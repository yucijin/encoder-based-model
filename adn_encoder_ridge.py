"""Predict context entry from ADn photometry using frozen encoder features and Ridge.

Train on CPU (Python 3.10+, numpy, pandas, scikit-learn, threadpoolctl, h5py, openpyxl):
    python adn_encoder_ridge.py

Extract features from adn_input.xlsx using the study's external mouse tokenizer:
    python adn_encoder_ridge.py extract --tokenizer-root /path/to/tokenizer-v2-fast-preview-20261023-batch2

BrainMU project: https://github.com/BAAI-Brain-Inspired-Group/Brainmu
Extraction uses the batch2 brainmu_tokenizer_v2 inference API and weights/mouse.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44, 45, 46)
SHUFFLE_SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)
CONDITIONS = ('Ordered', 'Time shuffle (matched alpha)')
FEATURE_NAMES = ('ordered', *(f'shuffle_{s}' for s in SEEDS))
ALPHAS = np.array([0.001, 0.01, 0.1, 1., 10., 100.])
RELEASE = 'tokenizer-v2-fast-preview-20261023-batch2'
WEIGHTS_SHA = '9349806e4bd48af58818f0674df03b10664c773b5ba79ce113d077d12fb01968'
SPEC_SHA = '6190c64c648ad201fb3a42ef00636b5bf8a2638a941562ffa247e3f39602e865'
META_FIELDS = ['mouse', 'trial', 'label']
SAMPLE_COLUMNS = [f'sample_{i:03d}' for i in range(80)]
FOLD_COLUMNS = [f'fold_{seed}' for seed in SEEDS]


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        yield staged
        staged.replace(path)


def check_trials(info, folds):
    if info.isna().any().any() or info.duplicated(['mouse', 'trial']).any():
        raise ValueError('Missing metadata or duplicate mouse/trial identities.')
    if not np.isin(info['label'], [0, 1]).all():
        raise ValueError('Expected binary labels 0 and 1.')
    if folds.shape != (len(SEEDS), len(info)) or not np.isin(folds, range(1, 6)).all():
        raise ValueError('Expected one outer fold (1–5) per seed and trial.')
    for mouse, rows in info.groupby('mouse').indices.items():
        if any(len(np.unique(fold[rows])) != 5 for fold in folds):
            raise ValueError(f'Missing test fold for {mouse}.')


def combined_kernel(x):
    """Average the dimension-normalized linear kernels for X and its absolute first differences."""
    def gram(values):
        flat = values.reshape(len(values), -1).astype(np.float64)
        return (flat @ flat.T) / flat.shape[1]

    return (gram(x) + gram(np.abs(np.diff(x, axis=1)))) / 2


def read_features(path):
    with h5py.File(path, 'r') as f:
        info = pd.DataFrame({'mouse': f['trials/mouse'].asstr()[:],
                             'trial': f['trials/trial'].asstr()[:], 'label': f['trials/label'][:]})
        folds = f['cv/fold_ids'][:]
        np.testing.assert_array_equal(f['cv/seeds'][:], SEEDS)
        np.testing.assert_array_equal(f['cv/shuffle_seeds'][:], SHUFFLE_SEEDS)
        check_trials(info, folds)
        kernels = []
        for name in FEATURE_NAMES:
            dataset = f['encoder/' + name]
            x = dataset[:]
            if x.shape != (len(info), 60, 1024) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f'Invalid Encoder 1024 array: {dataset.name}')
            if sha256(memoryview(x)).hexdigest() != dataset.attrs['sha256']:
                raise ValueError(f'Feature checksum mismatch: {dataset.name}')
            kernels.append(combined_kernel(x))
    return info, folds, kernels


def ridge_scores(g, y, train, test):
    """Six linear-Ridge fits in dual form. Centering, scale and class weights use training rows."""
    labels = y[train]
    counts = np.bincount(labels, minlength=2)
    weights = len(train) / (2 * counts[labels])
    weights /= weights.mean()
    train_kernel, test_kernel = g[np.ix_(train, train)], g[np.ix_(test, train)]
    center = train_kernel @ weights / weights.sum()
    grand = weights @ center / weights.sum()
    k = train_kernel - center[:, None] - center[None, :] + grand
    ktest = test_kernel - (test_kernel @ weights / weights.sum())[:, None] - center[None, :] + grand
    scale = np.trace(k) / len(train)
    scale = scale if scale > 1e-14 else 1.
    root = np.sqrt(weights)
    weighted = root[:, None] * (k / scale) * root[None, :]
    eigenvalues, basis = np.linalg.eigh((weighted + weighted.T) / 2)
    target = 2. * labels - 1.
    intercept = np.average(target, weights=weights)
    rhs = basis.T @ (root * (target - intercept))
    test_projection = (ktest / scale * root) @ basis
    coefficients = rhs[:, None] / (eigenvalues[:, None] + ALPHAS)
    return intercept + (test_projection @ coefficients).T


def inner_folds(y, mice, train, seed, fold):
    parts = [[] for _ in range(3)]
    for mi, mouse in enumerate(np.unique(mice)):
        ix = train[mice[train] == mouse]
        cv = StratifiedKFold(3, shuffle=True, random_state=seed * 100 + fold + mi * 1000)
        for j, (it, iv) in enumerate(cv.split(ix, y[ix])):
            parts[j].append((ix[it], ix[iv]))
    return [(np.concatenate([p[0] for p in group]), np.concatenate([p[1] for p in group]))
            for group in parts]


def choose_alpha(g, y, mice, train, inner):
    scores = np.full((len(ALPHAS), len(y)), np.nan)
    for fit, validation in inner:
        scores[:, validation] = ridge_scores(g, y, fit, validation)
    assert np.isfinite(scores[:, train]).all()
    correct = (scores > 0) == y
    criterion = np.zeros(len(ALPHAS))
    for mouse in np.unique(mice):
        ix = train[mice[train] == mouse]
        criterion += (correct[:, ix[y[ix] == 0]].mean(1) + correct[:, ix[y[ix] == 1]].mean(1)) / 2
    return int(np.argmax(criterion))


def fit_models(kernels, info, folds):
    """Select alpha inside each outer training fold and reuse it for the shuffled control."""
    mice, y = info['mouse'].to_numpy(), info['label'].to_numpy()
    original, *shuffled_kernels = kernels
    scores = np.full((2, len(SEEDS), len(y)), np.nan)  # condition, seed, trial
    alpha = np.full((2, len(SEEDS), 5), np.nan)
    for si, seed in enumerate(SEEDS):
        shuffled = shuffled_kernels[si]
        for fold in range(1, 6):
            train, test = np.flatnonzero(folds[si] != fold), np.flatnonzero(folds[si] == fold)
            inner = inner_folds(y, mice, train, seed, fold)
            alpha_index = choose_alpha(original, y, mice, train, inner)
            scores[0, si, test] = ridge_scores(original, y, train, test)[alpha_index]
            scores[1, si, test] = ridge_scores(shuffled, y, train, test)[alpha_index]
            alpha[:, si, fold - 1] = ALPHAS[alpha_index]
    assert np.isfinite(scores).all() and np.isfinite(alpha).all()
    return scores, alpha


def ba(y, correct):
    """Mean recall over classes present in the test set, including single-class folds."""
    return np.mean([correct[y == label].mean() for label in np.unique(y)])


def performance(info, folds, scores, alpha):
    y, mice = info['label'].to_numpy(), info['mouse'].to_numpy()
    rows, fold_rows = [], []
    for ci, condition in enumerate(CONDITIONS):
        for si, seed in enumerate(SEEDS):
            correct = (scores[ci, si] > 0) == y
            for mouse in np.unique(mice):
                mouse_ix = mice == mouse
                fold_ba = []
                for f in range(1, 6):
                    ix = mouse_ix & (folds[si] == f)
                    fold_ba.append(ba(y[ix], correct[ix]))
                    n0, n1 = np.bincount(y[ix], minlength=2)
                    fold_rows.append([condition, mouse, seed, f,
                                      (folds[si] != f).sum(), n0, n1,
                                      alpha[ci, si, f - 1], fold_ba[-1]])
                rows.append([condition, mouse, seed, np.mean(fold_ba),
                             ba(y[mouse_ix], correct[mouse_ix])])
    per_mouse_seed = pd.DataFrame(rows, columns=['condition', 'mouse', 'seed',
                                                'mean_fold_BA', 'pooled_OOF_BA'])
    summary = per_mouse_seed.groupby('condition', sort=False, as_index=False)[
        ['mean_fold_BA', 'pooled_OOF_BA']].mean()
    folds_table = pd.DataFrame(fold_rows, columns=['condition', 'mouse', 'seed', 'fold',
                                                  'training_trials', 'test_label0', 'test_label1',
                                                  'alpha', 'fold_BA'])
    return summary, per_mouse_seed, folds_table


def trial_predictions(info, folds, scores):
    y = info['label'].to_numpy()
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for si, seed in enumerate(SEEDS):
            score = scores[ci, si]
            frame = info.copy()
            frame['condition'] = condition
            frame['seed'] = seed
            frame['fold'] = folds[si]
            frame['score'] = score
            frame['prediction'] = (score > 0).astype(int)
            frame['correct'] = (frame['prediction'].to_numpy() == y).astype(int)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def train(features, output):
    with threadpool_limits(limits=1):
        info, folds, kernels = read_features(features)
        scores, alpha = fit_models(kernels, info, folds)
    summary, mouse_seed, fold_table = performance(info, folds, scores, alpha)
    by_mouse = mouse_seed.groupby(['condition', 'mouse'], sort=False, as_index=False)[
        ['mean_fold_BA', 'pooled_OOF_BA']].mean()
    fold_table['single_class_test'] = (fold_table[['test_label0', 'test_label1']].min(axis=1) == 0)
    tables = {'summary': summary, 'per_mouse': by_mouse, 'per_mouse_seed': mouse_seed,
              'fold_and_alpha': fold_table, 'trial_predictions': trial_predictions(info, folds, scores)}
    with atomic_output(output) as staged, pd.ExcelWriter(staged, engine='openpyxl') as writer:
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
        for name, tab in tables.items():
            tab.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = 'B2'
            ws.auto_filter.ref = ws.dimensions
            for j, col in enumerate(tab.columns, 1):
                ws.cell(1, j).fill = PatternFill('solid', fgColor='285563')
                ws.cell(1, j).font = Font(color='FFFFFF', bold=True)
                ws.column_dimensions[get_column_letter(j)].width = max(
                    len(col) + 3, 29 if col == 'condition' else 12)
                fmt = '0.00%' if col.endswith('BA') else '0.000000' if col == 'score' else None
                if fmt:
                    for row in range(2, len(tab) + 2):
                        ws.cell(row, j).number_format = fmt
    print(summary.to_string(index=False))
    print(output)


def read_signals(path):
    with pd.ExcelFile(path) as workbook:
        rows = pd.read_excel(workbook, sheet_name='trials').sort_values('trial_index')
    np.testing.assert_array_equal(rows['trial_index'], np.arange(len(rows)))
    info = rows[META_FIELDS].reset_index(drop=True)
    folds = rows[FOLD_COLUMNS].to_numpy().T
    check_trials(info, folds)
    x = np.ascontiguousarray(rows[SAMPLE_COLUMNS], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite calcium samples.')
    return info, folds.astype(np.int8), x


def encode_condition(tokenizer_root, device, raw, name):
    from unittest.mock import patch
    import torch

    package = tokenizer_root / 'inference'
    weights = tokenizer_root / 'weights' / 'mouse'
    if not (package / 'brainmu_tokenizer_v2').is_dir() or not (weights / 'COMPLETE.json').is_file():
        raise ValueError('Expected inference/brainmu_tokenizer_v2 and weights/mouse under --tokenizer-root.')
    identity = json.loads((weights / 'COMPLETE.json').read_text())
    if (identity['sha256'], identity['encoding_spec_sha256']) != (WEIGHTS_SHA, SPEC_SHA):
        raise ValueError('The mouse checkpoint or encoding spec differs from the study release.')
    sys.path.insert(0, str(package))
    import brainmu_tokenizer_v2 as bt

    torch.set_num_threads(1)
    tokenizer = bt.load(weights, device=device)
    tokenizer.model.eval()
    # Capture the continuous 1024-D representation immediately before quantization projection.
    encode_features = tokenizer.model._encode_features
    captured = []

    def capture(*args, **kwargs):
        result = encode_features(*args, **kwargs)
        captured.append(result.tokens.detach().float().cpu().numpy())
        return result

    signals = raw
    if name != 'ordered':
        seed = SHUFFLE_SEEDS[FEATURE_NAMES.index(name) - 1]
        rng = np.random.default_rng(seed)
        signals = np.stack([row[rng.permutation(80)] for row in raw])
    batch_size = 8 if name == 'shuffle_44' else 32
    features = np.empty((len(raw), 60, 1024), dtype=np.float32)
    expected_coords = np.column_stack((np.zeros(60, dtype=int), np.arange(60)))
    with torch.inference_mode(), patch.object(tokenizer.model, '_encode_features', capture):
        for start in range(0, len(raw), batch_size):
            batch = list(signals[start:start + batch_size])
            keep = len(batch)
            # Preserve the inference batch shape and discard padding before storing trial features.
            batch += [batch[-1]] * (batch_size - keep)
            prepared = [tokenizer.prepare('mouse/calcium', row[None, :], fs=40., group='dff',
                        unit='dff', signal_kind='dff', channel_names=['calcium_signal'],
                        rate_hz=30., n_time=60, window_start_s=0.)[0] for row in batch]
            encoded = tokenizer.encode(prepared)
            if len(encoded) != batch_size or len(captured) != 1:
                raise ValueError('Unexpected tokenizer batching or API.')
            for item in encoded[:keep]:
                np.testing.assert_array_equal(item.coordinates.cpu().numpy(), expected_coords)
            tokens = captured.pop()
            if tokens.shape != (batch_size * 60, 1024):
                raise ValueError(f'Unexpected encoder layer shape: {tokens.shape}')
            features[start:start + keep] = tokens.reshape(batch_size, 60, 1024)[:keep]
    return features


def write_features(path, info, folds, arrays):
    with atomic_output(path) as staged, h5py.File(staged, 'w') as f:
        f.attrs.update({'release': RELEASE, 'project_url': 'https://github.com/BAAI-Brain-Inspired-Group/Brainmu',
                        'entry': 'mouse/calcium', 'layer': 'normalized encoder vectors before quant_proj',
                        'axes': 'trial, time, channel', 'input_rate_hz': 40, 'feature_rate_hz': 30,
                        'weights_sha256': WEIGHTS_SHA, 'encoding_spec_sha256': SPEC_SHA,
                        'input': 'First 80 supplied segment-zscore calcium samples preceding context entry',
                        'shuffle': 'Permute 80 samples within each trial, then re-encode. Names use CV seed.'})
        trials = f.create_group('trials')
        for col in ('mouse', 'trial'):
            trials.create_dataset(col, data=info[col].tolist(), dtype=h5py.string_dtype('utf-8'))
        trials.create_dataset('label', data=info['label'].to_numpy(dtype=np.int8))
        cv = f.create_group('cv')
        cv.create_dataset('fold_ids', data=folds.astype(np.int8))  # [seed, trial]
        cv.create_dataset('seeds', data=SEEDS)
        cv.create_dataset('shuffle_seeds', data=SHUFFLE_SEEDS)
        group = f.create_group('encoder')
        for name, array in arrays:
            x = np.ascontiguousarray(array)
            if x.shape != (len(info), 60, 1024) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f'Invalid encoder features: {name}')
            ds = group.create_dataset(name, data=x, chunks=(1, 60, 1024),
                                      compression='gzip', compression_opts=4, shuffle=True)
            ds.attrs['sha256'] = sha256(memoryview(x)).hexdigest()
        if set(group.keys()) != set(FEATURE_NAMES):
            raise ValueError('Expected the ordered input and one shuffle per CV seed.')


def extract(args):
    from multiprocessing import get_context

    info, folds, raw = read_signals(args.signals)
    # Isolate GPU inference state for each time permutation.
    with get_context('spawn').Pool(1, maxtasksperchild=1) as worker:
        def arrays():
            for name in FEATURE_NAMES:
                yield name, worker.apply(encode_condition, (args.tokenizer_root, args.device, raw, name))
                print(f'Encoded {name}', flush=True)
        write_features(args.output, info, folds, arrays())
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command')
    fit = subs.add_parser('train', help='Train/test the Ridge readout on saved features (CPU).')
    fit.add_argument('--features', type=Path, default=HERE / 'adn_encoder1024.h5')
    fit.add_argument('--output', type=Path, default=HERE / 'adn_ridge_results.xlsx')
    enc = subs.add_parser('extract', help='Encode the signal workbook with an external matching tokenizer.')
    enc.add_argument('--signals', type=Path, default=HERE / 'adn_input.xlsx')
    enc.add_argument('--tokenizer-root', type=Path, required=True)
    enc.add_argument('--device', default='cuda:0')
    enc.add_argument('--output', type=Path, default=HERE / 'adn_encoder1024_extracted.h5')
    args = parser.parse_args(sys.argv[1:] or ['train'])
    if args.command == 'extract':
        extract(args)
    else:
        train(args.features, args.output)


if __name__ == '__main__':
    main()
