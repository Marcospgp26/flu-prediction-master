"""Compares models that predict next season's influenza HA/NA gene sequences.

Examples:
    python main.py                                       # H3N2-NA, every model
    python main.py --dataset all --align                 # every dataset, aligned sequences
    python main.py --dataset all --align --seeds 5       # repeats every model with 5 seeds
    python main.py --dataset h1n1-ha --models classical  # only the original research's models
    python main.py --models copy,random-forest,transformer --epochs 300
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dataset import DATASETS, load_dataset
from evaluation import ChunkModel, CopyBaseline, SequenceModel, evaluate
from prediction_model import decisionTree, extraTrees, naiveBayes, randomForest

# Available models, by their command-line name
CLASSICAL = {
    'decision-tree': decisionTree,
    'random-forest': randomForest,
    'extra-trees': extraTrees,
    'naive-bayes': naiveBayes,
}
DEEP = ['cnn', 'transformer', 'lstm']
GROUPS = {'baseline': ['copy'], 'classical': list(CLASSICAL), 'deep': DEEP}
GROUPS['all'] = GROUPS['baseline'] + GROUPS['classical'] + GROUPS['deep']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='h3n2-na', choices=[*DATASETS, 'all'])
    parser.add_argument('--models', default='all',
                        help='comma-separated models or groups: ' + ', '.join([*GROUPS, *GROUPS['all']]))
    parser.add_argument('--align', action='store_true',
                        help='trims every sequence at its start codon (ATG) so positions line up')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs for the neural networks')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto', help="'auto', 'cuda' (GPU) or 'cpu'")
    parser.add_argument('--seed', type=int, default=0, help='first random seed')
    parser.add_argument('--seeds', type=int, default=1,
                        help='how many seeds to use (seed, seed+1, ...); with more than one, saves mean and spread')
    parser.add_argument('--verbose', action='store_true', help='prints the neural networks\' training loss')
    parser.add_argument('--output', default='results', help='folder where the CSV files are saved')
    return parser.parse_args()


def resolve_models(spec):
    """'classical,cnn' -> ['decision-tree', 'random-forest', 'extra-trees', 'naive-bayes', 'cnn']"""
    names = []
    for item in (s.strip() for s in spec.split(',') if s.strip()):
        if item not in GROUPS and item not in GROUPS['all']:
            raise SystemExit(f"Unknown model '{item}'. Options: {', '.join([*GROUPS, *GROUPS['all']])}")
        for name in GROUPS.get(item, [item]):
            if name not in names:
                names.append(name)
    return names


def load_deep(names, args):
    """Imports the neural networks only when they are used, so the rest runs even without PyTorch."""
    if not any(n in DEEP for n in names):
        return None
    try:
        import prediction_model.deep as deep
    except ImportError as error:
        print(f'Skipping the neural networks: PyTorch is not installed ({error}). See README.md.')
        return None
    print('Neural networks will run on:', deep.resolve_device(args.device))
    return deep


def build_models(names, args, seed, deep):
    models = []
    for name in names:
        if name == 'copy':
            models.append(SequenceModel('Copy previous season (baseline)', CopyBaseline))
        elif name in CLASSICAL:
            module = CLASSICAL[name]
            models.append(ChunkModel(module.NAME, lambda m=module: m.build_model(seed)))
        elif deep is not None:
            label, network, lr = deep.DEEP_MODELS[name]
            models.append(SequenceModel(label, lambda net=network, lr=lr: deep.NeuralSequenceModel(
                net, epochs=args.epochs, lr=lr, batch_size=args.batch_size,
                device=args.device, seed=seed, verbose=args.verbose)))
    return models


# ------------------------------------------------------------------ output for a single seed

def print_report(r):
    cv = np.array(r['cv_r2'])
    print(f"\n----- {r['model']} -----")
    print('Cross-validation R² (2 folds):', np.round(cv, 4))
    print('Average R²: %0.2f (+/- %0.2f)' % (cv.mean() * 100, cv.std() * 100))
    print(f"Test ({r['n_train']} train pairs / {r['n_test']} test pairs):")
    print('  R²:', round(r['r2'], 4))
    print('  R² (variance weighted):', round(r['r2_variance_weighted'], 4))
    print('  Mean absolute error (MAE): %.4g' % r['mae'])
    print('  Mean squared error (MSE): %.4g' % r['mse'])
    print('  Nucleotide accuracy: %.2f%%' % (r['nt_accuracy'] * 100))
    print('  Mutation accuracy: %.2f%% (%d positions)' % (r['mutation_accuracy'] * 100, r['mutation_sites']))


def print_summary(results):
    # Plain R² can reach -1e+27 when a 15-digit column barely varies, so it is shown in
    # scientific notation. The variance-weighted R² is the steadier of the two.
    header = (f"{'Model':<34}{'CV R²':>11}{'Test R²':>11}{'R² (vw)':>10}{'MAE':>10}"
              f"{'NT acc %':>10}{'Mut acc %':>11}{'Time s':>8}")
    print('\n' + header)
    print('-' * len(header))
    for r in results:
        print(f"{r['model']:<34}{np.mean(r['cv_r2']):>11.2e}{r['r2']:>11.2e}{r['r2_variance_weighted']:>10.3f}"
              f"{r['mae']:>10.2e}{r['nt_accuracy'] * 100:>10.2f}{r['mutation_accuracy'] * 100:>11.2f}"
              f"{r['seconds']:>8.1f}")


def save_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['dataset', 'model', 'cv_r2_mean', 'cv_r2_std', 'cv_nt_accuracy_mean', 'r2', 'r2_variance_weighted',
               'mae', 'mse', 'nt_accuracy', 'mutation_accuracy', 'mutation_sites', 'n_train', 'n_test', 'seconds']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow({**r, 'cv_r2_mean': np.mean(r['cv_r2']), 'cv_r2_std': np.std(r['cv_r2']),
                             'cv_nt_accuracy_mean': np.mean(r['cv_nt_accuracy'])})
    print(f'\nSaved {path}')


# ------------------------------------------------------------------ output for several seeds

SEED_METRICS = ['nt_accuracy', 'mutation_accuracy', 'cv_nt_accuracy_mean', 'r2_variance_weighted', 'mae', 'seconds']


def summarize_seeds(runs):
    """One row per model, with the mean and standard deviation of each score across seeds."""
    rows = []
    for model in dict.fromkeys(r['model'] for r in runs):
        group = [r for r in runs if r['model'] == model]
        row = {'dataset': group[0]['dataset'], 'model': model, 'seeds': len(group)}
        for key in SEED_METRICS:
            values = [np.mean(r['cv_nt_accuracy']) if key == 'cv_nt_accuracy_mean' else r[key] for r in group]
            row[f'{key}_mean'], row[f'{key}_std'] = float(np.mean(values)), float(np.std(values))
        rows.append(row)
    return rows


def print_seed_summary(rows):
    header = (f"{'Model':<34}{'Seeds':>6}{'NT acc % (mean +/- sd)':>25}{'Mut acc % (mean +/- sd)':>26}"
              f"{'CV NT acc %':>13}")
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        nt = f"{r['nt_accuracy_mean'] * 100:.2f} +/- {r['nt_accuracy_std'] * 100:.2f}"
        mut = f"{r['mutation_accuracy_mean'] * 100:.2f} +/- {r['mutation_accuracy_std'] * 100:.2f}"
        print(f"{r['model']:<34}{r['seeds']:>6}{nt:>25}{mut:>26}{r['cv_nt_accuracy_mean_mean'] * 100:>13.2f}")


def save_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path}')


# ------------------------------------------------------------------ main program

def keep_awake():
    """Stops Windows from going to sleep during the run (back to normal when Python exits)."""
    if sys.platform == 'win32':
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    args = parse_args()
    keep_awake()
    names = resolve_models(args.models)
    deep = load_deep(names, args)
    seeds = [args.seed + k for k in range(max(1, args.seeds))]
    datasets = list(DATASETS) if args.dataset == 'all' else [args.dataset]

    for name in datasets:
        data = load_dataset(name, align=args.align)
        stem = f"{name}{'-aligned' if args.align else ''}"
        print(f"\n========== {name.upper()}{' (aligned)' if args.align else ''}: {len(data)} sequence pairs, "
              f'{data.length} nucleotides, {data.chunk_width} chunks ==========', flush=True)

        # One seed: a full report for each model, as in the original code
        if len(seeds) == 1:
            results = []
            for model in build_models(names, args, seeds[0], deep):
                result = evaluate(model, data)
                print_report(result)
                results.append(result)
            print_summary(results)
            save_csv(results, Path(args.output) / f'{stem}.csv')
            continue

        # Several seeds: one line per run and, at the end, the mean and spread of each model
        runs = []
        for seed in seeds:
            for model in build_models(names, args, seed, deep):
                result = evaluate(model, data)
                result['seed'] = seed
                runs.append(result)
                print(f"  seed {seed}  {result['model']:<34} nt {result['nt_accuracy'] * 100:6.2f}%  "
                      f"mut {result['mutation_accuracy'] * 100:6.2f}%", flush=True)
        summary = summarize_seeds(runs)
        print_seed_summary(summary)
        per_run = [{'seed': r['seed'], 'dataset': r['dataset'], 'model': r['model'],
                    'cv_nt_accuracy_mean': np.mean(r['cv_nt_accuracy']),
                    **{k: r[k] for k in SEED_METRICS if k != 'cv_nt_accuracy_mean'}} for r in runs]
        save_rows(per_run, Path(args.output) / f'{stem}-seeds.csv')
        save_rows(summary, Path(args.output) / f'{stem}-seeds-summary.csv')


if __name__ == '__main__':
    main()
