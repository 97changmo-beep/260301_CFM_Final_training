#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_5fold.py
==================
Evaluate 5-repeat CFM-ID transfer learning results.

For each fold:
1. Use cfm-predict (Docker) to predict test compound spectra
2. Compare predicted vs experimental spectra (cosine similarity)
3. Compare with baseline (default CFM-ID model)

Reports: per-fold results, 5-fold mean +/- std, Wilcoxon test

Usage:
    python evaluate_5fold.py [--skip-predict]   # skip cfm-predict if already done
"""
import sys, io, os, re, math, csv, subprocess
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parent
N_FOLDS = 5
IMAGE_NAME = "cfmid-final"


# ============================================================
# Spectrum parsing
# ============================================================
def parse_cfmid_output(path, energy=2):
    """Parse CFM-ID prediction output (energy0/1/2 sections)."""
    peaks = []
    target = f'energy{energy}'
    in_target = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == target:
                in_target = True
                peaks = []
                continue
            if in_target:
                if line.startswith('energy') or line == '':
                    break
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return peaks


def parse_experimental_spectrum(path, energy=0):
    """Parse experimental spectrum file (same format as CFM-ID output)."""
    return parse_cfmid_output(path, energy)


def cosine_similarity(spec1, spec2, mz_tol=0.01):
    """Compute cosine similarity between two spectra."""
    if not spec1 or not spec2:
        return 0.0, 0

    # Normalize to max intensity
    max1 = max(p[1] for p in spec1)
    max2 = max(p[1] for p in spec2)
    if max1 <= 0 or max2 <= 0:
        return 0.0, 0

    s1 = [(mz, i / max1) for mz, i in spec1]
    s2 = [(mz, i / max2) for mz, i in spec2]

    used2 = set()
    matched_prod = 0.0
    n_matched = 0

    for mz1, i1 in s1:
        best_j = -1
        best_diff = mz_tol + 1
        for j, (mz2, i2) in enumerate(s2):
            if j in used2:
                continue
            diff = abs(mz1 - mz2)
            if diff <= mz_tol and diff < best_diff:
                best_j = j
                best_diff = diff
        if best_j >= 0:
            used2.add(best_j)
            matched_prod += i1 * s2[best_j][1]
            n_matched += 1

    sum_sq1 = sum(i ** 2 for _, i in s1)
    sum_sq2 = sum(i ** 2 for _, i in s2)
    denom = math.sqrt(sum_sq1 * sum_sq2)
    if denom == 0:
        return 0.0, 0

    return matched_prod / denom, n_matched


# ============================================================
# Evaluation
# ============================================================
def evaluate_fold(fold_idx):
    """Evaluate a single fold's predictions against experimental spectra."""
    fold_dir = BASE / f'fold_{fold_idx}'
    test_file = fold_dir / 'test_compounds.txt'
    test_spectra_dir = fold_dir / 'test_spectra'
    predict_dir = fold_dir / 'predictions'
    model_file = fold_dir / 'param_output.log'

    if not model_file.exists():
        print(f'  Fold {fold_idx}: param_output.log not found (not trained yet)')
        return None

    # Read test compounds
    test_compounds = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            test_compounds.append({
                'id': parts[0],
                'smiles': parts[1] if len(parts) > 1 else '',
                'pepmass': parts[2] if len(parts) > 2 else '',
            })

    # Evaluate each test compound at each energy level
    results = []
    for comp in test_compounds:
        cid = comp['id']
        exp_file = test_spectra_dir / f'{cid}.txt'
        pred_file = predict_dir / f'{cid}.log'

        if not exp_file.exists():
            continue

        row = {'compound_id': cid}
        for energy in [0, 1, 2]:
            exp_spec = parse_experimental_spectrum(exp_file, energy)
            if pred_file.exists():
                pred_spec = parse_cfmid_output(pred_file, energy)
                cos, n_match = cosine_similarity(exp_spec, pred_spec)
            else:
                cos, n_match = 0.0, 0

            row[f'cos_e{energy}'] = cos
            row[f'matched_e{energy}'] = n_match

        results.append(row)

    return results


def evaluate_baseline(fold_idx, default_model_path):
    """Evaluate default (baseline) model on the same test set."""
    fold_dir = BASE / f'fold_{fold_idx}'
    test_spectra_dir = fold_dir / 'test_spectra'
    baseline_dir = fold_dir / 'baseline_predictions'
    test_file = fold_dir / 'test_compounds.txt'

    test_compounds = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            test_compounds.append({'id': parts[0]})

    results = []
    for comp in test_compounds:
        cid = comp['id']
        exp_file = test_spectra_dir / f'{cid}.txt'
        pred_file = baseline_dir / f'{cid}.log'

        if not exp_file.exists():
            continue

        row = {'compound_id': cid}
        for energy in [0, 1, 2]:
            exp_spec = parse_experimental_spectrum(exp_file, energy)
            if pred_file.exists():
                pred_spec = parse_cfmid_output(pred_file, energy)
                cos, n_match = cosine_similarity(exp_spec, pred_spec)
            else:
                cos, n_match = 0.0, 0

            row[f'cos_e{energy}'] = cos
            row[f'matched_e{energy}'] = n_match

        results.append(row)

    return results


# ============================================================
# Main
# ============================================================
def main():
    skip_predict = '--skip-predict' in sys.argv

    print('=' * 70)
    print('  CFM-ID Final Training — 5-Repeat Evaluation')
    print('=' * 70)

    all_fold_results = []
    all_baseline_results = []

    for fold in range(N_FOLDS):
        fold_dir = BASE / f'fold_{fold}'

        if not (fold_dir / 'param_output.log').exists():
            print(f'\n  Fold {fold}: SKIPPED (not trained)')
            continue

        print(f'\n  Fold {fold}:')

        # Evaluate trained model
        results = evaluate_fold(fold)
        if results:
            all_fold_results.append(results)
            cos_means = {}
            for e in [0, 1, 2]:
                vals = [r[f'cos_e{e}'] for r in results]
                cos_means[e] = np.mean(vals)
                print(f'    Trained  e{e}: mean cosine = {np.mean(vals):.4f} '
                      f'(+/-{np.std(vals):.4f}, n={len(vals)})')

        # Evaluate baseline
        baseline = evaluate_baseline(fold, None)
        if baseline:
            all_baseline_results.append(baseline)
            for e in [0, 1, 2]:
                vals = [r[f'cos_e{e}'] for r in baseline]
                print(f'    Baseline e{e}: mean cosine = {np.mean(vals):.4f} '
                      f'(+/-{np.std(vals):.4f})')

    # Cross-fold summary
    if not all_fold_results:
        print('\nNo trained folds found. Run training first.')
        return

    print(f'\n{"=" * 70}')
    print(f'  Cross-Fold Summary ({len(all_fold_results)} folds)')
    print(f'{"=" * 70}')

    for e in [0, 1, 2]:
        fold_means = []
        baseline_means = []
        for i, results in enumerate(all_fold_results):
            fold_means.append(np.mean([r[f'cos_e{e}'] for r in results]))
        for i, results in enumerate(all_baseline_results):
            baseline_means.append(np.mean([r[f'cos_e{e}'] for r in results]))

        trained_mean = np.mean(fold_means)
        trained_std = np.std(fold_means)

        print(f'\n  Energy {e}:')
        print(f'    Trained:  {trained_mean:.4f} +/- {trained_std:.4f}')

        if baseline_means:
            baseline_mean = np.mean(baseline_means)
            baseline_std = np.std(baseline_means)
            delta = trained_mean - baseline_mean
            print(f'    Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}')
            print(f'    Delta:    +{delta:.4f}')

            # Wilcoxon signed-rank test (if scipy available)
            if HAS_SCIPY and len(fold_means) == len(baseline_means) and len(fold_means) >= 5:
                try:
                    stat, pval = wilcoxon(fold_means, baseline_means)
                    print(f'    Wilcoxon: p = {pval:.4f}')
                except Exception:
                    pass

    # Save results
    rows = []
    for fold_idx, results in enumerate(all_fold_results):
        for r in results:
            r['fold'] = fold_idx
            r['model'] = 'trained'
            rows.append(r)
    for fold_idx, results in enumerate(all_baseline_results):
        for r in results:
            r['fold'] = fold_idx
            r['model'] = 'baseline'
            rows.append(r)

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        out_csv = BASE / 'evaluation_results.csv'
        df.to_csv(out_csv, index=False)
        print(f'\nSaved: {out_csv}')
        try:
            out_xlsx = BASE / 'evaluation_results.xlsx'
            df.to_excel(out_xlsx, index=False)
            print(f'Saved: {out_xlsx}')
        except Exception:
            pass
    elif rows:
        out_csv = BASE / 'evaluation_results.csv'
        fieldnames = ['fold', 'model', 'compound_id',
                      'cos_e0', 'matched_e0', 'cos_e1', 'matched_e1',
                      'cos_e2', 'matched_e2']
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved: {out_csv}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
