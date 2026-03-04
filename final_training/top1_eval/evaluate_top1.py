#!/usr/bin/env python3
"""
evaluate_top1.py
================
Evaluate TOP1 accuracy for 3 models × 7 energy combinations × 3 metrics.

Energy combinations: e0, e1, e2, e0+e1, e0+e2, e1+e2, e0+e1+e2
Metrics: cosine similarity, DICE coefficient, matched peak count
"""
import sys, os, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODELS = ['cfm_default', 'param_jjy', 'full_model']
MODEL_LABELS = {
    'cfm_default': 'CFM-ID Default',
    'param_jjy': 'Param_JJY',
    'full_model': 'Full Model (Final)',
}
ENERGY_COMBOS = {
    'e0':       [0],
    'e1':       [1],
    'e2':       [2],
    'e0+e1':    [0, 1],
    'e0+e2':    [0, 2],
    'e1+e2':    [1, 2],
    'e0+e1+e2': [0, 1, 2],
}


# ============================================================
# Spectrum parsing
# ============================================================
def parse_spectrum(path, energy=0):
    """Parse CFM-ID spectrum file for a given energy level."""
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


def normalize_spectrum(peaks):
    """Normalize peak intensities to max=1."""
    if not peaks:
        return []
    max_i = max(p[1] for p in peaks)
    if max_i <= 0:
        return []
    return [(mz, i / max_i) for mz, i in peaks]


def match_peaks(spec1, spec2, mz_tol=0.01):
    """Match peaks between two spectra. Returns list of (i1, i2) matched pairs."""
    s1 = normalize_spectrum(spec1)
    s2 = normalize_spectrum(spec2)
    if not s1 or not s2:
        return [], s1, s2

    used2 = set()
    matched = []

    for idx1, (mz1, i1) in enumerate(s1):
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
            matched.append((i1, s2[best_j][1]))

    return matched, s1, s2


def cosine_similarity(spec1, spec2, mz_tol=0.01):
    """Cosine similarity between two spectra."""
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not matched or not s1 or not s2:
        return 0.0

    dot = sum(a * b for a, b in matched)
    sum_sq1 = sum(i ** 2 for _, i in s1)
    sum_sq2 = sum(i ** 2 for _, i in s2)
    denom = math.sqrt(sum_sq1 * sum_sq2)
    if denom == 0:
        return 0.0
    return dot / denom


def dice_similarity(spec1, spec2, mz_tol=0.01):
    """DICE similarity: 2*|matched| / (|spec1| + |spec2|)."""
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    total = len(s1) + len(s2)
    if total == 0:
        return 0.0
    return 2.0 * len(matched) / total


def matched_peak_count(spec1, spec2, mz_tol=0.01):
    """Number of matched peaks."""
    matched, _, _ = match_peaks(spec1, spec2, mz_tol)
    return len(matched)


# ============================================================
# Multi-energy scoring
# ============================================================
def combined_score(exp_spectra, pred_spectra, energies, metric_fn):
    """Compute combined score across multiple energy levels."""
    scores = []
    for e in energies:
        exp = exp_spectra.get(e, [])
        pred = pred_spectra.get(e, [])
        scores.append(metric_fn(exp, pred))
    return np.mean(scores) if scores else 0.0


# ============================================================
# Main evaluation
# ============================================================
def main():
    print('=' * 80)
    print('  TOP1 Accuracy Evaluation')
    print('  3 Models × 7 Energy Combos × 3 Metrics')
    print('=' * 80)

    # Load candidate data
    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'

    # Results storage
    results = []

    for model in MODELS:
        pred_dir = BASE / model / 'predictions'
        n_pred = len(list(pred_dir.glob('*.log')))
        print(f'\n  {MODEL_LABELS[model]}: {n_pred} prediction files')

        if n_pred == 0:
            print('    SKIP: no predictions')
            continue

        # For each library compound
        for combo_name, energies in ENERGY_COMBOS.items():
            top1_cos = 0
            top1_dice = 0
            top1_matched = 0
            n_evaluated = 0
            n_correct_in_cands = 0

            for lib_name in df['lib_name'].unique():
                sub = df[df['lib_name'] == lib_name]
                correct_smi = sub['lib_dtb_smiles'].iloc[0]

                # Parse experimental spectra
                lib_id = lib_name  # e.g., HMDB0004825_350.20743_12.3
                exp_file = exp_spectra_dir / f'{lib_id}.txt'
                if not exp_file.exists():
                    continue

                exp_specs = {}
                for e in energies:
                    exp_specs[e] = parse_spectrum(exp_file, e)

                # Score all candidates
                cand_scores_cos = []
                cand_scores_dice = []
                cand_scores_matched = []

                for _, row in sub.iterrows():
                    cand_id = row['cand_id']
                    cand_smi = row['cand_dtb_smiles']
                    pred_file = pred_dir / f'{cand_id}.log'

                    if not pred_file.exists():
                        cand_scores_cos.append((cand_smi, 0.0))
                        cand_scores_dice.append((cand_smi, 0.0))
                        cand_scores_matched.append((cand_smi, 0))
                        continue

                    pred_specs = {}
                    for e in energies:
                        pred_specs[e] = parse_spectrum(pred_file, e)

                    cos = combined_score(exp_specs, pred_specs, energies, cosine_similarity)
                    dice = combined_score(exp_specs, pred_specs, energies, dice_similarity)
                    matched = combined_score(exp_specs, pred_specs, energies,
                                             lambda a, b: matched_peak_count(a, b))

                    cand_scores_cos.append((cand_smi, cos))
                    cand_scores_dice.append((cand_smi, dice))
                    cand_scores_matched.append((cand_smi, matched))

                if not cand_scores_cos:
                    continue

                n_evaluated += 1
                has_correct = correct_smi in [c[0] for c in cand_scores_cos]
                if has_correct:
                    n_correct_in_cands += 1

                # TOP1
                best_cos = max(cand_scores_cos, key=lambda x: x[1])
                best_dice = max(cand_scores_dice, key=lambda x: x[1])
                best_matched = max(cand_scores_matched, key=lambda x: x[1])

                if best_cos[0] == correct_smi:
                    top1_cos += 1
                if best_dice[0] == correct_smi:
                    top1_dice += 1
                if best_matched[0] == correct_smi:
                    top1_matched += 1

            if n_evaluated > 0:
                results.append({
                    'model': MODEL_LABELS[model],
                    'energy_combo': combo_name,
                    'n_evaluated': n_evaluated,
                    'n_correct_in_cands': n_correct_in_cands,
                    'top1_cosine': top1_cos,
                    'top1_dice': top1_dice,
                    'top1_matched': top1_matched,
                    'acc_cosine': top1_cos / n_evaluated * 100,
                    'acc_dice': top1_dice / n_evaluated * 100,
                    'acc_matched': top1_matched / n_evaluated * 100,
                })

    # Print results
    print(f'\n\n{"=" * 80}')
    print(f'  Results (TOP1 Accuracy %)')
    print(f'{"=" * 80}')

    print(f'\n{"Model":<22} {"Energy":<10} {"Cosine":>10} {"DICE":>10} {"Matched":>10} '
          f'{"Eval":>6} {"InCand":>7}')
    print('-' * 80)

    for r in results:
        print(f'{r["model"]:<22} {r["energy_combo"]:<10} '
              f'{r["acc_cosine"]:>9.1f}% {r["acc_dice"]:>9.1f}% {r["acc_matched"]:>9.1f}% '
              f'{r["n_evaluated"]:>6} {r["n_correct_in_cands"]:>7}')

    # Best combo per model
    print(f'\n{"=" * 80}')
    print(f'  Best Energy Combination per Model (by Cosine TOP1)')
    print(f'{"=" * 80}')
    for model in MODELS:
        model_results = [r for r in results if r['model'] == MODEL_LABELS[model]]
        if model_results:
            best = max(model_results, key=lambda x: x['acc_cosine'])
            print(f'  {best["model"]:<22}: {best["energy_combo"]:<10} '
                  f'Cosine={best["acc_cosine"]:.1f}%  DICE={best["acc_dice"]:.1f}%  '
                  f'Matched={best["acc_matched"]:.1f}%')

    # Save CSV
    out_csv = BASE / 'top1_results.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')


if __name__ == '__main__':
    main()
