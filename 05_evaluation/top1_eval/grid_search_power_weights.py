#!/usr/bin/env python3
"""
grid_search_power_weights.py
============================
Grid search over scoring formula:  entropy^c * cos^a * dice^b
with fixed energy weights e0=0.3, e1=0.3, e2=0.4.

Uses the SAME metric functions as evaluate_top1_detail.py to ensure
consistent baseline (61/143 = 42.7% for a=1, b=1, c=1).

Strategy:
  1. Pre-compute per-compound per-candidate weighted cos, dice, ent scores ONCE
  2. For each (a, b, c) combination, re-score all candidates using the power formula
  3. Report TOP1 and TOP3 accuracy
"""

import sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ENERGIES = [0, 1, 2]
ENERGY_WEIGHTS = {0: 0.3, 1: 0.3, 2: 0.4}


# ============================================================
# Spectrum parsing & metrics  (identical to evaluate_top1_detail.py)
# ============================================================
def parse_spectrum(path, energy=0):
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
    if not peaks:
        return []
    max_i = max(p[1] for p in peaks)
    if max_i <= 0:
        return []
    return [(mz, i / max_i) for mz, i in peaks]


def match_peaks(spec1, spec2, mz_tol=0.01):
    s1 = normalize_spectrum(spec1)
    s2 = normalize_spectrum(spec2)
    if not s1 or not s2:
        return [], s1, s2
    used2 = set()
    matched = []
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
            matched.append((i1, s2[best_j][1]))
    return matched, s1, s2


def cosine_similarity(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not matched or not s1 or not s2:
        return 0.0
    dot = sum(a * b for a, b in matched)
    denom = math.sqrt(sum(i**2 for _, i in s1) * sum(i**2 for _, i in s2))
    return dot / denom if denom > 0 else 0.0


def dice_similarity(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    total = len(s1) + len(s2)
    return 2.0 * len(matched) / total if total > 0 else 0.0


def spectral_entropy_sim(spec1, spec2, mz_tol=0.01):
    """Entropy-based similarity (JSD) -- identical to evaluate_top1_detail.py."""
    s1 = normalize_spectrum(spec1)
    s2 = normalize_spectrum(spec2)
    if not s1 or not s2:
        return 0.0
    sum1 = sum(i for _, i in s1)
    sum2 = sum(i for _, i in s2)
    if sum1 == 0 or sum2 == 0:
        return 0.0
    p1 = {round(mz, 2): i / sum1 for mz, i in s1}
    p2 = {round(mz, 2): i / sum2 for mz, i in s2}
    all_mz = set(p1.keys()) | set(p2.keys())

    def H(probs):
        return -sum(p * math.log2(p) for p in probs if p > 0)

    vals = [(p1.get(mz, 0), p2.get(mz, 0)) for mz in all_mz]
    h1 = H([v[0] for v in vals])
    h2 = H([v[1] for v in vals])
    h_mix = H([(v[0] + v[1]) / 2 for v in vals])
    jsd = h_mix - (h1 + h2) / 2
    return 1 - jsd if jsd < 1 else 0.0


def weighted_combined(exp_specs, pred_specs, metric_fn):
    """Weighted combined score with energy weights 0.3:0.3:0.4."""
    s = 0
    for e in ENERGIES:
        s += ENERGY_WEIGHTS[e] * metric_fn(exp_specs.get(e, []), pred_specs.get(e, []))
    return s


# ============================================================
# Pre-compute per-compound per-candidate scores
# ============================================================
def precompute_scores(df, exp_spectra_dir, pred_dir):
    """
    Returns:
      compounds: list of dicts, each with:
        'hmdb_id': str  (correct HMDB ID for this compound)
        'lib_name': str
        'candidates': list of dicts, each with:
            'cand_hmdb': str
            'cos': float   (weighted combined cosine)
            'dice': float  (weighted combined dice)
            'ent': float   (weighted combined entropy)
    """
    compounds = []
    lib_names = sorted(df['lib_name'].unique())
    n_total = len(lib_names)
    n_skipped = 0

    print(f"Pre-computing scores for {n_total} compounds...")

    for li, lib_name in enumerate(lib_names):
        if (li + 1) % 30 == 0:
            print(f"  {li+1}/{n_total}...")

        sub = df[df['lib_name'] == lib_name]
        hmdb_id = str(sub['hmdb_id'].iloc[0]) if pd.notna(sub['hmdb_id'].iloc[0]) else ''

        exp_file = exp_spectra_dir / f'{lib_name}.txt'
        if not exp_file.exists():
            n_skipped += 1
            continue

        exp_specs = {e: parse_spectrum(exp_file, e) for e in ENERGIES}

        cand_list = []
        for _, row in sub.iterrows():
            cand_id = row['cand_id']
            cand_hmdb = str(row['cand_hmdb_id']) if pd.notna(row.get('cand_hmdb_id', None)) else ''
            pred_file = pred_dir / f'{cand_id}.log'

            if not pred_file.exists():
                cand_list.append({'cand_hmdb': cand_hmdb, 'cos': 0.0, 'dice': 0.0, 'ent': 0.0})
                continue

            pred_specs = {e: parse_spectrum(pred_file, e) for e in ENERGIES}
            cos = weighted_combined(exp_specs, pred_specs, cosine_similarity)
            dice = weighted_combined(exp_specs, pred_specs, dice_similarity)
            ent = weighted_combined(exp_specs, pred_specs, spectral_entropy_sim)

            cand_list.append({'cand_hmdb': cand_hmdb, 'cos': cos, 'dice': dice, 'ent': ent})

        if not cand_list:
            n_skipped += 1
            continue

        compounds.append({
            'hmdb_id': hmdb_id,
            'lib_name': lib_name,
            'candidates': cand_list,
        })

    print(f"  Done. {len(compounds)} compounds loaded, {n_skipped} skipped.")
    return compounds


# ============================================================
# Evaluate a single (a, b, c) combination
# ============================================================
def evaluate_power(compounds, a, b, c):
    """
    Score = ent^c * cos^a * dice^b
    Returns (top1_correct, top3_correct, n_total)
    """
    top1_correct = 0
    top3_correct = 0
    n_total = 0

    for comp in compounds:
        hmdb_id = comp['hmdb_id']
        if not hmdb_id:
            continue
        n_total += 1

        # Score all candidates
        scored = []
        for cand in comp['candidates']:
            # Use power formula; handle 0^0 = 1.0 (math.pow behavior)
            try:
                score = (cand['ent'] ** c) * (cand['cos'] ** a) * (cand['dice'] ** b)
            except (ValueError, ZeroDivisionError):
                score = 0.0
            scored.append((score, cand['cand_hmdb']))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # TOP1 check
        if scored and scored[0][1] == hmdb_id:
            top1_correct += 1

        # TOP3 check
        for rank_i in range(min(3, len(scored))):
            if scored[rank_i][1] == hmdb_id:
                top3_correct += 1
                break

    return top1_correct, top3_correct, n_total


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    print("=" * 90)
    print("  Grid Search: entropy^c * cos^a * dice^b")
    print("  Energy weights: e0=0.3, e1=0.3, e2=0.4 (fixed)")
    print("=" * 90)

    # Load data
    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'
    pred_dir = BASE / 'full_model' / 'predictions'

    # Pre-compute weighted scores
    compounds = precompute_scores(df, exp_spectra_dir, pred_dir)

    t_precompute = time.time()
    print(f"Pre-computation took {t_precompute - t0:.1f}s")

    # Define grid
    a_values = np.arange(0.0, 3.01, 0.25)  # cos power
    b_values = np.arange(0.0, 3.01, 0.25)  # dice power
    c_values = np.arange(0.5, 2.01, 0.25)  # entropy power

    total_combos = len(a_values) * len(b_values) * len(c_values)
    print(f"\nGrid: a(cos) in [{a_values[0]:.2f}, {a_values[-1]:.2f}] step 0.25 ({len(a_values)} values)")
    print(f"      b(dice) in [{b_values[0]:.2f}, {b_values[-1]:.2f}] step 0.25 ({len(b_values)} values)")
    print(f"      c(ent) in [{c_values[0]:.2f}, {c_values[-1]:.2f}] step 0.25 ({len(c_values)} values)")
    print(f"Total combinations: {total_combos}")

    # Run grid search
    results = []
    done = 0

    print(f"\nRunning grid search...")
    for a in a_values:
        for b in b_values:
            for c in c_values:
                top1, top3, n_total = evaluate_power(compounds, a, b, c)
                results.append({
                    'a_cos': round(a, 2),
                    'b_dice': round(b, 2),
                    'c_ent': round(c, 2),
                    'top1': top1,
                    'top3': top3,
                    'n_total': n_total,
                    'top1_pct': round(top1 / n_total * 100, 1) if n_total > 0 else 0,
                    'top3_pct': round(top3 / n_total * 100, 1) if n_total > 0 else 0,
                })
                done += 1

        # Progress every 13 a-values (i.e., every ~1000 combos)
        pct = done / total_combos * 100
        best_so_far = max(r['top1'] for r in results)
        print(f"  a={a:.2f} done ({done}/{total_combos}, {pct:.0f}%) -- best TOP1 so far: {best_so_far}/{n_total}")

    t_search = time.time()
    print(f"\nGrid search took {t_search - t_precompute:.1f}s")
    print(f"Total time: {t_search - t0:.1f}s")

    # Sort: primary by TOP1 desc, secondary by TOP3 desc
    results.sort(key=lambda x: (-x['top1'], -x['top3'], x['a_cos'], x['b_dice'], x['c_ent']))

    # Save all results to CSV
    out_csv = BASE / 'power_grid_search_results.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved all {len(results)} results to {out_csv}")

    # Print top 20
    print(f"\n{'='*90}")
    print(f"  TOP 20 COMBINATIONS (sorted by TOP1, then TOP3)")
    print(f"  Baseline: ent^1 * cos^1 * dice^1 -> 61/143 = 42.7%")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'a(cos)':>7} {'b(dice)':>8} {'c(ent)':>7} {'TOP1':>10} {'TOP1%':>7} {'TOP3':>10} {'TOP3%':>7}")
    print('-' * 90)

    for rank, r in enumerate(results[:20], 1):
        print(f"{rank:<5} {r['a_cos']:>7.2f} {r['b_dice']:>8.2f} {r['c_ent']:>7.2f} "
              f"{r['top1']:>4}/{r['n_total']:<5} {r['top1_pct']:>6.1f}% "
              f"{r['top3']:>4}/{r['n_total']:<5} {r['top3_pct']:>6.1f}%")

    # Also print: what is the best TOP3 regardless of TOP1?
    results_by_top3 = sorted(results, key=lambda x: (-x['top3'], -x['top1']))
    print(f"\n{'='*90}")
    print(f"  TOP 10 by TOP3 (regardless of TOP1)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'a(cos)':>7} {'b(dice)':>8} {'c(ent)':>7} {'TOP1':>10} {'TOP1%':>7} {'TOP3':>10} {'TOP3%':>7}")
    print('-' * 90)
    for rank, r in enumerate(results_by_top3[:10], 1):
        print(f"{rank:<5} {r['a_cos']:>7.2f} {r['b_dice']:>8.2f} {r['c_ent']:>7.2f} "
              f"{r['top1']:>4}/{r['n_total']:<5} {r['top1_pct']:>6.1f}% "
              f"{r['top3']:>4}/{r['n_total']:<5} {r['top3_pct']:>6.1f}%")

    # Verify baseline
    print(f"\n{'='*90}")
    print(f"  BASELINE VERIFICATION (a=1.0, b=1.0, c=1.0)")
    print(f"{'='*90}")
    baseline = [r for r in results if r['a_cos'] == 1.0 and r['b_dice'] == 1.0 and r['c_ent'] == 1.0]
    if baseline:
        bl = baseline[0]
        print(f"  TOP1: {bl['top1']}/{bl['n_total']} ({bl['top1_pct']}%)")
        print(f"  TOP3: {bl['top3']}/{bl['n_total']} ({bl['top3_pct']}%)")
    else:
        print("  WARNING: baseline combination not found!")

    # Show how many unique TOP1 values exist
    unique_top1 = sorted(set(r['top1'] for r in results), reverse=True)
    print(f"\n  Unique TOP1 values: {unique_top1[:15]}...")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
