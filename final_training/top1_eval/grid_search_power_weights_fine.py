#!/usr/bin/env python3
"""
grid_search_power_weights_fine.py
=================================
Fine-grained grid search around the best region found in coarse search.

Best coarse result: a=0.50, b=1.00, c=0.75 -> 62/143 TOP1 (43.4%)

Now search with step 0.05 in a wider range to find any hidden peaks.
Also extends a up to 4.0 and b up to 4.0 to check extreme values.
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
    s = 0
    for e in ENERGIES:
        s += ENERGY_WEIGHTS[e] * metric_fn(exp_specs.get(e, []), pred_specs.get(e, []))
    return s


# ============================================================
# Pre-compute
# ============================================================
def precompute_scores(df, exp_spectra_dir, pred_dir):
    compounds = []
    lib_names = sorted(df['lib_name'].unique())

    print(f"Pre-computing scores for {len(lib_names)} compounds...")
    for li, lib_name in enumerate(lib_names):
        if (li + 1) % 30 == 0:
            print(f"  {li+1}/{len(lib_names)}...")

        sub = df[df['lib_name'] == lib_name]
        hmdb_id = str(sub['hmdb_id'].iloc[0]) if pd.notna(sub['hmdb_id'].iloc[0]) else ''

        exp_file = exp_spectra_dir / f'{lib_name}.txt'
        if not exp_file.exists():
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

        if cand_list:
            compounds.append({'hmdb_id': hmdb_id, 'lib_name': lib_name, 'candidates': cand_list})

    print(f"  Done. {len(compounds)} compounds loaded.")
    return compounds


def evaluate_power(compounds, a, b, c):
    top1_correct = 0
    top3_correct = 0
    n_total = 0

    for comp in compounds:
        hmdb_id = comp['hmdb_id']
        if not hmdb_id:
            continue
        n_total += 1

        scored = []
        for cand in comp['candidates']:
            try:
                score = (cand['ent'] ** c) * (cand['cos'] ** a) * (cand['dice'] ** b)
            except (ValueError, ZeroDivisionError):
                score = 0.0
            scored.append((score, cand['cand_hmdb']))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored and scored[0][1] == hmdb_id:
            top1_correct += 1

        for rank_i in range(min(3, len(scored))):
            if scored[rank_i][1] == hmdb_id:
                top3_correct += 1
                break

    return top1_correct, top3_correct, n_total


def main():
    t0 = time.time()

    print("=" * 90)
    print("  Fine Grid Search: entropy^c * cos^a * dice^b")
    print("  Step = 0.05, extended range")
    print("=" * 90)

    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'
    pred_dir = BASE / 'full_model' / 'predictions'

    compounds = precompute_scores(df, exp_spectra_dir, pred_dir)

    t_precompute = time.time()
    print(f"Pre-computation took {t_precompute - t0:.1f}s")

    # Fine grid: focus on ranges where best results were found
    # Coarse search showed best at a~0.5-2.0, b~1.0-3.0, c~0.5-2.0
    a_values = np.arange(0.0, 4.01, 0.10)  # cos power, finer step
    b_values = np.arange(0.0, 4.01, 0.10)  # dice power, finer step
    c_values = np.arange(0.0, 3.01, 0.10)  # entropy power, finer step

    total_combos = len(a_values) * len(b_values) * len(c_values)
    print(f"\nGrid: a(cos)  in [0.00, 4.00] step 0.10 ({len(a_values)} values)")
    print(f"      b(dice) in [0.00, 4.00] step 0.10 ({len(b_values)} values)")
    print(f"      c(ent)  in [0.00, 3.00] step 0.10 ({len(c_values)} values)")
    print(f"Total combinations: {total_combos}")

    results = []
    done = 0
    best_top1 = 0

    print(f"\nRunning grid search...")
    for a in a_values:
        for b in b_values:
            for c in c_values:
                top1, top3, n_total = evaluate_power(compounds, a, b, c)
                if top1 > best_top1:
                    best_top1 = top1
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

        if int(a * 10) % 5 == 0:  # Print every 0.5 step
            pct = done / total_combos * 100
            elapsed = time.time() - t_precompute
            rate = done / elapsed if elapsed > 0 else 1
            eta = (total_combos - done) / rate if rate > 0 else 0
            print(f"  a={a:.2f} done ({done}/{total_combos}, {pct:.0f}%) "
                  f"-- best TOP1: {best_top1}/{n_total} -- "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    t_search = time.time()
    print(f"\nGrid search took {t_search - t_precompute:.1f}s")
    print(f"Total time: {t_search - t0:.1f}s")

    # Sort
    results.sort(key=lambda x: (-x['top1'], -x['top3'], x['a_cos'], x['b_dice'], x['c_ent']))

    # Save
    out_csv = BASE / 'power_grid_search_fine_results.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved all {len(results)} results to {out_csv}")

    # Print top 30
    print(f"\n{'='*90}")
    print(f"  TOP 30 COMBINATIONS (sorted by TOP1, then TOP3)")
    print(f"  Baseline: ent^1 * cos^1 * dice^1 -> 61/143 = 42.7%")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'a(cos)':>7} {'b(dice)':>8} {'c(ent)':>7} {'TOP1':>10} {'TOP1%':>7} {'TOP3':>10} {'TOP3%':>7}")
    print('-' * 90)

    for rank, r in enumerate(results[:30], 1):
        print(f"{rank:<5} {r['a_cos']:>7.2f} {r['b_dice']:>8.2f} {r['c_ent']:>7.2f} "
              f"{r['top1']:>4}/{r['n_total']:<5} {r['top1_pct']:>6.1f}% "
              f"{r['top3']:>4}/{r['n_total']:<5} {r['top3_pct']:>6.1f}%")

    # Best TOP3
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
    baseline = [r for r in results if r['a_cos'] == 1.0 and r['b_dice'] == 1.0 and r['c_ent'] == 1.0]
    if baseline:
        bl = baseline[0]
        print(f"\nBaseline (a=1, b=1, c=1): TOP1={bl['top1']}/{bl['n_total']} ({bl['top1_pct']}%), "
              f"TOP3={bl['top3']}/{bl['n_total']} ({bl['top3_pct']}%)")

    # Count how many combos achieve the best TOP1
    best_combos = [r for r in results if r['top1'] == best_top1]
    print(f"\nTotal combinations achieving best TOP1 ({best_top1}/{n_total}): {len(best_combos)}")

    unique_top1 = sorted(set(r['top1'] for r in results), reverse=True)
    print(f"Unique TOP1 values: {unique_top1[:20]}...")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
