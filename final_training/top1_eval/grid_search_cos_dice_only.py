#!/usr/bin/env python3
"""
Grid search: cos^a * dice^b with various energy weights (no entropy).
Pre-compute per-energy cos/dice once, then sweep (w0, w1, w2, a, b).
"""
import math, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ENERGIES = [0, 1, 2]

def parse_spectrum(path, energy=0):
    peaks = []
    target = f'energy{energy}'
    in_target = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == target:
                in_target = True; peaks = []; continue
            if in_target:
                if line.startswith('energy') or line == '': break
                parts = line.split()
                if len(parts) >= 2:
                    try: peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError: pass
    return peaks

def normalize_spectrum(peaks):
    if not peaks: return []
    mx = max(p[1] for p in peaks)
    if mx <= 0: return []
    return [(mz, i/mx) for mz, i in peaks]

def match_peaks(spec1, spec2, mz_tol=0.01):
    s1, s2 = normalize_spectrum(spec1), normalize_spectrum(spec2)
    if not s1 or not s2: return [], s1, s2
    used2 = set(); matched = []
    for mz1, i1 in s1:
        best_j, best_diff = -1, mz_tol + 1
        for j, (mz2, i2) in enumerate(s2):
            if j in used2: continue
            diff = abs(mz1 - mz2)
            if diff <= mz_tol and diff < best_diff:
                best_j, best_diff = j, diff
        if best_j >= 0:
            used2.add(best_j); matched.append((i1, s2[best_j][1]))
    return matched, s1, s2

def cosine_similarity(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not matched or not s1 or not s2: return 0.0
    dot = sum(a*b for a,b in matched)
    denom = math.sqrt(sum(i**2 for _,i in s1) * sum(i**2 for _,i in s2))
    return dot/denom if denom > 0 else 0.0

def dice_similarity(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    total = len(s1) + len(s2)
    return 2.0*len(matched)/total if total > 0 else 0.0

def precompute(df, exp_dir, pred_dir):
    compounds = []
    for lib_name in sorted(df['lib_name'].unique()):
        sub = df[df['lib_name'] == lib_name]
        hmdb_id = str(sub['hmdb_id'].iloc[0]) if pd.notna(sub['hmdb_id'].iloc[0]) else ''
        exp_file = exp_dir / f'{lib_name}.txt'
        if not exp_file.exists(): continue
        exp_specs = {e: parse_spectrum(exp_file, e) for e in ENERGIES}
        cands = []
        for _, row in sub.iterrows():
            cand_hmdb = str(row['cand_hmdb_id']) if pd.notna(row.get('cand_hmdb_id', None)) else ''
            pred_file = pred_dir / f'{row["cand_id"]}.log'
            if not pred_file.exists():
                cands.append({'hmdb': cand_hmdb, 'cos': {e: 0.0 for e in ENERGIES}, 'dice': {e: 0.0 for e in ENERGIES}})
                continue
            pred_specs = {e: parse_spectrum(pred_file, e) for e in ENERGIES}
            cos_e = {e: cosine_similarity(exp_specs[e], pred_specs[e]) for e in ENERGIES}
            dice_e = {e: dice_similarity(exp_specs[e], pred_specs[e]) for e in ENERGIES}
            cands.append({'hmdb': cand_hmdb, 'cos': cos_e, 'dice': dice_e})
        if cands:
            compounds.append({'hmdb_id': hmdb_id, 'lib_name': lib_name, 'candidates': cands})
    return compounds

def evaluate(compounds, w0, w1, w2, a, b):
    top1 = top3 = 0; n = 0
    for comp in compounds:
        if not comp['hmdb_id']: continue
        n += 1
        scored = []
        for c in comp['candidates']:
            cos_w = w0*c['cos'][0] + w1*c['cos'][1] + w2*c['cos'][2]
            dice_w = w0*c['dice'][0] + w1*c['dice'][1] + w2*c['dice'][2]
            try: s = (cos_w**a) * (dice_w**b)
            except: s = 0.0
            scored.append((s, c['hmdb']))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][1] == comp['hmdb_id']: top1 += 1
        for i in range(min(3, len(scored))):
            if scored[i][1] == comp['hmdb_id']: top3 += 1; break
    return top1, top3, n

def main():
    t0 = time.time()
    print("=" * 90)
    print("  Grid Search: cos^a * dice^b  (NO entropy)")
    print("  Sweeping energy weights + power exponents")
    print("=" * 90)

    df = pd.read_pickle(BASE / 'candidates.pkl')
    compounds = precompute(df, BASE.parent / 'full_model' / 'spectra',
                           BASE / 'full_model' / 'predictions')
    print(f"Pre-compute: {time.time()-t0:.1f}s, {len(compounds)} compounds")

    # Energy weight grid (step 0.1, sum=1)
    w_grid = []
    for w0 in np.arange(0.0, 1.01, 0.1):
        for w1 in np.arange(0.0, 1.01 - w0, 0.1):
            w2 = round(1.0 - w0 - w1, 2)
            if w2 >= 0:
                w_grid.append((round(w0,2), round(w1,2), round(w2,2)))
    # Power grid
    a_vals = np.arange(0.0, 3.01, 0.25)
    b_vals = np.arange(0.0, 3.01, 0.25)
    total = len(w_grid) * len(a_vals) * len(b_vals)
    print(f"Energy weight combos: {len(w_grid)}, Power combos: {len(a_vals)*len(b_vals)}")
    print(f"Total: {total} combinations\n")

    results = []; done = 0
    for w0, w1, w2 in w_grid:
        for a in a_vals:
            for b in b_vals:
                top1, top3, n = evaluate(compounds, w0, w1, w2, a, b)
                results.append({'w0':w0,'w1':w1,'w2':w2,'a_cos':round(a,2),'b_dice':round(b,2),
                                'top1':top1,'top3':top3,'n':n,
                                'top1_pct':round(top1/n*100,1),'top3_pct':round(top3/n*100,1)})
                done += 1
        if done % 5000 < len(a_vals)*len(b_vals):
            best = max(r['top1'] for r in results)
            print(f"  {done}/{total} ({done/total*100:.0f}%) w=({w0},{w1},{w2}) best={best}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    results.sort(key=lambda x: (-x['top1'], -x['top3']))
    out = BASE / 'grid_cos_dice_only.csv'
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved {len(results)} results to {out}")

    print(f"\n{'='*90}")
    print(f"  TOP 20 (cos^a * dice^b, no entropy)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'w0':>4} {'w1':>4} {'w2':>4} {'a':>5} {'b':>6} {'TOP1':>10} {'TOP1%':>7} {'TOP3':>10} {'TOP3%':>7}")
    print('-'*90)
    for i, r in enumerate(results[:20], 1):
        print(f"{i:<5} {r['w0']:>4} {r['w1']:>4} {r['w2']:>4} {r['a_cos']:>5} {r['b_dice']:>6} "
              f"{r['top1']:>4}/{r['n']:<5} {r['top1_pct']:>6.1f}% {r['top3']:>4}/{r['n']:<5} {r['top3_pct']:>6.1f}%")

    # Best by TOP3
    by_t3 = sorted(results, key=lambda x: (-x['top3'], -x['top1']))
    print(f"\n{'='*90}")
    print(f"  TOP 10 by TOP3")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'w0':>4} {'w1':>4} {'w2':>4} {'a':>5} {'b':>6} {'TOP1':>10} {'TOP1%':>7} {'TOP3':>10} {'TOP3%':>7}")
    print('-'*90)
    for i, r in enumerate(by_t3[:10], 1):
        print(f"{i:<5} {r['w0']:>4} {r['w1']:>4} {r['w2']:>4} {r['a_cos']:>5} {r['b_dice']:>6} "
              f"{r['top1']:>4}/{r['n']:<5} {r['top1_pct']:>6.1f}% {r['top3']:>4}/{r['n']:<5} {r['top3_pct']:>6.1f}%")

    # Baseline comparison
    print(f"\n{'='*90}")
    print(f"  BASELINES")
    print(f"{'='*90}")
    for label, w0,w1,w2,a,b in [
        ("cos only (equal)",       0.33,0.33,0.34, 1.0, 0.0),
        ("cos*dice (equal)",       0.33,0.33,0.34, 1.0, 1.0),
        ("cos*dice (0.3:0.3:0.4)",0.3, 0.3, 0.4,  1.0, 1.0),
        ("ent*cos*dice (0.3:0.3:0.4)", 0.3,0.3,0.4, 1.0, 1.0),
    ]:
        top1, top3, n = evaluate(compounds, w0, w1, w2, a, b)
        print(f"  {label:<35} TOP1={top1}/{n} ({top1/n*100:.1f}%)  TOP3={top3}/{n} ({top3/n*100:.1f}%)")

if __name__ == '__main__':
    main()
