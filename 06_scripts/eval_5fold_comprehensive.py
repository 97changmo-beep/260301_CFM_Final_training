#!/usr/bin/env python3
"""
5-Fold CV test set: comprehensive energy weight evaluation.
Compute Cosine, DICE, Matched Peaks, WCS per energy combo from raw spectra.
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ENERGIES = [0, 1, 2]

def parse_spectrum(path, energy=0):
    peaks = []; target = f'energy{energy}'; in_target = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == target: in_target = True; peaks = []; continue
            if in_target:
                if line.startswith('energy') or line.startswith('#') or line == '': break
                parts = line.split()
                if len(parts) >= 2:
                    try: peaks.append((float(parts[0]), float(parts[1])))
                    except: pass
    return peaks

def normalize_spectrum(peaks):
    if not peaks: return []
    mx = max(p[1] for p in peaks)
    return [(mz, i/mx) for mz, i in peaks] if mx > 0 else []

def match_peaks(spec1, spec2, mz_tol=0.01):
    s1, s2 = normalize_spectrum(spec1), normalize_spectrum(spec2)
    if not s1 or not s2: return [], s1, s2
    used2 = set(); matched = []
    for mz1, i1 in s1:
        bj, bd = -1, mz_tol+1
        for j,(mz2,i2) in enumerate(s2):
            if j in used2: continue
            d = abs(mz1-mz2)
            if d <= mz_tol and d < bd: bj, bd = j, d
        if bj >= 0: used2.add(bj); matched.append((i1, s2[bj][1]))
    return matched, s1, s2

def cosine_sim(s1, s2):
    m, n1, n2 = match_peaks(s1, s2)
    if not m or not n1 or not n2: return 0.0
    dot = sum(a*b for a,b in m)
    denom = math.sqrt(sum(i**2 for _,i in n1)*sum(i**2 for _,i in n2))
    return dot/denom if denom > 0 else 0.0

def dice_sim(s1, s2):
    m, n1, n2 = match_peaks(s1, s2)
    t = len(n1)+len(n2)
    return 2.0*len(m)/t if t > 0 else 0.0

def matched_count(s1, s2):
    m, _, _ = match_peaks(s1, s2)
    return len(m)

# ── Collect per-energy scores for each compound ──
rows = []
for fold in range(5):
    fd = BASE / f'fold_{fold}'
    test_dir = fd / 'test_spectra'
    for model_type, pred_subdir in [('trained','predictions'), ('baseline','baseline_predictions')]:
        pred_dir = fd / pred_subdir
        for exp_file in sorted(test_dir.glob('*.txt')):
            comp = exp_file.stem
            pred_file = pred_dir / f'{comp}.log'
            if not pred_file.exists(): continue
            r = {'fold': fold, 'model': model_type, 'compound': comp}
            for e in ENERGIES:
                es = parse_spectrum(exp_file, e)
                ps = parse_spectrum(pred_file, e)
                r[f'cos_e{e}'] = cosine_sim(es, ps)
                r[f'dice_e{e}'] = dice_sim(es, ps)
                r[f'mp_e{e}'] = matched_count(es, ps)
            rows.append(r)

df = pd.DataFrame(rows)
print(f"Total: {len(df)} rows ({len(df[df['model']=='trained'])} trained, {len(df[df['model']=='baseline'])} baseline)")

# ── Define energy combos ──
combos = {
    'E0':       {0:1.0, 1:0.0, 2:0.0},
    'E1':       {0:0.0, 1:1.0, 2:0.0},
    'E2':       {0:0.0, 1:0.0, 2:1.0},
    'E0+E1':    {0:0.5, 1:0.5, 2:0.0},
    'E0+E2':    {0:0.5, 1:0.0, 2:0.5},
    'E1+E2':    {0:0.0, 1:0.5, 2:0.5},
    'E0+E1+E2': {0:1/3, 1:1/3, 2:1/3},
}

# ── Evaluate each combo × model ──
print(f"\n{'='*120}")
print(f"  5-FOLD CV TEST SET: COMPREHENSIVE METRIC COMPARISON")
print(f"{'='*120}")

for model_type in ['trained', 'baseline']:
    mdf = df[df['model'] == model_type]
    print(f"\n{'─'*120}")
    print(f"  MODEL: {model_type.upper()} ({len(mdf)} compounds)")
    print(f"{'─'*120}")
    print(f"{'Combo':<12} {'Cosine':>8} {'DICE':>8} {'MP':>8} {'WCS':>10} {'cos*dice':>10} {'cos*sqrtMP':>12}")
    print(f"{'':12} {'':>8} {'':>8} {'':>8} {'(cos*mp)':>10} {'':>10} {'':>12}")
    print("-"*72)

    for cname, weights in combos.items():
        wcos = sum(weights[e]*mdf[f'cos_e{e}'] for e in ENERGIES)
        wdice = sum(weights[e]*mdf[f'dice_e{e}'] for e in ENERGIES)
        wmp = sum(weights[e]*mdf[f'mp_e{e}'] for e in ENERGIES)
        wcs = (wcos * wmp).mean()
        cos_dice = (wcos * wdice).mean()
        cos_sqrtmp = (wcos * np.sqrt(wmp)).mean()
        print(f"{cname:<12} {wcos.mean():>8.4f} {wdice.mean():>8.4f} {wmp.mean():>8.2f} "
              f"{wcs:>10.4f} {cos_dice:>10.4f} {cos_sqrtmp:>12.4f}")

# ── Fine grid search for weighted combos (trained only) ──
print(f"\n{'='*120}")
print(f"  ENERGY WEIGHT GRID SEARCH (trained, step=0.05)")
print(f"{'='*120}")

tr = df[df['model']=='trained']
w_grid = []
for w0 in np.arange(0.0, 1.01, 0.05):
    for w1 in np.arange(0.0, 1.01-w0, 0.05):
        w2 = round(1.0-w0-w1, 3)
        if w2 >= -0.001:
            w_grid.append((round(w0,2), round(w1,2), max(0,round(w2,2))))

grid_results = []
for w0, w1, w2 in w_grid:
    wcos = w0*tr['cos_e0']+w1*tr['cos_e1']+w2*tr['cos_e2']
    wdice = w0*tr['dice_e0']+w1*tr['dice_e1']+w2*tr['dice_e2']
    wmp = w0*tr['mp_e0']+w1*tr['mp_e1']+w2*tr['mp_e2']
    grid_results.append({
        'w0':w0,'w1':w1,'w2':w2,
        'cos': wcos.mean(),
        'dice': wdice.mean(),
        'mp': wmp.mean(),
        'wcs': (wcos*wmp).mean(),
        'cos_dice': (wcos*wdice).mean(),
        'cos_sqrtmp': (wcos*np.sqrt(wmp)).mean(),
    })

gdf = pd.DataFrame(grid_results)

for metric, label in [
    ('cos','Cosine'), ('dice','DICE'), ('mp','Matched Peaks'),
    ('wcs','WCS (cos*mp)'), ('cos_dice','cos*dice'), ('cos_sqrtmp','cos*sqrt(mp)')
]:
    top5 = gdf.nlargest(5, metric)
    print(f"\n--- Best by {label} ---")
    print(f"{'Rank':<5} {'w0':>5} {'w1':>5} {'w2':>5}  {'Cos':>7} {'DICE':>7} {'MP':>6} {'WCS':>8} {'cos*dice':>9} {'cos*sqrtMP':>11}")
    print("-"*80)
    for i,(_, r) in enumerate(top5.iterrows(), 1):
        print(f"{i:<5} {r['w0']:>5.2f} {r['w1']:>5.2f} {r['w2']:>5.2f}  "
              f"{r['cos']:>7.4f} {r['dice']:>7.4f} {r['mp']:>6.2f} "
              f"{r['wcs']:>8.4f} {r['cos_dice']:>9.4f} {r['cos_sqrtmp']:>11.4f}")

# ── Summary ──
print(f"\n{'='*120}")
print(f"  SUMMARY: Best energy weights per metric (Trained)")
print(f"{'='*120}")
print(f"{'Metric':<20} {'w0(E0)':>8} {'w1(E1)':>8} {'w2(E2)':>8} {'Value':>10}")
print("-"*60)
for metric, label in [
    ('cos','Cosine'), ('dice','DICE'), ('mp','Matched Peaks'),
    ('wcs','WCS'), ('cos_dice','cos*dice'), ('cos_sqrtmp','cos*sqrt(mp)')
]:
    best = gdf.loc[gdf[metric].idxmax()]
    print(f"{label:<20} {best['w0']:>8.2f} {best['w1']:>8.2f} {best['w2']:>8.2f} {best[metric]:>10.4f}")

