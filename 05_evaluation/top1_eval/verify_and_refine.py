#!/usr/bin/env python3
"""
verify_and_refine.py
====================
1. Verify the top-3 results from the exhaustive search
2. Do ultra-fine grid search (step=0.01) around best energy weights for top methods
3. Try a few more creative approaches:
   - rank-based fusion
   - per-energy product scoring (not averaged)
   - additional metric/power combos near the optimum
"""

import sys, os, math, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ENERGIES = [0, 1, 2]
MZ_TOL = 0.01

# ============================================================
# Reuse functions from exhaustive search
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
                if line.startswith('energy') or line == '' or line.startswith('#'):
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

def match_peaks(s1_norm, s2_norm, mz_tol=MZ_TOL):
    if not s1_norm or not s2_norm:
        return [], s1_norm, s2_norm
    used2 = set()
    matched = []
    for mz1, i1 in s1_norm:
        best_j = -1
        best_diff = mz_tol + 1
        for j, (mz2, i2) in enumerate(s2_norm):
            if j in used2:
                continue
            diff = abs(mz1 - mz2)
            if diff <= mz_tol and diff < best_diff:
                best_j = j
                best_diff = diff
        if best_j >= 0:
            used2.add(best_j)
            matched.append((mz1, i1, s2_norm[best_j][0], s2_norm[best_j][1]))
    return matched, s1_norm, s2_norm

# Metric functions
def calc_cosine(matched, s1, s2):
    if not matched or not s1 or not s2: return 0.0
    dot = sum(i1 * i2 for _, i1, _, i2 in matched)
    denom = math.sqrt(sum(i**2 for _, i in s1) * sum(i**2 for _, i in s2))
    return dot / denom if denom > 0 else 0.0

def calc_weighted_cosine(matched, s1, s2, mz_power=0.5):
    if not matched or not s1 or not s2: return 0.0
    dot = sum((mz1**mz_power)*i1*(mz2**mz_power)*i2 for mz1, i1, mz2, i2 in matched)
    s1s = sum((mz**mz_power*i)**2 for mz, i in s1)
    s2s = sum((mz**mz_power*i)**2 for mz, i in s2)
    denom = math.sqrt(s1s * s2s)
    return dot / denom if denom > 0 else 0.0

def calc_dice(matched, s1, s2):
    total = len(s1) + len(s2)
    return 2.0 * len(matched) / total if total > 0 else 0.0

def calc_matched_ratio(matched, s1, s2):
    if not s1 or not s2: return 0.0
    denom = min(len(s1), len(s2))
    return len(matched) / denom if denom > 0 else 0.0

def calc_jaccard(matched, s1, s2):
    denom = len(s1) + len(s2) - len(matched)
    return len(matched) / denom if denom > 0 else 0.0

def _to_prob(intensities):
    s = sum(intensities)
    if s <= 0: return []
    return [x / s for x in intensities]

def _entropy(probs):
    return -sum(p * math.log(p + 1e-30) for p in probs if p > 0)

def calc_spectral_entropy(matched, s1, s2):
    if not s1 or not s2: return 0.0
    vec1, vec2 = [], []
    matched_idx_s1, matched_idx_s2 = set(), set()
    for mz1, i1, mz2, i2 in matched:
        vec1.append(i1); vec2.append(i2)
        for idx, (m, ii) in enumerate(s1):
            if abs(m-mz1)<1e-10 and abs(ii-i1)<1e-10 and idx not in matched_idx_s1:
                matched_idx_s1.add(idx); break
        for idx, (m, ii) in enumerate(s2):
            if abs(m-mz2)<1e-10 and abs(ii-i2)<1e-10 and idx not in matched_idx_s2:
                matched_idx_s2.add(idx); break
    for idx, (mz, i) in enumerate(s1):
        if idx not in matched_idx_s1:
            vec1.append(i); vec2.append(0.0)
    for idx, (mz, i) in enumerate(s2):
        if idx not in matched_idx_s2:
            vec1.append(0.0); vec2.append(i)
    if not vec1: return 0.0
    p, q = _to_prob(vec1), _to_prob(vec2)
    if not p or not q: return 0.0
    m_dist = [(pi+qi)/2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi*math.log((pi+1e-30)/(mi+1e-30)) for pi, mi in zip(p, m_dist) if pi > 0)
    kl_qm = sum(qi*math.log((qi+1e-30)/(mi+1e-30)) for qi, mi in zip(q, m_dist) if qi > 0)
    return max(0.0, min(1.0, 1.0 - (0.5*kl_pm + 0.5*kl_qm)/math.log(2)))

def calc_iw_dice(matched, s1, s2):
    if not s1 or not s2: return 0.0
    matched_sum = sum(i1 + i2 for _, i1, _, i2 in matched)
    total_sum = sum(i for _, i in s1) + sum(i for _, i in s2)
    return matched_sum / total_sum if total_sum > 0 else 0.0

def calc_sqrt_cosine(matched, s1, s2):
    if not matched or not s1 or not s2: return 0.0
    dot = sum(math.sqrt(i1)*math.sqrt(i2) for _, i1, _, i2 in matched)
    denom = math.sqrt(sum(i for _, i in s1) * sum(i for _, i in s2))
    return dot / denom if denom > 0 else 0.0

def calc_weighted_entropy(matched, s1, s2):
    if not s1 or not s2: return 0.0
    vec1, vec2 = [], []
    matched_idx_s1, matched_idx_s2 = set(), set()
    for mz1, i1, mz2, i2 in matched:
        w = math.sqrt((mz1+mz2)/2)
        vec1.append(i1*w); vec2.append(i2*w)
        for idx, (m, ii) in enumerate(s1):
            if abs(m-mz1)<1e-10 and abs(ii-i1)<1e-10 and idx not in matched_idx_s1:
                matched_idx_s1.add(idx); break
        for idx, (m, ii) in enumerate(s2):
            if abs(m-mz2)<1e-10 and abs(ii-i2)<1e-10 and idx not in matched_idx_s2:
                matched_idx_s2.add(idx); break
    for idx, (mz, i) in enumerate(s1):
        if idx not in matched_idx_s1:
            vec1.append(i*math.sqrt(mz)); vec2.append(0.0)
    for idx, (mz, i) in enumerate(s2):
        if idx not in matched_idx_s2:
            vec1.append(0.0); vec2.append(i*math.sqrt(mz))
    if not vec1: return 0.0
    p, q = _to_prob(vec1), _to_prob(vec2)
    if not p or not q: return 0.0
    m_dist = [(pi+qi)/2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi*math.log((pi+1e-30)/(mi+1e-30)) for pi, mi in zip(p, m_dist) if pi > 0)
    kl_qm = sum(qi*math.log((qi+1e-30)/(mi+1e-30)) for qi, mi in zip(q, m_dist) if qi > 0)
    return max(0.0, min(1.0, 1.0 - (0.5*kl_pm + 0.5*kl_qm)/math.log(2)))

def calc_rev_entropy(matched, s1, s2):
    if not matched: return 0.0
    intensities = [i1*i2 for _, i1, _, i2 in matched]
    if not intensities or sum(intensities) <= 0: return 0.0
    p = _to_prob(intensities)
    ent = _entropy(p)
    max_ent = math.log(len(p)) if len(p) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


# ============================================================
# Precompute
# ============================================================
def precompute(df, exp_spectra_dir, pred_dir):
    metric_fns = {
        'cos': calc_cosine, 'wcos': calc_weighted_cosine, 'dice': calc_dice,
        'jaccard': calc_jaccard, 'entropy': calc_spectral_entropy,
        'w_entropy': calc_weighted_entropy, 'rev_entropy': calc_rev_entropy,
        'matched_ratio': calc_matched_ratio, 'iw_dice': calc_iw_dice,
        'sqrt_cos': calc_sqrt_cosine,
    }
    metric_names = list(metric_fns.keys())
    mi = {name: idx for idx, name in enumerate(metric_names)}

    all_metrics = {}
    lib_info = {}

    for li, lib_name in enumerate(sorted(df['lib_name'].unique())):
        sub = df[df['lib_name'] == lib_name]
        hmdb_id = str(sub['hmdb_id'].iloc[0]) if pd.notna(sub['hmdb_id'].iloc[0]) else ''

        exp_file = exp_spectra_dir / f'{lib_name}.txt'
        if not exp_file.exists():
            continue

        exp_specs = {e: normalize_spectrum(parse_spectrum(exp_file, e)) for e in ENERGIES}

        n_cands = len(sub)
        cand_data = np.zeros((n_cands, 3, len(metric_names)))
        cand_hmdb_list = []

        for ci, (_, row) in enumerate(sub.iterrows()):
            cand_hmdb = str(row['cand_hmdb_id']) if pd.notna(row['cand_hmdb_id']) else ''
            cand_hmdb_list.append(cand_hmdb)
            pred_file = pred_dir / f'{row["cand_id"]}.log'
            if not pred_file.exists():
                continue
            for e in ENERGIES:
                pred_peaks = normalize_spectrum(parse_spectrum(pred_file, e))
                matched, s1, s2 = match_peaks(exp_specs[e], pred_peaks)
                for m_name, m_fn in metric_fns.items():
                    cand_data[ci, e, mi[m_name]] = m_fn(matched, s1, s2)

        all_metrics[lib_name] = cand_data
        lib_info[lib_name] = (hmdb_id, cand_hmdb_list)

    return all_metrics, lib_info, metric_names, mi


def evaluate(all_metrics, lib_info, score_fn, ew):
    w0, w1, w2 = ew
    n_correct = 0
    n_total = 0
    for lib_name, cand_data in all_metrics.items():
        hmdb_id, cand_hmdb_list = lib_info[lib_name]
        if not hmdb_id: continue
        n_total += 1
        weighted = w0*cand_data[:,0,:] + w1*cand_data[:,1,:] + w2*cand_data[:,2,:]
        scores = score_fn(weighted)
        best_idx = np.argmax(scores)
        if cand_hmdb_list[best_idx] == hmdb_id:
            n_correct += 1
    return n_correct, n_total


def evaluate_per_energy(all_metrics, lib_info, score_fn, ew):
    n_correct = 0
    n_total = 0
    for lib_name, cand_data in all_metrics.items():
        hmdb_id, cand_hmdb_list = lib_info[lib_name]
        if not hmdb_id: continue
        n_total += 1
        scores = score_fn(cand_data, ew)
        best_idx = np.argmax(scores)
        if cand_hmdb_list[best_idx] == hmdb_id:
            n_correct += 1
    return n_correct, n_total


def main():
    t0 = time.time()
    sys.stdout.write("="*80 + "\n")
    sys.stdout.write("  Verify & Ultra-Fine Refinement\n")
    sys.stdout.write("="*80 + "\n\n")
    sys.stdout.flush()

    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'
    pred_dir = BASE / 'full_model' / 'predictions'

    sys.stdout.write("Precomputing...\n")
    sys.stdout.flush()
    all_metrics, lib_info, metric_names, mi = precompute(df, exp_spectra_dir, pred_dir)
    sys.stdout.write(f"Precomputation done in {time.time()-t0:.1f}s\n\n")
    sys.stdout.flush()

    # ============================================================
    # PART 1: Verify top results
    # ============================================================
    sys.stdout.write("="*80 + "\n")
    sys.stdout.write("  VERIFICATION of top results\n")
    sys.stdout.write("="*80 + "\n")
    sys.stdout.flush()

    # Verify baseline: entropy * cos * dice, e0=0.3 e1=0.3 e2=0.4
    baseline_fn = lambda w: w[:, mi['entropy']] * w[:, mi['cos']] * w[:, mi['dice']]
    nc, nt = evaluate(all_metrics, lib_info, baseline_fn, (0.3, 0.3, 0.4))
    sys.stdout.write(f"Baseline (entropy*cos*dice, 0.3/0.3/0.4): {nc}/{nt} = {nc/nt*100:.1f}%\n")

    # Verify top-1: entropy^1.0*cos^0.5*iw_dice^1.0, e0=0.40 e1=0.35 e2=0.25
    top1_fn = lambda w: np.power(np.maximum(w[:, mi['entropy']], 0), 1.0) * \
                         np.power(np.maximum(w[:, mi['cos']], 0), 0.5) * \
                         np.power(np.maximum(w[:, mi['iw_dice']], 0), 1.0)
    nc, nt = evaluate(all_metrics, lib_info, top1_fn, (0.40, 0.35, 0.25))
    sys.stdout.write(f"Top-1 (entropy*cos^0.5*iw_dice, 0.40/0.35/0.25): {nc}/{nt} = {nc/nt*100:.1f}%\n")

    # Verify top-2: entropy^1.5*cos^1.0*iw_dice^2.0
    top2_fn = lambda w: np.power(np.maximum(w[:, mi['entropy']], 0), 1.5) * \
                         np.power(np.maximum(w[:, mi['cos']], 0), 1.0) * \
                         np.power(np.maximum(w[:, mi['iw_dice']], 0), 2.0)
    nc, nt = evaluate(all_metrics, lib_info, top2_fn, (0.40, 0.35, 0.25))
    sys.stdout.write(f"Top-2 (entropy^1.5*cos*iw_dice^2, 0.40/0.35/0.25): {nc}/{nt} = {nc/nt*100:.1f}%\n")
    sys.stdout.flush()

    # ============================================================
    # PART 2: Ultra-fine grid around best weights
    # ============================================================
    sys.stdout.write("\n" + "="*80 + "\n")
    sys.stdout.write("  ULTRA-FINE GRID (step=0.01) around best weights for top methods\n")
    sys.stdout.write("="*80 + "\n")
    sys.stdout.flush()

    # Focus: e0 in [0.25-0.55], e1 in [0.20-0.50], step=0.01
    ultra_fine_grid = []
    for w0 in np.arange(0.20, 0.56, 0.01):
        for w1 in np.arange(0.15, 0.51, 0.01):
            w2 = round(1.0 - w0 - w1, 4)
            if w2 < 0.05 or w2 > 0.55:
                continue
            ultra_fine_grid.append((round(w0, 2), round(w1, 2), round(w2, 2)))

    sys.stdout.write(f"Ultra-fine grid: {len(ultra_fine_grid)} weight combos\n")
    sys.stdout.flush()

    # Top methods to refine - extend with more power combos around the best
    top_methods = {}

    # Family 1: entropy^a * cos^b * iw_dice^c - explore more powers
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for b in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for c in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
                name = f'entropy^{a}*cos^{b}*iw_dice^{c}'
                top_methods[name] = lambda w, _a=a, _b=b, _c=c: \
                    np.power(np.maximum(w[:, mi['entropy']], 0), _a) * \
                    np.power(np.maximum(w[:, mi['cos']], 0), _b) * \
                    np.power(np.maximum(w[:, mi['iw_dice']], 0), _c)

    # Family 2: w_entropy^a * cos^b * iw_dice^c
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for b in [0.25, 0.5, 0.75, 1.0, 1.25]:
            for c in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
                name = f'w_entropy^{a}*cos^{b}*iw_dice^{c}'
                top_methods[name] = lambda w, _a=a, _b=b, _c=c: \
                    np.power(np.maximum(w[:, mi['w_entropy']], 0), _a) * \
                    np.power(np.maximum(w[:, mi['cos']], 0), _b) * \
                    np.power(np.maximum(w[:, mi['iw_dice']], 0), _c)

    # Family 3: entropy * sqrt_cos * iw_dice
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for b in [0.5, 0.75, 1.0, 1.25, 1.5]:
            for c in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                name = f'entropy^{a}*sqrt_cos^{b}*iw_dice^{c}'
                top_methods[name] = lambda w, _a=a, _b=b, _c=c: \
                    np.power(np.maximum(w[:, mi['entropy']], 0), _a) * \
                    np.power(np.maximum(w[:, mi['sqrt_cos']], 0), _b) * \
                    np.power(np.maximum(w[:, mi['iw_dice']], 0), _c)

    # Family 4: entropy * wcos * iw_dice
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for b in [0.5, 0.75, 1.0, 1.25, 1.5]:
            for c in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                name = f'entropy^{a}*wcos^{b}*iw_dice^{c}'
                top_methods[name] = lambda w, _a=a, _b=b, _c=c: \
                    np.power(np.maximum(w[:, mi['entropy']], 0), _a) * \
                    np.power(np.maximum(w[:, mi['wcos']], 0), _b) * \
                    np.power(np.maximum(w[:, mi['iw_dice']], 0), _c)

    # Family 5: entropy * cos * iw_dice * dice (4-metric)
    for a in [0.5, 1.0, 1.5]:
        for b in [0.25, 0.5, 1.0]:
            for c in [0.5, 1.0, 1.5, 2.0]:
                for d in [0.25, 0.5, 1.0]:
                    name = f'entropy^{a}*cos^{b}*iw_dice^{c}*dice^{d}'
                    top_methods[name] = lambda w, _a=a, _b=b, _c=c, _d=d: \
                        np.power(np.maximum(w[:, mi['entropy']], 0), _a) * \
                        np.power(np.maximum(w[:, mi['cos']], 0), _b) * \
                        np.power(np.maximum(w[:, mi['iw_dice']], 0), _c) * \
                        np.power(np.maximum(w[:, mi['dice']], 0), _d)

    # Family 6: entropy * cos * iw_dice * matched_ratio
    for a in [0.5, 1.0, 1.5]:
        for b in [0.25, 0.5, 1.0]:
            for c in [0.5, 1.0, 1.5, 2.0]:
                for d in [0.25, 0.5, 1.0]:
                    name = f'entropy^{a}*cos^{b}*iw_dice^{c}*mr^{d}'
                    top_methods[name] = lambda w, _a=a, _b=b, _c=c, _d=d: \
                        np.power(np.maximum(w[:, mi['entropy']], 0), _a) * \
                        np.power(np.maximum(w[:, mi['cos']], 0), _b) * \
                        np.power(np.maximum(w[:, mi['iw_dice']], 0), _c) * \
                        np.power(np.maximum(w[:, mi['matched_ratio']], 0), _d)

    # Family 7: Linear combos with iw_dice
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -0.001: continue
            w3 = max(0.0, w3)
            w1r, w2r = round(w1, 2), round(w2, 2)
            name = f'lin({w1r}*cos+{w2r}*iw_dice+{w3}*entropy)'
            top_methods[name] = lambda w, _w1=w1r, _w2=w2r, _w3=w3: \
                _w1*w[:, mi['cos']] + _w2*w[:, mi['iw_dice']] + _w3*w[:, mi['entropy']]

    # Family 8: Additive: entropy*iw_dice + cos
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5]:
            name = f'entropy^{a}*iw_dice+cos^{b}'
            top_methods[name] = lambda w, _a=a, _b=b: \
                np.power(np.maximum(w[:, mi['entropy']], 0), _a) * w[:, mi['iw_dice']] + \
                np.power(np.maximum(w[:, mi['cos']], 0), _b)

    # Family 9: Per-energy product for iw_dice combos
    per_energy_methods = {}
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for b in [0.25, 0.5, 0.75, 1.0]:
            for c in [0.5, 0.75, 1.0, 1.5, 2.0]:
                name = f'PE_entropy^{a}*cos^{b}*iw_dice^{c}'
                def make_pe_fn(_a=a, _b=b, _c=c):
                    def fn(cand_data, ew):
                        n_cands = cand_data.shape[0]
                        scores = np.zeros(n_cands)
                        for e_idx in range(3):
                            e_score = np.power(np.maximum(cand_data[:, e_idx, mi['entropy']], 0), _a) * \
                                      np.power(np.maximum(cand_data[:, e_idx, mi['cos']], 0), _b) * \
                                      np.power(np.maximum(cand_data[:, e_idx, mi['iw_dice']], 0), _c)
                            scores += ew[e_idx] * e_score
                        return scores
                    return fn
                per_energy_methods[name] = make_pe_fn()

    # Also PE with jaccard (was good in phase 1)
    for a in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for b in [0.5, 0.75, 1.0, 1.5]:
            for c in [0.5, 0.75, 1.0, 1.5]:
                name = f'PE_entropy^{a}*cos^{b}*jaccard^{c}'
                def make_pe_fn2(_a=a, _b=b, _c=c):
                    def fn(cand_data, ew):
                        n_cands = cand_data.shape[0]
                        scores = np.zeros(n_cands)
                        for e_idx in range(3):
                            e_score = np.power(np.maximum(cand_data[:, e_idx, mi['entropy']], 0), _a) * \
                                      np.power(np.maximum(cand_data[:, e_idx, mi['cos']], 0), _b) * \
                                      np.power(np.maximum(cand_data[:, e_idx, mi['jaccard']], 0), _c)
                            scores += ew[e_idx] * e_score
                        return scores
                    return fn
                per_energy_methods[name] = make_pe_fn2()

    sys.stdout.write(f"Methods to refine: {len(top_methods)} weighted-avg + {len(per_energy_methods)} per-energy\n")
    sys.stdout.write(f"Total evaluations: {(len(top_methods) + len(per_energy_methods)) * len(ultra_fine_grid)}\n\n")
    sys.stdout.flush()

    results = []
    done = 0
    total = (len(top_methods) + len(per_energy_methods)) * len(ultra_fine_grid)

    for score_name, score_fn in top_methods.items():
        for ew in ultra_fine_grid:
            nc, nt = evaluate(all_metrics, lib_info, score_fn, ew)
            results.append({
                'method': score_name,
                'e0_weight': ew[0], 'e1_weight': ew[1], 'e2_weight': ew[2],
                'n_correct': nc, 'n_total': nt,
                'accuracy': nc/nt*100 if nt > 0 else 0,
            })
            done += 1
        if done % 10000 < len(ultra_fine_grid):
            elapsed = time.time() - t0
            best = max(r['n_correct'] for r in results)
            sys.stdout.write(f"  {done}/{total} ({done/total*100:.1f}%) - {elapsed:.0f}s - best: {best}/{nt}\n")
            sys.stdout.flush()

    for score_name, score_fn in per_energy_methods.items():
        for ew in ultra_fine_grid:
            nc, nt = evaluate_per_energy(all_metrics, lib_info, score_fn, ew)
            results.append({
                'method': score_name,
                'e0_weight': ew[0], 'e1_weight': ew[1], 'e2_weight': ew[2],
                'n_correct': nc, 'n_total': nt,
                'accuracy': nc/nt*100 if nt > 0 else 0,
            })
            done += 1

    t_done = time.time()
    sys.stdout.write(f"\nRefinement done in {t_done - t0:.1f}s\n")
    sys.stdout.flush()

    results.sort(key=lambda x: (-x['accuracy'], x['method']))

    # Save
    out_csv = BASE / 'scoring_search_results_refined.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    sys.stdout.write(f"Saved {len(results)} results to {out_csv}\n")

    # Print top 30
    sys.stdout.write(f"\n{'='*110}\n")
    sys.stdout.write(f"  TOP 30 SCORING COMBINATIONS (Ultra-Fine Refinement)\n")
    sys.stdout.write(f"  (Baseline: entropy*cos*dice, e0=0.3 e1=0.3 e2=0.4 -> 61/143 = 42.7%)\n")
    sys.stdout.write(f"{'='*110}\n")
    sys.stdout.write(f"{'Rank':<5} {'Correct':>7} {'Acc%':>6} {'e0':>5} {'e1':>5} {'e2':>5}  {'Method'}\n")
    sys.stdout.write('-' * 110 + '\n')

    seen = set()
    rank = 0
    for r in results:
        key = (r['n_correct'], r['method'])
        if key in seen: continue
        seen.add(key)
        rank += 1
        if rank > 30: break
        sys.stdout.write(f"{rank:<5} {r['n_correct']:>4}/{r['n_total']:<3} {r['accuracy']:>5.1f}% "
                         f"{r['e0_weight']:>5.2f} {r['e1_weight']:>5.2f} {r['e2_weight']:>5.2f}  {r['method']}\n")

    # Show all combos achieving best accuracy
    best_acc = results[0]['accuracy']
    best_n = results[0]['n_correct']
    best_results = [r for r in results if r['n_correct'] == best_n]
    sys.stdout.write(f"\n{'='*110}\n")
    sys.stdout.write(f"  ALL COMBOS ACHIEVING BEST ({best_n}/{results[0]['n_total']} = {best_acc:.1f}%)\n")
    sys.stdout.write(f"{'='*110}\n")

    method_counts = Counter(r['method'] for r in best_results)
    sys.stdout.write(f"  {len(best_results)} total combos, {len(method_counts)} unique methods\n\n")
    for method, count in method_counts.most_common(50):
        mresults = [r for r in best_results if r['method'] == method]
        sys.stdout.write(f"  {method}: {count} weight combos\n")
        for mr in mresults[:3]:
            sys.stdout.write(f"    e0={mr['e0_weight']:.2f}, e1={mr['e1_weight']:.2f}, e2={mr['e2_weight']:.2f}\n")

    sys.stdout.write(f"\nTotal time: {time.time()-t0:.1f}s\n")
    sys.stdout.write("Done!\n")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
