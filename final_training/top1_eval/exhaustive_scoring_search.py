#!/usr/bin/env python3
"""
exhaustive_scoring_search.py
============================
Exhaustively search for the best scoring combination + energy weights
to maximize TOP1 accuracy (HMDB-ID matching) for full_model predictions.

Two-phase approach:
  Phase 1: Coarse search (energy weight step 0.1) over ALL scoring functions
  Phase 2: Fine search (energy weight step 0.05) for top methods from Phase 1
"""

import sys, os, math, csv, time, itertools, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
ENERGIES = [0, 1, 2]
MZ_TOL = 0.01

# ============================================================
# Spectrum parsing
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


# ============================================================
# Metric functions
# ============================================================
def calc_cosine(matched, s1, s2):
    if not matched or not s1 or not s2:
        return 0.0
    dot = sum(i1 * i2 for _, i1, _, i2 in matched)
    denom = math.sqrt(sum(i**2 for _, i in s1) * sum(i**2 for _, i in s2))
    return dot / denom if denom > 0 else 0.0


def calc_weighted_cosine(matched, s1, s2, mz_power=0.5):
    if not matched or not s1 or not s2:
        return 0.0
    dot = sum((mz1**mz_power) * i1 * (mz2**mz_power) * i2 for mz1, i1, mz2, i2 in matched)
    sum_sq1 = sum((mz**mz_power * i)**2 for mz, i in s1)
    sum_sq2 = sum((mz**mz_power * i)**2 for mz, i in s2)
    denom = math.sqrt(sum_sq1 * sum_sq2)
    return dot / denom if denom > 0 else 0.0


def calc_dice(matched, s1, s2):
    total = len(s1) + len(s2)
    return 2.0 * len(matched) / total if total > 0 else 0.0


def calc_matched_ratio(matched, s1, s2):
    if not s1 or not s2:
        return 0.0
    denom = min(len(s1), len(s2))
    return len(matched) / denom if denom > 0 else 0.0


def calc_jaccard(matched, s1, s2):
    denom = len(s1) + len(s2) - len(matched)
    return len(matched) / denom if denom > 0 else 0.0


def _to_prob(intensities):
    s = sum(intensities)
    if s <= 0:
        return []
    return [x / s for x in intensities]


def _entropy(probs):
    return -sum(p * math.log(p + 1e-30) for p in probs if p > 0)


def calc_spectral_entropy(matched, s1, s2):
    if not s1 or not s2:
        return 0.0
    matched_mzs_s1 = set()
    matched_mzs_s2 = set()
    vec1, vec2 = [], []
    for mz1, i1, mz2, i2 in matched:
        vec1.append(i1); vec2.append(i2)
        matched_mzs_s1.add(id((mz1, i1)))
        matched_mzs_s2.add(id((mz2, i2)))

    # Use index-based tracking for unmatched
    matched_idx_s1 = set()
    matched_idx_s2 = set()
    for mz1, i1, mz2, i2 in matched:
        for idx, (m, ii) in enumerate(s1):
            if abs(m - mz1) < 1e-10 and abs(ii - i1) < 1e-10 and idx not in matched_idx_s1:
                matched_idx_s1.add(idx)
                break
        for idx, (m, ii) in enumerate(s2):
            if abs(m - mz2) < 1e-10 and abs(ii - i2) < 1e-10 and idx not in matched_idx_s2:
                matched_idx_s2.add(idx)
                break

    for idx, (mz, i) in enumerate(s1):
        if idx not in matched_idx_s1:
            vec1.append(i); vec2.append(0.0)
    for idx, (mz, i) in enumerate(s2):
        if idx not in matched_idx_s2:
            vec1.append(0.0); vec2.append(i)

    if not vec1:
        return 0.0
    p = _to_prob(vec1)
    q = _to_prob(vec2)
    if not p or not q:
        return 0.0

    m_dist = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log((pi + 1e-30) / (mi + 1e-30)) for pi, mi in zip(p, m_dist) if pi > 0)
    kl_qm = sum(qi * math.log((qi + 1e-30) / (mi + 1e-30)) for qi, mi in zip(q, m_dist) if qi > 0)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, min(1.0, 1.0 - jsd / math.log(2)))


def calc_weighted_entropy(matched, s1, s2):
    if not s1 or not s2:
        return 0.0
    vec1, vec2 = [], []
    matched_idx_s1 = set()
    matched_idx_s2 = set()

    for mz1, i1, mz2, i2 in matched:
        w = math.sqrt((mz1 + mz2) / 2)
        vec1.append(i1 * w); vec2.append(i2 * w)
        for idx, (m, ii) in enumerate(s1):
            if abs(m - mz1) < 1e-10 and abs(ii - i1) < 1e-10 and idx not in matched_idx_s1:
                matched_idx_s1.add(idx)
                break
        for idx, (m, ii) in enumerate(s2):
            if abs(m - mz2) < 1e-10 and abs(ii - i2) < 1e-10 and idx not in matched_idx_s2:
                matched_idx_s2.add(idx)
                break

    for idx, (mz, i) in enumerate(s1):
        if idx not in matched_idx_s1:
            vec1.append(i * math.sqrt(mz)); vec2.append(0.0)
    for idx, (mz, i) in enumerate(s2):
        if idx not in matched_idx_s2:
            vec1.append(0.0); vec2.append(i * math.sqrt(mz))

    if not vec1:
        return 0.0
    p = _to_prob(vec1)
    q = _to_prob(vec2)
    if not p or not q:
        return 0.0

    m_dist = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log((pi + 1e-30) / (mi + 1e-30)) for pi, mi in zip(p, m_dist) if pi > 0)
    kl_qm = sum(qi * math.log((qi + 1e-30) / (mi + 1e-30)) for qi, mi in zip(q, m_dist) if qi > 0)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, min(1.0, 1.0 - jsd / math.log(2)))


def calc_reverse_entropy(matched, s1, s2):
    if not matched:
        return 0.0
    intensities = [i1 * i2 for _, i1, _, i2 in matched]
    if not intensities or sum(intensities) <= 0:
        return 0.0
    p = _to_prob(intensities)
    ent = _entropy(p)
    max_ent = math.log(len(p)) if len(p) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


def calc_iw_dice(matched, s1, s2):
    if not s1 or not s2:
        return 0.0
    matched_sum = sum(i1 + i2 for _, i1, _, i2 in matched)
    total_sum = sum(i for _, i in s1) + sum(i for _, i in s2)
    return matched_sum / total_sum if total_sum > 0 else 0.0


def calc_sqrt_cosine(matched, s1, s2):
    if not matched or not s1 or not s2:
        return 0.0
    dot = sum(math.sqrt(i1) * math.sqrt(i2) for _, i1, _, i2 in matched)
    sum_sq1 = sum(i for _, i in s1)
    sum_sq2 = sum(i for _, i in s2)
    denom = math.sqrt(sum_sq1 * sum_sq2)
    return dot / denom if denom > 0 else 0.0


# ============================================================
# Precompute
# ============================================================
def precompute_all_metrics(df, exp_spectra_dir, pred_dir):
    metric_fns = {
        'cos': calc_cosine,
        'wcos': calc_weighted_cosine,
        'dice': calc_dice,
        'jaccard': calc_jaccard,
        'entropy': calc_spectral_entropy,
        'w_entropy': calc_weighted_entropy,
        'rev_entropy': calc_reverse_entropy,
        'matched_ratio': calc_matched_ratio,
        'iw_dice': calc_iw_dice,
        'sqrt_cos': calc_sqrt_cosine,
    }

    # all_metrics[lib_name] = numpy array shape (n_cands, 3, n_metrics)
    all_metrics = {}
    lib_info = {}
    metric_names = list(metric_fns.keys())

    lib_names = sorted(df['lib_name'].unique())
    n_total = len(lib_names)
    n_skipped = 0

    sys.stdout.write(f"Precomputing metrics for {n_total} compounds...\n")
    sys.stdout.flush()

    for li, lib_name in enumerate(lib_names):
        if (li + 1) % 20 == 0:
            sys.stdout.write(f"  {li+1}/{n_total}...\n")
            sys.stdout.flush()

        sub = df[df['lib_name'] == lib_name]
        hmdb_id = str(sub['hmdb_id'].iloc[0]) if pd.notna(sub['hmdb_id'].iloc[0]) else ''

        exp_file = exp_spectra_dir / f'{lib_name}.txt'
        if not exp_file.exists():
            n_skipped += 1
            continue

        exp_specs = {}
        for e in ENERGIES:
            exp_specs[e] = normalize_spectrum(parse_spectrum(exp_file, e))

        n_cands = len(sub)
        cand_data = np.zeros((n_cands, 3, len(metric_names)))
        cand_hmdb_list = []

        for ci, (_, row) in enumerate(sub.iterrows()):
            cand_id = row['cand_id']
            cand_hmdb = str(row['cand_hmdb_id']) if pd.notna(row['cand_hmdb_id']) else ''
            cand_hmdb_list.append(cand_hmdb)

            pred_file = pred_dir / f'{cand_id}.log'
            if not pred_file.exists():
                continue

            for e in ENERGIES:
                pred_peaks = normalize_spectrum(parse_spectrum(pred_file, e))
                matched, s1, s2 = match_peaks(exp_specs[e], pred_peaks)
                for mi, (m_name, m_fn) in enumerate(metric_fns.items()):
                    cand_data[ci, e, mi] = m_fn(matched, s1, s2)

        all_metrics[lib_name] = cand_data
        lib_info[lib_name] = (hmdb_id, cand_hmdb_list)

    sys.stdout.write(f"  Done. {n_total - n_skipped} compounds loaded, {n_skipped} skipped.\n")
    sys.stdout.flush()
    return all_metrics, lib_info, metric_names


# ============================================================
# Fast vectorized evaluation
# ============================================================
def evaluate_scoring_fast(all_metrics, lib_info, score_fn_vec, energy_weights):
    """
    Vectorized evaluation.
    score_fn_vec: takes array (n_cands, n_metrics) -> array (n_cands,) of scores
    energy_weights: (w0, w1, w2) tuple
    """
    w0, w1, w2 = energy_weights
    n_correct = 0
    n_total = 0

    for lib_name, cand_data in all_metrics.items():
        hmdb_id, cand_hmdb_list = lib_info[lib_name]
        if not hmdb_id:
            continue
        n_total += 1

        # cand_data shape: (n_cands, 3, n_metrics)
        # Weighted average across energies
        weighted = w0 * cand_data[:, 0, :] + w1 * cand_data[:, 1, :] + w2 * cand_data[:, 2, :]
        # weighted shape: (n_cands, n_metrics)

        scores = score_fn_vec(weighted)  # (n_cands,)
        best_idx = np.argmax(scores)
        if cand_hmdb_list[best_idx] == hmdb_id:
            n_correct += 1

    return n_correct, n_total


def evaluate_scoring_per_energy(all_metrics, lib_info, score_fn_per_energy, energy_weights):
    """
    Evaluation where scoring uses per-energy metrics separately (not pre-averaged).
    score_fn_per_energy: takes (n_cands, 3, n_metrics) -> (n_cands,) scores
    """
    w0, w1, w2 = energy_weights
    n_correct = 0
    n_total = 0

    for lib_name, cand_data in all_metrics.items():
        hmdb_id, cand_hmdb_list = lib_info[lib_name]
        if not hmdb_id:
            continue
        n_total += 1

        scores = score_fn_per_energy(cand_data, (w0, w1, w2))
        best_idx = np.argmax(scores)
        if cand_hmdb_list[best_idx] == hmdb_id:
            n_correct += 1

    return n_correct, n_total


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.write("  Exhaustive Scoring Search for TOP1 Accuracy\n")
    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.flush()

    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'
    pred_dir = BASE / 'full_model' / 'predictions'

    all_metrics, lib_info, metric_names = precompute_all_metrics(df, exp_spectra_dir, pred_dir)

    n_total = sum(1 for ln in all_metrics if lib_info[ln][0])
    sys.stdout.write(f"\nTotal compounds for evaluation: {n_total}\n")

    t_precompute = time.time()
    sys.stdout.write(f"Precomputation took {t_precompute - t0:.1f}s\n")
    sys.stdout.flush()

    # Map metric names to indices
    mi = {name: idx for idx, name in enumerate(metric_names)}
    sys.stdout.write(f"Metric indices: {mi}\n")
    sys.stdout.flush()

    # ============================================================
    # Define energy weight grids
    # ============================================================
    def make_weight_grid(step):
        grid = []
        for w0 in np.arange(0, 1.001, step):
            for w1 in np.arange(0, 1.001 - w0, step):
                w2 = round(1.0 - w0 - w1, 4)
                if w2 < -0.001:
                    continue
                w2 = max(0.0, w2)
                grid.append((round(w0, 2), round(w1, 2), round(w2, 2)))
        return grid

    coarse_grid = make_weight_grid(0.1)
    fine_grid = make_weight_grid(0.05)

    sys.stdout.write(f"Coarse grid: {len(coarse_grid)} combos, Fine grid: {len(fine_grid)} combos\n")
    sys.stdout.flush()

    # ============================================================
    # Define scoring functions (vectorized)
    # Each takes array (n_cands, n_metrics) -> (n_cands,) scores
    # ============================================================
    scoring_functions = {}

    # Single metrics
    for m_name in metric_names:
        idx = mi[m_name]
        scoring_functions[m_name] = lambda w, _i=idx: w[:, _i]

    # Product combinations with powers
    powers = [0.5, 1.0, 1.5, 2.0]

    # Helper to create product scoring functions
    def make_product_fn(indices_powers):
        """indices_powers: list of (metric_idx, power)"""
        def fn(w):
            result = np.ones(w.shape[0])
            for idx, p in indices_powers:
                result *= np.power(np.maximum(w[:, idx], 0), p)
            return result
        return fn

    # entropy_type * cos * dice with powers
    for ent_name in ['entropy', 'w_entropy', 'rev_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'{ent_name}^{a}*cos^{b}*dice^{c}'
                    scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['cos'], b), (mi['dice'], c)])

    # cos * dice with powers
    for b in powers:
        for c in powers:
            name = f'cos^{b}*dice^{c}'
            scoring_functions[name] = make_product_fn([(mi['cos'], b), (mi['dice'], c)])

    # cos * jaccard with powers
    for b in powers:
        for c in powers:
            name = f'cos^{b}*jaccard^{c}'
            scoring_functions[name] = make_product_fn([(mi['cos'], b), (mi['jaccard'], c)])

    # entropy * cos with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                name = f'{ent_name}^{a}*cos^{b}'
                scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['cos'], b)])

    # entropy * dice with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for c in powers:
                name = f'{ent_name}^{a}*dice^{c}'
                scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['dice'], c)])

    # cos * matched_ratio with powers
    for b in powers:
        for c in powers:
            name = f'cos^{b}*matched_ratio^{c}'
            scoring_functions[name] = make_product_fn([(mi['cos'], b), (mi['matched_ratio'], c)])

    # wcos * dice with powers
    for b in powers:
        for c in powers:
            name = f'wcos^{b}*dice^{c}'
            scoring_functions[name] = make_product_fn([(mi['wcos'], b), (mi['dice'], c)])

    # sqrt_cos * dice with powers
    for b in powers:
        for c in powers:
            name = f'sqrt_cos^{b}*dice^{c}'
            scoring_functions[name] = make_product_fn([(mi['sqrt_cos'], b), (mi['dice'], c)])

    # wcos * entropy * dice with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'{ent_name}^{a}*wcos^{b}*dice^{c}'
                    scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['wcos'], b), (mi['dice'], c)])

    # sqrt_cos * entropy * dice with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'{ent_name}^{a}*sqrt_cos^{b}*dice^{c}'
                    scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['sqrt_cos'], b), (mi['dice'], c)])

    # cos * iw_dice with powers
    for b in powers:
        for c in powers:
            name = f'cos^{b}*iw_dice^{c}'
            scoring_functions[name] = make_product_fn([(mi['cos'], b), (mi['iw_dice'], c)])

    # entropy * cos * iw_dice with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'{ent_name}^{a}*cos^{b}*iw_dice^{c}'
                    scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['cos'], b), (mi['iw_dice'], c)])

    # entropy * cos * jaccard with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'{ent_name}^{a}*cos^{b}*jaccard^{c}'
                    scoring_functions[name] = make_product_fn([(mi[ent_name], a), (mi['cos'], b), (mi['jaccard'], c)])

    # 4-metric products: entropy * cos * dice * matched_ratio
    for ent_name in ['entropy', 'w_entropy']:
        for a in [0.5, 1.0]:
            for b in [0.5, 1.0, 1.5]:
                for c in [0.5, 1.0, 1.5]:
                    for d in [0.5, 1.0]:
                        name = f'{ent_name}^{a}*cos^{b}*dice^{c}*mr^{d}'
                        scoring_functions[name] = make_product_fn([
                            (mi[ent_name], a), (mi['cos'], b), (mi['dice'], c), (mi['matched_ratio'], d)])

    # --- Linear (additive) combinations ---
    linear_w_step = 0.1
    for ent_name in ['entropy', 'w_entropy']:
        for w1 in np.arange(0, 1.001, linear_w_step):
            for w2 in np.arange(0, 1.001 - w1, linear_w_step):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < -0.001:
                    continue
                w3 = max(0.0, w3)
                w1r, w2r = round(w1, 2), round(w2, 2)
                name = f'lin({w1r}*cos+{w2r}*dice+{w3}*{ent_name})'
                idx_c, idx_d, idx_e = mi['cos'], mi['dice'], mi[ent_name]
                scoring_functions[name] = lambda w, _w1=w1r, _w2=w2r, _w3=w3, _ic=idx_c, _id=idx_d, _ie=idx_e: \
                    _w1 * w[:, _ic] + _w2 * w[:, _id] + _w3 * w[:, _ie]

    # wcos linear
    for ent_name in ['entropy', 'w_entropy']:
        for w1 in np.arange(0, 1.001, linear_w_step):
            for w2 in np.arange(0, 1.001 - w1, linear_w_step):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < -0.001:
                    continue
                w3 = max(0.0, w3)
                w1r, w2r = round(w1, 2), round(w2, 2)
                name = f'lin({w1r}*wcos+{w2r}*dice+{w3}*{ent_name})'
                idx_c, idx_d, idx_e = mi['wcos'], mi['dice'], mi[ent_name]
                scoring_functions[name] = lambda w, _w1=w1r, _w2=w2r, _w3=w3, _ic=idx_c, _id=idx_d, _ie=idx_e: \
                    _w1 * w[:, _ic] + _w2 * w[:, _id] + _w3 * w[:, _ie]

    # --- Harmonic mean ---
    def harmonic2(a, b):
        denom = a + b
        return np.where(denom > 0, 2 * a * b / denom, 0.0)

    def harmonic3(a, b, c):
        denom = a*b + b*c + a*c
        return np.where(denom > 0, 3 * a * b * c / denom, 0.0)

    for ent_name in ['entropy', 'w_entropy']:
        scoring_functions[f'harmonic(cos,dice,{ent_name})'] = lambda w, _e=mi[ent_name]: \
            harmonic3(w[:, mi['cos']], w[:, mi['dice']], w[:, _e])
        scoring_functions[f'harmonic(wcos,dice,{ent_name})'] = lambda w, _e=mi[ent_name]: \
            harmonic3(w[:, mi['wcos']], w[:, mi['dice']], w[:, _e])

    scoring_functions['harmonic(cos,dice)'] = lambda w: harmonic2(w[:, mi['cos']], w[:, mi['dice']])
    scoring_functions['harmonic(cos,entropy)'] = lambda w: harmonic2(w[:, mi['cos']], w[:, mi['entropy']])
    scoring_functions['harmonic(wcos,dice)'] = lambda w: harmonic2(w[:, mi['wcos']], w[:, mi['dice']])

    # --- Geometric mean ---
    scoring_functions['geomean(cos,dice)'] = lambda w: np.sqrt(np.maximum(w[:, mi['cos']], 0) * np.maximum(w[:, mi['dice']], 0))
    for ent_name in ['entropy', 'w_entropy']:
        scoring_functions[f'geomean(cos,dice,{ent_name})'] = lambda w, _e=mi[ent_name]: \
            np.cbrt(np.maximum(w[:, mi['cos']], 0) * np.maximum(w[:, mi['dice']], 0) * np.maximum(w[:, _e], 0))
        scoring_functions[f'geomean(wcos,dice,{ent_name})'] = lambda w, _e=mi[ent_name]: \
            np.cbrt(np.maximum(w[:, mi['wcos']], 0) * np.maximum(w[:, mi['dice']], 0) * np.maximum(w[:, _e], 0))

    # --- Additive creative combos ---
    scoring_functions['(cos+dice)*entropy'] = lambda w: (w[:, mi['cos']] + w[:, mi['dice']]) * w[:, mi['entropy']]
    scoring_functions['(cos+dice)*w_entropy'] = lambda w: (w[:, mi['cos']] + w[:, mi['dice']]) * w[:, mi['w_entropy']]
    scoring_functions['cos*(dice+entropy)'] = lambda w: w[:, mi['cos']] * (w[:, mi['dice']] + w[:, mi['entropy']])
    scoring_functions['cos*(dice+w_entropy)'] = lambda w: w[:, mi['cos']] * (w[:, mi['dice']] + w[:, mi['w_entropy']])
    scoring_functions['cos*dice+entropy'] = lambda w: w[:, mi['cos']] * w[:, mi['dice']] + w[:, mi['entropy']]
    scoring_functions['cos*dice+w_entropy'] = lambda w: w[:, mi['cos']] * w[:, mi['dice']] + w[:, mi['w_entropy']]
    scoring_functions['cos+dice+entropy'] = lambda w: w[:, mi['cos']] + w[:, mi['dice']] + w[:, mi['entropy']]
    scoring_functions['cos+dice+w_entropy'] = lambda w: w[:, mi['cos']] + w[:, mi['dice']] + w[:, mi['w_entropy']]
    scoring_functions['wcos+dice+entropy'] = lambda w: w[:, mi['wcos']] + w[:, mi['dice']] + w[:, mi['entropy']]
    scoring_functions['wcos+dice+w_entropy'] = lambda w: w[:, mi['wcos']] + w[:, mi['dice']] + w[:, mi['w_entropy']]
    scoring_functions['cos^2*dice+entropy*dice'] = lambda w: w[:, mi['cos']]**2 * w[:, mi['dice']] + w[:, mi['entropy']] * w[:, mi['dice']]

    # min / max
    scoring_functions['max(cos,dice)'] = lambda w: np.maximum(w[:, mi['cos']], w[:, mi['dice']])
    scoring_functions['min(cos,dice)'] = lambda w: np.minimum(w[:, mi['cos']], w[:, mi['dice']])
    scoring_functions['cos*min(dice,entropy)'] = lambda w: w[:, mi['cos']] * np.minimum(w[:, mi['dice']], w[:, mi['entropy']])
    scoring_functions['cos*max(dice,entropy)'] = lambda w: w[:, mi['cos']] * np.maximum(w[:, mi['dice']], w[:, mi['entropy']])

    # exp combos
    scoring_functions['exp(cos+dice+entropy)'] = lambda w: np.exp(w[:, mi['cos']] + w[:, mi['dice']] + w[:, mi['entropy']])

    # ============================================================
    # Per-energy product scoring (product of per-energy scores, then weighted)
    # This is different from averaging metrics first then combining
    # ============================================================
    per_energy_fns = {}

    def make_per_energy_product(metric_indices_powers):
        """For each energy, compute product of metrics^powers, then weight-average across energies."""
        def fn(cand_data, ew):
            # cand_data: (n_cands, 3, n_metrics)
            n_cands = cand_data.shape[0]
            scores = np.zeros(n_cands)
            for e_idx in range(3):
                e_score = np.ones(n_cands)
                for m_idx, p in metric_indices_powers:
                    e_score *= np.power(np.maximum(cand_data[:, e_idx, m_idx], 0), p)
                scores += ew[e_idx] * e_score
            return scores
        return fn

    # Per-energy: entropy * cos * dice with powers
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'PE_{ent_name}^{a}*cos^{b}*dice^{c}'
                    per_energy_fns[name] = make_per_energy_product([
                        (mi[ent_name], a), (mi['cos'], b), (mi['dice'], c)])

    # Per-energy: cos * dice
    for b in powers:
        for c in powers:
            name = f'PE_cos^{b}*dice^{c}'
            per_energy_fns[name] = make_per_energy_product([(mi['cos'], b), (mi['dice'], c)])

    # Per-energy: wcos * dice
    for b in powers:
        for c in powers:
            name = f'PE_wcos^{b}*dice^{c}'
            per_energy_fns[name] = make_per_energy_product([(mi['wcos'], b), (mi['dice'], c)])

    # Per-energy: entropy * cos * jaccard
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'PE_{ent_name}^{a}*cos^{b}*jaccard^{c}'
                    per_energy_fns[name] = make_per_energy_product([
                        (mi[ent_name], a), (mi['cos'], b), (mi['jaccard'], c)])

    # Per-energy: entropy * wcos * dice
    for ent_name in ['entropy', 'w_entropy']:
        for a in powers:
            for b in powers:
                for c in powers:
                    name = f'PE_{ent_name}^{a}*wcos^{b}*dice^{c}'
                    per_energy_fns[name] = make_per_energy_product([
                        (mi[ent_name], a), (mi['wcos'], b), (mi['dice'], c)])

    sys.stdout.write(f"\nTotal scoring functions (weighted-avg): {len(scoring_functions)}\n")
    sys.stdout.write(f"Total per-energy scoring functions: {len(per_energy_fns)}\n")
    sys.stdout.flush()

    # ============================================================
    # PHASE 1: Coarse grid search
    # ============================================================
    sys.stdout.write(f"\n{'='*80}\n")
    sys.stdout.write(f"  PHASE 1: Coarse Grid Search (step=0.1)\n")
    sys.stdout.write(f"  {len(scoring_functions)} scoring fns x {len(coarse_grid)} weight combos = "
                     f"{len(scoring_functions) * len(coarse_grid)} evaluations\n")
    sys.stdout.write(f"  + {len(per_energy_fns)} per-energy fns x {len(coarse_grid)} = "
                     f"{len(per_energy_fns) * len(coarse_grid)} evaluations\n")
    sys.stdout.write(f"{'='*80}\n")
    sys.stdout.flush()

    results = []
    done = 0
    total_p1 = len(scoring_functions) * len(coarse_grid) + len(per_energy_fns) * len(coarse_grid)

    # Weighted-average scoring functions
    for score_name, score_fn in scoring_functions.items():
        for ew in coarse_grid:
            n_correct, n_eval = evaluate_scoring_fast(all_metrics, lib_info, score_fn, ew)
            results.append({
                'method': score_name,
                'e0_weight': ew[0],
                'e1_weight': ew[1],
                'e2_weight': ew[2],
                'n_correct': n_correct,
                'n_total': n_eval,
                'accuracy': n_correct / n_eval * 100 if n_eval > 0 else 0,
            })
            done += 1

        if done % 5000 < len(coarse_grid):
            elapsed = time.time() - t_precompute
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_p1 - done) / rate if rate > 0 else 0
            best_so_far = max(r['n_correct'] for r in results)
            sys.stdout.write(f"  {done}/{total_p1} ({done/total_p1*100:.1f}%) - "
                             f"{elapsed:.0f}s - ETA {eta:.0f}s - best: {best_so_far}/{n_eval}\n")
            sys.stdout.flush()

    # Per-energy scoring functions
    for score_name, score_fn in per_energy_fns.items():
        for ew in coarse_grid:
            n_correct, n_eval = evaluate_scoring_per_energy(all_metrics, lib_info, score_fn, ew)
            results.append({
                'method': score_name,
                'e0_weight': ew[0],
                'e1_weight': ew[1],
                'e2_weight': ew[2],
                'n_correct': n_correct,
                'n_total': n_eval,
                'accuracy': n_correct / n_eval * 100 if n_eval > 0 else 0,
            })
            done += 1

        if done % 5000 < len(coarse_grid):
            elapsed = time.time() - t_precompute
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_p1 - done) / rate if rate > 0 else 0
            best_so_far = max(r['n_correct'] for r in results)
            sys.stdout.write(f"  {done}/{total_p1} ({done/total_p1*100:.1f}%) - "
                             f"{elapsed:.0f}s - ETA {eta:.0f}s - best: {best_so_far}/{n_eval}\n")
            sys.stdout.flush()

    t_phase1 = time.time()
    sys.stdout.write(f"\nPhase 1 done in {t_phase1 - t_precompute:.1f}s\n")

    # Find top methods from phase 1
    results.sort(key=lambda x: -x['accuracy'])
    best_accuracy = results[0]['accuracy']
    best_n_correct = results[0]['n_correct']

    sys.stdout.write(f"Phase 1 best: {best_n_correct}/{results[0]['n_total']} = {best_accuracy:.1f}%\n")

    # Collect top methods (those within 2 of best correct count)
    threshold = best_n_correct - 3
    top_methods_wavg = set()
    top_methods_pe = set()
    for r in results:
        if r['n_correct'] >= threshold:
            if r['method'].startswith('PE_'):
                top_methods_pe.add(r['method'])
            else:
                top_methods_wavg.add(r['method'])

    sys.stdout.write(f"\nTop methods (within 3 of best): {len(top_methods_wavg)} weighted-avg, {len(top_methods_pe)} per-energy\n")
    sys.stdout.flush()

    # ============================================================
    # PHASE 2: Fine grid search for top methods
    # ============================================================
    sys.stdout.write(f"\n{'='*80}\n")
    sys.stdout.write(f"  PHASE 2: Fine Grid Search (step=0.05) for top methods\n")
    sys.stdout.write(f"  {len(top_methods_wavg)} wavg fns x {len(fine_grid)} + "
                     f"{len(top_methods_pe)} PE fns x {len(fine_grid)} = "
                     f"{(len(top_methods_wavg) + len(top_methods_pe)) * len(fine_grid)} evaluations\n")
    sys.stdout.write(f"{'='*80}\n")
    sys.stdout.flush()

    fine_results = []
    done = 0
    total_p2 = (len(top_methods_wavg) + len(top_methods_pe)) * len(fine_grid)

    for method_name in top_methods_wavg:
        score_fn = scoring_functions[method_name]
        for ew in fine_grid:
            n_correct, n_eval = evaluate_scoring_fast(all_metrics, lib_info, score_fn, ew)
            fine_results.append({
                'method': method_name,
                'e0_weight': ew[0],
                'e1_weight': ew[1],
                'e2_weight': ew[2],
                'n_correct': n_correct,
                'n_total': n_eval,
                'accuracy': n_correct / n_eval * 100 if n_eval > 0 else 0,
            })
            done += 1

    for method_name in top_methods_pe:
        score_fn = per_energy_fns[method_name]
        for ew in fine_grid:
            n_correct, n_eval = evaluate_scoring_per_energy(all_metrics, lib_info, score_fn, ew)
            fine_results.append({
                'method': method_name,
                'e0_weight': ew[0],
                'e1_weight': ew[1],
                'e2_weight': ew[2],
                'n_correct': n_correct,
                'n_total': n_eval,
                'accuracy': n_correct / n_eval * 100 if n_eval > 0 else 0,
            })
            done += 1

    t_phase2 = time.time()
    sys.stdout.write(f"Phase 2 done in {t_phase2 - t_phase1:.1f}s\n")
    sys.stdout.flush()

    # ============================================================
    # Merge and report
    # ============================================================
    all_results = results + fine_results
    all_results.sort(key=lambda x: (-x['accuracy'], x['method']))

    # Save all results
    out_csv = BASE / 'scoring_search_results.csv'
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    sys.stdout.write(f"\nSaved {len(all_results)} results to {out_csv}\n")

    # Print top 50 unique method+accuracy combos
    sys.stdout.write(f"\n{'='*110}\n")
    sys.stdout.write(f"  TOP 50 SCORING COMBINATIONS\n")
    sys.stdout.write(f"  (Baseline: entropy*cos*dice, e0=0.3 e1=0.3 e2=0.4 -> 61/143 = 42.7%)\n")
    sys.stdout.write(f"{'='*110}\n")
    sys.stdout.write(f"{'Rank':<5} {'Correct':>7} {'Acc%':>6} {'e0':>5} {'e1':>5} {'e2':>5}  {'Method'}\n")
    sys.stdout.write('-' * 110 + '\n')

    seen = set()
    rank = 0
    for r in all_results:
        key = (r['n_correct'], r['method'])
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        if rank > 50:
            break
        sys.stdout.write(f"{rank:<5} {r['n_correct']:>4}/{r['n_total']:<3} {r['accuracy']:>5.1f}% "
                         f"{r['e0_weight']:>5.2f} {r['e1_weight']:>5.2f} {r['e2_weight']:>5.2f}  {r['method']}\n")

    # Show all combos achieving the best accuracy
    best_acc = all_results[0]['accuracy']
    best_results = [r for r in all_results if r['accuracy'] == best_acc]

    sys.stdout.write(f"\n{'='*110}\n")
    sys.stdout.write(f"  ALL COMBINATIONS ACHIEVING BEST ACCURACY ({all_results[0]['n_correct']}/{all_results[0]['n_total']} = {best_acc:.1f}%)\n")
    sys.stdout.write(f"{'='*110}\n")

    method_counts = Counter(r['method'] for r in best_results)
    sys.stdout.write(f"\n  {len(best_results)} total combos, {len(method_counts)} unique methods\n")

    for method, count in method_counts.most_common(40):
        mresults = [r for r in best_results if r['method'] == method]
        sys.stdout.write(f"\n  {method}: {count} weight combos\n")
        for mr in mresults[:5]:
            sys.stdout.write(f"    e0={mr['e0_weight']:.2f}, e1={mr['e1_weight']:.2f}, e2={mr['e2_weight']:.2f}\n")

    # Show 2nd-best tier
    if len(all_results) > 1:
        second_best_n = all_results[0]['n_correct'] - 1
        second_results = [r for r in all_results if r['n_correct'] == second_best_n]
        if second_results:
            sys.stdout.write(f"\n{'='*110}\n")
            sys.stdout.write(f"  SECOND-BEST TIER: {second_best_n}/{all_results[0]['n_total']} = {second_best_n/all_results[0]['n_total']*100:.1f}%\n")
            sys.stdout.write(f"{'='*110}\n")
            method_counts2 = Counter(r['method'] for r in second_results)
            sys.stdout.write(f"  {len(second_results)} combos, {len(method_counts2)} unique methods\n")
            for method, count in method_counts2.most_common(20):
                sys.stdout.write(f"  {method}: {count} weight combos\n")

    total_time = time.time() - t0
    sys.stdout.write(f"\n\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
    sys.stdout.write("Done!\n")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
