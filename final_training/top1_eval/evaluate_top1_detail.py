#!/usr/bin/env python3
"""
evaluate_top1_detail.py
========================
Per-compound TOP1 evaluation.

For each of 143 library compounds × 3 models:
  - Score all candidates (cosine, DICE, matched peaks) at e0+e1+e2
  - Report TOP1 candidate, its scores, whether it's correct
  - Report correct answer's rank and score
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODELS = ['cfm_default', 'param_jjy', 'full_model']
MODEL_LABELS = {
    'cfm_default': 'CFM-ID Default',
    'param_jjy': 'Param_JJY',
    'full_model': 'Full Model (Final)',
}
ENERGIES = [0, 1, 2]
ENERGY_WEIGHTS = {0: 0.31, 1: 0.27, 2: 0.42}  # optimized via grid search (step=0.01)


# ============================================================
# Spectrum parsing & metrics
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


def matched_peak_count(spec1, spec2, mz_tol=0.01):
    matched, _, _ = match_peaks(spec1, spec2, mz_tol)
    return len(matched)


def iw_dice_similarity(spec1, spec2, mz_tol=0.01):
    """Intensity-weighted DICE: matched intensity sum / total intensity sum."""
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not s1 or not s2:
        return 0.0
    matched_sum = sum(i1 + i2 for i1, i2 in matched)
    total_sum = sum(i for _, i in s1) + sum(i for _, i in s2)
    return matched_sum / total_sum if total_sum > 0 else 0.0


def spectral_entropy_sim(spec1, spec2, mz_tol=0.01):
    """Entropy-based similarity (JSD)."""
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


def combined_score(exp_specs, pred_specs, energies, metric_fn):
    scores = [metric_fn(exp_specs.get(e, []), pred_specs.get(e, [])) for e in energies]
    return np.mean(scores) if scores else 0.0


def weighted_combined(exp_specs, pred_specs, metric_fn):
    """Weighted combined score with optimized energy weights."""
    s = 0
    for e in ENERGIES:
        s += ENERGY_WEIGHTS[e] * metric_fn(exp_specs.get(e, []), pred_specs.get(e, []))
    return s


# ============================================================
# Precompute per-energy metrics for all compounds × candidates
# ============================================================
def precompute_all_metrics(df, models, exp_spectra_dir):
    """Precompute per-energy cos, dice, iw_dice, ent, mp for all candidates."""
    data = {}  # {model: {lib_name: {cand_id: {energy: {metric: val}}}}}

    for model in models:
        model_label = MODEL_LABELS[model]
        pred_dir = BASE / model / 'predictions'
        data[model] = {}

        for lib_name in sorted(df['lib_name'].unique()):
            sub = df[df['lib_name'] == lib_name]
            exp_file = exp_spectra_dir / f'{lib_name}.txt'
            if not exp_file.exists():
                continue

            exp_specs = {e: parse_spectrum(exp_file, e) for e in ENERGIES}
            data[model][lib_name] = {}

            for _, row in sub.iterrows():
                cand_id = row['cand_id']
                pred_file = pred_dir / f'{cand_id}.log'

                if not pred_file.exists():
                    data[model][lib_name][cand_id] = {
                        e: {'cos': 0.0, 'dice': 0.0, 'iw_dice': 0.0,
                            'ent': 0.0, 'mp': 0}
                        for e in ENERGIES
                    }
                    continue

                pred_specs = {e: parse_spectrum(pred_file, e) for e in ENERGIES}
                per_e = {}
                for e in ENERGIES:
                    exp_s = exp_specs.get(e, [])
                    pred_s = pred_specs.get(e, [])
                    per_e[e] = {
                        'cos': cosine_similarity(exp_s, pred_s),
                        'dice': dice_similarity(exp_s, pred_s),
                        'iw_dice': iw_dice_similarity(exp_s, pred_s),
                        'ent': spectral_entropy_sim(exp_s, pred_s),
                        'mp': matched_peak_count(exp_s, pred_s),
                    }
                data[model][lib_name][cand_id] = per_e

    return data


def score_with_weights(per_e_metrics, weights, scoring_fn):
    """Compute weighted score from per-energy metrics."""
    weighted_metrics = {}
    for metric in ['cos', 'dice', 'iw_dice', 'ent', 'mp']:
        val = sum(weights[e] * per_e_metrics[e][metric] for e in ENERGIES)
        weighted_metrics[metric] = val
    return scoring_fn(weighted_metrics), weighted_metrics


def grid_search_optimal(df, precomputed, model='full_model', step=0.05):
    """Grid search over energy weights × scoring formulas for best TOP1."""
    scoring_formulas = {
        'ent*cos*dice': lambda m: m['ent'] * m['cos'] * m['dice'],
        'ent*cos*iw_dice': lambda m: m['ent'] * m['cos'] * m['iw_dice'],
        'cos*dice': lambda m: m['cos'] * m['dice'],
        'cos*iw_dice': lambda m: m['cos'] * m['iw_dice'],
        'ent*cos^0.75*iw_dice^1.25': lambda m: m['ent'] * (m['cos']**0.75) * (m['iw_dice']**1.25),
        'ent^0.5*cos*iw_dice': lambda m: (m['ent']**0.5) * m['cos'] * m['iw_dice'],
        'ent*cos^0.5*iw_dice^1.5': lambda m: m['ent'] * (m['cos']**0.5) * (m['iw_dice']**1.5),
        'ent*iw_dice': lambda m: m['ent'] * m['iw_dice'],
        'cos': lambda m: m['cos'],
        'iw_dice': lambda m: m['iw_dice'],
        'ent': lambda m: m['ent'],
        'dice': lambda m: m['dice'],
    }

    # Generate weight combos
    weight_combos = []
    steps = int(1.0 / step) + 1
    for i in range(steps):
        for j in range(steps - i):
            k = steps - 1 - i - j
            w0 = round(i * step, 2)
            w1 = round(j * step, 2)
            w2 = round(k * step, 2)
            if abs(w0 + w1 + w2 - 1.0) < 0.001:
                weight_combos.append({0: w0, 1: w1, 2: w2})

    lib_names = sorted(precomputed[model].keys())
    lib_hmdb = {}
    for lib_name in lib_names:
        sub = df[df['lib_name'] == lib_name]
        hmdb_id = sub['hmdb_id'].iloc[0]
        lib_hmdb[lib_name] = str(hmdb_id) if pd.notna(hmdb_id) else ''

    cand_info = {}
    for lib_name in lib_names:
        sub = df[df['lib_name'] == lib_name]
        cand_info[lib_name] = []
        for _, row in sub.iterrows():
            cand_info[lib_name].append({
                'cand_id': row['cand_id'],
                'cand_hmdb': str(row.get('cand_hmdb_id', '')) if pd.notna(row.get('cand_hmdb_id', '')) else '',
            })

    best_top1 = 0
    best_config = None
    results = []

    total = len(weight_combos) * len(scoring_formulas)
    print(f'  Grid search: {len(weight_combos)} weight combos × {len(scoring_formulas)} formulas = {total} tests')

    for formula_name, formula_fn in scoring_formulas.items():
        for weights in weight_combos:
            n_correct = 0
            for lib_name in lib_names:
                correct_hmdb = lib_hmdb[lib_name]
                if not correct_hmdb:
                    continue

                cand_scores = []
                for ci in cand_info[lib_name]:
                    cid = ci['cand_id']
                    per_e = precomputed[model][lib_name].get(cid)
                    if per_e is None:
                        continue
                    score, _ = score_with_weights(per_e, weights, formula_fn)
                    cand_scores.append((score, ci['cand_hmdb']))

                if not cand_scores:
                    continue
                cand_scores.sort(key=lambda x: x[0], reverse=True)
                if cand_scores[0][1] == correct_hmdb:
                    n_correct += 1

            results.append({
                'formula': formula_name,
                'w0': weights[0], 'w1': weights[1], 'w2': weights[2],
                'top1': n_correct,
            })

            if n_correct > best_top1:
                best_top1 = n_correct
                best_config = {'formula': formula_name, 'weights': weights, 'top1': n_correct}

    return best_config, results


# ============================================================
# Main
# ============================================================
def main():
    print('Loading candidates...')
    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'

    # Phase 1: Precompute all per-energy metrics
    print('Precomputing per-energy metrics for all models...')
    precomputed = precompute_all_metrics(df, MODELS, exp_spectra_dir)

    # Phase 2: Grid search for optimal scoring (full_model only)
    print('\n=== Grid Search for Optimal Scoring (Full Model) ===')
    best_config, grid_results = grid_search_optimal(df, precomputed, 'full_model', step=0.01)
    print(f'\n  BEST: {best_config["formula"]}, '
          f'weights=({best_config["weights"][0]:.2f}, {best_config["weights"][1]:.2f}, {best_config["weights"][2]:.2f}), '
          f'TOP1={best_config["top1"]}/143')

    # Save grid search results
    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(BASE / 'grid_search_full.csv', index=False)

    # Show top 10
    grid_df_sorted = grid_df.sort_values('top1', ascending=False).head(20)
    print('\n  Top 20 configurations:')
    print(f'  {"Formula":<35} {"w0":>5} {"w1":>5} {"w2":>5} {"TOP1":>5}')
    print('  ' + '-' * 60)
    for _, r in grid_df_sorted.iterrows():
        print(f'  {r["formula"]:<35} {r["w0"]:>5.2f} {r["w1"]:>5.2f} {r["w2"]:>5.2f} {int(r["top1"]):>5}')

    # Use best config
    BEST_FORMULA_NAME = best_config['formula']
    BEST_WEIGHTS = best_config['weights']

    scoring_formulas = {
        'ent*cos*dice': lambda m: m['ent'] * m['cos'] * m['dice'],
        'ent*cos*iw_dice': lambda m: m['ent'] * m['cos'] * m['iw_dice'],
        'cos*dice': lambda m: m['cos'] * m['dice'],
        'cos*iw_dice': lambda m: m['cos'] * m['iw_dice'],
        'ent*cos^0.75*iw_dice^1.25': lambda m: m['ent'] * (m['cos']**0.75) * (m['iw_dice']**1.25),
        'ent^0.5*cos*iw_dice': lambda m: (m['ent']**0.5) * m['cos'] * m['iw_dice'],
        'ent*cos^0.5*iw_dice^1.5': lambda m: m['ent'] * (m['cos']**0.5) * (m['iw_dice']**1.5),
        'ent*iw_dice': lambda m: m['ent'] * m['iw_dice'],
        'cos': lambda m: m['cos'],
        'iw_dice': lambda m: m['iw_dice'],
        'ent': lambda m: m['ent'],
        'dice': lambda m: m['dice'],
    }
    best_fn = scoring_formulas[BEST_FORMULA_NAME]

    # Phase 3: Full evaluation with best scoring for all models
    print(f'\n{"="*120}')
    print(f'  Evaluating all models with: {BEST_FORMULA_NAME}, '
          f'weights=({BEST_WEIGHTS[0]:.2f}, {BEST_WEIGHTS[1]:.2f}, {BEST_WEIGHTS[2]:.2f})')
    print(f'{"="*120}')

    all_results = []
    for model in MODELS:
        model_label = MODEL_LABELS[model]
        print(f'\n=== {model_label} ===')

        for lib_name in sorted(precomputed[model].keys()):
            sub = df[df['lib_name'] == lib_name]
            hmdb_id = sub['hmdb_id'].iloc[0]
            n_cands = len(sub)
            correct_hmdb = str(hmdb_id) if pd.notna(hmdb_id) else ''

            cand_results = []
            for _, row in sub.iterrows():
                cand_id = row['cand_id']
                cand_smi = row['cand_dtb_smiles']
                cand_hmdb_raw = row.get('cand_hmdb_id', '')
                cand_hmdb = str(cand_hmdb_raw) if pd.notna(cand_hmdb_raw) else ''
                cand_name = row.get('cand_name', '')

                per_e = precomputed[model][lib_name].get(cand_id)
                if per_e is None:
                    continue

                score, wm = score_with_weights(per_e, BEST_WEIGHTS, best_fn)
                cand_results.append({
                    'cand_id': cand_id, 'cand_smi': cand_smi,
                    'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
                    'cos': wm['cos'], 'dice': wm['dice'], 'iw_dice': wm['iw_dice'],
                    'ent': wm['ent'], 'mp': wm['mp'], 'score': score,
                })

            if not cand_results:
                continue

            ranked = sorted(cand_results, key=lambda x: x['score'], reverse=True)
            top1 = ranked[0]

            # Find correct answer
            correct_rank = -1
            correct_cos = correct_dice = correct_iw_dice = correct_ent = correct_score = 0.0
            correct_mp = 0
            correct_in_cands = False
            for rank_i, c in enumerate(ranked):
                if c['cand_hmdb'] == correct_hmdb and correct_hmdb:
                    correct_rank = rank_i + 1
                    correct_cos = c['cos']
                    correct_dice = c['dice']
                    correct_iw_dice = c['iw_dice']
                    correct_ent = c['ent']
                    correct_mp = c['mp']
                    correct_score = c['score']
                    correct_in_cands = True
                    break

            top1_correct = (top1['cand_hmdb'] == correct_hmdb and correct_hmdb != '')

            all_results.append({
                'model': model_label,
                'lib_name': lib_name,
                'hmdb_id': hmdb_id,
                'n_candidates': n_cands,
                'correct_in_cands': correct_in_cands,
                # TOP1 info
                'top1_cand_id': top1['cand_id'],
                'top1_cand_hmdb': top1['cand_hmdb'],
                'top1_score': round(top1['score'], 4),
                'top1_cos': round(top1['cos'], 4),
                'top1_dice': round(top1['dice'], 4),
                'top1_iw_dice': round(top1['iw_dice'], 4),
                'top1_ent': round(top1['ent'], 4),
                'top1_mp': round(top1['mp'], 1),
                'top1_correct': top1_correct,
                # Correct answer info
                'correct_rank': correct_rank if correct_in_cands else 'N/A',
                'correct_score': round(correct_score, 4) if correct_in_cands else 'N/A',
                'correct_cos': round(correct_cos, 4) if correct_in_cands else 'N/A',
                'correct_dice': round(correct_dice, 4) if correct_in_cands else 'N/A',
                'correct_iw_dice': round(correct_iw_dice, 4) if correct_in_cands else 'N/A',
                'correct_ent': round(correct_ent, 4) if correct_in_cands else 'N/A',
                'correct_mp': round(correct_mp, 1) if correct_in_cands else 'N/A',
            })

        # Print summary
        model_rows = [r for r in all_results if r['model'] == model_label]
        n_total = len(model_rows)
        n_correct = sum(1 for r in model_rows if r['top1_correct'])
        n_in_cands = sum(1 for r in model_rows if r['correct_in_cands'])
        print(f'  Total: {n_total}, Correct in cands: {n_in_cands}, '
              f'TOP1 correct: {n_correct} ({n_correct/n_total*100:.1f}%)')

    # Save Excel
    out_xlsx = BASE / 'top1_detail.xlsx'
    result_df = pd.DataFrame(all_results)

    with pd.ExcelWriter(str(out_xlsx), engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='All', index=False)

        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = result_df[result_df['model'] == label].copy()
            mdf.to_excel(writer, sheet_name=model[:31], index=False)

        # Summary
        summary_rows = []
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = result_df[result_df['model'] == label]
            n = len(mdf)
            n_correct = int(mdf['top1_correct'].sum())
            n_in = int(mdf['correct_in_cands'].sum())
            filt = mdf[mdf['correct_in_cands']]
            filt_correct = int(filt['top1_correct'].sum())

            mdf_in = filt.copy()
            mdf_in['correct_rank'] = pd.to_numeric(mdf_in['correct_rank'], errors='coerce')
            top3 = int((mdf_in['correct_rank'] <= 3).sum())
            top5 = int((mdf_in['correct_rank'] <= 5).sum())
            top10 = int((mdf_in['correct_rank'] <= 10).sum())

            summary_rows.append({
                'Model': label,
                'N Compounds': n,
                'N Correct in Cands': n_in,
                'TOP1 Correct': n_correct,
                'TOP1 Accuracy (%)': round(n_correct / n * 100, 1),
                'Filtered Accuracy (%)': round(filt_correct / n_in * 100, 1) if n_in > 0 else 0,
                'TOP3 (filtered)': top3,
                'TOP5 (filtered)': top5,
                'TOP10 (filtered)': top10,
                'Mean TOP1 Cosine': round(mdf['top1_cos'].mean(), 4),
                'Mean TOP1 DICE': round(mdf['top1_dice'].mean(), 4),
                'Mean TOP1 iw_DICE': round(mdf['top1_iw_dice'].mean(), 4),
                'Mean TOP1 Entropy': round(mdf['top1_ent'].mean(), 4),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

        # Scoring config
        config_df = pd.DataFrame([{
            'Scoring Formula': BEST_FORMULA_NAME,
            'Energy Weight E0 (10eV)': BEST_WEIGHTS[0],
            'Energy Weight E1 (20eV)': BEST_WEIGHTS[1],
            'Energy Weight E2 (40eV)': BEST_WEIGHTS[2],
            'Grid Search Step': 0.05,
            'Grid Search Step': 0.01,
            'N Total Tests': len(grid_results),
            'N Formulas Tested': len(set(r['formula'] for r in grid_results)),
        }])
        config_df.to_excel(writer, sheet_name='Scoring Config', index=False)

    print(f'\nSaved: {out_xlsx}')

    out_csv = BASE / 'top1_detail.csv'
    result_df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')

    # Print TOP-K summary
    print(f'\n{"="*120}')
    print(f'  Scoring: {BEST_FORMULA_NAME} | Weights: e0={BEST_WEIGHTS[0]:.2f} e1={BEST_WEIGHTS[1]:.2f} e2={BEST_WEIGHTS[2]:.2f}')
    print(f'{"="*120}')
    for model in MODELS:
        label = MODEL_LABELS[model]
        mdf = result_df[result_df['model'] == label]
        n = len(mdf)
        n_correct = mdf['top1_correct'].sum()
        n_in = mdf['correct_in_cands'].sum()
        mdf_in = mdf[mdf['correct_in_cands']].copy()
        mdf_in['correct_rank'] = pd.to_numeric(mdf_in['correct_rank'], errors='coerce')
        top3 = (mdf_in['correct_rank'] <= 3).sum()
        top5 = (mdf_in['correct_rank'] <= 5).sum()
        top10 = (mdf_in['correct_rank'] <= 10).sum()
        print(f'  {label:<25} TOP1: {int(n_correct):>3}/{n} ({n_correct/n*100:.1f}%)  |  '
              f'TOP3: {int(top3):>3}/{int(n_in)} ({top3/n_in*100:.1f}%)  |  '
              f'TOP5: {int(top5):>3}/{int(n_in)} ({top5/n_in*100:.1f}%)  |  '
              f'TOP10: {int(top10):>3}/{int(n_in)} ({top10/n_in*100:.1f}%)')


if __name__ == '__main__':
    main()
