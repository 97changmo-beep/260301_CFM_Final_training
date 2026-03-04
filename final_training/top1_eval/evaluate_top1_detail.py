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
ENERGY_WEIGHTS = {0: 0.3, 1: 0.3, 2: 0.4}  # optimized weights


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
# Main
# ============================================================
def main():
    print('Loading candidates...')
    df = pd.read_pickle(BASE / 'candidates.pkl')
    exp_spectra_dir = BASE.parent / 'full_model' / 'spectra'

    all_results = []

    for model in MODELS:
        model_label = MODEL_LABELS[model]
        pred_dir = BASE / model / 'predictions'
        print(f'\n=== {model_label} ===')

        for lib_name in sorted(df['lib_name'].unique()):
            sub = df[df['lib_name'] == lib_name]
            correct_smi = sub['lib_dtb_smiles'].iloc[0]
            hmdb_id = sub['hmdb_id'].iloc[0]
            n_cands = len(sub)

            # Experimental spectra
            exp_file = exp_spectra_dir / f'{lib_name}.txt'
            if not exp_file.exists():
                continue

            exp_specs = {e: parse_spectrum(exp_file, e) for e in ENERGIES}

            # Score all candidates
            cand_results = []
            for _, row in sub.iterrows():
                cand_id = row['cand_id']
                cand_smi = row['cand_dtb_smiles']
                cand_hmdb = row.get('cand_hmdb_id', '')
                cand_name = row.get('cand_name', '')
                pred_file = pred_dir / f'{cand_id}.log'

                if not pred_file.exists():
                    cand_results.append({
                        'cand_id': cand_id, 'cand_smi': cand_smi,
                        'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
                        'cos': 0.0, 'dice': 0.0, 'mp': 0, 'ent': 0.0, 'score': 0.0,
                    })
                    continue

                pred_specs = {e: parse_spectrum(pred_file, e) for e in ENERGIES}
                cos = weighted_combined(exp_specs, pred_specs, cosine_similarity)
                dice = weighted_combined(exp_specs, pred_specs, dice_similarity)
                ent = weighted_combined(exp_specs, pred_specs, spectral_entropy_sim)
                mp = combined_score(exp_specs, pred_specs, ENERGIES,
                                    lambda a, b: matched_peak_count(a, b))
                score = ent * cos * dice  # optimized scoring

                cand_results.append({
                    'cand_id': cand_id, 'cand_smi': cand_smi,
                    'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
                    'cos': cos, 'dice': dice, 'mp': mp, 'ent': ent, 'score': score,
                })

            if not cand_results:
                continue

            # Rank by entropy*cos*dice score (primary)
            ranked = sorted(cand_results, key=lambda x: x['score'], reverse=True)

            # TOP1
            top1 = ranked[0]

            # Check correctness: HMDB ID matching
            correct_hmdb = str(hmdb_id) if pd.notna(hmdb_id) else ''

            # Find correct answer's rank
            correct_rank = -1
            correct_cos = 0.0
            correct_dice = 0.0
            correct_mp = 0
            correct_score = 0.0
            correct_in_cands = False
            for rank_i, c in enumerate(ranked):
                c_hmdb = str(c['cand_hmdb']) if pd.notna(c['cand_hmdb']) else ''
                if c_hmdb == correct_hmdb and correct_hmdb:
                    correct_rank = rank_i + 1
                    correct_cos = c['cos']
                    correct_dice = c['dice']
                    correct_mp = c['mp']
                    correct_score = c['score']
                    correct_in_cands = True
                    break

            # Check if TOP1 is correct (by HMDB ID)
            top1_hmdb = str(top1['cand_hmdb']) if pd.notna(top1['cand_hmdb']) else ''
            top1_correct = (top1_hmdb == correct_hmdb and correct_hmdb != '')

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
                'top1_ent': round(top1['ent'], 4),
                'top1_mp': round(top1['mp'], 1),
                'top1_correct': top1_correct,
                # Correct answer info
                'correct_rank': correct_rank if correct_in_cands else 'N/A',
                'correct_cos': round(correct_cos, 4) if correct_in_cands else 'N/A',
                'correct_dice': round(correct_dice, 4) if correct_in_cands else 'N/A',
                'correct_mp': round(correct_mp, 1) if correct_in_cands else 'N/A',
            })

        # Print summary for this model
        model_rows = [r for r in all_results if r['model'] == model_label]
        n_total = len(model_rows)
        n_correct = sum(1 for r in model_rows if r['top1_correct'])
        n_in_cands = sum(1 for r in model_rows if r['correct_in_cands'])
        print(f'  Total: {n_total}, Correct in cands: {n_in_cands}, '
              f'TOP1 correct: {n_correct} ({n_correct/n_total*100:.1f}%)')

    # Save to Excel with separate sheets per model
    out_xlsx = BASE / 'top1_detail.xlsx'
    result_df = pd.DataFrame(all_results)

    with pd.ExcelWriter(str(out_xlsx), engine='openpyxl') as writer:
        # All results
        result_df.to_excel(writer, sheet_name='All', index=False)

        # Per-model sheets
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = result_df[result_df['model'] == label].copy()
            sheet_name = model[:31]  # Excel sheet name limit
            mdf.to_excel(writer, sheet_name=sheet_name, index=False)

        # Summary sheet
        summary_rows = []
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = result_df[result_df['model'] == label]
            n = len(mdf)
            n_correct = mdf['top1_correct'].sum()
            n_in = mdf['correct_in_cands'].sum()
            cos_mean = mdf['top1_cos'].mean()
            dice_mean = mdf['top1_dice'].mean()
            mp_mean = mdf['top1_mp'].mean()

            # For those correct in candidates
            filt = mdf[mdf['correct_in_cands']]
            filt_correct = filt['top1_correct'].sum()

            summary_rows.append({
                'Model': label,
                'N Compounds': n,
                'N Correct in Cands': int(n_in),
                'TOP1 Correct': int(n_correct),
                'TOP1 Accuracy (%)': round(n_correct / n * 100, 1),
                'Filtered Accuracy (%)': round(filt_correct / len(filt) * 100, 1) if len(filt) > 0 else 0,
                'Mean TOP1 Cosine': round(cos_mean, 4),
                'Mean TOP1 DICE': round(dice_mean, 4),
                'Mean TOP1 Matched Peaks': round(mp_mean, 1),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    print(f'\nSaved: {out_xlsx}')

    # Also save CSV
    out_csv = BASE / 'top1_detail.csv'
    result_df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')

    # Print detailed view
    print(f'\n{"="*110}')
    print(f'  Per-Compound TOP1 Results (Scoring: entropy*cos*dice, Energy weights: e0=0.3 e1=0.3 e2=0.4)')
    print(f'{"="*110}')
    for model in MODELS:
        label = MODEL_LABELS[model]
        mdf = result_df[result_df['model'] == label]
        print(f'\n--- {label} ---')
        print(f'{"Compound":<35} {"#Cand":>5} {"Score":>8} {"Cos":>7} {"DICE":>7} {"Ent":>7} '
              f'{"OK?":>4} {"Rank":>5} {"AnsScore":>9}')
        print('-' * 110)
        for _, r in mdf.iterrows():
            mark = 'O' if r['top1_correct'] else 'X'
            crank = str(r['correct_rank']) if r['correct_in_cands'] else '-'
            ccos = f'{r["correct_cos"]:.4f}' if r['correct_in_cands'] and r['correct_cos'] != 'N/A' else '-'
            print(f'{r["lib_name"]:<35} {r["n_candidates"]:>5} '
                  f'{r["top1_score"]:>8.4f} {r["top1_cos"]:>7.4f} {r["top1_dice"]:>7.4f} {r["top1_ent"]:>7.4f} '
                  f'{mark:>4} {crank:>5} {ccos:>9}')

        n_correct = mdf['top1_correct'].sum()
        n = len(mdf)
        n_in = mdf['correct_in_cands'].sum()
        # Top-K accuracy
        mdf_in = mdf[mdf['correct_in_cands']].copy()
        mdf_in['correct_rank'] = pd.to_numeric(mdf_in['correct_rank'], errors='coerce')
        top3 = (mdf_in['correct_rank'] <= 3).sum()
        top5 = (mdf_in['correct_rank'] <= 5).sum()
        top10 = (mdf_in['correct_rank'] <= 10).sum()
        print(f'\nTOP1: {n_correct}/{n} ({n_correct/n*100:.1f}%)  |  '
              f'TOP3: {top3}/{int(n_in)} ({top3/n_in*100:.1f}%)  |  '
              f'TOP5: {top5}/{int(n_in)} ({top5/n_in*100:.1f}%)  |  '
              f'TOP10: {top10}/{int(n_in)} ({top10/n_in*100:.1f}%)')


if __name__ == '__main__':
    main()
