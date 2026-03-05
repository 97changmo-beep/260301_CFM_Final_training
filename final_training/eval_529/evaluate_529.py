#!/usr/bin/env python3
"""
evaluate_529.py
===============
Score 529 feature candidates using 3 models.

Experimental spectra: single CE (~28 eV, HCD) from MGF file.
Predicted spectra: 3 energy levels (energy0=10eV, energy1=20eV, energy2=40eV).

For each feature × model:
  - Compare experimental spectrum vs predicted at each energy level
  - Score using optimized formula: ent * cos^0.75 * iw_dice^1.25
  - Rank candidates and report TOP1
"""
import math
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
MODELS = ['cfm_default', 'param_jjy', 'full_model']
MODEL_LABELS = {
    'cfm_default': 'CFM-ID Default',
    'param_jjy': 'Param_JJY',
    'full_model': 'Full Model (Final)',
}
ENERGIES = [0, 1, 2]
MGF_PATH = BASE.parent.parent / '529_invitroDESTNI_features.mgf'


# ============================================================
# Spectrum parsing
# ============================================================
def parse_mgf(mgf_path):
    """Parse MGF file, return {feature_id: [(mz, intensity), ...]}."""
    spectra = {}
    current_id = None
    peaks = []
    in_peaks = False

    with open(mgf_path) as f:
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                current_id = None
                peaks = []
                in_peaks = False
            elif line.startswith('FEATURE_ID='):
                current_id = line.split('=')[1]
            elif line.startswith('Num peaks='):
                in_peaks = True
            elif line == 'END IONS':
                if current_id and peaks:
                    spectra[current_id] = peaks
                in_peaks = False
            elif in_peaks and line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        spectra.setdefault(current_id, [])
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return spectra


def parse_predicted_spectrum(path, energy=0):
    """Parse CFM-ID prediction .log file for a specific energy level."""
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


# ============================================================
# Metrics (same as evaluate_top1_detail.py)
# ============================================================
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


def iw_dice_similarity(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not s1 or not s2:
        return 0.0
    matched_sum = sum(i1 + i2 for i1, i2 in matched)
    total_sum = sum(i for _, i in s1) + sum(i for _, i in s2)
    return matched_sum / total_sum if total_sum > 0 else 0.0


def matched_peak_count(spec1, spec2, mz_tol=0.01):
    matched, _, _ = match_peaks(spec1, spec2, mz_tol)
    return len(matched)


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


def scoring_fn(metrics):
    """Optimized scoring: ent * cos^0.75 * iw_dice^1.25"""
    return metrics['ent'] * (metrics['cos'] ** 0.75) * (metrics['iw_dice'] ** 1.25)


# ============================================================
# Main
# ============================================================
def main():
    print('Loading data...')
    df = pd.read_pickle(BASE / 'candidates_529.pkl')
    smiles_map = pd.read_csv(BASE / 'smiles_mapping.csv')
    smi_to_cand = dict(zip(smiles_map['dtb_smiles'], smiles_map['cand_id']))

    # Add cand_id to df
    df['cand_id'] = df['dtb_smiles'].map(smi_to_cand)

    print(f'Features: {df["feature_id"].nunique()}, Candidates: {len(df)}, '
          f'Unique SMILES: {df["dtb_smiles"].nunique()}')

    # Parse experimental spectra
    print('Parsing MGF...')
    exp_spectra = parse_mgf(MGF_PATH)
    print(f'Experimental spectra: {len(exp_spectra)}')

    # Optimized energy weights from 143-compound evaluation
    OPT_WEIGHTS = {0: 0.40, 1: 0.34, 2: 0.26}

    all_results = []

    for model in MODELS:
        model_label = MODEL_LABELS[model]
        pred_dir = BASE / model / 'predictions'
        print(f'\n=== {model_label} ===')

        n_features = 0
        n_with_scores = 0

        for feat_id in sorted(df['feature_id'].unique()):
            sub = df[df['feature_id'] == feat_id]
            n_cands = len(sub)

            # Get experimental spectrum
            exp_spec = exp_spectra.get(str(feat_id))
            if not exp_spec:
                continue

            n_features += 1

            # Score all candidates
            cand_results = []
            for _, row in sub.iterrows():
                cand_id = row['cand_id']
                cand_smi = row['dtb_smiles']
                cand_hmdb = row.get('hmdb_id', '')
                cand_name = row.get('candidate_name', '')
                pred_file = pred_dir / f'{cand_id}.log'

                if not pred_file.exists() or pd.isna(cand_id):
                    cand_results.append({
                        'cand_id': cand_id, 'cand_smi': cand_smi,
                        'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
                        'score': 0.0, 'cos': 0.0, 'dice': 0.0,
                        'iw_dice': 0.0, 'ent': 0.0, 'mp': 0,
                        'best_energy': -1,
                        'cos_e0': 0.0, 'cos_e1': 0.0, 'cos_e2': 0.0,
                    })
                    continue

                # Compute metrics at each energy level
                per_e = {}
                for e in ENERGIES:
                    pred_spec = parse_predicted_spectrum(pred_file, e)
                    per_e[e] = {
                        'cos': cosine_similarity(exp_spec, pred_spec),
                        'dice': dice_similarity(exp_spec, pred_spec),
                        'iw_dice': iw_dice_similarity(exp_spec, pred_spec),
                        'ent': spectral_entropy_sim(exp_spec, pred_spec),
                        'mp': matched_peak_count(exp_spec, pred_spec),
                    }

                # Weighted combination (same as 143-compound optimization)
                weighted_metrics = {}
                for metric in ['cos', 'dice', 'iw_dice', 'ent', 'mp']:
                    weighted_metrics[metric] = sum(
                        OPT_WEIGHTS[e] * per_e[e][metric] for e in ENERGIES
                    )

                score = scoring_fn(weighted_metrics)

                # Also find best single energy
                e_scores = {e: scoring_fn(per_e[e]) for e in ENERGIES}
                best_e = max(e_scores, key=e_scores.get)

                cand_results.append({
                    'cand_id': cand_id, 'cand_smi': cand_smi,
                    'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
                    'score': score,
                    'cos': weighted_metrics['cos'],
                    'dice': weighted_metrics['dice'],
                    'iw_dice': weighted_metrics['iw_dice'],
                    'ent': weighted_metrics['ent'],
                    'mp': weighted_metrics['mp'],
                    'best_energy': best_e,
                    'cos_e0': per_e[0]['cos'],
                    'cos_e1': per_e[1]['cos'],
                    'cos_e2': per_e[2]['cos'],
                })

            if not cand_results:
                continue

            # Rank by score
            ranked = sorted(cand_results, key=lambda x: x['score'], reverse=True)
            top1 = ranked[0]

            # Get feature info
            feat_mz = sub['feature_mz'].iloc[0]
            feat_rt = sub['feature_rt_min'].iloc[0]
            feat_name = sub['feature_name'].iloc[0] if 'feature_name' in sub.columns else ''

            if top1['score'] > 0:
                n_with_scores += 1

            # Store top-K results
            for rank_i, c in enumerate(ranked[:10]):
                all_results.append({
                    'model': model_label,
                    'feature_id': feat_id,
                    'feature_mz': round(feat_mz, 5),
                    'feature_rt_min': round(feat_rt, 2) if pd.notna(feat_rt) else '',
                    'n_candidates': n_cands,
                    'rank': rank_i + 1,
                    'cand_id': c['cand_id'],
                    'cand_smiles': c['cand_smi'],
                    'cand_hmdb': c['cand_hmdb'] if pd.notna(c['cand_hmdb']) else '',
                    'cand_name': c['cand_name'] if pd.notna(c['cand_name']) else '',
                    'score': round(c['score'], 6),
                    'cosine': round(c['cos'], 4),
                    'dice': round(c['dice'], 4),
                    'iw_dice': round(c['iw_dice'], 4),
                    'entropy': round(c['ent'], 4),
                    'matched_peaks': round(c['mp'], 1),
                    'cos_e0': round(c['cos_e0'], 4),
                    'cos_e1': round(c['cos_e1'], 4),
                    'cos_e2': round(c['cos_e2'], 4),
                })

        print(f'  Features: {n_features}, With scores: {n_with_scores}')

    # Save results
    result_df = pd.DataFrame(all_results)
    top1_df = result_df[result_df['rank'] == 1].copy()

    out_xlsx = BASE / 'eval_529_results.xlsx'
    with pd.ExcelWriter(str(out_xlsx), engine='openpyxl') as writer:
        # TOP1 per model
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = top1_df[top1_df['model'] == label].copy()
            sheet = model[:31]
            mdf.drop(columns=['model'], inplace=True)
            mdf.to_excel(writer, sheet_name=sheet, index=False)

        # Model comparison (TOP1 side by side)
        comparison = []
        for feat_id in sorted(top1_df['feature_id'].unique()):
            row = {'feature_id': feat_id}
            feat_sub = top1_df[top1_df['feature_id'] == feat_id]
            row['feature_mz'] = feat_sub['feature_mz'].iloc[0]
            row['feature_rt'] = feat_sub['feature_rt_min'].iloc[0]
            row['n_candidates'] = feat_sub['n_candidates'].iloc[0]

            for model in MODELS:
                label = MODEL_LABELS[model]
                msub = feat_sub[feat_sub['model'] == label]
                if len(msub) > 0:
                    r = msub.iloc[0]
                    short = model.replace('cfm_', '').replace('param_', '')
                    row[f'{short}_name'] = r['cand_name']
                    row[f'{short}_hmdb'] = r['cand_hmdb']
                    row[f'{short}_score'] = r['score']
                    row[f'{short}_cos'] = r['cosine']

            # Check consensus
            names = []
            for model in MODELS:
                label = MODEL_LABELS[model]
                msub = feat_sub[feat_sub['model'] == label]
                if len(msub) > 0:
                    names.append(msub.iloc[0]['cand_smiles'])
            row['consensus'] = 'YES' if len(set(names)) == 1 and len(names) == 3 else \
                               'PARTIAL' if len(set(names)) == 2 else 'NO'

            comparison.append(row)

        comp_df = pd.DataFrame(comparison)
        comp_df.to_excel(writer, sheet_name='Model Comparison', index=False)

        # All top-10 results
        result_df.to_excel(writer, sheet_name='All Top10', index=False)

        # Summary
        summary = []
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = top1_df[top1_df['model'] == label]
            summary.append({
                'Model': label,
                'N Features': len(mdf),
                'Mean Score': round(mdf['score'].mean(), 4),
                'Mean Cosine': round(mdf['cosine'].mean(), 4),
                'Mean DICE': round(mdf['dice'].mean(), 4),
                'Mean iw_DICE': round(mdf['iw_dice'].mean(), 4),
                'Mean Entropy': round(mdf['entropy'].mean(), 4),
                'Mean Matched Peaks': round(mdf['matched_peaks'].mean(), 1),
            })
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

        # Consensus stats
        n_total = len(comp_df)
        n_yes = (comp_df['consensus'] == 'YES').sum()
        n_partial = (comp_df['consensus'] == 'PARTIAL').sum()
        n_no = (comp_df['consensus'] == 'NO').sum()
        cons_df = pd.DataFrame([{
            'Total Features': n_total,
            'Full Consensus (3/3)': n_yes,
            'Partial (2/3)': n_partial,
            'No Consensus': n_no,
            'Consensus Rate (%)': round(n_yes / n_total * 100, 1),
            'Scoring Formula': 'ent * cos^0.75 * iw_dice^1.25',
            'Energy Weights': 'e0=0.40, e1=0.34, e2=0.26',
        }])
        cons_df.to_excel(writer, sheet_name='Consensus', index=False)

    print(f'\nSaved: {out_xlsx}')

    # CSV
    out_csv = BASE / 'eval_529_top1.csv'
    top1_df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')

    # Print summary
    print(f'\n{"="*100}')
    print(f'  529 Feature Evaluation — Scoring: ent*cos^0.75*iw_dice^1.25')
    print(f'  Energy weights: e0=0.40, e1=0.34, e2=0.26')
    print(f'{"="*100}')
    for model in MODELS:
        label = MODEL_LABELS[model]
        mdf = top1_df[top1_df['model'] == label]
        print(f'\n  {label}:')
        print(f'    Features: {len(mdf)}')
        print(f'    Mean Score: {mdf["score"].mean():.4f}')
        print(f'    Mean Cosine: {mdf["cosine"].mean():.4f}')
        print(f'    Mean iw_DICE: {mdf["iw_dice"].mean():.4f}')
        print(f'    Mean Entropy: {mdf["entropy"].mean():.4f}')

    # Consensus
    print(f'\n  Model Consensus:')
    print(f'    Full (3/3 agree): {n_yes}/{n_total} ({n_yes/n_total*100:.1f}%)')
    print(f'    Partial (2/3):    {n_partial}/{n_total} ({n_partial/n_total*100:.1f}%)')
    print(f'    No consensus:     {n_no}/{n_total} ({n_no/n_total*100:.1f}%)')


if __name__ == '__main__':
    main()
