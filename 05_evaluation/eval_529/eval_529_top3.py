#!/usr/bin/env python3
"""
eval_529_top3.py
================
3개 모델(default, JJY, final)의 529 feature prediction에 대해
compound_id(DMID/HMDB) 포함, TOP3 후보 + 평가지표 엑셀 생성.

지표: cosine, dice, iw_dice, matched_peaks, entropy, composite score
에너지별(e0/e1/e2) + weighted average
"""
import math
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
MODELS = ['cfm_default', 'param_jjy', 'full_model']
MODEL_LABELS = {
    'cfm_default': 'Default',
    'param_jjy': 'JJY',
    'full_model': 'Final',
}
ENERGIES = [0, 1, 2]
OPT_WEIGHTS = {0: 0.40, 1: 0.34, 2: 0.26}
MGF_PATH = BASE.parent.parent / '01_library' / '529_invitroDESTNI_features.mgf'
TOP_K = 3


# ============================================================
# Spectrum parsing
# ============================================================
def parse_mgf(mgf_path):
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
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return spectra


def parse_predicted_spectrum(path, energy=0):
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
# Metrics
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


def scoring_fn(m):
    return m['ent'] * (m['cos'] ** 0.75) * (m['iw_dice'] ** 1.25)


# ============================================================
# Main
# ============================================================
def main():
    print('Loading candidates...')
    cands = pd.read_csv(BASE / 'candidates_529.csv')
    smiles_map = pd.read_csv(BASE / 'smiles_mapping.csv')
    smi_to_cand = dict(zip(smiles_map['dtb_smiles'], smiles_map['cand_id']))
    cands['cand_id'] = cands['dtb_smiles'].map(smi_to_cand)

    n_features = cands['feature_id'].nunique()
    print(f'Features: {n_features}, Candidates: {len(cands)}')

    print('Parsing experimental MGF...')
    exp_spectra = parse_mgf(MGF_PATH)
    print(f'Experimental spectra: {len(exp_spectra)}')

    # === Process each model ===
    all_results = []  # for All_Top3 sheet
    model_top3 = {m: [] for m in MODELS}  # per-model sheets

    feature_ids = sorted(cands['feature_id'].unique())

    for model in MODELS:
        label = MODEL_LABELS[model]
        pred_dir = BASE / model / 'predictions'
        print(f'\n=== {label} ({model}) ===')

        n_scored = 0
        for feat_id in feature_ids:
            sub = cands[cands['feature_id'] == feat_id]
            exp_spec = exp_spectra.get(str(feat_id))
            if not exp_spec:
                continue

            feat_mz = sub['feature_mz'].iloc[0]
            feat_rt = sub['feature_rt_min'].iloc[0]
            n_cands = len(sub)

            cand_scores = []
            for _, row in sub.iterrows():
                cand_id = row['cand_id']
                compound_id = row.get('compound_id', '')
                dtb_smi = row['dtb_smiles']
                orig_smi = row.get('original_smiles', '')
                hmdb_id = row.get('hmdb_id', '')
                cand_name = row.get('candidate_name', '')
                source = row.get('source', '')

                pred_file = pred_dir / f'{cand_id}.log'
                if not pred_file.exists() or pd.isna(cand_id):
                    cand_scores.append({
                        'compound_id': compound_id,
                        'cand_id': cand_id,
                        'dtb_smiles': dtb_smi,
                        'original_smiles': orig_smi,
                        'hmdb_id': hmdb_id,
                        'candidate_name': cand_name,
                        'source': source,
                        'score': 0.0,
                        'cos_w': 0.0, 'dice_w': 0.0, 'iw_dice_w': 0.0,
                        'ent_w': 0.0, 'mp_w': 0.0,
                        'cos_e0': 0.0, 'cos_e1': 0.0, 'cos_e2': 0.0,
                        'dice_e0': 0.0, 'dice_e1': 0.0, 'dice_e2': 0.0,
                        'mp_e0': 0, 'mp_e1': 0, 'mp_e2': 0,
                        'ent_e0': 0.0, 'ent_e1': 0.0, 'ent_e2': 0.0,
                    })
                    continue

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

                wm = {}
                for metric in ['cos', 'dice', 'iw_dice', 'ent', 'mp']:
                    wm[metric] = sum(OPT_WEIGHTS[e] * per_e[e][metric] for e in ENERGIES)

                score = scoring_fn(wm)

                cand_scores.append({
                    'compound_id': compound_id,
                    'cand_id': cand_id,
                    'dtb_smiles': dtb_smi,
                    'original_smiles': orig_smi,
                    'hmdb_id': hmdb_id,
                    'candidate_name': cand_name,
                    'source': source,
                    'score': score,
                    'cos_w': wm['cos'], 'dice_w': wm['dice'],
                    'iw_dice_w': wm['iw_dice'], 'ent_w': wm['ent'],
                    'mp_w': wm['mp'],
                    'cos_e0': per_e[0]['cos'], 'cos_e1': per_e[1]['cos'],
                    'cos_e2': per_e[2]['cos'],
                    'dice_e0': per_e[0]['dice'], 'dice_e1': per_e[1]['dice'],
                    'dice_e2': per_e[2]['dice'],
                    'mp_e0': per_e[0]['mp'], 'mp_e1': per_e[1]['mp'],
                    'mp_e2': per_e[2]['mp'],
                    'ent_e0': per_e[0]['ent'], 'ent_e1': per_e[1]['ent'],
                    'ent_e2': per_e[2]['ent'],
                })

            if not cand_scores:
                continue

            ranked = sorted(cand_scores, key=lambda x: x['score'], reverse=True)
            if ranked[0]['score'] > 0:
                n_scored += 1

            for rank_i, c in enumerate(ranked[:TOP_K]):
                row_out = {
                    'model': label,
                    'feature_id': feat_id,
                    'feature_mz': round(feat_mz, 5),
                    'feature_rt': round(feat_rt, 2) if pd.notna(feat_rt) else '',
                    'n_candidates': n_cands,
                    'rank': rank_i + 1,
                    'compound_id': c['compound_id'] if pd.notna(c['compound_id']) else '',
                    'hmdb_id': c['hmdb_id'] if pd.notna(c['hmdb_id']) else '',
                    'candidate_name': c['candidate_name'] if pd.notna(c['candidate_name']) else '',
                    'source': c['source'] if pd.notna(c['source']) else '',
                    'original_smiles': c['original_smiles'] if pd.notna(c['original_smiles']) else '',
                    'dtb_smiles': c['dtb_smiles'],
                    'score': round(c['score'], 6),
                    'cos_weighted': round(c['cos_w'], 4),
                    'dice_weighted': round(c['dice_w'], 4),
                    'iw_dice_weighted': round(c['iw_dice_w'], 4),
                    'entropy_weighted': round(c['ent_w'], 4),
                    'matched_peaks_weighted': round(c['mp_w'], 1),
                    'cos_e0': round(c['cos_e0'], 4),
                    'cos_e1': round(c['cos_e1'], 4),
                    'cos_e2': round(c['cos_e2'], 4),
                    'dice_e0': round(c['dice_e0'], 4),
                    'dice_e1': round(c['dice_e1'], 4),
                    'dice_e2': round(c['dice_e2'], 4),
                    'matched_peaks_e0': c['mp_e0'],
                    'matched_peaks_e1': c['mp_e1'],
                    'matched_peaks_e2': c['mp_e2'],
                    'entropy_e0': round(c['ent_e0'], 4),
                    'entropy_e1': round(c['ent_e1'], 4),
                    'entropy_e2': round(c['ent_e2'], 4),
                }
                all_results.append(row_out)
                model_top3[model].append(row_out)

        print(f'  Features with score > 0: {n_scored}')

    # === Build Excel ===
    print('\nBuilding Excel...')
    result_df = pd.DataFrame(all_results)
    out_path = BASE / 'eval_529_top3_results.xlsx'

    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_rows = []
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = result_df[(result_df['model'] == label) & (result_df['rank'] == 1)]
            summary_rows.append({
                'Model': label,
                'N_Features': len(mdf),
                'Mean_Score': round(mdf['score'].mean(), 4),
                'Mean_Cosine': round(mdf['cos_weighted'].mean(), 4),
                'Mean_DICE': round(mdf['dice_weighted'].mean(), 4),
                'Mean_iw_DICE': round(mdf['iw_dice_weighted'].mean(), 4),
                'Mean_Entropy': round(mdf['entropy_weighted'].mean(), 4),
                'Mean_Matched_Peaks': round(mdf['matched_peaks_weighted'].mean(), 1),
                'Score>0.1': (mdf['score'] > 0.1).sum(),
                'Score>0.3': (mdf['score'] > 0.3).sum(),
                'Cos>0.5': (mdf['cos_weighted'] > 0.5).sum(),
                'Scoring': 'ent*cos^0.75*iw_dice^1.25',
                'Weights': 'e0=0.40,e1=0.34,e2=0.26',
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2-4: Per-model TOP3
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = pd.DataFrame(model_top3[model])
            mdf.drop(columns=['model'], inplace=True, errors='ignore')
            mdf.to_excel(writer, sheet_name=f'TOP3_{label}', index=False)

        # Sheet 5: Model Comparison (TOP1 side-by-side)
        top1_df = result_df[result_df['rank'] == 1]
        comparison = []
        for feat_id in sorted(top1_df['feature_id'].unique()):
            row = {'feature_id': feat_id}
            feat_sub = top1_df[top1_df['feature_id'] == feat_id]
            row['feature_mz'] = feat_sub['feature_mz'].iloc[0]
            row['feature_rt'] = feat_sub['feature_rt'].iloc[0]

            for model in MODELS:
                label = MODEL_LABELS[model]
                msub = feat_sub[feat_sub['model'] == label]
                if len(msub) > 0:
                    r = msub.iloc[0]
                    row[f'{label}_compound_id'] = r['compound_id']
                    row[f'{label}_name'] = r['candidate_name']
                    row[f'{label}_score'] = r['score']
                    row[f'{label}_cos'] = r['cos_weighted']
                    row[f'{label}_dice'] = r['dice_weighted']
                    row[f'{label}_entropy'] = r['entropy_weighted']
                    row[f'{label}_mp'] = r['matched_peaks_weighted']

            # Consensus
            cids = []
            for model in MODELS:
                label = MODEL_LABELS[model]
                msub = feat_sub[feat_sub['model'] == label]
                if len(msub) > 0:
                    cids.append(msub.iloc[0]['compound_id'])
            unique_cids = set(c for c in cids if c)
            if len(unique_cids) == 1 and len(cids) == 3:
                row['consensus'] = 'YES'
            elif len(unique_cids) <= 2 and len(cids) >= 2:
                row['consensus'] = 'PARTIAL'
            else:
                row['consensus'] = 'NO'

            comparison.append(row)

        pd.DataFrame(comparison).to_excel(writer, sheet_name='Comparison', index=False)

        # Sheet 6: All results
        result_df.to_excel(writer, sheet_name='All_Top3', index=False)

    print(f'\nSaved: {out_path}')
    print(f'Total rows: {len(result_df)}')

    # Print summary
    print(f'\n{"="*80}')
    print(f'  529 Feature Evaluation - TOP3 per model')
    print(f'  Scoring: ent*cos^0.75*iw_dice^1.25 | Weights: e0=0.40,e1=0.34,e2=0.26')
    print(f'{"="*80}')
    for model in MODELS:
        label = MODEL_LABELS[model]
        mdf = result_df[(result_df['model'] == label) & (result_df['rank'] == 1)]
        print(f'\n  {label}:')
        print(f'    Features: {len(mdf)}')
        print(f'    Mean Score: {mdf["score"].mean():.4f}')
        print(f'    Mean Cosine: {mdf["cos_weighted"].mean():.4f}')
        print(f'    Mean DICE: {mdf["dice_weighted"].mean():.4f}')
        print(f'    Mean Entropy: {mdf["entropy_weighted"].mean():.4f}')
        print(f'    Mean Matched Peaks: {mdf["matched_peaks_weighted"].mean():.1f}')
        print(f'    Score > 0.1: {(mdf["score"] > 0.1).sum()}/{len(mdf)}')
        print(f'    Cosine > 0.5: {(mdf["cos_weighted"] > 0.5).sum()}/{len(mdf)}')


if __name__ == '__main__':
    main()
