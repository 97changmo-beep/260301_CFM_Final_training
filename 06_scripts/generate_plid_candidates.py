#!/usr/bin/env python3.10
"""
Generate DTB-modified SMILES candidates for PLID features (819 + 1066).

Workflow:
1. Parse MGF → extract feature m/z
2. Match against union DB (metabolites): generate DTB-SMILES at primary (score=5)
   and secondary amine (score=4) sites
3. Match against peptide DTB DB: m/z match only (predictions already exist)
4. Reuse existing 529 predictions where SMILES overlap
5. Only predict new SMILES that don't exist in 529 or peptide predictions

Usage:
    python3.10 generate_plid_candidates.py
"""

import csv
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

# ─── Paths ───
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
UNION_DB_PATH = os.path.join(PROJECT, '01_library', 'union_db_260301.xlsx')
PEPTIDE_DB_PATH = os.path.join(PROJECT, '05_evaluation', 'peptide_dtb', 'peptide_dtb_candidates.csv')
MGF_DIR = os.path.join(PROJECT, '05_evaluation', 'enriched_mgf')

# Existing 529 predictions (to reuse)
EXISTING_529_MAPPING = os.path.join(PROJECT, '05_evaluation', 'eval_529', 'smiles_mapping.csv')
EXISTING_529_PRED_DIR = os.path.join(PROJECT, '05_evaluation', 'eval_529', 'full_model', 'predictions')

# Existing peptide DTB predictions
EXISTING_PEP_PRED_DIR = os.path.join(PROJECT, '05_evaluation', 'peptide_dtb', 'predictions')

# DTB modification mass
DTB_ACID_MASS = 214.13175
DTB_MOD_MASS = DTB_ACID_MASS - 18.01056  # = 196.12119

PPM_TOLERANCE = 10

# SMARTS
PRIMARY_AMINE = Chem.MolFromSmarts('[NH2;!$([NH2]C=O);!$([NH2]C(=N))]')
SECONDARY_AMINE = Chem.MolFromSmarts('[NH1;!$([NH1]C=O);!$([NH1]C(=N));!$([nH])]')


def parse_mgf(mgf_path):
    """Parse MGF file → list of feature dicts."""
    features = []
    with open(mgf_path) as f:
        in_ions = False
        title = pepmass = rt = fid = None
        peaks = []
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                in_ions = True
                title = pepmass = rt = fid = None
                peaks = []
            elif line == 'END IONS':
                if pepmass is not None:
                    features.append({
                        'title': title or '',
                        'pepmass': pepmass,
                        'rt_seconds': rt or 0,
                        'feature_id': fid or '',
                        'n_peaks': len(peaks),
                    })
                in_ions = False
            elif in_ions:
                if line.startswith('PEPMASS='):
                    pepmass = float(line.split('=')[1].split()[0])
                elif line.startswith('TITLE='):
                    title = line.split('=', 1)[1]
                elif line.startswith('RTINSECONDS='):
                    rt = float(line.split('=')[1])
                elif line.startswith('FEATURE_ID='):
                    fid = line.split('=')[1]
                elif line and line[0].isdigit():
                    peaks.append(1)  # just count
    return features


def load_union_db():
    """Load union DB with precomputed DTB [M+H]+."""
    print("Loading union DB...")
    df = pd.read_excel(UNION_DB_PATH)
    df['dtb_mh'] = df['mass'] + DTB_MOD_MASS + 1.007276
    print(f"  {len(df)} compounds")
    return df


def load_peptide_db():
    """Load peptide DTB candidates (already have DTB SMILES + [M+H]+)."""
    print("Loading peptide DTB DB...")
    df = pd.read_csv(PEPTIDE_DB_PATH)
    df['mh_plus'] = df['mh_plus'].astype(float)
    print(f"  {len(df)} peptide candidates")
    return df


def load_existing_529_smiles():
    """Load SMILES already predicted for 529 evaluation."""
    if not os.path.exists(EXISTING_529_MAPPING):
        return {}
    sm = pd.read_csv(EXISTING_529_MAPPING)
    # Map: dtb_smiles -> cand_id (for finding prediction files)
    return dict(zip(sm['dtb_smiles'], sm['cand_id']))


def load_existing_peptide_smiles():
    """Load peptide SMILES already predicted."""
    pep_df = pd.read_csv(PEPTIDE_DB_PATH)
    # Map: dtb_smiles -> candidate_id (pep_0, pep_1, ...)
    return dict(zip(pep_df['smiles'], pep_df['candidate_id']))


def generate_dtb_smiles(smiles):
    """
    Generate DTB-modified SMILES at primary (score=5) and secondary (score=4) amines.
    Returns list of (dtb_smiles, site_label, score).
    """
    if not isinstance(smiles, str) or not smiles:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    results = []
    seen = set()

    # Primary amines (score=5)
    if mol.HasSubstructMatch(PRIMARY_AMINE):
        rxn = AllChem.ReactionFromSmarts('[NH2:1]>>[NH1:1]C(=O)CCCCC[C@@H]1NC(=O)N[C@@H]1C')
        if rxn:
            products = rxn.RunReactants((mol,))
            for prod_set in products:
                for prod in prod_set:
                    try:
                        Chem.SanitizeMol(prod)
                        smi = Chem.MolToSmiles(prod)
                        if smi not in seen:
                            seen.add(smi)
                            results.append((smi, 'Primary(score=5)', 5))
                    except:
                        pass

    # Secondary amines (score=4)
    if mol.HasSubstructMatch(SECONDARY_AMINE):
        rxn = AllChem.ReactionFromSmarts('[NH1:1]>>[N:1]C(=O)CCCCC[C@@H]1NC(=O)N[C@@H]1C')
        if rxn:
            products = rxn.RunReactants((mol,))
            for prod_set in products:
                for prod in prod_set:
                    try:
                        Chem.SanitizeMol(prod)
                        smi = Chem.MolToSmiles(prod)
                        if smi not in seen:
                            seen.add(smi)
                            results.append((smi, 'Secondary(score=4)', 4))
                    except:
                        pass

    return results


def process_compound(args):
    """Worker: generate DTB variants for one compound."""
    idx, smiles, source, inchikey, hmdb_id, name, mass, freq = args
    variants = generate_dtb_smiles(smiles)
    out = []
    for dtb_smi, site_label, score in variants:
        dtb_mol = Chem.MolFromSmiles(dtb_smi)
        if dtb_mol:
            dtb_mass = Descriptors.ExactMolWt(dtb_mol)
            dtb_mh = dtb_mass + 1.007276
            formula = rdMolDescriptors.CalcMolFormula(dtb_mol)
            cid = hmdb_id if isinstance(hmdb_id, str) and hmdb_id.startswith('HMDB') else f'DMID_{inchikey[:14]}' if isinstance(inchikey, str) else f'IDX_{idx}'
            out.append({
                'source': source,
                'source_id': inchikey if isinstance(inchikey, str) else '',
                'hmdb_id': hmdb_id if isinstance(hmdb_id, str) else '',
                'candidate_name': name if isinstance(name, str) else '',
                'original_smiles': smiles,
                'original_mass': mass,
                'frequency': freq if not pd.isna(freq) else 0,
                'dtb_smiles': dtb_smi,
                'dtb_formula': formula,
                'dtb_mh': dtb_mh,
                'site_label': site_label,
                'score': score,
                'compound_id': cid,
                'db_type': 'union_db',
            })
    return out


def main():
    union_db = load_union_db()
    peptide_db = load_peptide_db()
    existing_529 = load_existing_529_smiles()
    existing_pep = load_existing_peptide_smiles()
    print(f"Existing 529 predictions: {len(existing_529)} SMILES")
    print(f"Existing peptide predictions: {len(existing_pep)} SMILES")

    union_mh = union_db['dtb_mh'].values
    pep_mh = peptide_db['mh_plus'].values

    mgf_files = {
        '24h_819': os.path.join(MGF_DIR, '24h_enriched_DTB_819.mgf'),
        'shTAUT_1066': os.path.join(MGF_DIR, 'shTAUT_enriched_DTB_1066.mgf'),
    }

    all_new_smiles = {}  # smiles -> set of dataset names that need it

    for ds_name, mgf_path in mgf_files.items():
        print(f"\n{'#'*60}")
        print(f"  {ds_name}")
        print(f"{'#'*60}")

        features = parse_mgf(mgf_path)
        print(f"  {len(features)} features parsed")

        # Mass match against union DB
        print("  Matching against union DB...")
        matched_indices = set()
        feat_union = {}
        feat_pep = {}

        for fi, feat in enumerate(features):
            mz = feat['pepmass']
            ppm_diff = np.abs(union_mh - mz) / mz * 1e6
            u_idx = np.where(ppm_diff <= PPM_TOLERANCE)[0]
            if len(u_idx):
                feat_union[fi] = u_idx
                matched_indices.update(u_idx.tolist())

            ppm_diff_p = np.abs(pep_mh - mz) / mz * 1e6
            p_idx = np.where(ppm_diff_p <= PPM_TOLERANCE)[0]
            if len(p_idx):
                feat_pep[fi] = p_idx

        print(f"  Union DB matched features: {len(feat_union)}")
        print(f"  Peptide DB matched features: {len(feat_pep)}")
        print(f"  Unique union compounds to process: {len(matched_indices)}")

        # Generate DTB variants for matched union compounds
        print("  Generating DTB variants (32 workers)...")
        args_list = []
        for idx in sorted(matched_indices):
            r = union_db.iloc[idx]
            args_list.append((idx, r['smiles'], r['source'], r.get('inchikey', ''),
                              r.get('hmdb_id', ''), r.get('name', ''), r['mass'], r.get('frequency', 0)))

        with Pool(min(32, cpu_count())) as pool:
            results_list = pool.map(process_compound, args_list)

        # Build variant lookup
        variant_map = {}
        for idx_val, results in zip(sorted(matched_indices), results_list):
            variant_map[idx_val] = results

        total_variants = sum(len(v) for v in variant_map.values())
        print(f"  DTB variants: {total_variants}")

        # Build candidates table
        candidates = []
        for fi, feat in enumerate(features):
            mz = feat['pepmass']
            rt_min = feat['rt_seconds'] / 60.0

            # Union DB
            if fi in feat_union:
                for uidx in feat_union[fi]:
                    for var in variant_map.get(uidx, []):
                        ppm = abs(var['dtb_mh'] - mz) / mz * 1e6
                        if ppm <= PPM_TOLERANCE:
                            candidates.append({
                                'feature_id': feat['feature_id'],
                                'feature_mz': mz,
                                'feature_rt_min': round(rt_min, 2),
                                'feature_name': feat['title'],
                                'num_ms2_peaks': feat['n_peaks'],
                                'ppm': round(ppm, 2),
                                **var,
                            })

            # Peptide DB
            if fi in feat_pep:
                for pidx in feat_pep[fi]:
                    prow = peptide_db.iloc[pidx]
                    ppm = abs(prow['mh_plus'] - mz) / mz * 1e6
                    if ppm <= PPM_TOLERANCE:
                        candidates.append({
                            'feature_id': feat['feature_id'],
                            'feature_mz': mz,
                            'feature_rt_min': round(rt_min, 2),
                            'feature_name': feat['title'],
                            'num_ms2_peaks': feat['n_peaks'],
                            'ppm': round(ppm, 2),
                            'source': 'peptide_dtb',
                            'source_id': prow['candidate_id'],
                            'hmdb_id': '',
                            'candidate_name': prow['name'],
                            'original_smiles': prow['sequence'],
                            'original_mass': prow['exact_mass'],
                            'frequency': 0,
                            'dtb_smiles': prow['smiles'],
                            'dtb_formula': '',
                            'dtb_mh': prow['mh_plus'],
                            'site_label': prow['site'],
                            'score': 5,
                            'compound_id': prow['candidate_id'],
                            'db_type': 'peptide_dtb',
                        })

        df = pd.DataFrame(candidates)
        # Dedup
        df = df.sort_values('score', ascending=False).drop_duplicates(
            subset=['feature_id', 'dtb_smiles'], keep='first').reset_index(drop=True)

        print(f"  Candidates after dedup: {len(df)}")

        # Output
        out_dir = os.path.join(PROJECT, '05_evaluation', f'eval_{ds_name}')
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f'candidates_{ds_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        # Determine which SMILES need new predictions
        unique_smiles = df['dtb_smiles'].unique()
        new_smiles = []
        reuse_529 = []
        reuse_pep = []
        for smi in unique_smiles:
            if smi in existing_529:
                reuse_529.append(smi)
            elif smi in existing_pep:
                reuse_pep.append(smi)
            else:
                new_smiles.append(smi)
                all_new_smiles[smi] = all_new_smiles.get(smi, set())
                all_new_smiles[smi].add(ds_name)

        print(f"\n  Prediction plan for {ds_name}:")
        print(f"    Total unique SMILES: {len(unique_smiles)}")
        print(f"    Reuse from 529: {len(reuse_529)}")
        print(f"    Reuse from peptide: {len(reuse_pep)}")
        print(f"    NEW (need prediction): {len(new_smiles)}")

        if len(df) > 0:
            n_union = len(df[df['db_type'] == 'union_db'])
            n_pep = len(df[df['db_type'] == 'peptide_dtb'])
            n_feat = df['feature_id'].nunique()
            print(f"\n  Stats:")
            print(f"    Features with candidates: {n_feat} / {len(features)}")
            print(f"    Union DB candidates: {n_union}")
            print(f"    Peptide DB candidates: {n_pep}")
            print(f"    Score 5: {len(df[df['score']==5])}, Score 4: {len(df[df['score']==4])}")

        # Write SMILES mapping and batch files for this dataset
        # Include ALL unique SMILES (for later scoring we need all predictions)
        smiles_map = {}
        for i, smi in enumerate(unique_smiles):
            smiles_map[smi] = f'cand_{ds_name}_{i}'

        mapping_path = os.path.join(out_dir, 'smiles_mapping.csv')
        with open(mapping_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cand_id', 'dtb_smiles', 'prediction_source'])
            for smi, cid in smiles_map.items():
                if smi in existing_529:
                    src = f'529:{existing_529[smi]}'
                elif smi in existing_pep:
                    src = f'peptide:{existing_pep[smi]}'
                else:
                    src = 'new'
                writer.writerow([cid, smi, src])

    # Write combined batch files for NEW SMILES only (shared across both datasets)
    print(f"\n{'='*60}")
    print(f"  NEW SMILES requiring prediction: {len(all_new_smiles)}")
    print(f"{'='*60}")

    if all_new_smiles:
        new_pred_dir = os.path.join(PROJECT, '05_evaluation', 'plid_new_predictions')
        os.makedirs(os.path.join(new_pred_dir, 'predictions'), exist_ok=True)

        # Write batch chunks
        new_list = list(all_new_smiles.keys())
        chunk_size = 500
        n_chunks = 0
        with open(os.path.join(new_pred_dir, 'new_smiles_list.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cand_id', 'smiles', 'datasets'])
            for i, smi in enumerate(new_list):
                ds = ','.join(all_new_smiles[smi])
                writer.writerow([f'new_{i}', smi, ds])

        for i in range(0, len(new_list), chunk_size):
            chunk = new_list[i:i+chunk_size]
            chunk_idx = i // chunk_size
            chunk_path = os.path.join(new_pred_dir, f'batch_chunk_{chunk_idx}.txt')
            with open(chunk_path, 'w') as f:
                for j, smi in enumerate(chunk):
                    f.write(f'new_{i+j} {smi}\n')
            n_chunks += 1

        print(f"  Batch files: {n_chunks} chunks in {new_pred_dir}")
    else:
        print("  No new predictions needed!")

    print("\nDone!")


if __name__ == '__main__':
    main()
