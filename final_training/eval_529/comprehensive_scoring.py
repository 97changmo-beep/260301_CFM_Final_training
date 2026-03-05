#!/usr/bin/env python3.10
"""
comprehensive_scoring.py
========================
Multi-method scoring for 529 feature candidates.

Methods:
  1. CFM-ID spectral matching (optimized: ent*cos^0.75*iw_dice^1.25)
  2. DTB-fragment-removed spectral matching
  3. sqrt-intensity spectral matching
  4. RDKit in-silico fragment matching (bond-breaking)
  5. Neutral loss matching
  6. Combined hybrid scoring

First validates on 143 library compounds, then applies to 529 features.
"""
import math, re, sys, warnings
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import FragmentCatalog, BRICS
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, fragment matching disabled")

BASE = Path(__file__).resolve().parent
MODELS = ['cfm_default', 'param_jjy', 'full_model']
MODEL_LABELS = {
    'cfm_default': 'CFM-ID Default',
    'param_jjy': 'Param_JJY',
    'full_model': 'Full Model (Final)',
}
ENERGIES = [0, 1, 2]
OPT_WEIGHTS = {0: 0.40, 1: 0.34, 2: 0.26}
MGF_529 = BASE.parent.parent / '529_invitroDESTNI_features.mgf'
MGF_143 = BASE.parent.parent / '143_final_library_260301.mgf'


# ============================================================
# Spectrum parsing
# ============================================================
def parse_mgf(mgf_path):
    spectra = {}
    meta = {}
    current_id = None
    peaks = []
    in_peaks = False
    cur_meta = {}

    with open(mgf_path) as f:
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                current_id = None
                peaks = []
                in_peaks = False
                cur_meta = {}
            elif line.startswith('FEATURE_ID='):
                current_id = line.split('=', 1)[1]
            elif line.startswith('NAME='):
                if current_id is None:
                    current_id = line.split('=', 1)[1]
                cur_meta['name'] = line.split('=', 1)[1]
            elif line.startswith('PEPMASS='):
                cur_meta['pepmass'] = float(line.split('=', 1)[1])
            elif line.startswith('SMILES='):
                cur_meta['smiles'] = line.split('=', 1)[1]
            elif line.startswith('Num peaks='):
                in_peaks = True
            elif line == 'END IONS':
                if current_id and peaks:
                    spectra[current_id] = peaks
                    meta[current_id] = cur_meta
                in_peaks = False
            elif in_peaks and line:
                parts = line.split('\t')
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return spectra, meta


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
# Spectrum preprocessing
# ============================================================
def find_common_peaks(spectra_dict, mz_tol=0.02, min_freq=0.3):
    """Find m/z values that appear in >min_freq fraction of spectra."""
    all_mz = []
    for peaks in spectra_dict.values():
        for mz, _ in peaks:
            all_mz.append(round(mz, 2))

    mz_counts = Counter(all_mz)
    n_spectra = len(spectra_dict)
    common = [mz for mz, cnt in mz_counts.items()
              if cnt / n_spectra >= min_freq]
    return sorted(common)


def remove_peaks(peaks, mz_list, mz_tol=0.02):
    """Remove peaks at specified m/z values."""
    filtered = []
    for mz, intensity in peaks:
        remove = False
        for common_mz in mz_list:
            if abs(mz - common_mz) <= mz_tol:
                remove = True
                break
        if not remove:
            filtered.append((mz, intensity))
    return filtered


def sqrt_transform(peaks):
    """Apply sqrt intensity transformation."""
    return [(mz, math.sqrt(abs(intensity))) for mz, intensity in peaks]


def remove_precursor(peaks, precursor_mz, tol=0.5):
    """Remove precursor peak and nearby."""
    return [(mz, i) for mz, i in peaks if abs(mz - precursor_mz) > tol]


def top_n_peaks(peaks, n=50):
    """Keep only top N peaks by intensity."""
    if len(peaks) <= n:
        return peaks
    sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
    return sorted(sorted_peaks[:n], key=lambda x: x[0])


# ============================================================
# Spectral metrics
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


def cosine_sim(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not matched or not s1 or not s2:
        return 0.0
    dot = sum(a * b for a, b in matched)
    denom = math.sqrt(sum(i**2 for _, i in s1) * sum(i**2 for _, i in s2))
    return dot / denom if denom > 0 else 0.0


def iw_dice_sim(spec1, spec2, mz_tol=0.01):
    matched, s1, s2 = match_peaks(spec1, spec2, mz_tol)
    if not s1 or not s2:
        return 0.0
    matched_sum = sum(i1 + i2 for i1, i2 in matched)
    total_sum = sum(i for _, i in s1) + sum(i for _, i in s2)
    return matched_sum / total_sum if total_sum > 0 else 0.0


def entropy_sim(spec1, spec2, mz_tol=0.01):
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


def matched_peaks_count(spec1, spec2, mz_tol=0.01):
    matched, _, _ = match_peaks(spec1, spec2, mz_tol)
    return len(matched)


# ============================================================
# RDKit fragment generation
# ============================================================
def generate_fragment_masses(smiles, max_cuts=3):
    """Generate possible fragment m/z values from SMILES using bond breaking."""
    if not HAS_RDKIT:
        return []
    if not isinstance(smiles, str) or not smiles:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)
    exact_mass = Descriptors.ExactMolWt(mol)

    fragment_masses = set()
    fragment_masses.add(round(exact_mass + 1.00728, 4))  # [M+H]+

    # BRICS decomposition
    try:
        brics_frags = list(BRICS.BRICSDecompose(Chem.RemoveHs(mol)))
        for frag_smi in brics_frags:
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol:
                frag_mol = Chem.AddHs(frag_mol)
                fm = Descriptors.ExactMolWt(frag_mol)
                fragment_masses.add(round(fm + 1.00728, 4))
    except:
        pass

    # Simple bond breaking: break each non-ring single bond
    mol_noH = Chem.RemoveHs(mol)
    ri = mol_noH.GetRingInfo()
    for bond in mol_noH.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if ri.NumBondRings(bond.GetIdx()) > 0:
            continue

        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()

        try:
            frag_mol = Chem.FragmentOnBonds(mol_noH, [bond.GetIdx()], addDummies=True)
            frags = Chem.GetMolFrags(frag_mol, asMols=True)
            for f in frags:
                # Replace dummy atoms ([1*], [2*], etc.) with [H] using regex
                f_smi = Chem.MolToSmiles(f)
                f_smi = re.sub(r'\[\d+\*\]', '[H]', f_smi)
                f_smi = re.sub(r'\[\*:\d+\]', '[H]', f_smi)
                f_smi = f_smi.replace('[*]', '[H]')
                f_mol = Chem.MolFromSmiles(f_smi)
                if f_mol:
                    f_mol = Chem.AddHs(f_mol)
                    fm = Descriptors.ExactMolWt(f_mol)
                    if fm > 50:  # ignore very small fragments
                        fragment_masses.add(round(fm + 1.00728, 4))  # [M+H]+
                        fragment_masses.add(round(fm, 4))  # radical cation
        except:
            pass

    # Common neutral losses from precursor
    common_losses = [
        18.010565,   # H2O
        17.002740,   # NH3
        27.994915,   # CO
        43.989829,   # CO2
        31.972071,   # CH3OH (MeOH) or S
        46.005479,   # CH2O2 (formic acid)
        17.026549,   # OH
        28.031300,   # C2H4
        44.026215,   # C2H4O
        36.021129,   # 2×H2O
    ]
    precursor = exact_mass + 1.00728
    for loss in common_losses:
        nl_mass = precursor - loss
        if nl_mass > 50:
            fragment_masses.add(round(nl_mass, 4))

    return sorted(fragment_masses)


def fragment_match_score(exp_peaks, candidate_masses, mz_tol=0.01):
    """Score: fraction of experimental peaks explained by candidate fragments."""
    if not exp_peaks or not candidate_masses:
        return 0.0, 0

    n_explained = 0
    total_intensity_explained = 0
    total_intensity = sum(i for _, i in exp_peaks)

    for mz, intensity in exp_peaks:
        for fm in candidate_masses:
            if abs(mz - fm) <= mz_tol:
                n_explained += 1
                total_intensity_explained += intensity
                break

    n_peaks = len(exp_peaks)
    peak_ratio = n_explained / n_peaks if n_peaks > 0 else 0.0
    intensity_ratio = total_intensity_explained / total_intensity if total_intensity > 0 else 0.0

    return peak_ratio, intensity_ratio


def neutral_loss_score(exp_peaks, precursor_mz, candidate_smiles, mz_tol=0.02):
    """Score based on how many neutral losses match expected losses from structure."""
    if not HAS_RDKIT or not exp_peaks:
        return 0.0
    if not isinstance(candidate_smiles, str) or not candidate_smiles:
        return 0.0

    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        return 0.0

    # Determine expected neutral losses from molecular structure
    expected_losses = set()
    expected_losses.add(18.0106)  # H2O (if OH/COOH present)
    expected_losses.add(17.0027)  # NH3 (if NH2 present)
    expected_losses.add(27.9949)  # CO
    expected_losses.add(43.9898)  # CO2

    # Check functional groups
    smarts_losses = {
        '[OH]': [18.0106],
        '[NH2]': [17.0027],
        'C(=O)O': [43.9898, 18.0106],
        'C(=O)N': [27.9949, 17.0027],
        'OC': [31.9721],  # methoxy
        'C=O': [27.9949],
    }
    for smarts, losses in smarts_losses.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            expected_losses.update(losses)

    # Compute observed neutral losses
    observed_losses = []
    for mz, intensity in exp_peaks:
        nl = precursor_mz - mz
        if 10 < nl < precursor_mz - 50:
            observed_losses.append((nl, intensity))

    if not observed_losses:
        return 0.0

    # Match
    n_matched = 0
    intensity_matched = 0
    total_intensity = sum(i for _, i in observed_losses)

    for nl, intensity in observed_losses:
        for expected_nl in expected_losses:
            if abs(nl - expected_nl) <= mz_tol:
                n_matched += 1
                intensity_matched += intensity
                break

    return intensity_matched / total_intensity if total_intensity > 0 else 0.0


# ============================================================
# Multi-method scoring
# ============================================================
def compute_all_scores(exp_spec, pred_specs_by_energy, candidate_smiles,
                       precursor_mz, common_peaks_mz=None):
    """Compute all scoring methods for one candidate."""
    results = {}

    # --- Method 1: Standard CFM-ID scoring (optimized) ---
    metrics_per_e = {}
    for e in ENERGIES:
        pred = pred_specs_by_energy.get(e, [])
        metrics_per_e[e] = {
            'cos': cosine_sim(exp_spec, pred),
            'iw_dice': iw_dice_sim(exp_spec, pred),
            'ent': entropy_sim(exp_spec, pred),
            'mp': matched_peaks_count(exp_spec, pred),
        }

    wm = {}
    for m in ['cos', 'iw_dice', 'ent', 'mp']:
        wm[m] = sum(OPT_WEIGHTS[e] * metrics_per_e[e][m] for e in ENERGIES)

    results['cfm_score'] = wm['ent'] * (wm['cos'] ** 0.75) * (wm['iw_dice'] ** 1.25)
    results['cfm_cos'] = wm['cos']
    results['cfm_iw_dice'] = wm['iw_dice']
    results['cfm_ent'] = wm['ent']

    # --- Method 2: DTB-removed scoring ---
    if common_peaks_mz:
        exp_clean = remove_peaks(exp_spec, common_peaks_mz)
        if exp_clean:
            clean_per_e = {}
            for e in ENERGIES:
                pred = pred_specs_by_energy.get(e, [])
                pred_clean = remove_peaks(pred, common_peaks_mz)
                clean_per_e[e] = {
                    'cos': cosine_sim(exp_clean, pred_clean),
                    'iw_dice': iw_dice_sim(exp_clean, pred_clean),
                    'ent': entropy_sim(exp_clean, pred_clean),
                }
            wm_clean = {}
            for m in ['cos', 'iw_dice', 'ent']:
                wm_clean[m] = sum(OPT_WEIGHTS[e] * clean_per_e[e][m] for e in ENERGIES)
            results['dtb_removed_score'] = wm_clean['ent'] * (wm_clean['cos']**0.75) * (wm_clean['iw_dice']**1.25)
            results['dtb_removed_cos'] = wm_clean['cos']
        else:
            results['dtb_removed_score'] = 0.0
            results['dtb_removed_cos'] = 0.0
    else:
        results['dtb_removed_score'] = results['cfm_score']
        results['dtb_removed_cos'] = results['cfm_cos']

    # --- Method 3: sqrt-transformed scoring ---
    exp_sqrt = sqrt_transform(exp_spec)
    sqrt_per_e = {}
    for e in ENERGIES:
        pred = pred_specs_by_energy.get(e, [])
        pred_sqrt = sqrt_transform(pred)
        sqrt_per_e[e] = {
            'cos': cosine_sim(exp_sqrt, pred_sqrt),
            'iw_dice': iw_dice_sim(exp_sqrt, pred_sqrt),
            'ent': entropy_sim(exp_sqrt, pred_sqrt),
        }
    wm_sqrt = {}
    for m in ['cos', 'iw_dice', 'ent']:
        wm_sqrt[m] = sum(OPT_WEIGHTS[e] * sqrt_per_e[e][m] for e in ENERGIES)
    results['sqrt_score'] = wm_sqrt['ent'] * (wm_sqrt['cos']**0.75) * (wm_sqrt['iw_dice']**1.25)
    results['sqrt_cos'] = wm_sqrt['cos']

    # --- Method 4: RDKit fragment matching ---
    if HAS_RDKIT and candidate_smiles:
        frag_masses = generate_fragment_masses(candidate_smiles)
        peak_ratio, intensity_ratio = fragment_match_score(exp_spec, frag_masses)
        results['frag_peak_ratio'] = peak_ratio
        results['frag_intensity_ratio'] = intensity_ratio
        results['n_frag_masses'] = len(frag_masses)
    else:
        results['frag_peak_ratio'] = 0.0
        results['frag_intensity_ratio'] = 0.0
        results['n_frag_masses'] = 0

    # --- Method 5: Neutral loss matching ---
    if HAS_RDKIT and candidate_smiles and precursor_mz:
        results['nl_score'] = neutral_loss_score(exp_spec, precursor_mz, candidate_smiles)
    else:
        results['nl_score'] = 0.0

    # --- Method 6: Hybrid combined score ---
    # Normalize and combine: CFM + fragment + neutral loss
    cfm = results['cfm_score']
    frag = results['frag_intensity_ratio']
    nl = results['nl_score']
    sqrt_s = results['sqrt_score']
    dtb_s = results['dtb_removed_score']

    # Hybrid: weighted combination (CFM-ID dominant, fragment/NL as bonus)
    results['hybrid_score'] = (
        0.4 * cfm +
        0.2 * sqrt_s +
        0.2 * dtb_s +
        0.1 * frag +
        0.1 * nl
    )

    return results


# ============================================================
# Parallel worker functions (must be top-level for pickle)
# ============================================================
def _score_one_candidate_143(args):
    """Worker: score one candidate for 143 validation."""
    cand_id, cand_smi, cand_hmdb, pred_dir, exp_spec, precursor_mz, common_peaks = args
    pred_specs = {}
    pred_file = Path(pred_dir) / f'{cand_id}.log'
    if pred_file.exists():
        for e in ENERGIES:
            pred_specs[e] = parse_predicted_spectrum(pred_file, e)
    scores = compute_all_scores(exp_spec, pred_specs, cand_smi, precursor_mz, common_peaks)
    return (scores, cand_hmdb)


def _score_one_lib_143(args):
    """Worker: score all candidates for one library compound."""
    lib_name, sub_rows, exp_spec, precursor_mz, common_peaks, pred_dir = args
    cand_all = []
    for row in sub_rows:
        cand_id = row['cand_id']
        cand_smi = row['cand_dtb_smiles']
        cand_hmdb = str(row.get('cand_hmdb_id', '')) if pd.notna(row.get('cand_hmdb_id', '')) else ''
        pred_specs = {}
        pred_file = Path(pred_dir) / f'{cand_id}.log'
        if pred_file.exists():
            for e in ENERGIES:
                pred_specs[e] = parse_predicted_spectrum(pred_file, e)
        scores = compute_all_scores(exp_spec, pred_specs, cand_smi, precursor_mz, common_peaks)
        cand_all.append((scores, cand_hmdb))
    return lib_name, cand_all


def _score_one_feature_529(args):
    """Worker: score all candidates for one 529 feature."""
    feat_id, sub_rows, exp_spec, precursor_mz, common_peaks, pred_dir, model, model_label = args
    cand_results = []
    for row in sub_rows:
        cand_id = row['cand_id']
        cand_smi = row['dtb_smiles']
        cand_hmdb = str(row.get('hmdb_id', '')) if pd.notna(row.get('hmdb_id', '')) else ''
        cand_name = str(row.get('candidate_name', '')) if pd.notna(row.get('candidate_name', '')) else ''

        pred_specs = {}
        pred_file = Path(pred_dir) / f'{cand_id}.log'
        if pred_file.exists() and pd.notna(cand_id):
            for e in ENERGIES:
                pred_specs[e] = parse_predicted_spectrum(pred_file, e)

        scores = compute_all_scores(exp_spec, pred_specs, cand_smi, precursor_mz, common_peaks)
        cand_results.append({
            'cand_id': cand_id, 'cand_smi': cand_smi,
            'cand_hmdb': cand_hmdb, 'cand_name': cand_name,
            **scores,
        })
    return feat_id, sub_rows[0].get('feature_mz', 0), sub_rows[0].get('feature_rt_min', ''), \
           len(sub_rows), cand_results, model_label


N_WORKERS = min(32, cpu_count())


# ============================================================
# Validate on 143 library compounds
# ============================================================
def validate_143():
    """Validate scoring methods on 143 library compounds."""
    print('\n' + '='*100)
    print('  PHASE 1: Validate on 143 Library Compounds')
    print(f'  Workers: {N_WORKERS}')
    print('='*100)

    candidates_csv = BASE.parent / 'top1_eval' / 'candidates.csv'
    if not candidates_csv.exists():
        print('  candidates.csv not found, skipping validation')
        return {}

    df = pd.read_csv(candidates_csv)
    exp_dir = BASE.parent / 'full_model' / 'spectra'
    pred_dir = BASE.parent / 'top1_eval' / 'full_model' / 'predictions'

    # Find common peaks in 143 library experimental spectra
    exp_spectra_143 = {}
    for f in exp_dir.glob('*.txt'):
        peaks = []
        with open(f) as fh:
            in_e0 = False
            for line in fh:
                line = line.strip()
                if line == 'energy0':
                    in_e0 = True; peaks = []; continue
                if in_e0:
                    if line.startswith('energy') or line == '':
                        break
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            peaks.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
        exp_spectra_143[f.stem] = peaks

    common_143 = find_common_peaks(exp_spectra_143, min_freq=0.3)
    print(f'  Common peaks in 143 spectra (>30%): {len(common_143)} m/z values')
    print(f'  Examples: {common_143[:10]}')

    # Prepare parallel tasks: one task per library compound
    tasks = []
    lib_info = {}  # lib_name -> correct_hmdb
    for lib_name in sorted(df['lib_name'].unique()):
        sub = df[df['lib_name'] == lib_name]
        hmdb_id = sub['hmdb_id'].iloc[0]
        correct_hmdb = str(hmdb_id) if pd.notna(hmdb_id) else ''

        exp_file = exp_dir / f'{lib_name}.txt'
        if not exp_file.exists():
            continue
        exp_spec = exp_spectra_143.get(lib_name, [])
        if not exp_spec:
            continue

        parts = lib_name.rsplit('_', 2)
        precursor_mz = float(parts[1]) if len(parts) >= 3 else 0

        sub_rows = sub.to_dict('records')
        tasks.append((lib_name, sub_rows, exp_spec, precursor_mz, common_143, str(pred_dir)))
        lib_info[lib_name] = correct_hmdb

    print(f'  Scoring {len(tasks)} library compounds with {N_WORKERS} workers...')

    # Parallel scoring
    all_lib_scores = {}
    with Pool(N_WORKERS) as pool:
        results = pool.map(_score_one_lib_143, tasks)
    for lib_name, cand_all in results:
        all_lib_scores[lib_name] = (cand_all, lib_info[lib_name])

    # Evaluate each scoring method
    scoring_methods = ['cfm_score', 'dtb_removed_score', 'sqrt_score',
                       'frag_intensity_ratio', 'nl_score', 'hybrid_score']
    method_correct = {m: 0 for m in scoring_methods}
    method_top3 = {m: 0 for m in scoring_methods}
    n_total = 0
    n_in_cands = 0

    for lib_name, (cand_all, correct_hmdb) in all_lib_scores.items():
        n_total += 1
        has_correct = any(h == correct_hmdb for _, h in cand_all if correct_hmdb)
        if has_correct:
            n_in_cands += 1

        for m in scoring_methods:
            ranked = sorted(cand_all, key=lambda x: x[0][m], reverse=True)
            if ranked and ranked[0][1] == correct_hmdb and correct_hmdb:
                method_correct[m] += 1
            for i, (_, h) in enumerate(ranked[:3]):
                if h == correct_hmdb and correct_hmdb:
                    method_top3[m] += 1
                    break

    print(f'\n  143 Library Validation Results (n={n_total}, in_cands={n_in_cands}):')
    print(f'  {"Method":<30} {"TOP1":>8} {"TOP1%":>8} {"TOP3":>8} {"TOP3%":>8}')
    print('  ' + '-'*70)
    for m in scoring_methods:
        t1 = method_correct[m]
        t3 = method_top3[m]
        print(f'  {m:<30} {t1:>8} {t1/n_total*100:>7.1f}% {t3:>8} {t3/n_in_cands*100:>7.1f}%')

    # Grid search over hybrid weights
    print('\n  Grid searching hybrid weights...')
    best_hybrid_top1 = 0
    best_weights = None

    for w_cfm in np.arange(0.1, 0.8, 0.1):
        for w_sqrt in np.arange(0.0, 0.5, 0.1):
            for w_dtb in np.arange(0.0, 0.5, 0.1):
                for w_frag in np.arange(0.0, 0.4, 0.1):
                    w_nl = round(1.0 - w_cfm - w_sqrt - w_dtb - w_frag, 2)
                    if w_nl < 0 or w_nl > 0.4:
                        continue
                    if abs(w_cfm + w_sqrt + w_dtb + w_frag + w_nl - 1.0) > 0.01:
                        continue

                    n_correct = 0
                    for lib_name, (cand_all, correct_hmdb) in all_lib_scores.items():
                        if not correct_hmdb:
                            continue
                        ranked = sorted(cand_all, key=lambda x: (
                            w_cfm * x[0]['cfm_score'] +
                            w_sqrt * x[0]['sqrt_score'] +
                            w_dtb * x[0]['dtb_removed_score'] +
                            w_frag * x[0]['frag_intensity_ratio'] +
                            w_nl * x[0]['nl_score']
                        ), reverse=True)
                        if ranked and ranked[0][1] == correct_hmdb:
                            n_correct += 1

                    if n_correct > best_hybrid_top1:
                        best_hybrid_top1 = n_correct
                        best_weights = {
                            'cfm': round(w_cfm, 2), 'sqrt': round(w_sqrt, 2),
                            'dtb': round(w_dtb, 2), 'frag': round(w_frag, 2),
                            'nl': round(w_nl, 2)
                        }

    print(f'\n  Best Hybrid: TOP1={best_hybrid_top1}/{n_total} ({best_hybrid_top1/n_total*100:.1f}%)')
    print(f'  Weights: {best_weights}')

    return best_weights or {'cfm': 0.4, 'sqrt': 0.2, 'dtb': 0.2, 'frag': 0.1, 'nl': 0.1}


# ============================================================
# Evaluate 529 features
# ============================================================
def evaluate_529(hybrid_weights):
    print('\n' + '='*100)
    print('  PHASE 2: Evaluate 529 Features')
    print(f'  Workers: {N_WORKERS}')
    print('='*100)

    df = pd.read_csv(BASE / 'candidates_529.csv')
    smiles_map = pd.read_csv(BASE / 'smiles_mapping.csv')
    smi_to_cand = dict(zip(smiles_map['dtb_smiles'], smiles_map['cand_id']))
    df['cand_id'] = df['dtb_smiles'].map(smi_to_cand)

    exp_spectra, exp_meta = parse_mgf(MGF_529)
    print(f'  Features: {df["feature_id"].nunique()}, Spectra: {len(exp_spectra)}')

    # Find DTB common peaks in 529 spectra
    common_529 = find_common_peaks(exp_spectra, min_freq=0.3)
    print(f'  Common peaks (>30%): {len(common_529)} m/z values')
    print(f'  Top common: {common_529[:15]}')

    all_results = []

    for model in MODELS:
        model_label = MODEL_LABELS[model]
        pred_dir = BASE / model / 'predictions'
        print(f'\n  === {model_label} ===')

        # Prepare parallel tasks
        tasks = []
        for feat_id in sorted(df['feature_id'].unique()):
            sub = df[df['feature_id'] == feat_id]
            exp_spec = exp_spectra.get(str(feat_id))
            if not exp_spec:
                continue
            meta = exp_meta.get(str(feat_id), {})
            precursor_mz = meta.get('pepmass', sub['feature_mz'].iloc[0])
            sub_rows = sub.to_dict('records')
            tasks.append((feat_id, sub_rows, exp_spec, precursor_mz,
                          common_529, str(pred_dir), model, model_label))

        print(f'    Scoring {len(tasks)} features with {N_WORKERS} workers...')

        with Pool(N_WORKERS) as pool:
            results = pool.map(_score_one_feature_529, tasks)

        for feat_id, feat_mz, feat_rt, n_cands, cand_results, mlabel in results:
            if not cand_results:
                continue

            # Compute hybrid scores
            for c in cand_results:
                c['hybrid_score'] = (
                    hybrid_weights['cfm'] * c['cfm_score'] +
                    hybrid_weights['sqrt'] * c['sqrt_score'] +
                    hybrid_weights['dtb'] * c['dtb_removed_score'] +
                    hybrid_weights['frag'] * c['frag_intensity_ratio'] +
                    hybrid_weights['nl'] * c['nl_score']
                )

            ranked = sorted(cand_results, key=lambda x: x['hybrid_score'], reverse=True)

            for rank_i, c in enumerate(ranked[:5]):
                all_results.append({
                    'model': mlabel,
                    'feature_id': feat_id,
                    'feature_mz': round(feat_mz, 5) if isinstance(feat_mz, (int, float)) else feat_mz,
                    'feature_rt_min': round(feat_rt, 2) if isinstance(feat_rt, (int, float)) and not (isinstance(feat_rt, float) and math.isnan(feat_rt)) else '',
                    'n_candidates': n_cands,
                    'rank': rank_i + 1,
                    'cand_smiles': c['cand_smi'],
                    'cand_hmdb': c['cand_hmdb'],
                    'cand_name': c['cand_name'],
                    'hybrid_score': round(c['hybrid_score'], 6),
                    'cfm_score': round(c['cfm_score'], 6),
                    'cfm_cos': round(c['cfm_cos'], 4),
                    'cfm_iw_dice': round(c['cfm_iw_dice'], 4),
                    'cfm_ent': round(c['cfm_ent'], 4),
                    'sqrt_score': round(c['sqrt_score'], 6),
                    'sqrt_cos': round(c['sqrt_cos'], 4),
                    'dtb_removed_score': round(c['dtb_removed_score'], 6),
                    'dtb_removed_cos': round(c['dtb_removed_cos'], 4),
                    'frag_peak_ratio': round(c['frag_peak_ratio'], 4),
                    'frag_intensity_ratio': round(c['frag_intensity_ratio'], 4),
                    'nl_score': round(c['nl_score'], 4),
                })

        print(f'    Done: {len([r for r in all_results if r["model"] == mlabel])} results')

    result_df = pd.DataFrame(all_results)
    top1_df = result_df[result_df['rank'] == 1].copy()

    # Save
    out_xlsx = BASE / 'comprehensive_529_results.xlsx'
    with pd.ExcelWriter(str(out_xlsx), engine='openpyxl') as writer:
        # Per-model TOP1
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = top1_df[top1_df['model'] == label].drop(columns=['model'])
            mdf.to_excel(writer, sheet_name=model[:31], index=False)

        # Model comparison
        comparison = []
        for feat_id in sorted(top1_df['feature_id'].unique()):
            row = {'feature_id': feat_id}
            feat_sub = top1_df[top1_df['feature_id'] == feat_id]
            row['feature_mz'] = feat_sub['feature_mz'].iloc[0]
            row['n_candidates'] = feat_sub['n_candidates'].iloc[0]

            smiles_list = []
            for model in MODELS:
                label = MODEL_LABELS[model]
                msub = feat_sub[feat_sub['model'] == label]
                short = model.replace('cfm_', '').replace('param_', '')
                if len(msub) > 0:
                    r = msub.iloc[0]
                    row[f'{short}_name'] = r['cand_name']
                    row[f'{short}_hybrid'] = r['hybrid_score']
                    row[f'{short}_cfm'] = r['cfm_score']
                    row[f'{short}_frag'] = r['frag_intensity_ratio']
                    smiles_list.append(r['cand_smiles'])

            row['consensus'] = 'YES' if len(set(smiles_list)) == 1 and len(smiles_list) == 3 else \
                               'PARTIAL' if len(set(smiles_list)) == 2 else 'NO'
            comparison.append(row)

        comp_df = pd.DataFrame(comparison)
        comp_df.to_excel(writer, sheet_name='Model Comparison', index=False)

        # Summary
        summary = []
        for model in MODELS:
            label = MODEL_LABELS[model]
            mdf = top1_df[top1_df['model'] == label]
            summary.append({
                'Model': label,
                'N Features': len(mdf),
                'Mean Hybrid Score': round(mdf['hybrid_score'].mean(), 4),
                'Mean CFM Score': round(mdf['cfm_score'].mean(), 4),
                'Mean CFM Cosine': round(mdf['cfm_cos'].mean(), 4),
                'Mean sqrt Cosine': round(mdf['sqrt_cos'].mean(), 4),
                'Mean DTB-rem Cosine': round(mdf['dtb_removed_cos'].mean(), 4),
                'Mean Frag Ratio': round(mdf['frag_intensity_ratio'].mean(), 4),
                'Mean NL Score': round(mdf['nl_score'].mean(), 4),
            })
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

        # Consensus
        n_total = len(comp_df)
        n_yes = (comp_df['consensus'] == 'YES').sum()
        n_partial = (comp_df['consensus'] == 'PARTIAL').sum()
        n_no = (comp_df['consensus'] == 'NO').sum()
        cons = pd.DataFrame([{
            'Total': n_total, 'Full (3/3)': n_yes,
            'Partial (2/3)': n_partial, 'None': n_no,
            'Consensus %': round(n_yes/n_total*100, 1),
            'Hybrid Weights': str(hybrid_weights),
        }])
        cons.to_excel(writer, sheet_name='Consensus', index=False)

        # All top-5
        result_df.to_excel(writer, sheet_name='All Top5', index=False)

    print(f'\n  Saved: {out_xlsx}')

    # Print summary
    print(f'\n  {"="*90}')
    print(f'  Hybrid weights: {hybrid_weights}')
    print(f'  {"="*90}')
    for model in MODELS:
        label = MODEL_LABELS[model]
        mdf = top1_df[top1_df['model'] == label]
        print(f'\n  {label}:')
        print(f'    Hybrid: {mdf["hybrid_score"].mean():.4f} | '
              f'CFM: {mdf["cfm_score"].mean():.4f} | '
              f'CFM-cos: {mdf["cfm_cos"].mean():.4f} | '
              f'sqrt-cos: {mdf["sqrt_cos"].mean():.4f} | '
              f'Frag: {mdf["frag_intensity_ratio"].mean():.4f} | '
              f'NL: {mdf["nl_score"].mean():.4f}')

    n_total = len(comp_df)
    n_yes = (comp_df['consensus'] == 'YES').sum()
    print(f'\n  Consensus: {n_yes}/{n_total} ({n_yes/n_total*100:.1f}%)')


# ============================================================
def main():
    print('='*100)
    print('  Comprehensive Multi-Method Scoring')
    print('  Methods: CFM-ID, DTB-removed, sqrt-transform, RDKit fragments, Neutral loss')
    print(f'  RDKit: {"Available" if HAS_RDKIT else "NOT available"}')
    print('='*100)

    # Phase 1: Validate on 143
    best_weights = validate_143()

    # Phase 2: Apply to 529
    evaluate_529(best_weights)

    print('\n  DONE.')


if __name__ == '__main__':
    main()
