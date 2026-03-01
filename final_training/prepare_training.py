#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_training.py
====================
143 DTB compound library → 5-repeat scaffold-aware 80/10/10 split
for CFM-ID transfer learning.

- Parses 143_final_library_260301.mgf
- Groups compounds by Murcko generic scaffold (RDKit)
- Creates 5 independent 80/10/10 splits (train/val/test)
- Generates CFM-ID training files for each repeat (fold_0 ~ fold_4)

Usage:
    python prepare_training.py
"""
import sys, io, os, re, shutil, random
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# RDKit for Murcko scaffold
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

# ============================================================
# Paths
# ============================================================
BASE = Path("C:/Users/Chang-Mo/Desktop/Git/260301_CFM_Final_training")
MGF_FILE = BASE / "143_final_library_260301.mgf"
OUTPUT_DIR = BASE / "final_training"

CONFIG_SRC = Path("C:/Users/Chang-Mo/Desktop/Git/archive/cfmid_v5/train/config.txt")
FEATURES_SRC = Path("C:/Users/Chang-Mo/Desktop/Git/archive/cfmid_v5/train/features.txt")
PRETRAINED_SRC = Path("C:/Users/Chang-Mo/Desktop/Git/archive/cfmid_legacy/cfm-id-code/"
                      "cfm-pretrained-models/cfmid4/[M+H]+/param_output.log")

N_REPEATS = 5
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEEDS = [42, 123, 256, 789, 1024]


# ============================================================
# Parse MGF
# ============================================================
def parse_mgf(path):
    """Parse 143 library MGF → list of compound dicts."""
    compounds = []
    with open(path, 'r', encoding='utf-8') as f:
        current = {}
        peaks = []
        in_entry = False
        for line in f:
            line = line.rstrip('\n')
            stripped = line.strip()

            if stripped == 'BEGIN IONS':
                current = {}
                peaks = []
                in_entry = True
                continue

            if stripped == 'END IONS':
                if current and peaks:
                    current['peaks'] = peaks
                    compounds.append(current)
                in_entry = False
                continue

            if not in_entry:
                continue

            # Metadata lines
            if '=' in stripped and not stripped[0].isdigit():
                key, val = stripped.split('=', 1)
                current[key] = val
            # Peak lines (tab or space separated)
            elif stripped and stripped[0].isdigit():
                parts = re.split(r'[\t ]+', stripped)
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    return compounds


# ============================================================
# Scaffold grouping
# ============================================================
def get_generic_scaffold(smiles):
    """Return generic Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'UNKNOWN'
    try:
        scaf = GetScaffoldForMol(mol)
        generic = MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(generic)
    except Exception:
        return 'UNKNOWN'


def group_by_scaffold(compounds):
    """Group compound indices by generic scaffold.
    Large groups (>20% of total) are subdivided into chunks of ~10
    to allow proper distribution across train/val/test.
    """
    raw_map = defaultdict(list)
    for i, comp in enumerate(compounds):
        smiles = comp.get('SMILES', '')
        scaffold = get_generic_scaffold(smiles)
        raw_map[scaffold].append(i)

    # Subdivide large groups (max ~10 to allow precise 80/10/10 distribution)
    max_group_size = 10
    scaffold_map = {}
    for scaffold, indices in raw_map.items():
        if len(indices) <= max_group_size:
            scaffold_map[scaffold] = indices
        else:
            # Split into subgroups of ~max_group_size
            for chunk_i in range(0, len(indices), max_group_size):
                chunk = indices[chunk_i:chunk_i + max_group_size]
                sub_key = f'{scaffold}__sub{chunk_i // max_group_size}'
                scaffold_map[sub_key] = chunk

    return scaffold_map


# ============================================================
# Scaffold-aware splitting
# ============================================================
def scaffold_split(compounds, scaffold_map, seed, train_ratio=0.80, val_ratio=0.10):
    """
    Split compounds into train/val/test by scaffold groups.
    Entire scaffold groups go to same partition.
    Uses greedy bin-packing to achieve target 80/10/10 ratio.
    """
    rng = random.Random(seed)
    n_total = len(compounds)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Shuffle scaffold groups
    groups = list(scaffold_map.items())
    rng.shuffle(groups)

    train_idx, val_idx, test_idx = [], [], []

    for scaffold, indices in groups:
        # Greedy assignment: add to the partition most below its target
        train_gap = n_train - len(train_idx)
        val_gap = n_val - len(val_idx)
        test_gap = n_test - len(test_idx)

        # Assign to partition with largest remaining gap
        if train_gap >= val_gap and train_gap >= test_gap:
            train_idx.extend(indices)
        elif val_gap >= test_gap:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return train_idx, val_idx, test_idx


# ============================================================
# Write CFM-ID training files
# ============================================================
def write_spectra_file(compound, filepath):
    """Write a single compound's spectrum in CFM-ID 3-energy format."""
    peaks = compound['peaks']
    with open(filepath, 'w', encoding='utf-8') as f:
        for energy in ['energy0', 'energy1', 'energy2']:
            f.write(f'{energy}\n')
            for mz, intensity in peaks:
                f.write(f'{mz:.4f} {intensity:.2f}\n')


def make_compound_id(compound):
    """Generate CFM-ID compound ID from MGF entry: HMDB_mz_logp"""
    name = compound.get('NAME', '')
    # NAME format: HMDB0004825_350.20743_12.3
    return name


def write_input_molecules(compounds, train_idx, val_idx, filepath):
    """Write input_molecules.txt with group assignments."""
    entries = []
    for idx in train_idx:
        comp = compounds[idx]
        cid = make_compound_id(comp)
        smiles = comp.get('SMILES', '')
        entries.append((cid, smiles, 0))  # group=0: train

    for idx in val_idx:
        comp = compounds[idx]
        cid = make_compound_id(comp)
        smiles = comp.get('SMILES', '')
        entries.append((cid, smiles, 1))  # group=1: validation

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'{len(entries)}\n')
        for i, (cid, smiles, group) in enumerate(entries):
            f.write(f'{cid} {smiles} {group}\n')

    return entries


def write_test_list(compounds, test_idx, filepath):
    """Write test compound list for post-training evaluation."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'# Test compounds for evaluation (NOT used during training)\n')
        f.write(f'# compound_id\tsmiles\tpepmass\n')
        for idx in test_idx:
            comp = compounds[idx]
            cid = make_compound_id(comp)
            smiles = comp.get('SMILES', '')
            pepmass = comp.get('PEPMASS', '')
            f.write(f'{cid}\t{smiles}\t{pepmass}\n')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 70)
    print('  CFM-ID Final Training: 143 Compound 5-Repeat Scaffold Split')
    print('=' * 70)

    # 1. Parse MGF
    compounds = parse_mgf(MGF_FILE)
    print(f'\nParsed: {len(compounds)} compounds from {MGF_FILE.name}')

    # 2. Scaffold grouping
    scaffold_map = group_by_scaffold(compounds)
    print(f'Scaffold groups: {len(scaffold_map)}')
    for scaf, indices in sorted(scaffold_map.items(), key=lambda x: -len(x[1])):
        names = [make_compound_id(compounds[i]).split('_')[0] for i in indices[:3]]
        more = f' +{len(indices)-3} more' if len(indices) > 3 else ''
        print(f'  {scaf[:60]:60s} ({len(indices):3d}) e.g. {", ".join(names)}{more}')

    # 3. Copy shared files
    config_dst = OUTPUT_DIR / 'config.txt'
    features_dst = OUTPUT_DIR / 'features.txt'

    if CONFIG_SRC.exists():
        shutil.copy2(CONFIG_SRC, config_dst)
        print(f'\nCopied config: {config_dst}')
    else:
        print(f'\nWARNING: config source not found: {CONFIG_SRC}')

    if FEATURES_SRC.exists():
        shutil.copy2(FEATURES_SRC, features_dst)
        print(f'Copied features: {features_dst}')
    else:
        print(f'WARNING: features source not found: {FEATURES_SRC}')

    # 4. Create 5 repeats
    print(f'\n{"=" * 70}')
    print(f'  Creating {N_REPEATS} repeats (80/10/10 scaffold split)')
    print(f'{"=" * 70}')

    all_splits = []
    for repeat in range(N_REPEATS):
        seed = SEEDS[repeat]
        fold_dir = OUTPUT_DIR / f'fold_{repeat}'

        # Clean up previous run's output directories to avoid file accumulation
        for subdir_name in ['spectra', 'test_spectra']:
            subdir = fold_dir / subdir_name
            if subdir.exists():
                shutil.rmtree(subdir)

        fold_dir.mkdir(parents=True, exist_ok=True)
        spectra_dir = fold_dir / 'spectra'
        spectra_dir.mkdir(exist_ok=True)
        pretrained_dir = fold_dir / 'pretrained'
        pretrained_dir.mkdir(exist_ok=True)

        # Split
        train_idx, val_idx, test_idx = scaffold_split(
            compounds, scaffold_map, seed, TRAIN_RATIO, VAL_RATIO)

        all_splits.append({
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx),
        })

        print(f'\n  Repeat {repeat} (seed={seed}):')
        print(f'    Train: {len(train_idx)} ({len(train_idx)/len(compounds)*100:.1f}%)')
        print(f'    Val:   {len(val_idx)} ({len(val_idx)/len(compounds)*100:.1f}%)')
        print(f'    Test:  {len(test_idx)} ({len(test_idx)/len(compounds)*100:.1f}%)')

        # Write input_molecules.txt (train + val only)
        entries = write_input_molecules(
            compounds, train_idx, val_idx,
            fold_dir / 'input_molecules.txt')
        print(f'    input_molecules.txt: {len(entries)} entries')

        # Write spectra for train + val
        n_spectra = 0
        for idx in train_idx + val_idx:
            comp = compounds[idx]
            cid = make_compound_id(comp)
            spec_file = spectra_dir / f'{cid}.txt'
            write_spectra_file(comp, spec_file)
            n_spectra += 1
        print(f'    spectra/: {n_spectra} files')

        # Write test compound list
        write_test_list(compounds, test_idx, fold_dir / 'test_compounds.txt')

        # Write test spectra (for evaluation, separate directory)
        test_spectra_dir = fold_dir / 'test_spectra'
        test_spectra_dir.mkdir(exist_ok=True)
        for idx in test_idx:
            comp = compounds[idx]
            cid = make_compound_id(comp)
            write_spectra_file(comp, test_spectra_dir / f'{cid}.txt')

        # Copy config and features
        shutil.copy2(config_dst, fold_dir / 'config.txt')
        shutil.copy2(features_dst, fold_dir / 'features.txt')

        # Copy pretrained model
        if PRETRAINED_SRC.exists():
            shutil.copy2(PRETRAINED_SRC, pretrained_dir / 'param_output.log')
        else:
            print(f'    WARNING: pretrained model not found: {PRETRAINED_SRC}')

    # 5. Write split summary
    print(f'\n{"=" * 70}')
    print(f'  Split Summary')
    print(f'{"=" * 70}')
    print(f'  {"Repeat":8s} {"Train":8s} {"Val":8s} {"Test":8s} {"Total":8s}')
    for i, s in enumerate(all_splits):
        total = s['train'] + s['val'] + s['test']
        print(f'  {i:8d} {s["train"]:8d} {s["val"]:8d} {s["test"]:8d} {total:8d}')

    # Save split info as CSV
    with open(OUTPUT_DIR / 'split_summary.csv', 'w', encoding='utf-8') as f:
        f.write('repeat,seed,train,val,test,total\n')
        for i, s in enumerate(all_splits):
            total = s['train'] + s['val'] + s['test']
            f.write(f'{i},{SEEDS[i]},{s["train"]},{s["val"]},{s["test"]},{total}\n')

    print(f'\nOutput: {OUTPUT_DIR}')
    print(f'  fold_0/ ~ fold_4/: training files per repeat')
    print(f'  config.txt, features.txt: shared configs')
    print(f'  split_summary.csv: split statistics')
    print(f'\nReady to train: bash run_5fold.sh')


if __name__ == '__main__':
    main()
