#!/usr/bin/env python3.10
"""
Generate DTB-modified di/tripeptide SMILES for CFM-ID prediction.

DTB modification: Dehydration condensation (탈수축합반응) at primary amines.
Target sites:
  - N-terminal amine (NOT Proline, which has secondary amine)
  - Lysine epsilon-amine (side chain -NH2)

Naming convention:
  - * marks the modified amino acid position
  - e.g., G*GG = DTB at N-terminal Gly (pos 1)
  - e.g., GK*G = DTB at Lys epsilon (pos 2)
  - e.g., G*K*G = DTB at both N-term and Lys (but we generate single-site only)

Output:
  - peptide_dtb_candidates.csv with columns: name, smiles, mass, site, peptide_sequence
  - batch_chunk_*.txt files for cfm-predict
"""

import csv
import os
import sys
from itertools import product
from multiprocessing import Pool, cpu_count

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

# ─── Constants ───
# DTB acid: desthiobiotin
# C[C@H]1NC(=O)N[C@H]1CCCCCC(=O)O
# DTB acyl (for amide bond formation - loses OH, amine loses H):
DTB_ACYL = "C[C@H]1NC(=O)N[C@H]1CCCCCC(=O)"

# 20 standard amino acids: (1-letter, 3-letter, side_chain_SMILES)
# Side chain is what hangs off the alpha carbon (Ca)
# Backbone: N-Ca(R)-C(=O)-  where R is the side chain
AMINO_ACIDS = {
    'G': ('Gly', ''),           # H (no side chain beyond H)
    'A': ('Ala', 'C'),
    'V': ('Val', 'C(C)C'),
    'L': ('Leu', 'CC(C)C'),
    'I': ('Ile', '[C@@H](CC)C'),
    'P': ('Pro', None),         # Special: cyclic, secondary amine at N-term
    'F': ('Phe', 'Cc1ccccc1'),
    'W': ('Trp', 'Cc1c[nH]c2ccccc12'),
    'M': ('Met', 'CCSC'),
    'S': ('Ser', 'CO'),
    'T': ('Thr', '[C@@H](O)C'),
    'C': ('Cys', 'CS'),
    'Y': ('Tyr', 'Cc1ccc(O)cc1'),
    'H': ('His', 'Cc1c[nH]cn1'),
    'D': ('Asp', 'CC(=O)O'),
    'E': ('Glu', 'CCC(=O)O'),
    'N': ('Asn', 'CC(=O)N'),    # Amide NH2 - NOT reactive
    'Q': ('Gln', 'CCC(=O)N'),   # Amide NH2 - NOT reactive
    'K': ('Lys', 'CCCCN'),      # Epsilon-amine - REACTIVE
    'R': ('Arg', 'CCCNC(=N)N'), # Guanidinium - NOT reactive
}

AA_LIST = list(AMINO_ACIDS.keys())


def build_peptide_smiles(sequence):
    """
    Build a linear peptide SMILES from amino acid sequence.
    Returns (smiles, n_term_is_primary, lys_positions).

    For Proline at N-terminus, n_term_is_primary = False.
    lys_positions = list of 0-indexed positions where Lys occurs.
    """
    # Build peptide backbone: H2N-CH(R1)-CO-NH-CH(R2)-CO-...-OH
    # We'll construct SMILES piece by piece

    n = len(sequence)
    if n < 2:
        return None, False, []

    parts = []
    lys_positions = []
    n_term_is_primary = (sequence[0] != 'P')

    for i, aa in enumerate(sequence):
        info = AMINO_ACIDS[aa]
        side_chain = info[1]

        if aa == 'P':
            # Proline: pyrrolidine ring
            # In peptide bond context: -N1CCCC1C(=O)- (simplified)
            if i == 0:
                # N-terminal Pro: secondary amine
                parts.append('O=C([C@@H]1CCCN1')
            else:
                # Internal/C-terminal Pro
                parts.append('N1CCC[C@@H]1C(=O)')
        elif aa == 'G':
            # Glycine: no chiral center
            if i == 0:
                parts.append('O=C(CN')
            else:
                parts.append('NCC(=O)')
        else:
            if i == 0:
                parts.append(f'O=C([C@@H]({side_chain})N')
            else:
                parts.append(f'N[C@@H]({side_chain})C(=O)')

        if aa == 'K':
            lys_positions.append(i)

    return sequence, n_term_is_primary, lys_positions


def make_peptide_smiles(sequence):
    """
    Build a proper peptide SMILES string from a sequence of amino acids.
    Returns canonical SMILES or None if invalid.
    """
    n = len(sequence)
    if n < 2:
        return None

    # Use RDKit to build the peptide properly
    # We'll build SMILES string manually with proper peptide bonds

    # Strategy: build the full SMILES string for H2N-aa1-aa2-...-aaN-COOH
    # then validate with RDKit

    fragments = []
    for i, aa in enumerate(sequence):
        side = AMINO_ACIDS[aa][1]

        if aa == 'P':
            if i == 0:
                # N-term Pro: H-N in ring
                frag = 'O=C([C@@H]1CCCN1'  # will close with ) or add peptide bond
            elif i == n - 1:
                # C-term Pro
                frag = 'N1CCC[C@@H]1C(=O)O'
            else:
                frag = 'N1CCC[C@@H]1C(=O)'
        elif aa == 'G':
            if i == 0:
                frag = 'O=C(CN'
            elif i == n - 1:
                frag = 'NCC(=O)O'
            else:
                frag = 'NCC(=O)'
        else:
            if i == 0:
                frag = f'O=C([C@@H]({side})N'
            elif i == n - 1:
                frag = f'N[C@@H]({side})C(=O)O'
            else:
                frag = f'N[C@@H]({side})C(=O)'

    # This approach is getting complicated. Let's use a simpler method:
    # Build SMILES from C-terminus to N-terminus

    return _build_peptide_v2(sequence)


def _build_peptide_v2(sequence):
    """
    Build peptide SMILES by constructing from scratch.
    Format: N-term ... C-term = H2N-CH(R1)-CO-NH-CH(R2)-CO-...-NH-CH(Rn)-COOH

    Written as SMILES (left to right, N to C):
    N[C@@H](R1)C(=O)N[C@@H](R2)C(=O)...N[C@@H](Rn)C(=O)O
    """
    n = len(sequence)
    parts = []

    for i, aa in enumerate(sequence):
        side = AMINO_ACIDS[aa][1]

        if aa == 'P':
            # Proline: cyclic - N is part of the pyrrolidine ring
            # Proline residue in peptide: ...C(=O)N1CCC[C@@H]1C(=O)...
            if i == 0:
                # N-terminal Pro (secondary amine)
                parts.append('C(=O)O')  # placeholder, will build differently
            else:
                parts.append(f'N1CCC[C@@H]1C(=O)')
        elif aa == 'G':
            if i == 0:
                parts.append('NCC(=O)')
            else:
                parts.append('NCC(=O)')
        else:
            if i == 0:
                parts.append(f'N[C@@H]({side})C(=O)')
            else:
                parts.append(f'N[C@@H]({side})C(=O)')

    # Handle proline at N-term specially
    if sequence[0] == 'P':
        # N-terminal Pro: ring structure
        # The full pyrrolidine: C1CC[C@H](N1)C(=O)...
        rest = ''.join(parts[1:])
        smiles = f'C1CCN[C@@H]1C(=O){rest}O'
    else:
        smiles = ''.join(parts) + 'O'

    # Validate with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol)


def apply_dtb_nterm(sequence):
    """
    Apply DTB modification at N-terminal amine.
    Returns (name, smiles) or None if N-term is Pro (secondary amine).

    DTB-NH-CH(R1)-CO-NH-...
    = C[C@H]1NC(=O)N[C@H]1CCCCCC(=O)N[C@@H](R1)C(=O)N...
    """
    if sequence[0] == 'P':
        return None  # Pro has secondary amine, not reactive

    n = len(sequence)
    parts = []

    # Start with DTB acyl connected to first amino acid's N
    first_aa = sequence[0]
    first_side = AMINO_ACIDS[first_aa][1]

    if first_aa == 'G':
        parts.append(f'{DTB_ACYL}NCC(=O)')
    else:
        parts.append(f'{DTB_ACYL}N[C@@H]({first_side})C(=O)')

    # Add remaining residues
    for i in range(1, n):
        aa = sequence[i]
        side = AMINO_ACIDS[aa][1]

        if aa == 'P':
            parts.append('N1CCC[C@@H]1C(=O)')
        elif aa == 'G':
            parts.append('NCC(=O)')
        else:
            parts.append(f'N[C@@H]({side})C(=O)')

    smiles = ''.join(parts) + 'O'

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canon = Chem.MolToSmiles(mol)

    # Name: mark first position with *
    name_parts = list(sequence)
    name_parts[0] = name_parts[0] + '*'
    name = ''.join(name_parts)

    return name, canon


def apply_dtb_lys(sequence, lys_pos):
    """
    Apply DTB modification at Lysine epsilon-amine at given position.

    Lys side chain: -CH2CH2CH2CH2NH2
    Modified:       -CH2CH2CH2CH2NH-CO-DTB
    = -CCCCNC(=O)CCCCC[C@@H]1NC(=O)N[C@@H]1C
    """
    n = len(sequence)
    parts = []

    # DTB-modified Lys side chain
    dtb_lys_side = f'CCCCNC(=O)CCCCC[C@@H]1NC(=O)N[C@@H]1C'

    for i, aa in enumerate(sequence):
        side = AMINO_ACIDS[aa][1]

        if i == lys_pos and aa == 'K':
            # Use DTB-modified Lys side chain
            if i == 0:
                parts.append(f'N[C@@H]({dtb_lys_side})C(=O)')
            else:
                parts.append(f'N[C@@H]({dtb_lys_side})C(=O)')
        elif aa == 'P':
            if i == 0:
                # N-term Pro
                rest = ''  # will handle below
                parts.append('__PROSTART__')
            else:
                parts.append('N1CCC[C@@H]1C(=O)')
        elif aa == 'G':
            if i == 0:
                parts.append('NCC(=O)')
            else:
                parts.append('NCC(=O)')
        else:
            if i == 0:
                parts.append(f'N[C@@H]({side})C(=O)')
            else:
                parts.append(f'N[C@@H]({side})C(=O)')

    # Handle N-terminal Pro
    if sequence[0] == 'P' and parts[0] == '__PROSTART__':
        rest = ''.join(parts[1:]) + 'O'
        smiles = f'C1CCN[C@@H]1C(=O){rest}'
    else:
        smiles = ''.join(parts) + 'O'

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canon = Chem.MolToSmiles(mol)

    # Name: mark Lys position with *
    name_parts = list(sequence)
    name_parts[lys_pos] = name_parts[lys_pos] + '*'
    name = ''.join(name_parts)

    return name, canon


def generate_for_sequence(sequence):
    """
    Generate all DTB-modified variants for a given peptide sequence.
    Returns list of (name, smiles, mass, site_description, sequence).
    """
    results = []
    seq_str = ''.join(sequence)

    # Find DTB modification sites
    n_term_reactive = (sequence[0] != 'P')
    lys_positions = [i for i, aa in enumerate(sequence) if aa == 'K']

    # Skip if no reactive sites
    if not n_term_reactive and not lys_positions:
        return results

    # 1. N-terminal DTB modification
    if n_term_reactive:
        result = apply_dtb_nterm(sequence)
        if result:
            name, smiles = result
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mass = Descriptors.ExactMolWt(mol)
                results.append((name, smiles, mass, 'N-term', seq_str))

    # 2. Lys epsilon-amine DTB modification
    for pos in lys_positions:
        result = apply_dtb_lys(sequence, pos)
        if result:
            name, smiles = result
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mass = Descriptors.ExactMolWt(mol)
                site = f'Lys{pos+1}-epsilon'
                results.append((name, smiles, mass, site, seq_str))

    return results


def generate_all_peptides(max_length=3):
    """Generate all dipeptide and tripeptide DTB candidates."""
    all_candidates = []
    seen_smiles = set()

    for length in range(2, max_length + 1):
        print(f"Generating {length}-mers...")
        sequences = list(product(AA_LIST, repeat=length))
        print(f"  {len(sequences)} sequences to process")

        n_workers = min(32, cpu_count())
        with Pool(n_workers) as pool:
            results_list = pool.map(generate_for_sequence, sequences)

        count = 0
        for results in results_list:
            for item in results:
                name, smiles, mass, site, seq = item
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    all_candidates.append(item)
                    count += 1

        print(f"  {count} unique DTB-peptide variants generated")

    return all_candidates


def write_batch_files(candidates, output_dir, chunk_size=500):
    """Write batch_chunk_*.txt files for cfm-predict."""
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    chunk_files = []
    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i:i+chunk_size]
        chunk_idx = i // chunk_size
        chunk_file = os.path.join(output_dir, f'batch_chunk_{chunk_idx}.txt')

        with open(chunk_file, 'w') as f:
            for j, (name, smiles, mass, site, seq) in enumerate(chunk):
                cand_id = f'pep_{i+j}'
                f.write(f'{cand_id} {smiles}\n')

        chunk_files.append(chunk_file)

    return chunk_files


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    PROJECT = os.path.dirname(BASE)
    OUTPUT_DIR = os.path.join(PROJECT, '05_evaluation', 'peptide_dtb')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  DTB-Modified Peptide Candidate Generation")
    print("  Di-peptides (20x20=400) + Tri-peptides (20^3=8000)")
    print("=" * 60)

    # Generate all candidates
    candidates = generate_all_peptides(max_length=3)

    print(f"\nTotal unique DTB-peptide candidates: {len(candidates)}")

    # Count by type
    nterm_count = sum(1 for c in candidates if c[3] == 'N-term')
    lys_count = sum(1 for c in candidates if 'Lys' in c[3])
    di_count = sum(1 for c in candidates if len(c[4]) == 2)
    tri_count = sum(1 for c in candidates if len(c[4]) == 3)

    print(f"  Dipeptides: {di_count}")
    print(f"  Tripeptides: {tri_count}")
    print(f"  N-terminal modifications: {nterm_count}")
    print(f"  Lys epsilon modifications: {lys_count}")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'peptide_dtb_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'name', 'smiles', 'exact_mass', 'mh_plus', 'site', 'sequence'])
        for i, (name, smiles, mass, site, seq) in enumerate(candidates):
            mh = mass + 1.007276
            writer.writerow([f'pep_{i}', name, smiles, f'{mass:.6f}', f'{mh:.6f}', site, seq])

    print(f"\nCSV written: {csv_path}")

    # Write batch files for cfm-predict
    chunk_files = write_batch_files(candidates, OUTPUT_DIR, chunk_size=500)
    print(f"Batch files written: {len(chunk_files)} chunks")

    # Mass range summary
    masses = [c[2] for c in candidates]
    print(f"\nMass range: {min(masses):.2f} - {max(masses):.2f} Da")
    print(f"[M+H]+ range: {min(masses)+1.007:.2f} - {max(masses)+1.007:.2f} Da")

    # Show some examples
    print(f"\nFirst 10 candidates:")
    print(f"{'Name':<12} {'Site':<15} {'Mass':>10} {'SMILES'}")
    print("-" * 80)
    for name, smiles, mass, site, seq in candidates[:10]:
        print(f"{name:<12} {site:<15} {mass:>10.4f} {smiles[:60]}")

    print(f"\nDone! Output in: {OUTPUT_DIR}")
    return candidates


if __name__ == '__main__':
    main()
