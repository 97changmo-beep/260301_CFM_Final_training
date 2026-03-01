# CFM-ID Final Transfer Learning: 143 DTB Compounds

## Overview

Transfer learning of CFM-ID 4.0 [M+H]+ model for DTB-derivatized metabolite MS/MS spectrum prediction.

**Pretrained model**: CFM-ID 4.0 default [M+H]+ (trained on ~4,055 METLIN metabolites)
**Fine-tuning data**: 143 DTB-derivatized endogenous metabolites with experimental Orbitrap MS/MS
**Evaluation**: 5-repeat scaffold-aware 80/10/10 split (MolPMoFiT-style)

## v5 vs Final

| Item | v5 (previous) | Final (current) |
|------|---------------|-----------------|
| Compounds | 121 | 143 (+22 new) |
| Split | Single hold-out (107T/14V) | 5-repeat 80/10/10 scaffold |
| Evaluation | 1 split | 5 repeats (mean +/- std) |
| Split method | Random group assignment | Scaffold-aware (MoleculeNet) |
| Pretrained | CFM-ID 4.0 default | CFM-ID 4.0 default (same) |
| v5 weights | N/A | NOT used (clean transfer) |

## Dataset Split Strategy

### 80/10/10 Scaffold-Aware Split

Following MoleculeNet (Wu et al., Chemical Science 2018) and MolPMoFiT (Li & Fourches, J Cheminform 2020):

```
143 compounds -> 5 independent splits (different random seeds)

Each repeat:
  Train (80%):  115 compounds (group=0) -> actual learning
  Val (10%):     14 compounds (group=1) -> CFM-ID early stopping
  Test (10%):    14 compounds           -> post-training evaluation only

Total in input_molecules.txt: 129 (train + val)
Test: NOT included in training files
```

### Scaffold Splitting

- RDKit Murcko generic scaffold used for grouping
- All compounds sharing the same backbone scaffold are assigned to the same partition
- Prevents information leakage between structurally similar compounds
- More challenging than random split (reflects real-world prediction scenarios)

### Literature References

| Reference | Split | Method |
|-----------|-------|--------|
| MoleculeNet (Wu 2018) | 80/10/10 | Scaffold split (Bemis-Murcko) |
| MolPMoFiT (Li 2020) | 80/10/10 | Scaffold + random (10 repeats) |
| MolPROP (2024) | 80/10/10 | Scaffold split |
| CFM-ID 3.0 (Wang 2019) | 80/20 | 5-fold CV |
| CFM-ID 4.0 (Wang 2022) | 90/10 | 10-fold CV |

## Training Configuration

### Hyperparameters (from v5 config, optimized for DTB TL)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 0.0001 -> 0.00005 | 10x lower than default (TL) |
| Lambda | 0.0 | No L2 regularization |
| EM max iterations | 100 | Allow full convergence |
| EM no-progress | 3 | Patience before early stop |
| GA max iterations | 20 | Converges in 10-20 steps |
| Minibatch nth | 2 | ~50% per batch (115 compounds) |
| NN architecture | 128-128-1 | Same as default CFM-ID |
| Dropout | 0.1 / 0.1 / 0 | Same as default |
| Optimizer | Adam (beta1=0.9, beta2=0.999) | |

### Transfer Learning Approach

```
CFM-ID 4.0 default [M+H]+  ----(-w flag)---->  DTB-specific model
  (376,707 parameters)                          (same architecture)
  (trained on METLIN Q-TOF)                     (fine-tuned on Orbitrap)
```

Key: the `-w` flag loads pretrained weights, allowing the model to start from a good initialization and fine-tune on the DTB-specific domain.

## OpenMP Parallelization

All 5 repeats use the same OpenMP-optimized Docker image (`cfmid-final`):

| Setting | Value |
|---------|-------|
| OMP_NUM_THREADS | 32 (auto-detected) |
| OMP_PROC_BIND | close |
| OMP_PLACES | threads/cores (HT-aware) |
| OMP_SCHEDULE | dynamic |
| OMP_STACKSIZE | 8M |
| LD_PRELOAD | mimalloc (multi-threaded allocator) |

OpenMP patches applied to CFM-ID source:
- `EmModel.cpp`: Parallel E/M-step computation
- `NNParam.cpp`: Thread-safe neural network parameters
- `cfm-train-main.cpp`: Parallel fragment graph generation

Expected speedup: 3-6x vs single-threaded (8+ cores)

## File Structure

```
final_training/
  prepare_training.py       # Data preparation + scaffold split
  run_5fold.sh              # Master training script (OpenMP)
  evaluate_5fold.py         # Cross-fold evaluation
  config.txt                # Training hyperparameters
  features.txt              # 6 molecular features
  split_summary.csv         # Split statistics
  README_final_training.md  # This file

  fold_0/                   # Repeat 0 (seed=42)
    input_molecules.txt     # 129 compounds (115 train + 14 val)
    spectra/                # 129 spectrum files (3-energy format)
    test_compounds.txt      # 14 test compound IDs
    test_spectra/           # 14 experimental test spectra
    config.txt              # Copy of training config
    features.txt            # Copy of features
    pretrained/
      param_output.log      # CFM-ID 4.0 default [M+H]+
    param_output.log        # [OUTPUT] trained model weights
    training.log            # [OUTPUT] Docker training log
    predictions/            # [OUTPUT] cfm-predict results
    baseline_predictions/   # [OUTPUT] default model predictions
  fold_1/ ... fold_4/       # Same structure for repeats 1-4
```

## How to Run

### Step 1: Prepare training data

```bash
cd C:/Users/Chang-Mo/Desktop/Git/260301_CFM_Final_training/final_training
python prepare_training.py
```

This creates fold_0 through fold_4 with all required files.

### Step 2: Run training (all 5 folds)

```bash
# Dry run first (see commands without executing)
bash run_5fold.sh --dry-run

# Run with all available cores
bash run_5fold.sh

# Or specify thread count
bash run_5fold.sh 32
```

Each fold takes approximately 6-10 hours on 32 threads.
Total: ~30-50 hours for all 5 folds (sequential).

### Step 3: Evaluate results

```bash
python evaluate_5fold.py
```

Outputs:
- `evaluation_results.csv`: Per-compound cosine scores (all folds)
- `evaluation_results.xlsx`: Same in Excel format
- Console: mean +/- std across folds, Wilcoxon p-values

## Expected Results

Based on v5 (121 compounds, single split):

| Energy | Default | v5 (121) | Final (143, expected) |
|--------|---------|----------|----------------------|
| e0 | 0.652 | 0.800 | ~0.80-0.85 |
| e1 | 0.470 | 0.816 | ~0.80-0.85 |
| e2 | 0.124 | 0.745 | ~0.70-0.80 |

Final training with 143 compounds (18% more data) may show slight improvement over v5.
Scaffold split is more challenging than random split, so test set cosine may be slightly lower.

## Evaluation Metrics

- **Cosine similarity**: Standard spectral similarity metric (0-1)
- **Per-energy reporting**: energy0 (10eV), energy1 (20eV), energy2 (40eV)
- **Statistical test**: Wilcoxon signed-rank (paired, non-parametric)
- **Baseline comparison**: Default CFM-ID vs fine-tuned, same test set

## 143 Compound Library

Source: `143_final_library_260301.mgf`

Compound classes include:
- Amino acids & derivatives (Ala, Ile, Leu, Phe, Trp, Tyr, ...)
- Polyamines (Putrescine, Spermidine, Spermine, Cadaverine)
- Aromatic metabolites (Histamine, Serotonin, Dopamine, Tyramine, ...)
- Peptide conjugates (Carnosine, Glutathione, gamma-glutamyl-*)
- Other DTB-reactive metabolites

All compounds are DTB-derivatized (desthiobiotin, +196.12 Da) [M+H]+ mode.
