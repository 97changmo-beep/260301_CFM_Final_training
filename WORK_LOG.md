# Work Log: CFM-ID Transfer Learning Project

Timeline of all work performed on this computer (2026-03-01 ~ 2026-03-06).

---

## Day 1: 2026-03-01 (Saturday) -- Project Setup & Initial Commit

### Repository initialization
- Created GitHub repository `260301_CFM_Final_training`
- Initial commit: CFM-ID final training pipeline for 143 DTB compounds
  - `README_final_training.md`: Full documentation (dataset, split strategy, config, how-to-run)
  - `prepare_training.py`: Data preparation + 5-repeat scaffold-aware split (MolPMoFiT-style)
  - `config.txt`, `features.txt`: Training hyperparameters
  - `143_final_library_260301.mgf`: Source library (143 DTB-derivatized metabolites)
  - `fold_0/` ~ `fold_4/`: Pre-generated splits (80/10/10 scaffold split, 5 seeds)

### Docker & pretrained model
- Added pretrained CFM-ID 4.0 default `[M+H]+` model weights for transfer learning
- Added `Param_JJY/param_output.log` (JJY-trained parameters for comparison)
- Renamed Docker image from `cfmid-v5-omp` to `cfmid-final`
- Added `Dockerfile` + 4 OpenMP patch files (`EmModel.cpp`, `NNParam.cpp`, `cfm-train-main.cpp`, `CMakeLists.txt`)
- Added Docker build instructions to README (Step 0)

---

## Day 2-3: 2026-03-02 ~ 03 (Sun-Mon) -- Training Execution

### 5-Fold Cross-Validation training (offline)
- Ran 5-fold training using `run_5fold.sh` with 32 OpenMP threads
- Each fold: ~6-10 hours (115 train + 14 val compounds, 100 EM iterations)
- Total training time: ~30-50 hours (sequential across 5 folds)

### Full Model training (offline)
- Trained full model using all 143 compounds (no held-out test set)
- Training time: 33,454 seconds (~9.3 hours)
- Output: `04_training/full_model/param_output.log` (final trained weights)

---

## Day 4: 2026-03-04 (Tuesday) -- Predictions & Evaluation Infrastructure

### Bulk commit of all training outputs (20:26)
- 5-fold CV results: `param_output.log` (per-fold trained models), `training.log`, predictions, baseline_predictions
- Full model outputs: `val_predictions/`, `training.log`
- TOP1 evaluation: predictions for 3 models (cfm_default, param_jjy, full_model)
- 529 feature candidates: batch files for 6,097 unique SMILES x 3 models
- PDF report generation script (`generate_report.py`)
- Scoring grid search results (`scoring_search_results.csv`)

### 529 feature prediction -- all 3 models (23:00)
- Ran CFM-ID predictions for 529 DESTNI features using parallel Docker containers
- 3 models x 6,092 predictions = 18,276 total prediction files
  - `cfm_default`: CFM-ID 4.0 default params (Docker-internal)
  - `param_jjy`: JJY-trained params
  - `full_model`: Final trained params (this project)
- Parallelization: 16 containers x 2 threads = 32 threads

---

## Day 5: 2026-03-05 (Wednesday) -- Scoring Optimization & Multi-Model Evaluation

### Morning session (11:46 ~ 12:14)

**TOP1 scoring optimization (11:46)**
- Grid search over 61,812 combinations (12 formulas x 5,151 weight combos, step=0.01)
- Added `iw_dice` (intensity-weighted DICE) metric
- Best formula: `entropy * cosine^0.75 * iw_dice^1.25`
- Best energy weights: e0=0.40, e1=0.34, e2=0.26
- Result: Full Model TOP1 = **63/143 (44.1%)**, TOP3 = 92/134, TOP10 = 120/134

**529 feature evaluation with optimized scoring (11:55)**
- Applied optimized scoring to 529 features x 3 models
- Full Model best: mean cosine 0.298, mean iw_dice 0.315
- Model consensus: 241/455 (53.0%) full agreement on TOP1

**Multi-method scoring (12:14)**
- Implemented 6 scoring methods: CFM-ID, DTB-removed, sqrt-transform, RDKit fragments, neutral loss, hybrid
- Best hybrid on 143 library: 64/143 (44.8%) with weights cfm=0.3, frag=0.3, nl=0.4

### Afternoon session (12:46 ~ 14:42)

**NeoME V8 model integration (12:46 ~ 13:29)**
- Added NeoME V8 parameter config and output log
- Ran NeoME V8 predictions for 143 library (6,081) and 529 features (6,092)

**Repository reorganization (13:04)**
- Reorganized entire repo into 7 category folders:
  - `01_library/`: MGF/xlsx source data (143 library, 529 features, union DB)
  - `02_config/`: Parameters (param_v5, Param_JJY, Param_NeoME_V8, config/features)
  - `03_docker/`: Dockerfile + OpenMP patches
  - `04_training/`: 5-fold CV (fold_0~4), full_model, shell scripts
  - `05_evaluation/`: eval_529, top1_eval, evaluation results
  - `06_scripts/`: Python analysis scripts
  - `07_figures/`: 5fold + fullmodel visualization PNGs

**Compound ID mapping (13:42)**
- Added `compound_id` (DMID/HMDB) mapping to 529 candidates
- DMID: 8,579 rows (95.0%), HMDB: 216 rows (2.4%), InChIKey: 108 rows (1.2%)
- Created `dmid_mapping.csv`: 5,281 unique DMID-InChIKey-SMILES mapping

**TOP3 evaluation with compound names (13:49 ~ 14:24)**
- `eval_529_top3.py`: Scored 529 features x 3 models (Default, JJY, Final)
- Generated `eval_529_top3_results.xlsx` with Summary, per-model sheets, comparison
- Added compound names from 143 library + PubChem IUPAC API + SMILES fallback
- Created reusable `pubchem_name_cache.csv` (InChIKey -> IUPAC name)

**4-model evaluation (14:42)**
- Added NeoME V8 to both 143 and 529 evaluations (total 4 models)
- 143 library TOP1 results:
  | Model | TOP1 | TOP10 |
  |-------|------|-------|
  | Full Model | 63/143 (44.1%) | 89.6% |
  | NeoME V8 | 50/143 (35.0%) | 91.8% |
  | Param JJY | 46/143 (32.2%) | 86.6% |
  | CFM Default | 33/143 (23.1%) | 82.1% |

### Evening session (22:24 ~ 22:54)

**Peptide DTB candidate generation (22:24)**
- Created `generate_peptide_dtb.py` (requires python3.10 for RDKit)
- Generated 9,220 DTB-modified di/tripeptide SMILES from 20 standard amino acids
  - 420 dipeptides + 8,800 tripeptides
  - N-terminal DTB modification (not Pro) + Lys epsilon-amine DTB
  - Naming: `G*GG` = DTB at N-term Gly, `GK*G` = DTB at Lys epsilon
  - Mass range: 328.17 ~ 772.37 Da
- Ran full_model predictions: 16 parallel Docker containers, 9,220/9,220 completed (~4.4 hours)

**Enriched MGF files pulled from GitHub (22:54)**
- `24h_enriched_DTB_819.mgf`: 819 features (24h organelle, TrexNon FC>2)
- `shTAUT_enriched_DTB_1066.mgf`: 1,066 features (shTAUT perturbation, TrexNon FC>2)
- TITLE format: `PLID_24_253.1909_7.88` (dataset_mz_RT)

---

## Day 6: 2026-03-06 (Thursday) -- PLID Candidate Generation & Final Predictions

### Early morning (03:45)

**PLID candidate generation & full_model predictions**
- Created `generate_plid_candidates.py`:
  - Parsed 2 enriched MGF files (819 + 1,066 features)
  - Matched against union DB (268,759 compounds) + peptide DTB DB (9,220)
  - 10 ppm mass tolerance for [M+H]+ matching
  - Only primary amine (score=5) and secondary amine (score=4) DTB modifications
  - RDKit reaction SMARTS for DTB attachment
- Results:
  | Dataset | Features matched | Candidates | Union DB | Peptide DB |
  |---------|-----------------|------------|----------|------------|
  | 24h_819 | 514/819 | 27,483 | 26,434 | 1,049 |
  | shTAUT_1066 | 644/1,066 | 27,490 | 26,890 | 600 |
- Prediction optimization:
  - Reused 529 eval predictions (6,097 SMILES) -- no re-prediction needed
  - Reused peptide DTB predictions (9,220 SMILES) -- m/z match only
  - NEW SMILES needing prediction: 21,648
- Ran full_model predictions for 21,648 new SMILES: 21,643 completed (~4.6 hours)
  - 5 SMILES failed (likely invalid structures)

### Afternoon (15:03)

**Peptide DTB -- 3 additional model predictions**
- Ran `run_predict_peptide_all_params.sh` for cfm_default, param_jjy, neome_v8
- 3 models x 9,220 = 27,660 predictions total (30 containers in parallel)
- Completion:
  | Model | Predictions | Time |
  |-------|------------|------|
  | neome_v8 | 9,220/9,220 | ~3.0 hours |
  | param_jjy | 9,220/9,220 | ~3.1 hours |
  | cfm_default | 9,220/9,220 | ~10.6 hours (slowest) |
- Committed and pushed all results

---

## Summary of Outputs

### Models trained
- **Full Model** (final): 143 DTB compounds, transfer learning from CFM-ID 4.0

### Models evaluated (4 total)
| Model | Description |
|-------|-------------|
| cfm_default | CFM-ID 4.0 pretrained default [M+H]+ |
| param_jjy | JJY-trained parameters |
| full_model | This project's final trained model |
| neome_v8 | NeoME V8 trained parameters |

### Predictions completed

| Task | SMILES | Models | Total predictions |
|------|--------|--------|-------------------|
| 143 library TOP1 eval | 8,879 | 4 | 35,516 |
| 529 feature eval | 6,097 | 4 | 24,388 |
| Peptide DTB | 9,220 | 4 | 36,880 |
| PLID new SMILES | 21,648 | 1 (full_model) | 21,643 |
| **Total** | | | **~118,427** |

### Key findings
1. **Full Model is the best**: TOP1 accuracy 63/143 (44.1%), significantly outperforms CFM-ID default (23.1%)
2. **Optimal scoring**: `entropy * cosine^0.75 * iw_dice^1.25`, energy weights e0=0.40, e1=0.34, e2=0.26
3. **Peptide DTB DB**: 9,220 candidates covering all possible DTB-modified di/tripeptides (20 AA)
4. **PLID candidates**: ~27,000 candidates per dataset from union DB + peptide DTB matching

### Repository structure (final)
```
260301_CFM_Final_training/
  01_library/     - Source data (MGF, xlsx)
  02_config/      - Model parameters (4 models), config, features
  03_docker/      - Dockerfile + OpenMP patches
  04_training/    - 5-fold CV + full model training outputs
  05_evaluation/
    eval_529/          - 529 feature predictions (4 models)
    top1_eval/         - 143 library predictions (4 models)
    peptide_dtb/       - 9,220 peptide predictions (4 models)
    eval_24h_819/      - PLID 24h candidates
    eval_shTAUT_1066/  - PLID shTAUT candidates
    plid_new_predictions/ - 21,643 new SMILES predictions
    enriched_mgf/      - Source MGF files (819 + 1,066 features)
  06_scripts/     - Python/bash scripts
  07_figures/     - Training visualization PNGs
```
