# CFM-ID v5 Transfer Learning Parameters

## Default vs v5 비교표

| Parameter | Default [M+H]+ | v5 Transfer | 변경 이유 |
|-----------|:-:|:-:|---|
| **starting_step_size** | **0.001** | **0.0001** | Default의 1/10. 0.001은 TL에서 oscillation, 0.00001은 stall |
| **ending_step_size** | 0.00025 | 0.00005 | LR decay 비율 유지 (1/4) |
| **em_no_progress_count** | 2 | 3 | Early stop 전 더 탐색 허용 |
| **ga_max_iterations** | 30 | 20 | M-step 15-20에서 수렴 확인 (v2 log) |
| **ga_minibatch_nth_size** | 20 | 2 | 121 compounds 소규모 → ~50% minibatch |
| allow_intermediate_peak | 1 | (미설정) | v5에서 제거 (큰 영향 없음) |
| allow_cyclization | 0 | (미설정) | v5에서 제거 (큰 영향 없음) |
| lambda | 0.0 | 0.0 | 동일 |
| em_max_iterations | 100 | 100 | 동일 |
| em_converge_thresh | 0.01 | 0.01 | 동일 |
| ga_method | 2 (Adam) | 2 (Adam) | 동일 |
| ga_adam_beta_1/2 | 0.9/0.999 | 0.9/0.999 | 동일 |
| NN architecture | 128-128-1 | 128-128-1 | 동일 |
| dropout | 0.1, 0.1, 0 | 0.1, 0.1, 0 | 동일 |
| model_depth | 2 | 2 | 동일 |
| ionization_mode | 1 [M+H]+ | 1 [M+H]+ | 동일 |

**핵심 차이**: LR을 10x 줄이고 (0.001 → 0.0001), minibatch를 키움 (nth=20 → nth=2).
Default는 대규모 데이터셋(수만 개)용, v5는 소규모 TL(121개)용으로 조정.

---

## JJY Parameter (참고)

`260301_CFM_Final_training/Param_JJY/param_config.txt`에 별도 config 존재:

| Parameter | JJY | 비고 |
|-----------|-----|------|
| starting_step_size | 0.00003 | 매우 보수적 |
| ending_step_size | 0.00003 | Decay 없음 (고정 LR) |
| em_converge_thresh | 0.005 | 더 엄격 |
| em_no_progress_count | 5 | 더 인내 |
| em_max_iterations | 30 | 짧음 |
| ga_max_iterations | 10 | 짧음 |
| ga_minibatch_nth_size | 3 | |
| default_predicted_peak_max | 10000 | 제한 없음 |
| default_predicted_min_intensity | 3.0 | |
| nn_layer_freeze | 1, 1, 0 | Hidden layer freeze (v5에서는 미사용) |
| spectrum_depth/weight | 1개만 설정 | 에너지 1개만 |

---

## Training Data

- **121 compounds** (111 train + 22 validation, group=0/1)
- **121 spectra files** (3 energy levels per file)
- Excluded: Nucleotides, Sugar conjugates, Amide-linked
- DTB-derivatized SMILES 사용

## cfm-train Command
```bash
cfm-train -i input_molecules.txt -f features.txt -c config.txt \
  -p spectra/ -g 1 -w pretrained/param_output.log -l tmp_data/status.log
```
- **`-w`**: pretrained model weight 로드 (transfer learning 핵심)
- **`-g 1`**: group 1을 validation으로 사용

## Results (v5 vs Default)
| Energy | Default | v5 | Delta |
|--------|---------|-----|-------|
| e0 (10eV) | 0.652 | **0.800** | +0.149 |
| e1 (20eV) | 0.470 | **0.816** | +0.347 |
| e2 (40eV) | 0.124 | **0.745** | +0.621 |

Test set (23 unseen): e0 = **0.815** (no overfitting)

## File Structure
```
param_v5/
  README_v5_parameters.md      <- 이 문서
  default_param_config.txt     <- CFM-ID 4.0 default [M+H]+ config
  v5_config.txt                <- v5 transfer learning config
  features.txt                 <- 6 molecular features

원본 (cfmid_v5 Git repo):
  https://github.com/97changmo-beep/cfmid_v5.git
  train/input_molecules.txt, train/param_output.log, train/spectra/
```
