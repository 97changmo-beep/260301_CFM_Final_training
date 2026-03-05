#!/usr/bin/env python3
"""
Grid search for optimal energy-level weight combination (w0, w1, w2)
that maximizes weighted-average cosine similarity across 5-fold CV test compounds.
"""

import pandas as pd
import numpy as np
from itertools import product

# ── Load data ────────────────────────────────────────────────────────────────
csv_path = "/home/rheelab/바탕화면/Changmo_FINAL/260301_CFM_Final_training/final_training/evaluation_results.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("DATA OVERVIEW")
print("="*80)
print(f"Shape : {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Models : {df['model'].unique()}")
print(f"Folds  : {sorted(df['fold'].unique())}")
print()

# Check for DICE columns
dice_cols = [c for c in df.columns if c.startswith("dice")]
if dice_cols:
    print(f"DICE columns found: {dice_cols}")
else:
    print("No DICE columns found. Only cosine similarity will be analyzed.")
print()

# ── Per-model summary (unweighted) ──────────────────────────────────────────
for model_name in ["trained", "baseline"]:
    sub = df[df["model"] == model_name]
    n = len(sub)
    print(f"--- {model_name.upper()} model  ({n} compound-fold observations) ---")
    for ecol in ["cos_e0", "cos_e1", "cos_e2"]:
        print(f"  {ecol}:  mean={sub[ecol].mean():.4f}  std={sub[ecol].std():.4f}  "
              f"min={sub[ecol].min():.4f}  max={sub[ecol].max():.4f}")
    print()

# ── Build weight grid (step 0.05, w0+w1+w2 = 1) ────────────────────────────
step = 0.05
ticks = np.arange(0.0, 1.0 + step/2, step)
ticks = np.round(ticks, 2)

weight_combos = []
for w0 in ticks:
    for w1 in ticks:
        w2 = round(1.0 - w0 - w1, 2)
        if w2 < -1e-9 or w2 > 1.0 + 1e-9:
            continue
        w2 = max(0.0, min(1.0, w2))  # clamp tiny float errors
        weight_combos.append((round(w0, 2), round(w1, 2), round(w2, 2)))

print(f"Total weight combinations to test: {len(weight_combos)}")
print()

# ── Grid search function ────────────────────────────────────────────────────
def grid_search(data, label):
    """Return sorted list of (mean_weighted_cos, w0, w1, w2) for a dataframe."""
    cos0 = data["cos_e0"].values
    cos1 = data["cos_e1"].values
    cos2 = data["cos_e2"].values

    results = []
    for w0, w1, w2 in weight_combos:
        weighted = w0 * cos0 + w1 * cos1 + w2 * cos2
        mean_val = weighted.mean()
        std_val  = weighted.std()
        median_val = np.median(weighted)
        results.append((mean_val, std_val, median_val, w0, w1, w2))

    results.sort(key=lambda x: -x[0])  # descending by mean
    return results

# ── Run for TRAINED model ────────────────────────────────────────────────────
print("="*80)
print("TRAINED MODEL — Optimal Energy Weights (Cosine Similarity)")
print("="*80)

trained_df = df[df["model"] == "trained"].copy()
trained_results = grid_search(trained_df, "trained")

print(f"\nTop 20 weight combinations (maximizing mean weighted cosine):\n")
print(f"{'Rank':>4}  {'w0':>5}  {'w1':>5}  {'w2':>5}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
print("-" * 60)
for i, (mean_v, std_v, med_v, w0, w1, w2) in enumerate(trained_results[:20], 1):
    print(f"{i:4d}  {w0:5.2f}  {w1:5.2f}  {w2:5.2f}  {mean_v:8.4f}  {std_v:8.4f}  {med_v:8.4f}")

# Best
best = trained_results[0]
print(f"\n>>> BEST for TRAINED: w0={best[3]:.2f}, w1={best[4]:.2f}, w2={best[5]:.2f}  "
      f"=> mean cosine = {best[0]:.4f}")

# ── Equal-weight reference ───────────────────────────────────────────────────
equal_idx = None
for i, (m, s, md, w0, w1, w2) in enumerate(trained_results):
    if abs(w0 - 0.35) < 0.001 and abs(w1 - 0.35) < 0.001 and abs(w2 - 0.30) < 0.001:
        equal_idx = i
    if abs(w0 - 1/3) < 0.02 and abs(w1 - 1/3) < 0.02 and abs(w2 - 1/3) < 0.02:
        # Closest to equal
        pass

# Find the ~equal weight entry (0.35, 0.35, 0.30) or (0.30, 0.35, 0.35), etc.
for i, (m, s, md, w0, w1, w2) in enumerate(trained_results):
    if abs(w0 - w1) < 0.06 and abs(w1 - w2) < 0.06 and abs(w0 - w2) < 0.06:
        print(f"\nNearest equal-weight entry: w0={w0:.2f}, w1={w1:.2f}, w2={w2:.2f}  "
              f"=> mean cosine = {m:.4f}  (rank {i+1})")
        break

# ── Run for BASELINE model ──────────────────────────────────────────────────
print("\n" + "="*80)
print("BASELINE MODEL — Optimal Energy Weights (Cosine Similarity)")
print("="*80)

baseline_df = df[df["model"] == "baseline"].copy()
baseline_results = grid_search(baseline_df, "baseline")

print(f"\nTop 20 weight combinations (maximizing mean weighted cosine):\n")
print(f"{'Rank':>4}  {'w0':>5}  {'w1':>5}  {'w2':>5}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
print("-" * 60)
for i, (mean_v, std_v, med_v, w0, w1, w2) in enumerate(baseline_results[:20], 1):
    print(f"{i:4d}  {w0:5.2f}  {w1:5.2f}  {w2:5.2f}  {mean_v:8.4f}  {std_v:8.4f}  {med_v:8.4f}")

best_b = baseline_results[0]
print(f"\n>>> BEST for BASELINE: w0={best_b[3]:.2f}, w1={best_b[4]:.2f}, w2={best_b[5]:.2f}  "
      f"=> mean cosine = {best_b[0]:.4f}")

# ── Compare trained vs baseline at their own optimal weights ─────────────────
print("\n" + "="*80)
print("COMPARISON: TRAINED vs BASELINE")
print("="*80)

# At trained-optimal weights
tw0, tw1, tw2 = best[3], best[4], best[5]
trained_at_opt = (tw0 * trained_df["cos_e0"] + tw1 * trained_df["cos_e1"] + tw2 * trained_df["cos_e2"]).mean()
baseline_at_trained_opt = (tw0 * baseline_df["cos_e0"] + tw1 * baseline_df["cos_e1"] + tw2 * baseline_df["cos_e2"]).mean()

print(f"\nAt TRAINED-optimal weights (w0={tw0:.2f}, w1={tw1:.2f}, w2={tw2:.2f}):")
print(f"  Trained  mean cosine = {trained_at_opt:.4f}")
print(f"  Baseline mean cosine = {baseline_at_trained_opt:.4f}")
print(f"  Improvement           = {trained_at_opt - baseline_at_trained_opt:+.4f} ({(trained_at_opt/baseline_at_trained_opt - 1)*100:+.1f}%)")

# At baseline-optimal weights
bw0, bw1, bw2 = best_b[3], best_b[4], best_b[5]
trained_at_bopt = (bw0 * trained_df["cos_e0"] + bw1 * trained_df["cos_e1"] + bw2 * trained_df["cos_e2"]).mean()
baseline_at_bopt = (bw0 * baseline_df["cos_e0"] + bw1 * baseline_df["cos_e1"] + bw2 * baseline_df["cos_e2"]).mean()

print(f"\nAt BASELINE-optimal weights (w0={bw0:.2f}, w1={bw1:.2f}, w2={bw2:.2f}):")
print(f"  Trained  mean cosine = {trained_at_bopt:.4f}")
print(f"  Baseline mean cosine = {baseline_at_bopt:.4f}")
print(f"  Improvement           = {trained_at_bopt - baseline_at_bopt:+.4f} ({(trained_at_bopt/baseline_at_bopt - 1)*100:+.1f}%)")

# At equal weights (1/3 each approximation: 0.35, 0.35, 0.30)
ew0, ew1, ew2 = 1/3, 1/3, 1/3
trained_at_equal = (ew0 * trained_df["cos_e0"] + ew1 * trained_df["cos_e1"] + ew2 * trained_df["cos_e2"]).mean()
baseline_at_equal = (ew0 * baseline_df["cos_e0"] + ew1 * baseline_df["cos_e1"] + ew2 * baseline_df["cos_e2"]).mean()

print(f"\nAt EQUAL weights (w0=0.333, w1=0.333, w2=0.333):")
print(f"  Trained  mean cosine = {trained_at_equal:.4f}")
print(f"  Baseline mean cosine = {baseline_at_equal:.4f}")
print(f"  Improvement           = {trained_at_equal - baseline_at_equal:+.4f} ({(trained_at_equal/baseline_at_equal - 1)*100:+.1f}%)")

# ── Per-energy-level comparison ──────────────────────────────────────────────
print("\n" + "="*80)
print("PER-ENERGY-LEVEL COMPARISON (mean cosine)")
print("="*80)
for ecol in ["cos_e0", "cos_e1", "cos_e2"]:
    t_mean = trained_df[ecol].mean()
    b_mean = baseline_df[ecol].mean()
    print(f"  {ecol}: trained={t_mean:.4f}  baseline={b_mean:.4f}  "
          f"improvement={t_mean - b_mean:+.4f} ({(t_mean/b_mean - 1)*100:+.1f}%)")

# ── Per-fold analysis for trained model at optimal weights ───────────────────
print("\n" + "="*80)
print("PER-FOLD ANALYSIS (Trained model at optimal weights)")
print("="*80)
tw0, tw1, tw2 = best[3], best[4], best[5]
print(f"Using weights: w0={tw0:.2f}, w1={tw1:.2f}, w2={tw2:.2f}\n")

for fold in sorted(trained_df["fold"].unique()):
    fold_data = trained_df[trained_df["fold"] == fold]
    weighted = tw0 * fold_data["cos_e0"] + tw1 * fold_data["cos_e1"] + tw2 * fold_data["cos_e2"]
    print(f"  Fold {fold}: n={len(fold_data):2d}  mean={weighted.mean():.4f}  "
          f"std={weighted.std():.4f}  min={weighted.min():.4f}  max={weighted.max():.4f}")

# ── Sensitivity analysis: how stable is the optimum? ─────────────────────────
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS (Trained model)")
print("="*80)
print("Mean cosine for selected weight combinations:\n")
test_weights = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),   # single energy
    (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),     # pair
    (1/3, 1/3, 1/3),                                          # equal
    (tw0, tw1, tw2),                                           # optimal
    (0.4, 0.3, 0.3), (0.3, 0.4, 0.3), (0.3, 0.3, 0.4),     # near-equal
    (0.5, 0.3, 0.2), (0.5, 0.2, 0.3),                        # e0-heavy
    (0.2, 0.5, 0.3), (0.3, 0.5, 0.2),                        # e1-heavy
    (0.2, 0.3, 0.5), (0.3, 0.2, 0.5),                        # e2-heavy
]

print(f"{'w0':>5}  {'w1':>5}  {'w2':>5}  {'Mean':>8}  {'Label'}")
print("-" * 55)
for ws in test_weights:
    w0, w1, w2 = ws
    val = (w0 * trained_df["cos_e0"] + w1 * trained_df["cos_e1"] + w2 * trained_df["cos_e2"]).mean()
    label = ""
    if w0 == tw0 and w1 == tw1 and w2 == tw2:
        label = "<-- OPTIMAL"
    elif abs(w0 - 1/3) < 0.01 and abs(w1 - 1/3) < 0.01:
        label = "<-- equal"
    elif w0 == 1.0:
        label = "e0 only"
    elif w1 == 1.0:
        label = "e1 only"
    elif w2 == 1.0:
        label = "e2 only"
    print(f"{w0:5.2f}  {w1:5.2f}  {w2:5.2f}  {val:8.4f}  {label}")

print("\n" + "="*80)
print("DONE")
print("="*80)
