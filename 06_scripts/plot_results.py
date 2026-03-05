#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py
===============
Generate evaluation figures for 5-fold CFM-ID transfer learning.

Figures:
1. Training loss vs EM iteration (per energy, all folds)
2. GA iteration loss within EM steps (detailed convergence)
3. Bar chart: Trained vs Baseline cosine similarity
4. Per-compound scatter plot
"""
import sys, os, re
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = Path(__file__).resolve().parent
N_FOLDS = 5
ENERGY_LABELS = ['Energy 0 (10 eV)', 'Energy 1 (20 eV)', 'Energy 2 (40 eV)']
ENERGY_COLORS = ['#2196F3', '#FF9800', '#E91E63']

# ============================================================
# Parse training logs
# ============================================================
def parse_training_log(fold_idx):
    """Parse training.log for a fold. Returns dict with loss data per energy level."""
    log_path = BASE / f'fold_{fold_idx}' / 'training.log'
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        text = f.read()

    result = {}
    for energy in range(3):
        result[energy] = {
            'em_loss': [],         # M-Step total loss per EM iteration
            'em_loss_avg': [],     # M-Step avg loss per EM iteration
            'val_loss': [],        # Validation loss per EM iteration
            'val_loss_avg': [],    # Validation avg loss per EM iteration
            'ga_losses': [],       # List of lists: GA losses per EM iteration
            'em_time': [],         # Time per EM iteration
        }

    # Split by energy training blocks
    energy_blocks = re.split(r'\[Training\] Starting Energy (\d+)', text)
    # energy_blocks[0] = preamble, then alternating (energy_num, block_text)

    for i in range(1, len(energy_blocks), 2):
        energy = int(energy_blocks[i])
        block = energy_blocks[i + 1] if i + 1 < len(energy_blocks) else ''

        if energy not in result:
            continue

        # Parse EM iterations within this block
        em_blocks = re.split(r'EM Iteration \d+', block)

        for em_block in em_blocks[1:]:  # skip text before first EM iteration
            # Parse GA losses within this EM iteration
            ga_losses = []
            for m in re.finditer(r'(\d+)\.\[T\+[\d.]+s\]Loss=([-\d.e+]+)', em_block):
                ga_losses.append(float(m.group(2)))

            if ga_losses:
                result[energy]['ga_losses'].append(ga_losses)

            # Parse M-Step summary
            m_step = re.search(
                r'\[M-Step\]\[T\+([\d.]+)s\]Loss=([-\d.e+]+)\s+Loss_Avg=([-\d.e+]+)',
                em_block
            )
            if m_step:
                result[energy]['em_time'].append(float(m_step.group(1)))
                result[energy]['em_loss'].append(float(m_step.group(2)))
                result[energy]['em_loss_avg'].append(float(m_step.group(3)))

            # Parse validation loss
            val = re.search(
                r'Validation_Loss_Total=([-\d.e+]+)\s+Validation_Loss_Avg=([-\d.e+]+)',
                em_block
            )
            if val:
                result[energy]['val_loss'].append(float(val.group(1)))
                result[energy]['val_loss_avg'].append(float(val.group(2)))

    return result


def parse_evaluation_csv():
    """Parse evaluation_results.csv for per-compound data."""
    csv_path = BASE / 'evaluation_results.csv'
    if not csv_path.exists():
        return None

    import csv
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
# Figure 1: Training Loss vs EM Iteration
# ============================================================
def plot_em_loss(all_logs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Loss vs EM Iteration (All Folds)', fontsize=14, fontweight='bold')

    for energy in range(3):
        ax = axes[energy]
        for fold in range(N_FOLDS):
            if fold not in all_logs or all_logs[fold] is None:
                continue
            data = all_logs[fold][energy]
            if not data['em_loss_avg']:
                continue

            iters = range(len(data['em_loss_avg']))
            ax.plot(iters, data['em_loss_avg'], 'o-', label=f'Fold {fold}',
                    alpha=0.8, markersize=4, linewidth=1.5)

        ax.set_title(ENERGY_LABELS[energy], fontsize=12, fontweight='bold')
        ax.set_xlabel('EM Iteration')
        ax.set_ylabel('Average Loss (negative log-likelihood)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = BASE / 'fig_em_loss.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Figure 2: Validation Loss vs EM Iteration
# ============================================================
def plot_val_loss(all_logs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Validation Loss vs EM Iteration (All Folds)', fontsize=14, fontweight='bold')

    for energy in range(3):
        ax = axes[energy]
        for fold in range(N_FOLDS):
            if fold not in all_logs or all_logs[fold] is None:
                continue
            data = all_logs[fold][energy]
            if not data['val_loss_avg']:
                continue

            iters = range(len(data['val_loss_avg']))
            ax.plot(iters, data['val_loss_avg'], 's-', label=f'Fold {fold}',
                    alpha=0.8, markersize=4, linewidth=1.5)

        ax.set_title(ENERGY_LABELS[energy], fontsize=12, fontweight='bold')
        ax.set_xlabel('EM Iteration')
        ax.set_ylabel('Avg Validation Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = BASE / 'fig_val_loss.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Figure 3: GA Loss within EM Steps (detailed convergence)
# ============================================================
def plot_ga_loss(all_logs):
    fig, axes = plt.subplots(3, N_FOLDS, figsize=(20, 12))
    fig.suptitle('Gradient Ascent Loss per EM Iteration', fontsize=14, fontweight='bold')

    for energy in range(3):
        for fold in range(N_FOLDS):
            ax = axes[energy][fold]
            if fold not in all_logs or all_logs[fold] is None:
                ax.set_visible(False)
                continue

            data = all_logs[fold][energy]
            if not data['ga_losses']:
                ax.set_visible(False)
                continue

            for em_iter, ga_loss in enumerate(data['ga_losses']):
                ax.plot(range(1, len(ga_loss) + 1), ga_loss,
                        '-', alpha=0.7, linewidth=1.2,
                        label=f'EM {em_iter}')

            if energy == 0:
                ax.set_title(f'Fold {fold}', fontsize=10, fontweight='bold')
            if fold == 0:
                ax.set_ylabel(f'{ENERGY_LABELS[energy]}\nLoss', fontsize=9)
            if energy == 2:
                ax.set_xlabel('GA Iteration')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)
            # X-axis: GA iterations are integers → use 2 or 5 step ticks
            x_max = max(len(g) for g in data['ga_losses'])
            if x_max <= 10:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            else:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            if len(data['ga_losses']) <= 8:
                ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    out = BASE / 'fig_ga_loss.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Figure 4: Bar chart - Trained vs Baseline
# ============================================================
def plot_comparison_bar():
    """Bar chart comparing trained vs baseline across energy levels."""
    # Re-run evaluation logic inline
    from evaluate_5fold import evaluate_fold, evaluate_baseline

    fold_trained = {e: [] for e in range(3)}
    fold_baseline = {e: [] for e in range(3)}

    for fold in range(N_FOLDS):
        results = evaluate_fold(fold)
        baseline = evaluate_baseline(fold, None)
        if results:
            for e in range(3):
                fold_trained[e].append(np.mean([r[f'cos_e{e}'] for r in results]))
        if baseline:
            for e in range(3):
                fold_baseline[e].append(np.mean([r[f'cos_e{e}'] for r in baseline]))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    trained_means = [np.mean(fold_trained[e]) for e in range(3)]
    trained_stds = [np.std(fold_trained[e]) for e in range(3)]
    baseline_means = [np.mean(fold_baseline[e]) for e in range(3)]
    baseline_stds = [np.std(fold_baseline[e]) for e in range(3)]

    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='CFM-ID 4.0 Default', color='#90CAF9', edgecolor='#1565C0',
                   capsize=5, linewidth=1.2)
    bars2 = ax.bar(x + width/2, trained_means, width, yerr=trained_stds,
                   label='Transfer Learning (DTB)', color='#EF5350', edgecolor='#B71C1C',
                   capsize=5, linewidth=1.2)

    ax.set_ylabel('Cosine Similarity', fontsize=13)
    ax.set_title('CFM-ID Default vs Transfer Learning\n(5-Fold Cross-Validation, 143 DTB Compounds)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['10 eV\n(Energy 0)', '20 eV\n(Energy 1)', '40 eV\n(Energy 2)'], fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars1, baseline_means, baseline_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, mean, std in zip(bars2, trained_means, trained_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#B71C1C')

    # Add delta annotations
    for i in range(3):
        delta = trained_means[i] - baseline_means[i]
        mid_x = x[i]
        mid_y = max(trained_means[i] + trained_stds[i],
                    baseline_means[i] + baseline_stds[i]) + 0.08
        ax.annotate(f'+{delta:.3f}', xy=(mid_x, mid_y),
                    fontsize=11, fontweight='bold', color='#2E7D32',
                    ha='center')

    plt.tight_layout()
    out = BASE / 'fig_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Figure 5: Per-compound paired scatter
# ============================================================
def plot_paired_scatter():
    """Scatter plot: baseline vs trained cosine for each compound."""
    from evaluate_5fold import evaluate_fold, evaluate_baseline

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Per-Compound: Transfer Learning vs CFM-ID Default',
                 fontsize=14, fontweight='bold')

    for energy in range(3):
        ax = axes[energy]
        all_baseline = []
        all_trained = []

        for fold in range(N_FOLDS):
            results = evaluate_fold(fold)
            baseline = evaluate_baseline(fold, None)
            if not results or not baseline:
                continue

            # Match by compound_id
            base_dict = {r['compound_id']: r[f'cos_e{energy}'] for r in baseline}
            for r in results:
                cid = r['compound_id']
                if cid in base_dict:
                    all_baseline.append(base_dict[cid])
                    all_trained.append(r[f'cos_e{energy}'])

        if not all_baseline:
            continue

        ax.scatter(all_baseline, all_trained, alpha=0.6, s=30,
                   c=ENERGY_COLORS[energy], edgecolors='k', linewidths=0.3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)

        # Stats
        above = sum(1 for b, t in zip(all_baseline, all_trained) if t > b)
        total = len(all_baseline)
        ax.text(0.05, 0.92, f'{above}/{total} improved ({100*above/total:.0f}%)',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('CFM-ID Default (cosine)', fontsize=11)
        ax.set_ylabel('Transfer Learning (cosine)', fontsize=11)
        ax.set_title(ENERGY_LABELS[energy], fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = BASE / 'fig_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Figure 6: Per-fold box plot
# ============================================================
def plot_boxplot():
    """Box plot showing distribution of cosine similarities per fold."""
    from evaluate_5fold import evaluate_fold, evaluate_baseline

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cosine Similarity Distribution per Fold', fontsize=14, fontweight='bold')

    for energy in range(3):
        ax = axes[energy]
        trained_data = []
        baseline_data = []

        for fold in range(N_FOLDS):
            results = evaluate_fold(fold)
            baseline = evaluate_baseline(fold, None)
            if results:
                trained_data.append([r[f'cos_e{energy}'] for r in results])
            if baseline:
                baseline_data.append([r[f'cos_e{energy}'] for r in baseline])

        positions_t = np.arange(N_FOLDS) * 2.5
        positions_b = positions_t + 0.8

        bp1 = ax.boxplot(baseline_data, positions=positions_b, widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor='#BBDEFB', edgecolor='#1565C0'),
                         medianprops=dict(color='#1565C0', linewidth=2))
        bp2 = ax.boxplot(trained_data, positions=positions_t, widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor='#FFCDD2', edgecolor='#B71C1C'),
                         medianprops=dict(color='#B71C1C', linewidth=2))

        ax.set_xticks(positions_t + 0.4)
        ax.set_xticklabels([f'Fold {i}' for i in range(N_FOLDS)])
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(ENERGY_LABELS[energy], fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend([bp2['boxes'][0], bp1['boxes'][0]],
                  ['Transfer Learning', 'CFM-ID Default'],
                  fontsize=9, loc='lower left')

    plt.tight_layout()
    out = BASE / 'fig_boxplot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('  Generating Evaluation Figures')
    print('=' * 60)

    # Parse all training logs
    print('\nParsing training logs...')
    all_logs = {}
    for fold in range(N_FOLDS):
        log = parse_training_log(fold)
        if log:
            all_logs[fold] = log
            for e in range(3):
                n_em = len(log[e]['em_loss'])
                n_ga = sum(len(g) for g in log[e]['ga_losses'])
                print(f'  Fold {fold} Energy {e}: {n_em} EM iterations, {n_ga} GA iterations')

    # Figure 1: EM loss
    print('\nPlotting EM loss...')
    plot_em_loss(all_logs)

    # Figure 2: Validation loss
    print('\nPlotting validation loss...')
    plot_val_loss(all_logs)

    # Figure 3: GA loss detail
    print('\nPlotting GA loss detail...')
    plot_ga_loss(all_logs)

    # Figure 4: Comparison bar chart
    print('\nPlotting comparison bar chart...')
    plot_comparison_bar()

    # Figure 5: Paired scatter
    print('\nPlotting paired scatter...')
    plot_paired_scatter()

    # Figure 6: Box plot
    print('\nPlotting box plot...')
    plot_boxplot()

    print('\nAll figures saved!')


if __name__ == '__main__':
    main()
