"""
Figure 6: Kernel Sensitivity Analysis.

This figure shows how Nystrom approximation error varies with kernel
bandwidth c for different datasets:
  (a) MNIST: c from 10 to 500
  (b) YearPredictionMSD: c from 10^3 to 10^5

Usage:
    python plot_figure6_kernel_sensitivity.py

Output:
    figures/figure6_kernel_sensitivity.pdf
    figures/figure6_kernel_sensitivity.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (12, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Colors
COLORS = {
    'mnist': '#1f77b4',
    'year': '#ff7f0e',
}


def format_pct(y, pos):
    """Format y-axis as percentages."""
    if y >= 0.01:
        return f'{y*100:.0f}%'
    elif y >= 0.001:
        return f'{y*100:.1f}%'
    elif y >= 0.0001:
        return f'{y*100:.2f}%'
    elif y >= 1e-6:
        return f'{y*100:.4f}%'
    else:
        return f'{y*100:.1e}%'


def load_json(filepath):
    """Load JSON data, return empty list if file not found."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_stats(data, key_field='c', value_field='error'):
    """Compute mean and std for each unique value of key_field."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item[key_field]].append(item[value_field])
    
    keys = sorted(grouped.keys())
    means = [np.mean(grouped[k]) for k in keys]
    stds = [np.std(grouped[k]) for k in keys]
    
    return np.array(keys), np.array(means), np.array(stds)


def plot_kernel_sensitivity(ax, data, title, color, recommended_c, dataset_name):
    """Plot error vs kernel bandwidth c for one dataset."""
    if not data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    c_vals, means, stds = compute_stats(data, 'c', 'error')
    
    # Confidence band
    ax.fill_between(c_vals, means - stds, means + stds, alpha=0.2, color=color)
    
    # Main curve
    ax.plot(c_vals, means, 'o-', color=color, markersize=6,
            markerfacecolor='white', markeredgewidth=1.5, label='Mean +/- Std Dev')

    ax.set_xlabel('Kernel Bandwidth c')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Parameters annotation
    if data:
        n = data[0].get('n', 'N/A')
        l = data[0].get('l', 'N/A')
        k = data[0].get('k', 'N/A')
        ax.text(0.02, 0.02, f'n={n}, l={l}, k={k}',
                transform=ax.transAxes, fontsize=8, style='italic',
                verticalalignment='bottom')
    
    

def main():
    """Generate Figure 6: Kernel Sensitivity Analysis."""
    print("=" * 60)
    print("Generating Figure 6: Kernel Sensitivity Analysis")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data...")
    data_mnist = load_json('error_vs_c_mnist.json')
    data_year = load_json('error_vs_c_year.json')
    
    print(f"  MNIST: {len(data_mnist)} entries")
    print(f"  YearPredictionMSD: {len(data_year)} entries")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # =========================================================================
    # Subplot (a): MNIST
    # =========================================================================
    plot_kernel_sensitivity(
        ax1, data_mnist,
        '(a) MNIST Dataset',
        COLORS['mnist'],
        recommended_c=100,
        dataset_name='mnist'
    )
    
    # =========================================================================
    # Subplot (b): YearPredictionMSD
    # =========================================================================
    plot_kernel_sensitivity(
        ax2, data_year,
        '(b) YearPredictionMSD Dataset',
        COLORS['year'],
        recommended_c=10000,
        dataset_name='year'
    )
    
    # Main title
    fig.suptitle('Kernel Bandwidth Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)

    
    plt.tight_layout()
    
    fig.savefig('figures/figure6_kernel_sensitivity.pdf', bbox_inches='tight')
    fig.savefig('figures/figure6_kernel_sensitivity.png', bbox_inches='tight')
    
    print("\nFigures saved:")
    print("  figures/figure6_kernel_sensitivity.pdf")
    print("  figures/figure6_kernel_sensitivity.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\nKernel Sensitivity Summary:")
    
    if data_mnist:
        c_vals, means, stds = compute_stats(data_mnist, 'c', 'error')
        min_idx = np.argmin(means)
        print(f"\n  MNIST:")
        print(f"    Optimal c: {c_vals[min_idx]:.0f}")
        print(f"    Best error: {means[min_idx]:.6e}")
    
    if data_year:
        c_vals, means, stds = compute_stats(data_year, 'c', 'error')
        min_idx = np.argmin(means)
        print(f"\n  YearPredictionMSD:")
        print(f"    Optimal c: {c_vals[min_idx]:.0e}")
        print(f"    Best error: {means[min_idx]:.6e}")
    
    plt.close()
    print("\n" + "=" * 60)
    print("Figure 6 generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
