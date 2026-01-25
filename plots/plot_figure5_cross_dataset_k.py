"""
Figure 5: Cross-Dataset k Comparison.

This figure compares error vs target rank k for MNIST and YearPredictionMSD,
showing how approximation quality improves with increasing k.

Usage:
    python plot_figure5_cross_dataset_k.py

Output:
    figures/figure5_cross_dataset_k.pdf
    figures/figure5_cross_dataset_k.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
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
    else:
        return f'{y*100:.3f}%'


def load_json(filepath):
    """Load JSON data, return empty list if file not found."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_stats(data, key_field='k', value_field='error'):
    """Compute mean and std for each unique value of key_field."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item[key_field]].append(item[value_field])
    
    keys = sorted(grouped.keys())
    means = [np.mean(grouped[k]) for k in keys]
    stds = [np.std(grouped[k]) for k in keys]
    
    return np.array(keys), np.array(means), np.array(stds)


def find_k_for_error(k_vals, means, target_error):
    """Find k value that achieves target error."""
    for i, (k, e) in enumerate(zip(k_vals, means)):
        if e <= target_error:
            return k
    return None


def main():
    """Generate Figure 5: Cross-Dataset k Comparison."""
    print("=" * 60)
    print("Generating Figure 5: Cross-Dataset k Comparison")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data...")
    data_mnist = load_json('error_vs_k_repeated.json')
    data_year = load_json('error_vs_k_year.json')
    
    print(f"  MNIST: {len(data_mnist)} entries")
    print(f"  YearPredictionMSD: {len(data_year)} entries")
    
    if not data_mnist and not data_year:
        print("\nNo data available!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # =========================================================================
    # Plot MNIST
    # =========================================================================
    k_mnist_for_01pct = None
    if data_mnist:
        k_vals, means, stds = compute_stats(data_mnist, 'k', 'error')
        
        ax.fill_between(k_vals, means - stds, means + stds,
                        alpha=0.2, color=COLORS['mnist'])
        
        ax.plot(k_vals, means, 'o-', color=COLORS['mnist'], markersize=6,
                markevery=2, markerfacecolor='white', markeredgewidth=1.5,
                label='MNIST (c=100, l=1000, n=5000)')
        
        k_mnist_for_01pct = find_k_for_error(k_vals, means, 0.001)
        l_mnist = data_mnist[0].get('l', 1000)
    
    # =========================================================================
    # Plot YearPredictionMSD
    # =========================================================================
    k_year_for_01pct = None
    if data_year:
        k_vals, means, stds = compute_stats(data_year, 'k', 'error')
        
        ax.fill_between(k_vals, means - stds, means + stds,
                        alpha=0.2, color=COLORS['year'])
        
        ax.plot(k_vals, means, 's-', color=COLORS['year'], markersize=6,
                markevery=2, markerfacecolor='white', markeredgewidth=1.5,
                label='YearPredictionMSD (c=10,000, l=1000, n=5000)')
        
        k_year_for_01pct = find_k_for_error(k_vals, means, 0.001)
        l_year = data_year[0].get('l', 1000)
    

    
    # =========================================================================
    # Formatting
    # =========================================================================
    ax.set_xlabel('Target Rank k')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error vs Target Rank k: Dataset Comparison')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    fig.savefig('figures/figure5_cross_dataset_k.pdf')
    fig.savefig('figures/figure5_cross_dataset_k.png')
    
    print("\nFigures saved:")
    print("  figures/figure5_cross_dataset_k.pdf")
    print("  figures/figure5_cross_dataset_k.png")
    
    plt.close()
    print("\n" + "=" * 60)
    print("Figure 5 generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
