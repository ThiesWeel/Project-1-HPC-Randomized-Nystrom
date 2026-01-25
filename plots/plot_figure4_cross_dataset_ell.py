"""
Figure 4: Cross-Dataset l Comparison.

This figure compares error vs sketch size l for MNIST and YearPredictionMSD,
showing how different data characteristics affect approximation quality.

Usage:
    python plot_figure4_cross_dataset_ell.py

Output:
    figures/figure4_cross_dataset_ell.pdf
    figures/figure4_cross_dataset_ell.png
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
        return f'{y*100:.1f}%'
    elif y >= 0.001:
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


def compute_stats(data, key_field='l', value_field='error'):
    """Compute mean and std for each unique value of key_field."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item[key_field]].append(item[value_field])
    
    keys = sorted(grouped.keys())
    means = [np.mean(grouped[k]) for k in keys]
    stds = [np.std(grouped[k]) for k in keys]
    
    return np.array(keys), np.array(means), np.array(stds)


def find_saturation_point(l_vals, means, threshold=0.03):
    """Find where improvement rate drops below threshold."""
    if len(l_vals) < 3:
        return None
    relative_change = np.abs(np.diff(means)) / means[:-1]
    sat_idx = np.where(relative_change < threshold)[0]
    if len(sat_idx) > 0:
        return l_vals[sat_idx[0] + 1]
    return None


def main():
    """Generate Figure 4: Cross-Dataset l Comparison."""
    print("=" * 60)
    print("Generating Figure 4: Cross-Dataset l Comparison")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data...")
    data_mnist = load_json('error_vs_ell_repeated.json')
    data_year = load_json('error_vs_ell_year.json')
    
    print(f"  MNIST: {len(data_mnist)} entries")
    print(f"  YearPredictionMSD: {len(data_year)} entries")
    
    if not data_mnist and not data_year:
        print("\nNo data available!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sat_points = {}
    
    # =========================================================================
    # Plot MNIST
    # =========================================================================
    if data_mnist:
        l_vals, means, stds = compute_stats(data_mnist, 'l', 'error')
        
        ax.fill_between(l_vals, means - stds, means + stds,
                        alpha=0.2, color=COLORS['mnist'])
        
        # Plot with markers every 3rd point
        ax.plot(l_vals, means, 'o-', color=COLORS['mnist'], markersize=6,
                markevery=3, markerfacecolor='white', markeredgewidth=1.5,
                label='MNIST (c=100, n=5000, k=100)')
        
        sat_points['mnist'] = find_saturation_point(l_vals, means)
    
    # =========================================================================
    # Plot YearPredictionMSD
    # =========================================================================
    if data_year:
        l_vals, means, stds = compute_stats(data_year, 'l', 'error')
        
        ax.fill_between(l_vals, means - stds, means + stds,
                        alpha=0.2, color=COLORS['year'])
        
        ax.plot(l_vals, means, 's-', color=COLORS['year'], markersize=6,
                markevery=3, markerfacecolor='white', markeredgewidth=1.5,
                label='YearPredictionMSD (c=10,000, n=5000, k=100)')
        
        sat_points['year'] = find_saturation_point(l_vals, means)
    
   
    # =========================================================================
    # Formatting
    # =========================================================================
    ax.set_xlabel('Sketch Size l')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error vs Sketch Size l: Dataset Comparison')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    fig.savefig('figures/figure4_cross_dataset_ell.pdf')
    fig.savefig('figures/figure4_cross_dataset_ell.png')
    
    print("\nFigures saved:")
    print("  figures/figure4_cross_dataset_ell.pdf")
    print("  figures/figure4_cross_dataset_ell.png")
    
    plt.close()
    print("\n" + "=" * 60)
    print("Figure 4 generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
