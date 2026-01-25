"""
Figure 2: Spectral Analysis (Eigenvalue Decay).

This figure shows the eigenvalue spectrum of kernel matrices:
  (a) Eigenvalue decay (log scale)
  (b) Cumulative spectral energy

Usage:
    python plot_figure2_spectral.py

Output:
    figures/figure2_spectral.pdf
    figures/figure2_spectral.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color scheme
COLORS = {
    'mnist': '#1f77b4',
    'year_10k': '#ff7f0e',
    'year_100k': '#2ca02c',
}


def load_json(filepath):
    """Load JSON data, return None if file not found."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    """Generate Figure 2: Spectral Analysis."""
    print("=" * 60)
    print("Generating Figure 2: Spectral Analysis")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    print("Loading eigenvalue data...")
    data_mnist = load_json('eigenvalues_mnist_c100.json')
    data_year_10k = load_json('eigenvalues_year_c10000.json')
    data_year_100k = load_json('eigenvalues_year_c100000.json')
    
    has_mnist = data_mnist is not None
    has_year_10k = data_year_10k is not None
    has_year_100k = data_year_100k is not None
    
    if not has_mnist and not has_year_10k and not has_year_100k:
        print("\nNo eigenvalue data available!")
        print("Please run: python run_eigenvalue_experiments.py")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # =========================================================================
    # Subplot (a): Eigenvalue Decay
    # =========================================================================
    max_idx = 500  # Show first 500 eigenvalues
    
    if has_mnist:
        eigenvalues = np.array(data_mnist['eigenvalues'])[:max_idx]
        indices = np.arange(1, len(eigenvalues) + 1)
        ax1.semilogy(indices, eigenvalues, color=COLORS['mnist'], 
                     linewidth=2, label='MNIST (c=100)')
        print(f"  MNIST: {len(data_mnist['eigenvalues'])} eigenvalues loaded")
    
    if has_year_10k:
        eigenvalues = np.array(data_year_10k['eigenvalues'])[:max_idx]
        indices = np.arange(1, len(eigenvalues) + 1)
        ax1.semilogy(indices, eigenvalues, color=COLORS['year_10k'],
                     linewidth=2, label='YearPred (c=10,000)')
        print(f"  Year (c=10,000): {len(data_year_10k['eigenvalues'])} eigenvalues loaded")
    
    if has_year_100k:
        eigenvalues = np.array(data_year_100k['eigenvalues'])[:max_idx]
        indices = np.arange(1, len(eigenvalues) + 1)
        ax1.semilogy(indices, eigenvalues, color=COLORS['year_100k'],
                     linewidth=2, linestyle='--', label='YearPred (c=100,000)')
        print(f"  Year (c=100,000): {len(data_year_100k['eigenvalues'])} eigenvalues loaded")
    

    
    ax1.set_xlabel('Eigenvalue Index i')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.set_title('(a) Eigenvalue Decay')
    ax1.set_xlim(1, max_idx)
    ax1.set_ylim(1e-12, 1e4)
    ax1.set_xscale('log')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Subplot (b): Cumulative Spectral Energy
    # =========================================================================
    if has_mnist:
        eigenvalues = np.array(data_mnist['eigenvalues'])[:max_idx]
        cumsum = np.cumsum(eigenvalues)
        total = np.sum(data_mnist['eigenvalues'])
        fraction = cumsum / total * 100
        indices = np.arange(1, len(eigenvalues) + 1)
        ax2.plot(indices, fraction, color=COLORS['mnist'], linewidth=2, label='MNIST (c=100)')
    
    if has_year_10k:
        eigenvalues = np.array(data_year_10k['eigenvalues'])[:max_idx]
        cumsum = np.cumsum(eigenvalues)
        total = np.sum(data_year_10k['eigenvalues'])
        fraction = cumsum / total * 100
        indices = np.arange(1, len(eigenvalues) + 1)
        ax2.plot(indices, fraction, color=COLORS['year_10k'], linewidth=2, label='YearPred (c=10,000)')
    
    if has_year_100k:
        eigenvalues = np.array(data_year_100k['eigenvalues'])[:max_idx]
        cumsum = np.cumsum(eigenvalues)
        total = np.sum(data_year_100k['eigenvalues'])
        fraction = cumsum / total * 100
        indices = np.arange(1, len(eigenvalues) + 1)
        ax2.plot(indices, fraction, color=COLORS['year_100k'], linewidth=2, linestyle='--', 
                 label='YearPred (c=100,000)')
    
    # Reference lines at 95% and 99%
    ax2.axhline(y=95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.text(max_idx * 0.98, 95.5, '95%', fontsize=8, color='gray', ha='right')
    
    ax2.set_xlabel('Number of Eigenvalues k')
    ax2.set_ylabel('Cumulative Energy (%)')
    ax2.set_title('(b) Cumulative Spectral Energy')
    ax2.set_xlim(1, max_idx)
    ax2.set_ylim(0, 102)
    ax2.set_xscale('log')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Main title
    fig.suptitle('Kernel Matrix Spectral Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    fig.savefig('figures/figure2_spectral.pdf', bbox_inches='tight')
    fig.savefig('figures/figure2_spectral.png', bbox_inches='tight')
    
    print("\nFigures saved:")
    print("  figures/figure2_spectral.pdf")
    print("  figures/figure2_spectral.png")
    
    plt.close()
    print("\n" + "=" * 60)
    print("Figure 2 generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
