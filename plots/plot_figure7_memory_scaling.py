"""
Figure 7: Memory Scaling Comparison (Explicit vs Streaming)

This script creates a bar chart comparing memory footprint of:
- Explicit kernel matrix method
- Streaming (matrix-free) method

For n in {5000, 20000, 50000, 100000}.

Output:
    figures/figure7_memory_scaling.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)


def main():
    # Load memory scaling data
    data_file = 'memory_scaling.json'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Run generate_memory_scaling_data.py first.")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract values
    n_values = [d['n'] for d in data]
    n_labels = [f"{n//1000}k" for n in n_values]
    explicit_mem = [d['explicit_peak_gb'] for d in data]
    streaming_mem = [d['streaming_peak_gb'] for d in data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(n_values))
    bar_width = 0.35
    
    # Bars
    bars1 = ax.bar(x - bar_width/2, explicit_mem, bar_width, 
                   label='Explicit A', color='#E74C3C', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width/2, streaming_mem, bar_width, 
                   label='Streaming (Matrix-Free)', color='#3498DB', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    def autolabel(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(offset, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            elif height > 0.01:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(offset, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Formatting
    ax.set_xlabel('Problem Size n', fontsize=12)
    ax.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax.set_title('Memory Footprint: Explicit vs Streaming', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    
    # Log scale for y-axis (optional, but useful for large range)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.set_ylabel('Peak Memory (GB, log scale)', fontsize=12)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    output_path = 'figures/figure7_memory_scaling.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also show comparison table
    print("\nMemory Savings Summary:")
    print("-" * 50)
    print(f"{'n':>10}  {'Explicit':>12}  {'Streaming':>12}  {'Savings':>10}")
    print("-" * 50)
    for d in data:
        print(f"{d['n']:>10,}  {d['explicit_peak_gb']:>12.2f}GB  {d['streaming_peak_gb']:>12.2f}GB  {d['savings_ratio']:>10.1f}x")
    
    plt.show()


if __name__ == '__main__':
    main()
