"""
Figure 8: Runtime vs n (3 Curves Comparison)

This script creates a runtime comparison plot showing:
1. Explicit A, no cache blocking (red, dashed)
2. Streaming Y, no cache blocking (blue, solid)  
3. Streaming Y, with cache blocking (green, solid)

For n in {5000, 20000, 50000, 100000}.

Output:
    figures/figure8_runtime_vs_n.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)


def main():
    # Load runtime data
    data_file = 'runtime_comparison.json'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Run run_runtime_comparison.py first.")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Organize data by mode
    explicit_data = [d for d in data if d['mode'] == 'explicit_no_cache']
    streaming_no_cache = [d for d in data if d['mode'] == 'streaming_no_cache']
    streaming_with_cache = [d for d in data if d['mode'] == 'streaming_with_cache']
    
    # Sort by n
    explicit_data.sort(key=lambda x: x['n'])
    streaming_no_cache.sort(key=lambda x: x['n'])
    streaming_with_cache.sort(key=lambda x: x['n'])
    
    # Extract values
    explicit_n = [d['n'] for d in explicit_data]
    explicit_t = [d['total'] for d in explicit_data]
    
    snc_n = [d['n'] for d in streaming_no_cache]
    snc_t = [d['total'] for d in streaming_no_cache]
    
    swc_n = [d['n'] for d in streaming_with_cache]
    swc_t = [d['total'] for d in streaming_with_cache]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot curves
    if explicit_n:
        ax.plot(explicit_n, explicit_t, 'o--', color='#E74C3C', linewidth=2, markersize=8,
                label='Explicit A (no cache blocking)', markeredgecolor='black', markeredgewidth=0.5)
    
    if snc_n:
        ax.plot(snc_n, snc_t, 's-', color='#3498DB', linewidth=2, markersize=8,
                label='Streaming Y (no cache blocking)', markeredgecolor='black', markeredgewidth=0.5)
    
    if swc_n:
        ax.plot(swc_n, swc_t, '^-', color='#27AE60', linewidth=2, markersize=8,
                label='Streaming Y (with cache blocking)', markeredgecolor='black', markeredgewidth=0.5)
    
    # Add annotations for speedup at largest common n
    if explicit_t and streaming_no_cache and streaming_with_cache:
        # Find common n values
        common_n = set(explicit_n) & set(snc_n) & set(swc_n)
        if common_n:
            max_common = max(common_n)
            exp_t = next(d['total'] for d in explicit_data if d['n'] == max_common)
            snc_t_val = next(d['total'] for d in streaming_no_cache if d['n'] == max_common)
            swc_t_val = next(d['total'] for d in streaming_with_cache if d['n'] == max_common)
            
            # Calculate speedups
            snc_speedup = exp_t / snc_t_val if snc_t_val > 0 else 0
            swc_speedup = exp_t / swc_t_val if swc_t_val > 0 else 0
            
            print(f"\nAt n = {max_common}:")
            print(f"  Explicit: {exp_t:.2f}s")
            print(f"  Streaming (no cache): {snc_t_val:.2f}s (speedup: {snc_speedup:.1f}x)")
            print(f"  Streaming (cache): {swc_t_val:.2f}s (speedup: {swc_speedup:.1f}x)")
    
    # Formatting
    ax.set_xlabel('Problem Size n', fontsize=12)
    ax.set_ylabel('Total Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison: Explicit vs Streaming Methods\n'
                 '(YearPredictionMSD, l=1000, k=100)', fontsize=13, fontweight='bold')
    
    # x-axis formatting
    ax.set_xticks([5000, 20000, 50000, 100000])
    ax.set_xticklabels(['5k', '20k', '50k', '100k'], fontsize=11)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Optional: log-log scale if range is large
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save
    output_path = 'figures/figure8_runtime_vs_n.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Runtime Summary (seconds)")
    print("=" * 70)
    print(f"{'n':>10}  {'Explicit':>12}  {'Stream(no)':>12}  {'Stream(cache)':>14}")
    print("-" * 70)
    
    all_n = sorted(set(explicit_n + snc_n + swc_n))
    for n in all_n:
        exp_t = next((d['total'] for d in explicit_data if d['n'] == n), None)
        snc_t = next((d['total'] for d in streaming_no_cache if d['n'] == n), None)
        swc_t = next((d['total'] for d in streaming_with_cache if d['n'] == n), None)
        
        exp_str = f"{exp_t:.2f}" if exp_t else "N/A"
        snc_str = f"{snc_t:.2f}" if snc_t else "N/A"
        swc_str = f"{swc_t:.2f}" if swc_t else "N/A"
        
        print(f"{n:>10,}  {exp_str:>12}  {snc_str:>12}  {swc_str:>14}")
    plt.show()
    


if __name__ == '__main__':
    main()
