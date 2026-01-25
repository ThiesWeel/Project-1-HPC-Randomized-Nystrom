"""
Plot Parallel Scaling Results: Runtime vs Number of Processors

Reads scaling_comparison_P{P}.json files and creates a plot showing:
- Runtime vs P
- Speedup vs P
- Efficiency vs P

Usage:
    python plot_scaling_comparison.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_results():
    """Load all scaling results."""
    P_values = [1, 4, 9, 16]
    results = []
    
    for P in P_values:
        filename = f'scaling_comparison_P{P}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded {filename}: T = {data['avg_timings']['t_total']:.3f} s")
        else:
            print(f"Warning: {filename} not found")
    
    return results


def main():
    """Generate scaling comparison plots."""
    results = load_results()
    
    if len(results) == 0:
        print("No results found! Run the experiments first.")
        return
    
    # Extract data
    P_values = [r['P'] for r in results]
    total_times = [r['avg_timings']['t_total'] for r in results]
    Y_times = [r['avg_timings']['t_Y_local'] for r in results]
    gather_times = [r['avg_timings']['t_gather'] for r in results]
    
    # Compute speedup and efficiency (relative to P=1)
    T1 = total_times[0] if P_values[0] == 1 else total_times[0] * P_values[0]
    speedups = [T1 / t for t in total_times]
    efficiencies = [s / p * 100 for s, p in zip(speedups, P_values)]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Runtime vs P
    ax1 = axes[0]
    ax1.plot(P_values, total_times, 'o-', color='blue', linewidth=2, markersize=10, label='Total')
    ax1.plot(P_values, Y_times, 's--', color='green', linewidth=1.5, markersize=8, label='Y computation')
    ax1.plot(P_values, gather_times, '^--', color='red', linewidth=1.5, markersize=8, label='Communication')
    
    ax1.set_xlabel('Number of Processors (P)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime vs Number of Processors', fontsize=14)
    ax1.set_xticks(P_values)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(P_values) + 1)
    ax1.set_ylim(bottom=0)
    
    # Add data labels
    for p, t in zip(P_values, total_times):
        ax1.annotate(f'{t:.2f}s', (p, t), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Speedup vs P
    ax2 = axes[1]
    ax2.plot(P_values, speedups, 'o-', color='blue', linewidth=2, markersize=10, label='Actual speedup')
    ax2.plot(P_values, P_values, 'k--', linewidth=1.5, label='Ideal (linear)')
    
    ax2.set_xlabel('Number of Processors (P)', fontsize=12)
    ax2.set_ylabel('Speedup (T₁/Tₚ)', fontsize=12)
    ax2.set_title('Speedup vs Number of Processors', fontsize=14)
    ax2.set_xticks(P_values)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(P_values) + 1)
    ax2.set_ylim(bottom=0)
    
    # Add efficiency labels
    for p, s, e in zip(P_values, speedups, efficiencies):
        ax2.annotate(f'{s:.1f}x\n({e:.0f}%)', (p, s), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/scaling_P.png', bbox_inches='tight', dpi=150)
    print("\nSaved figures/scaling_P.png")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Scaling Summary")
    print("=" * 60)
    print(f"{'P':>4} {'Time (s)':>12} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 60)
    for p, t, s, e in zip(P_values, total_times, speedups, efficiencies):
        print(f"{p:>4} {t:>12.3f} {s:>10.2f}x {e:>11.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
