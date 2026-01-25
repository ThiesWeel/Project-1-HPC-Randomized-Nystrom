"""
Generate all figures from pre-computed data.

Usage:
    cd plots
    python generate_all_figures.py

Output figures:
    figures/figure2_spectral.png
    figures/figure4_cross_dataset_ell.png
    figures/figure5_cross_dataset_k.png
    figures/figure6_kernel_sensitivity.png
    figures/figure7_memory_scaling.png
    figures/figure8_runtime_vs_n.png
    figures/scaling_P.png
    plots/heatmap_runtime.pdf (from scaling_analysis.py)
"""

import os
import subprocess
import sys


def run_script(script_name):
    """Run a plotting script."""
    print(f"\nRunning: {script_name}")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    return result.returncode == 0


def main():
    print("Generating figures for Nystrom approximation analysis")
    print("=" * 50)

    os.makedirs('figures', exist_ok=True)

    scripts = [
        'plot_figure2_spectral.py',
        'plot_figure4_cross_dataset_ell.py',
        'plot_figure5_cross_dataset_k.py',
        'plot_figure6_kernel_sensitivity.py',
        'plot_figure7_memory_scaling.py',
        'plot_figure8_runtime_vs_n.py',
        'plot_scaling_comparison.py',
    ]

    success = 0
    for script in scripts:
        if os.path.exists(script):
            if run_script(script):
                success += 1
        else:
            print(f"Warning: {script} not found")

    # Run heatmap script from parent directory
    print("\nRunning: ../scaling_analysis.py")
    os.chdir('..')
    subprocess.run([sys.executable, 'scaling_analysis.py'], capture_output=False)

    print(f"\nDone. {success}/{len(scripts)} scripts completed.")


if __name__ == '__main__':
    main()
