"""
Scaling Analysis: MPI Ranks vs BLAS Threads - Heatmap Visualization
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_data(filepath: str) -> list:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['experiments']


def extract_results(experiments: list) -> dict:
    """Extract and organize results by dataset."""
    results = {'mnist': {}, 'year': {}}
    for exp in experiments:
        dataset = exp['dataset']
        key = (exp['mpi_ranks'], exp['blas_threads'])
        runtime = exp['runtime_sec']
        if key not in results[dataset]:
            results[dataset][key] = runtime
        else:
            results[dataset][key] = min(results[dataset][key], runtime)
    return results


def plot_heatmap(results: dict, save_path_png: str, save_path_pdf: str):
    """Create heatmap of runtime vs (MPI ranks, BLAS threads)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ['mnist', 'year']
    titles = ['MNIST (d=780)', 'YearPredMSD (d=90)']

    for ax, dataset, title in zip(axes, datasets, titles):
        data = results[dataset]

        # Get unique P and BLAS values
        p_values = sorted(set(k[0] for k in data.keys()))
        blas_values = sorted(set(k[1] for k in data.keys()))

        # Create matrix for heatmap
        matrix = np.full((len(p_values), len(blas_values)), np.nan)
        for i, p in enumerate(p_values):
            for j, blas in enumerate(blas_values):
                if (p, blas) in data:
                    matrix[i, j] = data[(p, blas)]

        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Runtime (seconds)', color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        # Set ticks
        ax.set_xticks(range(len(blas_values)))
        ax.set_xticklabels(blas_values, color='black')
        ax.set_yticks(range(len(p_values)))
        ax.set_yticklabels(p_values, color='black')

        ax.set_xlabel('BLAS Threads', color='black')
        ax.set_ylabel('MPI Ranks', color='black')
        ax.set_title(title, color='black')

        # Add text annotations (all black)
        for i, p in enumerate(p_values):
            for j, blas in enumerate(blas_values):
                if (p, blas) in data:
                    total = p * blas
                    runtime = data[(p, blas)]
                    ax.text(j, i, f'{runtime:.1f}s\n({total}T)',
                           ha='center', va='center', fontsize=9, color='black')

        # Tick colors
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

    plt.tight_layout()

    # Save as PNG and PDF
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path_png}")
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Saved: {save_path_pdf}")

    return fig


def main():
    script_dir = Path('.')
    data_file = script_dir / 'optimized_experiment_results.json'
    plots_dir = script_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("Loading data...")
    experiments = load_experiment_data(data_file)
    results = extract_results(experiments)

    print("Generating heatmap...")
    plot_heatmap(
        results,
        save_path_png=plots_dir / 'heatmap_runtime.png',
        save_path_pdf=plots_dir / 'heatmap_runtime.pdf'
    )
    print("Done!")


if __name__ == '__main__':
    main()
