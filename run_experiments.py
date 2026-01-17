"""
Experiment runner for Randomized Nyström approximation.

This script runs experiments for:
1. Varying sketch dimension l
2. Varying target rank k
3. Varying kernel bandwidth c (sigma)
4. Scaling with number of MPI processes

Supports both MNIST and YearPredictionMSD datasets.

Usage:
    mpiexec -n P python run_experiments.py [--dataset mnist|year] [--n N] [--mode MODE]

Modes:
    - accuracy: Vary l, k, c and measure error
    - scaling: Vary P and measure time
    - single: Single run with specified parameters
"""

import sys
import os
import argparse
import math
import numpy as np
from mpi4py import MPI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.kernel import generate_features, rbf_kernel_block
from data.sketch import generate_omega_block
from data.libsvm_loader import read_dataset
from nystrom import (
    create_row_communicator,
    create_col_communicator,
    compute_nystrom_full_pipeline,
    estimate_nuclear_norm_A,
    compute_relative_error,
    measure_time,
)


def validate_process_count(size):
    """Validate that the number of MPI processes is a perfect square."""
    p = int(math.sqrt(size))
    if p * p != size:
        return None
    return p


def rank_to_grid_coords(rank, p):
    """Map MPI rank to 2D grid coordinates."""
    i = rank // p
    j = rank % p
    return i, j


def compute_block_range(index, p, n):
    """Compute the global index range owned by a process."""
    base_size = n // p
    remainder = n % p
    
    if index < remainder:
        start = index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (index - remainder) * base_size
        end = start + base_size
    
    return start, end


def load_features_for_block(
    dataset: str,
    global_indices: range,
    use_synthetic: bool = False,
    d: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Load feature vectors for the given global indices.
    
    Args:
        dataset: Dataset name ('mnist', 'year', or 'synthetic')
        global_indices: Range of global row indices
        use_synthetic: If True, use synthetic random features
        d: Feature dimension for synthetic data
        seed: Random seed
        
    Returns:
        Feature matrix of shape (len(global_indices), d)
    """
    if use_synthetic or dataset == 'synthetic':
        return generate_features(np.array(list(global_indices)), d, seed=seed)
    
    # Map dataset names to file paths and feature counts
    dataset_info = {
        'mnist': ('data/datasets/mnist.scale', 780),
        'year': ('data/datasets/YearPredictionMSD', 90),
    }
    
    if dataset not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist', 'year', or 'synthetic'")
    
    filepath, feature_count = dataset_info[dataset]
    
    if not os.path.exists(filepath):
        print(f"Warning: Dataset file not found: {filepath}. Using synthetic data.")
        return generate_features(np.array(list(global_indices)), feature_count, seed=seed)
    
    skip_lines = global_indices.start
    line_count = len(global_indices)
    
    return read_dataset(filepath, skip_lines, line_count, feature_count)


def run_single_experiment(
    comm: MPI.Comm,
    n: int,
    l: int,
    k: int,
    sigma: float,
    dataset: str = 'synthetic',
    d: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run a single Nyström experiment.
    
    Returns:
        Dictionary with results (error, timing, etc.)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    p = validate_process_count(size)
    if p is None:
        if rank == 0:
            print(f"Error: {size} is not a perfect square.", file=sys.stderr)
        comm.Abort(1)
    
    i, j = rank_to_grid_coords(rank, p)
    row_range = compute_block_range(i, p, n)
    col_range = compute_block_range(j, p, n)
    row_start, row_end = row_range
    col_start, col_end = col_range
    
    # Load features
    use_synthetic = (dataset == 'synthetic')
    row_indices = range(row_start, row_end)
    col_indices = range(col_start, col_end)
    
    X_rows = load_features_for_block(dataset, row_indices, use_synthetic, d)
    X_cols = load_features_for_block(dataset, col_indices, use_synthetic, d)
    
    # Compute local kernel block
    A_local = rbf_kernel_block(X_rows, X_cols, sigma)
    
    # Generate sketching matrix blocks
    Omega_row_local = generate_omega_block(row_indices, l, seed=42)
    Omega_col_block = generate_omega_block(col_indices, l, seed=42)
    
    # Create communicators
    row_comm = create_row_communicator(comm, i, j, p)
    col_comm = create_col_communicator(comm, i, j, p)
    
    # Run Nyström pipeline
    comm.Barrier()
    t_start = measure_time(comm)
    
    U_k_local, Lambda_k, info = compute_nystrom_full_pipeline(
        A_local=A_local,
        Omega_row_local=Omega_row_local,
        Omega_col_block=Omega_col_block,
        k=k,
        comm=comm,
        row_comm=row_comm,
        col_comm=col_comm,
        verbose=False,
        rank=rank,
    )
    
    t_end = measure_time(comm)
    elapsed = t_end - t_start
    
    # Compute error metrics
    is_diagonal_block = (i == j)
    nuclear_norm_A = estimate_nuclear_norm_A(comm, A_local, is_diagonal_block, n)
    rel_error = compute_relative_error(nuclear_norm_A, Lambda_k, U_k_local, col_comm)
    
    # Gather timing
    max_time = comm.allreduce(elapsed, op=MPI.MAX)
    
    # Clean up
    row_comm.Free()
    col_comm.Free()
    
    results = {
        'n': n,
        'l': l,
        'k': k,
        'sigma': sigma,
        'P': size,
        'time': max_time,
        'nuclear_norm_A': nuclear_norm_A,
        'nuclear_norm_approx': np.sum(Lambda_k),
        'relative_error': rel_error,
        'lambda_k': Lambda_k.tolist(),
    }
    
    if verbose and rank == 0:
        print(f"n={n}, l={l}, k={k}, sigma={sigma:.1f}, P={size}: "
              f"error={rel_error:.4f}, time={max_time:.4f}s")
    
    return results


def run_accuracy_experiments(comm: MPI.Comm, args):
    """Run experiments varying l, k, and sigma."""
    rank = comm.Get_rank()
    
    results = []
    
    # Experiment 1: Vary l (sketch dimension)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Experiment 1: Varying sketch dimension l")
        print("=" * 60)
    
    l_values = [20, 50, 100, 150, 200]
    for l in l_values:
        k = min(l // 2, 50)  # k <= l/2, capped at 50
        res = run_single_experiment(
            comm, n=args.n, l=l, k=k, sigma=args.sigma,
            dataset=args.dataset, d=args.d
        )
        res['experiment'] = 'vary_l'
        results.append(res)
    
    # Experiment 2: Vary k (target rank) with fixed l
    if rank == 0:
        print("\n" + "=" * 60)
        print("Experiment 2: Varying target rank k")
        print("=" * 60)
    
    l_fixed = 100
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for k in k_values:
        if k <= l_fixed:
            res = run_single_experiment(
                comm, n=args.n, l=l_fixed, k=k, sigma=args.sigma,
                dataset=args.dataset, d=args.d
            )
            res['experiment'] = 'vary_k'
            results.append(res)
    
    # Experiment 3: Vary sigma (kernel bandwidth)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Experiment 3: Varying kernel bandwidth sigma")
        print("=" * 60)
    
    if args.dataset == 'mnist':
        sigma_values = [10, 50, 100, 200, 500]
    elif args.dataset == 'year':
        sigma_values = [1e3, 1e4, 5e4, 1e5, 5e5]
    else:
        sigma_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for sigma in sigma_values:
        res = run_single_experiment(
            comm, n=args.n, l=100, k=50, sigma=sigma,
            dataset=args.dataset, d=args.d
        )
        res['experiment'] = 'vary_sigma'
        results.append(res)
    
    return results


def run_scaling_experiments(comm: MPI.Comm, args):
    """
    Run scaling experiments.
    
    Note: This measures scaling within a single run.
    For true weak/strong scaling, run the script multiple times with different -n P.
    """
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Scaling Experiment")
        print("=" * 60)
    
    results = run_single_experiment(
        comm, n=args.n, l=args.l, k=args.k, sigma=args.sigma,
        dataset=args.dataset, d=args.d
    )
    results['experiment'] = 'scaling'
    
    return [results]


def main():
    parser = argparse.ArgumentParser(description='Randomized Nystrom Experiments')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['mnist', 'year', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--n', type=int, default=1000,
                        help='Matrix dimension')
    parser.add_argument('--l', type=int, default=100,
                        help='Sketch dimension')
    parser.add_argument('--k', type=int, default=50,
                        help='Target rank')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Kernel bandwidth (default varies by dataset)')
    parser.add_argument('--d', type=int, default=10,
                        help='Feature dimension for synthetic data')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'accuracy', 'scaling'],
                        help='Experiment mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Set default sigma based on dataset
    if args.sigma == 1.0:  # Default value
        if args.dataset == 'mnist':
            args.sigma = 100.0
        elif args.dataset == 'year':
            args.sigma = 1e4
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("Randomized Nystrom Experiment Runner")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"n = {args.n}, l = {args.l}, k = {args.k}")
        print(f"sigma = {args.sigma}")
        print(f"Mode: {args.mode}")
        print(f"MPI processes: {comm.Get_size()}")
        print("=" * 60)
    
    # Run experiments
    if args.mode == 'single':
        results = [run_single_experiment(
            comm, n=args.n, l=args.l, k=args.k, sigma=args.sigma,
            dataset=args.dataset, d=args.d
        )]
    elif args.mode == 'accuracy':
        results = run_accuracy_experiments(comm, args)
    elif args.mode == 'scaling':
        results = run_scaling_experiments(comm, args)
    
    # Save results
    if args.output and rank == 0:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Experiments complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()
