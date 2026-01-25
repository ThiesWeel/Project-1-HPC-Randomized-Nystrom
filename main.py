"""
MPI Skeleton for Randomized Nyström Low-Rank Approximation

This module implements the distributed data layout for a randomized Nyström
approximation of a dense symmetric positive semidefinite matrix.

Current implementation: MPI process grid and block distribution only.
Future work: RBF kernel construction, A @ Omega computation, Nyström truncation.
"""

from mpi4py import MPI
import sys
import math
import numpy as np
import time
from data.kernel import generate_features, rbf_kernel_block
from data.sketch import generate_omega_block
from nystrom import (
    create_row_communicator,
    create_col_communicator,
    compute_nystrom_full_pipeline,
    estimate_nuclear_norm_A,
    compute_relative_error,
    measure_time,
)


def validate_process_count(size):
    """
    Validate that the number of MPI processes is a perfect square.
    
    Args:
        size: Total number of MPI processes
        
    Returns:
        p: Square root of size (grid dimension)
        
    Aborts if size is not a perfect square.
    """
    p = int(math.sqrt(size))
    if p * p != size:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Error: Number of MPI processes must be a perfect square.", file=sys.stderr)
            print(f"       Requested {size} processes, but {size} is not a perfect square.", file=sys.stderr)
            print(f"       Try using 1, 4, 9, 16, 25, 36, ... processes.", file=sys.stderr)
        MPI.COMM_WORLD.Abort(1)
    return p


def rank_to_grid_coords(rank, p):
    """
    Map MPI rank to 2D grid coordinates using row-major ordering.
    
    Args:
        rank: MPI rank (0 to p^2 - 1)
        p: Grid dimension
        
    Returns:
        (i, j): Grid coordinates where i is row index, j is column index
    """
    i = rank // p  # Row index
    j = rank % p   # Column index
    return i, j


def compute_block_range(index, p, n):
    """
    Compute the global index range owned by a process at grid position 'index'.
    
    Args:
        index: Grid index (i for rows, j for columns)
        p: Grid dimension
        n: Total matrix dimension
        
    Returns:
        (start, end): Half-open interval [start, end) of global indices
    """
    # Base block size
    base_size = n // p
    # Number of processes that get an extra element
    remainder = n % p
    
    if index < remainder:
        # This process gets an extra element
        start = index * (base_size + 1)
        end = start + base_size + 1
    else:
        # This process gets base_size elements
        start = remainder * (base_size + 1) + (index - remainder) * base_size
        end = start + base_size
    
    return start, end


def print_block_distribution(rank, size, p, i, j, row_range, col_range, n):
    """
    Print block distribution information for each rank in order.
    
    Args:
        rank: MPI rank
        size: Total number of processes
        p: Grid dimension
        i, j: Grid coordinates
        row_range: (start, end) for owned rows
        col_range: (start, end) for owned columns
        n: Total matrix dimension
    """
    comm = MPI.COMM_WORLD
    
    # Rank 0 prints header
    if rank == 0:
        print("=" * 80)
        print(f"MPI Block Distribution for {n} x {n} matrix on {p} x {p} process grid")
        print("=" * 80)
        print()
    
    # Each rank prints in order
    for r in range(size):
        if rank == r:
            row_start, row_end = row_range
            col_start, col_end = col_range
            row_count = row_end - row_start
            col_count = col_end - col_start
            
            print(f"Rank {rank:2d}:")
            print(f"  Grid coordinates:  ({i}, {j})")
            print(f"  Row range:         [{row_start:4d}, {row_end:4d})  (count: {row_count})")
            print(f"  Column range:      [{col_start:4d}, {col_end:4d})  (count: {col_count})")
            print()
            sys.stdout.flush()
        
        # Barrier to ensure ordered printing
        comm.Barrier()
    
    # Rank 0 prints footer
    if rank == 0:
        print("=" * 80)
        print("Block distribution complete and verified.")
        print("=" * 80)


def main():
    """
    Main entry point: Initialize MPI and demonstrate block distribution.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse command line arguments (basic - for full experiments use run_experiments.py)
    import argparse
    parser = argparse.ArgumentParser(description='Randomized Nystrom Demo')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['mnist', 'year', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--n', type=int, default=1000,
                        help='Matrix dimension')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Kernel bandwidth (auto-selected if not provided)')
    parser.add_argument('--use-cache-blocking', action='store_true',
                        help='Enable L2 cache blocking (tile=256)')
    args = parser.parse_args()
    
    # Validate that size is a perfect square
    p = validate_process_count(size)
    
    # Map rank to grid coordinates
    i, j = rank_to_grid_coords(rank, p)
    
    # Global matrix dimension
    n = args.n
    
    # Compute owned row and column ranges
    row_range = compute_block_range(i, p, n)
    col_range = compute_block_range(j, p, n)
    
    # Print block distribution for all ranks
    print_block_distribution(rank, size, p, i, j, row_range, col_range, n)
    
    # ========================================================================
    # Step 1: Local kernel block construction
    # ========================================================================
    
    # Feature dimension and RBF kernel bandwidth
    d = 10      # Feature dimension
    
    # Set sigma/c based on dataset (per project specifications)
    if args.sigma is not None:
        sigma = args.sigma
    else:
        # Auto-select based on dataset
        if args.dataset == 'mnist':
            sigma = 100.0  # c = 100 for MNIST
        elif args.dataset == 'year':
            sigma = 1e4    # c = 10,000 for YearPredictionMSD (can also use 1e5)
        else:
            sigma = 1.0    # Default for synthetic data
    
    # Generate feature vectors for local rows and columns
    row_start, row_end = row_range
    col_start, col_end = col_range
    
    row_indices = np.arange(row_start, row_end)
    col_indices = np.arange(col_start, col_end)
    
    X_rows = generate_features(row_indices, d, seed=42)
    X_cols = generate_features(col_indices, d, seed=42)
    
    # Compute local kernel block with optional cache blocking
    A_local = rbf_kernel_block(X_rows, X_cols, sigma, use_cache_blocking=args.use_cache_blocking)
    
    # Print local block shape
    comm.Barrier()
    if rank == 0:
        print("\n" + "=" * 80)
        print("Local Kernel Block Construction (Step 1)")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Matrix dimension n = {n}")
        print(f"Feature dimension d = {d}")
        print(f"RBF bandwidth c (sigma) = {sigma}")
        print("=" * 80)
        print()
    
    for r in range(size):
        if rank == r:
            print(f"Rank {rank}: A_local shape = {A_local.shape}")
            sys.stdout.flush()
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Step 1 complete: Local kernel blocks constructed.")
        print("=" * 80)
    
    # ========================================================================
    # Step 2: Sketching matrix Omega construction
    # ========================================================================
    
    # Sketch dimension and target rank
    l = 50   # Sketch dimension (k <= l << n)
    k = 20   # Target rank for truncation
    
    # Generate local block of Omega (rows owned by this rank)
    row_indices_range = range(row_start, row_end)
    Omega_row_local = generate_omega_block(row_indices_range, l, seed=42)
    
    # Also generate Omega block for column indices (needed for A @ Omega)
    col_indices_range = range(col_start, col_end)
    Omega_col_block = generate_omega_block(col_indices_range, l, seed=42)
    
    # Print local Omega block shape
    comm.Barrier()
    if rank == 0:
        print("\n" + "=" * 80)
        print("Sketching Matrix Construction (Step 2)")
        print("=" * 80)
        print(f"Sketch dimension l = {l}")
        print(f"Target rank k = {k}")
        print("=" * 80)
        print()
    
    for r in range(size):
        if rank == r:
            print(f"Rank {rank}: Omega_row_local shape = {Omega_row_local.shape}, "
                  f"Omega_col_block shape = {Omega_col_block.shape}")
            sys.stdout.flush()
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Step 2 complete: Sketching matrix blocks constructed.")
        print("=" * 80)
    
    # ========================================================================
    # Create MPI communicators for row and column operations
    # ========================================================================
    
    row_comm = create_row_communicator(comm, i, j, p)
    col_comm = create_col_communicator(comm, i, j, p)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MPI Communicators Created")
        print("=" * 80)
        print(f"Row communicator size: {row_comm.Get_size()}")
        print(f"Column communicator size: {col_comm.Get_size()}")
        print("=" * 80)
    
    # ========================================================================
    # Steps 3-7: Full Nystrom Pipeline
    # ========================================================================
    
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
        verbose=True,
        rank=rank,
    )
    
    t_end = measure_time(comm)
    elapsed = t_end - t_start
    
    comm.Barrier()
    if rank == 0:
        print("\n" + "=" * 80)
        print("Nystrom Approximation Complete (Steps 3-7)")
        print("=" * 80)
        print(f"Y_local shape: {info['Y_local_shape']}")
        print(f"B shape: {info['B_shape']}")
        print(f"Z_local shape: {info['Z_local_shape']}")
        print(f"U_k_local shape: {U_k_local.shape}")
        print(f"Lambda_k (top {k} eigenvalues): {Lambda_k[:min(5, k)]}...")
        print(f"Time elapsed: {elapsed:.4f} seconds")
        print("=" * 80)
    
    # ========================================================================
    # Step 8: Error computation
    # ========================================================================
    
    # Compute nuclear norm of A (trace for SPSD)
    is_diagonal_block = (i == j)
    nuclear_norm_A = estimate_nuclear_norm_A(comm, A_local, is_diagonal_block, n)
    
    # Compute relative error
    rel_error = compute_relative_error(nuclear_norm_A, Lambda_k, U_k_local, col_comm)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Error Analysis (Step 8)")
        print("=" * 80)
        print(f"||A||_* (nuclear norm via trace): {nuclear_norm_A:.6f}")
        print(f"||[[A_Nyst]]_k||_* (approx): {np.sum(Lambda_k):.6f}")
        print(f"Relative error (approx): {rel_error:.6f}")
        print("=" * 80)
    
    # ========================================================================
    # Step 9: Performance summary
    # ========================================================================
    
    # Gather timing from all ranks
    all_times = comm.gather(elapsed, root=0)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Performance Summary (Step 9)")
        print("=" * 80)
        print(f"Number of MPI processes: {size}")
        print(f"Process grid: {p} x {p}")
        print(f"Matrix dimension n = {n}")
        print(f"Sketch dimension l = {l}")
        print(f"Target rank k = {k}")
        print(f"Feature dimension d = {d}")
        print()
        print(f"Max time across ranks: {max(all_times):.4f} seconds")
        print(f"Min time across ranks: {min(all_times):.4f} seconds")
        print(f"Avg time across ranks: {np.mean(all_times):.4f} seconds")
        print("=" * 80)
        print("\nRandomized Nystrom approximation complete!")
        print("=" * 80)
    
    # Clean up communicators
    row_comm.Free()
    col_comm.Free()


if __name__ == "__main__":
    main()
