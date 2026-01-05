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
from data.kernel import generate_features, rbf_kernel_block


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
    
    # Validate that size is a perfect square
    p = validate_process_count(size)
    
    # Map rank to grid coordinates
    i, j = rank_to_grid_coords(rank, p)
    
    # Global matrix dimension (can be modified or read from command line)
    n = 1000  # Example: 1000 x 1000 matrix
    
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
    sigma = 1.0 # RBF bandwidth parameter
    
    # Generate feature vectors for local rows and columns
    row_start, row_end = row_range
    col_start, col_end = col_range
    
    row_indices = np.arange(row_start, row_end)
    col_indices = np.arange(col_start, col_end)
    
    X_rows = generate_features(row_indices, d, seed=42)
    X_cols = generate_features(col_indices, d, seed=42)
    
    # Compute local kernel block
    A_local = rbf_kernel_block(X_rows, X_cols, sigma)
    
    # Print local block shape
    comm.Barrier()
    if rank == 0:
        print("\n" + "=" * 80)
        print("Local Kernel Block Construction (Step 1)")
        print("=" * 80)
        print(f"Feature dimension d = {d}")
        print(f"RBF bandwidth sigma = {sigma}")
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


if __name__ == "__main__":
    main()
