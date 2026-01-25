"""
Sketching matrix construction for Randomized Nyström approximation.

This module provides functions for generating the random sketching matrix Ω
used in the Nyström low-rank approximation.

The sketching matrix is distributed by rows: each MPI rank owns the rows
of Ω corresponding to its owned rows of the kernel matrix A.

Supported sketching methods:
1. Gaussian: Dense random matrix with i.i.d. N(0,1) entries
2. SRHT: Subsampled Randomized Hadamard Transform (faster application)
"""

import numpy as np
from typing import Literal, Optional


# Gaussian sketching

def generate_omega_block(
    global_row_indices: range,
    l: int,
    seed: int = 42,
    method: Literal['gaussian', 'srht', 'sparse'] = 'gaussian',
    sparsity: float = 0.1,
) -> np.ndarray:
    """
    Generate the local block of the sketching matrix Ω.
    
    Each rank generates only its owned rows of Ω. The construction is
    deterministic: the same global row index always produces the same
    row of Ω, regardless of which rank generates it or execution order.
    
    Args:
        global_row_indices: Range of global row indices owned by this rank.
        l: Sketch dimension (number of columns in Ω)
        seed: Global base seed for reproducibility (default: 42)
        method: Sketching method ('gaussian', 'srht', or 'sparse')
        sparsity: For sparse sketching, fraction of non-zero entries (default: 0.1)
        
    Returns:
        Omega_local: NumPy array of shape (len(global_row_indices), l), dtype float64
    """
    if method == 'gaussian':
        return generate_gaussian_block(global_row_indices, l, seed)
    elif method == 'srht':
        return generate_srht_block(global_row_indices, l, seed)
    elif method == 'sparse':
        return generate_sparse_block(global_row_indices, l, seed, sparsity)
    else:
        raise ValueError(f"Unknown sketching method: {method}. Use 'gaussian', 'srht', or 'sparse'.")


def generate_gaussian_block(
    global_row_indices: range,
    l: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate the local block of the Gaussian sketching matrix Ω.
    
    Each row is a standard normal random vector.
    
    Mathematical context:
        Ω ∈ R^{n × l} is the full sketching matrix
        Ω_local = Ω[I_i, :] where I_i are the global row indices owned by this rank
    
    Args:
        global_row_indices: Range of global row indices owned by this rank.
        l: Sketch dimension (number of columns in Ω)
        seed: Global base seed for reproducibility (default: 42)
        
    Returns:
        Omega_local: NumPy array of shape (len(global_row_indices), l), dtype float64
    """
    n_local = len(global_row_indices)
    
    # Pre-allocate output matrix
    Omega_local = np.zeros((n_local, l), dtype=np.float64)
    
    # Generate each row deterministically using (seed + global_index)
    for local_idx, global_idx in enumerate(global_row_indices):
        rng = np.random.default_rng(seed + global_idx)
        Omega_local[local_idx, :] = rng.standard_normal(l)
    
    return Omega_local


# SRHT sketching (Subsampled Randomized Hadamard Transform)

def hadamard_transform_power_of_2(x: np.ndarray) -> np.ndarray:
    """
    Apply the Hadamard transform to a vector of length 2^k.
    
    Uses the fast Walsh-Hadamard transform (O(n log n)).
    
    Args:
        x: Input vector of length 2^k
        
    Returns:
        Hadamard-transformed vector
    """
    n = len(x)
    
    # Fast Walsh-Hadamard transform
    result = x.copy().astype(np.float64)
    
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    
    # Normalize
    return result / np.sqrt(n)


def generate_srht_block(
    global_row_indices: range,
    l: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate the local block of the SRHT (Subsampled Randomized Hadamard Transform) matrix.
    
    SRHT is faster to apply than Gaussian: O(n log l) vs O(nl).
    
    The SRHT matrix is: Ω = sqrt(n/l) * D * H * S
    where:
        - D is a diagonal matrix of random signs
        - H is the Hadamard transform matrix
        - S is a subsampling matrix (selects l columns)
    
    For distributed computation, we generate local rows directly.
    
    Args:
        global_row_indices: Range of global row indices
        l: Sketch dimension
        seed: Random seed
        
    Returns:
        Omega_local: Shape (len(global_row_indices), l)
    """
    n_local = len(global_row_indices)
    global_start = global_row_indices.start if hasattr(global_row_indices, 'start') else min(global_row_indices)
    global_end = global_row_indices.stop if hasattr(global_row_indices, 'stop') else max(global_row_indices) + 1
    n_global_approx = global_end  # Approximation of global n
    
    # Use a master RNG for consistent column selection across ranks
    master_rng = np.random.default_rng(seed)
    
    # Find next power of 2 >= l for Hadamard transform
    l_padded = 1
    while l_padded < l:
        l_padded *= 2
    
    # Select l columns from the Hadamard matrix
    selected_cols = master_rng.choice(l_padded, size=l, replace=False)
    selected_cols.sort()
    
    # Generate random signs for D (deterministic per row)
    Omega_local = np.zeros((n_local, l), dtype=np.float64)
    
    for local_idx, global_idx in enumerate(global_row_indices):
        # Random sign for this row
        row_rng = np.random.default_rng(seed + global_idx + 1000000)
        sign = row_rng.choice([-1.0, 1.0])
        
        # Generate row of Hadamard matrix at selected columns
        # H[i, j] = (-1)^(popcount(i & j)) / sqrt(n)
        # For efficiency, we compute only the selected columns
        
        for col_idx, j in enumerate(selected_cols):
            # Hadamard entry: (-1)^(popcount(global_idx & j))
            # popcount counts the number of 1 bits in (i & j)
            bit_and = global_idx & j
            popcount = bin(bit_and).count('1')
            h_entry = (-1.0) ** popcount
            
            # Apply sign and scaling
            Omega_local[local_idx, col_idx] = sign * h_entry / np.sqrt(l_padded)
    
    # Scale by sqrt(n/l) - but we use normalized version
    scale = np.sqrt(n_local)  # Local scaling
    Omega_local *= scale / np.sqrt(l)
    
    return Omega_local


# Sparse sketching

def generate_sparse_block(
    global_row_indices: range,
    l: int,
    seed: int = 42,
    sparsity: float = 0.1,
) -> np.ndarray:
    """
    Generate a sparse sketching matrix.
    
    Each entry is non-zero with probability 'sparsity'.
    Non-zero entries are ±1/sqrt(sparsity * l).
    
    Advantages:
        - Memory efficient for large l
        - Fast matrix-vector products
    
    Args:
        global_row_indices: Range of global row indices
        l: Sketch dimension
        seed: Random seed
        sparsity: Fraction of non-zero entries (default: 0.1)
        
    Returns:
        Omega_local: Shape (len(global_row_indices), l)
    """
    n_local = len(global_row_indices)
    Omega_local = np.zeros((n_local, l), dtype=np.float64)
    
    # Scaling factor for non-zero entries
    scale = 1.0 / np.sqrt(sparsity * l)
    
    for local_idx, global_idx in enumerate(global_row_indices):
        rng = np.random.default_rng(seed + global_idx)
        
        # Decide which entries are non-zero
        mask = rng.random(l) < sparsity
        
        # Random signs for non-zero entries
        signs = rng.choice([-1.0, 1.0], size=l)
        
        Omega_local[local_idx, mask] = scale * signs[mask]
    
    return Omega_local


# Utility functions

def get_sketching_method_info(method: str) -> dict:
    """
    Get information about a sketching method.
    
    Args:
        method: Sketching method name
        
    Returns:
        Dictionary with method properties
    """
    info = {
        'gaussian': {
            'name': 'Gaussian',
            'description': 'Dense random matrix with i.i.d. N(0,1) entries',
            'time_complexity': 'O(nl) to apply',
            'space_complexity': 'O(nl) dense',
            'pros': ['Strong theoretical guarantees', 'Simple implementation'],
            'cons': ['Slow for large l', 'High memory usage'],
        },
        'srht': {
            'name': 'Subsampled Randomized Hadamard Transform',
            'description': 'Fast structured random matrix using Hadamard transform',
            'time_complexity': 'O(n log l) to apply',
            'space_complexity': 'O(l) storage for indices',
            'pros': ['Fast application', 'Low memory', 'Good approximation'],
            'cons': ['Requires power-of-2 sizes', 'Slightly weaker guarantees'],
        },
        'sparse': {
            'name': 'Sparse Sketching',
            'description': 'Sparse random matrix with ±1 entries',
            'time_complexity': 'O(nnz) to apply where nnz = sparsity * nl',
            'space_complexity': 'O(sparsity * nl)',
            'pros': ['Memory efficient', 'Fast for low sparsity'],
            'cons': ['May need larger l for same accuracy'],
        },
    }
    
    return info.get(method, {'name': 'Unknown', 'description': 'Unknown method'})


def compare_sketching_methods(
    n: int,
    l: int,
    seed: int = 42,
) -> dict:
    """
    Generate and compare different sketching methods.
    
    Returns timing and memory statistics for each method.
    
    Args:
        n: Number of rows to generate
        l: Sketch dimension
        seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    import time
    
    results = {}
    indices = range(n)
    
    for method in ['gaussian', 'srht', 'sparse']:
        start = time.perf_counter()
        Omega = generate_omega_block(indices, l, seed, method=method)
        elapsed = time.perf_counter() - start
        
        # Count non-zeros for sparse
        nnz = np.count_nonzero(Omega)
        
        results[method] = {
            'time': elapsed,
            'shape': Omega.shape,
            'nnz': nnz,
            'density': nnz / (n * l),
            'memory_mb': Omega.nbytes / (1024 * 1024),
            'frobenius_norm': np.linalg.norm(Omega, 'fro'),
        }
    
    return results
