"""
Sketching matrix construction for Randomized Nyström approximation.

This module provides functions for generating the random sketching matrix Ω
used in the Nyström low-rank approximation.

The sketching matrix is distributed by rows: each MPI rank owns the rows
of Ω corresponding to its owned rows of the kernel matrix A.
"""

import numpy as np


def generate_omega_block(
    global_row_indices: range,
    l: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate the local block of the Gaussian sketching matrix Ω.
    
    Each rank generates only its owned rows of Ω. The construction is
    deterministic: the same global row index always produces the same
    row of Ω, regardless of which rank generates it or execution order.
    
    Mathematical context:
        Ω ∈ R^{n × l} is the full sketching matrix
        Ω_local = Ω[I_i, :] where I_i are the global row indices owned by this rank
    
    Args:
        global_row_indices: Range of global row indices owned by this rank.
                            Example: range(500, 1000)
        l: Sketch dimension (number of columns in Ω)
        seed: Global base seed for reproducibility (default: 42)
        
    Returns:
        Omega_local: NumPy array of shape (len(global_row_indices), l), dtype float64
                     Each row is a standard normal random vector.
    """
    n_local = len(global_row_indices)
    
    # Pre-allocate output matrix
    Omega_local = np.zeros((n_local, l), dtype=np.float64)
    
    # Generate each row deterministically using (seed + global_index)
    for local_idx, global_idx in enumerate(global_row_indices):
        rng = np.random.default_rng(seed + global_idx)
        Omega_local[local_idx, :] = rng.standard_normal(l)
    
    return Omega_local
