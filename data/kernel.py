"""
Kernel matrix construction for Randomized Nyström approximation.

This module provides functions for:
1. Deterministic feature vector generation
2. RBF (Gaussian) kernel block computation

No MPI communication is performed in this module - each rank computes
its local kernel block independently.
"""

import numpy as np


def generate_features(global_indices, d, seed=42):
    """
    NOTE: simple placeholder for MPI skeleton debugging.
    Generate deterministic feature vectors for given global indices.
    
    Each global index i produces the same feature vector x_i ∈ R^d
    regardless of which MPI rank generates it.
    
    Args:
        global_indices: Array-like of global row/column indices
        d: Feature dimension
        seed: Base random seed for reproducibility
        
    Returns:
        X: NumPy array of shape (len(global_indices), d)
           Each row is a feature vector
    """
    n = len(global_indices)
    X = np.zeros((n, d))
    
    for i, global_idx in enumerate(global_indices):
        # Use global index to seed the RNG for this specific feature vector
        rng = np.random.RandomState(seed + global_idx)
        X[i, :] = rng.randn(d)
    
    return X


def rbf_kernel_block(X_rows, X_cols, sigma):
    """
    Compute RBF (Gaussian) kernel block between two sets of feature vectors.
    
    K[i, j] = exp(-||x_i - y_j||^2 / sigma^2)
    
    Uses vectorized computation:
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
    
    Args:
        X_rows: NumPy array of shape (n_rows, d) - row features
        X_cols: NumPy array of shape (n_cols, d) - column features
        sigma: RBF kernel bandwidth parameter
        
    Returns:
        K: NumPy array of shape (n_rows, n_cols)
           Kernel matrix block
    """
    # Ensure feature dimensions match
    assert X_rows.shape[1] == X_cols.shape[1], \
        f"Feature dimensions must match: {X_rows.shape[1]} != {X_cols.shape[1]}"
    
    # Compute squared norms
    # X_rows_sq: (n_rows, 1)
    X_rows_sq = np.sum(X_rows**2, axis=1, keepdims=True)
    
    # X_cols_sq: (1, n_cols)
    X_cols_sq = np.sum(X_cols**2, axis=1, keepdims=True).T
    
    # Cross term: (n_rows, n_cols)
    X_cross = X_rows @ X_cols.T
    
    # Squared distances: ||x - y||^2
    # Broadcasting: (n_rows, 1) + (1, n_cols) - (n_rows, n_cols)
    sq_distances = X_rows_sq + X_cols_sq - 2 * X_cross
    
    # Ensure numerical stability (distances should be non-negative)
    sq_distances = np.maximum(sq_distances, 0.0)
    
    # RBF kernel
    K = np.exp(-sq_distances / (sigma**2))
    
    return K
