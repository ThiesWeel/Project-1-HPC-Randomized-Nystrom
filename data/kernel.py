"""
Kernel matrix construction for Randomized Nystrom approximation.

Provides:
- RBF (Gaussian) kernel block computation
- Chunked kernel computation for memory efficiency
- L2 cache-blocked kernel computation
"""

import numpy as np
from typing import Optional, Callable
import gc


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


def rbf_kernel_block_naive(X_rows, X_cols, sigma):
    """
    Compute RBF kernel: K(x, y) = exp(-||x - y||^2 / sigma^2)
    
    Standard vectorized version without cache optimization.
    Good for small matrices, but causes cache misses for large ones.
    
    Args:
        X_rows: (n_rows, d) feature matrix
        X_cols: (n_cols, d) feature matrix
        sigma: Bandwidth parameter (c in project spec)
    
    Returns:
        K: (n_rows, n_cols) kernel matrix
    """
    # Use identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    row_sq_norms = np.sum(X_rows**2, axis=1, keepdims=True)
    col_sq_norms = np.sum(X_cols**2, axis=1, keepdims=True)
    
    sq_dists = row_sq_norms + col_sq_norms.T - 2 * X_rows @ X_cols.T
    sq_dists = np.maximum(sq_dists, 0.0)  # Fix numerical errors
    
    return np.exp(-sq_dists / (sigma * sigma))


def rbf_kernel_block_cache_blocked(X_rows, X_cols, sigma):
    """
    Compute RBF kernel with L2 cache blocking.
    
    Optimized for i7-11800H (1.25 MB L2 cache).
    Uses 256×256 tiles (512 KB) to fit in L2 cache.
    
    Reduces cache misses by processing small tiles that stay in cache.
    
    Args:
        X_rows: (n_rows, d) feature matrix
        X_cols: (n_cols, d) feature matrix
        sigma: Bandwidth parameter
    
    Returns:
        K: (n_rows, n_cols) kernel matrix
    """
    n_rows, d = X_rows.shape
    n_cols = X_cols.shape[0]
    
    TILE_SIZE = 256  # 256×256 × 8 bytes = 512 KB < 800 KB target
    
    # Precompute norms once
    row_sq_norms = np.sum(X_rows**2, axis=1)
    col_sq_norms = np.sum(X_cols**2, axis=1)
    
    K = np.empty((n_rows, n_cols), dtype=np.float64)
    sigma_sq = sigma * sigma
    
    # Process in tiles
    for i in range(0, n_rows, TILE_SIZE):
        i_end = min(i + TILE_SIZE, n_rows)
        X_tile_rows = X_rows[i:i_end, :]
        tile_row_norms = row_sq_norms[i:i_end]
        
        for j in range(0, n_cols, TILE_SIZE):
            j_end = min(j + TILE_SIZE, n_cols)
            X_tile_cols = X_cols[j:j_end, :]
            tile_col_norms = col_sq_norms[j:j_end]
            
            # Compute tile (fits in L2 cache)
            sq_dists = (tile_row_norms[:, None] + tile_col_norms[None, :] 
                        - 2 * X_tile_rows @ X_tile_cols.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            
            K[i:i_end, j:j_end] = np.exp(-sq_dists / sigma_sq)
    
    return K


def rbf_kernel_block(X_rows, X_cols, sigma, use_cache_blocking=False):
    """
    Compute RBF kernel with optional cache blocking.
    
    Args:
        X_rows: (n_rows, d) feature matrix
        X_cols: (n_cols, d) feature matrix
        sigma: Bandwidth parameter
        use_cache_blocking: If True, use L2-optimized version
    
    Returns:
        K: (n_rows, n_cols) kernel matrix
    """
    if use_cache_blocking:
        return rbf_kernel_block_cache_blocked(X_rows, X_cols, sigma)
    else:
        return rbf_kernel_block_naive(X_rows, X_cols, sigma)


# Chunked kernel computation for memory efficiency

def rbf_kernel_block_chunked(
    X_rows: np.ndarray,
    X_cols: np.ndarray,
    sigma: float,
    chunk_size: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute RBF kernel in chunks to reduce peak memory usage.
    
    Instead of forming the full (n_rows x n_cols) intermediate matrices
    at once, compute in chunks of size (chunk_size x n_cols).
    
    This enables processing of larger matrices on limited RAM.
    
    Memory optimization:
        - Full computation: O(n_rows * n_cols) peak memory
        - Chunked computation: O(chunk_size * n_cols) peak memory
    
    Args:
        X_rows: Feature matrix for rows, shape (n_rows, d)
        X_cols: Feature matrix for columns, shape (n_cols, d)
        sigma: RBF kernel bandwidth parameter
        chunk_size: Number of rows to process at once (default: 1000)
        verbose: Print progress information
        
    Returns:
        K: Kernel matrix of shape (n_rows, n_cols)
    """
    n_rows = X_rows.shape[0]
    n_cols = X_cols.shape[0]
    
    # Pre-allocate output matrix
    K = np.empty((n_rows, n_cols), dtype=np.float64)
    
    # Pre-compute column squared norms (shared across all chunks)
    X_cols_sq = np.sum(X_cols**2, axis=1, keepdims=True).T  # (1, n_cols)
    
    # Process in chunks
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, n_rows)
        
        if verbose:
            print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} (rows {start_row}:{end_row})")
        
        # Compute chunk using naive method
        K[start_row:end_row, :] = rbf_kernel_block_naive(
            X_rows[start_row:end_row, :], X_cols, sigma
        )
    
    return K


def estimate_kernel_memory(n_rows: int, n_cols: int, dtype=np.float64) -> dict:
    """
    Estimate memory requirements for kernel computation.
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        dtype: Data type (default: float64 = 8 bytes)
        
    Returns:
        Dictionary with memory estimates
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Output matrix
    output_bytes = n_rows * n_cols * bytes_per_element
    
    # Intermediate matrices (worst case: all at once)
    # X_rows_sq: (n_rows, 1)
    # X_cols_sq: (1, n_cols)
    # X_cross: (n_rows, n_cols)
    # sq_distances: (n_rows, n_cols)
    intermediate_bytes = (
        n_rows * bytes_per_element +  # X_rows_sq
        n_cols * bytes_per_element +  # X_cols_sq
        n_rows * n_cols * bytes_per_element +  # X_cross
        n_rows * n_cols * bytes_per_element    # sq_distances
    )
    
    peak_bytes = output_bytes + intermediate_bytes
    
    return {
        'output_mb': output_bytes / (1024 * 1024),
        'intermediate_mb': intermediate_bytes / (1024 * 1024),
        'peak_mb': peak_bytes / (1024 * 1024),
        'peak_gb': peak_bytes / (1024 * 1024 * 1024),
    }


def recommend_chunk_size(
    n_rows: int,
    n_cols: int,
    available_memory_mb: float = 1024,
    dtype=np.float64,
) -> int:
    """
    Recommend chunk size based on available memory.
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        available_memory_mb: Available RAM in MB (default: 1024 = 1 GB)
        dtype: Data type
        
    Returns:
        Recommended chunk size
    """
    bytes_per_element = np.dtype(dtype).itemsize
    available_bytes = available_memory_mb * 1024 * 1024
    
    # Memory per row of output + intermediates
    # For each chunk row: n_cols (output) + n_cols (cross) + n_cols (sq_dist) + 1 (row_sq)
    bytes_per_chunk_row = n_cols * bytes_per_element * 3 + bytes_per_element
    
    # Reserve 50% for other operations
    usable_bytes = available_bytes * 0.5
    
    chunk_size = int(usable_bytes / bytes_per_chunk_row)
    
    # Clamp to reasonable range
    chunk_size = max(100, min(chunk_size, n_rows))
    
    return chunk_size


# Memory-efficient kernel-Omega product

def kernel_omega_product_chunked(
    X_rows: np.ndarray,
    X_all: np.ndarray,
    Omega: np.ndarray,
    sigma: float,
    chunk_size: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute Y = A @ Omega without forming full kernel matrix A.
    
    Instead of:
        A = rbf_kernel(X_rows, X_all)  # n_rows x n memory
        Y = A @ Omega                   # Another matrix multiply
    
    We compute:
        Y[i, :] = sum_j K(x_i, x_j) * Omega[j, :]
    
    In chunks to save memory.
    
    Args:
        X_rows: Row features, shape (n_rows, d)
        X_all: All features (for columns), shape (n, d)
        Omega: Sketching matrix, shape (n, l)
        sigma: RBF kernel bandwidth
        chunk_size: Chunk size for rows
        verbose: Print progress
        
    Returns:
        Y: Matrix of shape (n_rows, l)
    """
    n_rows = X_rows.shape[0]
    n = X_all.shape[0]
    l = Omega.shape[1]
    
    # Pre-allocate output
    Y = np.zeros((n_rows, l), dtype=np.float64)
    
    # Pre-compute column squared norms
    X_all_sq = np.sum(X_all**2, axis=1, keepdims=True).T  # (1, n)
    
    # Process in row chunks
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, n_rows)
        
        if verbose:
            print(f"  Y chunk {chunk_idx + 1}/{n_chunks}")
        
        # Compute kernel chunk using naive method and multiply
        K_chunk = rbf_kernel_block_naive(X_rows[start_row:end_row, :], X_all, sigma)
        Y[start_row:end_row, :] = K_chunk @ Omega
        
        del K_chunk
        gc.collect()
    
    return Y


# Memory profiling utilities

def profile_memory_usage():
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory statistics
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'available': True,
        }
    except ImportError:
        return {
            'available': False,
            'message': 'psutil not installed - memory profiling unavailable',
        }


def compare_kernel_methods(X_rows, X_cols, sigma):
    """
    Compare naive vs cache-blocked kernel computation.
    
    Args:
        X_rows: (n_rows, d) feature matrix
        X_cols: (n_cols, d) feature matrix
        sigma: Bandwidth parameter
    
    Returns:
        Dictionary with timing and accuracy results
    """
    import time
    
    n_rows, n_cols = X_rows.shape[0], X_cols.shape[0]
    mem = estimate_kernel_memory(n_rows, n_cols)
    mem_mb = mem['output_mb']
    print(f"Testing {n_rows}×{n_cols} matrix ({mem_mb:.1f} MB)")
    
    # Naive version
    print("  Naive version...", end=" ")
    start = time.time()
    K_naive = rbf_kernel_block_naive(X_rows, X_cols, sigma)
    time_naive = time.time() - start
    print(f"{time_naive:.4f} s")
    
    # Cache-blocked version
    print("  Cache-blocked...", end=" ")
    start = time.time()
    K_blocked = rbf_kernel_block_cache_blocked(X_rows, X_cols, sigma)
    time_blocked = time.time() - start
    print(f"{time_blocked:.4f} s")
    
    # Check correctness
    rel_error = np.linalg.norm(K_naive - K_blocked) / np.linalg.norm(K_naive)
    speedup = time_naive / time_blocked
    
    print(f"  Speedup: {speedup:.2f}×, Error: {rel_error:.2e}")
    
    return {
        'naive_time': time_naive,
        'blocked_time': time_blocked,
        'speedup': speedup,
        'relative_error': rel_error,
        'matrix_size': (n_rows, n_cols),
        'tile_size': 256,
    }


if __name__ == "__main__":
    print("RBF Kernel Cache Blocking Test (i7-11800H, 1.25 MB L2)")
    print("="*60)
    
    # Test different sizes
    for n in [500, 2000, 5000]:
        print(f"\n{n}×{n} matrix:")
        X = np.random.randn(n, 10)
        compare_kernel_methods(X, X, sigma=1.0)
    
    print("\n" + "="*60)
    print("✓ Tests completed!")
