"""
Randomized Nyström Low-Rank Approximation

This module implements the distributed Nyström algorithm for approximating
symmetric positive semidefinite (SPSD) kernel matrices.

Algorithm overview:
    Given A ∈ R^{n×n} (SPSD), Ω ∈ R^{n×l} (sketching matrix):
    
    1. Compute Y = A @ Ω                    (distributed matrix-matrix multiply)
    2. Compute B = Ωᵀ @ Y                   (small l×l matrix, Allreduce)
    3. Compute B⁺ (pseudoinverse of B)      (local, SVD-based)
    4. Form A_Nyst = Y @ B⁺ @ Yᵀ            (implicit representation)
    5. Compute rank-k truncation [[A_Nyst]]_k

Reference: Tropp et al. (2017), "Fixed-Rank Approximation of a Positive-Semidefinite Matrix from Streaming Data"
"""

import numpy as np
from mpi4py import MPI
from typing import Tuple, Optional


def create_row_communicator(comm: MPI.Comm, i: int, j: int, p: int) -> MPI.Comm:
    """
    Create a communicator for processes in the same row of the process grid.
    
    Processes in the same row (same i coordinate) need to communicate
    to sum partial contributions to Y.
    
    Args:
        comm: Global MPI communicator
        i: Row index in process grid
        j: Column index in process grid
        p: Grid dimension
        
    Returns:
        Row communicator
    """
    # Color by row index - processes with same i are in same communicator
    row_comm = comm.Split(color=i, key=j)
    return row_comm


def create_col_communicator(comm: MPI.Comm, i: int, j: int, p: int) -> MPI.Comm:
    """
    Create a communicator for processes in the same column of the process grid.
    
    Args:
        comm: Global MPI communicator
        i: Row index in process grid
        j: Column index in process grid
        p: Grid dimension
        
    Returns:
        Column communicator
    """
    # Color by column index - processes with same j are in same communicator
    col_comm = comm.Split(color=j, key=i)
    return col_comm


def compute_Y_local(
    A_local: np.ndarray,
    Omega_col_block: np.ndarray,
    row_comm: MPI.Comm,
) -> np.ndarray:
    """
    Compute local contribution to Y = A @ Ω and reduce across row.
    
    Each rank (i, j) computes:
        Y_partial = A_ij @ Ω_{J_j}
    
    Then sum across all j in the same row to get Y_{I_i}.
    
    Args:
        A_local: Local kernel block A_ij of shape (n_rows_local, n_cols_local)
        Omega_col_block: Ω rows corresponding to column indices, shape (n_cols_local, l)
        row_comm: MPI communicator for processes in the same row
        
    Returns:
        Y_local: Local block of Y, shape (n_rows_local, l)
    """
    # Local contribution: A_ij @ Omega_{J_j}
    Y_partial = A_local @ Omega_col_block
    
    # Sum across row communicator
    Y_local = np.zeros_like(Y_partial)
    row_comm.Allreduce(Y_partial, Y_local, op=MPI.SUM)
    
    return Y_local


def compute_B(
    Omega_local: np.ndarray,
    Y_local: np.ndarray,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    Compute B = Ωᵀ @ Y (small l×l matrix).
    
    Each rank computes local contribution:
        B_local = Ω_local.T @ Y_local
    
    Then Allreduce to get global B (replicated on all ranks).
    
    Args:
        Omega_local: Local Ω block, shape (n_rows_local, l)
        Y_local: Local Y block, shape (n_rows_local, l)
        comm: MPI communicator (use column communicator or subset)
        
    Returns:
        B: Global B matrix, shape (l, l), replicated on all ranks
    """
    # Local contribution
    B_local = Omega_local.T @ Y_local
    
    # Allreduce to sum contributions
    l = Omega_local.shape[1]
    B = np.zeros((l, l), dtype=np.float64)
    comm.Allreduce(B_local, B, op=MPI.SUM)
    
    return B


def compute_B_pseudoinverse(
    B: np.ndarray,
    regularization: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the pseudoinverse of B using SVD.
    
    B⁺ = V @ Σ⁺ @ Uᵀ
    
    Also returns the SVD factors for potential use in truncation.
    
    Args:
        B: Symmetric matrix of shape (l, l)
        regularization: Small value to add for numerical stability
        
    Returns:
        B_pinv: Pseudoinverse of B, shape (l, l)
        U: Left singular vectors, shape (l, l)
        S: Singular values, shape (l,)
    """
    # Symmetrize B for numerical stability
    B_sym = 0.5 * (B + B.T)
    
    # SVD of B
    U, S, Vt = np.linalg.svd(B_sym, full_matrices=False)
    
    # Compute pseudoinverse with regularization
    # S_inv[i] = 1/S[i] if S[i] > threshold, else 0
    threshold = regularization * np.max(S) if np.max(S) > 0 else regularization
    S_inv = np.where(S > threshold, 1.0 / S, 0.0)
    
    # B⁺ = V @ diag(S_inv) @ Uᵀ = U @ diag(S_inv) @ Uᵀ (since B is symmetric)
    B_pinv = (U * S_inv) @ U.T
    
    return B_pinv, U, S


def compute_nystrom_factors(
    Y_local: np.ndarray,
    B_pinv: np.ndarray,
) -> np.ndarray:
    """
    Compute local factor for Nyström approximation.
    
    A_Nyst = Y @ B⁺ @ Yᵀ = (Y @ B⁺^{1/2}) @ (Y @ B⁺^{1/2})ᵀ
    
    We compute Z_local = Y_local @ L where B⁺ = L @ Lᵀ (Cholesky-like).
    
    For numerical stability, use eigendecomposition of B⁺.
    
    Args:
        Y_local: Local Y block, shape (n_rows_local, l)
        B_pinv: Pseudoinverse of B, shape (l, l)
        
    Returns:
        Z_local: Factor such that A_Nyst ≈ Z @ Zᵀ, shape (n_rows_local, l)
    """
    # Eigendecomposition of B⁺ (which is SPSD)
    eigvals, eigvecs = np.linalg.eigh(B_pinv)
    
    # Clamp negative eigenvalues to zero (numerical artifacts)
    eigvals = np.maximum(eigvals, 0.0)
    
    # B⁺^{1/2} = V @ diag(sqrt(eigvals)) @ Vᵀ
    sqrt_eigvals = np.sqrt(eigvals)
    B_pinv_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    
    # Z_local = Y_local @ B⁺^{1/2}
    Z_local = Y_local @ B_pinv_sqrt
    
    return Z_local


def truncate_nystrom_rank_k(
    Z_local: np.ndarray,
    k: int,
    col_comm: MPI.Comm,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rank-k truncation of A_Nyst = Z @ Zᵀ.
    
    This implements the SECOND approach from the project:
    Truncate A_Nyst (not B).
    
    Algorithm:
    1. Compute C = Zᵀ @ Z (small l×l, use Allreduce)
    2. Eigendecompose C = W @ Λ @ Wᵀ
    3. Take top k eigenvalues/vectors
    4. Final factor: U_k = Z @ W_k @ Λ_k^{-1/2}
    
    Then: [[A_Nyst]]_k = U_k @ Λ_k @ U_kᵀ
    
    Args:
        Z_local: Local Nyström factor, shape (n_rows_local, l)
        k: Target rank for truncation
        col_comm: Column communicator for Allreduce
        
    Returns:
        U_k_local: Local block of rank-k left singular vectors, shape (n_rows_local, k)
        Lambda_k: Top k eigenvalues, shape (k,)
    """
    l = Z_local.shape[1]
    k = min(k, l)  # Cannot exceed l
    
    # Compute C = Zᵀ @ Z (l×l matrix)
    C_local = Z_local.T @ Z_local
    C = np.zeros((l, l), dtype=np.float64)
    col_comm.Allreduce(C_local, C, op=MPI.SUM)
    
    # Symmetrize for numerical stability
    C = 0.5 * (C + C.T)
    
    # Eigendecomposition (eigenvalues in ascending order)
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Reverse to get descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    # Take top k
    Lambda_k = eigvals[:k]
    W_k = eigvecs[:, :k]
    
    # Compute U_k = Z @ W_k @ diag(1/sqrt(Lambda_k))
    # Handle zero eigenvalues
    Lambda_k_safe = np.maximum(Lambda_k, 1e-15)
    inv_sqrt_Lambda_k = 1.0 / np.sqrt(Lambda_k_safe)
    
    # Zero out contributions from negligible eigenvalues
    inv_sqrt_Lambda_k[Lambda_k < 1e-15] = 0.0
    
    U_k_local = Z_local @ W_k @ np.diag(inv_sqrt_Lambda_k)
    
    return U_k_local, Lambda_k


def compute_nystrom_full_pipeline(
    A_local: np.ndarray,
    Omega_row_local: np.ndarray,
    Omega_col_block: np.ndarray,
    k: int,
    comm: MPI.Comm,
    row_comm: MPI.Comm,
    col_comm: MPI.Comm,
    verbose: bool = False,
    rank: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full Nyström approximation pipeline.
    
    Computes rank-k truncation [[A_Nyst]]_k where:
        A_Nyst = (A @ Ω) @ (Ωᵀ @ A @ Ω)⁺ @ (Ωᵀ @ A)
    
    Returns the implicit representation:
        [[A_Nyst]]_k = U_k @ diag(Lambda_k) @ U_kᵀ
    
    Args:
        A_local: Local kernel block A_ij
        Omega_row_local: Ω rows for this rank's row indices
        Omega_col_block: Ω rows for this rank's column indices
        k: Target rank
        comm: Global communicator
        row_comm: Row communicator
        col_comm: Column communicator
        verbose: Print debug info
        rank: MPI rank for printing
        
    Returns:
        U_k_local: Local block of eigenvectors, shape (n_rows_local, k)
        Lambda_k: Top k eigenvalues, shape (k,)
        info: Dictionary with intermediate values
    """
    info = {}
    
    # Step 3: Compute Y = A @ Ω
    Y_local = compute_Y_local(A_local, Omega_col_block, row_comm)
    info['Y_local_shape'] = Y_local.shape
    
    # Step 4: Compute B = Ωᵀ @ Y (only processes in column 0 have unique row data)
    # Use column communicator to avoid double-counting
    B = compute_B(Omega_row_local, Y_local, col_comm)
    info['B_shape'] = B.shape
    
    # Step 5: Compute B⁺
    B_pinv, U_B, S_B = compute_B_pseudoinverse(B)
    info['B_singular_values'] = S_B
    
    # Step 6: Compute Nyström factor Z
    Z_local = compute_nystrom_factors(Y_local, B_pinv)
    info['Z_local_shape'] = Z_local.shape
    
    # Step 7: Rank-k truncation
    U_k_local, Lambda_k = truncate_nystrom_rank_k(Z_local, k, col_comm)
    info['Lambda_k'] = Lambda_k
    
    return U_k_local, Lambda_k, info


# =============================================================================
# Error computation utilities
# =============================================================================

def compute_nuclear_norm_approx(
    Lambda: np.ndarray,
) -> float:
    """
    Compute nuclear norm from eigenvalues.
    
    ||M||_* = sum of singular values = sum of eigenvalues (for SPSD)
    
    Args:
        Lambda: Eigenvalues of SPSD matrix
        
    Returns:
        Nuclear norm
    """
    return np.sum(np.maximum(Lambda, 0.0))


def compute_local_trace(A_local_diag: np.ndarray) -> float:
    """
    Compute local contribution to trace of A.
    
    For SPSD matrices, trace(A) = ||A||_* if A has unit eigenvalues,
    but in general we need eigenvalues.
    
    Args:
        A_local_diag: Diagonal elements of A_local (for diagonal blocks only)
        
    Returns:
        Local trace contribution
    """
    return np.sum(A_local_diag)


def estimate_nuclear_norm_A(
    comm: MPI.Comm,
    A_local: np.ndarray,
    is_diagonal_block: bool,
    n: int,
    sample_size: int = 100,
) -> float:
    """
    Estimate nuclear norm of A using trace (upper bound for SPSD).
    
    For SPSD matrices: ||A||_* = trace(A) = sum of eigenvalues
    
    This only works correctly when we sum diagonal blocks.
    
    Args:
        comm: MPI communicator
        A_local: Local kernel block
        is_diagonal_block: True if this is a diagonal block (i == j)
        n: Global matrix dimension
        sample_size: Not used for exact trace computation
        
    Returns:
        Nuclear norm of A (exact for SPSD via trace)
    """
    # For SPSD, nuclear norm = trace = sum of diagonal elements
    if is_diagonal_block:
        local_trace = np.trace(A_local)
    else:
        local_trace = 0.0
    
    # Sum traces across all ranks
    global_trace = comm.allreduce(local_trace, op=MPI.SUM)
    
    return global_trace


def compute_relative_error(
    Lambda_A: float,
    Lambda_k: np.ndarray,
    U_k_local: np.ndarray,
    col_comm: MPI.Comm,
) -> float:
    """
    Compute relative error in nuclear norm.
    
    For SPSD matrices:
        ||A - [[A_Nyst]]_k||_* = ||A||_* - ||[[A_Nyst]]_k||_*    (approximately)
    
    This is an approximation; exact error requires full eigendecomposition.
    
    Args:
        Lambda_A: Nuclear norm of A (= trace for SPSD)
        Lambda_k: Top k eigenvalues of Nyström approximation
        U_k_local: Local eigenvectors
        col_comm: Column communicator
        
    Returns:
        Relative error ||A - [[A_Nyst]]_k||_* / ||A||_*
    """
    # Nuclear norm of approximation
    nystrom_nuclear = compute_nuclear_norm_approx(Lambda_k)
    
    # Relative error (approximate)
    if Lambda_A > 0:
        rel_error = (Lambda_A - nystrom_nuclear) / Lambda_A
    else:
        rel_error = 0.0
    
    return max(rel_error, 0.0)  # Ensure non-negative


# =============================================================================
# Performance measurement utilities
# =============================================================================

def measure_time(comm: MPI.Comm) -> float:
    """
    Get synchronized wall-clock time.
    
    Args:
        comm: MPI communicator
        
    Returns:
        Wall-clock time after barrier
    """
    comm.Barrier()
    return MPI.Wtime()
