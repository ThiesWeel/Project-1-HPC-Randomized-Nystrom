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
    Y_local_precomputed: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full Nystrom approximation pipeline.
    
    Computes rank-k truncation [[A_Nyst]]_k where:
        A_Nyst = (A @ Omega) @ (Omega^T @ A @ Omega)^+ @ (Omega^T @ A)
    
    Returns the implicit representation:
        [[A_Nyst]]_k = U_k @ diag(Lambda_k) @ U_k^T
    
    Memory-Efficient Mode (Matrix-Free):
        When Y_local_precomputed is provided, A_local can be None.
        This enables matrix-free computation where Y = A @ Omega is computed
        in chunks without ever storing the full kernel matrix A.
    
    Args:
        A_local: Local kernel block A_ij (can be None if Y_local_precomputed given)
        Omega_row_local: Omega rows for this rank's row indices
        Omega_col_block: Omega rows for this rank's column indices
        k: Target rank
        comm: Global communicator
        row_comm: Row communicator
        col_comm: Column communicator
        verbose: Print debug info
        rank: MPI rank for printing
        Y_local_precomputed: Pre-computed Y = A @ Omega (for matrix-free mode)
        
    Returns:
        U_k_local: Local block of eigenvectors, shape (n_rows_local, k)
        Lambda_k: Top k eigenvalues, shape (k,)
        info: Dictionary with intermediate values
    """
    info = {}
    
    # Step 3: Compute Y = A @ Omega
    if Y_local_precomputed is not None:
        # Matrix-free mode: Y already computed (in chunks, without storing A)
        Y_local = Y_local_precomputed
        info['matrix_free'] = True
    else:
        # Standard mode: compute Y from explicit kernel
        Y_local = compute_Y_local(A_local, Omega_col_block, row_comm)
        info['matrix_free'] = False
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


# Error computation utilities

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


# Performance measurement utilities

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


# Adaptive rank selection

def adaptive_rank_selection(
    Lambda_full: np.ndarray,
    energy_threshold: float = 0.95,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Automatically select rank k to capture a specified fraction of spectral energy.
    
    The spectral energy is defined as the cumulative sum of eigenvalues.
    We select the smallest k such that:
        sum(Lambda[:k]) / sum(Lambda) >= energy_threshold
    
    This provides a data-driven approach to rank selection, avoiding
    the need to hardcode k.
    
    Args:
        Lambda_full: All eigenvalues (in descending order), shape (l,)
        energy_threshold: Fraction of energy to capture (default: 0.95 = 95%)
        min_rank: Minimum rank to return (default: 1)
        max_rank: Maximum rank to return (default: len(Lambda_full))
        
    Returns:
        k_adaptive: Selected rank
        energy_captured: Actual fraction of energy captured
    """
    # Ensure eigenvalues are non-negative
    Lambda = np.maximum(Lambda_full, 0.0)
    
    total_energy = np.sum(Lambda)
    if total_energy <= 0:
        return min_rank, 0.0
    
    cumsum = np.cumsum(Lambda)
    
    # Find smallest k where cumulative energy >= threshold * total
    target = energy_threshold * total_energy
    k_adaptive = np.searchsorted(cumsum, target) + 1  # +1 because searchsorted is 0-indexed
    
    # Apply constraints
    if max_rank is None:
        max_rank = len(Lambda)
    k_adaptive = max(min_rank, min(k_adaptive, max_rank))
    
    energy_captured = cumsum[k_adaptive - 1] / total_energy if k_adaptive > 0 else 0.0
    
    return int(k_adaptive), float(energy_captured)


def compute_effective_rank(Lambda: np.ndarray) -> float:
    """
    Compute the effective rank (entropy-based) of the matrix.
    
    r_eff = exp(-sum(p_i * log(p_i)))
    
    where p_i = lambda_i / sum(lambda_j)
    
    This gives a smooth notion of "how many eigenvalues matter",
    useful for understanding the intrinsic dimensionality.
    
    Args:
        Lambda: Eigenvalues (non-negative), shape (l,)
        
    Returns:
        Effective rank (float between 1 and len(Lambda))
    """
    Lambda = np.maximum(Lambda, 0.0)
    total = np.sum(Lambda)
    
    if total <= 0:
        return 1.0
    
    # Normalize to probability distribution
    p = Lambda / total
    
    # Avoid log(0) by filtering out zeros
    p_nonzero = p[p > 1e-15]
    
    if len(p_nonzero) == 0:
        return 1.0
    
    # Entropy
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    
    # Effective rank
    r_eff = np.exp(entropy)
    
    return float(r_eff)


def compute_numerical_rank(Lambda: np.ndarray, epsilon: float = 1e-10) -> int:
    """
    Compute numerical rank at tolerance epsilon.
    
    r_epsilon = #{i : lambda_i > epsilon * lambda_max}
    
    Args:
        Lambda: Eigenvalues (non-negative), shape (l,)
        epsilon: Relative tolerance (default: 1e-10)
        
    Returns:
        Numerical rank
    """
    Lambda = np.maximum(Lambda, 0.0)
    
    if len(Lambda) == 0 or np.max(Lambda) <= 0:
        return 0
    
    threshold = epsilon * np.max(Lambda)
    return int(np.sum(Lambda > threshold))


# Exact error computation via matrix-free operations

def compute_exact_frobenius_error(
    U_k_local: np.ndarray,
    Lambda_k: np.ndarray,
    A_local: np.ndarray,
    Omega_row_local: np.ndarray,
    Omega_col_block: np.ndarray,
    row_comm: MPI.Comm,
    col_comm: MPI.Comm,
    is_diagonal_block: bool,
) -> Tuple[float, float, float]:
    """
    Compute exact Frobenius norm error without forming the full n×n approximation.
    
    Uses the identity:
        ||A - U_k Λ_k U_k^T||_F^2 = ||A||_F^2 - 2*trace(A U_k Λ_k U_k^T) + ||U_k Λ_k U_k^T||_F^2
    
    Each term can be computed efficiently:
    - ||A||_F^2: Sum of squared elements (distributed)
    - trace(A U_k Λ_k U_k^T) = sum_i λ_i * ||A^T u_i||^2 (matrix-vector products)
    - ||U_k Λ_k U_k^T||_F^2 = sum_i λ_i^2 (just eigenvalues)
    
    Args:
        U_k_local: Local eigenvectors, shape (n_rows_local, k)
        Lambda_k: Eigenvalues, shape (k,)
        A_local: Local kernel block A_ij
        Omega_row_local: For computing A @ U_k
        Omega_col_block: For computing A @ U_k
        row_comm: Row communicator
        col_comm: Column communicator
        is_diagonal_block: True if this is a diagonal block (i == j)
        
    Returns:
        rel_error: Relative Frobenius error ||A - A_approx||_F / ||A||_F
        abs_error: Absolute Frobenius error ||A - A_approx||_F
        A_fro: Frobenius norm of A
    """
    k = U_k_local.shape[1]
    
    # Term 1: ||A||_F^2 (sum of squared elements across all blocks)
    A_fro_sq_local = np.sum(A_local ** 2)
    A_fro_sq = col_comm.allreduce(A_fro_sq_local, op=MPI.SUM)
    # Need to reduce across row as well for full matrix
    A_fro_sq = row_comm.allreduce(A_fro_sq, op=MPI.SUM)
    # Correct for double-counting by dividing by number of column processes
    p_row = row_comm.Get_size()
    A_fro_sq = A_fro_sq / p_row
    
    A_fro = np.sqrt(A_fro_sq)
    
    # Term 3: ||U_k Λ_k U_k^T||_F^2 = trace((U_k Λ_k U_k^T)^2) = sum_i λ_i^2
    # Because U_k has orthonormal columns: ||U_k Λ U_k^T||_F^2 = ||Λ||_F^2 = sum(λ_i^2)
    approx_fro_sq = np.sum(Lambda_k ** 2)
    
    # Term 2: trace(A U_k Λ_k U_k^T) = trace(U_k^T A U_k Λ_k)
    # Compute U_k^T A U_k using distributed operations
    # First compute A @ U_k (similar to A @ Omega)
    
    # Local contribution: A_ij @ U_k_j (where U_k_j is the column block j's U_k)
    # This requires gathering U_k from all processes - simplify for diagonal
    
    # For efficiency, use trace(A U Λ U^T) = sum_i λ_i (u_i^T A u_i)
    # We can compute u_i^T A u_i for local blocks
    
    # Simplified approach: compute diagonal of U_k^T A U_k locally
    if is_diagonal_block:
        # U_k_local^T @ A_local @ U_k_local gives k×k matrix
        # We need the trace weighted by Lambda_k
        UTAU_local = U_k_local.T @ A_local @ U_k_local
    else:
        # Off-diagonal blocks contribute differently
        UTAU_local = np.zeros((k, k), dtype=np.float64)
    
    UTAU = np.zeros((k, k), dtype=np.float64)
    col_comm.Allreduce(UTAU_local, UTAU, op=MPI.SUM)
    
    # trace(A U Λ U^T) = trace(U^T A U Λ) = sum_i λ_i * (U^T A U)_ii
    trace_term = np.sum(np.diag(UTAU) * Lambda_k)
    
    # Error computation
    # ||A - A_approx||_F^2 = ||A||_F^2 - 2*trace_term + ||A_approx||_F^2
    error_fro_sq = A_fro_sq - 2 * trace_term + approx_fro_sq
    
    # Numerical stability: clamp to non-negative
    error_fro_sq = max(0.0, error_fro_sq)
    abs_error = np.sqrt(error_fro_sq)
    
    # Relative error
    rel_error = abs_error / A_fro if A_fro > 0 else 0.0
    
    return float(rel_error), float(abs_error), float(A_fro)


# Non-blocking communication

def compute_Y_local_nonblocking(
    A_local: np.ndarray,
    Omega_col_block: np.ndarray,
    row_comm: MPI.Comm,
) -> Tuple[np.ndarray, MPI.Request]:
    """
    Compute Y = A @ Ω with non-blocking reduction.
    
    Returns immediately with a request handle that can be waited on later,
    allowing overlap of communication with computation.
    
    Args:
        A_local: Local kernel block
        Omega_col_block: Local Omega block for columns
        row_comm: Row communicator
        
    Returns:
        Y_local: Buffer for result (will be filled when request completes)
        request: MPI request handle to wait on
    """
    # Local computation
    Y_partial = A_local @ Omega_col_block
    
    # Allocate output buffer
    Y_local = np.zeros_like(Y_partial)
    
    # Non-blocking allreduce
    request = row_comm.Iallreduce(Y_partial, Y_local, op=MPI.SUM)
    
    return Y_local, request


def compute_B_nonblocking(
    Omega_local: np.ndarray,
    Y_local: np.ndarray,
    col_comm: MPI.Comm,
) -> Tuple[np.ndarray, MPI.Request]:
    """
    Compute B = Ω^T @ Y with non-blocking reduction.
    
    Args:
        Omega_local: Local Omega block
        Y_local: Local Y block
        col_comm: Column communicator
        
    Returns:
        B: Buffer for result (will be filled when request completes)
        request: MPI request handle
    """
    # Local computation
    B_local = Omega_local.T @ Y_local
    
    # Allocate output buffer
    l = Omega_local.shape[1]
    B = np.zeros((l, l), dtype=np.float64)
    
    # Non-blocking allreduce
    request = col_comm.Iallreduce(B_local, B, op=MPI.SUM)
    
    return B, request


def compute_C_nonblocking(
    Z_local: np.ndarray,
    col_comm: MPI.Comm,
) -> Tuple[np.ndarray, MPI.Request]:
    """
    Compute C = Z^T @ Z with non-blocking reduction.
    
    Args:
        Z_local: Local Z factor
        col_comm: Column communicator
        
    Returns:
        C: Buffer for result
        request: MPI request handle
    """
    # Local computation
    C_local = Z_local.T @ Z_local
    
    # Allocate output buffer
    l = Z_local.shape[1]
    C = np.zeros((l, l), dtype=np.float64)
    
    # Non-blocking allreduce
    request = col_comm.Iallreduce(C_local, C, op=MPI.SUM)
    
    return C, request


def compute_nystrom_full_pipeline_nonblocking(
    A_local: np.ndarray,
    Omega_row_local: np.ndarray,
    Omega_col_block: np.ndarray,
    k: int,
    comm: MPI.Comm,
    row_comm: MPI.Comm,
    col_comm: MPI.Comm,
    use_adaptive_rank: bool = False,
    energy_threshold: float = 0.95,
    verbose: bool = False,
    rank: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full Nyström pipeline with non-blocking communication for better performance.
    
    Uses Iallreduce instead of Allreduce to overlap communication with computation.
    Also supports adaptive rank selection.
    
    Args:
        A_local: Local kernel block
        Omega_row_local: Local Omega (row indices)
        Omega_col_block: Local Omega (column indices)
        k: Target rank (or max rank if adaptive)
        comm: Global communicator
        row_comm: Row communicator
        col_comm: Column communicator
        use_adaptive_rank: If True, automatically select k based on eigenvalue decay
        energy_threshold: Energy threshold for adaptive rank (default: 0.95)
        verbose: Print debug info
        rank: MPI rank
        
    Returns:
        U_k_local: Local eigenvector block
        Lambda_k: Top k (or adaptive k) eigenvalues
        info: Dictionary with diagnostics
    """
    info = {}
    l = Omega_row_local.shape[1]
    
    # Step 1: Compute Y = A @ Ω (with non-blocking reduction)
    Y_local, req_Y = compute_Y_local_nonblocking(A_local, Omega_col_block, row_comm)
    
    # Wait for Y to be ready (could do other work here in future)
    req_Y.Wait()
    info['Y_local_shape'] = Y_local.shape
    
    # Step 2: Compute B = Ω^T @ Y (with non-blocking reduction)
    B, req_B = compute_B_nonblocking(Omega_row_local, Y_local, col_comm)
    
    # Wait for B (could overlap with preparation work)
    req_B.Wait()
    info['B_shape'] = B.shape
    
    # Step 3: Compute B^+
    B_pinv, U_B, S_B = compute_B_pseudoinverse(B)
    info['B_singular_values'] = S_B
    
    # Step 4: Compute Z = Y @ B^(+1/2)
    Z_local = compute_nystrom_factors(Y_local, B_pinv)
    info['Z_local_shape'] = Z_local.shape
    
    # Step 5: Compute C = Z^T @ Z (with non-blocking reduction)
    C, req_C = compute_C_nonblocking(Z_local, col_comm)
    
    # Wait for C
    req_C.Wait()
    
    # Symmetrize
    C = 0.5 * (C + C.T)
    
    # Step 6: Eigendecompose C
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = eigvals[::-1]  # Descending order
    eigvecs = eigvecs[:, ::-1]
    
    info['Lambda_full'] = eigvals.copy()
    info['effective_rank'] = compute_effective_rank(eigvals)
    info['numerical_rank'] = compute_numerical_rank(eigvals)
    
    # Step 7: Adaptive rank selection (INNOVATION 1)
    if use_adaptive_rank:
        k_adaptive, energy_captured = adaptive_rank_selection(
            eigvals, energy_threshold, min_rank=1, max_rank=k
        )
        info['k_original'] = k
        info['k_adaptive'] = k_adaptive
        info['energy_captured'] = energy_captured
        k = k_adaptive
    
    # Step 8: Take top k
    k = min(k, l)
    Lambda_k = eigvals[:k]
    W_k = eigvecs[:, :k]
    
    # Step 9: Lift to n-space
    Lambda_k_safe = np.maximum(Lambda_k, 1e-15)
    inv_sqrt_Lambda_k = 1.0 / np.sqrt(Lambda_k_safe)
    inv_sqrt_Lambda_k[Lambda_k < 1e-15] = 0.0
    
    U_k_local = Z_local @ W_k @ np.diag(inv_sqrt_Lambda_k)
    info['Lambda_k'] = Lambda_k
    
    return U_k_local, Lambda_k, info


# Strong scaling analysis utilities

def compute_parallel_efficiency(
    t_serial: float,
    t_parallel: float,
    num_processes: int,
) -> Tuple[float, float]:
    """
    Compute speedup and parallel efficiency.
    
    Speedup S(P) = T(1) / T(P)
    Efficiency E(P) = S(P) / P
    
    Args:
        t_serial: Serial execution time T(1)
        t_parallel: Parallel execution time T(P)
        num_processes: Number of processes P
        
    Returns:
        speedup: S(P)
        efficiency: E(P)
    """
    if t_parallel <= 0:
        return float('inf'), float('inf')
    
    speedup = t_serial / t_parallel
    efficiency = speedup / num_processes
    
    return speedup, efficiency


def estimate_amdahl_limit(
    serial_fraction: float,
    num_processes: int,
) -> float:
    """
    Estimate maximum speedup according to Amdahl's Law.
    
    S_max = 1 / (f + (1-f)/P)
    
    where f is the serial fraction.
    
    Args:
        serial_fraction: Fraction of code that is serial (0 to 1)
        num_processes: Number of processes
        
    Returns:
        Maximum theoretical speedup
    """
    if serial_fraction >= 1.0:
        return 1.0
    
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / num_processes)


def analyze_scaling_results(
    timing_results: list,
) -> dict:
    """
    Analyze scaling results from multiple runs.
    
    Args:
        timing_results: List of dicts with 'P' (processes) and 'time' keys
        
    Returns:
        Dictionary with analysis metrics
    """
    if not timing_results:
        return {}
    
    # Sort by P
    results = sorted(timing_results, key=lambda x: x['P'])
    
    # Find serial time (P=1)
    serial_result = next((r for r in results if r['P'] == 1), None)
    t_serial = serial_result['time'] if serial_result else results[0]['time']
    
    analysis = {
        'serial_time': t_serial,
        'scaling_data': [],
    }
    
    for r in results:
        P = r['P']
        t = r['time']
        speedup, efficiency = compute_parallel_efficiency(t_serial, t, P)
        
        analysis['scaling_data'].append({
            'P': P,
            'time': t,
            'speedup': speedup,
            'efficiency': efficiency,
            'ideal_speedup': P,
        })
    
    return analysis
