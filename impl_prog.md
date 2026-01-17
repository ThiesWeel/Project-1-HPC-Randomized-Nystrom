# Implementation Documentation: Randomized Nystrom Low-Rank Approximation

This document describes the implementation of the Randomized Nystrom algorithm for approximating symmetric positive semidefinite (SPSD) kernel matrices. The implementation is distributed using MPI and follows the algorithmic structure specified in the project requirements.

---

## Phase 0 — MPI Infrastructure and Block Ownership

The implementation begins with MPI initialization and the construction of a distributed data layout suitable for 2D block-cyclic matrix operations.

### Process Grid Construction

MPI is initialized via `mpi4py`. The total number of MPI ranks P must be a perfect square. This constraint is enforced by the function `validate_process_count` in `main.py`, which computes p = sqrt(P) and aborts execution if P is not a perfect square.

Each MPI rank is assigned a position (i, j) in a logical p x p process grid using row-major ordering. The mapping is implemented in `rank_to_grid_coords`:

- Row index: i = rank // p
- Column index: j = rank % p

### Block Ownership

The global n x n kernel matrix A is distributed in 2D blocks across the process grid. Each rank (i, j) owns a submatrix A_ij corresponding to:

- Row indices in the range [row_start, row_end)
- Column indices in the range [col_start, col_end)

The function `compute_block_range` computes these ranges, handling the case where n is not evenly divisible by p. Processes with lower grid indices receive one additional row or column when there is a remainder.

This 2D block distribution ensures that:
- No rank stores the full matrix A
- Memory usage scales with the local block size only
- The distribution is compatible with subsequent MPI collective operations

---

## Phase 1 — Dataset-Backed Feature Loading

The implementation supports loading feature vectors from LIBSVM-format datasets, as required for the MNIST and YearPredictionMSD benchmarks.

### LIBSVM Parsing

The module `data/libsvm_loader.py` provides two functions for dataset loading:

**parse_libsvm_line**: Parses a single line of LIBSVM format into a dense NumPy vector. The LIBSVM format is:

```
<label> <index>:<value> <index>:<value> ...
```

Indices in the file are 1-based and are converted to 0-based for NumPy. The label is ignored; only feature values are extracted. The function initializes a zero vector of the specified feature dimension and populates it with the sparse values.

**read_dataset**: Loads a contiguous subset of rows from a dataset file. It accepts parameters for:
- Number of lines to skip (skip_lines)
- Number of lines to read (line_count)
- Feature dimension (inferred from filename if not provided)

This design allows each MPI rank to load only its assigned rows without reading the entire file into memory.

### Supported Datasets

- MNIST: 780 features, recommended kernel bandwidth sigma = 100
- YearPredictionMSD: 90 features, recommended kernel bandwidth sigma = 1e4 or 1e5

The choice of n (number of data points to use) is determined by memory constraints and is passed as a parameter.

### Synthetic Data Fallback

For testing and debugging, the module `data/kernel.py` provides `generate_features`, which generates deterministic random feature vectors based on global indices. Each global index i is seeded with (seed + i) to ensure reproducibility across ranks.

---

## Phase 2 — Local Kernel Block Construction

Each MPI rank independently constructs its local kernel block A_ij without any inter-process communication.

### RBF Kernel Formula

The implementation uses the Radial Basis Function (RBF) kernel, also known as the Gaussian kernel:

K(x, y) = exp(-||x - y||^2 / sigma^2)

where sigma is the kernel bandwidth parameter.

### Vectorized Computation

The function `rbf_kernel_block` in `data/kernel.py` computes the kernel block using the identity:

||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y

This allows efficient vectorized computation:
1. Compute squared norms of row features: ||x||^2
2. Compute squared norms of column features: ||y||^2
3. Compute the cross term: X_rows @ X_cols.T
4. Combine using broadcasting to obtain squared distances
5. Apply the exponential function

Numerical stability is ensured by clamping squared distances to be non-negative before applying the exponential.

### Independence

This phase requires no MPI communication. Each rank loads its own row and column features and computes A_ij locally.

---

## Phase 3 — Sketching Matrix Construction

The sketching matrix Omega is used to project the high-dimensional kernel matrix into a lower-dimensional space.

### Gaussian Sketch

The implementation uses a Gaussian sketching matrix where each entry is drawn from a standard normal distribution. The matrix Omega has dimensions n x l, where l is the sketch dimension satisfying k <= l << n.

### Block-Row Distribution

Omega is distributed by block rows, matching the row distribution of A. Each rank (i, j) generates the rows of Omega corresponding to its row indices I_i. This is implemented in `generate_omega_block` in `data/sketch.py`.

### Deterministic Construction

To ensure reproducibility without communication, each row of Omega is generated deterministically based on its global row index. For global row index i, the random number generator is seeded with (seed + i). This guarantees that:

- The same global row index always produces the same Omega row
- Different ranks can independently generate consistent subsets of Omega
- Results are reproducible across different numbers of MPI processes

### No Communication

This phase requires no MPI communication. Each rank generates its local Omega block independently.

---

## Phase 4 — Distributed Computation of Y = A * Omega

The product Y = A * Omega is computed in a distributed manner, requiring MPI communication to combine partial results.

### Local Partial Products

Each rank (i, j) computes a partial contribution to Y:

Y_partial = A_ij @ Omega_{J_j}

where Omega_{J_j} denotes the rows of Omega corresponding to the column indices J_j owned by this rank. Note that each rank must generate both its row block and column block of Omega.

### Row Communicators

To combine partial products, the implementation creates MPI sub-communicators for processes in the same row of the process grid. This is done via `create_row_communicator` in `nystrom.py`, which uses `comm.Split` with the row index i as the color.

### Reduction

The function `compute_Y_local` performs the local matrix multiplication and then uses `MPI_Allreduce` with the row communicator to sum partial contributions:

Y_local = sum over j of (A_ij @ Omega_{J_j})

After this reduction, each rank in row i holds the complete Y block for row indices I_i.

### Distribution of Y

The result Y is distributed by block rows, matching the row distribution of A and Omega. Each rank stores Y_local with shape (local_row_count, l).

---

## Phase 5 — Computation of B = Omega^T * A * Omega

The small l x l matrix B is computed using the identity B = Omega^T @ Y.

### Local Contributions

Each rank computes its local contribution:

B_local = Omega_local^T @ Y_local

where Omega_local and Y_local are the local blocks owned by this rank.

### Global Reduction

The function `compute_B` uses `MPI_Allreduce` with the column communicator to sum local contributions across all ranks in the same column of the process grid. This avoids double-counting since ranks in the same column share the same row indices.

### Replication

After the reduction, B is replicated on all ranks. This is acceptable because B has dimensions l x l, where l is small (typically 20-200).

---

## Phase 6 — Pseudoinverse of B

The Moore-Penrose pseudoinverse of B is computed locally on each rank (since B is replicated).

### SVD-Based Computation

The function `compute_B_pseudoinverse` in `nystrom.py` computes B^+ using the singular value decomposition:

1. Symmetrize B for numerical stability: B_sym = 0.5 * (B + B^T)
2. Compute SVD: B_sym = U @ diag(S) @ V^T
3. Compute pseudoinverse: B^+ = V @ diag(S^+) @ U^T

For symmetric matrices, U = V, so B^+ = U @ diag(S^+) @ U^T.

### Regularization

Singular values below a threshold are treated as zero to ensure numerical stability. The threshold is set to:

threshold = regularization * max(S)

where regularization = 1e-10 by default. Singular values below this threshold contribute zero to the pseudoinverse.

---

## Phase 7 — Nystrom Approximation and Rank-k Truncation

The Nystrom approximation is formed implicitly, and a rank-k truncation is computed.

### Implicit Representation

The Nystrom approximation is:

A_Nyst = Y @ B^+ @ Y^T

This is never formed explicitly as an n x n matrix. Instead, the implementation works with factors.

### Nystrom Factor Z

The function `compute_nystrom_factors` computes a factor Z such that:

A_Nyst = Z @ Z^T

This is achieved by computing:

Z = Y @ (B^+)^{1/2}

The square root of B^+ is computed via eigendecomposition:

1. Compute eigenvalues and eigenvectors of B^+
2. Clamp negative eigenvalues to zero
3. Compute (B^+)^{1/2} = V @ diag(sqrt(eigenvalues)) @ V^T
4. Multiply: Z_local = Y_local @ (B^+)^{1/2}

### Rank-k Truncation

The rank-k truncation is computed by the function `truncate_nystrom_rank_k`, which implements the second approach specified in the project: truncation of A_Nyst rather than truncation of B.

The algorithm:
1. Compute C = Z^T @ Z using MPI Allreduce (C is l x l)
2. Eigendecompose C = W @ Lambda @ W^T
3. Select the top k eigenvalues and corresponding eigenvectors
4. Compute U_k = Z @ W_k @ diag(Lambda_k^{-1/2})

The result is an implicit representation:

[[A_Nyst]]_k = U_k @ diag(Lambda_k) @ U_k^T

Each rank stores its local block of U_k with shape (local_row_count, k).

---

## Phase 8 — Error Measurement

The approximation quality is measured using the nuclear norm error ratio.

### Nuclear Norm Definition

For SPSD matrices, the nuclear norm equals the trace:

||A||_* = trace(A) = sum of eigenvalues

### Computing ||A||_*

The function `estimate_nuclear_norm_A` computes the nuclear norm of A by summing the traces of diagonal blocks:

1. Each rank with i == j (diagonal block) computes trace(A_local)
2. These local traces are summed via MPI Allreduce
3. The result is the exact trace (and nuclear norm) of A

### Relative Error

The function `compute_relative_error` computes an approximate relative error:

error = (||A||_* - ||[[A_Nyst]]_k||_*) / ||A||_*

where ||[[A_Nyst]]_k||_* = sum(Lambda_k).

This is an approximation. The exact error ||A - [[A_Nyst]]_k||_* would require computing the full eigendecomposition of the error matrix, which is not tractable for large n.

### Limitations

For large n, the error estimate is approximate. The implementation assumes that the approximation captures the largest eigenvalues well, making the difference in nuclear norms a reasonable proxy for the true error.

---

## Phase 9 — Experimental Framework

The script `run_experiments.py` provides a flexible framework for running experiments and collecting data for analysis.

### Supported Modes

**single**: Runs a single Nystrom approximation with specified parameters and reports the error and timing.

**accuracy**: Runs a series of experiments varying:
- Sketch dimension l (with k = l/2)
- Target rank k (with fixed l)
- Kernel bandwidth sigma

**scaling**: Runs a single experiment for timing measurements. For true scaling studies, the script should be invoked multiple times with different numbers of MPI processes.

### Parameters

The following parameters can be specified via command line:
- `--dataset`: Dataset to use (mnist, year, or synthetic)
- `--n`: Matrix dimension (number of data points)
- `--l`: Sketch dimension
- `--k`: Target rank
- `--sigma`: Kernel bandwidth
- `--d`: Feature dimension for synthetic data
- `--output`: Output file for JSON results

### Output Format

Results are collected in a list of dictionaries containing:
- Input parameters (n, l, k, sigma, P)
- Timing (wall-clock time)
- Nuclear norms (exact for A, approximate for approximation)
- Relative error
- Top k eigenvalues

Results can be saved to a JSON file for subsequent analysis and plotting.

---

## Final Notes

### Alignment with Project Requirements

This implementation follows the algorithmic structure specified in the project PDF:
- 2D block distribution of A
- Block-row distribution of Omega
- Distributed computation of Y = A @ Omega with row-wise reduction
- Small matrix B computed and replicated
- Rank-k truncation of A_Nyst (not B)
- Nuclear norm error measurement

### Supported Datasets

- MNIST (mnist.scale): 780 features, up to 60,000 samples
- YearPredictionMSD: 90 features, larger sample count
- Synthetic: Configurable feature dimension, any sample count

### Parallel Scalability

The implementation scales with the number of MPI processes as follows:
- Local storage: O(n^2 / P) for kernel blocks, O(n * l / sqrt(P)) for Y and Omega blocks
- Communication: O(l^2) for B computation (independent of n), O(n * l / sqrt(P)) for Y reduction
- Computation: Dominated by local matrix multiplications

The algorithm is well-suited for moderate parallelism (P up to ~100) where l remains small relative to n/sqrt(P).
