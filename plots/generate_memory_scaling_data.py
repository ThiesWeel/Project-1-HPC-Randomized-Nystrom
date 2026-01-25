"""
Generate Memory Scaling Data for Explicit vs Streaming Methods

This script computes theoretical memory footprints for:
1. Explicit kernel matrix storage
2. Streaming (matrix-free) computation

For n in {5000, 20000, 50000, 100000}, l=1000, k=100.

Output:
    memory_scaling.json
"""

import json


def compute_memory_footprint(n: int, l: int, k: int) -> dict:
    """
    Compute theoretical memory footprint for each mode.
    All values in GB (assuming float64 = 8 bytes).
    """
    bytes_per_float = 8
    
    # Mode 1: Explicit kernel matrix
    # - Store full A matrix: n x n
    # - Also need Y, B, C, etc.
    A_bytes = n * n * bytes_per_float  # Full kernel matrix
    Omega_bytes = n * l * bytes_per_float  # Sketch matrix
    Y_bytes = n * l * bytes_per_float  # Y = A @ Omega
    B_bytes = l * l * bytes_per_float  # B = Omega' @ Y
    C_bytes = n * l * bytes_per_float  # C = Y @ B_pinv
    
    explicit_peak_bytes = A_bytes + Omega_bytes + Y_bytes + B_bytes
    explicit_total = {
        'A': A_bytes / 1e9,
        'Omega': Omega_bytes / 1e9,
        'Y': Y_bytes / 1e9,
        'B': B_bytes / 1e9,
        'peak_gb': explicit_peak_bytes / 1e9,
    }
    
    # Mode 2 & 3: Streaming (matrix-free)
    # - NO A matrix stored!
    # - Only chunk of A: chunk_size x n per iteration
    chunk_size = 1000
    K_chunk_bytes = chunk_size * n * bytes_per_float  # One chunk at a time
    
    streaming_peak_bytes = K_chunk_bytes + Omega_bytes + Y_bytes + B_bytes
    streaming_total = {
        'K_chunk': K_chunk_bytes / 1e9,
        'Omega': Omega_bytes / 1e9,
        'Y': Y_bytes / 1e9,
        'B': B_bytes / 1e9,
        'peak_gb': streaming_peak_bytes / 1e9,
    }
    
    return {
        'n': n,
        'l': l,
        'k': k,
        'chunk_size': chunk_size,
        'explicit': explicit_total,
        'streaming': streaming_total,
        'explicit_peak_gb': explicit_peak_bytes / 1e9,
        'streaming_peak_gb': streaming_peak_bytes / 1e9,
        'savings_ratio': explicit_peak_bytes / streaming_peak_bytes if streaming_peak_bytes > 0 else 0
    }


def main():
    print("=" * 60)
    print("Memory Scaling Analysis")
    print("=" * 60)
    
    n_values = [5000, 20000, 50000, 100000]
    l = 1000
    k = 100
    
    results = []
    
    print(f"\nParameters: l={l}, k={k}, chunk_size=1000")
    print("-" * 60)
    print(f"{'n':>10}  {'Explicit (GB)':>14}  {'Streaming (GB)':>14}  {'Savings':>10}")
    print("-" * 60)
    
    for n in n_values:
        mem = compute_memory_footprint(n, l, k)
        results.append(mem)
        
        print(f"{n:>10,}  {mem['explicit_peak_gb']:>14.2f}  {mem['streaming_peak_gb']:>14.2f}  {mem['savings_ratio']:>10.1f}x")
    
    print("-" * 60)
    
    # Save results
    output_file = 'memory_scaling.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
