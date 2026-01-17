"""
LIBSVM dataset loader for Randomized Nyström approximation.

This module provides functions for loading LIBSVM-format datasets
(such as MNIST, YearPredictionMSD) into dense NumPy arrays.

Each MPI rank can load only its assigned rows without reading the full file.
"""

import numpy as np
import re
from typing import Optional


def parse_libsvm_line(line: str, feature_count: int, sep: str = " ") -> np.ndarray:
    """
    Parse one LIBSVM line into a dense NumPy feature vector.
    
    LIBSVM format:
        <label> <index>:<value> <index>:<value> ...
    
    Indices in LIBSVM files are 1-based; this function converts to 0-based.
    The label (first token) is ignored.
    
    Args:
        line: A single line from a LIBSVM file
        feature_count: Total number of features (dimension of output vector)
        sep: Separator between tokens (default: space)
        
    Returns:
        Dense NumPy array of shape (feature_count,), dtype float64
    """
    # Initialize dense zero vector
    features = np.zeros(feature_count, dtype=np.float64)
    
    # Split line into tokens
    tokens = line.strip().split(sep)
    
    # Skip the first token (label) and process index:value pairs
    for token in tokens[1:]:
        if not token:
            continue
        
        # Split on colon to get index and value
        if ':' not in token:
            continue
            
        idx_str, val_str = token.split(':', 1)
        
        # Convert 1-based index to 0-based
        idx = int(idx_str) - 1
        val = float(val_str)
        
        # Store value at the correct position
        if 0 <= idx < feature_count:
            features[idx] = val
    
    return features


def infer_feature_count(filename: str) -> int:
    """
    Infer feature count from filename suffix.
    
    Examples:
        'mnist_780' -> 780
        'YearPredictionMSD_90' -> 90
        
    Args:
        filename: Path to the dataset file
        
    Returns:
        Inferred feature count
        
    Raises:
        ValueError: If feature count cannot be inferred from filename
    """
    # Extract the base filename (without path and extension)
    import os
    base = os.path.basename(filename)
    
    # Remove common extensions
    for ext in ['.bz2', '.gz', '.txt', '.libsvm']:
        if base.endswith(ext):
            base = base[:-len(ext)]
    
    # Look for trailing number after underscore
    match = re.search(r'_(\d+)$', base)
    if match:
        return int(match.group(1))
    
    raise ValueError(
        f"Cannot infer feature count from filename '{filename}'. "
        f"Please provide feature_count explicitly."
    )


def read_dataset(
    filename: str,
    skip_lines: int,
    line_count: int,
    feature_count: Optional[int] = None,
    sep: str = " ",
) -> np.ndarray:
    """
    Load a subset of rows from a LIBSVM dataset file.
    
    This function reads only the requested portion of the file,
    making it suitable for distributed loading where each MPI rank
    loads only its assigned rows.
    
    Args:
        filename: Path to the LIBSVM dataset file
        skip_lines: Number of lines to skip from the beginning
        line_count: Number of lines to read
        feature_count: Number of features per sample. If None, inferred
                       from filename suffix (e.g., 'mnist_780' -> 780)
        sep: Separator between tokens (default: space)
        
    Returns:
        Dense NumPy array of shape (line_count, feature_count), dtype float64
    """
    # Infer feature count from filename if not provided
    if feature_count is None:
        feature_count = infer_feature_count(filename)
    
    # Pre-allocate output matrix
    X = np.zeros((line_count, feature_count), dtype=np.float64)
    
    # Open file and read the requested subset
    with open(filename, 'r', encoding='utf-8') as f:
        # Skip the first skip_lines lines
        for _ in range(skip_lines):
            next(f)
        
        # Read exactly line_count lines
        for i in range(line_count):
            line = next(f)
            X[i, :] = parse_libsvm_line(line, feature_count, sep)
    
    return X
