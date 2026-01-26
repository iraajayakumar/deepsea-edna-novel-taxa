"""
Multi-k Frequency Chaos Game Representation (FCGR) encoding.

Converts fixed-length DNA sequences into normalized FCGR matrices
for k = 4, 5, 6.

Designed for deep-sea eDNA clustering and self-supervised learning.
"""

import numpy as np
from typing import Dict, List

# Fixed nucleotide encoding
NUC_TO_INT = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}


def _kmer_to_index(kmer: str) -> int:
    """
    Convert k-mer string to integer index using base-4 encoding.
    """
    idx = 0
    for nt in kmer:
        idx = (idx << 2) | NUC_TO_INT.get(nt, 0)
    return idx


def compute_fcgr(sequence: str, k: int) -> np.ndarray:
    """
    Compute FCGR matrix for a single DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence (length ~301 bp).
    k : int
        k-mer length.

    Returns
    -------
    np.ndarray
        Normalized FCGR matrix of shape (2^k, 2^k).
    """
    size = 2 ** k
    fcgr = np.zeros((size, size), dtype=np.float32)

    seq_len = len(sequence)
    if seq_len < k:
        return fcgr

    for i in range(seq_len - k + 1):
        kmer = sequence[i:i + k]
        idx = _kmer_to_index(kmer)

        # Split index into 2D coordinates
        x = idx >> k
        y = idx & ((1 << k) - 1)

        fcgr[x, y] += 1.0

    # Normalize by total k-mers (frequency-based)
    total = fcgr.sum()
    if total > 0:
        fcgr /= total

    return fcgr


def compute_multi_k_fcgr(
    sequence: str,
    k_values: List[int] = [4, 5, 6]
) -> Dict[int, np.ndarray]:
    """
    Compute FCGRs for multiple k values.

    Parameters
    ----------
    sequence : str
        DNA sequence.
    k_values : List[int]
        List of k values.

    Returns
    -------
    Dict[int, np.ndarray]
        Mapping k -> FCGR matrix.
    """
    return {k: compute_fcgr(sequence, k) for k in k_values}
