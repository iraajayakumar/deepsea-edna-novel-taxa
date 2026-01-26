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
    Compute true Frequency Chaos Game Representation (FCGR).

    Parameters
    ----------
    sequence : str
        DNA sequence.
    k : int
        k-mer length.

    Returns
    -------
    np.ndarray
        Normalized FCGR matrix of shape (2^k, 2^k).
    """
    size = 2 ** k
    fcgr = np.zeros((size, size), dtype=np.float32)

    corners = {
        'A': (0.0, 0.0),
        'C': (0.0, 1.0),
        'G': (1.0, 1.0),
        'T': (1.0, 0.0)
    }

    seq_len = len(sequence)
    if seq_len < k:
        return fcgr

    for i in range(seq_len - k + 1):
        x, y = 0.5, 0.5  # CGR center
        valid = True

        for nt in sequence[i:i + k]:
            if nt not in corners:
                valid = False
                break
            cx, cy = corners[nt]
            x = (x + cx) / 2
            y = (y + cy) / 2

        if not valid:
            continue

        ix = int(x * size)
        iy = int(y * size)
        fcgr[ix, iy] += 1.0

    if fcgr.sum() > 0:
        fcgr /= fcgr.sum()

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
