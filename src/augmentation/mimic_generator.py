"""
Mimic Sequence Generator for Self-Supervised eDNA Clustering

This module generates biologically plausible mimic sequences
from original DNA reads using abundance-aware augmentations.

Mimic pairs (original, mimic) are used as positive pairs
in self-supervised learning with mutual information maximization.

Designed for:
- Deep-sea eDNA
- Long-tail abundance distributions
- Novel taxa preservation
"""

import random
import math
from typing import Dict, List, Tuple

DNA_ALPHABET = ["A", "C", "G", "T"]


# ---------------------------------------------------------
# Abundance → mutation rate mapping
# ---------------------------------------------------------

def mutation_rate_from_abundance(
    abundance: int,
    min_rate: float = 0.001,
    max_rate: float = 0.02
) -> float:
    """
    Compute mutation rate as a smooth function of abundance.

    Rare sequences → very weak augmentation
    Abundant sequences → stronger augmentation

    Uses log-scaling to avoid over-penalizing extreme abundances.
    """
    if abundance <= 1:
        return min_rate

    log_ab = math.log10(abundance + 1)
    scaled = min(log_ab / 4.0, 1.0)

    return min_rate + scaled * (max_rate - min_rate)


# ---------------------------------------------------------
# Core mutation operations
# ---------------------------------------------------------

def point_mutation(sequence: str, mutation_rate: float) -> str:
    """
    Apply random point mutations to a DNA sequence.

    Mutation count scales with sequence length and mutation_rate.
    """
    seq = list(sequence)
    n_mutations = max(1, int(len(seq) * mutation_rate))

    positions = random.sample(range(len(seq)), n_mutations)

    for pos in positions:
        original = seq[pos]
        alternatives = [b for b in DNA_ALPHABET if b != original]
        seq[pos] = random.choice(alternatives)

    return "".join(seq)


def indel_mutation(sequence: str, indel_rate: float = 0.002) -> str:
    """
    Optional small insertions/deletions to simulate
    sequencing / fragmentation noise.

    Very conservative by default.
    """
    seq = list(sequence)
    if random.random() > indel_rate:
        return sequence

    pos = random.randint(0, len(seq) - 1)

    if random.random() < 0.5 and len(seq) > 50:
        # deletion
        del seq[pos]
    else:
        # insertion
        seq.insert(pos, random.choice(DNA_ALPHABET))

    return "".join(seq)


# ---------------------------------------------------------
# Main mimic generation logic
# ---------------------------------------------------------

def generate_mimic(
    sequence: str,
    abundance: int,
    allow_indels: bool = False
) -> str:
    """
    Generate a single mimic sequence from an original read.

    Augmentation strength depends on abundance.
    """
    mutation_rate = mutation_rate_from_abundance(abundance)

    mimic = point_mutation(sequence, mutation_rate)

    if allow_indels:
        mimic = indel_mutation(mimic)

    return mimic


def generate_mimic_pairs(
    sequences: List[str],
    abundance_map: Dict[str, int],
    n_mimics: int = 1,
    allow_indels: bool = False
) -> List[Tuple[str, str]]:
    """
    Generate (original, mimic) pairs for self-supervised training.

    Each sequence can produce one or more mimics.
    """
    pairs = []

    for seq in sequences:
        abundance = abundance_map.get(seq, 1)

        for _ in range(n_mimics):
            mimic = generate_mimic(
                sequence=seq,
                abundance=abundance,
                allow_indels=allow_indels
            )
            pairs.append((seq, mimic))

    return pairs


# ---------------------------------------------------------
# Debug / sanity check
# ---------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    example_seq = "ACGT" * 75  # length ~300
    example_abundance = 5000

    mimic = generate_mimic(example_seq, example_abundance)
    print("Original:", example_seq[:50], "...")
    print("Mimic:   ", mimic[:50], "...")
