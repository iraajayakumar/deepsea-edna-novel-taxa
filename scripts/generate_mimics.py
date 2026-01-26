"""
Generate mimic sequences for self-supervised eDNA training.

Reads clean.fasta and produces original–mimic sequence pairs.
"""

import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import jsonS

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.augmentation.mimic_generator import generate_mimic_pairs


# ---------------------------------------------------------
# FASTA loader
# ---------------------------------------------------------

def load_fasta(fasta_path: Path) -> List[str]:
    sequences = []
    seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append("".join(seq))
                    seq = []
            else:
                seq.append(line.upper())

        if seq:
            sequences.append("".join(seq))

    return sequences


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    fasta_path = Path("data/processed/clean.fasta (Non-chimera_final)")
    output_dir = Path("data/interim")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sequences...")
    sequences = load_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences")

    # -------------------------------------------------
    # For now: uniform abundance = 1
    # -------------------------------------------------
    abundance_map: Dict[str, int] = {}

    print("Generating mimic pairs...")
    pairs = generate_mimic_pairs(
        sequences=sequences,
        abundance_map=abundance_map,
        n_mimics=1,
        allow_indels=False
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    originals = [o for o, _ in pairs]
    mimics = [m for _, m in pairs]

    with open(output_dir / "original_sequences.txt", "w") as f:
        for seq in originals:
            f.write(seq + "\n")

    with open(output_dir / "mimic_sequences.txt", "w") as f:
        for seq in mimics:
            f.write(seq + "\n")

    print(f"Saved {len(pairs)} mimic pairs to {output_dir}")


if __name__ == "__main__":
    main()
