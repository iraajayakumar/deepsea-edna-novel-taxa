"""
Generate Multi-k FCGR representations for mimic sequences.

Reads mimic sequences from a text file (one sequence per line),
computes FCGRs using the shared preprocessing module,
and saves them as a pickle file.

Output format matches original FCGRs exactly.
"""
import sys
import pickle
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.preprocessing.fcgr import compute_multi_k_fcgr


def load_mimic_sequences(mimic_path: Path):
    """
    Load mimic sequences from a text file.

    Assumes:
    - One DNA sequence per line
    - No FASTA headers
    """
    sequences = []
    with open(mimic_path, "r") as f:
        for line in f:
            seq = line.strip().upper()
            if seq:
                sequences.append(seq)
    return sequences


def main():
    # -------- Paths --------
    mimic_path = Path("data/interim/mimic_sequences.txt")
    output_path = Path("data/interim/fcgr_mimic.pkl")

    # -------- Load mimics --------
    print("Loading mimic sequences...")
    mimic_sequences = load_mimic_sequences(mimic_path)
    print(f"Loaded {len(mimic_sequences)} mimic sequences")

    # -------- Compute FCGRs --------
    fcgr_data = []

    print("Computing FCGRs for mimic sequences...")
    for seq in tqdm(mimic_sequences):
        fcgr = compute_multi_k_fcgr(seq)
        fcgr_data.append(fcgr)

    # -------- Save --------
    with open(output_path, "wb") as f:
        pickle.dump(fcgr_data, f)

    print(f"Saved mimic FCGRs to {output_path}")


if __name__ == "__main__":
    main()
