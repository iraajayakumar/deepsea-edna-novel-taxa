import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
FCGR_PATH = Path("data/interim/fcgr.pkl")
K_VALUES = [4, 5, 6]
N_READS = 1000          # tune as needed
RANDOM_SEED = 42
OUT_DIR = Path("outputs/aggregated_fcgr")
# ----------------------------------------


def log_normalize(mat):
    """Log-scale and normalize for visualization only."""
    mat = np.log1p(mat)
    return mat / mat.max() if mat.max() > 0 else mat


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    with open(FCGR_PATH, "rb") as f:
        fcgr_data = pickle.load(f)

    seq_ids = list(fcgr_data.keys())

    if N_READS > len(seq_ids):
        raise ValueError("N_READS exceeds number of sequences")

    sampled_ids = random.sample(seq_ids, N_READS)

    # Initialize aggregation
    aggregated = {k: None for k in K_VALUES}

    for sid in sampled_ids:
        fcgr_maps = fcgr_data[sid]["fcgr"]

        for k in K_VALUES:
            mat = fcgr_maps[k]
            aggregated[k] = mat if aggregated[k] is None else aggregated[k] + mat

    # Normalize to frequencies
    for k in K_VALUES:
        total = aggregated[k].sum()
        if total > 0:
            aggregated[k] /= total

    # Save images
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for k in K_VALUES:
        mat = log_normalize(aggregated[k])

        fig = plt.figure(figsize=(4, 4))
        plt.imshow(
            mat,
            cmap="gray",
            origin="lower",
            interpolation="nearest"
        )
        plt.axis("off")

        out_path = OUT_DIR / f"fcgr_k{k}.png"
        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
