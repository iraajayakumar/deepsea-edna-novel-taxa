import pickle
from pathlib import Path

FCGR_PATH = Path("data/interim/fcgr.pkl")
N_SHOW = 3  # how many sequences to inspect


def main():
    with open(FCGR_PATH, "rb") as f:
        fcgr_data = pickle.load(f)

    print(f"Total sequences in fcgr.pkl: {len(fcgr_data)}\n")

    for i, (seq_id, record) in enumerate(fcgr_data.items()):
        if i >= N_SHOW:
            break

        abundance = record.get("abundance", None)
        multi_k_fcgr = record["fcgr"]

        print(f"Sequence ID: {seq_id}")
        print(f"  Abundance: {abundance}")

        for k, mat in multi_k_fcgr.items():
            print(
                f"  k={k}: shape={mat.shape}, "
                f"sum={mat.sum():.4f}, "
                f"min={mat.min():.4e}, "
                f"max={mat.max():.4e}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()
