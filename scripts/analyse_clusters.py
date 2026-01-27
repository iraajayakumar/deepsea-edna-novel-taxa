import torch
import pickle
import numpy as np
import sys
from pathlib import Path
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.network import MultiKFCGRNet

MODEL_PATH = "models/iic_model01.pt"
FCGR_PATH  = "data/interim/fcgr.pkl"

# Load model
model = MultiKFCGRNet(
    k_values=(4, 5, 6),
    embed_dim=128,
    n_clusters=80
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load FCGRs
with open(FCGR_PATH, "rb") as f:
    fcgr = pickle.load(f)

cluster_ids = []

with torch.no_grad():
    for seq_id, entry in fcgr.items():
        x = {
            k: torch.tensor(entry[k], dtype=torch.float32)
                  .unsqueeze(0)  # channel
                  .unsqueeze(0)  # batch
                  .to(DEVICE)
            for k in (4, 5, 6)
        }

        p = model(x)                # (1, n_clusters)
        cluster = p.argmax(dim=1).item()
        cluster_ids.append(cluster)

counts = Counter(cluster_ids)

print("\nCluster usage:")
for k, v in counts.most_common():
    print(f"Cluster {k:02d}: {v}")
