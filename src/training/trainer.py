import os
import sys
import yaml
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

# --- Fix PYTHONPATH ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.models.network import MultiKFCGRNet
from src.models.iic_loss import iic_loss


class DeepSeaEDNAPairs(Dataset):
    """
    Returns (orig_fcgr_dict, mimic_fcgr_dict)
    Each dict: {k: Tensor[1, H, W]}
    """
    def __init__(self, fcgr_orig, fcgr_mimic, k_values):
        self.keys = list(fcgr_orig.keys())
        self.orig = fcgr_orig
        self.mimic = fcgr_mimic
        self.k_values = k_values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        def to_tensor(entry):
            return {
                k: torch.tensor(entry[k], dtype=torch.float32).unsqueeze(0)
                for k in self.k_values
            }

        return to_tensor(self.orig[key]), to_tensor(self.mimic[key])


def train():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load FCGR data
    with open(config["data"]["fcgr_orig"], "rb") as f:
        fcgr_orig = pickle.load(f)

    with open(config["data"]["fcgr_mimic"], "rb") as f:
        fcgr_mimic = pickle.load(f)

    print(f"Loaded {len(fcgr_orig)} sequences.")

    k_values = config["model"]["k_values"]

    dataset = DeepSeaEDNAPairs(fcgr_orig, fcgr_mimic, k_values)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        drop_last=True
    )

    model = MultiKFCGRNet(
        k_values=tuple(k_values),
        embed_dim=config["model"]["embed_dim"],
        n_clusters=config["model"]["n_clusters"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0.0

        for x1, x2 in loader:
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            p1 = model(x1)
            p2 = model(x2)

            loss = iic_loss(p1, p2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], "
              f"IIC Loss: {epoch_loss:.4f}")

    os.makedirs(config["output"]["model_dir"], exist_ok=True)
    model_path = os.path.join(config["output"]["model_dir"], "iic_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
