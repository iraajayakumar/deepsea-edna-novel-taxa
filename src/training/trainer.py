import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from src.models.network import MultiKFCGRNet
from src.models.iic_loss import iic_loss


# -------------------------
# Dataset
# -------------------------

class FCGRPairDataset(Dataset):
    def __init__(self, fcgr_orig, fcgr_mimic, k_values):
        self.keys = list(fcgr_orig.keys())
        self.fcgr_orig = fcgr_orig
        self.fcgr_mimic = fcgr_mimic
        self.k_values = k_values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        def to_tensor(fcgr_dict):
            return {
                k: torch.tensor(fcgr_dict[k], dtype=torch.float32).unsqueeze(0)
                for k in self.k_values
            }

        return (
            to_tensor(self.fcgr_orig[key]),
            to_tensor(self.fcgr_mimic[key])
        )


# -------------------------
# Training
# -------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Drive paths
    fcgr_orig_path = "/content/drive/MyDrive/deepsea_edna_data/fcgr.pkl"
    fcgr_mimic_path = "/content/drive/MyDrive/deepsea_edna_data/fcgr_mimic.pkl"
    model_save_path = "/content/drive/MyDrive/deepsea_edna_models/iic_model_1.pt"

    with open(fcgr_orig_path, "rb") as f:
        fcgr_orig = pickle.load(f)

    with open(fcgr_mimic_path, "rb") as f:
        fcgr_mimic = pickle.load(f)

    k_values = [4, 5, 6]

    dataset = FCGRPairDataset(fcgr_orig, fcgr_mimic, k_values)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    model = MultiKFCGRNet(
        k_values=k_values,
        embed_dim=128,
        n_clusters=80
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x1, x2 in loader:
            x1 = {k: v.to(device) for k, v in x1.items()}
            x2 = {k: v.to(device) for k, v in x2.items()}

            p1 = model(x1)
            p2 = model(x2)

            loss = iic_loss(p1, p2, lambda_entropy=0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print("Model saved to:", model_save_path)


if __name__ == "__main__":
    train()
