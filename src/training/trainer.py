
import os
import sys
import yaml
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm # Import tqdm

# --- Fix PYTHONPATH ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

# Placeholder for network and loss - these would typically be imported from your project structure
# For this fix, we'll use a simple sequential model and BCEWithLogitsLoss
# You will replace these with your actual model (MultiKFCGRNet) and loss (iic_loss) imports

class CustomDataset(Dataset):
    def __init__(self, pickle_file_path, k_mer_size):
        with open(pickle_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Debug prints from previous iteration
        print(f"DEBUG: Type of loaded data in CustomDataset: {type(loaded_data)}")

        if isinstance(loaded_data, dict):
            if all(isinstance(k, int) for k in loaded_data.keys()):
                self.data_dict = loaded_data
                self.keys = list(sorted(loaded_data.keys()))
                print(f"DEBUG: Loaded data is a dict with integer keys. Number of items: {len(self.keys)}")
            else:
                raise TypeError("Dictionary keys must be integers if loaded as a dict.")
        elif isinstance(loaded_data, list):
            self.data_list = loaded_data
            print(f"DEBUG: Loaded data is a list. Number of items: {len(self.data_list)}")
        else:
            raise TypeError(f"Unsupported data type in pickle file: {type(loaded_data)}")
        self.k_mer_size = k_mer_size

    def __len__(self):
        if hasattr(self, 'data_list'):
            return len(self.data_list)
        elif hasattr(self, 'data_dict'):
            return len(self.keys)
        return 0

    def __getitem__(self, idx):
        try:
            if hasattr(self, 'data_list'):
                entry = self.data_list[idx]
            elif hasattr(self, 'data_dict'):
                actual_key = self.keys[idx]
                entry = self.data_dict[actual_key]
            else:
                raise RuntimeError("Dataset not properly initialized.")

        except (KeyError, IndexError) as e:
            print(f"ERROR in __getitem__: Failed to retrieve item for index {idx}. Exception: {e}")
            raise e

        # --- Extract FCGR data from the 'entry' ---
        if self.k_mer_size in entry:
            fcgr_array = entry[self.k_mer_size]
        else:
            print(f"WARNING: k_mer_size {self.k_mer_size} not found for index {idx}. Using first available array.")
            fcgr_array = next(iter(entry.values()))

        x = torch.from_numpy(fcgr_array).float().unsqueeze(0)

        # --- Dummy Label (replace with your actual label extraction) ---
        y = torch.tensor([0]).float() # Change to float for BCEWithLogitsLoss

        return x, y

def train():
    # Load config from the correct path relative to project root
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load FCGR data - using the 'fcgr_mimic' for this example
    pickle_file_path = config["data"]["fcgr_mimic"]
    dataset = CustomDataset(pickle_file_path, k_mer_size=config['dataset']['k_mer_size'])

    print(f"DEBUG: Final dataset length for DataLoader: {len(dataset)}")

    batch_size = config["training"]["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # --- Dummy Model, Optimizer, and Loss (replace with your actual components) ---
    # This simple linear model assumes FCGR arrays are flattened for input
    example_fcgr_shape = dataset[0][0].shape # Get shape of a single FCGR (C, H, W)
    input_features = example_fcgr_shape[1] * example_fcgr_shape[2]

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_features, config['model']['output_classes'])
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss() 

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Labels must be float and match output shape for BCEWithLogitsLoss
            # Removed .unsqueeze(1) as labels are already [batch_size, 1]
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished. Final Loss: {loss.item():.4f}")

    os.makedirs(os.path.join(PROJECT_ROOT, config["output"]["model_dir"]), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, config["output"]["model_dir"], "iic_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()
