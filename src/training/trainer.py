import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import pickle
import os

def to_tensor(fcgr_data_entry):
    """Converts the 'fcgr' part of a data entry to a PyTorch tensor."""
    if 'fcgr' in fcgr_data_entry and isinstance(fcgr_data_entry['fcgr'], (list, torch.Tensor)):
        return torch.tensor(fcgr_data_entry['fcgr'], dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("fcgr_data_entry must contain a 'fcgr' key with a tensor-convertible value")


class DeepSeaEDNA(Dataset):
    """Custom Dataset for DeepSeaEDNA project."""
    def __init__(self, fcgr_data):
        self.fcgr_data = fcgr_data
        self.keys = list(fcgr_data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data_entry = self.fcgr_data[key]
        # Assuming `to_tensor` correctly processes the `data_entry` dictionary
        # and returns the 'fcgr' tensor, and we also need the abundance.
        # The model likely needs both fcgr and abundance.
        # Let's return both for now, assuming the model will handle them.
        fcgr_tensor = to_tensor(data_entry)
        abundance = torch.tensor(data_entry['abundance'], dtype=torch.float32)
        return fcgr_tensor, abundance


class DeepSeaClassifier(nn.Module):
    """A simple CNN-based classifier for FCGR data."""
    def __init__(self):
        super(DeepSeaClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Assuming input FCGR image is 256x256. After two pooling layers, it becomes 64x64.
        # This might need adjustment based on the actual FCGR image size.
        self.fc1 = nn.Linear(32 * 64 * 64, 128) # Adjust input features based on actual image size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1) # Output a single value (e.g., predicted abundance)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64) # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join(config['data']['path'], config['data']['fcgr_file'])
    with open(data_path, 'rb') as f:
        fcgr_data = pickle.load(f)
    print(f"Loaded {len(fcgr_data)} FCGR entries.")

    # Create dataset and dataloader
    dataset = DeepSeaEDNA(fcgr_data)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])

    # Initialize model, loss, and optimizer
    model = DeepSeaClassifier().to(device)
    criterion = nn.MSELoss() # Using MSE for abundance prediction
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1) # Reshape targets to match output of fc2

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {running_loss/len(loader):.4f}")

    print("Training complete.")
    # Save model (optional)
    # torch.save(model.state_dict(), config['model']['save_path'])

if __name__ == '__main__':
    train()