import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Fix PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.models.paired_dataset import PairedFCGRDataset
from src.models.network import MultiKFCGRNet
from src.models.iic_loss import iic_loss

def train():
    # Config (create configs/config.yaml if missing)
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = {
        'data': {'fcgr_orig': 'data/interim/fcgr.pkl', 'fcgr_mimic': 'data/interim/fcgr_mimic.pkl'},
        'model': {'n_clusters': 300},  # ← CHANGE 80→300
        'training': {'batch_size': 512, 'learning_rate': 0.0001, 'epochs': 150, 'lambda_entropy': 2.0}  # ← FULL UPDATE
    }

    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config['data'].update(full_config.get('data', {}))
            config['model'].update(full_config.get('model', {}))
            config['training'].update(full_config.get('training', {}))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    print(f"Config: {config['training']}")

    # Dataset + Dataloader
    dataset = PairedFCGRDataset(config['data']['fcgr_orig'], config['data']['fcgr_mimic'])
    loader = DataLoader(dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        num_workers=0,              # ← CHANGE: 4→0 (Colab safe)
        pin_memory=torch.cuda.is_available()  # ← ADD
    )


    # Model + Optimizer
    model = MultiKFCGRNet(
        k_values=[4,5,6], 
        embed_dim=128, 
        n_clusters=config['model']['n_clusters']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # Training Loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for orig_fcgr, mimic_fcgr in pbar:
            # To device (dict of tensors)
            orig_fcgr = {k: v.to(device) for k, v in orig_fcgr.items()}
            mimic_fcgr = {k: v.to(device) for k, v in mimic_fcgr.items()}
            
            optimizer.zero_grad()
            
            # Forward pass (same model, different augmentations)
            p1 = model(orig_fcgr)  # (B, K) log probs
            p2 = model(mimic_fcgr) # Paired
            
            # IIC Loss - maximize MI between pairs
            loss = -iic_loss(p1, p2, lambda_entropy=config['training']['lambda_entropy'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
    
    # Save
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, "models", "iic_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")

if __name__ == '__main__':
    train()
