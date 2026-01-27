import torch
import torch.nn as nn
import torch.nn.functional as F

class FCGRBranch(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class MultiKFCGRNet(nn.Module):
    def __init__(self, k_values=[4,5,6], embed_dim=128, n_clusters=80):
        super().__init__()
        self.k_values = [str(k) for k in k_values]  # Store as strings
        self.branches = nn.ModuleDict({
            str(k): FCGRBranch(embed_dim=embed_dim) for k in k_values
        })
        self.fusion_weights = nn.Parameter(torch.ones(len(self.k_values)))
        self.cluster_head = nn.Linear(embed_dim, n_clusters)
    
    def forward(self, x_dict):
        """x_dict keys: ['4','5','6'] -> tensors (B,1,H,W)"""
        features = []
        for k_str in self.k_values:
            feat = self.branches[k_str](x_dict[k_str])
            features.append(feat)
        
        # Learnable fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, features))
        logits = self.cluster_head(fused)
        probs = F.softmax(logits, dim=1)      # ✅ NORMAL probs [0,1]
        return probs

if __name__ == "__main__":
    model = MultiKFCGRNet()
    print("Model created successfully!")
