import torch
import torch.nn as nn
import torch.nn.functional as F


class FCGRBranch(nn.Module):
    """
    CNN feature extractor for a single k-FCGR.
    """
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
    """
    Multi-branch FCGR network with learnable fusion
    and clustering head.
    """
    def __init__(
        self,
        k_values=(4, 5, 6),
        embed_dim=128,
        n_clusters=80
    ):
        super().__init__()

        self.k_values = k_values

        self.branches = nn.ModuleDict({
            str(k): FCGRBranch(embed_dim=embed_dim)
            for k in k_values
        })

        # Learnable fusion weights (attention-like)
        self.fusion_weights = nn.Parameter(
            torch.ones(len(k_values))
        )

        self.cluster_head = nn.Linear(embed_dim, n_clusters)

    def forward(self, x_dict):
        """
        x_dict: {k: tensor [B, 1, H, W]}
        """
        features = []

        for i, k in enumerate(self.k_values):
            feat = self.branches[str(k)](x_dict[k])
            features.append(feat)

        weights = F.softmax(self.fusion_weights, dim=0)

        fused = sum(w * f for w, f in zip(weights, features))

        logits = self.cluster_head(fused)
        probs = F.softmax(logits, dim=1)

        return probs
