import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Learnable weighted fusion of multi-k features.
    """

    def __init__(self, num_branches, feature_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_branches))

    def forward(self, features):
        """
        features: list of tensors [B, D]
        """
        weights = F.softmax(self.weights, dim=0)

        fused = 0
        for w, f in zip(weights, features):
            fused = fused + w * f

        return fused
