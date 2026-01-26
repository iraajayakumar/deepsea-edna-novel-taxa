import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FCGRBranch
from .fusion import FeatureFusion


class MultiKFCGRNet(nn.Module):
    """
    Multi-branch FCGR encoder + clustering head.
    """

    def __init__(self, num_clusters, k_values, feature_dim=128):
        super().__init__()

        self.k_values = k_values
        self.branches = nn.ModuleList(
            [FCGRBranch(feature_dim) for _ in k_values]
        )

        self.fusion = FeatureFusion(
            num_branches=len(k_values),
            feature_dim=feature_dim
        )

        self.cluster_head = nn.Linear(feature_dim, num_clusters)

    def forward(self, fcgr_list):
        """
        fcgr_list: list of FCGR tensors, one per k
        Each tensor: [B, 1, H, W]
        """
        branch_features = []

        for fcgr, branch in zip(fcgr_list, self.branches):
            feat = branch(fcgr)
            branch_features.append(feat)

        fused = self.fusion(branch_features)

        logits = self.cluster_head(fused)
        probs = F.softmax(logits, dim=1)

        return probs
