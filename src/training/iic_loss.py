import torch
import torch.nn.functional as F


def compute_joint(p, q, weights=None, eps=1e-8):
    """
    Compute joint probability matrix P_ij.
    p, q: [B, K] cluster probabilities
    weights: [B] optional abundance-based weights
    """
    B, K = p.size()

    if weights is not None:
        weights = weights.view(B, 1)
        p = p * weights
        q = q * weights

    joint = torch.matmul(p.t(), q)
    joint = joint / joint.sum()

    joint = torch.clamp(joint, eps, 1.0)
    return joint


def iic_loss(
    p,
    q,
    weights=None,
    lambda_entropy=0.2,
    eps=1e-8
):
    """
    Invariant Information Clustering loss.

    Adaptations:
    - relaxed entropy regularization
    - abundance-weighted joint distribution
    """

    joint = compute_joint(p, q, weights)

    marginal_p = joint.sum(dim=1, keepdim=True)
    marginal_q = joint.sum(dim=0, keepdim=True)

    mi = joint * (
        torch.log(joint)
        - torch.log(marginal_p)
        - torch.log(marginal_q)
    )

    mi = mi.sum()

    # Optional entropy regularization (softened)
    entropy = -torch.sum(marginal_p * torch.log(marginal_p + eps))

    loss = -mi + lambda_entropy * entropy
    return loss
