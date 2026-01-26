import torch


def iic_loss(
    p1,
    p2,
    eps=1e-8,
    lambda_entropy=0.0
):
    """
    Invariant Information Clustering loss.

    Parameters
    ----------
    p1, p2 : torch.Tensor
        Shape [B, K] cluster probability distributions.
    lambda_entropy : float
        Set >0 only if you want balanced clusters.
        For novel taxa discovery, keep this LOW or 0.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    B, K = p1.size()

    # Joint probability matrix
    joint = torch.matmul(p1.T, p2)
    joint = joint / joint.sum()

    # Marginals
    p_i = joint.sum(dim=1, keepdim=True)
    p_j = joint.sum(dim=0, keepdim=True)

    # Mutual Information
    mi = joint * (
        torch.log(joint + eps)
        - torch.log(p_i + eps)
        - torch.log(p_j + eps)
    )
    mi = mi.sum()

    # Optional entropy regularization
    entropy = -(p_i * torch.log(p_i + eps)).sum()

    loss = -mi + lambda_entropy * entropy
    return loss
