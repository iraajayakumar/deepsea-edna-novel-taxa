import torch
import torch.nn.functional as F

def iic_loss(p1, p2, eps=1e-8, lambda_entropy=2.0, temperature=0.5):
    """
    FIXED IIC: Accepts LOGITS, applies temperature-controlled softmax
    """
    # Convert logits → temperature-scaled probs
    probs1 = F.softmax(p1 / temperature, dim=1)
    probs2 = F.softmax(p2 / temperature, dim=1)
    
    B, K = probs1.size()
    joint = torch.matmul(probs1.T, probs2) / B
    
    p_i = joint.sum(dim=1, keepdim=True)
    p_j = joint.sum(dim=0, keepdim=True)
    
    mi = (joint * (torch.log(joint + eps) - torch.log(p_i + eps) - torch.log(p_j + eps))).sum()
    entropy = -(p_i * torch.log(p_i + eps)).sum()
    
    loss = -mi + lambda_entropy * entropy
    return loss
