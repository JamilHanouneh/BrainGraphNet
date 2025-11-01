"""
Custom loss functions for brain connectivity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def connectivity_mse_loss(pred, target, mask=None):
    """
    MSE loss for connectivity matrices
    
    Args:
        pred: Predicted connectivity (batch_size, N, N)
        target: Target connectivity (batch_size, N, N)
        mask: Optional mask for valid connections
    
    Returns:
        loss: Scalar loss value
    """
    if mask is not None:
        diff = (pred - target) * mask
        loss = (diff ** 2).sum() / mask.sum()
    else:
        loss = F.mse_loss(pred, target)
    
    return loss


def temporal_consistency_loss(pred_sequence, lambda_tc=0.1):
    """
    Encourage temporal smoothness in predictions
    
    Args:
        pred_sequence: List of predictions over time
        lambda_tc: Weight for temporal consistency
    
    Returns:
        loss: Temporal consistency loss
    """
    if len(pred_sequence) < 2:
        return torch.tensor(0.0, device=pred_sequence[0].device)
    
    tc_loss = 0.0
    for t in range(len(pred_sequence) - 1):
        diff = pred_sequence[t+1] - pred_sequence[t]
        tc_loss += torch.mean(diff ** 2)
    
    tc_loss = tc_loss / (len(pred_sequence) - 1)
    return lambda_tc * tc_loss


def connectivity_correlation_loss(pred, target):
    """
    Correlation-based loss for connectivity
    Maximizes correlation between predicted and target
    
    Returns:
        loss: 1 - correlation (to minimize)
    """
    # Flatten matrices
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Compute correlation
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1))
    
    correlation = numerator / (pred_std * target_std + 1e-8)
    
    # Loss = 1 - correlation (minimize)
    loss = 1 - correlation.mean()
    
    return loss


class CombinedConnectivityLoss(nn.Module):
    """
    Combined loss for connectivity prediction
    MSE + Correlation + Temporal Consistency
    """
    
    def __init__(self, lambda_corr=0.5, lambda_tc=0.1):
        super(CombinedConnectivityLoss, self).__init__()
        self.lambda_corr = lambda_corr
        self.lambda_tc = lambda_tc
    
    def forward(self, pred, target, pred_sequence=None):
        # MSE loss
        mse = connectivity_mse_loss(pred, target)
        
        # Correlation loss
        corr_loss = connectivity_correlation_loss(pred, target)
        
        # Temporal consistency (if sequence provided)
        tc_loss = 0.0
        if pred_sequence is not None:
            tc_loss = temporal_consistency_loss(pred_sequence, self.lambda_tc)
        
        # Combined
        total_loss = mse + self.lambda_corr * corr_loss + tc_loss
        
        return total_loss, {
            'mse': mse.item(),
            'correlation': corr_loss.item(),
            'temporal_consistency': tc_loss.item() if isinstance(tc_loss, torch.Tensor) else tc_loss
        }
