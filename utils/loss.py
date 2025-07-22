import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.student.layer import SoftMaskedConv2d

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

    def forward(self, logits_teacher, logits_student, temperature):
        kd_loss = F.binary_cross_entropy_with_logits(
            logits_student / temperature,
            torch.sigmoid(logits_teacher / temperature), 
            reduction='mean'
        )
        return kd_loss

class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()

import warnings

def compute_filter_correlation(filters, mask_weight, gumbel_temperature=1.0):
    if torch.isnan(filters).any():
        warnings.warn("Filters contain NaN.")
    if torch.isinf(filters).any():
        warnings.warn("Filters contain Inf values.")
    if torch.isnan(mask_weight).any():
        warnings.warn("Mask weights contain NaN.")
    if torch.isinf(mask_weight).any():
        warnings.warn("Mask weights contain Inf values.")
    
    num_filters = filters.shape[0]
    
    if num_filters < 2:
        warnings.warn("Less than 2 filters, returning zero loss.")
        return torch.tensor(0.0, device=filters.device)
    
    filters_flat = filters.view(num_filters, -1)
    variance = torch.var(filters_flat, dim=1)
    valid_indices = torch.where(variance > 0)[0]
    
    if len(valid_indices) < 2:
        warnings.warn(f"Only {len(valid_indices)} filters with non-zero variance found.")
    
    if len(valid_indices) < num_filters:
        warnings.warn(f"{num_filters - len(valid_indices)} filters have zero variance.")
    
    active_filters_flat = filters_flat[valid_indices]
    mean = torch.mean(active_filters_flat, dim=1, keepdim=True)
    centered = active_filters_flat - mean
    cov_matrix = torch.matmul(centered, centered.t()) / (active_filters_flat.size(1) - 1)
    std = torch.sqrt(variance[valid_indices])
    epsilon = 1e-4  
    std_outer = std.unsqueeze(1) * std.unsqueeze(0)
    corr_matrix = cov_matrix / (std_outer + epsilon)
    
    if torch.isnan(corr_matrix).any():
        warnings.warn("Correlation matrix contains NaN values.")
    if torch.isinf(corr_matrix).any():
        warnings.warn("Correlation matrix contains Inf values.")
    
    mask = ~torch.eye(len(valid_indices), len(valid_indices), device=filters.device).bool()
    
    correlation_scores = torch.sum((corr_matrix * mask.float())**2, dim=1) / max(len(valid_indices) - 1, 1)
    
    if torch.isnan(correlation_scores).any():
        warnings.warn("Correlation scores contain NaN values.")
    if torch.isinf(correlation_scores).any():
        warnings.warn("Correlation scores contain Inf values.")
    
    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1, :, :]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)
    mask_probs = mask_probs[valid_indices]
    
    if mask_probs.shape[0] != correlation_scores.shape[0]:
        warnings.warn(f"Shape mismatch between mask_probs ({mask_probs.shape[0]}) and correlation_scores ({correlation_scores.shape[0]}).")
    
    correlation_loss = torch.mean(correlation_scores * mask_probs)
    
    return correlation_loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, model):
   
        total_pruning_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
        
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight  
                mask_weight = m.mask_weight 
                gumbel_temperature = m.gumbel_temperature 
                pruning_loss = compute_filter_correlation(filters, mask_weight, gumbel_temperature)
                total_pruning_loss += pruning_loss
                num_layers += 1
        
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")

        total_loss = total_pruning_loss / num_layers
        return total_loss

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
