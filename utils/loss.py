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

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.student.layer import SoftMaskedConv2d

class MaskLoss(nn.Module):
    def __init__(self, correlation_weight=0.1):
        super(MaskLoss, self).__init__()
        self.correlation_weight = correlation_weight  # Weight for correlation-based pruning term

    def forward(self, model):
        total_pruning_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device

        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                # 1. Compute correlation between filters
                filters = m.weight  # Shape: (out_channels, in_channels, kernel_size, kernel_size)
                num_filters = filters.shape[0]

                if num_filters < 2:
                    # If fewer than 2 filters, no pruning loss
                    pruning_loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    # Normalize filters
                    filters = filters.view(num_filters, -1)
                    filters = (filters - filters.mean(dim=1, keepdim=True)) / (
                        filters.std(dim=1, keepdim=True) + 1e-8
                    )
                    norm = torch.norm(filters, dim=1, keepdim=True)
                    filters_normalized = filters / (norm + 1e-8)

                    # Compute correlation matrix (Pearson correlation)
                    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t())
                    
                    # Compute correlation scores for each filter
                    correlation_scores = torch.sum(torch.abs(corr_matrix), dim=1) - 1
                    correlation_scores = correlation_scores / max(num_filters - 1, 1)

                    # 2. Pruning loss based on correlation scores
                    mask_probs = F.softmax(m.mask_weight, dim=1)[:, 1, :, :]  # Shape: (out_channels, 1, 1)
                    mask_probs = mask_probs.squeeze(-1).squeeze(-1)  # Shape: (out_channels,)
                    
                    # Ensure shapes match
                    if mask_probs.shape[0] == correlation_scores.shape[0]:
                        pruning_loss = torch.mean(correlation_scores * mask_probs)
                    else:
                        pruning_loss = torch.tensor(0.0, device=device, requires_grad=True)

                total_pruning_loss += pruning_loss
                num_layers += 1

        if num_layers == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Total loss
        total_loss = self.correlation_weight * (total_pruning_loss / num_layers)
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
