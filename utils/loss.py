import torch
import torch.nn as nn
import torch.nn.functional as F

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

def compute_active_filters_correlation(filters, m, rank=0):
    device = filters.device

    # بررسی NaN یا Inf در فیلترها
    if torch.isnan(filters).any() or torch.isinf(filters).any():
        if rank == 0:
            warnings.warn("Filters contain NaN or Inf.")

    # بررسی NaN یا Inf در ماسک
    if torch.isnan(m).any() or torch.isinf(m).any():
        if rank == 0:
            warnings.warn("Mask contains NaN or Inf.")

    active_indices = torch.where(m.squeeze() > 0.5)[0]
    
    if len(active_indices) < 2:
        if rank == 0:
            warnings.warn(f"Fewer than 2 active filters found: {len(active_indices)}")
        return torch.tensor(0.0, device=device), active_indices

    # انتخاب فیلترهای فعال
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(len(active_indices), -1)

    # بررسی NaN یا Inf در فیلترهای فعال
    if torch.isnan(active_filters_flat).any() or torch.isinf(active_filters_flat).any():
        if rank == 0:
            warnings.warn("Active filters contain NaN or Inf.")

    # محاسبه واریانس با اپسیلون بزرگ‌تر
    variance = torch.var(active_filters_flat, dim=1, unbiased=True) + 1e-4
    valid_indices = torch.where(variance > 1e-6)[0]
    if len(valid_indices) < 2:
        if rank == 0:
            warnings.warn(f"Fewer than 2 filters with non-zero variance: {len(valid_indices)}")

    active_filters_flat = active_filters_flat[valid_indices]
    mean = torch.mean(active_filters_flat, dim=1, keepdim=True)
    centered = active_filters_flat - mean
    std = torch.sqrt(variance[valid_indices])

    # محاسبه ماتریس کوواریانس
    cov_matrix = torch.matmul(centered, centered.t()) / (active_filters_flat.size(1) - 1 + 1e-6)
    std_outer = std.unsqueeze(1) * std.unsqueeze(0)+1e-4
    correlation_matrix = cov_matrix / (std_outer + 1e-6)

    if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
        if rank == 0:
            warnings.warn("Correlation matrix contains NaN or Inf.")

    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))
    num_valid_filters = len(valid_indices)
    normalized_correlation = sum_of_squares / (num_valid_filters * (num_valid_filters - 1) / 2 + 1e-6)

    if torch.isnan(normalized_correlation) or torch.isinf(normalized_correlation):
        if rank == 0:
            warnings.warn(f"Normalized correlation is NaN or Inf: {normalized_correlation}")
        return torch.tensor(0.0, device=device), active_indices

    return normalized_correlation, active_indices


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, filters, mask):
        correlation, active_indices = compute_active_filters_correlation(filters, mask)
        return correlation, active_indices

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
