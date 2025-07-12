import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

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


def compute_active_filters_correlation(filters, m):
    if torch.isnan(filters).any():
        print("Filters contain NaN")
        
    if torch.isinf(filters).any():
        print("Filters contain Inf")
        
    if torch.isnan(m).any():
        print("Masks contain NaN")
        
    if torch.isinf(m).any():
        print("Masks contain Inf")
    
    active_indices = torch.where(m == 1)[0]

    if len(active_indices) < 2:
        print("Fewer than 2 active filters found")
        
    active_filters = filters[active_indices]
    active_filters_flat = active_filters.view(active_filters.size(0), -1)

    if torch.isnan(active_filters_flat).any() :
        warnings.warn("Active filters contain NaN.")
        
    if torch.isinf(active_filters_flat).any():
        warnings.warn("Active filters contain Inf values.")

    # Calculate variance for each filter
    variance = torch.var(active_filters_flat, dim=1)
    
    # Identify filters with zero variance
    zero_variance_indices = torch.where(variance == 0)[0]
    if len(zero_variance_indices) > 0:
        print("The following filters have zero variance:")
        for idx in zero_variance_indices:
            filter_weights = active_filters_flat[idx]
            print(f"Filter at index {active_indices[idx].item()}: values = {filter_weights.tolist()}")
    
    # Check the number of valid filters (non-zero variance)
    valid_indices = torch.where(variance > 0)[0]
    if len(valid_indices) < 2:
        warnings.warn("Fewer than 2 filters with non-zero variance found.")

    # Continue calculations only with valid filters
    active_filters_flat = active_filters_flat[valid_indices]
    mean = torch.mean(active_filters_flat, dim=1, keepdim=True)
    centered = active_filters_flat - mean
    cov_matrix = torch.matmul(centered, centered.t()) / (active_filters_flat.size(1) - 1)
    std = torch.sqrt(variance[valid_indices])

    epsilon = 1e-6
    std_outer = std.unsqueeze(1) * std.unsqueeze(0)
    correlation_matrix = cov_matrix / (std_outer + epsilon)

    if torch.isnan(correlation_matrix).any():
        warnings.warn("Correlation matrix contains NaN values.")

    if torch.isinf(correlation_matrix).any():
        warnings.warn("Correlation matrix contains Inf values.")

    upper_tri = torch.triu(correlation_matrix, diagonal=1)
    sum_of_squares = torch.sum(torch.pow(upper_tri, 2))

    num_active_filters = len(valid_indices)
    normalized_correlation = sum_of_squares / num_active_filters
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
