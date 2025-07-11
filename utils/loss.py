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

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

def compute_active_filters_correlation(filters, mask_weight, gumbel_temperature=1.0):
    if torch.isnan(filters).any():
        warnings.warn("Filters contain NaN.")
    if torch.isinf(filters).any():
        warnings.warn("Filters contain Inf values.")
    if torch.isnan(mask_weight).any():
        warnings.warn("Mask weights contain NaN.")
    if torch.isinf(mask_weight).any():
        warnings.warn("Mask weights contain Inf values.")
    
    # تعداد فیلترها
    num_filters = filters.shape[0]
    
    if num_filters < 2:
        device = filters.device
        print('less than 2')
        return torch.tensor(0.0, device=device)
    
    # تغییر شکل فیلترها به بردار
    filters_flat = filters.view(num_filters, -1)
    
    # بررسی واریانس فیلترها
    variance = torch.var(filters_flat, dim=1)
    zero_variance_indices = torch.where(variance == 0)[0]
    if len(zero_variance_indices) > 0:
        warnings.warn(f"{len(zero_variance_indices)} filters have zero variance.")
    
    # نرمال‌سازی فیلترها (استانداردسازی)
    mean = torch.mean(filters_flat, dim=1, keepdim=True)
    centered = filters_flat - mean
    std = torch.std(filters_flat, dim=1, keepdim=True)
    epsilon = 1e-6
    filters_normalized = centered / (std + epsilon)
    
    # نرمال‌سازی با نرم (به طول واحد)
    norm = torch.norm(filters_normalized, dim=1, keepdim=True)
    filters_normalized = filters_normalized / (norm + epsilon)
    
    # بررسی مقادیر غیرمعتبر پس از نرمال‌سازی
    if torch.isnan(filters_normalized).any():
        warnings.warn("Normalized filters contain NaN.")
    if torch.isinf(filters_normalized).any():
        warnings.warn("Normalized filters contain Inf values.")
    
    # محاسبه‌ی ماتریس همبستگی
    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t())
    
    # بررسی مقادیر غیرمعتبر در ماتریس همبستگی
    if torch.isnan(corr_matrix).any():
        warnings.warn("Correlation matrix contains NaN values.")
    if torch.isinf(corr_matrix).any():
        warnings.warn("Correlation matrix contains Inf values.")
    
    # محاسبه‌ی امتیاز همبستگی فقط برای عناصر بالای قطر اصلی
    correlation_scores = torch.sum(torch.abs(corr_matrix.triu(diagonal=1)), dim=1)
    correlation_scores = correlation_scores / max(num_filters - 1, 1)
    
    # محاسبه‌ی احتمالات ماسک با gumbel_softmax
    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1, :, :]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)  # شکل: (out_channels,)
    
    # بررسی تطابق شکل‌ها
    if mask_probs.shape[0] != correlation_scores.shape[0]:
        warnings.warn("Shape mismatch between mask_probs and correlation_scores.")
        device = filters.device
        return torch.tensor(0.0, device=device)
    
    # محاسبه‌ی هزینه همبستگی
    correlation_loss = torch.mean(correlation_scores * mask_probs)
    
    return correlation_loss

class MaskLoss(nn.Module):
    def __init__(self, correlation_weight=0.1):
        super(MaskLoss, self).__init__()
        self.correlation_weight = correlation_weight
    
    def forward(self, model):
        total_pruning_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
        
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight  # وزن‌های فیلتر
                mask_weight = m.mask_weight  # وزن‌های ماسک
                gumbel_temperature = m.gumbel_temperature  # دمای گامبل
                pruning_loss = compute_active_filters_correlation(filters, mask_weight, gumbel_temperature)
                total_pruning_loss += pruning_loss
                num_layers += 1
        
        if num_layers == 0:
            print('0 layers')
            return torch.tensor(0.0, device=device)
        
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
