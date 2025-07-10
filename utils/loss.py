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

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, model):
    
        total_corr_loss = 0.0
        num_layers = 0

        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):

                active_filters = m.weight[m.mask.squeeze() == 1]
                if active_filters.numel() == 0:  
                    print('no active filters found')

                num_active_filters = active_filters.shape[0]
                if num_active_filters < 2:  
                    print('active filteers less than 2')

                active_filters = active_filters.view(num_active_filters, -1)

                active_filters = (active_filters - active_filters.mean(dim=1, keepdim=True)) / (
                    active_filters.std(dim=1, keepdim=True) + 1e-8
                )

                corr_matrix = torch.matmul(active_filters, active_filters.t()) / active_filters.shape[1]

                triu_indices = torch.triu_indices(row=corr_matrix.shape[0], col=corr_matrix.shape[0], offset=1)
                corr_values = corr_matrix[triu_indices[0], triu_indices[1]]

                norm2 = torch.sum(corr_values ** 2)

                normalized_loss = norm2 / num_active_filters

                total_corr_loss += normalized_loss
                num_layers += 1

        if num_layers == 0:
            print('err')
        return total_corr_loss / num_layers

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
