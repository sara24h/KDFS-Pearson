import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from thop import profile
from model.student.ResNet_sparse import SoftMaskedConv2d
from data.dataset import FaceDataset, Dataset_selector
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet_50_sparse_hardfakevsreal()
model.dataset_type = 'rvf10k'  
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)


checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

if 'student' in checkpoint:
    state_dict = checkpoint['student']

    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('feat')}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
else:
    raise KeyError("'student' key not found in checkpoint")

# انتقال مدل به دستگاه و فعال کردن پرونینگ
model = model.to(device)
model.ticket = True  # اعمال ماسک‌های هرس
model.eval()

# بارگذاری دیتاست جدید (مثلاً rvf10k)
data_dir = '/kaggle/input/rvf10k'  # مسیر دیتاست را تنظیم کنید
dataset = Dataset_selector(
    dataset_mode='rvf10k',
    rvf10k_train_csv=os.path.join(data_dir, 'train.csv'),
    rvf10k_valid_csv=os.path.join(data_dir, 'valid.csv'),
    rvf10k_root_dir=data_dir,
    train_batch_size=64,
    eval_batch_size=64,
    num_workers=4,
    pin_memory=True,
    ddp=False
)
test_loader = dataset.loader_test

# تعریف معیار
criterion = nn.BCEWithLogitsLoss()

# ارزیابی روی دیتاست جدید
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs, _ = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f'Test Loss on rvf10k: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')
