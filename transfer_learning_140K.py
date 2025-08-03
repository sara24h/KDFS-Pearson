import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import math
import copy
import pandas as pd
import os
from PIL import Image
from thop import profile

# ایمپورت از فایل‌های مدل و دیتاست
from model.student.layer import SoftMaskedConv2d
from model.student.ResNet_sparse import ResNet_50_sparse_140k
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from data.dataset import FaceDataset, Dataset_selector

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ایجاد مدل
model = ResNet_50_sparse_140k(
    gumbel_start_temperature=2.0,
    gumbel_end_temperature=0.5,
    num_epochs=200
)

# لود فایل وزن‌ها
checkpoint_path = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print("وزن‌ها از کلید 'student' با موفقیت لود شدند.")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print("وزن‌ها مستقیماً لود شدند.")
except Exception as e:
    print(f"خطا در لود وزن‌ها: {e}")
    checkpoint_keys = set(checkpoint['student'].keys() if 'student' in checkpoint else checkpoint.keys())
    model_keys = set(model.state_dict().keys())
    print("کلیدهای گمشده:", model_keys - checkpoint_keys)
    print("کلیدهای غیرمنتظره:", checkpoint_keys - model_keys)
    exit()

# انتقال مدل به دستگاه و تنظیم حالت ticket
model = model.to(device)
model.ticket = True
model.eval()

# لیست دیتاست‌ها برای ارزیابی
datasets_to_evaluate = [
    {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
        'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces',
    },
    {
        'dataset_mode': 'rvf10k',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
    },
    {
        'dataset_mode': '200k',
        'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
        'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
    },
    {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
    },
    {
        'dataset_mode': '330k',
        'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
    },
]

# ذخیره نتایج
results = {}

# ارزیابی روی هر دیتاست
for dataset_config in datasets_to_evaluate:
    dataset_mode = dataset_config['dataset_mode']
    print(f"\nارزیابی روی دیتاست تست {dataset_mode}...")

    try:
        # بارگذاری دیتاست تست
        dataset = Dataset_selector(
            dataset_mode=dataset_mode,
            hardfake_csv_file=dataset_config.get('hardfake_csv_file'),
            hardfake_root_dir=dataset_config.get('hardfake_root_dir'),
            rvf10k_valid_csv=dataset_config.get('rvf10k_valid_csv'),
            rvf10k_root_dir=dataset_config.get('rvf10k_root_dir'),
            realfake200k_test_csv=dataset_config.get('realfake200k_test_csv'),
            realfake200k_root_dir=dataset_config.get('realfake200k_root_dir'),
            realfake190k_root_dir=dataset_config.get('realfake190k_root_dir'),
            realfake330k_root_dir=dataset_config.get('realfake330k_root_dir'),
            eval_batch_size=64,
            num_workers=8,
            pin_memory=True,
            ddp=False,
        )

        # ارزیابی مدل
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataset.loader_test:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # محاسبه معیارهای ارزیابی
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        # ذخیره نتایج
        results[dataset_mode] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }

        print(f"نتایج ارزیابی روی دیتاست تست {dataset_mode}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    except Exception as e:
        print(f"خطا در ارزیابی دیتاست {dataset_mode}: {e}")

# محاسبه FLOPs و تعداد پارامترها
flops = model.get_flops()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nFLOPs: {flops / 1e6:.2f} MFLOPs")
print(f"تعداد پارامترها: {total_params / 1e6:.2f} M")

# ذخیره نتایج در فایل
with open('evaluation_results.txt', 'w') as f:
    for dataset_mode, metrics in results.items():
        f.write(f"\nDataset: {dataset_mode}\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['F1-Score']:.4f}\n")
    f.write(f"\nFLOPs: {flops / 1e6:.2f} MFLOPs\n")
    f.write(f"تعداد پارامترها: {total_params / 1e6:.2f} M\n")
