import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import math
import copy
from thop import profile

# ایمپورت از فایل‌های مدل
from model.student.layer import SoftMaskedConv2d
from model.student.ResNet_sparse import ResNet_50_sparse_140k
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

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
    # لود وزن‌ها از کلید 'student'
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
        # بررسی پیشوند 'module.'
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print("وزن‌ها از کلید 'student' با موفقیت لود شدند.")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print("وزن‌ها مستقیماً لود شدند.")
except Exception as e:
    print(f"خطا در لود وزن‌ها: {e}")
    # چاپ کلیدهای گمشده یا اضافی برای دیباگ
    checkpoint_keys = set(checkpoint['student'].keys() if 'student' in checkpoint else checkpoint.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    print("کلیدهای گمشده:", missing_keys)
    print("کلیدهای غیرمنتظره:", unexpected_keys)
    exit()

# انتقال مدل به دستگاه و تنظیم حالت ticket
model = model.to(device)
model.ticket = True  # استفاده از ماسک‌های سخت
model.eval()

# تعریف پیش‌پردازش
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# بارگذاری دیتاست تست (مسیر را اصلاح کنید)
test_dataset = datasets.ImageFolder(root='path_to_test_dataset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ارزیابی مدل
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()  # برای خروجی باینری
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# محاسبه معیارهای ارزیابی
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print(f"نتایج ارزیابی روی دیتاست تست:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# محاسبه FLOPs
flops = model.get_flops()
print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")

# محاسبه تعداد پارامترها
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {total_params / 1e6:.2f} M")
