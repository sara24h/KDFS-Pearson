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

# ایمپورت از فایل‌های ارائه‌شده
from model.student.layer import SoftMaskedConv2d
from model.student.ResNet_sparse import ResNet_50_sparse_140k
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# تنظیم دستگاه (GPU یا CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ایجاد مدل
model = ResNet_50_sparse_140k(
    gumbel_start_temperature=2.0,
    gumbel_end_temperature=0.5,
    num_epochs=200
)

# لود وزن‌ها
checkpoint_path = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("وزن‌ها با موفقیت لود شدند.")
except Exception as e:
    print(f"خطا در لود وزن‌ها: {e}")
    exit()

# انتقال مدل به دستگاه و تنظیم حالت ticket
model = model.to(device)
model.ticket = True  # استفاده از ماسک‌های سخت
model.eval()


# بارگذاری دیتاست تست
test_dataset = datasets.ImageFolder(root='path_to_test_dataset', transform=transform)  # مسیر دیتاست تست را وارد کنید
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

# اختیاری: ایجاد مدل هرس‌شده (pruned) و انتقال ماسک‌ها
masks = [m.mask for m in model.mask_modules]
pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
pruned_model = pruned_model.to(device)
pruned_model.eval()
print("مدل هرس‌شده (pruned) با موفقیت ایجاد شد.")
