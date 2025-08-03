import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from data.dataset import Dataset_selector

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# تعریف مدل
model = ResNet_50_sparse_hardfakevsreal()
model.dataset_type = 'rvf10k'
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# مسیر فایل وزن‌ها
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'

# بررسی وجود فایل وزن‌ها
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")

# بارگذاری وزن‌ها
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

if 'student' not in checkpoint:
    raise KeyError("'student' key not found in checkpoint")

# لود وزن‌ها بدون فیلتر کردن
state_dict = checkpoint['student']
missing, unexpected = model.load_state_dict(state_dict, strict=True)

# بررسی کلیدهای گم‌شده و غیرمنتظره
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

if missing or unexpected:
    print("Warning: There are missing or unexpected keys. Please verify checkpoint compatibility with the model architecture.")

# بررسی نمونه‌ای از وزن‌ها برای اطمینان از لود صحیح
print("Sample weights from conv1:", model.conv1.weight[0, 0, :3, :3])
print("Sample weights from feat1:", model.feat1.weight[0, 0, :3, :3])

# انتقال مدل به دستگاه و فعال کردن پرونینگ
model = model.to(device)
model.ticket = True  # اعمال ماسک‌های هرس
model.eval()

# محاسبه تعداد پارامترها و FLOPs
total_params = sum(p.numel() for p in model.parameters())
pruned_params = sum(m.mask.sum().item() for m in model.mask_modules if hasattr(m, 'mask'))
pruning_ratio = (total_params - pruned_params) / total_params * 100 if total_params > 0 else 0
flops = model.get_flops()

print(f"Total parameters: {total_params:,}")
print(f"Active (non-pruned) parameters: {pruned_params:,}")
print(f"Pruning ratio: {pruning_ratio:.2f}%")
print(f"Total FLOPs: {flops.item():,.0f}")

# بارگذاری دیتاست
data_dir = '/kaggle/input/rvf10k'
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

# ارزیابی روی دیتاست
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs, feature_list = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# چاپ نتایج
print(f'Test Loss on rvf10k: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')
print("Feature list shapes:", [f.shape for f in feature_list])
