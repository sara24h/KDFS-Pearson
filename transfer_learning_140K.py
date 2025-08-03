import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from data.dataset import Dataset_selector
from thop import profile

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# تعریف مدل sparse برای استخراج ماسک‌ها
sparse_model = ResNet_50_sparse_hardfakevsreal()
sparse_model.dataset_type = 'rvf10k'

# بارگذاری وزن‌ها
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

if 'student' not in checkpoint:
    raise KeyError("'student' key not found in checkpoint")

# لود وزن‌ها به مدل sparse
sparse_model.load_state_dict(checkpoint['student'], strict=True)

# بررسی کلیدهای گم‌شده و غیرمنتظره
missing, unexpected = sparse_model.load_state_dict(checkpoint['student'], strict=True)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# بررسی وزن‌های feat1
print("Sample weights from feat1 in checkpoint:", checkpoint['student']['feat1.weight'][0, 0, :3, :3])
print("feat1 weight shape in checkpoint:", checkpoint['student']['feat1.weight'].shape)

# استخراج ماسک‌ها
mask_weights = [m.mask_weight for m in sparse_model.mask_modules]
masks = [torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1) for mask_weight in mask_weights]

# تعریف مدل پرون‌شده
pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
pruned_model.dataset_type = 'rvf10k'

# انتقال وزن‌های غیر پرون‌شده به مدل پرون‌شده
state_dict_pruned = {}
for key, value in checkpoint['student'].items():
    if 'mask_weight' not in key:  # نادیده گرفتن ماسک‌ها
        state_dict_pruned[key] = value
missing, unexpected = pruned_model.load_state_dict(state_dict_pruned, strict=True)
print(f"Missing keys in pruned model: {missing}")
print(f"Unexpected keys in pruned model: {unexpected}")

# انتقال مدل پرون‌شده به دستگاه و فعال کردن حالت ارزیابی
pruned_model = pruned_model.to(device)
pruned_model.eval()

# محاسبه FLOPs و پارامترها با thop
input_size = (1, 3, 256, 256)  # اندازه تصویر برای rvf10k
input_tensor = torch.randn(input_size).to(device)
macs, params = profile(pruned_model, inputs=(input_tensor,), verbose=False)

# محاسبه تعداد پارامترهای فعال
total_params = sum(p.numel() for p in pruned_model.parameters())
active_params = total_params  # در مدل پرون‌شده، تمام پارامترها فعال هستند
pruning_ratio = (29_129_793 - active_params) / 29_129_793 * 100  # مقایسه با مدل اصلی

print(f"Total parameters (pruned model): {total_params:,}")
print(f"Active parameters (pruned model): {active_params:,}")
print(f"Pruning ratio (compared to sparse model): {pruning_ratio:.2f}%")
print(f"Total FLOPs (pruned model): {macs / 1e6:.2f} MFLOPs")

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

# ارزیابی روی دیتاست با مدل پرون‌شده
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = pruned_model(images).squeeze(1)  # مدل پرون‌شده feature_list ندارد
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# چاپ نتایج
print(f'Test Loss on rvf10k: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')
