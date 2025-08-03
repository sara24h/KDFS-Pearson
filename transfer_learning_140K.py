import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from thop import profile
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# تابع برای پرون کردن وزن‌ها
def prune_weights(state_dict, masks):
    pruned_state_dict = {}
    mask_idx = 0
    block_mask_indices = []  # ذخیره اندیس‌های ماسک برای هر بلوک
    current_block = None
    pruned_channels = {}  # ذخیره تعداد کانال‌های پرون‌شده برای هر لایه

    for name, param in state_dict.items():
        print(f"Processing layer: {name}, mask_idx: {mask_idx}, block_mask_indices: {block_mask_indices}, current_block: {current_block}")

        # نادیده گرفتن کلیدهای feat و mask_weight
        if 'feat' in name or 'mask_weight' in name:
            continue

        # کپی مستقیم لایه‌های conv1 و bn1 (لایه‌های ابتدایی بدون ماسک)
        if 'conv1' in name and 'layer' not in name:
            pruned_state_dict[name] = param
            pruned_channels[name] = param.shape[0]
            continue
        if 'bn1' in name and 'layer' not in name:
            pruned_state_dict[name] = param
            pruned_channels[name] = param.shape[0] if 'num_batches_tracked' not in name else 0
            continue

        # شناسایی بلوک فعلی (مثل layer1.0, layer1.1, ...)
        block_name = '.'.join(name.split('.')[:2]) if 'layer' in name else None
        if block_name and block_name != current_block:
            print(f"New block detected: {block_name}, resetting block_mask_indices")
            current_block = block_name
            block_mask_indices = []  # ریست کردن اندیس‌های ماسک برای بلوک جدید

        # پرون کردن لایه‌های conv در layer1 تا layer4
        if 'conv' in name and 'layer' in name:
            if 'conv1' in name:
                if mask_idx >= len(masks):
                    raise ValueError(f"mask_idx {mask_idx} out of range for masks with length {len(masks)}")
                if param.shape[0] != masks[mask_idx].shape[0]:
                    raise ValueError(f"Dimension mismatch for {name}: weight shape {param.shape[0]}, mask shape {masks[mask_idx].shape[0]}")
                pruned_param = param[masks[mask_idx] == 1]
                pruned_state_dict[name] = pruned_param
                pruned_channels[name] = pruned_param.shape[0]
                print(f"Pruned {name} to {pruned_param.shape}")
                block_mask_indices.append(mask_idx)
                mask_idx += 1
            elif 'conv2' in name:
                if mask_idx >= len(masks):
                    raise ValueError(f"mask_idx {mask_idx} out of range for masks with length {len(masks)}")
                if param.shape[0] != masks[mask_idx].shape[0] or param.shape[1] != masks[mask_idx - 1].shape[0]:
                    raise ValueError(f"Dimension mismatch for {name}: weight shape {param.shape}, mask shape {masks[mask_idx].shape[0]}x{masks[mask_idx - 1].shape[0]}")
                pruned_param = param[masks[mask_idx] == 1][:, masks[mask_idx - 1] == 1]
                pruned_state_dict[name] = pruned_param
                pruned_channels[name] = pruned_param.shape[0]
                print(f"Pruned {name} to {pruned_param.shape}")
                block_mask_indices.append(mask_idx)
                mask_idx += 1
            elif 'conv3' in name:
                if mask_idx >= len(masks):
                    raise ValueError(f"mask_idx {mask_idx} out of range for masks with length {len(masks)}")
                if param.shape[0] != masks[mask_idx].shape[0] or param.shape[1] != masks[mask_idx - 1].shape[0]:
                    raise ValueError(f"Dimension mismatch for {name}: weight shape {param.shape}, mask shape {masks[mask_idx].shape[0]}x{masks[mask_idx - 1].shape[0]}")
                pruned_param = param[masks[mask_idx] == 1][:, masks[mask_idx - 1] == 1]
                pruned_state_dict[name] = pruned_param
                pruned_channels[name] = pruned_param.shape[0]
                print(f"Pruned {name} to {pruned_param.shape}")
                block_mask_indices.append(mask_idx)
                mask_idx += 1
        # پرون کردن لایه‌های bn و downsample
        elif ('bn' in name or 'downsample' in name) and 'layer' in name:
            if not block_mask_indices:
                raise ValueError(f"No mask indices available for {name} in block {current_block}")
            # انتخاب ماسک مناسب بر اساس نوع لایه
            if 'bn1' in name:
                if len(block_mask_indices) < 1:
                    raise ValueError(f"Expected at least 1 mask index for {name}, got {block_mask_indices}")
                prev_mask_idx = block_mask_indices[0]  # ماسک conv1
            elif 'bn2' in name:
                if len(block_mask_indices) < 2:
                    raise ValueError(f"Expected at least 2 mask indices for {name}, got {block_mask_indices}")
                prev_mask_idx = block_mask_indices[1]  # ماسک conv2
            elif 'bn3' in name or 'downsample' in name:
                if len(block_mask_indices) < 3:
                    raise ValueError(f"Expected at least 3 mask indices for {name}, got {block_mask_indices}")
                prev_mask_idx = block_mask_indices[2]  # ماسک conv3
            else:
                raise ValueError(f"Unexpected layer name: {name}")
            # بررسی اینکه آیا param یک تنسور با ابعاد معتبر است
            if 'num_batches_tracked' in name:
                pruned_state_dict[name] = param
                pruned_channels[name] = 0
            else:
                if prev_mask_idx >= len(masks):
                    raise ValueError(f"prev_mask_idx {prev_mask_idx} out of range for masks with length {len(masks)}")
                if param.shape[0] != masks[prev_mask_idx].shape[0]:
                    raise ValueError(f"Dimension mismatch for {name}: weight shape {param.shape[0]}, mask shape {masks[prev_mask_idx].shape[0]}")
                pruned_param = param[masks[prev_mask_idx] == 1]
                pruned_state_dict[name] = pruned_param
                pruned_channels[name] = pruned_param.shape[0]
                print(f"Pruned {name} to {pruned_param.shape}")
        # کپی مستقیم سایر لایه‌ها (مثل fc)
        else:
            pruned_state_dict[name] = param
            pruned_channels[name] = param.shape[0] if len(param.shape) > 0 else 0
            print(f"Copied {name} with shape {param.shape}")

    if mask_idx != len(masks):
        print(f"Warning: mask_idx {mask_idx} does not match total number of masks {len(masks)}")
    return pruned_state_dict, pruned_channels

# تعریف مدل sparse برای استخراج ماسک‌ها
sparse_model = ResNet_50_sparse_hardfakevsreal()
sparse_model.dataset_type = 'rvf10k'

# بارگذاری چک‌پوینت
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

if 'student' not in checkpoint:
    raise KeyError("'student' key not found in checkpoint")

# لود وزن‌ها به مدل sparse
sparse_model.load_state_dict(checkpoint['student'], strict=True)
sparse_model.eval()

# بررسی وزن‌های feat1 (برای دیباگ)
print("Sample weights from feat1 in checkpoint:", checkpoint['student']['feat1.weight'][0, 0, :3, :3])
print("feat1 weight shape in checkpoint:", checkpoint['student']['feat1.weight'].shape)

# استخراج ماسک‌ها
mask_weights = [m.mask_weight for m in sparse_model.mask_modules]
masks = [torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1) for mask_weight in mask_weights]

# بررسی تعداد و ابعاد ماسک‌ها
print(f"Number of masks: {len(masks)}")
for i, mask in enumerate(masks):
    print(f"Mask {i} shape: {mask.shape}, kept channels: {(masks[i] == 1).sum().item()}")

# تعریف مدل پرون‌شده
pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
pruned_model.dataset_type = 'rvf10k'

# پرون کردن وزن‌ها
try:
    pruned_state_dict, pruned_channels = prune_weights(checkpoint['student'], masks)
except ValueError as e:
    print(f"Error in pruning weights: {e}")
    raise

# چاپ تعداد کانال‌های پرون‌شده برای دیباگ
print("Pruned channels for each layer:")
for name, num_channels in pruned_channels.items():
    print(f"{name}: {num_channels} channels")

# بررسی ابعاد مدل پرون‌شده
print("Expected shapes in pruned model:")
for name, param in pruned_model.state_dict().items():
    print(f"{name}: {param.shape}")

# بارگذاری وزن‌های پرون‌شده به مدل
missing, unexpected = pruned_model.load_state_dict(pruned_state_dict, strict=True)
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
    rvf10k_test_csv=os.path.join(data_dir, 'test.csv'),
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
feature_list_shapes = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs, feature_list = pruned_model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        feature_list_shapes.append([f.shape for f in feature_list])

print(f'Test Loss on rvf10k: {test_loss / len(test_loader):.4f}')
print(f'Test Accuracy on rvf10k: {100 * correct / total:.2f}%')
print(f"Feature list shapes: {feature_list_shapes[-1]}")
