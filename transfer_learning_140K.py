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
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k, SoftMaskedConv2d
from data.dataset import FaceDataset, Dataset_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer learning with pruned ResNet50 model.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k'],
                        help='Dataset to use: hardfake, rvf10k, or 140k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save trained model and outputs')
    parser.add_argument('--img_height', type=int, default=300,
                        help='Height of input images (default: 300 for hardfake, 256 for rvf10k/140k)')
    parser.add_argument('--img_width', type=int, default=300,
                        help='Width of input images (default: 300 for hardfake, 256 for rvf10k/140k)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer (fc layer)')
    parser.add_argument('--lr_layer4', type=float, default=1e-4,
                        help='Learning rate for layer4 parameters')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='Weight decay for the optimizer')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set environment variables
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Assign arguments to variables
dataset_mode = args.dataset_mode
data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_height
img_width = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
lr_layer4 = args.lr_layer4
weight_decay = args.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Validate directories
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
os.makedirs(teacher_dir, exist_ok=True)

# Initialize dataset
if dataset_mode == 'hardfake':
    dataset = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file=os.path.join(data_dir, 'data.csv'),
        hardfake_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == 'rvf10k':
    dataset = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv=os.path.join(data_dir, 'train.csv'),
        rvf10k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        rvf10k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == '140k':
    dataset = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv=os.path.join(data_dir, 'train.csv'),
        realfake140k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        realfake140k_test_csv=os.path.join(data_dir, 'test.csv'),
        realfake140k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
else:
    raise ValueError("Invalid dataset_mode. Choose 'hardfake', 'rvf10k', or '140k'.")

train_loader = dataset.loader_train
val_loader = dataset.loader_val
test_loader = dataset.loader_test

# Initialize student model
model = ResNet_50_sparse_rvf10k(
    gumbel_start_temperature=1.0,  # مقادیر پیش‌فرض، در صورت نیاز تنظیم کنید
    gumbel_end_temperature=0.1,
    num_epochs=epochs
)
model.dataset_type = 'rvf10k'
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Load checkpoint
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

if 'student' in checkpoint:
    state_dict = checkpoint['student']
    # فیلتر کردن کلیدهای اضافی
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('feat')}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
else:
    raise KeyError("'student' key not found in checkpoint")

model = model.to(device)
model.ticket = True  # فعال کردن حالت پرونینگ

# Calculate active parameters
total_params = 0
for name, module in model.named_modules():
    if isinstance(module, SoftMaskedConv2d):
        weight = module.weight
        mask_key = name + '.mask_weight'
        if mask_key in state_dict:
            mask = state_dict[mask_key].to(device)  # انتقال ماسک به دستگاه
            # بررسی ساختار ماسک
            print(f"{mask_key} shape: {mask.shape}, Sample values: {mask.flatten()[:10]}")
            # تبدیل ماسک به باینری
            if len(mask.shape) >= 2:  # فرض می‌کنیم ماسک‌ها دوبعدی هستند (مانند [64, 2, 1, 1])
                mask_binary = torch.argmax(mask, dim=1)  # انتخاب کانال‌های فعال
                mask_binary = (mask_binary != 0).float()  # تبدیل به 0/1
                mask_binary = mask_binary.view(-1, 1, 1, 1)  # تنظیم شکل برای ضرب
                # بررسی تطابق ابعاد
                if mask_binary.shape[0] != weight.shape[0]:
                    print(f"Dimension mismatch: mask_binary {mask_binary.shape[0]} vs weight {weight.shape[0]} for {name}")
                    active_params = weight.count_nonzero().item()  # استفاده از تمام وزن‌ها
                else:
                    active_params = (weight * mask_binary).abs().count_nonzero().item()
                    print(f"{name}.weight: {int(mask_binary.sum().item())}/{weight.shape[0]} channels kept")
            else:
                # اگر ماسک تک‌بعدی است
                print(f"Unexpected mask shape for {mask_key}: {mask.shape}")
                mask_binary = (mask > 0.5).float().view(-1, 1, 1, 1)
                active_params = (weight * mask_binary).abs().count_nonzero().item()
                print(f"{name}.weight: {int(mask_binary.sum().item())}/{weight.shape[0]} channels kept")
        else:
            print(f"{name}.weight: No mask available, using all {weight.shape[0]} channels")
            active_params = weight.count_nonzero().item()
        total_params += active_params
    elif isinstance(module, nn.Linear):
        total_params += module.weight.count_nonzero().item()
        if module.bias is not None:
            total_params += module.bias.count_nonzero().item()
print(f"Active Parameters: {total_params / 1e6:.2f} M")

# Calculate FLOPs
try:
    flops = model.get_flops()  # بدون input_shape
    print(f"Pruned FLOPs: {flops / 1e9:.2f} GMac")
except (AttributeError, TypeError) as e:
    print(f"Warning: get_flops method failed ({e}), using thop for FLOPs calculation")
    input_tensor = torch.randn(1, 3, img_height, img_width).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"THOP FLOPs: {flops / 1e9:.2f} GMac")
    print(f"THOP Parameters: {params / 1e6:.2f} M")

# Freeze all parameters except layer4 and fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Initialize optimizer
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': lr_layer4},
    {'params': model.fc.parameters(), 'lr': lr}
], weight_decay=weight_decay)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

if device.type == 'cuda':
    torch.cuda.empty_cache()

# Training loop
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
best_val_acc = 0.0
best_model_path = os.path.join(teacher_dir, 'teacher_model_best.pth')

for epoch in range(epochs):
    model.train()
    model.ticket = True
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs, _ = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Validation
    model.eval()
    model.ticket = True
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs, _ = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with validation accuracy: {val_accuracy:.2f}% at epoch {epoch+1}')

# Plot metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
metrics_plot_path = os.path.join(teacher_dir, 'training_metrics.png')
plt.savefig(metrics_plot_path)
display(IPImage(filename=metrics_plot_path))
plt.close()

# Save final model
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model_final.pth'))
print(f'Saved final model at epoch {epochs}')

# Test evaluation
model.eval()
model.ticket = True
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
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

# Visualize test samples
test_data = dataset.loader_test.dataset.data
transform_test = dataset.loader_test.dataset.transform
random_indices = random.sample(range(len(test_data)), min(20, len(test_data)))
fig, axes = plt.subplots(4, 5, figsize=(15, 8))
axes = axes.ravel()

label_map = {'fake': 0, 'real': 1, 0: 0, 1: 1, 'Fake': 0, 'Real': 1}
inverse_label_map = {0: 'fake', 1: 'real'}

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = test_data.iloc[idx]
        img_column = 'path' if dataset_mode == '140k' else 'images_id'
        img_name = row[img_column]
        label = row['label']
        img_path = os.path.join(data_dir, 'real_vs_fake', 'real-vs-fake', img_name) if dataset_mode == '140k' else os.path.join(data_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            axes[i].set_title("Image not found")
            axes[i].axis('off')
            continue
        
        image = Image.open(img_path).convert('RGB')
        image_transformed = transform_test(image).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            output, _ = model(image_transformed)
            output = output.squeeze(1)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        
        true_label = inverse_label_map[int(label)] if isinstance(label, (int, float)) else inverse_label_map[label_map[label]]
        
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"Image: {img_path}, True Label: {true_label}, Predicted: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))
