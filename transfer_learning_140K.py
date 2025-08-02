import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from thop import profile  # استفاده از thop به جای ptflops
from data.dataset import FaceDataset, Dataset_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model for fake vs real face classification.')
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

# Initialize model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Load checkpoint
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'student' in checkpoint:
    state_dict = checkpoint['student']
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('mask_weight') and k not in ['feat1.weight', 'feat1.bias', 'feat2.weight', 'feat2.bias', 'feat3.weight', 'feat3.bias', 'feat4.weight', 'feat4.bias']}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
else:
    raise KeyError("'student' key not found in checkpoint")

model = model.to(device)

# Calculate active parameters considering pruning masks
total_params = 0
for name, param in model.named_parameters():
    if '.weight' in name:  # فقط وزن‌ها را بررسی می‌کنیم، نه بایاس‌ها
        mask_key = name.replace('.weight', '.mask_weight')
        if mask_key in state_dict:
            mask = state_dict[mask_key].to(device)
            # تبدیل ماسک به باینری (0 یا 1) در صورت غیرباینری بودن
            mask = (mask != 0).float()
            # بررسی تطابق ابعاد
            if mask.shape[0] == param.shape[0]:  # اطمینان از تطابق تعداد کانال‌های خروجی
                # برای لایه‌های کانولوشنی، ماسک را به ابعاد وزن گسترش می‌دهیم
                if len(param.shape) == 4:  # [out_channels, in_channels, kernel_h, kernel_w]
                    mask = mask.view(-1, 1, 1, 1)  # [out_channels, 1, 1, 1]
                elif len(param.shape) == 2:  # برای لایه fc
                    mask = mask.view(-1, 1)  # [out_features, 1]
                try:
                    active_params = (param * mask).abs().count_nonzero().item()
                except RuntimeError as e:
                    print(f"Error in applying mask for {name}: {e}")
                    active_params = param.count_nonzero().item()
            else:
                print(f"Dimension mismatch for {name}: param.shape={param.shape}, mask.shape={mask.shape}")
                active_params = param.count_nonzero().item()
        else:
            active_params = param.count_nonzero().item()
    else:
        active_params = param.count_nonzero().item()
    total_params += active_params
print(f"Active Parameters: {total_params / 1e6:.2f} M")

# Custom FLOPs calculation considering pruning masks
def calculate_flops_pruned(model, state_dict, input_shape=(3, 256, 256)):
    flops = 0
    input_size = input_shape[1:]  # (height, width)

    def conv_flops(module, input_size, mask=None):
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            if mask is not None:
                out_channels = int(mask.sum().item())  # تعداد کانال‌های فعال
            k_h, k_w = module.kernel_size
            stride_h, stride_w = module.stride
            padding_h, padding_w = module.padding
            groups = module.groups

            # محاسبه اندازه خروجی
            out_h = (input_size[0] + 2 * padding_h - k_h) // stride_h + 1
            out_w = (input_size[1] + 2 * padding_w - k_w) // stride_w + 1

            # FLOPs برای یک لایه کانولوشنی
            flops_per_instance = k_h * k_w * in_channels * out_channels / groups
            total_flops = flops_per_instance * out_h * out_w
            return total_flops, (out_h, out_w)
        return 0, input_size

    for name, module in model.named_modules():
        mask_key = f"{name}.mask_weight"
        mask = state_dict.get(mask_key, None)
        if mask is not None:
            mask = (mask != 0).float()  # تبدیل به باینری
        if isinstance(module, nn.Conv2d):
            module_flops, input_size = conv_flops(module, input_size, mask)
            flops += module_flops
        elif isinstance(module, nn.Linear):
            # FLOPs برای لایه fc
            mask_key = 'fc.mask_weight'
            if mask_key in state_dict:
                mask = state_dict[mask_key].to(device)
                mask = (mask != 0).float()
                out_features = int(mask.sum().item())
            else:
                out_features = module.out_features
            flops += module.in_features * out_features
    return flops / 1e9  # تبدیل به گیگا فلاپس

flops = calculate_flops_pruned(model, state_dict, input_shape=(3, img_height, img_width))
print(f"Pruned FLOPs: {flops:.2f} GMac")

# Freeze all parameters except layer4 and fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Initialize optimizer with configurable weight decay and learning rates
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
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images).squeeze(1)
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
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(images).squeeze(1)
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
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images).squeeze(1)
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
            output = model(image_transformed).squeeze(1)
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
