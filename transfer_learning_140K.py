import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from thop import profile
import argparse
from torch.amp import autocast
import copy
from torchvision import transforms
from data.dataset import Dataset_selector
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal, get_preserved_filter_num

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune and test generalization of pruned ResNet50 student model.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the student model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing dataset folders')
    parser.add_argument('--datasets', type=str, nargs='+', default=['hardfake', 'rvf10k'],
                        help='List of datasets to fine-tune and test (e.g., hardfake rvf10k)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--lr_layer4', type=float, default=1e-4,
                        help='Learning rate for layer4 during fine-tuning')
    parser.add_argument('--lr_fc', type=float, default=0.0005,
                        help='Learning rate for fc layer during fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='Weight decay for optimizer')
    return parser.parse_args()

# تنظیمات اولیه
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# تعریف Data Augmentation برای train
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),  # برای سازگاری با hardfake
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. لود چک‌پوینت و استخراج ماسک‌ها
checkpoint_path = args.checkpoint_path
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

if 'student' in checkpoint:
    state_dict = checkpoint['student']
else:
    raise KeyError("'student' key not found in checkpoint")

# استخراج ماسک‌ها
masks = []
mask_dict = {}
for key, value in state_dict.items():
    if 'mask_weight' in key:
        mask_binary = (torch.argmax(value, dim=1).squeeze(1).squeeze(1) != 0).float()
        masks.append(mask_binary)
        mask_dict[key] = mask_binary.clone()
print(f"Number of masks extracted: {len(masks)}")
if len(masks) != 48:
    print(f"Warning: Expected 48 masks for ResNet50, got {len(masks)}")

# 2. ساخت مدل پرون‌شده
model = ResNet_50_pruned_hardfakevsreal(masks=masks)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# 3. فیلتر کردن و لود وزن‌ها
pruned_state_dict = {}
for key, value in state_dict.items():
    if 'mask_weight' in key or 'feat' in key:
        continue
    if 'weight' in key and 'conv' in key:
        mask_key = key.replace('.weight', '.mask_weight')
        if mask_key in state_dict:
            mask_binary = mask_dict[mask_key].float()
            if len(value.shape) == 4:
                out_channels = mask_binary.sum().int().item()
                pruned_weight = value[mask_binary.bool()].clone()
                if 'conv2' in key:
                    prev_mask_key = key.replace('conv2', 'conv1').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
                        if prev_mask_binary.shape[0] != value.shape[1]:
                            print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                            pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()].clone()
                        else:
                            pruned_weight = pruned_weight[:, prev_mask_binary.bool()].clone()
                elif 'conv3' in key:
                    prev_mask_key = key.replace('conv3', 'conv2').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
                        if prev_mask_binary.shape[0] != value.shape[1]:
                            print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                            pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()].clone()
                        else:
                            pruned_weight = pruned_weight[:, prev_mask_binary.bool()].clone()
                pruned_state_dict[key] = pruned_weight
            else:
                pruned_state_dict[key] = value.clone()
        else:
            pruned_state_dict[key] = value.clone()
    elif 'bn' in key and any(s in key for s in ['weight', 'bias', 'running_mean', 'running_var']):
        conv_key = key.replace('.bn1.', '.conv1.').replace('.bn2.', '.conv2.').replace('.bn3.', '.conv3.')
        conv_key = conv_key.replace('.weight', '.mask_weight').replace('.bias', '.mask_weight').replace('.running_mean', '.mask_weight').replace('.running_var', '.mask_weight')
        if conv_key in state_dict:
            mask_binary = mask_dict[conv_key].float()
            pruned_param = value[mask_binary.bool()].clone()
            pruned_state_dict[key] = pruned_param
        else:
            pruned_state_dict[key] = value.clone()
    else:
        pruned_state_dict[key] = value.clone()

missing, unexpected = model.load_state_dict(pruned_state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# 4. محاسبه تعداد پارامترها و FLOPs
input = torch.randn(1, 3, 300, 300).to(device)  # برای hardfake
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops / 1e9:.2f} GMac, Parameters: {params / 1e6:.2f} M")

# 5. تعریف تنظیمات دیتاست‌ها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': os.path.join(args.data_dir, 'hardfakevsrealfaces/data.csv'),
        'hardfake_root_dir': os.path.join(args.data_dir, 'hardfakevsrealfaces'),
        'train_transform': train_transform,
        'eval_transform': eval_transform
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': os.path.join(args.data_dir, 'rvf10k/train.csv'),
        'rvf10k_valid_csv': os.path.join(args.data_dir, 'rvf10k/valid.csv'),
        'rvf10k_root_dir': os.path.join(args.data_dir, 'rvf10k'),
        'train_transform': train_transform,
        'eval_transform': eval_transform
    }
}

# 6. بررسی دیتاست‌های انتخاب‌شده و وجود فایل‌ها
selected_datasets = args.datasets
valid_datasets = []
for ds in selected_datasets:
    if ds not in dataset_configs:
        print(f"Warning: Dataset '{ds}' is invalid and will be ignored. Valid datasets: {list(dataset_configs.keys())}")
        continue
    config = dataset_configs[ds]
    all_files_exist = True
    if ds == 'hardfake':
        if not os.path.exists(config.get('hardfake_csv_file', '')):
            print(f"Error: CSV file not found: {config.get('hardfake_csv_file')}")
            all_files_exist = False
        if not os.path.exists(config.get('hardfake_root_dir', '')):
            print(f"Error: Directory not found: {config.get('hardfake_root_dir')}")
            all_files_exist = False
    elif ds == 'rvf10k':
        if not os.path.exists(config.get('rvf10k_train_csv', '')):
            print(f"Error: CSV file not found: {config.get('rvf10k_train_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('rvf10k_valid_csv', '')):
            print(f"Error: CSV file not found: {config.get('rvf10k_valid_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('rvf10k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('rvf10k_root_dir')}")
            all_files_exist = False
    if all_files_exist:
        valid_datasets.append(ds)
    else:
        print(f"Dataset '{ds}' will be skipped due to missing files or directories.")

if not valid_datasets:
    raise ValueError(f"No valid datasets selected or available! Choose from {list(dataset_configs.keys())}")

# 7. تابع ارزیابی
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs, _ = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / len(data_loader) if total > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

# 8. فاین‌تیونینگ و تست
results = {}
criterion = nn.BCEWithLogitsLoss()

for dataset_name in valid_datasets:
    config = dataset_configs[dataset_name]
    print(f"\nProcessing {dataset_name} dataset...")

    # لود دیتاست
    try:
        dataset = Dataset_selector(
            dataset_mode=config['dataset_mode'],
            hardfake_csv_file=config.get('hardfake_csv_file'),
            hardfake_root_dir=config.get('hardfake_root_dir'),
            rvf10k_train_csv=config.get('rvf10k_train_csv'),
            rvf10k_valid_csv=config.get('rvf10k_valid_csv'),
            rvf10k_root_dir=config.get('rvf10k_root_dir'),
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ddp=False,
            train_transform=config.get('train_transform'),
            eval_transform=config.get('eval_transform')
        )
        train_loader = dataset.loader_train
        val_loader = dataset.loader_val
        test_loader = dataset.loader_test
        print(f"{dataset_name} train dataset size: {len(train_loader.dataset)}")
        print(f"{dataset_name} validation dataset size: {len(val_loader.dataset)}")
        print(f"{dataset_name} test dataset size: {len(test_loader.dataset)}")
        print(f"{dataset_name} test loader batches: {len(test_loader)}")
        try:
            sample = next(iter(test_loader))
            print(f"Sample test batch image shape: {sample[0].shape}")
            print(f"Sample test batch labels: {sample[1][:5]}")
        except Exception as e:
            print(f"Error loading sample test batch for {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
            continue
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        results[dataset_name] = {'error': str(e)}
        continue

    # ارزیابی قبل از فاین‌تیونینگ
    try:
        test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Before Fine-tuning - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        results[dataset_name] = {'before_finetune_loss': test_loss, 'before_finetune_accuracy': test_accuracy}
    except Exception as e:
        print(f"Error evaluating {dataset_name} before fine-tuning: {e}")
        results[dataset_name] = {'error': str(e)}
        continue

    # فاین‌تیونینگ
    try:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {'params': model.layer4.parameters(), 'lr': args.lr_layer4},
            {'params': model.fc.parameters(), 'lr': args.lr_fc}
        ], weight_decay=args.weight_decay)

        model_state = copy.deepcopy(model.state_dict())

        model.train()
        for epoch in range(args.epochs):
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    outputs, _ = model(images)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print(f"{dataset_name} - Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

        val_loss, val_accuracy = evaluate_model(model, val_loader, device, criterion)
        print(f"{dataset_name} - After Fine-tuning - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        results[dataset_name]['val_loss'] = val_loss
        results[dataset_name]['val_accuracy'] = val_accuracy

        test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - After Fine-tuning - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        results[dataset_name]['test_loss'] = test_loss
        results[dataset_name]['test_accuracy'] = test_accuracy

        torch.save(model.state_dict(), os.path.join('results', f'finetuned_model_{dataset_name}.pt'))
        print(f"Fine-tuned model for {dataset_name} saved to results/finetuned_model_{dataset_name}.pt")
    except Exception as e:
        print(f"Error during fine-tuning {dataset_name}: {e}")
        results[dataset_name]['error'] = str(e)

    model.load_state_dict(model_state)

# 9. ذخیره نتایج
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'generalization_results.txt'), 'w') as f:
    f.write(f"FLOPs: {flops / 1e9:.2f} GMac\n")
    f.write(f"Parameters: {params / 1e6:.2f} M\n\n")
    for dataset_name in valid_datasets:
        result = results.get(dataset_name, {'error': 'Not evaluated'})
        f.write(f"Dataset: {dataset_name}\n")
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
        else:
            f.write(f"Before Fine-tuning - Test Loss: {result['before_finetune_loss']:.4f}\n")
            f.write(f"Before Fine-tuning - Test Accuracy: {result['before_finetune_accuracy']:.2f}%\n")
            f.write(f"After Fine-tuning - Validation Loss: {result['val_loss']:.4f}\n")
            f.write(f"After Fine-tuning - Validation Accuracy: {result['val_accuracy']:.2f}%\n")
            f.write(f"After Fine-tuning - Test Loss: {result['test_loss']:.4f}\n")
            f.write(f"After Fine-tuning - Test Accuracy: {result['test_accuracy']:.2f}%\n")
        f.write("\n")
print(f"Results saved to {os.path.join(results_dir, 'generalization_results.txt')}")

# 10. چاپ نتایج نهایی
print("\nFinal Generalization Results:")
for dataset_name in valid_datasets:
    print(f"\nDataset: {dataset_name}")
    result = results.get(dataset_name, {'error': 'Not evaluated'})
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Before Fine-tuning - Test Loss: {result['before_finetune_loss']:.4f}, Test Accuracy: {result['before_finetune_accuracy']:.2f}%")
        print(f"After Fine-tuning - Validation Loss: {result['val_loss']:.4f}, Validation Accuracy: {result['val_accuracy']:.2f}%")
        print(f"After Fine-tuning - Test Loss: {result['test_loss']:.4f}, Test Accuracy: {result['test_accuracy']:.2f}%")
