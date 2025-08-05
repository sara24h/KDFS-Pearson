import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from thop import profile
import argparse
from torch.amp import autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from data.dataset import Dataset_selector
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal, get_preserved_filter_num

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune and test generalization of pruned ResNet50 student model.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the student model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing dataset folders')
    parser.add_argument('--datasets', type=str, nargs='+', default=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help='List of datasets to fine-tune')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Dataset to use for testing (if not specified, test on all datasets)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    return parser.parse_args()

# تنظیمات اولیه
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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
        mask_dict[key] = mask_binary
print(f"Number of masks extracted: {len(masks)}")
if len(masks) != 48:
    print(f"Warning: Expected 48 masks for ResNet50, got {len(masks)}")

# ذخیره ماسک‌ها
masks_path = os.path.join('checkpoints', 'pruned_masks.pth')
os.makedirs('checkpoints', exist_ok=True)
torch.save(masks, masks_path)
print(f"Masks saved to {masks_path}")

# 2. ساخت و ذخیره مدل هرس‌شده
model = ResNet_50_pruned_hardfakevsreal(masks=masks)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# لود وزن‌های هرس‌شده
pruned_state_dict = {}
for key, value in state_dict.items():
    if 'mask_weight' in key or 'feat' in key:
        continue
    if 'weight' in key and 'conv' in key:
        mask_key = key.replace('.weight', '.mask_weight')
        if mask_key in state_dict:
            mask_binary = mask_dict[mask_key].float()
            out_channels = mask_binary.sum().int().item()
            pruned_weight = value[mask_binary.bool()]
            if 'conv2' in key:
                prev_mask_key = key.replace('conv2', 'conv1').replace('.weight', '.mask_weight')
                if prev_mask_key in state_dict:
                    prev_mask_binary = mask_dict[prev_mask_key].float()
                    if prev_mask_binary.shape[0] != value.shape[1]:
                        print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                        pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()]
                    else:
                        pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
            elif 'conv3' in key:
                prev_mask_key = key.replace('conv3', 'conv2').replace('.weight', '.mask_weight')
                if prev_mask_key in state_dict:
                    prev_mask_binary = mask_dict[prev_mask_key].float()
                    if prev_mask_binary.shape[0] != value.shape[1]:
                        print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                        pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()]
                    else:
                        pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
            pruned_state_dict[key] = pruned_weight
        else:
            pruned_state_dict[key] = value
    elif 'bn' in key and any(s in key for s in ['weight', 'bias', 'running_mean', 'running_var']):
        conv_key = key.replace('.bn1.', '.conv1.').replace('.bn2.', '.conv2.').replace('.bn3.', '.conv3.')
        conv_key = conv_key.replace('.weight', '.mask_weight').replace('.bias', '.mask_weight').replace('.running_mean', '.mask_weight').replace('.running_var', '.mask_weight')
        if conv_key in state_dict:
            mask_binary = mask_dict[conv_key].float()
            pruned_param = value[mask_binary.bool()]
            pruned_state_dict[key] = pruned_param
        else:
            pruned_state_dict[key] = value
    else:
        pruned_state_dict[key] = value

missing, unexpected = model.load_state_dict(pruned_state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# ذخیره مدل هرس‌شده
pruned_checkpoint_path = os.path.join('checkpoints', 'pruned_model.pth')
torch.save(model.state_dict(), pruned_checkpoint_path)
print(f"Pruned model saved to {pruned_checkpoint_path}")

# تست لود چک‌پوینت برای اطمینان
test_model = ResNet_50_pruned_hardfakevsreal(masks=torch.load(masks_path, map_location='cpu'))
test_model.load_state_dict(torch.load(pruned_checkpoint_path, map_location='cpu', weights_only=True))
print("Pruned model checkpoint loaded successfully for testing!")

# 3. تعریف تابع فاین‌تیونینگ
def fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, dataset_name):
    best_acc = 0.0
    best_model_path = os.path.join('checkpoints', f'finetuned_{dataset_name}.pth')
    os.makedirs('checkpoints', exist_ok=True)
    
    # فریز کردن تمام لایه‌ها به‌جز لایه fc
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
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
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader) if total > 0 else 0
        train_accuracy = 100 * correct / total if total > 0 else 0
        
        # ارزیابی روی دیتاست اعتبارسنجی
        valid_loss, valid_accuracy, _, _, _, _ = evaluate_model(model, valid_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} - {dataset_name} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")
        
        # ذخیره بهترین مدل
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model for {dataset_name} with Valid Acc: {best_acc:.2f}%")
    
    return best_model_path

# 4. تابع ارزیابی با معیارهای اضافی
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
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
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = test_loss / len(data_loader) if total > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    
    # ترکیب پیش‌بینی‌ها و برچسب‌ها
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # محاسبه Precision، Recall و F1-Score
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # محاسبه Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, cm

# 5. تعریف تنظیمات دیتاست‌ها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': os.path.join(args.data_dir, 'hardfakevsrealfaces/data.csv'),
        'hardfake_root_dir': os.path.join(args.data_dir, 'hardfakevsrealfaces'),
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': os.path.join(args.data_dir, 'rvf10k/train.csv'),
        'rvf10k_valid_csv': os.path.join(args.data_dir, 'rvf10k/valid.csv'),
        'rvf10k_root_dir': os.path.join(args.data_dir, 'rvf10k'),
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_train_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/train.csv'),
        'realfake140k_valid_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/valid.csv'),
        'realfake140k_test_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/test.csv'),
        'realfake140k_root_dir': os.path.join(args.data_dir, '140k-real-and-fake-faces'),
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': os.path.join(args.data_dir, 'deepfake-and-real-images/Dataset'),
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_train_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/train_labels.csv'),
        'realfake200k_val_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/val_labels.csv'),
        'realfake200k_test_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/test_labels.csv'),
        'realfake200k_root_dir': os.path.join(args.data_dir, '200k-real-and-fake-faces'),
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': os.path.join(args.data_dir, 'deepfake-dataset'),
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
        if not os.path.exists(config.get('rvf10k_train_csv', '')) or not os.path.exists(config.get('rvf10k_valid_csv', '')):
            print(f"Error: CSV file not found: {config.get('rvf10k_train_csv')} or {config.get('rvf10k_valid_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('rvf10k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('rvf10k_root_dir')}")
            all_files_exist = False
    elif ds == '140k':
        if not os.path.exists(config.get('realfake140k_train_csv', '')) or not os.path.exists(config.get('realfake140k_valid_csv', '')) or not os.path.exists(config.get('realfake140k_test_csv', '')):
            print(f"Error: CSV file not found: {config.get('realfake140k_train_csv')} or {config.get('realfake140k_valid_csv')} or {config.get('realfake140k_test_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('realfake140k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake140k_root_dir')}")
            all_files_exist = False
    elif ds == '190k':
        if not os.path.exists(config.get('realfake190k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake190k_root_dir')}")
            all_files_exist = False
    elif ds == '200k':
        if not os.path.exists(config.get('realfake200k_train_csv', '')) or not os.path.exists(config.get('realfake200k_val_csv', '')) or not os.path.exists(config.get('realfake200k_test_csv', '')):
            print(f"Error: CSV file not found: {config.get('realfake200k_train_csv')} or {config.get('realfake200k_val_csv')} or {config.get('realfake200k_test_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('realfake200k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake200k_root_dir')}")
            all_files_exist = False
    elif ds == '330k':
        if not os.path.exists(config.get('realfake330k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake330k_root_dir')}")
            all_files_exist = False
    if all_files_exist:
        valid_datasets.append(ds)
    else:
        print(f"Dataset '{ds}' will be skipped due to missing files or directories.")

# بررسی دیتاست تست
test_dataset = args.test_dataset
if test_dataset and test_dataset not in dataset_configs:
    print(f"Warning: Test dataset '{test_dataset}' is invalid. Valid datasets: {list(dataset_configs.keys())}")
    test_dataset = None
if test_dataset:
    config = dataset_configs[test_dataset]
    all_files_exist = True
    if test_dataset == 'hardfake':
        if not os.path.exists(config.get('hardfake_csv_file', '')):
            print(f"Error: CSV file not found: {config.get('hardfake_csv_file')}")
            all_files_exist = False
        if not os.path.exists(config.get('hardfake_root_dir', '')):
            print(f"Error: Directory not found: {config.get('hardfake_root_dir')}")
            all_files_exist = False
    elif test_dataset == 'rvf10k':
        if not os.path.exists(config.get('rvf10k_valid_csv', '')):
            print(f"Error: CSV file not found: {config.get('rvf10k_valid_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('rvf10k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('rvf10k_root_dir')}")
            all_files_exist = False
    elif test_dataset == '140k':
        if not os.path.exists(config.get('realfake140k_test_csv', '')):
            print(f"Error: CSV file not found: {config.get('realfake140k_test_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('realfake140k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake140k_root_dir')}")
            all_files_exist = False
    elif test_dataset == '190k':
        if not os.path.exists(config.get('realfake190k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake190k_root_dir')}")
            all_files_exist = False
    elif test_dataset == '200k':
        if not os.path.exists(config.get('realfake200k_test_csv', '')):
            print(f"Error: CSV file not found: {config.get('realfake200k_test_csv')}")
            all_files_exist = False
        if not os.path.exists(config.get('realfake200k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake200k_root_dir')}")
            all_files_exist = False
    elif test_dataset == '330k':
        if not os.path.exists(config.get('realfake330k_root_dir', '')):
            print(f"Error: Directory not found: {config.get('realfake330k_root_dir')}")
            all_files_exist = False
    if not all_files_exist:
        print(f"Test dataset '{test_dataset}' will be ignored due to missing files or directories.")
        test_dataset = None

if not valid_datasets and not test_dataset:
    raise ValueError(f"No valid datasets or test dataset selected! Choose from {list(dataset_configs.keys())}")

# 7. حلقه فاین‌تیونینگ
results = {}
criterion = nn.BCEWithLogitsLoss()

for dataset_name in valid_datasets:
    print(f"\nFine-tuning on {dataset_name} dataset...")
    
    # لود دیتاست
    config = dataset_configs[dataset_name]
    try:
        dataset = Dataset_selector(
            dataset_mode=config['dataset_mode'],
            hardfake_csv_file=config.get('hardfake_csv_file'),
            hardfake_root_dir=config.get('hardfake_root_dir'),
            rvf10k_train_csv=config.get('rvf10k_train_csv'),
            rvf10k_valid_csv=config.get('rvf10k_valid_csv'),
            rvf10k_root_dir=config.get('rvf10k_root_dir'),
            realfake140k_train_csv=config.get('realfake140k_train_csv'),
            realfake140k_valid_csv=config.get('realfake140k_valid_csv'),
            realfake140k_test_csv=config.get('realfake140k_test_csv'),
            realfake140k_root_dir=config.get('realfake140k_root_dir'),
            realfake200k_train_csv=config.get('realfake200k_train_csv'),
            realfake200k_val_csv=config.get('realfake200k_val_csv'),
            realfake200k_test_csv=config.get('realfake200k_test_csv'),
            realfake200k_root_dir=config.get('realfake200k_root_dir'),
            realfake190k_root_dir=config.get('realfake190k_root_dir'),
            realfake330k_root_dir=config.get('realfake330k_root_dir'),
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ddp=False
        )
        train_loader = dataset.loader_train if hasattr(dataset, 'loader_train') else None
        valid_loader = dataset.loader_valid if hasattr(dataset, 'loader_valid') else dataset.loader_test
        print(f"{dataset_name} train dataset size: {len(train_loader.dataset) if train_loader else 0}")
        print(f"{dataset_name} valid dataset size: {len(valid_loader.dataset)}")
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # بازسازی مدل با ماسک‌های ذخیره‌شده و لود چک‌پوینت هرس‌شده
    model = ResNet_50_pruned_hardfakevsreal(masks=torch.load(masks_path, map_location='cpu'))
    model.load_state_dict(torch.load(pruned_checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    
    # فاین‌تیونینگ فقط لایه fc
    if train_loader:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        print(f"Fine-tuning only fc layer on {dataset_name}...")
        best_model_path = fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, dataset_name)
    else:
        print(f"No training data available for {dataset_name}. Skipping fine-tuning.")
        best_model_path = pruned_checkpoint_path  # استفاده از مدل هرس‌شده برای تست

# 8. حلقه تست روی دیتاست انتخاب‌شده
test_datasets = [test_dataset] if test_dataset else valid_datasets
for dataset_name in test_datasets:
    print(f"\nTesting on {dataset_name} dataset...")
    
    # لود دیتاست تست
    config = dataset_configs[dataset_name]
    try:
        dataset = Dataset_selector(
            dataset_mode=config['dataset_mode'],
            hardfake_csv_file=config.get('hardfake_csv_file'),
            hardfake_root_dir=config.get('hardfake_root_dir'),
            rvf10k_train_csv=config.get('rvf10k_train_csv'),
            rvf10k_valid_csv=config.get('rvf10k_valid_csv'),
            rvf10k_root_dir=config.get('rvf10k_root_dir'),
            realfake140k_train_csv=config.get('realfake140k_train_csv'),
            realfake140k_valid_csv=config.get('realfake140k_valid_csv'),
            realfake140k_test_csv=config.get('realfake140k_test_csv'),
            realfake140k_root_dir=config.get('realfake140k_root_dir'),
            realfake200k_train_csv=config.get('realfake200k_train_csv'),
            realfake200k_val_csv=config.get('realfake200k_val_csv'),
            realfake200k_test_csv=config.get('realfake200k_test_csv'),
            realfake200k_root_dir=config.get('realfake200k_root_dir'),
            realfake190k_root_dir=config.get('realfake190k_root_dir'),
            realfake330k_root_dir=config.get('realfake330k_root_dir'),
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ddp=False
        )
        test_loader = dataset.loader_test
        print(f"{dataset_name} test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading {dataset_name} test dataset: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # تست مدل هرس‌شده (قبل از فاین‌تیونینگ)
    model = ResNet_50_pruned_hardfakevsreal(masks=torch.load(masks_path, map_location='cpu'))
    model.load_state_dict(torch.load(pruned_checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    
    try:
        pre_finetune_loss, pre_finetune_accuracy, pre_precision, pre_recall, pre_f1, pre_cm = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Pre-Finetune Metrics:")
        print(f"  Test Loss: {pre_finetune_loss:.4f}")
        print(f"  Test Accuracy: {pre_finetune_accuracy:.2f}%")
        print(f"  Precision: {pre_precision:.4f}")
        print(f"  Recall: {pre_recall:.4f}")
        print(f"  F1-Score: {pre_f1:.4f}")
        print(f"  Confusion Matrix:\n{pre_cm}")
        results[dataset_name] = {
            'pre_finetune_loss': pre_finetune_loss,
            'pre_finetune_accuracy': pre_finetune_accuracy,
            'pre_precision': pre_precision,
            'pre_recall': pre_recall,
            'pre_f1': pre_f1,
            'pre_cm': pre_cm.tolist()
        }
    except Exception as e:
        print(f"Error evaluating {dataset_name} before fine-tuning: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # تست مدل فاین‌تیون‌شده (برای هر دیتاست فاین‌تیون‌شده)
    for finetune_dataset in valid_datasets:
        best_model_path = os.path.join('checkpoints', f'finetuned_{finetune_dataset}.pth')
        if not os.path.exists(best_model_path):
            print(f"No fine-tuned model found for {finetune_dataset}. Skipping.")
            continue
        
        model = ResNet_50_pruned_hardfakevsreal(masks=torch.load(masks_path, map_location='cpu'))
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        model.to(device)
        
        try:
            post_finetune_loss, post_finetune_accuracy, post_precision, post_recall, post_f1, post_cm = evaluate_model(model, test_loader, device, criterion)
            print(f"{dataset_name} - Post-Finetune Metrics (fine-tuned on {finetune_dataset}):")
            print(f"  Test Loss: {post_finetune_loss:.4f}")
            print(f"  Test Accuracy: {post_finetune_accuracy:.2f}%")
            print(f"  Precision: {post_precision:.4f}")
            print(f"  Recall: {post_recall:.4f}")
            print(f"  F1-Score: {post_f1:.4f}")
            print(f"  Confusion Matrix:\n{post_cm}")
            results[dataset_name][f'post_finetune_loss_{finetune_dataset}'] = post_finetune_loss
            results[dataset_name][f'post_finetune_accuracy_{finetune_dataset}'] = post_finetune_accuracy
            results[dataset_name][f'post_precision_{finetune_dataset}'] = post_precision
            results[dataset_name][f'post_recall_{finetune_dataset}'] = post_recall
            results[dataset_name][f'post_f1_{finetune_dataset}'] = post_f1
            results[dataset_name][f'post_cm_{finetune_dataset}'] = post_cm.tolist()
        except Exception as e:
            print(f"Error evaluating {dataset_name} after fine-tuning on {finetune_dataset}: {e}")
            results[dataset_name][f'error_{finetune_dataset}'] = str(e)
    
    # محاسبه FLOPs و پارامترها
    input = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    results[dataset_name]['flops'] = flops / 1e9  # GMac
    results[dataset_name]['params'] = params / 1e6  # M

# 9. ذخیره نتایج
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'finetune_results.txt'), 'w') as f:
    for dataset_name in test_datasets:
        f.write(f"Test Dataset: {dataset_name}\n")
        result = results.get(dataset_name, {'error': 'Not evaluated'})
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
        else:
            f.write(f"Pre-Finetune Metrics:\n")
            f.write(f"  Test Loss: {result['pre_finetune_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {result['pre_finetune_accuracy']:.2f}%\n")
            f.write(f"  Precision: {result['pre_precision']:.4f}\n")
            f.write(f"  Recall: {result['pre_recall']:.4f}\n")
            f.write(f"  F1-Score: {result['pre_f1']:.4f}\n")
            f.write(f"  Confusion Matrix: {result['pre_cm']}\n")
            for finetune_dataset in valid_datasets:
                if f'post_finetune_loss_{finetune_dataset}' in result:
                    f.write(f"Post-Finetune Metrics (fine-tuned on {finetune_dataset}):\n")
                    f.write(f"  Test Loss: {result[f'post_finetune_loss_{finetune_dataset}']:.4f}\n")
                    f.write(f"  Test Accuracy: {result[f'post_finetune_accuracy_{finetune_dataset}']:.2f}%\n")
                    f.write(f"  Precision: {result[f'post_precision_{finetune_dataset}']:.4f}\n")
                    f.write(f"  Recall: {result[f'post_recall_{finetune_dataset}']:.4f}\n")
                    f.write(f"  F1-Score: {result[f'post_f1_{finetune_dataset}']:.4f}\n")
                    f.write(f"  Confusion Matrix: {result[f'post_cm_{finetune_dataset}']}\n")
                if f'error_{finetune_dataset}' in result:
                    f.write(f"Error (fine-tuned on {finetune_dataset}): {result[f'error_{finetune_dataset}']}\n")
            f.write(f"FLOPs: {result['flops']:.2f} GMac\n")
            f.write(f"Parameters: {result['params']:.2f} M\n")
        f.write("\n")
print(f"Results saved to {os.path.join(results_dir, 'finetune_results.txt')}")

# 10. چاپ نتایج نهایی
print("\nFinal Fine-tuning Results:")
for dataset_name in test_datasets:
    print(f"\nTest Dataset: {dataset_name}")
    result = results.get(dataset_name, {'error': 'Not evaluated'})
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Pre-Finetune Metrics:")
        print(f"  Test Loss: {result['pre_finetune_loss']:.4f}")
        print(f"  Test Accuracy: {result['pre_finetune_accuracy']:.2f}%")
        print(f"  Precision: {result['pre_precision']:.4f}")
        print(f"  Recall: {result['pre_recall']:.4f}")
        print(f"  F1-Score: {result['pre_f1']:.4f}")
        print(f"  Confusion Matrix:\n{result['pre_cm']}")
        for finetune_dataset in valid_datasets:
            if f'post_finetune_loss_{finetune_dataset}' in result:
                print(f"Post-Finetune Metrics (fine-tuned on {finetune_dataset}):")
                print(f"  Test Loss: {result[f'post_finetune_loss_{finetune_dataset}']:.4f}")
                print(f"  Test Accuracy: {result[f'post_finetune_accuracy_{finetune_dataset}']:.2f}%")
                print(f"  Precision: {result[f'post_precision_{finetune_dataset}']:.4f}")
                print(f"  Recall: {result[f'post_recall_{finetune_dataset}']:.4f}")
                print(f"  F1-Score: {result[f'post_f1_{finetune_dataset}']:.4f}")
                print(f"  Confusion Matrix:\n{result[f'post_cm_{finetune_dataset}']}")
            if f'error_{finetune_dataset}' in result:
                print(f"Error (fine-tuned on {finetune_dataset}): {result[f'error_{finetune_dataset}']}")
        print(f"FLOPs: {result['flops']:.2f} GMac, Parameters: {result['params']:.2f} M")
