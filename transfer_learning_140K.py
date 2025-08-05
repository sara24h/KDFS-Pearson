import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from thop import profile
import argparse
from torch.amp import autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
from data.dataset import Dataset_selector
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def parse_args():
    parser = argparse.ArgumentParser(description='Test generalization of pruned ResNet50 model.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the student model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing dataset folders')
    parser.add_argument('--datasets', type=str, nargs='+', default=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help='List of datasets to test')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    return parser.parse_args()

# تنظیمات اولیه
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# بررسی فضای دیسک
def check_disk_space(path='/kaggle/working'):
    disk = psutil.disk_usage(path)
    free_space = disk.free / (1024 ** 3)  # به گیگابایت
    print(f"Disk space check for {path}: {free_space:.2f} GB free")
    if free_space < 1:
        raise RuntimeError(f"Insufficient disk space at {path}: {free_space:.2f} GB free")

# تابع برای چاپ محتوای دایرکتوری
def list_directory(path):
    print(f"Listing contents of {path}:")
    try:
        os.system(f"ls -lh {path}")
    except Exception as e:
        print(f"Error listing directory {path}: {e}")

# تابع برای رسم و ذخیره ماتریس درهم‌ریختگی
def plot_confusion_matrix(cm, dataset_name, phase, results_dir):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {dataset_name} ({phase})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(results_dir, f'cm_{dataset_name}_{phase}.png')
    check_disk_space(results_dir)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    list_directory(results_dir)

# تابع برای لود و آماده‌سازی مدل هرس‌شده
def load_pruned_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'student' not in checkpoint:
        raise KeyError("'student' key not found in checkpoint")
    state_dict = checkpoint['student']

    masks = []
    mask_dict = {}
    for key, value in state_dict.items():
        if 'mask_weight' in key:
            mask_binary = (torch.argmax(value, dim=1).squeeze(1).squeeze(1) != 0).float()
            masks.append(mask_binary)
            mask_dict[key] = mask_binary
    if len(masks) != 48:
        raise ValueError(f"Expected 48 masks for ResNet50, got {len(masks)}")

    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    pruned_state_dict = {}
    for key, value in state_dict.items():
        if 'mask_weight' in key or 'feat' in key:
            continue
        if 'weight' in key and 'conv' in key:
            mask_key = key.replace('.weight', '.mask_weight')
            if mask_key in state_dict:
                mask_binary = mask_dict[mask_key].float()
                pruned_weight = value[mask_binary.bool()]
                if 'conv2' in key:
                    prev_mask_key = key.replace('conv2', 'conv1').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
                        pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
                elif 'conv3' in key:
                    prev_mask_key = key.replace('conv3', 'conv2').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
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
    if missing or unexpected:
        raise ValueError(f"State dict loading issues - Missing: {missing}, Unexpected: {unexpected}")

    checkpoints_dir = '/kaggle/working/checkpoints'
    backup_dir = '/kaggle/working/backup'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    pruned_model_path = os.path.join(checkpoints_dir, 'pruned_model.pth')
    backup_model_path = os.path.join(backup_dir, 'pruned_model_backup.pth')
    
    torch.cuda.empty_cache()
    check_disk_space(checkpoints_dir)
    try:
        torch.save({'model_state_dict': model.state_dict(), 'masks': masks}, pruned_model_path)
        if not os.path.exists(pruned_model_path):
            raise FileNotFoundError(f"Failed to confirm existence of {pruned_model_path}")
        print(f"Pruned model successfully saved to {pruned_model_path}")
        # ذخیره نسخه پشتیبان
        torch.save({'model_state_dict': model.state_dict(), 'masks': masks}, backup_model_path)
        if not os.path.exists(backup_model_path):
            raise FileNotFoundError(f"Failed to confirm existence of {backup_model_path}")
        print(f"Backup pruned model saved to {backup_model_path}")
        list_directory(checkpoints_dir)
        list_directory(backup_dir)
    except Exception as e:
        raise RuntimeError(f"Error saving pruned model to {pruned_model_path}: {e}")

    return model, masks, pruned_model_path

# تابع فاین‌تیونینگ فقط روی لایه fc
def fine_tune_model(model, test_loader, device, criterion, optimizer, epochs, dataset_name, results_dir):
    checkpoints_dir = '/kaggle/working/checkpoints'
    backup_dir = '/kaggle/working/backup'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    finetuned_model_path = os.path.join(checkpoints_dir, f'finetuned_{dataset_name}.pth')
    finetuned_backup_path = os.path.join(backup_dir, f'finetuned_{dataset_name}_backup.pth')
    
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in test_loader:
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
        
        avg_train_loss = train_loss / len(test_loader) if total > 0 else 0
        train_accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - {dataset_name} - Fine-tune Loss: {avg_train_loss:.4f}, Fine-tune Acc: {train_accuracy:.2f}%")
    
    check_disk_space(checkpoints_dir)
    try:
        torch.save(model.state_dict(), finetuned_model_path)
        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"Failed to confirm existence of {finetuned_model_path}")
        print(f"Fine-tuned model saved to {finetuned_model_path}")
        # ذخیره نسخه پشتیبان
        torch.save(model.state_dict(), finetuned_backup_path)
        if not os.path.exists(finetuned_backup_path):
            raise FileNotFoundError(f"Failed to confirm existence of {finetuned_backup_path}")
        print(f"Backup fine-tuned model saved to {finetuned_backup_path}")
        list_directory(checkpoints_dir)
        list_directory(backup_dir)
    except Exception as e:
        raise RuntimeError(f"Error saving fine-tuned model to {finetuned_model_path}: {e}")
    
    return finetuned_model_path

# تابع ارزیابی
def evaluate_model(model, data_loader, device, criterion, dataset_name, phase, results_dir):
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
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    if phase == 'post_finetune':
        plot_confusion_matrix(cm, dataset_name, phase, results_dir)
    
    return avg_loss, accuracy, precision, recall, f1, cm

# تنظیمات دیتاست‌ها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': os.path.join(args.data_dir, 'hardfakevsrealfaces/data.csv'),
        'hardfake_root_dir': os.path.join(args.data_dir, 'hardfakevsrealfaces'),
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_valid_csv': os.path.join(args.data_dir, 'rvf10k/valid.csv'),
        'rvf10k_root_dir': os.path.join(args.data_dir, 'rvf10k'),
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_test_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/test.csv'),
        'realfake140k_root_dir': os.path.join(args.data_dir, '140k-real-and-fake-faces'),
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': os.path.join(args.data_dir, 'deepfake-and-real-images/Dataset'),
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_test_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/test_labels.csv'),
        'realfake200k_root_dir': os.path.join(args.data_dir, '200k-real-and-fake-faces'),
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': os.path.join(args.data_dir, 'deepfake-dataset'),
    }
}

# بررسی دیتاست‌های انتخاب‌شده
valid_datasets = []
for ds in args.datasets:
    if ds not in dataset_configs:
        print(f"Warning: Dataset '{ds}' is invalid and will be ignored. Valid datasets: {list(dataset_configs.keys())}")
        continue
    config = dataset_configs[ds]
    all_files_exist = True
    for key, path in config.items():
        if 'csv' in key or 'dir' in key:
            if not os.path.exists(path):
                print(f"Error: {key} not found: {path}")
                all_files_exist = False
    if all_files_exist:
        valid_datasets.append(ds)
    else:
        print(f"Dataset '{ds}' will be skipped due to missing files or directories.")

if not valid_datasets:
    raise ValueError(f"No valid datasets selected or available! Choose from {list(dataset_configs.keys())}")

# لود مدل هرس‌شده
try:
    model, masks, pruned_model_path = load_pruned_model(args.checkpoint_path, device)
except Exception as e:
    print(f"Failed to load or save pruned model: {e}")
    raise

criterion = nn.BCEWithLogitsLoss()
results = {}
results_dir = '/kaggle/working/results'
os.makedirs(results_dir, exist_ok=True)

# حلقه اصلی برای ارزیابی
for dataset_name in valid_datasets:
    print(f"\nProcessing {dataset_name} dataset...")
    
    # لود دیتاست تست
    config = dataset_configs[dataset_name]
    try:
        dataset = Dataset_selector(
            dataset_mode=config['dataset_mode'],
            hardfake_csv_file=config.get('hardfake_csv_file'),
            hardfake_root_dir=config.get('hardfake_root_dir'),
            rvf10k_valid_csv=config.get('rvf10k_valid_csv'),  # فقط valid_csv برای rvf10k
            rvf10k_root_dir=config.get('rvf10k_root_dir'),
            realfake140k_test_csv=config.get('realfake140k_test_csv'),
            realfake140k_root_dir=config.get('realfake140k_root_dir'),
            realfake200k_test_csv=config.get('realfake200k_test_csv'),
            realfake200k_root_dir=config.get('realfake200k_root_dir'),
            realfake190k_root_dir=config.get('realfake190k_root_dir'),
            realfake330k_root_dir=config.get('realfake330k_root_dir'),
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ddp=False
        )
        test_loader = dataset.loader_test
        print(f"{dataset_name} test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {str(e)}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # بازسازی مدل برای هر دیتاست
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    try:
        checkpoint = torch.load(pruned_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded pruned model from {pruned_model_path}")
    except Exception as e:
        print(f"Error loading pruned model from {pruned_model_path}: {e}")
        results[dataset_name] = {'error': f"Failed to load pruned model: {str(e)}"}
        continue

    # ارزیابی مدل قبل از فاین‌تیونینگ
    try:
        pre_finetune_loss, pre_finetune_accuracy, pre_precision, pre_recall, pre_f1, pre_cm = evaluate_model(
            model, test_loader, device, criterion, dataset_name, 'pre_finetune', results_dir)
        print(f"{dataset_name} - Pre-Finetune Metrics:")
        print(f"  Test Loss: {pre_finetune_loss:.4f}")
        print(f"  Test Accuracy: {pre_finetune_accuracy:.2f}%")
        print(f"  Precision (Negative, Positive): {pre_precision[0]:.4f}, {pre_precision[1]:.4f}")
        print(f"  Recall (Negative, Positive): {pre_recall[0]:.4f}, {pre_recall[1]:.4f}")
        print(f"  F1-Score (Negative, Positive): {pre_f1[0]:.4f}, {pre_f1[1]:.4f}")
        print(f"  Confusion Matrix:\n{pre_cm}")
        results[dataset_name] = {
            'pre_finetune_loss': pre_finetune_loss,
            'pre_finetune_accuracy': pre_finetune_accuracy,
            'pre_precision_neg': pre_precision[0],
            'pre_precision_pos': pre_precision[1],
            'pre_recall_neg': pre_recall[0],
            'pre_recall_pos': pre_recall[1],
            'pre_f1_neg': pre_f1[0],
            'pre_f1_pos': pre_f1[1],
            'pre_cm': pre_cm.tolist()
        }
    except Exception as e:
        print(f"Error evaluating {dataset_name} before fine-tuning: {str(e)}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # فاین‌تیونینگ فقط لایه fc روی دیتاست تست
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(f"Fine-tuning only fc layer on {dataset_name} test set...")
    finetuned_model_path = fine_tune_model(model, test_loader, device, criterion, optimizer, args.epochs, dataset_name, results_dir)
    
    # لود مدل فاین‌تیون‌شده برای ارزیابی
    try:
        model.load_state_dict(torch.load(finetuned_model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded fine-tuned model from {finetuned_model_path}")
    except Exception as e:
        print(f"Error loading fine-tuned model from {finetuned_model_path}: {e}")
        results[dataset_name].update({'error': f"Failed to load fine-tuned model: {str(e)}"})
        continue
    
    # ارزیابی مدل بعد از فاین‌تیونینگ
    try:
        post_finetune_loss, post_finetune_accuracy, post_precision, post_recall, post_f1, post_cm = evaluate_model(
            model, test_loader, device, criterion, dataset_name, 'post_finetune', results_dir)
        print(f"{dataset_name} - Post-Finetune Metrics:")
        print(f"  Test Loss: {post_finetune_loss:.4f}")
        print(f"  Test Accuracy: {post_finetune_accuracy:.2f}%")
        print(f"  Precision (Negative, Positive): {post_precision[0]:.4f}, {post_precision[1]:.4f}")
        print(f"  Recall (Negative, Positive): {post_recall[0]:.4f}, {post_recall[1]:.4f}")
        print(f"  F1-Score (Negative, Positive): {post_f1[0]:.4f}, {post_f1[1]:.4f}")
        print(f"  Confusion Matrix:\n{post_cm}")
        results[dataset_name].update({
            'post_finetune_loss': post_finetune_loss,
            'post_finetune_accuracy': post_finetune_accuracy,
            'post_precision_neg': post_precision[0],
            'post_precision_pos': post_precision[1],
            'post_recall_neg': post_recall[0],
            'post_recall_pos': post_recall[1],
            'post_f1_neg': post_f1[0],
            'post_f1_pos': post_f1[1],
            'post_cm': post_cm.tolist()
        })
    except Exception as e:
        print(f"Error evaluating {dataset_name} after fine-tuning: {str(e)}")
        results[dataset_name].update({'error': str(e)})
    
    # محاسبه FLOPs و پارامترها
    input = torch.randn(1, 3, 256, 256).to(device)
    try:
        flops, params = profile(model, inputs=(input,))
        results[dataset_name]['flops'] = flops / 1e9  # GMac
        results[dataset_name]['params'] = params / 1e6  # M
    except Exception as e:
        print(f"Error calculating FLOPs and params for {dataset_name}: {str(e)}")
        results[dataset_name]['flops_error'] = str(e)

# ذخیره نتایج
results_path = os.path.join(results_dir, 'test_results.txt')
check_disk_space(results_dir)
try:
    with open(results_path, 'w') as f:
        for dataset_name in valid_datasets:
            f.write(f"Dataset: {dataset_name}\n")
            result = results.get(dataset_name, {'error': 'Not evaluated'})
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Pre-Finetune Metrics:\n")
                f.write(f"  Test Loss: {result['pre_finetune_loss']:.4f}\n")
                f.write(f"  Test Accuracy: {result['pre_finetune_accuracy']:.2f}%\n")
                f.write(f"  Precision (Negative): {result['pre_precision_neg']:.4f}\n")
                f.write(f"  Precision (Positive): {result['pre_precision_pos']:.4f}\n")
                f.write(f"  Recall (Negative): {result['pre_recall_neg']:.4f}\n")
                f.write(f"  Recall (Positive): {result['pre_recall_pos']:.4f}\n")
                f.write(f"  F1-Score (Negative): {result['pre_f1_neg']:.4f}\n")
                f.write(f"  F1-Score (Positive): {result['pre_f1_pos']:.4f}\n")
                f.write(f"  Confusion Matrix: {result['pre_cm']}\n")
                f.write(f"Post-Finetune Metrics:\n")
                f.write(f"  Test Loss: {result['post_finetune_loss']:.4f}\n")
                f.write(f"  Test Accuracy: {result['post_finetune_accuracy']:.2f}%\n")
                f.write(f"  Precision (Negative): {result['post_precision_neg']:.4f}\n")
                f.write(f"  Precision (Positive): {result['post_precision_pos']:.4f}\n")
                f.write(f"  Recall (Negative): {result['post_recall_neg']:.4f}\n")
                f.write(f"  Recall (Positive): {result['post_recall_pos']:.4f}\n")
                f.write(f"  F1-Score (Negative): {result['post_f1_neg']:.4f}\n")
                f.write(f"  F1-Score (Positive): {result['post_f1_pos']:.4f}\n")
                f.write(f"  Confusion Matrix: {result['post_cm']}\n")
                f.write(f"FLOPs: {result.get('flops', 'N/A'):.2f} GMac\n")
                f.write(f"Parameters: {result.get('params', 'N/A'):.2f} M\n")
            f.write("\n")
    print(f"Results saved to {results_path}")
    list_directory(results_dir)
except Exception as e:
    print(f"Error saving results to {results_path}: {str(e)}")

# چاپ نتایج نهایی
print("\nFinal Test Results:")
for dataset_name in valid_datasets:
    print(f"\nDataset: {dataset_name}")
    result = results.get(dataset_name, {'error': 'Not evaluated'})
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Pre-Finetune Metrics:")
        print(f"  Test Loss: {result['pre_finetune_loss']:.4f}")
        print(f"  Test Accuracy: {result['pre_finetune_accuracy']:.2f}%")
        print(f"  Precision (Negative, Positive): {result['pre_precision_neg']:.4f}, {result['pre_precision_pos']:.4f}")
        print(f"  Recall (Negative, Positive): {result['pre_recall_neg']:.4f}, {result['pre_recall_pos']:.4f}")
        print(f"  F1-Score (Negative, Positive): {result['pre_f1_neg']:.4f}, {result['pre_f1_pos']:.4f}")
        print(f"  Confusion Matrix:\n{result['pre_cm']}")
        print(f"Post-Finetune Metrics:")
        print(f"  Test Loss: {result['post_finetune_loss']:.4f}")
        print(f"  Test Accuracy: {result['post_finetune_accuracy']:.2f}%")
        print(f"  Precision (Negative, Positive): {result['post_precision_neg']:.4f}, {result['post_precision_pos']:.4f}")
        print(f"  Recall (Negative, Positive): {result['post_recall_neg']:.4f}, {result['post_recall_pos']:.4f}")
        print(f"  F1-Score (Negative, Positive): {result['post_f1_neg']:.4f}, {result['post_f1_pos']:.4f}")
        print(f"  Confusion Matrix:\n{result['post_cm']}")
        print(f"FLOPs: {result.get('flops', 'N/A'):.2f} GMac, Parameters: {result.get('params', 'N/A'):.2f} M")
