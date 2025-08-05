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
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def parse_args():
    parser = argparse.ArgumentParser(description='Test generalization of pruned ResNet50 model.')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/', help='Base directory containing dataset folders')
    parser.add_argument('--datasets', type=str, nargs='+', default=['rvf10k'], help='List of datasets to test')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    return parser.parse_args()

# تنظیمات اولیه
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# مسیر مدل
CHECKPOINT_PATH = '/kaggle/input/pruned_140k_resnet50/pytorch/default/1/pruned_model.pth'
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file {CHECKPOINT_PATH} not found!")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
state_dict = checkpoint['student'] if 'student' in checkpoint else checkpoint

# تابع فاین‌تیونینگ
def fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, dataset_name):
    results_dir = '/kaggle/working/results'
    best_model_path = os.path.join(results_dir, f'finetuned_{dataset_name}.pth')
    os.makedirs(results_dir, exist_ok=True)
    
    # فریز کردن تمام لایه‌ها به‌جز fc
    for name, param in model.named_parameters():
        param.requires_grad = 'fc' in name
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            with autocast('cuda', enabled=device.type == 'cuda'):
                outputs, _ = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_accuracy = 100 * correct / total if total > 0 else 0
        valid_loss, valid_accuracy, _, _, _, _ = evaluate_model(model, valid_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} - {dataset_name} - Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%")
        
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(), best_model_path)
    
    return best_model_path

# تابع ارزیابی
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
            with autocast('cuda', enabled=device.type == 'cuda'):
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
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
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

# بررسی دیتاست‌ها
valid_datasets = []
for ds in args.datasets:
    if ds not in dataset_configs:
        print(f"Warning: Dataset '{ds}' not supported. Supported: {list(dataset_configs.keys())}")
        continue
    config = dataset_configs[ds]
    all_files_exist = True
    for key, path in config.items():
        if ('csv' in key or 'dir' in key) and not os.path.exists(path):
            print(f"Error: Path not found: {path}")
            all_files_exist = False
    if all_files_exist:
        valid_datasets.append(ds)
if not valid_datasets:
    raise ValueError("No valid datasets found!")

# حلقه اصلی
results = {}
criterion = nn.BCEWithLogitsLoss()

for dataset_name in valid_datasets:
    print(f"\nProcessing {dataset_name}...")
    
    # بازسازی مدل با فرض بدون ماسک
    model = ResNet_50_pruned_hardfakevsreal(masks=[])  # استفاده از لیست خالی برای masks
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    
    # لود وزن‌های پرون‌شده مستقیماً
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
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
        test_loader = dataset.loader_test
        print(f"{dataset_name} - Train: {len(train_loader.dataset) if train_loader else 0}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # ارزیابی بدون فاین‌تیونینگ
    try:
        pre_finetune_loss, pre_finetune_accuracy, pre_precision, pre_recall, pre_f1, pre_cm = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Pre-Finetune: Loss: {pre_finetune_loss:.4f}, Acc: {pre_finetune_accuracy:.2f}%, Precision: {pre_precision:.4f}, Recall: {pre_recall:.4f}, F1: {pre_f1:.4f}")
        results[dataset_name] = {
            'pre_finetune_loss': pre_finetune_loss,
            'pre_finetune_accuracy': pre_finetune_accuracy,
            'pre_precision': pre_precision,
            'pre_recall': pre_recall,
            'pre_f1': pre_f1,
            'pre_cm': pre_cm.tolist()
        }
    except Exception as e:
        print(f"Error evaluating {dataset_name} pre-finetune: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # فاین‌تیونینگ
    if train_loader:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        best_model_path = fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, dataset_name)
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        print(f"No training data for {dataset_name}. Skipping fine-tuning.")
    
    # ارزیابی با فاین‌تیونینگ
    try:
        post_finetune_loss, post_finetune_accuracy, post_precision, post_recall, post_f1, post_cm = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Post-Finetune: Loss: {post_finetune_loss:.4f}, Acc: {post_finetune_accuracy:.2f}%, Precision: {post_precision:.4f}, Recall: {post_recall:.4f}, F1: {post_f1:.4f}")
        results[dataset_name].update({
            'post_finetune_loss': post_finetune_loss,
            'post_finetune_accuracy': post_finetune_accuracy,
            'post_precision': post_precision,
            'post_recall': post_recall,
            'post_f1': post_f1,
            'post_cm': post_cm.tolist()
        })
    except Exception as e:
        print(f"Error evaluating {dataset_name} post-finetune: {e}")
        results[dataset_name].update({'error': str(e)})
    
    # محاسبه FLOPs و پارامترها
    try:
        input = torch.randn(1, 3, 256, 256).to(device)
        flops, params = profile(model, inputs=(input,))
        results[dataset_name]['flops'] = flops / 1e9
        results[dataset_name]['params'] = params / 1e6
    except Exception as e:
        print(f"Error calculating FLOPs/params for {dataset_name}: {e}")
        results[dataset_name]['flops'] = None
        results[dataset_name]['params'] = None

# ذخیره نتایج
results_dir = '/kaggle/working/results'
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'finetune_results.txt'), 'w') as f:
    for dataset_name in valid_datasets:
        f.write(f"Dataset: {dataset_name}\n")
        result = results.get(dataset_name, {'error': 'Not evaluated'})
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
        else:
            f.write(f"Pre-Finetune: Loss: {result['pre_finetune_loss']:.4f}, Acc: {result['pre_finetune_accuracy']:.2f}%, Precision: {result['pre_precision']:.4f}, Recall: {result['pre_recall']:.4f}, F1: {result['pre_f1']:.4f}\n")
            f.write(f"Post-Finetune: Loss: {result['post_finetune_loss']:.4f}, Acc: {result['post_finetune_accuracy']:.2f}%, Precision: {result['post_precision']:.4f}, Recall: {result['post_recall']:.4f}, F1: {result['post_f1']:.4f}\n")
            f.write(f"FLOPs: {result['flops']:.2f} GMac, Parameters: {result['params']:.2f} M\n")
        f.write("\n")
print(f"Results saved to {os.path.join(results_dir, 'finetune_results.txt')}")

# چاپ نتایج نهایی
print("\nFinal Results:")
for dataset_name in valid_datasets:
    print(f"\nDataset: {dataset_name}")
    result = results.get(dataset_name, {'error': 'Not evaluated'})
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Pre-Finetune: Loss: {result['pre_finetune_loss']:.4f}, Acc: {result['pre_finetune_accuracy']:.2f}%, Precision: {result['pre_precision']:.4f}, Recall: {result['pre_recall']:.4f}, F1: {result['pre_f1']:.4f}")
        print(f"Post-Finetune: Loss: {result['post_finetune_loss']:.4f}, Acc: {result['post_finetune_accuracy']:.2f}%, Precision: {result['post_precision']:.4f}, Recall: {result['post_recall']:.4f}, F1: {result['post_f1']:.4f}")
        print(f"FLOPs: {result['flops']:.2f} GMac, Parameters: {result['params']:.2f} M")
