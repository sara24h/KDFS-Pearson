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
                        help='List of datasets to fine-tune and test')
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

# تابع برای لود و آماده‌سازی مدل هرس‌شده
def load_pruned_model(checkpoint_path, device):
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

    # بازسازی مدل
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # آماده‌سازی وزن‌های هرس‌شده
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
    checkpoints_dir = '/kaggle/working/checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    pruned_model_path = os.path.join(checkpoints_dir, 'pruned_model.pth')
    try:
        torch.save(model.state_dict(), pruned_model_path)
        print(f"Pruned model successfully saved to {pruned_model_path}")
        if os.path.exists(pruned_model_path):
            print(f"Confirmed: Pruned model exists at {pruned_model_path}")
        else:
            print(f"Error: Pruned model not found at {pruned_model_path}")
    except Exception as e:
        print(f"Error saving pruned model to {pruned_model_path}: {e}")
        raise

    return model, masks

# تابع فاین‌تیونینگ
def fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, dataset_name):
    best_acc = 0.0
    checkpoints_dir = '/kaggle/working/checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoints_dir, f'finetuned_{dataset_name}.pth')
    
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
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model for {dataset_name} with Valid Acc: {best_acc:.2f}% at {best_model_path}")
            except Exception as e:
                print(f"Error saving fine-tuned model to {best_model_path}: {e}")
    
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

# بررسی دیتاست‌های انتخاب‌شده
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

if not valid_datasets:
    raise ValueError(f"No valid datasets selected or available! Choose from {list(dataset_configs.keys())}")

# لود مدل هرس‌شده
model, masks = load_pruned_model(args.checkpoint_path, device)

# حلقه اصلی برای فاین‌تیونینگ و ارزیابی
results = {}
criterion = nn.BCEWithLogitsLoss()

for dataset_name in valid_datasets:
    print(f"\nProcessing {dataset_name} dataset...")
    
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
        print(f"{dataset_name} train dataset size: {len(train_loader.dataset) if train_loader else 0}")
        print(f"{dataset_name} valid dataset size: {len(valid_loader.dataset)}")
        print(f"{dataset_name} test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        results[dataset_name] = {'error': str(e)}
        continue
    
    # بازسازی مدل برای هر دیتاست (برای اطمینان از ریست شدن)
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    pruned_model_path = os.path.join('/kaggle/working/checkpoints', 'pruned_model.pth')
    try:
        model.load_state_dict(torch.load(pruned_model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded pruned model from {pruned_model_path}")
    except Exception as e:
        print(f"Error loading pruned model from {pruned_model_path}: {e}")
        results[dataset_name] = {'error': f"Failed to load pruned model: {str(e)}"}
        continue

    # ارزیابی مدل قبل از فاین‌تیونینگ
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
    
    # فاین‌تیونینگ فقط لایه fc
    if train_loader:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        print(f"Fine-tuning only fc layer on {dataset_name}...")
        best_model_path = fine_tune_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, dataset_name)
        
        # لود بهترین مدل برای ارزیابی
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            print(f"Successfully loaded fine-tuned model from {best_model_path}")
        except Exception as e:
            print(f"Error loading fine-tuned model from {best_model_path}: {e}")
            results[dataset_name].update({'error': f"Failed to load fine-tuned model: {str(e)}"})
            continue
    else:
        print(f"No training data available for {dataset_name}. Skipping fine-tuning.")
        best_model_path = None
    
    # ارزیابی روی دیتاست تست پس از فاین‌تیونینگ
    try:
        post_finetune_loss, post_finetune_accuracy, post_precision, post_recall, post_f1, post_cm = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Post-Finetune Metrics:")
        print(f"  Test Loss: {post_finetune_loss:.4f}")
        print(f"  Test Accuracy: {post_finetune_accuracy:.2f}%")
        print(f"  Precision: {post_precision:.4f}")
        print(f"  Recall: {post_recall:.4f}")
        print(f"  F1-Score: {post_f1:.4f}")
        print(f"  Confusion Matrix:\n{post_cm}")
        results[dataset_name].update({
            'post_finetune_loss': post_finetune_loss,
            'post_finetune_accuracy': post_finetune_accuracy,
            'post_precision': post_precision,
            'post_recall': post_recall,
            'post_f1': post_f1,
            'post_cm': post_cm.tolist()
        })
    except Exception as e:
        print(f"Error evaluating {dataset_name} after fine-tuning: {e}")
        results[dataset_name].update({'error': str(e)})
    
    # محاسبه FLOPs و پارامترها
    input = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    results[dataset_name]['flops'] = flops / 1e9  # GMac
    results[dataset_name]['params'] = params / 1e6  # M

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
            f.write(f"Pre-Finetune Metrics:\n")
            f.write(f"  Test Loss: {result['pre_finetune_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {result['pre_finetune_accuracy']:.2f}%\n")
            f.write(f"  Precision: {result['pre_precision']:.4f}\n")
            f.write(f"  Recall: {result['pre_recall']:.4f}\n")
            f.write(f"  F1-Score: {result['pre_f1']:.4f}\n")
            f.write(f"  Confusion Matrix: {result['pre_cm']}\n")
            f.write(f"Post-Finetune Metrics:\n")
            f.write(f"  Test Loss: {result['post_finetune_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {result['post_finetune_accuracy']:.2f}%\n")
            f.write(f"  Precision: {result['post_precision']:.4f}\n")
            f.write(f"  Recall: {result['post_recall']:.4f}\n")
            f.write(f"  F1-Score: {result['post_f1']:.4f}\n")
            f.write(f"  Confusion Matrix: {result['post_cm']}\n")
            f.write(f"FLOPs: {result['flops']:.2f} GMac\n")
            f.write(f"Parameters: {result['params']:.2f} M\n")
        f.write("\n")
print(f"Results saved to {os.path.join(results_dir, 'finetune_results.txt')}")

# چاپ نتایج نهایی
print("\nFinal Fine-tuning Results:")
for dataset_name in valid_datasets:
    print(f"\nDataset: {dataset_name}")
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
        print(f"Post-Finetune Metrics:")
        print(f"  Test Loss: {result['post_finetune_loss']:.4f}")
        print(f"  Test Accuracy: {result['post_finetune_accuracy']:.2f}%")
        print(f"  Precision: {result['post_precision']:.4f}")
        print(f"  Recall: {result['post_recall']:.4f}")
        print(f"  F1-Score: {result['post_f1']:.4f}")
        print(f"  Confusion Matrix:\n{result['post_cm']}")
        print(f"FLOPs: {result['flops']:.2f} GMac, Parameters: {result['params']:.2f} M")
