import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# تابع ارزیابی مدل
def evaluate_model(model, test_loader, device, dataset_name):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs, _ = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # پیش‌بینی‌ها
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\nResults for {dataset_name}:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1

# مسیر فایل وزن‌ها
model_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'

# لود checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# بررسی کلیدها
print("Keys in checkpoint:", list(checkpoint.keys()))

# فرض می‌کنیم کلید ماسک‌ها 'prune_masks' است (بعد از بررسی تغییر دهید)
masks = checkpoint.get('prune_masks', [])  # اگر کلید وجود نداشت، لیست خالی
state_dict = checkpoint['state_dict']

# تعریف مدل
model = ResNet_50_pruned_hardfakevsreal(masks=masks)
model.load_state_dict(state_dict, strict=False)

# انتقال مدل به دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# تعریف دیتاست‌های تست
datasets = [
    {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
    },
    {
        'dataset_mode': '140k',
        'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces',
    },
    {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
    },
    {
        'dataset_mode': '200k',
        'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
        'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
        'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
        'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
    },
    {
        'dataset_mode': '330k',
        'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
    }
]

# ارزیابی روی هر دیتاست
results = {}
for dataset_config in datasets:
    dataset_mode = dataset_config['dataset_mode']
    print(f"\nLoading test dataset for {dataset_mode}...")
    
    # لود دیتاست
    dataset = Dataset_selector(
        dataset_mode=dataset_mode,
        train_batch_size=64,
        eval_batch_size=64,
        ddp=False,
        **dataset_config
    )
    
    # لود دیتاست تست
    test_loader = dataset.loader_test
    
    # ارزیابی مدل
    avg_loss, accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device, dataset_mode)
    results[dataset_mode] = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# چاپ نتایج کلی
print("\nSummary of Results:")
for dataset_mode, metrics in results.items():
    print(f"\n{dataset_mode}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
