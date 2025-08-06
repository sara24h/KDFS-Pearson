import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# وارد کردن کلاس‌های دیتاست از فایل data.dataset.py
sys.path.append('/kaggle/working/')  # مسیر فایل data.dataset.py
from data.dataset import FaceDataset, Dataset_selector

# تابع آموزش و فاین‌تیون مدل
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    for epoch in range(num_epochs):
        # حالت آموزش
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # حذف ابعاد اضافی برای BCEWithLogitsLoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}')

        # حالت اعتبارسنجی
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%, Validation Loss: {val_loss/len(val_loader):.4f}')

# تابع ارزیابی مدل با معیارهای اضافی
def evaluate_model(model, test_loader, dataset_name, device='cuda'):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # تبدیل به آرایه numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # محاسبه معیارها
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = running_loss / len(test_loader)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # محاسبه ماتریس درهم‌ریختگی
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # چاپ نتایج
    print(f'\n{dataset_name} Evaluation:')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Loss: {avg_loss:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm
    }

# تنظیم دستگاه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# بارگذاری مدل
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # تنظیم برای طبقه‌بندی باینری
model_path = "/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt"
state_dict = torch.load(model_path, map_location=device)
try:
    model.load_state_dict(state_dict)
except RuntimeError as e:
    print(f"Error loading state dict: {e}")
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)
model = model.to(device)

# تعریف دیتاست‌ها برای تست
datasets = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
        'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces',
        'eval_batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False,
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
        'eval_batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False,
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
        'eval_batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False,
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
        'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
        'eval_batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False,
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
        'eval_batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False,
    },
}

# ارزیابی مدل قبل از فاین‌تیون
print("\nEvaluating model before fine-tuning...")
results_before = {}
for dataset_name, dataset_params in datasets.items():
    print(f"\nEvaluating on {dataset_name} dataset (before fine-tuning)...")
    try:
        dataset = Dataset_selector(**dataset_params)
        results_before[dataset_name] = evaluate_model(model, dataset.loader_test, dataset_name, device=device)
    except Exception as e:
        print(f"Error evaluating {dataset_name} (before fine-tuning): {e}")

# تعریف دیتاست برای فاین‌تیون (140k)
finetune_dataset = Dataset_selector(
    dataset_mode='140k',
    realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
    realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
    realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
    realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
    train_batch_size=64,
    eval_batch_size=64,
    num_workers=4,
    pin_memory=True,
    ddp=False,
)

# دسترسی به DataLoaderها برای فاین‌تیون
train_loader = finetune_dataset.loader_train
val_loader = finetune_dataset.loader_val

# تنظیم معیار خطا و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# فاین‌تیون مدل
print("\nStarting fine-tuning on 140k dataset...")
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

# ذخیره مدل فاین‌تیون‌شده
torch.save(model.state_dict(), '/kaggle/working/finetuned_resnet50_140k.pt')
print("Fine-tuned model saved to /kaggle/working/finetuned_resnet50_140k.pt")

# ارزیابی مدل بعد از فاین‌تیون
print("\nEvaluating model after fine-tuning...")
results_after = {}
for dataset_name, dataset_params in datasets.items():
    print(f"\nEvaluating on {dataset_name} dataset (after fine-tuning)...")
    try:
        dataset = Dataset_selector(**dataset_params)
        results_after[dataset_name] = evaluate_model(model, dataset.loader_test, dataset_name, device=device)
    except Exception as e:
        print(f"Error evaluating {dataset_name} (after fine-tuning): {e}")

# نمایش مقایسه نتایج قبل و بعد از فاین‌تیون
print("\nComparison of Evaluation Results (Before vs After Fine-Tuning):")
for dataset_name in datasets.keys():
    print(f"\n{dataset_name}:")
    if dataset_name in results_before:
        print("Before Fine-Tuning:")
        print(f"  Accuracy: {results_before[dataset_name]['accuracy']:.2f}%")
        print(f"  Loss: {results_before[dataset_name]['loss']:.4f}")
        print(f"  Precision: {results_before[dataset_name]['precision']:.4f}")
        print(f"  Recall: {results_before[dataset_name]['recall']:.4f}")
        print(f"  F1 Score: {results_before[dataset_name]['f1']:.4f}")
        print(f"  Specificity: {results_before[dataset_name]['specificity']:.4f}")
        print(f"  Confusion Matrix:\n{results_before[dataset_name]['confusion_matrix']}")
    else:
        print("Before Fine-Tuning: Evaluation failed")
    
    if dataset_name in results_after:
        print("After Fine-Tuning:")
        print(f"  Accuracy: {results_after[dataset_name]['accuracy']:.2f}%")
        print(f"  Loss: {results_after[dataset_name]['loss']:.4f}")
        print(f"  Precision: {results_after[dataset_name]['precision']:.4f}")
        print(f"  Recall: {results_after[dataset_name]['recall']:.4f}")
        print(f"  F1 Score: {results_after[dataset_name]['f1']:.4f}")
        print(f"  Specificity: {results_after[dataset_name]['specificity']:.4f}")
        print(f"  Confusion Matrix:\n{results_after[dataset_name]['confusion_matrix']}")
    else:
        print("After Fine-Tuning: Evaluation failed")
