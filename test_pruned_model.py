import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import Dataset_selector
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """بارگذاری مدل از فایل .pt"""
    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),weights_only=False)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # فقط تنسور اصلی را بگیر

            predicted = (torch.sigmoid(outputs) > 0.5).float().squeeze()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # محاسبه معیارها
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, dataset_name):
    """رسم ماتریس اشتباه"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # تنظیمات argparse
    parser = argparse.ArgumentParser(description='Test ResNet_pruned model on datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                       help='Dataset to test: hardfake, rvf10k, 140k, 190k, 200k, 330k')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the pruned model file (e.g., pruned_model.pt)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation (default: 64)')
    args = parser.parse_args()

    # دستگاه (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # بارگذاری مدل
    model = load_model(args.model_path).to(device)
    print("Model loaded successfully")

    # بارگذاری دیتاست
    dataset_config = {
        'dataset_mode': args.dataset,
        'eval_batch_size': args.batch_size,
        'ddp': False  # غیرفعال کردن DDP برای ارزیابی
    }

    # تنظیم مسیرهای دیتاست بر اساس انتخاب کاربر
    if args.dataset == 'hardfake':
        dataset_config.update({
            'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
            'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces'
        })
    elif args.dataset == 'rvf10k':
        dataset_config.update({
            'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
            'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
            'rvf10k_root_dir': '/kaggle/input/rvf10k'
        })
    elif args.dataset == '140k':
        dataset_config.update({
            'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
            'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
            'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
            'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces'
        })
    elif args.dataset == '190k':
        dataset_config.update({
            'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset'
        })
    elif args.dataset == '200k':
        dataset_config.update({
            'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
            'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
            'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
            'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces'
        })
    elif args.dataset == '330k':
        dataset_config.update({
            'realfake330k_root_dir': '/kaggle/input/deepfake-dataset'
        })

    dataset = Dataset_selector(**dataset_config)
    test_loader = dataset.loader_test

    # ارزیابی مدل
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, device)

    # نمایش نتایج
    print(f"\nResults on {args.dataset} dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # رسم ماتریس اشتباه
    plot_confusion_matrix(cm, args.dataset)

if __name__ == "__main__":
    main()
