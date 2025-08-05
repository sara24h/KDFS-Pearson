import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import argparse
from data_dataset import Dataset_selector  # فرض می‌کنیم data_dataset.py همان کد شماست

def evaluate_model(model, dataloader, device, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze()  # برای طبقه‌بندی باینری
            preds = (outputs >= 0.5).float()  # آستانه 0.5 برای پیش‌بینی

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nEvaluation on {dataset_name} dataset:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Generalization (Test Accuracy as proxy): {accuracy:.4f}")

    return accuracy, precision, recall, f1, cm

def fine_tune_model(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # ارزیابی روی validation برای بررسی پیشرفت
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs).squeeze()
                preds = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_accuracy = correct / total
        print(f"Validation Accuracy after epoch {epoch+1}: {val_accuracy:.4f}")
        model.train()

    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on selected dataset")
    parser.add_argument('--dataset_mode', type=str, required=True, 
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help="Dataset to evaluate: hardfake, rvf10k, 140k, 190k, 200k, 330k")
    parser.add_argument('--model_path', type=str, default='/kaggle/input/pruned_140k_resnet50/pytorch/default/1/pruned_model.pth',
                        help="Path to the pre-trained model")
    parser.add_argument('--epochs', type=int, default=5, help="Number of fine-tuning epochs")
    args = parser.parse_args()

    # تنظیم دستگاه
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # تنظیمات دیتاست
    dataset_args = {
        'dataset_mode': args.dataset_mode,
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'num_workers': 8,
        'pin_memory': True,
        'ddp': False,  # برای سادگی، DDP غیرفعال است
    }

    # مسیرهای دیتاست‌ها
    if args.dataset_mode == 'hardfake':
        dataset_args.update({
            'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
            'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces'
        })
    elif args.dataset_mode == 'rvf10k':
        dataset_args.update({
            'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
            'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
            'rvf10k_root_dir': '/kaggle/input/rvf10k'
        })
    elif args.dataset_mode == '140k':
        dataset_args.update({
            'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
            'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
            'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
            'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces'
        })
    elif args.dataset_mode == '190k':
        dataset_args.update({
            'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset'
        })
    elif args.dataset_mode == '200k':
        dataset_args.update({
            'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
            'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
            'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
            'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces'
        })
    elif args.dataset_mode == '330k':
        dataset_args.update({
            'realfake330k_root_dir': '/kaggle/input/deepfake-dataset'
        })

    # بارگذاری دیتاست
    try:
        dataset = Dataset_selector(**dataset_args)
        train_loader = dataset.loader_train
        val_loader = dataset.loader_val
        test_loader = dataset.loader_test
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # بارگذاری مدل از مسیر مشخص‌شده
    try:
        model = torch.load(args.model_path, map_location=device)
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model = model.to(device)

    # بررسی سازگاری مدل با دیتاست
    try:
        sample_batch = next(iter(test_loader))
        sample_images = sample_batch[0].to(device)
        model.eval()
        with torch.no_grad():
            _ = model(sample_images)  # تست ورودی
        print("Model structure is compatible with input data.")
    except Exception as e:
        print(f"Error: Model is not compatible with input data: {e}")
        return

    print("\n=== Evaluating model before fine-tuning ===")
    evaluate_model(model, test_loader, device, args.dataset_mode)

    # فاین‌تیونینگ مدل
    print("\n=== Fine-tuning model ===")
    model = fine_tune_model(model, train_loader, val_loader, device, epochs=args.epochs)

    # ارزیابی مدل بعد از فاین‌تیونینگ
    print("\n=== Evaluating model after fine-tuning ===")
    evaluate_model(model, test_loader, device, args.dataset_mode)

if __name__ == "__main__":
    main()
