import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector  # Assuming dataset.py is available

# Function to evaluate the model and collect predictions for confusion matrix
def evaluate_model(model, data_loader, device, dataset_name=""):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate metrics for each class (remove average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics for each class
    print(f"\n{dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Fake): {precision[0]:.4f}, Precision (Real): {precision[1]:.4f}")
    print(f"Recall (Fake): {recall[0]:.4f}, Recall (Real): {recall[1]:.4f}")
    print(f"F1-Score (Fake): {f1[0]:.4f}, F1-Score (Real): {f1[1]:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,  # Array of precision for each class
        'recall': recall,        # Array of recall for each class
        'f1_score': f1,         # Array of F1-score for each class
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

# Function to freeze layers
def freeze_layers(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return model

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and finetune a model with frozen layers on selected dataset.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'])
    parser.add_argument('--hardfake_csv_file', type=str, default=None)
    parser.add_argument('--hardfake_root_dir', type=str, default=None)
    parser.add_argument('--rvf10k_train_csv', type=str, default=None)
    parser.add_argument('--rvf10k_valid_csv', type=str, default=None)
    parser.add_argument('--rvf10k_root_dir', type=str, default=None)
    parser.add_argument('--realfake140k_train_csv', type=str, default=None)
    parser.add_argument('--realfake140k_valid_csv', type=str, default=None)
    parser.add_argument('--realfake140k_test_csv', type=str, default=None)
    parser.add_argument('--realfake140k_root_dir', type=str, default=None)
    parser.add_argument('--realfake190k_root_dir', type=str, default=None)
    parser.add_argument('--realfake200k_train_csv', type=str, default=None)
    parser.add_argument('--realfake200k_val_csv', type=str, default=None)
    parser.add_argument('--realfake200k_test_csv', type=str, default=None)
    parser.add_argument('--realfake200k_root_dir', type=str, default=None)
    parser.add_argument('--realfake330k_root_dir', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = Dataset_selector(
        dataset_mode=args.dataset_mode,
        hardfake_csv_file=args.hardfake_csv_file,
        hardfake_root_dir=args.hardfake_root_dir,
        rvf10k_train_csv=args.rvf10k_train_csv,
        rvf10k_valid_csv=args.rvf10k_valid_csv,
        rvf10k_root_dir=args.rvf10k_root_dir,
        realfake140k_train_csv=args.realfake140k_train_csv,
        realfake140k_valid_csv=args.realfake140k_valid_csv,
        realfake140k_test_csv=args.realfake140k_test_csv,
        realfake140k_root_dir=args.realfake140k_root_dir,
        realfake190k_root_dir=args.realfake190k_root_dir,
        realfake200k_train_csv=args.realfake200k_train_csv,
        realfake200k_val_csv=args.realfake200k_val_csv,
        realfake200k_test_csv=args.realfake200k_test_csv,
        realfake200k_root_dir=args.realfake200k_root_dir,
        realfake330k_root_dir=args.realfake330k_root_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        ddp=False
    )

    train_loader = DataLoader(dataset.loader_train.dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset.loader_val.dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset.loader_test.dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # Load model
    try:
        model = torch.load('/kaggle/input/pruned_140k_resnet50/pytorch/default/1/full_model.pth', map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    model = model.to(device)
    model = freeze_layers(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after freezing: {trainable_params}")

    # Check model compatibility
    try:
        sample_batch = next(iter(test_loader))
        sample_images = sample_batch[0].to(device)
        with torch.no_grad():
            sample_output = model(sample_images)
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]
        print(f"Sample model output for dataset {args.dataset_mode}: {sample_output.shape}")
    except Exception as e:
        print(f"Error running model on sample data from dataset {args.dataset_mode}: {e}")
        exit()

    # Evaluate before finetuning
    print(f"\nEvaluating pruned model before finetuning on dataset {args.dataset_mode} (test data):")
    metrics_before = evaluate_model(model, test_loader, device, f"Test Data Before Finetuning ({args.dataset_mode})")
    plot_confusion_matrix(metrics_before['confusion_matrix'], 
                         f"Confusion Matrix Before Finetuning ({args.dataset_mode})",
                         "cm_before_finetuning.png")

    # Finetune model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model.train()
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Evaluate after finetuning
    print(f"\nEvaluating model after finetuning on dataset {args.dataset_mode} (test data):")
    metrics_after = evaluate_model(model, test_loader, device, f"Test Data After Finetuning ({args.dataset_mode})")
    plot_confusion_matrix(metrics_after['confusion_matrix'], 
                         f"Confusion Matrix After Finetuning ({args.dataset_mode})",
                         "cm_after_finetuning.png")

if __name__ == "__main__":
    main()
