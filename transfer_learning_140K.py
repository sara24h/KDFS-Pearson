import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import argparse
from data.dataset import Dataset_selector  # Assuming dataset.py is available

# Function to evaluate the model
def evaluate_model(model, data_loader, device, dataset_name=""):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Handle tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Select the primary output tensor
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"\nEvaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Function to freeze layers
def freeze_layers(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:  # Only the fc layer remains unfrozen
            param.requires_grad = False
    return model

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and finetune a model with frozen layers on selected dataset.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help='Dataset mode to use (hardfake, rvf10k, 140k, 190k, 200k, 330k)')
    parser.add_argument('--hardfake_csv_file', type=str, default=None, help='Path to hardfake CSV file')
    parser.add_argument('--hardfake_root_dir', type=str, default=None, help='Root directory for hardfake dataset')
    parser.add_argument('--rvf10k_train_csv', type=str, default=None, help='Path to rvf10k train CSV')
    parser.add_argument('--rvf10k_valid_csv', type=str, default=None, help='Path to rvf10k valid CSV')
    parser.add_argument('--rvf10k_root_dir', type=str, default=None, help='Root directory for rvf10k dataset')
    parser.add_argument('--realfake140k_train_csv', type=str, default=None, help='Path to 140k train CSV')
    parser.add_argument('--realfake140k_valid_csv', type=str, default=None, help='Path to 140k valid CSV')
    parser.add_argument('--realfake140k_test_csv', type=str, default=None, help='Path to 140k test CSV')
    parser.add_argument('--realfake140k_root_dir', type=str, default=None, help='Root directory for 140k dataset')
    parser.add_argument('--realfake190k_root_dir', type=str, default=None, help='Root directory for 190k dataset')
    parser.add_argument('--realfake200k_train_csv', type=str, default=None, help='Path to 200k train CSV')
    parser.add_argument('--realfake200k_val_csv', type=str, default=None, help='Path to 200k validation CSV')
    parser.add_argument('--realfake200k_test_csv', type=str, default=None, help='Path to 200k test CSV')
    parser.add_argument('--realfake200k_root_dir', type=str, default=None, help='Root directory for 200k dataset')
    parser.add_argument('--realfake330k_root_dir', type=str, default=None, help='Root directory for 330k dataset')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for finetuning')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for finetuning')
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

    # Set up DataLoader with optimal number of workers
    train_loader = DataLoader(dataset.loader_train.dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset.loader_val.dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset.loader_test.dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # Load model
    try:
        model = torch.load('/kaggle/input/pruned_140k_resnet50/pytorch/default/1/full_model.pth', map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file contains the complete model object and is compatible with the PyTorch version.")
        exit()

    model = model.to(device)

    # Freeze layers
    model = freeze_layers(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after freezing: {trainable_params}")

    # Check model compatibility with data
    try:
        sample_batch = next(iter(test_loader))
        sample_images = sample_batch[0].to(device)
        with torch.no_grad():
            sample_output = model(sample_images)
            # Handle tuple output
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]  # Select the primary output tensor
        print(f"Sample model output for dataset {args.dataset_mode}: {sample_output.shape}")
    except Exception as e:
        print(f"Error running model on sample data from dataset {args.dataset_mode}: {e}")
        print("Please ensure the model architecture is compatible with the input data.")
        exit()

    # Evaluate model before finetuning on test set
    print(f"\nEvaluating pruned model before finetuning on dataset {args.dataset_mode} (test data):")
    metrics_before = evaluate_model(model, test_loader, device, f"test data ({args.dataset_mode})")

    # Initial evaluation on validation set
    print(f"\nEvaluating pruned model before finetuning on dataset {args.dataset_mode} (validation data):")
    evaluate_model(model, val_loader, device, f"validation data ({args.dataset_mode})")

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
            # Handle tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Select the primary output tensor
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Evaluate on validation set after each epoch
        print(f"\nEvaluation at epoch {epoch+1} on validation data ({args.dataset_mode}):")
        evaluate_model(model, val_loader, device, f"validation data ({args.dataset_mode})")

    # Evaluate model after finetuning on test set
    print(f"\nEvaluating model after finetuning on dataset {args.dataset_mode} (test data):")
    metrics_after = evaluate_model(model, test_loader, device, f"test data ({args.dataset_mode})")

    # Compare performance before and after finetuning on test set
    print(f"\nPerformance comparison before and after finetuning on test data ({args.dataset_mode}):")
    print(f"Change in accuracy: {(metrics_after['accuracy'] - metrics_before['accuracy']):.4f}")
    print(f"Change in precision: {(metrics_after['precision'] - metrics_before['precision']):.4f}")
    print(f"Change in recall: {(metrics_after['recall'] - metrics_before['recall']):.4f}")
    print(f"Change in F1-Score: {(metrics_after['f1_score'] - metrics_before['f1_score']):.4f}")

    # Save finetuned model
    torch.save(model, f'/kaggle/working/finetuned_pruned_model_{args.dataset_mode}.pth')

if __name__ == "__main__":
    main()
