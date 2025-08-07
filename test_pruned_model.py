import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from thop import profile
import os
import warnings
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Test a model on a specified dataset.')
parser.add_argument('--dataset', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                    help='Name of the dataset to test (e.g., hardfake, rvf10k, 140k, 190k, 200k, 330k)')
args = parser.parse_args()

# Load model
try:
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device, weights_only=False)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise Exception("Failed to load the model.")
model = model.to(device)
model.eval()

# Calculate FLOPs and parameters
input = torch.randn(1, 3, 256, 256).to(device)  # Match dataset image size
try:
    flops, params = profile(model, inputs=(input,))
    logging.info(f"Params: {params/1e6:.2f}M, FLOPs: {flops/1e6:.2f}M")
except Exception as e:
    logging.error(f"Error profiling model: {e}")

# Dataset configurations
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
        'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200kroom_data = Dataset_selector(**dataset_configs[dataset_name])
    logging.info(f"{dataset_name} dataset loaded successfully")
except Exception as e:
    logging.error(f"Error loading {dataset_name} dataset: {e}")
    raise

# Function to process model output
def process_model_output(outputs):
    try:
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Select first element if tuple
        if outputs.dim() > 1:
            outputs = outputs.squeeze(1)
        return outputs
    except Exception as e:
        logging.error(f"Error processing model output: {e}")
        raise

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            try:
                outputs = process_model_output(model(inputs))
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logging.error(f"Error during evaluation forward pass: {e}")
                raise
    accuracy = accuracy_score(true_labels, predictions) * 100
    f1 = f1_score(true_labels, predictions, average='binary') * 100
    avg_loss = running_loss / len(dataloader)
    return accuracy, f1, avg_loss

# Training function
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=5):
    metrics = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            try:
                outputs = process_model_output(model(inputs))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                logging.error(f"Error during training forward pass: {e}")
                raise
        metrics['train_loss'].append(running_loss / len(trainloader))
        logging.info(f"Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader):.4f}")
        
        # Evaluate on validation set
        accuracy, f1, val_loss = evaluate(model, valloader, criterion)
        metrics['val_accuracy'].append(accuracy)
        metrics['val_f1'].append(f1)
        metrics['val_loss'].append(val_loss)
        logging.info(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Validation Loss: {val_loss:.4f}")
    return metrics

# Test without fine-tuning
criterion = nn.BCEWithLogitsLoss()
logging.info(f"\nTesting {dataset_name} without fine-tuning...")
try:
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_without_finetune = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    logging.info(f"{dataset_name} Without Fine-tuning - Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Loss: {loss:.4f}")
except Exception as e:
    logging.error(f"Error during evaluation without fine-tuning: {e}")
    raise

# Fine-tuning
logging.info(f"\nFine-tuning on {dataset_name}...")
try:
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device, weights_only=False)
    model = model.to(device)
except Exception as e:
    logging.error(f"Error loading model for fine-tuning: {e}")
    raise

# Freeze initial layers
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Train the model
metrics = train_model(model, dataset.loader_train, dataset.loader_val, criterion, optimizer, num_epochs=5)

# Test on test set
try:
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_with_finetune = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    logging.info(f"{dataset_name} With Fine-tuning - Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Loss: {loss:.4f}")
except Exception as e:
    logging.error(f"Error during evaluation with fine-tuning: {e}")
    raise

# Save the fine-tuned model
try:
    torch.save(model, f'finetuned_{dataset_name}.pt')
    logging.info(f"Fine-tuned model saved as finetuned_{dataset_name}.pt")
except Exception as e:
    logging.error(f"Error saving fine-tuned model: {e}")
    raise

# Summary of results
logging.info("\nSummary of Results:")
logging.info("\nWithout Fine-tuning:")
logging.info(f"{dataset_name}: Accuracy = {results_without_finetune['accuracy']:.2f}%, F1-Score = {results_without_finetune['f1_score']:.2f}%, Loss = {results_without_finetune['loss']:.4f}")
logging.info("\nWith Fine-tuning:")
logging.info(f"{dataset_name}: Accuracy = {results_with_finetune['accuracy']:.2f}%, F1-Score = {results_with_finetune['f1_score']:.2f}%, Loss = {results_with_finetune['loss']:.4f}")
