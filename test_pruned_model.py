import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
import torchvision.models as models

# Define the pruned ResNet architecture
class Bottleneck_pruned(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes1, planes2, planes3, stride=1, downsample=None):
        super(Bottleneck_pruned, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes1)
        self.conv2 = nn.Conv2d(planes1, planes2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes2)
        self.conv3 = nn.Conv2d(planes2, planes3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNet_pruned(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet_pruned, self).__init__()
        self.inplanes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1
        self.layer1 = nn.Sequential(
            Bottleneck_pruned(64, 11, 7, 57, downsample=nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )),
            Bottleneck_pruned(256, 7, 10, 43),
            Bottleneck_pruned(256, 10, 8, 41)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            Bottleneck_pruned(256, 28, 19, 96, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )),
            Bottleneck_pruned(512, 25, 22, 94),
            Bottleneck_pruned(512, 25, 13, 77),
            Bottleneck_pruned(512, 18, 16, 59)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            Bottleneck_pruned(512, 30, 39, 113, stride=2, downsample=nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(1024)
            )),
            Bottleneck_pruned(1024, 51, 18, 112),
            Bottleneck_pruned(1024, 46, 15, 71),
            Bottleneck_pruned(1024, 47, 15, 67),
            Bottleneck_pruned(1024, 28, 8, 51),
            Bottleneck_pruned(1024, 30, 11, 44)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            Bottleneck_pruned(1024, 35, 50, 66, stride=2, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(2048)
            )),
            Bottleneck_pruned(2048, 86, 12, 64),
            Bottleneck_pruned(2048, 57, 18, 102)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def load_model(model_path, device):
    """Load the pruned model"""
    model = ResNet_pruned(num_classes=1)
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def test_model(model, test_loader, device):
    """Test the model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def calculate_metrics(preds, labels, probs):
    """Calculate various metrics"""
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Pruned ResNet50 Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pruned model file (.pt)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help='Dataset to test on')
    
    # Dataset-specific paths
    parser.add_argument('--hardfake_csv', type=str,
                        help='Path to hardfake CSV file')
    parser.add_argument('--hardfake_root', type=str,
                        help='Root directory for hardfake dataset')
    
    parser.add_argument('--rvf10k_train_csv', type=str,
                        help='Path to RVF10k train CSV')
    parser.add_argument('--rvf10k_valid_csv', type=str,
                        help='Path to RVF10k valid CSV')
    parser.add_argument('--rvf10k_root', type=str,
                        help='Root directory for RVF10k dataset')
    
    parser.add_argument('--realfake140k_train_csv', type=str,
                        help='Path to 140k train CSV')
    parser.add_argument('--realfake140k_valid_csv', type=str,
                        help='Path to 140k valid CSV')
    parser.add_argument('--realfake140k_test_csv', type=str,
                        help='Path to 140k test CSV')
    parser.add_argument('--realfake140k_root', type=str,
                        help='Root directory for 140k dataset')
    
    parser.add_argument('--realfake200k_train_csv', type=str,
                        help='Path to 200k train CSV')
    parser.add_argument('--realfake200k_val_csv', type=str,
                        help='Path to 200k val CSV')
    parser.add_argument('--realfake200k_test_csv', type=str,
                        help='Path to 200k test CSV')
    parser.add_argument('--realfake200k_root', type=str,
                        help='Root directory for 200k dataset')
    
    parser.add_argument('--realfake190k_root', type=str,
                        help='Root directory for 190k dataset')
    
    parser.add_argument('--realfake330k_root', type=str,
                        help='Root directory for 330k dataset')
    
    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    
    # Output arguments
    parser.add_argument('--save_results', type=str,
                        help='Path to save results (optional)')
    parser.add_argument('--plot_cm', action='store_true',
                        help='Plot confusion matrix')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    print("Model loaded successfully!")
    
    # Prepare dataset arguments
    dataset_kwargs = {
        'dataset_mode': args.dataset,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True if device.type == 'cuda' else False,
        'ddp': False
    }
    
    # Add dataset-specific arguments
    if args.dataset == 'hardfake':
        dataset_kwargs.update({
            'hardfake_csv_file': args.hardfake_csv,
            'hardfake_root_dir': args.hardfake_root
        })
    elif args.dataset == 'rvf10k':
        dataset_kwargs.update({
            'rvf10k_train_csv': args.rvf10k_train_csv,
            'rvf10k_valid_csv': args.rvf10k_valid_csv,
            'rvf10k_root_dir': args.rvf10k_root
        })
    elif args.dataset == '140k':
        dataset_kwargs.update({
            'realfake140k_train_csv': args.realfake140k_train_csv,
            'realfake140k_valid_csv': args.realfake140k_valid_csv,
            'realfake140k_test_csv': args.realfake140k_test_csv,
            'realfake140k_root_dir': args.realfake140k_root
        })
    elif args.dataset == '200k':
        dataset_kwargs.update({
            'realfake200k_train_csv': args.realfake200k_train_csv,
            'realfake200k_val_csv': args.realfake200k_val_csv,
            'realfake200k_test_csv': args.realfake200k_test_csv,
            'realfake200k_root_dir': args.realfake200k_root
        })
    elif args.dataset == '190k':
        dataset_kwargs.update({
            'realfake190k_root_dir': args.realfake190k_root
        })
    elif args.dataset == '330k':
        dataset_kwargs.update({
            'realfake330k_root_dir': args.realfake330k_root
        })
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = Dataset_selector(**dataset_kwargs)
    test_loader = dataset.loader_test
    
    # Test model
    print("Testing model...")
    preds, labels, probs = test_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(preds, labels, probs)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {len(labels)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    print("="*50)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Negatives (Fake): {metrics['confusion_matrix'][0,0]}")
    print(f"False Positives: {metrics['confusion_matrix'][0,1]}")
    print(f"False Negatives: {metrics['confusion_matrix'][1,0]}")
    print(f"True Positives (Real): {metrics['confusion_matrix'][1,1]}")
    
    # Plot confusion matrix if requested
    if args.plot_cm:
        save_path = None
        if args.save_results:
            save_path = args.save_results.replace('.txt', '_cm.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            f.write(f"Test Results for {args.dataset} Dataset\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Test samples: {len(labels)}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['auc']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"True Negatives (Fake): {metrics['confusion_matrix'][0,0]}\n")
            f.write(f"False Positives: {metrics['confusion_matrix'][0,1]}\n")
            f.write(f"False Negatives: {metrics['confusion_matrix'][1,0]}\n")
            f.write(f"True Positives (Real): {metrics['confusion_matrix'][1,1]}\n")
        
        print(f"\nResults saved to: {args.save_results}")

if __name__ == "__main__":
    main()
