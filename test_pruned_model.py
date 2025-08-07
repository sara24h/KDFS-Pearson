import torch
import torch.nn as nn
import argparse
import os
import sys
import time
from data.dataset import Dataset_selector  # Import the dataset class

# کلاس‌های مدل
class Bottleneck_pruned(nn.Module):
    def __init__(self, in_channels, planes1, planes2, planes3, stride=1, downsample=None):
        super(Bottleneck_pruned, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes1)
        self.conv2 = nn.Conv2d(planes1, planes2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes2)
        self.conv3 = nn.Conv2d(planes2, planes3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_pruned(nn.Module):
    def __init__(self):
        super(ResNet_pruned, self).__init__()
        # Layer 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1 (fixed channel dimensions)
        self.layer1 = nn.Sequential(
            Bottleneck_pruned(
                in_channels=64, planes1=11, planes2=7, planes3=57,
                downsample=nn.Sequential(
                    nn.Conv2d(64, 57, kernel_size=1, stride=1, bias=False),  # Fixed: 64->57
                    nn.BatchNorm2d(57)
                )
            ),
            Bottleneck_pruned(in_channels=57, planes1=7, planes2=10, planes3=43),  # Fixed: 57->43
            Bottleneck_pruned(in_channels=43, planes1=10, planes2=8, planes3=41)   # Fixed: 43->41
        )
        
        # Layer 2 (fixed channel dimensions)
        self.layer2 = nn.Sequential(
            Bottleneck_pruned(
                in_channels=41, planes1=28, planes2=19, planes3=96, stride=2,  # Fixed: 41->96
                downsample=nn.Sequential(
                    nn.Conv2d(41, 96, kernel_size=1, stride=2, bias=False),   # Fixed: 41->96
                    nn.BatchNorm2d(96))
            ),
            Bottleneck_pruned(in_channels=96, planes1=25, planes2=22, planes3=94),  # Fixed: 96->94
            Bottleneck_pruned(in_channels=94, planes1=25, planes2=13, planes3=77),   # Fixed: 94->77
            Bottleneck_pruned(in_channels=77, planes1=18, planes2=16, planes3=59)    # Fixed: 77->59
        )
        
        # Layer 3 (fixed channel dimensions)
        self.layer3 = nn.Sequential(
            Bottleneck_pruned(
                in_channels=59, planes1=30, planes2=39, planes3=113, stride=2,  # Fixed: 59->113
                downsample=nn.Sequential(
                    nn.Conv2d(59, 113, kernel_size=1, stride=2, bias=False),   # Fixed: 59->113
                    nn.BatchNorm2d(113))
            ),
            Bottleneck_pruned(in_channels=113, planes1=51, planes2=18, planes3=112),  # Fixed: 113->112
            Bottleneck_pruned(in_channels=112, planes1=46, planes2=15, planes3=71),   # Fixed: 112->71
            Bottleneck_pruned(in_channels=71, planes1=47, planes2=15, planes3=67),    # Fixed: 71->67
            Bottleneck_pruned(in_channels=67, planes1=28, planes2=8, planes3=51),     # Fixed: 67->51
            Bottleneck_pruned(in_channels=51, planes1=30, planes2=11, planes3=44)     # Fixed: 51->44
        )
        
        # Layer 4 (fixed channel dimensions)
        self.layer4 = nn.Sequential(
            Bottleneck_pruned(
                in_channels=44, planes1=35, planes2=50, planes3=66, stride=2,  # Fixed: 44->66
                downsample=nn.Sequential(
                    nn.Conv2d(44, 66, kernel_size=1, stride=2, bias=False),   # Fixed: 44->66
                    nn.BatchNorm2d(66))
            ),
            Bottleneck_pruned(in_channels=66, planes1=86, planes2=12, planes3=64),  # Fixed: 66->64
            Bottleneck_pruned(in_channels=64, planes1=57, planes2=18, planes3=102)   # Fixed: 64->102
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(102, 1)  # Fixed: 2048->102

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

def load_pruned_model(model_path):
    # ایجاد نمونه مدل با معماری دستی
    model = ResNet_pruned()
    
    # بارگذاری state_dict از فایل مدل
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model file: {e}")
        sys.exit(1)
    
    # استخراج state_dict
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # بارگذاری وزن‌ها روی مدل (با امکان عدم تطابق برخی لایه‌ها)
    try:
        model.load_state_dict(state_dict, strict=False)  # Allow partial loading
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        sys.exit(1)
    
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Add dimension for BCE loss
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    return accuracy, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate or test pruned model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pruned model file')
    parser.add_argument('--dataset_mode', type=str, default=None, 
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k', None],
                        help='Dataset to use for evaluation (optional for test mode)')
    parser.add_argument('--test_only', action='store_true', help='Run test mode with random input')
    parser.add_argument('--hardfake_csv_file', type=str, default='', help='CSV file for hardfake dataset')
    parser.add_argument('--hardfake_root_dir', type=str, default='', help='Root directory for hardfake dataset')
    parser.add_argument('--rvf10k_train_csv', type=str, default='', help='Train CSV for RVF10K dataset')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='', help='Validation CSV for RVF10K dataset')
    parser.add_argument('--rvf10k_root_dir', type=str, default='', help='Root directory for RVF10K dataset')
    parser.add_argument('--realfake140k_train_csv', type=str, default='', help='Train CSV for 140K dataset')
    parser.add_argument('--realfake140k_valid_csv', type=str, default='', help='Validation CSV for 140K dataset')
    parser.add_argument('--realfake140k_test_csv', type=str, default='', help='Test CSV for 140K dataset')
    parser.add_argument('--realfake140k_root_dir', type=str, default='', help='Root directory for 140K dataset')
    parser.add_argument('--realfake200k_train_csv', type=str, default='', help='Train CSV for 200K dataset')
    parser.add_argument('--realfake200k_val_csv', type=str, default='', help='Validation CSV for 200K dataset')
    parser.add_argument('--realfake200k_test_csv', type=str, default='', help='Test CSV for 200K dataset')
    parser.add_argument('--realfake200k_root_dir', type=str, default='', help='Root directory for 200K dataset')
    parser.add_argument('--realfake190k_root_dir', type=str, default='', help='Root directory for 190K dataset')
    parser.add_argument('--realfake330k_root_dir', type=str, default='', help='Root directory for 330K dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_pruned_model(args.model_path)
    model = model.to(device)

    if args.test_only:
        input_tensor = torch.randn(2, 3, 256, 256).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        print("=" * 50)
        print("مدل با موفقیت لود شد و تست گذر جلو انجام شد!")
        print(f"دستگاه: {device}")
        print(f"شکل ورودی: {input_tensor.shape}")
        print(f"شکل خروجی: {output.shape}")
        print(f"مقدار خروجی نمونه:\n{output.cpu().numpy()}")
        print("=" * 50)
    else:
        if not args.dataset_mode:
            print("Error: dataset_mode is required for evaluation")
            sys.exit(1)
        print(f"\nCreating {args.dataset_mode} dataset loader...")
        start_time = time.time()
        dataset_args = {
            'dataset_mode': args.dataset_mode,
            'train_batch_size': args.batch_size,
            'eval_batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True,
            'ddp': False,
        }
        if args.dataset_mode == 'hardfake':
            dataset_args['hardfake_csv_file'] = args.hardfake_csv_file
            dataset_args['hardfake_root_dir'] = args.hardfake_root_dir
        elif args.dataset_mode == 'rvf10k':
            dataset_args['rvf10k_train_csv'] = args.rvf10k_train_csv
            dataset_args['rvf10k_valid_csv'] = args.rvf10k_valid_csv
            dataset_args['rvf10k_root_dir'] = args.rvf10k_root_dir
        elif args.dataset_mode == '140k':
            dataset_args['realfake140k_train_csv'] = args.realfake140k_train_csv
            dataset_args['realfake140k_valid_csv'] = args.realfake140k_valid_csv
            dataset_args['realfake140k_test_csv'] = args.realfake140k_test_csv
            dataset_args['realfake140k_root_dir'] = args.realfake140k_root_dir
        elif args.dataset_mode == '200k':
            dataset_args['realfake200k_train_csv'] = args.realfake200k_train_csv
            dataset_args['realfake200k_val_csv'] = args.realfake200k_val_csv
            dataset_args['realfake200k_test_csv'] = args.realfake200k_test_csv
            dataset_args['realfake200k_root_dir'] = args.realfake200k_root_dir
        elif args.dataset_mode == '190k':
            dataset_args['realfake190k_root_dir'] = args.realfake190k_root_dir
        elif args.dataset_mode == '330k':
            dataset_args['realfake330k_root_dir'] = args.realfake330k_root_dir
        
        try:
            dataset_selector = Dataset_selector(**dataset_args)
            test_loader = dataset_selector.loader_test
            print(f"Test dataset loaded successfully! Size: {len(test_loader.dataset)} samples")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

        print("\nEvaluating model...")
        start_time = time.time()
        accuracy, avg_loss = evaluate_model(model, test_loader, device)
        print("\n" + "=" * 50)
        print(f"Evaluation Results on {args.dataset_mode} dataset:")
        print(f"- Test Loss: {avg_loss:.4f}")
        print(f"- Test Accuracy: {accuracy * 100:.2f}%")
        print(f"- Evaluation Time: {time.time() - start_time:.2f} seconds")
        print("=" * 50)

if __name__ == "__main__":
    main()
