import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from thop import profile
import argparse
import pandas as pd
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, SoftMaskedConv2d
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from data.dataset import Dataset_selector

# تنظیمات پایه برای فلاپس و پارامترها
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5390.0,
        "140k": 5390.0,
        "190k": 5390.0,
        "200k": 5390.0,
        "330k": 5390.0,
        "125k": 2100.0,
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 23.51,
        "140k": 23.51,
        "190k": 23.51,
        "200k": 23.51,
        "330k": 23.51,
        "125k": 23.51,
    }
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,
    "190k": 256,
    "200k": 256,
    "330k": 256,
    "125k": 160,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate generalization of pruned ResNet50 student model.')
    parser.add_argument('--dataset_modes', type=str, nargs='+', default=['hardfake', 'rvf10k', '140k'],
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k', '125k'],
                        help='Datasets to evaluate generalization on')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='output_dir',
                        help='Directory to save evaluation results')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt',
                        help='Path to the sparse student checkpoint')
    return parser.parse_args()

def extract_masks_from_sparse_model(sparse_model):
    """Extract binary masks from sparse model's mask_weight"""
    print("\n" + "="*60)
    print("EXTRACTING MASKS FROM SPARSE MODEL")
    print("="*60)
    
    masks = []
    mask_weights = [m.mask_weight for m in sparse_model.mask_modules]
    for i, mask_weight in enumerate(mask_weights):
        mask = torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1) > 0
        masks.append(mask)
        print(f"  Mask {i}: {mask.sum().item()} filters preserved out of {len(mask)}")
    
    print(f"\nTotal masks created: {len(masks)}")
    return masks

def adjust_weights_for_pruned_model(sparse_state_dict, pruned_model, masks):
    """Adjust weights from sparse model to match pruned model's reduced dimensions"""
    print("\n" + "="*60)
    print("ADJUSTING WEIGHTS FOR PRUNED MODEL")
    print("="*60)
    
    pruned_state_dict = pruned_model.state_dict()
    layer_patterns = [
        ('layer1', 3, [64, 64, 256]),
        ('layer2', 4, [128, 128, 512]),
        ('layer3', 6, [256, 256, 1024]),
        ('layer4', 3, [512, 512, 2048]),
    ]
    mask_idx = 0
    
    # تنظیم وزن‌های conv1
    if 'conv1.weight' in sparse_state_dict:
        out_mask = masks[0]
        pruned_state_dict['conv1.weight'] = sparse_state_dict['conv1.weight'][out_mask]
        for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
            bn_key = f'bn1.{bn_param}'
            if bn_key in sparse_state_dict:
                pruned_state_dict[bn_key] = sparse_state_dict[bn_key][out_mask]
        mask_idx += 1
    
    # تنظیم وزن‌های لایه‌های Bottleneck
    for layer_name, num_blocks, channels in layer_patterns:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):
                conv_key = f"{layer_name}.{block_idx}.conv{conv_idx}.weight"
                bn_key = f"{layer_name}.{block_idx}.bn{conv_idx}.weight"
                if conv_key in sparse_state_dict and conv_key in pruned_state_dict:
                    out_mask = masks[mask_idx]
                    out_channels = out_mask.sum().item()
                    
                    # تنظیم ورودی‌ها برای conv2 و conv3
                    if conv_idx > 1:
                        prev_mask = masks[mask_idx - 1]
                        in_channels = prev_mask.sum().item()
                    else:
                        in_channels = sparse_state_dict[conv_key].shape[1]
                        prev_mask = slice(None)
                    
                    # انتخاب وزن‌های فعال
                    weight = sparse_state_dict[conv_key]
                    weight = weight[out_mask][:, prev_mask]
                    pruned_state_dict[conv_key] = weight
                    print(f"  Adjusted {conv_key}: {weight.shape}")
                    
                    # تنظیم پارامترهای BatchNorm
                    for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                        bn_key_full = f"{layer_name}.{block_idx}.bn{conv_idx}.{bn_param}"
                        if bn_key_full in sparse_state_dict:
                            pruned_state_dict[bn_key_full] = sparse_state_dict[bn_key_full][out_mask]
                    
                    mask_idx += 1
            
            # تنظیم لایه‌های downsample
            if block_idx == 0:
                downsample_key = f"{layer_name}.{block_idx}.downsample.0.weight"
                if downsample_key in sparse_state_dict:
                    out_mask = masks[mask_idx - 1]  # ماسک conv3
                    in_mask = masks[mask_idx - 3] if layer_name != 'layer1' else slice(None)
                    pruned_state_dict[downsample_key] = sparse_state_dict[downsample_key][out_mask][:, in_mask]
                    for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                        bn_key = f"{layer_name}.{block_idx}.downsample.1.{bn_param}"
                        if bn_key in sparse_state_dict:
                            pruned_state_dict[bn_key] = sparse_state_dict[bn_key][out_mask]
    
    # تنظیم لایه fc
    fc_key = 'fc.weight'
    if fc_key in sparse_state_dict and fc_key in pruned_state_dict:
        in_mask = masks[-1]  # ماسک آخرین لایه conv
        pruned_state_dict[fc_key] = sparse_state_dict[fc_key][:, in_mask]
        pruned_state_dict['fc.bias'] = sparse_state_dict['fc.bias']
        print(f"  Adjusted {fc_key}: {pruned_state_dict[fc_key].shape}")
    
    return pruned_state_dict

def evaluate_model(model, loader, criterion, device):
    """Evaluate model on a given data loader"""
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(images)
                outputs = outputs.squeeze(1)
                loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    return loss / len(loader), 100 * correct / total

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load sparse student model
    sparse_model = ResNet_50_sparse_hardfakevsreal()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=True)
    if 'student' in checkpoint:
        sparse_model.load_state_dict(checkpoint['student'], strict=True)
        print("✓ Sparse model loaded successfully")
    else:
        raise KeyError("'student' key not found in checkpoint")

    # Extract masks
    masks = extract_masks_from_sparse_model(sparse_model)

    # Initialize pruned model
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = pruned_model.fc.in_features
    pruned_model.fc = nn.Linear(num_ftrs, 1)

    # Adjust weights
    pruned_state_dict = adjust_weights_for_pruned_model(checkpoint['student'], pruned_model, masks)
    pruned_model.load_state_dict(pruned_state_dict, strict=True)
    pruned_model = pruned_model.to(device)
    print("✓ Pruned model weights loaded successfully")

    # Calculate FLOPs and parameters
    dataset_type = {'hardfake': 'hardfakevsreal', 'rvf10k': 'rvf10k', '140k': '140k'}.get(args.dataset_modes[0], '140k')
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]]).to(device)
    flops, params = profile(pruned_model, inputs=(input,), verbose=False)
    print(f"Total Parameters: {params / 1e6:.2f} M")
    print(f"Pruned FLOPs: {flops / 1e9:.2f} GMac")

    # Evaluate generalization on multiple datasets
    criterion = nn.BCEWithLogitsLoss()
    results = {}
    
    for dataset_mode in args.dataset_modes:
        print(f"\nEvaluating generalization on dataset: {dataset_mode}")
        
        # Initialize dataset
        if dataset_mode == 'hardfake':
            dataset = Dataset_selector(
                dataset_mode='hardfake',
                hardfake_csv_file=os.path.join(args.data_dir, 'data.csv'),
                hardfake_root_dir=args.data_dir,
                train_batch_size=64,
                eval_batch_size=64,
                num_workers=4,
                pin_memory=True,
                ddp=False
            )
        elif dataset_mode == 'rvf10k':
            dataset = Dataset_selector(
                dataset_mode='rvf10k',
                rvf10k_train_csv=os.path.join(args.data_dir, 'train.csv'),
                rvf10k_valid_csv=os.path.join(args.data_dir, 'valid.csv'),
                rvf10k_root_dir=args.data_dir,
                train_batch_size=64,
                eval_batch_size=64,
                num_workers=4,
                pin_memory=True,
                ddp=False
            )
        elif dataset_mode == '140k':
            dataset = Dataset_selector(
                dataset_mode='140k',
                realfake140k_train_csv=os.path.join(args.data_dir, 'train.csv'),
                realfake140k_valid_csv=os.path.join(args.data_dir, 'valid.csv'),
                realfake140k_test_csv=os.path.join(args.data_dir, 'test.csv'),
                realfake140k_root_dir=args.data_dir,
                train_batch_size=64,
                eval_batch_size=64,
                num_workers=4,
                pin_memory=True,
                ddp=False
            )
        else:
            print(f"Skipping unsupported dataset: {dataset_mode}")
            continue
        
        # Evaluate on validation and test sets
        val_loader = dataset.loader_val
        test_loader = dataset.loader_test
        
        val_loss, val_accuracy = evaluate_model(pruned_model, val_loader, criterion, device)
        test_loss, test_accuracy = evaluate_model(pruned_model, test_loader, criterion, device)
        
        results[dataset_mode] = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        print(f"Dataset: {dataset_mode}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save results
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(args.output_dir, 'generalization_results.csv'))
    print(f"\nGeneralization results saved to {os.path.join(args.output_dir, 'generalization_results.csv')}")

if __name__ == "__main__":
    main()
