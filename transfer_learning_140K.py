import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from ptflops import get_model_complexity_info
import argparse

# Import dataset classes from data.dataset
from data.dataset import FaceDataset, Dataset_selector

# Assuming these are your model definitions
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pruned ResNet50 model on multiple datasets.')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt',
                        help='Path to the student model checkpoint')
    parser.add_argument('--hardfake_csv_file', type=str, default='/kaggle/input/hardfakevsrealfaces/data.csv',
                        help='Path to hardfake dataset CSV')
    parser.add_argument('--hardfake_root_dir', type=str, default='/kaggle/input/hardfakevsrealfaces',
                        help='Root directory for hardfake dataset')
    parser.add_argument('--rvf10k_train_csv', type=str, default='/kaggle/input/rvf10k/train.csv',
                        help='Path to rvf10k train CSV')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='/kaggle/input/rvf10k/valid.csv',
                        help='Path to rvf10k validation CSV')
    parser.add_argument('--rvf10k_root_dir', type=str, default='/kaggle/input/rvf10k',
                        help='Root directory for rvf10k dataset')
    parser.add_argument('--realfake140k_train_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/train.csv',
                        help='Path to 140k train CSV')
    parser.add_argument('--realfake140k_valid_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/valid.csv',
                        help='Path to 140k validation CSV')
    parser.add_argument('--realfake140k_test_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/test.csv',
                        help='Path to 140k test CSV')
    parser.add_argument('--realfake140k_root_dir', type=str, default='/kaggle/input/140k-real-and-fake-faces',
                        help='Root directory for 140k dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loaders')
    return parser.parse_args()

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load the student model checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint['student']

    # Step 2: Extract pruning masks from the sparse model
    student = ResNet_50_sparse_hardfakevsreal()
    student.load_state_dict(state_dict)
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = []
    active_filters = []
    for i, mask_weight in enumerate(mask_weights):
        print(f"Mask {i} original shape: {mask_weight.shape}")
        if mask_weight.dim() > 1:
            mask = torch.argmax(mask_weight, dim=1).squeeze()
            if mask.dim() > 1:
                mask = mask.squeeze()
        else:
            mask = (mask_weight > 0.5).float()
        masks.append(mask)
        num_active = torch.sum(mask == 1).item()
        active_filters.append(num_active)
        print(f"Mask {i} processed shape: {mask.shape}, Active filters: {num_active}")

    # Step 3: Build the pruned model with active filters
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)

    # Step 4: Verify pruned model architecture
    print("\nPruned model architecture:")
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Layer {name}: out_channels={module.out_channels}, in_channels={module.in_channels}")
        elif isinstance(module, nn.BatchNorm2d):
            print(f"Layer {name}: num_features={module.num_features}")

    # Step 5: Filter and trim weights
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('mask_weight') and k not in ['feat1.weight', 'feat1.bias', 'feat2.weight', 'feat2.bias', 'feat3.weight', 'feat3.bias', 'feat4.weight', 'feat4.bias']}
    
    # Keep track of input channels for each layer
    in_channels_list = [64] + active_filters[:-1]  # conv1 has 64 output channels, followed by active filters
    mask_idx = 0
    prev_mask = torch.ones(64).to(device)  # Initial mask for conv1 input (no pruning)
    
    for layer_idx in range(1, 5):
        num_blocks = 3 if layer_idx == 1 else 4 if layer_idx == 2 else 6 if layer_idx == 3 else 3
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):
                if mask_idx >= len(masks):
                    break
                m = masks[mask_idx]
                layer_prefix = f'layer{layer_idx}.{block_idx}.conv{conv_idx}'
                bn_prefix = f'layer{layer_idx}.{block_idx}.bn{conv_idx}'
                
                if f'{layer_prefix}.weight' in filtered_state_dict:
                    weight = filtered_state_dict[f'{layer_prefix}.weight']
                    weight_shape = weight.shape
                    expected_in_channels = in_channels_list[mask_idx]
                    expected_out_channels = active_filters[mask_idx]
                    print(f"Weight shape for {layer_prefix}: {weight_shape}, Mask shape: {m.shape}, Active filters: {expected_out_channels}, Expected in_channels: {expected_in_channels}")
                    if weight_shape[0] == m.shape[0] and weight_shape[1] >= expected_in_channels:
                        # Trim output channels
                        filtered_state_dict[f'{layer_prefix}.weight'] = weight[m == 1]
                        # Trim input channels based on previous layer's active filters
                        filtered_state_dict[f'{layer_prefix}.weight'] = filtered_state_dict[f'{layer_prefix}.weight'][:, prev_mask == 1]
                        if f'{layer_prefix}.bias' in filtered_state_dict:
                            filtered_state_dict[f'{layer_prefix}.bias'] = filtered_state_dict[f'{layer_prefix}.bias'][m == 1]
                    else:
                        print(f"Warning: Shape mismatch for {layer_prefix}. Using original weights.")
                
                if f'{bn_prefix}.weight' in filtered_state_dict:
                    bn_weight = filtered_state_dict[f'{bn_prefix}.weight']
                    bn_weight_shape = bn_weight.shape
                    print(f"BN weight shape for {bn_prefix}: {bn_weight_shape}, Mask shape: {m.shape}")
                    if bn_weight_shape[0] == m.shape[0]:
                        filtered_state_dict[f'{bn_prefix}.weight'] = bn_weight[m == 1]
                        filtered_state_dict[f'{bn_prefix}.bias'] = filtered_state_dict[f'{bn_prefix}.bias'][m == 1]
                        filtered_state_dict[f'{bn_prefix}.running_mean'] = filtered_state_dict[f'{bn_prefix}.running_mean'][m == 1]
                        filtered_state_dict[f'{bn_prefix}.running_var'] = filtered_state_dict[f'{bn_prefix}.running_var'][m == 1]
                    else:
                        print(f"Warning: Shape mismatch for {bn_prefix}. Using original BN weights.")
                
                prev_mask = m  # Update previous mask for the next layer
                mask_idx += 1

    # Handle downsample layers
    downsample_mask_indices = [2, 8, 20, 38]  # Indices of conv3 masks for layer1, layer2, layer3, layer4
    downsample_in_channels = [64, 256, 512, 1024]  # Input channels for downsample layers
    for layer_idx in range(1, 5):
        downsample_prefix = f'layer{layer_idx}.0.downsample'
        if f'{downsample_prefix}.0.weight' in filtered_state_dict:
            mask_idx = downsample_mask_indices[layer_idx-1]
            prev_mask = masks[mask_idx]
            weight = filtered_state_dict[f'{downsample_prefix}.0.weight']
            weight_shape = weight.shape
            expected_in_channels = in_channels_list[mask_idx]
            expected_out_channels = active_filters[mask_idx]
            print(f"Downsample weight shape for {downsample_prefix}.0: {weight_shape}, Mask shape: {prev_mask.shape}, Active filters: {expected_out_channels}, Expected in_channels: {expected_in_channels}")
            if weight_shape[0] == prev_mask.shape[0] and weight_shape[1] >= expected_in_channels:
                filtered_state_dict[f'{downsample_prefix}.0.weight'] = weight[prev_mask == 1]
                filtered_state_dict[f'{downsample_prefix}.0.weight'] = filtered_state_dict[f'{downsample_prefix}.0.weight'][:, in_channels_list[mask_idx-1] if mask_idx > 0 else 64]
                filtered_state_dict[f'{downsample_prefix}.1.weight'] = filtered_state_dict[f'{downsample_prefix}.1.weight'][prev_mask == 1]
                filtered_state_dict[f'{downsample_prefix}.1.bias'] = filtered_state_dict[f'{downsample_prefix}.1.bias'][prev_mask == 1]
                filtered_state_dict[f'{downsample_prefix}.1.running_mean'] = filtered_state_dict[f'{downsample_prefix}.1.running_mean'][prev_mask == 1]
                filtered_state_dict[f'{downsample_prefix}.1.running_var'] = filtered_state_dict[f'{downsample_prefix}.1.running_var'][prev_mask == 1]
            else:
                print(f"Warning: Shape mismatch for {downsample_prefix}. Using original downsample weights.")

    # Step 6: Load weights onto the pruned model
    missing, unexpected = pruned_model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    pruned_model.to(device)

    # Step 7: Compute FLOPs and Parameters
    image_sizes = {"hardfake": 300, "rvf10k": 256, "140k": 256}
    input_size = image_sizes["140k"]  # Assuming checkpoint is trained on 140k
    flops, params = get_model_complexity_info(pruned_model, (3, input_size, input_size), as_strings=True, print_per_layer_stat=True)
    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    # Step 8: Evaluate generalization on datasets
    datasets = {
        "hardfake": Dataset_selector(
            dataset_mode='hardfake',
            hardfake_csv_file=args.hardfake_csv_file,
            hardfake_root_dir=args.hardfake_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False,
        ),
        "rvf10k": Dataset_selector(
            dataset_mode='rvf10k',
            rvf10k_train_csv=args.rvf10k_train_csv,
            rvf10k_valid_csv=args.rvf10k_valid_csv,
            rvf10k_root_dir=args.rvf10k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False,
        ),
        "140k": Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv=args.realfake140k_train_csv,
            realfake140k_valid_csv=args.realfake140k_valid_csv,
            realfake140k_test_csv=args.realfake140k_test_csv,
            realfake140k_root_dir=args.realfake140k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False,
        )
    }

    for name, dataset in datasets.items():
        print(f"\nEvaluating on {name} dataset:")
        val_accuracy = evaluate_model(pruned_model, dataset.loader_val, device)
        test_accuracy = evaluate_model(pruned_model, dataset.loader_test, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
