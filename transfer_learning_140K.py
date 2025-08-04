import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info
from data.dataset import FaceDataset, Dataset_selector

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal, Bottleneck_pruned, BasicBlock_pruned

def extract_masks_from_checkpoint(checkpoint):
    """استخراج masks از چک‌پوینت sparse model"""
    masks = []
    mask_keys = []
    
    # جمع‌آوری تمام mask_weight ها به ترتیب
    for key in sorted(checkpoint['student'].keys()):
        if key.endswith('mask_weight'):
            mask_weight = checkpoint['student'][key]
            # تبدیل به binary mask
            mask = torch.argmax(mask_weight, dim=1).float().squeeze()
            masks.append(mask)
            mask_keys.append(key)
            print(f"Extracted mask from {key}: shape {mask.shape}, preserved filters: {int(mask.sum())}")
    
    return masks, mask_keys

def create_pruned_state_dict(sparse_checkpoint, masks, mask_keys):
    """تبدیل state_dict sparse به state_dict مناسب برای مدل هرس شده"""
    pruned_state_dict = {}
    sparse_state_dict = sparse_checkpoint['student']
    
    mask_idx = 0
    
    for key, value in sparse_state_dict.items():
        # حذف mask_weight ها و feat layers
        if key.endswith('mask_weight') or key.startswith('feat'):
            continue
        
        # برای conv layers که باید pruned شوند
        if 'conv' in key and 'weight' in key and any(layer in key for layer in ['layer1', 'layer2', 'layer3', 'layer4']):
            # یافتن mask مربوطه
            corresponding_mask_key = key.replace('.weight', '.mask_weight')
            if corresponding_mask_key in mask_keys:
                current_mask_idx = mask_keys.index(corresponding_mask_key)
                current_mask = masks[current_mask_idx]
                
                # اعمال mask روی output channels
                pruned_weight = value[current_mask == 1]
                
                # اگر این conv دوم یا سوم bottleneck است، input channels هم باید pruned شوند
                if mask_idx > 0:
                    # پیدا کردن mask قبلی برای input channels
                    prev_mask = masks[current_mask_idx - 1] if current_mask_idx > 0 else None
                    if prev_mask is not None and 'conv1' not in key:
                        # اعمال mask روی input channels
                        pruned_weight = pruned_weight[:, prev_mask == 1]
                
                pruned_state_dict[key] = pruned_weight
                print(f"Pruned {key}: {value.shape} -> {pruned_weight.shape}")
            else:
                pruned_state_dict[key] = value
        
        # برای batch norm layers که باید pruned شوند
        elif ('bn' in key and any(layer in key for layer in ['layer1', 'layer2', 'layer3', 'layer4'])) or \
             ('downsample' in key and 'weight' in key):
            # یافتن mask مربوطه
            layer_part = key.split('.')[:-1]  # حذف آخرین قسمت (weight/bias/etc)
            possible_conv_key = '.'.join(layer_part + ['conv3', 'mask_weight'])  # برای bottleneck
            if possible_conv_key not in mask_keys:
                possible_conv_key = '.'.join(layer_part + ['conv2', 'mask_weight'])  # برای basic block
            
            if possible_conv_key in mask_keys:
                current_mask_idx = mask_keys.index(possible_conv_key)
                current_mask = masks[current_mask_idx]
                
                # اعمال mask
                pruned_param = value[current_mask == 1]
                pruned_state_dict[key] = pruned_param
                print(f"Pruned BN {key}: {value.shape} -> {pruned_param.shape}")
                
            elif 'downsample' in key:
                # برای downsample layers، از mask آخرین conv همان block استفاده می‌کنیم
                block_key = '.'.join(key.split('.')[:2])  # مثل layer1.0
                possible_mask_key = block_key + '.conv3.mask_weight'
                if possible_mask_key not in mask_keys:
                    possible_mask_key = block_key + '.conv2.mask_weight'
                
                if possible_mask_key in mask_keys:
                    current_mask_idx = mask_keys.index(possible_mask_key)
                    current_mask = masks[current_mask_idx]
                    
                    if 'weight' in key:
                        # برای downsample conv weight
                        pruned_param = value[current_mask == 1]
                    else:
                        # برای downsample bn parameters
                        pruned_param = value[current_mask == 1]
                    
                    pruned_state_dict[key] = pruned_param
                    print(f"Pruned downsample {key}: {value.shape} -> {pruned_param.shape}")
                else:
                    pruned_state_dict[key] = value
            else:
                pruned_state_dict[key] = value
        else:
            # برای سایر layers (conv1, bn1, fc) تغییری نمی‌دهیم
            pruned_state_dict[key] = value
    
    return pruned_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Load weights on pruned ResNet50 model.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k'],
                        help='Dataset to use: hardfake, rvf10k, or 140k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save trained model and outputs')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the sparse model checkpoint')
    parser.add_argument('--img_height', type=int, default=300,
                        help='Height of input images')
    parser.add_argument('--img_width', type=int, default=300,
                        help='Width of input images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # تنظیمات
    dataset_mode = args.dataset_mode
    data_dir = args.data_dir
    teacher_dir = args.teacher_dir
    checkpoint_path = args.checkpoint_path
    img_height = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_height
    img_width = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_width
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # بارگذاری چک‌پوینت
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'student' not in checkpoint:
        raise KeyError("'student' key not found in checkpoint")
    
    print("Checkpoint keys:", list(checkpoint.keys()))
    print("Number of parameters in sparse model:", len(checkpoint['student'].keys()))
    
    # استخراج masks
    print("\n=== Extracting Masks ===")
    masks, mask_keys = extract_masks_from_checkpoint(checkpoint)
    print(f"Total masks extracted: {len(masks)}")
    
    # محاسبه تعداد کل فیلترهای preserved
    total_original = sum([64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512] * 3)  # تقریبی برای bottleneck
    total_preserved = sum([int(mask.sum()) for mask in masks])
    compression_ratio = total_preserved / total_original if total_original > 0 else 0
    
    print(f"Compression ratio: {compression_ratio:.3f} ({total_preserved}/{total_original})")
    
    # ساخت مدل هرس شده
    print("\n=== Creating Pruned Model ===")
    if dataset_mode == 'hardfake':
        pruned_model = ResNet_50_pruned_hardfakevsreal(masks)
    else:
        # برای سایر dataset ها، نیاز به تعریف functions مشابه دارید
        print("Warning: Using hardfake model for other datasets. You may need to define specific functions.")
        pruned_model = ResNet_50_pruned_hardfakevsreal(masks)
    
    # تبدیل state_dict
    print("\n=== Converting State Dict ===")
    pruned_state_dict = create_pruned_state_dict(checkpoint, masks, mask_keys)
    
    # لود کردن وزن‌ها
    print("\n=== Loading Weights ===")
    missing_keys, unexpected_keys = pruned_model.load_state_dict(pruned_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print("Missing keys:", missing_keys[:5], "..." if len(missing_keys) > 5 else "")
    
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys[:5], "..." if len(unexpected_keys) > 5 else "")
    
    # انتقال به device
    pruned_model = pruned_model.to(device)
    pruned_model.eval()
    
    print(f"\n=== Model Successfully Loaded ===")
    print(f"Model moved to: {device}")
    
    # محاسبه پارامترها و FLOPs
    print("\n=== Model Statistics ===")
    
    # شمارش پارامترها
    total_params = sum(p.numel() for p in pruned_model.parameters())
    trainable_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # محاسبه FLOPs
    try:
        flops, params = get_model_complexity_info(
            pruned_model, 
            (3, img_height, img_width), 
            as_strings=True, 
            print_per_layer_stat=False
        )
        print(f'FLOPs: {flops}')
        print(f'Parameters (from ptflops): {params}')
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
    
    # تست ساده
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_height, img_width).to(device)
            output, features = pruned_model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Number of feature maps: {len(features)}")
            print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    # ذخیره مدل هرس شده
    save_path = os.path.join(teacher_dir, 'pruned_model.pth')
    torch.save(pruned_model.state_dict(), save_path)
    print(f"\nPruned model saved to: {save_path}")
    
    # ذخیره اطلاعات masks
    masks_info = {
        'masks': masks,
        'mask_keys': mask_keys,
        'compression_ratio': compression_ratio,
        'total_preserved': total_preserved,
        'total_original': total_original
    }
    masks_save_path = os.path.join(teacher_dir, 'masks_info.pth')
    torch.save(masks_info, masks_save_path)
    print(f"Masks info saved to: {masks_save_path}")

if __name__ == "__main__":
    main()
