import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import pickle
import os

# Import از فایل‌های موجود شما
from data.dataset import Dataset_selector
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def inspect_checkpoint_structure(checkpoint_path):
    """
    ساختار checkpoint را بررسی می‌کند تا ماسک‌ها را پیدا کند
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=== Checkpoint Structure Analysis ===")
    print(f"Total keys in checkpoint: {len(checkpoint)}")
    
    # نمایش تمام کلیدها
    for key in sorted(checkpoint.keys()):
        if hasattr(checkpoint[key], 'shape'):
            print(f"{key}: {checkpoint[key].shape}")
        else:
            print(f"{key}: {type(checkpoint[key])}")
    
    # جستجو برای ماسک‌ها
    mask_keys = [key for key in checkpoint.keys() if 'mask' in key.lower()]
    print(f"\nFound {len(mask_keys)} potential mask keys:")
    for key in mask_keys:
        print(f"  {key}")
    
    return checkpoint

def extract_masks_from_student_checkpoint(checkpoint):
    """
    ماسک‌ها را از checkpoint مدل student استخراج می‌کند
    """
    masks = []
    
    # بررسی انواع مختلف کلیدهای ممکن برای ماسک‌ها
    possible_mask_patterns = [
        'masks',  # مستقیم
        'pruning_masks',  # الگوی دیگر
        'model.masks',  # اگر در model state قرار دارد
        'student_masks',  # خاص مدل student
    ]
    
    # جستجو برای ماسک‌ها
    found_masks = None
    for pattern in possible_mask_patterns:
        if pattern in checkpoint:
            found_masks = checkpoint[pattern]
            print(f"Found masks with key: {pattern}")
            break
    
    if found_masks is not None:
        if isinstance(found_masks, list):
            masks = found_masks
        elif isinstance(found_masks, dict):
            # اگر ماسک‌ها در dictionary هستند
            mask_keys = sorted([k for k in found_masks.keys() if 'mask' in k])
            masks = [found_masks[k] for k in mask_keys]
    
    # اگر ماسک پیدا نشد، از state_dict جستجو کن
    if not masks:
        print("Searching for individual mask tensors in state_dict...")
        
        # الگوهای مختلف نام‌گذاری برای ماسک‌ها
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        block_counts = [3, 4, 6, 3]
        
        for layer_idx, (layer_name, num_blocks) in enumerate(zip(layer_names, block_counts)):
            for block_idx in range(num_blocks):
                for conv_idx in [1, 2, 3]:
                    # الگوهای مختلف نام‌گذاری
                    possible_keys = [
                        f"{layer_name}.{block_idx}.conv{conv_idx}.mask",
                        f"{layer_name}.{block_idx}.conv{conv_idx}.mask_weight",
                        f"module.{layer_name}.{block_idx}.conv{conv_idx}.mask",
                        f"backbone.{layer_name}.{block_idx}.conv{conv_idx}.mask",
                        f"model.{layer_name}.{block_idx}.conv{conv_idx}.mask",
                    ]
                    
                    mask_found = False
                    for key in possible_keys:
                        if key in checkpoint:
                            mask_tensor = checkpoint[key]
                            if len(mask_tensor.shape) == 4:  # Conv weight mask
                                mask_bool = (mask_tensor.abs().sum(dim=[1,2,3]) > 1e-8)
                            elif len(mask_tensor.shape) == 1:  # Already boolean or 1D
                                mask_bool = mask_tensor.bool()
                            else:
                                mask_bool = mask_tensor.bool()
                            
                            masks.append(mask_bool)
                            print(f"Found mask: {key} -> {mask_bool.sum()}/{len(mask_bool)} filters preserved")
                            mask_found = True
                            break
                    
                    if not mask_found:
                        print(f"Warning: No mask found for {layer_name}.{block_idx}.conv{conv_idx}")
    
    if not masks:
        print("ERROR: No masks found in checkpoint!")
        print("Available keys in checkpoint:")
        for key in sorted(checkpoint.keys()):
            print(f"  {key}")
        return None
    
    print(f"Successfully extracted {len(masks)} masks")
    
    # بررسی تعداد پارامترهای باقی‌مانده
    total_preserved = sum(mask.sum().item() for mask in masks)
    print(f"Total preserved filters across all layers: {total_preserved}")
    
    return masks

def load_student_model_properly(model_path):
    """
    مدل student را با ماسک‌های درست بارگذاری می‌کند
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ابتدا ساختار checkpoint را بررسی کن
    print("Inspecting checkpoint structure...")
    checkpoint = inspect_checkpoint_structure(model_path)
    
    # استخراج ماسک‌ها
    print("\nExtracting masks...")
    masks = extract_masks_from_student_checkpoint(checkpoint)
    
    if masks is None:
        print("CRITICAL ERROR: Could not extract masks from checkpoint!")
        print("The student model requires proper pruning masks to work correctly.")
        return None, None
    
    # ایجاد مدل با ماسک‌های استخراج شده
    print(f"\nCreating pruned model with {len(masks)} masks...")
    model = ResNet_50_pruned_hardfakevsreal(masks)
    
    # بارگذاری وزن‌ها
    try:
        # حذف کلیدهای مربوط به ماسک برای بارگذاری وزن‌ها
        state_dict = {}
        for key, value in checkpoint.items():
            if not any(pattern in key.lower() for pattern in ['mask', 'pruning']):
                # حذف پیشوند module. اگر وجود دارد
                clean_key = key.replace('module.', '').replace('model.', '')
                state_dict[clean_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:  # نمایش 5 کلید اول
                print(f"  {key}")
        
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # نمایش 5 کلید اول
                print(f"  {key}")
        
        print("Model state loaded successfully")
        
    except Exception as e:
        print(f"Error loading model state: {e}")
        return None, None
    
    model.to(device)
    model.eval()
    
    # نمایش آمار مدل
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # بررسی اینکه آیا این واقعاً مدل pruned است
    if total_params > 10_000_000:  # اگر بیش از 10 میلیون پارامتر دارد
        print("WARNING: Model seems to have too many parameters for a pruned student model!")
        print("Expected: ~3 million parameters")
        print("Current: {:.1f} million parameters".format(total_params / 1_000_000))
    else:
        print("✓ Model parameter count looks correct for a pruned student model")
    
    return model, device

def evaluate_model(model, test_loader, device):
    """
    مدل را ارزیابی می‌کند
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            
            # پیش‌بینی
            output, _ = model(data)
            
            # تبدیل به احتمال (برای binary classification)
            prob = torch.sigmoid(output.squeeze())
            pred = (prob > 0.5).float()
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(prob.cpu().numpy())
    
    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def main():
    # مسیرها و تنظیمات
    MODEL_PATH = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
    
    # تنظیمات دیتاست
    dataset_config = {
        'dataset_mode': '140k',
        'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces',
        'eval_batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'ddp': False
    }
    
    print("=== Evaluating Pruned ResNet50 Student Model ===")
    
    # 1. بارگذاری مدل با ماسک‌های درست
    print("Loading student model with proper masks...")
    model, device = load_student_model_properly(MODEL_PATH)
    
    if model is None:
        print("FAILED: Could not load the student model properly!")
        print("\nTroubleshooting suggestions:")
        print("1. Check if the checkpoint contains the pruning masks")
        print("2. Verify the checkpoint was saved correctly during training")
        print("3. Make sure you're using the correct student model checkpoint")
        return
    
    # 2. آماده‌سازی دیتا
    print("\nPreparing test data...")
    try:
        dataset = Dataset_selector(**dataset_config)
        test_loader = dataset.loader_test
        print(f"Test dataset loaded: {len(test_loader)} batches")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 3. ارزیابی مدل
    print("\nEvaluating model...")
    test_results = evaluate_model(model, test_loader, device)
    
    # 4. نمایش نتایج
    print("\n=== Final Results ===")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1-Score: {test_results['f1_score']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    
    # محاسبه compression ratio
    original_resnet50_params = 25_557_032  # تعداد پارامترهای ResNet-50 اصلی
    current_params = sum(p.numel() for p in model.parameters())
    compression_ratio = original_resnet50_params / current_params
    
    print(f"\n=== Compression Statistics ===")
    print(f"Original ResNet-50 parameters: {original_resnet50_params:,}")
    print(f"Pruned model parameters: {current_params:,}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Parameter reduction: {(1 - current_params/original_resnet50_params)*100:.1f}%")

if __name__ == "__main__":
    main()
