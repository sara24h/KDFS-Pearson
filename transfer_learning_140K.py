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
# یا اگر از student model استفاده می‌کنید:
# from model.student.ResNet_sparse import ResNet_50_pruned_hardfakevsreal

def extract_masks_from_checkpoint(checkpoint):
    """
    ماسک‌ها را از checkpoint استخراج می‌کند
    """
    masks = []
    
    # ترتیب layers در ResNet50: layer1, layer2, layer3, layer4
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    block_counts = [3, 4, 6, 3]  # تعداد blocks در هر layer
    
    for layer_idx, (layer_name, num_blocks) in enumerate(zip(layer_names, block_counts)):
        for block_idx in range(num_blocks):
            # هر Bottleneck block دارای 3 conv layer است
            for conv_idx in [1, 2, 3]:  # conv1, conv2, conv3
                mask_key = f"{layer_name}.{block_idx}.conv{conv_idx}.mask_weight"
                
                if mask_key in checkpoint:
                    mask_tensor = checkpoint[mask_key]
                    # تبدیل mask tensor به boolean mask
                    # معمولاً mask_weight برای فیلترهای فعال 1 و برای غیرفعال 0 دارد
                    mask_bool = (mask_tensor.abs().sum(dim=[1,2,3]) > 1e-8)  # فیلترهای غیرصفر
                    masks.append(mask_bool)
                    print(f"Extracted mask for {mask_key}: {mask_bool.sum()}/{len(mask_bool)} filters preserved")
                else:
                    print(f"Warning: {mask_key} not found in checkpoint")
                    # ایجاد mask پیش‌فرض (همه فیلترها فعال)
                    if conv_idx == 1:
                        num_filters = [64, 128, 256, 512][layer_idx]
                    elif conv_idx == 2:
                        num_filters = [64, 128, 256, 512][layer_idx]
                    else:  # conv3
                        num_filters = [64, 128, 256, 512][layer_idx] * 4
                    
                    mask_bool = torch.ones(num_filters, dtype=torch.bool)
                    masks.append(mask_bool)
    
    print(f"Total masks extracted: {len(masks)}")
    return masks

def create_dummy_masks_for_resnet50():
    """
    ماسک‌های dummy برای ResNet50 ایجاد می‌کند (همه فیلترها فعال)
    """
    masks = []
    block_configs = [3, 4, 6, 3]  # تعداد blocks در هر layer
    channels = [64, 128, 256, 512]  # تعداد کانال‌های خروجی هر layer
    
    for layer_idx, (num_blocks, base_channels) in enumerate(zip(block_configs, channels)):
        for block_idx in range(num_blocks):
            # هر Bottleneck block دارای 3 conv layer است
            mask1 = torch.ones(base_channels, dtype=torch.bool)  # Conv1
            mask2 = torch.ones(base_channels, dtype=torch.bool)  # Conv2
            mask3 = torch.ones(base_channels * 4, dtype=torch.bool)  # Conv3
            
            masks.extend([mask1, mask2, mask3])
    
    print(f"Created {len(masks)} dummy masks (all filters preserved)")
    return masks

def load_pruned_model(model_path, masks_path=None, use_dummy_masks=True):
    """
    مدل pruned شده را بارگذاری می‌کند
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # بارگذاری checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print("Checkpoint loaded successfully")
        print(f"Checkpoint contains {len(checkpoint)} keys")
        
        # استخراج masks از checkpoint
        masks = None
        if 'masks' in checkpoint:
            masks = checkpoint['masks']
            print("Masks found in checkpoint")
        elif any('mask_weight' in key for key in checkpoint.keys()):
            print("Found mask_weight tensors in checkpoint, extracting masks...")
            masks = extract_masks_from_checkpoint(checkpoint)
        elif masks_path and os.path.exists(masks_path):
            with open(masks_path, 'rb') as f:
                masks = pickle.load(f)
            print("Masks loaded from separate file")
        elif use_dummy_masks:
            print("Warning: No masks found. Creating dummy masks (all filters preserved)")
            masks = create_dummy_masks_for_resnet50()
        else:
            print("Error: No masks found and dummy masks disabled")
            return None, None
        
        # ایجاد مدل
        model = ResNet_50_pruned_hardfakevsreal(masks)
        
        # حذف mask_weight keys از checkpoint برای بارگذاری وزن‌ها
        state_dict = {}
        for key, value in checkpoint.items():
            if 'mask_weight' not in key:
                state_dict[key] = value
        
        # بارگذاری وزن‌ها
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            print("Continuing with partially loaded model...")
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def evaluate_model(model, test_loader, device):
    """
    مدل را ارزیابی می‌کند و معیارهای مختلف generalization محاسبه می‌کند
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
        auc = 0.0  # در صورت مشکل در محاسبه AUC
    
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

def calculate_generalization_metrics(train_acc, test_results):
    """
    معیارهای generalization را محاسبه می‌کند
    """
    test_acc = test_results['accuracy']
    
    # Generalization Gap
    generalization_gap = train_acc - test_acc
    
    # Generalization Ratio
    if train_acc > 0:
        generalization_ratio = test_acc / train_acc
    else:
        generalization_ratio = 0
    
    return {
        'generalization_gap': generalization_gap,
        'generalization_ratio': generalization_ratio,
        'overfitting_indicator': 'High' if generalization_gap > 0.1 else 'Low'
    }

def main():
    # مسیرها و تنظیمات
    MODEL_PATH = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
    
    # تنظیمات دیتاست - برای دیتاست 140k
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
    
    # 1. بارگذاری مدل
    print("Loading pruned model...")
    model, device = load_pruned_model(MODEL_PATH)
    
    if model is None:
        print("Failed to load model. Please check the model path and masks.")
        return
    
    # 2. آماده‌سازی دیتا
    print("Preparing test data...")
    try:
        # ایجاد دیتاست
        dataset = Dataset_selector(**dataset_config)
        test_loader = dataset.loader_test
        
        print(f"Test dataset loaded successfully")
        print(f"Test loader batches: {len(test_loader)}")
        
        # تست یک batch
        sample_batch = next(iter(test_loader))
        print(f"Sample batch shape: {sample_batch[0].shape}")
        print(f"Sample labels: {sample_batch[1][:5]}")  # نمایش 5 label اول
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset paths are correct and files exist.")
        return
    
    # 3. ارزیابی مدل
    print("Evaluating model on test set...")
    test_results = evaluate_model(model, test_loader, device)
    
    # 4. نمایش نتایج
    print("\n=== Test Results ===")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    print(f"AUC: {test_results['auc']:.4f}")
    
    # 5. محاسبه معیارهای generalization
    # اگر accuracy روی train set دارید، این خط را فعال کنید
    train_accuracy = 0.95  # مقدار واقعی accuracy روی train set را وارد کنید
    gen_metrics = calculate_generalization_metrics(train_accuracy, test_results)
    print(f"\n=== Generalization Metrics ===")
    print(f"Generalization Gap: {gen_metrics['generalization_gap']:.4f}")
    print(f"Generalization Ratio: {gen_metrics['generalization_ratio']:.4f}")
    print(f"Overfitting Indicator: {gen_metrics['overfitting_indicator']}")
    
    # 6. ارزیابی روی validation set نیز
    print("\n=== Validation Set Evaluation ===")
    val_results = evaluate_model(model, dataset.loader_val, device)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation F1-Score: {val_results['f1_score']:.4f}")
    print(f"Validation AUC: {val_results['auc']:.4f}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
