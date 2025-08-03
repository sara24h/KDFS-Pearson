import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from thop import profile
import torchvision.transforms as transforms

# ایمپورت از فایل‌های مدل و دیتاست
from model.student.layer import SoftMaskedConv2d
from model.student.ResNet_sparse import ResNet_50_sparse_140k
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from data.dataset import FaceDataset, Dataset_selector

# تابع ارزیابی مدل
def evaluate_model(dataset_mode):
    # تنظیم دستگاه
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ایجاد مدل sparse
    sparse_model = ResNet_50_sparse_140k(
        gumbel_start_temperature=2.0,
        gumbel_end_temperature=0.5,
        num_epochs=200
    )

    # لود فایل وزن‌ها
    checkpoint_path = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            sparse_model.load_state_dict(state_dict, strict=True)
            print("وزن‌ها از کلید 'student' با موفقیت لود شدند.")
        else:
            sparse_model.load_state_dict(checkpoint, strict=True)
            print("وزن‌ها مستقیماً لود شدند.")
    except Exception as e:
        print(f"خطا در لود وزن‌ها: {e}")
        checkpoint_keys = set(checkpoint['student'].keys() if 'student' in checkpoint else checkpoint.keys())
        model_keys = set(sparse_model.state_dict().keys())
        print("کلیدهای گمشده:", model_keys - checkpoint_keys)
        print("کلیدهای غیرمنتظره:", checkpoint_keys - model_keys)
        return None

    # تنظیم حالت ticket برای اعمال ماسک‌ها
    sparse_model.ticket = True
    sparse_model.eval()

    # تنظیم ابعاد مکانی برای دیتاست‌های 256x256
    image_size = 256
    conv1_h = (image_size - 7 + 2 * 3) // 2 + 1  # = 128
    maxpool_h = (conv1_h - 3 + 2 * 1) // 2 + 1  # ≈ 64

    # استخراج ماسک‌ها
    masks = []
    for module in sparse_model.modules():
        if isinstance(module, SoftMaskedConv2d):
            mask = module.mask
            if mask is not None:
                # کپی تنسور با clone().detach() و انتقال به دستگاه
                mask = mask.clone().detach().to(device)
                # گسترش ماسک به ابعاد [1, C, 64, 64]
                mask = mask.expand(1, -1, maxpool_h, maxpool_h)  # [1, C, 64, 64]
                masks.append(mask)
            else:
                print(f"هشدار: ماسک برای لایه {module} None است")
                masks.append(None)
    print(f"تعداد ماسک‌ها: {len(masks)}")

    # ایجاد مدل هرس‌شده
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    pruned_model = pruned_model.to(device)
    pruned_model.eval()

    # تابع کمکی برای تنظیم تعداد کانال‌های downsample
    def adjust_downsample(pruned_model, masks):
        def get_preserved_filter_num(mask):
            return int(mask.sum(dim=(1, 2, 3)))  # جمع تعداد فیلترهای حفظ‌شده

        mask_idx = 0
        for layer_name, layer in pruned_model.named_children():
            if layer_name.startswith('layer'):
                for block_idx, block in enumerate(layer):
                    if isinstance(block, Bottleneck_pruned):
                        preserved_filter_num3 = get_preserved_filter_num(masks[mask_idx + 2])
                        if block.downsample:
                            downsample_conv = block.downsample[0]
                            downsample_bn = block.downsample[1]
                            # ایجاد downsample جدید با تعداد کانال‌های مناسب
                            new_conv = nn.Conv2d(
                                downsample_conv.in_channels,
                                preserved_filter_num3,
                                kernel_size=1,
                                stride=downsample_conv.stride,
                                bias=False
                            ).to(device)
                            new_bn = nn.BatchNorm2d(preserved_filter_num3).to(device)
                            # کپی وزن‌های مرتبط
                            with torch.no_grad():
                                min_channels = min(preserved_filter_num3, downsample_conv.out_channels)
                                new_conv.weight[:min_channels] = downsample_conv.weight[:min_channels]
                                new_bn.weight[:min_channels] = downsample_bn.weight[:min_channels]
                                new_bn.bias[:min_channels] = downsample_bn.bias[:min_channels]
                                new_bn.running_mean[:min_channels] = downsample_bn.running_mean[:min_channels]
                                new_bn.running_var[:min_channels] = downsample_bn.running_var[:min_channels]
                            block.downsample = nn.Sequential(new_conv, new_bn)
                        mask_idx += 3  # هر بلوک Bottleneck سه ماسک دارد

    # تنظیم downsample
    adjust_downsample(pruned_model, masks)

    # انتقال وزن‌های هرس‌شده از مدل sparse به مدل pruned
    pruned_state_dict = pruned_model.state_dict()
    sparse_state_dict = sparse_model.state_dict()
    for name, param in sparse_state_dict.items():
        if name in pruned_state_dict:
            pruned_state_dict[name].copy_(param)
    pruned_model.load_state_dict(pruned_state_dict)

    # تنظیمات دیتاست‌ها
    dataset_configs = {
        'rvf10k': {
            'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
            'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
            'rvf10k_root_dir': '/kaggle/input/rvf10k',
        },
        '200k': {
            'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
            'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
            'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
            'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
        },
        '190k': {
            'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
        },
        '330k': {
            'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
        },
    }

    # بررسی وجود دیتاست
    if dataset_mode not in dataset_configs:
        print(f"خطا: dataset_mode باید یکی از {list(dataset_configs.keys())} باشد")
        return None

    # بارگذاری دیتاست تست
    print(f"\nبارگذاری دیتاست تست {dataset_mode}...")
    try:
        dataset = Dataset_selector(
            dataset_mode=dataset_mode,
            rvf10k_train_csv=dataset_configs[dataset_mode].get('rvf10k_train_csv'),
            rvf10k_valid_csv=dataset_configs[dataset_mode].get('rvf10k_valid_csv'),
            rvf10k_root_dir=dataset_configs[dataset_mode].get('rvf10k_root_dir'),
            realfake200k_train_csv=dataset_configs[dataset_mode].get('realfake200k_train_csv'),
            realfake200k_val_csv=dataset_configs[dataset_mode].get('realfake200k_val_csv'),
            realfake200k_test_csv=dataset_configs[dataset_mode].get('realfake200k_test_csv'),
            realfake200k_root_dir=dataset_configs[dataset_mode].get('realfake200k_root_dir'),
            realfake190k_root_dir=dataset_configs[dataset_mode].get('realfake190k_root_dir'),
            realfake330k_root_dir=dataset_configs[dataset_mode].get('realfake330k_root_dir'),
            eval_batch_size=64,
            num_workers=4,
            pin_memory=True,
            ddp=False,
        )
    except Exception as e:
        print(f"خطا در بارگذاری دیتاست {dataset_mode}: {e}")
        return None

    # ارزیابی مدل فقط با loader_test
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataset.loader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = pruned_model(inputs)  # استفاده از مدل هرس‌شده
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    # محاسبه FLOPs و تعداد پارامترها
    input_size = (1, 3, 256, 256)  # برای دیتاست‌های غیر از hardfake
    flops, params = profile(pruned_model, inputs=(torch.randn(input_size).to(device),), verbose=False)

    # چاپ نتایج
    print(f"\nنتایج ارزیابی روی دیتاست تست {dataset_mode}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")
    print(f"تعداد پارامترها: {params / 1e6:.2f} M")

    # ذخیره نتایج
    with open(f'evaluation_results_{dataset_mode}.txt', 'w') as f:
        f.write(f"Dataset: {dataset_mode}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"FLOPs: {flops / 1e6:.2f} MFLOPs\n")
        f.write(f"تعداد پارامترها: {params / 1e6:.2f} M\n")

    return {
        'dataset': dataset_mode,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flops': flops / 1e6,
        'params': params / 1e6
    }

if __name__ == "__main__":
    # لیست دیتاست‌ها (بدون hardfake)
    datasets = ['rvf10k', '190k', '200k', '330k']

    # ذخیره نتایج همه دیتاست‌ها
    all_results = []

    # ارزیابی روی هر دیتاست
    for dataset_mode in datasets:
        print(f"\n=== شروع ارزیابی روی دیتاست {dataset_mode} ===")
        result = evaluate_model(dataset_mode)
        if result:
            all_results.append(result)
        print(f"=== ارزیابی روی دیتاست {dataset_mode} به پایان رسید ===")

    # چاپ خلاصه نتایج
    print("\nخلاصه نتایج ارزیابی روی همه دیتاست‌ها:")
    for result in all_results:
        print(f"\nDataset: {result['dataset']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1-Score: {result['f1']:.4f}")
        print(f"FLOPs: {result['flops']:.2f} MFLOPs")
        print(f"تعداد پارامترها: {result['params']:.2f} M")

    # ذخیره خلاصه نتایج در فایل
    with open('evaluation_results_summary.txt', 'w') as f:
        f.write("خلاصه نتایج ارزیابی روی همه دیتاست‌ها:\n")
        for result in all_results:
            f.write(f"\nDataset: {result['dataset']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1-Score: {result['f1']:.4f}\n")
            f.write(f"FLOPs: {result['flops']:.2f} MFLOPs\n")
            f.write(f"تعداد پارامترها: {result['params']:.2f} M\n")
