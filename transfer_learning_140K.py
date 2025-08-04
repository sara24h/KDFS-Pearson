import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from thop import profile
import argparse
import os
from PIL import Image
import sys

# لود معماری‌ها
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal 

# مقادیر پایه FLOPs و پارامترها
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

# تابع اصلاح‌شده برای محاسبه تعداد فیلترهای فعال
def get_preserved_filter_num(mask):
    # اطمینان از اینکه ماسک باینری است (فقط ۰ و ۱)
    mask = (mask > 0).float()  # تبدیل مقادیر منفی یا غیرباینری به ۰ و ۱
    num_filters = int(mask.sum())
    if num_filters <= 0:
        print(f"هشدار: تعداد فیلترهای فعال برای ماسک برابر {num_filters} است. تنظیم به 1.")
        num_filters = 1  # حداقل یک فیلتر برای جلوگیری از خطا
    return num_filters

# تابع برای نگاشت وزن‌های مدل دانش‌آموز به مدل هرس‌شده
def map_weights_to_pruned_model(sparse_state_dict, pruned_model, masks):
    pruned_state_dict = {}
    mask_idx = 0
    for name, param in pruned_model.named_parameters():
        sparse_name = name.replace('module.', '')  # حذف پیشوند module. در صورت وجود
        if sparse_name in sparse_state_dict:
            if 'conv' in sparse_name and 'weight' in sparse_name:
                # نگاشت وزن‌ها با توجه به ماسک‌ها
                mask = masks[mask_idx]
                active_filters = (mask > 0).float()
                if len(mask.shape) > 1:
                    active_filters = active_filters.squeeze()
                active_indices = torch.nonzero(active_filters).squeeze()
                pruned_weight = sparse_state_dict[sparse_name][active_indices]
                if pruned_weight.shape == param.shape:
                    pruned_state_dict[name] = pruned_weight
                else:
                    print(f"هشدار: ناسازگاری شکل در {name}: انتظار {param.shape}، دریافت {pruned_weight.shape}")
            else:
                # برای لایه‌های غیرکانولوشنی (مثل bn یا fc)
                if sparse_state_dict[sparse_name].shape == param.shape:
                    pruned_state_dict[name] = sparse_state_dict[sparse_name]
                else:
                    print(f"هشدار: ناسازگاری شکل در {name}: انتظار {param.shape}، دریافت {sparse_state_dict[sparse_name].shape}")
            if 'conv3.weight' in sparse_name or 'conv2.weight' in sparse_name:
                mask_idx += 1
    return pruned_state_dict

# تابع برای محاسبه FLOPs و پارامترها
def get_flops_and_params(args, masks, device):
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k",
        "200k": "200k",
        "330k": "330k",
        "125k": "125k"
    }[args.dataset_mode]

    # تعریف مدل هرس‌شده
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks).to(device)

    # لود چک‌پوینت دانش‌آموز
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location=device, weights_only=True)
    sparse_state_dict = ckpt_student["student"]

    # نگاشت وزن‌ها به مدل هرس‌شده
    pruned_state_dict = map_weights_to_pruned_model(sparse_state_dict, pruned_model, masks)
    missing_keys, unexpected_keys = pruned_model.load_state_dict(pruned_state_dict, strict=False)
    print(f"کلیدهای گمشده: {missing_keys}")
    print(f"کلیدهای غیرمنتظره: {unexpected_keys}")

    # محاسبه FLOPs و پارامترها
    input_size = image_sizes[dataset_type]
    input = torch.rand([1, 3, input_size, input_size]).to(device)
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    Flops_baseline = Flops_baselines["ResNet_50"][dataset_type]
    Params_baseline = Params_baselines["ResNet_50"][dataset_type]
    Flops_reduction = ((Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0)
    Params_reduction = ((Params_baseline - Params / (10**6)) / Params_baseline * 100.0)

    # محاسبه تعداد پارامترهای غیرصفر
    non_zero_params = sum(torch.count_nonzero(p).item() for p in pruned_model.parameters() if p.requires_grad)
    print(f"تعداد پارامترهای غیرصفر: {non_zero_params}")

    return Flops_baseline, Flops / (10**6), Flops_reduction, Params_baseline, Params / (10**6), Params_reduction

# تابع برای ارزیابی تعمیم‌پذیری
def evaluate_model(test_loader, model, criterion=nn.BCEWithLogitsLoss(), device='cuda'):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs, _ = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'زیان آزمایشی: {test_loss:.4f}, دقت آزمایشی: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

# تابع برای تست تصویر نمونه
def test_single_image(img_path, model, transform, device='cuda'):
    if not os.path.exists(img_path):
        print(f"تصویر در مسیر {img_path} یافت نشد!")
        return
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(image)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        print(f"تصویر: {img_path}")
        print(f"پیش‌بینی: {predicted_label}, احتمال: {prob:.4f}")

# تابع برای پردازش آرگومان‌ها
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="rvf10k",
        choices=("hardfake", "rvf10k", "140k", "190k", "200k", "330k", "125k"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default="/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt",
        help="The path where to load the sparsed student ckpt",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/kaggle/input/rvf10k/test",
        help="Path to the test dataset",
    )
    return parser.parse_args(args=[])  # برای محیط‌هایی مثل Jupyter که آرگومان‌ها از خط فرمان دریافت نمی‌شوند

# تابع اصلی
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"دستگاه مورد استفاده: {device}")

    # لود مدل دانش‌آموز برای استخراج ماسک‌ها
    student = ResNet_50_sparse_hardfakevsreal().to(device)
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location=device, weights_only=True)
    student.load_state_dict(ckpt_student["student"])

    # استخراج ماسک‌ها
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [(mask_weight > 0).float() for mask_weight in mask_weights]  # تبدیل به باینری
    print(f"ماسک‌های هرس یافت‌شده: {[f'layer{i//3+1}.{i%3}.conv{(i%3)+1}.mask_weight' for i in range(len(masks))]}")
    for i, mask in enumerate(masks):
        print(f"ماسک layer{i//3+1}.{i%3}.conv{(i%3)+1}.mask_weight: تعداد فیلترهای فعال = {int(mask.sum())}")

    # محاسبه FLOPs و پارامترها
    Flops_baseline, Flops, Flops_reduction, Params_baseline, Params, Params_reduction = get_flops_and_params(args, masks, device)
    print(f"\nنتایج برای دیتاست: {args.dataset_mode}")
    print(f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%")
    print(f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%")

    # تعریف مدل هرس‌شده برای ارزیابی
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks).to(device)
    pruned_state_dict = map_weights_to_pruned_model(ckpt_student["student"], pruned_model, masks)
    missing_keys, unexpected_keys = pruned_model.load_state_dict(pruned_state_dict, strict=False)
    print(f"کلیدهای گمشده: {missing_keys}")
    print(f"کلیدهای غیرمنتظره: {unexpected_keys}")

    # تست تصویر نمونه
    sample_image_path = input(f"لطفاً مسیر تصویر نمونه را وارد کنید (مثلاً /kaggle/input/{args.dataset_mode}/test/real/image.jpg): ")
    test_single_image(sample_image_path, pruned_model, transform, device)

    # ارزیابی مدل روی دیتاست آزمایشی
    test_dataset_path = args.test_dataset_path
    if os.path.exists(test_dataset_path):
        test_dataset = ImageFolder(root=test_dataset_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        evaluate_model(test_loader, pruned_model, device=device)
    else:
        print(f"دیتاست آزمایشی در مسیر {test_dataset_path} یافت نشد!")

    # ذخیره مدل
    save_path = f'./resnet50_pruned_student_{args.dataset_mode}.pth'
    torch.save(pruned_model.state_dict(), save_path)
    print(f"مدل در مسیر {save_path} ذخیره شد!")

if __name__ == "__main__":
    main()
