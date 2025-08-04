import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ptflops import get_model_complexity_info
import os
from PIL import Image
import sys


# لود معماری هرس‌شده
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه مورد استفاده: {device}")

# لود چک‌پوینت
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
except FileNotFoundError:
    raise FileNotFoundError(f"فایل چک‌پوینت در مسیر {checkpoint_path} یافت نشد!")

# استخراج state_dict و ماسک‌ها
if 'student' in checkpoint:
    state_dict = checkpoint['student']
else:
    state_dict = checkpoint

# استخراج ماسک‌های هرس
mask_keys = [k for k in state_dict.keys() if k.endswith('mask_weight')]
masks = [state_dict[k] for k in mask_keys]
print(f"ماسک‌های هرس یافت‌شده: {mask_keys}")

# بررسی تعداد فیلترهای فعال در هر ماسک
for i, (key, mask) in enumerate(zip(mask_keys, masks)):
    print(f"ماسک {key}: تعداد فیلترهای فعال = {int(mask.sum())}")

# تعریف مدل هرس‌شده
model = ResNet_50_pruned_hardfakevsreal(masks=masks).to(device)

# فیلتر کردن کلیدهای غیرمرتبط
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('mask_weight') and k not in ['feat1.weight', 'feat1.bias', 'feat2.weight', 'feat2.bias', 'feat3.weight', 'feat3.bias', 'feat4.weight', 'feat4.bias']}

# لود وزن‌ها به مدل
missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
print(f"کلیدهای گمشده: {missing_keys}")
print(f"کلیدهای غیرمنتظره: {unexpected_keys}")

# تنظیم مدل در حالت ارزیابی
model.eval()

# محاسبه تعداد پارامترها و FLOPs
total_params = sum(p.numel() for p in model.parameters())
non_zero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
print(f"تعداد کل پارامترها: {total_params}")
print(f"تعداد پارامترهای غیرصفر: {non_zero_params}")

flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print(f'FLOPs: {flops}')
print(f'Parameters: {params}')

# بررسی وزن‌های یک لایه نمونه
print("وزن‌های conv1 (نمونه):", model.conv1.weight.data[:2])
print(f"تعداد وزن‌های غیرصفر در conv1: {torch.count_nonzero(model.conv1.weight.data).item()}")

# تعریف تبدیل برای تصویر ورودی
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# تابع برای تست مدل روی تصویر نمونه
def test_single_image(img_path, model):
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

# تابع برای ارزیابی مدل روی دیتاست آزمایشی
def evaluate_model(test_loader, model, criterion=nn.BCEWithLogitsLoss()):
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

# ورودی مسیر تصویر و دیتاست
sample_image_path = input("لطفاً مسیر تصویر نمونه را وارد کنید (مثلاً /kaggle/input/rvf10k/test/real/image.jpg): ")
test_dataset_path = input("لطفاً مسیر دیتاست آزمایشی را وارد کنید (مثلاً /kaggle/input/rvf10k/test): ")

# تست تصویر نمونه
test_single_image(sample_image_path, model)

# تعریف و ارزیابی دیتالودر آزمایشی
if os.path.exists(test_dataset_path):
    test_dataset = ImageFolder(root=test_dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    evaluate_model(test_loader, model)
else:
    print(f"دیتاست آزمایشی در مسیر {test_dataset_path} یافت نشد!")

# ذخیره مدل
save_path = './resnet50_pruned_student_model.pth'
torch.save(model.state_dict(), save_path)
print(f"مدل در مسیر {save_path} ذخیره شد!")
