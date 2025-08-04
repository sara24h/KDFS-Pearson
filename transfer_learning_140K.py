import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from ptflops import get_model_complexity_info
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# تنظیم دستگاه (GPU یا CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه مورد استفاده: {device}")

# تعریف مدل ResNet50
model = models.resnet50(pretrained=False)  # بدون وزن‌های پیش‌فرض ImageNet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # تنظیم لایه خروجی برای طبقه‌بندی دودویی (real vs. fake)

# مسیر فایل چک‌پوینت
checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'

# لود چک‌پوینت
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
except FileNotFoundError:
    raise FileNotFoundError(f"فایل چک‌پوینت در مسیر {checkpoint_path} یافت نشد!")

# بررسی وجود کلید 'student' و فیلتر کردن کلیدهای غیرضروری
if 'student' in checkpoint:
    state_dict = checkpoint['student']
else:
    state_dict = checkpoint

# فیلتر کردن کلیدهای غیرضروری (مثل mask_weight یا کلیدهای اضافی)
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('mask_weight') and k not in ['feat1.weight', 'feat1.bias', 'feat2.weight', 'feat2.bias', 'feat3.weight', 'feat3.bias', 'feat4.weight', 'feat4.bias']}

# لود state_dict به مدل
missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
print(f"کلیدهای گمشده: {missing_keys}")
print(f"کلیدهای غیرمنتظره: {unexpected_keys}")

# اعمال ماسک‌های هرس (در صورت وجود)
# اگر ماسک‌های هرس در چک‌پوینت وجود دارند، آن‌ها را اعمال کنید
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        mask_key = f"{name}.weight_mask"
        if mask_key in state_dict:
            mask = state_dict[mask_key]
            prune.custom_from_mask(module, name='weight', mask=mask)
            print(f"ماسک هرس برای {name} اعمال شد")

# انتقال مدل به دستگاه
model = model.to(device)

# تنظیم مدل در حالت ارزیابی (برای تست یا استنتاج)
model.eval()

print("مدل دانش‌آموز با موفقیت لود شد!")

# محاسبه تعداد پارامترهای کل و غیرصفر
total_params = sum(p.numel() for p in model.parameters())
non_zero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
print(f"تعداد کل پارامترها: {total_params}")
print(f"تعداد پارامترهای غیرصفر: {non_zero_params}")

# محاسبه FLOPs
flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print(f'FLOPs: {flops}')
print(f'Parameters: {params}')

# بررسی وزن‌های یک لایه نمونه (برای تأیید هرس)
print("وزن‌های conv1 (نمونه):", model.conv1.weight.data[:2])
print(f"تعداد وزن‌های غیرصفر در conv1: {torch.count_nonzero(model.conv1.weight.data).item()}")

# تعریف تبدیل برای تصویر ورودی
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# تابع برای تست مدل روی یک تصویر نمونه
def test_single_image(img_path, model):
    if not os.path.exists(img_path):
        print(f"تصویر در مسیر {img_path} یافت نشد!")
        return
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image).squeeze(1)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        print(f"تصویر: {img_path}")
        print(f"پیش‌بینی: {predicted_label}, احتمال: {prob:.4f}")

# تابع برای ارزیابی مدل روی مجموعه داده آزمایشی
def evaluate_model(test_loader, model, criterion=nn.BCEWithLogitsLoss()):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'زیان آزمایشی: {test_loss:.4f}, دقت آزمایشی: {test_accuracy:.2f}%')

# تست روی یک تصویر نمونه (مسیر را جایگزین کنید)
sample_image_path = '/path/to/sample/image.jpg'  # مسیر تصویر نمونه را وارد کنید
test_single_image(sample_image_path, model)

# تعریف دیتالودر آزمایشی (اختیاری)
try:
    test_dataset = ImageFolder(
        root='/path/to/test/dataset',  # مسیر مجموعه داده آزمایشی را وارد کنید
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    evaluate_model(test_loader, model)
except FileNotFoundError:
    print("مجموعه داده آزمایشی یافت نشد. لطفاً مسیر صحیح را وارد کنید.")

# ذخیره مدل (اختیاری)
save_path = './resnet50_student_model.pth'
torch.save(model.state_dict(), save_path)
print(f"مدل در مسیر {save_path} ذخیره شد!")
