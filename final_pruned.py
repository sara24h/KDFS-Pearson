import torch
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k

# مسیر فایل مدل اصلی
model_path = "/kaggle/input/kdfs-5-mordad-330k-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_best.pt"

# مسیر فایل جدید برای ذخیره
output_path = "/kaggle/working/ResNet_50_sparse_pruned_weights_only.pt"

# بارگذاری فایل مدل
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# آستانه برای اعمال ماسک‌های باینری
mask_threshold = 0.1

# بارگذاری مدل برای دسترسی به معماری
model = ResNet_50_sparse_rvf10k(
    gumbel_start_temperature=1.0,  # مقادیر پیش‌فرض
    gumbel_end_temperature=0.1,
    num_epochs=100
)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(checkpoint['student'])
model.eval()

# ایجاد دیکشنری جدید برای وزن‌های هرس‌شده
new_state_dict = {}

# بررسی و اعمال ماسک‌ها
state_dict = checkpoint['student']
for name, param in state_dict.items():
    if 'mask_weight' in name:
        # یافتن وزن مرتبط با ماسک
        weight_name = name.replace('mask_weight', 'weight')
        if weight_name in state_dict:
            weight = state_dict[weight_name]
            mask = param
            if mask.dim() == 4 and weight.dim() == 4:  # اطمینان از کانولوشنی بودن
                # تبدیل ماسک به باینری (0 یا 1)
                binary_mask = (mask > mask_threshold).float()
                # اعمال ماسک به وزن‌ها
                pruned_weight = weight * binary_mask
                # شناسایی کانال‌های غیرصفر
                non_zero_channels = [i for i in range(pruned_weight.size(0)) if not torch.all(pruned_weight[i] == 0)]
                if non_zero_channels:
                    # فقط کانال‌های غیرصفر را نگه می‌داریم
                    new_state_dict[weight_name] = pruned_weight[non_zero_channels]
                    # ذخیره ماسک باینری (اختیاری، برای سازگاری)
                    new_state_dict[name] = binary_mask[non_zero_channels]
                else:
                    # اگر همه کانال‌ها صفر هستند، وزن را نگه می‌داریم (برای سازگاری)
                    new_state_dict[weight_name] = pruned_weight
                    new_state_dict[name] = binary_mask
            else:
                # برای ماسک‌ها یا وزن‌های غیرکانولوشنی
                new_state_dict[name] = param
        else:
            # اگر وزن مرتبط وجود نداشته باشد
            new_state_dict[name] = param
    elif 'weight' in name and name.replace('weight', 'mask_weight') not in state_dict:
        # وزن‌هایی که ماسک ندارند (مثل لایه‌های fc یا bn)
        new_state_dict[name] = param
    else:
        # سایر پارامترها (مثل bias یا bn)
        new_state_dict[name] = param

# ذخیره مدل جدید
new_checkpoint = {
    'student': new_state_dict,
    'best_prec1': checkpoint.get('best_prec1', 0.0)
}
torch.save(new_checkpoint, output_path)
print(f"مدل هرس‌شده با وزن‌های باقی‌مانده در مسیر {output_path} ذخیره شد.")

# بررسی حجم فایل جدید
import os
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # تبدیل به مگابایت
print(f"حجم فایل جدید: {file_size_mb:.2f} مگابایت")

# گزارش تعداد پارامترهای باقی‌مانده
total_params = sum(p.numel() for p in new_state_dict.values())
print(f"تعداد پارامترهای باقی‌مانده: {total_params:,}")
