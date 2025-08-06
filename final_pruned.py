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
    gumbel_start_temperature=1.0,
    gumbel_end_temperature=0.1,
    num_epochs=100
)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(checkpoint['student'])
model.eval()

# ایجاد دیکشنری جدید برای وزن‌های هرس‌شده
new_state_dict = {}
channel_counts = {}  # برای ذخیره تعداد کانال‌های باقی‌مانده در هر لایه

# بررسی و اعمال ماسک‌ها
state_dict = checkpoint['student']
for name, param in state_dict.items():
    if 'mask_weight' in name:
        weight_name = name.replace('mask_weight', 'weight')
        if weight_name in state_dict:
            weight = state_dict[weight_name]
            mask = param
            print(f"لایه: {name}, ابعاد وزن: {weight.shape}, ابعاد ماسک: {mask.shape}")
            
            if weight.dim() == 4:  # برای لایه‌های کانولوشنی
                out_channels = weight.size(0)
                if mask.dim() == 4 and mask.shape[0] == out_channels and mask.shape[1] == 2:
                    # انتخاب مقدار اول از بُعد دوم ماسک (یا حداکثر مقدار)
                    mask_selected = mask[:, 0, :, :].unsqueeze(1)  # تبدیل به [out_channels, 1, 1, 1]
                    binary_mask = (mask_selected > mask_threshold).float()
                    pruned_weight = weight * binary_mask
                    # شناسایی کانال‌های غیرصفر
                    non_zero_channels = [i for i in range(out_channels) if not torch.all(pruned_weight[i] == 0)]
                    if non_zero_channels:
                        new_state_dict[weight_name] = pruned_weight[non_zero_channels]
                        new_state_dict[name] = binary_mask[non_zero_channels]
                        channel_counts[weight_name] = len(non_zero_channels)
                        print(f"لایه {weight_name}: {len(non_zero_channels)} کانال از {out_channels} باقی ماند.")
                    else:
                        new_state_dict[weight_name] = pruned_weight
                        new_state_dict[name] = binary_mask
                        channel_counts[weight_name] = out_channels
                        print(f"لایه {weight_name}: هیچ کانالی حذف نشد.")
                else:
                    print(f"هشدار: ابعاد ماسک ({mask.shape}) با وزن ({weight.shape}) سازگار نیست. لایه بدون هرس ذخیره می‌شود.")
                    new_state_dict[weight_name] = weight
                    new_state_dict[name] = mask
                    channel_counts[weight_name] = out_channels
            else:
                new_state_dict[weight_name] = weight
                new_state_dict[name] = mask
                channel_counts[weight_name] = weight.size(0) if weight.dim() > 0 else 0
        else:
            new_state_dict[name] = param
    elif 'weight' in name and name.replace('weight', 'mask_weight') not in state_dict:
        new_state_dict[name] = param
        if 'weight' in name and param.dim() > 0:
            channel_counts[name] = param.size(0)
    else:
        new_state_dict[name] = param

# ذخیره مدل جدید
new_checkpoint = {
    'student': new_state_dict,
    'best_prec1': checkpoint.get('best_prec1', 0.0),
    'channel_counts': channel_counts  # ذخیره تعداد کانال‌ها برای بازتعریف مدل
}
torch.save(new_checkpoint, output_path)
print(f"مدل هرس‌شده با وزن‌های باقی‌مانده در مسیر {output_path} ذخیره شد.")

# بررسی حجم فایل جدید و تعداد پارامترها
import os
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"حجم فایل جدید: {file_size_mb:.2f} مگابایت")
total_params = sum(p.numel() for p in new_state_dict.values())
print(f"تعداد پارامترهای باقی‌مانده: {total_params:,}")
