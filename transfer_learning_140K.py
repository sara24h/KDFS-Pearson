import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from thop import profile
import argparse
from torch.amp import autocast

# فرض می‌کنیم این ماژول‌ها از پروژه شما وارد شده‌اند
from model.student.ResNet_pruned import ResNet_50_pruned_hardfakevsreal, get_preserved_filter_num
from data.dataset import Dataset_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Test generalization of pruned ResNet50 student model on multiple datasets.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the student model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing dataset folders')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    return parser.parse_args()

# تنظیمات اولیه
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# 1. لود چک‌پوینت و استخراج ماسک‌ها
checkpoint_path = args.checkpoint_path
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

if 'student' in checkpoint:
    state_dict = checkpoint['student']
else:
    raise KeyError("'student' key not found in checkpoint")

# استخراج ماسک‌ها
masks = []
for key, value in state_dict.items():
    if 'mask_weight' in key:
        mask_binary = (torch.argmax(value, dim=1).squeeze(1).squeeze(1) != 0).float()
        masks.append(mask_binary)
print(f"Number of masks extracted: {len(masks)}")
if len(masks) != 48:  # برای ResNet50 باید 48 ماسک باشد
    print(f"Warning: Expected 48 masks for ResNet50, got {len(masks)}")

# 2. ساخت مدل پرون‌شده
model = ResNet_50_pruned_hardfakevsreal(masks=masks)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # برای مسائل باینری (real vs fake)
model = model.to(device)

# 3. فیلتر کردن و لود وزن‌ها
pruned_state_dict = {}
for key, value in state_dict.items():
    if 'mask_weight' in key or 'feat' in key:
        continue  # نادیده گرفتن ماسک‌ها و لایه‌های feat
    if 'weight' in key and 'conv' in key:
        mask_key = key.replace('.weight', '.mask_weight')
        if mask_key in state_dict:
            mask = state_dict[mask_key]
            mask_binary = (torch.argmax(mask, dim=1).squeeze(1).squeeze(1) != 0).float()
            if len(value.shape) == 4:  # برای لایه‌های کانولوشنی
                out_channels = mask_binary.sum().int().item()
                pruned_weight = value[mask_binary.bool()]
                if 'conv2' in key or 'conv3' in key:
                    prev_mask_key = key.replace('conv2', 'conv1').replace('conv3', 'conv2')
                    if prev_mask_key in state_dict:
                        prev_mask = state_dict[prev_mask_key]
                        prev_mask_binary = (torch.argmax(prev_mask, dim=1).squeeze(1).squeeze(1) != 0).float()
                        pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
                pruned_state_dict[key] = pruned_weight
            else:
                pruned_state_dict[key] = value
        else:
            pruned_state_dict[key] = value
    else:
        pruned_state_dict[key] = value

missing, unexpected = model.load_state_dict(pruned_state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# 4. محاسبه تعداد پارامترها و FLOPs
input = torch.randn(1, 3, 256, 256).to(device)  # اندازه پیش‌فرض برای اکثر دیتاست‌ها
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops / 1e9:.2f} GMac, Parameters: {params / 1e6:.2f} M")

# 5. تعریف دیتاست‌ها و لودرها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': os.path.join(args.data_dir, 'hardfakevsrealfaces/data.csv'),
        'hardfake_root_dir': os.path.join(args.data_dir, 'hardfakevsrealfaces'),
        'image_size': (300, 300),
        'mean': (0.5124, 0.4165, 0.3684),
        'std': (0.2363, 0.2087, 0.2029),
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': os.path.join(args.data_dir, 'rvf10k/train.csv'),
        'rvf10k_valid_csv': os.path.join(args.data_dir, 'rvf10k/valid.csv'),
        'rvf10k_root_dir': os.path.join(args.data_dir, 'rvf10k'),
        'image_size': (256, 256),
        'mean': (0.5212, 0.4260, 0.3811),
        'std': (0.2486, 0.2238, 0.2211),
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_train_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/train.csv'),
        'realfake140k_valid_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/valid.csv'),
        'realfake140k_test_csv': os.path.join(args.data_dir, '140k-real-and-fake-faces/test.csv'),
        'realfake140k_root_dir': os.path.join(args.data_dir, '140k-real-and-fake-faces'),
        'image_size': (256, 256),
        'mean': (0.5207, 0.4258, 0.3806),
        'std': (0.2490, 0.2239, 0.2212),
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': os.path.join(args.data_dir, 'deepfake-and-real-images/Dataset'),
        'image_size': (256, 256),
        'mean': (0.4668, 0.3816, 0.3414),
        'std': (0.2410, 0.2161, 0.2081),
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_train_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/train_labels.csv'),
        'realfake200k_val_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/val_labels.csv'),
        'realfake200k_test_csv': os.path.join(args.data_dir, '200k-real-and-fake-faces/test_labels.csv'),
        'realfake200k_root_dir': os.path.join(args.data_dir, '200k-real-and-fake-faces'),
        'image_size': (256, 256),
        'mean': (0.4868, 0.3972, 0.3624),
        'std': (0.2296, 0.2066, 0.2009),
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': os.path.join(args.data_dir, 'deepfake-dataset'),
        'image_size': (256, 256),
        'mean': (0.4923, 0.4042, 0.3624),
        'std': (0.2446, 0.2198, 0.2141),
    }
}

# 6. تابع ارزیابی
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs, _ = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / len(data_loader) if total > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

# 7. تست روی همه دیتاست‌ها
results = {}
criterion = nn.BCEWithLogitsLoss()

for dataset_name, config in dataset_configs.items():
    print(f"\nTesting on {dataset_name} dataset...")
    
    # تعریف transform
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])

    # لود دیتاست
    try:
        dataset = Dataset_selector(
            dataset_mode=config['dataset_mode'],
            hardfake_csv_file=config.get('hardfake_csv_file'),
            hardfake_root_dir=config.get('hardfake_root_dir'),
            rvf10k_train_csv=config.get('rvf10k_train_csv'),
            rvf10k_valid_csv=config.get('rvf10k_valid_csv'),
            rvf10k_root_dir=config.get('rvf10k_root_dir'),
            realfake140k_train_csv=config.get('realfake140k_train_csv'),
            realfake140k_valid_csv=config.get('realfake140k_valid_csv'),
            realfake140k_test_csv=config.get('realfake140k_test_csv'),
            realfake140k_root_dir=config.get('realfake140k_root_dir'),
            realfake200k_train_csv=config.get('realfake200k_train_csv'),
            realfake200k_val_csv=config.get('realfake200k_val_csv'),
            realfake200k_test_csv=config.get('realfake200k_test_csv'),
            realfake200k_root_dir=config.get('realfake200k_root_dir'),
            realfake190k_root_dir=config.get('realfake190k_root_dir'),
            realfake330k_root_dir=config.get('realfake330k_root_dir'),
            train_batch_size=0,  # فقط تست
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ddp=False
        )
        test_loader = dataset.loader_test
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        results[dataset_name] = {'error': str(e)}
        continue

    # ارزیابی
    try:
        test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
        print(f"{dataset_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        results[dataset_name] = {'loss': test_loss, 'accuracy': test_accuracy}
    except Exception as e:
        print(f"Error evaluating {dataset_name} dataset: {e}")
        results[dataset_name] = {'error': str(e)}

# 8. ذخیره نتایج
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'generalization_results.txt'), 'w') as f:
    f.write(f"FLOPs: {flops / 1e9:.2f} GMac\n")
    f.write(f"Parameters: {params / 1e6:.2f} M\n\n")
    for dataset_name, result in results.items():
        f.write(f"Dataset: {dataset_name}\n")
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
        else:
            f.write(f"Test Loss: {result['loss']:.4f}\n")
            f.write(f"Test Accuracy: {result['accuracy']:.2f}%\n")
        f.write("\n")
print(f"Results saved to {os.path.join(results_dir, 'generalization_results.txt')}")

# 9. چاپ نتایج نهایی
print("\nFinal Generalization Results:")
for dataset_name, result in results.items():
    print(f"\nDataset: {dataset_name}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Test Loss: {result['loss']:.4f}, Test Accuracy: {result['accuracy']:.2f}%")
