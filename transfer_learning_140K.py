import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from thop import profile

# افزودن مسیر فایل ResNet_pruned.py به sys.path
sys.path.append('/model/pruned_model')

# ایمپورت کلاس‌های لازم از ResNet_pruned.py
try:
    from ResNet_pruned import ResNet_pruned, Bottleneck_pruned
except ImportError as e:
    print(f"Error importing ResNet_pruned or Bottleneck_pruned: {e}")
    raise

# تابع برای محاسبه تعداد فیلترهای نگه‌داشته‌شده
def get_preserved_filter_num(mask):
    return int(mask.sum())

# تعریف ماسک‌ها بر اساس اطلاعات ارائه‌شده
masks = [
    torch.ones(11), torch.ones(7), torch.ones(57),   # layer1.0
    torch.ones(7), torch.ones(10), torch.ones(43),   # layer1.1
    torch.ones(10), torch.ones(8), torch.ones(41),   # layer1.2
    torch.ones(28), torch.ones(19), torch.ones(96),  # layer2.0
    torch.ones(25), torch.ones(22), torch.ones(94),  # layer2.1
    torch.ones(25), torch.ones(13), torch.ones(77),  # layer2.2
    torch.ones(18), torch.ones(16), torch.ones(59),  # layer2.3
    torch.ones(30), torch.ones(39), torch.ones(113), # layer3.0
    torch.ones(51), torch.ones(18), torch.ones(112), # layer3.1
    torch.ones(46), torch.ones(15), torch.ones(71),  # layer3.2
    torch.ones(47), torch.ones(15), torch.ones(67),  # layer3.3
    torch.ones(28), torch.ones(8), torch.ones(51),   # layer3.4
    torch.ones(30), torch.ones(11), torch.ones(44),  # layer3.5
    torch.ones(35), torch.ones(50), torch.ones(66),  # layer4.0
    torch.ones(86), torch.ones(12), torch.ones(64),  # layer4.1
    torch.ones(57), torch.ones(18), torch.ones(102), # layer4.2
]

# تعریف مدل
model = ResNet_pruned(block=Bottleneck_pruned, num_blocks=[3, 4, 6, 3], masks=masks, num_classes=1)

# اصلاح تعداد ورودی‌های لایه fc
model.fc = nn.Linear(get_preserved_filter_num(masks[-1]), 1)  # layer4.2.conv3: 102 کانال

# لود کردن وزن‌ها
model_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
try:
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model_state_dict = model.state_dict()
    state_dict_filtered = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(state_dict_filtered)
    model.load_state_dict(model_state_dict)
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# انتقال مدل به دستگاه مناسب
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# آماده‌سازی دیتاست تست
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    test_dataset = ImageFolder(root='/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
except FileNotFoundError as e:
    print(f"Error: Test dataset directory not found. Please verify the path: {e}")
    raise

# تابع ارزیابی مدل
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)  # خروجی مدل شامل feature_list است
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.int32).flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC-ROC: {auc:.4f}')
    
    return accuracy, precision, recall, f1, auc

# اجرای ارزیابی
try:
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_loader)
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

# محاسبه تعداد پارامترها و FLOPs
try:
    flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256).to(device),))
    print(f'Params: {params/1e6:.2f}M, FLOPs: {flops/1e6:.2f}M')
except Exception as e:
    print(f"Error calculating FLOPs/Params: {e}")

# ذخیره نتایج
with open('evaluation_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nAUC-ROC: {auc:.4f}\n')
    f.write(f'Params: {params/1e6:.2f}M\nFLOPs: {flops/1e6:.2f}M\n')
