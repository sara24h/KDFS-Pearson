import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# کد مدل شما (همان کدی که دادید)
import torch.nn.functional as F

def get_preserved_filter_num(mask):
    return int(mask.sum())

class BasicBlock_pruned(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes,
            preserved_filter_num1,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut_out = self.downsample(x).clone()
        padded_out = torch.zeros_like(shortcut_out).clone()
        for padded_feature_map, feature_map in zip(padded_out, out):
            padded_feature_map[self.masks[1] == 1] = feature_map

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out

class Bottleneck_pruned(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes, preserved_filter_num1, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)
        preserved_filter_num3 = get_preserved_filter_num(masks[2])
        self.conv3 = nn.Conv2d(
            preserved_filter_num2,
            preserved_filter_num3,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(preserved_filter_num3)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        shortcut_out = self.downsample(x).clone()
        padded_out = torch.zeros_like(shortcut_out).clone()
        for padded_feature_map, feature_map in zip(padded_out, out):
            padded_feature_map[self.masks[2] == 1] = feature_map

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out

class ResNet_pruned(nn.Module):
    def __init__(self, block, num_blocks, masks=[], num_classes=1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3
        num = 0
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            masks=masks[0 : coef * num_blocks[0]],
        )
        num = num + coef * num_blocks[0]

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            masks=masks[num : num + coef * num_blocks[1]],
        )
        num = num + coef * num_blocks[1]

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            masks=masks[num : num + coef * num_blocks[2]],
        )
        num = num + coef * num_blocks[2]

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            masks=masks[num : num + coef * num_blocks[3]],
        )
        num = num + coef * num_blocks[3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, masks=[]):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3

        for i, stride in enumerate(strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    masks[coef * i : coef * i + coef],
                    stride,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out)
        feature_list.append(out)

        for block in self.layer2:
            out = block(out)
        feature_list.append(out)

        for block in self.layer3:
            out = block(out)
        feature_list.append(out)

        for block in self.layer4:
            out = block(out)
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_50_pruned_hardfakevsreal(masks):
    return ResNet_pruned(
        block=Bottleneck_pruned, num_blocks=[3, 4, 6, 3], masks=masks, num_classes=1
    )

def load_pruned_model(model_path, masks_path=None):
    """
    مدل pruned شده را بارگذاری می‌کند
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # بارگذاری checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # استخراج masks (اگر در checkpoint موجود باشد)
    if 'masks' in checkpoint:
        masks = checkpoint['masks']
    elif masks_path:
        # اگر masks جداگانه ذخیره شده باشد
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f)
    else:
        print("Warning: No masks found. You need to provide masks for the pruned model.")
        return None
    
    # ایجاد مدل
    model = ResNet_50_pruned_hardfakevsreal(masks)
    
    # بارگذاری وزن‌ها
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, device

# کد دیتاست شما
import pandas as pd
from sklearn.model_selection import train_test_split

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

def prepare_test_dataloader(dataset_mode='140k', batch_size=32, **dataset_kwargs):
    """
    دیتالودر تست را با استفاده از Dataset_selector آماده می‌کند
    """
    # Dataset_selector را import کنید یا کد آن را اینجا قرار دهید
    from your_dataset_file import Dataset_selector  # نام فایل دیتاست خود را جایگزین کنید
    
    # ایجاد دیتاست
    dataset = Dataset_selector(
        dataset_mode=dataset_mode,
        eval_batch_size=batch_size,
        **dataset_kwargs
    )
    
    return dataset.loader_test, dataset

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
        # ایجاد دیتاست به صورت مستقیم
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
