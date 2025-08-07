import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from thop import profile
import os
import warnings
import argparse
import logging

# تنظیم لاگ‌گیری
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"استفاده از دستگاه: {device}")

parser = argparse.ArgumentParser(description='آزمایش یک مدل روی مجموعه داده مشخص')
parser.add_argument('--dataset', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                    help='نام مجموعه داده برای آزمایش (مثل hardfake، rvf10k، 140k، 190k، 200k، 330k)')
args = parser.parse_args()

# بارگذاری مدل
try:
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device, weights_only=False)
    logging.info("مدل با موفقیت بارگذاری شد.")
except Exception as e:
    logging.error(f"خطا در بارگذاری مدل: {e}")
    raise Exception("بارگذاری مدل ناموفق بود.")
model = model.to(device)
model.eval()

# دیباگ خروجی مدل
sample_input = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    sample_output = model(sample_input)
    logging.info(f"نوع خروجی مدل: {type(sample_output)}, شکل: {sample_output[0].shape if isinstance(sample_output, tuple) else sample_output.shape}")

# محاسبه FLOPs و پارامترها
input = torch.randn(1, 3, 256, 256).to(device)  # تطبیق با اندازه تصویر مجموعه داده
try:
    flops, params = profile(model, inputs=(input,))
    logging.info(f"پارامترها: {params/1e6:.2f}M, FLOPs: {flops/1e6:.2f}M")
except Exception as e:
    logging.error(f"خطا در پروفایلینگ مدل: {e}")

# تنظیمات مجموعه داده‌ها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
        'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
        'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
        'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
        'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    }
}

# انتخاب مجموعه داده
dataset_name = args.dataset
if dataset_name not in dataset_configs:
    logging.error(f"مجموعه داده نامعتبر: {dataset_name}. مجموعه داده‌های موجود: {list(dataset_configs.keys())}")
    raise ValueError(f"مجموعه داده نامعتبر: {dataset_name}")

# بارگذاری مجموعه داده
from data.dataset import Dataset_selector
try:
    dataset = Dataset_selector(**dataset_configs[dataset_name])
    logging.info(f"مجموعه داده {dataset_name} با موفقیت بارگذاری شد")
except Exception as e:
    logging.error(f"خطا در بارگذاری مجموعه داده {dataset_name}: {e}")
    raise

# تابع پردازش خروجی مدل
def process_model_output(outputs):
    try:
        if isinstance(outputs, tuple):
            outputs = outputs[0].clone()  # انتخاب اولین عنصر تاپل و کپی برای جلوگیری از عملیات درجا
        if outputs.dim() > 1:
            outputs = outputs.squeeze(1)
        return outputs
    except Exception as e:
        logging.error(f"خطا در پردازش خروجی مدل: {e}")
        raise

# تابع ارزیابی
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            try:
                outputs = process_model_output(model(inputs))
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logging.error(f"خطا در پاس رو به جلو ارزیابی: {e}")
                raise
    accuracy = accuracy_score(true_labels, predictions) * 100
    f1 = f1_score(true_labels, predictions, average='binary') * 100
    avg_loss = running_loss / len(dataloader)
    return accuracy, f1, avg_loss

# تابع آموزش
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=5):
    metrics = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            try:
                # غیرفعال کردن موقت ردیابی گرادیان برای جلوگیری از خطای عملیات درجا
                with torch.no_grad():
                    raw_outputs = model(inputs)
                outputs = process_model_output(raw_outputs)  # کپی خروجی برای جلوگیری از تغییرات درجا
                outputs.requires_grad_(True)  # فعال کردن گرادیان برای لوجیت‌های پردازش‌شده
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                logging.error(f"خطا در پاس رو به جلو آموزش: {e}")
                raise
        metrics['train_loss'].append(running_loss / len(trainloader))
        logging.info(f"اپوک {epoch+1}, خطای آموزشی: {running_loss / len(trainloader):.4f}")
        
        # ارزیابی روی مجموعه اعتبارسنجی
        accuracy, f1, val_loss = evaluate(model, valloader, criterion)
        metrics['val_accuracy'].append(accuracy)
        metrics['val_f1'].append(f1)
        metrics['val_loss'].append(val_loss)
        logging.info(f"اپوک {epoch+1}, دقت اعتبارسنجی: {accuracy:.2f}%, امتیاز F1: {f1:.2f}%, خطای اعتبارسنجی: {val_loss:.4f}")
    return metrics

# آزمایش بدون فاین‌تیونینگ
criterion = nn.BCEWithLogitsLoss()
logging.info(f"\nآزمایش {dataset_name} بدون فاین‌تیونینگ...")
try:
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_without_finetune = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    logging.info(f"{dataset_name} بدون فاین‌تیونینگ - دقت: {accuracy:.2f}%, امتیاز F1: {f1:.2f}%, خطا: {loss:.4f}")
except Exception as e:
    logging.error(f"خطا در ارزیابی بدون فاین‌تیونینگ: {e}")
    raise

# فاین‌تیونینگ
logging.info(f"\nفاین‌تیونینگ روی {dataset_name}...")
try:
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device, weights_only=False)
    model = model.to(device)
except Exception as e:
    logging.error(f"خطا در بارگذاری مدل برای فاین‌تیونینگ: {e}")
    raise

# فریز کردن لایه‌های اولیه
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# آموزش مدل
metrics = train_model(model, dataset.loader_train, dataset.loader_val, criterion, optimizer, num_epochs=5)

# آزمایش روی مجموعه تست
try:
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_with_finetune = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    logging.info(f"{dataset_name} با فاین‌تیونینگ - دقت: {accuracy:.2f}%, امتیاز F1: {f1:.2f}%, خطا: {loss:.4f}")
except Exception as e:
    logging.error(f"خطا در ارزیابی با فاین‌تیونینگ: {e}")
    raise

# ذخیره مدل فاین‌تیون‌شده
try:
    torch.save(model, f'finetuned_{dataset_name}.pt')
    logging.info(f"مدل فاین‌تیون‌شده با نام finetuned_{dataset_name}.pt ذخیره شد")
except Exception as e:
    logging.error(f"خطا در ذخیره مدل فاین‌تیون‌شده: {e}")
    raise

# خلاصه نتایج
logging.info("\nخلاصه نتایج:")
logging.info("\nبدون فاین‌تیونینگ:")
logging.info(f"{dataset_name}: دقت = {results_without_finetune['accuracy']:.2f}%, امتیاز F1 = {results_without_finetune['f1_score']:.2f}%, خطا = {results_without_finetune['loss']:.4f}")
logging.info("\nبا فاین‌تیونینگ:")
logging.info(f"{dataset_name}: دقت = {results_with_finetune['accuracy']:.2f}%, امتیاز F1 = {results_with_finetune['f1_score']:.2f}%, خطا = {results_with_finetune['loss']:.4f}")
