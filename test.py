import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from utils import meter
from get_flops_and_params import get_flops_and_params
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from data.dataset import Dataset_selector

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch  # Expected to be 'ResNet_50'
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode  # 'hardfake', 'rvf10k', '140k', '200k', '190k', '330k'

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA در دسترس نیست! لطفاً تنظیمات GPU را بررسی کنید.")

    def dataload(self):
        print("==> در حال بارگذاری دیتاست تست...")
        try:
            # Verify dataset paths
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"فایل CSV پیدا نشد: {csv_path}")
            elif self.dataset_mode == 'rvf10k':
                train_csv = os.path.join(self.dataset_dir, 'train.csv')
                valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"فایل‌های CSV پیدا نشدند: {train_csv}, {valid_csv}")
            elif self.dataset_mode == '140k':
                test_csv = os.path.join(self.dataset_dir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"فایل CSV پیدا نشد: {test_csv}")
            elif self.dataset_mode == '200k':
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"فایل CSV پیدا نشد: {test_csv}")
            elif self.dataset_mode == '330k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"پوشه دیتاست پیدا نشد: {self.dataset_dir}")

            # Initialize dataset based on mode
            if self.dataset_mode == 'hardfake':
                dataset = Dataset_selector(
                    dataset_mode='hardfake',
                    hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv'),
                    hardfake_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == 'rvf10k':
                dataset = Dataset_selector(
                    dataset_mode='rvf10k',
                    rvf10k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                    rvf10k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                    rvf10k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '140k':
                dataset = Dataset_selector(
                    dataset_mode='140k',
                    realfake140k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                    realfake140k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                    realfake140k_test_csv=os.path.join(self.dataset_dir, 'test.csv'),
                    realfake140k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '200k':
                dataset = Dataset_selector(
                    dataset_mode='200k',
                    realfake200k_train_csv=os.path.join(self.dataset_dir, 'train_labels.csv'),
                    realfake200k_val_csv=os.path.join(self.dataset_dir, 'val_labels.csv'),
                    realfake200k_test_csv=os.path.join(self.dataset_dir, 'test_labels.csv'),
                    realfake200k_root_dir=os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset/my_real_vs_ai_dataset'),
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '330k':
                dataset = Dataset_selector(
                    dataset_mode='330k',
                    realfake330k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )

            self.test_loader = dataset.loader_test
            print(f"دیتاست تست {self.dataset_mode} بارگذاری شد! تعداد دسته‌ها: {len(self.test_loader)}")
        except Exception as e:
            print(f"خطا در بارگذاری دیتاست: {str(e)}")
            raise

    def build_model(self, fine_tuned=True):
        print(f"==> در حال ساخت مدل {'فاین‌تیون‌شده (با فریز لایه‌ها)' if fine_tuned else 'بدون فاین‌تیون'}...")
        try:
            model = ResNet_50_sparse_hardfakevsreal()
        
        # بارگذاری چک‌پوینت
            if not os.path.exists(self.sparsed_student_ckpt_path):
                raise FileNotFoundError(f"فایل چک‌پوینت پیدا نشد: {self.sparsed_student_ckpt_path}")
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            try:
               model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"بارگذاری state_dict با strict=True ناموفق بود: {str(e)}")
                print("تلاش با strict=False برای شناسایی کلیدهای ناسازگار...")
                model.load_state_dict(state_dict, strict=False)
                print("با strict=False بارگذاری شد؛ کلیدهای گمشده یا غیرمنتظره را بررسی کنید.")

            if fine_tuned:
            # فریز کردن همه لایه‌ها به جز layer4 و fc
                for name, param in model.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False
                print("لایه‌های مدل به جز layer4 و fc فریز شدند.")
            else:
            # بدون فریز کردن لایه‌ها
                for param in model.parameters():
                    param.requires_grad = True
                print("هیچ لایه‌ای فریز نشد.")

            model.to(self.device)
            print(f"مدل روی {self.device} بارگذاری شد")
            return model
        except Exception as e:
            print(f"خطا در ساخت مدل: {str(e)}")
            raise

    def compute_metrics(self, y_true, y_pred):
        # محاسبه معیارها برای هر کلاس
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # محاسبه معیارها برای کل داده‌ها (میانگین وزنی)
        precision_overall = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_overall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_overall = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # محاسبه specificity برای هر کلاس
        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # True Negatives
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])  # False Positives
            specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        return {
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'precision_overall': precision_overall,
            'recall_overall': recall_overall,
            'f1_overall': f1_overall,
            'specificity_per_class': specificity_per_class,
            'confusion_matrix': cm
        }

    def test(self, model, model_name=""):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        all_preds = []
        all_targets = []

        model.eval()
        model.ticket = True  # فعال‌سازی حالت ticket برای مدل اسپارس
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc=f"تست {model_name}") as _tqdm:
                    for images, targets in self.test_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        logits, _ = model(images)
                        logits = logits.squeeze()
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100.0 * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            print(f"[تست {model_name}] دیتاست: {self.dataset_mode}, دقت@1: {meter_top1.avg:.2f}%")

            # محاسبه معیارها
            metrics = self.compute_metrics(all_targets, all_preds)
            print(f"\nمعیارهای {model_name}:")
            print(f"--- معیارهای کلی (میانگین وزنی) ---")
            print(f"Precision (کلی): {metrics['precision_overall']:.4f}")
            print(f"Recall (کلی): {metrics['recall_overall']:.4f}")
            print(f"F1 Score (کلی): {metrics['f1_overall']:.4f}")
            print(f"\n--- معیارهای به تفکیک هر کلاس (0=جعلی, 1=واقعی) ---")
            print(f"Precision برای هر کلاس: {metrics['precision_per_class']}")
            print(f"Recall برای هر کلاس: {metrics['recall_per_class']}")
            print(f"F1 Score برای هر کلاس: {metrics['f1_per_class']}")
            print(f"Specificity برای هر کلاس: {metrics['specificity_per_class']}")
            print(f"ماتریس درهم‌ریختگی:\n{metrics['confusion_matrix']}")

            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = get_flops_and_params(args=self.args)
            print(
                f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
                f"کاهش پارامترها: {Params_reduction:.2f}%"
            )
            print(
                f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"کاهش Flops: {Flops_reduction:.2f}%"
            )

        except Exception as e:
            print(f"خطا در حین تست {model_name}: {str(e)}")
            raise

    def main(self):
        print(f"شروع فرآیند تست با حالت دیتاست: {self.dataset_mode}")
        try:
            print(f"نسخه PyTorch: {torch.__version__}")
            print(f"CUDA در دسترس: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"نسخه CUDA: {torch.version.cuda}")
                print(f"نام دستگاه: {torch.cuda.get_device_name(0)}")

            self.dataload()
            
            print("\n=== تست مدل بدون فاین‌تیون (همه لایه‌ها آزاد) ===")
            model_no_ft = self.build_model(fine_tuned=False)
            self.test(model_no_ft, model_name="بدون فاین‌تیون")

            print("\n=== تست مدل فاین‌تیون‌شده (با فریز لایه‌ها) ===")
            model_ft = self.build_model(fine_tuned=True)
            self.test(model_ft, model_name="فاین‌تیون‌شده")

        except Exception as e:
            print(f"خطا در فرآیند تست: {str(e)}")
            raise
