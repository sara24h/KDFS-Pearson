import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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
        self.finetuned_ckpt_path = "finetuned_checkpoint.pth"  # مسیر برای ذخیره چک‌پوینت فاین‌تیون‌شده

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA در دسترس نیست! لطفاً تنظیمات GPU را بررسی کنید.")

    def dataload(self):
        print("==> در حال بارگذاری دیتاست...")
        try:
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"فایل CSV پیدا نشد: {csv_path}")
                
                dataset = Dataset_selector(
                    dataset_mode='hardfake',
                    hardfake_csv_file=csv_path,
                    hardfake_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
                self.train_loader = dataset.loader_train
                self.val_loader = dataset.loader_val
                self.test_loader = dataset.loader_test
                
                # چاپ آمار دیتاست
                print("hardfake dataset statistics:")
                print("Sample train image paths:")
                print(dataset.df_train['images_id'].head())
                print(f"Total train dataset size: {len(dataset.df_train)}")
                print("Train label distribution:")
                print(dataset.df_train['label'].value_counts())
                print("Sample validation image paths:")
                print(dataset.df_val['images_id'].head())
                print(f"Total validation dataset size: {len(dataset.df_val)}")
                print("Validation label distribution:")
                print(dataset.df_val['label'].value_counts())
                print("Sample test image paths:")
                print(dataset.df_test['images_id'].head())
                print(f"Total test dataset size: {len(dataset.df_test)}")
                print("Test label distribution:")
                print(dataset.df_test['label'].value_counts())
                
                print(f"Train loader batches: {len(self.train_loader)}")
                print(f"Validation loader batches: {len(self.val_loader)}")
                print(f"Test loader batches: {len(self.test_loader)}")
                
                # چاپ نمونه‌ای از شکل داده‌ها
                for images, targets in self.train_loader:
                    print(f"Sample train batch image shape: {images.shape}")
                    print(f"Sample train batch labels: {targets}")
                    break
                for images, targets in self.val_loader:
                    print(f"Sample validation batch image shape: {images.shape}")
                    print(f"Sample validation batch labels: {targets}")
                    break
                for images, targets in self.test_loader:
                    print(f"Sample test batch image shape: {images.shape}")
                    print(f"Sample test batch labels: {targets}")
                    break
                
            else:
                raise ValueError(f"حالت دیتاست {self.dataset_mode} پشتیبانی نمی‌شود.")
            
            print(f"دیتاست تست {self.dataset_mode} بارگذاری شد! تعداد دسته‌ها: {len(self.test_loader)}")
        except Exception as e:
            print(f"خطا در بارگذاری دیتاست: {str(e)}")
            raise

    def build_model(self, fine_tuned=True):
        print(f"==> در حال ساخت مدل {'فاین‌تیون‌شده (با فریز لایه‌ها)' if fine_tuned else 'بدون فاین‌تیون'}...")
        try:
            model = ResNet_50_sparse_hardfakevsreal()
            ckpt_path = self.finetuned_ckpt_path if fine_tuned else self.sparsed_student_ckpt_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"فایل چک‌پوینت پیدا نشد: {ckpt_path}")
            
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt["student"] if "student" in ckpt else ckpt
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"بارگذاری state_dict با strict=True ناموفق بود: {str(e)}")
                print("تلاش با strict=False...")
                model.load_state_dict(state_dict, strict=False)
                print("با strict=False بارگذاری شد.")

            if fine_tuned:
                for name, param in model.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False
                print("لایه‌های مدل به جز layer4 و fc فریز شدند.")
            else:
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
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_overall = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_overall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_overall = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
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
        model.ticket = True
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

            Flops_baseline, Flops, Flops_reduction, Params_baseline, Params, Params_reduction = get_flops_and_params(args=self.args)
            print(f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, کاهش پارامترها: {Params_reduction:.2f}%")
            print(f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, کاهش Flops: {Flops_reduction:.2f}%")
        except Exception as e:
            print(f"خطا در حین تست {model_name}: {str(e)}")
            raise

    def fine_tune(self, model, num_epochs=10, lr=1e-4):
        print("==> در حال فاین‌تیون مدل...")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1.5, 1.0]).to(self.device))  # وزن‌دهی به کلاس جعلی
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        model.train()
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                optimizer.zero_grad()
                logits, _ = model(images)
                loss = criterion(logits.squeeze(), targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # اعتبارسنجی
            val_loss = self.validate(model)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss / len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # تنظیم نرخ یادگیری
            scheduler.step(val_loss)
            
            # ذخیره بهترین مدل
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.finetuned_ckpt_path)
                print(f"چک‌پوینت فاین‌تیون‌شده ذخیره شد: {self.finetuned_ckpt_path}")

    def validate(self, model):
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        running_loss = 0.0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                logits, _ = model(images)
                loss = criterion(logits.squeeze(), targets)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def main(self):
        print(f"شروع فرآیند تست با حالت دیتاست: {self.dataset_mode}")
        try:
            print(f"نسخه PyTorch: {torch.__version__}")
            print(f"CUDA در دسترس: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"نسخه CUDA: {torch.version.cuda}")
                print(f"نام دستگاه: {torch.cuda.get_device_name(0)}")

            self.dataload()

            # فاین‌تیون مدل
            print("\n=== فاین‌تیون مدل (با فریز لایه‌ها) ===")
            model_ft = ResNet_50_sparse_hardfakevsreal()
            for name, param in model_ft.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
            model_ft.to(self.device)
            
            # بارگذاری چک‌پوینت اولیه
            ckpt = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt["student"] if "student" in ckpt else ckpt
            model_ft.load_state_dict(state_dict, strict=True)
            print(f"چک‌پوینت اولیه بارگذاری شد: {self.sparsed_student_ckpt_path}")

            # اجرای فاین‌تیون
            self.fine_tune(model_ft, num_epochs=10, lr=1e-4)

            # تست مدل بدون فاین‌تیون
            print("\n=== تست مدل بدون فاین‌تیون (همه لایه‌ها آزاد) ===")
            model_no_ft = self.build_model(fine_tuned=False)
            self.test(model_no_ft, model_name="بدون فاین‌تیون")

            # تست مدل فاین‌تیون‌شده
            print("\n=== تست مدل فاین‌تیون‌شده (با فریز لایه‌ها) ===")
            model_ft = self.build_model(fine_tuned=True)
            self.test(model_ft, model_name="فاین‌تیون‌شده")

        except Exception as e:
            print(f"خطا در فرآیند تست: {str(e)}")
            raise

# تنظیمات نمونه
class Args:
    dataset_dir = "path/to/hardfake/dataset"  # مسیر دیتاست
    num_workers = 8
    pin_memory = True
    arch = "ResNet_50"
    device = "cuda"
    test_batch_size = 256
    sparsed_student_ckpt_path = "path/to/sparsed_student_ckpt.pth"  # مسیر چک‌پوینت اولیه
    dataset_mode = "hardfake"

if __name__ == "__main__":
    args = Args()
    test = Test(args)
    test.main()
