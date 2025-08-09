import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Test:

    def __init__(self, dataset_dir, sparsed_student_ckpt_path, dataset_mode, result_dir, 
                 train_batch_size, test_batch_size, num_workers, pin_memory, 
                 device, f_epochs, f_lr):
        
        self.dataset_dir = dataset_dir
        self.sparsed_student_ckpt_path = sparsed_student_ckpt_path
        self.dataset_mode = dataset_mode
        self.result_dir = result_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.f_epochs = f_epochs
        self.f_lr = f_lr

        print(f"**Device selected:** {self.device}")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.student = None

    def dataload(self):
        """بارگذاری مجموعه داده‌ها با استفاده از تنظیمات از پیش تعیین‌شده."""
        print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_140k = [0.5207, 0.4258, 0.3806]
        std_140k = [0.2490, 0.2239, 0.2212]

        transform_train_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        transform_val_test_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False
        }

        if self.dataset_mode == 'hardfake':
            params['hardfake_csv_file'] = os.path.join(self.dataset_dir, 'data.csv')
            params['hardfake_root_dir'] = self.dataset_dir
        elif self.dataset_mode == 'rvf10k':
            params['rvf10k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['rvf10k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['rvf10k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '140k':
            params['realfake140k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['realfake140k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['realfake140k_test_csv'] = os.path.join(self.dataset_dir, 'test.csv')
            params['realfake140k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '200k':
            image_root_dir = os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset', 'my_real_vs_ai_dataset')
            params['realfake200k_root_dir'] = image_root_dir
            params['realfake200k_train_csv'] = os.path.join(self.dataset_dir, 'train_labels.csv')
            params['realfake200k_val_csv'] = os.path.join(self.dataset_dir, 'val_labels.csv')
            params['realfake200k_test_csv'] = os.path.join(self.dataset_dir, 'test_labels.csv')
        elif self.dataset_mode == '190k':
            params['realfake190k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '330k':
            params['realfake330k_root_dir'] = self.dataset_dir

        dataset_manager = Dataset_selector(**params)

        print("Overriding transforms to use consistent 140k normalization stats for all datasets.")
        dataset_manager.loader_train.dataset.transform = transform_train_140k
        dataset_manager.loader_val.dataset.transform = transform_val_test_140k
        dataset_manager.loader_test.dataset.transform = transform_val_test_140k

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        print(f"All loaders for '{self.dataset_mode}' are now configured with 140k normalization.")

    def build_model(self):
        """ساخت مدل دانشجوی (student) و بارگذاری وزن‌های از پیش آموزش‌دیده."""
        print("==> Building student model...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        # بارگذاری وزن‌ها، با نادیده گرفتن لایه‌هایی که تطابق ندارند
        self.student.load_state_dict(state_dict, strict=False)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def test(self, loader, description="Test", is_finetuned=False):
        """ارزیابی مدل بر روی یک مجموعه داده."""
        self.student.eval()
        
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=description, ncols=100):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        # محاسبه معیارهای کلی (Overall Metrics)
        test_accuracy = accuracy_score(all_targets, all_preds)
        precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"\n--- Overall Metrics for {description} ---")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision (Macro Avg): {precision_macro:.4f}")
        print(f"Recall (Macro Avg): {recall_macro:.4f}")
        print(f"F1 Score (Macro Avg): {f1_macro:.4f}")
        
        # محاسبه معیارهای مربوط به هر کلاس (Class-specific Metrics)
        class_names = ['Real', 'Fake'] # فرض بر این است که 0=Real, 1=Fake
        
        print("\n--- Class-specific Metrics ---")
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        # معیارها برای کلاس 'Real' (لیبل 0)
        prec_real = precision_score(all_targets, all_preds, pos_label=0, zero_division=0)
        rec_real = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
        f1_real = f1_score(all_targets, all_preds, pos_label=0, zero_division=0)
        specificity_real = tn / (tn + fp) if (tn + fp) != 0 else 0.0

        print("Metrics for class 'Real' (label 0):")
        print(f"  - Precision: {prec_real:.4f}")
        print(f"  - Recall: {rec_real:.4f}")
        print(f"  - F1 Score: {f1_real:.4f}")
        print(f"  - Specificity: {specificity_real:.4f}")
        
        # معیارها برای کلاس 'Fake' (لیبل 1)
        prec_fake = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        rec_fake = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1_fake = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
        specificity_fake = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        
        print("Metrics for class 'Fake' (label 1):")
        print(f"  - Precision: {prec_fake:.4f}")
        print(f"  - Recall: {rec_fake:.4f}")
        print(f"  - F1 Score: {f1_fake:.4f}")
        print(f"  - Specificity: {specificity_fake:.4f}")
            
        # رسم ماتریس درهم‌ریختگی (Confusion Matrix)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix {description}')
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        if is_finetuned:
            plt.savefig(os.path.join(self.result_dir, 'confusion_matrix_after_finetune.png'))
        else:
            plt.savefig(os.path.join(self.result_dir, 'confusion_matrix_before_finetune.png'))
        plt.close()

        print("\n--- 30 Random Test Samples ---")
        indices = np.random.choice(len(all_targets), 30, replace=False)
        for i, idx in enumerate(indices):
            true_label = "Real" if all_targets[idx] == 0 else "Fake"
            pred_label = "Real" if all_preds[idx] == 0 else "Fake"
            print(f"Sample {i+1}: True Label -> {true_label}, Predicted Label -> {pred_label}")
            
        return test_accuracy

    def finetune(self):
        """
        اجرای استراتژی fine-tuning بر روی لایه‌های 'fc' و 'layer4' مدل.
        """
        print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # فریز کردن تمام پارامترها به جز 'fc' و 'layer4'
        for name, param in self.student.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.f_lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.ticket = False # غیرفعال کردن مکانیسم‌های خاص مدل اسپارس در حین fine-tuning
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.f_epochs} [Train]", ncols=100):
                images, targets = images.to(self.device), targets.to(self.device).float()
                optimizer.zero_grad()
                logits, _ = self.student(images)
                logits = logits.squeeze()
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_loss.update(loss.item(), images.size(0))
                meter_top1_train.update(prec1, images.size(0))

            val_acc = self.test(self.val_loader, description=f"Epoch {epoch+1}/{self.f_epochs} [Val]")
            
            print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving to {best_model_path}")
                torch.save(self.student.state_dict(), best_model_path)
        
        print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
        if os.path.exists(best_model_path):
            self.student.load_state_dict(torch.load(best_model_path))
        else:
            print("Warning: No best model was saved. The model from the last epoch will be used for testing.")

    def main(self):
        """
        اجرای کامل پایپ‌لاین: بارگذاری داده، ساخت مدل، تست اولیه، fine-tuning و تست نهایی.
        """
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        self.test(self.test_loader, "Initial Test", is_finetuned=False)
        
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        print("\n--- Testing AFTER fine-tuning with best model ---")
        self.test(self.test_loader, "Final Test", is_finetuned=True)

if __name__ == '__main__':
    
    # تنظیمات برنامه
    dataset_dir = '/kaggle/input/rvf10k' # مسیر دایرکتوری مجموعه داده
    # توجه: شما باید مسیر فایل checkpoint را به فایل صحیح خود تغییر دهید
    sparsed_student_ckpt_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
    dataset_mode = '140k' 
    result_dir = './results' # دایرکتوری برای ذخیره نتایج
    train_batch_size = 32
    test_batch_size = 64
    num_workers = 4
    pin_memory = True
    device = 'cuda'  # 'cuda' برای استفاده از GPU، 'cpu' برای استفاده از CPU
    f_epochs = 10    # تعداد دورهای (epoch) fine-tuning
    f_lr = 0.0001    # نرخ یادگیری (learning rate) برای fine-tuning

    pipeline = Test(dataset_dir=dataset_dir, 
                    sparsed_student_ckpt_path=sparsed_student_ckpt_path, 
                    dataset_mode=dataset_mode, 
                    result_dir=result_dir, 
                    train_batch_size=train_batch_size, 
                    test_batch_size=test_batch_size, 
                    num_workers=num_workers, 
                    pin_memory=pin_memory, 
                    device=device, 
                    f_epochs=f_epochs, 
                    f_lr=f_lr)
    pipeline.main()
