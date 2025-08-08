import os
import time
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# فایل‌های خودتان را ایمپورت کنید
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
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.student = None

    def dataload(self):
        print("==> Loading datasets using Dataset_selector...")
        
        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.test_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False
        }
        
        # بر اساس دیتاست، مسیرهای مربوطه را به دیکشنری اضافه می‌کنیم
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
        # در متد dataload
        elif self.dataset_mode == '200k':
    # ساختن مسیر صحیح و عمیق‌تر به پوشه تصاویر
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

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        print(f"Train/Val loaders ready (using {self.dataset_mode} normalization).")

        print("==> Creating the test loader with 330k normalization...")
        image_size = (256, 256) if self.dataset_mode != 'hardfake' else (300, 300)
        
        transform_test_330k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4923, 0.4042, 0.3624], std=[0.2446, 0.2198, 0.2141]
            ),
        ])
        
        test_dataset_raw = dataset_manager.loader_test.dataset
        test_dataset_raw.transform = transform_test_330k
        
        self.test_loader = DataLoader(
            test_dataset_raw, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        print("Test loader is ready (using 330k normalization).")

    def build_model(self):
        print("==> Building student model..")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def test(self):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        desc = "Test (with 330k norm)"
        self.student.eval()
        self.student.ticket = True
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc=desc, ncols=100):
                # هر تانسور را جداگانه به دستگاه منتقل می‌کنیم
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits_student, _ = self.student(images)
                logits_student = logits_student.squeeze()
                preds = (torch.sigmoid(logits_student) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        print(f"[{desc}] Dataset: {self.dataset_mode}, Final Prec@1: {meter_top1.avg:.2f}%")

    def finetune(self):
        print("==> Fine-tuning using FEATURE EXTRACTOR strategy...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # استراتژی Feature Extractor: فقط لایه آخر (fc) آموزش داده می‌شود
        for name, param in self.student.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
                print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.args.f_lr,
            weight_decay=1e-4
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.ticket = False
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.args.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs} [Train]", ncols=100):
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

            self.student.eval()
            meter_top1_val = meter.AverageMeter("Val Acc@1", ":6.2f")
            with torch.no_grad():
                for images, targets in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs} [Val]", ncols=100):
                    images, targets = images.to(self.device), targets.to(self.device).float()
                    logits, _ = self.student(images)
                    logits = logits.squeeze()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100.0 * correct / images.size(0)
                    meter_top1_val.update(prec1, images.size(0))
            
            print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {meter_top1_val.avg:.2f}%")

            if meter_top1_val.avg > best_val_acc:
                best_val_acc = meter_top1_val.avg
                print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving to {best_model_path}")
                torch.save(self.student.state_dict(), best_model_path)
        
        print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
        if os.path.exists(best_model_path):
            self.student.load_state_dict(torch.load(best_model_path))
        else:
            print("Warning: No best model was saved. The model from the last epoch will be used for testing.")

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        print("\n--- Testing BEFORE fine-tuning ---")
        self.test()
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        print("\n--- Testing AFTER fine-tuning ---")
        self.test()
