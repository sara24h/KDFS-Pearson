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
        self.test_loader = None # فقط یک لودر تست خواهیم داشت
        self.student = None

    def dataload(self):
        print("==> Loading datasets using Dataset_selector...")
        
        # ۱. ساخت کارخانه دیتاست برای گرفتن لودر آموزش و دیتاست خام تست
        dataset_manager = Dataset_selector(
            dataset_mode=self.dataset_mode,
            realfake200k_root_dir=self.dataset_dir,
            realfake200k_train_csv=os.path.join(self.dataset_dir, 'train_labels.csv'),
            realfake200k_val_csv=os.path.join(self.dataset_dir, 'val_labels.csv'),
            realfake200k_test_csv=os.path.join(self.dataset_dir, 'test_labels.csv'),
            train_batch_size=self.args.f_train_batch_size,
            eval_batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=False
        )

        # ۲. لودر آموزش را برای فاین‌تیون برمی‌داریم (با نرمال‌سازی دیتاست هدف)
        self.train_loader = dataset_manager.loader_train
        print(f"Train loader for fine-tuning is ready (using {self.dataset_mode} normalization).")

        # ۳. لودر تست را با نرمال‌سازی ثابت 330k می‌سازیم
        print("==> Creating the test loader with 330k normalization...")
        image_size = (256, 256) if self.dataset_mode != 'hardfake' else (300, 300)
        
        transform_test_330k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4923, 0.4042, 0.3624],  # 330k mean
                std=[0.2446, 0.2198, 0.2141]   # 330k std
            ),
        ])
        
        # از دیتاست خام تست که توسط کارخانه ساخته شده استفاده می‌کنیم
        test_dataset_raw = dataset_manager.loader_test.dataset
        test_dataset_raw.transform = transform_test_330k # و فقط transform آن را عوض می‌کنیم
        
        self.test_loader = DataLoader(
            test_dataset_raw,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=None
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

    def test(self): # <--- متد تست ساده‌تر شد، دیگر ورودی ندارد
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        desc = "Test (with 330k norm)"

        self.student.eval()
        self.student.ticket = True
        
        with torch.no_grad():
            # همیشه از self.test_loader استفاده می‌کند
            for images, targets in tqdm(self.test_loader, desc=desc, ncols=100):
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
        # این متد بدون تغییر باقی می‌ماند
        print("==> Fine-tuning the model..")
        for name, param in self.student.named_parameters():
            param.requires_grad = 'layer4' in name or 'fc' in name

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.args.f_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.ticket = False
        
        for epoch in range(self.args.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
            
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs}", ncols=100):
                images = images.to(self.device)
                targets = targets.to(self.device).float()
                
                optimizer.zero_grad()
                logits_student, _ = self.student(images)
                logits_student = logits_student.squeeze()
                
                loss = criterion(logits_student, targets)
                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits_student) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                
                meter_loss.update(loss.item(), images.size(0))
                meter_top1.update(prec1, images.size(0))
                
            print(f"Epoch {epoch+1}/{self.args.f_epochs}, Loss: {meter_loss.avg:.4f}, Acc@1: {meter_top1.avg:.2f}%")
        
        save_path = os.path.join(self.result_dir, f'finetuned_model_{self.dataset_mode}.pth')
        torch.save(self.student.state_dict(), save_path)
        print(f"Finetuned model saved to {save_path}")

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        self.test() # <--- فقط یک نوع تست انجام می‌شود
        
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        print("\n--- Testing AFTER fine-tuning ---")
        self.test() # <--- دوباره همان تست برای مقایسه انجام می‌شود
