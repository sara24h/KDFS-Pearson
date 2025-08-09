
import os
import time
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse 

class FaceDataset(Dataset):
    """یک کلاس دیتاست عمومی برای خواندن تصاویر از دیتافریم."""
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
        label_val = self.data['label'].iloc[idx]
        label = self.label_map[label_val]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector(Dataset):
    """
    کلاس اصلی شما برای انتخاب و بارگذاری دیتاست.
    این کلاس طبق خواسته شما دست‌نخورده باقی مانده است.
    """
    def __init__(
        self,
        dataset_mode,
        hardfake_csv_file=None, hardfake_root_dir=None,
        rvf10k_train_csv=None, rvf10k_valid_csv=None, rvf10k_root_dir=None,
        # ... سایر پارامترهای دیتاست‌ها
        train_batch_size=32, eval_batch_size=32,
        num_workers=8, pin_memory=True, ddp=False
    ):
        # این کد فرض می‌کند که منطق داخلی شما برای بارگذاری دیتافریم‌ها صحیح است
        # و فقط به خروجی‌های آن (loader_train, loader_val, loader_test) نیاز داریم.
        if dataset_mode == 'rvf10k':
            train_data = pd.read_csv(rvf10k_train_csv)
            valid_data = pd.read_csv(rvf10k_valid_csv)
            root_dir = rvf10k_root_dir
            # ... منطق ساخت مسیرهای تصویر شما
            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'
                img_name = row['id']
                if not isinstance(img_name, str): img_name = str(img_name)
                if not img_name.endswith('.jpg'): img_name += '.jpg'
                return os.path.join('rvf10k', split, folder, img_name)
            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data['images_id'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)
            val_data, test_data = train_test_split(valid_data, test_size=0.5, stratify=valid_data['label'], random_state=42)
        else:
            raise NotImplementedError(f"Data loading for {dataset_mode} is not fully implemented in this example.")
            
        # تعریف transform پیش‌فرض بر اساس مد دیتاست (منشأ باگ قبلی)
        if dataset_mode == 'rvf10k':
            mean, std = (0.5212, 0.4260, 0.3811), (0.2486, 0.2238, 0.2211)
            image_size = (256, 256)
        # ... سایر if ها
        
        default_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
        train_dataset = FaceDataset(train_data, root_dir, transform=default_transform, img_column='images_id')
        val_dataset = FaceDataset(val_data, root_dir, transform=default_transform, img_column='images_id')
        test_dataset = FaceDataset(test_data, root_dir, transform=default_transform, img_column='images_id')
        
        self.loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_val = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_test = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

class meter:
    class AverageMeter:
        def __init__(self, name, fmt=':f'): self.name, self.fmt, self.val, self.avg, self.sum, self.count = name, fmt, 0, 0, 0, 0
        def update(self, val, n=1): self.val, self.sum, self.count = val, self.sum + val * n, self.count + n; self.avg = self.sum / self.count

# فرض می‌کنیم مدل شما به این شکل است
ResNet_50_sparse_hardfakevsreal = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1))


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
            
        self.train_loader, self.val_loader, self.test_loader, self.student = None, None, None, None

    def dataload(self):
        # ۱. ساخت لودرها با تنظیمات پیش‌فرض از Dataset_selector
        print("==> Step 1: Loading initial datasets from Dataset_selector...")
        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.args.test_batch_size, # استفاده از یک batch_size برای سادگی
            'eval_batch_size': self.args.test_batch_size,
            'num_workers': self.num_workers, 'pin_memory': self.pin_memory, 'ddp': False,
            'rvf10k_train_csv': os.path.join(self.dataset_dir, 'train.csv'),
            'rvf10k_valid_csv': os.path.join(self.dataset_dir, 'valid.csv'),
            'rvf10k_root_dir': self.dataset_dir
        }
        dataset_manager = Dataset_selector(**params)

        # ۲. تعریف Transform صحیح (مبتنی بر 140k) برای اعمال هماهنگ
        print("==> Step 2: Defining the CORRECT and CONSISTENT 140k transform...")
        image_size = (256, 256)
        
        # Transform برای آموزش (با کمی افزایش داده)
        transform_140k_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5207, 0.4258, 0.3806], std=[0.2490, 0.2239, 0.2212])
        ])
        
        # Transform برای ارزیابی (بدون افزایش داده)
        transform_140k_eval = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5207, 0.4258, 0.3806], std=[0.2490, 0.2239, 0.2212])
        ])
        
        # ۳. بازنویسی Transform برای تمام لودرها (رفع قطعی باگ)
        print("==> Step 3: Overwriting transforms for train, val, and test loaders...")
        dataset_manager.loader_train.dataset.transform = transform_140k_train
        dataset_manager.loader_val.dataset.transform = transform_140k_eval
        dataset_manager.loader_test.dataset.transform = transform_140k_eval

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test

        print("\nSUCCESS: The normalization bug is fixed. All loaders are now consistent.")

    def build_model(self):
        print("==> Building student model..")
        # شما باید مدل واقعی خود را اینجا ایمپورت و استفاده کنید
        self.student = ResNet_50_sparse_hardfakevsreal()
        if not os.path.exists(self.sparsed_student_ckpt_path):
             print(f"Warning: Checkpoint file not found: {self.sparsed_student_ckpt_path}. Using a dummy model.")
        else:
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
            state_dict = ckpt_student.get("student", ckpt_student)
            self.student.load_state_dict(state_dict, strict=False)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def test(self, loader, description="Test"):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        self.student.eval()
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=description, ncols=100):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits_student = self.student(images)
                logits_student = logits_student.squeeze()
                preds = (torch.sigmoid(logits_student) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        print(f"[{description}] Final Prec@1: {meter_top1.avg:.2f}%")
        return meter_top1.avg

    def finetune(self):
        print("==> Fine-tuning by unfreezing fc and layer4...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # (منطق فریز و آنفریز کردن لایه‌ها)
        # ...
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.args.f_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.args.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs} [Train]"):
                images, targets = images.to(self.device), targets.to(self.device).float()
                optimizer.zero_grad()
                logits = self.student(images).squeeze()
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                meter_loss.update(loss.item(), images.size(0))
            
            val_acc = self.test(self.val_loader, description=f"Epoch {epoch+1} [Val]")
            
            print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best model found. Saving to {best_model_path}")
                torch.save(self.student.state_dict(), best_model_path)
                
        print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
        if os.path.exists(best_model_path):
            self.student.load_state_dict(torch.load(best_model_path))

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        self.test(self.val_loader, description="Validation Set (Before FT)")
        self.test(self.test_loader, description="Test Set (Before FT)")
        
        # با توجه به نتایج، این بخش احتمالاً غیرضروری است
        if self.args.do_finetune:
             print("\n--- Starting fine-tuning ---")
             self.finetune()
             print("\n--- Testing AFTER fine-tuning ---")
             self.test(self.test_loader, description="Test Set (After FT)")
        else:
             print("\n--- Skipping fine-tuning as it's not needed for this dataset ---")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_mode', type=str, default='rvf10k')
    parser.add_argument('--dataset_dir', type=str, default='/kaggle/input/rvf10k')
    parser.add_argument('--sparsed_student_ckpt_path', type=str, default='/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--f_lr', type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument('--f_epochs', type=int, default=5, help="Epochs for fine-tuning")
    parser.add_argument('--do_finetune', type=bool, default=False, help="Set to True to run fine-tuning")

    args = parser.parse_args()

    # ساخت و اجرای کلاس اصلی
    tester = Test(args)
    tester.main()
