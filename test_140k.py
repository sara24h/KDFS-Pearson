import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# فرض بر این است که این فایل‌ها در مسیر درست پروژه شما قرار دارند
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.train_batch_size = args.train_batch_size
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
        print("==> Building student model...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def test(self, loader, description="Test"):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        self.student.eval()
        self.student.ticket = True
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=description, ncols=100):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        print(f"[{description}] Final Prec@1: {meter_top1.avg:.2f}%")
        return meter_top1.avg

    def finetune(self):
        print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        for name, param in self.student.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.args.f_lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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

            val_acc = self.test(self.val_loader, description=f"Epoch {epoch+1}/{self.args.f_epochs} [Val]")
            
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
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        self.test(self.test_loader, "Initial Test")
        
        print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        print("\n--- Testing AFTER fine-tuning with best model ---")
        self.test(self.test_loader, "Final Test")
