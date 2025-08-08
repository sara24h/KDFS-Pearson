import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
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
        self.result_dir = args.result_dir  # Directory to save finetuned model

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print("==> Loading datasets..")
        try:
            # Verify dataset paths
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
                mean = [0.5124, 0.4165, 0.3684]
                std = [0.2363, 0.2087, 0.2029]
            elif self.dataset_mode == 'rvf10k':
                train_csv = os.path.join(self.dataset_dir, 'train.csv')
                valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"CSV files not found: {train_csv}, {valid_csv}")
                mean = [0.5212, 0.4260, 0.3811]
                std = [0.2486, 0.2238, 0.2211]
            elif self.dataset_mode == '140k':
                test_csv = os.path.join(self.dataset_dir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
                mean = [0.5207, 0.4258, 0.3806]
                std = [0.2490, 0.2239, 0.2212]
            elif self.dataset_mode == '200k':
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
                mean = [0.4868, 0.3972, 0.3624]
                std = [0.2296, 0.2066, 0.2009]
            elif self.dataset_mode == '190k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
                mean = [0.4668, 0.3816, 0.3414]
                std = [0.2410, 0.2161, 0.2081]
            elif self.dataset_mode == '330k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
                mean = [0.4923, 0.4042, 0.3624]
                std = [0.2446, 0.2198, 0.2141]

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
            elif self.dataset_mode == '190k':
                dataset = Dataset_selector(
                    dataset_mode='190k',
                    realfake190k_root_dir=self.dataset_dir,
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

            # Define transforms with dataset-specific normalization for training and testing
            image_size = (256, 256) if self.dataset_mode in ['rvf10k', '140k', '190k', '200k', '330k'] else (300, 300)
            transform_train = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size[0], padding=8),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),  # Dataset-specific normalization
            ])
            transform_test = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),  # Dataset-specific normalization
            ])

            # Optional: Test loader with 330k normalization for comparison
            transform_test_330k = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4923, 0.4042, 0.3624],  # 330k mean
                    std=[0.2446, 0.2198, 0.2141]    # 330k std
                ),
            ])

            # Load train and test datasets
            train_dataset = dataset.loader_train.dataset
            train_dataset.transform = transform_train  # Apply dataset-specific normalization
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.test_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=dataset.loader_train.sampler if hasattr(dataset.loader_train, 'sampler') else None
            )
            test_dataset = dataset.loader_test.dataset
            test_dataset.transform = transform_test  # Apply dataset-specific normalization
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=dataset.loader_test.sampler if hasattr(dataset.loader_test, 'sampler') else None
            )
            # Optional: Test loader with 330k normalization
            test_dataset_330k = dataset.loader_test.dataset
            test_dataset_330k.transform = transform_test_330k  # Apply 330k normalization
            self.test_loader_330k = DataLoader(
                test_dataset_330k,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=dataset.loader_test.sampler if hasattr(dataset.loader_test, 'sampler') else None
            )
            print(f"{self.dataset_mode} datasets loaded! Train batches: {len(self.train_loader)}, "
                  f"Test batches: {len(self.test_loader)}, Test batches (330k norm): {len(self.test_loader_330k)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self):
        print("==> Building student model..")
        try:
            print(f"Loading sparse student model for dataset mode: {self.dataset_mode}")
            self.student = ResNet_50_sparse_hardfakevsreal()
            # Load checkpoint
            if not os.path.exists(self.sparsed_student_ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            try:
                self.student.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"State dict loading failed with strict=True: {str(e)}")
                print("Trying with strict=False to identify mismatched keys...")
                self.student.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False; check for missing or unexpected keys.")

            self.student.to(self.device)
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self, use_330k_norm=False):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        loader = self.test_loader_330k if use_330k_norm else self.test_loader
        desc = "Test (330k norm)" if use_330k_norm else "Test (dataset-specific norm)"

        self.student.eval()
        self.student.ticket = True  # Enable ticket mode for sparse model
        try:
            with torch.no_grad():
                with tqdm(total=len(loader), ncols=100, desc=desc) as _tqdm:
                    for images, targets in loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze()
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100.0 * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            print(f"[{desc}] Dataset: {self.dataset_mode}, Prec@1: {meter_top1.avg:.2f}%")

            # Calculate FLOPs and parameters
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
                f"Params reduction: {Params_reduction:.2f}%"
            )
            print(
                f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"Flops reduction: {Flops_reduction:.2f}%"
            )
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

    def finetune(self):
        print("==> Fine-tuning the model..")
        # Freeze all layers except layer4 and fc
        for name, param in self.student.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Set optimizer for updating unfrozen parameters
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.args.f_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Define loss function (for binary classification)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Set model to training mode
        self.student.ticket = False  # Disable ticket mode for training
        
        # Training loop
        for epoch in range(self.args.f_epochs):
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
            
            with tqdm(total=len(self.train_loader), ncols=100, desc=f"Epoch {epoch+1}/{self.args.f_epochs}") as _tqdm:
                for images, targets in self.train_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    
                    optimizer.zero_grad()
                    logits_student, _ = self.student(images)
                    logits_student = logits_student.squeeze()
                    loss = criterion(logits_student, targets)
                    loss.backward()
                    optimizer.step()
                    
                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100.0 * correct / images.size(0)
                    n = images.size(0)
                    meter_loss.update(loss.item(), n)
                    meter_top1.update(prec1, n)
                    
                    _tqdm.set_postfix(loss=f"{meter_loss.avg:.4f}", top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)
                    time.sleep(0.01)
            
            print(f"Epoch {epoch+1}/{self.args.f_epochs}, Loss: {meter_loss.avg:.4f}, Acc@1: {meter_top1.avg:.2f}%")
        
        # Save finetuned model
        save_path = os.path.join(self.args.result_dir, f'finetuned_model_{self.dataset_mode}.pth')
        torch.save(self.student.state_dict(), save_path)
        print(f"Finetuned model saved to {save_path}")

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            self.dataload()
            self.build_model()
            print("تست قبل از فاین‌تیونینگ (با نرمال‌سازی خاص دیتاست):")
            self.test(use_330k_norm=False)
            print("تست قبل از فاین‌تیونینگ (با نرمال‌سازی 330k):")
            self.test(use_330k_norm=True)
            self.finetune()
            print("تست بعد از فاین‌تیونینگ (با نرمال‌سازی خاص دیتاست):")
            self.test(use_330k_norm=False)
            print("تست بعد از فاین‌تیونینگ (با نرمال‌سازی 330k):")
            self.test(use_330k_norm=True)
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise
