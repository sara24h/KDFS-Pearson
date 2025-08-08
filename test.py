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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test and Fine-Tune Sparse ResNet-50 on Hardfake Dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory for data loading')
    parser.add_argument('--arch', type=str, default='ResNet_50', help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sparsed_student_ckpt_path', type=str, default='checkpoint.pth', help='Path to save/load sparse student checkpoint')
    parser.add_argument('--dataset_mode', type=str, default='hardfake', choices=['hardfake', 'rvf10k', '140k', '200k', '330k'], help='Dataset mode')
    parser.add_argument('--f_epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    parser.add_argument('--f_lr', type=float, default=0.001, help='Learning rate for fine-tuning')
    return parser.parse_args()

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.train_batch_size = args.train_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.f_epochs = args.f_epochs
        self.f_lr = args.f_lr

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print("==> Loading dataset..")
        try:
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
            elif self.dataset_mode == 'rvf10k':
                train_csv = os.path.join(self.dataset_dir, 'train.csv')
                valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"CSV files not found: {train_csv}, {valid_csv}")
            elif self.dataset_mode == '140k':
                test_csv = os.path.join(self.dataset_dir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
            elif self.dataset_mode == '200k':
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
            elif self.dataset_mode == '330k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

            dataset = Dataset_selector(
                dataset_mode=self.dataset_mode,
                hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv') if self.dataset_mode == 'hardfake' else None,
                hardfake_root_dir=self.dataset_dir if self.dataset_mode == 'hardfake' else None,
                rvf10k_train_csv=os.path.join(self.dataset_dir, 'train.csv') if self.dataset_mode == 'rvf10k' else None,
                rvf10k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv') if self.dataset_mode == 'rvf10k' else None,
                rvf10k_root_dir=self.dataset_dir if self.dataset_mode == 'rvf10k' else None,
                realfake140k_train_csv=os.path.join(self.dataset_dir, 'train.csv') if self.dataset_mode == '140k' else None,
                realfake140k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv') if self.dataset_mode == '140k' else None,
                realfake140k_test_csv=os.path.join(self.dataset_dir, 'test.csv') if self.dataset_mode == '140k' else None,
                realfake140k_root_dir=self.dataset_dir if self.dataset_mode == '140k' else None,
                realfake200k_train_csv=os.path.join(self.dataset_dir, 'train_labels.csv') if self.dataset_mode == '200k' else None,
                realfake200k_val_csv=os.path.join(self.dataset_dir, 'val_labels.csv') if self.dataset_mode == '200k' else None,
                realfake200k_test_csv=os.path.join(self.dataset_dir, 'test_labels.csv') if self.dataset_mode == '200k' else None,
                realfake200k_root_dir=os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset/my_real_vs_ai_dataset') if self.dataset_mode == '200k' else None,
                realfake330k_root_dir=self.dataset_dir if self.dataset_mode == '330k' else None,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False
            )

            self.train_loader = dataset.loader_train
            self.test_loader = dataset.loader_test
            print(f"{self.dataset_mode} dataset loaded! Train batches: {len(self.train_loader)}, Test batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self, fine_tuned=True):
        print(f"==> Building {'fine-tuned' if fine_tuned else 'non-fine-tuned'} student model..")
        try:
            model = ResNet_50_sparse_hardfakevsreal()
            if fine_tuned:
                print(f"Loading fine-tuned sparse student model for dataset mode: {self.dataset_mode}")
                if os.path.exists(self.sparsed_student_ckpt_path):
                    ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
                    state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
                    try:
                        model.load_state_dict(state_dict, strict=True)
                    except RuntimeError as e:
                        print(f"State dict loading failed with strict=True: {str(e)}")
                        print("Trying with strict=False to identify mismatched keys...")
                        model.load_state_dict(state_dict, strict=False)
                        print("Loaded with strict=False; check for missing or unexpected keys.")
                else:
                    print(f"No checkpoint found at {self.sparsed_student_ckpt_path}. Starting with fresh model for fine-tuning.")
            else:
                print(f"Using non-fine-tuned model with random or pre-trained weights for dataset mode: {self.dataset_mode}")
            
            model.to(self.device)
            print(f"Model loaded on {self.device}")
            return model
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self, model, test_name):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        model.eval()
        model.ticket = True
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc=f"Test ({test_name})") as _tqdm:
                    for images, targets in self.test_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        logits_student, _ = model(images)
                        logits_student = logits_student.squeeze()
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100.0 * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            print(f"[Test] Dataset: {self.dataset_mode}, Mode: {test_name}, Prec@1: {meter_top1.avg:.2f}%")

            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = get_flops_and_params(args=self.args)
            print(
                f"[{test_name}] Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
                f"Params reduction: {Params_reduction:.2f}%"
            )
            print(
                f"[{test_name}] Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"Flops reduction: {Flops_reduction:.2f}%"
            )
        except Exception as e:
            print(f"Error during testing ({test_name}): {str(e)}")
            raise

    def finetune(self, model):
        print(f"==> Starting fine-tuning for {self.f_epochs} epochs with learning rate {self.f_lr}..")
        try:
            model.train()
            model.ticket = True
            optimizer = torch.optim.Adam(model.parameters(), lr=self.f_lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            for epoch in range(self.f_epochs):
                meter_loss = meter.AverageMeter("Loss", ":6.4f")
                meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
                
                with tqdm(total=len(self.train_loader), ncols=100, desc=f"Epoch {epoch+1}/{self.f_epochs}") as _tqdm:
                    for images, targets in self.train_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        optimizer.zero_grad()
                        logits_student, _ = model(images)
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

                print(f"[Fine-Tune Epoch {epoch+1}] Loss: {meter_loss.avg:.4f}, Acc@1: {meter_top1.avg:.2f}%")

            # Save the fine-tuned model
            torch.save({"student": model.state_dict()}, self.sparsed_student_ckpt_path)
            print(f"Fine-tuned model saved to {self.sparsed_student_ckpt_path}")
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            raise

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            self.dataload()

            # Fine-tune the model
            model_ft = self.build_model(fine_tuned=False)  # Start with non-fine-tuned model
            self.finetune(model_ft)

            # Test without fine-tuning
            model_no_ft = self.build_model(fine_tuned=False)
            self.test(model_no_ft, "Without Fine-Tuning")

            # Test with fine-tuning
            model_ft = self.build_model(fine_tuned=True)
            self.test(model_ft, "With Fine-Tuning")
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    args = parse_args()
    tester = Test(args)
    tester.main()
