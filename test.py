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

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            # Verify dataset paths
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
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
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
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self, fine_tuned=True):
        print(f"==> Building {'fine-tuned' if fine_tuned else 'non-fine-tuned'} student model..")
        try:
            model = ResNet_50_sparse_hardfakevsreal()
            if fine_tuned:
                print(f"Loading fine-tuned sparse student model for dataset mode: {self.dataset_mode}")
                if not os.path.exists(self.sparsed_student_ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
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
                print(f"Using non-fine-tuned model with random or pre-trained weights for dataset mode: {self.dataset_mode}")
                # Optionally, load pre-trained ImageNet weights here if available
                # For example: model.load_state_dict(torch.hub.load_state_dict_from_url('url_to_imagenet_weights'), strict=False)
            
            model.to(self.device)
            print(f"Model loaded on {self.device}")
            return model
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self, model, test_name):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        model.eval()
        model.ticket = True  # Enable ticket mode for sparse model
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

            # Calculate FLOPs and parameters
            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = get_flops_and_params(args=self.args, model=model)
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

    def main(self):
        print(f"Starting test pipeline with dataset mode: {self.dataset_mode}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            # Load dataset
            self.dataload()

            # Test without fine-tuning
            model_no_ft = self.build_model(fine_tuned=False)
            self.test(model_no_ft, "Without Fine-Tuning")

            # Test with fine-tuning
            model_ft = self.build_model(fine_tuned=True)
            self.test(model_ft, "With Fine-Tuning")
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise
