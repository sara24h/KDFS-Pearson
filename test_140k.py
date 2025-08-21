import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import random
from datetime import datetime
import torch.multiprocessing as mp
import logging

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch  
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir
        self.new_dataset_dir = getattr(args, 'new_dataset_dir', None)
        self.f_lr = args.f_lr
        self.f_epochs = args.f_epochs
        self.gpu = args.gpu
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.seed = args.seed if hasattr(args, 'seed') else 42

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    def setup_seed(self):
        self.seed = self.seed + self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.logger = logging.getLogger("train_logger")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.result_dir, "train_logger.log"))
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.info("Fine-tune config:")
            self.logger.info(str(vars(self.args)))

    def dataload(self):
        if self.rank == 0:
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
            'ddp': True
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

        if self.rank == 0:
            print("Overriding transforms to use consistent 140k normalization stats for all datasets.")
        dataset_manager.loader_train.dataset.transform = transform_train_140k
        dataset_manager.loader_val.dataset.transform = transform_val_test_140k
        dataset_manager.loader_test.dataset.transform = transform_val_test_140k

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        if self.rank == 0:
            print(f"All loaders for '{self.dataset_mode}' are now configured with 140k normalization.")

        if self.new_dataset_dir:
            if self.rank == 0:
                print("==> Loading new test dataset...")
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir,
                'ddp': True
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_dataset_manager.loader_test.dataset.transform = transform_val_test_140k
            self.new_test_loader = new_dataset_manager.loader_test
            if self.rank == 0:
                print(f"New test dataset loader configured with 140k normalization.")

    def build_model(self):
        if self.rank == 0:
            print(f"==> Building student model: {self.arch}...")
        
        if self.arch == 'resnet50_sparse':
            self.student = ResNet_50_sparse_hardfakevsreal()
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}. Supported: 'resnet50_sparse'")
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        if self.rank == 0:
            print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        torch.cuda.set_device(self.gpu)
        self.student = self.student.cuda(self.gpu)
        self.student = DDP(self.student, device_ids=[self.gpu])
        if self.rank == 0:
            print(f"Model loaded on GPU {self.gpu}")

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def compute_metrics(self, loader, description="Test", print_metrics=True, save_confusion_matrix=True):
        all_preds = []
        all_targets = []
        sample_info = []
        
        self.student.eval()
        self.student.module.ticket = True
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100, disable=self.rank != 0)):
                images = images.cuda(self.gpu, non_blocking=True)
                targets = targets.cuda(self.gpu, non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                batch_size = images.size(0)
                for i in range(batch_size):
                    try:
                        img_path = loader.dataset.samples[batch_idx * loader.batch_size + i][0]
                    except (AttributeError, IndexError):
                        img_path = f"Sample_{batch_idx * loader.batch_size + i}"
                    sample_info.append({
                        'id': img_path,
                        'true_label': targets[i].item(),
                        'pred_label': preds[i].item()
                    })
        
        all_preds_gather = torch.tensor(all_preds).cuda(self.gpu)
        all_targets_gather = torch.tensor(all_targets).cuda(self.gpu)
        dist.all_reduce(all_preds_gather)
        dist.all_reduce(all_targets_gather)
        all_preds = all_preds_gather.cpu().numpy()
        all_targets = all_targets_gather.cpu().numpy()

        accuracy = 100.0 * np.sum(all_preds == all_targets) / len(all_targets) if len(all_targets) > 0 else 0
        precision = precision_score(all_targets, all_preds, average='binary') if len(all_targets) > 0 else 0
        recall = recall_score(all_targets, all_preds, average='binary') if len(all_targets) > 0 else 0
        
        precision_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1]) if len(all_targets) > 0 else [0, 0]
        recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1]) if len(all_targets) > 0 else [0, 0]
        
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if print_metrics and self.rank == 0:
            print(f"[{description}] Overall Metrics:")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Specificity: {specificity_real:.4f}")
            
            print(f"\n[{description}] Per-Class Metrics:")
            print(f"Class Real (0):")
            print(f"  Precision: {precision_per_class[0]:.4f}")
            print(f"  Recall: {recall_per_class[0]:.4f}")
            print(f"  Specificity: {specificity_real:.4f}")
            print(f"Class Fake (1):")
            print(f"  Precision: {precision_per_class[1]:.4f}")
            print(f"  Recall: {recall_per_class[1]:.4f}")
            print(f"  Specificity: {specificity_fake:.4f}")
        
        classes = ['Real', 'Fake']
        
        if save_confusion_matrix and self.rank == 0:
            print(f"\n[{description}] Confusion Matrix:")
            print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
            print(f"{'Actual Real':>10} {cm[0,0]:>15} {cm[0,1]:>15}")
            print(f"{'Actual Fake':>10} {cm[1,0]:>15} {cm[1,1]:>15}")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix - {description}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            sanitized_description = description.lower().replace(" ", "_").replace("/", "_")
            plot_path = os.path.join(self.result_dir, f'confusion_matrix_{sanitized_description}.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            print(f"Confusion matrix saved to: {plot_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity_real,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'specificity_per_class': [specificity_real, specificity_fake],
            'confusion_matrix': cm,
            'sample_info': sample_info
        }

    def display_samples(self, sample_info, description="Test", num_samples=30):
        if self.rank == 0:
            print(f"\n[{description}] Displaying first {num_samples} test samples:")
            print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
            print("-" * 80)
            for i, sample in enumerate(sample_info[:num_samples]):
                true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
                pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
                print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def finetune(self):
        if self.rank == 0:
            print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
        
        for name, param in self.student.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                if self.rank == 0:
                    print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=self.args.f_lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.BCEWithLogitsLoss()
        
        self.student.module.ticket = False
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        scaler = GradScaler()

        for epoch in range(self.f_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.f_epochs} [Train]", ncols=100, disable=self.rank != 0) as pbar:
                for images, targets in self.train_loader:
                    images = images.cuda(self.gpu)
                    targets = targets.cuda(self.gpu).float()
                    optimizer.zero_grad()
                    
                    with autocast():
                        logits, _ = self.student(images)
                        logits = logits.squeeze()
                        loss = criterion(logits, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    train_correct += correct
                    train_total += images.size(0)
                    train_loss += loss.item() * images.size(0)
                    
                    pbar.update(1)

            train_loss_tensor = torch.tensor(train_loss).cuda(self.gpu)
            train_correct_tensor = torch.tensor(train_correct).cuda(self.gpu)
            train_total_tensor = torch.tensor(train_total).cuda(self.gpu)
            dist.reduce(train_loss_tensor, dst=0)
            dist.reduce(train_correct_tensor, dst=0)
            dist.reduce(train_total_tensor, dst=0)
            if self.rank == 0:
                avg_train_loss = train_loss_tensor.item() / train_total_tensor.item()
                train_acc = 100.0 * train_correct_tensor.item() / train_total_tensor.item()
            else:
                avg_train_loss = 0
                train_acc = 0

            val_metrics = self.compute_metrics(self.val_loader, description=f"Epoch_{epoch+1}/{self.f_epochs} Val", print_metrics=False, save_confusion_matrix=False)
            val_acc = val_metrics['accuracy']
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            scheduler.step()

            if self.rank == 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving to {best_model_path}")
                torch.save(self.student.module.state_dict(), best_model_path)
        
        dist.barrier()
        if self.rank == 0:
            print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path)
                self.student.module.load_state_dict(state_dict)
        dist.barrier()

        final_test_metrics = self.compute_metrics(self.test_loader, description="Final_Test", print_metrics=True, save_confusion_matrix=True)
        if self.rank == 0:
            print(f"\nFinal Test Metrics after Fine-tuning:")
            print(f"Accuracy: {final_test_metrics['accuracy']:.2f}%")
            print(f"Precision: {final_test_metrics['precision']:.4f}")
            print(f"Recall: {final_test_metrics['recall']:.4f}")
            print(f"Specificity: {final_test_metrics['specificity']:.4f}")
            print(f"\nPer-Class Metrics:")
            print(f"Class Real (0):")
            print(f"  Precision: {final_test_metrics['precision_per_class'][0]:.4f}")
            print(f"  Recall: {final_test_metrics['recall_per_class'][0]:.4f}")
            print(f"  Specificity: {final_test_metrics['specificity_per_class'][0]:.4f}")
            print(f"Class Fake (1):")
            print(f"  Precision: {final_test_metrics['precision_per_class'][1]:.4f}")
            print(f"  Recall: {final_test_metrics['recall_per_class'][1]:.4f}")
            print(f"  Specificity: {final_test_metrics['specificity_per_class'][1]:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
            print(f"{'Actual Real':>10} {final_test_metrics['confusion_matrix'][0,0]:>15} {final_test_metrics['confusion_matrix'][0,1]:>15}")
            print(f"{'Actual Fake':>10} {final_test_metrics['confusion_matrix'][1,0]:>15} {final_test_metrics['confusion_matrix'][1,1]:>15}")

    def main(self):
        self.setup_seed()
        self.result_init()
        self.dataload()
        self.build_model()
        
        if self.rank == 0:
            print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
            print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics['sample_info'], "Initial Test", num_samples=30)
        
        if self.rank == 0:
            print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        if self.rank == 0:
            print("\n--- Testing AFTER fine-tuning with best model ---")
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test", print_metrics=False)
        self.display_samples(final_metrics['sample_info'], "Final Test", num_samples=30)
        
        if self.new_test_loader:
            if self.rank == 0:
                print("\n--- Testing on NEW dataset ---")
            new_metrics = self.compute_metrics(self.new_test_loader, "New_Dataset_Test")
            self.display_samples(new_metrics['sample_info'], "New Dataset Test", num_samples=30)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    print(f"Use GPU: {args.gpu} for processing")
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=gpu)
    test = Test(args)
    test.main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune model for real vs fake image detection')
    
    # Dataset selection
    parser.add_argument('--dataset_mode', type=str, default='140k', 
                        choices=['hardfake', 'rvf10k', '140k', '200k', '190k', '330k'],
                        help='Dataset mode to use (e.g., 140k, 200k)')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the dataset directory')
    
    # Model selection
    parser.add_argument('--arch', type=str, default='resnet50_sparse', 
                        choices=['resnet50_sparse'], 
                        help='Model architecture to use')
    parser.add_argument('--sparsed_student_ckpt_path', type=str, required=True, 
                        help='Path to the pre-trained sparse student checkpoint')
    
    # Other required arguments
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', 
                        help='Pin memory for data loaders')
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help='Batch size for testing')
    parser.add_argument('--result_dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--new_dataset_dir', type=str, default=None, 
                        help='Optional new dataset directory for additional testing')
    parser.add_argument('--f_lr', type=float, default=0.001, 
                        help='Learning rate for fine-tuning')
    parser.add_argument('--f_epochs', type=int, default=10, 
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.set_start_method('fork', force=True)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
