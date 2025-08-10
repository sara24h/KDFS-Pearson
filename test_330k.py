import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # سرکوب warningهای TF/CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force GPU device

import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter

# واردات برای تیونینگ
from ray import tune, train
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import Checkpoint
import ray

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
        self.new_dataset_dir = getattr(args, 'new_dataset_dir', None)

        # چک CUDA بدون raise
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            self.device = 'cpu'
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    def dataload(self, batch_size=None):
        print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_330k = [0.4923, 0.4042, 0.3624]
        std_330k = [0.2446, 0.2198, 0.2141]

        transform_train_330k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_330k, std=std_330k),
        ])

        transform_val_test_330k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_330k, std=std_330k),
        ])

        if batch_size is None:
            batch_size = self.train_batch_size

        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': batch_size,
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

        print("Overriding transforms to use consistent 330k normalization stats for all datasets.")
        dataset_manager.loader_train.dataset.transform = transform_train_330k
        dataset_manager.loader_val.dataset.transform = transform_val_test_330k
        dataset_manager.loader_test.dataset.transform = transform_val_test_330k

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        print(f"All loaders for '{self.dataset_mode}' are now configured with 330k normalization.")

        if self.new_dataset_dir:
            print("==> Loading new test dataset...")
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_dataset_manager.loader_test.dataset.transform = transform_val_test_330k
            self.new_test_loader = new_dataset_manager.loader_test
            print(f"New test dataset loader configured with 330k normalization.")

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

    def compute_metrics(self, loader, description="Test", print_metrics=True, save_confusion_matrix=True):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        all_preds = []
        all_targets = []
        sample_info = []
        
        self.student.eval()
        self.student.ticket = True
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100)):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
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
                
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        accuracy = meter_top1.avg
        precision = precision_score(all_targets, all_preds, average='binary')
        recall = recall_score(all_targets, all_preds, average='binary')
        
        precision_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1])
        recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1])
        
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if print_metrics:
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
        
        cm = confusion_matrix(all_targets, all_preds)
        classes = ['Real', 'Fake']
        
        if save_confusion_matrix:
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
        print(f"\n[{description}] Displaying first {num_samples} test samples:")
        print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
        print("-" * 80)
        for i, sample in enumerate(sample_info[:num_samples]):
            true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
            pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
            print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def tune_finetune(self):
        print("==> Starting hyperparameter tuning for fine-tuning...")
        
        ray.init(num_gpus=1, ignore_reinit_error=True)
        
        config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([8, 16, 32, 64]),
            "epochs": tune.choice([5, 10, 15, 20]),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "unfreeze_layers": tune.choice([['fc'], ['fc', 'layer4'], ['fc', 'layer3', 'layer4']])
        }
        
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            grace_period=3,
            reduction_factor=2
        )
        
        search_alg = OptunaSearch(metric="val_acc", mode="max")
        
        trainable_with_params = tune.with_parameters(train_function, args=self.args)
        
        trainable_with_resources = tune.with_resources(
            trainable_with_params,
            resources={"cpu": 2, "gpu": 1}
        )
        
        tuner = Tuner(
            trainable_with_resources,
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=10,
                scheduler=scheduler,
                search_alg=search_alg
            ),
            run_config=train.RunConfig(
                name=f"fine_tune_{self.dataset_mode}",
                stop={"val_acc": 99.0}
            )
        )
        
        results = tuner.fit()
        
        best_result = results.get_best_result(metric="val_acc", mode="max")
        best_config = best_result.config
        print("Best hyperparameters found:", best_config)
        
        if best_result.checkpoint:
            checkpoint_dict = best_result.checkpoint.to_dict()
            if self.student is None:
                self.build_model()
            self.student.load_state_dict(checkpoint_dict["model_state"])
            print("Loaded best model from checkpoint for final testing.")
        
        return best_config

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        
        print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics['sample_info'], "Initial Test", num_samples=30)
        
        print("\n--- Starting tuned fine-tuning ---")
        best_config = self.tune_finetune()
        
        print("\n--- Testing AFTER fine-tuning with best model ---")
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test", print_metrics=False)
        self.display_samples(final_metrics['sample_info'], "Final Test", num_samples=30)
        
        if self.new_test_loader:
            print("\n--- Testing on NEW dataset ---")
            new_metrics = self.compute_metrics(self.new_test_loader, "New_Dataset_Test")
            self.display_samples(new_metrics['sample_info'], "New Dataset Test", num_samples=30)

def train_function(config, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not torch.cuda.is_available():
        print("Warning: CUDA not available in this worker, using CPU.")
    
    test_instance = Test(args)
    
    test_instance.dataload(batch_size=config["batch_size"])
    
    test_instance.build_model()
    
    for name, param in test_instance.student.named_parameters():
        param.requires_grad = any(layer in name for layer in config["unfreeze_layers"])
        if param.requires_grad:
            print(f"Unfreezing for training: {name}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, test_instance.student.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    test_instance.student.ticket = False
    
    best_val_acc = 0.0
    best_model_path = os.path.join(args.result_dir, f'finetuned_model_best_{args.dataset_mode}.pth')

    for epoch in range(config["epochs"]):
        test_instance.student.train()
        meter_loss = meter.AverageMeter("Loss", ":6.4f")
        meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
        
        for images, targets in tqdm(test_instance.train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", ncols=100):
            images, targets = images.to(test_instance.device), targets.to(test_instance.device).float()
            optimizer.zero_grad()
            logits, _ = test_instance.student(images)
            logits = logits.squeeze()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == targets).sum().item()
            prec1 = 100.0 * correct / images.size(0)
            meter_loss.update(loss.item(), images.size(0))
            meter_top1_train.update(prec1, images.size(0))

        val_metrics = test_instance.compute_metrics(test_instance.val_loader, description=f"Epoch_{epoch+1}_{config['epochs']}_Val", print_metrics=False, save_confusion_matrix=False)
        val_acc = val_metrics['accuracy']
        
        print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        report_dict = {"val_acc": val_acc, "epoch": epoch}
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving checkpoint.")
            checkpoint = Checkpoint.from_dict({
                "model_state": test_instance.student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc
            })
            train.report(report_dict, checkpoint=checkpoint)
        else:
            train.report(report_dict)
    
    torch.save(test_instance.student.state_dict(), best_model_path)
