import os
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

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    # dataload, build_model, compute_metrics, display_samples مثل قبل

    def tune_finetune(self):
        print("==> Starting hyperparameter tuning for fine-tuning...")
        
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
        
        # استفاده از with_parameters برای پاس args کوچک
        trainable_with_params = tune.with_parameters(train_function, args=self.args)
        
        tuner = Tuner(
            trainable_with_params,
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=50,
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
        
        # لود بهترین checkpoint به self.student
        if best_result.checkpoint:
            checkpoint_dict = best_result.checkpoint.to_dict()
            if self.student is None:
                self.build_model()  # اگر مدل لود نشده، اول بساز
            self.student.load_state_dict(checkpoint_dict["model_state"])
            print("Loaded best model from checkpoint for final testing.")
        
        return best_config

    def main(self):
        print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
        self.dataload()  # با default برای initial test
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

# فانکشن standalone خارج از کلاس
def train_function(config, args):
    """Standalone trainable function"""
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    # recreate Test instance داخل فانکشن
    test_instance = Test(args)
    
    # لود دیتا با batch_size از config
    test_instance.dataload(batch_size=config["batch_size"])
    
    # ساخت مدل
    test_instance.build_model()
    
    # Unfreeze لایه‌ها
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

        # Compute validation metrics
        val_metrics = test_instance.compute_metrics(test_instance.val_loader, description=f"Epoch_{epoch+1}_{config['epochs']}_Val", print_metrics=False, save_confusion_matrix=False)
        val_acc = val_metrics['accuracy']
        
        print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # گزارش به Ray Tune
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
    
    # ذخیره آخرین مدل
    torch.save(test_instance.student.state_dict(), best_model_path)
