import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from utils import utils, loss, meter, scheduler


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class MobileNetV2WithFeatures(nn.Module):

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.feature_list = []
        
        layers_to_hook = ['features.1', 'features.6', 'features.13', 'features.17']
        
        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                layer.register_forward_hook(self.save_features_hook())

    def save_features_hook(self):
        def hook(module, input, output):
            self.feature_list.append(output)
        return hook

    def forward(self, x):
        self.feature_list = []
        output = self.model(x)
        return output, self.feature_list

class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch.lower()
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        self.resume = args.resume
        self.start_epoch = 0
        self.best_prec1 = 0
        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

        if self.dataset_mode == "hardfake":
            self.args.dataset_type = "hardfakevsrealfaces"
        elif self.dataset_mode == "rvf10k":
            self.args.dataset_type = "rvf10k"
        elif self.dataset_mode == "140k":
            self.args.dataset_type = "140k"
        elif self.dataset_mode == "200k":
            self.args.dataset_type = "200k"
        elif self.dataset_mode == "190k":
            self.args.dataset_type = "190k"
        elif self.dataset_mode == "330k":
            self.args.dataset_type = "330k"
        else:
             raise ValueError("dataset_mode not recognized")

        if self.arch not in ['resnet50', 'mobilenetv2']:
            raise ValueError("arch must be 'resnet50' or 'mobilenetv2'")

    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "train_logger.log"), "train_logger"
            )
            self.logger.info("train config:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "train_config.txt")
            )
            self.logger.info("--------- Train -----------")

    def setup_seed(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        seed = self.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        dataset_instance = Dataset_selector(
            dataset_mode=self.dataset_mode,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=True,
            dataset_dir=self.dataset_dir,
            rvf10k_train_csv=self.args.rvf10k_train_csv,
            rvf10k_valid_csv=self.args.rvf10k_valid_csv
        )
        self.train_loader = dataset_instance.loader_train
        self.val_loader = dataset_instance.loader_val
        self.test_loader = dataset_instance.loader_test
        if self.rank == 0:
            self.logger.info("Dataset has been loaded!")
            
    # <<< تغییر ۴: اصلاح کامل تابع build_model >>>
    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")

        # --- ساخت مدل معلم (Teacher Model) ---
        if self.arch == 'resnet50':
            teacher_model = ResNet_50_hardfakevsreal()
            if self.rank == 0:
                self.logger.info("Loading weights for custom ResNet-50 teacher...")
            ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_teacher.get('config_state_dict', ckpt_teacher)
            teacher_model.load_state_dict(state_dict, strict=True)
            if self.rank == 0:
                self.logger.info("Weights loaded successfully for custom ResNet-50.")

        elif self.arch == 'mobilenetv2':
            standard_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            num_ftrs = standard_model.classifier[1].in_features
            standard_model.classifier[1] = nn.Linear(num_ftrs, 1)
            teacher_model = MobileNetV2WithFeatures(standard_model)
            if self.rank == 0:
                self.logger.info("Loaded standard MobileNetV2 from torchvision and wrapped it for feature extraction.")
        
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        self.teacher = teacher_model.cuda()
        
        # --- ساخت مدل دانش‌آموز (Student Model) ---
        if self.rank == 0:
            self.logger.info("Building student model")
        
        if self.arch == 'resnet50':
            if self.dataset_mode == "hardfake":
                self.student = ResNet_50_sparse_hardfakevsreal()
            else:
                self.student = ResNet_50_sparse_rvf10k()
        elif self.arch == 'mobilenetv2':
            self.student = MobileNetV2_sparse_deepfake(
                gumbel_start_temperature=self.gumbel_start_temperature,
                gumbel_end_temperature=self.gumbel_end_temperature,
                num_epochs=self.num_epochs,
            )
        
        self.student = self.student.cuda()
        self.student = DDP(self.student, device_ids=[self.local_rank])

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().cuda()
        self.kd_loss = loss.KDLoss().cuda()
        self.rc_loss = loss.RCLoss().cuda()
        self.mask_loss = loss.MaskLoss().cuda()

    def define_optim(self):
        weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and "mask" not in p[0], self.student.module.named_parameters()))
        mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and "mask" in p[0], self.student.module.named_parameters()))
        self.optim_weight = torch.optim.Adamax(weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7)
        self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)
        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(self.optim_weight, T_max=self.lr_decay_T_max, eta_min=self.lr_decay_eta_min, last_epoch=-1, warmup_steps=self.warmup_steps, warmup_start_lr=self.warmup_start_lr)
        self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(self.optim_mask, T_max=self.lr_decay_T_max, eta_min=self.lr_decay_eta_min, last_epoch=-1, warmup_steps=self.warmup_steps, warmup_start_lr=self.warmup_start_lr)
        
    def resume_student_ckpt(self):
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.module.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.scheduler_student_weight.load_state_dict(ckpt_student["scheduler_student_weight"])
        self.scheduler_student_mask.load_state_dict(ckpt_student["scheduler_student_mask"])
        if self.rank == 0:
            self.logger.info("=> Continue from epoch {}...".format(self.start_epoch + 1))

    def save_student_ckpt(self, is_best, epoch):
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            ckpt_student = {
                "best_prec1": self.best_prec1,
                "start_epoch": epoch,
                "student": self.student.module.state_dict(),
                "optim_weight": self.optim_weight.state_dict(),
                "optim_mask": self.optim_mask.state_dict(),
                "scheduler_student_weight": self.scheduler_student_weight.state_dict(),
                "scheduler_student_mask": self.scheduler_student_mask.state_dict(),
            }
            
            if is_best:
                torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_best.pt"))
            torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))
        
    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def train(self):
        if self.rank == 0:
            self.logger.info(f"Starting training from epoch: {self.start_epoch + 1}")
        
        torch.cuda.empty_cache()
        self.teacher.eval()
        scaler = GradScaler()

        if self.resume:
            self.resume_student_ckpt()

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
            meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
            meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.train()
            if hasattr(self.student.module, 'ticket'):
                self.student.module.ticket = False
            
            if self.rank == 0:
                meter_oriloss.reset(); meter_kdloss.reset(); meter_rcloss.reset()
                meter_maskloss.reset(); meter_loss.reset(); meter_top1.reset()
                lr = self.optim_weight.state_dict()["param_groups"][0]["lr"] if epoch > 1 else self.warmup_start_lr
            
            if hasattr(self.student.module, 'update_gumbel_temperature'):
                self.student.module.update_gumbel_temperature(epoch)
            
            with tqdm(total=len(self.train_loader), ncols=100, disable=self.rank != 0) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                
                for images, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    images = images.cuda(self.local_rank, non_blocking=True)
                    targets = targets.cuda(self.local_rank, non_blocking=True).float()

                    with autocast():
                        logits_student, feature_list_student = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        with torch.no_grad():
                            logits_teacher, feature_list_teacher = self.teacher(images)
                            logits_teacher = logits_teacher.squeeze(1)

                        ori_loss = self.ori_loss(logits_student, targets)
                        kd_loss = (self.target_temperature**2) * self.kd_loss(logits_teacher, logits_student, self.target_temperature)
                        
                        rc_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                        if feature_list_student and feature_list_teacher and len(feature_list_student) == len(feature_list_teacher):
                            for i in range(len(feature_list_student)):
                                rc_loss = rc_loss + self.rc_loss(feature_list_student[i], feature_list_teacher[i])
                        
                        mask_loss = self.mask_loss(self.student.module)
                        
                        total_loss = ori_loss + self.coef_kdloss * kd_loss
                        if torch.numel(rc_loss) > 0 and feature_list_student and len(feature_list_student) > 0:
                             total_loss += self.coef_rcloss * rc_loss / len(feature_list_student)
                        total_loss += self.coef_maskloss * mask_loss

                    scaler.scale(total_loss).backward()
                    scaler.step(self.optim_weight)
                    scaler.step(self.optim_mask)
                    scaler.update()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum()
                    prec1 = 100. * correct / images.size(0)

                    dist.barrier()
                    reduced_loss = self.reduce_tensor(total_loss.data)
                    reduced_prec1 = self.reduce_tensor(prec1.data)
                    
                    if self.rank == 0:
                        meter_loss.update(reduced_loss.item(), images.size(0))
                        meter_top1.update(reduced_prec1.item(), images.size(0))
                        _tqdm.set_postfix(loss="{:.4f}".format(meter_loss.avg), train_acc="{:.2f}".format(meter_top1.avg))
                        _tqdm.update(1)

            self.scheduler_student_weight.step()
            self.scheduler_student_mask.step()

            if self.rank == 0:
                self.student.eval()
                if hasattr(self.student.module, 'ticket'):
                    self.student.module.ticket = True
                
                meter_val_top1 = meter.AverageMeter("ValAcc@1", ":6.2f")
                with torch.no_grad():
                    for images, targets in self.val_loader:
                        images = images.cuda(self.local_rank, non_blocking=True)
                        targets = targets.cuda(self.local_rank, non_blocking=True).float()

                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum()
                        prec1 = 100. * correct / images.size(0)
                        meter_val_top1.update(prec1.item(), images.size(0))

                self.logger.info(f"[Validation] Epoch {epoch}: Val_Acc {meter_val_top1.avg:.2f}%")
                
                is_best = meter_val_top1.avg > self.best_prec1
                if is_best:
                    self.best_prec1 = meter_val_top1.avg
                
                self.save_student_ckpt(is_best, epoch)
                self.writer.add_scalar("val/acc/top1", meter_val_top1.avg, global_step=epoch)
        
        if self.rank == 0:
            self.logger.info("Train finished!")

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
