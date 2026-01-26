# encoding: utf-8
import math
import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + MultiStepLR"""
    def __init__(self, optimizer, milestones, gamma=0.1,
                 warmup_factor=1.0/3, warmup_iters=10,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers.")
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + CosineAnnealing"""
    def __init__(self, optimizer, max_epochs, warmup_epochs=10, eta_min=1e-7, last_epoch=-1):
        self.max_epochs = max_epochs - 1
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]


try:
    from .ranger import Ranger
    from .swa import SWA
except ImportError:
    Ranger, SWA = None, None


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if 'classifier' in key or 'arcface' in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.FC_LR_FACTOR

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=getattr(cfg.SOLVER, 'MOMENTUM', 0.9))
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger' and Ranger is not None:
        print('Using Ranger optimizer')
        optimizer = Ranger(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SWA' and SWA is not None:
        print('Using SWA optimizer (wraps SGD)')
        base_opt = torch.optim.SGD(params, momentum=getattr(cfg.SOLVER, 'MOMENTUM', 0.9))
        optimizer = SWA(base_opt, swa_start=0, swa_freq=1)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    return optimizer


def make_scheduler(cfg, optimizer):
    lr_type = getattr(cfg.SOLVER, 'LR_SCHEDULER', 'multi_step')

    if lr_type == 'warmup_multi_step':
        return WarmupMultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA,
            warmup_factor=getattr(cfg.SOLVER, 'WARMUP_FACTOR', 1.0/3),
            warmup_iters=getattr(cfg.SOLVER, 'WARMUP_ITERS', 10),
            warmup_method=getattr(cfg.SOLVER, 'WARMUP_METHOD', 'linear')
        )
    elif lr_type == 'warmup_cosine':
        return WarmupCosineLR(
            optimizer,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            warmup_epochs=getattr(cfg.SOLVER, 'WARMUP_ITERS', 10),
            eta_min=getattr(cfg.SOLVER, 'ETA_MIN_LR', 1e-7)
        )
    elif lr_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=float(cfg.SOLVER.MAX_EPOCHS),
            eta_min=getattr(cfg.SOLVER, 'ETA_MIN_LR', 0)
        )
    else:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA
        )
