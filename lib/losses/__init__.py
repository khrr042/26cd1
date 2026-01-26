import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .triplet_loss import TripletLoss
from .crossentropy import CrossEntropyLabelSmooth
from .supcon import SupConLoss


class MomentumAdaptiveLoss(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()

        self.update_iter_interval = 500
        self.momentum = 0.9

        self.ID_LOSS_WEIGHT = cfg.MODEL.ID_LOSS_WEIGHT
        self.METRIC_LOSS_WEIGHT = cfg.MODEL.TRIPLET_LOSS_WEIGHT

        self.id_loss_history = []
        self.metric_loss_history = []

        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            self.metric_loss_func = TripletLoss(cfg.SOLVER.MARGIN)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'supconloss':
            self.metric_loss_func = SupConLoss(
                num_ids=int(cfg.SOLVER.IMS_PER_BATCH / cfg.DATALOADER.NUM_INSTANCE),
                views=cfg.DATALOADER.NUM_INSTANCE
            )

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'none':
            self.metric_loss_func = None

        else:
            raise ValueError(f"Unsupported metric loss {cfg.MODEL.METRIC_LOSS_TYPE}")


        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            self.id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            self.id_loss_func = nn.CrossEntropyLoss()

    def forward(self, score, feat, target):
        id_loss = self.id_loss_func(score, target)

        if self.metric_loss_func is None:
            metric_loss = torch.zeros((), device=feat.device)
        else:
            if isinstance(self.metric_loss_func, SupConLoss):
                metric_loss = self.metric_loss_func(feat, labels=target)
            else:
                metric_loss = self.metric_loss_func(feat, target)

        self.id_loss_history.append(id_loss.detach().item())
        self.metric_loss_history.append(metric_loss.detach().item())

        if len(self.id_loss_history) % self.update_iter_interval == 0:
            id_std = np.std(self.id_loss_history)
            metric_std = np.std(self.metric_loss_history)

            self.id_loss_history.clear()
            self.metric_loss_history.clear()

            if id_std > metric_std:
                safe_std = max(id_std, 1e-6)
                new_lambda = 1.0 - (id_std - metric_std) / safe_std
                self.ID_LOSS_WEIGHT = (
                    self.momentum * self.ID_LOSS_WEIGHT
                    + (1 - self.momentum) * new_lambda
                )

            if torch.distributed.is_initialized():
                w = torch.tensor(self.ID_LOSS_WEIGHT, device=feat.device)
                torch.distributed.broadcast(w, src=0)
                self.ID_LOSS_WEIGHT = w.item()

        total_loss = (
            self.ID_LOSS_WEIGHT * id_loss
            + self.METRIC_LOSS_WEIGHT * metric_loss
        )

        return total_loss
    
    def make_loss(cfg, num_classes):
        return MomentumAdaptiveLoss(cfg, num_classes)
