import os
import argparse
import sys
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from lib.config import cfg
from lib.data import make_data_loader
from lib.models import build_model
from lib.losses import MomentumAdaptiveLoss
from lib.solver.solver import make_optimizer, make_scheduler
from lib.utils.utils import setup_logger
from lib.engine.hooks import SWAHook, ValidationHook, CheckpointHook


class Trainer:
    def __init__(
        self,
        cfg,
        model: Optional[nn.Module],
        optimizer,
        scheduler,
        train_loader,
        loss_func,
        hooks: Optional[List] = None
    ):
        if model is None:
            raise ValueError("Trainer requires a valid nn.Module, got None")
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.loss_func = loss_func
        self.hooks = hooks or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = amp.GradScaler(enabled=cfg.SOLVER.FP16)

        self.model.to(self.device)
        for hook in self.hooks:
            hook.trainer = self

        self._backbone_frozen = False

    def _set_backbone_trainable(self, trainable: bool):
        if not hasattr(self.model, "base"):
            return
        for p in self.model.base.parameters():
            p.requires_grad = trainable
        if trainable:
            self.model.base.train()
        else:
            self.model.base.eval()
        self._backbone_frozen = not trainable

    def train(self):
        max_epochs = self.cfg.SOLVER.MAX_EPOCHS
        for epoch in range(1, max_epochs + 1):
            self.model.train()

            freeze_epochs = getattr(self.cfg.SOLVER, "FREEZE_BACKBONE_EPOCHS", 0)
            if freeze_epochs > 0:
                if epoch <= freeze_epochs and not self._backbone_frozen:
                    self._set_backbone_trainable(False)
                elif epoch > freeze_epochs and self._backbone_frozen:
                    self._set_backbone_trainable(True)
            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{max_epochs}]")
            for batch in loop:
                img, target = batch[0].to(self.device), batch[1].to(self.device)

                self.optimizer.zero_grad()
                with amp.autocast(enabled=self.cfg.SOLVER.FP16):
                    outputs = self.model(img)
                    score, feat = outputs[:2] if isinstance(outputs, tuple) else (outputs, outputs)
                    loss = self.loss_func(score, feat, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loop.set_postfix(loss=loss.item())

            self.scheduler.step()

            for hook in self.hooks:
                hook.after_epoch(epoch)

        for hook in self.hooks:
            hook.after_train()


def do_train(cfg):
    output_dir = cfg.OUTPUT_DIR
    setup_logger("reid_baseline", output_dir, is_train=True)


    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    model: Optional[nn.Module] = build_model(cfg, num_classes)
    if model is None:
        raise ValueError("build_model returned None")

    swa_model = AveragedModel(model) if getattr(cfg.SOLVER, "USE_SWA", True) else None

    loss_func = MomentumAdaptiveLoss(cfg, num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    hooks = []
    if getattr(cfg.SOLVER, "USE_SWA", True):
        hooks.append(SWAHook(swa_model, int(cfg.SOLVER.MAX_EPOCHS * 0.75), train_loader))
    hooks += [
        CheckpointHook(output_dir, interval=getattr(cfg.SOLVER, 'CHECKPOINT_PERIOD', 5)),
        ValidationHook(val_loader, num_query, interval=getattr(cfg.SOLVER, 'EVAL_PERIOD', 5)),
    ]

    trainer = Trainer(cfg, model, optimizer, scheduler, train_loader, loss_func, hooks)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    do_train(cfg)
