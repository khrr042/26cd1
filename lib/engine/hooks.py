# encoding: utf-8
import os
import torch
import numpy as np
from typing import Any, Optional
from torch.optim.swa_utils import update_bn
from lib.utils.metrics import eval_func

class HookBase:
    def __init__(self):
        self.trainer: Any = None 

    def before_train(self): pass
    def after_train(self): pass
    def before_epoch(self, epoch): pass
    def after_epoch(self, epoch): pass
    def after_step(self, epoch, iteration, loss): pass

class SWAHook(HookBase):
    def __init__(self, swa_model, swa_start_epoch, train_loader):
        super().__init__()
        self.swa_model = swa_model
        self.swa_start_epoch = swa_start_epoch
        self.train_loader = train_loader

    def after_epoch(self, epoch):
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(self.trainer.model)

    def after_train(self):
        device = next(self.swa_model.parameters()).device
        update_bn(self.train_loader, self.swa_model, device=device)

class ValidationHook(HookBase):
    def __init__(self, val_loader, num_query, interval=10):
        super().__init__()
        self.val_loader = val_loader
        self.num_query = num_query
        self.interval = interval

    def after_epoch(self, epoch):
        if epoch % self.interval == 0:
            self.evaluate(epoch)

    def evaluate(self, epoch):
        if self.trainer is None:
            return

        model = self.trainer.model
        device = self.trainer.device
        model.eval()
        
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                img, pid, camid = batch[0].to(device), batch[1], batch[2]
                feat = model(img)
                feats.append(feat.cpu())
                pids.extend(np.asarray(pid))
                camids.extend(np.asarray(camid))

        feats = torch.cat(feats, dim=0)
        pids = np.array(pids)
        camids = np.array(camids)

        q_feats = feats[:self.num_query]
        q_pids = pids[:self.num_query]
        q_camids = camids[:self.num_query]

        g_feats = feats[self.num_query:]
        g_pids = pids[self.num_query:]
        g_camids = camids[self.num_query:]

        distmat = 1 - torch.mm(q_feats, g_feats.t())
        cmc, mAP = eval_func(distmat.numpy(), q_pids, g_pids, q_camids, g_camids)
        
        print(f"\n[Epoch {epoch}] Validation - mAP: {mAP:.1%}, Rank-1: {cmc[0]:.1%}")

class CheckpointHook(HookBase):
    def __init__(self, save_dir, interval=10):
        super().__init__()
        self.save_dir = save_dir
        self.interval = interval

    def after_epoch(self, epoch):
        if self.trainer is not None and epoch % self.interval == 0:
            save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.trainer.model.state_dict(), save_path)