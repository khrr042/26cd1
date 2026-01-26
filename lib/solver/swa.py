# swa_modern.py
# encoding: utf-8
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        self._auto_mode = swa_start is not None and swa_freq is not None
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr if swa_lr is not None else None

        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = defaultdict(dict)

        for group in self.param_groups:
            group.setdefault('step_counter', 0)
            group.setdefault('n_avg', 0)

        super().__init__(optimizer.param_groups, optimizer.defaults)

    def swap_swa_sgd(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'swa_buffer' not in self.state[p]:
                    continue
                buf = self.state[p]['swa_buffer']
                tmp = p.clone()
                p.copy_(buf)
                self.state[p]['swa_buffer'] = tmp

    def update_swa(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'swa_buffer' not in self.state[p]:
                    self.state[p]['swa_buffer'] = p.clone().detach()
                    self.state[p]['swa_buffer'].requires_grad_(False)

                buf = self.state[p]['swa_buffer']
                n = group['n_avg']
                buf += (p - buf) / (n + 1)
            group['n_avg'] += 1

    def step(self, closure=None):
        if self.swa_lr is not None:
            for group in self.param_groups:
                if group['step_counter'] >= self.swa_start:
                    group['lr'] = self.swa_lr

        loss = self.optimizer.step(closure)

        for group in self.param_groups:
            group['step_counter'] += 1
            if self._auto_mode and group['step_counter'] >= self.swa_start:
                if (group['step_counter'] - self.swa_start) % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def update_swa_group(self, group):
        for p in group['params']:
            if 'swa_buffer' not in self.state[p]:
                self.state[p]['swa_buffer'] = p.clone().detach()
                self.state[p]['swa_buffer'].requires_grad_(False)
            buf = self.state[p]['swa_buffer']
            n = group['n_avg']
            buf += (p - buf) / (n + 1)
        group['n_avg'] += 1

    @staticmethod
    @torch.no_grad()
    def update_bn(loader, model, device=None):
        if device is None:
            device = next(model.parameters()).device
        model.train()
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                momenta[module] = module.momentum
                module.running_mean.zero_()
                module.running_var.fill_(1)

        n = 0
        for inputs, *_ in loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            b = inputs.size(0)
            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum
            inputs = inputs.to(device)
            model(inputs)
            n += b

        for module in momenta.keys():
            module.momentum = momenta[module]
        model.eval()
