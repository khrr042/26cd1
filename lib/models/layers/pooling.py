import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super().__init__()
        self.eps = eps

        if freeze_p:
            self.register_buffer("p", torch.tensor(p))
        else:
            self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1. / self.p)

    def __repr__(self):
        p = float(self.p.detach())
        return f"{self.__class__.__name__}(p={p:.4f}, eps={self.eps})"
