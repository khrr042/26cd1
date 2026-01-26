import random
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()

        x_normed = (x - mu) / sig

        gamma1 = torch._standard_gamma(
            torch.full((B, 1, 1, 1), self.alpha, device=x.device)
        )
        gamma2 = torch._standard_gamma(
            torch.full((B, 1, 1, 1), self.alpha, device=x.device)
        )
        lmda = gamma1 / (gamma1 + gamma2)

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class MixStyle2(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()

        x_normed = (x - mu) / sig

        gamma1 = torch._standard_gamma(
            torch.full((B, 1, 1, 1), self.alpha, device=x.device)
        )
        gamma2 = torch._standard_gamma(
            torch.full((B, 1, 1, 1), self.alpha, device=x.device)
        )
        lmda = gamma1 / (gamma1 + gamma2)

        perm = torch.arange(B - 1, -1, -1, device=x.device)
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2, device=x.device)]
        perm_a = perm_a[torch.randperm(B // 2, device=x.device)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu2, sig2 = mu[perm], sig[perm]

        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
