# ranger.py
import math
import torch
from torch.optim.optimizer import Optimizer

class Ranger(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5,
                 betas=(0.95, 0.999), eps=1e-5, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold
        self.radam_buffer = [[None, None, None] for _ in range(10)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger does not support sparse gradients.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['slow_buffer'] = p.detach().clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1

                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p.add_(exp_avg, alpha=-step_size * group['lr'])

                # Lookahead
                if state['step'] % group['k'] == 0:
                    slow = state['slow_buffer']
                    slow.add_(p - slow, alpha=self.alpha)
                    p.copy_(slow)

        return loss
