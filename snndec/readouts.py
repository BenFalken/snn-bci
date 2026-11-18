"""
The host readouts: non-spiking, they turn the read layer's spikes into the decoder's output every step.

LeakyReadout  (every LSNN, the speech Speck network)
    leaky output neurons: y_k <- a_k y_k + (1 - a_k) (W r + b)_k, `stages` of them in series, one trained time
    constant per output and stage; r = the read layers' spikes (optionally with leaky traces of them).
    head 'affine': r is a spiking output layer with one neuron per output, and W r + b becomes gain_k r_k + b_k.
    The output neurons start at 0 ('zero') or at the bias ('bias').

EMAReadout  (the motor and audio Speck networks)
    each read neuron's spikes smoothed by `stages` exponential moving averages in series (one decay, or one per
    neuron), then a head: 'linear' (W s + b), 'affine' (gain * s + offset, one per output neuron), or 'population'
    (the motor decoder: 2 groups of output neurons with preferred values 0..1, the prediction per group is their
    activity-weighted mean, `neutral` while a group is silent).
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class LeakyReadout(nn.Module):
    def __init__(self, n_read: int, n_out: int, tau: float, stages: int = 1, init: str = 'zero',
                 bias_init: float = 0.0, traces: Optional[Sequence[float]] = None, dropout: float = 0.0,
                 head: str = 'linear'):
        super().__init__()
        if head not in ('linear', 'affine'):
            raise ValueError(f"head must be 'linear' or 'affine', got {head!r}")
        if head == 'affine' and (traces or n_read != n_out):
            raise ValueError("head 'affine' reads one output neuron per output, without traces")
        self.head = head
        self.gain = nn.Parameter(torch.ones(n_out)) if head == 'affine' else None
        self.traces = [float(t) for t in traces] if traces else None
        self.trace_alphas = [0.0 if t <= 0 else math.exp(-1.0 / t) for t in self.traces] if self.traces else None
        self.linear = nn.Linear(n_read * (len(self.traces) if self.traces else 1), n_out)
        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(bias_init))
        self.log_tau = nn.Parameter(torch.full((stages, n_out), math.log(tau)))
        if init not in ('zero', 'bias'):
            raise ValueError(f"init must be 'zero' or 'bias', got {init!r}")
        self.init = init
        self.dropout = nn.Dropout(dropout)

    def init_state(self, n: int, device) -> dict:
        y0 = torch.zeros(n, self.linear.out_features, device=device)
        if self.init == 'bias':
            y0 = y0 + self.linear.bias.detach()
        return {'y': [y0.clone() for _ in range(self.log_tau.shape[0])], 'tr': None}

    def step(self, r: torch.Tensor, st: dict) -> torch.Tensor:
        if self.traces:
            if st['tr'] is None:
                st['tr'] = [torch.zeros_like(r) for _ in self.trace_alphas]
            st['tr'] = [r if a == 0.0 else a * y + (1 - a) * r for a, y in zip(self.trace_alphas, st['tr'])]
            r = torch.cat(st['tr'], -1)
        a = torch.exp(-1.0 / self.log_tau.exp())
        r = self.dropout(r)
        u = self.gain * r + self.linear.bias if self.head == 'affine' else self.linear(r)
        for k in range(a.shape[0]):
            st['y'][k] = a[k] * st['y'][k] + (1 - a[k]) * u
            u = st['y'][k]
        return u


class EMAReadout(nn.Module):
    def __init__(self, n_read: int, n_out: int, decay: float, stages: int = 2, per_neuron: bool = False,
                 head: str = 'linear', normalize: bool = True, neutral: float = 0.5, bias_init: float = 0.0):
        super().__init__()
        init = torch.logit(torch.tensor(float(decay)))
        self.decay_raw = nn.Parameter(init.repeat(n_read) if per_neuron else init)
        self.stages, self.normalize, self.head = int(stages), normalize, head
        if head == 'linear':
            self.linear = nn.Linear(n_read, n_out)
            nn.init.constant_(self.linear.bias, float(bias_init))
        elif head == 'affine':
            self.gain, self.offset = nn.Parameter(torch.ones(n_out)), nn.Parameter(torch.zeros(n_out))
        elif head == 'population':
            if n_read % n_out:
                raise ValueError('population: the read neurons must split evenly over the outputs')
            self.register_buffer('positions', torch.linspace(0.0, 1.0, n_read // n_out))
            self.register_buffer('neutral', torch.tensor(float(neutral)))
        else:
            raise ValueError(f"head must be 'linear', 'affine' or 'population', got {head!r}")
        self.n_out = n_out

    def init_state(self, n: int, device) -> dict:
        return {'e': [None] * self.stages}                              # None: zero, until the first step

    def step(self, r: torch.Tensor, st: dict) -> torch.Tensor:
        d = torch.sigmoid(self.decay_raw)
        u = r
        for k in range(self.stages):
            st['e'][k] = u if st['e'][k] is None else d * st['e'][k] + u
            u = st['e'][k]
        if self.normalize:
            u = u * (1 - d) ** self.stages
        if self.head == 'linear':
            return self.linear(u)
        if self.head == 'affine':
            return self.gain * u + self.offset
        g = u.reshape(u.shape[0], self.n_out, -1)                        # (N, outputs, neurons per output)
        tot = g.sum(-1)
        pred = (g * self.positions).sum(-1) / tot.clamp(min=1)
        return torch.where(tot > 0, pred, self.neutral)
