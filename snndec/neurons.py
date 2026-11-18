"""
The one spiking neuron of every decoder: the LSNN's layers and the Speck chip's layers are the same layer with
different switches.

    current  = W x  (+ W_rec s_prev                    if recurrent)
    i        = a_syn i + (1 - a_syn) current          if a synaptic filter (unit gain), else i = current
    v        = v + i                                   integrate-and-fire: no leak
    thr      = threshold (+ adapt_scale b              on the adaptive neurons)
    s        = clamp(floor(v / thr), 0, max_spikes)   spikes per step (max_spikes None: unbounded, sinabs MultiSpike)
    v        = v - s thr  (soft)  or  v (s == 0)  (hard)
    v        = clamp(v, min_vmem, max_vmem)
    b        = a_ad b + (1 - a_ad) s                   threshold adaptation

The Speck chip runs this layer with recurrent=False, no filter, no adaptation and reset='hard'. Everything is in
steps; the models convert milliseconds to steps.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------- surrogate gradients
def surrogate(name: str, width: float = 0.5):
    """d spikes / d v as a function of (v, threshold). 'lsnn': the motor and audio LSNNs' single exponential,
    exp(-|v - thr| / width) / thr; 'single_exponential' and 'periodic_exponential': sinabs' functions (the speech
    decoders and the motor Speck SNN)."""
    if name == 'lsnn':
        return lambda v, thr: torch.exp(-torch.abs(v - thr) / width) / thr
    import sinabs.activation as sa
    fn = {'single_exponential': sa.SingleExponential, 'periodic_exponential': sa.PeriodicExponential}[name]()
    return fn


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr, max_spikes, grad_fn):
        ctx.save_for_backward(v, thr if torch.is_tensor(thr) else torch.as_tensor(thr, device=v.device))
        ctx.grad_fn = grad_fn
        s = torch.floor(v / thr).clamp(min=0)
        return s if max_spikes is None else s.clamp(max=max_spikes)

    @staticmethod
    def backward(ctx, grad):
        v, thr = ctx.saved_tensors
        return grad * ctx.grad_fn(v, thr), None, None, None


# ---------------------------------------------------------------------------------------------------- the layer
class SpikingLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, recurrent: bool = False, reset: str = 'hard',
                 tau_syn: Optional[Sequence[float]] = None, adaptive_fraction: float = 0.0,
                 tau_adapt: float = 25.0, adapt_scale: float = 1.8, threshold: float = 1.0,
                 max_spikes: Optional[int] = None, min_vmem: Optional[float] = -1.0, max_vmem: Optional[float] = None,
                 rec_init_scale: float = 0.5, adapt_seed: int = 5, surrogate_fn: str = 'lsnn', dropout: float = 0.0):
        """tau_syn (lo, hi) steps: one trained synaptic time constant per neuron, log-spaced; None = no filter.
        dropout: on this layer's input (training only)."""
        super().__init__()
        if reset not in ('soft', 'hard'):
            raise ValueError(f"reset must be 'soft' or 'hard', got {reset!r}")
        self.width, self.reset, self.threshold = n_out, reset, float(threshold)
        self.max_spikes, self.min_vmem, self.max_vmem = max_spikes, min_vmem, max_vmem
        self.ff = nn.Linear(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight)
        self.rec = nn.Linear(n_out, n_out, bias=False) if recurrent else None
        if self.rec is not None:
            nn.init.orthogonal_(self.rec.weight)
            with torch.no_grad():
                self.rec.weight.mul_(rec_init_scale)
        self.log_tau_syn = None
        if tau_syn:
            lo, hi = tau_syn
            self.log_tau_syn = nn.Parameter(torch.log(torch.logspace(math.log10(lo), math.log10(hi), n_out)))
        mask = None
        if adaptive_fraction > 0:
            mask = torch.zeros(n_out)
            order = torch.randperm(n_out, generator=torch.Generator().manual_seed(adapt_seed))
            mask[order[:int(round(adaptive_fraction * n_out))]] = 1.0
        self.register_buffer('adapt_mask', mask)
        self.alpha_adapt = math.exp(-1.0 / tau_adapt)
        self.adapt_scale = float(adapt_scale)
        self.surrogate_fn = surrogate_fn
        self._grad = surrogate(surrogate_fn)
        self.dropout = nn.Dropout(dropout)

    @property
    def chip_compatible(self) -> bool:
        """What the Speck chip runs: feedforward, no filter, no adaptation; either reset (the chip zeroes the membrane
        or subtracts the threshold: sinabs' make_config sets return_to_zero from the layer's reset function)."""
        return self.rec is None and self.log_tau_syn is None and self.adapt_mask is None

    def init_state(self, n: int, device) -> dict:
        z = lambda: torch.zeros(n, self.width, device=device)
        return {'i': z(), 'v': z(), 'b': z(), 's': z()}

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward current for many steps at once: (..., n_in) -> (..., n_out)."""
        return self.ff(self.dropout(x))

    def step(self, current: torch.Tensor, st: dict) -> torch.Tensor:
        """One step given its feedforward current; updates st, returns the spikes."""
        if self.rec is not None:
            current = current + self.rec(st['s'])
        if self.log_tau_syn is not None:
            a = torch.exp(-1.0 / self.log_tau_syn.exp())
            st['i'] = a * st['i'] + (1 - a) * current
            current = st['i']
        v = st['v'] + current
        thr = self.threshold if self.adapt_mask is None else self.threshold + self.adapt_scale * st['b'] * self.adapt_mask
        s = _Spike.apply(v, thr, self.max_spikes, self._grad)
        v = v - s * thr if self.reset == 'soft' else v * (s == 0)
        if self.min_vmem is not None or self.max_vmem is not None:
            v = torch.clamp(v, min=self.min_vmem, max=self.max_vmem)
        st['v'], st['s'] = v, s
        if self.adapt_mask is not None:
            st['b'] = self.alpha_adapt * st['b'] + (1 - self.alpha_adapt) * s
        return s
