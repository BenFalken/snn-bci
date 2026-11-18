"""Learnable synaptic delays, after Hammouamri, Khalfaoui-Hassani & Masquelier ("Learning delays in spiking
neural networks using dilated convolutions with learnable spacings", ICLR 2024).

Each synapse's delay is a continuous position in [0, D] steps (20 ms each), starting at the fixed random delay
the model would otherwise use, so a run with learnable delays starts exactly where the fixed-delay model does.
In training the synapse's weight is spread over the whole-step delays by a Gaussian of width sigma centred on
its position, which gives the loss a gradient with respect to the position. sigma shrinks linearly over
training (from max(1, sigma_frac * D) steps to 0.5) and, from a set point on, the delays are rounded to whole
steps and frozen ("hard"), so the remaining epochs train the weights with integer delays, as hardware would run
them. The model that is saved and evaluated at the end has integer delays only.

The kernel takes the place of the fixed models' 0/1 masks (D + 1, *synapse shape): fixed delays are the hard
case of the same thing.
"""
import math

import torch
import torch.nn as nn


class DelayPositions(nn.Module):
    """One learnable delay per synapse: pos (*synapse shape), in steps, kept within [0, D]."""

    def __init__(self, init_delays: torch.Tensor, max_delay: int):
        super().__init__()
        self.D = int(max_delay)
        self.pos = nn.Parameter(init_delays.float().clone())
        self.register_buffer('pos_init', init_delays.to(torch.uint8).clone())   # for the report on how they moved
        self.register_buffer('sigma', torch.tensor(0.0))                         # 0: hard (rounded, frozen)

    def kernel(self) -> torch.Tensor:
        """(D + 1, *synapse shape): the weight share of each whole-step delay; one-hot once hard."""
        d = torch.arange(self.D + 1, device=self.pos.device, dtype=self.pos.dtype).view(-1, *[1] * self.pos.dim())
        p = self.pos.clamp(0, self.D)
        sigma = float(self.sigma)
        if sigma <= 0:
            return (p.detach().round().unsqueeze(0) == d).to(self.pos.dtype)
        g = torch.exp(-0.5 * ((d - p) / sigma) ** 2)
        return g / g.sum(0, keepdim=True).clamp_min(1e-12)

    def delays(self) -> torch.Tensor:
        """The whole-step delays (as deployed)."""
        return self.pos.detach().clamp(0, self.D).round().long()


def delay_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DelayPositions)]


def set_delay_sigma(model, epoch: int, n_epochs: int, sigma_frac: float = 0.25, hard_from: float = 0.75):
    """Schedule for this epoch (0-based): sigma from max(1, sigma_frac * D) down to 0.5 at hard_from * n_epochs,
    then 0 (rounded and frozen). Returns True once the delays are hard."""
    hard_epoch = max(1, int(round(hard_from * n_epochs)))
    hard = epoch >= hard_epoch
    for _, m in delay_modules(model):
        s0 = max(1.0, sigma_frac * m.D)
        m.sigma.fill_(0.0 if hard else s0 + (0.5 - s0) * epoch / hard_epoch)
        if hard:
            with torch.no_grad():
                m.pos.copy_(m.pos.clamp(0, m.D).round())
        m.pos.requires_grad_(not hard)
    return hard


@torch.no_grad()
def clamp_positions(model):
    for _, m in delay_modules(model):
        m.pos.clamp_(0, m.D)


@torch.no_grad()
def delay_summary(model, step_ms: int = 20) -> str:
    """One line per epoch: per group, mean delay and mean distance moved from the start, in ms."""
    parts = []
    for name, m in delay_modules(model):
        p = m.pos.clamp(0, m.D)
        moved = (p - m.pos_init.float()).abs().mean()
        parts.append(f"{name.replace('_delay', '').replace('layers.', 'L')}: {p.mean() * step_ms:.0f} ms "
                     f"(moved {moved * step_ms:.1f})")
    if not parts:
        return ''
    s = float(delay_modules(model)[0][1].sigma)
    return f"sigma {'hard' if s <= 0 else f'{s:.2f} steps (first group)'} | " + ', '.join(parts)


def delay_bits(m: DelayPositions) -> int:
    """Storage of the deployed (whole-step) delays."""
    return m.pos.numel() * max(1, math.ceil(math.log2(m.D + 1)))
