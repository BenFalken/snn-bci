"""
What the host does to the input before the first spiking layer, in this order:

    channels       keep a list of channels
    dropout        nn.Dropout on the values (training only)
    day gain       one trained gain per channel and recording day (speech)
    encoding       'none'; 'clip' (counts clipped to 1, the motor decoders' binarized input);
                   'threshold' (one event when value / threshold >= 1, straight-through gradient);
                   'spread' (substeps > 1: each step's count, in units of threshold, spread evenly over
                   `substeps` steps; with binarize, at most one event per step)
    event dropout  events dropped at random without rescaling (training only)
    input scale    a fixed factor per channel (folds into the first layer's weights on the chip)
    delays         the channels now and delays[k] steps earlier, side by side (copy-major); on the chip's side
                   of the host, the only memory a Speck network has

With substeps K > 1 every input step becomes K model steps: delays are given in input steps and scaled by K here.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class HostInput(nn.Module):
    def __init__(self, n_channels: int, channels: Optional[Sequence[int]] = None, dropout: float = 0.0,
                 n_days: int = 0, encoding: str = 'none', threshold: float = 1.0, substeps: int = 1,
                 binarize: bool = True, event_dropout: float = 0.0, input_scale=None,
                 delays: Optional[Sequence[int]] = None):
        super().__init__()
        if encoding not in ('none', 'clip', 'threshold', 'spread'):
            raise ValueError(f"encoding must be 'none', 'clip', 'threshold' or 'spread', got {encoding!r}")
        if (encoding == 'spread') != (substeps > 1):
            raise ValueError("encoding 'spread' goes with substeps > 1, and only it")
        self.channels = [int(c) for c in channels] if channels is not None else None
        if self.channels is not None:
            self.register_buffer('channel_index', torch.tensor(self.channels, dtype=torch.long), persistent=False)
        self.n_in = len(self.channels) if self.channels is not None else n_channels
        self.dropout = nn.Dropout(dropout)
        self.day_gain = nn.Parameter(torch.ones(n_days, self.n_in)) if n_days else None
        self.encoding, self.threshold, self.substeps, self.binarize = encoding, float(threshold), int(substeps), binarize
        self.event_dropout = float(event_dropout)
        self.register_buffer('input_scale', torch.as_tensor(input_scale, dtype=torch.float32)
                             if input_scale is not None else None)
        self.delays = sorted({int(d) * self.substeps for d in delays}) if delays else [0]
        self.n_out = self.n_in * len(self.delays)

    def encode(self, x: torch.Tensor, day: Optional[int] = None) -> torch.Tensor:
        """(N, T, C) -> (N, T * substeps, n_in): everything but the delays."""
        if self.channels is not None:
            x = x.index_select(-1, self.channel_index)
        x = self.dropout(x)
        if self.day_gain is not None:
            x = x * self.day_gain[day]
        if self.encoding == 'clip':
            x = x.clamp(max=1.0)
        elif self.encoding == 'threshold':
            u = x / self.threshold
            x = u + (torch.clamp(torch.floor(u.detach()), 0, 1) - u).detach()
        elif self.encoding == 'spread':
            x = self.spread(x)
        if self.training and self.event_dropout > 0:
            x = x * (torch.rand_like(x) >= self.event_dropout)
        if self.input_scale is not None:
            x = x * self.input_scale
        return x

    def spread(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, C) values in units of threshold -> (N, T * K, C): round(value) events spread evenly over each
        step's K sub-steps (step j gets floor((j + 1) q / K) - floor(j q / K)), at most one per sub-step with
        binarize. Gradient straight through, each sub-step carrying value / K."""
        K = self.substeps
        u = x.unsqueeze(2) / (self.threshold * K)                              # (N, T, 1, C), per sub-step
        q = torch.round(u.detach() * K).clamp(min=0)
        j = torch.arange(K, device=x.device, dtype=x.dtype).view(1, 1, K, 1)
        ev = torch.floor((j + 1) * q / K) - torch.floor(j * q / K)
        if self.binarize:
            ev = ev.clamp(max=1.0)
        out = u + (ev - u).detach()
        return out.reshape(x.shape[0], -1, x.shape[-1])

    def delay(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, C) -> (N, T, C * len(delays)): copy-major, zeros before the start."""
        if self.delays == [0]:
            return x
        T = x.shape[1]
        return torch.cat([x if d == 0 else torch.cat([x.new_zeros(x.shape[0], min(d, T), x.shape[2]),
                                                      x[:, :max(T - d, 0)]], 1) for d in self.delays], -1)

    def forward(self, x: torch.Tensor, day: Optional[int] = None) -> torch.Tensor:
        return self.delay(self.encode(x, day))

    # streaming: one input step at a time
    def init_state(self) -> dict:
        return {'past': []}

    def step(self, x: torch.Tensor, st: dict, day: Optional[int] = None) -> List[torch.Tensor]:
        """One input step (N, C) -> its substeps' first-layer inputs, each (N, n_out)."""
        out = []
        for u in self.encode(x[:, None], day).unbind(1):
            if self.delays != [0]:
                st['past'] = ([u] + st['past'])[:self.delays[-1] + 1]
                u = torch.cat([st['past'][d] if d < len(st['past']) else torch.zeros_like(u) for d in self.delays], -1)
            out.append(u)
        return out
