"""
Input front end of the recurrent SNN (rsnn.SNN_Recurrent): turning the per-bin features into events, and giving
the first layer memory of past input in ways a spiking chip supports, other than stacking a long window of bins.

EventEncoder (input_encoding): each feature channel of each 20 ms bin becomes a signed event count, so the first
layer adds a weight per event instead of multiplying every input -- an addition per event that occurs.
    ternary      memoryless dead zone: n = sign(x) * min(floor(|x| / threshold), max_events). Values below
                 the threshold send nothing; the larger the threshold, the sparser.
    delta        send-on-delta: events when the value has moved by a threshold since the last event
                 (n = the whole steps moved, signed); a slowly changing feature sends little. The events
                 carry changes, not levels.
    sigma_delta  the residual (value minus what was sent) is carried to the next bin, so the events'
                 running sum follows the value with an error below one threshold (a sigma-delta modulator).
    counts       (2026-11-26, for Speck, which takes no negative events) n = min(floor(max(x, 0) / threshold),
                 max_events) events on the channel's one line: graded input as repeated events in a step. Under
                 soft reset the chip keeps what each repeat adds; under hard reset the excess was lost at a spike.
    offset       (2026-11-27, for Speck) the SIGNED value on one positive line: n = round((x + offset) / threshold),
                 clipped to 0..max_events. The channel's mean (x = 0) sends offset / threshold events every bin, a
                 quieter bin fewer and a busier one more, so below-mean activity reaches the chip too; layer 1 sees
                 w (x + offset) / threshold, the constant part absorbed in training like a bias.
    onoff        (2026-11-26) send-on-delta split into two positive lines per channel: 'rise' events when the value
                 has risen by a threshold since the last event, 'fall' when it has fallen (at most max_events each);
                 the slope of the feature, laid out as two levels (all rise lines, then all fall lines).
Training passes gradients straight through (d events / dx = 1 / threshold), as quantization-aware training does.

InputMemory, on the stacked frames (one frame = input_window bins, every input_stride bins):
    delays       synaptic delays: frame t also delivers frames t - d for each d in input_delays (in frames),
                 each through its own weights. Delays [0, 1, 2, 3, 4, 5, 6, 7] over 4-bin frames are the
                 32-bin window again; [0, 1, 2, 4, 8] cover 720 ms with 5 taps.
    traces       synaptic currents with several time constants: frame t is low-pass filtered with each tau in
                 input_traces (in frames; 0 = the frame itself), y = a y + (1 - a) x, a = exp(-1 / tau),
                 each filtered copy through its own weights: a neuron with a few synapse types of different
                 speeds, as many neuromorphic chips offer. Memory of the past in M copies instead of a window.
                 Event-driven cost: each event adds to M currents (M accumulates per target neuron), and each
                 current decays once per step.
Delays apply before traces when both are given. Both are causal; step() matches forward() bin by bin.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

ENCODINGS = ('none', 'ternary', 'delta', 'sigma_delta', 'binary', 'levels', 'counts', 'onoff', 'offset')


class EventEncoder(nn.Module):
    def __init__(self, mode: str = 'none', threshold: float = 0.25, max_events: int = 3,
                 levels: Optional[List[float]] = None, offset: float = 3.0):
        super().__init__()
        if mode not in ENCODINGS:
            raise ValueError(f'input_encoding must be one of {ENCODINGS}, got {mode!r}')
        if mode != 'none' and threshold <= 0:
            raise ValueError('encoding_threshold must be > 0')
        self.mode, self.threshold, self.max_events = mode, float(threshold), int(max_events)
        self.offset = float(offset)
        # 'levels': one on/off line per channel and level (a thermometer code): line l is on when the feature is
        # at or above levels[l], so a channel's value reaches the chip as how many of its lines are on, with no
        # sub-steps. The lines are laid out level by level (all channels at levels[0], then levels[1], ...)
        if mode == 'levels' and not levels:
            raise ValueError("input_encoding 'levels' needs encoding_levels")
        self.levels = [float(v) for v in levels] if mode == 'levels' else None

    @property
    def n_lines(self) -> int:
        """Input lines per channel: one, or one per level."""
        return len(self.levels) if self.levels else 2 if self.mode == 'onoff' else 1

    @property
    def active(self) -> bool:
        return self.mode != 'none'

    def _quantize(self, u: torch.Tensor) -> torch.Tensor:
        """u in threshold units -> signed whole events, at most max_events per bin."""
        return torch.sign(u) * torch.clamp(torch.floor(u.abs()), max=self.max_events)

    def init_state(self, n: int, c: int, device) -> torch.Tensor:
        return torch.zeros(n, c, device=device)

    def step(self, x: torch.Tensor, state: Optional[torch.Tensor]):
        """One bin x (N, C) -> (events (N, C), new state). state: the delta encoder's last sent level or the
        sigma-delta encoder's residual, in threshold units."""
        if self.mode == 'levels':                              # straight through per line, as 'binary'
            out = []
            for v in self.levels:
                u = x / v
                q = torch.clamp(torch.floor(u.detach()), 0, 1)
                out.append(u + (q - u).detach())
            return torch.cat(out, -1), state
        u = x / self.threshold
        if self.mode == 'binary':                              # at most one positive event per channel per bin
            q = torch.clamp(torch.floor(u.detach()), 0, 1)                # (raw counts, threshold 1: count > 0)
            return u + (q - u).detach(), state                 # straight through, so the per-day gain still learns
        if self.mode == 'ternary':
            q = self._quantize(u.detach())
            return u + (q - u).detach(), state
        if self.mode == 'offset':                              # signed value on one line, about a baseline
            u = (x + self.offset) / self.threshold
            q = torch.clamp(torch.floor(u.detach() + 0.5), 0, self.max_events)
            return u + (q - u).detach(), state
        if self.mode == 'counts':                              # positive whole events, at most max_events
            q = torch.clamp(torch.floor(u.detach()), 0, self.max_events)
            return u + (q - u).detach(), state
        if self.mode == 'onoff':                               # send-on-delta, rises and falls on separate lines
            d = u - state
            q = self._quantize(d.detach())
            up, dn = torch.relu(d), torch.relu(-d)
            on, off = torch.clamp(q, min=0), torch.clamp(-q, min=0)
            return torch.cat([up + (on - up).detach(), dn + (off - dn).detach()], -1), state + q
        if self.mode == 'delta':
            d = u - state                                      # the change since the last sent level
            q = self._quantize(d.detach())
            return d + (q - d).detach(), state + q
        r = state + u.detach()                                 # sigma-delta: carry the residual
        q = self._quantize(r)
        return u + (q - u).detach(), r - q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, C) -> signed event counts (N, T, C)."""
        if not self.active:
            return x
        if self.mode in ('ternary', 'binary', 'levels', 'counts', 'offset'):
            return self.step(x, None)[0]
        state = self.init_state(x.shape[0], x.shape[2], x.device)
        out = []
        for t in range(x.shape[1]):
            q, state = self.step(x[:, t], state)
            out.append(q)
        return torch.stack(out, 1)


class InputMemory(nn.Module):
    def __init__(self, delays: Optional[List[int]] = None, traces: Optional[List[float]] = None):
        super().__init__()
        self.delays = sorted({int(d) for d in delays}) if delays else None
        if self.delays and self.delays[0] < 0:
            raise ValueError('input_delays must be >= 0 (causal)')
        self.traces = [float(t) for t in traces] if traces else None
        self.alphas = [0.0 if t <= 0 else math.exp(-1.0 / t) for t in self.traces] if self.traces else None

    @property
    def n_delays(self) -> int:
        return len(self.delays) if self.delays else 1

    @property
    def n_traces(self) -> int:
        return len(self.traces) if self.traces else 1

    @property
    def n_decaying(self) -> int:
        """Filtered copies that decay each step (tau > 0)."""
        return sum(a > 0 for a in self.alphas) if self.alphas else 0

    def expand(self, f: int) -> int:
        return f * self.n_delays * self.n_traces

    def delay(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, F) -> (N, T, F * n_delays): frame t, then frame t - d for each delay."""
        if not self.delays:
            return x
        T = x.shape[1]
        shifted = [x if d == 0 else torch.cat([torch.zeros_like(x[:, :min(d, T)]), x[:, :max(T - d, 0)]], 1)
                   for d in self.delays]
        return torch.cat(shifted, -1)

    def trace(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, F) -> (N, T, F * n_traces): each copy low-pass filtered with its time constant."""
        if not self.traces:
            return x
        copies = []
        for a in self.alphas:
            if a == 0.0:
                copies.append(x)
                continue
            y, ys = torch.zeros_like(x[:, 0]), []
            for t in range(x.shape[1]):
                y = a * y + (1 - a) * x[:, t]
                ys.append(y)
            copies.append(torch.stack(ys, 1))
        return torch.cat(copies, -1)

    def init_state(self) -> dict:
        return {'past': [], 'y': None}

    def step(self, f: torch.Tensor, st: dict) -> torch.Tensor:
        """One frame f (N, F) -> the first layer's input (N, F * n_delays * n_traces)."""
        if self.delays:
            st['past'] = ([f] + st['past'])[:self.delays[-1] + 1]          # newest first
            f = torch.cat([st['past'][d] if d < len(st['past']) else torch.zeros_like(f)
                           for d in self.delays], -1)
        if self.traces:
            if st['y'] is None:
                st['y'] = [torch.zeros_like(f) for _ in self.alphas]
            st['y'] = [f if a == 0.0 else a * y + (1 - a) * f for a, y in zip(self.alphas, st['y'])]
            f = torch.cat(st['y'], -1)
        return f
