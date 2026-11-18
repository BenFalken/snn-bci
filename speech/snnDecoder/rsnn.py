"""
Recurrent spiking phoneme decoder shaped like the paper's RNN
(NeuralDecoder, the cached derived/rnns/baselineRelease):

    per-session input layer  Linear(256, 256) + softsign + dropout 0.2      as the RNN
                             (input_layer 'gain': one gain per channel instead)
    input events             optional (input_encoding, frontend.py): each bin's features become signed
                             event counts, so the first layer adds a weight per event
    input stacking           the last input_window bins, every input_stride   as the RNN (32 bins,
                             bins, flattened (window * 256 inputs)            stride 4: one step per 80 ms)
    input memory             optional (frontend.py): synaptic delays (input_delays) and synaptic
                             currents of several time constants (input_traces) on the first layer
    spiking layers           hidden_dims, each                                RNN: 5 GRU layers of 1024
                               I(t) = W x(t) + U s(t-1)   (U if recurrent)
                               -> optional synaptic filter (tau_syn, unit gain)
                               -> IAF / LIF / ALIF neuron, soft or hard reset
                             with dropout (layer_dropout) on each layer's input   RNN: 0.4
    readout                  optional leaky stages (as SNN_Speech's EMA) of the
                             last layer's spikes -> Linear(width, n_classes)

Stacking uses only past bins, like the RNN's ('valid'): output frame k sees
bins [k*stride, k*stride + window), so a trial of T bins gives
(T - window) // stride + 1 outputs -- the RNN's frames exactly, so its CTC
language-model settings carry over. With input_window = input_stride = 1 the
network steps every 20 ms bin.

Time constants (tau_mem, tau_syn, tau_adapt, the readout decay) are in model
steps: 80 ms with stride 4, 20 ms without stacking.

Same interface as model.SNN_Speech: forward() over padded batches with
lengths, layer_spikes / total_spikes / op_counts, reset_stream() + step()
for streaming (step() returns logits when a frame completes, else None),
plus output_lengths(), frame_kernel and frame_stride.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import sinabs.activation

from .delays import DelayPositions
from .frontend import EventEncoder, InputMemory
from .model import N_CLASSES, get_surrogate_grad_fn


class SpikingLayer(nn.Module):
    """One layer of (optionally recurrent) spiking neurons, stepped in time."""

    def __init__(self, n_in: int, n_out: int, recurrent: bool, neuron_type: str, reset_type: str,
                 threshold: float, tau_mem: float, tau_syn: Optional[float], tau_adapt: float,
                 adapt_scale: float, min_vmem: Optional[float], spike_fn, surrogate_grad, norm_syn: bool,
                 dropout: float, rec_init_scale: float = 1.0, rec_init: str = 'default',
                 learn_threshold: str = 'none', tau_mem_range=None, tau_adapt_range=None, substeps: int = 1,
                 tau_syn_range=None, max_vmem: Optional[float] = None, rec_delays=None,
                 rec_delay_fractions=None, rec_no_self: bool = False, rec_syn_delay: int = 0,
                 ff_syn_delay: int = 0, adaptive_fraction: float = 0.0, learn_delays: bool = False,
                 seadlif: Optional[dict] = None):
        super().__init__()
        self.width = n_out
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight)
        self.rec = nn.Linear(n_out, n_out, bias=False) if recurrent else None
        if self.rec is not None:
            # 'default': PyTorch's uniform init; 'orthogonal': as the RNN's GRU recurrent weights.
            # Scaled by rec_init_scale (< 1: weaker initial feedback).
            if rec_init == 'orthogonal':
                nn.init.orthogonal_(self.rec.weight)
            with torch.no_grad():
                self.rec.weight.mul_(rec_init_scale)
        # rec_no_self: no neuron feeds its own spikes back (diagonal of the recurrent weights held at 0)
        self.rec_no_self = bool(rec_no_self) and self.rec is not None
        if self.rec_no_self:
            self.register_buffer('rec_offdiag', 1.0 - torch.eye(n_out))
            with torch.no_grad():
                self.rec.weight.mul_(self.rec_offdiag)
        # rec_delays: per-neuron delays (in steps) of the recurrent connections, as DelayNet (Balafrej et al.,
        # arXiv 2310.19067): neuron j's spikes reach the layer's neurons d_j steps later than usual, d_j drawn
        # from rec_delays in the proportions rec_delay_fractions (default equal). The assignment is fixed (seed 0)
        # and saved with the weights. In training, a delay lets the gradient skip d_j steps at once.
        self.rec_delay_values = None
        if rec_delays and self.rec is not None and max(rec_delays) > 0:
            values = [int(d) for d in rec_delays]
            fr = list(rec_delay_fractions) if rec_delay_fractions else [1.0] * len(values)
            if len(fr) != len(values) or min(values) < 0:
                raise ValueError('rec_delays (>= 0) and rec_delay_fractions need the same number of values')
            counts = [int(round(f / sum(fr) * n_out)) for f in fr]
            counts[-1] = n_out - sum(counts[:-1])
            order = torch.randperm(n_out, generator=torch.Generator().manual_seed(0))
            delay_of = torch.zeros(n_out, dtype=torch.long)
            start = 0
            for d, c in zip(values, counts):
                delay_of[order[start:start + c]] = d
                start += c
            self.rec_delay_values = sorted(set(values))
            self.register_buffer('rec_delay_masks', torch.stack([(delay_of == d).float() for d in self.rec_delay_values]))
        # rec_syn_delay D > 0: heterogeneous synaptic delays on the recurrent connections -- every synapse its own
        # conduction delay, drawn uniformly from 0..D steps (fixed, seed 1, saved with the weights), as axons of
        # different lengths. Still one weight per synapse, and on event-driven hardware one accumulate per spike
        # and synapse, as without delays (Izhikevich 2006; learned delays: Hammouamri et al., ICLR 2024).
        self.rec_syn_delay = int(rec_syn_delay) if self.rec is not None else 0
        if self.rec_syn_delay > 0:
            if self.rec_delay_values:
                raise ValueError('rec_syn_delay and rec_delays (per-neuron) are alternatives; use one')
            idx = torch.randint(0, self.rec_syn_delay + 1, (n_out, n_out), generator=torch.Generator().manual_seed(1))
            if learn_delays:                         # learnable, starting from these delays (delays.py)
                self.rec_delay = DelayPositions(idx, self.rec_syn_delay)
            else:
                self.register_buffer('rec_syn_masks', torch.stack([idx == d for d in range(self.rec_syn_delay + 1)]))
        # ff_syn_delay D > 0: the same for the feedforward synapses from the layer below (the native model's layers
        # 2+): each synapse its own delay, uniform over 0..D steps (fixed, seed 3), still one weight per synapse
        self.ff_syn_delay = int(ff_syn_delay)
        if self.ff_syn_delay > 0:
            idx = torch.randint(0, self.ff_syn_delay + 1, (n_out, n_in), generator=torch.Generator().manual_seed(3))
            if learn_delays:
                self.ff_delay = DelayPositions(idx, self.ff_syn_delay)
            else:
                self.register_buffer('ff_syn_masks', torch.stack([idx == d for d in range(self.ff_syn_delay + 1)]))
        self.neuron_type, self.reset_type = neuron_type, reset_type
        self.threshold, self.adapt_scale, self.min_vmem = threshold, adapt_scale, min_vmem
        # adaptive_fraction f in (0, 1): a mixed population, as the LSNN of Bellec et al. (NeurIPS 2018): a fixed
        # random fraction f of the layer's neurons has threshold adaptation (tau_adapt, adapt_scale), the rest none,
        # all with this layer's membrane (neuron_type: 'iaf' = no leak, 'lif' = leak). The adaptive neurons' slowly
        # decaying thresholds hold memory; the others stay fast. 0 = off (neuron_type alone decides)
        mask = None
        if adaptive_fraction and 0.0 < adaptive_fraction < 1.0:
            if neuron_type == 'alif':
                raise ValueError("adaptive_fraction mixes adaptive and plain neurons: use neuron_type 'iaf' or 'lif'")
            n_ad = int(round(adaptive_fraction * n_out))
            order = torch.randperm(n_out, generator=torch.Generator().manual_seed(5))
            mask = torch.zeros(n_out)
            mask[order[:n_ad]] = 1.0
        self.register_buffer('adapt_mask', mask)          # None when off (nothing saved)
        # max_vmem: a ceiling on the membrane potential (a safety net; with a spike cap and no leak, charge
        # beyond what can be fired would otherwise accumulate without bound)
        self.max_vmem = max_vmem
        # learn_threshold 'layer' / 'neuron': the base threshold is trained (one per layer / per neuron),
        # kept positive as exp(log_threshold), starting at `threshold`. sinabs' spike functions pass no
        # gradient to the threshold, so the spikes, the reset and the membrane gradient stay exactly
        # sinabs' (spike_fn(v, threshold) as for a fixed threshold) and a zero-valued term adds only the
        # base threshold's gradient, ds/dthreshold = -(v / threshold) * surrogate(v, threshold) -- the
        # chain rule of the same surrogate for s ~ v / threshold. ALIF's adaptation part of the threshold
        # stays without gradient, as with a fixed threshold.
        if learn_threshold not in ('none', 'layer', 'neuron'):
            raise ValueError(f"learn_threshold must be 'none', 'layer' or 'neuron', got {learn_threshold!r}")
        self.learn_threshold = learn_threshold
        self.log_threshold = None if learn_threshold == 'none' else nn.Parameter(
            torch.full((1 if learn_threshold == 'layer' else n_out,), math.log(threshold)))
        self.spike_fn = spike_fn
        self.surrogate = get_surrogate_grad_fn(surrogate_grad)
        self.norm_syn = norm_syn
        # Time constants are in frames (80 ms with stride 4); with `substeps` steps per frame each one is
        # applied per step as tau * substeps, so it keeps its length in real time.
        # IAF: no leak; LIF/ALIF: leak at tau_mem. Fixed, unless a range is given: tau_mem_range /
        # tau_adapt_range (lo, hi) = one trainable time constant per neuron, log-spaced over [lo, hi]
        # (kept positive as exp(log_tau)); a tau_mem_range gives every neuron type a leak. tau_syn is trained.
        self.substeps = int(substeps)
        k = self.substeps
        spread = lambda lo, hi: torch.log(torch.logspace(math.log10(lo), math.log10(hi), n_out))
        self.log_tau_mem = nn.Parameter(spread(*tau_mem_range)) if tau_mem_range else None
        self.log_tau_adapt = nn.Parameter(spread(*tau_adapt_range)) if tau_adapt_range else None
        self.alpha_mem = 1.0 if neuron_type == 'iaf' else math.exp(-1.0 / (tau_mem * k))
        self.alpha_adapt = math.exp(-1.0 / (tau_adapt * k))
        self.tau_syn = nn.Parameter(torch.tensor(float(tau_syn))) if tau_syn else None
        # tau_syn_range (lo, hi): one trainable synaptic time constant per neuron, log-spaced over [lo, hi]
        # (replaces the single tau_syn)
        tau_syn_range = tau_syn_range if tau_syn_range and min(tau_syn_range) > 0 else None   # (0, 0) = off
        self.log_tau_syn = nn.Parameter(spread(*tau_syn_range)) if tau_syn_range else None
        if self.log_tau_syn is not None:
            self.tau_syn = None
        # neuron_type 'seadlif': adaptive LIF with subthreshold coupling, symplectic-Euler discretization (Baronig
        # et al., Nat. Commun. 2025). Per neuron, trained: membrane and adaptation time constants, tau = lo +
        # theta (hi - lo) with theta in [0, 1]; coupling a = a_max * a_hat (a_hat in [0, 1]) and spike-triggered
        # adaptation b = b_max * b_hat (b_hat in [0, 1]); all drawn uniformly at the start (fixed seed) and kept
        # in range by clamping. Ranges in steps. Defaults for 20 ms steps: membrane 20-500 ms and adaptation
        # 30-500 ms (the paper's 30-300 ms, reaching the LSNN's 500 ms); a, b <= 1, which keeps both layers firing
        # near the LSNN's rates at the start (the paper's a up to 120 made gradients explode at 20 ms steps).
        if neuron_type == 'seadlif':
            if adaptive_fraction:
                raise ValueError("neuron_type 'seadlif' adapts every neuron; leave adaptive_fraction at 0")
            cfg = {**dict(tau_u=(1.0, 25.0), tau_w=(1.5, 25.0), a_max=1.0, b_max=1.0), **(seadlif or {})}
            self.sead_tau_u, self.sead_tau_w = tuple(cfg['tau_u']), tuple(cfg['tau_w'])
            self.sead_a_max, self.sead_b_max = float(cfg['a_max']), float(cfg['b_max'])
            g = torch.Generator().manual_seed(7)
            for name in ('sead_theta_u', 'sead_theta_w', 'sead_a', 'sead_b'):
                setattr(self, name, nn.Parameter(torch.rand(n_out, generator=g)))

    def seadlif_params(self):
        """SE-adLIF per-neuron decays (membrane, adaptation) and couplings a, b, from the clamped parameters."""
        span = lambda lo_hi, th: lo_hi[0] + th.clamp(0, 1) * (lo_hi[1] - lo_hi[0])
        alpha_u = torch.exp(-1.0 / (span(self.sead_tau_u, self.sead_theta_u) * self.substeps))
        alpha_w = torch.exp(-1.0 / (span(self.sead_tau_w, self.sead_theta_w) * self.substeps))
        return alpha_u, alpha_w, self.sead_a_max * self.sead_a.clamp(0, 1), self.sead_b_max * self.sead_b.clamp(0, 1)

    def init_state(self, n: int, device) -> dict:
        z = lambda: torch.zeros(n, self.width, device=device)
        st = {'v': z(), 'i': z(), 'b': z(), 's': z(), 'v_pre': z()}
        if self.rec_delay_values or self.rec_syn_delay:
            # spikes of the last max-delay + 1 steps, newest first (hist[k] = spikes k + 1 steps ago)
            d_max = self.rec_delay_values[-1] if self.rec_delay_values else self.rec_syn_delay
            st['hist'] = torch.zeros(d_max + 1, n, self.width, device=device)
        if self.ff_syn_delay:
            st['ff_hist'] = torch.zeros(self.ff_syn_delay + 1, n, self.ff.in_features, device=device)
        return st

    def delayed_drive(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        """One step's feedforward current through the per-synapse delays (ff_syn_delay): x (N, n_in) is this
        step's input; the inputs of the last D steps are kept in st['ff_hist'], newest first."""
        st['ff_hist'] = torch.cat([self.dropout(x)[None], st['ff_hist'][:-1]], 0)
        if '_ffwd' not in st:                       # delay-sorted weights, built once per sequence
            masks = self.ff_delay.kernel() if hasattr(self, 'ff_delay') else self.ff_syn_masks
            st['_ffwd'] = (self.ff.weight.unsqueeze(0) * masks).transpose(1, 2).reshape(-1, self.width)
        h = st['ff_hist']
        return h.transpose(0, 1).reshape(h.shape[1], -1) @ st['_ffwd']

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward input current for every step at once: (N, T, n_in) -> (N, T, n_out)."""
        return self.ff(self.dropout(x))

    def step(self, current: torch.Tensor, st: dict) -> torch.Tensor:
        """One time step given this step's feedforward current; updates st, returns spikes."""
        if self.rec is not None:
            s_rec = st['s']
            if self.rec_delay_values:
                s_rec = sum(st['hist'][d] * m for d, m in zip(self.rec_delay_values, self.rec_delay_masks))
            if self.rec_no_self:
                # the masked recurrent weights, built once per sequence (st['_w']): rebuilt every step, training
                # stored one copy per step (~4 MB per step and layer at 1024 wide)
                if '_w' not in st:
                    st['_w'] = self.rec.weight * self.rec_offdiag
                w = st['_w']
            else:
                w = self.rec.weight
            if self.rec_syn_delay:
                # the delay-sorted weights as one (delays x inputs, outputs) matrix, built once per sequence
                # (st['_wd'], kept through truncation): each step is then one matrix product of the spike
                # history with that shared matrix, so training stores only the history per step, not a copy
                # of the weights (a per-step einsum copied the weights, ~24 MB per step and layer)
                if '_wd' not in st:
                    masks = self.rec_delay.kernel() if hasattr(self, 'rec_delay') else self.rec_syn_masks
                    st['_wd'] = (w.unsqueeze(0) * masks).transpose(1, 2).reshape(-1, w.shape[0])
                hist = st['hist']
                current = current + hist.transpose(0, 1).reshape(hist.shape[1], -1) @ st['_wd']
            else:
                current = current + torch.nn.functional.linear(s_rec, w)
        if self.tau_syn is not None or self.log_tau_syn is not None:
            tau = self.tau_syn.clamp(min=0.05) if self.log_tau_syn is None else self.log_tau_syn.exp()
            alpha = torch.exp(-1.0 / (tau * self.substeps))
            if self.norm_syn:
                current = current * (1 - alpha) / alpha
            st['i'] = alpha * (st['i'] + current)
            current = st['i']
        if self.neuron_type == 'seadlif':
            # SE-adLIF (Baronig et al., Nat. Commun. 2025), symplectic-Euler order: membrane from the previous
            # adaptation current, spike, reset, then the adaptation current from the reset membrane and the spikes
            # u <- a_u u + (I - w);  w <- a_w w + (1 - a_w)(a u + b s). The input enters at full strength, as in
            # this model's other neurons (the paper scales it by 1 - a_u, which at 20 ms steps left the layers
            # nearly silent); the decay per step is still sqrt(a_u a_w), independent of a, for moderate a.
            alpha_u, alpha_w, a, b = self.seadlif_params()
            v = alpha_u * st['v'] + (current - st['b'])
        else:
            alpha_mem = self.alpha_mem if self.log_tau_mem is None else \
                torch.exp(-1.0 / (self.log_tau_mem.exp() * self.substeps))
            v = alpha_mem * st['v'] + current
        st['v_pre'] = v                                  # before spiking and reset (membrane readout)
        base = self.threshold if self.log_threshold is None else self.log_threshold.exp()
        if self.neuron_type == 'seadlif':
            thr = torch.zeros_like(v) + base             # adaptation acts through the current, not the threshold
        elif self.neuron_type == 'alif':
            thr = base + self.adapt_scale * st['b']
        elif self.adapt_mask is not None:                 # mixed population: only the adaptive neurons adapt
            thr = base + self.adapt_scale * st['b'] * self.adapt_mask
        else:
            thr = torch.zeros_like(v) + base
        if self.log_threshold is None:
            s = self.spike_fn.apply(v, thr, self.surrogate)
        else:
            thr_d = thr.detach()
            s = self.spike_fn.apply(v, thr_d, self.surrogate)
            with torch.no_grad():
                ds_dthr = -(v / thr_d) * self.surrogate(v, thr_d)
            s = s + ds_dthr * (base - base.detach())     # zero in value; gradient to the base threshold only
        if self.reset_type == 'soft':
            v = v - s * thr
        else:
            v = v * (s == 0).float()
        if self.min_vmem is not None:
            v = torch.relu(v - self.min_vmem) + self.min_vmem
        if self.max_vmem is not None:
            v = torch.clamp(v, max=self.max_vmem)
        if self.neuron_type == 'seadlif':
            st['b'] = alpha_w * st['b'] + (1 - alpha_w) * (a * v + b * s)
        elif self.neuron_type == 'alif' or self.adapt_mask is not None:
            alpha_adapt = self.alpha_adapt if self.log_tau_adapt is None else \
                torch.exp(-1.0 / (self.log_tau_adapt.exp() * self.substeps))
            st['b'] = alpha_adapt * st['b'] + (1 - alpha_adapt) * s
        st['v'], st['s'] = v, s
        if self.rec_delay_values or self.rec_syn_delay:
            st['hist'] = torch.cat([s[None], st['hist'][:-1]], 0)
        return s


class SNN_Recurrent(nn.Module):
    def __init__(
        self,
        n_sessions: int,
        num_input_channels: int = 256,
        n_classes: int = N_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        input_window: int = 32,
        input_stride: int = 4,
        recurrent: bool = True,
        neuron_type: str = 'alif',
        reset_type: str = 'soft',
        spike_thresholds: Optional[List[float]] = None,
        tau_mem: float = 1000.0,
        tau_syn: Optional[List[float]] = None,
        tau_adapt: float = 6.0,
        adapt_scale: float = 1.8,
        min_vmem: Optional[float] = -1.0,
        spike_fn=None,
        surrogate_grad='periodic_exponential',
        norm_syn: bool = True,
        input_dropout: float = 0.2,
        layer_dropout: float = 0.4,
        temporal_decay_init: float = 0.37,
        learnable_temporal_decay: bool = True,
        temporal_decay_stages: int = 1,
        rec_init_scale: float = 1.0,
        rec_init: str = 'default',
        learn_threshold: str = 'none',
        substeps: int = 1,
        readout_source: str = 'spikes',
        tau_mem_range=None,
        tau_adapt_range=None,
        tau_syn_range=None,
        input_layer: str = 'linear',
        input_encoding: str = 'none',
        encoding_threshold: float = 0.25,
        encoding_max: int = 3,
        input_delays=None,
        input_traces=None,
        rec_delays=None,
        rec_delay_fractions=None,
        rec_no_self: bool = False,
        **unused,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [1024] * 5)
        n = len(hidden_dims)
        thresholds = list(spike_thresholds) if spike_thresholds else [1.0]
        thresholds = thresholds * n if len(thresholds) == 1 else thresholds
        taus = list(tau_syn) if isinstance(tau_syn, (list, tuple)) else [tau_syn]
        taus = taus * n if len(taus) == 1 else taus
        if len(thresholds) != n or len(taus) != n:
            raise ValueError(f'spike_thresholds and tau_syn need 1 or {n} values (one per layer)')
        self.frame_kernel, self.frame_stride = int(input_window), int(input_stride)
        self.n_classes = n_classes
        self.recurrent = recurrent
        # substeps: the spiking layers step this many times per output frame, the frame's input held
        # throughout; readout_source 'vmem': the readout reads the last layer's membrane potential
        # (before its spikes and reset) instead of its spikes
        if substeps < 1:
            raise ValueError(f'substeps must be >= 1, got {substeps}')
        if readout_source not in ('spikes', 'vmem'):
            raise ValueError(f"readout_source must be 'spikes' or 'vmem', got {readout_source!r}")
        self.substeps, self.readout_source = int(substeps), readout_source

        # per-session input: 'linear' (the RNN's Linear + softsign) or 'gain' (one gain per channel and session,
        # no mixing: C multiplies per bin instead of C * C)
        if input_layer not in ('linear', 'gain'):
            raise ValueError(f"input_layer must be 'linear' or 'gain', got {input_layer!r}")
        self.input_layer, self.num_input_channels = input_layer, int(num_input_channels)
        self.day_layers = None
        if input_layer == 'linear':
            self.day_layers = nn.ModuleList()
            for _ in range(n_sessions):
                layer = nn.Linear(num_input_channels, num_input_channels)
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.day_layers.append(layer)
        else:
            self.day_gain = nn.Parameter(torch.ones(n_sessions, num_input_channels))
        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)
        # input events and first-layer memory (frontend.py)
        self.encoder = EventEncoder(input_encoding, encoding_threshold, encoding_max)
        self.memory = InputMemory(input_delays, input_traces)

        spike_fn = spike_fn or sinabs.activation.MultiSpike
        dims = [self.memory.expand(num_input_channels * self.frame_kernel)] + hidden_dims
        self.layers = nn.ModuleList([
            SpikingLayer(dims[i], dims[i + 1], recurrent, neuron_type, reset_type, thresholds[i], tau_mem,
                         taus[i], tau_adapt, adapt_scale, min_vmem, spike_fn, surrogate_grad, norm_syn,
                         layer_dropout, rec_init_scale, rec_init, learn_threshold, tau_mem_range,
                         tau_adapt_range, substeps, tau_syn_range,
                         rec_delays=rec_delays, rec_delay_fractions=rec_delay_fractions, rec_no_self=rec_no_self)
            for i in range(n)])
        self.layer_widths = hidden_dims
        self.total_neuron_units = sum(hidden_dims)

        if temporal_decay_stages < 0:
            raise ValueError(f'temporal_decay_stages must be >= 0, got {temporal_decay_stages}')
        self.temporal_decay_stages = temporal_decay_stages
        self._temporal_decay_raw = nn.Parameter(torch.logit(torch.tensor(float(temporal_decay_init))),
                                                requires_grad=learnable_temporal_decay)
        self.readout = nn.Linear(hidden_dims[-1], n_classes)
        nn.init.kaiming_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self._stream = None

    def tau_summary(self) -> Optional[str]:
        """Trained per-neuron time constants per layer, in frames: mean [min, max], or None when fixed."""
        parts = []
        for i, l in enumerate(self.layers):
            for name, p in (('syn', l.log_tau_syn), ('mem', l.log_tau_mem), ('adapt', l.log_tau_adapt)):
                if p is not None:
                    t = p.detach().exp()
                    parts.append(f"L{i + 1} {name} {t.mean():.2f} [{t.min():.2f}, {t.max():.2f}]")
        return ' | '.join(parts) if parts else None

    def threshold_summary(self) -> Optional[str]:
        """Learned base thresholds per layer (mean [min, max]), or None when they are fixed."""
        parts = [f"{t.mean():.3f} [{t.min():.3f}, {t.max():.3f}]"
                 for t in (l.log_threshold.detach().exp() for l in self.layers if l.log_threshold is not None)]
        return ' | '.join(parts) if parts else None

    # ------------------------------------------------------------------ framing
    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Output frames for trials of `lengths` input bins (the RNN's 'valid' stacking)."""
        return torch.clamp((lengths - self.frame_kernel) // self.frame_stride + 1, min=0)

    def _day(self, x: torch.Tensor, day_idx: int) -> torch.Tensor:
        """Per-session input layer and dropout, per bin: (..., C) -> (..., C)."""
        if self.day_layers is not None:
            return self.day_dropout(self.day_activation(self.day_layers[day_idx](x)))
        return self.day_dropout(x * self.day_gain[day_idx])

    def _day_ops(self) -> int:
        """Multiplies per bin of the per-session input layer."""
        c = self.num_input_channels
        return c * c if self.day_layers is not None else c

    def _stack(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, C) -> (N, T_out, window * C), frame k = bins [k*stride, k*stride + window)."""
        if self.frame_kernel == 1 and self.frame_stride == 1:
            return x
        frames = x.unfold(1, self.frame_kernel, self.frame_stride)       # (N, T_out, C, window)
        return frames.transpose(2, 3).reshape(x.shape[0], frames.shape[1], -1)

    @property
    def temporal_decay(self) -> float:
        return torch.sigmoid(self._temporal_decay_raw).item()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, day_idx: int, lengths: Optional[torch.Tensor] = None,
                count_ops: bool = False, reset_state: bool = True) -> torch.Tensor:
        """x (N, T, C) -> logits (N, T_out, n_classes); see output_lengths()."""
        N = x.shape[0]
        if lengths is None:
            lengths = torch.full((N,), x.shape[1], dtype=torch.long)
        out_len = self.output_lengths(lengths.to(x.device))
        u = self.encoder(self._day(x, day_idx))                  # per bin; signed event counts if encoding
        if self.encoder.active:
            bins = (torch.arange(u.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]).float()
            self.input_events_soft = (u.abs() * bins[..., None]).sum()     # its gradient: straight through
            self.input_events = self.input_events_soft.detach()
            self.input_event_slots = float(bins.sum().item()) * u.shape[2]     # channels x valid bins
        pre = self.memory.delay(self._stack(u))                   # frames, with delayed copies
        x = self.memory.trace(pre)                                # + filtered copies: the first layer's input
        T, K = x.shape[1], self.substeps
        mask = (torch.arange(T, device=x.device)[None, :] < out_len[:, None]).float()[..., None]
        mask_k = mask.repeat_interleave(K, dim=1)                 # per step (T * K steps)
        n_valid = int(mask.sum().item())
        n_valid_k = n_valid * K

        op_mac, op_acc, op_elementwise = 0, 0.0, 0
        if count_ops:
            # input layer: MAC per nonzero input per output, softsign per output (per 20 ms bin)
            n_bins = int(lengths.sum().item())
            op_mac += n_bins * self._day_ops()
            op_elementwise += n_bins * self.num_input_channels

        self.layer_spikes, self.layer_step_rates = [], []
        self.step_mask = mask_k[..., 0]
        for li, layer in enumerate(self.layers):
            if li == 0:
                if count_ops:
                    if self.encoder.active:
                        # events: each adds its weight column (one per delay copy, inside `pre`) to each of the
                        # n_traces synaptic currents; the filtered currents decay once per frame
                        acc0 = float((pre.abs() * mask).sum().item()) * layer.width * self.memory.n_traces
                        mac0 = n_valid * layer.width * self.memory.n_decaying
                    else:
                        acc0, mac0 = 0.0, int(torch.count_nonzero(x * mask).item()) * layer.width   # dense
                    op_acc += acc0
                    op_mac += mac0
                    self.input_ops = {'mac': mac0, 'acc': acc0, 'frames': n_valid}
                drive = layer.drive(x).repeat_interleave(K, dim=1)    # the frame's input held for its K steps
            else:
                if count_ops:
                    op_acc += float((x * mask_k).sum().item()) * layer.width           # one ACC per spike
                drive = layer.drive(x)
            st = layer.init_state(N, x.device)
            spikes, vmem = [], []
            for t in range(T * K):
                spikes.append(layer.step(drive[:, t], st))
                vmem.append(st['v_pre'])
            x = torch.stack(spikes, dim=1)
            self.layer_spikes.append((x * mask_k).sum().clamp(min=0))  # per frame: summed over its steps
            self.layer_step_rates.append(x.mean(-1))                     # (N, steps), for the branching loss
            if count_ops:
                if layer.rec is not None:   # last step's spikes, one ACC each per neuron
                    prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
                    op_acc += float((prev * mask_k).sum().item()) * layer.width
                if layer.tau_syn is not None or layer.log_tau_syn is not None:
                    op_mac += n_valid_k * layer.width
                op_elementwise += 2 * n_valid_k * layer.width
                if layer.neuron_type != 'iaf' or layer.log_tau_mem is not None:
                    op_mac += n_valid_k * layer.width
                if layer.neuron_type == 'alif':
                    op_mac += 2 * n_valid_k * layer.width
        self.total_spikes = sum(self.layer_spikes)
        if self.readout_source == 'vmem':
            x = torch.stack(vmem, dim=1)

        decay = torch.sigmoid(self._temporal_decay_raw) ** (1.0 / K)   # per step: the same time per frame
        for _ in range(self.temporal_decay_stages):
            state = torch.zeros(N, x.shape[-1], device=x.device)
            smoothed = []
            for t in range(T * K):
                state = state + (1 - decay) * (x[:, t] - state)
                smoothed.append(state)
            x = torch.stack(smoothed, dim=1)
        x = x[:, K - 1::K]                                        # each frame's last step
        if count_ops:
            op_mac += self.temporal_decay_stages * n_valid_k * x.shape[-1]
            op_elementwise += self.temporal_decay_stages * n_valid_k * x.shape[-1]
            op_mac += int(torch.count_nonzero(x * mask).item()) * self.n_classes
            self.op_counts = {'mac': op_mac, 'acc': op_acc, 'elementwise': op_elementwise}
        return self.readout(x)

    # ------------------------------------------------------------------ streaming
    def reset_stream(self):
        """Start a new stream (e.g. a new sentence) from rest."""
        self._stream = None

    def step(self, x: torch.Tensor, day_idx: int) -> Optional[torch.Tensor]:
        """One 20 ms bin x (N, C). Returns logits (N, n_classes) when it completes
        an output frame (every frame_stride bins once frame_kernel bins are in),
        else None. Equivalent to forward() over the same stream."""
        x = self._day(x, day_idx)
        if self._stream is None:
            self._stream = {'buffer': [], 'n': 0, 'enc': self.encoder.init_state(x.shape[0], x.shape[1], x.device),
                            'mem': self.memory.init_state(),
                            'states': [layer.init_state(x.shape[0], x.device) for layer in self.layers],
                            'ema': [torch.zeros(x.shape[0], self.layer_widths[-1], device=x.device)
                                    for _ in range(self.temporal_decay_stages)]}
        st = self._stream
        if self.encoder.active:
            x, st['enc'] = self.encoder.step(x, st['enc'])
        st['buffer'] = (st['buffer'] + [x])[-self.frame_kernel:]
        st['n'] += 1
        if st['n'] < self.frame_kernel or (st['n'] - self.frame_kernel) % self.frame_stride:
            return None
        frame = torch.stack(st['buffer'], 1).reshape(x.shape[0], -1)   # (N, window * C), time-major
        frame = self.memory.step(frame, st['mem'])
        drive0 = self.layers[0].drive(frame)                          # held for the frame's steps
        decay = torch.sigmoid(self._temporal_decay_raw) ** (1.0 / self.substeps)
        for _ in range(self.substeps):
            h = None
            for li, (layer, ls) in enumerate(zip(self.layers, st['states'])):
                h = layer.step(drive0 if li == 0 else layer.drive(h), ls)
            if self.readout_source == 'vmem':
                h = st['states'][-1]['v_pre']
            for s in range(self.temporal_decay_stages):
                st['ema'][s] = st['ema'][s] + (1 - decay) * (h - st['ema'][s])
                h = st['ema'][s]
        return self.readout(h)
