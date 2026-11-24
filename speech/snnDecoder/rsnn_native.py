"""
Spike-native recurrent decoder (`--arch native`): the SNN reads raw threshold-crossing counts of the 128
area-6v electrodes in 20 ms bins (speech/analysis/exportRawCountsToNpz.py, derived/snnDataTx1) and steps
once per bin, so every layer -- the first included -- is driven by events:

    input           raw counts, (N, T, 128) integers; optionally a channel range of wider data
                    (input_channels, e.g. the first 128 = normalized tx1 of derived/snnData)
    per-day input   'gain': one learned gain per channel and recording day (starting at 1), which folds into
                    the first layer's weights, so a count still costs one accumulate per synapse;
                    'linear': a dense per-day Linear + softsign (the RNN's input layer, not spike-native);
                    'none'. Then dropout.
    input events    optional (input_encoding, frontend.py): continuous features (e.g. the 256 normalized
                    features of derived/snnData) become signed event counts, one step per bin, so the first
                    layer adds a weight per event as it does for raw counts
    spiking layers  hidden_dims recurrent ALIF layers (rsnn.SpikingLayer): optional per-neuron trainable
                    synaptic time constants (tau_syn_range, in 20 ms steps), threshold adaptation, soft reset;
                    feedback=True adds top-down input from the last layer's previous-step spikes into layer 1
    output          41 leaky-integrator neurons that never spike, one per class (39 phonemes, SIL, blank):
                    v_k <- a_k v_k + (1 - a_k) (W s(t) + b)_k with one trainable time constant per class
                    (out_tau_init steps, trained unless out_tau_train=False); the logits are v. Softmax only
                    turns them into probabilities in the CTC loss.

Training on whole sentences backpropagates through hundreds of 20 ms steps, where gradients overflow; tbptt=K
(truncated backpropagation through time) keeps the forward pass over the whole sentence but cuts the gradient
path every K steps (the carried state keeps its value). It only acts in training mode.

One output per 20 ms bin. Every operation is causal; step() takes one bin and returns its logits, the
same as forward() over the stream. Same interface as model.SNN_Speech / rsnn.SNN_Recurrent.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import sinabs.activation

from .frontend import EventEncoder, InputMemory
from .model import N_CLASSES
from .delays import DelayPositions
from .rsnn import SpikingLayer



class _Mask(nn.Module):
    """A fixed 0/1 mask on a weight (torch parametrization): the masked weights stay zero and get no gradient."""
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.mask

class SNN_Native(nn.Module):
    def __init__(
        self,
        n_sessions: int,
        num_input_channels: int = 128,
        n_classes: int = N_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        input_channels: Optional[List[int]] = None,
        input_channel_list: Optional[List[int]] = None,
        day_input: str = 'gain',
        input_dropout: float = 0.2,
        layer_dropout: float = 0.2,
        neuron_type: str = 'alif',
        reset_type: str = 'soft',
        spike_thresholds: Optional[List[float]] = None,
        tau_mem: float = 1000.0,
        tau_adapt: float = 24.0,
        adapt_scale: float = 1.8,
        tau_syn_range: Optional[List[float]] = None,
        min_vmem: Optional[float] = -1.0,
        spike_fn=None,
        surrogate_grad='periodic_exponential',
        norm_syn: bool = True,
        recurrent: bool = True,
        rec_init: str = 'orthogonal',
        rec_init_scale: float = 1.0,
        feedback: bool = False,
        out_tau_init: float = 2.0,
        out_tau_train: bool = True,
        tau_mem_range=None,
        tau_adapt_range=None,
        learn_threshold: str = 'none',
        tbptt: int = 0,
        max_vmem: Optional[float] = None,
        input_encoding: str = 'none',
        encoding_threshold: float = 0.25,
        encoding_max: int = 3,
        encoding_levels: Optional[List[float]] = None,
        input_delays=None,
        input_traces=None,
        output_stride: int = 1,
        rec_delays=None,
        rec_delay_fractions=None,
        rec_no_self: bool = False,
        syn_delay_in: int = 0,
        syn_delay_rec=0,
        syn_delay_ff: int = 0,
        syn_delay_in_mult: int = 1,
        adaptive_fraction: float = 0.0,
        seadlif: Optional[dict] = None,
        out_stages: int = 1,
        learn_delays: bool = False,
        readout_all: bool = False,
        readout_source: str = 'spikes',
        readout_traces: Optional[List[float]] = None,
        input_substeps: int = 1,
        branches: int = 1,
        branch_split: str = 'delay',
        branch_groups: Optional[list] = None,
        **unused,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [1024, 1024])
        n = len(hidden_dims)
        thresholds = list(spike_thresholds) if spike_thresholds else [1.0]
        thresholds = thresholds * n if len(thresholds) == 1 else thresholds
        if len(thresholds) != n:
            raise ValueError(f'spike_thresholds needs 1 or {n} values')
        if day_input not in ('gain', 'linear', 'none'):
            raise ValueError(f"day_input must be 'gain', 'linear' or 'none', got {day_input!r}")
        self.input_channels = list(input_channels) if input_channels else None
        # input_channel_list: an explicit channel subset (e.g. the top-ranked electrodes, scripts/rank_channels.py)
        # in place of the contiguous range
        self.input_channel_list = [int(c) for c in input_channel_list] if input_channel_list else None
        if self.input_channel_list:
            self.register_buffer('channel_index', torch.tensor(self.input_channel_list, dtype=torch.long),
                                 persistent=False)
            n_in = len(self.input_channel_list)
        else:
            n_in = (self.input_channels[1] - self.input_channels[0]) if self.input_channels else num_input_channels
        self.n_in, self.n_classes, self.day_input = n_in, n_classes, day_input
        # one output per output_stride steps (1: every 20 ms bin; 4: every 80 ms, as the RNN's CTC frames),
        # the network still stepping every bin; substeps = steps per output (for op counting)
        self.output_stride = int(output_stride)
        if self.output_stride < 1:
            raise ValueError('output_stride must be >= 1')
        # input_substeps K > 1: each 20 ms bin becomes K chip steps, its count (after the per-day gain, in units of
        # encoding_threshold, at most K) spread evenly over them as single events: a binary input stream that keeps
        # the counts, as the audio Speck decoder's sub-steps. Delays, the output time constants, tbptt and the
        # output stride stay in 20 ms bins (scaled by K inside); for the chip network (IAF, no filters) only
        self.input_substeps = int(input_substeps)
        if self.input_substeps > 1 and (neuron_type != 'iaf' or input_encoding != 'none' or input_traces):
            raise ValueError("input_substeps is for the chip network: neuron_type 'iaf', no input_encoding or traces")
        self._ostride = self.output_stride * self.input_substeps        # chip steps per output
        self.frame_kernel, self.frame_stride, self.substeps = 1, self.output_stride, self._ostride

        if day_input == 'gain':
            self.day_gain = nn.Parameter(torch.ones(n_sessions, n_in))
        elif day_input == 'linear':
            self.day_layers = nn.ModuleList()
            for _ in range(n_sessions):
                layer = nn.Linear(n_in, n_in)
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.day_layers.append(layer)
        self.in_dropout = nn.Dropout(input_dropout)
        self.encoder = EventEncoder(input_encoding, encoding_threshold, encoding_max, encoding_levels)
        # first-layer memory of past input, in 20 ms steps (frontend.py): synaptic delays, and synaptic currents
        # with several time constants
        self.memory = InputMemory([d * self.input_substeps for d in input_delays] if input_delays else None,
                                  input_traces)

        spike_fn = spike_fn or sinabs.activation.MultiSpike
        dims = [self.memory.expand(n_in * self.encoder.n_lines)] + hidden_dims
        # syn_delay_rec: one maximum recurrent delay for every layer, or one per layer (e.g. growing with depth)
        rec_d = list(syn_delay_rec) if isinstance(syn_delay_rec, (list, tuple)) else [syn_delay_rec or 0]
        rec_d = rec_d * n if len(rec_d) == 1 else rec_d
        if len(rec_d) != n:
            raise ValueError(f'syn_delay_rec needs 1 or {n} values (one per layer)')
        self.layers = nn.ModuleList([
            SpikingLayer(dims[i], dims[i + 1], recurrent, neuron_type, reset_type, thresholds[i], tau_mem,
                         None, tau_adapt, adapt_scale, min_vmem, spike_fn, surrogate_grad, norm_syn,
                         input_dropout if i == 0 else layer_dropout, rec_init_scale, rec_init, learn_threshold,
                         tau_mem_range, tau_adapt_range, 1, tau_syn_range, max_vmem,
                         rec_delays=rec_delays, rec_delay_fractions=rec_delay_fractions, rec_no_self=rec_no_self,
                         rec_syn_delay=int(rec_d[i]), ff_syn_delay=int(syn_delay_ff) if i > 0 else 0,
                         adaptive_fraction=adaptive_fraction, learn_delays=learn_delays,
                         seadlif=seadlif)
            for i in range(n)])
        # the first layer's input dropout is applied once, before the per-day gain
        self.layers[0].dropout = nn.Identity()
        # branches K > 1: K separate feedforward chains side by side (Speck: each its own cores), joined only at the
        # host readout, which reads every chain's last layer. Each layer's weights are block-diagonal: chain k's
        # neurons are the k-th block of every hidden layer, and layer 1 of chain k reads only its share of the input
        # lines, split by delay, level or channel (lines are laid out delay by delay, level by level, channel by
        # channel). Trained together, so the chains share the output timing (separately trained nets do not)
        self.branches, self.branch_split = int(branches), branch_split   # masked below, once feedback is known
        # branch_groups: chains of unequal size, each with its own input lines: one dict per chain, {'channels':
        # feature ids (a subset of input_channel_list), 'delays': a subset of input_delays, 'dims': its width in
        # each hidden layer}; hidden_dims must be the per-layer sums. Overrides branches/branch_split
        self.branch_groups = [dict(g) for g in branch_groups] if branch_groups else None
        if self.branch_groups:
            self.branches = len(self.branch_groups)
        self.branch_widths = None
        # syn_delay_in D > 0: heterogeneous synaptic delays on the input connections -- every input synapse of
        # layer 1 its own delay, uniform over 0..D steps (fixed, seed 2, saved with the weights). One weight per
        # synapse; on event-driven hardware each input event costs one accumulate per synapse, as without delays
        self.syn_delay_in = int(syn_delay_in)
        if self.syn_delay_in > 0:
            w0 = self.layers[0].ff.weight
            idx = torch.randint(0, self.syn_delay_in + 1, tuple(w0.shape), generator=torch.Generator().manual_seed(2))
            if learn_delays:                         # learnable, starting from these delays (delays.py)
                self.in_delay = DelayPositions(idx, self.syn_delay_in)
            else:
                self.register_buffer('in_syn_masks', torch.stack([idx == d for d in range(self.syn_delay_in + 1)]))
        # syn_delay_in_mult M > 1: each input channel reaches each layer-1 neuron through M synapses, each with
        # its own weight and its own random delay (several synaptic contacts along axon branches of different
        # lengths): M samples of the input's past per pair instead of one. M times the first layer's weights; all
        # start at 1/sqrt(M) of the usual scale so the summed drive starts the same
        self.in_mult = int(syn_delay_in_mult)
        if self.in_mult > 1:
            if not self.syn_delay_in:
                raise ValueError('syn_delay_in_mult needs syn_delay_in > 0')
            w0 = self.layers[0].ff.weight
            extra = torch.empty(self.in_mult - 1, *w0.shape)
            for k in range(self.in_mult - 1):
                nn.init.kaiming_uniform_(extra[k])
            self.in_extra = nn.Parameter(extra / math.sqrt(self.in_mult))
            with torch.no_grad():
                w0.div_(math.sqrt(self.in_mult))
            idx_x = torch.randint(0, self.syn_delay_in + 1, (self.in_mult - 1, *w0.shape),
                                  generator=torch.Generator().manual_seed(4))
            if learn_delays:
                self.in_extra_delay = DelayPositions(idx_x, self.syn_delay_in)
            else:
                self.register_buffer('in_extra_masks',
                                     torch.stack([idx_x == d for d in range(self.syn_delay_in + 1)], 1))
        self.feedback = nn.Linear(hidden_dims[-1], hidden_dims[0], bias=False) if feedback and n > 1 else None
        if self.feedback is not None:
            nn.init.orthogonal_(self.feedback.weight)
            with torch.no_grad():
                self.feedback.weight.mul_(0.5)
        if self.branch_groups:
            self._mask_branch_groups(n_in, hidden_dims)
        elif self.branches > 1:
            self._mask_branches(n_in, hidden_dims)
        self.layer_widths = hidden_dims
        self.total_neuron_units = sum(hidden_dims)

        # readout_all: the output neurons read every spiking layer, not only the last (as the 3-layer LSNN of
        # Bellec et al., Nat Commun 2020, on TIMIT): each layer's spikes reach the outputs directly
        self.readout_all = bool(readout_all)
        # readout_source 'vmem': the output neurons read the last layer's membrane potentials (before spiking and
        # reset) instead of its spikes: continuous values, so a multiply-add per neuron and class every step rather
        # than an add per spike. Tests whether the spikes' few bits per step limit accuracy
        if readout_source not in ('spikes', 'vmem'):
            raise ValueError(f"readout_source must be 'spikes' or 'vmem', got {readout_source!r}")
        self.readout_vmem = readout_source == 'vmem'
        # readout_traces: the readout also reads leaky traces of the spikes it reads, one copy per time constant
        # (20 ms steps; 0 = the spikes themselves): host-side memory for a chip network that has none. Each
        # decaying copy costs a multiply-add per neuron and step, and is read densely
        self.readout_traces = [float(t) for t in readout_traces] if readout_traces else None
        self.readout_alphas = [0.0 if t <= 0 else math.exp(-1.0 / (t * self.input_substeps))
                               for t in self.readout_traces] \
            if self.readout_traces else None
        n_read = sum(hidden_dims) if self.readout_all else hidden_dims[-1]
        self.readout = nn.Linear(n_read * (len(self.readout_traces) if self.readout_traces else 1), n_classes)
        nn.init.kaiming_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.out_log_tau = nn.Parameter(torch.full((n_classes,), math.log(out_tau_init * self.input_substeps)),
                                        requires_grad=out_tau_train)
        # out_stages S > 1: S leaky stages in a row per class (stage 1 integrates the readout, each next stage
        # the one before, all with the class's time constant), the logits = the last stage: a smoother,
        # second-order low-pass output, as the EMA stages of the earlier SNN
        self.out_stages = int(out_stages)
        if self.out_stages < 1:
            raise ValueError('out_stages must be >= 1')
        self.tbptt = int(tbptt) * self.input_substeps
        self._stream = None

    def _mask_branches(self, n_in: int, hidden_dims: List[int]):
        import torch.nn.utils.parametrize as P
        K = self.branches
        if any(h % K for h in hidden_dims):
            raise ValueError(f'every hidden width must divide into {K} branches')
        if self.recurrent_any() or self.feedback is not None or self.syn_delay_in or self.in_mult > 1:
            raise ValueError('branches are for the feedforward chip network')
        D = self.memory.n_delays if self.memory.delays else 1
        L, C = self.encoder.n_lines, n_in
        j = torch.arange(D * L * C)
        key, n_key = {'delay': (j // (L * C), D), 'level': ((j // C) % L, L), 'channel': (j % C, C)}[self.branch_split]
        if n_key % K:
            raise ValueError(f'{n_key} {self.branch_split}s do not split into {K} branches')
        in_branch = key * K // n_key
        self.branch_widths = [[h // K] * K for h in hidden_dims]
        for i, layer in enumerate(self.layers):
            w = layer.ff.weight
            out_branch = torch.arange(w.shape[0]) * K // w.shape[0]
            src = in_branch if i == 0 else torch.arange(w.shape[1]) * K // w.shape[1]
            mask = (out_branch[:, None] == src[None, :]).float()
            with torch.no_grad():
                w.mul_(math.sqrt(K))          # each neuron reads 1/K of the inputs: keep the initial drive
            P.register_parametrization(layer.ff, 'weight', _Mask(mask))

    def _mask_branch_groups(self, n_in: int, hidden_dims: List[int]):
        """Block-diagonal layers for chains of unequal size (branch_groups): chain g's neurons are the g-th block of
        every hidden layer (widths g['dims']), and its layer 1 reads only its channels at its delays."""
        import torch.nn.utils.parametrize as P
        G = self.branch_groups
        if self.recurrent_any() or self.syn_delay_in or self.in_mult > 1:
            raise ValueError('branch groups are for the feedforward chip network')
        if any(len(g['dims']) != len(hidden_dims) for g in G) or \
                [sum(g['dims'][i] for g in G) for i in range(len(hidden_dims))] != list(hidden_dims):
            raise ValueError(f'hidden_dims {hidden_dims} must be the per-layer sums of the groups\' dims')
        delays = self.memory.delays or [0]
        L, C = self.encoder.n_lines, n_in
        chan = self.input_channel_list or list(range(C))
        member = torch.zeros(len(G), len(delays) * L * C)    # a line may feed several chains (sent to each)
        for k, g in enumerate(G):
            for d in g['delays']:
                if int(d) not in delays:
                    raise ValueError(f'group {k}: delay {d} is not one of input_delays {delays}')
                for f in g['channels']:
                    if int(f) not in chan:
                        raise ValueError(f'group {k}: feature {f} is not an input channel')
                    c = chan.index(int(f))
                    for lv in range(L):
                        member[k, delays.index(int(d)) * L * C + lv * C + c] = 1.0
        self.branch_widths = [[g['dims'][i] for g in G] for i in range(len(hidden_dims))]
        for i, layer in enumerate(self.layers):
            w = layer.ff.weight
            out_g = torch.repeat_interleave(torch.arange(len(G)), torch.tensor(self.branch_widths[i]))
            if i == 0:
                mask = member[out_g]
            else:
                src = torch.repeat_interleave(torch.arange(len(G)), torch.tensor(self.branch_widths[i - 1]))
                mask = (out_g[:, None] == src[None, :]).float()
            with torch.no_grad():                 # keep each neuron's initial drive: scale by its fan-in share
                w.mul_(torch.sqrt(w.shape[1] / mask.sum(1).clamp(min=1))[:, None])
            P.register_parametrization(layer.ff, 'weight', _Mask(mask))
        if self.feedback is not None:
            # host-loop recurrence on Speck: the host writes a chain's last-layer spikes back into the ring buffer and
            # sends them, one 20 ms bin later, as binary input lines of the same chain's layer 1 (each line one
            # last-layer neuron; it costs layer 1's core one weight row per line, like any input line). Only chains
            # with 'feedback': true get it; the chains stay independent on the chip
            rows = torch.repeat_interleave(torch.arange(len(G)), torch.tensor(self.branch_widths[0]))
            cols = torch.repeat_interleave(torch.arange(len(G)), torch.tensor(self.branch_widths[-1]))
            fb = torch.tensor([bool(g.get('feedback', False)) for g in G])
            mask = ((rows[:, None] == cols[None, :]) & fb[rows][:, None]).float()
            with torch.no_grad():
                self.feedback.weight.mul_(torch.sqrt(mask.shape[1] / mask.sum(1).clamp(min=1))[:, None])
            P.register_parametrization(self.feedback, 'weight', _Mask(mask))
            self.feedback_binary = True

    def recurrent_any(self) -> bool:
        return any(getattr(l, 'rec', None) is not None for l in self.layers)

    # ------------------------------------------------------------------ helpers
    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return lengths // self.output_stride if self.output_stride > 1 else lengths

    def tau_summary(self) -> Optional[str]:
        parts = []
        for i, l in enumerate(self.layers):
            if l.log_tau_syn is not None:
                t = l.log_tau_syn.detach().exp()
                parts.append(f"L{i + 1} syn {t.mean():.2f} [{t.min():.2f}, {t.max():.2f}]")
        t = self.out_log_tau.detach().exp()
        parts.append(f"out {t.mean():.2f} [{t.min():.2f}, {t.max():.2f}]")
        return ' | '.join(parts)

    def select_channels(self, x: torch.Tensor) -> torch.Tensor:
        """The channels the model reads: (..., C) -> (..., n_in)."""
        if self.input_channel_list:
            return x.index_select(-1, self.channel_index)
        if self.input_channels:
            return x[..., self.input_channels[0]:self.input_channels[1]]
        return x

    def _spread(self, u: torch.Tensor) -> torch.Tensor:
        """(N, T, C) gain-scaled counts -> (N, T * K, C) single events: round(u / threshold), at most K, spread
        evenly over each bin's K steps. Gradient straight through, each step carrying u / (threshold K)."""
        K = self.input_substeps
        u = u.unsqueeze(2) / (self.encoder.threshold * K)                  # (N, T, 1, C), per step
        q = torch.clamp(torch.round(u.detach() * K), 0, K)
        j = torch.arange(K, device=u.device, dtype=u.dtype).view(1, 1, K, 1)
        ev = torch.floor((j + 1) * q / K) - torch.floor(j * q / K)        # 0/1, summing to q over the K steps
        out = u + (ev - u).detach()
        return out.reshape(out.shape[0], -1, out.shape[-1])

    def _input(self, x: torch.Tensor, day_idx: int) -> torch.Tensor:
        """Channel selection, dropout, per-day input: (..., C) -> (..., n_in)."""
        x = self.in_dropout(self.select_channels(x))
        if self.day_input == 'gain':
            return x * self.day_gain[day_idx]
        if self.day_input == 'linear':
            return torch.nn.functional.softsign(self.day_layers[day_idx](x))
        return x

    def _drive0(self, x: torch.Tensor) -> torch.Tensor:
        """Layer 1's input current for a whole sequence (N, T, n_in) -> (N, T, width), through the
        heterogeneous input delays if set: sum over d of (W masked to delay-d synapses) x(t - d)."""
        if not self.syn_delay_in:
            return self.layers[0].drive(x)
        wd, T = self._in_wd(), x.shape[1]
        out = 0
        for d in range(self.syn_delay_in + 1):
            xd = x if d == 0 else torch.cat([torch.zeros_like(x[:, :min(d, T)]), x[:, :max(T - d, 0)]], 1)
            out = out + torch.nn.functional.linear(xd, wd[d])
        return out

    def _in_wd(self) -> torch.Tensor:
        """Layer 1's input weights sorted by delay, (D + 1, width, n_in): for each delay, the sum of the weights
        of a pair's synapses that have that delay."""
        masks = self.in_delay.kernel() if hasattr(self, 'in_delay') else self.in_syn_masks
        wd = self.layers[0].ff.weight.unsqueeze(0) * masks
        if self.in_mult > 1:
            extra = self.in_extra_delay.kernel().transpose(0, 1) if hasattr(self, 'in_extra_delay') \
                else self.in_extra_masks                  # (M - 1, D + 1, width, n_in)
            wd = wd + (self.in_extra.unsqueeze(1) * extra).sum(0)
        return wd

    def _out_alpha(self) -> torch.Tensor:
        return torch.exp(-1.0 / self.out_log_tau.exp())

    def _step_all(self, drive0: torch.Tensor, states: list, out_v: torch.Tensor):
        """One 20 ms step of every layer and the output neurons; returns (top-layer spikes, new out_v)."""
        h, hs = None, []
        for li, (layer, st) in enumerate(zip(self.layers, states)):
            if li == 0:
                cur = drive0
                if self.feedback is not None:
                    fb = states[-1]['s']
                    if getattr(self, 'feedback_binary', False):      # chip input lines: one event or none
                        fb = fb.clamp(max=1.0)
                    cur = cur + self.feedback(fb)     # top-down: last layer, previous step
            else:
                cur = layer.delayed_drive(h, st) if layer.ff_syn_delay else layer.drive(h)
            h = layer.step(cur, st)
            hs.append(st['v_pre'] if self.readout_vmem else h)
        a = self._out_alpha()
        r = torch.cat(hs, -1) if self.readout_all else hs[-1]
        if self.readout_traces:                                # traces live with the last layer's state
            st = states[-1]
            if 'rtr' not in st:
                st['rtr'] = torch.zeros(len(self.readout_alphas), *r.shape, device=r.device)
            st['rtr'] = torch.stack([r if a == 0.0 else a * y + (1 - a) * r
                                     for a, y in zip(self.readout_alphas, st['rtr'])])
            r = st['rtr'].transpose(0, 1).reshape(r.shape[0], -1)
        stages, prev = [], self.readout(r)
        for k in range(self.out_stages):                       # out_v: (stages, N, classes)
            prev = a * out_v[k] + (1 - a) * prev
            stages.append(prev)
        return h, torch.stack(stages)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, day_idx: int, lengths: Optional[torch.Tensor] = None,
                count_ops: bool = False, reset_state: bool = True) -> torch.Tensor:
        """x (N, T, C) raw counts -> logits (N, T, n_classes): the output neurons' potentials."""
        N, T = x.shape[0], x.shape[1]
        if lengths is None:
            lengths = torch.full((N,), T, dtype=torch.long)
        K = self.input_substeps
        if K > 1:                                              # K chip steps per bin, single events
            u = self._spread(self._input(x, day_idx))
            T, bin_lengths, lengths = T * K, lengths, lengths * K
        else:
            u, bin_lengths = self.encoder(self._input(x, day_idx)), lengths   # signed event counts if encoding
        mask = (torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None]).float()[..., None]
        n_valid = int(mask.sum().item())
        if self.encoder.active or K > 1:
            self.input_events_soft = (u.abs() * mask).sum()     # its gradient: straight through
            self.input_events = self.input_events_soft.detach()
            self.input_event_slots = float(n_valid) * u.shape[2]
        pre = self.memory.delay(u)                              # + delayed copies
        drive0 = self._drive0(self.memory.trace(pre))          # + filtered copies: first layer's input current
        states = [layer.init_state(N, x.device) for layer in self.layers]
        out_v = torch.zeros(self.out_stages, N, self.n_classes, device=x.device)
        spikes = [[] for _ in self.layers]
        outs = []
        for t in range(T):
            if self.training and self.tbptt > 0 and t > 0 and t % self.tbptt == 0:
                # truncated backpropagation: same values, gradients stop here
                for st in states:
                    for k in st:
                        if not k.startswith('_'):              # '_' entries: weights, not state
                            st[k] = st[k].detach()
                out_v = out_v.detach()
            _, out_v = self._step_all(drive0[:, t], states, out_v)
            for li, st in enumerate(states):
                spikes[li].append(st['s'])
            outs.append(out_v[-1])
        layer_out = [torch.stack(s, 1) for s in spikes]
        # each layer's mean spikes per neuron at every step (N, T), for the branching-factor loss
        self.layer_step_rates = [s.mean(-1) for s in layer_out]
        self.step_mask = mask[..., 0]
        self.layer_spikes = [(s * mask).sum().clamp(min=0) for s in layer_out]
        self.total_spikes = sum(self.layer_spikes)

        if count_ops:
            op_mac, op_acc, op_elem = 0, 0.0, 0
            xin = self.select_channels(x)
            if self.day_input == 'linear':
                op_mac += int((xin * mask).count_nonzero().item()) * self.n_in + n_valid * self.n_in
            elif (self.encoder.active or K > 1) and self.day_input == 'gain':
                op_mac += n_valid // K * self.n_in               # the gain no longer folds: one multiply per channel
            W, M = self.layer_widths[0] * self.in_mult, self.memory.n_traces     # in_mult synapses per pair
            events = u.detach() if self.encoder.active or K > 1 else \
                xin if self.day_input != 'linear' and bool((xin == xin.round()).all()) else None
            if events is not None:
                # events or raw counts: each adds its (gain-folded) weight column to each of the M synaptic
                # currents, once per delay copy; the filtered currents decay once per step
                op_acc += float((self.memory.delay(events).abs() * mask).sum().item()) * W * M
                op_mac += n_valid * W * self.memory.n_decaying
            else:
                # continuous input: a multiply-add per nonzero input value and neuron
                op_mac += int((self.memory.trace(pre).detach() * mask).count_nonzero().item()) * W
            n_out = int(self.output_lengths(bin_lengths.to(x.device)).sum().item())
            self.input_ops = {'mac': op_mac, 'acc': op_acc, 'frames': max(n_out, 1)}
            for li, (layer, s) in enumerate(zip(self.layers, layer_out)):
                prev = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1)
                if li > 0:
                    op_acc += float((layer_out[li - 1] * mask).sum().item()) * layer.width
                if layer.rec is not None:
                    op_acc += float((prev * mask).sum().item()) * layer.width
                if layer.log_tau_syn is not None:
                    op_mac += n_valid * layer.width
                op_elem += 2 * n_valid * layer.width
                if layer.neuron_type != 'iaf' or layer.log_tau_mem is not None:
                    op_mac += n_valid * layer.width
                if layer.neuron_type == 'alif':
                    op_mac += 2 * n_valid * layer.width
                elif layer.neuron_type == 'seadlif':          # input scaling, a u, b s, adaptation decay
                    op_mac += 4 * n_valid * layer.width
                elif layer.adapt_mask is not None:            # mixed population: only the adaptive neurons
                    op_mac += 2 * n_valid * int(layer.adapt_mask.sum().item())
            if self.feedback is not None:
                top = layer_out[-1]
                op_acc += float((torch.cat([torch.zeros_like(top[:, :1]), top[:, :-1]], 1) * mask).sum().item()) \
                    * self.layer_widths[0]
            read = layer_out if self.readout_all else layer_out[-1:]
            n_dec = sum(a > 0 for a in self.readout_alphas) if self.readout_traces else 0
            n_plain = len(self.readout_alphas) - n_dec if self.readout_traces else 1
            if n_dec:                                 # readout traces: each decays and is read densely, every step
                op_mac += n_valid * sum(s.shape[-1] for s in read) * n_dec * (1 + self.n_classes)
            if self.readout_vmem:                     # continuous: every neuron, every step
                op_mac += n_valid * sum(s.shape[-1] for s in read) * self.n_classes * n_plain
            else:
                op_acc += sum(float((s * mask).sum().item()) for s in read) * self.n_classes * n_plain  # per spike
            op_mac += 2 * n_valid * self.n_classes * self.out_stages                     # output leak, per stage
            self.op_counts = {'mac': op_mac, 'acc': op_acc, 'elementwise': op_elem}
        out = torch.stack(outs, 1)
        if self._ostride > 1:
            out = out[:, self._ostride - 1::self._ostride]           # the potential at each 80 ms frame's end
        return out

    # ------------------------------------------------------------------ streaming
    def reset_stream(self):
        self._stream = None

    def step(self, x: torch.Tensor, day_idx: int) -> torch.Tensor:
        """One 20 ms bin x (N, C) -> logits (N, n_classes) when an output is due (every output_stride bins),
        else None. Equivalent to forward() over the stream."""
        if self._stream is None:
            self._stream = {'states': [layer.init_state(x.shape[0], x.device) for layer in self.layers],
                            'out_v': torch.zeros(self.out_stages, x.shape[0], self.n_classes, device=x.device),
                            'enc': self.encoder.init_state(x.shape[0], self.n_in, x.device),
                            'mem': self.memory.init_state(), 'n': 0}
        st = self._stream
        u = self._input(x, day_idx)
        if self.input_substeps > 1:
            us = self._spread(u.unsqueeze(1)).unbind(1)       # the bin's K chip steps
        else:
            if self.encoder.active:
                u, st['enc'] = self.encoder.step(u, st['enc'])
            us = (u,)
        out = None
        for u in us:
            xin = self.memory.step(u, st['mem'])
            if self.syn_delay_in:
                st['xin'] = ([xin] + st.get('xin', []))[:self.syn_delay_in + 1]          # newest first
                if '_in_wd' not in st:
                    st['_in_wd'] = self._in_wd()
                drive0 = sum(torch.nn.functional.linear(st['xin'][d], st['_in_wd'][d]) for d in range(len(st['xin'])))
            else:
                drive0 = self.layers[0].drive(xin)
            _, st['out_v'] = self._step_all(drive0, st['states'], st['out_v'])
            st['n'] += 1
            if st['n'] % self._ostride == 0:
                out = st['out_v'][-1]
        return out
