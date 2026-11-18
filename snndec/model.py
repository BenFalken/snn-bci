"""
SNNDecoder: host input -> spiking layers -> host readout, for every experiment and both designs.

    kind 'lsnn'   recurrent layers, synaptic filters, threshold adaptation in a fraction of the neurons, soft reset,
                  a spike cap; read by leaky output neurons
    kind 'speck'  what maps onto the Speck2f chip: feedforward integrate-and-fire layers, hard reset, no filters, no
                  adaptation (checked); all memory on the host (input delays, readout smoothing)

Built from a spec (a dict, the configs' `model:` section). Time constants are in milliseconds; `step_ms` is the
length of one input step and `substeps` model steps make one input step.

    spec = {kind, step_ms, n_channels, n_outputs, n_days (0: no day gain), out_every (input steps per output),
            input:   {channels, dropout, encoding, threshold, substeps, binarize, event_dropout, input_scale, delays},
            layers:  {hidden, recurrent, reset, tau_syn_ms, adaptive_fraction, tau_adapt_ms, adapt_scale, thresholds,
                      max_spikes, min_vmem, max_vmem, rec_init_scale, adapt_seed, surrogate, layer_dropout},
            readout: {type: leaky | ema, read: last | all, ...  (readouts.py)}}

forward(x, day) takes (N, T, n_channels) input steps and returns (N, T // out_every, n_outputs); step(x_t, day)
does the same one input step at a time and returns an output when one is due, else None.
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .frontend import HostInput
from .neurons import SpikingLayer
from .readouts import EMAReadout, LeakyReadout

LAYER_DEFAULTS = dict(recurrent=False, reset='hard', tau_syn_ms=None, adaptive_fraction=0.0, tau_adapt_ms=500.0,
                      adapt_scale=1.8, thresholds=1.0, max_spikes=None, min_vmem=-1.0, max_vmem=None,
                      rec_init_scale=0.5, adapt_seed='per_layer', surrogate='lsnn', layer_dropout=0.0)


class SNNDecoder(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        s = self.spec
        if s['kind'] not in ('lsnn', 'speck'):
            raise ValueError(f"kind must be 'lsnn' or 'speck', got {s['kind']!r}")
        inp = dict(s.get('input', {}))
        self.substeps = int(inp.get('substeps', 1))
        self.dt = float(s['step_ms']) / self.substeps                       # ms per model step
        self.out_every = int(s.get('out_every', 1)) * self.substeps         # model steps per output
        self.host = HostInput(s['n_channels'], n_days=s.get('n_days', 0), **inp)

        L = {**LAYER_DEFAULTS, **s.get('layers', {})}
        hidden = list(L['hidden'])
        thr = L['thresholds'] if isinstance(L['thresholds'], (list, tuple)) else [L['thresholds']] * len(hidden)
        dims = [self.host.n_out] + hidden
        tau_syn = [t / self.dt for t in L['tau_syn_ms']] if L['tau_syn_ms'] else None
        self.layers = nn.ModuleList(SpikingLayer(
            dims[i], dims[i + 1], recurrent=L['recurrent'], reset=L['reset'], tau_syn=tau_syn,
            adaptive_fraction=L['adaptive_fraction'], tau_adapt=L['tau_adapt_ms'] / self.dt,
            adapt_scale=L['adapt_scale'], threshold=thr[i], max_spikes=L['max_spikes'], min_vmem=L['min_vmem'],
            max_vmem=L['max_vmem'], rec_init_scale=L['rec_init_scale'],
            adapt_seed=5 + i if L['adapt_seed'] == 'per_layer' else int(L['adapt_seed']),
            surrogate_fn=L['surrogate'], dropout=L['layer_dropout'] if i > 0 else 0.0) for i in range(len(hidden)))
        if s['kind'] == 'speck':
            bad = [i for i, l in enumerate(self.layers) if not l.chip_compatible]
            if bad:
                raise ValueError(f"kind 'speck': layers {bad} are not chip-compatible (need feedforward, no filter, "
                                 "no adaptation)")

        self.branches = None
        if s.get('branches') and int(s['branches'].get('n', 1)) > 1:
            self._make_branches(dict(s['branches']))

        R = dict(s['readout'])
        self.read_all = R.pop('read', 'last') == 'all'
        n_read = sum(hidden) if self.read_all else hidden[-1]
        kind = R.pop('type')
        if kind == 'leaky':
            R['tau'] = R.pop('tau_ms') / self.dt
            if R.get('traces_ms'):
                R['traces'] = [t / self.dt for t in R.pop('traces_ms')]
            R.pop('traces_ms', None)
            self.readout = LeakyReadout(n_read, s['n_outputs'], **R)
        elif kind == 'ema':
            R['decay'] = R['decay'] ** (1.0 / self.substeps)              # given per input step
            self.readout = EMAReadout(n_read, s['n_outputs'], **R)
        else:
            raise ValueError(f"readout type must be 'leaky' or 'ema', got {kind!r}")
        self._stream = None

    def _make_branches(self, B: dict):
        """Several independent networks side by side (kind 'speck'), as one model whose weights between them are
        zero and stay zero (their gradients are masked), so they train jointly into one readout yet each maps onto its
        own chip cores (chip.branch_stacks). branches: {n, split}, split = 'channels_blocks' (the ranked channels
        cut into n contiguous groups), 'channels_interleave' (channel c to network c mod n) or 'delays' (every
        channel in every network, the delay taps split: delay_groups = one list of delays per network, input steps).
        Each layer's neurons are split into n equal contiguous groups. Initial weights are rescaled by the square
        root of the fan-in lost, so each network starts at the scale the full layer would."""
        n, split = int(B['n']), B.get('split', 'channels_blocks')
        h, K = self.host, self.host.substeps
        n_in, delays = h.n_in, [d // K for d in h.delays]
        line = np.empty(n_in * len(delays), dtype=int)
        for di, d in enumerate(delays):
            for c in range(n_in):
                if split == 'channels_blocks':
                    b = c * n // n_in
                elif split == 'channels_interleave':
                    b = c % n
                elif split == 'delays':
                    b = next(k for k, g in enumerate(B['delay_groups']) if d in [int(x) for x in g])
                else:
                    raise ValueError(f'unknown branch split {split!r}')
                line[di * n_in + c] = b
        neuron = [np.arange(l.width) * n // l.width for l in self.layers]
        masks = [torch.as_tensor(neuron[0][:, None] == line[None, :], dtype=torch.float32)]
        masks += [torch.as_tensor(neuron[i][:, None] == neuron[i - 1][None, :], dtype=torch.float32)
                  for i in range(1, len(self.layers))]
        for l, m in zip(self.layers, masks):
            with torch.no_grad():
                fan = m.sum(1, keepdim=True).clamp(min=1)
                l.ff.weight.mul_(m * torch.sqrt(m.shape[1] / fan))
            l.ff.weight.register_hook(lambda g, m=m: g * m)
            l.register_buffer('branch_mask', m, persistent=False)
        self.branches = {'n': n, 'line': line, 'neuron': neuron}

    @property
    def widths(self):
        return [l.width for l in self.layers]

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Outputs for sequences of `lengths` input steps."""
        return lengths * self.substeps // self.out_every

    # ------------------------------------------------------------------------------------------------- forward
    def _run(self, drive0, states, rst, tbptt=0):
        """Steps through drive0 (N, T, width0); returns outputs (N, T, n_outputs) and spikes per layer."""
        T = drive0.shape[1]
        outs, spikes = [], [[] for _ in self.layers]
        for t in range(T):
            if tbptt and self.training and t and t % tbptt == 0:            # truncated backpropagation
                for st in states:
                    for k in st:
                        st[k] = st[k].detach()
                for k in ('y', 'tr', 'e'):
                    if rst.get(k) is not None:
                        rst[k] = [v if v is None else v.detach() for v in rst[k]]
            cur, read = drive0[:, t], []
            for li, (layer, st) in enumerate(zip(self.layers, states)):
                if li:
                    cur = layer.drive(h)
                h = layer.step(cur, st)
                spikes[li].append(h)
                read.append(h)
            outs.append(self.readout.step(torch.cat(read, -1) if self.read_all else h, rst))
        return torch.stack(outs, 1), [torch.stack(s, 1) for s in spikes]

    def forward(self, x: torch.Tensor, day: Optional[int] = None, tbptt: int = 0) -> torch.Tensor:
        """x (N, T, n_channels) -> (N, T * substeps // out_every, n_outputs). Starts from rest. After the call,
        self.layer_spikes holds each layer's spikes (N, T', width) for spike-rate penalties and op counts."""
        N = x.shape[0]
        u = self.host(x, day)
        self.input_events = u                                              # what the first layer reads
        states = [l.init_state(N, x.device) for l in self.layers]
        y, self.layer_spikes = self._run(self.layers[0].drive(u), states,
                                         self.readout.init_state(N, x.device), tbptt * self.substeps)
        return y[:, self.out_every - 1::self.out_every]

    # ----------------------------------------------------------------------------------------------- streaming
    def reset_stream(self):
        self._stream = None

    def step(self, x: torch.Tensor, day: Optional[int] = None) -> Optional[torch.Tensor]:
        """One input step (N, n_channels) -> (N, n_outputs) when an output is due, else None. Equivalent to
        forward() over the stream."""
        N = x.shape[0]
        if self._stream is None:
            self._stream = {'host': self.host.init_state(), 'n': 0,
                            'layers': [l.init_state(N, x.device) for l in self.layers],
                            'readout': self.readout.init_state(N, x.device)}
        st, out = self._stream, None
        for u in self.host.step(x, st['host'], day):
            y, _ = self._run(self.layers[0].drive(u)[:, None], st['layers'], st['readout'])
            st['n'] += 1
            if st['n'] % self.out_every == 0:
                out = y[:, 0]
        return out

    # ------------------------------------------------------------------------------------------------ the chip
    def chip_network(self) -> nn.Sequential:
        """The spiking layers as sinabs layers, as they would run on Speck (kind 'speck' only)."""
        import sinabs.activation as sa
        import sinabs.layers as sl
        if self.spec['kind'] != 'speck':
            raise ValueError("only a 'speck' network maps onto the chip")
        mods = []
        for l in self.layers:
            conv = nn.Conv2d(l.ff.in_features, l.width, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(l.ff.weight[:, :, None, None])
            mods += [conv, sl.IAFSqueeze(batch_size=1, spike_threshold=l.threshold, min_v_mem=l.min_vmem,
                                         spike_fn=sa.MultiSpike,
                                         reset_fn=sa.MembraneSubtract() if l.reset == 'soft' else sa.MembraneReset())]
        return nn.Sequential(*mods)

    def check_chip(self, device: str = 'speck2fdevkit') -> bool:
        """True if the spiking layers map onto the chip (sinabs' DynapcnnNetwork mapper); raises with the mapper's
        reason otherwise."""
        from sinabs.backend.dynapcnn import DynapcnnNetwork
        DynapcnnNetwork(self.chip_network(), input_shape=(self.host.n_out, 1, 1),
                        discretize=True).make_config(device=device)
        return True
