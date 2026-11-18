import os
os.environ["PYTHONHASHSEED"] = "0"
import sys
import math
import random

import numpy as np
import torch
import torch.nn as nn
import sinabs.layers as sl
import sinabs.activation as sa

from base_dockex.BaseModel import BaseModel


class SpeckSNN(nn.Module):
    """A network that maps onto the Speck2f chip, as the motor decoder's speck_delay: feedforward integrate-and-fire
    layers, hard reset, no synaptic filters, no recurrence, spiking outputs (one per target band). Everything with
    memory or continuous values runs on the host: the input delay bank in front of the chip, and behind it a two-stage
    exponential moving average of each output's spikes and one gain and offset per band.

    forward(x): x (N, T, C) spike counts per step -> (N, T, n_outputs) the host readout at every step, and the mean
    spikes per neuron per step. Every call starts from rest."""

    def __init__(self, n_inputs, n_outputs, hidden_dims, delays, ema_decay=0.3, readout="band", thresholds=None,
                 input_scale=None, ema_stages=2, ema_per_neuron=False):
        super().__init__()
        self.delays = [int(d) for d in delays]
        # input_scale (one per channel): the counts are multiplied by it before the first layer; on the chip this is a
        # fixed factor folded into that layer's weights (input events stay events)
        self.register_buffer("input_scale", torch.as_tensor(input_scale, dtype=torch.float32) if input_scale is not None
                             else torch.ones(n_inputs))
        self.n_bands = n_outputs
        # readout 'pushpull': two output neurons per band, one for above its mean and one for below (spiking outputs
        # can only signal increases); the host takes their difference
        n_out_neurons = 2 * n_outputs if readout == "pushpull" else n_outputs
        # readout 'hidden': no spiking output layer; the host maps the last hidden layer's smoothed spikes linearly
        # to the bands (the chip's output layer was mostly silent, its neurons too far below threshold to learn)
        dims = [n_inputs * len(self.delays)] + list(hidden_dims) + ([] if readout == "hidden" else [n_out_neurons])
        self.linears = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        for lin in self.linears:
            nn.init.kaiming_uniform_(lin.weight)
        thresholds = list(thresholds) if thresholds else [1.0] * len(self.linears)
        if len(thresholds) != len(self.linears):
            raise ValueError(f"thresholds: one per layer ({len(self.linears)}), got {len(thresholds)}")
        self.neurons = nn.ModuleList(sl.IAF(spike_threshold=float(t), spike_fn=sa.MultiSpike,
                                            reset_fn=sa.MembraneReset(), min_v_mem=-1.0) for t in thresholds)
        # host smoothing of the read layer's spikes (a leaky integrator, the membrane of a non-spiking readout neuron):
        # ema_stages in series; ema_per_neuron gives every read neuron its own trained time constant
        self.ema_stages = int(ema_stages)
        width = dims[-1]
        init = torch.logit(torch.tensor(float(ema_decay)))
        self.decay_raw = nn.Parameter(init.repeat(width) if ema_per_neuron else init)
        # readout 'band': one output neuron per band, a gain and offset each; 'linear': every band a weighted sum of
        # all the output neurons' smoothed spikes (n_outputs^2 multiply-adds per step on the host)
        self.readout = readout
        if readout == "linear":
            self.host_linear = nn.Linear(n_outputs, n_outputs)
        elif readout == "hidden":
            self.host_linear = nn.Linear(dims[-1], n_outputs)
        elif readout in ("band", "pushpull"):
            self.gain = nn.Parameter(torch.ones(n_outputs))
            self.offset = nn.Parameter(torch.zeros(n_outputs))
        else:
            raise ValueError(f"readout must be 'band', 'pushpull', 'linear' or 'hidden', got {readout!r}")
        self.widths = dims[1:]

    def delay_bank(self, x):
        """(N, T, C) -> (N, T, C * len(delays)): each channel now and delays[k] steps earlier (zeros before the
        start), side by side. Runs on the host."""
        T = x.shape[1]
        copies = [x if d == 0 else torch.cat([x.new_zeros(x.shape[0], min(d, T), x.shape[2]), x[:, :max(T - d, 0)]], 1)
                  for d in self.delays]
        return torch.cat(copies, -1)

    def forward(self, x):
        for n in self.neurons:
            n.reset_states()
        h = self.delay_bank(x * self.input_scale)
        spikes = x.new_zeros(())
        self.layer_stats = []                                 # per layer: (mean spikes/neuron/step, share of silent neurons)
        for lin, neu in zip(self.linears, self.neurons):
            h = neu(lin(h))                                   # (N, T, width) spike counts
            spikes = spikes + h.sum()
            with torch.no_grad():
                self.layer_stats.append((float(h.mean()), float((h.sum((0, 1)) == 0).float().mean())))
        decay = torch.sigmoid(self.decay_raw)
        stages = [h.new_zeros(h.shape[0], h.shape[2]) for _ in range(self.ema_stages)]
        out = []
        for t in range(h.shape[1]):                           # host: EMA stages, then the readout
            inp = h[:, t]
            for k in range(self.ema_stages):
                stages[k] = decay * stages[k] + inp
                inp = stages[k]
            s = inp * (1 - decay) ** self.ema_stages
            if self.readout == "pushpull":
                s = s[:, :self.n_bands] - s[:, self.n_bands:]
            out.append(self.host_linear(s) if self.readout in ("linear", "hidden") else self.gain * s + self.offset)
        rate = spikes / (x.shape[0] * x.shape[1] * sum(self.widths))
        return torch.stack(out, 1), rate


class SpeckSNNDecoder(BaseModel):
    """Takes the same window arrays as the Keras RNNs (window i = frames i..i+W-1, target = frame i+W//2) and the
    dataset's raw spike counts (raw_features_npy: features.npy, one row per frame, the windows' source). The chip reads
    the counts as events, one 40 ms frame per step, without resets; window i's prediction is the readout at its last
    frame, so it sees the same frames (the delays sample the window) and the same look-ahead as the other decoders.
    Trained on random stretches of the training frames, the first burn-in steps of each not scored."""

    def __init__(self, input_args):
        super().__init__(input_args)
        p = lambda k, d: self.params[k] if self.params[k] is not None else d
        self.hidden = p('hidden_dims', [128, 256, 256])
        self.delays = p('delays', [0, 4, 8, 12, 16])
        # n_delays: that many delays spread evenly over the window instead (0 .. window length - 1)
        self.n_delays = p('n_delays', None)
        # n_channels: keep only the channels that best predict the targets on the TRAINING frames (each channel's
        # window -> every band by ridge regression, ranked by variance explained); the chip's 480 inputs are then
        # channels x delays, so fewer channels buy more delays
        self.n_channels = p('n_channels', None)
        self.channels = None
        self.epochs = p('epochs', 2048)
        self.lr = p('learning_rate', 1e-3)
        self.patience = p('patience', 10)
        self.chunk_steps = p('chunk_steps', 150)
        self.burn_in = p('burn_in_steps', 25)
        self.batch = p('batch_size', None) or 16
        self.coverage = p('coverage', 4)
        self.spike_rate_weight = p('spike_rate_weight', 0.0)
        self.spike_rate_target = p('spike_rate_target', 0.1)
        self.input_dropout = p('input_dropout', 0.0)
        self.ema_decay = p('ema_decay', 0.3)                 # per 40 ms step: 0.3 ~ a 33 ms time constant per stage
        self.readout = p('readout', 'band')
        # substeps K: each 40 ms frame is K chip steps of 40/K ms, its counts spread as evenly as possible over them
        # (as the motor decoder's 4 ms bins); delays keep their length in ms (K x the steps), the EMA its time constant
        # (decay ** (1/K) per step), and a frame's prediction is the readout at its last step, so the data and the
        # look-ahead are unchanged
        self.substeps = int(p('substeps', 1))
        self.eval_chunked = bool(p('eval_chunked', False))  # inference (and validation) in reset stretches
        self.input_norm = bool(p('input_norm', False))      # scale each channel by 1 / std of its TRAINING counts
        # binarize: at most one event per channel per chip step (after spreading over the substeps), as the motor
        # decoder's binarized input; bounds the drive, so bursts of activity saturate instead of multiplying it
        self.binarize = bool(p('binarize', False))
        self.ema_stages = int(p('ema_stages', 2))
        self.ema_per_neuron = bool(p('ema_per_neuron', False))
        self.thresholds = p('thresholds', None)              # spike threshold per chip layer (default 1)
        self.delay_max = p('delay_max', None)                # n_delays spread over 0 .. delay_max frames (default:
                                                             # the window, 0 .. W - 1); beyond W - 1 reaches further back
        self.device = torch.device(p('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    def set_random_seeds(self):
        super().set_random_seeds()
        torch.manual_seed(self.random_seed)

    def load_data(self):
        super().load_data()
        self.raw = np.load(self.input_pathnames['raw_features_npy']).astype(np.float32)
        self.lag = self.X_train.shape[1] - 1                 # window i is read at frame i + lag
        n = sum(len(a) for a in (self.X_train, self.X_valid, self.X_test) if a is not None)
        if len(self.raw) != n + self.lag:
            raise ValueError(f"raw_features_npy has {len(self.raw)} frames; the windows need {n + self.lag}")
        if self.raw.min() < 0:
            raise ValueError("raw_features_npy must hold spike counts (>= 0): the chip takes events")
        W = self.X_train.shape[1]
        if self.n_delays:
            top = self.delay_max if self.delay_max is not None else W - 1
            self.delays = sorted({int(round(d)) for d in np.linspace(0, top, self.n_delays)})
        if self.n_channels and self.n_channels < self.raw.shape[1]:
            self.channels = self.rank_channels()[:self.n_channels]
            self.raw = self.raw[:, self.channels]

    def rank_channels(self, ridge=1.0):
        """Channels ordered by how much target variance each one's window explains on the training frames alone."""
        n_tr, W = len(self.X_train), self.X_train.shape[1]
        y = self.y_train - self.y_train.mean(0)
        score = []
        for c in range(self.raw.shape[1]):
            x = np.lib.stride_tricks.sliding_window_view(self.raw[:n_tr + W - 1, c], W)    # (n_tr, W)
            x = x - x.mean(0)
            w = np.linalg.solve(x.T @ x + ridge * np.eye(W), x.T @ y)
            score.append(1 - ((y - x @ w) ** 2).sum() / (y ** 2).sum())
        order = list(np.argsort(score)[::-1])
        print(f"channels ranked on the training frames: best {order[:8]} (variance explained "
              f"{score[order[0]]:.3f} .. {score[order[min(len(order), self.n_channels or 1) - 1]]:.3f})")
        return order

    def spread(self, x):
        """(..., L, C) counts per frame -> (..., L * K, C) counts per chip step, each frame's spread evenly."""
        K = self.substeps
        if K == 1:
            return np.minimum(x, 1.0) if self.binarize else x
        j = np.arange(K, dtype=np.float32)[:, None]
        c = x[..., :, None, :]
        y = np.floor(c * (j + 1) / K) - np.floor(c * j / K)                 # (..., L, K, C), sums to c
        if self.binarize:
            y = np.minimum(y, 1.0)
        return y.reshape(*x.shape[:-2], x.shape[-2] * K, x.shape[-1])

    def instantiate_model(self):
        K = self.substeps
        scale = None
        if self.input_norm:                                   # of what the chip receives (binarized, if so)
            sd = self.spread(self.raw[:len(self.X_train) + self.lag]).std(0)
            scale = 1.0 / np.maximum(sd, 1e-3)
        self.model = SpeckSNN(self.raw.shape[1], self.y_train.shape[1], self.hidden, [d * K for d in self.delays],
                              self.ema_decay ** (1 / K), self.readout, self.thresholds, scale,
                              self.ema_stages, self.ema_per_neuron).to(self.device)
        print(f"Speck SNN: {self.raw.shape[1]} channels x {len(self.delays)} delays {self.delays} frames "
              f"({self.substeps} chip step(s) per 40 ms frame) = "
              f"{self.raw.shape[1] * len(self.delays)} chip inputs -> "
              f"{' -> '.join(map(str, self.model.widths))}; {sum(q.numel() for q in self.model.parameters()):,} parameters")

    def _t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def _run(self, x):
        """A (T, C) recording of frames, no gradients: (T, n_outputs) readout at each frame's last chip step. One
        continuous pass, or with eval_chunked: stretches as in training (chunk_steps frames, from rest), overlapping by
        burn_in frames whose outputs are dropped, so inference matches training."""
        self.model.eval()
        K = self.substeps
        with torch.no_grad():
            if not self.eval_chunked:
                return self.model(self._t(self.spread(x)[None]))[0][0, K - 1::K].cpu().numpy()
            L, B = self.chunk_steps, self.burn_in
            out = np.zeros((len(x), self.model.n_bands), np.float32)
            start = 0
            while start < len(x):
                a = max(start - B, 0)
                seg = self.model(self._t(self.spread(x[a:start + L - B])[None]))[0][0, K - 1::K].cpu().numpy()
                out[start:start + len(seg) - (start - a)] = seg[start - a:]
                start += L - B
            return out

    def fit(self):
        print(f'Fitting model ({self.chunk_steps}-step stretches, first {self.burn_in} not scored)')
        n_tr = len(self.X_train)
        xs = self.raw[:n_tr + self.lag]
        ys = np.concatenate([np.zeros((self.lag, self.y_train.shape[1]), np.float32), self.y_train], 0)
        L = min(self.chunk_steps, len(xs))
        has_valid = self.X_valid is not None
        x_tv = self.raw[:n_tr + len(self.X_valid) + self.lag] if has_valid else None
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, waited = math.inf, None, 0
        n_chunks = max(1, int(round(self.coverage * len(xs) / L)))
        for epoch in range(self.epochs):
            self.model.train()
            tot, rate_sum, nb, skipped, gmax = 0.0, 0.0, 0, 0, 0.0
            starts = np.random.randint(0, len(xs) - L + 1, n_chunks)
            for i in range(0, n_chunks, self.batch):
                s = starts[i:i + self.batch]
                steps = s[:, None] + np.arange(L)[None]
                x = self._t(self.spread(xs[steps]))
                if self.input_dropout > 0:
                    x = x * (torch.rand_like(x) >= self.input_dropout)        # drop events
                mask = torch.as_tensor((steps >= self.lag) & (np.arange(L)[None] >= self.burn_in), device=self.device)
                if not bool(mask.any()):
                    continue
                pred, rate = self.model(x)
                pred = pred[:, self.substeps - 1::self.substeps]               # each frame's last chip step
                loss = ((pred - self._t(ys[steps])) ** 2)[mask].mean()
                total = loss + self.spike_rate_weight * (rate - self.spike_rate_target).clamp(min=0) ** 2
                opt.zero_grad()
                total.backward()
                if not all(torch.isfinite(q.grad).all() for q in self.model.parameters() if q.grad is not None):
                    skipped += 1
                    continue
                gn = float(nn.utils.clip_grad_norm_(self.model.parameters(), 10.0))
                opt.step()
                tot, rate_sum, nb = tot + float(loss), rate_sum + float(rate), nb + 1
                gmax = max(gmax, gn)
            if has_valid:
                out = self._run(x_tv)[n_tr + self.lag:]
                val = float(((out - self.y_valid) ** 2).mean())
            else:
                val = tot / max(nb, 1)
            print(f'Epoch {epoch + 1}/{self.epochs} - loss: {tot / max(nb, 1):.4f} - val_loss: {val:.4f} - '
                  f'spikes/neuron/step: {rate_sum / max(nb, 1):.3f}' + (f' - skipped {skipped}' if skipped else ''),
                  flush=True)
            if has_valid:                                     # the validation pass, layer by layer
                print('    layers (spikes/neuron/step, silent share): ' + '  '.join(
                    f'{r:.3f}/{z:.0%}' for r, z in self.model.layer_stats) + f' | max grad norm {gmax:.1f}', flush=True)
            if val < best - 1e-4:
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    print(f'Early stopping; restoring the weights of the best epoch (val_loss {best:.4f})')
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self):
        print('Generating predictions')
        out = self._run(self.raw)[self.lag:].astype(np.float64)       # window i -> readout at frame i + lag
        i = 0
        for name, a in (('train', self.X_train), ('valid', self.X_valid), ('test', self.X_test)):
            if a is not None:
                setattr(self, f'predict_{name}', out[i:i + len(a)])
                i += len(a)

    def save(self):
        if self.method == 'predict':
            return
        torch.save({'state_dict': self.model.state_dict(), 'hidden': self.hidden, 'delays': self.delays,
                    'ema_decay': self.ema_decay, 'readout': self.readout, 'substeps': self.substeps,
                    'thresholds': self.thresholds, 'binarize': self.binarize, 'ema_stages': self.ema_stages,
                    'ema_per_neuron': self.ema_per_neuron,
                    'n_inputs': self.raw.shape[1], 'n_outputs': self.y_train.shape[1],
                    'channels': None if self.channels is None else [int(c) for c in self.channels]},
                   self.output_pathnames['model_pt'])

    def load(self):
        ck = torch.load(self.input_pathnames['model_pt'], map_location=self.device, weights_only=False)
        self.substeps = K = ck.get('substeps', 1)
        self.delays = ck['delays']
        self.model = SpeckSNN(ck['n_inputs'], ck['n_outputs'], ck['hidden'], [d * K for d in ck['delays']],
                              ck.get('ema_decay', 0.3) ** (1 / K), ck.get('readout', 'band'),
                              ck.get('thresholds'), ck['state_dict'].get('input_scale'), ck.get('ema_stages', 2),
                              ck.get('ema_per_neuron', False)).to(self.device)
        self.binarize = ck.get('binarize', False)
        # weights only: sinabs keeps each layer's membrane state (v_mem) with the weights, shaped by the last call
        self.model.load_state_dict({k: v for k, v in ck['state_dict'].items() if not k.endswith('v_mem')}, strict=False)
        if ck.get('channels') is not None:                    # the channels it was trained on, not a new ranking
            self.channels = ck['channels']
            self.raw = np.load(self.input_pathnames['raw_features_npy']).astype(np.float32)[:, self.channels]


if __name__ == '__main__':
    SpeckSNNDecoder(sys.argv).run()
