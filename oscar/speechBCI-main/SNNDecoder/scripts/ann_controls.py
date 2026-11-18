"""
Non-spiking controls that isolate why the feedforward SNN does not learn
while a GRU does. Each trains through the SNN's own pipeline (same .npz
data, per-session input layers, augmentation, CTC loss, greedy PER, one
output per 20 ms bin) on the SNN sweep's sessions:

    gru_512       2 x 512 GRU, causal: the reference that learns (~2.7M params)
    gru_1024      2 x 1024 GRU, causal (on the raw-count data: the native SNN's reference)
    gru_small     2 x 148 GRU (~0.31M params, the SNN core's size): is it capacity?
    ff_twin       the SNN's architecture without spikes: the same widths
                  (512-256-128-128), each layer Linear -> unit-gain leaky
                  filter with trained tau (init 2/4/8/16 bins) -> ReLU, then
                  the 2-stage EMA (0.8) and readout. Same memory, continuous
                  activations and exact gradients: is it the spiking?
    window32      causal temporal convolution over the last 32 bins (640 ms,
                  the RNN's input window, per-lag weights), then pointwise
                  512-256-128 ReLU layers, no recurrence: can a feedforward
                  network learn with explicit lags? (~4.4M params)
    window32_small  the same with 128 conv channels (~1.1M params)
    gru_stack_2x512  the paper RNN's front end (causal 32-bin stacking, stride 4: 80 ms
                  frames, its output frames) with 2 x 512 GRU layers: the ANN twin of the
                  recurrent SNN R2 (rsnn_ladder / rsnn_round2)
    gru_stack_5x1024 the paper RNN's full shape (5 x 1024 GRU, dropout 0.4 between layers)
                  trained by this pipeline; run it with --smooth-sd 2 as well to match the
                  RNN's non-causal input smoothing. Against the cached RNN's own test PER,
                  it shows how much of the SNN's gap comes from this training pipeline
                  rather than from spiking.

Reading the results (best test PER after the same epochs):
    ff_twin learns, SNN does not  -> spikes / surrogate gradients
    ff_twin fails, window32 learns -> exponential memory; feedforward works
                                      with explicit lags (input windows, delays)
    both fail, GRUs learn          -> this task needs recurrence
    gru_small fails, gru_512 learns -> capacity matters too

    python ann_controls.py --model ff_twin --data-dir $BASE_DIR/derived/snnData --out-dir $BASE_DIR/derived/snns/controls
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snnDecoder.dataset import SessionBatches, augment, load_partitions, smooth_inputs
from snnDecoder.train import ctc_loss
from snnDecoder.utils import edit_distance, greedy_decode

BLANK, N_CLASSES = 40, 41
N_IN = 256   # input channels; set from the data in main() (128 for the raw-count export)
SWEEP_SESSIONS = ['t12.2022.05.05', 't12.2022.05.24', 't12.2022.06.07', 't12.2022.06.16',
                  't12.2022.06.28', 't12.2022.07.14', 't12.2022.07.27', 't12.2022.08.11']


class DayInput(nn.Module):
    """The SNN's per-session input layer: identity-initialized Linear, softsign, dropout."""

    def __init__(self, n_sessions, dropout):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(N_IN, N_IN) for _ in range(n_sessions)])
        for layer in self.layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, day_idx):
        return self.dropout(F.softsign(self.layers[day_idx](x)))


def causal_filter(x, decay, length=128):
    """Unit-gain first-order low-pass y[t] = d*y[t-1] + (1-d)*x[t] along time
    of x (N, T, C), one decay per call (scalar tensor), as a causal
    convolution truncated at `length` bins (exact to ~1e-3 for tau <= 20)."""
    k = torch.arange(length, device=x.device, dtype=x.dtype)
    kernel = ((1 - decay) * decay ** k).flip(0)                    # oldest tap first
    C = x.shape[-1]
    w = kernel.view(1, 1, -1).repeat(C, 1, 1)
    out = F.conv1d(F.pad(x.transpose(1, 2), (length - 1, 0)), w, groups=C)
    return out.transpose(1, 2)


class GRUNet(nn.Module):
    """Per-session input layer, then a GRU stepping every 20 ms bin. out_stride > 1: one output per out_stride
    steps (the last step's), as the native SNN's --output-stride."""

    def __init__(self, n_sessions, hidden, layers=2, dropout=0.2, out_stride=1):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        self.gru = nn.GRU(N_IN, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, N_CLASSES)
        self.out_stride = int(out_stride)

    def output_lengths(self, lengths):
        return lengths // self.out_stride if self.out_stride > 1 else lengths

    def forward(self, x, day_idx):
        y = self.out(self.gru(self.inp(x, day_idx))[0])
        return y[:, self.out_stride - 1::self.out_stride] if self.out_stride > 1 else y


class StackGRU(nn.Module):
    """The paper RNN's shape: per-session input layer, causal 32-bin stacking with
    stride 4 (output frame k sees bins [4k, 4k + 32)), stacked GRU layers."""

    def __init__(self, n_sessions, hidden, layers, window=32, stride=4, dropout=0.2, layer_dropout=0.4):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        self.window, self.stride = window, stride
        self.gru = nn.GRU(N_IN * window, hidden, layers, batch_first=True,
                          dropout=layer_dropout if layers > 1 else 0.0)
        # Keras GRU initialization, as the paper RNN: Glorot-uniform input weights, orthogonal
        # recurrent weights (per gate), zero biases
        for name, w in self.gru.named_parameters():
            if name.startswith('weight_ih'):
                for gate in w.data.chunk(3, 0):
                    nn.init.xavier_uniform_(gate)
            elif name.startswith('weight_hh'):
                for gate in w.data.chunk(3, 0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(w)
        self.in_dropout = nn.Dropout(layer_dropout)
        self.out = nn.Linear(hidden, N_CLASSES)

    def output_lengths(self, lengths):
        return torch.clamp((lengths - self.window) // self.stride + 1, min=0)

    def forward(self, x, day_idx):
        x = self.inp(x, day_idx)
        frames = x.unfold(1, self.window, self.stride)                     # (N, T_out, C, window)
        x = frames.transpose(2, 3).reshape(x.shape[0], frames.shape[1], -1)
        return self.out(self.gru(self.in_dropout(x))[0])


class FFTwin(nn.Module):
    """The SNN (configuration C) with ReLU units in place of IAF neurons."""

    def __init__(self, n_sessions, widths=(512, 256, 128, 128), taus=(2, 4, 8, 16),
                 ema_stages=2, ema_decay=0.8, dropout=0.2):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        dims = [N_IN] + list(widths)
        self.linears = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:])])
        for lin in self.linears:
            nn.init.kaiming_uniform_(lin.weight)
        # trained time constants, kept in [0.2, 20] bins
        self.tau_raw = nn.Parameter(torch.tensor([math.log(t) for t in taus], dtype=torch.float32))
        self.ema_raw = nn.Parameter(torch.logit(torch.tensor(float(ema_decay))))
        self.ema_stages = ema_stages
        self.out = nn.Linear(widths[-1], N_CLASSES)

    def taus(self):
        return self.tau_raw.exp().clamp(0.2, 20.0)

    def forward(self, x, day_idx):
        x = self.inp(x, day_idx)
        for lin, tau in zip(self.linears, self.taus()):
            x = F.relu(causal_filter(lin(x), torch.exp(-1.0 / tau)))
        decay = torch.sigmoid(self.ema_raw)
        for _ in range(self.ema_stages):
            x = causal_filter(x, decay)
        return self.out(x)


class Window(nn.Module):
    """Causal temporal convolution over the last `window` bins, then pointwise layers."""

    def __init__(self, n_sessions, channels=512, window=32, widths=(256, 128), dropout=0.2):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        self.window = window
        self.conv = nn.Conv1d(N_IN, channels, window)
        dims = [channels] + list(widths)
        self.pointwise = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(dims[-1], N_CLASSES)

    def forward(self, x, day_idx):
        x = self.inp(x, day_idx).transpose(1, 2)
        x = F.relu(self.conv(F.pad(x, (self.window - 1, 0)))).transpose(1, 2)
        for lin in self.pointwise:
            x = F.relu(lin(x))
        return self.out(x)


# ---------------------------------------------------------------------------------------------- benchmark decoders
# The classic decoders of motor-BCI benchmarks (Wiener filter, Kalman filter, LSTM, QRNN), adapted to this task: the
# same 20 ms input and one 41-class output every 80 ms, scored with the same CTC decoding and language model. Each has
# frame_kernel / frame_stride (input bins an output sees / bins between outputs), output_lengths, reset_stream and
# stream_step (one bin at a time, for timing) and frame_counts (operations per output, for scripts/efficiency.py).

def _zero_counts():
    return {k: 0.0 for k in ('mac_in', 'acc_in', 'mac_syn', 'acc_syn', 'mac_neuron', 'mac_readout', 'acc_readout',
                             'elementwise', 'state_words')}


class LSTMNet(nn.Module):
    """Per-session input layer, then an LSTM stepping every 20 ms bin, one output every `every` bins (the last
    step's): the GRU control with LSTM cells."""

    def __init__(self, n_sessions, hidden, layers=2, dropout=0.2, every=4):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        self.rnn = nn.LSTM(N_IN, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, N_CLASSES)
        self.frame_kernel = self.frame_stride = int(every)
        self.hidden, self.depth, self.input_buffer = hidden, layers, False

    def output_lengths(self, lengths):
        return lengths // self.frame_stride

    def forward(self, x, day_idx):
        y = self.out(self.rnn(self.inp(x, day_idx))[0])
        return y[:, self.frame_stride - 1::self.frame_stride]

    def reset_stream(self):
        self._h, self._n = None, 0

    def stream_step(self, x, day_idx):
        y, self._h = self.rnn(self.inp(x, day_idx)[:, None], self._h)
        self._n += 1
        return self.out(y[:, -1]) if self._n % self.frame_stride == 0 else None

    def frame_counts(self):
        H, L, C, k = self.hidden, self.depth, N_IN, self.frame_stride
        c = _zero_counts()
        c['mac_in'] = k * (C * C + C * 4 * H)                              # day layer, first layer's input weights
        c['mac_syn'] = k * (L * H * 4 * H + (L - 1) * H * 4 * H)           # recurrent, and inputs of layers 2..L
        c['mac_neuron'] = k * L * 3 * H                                     # f*c, i*g, o*tanh(c)
        c['elementwise'] = k * L * 5 * H                                    # 3 sigmoids, 2 tanh per unit
        c['mac_readout'] = H * N_CLASSES
        c['state_words'] = k * L * 2 * H
        return c


class QRNNLayer(nn.Module):
    """Quasi-recurrent layer (Bradbury et al. 2017): a causal convolution over the last `window` steps gives the
    candidate z and forget/output gates f, o for every step at once; the only recurrence is elementwise
    ("fo-pooling"): c_t = f_t * c_{t-1} + (1 - f_t) * z_t, h_t = o_t * c_t."""

    def __init__(self, n_in, hidden, window=2, chunk=64):
        super().__init__()
        self.conv = nn.Linear(n_in * window, 3 * hidden)
        self.n_in, self.hidden, self.window, self.chunk = n_in, hidden, window, chunk

    def forward(self, x, state=None):
        N, T, C = x.shape
        prev, c = state if state is not None else (x.new_zeros(N, self.window - 1, C), x.new_zeros(N, self.hidden))
        xp = torch.cat([prev, x], 1)
        frames = xp.unfold(1, self.window, 1).transpose(2, 3).reshape(N, T, -1)   # oldest step first
        z, fa, o = self.conv(frames).chunk(3, -1)
        z, o = torch.tanh(z), torch.sigmoid(o)
        if T <= 2:                                       # streaming (one step at a time): the plain recurrence
            f = torch.sigmoid(fa)
            hs = []
            for t in range(T):
                c = f[:, t] * c + (1 - f[:, t]) * z[:, t]
                hs.append(o[:, t] * c)
            return torch.stack(hs, 1), (xp[:, xp.shape[1] - (self.window - 1):], c)
        # training: the same recurrence without a step-by-step loop. Unrolled from c0 over a chunk of steps,
        #   c_t = exp(F_t) c0 + sum_{s<=t} exp(F_t - F_s) b_s,   F_t = sum_{r<=t} log f_r,   b_s = (1 - f_s) z_s,
        # computed in log space with cumulative log-sum-exps, b split into its positive and negative parts (each
        # part of c stays within [0, 1], so the difference keeps full precision). Chunks of `chunk` steps keep
        # |F| small; the Python loop runs T/chunk times instead of T (same model, same values to float rounding)
        logf = F.logsigmoid(fa)
        b = torch.sigmoid(-fa) * z
        lpos, lneg = torch.relu(b).clamp_min(1e-30).log(), torch.relu(-b).clamp_min(1e-30).log()
        hs = []
        for t0 in range(0, T, self.chunk):
            sl = slice(t0, t0 + self.chunk)
            Fc = logf[:, sl].cumsum(1)
            ck = torch.exp(Fc + torch.logcumsumexp(lpos[:, sl] - Fc, 1)) \
                - torch.exp(Fc + torch.logcumsumexp(lneg[:, sl] - Fc, 1)) + torch.exp(Fc) * c[:, None]
            hs.append(ck)
            c = ck[:, -1]
        return torch.cat(hs, 1) * o, (xp[:, xp.shape[1] - (self.window - 1):], c)


class QRNNNet(nn.Module):
    """Per-session input layer, QRNN layers stepping every 20 ms bin, one output every `every` bins."""

    def __init__(self, n_sessions, hidden, layers=2, window=2, dropout=0.2, every=4):
        super().__init__()
        self.inp = DayInput(n_sessions, dropout)
        self.layers_q = nn.ModuleList([QRNNLayer(N_IN if i == 0 else hidden, hidden, window) for i in range(layers)])
        self.out = nn.Linear(hidden, N_CLASSES)
        self.frame_kernel = self.frame_stride = int(every)
        self.hidden, self.depth, self.qwindow, self.input_buffer = hidden, layers, window, False

    def output_lengths(self, lengths):
        return lengths // self.frame_stride

    def forward(self, x, day_idx):
        h = self.inp(x, day_idx)
        for layer in self.layers_q:
            h, _ = layer(h)
        return self.out(h)[:, self.frame_stride - 1::self.frame_stride]

    def reset_stream(self):
        self._states, self._n = [None] * len(self.layers_q), 0

    def stream_step(self, x, day_idx):
        h = self.inp(x, day_idx)[:, None]
        for i, layer in enumerate(self.layers_q):
            h, self._states[i] = layer(h, self._states[i])
        self._n += 1
        return self.out(h[:, -1]) if self._n % self.frame_stride == 0 else None

    def frame_counts(self):
        H, L, C, k, w = self.hidden, self.depth, N_IN, self.frame_stride, self.qwindow
        c = _zero_counts()
        c['mac_in'] = k * (C * C + w * C * 3 * H)
        c['mac_syn'] = k * (L - 1) * w * H * 3 * H                          # no recurrent weights
        c['mac_neuron'] = k * L * 3 * H                                     # f*c, (1-f)*z, o*c
        c['elementwise'] = k * L * 3 * H
        c['mac_readout'] = H * N_CLASSES
        c['state_words'] = k * (L * H + (w - 1) * (C + (L - 1) * H))        # c, and the previous step's inputs
        return c


class WienerNet(nn.Module):
    """Wiener filter: every 4 bins, one linear map from the last `taps` 20 ms bins (taps x channels values) to the
    41 class scores, fitted in closed form (ridge regression onto force-aligned frame labels) by
    classic_decoders.py; `scale` turns the scores into CTC logits."""

    def __init__(self, n_sessions, taps=8, every=4):
        super().__init__()
        self.lin = nn.Linear(N_IN * taps, N_CLASSES)
        self.register_buffer('scale', torch.ones(()))
        self.frame_kernel, self.frame_stride = int(taps), int(every)
        self.num_input_channels, self.depth, self.input_buffer = N_IN, 1, True

    def output_lengths(self, lengths):
        return lengths // self.frame_stride

    def frames(self, x):
        """(N, T, C) -> (N, T // every, taps * C): frame k = bins [4k + 4 - taps, 4k + 4), zeros before the start."""
        K, S = self.frame_kernel, self.frame_stride
        xp = F.pad(x, (0, 0, max(K - S, 0), 0))
        f = xp.unfold(1, K, S)[:, :x.shape[1] // S]                       # (N, F, C, K)
        return f.transpose(2, 3).reshape(x.shape[0], f.shape[1], -1)

    def forward(self, x, day_idx):
        return self.scale * self.lin(self.frames(x))

    def reset_stream(self):
        self._buf, self._n = [], 0

    def stream_step(self, x, day_idx):
        self._buf = (self._buf + [x])[-self.frame_kernel:]
        self._n += 1
        if self._n % self.frame_stride:
            return None
        pad = [torch.zeros_like(x)] * (self.frame_kernel - len(self._buf))
        return self.scale * self.lin(torch.cat(pad + self._buf, -1))

    def frame_counts(self):
        c = _zero_counts()
        c['mac_readout'] = self.frame_kernel * N_IN * N_CLASSES + N_CLASSES
        return c


class KalmanNet(nn.Module):
    """Steady-state Kalman filter: the state is the 41 class scores; each 80 ms frame's observation is the mean of
    its 4 bins. With the converged gain K (fitted by classic_decoders.py with the transition A and observation H),
    x_t = M x_{t-1} + K (z_t - mean_z), M = A - K H A, on centred states; logits = scale * (x_t + mean_x)."""

    def __init__(self, n_sessions, every=4):
        super().__init__()
        self.M = nn.Parameter(torch.zeros(N_CLASSES, N_CLASSES))
        self.K = nn.Parameter(torch.zeros(N_CLASSES, N_IN))
        self.register_buffer('mean_x', torch.zeros(N_CLASSES))
        self.register_buffer('mean_z', torch.zeros(N_IN))
        self.register_buffer('scale', torch.ones(()))
        self.frame_kernel = self.frame_stride = int(every)
        self.num_input_channels, self.depth, self.input_buffer = N_IN, 1, False

    def output_lengths(self, lengths):
        return lengths // self.frame_stride

    def forward(self, x, day_idx):
        N, T, C = x.shape
        n = T // self.frame_stride
        z = x[:, :n * self.frame_stride].reshape(N, n, self.frame_stride, C).mean(2) - self.mean_z
        drive = z @ self.K.T                                                 # (N, n, classes)
        s, out = x.new_zeros(N, N_CLASSES), []
        for t in range(n):
            s = s @ self.M.T + drive[:, t]
            out.append(s)
        return self.scale * (torch.stack(out, 1) + self.mean_x)

    def reset_stream(self):
        self._s, self._acc, self._n = None, None, 0

    def stream_step(self, x, day_idx):
        self._acc = x if self._acc is None else self._acc + x
        self._n += 1
        if self._n % self.frame_stride:
            return None
        z = self._acc / self.frame_stride - self.mean_z
        self._acc = None
        self._s = z @ self.K.T if self._s is None else self._s @ self.M.T + z @ self.K.T
        return self.scale * (self._s + self.mean_x)

    def frame_counts(self):
        c = _zero_counts()
        c['acc_in'] = self.frame_stride * N_IN                               # summing the frame's bins
        c['mac_syn'] = N_CLASSES * N_CLASSES + N_CLASSES * N_IN              # M x and K z
        c['mac_readout'] = N_CLASSES
        c['state_words'] = N_CLASSES + N_IN
        return c


class WienerDayNet(nn.Module):
    """Wiener filter with one set of weights per recording session (a daily-recalibrated linear decoder, as
    classic BCI decoders are used): the same as WienerNet, the weights chosen by the session."""

    def __init__(self, n_sessions, taps=8, every=4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_sessions, N_CLASSES, N_IN * taps))
        self.bias = nn.Parameter(torch.zeros(n_sessions, N_CLASSES))
        self.register_buffer('scale', torch.ones(()))
        self.frame_kernel, self.frame_stride = int(taps), int(every)
        self.num_input_channels, self.depth, self.input_buffer = N_IN, 1, True

    output_lengths = WienerNet.output_lengths
    frames = WienerNet.frames

    def forward(self, x, day_idx):
        return self.scale * (self.frames(x) @ self.weight[day_idx].T + self.bias[day_idx])

    def reset_stream(self):
        self._buf, self._n = [], 0

    def stream_step(self, x, day_idx):
        self._buf = (self._buf + [x])[-self.frame_kernel:]
        self._n += 1
        if self._n % self.frame_stride:
            return None
        pad = [torch.zeros_like(x)] * (self.frame_kernel - len(self._buf))
        return self.scale * (torch.cat(pad + self._buf, -1) @ self.weight[day_idx].T + self.bias[day_idx])

    def frame_counts(self):
        return WienerNet.frame_counts(self)

    def per_session_params(self):
        return self.weight[0].numel() + self.bias[0].numel()


class KalmanDayNet(nn.Module):
    """Steady-state Kalman filter with one model per recording session (transition, observation model, noise and
    gain fitted per day); otherwise as KalmanNet."""

    def __init__(self, n_sessions, every=4):
        super().__init__()
        self.M = nn.Parameter(torch.zeros(n_sessions, N_CLASSES, N_CLASSES))
        self.K = nn.Parameter(torch.zeros(n_sessions, N_CLASSES, N_IN))
        self.register_buffer('mean_x', torch.zeros(n_sessions, N_CLASSES))
        self.register_buffer('mean_z', torch.zeros(n_sessions, N_IN))
        self.register_buffer('scale', torch.ones(()))
        self.frame_kernel = self.frame_stride = int(every)
        self.num_input_channels, self.depth, self.input_buffer = N_IN, 1, False

    output_lengths = KalmanNet.output_lengths

    def forward(self, x, day_idx):
        N, T, C = x.shape
        n = T // self.frame_stride
        M, K = self.M[day_idx], self.K[day_idx]
        z = x[:, :n * self.frame_stride].reshape(N, n, self.frame_stride, C).mean(2) - self.mean_z[day_idx]
        drive = z @ K.T
        s, out = x.new_zeros(N, N_CLASSES), []
        for t in range(n):
            s = s @ M.T + drive[:, t]
            out.append(s)
        return self.scale * (torch.stack(out, 1) + self.mean_x[day_idx])

    def reset_stream(self):
        self._s, self._acc, self._n = None, None, 0

    def stream_step(self, x, day_idx):
        self._acc = x if self._acc is None else self._acc + x
        self._n += 1
        if self._n % self.frame_stride:
            return None
        z = self._acc / self.frame_stride - self.mean_z[day_idx]
        self._acc = None
        K, M = self.K[day_idx], self.M[day_idx]
        self._s = z @ K.T if self._s is None else self._s @ M.T + z @ K.T
        return self.scale * (self._s + self.mean_x[day_idx])

    def frame_counts(self):
        return KalmanNet.frame_counts(self)

    def per_session_params(self):
        return self.M[0].numel() + self.K[0].numel()


MODELS = {
    'gru_512': lambda n: GRUNet(n, 512),
    'gru_1024': lambda n: GRUNet(n, 1024),
    'gru_1024_out80': lambda n: GRUNet(n, 1024, out_stride=4),   # steps every 20 ms, outputs every 80 ms
    'gru_small': lambda n: GRUNet(n, 148),
    'ff_twin': lambda n: FFTwin(n),
    'window32': lambda n: Window(n, 512),
    'window32_small': lambda n: Window(n, 128),
    'gru_stack_2x512': lambda n: StackGRU(n, 512, 2),
    'gru_stack_2x896': lambda n: StackGRU(n, 896, 2),     # ~29M: the size of the 2 x 2048 recurrent SNN
    'gru_stack_5x1024': lambda n: StackGRU(n, 1024, 5),
    # the paper's real-time decoder (Extended Data Fig. 5: 14-bin kernel, stride 4, 5 layers of 512 units)
    'gru_stack_5x512_k14': lambda n: StackGRU(n, 512, 5, window=14),
    # the GRU twin of the short-window SNN runs (rsnn_window): 4 bins, every 4, so each bin enters once
    'gru_stack_2x512_k4': lambda n: StackGRU(n, 512, 2, window=4),
    # benchmark decoders, same input and outputs as gru_1024_out80 (classic_decoders.py fits the linear two)
    'lstm_1024_out80': lambda n: LSTMNet(n, 1024),
    'qrnn_1024_out80': lambda n: QRNNNet(n, 1024),
    'wiener_160ms': lambda n: WienerNet(n, taps=8),
    'kalman_80ms': lambda n: KalmanNet(n),
    'wiener_160ms_day': lambda n: WienerDayNet(n, taps=8),       # one filter per session
    'kalman_80ms_day': lambda n: KalmanDayNet(n),
}


def out_lengths(model, lengths):
    return model.output_lengths(lengths) if hasattr(model, 'output_lengths') else lengths


def evaluate(model, batches, device, smooth_sd=0.0):
    model.eval()
    loss_sum, n_batches, edit, phones = 0.0, 0, 0, 0
    with torch.no_grad():
        for b in batches:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            logits = model(smooth_inputs(b['x'], smooth_sd), b['day_idx'])
            lengths = out_lengths(model, b['lengths'])
            loss_sum += ctc_loss(logits, b, BLANK, lengths).item()
            n_batches += 1
            logits = logits.cpu().numpy()
            targets = np.split(b['targets'].cpu().numpy(), np.cumsum(b['target_lengths'].cpu().numpy())[:-1])
            for i, (length, target) in enumerate(zip(lengths.cpu().numpy(), targets)):
                edit += edit_distance(greedy_decode(logits[i, :length], BLANK), target)
                phones += len(target)
    return loss_sum / n_batches, edit / phones


def main():
    global N_IN
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', required=True, help='writes <out-dir>/<model>.json (history) and <model>.pt (weights)')
    p.add_argument('--sessions', nargs='+', default=SWEEP_SESSIONS)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--recipe', default='default', choices=['default', 'rnn'],
                   help="default: AdamW lr 1e-3, wd 1e-3, clip 1; rnn: the paper RNN's -- Adam eps 0.1, "
                        "L2 1e-5 in the gradient, lr 0.02 decayed linearly to 0 over all batches, clip 10")
    p.add_argument('--lr', type=float, default=None, help='default 1e-3, or 0.02 with --recipe rnn')
    p.add_argument('--weight-decay', type=float, default=None, help='default 1e-3, or 1e-5 with --recipe rnn')
    p.add_argument('--white-noise-sd', type=float, default=0.3, help='as SNN configuration C')
    p.add_argument('--constant-offset-sd', type=float, default=0.2)
    p.add_argument('--smooth-sd', type=float, default=0.0,
                   help="the RNN's (non-causal) Gaussian input smoothing, in bins; 2 = the RNN's")
    p.add_argument('--tag', default='', help='suffix for the run name and its .json (e.g. _smoothed)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--index', type=int, default=0, help='array index, for the log header')
    args = p.parse_args()
    name = args.model + args.tag

    torch.manual_seed(args.seed)
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = SessionBatches(load_partitions(args.data_dir, args.sessions, 'train'), args.batch_size, True, args.seed)
    test = SessionBatches(load_partitions(args.data_dir, args.sessions, 'test'), args.batch_size, False)
    N_IN = int(train.parts[0].features[0].shape[1])      # 256 (derived/snnData) or 128 (snnDataTx1)
    model = MODELS[args.model](len(args.sessions)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_core = n_params - sum(p.numel() for p in model.inp.parameters())
    rnn_recipe = args.recipe == 'rnn'
    args.lr = args.lr if args.lr is not None else (0.02 if rnn_recipe else 1e-3)
    args.weight_decay = args.weight_decay if args.weight_decay is not None else (1e-5 if rnn_recipe else 1e-3)
    clip = 10.0 if rnn_recipe else 1.0
    if rnn_recipe:
        opt = torch.optim.Adam(model.parameters(), args.lr, eps=0.1, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    total_steps, step = max(1, args.epochs * len(train)), 0

    print(f'[{args.index}] {name}: {n_core:,} params excluding the per-session input layers '
          f'({n_params:,} total) | {args.epochs} epochs | sessions: {" ".join(args.sessions)}', flush=True)
    print(f'device {device}, {torch.get_num_threads()} threads | {train.n_trials} train, {test.n_trials} test trials | '
          f'noise {args.white_noise_sd}, offset {args.constant_offset_sd}, smoothing sd {args.smooth_sd} | '
          f'recipe {args.recipe}: lr {args.lr}, wd {args.weight_decay}, clip {clip}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f'{name}.json')
    history = {'train_loss': [], 'test_loss': [], 'test_per': [], 'epoch_s': []}
    for epoch in range(1, args.epochs + 1):
        t0, loss_sum, n = time.time(), 0.0, 0
        model.train()
        for b in train:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            x = smooth_inputs(augment(b['x'], args.white_noise_sd, args.constant_offset_sd), args.smooth_sd)
            loss = ctc_loss(model(x, b['day_idx']), b, BLANK, out_lengths(model, b['lengths']))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if rnn_recipe:
                step += 1
                for g in opt.param_groups:
                    g['lr'] = args.lr * max(0.0, 1.0 - step / total_steps)
            loss_sum, n = loss_sum + loss.item(), n + 1
        test_loss, test_per = evaluate(model, test, device, args.smooth_sd)
        dt = time.time() - t0
        for k, v in zip(history, (loss_sum / n, test_loss, test_per, dt)):
            history[k].append(v)
        extra = ''
        if isinstance(model, FFTwin):
            extra = (f' | tau {[round(t, 2) for t in model.taus().tolist()]}'
                     f' ema {torch.sigmoid(model.ema_raw).item():.3f}')
        print(f'Epoch {epoch}/{args.epochs} ({dt:.1f}s) | train loss {loss_sum / n:.4f} | '
              f'test loss {test_loss:.4f} | test PER {test_per:.4f}{extra}', flush=True)
        with open(out_file, 'w') as f:
            json.dump({'model': name, 'params_core': n_core, 'params_total': n_params,
                       'args': vars(args), 'history': history}, f, indent=1)
    # the trained weights, for scripts/efficiency.py (quantization, re-evaluation)
    torch.save(model.state_dict(), os.path.splitext(out_file)[0] + '.pt')
    print(f'Training finished; best test PER {min(history["test_per"]):.4f} | saved {out_file} and its .pt', flush=True)


if __name__ == '__main__':
    main()
