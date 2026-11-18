"""
Latency of every decoder of the three experiments on one machine: compute time per output, one CPU thread, one
sample at a time, the way each decoder runs online. Every network is rebuilt at its trained size with random weights
(time does not depend on the weights), so no data or checkpoints are needed. All decoders are timed in PyTorch or
NumPy, so framework overhead is the same for all (the earlier motor laptop run timed the ANNs in Keras, whose
per-call overhead dominated).

    motor  (an output every 4 ms)   kf, wf, lstm, qrnn, lsnn, speck_host
    audio  (every 40 ms)            kf, wf, dense_nn, lstm, gru, lsnn, speck_host
    speech (every 80 ms)            lstm, paper_rnn, lsnn

kf is the steady-state Kalman filter (x <- A x + K (z - H x)); kf_full also recomputes the gain every step, as
spike_bmi's predict() does. speck_host is the host's share of a Speck decoder (input buffer and delay taps, and the
readout); the chip's own time is measured on the devkit (batches 1 and 2).

    python experiments/chip/latency_all.py --out latency.json
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from snndec.model import SNNDecoder  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs')


def timeit(fn, seconds=2.0, warmup=30):
    """Per-call times in ms: warm up, then call for about `seconds`."""
    for _ in range(warmup):
        fn()
    ts, start = [], time.perf_counter()
    while time.perf_counter() - start < seconds or len(ts) < 50:
        t = time.perf_counter()
        fn()
        ts.append(1e3 * (time.perf_counter() - t))
    ts.sort()
    return {'median_ms': statistics.median(ts), 'p95_ms': ts[int(0.95 * (len(ts) - 1))], 'calls': len(ts)}


def kf(s, c, full=False):
    rng = np.random.default_rng(0)
    A, H = rng.standard_normal((s, s)).astype(np.float32) * 0.1, rng.standard_normal((c, s)).astype(np.float32)
    K, z = rng.standard_normal((s, c)).astype(np.float32) * 0.01, rng.standard_normal(c).astype(np.float32)
    W, Q = np.eye(s, dtype=np.float32), np.eye(c, dtype=np.float32)
    st = {'x': np.zeros(s, np.float32), 'P': np.eye(s, dtype=np.float32)}

    def steady():
        xm = A @ st['x']
        st['x'] = xm + K @ (z - H @ xm)

    def recursion():
        Pm = A @ st['P'] @ A.T + W
        G = Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + Q)
        xm = A @ st['x']
        st['x'] = xm + G @ (z - H @ xm)
        st['P'] = (np.eye(s, dtype=np.float32) - G @ H) @ Pm
    return recursion if full else steady


def wf(n_in, n_out):
    rng = np.random.default_rng(0)
    Wm, x = rng.standard_normal((n_in, n_out)).astype(np.float32), rng.standard_normal(n_in).astype(np.float32)
    return lambda: x @ Wm


class QRNN(nn.Module):
    """The motor QRNN: a width-2 convolution over time to (z, f, o), then fo-pooling over the window."""
    def __init__(self, n_in, units, window=2):
        super().__init__()
        self.conv = nn.Conv1d(n_in, 3 * units, window)
        self.units = units

    def forward(self, x):                                        # (1, T, C) -> (1, units)
        g = self.conv(nn.functional.pad(x.transpose(1, 2), (1, 0)))
        z, f, o = torch.tanh(g[:, :self.units]), torch.sigmoid(g[:, self.units:2 * self.units]), torch.sigmoid(g[:, 2 * self.units:])
        c = torch.zeros(1, self.units)
        for t in range(x.shape[1]):
            c = f[:, :, t] * c + (1 - f[:, :, t]) * z[:, :, t]
        return o[:, :, -1] * c


@torch.no_grad()
def windowed(rnn, head, T, C):
    """An RNN run over the whole window for every output (how the pipelines' windowed decoders predict)."""
    x = torch.randn(1, T, C)
    if isinstance(rnn, QRNN):
        return lambda: head(rnn(x))
    return lambda: head(rnn(x)[0][:, -1])


@torch.no_grad()
def streaming_rnn(rnn, head, steps, C):
    """A stateful RNN fed `steps` new input steps per output (the speech decoders)."""
    x, st = torch.randn(1, steps, C), {'h': None}

    def f():
        y, st['h'] = rnn(x, st['h'])
        return head(y[:, -1])
    return f


def snn(config, steps_per_output, **overrides):
    """An snndec network built from its config, stepped online (model.step), one output's worth of input steps."""
    cfg = yaml.safe_load(open(os.path.join(CONFIGS, config)))
    spec = cfg['model']
    for k, v in overrides.items():
        d = spec
        *path, last = k.split('.')
        for q in path:
            d = d[q]
        d[last] = v
    for k in ('n_channels', 'n_delays', 'input_norm', 'delay_max'):     # resolved from data at training time
        spec.get('input', {}).pop(k, None)
    model = SNNDecoder(spec).eval()
    x = (torch.rand(steps_per_output, 1, spec['n_channels']) < 0.1).float()
    day = 0 if spec.get('n_days') else None

    @torch.no_grad()
    def f():
        for t in range(steps_per_output):
            model.step(x[t], day)
    return f, model


def speck_host(config, steps_per_output, **overrides):
    """The host's share of a Speck decoder per output: the input encoding and delay taps (model.host) and the
    readout, without the chip's layers."""
    _, model = snn(config, steps_per_output, **overrides)
    widths = model.widths
    st = {'host': model.host.init_state(), 'ro': model.readout.init_state(1, 'cpu')}
    x = (torch.rand(steps_per_output, 1, model.spec['n_channels']) < 0.1).float()
    h = (torch.rand(1, widths[-1]) < 0.05).float()

    @torch.no_grad()
    def f():
        for t in range(steps_per_output):
            for _ in model.host.step(x[t], st['host']):
                model.readout.step(h, st['ro'])
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=2.0)
    a = p.parse_args()
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    T = lambda fn: timeit(fn, a.seconds)
    audio_speck = {'input.channels': list(range(48)), 'input.delays': [0, 2, 4, 5, 7, 9, 11, 12, 14, 16],
                   'input.input_scale': [1.0] * 48}
    plan = {
        'motor': {
            'interval_ms': 4,
            'kf': lambda: kf(6, 96), 'kf_full': lambda: kf(6, 96, full=True), 'wf': lambda: wf(15 * 96, 2),
            'lstm': lambda: windowed(nn.LSTM(96, 200, batch_first=True), nn.Linear(200, 2), 5, 96),
            'qrnn': lambda: windowed(QRNN(96, 400), nn.Linear(400, 2), 5, 96),
            'lsnn': lambda: snn('motor_lsnn.yaml', 1, **{'readout.bias_init': 0.5})[0],
            'speck_host': lambda: speck_host('motor_speck.yaml', 1, **{'readout.neutral': 0.5}),
        },
        'audio': {
            'interval_ms': 40,
            'kf': lambda: kf(128, 96), 'kf_full': lambda: kf(128, 96, full=True), 'wf': lambda: wf(17 * 96, 128),
            'dense_nn': lambda: (lambda m, x: (lambda: m(x)))(
                nn.Sequential(nn.Linear(17 * 96, 1024), nn.ReLU(), nn.Linear(1024, 128)).eval(), torch.randn(1, 17 * 96)),
            'lstm': lambda: windowed(nn.LSTM(96, 1024, batch_first=True), nn.Linear(1024, 128), 17, 96),
            'gru': lambda: windowed(nn.GRU(96, 1024, batch_first=True), nn.Linear(1024, 128), 17, 96),
            # the pipeline's LSNN predicts each window from rest: 17 steps per output
            'lsnn': lambda: snn('audio_lsnn.yaml', 17)[0],
            'speck_host': lambda: speck_host('audio_speck.yaml', 1, **audio_speck),
        },
        'speech': {
            'interval_ms': 80,
            # 4 new 20 ms bins per 80 ms output; the LSTM steps through them, the paper's GRU reads a 32-bin
            # window (8192 inputs) once per output
            'lstm': lambda: streaming_rnn(nn.LSTM(256, 1024, 2, batch_first=True), nn.Linear(1024, 41), 4, 256),
            'paper_rnn': lambda: streaming_rnn(nn.GRU(256 * 32, 1024, 5, batch_first=True), nn.Linear(1024, 41), 1, 256 * 32),
            'lsnn': lambda: snn('speech_lsnn.yaml', 4)[0],
        },
    }
    out = {'machine': {'platform': platform.platform(), 'processor': platform.processor() or platform.machine(),
                       'python': platform.python_version(), 'torch': torch.__version__, 'threads': 1}, 'results': {}}
    for exp, decs in plan.items():
        iv = decs.pop('interval_ms')
        for name, build in decs.items():
            r = T(build())
            r['interval_ms'] = iv
            out['results'][f'{exp}/{name}'] = r
            print(f"{exp:6s} {name:10s} {r['median_ms']:9.4f} ms (p95 {r['p95_ms']:.4f})  budget {iv} ms")
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
