"""
The final latency and energy measurement of every decoder of the three experiments, on the Speck laptop: compute time
per output and RAPL energy per output, one CPU thread, one output at a time, the way each decoder runs online.

Every network is rebuilt at its trained size with random weights: time and energy depend on the sizes, not the
weights, so no data or checkpoints are needed (accuracy comes from the test-set runs, Speck's chip power and time from
the devkit runs). All decoders run in PyTorch or NumPy, so framework overhead is the same for all.

Energy: Intel RAPL package counters (/sys/class/powercap/intel-rapl:N, the 'package-N' domains), idle-subtracted:
the laptop's idle package power is measured before and after (the mean of the two), and a decoder's energy per output
is (package energy over its run - idle power x run time) / outputs. The raw energy and mean power are kept too.

    python experiments/chip/final_bench.py --out final_bench.json             (RAPL needs read access; see below)
    python experiments/chip/final_bench.py --out x.json --only motor/lsnn --seconds 2

RAPL counters are root-only on recent Linux kernels: run with sudo, or once per boot
    sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj

What each decoder is (sizes as trained):
    motor  indy/loco: 96 channels, an output every 4 ms; jenkins: 192 channels
        kf (steady-state gain) / kf_full (the gain recomputed every step, as spike_bmi's predict()), wf (15 taps),
        lstm (200 units, 5-step window), qrnn (400 units, window 5), lsnn (2 x 256), speck_host (host buffer + readout)
    audio  96 channels, an output every 40 ms, 17-frame windows (8 past, centre, 8 ahead)
        kf, kf_full, wf, wiener_cascade (wf + a cubic per band), dense_nn (1024), simple_rnn / gru / lstm (1024, over
        the window), lsnn (2 x 256, the window from rest), speck_host
    speech 256 features per 20 ms bin, an output every 80 ms (4 bins)
        kalman (41 states), wiener (8 bins), gru / lstm (1024, stepping every bin), paper_rnn (5 x 1024 GRU on a
        32-bin window, stride 4), lsnn (2 x 1024 recurrent, readout of both layers), speck_host
"""

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latency_all import QRNN, kf, snn, speck_host, streaming_rnn, wf, windowed  # noqa: E402


# --------------------------------------------------------------------------------------------------------- RAPL
def _rapl_paths():
    paths = []
    for d in sorted(glob.glob('/sys/class/powercap/intel-rapl/intel-rapl:*')):
        if ':' in os.path.basename(d).split('intel-rapl:')[1]:
            continue                                                  # sub-domains (core, uncore, dram)
        try:
            name = open(f'{d}/name').read().strip()
        except OSError:
            continue
        if name.startswith('package'):
            mx = int(open(f'{d}/max_energy_range_uj').read())
            paths.append((f'{d}/energy_uj', mx))
    return paths


RAPL = _rapl_paths()


def rapl_uj():
    """Per-package counters (µJ), or None when unreadable."""
    try:
        return [int(open(p).read()) for p, _ in RAPL] if RAPL else None
    except OSError:
        return None


def rapl_delta_j(e0, e1):
    return sum(((b - a) % mx) for (a, b, (_, mx)) in zip(e0, e1, RAPL)) * 1e-6


def idle_power_w(seconds):
    e0, t0 = rapl_uj(), time.perf_counter()
    time.sleep(seconds)
    e1, t1 = rapl_uj(), time.perf_counter()
    return rapl_delta_j(e0, e1) / (t1 - t0)


def measure(fn, seconds, idle_w, warmup=30):
    """Per-call times (ms) and energy over a run of about `seconds`."""
    for _ in range(warmup):
        fn()
    ts = []
    e0, start = rapl_uj(), time.perf_counter()
    while time.perf_counter() - start < seconds or len(ts) < 50:
        t = time.perf_counter()
        fn()
        ts.append(1e3 * (time.perf_counter() - t))
    run_s, e1 = time.perf_counter() - start, rapl_uj()
    ts.sort()
    r = {'median_ms': statistics.median(ts), 'p95_ms': ts[int(0.95 * (len(ts) - 1))], 'calls': len(ts), 'run_s': run_s}
    if e0 is not None and e1 is not None:
        e = rapl_delta_j(e0, e1)
        r.update(energy_raw_uj=1e6 * e / len(ts), power_w=e / run_s,
                 energy_uj=1e6 * max(e - idle_w * run_s, 0.0) / len(ts))
    return r


# ------------------------------------------------------------------------------------------------- extra decoders
def wiener_cascade(n_in, n_out, degree=3):
    """A Wiener filter, then a polynomial per output (the audio pipeline's Wiener cascade)."""
    rng = np.random.default_rng(0)
    Wm, x = rng.standard_normal((n_in, n_out)).astype(np.float32), rng.standard_normal(n_in).astype(np.float32)
    P = rng.standard_normal((degree + 1, n_out)).astype(np.float32)

    def f():
        y = x @ Wm
        out = P[degree].copy()
        for k in range(degree - 1, -1, -1):                       # Horner, per band
            out = out * y + P[k]
        return out
    return f


@torch.no_grad()
def kalman_speech(n_states=41, n_in=256):
    """speech's Kalman control: x <- M x + K z, once per 80 ms output, z the mean of its 4 bins."""
    M, K = torch.randn(n_states, n_states), torch.randn(n_states, n_in)
    st, z = {'x': torch.zeros(n_states)}, torch.randn(4, n_in)

    def f():
        st['x'] = M @ st['x'] + K @ z.mean(0)
        return st['x']
    return f


@torch.no_grad()
def speech_speck_host(levels=(0.5, 1.5), delays=(0, 13), n_ch=128, layer_widths=(128, 256, 256), read_all=True,
                      traces=(0, 5, 15, 30, 60), n_out=41, extra_taps=()):
    """The host's share of the speech Speck decoder per 80 ms output (4 bins): per bin, the per-day gain, the level
    lines, the delay ring buffer and its taps; the readout of every layer's spikes with their traces, and the leaky
    output neurons."""
    D = max([*delays] + [d for _, ds in extra_taps for d in ds]) + 1    # extra_taps: (first n lines, delays) more
    gain, x = torch.rand(n_ch) + 0.5, torch.randn(4, n_ch)
    n_read = sum(layer_widths) if read_all else layer_widths[-1]
    R = nn.Linear(n_read * len(traces), n_out)
    alphas = torch.tensor([0.0 if t == 0 else float(np.exp(-1.0 / t)) for t in traces])[:, None]
    st = {'buf': torch.zeros(D, n_ch * len(levels)), 'i': 0, 'tr': torch.zeros(len(traces), n_read),
          'out': torch.zeros(n_out)}
    spikes = (torch.rand(4, n_read) < 0.4).float()                 # the chip's spikes, as read back

    def f():
        for b in range(4):
            u = x[b] * gain
            lines = torch.cat([(u >= v).float() for v in levels])
            st['buf'][st['i'] % D] = lines
            taps = torch.cat([st['buf'][(st['i'] - d) % D] for d in delays]           # -> the chip
                             + [st['buf'][(st['i'] - d) % D][:k] for k, ds in extra_taps for d in ds])
            st['i'] += 1
            st['tr'] = alphas * st['tr'] + (1 - alphas) * spikes[b]
            st['out'] = 0.6 * st['out'] + 0.4 * R(st['tr'].reshape(-1))
        return taps, st['out']
    return f


# round 16's one-chip speech networks (speechBCI snnDecoder configs/speck_r16.txt); SPEECH_SPECK picks the final one
SPEECH_SPECK = {'R1': {}, 'R2': {'n_ch': 256, 'delays': (0,)}, 'R3': {'n_ch': 256, 'levels': (0.5,)},
                'R4': {'n_ch': 256, 'delays': (0,)},
                # round 17 S3: the top 128 features x 4 delays, one 128-256-256 network
                'S3': {'n_ch': 128, 'levels': (0.5,), 'delays': (0, 4, 13, 25)},
                # round 19 V2 (final): all 256 features x 4 delays, two 128-256-64 networks on one chip, every layer
                # of both read with traces (896 neurons)
                'V2': {'n_ch': 256, 'levels': (0.5,), 'delays': (0, 4, 13, 25), 'layer_widths': (256, 512, 128)},
                # round 20 V2+C (final): V2 plus network C, the top 64 features x 0/300/600/1200 ms, 64-256-64; every
                # layer of all three read with traces (1,280 neurons)
                'V2C': {'n_ch': 256, 'levels': (0.5,), 'delays': (0, 4, 13, 25), 'layer_widths': (320, 768, 192),
                        'extra_taps': ((64, (0, 15, 30, 60)),)}}


def plan():
    audio_speck = {'input.channels': list(range(48)), 'input.delays': [0, 2, 4, 5, 7, 9, 11, 12, 14, 16],
                   'input.input_scale': [1.0] * 48}
    hkm_speck = {'n_channels': 192, 'input.delays': [0, 25], 'readout.neutral': 0.5}
    return {
        'motor': {
            'interval_ms': 4,
            'kf': lambda: kf(6, 96), 'kf_full': lambda: kf(6, 96, full=True), 'wf': lambda: wf(15 * 96, 2),
            'lstm': lambda: windowed(nn.LSTM(96, 200, batch_first=True), nn.Linear(200, 2), 5, 96),
            'qrnn': lambda: windowed(QRNN(96, 400), nn.Linear(400, 2), 5, 96),
            'lsnn': lambda: snn('motor_lsnn.yaml', 1, **{'readout.bias_init': 0.5})[0],
            'speck_host': lambda: speck_host('motor_speck.yaml', 1, **{'readout.neutral': 0.5}),
        },
        'motor_jenkins': {
            'interval_ms': 4,
            'kf': lambda: kf(6, 192), 'kf_full': lambda: kf(6, 192, full=True), 'wf': lambda: wf(15 * 192, 2),
            'lstm': lambda: windowed(nn.LSTM(192, 200, batch_first=True), nn.Linear(200, 2), 5, 192),
            'qrnn': lambda: windowed(QRNN(192, 400), nn.Linear(400, 2), 5, 192),
            'lsnn': lambda: snn('motor_lsnn.yaml', 1, **{'n_channels': 192, 'readout.bias_init': 0.5})[0],
            'speck_host': lambda: speck_host('motor_speck.yaml', 1, **hkm_speck),
        },
        'audio': {
            'interval_ms': 40,
            'kf': lambda: kf(128, 96), 'kf_full': lambda: kf(128, 96, full=True), 'wf': lambda: wf(17 * 96, 128),
            'wiener_cascade': lambda: wiener_cascade(17 * 96, 128),
            'dense_nn': lambda: (lambda m, x: (lambda: m(x)))(
                nn.Sequential(nn.Linear(17 * 96, 1024), nn.ReLU(), nn.Linear(1024, 128)).eval(), torch.randn(1, 17 * 96)),
            'simple_rnn': lambda: windowed(nn.RNN(96, 1024, batch_first=True), nn.Linear(1024, 128), 17, 96),
            'gru': lambda: windowed(nn.GRU(96, 1024, batch_first=True), nn.Linear(1024, 128), 17, 96),
            'lstm': lambda: windowed(nn.LSTM(96, 1024, batch_first=True), nn.Linear(1024, 128), 17, 96),
            'lsnn': lambda: snn('audio_lsnn.yaml', 17)[0],              # each window from rest: 17 steps per output
            'speck_host': lambda: speck_host('audio_speck.yaml', 1, **audio_speck),
        },
        'speech': {
            'interval_ms': 80,
            'kalman': lambda: kalman_speech(),
            'wiener': lambda: wf(8 * 256, 41),
            'gru': lambda: streaming_rnn(nn.GRU(256, 1024, batch_first=True), nn.Linear(1024, 41), 4, 256),
            'lstm': lambda: streaming_rnn(nn.LSTM(256, 1024, batch_first=True), nn.Linear(1024, 41), 4, 256),
            'paper_rnn': lambda: streaming_rnn(nn.GRU(256 * 32, 1024, 5, batch_first=True), nn.Linear(1024, 41), 1,
                                               256 * 32),
            'lsnn': lambda: snn('speech_lsnn.yaml', 4)[0],
            'speck_host': lambda: speech_speck_host(**SPEECH_SPECK[os.environ.get('SPEECH_SPECK', 'R1')]),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=5.0, help='run length per decoder')
    p.add_argument('--idle-seconds', type=float, default=15.0, help='idle measurement, before and after')
    p.add_argument('--only', nargs='*', default=None, help='e.g. motor/lsnn audio (an experiment or exp/decoder)')
    a = p.parse_args()
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if rapl_uj() is None:
        print('WARNING: RAPL not readable: times only. Fix: sudo chmod a+r '
              '/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj (or run with sudo)')
    idle0 = idle_power_w(a.idle_seconds) if rapl_uj() is not None else None
    if idle0 is not None:
        print(f'idle package power before: {idle0:.2f} W')
    out = {'machine': {'platform': platform.platform(), 'processor': platform.processor() or platform.machine(),
                       'python': platform.python_version(), 'torch': torch.__version__, 'threads': 1,
                       'rapl_packages': len(RAPL)}, 'results': {}}
    for exp, decs in plan().items():
        iv = decs.pop('interval_ms')
        for name, build in decs.items():
            key = f'{exp}/{name}'
            if a.only and not any(key == o or exp == o for o in a.only):
                continue
            r = measure(build(), a.seconds, idle0 or 0.0)
            r['interval_ms'] = iv
            out['results'][key] = r
            e = f"  {r['energy_uj']:10.2f} µJ/output (raw {r['energy_raw_uj']:.1f}, {r['power_w']:.2f} W)" \
                if 'energy_uj' in r else ''
            print(f"{exp:13s} {name:14s} {r['median_ms']:9.4f} ms (p95 {r['p95_ms']:.4f}, budget {iv} ms){e}", flush=True)
    if idle0 is not None:
        idle1 = idle_power_w(a.idle_seconds)
        out['machine'].update(idle_w_before=idle0, idle_w_after=idle1)
        print(f'idle package power after: {idle1:.2f} W (energies used {idle0:.2f} W)')
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
