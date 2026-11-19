"""
Inputs and splits for the GRU references, under two protocols.

Both build 4 ms spike-count bins, bin k = spikes in [t[k], t[k+1]), and decode the velocity at sample T from the
WS x N bins before it (bins T - WS*N .. T-1), summed into N sub-windows of WS bins (the paper's Fig. 2), then
x' = log(softplus(x)) (their eq. 16). The reconstruction target is the raw sub-window count x.

  neurobench : the BioCAS 2024 challenge / NeuroBench primate-reaching protocol, from the raw .mat: all channels
               (96 for indy, 192 for loco), units merged per channel as NeuroBench does (a bin is 1 if any unit
               fired), velocity = gradient of cursor_pos per sample (NeuroBench's labels), reach segments from target
               changes, split by segment order 50 / 25 / 25 (train / validation / test).
  ours       : this paper's motor protocol, from process_data.py output: the 96 M1 channels' multi-unit spike
               counts, velocity = task_data[:, 2:4], and the same chronological boundary as every other decoder
               (aligned_train_boundary, last 10% test). ANN row r decodes sample r + 65, so test row r is directly
               comparable with results/final_v2 (arrays y_true). Training rows end GAP rows before the boundary.
"""
import math
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.preprocessing import aligned_train_boundary  # noqa: E402

ANN_WINDOW = 65          # ANN row r's window is raw samples [r, r + 65); it decodes sample r + 65
TEST_FRAC = 0.1
GAP_ROWS = 136           # the LSTM's purge gap (split_gap_samples); the GRU's 400 ms input needs 100


def bin_counts(spike_lists, t, binary=False):
    """(len(t), n_channels) counts per 4 ms bin [t[k], t[k+1]); the last bin is half-open to t[-1] + dt."""
    edges = np.append(t, t[-1] + np.median(np.diff(t)))
    out = np.zeros((len(t), len(spike_lists)), dtype=np.float32)
    for c, s in enumerate(spike_lists):
        if len(s):
            out[:, c] = np.histogram(s, bins=edges)[0]
    return (out > 0).astype(np.float32) if binary else out


def windows(bins, targets, ws, n):
    """Sub-window counts (len(targets), n, C) for each target sample T: bins T-ws*n .. T-1, summed by ws."""
    span = ws * n
    targets = np.asarray(targets)
    assert targets.min() >= span, "a target lacks a full history"
    cs = np.concatenate([np.zeros((1, bins.shape[1]), np.float32), np.cumsum(bins, axis=0, dtype=np.float64)])
    starts = targets[:, None] - span + ws * np.arange(n)[None, :]          # (M, n) first bin of each sub-window
    return (cs[starts + ws] - cs[starts]).astype(np.float32)


def transform(x):
    """x' = log(softplus(x)) (paper eq. 16)."""
    return np.log(np.log1p(np.exp(x))).astype(np.float32)


def load_neurobench(mat_path, binary=True):
    """(bins (T, C) binary, velocity (T, 2), split {train, val, test}: target sample indices)."""
    with h5py.File(mat_path, 'r') as f:
        cursor = f['cursor_pos'][()].T.astype(np.float64)       # (T, 2)
        target = f['target_pos'][()].T
        t = f['t'][()].squeeze()
        refs = f['spikes'][()].T                                # (channels, units)
        spikes = [np.concatenate([f[refs[c, u]][()].ravel() for u in range(refs.shape[1])
                                  if f[refs[c, u]].ndim == 2] or [np.empty(0)])
                  for c in range(refs.shape[0])]
    bins = bin_counts(spikes, t, binary=binary)
    vel = np.gradient(cursor, axis=0)                           # per sample, as NeuroBench's torch.gradient
    change = np.flatnonzero(np.abs(np.diff(target, axis=0)).sum(axis=1)) + 1
    bounds = np.concatenate([[0], change, [len(t)]])
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    n_tr = math.floor(0.5 * len(segs))
    n_va = math.floor((len(segs) - n_tr) / 2)
    parts = {'train': segs[:n_tr], 'val': segs[n_tr:n_tr + n_va], 'test': segs[n_tr + n_va:]}
    split = {k: np.concatenate([np.arange(a, b) for a, b in v]) for k, v in parts.items()}
    return bins, vel.astype(np.float32), split, {'n_segments': len(segs), 'n_channels': bins.shape[1]}


def load_ours(processed_path, span):
    """(bins (T, 96) counts, velocity (T, 2), split {train, test}: target samples, info).
    test: samples n_train + 65 + r for every ANN test row r; train: every row r < n_train - GAP_ROWS, as for the other
    reference decoders (a fixed schedule; nothing is selected on held-out data, so there is no validation split)."""
    with h5py.File(processed_path, 'r') as f:
        t = f['task_time'][()]
        vel = f['task_data'][()][:, 2:4].astype(np.float32)
        spikes = [np.asarray(s) for s in f['mua_trains'][()]]
    T = len(t)
    n_train = aligned_train_boundary(T, TEST_FRAC, ANN_WINDOW)
    n_rows = T - ANN_WINDOW
    test = np.arange(n_train, n_rows) + ANN_WINDOW
    train_all = np.arange(0, n_train - GAP_ROWS) + ANN_WINDOW
    train_all = train_all[train_all >= span]
    split = {'train': train_all, 'test': test}
    return bin_counts(spikes, t), vel, split, {'n_train_rows': int(n_train), 'n_channels': len(spikes),
                                                'first_test_row': int(n_train)}
