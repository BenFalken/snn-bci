"""
The three experiments' preprocessed data, as each project wrote it, read into one shape: a split is a list of
trials, each (x (T, C) float32 per input step, y targets, day index).

    motor   spike_bmi snn_datasets/<experiment>/<subject>/<group>/<session>/{train,test}/*.pkl: input_spikes (C, T)
            counts per 4 ms bin, velocity (T, 2); targets scaled to [margin, 1 - margin] with the subject's bounds
            (velocity_scalers.json), as in training
    audio   AuditoryDecoding pipeline_data/<subject>/<region>/<date>/pipeline: window_features_<split>.npy
            (n, W, C) and window_targets_<split>.npy (n, bands) for the LSNN; features.npy (frames, C) spike counts,
            the windows' source, for the Speck network (window i = frames i .. i + W - 1)
    speech  SNNDecoder data dirs <dir>/<session>/<partition>.npz: features (sum T, C) per 20 ms bin, featureLens,
            labels (phoneme ids), labelLens
"""

import glob
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Trial:
    x: np.ndarray                 # (T, C)
    y: np.ndarray                 # motor (T, 2) scaled velocity; audio (bands,) or (T, bands); speech (L,) labels
    day: Optional[int] = None


# ------------------------------------------------------------------------------------------------------ motor
def velocity_scaler(session_dir: str, scalers_json: str):
    """(v_lo, v_hi, margin) of the session's experiment and subject, read from its path."""
    m = re.search(r'/(bmi|hkm)/([^/]+)/', session_dir + '/')
    if not m:
        raise ValueError(f'{session_dir}: no .../<experiment>/<subject>/... in the path')
    e = json.load(open(scalers_json))[m.group(1)][m.group(2)]
    return e['v_lo'], e['v_hi'], e['margin']


def scale_velocity(v, lo, hi, margin):
    return margin + (1 - 2 * margin) * (v - lo) / (hi - lo)


def unscale_velocity(s, lo, hi, margin):
    return lo + (s - margin) * (hi - lo) / (1 - 2 * margin)


def motor_split(session_dir: str, split: str, scalers_json: str) -> List[Trial]:
    scale = velocity_scaler(session_dir, scalers_json)
    files = sorted(glob.glob(f'{session_dir}/{split}/*.pkl'), key=lambda f: int(re.findall(r'(\d+)\.pkl$', f)[0]))
    out = []
    for f in files:
        d = pickle.load(open(f, 'rb'))
        out.append(Trial(np.asarray(d['input_spikes'], np.float32).T,
                         scale_velocity(np.asarray(d['velocity'], np.float32), *scale)))
    return out


# ------------------------------------------------------------------------------------------------------ audio
PIPELINE_PAST = 8                                                    # the pipeline's samples_before (= samples_after)


def audio_windows(pipeline_dir: str, split: str, past: Optional[int] = None):
    """(X (n, W, C), y (n, bands)) windows of one split. past: frames before each window's centre (the pipeline's
    windows have 8, then the centre, then 8 after); another value rebuilds the windows from features_scaled_full.npy
    (window k is its frames k .. k + 16; with past P, k + 8 - P .. k + 16, zeros before the recording's start), with
    the same targets and the same 8 frames of look-ahead."""
    y = np.load(f'{pipeline_dir}/window_targets_{split}.npy').astype(np.float32)
    if past is None or past == PIPELINE_PAST:
        return np.load(f'{pipeline_dir}/window_features_{split}.npy').astype(np.float32), y
    F = np.load(f'{pipeline_dir}/features_scaled_full.npy').astype(np.float32)
    n = {s: len(np.load(f'{pipeline_dir}/window_targets_{s}.npy', mmap_mode='r')) for s in ('train', 'valid', 'test')}
    if len(F) != sum(n.values()) + 2 * PIPELINE_PAST:
        raise ValueError(f'features_scaled_full.npy has {len(F)} frames; the windows need {sum(n.values()) + 16}')
    start = {'train': 0, 'valid': n['train'], 'test': n['train'] + n['valid']}[split]
    F = np.concatenate([np.zeros((past, F.shape[1]), np.float32), F])            # frame f is now at f + past
    k = np.arange(start, start + n[split])
    idx = (k + PIPELINE_PAST)[:, None] + np.arange(past + 1 + PIPELINE_PAST)[None, :]   # padded: k + 8 - P + P
    return F[idx], y


def audio_recording(pipeline_dir: str, features: str = 'features.npy'):
    """features.npy (frames, C) counts (or another per-frame input file of the same length, e.g. a dataset's
    features_counts_daynorm.npy), and the window length and split sizes to cut its outputs back into windows."""
    raw = np.load(f'{pipeline_dir}/{features}').astype(np.float32)
    if os.path.isfile(f'{pipeline_dir}/lengths.json'):                  # a copy that carries only the lengths
        meta = json.load(open(f'{pipeline_dir}/lengths.json'))
        n, W = meta['n'], meta['W']
    else:
        n = {s: len(np.load(f'{pipeline_dir}/window_targets_{s}.npy', mmap_mode='r')) for s in ('train', 'valid', 'test')}
        W = np.load(f'{pipeline_dir}/window_features_train.npy', mmap_mode='r').shape[1]
    if len(raw) != sum(n.values()) + W - 1:
        raise ValueError(f'features.npy has {len(raw)} frames; the windows need {sum(n.values()) + W - 1}')
    return raw, W, n


# ----------------------------------------------------------------------------------------------------- speech
def speech_split(data_dir: str, sessions: List[str], partition: str) -> List[Trial]:
    out = []
    for day, s in enumerate(sessions):
        path = f'{data_dir}/{s}/{partition}.npz'
        if not os.path.isfile(path):
            continue
        with np.load(path) as f:
            xs = np.split(f['features'], np.cumsum(f['featureLens'])[:-1])
            ys = np.split(f['labels'], np.cumsum(f['labelLens'])[:-1])
        out += [Trial(x.astype(np.float32), y, day) for x, y in zip(xs, ys)]
    return out


def audio_rank_channels(raw: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, ridge: float = 1.0) -> list:
    """Channels ordered by how much target variance each one's window of counts explains on the training frames
    (ridge regression per channel; the audio Speck decoder's ranking)."""
    n_tr, W = len(X_train), X_train.shape[1]
    y = y_train - y_train.mean(0)
    score = []
    for c in range(raw.shape[1]):
        x = np.lib.stride_tricks.sliding_window_view(raw[:n_tr + W - 1, c], W)
        x = x - x.mean(0)
        w = np.linalg.solve(x.T @ x + ridge * np.eye(W), x.T @ y)
        score.append(1 - ((y - x @ w) ** 2).sum() / (y ** 2).sum())
    return [int(c) for c in np.argsort(score)[::-1]]
