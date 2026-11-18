"""
Sentence data for the SNN, exported from the RNN's TFRecords by
AnalysisExamples/exportTFRecordsToNpz.py, so both decoders see identical
features and labels:

    {dataDir}/{session}/{partition}.npz   partition: train, test, competitionHoldOut
        features        (sum T_i, 256) float32  block-normalized tx1 + spikePow (area 6v),
                                                every trial's 20 ms bins concatenated
        featureLens     (N,)                    T_i per trial
        labels          (sum L_i,) int32        phoneme IDs 0-39 (PHONE_DEF_SIL index;
                                                the RNN's seqClassIDs - 1), concatenated
        labelLens       (N,)                    L_i per trial (competitionHoldOut: as in its
                                                TFRecords, whose sentences are redacted)
        transcriptions  (N, 500) int32          ASCII codes, zero-padded (as in the TFRecords)

Optional distillation targets (AnalysisExamples/exportTeacherLogits.py):
{teacherDir}/{session}/{partition}.npz with a trained RNN's per-frame logits
(logits (sum F_i, 41), frameLens (N,)); batches then also carry 'teacher'
(N, F_max, 41) and 'teacher_lengths'.

Batches hold trials of one session only (each session has its own input
layer, as in the RNN), zero-padded to the longest trial in the batch.
"""

import os
from typing import List, Optional

import numpy as np
import torch

# NeuralDecoder/neuralDecoder/configs/dataset/speech_release_baseline.yaml
BASELINE_SESSIONS = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19', 't12.2022.05.24',
    't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07', 't12.2022.06.14', 't12.2022.06.16',
    't12.2022.06.21', 't12.2022.06.28', 't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21',
    't12.2022.07.27', 't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]


class SessionPartition:
    """One session's trials of one partition, held in memory."""

    def __init__(self, path: str, session: str, day_idx: int, teacher_path: Optional[str] = None):
        with np.load(path) as f:
            features, feature_lens = f['features'], f['featureLens']
            labels, label_lens = f['labels'], f['labelLens']
            self.transcriptions = f['transcriptions']
        self.session, self.day_idx = session, day_idx
        self.features = np.split(features, np.cumsum(feature_lens)[:-1])
        self.labels = np.split(labels, np.cumsum(label_lens)[:-1])
        self.teacher = None
        if teacher_path is not None:
            with np.load(teacher_path) as f:
                logits, frame_lens = f['logits'], f['frameLens']
            if len(frame_lens) != len(self.features):
                raise ValueError(f'{teacher_path}: {len(frame_lens)} trials, the data has {len(self.features)}')
            self.teacher = np.split(logits, np.cumsum(frame_lens)[:-1])

    def __len__(self):
        return len(self.features)


def load_partitions(data_dir: str, sessions: List[str], partition: str,
                    day_indices: Optional[List[int]] = None, required: bool = False,
                    teacher_dir: Optional[str] = None) -> List[SessionPartition]:
    """The partition of each session that has one, in session order;
    day_indices (default 0..n-1) gives each session's input layer."""
    day_indices = list(range(len(sessions))) if day_indices is None else day_indices
    parts = []
    for session, day_idx in zip(sessions, day_indices):
        path = os.path.join(data_dir, session, f'{partition}.npz')
        if not os.path.isfile(path):
            if required:
                raise FileNotFoundError(f'{path} not found; run AnalysisExamples/exportTFRecordsToNpz.py')
            print(f'  [skip] {session}: no {partition} partition ({path})')
            continue
        teacher_path = None
        if teacher_dir is not None:
            teacher_path = os.path.join(teacher_dir, session, f'{partition}.npz')
            if not os.path.isfile(teacher_path):
                raise FileNotFoundError(f'{teacher_path} not found; run AnalysisExamples/exportTeacherLogits.py')
        parts.append(SessionPartition(path, session, day_idx, teacher_path))
    if not parts:
        raise FileNotFoundError(f'No {partition} data for any of {sessions} under {data_dir}')
    return parts


def collate(part: SessionPartition, idx) -> dict:
    """Trials idx of one session -> a zero-padded batch."""
    feats = [part.features[i] for i in idx]
    labels = [part.labels[i] for i in idx]
    lengths = np.array([len(f) for f in feats])
    x = np.zeros((len(idx), lengths.max(), feats[0].shape[1]), dtype=np.float32)
    for b, f in enumerate(feats):
        x[b, :len(f)] = f
    batch = {
        'x': torch.from_numpy(x),
        'lengths': torch.from_numpy(lengths).long(),
        'targets': torch.from_numpy(np.concatenate(labels).astype(np.int64)),
        'target_lengths': torch.tensor([len(l) for l in labels], dtype=torch.long),
        'day_idx': part.day_idx,
        'trial_idx': list(idx),
    }
    if part.teacher is not None:
        frames = [part.teacher[i] for i in idx]
        teacher = np.zeros((len(idx), max(len(t) for t in frames), frames[0].shape[1]), dtype=np.float32)
        for b, t in enumerate(frames):
            teacher[b, :len(t)] = t
        batch['teacher'] = torch.from_numpy(teacher)
        batch['teacher_lengths'] = torch.tensor([len(t) for t in frames], dtype=torch.long)
    return batch


class SessionBatches:
    """Batches over every trial of the given partitions, each from one session.

    shuffle=True (training): trials are shuffled within each session, cut
    into batches, and the batches of all sessions interleaved in random
    order, so an epoch visits every training trial once."""

    def __init__(self, parts: List[SessionPartition], batch_size: int, shuffle: bool, seed: int = 0):
        self.parts, self.batch_size, self.shuffle = parts, batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def _plan(self):
        plan = []
        for p, part in enumerate(self.parts):
            order = self.rng.permutation(len(part)) if self.shuffle else np.arange(len(part))
            plan += [(p, order[i:i + self.batch_size]) for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            plan = [plan[i] for i in self.rng.permutation(len(plan))]
        return plan

    def __len__(self):
        return sum(int(np.ceil(len(p) / self.batch_size)) for p in self.parts)

    def __iter__(self):
        for p, idx in self._plan():
            yield collate(self.parts[p], idx)

    @property
    def n_trials(self):
        return sum(len(p) for p in self.parts)


def gauss_kernel(sd: float) -> np.ndarray:
    """The RNN's input-smoothing kernel (neuralSequenceDecoder.gaussSmooth)."""
    from scipy.ndimage import gaussian_filter1d
    inp = np.zeros(100, dtype=np.float32)
    inp[50] = 1
    kernel = gaussian_filter1d(inp, sd)
    kernel = kernel[kernel > 0.01]
    return kernel / kernel.sum()


def smooth_inputs(x: torch.Tensor, sd: float) -> torch.Tensor:
    """Gaussian smoothing over time of x (N, T, C), as the RNN applies to its
    input (smoothInputs/smoothKernelSD). The kernel is centred, so it is NOT
    causal: each bin sees about 3 * sd bins ahead. sd <= 0: unchanged."""
    if sd <= 0:
        return x
    kernel = torch.tensor(gauss_kernel(sd), dtype=x.dtype, device=x.device)
    C = x.shape[-1]
    weight = kernel.flip(0)[None, None, :].repeat(C, 1, 1)
    out = torch.nn.functional.conv1d(x.transpose(1, 2), weight, padding=len(kernel) // 2, groups=C)
    return out.transpose(1, 2)


def augment(x: torch.Tensor, white_noise_sd: float, constant_offset_sd: float) -> torch.Tensor:
    """The RNN's training-time augmentation: white noise on every bin plus a
    constant offset per trial and feature."""
    if white_noise_sd > 0:
        x = x + torch.randn_like(x) * white_noise_sd
    if constant_offset_sd > 0:
        x = x + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * constant_offset_sd
    return x
