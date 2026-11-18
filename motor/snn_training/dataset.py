"""
Dataset and DataLoaders for the SNN's .pkl trials.

A dataset directory has train/ and test/ subfolders of trials named 0.pkl,
1.pkl, ... (or {session}_{i}.pkl in a pooled directory), each a dict with
    input_spikes  (n_units, T)  spike counts per 4 ms bin
    velocity      (T, 2)        hand velocity in physical units

bmi trials of one split share one length. hkm trials are whole reaches of
different lengths (nwb_conversion/make_snn_dataset_whole_trial.py), so they
can only be loaded one per batch (--batch-size 1).

With binarize=True (train_snn.py's default) spike counts are clipped to 0/1.

Each split's trials are read from disk once, when the dataset is built, and
kept in memory (spikes as small integers, the scaled velocity as float32);
every epoch then reads memory rather than one .pkl per trial from scratch.
Items are the same tensors either way. SNN_CACHE_TRIALS=0 reads from disk
at every access instead (the earlier behaviour).

Velocity is scaled to [margin, 1 - margin] using per-subject bounds from
velocity_scalers.json (written by preprocessing_training/
compute_velocity_scalers.py; override the path with
BMI_VELOCITY_SCALERS_PATH). The experiment and subject are read from the
dataset path, which must contain .../{experiment}/{subject}/... .
"""

import json
import os
import pickle as pkl
import re
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:                                    # NumPy 2 pickles under NumPy 1 too (utils/pickle_compat.py)
    from utils.pickle_compat import load_pickle
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.pickle_compat import load_pickle

BMI_VELOCITY_SCALERS_PATH = os.environ.get(
    "BMI_VELOCITY_SCALERS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_scalers.json"),
)

CACHE_TRIALS = os.environ.get("SNN_CACHE_TRIALS", "1") != "0"

_velocity_scalers_cache = None


def _load_velocity_scalers():
    """{experiment: {subject: {v_lo, v_hi, margin, ...}}}, read once per process."""
    global _velocity_scalers_cache
    if _velocity_scalers_cache is None:
        if not os.path.isfile(BMI_VELOCITY_SCALERS_PATH):
            raise FileNotFoundError(
                f"{BMI_VELOCITY_SCALERS_PATH} does not exist; run compute_velocity_scalers.py "
                f"or set BMI_VELOCITY_SCALERS_PATH.")
        with open(BMI_VELOCITY_SCALERS_PATH, "r") as f:
            _velocity_scalers_cache = json.load(f)
    return _velocity_scalers_cache


def get_velocity_scalers(experiment: str, subject: str) -> Tuple[float, float, float]:
    """(v_lo, v_hi, margin) for one experiment/subject."""
    scalers = _load_velocity_scalers()
    if experiment not in scalers or subject not in scalers[experiment]:
        available = {exp: list(subs) for exp, subs in scalers.items()}
        raise KeyError(f"No velocity scalers for {experiment}/{subject} in "
                       f"{BMI_VELOCITY_SCALERS_PATH}; available: {available}")
    entry = scalers[experiment][subject]
    return entry["v_lo"], entry["v_hi"], entry["margin"]


def experiment_and_subject_from_path(path: str) -> Tuple[str, str]:
    """'.../bmi/indy/mua_8_group/indy_20160407_02' -> ('bmi', 'indy').

    Finds the first path component that is an experiment in
    velocity_scalers.json and is followed by one of its subjects."""
    scalers = _load_velocity_scalers()
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts[:-1]):
        if part in scalers and parts[i + 1] in scalers[part]:
            return part, parts[i + 1]
    raise ValueError(f"No '.../{{experiment}}/{{subject}}/...' segment in {path!r} matching "
                     f"velocity_scalers.json (experiments: {list(scalers)})")


class CustomDataset(Dataset):
    """One split (train/ or test/) of a dataset directory.

    Items are (label, spikes (T, n_units), scaled velocity (T, 2)); label is
    an unused NaN placeholder kept for the training loop's signature.
    experiment/subject default to the values derived from `path`."""

    def __init__(self, path: str, train: bool = True,
                 experiment: Optional[str] = None, subject: Optional[str] = None,
                 binarize: bool = False):
        super().__init__()
        self.binarize = binarize
        self.split_dir = os.path.join(path, 'train' if train else 'test')
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        files = [f for f in os.listdir(self.split_dir) if f.endswith('.pkl')]
        self.files = sorted(files, key=_natural_key)

        if experiment is None or subject is None:
            derived_experiment, derived_subject = experiment_and_subject_from_path(path)
            experiment = experiment or derived_experiment
            subject = subject or derived_subject
        self.experiment, self.subject = experiment, subject
        self.v_lo, self.v_hi, self.v_margin = get_velocity_scalers(experiment, subject)
        self.cache = [self._read(f) for f in self.files] if CACHE_TRIALS else None
        print(f"Loaded {'train' if train else 'test'} split: {len(self.files)} trials from "
              f"{self.split_dir} ({experiment}/{subject}, v_lo={self.v_lo:.2f}, "
              f"v_hi={self.v_hi:.2f}, margin={self.v_margin})"
              + (f", in memory ({self._cache_mb():.0f} MB)" if self.cache is not None else ""))

    def _read(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """One trial from disk: (spikes (T, n_units), scaled velocity (T, 2)).
        Spikes are kept as the smallest integer type that holds their counts
        exactly (counts are whole numbers), and become float in __getitem__."""
        with open(os.path.join(self.split_dir, name), 'rb') as f:
            data = load_pickle(f)
        counts = data['input_spikes'].T
        if np.issubdtype(counts.dtype, np.integer) or np.array_equal(counts, np.round(counts)):
            hi = counts.max() if counts.size else 0
            lo = counts.min() if counts.size else 0
            dtype = np.uint8 if 0 <= lo and hi <= 255 else np.int32
            spikes = torch.from_numpy(np.ascontiguousarray(counts.astype(dtype)))
        else:                                             # not counts: keep the values exactly
            spikes = torch.from_numpy(np.ascontiguousarray(counts)).float()
        # Inverted by train_snn.unscale_velocity().
        velocity = torch.from_numpy(
            self.v_margin + (1 - 2 * self.v_margin) * (data['velocity'] - self.v_lo) / (self.v_hi - self.v_lo)
        ).float()
        return spikes, velocity

    def _cache_mb(self) -> float:
        return sum(s.element_size() * s.nelement() + v.element_size() * v.nelement()
                   for s, v in self.cache) / 2 ** 20

    def __len__(self) -> int:
        return len(self.files)

    def trial_length(self, idx: int) -> int:
        """Timesteps of trial idx."""
        if self.cache is not None:
            return self.cache[idx][0].shape[0]
        with open(os.path.join(self.split_dir, self.files[idx]), 'rb') as f:
            return load_pickle(f)['input_spikes'].shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spikes, velocity = self.cache[idx] if self.cache is not None else self._read(self.files[idx])
        spikes = spikes.float()
        if self.binarize:
            spikes = (spikes > 0).float()
        return torch.tensor([float('nan')]), spikes, velocity.clone()


def _natural_key(name: str):
    """'s_10.pkl' sorts after 's_9.pkl'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', name)]


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels, spikes, velocity = zip(*batch)
    lengths = sorted({s.shape[0] for s in spikes})
    if len(lengths) > 1:
        raise ValueError(f"Cannot batch trials of different lengths ({lengths[:5]}...); "
                         f"whole-trial datasets (hkm) need --batch-size 1")
    return torch.stack(labels), torch.stack(spikes), torch.stack(velocity)


def create_dataloaders(
    data_path: str,
    batch_size: int = 20,
    num_workers: int = 4,
    shuffle_train: bool = True,
    drop_last: bool = True,
    experiment: Optional[str] = None,
    subject: Optional[str] = None,
    binarize: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """(train_loader, test_loader) for one dataset directory; both splits use
    the same velocity scalers and input binarization."""
    train_dataset = CustomDataset(data_path, train=True, experiment=experiment, subject=subject,
                                  binarize=binarize)
    test_dataset = CustomDataset(data_path, train=False, experiment=experiment, subject=subject,
                                 binarize=binarize)
    common = dict(batch_size=batch_size, num_workers=num_workers, drop_last=drop_last,
                  pin_memory=True, collate_fn=collate_fn)
    return (DataLoader(train_dataset, shuffle=shuffle_train, **common),
            DataLoader(test_dataset, shuffle=False, **common))
