"""
The local Indy recordings for the motor figures: each session's held-out segment as the decoders use it
(speck_per_session/data/dataset/bmi/indy/mua/<session>_binning.h5): multi-unit activity of 96 channels in 4 ms bins
(X_mua) and the hand's state (y_task: position x, y, velocity x, y, ...), about 80 s per session.

    sessions()                  session names in date order
    load(session)               (rates (T, 96), velocity (T, 2)) with rates smoothed by a 40 ms Gaussian
    date(session)               datetime.date of the session
"""

from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import datetime
import glob
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = WORKSPACE / "speck_per_session/data/dataset/bmi/indy/mua"
STEP_MS = 4.0


def sessions():
    return sorted(os.path.basename(f)[:-len("_binning.h5")] for f in glob.glob(str(ROOT / "*_binning.h5")))


def date(session):
    d = session.split("_")[1]
    return datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def load(session, smooth_ms=40.0):
    with h5py.File(ROOT / f"{session}_binning.h5", "r") as f:
        X = f["X_mua"][:].astype(np.float64)
        y = f["y_task"][:].astype(np.float64)
    if smooth_ms:
        X = gaussian_filter1d(X, smooth_ms / STEP_MS, axis=0)
    return X, y[:, 2:4]


def lead(X, V, lead_ms):
    """Rates at t paired with velocity at t + lead (the activity leads the movement it encodes)."""
    k = int(round(lead_ms / STEP_MS))
    return (X[:-k], V[k:]) if k > 0 else (X, V)
