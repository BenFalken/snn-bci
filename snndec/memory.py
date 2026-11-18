"""
How far back a trained decoder remembers: the silence response. At many random moments t0 of real test data, the
input at t0 alone is silenced (every channel set to 0), and the outputs that follow are compared with the unperturbed
run, at each lag after t0, in units of the outputs' SD:

    sensitivity  the mean absolute change. Includes knock-on effects: an integrate-and-fire neuron without leak keeps
                 any small change of its membrane until it fires, so later spikes shift and the change spreads
    kernel       the size of the mean signed change (RMS over outputs). Knock-on effects point every which way and
                 average out; what remains is the consistent effect of t0's input, the memory the decoder uses:
                 input delays bring t0 back at each delay, a leaky readout lets it fade
    floor        what the kernel reads from knock-on effects alone (their spread / sqrt(moments)): memory is the
                 kernel above it
"""

import numpy as np
import torch

from .model import SNNDecoder


@torch.no_grad()
def silence_response(model: SNNDecoder, segments: np.ndarray, burn_in: int, day=None) -> np.ndarray:
    """segments (K, burn_in + L, C): input steps, silenced at step burn_in. Returns {'sensitivity', 'kernel'}, each
    (n_lags,) at the output steps from t0 on (lag 0 = the output at or right after t0)."""
    model.eval()
    x = torch.as_tensor(segments, dtype=torch.float32)
    x0 = x.clone()
    x0[:, burn_in] = 0.0
    y, y0 = model(x, day), model(x0, day)
    first = (burn_in * model.substeps) // model.out_every                  # the first output that can see t0
    d = (y - y0)[:, first:]
    sd = y[:, first:].std().clamp(min=1e-8)
    K = d.shape[0]
    return {'sensitivity': (d.abs().mean((0, 2)) / sd).numpy(),
            'kernel': (d.mean(0).pow(2).mean(-1).sqrt() / sd).numpy(),
            # what the kernel would read if the changes had no consistent direction: their spread / sqrt(K)
            'floor': (d.std(0).pow(2).mean(-1).sqrt() / (K ** 0.5) / sd).numpy()}


def sample_segments(recording: np.ndarray, n: int, burn_in: int, length: int, seed: int = 0) -> np.ndarray:
    """n random (burn_in + length)-step stretches of a (T, C) recording."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(recording) - burn_in - length, n)
    return np.stack([recording[s:s + burn_in + length] for s in starts])
