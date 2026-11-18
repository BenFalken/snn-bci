"""
The three tasks: what the decoder's outputs mean, the training loss, and the reported metric.

    motor   2 outputs per 4 ms step, the scaled hand velocity. Loss and metric: RMSE in physical units (sqrt of the
            mean squared error over steps and both axes, after unscaling), as spike_bmi's train_snn.py
    audio   one output per spectrogram band. Loss: MSE. Metric: the mean Pearson correlation over bands, averaged
            in Fisher-z space (AuditoryDecoding's regression_metrics; bands without variance left out)
    speech  41 outputs (39 phonemes, silence, CTC blank last) per 80 ms. Loss: CTC. Metric: phoneme error rate,
            greedy (best-path) decoding, edit distance summed over trials / reference length summed
"""

import numpy as np
import torch
import torch.nn.functional as F

from .data import unscale_velocity


# ------------------------------------------------------------------------------------------------------ motor
def motor_loss(pred, target, scale, eps=0.0):
    return torch.sqrt(F.mse_loss(unscale_velocity(pred, *scale), unscale_velocity(target, *scale)) + eps)


# ------------------------------------------------------------------------------------------------------ audio
def mean_correlation(y: np.ndarray, p: np.ndarray) -> float:
    r = np.array([np.corrcoef(y[:, i], p[:, i])[0, 1] for i in range(y.shape[1])])
    z = np.arctanh(np.clip(r, -0.9999999, 0.9999999))
    return float(np.tanh(np.nanmean(z)))


# ----------------------------------------------------------------------------------------------------- speech
BLANK = 40


def ctc_loss(logits, out_lengths, labels, label_lengths):
    """logits (N, T, 41); labels concatenated (sum L,). Mean over trials of the per-trial loss."""
    lp = logits.log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(lp, labels, out_lengths, label_lengths, blank=BLANK, reduction='mean', zero_infinity=True)


def greedy_decode(logits: np.ndarray) -> np.ndarray:
    best = logits.argmax(-1)
    keep = np.ones(len(best), bool)
    keep[1:] = best[1:] != best[:-1]
    best = best[keep]
    return best[best != BLANK]


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def phoneme_error_rate(logits_list, labels_list) -> float:
    errs = sum(edit_distance(greedy_decode(l), y) for l, y in zip(logits_list, labels_list))
    return errs / max(sum(len(y) for y in labels_list), 1)
