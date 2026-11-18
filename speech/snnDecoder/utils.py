"""Training helpers: early stopping, checkpoints, training curves, and CTC
greedy decoding / phoneme error rate."""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Stop after `patience` epochs without improvement (mode 'min' or 'max')."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.mode == 'min':
            score = -score
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float,
                    checkpoint_path: str, additional_info: Optional[Dict] = None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if additional_info:
        checkpoint.update(additional_info)
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: torch.device = torch.device('cpu')) -> Dict:
    """Restore model weights (skipping neuron state buffers, see
    model.load_model_weights) and, if given, the optimizer state."""
    from .model import load_model_weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model_weights(model, checkpoint['model_state_dict'], source_description=checkpoint_path)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def plot_training_curves(train_losses: List[float], test_losses: List[float], save_path: str,
                         title: str = "Training Progress", lr_history: Optional[List[float]] = None,
                         test_per: Optional[List[float]] = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = 1 + bool(test_per) + bool(lr_history)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True, squeeze=False)
    axes = list(axes[:, 0])
    epochs = range(1, len(train_losses) + 1)

    ax = axes.pop(0)
    ax.plot(epochs, train_losses, 'b-', label='Train CTC loss')
    ax.plot(epochs, test_losses, 'r-', label='Test CTC loss')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if test_per:
        ax = axes.pop(0)
        ax.plot(epochs, test_per, 'k-')
        best = int(np.argmin(test_per))
        ax.scatter(best + 1, test_per[best], color='green', s=60, zorder=5,
                   label=f'Best test PER {test_per[best]:.4f}')
        ax.set_ylabel('Phoneme error rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if lr_history:
        ax = axes.pop(0)
        ax.plot(epochs, lr_history, 'g-', linewidth=2)
        ax.set_ylabel('Learning rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def greedy_decode(logits: np.ndarray, blank: int) -> np.ndarray:
    """Best-path CTC decoding of one sequence's logits (T, K): argmax per
    frame, merge repeats, drop blanks. Streaming-compatible: each emitted
    class depends only on frames up to it."""
    best = np.argmax(logits, axis=-1)
    keep = np.ones(len(best), dtype=bool)
    keep[1:] = best[1:] != best[:-1]
    best = best[keep]
    return best[best != blank]


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]
