"""
Ranking input channels by how much a trained network uses them: the L2 norm of the first layer's weights from each
channel, over every neuron and input delay, times the RMS over days of the channel's day gain (when there is one).
With inputs of comparable scale, this is how strongly the channel drives the network. The same rule for every
experiment, so the channel/delay sweeps reduce channels the same way.
"""

from typing import List

import torch

from .model import SNNDecoder


def rank_channels(model: SNNDecoder) -> List[int]:
    """The data's channel indices, most used first."""
    host = model.host
    C, D = host.n_in, len(host.delays)
    w = model.layers[0].ff.weight.detach().float().reshape(-1, D, C)        # delay-major, as HostInput.delay
    score = w.pow(2).sum((0, 1)).sqrt()
    if host.input_scale is not None:                                        # weights act on scaled inputs
        score = score * host.input_scale.abs()
    if host.day_gain is not None:
        score = score * host.day_gain.detach().pow(2).mean(0).sqrt()
    cols = host.channels if host.channels is not None else list(range(C))
    return [int(cols[i]) for i in torch.argsort(score, descending=True)]
