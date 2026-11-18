"""
Rank input channels by how much a trained native SNN uses them, and write channel lists for
train.py --input-channel-file.

A channel's weight: the L2 norm of the first layer's weights from it, over every neuron and input delay,
times the RMS over days of its per-day input gain. With z-scored inputs (unit variance per channel), this
is a fair measure of how strongly the channel drives the network.

    python rank_channels.py CKPT --sizes 96 64 32 --out-dir configs/channels [--random-seed 0]

CKPT: a native run's best_model_weights.pth (e.g. the Speck screen's continuous SP0 run). Writes
top<K>.txt for each size, and random<K>.txt (a control drawn from the same channels) when --random-seed is
given. Indices are columns of the data file, as --input-channels uses.
"""

import argparse
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--sizes', type=int, nargs='+', default=[96, 64, 32])
    p.add_argument('--out-dir', default='configs/channels')
    p.add_argument('--random-seed', type=int, default=None)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd, args = ck['model_state_dict'], ck.get('args', {})
    if args.get('input_channel_list'):
        cols = np.array(args['input_channel_list'])
    elif args.get('input_channels'):
        cols = np.arange(*args['input_channels'])
    else:
        cols = np.arange(sd['day_gain'].shape[1])
    C = len(cols)
    w = sd['layers.0.ff.weight'].float()                        # (width, C * n_delays): delay-major
    n_delays = w.shape[1] // C
    w = w.reshape(w.shape[0], n_delays, C)
    score = w.pow(2).sum((0, 1)).sqrt()
    if 'day_gain' in sd:
        score = score * sd['day_gain'].float().pow(2).mean(0).sqrt()
    score = score.numpy()
    order = np.argsort(-score)

    os.makedirs(a.out_dir, exist_ok=True)
    print(f'{a.ckpt}: {C} channels x {n_delays} delays; weight spread: top 10% {score[order[:C // 10]].mean():.3f}, '
          f'median {np.median(score):.3f}, bottom 10% {score[order[-C // 10:]].mean():.3f}')
    for k in a.sizes:
        top = np.sort(cols[order[:k]])
        np.savetxt(os.path.join(a.out_dir, f'top{k}.txt'), top[None], fmt='%d')
        half = int((top < cols[0] + C // 2).sum())
        print(f'top{k}: {half} from the first array (columns {cols[0]}-{cols[0] + C // 2 - 1}), {k - half} from the second; '
              f'{100 * score[order[:k]].sum() / score.sum():.0f}% of the total weight')
        if a.random_seed is not None:
            rnd = np.sort(np.random.default_rng(a.random_seed).choice(cols, k, replace=False))
            np.savetxt(os.path.join(a.out_dir, f'random{k}.txt'), rnd[None], fmt='%d')
    np.savetxt(os.path.join(a.out_dir, 'scores.txt'), np.stack([cols, score], 1), fmt=['%d', '%.5f'])


if __name__ == '__main__':
    main()
