"""How learned synaptic delays ended up (--learn-delays runs; snnDecoder/delays.py).

For every group of delays (input synapses, recurrent synapses per layer, synapses between layers): where the
delays started (fixed random, uniform over 0..D steps) against where training left them, how far they moved,
whether strong synapses prefer particular delays, and, for the input, threshold crossings against spike-band
power. With the run's log, also how the mean delay of each group moved epoch by epoch.

example usage (on Oscar):
    python delay_report.py --model $BASE_DIR/derived/snns/learndelay36/*_LD_modelA/final_model.pth \
        --log snn_logs/SNN_RUN_1234_0.out --out $BASE_DIR/derived/delay_report/LD_modelA
writes <out>.md (tables), <out>.png and <out>.svg (figure)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from snnDecoder.delays import delay_modules            # noqa: E402
from snnDecoder.inference import load_snn_model         # noqa: E402

STEP_MS = 20


def synapse_weights(model, name):
    """|weight| of each synapse, in the delay group's shape."""
    if name == 'in_delay':
        return model.layers[0].ff.weight.detach().abs()
    if name == 'in_extra_delay':
        return model.in_extra.detach().abs()
    layer = model.layers[int(name.split('.')[1])]
    return (layer.rec.weight if name.endswith('rec_delay') else layer.ff.weight).detach().abs()


def label(name):
    if name in ('in_delay', 'in_extra_delay'):
        return 'input → layer 1'
    li = int(name.split('.')[1]) + 1
    return f'layer {li} recurrent' if name.endswith('rec_delay') else f'layer {li - 1} → {li}'


def groups(model):
    """Merge the input's first and extra synapses into one group: -> {label: (D, start, final, |w|, name)}"""
    out = {}
    for name, m in delay_modules(model):
        lab = label(name)
        start, final = m.pos_init.long().flatten(), m.delays().flatten()
        w = synapse_weights(model, name).flatten()
        if lab in out:
            D, s0, f0, w0, _ = out[lab]
            out[lab] = (D, torch.cat([s0, start]), torch.cat([f0, final]), torch.cat([w0, w]), 'in')
        else:
            out[lab] = (m.D, start, final, w, name)
    return out


def input_by_feature(model):
    """Mean learned input delay (ms) for threshold crossings (channels 0-127) and spike-band power (128-255),
    over all synapses and weighted by |weight|."""
    rows = []
    for name in ('in_delay', 'in_extra_delay'):
        m = dict(delay_modules(model)).get(name)
        if m is None:
            continue
        d, w = m.delays().float(), synapse_weights(model, name)
        rows.append((d, w))
    if not rows:
        return None
    d = torch.cat([r[0].reshape(-1, r[0].shape[-1]) for r in rows])     # (synapses per channel, channels)
    w = torch.cat([r[1].reshape(-1, r[1].shape[-1]) for r in rows])
    half = d.shape[1] // 2
    res = {}
    for lab, sl in (('threshold crossings', slice(0, half)), ('spike-band power', slice(half, None))):
        dd, ww = d[:, sl], w[:, sl]
        res[lab] = (float(dd.mean()) * STEP_MS, float((dd * ww).sum() / ww.sum()) * STEP_MS)
    return res


def parse_log(path):
    """Per epoch: {group: mean delay ms} from the '  learned delays:' lines."""
    if not path:
        return []
    epochs = []
    for line in open(path):
        if line.strip().startswith('learned delays:'):
            epochs.append({k: float(v) for k, v in re.findall(r'(\S+): (\d+) ms', line)})
    return epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='checkpoint of a --learn-delays run (a glob is fine)')
    p.add_argument('--log', default=None, help="the run's training log, for the epoch-by-epoch trend")
    p.add_argument('--out', default='delay_report')
    a = p.parse_args()
    path = sorted(glob.glob(a.model))[-1]
    model, _ = load_snn_model(path, 'cpu')
    G = groups(model)
    if not G:
        sys.exit('no learned delays in this model (was it trained with --learn-delays true?)')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    L = [f'# Learned delays: {path}', '',
         '| Group | Range | Synapses | Mean start → end (ms) | Moved (ms, mean) | Unchanged | '
         'At 0 / at max | Weighted by \\|w\\| (ms) |',
         '|---|---|---|---|---|---|---|---|']
    for lab, (D, s, f, w, _) in G.items():
        moved = (f - s).abs().float()
        wmean = float((f.float() * w).sum() / w.sum()) * STEP_MS
        L.append(f'| {lab} | 0–{D * STEP_MS} ms | {s.numel():,} | {s.float().mean() * STEP_MS:.0f} → '
                 f'{f.float().mean() * STEP_MS:.0f} | {moved.mean() * STEP_MS:.1f} | '
                 f'{float((moved == 0).float().mean()) * 100:.0f}% | {float((f == 0).float().mean()) * 100:.0f}% / '
                 f'{float((f == D).float().mean()) * 100:.0f}% | {wmean:.0f} |')
    feat = input_by_feature(model)
    if feat:
        L += ['', '| Input feature | Mean delay (ms) | Weighted by \\|w\\| (ms) |', '|---|---|---|']
        L += [f'| {k} | {v[0]:.0f} | {v[1]:.0f} |' for k, v in feat.items()]
    L += ['', 'Start: the fixed random delays (uniform over the range) the run began from. "Weighted by |w|": the '
          'mean delay counting each synapse by its weight, which shows where the strong synapses sit.']
    open(a.out + '.md', 'w').write('\n'.join(L) + '\n')

    trend = parse_log(a.log)
    n = len(G) + (1 if trend else 0)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, (lab, (D, s, f, w, _)) in zip(axs.flat, G.items()):
        bins = np.arange(D + 2) - 0.5
        ax.hist(s.numpy(), bins=bins, color='0.8', label='start (random)')
        ax.hist(f.numpy(), bins=bins, histtype='step', color='black', lw=1.6, label='learned')
        ax.set_title(lab, fontsize=12)
        ax.set_xlabel('delay (steps of 20 ms)')
        ax.set_ylabel('synapses')
    axs.flat[0].legend(frameon=False, fontsize=10)
    if trend:
        ax = axs.flat[len(G)]
        for k in trend[0]:
            ax.plot(range(1, len(trend) + 1), [e.get(k, np.nan) for e in trend], lw=1.6, label=k)
        ax.set_xlabel('epoch')
        ax.set_ylabel('mean delay (ms)')
        ax.set_title('mean delay during training', fontsize=12)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(f'{a.out}.{ext}', dpi=200, facecolor='white')
    print('\n'.join(L))
    print(f'wrote {a.out}.md, .png, .svg')


if __name__ == '__main__':
    main()
