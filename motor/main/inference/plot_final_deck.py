"""
The final motor comparison in the slide deck's style (Palatino, black on white): velocity RMSE per session for
every decoder on both monkeys, and the measured energy per decoded sample.

    python plot_final_deck.py <results/final> <out_dir>

Reads <results/final>/<subject>/combined_metrics.json (per session, the deck's per-axis RMSE) and
efficiency_summary.json (energy per sample: RAPL on the laptop CPU, the chip's power monitor for Speck).
Writes motor_final.png/.svg.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

DECODERS = [('kf', 'Kalman\nfilter'), ('wf', 'Wiener\nfilter'), ('lstm', 'LSTM'), ('qrnn', 'QRNN'),
            ('snn', 'LSNN\n(CPU)'), ('speck', 'Speck SNN\n(chip)')]
RED = '#C0392B'
FACE = {'kf': '0.85', 'wf': '0.85', 'lstm': '0.6', 'qrnn': '0.6', 'snn': RED, 'speck': '0.15'}


def ax_floor(mean):
    """Height of a bar's mean label: inside the bar, near its base, clear of the session dots."""
    return mean * 0.06


def main(root, out_dir):
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 15, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={'width_ratios': [1, 1, 1.05]})
    x = np.arange(len(DECODERS))
    for ax, subj, title in zip(axes[:2], ('indy', 'loco'), ('a   Monkey I (indy), 36 sessions',
                                                            'b   Monkey L (loco), 9 sessions')):
        m = json.load(open(os.path.join(root, subj, 'combined_metrics.json')))
        sessions = sorted(m)
        vals = {k: np.array([m[s][k]['rmse'] for s in sessions]) for k, _ in DECODERS}
        rng = np.random.default_rng(0)
        for i, (k, _) in enumerate(DECODERS):
            v = vals[k]
            ax.bar(i, v.mean(), 0.66, color=FACE[k], edgecolor='black', hatch='//' if k == 'speck' else None, zorder=1)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color='white', edgecolor='black', lw=0.7, zorder=3)
            dark = k in ('snn', 'speck')
            ax.text(i, ax_floor(v.mean()), f'{v.mean():.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color='white' if dark else 'black',
                    bbox=dict(facecolor='black' if k == 'speck' else FACE[k], edgecolor='none', pad=2))
        p = wilcoxon(vals['snn'], vals['lstm']).pvalue
        print(f'{subj}: LSNN vs LSTM p = {p:.2g}; LSNN better in {np.sum(vals["snn"] < vals["lstm"])}/{len(sessions)}; '
              f'Speck vs LSTM p = {wilcoxon(vals["speck"], vals["lstm"]).pvalue:.2g}, Speck better in '
              f'{np.sum(vals["speck"] < vals["lstm"])}/{len(sessions)}')
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in DECODERS], fontsize=13)
        ax.set_ylabel('hand-velocity RMSE (test split), lower is better')
        ax.set_ylim(0, max(v.max() for v in vals.values()) * 1.08)
        ax.set_title(title, loc='left', fontsize=16, fontweight='bold')

    ax = axes[2]
    e = json.load(open(os.path.join(root, 'indy', 'efficiency_summary.json')))
    e = next(iter(e.values()))
    for i, (k, _) in enumerate(DECODERS):
        r = e[k]
        ax.bar(i, r['energy_j'] * 1e6, 0.66, color=FACE[k], edgecolor='black', hatch='//' if k == 'speck' else None)
        j = r['energy_j']
        lab = f"{j:.2f} J" if j >= 0.1 else f"{j * 1e3:.1f} mJ" if j >= 1e-4 else f"{j * 1e6:.1f} µJ"
        ax.text(i, r['energy_j'] * 1e6 * 1.6, lab, ha='center', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1, 2e7)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in DECODERS], fontsize=13)
    ax.set_ylabel('measured energy per decoded sample (µJ, log scale)')
    ax.set_title('c   Energy per sample (indy)', loc='left', fontsize=16, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'motor_final.{ext}'), dpi=200, facecolor='white', bbox_inches='tight')
    print('wrote motor_final')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
