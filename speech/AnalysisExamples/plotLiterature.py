#figures for the literature slides: DelayNet (Balafrej, Alibart & Rouat 2023) and e-prop / LSNN on TIMIT
#(Bellec et al., Nat Commun 2020), redrawn in the deck's style (black on white, Palatino).
#
#example usage:
#   python plotLiterature.py --outDir ../figures
#writes delaynet_memory.png, delaynet_unrolled.png, eprop_timit.png and gap_compare.png (and .svg)
#
#numbers: DelayNet Fig. 5 curves are read off the published figure (means only, error bars omitted);
#e-prop bars are the values printed in Bellec et al. Fig. 2c; LSTM references are those quoted in its
#Supplementary Fig. 4 (Graves & Schmidhuber 2005; Graves et al. 2013); our numbers are 36-epoch test PERs.
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

GRAY, LIGHT = '0.45', '0.75'

def style():
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 15, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': 'black', 'axes.linewidth': 1.0,
                         'xtick.color': 'black', 'ytick.color': 'black', 'svg.fonttype': 'none'})

def save(fig, out_dir, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=220, facecolor='white')
    plt.close(fig)
    print('wrote', name)

def delaynet_memory(out_dir):
    # DelayNet Fig. 5: the cue-accumulation task is solved by slow membrane decay alone, recurrence or not
    rec_slow = [0.555, 0.615, 0.662, 0.684, 0.714, 0.746, 0.770, 0.782, 0.808, 0.794, 0.820, 0.834, 0.832, 0.834,
                0.816, 0.816, 0.818, 0.812, 0.810, 0.816, 0.826, 0.832, 0.836, 0.844, 0.856]
    ff_slow = [0.555, 0.610, 0.664, 0.702, 0.722, 0.732, 0.750, 0.774, 0.792, 0.800, 0.818, 0.824, 0.832, 0.832,
               0.832, 0.828, 0.824, 0.826, 0.828, 0.822, 0.832, 0.832, 0.832, 0.830, 0.826]
    rec_fast = [0.452, 0.444, 0.434, 0.442, 0.442, 0.442, 0.436, 0.426, 0.428, 0.434, 0.432, 0.444, 0.452, 0.452,
                0.444, 0.444, 0.444, 0.450, 0.458, 0.444, 0.450, 0.458, 0.430, 0.440, 0.438]
    ep = np.arange(1, 26)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(ep, rec_slow, color='black', lw=2.4)
    ax.plot(ep, ff_slow, color=GRAY, lw=2.4, ls='--')
    ax.plot(ep, rec_fast, color=LIGHT, lw=2.4)
    ax.axhline(0.5, color='0.85', lw=1, zorder=0)
    ax.text(25.4, rec_slow[-1] + 0.012, 'recurrent, slow decay (2 s)', va='center', fontsize=14)
    ax.text(25.4, ff_slow[-1] - 0.016, 'no recurrence, slow decay (2 s)', va='center', fontsize=14, color=GRAY)
    ax.text(25.4, rec_fast[-1], 'recurrent, fast decay (20 ms)', va='center', fontsize=14, color='0.55')
    ax.text(1, 0.506, 'chance', fontsize=12, color='0.6')
    ax.set_xlim(1, 25)
    ax.set_ylim(0.38, 0.9)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy (cue accumulation)')
    fig.subplots_adjust(left=0.1, right=0.66, bottom=0.13, top=0.96)
    save(fig, out_dir, 'delaynet_memory')

def delaynet_unrolled(out_dir):
    # a recurrent network unrolled in time; a delayed synapse skips steps, like a residual connection
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis('off')
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-1.55, 2.6)
    xs = range(6)
    for x in xs:
        ax.add_patch(FancyBboxPatch((x - 0.32, -0.25), 0.64, 0.5, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc='white', ec='black', lw=1.5))
        ax.text(x, 0, ('t' if x == 3 else f"t {'+' if x > 3 else '−'} {abs(x - 3)}"), ha='center', va='center', fontsize=19,
                style='italic')
        ax.add_patch(FancyArrowPatch((x, -0.95), (x, -0.3), arrowstyle='-|>', mutation_scale=14, color=GRAY, lw=1.2))
        ax.text(x, -1.25, 'bin in', ha='center', va='top', fontsize=12, color=GRAY)
    for x in range(5):
        ax.add_patch(FancyArrowPatch((x + 0.33, 0), (x + 0.67, 0), arrowstyle='-|>', mutation_scale=16,
                                     color='black', lw=1.6))
    # delayed synapse: skips three steps
    ax.add_patch(FancyArrowPatch((1, 0.3), (4, 0.3), connectionstyle='arc3,rad=-0.45', arrowstyle='-|>',
                                 mutation_scale=18, color='black', lw=1.8, ls='--'))
    ax.text(2.5, 1.95, 'delayed synapse: a spike sent at t − 2 arrives at t + 1', ha='center', fontsize=15)
    ax.text(2.5, 1.65, 'the gradient takes the same shortcut back in training', ha='center', fontsize=13, color=GRAY)
    ax.text(5.55, 0.42, 'ordinary recurrence:\none step at a time', ha='center', fontsize=12, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    save(fig, out_dir, 'delaynet_unrolled')

def eprop_timit(out_dir):
    # Bellec et al. 2020 Fig. 2c, with the LSTM references from its Supplementary Fig. 4
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    labels = ['global\nsignal', 'e-prop', 'BPTT', 'LSTM\n(reference)']
    panels = [('framewise error (%)', [52, 34.6, 32.9, 26.9], '800 neurons, bidirectional\n61 classes per 10 ms frame'),
              ('phoneme error rate (%)', [60, 26.4, 24.7, 18.6], '3 layers × 800 neurons, CTC\n39 classes, no language model')]
    for ax, (yl, vals, sub) in zip(axs, panels):
        fcs = [LIGHT, 'black', 'white', 'white']
        hatches = [None, None, '///', None]
        for i, (v, fc, h) in enumerate(zip(vals, fcs, hatches)):
            ax.bar(i, v, width=0.62, color=fc, edgecolor='black', lw=1.3, hatch=h)
            ax.text(i, v + 1.2, f'{v:g}', ha='center', va='bottom', fontsize=14)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel(yl)
        ax.set_ylim(0, 68)
        ax.set_title(sub, fontsize=13, color=GRAY, pad=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.86, wspace=0.3)
    save(fig, out_dir, 'eprop_timit')

def gap_compare(out_dir):
    # spiking network against its non-spiking reference, both studies (PER, %); different data, so only the gap compares
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rows = [('TIMIT speech audio\n(Bellec et al. 2020)', [(18.6, 'LSTM'), (24.7, 'LSNN')]),
            ('Our speech BCI, same input\n(20 ms steps, 96 epochs)', [(21.1, 'GRU'), (24.5, 'our SNN')]),
            ('Our SNN against the paper\'s RNN\n(windowed, 5 × 1,024)', [(18.8, 'RNN'), (24.5, 'our SNN')])]
    for y, (lab, pts) in enumerate(rows[::-1]):
        (a, la), (b, lb) = pts
        ax.plot([a, b], [y, y], color='0.7', lw=3, zorder=0)
        ax.scatter([a], [y], s=130, facecolor='white', edgecolor='black', lw=1.6, zorder=2)
        ax.scatter([b], [y], s=130, color='black', zorder=2)
        ax.text(a - 0.35, y, la, ha='right', va='center', fontsize=13, color=GRAY)
        ax.text(b + 0.35, y, f'{lb}   +{b - a:.1f}', ha='left', va='center', fontsize=13)
        ax.text(12.6, y, lab, ha='right', va='center', fontsize=13)
    ax.set_xlim(13, 33)
    ax.set_xticks([15, 20, 25, 30])
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('phoneme error rate (%)')
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.17, top=0.97)
    save(fig, out_dir, 'gap_compare')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    style()
    delaynet_memory(a.outDir)
    delaynet_unrolled(a.outDir)
    eprop_timit(a.outDir)
    gap_compare(a.outDir)

if __name__ == '__main__':
    main()
