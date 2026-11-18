#figure: the final decoder, a two-layer recurrent LSNN -- topology, and one neuron with and without threshold
#adaptation -- in the deck's style (black on white, Palatino), as PNG and SVG:
#   python drawModel.py --outDir ../figures
#The neuron in panel B is simulated with the model's own update (rsnn.SpikingLayer): unit-gain synaptic filter
#i <- a i + (1 - a) x, integrate-and-fire without leak v <- v + i, threshold theta = 1 + 1.8 b, at most 3 spikes per
#step, soft reset v <- v - n theta, adaptation b <- a_b b + (1 - a_b) n with a 500 ms (25-step) time constant.
import argparse
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch

INK, GRAY, LIGHT, RED = '0.1', '0.45', '0.88', '#C0392B'
TAU_SYN, TAU_ADAPT, ADAPT_SCALE, MAX_SPIKES = 3.0, 25.0, 1.8, 3


def style():
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 14, 'svg.fonttype': 'none'})


def capsule(ax, cx, cy, h, count, name, sub, w=3.2, edge=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={w / 2}',
                                fc='white', ec=edge, lw=1.6, zorder=3))
    n = 3
    for i in range(n):
        for y in (cy + h / 2 - 2.2 - i * 2.4, cy - h / 2 + 2.2 + i * 2.4):
            ax.add_patch(Circle((cx, y), 0.75, fc='white', ec=edge, lw=1.2, zorder=4))
    for dy in (-1.0, 0, 1.0):                                          #a vertical ellipsis (no glyph in Palatino)
        ax.add_patch(Circle((cx, cy + dy), 0.18, fc=GRAY, ec='none', zorder=4))
    ax.text(cx, cy + h / 2 + 1.6, count, ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.text(cx, cy - h / 2 - 1.6, name, ha='center', va='top', fontsize=15)
    ax.text(cx, cy - h / 2 - 4.4, sub, ha='center', va='top', fontsize=12.5, color=GRAY, linespacing=1.3)


def fan(ax, x1, h1, x2, h2, cy, w=3.2):
    ax.add_patch(Polygon([(x1 + w / 2, cy + h1 / 2 - 1.5), (x2 - w / 2, cy + h2 / 2 - 1.5),
                          (x2 - w / 2, cy - h2 / 2 + 1.5), (x1 + w / 2, cy - h1 / 2 + 1.5)],
                         fc=LIGHT, ec='none', alpha=0.6, zorder=1))


def box(ax, x, cy, w, lines, sub=None):
    ax.add_patch(FancyBboxPatch((x, cy - 4.2), w, 8.4, boxstyle='round,pad=0,rounding_size=0.8', fc='white', ec=INK,
                                lw=1.4, zorder=3))
    ax.text(x + w / 2, cy + (1.2 if sub else 0), lines, ha='center', va='center', fontsize=14, linespacing=1.25)
    if sub:
        ax.text(x + w / 2, cy - 2.6, sub, ha='center', va='center', fontsize=12, color=GRAY)


def arrow(ax, x1, y1, x2, y2, **kw):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16, lw=1.5, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=5, **kw))


def panel_a(ax):
    ax.set_xlim(0, 166)
    ax.set_ylim(-14, 34)
    ax.axis('off')
    ax.text(0, 33, 'A', fontsize=22, fontweight='bold', va='top')
    ax.text(4, 33, 'Network', fontsize=18, va='top')
    cy = 12
    cols = {'in': 10, 'h1': 38, 'h2': 62, 'out': 86}
    hs = {'in': 22, 'h1': 28, 'h2': 28, 'out': 18}
    fan(ax, cols['in'], hs['in'], cols['h1'], hs['h1'], cy)
    fan(ax, cols['h1'], hs['h1'], cols['h2'], hs['h2'], cy)
    fan(ax, cols['h2'], hs['h2'], cols['out'], hs['out'], cy)
    capsule(ax, cols['in'], cy, hs['in'], '256', 'Input', 'one 20 ms bin per step\ngain per channel and day')
    for k, name in (('h1', 'Hidden 1'), ('h2', 'Hidden 2')):
        capsule(ax, cols[k], cy, hs[k], '1,024', name, 'recurrent\n90% adaptive', edge=RED)
        x, top = cols[k] + 1.6, cy + hs[k] / 2 - 3
        ax.add_patch(FancyArrowPatch((x, top), (x, top - 5.5), connectionstyle='arc3,rad=-1.6', arrowstyle='-|>',
                                     mutation_scale=14, lw=1.4, color=INK, zorder=5))
    capsule(ax, cols['out'], cy, hs['out'], '41', 'Leaky outputs', 'never spike')
    #the outputs read both hidden layers: hidden 1's spikes reach them over hidden 2
    ax.add_patch(FancyArrowPatch((cols['h1'] + 1.8, cy + hs['h1'] / 2 - 2), (cols['out'] - 1.8, cy + hs['out'] / 2 - 2),
                                 connectionstyle='arc3,rad=-0.28', arrowstyle='-|>', mutation_scale=14, lw=1.4,
                                 color=INK, zorder=5))
    ax.text(cols['h2'] + 9, cy + 17.2, 'both layers\nread out', ha='left', va='bottom', fontsize=12, color=GRAY, linespacing=1.2)
    ox = cols['out'] + 1.6
    arrow(ax, ox, cy, ox + 9.5, cy)
    ax.text(ox + 4.75, cy + 1.2, 'every', ha='center', va='bottom', fontsize=12, color=GRAY)
    ax.text(ox + 4.75, cy - 1.2, '80 ms', ha='center', va='top', fontsize=12, fontweight='bold')
    bx = ox + 10
    box(ax, bx, cy, 20, 'phoneme\nprobabilities', None)
    arrow(ax, bx + 20, cy, bx + 23.5, cy)
    box(ax, bx + 23.5, cy, 23, '3-gram\nlanguage model', None)
    arrow(ax, bx + 46.5, cy, bx + 50, cy)
    box(ax, bx + 50, cy, 15, 'decoded\nsentence', None)


def simulate(T=50, adaptive=True):
    """One neuron driven by two identical 160 ms input bursts, starting 360 ms apart."""
    drive = [2.5 if (4 <= t < 12 or 22 <= t < 30) else 0.0 for t in range(T)]
    a_syn, a_ad = math.exp(-1 / TAU_SYN), math.exp(-1 / TAU_ADAPT)
    i = v = b = 0.0
    cur, thr, spikes = [], [], []
    for x in drive:
        i = a_syn * i + (1 - a_syn) * x
        v += i
        th = 1.0 + (ADAPT_SCALE * b if adaptive else 0.0)
        n = min(MAX_SPIKES, int(v // th)) if v >= th else 0
        v -= n * th
        if adaptive:
            b = a_ad * b + (1 - a_ad) * n
        cur.append(i)
        thr.append(th)
        spikes.append(n)
    return drive, cur, thr, spikes


def panel_b(fig, gs):
    T = 50
    drive, cur, _, s_iaf = simulate(T, adaptive=False)
    _, _, thr, s_ad = simulate(T, adaptive=True)
    t = [k * 20 for k in range(T)]                                   # ms
    axs = [fig.add_subplot(gs[k]) for k in range(4)]
    for ax in axs:
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, T * 20)
        ax.tick_params(labelsize=12)
    axs[0].text(-0.105, 1.25, 'B', transform=axs[0].transAxes, fontsize=22, fontweight='bold', va='top')
    axs[0].text(-0.07, 1.25, 'One neuron, the same input twice: the adaptive neuron answers the repeat with fewer spikes', transform=axs[0].transAxes,
                fontsize=18, va='top')
    axs[0].step(t, cur, where='post', color=INK, lw=1.8)
    axs[0].set_ylabel('synaptic\ncurrent', fontsize=13)
    for k, (ax, sp, col, lab) in enumerate(((axs[1], s_iaf, GRAY, 'plain IAF\nspikes'),
                                           (axs[3], s_ad, RED, 'adaptive\nspikes'))):
        ax.bar([x + 10 for x in t], sp, width=14, color=col)
        ax.set_ylim(0, MAX_SPIKES + 0.5)
        ax.set_yticks([0, MAX_SPIKES])
        ax.set_ylabel(lab, fontsize=13)
        first, second = sum(sp[:22]), sum(sp[22:])                  #the second burst starts at step 22
        ax.text(1.01, 0.5, f'{first} then {second}\nspikes', transform=ax.transAxes, fontsize=13, va='center',
                color=col if col == RED else INK)
    axs[2].step(t, thr, where='post', color=RED, lw=2.0)
    axs[2].axhline(1.0, color=GRAY, ls='--', lw=1.2)
    axs[2].text(T * 20 * 0.995, 1.04, 'plain IAF threshold', ha='right', va='bottom', fontsize=12, color=GRAY)
    axs[2].set_ylabel('adaptive\nthreshold', fontsize=13)
    axs[2].text(1.01, 0.5, 'rises with each\nspike, recovers\nover 500 ms', transform=axs[2].transAxes, fontsize=13,
                va='center', color=RED)
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[-1].set_xlabel('time (ms; one step = 20 ms)', fontsize=13)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    style()
    fig = plt.figure(figsize=(16, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.12, left=0.1, right=0.86, top=0.98, bottom=0.07)
    panel_a(fig.add_subplot(outer[0]))
    panel_b(fig, outer[1].subgridspec(4, 1, hspace=0.35, height_ratios=[1, 0.8, 1, 0.8]))
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(a.outDir, f'model_final_figure.{ext}'), dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote model_final_figure.png/.svg')


if __name__ == '__main__':
    main()
