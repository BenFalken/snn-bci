#figure: the two motor (hand-velocity) decoders of spike_bmi side by side, in the deck's style (black on white,
#Palatino), as PNG and SVG:
#   python drawMotorModels.py --outDir ../figures
#A: the earlier SNN (models/model_bmi.py SNN_Speck, snn_config.sh 'pooled'): feedforward IAF 512 -> 256 -> 128, a
#synaptic filter per layer (initialized at 8, 16, 32, 64 ms, trained), 36 spiking outputs (18 per axis) smoothed by a
#two-stage EMA and decoded as a population vector. B: the LSNN ported from the speech decoder (models/model_bmi_lsnn.py,
#snn_config.sh 'lsnn'): two recurrent layers of 256, 90% adaptive, per-neuron synaptic filters 20 ms-1.3 s, two leaky
#outputs reading both layers through two leaky stages. Parameter counts are the models' own (no biases in the SNN).
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from drawModel import INK, GRAY, RED, style, capsule, fan, box, arrow


def recurrent_loop(ax, cx, cy, h):
    x, top = cx + 1.6, cy + h / 2 - 3
    ax.add_patch(FancyArrowPatch((x, top), (x, top - 5.5), connectionstyle='arc3,rad=-1.6', arrowstyle='-|>',
                                 mutation_scale=14, lw=1.4, color=INK, zorder=5))


def pair(ax, cx, cy, count, name, sub, w=3.2):
    """A small layer drawn as its two neurons."""
    from matplotlib.patches import FancyBboxPatch, Circle
    h = 7
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={w / 2}',
                                fc='white', ec=INK, lw=1.6, zorder=3))
    for y in (cy + 1.3, cy - 1.3):
        ax.add_patch(Circle((cx, y), 0.75, fc='white', ec=INK, lw=1.2, zorder=4))
    ax.text(cx, cy + h / 2 + 1.6, count, ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.text(cx, cy - h / 2 - 1.6, name, ha='center', va='top', fontsize=15)
    ax.text(cx, cy - h / 2 - 4.4, sub, ha='center', va='top', fontsize=12.5, color=GRAY, linespacing=1.3)


def tail(ax, ox, cy, stages, decode):
    """Every 4 ms: the output stages, then the velocity."""
    arrow(ax, ox, cy, ox + 9.5, cy)
    ax.text(ox + 4.75, cy + 1.2, 'every', ha='center', va='bottom', fontsize=12, color=GRAY)
    ax.text(ox + 4.75, cy - 1.2, '4 ms', ha='center', va='top', fontsize=12, fontweight='bold')
    bx = ox + 10
    box(ax, bx, cy, 22, stages, None)
    arrow(ax, bx + 22, cy, bx + 25.5, cy)
    if decode:
        box(ax, bx + 25.5, cy, 22, decode, None)
        bx += 25.5
    arrow(ax, bx + 22, cy, bx + 25.5, cy)
    box(ax, bx + 25.5, cy, 17, 'hand\nvelocity', 'x, y')


def panel(ax, letter, title):
    ax.set_xlim(0, 166)
    ax.set_ylim(-16, 34)
    ax.axis('off')
    ax.text(0, 33, letter, fontsize=22, fontweight='bold', va='top')
    ax.text(4, 33, title, fontsize=18, va='top')


def panel_snn(ax):
    panel(ax, 'A', 'Earlier SNN: feedforward, 218 k parameters')
    cy = 10
    cols = [8, 28, 46, 62, 78]
    hs = [20, 30, 24, 20, 18.5]
    for i in range(4):
        fan(ax, cols[i], hs[i], cols[i + 1], hs[i + 1], cy)
    capsule(ax, cols[0], cy, hs[0], '96', 'Input', 'multi-unit\nchannels')
    for c, h, n, tau in zip(cols[1:4], hs[1:4], ('512', '256', '128'), ('8', '16', '32')):
        capsule(ax, c, cy, h, n, 'IAF', f'synapse\n{tau} ms')
    capsule(ax, cols[4], cy, hs[4], '36', 'Spiking out', '18 per axis\n64 ms')
    tail(ax, cols[4] + 1.6, cy, 'two-stage\nEMA', 'population\nvector')


def panel_lsnn(ax):
    panel(ax, 'B', 'LSNN (from the speech decoder): recurrent, 223 k parameters')
    cy = 10
    cols = {'in': 8, 'h1': 34, 'h2': 58, 'out': 80}
    hs = {'in': 20, 'h1': 26, 'h2': 26, 'out': 7}
    fan(ax, cols['in'], hs['in'], cols['h1'], hs['h1'], cy)
    fan(ax, cols['h1'], hs['h1'], cols['h2'], hs['h2'], cy)
    fan(ax, cols['h2'], hs['h2'], cols['out'], hs['out'], cy)
    capsule(ax, cols['in'], cy, hs['in'], '96', 'Input', 'multi-unit\nchannels')
    for k, name in (('h1', 'Hidden 1'), ('h2', 'Hidden 2')):
        capsule(ax, cols[k], cy, hs[k], '256', name, 'recurrent, 90% adaptive\nsynapses 20 ms–1.3 s', edge=RED)
        recurrent_loop(ax, cols[k], cy, hs[k])
    pair(ax, cols['out'], cy, '2', 'Leaky out', 'one per axis\nnever spike')
    ax.add_patch(FancyArrowPatch((cols['h1'] + 1.8, cy + hs['h1'] / 2 - 2), (cols['out'] - 1.8, cy + hs['out'] / 2 - 2),
                                 connectionstyle='arc3,rad=-0.3', arrowstyle='-|>', mutation_scale=14, lw=1.4,
                                 color=INK, zorder=5))
    ax.text(cols['h2'] + 8, cy + 15, 'both layers\nread out', ha='left', va='bottom', fontsize=12, color=GRAY,
            linespacing=1.2)
    tail(ax, cols['out'] + 1.6, cy, 'two leaky\nstages', None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    style()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, hspace=0.08, left=0.03, right=0.99, top=0.98, bottom=0.02)
    panel_snn(fig.add_subplot(gs[0]))
    panel_lsnn(fig.add_subplot(gs[1]))
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(a.outDir, f'motor_models_figure.{ext}'), dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote motor_models_figure.png/.svg')


if __name__ == '__main__':
    main()
