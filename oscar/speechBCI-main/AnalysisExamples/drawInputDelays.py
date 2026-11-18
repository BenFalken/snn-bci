#figure: DelayNet's delayed input synapses -- one input channel over the last 600 ms (30 bins of 20 ms) reaching one
#first-layer neuron through 4 synapses, each with its own weight and fixed delay -- in the deck's style (Palatino):
#   python drawInputDelays.py --outDir ../figures
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

INK, GRAY, LIGHT, ACCENT = '0.1', '0.45', '0.85', '0.35'
TAPS = [(29, 'w₁, 0 ms'), (25, 'w₂, 80 ms'), (16, 'w₃, 260 ms'), (4, 'w₄, 500 ms')]   #(bin, label); bin 29 = now


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 15})
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(-3.0, 31)
    ax.set_ylim(-5.0, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    y0 = 1.0
    taps = {b for b, _ in TAPS}
    for b in range(30):                                         #one input channel, the last 30 bins
        ax.add_patch(Rectangle((b + 0.06, y0), 0.88, 0.9, fc='0.15' if b in taps else 'white',
                               ec=INK if b in taps else LIGHT, lw=1.4 if b in taps else 1.0))
    ax.text(-0.4, y0 + 0.45, 'input\nchannel', ha='right', va='center', fontsize=14)
    ax.text(0, y0 - 0.25, '600 ms ago', ha='left', va='top', fontsize=13, color=GRAY)
    ax.text(30, y0 - 0.25, 'now', ha='right', va='top', fontsize=13, color=GRAY)
    nx, ny = 17.0, -3.8                                         #the first-layer neuron
    ax.add_patch(Circle((nx, ny), 0.75, fc='white', ec=INK, lw=1.8))
    ax.text(nx, ny, 'j', ha='center', va='center', fontsize=18, style='italic')
    ax.text(nx + 1.2, ny, 'one first-layer neuron', ha='left', va='center', fontsize=14)
    for b, lab in TAPS:
        ax.add_patch(FancyArrowPatch((b + 0.5, y0 - 0.05), (nx, ny + 0.8), connectionstyle='arc3,rad=0.12',
                                     arrowstyle='-|>', mutation_scale=15, lw=1.5, color=ACCENT))
        ax.text(b + 0.5, y0 + 1.2, lab, ha='center', va='bottom', fontsize=13)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(a.outDir, f'input_delays.{ext}'), dpi=200, facecolor='white', bbox_inches='tight')
    print('wrote input_delays.png/.svg')


if __name__ == '__main__':
    main()
