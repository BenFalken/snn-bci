#figure: every avenue explored and set aside, as the change in test phoneme error against the run it modified (each test
#its own base), grouped by what it tried to improve; open markers 36-epoch screens, filled 96-epoch checks; the band is
#the seed-to-seed spread (about +-0.005). In the deck's style (Palatino, white):
#   python plotExplorations.py --outDir ../figures
#numbers: the training logs of each round (base -> variant, seed 42 unless noted)
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

INK, GRAY, BAND = '0.1', '0.5', '0.93'
NOISE = 0.005
#two output smoothing stages on the final LSNN (F_lsnn_s42/43/44) minus the final LSNN of the same seed
#(C_lsnn_nodelay96 0.2499, Q_lsnn_s43 0.2446, Q_lsnn_s44 0.2450); None while a seed is training
OUT2_PAIRS = [(0.2497, 0.2499), (0.2486, 0.2446), (None, 0.2450)]
OUT2 = sum(a - b for a, b in OUT2_PAIRS if a is not None) / sum(a is not None for a, _ in OUT2_PAIRS)
#(group, label, change in PER = variant - base, epochs); bases in the comments
TESTS = [
    ('More memory', 'Recurrent + between-layer delays, on top of input delays', 0.270 - 0.274, 36),   #model A vs B1_in4
    ('More memory', 'Adaptive neurons, on top of input delays', 0.273 - 0.279, 36),                   #R_lsnn90 vs B1_in4
    ('More memory', 'Between-layer delays', 0.318 - 0.314, 36),                                       #D_ff vs C_base
    ('More memory', 'Every delay type and adaptation together', 0.342 - 0.314, 36),                   #C_all vs C_base
    ('More memory', 'Adaptation mix 40%, 1 s instead of none', 0.301 - 0.298, 36),                    #LSNN sweep vs C_iaf
    ('More memory', 'Learned delay positions', 0.2511 - 0.2449, 96),                                  #R_readall_learn96 vs R_readall96
    ('More memory', 'Gradient horizon 4 s instead of 2 s', 0.334 - 0.314, 36),                        #C_tbptt200 vs C_base
    ('More memory', 'Input delays up to 300 ms instead of 600 ms', 0.2875 - 0.2745, 36),              #R_in300 vs B1_in4 (2 seeds)
    ('Wider spike channel', 'Up to 7 spikes per step instead of 3', 0.3003 - 0.2465, 96),             #F_spikes7 vs R_full96
    ('Wider spike channel', 'Membrane readout instead of spikes', 0.2432 - 0.2465, 96),               #F_vmem vs R_full96
    ('Wider spike channel', 'Event-encoded input', 0.328 - 0.314, 36),                                #C_events vs C_base
    ('Neuron model', 'SE-adLIF neurons instead of adaptive IAF', 0.2522 - 0.2516, 96),               #E_sead vs B_noro_s42
    ('Neuron model', 'SE-adLIF, slower time constants', 0.2870 - 0.2516, 96),                        #E_sead_long vs B_noro_s42
    ('Readout and training', 'Last layer only instead of both layers (3 seed pairs)', 0.2502 - 0.2465, 96),  #B_noro vs LSNN
    ('Readout and training', 'Two output smoothing stages (seed pairs)', OUT2, 96),                   #F_lsnn vs LSNN, same seeds
    ('Readout and training', 'Extra dropout', 0.2471 - 0.2449, 96),                                   #T_drop96 vs R_readall96
    ('Readout and training', 'Adam epsilon 1e-4, learning rate 0.002', 0.2700 - 0.2499, 96),          #Q_lsnn_eps vs C_lsnn_nodelay96
    ('Size', 'LSNN 2 × 1,152 instead of 2 × 1,024 (3 seeds)', 0.2438 - 0.2465, 96),                  #Q_w1152 vs LSNN, 3-seed means
    ('Size', '8 input synapses per pair instead of 4', 0.2745 - 0.2745, 36),                          #R_in8 vs B1_in4
    ('Size', '2 × 2,048 instead of 2 × 1,024', 0.430 - 0.424, 36),                                    #N_wide2048 vs native36 base
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False, 'svg.fonttype': 'none'})
    rows, y, ticks, groups = [], 0, [], []
    for i, (g, lab, d, ep) in enumerate(TESTS):
        if i == 0 or g != TESTS[i - 1][0]:
            if i:
                y += 0.8                                            #a gap between groups
            groups.append((g, y))
            y += 0.9
        rows.append((y, lab, d, ep))
        y += 1
    fig, ax = plt.subplots(figsize=(15, 0.31 * y + 1.1))
    ax.axvspan(-NOISE, NOISE, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=GRAY, lw=1.0, zorder=1)
    for yy, lab, d, ep in rows:
        ax.plot([0, d], [yy, yy], color=INK, lw=1.1, zorder=2, solid_capstyle='butt')
        ax.scatter([d], [yy], s=54, zorder=3, marker='o', facecolor=INK if ep == 96 else 'white', edgecolor=INK, lw=1.3)
        ax.text(d + (0.0015 if d >= 0 else -0.0015), yy, f'{d:+.3f}', va='center', ha='left' if d >= 0 else 'right',
                fontsize=12, color=GRAY)
    label_col = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for g, gy in groups:                                        #group headings in the label column
        ax.text(-0.012, gy, g, ha='right', va='center', fontsize=15, fontweight='bold', transform=label_col)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=13)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y, -0.2)
    ax.set_xlim(-0.02, 0.062)
    ax.set_xlabel('change in phoneme error against the run it modified  (left: better, right: worse)', fontsize=13)
    ax.text(0, -0.15, 'seed-to-seed spread', ha='center', va='bottom', fontsize=11, color=GRAY)
    ax.scatter([], [], s=54, facecolor='white', edgecolor=INK, label='36 epochs')
    ax.scatter([], [], s=54, facecolor=INK, edgecolor=INK, label='96 epochs')
    ax.legend(loc='lower right', frameon=False, fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(a.outDir, f'explorations.{ext}'), dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote explorations.png/.svg')


if __name__ == '__main__':
    main()
