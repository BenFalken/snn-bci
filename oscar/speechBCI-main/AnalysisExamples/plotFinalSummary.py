#summary figures for the deck, in its style (black on white, Palatino), from the final runs' reports:
#   final_cost.png      estimated power (fp32 and int8 weights, on-chip memory) and the time until a decision
#   final_branch.png    the build-up ladder and its fork into the LSNN and DelayNet, every step a 96-epoch run
#
#numbers: derived/efficiency/efficiency_readall96.md (power, parameters, compute time per output, lookahead) and the
#training logs (phoneme error on the test sentences of all 19 sessions; 36-epoch steps unless marked)
#
#example usage:
#   python plotFinalSummary.py --outDir ../figures
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAY, LIGHT, BLUE = '0.45', '0.8', '#2F5FA7'
#name: (power mW fp32 + on-chip, power mW int8 + on-chip, parameters M, compute ms per output, lookahead ms)
COST = {'Paper RNN': (6.46, 1.04, 53.6, 17.25, 80), 'GRU': (4.96, 0.80, 10.3, 12.50, 0),
        'SNN': (0.60, 0.10, 4.3, 19.85, 0)}
COLORS = {'Paper RNN': '0.25', 'GRU': '0.6', 'SNN': BLUE}

def style():
    plt.rcParams.update({'font.family': 'Palatino', 'font.size': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})


def save(fig, out_dir, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def cost(out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.1, 1]})
    names = list(COST)
    for i, n in enumerate(names):
        p32, p8, params, _, _ = COST[n]
        a.bar(i - 0.2, p32, 0.38, color=COLORS[n], edgecolor='black')
        a.bar(i + 0.2, p8, 0.38, color=COLORS[n], edgecolor='black', hatch='///', alpha=0.6)
        a.text(i, p32 * 1.35, f'{p32:.2f} mW\n{params:.1f}M parameters', ha='center', fontsize=12)
    a.set_yscale('log')
    a.set_ylim(0.05, 40)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names)
    a.set_ylabel('estimated power (mW, log scale)')
    a.set_title('a   Power: solid 32-bit weights, hatched 8-bit', loc='left', fontsize=14, fontweight='bold')
    for i, n in enumerate(names):
        _, _, _, comp, look = COST[n]
        b.barh(i, 80, color=COLORS[n], edgecolor='black')
        b.barh(i, look, left=80, color='white', edgecolor='black', hatch='///')
        b.barh(i, comp, left=80 + look, color=LIGHT, edgecolor='black')
        b.text(80 + look + comp + 4, i, f'{80 + look + comp:.0f} ms', va='center', fontsize=13)
    b.set_yticks(range(len(names)))
    b.set_yticklabels(names)
    b.invert_yaxis()
    b.set_xlim(0, 230)
    b.set_xlabel('ms from the start of 80 ms of speech to its decoded output')
    b.set_title('b   Time until a decision', loc='left', fontsize=14, fontweight='bold')
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(facecolor='0.5', edgecolor='black', label='output interval'),
                      Patch(facecolor='white', edgecolor='black', hatch='///', label='waiting for future input'),
                      Patch(facecolor=LIGHT, edgecolor='black', label='compute, 1 CPU thread')],
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, 'final_cost')


#the build-up as a fork, every point a 96-epoch run (seed 42 on the trunk), output from the last layer only, one
#change per rung: width first, then recurrence (recurrence only pays off once the layers are wide: the narrow
#recurrent network, R1_rec, barely beats the original), 80 ms outputs, the readout from both layers, trained synaptic
#time constants (the root); then memory from adaptive neurons (red, the final LSNN) or from synaptic delays (light
#gray). Trunk: L0_orig, L1_width (ladder96.txt), S2_rec, S3_out80, S4_readall (fork96.txt), C_nodelay96
#(nodelay96.txt); None = not finished, drawn as a pending rung. Branches, one entry per seed (42, 43, 44), both reading
#both layers: LSNN C_lsnn_nodelay96, Q_lsnn_s43/s44; DelayNet R_readall96, R_readall_s43/s44
RED = '#C0392B'
TRUNK = [('Original SNN\n(feedforward,\n512-256-128)', 0.900), ('+ width\n(2 × 1,024)', 0.7226),
         ('+ recurrence', 0.3423), ('+ outputs every\n80 ms', 0.2636),
         ('+ readout from\nboth layers', 0.2639), ('+ trained synaptic\ntime constants', 0.2627)]
BRANCHES = {'delays': ('Model A, DelayNet', '0.7', [0.2449, 0.2492, 0.2504]),
            'lsnn': ('Model B, LSNN', RED, [0.2499, 0.2446, 0.2450])}
WIDE_LSNN = []                                                   #LSNN at 2 × 1,152: only run with both-layer readout


def branch(out_dir):
    import numpy as np
    fig, ax = plt.subplots(figsize=(15, 6.4))
    known = [(i, p) for i, (_, p) in enumerate(TRUNK) if p is not None]
    for (i0, y0), (i1, y1) in zip(known, known[1:]):            #dashed across rungs still training
        ax.plot([i0, i1], [y0, y1], '-' if i1 == i0 + 1 else '--', color='0.25', lw=2.2, zorder=1)
    for i, y in known:
        ax.scatter([i], [y], s=110, color='0.25', zorder=3)
        ax.text(i, y + 0.04, f'{y:.3f}', ha='center', fontsize=15)
    xs, ys = [i for i, _ in known], [p for _, p in known]
    end = len(TRUNK)
    means = []
    for (k, (label, color, seeds)), dx, lw in zip(BRANCHES.items(), (-0.07, 0.07), (6, 2.2)):
        m = np.mean([s for s in seeds if s is not None])   #the two branches land together: a wide gray rail under a
        means.append(m)                                     #thin red one, and side-by-side end points
        ax.plot([xs[-1], end + dx], [ys[-1], m], '-', color=color, lw=lw, zorder=2, solid_capstyle='round')
        ax.scatter([end + dx], [m], s=150, color=color, edgecolor='0.25', zorder=4)
    ax.text(end, max(means) + 0.04, f'{np.mean(means):.3f}', ha='center', fontsize=15, fontweight='bold')
    ax.axhline(0.208, color=GRAY, ls='--', lw=1.3)
    ax.axhline(0.188, color=GRAY, ls=':', lw=1.3)
    ax.text(-0.3, 0.208 + 0.012, 'LSTM, same input, 96 epochs: 0.208', ha='left', fontsize=13, color=GRAY)
    ax.text(-0.3, 0.188 - 0.035, 'paper RNN (640 ms window, 5 × 1,024): 0.188', ha='left', fontsize=13, color=GRAY)
    ax.set_xticks(list(range(len(TRUNK))) + [end])
    ax.set_xticklabels([l + ('\n(training)' if p is None else '') for l, p in TRUNK] + ['Model A: DelayNet\nor\nModel B: LSNN'],
                       fontsize=13)
    ax.set_ylabel('phoneme error (test, 19 sessions)')
    ax.set_ylim(0.1, 1.0)
    ax.set_xlim(-0.4, end + 0.5)

    #zoom on the fork: every seed, and the branch means
    z = ax.inset_axes([0.55, 0.52, 0.43, 0.45])
    z.plot([0], [ys[-1]], 'o', color='0.25', ms=11)
    z.text(0, ys[-1] + 0.0018, f'{ys[-1]:.3f}', ha='center', fontsize=12)
    rows = [('delays', 1.0), ('lsnn', 2.0)]
    for k, x in rows:
        label, color, seeds = BRANCHES[k]
        done = [s for s in seeds if s is not None]
        m = np.mean(done)
        z.plot([0, x], [ys[-1], m], '-', color=color, lw=2)
        z.scatter(x + np.linspace(-0.08, 0.08, len(done)), done, s=34, color=color, edgecolor='0.25', lw=0.6, zorder=3)
        z.plot([x - 0.18, x + 0.18], [m, m], color=color, lw=3, zorder=2)
        z.text(x, max(done) + 0.0018, f'{m:.3f}\n({len(done)} seed{"s" if len(done) > 1 else ""})', ha='center', fontsize=11)
    wide = [s for s in WIDE_LSNN if s is not None]
    if wide:
        z.scatter(2.42 + np.linspace(-0.05, 0.05, len(wide)), wide, s=34, facecolor='white', edgecolor=RED, lw=1.4, zorder=3)
        z.text(2.42, min(wide) - 0.004, f'2 × 1,152\n{np.mean(wide):.3f}', ha='center', fontsize=10, color=RED)
    z.set_xticks([0, 1, 2])
    z.set_xticklabels(['IAF, no delays', 'Model A: DelayNet\n(delayed input\nsynapses)', 'Model B: LSNN\n(adaptive\nneurons)'], fontsize=11)
    z.set_xlim(-0.35, 2.7)
    z.set_ylim(0.236, 0.268)
    z.tick_params(labelsize=10)
    z.set_title('the fork, every seed (96 epochs)', fontsize=12, loc='left')
    fig.tight_layout()
    save(fig, out_dir, 'final_branch')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default='.')
    a = p.parse_args()
    style()
    cost(a.outDir)
    branch(a.outDir)


if __name__ == '__main__':
    main()
