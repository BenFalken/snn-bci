#figure: what one input to the paper's GRU looks like. One real test sentence's 256 features (20 ms bins), with the
#sliding 640 ms windows the GRU reads (32 bins, one step every 4 bins = 80 ms), and one window zoomed in as the
#32 x 256 snapshot that is flattened into 8,192 inputs for a single GRU step.
#
#example usage (on Oscar, where the data are):
#   python plotGRUInput.py --dataDir /users/$USER/scratch/speechBCI_dryad/derived/snnData \
#       --session t12.2022.08.13 --trial 0 --out gru_input
#writes gru_input.svg, gru_input.pdf and gru_input.png
#
#features (derived/snnData, as every model in this project reads them): columns 0-127 = threshold crossings,
#128-255 = spike band power, both from the area 6v arrays, normalized within each recording block
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import to_rgba

BIN_MS, WINDOW, STRIDE = 20, 32, 4
NEW_COL, OLD_COL = '#1f5fbf', '#555b66'   # incoming bins (blue), leaving bins (gray)

def load_trial(data_dir, session, trial):
    with np.load(os.path.join(data_dir, session, 'test.npz')) as f:
        feats = np.split(f['features'], np.cumsum(f['featureLens'])[:-1])[trial]
        codes = np.asarray(f['transcriptions'][trial])
    end = np.argwhere(codes == 0)
    text = ''.join(chr(c) for c in (codes[:end[0, 0]] if len(end) else codes))
    return feats.astype(np.float32), text.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataDir', required=True)
    p.add_argument('--session', default='t12.2022.08.13')
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--window', type=int, default=None, help='which window to zoom on (default: mid-sentence)')
    p.add_argument('--out', default='gru_input')
    a = p.parse_args()

    x, text = load_trial(a.dataDir, a.session, a.trial)
    T = len(x)
    n_win = (T - WINDOW) // STRIDE + 1
    k = a.window if a.window is not None else max(1, min(n_win - 2, (T // 2 - WINDOW) // STRIDE))
    s0 = k * STRIDE                       # window k covers bins [s0, s0 + 32)
    t = lambda b: b * BIN_MS / 1000.0
    vmax = 3.0
    cmap = 'RdBu_r'

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(12, 9.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.22, 1.15], hspace=0.32, left=0.11, right=0.97, top=0.9,
                             bottom=0.13)
    top_gs = outer[0].subgridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.06)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.16)

    # ---- A (top strip): three consecutive windows as bars on the sentence's time axis
    axw = fig.add_subplot(top_gs[0])
    for row, (kk, lab, col) in enumerate([(k + 1, 'step k+1', '0.6'), (k, 'step k', 'k'), (k - 1, 'step k-1', '0.6')]):
        st = kk * STRIDE
        axw.barh(row, t(WINDOW), left=t(st), height=0.62, color=col if kk == k else '0.85', edgecolor=col, lw=1.2)
        axw.text(t(st) - 0.02, row, lab, ha='right', va='center', fontsize=9, color='k' if kk == k else '0.35')
    axw.annotate('', xy=(t(s0 + STRIDE + WINDOW) + 0.01, 2.0), xytext=(t(s0 + WINDOW) + 0.01, 2.0),
                 arrowprops=dict(arrowstyle='->', color='k', lw=1))
    axw.text(t(s0 + STRIDE + WINDOW) + 0.04, 1.5, 'each step: +80 ms (4 bins)', va='center', fontsize=9)
    axw.text(t(s0 + WINDOW / 2), 1.0, '640 ms', ha='center', va='center', fontsize=9, color='white',
             fontweight='bold')
    axw.set_xlim(0, t(T))
    axw.set_ylim(-0.6, 2.6)
    axw.axis('off')
    axw.set_title(f'A   One sentence as the decoder sees it: "{text}"', loc='left', fontsize=11,
                  fontweight='bold', pad=8)

    # ---- A: the whole sentence's features
    ax = fig.add_subplot(top_gs[1])
    ax.imshow(x.T, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest',
              extent=[0, t(T), 256, 0])
    ax.axhline(128, color='k', lw=0.6)
    ax.add_patch(Rectangle((t(s0), 1), t(WINDOW), 254, fill=False, ec='k', lw=2))
    ax.set_yticks([64, 192])
    ax.set_yticklabels(['threshold\ncrossings\n(128)', 'spike band\npower\n(128)'])
    ax.set_xlabel('time in the sentence (s); one column = one 20 ms bin')
    ax.set_xlim(0, t(T))
    ax.set_ylim(256, 0)

    # ---- B: the zoomed window: 32 bins x 256 features
    axz = fig.add_subplot(bot_gs[0])
    win = x[s0:s0 + WINDOW]
    axz.imshow(win.T, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest',
               extent=[-WINDOW * BIN_MS, 0, 256, 0])
    axz.axhline(128, color='k', lw=0.6)
    # incoming (newest 4 bins) and leaving (oldest 4 bins): dotted boxes with a faint fill of the border color
    for x_left, col in ((-STRIDE * BIN_MS, NEW_COL), (-WINDOW * BIN_MS, OLD_COL)):
        axz.add_patch(Rectangle((x_left, 1), STRIDE * BIN_MS, 254, fc=to_rgba(col, 0.22), ec=col, lw=2.2, ls=':'))
    axz.set_xlabel('ms before the step (each column = one 20 ms bin)')
    axz.set_yticks([64, 192])
    axz.set_yticklabels(['crossings', 'band\npower'])
    axz.set_title('B   Window k: a 32 \u00d7 256 snapshot of the last 640 ms', loc='left', fontsize=11,
                  fontweight='bold', pad=22)
    axz.text(-STRIDE * BIN_MS / 2, -4, 'new bins', color=NEW_COL, ha='center', va='bottom', fontsize=9,
             fontweight='bold')
    axz.text(-WINDOW * BIN_MS + STRIDE * BIN_MS / 2, -4, 'leave next step', color=OLD_COL, ha='center',
             va='bottom', fontsize=9, fontweight='bold')
    axz.set_ylim(256, 0)

    # ---- C: what the GRU does with it
    axc = fig.add_subplot(bot_gs[1])
    axc.axis('off')
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_title('C   One GRU step', loc='left', fontsize=11, fontweight='bold', pad=22)
    box = dict(boxstyle='round,pad=0.5', fc='white', ec='0.3', lw=1)
    axc.text(0.5, 0.88, '32 bins \u00d7 256 features\nflattened into 8,192 numbers', ha='center', va='center',
             bbox=box, fontsize=10)
    axc.text(0.5, 0.56, 'first GRU layer: one weight per number\n(a separate weight for every feature\nat every bin of the window)',
             ha='center', va='center', bbox=box, fontsize=10)
    axc.text(0.5, 0.23, '5 GRU layers \u2192 41 phoneme probabilities\nfor this 80 ms step', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='#eef3fb', ec='#2f5fa7', lw=1.2), fontsize=10)
    for y0, y1 in ((0.79, 0.68), (0.44, 0.33)):
        axc.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle='->', mutation_scale=14, color='0.25'))
    axc.text(0.5, 0.02, 'Consecutive windows share 560 ms,\nso each bin is processed 8 times.', ha='center',
             va='bottom', fontsize=9, color='0.3')

    cax = fig.add_axes([0.11, 0.045, 0.25, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap), cax=cax,
                      orientation='horizontal')
    cb.set_label('feature value (normalized within block)', fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('Input to the reference GRU: a sliding 640 ms window, stepped every 80 ms', x=0.11, ha='left',
                 fontsize=13, fontweight='bold')
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=200)
    print(f'wrote {a.out}.svg/.pdf/.png  ({a.session} test trial {a.trial}, {T} bins = {t(T):.2f} s, window k = {k})')

if __name__ == '__main__':
    main()
