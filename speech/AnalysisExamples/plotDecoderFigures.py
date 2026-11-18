#figures comparing the final decoders, from compareDecoders.py (fidelity per session, with the language model) and
#SNNDecoder/scripts/efficiency.py (latency, parameters, energy):
#
#   <out>_fidelity.png   a/b  per-session phoneme error (greedy) and word error (+ 3-gram LM), one box per decoder;
#                             stars: paired Wilcoxon signed-rank test across sessions against the best decoder
#                        c/d  pairwise: each cell = mean (row - column) error over sessions, coloured by the share of
#                             sessions where the row decoder is better (lower error), stars for the paired test
#   <out>_latency.png         error against compute time per 80 ms output (log scale, 1 CPU thread, batch 1,
#                             state carried), marker size = parameters deployed for one session
#   <out>_energy.png          estimated energy per decoded output (80 ms) and average power, per decoder, under the
#                             efficiency report's accounting (45 nm, fp32 weights in on-chip memory; and int8 hatched)
#
#example usage (on Oscar):
#   B=/users/$USER/scratch/speechBCI_dryad
#   python plotDecoderFigures.py --compare $B/derived/decoderComparison_final96.json \
#       --efficiency $B/derived/efficiency/efficiency_readall96.json --out $B/derived/figures/final96 \
#       --match "GRU96=GRU 2x1024 out80, same input, 96 ep" --match "SNN96=Reference, 96 ep" \
#       --match "SNN96_readall=Reference + readout from both layers, 96 ep" --match "RNN=Paper RNN"
#--match NAME=LABEL ties a decoder of the comparison to its row of the efficiency report (same model); --label
#NAME=TEXT sets the name shown in the figures.
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_S = 0.08                                                     #one decoded output every 80 ms
SHOW_MEASURED = False   #--measuredEnergy: add the CPU 'measured' panel (a CPU cannot skip silent neurons, and on a
                        #cluster node the meter is often only a utilisation-based proxy)
SNN_COLORS = ['#2F5FA7', '#7FA6DB', '#1B3A6B', '#A9C3E8']        #spiking decoders: blues
ANN_COLORS = ['#555555', '#A0A0A0', '#C0661E', '#7A7A7A']        #non-spiking decoders: grays (and orange)


#one fixed colour per decoder family, the same in every figure and in the deck: the final LSNN red, the other spiking
#model a light red, recurrent ANNs blues (the paper RNN near black), the linear filters ochres; differing in lightness
FAMILY_COLORS = [('lsnn', '#C0392B'), ('delaynet', '#E8A39B'), ('qrnn', '#5B7DB1'), ('lstm', '#2F5FA7'),
                 ('gru', '#9DBBE3'), ('wiener', '#D4AC2B'), ('kalman', '#8A6512')]
PAPER_RNN_COLOR = '#262626'


def colors_for(names, labels):
    c, i, j = {}, 0, 0
    for n in names:
        key = (n + ' ' + labels[n]).lower()
        fam = next((col for k, col in FAMILY_COLORS if k in key), None)
        if fam is None and n == 'RNN':
            fam = PAPER_RNN_COLOR
        if fam is not None:
            c[n] = fam
        elif 'snn' in key:
            c[n], i = SNN_COLORS[i % len(SNN_COLORS)], i + 1
        else:
            c[n], j = ANN_COLORS[j % len(ANN_COLORS)], j + 1
    return c


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def paired_p(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def session_matrix(entries, names, key):
    sessions = list(entries[names[0]][key])
    return sessions, np.array([[entries[n][key][s] for s in sessions] for n in names])


def fidelity(entries, names, labels, out, COL):
    keys = [('per_session', 'Phoneme error rate (greedy)'), ('wer_session', 'Word error rate (+ 3-gram LM)')]
    keys = [k for k in keys if all(k[0] in entries[n] for n in names)]
    fig = plt.figure(figsize=(6.2 * len(keys), 11))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1, 1.05], hspace=1.0, wspace=0.3)
    cmap = LinearSegmentedColormap.from_list('rb', ['#8B1A1A', '#F2F2F2', '#1F3F7A'])
    for col, (key, title) in enumerate(keys):
        sessions, M = session_matrix(entries, names, key)
        M = M * 100
        ax = fig.add_subplot(gs[0, col])
        bp = ax.boxplot(M.T, patch_artist=True, widths=0.6, medianprops=dict(color='black', lw=1.6),
                        flierprops=dict(marker='o', mfc='white', mec='black', ms=5), showmeans=True,
                        meanprops=dict(marker='o', mfc='white', mec='black', ms=6))
        for patch, c in zip(bp['boxes'], [COL[x] for x in names]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        best = int(np.argmin(M.mean(1)))
        top = M.max()
        for i in range(len(names)):
            if i != best:
                ax.text(i + 1, top * 1.04, stars(paired_p(M[i], M[best])), ha='center', fontsize=10)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([labels[n] for n in names], rotation=20, ha='right')
        ax.set_ylabel(f'{title} (%)')
        ax.set_ylim(min(0, M.min() * 0.9), top * 1.12)
        ax.set_title(f'{"ab"[col]}   {title}, {len(sessions)} sessions', loc='left', fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

        ax = fig.add_subplot(gs[1, col])
        n = len(names)
        share = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i != j:
                    share[i, j] = 100 * np.mean(M[i] < M[j])
        ax.imshow(share, cmap=cmap, vmin=0, vmax=100)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#DDDDDD'))
                    continue
                d = (M[i] - M[j]).mean()
                ax.text(j, i - 0.08, f'{d:+.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(share[i, j] - 50) > 25 else 'black')
                ax.text(j, i + 0.22, stars(paired_p(M[i], M[j])), ha='center', va='center', fontsize=7,
                        color='white' if abs(share[i, j] - 50) > 25 else 'black')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([labels[x] for x in names], rotation=35, ha='left')
        ax.set_yticklabels([labels[x] for x in names])
        ax.xaxis.tick_top()
        ax.set_title(f'{"cd"[col]}   row − column, points of error', loc='left', fontsize=12, fontweight='bold',
                     pad=105)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
        cb.set_label('Sessions where the row decoder is better (%)')
    fig.savefig(out + '_fidelity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def eff_rows(eff, match):
    rows = {r['label']: r for r in eff['rows']}
    return {name: rows[label] for name, label in match.items() if label in rows}


def latency(entries, names, labels, rows, out, metric, COL):
    pts = []
    for n in names:
        r = rows.get(n)
        t = (r or {}).get('timing', {}).get('s_per_output')
        if t is None:
            continue
        err = entries[n][metric] if metric != 'wer' else entries[n]['wer']
        err = err[0] if isinstance(err, (list, tuple)) else err
        pts.append((n, t * 1e3, 100 * err, r['memory']['params_deployed']))
    if not pts:
        print('latency figure: no decoder has a timing in the efficiency report; skipped')
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 8.2), gridspec_kw={'width_ratios': [1.2, 1]})
    pmax = max(p for *_, p in pts)
    size = lambda p: 120 + 1300 * p / pmax                     # marker area by parameters
    ts = [t for _, t, _, _ in pts]
    es = [e for _, _, e, _ in pts]
    y0, y1 = min(es) - 6, max(es) + 6
    ax.set_xscale('log')
    ax.set_xlim(min(ts) / 4, max(ts) * 40)
    ax.set_ylim(y0, y1)
    # labels to the right of their points, pushed apart vertically so none overlap (min gap 7% of the axis)
    order = sorted(range(len(pts)), key=lambda i: pts[i][2])
    ly = {i: pts[i][2] for i in order}
    gap = 0.09 * (y1 - y0)
    for _ in range(50):
        moved = False
        for a_, b_ in zip(order, order[1:]):
            if ly[b_] - ly[a_] < gap:
                mid = (ly[a_] + ly[b_]) / 2
                ly[a_], ly[b_] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    for i, (n, t, e, p) in enumerate(pts):
        ax.scatter(t, e, s=size(p), color=COL[n], edgecolor='black', alpha=0.8, zorder=3)
        # one label column per cluster of nearby points (within a decade), right of the cluster's last marker
        lx = max(t2 for t2 in ts if abs(math.log10(t2 / t)) < 1) * 2.6
        ax.annotate(f'{labels[n]}  {t:.2f} ms', (t, e), xytext=(lx, ly[i]), textcoords='data', fontsize=11,
                    fontweight='bold', va='center',
                    arrowprops=dict(arrowstyle='-', color='0.55', lw=0.7, shrinkA=0, shrinkB=6))
    ax.set_xlabel('Compute time per 80 ms output (ms, log scale; 1 CPU thread, batch 1)')
    ax.set_ylabel(('Phoneme error rate' if metric == 'per' else 'Word error rate, + LM') + ' (%)')
    ax.grid(True, which='both', ls=':', color='0.85')
    ax.set_title('a   Simulation time on a CPU (marker size = parameters)', loc='left', fontsize=12,
                 fontweight='bold')
    # b: when a decision is available after the speech it is about: the output interval (a frame is emitted at the
    # end of its 80 ms), any look into the future (the paper RNN's centred input smoothing), and the compute time
    rows_b = []
    for n, t, e, p in pts:
        look = float(entries[n].get('lookahead_s') or 0.0) * 1e3
        rows_b.append((n, OUT_S * 1e3, look, t, e))
    for n in names:                                   #decoders without a timing (e.g. the paper RNN's spec)
        if n not in [r[0] for r in rows_b] and entries[n].get('lookahead_s') is not None:
            err = entries[n][metric] if metric != 'wer' else entries[n]['wer']
            err = err[0] if isinstance(err, (list, tuple)) else err
            prof = (entries[n].get('profile') or {}).get('latency_per_output_s')
            rows_b.append((n, OUT_S * 1e3, float(entries[n]['lookahead_s']) * 1e3,
                           (prof or 0.0) * 1e3, 100 * err))
    rows_b.sort(key=lambda r: names.index(r[0]))
    ys = range(len(rows_b))
    for y, (n, frame, look, comp, e) in zip(ys, rows_b):
        ax2.barh(y, frame, color=COL[n], edgecolor='black', alpha=0.9)
        ax2.barh(y, look, left=frame, color='white', edgecolor='black', hatch='///')
        ax2.barh(y, comp, left=frame + look, color='0.85', edgecolor='black')
        ax2.text(frame + look + comp + 4, y, f'{frame + look + comp:.0f} ms   ({e:.1f}% error)', va='center',
                 fontsize=10)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([labels[r[0]] for r in rows_b])
    ax2.invert_yaxis()
    ax2.set_xlabel('ms from the start of an 80 ms stretch of speech to its decoded output')
    ax2.set_xlim(0, max(r[1] + r[2] + r[3] for r in rows_b) * 1.45)
    ax2.set_title('b   Time until a decision is available', loc='left', fontsize=12, fontweight='bold')
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor='0.4', edgecolor='black', label='output interval (80 ms)'),
                        Patch(facecolor='white', edgecolor='black', hatch='///', label='waiting for future input'),
                        Patch(facecolor='0.85', edgecolor='black', label='compute (1 CPU thread)')],
               loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    for p in sorted({round(pmax / 4, -5), round(pmax / 2, -5), round(pmax, -5)}):
        if p > 0:
            ax.scatter([], [], s=size(p), color='0.75', edgecolor='black', label=f'{p / 1e6:.1f} M')
    ax.legend(title='parameters', loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
              labelspacing=2, borderpad=1.5, columnspacing=3, handletextpad=1.5)
    fig.savefig(out + '_latency.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def energy(names, labels, rows, out, entries, COL):
    bars = [(n, rows[n]['ops']['energy']) for n in names if n in rows]
    if not bars:
        print('energy figure: no decoder matched to the efficiency report; skipped')
        return
    meas = {n: entries[n].get('profile') for n in names if (entries[n].get('profile') or {}).get('energy_j')} \
        if SHOW_MEASURED else {}
    ncol = 2 if meas else 1
    fig, axs = plt.subplots(1, ncol, figsize=(7.5 * ncol, 6), squeeze=False)
    ax = axs[0, 0]
    for i, (n, e) in enumerate(bars):
        c = COL[n]
        uj = e['sram_fp32']['uw'] * OUT_S                       #uW x 80 ms = uJ per output
        uj8 = e['sram_int8']['uw'] * OUT_S
        ax.bar(i - 0.2, uj, 0.38, color=c, edgecolor='black')
        ax.bar(i + 0.2, uj8, 0.38, color=c, edgecolor='black', hatch='///', alpha=0.7)
        ax.text(i, max(uj, uj8) * 1.6, f'{labels[n]}\n{uj:.3g} µJ / output\n{e["sram_fp32"]["uw"] / 1e3:.3g} mW',
                ha='center', fontsize=9)
    ax.set_yscale('log')
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([labels[n] for n, _ in bars], rotation=15)
    ax.set_ylabel('Energy per decoded output (µJ, log scale)')
    lo = min(rows[n]['ops']['energy']['sram_int8']['uw'] * OUT_S for n, _ in bars)
    hi = max(rows[n]['ops']['energy']['sram_fp32']['uw'] * OUT_S for n, _ in bars)
    ax.set_ylim(lo / 4, hi * 40)
    ax.set_title('Estimated (45 nm; solid fp32, hatched int8; weights on chip)', fontsize=11)
    ax.grid(True, axis='y', which='both', ls=':', color='0.85')
    if meas:
        ax = axs[0, 1]
        ks = list(meas)
        for i, n in enumerate(ks):
            p = meas[n]
            uj = p['energy_j'] * 1e6 * (OUT_S / 0.02)            #per 20 ms bin -> per 80 ms output
            ax.bar(i, uj, 0.6, color=COL[n], edgecolor='black')
            ax.text(i, uj * 1.6, f'{labels[n]}\n{uj:.3g} µJ / output' +
                    (f"\n{p['power_w']:.3g} W" if p.get('power_w') else ''), ha='center', fontsize=9)
        ax.set_yscale('log')
        vals = [meas[n]['energy_j'] * 1e6 * (OUT_S / 0.02) for n in ks]
        ax.set_ylim(min(vals) / 4, max(vals) * 40)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([labels[n] for n in ks], rotation=15)
        ax.set_ylabel('Energy per decoded output (µJ, log scale)')
        ax.set_title(f"Measured on CPU ({next(iter(meas.values())).get('energy_method', 'meter')})", fontsize=11)
        ax.grid(True, axis='y', which='both', ls=':', color='0.85')
    fig.suptitle('Energy per decoded output (one every 80 ms)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out + '_energy.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', required=True, help='compareDecoders.py output (.json)')
    ap.add_argument('--efficiency', default=None, help='efficiency.py output (.json), for latency/parameters/energy')
    ap.add_argument('--match', action='append', default=[], help='NAME=efficiency row label')
    ap.add_argument('--label', action='append', default=[], help='NAME=label shown in the figures')
    ap.add_argument('--decoders', nargs='+', default=None, help='which decoders, in this order (default: all)')
    ap.add_argument('--latencyMetric', default='per', choices=['per', 'wer'])
    ap.add_argument('--sort', default='none', choices=['none', 'per', 'wer'],
                    help='order the decoders by mean per-session phoneme (per) or word (wer) error, best first')
    ap.add_argument('--measuredEnergy', action='store_true', help='add the CPU-measured energy panel')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plt.rcParams.update({'font.family': ['Palatino', 'TeX Gyre Pagella', 'URW Palladio L', 'P052', 'serif'],
                         'font.size': 11})              # the deck's face; Palatino clones where it is missing
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    entries = json.load(open(a.compare))
    names = a.decoders or list(entries)
    if a.sort != 'none':                                  # most accurate first (lowest mean error over sessions)
        key = {'per': 'per_session', 'wer': 'wer_session'}[a.sort]
        names = sorted(names, key=lambda n: np.mean(list(entries[n][key].values())))
    labels = {n: n for n in names}
    labels.update(dict(x.split('=', 1) for x in a.label))
    global SHOW_MEASURED
    SHOW_MEASURED = a.measuredEnergy
    COL = colors_for(names, labels)
    fidelity(entries, names, labels, a.out, COL)
    if a.efficiency:
        rows = eff_rows(json.load(open(a.efficiency)), dict(x.split('=', 1) for x in a.match))
        missing = [n for n in names if n not in rows]
        if missing:
            print(f'not matched to the efficiency report (no latency/energy): {missing}')
        latency(entries, names, labels, rows, a.out, a.latencyMetric, COL)
        energy(names, labels, rows, a.out, entries, COL)
    print(f'wrote {a.out}_fidelity.png' + (', _latency.png, _energy.png' if a.efficiency else ''))


if __name__ == '__main__':
    main()
