"""Learning curves of every run in the logs: fit PER against epoch, and find the earliest epoch at which runs can be
compared, for short "speed-run" sweeps.

What it does:
  1. Reads every training log (snn_logs/*.out: SNN runs and GRU controls), one run per log, keeping the longest log
     when a run was restarted. Each run's schedule is noted: with the RNN recipe the learning rate falls linearly to
     0 over the run's own number of epochs E, so a 36- and a 96-epoch run are on different schedules.
  2. Fits each run's test PER curve with:
        power         a * x^-b
        power+floor   a * x^-b + c          (c: the PER it levels off at)
        exp+floor     a * exp(-b x) + c
        power(lr)     a * s^-b + c, with s the learning rate summed so far (x - x^2 / 2E epochs at full rate):
                      the same curve family on the "training done" axis, which puts runs of different lengths
                      on one footing
     and compares them by error and AIC.
  3. Horizon: among runs with the same schedule length E, how well the ranking at epoch k agrees with the final
     ranking (Spearman), how often the final best few are already the best few at k, and how far a curve fitted to
     epochs 1..k misses the final PER. The earliest k where the ranking is reliable is a candidate comparison point.

Caveat, written into the report: truncating a run at epoch k is not the same as a short run. A run cut at k is still
at a high learning rate; a short schedule (say 12 epochs decaying to 0) ends with the annealing drop. To speed-run a
sweep, the safer design is a short full schedule, checked once against 36-epoch results.

example usage (on Oscar):
    python learning_curves.py --logs 'snn_logs/*.out' --out $BASE_DIR/derived/curves/curves
writes <out>.md, <out>_curves.png, <out>_fits.png, <out>_horizon.png, <out>.csv
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HEADER = re.compile(r'^\[(\d+)\] (\S+?):(.*)$')
EPOCH = re.compile(r'^Epoch (\d+)/(\d+) \([\d.]+s\) \| train loss ([\d.]+) \| \w+ loss ([\d.]+) \| \w+ PER ([\d.]+)')


# ---------------------------------------------------------------------------------------------- reading
def read_log(path):
    """-> list of runs in the file: dict(name, group, flags, epochs (E), rnn_recipe, x, per, train, test)."""
    runs, cur = [], None
    for line in open(path, errors='ignore'):
        m = HEADER.match(line)
        if m:
            full = m.group(2)
            group, name = full.split('/', 1) if '/' in full else ('controls', full)
            if cur is None or cur['full'] != full:
                cur = {'full': full, 'group': group, 'name': name, 'flags': m.group(3), 'file': path,
                       'rnn_recipe': False, 'rows': {}}
                runs.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith('RNN recipe'):
            cur['rnn_recipe'] = True
        m = EPOCH.match(line)
        if m:
            ep, E = int(m.group(1)), int(m.group(2))
            cur['E'] = E
            cur['rows'][ep] = (float(m.group(5)), float(m.group(3)), float(m.group(4)))   # PER, train, test
    out = []
    for r in runs:
        if not r['rows']:
            continue
        eps = sorted(r['rows'])
        r['x'] = np.array(eps, float)
        r['per'] = np.array([r['rows'][e][0] for e in eps])
        r['train'] = np.array([r['rows'][e][1] for e in eps])
        r['test'] = np.array([r['rows'][e][2] for e in eps])
        if 'gru' in r['full'].lower() or r['group'] == 'controls':
            r['rnn_recipe'] = True                      # the GRU controls use the RNN recipe's linear decay
        out.append(r)
    return out


def collect(pattern, min_epochs):
    best = {}
    for path in sorted(glob.glob(pattern)):
        for r in read_log(path):
            key = r['full']
            if len(r['x']) < min_epochs:
                continue
            if key not in best or len(r['x']) > len(best[key]['x']):
                best[key] = r
    return list(best.values())


def finished(r):
    return r['x'][-1] >= r['E']


# ---------------------------------------------------------------------------------------------- fitting
def lr_sum(x, E):
    """Learning rate summed over epochs 0..x for a linear decay to 0 at E (in epochs at full rate)."""
    x = np.minimum(x, E)
    return x - x ** 2 / (2.0 * E)


MODELS = {
    'power':       (lambda x, a, b: a * x ** -b, (0.5, 0.2)),
    'power+floor': (lambda x, a, b, c: a * x ** -b + c, (0.5, 0.5, 0.25)),
    'exp+floor':   (lambda x, a, b, c: a * np.exp(-b * x) + c, (0.5, 0.2, 0.25)),
}


def fit(model, x, y):
    """Least squares; -> (params, rmse) or (None, inf)."""
    from scipy.optimize import curve_fit
    f, p0 = MODELS[model]
    try:
        bounds = ([0] * len(p0), [np.inf] * len(p0))
        p, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return p, float(np.sqrt(np.mean((f(x, *p) - y) ** 2)))
    except Exception:
        return None, math.inf


def aic(rmse, n, k):
    return n * math.log(max(rmse, 1e-12) ** 2) + 2 * k


def axis_of(r, kind):
    """x for the fit: epochs, or the summed learning rate for the RNN recipe."""
    return lr_sum(r['x'], r['E']) if kind == 'lr' and r['rnn_recipe'] else r['x']


def fit_run(r):
    res = {}
    for m in MODELS:
        p, e = fit(m, r['x'], r['per'])
        res[m] = (p, e)
    if r['rnn_recipe']:
        p, e = fit('power+floor', np.maximum(axis_of(r, 'lr'), 1e-3), r['per'])
        res['power(lr)'] = (p, e)
    return res


# ---------------------------------------------------------------------------------------------- horizon
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def horizon(runs, E, top=3):
    """Per epoch k: Spearman(PER at k, final PER), top-`top` overlap, and the median miss of a power+floor fit
    (on summed learning rate when the recipe decays) from epochs 1..k extrapolated to E."""
    group = [r for r in runs if r['E'] == E and finished(r) and len(r['x']) == E]
    if len(group) < 4:
        return None, group
    final = np.array([r['per'][-1] for r in group])
    best_final = set(np.argsort(final)[:top])
    rows = []
    for k in range(2, E + 1):
        at_k = np.array([r['per'][k - 1] for r in group])
        rho = spearman(at_k, final)
        overlap = len(best_final & set(np.argsort(at_k)[:top])) / top
        misses = []
        if k >= 5:
            for r in group:
                xs = axis_of(r, 'lr')
                p, _ = fit('power+floor', np.maximum(xs[:k], 1e-3), r['per'][:k])
                if p is not None:
                    pred = MODELS['power+floor'][0](max(xs[-1], 1e-3), *p)
                    misses.append(abs(pred - r['per'][-1]))
        close = [(i, j) for i in range(len(group)) for j in range(i + 1, len(group))
                 if 0.005 <= abs(final[i] - final[j]) <= 0.03]
        pairs = (sum((at_k[i] - at_k[j]) * (final[i] - final[j]) > 0 for i, j in close) / len(close)) if close \
            else math.nan
        rows.append((k, rho, overlap, float(np.median(misses)) if misses else math.nan, pairs, len(close)))
    return rows, group


# ---------------------------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='snn_logs/*.out')
    ap.add_argument('--out', default='curves')
    ap.add_argument('--min-epochs', type=int, default=5, help='ignore runs with fewer logged epochs')
    ap.add_argument('--max-per', type=float, default=1.0, help='ignore runs whose final PER is above this '
                    '(diverged or broken runs)')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    runs = [r for r in collect(a.logs, a.min_epochs) if r['per'][-1] <= a.max_per]
    if not runs:
        raise SystemExit(f'no runs found in {a.logs}')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})

    with open(a.out + '.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'group', 'E', 'rnn_recipe', 'epoch', 'test_per', 'train_loss', 'test_loss'])
        for r in runs:
            for i, x in enumerate(r['x']):
                w.writerow([r['full'], r['group'], r['E'], r['rnn_recipe'], int(x), r['per'][i], r['train'][i],
                            r['test'][i]])

    # 1. all curves, one panel per schedule length
    lengths = sorted({r['E'] for r in runs}, key=lambda E: -sum(r['E'] == E for r in runs))[:4]
    fig, axs = plt.subplots(1, len(lengths), figsize=(5.2 * len(lengths), 4.2), squeeze=False)
    for ax, E in zip(axs[0], lengths):
        for r in runs:
            if r['E'] == E:
                ax.plot(r['x'], r['per'], lw=0.9, alpha=0.6)
        ax.set_title(f'{E}-epoch schedule ({sum(r["E"] == E for r in runs)} runs)')
        ax.set_xlabel('epoch')
        ax.set_ylabel('test PER')
        ax.set_ylim(0.2, min(1.0, max(r['per'][min(2, len(r['per']) - 1)] for r in runs if r['E'] == E) + 0.05))
    fig.tight_layout()
    fig.savefig(a.out + '_curves.png', dpi=180)
    plt.close(fig)

    # 2. fits
    fin = [r for r in runs if finished(r) and len(r['x']) >= 10]
    table = defaultdict(list)
    for r in fin:
        r['fits'] = fit_run(r)
        n = len(r['x'])
        for m, (p, e) in r['fits'].items():
            if p is not None:
                table[m].append((e, aic(e, n, len(p))))
    wins = defaultdict(int)
    for r in fin:
        scored = {m: aic(e, len(r['x']), len(p)) for m, (p, e) in r['fits'].items() if p is not None}
        if scored:
            wins[min(scored, key=scored.get)] += 1

    examples = sorted(fin, key=lambda r: r['per'][-1])[:: max(1, len(fin) // 6)][:6]
    if examples:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for ax, r in zip(axs.flat, examples):
            ax.plot(r['x'], r['per'], 'o', ms=3, color='black', label='test PER')
            xx = np.linspace(r['x'][0], r['x'][-1], 200)
            for m, ls in (('power', ':'), ('power+floor', '--'), ('exp+floor', '-.')):
                p, e = r['fits'][m]
                if p is not None:
                    ax.plot(xx, MODELS[m][0](xx, *p), ls, lw=1.3, label=f'{m} ({e:.4f})')
            if 'power(lr)' in r['fits'] and r['fits']['power(lr)'][0] is not None:
                p, e = r['fits']['power(lr)']
                ax.plot(xx, MODELS['power+floor'][0](np.maximum(lr_sum(xx, r['E']), 1e-3), *p), '-', lw=1.3,
                        label=f'power(lr) ({e:.4f})')
            ax.set_title(f"{r['full']} (E={r['E']})", fontsize=10)
            ax.set_xlabel('epoch')
            ax.set_ylabel('test PER')
            ax.legend(fontsize=7, frameon=False)
        for ax in list(axs.flat)[len(examples):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(a.out + '_fits.png', dpi=180)
        plt.close(fig)

    # 3. horizon
    L = ['# Learning curves', '',
         f'{len(runs)} runs read from `{a.logs}` ({len(fin)} finished with 10+ epochs). Schedule lengths: ' +
         ', '.join(f'{E} epochs ({sum(r["E"] == E for r in runs)} runs)' for E in lengths) + '.', '',
         '## Which function fits', '',
         '| Function | Runs fitted | Median RMSE (PER) | Best by AIC |', '|---|---|---|---|']
    for m in ('power', 'power+floor', 'exp+floor', 'power(lr)'):
        if table[m]:
            L.append(f'| {m} | {len(table[m])} | {np.median([e for e, _ in table[m]]):.4f} | {wins[m]} |')
    L += ['', 'power: a·x^-b; power+floor: a·x^-b + c; exp+floor: a·e^(-bx) + c; power(lr): a·s^-b + c with s the '
          'learning rate summed so far (runs on the RNN recipe, whose rate falls linearly to 0 at the last epoch).', '']
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for E in lengths:
        rows, group = horizon(runs, E)
        if rows is None:
            continue
        ks, rho, ov, miss, pairs, n_close = zip(*rows)
        ax.plot(ks, rho, lw=1.8, label=f'{E}-epoch runs (n={len(group)}): Spearman with final')
        ax.plot(ks, pairs, lw=1.2, ls='--', label=f'{E}-epoch: close pairs (0.005-0.03 apart) in order')
        reliable = next((k for k, r_, *_ in rows if all(rr[1] >= 0.9 for rr in rows if rr[0] >= k)), None)
        close_ok = next((k for k, *_ in rows if all(rr[4] >= 0.85 for rr in rows if rr[0] >= k)), None)
        L += [f'## Horizon, {E}-epoch runs ({len(group)} finished)', '',
              '| Epoch k | Spearman (rank at k vs final) | Final top 3 already top 3 | '
              f'Close pairs in final order ({n_close[0]} pairs) | Median miss of fit to 1..k |',
              '|---|---|---|---|---|']
        for k, r_, o, mm, pp, _n in rows:
            if k in (3, 5, 8, 10, 12, 15, 18, 20, 24, 27, 30, 33, E):
                L.append(f'| {k} | {r_:.2f} | {o:.0%} | {"" if math.isnan(pp) else f"{pp:.0%}"} | '
                         f'{"" if math.isnan(mm) else f"{mm:.4f}"} |')
        L += ['', f'Earliest epoch from which the ranking stays at Spearman ≥ 0.9: '
              f'**{reliable if reliable else "not reached"}** of {E}. Earliest from which at least 85% of close '
              f'pairs (final PERs 0.005-0.03 apart) stay in their final order: '
              f'**{close_ok if close_ok else "not reached"}** of {E}.', '']
    ax.axhline(0.9, color='0.7', lw=0.8)
    ax.set_xlabel('epoch k')
    ax.set_ylabel('agreement with the final ranking')
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out + '_horizon.png', dpi=180)
    plt.close(fig)
    L += ['## Caveat', '',
          'A run cut at epoch k is still at a high learning rate; a short run whose rate decays to 0 at k ends with '
          'the annealing drop and usually scores better than the same run cut at k. The ranking agreement above is '
          'for cutting long runs. For speed-run sweeps, a short full schedule is the cleaner design: validate it '
          'once by re-running a few known 36-epoch configurations at the short length and checking the ranking.']
    open(a.out + '.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'wrote {a.out}.md, .csv, _curves.png, _fits.png, _horizon.png')


if __name__ == '__main__':
    main()
