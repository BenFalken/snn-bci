#compares runs epoch by epoch from their training logs: test phoneme error per epoch side by side, each run's
#difference from a reference run at the same epoch, and optionally a plot
#
#   python compare_runs.py C_lsnn_nodelay96 Q_adapt2s Q_adaptrange Q_frac50 --ref C_lsnn_nodelay96 --every 2
#   python compare_runs.py C_lsnn_nodelay96 Q_adapt2s Q_adaptrange --ref C_lsnn_nodelay96 --plot lsnn_trend.png
import argparse
import glob
import re

EPOCH = re.compile(r'^Epoch (\d+)/(\d+) .*?test PER ([0-9.]+)')


def find_log(name, logs):
    """The SNN_RUN log of a run (the line '[i] group/name: ...' names it); the newest if it ran more than once."""
    hits = []
    for f in glob.glob(f'{logs}/SNN_RUN_*_[0-9]*.out'):
        with open(f, errors='replace') as fh:
            head = fh.read(4000)
        if re.search(rf'^\[\d+\] [A-Za-z0-9_]+/{re.escape(name)}:', head, re.M):
            hits.append(f)
    return sorted(hits)[-1] if hits else None


def read_per(path):
    per = {}
    with open(path, errors='replace') as fh:
        for line in fh:
            m = EPOCH.match(line)
            if m:
                per[int(m.group(1))] = float(m.group(3))   # a resumed run repeats epochs: the last value wins
    return per


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--ref', default=None, help='run to subtract at each epoch (default: the first)')
    p.add_argument('--logs', default='snn_logs')
    p.add_argument('--every', type=int, default=1, help='print every n-th epoch (and always the last)')
    p.add_argument('--plot', default=None, help='also save a plot of test PER against epoch here')
    a = p.parse_args()
    ref = a.ref or a.runs[0]
    data = {}
    for r in a.runs:
        f = find_log(r, a.logs)
        if f is None:
            print(f'{r}: no log found (not started?)')
            continue
        data[r] = read_per(f)
        last = max(data[r]) if data[r] else 0
        print(f'{r}: {f}, {len(data[r])} epochs, last {last}' + (f', PER {data[r][last]:.4f}' if last else ''))
    runs = [r for r in a.runs if r in data and data[r]]
    if not runs:
        return
    epochs = sorted(set().union(*(data[r] for r in runs)))
    last = epochs[-1]
    others = [r for r in runs if r != ref and ref in data]
    w = max(10, max(len(r) for r in runs) + 1)
    print('\ntest PER by epoch' + (f'; Δ = run − {ref} at the same epoch (negative = better)' if others else ''))
    print(f'{"epoch":>5} ' + ''.join(f'{r:>{w}}' for r in runs) + ''.join(f'{"Δ " + r[:w - 3]:>{w}}' for r in others))
    for e in epochs:
        if e % a.every and e != last and e != 1:
            continue
        row = ''.join(f'{data[r][e]:>{w}.4f}' if e in data[r] else f'{"":>{w}}' for r in runs)
        row += ''.join(f'{data[r][e] - data[ref][e]:>+{w}.4f}' if e in data[r] and e in data[ref] else f'{"":>{w}}'
                       for r in others)
        print(f'{e:>5} {row}')
    print('\nbest so far: ' + ', '.join(f'{r} {min(data[r].values()):.4f} (epoch {min(data[r], key=data[r].get)})'
                                         for r in runs))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5))
        for r in runs:
            es = sorted(data[r])
            ax.plot(es, [data[r][e] for e in es], label=r, lw=2 if r == ref else 1.5)
            if r != ref and ref in data:
                common = [e for e in es if e in data[ref]]
                bx.plot(common, [data[r][e] - data[ref][e] for e in common], label=r)
        ax.set_xlabel('epoch'); ax.set_ylabel('test phoneme error'); ax.set_ylim(0.2, 0.5); ax.legend()
        bx.axhline(0, color='k', lw=1); bx.set_xlabel('epoch'); bx.set_ylabel(f'difference from {ref}'); bx.legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print('wrote', a.plot)


if __name__ == '__main__':
    main()
