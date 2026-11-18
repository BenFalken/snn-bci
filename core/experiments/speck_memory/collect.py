"""
Collect the speck_memory runs into one table, and summarize them.

    python experiments/speck_memory/collect.py --root RUNS_DIR --out results.csv [--memory memory.csv]

results.csv: one row per run (experiment, unit, variant, readout, channels, delays, span_ms, inputs, seed, the test
metric, chip mapping, minutes). memory.csv: the silence response per run (lag_ms, value). Prints, per experiment,
the mean test metric of every variant over units and seeds, and the channel/delay split with the best mean.
"""

import argparse
import glob
import json

import pandas as pd

# the shapes verified to map onto the chip (run.py, CHIP_VERIFIED); runs whose in-job check failed only because its
# samna import broke on the cluster count as mapping when their shape is one of these
CHIP_VERIFIED = {(n_in, tuple([128, 256, 256] + out)) for n_in in (96, 480, 128, 512) for out in ([], [36], [128], [41])}
METRIC = {'motor': ('rmse_test', 'lower'), 'audio': ('r_test', 'higher'), 'speech': ('per_test', 'lower')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--memory', default=None)
    a = p.parse_args()
    rows, mem = [], []
    for f in sorted(glob.glob(f'{a.root}/*/*/*/result.json')):
        r = json.load(open(f))
        key, _ = METRIC[r['experiment']]
        rows.append({k: r[k] for k in ('experiment', 'unit', 'variant', 'readout', 'channels', 'n_delays', 'span_ms',
                                       'inputs', 'seed', 'parameters', 'buffer_values', 'train_minutes')}
                    | {'metric': key, 'value': r['metrics'][key],
                       'chip_maps': r['chip_maps'] is True or (r['chip_inputs'], tuple(r['widths'])) in CHIP_VERIFIED})
        sr = r['silence_response']
        mem += [{'experiment': r['experiment'], 'unit': r['unit'], 'variant': r['variant'], 'seed': r['seed'],
                 'lag_ms': l, 'kernel': k, 'floor': fl, 'sensitivity': s}
                for l, k, fl, s in zip(r['lag_ms'], sr['kernel'], sr['floor'], sr['sensitivity'])]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    if a.memory:
        pd.DataFrame(mem).to_csv(a.memory, index=False)
    print(f'{len(df)} runs -> {a.out}')
    for exp, g in df.groupby('experiment'):
        key, better = METRIC[exp]
        s = g.groupby(['variant', 'readout', 'channels', 'n_delays', 'span_ms'])['value'].agg(['mean', 'std', 'count'])
        print(f'\n== {exp} ({key}, {better} is better)\n{s.round(4).to_string()}')
        sweep = s.reset_index()
        sweep = sweep[sweep.readout == 'hidden']
        sweep = sweep[sweep.n_delays > 1]
        if len(sweep):
            best = sweep.loc[sweep['mean'].idxmin() if better == 'lower' else sweep['mean'].idxmax()]
            print(f"best split: {best.channels} channels x {best.n_delays} delays ({best.span_ms:.0f} ms of memory), "
                  f"{key} {best['mean']:.4f}")


if __name__ == '__main__':
    main()
