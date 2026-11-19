"""
Score the GRU references (Oscar results, copied to the Mac) on the rows every other motor decoder was scored on.

    python score_gru.py [--gru ~/Downloads/gru_reference] [--final ~/Downloads/final_v2]

Our protocol: each session's test predictions are cut to results/final_v2's scored rows [start_raw, end_raw)
(their y_true must match, which is checked), scored with decoder_eval.decoder_metrics, and written beside
combined_metrics.json as gru_metrics.json (nothing existing is changed): per session, per model, per seed and the
seed mean. Prints the six NeuroBench sessions' R^2 for both protocols against the published numbers.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decoder_eval import decoder_metrics  # noqa: E402

PUBLISHED = {  # Wang et al. 2024, Table III (5 seeds): vanilla GRU, AEGRU
    'indy_20160622_01': (0.74, 0.76), 'indy_20160630_01': (0.64, 0.65), 'indy_20170131_02': (0.75, 0.76),
    'loco_20170210_03': (0.61, 0.68), 'loco_20170215_02': (0.60, 0.62), 'loco_20170301_05': (0.66, 0.70)}


def main(a):
    nb = {}
    for f in glob.glob(os.path.join(a.gru, 'neurobench', '*.json')):
        r = json.load(open(f)); nb.setdefault((r['session'], r['config']['model']), []).append(r['test']['r2'])
    out_all = {}
    for subj in ('indy', 'loco'):
        cm = json.load(open(os.path.join(a.final, subj, 'combined_metrics.json')))
        out = {}
        for s in sorted(cm):
            meta = json.load(open(os.path.join(a.final, subj, 'sessions', f'{s}.json')))['full']
            arr = np.load(os.path.join(a.final, subj, 'sessions', f'{s}_arrays.npz'))
            lo, hi = meta['start_raw'], meta['end_raw']
            for model in ('aegru', 'gru'):
                runs = sorted(glob.glob(os.path.join(a.gru, 'ours', f'{s}__ours_{model}_*.npz')))
                if not runs:
                    continue
                per_seed = {}
                for f in runs:
                    z = np.load(f)
                    assert np.allclose(z['y_true'][lo:hi], arr['y_true']), f"{f}: test rows do not match final_v2"
                    per_seed[f.split('_s')[-1][:-4]] = decoder_metrics(arr['y_true'], z['y_pred'][lo:hi])
                keys = [k for k, v in next(iter(per_seed.values())).items() if isinstance(v, float)]
                mean = {k: float(np.mean([m[k] for m in per_seed.values()])) for k in keys}
                out.setdefault(s, {})[model] = {'seed_mean': mean, 'seeds': per_seed}
        json.dump(out, open(os.path.join(a.final, subj, 'gru_metrics.json'), 'w'), indent=1)
        out_all.update(out)

    print(f"{'session':18s} | published GRU AEGRU | their protocol GRU AEGRU | our protocol GRU AEGRU LSNN")
    lsnn = {}
    for subj in ('indy', 'loco'):
        cm = json.load(open(os.path.join(a.final, subj, 'combined_metrics.json')))
        lsnn.update({s: (v['snn']['r2_x'] + v['snn']['r2_y']) / 2 for s, v in cm.items()})
    r2 = lambda s, m: (lambda d: (d['r2_x'] + d['r2_y']) / 2)(out_all[s][m]['seed_mean']) if m in out_all.get(s, {}) else np.nan
    for s, (pg, pa) in PUBLISHED.items():
        th = [np.mean(nb.get((s, m), [np.nan])) for m in ('gru', 'aegru')]
        print(f"{s:18s} |        {pg:.2f}  {pa:.2f} |             {th[0]:.3f} {th[1]:.3f} |"
              f"          {r2(s, 'gru'):.3f} {r2(s, 'aegru'):.3f} {lsnn[s]:.3f}")
    for m in ('gru', 'aegru'):
        v = [r2(s, m) for s in out_all if m in out_all[s]]
        print(f"our protocol, {m}: mean R2 over {len(v)} sessions {np.mean(v):.3f}" if v else f"no {m} results")
    print(f"LSNN mean R2 over {len(lsnn)} sessions {np.mean(list(lsnn.values())):.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gru', default=os.path.expanduser('~/Downloads/gru_reference'))
    p.add_argument('--final', default=os.path.expanduser('~/Downloads/final_v2'))
    main(p.parse_args())
