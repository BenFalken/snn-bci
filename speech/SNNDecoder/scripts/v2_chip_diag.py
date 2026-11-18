"""
V2 on the devkit, second diagnostic. The first (results/V2_diag) found two faults:

  1. Both chains on one chip (MultiDevkit): chain B's cores ran at samna's defaults (threshold 0, no destination),
     because whole-layer assignment into the merged configuration does not persist on this samna. Fixed in
     speck_io (copy_layer, setting by setting, verified before the configuration is applied).
  2. Each chain alone: the chip's first 1-2 s of every sentence came out almost silent, then in a burst. V2's chains
     use cores 0 and 1 (64K neurons each), and resetting every neuron there before each sentence may still be
     draining when the sentence starts.

This script (a) runs chain A alone under four resets (whole neuron memory or used neurons only; 0.3 s or 2 s
settle) and reports chain A's spikes by quarter of the sentence and their agreement with the chip model, picks the
reset whose first quarter best matches the model; then (b) runs both chains on one chip with that reset and the fixed
merge, and reports spikes per layer, phoneme error and logit r against the chip model (sim) as before.

    python3 scripts/v2_chip_diag.py --ckpt V2.pth --data-dir data --scale 2.5 --out results/V2_diag2/diag.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speech_chip as sc  # noqa: E402
import speck_io  # noqa: E402
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode  # noqa: E402


def quarters(x):
    q = len(x) // 4
    return [float(x[j * q:(j + 1) * q].sum()) for j in range(4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--scale', type=float, default=2.5)
    p.add_argument('--sessions', nargs='+', default=['t12.2022.05.24', 't12.2022.07.27'])
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--wait', type=float, default=0.003)
    p.add_argument('--devkit', default='speck2fdevkit:0')
    p.add_argument('--out', default='results/V2_diag2/diag.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    model, ck = sc.load_snn_model(a.ckpt)
    S = list(ck['sessions'])
    CH = sc.chains(model)
    blank = model.n_classes - 1
    widths = [[len(u) for u in c[1]] for c in CH]
    trials = []
    for sess in a.sessions:
        part = load_partitions(a.data_dir, [sess], 'test', [S.index(sess)])[0]
        for i in range(min(a.trials, len(part))):
            trials.append((sess, i, sc.host_events(model, part.features[i], part.day_idx), part.labels[i],
                           sc.reference_logits(model, part.features[i], part.day_idx)))
    nets = [speck_io.discretized(sc.spiking_stack(model, 1.0, chain=c), len(c[0])) for c in CH]
    sim = [[sc.run_events(n, ev[:, c[0]], a.scale, True) for n, c in zip(nets, CH)] for _, _, ev, _, _ in trials]
    offA = np.cumsum([0] + widths[0])
    sl = lambda x, l: x[:, offA[l]:offA[l + 1]].sum(1)
    report = dict(widths=widths, sentences=[(s, i) for s, i, *_ in trials])

    # (a) chain A alone, four resets
    print('(a) chain A alone: layer-1 spikes by quarter of the sentence (summed over sentences) and agreement with '
          'the chip model (per-bin counts, layers 1-3)')
    simq = np.sum([quarters(sl(r[0], 0)) for r in sim], 0)
    print(f'  {"chip model":24s} Q1-Q4 ' + ' '.join(f'{v:7.0f}' for v in simq))
    variants = {}
    for used in (False, True):
        for settle in (0.3, 2.0):
            speck_io.RESET_USED_ONLY, speck_io.RESET_SETTLE_S = used, settle
            dev = speck_io.Devkit(sc.spiking_stack(model, a.scale, chain=CH[0]), len(CH[0][0]), devkit=a.devkit,
                                  wait_time=a.wait, all_layers=True)
            runs = []
            for _, _, ev, _, _ in trials:
                dev.reset()
                runs.append(dev.run(ev[:, CH[0][0]], CH[0][1][-1].size)[0])
            dev.close()
            del dev
            q = np.sum([quarters(sl(r, 0)) for r in runs], 0)
            r_ = float(np.mean([np.mean([np.corrcoef(sl(c, l), sl(s_[0], l))[0, 1] for l in range(3)])
                                for c, s_ in zip(runs, sim)]))
            name = f'{"used neurons" if used else "all neurons"}, {settle:g} s'
            variants[name] = dict(used=used, settle=settle, quarters=q.tolist(), agreement=r_,
                                  q1_ratio=float(q[0] / max(simq[0], 1)))
            print(f'  {name:24s} Q1-Q4 ' + ' '.join(f'{v:7.0f}' for v in q) + f'   agreement r {r_:.3f}')
    best = max(variants, key=lambda k: (variants[k]['agreement'], -abs(1 - variants[k]['q1_ratio'])))
    print(f'  -> best reset: {best}')
    report['reset_variants'], report['best_reset'] = variants, best
    speck_io.RESET_USED_ONLY, speck_io.RESET_SETTLE_S = variants[best]['used'], variants[best]['settle']

    # (b) both chains on one chip, fixed merge, best reset
    dev = speck_io.MultiDevkit([sc.spiking_stack(model, a.scale, chain=c) for c in CH], [len(c[0]) for c in CH],
                               devkit=a.devkit, wait_time=a.wait, all_layers=True)
    multi = []
    for _, _, ev, _, _ in trials:
        dev.reset()
        multi.append(dev.run([ev[:, c[0]] for c in CH])[0])
    dev.close()
    print('\n(b) both chains on one chip: mean spikes per bin (A1 A2 A3 | B1 B2 B3)')
    table = {}
    for name, res in (('chip model', sim), ('one chip', multi)):
        row = []
        for k, wk in enumerate(widths):
            off = np.cumsum([0] + wk)
            row += [float(np.mean([r[k][:, off[l]:off[l + 1]].sum(1).mean() for r in res])) for l in range(len(wk))]
        table[name] = row
        print(f'  {name:10s} ' + ' '.join(f'{v:7.1f}' for v in row))
    scores = {}
    for name, res in (('chip model', sim), ('one chip', multi)):
        e = n_ = 0
        rr, qq = [], []
        for (sess, i, ev, y, ref), counts in zip(trials, res):
            lg = sc.readout(model, sc.assemble(model, [np.asarray(c, np.float32) for c in counts], True))
            n = min(len(lg), len(ref))
            e += edit_distance(greedy_decode(lg[:n], blank), y)
            n_ += len(y)
            rr.append(float(np.corrcoef(lg[:n].ravel(), ref[:n].ravel())[0, 1]))
            qq.append(sc.quarters(lg[:n], ref[:n]))
        scores[name] = dict(per=e / n_, logit_r=float(np.mean(rr)), by_quarter=np.nanmean(qq, 0).tolist())
        print(f'  {name:10s} PER {e / n_:.3f}  logit r {np.mean(rr):.3f}  by quarter '
              + ' '.join(f'{v:.2f}' for v in scores[name]['by_quarter']))
    report.update(spikes_per_bin=table, scores=scores)
    json.dump(report, open(a.out, 'w'), indent=1)
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'best_reset.sh'), 'w') as f:
        f.write(f"RESET_SETTLE={variants[best]['settle']}\nRESET_USED={'--reset-used' if variants[best]['used'] else ''}\n")
    np.savez_compressed(a.out.replace('.json', '_counts.npz'),
                        **{f'{name}/{s}/{i}/{k}': np.asarray(c[k], np.uint16)
                           for name, res in (('sim', sim), ('multi', multi))
                           for (s, i, *_), c in zip(trials, res) for k in range(len(CH))})
    print('wrote', a.out)


if __name__ == '__main__':
    main()
