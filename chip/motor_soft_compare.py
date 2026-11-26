"""
EXPLORATORY (2026-11-26): the motor Speck SNN with soft reset and a membrane floor of -3 (speck_delay_soft) against
the final hard-reset one (speck_delay), both selected on the validation split, compared as they would run on the chip.

Per session:
  1. the soft network's layer-1 factor, chosen without labels as the hard network's was: chip/motor_l1_sweep.py on the
     first 20 training trials, scales 0.5-3, max agreement with the trained network (the hard network's factor is read
     from motor/speck_l1_factors_val.txt, or for monkey J from its l1_sweep json)
  2. both networks in the chip model (snndec.chip.run_events) at their factors on the VALIDATION trials, the last
     10% of the session's training trials that training held out to choose the checkpoint, scored by the paper's
     metric (velocity correlation, mean of x and y, over the validation trials concatenated), read from the dataset
     training used (mua_8_group for monkeys I and L, mua for J), so they are training's own validation trials, all of
     them (on Oscar: the laptop's copies hold only 20 training trials); the trained readout,
     as deployed for monkeys I and L (monkey J's refit is a later step for whichever network is kept)

Decision rule, fixed before the scores: soft replaces hard only if its mean validation correlation is higher for
Motor (I) (indy + loco, 45 sessions) and for Motor (II) (jenkins_m1, 4 sessions). Test is not read here.

    python chip/motor_soft_compare.py --subject indy --hard-root ~/Downloads/motor_val/per_session_delay_binarized_val \
        --soft-root ~/Downloads/motor_val_soft/indy --out ~/Downloads/motor_soft_compare [--jobs 6]
    python chip/motor_soft_compare.py --summary ~/Downloads/motor_soft_compare
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from multiprocessing import get_context

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'chip'))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402
from motor_l1_sweep import decode  # noqa: E402

SCALERS = os.path.join(ROOT, 'motor/snn_training/velocity_scalers.json')
FACTORS = os.path.join(ROOT, 'motor/speck_l1_factors_val.txt')
SCALES = ['0.5', '0.75', '1', '1.5', '2', '3']
GROUP = {'indy': 'Motor (I)', 'loco': 'Motor (I)', 'jenkins_m1': 'Motor (II)'}


def hard_factor(subject, session, hard_root):
    for line in open(FACTORS):
        f = line.split()
        if len(f) == 3 and f[0] == subject and f[1] == session:
            return float(f[2])
    j = os.path.join(os.path.dirname(hard_root), 'l1_sweep', f'{session}.json')     # monkey J
    if os.path.isfile(j):
        return float(json.load(open(j))['chosen_scale'])
    raise SystemExit(f'no layer-1 factor for the hard network of {subject} {session}')


@torch.no_grad()
def val_corr(ckpt, val_dir, factor):
    model, _ = load(ckpt)
    model.eval()
    trials = data.motor_split(val_dir, 'train', SCALERS)
    n = len(trials)
    val = trials[n - max(1, int(round(0.1 * n))):]
    net = chip.discretized(chip.linear_stack(model, factor), model.host.n_out)
    ys, ts, ysw = [], [], []
    for t in val:
        ysw.append(model(torch.as_tensor(t.x)[None])[0].numpy())
        ev = chip.chip_events(model, model.input_events[0].numpy())
        ys.append(decode(model, chip.run_events(net, ev)))
        ts.append(t.y)
    y, tr, sw = np.concatenate(ys), np.concatenate(ts), np.concatenate(ysw)
    r = lambda p: float(np.mean([np.corrcoef(p[:, k], tr[:, k])[0, 1] for k in range(2)]))
    return {'r_chip': r(y), 'r_software': r(sw), 'val_trials': len(val)}


def one(job):
    subject, session, hard_ckpt, soft_ckpt, session_dir, val_dir, out, hard_root = job
    f = os.path.join(out, subject, f'{session}.json')
    if os.path.isfile(f):
        return json.load(open(f))
    os.makedirs(os.path.join(out, subject, 'l1'), exist_ok=True)
    sw = os.path.join(out, subject, 'l1', f'{session}.json')
    if not os.path.isfile(sw):
        p = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'chip/motor_l1_sweep.py'), session_dir,
                            soft_ckpt, '--scalers', SCALERS, '--scales', *SCALES, '--trials', '20', '--out', sw],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if p.returncode:                       # one session's failure is reported, not fatal to the others
            err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f'exit {p.returncode}'
            print(f'{session}: FACTOR SWEEP FAILED: {err}\n{p.stderr[-2000:]}', flush=True)
            return None
    k_soft = float(json.load(open(sw))['chosen_scale'])
    k_hard = hard_factor(subject, session, hard_root)
    res = {'subject': subject, 'session': session, 'factor_hard': k_hard, 'factor_soft': k_soft,
           'hard': val_corr(hard_ckpt, val_dir, k_hard), 'soft': val_corr(soft_ckpt, val_dir, k_soft)}
    json.dump(res, open(f, 'w'), indent=1)
    print(f"{session}: hard x{k_hard:g} r {res['hard']['r_chip']:.3f} | soft x{k_soft:g} r {res['soft']['r_chip']:.3f}",
          flush=True)
    return res


def summary(out):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, '*', '*.json')))]
    print('Chip model, VALIDATION trials, velocity correlation (mean of x and y); trained readout')
    ok = {}
    for g in ('Motor (I)', 'Motor (II)'):
        rs = [r for r in rows if GROUP[r['subject']] == g]
        if not rs:
            print(f'  {g}: no sessions yet'); ok[g] = False; continue
        h = np.array([r['hard']['r_chip'] for r in rs]); s = np.array([r['soft']['r_chip'] for r in rs])
        hs = np.mean([r['hard']['r_software'] for r in rs]); ss = np.mean([r['soft']['r_software'] for r in rs])
        print(f'  {g:10s} {len(rs):2d} sessions: chip soft {s.mean():.3f} vs hard {h.mean():.3f} '
              f'(median {np.median(s):.3f} vs {np.median(h):.3f}; soft higher in {int((s > h).sum())} of {len(rs)}) '
              f'| software soft {ss:.3f} vs hard {hs:.3f}')
        ok[g] = s.mean() > h.mean()
    n = {g: sum(GROUP[r['subject']] == g for r in rows) for g in ok}
    print('Decision (both groups, all sessions): ' + ('SOFT BEATS HARD' if all(ok.values()) and n['Motor (I)'] == 45
                                                      and n['Motor (II)'] == 4 else
                                                      'soft does not (yet) beat hard' + f' [{n}]'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', choices=list(GROUP))
    p.add_argument('--hard-root')
    p.add_argument('--soft-root')
    p.add_argument('--out', default=None)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--summary', default=None)
    p.add_argument('--data-root', default=None, help='motor data root (default $SNNBCI_MOTOR_DATA)')
    a = p.parse_args()
    if a.summary:
        return summary(a.summary)
    exp = 'hkm' if a.subject == 'jenkins_m1' else 'bmi'
    base = a.data_root or os.environ['SNNBCI_MOTOR_DATA']
    D = os.path.join(base, 'snn_datasets', exp, a.subject, 'mua')                   # the factor sweep's trials
    V = os.path.join(base, 'snn_datasets', exp, a.subject, 'mua' if exp == 'hkm' else 'mua_8_group')   # training's
    jobs = []
    for d in sorted(glob.glob(os.path.join(a.soft_root, '*', 'best_model_weights.pth'))):
        s = os.path.basename(os.path.dirname(d))
        h = os.path.join(a.hard_root, s, 'best_model_weights.pth')
        if os.path.isfile(h):
            jobs.append((a.subject, s, h, d, os.path.join(D, s), os.path.join(V, s), a.out, a.hard_root))
    print(f'{a.subject}: {len(jobs)} sessions with both networks')
    torch.set_num_threads(1)
    with get_context('fork').Pool(a.jobs) as pool:
        done = pool.map(one, jobs)
    failed = [j[1] for j, r in zip(jobs, done) if r is None]
    if failed:
        print(f'{a.subject}: {len(failed)} session(s) failed and are left out: {failed}')
    summary(a.out)


if __name__ == '__main__':
    main()
