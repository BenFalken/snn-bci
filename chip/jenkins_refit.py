"""
Refit a motor Speck SNN's host readout to the chip's firing, from TRAINING trials only, and score it on the real
chip's test output (the speech W1 recipe, for motor).

The deployed readout is fixed: the 36 output neurons' spikes, smoothed by two EMA stages, decoded per axis as a
rate-weighted population vector. The refit readout reads the same smoothed spikes (each axis's share per neuron
plus its log total, as diagnose_speck's refit) through a linear map, optionally from a few past steps too (lags, in
4 ms steps: room to absorb the chip's late spikes). It is fitted by ridge regression on the training trials' spikes,
from either source:

    sim     the chip model (snndec.chip.run_events at the session's layer-1 factor): free, on this computer
    chip    the devkit's spikes for the same training trials (--train-chip FILE.npz from motor_train_chip.py)

the ridge strength chosen by leave-trials-out cross-validation within the training trials; then applied once to the
devkit's test output (the final run's speck_counts) and scored against the trained readout's RMSE on the same rows.

    python chip/jenkins_refit.py --session SESSION_DIR --ckpt CKPT --scalers velocity_scalers.json \
        --arrays final_v2/jenkins_m1/sessions/<session>_arrays.npz --scale 3 [--train-chip counts.npz] --out OUT.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402

RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
WARMUP = 25


def smooth(counts, d, stages):
    out = counts.astype(np.float64)
    for _ in range(stages):
        e, acc = np.empty_like(out), np.zeros(out.shape[1])
        for t in range(len(out)):
            acc = d * acc + out[t]
            e[t] = acc
        out = e
    return out


def feats(sm, lags):
    n = sm.shape[1] // 2
    f = []
    for ax in (sm[:, :n], sm[:, n:]):
        tot = ax.sum(1, keepdims=True)
        f += [np.divide(ax, tot, out=np.zeros_like(ax), where=tot > 0), np.log1p(tot)]
    f = np.hstack(f)
    return np.hstack([np.vstack([np.zeros((L, f.shape[1])), f[:len(f) - L]]) if L else f for L in lags])


def fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = np.inf
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    R = lam * len(A) * np.eye(A.shape[1])
    R[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ Y)
    g = lambda Z: np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w
    g.params = dict(mu=mu, sd=sd, w=w)
    return g


def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--scalers', required=True)
    p.add_argument('--arrays', required=True, help="the final run's <session>_arrays.npz (speck_counts, y_true)")
    p.add_argument('--scale', type=float, required=True)
    p.add_argument('--train-chip', default=None, help='devkit spikes of the training trials (motor_train_chip.py)')
    p.add_argument('--lags', type=int, nargs='+', default=[0, 2, 4, 6])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load(a.ckpt)
    model.eval()
    d, stages = float(torch.sigmoid(model.readout.decay_raw).mean()), model.readout.stages
    scale = data.velocity_scaler(a.session, a.scalers)
    trials = data.motor_split(a.session, 'train', a.scalers)
    net = chip.discretized(chip.linear_stack(model, a.scale), model.host.n_out)
    tc = np.load(a.train_chip) if a.train_chip else None
    src = {'sim': [], 'chip': []} if tc is not None else {'sim': []}
    ys = []
    for i, t in enumerate(trials):
        model(torch.as_tensor(t.x)[None])
        ev = chip.chip_events(model, model.input_events[0].numpy())
        src['sim'].append(np.asarray(chip.run_events(net, ev)))
        if tc is not None:
            src['chip'].append(tc[str(i)].astype(np.float64))
        ys.append(data.unscale_velocity(t.y, *scale))
    z = np.load(a.arrays)
    test_counts, y_test = z['speck_counts'].astype(np.float64), z['y_true'].astype(np.float64)
    res = {'session': os.path.basename(a.session), 'scale': a.scale, 'lags': a.lags, 'train_trials': len(trials),
           'rmse_chip_trained_readout': rmse(z['pred_speck'], y_test)}
    sm_test = smooth(test_counts, d, stages)
    preds, params = {}, {}
    for name, counts in src.items():
        for lags in ([0], a.lags):
            Xs = [feats(smooth(c, d, stages), lags)[WARMUP:] for c in counts]
            Ys = [y[WARMUP:] for y in ys]
            folds = np.array_split(np.arange(len(Xs)), 5)
            cv = {}
            for lam in RIDGES:
                err = []
                for f in folds:
                    tr = [j for j in range(len(Xs)) if j not in f]
                    g = fit(np.vstack([Xs[j] for j in tr]), np.vstack([Ys[j] for j in tr]), lam)
                    err += [((g(Xs[j]) - Ys[j]) ** 2).ravel() for j in f]
                cv[lam] = float(np.sqrt(np.concatenate(err).mean()))
            lam = min(cv, key=cv.get)
            g = fit(np.vstack(Xs), np.vstack(Ys), lam)
            key = f"{name}_lags{'-'.join(map(str, lags))}"
            pred = g(feats(sm_test, lags))
            res[key] = {'ridge': lam, 'cv_rmse_train': cv[lam], 'rmse_chip_test': rmse(pred, y_test),
                        'cc_x': float(np.corrcoef(pred[:, 0], y_test[:, 0])[0, 1]),
                        'cc_y': float(np.corrcoef(pred[:, 1], y_test[:, 1])[0, 1])}
            preds[key] = pred
            params[key] = dict(**g.params, lags=np.array(lags), decay=d, stages=stages)
            print(f"{res['session'][:30]} {key:18s} train CV {cv[lam]:6.2f} | chip test RMSE {res[key]['rmse_chip_test']:6.2f} "
                  f"(trained readout {res['rmse_chip_trained_readout']:.2f})", flush=True)
    # the variant kept: the lower cross-validated error on the training trials (never the test score)
    keys = [k for k in res if isinstance(res[k], dict) and 'cv_rmse_train' in res[k] and k.startswith('chip' if tc is not None else 'sim')]
    best = min(keys, key=lambda k: res[k]['cv_rmse_train'])
    res['chosen'] = best
    res['rmse_chip_trained_readout_cc'] = [float(np.corrcoef(z['pred_speck'][:, i], y_test[:, i])[0, 1]) for i in (0, 1)]
    np.save(a.out.replace('.json', '_pred.npy'), preds[best].astype(np.float32))
    sdv = params[best]['sd']                         # the deployed readout (motor_chip.py --readout)
    np.savez(a.out.replace('.json', '_readout.npz'), **{**params[best], 'sd': np.where(np.isinf(sdv), 0.0, sdv)})
    print(f"chosen on training CV: {best}: chip test RMSE {res[best]['rmse_chip_test']:.2f}")
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
