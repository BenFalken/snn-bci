"""
A motor Speck SNN on the Speck2f devkit, end to end: the session's test trials streamed one after another (one reset
before the first, as the final runs), the chip's output neurons read back every 4 ms step, and the velocity decoded on
the host, live, two ways:

    trained    the network's own readout: two EMA stages and the population vector of the 36 output neurons
    refitted   with --readout: the readout refitted to the chip's firing on TRAINING trials (jenkins_refit.py ->
               <session>_readout.npz: the same smoothing, each axis's share per neuron and its log total, from a few
               past steps, through a ridge-fitted linear map), the deployed readout for Jenkins

    python chip/motor_chip.py --session SESSION_DIR --ckpt CKPT --scalers velocity_scalers.json \
        --scale 3 [--readout <session>_readout.npz] [--nochip] --out results/<session>.json

--nochip runs the chip model in software (snndec.chip.run_events) instead of the devkit. Scores: RMSE (mm/s, over
both axes) and the correlation per axis, for each readout; for the chip, mean power and time per step. The step-by-step
output goes next to the json as <out>_arrays.npz: counts (the output neurons' spikes), y_true (mm/s), pred_trained,
pred_refit (with --readout) and trial_lens (steps per test trial, in order).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402
from jenkins_refit import feats, smooth  # noqa: E402


@torch.no_grad()
def trained_decode(model, counts):
    st = model.readout.init_state(1, 'cpu')
    return torch.stack([model.readout.step(torch.as_tensor(c, dtype=torch.float32)[None], st)[0]
                        for c in counts]).numpy()


def refit_decode(counts, path):
    r = np.load(path)
    Z = feats(smooth(counts.astype(np.float64), float(r['decay']), int(r['stages'])), [int(x) for x in r['lags']])
    sd = np.where(r['sd'] > 0, r['sd'], np.inf)
    return np.hstack([(Z - r['mu']) / sd, np.ones((len(Z), 1))]) @ r['w']


def scores(pred, y):
    return dict(rmse=float(np.sqrt(((pred - y) ** 2).mean())),
                cc_x=float(np.corrcoef(pred[:, 0], y[:, 0])[0, 1]), cc_y=float(np.corrcoef(pred[:, 1], y[:, 1])[0, 1]))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--scalers', required=True)
    p.add_argument('--scale', type=float, required=True, help="layer 1's threshold factor on the chip")
    p.add_argument('--readout', default=None)
    p.add_argument('--nochip', action='store_true')
    p.add_argument('--wait', type=float, default=0.003)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--devkit', default='speck2fdevkit:0')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load(a.ckpt)
    model.eval()
    scale = data.velocity_scaler(a.session, a.scalers)
    trials = data.motor_split(a.session, 'test', a.scalers)
    evs, ys, lens = [], [], []
    for t in trials:
        model(torch.as_tensor(t.x)[None])
        evs.append(chip.chip_events(model, model.input_events[0].numpy()))
        ys.append(data.unscale_velocity(t.y, *scale))
        lens.append(len(t.y))
    ev, y = np.concatenate(evs), np.concatenate(ys)
    seq = chip.linear_stack(model, a.scale)
    res = {'session': os.path.basename(a.session), 'trials': len(trials), 'steps': len(ev), 'scale': a.scale,
           'readout': a.readout, 'source': 'chip model (software)' if a.nochip else 'devkit'}
    t0 = time.time()
    if a.nochip:
        counts = np.asarray(chip.run_events(chip.discretized(seq, model.host.n_out), ev))
    else:
        chip.RESET_SETTLE_S = a.settle
        dev = chip.Devkit(seq, model.host.n_out, devkit=a.devkit, wait_time=a.wait)
        dev.reset()
        counts, loop_s, power_w, n_ev = dev.run(ev, model.layers[-1].width)
        dev.close()
        res.update(ms_per_step=1e3 * loop_s / len(ev), power_mw=1e3 * power_w, input_events_per_step=n_ev / len(ev))
    arr = dict(counts=np.asarray(counts), y_true=y, trial_lens=np.array(lens),
               pred_trained=data.unscale_velocity(trained_decode(model, counts), *scale))
    res['trained'] = scores(arr['pred_trained'], y)
    if a.readout:
        arr['pred_refit'] = refit_decode(np.asarray(counts), a.readout)
        res['refitted'] = scores(arr['pred_refit'], y)
    print(f"{res['session'][:32]} ({res['source']}, {len(trials)} trials, {time.time() - t0:.0f} s): trained readout "
          f"RMSE {res['trained']['rmse']:.1f}" + (f", refitted RMSE {res['refitted']['rmse']:.1f}" if a.readout else '')
          + (f" | {res['power_mw']:.2f} mW, {res['ms_per_step']:.2f} ms per step" if 'power_mw' in res else ''))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    np.savez_compressed(a.out.replace('.json', '') + '_arrays.npz', **arr)


if __name__ == '__main__':
    main()
