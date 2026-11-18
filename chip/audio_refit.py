"""
Refit the audio Speck SNN's host readout to the chip's own firing (the speech W1 recipe, for audio).

The chip network stays as trained; the readout (Hidden 3's spikes -> one EMA stage -> linear map to the 128 mel
bands) was fitted to the network's software spikes. Here the whole recording is run through the chip model
(snndec.chip.run_events: 8-bit weights, layer-1 threshold x the chip's factor, input events applied one at a time),
Hidden 3's spikes are smoothed exactly as the readout does, and the linear map is refitted by ridge regression on
the TRAINING rows; the ridge strength is chosen on the VALIDATION rows (as every audio decoder is selected); test is
only scored. The control does the same on the software network's spikes (run_software: a step's input summed).

    python chip/audio_refit.py --pipeline DIR --ckpt words_P5_s11/model.pt --scale 4 --out OUT

OUT/: spikes_{event,software}.npz (Hidden 3 per chip step, uint8), refit.json, readout_{event,software}.npz
(W, b), preds_{float,event_trained,event_refit,software_refit}.npy (test rows). A devkit run's Hidden 3 counts
(audio_chip.py --save-counts) can be decoded with readout_event.npz.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snndec import chip, data, tasks  # noqa: E402
from snndec.convert import load  # noqa: E402
from snndec.evaluate import recording_file  # noqa: E402

RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def features(model, spikes):
    """Hidden 3's spikes per chip step -> the readout's input per frame: one EMA per stage, normalized, sampled at
    each frame's last step (as EMAReadout.step, then out_every)."""
    ro = model.readout
    d = float(torch.sigmoid(ro.decay_raw).mean())
    u = spikes.astype(np.float64)
    for _ in range(ro.stages):
        e = np.empty_like(u)
        acc = np.zeros(u.shape[1])
        for t in range(len(u)):
            acc = u[t] if t == 0 else d * acc + u[t]
            e[t] = acc
        u = e
    if ro.normalize:
        u = u * (1 - d) ** ro.stages
    k = model.out_every
    return u[k - 1::k].astype(np.float32)


def ridge(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = np.hstack([Z, np.ones((len(Z), 1))])
    R = lam * len(A) * np.eye(A.shape[1])
    R[-1, -1] = 0
    Wz = np.linalg.solve(A.T @ A + R, A.T @ Y)
    W = Wz[:-1] / sd[:, None]                       # back to raw features: y = X W + b
    b = Wz[-1] - mu @ W
    return W.astype(np.float32), b.astype(np.float32)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline', required=True, help='the dataset folder with window_targets_{train,valid,test}')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--scale', type=float, required=True, help="layer 1's threshold factor on the chip")
    p.add_argument('--out', required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    model, info = load(a.ckpt)
    model.eval()
    rec, W, n = data.audio_recording(a.pipeline, recording_file(info))
    if info.get('channels') is not None:
        rec = rec[:, list(info['channels'])]
    y_float = model(torch.as_tensor(rec)[None])[0].numpy()                 # one output per frame, from rest
    events = chip.chip_events(model, model.input_events[0].numpy())
    rows = {'train': (W - 1, n['train']), 'valid': (n['train'] + W - 1, n['valid']),
            'test': (n['train'] + n['valid'] + W - 1, n['test'])}
    Y = {s: np.load(f"{a.pipeline}/window_targets_{s}.npy").astype(np.float32) for s in rows}
    cut = lambda arr, s: arr[rows[s][0]:rows[s][0] + rows[s][1]]

    spk = {}
    for kind in ('event', 'software'):
        f = os.path.join(a.out, f'spikes_{kind}.npz')
        if os.path.isfile(f):
            spk[kind] = np.load(f)['h3']
            continue
        if kind == 'event':
            s = chip.simulate(model, events, a.scale)
        else:
            s = chip.simulate(model, events, 1.0, mode='software')
        spk[kind] = np.minimum(np.asarray(s), 255).astype(np.uint8)
        np.savez_compressed(f, h3=spk[kind])
        print(f'{kind} spikes: {spk[kind].shape}, {spk[kind].sum(1).mean():.1f} per chip step', flush=True)

    lin = model.readout.linear
    W0, b0 = lin.weight.numpy().T, lin.bias.numpy()
    res = {'ckpt': os.path.abspath(a.ckpt), 'scale': a.scale,
           'r_float_test': tasks.mean_correlation(Y['test'], cut(y_float, 'test'))}
    preds = {'float': cut(y_float, 'test')}
    for kind in ('event', 'software'):
        X = features(model, spk[kind])
        trained = X @ W0 + b0
        res[f'r_{kind}_trained'] = {s: tasks.mean_correlation(Y[s], cut(trained, s)) for s in rows}
        best = None
        for lam in RIDGES:
            Wf, bf = ridge(cut(X, 'train'), Y['train'], lam)
            rv = tasks.mean_correlation(Y['valid'], cut(X, 'valid') @ Wf + bf)
            if best is None or rv > best[0]:
                best = (rv, lam, Wf, bf)
        rv, lam, Wf, bf = best
        refit = X @ Wf + bf
        res[f'r_{kind}_refit'] = {s: tasks.mean_correlation(Y[s], cut(refit, s)) for s in rows}
        res[f'ridge_{kind}'] = lam
        np.savez(os.path.join(a.out, f'readout_{kind}.npz'), W=Wf, b=bf)
        preds[f'{kind}_trained'], preds[f'{kind}_refit'] = cut(trained, 'test'), cut(refit, 'test')
        print(f"{kind:8s} trained readout r valid {res[f'r_{kind}_trained']['valid']:.3f} test "
              f"{res[f'r_{kind}_trained']['test']:.3f} | refit (ridge {lam:g}) valid "
              f"{res[f'r_{kind}_refit']['valid']:.3f} test {res[f'r_{kind}_refit']['test']:.3f}", flush=True)
    for k, v in preds.items():
        np.save(os.path.join(a.out, f'preds_{k}.npy'), v.astype(np.float32))
    json.dump(res, open(os.path.join(a.out, 'refit.json'), 'w'), indent=1)
    print(f"float test r {res['r_float_test']:.3f}; wrote {a.out}")


if __name__ == '__main__':
    main()
