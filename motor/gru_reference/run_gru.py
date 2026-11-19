"""
Train and test one GRU reference decoder (AEGRU or plain GRU) on one session.

    python run_gru.py --protocol neurobench --input RAW.mat      --model aegru --seed 0 --out DIR
    python run_gru.py --protocol ours       --input PROCESSED.h5 --model gru   --seed 0 --out DIR

Training follows the paper: 50 epochs, Adam (lr 1e-3, weight decay 1e-3), cosine annealing, one model per session;
the last epoch's model is tested (no checkpoint selection). Velocity is z-scored on the training split and the
prediction mapped back. Writes DIR/<session>__<tag>.json (R^2, CC, RMSE per axis on each split, the config, timing)
and DIR/<session>__<tag>.npz (test target samples, y_true, y_pred).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from data import load_neurobench, load_ours, transform, windows
from model import AEGRU, PlainGRU, macs_per_output


def r2(y, p):
    return [float(1 - ((y[:, i] - p[:, i]) ** 2).sum() / ((y[:, i] - y[:, i].mean()) ** 2).sum()) for i in range(2)]


def scores(y, p):
    cc = [float(np.corrcoef(y[:, i], p[:, i])[0, 1]) for i in range(2)]
    rx, ry = r2(y, p)
    return {'r2': (rx + ry) / 2, 'r2_x': rx, 'r2_y': ry, 'cc_x': cc[0], 'cc_y': cc[1],
            'rmse': float(np.sqrt(((y - p) ** 2).mean()))}


def predict(model, X, batch=4096):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + batch]))[0].numpy() for i in range(0, len(X), batch)])


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    span = a.ws * a.n
    session = os.path.splitext(os.path.basename(a.input))[0]
    if a.protocol == 'neurobench':
        bins, vel, split, info = load_neurobench(a.input, binary=not a.counts)
        for k in split:
            split[k] = split[k][split[k] >= span]
    else:
        bins, vel, split, info = load_ours(a.input, span)
    tag = a.tag or (f"{a.protocol}_{a.model}_ws{a.ws}n{a.n}_wx{a.wx:g}_b{a.batch}" + ("_counts" if a.counts else "") + f"_s{a.seed}")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"{session}__{tag}")

    raw = {k: windows(bins, v, a.ws, a.n) for k, v in split.items()}
    X = {k: transform(v) for k, v in raw.items()}
    Y = {k: vel[v] for k, v in split.items()}
    mu, sd = Y['train'].mean(0), Y['train'].std(0)
    n_in = bins.shape[1]
    model = AEGRU(n_in) if a.model == 'aegru' else PlainGRU(n_in)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    Xt, Rt = torch.from_numpy(X['train']), torch.from_numpy(raw['train'])
    Yt = torch.from_numpy((Y['train'] - mu) / sd)
    poisson = nn.PoissonNLLLoss(log_input=True, full=False)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:
                continue
            v, log_r = model(Xt[idx])
            loss = ((v - Yt[idx]) ** 2).mean()
            if log_r is not None and a.wx > 0:
                loss = loss + a.wx * poisson(log_r, Rt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep + 1:3d}/{a.epochs}  loss {tot / len(perm):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    res = {'session': session, 'tag': tag, 'config': vars(a), 'info': info,
           'n': {k: int(len(v)) for k, v in split.items()},
           'inference_params': model.inference_params(), 'macs_per_output': macs_per_output(model, n_in, a.n),
           'train_seconds': time.time() - t0}
    preds = {}
    for k in split:
        preds[k] = predict(model, X[k]) * sd + mu
        res[k] = scores(Y[k], preds[k])
    np.savez_compressed(stem + '.npz', test_samples=split['test'], y_true=Y['test'], y_pred=preds['test'])
    json.dump(res, open(stem + '.json', 'w'), indent=1)
    print(f"{session} {tag}: " + "  ".join(f"{k} R2 {res[k]['r2']:.3f}" for k in split), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', choices=['neurobench', 'ours'], required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--model', choices=['aegru', 'gru'], default='aegru')
    p.add_argument('--ws', type=int, default=20)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--wx', type=float, default=1.0, help='weight of the Poisson reconstruction loss (AEGRU)')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--counts', action='store_true', help='neurobench: spike counts per bin, not NeuroBench binary bins')
    p.add_argument('--tag', default=None)
    p.add_argument('--out', required=True)
    main(p.parse_args())
