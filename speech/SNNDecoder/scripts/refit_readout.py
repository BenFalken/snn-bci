"""
Round 20 W1: refit the host readout of a speech Speck network to the chip's own firing.

The spiking layers were trained in software, a step's input summed at once; on the chip each input event lands on
its own (speech_chip.py --mode events), so the spikes the readout sees differ from the ones it was trained on. The
chip's weights stay as they are. Only the host readout (its weights, bias and output time constants) is retrained,
with CTC, on the chip model's spikes for the TRAINING sentences (every 10th held out to stop on); the test
sentences are only scored.

    python refit_readout.py spikes --ckpt V2.pth --data-dir snnData --out R --scale 2.5     chip-model spikes
    python refit_readout.py fit    --ckpt V2.pth --data-dir snnData --out R --scale 2.5     refit, score on test
    python refit_readout.py apply  --ckpt V2.pth --data-dir snnData --out R --counts chip_counts.npz
                                   (devkit spikes from speech_chip.py --save-counts: trained vs refit readout)

R/readout_refit.pt holds the refit readout (speech_chip.py --readout R/readout_refit.pt uses it); R/refit.json the
phoneme errors.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import speech_chip as sc  # noqa: E402
import speck_io  # noqa: E402
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.inference import load_snn_model  # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode  # noqa: E402

torch.set_num_threads(1)


def _session_spikes(job):
    ckpt, data_dir, session, partition, scale, path, source = job
    if os.path.isfile(path):
        return session, 'cached'
    model, ck = load_snn_model(ckpt)
    sessions = list(ck['sessions'])
    part = load_partitions(data_dir, [session], partition, [sessions.index(session)])[0]
    CH = sc.chains(model)
    nets = [speck_io.discretized(sc.spiking_stack(model, 1.0, chain=c), len(c[0])) for c in CH] \
        if source == 'events' else [sc.spiking_stack(model, 1.0, as_trained=True, chain=c) for c in CH]
    ALL = bool(getattr(model, 'readout_all', False))
    out = {}
    for i in range(len(part)):
        ev = sc.host_events(model, part.features[i], part.day_idx)
        sp = sc.assemble(model, [sc.run_events(n, ev[:, c[0]], scale, ALL) if source == 'events'
                                 else sc.run_stack(n, ev[:, c[0]], ALL) for n, c in zip(nets, CH)], ALL)
        out[str(i)] = np.minimum(sp, 255).astype(np.uint8)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return session, len(out)


SOURCE = 'events'


def spike_path(out, partition, scale, session):
    tag = f'x{scale:g}' if SOURCE == 'events' else 'software'
    return os.path.join(out, f'spikes_{partition}_{tag}', f'{session}.npz')


def load_set(a, model_sessions, partition, counts=None):
    """[(spikes (T, n) float, labels, session)] for the checkpoint's sessions."""
    parts = load_partitions(a.data_dir, model_sessions, partition, list(range(len(model_sessions))))
    items = []
    for p in parts:
        z = counts if counts is not None else np.load(spike_path(a.out, partition, a.scale, p.session))
        for i in range(len(p)):
            key = f'{p.session}/{i}' if counts is not None else str(i)
            if key in z:
                items.append((z[key].astype(np.float32), np.asarray(p.labels[i]), p.session))
    return items


class Readout(torch.nn.Module):
    """The host readout of speech_chip.readout, batched and differentiable: leaky traces of the spikes (fixed time
    constants), a linear map, leaky output neurons (learned time constants), one output every _ostride bins."""

    def __init__(self, model):
        super().__init__()
        self.alphas = model.readout_alphas or [0.0]
        self.lin = torch.nn.Linear(model.readout.in_features, model.readout.out_features)
        self.lin.load_state_dict(model.readout.state_dict())
        self.out_log_tau = torch.nn.Parameter(model.out_log_tau.detach().clone())
        self.stages, self.k = model.out_stages, model._ostride

    def traces(self, s):                                   # (B, T, n) -> (B, T, n * len(alphas)), no grad
        with torch.no_grad():
            tr = [torch.zeros_like(s[:, 0]) for _ in self.alphas]
            out = []
            for t in range(s.shape[1]):
                tr = [s[:, t] if al == 0.0 else al * y + (1 - al) * s[:, t] for al, y in zip(self.alphas, tr)]
                out.append(torch.stack(tr, 1).reshape(s.shape[0], -1))
            return torch.stack(out, 1)

    def forward(self, feats):                              # traces -> logits (B, T // k, classes)
        a = torch.exp(-1.0 / self.out_log_tau.exp())
        x = self.lin(feats)
        v = [torch.zeros_like(x[:, 0]) for _ in range(self.stages)]
        outs = []
        for t in range(x.shape[1]):
            prev = x[:, t]
            for j in range(self.stages):
                v[j] = a * v[j] + (1 - a) * prev
                prev = v[j]
            outs.append(prev)
        out = torch.stack(outs, 1)
        return out[:, self.k - 1::self.k] if self.k > 1 else out


def batches(items, size, rng=None):
    order = np.arange(len(items)) if rng is None else rng.permutation(len(items))
    for b in range(0, len(order), size):
        yield [items[j] for j in order[b:b + size]]


def collate(chunk):
    T = max(len(s) for s, _, _ in chunk)
    S = torch.zeros(len(chunk), T, chunk[0][0].shape[1])
    for j, (s, _, _) in enumerate(chunk):
        S[j, :len(s)] = torch.from_numpy(s)
    return S


def per_of(readout, items, blank, k, size=32):
    edits = lens = 0
    by = {}
    with torch.no_grad():
        for chunk in batches(items, size):
            logits = readout(readout.traces(collate(chunk))).numpy()
            for (s, y, sess), lg in zip(chunk, logits):
                n = len(s) // k
                e = edit_distance(greedy_decode(lg[:n], blank), y)
                edits, lens = edits + e, lens + len(y)
                a, b = by.get(sess, (0, 0))
                by[sess] = (a + e, b + len(y))
    return edits / max(lens, 1), {s: a / b for s, (a, b) in by.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['spikes', 'fit', 'apply'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scale', type=float, default=2.5)
    p.add_argument('--jobs', type=int, default=12)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--source', choices=['events', 'software'], default='events',
                   help="spikes: the chip model's (events) or the trained network's own (software, the control)")
    p.add_argument('--counts', default=None, help='apply: .npz of devkit spikes (speech_chip.py --save-counts)')
    a = p.parse_args()
    global SOURCE
    SOURCE = a.source
    os.makedirs(a.out, exist_ok=True)
    model, ck = load_snn_model(a.ckpt)
    sessions = list(ck['sessions'])
    blank = model.n_classes - 1

    if a.step == 'spikes':
        jobs = []
        for part in ('train', 'test'):
            os.makedirs(os.path.dirname(spike_path(a.out, part, a.scale, 'x')), exist_ok=True)
            jobs += [(a.ckpt, a.data_dir, s, part, a.scale, spike_path(a.out, part, a.scale, s), a.source)
                     for s in sessions]
        t0 = time.time()
        with Pool(a.jobs) as pool:
            for s, n in pool.imap_unordered(_session_spikes, jobs):
                print(f'  {s}: {n} ({time.time() - t0:.0f} s)', flush=True)
        return

    readout = Readout(model)
    if a.step == 'apply':
        z = np.load(a.counts)
        test = load_set(a, sessions, 'test', counts=z)
        res = {'counts': os.path.abspath(a.counts), 'sentences': len(test)}
        res['trained_readout'], _ = per_of(readout, test, blank, readout.k)
        readout.load_state_dict(torch.load(os.path.join(a.out, 'readout_refit.pt')))
        res['refit_readout'], res['refit_per_session'] = per_of(readout, test, blank, readout.k)
        print(f"devkit spikes, {len(test)} sentences: trained readout PER {res['trained_readout']:.4f}, "
              f"refit readout {res['refit_readout']:.4f}")
        json.dump(res, open(os.path.join(a.out, 'refit_apply.json'), 'w'), indent=1)
        return

    train_all = load_set(a, sessions, 'train')
    test = load_set(a, sessions, 'test')
    val = train_all[::10]
    train = [x for j, x in enumerate(train_all) if j % 10]
    src = f'chip model, x{a.scale:g}' if SOURCE == 'events' else 'software spikes'
    print(f'{len(train)} train / {len(val)} held-out train / {len(test)} test sentences ({src})')
    res = {'ckpt': os.path.abspath(a.ckpt), 'source': SOURCE, 'scale': a.scale, 'epochs': []}
    res['test_trained_readout'], _ = per_of(readout, test, blank, readout.k)
    res['val_trained_readout'], _ = per_of(readout, val, blank, readout.k)
    print(f"trained readout on these spikes: held-out train PER {res['val_trained_readout']:.4f}, "
          f"test PER {res['test_trained_readout']:.4f}", flush=True)
    opt = torch.optim.Adam(readout.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    rng = np.random.default_rng(0)
    best, bad = res['val_trained_readout'], 0
    torch.save(readout.state_dict(), os.path.join(a.out, 'readout_refit.pt'))
    for ep in range(1, a.epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        for chunk in batches(train, a.batch_size, rng):
            S = collate(chunk)
            logits = readout(readout.traces(S))
            lengths = torch.tensor([len(s) // readout.k for s, _, _ in chunk])
            targets = torch.cat([torch.as_tensor(y, dtype=torch.long) for _, y, _ in chunk])
            tlen = torch.tensor([len(y) for _, y, _ in chunk])
            loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), targets, lengths, tlen, blank=blank,
                              reduction='none', zero_infinity=True).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 10.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        v, _ = per_of(readout, val, blank, readout.k)
        res['epochs'].append({'epoch': ep, 'train_ctc': tot / nb, 'val_per': v})
        print(f'epoch {ep}: train CTC {tot / nb:.3f}, held-out train PER {v:.4f} ({time.time() - t0:.0f} s)',
              flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            torch.save(readout.state_dict(), os.path.join(a.out, 'readout_refit.pt'))
        else:
            bad += 1
            if bad >= a.patience:
                break
    readout.load_state_dict(torch.load(os.path.join(a.out, 'readout_refit.pt')))
    res['val_refit_readout'] = best
    res['test_refit_readout'], res['test_refit_per_session'] = per_of(readout, test, blank, readout.k)
    print(f"refit readout (chosen on held-out TRAIN sentences): test PER {res['test_refit_readout']:.4f} "
          f"vs trained readout {res['test_trained_readout']:.4f}")
    json.dump(res, open(os.path.join(a.out, 'refit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
