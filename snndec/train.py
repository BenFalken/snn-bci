"""
One trainer for every experiment and design; each experiment keeps its own recipe, written out in its config
(configs/*.yaml):

    python -m snndec.train snndec/configs/motor_lsnn.yaml --data SESSION_DIR --scalers velocity_scalers.json --out DIR
    python -m snndec.train snndec/configs/audio_speck.yaml --data PIPELINE_DIR --out DIR
    python -m snndec.train snndec/configs/speech_lsnn.yaml --data SNNDATA_DIR --out DIR

Config:
    experiment: motor | audio | speech
    model: the SNNDecoder spec (model.py)
    train:
      batches:     trials (motor, speech: whole trials; speech batches hold one session) | windows (audio LSNN: the
                   window's last output scored) | chunks (audio Speck and streaming LSNN: random stretches of the
                   recording, the first burn_in steps not scored)
      batch_size, chunk_steps, burn_in, coverage
      optimizer:   adam | adamw;  lr, eps, weight_decay;  tau_lr_mult (synaptic time constants)
      schedule:    none | plateau (factor, scheduler_patience, min_lr) | linear (to 0 over all batches, per batch)
      epochs, patience (early stopping; null: none), grad_clip, tbptt (input steps)
      augment:     white_noise_sd, constant_offset_sd (on the input, training only)
      spike_rate:  weight, targets (one or per layer), form: layer_relu (sum over layers of max(0, rate - target),
                   motor and speech) | total_sq (max(0, rate - target)^2 of all layers together, audio);
                   per: step (rate per model step) | output (per output frame, speech)
      select_on:   the split scored each epoch for the best model (motor: test, as spike_bmi; speech: test;
                   audio: valid)
The best model (lowest loss on select_on) is saved as a unified checkpoint (convert.save).
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from . import data, tasks
from .convert import save
from .model import SNNDecoder


# ------------------------------------------------------------------------------------------------- batches
def motor_batches(trials, batch_size, shuffle, rng):
    order = rng.permutation(len(trials)) if shuffle else np.arange(len(trials))
    lengths = {len(t.x) for t in trials}
    bs = batch_size if len(lengths) == 1 else 1                    # unequal lengths (hkm): one trial per batch
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        yield dict(x=torch.as_tensor(np.stack([trials[j].x for j in idx])),
                   y=torch.as_tensor(np.stack([trials[j].y for j in idx])), day=None)


def speech_batches(trials, batch_size, shuffle, rng):
    by_day = {}
    for i, t in enumerate(trials):
        by_day.setdefault(t.day, []).append(i)
    groups = []
    for day, idx in by_day.items():
        idx = list(rng.permutation(idx)) if shuffle else idx
        groups += [(day, idx[i:i + batch_size]) for i in range(0, len(idx), batch_size)]
    if shuffle:
        groups = [groups[i] for i in rng.permutation(len(groups))]
    for day, idx in groups:
        L = max(len(trials[j].x) for j in idx)
        x = np.zeros((len(idx), L, trials[idx[0]].x.shape[1]), np.float32)
        for k, j in enumerate(idx):
            x[k, :len(trials[j].x)] = trials[j].x
        yield dict(x=torch.as_tensor(x), day=day, lengths=torch.tensor([len(trials[j].x) for j in idx]),
                   labels=torch.as_tensor(np.concatenate([trials[j].y for j in idx])).long(),
                   label_lengths=torch.tensor([len(trials[j].y) for j in idx]))


def window_batches(X, Y, batch_size, shuffle, rng):
    order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(order), batch_size):
        idx = np.sort(order[i:i + batch_size])
        yield dict(x=torch.as_tensor(X[idx]), y=torch.as_tensor(Y[idx]), day=None)


def window_pair_batches(X, Y, batch_size, shuffle, rng):
    """As window_batches, each window i with its neighbour i + 1 (the next output frame): the first half of a batch
    is the windows, the second their neighbours, for the frame-to-frame term of the loss (train.deriv_weight)."""
    n = len(X) - 1
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        idx = np.sort(order[i:i + batch_size])
        both = np.concatenate([idx, idx + 1])
        yield dict(x=torch.as_tensor(X[both]), y=torch.as_tensor(Y[both]), day=None, pairs=len(idx))


def chunk_batches(rec, ys, lag, L, burn_in, coverage, batch_size, rng):
    """Random stretches of a recording; output step f is scored against ys[f] once f >= lag and past the burn-in."""
    L = min(L, len(rec))
    n = max(1, int(round(coverage * len(rec) / L)))
    starts = rng.integers(0, len(rec) - L + 1, n)
    for i in range(0, n, batch_size):
        steps = starts[i:i + batch_size, None] + np.arange(L)[None]
        mask = (steps >= lag) & (np.arange(L)[None] >= burn_in)
        if mask.any():
            yield dict(x=torch.as_tensor(rec[steps]), y=torch.as_tensor(ys[steps]), mask=torch.as_tensor(mask), day=None)


# ------------------------------------------------------------------------------------------------- penalties
def spike_penalty(model, cfg, n_units_steps, mask=None):
    """cfg: weight, targets, form, per. n_units_steps: valid steps (x batch) the rate is taken over."""
    if not cfg or not cfg.get('weight'):
        return torch.zeros(())
    spikes = [s if mask is None else s * mask[..., None] for s in model.layer_spikes]
    widths = model.widths
    if cfg.get('form', 'layer_relu') == 'total_sq':
        rate = sum(s.sum() for s in spikes) / (n_units_steps * sum(widths))
        return cfg['weight'] * torch.relu(rate - cfg['targets'][0]) ** 2
    tg = cfg['targets'] * len(widths) if len(cfg['targets']) == 1 else cfg['targets']
    return cfg['weight'] * sum(torch.relu(s.sum() / (n_units_steps * w) - t) for s, w, t in zip(spikes, widths, tg))


def augment(x, cfg):
    if cfg.get('white_noise_sd'):
        x = x + torch.randn_like(x) * cfg['white_noise_sd']
    if cfg.get('constant_offset_sd'):
        x = x + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * cfg['constant_offset_sd']
    return x


def T_device(cfg):
    return cfg['train'].get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------------------------- experiments
class Experiment:
    """Data, batches, loss and metric of one experiment."""

    def __init__(self, cfg, args):
        self.cfg, self.T = cfg, cfg['train']
        self.kind = cfg['experiment']
        self.dev = torch.device(getattr(args, 'device', None) or T_device(cfg))
        self.rng = np.random.default_rng(self.T.get('seed', 0))
        e = self.kind
        if e == 'motor':
            dirs = args.data if isinstance(args.data, list) else [args.data]
            self.scale = data.velocity_scaler(dirs[0], args.scalers)
            self.splits = {s: [t for d in dirs for t in data.motor_split(d, s, args.scalers)] for s in ('train', 'test')}
        elif e == 'speech':
            args.data = args.data[0] if isinstance(args.data, list) else args.data
            self.sessions = cfg['data']['sessions']
            self.splits = {'train': data.speech_split(args.data, self.sessions, 'train'),
                           'test': data.speech_split(args.data, self.sessions, 'test')}
        elif e == 'audio':
            args.data = args.data[0] if isinstance(args.data, list) else args.data
            self.windows = {s: data.audio_windows(args.data, s, cfg['data'].get('past'))
                            for s in ('train', 'valid', 'test')}
            if self.T['batches'] == 'chunks':
                if cfg['data'].get('input', 'counts') == 'counts':          # the Speck network: features.npy counts
                    self.rec, self.W, _ = data.audio_recording(args.data, cfg['data'].get('features', 'features.npy'))
                else:                                                       # the LSNN streaming: window features
                    X = np.concatenate([self.windows[s][0] for s in ('train', 'valid', 'test')])
                    self.rec, self.W = np.concatenate([X[:, 0], X[-1, 1:]]), X.shape[1]
        else:
            raise ValueError(f'experiment {e!r}')

    def resolve(self, spec):
        """Settings the originals derived from the training data: the motor readout's starting value (scaled zero
        velocity); the audio Speck network's channels (n_channels best on the training frames), delays (n_delays
        spread over the window) and input scale (1 / SD of each channel's training input)."""
        spec = copy.deepcopy(spec)
        R, inp = spec['readout'], spec.setdefault('input', {})
        if self.kind == 'motor':
            zero = float(data.scale_velocity(0.0, *self.scale))
            if R.get('bias_init') == 'neutral':
                R['bias_init'] = zero
            if R.get('neutral') == 'neutral':
                R['neutral'] = zero
        if self.kind == 'audio' and spec['kind'] == 'lsnn':
            spec['n_channels'] = self.windows['train'][0].shape[2]
        if self.kind == 'audio':
            spec['n_outputs'] = self.windows['train'][1].shape[1]
        if self.kind == 'audio' and spec['kind'] == 'speck':
            X, Y = self.windows['train']
            K = int(inp.get('substeps', 1)) if inp.get('encoding') == 'fine' else 1
            frames = self.rec.reshape(len(self.rec), K, -1).sum(1) if K > 1 else self.rec   # 'fine': per-frame sums
            if inp.get('n_channels'):                              # in ranked order, as the original
                inp['channels'] = data.audio_rank_channels(frames, X, Y)[:inp.pop('n_channels')]
            if inp.get('n_delays'):
                top = inp.pop('delay_max', None) or self.W - 1
                inp['delays'] = sorted({int(round(d)) for d in np.linspace(0, top, inp.pop('n_delays'))})
            else:                                                  # explicit model.input.delays (n_delays 0 or unset)
                inp.pop('n_delays', None); inp.pop('delay_max', None)
            if inp.pop('input_norm', False):
                probe = SNNDecoder({**spec, 'input': {k: v for k, v in inp.items() if k != 'input_scale'}})
                probe.eval()
                with torch.no_grad():
                    u = probe.host.encode(torch.as_tensor(self.rec[:len(Y) + self.W - 1])[None])[0]
                inp['input_scale'] = (1.0 / u.double().std(0, unbiased=False).clamp(min=1e-3)).tolist()
            spec['n_channels'] = frames.shape[1]
        if self.kind == 'audio' and spec['kind'] == 'lsnn' and inp.get('encoding') == 'fine':
            spec['n_channels'] = self.rec.shape[1] // int(inp['substeps'])     # streaming LSNN on 4 ms inputs
        return spec

    # ----- training batches and their loss
    def train_batches(self):
        T = self.T
        if self.kind == 'motor':
            yield from motor_batches(self.splits['train'], T['batch_size'], True, self.rng)
        elif self.kind == 'speech':
            yield from speech_batches(self.splits['train'], T['batch_size'], True, self.rng)
        elif T['batches'] == 'windows' and T.get('deriv_weight'):
            yield from window_pair_batches(*self.windows['train'], T['batch_size'], True, self.rng)
        elif T['batches'] == 'windows':
            yield from window_batches(*self.windows['train'], T['batch_size'], True, self.rng)
        else:
            n_tr, lag = len(self.windows['train'][1]), self.W - 1
            ys = np.concatenate([np.zeros((lag, self.windows['train'][1].shape[1]), np.float32),
                                 self.windows['train'][1]])
            yield from chunk_batches(self.rec[:n_tr + lag], ys, lag, T['chunk_steps'], T['burn_in'],
                                     T.get('coverage', 4), T['batch_size'], self.rng)

    def to_dev(self, b):
        return {k: v.to(self.dev) if torch.is_tensor(v) else v for k, v in b.items()}

    def batch_loss(self, model, b):
        b = self.to_dev(b)
        x = augment(b['x'], self.T.get('augment', {}))
        out = model(x, b['day'], tbptt=self.T.get('tbptt', 0))
        sr = self.T.get('spike_rate', {})
        N, Tm = out.shape[0], model.layer_spikes[0].shape[1]
        if self.kind == 'motor':
            loss = tasks.motor_loss(out, b['y'], self.scale, self.T.get('loss_eps', 0.0))
            return loss, loss + spike_penalty(model, sr, N * Tm)
        if self.kind == 'speech':
            ol = model.output_lengths(b['lengths'])
            loss = tasks.ctc_loss(out, ol, b['labels'], b['label_lengths'])
            steps = (torch.arange(Tm, device=self.dev)[None] < (b['lengths'] * model.substeps)[:, None]).float()
            n = ol.sum() if sr.get('per') == 'output' else steps.sum()
            return loss, loss + spike_penalty(model, sr, n, steps)
        w = self.T.get('deriv_weight', 0.0)                          # frame-to-frame change, matched to the target's
        if 'mask' in b:                                              # chunks
            loss = ((out - b['y']) ** 2)[b['mask']].mean()
            if w:
                m = b['mask'][:, 1:] & b['mask'][:, :-1]
                d = (out[:, 1:] - out[:, :-1]) - (b['y'][:, 1:] - b['y'][:, :-1])
                loss_total = loss + w * (d ** 2)[m].mean()
                return loss, loss_total + spike_penalty(model, sr, N * Tm)
        else:                                                        # windows: the last step
            out = out[:, -1]
            loss = ((out - b['y']) ** 2).mean()
            if w and 'pairs' in b:
                k = b['pairs']
                d = (out[k:] - out[:k]) - (b['y'][k:] - b['y'][:k])
                return loss, loss + w * (d ** 2).mean() + spike_penalty(model, sr, N * Tm)
        return loss, loss + spike_penalty(model, sr, N * Tm)

    # ----- the selection score (lower is better) and the reported metric
    @torch.no_grad()
    def score(self, model):
        from . import evaluate as ev
        split = self.T.get('select_on', 'test')
        dev = self.dev
        if self.kind == 'motor':
            preds = [model(torch.as_tensor(t.x, device=dev)[None])[0].cpu().numpy() for t in self.splits[split]]
            losses = [float(tasks.motor_loss(torch.as_tensor(p), torch.as_tensor(t.y), self.scale))
                      for p, t in zip(preds, self.splits[split])]
            return float(np.mean(losses)), {f'rmse_{split}': float(np.mean(losses))}
        if self.kind == 'speech':
            tot, n, logits = 0.0, 0, []
            for b in speech_batches(self.splits[split], self.T['batch_size'], False, self.rng):
                out = model(b['x'].to(dev), b['day']).cpu()
                ol = model.output_lengths(b['lengths'])
                tot += float(tasks.ctc_loss(out, ol, b['labels'], b['label_lengths'])) * len(ol)
                n += len(ol)
                logits += [o[:l].numpy() for o, l in zip(out, ol)]
            per = tasks.phoneme_error_rate(logits, [t.y for t in self.splits[split]])
            return tot / n, {f'ctc_{split}': tot / n, f'per_{split}': per}
        if self.T['batches'] == 'windows':
            X, Y = self.windows[split]
            p = np.concatenate([model(torch.as_tensor(X[i:i + 1024], device=dev))[:, -1].cpu().numpy()
                                for i in range(0, len(X), 1024)])
        else:                                                        # one pass over train + valid (+ test)
            n = {s: len(self.windows[s][1]) for s in ('train', 'valid', 'test')}
            lag = self.W - 1
            end = n['train'] + n['valid'] + (n['test'] if split == 'test' else 0) + lag
            y = model(torch.as_tensor(self.rec[:end], device=dev)[None])[0].cpu().numpy()[lag:]
            start = n['train'] + (n['valid'] if split == 'test' else 0)
            p, Y = y[start:start + n[split]], self.windows[split][1]
        return float(((p - Y) ** 2).mean()), {f'mse_{split}': float(((p - Y) ** 2).mean()),
                                              f'r_{split}': tasks.mean_correlation(Y, p)}


# ------------------------------------------------------------------------------------------------- the loop
def optimizer_for(model, T):
    tau = [p for n, p in model.named_parameters() if n.endswith('log_tau_syn')]
    rest = [p for n, p in model.named_parameters() if not n.endswith('log_tau_syn')]
    groups = [{'params': rest, 'lr_mult': 1.0}]
    if tau:
        groups.append({'params': tau, 'lr_mult': T.get('tau_lr_mult', 1.0)})
    for g in groups:
        g['lr'] = T['lr'] * g['lr_mult']
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[T.get('optimizer', 'adam')]
    return cls(groups, lr=T['lr'], eps=T.get('eps', 1e-8), weight_decay=T.get('weight_decay', 0.0))


def train(cfg, args):
    T = cfg['train']
    torch.manual_seed(T.get('seed', 0))
    exp = Experiment(cfg, args)
    model = SNNDecoder(exp.resolve(cfg['model'])).to(exp.dev)
    if args.init:
        from .convert import load
        model.load_state_dict(load(args.init)[0].to(exp.dev).state_dict())
    opt = optimizer_for(model, T)
    sched = T.get('schedule', 'none')
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=T.get('factor', 0.1), patience=T.get(
        'scheduler_patience', 10), min_lr=T.get('min_lr', 1e-6)) if sched == 'plateau' else None
    n_batches = sum(1 for _ in exp.train_batches()) if sched == 'linear' else None
    total, step = (T['epochs'] * n_batches if n_batches else None), 0
    best, best_state, waited = math.inf, None, 0
    os.makedirs(args.out, exist_ok=True)
    print(f"{cfg['experiment']} {cfg['model']['kind']}: {' -> '.join(map(str, [model.host.n_out] + model.widths))}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(T['epochs']):
        t0, run, nb, skipped = time.time(), 0.0, 0, 0
        model.train()
        for b in exp.train_batches():
            loss, total_loss = exp.batch_loss(model, b)
            opt.zero_grad()
            total_loss.backward()
            if all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                nn.utils.clip_grad_norm_(model.parameters(), T.get('grad_clip', 10.0))
                opt.step()
                run, nb = run + float(loss.detach()), nb + 1
            else:
                skipped += 1
            if sched == 'linear':
                step += 1
                for g in opt.param_groups:
                    g['lr'] = T['lr'] * g['lr_mult'] * max(0.0, 1 - step / total)
        model.eval()
        score, metrics = exp.score(model)
        if plateau is not None:
            plateau.step(score)
        print(f"epoch {epoch + 1}/{T['epochs']} ({time.time() - t0:.0f}s) | train loss {run / max(nb, 1):.4f} | "
              + ' | '.join(f'{k} {v:.4f}' for k, v in metrics.items()) + f" | lr {opt.param_groups[0]['lr']:.2e}"
              + (f' | skipped {skipped}' if skipped else ''), flush=True)
        if score < best - T.get('min_delta', 0.0):
            best, waited, best_state = score, 0, copy.deepcopy(model.state_dict())
            save(model, os.path.join(args.out, 'best.pt'), {'epoch': epoch + 1, 'metrics': metrics, 'config': cfg,
                                                             'sessions': cfg.get('data', {}).get('sessions')})
        else:
            waited += 1
            if T.get('patience') and waited >= T['patience']:
                print(f'early stop: no improvement for {waited} epochs')
                break
    model.load_state_dict(best_state)
    return model.cpu()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config')
    p.add_argument('--data', required=True, nargs='+', help='motor: one or more session dirs (several: pooled)')
    p.add_argument('--scalers', help='motor: velocity_scalers.json')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None, help='cuda | cpu (default: cuda when there is one)')
    p.add_argument('--init', help='start from this checkpoint (any kind convert.load reads): fine-tuning')
    p.add_argument('--set', nargs='*', default=[], help='override config values, e.g. train.epochs=2')
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split('=', 1)
        d = cfg
        *path, last = k.split('.')
        for q in path:
            d = d.setdefault(q, {})
        d[last] = yaml.safe_load(v)
    train(cfg, a)


if __name__ == '__main__':
    main()
