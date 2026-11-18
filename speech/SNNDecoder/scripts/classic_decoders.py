"""Fit the two classic benchmark decoders, a Wiener filter and a Kalman filter, for phoneme decoding, in closed form.

The benchmark versions regress continuous kinematics; here the target is the 41 CTC classes (39 phonemes, silence,
blank) every 80 ms, so they are scored with the same CTC decoding and language model as every other decoder. The data
have no phoneme timing, so each training sentence's 80 ms frames get labels by CTC forced alignment (Viterbi) of a
trained decoder's output (default: the GRU control on the same input) to the true phoneme sequence, and each frame
is labelled with the phoneme it belongs to (frame-wise targets; repeats collapse in CTC decoding). Then:

  Wiener filter   class scores of frame k = W [the last `taps` 20 ms bins, taps x 256 values] + b. Ridge regression
                  onto the one-hot frame labels, from the normal equations (X'X, X'Y accumulated batch by batch: the
                  2,048 x 2,048 X'X is the largest object, never the full design matrix).
  Kalman filter   state = the 41 class scores; observation = the frame's mean over its 4 bins (256 values).
                  Transition A (41 x 41) and observation H (256 x 41) by least squares, noise covariances W, Q from
                  the residuals; the gain is iterated to its steady state once, so decoding a frame costs two small
                  matrix-vector products (no 256 x 256 inversion per step, as the benchmark's per-step update has).

Both read the normalized features as every other decoder does (no per-session layer: they are z-scored per block).
A scale per decoder, fitted on training frames, turns the scores into CTC logits. Each is saved like a GRU control
(<out-dir>/<name>.json + .pt), so snnDecoder.inference.gru_inference, compareDecoders.py --gru, plotExampleTrial.py
--gruJson and efficiency.py ("gru" rows) all take them unchanged.

example usage (on Oscar; a GPU helps the alignment, the fits run in minutes on CPU):
    python classic_decoders.py --data-dir $B/derived/snnData --align-with $B/derived/snns/controls/gru_1024_out80_full96.json \
        --out-dir $B/derived/snns/controls
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ann_controls                                                    # noqa: E402
from snnDecoder.dataset import BASELINE_SESSIONS, SessionBatches, load_partitions   # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode              # noqa: E402

BLANK, NC = ann_controls.BLANK, ann_controls.N_CLASSES
SIL = 39                                                              # the silence class (PHONE_DEF_SIL[-1])


def ctc_align(logp, target):
    """Viterbi CTC alignment: logp (T, classes) log-probabilities, target phoneme ids -> frame labels (T,)."""
    T, L = logp.shape[0], len(target)
    ext = np.full(2 * L + 1, BLANK, dtype=np.int64)
    ext[1::2] = target
    S = len(ext)
    neg = -1e30
    dp = np.full((T, S), neg)
    back = np.zeros((T, S), dtype=np.int8)                                  # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[neg], prev[:-1]])
        c2 = np.where(skip, np.concatenate([[neg, neg], prev[:-2]]), neg)
        stacked = np.stack([c0, c1, c2])
        back[t] = stacked.argmax(0)
        dp[t] = stacked.max(0) + logp[t, ext]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    labels = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        labels[t] = ext[s]
        s -= int(back[t, s])
    return labels


def dense_labels(path):
    """A CTC path (mostly blanks, one frame per phoneme) -> frame-wise phoneme labels: every frame takes the phoneme
    it belongs to, i.e. the last one emitted. Frames before the first phoneme are silence: the sentences' phoneme
    sequences do not start with SIL, so the pre-speech period has no phoneme of its own (labelling it with the first
    phoneme taught the filters to output it during silence). The frame-wise targets of classic phoneme
    classification; a linear decoder cannot reproduce CTC's one-frame spikes, because the input hardly changes
    within a phoneme."""
    out = path.copy()
    emitted = np.flatnonzero(path != BLANK)
    if len(emitted) == 0:
        out[:] = SIL
        return out
    out[:emitted[0]] = SIL
    last = path[emitted[0]]
    for t in range(emitted[0], len(path)):
        last = path[t] if path[t] != BLANK else last
        out[t] = last
    return out


def aligned_labels(align_json, data_dir, sessions, device, dense=True, partition='train'):
    """Frame labels for every sentence of the partition, from the alignment decoder's output: {session: [labels per trial]}."""
    from snnDecoder.inference import gru_inference
    out = {}
    for si, s in enumerate(sessions):
        o = gru_inference(align_json, data_dir, [s], partition, device=device)
        labs = []
        for i in range(len(o['logitLengths'])):
            n = int(o['logitLengths'][i])
            lg = o['logits'][i, :n].astype(np.float64)
            logp = lg - lg.max(1, keepdims=True)
            logp -= np.log(np.exp(logp).sum(1, keepdims=True))
            tgt = o['trueSeqs'][i]
            tgt = tgt[tgt >= 0]
            path = ctc_align(logp, tgt) if 0 < len(tgt) <= n else np.argmax(lg, 1)
            labs.append(dense_labels(path) if dense else path)
        out[s] = labs
        print(f'  aligned {s} ({partition}): {len(labs)} sentences', flush=True)
    return out


def session_frames(model, part, labels, device):
    """For each trial: (Wiener frame inputs (n, taps*C) or Kalman observations (n, C), one-hot labels (n, 41))."""
    for i, feats in enumerate(part.features):
        x = torch.from_numpy(np.asarray(feats, np.float32))[None].to(device)
        if isinstance(model, ann_controls.WienerNet):
            f = model.frames(x)[0]
        else:
            n = x.shape[1] // model.frame_stride
            f = x[0, :n * model.frame_stride].reshape(n, model.frame_stride, -1).mean(1)
        lab = labels[i][:f.shape[0]]
        f = f[:len(lab)]
        y = torch.zeros(len(lab), NC, device=device)
        y[torch.arange(len(lab)), torch.from_numpy(lab).to(device)] = 1.0
        yield f.double(), y.double()


def fit_wiener(parts, labels, device, taps, ridge, verbose=True):
    model = ann_controls.WienerNet(1, taps=taps).to(device)
    D = model.lin.in_features
    xtx = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=device)    # with a constant column (bias)
    xty = torch.zeros(D + 1, NC, dtype=torch.float64, device=device)
    n = 0
    for part in parts:
        for f, y in session_frames(model, part, labels[part.session], device):
            f1 = torch.cat([f, torch.ones(f.shape[0], 1, dtype=f.dtype, device=device)], 1)
            xtx += f1.T @ f1
            xty += f1.T @ y
            n += f.shape[0]
    lam = ridge * torch.diagonal(xtx)[:-1].mean()
    reg = torch.eye(D + 1, dtype=xtx.dtype, device=device) * lam
    reg[-1, -1] = 0                                                         # no penalty on the bias
    wb = torch.linalg.solve(xtx + reg, xty)                                 # (D + 1, classes)
    with torch.no_grad():
        model.lin.weight.copy_(wb[:-1].T.float())
        model.lin.bias.copy_(wb[-1].float())
    if verbose:
        print(f'  Wiener filter: {n:,} frames, {D:,} inputs, ridge {lam.item():.3g}', flush=True)
    return model


def fit_kalman(parts, labels, device, ridge, iters=500, verbose=True):
    model = ann_controls.KalmanNet(1).to(device)
    C = model.K.shape[1]
    d = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
    frames = lambda: (fy for part in parts for fy in session_frames(model, part, labels[part.session], device))
    Sx, Sz, n = d(NC), d(C), 0
    for f, y in frames():                                                   # pass 1: means
        Sx += y.sum(0)
        Sz += f.sum(0)
        n += f.shape[0]
    mx, mz = Sx / n, Sz / n
    Sxx, Szx, Szz = d(NC, NC), d(C, NC), d(C, C)
    S11, S21, S22, m = d(NC, NC), d(NC, NC), d(NC, NC), 0
    for f, y in frames():                                                   # pass 2: centred second moments
        xc, zc = y - mx, f - mz
        Sxx += xc.T @ xc
        Szx += zc.T @ xc
        Szz += zc.T @ zc
        S11 += xc[:-1].T @ xc[:-1]
        S21 += xc[1:].T @ xc[:-1]
        S22 += xc[1:].T @ xc[1:]
        m += xc.shape[0] - 1
    I = torch.eye(NC, dtype=torch.float64, device=device)
    eps = 1e-6 * torch.trace(Sxx) / NC                                       # one-hot states are collinear
    A = S21 @ torch.linalg.inv(S11 + eps * I)
    W = (S22 - A @ S21.T) / m
    H = Szx @ torch.linalg.inv(Sxx + eps * I)                                # (C, classes)
    Q = (Szz - H @ Szx.T) / n
    Q = Q + ridge * torch.diagonal(Q).mean() * torch.eye(C, dtype=Q.dtype, device=device)
    W = (W + W.T) / 2 + eps / n * I
    P = W.clone()
    for it in range(iters):                                                 # Riccati iteration to the steady-state gain
        Pm = A @ P @ A.T + W
        K = torch.linalg.solve(H @ Pm @ H.T + Q, H @ Pm).T                    # (classes, C)
        P_new = (I - K @ H) @ Pm
        if torch.norm(P_new - P) < 1e-9 * torch.norm(P):
            break
        P = P_new
    with torch.no_grad():
        model.M.copy_((A - K @ H @ A).float())
        model.K.copy_(K.float())
        model.mean_x.copy_(mx.float())
        model.mean_z.copy_(mz.float())
    if verbose:
        print(f'  Kalman filter: {n:,} frames, steady-state gain after {it + 1} iterations', flush=True)
    return model


@torch.no_grad()
def fit_scale(model, parts, labels, device):
    """The logit scale maximizing the aligned labels' log-likelihood on training frames (1-D search)."""
    model.scale.fill_(1.0)
    scores, labs = [], []
    for part in parts[:6]:                                                  # a subset is plenty for one number
        for i, feats in enumerate(part.features[:60]):
            out = model(torch.from_numpy(np.asarray(feats, np.float32))[None].to(device), part.day_idx)[0]
            lab = labels[part.session][i][:out.shape[0]]
            scores.append(out[:len(lab)])
            labs.append(torch.from_numpy(lab).to(device))
    s, y = torch.cat(scores), torch.cat(labs)
    best = max(np.logspace(-1, 3, 81), key=lambda b: torch.log_softmax(b * s, 1)[torch.arange(len(y)), y].mean().item())
    model.scale.fill_(float(best))
    return float(best)


@torch.no_grad()
def test_per(model, parts, device):
    edit = phones = 0
    for part in parts:
        for i, feats in enumerate(part.features):
            out = model(torch.from_numpy(np.asarray(feats, np.float32))[None].to(device), part.day_idx)[0].cpu().numpy()
            edit += edit_distance(greedy_decode(out, BLANK), part.labels[i])
            phones += len(part.labels[i])
    return edit / phones


def fit_per_session(which, parts, labels, device, taps, ridge, n_sessions):
    """One filter per session (as a daily-recalibrated decoder), packed into WienerDayNet / KalmanDayNet."""
    day = (ann_controls.WienerDayNet(n_sessions, taps=taps) if which == 'wiener' else
           ann_controls.KalmanDayNet(n_sessions)).to(device)
    frames = 0
    for part in parts:
        m = fit_wiener([part], labels, device, taps, ridge, verbose=False) if which == 'wiener' else \
            fit_kalman([part], labels, device, ridge, verbose=False)
        with torch.no_grad():
            if which == 'wiener':
                day.weight[part.day_idx].copy_(m.lin.weight)
                day.bias[part.day_idx].copy_(m.lin.bias)
            else:
                for k in ('M', 'K', 'mean_x', 'mean_z'):
                    getattr(day, k)[part.day_idx].copy_(getattr(m, k))
        frames += sum(len(l) for l in labels[part.session])
    print(f'  {which} per session: {len(parts)} filters, {frames:,} frames', flush=True)
    return day


def collapse_flicker(seq):
    """Per-frame argmax labels with every one-frame run replaced by the frame before it."""
    out = seq.copy()
    for t in range(1, len(seq) - 1):
        if seq[t] != seq[t - 1] and seq[t] != seq[t + 1]:
            out[t] = out[t - 1]
    return out


@torch.no_grad()
def diagnose(model, parts, labels, device):
    """Frame accuracy against the aligned frame labels, greedy PER, and greedy PER with one-frame flickers removed."""
    right = frames = edit = edit_f = phones = 0
    for part in parts:
        for i, feats in enumerate(part.features):
            out = model(torch.from_numpy(np.asarray(feats, np.float32))[None].to(device), part.day_idx)[0].cpu().numpy()
            lab = labels[part.session][i][:out.shape[0]]
            am = out.argmax(1)
            right += int((am[:len(lab)] == lab).sum())
            frames += len(lab)
            truth = part.labels[i]
            edit += edit_distance(greedy_decode(out, BLANK), truth)
            sm = collapse_flicker(am)
            dec = [int(c) for j, c in enumerate(sm) if c != BLANK and (j == 0 or c != sm[j - 1])]
            edit_f += edit_distance(dec, truth)
            phones += len(truth)
    return {'frame_acc': right / max(frames, 1), 'per': edit / phones, 'per_no_flicker': edit_f / phones}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--align-with', required=True, help='a trained control (.json + .pt) whose output is force-aligned')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--sessions', nargs='+', default=None, help='default: the 19 baseline sessions')
    p.add_argument('--taps', type=int, default=8, help='Wiener filter history in 20 ms bins (8 = 160 ms)')
    p.add_argument('--ridge', type=float, default=1e-3, help='ridge strength, relative to the mean input variance')
    p.add_argument('--which', nargs='+', default=['wiener', 'kalman'], choices=['wiener', 'kalman'])
    p.add_argument('--variants', nargs='+', default=['shared', 'session'], choices=['shared', 'session'],
                   help='shared: one filter for all sessions; session: one per session (daily recalibration)')
    p.add_argument('--targets', default='dense', choices=['dense', 'ctc'],
                   help='dense: each frame labelled with its phoneme (default); ctc: the aligned CTC path itself')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    sessions = a.sessions or list(json.load(open(a.align_with))['args']['sessions']) or list(BASELINE_SESSIONS)
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    train = load_partitions(a.data_dir, sessions, 'train', list(range(len(sessions))))
    test = load_partitions(a.data_dir, sessions, 'test', list(range(len(sessions))))
    ann_controls.N_IN = int(train[0].features[0].shape[1])
    print(f'{len(sessions)} sessions | aligning training sentences with {a.align_with}', flush=True)
    labels = aligned_labels(a.align_with, a.data_dir, sessions, device, dense=a.targets == 'dense')
    allab = np.concatenate([l for s_ in sessions for l in labels[s_]])
    majority = float(np.bincount(allab, minlength=NC).max() / len(allab))
    test_labels = aligned_labels(a.align_with, a.data_dir, sessions, device, dense=True, partition='test')
    print(f'alignment done ({time.time() - t0:.0f}s)', flush=True)
    report = []
    for which, variant in [(w, v) for v in a.variants for w in a.which]:
        t1 = time.time()
        if variant == 'shared':
            model = fit_wiener(train, labels, device, a.taps, a.ridge) if which == 'wiener' else \
                fit_kalman(train, labels, device, a.ridge)
        else:
            model = fit_per_session(which, train, labels, device, a.taps, a.ridge, len(sessions))
        scale = fit_scale(model, train, labels, device)
        per = test_per(model, test, device)
        name = (f'wiener_{a.taps * 20}ms' if which == 'wiener' else 'kalman_80ms') + ('_day' if variant == 'session' else '')
        tr = diagnose(model, train, labels, device)
        te = diagnose(model, test, test_labels, device)
        report.append((name, tr, te))
        if name not in ann_controls.MODELS:
            raise SystemExit(f'add {name} to ann_controls.MODELS')
        path = os.path.join(a.out_dir, name + '.json')
        n_params = sum(p.numel() for p in model.parameters())
        json.dump({'model': name, 'params_core': n_params, 'params_total': n_params,
                   'args': {'model': name, 'sessions': sessions, 'smooth_sd': 0.0, 'epochs': 1, 'recipe': 'closed form',
                            'align_with': a.align_with, 'taps': a.taps, 'ridge': a.ridge, 'scale': scale,
                            'targets': a.targets},
                   'history': {'test_per': [per], 'train_loss': [], 'test_loss': [], 'epoch_s': [time.time() - t1]}},
                  open(path, 'w'), indent=1)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.splitext(path)[0] + '.pt')
        print(f'{name}: test PER {per:.4f} (greedy) | logit scale {scale:.3g} | fitted in {time.time() - t1:.0f}s | '
              f'saved {path} and its .pt', flush=True)
    print('\nDiagnostics (frame accuracy against the aligned frame labels; PER greedy, and with one-frame flickers removed)')
    print(f"{'decoder':22s} {'frame acc train':>16s} {'frame acc test':>15s} {'PER train':>10s} {'PER test':>9s} {'PER test, no flicker':>21s}")
    for name, tr, te in report:
        print(f"{name:22s} {tr['frame_acc']:16.3f} {te['frame_acc']:15.3f} {tr['per']:10.3f} {te['per']:9.3f} {te['per_no_flicker']:21.3f}")
    print('frame accuracy of always predicting the most common label (train):', f"{majority:.3f}")


if __name__ == '__main__':
    main()
