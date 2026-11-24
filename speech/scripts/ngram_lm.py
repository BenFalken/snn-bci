"""
A phoneme n-gram language model for CTC phoneme decoders, applied to saved logits by prefix beam search.

The n-gram (interpolated Kneser-Ney, word boundaries included as a phoneme) is fitted on the TRAINING sentences'
phoneme labels only, leaving out any training sentence identical to a scored one (2.8% of test sentences repeat a
training sentence). Decoding adds alpha * log P_lm per emitted phoneme and beta per phoneme (a length bonus) to the
CTC prefix score. alpha and beta are chosen by two-fold cross-tuning on the scored sentences (alternate sentences of
each session): each half is decoded with the setting that was best on the other half, so no sentence is scored with a
setting tuned on itself. alpha = 0, beta = 0 is the beam search without a language model (close to greedy).

    python ngram_lm.py --logits chip_logits.npz --data-dir snnData --order 4 --out lm.json

The logits file is the one speech_chip.py --save-logits writes: keys 'session/i', (T, n_classes), blank last.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from snnDecoder.dataset import load_partitions  # noqa: E402
from speech_chip import edit_distance, greedy_decode  # noqa: E402

BOS, EOS = -1, -2


class KneserNey:
    """Interpolated Kneser-Ney n-gram over integer tokens (discount 0.75), with BOS/EOS padding."""

    def __init__(self, sentences, order=4, d=0.75):
        self.n, self.d = order, d
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]   # [k][context][token]
        for s in sentences:
            seq = [BOS] * (order - 1) + [int(x) for x in s] + [EOS]
            for i in range(order - 1, len(seq)):
                for k in range(order):            # context length k
                    self.counts[k][tuple(seq[i - k:i])][seq[i]] += 1
        # continuation counts for lower orders: distinct left contexts per (context, token)
        self.cont = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        for k in range(1, order):
            for ctx, toks in self.counts[k].items():
                for t in toks:
                    self.cont[k - 1][ctx[1:]][t] += 1
        self.vocab = sorted({t for toks in self.counts[0].values() for t in toks})
        self.cache = {}

    def _table(self, k):
        return self.counts[k] if k == self.n - 1 else self.cont[k]

    def prob(self, ctx, tok, k=None):
        ctx = tuple(ctx[-(self.n - 1):]) if self.n > 1 else ()
        k = len(ctx) if k is None else k
        key = (ctx[len(ctx) - k:], tok, k)
        if key in self.cache:
            return self.cache[key]
        c = ctx[len(ctx) - k:]
        if k == 0:
            tab = self._table(0)[()]
            total = sum(tab.values())
            p = (max(tab.get(tok, 0) - self.d, 0) + self.d * len(tab) / len(self.vocab)) / total
        else:
            tab = self._table(k).get(c)
            lower = self.prob(ctx, tok, k - 1)
            if not tab:
                p = lower
            else:
                total = sum(tab.values())
                p = max(tab.get(tok, 0) - self.d, 0) / total + self.d * len(tab) / total * lower
        self.cache[key] = p
        return p

    def logp(self, history, tok):
        ctx = [BOS] * (self.n - 1) + list(history)
        return math.log(max(self.prob(ctx, tok), 1e-12))


def _lse(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(logits, lm, alpha, beta, beam=16, prune=1e-3):
    """CTC prefix beam search; each prefix keeps (log P_blank-ending, log P_nonblank-ending) and its LM score."""
    x = logits - logits.max(1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(1, keepdims=True))
    blank = logits.shape[1] - 1
    beams = {(): (0.0, -math.inf)}
    lm_score = {(): 0.0}
    for t in range(len(logp)):
        row = logp[t]
        cand = [c for c in np.flatnonzero(row > math.log(prune)) if c != blank] or [int(np.argmax(row[:-1]))]
        nxt = defaultdict(lambda: [-math.inf, -math.inf])
        for pre, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            e = nxt[pre]                                          # blank: prefix unchanged
            e[0] = _lse(e[0], tot + row[blank])
            for c in cand:
                c = int(c)
                p = row[c]
                if pre and pre[-1] == c:
                    e[1] = _lse(e[1], pnb + p)                    # repeat without blank: same prefix
                    ext = pb                                      # a second c needs a blank between
                else:
                    ext = tot
                new = pre + (c,)
                if new not in lm_score:
                    lm_score[new] = lm_score[pre] + (alpha * lm.logp(pre, c) + beta if alpha or beta else 0.0)
                n = nxt[new]
                n[1] = _lse(n[1], ext + p)
        scored = sorted(nxt.items(), key=lambda kv: -(_lse(*kv[1]) + lm_score[kv[0]]))[:beam]
        beams = {k: tuple(v) for k, v in scored}
    final = max(beams, key=lambda k: _lse(*beams[k]) + lm_score[k] + (alpha * lm.logp(k, EOS) if alpha else 0.0))
    return list(final)


_G = {}


def _work(args):
    key, alpha, beta, beam = args
    return key, alpha, beta, beam_decode(_G['logits'][key], _G['lm'], alpha, beta, beam)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--logits', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--partition', default='test')
    p.add_argument('--keys', default=None, help='perTrialPER json: score only its sentences (the paper\'s 600)')
    p.add_argument('--order', type=int, default=4)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.8, 1.0, 1.5])
    p.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument('--beam', type=int, default=16)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    z = np.load(a.logits)
    logits = {k: z[k].astype(np.float64) for k in z.keys()}
    sessions = sorted({k.split('/')[0] for k in logits})
    train = load_partitions(a.data_dir, sessions, 'train')
    test = load_partitions(a.data_dir, sessions, a.partition)
    target = {f'{part.session}/{i}': list(part.labels[i]) for part in test for i in range(len(part))}
    # never fit the LM on a sentence that is scored: training sentences identical to any scored sentence are left out
    scored = {tuple(int(x) for x in t) for t in target.values()}
    corpus = [s for part in train for s in part.labels]
    kept = [s for s in corpus if tuple(int(x) for x in s) not in scored]
    lm = KneserNey(kept, a.order)
    keys = [k for k in logits if k in target]
    if a.keys:
        wanted = set(json.load(open(a.keys))['keys'])
        keys = [k for k in keys if k in wanted]
        assert len(keys) == len(wanted), f'{len(wanted) - len(keys)} of the --keys sentences have no logits'
    fold = {k: int(k.split('/')[1]) % 2 for k in keys}
    greedy = sum(edit_distance(greedy_decode(logits[k], logits[k].shape[1] - 1), target[k]) for k in keys) / \
        sum(len(target[k]) for k in keys)
    print(f'{len(keys)} sentences; {a.order}-gram on {len(kept)} training sentences ({len(corpus) - len(kept)} '
          f'identical to a scored sentence left out); '
          f'greedy PER {greedy:.4f}', flush=True)

    _G.update(logits=logits, lm=lm)
    grid = [(al, be) for al in a.alphas for be in a.betas]
    with mp.get_context('fork').Pool(a.jobs) as pool:     # forked workers share the logits and the LM
        decoded = pool.map(_work, [(k, al, be, a.beam) for (al, be) in grid for k in keys], chunksize=8)
    err = defaultdict(lambda: [0, 0, 0, 0])                    # (alpha, beta) -> [edits f0, len f0, edits f1, len f1]
    for k, al, be, dec in decoded:
        f = fold[k]
        err[(al, be)][2 * f] += edit_distance(dec, target[k])
        err[(al, be)][2 * f + 1] += len(target[k])
    per = {g: ((e[0] + e[2]) / (e[1] + e[3]), e[0] / e[1], e[2] / e[3]) for g, e in err.items()}
    best = [min(grid, key=lambda g: per[g][1 + f]) for f in (0, 1)]           # best on each half
    cross = (err[best[1]][0] + err[best[0]][2]) / (err[best[1]][1] + err[best[0]][3])   # each half: the other's best
    no_lm = per[(0.0, 0.0)][0] if (0.0, 0.0) in per else None
    print(f'beam without LM {no_lm:.4f} | cross-tuned with LM {cross:.4f} '
          f'(half A scored at alpha, beta = {best[1]}, half B at {best[0]})')
    json.dump({'logits': a.logits, 'order': a.order, 'lm_sentences': len(kept), 'left_out': len(corpus) - len(kept), 'beam': a.beam, 'greedy_per': greedy, 'beam_no_lm_per': no_lm,
               'cross_tuned_per': cross, 'chosen': {'half0': best[1], 'half1': best[0]},
               'grid': {f'{al},{be}': per[(al, be)][0] for al, be in grid}}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
