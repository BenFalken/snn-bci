"""
Speech: what kind of phoneme errors each decoder makes (substituted, deleted, inserted) and where, over the 600 test
sentences, from the same saved logits as collect_speech_lm.py (paper RNN, LSNN, Speck SNN = round 21 S3 on the chip).

    greedy      greedy CTC (argmax, repeats merged, blanks dropped), each sentence aligned to its true phonemes
                (Levenshtein; every alignment's error count equals snnDecoder.utils.edit_distance)
    position    error rate of the true phonemes by place in the word (first / middle / last; word boundaries are the
                labels' SIL tokens), deletions and substitutions apart
    rate        error rate by speaking rate (true phonemes per second of the trial, from the Speck logits' 80 ms
                frames), thirds of the sentences
    blank       greedy decoding with the blank's log-probability lowered by b: does a deleted phoneme come back right?
                b chosen on one half of the sentences (trial index parity, as the 5-gram's settings) and scored on
                the other
    lm          the same alignment after the phoneme 5-gram, decoded exactly as collect_speech_lm.py does

    python3 collect_speech_errors.py   -> data/speech_errors.json
"""
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

import collect_speech_lm as C
from collect_speech_lm import KneserNey, beam_decode, load_partitions, edit_distance

SIL = 39
POS = ("first", "middle", "last")
PENALTY = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
NAMES = {"paper_rnn_lm": "paper_rnn", "lsnn_lm": "lsnn", "speck_lm": "speck"}
_G = {}


def greedy(lg):
    a = lg.argmax(1)
    keep = np.r_[True, a[1:] != a[:-1]] & (a != lg.shape[1] - 1)
    return [int(c) for c in a[keep]]


def align(ref, hyp):
    """Per true phoneme 'C', 'S' or 'D', and the number of insertions."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, ins, i, j = [None] * n, 0, n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops[i - 1] = "C" if ref[i - 1] == hyp[j - 1] else "S"; i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops[i - 1] = "D"; i -= 1
        else:
            ins += 1; j -= 1
    return ops, ins


def word_position(ref):
    """Place of each true phoneme in its word (None for SIL and one-phoneme words)."""
    out, word = [None] * len(ref), []
    for k, t in enumerate(list(ref) + [SIL]):
        if t != SIL:
            word.append(k); continue
        if len(word) > 1:
            for q, w in enumerate(word):
                out[w] = "first" if q == 0 else "last" if q == len(word) - 1 else "middle"
        word = []
    return out


def tally(pairs):
    """pairs: (ref, hyp) -> rates of S, D, I per true phoneme, by word position too."""
    t, pos = defaultdict(int), {p: defaultdict(int) for p in POS}
    for ref, hyp in pairs:
        ops, ins = align(ref, hyp)
        assert ops.count("S") + ops.count("D") + ins == edit_distance(hyp, ref)
        t["N"] += len(ref); t["I"] += ins
        for o, p in zip(ops, word_position(ref)):
            t[o] += 1
            if p:
                pos[p]["N"] += 1; pos[p][o] += 1
    r = {x: t[x] / t["N"] for x in "SDI"}
    r["per"] = r["S"] + r["D"] + r["I"]
    r["position"] = {p: {x: pos[p][x] / pos[p]["N"] for x in "SD"} | {"n": pos[p]["N"]} for p in POS}
    return r


def _beam(job):
    n, k = job
    al, be = _G["chosen"][n][f"half{int(k.split('/')[1]) % 2}"]
    return n, k, beam_decode(_G["logits"][n][k].astype(np.float64), _G["lm"], al, be, _G["beam"][n])


def main():
    keys = json.load(open(C.KEYS))["keys"]
    D = str(C.DATA)
    sessions = sorted(x for x in os.listdir(D) if os.path.isfile(f"{D}/{x}/train.npz"))
    test = load_partitions(D, sessions, "test")
    target = {f"{p.session}/{i}": [int(x) for x in p.labels[i]] for p in test for i in range(len(p))}
    logits = {n: dict(np.load(f)) for n, (f, _) in C.RUNS.items()}
    halves = [[k for k in keys if int(k.split("/")[1]) % 2 == h] for h in (0, 1)]
    speck = logits["speck_lm"]
    rate = {k: sum(t != SIL for t in target[k]) / (speck[k].shape[0] * 0.08) for k in keys}
    cuts = np.quantile(list(rate.values()), [1 / 3, 2 / 3])
    third = {k: int(np.digitize(rate[k], cuts)) for k in keys}
    out = {"n_sentences": len(keys), "rate_cuts_per_s": cuts.tolist(), "penalty": PENALTY, "decoders": {}}

    lm = {n: json.load(open(j)) for n, (_, j) in C.RUNS.items()}
    scored = {tuple(t) for t in target.values()}
    kept = [s for p in load_partitions(D, sessions, "train") for s in p.labels
            if tuple(int(x) for x in s) not in scored]
    order = {j["order"] for j in lm.values()}
    assert len(order) == 1, order
    _G.update(lm=KneserNey(kept, order.pop()), chosen={n: j["chosen"] for n, j in lm.items()},
              beam={n: j.get("beam", 32) for n, j in lm.items()}, logits=logits)
    with mp.get_context("fork").Pool(8) as pool:
        beams = pool.map(_beam, [(n, k) for n in C.RUNS for k in keys], chunksize=8)

    for n, L in logits.items():
        dec = {k: greedy(L[k]) for k in keys}
        g = tally((target[k], dec[k]) for k in keys)
        g["rate"] = [tally((target[k], dec[k]) for k in keys if third[k] == h) for h in range(3)]
        sweep = []
        for b in PENALTY:
            def pen(k):
                lg = L[k].astype(np.float64).copy(); lg[:, -1] -= b
                return greedy(lg)
            sweep.append([tally((target[k], pen(k)) for k in hk) for hk in halves] + [None])
            sweep[-1][2] = tally((target[k], pen(k)) for k in keys)
        chosen = [int(np.argmin([s[1 - h]["per"] for s in sweep])) for h in (0, 1)]
        n_half = [sum(len(target[k]) for k in hk) for hk in halves]
        cross = sum(sweep[chosen[h]][h]["per"] * n_half[h] for h in (0, 1)) / sum(n_half)
        lmr = tally((target[k], d) for m, k, d in beams if m == n)
        assert abs(lmr["per"] - lm[n]["cross_tuned_per"]) < 1e-9, (n, lmr["per"], lm[n]["cross_tuned_per"])
        out["decoders"][NAMES[n]] = {"greedy": g, "lm": lmr,
                                     "blank": {"all": [s[2] for s in sweep],
                                               "chosen": [PENALTY[c] for c in chosen], "cross_per": cross}}
        print(f"{NAMES[n]:10s} greedy PER {g['per']:.3f} (S {g['S']:.3f} D {g['D']:.3f} I {g['I']:.3f}) | "
              f"blank penalty {[PENALTY[c] for c in chosen]} -> {cross:.3f} | 5-gram {lmr['per']:.3f} "
              f"(S {lmr['S']:.3f} D {lmr['D']:.3f} I {lmr['I']:.3f})")
    json.dump(out, open(C.HERE / "data" / "speech_errors.json", "w"), indent=1)
    print("wrote data/speech_errors.json")


if __name__ == "__main__":
    main()
