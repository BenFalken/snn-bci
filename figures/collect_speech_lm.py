"""
Speech, per test sentence, with the phoneme 5-gram (speech/scripts/ngram_lm.py): each decoder's saved logits decoded
by beam search with the one language model every decoder shares, each sentence at the setting (alpha, beta) chosen on
the other half of the sentences (ngram_lm.py's 'chosen': half<trial % 2>), exactly as the pooled scores were made.
The pooled phoneme error over the 600 sentences is checked against ngram_lm.py's cross_tuned_per.

    python3 collect_speech_lm.py   -> data/speech_lm_sentences.csv (experiment, group, session, decoder, metric, value)
                                      decoders paper_rnn_lm, lsnn_lm, speck_lm (Speck = round 21 S3 on the real chip)
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "speech"), str(HERE.parent / "speech" / "scripts")]
from ngram_lm import KneserNey, beam_decode  # noqa: E402
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.utils import edit_distance  # noqa: E402

N = WORKSPACE / "exploratory_ngram_2026-11-24"
RUNS = {"paper_rnn_lm": (WORKSPACE / "logits_final/RNN_logits.npz", N / "same_lm_RNN_600_o5.json"),
        "lsnn_lm": (WORKSPACE / "logits_final/LSNN_logits.npz", N / "same_lm_LSNN_600_o5.json"),
        "speck_lm": (WORKSPACE / "final_results/speech_r21s3/chip_refit_logits.npz", N / "same_lm_S3_chip_600_o5.json")}
DATA = WORKSPACE / "speech_rebuild/derived/snnData"
KEYS = WORKSPACE / "final_results/perTrialPER_final_v2c.json"
_G = {}


def _work(job):
    name, key = job
    al, be = _G["chosen"][name][f"half{int(key.split('/')[1]) % 2}"]
    dec = beam_decode(_G["logits"][name][key].astype(np.float64), _G["lm"], al, be, _G["beam"][name])
    return name, key, edit_distance(dec, _G["target"][key])


def main():
    keys = json.load(open(KEYS))["keys"]
    sessions = sorted(x for x in os.listdir(DATA) if os.path.isfile(DATA / x / "train.npz"))
    test = load_partitions(str(DATA), sessions, "test")
    target = {f"{p.session}/{i}": list(p.labels[i]) for p in test for i in range(len(p))}
    scored = {tuple(int(x) for x in t) for p in test for t in p.labels}
    lm_json = {n: json.load(open(j)) for n, (_, j) in RUNS.items()}
    order = {j["order"] for j in lm_json.values()}
    assert len(order) == 1, order
    kept = [s for p in load_partitions(str(DATA), sessions, "train") for s in p.labels
            if tuple(int(x) for x in s) not in scored]
    _G.update(lm=KneserNey(kept, order.pop()), target=target,
              chosen={n: j["chosen"] for n, j in lm_json.items()}, beam={n: j.get("beam", 32) for n, j in lm_json.items()},
              logits={n: dict(np.load(f)) for n, (f, _) in RUNS.items()})
    with mp.get_context("fork").Pool(8) as pool:
        out = pool.map(_work, [(n, k) for n in RUNS for k in keys], chunksize=8)
    rows = []
    for n in RUNS:
        e = {k: d for m, k, d in out if m == n}
        pooled = sum(e.values()) / sum(len(target[k]) for k in keys)
        ref = lm_json[n]["cross_tuned_per"]
        print(f"{n:13s} pooled {pooled:.4f} (ngram_lm.py {ref:.4f}) | median per sentence "
              f"{np.median([e[k] / len(target[k]) for k in keys]):.3f}")
        assert abs(pooled - ref) < 1e-9, (n, pooled, ref)
        rows += [("speech", "sentence", k, n, "per", e[k] / len(target[k])) for k in keys]
    pd.DataFrame(rows, columns=["experiment", "group", "session", "decoder", "metric", "value"]).to_csv(
        HERE / "data" / "speech_lm_sentences.csv", index=False)
    print("wrote data/speech_lm_sentences.csv")


if __name__ == "__main__":
    main()
