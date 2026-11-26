"""
Sentence files for video_speech.py, built on this machine (the speechBCI viewer's export needs the paper RNN, whose
checkpoint is no longer on Oscar): for each chosen test sentence, the neural features (threshold crossings and
spike-band power, z-scored over the sentence, as an 8-bit image), the true phonemes timed by CTC forced alignment to a
reference decoder's logits (here the LSNN's: 'alignedBy'), and the words timed by the runs of phonemes between
silences. Decoder lanes are added by the video script itself.

Sentences: --n drawn at random (--seed) from the paper's 600 scored test sentences, with no selection on any score.

    python3 speech_trials_local.py --align-logits LSNN_logits.npz --keys perTrialPER_final_v2c.json --n 12 --out DIR
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "speech"), str(HERE)]
from snnDecoder.dataset import load_partitions  # noqa: E402
from video_speech import ctc_align  # noqa: E402

NAMES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
         'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
         'SIL', 'blank']


def text(ascii_row):
    return "".join(chr(int(c)) for c in ascii_row if int(c) > 0).strip()


def heat(features):
    x = (features - features.mean(0)) / (features.std(0) + 1e-6)
    img = np.clip((x.T + 2.0) / 5.0, 0, 1) * 255                     # rows: features, cols: 20 ms bins
    return {"rows": int(img.shape[0]), "cols": int(img.shape[1]),
            "data": base64.b64encode(img.astype(np.uint8).tobytes()).decode(),
            "groups": [["threshold crossings", 0, 128], ["spike-band power", 128, 256]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/Downloads/speech_rebuild/derived/snnData"))
    ap.add_argument("--align-logits", required=True)
    ap.add_argument("--align-name", default="LSNN")
    ap.add_argument("--keys", required=True, help="perTrialPER json: draw from its sentences")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = json.load(open(a.keys))["keys"]
    pick = sorted(np.random.default_rng(a.seed).choice(len(keys), a.n, replace=False))
    chosen = [keys[i] for i in pick]
    z = np.load(a.align_logits)
    os.makedirs(a.out, exist_ok=True)
    parts = {}
    for k in chosen:
        s = k.split("/")[0]
        if s not in parts:
            parts[s] = load_partitions(a.data_dir, [s], "test")[0]
    for k in chosen:
        s, i = k.split("/")[0], int(k.split("/")[1])
        p = parts[s]
        feats, lab = p.features[i], [int(c) for c in p.labels[i]]
        sent = text(p.transcriptions[i])
        ph = [[NAMES[c], 80.0 * f0, 80.0 * (f1 + 1)] for c, f0, f1 in ctc_align(z[k], lab)]
        runs, cur = [], []
        for seg in ph + [["SIL", 0, 0]]:
            if seg[0] == "SIL":
                if cur:
                    runs.append((cur[0][1], cur[-1][2]))
                cur = []
            else:
                cur.append(seg)
        words = [[w, a_, b_] for w, (a_, b_) in zip(sent.split(), runs)]
        d = {"session": s, "partition": "test", "trial": i, "sentence": sent, "binMs": 20.0,
             "nBins": int(feats.shape[0]), "classNames": NAMES, "heat": heat(np.asarray(feats, float)),
             "truth": {"phonemes": ph, "words": words, "alignedBy": a.align_name}, "decoders": []}
        json.dump(d, open(os.path.join(a.out, f"trial_{s}_test_{i}.json"), "w"))
        print(f"{k}: '{sent}' ({feats.shape[0] * 0.02:.1f} s, {len(sent.split())} words, {len(runs)} phoneme runs)")


if __name__ == "__main__":
    main()
