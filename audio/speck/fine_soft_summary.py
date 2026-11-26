"""
EXPLORATORY: soft reset for the 4 ms audio Speck SNN (runs_fine_soft.txt) against FT1 (hard reset, the final audio
Speck SNN), on VALIDATION, three seeds each. Every number is the chip model's event-driven output with the readout
refitted, as FT1 was calibrated.

Decision rule, fixed before the runs: a variant counts as an improvement only if its mean per-word band r (the paper's
metric) beats FT1's for both monkeys on validation. Test scores only with --show-test, after that decision.

    python3 audio/speck/fine_soft_summary.py [--show-test]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fine_timing_summary as fts  # noqa: E402  (words, per_word, line, FINAL, SEEDS)

VARIANTS = {"FT1": "FT1_spikes_{s}", "FS1": "FS1_soft_{s}", "FS3": "FS3_soft_floor3_{s}"}


def whole(monkey, v, split):
    out = []
    for s in fts.SEEDS:
        f = fts.FINAL / "audio_fine" / f"{monkey}_{VARIANTS[v].format(s=s)}" / "refit" / "refit.json"
        if f.is_file():
            out.append(json.load(open(f))["r_event_refit"][split])
    return out


def per_word(monkey, v, split):
    y, spans = fts.words(monkey, split)
    suf = "_valid" if split == "valid" else ""
    out = []
    for s in fts.SEEDS:
        q = fts.FINAL / "audio_fine" / f"{monkey}_{VARIANTS[v].format(s=s)}" / "refit" / f"preds_event_refit{suf}.npy"
        if q.is_file():
            out.append(fts.per_word(y, spans, np.load(q)))
    return out


def scale(monkey, v):
    out = []
    for s in fts.SEEDS:
        f = fts.FINAL / "audio_fine" / f"{monkey}_{VARIANTS[v].format(s=s)}" / "sweep.json"
        if f.is_file():
            out.append(json.load(open(f))["chosen_scale"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-test", action="store_true")
    a = ap.parse_args()
    ok = {v: True for v in VARIANTS if v != "FT1"}
    for split in ["valid"] + (["test"] if a.show_test else []):
        for title, fn in [("whole-recording r", whole), ("per-word band r (the paper's metric)", per_word)]:
            print(f"\nSpeck, chip model, readout refitted: {title} on {split} (mean over seeds; difference to FT1)")
            for m in ("Neil", "Gomez"):
                base = fn(m, "FT1", split)
                for v in VARIANTS:
                    r = fn(m, v, split)
                    print(fts.line(m, v, r, base if v != "FT1" else None)
                          + (f"   layer-1 factors {scale(m, v)}" if fn is whole and split == "valid" else ""))
                    if fn is per_word and split == "valid" and v != "FT1" and \
                            (len(r) < 3 or len(base) < 3 or np.mean(r) <= np.mean(base)):
                        ok[v] = False
    print("\nDecision on the paper's metric (validation, both monkeys, 3 seeds): " + "; ".join(
        f"{v} {'BEATS FT1' if b else 'does not beat FT1'}" for v, b in ok.items()))


if __name__ == "__main__":
    main()
