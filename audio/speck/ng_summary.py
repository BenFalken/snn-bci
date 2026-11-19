"""
EXPLORATORY: the next-generation audio variants (runs_speck_ng.txt) against P5 (its seeds s1337 and s11), on VALIDATION (the chip model's event-
driven output with its readout refitted, r_event_refit, the number P5's calibration rests on). The decision rule, fixed
before the runs: a variant replaces P5 only if its mean validation r beats P5's for both monkeys. Test scores are
printed only with --show-test, after that decision.

    python3 audio/speck/explore_96ch_summary.py [--show-test]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import WORKSPACE  # noqa: E402

FINAL = WORKSPACE / "final_results"
SEEDS = ["s1337", "s11"]
VARIANTS = {"P5": ("audio_novel", "P5_{s}"), "NG2": ("audio_ng", "NG2_centre_ema1_deriv_{s}"),
            "NG4": ("audio_ng", "NG4_centre_ema1_deriv_{s}")}   # NG10 dropped 2026-11-19 (runs_speck_ng.txt)


def score(monkey, variant, split):
    folder, pattern = VARIANTS[variant]
    out = []
    for s in SEEDS:
        f = FINAL / folder / f"{monkey}_{pattern.format(s=s)}" / "refit" / "refit.json"
        if f.is_file():
            out.append(json.load(open(f))["r_event_refit"][split])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-test", action="store_true")
    a = ap.parse_args()
    splits = ["valid"] + (["test"] if a.show_test else [])
    beats = {v: True for v in VARIANTS if v != "P5"}
    for split in splits:
        print(f"\nchip model, readout refitted: r on {split} (mean over seeds; seeds in brackets)")
        for m in ("Neil", "Gomez"):
            base = score(m, "P5", split)
            for v in VARIANTS:
                r = score(m, v, split)
                if not r:
                    print(f"  {m:5s} {v:6s} (no results yet)")
                    if v != "P5":
                        beats[v] = False
                    continue
                d = f"  {np.mean(r) - np.mean(base):+.3f} vs P5" if v != "P5" and base else ""
                print(f"  {m:5s} {v:6s} {np.mean(r):.3f}  [{', '.join(f'{x:.3f}' for x in r)}]{d}")
                if split == "valid" and v != "P5" and (len(r) < 2 or not base or np.mean(r) <= np.mean(base)):
                    beats[v] = False
    print("\nDecision (validation, both monkeys): " + "; ".join(
        f"{v} {'BEATS P5' if ok else 'does not beat P5'}" for v, ok in beats.items()))


if __name__ == "__main__":
    main()
