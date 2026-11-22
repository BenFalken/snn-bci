"""
EXPLORATORY: the spike-timing pilot (runs_fine_timing.txt) on VALIDATION, seeds s1337 and s11 (P5's calibrated seeds).

  Speck  FT1 (real 4 ms spikes, chip-ready): the chip model's event-driven output, readout refitted (r_event_refit),
         against P5's. FT2 (events weighted per day; not chip-ready as is): the software network, readout refitted
         (r_software_refit), against P5's software number.
  LSNN   LS1 (streaming at 4 ms) against LS0 (streaming at 40 ms, the same batches), each scored streaming over the
         whole recording (its validation r equals the training log's best).

Decision rule, fixed before the runs: a variant counts as an improvement only if its mean validation r beats its
control's for both monkeys. Test scores only with --show-test, after that decision.

    python3 audio/speck/fine_timing_summary.py [--show-test]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import WORKSPACE  # noqa: E402

FINAL = WORKSPACE / "final_results"
SEEDS = ["s1337", "s11"]
SPECK = {"P5": ("audio_novel", "P5_{s}"), "FT1": ("audio_fine", "FT1_spikes_{s}"),
         "FT2": ("audio_fine", "FT2_spikes_daynorm_{s}")}
LSNN = {"LS0": "LS0_stream40_{s}", "LS1": "LS1_fine_{s}"}


def speck(monkey, v, key, split):
    folder, pat = SPECK[v]
    out = []
    for s in SEEDS:
        f = FINAL / folder / f"{monkey}_{pat.format(s=s)}" / "refit" / "refit.json"
        if f.is_file():
            out.append(json.load(open(f))[key][split])
    return out


def lsnn(monkey, v, split):
    """Each seed's r, streaming over the whole recording (snndec.evaluate, as fine_timing.sh writes it)."""
    out = []
    for s in SEEDS:
        f = FINAL / "audio_fine" / f"{monkey}_{LSNN[v].format(s=s)}" / "eval.txt"
        m = re.search(rf"r_{split}: ([0-9.]+)", f.read_text()) if f.is_file() else None
        if m:
            out.append(float(m.group(1)))
    return out


def line(m, name, r, base):
    if not r:
        return f"  {m:5s} {name:4s} (no results yet)"
    d = f"  {np.mean(r) - np.mean(base):+.3f}" if base and r is not base else ""
    return f"  {m:5s} {name:4s} {np.mean(r):.3f}  [{', '.join(f'{x:.3f}' for x in r)}]{d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-test", action="store_true")
    a = ap.parse_args()
    ok = {"FT1": True, "FT2": True, "LS1": True}
    for split in ["valid"] + (["test"] if a.show_test else []):
        groups = [("Speck, chip model, readout refitted", [("P5", "r_event_refit"), ("FT1", "r_event_refit")]),
                  ("Speck, software, readout refitted", [("P5", "r_software_refit"), ("FT2", "r_software_refit")])]
        for title, rows in groups:
            print(f"\n{title}: r on {split} (mean over seeds; seeds in brackets; difference to P5)")
            for m in ("Neil", "Gomez"):
                base = speck(m, "P5", rows[0][1], split)
                for v, key in rows:
                    r = speck(m, v, key, split)
                    print(line(m, v, r, base if v != "P5" else None))
                    if split == "valid" and v != "P5" and (len(r) < 2 or not base or np.mean(r) <= np.mean(base)):
                        ok[v] = False
        print(f"\nLSNN, streaming: r on {split} (difference to LS0)")
        for m in ("Neil", "Gomez"):
            base = lsnn(m, "LS0", split)
            for v in LSNN:
                r = lsnn(m, v, split)
                print(line(m, v, r, base if v != "LS0" else None))
                if split == "valid" and v == "LS1" and (len(r) < 2 or not base or np.mean(r) <= np.mean(base)):
                    ok["LS1"] = False
    print("\nDecision (validation, both monkeys): " + "; ".join(
        f"{v} {'BEATS its control' if b else 'does not beat its control'}" for v, b in ok.items()))


if __name__ == "__main__":
    main()
