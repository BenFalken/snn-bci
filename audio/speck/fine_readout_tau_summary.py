"""
EXPLORATORY: the Speck SNN's readout time constant (fine_readout_tau.sh), on VALIDATION, by the paper's metric
(per-word band r, median over words), mean over each run's calibrated seeds, chip model, readout refitted.

Rule, fixed before the refits: for FT1 (the 4 ms network), the factor with the best mean over both monkeys replaces
the trained time constant (factor 1) only if it beats factor 1 for BOTH monkeys; otherwise factor 1 stays. P5 is
shown as the control (is any gain specific to real spike timing?) and is not changed. Test only with --show-test.

    python3 audio/speck/fine_readout_tau_summary.py [--show-test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fine_timing_summary as fs  # noqa: E402

FACTORS = ["1", "0.5", "0.25", "0.125"]
RUNS = {"FT1": "FT1_spikes", "P5": "P5"}


def score(m, run, f, split):
    y, spans = fs.words(m, split)
    folder = "refit" if f == "1" else f"refit_tau{f}"
    name = "preds_event_refit" + ("_valid" if split == "valid" else "") + ".npy"
    r = []
    for s in fs.SEEDS:
        q = fs.FINAL / "audio_fine" / f"{m}_{run}_{s}" / folder / name
        if q.is_file():
            r.append(fs.per_word(y, spans, np.load(q)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-test", action="store_true")
    a = ap.parse_args()
    for split in ["valid"] + (["test"] if a.show_test else []):
        print(f"\nper-word band r on {split}, chip model, readout refitted (mean over seeds; n seeds)")
        print("                 " + "  ".join(f"x{f:<6s}" for f in FACTORS))
        tab = {}
        for name, run in RUNS.items():
            for m in ("Neil", "Gomez"):
                row = [score(m, run, f, split) for f in FACTORS]
                tab[(name, m)] = [np.mean(r) if r else np.nan for r in row]
                print(f"  {name:4s} {m:6s}    " + "  ".join(f"{np.mean(r):.3f}({len(r)})" if r else "   --   " for r in row))
        if split == "valid":
            ft = np.array([tab[("FT1", m)] for m in ("Neil", "Gomez")])
            best = int(np.nanargmax(ft[:, 1:].mean(0))) + 1
            ok = all(ft[i, best] > ft[i, 0] for i in range(2))
            choice = FACTORS[best] if ok else "1"
            print(f"\nDecision (FT1, validation): factor x{choice}" + (f" (beats x1 for both monkeys)" if ok else
                  f" (x{FACTORS[best]} was best on average but does not beat x1 for both monkeys)"))
            print(f"FT1_TAU_FACTOR={choice}")


if __name__ == "__main__":
    main()
