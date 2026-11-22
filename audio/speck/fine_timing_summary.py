"""
EXPLORATORY: the spike-timing pilot (runs_fine_timing.txt) on VALIDATION, seeds s1337 and s11 (P5's calibrated seeds; s99 added later).

  Speck  FT1 (real 4 ms spikes, chip-ready): the chip model's event-driven output, readout refitted (r_event_refit),
         against P5's. FT2 (events weighted per day; not chip-ready as is): the software network, readout refitted
         (r_software_refit), against P5's software number.
  LSNN   LS1 (streaming at 4 ms) against LS0 (streaming at 40 ms, the same batches), each scored streaming over the
         whole recording (its validation r equals the training log's best).

Then the same on the paper's audio metric, per-word band r (each word's Fisher-mean band correlation; the median over
the validation words), with every pipeline decoder's for reference. The decision is made on this metric.

Decision rule, fixed before the runs: a variant counts as an improvement only if its mean validation r beats its
control's for both monkeys. Test scores only with --show-test, after that decision.

    python3 audio/speck/fine_timing_summary.py [--show-test]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import AUDITORY_DATA, FIGURES, WORKSPACE  # noqa: E402

sys.path.insert(0, str(FIGURES))
from collect_fidelity import fisher_mean_r  # noqa: E402

FINAL = WORKSPACE / "final_results"
SEEDS = ["s1337", "s11", "s99"]                      # s99 added after the two-seed result
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


SAMPLES_BEFORE = 8
PIPELINE = ["lstm_rnn", "gru_rnn", "lsnn", "simple_rnn", "wiener_filter", "kalman_filter"]


def words(monkey, split):
    """The split's window targets and each word's rows in them (window i targets frame i + 8)."""
    base = Path(AUDITORY_DATA) / monkey / "aSTG" / "ALLWORDS_NOVEL_DAYNORM"
    eps = [r for r in csv.DictReader(open(base / "prep" / "epochs.csv")) if r["split"] == split]
    b0 = min(int(r["frame_start"]) for r in eps) - SAMPLES_BEFORE
    y = np.load(base / "pipeline" / f"window_targets_{split}.npy")
    spans = [(int(r["frame_start"]) - SAMPLES_BEFORE - b0, int(r["frame_end"]) - SAMPLES_BEFORE - b0) for r in eps]
    return y, [(a, b) for a, b in spans if a >= 0 and b <= len(y)]


def per_word(y, spans, p):
    return float(np.median([fisher_mean_r(y[a:b], p[a:b]) for a, b in spans]))


def word_scores(monkey, split):
    """{name: [per-seed per-word median]}: the pilot's runs, P5 (refit again in audio_fine) and the pipeline's."""
    y, spans = words(monkey, split)
    suf = "_valid" if split == "valid" else ""
    out = {}
    runs = {"P5 chip": ("P5_{s}", f"refit/preds_event_refit{suf}.npy"),
            "P5 software": ("P5_{s}", f"refit/preds_software_refit{suf}.npy"),
            "FT1 chip": ("FT1_spikes_{s}", f"refit/preds_event_refit{suf}.npy"),
            "FT2 software": ("FT2_spikes_daynorm_{s}", f"refit/preds_software_refit{suf}.npy"),
            "LS0": ("LS0_stream40_{s}", f"preds/{split}.npy"), "LS1": ("LS1_fine_{s}", f"preds/{split}.npy")}
    for name, (pat, f) in runs.items():
        r = []
        for s in SEEDS:
            q = FINAL / "audio_fine" / f"{monkey}_{pat.format(s=s)}" / f
            if q.is_file():
                r.append(per_word(y, spans, np.load(q)))
        out[name] = r
    base = Path(AUDITORY_DATA) / monkey / "aSTG" / "ALLWORDS_NOVEL_DAYNORM" / "pipeline"
    for d in PIPELINE:
        q = base / f"{d}_predict_{split}.npy"
        out[d] = [per_word(y, spans, np.load(q))] if q.is_file() else []
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
    word_ok = {"FT1": True, "FT2": True, "LS1": True}
    pairs = {"FT1": ("FT1 chip", "P5 chip"), "FT2": ("FT2 software", "P5 software"), "LS1": ("LS1", "LS0")}
    for split in ["valid"] + (["test"] if a.show_test else []):
        print(f"\nThe paper's metric, per-word band r on {split} (median over words; mean over seeds)")
        for m in ("Neil", "Gomez"):
            w = word_scores(m, split)
            for name, r in w.items():
                print(f"  {m:5s} {name:13s} " + (f"{np.mean(r):.3f}" + (f"  [{', '.join(f'{x:.3f}' for x in r)}]"
                                                  if len(r) > 1 else "") if r else "(no results yet)"))
            if split == "valid":
                for v, (x, c) in pairs.items():
                    if len(w[x]) < 2 or not w[c] or np.mean(w[x]) <= np.mean(w[c]):
                        word_ok[v] = False
    print("\nDecision on the paper's metric (validation, both monkeys): " + "; ".join(
        f"{v} {'BEATS its control' if b else 'does not beat its control'}" for v, b in word_ok.items()))
    print("Decision on whole-recording r (validation, both monkeys): " + "; ".join(
        f"{v} {'BEATS its control' if b else 'does not beat its control'}" for v, b in ok.items()))


if __name__ == "__main__":
    main()
