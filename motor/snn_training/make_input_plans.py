"""
Per-session input plans for the Speck SNN (train_snn.py --input-plan-dir), from
channel_lag_tuning.py's per-channel results: which channels the chip gets, and
each one's delays (in 4 ms steps). The analysis used only each session's
training portion, so the plans never see test data.

    prune5   the channels tuned to velocity (peak R^2 >= --min-r2; silent and
             untuned ones dropped), each with the same 5 delays: 0, 60, 120,
             180, 240 ms (the span where the channels' information lies)
    tuned    every channel; a tuned one gets "now" plus 3 delays around its
             own best causal lead d: d - 60, d, d + 60 ms (shifted up to stay
             >= 4 ms and distinct); an untuned one the shared 0, 40, 100, 200 ms

Writes <out-dir>/<plan>/<session>.json = {"channels": [...], "delays": [[...] per copy]}.
    python snn_training/make_input_plans.py --lag-dir .../lag_tuning_all --out-dir .../input_plans
"""

import argparse
import glob
import json
import os

STEP_MS = 4
SHARED = [0, 10, 25, 50]                      # 0, 40, 100, 200 ms: speck_delay's delays
PRUNE5 = [0, 15, 30, 45, 60]                  # 0, 60, 120, 180, 240 ms
SPREAD = 15                                   # 60 ms either side of a channel's own lead
MAX_CHIP_INPUTS = 512                         # first layer of 128 on a 64 Ki-word core


def tuned_delays(lead_steps):
    """'now' plus 3 distinct delays >= 1 step centred on lead_steps."""
    d = [lead_steps - SPREAD, lead_steps, lead_steps + SPREAD]
    shift = max(0, 1 - d[0])
    return [0] + [x + shift for x in d]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lag-dir", required=True, help="channel_lag_tuning.py --out-dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-r2", type=float, default=0.02)
    args = p.parse_args()
    for plan in ("prune5", "tuned"):
        os.makedirs(os.path.join(args.out_dir, plan), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.lag_dir, "lag_tuning_*.json"))):
        if f.endswith("summary.json"):
            continue
        d = json.load(open(f))
        s, ch = d["session"], d["channels"]
        good = [c["channel"] for c in ch if c["peak_r2"] >= args.min_r2]
        if len(good) * len(PRUNE5) > MAX_CHIP_INPUTS:
            good = sorted(sorted(good, key=lambda c: -ch[c]["peak_r2"])[:MAX_CHIP_INPUTS // len(PRUNE5)])
        prune5 = {"channels": good, "delays": [[x] * len(good) for x in PRUNE5]}
        rows = [tuned_delays(int(round(c["best_causal_delay_ms"] / STEP_MS))) if c["peak_r2"] >= args.min_r2
                else SHARED for c in ch]
        tuned = {"channels": list(range(len(ch))), "delays": [list(r) for r in zip(*rows)]}
        for plan, content in (("prune5", prune5), ("tuned", tuned)):
            with open(os.path.join(args.out_dir, plan, f"{s}.json"), "w") as fh:
                json.dump(content | {"source": os.path.basename(f), "min_r2": args.min_r2}, fh)
        print(f"{s}: prune5 {len(good)} channels x 5 = {5 * len(good)} chip inputs; tuned 96 x 4 = 384, "
              f"leads {sorted({r[2] * STEP_MS for r in rows if r is not SHARED})[:3]}... ms")


if __name__ == "__main__":
    main()
