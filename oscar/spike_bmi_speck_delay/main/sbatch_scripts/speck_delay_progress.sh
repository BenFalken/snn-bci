#!/bin/bash
# Progress of the Speck SNN with input delays (SPECK_MODEL=speck_delay) against the per-session Speck SNN.
#   [SPECK_MODEL=speck_delay_long] bash speck_delay_progress.sh [jobid ...]   (job ids: that model's arrays)
# 1. live: per task, the newest epoch, its test loss and the best so far (from the logs; Python buffers its
#    output, so a log can lag its checkpoints by a few epochs -- section 2 is the up-to-date one)
# 2. per session, the best test loss of both networks from their checkpoints: sqrt(MSE) of the velocity
#    in physical units, averaged over x and y (per-axis RMSE), on that session's own test split
DATA_ROOT="${DATA_ROOT:-/users/$USER/scratch/$USER/data}"
LOGS="$(cd "$(dirname "$0")" && pwd)/snn_logs"
for job in "$@"; do
    echo "######## job $job"
    for f in $(ls "$LOGS"/SNN_PER_SESSION_${job}_*.out 2>/dev/null | sort -t_ -k5 -n); do
        t=${f##*_}; t=${t%.out}
        awk -v t="$t" '
            /^Per-session training/ { split($3, p, "/"); s=p[3] }
            /^Epoch [0-9]+\// { split($2,e,"/"); ep=e[1]; tot=e[2]; for (i=1;i<=NF;i++) if ($i=="test") test=$(i+1) }
            /Saved best model/ { best=$NF }
            /Training finished|Early stopping/ { done="DONE" }
            /Error|Traceback|CANCELLED|oom-kill/ { err=$0 }
            END { if (!ep) { printf "task %-3s %-18s no epoch finished yet %s\n", t, s, err; exit }
                  printf "task %-3s %-18s epoch %2d/%-3d test %-9s best %-9s %s %s\n", t, s, ep, tot, test, best, done, err }' "$f"
    done
done
echo "######## best test loss per session (per-axis RMSE, own test split): current Speck SNN vs speck_delay"
SPECK_MODEL="${SPECK_MODEL:-speck_delay}"
echo "(model: $SPECK_MODEL)"
python3 - "$DATA_ROOT" "per_session_${SPECK_MODEL#speck_}_binarized" <<'PY'
import glob, os, sys
import numpy as np
import torch
root, new_dir = sys.argv[1], sys.argv[2]
for subj in ("indy", "loco"):
    base = f"{root}/snn_checkpoints/bmi/{subj}"
    rows = []
    for path in sorted(glob.glob(f"{base}/{new_dir}/*/best_model_weights.pth")):
        s = path.split("/")[-2]
        new = torch.load(path, map_location="cpu", weights_only=False)
        old_path = f"{base}/per_session_binarized/{s}/best_model_weights.pth"
        old = torch.load(old_path, map_location="cpu", weights_only=False)["best_loss"] if os.path.isfile(old_path) else np.nan
        done = os.path.isfile(os.path.join(os.path.dirname(path), "final_model_bmi_iaf_hard.pth"))
        rows.append((s, old, new["best_loss"], new["epoch"] + 1, done))
    if not rows:
        print(f"{subj}: no speck_delay checkpoints yet")
        continue
    print(f"{subj}:")
    for s, o, n, ep, done in rows:
        if done:
            print(f"  {s:20s} current {o:6.2f}  delay {n:6.2f}  ({n - o:+.2f})  best epoch {ep:2d}  done")
        else:
            print(f"  {s:20s} current {o:6.2f}  delay {n:6.2f}  (still training: best so far, at epoch {ep})")
    fin = [r for r in rows if r[4] and not np.isnan(r[1])]
    if fin:
        d = np.array([r[2] - r[1] for r in fin])
        print(f"  finished {len(fin)}: mean current {np.mean([r[1] for r in fin]):.2f}, delay {np.mean([r[2] for r in fin]):.2f}, "
              f"difference {d.mean():+.2f}; delay lower on {(d < 0).sum()}/{len(fin)}")
PY
