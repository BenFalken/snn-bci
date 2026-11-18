#!/bin/bash
# Every Speck SNN variant at a glance, from the checkpoints (no job ids needed):
#   bash speck_all_progress.sh            (add -v for one line per session)
# 1. the queue: tasks per job name and state
# 2. per variant and subject: sessions finished / training, and the mean best test loss of the finished ones
#    (sqrt(MSE) of the velocity, physical units, per axis, on each session's own test split), next to the current
#    Speck SNN's on the same sessions, with the number of sessions the variant has the lower loss
# 3. errors: the last error line of each speck variant's newest failed log
DATA_ROOT="${DATA_ROOT:-/users/$USER/scratch/$USER/data}"
LOGS="$(cd "$(dirname "$0")" && pwd)/snn_logs"
echo "######## queue (tasks by job and state)"
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo
echo "######## Speck SNN variants vs the current Speck SNN (per_session_binarized)"
python3 - "$DATA_ROOT" "${1:-}" <<'PY'
import glob, os, sys
import numpy as np
import torch
root, verbose = sys.argv[1], sys.argv[2] == "-v"
VARIANTS = [("speck_delay   0-200 ms, 384-128-256-256", "per_session_delay_binarized"),
            ("speck_delay4  0-200 ms, 384-128-128-64", "per_session_delay4_binarized"),
            ("speck_delay_long  0-600 ms, 384-128-256-256", "per_session_delay_long_binarized"),
            ("speck_delay_xlong 0-1200 ms, 384-128-256-256", "per_session_delay_xlong_binarized"),
            ("speck_delay_random 0 + 3 random 4-200 ms, 384-128-256-256", "per_session_delay_random_binarized"),
            ("speck_delay_shift 0-240 ms, 384-128-256-256", "per_session_delay_shift_binarized"),
            ("speck_plan_prune5 tuned channels x 5 delays, wide", "per_session_plan_prune5_binarized"),
            ("speck_plan_tuned each channel's own delays, wide", "per_session_plan_tuned_binarized"),
            ("speck_delay_learn learned per channel, 0-300 ms, wide", "per_session_delay_learn_binarized"),
            ("speck_delay_learn_shared learned shared, 0-300 ms, wide", "per_session_delay_learn_shared_binarized"),
            ("speck_delay PRETRAINED (LOSO) + fine-tuned", "speckdelay_loso_finetuned_binarized"),
            # the motor ladder (snn_config.sh ladder_*), one change per step from per_session
            ("ladder 1 + synaptic filters, 256-128-64", "per_session_ladder_filters_binarized"),
            ("ladder 1b + soft reset (original network, feedforward)", "per_session_ladder_filters_soft_binarized"),
            ("LSNN ladder 1: LSNN neurons, feedforward 256-128-64, hard", "per_session_lsnn_ladder_ff_binarized"),
            ("LSNN ladder 2 + soft reset", "per_session_lsnn_ladder_soft_binarized"),
            ("LSNN ladder 3 + recurrence", "per_session_lsnn_ladder_rec_binarized"),
            ("LSNN ladder 4 + width, 2 x 256", "per_session_lsnn_ladder_width_binarized"),
            ("LSNN ladder 5 + adaptation (LSNN per session)", "per_session_lsnn_ladder_adapt_binarized"),
            ("ladder chip: 128-256-256, no delays", "per_session_ladder_chip_binarized"),
            ("LSNN (final: LOSO pretrained + fine-tuned)", "lsnn_loso_finetuned_binarized")]

def best(path):
    """(best test loss, best epoch, finished) of one checkpoint directory, or None."""
    f = os.path.join(path, "best_model_weights.pth")
    if not os.path.isfile(f):
        return None
    ck = torch.load(f, map_location="cpu", weights_only=False)
    return ck["best_loss"], ck["epoch"] + 1, bool(glob.glob(os.path.join(path, "final_model_*.pth")))

for subj in ("indy", "loco"):
    base = f"{root}/snn_checkpoints/bmi/{subj}"
    current = {os.path.basename(d): best(d) for d in glob.glob(f"{base}/per_session_binarized/*")}
    print(f"{subj}: current Speck SNN finished on {sum(1 for v in current.values() if v and v[2])} sessions")
    for name, folder in VARIANTS:
        runs = {os.path.basename(d): best(d) for d in sorted(glob.glob(f"{base}/{folder}/*"))}
        runs = {s: r for s, r in runs.items() if r}
        if not runs:
            print(f"  {name:58s} not started")
            continue
        done = [s for s, r in runs.items() if r[2] and current.get(s) and current[s][2]]
        line = f"  {name:58s} finished {sum(r[2] for r in runs.values()):2d}, training {sum(not r[2] for r in runs.values()):2d}"
        if done:
            new = np.array([runs[s][0] for s in done]); old = np.array([current[s][0] for s in done])
            line += (f" | {len(done)} sessions: {new.mean():.2f} vs current {old.mean():.2f} "
                     f"({100 * (new.mean() / old.mean() - 1):+.1f}%), lower on {(new < old).sum()}/{len(done)}")
        print(line)
        if verbose:
            for s, (loss, ep, fin) in runs.items():
                c = current.get(s)
                diff = f"({loss - c[0]:+.2f})" if c and fin else ""
                print(f"      {s:20s} {loss:6.2f} {diff:9s} best epoch {ep:2d} {'done' if fin else 'training'}")
PY
echo
echo "######## errors in speck variant logs (newest per job)"
for job in $(ls "$LOGS"/SNN_PER_SESSION_*.out 2>/dev/null | sed -E 's/.*_([0-9]+)_[0-9]+\.out/\1/' | sort -u); do
    f=$(grep -lE "ERROR|Error|Traceback|CANCELLED|oom" "$LOGS"/SNN_PER_SESSION_${job}_*.out 2>/dev/null | tail -1)
    [ -n "$f" ] && echo "  job $job ($(grep -lE "ERROR|Error|Traceback|CANCELLED|oom" "$LOGS"/SNN_PER_SESSION_${job}_*.out | wc -l) tasks): $(grep -E "ERROR|Error|Traceback|CANCELLED|oom" "$f" | tail -1 | cut -c1-150)"
done
echo "  (none listed = no errors)"
