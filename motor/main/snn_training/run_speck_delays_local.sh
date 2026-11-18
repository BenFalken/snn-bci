#!/bin/bash
# compare_speck_delays.py over several sessions on a laptop (datasets as built by run_sessions_local.sh), then a
# summary over sessions: $ROOT/lsnn_vs_snn/speck_delays_summary.md
#   ROOT=~/Downloads/spike_bmi_lsnn_data bash run_speck_delays_local.sh indy_20160411_01 indy_20160921_01 ...
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
ROOT="${ROOT:-$SNNBCI_MOTOR_LSNN_DATA}"
DEVICE="${DEVICE:-cpu}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for S in "$@"; do
    SUBJ="${S%%_*}"
    OUT="$ROOT/lsnn_vs_snn/$S"
    mkdir -p "$OUT"
    echo "=== $S"
    MPLBACKEND=Agg python3 "$HERE/compare_speck_delays.py" --data-path "$ROOT/snn_datasets/bmi/$SUBJ/mua_8_group/$S" \
        --out-dir "$OUT" --device "$DEVICE" > "$OUT/run_speck_delays.log" 2>&1
done
python3 - "$ROOT/lsnn_vs_snn" "$@" <<'PY'
import json, os, sys
import numpy as np
root, sessions = sys.argv[1], sys.argv[2:]
res = {s: json.load(open(os.path.join(root, s, "speck_delays.json"))) for s in sessions}
names = list(dict.fromkeys(n for s in sessions for n in res[s]))          # every network run on any session
def cell(s, n):
    r = res[s].get(n)
    return f"{r['rmse']:.1f} / {r['speck'].get('quantized_rmse', float('nan')):.1f}" if r else "-"
def mean(n, f):
    v = [f(res[s][n]) for s in sessions if n in res[s]]
    return np.mean(v), len(v)
lines = ["# Speck SNN with input delays, over sessions (each trained on its session alone, seed 42)", "",
         "Velocity error (RMS of the velocity vector), PyTorch / quantized (8-bit, emulated on the host). "
         "Means over the sessions each network ran on (n).", "",
         "| network | " + " | ".join(sessions) + " | mean (n) | R² mean |", "|---|" + "---|" * (len(sessions) + 2)]
for n in names:
    m, k = mean(n, lambda r: r['rmse'])
    q, _ = mean(n, lambda r: r['speck'].get('quantized_rmse', np.nan))
    r2, _ = mean(n, lambda r: r['r2_mean'])
    lines.append(f"| {n} | " + " | ".join(cell(s, n) for s in sessions) + f" | {m:.1f} / {q:.1f} ({k}) | {r2:.3f} |")
open(os.path.join(root, "speck_delays_summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
