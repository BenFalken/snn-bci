#!/bin/bash
# Submits run files (3 runs per GPU) and, after all of them, the efficiency report. From this directory on Oscar:
#   bash submit_round.sh configs/rsnn_events.txt configs/rsnn_memory.txt
# Report: $BASE_DIR/derived/efficiency/efficiency_<tag>.md (tag: TAG, default today's date); EFF_RUNS picks its
# model list (default configs/efficiency_runs.txt). AFTER=<job ids, colon-separated> holds the runs until those jobs
# have ended (e.g. a round still training).
set -euo pipefail
B=${BASE_DIR:-/users/$USER/scratch/speechBCI_dryad}
PACK=${PACK:-3}
TAG=${TAG:-$(date +%m%d)}
cd "$(dirname "$0")"
mkdir -p snn_logs
[ $# -gt 0 ] || { echo "usage: bash submit_round.sh configs/<runs>.txt [...]" >&2; exit 1; }
[ -f ../snnDecoder/frontend.py ] || { echo "ERROR: ../snnDecoder/frontend.py missing; rsync SNNDecoder/ first" >&2; exit 1; }
deps=""
hold=${AFTER:+--dependency=afterany:$AFTER}
for runs in "$@"; do
    [ -f "$runs" ] || { echo "ERROR: $runs not found (rsync first?)" >&2; exit 1; }
    n=$(grep -cv '^[[:space:]]*\(#\|$\)' "$runs")
    j=$(sbatch --parsable $hold --cpus-per-task=4 --export=ALL,BASE_DIR=$B,RUNS=$runs,PACK=$PACK,SESSIONS=all,EPOCHS=96 \
        --array=0-$(( (n + PACK - 1) / PACK - 1 )) run_snn_configs.sbatch)
    echo "$runs: $n runs -> job $j (logs snn_logs/SNN_RUN_${j}_<line>.out)"
    deps="$deps:$j"
done
e=$(sbatch --parsable -J EFF_$TAG -t 24:00:00 --dependency=afterany${deps} \
    --export=ALL,BASE_DIR=$B,OUT=$B/derived/efficiency/efficiency_$TAG,QUANT_BITS="8 4",RUNS=${EFF_RUNS:-configs/efficiency_runs.txt} run_efficiency.sbatch)
echo "efficiency report: job $e, after the runs -> $B/derived/efficiency/efficiency_$TAG.md"
squeue -u "$USER" -o "%.14i %.16j %.3t %.10M %R"
