#!/bin/bash
# Progress of the newest run_snn_configs.sbatch job (or JOB=<id>): for each run its latest epoch, test PER,
# the mean of its last 5 epochs, the baseline's mean over the same 5 epochs, spike rate, and any trained
# time constants / thresholds; errors if a run crashed.
#   bash sweep_progress.sh                 # from SNNDecoder/scripts
#   JOB=7480000 bash sweep_progress.sh
#   BASE_RUN=R2x1280_alif bash sweep_progress.sh    # the baseline run's name (default R2x1280_alif)
cd "$(dirname "$0")"
JOB=${JOB:-$(ls -t snn_logs/SNN_RUN_*_task*.out 2>/dev/null | head -1 | sed 's/.*SNN_RUN_\([0-9]*\)_task.*/\1/')}
BASE_RUN=${BASE_RUN:-R2x1280_alif}
BASE=$(grep -l "/${BASE_RUN}:" snn_logs/SNN_RUN_*_[0-9]*.out snn_logs/archive_*/SNN_RUN_*_[0-9]*.out 2>/dev/null | head -1)
mean5() { grep -o "test PER [0-9.]*" | awk '{s+=$3; n++} END {if (n) printf "%.4f", s/n; else printf "-"}'; }
echo "job $JOB | baseline: $BASE_RUN (${BASE:-not found})"
printf "%-16s %6s %8s %10s %12s %7s\n" run epoch PER last5 "base same5" spikes
for f in snn_logs/SNN_RUN_${JOB}_[0-9]*.out; do
  [ -f "$f" ] || continue
  name=$(grep -m1 "^\[" "$f" | sed 's/^\[[0-9]*\] [^/]*\/\([^:]*\):.*/\1/')
  last=$(grep "^Epoch" "$f" | tail -1)
  if [ -z "$last" ]; then printf "%-16s (no epoch yet)\n" "$name"; grep -m1 "Traceback\|out of memory\|Error" "$f"; continue; fi
  ep=$(echo "$last" | sed 's/^Epoch \([0-9]*\)\/.*/\1/')
  per=$(echo "$last" | grep -o "test PER [0-9.]*" | cut -d' ' -f3)
  rate=$(echo "$last" | grep -o "spike rate [0-9.a-z]*" | cut -d' ' -f3)
  mine=$(grep "^Epoch" "$f" | tail -5 | mean5)
  base=$([ -n "$BASE" ] && grep "^Epoch" "$BASE" | awk -v e="$ep" -F'[ /]' '$2>e-5 && $2<=e' | mean5)
  printf "%-16s %6s %8s %10s %12s %7s\n" "$name" "$ep" "$per" "$mine" "${base:--}" "$rate"
  grep "trained time constants\|learned thresholds" "$f" | tail -1 | sed 's/^ */                 /'
  grep -m1 "Traceback\|out of memory\|nan" "$f"
done
