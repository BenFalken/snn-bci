#!/bin/bash
# Progress of the final comparison: Model B (LSNN, the final model: 2 x 1,024 recurrent, 90% adaptive) and Model A
# (DelayNet: 2 x 1,024 IAF, 4 delayed input synapses), both reading both layers, over three seeds, with the variants
# set aside; the build-up ladder; the references; the queue.
# Install as the progress command:  cp progress_report.sh ~/new_progress.sh   (then: bash ~/new_progress.sh)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
cd "${SCRIPTS_DIR:-$SNNBCI_ROOT/speech/SNNDecoder/scripts}" || exit 1

log_of() {   # newest log of a run name (group/name in the run-file logs); nothing when it has none
  local files
  files=$(grep -l "^\[[0-9]*\] [A-Za-z0-9_]*/$1\b" snn_logs/SNN_RUN_*_[0-9]*.out 2>/dev/null)
  [ -n "$files" ] && ls -t $files | head -1
}
per_of() {   # test PER of a finished (96/96) run, else nothing
  local f; f=$(log_of "$1"); [ -z "$f" ] && return
  grep '^Epoch 96/96' "$f" | tail -1 | grep -oE 'test PER [0-9.]+' | cut -d' ' -f3
}
show() {     # label: epoch, test PER, spike rate; "not started" without a log; any error flagged
  local label=$1 name=$2 f last
  f=$(log_of "$name")
  if [ -z "$f" ]; then printf '  %-30s %s\n' "$label" "not started"; return; fi
  last=$(grep '^Epoch' "$f" | tail -1)
  if [ -z "$last" ]; then printf '  %-30s %s\n' "$label" "starting"; else
    printf '  %-30s epoch %-6s  PER %s  spikes %s\n' "$label" \
      "$(echo "$last" | grep -oE 'Epoch [0-9]+/[0-9]+' | cut -d' ' -f2)" \
      "$(echo "$last" | grep -oE 'test PER [0-9.]+' | cut -d' ' -f3)" \
      "$(echo "$last" | grep -oE 'spike rate [0-9.]+' | cut -d' ' -f3)"
  fi
  grep -qE "Traceback|OutOfMemory" "$f" && printf '  %-30s %s\n' "" "!! error in $f"
}
summary() {  # mean and range of the finished seeds' test PER
  local label=$1 pers=() p n; shift
  for n in "$@"; do p=$(per_of "$n"); [ -n "$p" ] && pers+=("$p"); done
  [ ${#pers[@]} -eq 0 ] && { printf '  %-30s %s\n' "$label" "no finished seeds"; return; }
  printf '%s\n' "${pers[@]}" | awk -v l="$label" '{s+=$1; if(NR==1||$1<lo)lo=$1; if(NR==1||$1>hi)hi=$1}
    END{printf "  %-30s mean PER %.4f over %d seed(s), range %.4f-%.4f\n", l, s/NR, NR, lo, hi}'
}
section() { echo; echo "######## $1"; }

section "FINAL MODELS (96 epochs, readout from both layers)"
summary "Model B, LSNN (final)" C_lsnn_nodelay96 Q_lsnn_s43 Q_lsnn_s44
summary "Model A, DelayNet" R_readall96 R_readall_s43 R_readall_s44
echo "  -- set aside: reading the last layer only"
summary "LSNN" B_noro_s42 B_noro_s43 B_noro_s44
summary "DelayNet" R_full96 A_noro_s43 A_noro_s44
echo "  -- set aside: two output smoothing stages on the final LSNN"
show "  seed 42 [vs .2499]" F_lsnn_s42
show "  seed 43 [vs .2446]" F_lsnn_s43
show "  seed 44 [vs .2450]" F_lsnn_s44
summary "LSNN + two output stages" F_lsnn_s42 F_lsnn_s43 F_lsnn_s44

section "LADDER (96 epochs, seed 42, IAF until the fork; the final model's path)"
show "0 original layout" L0_orig
show "1 + width (2 x 1,024)" L1_width
show "2 + recurrence" S2_rec
show "3 + 80 ms outputs" S3_out80
show "4 + readout from both layers" S4_readall
show "5 + trained tau (root)" C_nodelay96
show "fork: + adaptation (LSNN)" C_lsnn_nodelay96
show "fork: + delays (DelayNet)" R_readall96
echo "  -- set aside (last-layer base, T4_tau)"
show "SE-adLIF" E_sead
show "SE-adLIF (slower)" E_sead_long
echo "  -- side check: recurrence without width (narrow 512-256-128)"
show "narrow + recurrence" R1_rec

section "REFERENCES (same input; networks 96 epochs, filters fitted in closed form)"
for ref in "GRU 2 x 1,024:GRU_full96" "LSTM 2 x 1,024:LSTM_full96" "QRNN 2 x 1,024:QRNN_full96"; do
  label=${ref%%:*}; f=$(ls -t snn_logs/${ref##*:}_*.out 2>/dev/null | head -1)
  if [ -z "$f" ]; then printf '  %-30s %s\n' "$label" "no log (${ref##*:})"; continue; fi
  done_per=$(grep -ohE 'best test PER [0-9.]+' "$f" | tail -1 | cut -d' ' -f4)
  last=$(grep '^Epoch' "$f" | tail -1)
  if [ -n "$done_per" ]; then printf '  %-30s PER %s (best, finished)\n' "$label" "$done_per"
  elif [ -n "$last" ]; then printf '  %-30s stopped or running at epoch %s, PER %s  (%s)\n' "$label" \
      "$(echo "$last" | grep -oE 'Epoch [0-9]+/[0-9]+' | cut -d' ' -f2)" \
      "$(echo "$last" | grep -oE 'test PER [0-9.]+' | cut -d' ' -f3)" "$(basename "$f")"
  else printf '  %-30s %s\n' "$label" "no epoch yet ($(basename "$f"))"; fi
done
f=$(ls -t snn_logs/CLASSIC_*.out 2>/dev/null | head -1)          # Wiener and Kalman: valid only if fitted after the
if [ -n "$f" ]; then                                             # 2026-10-22 silence-label fix
  printf '  %-30s %s\n' "Wiener / Kalman filters" "fit $(date -r "$f" '+%Y-%m-%d %H:%M') ($(basename "$f"))"
  for n in wiener_160ms kalman_80ms_day; do
    p=$(grep -ohE "$n: test PER [0-9.]+" "$f" | tail -1 | grep -oE '[0-9.]+$')
    printf '  %-30s PER %s\n' "  $n" "${p:-not in log}"
  done
else printf '  %-30s %s\n' "Wiener / Kalman filters" "no log (CLASSIC)"; fi
printf '  %-30s PER %s\n' "paper RNN (5 x 1,024, window)" "0.188"

section "INFERENCE OUTPUTS (accuracy comparison, efficiency, figures, trial export)"
B=${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}
AE=${SCRIPTS_DIR:-$SNNBCI_ROOT/speech/SNNDecoder/scripts}/../../AnalysisExamples
for out in "accuracy comparison:$B/derived/decoderComparison_final.json" \
           "efficiency report:$B/derived/efficiency/efficiency_final.md" \
           "comparison figures:$B/derived/figures/final_fidelity.png" \
           "trial figures:$AE/trialPlots_final/LM" "viewer export:$B/derived/viewer_final"; do
  p=${out#*:}
  if [ -e "$p" ]; then printf '  %-30s ready  (%s)\n' "${out%%:*}" "$(date -r "$p" '+%m-%d %H:%M')"
  else printf '  %-30s not yet\n' "${out%%:*}"; fi
done
for job in "COMPARE:$AE/logs" "PLOT_TRIALS:$AE/logs" "EFFICIENCY:snn_logs"; do  # last line of each job's newest log
  f=$(ls -t ${job#*:}/${job%%:*}_*.out 2>/dev/null | head -1)
  [ -n "$f" ] && printf '  %-30s %s\n' "${job%%:*} log" "$(grep -v '^\s*$' "$f" | tail -1 | cut -c1-90)"
done

section "QUEUE"
command -v squeue >/dev/null && squeue -u $USER -o "%.14i %.16j %.3t %.10M %R"
true
