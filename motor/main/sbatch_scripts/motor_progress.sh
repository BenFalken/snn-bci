#!/bin/bash
# Progress of train_all.sh for the bmi subjects: per subject, how many sessions have reached each stage (raw file,
# SNN dataset, KF/WF/LSTM/QRNN final models, LSNN pretraining, LSNN fine-tuning, per-session Speck SNN), the queue by job
# and state, held tasks, and the newest error of each job type.
#   bash motor_progress.sh                 # SUBJECTS="indy loco" by default
# Install as a command:  cp motor_progress.sh ~/motor_progress.sh   (then: bash ~/motor_progress.sh)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
cd "${SCRIPTS_DIR:-$SNNBCI_MOTOR/main/sbatch_scripts}" || exit 1
DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
SUBJECTS="${SUBJECTS:-indy loco}"

count() { find "$@" 2>/dev/null | wc -l | tr -d ' '; }   # files matching a find expression

for s in $SUBJECTS; do
  n=$(grep -cv '^[[:space:]]*\(#\|$\)' "sessions/bmi_$s.txt" 2>/dev/null || echo "?")
  C="$DATA_ROOT/snn_checkpoints/bmi/$s"
  echo
  echo "######## bmi/$s: $n sessions"
  printf '  %-34s %s\n' "raw .mat downloaded"        "$(count "$DATA_ROOT/raw/bmi/$s" -maxdepth 1 -name '*.mat')"
  printf '  %-34s %s\n' "SNN datasets built"         "$(count "$DATA_ROOT/snn_datasets/bmi/$s/mua_8_group" -path '*/train/0.pkl')"
  # the decoders' final models (results/model_cache/bmi/<subject>/<session>/mua/); with no CV folds (the pipeline's
  # default) they write no results/.../decoder/*.h5, only these bundles
  M="$DATA_ROOT/results/model_cache/bmi/$s"
  printf '  %-34s %s\n' "KF final models"    "$(count "$M" -name 'kf_config.json')"
  printf '  %-34s %s\n' "WF final models"    "$(count "$M" -name 'wf_config.json')"
  printf '  %-34s %s\n' "LSTM final models"  "$(count "$M" -name 'lstm_config.json')"
  printf '  %-34s %s\n' "QRNN final models"  "$(count "$M" -name 'qrnn_config.json')"
  printf '  %-34s %s\n' "LSNN pretrained (LOSO)"     "$(count "$C/lsnn_loso_pretrained_binarized" -name 'final_model_*.pth')"
  printf '  %-34s %s\n' "LSNN fine-tuned"            "$(count "$C/lsnn_loso_finetuned_binarized" -name 'final_model_*.pth')"
  printf '  %-34s %s\n' "per-session Speck SNN"      "$(count "$C/per_session_binarized" -name 'final_model_*.pth')"
done

echo
echo "######## QUEUE (tasks by job and state)"
squeue -u "$USER" -h -r -o "%j %T %r" | awk '{st=$2; if ($3 ~ /held|Held/) st="HELD"; k=$1" "st; c[k]++}
  END {for (k in c) printf "  %-26s %-10s %s\n", substr(k,1,index(k," ")-1), substr(k,index(k," ")+1), c[k]}' | sort
held=$(squeue -u "$USER" -h -r -o "%i %r" | grep -i held | awk '{print $1}')
[ -n "$held" ] && echo "  -> held: $(echo $held | tr ' ' ',')   release with  scontrol release <jobid>"

echo
# only logs of the current run: job ids at least the oldest one still queued (SINCE_JOB overrides; with an empty
# queue, every log); the job id is the first number of 5+ digits in the log's file name (name_<jobid>[_<task>])
since="${SINCE_JOB:-$(squeue -u "$USER" -h -o "%F" 2>/dev/null | sort -n | head -1)}"
echo "######## NEWEST ERROR PER JOB TYPE${since:+ (jobs from $since on)}: last 3 matching lines of the newest log that has one"
for pat in "bmi_logs/bmi_pipeline_*.err" "bmi_logs/bmi_pipeline_*.out" "logs/VELOCITY_SCALERS_*.out" \
           "snn_logs/SNN_POOLED_PRETRAIN_*.out" "snn_logs/SNN_POOLED_FINETUNE_*.out" "snn_logs/SNN_PER_SESSION_*.out"; do
  f=$(ls -t $pat 2>/dev/null | awk -v s="${since:-0}" '{if (match($0, /_[0-9][0-9][0-9][0-9][0-9]+/) &&
        substr($0, RSTART + 1, RLENGTH - 1) + 0 >= s + 0) print}' | head -50 \
      | xargs grep -lE "Traceback|ERROR|Error:|FAILED" 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "  $f"; grep -E "Error|ERROR|FAILED" "$f" | tail -3 | sed 's/^/      /'; }
done
true
