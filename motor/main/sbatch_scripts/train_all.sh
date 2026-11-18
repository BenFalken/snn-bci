#!/bin/bash
# Train every decoder of the final pipeline for several subjects: one command
# that submits all the Slurm jobs, chained by dependencies. No inference.
#
#   bash train_all.sh                      # bmi indy, bmi loco, hkm jenkins
#   DRY_RUN=1 bash train_all.sh            # print the sbatch commands, submit nothing
#   SUBJECTS="hkm:jenkins" bash train_all.sh
#   SUBJECTS="hkm:nitschke" bash train_all.sh   # the fourth subject, on its own
#
# Per subject (sessions = {DATA_ROOT}/raw/<exp>/<subject>/*.mat or *.nwb; for bmi
# also sessions/bmi_<subject>.txt, so sessions purged from scratch are downloaded
# again; a subject with no sessions is skipped with a warning):
#   1. datasets      bmi: run_bmi_subject_pipeline_array.sbatch STAGE=datasets
#                    hkm: nwb_conversion/run_hkm_nwb_pipeline_array.sbatch (skips
#                         sessions already built with trial IDs)
#   2. KF, WF, LSTM, QRNN, after 1: run_{bmi,hkm}_subject_pipeline_array.sbatch
#      (one fit each on the training split; no CV folds, no durations)
#   3. velocity scalers, once, after every subject's datasets:
#      compute_velocity_scalers.py -> ../snn_training/velocity_scalers.json (merged)
#   4. PyTorch SNN (MODEL=lsnn by default: the recurrent LSNN), after 3: run_snn_pooled_pretrain.sbatch, then per session
#      run_snn_pooled_finetune_array.sbatch (aftercorr: session i starts when its
#      pretraining succeeds); LOSO=1 by default
#   5. Speck SNN, after 3: run_snn_per_session_array.sbatch
# Networks and epochs: snn_config.sh. Steps 2 and 4-5 run in parallel.
#
# Re-running is safe and is how to resume: every job skips finished work and
# resumes interrupted training. Cancel this script's pending jobs first
# (squeue -u $USER), or the old and new submissions would run side by side.
# A dependency on a job that failed stays pending forever: cancel it and re-run.
#
# Settings (environment): DATA_ROOT, SUBJECTS ("exp:subject ..."), LOSO (0/1),
# MODEL (step 4's network: lsnn, the default, or pooled, the earlier SNN),
# AFTER (job ids, colon-separated: nothing starts until they have ended),
# HKM_TIME (Slurm time limit for hkm SNN jobs, default 48:00:00; an hkm epoch
# takes about an hour per session), DRY_RUN=1.

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

export DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
SUBJECTS="${SUBJECTS:-bmi:indy bmi:loco hkm:jenkins}"
export LOSO="${LOSO:-1}"
export MODEL="${MODEL:-lsnn}"
AFTER_ARGS=()
[ -n "${AFTER:-}" ] && AFTER_ARGS=(--dependency=afterany:"$AFTER")
HKM_TIME="${HKM_TIME:-48:00:00}"

# submit [sbatch args...] -> prints the job id (DRY_RUN: prints the command to stderr)
submit() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  sbatch $*" >&2
        echo "DRY${RANDOM}"
    else
        sbatch --parsable "$@"
    fi
}

mkdir -p logs snn_logs bmi_logs hkm_logs ../nwb_conversion/logs
echo "DATA_ROOT=$DATA_ROOT  subjects: $SUBJECTS  LOSO=$LOSO  MODEL=$MODEL${AFTER:+  after: $AFTER}$([ "${DRY_RUN:-0}" = "1" ] && echo "  (dry run)")"

# --- 1-2: datasets, then the classical and DL decoders, per subject ---
source ./bmi_sessions.sh
declare -A N_SESSIONS
DATASET_JOBS=()
ACTIVE=()
for entry in $SUBJECTS; do
    exp="${entry%%:*}"; subject="${entry#*:}"
    ext=$([ "$exp" = "hkm" ] && echo nwb || echo mat)
    raw_dir="${DATA_ROOT}/raw/${exp}/${subject}"
    if [ "$exp" = "bmi" ]; then
        # raw .mat files plus sessions/bmi_<subject>.txt: a purged raw file is downloaded again
        n=$(bmi_sessions "$subject" | wc -l)
    else
        n=$(find "$raw_dir" -maxdepth 1 -name "*.${ext}" 2>/dev/null | wc -l)
    fi
    if [ "$n" -eq 0 ]; then
        echo "WARNING: ${exp}/${subject}: no sessions (no *.${ext} in ${raw_dir}$([ "$exp" = "bmi" ] && echo " and no sessions/bmi_${subject}.txt")); skipped" >&2
        continue
    fi
    N_SESSIONS[$entry]=$n
    ACTIVE+=("$entry")
    array="--array=0-$((n - 1))"
    echo "${exp}/${subject}: ${n} sessions"
    if [ "$exp" = "hkm" ]; then
        datasets=$(cd ../nwb_conversion && submit "${AFTER_ARGS[@]}" --export=ALL,SUBJECT="$subject" "$array" \
                   run_hkm_nwb_pipeline_array.sbatch)
        decoders=$(submit --dependency=afterany:"$datasets" --export=ALL,SUBJECT="$subject" "$array" \
                   run_hkm_subject_pipeline_array.sbatch)
    else
        datasets=$(submit "${AFTER_ARGS[@]}" --export=ALL,SUBJECT="$subject",STAGE=datasets "$array" \
                   run_bmi_subject_pipeline_array.sbatch)
        decoders=$(submit --dependency=afterany:"$datasets" --export=ALL,SUBJECT="$subject" "$array" \
                   run_bmi_subject_pipeline_array.sbatch)
    fi
    echo "  datasets: job $datasets   KF/WF/LSTM/QRNN: job $decoders"
    DATASET_JOBS+=("$datasets")
done

if [ ${#ACTIVE[@]} -eq 0 ]; then
    echo "ERROR: no subject has sessions; nothing submitted" >&2
    exit 1
fi
SUBJECTS="${ACTIVE[*]}"

# --- 3: velocity scalers, once every dataset job has ended; the job fails when a
# subject has no SNN training data, so the SNN jobs (afterok) never start on an
# empty pool ---
experiments=$(for e in $SUBJECTS; do echo "${e%%:*}"; done | sort -u | tr '\n' ' ')
subjects=$(for e in $SUBJECTS; do echo "${e#*:}"; done | tr '\n' ' ')
scalers=$(submit --dependency=afterany:"$(IFS=:; echo "${DATASET_JOBS[*]}")" -J VELOCITY_SCALERS \
          --time=1:00:00 --mem=16G --cpus-per-task=1 -o logs/VELOCITY_SCALERS_%j.out \
          --wrap="source \$HOME/ann-env/bin/activate && python3 ../preprocessing_training/compute_velocity_scalers.py \
--snn-datasets-root ${DATA_ROOT}/snn_datasets --experiments ${experiments} --subjects ${subjects} \
--output ../snn_training/velocity_scalers.json \
&& for e in ${SUBJECTS}; do d=${DATA_ROOT}/snn_datasets/\${e%%:*}/\${e#*:}; \
if [ -z \"\$(find \$d -path '*/train/*.pkl' -print -quit 2>/dev/null)\" ]; then \
echo \"ERROR: no SNN training data under \$d (dataset jobs failed?): stopping before the SNNs\" >&2; exit 1; fi; done")
echo "velocity scalers: job $scalers"

# --- 4-5: the two SNNs, per subject ---
for entry in $SUBJECTS; do
    exp="${entry%%:*}"; subject="${entry#*:}"
    array="--array=0-$((N_SESSIONS[$entry] - 1))"
    env_args="--export=ALL,EXPERIMENT=${exp},SUBJECT=${subject},LOSO=${LOSO},MODEL=${MODEL}"
    time_args=()
    [ "$exp" = "hkm" ] && time_args=(--time="$HKM_TIME")
    if [ "$LOSO" = "1" ]; then
        pretrain=$(submit --dependency=afterok:"$scalers" "$env_args" "$array" "${time_args[@]}" \
                   run_snn_pooled_pretrain.sbatch)
        finetune_dep="aftercorr:${pretrain}"
    else
        pretrain=$(submit --dependency=afterok:"$scalers" "$env_args" "${time_args[@]}" \
                   run_snn_pooled_pretrain.sbatch)
        finetune_dep="afterok:${pretrain}"
    fi
    finetune=$(submit --dependency="$finetune_dep" "$env_args" "$array" "${time_args[@]}" \
               run_snn_pooled_finetune_array.sbatch)
    per_session=$(submit --dependency=afterok:"$scalers" "$env_args" "$array" "${time_args[@]}" \
                  run_snn_per_session_array.sbatch)
    echo "${exp}/${subject}: SNN pretrain job $pretrain, fine-tune job $finetune, per-session job $per_session"
done
echo "Submitted. Follow with: squeue -u \$USER"
