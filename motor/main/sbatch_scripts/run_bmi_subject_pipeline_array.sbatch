#!/bin/bash
#SBATCH --job-name=bmi_pipeline
#SBATCH --output=bmi_logs/%x_%A_%a.out
#SBATCH --error=bmi_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# bmi sessions, one array task per session (the raw .mat files under
# {DATA_ROOT}/raw/bmi/<subject> plus sessions/bmi_<subject>.txt, sorted; see
# bmi_sessions.sh): single_subject_pipeline.py downloads the session if its raw
# file is missing, then processes it,
# builds its ANN and SNN datasets and trains KF, WF, LSTM and QRNN on them.
#
#   SUBJECT=<name>   only that subject's sessions (default: every subject)
#   STAGE=datasets   build the datasets only (SNN training can then start while
#                    the decoders train in a second run)
#   OVERWRITE=1      recompute every stage
#
# Every stage skips outputs that already exist, so re-submitting resumes.
#
# Submit from this directory:
#   mkdir -p bmi_logs
#   N=$(source bmi_sessions.sh; DATA_ROOT=$DATA_ROOT bmi_sessions indy | wc -l)
#   sbatch --export=ALL,SUBJECT=indy --array=0-$((N-1)) run_bmi_subject_pipeline_array.sbatch

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail

deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# Cap each task's threads to its allocation (TF/PyTorch read these at import).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
export BMI_DATA_ROOT="$DATA_ROOT"
RAW_DATA_DIR="${DATA_ROOT}/raw/bmi"

source ./bmi_sessions.sh
if [ -n "${SUBJECT:-}" ]; then
    SUBJECTS_HERE="$SUBJECT"
else
    SUBJECTS_HERE=$( { { find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true; } | sed 's|.*/||'
                       { ls sessions/bmi_*.txt 2>/dev/null || true; } | sed 's|sessions/bmi_||; s|\.txt$||'; } | sort -u)
fi
mapfile -t STEMS < <(for s in $SUBJECTS_HERE; do bmi_sessions "$s"; done)
if [ "${#STEMS[@]}" -eq 0 ]; then
    echo "ERROR: no sessions: no <subject>/<session>.mat under $RAW_DATA_DIR and no sessions/bmi_<subject>.txt${SUBJECT:+ for SUBJECT=$SUBJECT}" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#STEMS[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#STEMS[@]} found); exiting."
    exit 0
fi
RAW_STEM="${STEMS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID / $((${#STEMS[@]} - 1)) -> $RAW_STEM${STAGE:+ (stage: $STAGE)}"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

ARGS=()
[ "${STAGE:-}" = "datasets" ] && ARGS+=(--datasets_only)
[ "${OVERWRITE:-0}" = "1" ] && ARGS+=(--overwrite)
python3 ../preprocessing_training/single_subject_pipeline.py --raw_stem "$RAW_STEM" "${ARGS[@]}"
