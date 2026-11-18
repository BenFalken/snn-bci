#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -J SNN_RUN
#SBATCH -o snn_logs/SNN_RUN_%A_task%a.out
#SBATCH -e snn_logs/SNN_RUN_%A_task%a.out
#
# Trains the SNN configurations listed in a run file. Each array task takes one
# GPU and runs PACK of the listed runs on it at the same time (default 1): an
# SNN run keeps a GPU mostly idle (sinabs steps through time in Python), so
# several share one well, and with the per-user limit of 2 GPUs, PACK=5 runs
# 10 configurations at once. On CPU the runs are ~17x slower per epoch.
#
# Run file (configs/<group>.txt): one run per line, "name|train.py flags";
# blank lines and lines starting with # are skipped.
#
# Submit from this directory:
#   mkdir -p snn_logs
#   RUNS=configs/c2_factors.txt; PACK=5
#   N=$(grep -cv '^[[:space:]]*\(#\|$\)' $RUNS)
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH,RUNS=$RUNS,PACK=$PACK \
#          --array=0-$(( (N + PACK - 1) / PACK - 1 )) run_snn_configs.sbatch
# Task t runs lines t*PACK .. t*PACK+PACK-1. Each run logs to snn_logs/SNN_RUN_<job>_<line>.out.
# Options: SESSIONS=all (all 19 baseline sessions) or a space-separated list
#   (default: the sweep's 8 sessions); EPOCHS (default 60).
#
# Checkpoints: $BASE_DIR/derived/snns/<group>/<NN>_<name>/   (group = run file name)
# Report:      python report_runs.py snn_logs --pattern 'SNN_RUN_*_[0-9]*.out' --ckpt-root $BASE_DIR/derived/snns --out runs_report
# Resubmitting skips finished runs and resumes interrupted ones.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
[ -n "${RUNS:-}" ] && [ -f "$RUNS" ] || { echo "ERROR: set RUNS to a run file, e.g. --export=ALL,...,RUNS=configs/c2.txt" >&2; exit 1; }
source ./snn_config.sh
activate_env

if [ "${SESSIONS:-}" = "all" ]; then
    SESSION_ARGS=()
    SESSION_LIST=(all)
else
    read -r -a SESSION_LIST <<< "${SESSIONS:-t12.2022.05.05 t12.2022.05.24 t12.2022.06.07 t12.2022.06.16 t12.2022.06.28 t12.2022.07.14 t12.2022.07.27 t12.2022.08.11}"
    SESSION_ARGS=(--sessions "${SESSION_LIST[@]}")
fi
EPOCHS=${EPOCHS:-60}

LINES=()
while IFS= read -r line; do LINES+=("$line"); done < <(grep -v '^[[:space:]]*\(#\|$\)' "$RUNS")
# run_line i: train the i-th run of the run file
run_line() {
    local i=$1 NAME FLAGS OUT latest
    NAME=${LINES[$i]%%|*}
    NAME=${NAME//[[:space:]]/}
    FLAGS=${LINES[$i]#*|}
    FLAGS=${FLAGS//\$USER/$USER}   # run lists write the Oscar home as $USER
    OUT="${SNN_CKPT_ROOT}/${GROUP}/$(printf '%02d' "$i")_${NAME}"
    mkdir -p "$OUT"
    printf '%s\n' "$FLAGS" > "$OUT/flags.txt"
    echo "[$i] ${GROUP}/${NAME}: ${FLAGS} | ${EPOCHS} epochs | sessions: ${SESSION_LIST[*]} | $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs, packed ${PACK}"
    if ls "$OUT"/final_model_*.pth >/dev/null 2>&1; then
        echo "[skip] finished: $(ls "$OUT"/final_model_*.pth)"
        return 0
    fi
    local RESUME_ARGS=()
    latest=$(ls "$OUT"/checkpoint_*_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$latest" ] && { echo "[resume] $latest"; RESUME_ARGS=(--resume "$latest"); }
    # shellcheck disable=SC2086  # FLAGS holds several flags
    "$PYTHON" -m snnDecoder.train --config "$CONFIG_YAML" --data-dir "$SNN_DATA_DIR" "${SESSION_ARGS[@]}" \
        --checkpoint-dir "$OUT" --epochs "$EPOCHS" --checkpoint-interval 5 \
        --selection-metric loss --disable-early-stopping $FLAGS "${RESUME_ARGS[@]}"
}

GROUP=$(basename "$RUNS" .txt)
PACK=${PACK:-1}
task=${SLURM_ARRAY_TASK_ID:-0}
job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
# split the job's CPUs between the runs sharing it
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$PACK} / PACK > 0 ? ${SLURM_CPUS_PER_TASK:-$PACK} / PACK : 1 ))
pids=()
for (( j = task * PACK; j < task * PACK + PACK && j < ${#LINES[@]}; j++ )); do
    log="snn_logs/SNN_RUN_${job}_${j}.out"
    echo "task $task: run $j -> $log"
    run_line "$j" > "$log" 2>&1 &
    pids+=($!)
done
[ ${#pids[@]} -gt 0 ] || { echo "no runs for task $task in $RUNS (has ${#LINES[@]})"; exit 0; }
# GPU load once the runs are going (in this task's log)
( sleep 900; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null ) &
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
