#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_DIAG
#SBATCH -o snn_logs/SNN_DIAG_%A_%a.out
#SBATCH -e snn_logs/SNN_DIAG_%A_%a.out
#SBATCH --array=0-5
#
# Short runs on 4 sessions that separate the possible reasons the pooled SNN
# stays at ~97% PER, one array task each. Every run trains a fixed number of
# epochs (no early stopping) and selects on loss.
#
#   0 snn_current    the pretraining configuration as is
#   1 snn_no_noise   no white noise / offset augmentation
#   2 snn_smooth2    the RNN's input smoothing (sd 2 bins; non-causal, ~120 ms lookahead)
#   3 snn_long_tau   longer synaptic time constants (tau_syn 2-16 bins, 40-320 ms) and EMA decay 0.8
#   4 snn_long_tau_low_noise   3 with white noise 0.3
#   5 gru_control    a causal 2x512 GRU through the same pipeline, augmentation as 0
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_snn_diagnostics.sbatch
#   grep -h "test PER" snn_logs/SNN_DIAG_<jobid>_*.out | less     # or: tail -n 3 snn_logs/SNN_DIAG_<jobid>_*.out
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env

SESSIONS=(t12.2022.06.02 t12.2022.06.07 t12.2022.06.14 t12.2022.06.16)
EPOCHS=${EPOCHS:-25}
NAMES=(snn_current snn_no_noise snn_smooth2 snn_long_tau snn_long_tau_low_noise gru_control)
EXTRA=(
    ""
    "--white-noise-sd 0 --constant-offset-sd 0"
    "--smooth-sd 2"
    "--tau-syn 2 4 8 16 --temporal-decay-init 0.8"
    "--tau-syn 2 4 8 16 --temporal-decay-init 0.8 --white-noise-sd 0.3"
    ""
)
i=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$i]}
OUT="${SNN_CKPT_ROOT}/diagnostics/${NAME}"
echo "[$i] ${NAME}: ${EXTRA[$i]:-(no changes)} | sessions ${SESSIONS[*]} | ${EPOCHS} epochs"

if [ "$NAME" = "gru_control" ]; then
    "$PYTHON" gru_control.py --data-dir "$SNN_DATA_DIR" --sessions "${SESSIONS[@]}" --epochs "$EPOCHS"
else
    mkdir -p "$OUT"
    printf '%s\n' "${EXTRA[$i]}" > "$OUT/flags.txt"
    # resubmitting skips a finished run and resumes an interrupted one from its newest checkpoint
    if ls "$OUT"/final_model_*.pth >/dev/null 2>&1; then
        echo "[skip] finished: $(ls "$OUT"/final_model_*.pth)"
        exit 0
    fi
    RESUME_ARGS=()
    latest=$(ls "$OUT"/checkpoint_*_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$latest" ] && { echo "[resume] $latest"; RESUME_ARGS=(--resume "$latest"); }
    # shellcheck disable=SC2086  # EXTRA holds several flags
    "$PYTHON" -m snnDecoder.train --config "$CONFIG_YAML" --data-dir "$SNN_DATA_DIR" \
        --sessions "${SESSIONS[@]}" --reset-type hard --checkpoint-dir "$OUT" \
        --epochs "$EPOCHS" --checkpoint-interval 5 --selection-metric loss --disable-early-stopping \
        ${EXTRA[$i]} "${RESUME_ARGS[@]}"
fi
