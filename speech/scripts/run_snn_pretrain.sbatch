#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_PRETRAIN
#SBATCH -o snn_logs/SNN_PRETRAIN_%j.out
#SBATCH -e snn_logs/SNN_PRETRAIN_%j.out
#
# Stage 1 of 2 of the SNN: pretrain the pooled network (snn_config.sh) on
# every training session, hard reset. run_snn_finetune.sbatch then
# fine-tunes it with soft reset.
#
# Submit (from this directory), the fine-tune starting when this succeeds:
#   mkdir -p snn_logs
#   PRE=$(sbatch --parsable --export=ALL,BASE_DIR=/path/to/speechPaperRelease_08_20 run_snn_pretrain.sbatch)
#   sbatch --dependency=afterok:$PRE --export=ALL,BASE_DIR=/path/to/speechPaperRelease_08_20 run_snn_finetune.sbatch
#
# Checkpoints: ${BASE_DIR}/derived/snns/pretrain_hard/
# Re-submitting skips a finished run and resumes an interrupted one.

set -eo pipefail
source ./snn_config.sh
select_stage pretrain

if [ -f "${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth exists; delete it to retrain."
    exit 0
fi
latest_resume_args "$PRETRAIN_DIR"
activate_env

echo "Pretraining (hard reset) -> ${PRETRAIN_DIR}"
mkdir -p "$PRETRAIN_DIR"
"$PYTHON" -m snnDecoder.train \
    --checkpoint-dir "$PRETRAIN_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
