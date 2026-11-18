#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_FINETUNE
#SBATCH -o snn_logs/SNN_FINETUNE_%A_%a.out
#SBATCH -e snn_logs/SNN_FINETUNE_%A_%a.out
#
# Stage 2 of 2 of the SNN: fine-tune the pretrained network (soft reset),
# starting from its best weights with a fresh optimizer.
#
#   default        one job, fine-tuned on every session -> one model, like the
#                  RNN (the 'snn' checkpoint rnn_step3 evaluates):
#                  ${BASE_DIR}/derived/snns/finetune_soft/
#   PER_SESSION=1  an array, task i = session i of the pretrained model, each
#                  fine-tuned on its own session only:
#                  ${BASE_DIR}/derived/snns/finetune_soft/<session>/
#                  (submit with --array=0-18 for the 19 baseline sessions)
#
# See run_snn_pretrain.sbatch for how to chain the two.
# Re-submitting skips finished runs and resumes interrupted ones.

set -eo pipefail
source ./snn_config.sh
select_stage finetune

PRETRAINED_CHECKPOINT="${PRETRAIN_DIR}/best_model_weights.pth"
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: $PRETRAINED_CHECKPOINT not found; run run_snn_pretrain.sbatch first." >&2
    exit 1
fi
activate_env

SESSION_ARGS=()
CHECKPOINT_DIR="$FINETUNE_DIR"
if [ "${PER_SESSION:-0}" = "1" ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: PER_SESSION=1 needs an array job, one task per session (--array=0-18)" >&2
        exit 1
    fi
    SESSION_ID=$("$PYTHON" -c "import torch, sys; s = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['sessions']; i = int(sys.argv[2]); print(s[i] if i < len(s) else '')" \
                 "$PRETRAINED_CHECKPOINT" "$SLURM_ARRAY_TASK_ID")
    if [ -z "$SESSION_ID" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID has no session; exiting."
        exit 0
    fi
    SESSION_ARGS=(--sessions "$SESSION_ID")
    CHECKPOINT_DIR="${FINETUNE_DIR}/${SESSION_ID}"
fi

if [ -f "${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth exists; delete it to redo."
    exit 0
fi
latest_resume_args "$CHECKPOINT_DIR"
INIT_ARGS=()
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    echo "[fresh] starting from $PRETRAINED_CHECKPOINT"
    INIT_ARGS=(--init-weights-from "$PRETRAINED_CHECKPOINT")
fi

echo "Fine-tuning (soft reset)${SESSION_ID:+ on ${SESSION_ID}} -> ${CHECKPOINT_DIR}"
mkdir -p "$CHECKPOINT_DIR"
"$PYTHON" -m snnDecoder.train \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${SESSION_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
