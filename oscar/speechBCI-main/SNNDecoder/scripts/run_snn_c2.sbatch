#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_C2
#SBATCH -o snn_logs/SNN_C2_%A_%a.out
#SBATCH -e snn_logs/SNN_C2_%A_%a.out
#SBATCH --array=0-1
#
# Side by side: the candidate configuration "C2" with adaptive neurons (0) and
# with plain IAF neurons (1), otherwise identical:
#   widths 1024 -> 512 -> 256 + 256 readout layer, soft reset, multi-spike
#   per-neuron tau_syn log-spaced x/3..x*3 around 1.5 / 3 / 6.5 / 14 bins
#     (the centres configuration C learned), trained
#   readout: one leaky stage with a 1-bin time constant (decay 0.37, trained)
#     instead of C's 2-stage EMA
#   white noise 0.3, offset 0.2, input dropout 0.2, 60 epochs, the sweep's 8 sessions
#   0 C2_alif: ALIF as an adaptive IAF (tau_mem 1000 bins, no input norm),
#              threshold adaptation tau 25 bins (500 ms), adapt_scale 1.8
#   1 C2_iaf:  plain IAF
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad run_snn_c2.sbatch
#   (SESSIONS=all for all 19 baseline sessions, ~2.5x slower; EPOCHS=... to change the length)
# Report:  python report_runs.py snn_logs --pattern 'SNN_C2_*.out' --ckpt-root $BASE_DIR/derived/snns --out c2_report
# Resubmitting skips finished runs and resumes interrupted ones.
set -eo pipefail
source ./snn_config.sh
activate_env

if [ "${SESSIONS:-}" = "all" ]; then
    SESSION_ARGS=()       # train.py default: every session in the data folder's baseline list
else
    read -r -a SESSION_LIST <<< "${SESSIONS:-t12.2022.05.05 t12.2022.05.24 t12.2022.06.07 t12.2022.06.16 t12.2022.06.28 t12.2022.07.14 t12.2022.07.27 t12.2022.08.11}"
    SESSION_ARGS=(--sessions "${SESSION_LIST[@]}")
fi
EPOCHS=${EPOCHS:-60}

C2="--hidden-dims 1024 512 256 --readout-width 256 --reset-type soft --tau-syn 1.5 3 6.5 14 --tau-syn-spread 3 --temporal-decay-stages 1 --temporal-decay-init 0.37 --white-noise-sd 0.3"
CONFIGS=(
    "C2_alif|$C2 --neuron-type alif --tau-mem 1000 --norm-input false --tau-adapt 25"
    "C2_iaf|$C2 --neuron-type iaf"
)

i=${SLURM_ARRAY_TASK_ID:-0}
[ "$i" -lt "${#CONFIGS[@]}" ] || { echo "no config $i (have ${#CONFIGS[@]})"; exit 0; }
NAME=${CONFIGS[$i]%%|*}
FLAGS=${CONFIGS[$i]#*|}
OUT="${SNN_CKPT_ROOT}/c2/$(printf '%02d' "$i")_${NAME}"
mkdir -p "$OUT"
printf '%s\n' "$FLAGS" > "$OUT/flags.txt"
echo "[$i] ${NAME}: ${FLAGS:-(reference)} | ${EPOCHS} epochs | sessions: ${SESSION_LIST[*]:-all}"

if ls "$OUT"/final_model_*.pth >/dev/null 2>&1; then
    echo "[skip] finished: $(ls "$OUT"/final_model_*.pth)"
    exit 0
fi
RESUME_ARGS=()
latest=$(ls "$OUT"/checkpoint_*_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
[ -n "$latest" ] && { echo "[resume] $latest"; RESUME_ARGS=(--resume "$latest"); }

# shellcheck disable=SC2086  # FLAGS holds several flags
"$PYTHON" -m snnDecoder.train --config "$CONFIG_YAML" --data-dir "$SNN_DATA_DIR" "${SESSION_ARGS[@]}" \
    --checkpoint-dir "$OUT" --epochs "$EPOCHS" --checkpoint-interval 5 \
    --selection-metric loss --disable-early-stopping $FLAGS "${RESUME_ARGS[@]}"
