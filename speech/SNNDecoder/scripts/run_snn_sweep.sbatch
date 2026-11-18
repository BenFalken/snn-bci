#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_SWEEP
#SBATCH -o snn_logs/SNN_SWEEP_%A_%a.out
#SBATCH -e snn_logs/SNN_SWEEP_%A_%a.out
#SBATCH --array=0-23%6
#
# Overnight sweep of the SNN's configuration, one array task per entry of
# CONFIGS below (at most 6 at a time: %6). Every run trains on the same 8
# sessions spread over the recording period for a fixed number of epochs (no
# early stopping, selected on test loss), so runs are directly comparable.
# Summarize with: python summarize_sweep.py $BASE_DIR/derived/snns/sweep
#
# Design: one factor at a time around two centres.
#   A  the current configuration (configs/speech_pooled.yaml), which stayed at
#      ~97% PER on the full data: noise and time-constant changes from it
#   C  a candidate built from A's two likeliest problems, input noise and
#      short memory: tau_syn 2-16 bins (the velocity decoder's step counts,
#      40-320 ms here), EMA decay 0.8, white noise 0.3. Width, neuron,
#      spiking and optimization changes are made from C, so they are not all
#      measured on a model that does not learn at all.
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad run_snn_sweep.sbatch
# Options: --array=0-23%3 (fewer GPUs at once), --array=6,11 (only some runs),
#   --export=...,EPOCHS=60 or SESSIONS=all (all 19 baseline sessions; ~2.5x slower per epoch).
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
EPOCHS=${EPOCHS:-40}

C="--tau-syn 2 4 8 16 --temporal-decay-init 0.8 --white-noise-sd 0.3"
CONFIGS=(
    # --- centre A: input SNR
    "A_reference|"
    "A_noise0.5|--white-noise-sd 0.5"
    "A_noise0.2|--white-noise-sd 0.2"
    "A_noise0_offset0|--white-noise-sd 0 --constant-offset-sd 0"
    "A_smooth2_noncausal|--smooth-sd 2"
    # --- centre A: memory (synaptic time constants, in 20 ms bins, and the readout EMA)
    "A_tau1-8|--tau-syn 1 2 4 8 --temporal-decay-init 0.6"
    "A_tau2-16|--tau-syn 2 4 8 16 --temporal-decay-init 0.8"
    "A_tau4-32|--tau-syn 4 8 16 32 --temporal-decay-init 0.9"
    "A_tau8_flat|--tau-syn 8 8 8 8 --temporal-decay-init 0.8"
    "A_tau16-2_reversed|--tau-syn 16 8 4 2 --temporal-decay-init 0.8"
    # --- combinations
    "C_centre|$C"
    "C_tau4-32|--tau-syn 4 8 16 32 --temporal-decay-init 0.9 --white-noise-sd 0.3"
    "C_ema3_0.9|$C --temporal-decay-stages 3 --temporal-decay-init 0.9"
    # --- centre C: width / depth
    "C_wide_1024|$C --hidden-dims 1024 512 256 --readout-width 256"
    "C_flat_512|$C --hidden-dims 512 512 512 --readout-width 256"
    "C_deep_4layers|--tau-syn 2 4 8 16 16 --temporal-decay-init 0.8 --white-noise-sd 0.3 --hidden-dims 512 512 256 128 --spike-thresholds 1 1 1 1 1"
    # --- centre C: neuron and spiking
    "C_soft_reset|$C --reset-type soft"
    "C_lif_tau5|$C --neuron-type lif --tau-mem 5 --norm-input false"
    "C_single_spike|$C --spike-fn single"
    "C_threshold0.5|$C --spike-thresholds 0.5 0.5 0.5 0.5"
    "C_rate_reg|$C --spike-rate-weight 10 --spike-rate-target 0.1"
    # --- centre C: optimization
    "C_lr3e-4|$C --lr 3e-4"
    "C_lr3e-3|$C --lr 3e-3"
    "C_no_dropout|$C --input-dropout 0"
)

i=${SLURM_ARRAY_TASK_ID:-0}
[ "$i" -lt "${#CONFIGS[@]}" ] || { echo "no config $i (have ${#CONFIGS[@]})"; exit 0; }
NAME=${CONFIGS[$i]%%|*}
FLAGS=${CONFIGS[$i]#*|}
OUT="${SNN_CKPT_ROOT}/sweep/$(printf '%02d' "$i")_${NAME}"
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
    --reset-type hard --checkpoint-dir "$OUT" --epochs "$EPOCHS" --checkpoint-interval 5 \
    --selection-metric loss --disable-early-stopping $FLAGS "${RESUME_ARGS[@]}"
