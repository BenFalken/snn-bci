#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J SNN_ABL
#SBATCH -o snn_logs/SNN_ABL_%A_%a.out
#SBATCH -e snn_logs/SNN_ABL_%A_%a.out
#SBATCH --array=0-13%6
#
# Ablations of the best feedforward SNN so far (run_snn_sweep.sbatch's C_centre:
# tau_syn 2-16 bins, EMA 2 x 0.8, white noise 0.3), one factor at a time, on
# the sweep's 8 sessions, 60 epochs, no early stopping, selected on loss:
#
#   readout EMA   none / 1 stage 0.5 / 1 stage 0.8 / 2 stages 0.5 (C: 2 stages 0.8).
#                 The EMA's mean lag is stages * d/(1-d) bins: C 8 bins (160 ms),
#                 1 x 0.5 one bin; none reads the current bin's spikes only.
#   spiking       ALIF (sinabs adaptive LIF: threshold rises with recent spikes)
#                 as an adaptive IAF (tau_mem 1000, no input norm) with
#                 adaptation tau 25 or 100 bins, a leaky ALIF (tau_mem 10),
#                 single-spike, spike-rate penalty
#   regularizer   no input dropout
#   combined      adaptive IAF + no EMA, with and without dropout
#
#   learned init  C started from the values C learned (tau_syn 1.4/3.0/6.5/14.2, EMA 0.57)
#
# 00_C_reference reruns C_centre with this code for a same-batch reference.
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_snn_ablation.sbatch
# Report:  python report_runs.py snn_logs --pattern 'SNN_ABL_*.out' --ckpt-root $BASE_DIR/derived/snns --out ablation_report
# Options as run_snn_sweep.sbatch (EPOCHS, SESSIONS, --array). Resubmitting skips finished runs and
# resumes interrupted ones.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
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

C="--tau-syn 2 4 8 16 --temporal-decay-init 0.8 --white-noise-sd 0.3"
AIAF="--neuron-type alif --tau-mem 1000 --norm-input false"
CONFIGS=(
    "C_reference|$C"
    # --- readout EMA
    "E_no_ema|$C --temporal-decay-stages 0"
    "E_1stage_0.5|$C --temporal-decay-stages 1 --temporal-decay-init 0.5"
    "E_1stage_0.8|$C --temporal-decay-stages 1 --temporal-decay-init 0.8"
    "E_2stage_0.5|$C --temporal-decay-init 0.5"
    # --- spiking / rate control
    "R_adaptive_iaf_25|$C $AIAF --tau-adapt 25"
    "R_adaptive_iaf_100|$C $AIAF --tau-adapt 100"
    "R_alif_leaky10|$C --neuron-type alif --tau-mem 10 --norm-input false --tau-adapt 25"
    "R_single_spike|$C --spike-fn single"
    "R_rate_reg|$C --spike-rate-weight 10 --spike-rate-target 0.1"
    # --- regularization
    "D_no_dropout|$C --input-dropout 0"
    # --- combined
    "X_adaptive_iaf_no_ema|$C $AIAF --tau-adapt 25 --temporal-decay-stages 0"
    "X_adaptive_iaf_no_ema_no_dropout|$C $AIAF --tau-adapt 25 --temporal-decay-stages 0 --input-dropout 0"
    # --- C started from the time constants C itself learned in 40 epochs (sweep 10_C_centre)
    "L_learned_init|--tau-syn 1.4 3.0 6.5 14.2 --temporal-decay-init 0.57 --white-noise-sd 0.3"
)

i=${SLURM_ARRAY_TASK_ID:-0}
[ "$i" -lt "${#CONFIGS[@]}" ] || { echo "no config $i (have ${#CONFIGS[@]})"; exit 0; }
NAME=${CONFIGS[$i]%%|*}
FLAGS=${CONFIGS[$i]#*|}
FLAGS=${FLAGS//\$USER/$USER}   # run lists write the Oscar home as $USER
OUT="${SNN_CKPT_ROOT}/ablation/$(printf '%02d' "$i")_${NAME}"
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
