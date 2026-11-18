#!/bin/bash
#SBATCH -p batch
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH -J ANN_CTRL
#SBATCH -o snn_logs/ANN_CTRL_%A_%a.out
#SBATCH -e snn_logs/ANN_CTRL_%A_%a.out
#SBATCH --array=0-4
#
# Non-spiking controls (ann_controls.py), one per array task, on the SNN
# sweep's 8 sessions with SNN configuration C's augmentation, 60 epochs:
#   0 gru_512   1 gru_small   2 ff_twin   3 window32   4 window32_small
# Runs on CPU (batch partition), so it does not take from the 2-GPU limit.
# To use a GPU instead: sbatch -p gpu --gres=gpu:1 --cpus-per-task=4 ...
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad run_ann_controls.sbatch
# Progress:  grep -h "^\[\|^Epoch" snn_logs/ANN_CTRL_<jobid>_*.out | less
# Report:    python report_runs.py snn_logs --pattern 'ANN_CTRL_*.out' --out controls_report
# History:   $BASE_DIR/derived/snns/controls/<model>.json
set -eo pipefail
source ./snn_config.sh
activate_env

MODELS=(gru_512 gru_small ff_twin window32 window32_small)
i=${SLURM_ARRAY_TASK_ID:-0}
"$PYTHON" ann_controls.py --model "${MODELS[$i]}" --index "$i" --epochs "${EPOCHS:-60}" \
    --data-dir "$SNN_DATA_DIR" --out-dir "${SNN_CKPT_ROOT}/controls"
