#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J GRU_REF
#SBATCH -o snn_logs/%x_%j.out
#SBATCH -e snn_logs/%x_%j.out
#
# One GRU reference (ann_controls.py) on the 19 sessions with the paper recipe, as the SNN runs. Submit from this
# directory; BASE_DIR must be exported (the job stops at once without it):
#   B=$SNNBCI_OSCAR_SCRATCH
#   sbatch -J GRU_2x896 --export=ALL,BASE_DIR=$B,MODEL=gru_stack_2x896 run_gru_ref.sbatch
#   sbatch -J GRU_TX1   --export=ALL,BASE_DIR=$B,MODEL=gru_1024,TAG=_tx1,DATA=snnDataTx1,EPOCHS=48 run_gru_ref.sbatch
# Options: MODEL (required), TAG (suffix of the run name), DATA (directory under derived/, default snnData),
# EPOCHS (default 96), EXTRA (more ann_controls.py flags). Logs to snn_logs/<job name>_<job id>.out; writes derived/snns/controls/<model><tag>.json and .pt.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
[ -n "${MODEL:-}" ] || { echo "ERROR: set MODEL, e.g. --export=ALL,BASE_DIR=...,MODEL=gru_stack_2x896" >&2; exit 1; }
source ./snn_config.sh
activate_env
S19=(t12.2022.04.28 t12.2022.05.05 t12.2022.05.17 t12.2022.05.19 t12.2022.05.24 t12.2022.05.26
     t12.2022.06.02 t12.2022.06.07 t12.2022.06.14 t12.2022.06.16 t12.2022.06.21 t12.2022.06.28
     t12.2022.07.05 t12.2022.07.14 t12.2022.07.21 t12.2022.07.27 t12.2022.08.02 t12.2022.08.11 t12.2022.08.13)
# shellcheck disable=SC2086  # EXTRA holds several flags
"$PYTHON" ann_controls.py --model "$MODEL" --tag "${TAG:-}" --recipe rnn --epochs "${EPOCHS:-96}" \
    --data-dir "${BASE_DIR}/derived/${DATA:-snnData}" --sessions "${S19[@]}" \
    --out-dir "${SNN_CKPT_ROOT}/controls" --white-noise-sd 1.0 ${EXTRA:-}
