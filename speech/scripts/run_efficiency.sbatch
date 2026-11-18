#!/bin/bash
#SBATCH -p batch
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J EFFICIENCY
#SBATCH -o snn_logs/EFFICIENCY_%j.out
#SBATCH -e snn_logs/EFFICIENCY_%j.out
#
# Accuracy against energy, memory and latency of the models in a run list (scripts/efficiency.py), on CPU so
# it does not wait for the 2-GPU limit. Submit from this directory:
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_efficiency.sbatch
# Options: RUNS (default configs/efficiency_runs.txt), OUT (default $BASE_DIR/derived/efficiency/efficiency),
# QUANT_BITS (default "8 6 4"; "" = skip quantization, the slowest part).
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env
export BASE_DIR
RUNS=${RUNS:-configs/efficiency_runs.txt}
OUT=${OUT:-${BASE_DIR}/derived/efficiency/efficiency}
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
# shellcheck disable=SC2086  # QUANT_BITS holds several numbers
"$PYTHON" efficiency.py --runs "$RUNS" --data-dir "$SNN_DATA_DIR" --out "$OUT" --device cpu \
    --quant-bits ${QUANT_BITS-8 6 4}
