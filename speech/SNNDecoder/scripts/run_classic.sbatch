#!/bin/bash
#SBATCH -p batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -J CLASSIC
#SBATCH -o snn_logs/%x_%j.out
#SBATCH -e snn_logs/%x_%j.out
#
# Fits the Wiener and Kalman filter benchmarks (classic_decoders.py) in closed form, on CPU (no GPU needed: the
# largest step is a 2,049 x 2,049 normal-equation solve). Frame labels come from force-aligning the 96-epoch GRU
# control on the same input. Submit from this directory:
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_classic.sbatch
# Writes derived/snns/controls/wiener_160ms.json/.pt and kalman_80ms.json/.pt (read like GRU controls).
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
[ -n "${BASE_DIR:-}" ] || { echo "ERROR: export BASE_DIR" >&2; exit 1; }
source ./snn_config.sh
activate_env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$PYTHON" classic_decoders.py --data-dir "${BASE_DIR}/derived/${DATA:-snnData}" \
    --align-with "${ALIGN:-${SNN_CKPT_ROOT}/controls/gru_1024_out80_full96.json}" \
    --out-dir "${SNN_CKPT_ROOT}/controls" ${EXTRA:-}
