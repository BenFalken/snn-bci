#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH -J DECLOGITS
#SBATCH -o logs/DECLOGITS_%j.out
#SBATCH -e logs/DECLOGITS_%j.out
#
# decodeSavedLogits.py as a batch job: word error with the 3-gram LM for saved logits (the Speck chip's), at the
# default settings (as every decoder) and at Speck's retuned ones. Submit from this directory:
#   mkdir -p logs
#   sbatch --export=ALL,LOGITS=/path/chip_logits.npz,OUT=/path/name run_decodeLogits.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
[ -f "${LOGITS:?set LOGITS}" ] || { echo "ERROR: no $LOGITS"; exit 1; }
if [ -n "$VIRTUAL_ENV" ]; then
  PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
  unset VIRTUAL_ENV
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-speechBCI}"
PYTHON="$(conda info --base)/envs/${CONDA_ENV:-speechBCI}/bin/python"
export PYTHONUNBUFFERED=1
cd $SNNBCI_ROOT/speech/AnalysisExamples
"$PYTHON" decodeSavedLogits.py --baseDir "$BASE_DIR" --logits "$LOGITS" --name "${NAME:-SpeckChip}" \
    --setting default=0.8,0.6931471805599453 --setting tuned=1.6,3.47 --out "${OUT:-$BASE_DIR/derived/decoderComparison_speckChip}"
