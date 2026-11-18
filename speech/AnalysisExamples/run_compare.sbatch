#!/bin/bash
#SBATCH -p batch
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH -J COMPARE
#SBATCH -o logs/COMPARE_%j.out
#SBATCH -e logs/COMPARE_%j.out
#
# compareDecoders.py as a batch job: the paper RNN against trained SNNs on the same test trials with the
# same 3-gram language model (needs the compiled lm_decoder; ~60 GB for the LM).
#
# Submit from this directory:
#   mkdir -p logs
#   sbatch run_compare.sbatch                                  # the two default SNNs below
#   sbatch --export=ALL,SNNS="name=/path/ckpt.pth other=/path/ckpt.pth" run_compare.sbatch
#   sbatch --export=ALL,SNNS="...",GRUS="GRU96=/path/controls/gru_1024_out80_full96.json" run_compare.sbatch
#   sbatch --export=ALL,SNNS="...",OUT=/path/name,EXTRA="--acousticScale 1.2 --snnBlankPenalty 0 --profileBins 0" run_compare.sbatch
#   (EXTRA: more compareDecoders.py flags)
#   (GRUS: GRU controls from SNNDecoder/scripts/ann_controls.py, weights in the .pt next to each .json)
# Results: the table at the end of logs/COMPARE_<jobid>.out, and
#          $BASE_DIR/derived/decoderComparison.json (+ _sentences.json, every decoded sentence)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
CK="$BASE_DIR/derived/snns/rsnn_recipe"
SNNS="${SNNS:-R2r_alif_lr02=$CK/00_R2r_alif_lr02/best_model_weights.pth R2r_iaf_lr005=$CK/02_R2r_iaf_lr005/best_model_weights.pth}"

# a venv active in the submitting shell (e.g. ann-env) would shadow the env's python
if [ -n "$VIRTUAL_ENV" ]; then
  PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
  unset VIRTUAL_ENV
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-speechBCI}"
PYTHON="$(conda info --base)/envs/${CONDA_ENV:-speechBCI}/bin/python"
export PYTHONUNBUFFERED=1
echo "python: $PYTHON | SNNs: $SNNS | GRUs: ${GRUS:-none}"

SNN_ARGS=()
for s in $SNNS; do
  [ -f "${s#*=}" ] || { echo "ERROR: no checkpoint at ${s#*=}"; exit 1; }
  SNN_ARGS+=(--snn "$s")
done
for g in ${GRUS:-}; do
  [ -f "${g#*=}" ] || { echo "ERROR: no GRU at ${g#*=}"; exit 1; }
  SNN_ARGS+=(--gru "$g")
done
cd $SNNBCI_ROOT/speech/AnalysisExamples
"$PYTHON" compareDecoders.py --baseDir "$BASE_DIR" "${SNN_ARGS[@]}" ${OUT:+--out "$OUT"} ${EXTRA:-}
