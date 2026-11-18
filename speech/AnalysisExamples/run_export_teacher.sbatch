#!/bin/bash
#SBATCH -p batch
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J TEACHER
#SBATCH -o logs/TEACHER_%j.out
#SBATCH -e logs/TEACHER_%j.out
#
# exportTeacherLogits.py as a batch job: the cached paper RNN's per-frame logits on every train and test trial of
# the 19 baseline sessions, for SNN distillation (train.py --teacher-dir). CPU only.
#   mkdir -p logs && sbatch run_export_teacher.sbatch
# Output: $BASE_DIR/derived/teacherLogits/baselineRelease/<session>/{train,test}.npz
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
if [ -n "$VIRTUAL_ENV" ]; then
  PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
  unset VIRTUAL_ENV
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-speechBCI}"
PYTHON="$(conda info --base)/envs/${CONDA_ENV:-speechBCI}/bin/python"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
cd $SNNBCI_ROOT/speech/AnalysisExamples
"$PYTHON" exportTeacherLogits.py --baseDir "$BASE_DIR"
