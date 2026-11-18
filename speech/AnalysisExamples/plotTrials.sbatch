#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH -J PLOT_TRIALS
#SBATCH -o logs/PLOT_TRIALS_%j.out
#SBATCH -e logs/PLOT_TRIALS_%j.out
#
# Trial figures from plotExampleTrial.py with the cached RNN, first without and then with the 3-gram language
# model; builds the lm_decoder module first if it does not import yet (needs gcc >= 10, cmake >= 3.14, internet).
#
# Submit from this directory:
#   mkdir -p logs && sbatch plotTrials.sbatch
#   tail -f logs/PLOT_TRIALS_<jobid>.out
#
# Options (sbatch --export=ALL,VAR=value,... plotTrials.sbatch):
#   BASE_DIR   the Dryad data folder
#   SESSION    session to plot (default t12.2022.06.02)
#   PARTITION  train, test or competitionHoldOut (default test)
#   TRIALS     trial indices, space separated (default "0 1 2 3 4")
#   LM_DIR     language model folder (default $BASE_DIR/languageModel; LM_DIR=none skips the LM figures)
#   SNN_CKPT   trained SNN checkpoint, to add the SNN's panels (optional); SNN_NAME its label (default SNN)
#   GRU_JSON   a GRU control's .json (SNNDecoder/scripts/ann_controls.py), to add its panels (optional);
#              GRU_NAME its label (default GRU)
#   CONTROLS   more controls, "NAME=path.json NAME=path.json" (e.g. Wiener and Kalman filters), drawn in order
#   SAVED      decoders from saved logits, "NAME=logits.npz ..." (e.g. the Speck chip's, speech_chip.py --save-logits)
#   EXTRA      more plotExampleTrial.py flags for the LM run, e.g. "--acousticScale 1.6 --snnBlankPenalty 3.47"
#   OUT_DIR    figure folder (default trialPlots)
#   EXPORT_DIR also write one JSON per trial for the interactive viewer (trialViewer/build_viewer.py), from the LM run
#
# Figures: trialPlots/noLM/ and trialPlots/LM/ (in this directory)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail

REPO_DIR=$SNNBCI_ROOT/speech
BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
SESSION="${SESSION:-t12.2022.06.02}"
PARTITION="${PARTITION:-test}"
TRIALS="${TRIALS:-0 1 2 3 4}"
LM_DIR="${LM_DIR:-$BASE_DIR/languageModel}"

#a venv active in the submitting shell (e.g. ann-env) would shadow the env's python
if [ -n "$VIRTUAL_ENV" ]; then
  PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
  unset VIRTUAL_ENV
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-speechBCI}"
export PATH="$(conda info --base)/envs/${CONDA_ENV:-speechBCI}/bin:$PATH"
echo "python: $(command -v python) ($(python --version 2>&1))"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "$REPO_DIR/AnalysisExamples"
echo "node $(hostname) | BASE_DIR=$BASE_DIR | $SESSION $PARTITION trials: $TRIALS"

SNN_ARGS=()
[ -n "$SNN_CKPT" ] && SNN_ARGS=(--snnCkpt "$SNN_CKPT" --snnName "${SNN_NAME:-SNN}")
[ -n "$GRU_JSON" ] && SNN_ARGS+=(--gruJson "$GRU_JSON" --gruName "${GRU_NAME:-GRU}")
for c in ${CONTROLS:-}; do SNN_ARGS+=(--control "$c"); done
for c in ${SAVED:-}; do SNN_ARGS+=(--savedLogits "$c"); done
OUT_DIR="${OUT_DIR:-trialPlots}"

echo "=== 1. Figures without the language model"
python plotExampleTrial.py --baseDir "$BASE_DIR" --session "$SESSION" --partition "$PARTITION" \
  --trials $TRIALS "${SNN_ARGS[@]}" --outDir "$OUT_DIR/noLM"

[ "$LM_DIR" = "none" ] && { echo "LM_DIR=none: done"; exit 0; }

echo "=== 2. lm_decoder module"
if python -c "import lm_decoder" 2>/dev/null; then
  echo "already built"
else
  gcc --version | head -1; cmake --version | head -1
  export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-8}
  cd "$REPO_DIR/LanguageModelDecoder/runtime/server/x86"
  echo "building with $CMAKE_BUILD_PARALLEL_LEVEL cores; full log: $PWD/lm_build.log"
  if ! python setup.py install > lm_build.log 2>&1; then
    tail -40 lm_build.log
    echo "ERROR: lm_decoder build failed (log above); the figures without the LM are in trialPlots/noLM"
    exit 1
  fi
  cd "$REPO_DIR/AnalysisExamples"
  python -c "import lm_decoder; print('lm_decoder OK')"
fi

echo "=== 3. Figures with the language model ($LM_DIR)"
python plotExampleTrial.py --baseDir "$BASE_DIR" --session "$SESSION" --partition "$PARTITION" \
  --trials $TRIALS "${SNN_ARGS[@]}" --lmDir "$LM_DIR" --outDir "$OUT_DIR/LM" ${EXPORT_DIR:+--exportJson "$EXPORT_DIR"} ${EXTRA:-}

ls -lh "$OUT_DIR"/*/
