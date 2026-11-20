# Shared settings of the SNN's two training stages (sourced by
# run_snn_pretrain.sbatch and run_snn_finetune.sbatch, from this directory):
#
#   pretrain   the pooled network (snnDecoder/configs/speech_pooled.yaml) on
#              every training session of the RNN baseline, hard reset
#   finetune   from the pretrained weights (--init-weights-from, fresh
#              optimizer), soft reset; the reset has no weights, so the
#              pretrained weights load unchanged
#
# Requires BASE_DIR (the folder holding derived/tfRecords); the .npz data
# come from speech/analysis/exportTFRecordsToNpz.py.

BASE_DIR="${BASE_DIR:?set BASE_DIR, e.g. sbatch --export=ALL,BASE_DIR=/path/to/speechPaperRelease_08_20 ...}"
SNN_DATA_DIR="${SNN_DATA_DIR:-${BASE_DIR}/derived/snnData}"
SNN_CKPT_ROOT="${SNN_CKPT_ROOT:-${BASE_DIR}/derived/snns}"
PRETRAIN_DIR="${SNN_CKPT_ROOT}/pretrain_hard"
FINETUNE_DIR="${SNN_CKPT_ROOT}/finetune_soft"
CONFIG_YAML="$(cd .. && pwd)/snnDecoder/configs/speech_pooled.yaml"

# select_stage pretrain|finetune -> RESET_TYPE, EXP_NAME, TRAIN_ARGS
select_stage() {
    case "$1" in
        pretrain) RESET_TYPE=hard ;;
        finetune) RESET_TYPE=soft ;;
        *) echo "ERROR: select_stage pretrain|finetune, got '$1'" >&2; exit 1 ;;
    esac
    EXP_NAME="speech_iaf_${RESET_TYPE}"   # file prefix train.py uses (speech_{neuron}_{reset})
    TRAIN_ARGS=(
        --config "$CONFIG_YAML"
        --data-dir "$SNN_DATA_DIR"
        --reset-type "$RESET_TYPE"
    )
}

# latest_resume_args DIR -> RESUME_ARGS resuming from DIR's newest periodic
# checkpoint, else from its best_model_weights.pth, if any.
latest_resume_args() {
    local latest
    RESUME_ARGS=()
    latest=$(ls "$1"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    [ -z "$latest" ] && [ -f "$1/best_model_weights.pth" ] && latest="$1/best_model_weights.pth"
    if [ -n "$latest" ]; then
        echo "[resume] $latest"
        RESUME_ARGS=(--resume "$latest")
    fi
}

# activate_env -> the conda env active, PYTHON its interpreter. sbatch --export=ALL hands the job the
# submitting shell's PATH, so a venv active there (e.g. ann-env) would otherwise shadow the env's python.
activate_env() {
    if [ -n "$VIRTUAL_ENV" ]; then
        PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
        unset VIRTUAL_ENV
    fi
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-speechBCI}"
    PYTHON="$(conda info --base)/envs/${CONDA_ENV:-speechBCI}/bin/python"
    [ -x "$PYTHON" ] || { echo "ERROR: no python at $PYTHON (conda env ${CONDA_ENV:-speechBCI} missing?)" >&2; exit 1; }
    export PATH="$(dirname "$PYTHON"):$PATH"
    export PYTHONUNBUFFERED=1   # log lines appear as they happen, not in 8 KB blocks
    # this repository's snnDecoder (speech/) before any copy installed in the environment (an editable install of an
    # older checkout would otherwise be imported by `python -m snnDecoder.train`)
    export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
    echo "python: $PYTHON ($("$PYTHON" --version 2>&1)); snnDecoder from $("$PYTHON" -c 'import snnDecoder, os; print(os.path.dirname(snnDecoder.__file__))' 2>/dev/null)"
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
}
