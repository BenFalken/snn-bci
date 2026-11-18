#!/bin/bash
# Run on the Speck-connected laptop: copy from Oscar what run_inference.sbatch --local needs to run the
# fine-tuned Speck SNN on the chip, for every session whose fine-tuning has finished (re-run later to add
# the sessions finished since; files already here are skipped). One ssh connection, so one password + Duo.
#
#   bash fetch_for_speck.sh [subject ...]            (default: indy loco)
#
# Settings (environment):
#   DATA_ROOT  local data root (default: the data/ folder next to main/, which run_inference.sbatch
#              --local uses without DATA_ROOT)
#   CKPT_DIR   checkpoint folder to fetch (default speckdelay_loso_finetuned_binarized: the Speck SNN
#              with input delays, pretrained leave-one-session-out and fine-tuned)
#   ANN=0      skip the decoders' model bundles (results/model_cache), needed only to run kf/wf/lstm/qrnn
#
# Per session it copies, into the same layout under DATA_ROOT as on Oscar:
#   snn_checkpoints/bmi/<subject>/<CKPT_DIR>/<session>/best_model_weights.pth
#   snn_datasets/bmi/<subject>/mua_8_group/<session>/test/            the SNN test trial(s)
#   dataset/bmi/<subject>/mua/<session>_binning.h5                    test rows only (export_test_split.py,
#                                                                     run on Oscar first when missing)
#   results/model_cache/bmi/<subject>/<session>/                      unless ANN=0
set -eo pipefail
REMOTE="${REMOTE:-$USER@ssh.ccv.brown.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/users/$USER/scratch/$USER/data}"
REMOTE_CODE="${REMOTE_CODE:-/users/$USER/spike_bmi_speck_delay/main}"
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/data}"
CKPT_DIR="${CKPT_DIR:-speckdelay_loso_finetuned_binarized}"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco)

SOCK="$HOME/.ssh/cm-fetch-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$SOCK" -o ControlPersist=15m)
"${SSH[@]}" -fN "$REMOTE"                      # log in once; later ssh/rsync calls reuse the connection
trap '"${SSH[@]}" -O exit "$REMOTE" 2>/dev/null || true' EXIT
RSYNC=(rsync -a -e "${SSH[*]}")
mkdir -p "$DATA_ROOT"
echo "Local DATA_ROOT: $DATA_ROOT"

for subj in "${SUBJECTS[@]}"; do
    # sessions with a finished fine-tune; their test-only ANN datasets exported on Oscar where missing
    sessions=$("${SSH[@]}" "$REMOTE" bash -s -- "$REMOTE_ROOT" "$REMOTE_CODE" "$subj" "$CKPT_DIR" <<'EOF'
root=$1; code=$2; subj=$3; ckpt=$4
dir="$root/snn_checkpoints/bmi/$subj/$ckpt"
export_dir="$root/laptop_export/dataset/bmi/$subj/mua"
done_sessions=()
for d in "$dir"/*/; do
    s=$(basename "$d")
    ls "$d"/final_model_*.pth >/dev/null 2>&1 && [ -f "$d/best_model_weights.pth" ] && done_sessions+=("$s")
done
missing=0
for s in "${done_sessions[@]}"; do [ -f "$export_dir/${s}_binning.h5" ] || missing=1; done
if [ "$missing" = 1 ]; then
    # the inference jobs' environment (run_inference.sbatch VENV, ~/ann-env): export_test_split.py imports
    # the inference code, which needs TensorFlow; a non-interactive login has neither it nor conda
    if [ -f ~/ann-env/bin/activate ]; then source ~/ann-env/bin/activate
    elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; conda activate snn_speck >&2; fi
    mkdir -p "$export_dir"
    python3 "$code/inference/export_test_split.py" --input_dir "$root/dataset/bmi/$subj/mua" \
        --output_dir "$export_dir" >&2
fi
echo "${done_sessions[@]}"
EOF
)
    if [ -z "$sessions" ]; then
        echo "$subj: no finished fine-tunes in $CKPT_DIR yet"
        continue
    fi
    n=$(echo $sessions | wc -w | tr -d ' ')
    echo "$subj: $n finished: $sessions"
    list=$(mktemp)
    for s in $sessions; do
        echo "snn_checkpoints/bmi/$subj/$CKPT_DIR/$s/best_model_weights.pth"
        echo "snn_datasets/bmi/$subj/mua_8_group/$s/test/"
        [ "${ANN:-1}" = "0" ] || echo "results/model_cache/bmi/$subj/$s/"
    done > "$list"
    "${RSYNC[@]}" -r --files-from="$list" "$REMOTE:$REMOTE_ROOT/" "$DATA_ROOT/"
    rm -f "$list"
    mkdir -p "$DATA_ROOT/dataset/bmi/$subj/mua"
    "${RSYNC[@]}" $(for s in $sessions; do echo "$REMOTE:$REMOTE_ROOT/laptop_export/dataset/bmi/$subj/mua/${s}_binning.h5"; done) \
        "$DATA_ROOT/dataset/bmi/$subj/mua/"
    echo "$subj: copied $n session(s) into $DATA_ROOT"
done
echo
echo "Run the chip check, per subject (on this laptop, devkit connected):"
echo "  cd $(cd "$(dirname "$0")" && pwd)"
echo "  bash run_inference.sbatch --local bmi indy DECODERS=snn,speck \\"
echo "      SNN_CHECKPOINT_ROOT=$DATA_ROOT/snn_checkpoints/bmi/indy/$CKPT_DIR \\"
echo "      SPECK_CHECKPOINT_ROOT=$DATA_ROOT/snn_checkpoints/bmi/indy/$CKPT_DIR \\"
echo "      RESULTS_DIR=$DATA_ROOT/results/test_all_decoders_speck/bmi/indy SNN_LABEL='Speck SNN (PyTorch)' \\"
echo "      SPECK_WAIT_TIME=0.003      # the delay bank's input needs 3 ms per step on the chip (1 ms loses accuracy)"
