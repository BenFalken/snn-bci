#!/bin/bash
#
# Fetches what run_final.sh needs to run Jenkins (hkm, 4 sessions) on this laptop and the devkit, into the same
# layout as indy and loco (data/ as on Oscar). Run ON YOUR MAC, from anywhere:
#
#   bash fetch_jenkins.sh             export the test splits on Oscar, then copy everything down
#   DRY_RUN=1 bash fetch_jenkins.sh   list what would be copied (the export still runs)
#   SUBJ=jenkins_m1 bash fetch_jenkins.sh   the M1-array subject instead (96 channels, the standard Speck SNN)
#
# One ssh connection is shared by every step (ControlMaster), so there is one password + Duo prompt.
#
# What it copies (Oscar: $SNNBCI_OSCAR_MOTOR_DATA/...):
#   1. dataset/hkm/jenkins/mua/<session>_binning.h5      test rows only, written on Oscar first by
#                                                         inference/export_test_split.py (keeps trial_id for hkm)
#   2. snn_datasets/hkm/jenkins/mua/<session>/**         everything except train/*.pkl (train/ is kept, empty:
#                                                         the SNN dataloader checks that it exists)
#   3. snn_checkpoints/hkm/jenkins/{lsnn_loso_finetuned_binarized,per_session_delay_binarized}/<session>/
#                                                         best_model_weights.pth and summary_*.txt only
#   4. results/model_cache/hkm/jenkins/**                the conventional decoders, minus the *min duration models
#   5. snn_training/velocity_scalers.json           Oscar's copy (this bundle's has bmi only; hkm's is needed)
#
# Then:  bash run_final.sh jenkins      (devkit plugged in; Speck at layer 1 threshold x2, 3 ms per step)

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

SUBJ="${SUBJ:-jenkins}"     # SUBJ=jenkins_m1: the M1 array alone (make_hkm_array_subset.py), same layout
REMOTE="$SNNBCI_OSCAR_USER@ssh.ccv.brown.edu"
REMOTE_ROOT="$SNNBCI_OSCAR_MOTOR_DATA"
REMOTE_CODE="$SNNBCI_OSCAR_HOME/spike_bmi_speck_delay/main"
EXPORT_DIR="$SNNBCI_OSCAR_HOME/scratch/$USER/laptop_export/dataset/hkm/$SUBJ/mua"
LOCAL_ROOT="$SNNBCI_MOTOR_DATA"

CTL="$HOME/.ssh/cm-jenkins-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=15m)
trap '"${SSH[@]}" -O exit "$REMOTE" 2>/dev/null || true' EXIT

RSYNC_OPTS=(-avz -e "${SSH[*]}")
[[ "${DRY_RUN:-0}" == "1" ]] && { RSYNC_OPTS+=(--dry-run); echo "*** DRY RUN: nothing will be copied ***"; }

echo "== 1/3  export Jenkins's test splits on Oscar"
"${SSH[@]}" "$REMOTE" bash -s <<EOF
set -e
source ~/ann-env/bin/activate 2>/dev/null || true
mkdir -p "$EXPORT_DIR"
python3 "$REMOTE_CODE/inference/export_test_split.py" \
    --input_dir "$REMOTE_ROOT/dataset/hkm/$SUBJ/mua" --output_dir "$EXPORT_DIR"
ls -la "$EXPORT_DIR"
EOF

echo "== 2/3  copy test splits, SNN test data, checkpoints and model cache"
mkdir -p "$LOCAL_ROOT/dataset/hkm/$SUBJ/mua"
rsync "${RSYNC_OPTS[@]}" --include='*_binning.h5' --exclude='*' \
    "$REMOTE:$EXPORT_DIR/" "$LOCAL_ROOT/dataset/hkm/$SUBJ/mua/"

FILTERS=(
    --include='/snn_checkpoints/' --include='/snn_checkpoints/hkm/' --include="/snn_checkpoints/hkm/$SUBJ/"
    --include='/snn_datasets/'    --include='/snn_datasets/hkm/'    --include="/snn_datasets/hkm/$SUBJ/"
    --include='/results/' --include='/results/model_cache/' --include='/results/model_cache/hkm/'
)
for c in lsnn_loso_finetuned_binarized per_session_delay_binarized; do
    FILTERS+=(
        --include="/snn_checkpoints/hkm/$SUBJ/$c/"
        --include="/snn_checkpoints/hkm/$SUBJ/$c/*/"
        --include="/snn_checkpoints/hkm/$SUBJ/$c/*/best_model_weights.pth"
        --include="/snn_checkpoints/hkm/$SUBJ/$c/*/summary_*.txt"
    )
done
FILTERS+=(
    # the excludes must come before the '**' include: the first matching rule wins
    --include="/snn_datasets/hkm/$SUBJ/mua/"
    --exclude="/snn_datasets/hkm/$SUBJ/mua/**/train/**.pkl"
    --include="/snn_datasets/hkm/$SUBJ/mua/**"
    --include="/results/model_cache/hkm/$SUBJ/"
    --exclude="/results/model_cache/hkm/$SUBJ/**[0-9]min*"
    --include="/results/model_cache/hkm/$SUBJ/**"
    --exclude='*'
)
rsync "${RSYNC_OPTS[@]}" "${FILTERS[@]}" "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"

echo "== 3/3  Oscar's velocity scalers (with hkm)"
cp "$HERE/snn_training/velocity_scalers.json" "$HERE/snn_training/velocity_scalers.json.bmi_only" 2>/dev/null || true
rsync "${RSYNC_OPTS[@]}" "$REMOTE:$REMOTE_CODE/snn_training/velocity_scalers.json" "$HERE/snn_training/"

echo "== 4/4  the first ${TRAIN_TRIALS:-20} training trials per session (for the label-free layer-1 factor; never test data)"
for d in "$LOCAL_ROOT"/snn_datasets/hkm/$SUBJ/mua/*/; do
    s=$(basename "$d")
    for i in $(seq 0 $(( ${TRAIN_TRIALS:-20} - 1 ))); do echo "$s/train/$i.pkl"; done
done | rsync "${RSYNC_OPTS[@]}" --files-from=- "$REMOTE:$REMOTE_ROOT/snn_datasets/hkm/$SUBJ/mua/" "$LOCAL_ROOT/snn_datasets/hkm/$SUBJ/mua/"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    find "$LOCAL_ROOT" -depth -mindepth 1 -type d -empty ! -name train -delete
    echo
    echo "Sessions with both checkpoints:"
    for d in "$LOCAL_ROOT"/snn_checkpoints/hkm/$SUBJ/lsnn_loso_finetuned_binarized/*/; do
        s=$(basename "$d")
        lsnn=$([ -f "$d/best_model_weights.pth" ] && echo ok || echo MISSING)
        speck=$([ -f "$LOCAL_ROOT/snn_checkpoints/hkm/$SUBJ/per_session_delay_binarized/$s/best_model_weights.pth" ] && echo ok || echo MISSING)
        test=$(ls "$LOCAL_ROOT/snn_datasets/hkm/$SUBJ/mua/$s/test" 2>/dev/null | wc -l | tr -d ' ')
        h5=$([ -f "$LOCAL_ROOT/dataset/hkm/$SUBJ/mua/${s}_binning.h5" ] && echo ok || echo MISSING)
        train=$(ls "$LOCAL_ROOT/snn_datasets/hkm/$SUBJ/mua/$s/train" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $s: LSNN $lsnn | Speck $speck | SNN test files $test | train trials $train | test split $h5"
    done
    python3 -c "import json; print('velocity scalers:', list(json.load(open('$HERE/snn_training/velocity_scalers.json'))))"
    echo
    echo "Next: bash run_final.sh $SUBJ"
fi
