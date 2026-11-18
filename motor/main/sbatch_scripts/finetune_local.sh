#!/bin/bash
# Fine-tune a leave-one-session-out pretrained Speck SNN (MODEL=speck_delay) on one session, on this machine,
# with the settings run_snn_pooled_finetune_array.sbatch uses on Oscar (snn_config.sh select_model speck_delay)
# and the pretrained checkpoint's own velocity scaling, so the result matches an Oscar fine-tune.
#
#   bash finetune_local.sh <session dataset dir> <pretrained best_model_weights.pth> <output dir>
set -eo pipefail
SESSION_DIR="$1"; PRETRAINED="$2"; OUT="$3"
[ -d "$SESSION_DIR/train" ] && [ -f "$PRETRAINED" ] && [ -n "$OUT" ] || {
    echo "usage: bash finetune_local.sh <session dataset dir> <pretrained best_model_weights.pth> <output dir>" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT=bmi SUBJECT="$(basename "$SESSION_DIR" | cut -d_ -f1)" DATA_ROOT="${DATA_ROOT:-/tmp}" source "$HERE/snn_config.sh"
select_model speck_delay
read -r LO HI MARGIN < <(python3 -W ignore -c "
import sys, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['args']
print(a['velocity_lo'], a['velocity_hi'], a['velocity_margin'])" "$PRETRAINED")
echo "Fine-tuning $(basename "$SESSION_DIR") from $PRETRAINED ($MODEL_DESC; velocity scale $LO $HI $MARGIN) -> $OUT"
mkdir -p "$OUT"
cd "$HERE"
MPLBACKEND=Agg python3 "$TRAIN_SCRIPT" --data-path "$SESSION_DIR" --checkpoint-dir "$OUT" \
    "${TRAIN_ARGS[@]}" --init-weights-from "$PRETRAINED" \
    --velocity-lo "$LO" --velocity-hi "$HI" --velocity-margin "$MARGIN" --device "${DEVICE:-cpu}"
