#!/bin/bash
# Layer-1 threshold factor for speech Speck networks, chosen without labels, and how well each network's chip model
# agrees with its software. For every checkpoint, on the same TRAINING sentences (never test):
#   torch    the trained network's own phoneme error (the reference)
#   events   the chip's event-by-event firing in software at each layer-1 factor (speech_chip.py --mode events):
#            phoneme error, and agreement = correlation of its logits with the trained network's
# The factor with the highest agreement is chosen (no labels); the table puts software and chip model side by side,
# with input events per 20 ms bin (what the chip must process each step).
#
#   bash speech_l1_sweep.sh DATA_DIR OUT_DIR CKPT [CKPT ...]
#   TRIALS=3 SCALES="1 1.5 2 3 4" bash speech_l1_sweep.sh $SNNBCI_WORKSPACE/speech_speck/snnData results/l1 \
#        $SNNBCI_WORKSPACE/speech_speck/02_R3_hard_txpow_binary_d2/best_model_weights.pth ...
# Each network's run folder name labels it. Finished parts are skipped on a rerun. Runs the networks in parallel
# (JOBS, default 4); events mode is slow, so TRIALS (sentences per session, default 3) sets the cost.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$1"; OUT="$2"; shift 2
[ $# -ge 1 ] || { sed -n 2,14p "$0"; exit 1; }
TRIALS="${TRIALS:-3}"; SCALES="${SCALES:-1 1.5 2 3 4}"; JOBS="${JOBS:-4}"; PY="${PYTHON:-python3}"
mkdir -p "$OUT"
one() {
    local ck="$1" name; name=$(basename "$(dirname "$ck")")
    [ -f "$OUT/${name}_torch.json" ] || "$PY" "$HERE/speech_chip.py" --ckpt "$ck" --data-dir "$DATA" --mode torch \
        --partition train --max-trials "$TRIALS" --out "$OUT/${name}_torch.json" > "$OUT/${name}_torch.log" 2>&1
    [ -f "$OUT/${name}_events.json" ] || "$PY" "$HERE/speech_chip.py" --ckpt "$ck" --data-dir "$DATA" --mode events \
        --partition train --max-trials "$TRIALS" --scales $SCALES --out "$OUT/${name}_events.json" \
        > "$OUT/${name}_events.log" 2>&1
    echo "done: $name"
}
n=0
for ck in "$@"; do
    [ -f "$ck" ] || { echo "missing $ck" >&2; exit 1; }
    one "$ck" & n=$((n + 1))
    [ $((n % JOBS)) -eq 0 ] && wait
done
wait
"$PY" "$HERE/speech_l1_summary.py" "$DATA" "$OUT" "$TRIALS" "$@"
