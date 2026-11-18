#!/bin/bash
# Every line of the run lists on this machine, P at a time (default 4, 3 threads each); same outputs as run_array.sbatch.
#   bash audio/speck/run_local.sh            (from the repo root)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
DATA="${DATA:-$SNNBCI_AUDITORY_DATA/Neil/aSTG/ALLWORDS_VERIFIED_DAYNORM/pipeline}"
ROOT="${ROOT:-$SNNBCI_SNN_RUNS/audio_final}"; P="${P:-4}"; export OMP_NUM_THREADS="${THREADS:-3}"
mkdir -p "$ROOT"
one() {
    NAME=$(echo "$1" | cut -d'|' -f1 | xargs); CFG=$(echo "$1" | cut -d'|' -f2 | xargs); SETS=$(echo "$1" | cut -d'|' -f3)
    OUT="$ROOT/$NAME"; mkdir -p "$OUT"
    [ -f "$OUT/done" ] && { echo "[skip] $NAME"; return 0; }
    echo "[start] $NAME $(date +%H:%M)"
    # shellcheck disable=SC2086
    python3 -m snndec.train "snndec/configs/$CFG.yaml" --data "$DATA" --out "$OUT" --device cpu \
        --set data.features=features_counts_daynorm.npy $SETS > "$OUT/train.log" 2>&1 && touch "$OUT/done"
    echo "[done] $NAME $(date +%H:%M): $(grep -c '^epoch' "$OUT/train.log") epochs, best $(grep -o 'r_valid [0-9.]*' "$OUT/train.log" | sort -k2 -n | tail -1)"
}
export -f one; export DATA ROOT
# LSNN runs first (fast), then the Speck runs
cat ${LISTS:-audio/speck/runs_lsnn.txt audio/speck/runs_speck.txt} | grep -v '^[[:space:]]*\(#\|$\)' \
    | tr '\n' '\0' | xargs -0 -P "$P" -n 1 bash -c 'one "$1"' _
