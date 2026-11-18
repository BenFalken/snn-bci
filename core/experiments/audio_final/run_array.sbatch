#!/bin/bash
#SBATCH -J AUDIO_FINAL
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH -c 2
#SBATCH -o experiments/audio_final/logs/AUDIO_FINAL_%A_%a.out
#
# One array task per line of a run list (runs_lsnn.txt, runs_speck.txt): trains with snndec.train on the word set and
# writes $ROOT/<name>/best.pt (with its valid/test metrics) and train.log; a run with a 'done' marker is skipped.
# Submit from the repo root:
#   mkdir -p experiments/audio_final/logs
#   L=experiments/audio_final/runs_lsnn.txt; N=$(grep -cv '^#' $L)
#   sbatch --export=ALL,RUNS=$L,ROOT=$SNNBCI_OSCAR_HOME/scratch/audio_final,DATA=$SNNBCI_OSCAR_HOME/scratch/audio_pipeline/ALLWORDS_VERIFIED_DAYNORM/pipeline --array=0-$((N - 1)) experiments/audio_final/run_array.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
: "${RUNS:?set RUNS}" "${ROOT:?set ROOT}" "${DATA:?set DATA}"
ENV_SCRIPT="${ENV_SCRIPT:-$SNNBCI_ROOT/speech/SNNDecoder/scripts/snn_config.sh}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
cd "$(dirname "$ENV_SCRIPT")" && source "$ENV_SCRIPT" && activate_env && cd "$REPO"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
line=$(grep -v '^[[:space:]]*\(#\|$\)' "$RUNS" | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")
[ -n "$line" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $RUNS"; exit 0; }
NAME=$(echo "$line" | cut -d'|' -f1 | xargs); CFG=$(echo "$line" | cut -d'|' -f2 | xargs); SETS=$(echo "$line" | cut -d'|' -f3)
OUT="$ROOT/$NAME"; mkdir -p "$OUT"
[ -f "$OUT/done" ] && { echo "[skip] $NAME finished"; exit 0; }
echo "$NAME: configs/$CFG.yaml --set $SETS"
# shellcheck disable=SC2086
"${PYTHON:-python}" -m snndec.train "configs/$CFG.yaml" --data "$DATA" --out "$OUT" --device cpu --set data.features=features_counts_daynorm.npy $SETS 2>&1 | tee "$OUT/train.log"
touch "$OUT/done"
