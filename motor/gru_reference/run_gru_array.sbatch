#!/bin/bash
#SBATCH -J GRU_REF
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH -o gru_logs/GRU_REF_%A_%a.out
#SBATCH -e gru_logs/GRU_REF_%A_%a.out
# One array task = one (session, model) pair, looping over SEEDS. Submit with submit_gru.sh.
#   PROTOCOL=neurobench|ours  SESSIONS=<file>  SEEDS="0 1 2"  OUT=<dir>
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate snn_speck
D=${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}
mapfile -t S < <(grep -v '^\s*$' "$SESSIONS")
MODELS=(aegru gru)
SESSION=${S[$((SLURM_ARRAY_TASK_ID / 2))]}; MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 2))]}
SUBJ=${SESSION%%_*}
if [ "$PROTOCOL" = neurobench ]; then IN=$D/raw/bmi/$SUBJ/$SESSION.mat; else IN=$D/processed/bmi/$SUBJ/mua/$SESSION.h5; fi
[ -f "$IN" ] || { echo "ERROR: missing $IN" >&2; exit 1; }
for SEED in $SEEDS; do
    python3 run_gru.py --protocol "$PROTOCOL" --input "$IN" --model "$MODEL" --seed "$SEED" --threads 4 --out "$OUT"
done
