#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J PER_R16
#SBATCH -o snn_logs/PER_R16_%j.out
#SBATCH -e snn_logs/PER_R16_%j.out
#
# Phoneme error (greedy, no language model) per test sentence for round 16's other one-chip networks, which were only
# ever compared on validation loss: R1 (counts only, 2 levels, 2 delays), R2 (counts + spike-band power, 2 levels, no
# delay), R4 (R2 sparser), with R3 (the chosen network) again as the anchor. Same trials as perTrialPER_final.json.
# Software only, no chip. Submit from this directory:
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad run_per_trial_r16.sbatch
set -eo pipefail
source ./snn_config.sh
activate_env
D="$BASE_DIR/derived"
R16="$D/snns/speck_r16"
OUT=${OUT:-$D/decoderComparison/perTrialPER_r16.json}
mkdir -p "$(dirname "$OUT")"

ckpt() {        # the run folder whose name ends in $1 (folders carry a numeric prefix, e.g. 02_R3_...)
    local d
    d=$(ls -d "$R16"/*"$1" 2>/dev/null | head -1)
    [ -n "$d" ] && [ -f "$d/best_model_weights.pth" ] || { echo "ERROR: no $R16/*$1/best_model_weights.pth" >&2; ls "$R16" >&2; exit 1; }
    echo "$d/best_model_weights.pth"
}
R1=$(ckpt R1_H6_hard); R2=$(ckpt R2_hard_txpow_levels_nodelay); R3=$(ckpt R3_hard_txpow_binary_d2)
R4=$(ckpt R4_hard_txpow_levels_nodelay_rate03)
echo "R1 $R1"; echo "R2 $R2"; echo "R3 $R3"; echo "R4 $R4"

export PYTHONPATH="$(cd .. && pwd):$(cd ../../NeuralDecoder && pwd):$(cd ../../AnalysisExamples && pwd):${PYTHONPATH:-}"
cd ../../AnalysisExamples
"$PYTHON" perTrialPER.py --baseDir "$BASE_DIR" --noRNN \
    --snn R1="$R1" --snn R2="$R2" --snn R3="$R3" --snn R4="$R4" \
    --out "$OUT"
