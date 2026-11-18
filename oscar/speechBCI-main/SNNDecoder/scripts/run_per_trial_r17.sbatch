#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J PER_R17
#SBATCH -o snn_logs/PER_R17_%j.out
#SBATCH -e snn_logs/PER_R17_%j.out
#
# Phoneme error (greedy, no language model) per test sentence for round 17 (S1-S4), with round 16's R3 as the
# anchor (should read 0.4968). Same trials as perTrialPER_final.json. Software only, no chip. Submit from here:
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad run_per_trial_r17.sbatch
set -eo pipefail
source ./snn_config.sh
activate_env
D="$BASE_DIR/derived"
R16="$D/snns/speck_r17"; R3DIR="$D/snns/speck_r16"
OUT=${OUT:-$D/decoderComparison/perTrialPER_r17.json}
mkdir -p "$(dirname "$OUT")"

ckpt() {        # the run folder whose name ends in $1 (folders carry a numeric prefix, e.g. 02_R3_...)
    local d
    d=$(ls -d "$R16"/*"$1" 2>/dev/null | head -1)
    [ -n "$d" ] && [ -f "$d/best_model_weights.pth" ] || { echo "ERROR: no $R16/*$1/best_model_weights.pth" >&2; ls "$R16" >&2; exit 1; }
    echo "$d/best_model_weights.pth"
}
S1=$(ckpt S1_power_d4); S2=$(ckpt S2_top64_d8_long); S3=$(ckpt S3_top128_d4); S4=$(ckpt S4_power_levels_d2)
R3=$(R16="$R3DIR" ckpt R3_hard_txpow_binary_d2)
for v in S1 S2 S3 S4 R3; do echo "$v ${!v}"; done

export PYTHONPATH="$(cd .. && pwd):$(cd ../../NeuralDecoder && pwd):$(cd ../../AnalysisExamples && pwd):${PYTHONPATH:-}"
cd ../../AnalysisExamples
"$PYTHON" perTrialPER.py --baseDir "$BASE_DIR" --noRNN \
    --snn S1="$S1" --snn S2="$S2" --snn S3="$S3" --snn S4="$S4" --snn R3="$R3" \
    --out "$OUT"
