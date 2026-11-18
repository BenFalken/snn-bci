#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J PER_TRIAL
#SBATCH -o snn_logs/PER_TRIAL_%j.out
#SBATCH -e snn_logs/PER_TRIAL_%j.out
#
# Greedy phoneme error per test sentence (speech/analysis/perTrialPER.py) for the final speech decoders: the paper
# RNN, the same-input GRU and LSTM, the LSNN, the Speck SNN (R3) in software and on the chip. No training, no language model. Submit from here:
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_per_trial_per.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env
D="$BASE_DIR/derived"
OUT=${OUT:-$D/decoderComparison/perTrialPER_final.json}
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$(cd .. && pwd):$(cd ../../external/willett_speechBCI/NeuralDecoder && pwd):$(cd ../analysis && pwd):${PYTHONPATH:-}"
cd ../analysis
"$PYTHON" perTrialPER.py --baseDir "$BASE_DIR" \
    --snn LSNN="$D/snns/nodelay96/01_C_lsnn_nodelay96/final_model_speech_iaf_soft.pth" \
    --snn SpeckR3="$D/snns/speck_r16/02_R3_hard_txpow_binary_d2/best_model_weights.pth" \
    --gru GRU96="$D/snns/controls/gru_1024_out80_full96.json" \
    --gru LSTM96="$D/snns/controls/lstm_1024_out80_full96.json" \
    --logits SpeckChip="$D/speck_chip_r3/chip_logits.npz" \
    --out "$OUT"
