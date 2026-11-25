#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J PER_LOGITS
#SBATCH -o snn_logs/PER_LOGITS_%j.out
#SBATCH -e snn_logs/PER_LOGITS_%j.out
#
# The final speech decoders' test logits, for the phoneme n-gram (scripts/ngram_lm.py): the paper
# RNN, the same-input GRU and LSTM and the LSNN, saved by perTrialPER.py --saveLogits (the Speck SNN's chip logits are
# already on the Mac). Its scores must reproduce perTrialPER_final.json (RNN 0.1689, GRU 0.1910, LSTM 0.1871, LSNN
# 0.2228); that file is not touched. Submit from here:
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_per_trial_logits.sbatch
# RNN_DIR=<folder holding the paper RNN's args.yaml> if it is not $BASE_DIR/derived/rnns/baselineRelease;
# NO_RNN=1 skips the paper RNN (TensorFlow) and saves the LSNN, GRU and LSTM only.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env
D="$BASE_DIR/derived"
OUT=${OUT:-$D/decoderComparison/perTrialPER_logits.json}; LOGITS=${LOGITS:-$D/decoderComparison/logits_final}
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$(cd .. && pwd):$(cd ../../external/willett_speechBCI/NeuralDecoder && pwd):$(cd ../analysis && pwd):${PYTHONPATH:-}"
cd ../analysis
# each decoder only if its checkpoint (and, once saved, not its logits again) is there: scratch may have lost some
ARGS=()
add() { [ -f "$LOGITS/${2}_logits.npz" ] && { echo "[have] $2"; return; }
        [ -f "$3" ] && ARGS+=("$1" "$2=$3") || echo "[missing] $2: $3"; }
add --snn LSNN "$D/snns/nodelay96/01_C_lsnn_nodelay96/final_model_speech_iaf_soft.pth"
add --gru GRU96 "$D/snns/controls/gru_1024_out80_full96.json"
add --gru LSTM96 "$D/snns/controls/lstm_1024_out80_full96.json"
"$PYTHON" perTrialPER.py --baseDir "$BASE_DIR" "${ARGS[@]}" \
    ${RNN_DIR:+--rnnDir "$RNN_DIR"} ${NO_RNN:+--noRNN} --saveLogits "$LOGITS" --out "$OUT"
