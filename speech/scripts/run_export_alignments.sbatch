#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J ALIGN
#SBATCH -o snn_logs/ALIGN_%j.out
#SBATCH -e snn_logs/ALIGN_%j.out
#
# Phoneme timing for the speech lag figure: the final LSNN (the same weights as the paper's speech scoring,
# run_per_trial_per.sbatch) over the training sentences of its 19 sessions, each sentence's phonemes force-aligned to
# its outputs (export_alignments.py). CPU only, no training. Submit from here:
#   mkdir -p snn_logs; sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_export_alignments.sbatch
# Then copy the one output file to the Mac (the command is printed at the end of the log).
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env
D="$BASE_DIR/derived"
CKPT="${CKPT:-$D/snns/nodelay96/01_C_lsnn_nodelay96/final_model_speech_iaf_soft.pth}"
OUT="${OUT:-$D/lag_alignments/alignments_lsnn_train.npz}"
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT (set CKPT=...)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
"$PYTHON" export_alignments.py --ckpt "$CKPT" --data-dir "$D/snnData" --out "$OUT"
echo
echo "copy to the Mac:  scp $SNNBCI_OSCAR_USER@ssh.ccv.brown.edu:$OUT ~/Downloads/lag_tuning/"
