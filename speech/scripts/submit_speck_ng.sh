#!/bin/bash
# EXPLORATORY: the "next-generation Speck" speech networks (configs/speck_ng.txt), on Oscar, from speech/scripts:
#   bash submit_speck_ng.sh
# Three runs, one GPU each, at most two at a time (the per-user GPU limit). Each trains V2+C's recipe as one chain with
# a larger input budget and wider layers (2x, 4x, 13x). Checkpoints: $BASE_DIR/derived/snns/speck_ng/<NN>_<name>/.
# A run that reaches the 24 h limit resumes from its latest checkpoint when this script is run again.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
B=$SNNBCI_OSCAR_SCRATCH
mkdir -p snn_logs
[ -f configs/channels/r3/all256_top128first.txt ] || { echo "ERROR: configs/channels/r3/all256_top128first.txt missing" >&2; exit 1; }
R=$(sbatch --parsable --export=ALL,BASE_DIR=$B,RUNS=configs/speck_ng.txt,PACK=1,SESSIONS=all,EPOCHS=96 \
    --array=0-2%2 run_snn_configs.sbatch)
echo "next-generation Speck job $R: logs snn_logs/SNN_RUN_${R}_{0,1,2}.out; checkpoints $B/derived/snns/speck_ng/"
echo "when they finish, copy each run's best_model_weights.pth to the Mac:"
echo "  scp -r $SNNBCI_OSCAR_USER@ssh.ccv.brown.edu:$B/derived/snns/speck_ng ~/Downloads/"
