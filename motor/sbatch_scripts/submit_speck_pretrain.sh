#!/bin/bash
# The motor Speck SNN with leave-one-session-out pretraining, from sbatch_scripts/:
#   bash submit_speck_pretrain.sh [subject ...]        default: indy loco
# Pretraining was the LSNN's largest single gain; this asks whether the chip network gets it too. Per subject:
#   1. pretrain the deployed Speck network (speck_delay: 128-256-256, hard reset, delays 0/40/100/200 ms) on every
#      session but one (array task i leaves session i out)  -> snn_checkpoints/bmi/<subj>/speckdelay_loso_pretrained_binarized/
#   2. fine-tune it on that session, the checkpoint chosen on a validation split (VAL_FRAC, default 0.1), each task
#      starting when its pretraining succeeds          -> .../speckdelay_loso_finetuned_binarized_val/
# Nothing existing is overwritten. Then: calibrate (label-free layer-1 sweep, readout refit) and score as the
# per-session Speck networks.
source "$(cd "$(dirname "$0")" && pwd)/../../snnbci_env.sh"
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p snn_logs
DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco)
for SUBJ in "${SUBJECTS[@]}"; do
    N=$(find "$DATA_ROOT/snn_datasets/bmi/$SUBJ/mua_8_group" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$N" -gt 0 ] || { echo "ERROR: no sessions for $SUBJ" >&2; exit 1; }
    PRE=$(sbatch --parsable --array=0-$((N - 1)) \
        --export=ALL,EXPERIMENT=bmi,SUBJECT=$SUBJ,MODEL=speck_delay,LOSO=1,DATA_ROOT=$DATA_ROOT \
        run_snn_pooled_pretrain.sbatch)
    FT=$(sbatch --parsable --array=0-$((N - 1)) --dependency=aftercorr:$PRE \
        --export=ALL,EXPERIMENT=bmi,SUBJECT=$SUBJ,MODEL=speck_delay,LOSO=1,VAL_FRAC=${VAL_FRAC:-0.1},DATA_ROOT=$DATA_ROOT \
        run_snn_pooled_finetune_array.sbatch)
    echo "$SUBJ ($N sessions): pretrain job $PRE, fine-tune job $FT"
done
