#!/bin/bash
# Retrain the final motor spiking decoders with model selection on a validation split, from sbatch_scripts/:
#   bash submit_val_reselect.sh [subject ...]        default: indy loco jenkins
# The original runs chose each checkpoint, the learning-rate schedule and early stopping on the test split. These
# runs hold out the chronologically last VAL_FRAC (default 0.1) of each session's training trials instead, and score
# the test split once, on the chosen checkpoint (train_snn.py --val-frac). Per subject, two arrays (one task per
# session):
#   the LSNN      fine-tuned from the existing leave-one-session-out pretraining (which never sees the session)
#                 -> snn_checkpoints/<exp>/<subject>/lsnn_loso_finetuned_binarized_val/<session>/
#   the Speck SNN trained per session with its input delays (bmi: speck_delay; jenkins: speck_delay2)
#                 -> snn_checkpoints/<exp>/<subject>/per_session_delay[2]_binarized_val/<session>/
# Nothing existing is overwritten. Re-running skips finished sessions and resumes interrupted ones.
source "$(cd "$(dirname "$0")" && pwd)/../../snnbci_env.sh"
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p snn_logs
VAL_FRAC=${VAL_FRAC:-0.1}
DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco jenkins)
for SUBJ in "${SUBJECTS[@]}"; do
    EXP=bmi; SUB=mua_8_group; SPECK=speck_delay
    [ "$SUBJ" = jenkins ] && { EXP=hkm; SUB=mua; SPECK=speck_delay2; }
    N=$(find "$DATA_ROOT/snn_datasets/$EXP/$SUBJ/$SUB" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$N" -gt 0 ] || { echo "ERROR: no sessions under $DATA_ROOT/snn_datasets/$EXP/$SUBJ/$SUB" >&2; exit 1; }
    L=$(sbatch --parsable --array=0-$((N - 1)) \
        --export=ALL,EXPERIMENT=$EXP,SUBJECT=$SUBJ,MODEL=lsnn,VAL_FRAC=$VAL_FRAC,DATA_ROOT=$DATA_ROOT \
        run_snn_pooled_finetune_array.sbatch)
    S=$(sbatch --parsable --array=0-$((N - 1)) \
        --export=ALL,EXPERIMENT=$EXP,SUBJECT=$SUBJ,SPECK_MODEL=$SPECK,VAL_FRAC=$VAL_FRAC,DATA_ROOT=$DATA_ROOT \
        run_snn_per_session_array.sbatch)
    echo "$SUBJ ($N sessions): LSNN job $L, Speck SNN job $S"
done
echo "logs: snn_logs/ ; each run's summary_*.txt gives its best validation loss and the test loss at that checkpoint"
