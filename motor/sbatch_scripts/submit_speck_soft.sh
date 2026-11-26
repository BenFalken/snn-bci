#!/bin/bash
# EXPLORATORY (2026-11-26): the motor Speck SNN with soft reset and a membrane floor of -3 (speck_delay_soft), as the
# final audio and speech Speck SNNs, trained per session exactly as the final motor Speck SNN (speck_delay,
# validation split VAL_FRAC=0.1), from sbatch_scripts/:
#   bash submit_speck_soft.sh [subject ...]        default: indy loco jenkins_m1
# -> snn_checkpoints/<exp>/<subject>/per_session_delay_soft_binarized_val/<session>/ ; nothing existing is touched.
source "$(cd "$(dirname "$0")" && pwd)/../../snnbci_env.sh"
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p snn_logs
VAL_FRAC=${VAL_FRAC:-0.1}
DATA_ROOT="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco jenkins_m1)
for SUBJ in "${SUBJECTS[@]}"; do
    EXP=bmi; SUB=mua_8_group
    [ "$SUBJ" = jenkins_m1 ] && { EXP=hkm; SUB=mua; }
    N=$(find "$DATA_ROOT/snn_datasets/$EXP/$SUBJ/$SUB" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$N" -gt 0 ] || { echo "ERROR: no sessions under $DATA_ROOT/snn_datasets/$EXP/$SUBJ/$SUB" >&2; exit 1; }
    S=$(sbatch --parsable --array=0-$((N - 1)) \
        --export=ALL,EXPERIMENT=$EXP,SUBJECT=$SUBJ,SPECK_MODEL=speck_delay_soft,VAL_FRAC=$VAL_FRAC,DATA_ROOT=$DATA_ROOT \
        run_snn_per_session_array.sbatch)
    echo "$SUBJ ($N sessions): Speck SNN soft reset job $S"
done
