#!/bin/bash
#SBATCH -p batch
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -J MOTOR_SOFT_CMP
#SBATCH -o motor_soft_compare_%j.out
# EXPLORATORY: motor Speck SNN, soft reset + floor -3 vs hard reset, in the chip model on the validation trials
# (chip/motor_soft_compare.py). From ~/snn-bci:  sbatch chip/motor_soft_compare.sbatch   [SUBJECTS="indy loco"]
source "${SLURM_SUBMIT_DIR:-.}/snnbci_env.sh"
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
DR="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"; OUT="${OUT:-$DR/motor_soft_compare}"
for SUBJ in ${SUBJECTS:-indy loco jenkins_m1}; do
    EXP=bmi; [ "$SUBJ" = jenkins_m1 ] && EXP=hkm
    C="$DR/snn_checkpoints/$EXP/$SUBJ"
    python3 -W ignore chip/motor_soft_compare.py --subject "$SUBJ" --data-root "$DR" --jobs 8 --out "$OUT" \
        --hard-root "$C/per_session_delay_binarized_val" --soft-root "$C/per_session_delay_soft_binarized_val"
done
python3 chip/motor_soft_compare.py --summary "$OUT"
