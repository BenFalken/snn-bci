#!/bin/bash
#SBATCH -p batch
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -J JENKINS_SOFT_REFIT
#SBATCH -o jenkins_soft_refit_%j.out
# Motor (II) soft-reset Speck SNNs: each readout refitted to the chip model's firing on ALL training trials, at the
# session's soft layer-1 factor (motor/speck_l1_factors_val_soft.txt), as the hard-reset networks' were (jenkins_refit.py,
# variant by cross-validation on training). From ~/snn-bci:  sbatch chip/jenkins_soft_refit.sbatch
# -> $DR/motor_val_jenkins_soft/{per_session_delay_soft_binarized_val,refit}/, ready to copy to the Mac.
source "${SLURM_SUBMIT_DIR:-.}/snnbci_env.sh"
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
DR="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}"; O="$DR/motor_val_jenkins_soft"; D=per_session_delay_soft_binarized_val
mkdir -p "$O/$D" "$O/refit"
while read -r subj S k; do
  [ "$subj" = jenkins_m1 ] || continue
  rsync -a "$DR/snn_checkpoints/hkm/jenkins_m1/$D/$S" "$O/$D/"
  python3 -W ignore chip/jenkins_refit.py --session "$DR/snn_datasets/hkm/jenkins_m1/mua/$S" \
      --ckpt "$O/$D/$S/best_model_weights.pth" --scalers motor/snn_training/velocity_scalers.json --scale "$k" \
      --out "$O/refit/$S.json" > "$O/refit/$S.log" 2>&1 &
done < <(grep -v '^#' motor/speck_l1_factors_val_soft.txt)
wait
ls -la "$O/refit"; tail -n 2 "$O"/refit/*.log
echo "copy to the Mac: rsync -a oscar:$O ~/Downloads/"
