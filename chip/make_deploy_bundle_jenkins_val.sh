#!/bin/bash
# Builds $SNNBCI_WORKSPACE/speck_deploy_jenkins_val.zip: Motor (II), Jenkins's M1 array, the VALIDATION-selected Speck SNNs
# (per_session_delay_binarized_val), each at its label-free layer-1 factor (motor_l1_sweep.py, 20 training trials;
# the jenkins_m1 lines of motor/speck_l1_factors_val.txt) with its readout refitted to the chip model's firing on
# training trials (jenkins_refit.py, the variant chosen by cross-validation there), decoded live on the devkit.
# Run ON THE MAC:  VAL=~/Downloads/motor_val_jenkins bash chip/make_deploy_bundle_jenkins_val.sh
# On the Speck laptop:  unzip speck_deploy_jenkins_val.zip && cd speck_deploy_jenkins_val && bash run.sh   (~30 min)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; DL="$SNNBCI_WORKSPACE"; VAL="${VAL:-$DL/motor_val_jenkins}"
T="$(mktemp -d)/speck_deploy_jenkins_val"
mkdir -p "$T"/{code/snn-decoders,models,data,results}
rsync -a --exclude __pycache__ "$REPO/snndec" "$REPO/chip" "$T/code/snn-decoders/"
cp "$REPO/motor/snn_training/velocity_scalers.json" "$T/models/"
: > "$T/models/sessions.txt"
for S in $(ls "$VAL/per_session_delay_binarized_val"); do
  k=$(awk -v s="$S" '$1 == "jenkins_m1" && $2 == s {print $3}' "$REPO/motor/speck_l1_factors_val.txt" | tail -1)
  [ -n "$k" ] || { echo "no factor for $S"; exit 1; }
  mkdir -p "$T/models/$S" "$T/data/$S"
  cp "$VAL/per_session_delay_binarized_val/$S/best_model_weights.pth" "$T/models/$S/"
  cp "$VAL/refit/${S}_readout.npz" "$T/models/$S/readout_refit.npz"
  rsync -a "$SNNBCI_MOTOR_DATA/snn_datasets/hkm/jenkins_m1/mua/$S/test" "$T/data/$S/"
  echo "$S $k" >> "$T/models/sessions.txt"
done
cat > "$T/run.sh" <<'RUN'
#!/bin/bash
#   bash run.sh check    packages and the devkit
#   bash run.sh          the 4 sessions on the devkit, their test trials, readout refitted, live -> results/
set -eo pipefail
cd "$(dirname "$0")"; PY="${PYTHON:-python3}"; SD=code/snn-decoders
if [ "${1:-}" = check ]; then
  $PY -c "import torch, sinabs; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
  $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
  exit 0
fi
while read -r S k; do
  $PY -W ignore $SD/chip/motor_chip.py --session data/$S --ckpt models/$S/best_model_weights.pth \
      --scalers models/velocity_scalers.json --scale "$k" --readout models/$S/readout_refit.npz --out results/$S.json
done < models/sessions.txt
echo "done: send back the results/ folder"
RUN
chmod +x "$T/run.sh"
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy_jenkins_val.zip" && zip -qr "$DL/speck_deploy_jenkins_val.zip" speck_deploy_jenkins_val )
du -h "$DL/speck_deploy_jenkins_val.zip"; cat "$T/models/sessions.txt"; echo "bundle: $T"
