#!/bin/bash
# Builds $SNNBCI_WORKSPACE/speck_deploy_jenkins_soft.zip: Motor (II), Jenkins's M1 array, the soft-reset + floor -3 Speck
# SNNs (speck_delay_soft, per_session_delay_soft_binarized_val; adopted 2026-11-27, motor_soft_compare: chip model on
# validation trials 0.936 vs 0.917, 4 of 4), each at its label-free layer-1 factor (the jenkins_m1 lines of
# motor/speck_l1_factors_val_soft.txt) with its readout refitted to the chip model's firing on training trials
# (jenkins_refit.py, as for the hard-reset networks), decoded live on the devkit; motor_chip.py saves each session's
# step-by-step output (<session>_arrays.npz) for the figures and video 2.
# Run ON THE MAC, after copying from Oscar (see jenkins_soft_refit.sbatch):
#   SOFT=~/Downloads/motor_val_jenkins_soft bash chip/make_deploy_bundle_jenkins_soft.sh
# On the Speck laptop:  unzip speck_deploy_jenkins_soft.zip && cd speck_deploy_jenkins_soft && bash run.sh   (~30 min)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; DL="$SNNBCI_WORKSPACE"; SOFT="${SOFT:-$DL/motor_val_jenkins_soft}"
D=per_session_delay_soft_binarized_val
[ -z "$(git -C "$REPO" status --porcelain -- chip snndec motor)" ] || { echo "commit chip/ snndec/ motor/ first"; exit 1; }
T="$(mktemp -d)/speck_deploy_jenkins_soft"
mkdir -p "$T"/{code/snn-decoders,models,data,results}
rsync -a --exclude __pycache__ "$REPO/snndec" "$REPO/chip" "$T/code/snn-decoders/"
git -C "$REPO" log -1 --format='%h %s' > "$T/VERSION"
cp "$REPO/motor/snn_training/velocity_scalers.json" "$T/models/"
: > "$T/models/sessions.txt"
for S in $(awk '$1 == "jenkins_m1" {print $2}' "$REPO/motor/speck_l1_factors_val_soft.txt"); do
  k=$(awk -v s="$S" '$1 == "jenkins_m1" && $2 == s {print $3}' "$REPO/motor/speck_l1_factors_val_soft.txt")
  for f in "$SOFT/$D/$S/best_model_weights.pth" "$SOFT/refit/${S}_readout.npz"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
  mkdir -p "$T/models/$S" "$T/data/hkm/jenkins_m1/$S"
  cp "$SOFT/$D/$S/best_model_weights.pth" "$T/models/$S/"
  cp "$SOFT/refit/${S}_readout.npz" "$T/models/$S/readout_refit.npz"
  rsync -a "$SNNBCI_MOTOR_DATA/snn_datasets/hkm/jenkins_m1/mua/$S/test" "$T/data/hkm/jenkins_m1/$S/"
  echo "$S $k" >> "$T/models/sessions.txt"
done
[ "$(wc -l < "$T/models/sessions.txt")" -eq 4 ] || { echo "expected 4 sessions"; exit 1; }
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
  $PY -W ignore $SD/chip/motor_chip.py --session data/hkm/jenkins_m1/$S --ckpt models/$S/best_model_weights.pth \
      --scalers models/velocity_scalers.json --scale "$k" --readout models/$S/readout_refit.npz --out results/$S.json
done < models/sessions.txt
echo "done: send back the results/ folder"
RUN
chmod +x "$T/run.sh"
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy_jenkins_soft.zip" && zip -qr "$DL/speck_deploy_jenkins_soft.zip" speck_deploy_jenkins_soft )
du -h "$DL/speck_deploy_jenkins_soft.zip"; cat "$T/VERSION" "$T/models/sessions.txt"; echo "bundle: $T"
