#!/bin/bash
# EXPLORATORY (motor soft reset, speck_delay_soft): builds $SNNBCI_WORKSPACE/speck_deploy_motor_soft.zip, the Motor (I)
# Speck SNNs (indy 36 + loco 9 sessions) trained with soft reset and a membrane floor of -3, validation-selected
# (per_session_delay_soft_binarized_val), each at its label-free layer-1 factor (motor/speck_l1_factors_val_soft.txt,
# from chip/motor_soft_compare.py), with the trained readout, on the devkit: the same pipeline, wait and test trials as
# the final hard-reset run (motor/final_val.sh, results/motor/final_v3), so the two compare session by session.
# Run ON THE MAC, from a clean checkout (the code is the committed HEAD):
#   SOFT=~/Downloads/motor_val_soft bash chip/make_deploy_bundle_motor_soft.sh
# On the Speck laptop, devkit plugged in (MOTOR_DATA = the speck_per_session/data folder the final run used):
#   unzip speck_deploy_motor_soft.zip && cd speck_deploy_motor_soft && bash run.sh check && bash run.sh motor MOTOR_DATA
# (~3 h, Speck only) then send back results/. Nothing final is touched: results go to results/motor/final_v3_soft/.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; DL="$SNNBCI_WORKSPACE"; SOFT="${SOFT:-$DL/motor_val_soft}"
D=per_session_delay_soft_binarized_val; F="$REPO/motor/speck_l1_factors_val_soft.txt"
[ -z "$(git -C "$REPO" status --porcelain -- motor chip snndec)" ] || { echo "commit motor/ chip/ snndec/ first"; exit 1; }
T="$(mktemp -d)/speck_deploy_motor_soft"
mkdir -p "$T/snn-bci" "$T/motor_val/$D" "$T/results"
git -C "$REPO" archive HEAD | tar -x -C "$T/snn-bci"
git -C "$REPO" log -1 --format='%h %s' > "$T/VERSION"
n=0
for S in $(awk '!/^#/ {print $2}' "$F"); do
  [ -f "$SOFT/$D/$S/best_model_weights.pth" ] || { echo "missing checkpoint: $SOFT/$D/$S"; exit 1; }
  cp -r "$SOFT/$D/$S" "$T/motor_val/$D/"; n=$((n + 1))
done
[ "$n" -eq 45 ] || { echo "expected 45 sessions, got $n"; exit 1; }
cat > "$T/run.sh" <<'RUN'
#!/bin/bash
# Motor (I) soft-reset Speck SNNs on the devkit (EXPLORATORY; see snn-bci/chip/make_deploy_bundle_motor_soft.sh):
#   bash run.sh check                        packages and the devkit
#   bash run.sh motor MOTOR_DATA [subj ...]  Speck only, default indy loco (~3 h); MOTOR_DATA = this laptop's
#                                            speck_per_session/data folder (snn_datasets/ and dataset/), as for final_v3
set -eo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; PY="${PYTHON:-python3}"
what="${1:-check}"; shift || true
mkdir -p "$B/results"
case "$what" in
  check)
    $PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
    $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
    ;;
  motor)
    [ -n "$1" ] || { echo "ERROR: give the motor data folder, e.g. bash run.sh motor ~/speck_per_session/data"; exit 1; }
    D="$(cd "$1" && pwd)"; shift
    for need in snn_datasets/bmi/indy snn_datasets/bmi/loco; do
      [ -d "$D/$need" ] || { echo "ERROR: $D has no $need: not the motor data folder?"; exit 1; }
    done
    printf 'motor_data: %s\nmotor_results: %s\n' "$D" "$B/results/motor" > "$B/snn-bci/paths.local.yaml"
    SUBJ=("$@"); [ ${#SUBJ[@]} -eq 0 ] && SUBJ=(indy loco)
    SPECK_DIR=per_session_delay_soft_binarized_val FACTORS=speck_l1_factors_val_soft.txt OUT_NAME=final_v3_soft \
      DECODERS=speck VAL_ROOT="$B/motor_val" bash "$B/snn-bci/motor/final_val.sh" "${SUBJ[@]}"
    ;;
  *) echo "usage: bash run.sh check | motor MOTOR_DATA [subject ...]"; exit 1 ;;
esac
echo "done: send back $B/results"
RUN
chmod +x "$T/run.sh"
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy_motor_soft.zip" && zip -qr "$DL/speck_deploy_motor_soft.zip" speck_deploy_motor_soft )
du -h "$DL/speck_deploy_motor_soft.zip"; cat "$T/VERSION"
awk '!/^#/ {print $1, $3}' "$F" | sort | uniq -c
echo "bundle: $T"
