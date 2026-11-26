#!/bin/bash
# EXPLORATORY (soft reset for the 4 ms audio Speck SNN, audio/speck/runs_fine_soft.txt): builds
# $SNNBCI_WORKSPACE/speck_deploy_fine_soft.zip, FS3 (FT1's recipe with soft reset and a membrane floor of -3, which beat
# FT1 on validation per-word r for both monkeys, 3 seeds: audio/speck/fine_soft_summary.py), monkeys N and G x 3 seeds,
# each with its own label-free layer-1 factor (sweep.json) and its readout refitted to the chip model's firing.
# Run ON THE MAC:  bash chip/make_deploy_bundle_fine_soft.sh
# On the Speck laptop (conda env "speck"), devkit plugged in (one Speck2f):
#   unzip speck_deploy_fine_soft.zip && cd speck_deploy_fine_soft && bash run.sh check && bash run.sh audio   (~35 min)
# then send back results/. FT1's chip run (speck_deploy_fine) is the comparison; nothing final is touched.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DL="$SNNBCI_WORKSPACE"; AD="$SNNBCI_AUDITORY_DATA"; CAL="$DL/final_results/audio_fine"
RF=refit
T="$(mktemp -d)/speck_deploy_fine_soft"
mkdir -p "$T"/{code/snn-decoders,models/audio,data,results}
rsync -a --exclude __pycache__ "$REPO/snndec" "$REPO/chip" "$T/code/snn-decoders/"
: > "$T/models/audio/scales.txt"
for m in Neil Gomez; do
  P="$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline"; Dp="$T/data/$m/pipeline"; mkdir -p "$Dp"
  cp "$P/features_4ms_counts.npy" "$P/window_targets_test.npy" "$Dp/"            # once per monkey (the 4 ms input)
  python3 -c "
import numpy as np, json
P='$P'; n={k: len(np.load(f'{P}/window_targets_{k}.npy', mmap_mode='r')) for k in ('train', 'valid', 'test')}
json.dump({'n': n, 'W': int(np.load(f'{P}/window_features_train.npy', mmap_mode='r').shape[1])}, open('$Dp/lengths.json', 'w'))"
  for s in s11 s1337 s99; do
    R="$CAL/${m}_FS3_soft_floor3_$s"; D="$T/models/audio/${m}_FS3_$s"; mkdir -p "$D"
    [ -f "$R/$RF/readout_event.npz" ] || { echo "not calibrated yet: $R/$RF"; exit 1; }
    cp "$SNNBCI_SNN_RUNS/audio_fine_$m/FS3_soft_floor3_$s/best.pt" "$D/model.pt"
    cp "$R/$RF/readout_event.npz" "$D/readout_refit.npz"
    echo "${m}_FS3_$s $(python3 -c "import json; print(json.load(open('$R/sweep.json'))['chosen_scale'])")" >> "$T/models/audio/scales.txt"
  done
done
cat > "$T/run.sh" <<'RUN'
#!/bin/bash
#   bash run.sh check     packages and the devkit
#   bash run.sh audio     FS3 (real 4 ms spikes, soft reset, floor -3), monkeys N and G x 3 seeds, held-out words -> results/audio/
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"; SD=code/snn-decoders
if [ "${1:-audio}" = check ]; then
  $PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
  $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
  exit 0
fi
mkdir -p results/audio
while read -r name k; do                                  # each network's own label-free layer-1 factor
  m=${name%%_*}; d=models/audio/$name
  $PY -W ignore $SD/chip/audio_chip.py --data "data/$m/pipeline=$d/model.pt=$d/readout_refit.npz" \
      --layer1-scale "$k" --out results/audio/$name.json --save-preds results/audio/preds_$name
done < models/audio/scales.txt
echo "done: send back the results/ folder"
RUN
chmod +x "$T/run.sh"
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy_fine_soft.zip" && zip -qr "$DL/speck_deploy_fine_soft.zip" speck_deploy_fine_soft )
du -h "$DL/speck_deploy_fine_soft.zip"; cat "$T/models/audio/scales.txt"
