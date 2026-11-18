#!/bin/bash
# Devkit bundle for the chosen two-network audio variant (summary.json), both monkeys x 3 seeds, readouts refitted and
# applied live; $SNNBCI_WORKSPACE/speck_audio_two(.zip).
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
SD=$(cd "$(dirname "$0")/../.." && pwd); AD=$SNNBCI_AUDITORY_DATA
OUT=$SNNBCI_WORKSPACE/final_results/audio_two; B=$SNNBCI_WORKSPACE/speck_audio_two
V=$(python3 -c "import json; print(json.load(open('$OUT/summary.json'))['chosen'])")
[ "$V" != None ] || { echo "no variant to bundle"; exit 1; }
rm -rf $B; mkdir -p $B/code $B/data $B/results
rsync -a --exclude __pycache__ $SD/snndec $B/code/; rsync -a --exclude __pycache__ $SD/chip $B/code/
LINES=""
for m in Neil Gomez; do for s in s11 s1337 s99; do
  r=${V}_$s; N=$OUT/${m}_$r; D=$B/data/${m}_$r; P=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline
  [ -f $N/refit/readout_event.npz ] || continue
  mkdir -p $D/pipeline
  cp $SNNBCI_SNN_RUNS/audio_two_$m/$r/best.pt $D/model.pt; cp $N/refit/readout_event.npz $D/readout_refit.npz
  cp $P/features_counts_daynorm.npy $P/window_targets_test.npy $D/pipeline/
  python3 -c "
import numpy as np, json
P='$P'; n={k: len(np.load(f'{P}/window_targets_{k}.npy', mmap_mode='r')) for k in ('train','valid','test')}
json.dump({'n': n, 'W': int(np.load(f'{P}/window_features_train.npy', mmap_mode='r').shape[1])}, open('$D/pipeline/lengths.json','w'))"
  k=$(python3 -c "import json; print(json.load(open('$N/sweep.json'))['chosen_scale'])")
  LINES="$LINES ${m}_$r:$k"
done; done
cat > $B/run.sh <<EOS
#!/bin/bash
# Audio Speck as TWO networks on one Speck2f (the speech V2 layout): variant $V, all 96 channels, both monkeys x 3 seeds,
# each network on its own cores; layer 1 factor per network chosen without labels; readout refitted to the chip
# model's spikes, applied live. ~35 min.
#   bash run.sh check | bash run.sh quick (one network, confirms both chains place) | bash run.sh
set -eo pipefail
cd "\$(dirname "\$0")"
PY="\${PYTHON:-python3}"
if [ "\${1:-}" = check ]; then
  \$PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
  \$PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
  exit 0
fi
mkdir -p results
ALL="$LINES"
[ "\${1:-}" = quick ] && ALL="\${ALL%% *}" && ALL="\$(echo $LINES | cut -d' ' -f1)"
for pair in \$ALL; do
  n=\${pair%%:*}; x=\${pair##*:}; d=data/\$n
  echo "== \$n, layer 1 x\$x"
  \$PY -W ignore code/chip/audio_chip.py --data "\$d/pipeline=\$d/model.pt=\$d/readout_refit.npz" \\
      --layer1-scale \$x --out results/\$n.json --save-preds results/preds_\$n
done
echo "Done: send back the results/ folder."
EOS
chmod +x $B/run.sh
cd $SNNBCI_WORKSPACE && rm -f speck_audio_two.zip && zip -qr speck_audio_two.zip speck_audio_two && du -h speck_audio_two.zip
echo "bundle: variant $V, networks:$LINES"
