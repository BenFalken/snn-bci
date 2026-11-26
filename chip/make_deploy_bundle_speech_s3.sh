#!/bin/bash
# Builds $SNNBCI_WORKSPACE/speck_deploy_s3.zip: round 21's S3 speech network (V2+C's three networks on one chip, with
# soft reset and a membrane floor of -3) on the Speck2f devkit, its readout refitted to the chip model's firing applied
# live. Exploratory: nothing of the final results is touched. Run ON THE MAC:
#   bash chip/make_deploy_bundle_speech_s3.sh
# On the Speck laptop (conda env "speck"), devkit plugged in:
#   unzip speck_deploy_s3.zip && cd speck_deploy_s3 && bash run.sh check && bash run.sh quick && bash run.sh full
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DL="$SNNBCI_WORKSPACE"
SB="$SNNBCI_SPEECH_SNN"
CK="$DL/speck_r21/02_S3_soft_floor3/best_model_weights.pth"
RO="$DL/speck_r21/S3_results/W1/readout_refit.pt"
for f in "$CK" "$RO"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
T="$(mktemp -d)/speck_deploy_s3"
mkdir -p "$T"/{code,models,data,results}

mkdir -p "$T/code/snn-decoders" "$T/code/speech/scripts"
rsync -a --exclude __pycache__ "$REPO/snndec" "$T/code/snn-decoders/"
rsync -a --exclude __pycache__ --exclude configs "$SB/snnDecoder" "$T/code/speech/"
cp "$SB/scripts/speech_chip.py" "$SB/scripts/speck_io.py" "$SB/scripts/refit_readout.py" "$T/code/speech/scripts/"
cp "$CK" "$T/models/S3.pth"
cp "$RO" "$T/models/S3_readout_refit.pt"
rsync -a --include='*/' --include='test.npz' --exclude='*' "$DL/speech_rebuild/derived/snnData/" "$T/data/"

cat > "$T/run.sh" <<'EOF'
#!/bin/bash
# Round 21 S3 on one Speck2f: soft reset, floor -3, layer 1 threshold x1.5 (chosen without labels on training
# sentences), refit readout live; only the used neurons reset, 2 s settling before each sentence (as V2+C).
#   bash run.sh check    packages and the devkit
#   bash run.sh quick    2 test sentences per session, 38 in all (~10 min): chip model PER 0.4432, logit r 0.990
#   bash run.sh full     all 680 test sentences (~1.5 h)  -> results/
# Chip model (Mac, same sentences): PER 0.3994 on 680 (trained readout 0.4131); V2+C on this chip was 0.4396.
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD/code/snn-decoders:$PWD/code/speech:${PYTHONPATH:-}"
what="${1:-check}"
if [ "$what" = check ]; then
  $PY -c "import torch, sinabs, numpy; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
  $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
  exit 0
fi
MAX=(); [ "$what" = quick ] && MAX=(--max-trials 2)
$PY code/speech/scripts/speech_chip.py --ckpt models/S3.pth --data-dir data --mode chip \
    --one-chip --reset-used --reset-settle 2 --scale 1.5 --readout models/S3_readout_refit.pt "${MAX[@]}" \
    --out results/chip_$what.json --save-logits results/chip_logits_$what.npz --save-counts results/chip_counts_$what.npz
echo "done: send back the results/ folder"
EOF
chmod +x "$T/run.sh"
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy_s3.zip" && zip -qr "$DL/speck_deploy_s3.zip" speck_deploy_s3 )
echo "$T"
du -h "$DL/speck_deploy_s3.zip"
