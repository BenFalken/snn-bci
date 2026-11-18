#!/bin/bash
# Packs the speech Speck SNN's chip run for the Speck laptop: $SNNBCI_WORKSPACE/speech_chip_bundle.zip with the snnDecoder
# package, speech_chip.py, the F3 checkpoint and every session's test split. Run ON THE MAC after fetching:
#
#   rsync ... $SNNBCI_WORKSPACE/speech_speck/        (snnData/<session>/test.npz and the F3 checkpoint folder)
#   bash make_speech_chip_bundle.sh
#   MODELS="G3_levels2_d2 G7_chip_multi_96ep" bash make_speech_chip_bundle.sh   (other run folders under $SRC)
#
# On the laptop (conda env speck, devkit plugged in):  unzip speech_chip_bundle.zip && cd speech_chip && bash run.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$SNNBCI_WORKSPACE/speech_speck}"
OUT="$(mktemp -d)/speech_chip"
MODELS="${MODELS:-F3_chip_soft_binary_rtr_lr01}"
DATA="${DATA:-$SRC/snnData}"              # folder of <session>/test.npz
ONE_CHIP="${ONE_CHIP:-}"                  # networks (bundle names, e.g. V2) whose chains share one chip: --one-chip
BUNDLE="${BUNDLE:-$SNNBCI_WORKSPACE/speech_chip_bundle.zip}"
mkdir -p "$OUT/scripts" "$OUT/data" "$OUT/results"
for m in $MODELS; do
  CKPT=$(find "$SRC" -path "*$m*" -name best_model_weights.pth | head -1)
  [ -n "$CKPT" ] || { echo "no $m checkpoint under $SRC"; exit 1; }
  cp "$CKPT" "$OUT/${m%%_*}.pth"; echo "${m%%_*}: $CKPT"
done
rsync -a --exclude __pycache__ --exclude configs "$HERE/../snnDecoder" "$OUT/"
cp "$HERE/speech_chip.py" "$HERE/speck_io.py" "$OUT/scripts/"
rsync -a --include='*/' --include='test.npz' --exclude='*' "$DATA/" "$OUT/data/"
for n in $ONE_CHIP; do touch "$OUT/$n.onechip"; done
cat > "$OUT/run.sh" <<'EOF'
#!/bin/bash
# The speech Speck SNN on the Speck2f devkit. Steps (each prints the phoneme error so far, session by session):
#   1. stack: the rebuilt chip network in software against the trained model, 3 trials a session (a check)
#   2. events: the chip's event-by-event firing in software, layer 1 threshold x1 to x4, 2 trials of 6 sessions;
#      pick the scale with the highest "logits vs trained r" (no labels needed)
#   3. chip: every test trial of every session on the devkit at that scale -> results/chip.json
#   bash run.sh            all three (step 3 uses SCALE, default: the best of step 2)
#   STEP=3 SCALE=2 bash run.sh
# Each network (*.pth here) is run in turn; its results land in results/<name>/.
set -eo pipefail
cd "$(dirname "$0")"
for CK in *.pth; do
NAME=${CK%.pth}; R=results/$NAME; mkdir -p $R; echo "######## $NAME"
PY="python3 scripts/speech_chip.py --ckpt $CK --data-dir data"
SESS6="t12.2022.04.28 t12.2022.05.24 t12.2022.06.14 t12.2022.07.05 t12.2022.07.27 t12.2022.08.13"
if [ "${STEP:-1}" -le 1 ]; then
  echo "== 1/3 software check"; $PY --mode stack --max-trials 3 --out $R/stack.json
fi
if [ "${STEP:-1}" -le 2 ]; then
  echo "== 2/3 chip firing in software, threshold scales"
  $PY --mode events --scales 1 1.5 2 3 4 --max-trials 2 --sessions $SESS6 --out $R/events.json
fi
S=${SCALE:-}
if [ -z "$S" ]; then
  S=$(python3 -c "import json;r=json.load(open('$R/events.json'))['results'];print(max(r,key=lambda k:r[k]['logit_corr_vs_trained']))")
fi
echo "== 3/3 on the devkit, layer 1 threshold x$S"
ONE=""; [ -f "$NAME.onechip" ] && ONE="--one-chip" && echo "   (all chains on one chip)"
$PY --mode chip $ONE --scale "$S" --out $R/chip.json --save-logits $R/chip_logits.npz --save-counts $R/chip_counts.npz
done
echo "done: copy results/ back to the Mac ($SNNBCI_WORKSPACE/speech_speck/results/)"
EOF
chmod +x "$OUT/run.sh"
cat > "$OUT/README.txt" <<'EOF'
Speech Speck SNN (T12; the network(s) *.pth here: 512 chip inputs -> 128-256-256 IAF on one Speck2f, hard reset)
on a Speck2f devkit. Needs the conda env "speck" (torch, sinabs, samna, numpy, pyyaml). Plug in the devkit, then:
    bash run.sh
A few seconds per test sentence on the chip (3 ms per 20 ms bin plus a 0.3 s reset); all sessions take one to two hours. Results land in
results/ (stack.json, events.json, chip.json): phoneme error overall and per session, chip power, ms per 20 ms bin,
energy per 80 ms output. Send results/ back.
EOF
( cd "$(dirname "$OUT")" && rm -f "$BUNDLE" && zip -qr "$BUNDLE" speech_chip )
du -h "$BUNDLE"
echo "sessions: $(ls "$OUT/data" | wc -l | tr -d ' ')  checkpoint: $CKPT"
