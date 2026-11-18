#!/bin/bash
# After round 16: builds the one final laptop batch, every experiment's Speck network on the chip plus the energy
# bench, as ~/Downloads/speck_final_all.zip. Run ON THE MAC once the winner (lowest final training loss of R1 to R4)
# is fetched to ~/Downloads/speech_speck/<NN>_<name>/best_model_weights.pth:
#
#   rsync -av $USER@ssh.ccv.brown.edu:/users/$USER/scratch/speechBCI_dryad/derived/snns/speck_r16/<NN>_<name> ~/Downloads/speech_speck/
#   bash finish_r16.sh <NN>_<name>
#
# 1. speech's layer 1 threshold factor, without labels, on TRAINING sentences (3 per session, 6 sessions): the scale
#    whose event-by-event logits agree best with the trained network's
# 2. the speech chip bundle for that network (make_speech_chip_bundle.sh)
# 3. the batch: motor (final_all.sh with every session's factor from training trials), audio (the hard-reset 48 x 10
#    network, 3 seeds, x4 chosen on validation), speech (step 1's factor), then final_bench.py with the final
#    speech network's host model
#
# On the laptop: unzip speck_final_all.zip next to speck_per_session/, then  cd speck_final_all && bash run_all.sh
set -euo pipefail
W="${1:?usage: bash finish_r16.sh <NN>_<name>, e.g. 01_R2_hard_txpow_levels_nodelay}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$HOME/Downloads/speech_speck}"
SND="${SND:-$HOME/Documents/GitHub/snn-decoders}"
SPS="${SPS:-$HOME/Downloads/speck_per_session}"
B2="${B2:-$HOME/Downloads/speck_final_batch2}"
RID=$(echo "$W" | grep -o 'R[1-4]' | head -1 || true); RID=${RID:-R1}
CK="$SRC/$W/best_model_weights.pth"
[ -f "$CK" ] || { echo "no $CK: fetch it first"; exit 1; }
R="$SRC/results/$W"; mkdir -p "$R"

echo "== 1/3 speech layer 1 factor on training sentences ($W, host model $RID)"
python3 -W ignore "$HERE/speech_chip.py" --ckpt "$CK" --data-dir "$SRC/snnData" --mode events --partition train \
    --scales 1 1.5 2 3 4 --max-trials 3 \
    --sessions t12.2022.04.28 t12.2022.05.24 t12.2022.06.14 t12.2022.07.05 t12.2022.07.27 t12.2022.08.13 \
    --out "$R/events_train.json"
S=$(python3 -c "import json;r=json.load(open('$R/events_train.json'))['results'];print(max(r,key=lambda k:r[k]['logit_corr_vs_trained']))")
echo "chosen: x$S"

echo "== 2/3 speech chip bundle"
MODELS="$W" bash "$HERE/make_speech_chip_bundle.sh"

echo "== 3/3 the batch"
OUT="$(mktemp -d)/speck_final_all"
mkdir -p "$OUT/code" "$OUT/audio" "$OUT/motor" "$OUT/results"
( cd "$OUT" && unzip -q ~/Downloads/speech_chip_bundle.zip )
rsync -a --exclude __pycache__ "$SND/snndec" "$SND/configs" "$OUT/code/"
mkdir -p "$OUT/code/experiments"; rsync -a --exclude __pycache__ "$SND/experiments/chip" "$OUT/code/experiments/"
for d in "$B2"/audio/words_old_*/; do cp -r "${d%/}" "$OUT/audio/"; done
cp "$SPS/final_all.sh" "$SPS/speck_l1_factors.txt" "$SPS/jenkins_chip_final.sh" "$OUT/motor/"
cp "$R/events_train.json" "$OUT/results/speech_events_train.json"
cat > "$OUT/run_all.sh" <<EOF
#!/bin/bash
# The final laptop batch, devkit plugged in. Every Speck network is hard reset, on one Speck2f, with its layer 1
# threshold factor chosen without labels on data the test never sees.
#   bash run_all.sh check | motor | audio | speech | bench     (no argument: all, in that order; several hours)
# Send back: results/, speech_chip/results/, and ../speck_per_session/results/final_v2/
set -eo pipefail
cd "\$(dirname "\$0")"
PY="\${PYTHON:-python3}"
SPS="\${SPS:-\$(cd .. && pwd)/speck_per_session}"
STEPS=("\$@"); [ \${#STEPS[@]} -eq 0 ] && STEPS=(check motor audio speech bench)
for st in "\${STEPS[@]}"; do case \$st in
check)
    \$PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
    \$PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
    [ -d "\$SPS" ] || { echo "no \$SPS (set SPS=/path/to/speck_per_session)"; exit 1; } ;;
motor)   # indy, loco and jenkins: every decoder, Speck at each session's factor
    cp motor/* "\$SPS/" && bash "\$SPS/final_all.sh" ;;
audio)   # the hard-reset 48 x 10 network, 3 seeds, layer 1 x4; every prediction saved
    ARGS=(); for d in audio/*/; do ARGS+=("\${d}pipeline=\${d}model.pt"); done
    \$PY -W ignore code/experiments/chip/audio_chip.py --data "\${ARGS[@]}" --out results/audio_chip_final.json \\
        --layer1-scale 4 --save-preds results/audio_preds ;;
speech)  # $W, every test sentence of the 19 sessions, layer 1 x$S; logits saved for the language model
    (cd speech_chip && STEP=3 SCALE=$S bash run.sh) ;;
bench)   # latency and RAPL energy for every decoder, idle subtracted
    SPEECH_SPECK=$RID \$PY -W ignore code/experiments/chip/final_bench.py --out results/final_bench.json ;;
*) echo "unknown step \$st"; exit 1 ;;
esac; done
echo "Done: send back results/, speech_chip/results/ and \$SPS/results/final_v2/"
EOF
chmod +x "$OUT/run_all.sh"
( cd "$(dirname "$OUT")" && rm -f ~/Downloads/speck_final_all.zip && zip -qr ~/Downloads/speck_final_all.zip speck_final_all )
du -h ~/Downloads/speck_final_all.zip
echo "speech: $W at x$S (host model $RID)"
