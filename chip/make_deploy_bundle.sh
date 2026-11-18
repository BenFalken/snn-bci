#!/bin/bash
# Builds $SNNBCI_WORKSPACE/speck_deploy.zip: the FINAL Speck SNNs, run end to end on a Speck2f devkit with their final
# readouts applied live (no post-processing step). Run ON THE MAC (it gathers checkpoints, readouts and data):
#   bash chip/make_deploy_bundle.sh
# On the Speck laptop (conda env "speck": torch, sinabs, samna, numpy, pyyaml), devkit plugged in:
#   unzip speck_deploy.zip && cd speck_deploy && bash run.sh check && bash run.sh all
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DL="$SNNBCI_WORKSPACE"
SB="$SNNBCI_SPEECH_SNN"
T="$(mktemp -d)/speck_deploy"
mkdir -p "$T"/{code,models,data,results}

# ---- code: snn-decoders (audio, motor, the host benchmark) and the speech decoder package
mkdir -p "$T/code/snn-decoders"
rsync -a --exclude __pycache__ "$REPO/snndec" "$T/code/snn-decoders/"
rsync -a --exclude __pycache__ "$REPO/chip" "$T/code/snn-decoders/"
mkdir -p "$T/code/speech"
rsync -a --exclude __pycache__ --exclude configs "$SB/snnDecoder" "$T/code/speech/"
mkdir -p "$T/code/speech/scripts"
cp "$SB/scripts/speech_chip.py" "$SB/scripts/speck_io.py" "$SB/scripts/refit_readout.py" "$T/code/speech/scripts/"

# ---- speech: round 20 V2+C (three networks on one chip, all nine cores) and its readout refitted to the chip
mkdir -p "$T/models/speech" "$T/data/speech"
cp "$DL/speck_r20/best_model_weights.pth" "$T/models/speech/V2C.pth"
cp "$DL/speck_r20/results/W1/readout_refit.pt" "$T/models/speech/V2C_readout_refit.pt"
rsync -a --include='*/' --include='test.npz' --exclude='*' "$DL/speech_rebuild/derived/snnData/" "$T/data/speech/"

# ---- audio: P5 on the expanded cohorts, monkeys N (Neil) and G (Gomez), 3 seeds each, each with its refitted
# readout; per network only what the run reads (the per-frame chip input, the test targets, the split lengths)
AD="$SNNBCI_AUDITORY_DATA"
for m in Neil Gomez; do for s in s11 s1337 s99; do
  D="$T/models/audio/${m}_P5_$s"; P="$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline"; mkdir -p "$D/pipeline"
  cp "$SNNBCI_SNN_RUNS/audio_novel_$m/P5_centre_ema1_deriv_$s/best.pt" "$D/model.pt"
  cp "$DL/final_results/audio_novel/${m}_P5_$s/refit/readout_event.npz" "$D/readout_refit.npz"
  cp "$P/features_counts_daynorm.npy" "$P/window_targets_test.npy" "$D/pipeline/"
  python3 -c "
import numpy as np, json
P='$P'; n={k: len(np.load(f'{P}/window_targets_{k}.npy', mmap_mode='r')) for k in ('train', 'valid', 'test')}
json.dump({'n': n, 'W': int(np.load(f'{P}/window_features_train.npy', mmap_mode='r').shape[1])}, open('$D/pipeline/lengths.json', 'w'))"
done; done

# ---- motor (II): Jenkins, M1 array, 4 sessions, each with its readout refitted in software
cp "$DL/speck_per_session/main/snn_training/velocity_scalers.json" "$T/models/"
for s in 20090912 20090916 20090918 20090923; do
  S=sub-Jenkins_ses-${s}_behavior+ecephys
  mkdir -p "$T/models/jenkins_m1/$S" "$T/data/hkm/jenkins_m1/$S"
  cp "$DL/speck_per_session/data/snn_checkpoints/hkm/jenkins_m1/per_session_delay_binarized/$S/best_model_weights.pth" \
     "$T/models/jenkins_m1/$S/"
  cp "$DL/final_results/jenkins_refit/${S}_readout.npz" "$T/models/jenkins_m1/$S/readout_refit.npz"
  rsync -a "$DL/speck_per_session/data/snn_datasets/hkm/jenkins_m1/mua/$S/test" "$T/data/hkm/jenkins_m1/$S/"
done

cat > "$T/run.sh" <<'EOF'
#!/bin/bash
# The final Speck SNNs on the Speck2f devkit, each decoded with its final readout LIVE (no post-processing):
#   bash run.sh check     packages and the devkit
#   bash run.sh speech    V2+C on one chip, all 680 test sentences (~1.5 h)       -> results/speech/
#   bash run.sh audio     P5, monkeys N and G x 3 seeds, held-out words (~35 min)  -> results/audio/
#   bash run.sh jenkins   Motor (II), 4 sessions, their test trials (~30 min)     -> results/jenkins/
#   bash run.sh bench     the host's share of the speech decoder, timed under RAPL (no devkit needed, ~2 min)
#   bash run.sh all       all of the above
# Settings are the deployed ones: layer 1's threshold factors chosen without labels on training data; readouts
# refitted to the chip's firing on training data; 3 ms per chip step (motor, audio); speech resets only the used
# neurons and waits 2 s before each sentence (its networks sit on 64K-neuron cores).
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
SD=code/snn-decoders
what="${1:-all}"
if [ "$what" = check ]; then
  $PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
  $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
  exit 0
fi
if [ "$what" = speech ] || [ "$what" = all ]; then
  mkdir -p results/speech
  $PY code/speech/scripts/speech_chip.py --ckpt models/speech/V2C.pth --data-dir data/speech --mode chip \
      --one-chip --reset-used --reset-settle 2 --scale 2.5 --readout models/speech/V2C_readout_refit.pt \
      --out results/speech/chip.json --save-logits results/speech/chip_logits.npz \
      --save-counts results/speech/chip_counts.npz
fi
if [ "$what" = audio ] || [ "$what" = all ]; then
  mkdir -p results/audio
  for m in Neil Gomez; do for s in s11 s1337 s99; do       # layer 1 x4 for all six (label-free, validation)
    d=models/audio/${m}_P5_$s
    $PY -W ignore $SD/chip/audio_chip.py --data "$d/pipeline=$d/model.pt=$d/readout_refit.npz" \
        --layer1-scale 4 --out results/audio/${m}_P5_$s.json --save-preds results/audio/preds_${m}_$s
  done; done
fi
if [ "$what" = jenkins ] || [ "$what" = all ]; then
  mkdir -p results/jenkins
  for sf in 20090912:3 20090916:3 20090918:3 20090923:2; do
    S=sub-Jenkins_ses-${sf%:*}_behavior+ecephys
    $PY -W ignore $SD/chip/motor_chip.py --session data/hkm/jenkins_m1/$S \
        --ckpt models/jenkins_m1/$S/best_model_weights.pth --scalers models/velocity_scalers.json --scale ${sf#*:} \
        --readout models/jenkins_m1/$S/readout_refit.npz --out results/jenkins/$S.json
  done
fi
if [ "$what" = bench ] || [ "$what" = all ]; then
  mkdir -p results/bench
  # energy needs RAPL readable: sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj
  SPEECH_SPECK=V2C $PY -W ignore $SD/chip/final_bench.py --only speech/speck_host \
      --out results/bench/speech_V2C_host.json
fi
echo "done: send back the results/ folder"
EOF
chmod +x "$T/run.sh"
cat > "$T/README.txt" <<'EOF'
The final Speck SNNs, ready to run on a Speck2f devkit (see run.sh for every option):

  speech    round 20 V2+C: all 256 features x 4 delay taps (0, 80, 260, 500 ms) as two 128-256-64 networks, plus the
            top 64 features at 0, 300, 600 and 1200 ms as a 64-256-64 network, all three on one chip (cores 5-3-0,
            6-4-1, 2-7-8); layer 1 threshold x2.5; host readout of every layer with traces, refitted to the chip's
            firing on training sentences; greedy CTC phonemes every 80 ms
  audio     P5 for two monkeys (N, G; 20 words + 3 novel words): 48 channels x 9 taps around the target frame,
            128-256-256; layer 1 x4 for all six networks; Hidden 3 smoothed once and mapped to 128 mel bands by a
            readout refitted on the training stream
  jenkins   Motor (II), M1 array: 96 channels x 4 taps, 128-256-256 + 36 output neurons; layer 1 x3/x3/x3/x2;
            velocity from the output neurons by a readout refitted on training trials
  Motor (I) (Indy, Loco) is the speck_per_session pipeline already on this laptop (run_final.sh): its trained
  population readout is used as is (refitting does not improve it).

Every calibration was made on training data in software (the chip model, snndec.chip.run_events); the chip's
weights are those trained. Each script prints the phoneme error / correlation / RMSE with the final readout; the
trained readout's score is reported beside it for reference.
EOF
( cd "$(dirname "$T")" && rm -f "$DL/speck_deploy.zip" && zip -qr "$DL/speck_deploy.zip" speck_deploy )
du -h "$DL/speck_deploy.zip"
echo "bundle: $T"
