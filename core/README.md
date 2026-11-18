# snn-decoders

One codebase for the spiking decoders of the three experiments: the **LSNN** and the **Speck-compatible SNN** for
motor (spike_bmi: hand velocity), audio (AuditoryDecoding: heard-speech spectrograms) and speech (speechBCI
SNNDecoder: attempted-speech phonemes). Data is read as each project already preprocessed it; this package does the
models, training, and inference.

## One model

Every decoder is `snndec.SNNDecoder(spec)`: host input → spiking layers → host readout.

| | LSNN | Speck SNN |
|---|---|---|
| spiking layers | recurrent, synaptic filters, 90% adaptive thresholds, soft reset, ≤ 3 spikes/step | feedforward integrate-and-fire, hard reset — what maps onto Speck2f (checked: `model.check_chip()`) |
| host input | optional channel subset, dropout, per-day gain | + input delays (the chip's only memory), event encoding (clip, threshold, or counts spread over sub-steps), input scale |
| host readout | leaky output neurons | leaky output neurons (speech) or smoothed spikes → linear / population decode (audio, motor) |

The neuron (`neurons.py`) is the same in both; a Speck network is the LSNN layer with recurrence, filters and
adaptation off and hard reset. Time constants are in milliseconds; each experiment sets its step (motor 4 ms, speech
20 ms, audio 40 ms).

## Layout

```
snndec/
  neurons.py    the spiking layer
  frontend.py   host input: channels, day gain, encoding, sub-steps, delays
  readouts.py   leaky output neurons; EMA + linear / affine / population heads
  model.py      SNNDecoder (forward over a sequence; step() for streaming; chip mapping check)
  convert.py    the three projects' checkpoints -> SNNDecoder, unchanged; load() reads any kind
  data.py       the projects' preprocessed data -> trials
  tasks.py      losses and metrics: motor RMSE, audio correlation, speech CTC + PER
  train.py      one trainer, each experiment's recipe in its config
  evaluate.py   any checkpoint on its data, with its experiment's metric
configs/        motor_lsnn, motor_speck, audio_lsnn, audio_speck, speech_lsnn, speech_speck (PROVISIONAL)
tests/          test_parity.py (local), check_speech_checkpoints.py (where the speech checkpoints are)
```

## Use

```bash
# evaluate any checkpoint, original or unified
python -m snndec.evaluate motor  CKPT --data SESSION_DIR --scalers velocity_scalers.json
python -m snndec.evaluate audio  CKPT --data PIPELINE_DIR
python -m snndec.evaluate speech CKPT --data SNNDATA_DIR

# train (each experiment's own recipe; --set overrides any config value)
python -m snndec.train configs/motor_lsnn.yaml --data SESSION_DIR --scalers velocity_scalers.json --out runs/x
python -m snndec.train configs/motor_lsnn.yaml --data S1 S2 ... --scalers ... --out runs/pool --set train.epochs=10
python -m snndec.train configs/motor_lsnn.yaml --data S1 --scalers ... --out runs/ft --init runs/pool/best.pt
python -m snndec.train configs/audio_speck.yaml --data PIPELINE_DIR --out runs/y
python -m snndec.train configs/speech_lsnn.yaml --data SNNDATA_DIR --out runs/z

# convert an original checkpoint to the unified format
python -c "import snndec; m, info = snndec.load('orig.pth'); snndec.save(m, 'unified.pt', info)"
```

## Verified (2026-11-04)

`python tests/test_parity.py`: 18/18 checks, outputs identical to the original code (max difference 0):
motor LSNN and Speck (real checkpoints), audio LSNN and Speck binB (real checkpoints), the final speech LSNN and two
speech Speck designs (original code, random weights); streaming `step()` equals `forward()` for all; every Speck
network maps onto Speck2f.

`snndec.evaluate` on real data reproduces the reported results: motor LSNN 32.3216 and Speck 35.5453 test RMSE
(indy_20160411_01); audio LSNN r = 0.8363 / 0.8195 / 0.7759 and Speck binB 0.7701 / 0.7827 / 0.7500 (07_31_13),
LSNN predictions bit-identical to the pipeline's saved ones.

Training: each recipe runs end to end (motor and audio on real data, speech on a synthetic set); motor's first two
epochs track the original runs' within seed noise.

## Recipes (configs)

| | motor | audio | speech |
|---|---|---|---|
| batches | whole trials, 20 | LSNN: windows, 64; Speck: 150-step stretches, 16 | one session's trials, 64 |
| optimizer | AdamW 1e-3, wd 1e-3 | Adam 1e-3 | Adam eps 0.1, L2 1e-5, lr 0.02 → 0 linear; τ_syn lr × 0.1 |
| schedule / stop | ÷10 on plateau; early stop 10 | early stop 10 (best restored) | 96 epochs (Speck screens 36) |
| loss | RMSE (physical units) | MSE | CTC (+ spike-rate penalty 30 × Σ max(0, rate − 0.4)) |
| selected on | test (as spike_bmi) | valid | test |
| augmentation | — | LSNN: noise 1.0, offset 0.2 | LSNN: noise 1.0, offset 0.2; Speck: offset only |

Differences from the originals, all deliberate: the audio Speck input scale is computed in double precision (the
original's float32 sum differs by ≤ 0.08%); the original options the final models never used (learned delays,
per-synapse delays, input plans, SE-adLIF, ...) are refused by the converters, not carried over.
