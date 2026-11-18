# Producing the data for every figure

The figures: **fidelity**, **efficiency** (energy, latency), **videos** (audio, speech) and the **Speck memory**
figure (how the chip works; memory from the hidden-layer readout and from input delays; the channel/delay trade-off
in all three experiments). Speech's Speck SNN is left out of every test.

`$` lines run on the laptop, `oscar$` lines on Oscar. Paths as on 2026-11-05.

---

## 0. Code and data in place

```bash
$ rsync -av --exclude __pycache__ --exclude runs ~/Documents/GitHub/snn-decoders/ \
      <oscar-user>@ssh.ccv.brown.edu:/users/$USER/snn-decoders/
# audio: the four days the sweep uses (spike counts and windows; 1.2 GB), one login; --rsync-path makes the
# destination's parent folders, which rsync itself does not
$ rsync -av --rsync-path="mkdir -p /users/$USER/scratch/audio_pipeline && rsync" \
    --include '/07_31_13/' --include '/08_03_10/' --include '/07_20_10/' --include '/08_06_13/' \
    --include '/*/pipeline/' --exclude '*_full.npy' --include '/*/pipeline/features.npy' \
    --include '/*/pipeline/window_*.npy' --exclude '*' \
    ~/Documents/GitHub/AuditoryDecoding/pipeline_data/Neil/aSTG/ \
    <oscar-user>@ssh.ccv.brown.edu:/users/$USER/scratch/audio_pipeline/
oscar$ cd /users/$USER/speechBCI-main/SNNDecoder/scripts && source ./snn_config.sh && activate_env && cd -
oscar$ $PYTHON -c "import torch, sinabs, yaml, pandas; print('env ok')"
```

---

## 1. Speck memory figure: the sweep (Oscar)

Per experiment, at the chip's input budget (motor and audio 480 inputs, speech 512), delays 40 ms apart:

| | phase A (all channels) | phase B (fewer channels, more delays, top channels by weight) |
|---|---|---|
| motor (8 indy sessions) | hid_D1, out_D1, out_D5, hid_D5 | 48×10, 32×15, 24×20, 16×30, 8×60 |
| audio (4 days) | hid_D1, out_D1, out_D5, hid_D5 | 48×10, 32×15, 24×20, 16×30, 8×60 |
| speech (19 sessions pooled, simulated chip) | hid_D1, out_D1, out_D4, hid_D4 | 64×8, 32×16, 16×32, 8×64 |

`hid` = the host reads the last hidden layer (synaptic skip); `out` = a spiking output layer on the chip. Three
seeds each: 96 + 120 motor, 48 + 60 audio, 12 + 12 speech runs. Each run trains, scores the test split, measures
the silence response (memory kernel) and checks the chip mapping.

```bash
oscar$ cd /users/$USER/snn-decoders && mkdir -p experiments/speck_memory/logs
oscar$ R=/users/$USER/scratch/speck_memory
oscar$ $PYTHON experiments/speck_memory/plan.py --out $R \
    --motor-root /users/$USER/scratch/$USER/data/snn_datasets/bmi/indy/mua_8_group \
    --scalers /users/$USER/spike_bmi_speck_delay/main/snn_training/velocity_scalers.json \
    --audio-root /users/$USER/scratch/audio_pipeline \
    --speech-data /users/$USER/scratch/speechBCI_dryad/derived/snnData
# phase A: CPU arrays for motor and audio, GPUs for speech
oscar$ sub() { sbatch --parsable "$@" experiments/speck_memory/run_array.sbatch; }
oscar$ n() { wc -l < $R/plan_$1.jsonl; }
oscar$ A1=$(sub --export=ALL,PLAN=$R/plan_motor_A.jsonl,ROOT=$R --array=0-$(($(n motor_A) - 1)))
oscar$ A2=$(sub --export=ALL,PLAN=$R/plan_audio_A.jsonl,ROOT=$R --array=0-$(($(n audio_A) - 1)))
oscar$ A3=$(sub -p gpu --gres=gpu:1 -c 8 --mem=48G --export=ALL,PLAN=$R/plan_speech_A.jsonl,ROOT=$R,PACK=2 \
              --array=0-$((($(n speech_A) + 1) / 2 - 1)))
# phase B: starts when its phase A has finished (it ranks channels with phase A's hid_D seed-0 models)
oscar$ sub --dependency=afterany:$A1 --export=ALL,PLAN=$R/plan_motor_B.jsonl,ROOT=$R --array=0-$(($(n motor_B) - 1))
oscar$ sub --dependency=afterany:$A2 --export=ALL,PLAN=$R/plan_audio_B.jsonl,ROOT=$R --array=0-$(($(n audio_B) - 1))
oscar$ sub --dependency=afterany:$A3 -p gpu --gres=gpu:1 -c 8 --mem=48G \
    --export=ALL,PLAN=$R/plan_speech_B.jsonl,ROOT=$R,PACK=2 --array=0-$((($(n speech_B) + 1) / 2 - 1))
```

Progress, and resubmitting (finished runs are skipped):

```bash
oscar$ R=$R python3 -c "
import json, os
R = os.environ['R']
for e in ('motor', 'audio', 'speech'):
    for p in 'AB':
        rows = [json.loads(l) for l in open(f'{R}/plan_{e}_{p}.jsonl')]
        done = sum(os.path.isfile(f\"{R}/{r['out']}/result.json\") for r in rows)
        print(f'{e:7s} {p}: {done:3d} / {len(rows)} done')"
oscar$ grep -l -E "Error|Traceback" experiments/speck_memory/logs/*.out | head
```

When all are done:

```bash
oscar$ $PYTHON experiments/speck_memory/collect.py --root $R --out $R/results.csv --memory $R/memory.csv
$ scp <oscar-user>@ssh.ccv.brown.edu:/users/$USER/scratch/speck_memory/{results,memory}.csv ~/Downloads/paper_figures/data/
```

Rough cost: motor ~1 h per run, audio ~2–3 h (10 chip steps per frame), speech ~2–3 h per run on a GPU; the per-user
CPU limit sets the pace (about a day for everything).

**Note on audio.** The audio benchmark's target is the centre frame of a 17-frame window and the decoders answer at
the window's last frame, so every audio decoder must remember 8 frames (320 ms) back to see the activity at the
target. Splits whose delays span less than that (96×5: 160 ms) are expected to fail for that reason, not for lack of
channels.

---

## 2. Models still to train elsewhere

| what | where | status / command |
|---|---|---|
| audio Speck (binB), 3 seeds × 4 days | laptop | running (`speck_audio/sweep_bin.py <day> B`); results in `speck_audio/r13B3_<day>.txt` |
| audio Speck (binB) on 07_19_11, 3 seeds | laptop or Oscar | `python -m snndec.train configs/audio_speck.yaml --data .../07_19_11/pipeline --out runs/audio_speck_07_19_11_s$S --set train.seed=$S` for S in 0 1 2 |
| motor ladder: last ladder_filters_soft session | Oscar | running; `bash sbatch_scripts/speck_all_progress.sh` |
| motor loco: per-session LSNN (9 sessions; the ladder was indy only) | Oscar | `cd /users/$USER/spike_bmi_speck_delay/main/sbatch_scripts && sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=loco,SPECK_MODEL=lsnn_ladder_adapt --array=0-8 run_snn_per_session_array.sbatch` |
| motor hkm (jenkins) LSNN fine-tunes | Oscar | pretraining finishing, fine-tunes queued (only if hkm is in the figures) |
| speech LSNN, speech baselines | — | done (Oscar `figures/inference_final`) |

---

## 3. Fidelity

**Motor** — every decoder on the Speck laptop (Speck on the chip), per session, then the report. The main per-session
LSNN and the Speck SNN with input delays (the fairness decision: both trained per session, as the baselines):

```bash
$ cd ~/Downloads/spike_bmi_lsnn/main/sbatch_scripts
$ CKPT_DIR=per_session_delay_binarized bash fetch_for_speck.sh indy loco
$ CKPT_DIR=per_session_lsnn_ladder_adapt_binarized ANN=0 bash fetch_for_speck.sh indy loco
$ D=../../data/snn_checkpoints/bmi
$ for s in indy loco; do
    SNN_CHECKPOINT_ROOT=$D/$s/per_session_lsnn_ladder_adapt_binarized \
    SPECK_CHECKPOINT_ROOT=$D/$s/per_session_delay_binarized bash run_inference.sbatch --local bmi $s; done
$ rsync -a ../../data/results/test_all_decoders/bmi/ ~/Downloads/results/final/     # combined_metrics.json etc.
```
(For the LOSO-pretrained LSNN instead, leave `SNN_CHECKPOINT_ROOT` at its default.)

**Audio** — the baselines and the LSNN are in the pipeline already; add the Speck SNN (the seed-1337 checkpoints, as
the pipeline's other decoders; the 3-seed spread separately):

```bash
$ S=~/Documents/GitHub/snn-decoders/runs/speck_audio_binB
$ cd ~/Documents/GitHub/snn-decoders
$ python experiments/figure_data/audio_speck_results.py --audio-root ~/Documents/GitHub/AuditoryDecoding/pipeline_data/Neil/aSTG \
    --ckpt $(for d in 07_31_13 08_03_10 07_20_10 08_06_13; do printf "$d=$S/${d}_r13_binB_s1337/model.pt "; done) \
           07_19_11=runs/audio_speck_07_19_11_s0/best.pt \
    --seeds $(for d in 07_31_13 08_03_10 07_20_10 08_06_13; do printf "$d=$S/${d}_r13_binB_s1337/model.pt,$S/${d}_r13_binB_s11/model.pt,$S/${d}_r13_binB_s99/model.pt "; done)
```
(The binB runs were copied out of the temporary scratch folder into `runs/speck_audio_binB/`.)

**Speech** — `~/Downloads/speechBCI-main/figures/inference_final/decoderComparison_final.json`, done.

Then `python ~/Downloads/paper_figures/collect_fidelity.py`.

---

## 4. Efficiency (energy, latency)

- motor: from step 3's laptop run (`efficiency_summary.json`: RAPL on the laptop, the chip's power monitor for Speck)
- speech: `figures/inference_final/efficiency_final.json`, done
- audio: computed by `collect_efficiency.py` from the pipeline's models

Then `python ~/Downloads/paper_figures/collect_efficiency.py`.

---

## 5. Videos

- audio: `python ~/Downloads/paper_figures/video_audio.py` (reads `<decoder>_predict_valid.npy`; Speck's appear after
  step 3)
- speech: the final model's trial exports from Oscar, then the video script:

```bash
oscar$ find /users/$USER/speechBCI-main -name 'trial_t12.2022.06.02_test_0.json' -newer \
    /users/$USER/speechBCI-main/SNNDecoder/scripts/configs/nodelay96.txt
$ rsync -av <oscar-user>@ssh.ccv.brown.edu:<the folder found>/ ~/Downloads/viewer_trials_final/
```

---

## Code still to write at the figure stage

- `paper_figures/fig_speck_memory.py`: the chip schematic, memory kernels, readout × delay ablation, channel/delay
  trade-off (from `results.csv`, `memory.csv`)
- `collect_fidelity.py` / `collect_efficiency.py`: add the audio Speck SNN ('speck_snn')
- `video_audio.py`: a Speck lane; the speech video in the paper style
- the same-laptop latency/power harness (snndec streaming `step()`), if every decoder is to be timed on one machine
