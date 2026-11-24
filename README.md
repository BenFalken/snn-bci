# snn-bci

Spiking neural decoders of movement (Motor I: monkeys Indy and Loco; Motor II: Jenkins), heard words (monkeys Neil and
Gomez) and attempted speech (participant T12): an LSNN (recurrent, adaptive) on a CPU, and a feedforward SNN on one
Speck2f neuromorphic chip.

## Layout

```
snndec/      the shared model: LSNN and Speck SNN in one class, their configs, training, checkpoint loaders
chip/        running the Speck SNNs on the chip: threshold calibration, readout refits, deploy bundle, benchmark
motor/       Motor I and II: datasets, training, evaluation of every decoder (incl. the chip)
audio/       heard words: cohorts from the raw recordings, label review (labels/), Speck SNN training (speck/)
speech/      attempted speech: the speech SNN package (snnDecoder/), training and chip scripts, scoring (analysis/)
figures/     every figure, table and video of the paper, built from stored results
tests/       parity: every original checkpoint gives the same outputs through snndec
external/    third-party code, with its own license: Willett et al.'s speech RNN; Brown's mikko toolkit (audio
             reference decoders in Docker, with our LSNN module added)
```

Each experiment keeps the training code its final networks were trained with (motor/snn_training, speech/snnDecoder,
snndec for the audio Speck SNN, external/mikko for the audio LSNN); `snndec` loads all of their checkpoints, and
`tests/test_parity.py` checks that it reproduces their outputs exactly.

## 1. Environment

```bash
pip install -r requirements.txt               # Python 3.13; samna only to run the chip
bash external/mikko/build_all_images.sh       # audio reference decoders (Docker)
```

The speech reference RNN needs Python 3.9 + TensorFlow (`external/willett_speechBCI/README.md`). Chip runs need one
Speck2f devkit over USB.

## 2. Data

None is in the repository.

| Experiment | Source | Prepare with |
|---|---|---|
| Motor I | O'Doherty et al., [Zenodo 3854034](https://zenodo.org/record/3854034) | `motor/sbatch_scripts/train_all.sh` (fetches and bins) |
| Motor II | Jenkins NWB sessions *(release to be added)* | `motor/nwb_conversion/`, then `train_all.sh` |
| Audio | Heelan et al. 2019 recordings *(available on request)* | `audio/prepare_auditory_data.py`, `audio/build_word_dataset_daynorm.py --include-novel` |
| Speech | Willett et al. 2023, [Dryad](https://doi.org/10.5061/dryad.x69p8czpq) | `speech/scripts/rebuild_snndata.sbatch` |

## 3. Paths

Every script reads its locations from `paths.yaml`; override them without editing it:

```bash
cp paths.yaml paths.local.yaml                # git-ignored; change only the roots that differ
python3 paths.py                              # lists every root and whether it exists
```

Shell scripts get the same roots as `$SNNBCI_*` from `snnbci_env.sh`. On a cluster the Oscar roots follow `$USER`; to
fetch from one, set `oscar_user` in `paths.local.yaml`.

## 4. Reproduce

| What | Command |
|---|---|
| Paper figures and tables, from stored results | `cd figures && python3 collect_four.py && python3 collect_final.py && python3 collect_speck_ops.py && python3 fig_results.py` |
| Motor: train every decoder (Slurm) | `bash motor/sbatch_scripts/train_all.sh` |
| Motor: evaluate, software and chip | `bash motor/final_all.sh indy loco` and `bash motor/jenkins_m1_final.sh` |
| Audio: reference decoders, LSNN and Speck SNN; calibration | `bash audio/speck/run_novel.sh`, then `bash audio/speck/calibrate_novel.sh` |
| Speech: LSNN (Slurm) | `speech/scripts/configs/final_lsnn.txt` (command in its header) |
| Speech: Speck SNN V2+C (Slurm) | `cd speech/scripts && bash submit_r20.sh` |
| Speech: chip run, readout refit, scoring | `speech/scripts/speech_chip.py`, `refit_readout.py`, then `figures/final_speech_v2c.py` |
| Speech: comparison with the paper RNN | `speech/analysis/run_compare.sbatch` (`LM_DIR=none`: no language model needed) |
| Every Speck network on a devkit | `bash chip/make_deploy_bundle.sh` (a self-contained bundle) |
| Parity check | `python3 tests/test_parity.py` |

Notes:

- The code exactly as it trained the final networks on the cluster, and every exploratory round, are in the git tag
  `as-assembled` (`git checkout as-assembled -- oscar/`).
- Evaluation on another machine matches the stored results to about 0.1% (PyTorch builds differ), not digit for
  digit; the figures are built from the stored results.

## 5. The chip model: SynSense's specksim

The chip model that chooses both calibrations is SynSense's simulator, specksim (samna 0.48, through sinabs), run on
the network exactly as quantized for the devkit, with the chip's reset imposed after each input event (specksim's
neuron only subtracts its threshold and fires at most once per event): `snndec.chip.run_specksim`. The sweeps and
refits ran a NumPy version (`snndec.chip.run_events`) that gives identical spikes. Choose the engine with

    SNNBCI_CHIP_SIM=numpy      # default, ~25x faster as specksim is called from Python
    SNNBCI_CHIP_SIM=specksim   # every chip-model call in specksim
    SNNBCI_CHIP_SIM=verify     # both on every call; stops at the first spike that differs

`OUT=dir bash chip/specksim_verify.sh` runs each experiment's own calibration script in verify mode on a deployed
network (motor I, monkey J, the whole audio validation split, every speech session): all identical.
`chip/specksim_compare.py` and `chip/specksim_hard.py` show specksim as shipped (+18% spikes) and with the reset.
