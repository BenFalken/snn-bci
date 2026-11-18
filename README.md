# snn-bci

Code for spiking neural decoders (an LSNN on a CPU, a feedforward SNN on a Speck2f chip) of movement (Motor I:
Indy and Loco; Motor II: Jenkins), heard words (monkeys Neil and Gomez) and attempted speech (T12).

## 1. Environment

```bash
pip install -r requirements.txt          # Python 3.13; samna is needed only for the chip
```

Two experiments need a second environment for their reference decoders:

- Speech RNN and language model: Python 3.9 + TensorFlow, as in `speech/README.md`.
- Audio reference decoders (Wiener, Kalman, GRU, LSTM, ...): Docker, `cd audio && bash build_all_images.sh`.

Chip runs need one Speck2f devkit over USB.

## 2. Data

None is in the repository. Download it, then say where it is (step 3).

| Experiment | Source | Prepare with |
|---|---|---|
| Motor I | O'Doherty et al., [Zenodo 3854034](https://zenodo.org/record/3854034) | `motor/main/sbatch_scripts/train_all.sh` (fetches and bins) |
| Motor II | Jenkins NWB sessions *(release to be added)* | `motor/main/nwb_conversion/`, `motor/main/sbatch_scripts/train_all.sh` |
| Audio | Heelan et al. 2019 recordings *(available on request)* | `audio/prepare_auditory_data.py`, then `audio/build_word_dataset_daynorm.py --include-novel` |
| Speech | Willett et al. 2023, [Dryad](https://doi.org/10.5061/dryad.x69p8czpq) | `speech/SNNDecoder/scripts/rebuild_snndata.sbatch` |

## 3. Paths

Every script reads its locations from `paths.yaml`. Override them on your machine without editing it:

```bash
cp paths.yaml paths.local.yaml           # git-ignored; change only the roots that differ
python3 paths.py                         # lists every root and whether it exists
```

Shell scripts get the same roots as `$SNNBCI_*` from `snnbci_env.sh`.

## 4. Reproduce

| What | Command |
|---|---|
| Paper figures and tables, from stored results | `cd figures && python3 collect_four.py && python3 collect_final.py && python3 collect_speck_ops.py && python3 fig_results.py` |
| Motor: train (Slurm) | `bash motor/main/sbatch_scripts/train_all.sh` |
| Motor: evaluate all decoders, software and chip | `bash motor/final_all.sh [indy loco jenkins]` |
| Audio: train and calibrate the Speck SNN | `bash core/experiments/audio_final/run_novel.sh`, then `calibrate_novel.sh` |
| Speech: train V2+C (Slurm) | `cd speech/SNNDecoder/scripts && bash submit_r20.sh` |
| Speech: chip run and scoring | `speech_chip.py`, then `figures/final_speech_v2c.py` |
| Every Speck network on a devkit | `bash core/experiments/chip/make_deploy_bundle.sh` (a self-contained bundle) |
| Model parity check | `python3 core/tests/test_parity.py` |

Notes:

- `oscar/` is the code exactly as it trained the final networks on the cluster; use it to retrain bit for bit.
  The rest of the repository is the later code used for evaluation.
- Evaluation on another machine matches the stored results to about 0.1% (PyTorch builds differ), not digit for
  digit; the figures are built from the stored results.
