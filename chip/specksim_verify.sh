#!/bin/bash
# specksim verification: each experiment's own calibration script, unchanged, on one deployed network at its chosen
# layer-1 factor, with SNNBCI_CHIP_SIM=verify: every chip-model call also runs in SynSense's specksim (with the chip's
# reset imposed: snndec.chip.run_specksim) and stops at the first spike that differs.
#   OUT=/some/dir bash chip/specksim_verify.sh
# Result (2026-11-24, samna 0.48), all identical, factors as chosen before:
#   motor I  indy_20160426_01 x1.5 (2 trials)          Jenkins 20090912 x3 (2 trials)
#   audio    Neil FT1 s11 x3, whole validation split: 52,890 steps, 2,678,163 spikes; agreement 0.9324 as before
#   speech   V2+C x2.5, 1 sentence per session, 3 chains, every layer, subtractive reset; agreement 0.976 as before
V=${OUT:?set OUT}; mkdir -p $V; : > $V/done
MD=$SNNBCI_MOTOR_DATA/snn_datasets; SC=motor/snn_training/velocity_scalers.json
AD=$SNNBCI_AUDITORY_DATA; J=sub-Jenkins_ses-20090912_behavior+ecephys
( /usr/bin/time -p python3 -W ignore chip/audio_l1_sweep.py $AD/Neil/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline \
    $SNNBCI_SNN_RUNS/audio_fine_Neil/FT1_spikes_s11/best.pt --scales 3 --out $V/audio.json > $V/audio.log 2>&1; echo "audio exit $?" >> $V/done ) &
( /usr/bin/time -p python3 -W ignore chip/motor_l1_sweep.py $MD/hkm/jenkins_m1/mua/$J \
    ~/Downloads/motor_val_jenkins/per_session_delay_binarized_val/$J/best_model_weights.pth --scalers $SC \
    --scales 3 --trials 2 --out $V/jenkins.json > $V/jenkins.log 2>&1; echo "jenkins exit $?" >> $V/done ) &
( /usr/bin/time -p python3 -W ignore chip/motor_l1_sweep.py $MD/bmi/indy/mua/indy_20160426_01 \
    ~/Downloads/motor_val/per_session_delay_binarized_val/indy_20160426_01/best_model_weights.pth --scalers $SC \
    --scales 1.5 --trials 2 --out $V/motor1.json > $V/motor1.log 2>&1; echo "motor1 exit $?" >> $V/done ) &
( cd speech/scripts && /usr/bin/time -p python3 -W ignore speech_chip.py --ckpt ~/Downloads/speck_r20/00_V2C_1chip_3chains/best_model_weights.pth \
    --data-dir ~/Downloads/speech_rebuild/derived/snnData --mode events --partition train --max-trials 1 --scales 2.5 \
    --out $V/speech.json > $V/speech.log 2>&1; echo "speech exit $?" >> $V/done ) &
wait
