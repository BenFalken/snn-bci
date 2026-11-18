Speck per-session deployment
============================

The Speck SNN with input delays, trained on each session alone on Oscar (SPECK_MODEL=speck_delay,
per_session_delay_binarized), run on every indy and loco session in PyTorch and on the Speck2f devkit.

The network: 96 channels, each also sent 40, 100 and 200 ms late by the host (384 chip inputs)
-> 128 -> 256 -> 256 -> 36 spiking outputs (IAF, hard reset), decoded on the host by the two-stage
EMA and population vector. About 157 k weights on 4 of the chip's cores.

Run (from this folder, in the Speck environment, devkit plugged in):
    bash run_speck.sh check          packages and devkit
    bash run_speck.sh                both monkeys, PyTorch and chip (about 1-2 hours)
    bash run_speck.sh chip indy      one monkey
    bash run_speck.sh nochip         PyTorch only, no devkit
An interrupted run resumes: finished sessions are skipped. The chip waits 3 ms per step
(SPECK_WAIT_TIME=0.003), the setting that halved the chip's loss in the mini test.

Results: results/chip/<subject>/, and a summary per monkey at the end
    combined_metrics.json        per session: RMSE, CC, R2 of 'snn' (PyTorch) and 'speck' (chip)
    efficiency_summary.json      chip latency, power and energy; PyTorch latency and RAPL energy
    decoder_*.png                report figures
    speck_diagnosis.json         PyTorch vs 8-bit quantized vs chip, per session
    speck_layer_activity_*.png   layer-by-layer activity, PyTorch vs chip

Layout (as on Oscar, so run_inference.sbatch finds everything):
    main/                        spike_bmi code
    data/dataset/bmi/<subject>/mua/<session>_binning.h5                    test rows only
    data/snn_datasets/bmi/<subject>/mua_8_group/<session>/test/            SNN test trial
    data/snn_checkpoints/bmi/<subject>/per_session_delay_binarized/<session>/best_model_weights.pth

Final comparison: every decoder (run_final.sh)
    Kalman filter, Wiener filter, LSTM, QRNN and the LSNN on this laptop's CPU, and the Speck SNN on the
    chip, on every session: accuracy, latency and measured energy (RAPL; the chip's power monitor).
    Needs, besides the above, from Oscar (fetch_for_speck.sh, on a machine that can log in to Oscar):
      data/snn_checkpoints/bmi/<subject>/lsnn_loso_finetuned_binarized/<session>/best_model_weights.pth
      data/results/model_cache/bmi/<subject>/<session>/        the conventional decoders' saved models
    Run:      bash run_final.sh            (both monkeys; several hours)
    Results:  results/final/<subject>/, and a per-decoder summary at the end

Jenkins (hkm, 4 sessions; added 2026-11-06)
    bash fetch_jenkins.sh        export the test splits on Oscar and copy checkpoints, test data, model cache and
                                 Oscar's velocity_scalers.json (with hkm) here; one Duo prompt
    bash run_final.sh jenkins    every decoder, Speck on the devkit (layer 1 threshold x2, 3 ms per step)
    Results: results/final/jenkins/
