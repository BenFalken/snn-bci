# Final results production (2026-11-17)

Final Speck models: Motor I as deployed (Indy, Loco); Motor II Jenkins with its readout refitted in software
(final_results/jenkins_refit); audio P5 with its readout refitted (audio_preds/words_P5_*/chip_refit.npy); speech
round 19 V2, two networks on one chip, readout refitted (final_results/speech_v2, perTrialPER_final_v2.json).

Data:   python final_speech_v2.py && python collect_four.py && python collect_final.py && python collect_speck_ops.py
Figures: python fig_results.py
  fig_results_fidelity          every decoder, each experiment's metric (Speck on the chip, software as a hollow diamond)
  fig_results_latency           time per output against the real-time budget
  fig_results_energy            estimated energy per output (MAC + ACC operation counts, 45 nm)
  fig_results_energy_measured   auxiliary: measured on the laptop (RAPL; Speck chip power monitor + host share)
Videos (four tracks each: ground truth, the source paper's best decoder, LSNN, Speck):
  video_motor.py -> figures/video_motor1.mp4 (Indy, QRNN), video_motor2.mp4 (Jenkins, QRNN)
  video_audio_words.py --all --ann gru_rnn -> figures/video_audio_words_final.mp4 (GRU)
  video_speech.py -> figures/video_speech_final.mp4 (the paper's RNN)
Backups of the collectors and tables before the switch: backup_2026-11-17_final/.
Speech Speck host share: V2's own RAPL run on the laptop (results_speck_deploy/bench, 7.0 mJ, 0.25 ms per output).
Energy billing (2026-11-17 late): every decoder charged energy while computing each output; speech chip changed from power x 80 ms (290 uJ) to power x loop time (84 uJ). collect_final.py backup backup_2026-11-17_final/collect_final.pre_busytime.py.
2026-11-18 AUDIO = two monkeys (expanded cohorts, ALLWORDS_NOVEL_DAYNORM, Neil 44 + Gomez 35 test utterances): collect_audio_novel.py -> data/audio_novel_fidelity.csv; collect_four.audio_rows reads it (group = monkey; the Neil-20 rows kept as audio_rows_neil20); audio Speck chip energy/time from results_audio_novel (6 devkit runs). KALMAN FIX: the audio KF (raw split) had been scored 7 frames out of step; collect_fidelity.align_raw aligns it (Neil KF 0.20 -> 0.38). Calibration gains panel: audio N 93->98%, G 96->102%.
V2+C host share measured on the laptop (results_bench_v2c, 9.18 mJ, 0.34 ms per output), 2026-11-18.

2026-11-18 CLEANUP: superseded scripts, figures, data, backups (paper_figures), old devkit bundles and returned results
(Downloads), and old snn-decoders runs moved to ~/Downloads/_archive_2026-11-18 (MANIFEST.txt lists every item and its
original path; nothing deleted). Every final collector and figure script was rerun afterwards: all ran, and
four_fidelity.csv, final_efficiency.csv and speck_ops.csv came out identical. speck_deploy.zip rebuilt with the final
models (speech V2+C, audio P5 for monkeys N and G, Jenkins refit).
