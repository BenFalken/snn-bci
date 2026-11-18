# Speck/LSNN methodology figures (2026-11-17)

| Fig | Script | What it shows |
|---|---|---|
| 1a | `fig_speck_design.py` | task demands (numbered) -> Speck SNN choices per task, incl. the deployed network on the chip's 9 cores |
| 1b | `fig_lsnn_design.py` | the same task demands -> LSNN choices per task (shared core + per-task settings) |
| 2 | `fig_calibration_overview.py` | one pre-deployment network tuned at the input end (layer-1 threshold vs the PyTorch model's output, no labels) and the output end (host readout vs the true targets) |
| 3 | `fig_calibration_results.py` | measured effect of both calibrations: firing, threshold sweeps (agreement, accuracy), per-neuron distortion, a real-chip sentence, gains |

Data: `calibration_data.py` -> `data/calibration_speech.{npz,json}`; threshold sweeps of the deployed networks in
`~/Downloads/final_results/l1_final`; speech V2 sweep and refit in `~/Downloads/speck_r19/results`.
Shared code: `calibration_panels.py` (panels, icons), `flowdraw.py` (flow-diagram kit); `fig_lsnn_design.py` imports
the header and helpers from `fig_speck_design.py`. Drafts superseded by these: `archive_2026-11-17_calibration_drafts/`.
Fig 3f is final: speech V2 real chip 680 sentences, audio P5 real chip 38 words (3 seeds), Jenkins software refit scored on the real chip.
