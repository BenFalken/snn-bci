# Decoder efficiency

Every model under the same accounting (scripts/efficiency.py). Energy is an estimate for hardware that skips silent neurons; average power = energy per second of speech.

## Accuracy and cost

| model | PER | params (deployed) | weights fp32 / int8 | MAC/s | ACC/s | power, compute only | power, + on-chip memory | power, + off-chip weights | power, int8 + on-chip | spikes/neuron/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| GRU 2x1024 out80, same input, 96 ep | 0.211 | 10.3M | 41 / 10 MB | 517M | 0.0M | 2.38 mW | 4.96 mW | 333.0 mW | 0.80 mW | - |
| LSTM 2x1024, 96 ep | 0.208 | 13.8M | 55 / 14 MB | 686M | 0.0M | 3.15 mW | 6.58 mW | 441.8 mW | 1.06 mW | - |
| QRNN 2x1024, 96 ep | 0.317 (epoch 49/96) | 8.0M | 32 / 8 MB | 397M | 0.0M | 1.83 mW | 3.81 mW | 255.9 mW | 0.62 mW | - |
| Wiener filter (160 ms) | 0.765 | 0.1M | 0 / 0 MB | 1M | 0.0M | 0.00 mW | 0.01 mW | 0.7 mW | 0.00 mW | - |
| Kalman filter per session | 0.710 | 0.0M | 0 / 0 MB | 0M | 0.0M | 0.00 mW | 0.00 mW | 0.1 mW | 0.00 mW | - |
| Paper RNN | 0.188 (given) | 53.6M | 214 / 54 MB | 672M | 0.0M | 3.09 mW | 6.46 mW | 433.3 mW | 1.04 mW | - |
| 0 original layout (L0_orig) | 0.900 | 0.3M | 1 / 0 MB | 7M | 2.6M | 0.03 mW | 0.08 mW | 5.9 mW | 0.01 mW | 0.30, 0.37, 0.39 |
| narrow + recurrence (R1_rec) | 0.869 | 0.6M | 3 / 1 MB | 7M | 9.6M | 0.04 mW | 0.12 mW | 10.4 mW | 0.02 mW | 0.38, 0.39, 0.21 |
| 1 + width (L1_width) | 0.723 | 1.4M | 5 / 1 MB | 13M | 16.7M | 0.08 mW | 0.23 mW | 19.1 mW | 0.04 mW | 0.31, 0.18 |
| 2 + recurrence (S2_rec) | 0.341 | 3.5M | 14 / 3 MB | 13M | 56.0M | 0.11 mW | 0.46 mW | 44.4 mW | 0.10 mW | 0.33, 0.40 |
| 3 + 80 ms outputs (S3_out80) | 0.263 | 3.5M | 14 / 3 MB | 13M | 14.5M | 0.07 mW | 0.21 mW | 17.8 mW | 0.04 mW | 0.35, 0.39 |
| 4 + readout from both layers (S4_readall) | 0.264 | 3.5M | 14 / 3 MB | 13M | 14.9M | 0.07 mW | 0.22 mW | 18.0 mW | 0.04 mW | 0.36, 0.39 |
| 5 + trained time constants (C_nodelay96) | 0.263 | 3.5M | 14 / 3 MB | 13M | 14.8M | 0.07 mW | 0.22 mW | 18.0 mW | 0.04 mW | 0.36, 0.38 |
| Model A, DelayNet (R_readall96) | 0.245 | 4.3M | 17 / 4 MB (+0.7 MB delays) | 53M | 14.8M | 0.26 mW | 0.60 mW | 43.4 mW | 0.10 mW | 0.35, 0.39 |
| Model B, LSNN, final (C_lsnn_nodelay96) | 0.249 | 3.5M | 14 / 3 MB | 13M | 15.2M | 0.08 mW | 0.22 mW | 18.2 mW | 0.04 mW | 0.36, 0.40 |

## Where the compute energy goes (fp32, compute only)

| model | input layers | hidden synapses | neuron updates | readout |
|---|---|---|---|---|
| GRU 2x1024 out80, same input, 96 ep | 8% | 91% | 0% | 0% |
| LSTM 2x1024, 96 ep | 8% | 92% | 0% | 0% |
| QRNN 2x1024, 96 ep | 21% | 79% | 0% | 0% |
| Wiener filter (160 ms) | 0% | 0% | 0% | 100% |
| Kalman filter per session | 2% | 98% | 0% | 0% |
| Paper RNN | 47% | 53% | 0% | 0% |
| 0 original layout (L0_orig) | 92% | 7% | 1% | 0% |
| narrow + recurrence (R1_rec) | 77% | 22% | 1% | 0% |
| 1 + width (L1_width) | 80% | 19% | 1% | 0% |
| 2 + recurrence (S2_rec) | 54% | 45% | 0% | 1% |
| 3 + 80 ms outputs (S3_out80) | 82% | 17% | 1% | 0% |
| 4 + readout from both layers (S4_readall) | 81% | 17% | 1% | 0% |
| 5 + trained time constants (C_nodelay96) | 81% | 17% | 1% | 0% |
| Model A, DelayNet (R_readall96) | 95% | 5% | 0% | 0% |
| Model B, LSNN, final (C_lsnn_nodelay96) | 80% | 18% | 2% | 1% |

## Latency

| model | lookahead | output every | input window | layers in sequence | compute per output (1 CPU thread) | real-time factor |
|---|---|---|---|---|---|---|
| GRU 2x1024 out80, same input, 96 ep | 0 ms | 80 ms | 20 ms | 2 | 8.56 ms | 0.107 |
| LSTM 2x1024, 96 ep | 0 ms | 80 ms | 80 ms | 2 | 20.74 ms | 0.259 |
| QRNN 2x1024, 96 ep | 0 ms | 80 ms | 80 ms | 2 | 12.19 ms | 0.152 |
| Wiener filter (160 ms) | 0 ms | 80 ms | 160 ms | 1 | 0.07 ms | 0.001 |
| Kalman filter per session | 0 ms | 80 ms | 80 ms | 1 | 0.09 ms | 0.001 |
| Paper RNN | 80 ms | 80 ms | 640 ms | 5 | 19.02 ms | 0.238 |
| 0 original layout (L0_orig) | 0 ms | 20 ms | 20 ms | 3 | 0.73 ms | 0.037 |
| narrow + recurrence (R1_rec) | 0 ms | 20 ms | 20 ms | 3 | 0.78 ms | 0.039 |
| 1 + width (L1_width) | 0 ms | 20 ms | 20 ms | 2 | 0.78 ms | 0.039 |
| 2 + recurrence (S2_rec) | 0 ms | 20 ms | 20 ms | 2 | 1.26 ms | 0.063 |
| 3 + 80 ms outputs (S3_out80) | 0 ms | 80 ms | 20 ms | 2 | 5.20 ms | 0.065 |
| 4 + readout from both layers (S4_readall) | 0 ms | 80 ms | 20 ms | 2 | 5.36 ms | 0.067 |
| 5 + trained time constants (C_nodelay96) | 0 ms | 80 ms | 20 ms | 2 | 5.38 ms | 0.067 |
| Model A, DelayNet (R_readall96) | 0 ms | 80 ms | 20 ms | 2 | 22.90 ms | 0.286 |
| Model B, LSNN, final (C_lsnn_nodelay96) | 0 ms | 80 ms | 20 ms | 2 | 5.47 ms | 0.068 |

## Post-training weight quantization (test PER)

| model | fp32 | int8 | int6 | int4 |
|---|---|---|---|---|
| GRU 2x1024 out80, same input, 96 ep | 0.211 | 0.212 | 0.212 | 0.222 |
| LSTM 2x1024, 96 ep | 0.208 | 0.208 | 0.208 | 0.219 |
| Wiener filter (160 ms) | 0.765 | 0.765 | 0.765 | 0.757 |
| Kalman filter per session | 0.710 | 0.710 | 0.710 | 0.710 |
| 0 original layout (L0_orig) | 0.900 | 0.899 (0.33 sp) | 0.926 (0.33 sp) | 0.898 (0.33 sp) |
| narrow + recurrence (R1_rec) | 0.869 | 0.870 (0.36 sp) | 0.867 (0.36 sp) | 0.787 (0.36 sp) |
| 1 + width (L1_width) | 0.723 | 0.720 (0.25 sp) | 0.728 (0.25 sp) | 0.714 (0.25 sp) |
| 2 + recurrence (S2_rec) | 0.341 | 0.344 (0.36 sp) | 0.342 (0.36 sp) | 0.410 (0.36 sp) |
| 3 + 80 ms outputs (S3_out80) | 0.263 | 0.265 (0.37 sp) | 0.266 (0.37 sp) | 0.284 (0.37 sp) |
| 4 + readout from both layers (S4_readall) | 0.264 | 0.264 (0.37 sp) | 0.264 (0.37 sp) | 0.321 (0.37 sp) |
| 5 + trained time constants (C_nodelay96) | 0.263 | 0.261 (0.37 sp) | 0.264 (0.37 sp) | 0.287 (0.37 sp) |
| Model A, DelayNet (R_readall96) | 0.245 | 0.244 (0.37 sp) | 0.246 (0.37 sp) | 0.271 (0.37 sp) |
| Model B, LSNN, final (C_lsnn_nodelay96) | 0.249 | 0.250 (0.38 sp) | 0.251 (0.38 sp) | 0.289 (0.38 sp) |

## Pareto fronts (no other model has both lower PER and lower cost)

- **PER vs power (fp32 + on-chip memory)**: GRU 2x1024 out80, same input, 96 ep, Kalman filter per session, Paper RNN, 3 + 80 ms outputs (S3_out80), 5 + trained time constants (C_nodelay96), Model A, DelayNet (R_readall96), Model B, LSNN, final (C_lsnn_nodelay96)
- **PER vs power (compute only)**: GRU 2x1024 out80, same input, 96 ep, Kalman filter per session, Paper RNN, 3 + 80 ms outputs (S3_out80), 5 + trained time constants (C_nodelay96), Model A, DelayNet (R_readall96), Model B, LSNN, final (C_lsnn_nodelay96)
- **PER vs weight memory**: GRU 2x1024 out80, same input, 96 ep, LSTM 2x1024, 96 ep, Kalman filter per session, Paper RNN, 3 + 80 ms outputs (S3_out80), Model A, DelayNet (R_readall96), Model B, LSNN, final (C_lsnn_nodelay96)

## Notes

- QRNN 2x1024, 96 ep: still training (49 of 96 epochs); its PER is not final.
- PER: greedy CTC phoneme error rate on the test partition of each model's sessions; rows marked (history) or (given) were not re-evaluated.
- GRU op counts are exact for its shape; SNN synaptic ACCs use the measured spikes on the test set.
- Per-session input layers: one session's counted in the deployed size and energy; the native SNN's per-session gains fold into its first layer's weights.
- Activation traffic between layers (spike events, GRU hidden vectors) is not counted.
