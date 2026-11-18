## SNNDecoder: a spiking phoneme decoder trained and evaluated alongside the RNN

A feedforward spiking network (forked from the velocity-decoding SNN in spike_bmi_main) that decodes the same
neural features as NeuralDecoder's RNN into per-20 ms phoneme probabilities, trained with CTC. Its outputs use the
RNN's class order (39 phonemes, SIL, CTC blank), so the same language-model decoders run on both.

### Architecture (`snnDecoder/model.py`)
- per-session input layer (256 -> 256, softsign, dropout 0.2), identity-initialized, as the RNN's day layers
- funnelled spiking stack, 512 -> 256 -> 128 -> 128, IAF neurons (threshold 1.0, multi-spike), each layer with its own
  trained synaptic time constant (cascading, 0.4 -> 3.2 bins = 8 -> 64 ms), hard reset in pretraining, soft in fine-tuning
- readout: 2-stage cascaded EMA of the last layer's spikes -> linear layer -> softmax over 41 classes

Every operation is causal and one 20 ms bin is one timestep, so it decodes in real time: `step()` takes one bin and
returns that bin's class probabilities (greedy CTC decoding emits a phoneme as soon as its frame wins). Training runs
whole padded batches layer by layer, which gives the same spikes as stepping.

### Recurrent, RNN-shaped variant (`snnDecoder/rsnn.py`, `--arch rsnn`, `configs/speech_rsnn.yaml`)
The paper RNN's shape with spiking units: the same per-session input layer, the RNN's causal 32-bin input stacking
with stride 4 (one step per 80 ms, the RNN's output frames), then recurrent spiking layers (default 5 x 1024,
I(t) = W x(t) + U s(t-1), ALIF as an adaptive IAF, soft reset) with dropout 0.4, and a one-step leaky readout.
`step()` takes one 20 ms bin and returns logits whenever a frame completes. `scripts/configs/rsnn_ladder.txt` runs
the full replica, a smoothed replica, a 2 x 512 version, one without stacking and one without recurrence.

### Pipeline (AnalysisExamples)
1. `rnn_step1_makeTFRecords.ipynb` (or the TFRecords in derived.tar.gz)
2. `rnn_step2_trainBaselineRNN.ipynb` and `rnn_step2b_trainSNN.ipynb` (exports the TFRecords to .npz with
   `exportTFRecordsToNpz.py`, then pretrains and fine-tunes the SNN); on Slurm: `scripts/run_snn_pretrain.sbatch`,
   `scripts/run_snn_finetune.sbatch`
3. `rnn_step3_baselineRNNInference.ipynb`: both decoders on the same trials, with the same language model, plus
   a side-by-side table of phoneme/word error rates, op-count energy estimates and measured streaming latency/energy

### Energy (`snnDecoder/energy/`)
- `op_energy_estimate.py`: effective MACs/ACCs per 20 ms bin -> memory accesses -> Horowitz (45 nm) energies, as a range
  from all-SRAM to all-DRAM. The SNN counts its own ops during inference (spike inputs cost ACCs, not MACs); the RNN's
  are computed from its configuration (input smoothing, input layer, 5 stacked GRU layers on the 32-bin window, every
  4 bins, and the output layer).
- `energy_meter.py` / `profiling.py`: measured per-bin latency and energy (Intel RAPL, else a CPU-utilization proxy),
  with each decoder run as a stream: the SNN one bin per call, the RNN one 4-bin step per call with its GRU state carried.

### Differences from the velocity SNN worth knowing
- **Time constants in physical time.** tau_syn and the EMA decay count timesteps. The velocity SNN's 2, 4, 8, 16 and
  0.8 were in 4 ms steps; at 20 ms bins the same time constants are 0.4, 0.8, 1.6, 3.2 and 0.8^5 = 0.33 (the defaults
  in `configs/speech_pooled.yaml`). Keeping the step counts (40-320 ms) made the network fail to learn a synthetic
  phoneme task entirely.
- **Unit-gain synapses (`--norm-syn`, default on).** sinabs updates `i_syn = alpha * (i_syn + input)`, a steady-state
  gain of alpha / (1 - alpha), from 0.09 (tau_syn 0.4) to 15.5 (tau_syn 16), so each layer's time constant also set how
  hard it drove the next layer. norm_syn rescales each layer's input to unit gain; it is one scalar per layer, so
  deployed it folds into the weights.
- **Unit-gain EMA readout.** Each EMA stage is `s += (1 - decay) * (x - s)` instead of `s = decay * s + x`: the
  population-vector readout cancelled that gain, a softmax readout would not.
- **Continuous input.** The input is the RNN's z-scored threshold crossings + spike-band power (no binarization); the
  first spiking layer's input is the continuous output of the session's input layer.
- **No input smoothing by default.** The RNN smooths its input with a centred Gaussian (80 ms lookahead);
  `--smooth-sd 2` gives the SNN the same, at the cost of real-time causality.
