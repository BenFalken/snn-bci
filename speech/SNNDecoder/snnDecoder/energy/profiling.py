"""
Streaming latency, measured energy and parameter count of the RNN and SNN
decoders, timed the way a real-time decoder runs: one call per new chunk of
neural input, never batched, state carried from call to call and reset at
each sentence start.

    SNN  one call per 20 ms bin (step()): one output per bin (SNN_Speech), or per
         completed frame (SNN_Recurrent, every frame_stride bins).
    RNN  one call per `strides` new bins (4 = 80 ms) -> one output frame,
         over the last `kernel_size` bins (32 = 640 ms) with the GRU state
         carried (compiled with tf.function, so per-call Python overhead is
         small). Its Gaussian input smoothing is centred (non-causal), so it
         cannot run in a stream and is skipped here; its cost is in the op
         estimate. Each call re-runs the input layer only on the new bins.

latency_s             median per-bin compute time over n_repeats timed passes,
                      after n_warmup untimed passes (TF traces on first call)
latency_per_output_s  compute time per output (RNN: per frame; SNN: per bin)
output_interval_s     time between outputs (RNN 80 ms; SNN 20 ms)
real_time_factor      latency_per_output_s / output_interval_s (< 1 keeps up)
energy_j              per-bin energy from one longer block of n_energy_repeats
                      passes inside an EnergyMeter (RAPL where readable, else
                      a CPU-utilization proxy, see energy_meter.py)
power_w               mean power over that block
param_count           parameters of the model as deployed for one session
                      (shared network + that session's input layer)
"""

import time

import numpy as np

BIN_S = 0.02


def time_per_sample(run_pass, n_samples, energy_meter_cls=None, n_repeats=3, n_warmup=2,
                    n_energy_repeats=10):
    """run_pass() processes n_samples bins. Returns (latency_s, energy_j,
    energy_method, power_w); latency and energy per bin, power_w the mean
    over the energy block."""
    for _ in range(n_warmup):
        run_pass()
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        run_pass()
        times.append(time.perf_counter() - start)
    latency_s = float(np.median(times)) / n_samples

    if energy_meter_cls is None:
        return latency_s, None, None, None
    with energy_meter_cls() as meter:
        for _ in range(n_energy_repeats):
            run_pass()
    if meter.latency_s < 0.01:
        print(f"    [caution] energy window was only {meter.latency_s * 1000:.2f} ms; "
              f"raise n_energy_repeats if energy looks noisy")
    energy_j = None if meter.energy_j is None else meter.energy_j / (n_energy_repeats * n_samples)
    power_w = None if meter.energy_j is None or meter.latency_s <= 0 else meter.energy_j / meter.latency_s
    return latency_s, energy_j, meter.energy_method, power_w


def _take_bins(trials, n_bins, min_len=1):
    """Leading trials (features (T, C), day/layer index), truncated so they
    total about n_bins bins."""
    out, total = [], 0
    for feats, idx in trials:
        if len(feats) < min_len:
            continue
        feats = feats[:max(min_len, n_bins - total)]
        out.append((feats, idx))
        total += len(feats)
        if total >= n_bins:
            break
    return out


def _result(latency_s, energy_j, method, power_w, n_bins, n_outputs, output_interval_s, param_count):
    per_output = latency_s * n_bins / n_outputs
    return {'latency_s': latency_s, 'latency_per_output_s': per_output,
            'output_interval_s': output_interval_s, 'real_time_factor': per_output / output_interval_s,
            'energy_j': energy_j, 'energy_method': method, 'power_w': power_w,
            'param_count': param_count, 'n_timed_bins': n_bins}


def profile_snn(model, trials, n_bins=500, energy_meter_cls=None, n_energy_repeats=10):
    """trials: [(features (T, C) float32, day_idx)], e.g. a session's test
    sentences. The model must be in eval mode."""
    import torch

    if getattr(model, 'smooth_sd', 0) > 0:
        print("  WARNING: this SNN was trained with (non-causal) input smoothing, which a stream "
              "cannot apply; it is timed without it")
    trials = [(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)), d) for f, d in _take_bins(trials, n_bins)]
    n = sum(len(f) for f, _ in trials)

    def run_pass():
        n_out = 0
        with torch.no_grad():
            for feats, day in trials:
                model.reset_stream()
                for t in range(len(feats)):
                    n_out += model.step(feats[t:t + 1], day) is not None
        return n_out

    n_outputs = run_pass()     # also a warm-up
    timing = time_per_sample(run_pass, n, energy_meter_cls, n_energy_repeats=n_energy_repeats)
    # parameters deployed for one session: the shared network plus one session's input layer (the native SNN has
    # one gain per channel and session, day_gain (sessions, channels), instead of per-session layers)
    total = sum(p.numel() for p in model.parameters())
    if getattr(model, 'day_layers', None) is not None:
        day_params = sum(p.numel() for p in model.day_layers[0].parameters())
        params = total - (len(model.day_layers) - 1) * day_params
    elif getattr(model, 'day_gain', None) is not None:
        params = total - (model.day_gain.shape[0] - 1) * model.day_gain.shape[1]
    else:
        params = total
    return _result(*timing, n_bins=n, n_outputs=max(n_outputs, 1),
                   output_interval_s=getattr(model, 'frame_stride', 1) * BIN_S, param_count=int(params))


def profile_rnn(nsd, trials, n_bins=500, energy_meter_cls=None, n_energy_repeats=10):
    """nsd: a NeuralSequenceDecoder with its checkpoint loaded (rnn_step3).
    trials: [(features (T, C) float32, input layer index)]."""
    import tensorflow as tf

    stack = nsd.args['model'].get('stack_kwargs') or {'kernel_size': 1, 'strides': 1}
    kernel, stride = int(stack['kernel_size']), int(stack['strides'])
    model = nsd.model

    steps = {}

    def get_steps(layer_idx):
        if layer_idx not in steps:
            norm, inp = nsd.normLayers[layer_idx], nsd.inputLayers[layer_idx]

            @tf.function
            def first(window):
                x = inp(norm(window), training=False)
                logits, states = model(x, training=False, returnState=True)
                return logits, x[:, stride:], states

            @tf.function
            def step(new_bins, buffer, states):
                window = tf.concat([buffer, inp(norm(new_bins), training=False)], axis=1)
                logits, states = model(window, states=states, training=False, returnState=True)
                return logits, window[:, stride:], states

            steps[layer_idx] = (first, step)
        return steps[layer_idx]

    plans, n, n_outputs = [], 0, 0
    for feats, layer_idx in _take_bins(trials, n_bins, min_len=kernel):
        feats = tf.constant(feats[None].astype(np.float32))
        n_steps = (feats.shape[1] - kernel) // stride
        chunks = [feats[:, kernel + i * stride: kernel + (i + 1) * stride] for i in range(n_steps)]
        plans.append((get_steps(layer_idx), feats[:, :kernel], chunks))
        n += kernel + n_steps * stride
        n_outputs += 1 + n_steps

    def run_pass():
        for (first, step), window, chunks in plans:
            _, buffer, states = first(window)
            for chunk in chunks:
                _, buffer, states = step(chunk, buffer, states)

    timing = time_per_sample(run_pass, n, energy_meter_cls, n_energy_repeats=n_energy_repeats)
    layer_idx = _take_bins(trials, n_bins, min_len=kernel)[0][1]
    params = int(model.count_params() + nsd.inputLayers[layer_idx].count_params())
    return _result(*timing, n_bins=n, n_outputs=n_outputs, output_interval_s=stride * BIN_S, param_count=params)
