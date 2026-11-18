"""
Side-by-side summary of the RNN and SNN decoders for rnn_step3: accuracy
(greedy phoneme error rate, word error rate with the language model), the
op-count energy estimate, the measured streaming latency and energy, and
each decoder's timing structure.

    entries = {'RNN': {...}, 'SNN': {...}}, each with any of
        per, wer           phoneme / word error rate (fractions)
        op_estimate        energy.op_energy_estimate result
        profile            energy.profiling result
        lookahead_s        future input each output depends on
    print_comparison(entries); save_comparison(entries, path)
"""

import json

import numpy as np

from .dataset import gauss_kernel


def rnn_lookahead_s(rnn_args, bin_s=0.02):
    """The RNN's centred input smoothing reads (taps - 1) / 2 bins ahead."""
    if not rnn_args.get('smoothInputs', 0):
        return 0.0
    return (len(gauss_kernel(rnn_args.get('smoothKernelSD', 2))) - 1) // 2 * bin_s


def snn_lookahead_s(smooth_sd, bin_s=0.02):
    return 0.0 if smooth_sd <= 0 else (len(gauss_kernel(smooth_sd)) - 1) // 2 * bin_s


def _fmt(value, scale=1.0, spec='.3f', suffix=''):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    return f"{value * scale:{spec}}{suffix}"


def _rows(entry):
    ops = entry.get('op_estimate') or {}
    per_sample = ops.get('per_sample', {})
    prof = entry.get('profile') or {}
    wer = entry.get('wer')
    wer = wer[0] if isinstance(wer, (tuple, list)) else wer
    return [
        ('PER (greedy)', _fmt(entry.get('per'), 100, '.2f', ' %')),
        ('WER (+ LM)', _fmt(wer, 100, '.2f', ' %')),
        ('MACs / bin', _fmt(per_sample.get('effective_macs'), 1e-6, '.3f', ' M')),
        ('ACCs / bin', _fmt(per_sample.get('effective_accs'), 1e-6, '.3f', ' M')),
        ('est. energy / bin (SRAM)', _fmt(per_sample.get('energy_total_j_low'), 1e6, '.3f', ' uJ')),
        ('est. energy / bin (DRAM)', _fmt(per_sample.get('energy_total_j_high'), 1e6, '.3f', ' uJ')),
        ('meas. latency / bin', _fmt(prof.get('latency_s'), 1e3, '.4f', ' ms')),
        ('meas. latency / output', _fmt(prof.get('latency_per_output_s'), 1e3, '.4f', ' ms')),
        ('output interval', _fmt(prof.get('output_interval_s'), 1e3, '.0f', ' ms')),
        ('real-time factor', _fmt(prof.get('real_time_factor'), 1, '.4f')),
        ('meas. energy / bin', _fmt(prof.get('energy_j'), 1e6, '.3f', ' uJ')
         + (f" ({prof['energy_method']})" if prof.get('energy_method') else '')),
        ('meas. power', _fmt(prof.get('power_w'), 1, '.2f', ' W')),
        ('parameters (1 session)', _fmt(prof.get('param_count'), 1e-6, '.2f', ' M')),
        ('input lookahead', _fmt(entry.get('lookahead_s'), 1e3, '.0f', ' ms')),
    ]


def print_comparison(entries):
    names = list(entries)
    rows = {name: _rows(entries[name]) for name in names}
    labels = [label for label, _ in rows[names[0]]]
    width = max(len(l) for l in labels) + 2
    col = max(16, max(len(v) for name in names for _, v in rows[name]) + 2)
    print(' ' * width + ''.join(f"{name:>{col}s}" for name in names))
    for i, label in enumerate(labels):
        print(f"{label:<{width}s}" + ''.join(f"{rows[name][i][1]:>{col}s}" for name in names))


def save_comparison(entries, path):
    def ready(x):
        if isinstance(x, dict):
            return {k: ready(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [ready(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    with open(path, 'w') as f:
        json.dump(ready(entries), f, indent=2)
    print(f"Saved {path}")
