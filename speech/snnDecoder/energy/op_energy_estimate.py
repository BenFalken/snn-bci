"""
Operation-count energy ESTIMATES for the RNN and SNN decoders, independent of
the machine they run on (energy_meter.py MEASURES energy instead).

For one evaluation, each estimate counts effective multiply-accumulates
(MACs, one per nonzero input x weight, since zero inputs can be skipped) and
accumulates (ACCs, for spike inputs, which need no multiply), derives memory
accesses (MAC = 4, ACC = 3; Liao et al.), and converts to joules with
Horowitz (ISSCC 2014) 45 nm FP32 figures: MAC 4.6 pJ, ACC 0.9 pJ, memory
access 5 pJ (on-chip SRAM) to 640 pJ (off-chip DRAM). Memory placement
depends on the hardware, so total energy is reported as a range,
energy_total_j_low (all SRAM) to energy_total_j_high (all DRAM), in total and
per sample. Elementwise operations (activations, gate products, membrane
updates) are reported but not included in the energy.

A sample is one 20 ms bin of neural input for both decoders, so per-sample
values compare directly even though the RNN emits one output every
`strides` bins and the SNN one per bin.

The SNN's counts come from SNN_Speech.forward(count_ops=True) and go
through finalize_snn_ops(); the RNN's are computed from its configuration
and the input lengths (estimate_ops_rnn()).
"""

import numpy as np

# Horowitz (2014), 45 nm, FP32
ENERGY_PER_MAC_PJ = 3.7 + 0.9          # multiply + add
ENERGY_PER_ACC_PJ = 0.9                # add only
ENERGY_PER_MEM_ACCESS_SRAM_PJ = 5.0
ENERGY_PER_MEM_ACCESS_DRAM_PJ = 640.0
_PJ_TO_J = 1e-12


def memory_accesses_from_ops(effective_macs, effective_accs):
    """MAC = 3 loads + 1 store; ACC = 2 loads + 1 store (Liao et al.)."""
    return effective_macs * 4 + effective_accs * 3


def estimate_energy_joules(effective_macs, effective_accs, memory_accesses):
    compute = (effective_macs * ENERGY_PER_MAC_PJ + effective_accs * ENERGY_PER_ACC_PJ) * _PJ_TO_J
    sram = memory_accesses * ENERGY_PER_MEM_ACCESS_SRAM_PJ * _PJ_TO_J
    dram = memory_accesses * ENERGY_PER_MEM_ACCESS_DRAM_PJ * _PJ_TO_J
    return {'energy_compute_j': compute, 'energy_memory_j_sram': sram, 'energy_memory_j_dram': dram,
            'energy_total_j_low': compute + sram, 'energy_total_j_high': compute + dram}


def _finalize(effective_macs, effective_accs, elementwise_ops=0, n_samples=None):
    """Totals plus per-sample values."""
    mem = memory_accesses_from_ops(effective_macs, effective_accs)
    energy = estimate_energy_joules(effective_macs, effective_accs, mem)
    result = {'effective_macs': effective_macs, 'effective_accs': effective_accs,
              'elementwise_ops': elementwise_ops, 'memory_accesses': mem, **energy}
    if n_samples:
        result['n_samples'] = n_samples
        result['per_sample'] = {
            'effective_macs': effective_macs / n_samples,
            'effective_accs': effective_accs / n_samples,
            'memory_accesses': mem / n_samples,
            'energy_total_j_low': energy['energy_total_j_low'] / n_samples,
            'energy_total_j_high': energy['energy_total_j_high'] / n_samples,
        }
    return result


def finalize_snn_ops(mac, acc, elementwise, n_samples):
    """Convert SNN_Speech.forward(count_ops=True) counts to the result
    format; n_samples = valid timesteps (20 ms bins) of the call(s)."""
    return _finalize(mac, acc, elementwise_ops=elementwise, n_samples=n_samples)


def smoothing_taps(kernel_sd):
    """Taps of the RNN's Gaussian input-smoothing kernel (gaussSmooth)."""
    from scipy.ndimage import gaussian_filter1d
    inp = np.zeros(100, dtype=np.float32)
    inp[50] = 1
    return int(np.count_nonzero(gaussian_filter1d(inp, kernel_sd) > 0.01))


def estimate_ops_rnn(rnn_args, input_lengths):
    """Op counts for NeuralDecoder's GRU (neuralDecoder/models.py) over trials
    of input_lengths 20 ms bins, from its args.yaml (a dict or OmegaConf).

    Per input bin:   Gaussian smoothing (if smoothInputs): taps MACs per feature
                     input network: in x out MACs per layer (dense: the
                     z-scored features are essentially never zero)
    Per output frame (one per `strides` bins once `kernel_size` bins are in):
                     GRU layer l: in_l x 3u + u x 3u MACs, in_1 = kernel_size x
                     input width (the stacked window), in_l = u after
                     (x2 directions if bidirectional); dense u x n_classes.
    Elementwise per frame and layer, ~14u (bias adds, 3 activations, the
    reset-gate product and the update interpolation), reported only."""
    model, dataset = rnn_args['model'], rnn_args['dataset']
    units, n_layers = int(model['nUnits']), int(model.get('nLayers', 2))
    n_dirs = 2 if model.get('bidirectional', False) else 1
    n_features = int(dataset['nInputFeatures'])
    n_classes = int(dataset['nClasses']) + 1
    stack = model.get('stack_kwargs')
    kernel, stride = (int(stack['kernel_size']), int(stack['strides'])) if stack else (1, 1)
    if int(model.get('subsampleFactor', 1)) != 1:
        print("  WARNING: subsampleFactor > 1 is not modelled; the RNN op estimate counts every frame")

    width = n_features
    input_macs_per_bin = 0
    if 'inputNetwork' in model:
        for size in model['inputNetwork']['inputLayerSizes']:
            input_macs_per_bin += width * int(size)
            width = int(size)
    smoothing_macs_per_bin = smoothing_taps(rnn_args.get('smoothKernelSD', 2)) * n_features \
        if rnn_args.get('smoothInputs', 0) else 0

    gru_in = [kernel * width] + [units * n_dirs] * (n_layers - 1)
    macs_per_frame = sum(n_dirs * (i * 3 * units + units * 3 * units) for i in gru_in) + \
        units * n_dirs * n_classes
    elementwise_per_frame = n_layers * n_dirs * 14 * units

    lengths = np.asarray(input_lengths, dtype=np.int64)
    n_frames = np.where(lengths >= kernel, (lengths - kernel) // stride + 1, 0)
    n_bins, frames = int(lengths.sum()), int(n_frames.sum())
    macs = n_bins * (input_macs_per_bin + smoothing_macs_per_bin) + frames * macs_per_frame
    return _finalize(macs, 0, elementwise_ops=frames * elementwise_per_frame, n_samples=n_bins)
