"""
Accuracy against energy, memory and latency for trained decoders, every model under the same accounting
rules, so that "rivals the RNN while using less energy, less memory and decoding faster" can be checked as a
Pareto statement: is there an SNN that no GRU beats on both PER and the cost?

    python efficiency.py --runs configs/efficiency_runs.txt --data-dir $BASE_DIR/derived/snnData \
        --out $BASE_DIR/derived/efficiency/efficiency

Run list (one model per line, "label | kind | path-or-shape [| key=value ...]"; # starts a comment):
    snn       a train.py checkpoint (final_model_*.pth; --arch rsnn or native). Evaluated on the test
              partition of its sessions: greedy PER and the spikes of every layer, measured.
    gru       an ann_controls.py history (<out-dir>/<model>.json). With its weights next to it
              (<model>.pt, saved since this script) it is evaluated like an SNN; without, PER is the
              history's final test PER and quantization is skipped. Its op counts do not depend on data.
    gru_spec  a GRU shape that was not trained here, e.g. the paper RNN:
              "Paper RNN | gru_spec | hidden=1024 layers=5 window=32 stride=4 smooth_sd=2 per=0.188"
Paths may use $VARIABLES and wildcards (each must match exactly one file; otherwise the run is skipped
with a message, so runs that have not finished yet can stay in the list). Options: data=<dir> (another data directory, e.g. derived/snnDataTx1 for raw counts), family=<name>
(groups the Pareto front; default snn / gru).

Accounting, per output frame and per second of speech (the 80 ms models emit 12.5 frames/s, the 20 ms
ones 50/s; per-second figures put both on one scale and µJ/s is the average power in µW):
    synaptic ops    MAC (multiply-accumulate) for a weight times a continuous value, ACC (accumulate only)
                    for a weight added because a spike arrived (a count of n adds it n times).
                    GRU: every weight of every gate, every frame. SNN: the first layer's input is the
                    stacked continuous features (dense MACs; raw counts for --arch native: ACCs), every
                    later synapse costs one ACC per spike that crosses it.
    neuron update   multiplies per neuron and step (SNN: membrane leak 1, threshold adaptation 2, synaptic
                    filter 1; GRU: the 3 gate products per unit), priced as MACs. Comparisons, resets and
                    sigmoid/tanh evaluations are counted (elementwise) but not priced.
    memory traffic  one weight read per synaptic op (a dense layer reads every weight each frame, an SNN
                    only the rows of neurons that spiked), plus a read and a write of every state variable
                    per step (SNN: membrane, adaptation, synaptic current; GRU: hidden state).
    energy          Horowitz (ISSCC 2014), 45 nm: fp32 MAC 4.6 pJ, ACC 0.9 pJ; int8 MAC 0.3 pJ, int32 ACC
                    0.1 pJ; memory 5 pJ per 32-bit word on chip (SRAM), 640 pJ off chip (DRAM), scaled by
                    word width. Scenarios: compute only, + SRAM, + DRAM weights, and int8 weights + SRAM.
                    This is an estimate for hardware that skips silent neurons (event-driven); on a GPU or
                    CPU, dense kernels do the same work whether a neuron spiked or not.
    memory          parameters as deployed for one session (shared network + that session's input layer)
                    at 32/16/8/4 bits, and the state carried between steps.
    latency         lookahead (future input a decision waits for: the RNN's centred smoothing), time
                    between outputs, input window, layers in sequence, and the measured compute time per
                    output: one call per new 20 ms bin, batch 1, one CPU thread, PyTorch for every model.
    quantization    per-row symmetric post-training quantization of every weight matrix to 8/6/4 bits
                    (biases and time constants kept), then the test PER again (and the SNN's spike rate).

Writes <out>.json (everything), <out>.md (tables) and <out>_pareto.png.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snnDecoder.compare import snn_lookahead_s                               # noqa: E402
from snnDecoder.dataset import SessionBatches, load_partitions, smooth_inputs  # noqa: E402
from snnDecoder.energy.op_energy_estimate import (ENERGY_PER_ACC_PJ, ENERGY_PER_MAC_PJ,  # noqa: E402
                                                  ENERGY_PER_MEM_ACCESS_DRAM_PJ, ENERGY_PER_MEM_ACCESS_SRAM_PJ)
from snnDecoder.utils import edit_distance, greedy_decode                    # noqa: E402

BIN_S = 0.02
N_CLASSES, BLANK = 41, 40
INT8_MAC_PJ, INT_ACC_PJ = 0.2 + 0.1, 0.1
SCENARIOS = {   # name: (MAC pJ, ACC pJ, weight word pJ, state word pJ)
    'compute_fp32': (ENERGY_PER_MAC_PJ, ENERGY_PER_ACC_PJ, 0.0, 0.0),
    'sram_fp32': (ENERGY_PER_MAC_PJ, ENERGY_PER_ACC_PJ, ENERGY_PER_MEM_ACCESS_SRAM_PJ, ENERGY_PER_MEM_ACCESS_SRAM_PJ),
    'dram_fp32': (ENERGY_PER_MAC_PJ, ENERGY_PER_ACC_PJ, ENERGY_PER_MEM_ACCESS_DRAM_PJ, ENERGY_PER_MEM_ACCESS_SRAM_PJ),
    'sram_int8': (INT8_MAC_PJ, INT_ACC_PJ, ENERGY_PER_MEM_ACCESS_SRAM_PJ / 4, ENERGY_PER_MEM_ACCESS_SRAM_PJ),
}
SCENARIO_LABELS = {'compute_fp32': 'compute only (fp32)', 'sram_fp32': 'fp32 + on-chip memory',
                   'dram_fp32': 'fp32 + off-chip weights', 'sram_int8': 'int8 weights + on-chip memory'}


# ---------------------------------------------------------------------------------------------- run list
def parse_runs(path):
    runs = []
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split('|')]
        if len(fields) < 3:
            raise ValueError(f'{path}: "{line}" needs "label | kind | path-or-shape"')
        label, kind, target = fields[:3]
        if kind not in ('snn', 'gru', 'gru_spec'):
            raise ValueError(f'{path}: unknown kind {kind!r} in "{line}"')
        opts = dict(kv.split('=', 1) for f in fields[3:] for kv in f.split())
        if kind == 'gru_spec':
            opts.update(dict(kv.split('=', 1) for kv in target.split()))
            target = None
        if target is not None:
            matches = sorted(glob.glob(os.path.expandvars(target)))
            if len(matches) != 1:
                print(f'[skip] {label}: {target} matches {len(matches)} files {matches[:4]}')
                continue
            target = matches[0]
        if 'data' in opts:
            opts['data'] = os.path.expandvars(opts['data'])
        runs.append({'label': label, 'kind': kind, 'path': target, 'opts': opts})
    return runs


# ---------------------------------------------------------------------------------------------- models
def load_run(run, default_data_dir, device):
    """-> dict(model, sessions, data_dir, smooth_sd, history_per) for a run; model None for gru_spec or
    a gru without saved weights."""
    import ann_controls
    opts = run['opts']
    if run['kind'] == 'snn':
        from snnDecoder.inference import load_snn_model
        model, ckpt = load_snn_model(run['path'], device)
        args = ckpt.get('args', {})
        data_dir = opts.get('data') or args.get('data_dir') or default_data_dir
        if not os.path.isdir(data_dir):
            data_dir = opts.get('data') or default_data_dir
        return {'model': model, 'sessions': list(ckpt['sessions']), 'data_dir': data_dir,
                'smooth_sd': model.smooth_sd, 'history_per': None, 'arch': args.get('arch', 'ff')}
    if run['kind'] == 'gru':
        info = json.load(open(run['path']))
        args = info['args']
        data_dir = opts.get('data') or args.get('data_dir') or default_data_dir
        if not os.path.isdir(data_dir):
            data_dir = opts.get('data') or default_data_dir
        sessions = list(args['sessions'])
        n_in = int(load_partitions(data_dir, sessions[:1], 'test')[0].features[0].shape[1])
        ann_controls.N_IN = n_in
        model = ann_controls.MODELS[args['model']](len(sessions))
        weights = os.path.splitext(run['path'])[0] + '.pt'
        has_weights = os.path.isfile(weights)
        if has_weights:
            model.load_state_dict(torch.load(weights, map_location='cpu'))
        return {'model': model.to(device).eval(), 'evaluate': has_weights, 'sessions': sessions,
                'data_dir': data_dir, 'smooth_sd': float(args.get('smooth_sd') or 0.0),
                'history_per': float(info['history']['test_per'][-1]), 'arch': 'gru',
                'epochs_done': len(info['history']['test_per']), 'epochs_planned': args.get('epochs')}
    # gru_spec: build the shape (one session) for its parameter and op counts
    ann_controls.N_IN = int(opts.get('n_in', 256))
    model = ann_controls.StackGRU(1, int(opts['hidden']), int(opts['layers']), window=int(opts.get('window', 32)),
                                  stride=int(opts.get('stride', 4)))
    return {'model': model.eval(), 'evaluate': False, 'sessions': [], 'data_dir': None,
            'smooth_sd': float(opts.get('smooth_sd', 0.0)), 'history_per': float(opts['per']), 'arch': 'gru'}


def is_snn(model):
    return hasattr(model, 'layers') and hasattr(model, 'layer_widths')


def frame_shape(model):
    if hasattr(model, 'frame_stride'):
        return int(model.frame_kernel), int(model.frame_stride)
    return int(getattr(model, 'window', 1)), int(getattr(model, 'stride', 1))


def out_lengths(model, lengths):
    return model.output_lengths(lengths) if hasattr(model, 'output_lengths') else lengths


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate(model, parts, smooth_sd, device, batch_size=32):
    """Greedy PER over the partitions; for an SNN also the spikes of every layer and the input counts."""
    snn = is_snn(model)
    edit = phones = frames = 0
    spikes = None
    in_count = 0.0
    in_ops = {'mac': 0.0, 'acc': 0.0, 'frames': 0}
    events = slots = 0.0
    with torch.no_grad():
        for part in parts:
            for b in SessionBatches([part], batch_size, shuffle=False):
                x = smooth_inputs(b['x'].to(device), smooth_sd)
                lengths = b['lengths']
                if snn:
                    logits = model(x, part.day_idx, lengths=lengths, count_ops=True)
                    if getattr(model, 'input_ops', None):                  # measured first-layer ops
                        for k in in_ops:
                            in_ops[k] += model.input_ops[k]
                    if getattr(getattr(model, 'encoder', None), 'active', False):
                        events += float(model.input_events)
                        slots += model.input_event_slots
                    ls = [float(s) for s in model.layer_spikes]
                    spikes = ls if spikes is None else [a + c for a, c in zip(spikes, ls)]
                    if getattr(model, 'day_input', None) is not None:     # native: the raw counts it reads
                        xin = model.select_channels(x)
                        mask = (torch.arange(x.shape[1], device=device)[None, :] < lengths.to(device)[:, None])
                        in_count += float((xin.abs() * mask[..., None]).sum())
                else:
                    logits = model(x, part.day_idx)
                olen = out_lengths(model, lengths.to(device)).cpu().numpy()
                logits = logits.cpu().numpy()
                targets = np.split(b['targets'].numpy(), np.cumsum(b['target_lengths'].numpy())[:-1])
                for i, (n, target) in enumerate(zip(olen, targets)):
                    edit += edit_distance(greedy_decode(logits[i, :n], BLANK), target)
                    phones += len(target)
                    frames += int(n)
    return {'per': edit / phones, 'frames': frames, 'layer_spikes': spikes, 'input_count': in_count,
            'input_ops': in_ops if in_ops['frames'] else None,
            'input_event_rate': events / slots if slots else None}


# ---------------------------------------------------------------------------------------------- accounting
def snn_counts(model, ev):
    """Per output frame: synaptic MAC/ACC, neuron-update MACs, elementwise, weight reads, state words."""
    kernel, stride = frame_shape(model)
    K = int(getattr(model, 'substeps', 1))
    widths = list(model.layer_widths)
    frames = max(ev['frames'], 1)
    s = [x / frames for x in ev['layer_spikes']]                 # spikes per frame, per layer
    native = getattr(model, 'day_input', None) is not None
    c = {k: 0.0 for k in ('mac_in', 'acc_in', 'mac_syn', 'acc_syn', 'mac_neuron', 'mac_readout', 'acc_readout',
                          'elementwise', 'state_words')}
    if native:
        n_in = model.n_in
        io = ev.get('input_ops')
        if io:                                                   # measured: per-day layer + first layer
            c['mac_in'] += io['mac'] / io['frames']
            c['acc_in'] += io['acc'] / io['frames']
        elif model.day_input == 'linear':
            c['mac_in'] += n_in * n_in + n_in * widths[0]
        else:                                                    # counts add their (gain-folded) weight
            c['acc_in'] += ev['input_count'] / frames * widths[0]
    else:
        C = model.num_input_channels
        c['mac_in'] += stride * model._day_ops()                 # the per-session input layer, per bin
        io = ev.get('input_ops')
        if io:                                                   # measured: dense MACs, or one ACC per event
            c['mac_in'] += io['mac'] / io['frames']
            c['acc_in'] += io['acc'] / io['frames']
        else:
            c['mac_in'] += C * kernel * widths[0]                # stacked continuous input: dense
    for li, layer in enumerate(model.layers):
        if li > 0:
            c['acc_syn'] += s[li - 1] * widths[li]
        if layer.rec is not None:
            c['acc_syn'] += s[li] * widths[li]
        has_syn = getattr(layer, 'tau_syn', None) is not None or getattr(layer, 'log_tau_syn', None) is not None
        leak = layer.neuron_type != 'iaf' or getattr(layer, 'log_tau_mem', None) is not None
        mask = getattr(layer, 'adapt_mask', None)               # mixed population: the adaptive share only
        alif = 1.0 if layer.neuron_type == 'alif' else (float(mask.mean()) if mask is not None else 0.0)
        sead = layer.neuron_type == 'seadlif'                   # input scaling, a u, b s, adaptation decay
        c['mac_neuron'] += K * widths[li] * (int(leak) + 2 * alif + 4 * int(sead) + int(has_syn))
        c['elementwise'] += K * widths[li] * 2                   # threshold comparison, reset
        c['state_words'] += K * widths[li] * (1 + alif + int(sead) + int(has_syn))
    if getattr(model, 'feedback', None) is not None:
        c['acc_syn'] += s[-1] * widths[0]
    if native:
        alphas = getattr(model, 'readout_alphas', None) or []
        n_dec = sum(a > 0 for a in alphas)
        n_plain = len(alphas) - n_dec if alphas else 1
        n_read = sum(widths) if getattr(model, 'readout_all', False) else widths[-1]
        if n_dec:                                                # readout traces: decay + dense read, every step
            c['mac_readout'] += K * n_read * n_dec * (1 + N_CLASSES)
            c['state_words'] += K * n_read * n_dec
        if getattr(model, 'readout_vmem', False):                # membrane readout: dense, every step
            c['mac_readout'] += K * n_read * N_CLASSES * n_plain
        else:
            c['acc_readout'] += (sum(s) if getattr(model, 'readout_all', False) else s[-1]) * N_CLASSES * n_plain
        S = int(getattr(model, 'out_stages', 1))
        c['mac_readout'] += 2 * N_CLASSES * K * S                # output leak, every step and stage
        c['state_words'] += N_CLASSES * K * S
        mem = getattr(model, 'memory', None)
        if mem is not None:                                      # first-layer synaptic currents that decay
            c['state_words'] += mem.n_decaying * widths[0] * K
    else:
        stages = int(model.temporal_decay_stages)
        if stages:                                               # smoothed spikes: continuous
            c['mac_readout'] += stages * K * widths[-1] + widths[-1] * N_CLASSES
            c['state_words'] += stages * K * widths[-1]
        else:
            c['acc_readout'] += s[-1] * N_CLASSES
    return c


def gru_counts(model):
    if hasattr(model, 'frame_counts'):                      # LSTM, QRNN, Wiener, Kalman (ann_controls.py)
        return model.frame_counts()
    kernel, stride = frame_shape(model)
    g = model.gru
    H, L, n_in = g.hidden_size, g.num_layers, g.input_size
    C = model.inp.layers[0].in_features
    c = {k: 0.0 for k in ('mac_in', 'acc_in', 'mac_syn', 'acc_syn', 'mac_neuron', 'mac_readout', 'acc_readout',
                          'elementwise', 'state_words')}
    c['mac_in'] += stride * C * C
    c['mac_in'] += n_in * 3 * H                                  # first layer's input weights (stacked input)
    c['mac_syn'] += L * H * 3 * H + (L - 1) * H * 3 * H          # recurrent, and inputs of layers 2..L
    c['mac_neuron'] += L * 3 * H                                 # r*h, z*h, (1-z)*n
    c['elementwise'] += L * 3 * H                                # 2 sigmoids, 1 tanh per unit
    c['mac_readout'] += H * N_CLASSES
    c['state_words'] += L * H
    return c


def summarize_counts(c, frames_per_s):
    mac = c['mac_in'] + c['mac_syn'] + c['mac_neuron'] + c['mac_readout']
    acc = c['acc_in'] + c['acc_syn'] + c['acc_readout']
    weight_reads = c['mac_in'] + c['mac_syn'] + c['mac_readout'] + acc
    state_access = 2 * c['state_words']
    energy = {}
    for name, (e_mac, e_acc, e_w, e_s) in SCENARIOS.items():
        pj = mac * e_mac + acc * e_acc + weight_reads * e_w + state_access * e_s
        energy[name] = {'uj_per_frame': pj * 1e-6, 'uw': pj * 1e-6 * frames_per_s}
    total_fp32 = mac * ENERGY_PER_MAC_PJ + acc * ENERGY_PER_ACC_PJ
    breakdown = {   # share of fp32 compute energy
        'input_layers': (c['mac_in'] * ENERGY_PER_MAC_PJ + c['acc_in'] * ENERGY_PER_ACC_PJ) / total_fp32,
        'hidden_synapses': (c['mac_syn'] * ENERGY_PER_MAC_PJ + c['acc_syn'] * ENERGY_PER_ACC_PJ) / total_fp32,
        'neuron_updates': c['mac_neuron'] * ENERGY_PER_MAC_PJ / total_fp32,
        'readout': (c['mac_readout'] * ENERGY_PER_MAC_PJ + c['acc_readout'] * ENERGY_PER_ACC_PJ) / total_fp32,
    }
    return {'per_frame': {**c, 'mac': mac, 'acc': acc, 'weight_reads': weight_reads, 'state_accesses': state_access},
            'per_second': {'mac': mac * frames_per_s, 'acc': acc * frames_per_s,
                           'weight_reads': weight_reads * frames_per_s},
            'energy': energy, 'compute_breakdown': breakdown}


def memory_footprint(model, c):
    day = 0
    if getattr(model, 'day_layers', None) is not None:
        day = sum(p.numel() for p in model.day_layers.parameters())
        per_session = sum(p.numel() for p in model.day_layers[0].parameters())
    elif hasattr(model, 'day_gain'):
        day, per_session = model.day_gain.numel(), model.day_gain.shape[1]
    elif hasattr(model, 'inp'):
        day = sum(p.numel() for p in model.inp.parameters())
        per_session = sum(p.numel() for p in model.inp.layers[0].parameters())
    elif hasattr(model, 'per_session_params'):                 # per-session classic filters: one session's
        per_session = model.per_session_params()
        day = sum(p.numel() for p in model.parameters())
    else:
        per_session = 0
    total = sum(p.numel() for p in model.parameters())
    deployed = total - day + per_session
    kernel, _ = frame_shape(model)
    n_feat = getattr(model, 'num_input_channels', None) or \
        (model.inp.layers[0].in_features if hasattr(model, 'inp') else 0)
    buffer_words = kernel * n_feat if kernel > 1 and getattr(model, 'input_buffer', True) else 0
    state_words = c['state_words'] / max(int(getattr(model, 'substeps', 1)), 1)
    delay_bits, history_words = delay_memory(model)
    buffer_words += history_words
    return {'params_total_all_sessions': total, 'params_deployed': deployed,
            'weights_mb': {b: deployed * b / 8 / 1e6 for b in (32, 16, 8, 4)},
            'delay_table_mb': delay_bits / 8 / 1e6,
            'state_words': state_words, 'input_buffer_words': buffer_words,
            'state_kb_fp32': (state_words + buffer_words) * 4 / 1e3}


def delay_memory(model):
    """Synaptic delays: the delay stored with each synapse (bits), and the past the delays read from (words).
    Per-synapse masks (D + 1, out, in) store one delay of ceil(log2(D + 1)) bits per synapse; per-neuron delays
    one per neuron. History: the input layer keeps D past input values per channel (continuous, one word each;
    the same convention as a window's input buffer); spiking layers keep D past steps of spikes (one word per
    neuron per step here, an upper bound: a count of at most 3 needs 2 bits)."""
    bits, words = 0, 0
    def table(masks):                                          # (D + 1, out, in): one delay per synapse
        return masks[0].numel() * max(1, math.ceil(math.log2(masks.shape[0])))
    for name, buf in model.named_buffers():
        leaf = name.split('.')[-1]
        if leaf in ('in_syn_masks', 'rec_syn_masks', 'ff_syn_masks'):
            bits += table(buf)
        elif leaf == 'in_extra_masks':                          # (M - 1, D + 1, out, in)
            bits += buf.shape[0] * buf[0, 0].numel() * max(1, math.ceil(math.log2(buf.shape[1])))
        elif leaf == 'rec_delay_masks':                         # per neuron: (n_values, width)
            bits += buf.shape[1] * max(1, math.ceil(math.log2(buf.shape[0])))
    from snnDecoder.delays import delay_bits, delay_modules    # learned delays: whole-step, as deployed
    bits += sum(delay_bits(m) for _, m in delay_modules(model))
    d_in = int(getattr(model, 'syn_delay_in', 0) or 0)
    if d_in:
        words += d_in * int(model.layers[0].ff.in_features)
    for layer in getattr(model, 'layers', []):
        d_rec = int(getattr(layer, 'rec_syn_delay', 0) or 0) or \
            (max(layer.rec_delay_values) if getattr(layer, 'rec_delay_values', None) else 0)
        words += d_rec * int(getattr(layer, 'width', 0))
        d_ff = int(getattr(layer, 'ff_syn_delay', 0) or 0)
        if d_ff:
            words += d_ff * int(layer.ff.in_features)
    return bits, words


# ---------------------------------------------------------------------------------------------- latency
def time_stream(model, trials, n_bins, device='cpu', repeats=3):
    """Median compute time per output: one call per new 20 ms bin, batch 1, state carried."""
    model = copy.deepcopy(model).to(device).eval()
    snn = is_snn(model)
    kernel, stride = frame_shape(model)
    bins = []
    for feats, day in trials:
        bins.append((torch.from_numpy(feats[:max(kernel, n_bins - sum(len(b[0]) for b in bins))]), day))
        if sum(len(b[0]) for b in bins) >= n_bins:
            break

    def run():
        outputs = 0
        for feats, day in bins:
            if snn:
                model.reset_stream()
                for t in range(len(feats)):
                    if model.step(feats[t:t + 1], day) is not None:
                        outputs += 1
            elif hasattr(model, 'stream_step'):              # LSTM, QRNN, Wiener, Kalman: one bin at a time
                model.reset_stream()
                for t in range(len(feats)):
                    if model.stream_step(feats[t:t + 1], day) is not None:
                        outputs += 1
            else:
                h, buf = None, []
                for t in range(len(feats)):
                    buf = (buf + [model.inp(feats[t:t + 1], day)])[-kernel:]
                    if len(buf) == kernel and (t + 1 - kernel) % stride == 0:
                        frame = torch.stack(buf, 2).reshape(1, 1, -1) if kernel > 1 else buf[-1][:, None]
                        y, h = model.gru(frame, h)
                        model.out(y[:, -1])
                        outputs += 1
        return outputs

    with torch.no_grad():
        run()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            n_out = run()
            times.append(time.perf_counter() - t0)
    n_in = sum(len(b[0]) for b in bins)
    t = float(np.median(times))
    k = int(getattr(model, 'out_stride', 1))      # a GRU stepping every bin, outputting every k bins: k steps per output
    if hasattr(model, 'stream_step'):             # these count their real outputs
        k = 1
    return {'s_per_output': k * t / max(n_out, 1), 's_per_bin': t / n_in, 'real_time_factor':
            (t / max(n_out, 1)) / (stride * BIN_S), 'bins_timed': n_in}


def latency_info(model, smooth_sd):
    kernel, stride = frame_shape(model)
    stride = stride * int(getattr(model, 'out_stride', 1))
    depth = len(model.layers) if is_snn(model) else getattr(model, 'depth', None) or model.gru.num_layers
    return {'lookahead_s': snn_lookahead_s(smooth_sd), 'output_interval_s': stride * BIN_S,
            'input_window_s': kernel * BIN_S, 'layers_in_sequence': depth}


# ---------------------------------------------------------------------------------------------- quantization
def quantize(model, bits):
    """Per-row symmetric fake quantization of every weight matrix (2-D parameter)."""
    q = copy.deepcopy(model)
    levels = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in q.named_parameters():
            if p.dim() == 2 and 'weight' in name:
                scale = p.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / levels
                p.copy_(torch.round(p / scale).clamp(-levels, levels) * scale)
    return q


# ---------------------------------------------------------------------------------------------- report
def pareto(rows, cost_key):
    """Labels of rows that no other row beats on both PER and the cost (lower is better on both)."""
    front = []
    for r in rows:
        if r.get('per') is None:
            continue
        beaten = [o['label'] for o in rows if o is not r and o.get('per') is not None
                  and o['per'] <= r['per'] and cost_key(o) <= cost_key(r)
                  and (o['per'] < r['per'] or cost_key(o) < cost_key(r))]
        r.setdefault('beaten_by', {})[cost_key.__name__] = beaten
        if not beaten:
            front.append(r['label'])
    return front


def power_sram(r):
    return r['ops']['energy']['sram_fp32']['uw']


def power_compute(r):
    return r['ops']['energy']['compute_fp32']['uw']


def weights_mb(r):
    return r['memory']['weights_mb'][32]


def write_markdown(rows, fronts, path, notes):
    L = ['# Decoder efficiency', '',
         'Every model under the same accounting (scripts/efficiency.py). Energy is an estimate for hardware '
         'that skips silent neurons; average power = energy per second of speech.', '']
    L += ['## Accuracy and cost', '',
          '| model | PER | params (deployed) | weights fp32 / int8 | MAC/s | ACC/s | power, compute only | '
          'power, + on-chip memory | power, + off-chip weights | power, int8 + on-chip | spikes/neuron/frame |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        e, m, o = r['ops']['energy'], r['memory'], r['ops']['per_second']
        rates = ', '.join(f'{x:.2f}' for x in r['spike_rates']) if r.get('spike_rates') else '-'
        if r.get('input_event_rate') is not None:
            rates = f"input {r['input_event_rate']:.2f} per channel per bin; {rates}"
        per = f"{r['per']:.3f}" if r.get('per') is not None else 'n/a'
        L.append(f"| {r['label']} | {per}{r.get('per_note', '')} | {m['params_deployed'] / 1e6:.1f}M | "
                 f"{m['weights_mb'][32]:.0f} / {m['weights_mb'][8]:.0f} MB"
                 f"{f' (+{dt:.1f} MB delays)' if (dt := m.get('delay_table_mb', 0)) else ''} | {o['mac'] / 1e6:.0f}M | "
                 f"{o['acc'] / 1e6:.1f}M | {e['compute_fp32']['uw'] / 1e3:.2f} mW | {e['sram_fp32']['uw'] / 1e3:.2f} mW | "
                 f"{e['dram_fp32']['uw'] / 1e3:.1f} mW | {e['sram_int8']['uw'] / 1e3:.2f} mW | {rates} |")
    L += ['', '## Where the compute energy goes (fp32, compute only)', '',
          '| model | input layers | hidden synapses | neuron updates | readout |', '|---|---|---|---|---|']
    for r in rows:
        b = r['ops']['compute_breakdown']
        L.append(f"| {r['label']} | {b['input_layers']:.0%} | {b['hidden_synapses']:.0%} | "
                 f"{b['neuron_updates']:.0%} | {b['readout']:.0%} |")
    L += ['', '## Latency', '',
          '| model | lookahead | output every | input window | layers in sequence | compute per output (1 CPU thread) | '
          'real-time factor |', '|---|---|---|---|---|---|---|']
    for r in rows:
        la, t = r['latency'], r.get('timing')
        tt = f"{t['s_per_output'] * 1e3:.2f} ms" if t else '-'
        rtf = f"{t['real_time_factor']:.3f}" if t else '-'
        L.append(f"| {r['label']} | {la['lookahead_s'] * 1e3:.0f} ms | {la['output_interval_s'] * 1e3:.0f} ms | "
                 f"{la['input_window_s'] * 1e3:.0f} ms | {la['layers_in_sequence']} | {tt} | {rtf} |")
    if any(r.get('quantized') for r in rows):
        bits = sorted({b for r in rows for b in (r.get('quantized') or {})}, reverse=True)
        L += ['', '## Post-training weight quantization (test PER)', '',
              '| model | fp32 | ' + ' | '.join(f'int{b}' for b in bits) + ' |',
              '|---|---|' + '---|' * len(bits)]
        for r in rows:
            qd = r.get('quantized')
            if not qd:
                continue
            L.append(f"| {r['label']} | {r['per']:.3f} | " + ' | '.join(
                f"{qd[b]['per']:.3f}" + (f" ({qd[b]['spike_rate']:.2f} sp)" if qd[b].get('spike_rate') is not None else '')
                if b in qd else '-' for b in bits) + ' |')
    L += ['', '## Pareto fronts (no other model has both lower PER and lower cost)', '']
    for k, v in fronts.items():
        L.append(f"- **{k}**: {', '.join(v)}")
    L += ['', '## Notes', ''] + [f'- {n}' for n in notes]
    open(path, 'w').write('\n'.join(L) + '\n')


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [(lambda r: power_sram(r) / 1e3, 'average power, fp32 + on-chip memory (mW)'),
              (lambda r: weights_mb(r), 'weights, fp32 (MB)'),
              (lambda r: r['timing']['s_per_output'] * 1e3 if r.get('timing') else None,
               'compute per output, 1 CPU thread (ms)')]
    colors = {'snn': 'tab:orange', 'gru': 'tab:blue'}
    for ax, (fx, label) in zip(axes, panels):
        for fam in sorted({r['family'] for r in rows}):
            pts = [(fx(r), r['per'], r['label']) for r in rows if r['family'] == fam
                   and r.get('per') is not None and fx(r) is not None]
            if not pts:
                continue
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=colors.get(fam), label=fam, zorder=3)
            for x, y, t in pts:
                ax.annotate(t, (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel('test PER (greedy)')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True)
    p.add_argument('--data-dir', required=True, help='default data directory (derived/snnData)')
    p.add_argument('--out', required=True, help='output prefix')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='for evaluation')
    p.add_argument('--quant-bits', type=int, nargs='*', default=[8, 6, 4])
    p.add_argument('--timing-bins', type=int, default=500, help='20 ms bins timed per model; 0 = skip')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows, notes = [], []
    for run in parse_runs(args.runs):
        print(f"== {run['label']} ({run['kind']})", flush=True)
        info = load_run(run, args.data_dir, device)
        model = info['model']
        _, stride = frame_shape(model)
        frames_per_s = 1.0 / (stride * BIN_S)
        family = run['opts'].get('family', 'snn' if is_snn(model) else 'gru')
        row = {'label': run['label'], 'kind': run['kind'], 'path': run['path'], 'family': family,
               'arch': info['arch'], 'latency': latency_info(model, info['smooth_sd'])}
        parts = None
        if info['sessions'] and (is_snn(model) or info.get('evaluate')):
            day_indices = list(range(len(info['sessions'])))
            parts = load_partitions(info['data_dir'], info['sessions'], 'test', day_indices)
        if is_snn(model):
            ev = evaluate(model, parts, info['smooth_sd'], device, args.batch_size)
            row['per'] = ev['per']
            row['spike_rates'] = [s / ev['frames'] / w for s, w in zip(ev['layer_spikes'], model.layer_widths)]
            row['input_event_rate'] = ev['input_event_rate']
            counts = snn_counts(model, ev)
        else:
            if info.get('evaluate'):
                ev = evaluate(model, parts, info['smooth_sd'], device, args.batch_size)
                row['per'] = ev['per']
                row['history_per'] = info['history_per']
            else:
                row['per'] = info['history_per']
                if run['kind'] == 'gru':
                    row['per_note'] = ' (history)'
                    if info.get('epochs_planned') and info['epochs_done'] < info['epochs_planned']:
                        row['per_note'] = f" (epoch {info['epochs_done']}/{info['epochs_planned']})"
                        notes.append(f"{run['label']}: still training ({info['epochs_done']} of "
                                     f"{info['epochs_planned']} epochs); its PER is not final.")
                else:
                    row['per_note'] = ' (given)'
            counts = gru_counts(model)
        row['ops'] = summarize_counts(counts, frames_per_s)
        row['memory'] = memory_footprint(model, counts)
        print(f"   PER {row['per']:.4f} | {row['memory']['params_deployed'] / 1e6:.1f}M params | "
              f"{row['ops']['energy']['sram_fp32']['uw'] / 1e3:.2f} mW (fp32 + on-chip)"
              + (f" | spikes/neuron/frame {[round(r, 3) for r in row['spike_rates']]}" if row.get('spike_rates') else ''),
              flush=True)

        if args.timing_bins:
            if parts is not None:
                trials = [(f, part.day_idx) for part in parts for f in part.features]
            elif info['data_dir'] and info['sessions']:
                trials = [(f, 0) for part in load_partitions(info['data_dir'], info['sessions'][:1], 'test')
                          for f in part.features]
            else:   # a shape only: random inputs of the right width (compute time does not depend on values)
                n_feat = model.inp.layers[0].in_features
                trials = [(np.random.randn(args.timing_bins, n_feat).astype(np.float32), 0)]
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            row['timing'] = time_stream(model, trials, args.timing_bins)
            torch.set_num_threads(threads)
            print(f"   {row['timing']['s_per_output'] * 1e3:.2f} ms per output on 1 CPU thread", flush=True)

        if args.quant_bits and parts is not None:
            row['quantized'] = {}
            for bits in args.quant_bits:
                qm = quantize(model, bits)
                ev = evaluate(qm, parts, info['smooth_sd'], device, args.batch_size)
                entry = {'per': ev['per']}
                if is_snn(qm):
                    entry['spike_rate'] = float(sum(ev['layer_spikes']) / ev['frames'] / sum(qm.layer_widths))
                row['quantized'][bits] = entry
                print(f"   int{bits}: PER {ev['per']:.4f}", flush=True)
        rows.append(row)
        del model, info

    fronts = {'PER vs power (fp32 + on-chip memory)': pareto(rows, power_sram),
              'PER vs power (compute only)': pareto(rows, power_compute),
              'PER vs weight memory': pareto(rows, weights_mb)}
    notes += ['PER: greedy CTC phoneme error rate on the test partition of each model\'s sessions; rows marked '
              '(history) or (given) were not re-evaluated.',
              'GRU op counts are exact for its shape; SNN synaptic ACCs use the measured spikes on the test set.',
              'Per-session input layers: one session\'s counted in the deployed size and energy; the native '
              'SNN\'s per-session gains fold into its first layer\'s weights.',
              'Activation traffic between layers (spike events, GRU hidden vectors) is not counted.']
    with open(args.out + '.json', 'w') as f:
        json.dump({'rows': rows, 'pareto': fronts, 'scenarios': {k: dict(zip(('mac_pj', 'acc_pj', 'weight_word_pj',
                   'state_word_pj'), v)) for k, v in SCENARIOS.items()}}, f, indent=1, default=str)
    write_markdown(rows, fronts, args.out + '.md', notes)
    try:
        plot(rows, args.out + '_pareto.png')
    except Exception as e:   # plotting is optional
        print(f'[plot skipped] {e}')
    print(f"wrote {args.out}.json, .md, _pareto.png")
    print(open(args.out + '.md').read())


if __name__ == '__main__':
    main()
