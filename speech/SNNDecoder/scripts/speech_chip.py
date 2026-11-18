"""
The speech Speck SNN run as it would be deployed: the host turns each 20 ms bin into 0/1 events and keeps the delayed
copies, the spiking layers run on a Speck2f devkit, and the host reads their spikes into the leaky output neurons,
one phoneme output every 80 ms, decoded by greedy CTC.

A model of several chains (--branches, e.g. round 14's N2: one chain on the threshold-crossing counts, one on the
spike-band powers) runs as that many independent networks, one Speck2f each (a chain's 512 x 128 first layer and
256 x 256 third layer need both of the chip's 64K-weight cores); the host gives each chain its share of the input
lines and puts their spikes back together for the readout.

The per-day input gain only scales each channel before its threshold, on the host, so one chip configuration
serves every session. Modes, each one step closer to the chip:

    torch    the trained model's own forward pass (the reference phoneme error)
    stack    host events -> the spiking layers rebuilt as Linear + IAF (float) -> host readout; equals torch
    quant    the chip's 8-bit weights and thresholds (DynapcnnNetwork), a step's input summed, and no cap on a
             neuron's spikes per step (training capped it at 3; the chip has no steps)
    events   8-bit weights, each input event applied on its own as on the chip, hard or soft reset as trained
             (slow: use --max-trials)
    chip     the Speck2f devkit: phoneme error, mean power, wall time per 20 ms bin

    python speech_chip.py --ckpt N2/best_model_weights.pth --data-dir snnData --mode stack --max-trials 5
    python speech_chip.py ... --mode events --scales 1 1.5 2 3 4 --max-trials 3    pick layer 1's threshold scale
    python speech_chip.py ... --mode chip --scale 2 --out results/chip.json

--scales (events mode) reports, per layer 1 threshold scale, the phoneme error and the correlation of the logits
with the trained model's (the label-free choice). Every mode also reports that correlation by quarter of each
sentence (does the chip drift from the trained network the longer it runs?). Results: per-session and overall
phoneme error, and for the chip power, latency and energy per 80 ms output (static power included), in --out.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.inference import load_snn_model  # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode  # noqa: E402

import speck_io  # noqa: E402  (snn-decoders snndec/chip.py: discretized, run_software, Devkit)


@torch.no_grad()
def host_events(model, x, day_idx):
    """One trial's features (T, C) -> the chip's input: 0/1 events (T, n_in x delays), delayed copies side by side."""
    u = model.encoder(model._input(torch.as_tensor(x, dtype=torch.float32)[None], day_idx))
    return (model.memory.delay(u)[0].numpy() > 0).astype(np.float32)


def chains(model):
    """The model's independent chains: per chain, its input lines and, per layer, its neurons (indices into the
    full layer). One chain for an ordinary model."""
    K = int(getattr(model, 'branches', 1) or 1)
    w0 = model.layers[0].ff.weight.detach()
    W = getattr(model, 'branch_widths', None) or [[l.width // K] * K for l in model.layers]
    out = []
    for k in range(K):
        units = [np.arange(sum(w[:k]), sum(w[:k + 1])) for w in W]
        inputs = np.flatnonzero((w0[units[0]] != 0).any(0).numpy()) if K > 1 else np.arange(w0.shape[1])
        out.append((inputs, units))
    return out


def spiking_stack(model, layer1_scale=1.0, as_trained=False, chain=None):
    """The model's spiking layers (or one chain's: chain = (inputs, units) from chains()) as nn.Sequential(Linear,
    IAFSqueeze, ...) with the chip's rules: no bias, no leak, the trained reset (subtract the threshold, or zero),
    membrane floored at min_vmem. layer1_scale multiplies layer 1's threshold and floor (the chip applies a step's
    events one at a time). Training caps a neuron's spikes per step (MaxSpike, 3); the chip has no steps and no cap,
    so the chip's stack fires as often as it crosses (MultiSpike); as_trained keeps the cap, to check the stack."""
    import sinabs.activation as sa
    import sinabs.layers as sl
    mods = []
    for k, l in enumerate(model.layers):
        if l.rec is not None or l.neuron_type != 'iaf' or l.log_tau_syn is not None or l.tau_syn is not None \
                or l.adapt_mask is not None or l.log_tau_mem is not None:
            raise ValueError('only a feedforward integrate-and-fire network without synaptic filters maps to Speck')
        w = l.ff.weight.detach()
        if chain is not None:
            inputs, units = chain
            w = w[units[k]][:, inputs if k == 0 else units[k - 1]]
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(w)
        s = layer1_scale if k == 0 else 1.0
        base = l.threshold if l.log_threshold is None else l.log_threshold.detach().exp()  # learned: one per layer
        if torch.as_tensor(base).numel() != 1:
            raise ValueError('a threshold per neuron does not map to Speck (one per layer)')
        thr = float(base) * s
        floor = None if l.min_vmem is None else float(l.min_vmem) * s
        mods += [lin, sl.IAFSqueeze(batch_size=1, spike_threshold=thr, min_v_mem=floor,
                                    spike_fn=l.spike_fn if as_trained else sa.MultiSpike,
                                    reset_fn=sa.MembraneSubtract() if l.reset_type == 'soft' else sa.MembraneReset())]
    return nn.Sequential(*mods)


@torch.no_grad()
def run_stack(seq, events, all_layers=False):
    """Float layers, a step's input summed (as in training): (T, n_in) -> last layer's spikes (T, n), or every
    layer's side by side (all_layers)."""
    speck_io.reset_states(seq)
    x = torch.as_tensor(events)
    rows = []
    for t in range(len(x)):
        h, outs = x[t:t + 1], []
        for m in seq:
            h = m(h)
            if not isinstance(m, nn.Linear):
                outs.append(h.reshape(1, -1))
        rows.append(torch.cat(outs, -1) if all_layers else outs[-1])
    return torch.cat(rows).numpy()


def run_events(network, events, threshold_scale=1.0, all_layers=False):
    """The chip's event-by-event dynamics: within a step the input events (address order) reach layer 1 one at a
    time; each adds its 8-bit weights; with soft reset every neuron at or above threshold fires once per threshold it
    holds and subtracts it, with hard reset it fires once and goes to zero; the membrane is floored, and the spikes
    reach the next layer in firing order."""
    import sinabs.activation as sa
    seq = speck_io.quantized_layers(network)
    W = [m.weight.detach().numpy()[:, :, 0, 0] for m in seq if hasattr(m, 'weight')]
    neurons = [m for m in seq if not hasattr(m, 'weight')]
    soft = [isinstance(getattr(n, 'reset_fn', None), sa.MembraneSubtract) for n in neurons]
    scale = [threshold_scale] + [1.0] * (len(W) - 1)
    thr = [float(n.spike_threshold) * k for n, k in zip(neurons, scale)]
    floor = [float(n.min_v_mem) * k if n.min_v_mem is not None else -np.inf for n, k in zip(neurons, scale)]
    v = [np.zeros(w.shape[0]) for w in W]
    offsets = np.cumsum([0] + [w.shape[0] for w in W])
    out = np.zeros((len(events), offsets[-1]))
    for t, row in enumerate(events):
        incoming = np.flatnonzero(row).tolist()
        for li, w in enumerate(W):
            fired = []
            for i in incoming:
                v[li] += w[:, i]
                if soft[li]:
                    n = np.floor(np.maximum(v[li], 0) / thr[li]).astype(int)
                    hot = np.flatnonzero(n)
                    for h in hot:
                        fired.extend([h] * n[h])
                    v[li][hot] -= n[hot] * thr[li]
                else:
                    hot = np.flatnonzero(v[li] >= thr[li])
                    fired.extend(hot.tolist())
                    v[li][hot] = 0.0
                np.maximum(v[li], floor[li], out=v[li])
            incoming = fired
            np.add.at(out[t], np.asarray(fired, dtype=int) + offsets[li], 1)
    return out if all_layers else out[:, offsets[-2]:]


def assemble(model, per_chain, all_layers):
    """Each chain's spikes (its layers side by side, or its last layer) -> the full model's readout input, in the
    model's own neuron order (layer by layer, each layer's neurons chain by chain)."""
    if len(per_chain) == 1:
        return per_chain[0]
    if not all_layers:
        return np.concatenate(per_chain, 1)
    K = len(per_chain)
    W = getattr(model, 'branch_widths', None) or [[l.width // K] * K for l in model.layers]
    offs = [np.cumsum([0] + [W[i][k] for i in range(len(W))]) for k in range(K)]    # within chain k's own output
    return np.concatenate([per_chain[k][:, offs[k][i]:offs[k][i + 1]] for i in range(len(W)) for k in range(K)], 1)


@torch.no_grad()
def readout(model, spikes):
    """The readout's input spikes (T, n), one row per 20 ms bin -> logits, one row per output (every output_stride
    bins): the host's traces, the readout's weights and the leaky output neurons, as model._step_all."""
    alphas = model.readout_alphas or [0.0]
    a = model._out_alpha()
    tr = torch.zeros(len(alphas), spikes.shape[1])
    out_v = torch.zeros(model.out_stages, model.n_classes)
    outs = []
    for s in torch.as_tensor(spikes, dtype=torch.float32):
        if model.readout_traces:
            tr = torch.stack([s if al == 0.0 else al * y + (1 - al) * s for al, y in zip(alphas, tr)])
            r = tr.reshape(-1)
        else:
            r = s
        prev = model.readout(r)
        stages = []
        for k in range(model.out_stages):
            prev = a * out_v[k] + (1 - a) * prev
            stages.append(prev)
        out_v = torch.stack(stages)
        outs.append(out_v[-1])
    out = torch.stack(outs)
    k = model._ostride
    return (out[k - 1::k] if k > 1 else out).numpy()


@torch.no_grad()
def reference_logits(model, x, day_idx):
    out = model(torch.as_tensor(x, dtype=torch.float32)[None], day_idx)[0]
    return out.numpy()


def quarters(a, b):
    """Correlation of two (T, classes) logit arrays in each quarter of the sentence (NaN if a quarter is too short)."""
    q = len(a) // 4
    if q < 2:
        return [float('nan')] * 4
    return [float(np.corrcoef(a[i * q:(i + 1) * q].ravel(), b[i * q:(i + 1) * q].ravel())[0, 1]) for i in range(4)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-dir', required=True, help='folder of <session>/<partition>.npz')
    p.add_argument('--mode', choices=['torch', 'stack', 'quant', 'events', 'chip'], default='stack')
    p.add_argument('--sessions', nargs='+', default=None, help='default: every session of the checkpoint')
    p.add_argument('--max-trials', type=int, default=None, help='per session')
    p.add_argument('--partition', default='test', help="'train' to choose layer 1's scale (never on test sentences)")
    p.add_argument('--scale', type=float, default=1.0, help="layer 1's threshold scale on the chip")
    p.add_argument('--scales', type=float, nargs='+', default=None, help='events mode: sweep these scales')
    p.add_argument('--wait', type=float, default=0.003, help='chip: seconds per 20 ms bin before reading output')
    p.add_argument('--reset-settle', type=float, default=0.3, help='chip: seconds to settle after a reset')
    p.add_argument('--reset-used', action='store_true', help="chip: reset only each core's used neurons, not its "
                   "whole neuron memory (cores 0-2 hold 64K)")
    p.add_argument('--devkit', default='speck2fdevkit:0')
    p.add_argument('--save-counts', default=None, help="chip: .npz of every sentence's spikes as the readout reads "
                   "them (keyed '<session>/<trial>'), to rescore with another readout (refit_readout.py apply)")
    p.add_argument('--readout', default=None, help="a host readout refitted to the chip's firing on training "
                   "sentences (refit_readout.py -> readout_refit.pt): replaces the trained readout's weights and "
                   "output time constants, so every mode decodes with it live")
    p.add_argument('--one-chip', action='store_true', help='chip: all chains on one Speck2f at once (MultiDevkit), '
                   'for chains small enough to share a chip; default: one chip per chain, measured chain by chain')
    p.add_argument('--out', default=None)
    p.add_argument('--save-logits', default=None, help='.npz of every sentence\'s logits (first scale), keyed '
                   '<session>/<trial>: for the language model and the video')
    a = p.parse_args()

    torch.set_num_threads(1)
    model, ck = load_snn_model(a.ckpt)
    if a.readout:                      # the deployed readout: refitted to the chip's own firing (training data only)
        r = torch.load(a.readout, map_location='cpu')
        with torch.no_grad():
            model.readout.weight.copy_(r['lin.weight'])
            model.readout.bias.copy_(r['lin.bias'])
            model.out_log_tau.copy_(r['out_log_tau'])
        print(f'readout: {a.readout} (refitted to the chip)')
    model_sessions = list(ck['sessions'])
    sessions = a.sessions or model_sessions
    parts = load_partitions(a.data_dir, sessions, a.partition, [model_sessions.index(s) for s in sessions])
    blank = model.n_classes - 1
    ALL = bool(getattr(model, 'readout_all', False))      # the readout reads every layer: the chip reports them all
    CH = chains(model)
    if ALL and a.mode == 'quant':
        raise SystemExit('quant mode reads the last layer only; use stack, events or chip for a readout_all model')
    print(f"{len(parts)} sessions; {len(CH)} chain(s) of {[len(c[0]) for c in CH]} chip inputs, layers "
          f"{[l.width for l in model.layers]} ({[l.reset_type for l in model.layers][0]} reset), readout of "
          f"{'every layer' if ALL else 'the last layer'}, traces {model.readout_traces}, output every "
          f"{model._ostride} bins; mode {a.mode}")

    scales = a.scales if a.mode == 'events' and a.scales else [a.scale]
    nets = {}
    if a.mode in ('quant', 'events'):
        nets = [speck_io.discretized(spiking_stack(model, 1.0, chain=c), len(c[0])) for c in CH]
    stacks = [spiking_stack(model, a.scale, as_trained=True, chain=c) for c in CH] if a.mode == 'stack' else None
    chip_counts, chip_meas = None, []
    if a.mode == 'chip':
        # One Speck2f holds one chain (each needs both 64K-weight cores, 5 and 6). The chains never interact, so a
        # model of K chains is K chips; measured here on one devkit, chain by chain over every sentence, then the
        # host joins their spikes: the same outputs as K chips side by side. Power: the sum over chips; time per bin:
        # one chip's (they run in parallel).
        speck_io.RESET_SETTLE_S = a.reset_settle
        speck_io.RESET_USED_ONLY = a.reset_used
        chip_counts = [dict() for _ in CH]
        if a.one_chip and len(CH) > 1:
            # Small chains (e.g. round 19's 512-128-256-64 x 2) fit one Speck2f together, each on its own cores:
            # all chains run at once on one chip, so power and time per bin are measured as deployed.
            dev = speck_io.MultiDevkit([spiking_stack(model, a.scale, chain=c) for c in CH], [len(c[0]) for c in CH],
                                       devkit=a.devkit, wait_time=a.wait, all_layers=ALL)
            s_k, bins_k, j_k, ev_k, t_k = 0.0, 0, 0.0, 0, time.time()
            for part in parts:
                idx = range(len(part)) if a.max_trials is None else range(min(a.max_trials, len(part)))
                for i in idx:
                    ev = host_events(model, part.features[i], part.day_idx)
                    dev.reset()
                    counts, loop_s, power_w, n_ev = dev.run([ev[:, c[0]] for c in CH])
                    for k in range(len(CH)):
                        chip_counts[k][(part.session, i)] = np.minimum(counts[k], 255).astype(np.uint8)
                    s_k, bins_k, j_k, ev_k = s_k + loop_s, bins_k + len(ev), j_k + power_w * loop_s, ev_k + n_ev
                print(f"  {len(CH)} chains on one chip, {part.session} done ({time.time() - t_k:.0f} s)", flush=True)
            dev.close()
            chip_meas.append({'power_mw': j_k / s_k * 1e3, 'ms_per_bin': s_k / bins_k * 1e3,
                              'input_events_per_bin': ev_k / bins_k, 'cores': dev.cores})
            print(f"  one chip: {j_k / s_k * 1e3:.2f} mW, {s_k / bins_k * 1e3:.2f} ms per bin", flush=True)
        for k, c in enumerate(CH if not (a.one_chip and len(CH) > 1) else []):
            dev = speck_io.Devkit(spiking_stack(model, a.scale, chain=c), len(c[0]), devkit=a.devkit,
                                  wait_time=a.wait, all_layers=ALL)
            s_k, bins_k, j_k, ev_k, t_k = 0.0, 0, 0.0, 0, time.time()
            for part in parts:
                idx = range(len(part)) if a.max_trials is None else range(min(a.max_trials, len(part)))
                for i in idx:
                    ev = host_events(model, part.features[i], part.day_idx)[:, c[0]]
                    dev.reset()
                    counts, loop_s, power_w, n_ev = dev.run(ev, c[1][-1].size)
                    chip_counts[k][(part.session, i)] = np.minimum(counts, 255).astype(np.uint8)
                    s_k, bins_k, j_k, ev_k = s_k + loop_s, bins_k + len(ev), j_k + power_w * loop_s, ev_k + n_ev
                print(f"  chain {k + 1}/{len(CH)}, {part.session} done ({time.time() - t_k:.0f} s)", flush=True)
            dev.close()
            chip_meas.append({'power_mw': j_k / s_k * 1e3, 'ms_per_bin': s_k / bins_k * 1e3,
                              'input_events_per_bin': ev_k / bins_k, 'cores': dev.cores})
            print(f"  chain {k + 1}: {j_k / s_k * 1e3:.2f} mW, {s_k / bins_k * 1e3:.2f} ms per bin", flush=True)

    res = {s: {'edits': {}, 'lengths': {}, 'corr': [], 'quarters': []} for s in scales}
    t0 = time.time()
    saved = {}
    for part in parts:
        idx = range(len(part)) if a.max_trials is None else range(min(a.max_trials, len(part)))
        for i in idx:
            x, target = part.features[i], part.labels[i]
            ref = reference_logits(model, x, part.day_idx)
            ev = host_events(model, x, part.day_idx) if a.mode != 'torch' else None
            evs = [ev[:, c[0]] for c in CH] if ev is not None else None
            for s in scales:
                if a.mode == 'torch':
                    logits = ref
                elif a.mode == 'stack':
                    logits = readout(model, assemble(model, [run_stack(st, e, ALL) for st, e in zip(stacks, evs)], ALL))
                elif a.mode == 'quant':
                    logits = readout(model, assemble(model, [speck_io.run_software(n, e) for n, e in zip(nets, evs)], ALL))
                elif a.mode == 'events':
                    logits = readout(model, assemble(model, [run_events(n, e, s, ALL) for n, e in zip(nets, evs)], ALL))
                else:
                    counts = [cc[(part.session, i)].astype(np.float32) for cc in chip_counts]
                    logits = readout(model, assemble(model, counts, ALL))
                n = min(len(logits), len(ref))
                if a.save_logits and s == scales[0]:
                    saved[f'{part.session}/{i}'] = logits[:n].astype(np.float32)
                dec = greedy_decode(logits[:n], blank)
                r = res[s]
                r['edits'][part.session] = r['edits'].get(part.session, 0) + edit_distance(dec, target)
                r['lengths'][part.session] = r['lengths'].get(part.session, 0) + len(target)
                r['corr'].append(float(np.corrcoef(logits[:n].ravel(), ref[:n].ravel())[0, 1]))
                r['quarters'].append(quarters(logits[:n], ref[:n]))
        done = {s: sum(res[s]['edits'].values()) / max(sum(res[s]['lengths'].values()), 1) for s in scales}
        print(f"  {part.session}: " + ', '.join(f"x{s:g} PER so far {v:.4f}" for s, v in done.items())
              + f"  ({time.time() - t0:.0f} s)", flush=True)

    out = {'ckpt': os.path.abspath(a.ckpt), 'readout': a.readout, 'mode': a.mode, 'max_trials': a.max_trials, 'chains': len(CH),
           'results': {}}
    for s in scales:
        r = res[s]
        per_session = {k: r['edits'][k] / r['lengths'][k] for k in r['edits']}
        q = np.nanmean(np.array(r['quarters']), 0).tolist()
        out['results'][f'{s:g}'] = {
            'per': sum(r['edits'].values()) / sum(r['lengths'].values()),
            'per_session': per_session,
            'mean_session_per': float(np.mean(list(per_session.values()))),
            'logit_corr_vs_trained': float(np.mean(r['corr'])),
            'logit_corr_by_quarter': q,
        }
        print(f"layer 1 threshold x{s:g}: PER {out['results'][f'{s:g}']['per']:.4f} "
              f"(mean over sessions {out['results'][f'{s:g}']['mean_session_per']:.4f}), "
              f"logits vs trained r {np.mean(r['corr']):.4f}; by quarter of the sentence "
              + ' '.join(f'{v:.3f}' for v in q))
    if a.mode == 'chip' and chip_meas:
        p_mw = sum(m['power_mw'] for m in chip_meas)                   # one chip per chain, all running
        ms = max(m['ms_per_bin'] for m in chip_meas)
        out['chip'] = {'chips': len(chip_meas), 'per_chip': chip_meas, 'power_mw': p_mw, 'ms_per_bin': ms,
                       'energy_uj_per_output_80ms': p_mw * 1e-3 * 0.08 * 1e6,
                       'input_events_per_bin': sum(m['input_events_per_bin'] for m in chip_meas),
                       'wait_s': a.wait, 'devkit': a.devkit}
        print(f"chip: {len(chip_meas)} chip(s), {p_mw:.2f} mW in all, {ms:.2f} ms per 20 ms bin, "
              f"{p_mw * 0.08 * 1e3:.0f} µJ per 80 ms output")
    if a.save_counts and chip_counts:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_counts)), exist_ok=True)
        keys = chip_counts[0].keys()
        np.savez_compressed(a.save_counts, **{f'{s}/{i}': np.minimum(assemble(model, [cc[(s, i)].astype(np.float32)
                            for cc in chip_counts], ALL), 255).astype(np.uint8) for s, i in keys})
        print('wrote', a.save_counts, f'({len(keys)} sentences)')
    if a.save_logits:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_logits)), exist_ok=True)
        np.savez_compressed(a.save_logits, **saved)
        print('wrote', a.save_logits, f'({len(saved)} sentences)')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
