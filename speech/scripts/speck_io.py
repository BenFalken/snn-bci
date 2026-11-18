"""
Running a Speck network on a Speck2f devkit: the spiking layers on the chip, everything else on the host, as in
spike_bmi's inference/speck.py (whose device calls this follows: the samna graph, the per-core configuration, the
membrane reset, the PowerMonitor at 100 Hz).

    seq = linear_stack(model)                     the spiking layers as Linear + IAF, input gain folded into layer 1
    events = chip_events(model, model.input_events[0])        0/1 events, one row per model step
    emu = discretized(seq, model.host.n_out)      the same network with the chip's 8-bit weights, in software
    dev = Devkit(seq, model.host.n_out, wait_time=0.003)
    counts, seconds, power_w, n_events = dev.run(events)      events (T, n_inputs) 0/1 -> last layer's spikes (T, n)

Each model step's input is written to the chip as events; after wait_time the last layer's output spikes are
counted per neuron. Latency is wall time per step of that loop (wait included); power is the PowerMonitor's mean
over the same loop, so energy per step = power x latency. samna is imported only by Devkit.
"""

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

POWER_SAMPLE_RATE_HZ = 100
RESET_SETTLE_S = 1.0
RESET_USED_ONLY = False      # True: reset only each core's used neurons (its layer width), not its whole neuron
                             # memory (cores 0-2 hold 64K neurons: writing them all can still be draining when a
                             # sentence starts, delaying the chip's first 1-2 s of spikes)


def input_gain(model) -> torch.Tensor:
    """Each chip input's fixed gain: the host's per-channel input scale, repeated for every delayed copy (the copies
    are concatenated copy by copy). On the chip the input is events, so this gain lives in the first layer's
    weights."""
    h = model.host
    if h.day_gain is not None:
        raise ValueError('a per-day input gain cannot be folded into fixed chip weights')
    scale = h.input_scale if h.input_scale is not None else torch.ones(h.n_in)
    return scale.repeat(len(h.delays))


def chip_events(model, input_events) -> np.ndarray:
    """What the host sends the chip: model.input_events (event x input gain) as 0/1 events."""
    return (np.asarray(input_events) > 0).astype(np.float32)


def linear_stack(model, layer1_scale: float = 1.0) -> nn.Sequential:
    """The model's spiking layers as nn.Sequential(Linear, IAFSqueeze, ...), weights copied, the input gain folded
    into the first layer (so the chip takes plain 0/1 events). layer1_scale multiplies layer 1's threshold and floor:
    the chip fires on each input event rather than on a step's summed input, and with dense input a higher first
    threshold restores what training saw (run_events simulates this; the scale is chosen on validation data)."""
    import sinabs.activation as sa
    import sinabs.layers as sl
    if model.spec['kind'] != 'speck':
        raise ValueError("only a 'speck' network maps onto the chip")
    mods = []
    for k, l in enumerate(model.layers):
        lin = nn.Linear(l.ff.in_features, l.width, bias=False)
        with torch.no_grad():
            lin.weight.copy_(l.ff.weight * input_gain(model)[None, :] if k == 0 else l.ff.weight)
        k_scale = layer1_scale if k == 0 else 1.0
        mods += [lin, sl.IAFSqueeze(batch_size=1, spike_threshold=l.threshold * k_scale,
                                    min_v_mem=None if l.min_vmem is None else l.min_vmem * k_scale,
                                    spike_fn=sa.MultiSpike, reset_fn=sa.MembraneReset())]
    return nn.Sequential(*mods)


def discretized(seq: nn.Sequential, n_inputs: int):
    """The DynapcnnNetwork deployed to the chip (weights and thresholds at the chip's precision); also runs in
    software, (T, n_inputs, 1, 1) -> (T, n_out, 1, 1)."""
    from sinabs.backend.dynapcnn import DynapcnnNetwork
    return DynapcnnNetwork(nn.Sequential(nn.Flatten(), *seq), input_shape=(n_inputs, 1, 1), discretize=True,
                           dvs_input=False)


def quantized_layers(network) -> nn.Sequential:
    """A DynapcnnNetwork's quantized layers as a plain nn.Sequential of Conv2d and IAF layers."""
    if hasattr(network, 'sequence'):                                        # sinabs < 3
        pairs = [(l.conv_layer, l.spk_layer) for l in network.sequence if hasattr(l, 'conv_layer')]
    else:
        layers = network._dynapcnn_module._dynapcnn_layers
        pairs = [(layers[k].conv, layers[k].spk) for k in sorted(layers, key=int)]
    return nn.Sequential(*[m for pair in pairs for m in pair])


def reset_states(seq):
    """sinabs.reset_states, after replacing neuron states a run left as views (sinabs cannot reset those)."""
    import sinabs
    for m in seq.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf._base is not None:
                m._buffers[name] = buf.detach().clone()
    sinabs.reset_states(seq)


@torch.no_grad()
def run_software(network, events: np.ndarray) -> np.ndarray:
    """Step the discretized network one model step at a time from rest, as the chip runs it; returns the last
    layer's spikes (T, n)."""
    seq = quantized_layers(network)
    reset_states(seq)
    x = torch.as_tensor(events, dtype=torch.float32)[:, :, None, None]
    return torch.cat([seq(x[t:t + 1]).reshape(1, -1) for t in range(len(x))]).numpy()


def run_events(network, events: np.ndarray, threshold_scale=(1.0,)) -> np.ndarray:
    """The chip's event-by-event dynamics in software: within each step the input events (in address order) reach
    layer 1 one at a time; each adds its 8-bit weights to the membranes, every neuron at or above threshold fires
    once at that moment and resets to zero (hard reset), the membrane is floored at min_v_mem, and the spikes reach
    the next layer in the order they fired. Unlike run_software (and training), a step's input is never summed
    before firing, so dense input loses the charge above threshold at every early spike. threshold_scale multiplies
    each layer's threshold and floor (one value per layer; missing layers 1.0), as a chip-side correction.
    Returns the last layer's spikes (T, n)."""
    seq = quantized_layers(network)
    W = [m.weight.detach().numpy()[:, :, 0, 0] for m in seq if hasattr(m, 'weight')]
    neurons = [m for m in seq if not hasattr(m, 'weight')]
    scale = list(threshold_scale) + [1.0] * (len(W) - len(threshold_scale))
    thr = [float(n.spike_threshold) * k for n, k in zip(neurons, scale)]
    floor = [float(n.min_v_mem) * k if n.min_v_mem is not None else -np.inf for n, k in zip(neurons, scale)]
    v = [np.zeros(w.shape[0]) for w in W]
    out = np.zeros((len(events), W[-1].shape[0]))
    for t, row in enumerate(events):
        incoming = np.flatnonzero(row)
        for li, w in enumerate(W):
            fired = []
            for i in incoming:
                v[li] += w[:, i]
                hot = np.flatnonzero(v[li] >= thr[li])
                if len(hot):
                    fired.extend(hot.tolist())
                    v[li][hot] = 0.0
                np.maximum(v[li], floor[li], out=v[li])
            incoming = fired
        np.add.at(out[t], incoming, 1)
    return out


def power_from_events(events, loop_s):
    by_channel = defaultdict(list)
    for ev in events:
        by_channel[ev.channel].append(ev.value)
    if not by_channel:
        return float('nan'), False
    expected = loop_s * POWER_SAMPLE_RATE_HZ * len(by_channel)
    return float(sum(np.mean(v) for v in by_channel.values())), abs(len(events) - expected) <= 0.2 * expected


class Devkit:
    """A network deployed on a Speck2f devkit; the last layer's spikes are read back (all_layers: every layer's,
    side by side in layer order, for a readout that reads them all)."""

    def __init__(self, seq, n_inputs, devkit='speck2fdevkit:0', wait_time=0.003, raster_dt=0.1, all_layers=False):
        import samna
        import sinabs.backend.dynapcnn.io as sio
        from sinabs.backend.dynapcnn.chip_factory import ChipFactory
        self.samna, self.wait_time, self.raster_dt = samna, wait_time, raster_dt
        self.all_layers = all_layers
        self.widths = [m.out_features for m in seq if isinstance(m, nn.Linear)]
        self.network = discretized(seq, n_inputs)
        self.device = sio.open_device(devkit)
        self.chip_factory = ChipFactory(devkit.split(':')[0])
        self.power_monitor = self.device.get_power_monitor()
        self.power_sink = samna.BasicSinkNode_unifirm_modules_events_measurement()
        self.input_source = samna.BasicSourceNode_speck2f_event_input_event()
        self.output_sink = samna.BasicSinkNode_speck2f_event_output_event()
        self.graph = samna.graph.EventFilterGraph()
        self.graph.sequential([self.input_source, self.device.get_model_sink_node()])
        self.graph.sequential([self.device.get_model_source_node(), self.output_sink])
        self.graph.sequential([self.power_monitor.get_source_node(), self.power_sink])
        self.graph.start()
        config = self.network.make_config(device=devkit)
        ordering = self.network.chip_layers_ordering
        self.cores = [ordering[k] for k in sorted(ordering)] if isinstance(ordering, dict) else list(ordering)
        config.dvs_layer.pass_sensor_events = False
        for core in self.cores:
            config.cnn_layers[core].monitor_enable = all_layers or core == self.cores[-1]
        self.device.get_model().apply_configuration(config)
        print(f"  [speck] {devkit}: layers on cores {self.cores}")

    def reset(self):
        """Zero every membrane potential and drop pending output and power events."""
        self.output_sink.get_events()
        builder = self.chip_factory.get_config_builder()
        module = builder.get_samna_module()
        for core, width in zip(self.cores, self.widths):
            events = []
            n_addr = builder.get_constraints()[core].neuron_memory
            for address in range(min(n_addr, width) if RESET_USED_ONLY else n_addr):
                ev = module.event.WriteNeuronValue()
                ev.address, ev.layer, ev.neuron_state = address, core, 0
                events.append(ev)
            source = builder.get_input_buffer()
            graph = self.samna.graph.EventFilterGraph()
            graph.sequential([source, self.device.get_model().get_sink_node()])
            graph.start()
            source.write(events)
            time.sleep(RESET_SETTLE_S)
            graph.stop()
        self.power_sink.get_events()

    def run(self, events: np.ndarray, n_out: int):
        """events (T, n_inputs) of 0/1, one row per model step -> (last layer's spike counts (T, n_out), or every
        layer's side by side with all_layers, loop
        seconds, mean power in W, input events sent)."""
        frames = torch.as_tensor(events, dtype=torch.float32)[:, None, :, None, None]
        spike = self.samna.speck2f.event.Spike
        last = self.cores[-1]
        reads = list(zip(self.cores, self.widths)) if self.all_layers else [(last, n_out)]
        offsets = np.cumsum([0] + [w for _, w in reads])
        counts = np.zeros((len(frames), offsets[-1]))
        n_events = 0
        self.power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
        start = time.perf_counter()
        for t, frame in enumerate(frames):
            if frame.sum() >= 1:                                            # 0/1 events (chip_events)
                evs = self.chip_factory.raster_to_events(frame, self.cores[0], dt=self.raster_dt)
                n_events += len(evs)
                self.input_source.write(evs)
            time.sleep(self.wait_time)
            evs = [e for e in self.output_sink.get_events() if isinstance(e, spike)]
            for (core, w), o in zip(reads, offsets):
                feats = [e.feature for e in evs if e.layer == core]
                counts[t, o:o + w] = np.bincount(feats, minlength=w)[:w]
        loop_s = time.perf_counter() - start
        self.power_monitor.stop_auto_power_measurement()
        power_w, ok = power_from_events(self.power_sink.get_events(), loop_s)
        if not ok:
            print("  WARNING: the power sample count is off; this run's power may be unreliable")
        return counts, loop_s, power_w, n_events

    def close(self):
        self.graph.stop()


def layer_summary(layer):
    """What must match after a copy: weights, threshold, floor, destinations."""
    w = np.asarray(layer.weights)
    return (w.shape, float(np.abs(w).sum()), int(layer.threshold_high), int(layer.threshold_low),
            tuple((d.enable, d.layer) for d in layer.destinations))


def copy_layer(dst, src):
    """Copy one core's configuration into another configuration's core, setting by setting (whole-element
    assignment, cfg.cnn_layers[c] = other.cnn_layers[c], does not persist with every samna version: on the devkit
    laptop's samna 0.42 it left the core at its defaults, threshold 0 and no destination)."""
    for name in dir(src):
        if name.startswith('_'):
            continue
        try:
            val = getattr(src, name)
        except Exception:
            continue
        if callable(val):
            continue
        try:
            setattr(dst, name, val)
        except Exception:
            pass


class MultiDevkit:
    """Several independent networks (the chains of a --branches model) on one Speck2f chip at once, each on its own
    cores: one configuration per chain, placed on cores no other chain uses, merged into one chip configuration.
    Each step, every chain's input events go to its first core; the spikes read back are sorted by core into each
    chain's layers. Power and time are those of all chains running together, as deployed."""

    def __init__(self, seqs, n_inputs, devkit='speck2fdevkit:0', wait_time=0.003, raster_dt=0.1, all_layers=False):
        import itertools
        import samna
        import sinabs.backend.dynapcnn.io as sio
        from sinabs.backend.dynapcnn.chip_factory import ChipFactory
        self.samna, self.wait_time, self.raster_dt, self.all_layers = samna, wait_time, raster_dt, all_layers
        self.widths = [[m.out_features for m in s if isinstance(m, nn.Linear)] for s in seqs]
        self.device = sio.open_device(devkit)
        self.chip_factory = ChipFactory(devkit.split(':')[0])
        self.power_monitor = self.device.get_power_monitor()
        self.power_sink = samna.BasicSinkNode_unifirm_modules_events_measurement()
        self.input_source = samna.BasicSourceNode_speck2f_event_input_event()
        self.output_sink = samna.BasicSinkNode_speck2f_event_output_event()
        self.graph = samna.graph.EventFilterGraph()
        self.graph.sequential([self.input_source, self.device.get_model_sink_node()])
        self.graph.sequential([self.device.get_model_source_node(), self.output_sink])
        self.graph.sequential([self.power_monitor.get_source_node(), self.power_sink])
        self.graph.start()
        self.cores, configs, used = [], [], set()
        for seq, n in zip(seqs, n_inputs):
            net = discretized(seq, n)
            n_layers = len(self.widths[len(self.cores)])
            tried, config = 0, None
            for order in itertools.permutations([c for c in range(9) if c not in used], n_layers):
                tried += 1
                try:
                    config = net.make_config(chip_layers_ordering=list(order), device=devkit)
                    break
                except Exception:
                    continue
            if config is None:
                raise RuntimeError(f'chain {len(self.cores)}: no free cores fit its layers ({tried} orderings tried)')
            self.cores.append(list(order))
            used.update(order)
            configs.append(config)
        merged = configs[0]
        for cfg, cores in zip(configs[1:], self.cores[1:]):
            for c in cores:
                copy_layer(merged.cnn_layers[c], cfg.cnn_layers[c])
                if layer_summary(merged.cnn_layers[c]) != layer_summary(cfg.cnn_layers[c]):
                    raise RuntimeError(f'core {c}: chain configuration did not copy into the merged chip configuration')
        merged.dvs_layer.pass_sensor_events = False
        for cores in self.cores:
            for c in cores:
                merged.cnn_layers[c].monitor_enable = all_layers or c == cores[-1]
        self.device.get_model().apply_configuration(merged)
        print(f"  [speck] {devkit}: {len(self.cores)} chains on cores {self.cores}")

    def reset(self):
        self.output_sink.get_events()
        builder = self.chip_factory.get_config_builder()
        module = builder.get_samna_module()
        for core, width in zip([c for cores in self.cores for c in cores], [w for ws in self.widths for w in ws]):
            events = []
            n_addr = builder.get_constraints()[core].neuron_memory
            for address in range(min(n_addr, width) if RESET_USED_ONLY else n_addr):
                ev = module.event.WriteNeuronValue()
                ev.address, ev.layer, ev.neuron_state = address, core, 0
                events.append(ev)
            source = builder.get_input_buffer()
            graph = self.samna.graph.EventFilterGraph()
            graph.sequential([source, self.device.get_model().get_sink_node()])
            graph.start()
            source.write(events)
            time.sleep(RESET_SETTLE_S)
            graph.stop()
        self.power_sink.get_events()

    def run(self, events_per_chain):
        """events_per_chain: one (T, n_inputs_k) 0/1 array per chain -> (one spike-count array per chain: its last
        layer's (T, n), or every layer's side by side with all_layers; loop seconds; mean power in W; input events)."""
        spike = self.samna.speck2f.event.Spike
        T = len(events_per_chain[0])
        frames = [torch.as_tensor(e, dtype=torch.float32)[:, None, :, None, None] for e in events_per_chain]
        reads = [list(zip(cores, w)) if self.all_layers else [(cores[-1], w[-1])]
                 for cores, w in zip(self.cores, self.widths)]
        offsets = [np.cumsum([0] + [w for _, w in r]) for r in reads]
        counts = [np.zeros((T, o[-1])) for o in offsets]
        where = {core: (k, o, w) for k, (r, off) in enumerate(zip(reads, offsets)) for (core, w), o in zip(r, off)}
        n_events = 0
        self.power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
        start = time.perf_counter()
        for t in range(T):
            evs = []
            for f, cores in zip(frames, self.cores):
                if f[t].sum() >= 1:
                    evs += self.chip_factory.raster_to_events(f[t], cores[0], dt=self.raster_dt)
            if evs:
                n_events += len(evs)
                self.input_source.write(evs)
            time.sleep(self.wait_time)
            for e in self.output_sink.get_events():
                if isinstance(e, spike) and e.layer in where:
                    k, o, w = where[e.layer]
                    if e.feature < w:
                        counts[k][t, o + e.feature] += 1
        loop_s = time.perf_counter() - start
        self.power_monitor.stop_auto_power_measurement()
        power_w, ok = power_from_events(self.power_sink.get_events(), loop_s)
        if not ok:
            print("  WARNING: the power sample count is off; this run's power may be unreliable")
        return counts, loop_s, power_w, n_events

    def close(self):
        self.graph.stop()
