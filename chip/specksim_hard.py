"""
specksim with hard reset imposed, against our chip model (snndec.chip.run_events), on chip/specksim_compare.py's
network and events. specksim's source is compiled inside samna, so the reset is added from outside: each input spike
goes through specksim's own convolution and integrate-and-fire layer (forward_single, forward), and the neurons that
fired are set to zero (get_v_mem / set_v_mem). In these fully connected layers each input spike reaches each neuron
once, so this is exactly a reset to zero at the spike. Spikes are passed on in neuron order, as ours.
    python3 chip/specksim_compare.py && python3 chip/specksim_hard.py
Result (2026-11-24, samna 0.48): identical spike trains, all 3,000 steps, every layer (weights and thresholds are whole
numbers, so float32 is exact). In specksim's own spike order: same rate (48.87 vs 48.85 per step), correlation 0.993.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
S = os.path.dirname(os.path.abspath(__file__))
exec(open('chip/specksim_compare.py').read().split("t0 = time.time(); ours")[0])     # same network, same events
ours = np.load('chip/specksim_compare.npz')['ours']
shape = (ev.shape[1], 1, 1); nodes = []; layers = []
for mod in q:
    node, shape = (SS.convert_convolutional_layer(mod, shape) if isinstance(mod, torch.nn.Conv2d)
                   else SS.convert_iaf_layer(mod, shape))
    nodes.append(node); layers.append(node.get_layer())
pairs = list(zip(layers[0::2], layers[1::2]))          # (convolution, integrate-and-fire) per layer
t0 = time.time(); hard = np.zeros_like(ours)
for t, row in enumerate(ev):
    spikes = [Spike(int(c), 0, 0, t) for c in np.flatnonzero(row)]
    for conv, iaf in pairs:
        out = []
        for sp in spikes:                              # one input spike at a time
            fired = iaf.forward(conv.forward_single(sp))   # specksim's own weights, sums and threshold test
            if fired:                                   # hard reset: the neurons that fired go to zero
                v = iaf.get_v_mem()
                for f in fired:
                    v[f.feature][0][0] = 0.0
                iaf.set_v_mem(v)
                out.extend(sorted(fired, key=lambda f: f.feature))   # passed on in neuron order, as ours
        spikes = out
    for sp in spikes:
        hard[t, sp.feature] += 1
print('time %.1fs' % (time.time() - t0))
f = lambda a: a.reshape(-1, 10, a.shape[1]).sum(1).ravel()
print('spikes per step: ours %.2f | specksim with hard reset %.2f' % (ours.sum(1).mean(), hard.sum(1).mean()))
print('per-frame count correlation %.4f' % np.corrcoef(f(ours), f(hard))[0, 1])
print('identical spike trains: %s | steps identical %.1f%% | max difference %d'
      % (np.array_equal(ours, hard), 100 * np.mean(np.all(ours == hard, axis=1)), np.abs(ours - hard).max()))
ws = [mm.weight.detach().numpy() for mm in q if hasattr(mm, 'weight')]; print('weights whole numbers:', all(np.allclose(w, np.round(w)) for w in ws), '| thresholds', [float(mm.spike_threshold) for mm in q if not hasattr(mm, 'weight')])
