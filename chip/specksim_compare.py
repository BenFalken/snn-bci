"""
Our chip model (snndec.chip.run_events) against SynSense's specksim (samna.specksim, through sinabs), on the same
network and the same input events: the 4 ms audio Speck SNN (FT1, monkey N, seed s11), 300 test frames (3,000 chip
steps), at its label-free layer-1 factor. specksim is run layer by layer on each step's events, as its graph does.
    python3 chip/specksim_compare.py
Result (2026-11-24, samna 0.48): last-layer spikes per step 48.85 (ours, reset to zero) vs 57.45 (specksim: subtract
reset, at most one spike per event); per-frame count correlation 0.989; ours 0.6 s, specksim 14.5 s. With specksim's
reset rule our loop gives 57.41 spikes per step (correlation 0.996): the reset rule explains the difference.
"""
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'chip'))
from snndec.convert import load
from snndec import chip, data
from snndec.evaluate import recording_file
from sinabs.backend.dynapcnn import specksim as SS
from samna.specksim.events import Spike
W=os.environ['SNNBCI_WORKSPACE']; R=os.environ['SNNBCI_SNN_RUNS']; AD=os.environ['SNNBCI_AUDITORY_DATA']
m, s = 'Neil', 's11'
model, info = load(f'{R}/audio_fine_{m}/FT1_spikes_{s}/best.pt'); model.eval()
k = float(json.load(open(f'{W}/final_results/audio_fine/{m}_FT1_spikes_{s}/sweep.json'))['chosen_scale'])
P = f'{AD}/{m}/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline'
rec, Wn, n = data.audio_recording(P, recording_file(info))
first = n['train'] + n['valid'] + Wn - 1
x = rec[first:first + 300]
with torch.no_grad():
    model(torch.as_tensor(x)[None])
ev = chip.chip_events(model, model.input_events[0].numpy())
net = chip.discretized(chip.linear_stack(model, k), model.host.n_out)
q = chip.quantized_layers(net)
print('layer-1 factor', k, '| steps', len(ev), '| input events per step %.1f' % ev.sum(1).mean(), flush=True)
t0 = time.time(); ours = chip.run_events(net, ev); t_ours = time.time() - t0
print('ours done %.1fs' % t_ours, flush=True)
shape = (ev.shape[1], 1, 1); layers = []; nodes = []   # keep the nodes alive: their layers are views
for mod in q:
    if isinstance(mod, torch.nn.Conv2d):
        node, shape = SS.convert_convolutional_layer(mod, shape)
    else:
        node, shape = SS.convert_iaf_layer(mod, shape)
    nodes.append(node); layers.append(node.get_layer())
print('specksim layers:', [type(L).__name__ for L in layers], flush=True)
t0 = time.time(); sim = np.zeros_like(ours)
for t, row in enumerate(ev):
    batch = [Spike(int(c), 0, 0, t) for c in np.flatnonzero(row)]
    for L in layers:
        batch = L.forward(batch)
    for sp in batch:
        sim[t, sp.feature] += 1
t_sim = time.time() - t0
print('Hidden-3 spikes per step: ours %.2f  specksim %.2f' % (ours.sum(1).mean(), sim.sum(1).mean()))
fo = ours.reshape(-1, 10, ours.shape[1]).sum(1); fs = sim.reshape(-1, 10, sim.shape[1]).sum(1)
print('per-frame, per-neuron count correlation ours vs specksim: %.3f' % np.corrcoef(fo.ravel(), fs.ravel())[0, 1])
print('identical spike trains:', np.array_equal(ours, sim), '| time ours %.1fs specksim %.1fs' % (t_ours, t_sim))
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'specksim_compare.npz'), ours=ours, sim=sim)
