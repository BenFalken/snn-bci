"""
Quick CPU check of the native model's synaptic-delay options before submitting a delay round (about a minute):
    python test_delays.py
For each option: streaming one bin at a time must equal the full forward pass ("stream diff" ~0), gradients must
reach the new weights, and a delayed synapse must deliver its spike exactly at its own delay. Ends with "ALL OK".
"""
import os
import sys

import torch
import sinabs.activation as A

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snnDecoder.rsnn_native import SNN_Native  # noqa: E402

torch.manual_seed(0)
ok = True
base = dict(num_input_channels=16, hidden_dims=[32, 32], spike_fn=A.MaxSpike(3), tau_mem=25, max_vmem=50)
cases = [dict(), dict(syn_delay_in=6, syn_delay_in_mult=3), dict(syn_delay_ff=4), dict(syn_delay_rec=[2, 5]),
         dict(hidden_dims=[32, 32, 32], syn_delay_rec=[1, 3, 6], syn_delay_ff=3, tbptt=9),
         dict(syn_delay_in=6, syn_delay_in_mult=2, syn_delay_ff=3, syn_delay_rec=[2], output_stride=4, out_stages=2)]
for kw in cases:
    m = SNN_Native(2, **{**base, **kw}).eval()
    x = torch.randn(2, 41, 16) * 2
    with torch.no_grad():
        f = m(x, 1, count_ops=True)
        m.reset_stream()
        st = [m.step(x[:, t], 1) for t in range(41)]
        st = torch.stack([o for o in st if o is not None], 1)
    m.train()
    m(x, 1).sum().backward()
    grads = bool(m.layers[-1].ff.weight.grad.abs().sum() > 0)
    if m.in_mult > 1:
        grads = grads and bool(m.in_extra.grad.abs().sum() > 0)
    diff = float((f - st).abs().max())
    ok = ok and diff < 1e-4 and grads
    print(kw, f'stream diff {diff:.1e}', 'params', sum(p.numel() for p in m.parameters()), 'grads ok', grads)

# a layer-2 neuron receives a layer-1 spike exactly at its synapse's delay
m = SNN_Native(1, num_input_channels=1, hidden_dims=[1, 3], syn_delay_ff=4, recurrent=False, day_input='none',
               tau_syn_range=None, spike_fn=A.MultiSpike).eval()
with torch.no_grad():
    m.layers[1].ff.weight[:] = 5.0
    s = m.layers[1].init_state(1, 'cpu')
    arr = [m.layers[1].delayed_drive(torch.ones(1, 1) if t == 0 else torch.zeros(1, 1), s)[0].tolist() for t in range(6)]
delays = [int(m.layers[1].ff_syn_masks[:, j, 0].float().argmax()) for j in range(3)]
arrive = [next(t for t in range(6) if arr[t][j] > 0) for j in range(3)]
ok = ok and delays == arrive
print('between-layer synapse delays', delays, '-> current arrives at steps', arrive)

# mixed population (LSNN): only the adaptive fraction's thresholds move; streaming still equals forward
for kw in (dict(neuron_type='iaf', adaptive_fraction=0.4, tau_adapt=50.0),
           dict(neuron_type='iaf', adaptive_fraction=0.4, tau_adapt=50.0, syn_delay_rec=[3], syn_delay_in=6)):
    m = SNN_Native(2, **{**base, **kw}).eval()
    x = torch.randn(2, 41, 16) * 3
    with torch.no_grad():
        f = m(x, 1)
        m.reset_stream()
        stv = torch.stack([m.step(x[:, t], 1) for t in range(41)], 1)
        st_b = m._stream['states'][0]['b']
    mask = m.layers[0].adapt_mask
    moved = bool((st_b[:, mask == 1].abs().sum() > 0)) if mask.sum() > 0 else True
    diff = float((f - stv).abs().max())
    ok = ok and diff < 1e-4 and moved and int(mask.sum()) == round(0.4 * 32)
    print(kw, f'stream diff {diff:.1e}', 'adaptive neurons', int(mask.sum()), 'of 32, adapting', moved)

# single-synapse checkpoints still load
SNN_Native(2, **base, syn_delay_in=6).load_state_dict(SNN_Native(2, **base, syn_delay_in=6).state_dict())

# learned delays (delays.py): once rounded, identical to the fixed delays they start from; while soft, the loss
# has a gradient with respect to every delay
from snnDecoder.delays import delay_modules, set_delay_sigma  # noqa: E402
kw = dict(n_sessions=1, num_input_channels=16, n_classes=5, hidden_dims=[32, 32], neuron_type='iaf',
          syn_delay_in=6, syn_delay_in_mult=2, syn_delay_rec=[2, 4], syn_delay_ff=3, output_stride=4)
torch.manual_seed(0)
fixed = SNN_Native(**kw).eval()
learned = SNN_Native(learn_delays=True, **kw)
learned.load_state_dict(fixed.state_dict(), strict=False)
learned.eval()
set_delay_sigma(learned, 9, 10)
x = torch.randn(2, 24, 16)
with torch.no_grad():
    same = torch.allclose(fixed(x, 0), learned(x, 0), atol=1e-6)
learned.train()
set_delay_sigma(learned, 0, 10)
learned(x, 0).pow(2).mean().backward()
grads = all(m.pos.grad is not None and m.pos.grad.abs().sum() > 0 for _, m in delay_modules(learned))
print('learned delays: rounded == fixed', same, '| gradient to every delay group', grads)
ok = ok and same and grads
print('ALL OK' if ok else 'FAILED')
