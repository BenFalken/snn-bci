"""
Layer-1 threshold scale for a motor Speck network, chosen without labels: the chip's event-by-event firing in
software (snndec.chip.run_events) at several scales, scored by agreement with the trained network's own output
(correlation of decoded velocities), with the RMSE alongside for reference. Also the discretized network stepped as in
training (run_software), which isolates quantization from event-by-event firing.

    python experiments/chip/motor_l1_sweep.py SESSION_DIR CKPT --scalers velocity_scalers.json \
        --scales 0.5 0.75 1 1.5 2 3 --trials 5 --out sweep.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402


@torch.no_grad()
def decode(model, spikes):
    """The last chip layer's spikes (T, n) -> the host readout's velocities (T, 2)."""
    st = model.readout.init_state(1, 'cpu')
    return torch.stack([model.readout.step(torch.as_tensor(s, dtype=torch.float32)[None], st)[0]
                        for s in spikes]).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session')
    p.add_argument('ckpt')
    p.add_argument('--scalers', required=True)
    p.add_argument('--scales', nargs='+', default=['0.5', '0.75', '1', '1.5', '2', '3'],
                   help='layer 1 scale, or per-layer scales a,b,c,d (e.g. 1,0.5,0.5,0.5)')
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--split', default='train', help='never test: the factor is chosen on data the evaluation does not use')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    model, info = load(a.ckpt)
    model.eval()
    scale = data.velocity_scaler(a.session, a.scalers)
    trials = data.motor_split(a.session, a.split, a.scalers)[:a.trials]
    rows = {s: {'sq': [], 'corr': [], 'spikes': 0.0} for s in ['software'] + a.scales}
    ref_spikes, steps = 0.0, 0
    def build(s):
        k = [float(v) for v in s.split(',')]
        seq = chip.linear_stack(model, k[0])
        if len(k) > 1:                                  # later layers' thresholds and floors too
            neurons = [m for m in seq if not isinstance(m, torch.nn.Linear)]
            for n, f in zip(neurons[1:], k[1:]):
                with torch.no_grad():
                    n.spike_threshold.mul_(f)
                    if n.min_v_mem is not None:
                        n.min_v_mem.mul_(f)
        return chip.discretized(seq, model.host.n_out)
    nets = {s: build(s) for s in a.scales}
    soft = chip.discretized(chip.linear_stack(model, 1.0), model.host.n_out)
    t0 = time.time()
    for t in trials:
        with torch.no_grad():
            y_ref = model(torch.as_tensor(t.x)[None])[0].numpy()
        ev = chip.chip_events(model, model.input_events[0].numpy())
        ref_spikes += float(model.layer_spikes[-1].sum())
        steps += len(ev)
        truth = data.unscale_velocity(t.y, *scale)
        for s in rows:
            out = chip.run_software(soft, ev) if s == 'software' else chip.run_events(nets[s], ev)
            y = decode(model, out)
            rows[s]['spikes'] += float(out.sum())
            rows[s]['sq'].append(np.sqrt(((data.unscale_velocity(y, *scale) - truth) ** 2).mean()))
            rows[s]['corr'].append(np.corrcoef(y.ravel(), y_ref.ravel())[0, 1])
        print(f'trial done ({time.time() - t0:.0f} s)', flush=True)
    ref_rmse = float(np.mean([np.sqrt(((data.unscale_velocity(model(torch.as_tensor(t.x)[None])[0].detach().numpy(),
                                                                *scale) - data.unscale_velocity(t.y, *scale)) ** 2).mean())
                              for t in trials]))
    res = {'session': os.path.basename(a.session), 'trials': len(trials), 'rmse_trained': ref_rmse,
           'out_spikes_per_step_trained': ref_spikes / steps, 'results': {}}
    print(f"trained network: RMSE {ref_rmse:.2f}, {ref_spikes / steps:.0f} output spikes per step")
    for s, r in rows.items():
        res['results'][str(s)] = {'rmse': float(np.mean(r['sq'])), 'agreement_r': float(np.mean(r['corr'])),
                                  'out_spikes_per_step': r['spikes'] / steps}
        print(f"{'8-bit, summed steps' if s == 'software' else f'event-by-event, x{s}':32s} RMSE "
              f"{np.mean(r['sq']):6.2f}  agreement r {np.mean(r['corr']):.3f}  output spikes/step {r['spikes'] / steps:.0f}")
    best = max(a.scales, key=lambda s: res['results'][str(s)]['agreement_r'])
    res['chosen_scale'] = best
    print(f'label-free choice: x{best}')
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
