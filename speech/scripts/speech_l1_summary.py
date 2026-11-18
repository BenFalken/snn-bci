"""Table for speech_l1_sweep.sh: per network, software (trained) phoneme error, the label-free layer-1 factor, the chip
model's phoneme error and agreement there, the gap, and input events per 20 ms bin; written to OUT/summary.json.

    python speech_l1_summary.py DATA_DIR OUT_DIR TRIALS CKPT [CKPT ...]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.inference import load_snn_model  # noqa: E402
from speech_chip import host_events  # noqa: E402


def events_per_bin(ck, data, trials):
    model, info = load_snn_model(ck)
    sessions = list(info['sessions'])
    n_ev = n_bins = 0
    for part in load_partitions(data, sessions, 'train', list(range(len(sessions)))):
        for i in range(min(trials, len(part))):
            ev = host_events(model, part.features[i], part.day_idx)
            n_ev, n_bins = n_ev + ev.sum(), n_bins + len(ev)
    return float(n_ev / max(n_bins, 1)), int(ev.shape[1])


def main():
    data, out, trials, ckpts = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    torch.set_num_threads(2)
    rows = {}
    print(f"{'network':38s} {'inputs':>6s} {'ev/bin':>7s} {'software':>8s} {'factor':>6s} {'chip model':>10s} "
          f"{'gap':>6s} {'agree r':>7s}")
    for ck in ckpts:
        name = os.path.basename(os.path.dirname(ck))
        sw = json.load(open(f'{out}/{name}_torch.json'))['results']['1']['per']
        ev = json.load(open(f'{out}/{name}_events.json'))['results']
        best = max(ev, key=lambda s: ev[s]['logit_corr_vs_trained'])
        epb, n_in = events_per_bin(ck, data, trials)
        rows[name] = {'ckpt': ck, 'inputs': n_in, 'input_events_per_bin': epb, 'software_per': sw,
                      'chosen_scale': float(best), 'chip_model_per': ev[best]['per'],
                      'agreement_r': ev[best]['logit_corr_vs_trained'], 'gap': ev[best]['per'] - sw,
                      'by_scale': {s: {'per': v['per'], 'agreement_r': v['logit_corr_vs_trained']} for s, v in ev.items()}}
        r = rows[name]
        print(f"{name:38s} {n_in:6d} {epb:7.1f} {sw:8.3f} {'x' + best:>6s} {r['chip_model_per']:10.3f} "
              f"{r['gap']:+6.3f} {r['agreement_r']:7.3f}")
    json.dump({'partition': 'train', 'trials_per_session': trials, 'networks': rows},
              open(f'{out}/summary.json', 'w'), indent=1)
    print(f'wrote {out}/summary.json  (training sentences, {trials} per session; the factor is chosen on agreement)')


if __name__ == '__main__':
    main()
