"""
Layer-1 threshold scale for an audio Speck network, chosen without labels on the validation split: the chip's
event-by-event firing in software (snndec.chip.run_events) at several scales, scored by agreement (correlation) with
the trained network's own output; spectrogram r against the targets alongside, for reference.

    python chip/audio_l1_sweep.py PIPELINE_DIR CKPT --scales 1 2 3 4 6 --out sweep.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snndec import chip, data, tasks  # noqa: E402
from snndec.convert import load  # noqa: E402
from snndec.evaluate import recording_file  # noqa: E402
from audio_chip import BURN, decode  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pipeline')
    p.add_argument('ckpt')
    p.add_argument('--scales', type=float, nargs='+', default=[1, 2, 3, 4, 6])
    p.add_argument('--split', default='valid', choices=['valid', 'test'])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(3)
    model, info = load(a.ckpt)
    model.eval()
    rec, W, n = data.audio_recording(a.pipeline, recording_file(info))
    if info.get('channels') is not None:
        rec = rec[:, list(info['channels'])]
    y = np.load(f'{a.pipeline}/window_targets_{a.split}.npy').astype(np.float32)
    first = (n['train'] if a.split == 'valid' else n['train'] + n['valid']) + W - 1
    start = max(0, first - BURN)
    seg = torch.as_tensor(rec[start:first + len(y)])[None]
    y_float = model(seg)[0].numpy()
    events = chip.chip_events(model, model.input_events[0].numpy())
    sel = slice(first - start, first - start + len(y))
    res = {'ckpt': a.ckpt, 'split': a.split, 'r_float': tasks.mean_correlation(y, y_float[sel]), 'results': {}}
    print(f"trained network: r {res['r_float']:.4f} on {a.split}")
    for s in a.scales:
        t0 = time.time()
        out = decode(model, chip.simulate(model, events, s))
        agree = float(np.corrcoef(out[sel].ravel(), y_float[sel].ravel())[0, 1])
        r = tasks.mean_correlation(y, out[sel])
        res['results'][f'{s:g}'] = {'agreement_r': agree, 'r': r}
        print(f'layer 1 x{s:g}: agreement r {agree:.4f}  r vs targets {r:.4f}  ({time.time() - t0:.0f} s)', flush=True)
    res['chosen_scale'] = max(a.scales, key=lambda s: res['results'][f'{s:g}']['agreement_r'])
    print(f"label-free choice: x{res['chosen_scale']:g}")
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
