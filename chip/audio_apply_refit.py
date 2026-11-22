"""
Decode a devkit run's Hidden 3 spikes (audio_chip.py --save-preds DIR -> DIR/chip_counts.npz) with the readout
refitted to the chip model on the training rows (audio_refit.py -> OUT/readout_event.npz), and save the test
predictions next to the deployed ones as chip_refit.npy.

    python chip/audio_apply_refit.py --ckpt words_P5_s11/model.pt --counts preds_s11/chip_counts.npz \
        --readout final_results/audio_refit/words_P5_s11/readout_event.npz --out final_results/audio_preds/words_P5_s11
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_refit import features  # noqa: E402
from snndec.convert import load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--counts', required=True)
    p.add_argument('--readout', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    model, _ = load(a.ckpt)
    z, r = np.load(a.counts), np.load(a.readout)
    X = features(model, z['h3'], r['decay'] if 'decay' in r.files else None)
    first, n = int(z['first_frame']), int(z['n_test'])
    pred = (X @ r['W'] + r['b'])[first:first + n]
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, 'chip_refit.npy'), pred.astype(np.float32))
    print(f'wrote {a.out}/chip_refit.npy {pred.shape}')


if __name__ == '__main__':
    main()
