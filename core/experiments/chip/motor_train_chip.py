"""
The devkit's output spikes for a motor Speck SNN's TRAINING trials, for refitting its host readout (jenkins_refit.py
--train-chip). Run as the final test run was: the trials streamed one after another without a reset between them
(hkm carries state across trials), one reset before the first, the session's layer-1 factor, 3 ms per chip step.
The reset clears every neuron of every core used and waits --settle seconds (the output layer sits on core 0, which
holds 64K neurons: 0.3 s was too short for speech's V2).

    python experiments/chip/motor_train_chip.py --session SESSION_DIR --ckpt CKPT --scalers velocity_scalers.json \
        --scale 3 --out train_chip_<session>.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--scalers', required=True)
    p.add_argument('--scale', type=float, required=True)
    p.add_argument('--wait', type=float, default=0.003)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--devkit', default='speck2fdevkit:0')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load(a.ckpt)
    model.eval()
    trials = data.motor_split(a.session, 'train', a.scalers)
    evs = []
    for t in trials:
        model(torch.as_tensor(t.x)[None])
        evs.append(chip.chip_events(model, model.input_events[0].numpy()))
    lengths = [len(e) for e in evs]
    chip.RESET_SETTLE_S = a.settle
    dev = chip.Devkit(chip.linear_stack(model, a.scale), model.host.n_out, devkit=a.devkit, wait_time=a.wait)
    dev.reset()
    t0 = time.time()
    counts, loop_s, power_w, n_ev = dev.run(np.concatenate(evs), model.layers[-1].width)
    dev.close()
    cuts = np.cumsum([0] + lengths)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **{str(i): np.minimum(counts[cuts[i]:cuts[i + 1]], 65535).astype(np.uint16)
                                  for i in range(len(trials))})
    print(f'{os.path.basename(a.session)}: {len(trials)} trials, {cuts[-1]} steps in {time.time() - t0:.0f} s, '
          f'{1e3 * loop_s / cuts[-1]:.2f} ms per step, {counts.sum() / cuts[-1]:.1f} output spikes per step; wrote {a.out}')


if __name__ == '__main__':
    main()
