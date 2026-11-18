"""
Parity on real speech checkpoints (run where they are, e.g. Oscar): for each SNNDecoder checkpoint, the original model
and the unified one on the same test trials -- the largest logit difference, and both PERs.

    python tests/check_speech_checkpoints.py --snn-decoder speech \
        --data /users/$USER/scratch/speechBCI_dryad/derived/snnData CKPT [CKPT ...]

Raw-count runs (snnDataTx1) need --data pointing there. --trials limits the trials per session (default 5).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snndec import convert, data, tasks  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpts', nargs='+')
    p.add_argument('--snn-decoder', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--trials', type=int, default=5)
    a = p.parse_args()
    sys.path.insert(0, a.snn_decoder)
    from snnDecoder.model import create_model
    from snnDecoder.train import model_kwargs

    ok = True
    for path in a.ckpts:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        args, sessions = ck['args'], ck.get('sessions') or ck['args'].get('model_sessions')
        ref = create_model(n_sessions=len(sessions), **model_kwargs(args)).eval()
        ref.load_state_dict({k: v for k, v in ck['model_state_dict'].items()}, strict=False)
        model, _ = convert.load(path)
        model.eval()
        trials = data.speech_split(a.data, sessions, 'test')
        trials = [t for d in range(len(sessions)) for t in [u for u in trials if u.day == d][:a.trials]]
        diff, L_ref, L_new = 0.0, [], []
        with torch.no_grad():
            for t in trials:
                x = torch.as_tensor(t.x)[None]
                r, n = ref(x, t.day)[0], model(x, t.day)[0]
                k = int(model.output_lengths(torch.tensor(len(t.x))))
                diff = max(diff, float((r[:k] - n[:k]).abs().max()))
                L_ref.append(r[:k].numpy())
                L_new.append(n[:k].numpy())
        ys = [t.y for t in trials]
        per_r, per_n = tasks.phoneme_error_rate(L_ref, ys), tasks.phoneme_error_rate(L_new, ys)
        good = diff < 1e-3 and per_r == per_n
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {os.path.basename(os.path.dirname(path))}: {len(trials)} trials, "
              f"max |logit diff| {diff:.2e}, PER original {per_r:.4f} unified {per_n:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
