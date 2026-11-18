"""
Test scores of every audio_final run: r over the test windows (the pipeline's metric) and the spectral match per
held-out word epoch on its sound frames (the deck's metric, paper_figures collect_fidelity.spectral_r), plus the
validation r the runs were chosen on.

    python experiments/audio_final/score.py runs/audio_final --out runs/audio_final_scores.json [--jobs 4]
"""

import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[3]))
from paths import AUDITORY_DATA, FIGURES  # noqa: E402  (the repository's paths.yaml)
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, str(FIGURES))

WORDS = AUDITORY_DATA / 'Neil/aSTG/ALLWORDS_VERIFIED_DAYNORM'
PIPE = WORDS / 'pipeline'


def word_slices(y):
    """(start, end, sound-frame mask) per held-out word epoch, as collect_fidelity.word_rows."""
    from collect_fidelity import SAMPLES_BEFORE, sound_frames
    eps = [r for r in csv.DictReader(open(WORDS / 'prep' / 'epochs.csv')) if r['split'] == 'test']
    base = min(int(r['frame_start']) for r in eps) - SAMPLES_BEFORE
    out = []
    for r in eps:
        a, b = int(r['frame_start']) - SAMPLES_BEFORE - base, int(r['frame_end']) - SAMPLES_BEFORE - base
        if 0 <= a and b <= len(y):
            out.append((a, b, sound_frames(y[a:b].mean(1))))
    return out


def score(run_dir):
    import torch
    torch.set_num_threads(3)
    from collect_fidelity import spectral_r
    from snndec import evaluate as ev
    from snndec.convert import load
    model, info = load(f'{run_dir}/best.pt')
    model.eval()
    mode = 'recording' if model.spec['kind'] == 'speck' else info.get('mode', 'window')
    past = (info.get('config') or {}).get('data', {}).get('past')
    metrics, preds = ev.evaluate_audio(model, str(PIPE), mode, info.get('channels'), ev.recording_file(info), past)
    y = np.load(PIPE / 'window_targets_test.npy')
    p = preds['test']
    words = [spectral_r(y[a:b][m], p[a:b][m]) for a, b, m in word_slices(y)]
    return os.path.basename(run_dir), {**{k: float(v) for k, v in metrics.items()},
                                       'rho_word_mean': float(np.mean(words)), 'n_words': len(words),
                                       'epoch': info.get('epoch')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    runs = sorted(str(d) for d in Path(a.root).iterdir() if (d / 'best.pt').is_file() and (d / 'done').is_file())
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(score, runs))
    json.dump(res, open(a.out, 'w'), indent=1)
    groups = {}
    for name, r in res.items():
        groups.setdefault(name.rsplit('_s', 1)[0], []).append(r)
    print(f"{'variant':24s} {'r_valid':>8s} {'r_test':>8s} {'word match':>11s}   (mean of seeds; per seed)")
    for g, rs in sorted(groups.items()):
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"{g:24s} {f('r_valid'):8.4f} {f('r_test'):8.4f} {f('rho_word_mean'):11.4f}   "
              + ' '.join(f"{r['r_test']:.3f}/{r['rho_word_mean']:.3f}" for r in rs))


if __name__ == '__main__':
    main()
