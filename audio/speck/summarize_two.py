"""
Two-network audio Speck variants against P5 (one network, 48 channels), both monkeys, in the chip model (event-by-event,
label-free layer-1 factor, readout refitted to the chip model's spikes on training data): the test stream's
correlation and the reference paper's per-utterance metric (median band_r), with the Wiener filter for reference.
Writes ~/Downloads/final_results/audio_two/summary.json, including the variant for the devkit: the one chosen on validation
during training (run_two.sh), with every variant's chip-model test scores alongside for the record.

    python audio/speck/summarize_two.py
"""
import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[2]))
from paths import WORKSPACE, AUDITORY_DATA, FIGURES  # noqa: E402  (the repository's paths.yaml)
import csv
import glob
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser('~')
AD = str(AUDITORY_DATA)
OUT = f'{WORKSPACE}/final_results/audio_two'
P5 = f'{WORKSPACE}/final_results/audio_novel'
sys.path.insert(0, str(FIGURES))
from collect_fidelity import fisher_mean_r  # noqa: E402

SAMPLES_BEFORE = 8


def words(monkey):
    base = f'{AD}/{monkey}/aSTG/ALLWORDS_NOVEL_DAYNORM'
    eps = [r for r in csv.DictReader(open(f'{base}/prep/epochs.csv')) if r['split'] == 'test']
    b0 = min(int(r['frame_start']) for r in eps) - SAMPLES_BEFORE
    y = np.load(f'{base}/pipeline/window_targets_test.npy')
    spans = [(int(r['frame_start']) - SAMPLES_BEFORE - b0, int(r['frame_end']) - SAMPLES_BEFORE - b0) for r in eps]
    return y, [(a, b) for a, b in spans if a >= 0 and b <= len(y)]


def per_word(y, spans, preds):
    return [np.mean([fisher_mean_r(y[a:b], p[a:b]) for p in preds]) for a, b in spans]


def main():
    out = {}
    for m in ('Neil', 'Gomez'):
        y, spans = words(m)
        wf = np.load(f'{AD}/{m}/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline/wiener_filter_predict_test.npy')
        res = {'wiener_filter': {'band_r_median': float(np.median(per_word(y, spans, [wf])))}}
        groups = {'P5': sorted(glob.glob(f'{P5}/{m}_P5_*/refit'))}
        for v in ('A_blocks', 'B_interleave', 'C_tapsplit'):
            groups[v] = sorted(glob.glob(f'{OUT}/{m}_{v}_s*/refit'))
        for v, dirs in groups.items():
            dirs = [d for d in dirs if os.path.isfile(f'{d}/refit.json')]
            if not dirs:
                continue
            js = [json.load(open(f'{d}/refit.json')) for d in dirs]
            pr = {k: [np.load(f'{d}/preds_{k}.npy') for d in dirs] for k in ('float', 'event_trained', 'event_refit')}
            res[v] = {'seeds': len(dirs),
                      'scale': [j['scale'] for j in js],
                      'stream_r_float': float(np.mean([j['r_float_test'] for j in js])),
                      'stream_r_chip_trained': float(np.mean([j['r_event_trained']['test'] for j in js])),
                      'stream_r_chip_refit': float(np.mean([j['r_event_refit']['test'] for j in js])),
                      **{f'band_r_median_{k}': float(np.median(per_word(y, spans, pr[k]))) for k in pr}}
        out[m] = res
    cand = [v for v in ('A_blocks', 'B_interleave', 'C_tapsplit') if all(v in out[m] for m in out)]
    score = {v: float(np.mean([out[m][v]['band_r_median_event_refit'] for m in out])) for v in cand}
    vf = f'{OUT}/chosen_on_validation.txt'                 # chosen on validation in training, not on test
    out['chosen'] = open(vf).read().strip() if os.path.isfile(vf) else (max(score, key=score.get) if score else None)
    out['chosen_score'] = score
    out['P5_score'] = float(np.mean([out[m]['P5']['band_r_median_event_refit'] for m in out if 'P5' in out[m]]))
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1)
    for m in ('Neil', 'Gomez'):
        print(f'\n{m}: per-utterance median band_r (chip model)  |  stream r', flush=True)
        for v, r in out[m].items():
            if v == 'wiener_filter':
                print(f"  {'Wiener filter':14s} {r['band_r_median']:.3f}")
                continue
            print(f"  {v:14s} software {r['band_r_median_float']:.3f}  chip trained {r['band_r_median_event_trained']:.3f}"
                  f"  chip refit {r['band_r_median_event_refit']:.3f}  |  stream refit {r['stream_r_chip_refit']:.3f}"
                  f"  (x{r['scale']})")
    print(f"\nchosen for the devkit: {out['chosen']} (mean over monkeys {score.get(out['chosen'], float('nan')):.3f}; "
          f"P5 {out['P5_score']:.3f})")


if __name__ == '__main__':
    main()
