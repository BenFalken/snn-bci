"""
Accuracy and temporal precision of every audio run, on the 38 held-out word epochs of the word set:

    band_r     the reference paper's metric (Heelan et al. 2019): per word, each of the 128 mel bands' Pearson r over
               the epoch, Fisher-z averaged (paper_figures collect_fidelity.fisher_mean_r); median over words
    rho_word   our frame-wise spectral match on the word's sound frames; mean over words
    sharpness  per word, the SD of the predicted broadband envelope's frame-to-frame change over the truth's (both
               z-scored); 1 = as sharp as the truth, lower = smoothed; median over words
    width      the effective temporal kernel h (predicted envelope at t ~ sum_l h[l] * true envelope at t - l, least
               squares over all words, lags -6..6 frames): its spread, 2 SD of the positive part, in ms
    centre     h[0]: the weight the prediction puts on the true current frame
    lag        the kernel's centre of mass in ms (> 0: the prediction follows the truth late)

Runs: snn-decoders run folders (best.pt; test predictions computed once and cached as predict_test.npy), Keras runs
(predict_test.npy), and the pipeline's own predictions as references.

    python experiments/audio_temporal/score_temporal.py --out runs/audio_temporal_scores.json [--jobs 4]
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FIGURES))
WORDS = AUDITORY_DATA / 'Neil/aSTG/ALLWORDS_VERIFIED_DAYNORM'
PIPE = WORDS / 'pipeline'
LAGS = np.arange(-6, 7)

# (group, run folder or prediction file); groups ending in a seed suffix are averaged over seeds
RUN_DIRS = [REPO / 'runs/audio_temporal', REPO / 'runs/audio_final']
CONTROLS = {f'Pctl_deployed_s{s}': REPO / f'runs/audio_speck_words_s{s}' for s in (1337, 11, 99)}
PIPELINE = {'ref_pipeline_gru': PIPE / 'gru_rnn_predict_test.npy', 'ref_pipeline_lstm': PIPE / 'lstm_rnn_predict_test.npy',
            'ref_pipeline_lsnn': PIPE / 'lsnn_predict_test.npy', 'ref_pipeline_wf': PIPE / 'wiener_filter_predict_test.npy'}
KEEP = ('L0_base', 'L3_adapt0', 'T', 'P', 'K_')


def words(y):
    from collect_fidelity import SAMPLES_BEFORE, sound_frames
    eps = [r for r in csv.DictReader(open(WORDS / 'prep' / 'epochs.csv')) if r['split'] == 'test']
    base = min(int(r['frame_start']) for r in eps) - SAMPLES_BEFORE
    out = []
    for r in eps:
        a, b = int(r['frame_start']) - SAMPLES_BEFORE - base, int(r['frame_end']) - SAMPLES_BEFORE - base
        if 0 <= a and b <= len(y):
            out.append((a, b, sound_frames(y[a:b].mean(1))))
    return out


def predictions(src):
    src = Path(src)
    if src.suffix == '.npy':
        return np.load(src)
    cache = src / 'predict_test.npy'
    if cache.is_file():
        return np.load(cache)
    import torch
    torch.set_num_threads(3)
    from snndec import evaluate as ev
    from snndec.convert import load
    model, info = load(str(src / 'best.pt'))
    model.eval()
    mode = 'recording' if model.spec['kind'] == 'speck' else info.get('mode', 'window')
    past = (info.get('config') or {}).get('data', {}).get('past')
    _, preds = ev.evaluate_audio(model, str(PIPE), mode, info.get('channels'), ev.recording_file(info), past)
    np.save(cache, preds['test'])
    return preds['test']


def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def metrics(p, y, ws):
    from collect_fidelity import fisher_mean_r, spectral_r
    band = [fisher_mean_r(y[a:b], p[a:b]) for a, b, _ in ws]
    rho = [spectral_r(y[a:b][m], p[a:b][m]) for a, b, m in ws]
    sharp, rows, pe = [], [], []
    for a, b, _ in ws:
        et, ep = z(y[a:b].mean(1)), z(p[a:b].mean(1))
        sharp.append(np.std(np.diff(ep)) / np.std(np.diff(et)))
        rows.append(np.array([[et[t - l] if 0 <= t - l < len(et) else 0.0 for l in LAGS] for t in range(len(et))]))
        pe.append(ep)
    h, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(pe), rcond=None)
    hp = np.clip(h, 0, None) / np.clip(h, 0, None).sum()
    m = float((LAGS * hp).sum())
    sd = float(np.sqrt(((LAGS - m) ** 2 * hp).sum()))
    return dict(band_r=float(np.median(band)), rho_word=float(np.mean(rho)), sharpness=float(np.median(sharp)),
                width_ms=80 * sd, centre=float(h[LAGS == 0][0]), lag_ms=40 * m, kernel=[float(v) for v in h],
                band_r_words=[float(v) for v in band])


def one(item):
    name, src = item
    y = np.load(PIPE / 'window_targets_test.npy')
    try:
        return name, metrics(predictions(src), y, words(y))
    except Exception as e:                                       # a run still training, or broken: reported, skipped
        return name, {'error': repr(e)}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--out', required=True)
    pa.add_argument('--jobs', type=int, default=4)
    a = pa.parse_args()
    items = dict(PIPELINE)
    items.update({k: v for k, v in CONTROLS.items() if (v / 'best.pt').is_file()})
    for root in RUN_DIRS:
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if d.name.startswith(KEEP) and (d / 'done').is_file() and ((d / 'best.pt').is_file()
                                                                          or (d / 'predict_test.npy').is_file()):
                    items[d.name] = d
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(one, items.items()))
    json.dump(res, open(a.out, 'w'), indent=1)

    groups = {}
    for name, r in res.items():
        if 'error' in r:
            print(f'[skip] {name}: {r["error"]}')
            continue
        g = name.rsplit('_s', 1)[0] if name.rsplit('_s', 1)[-1].isdigit() else name
        groups.setdefault(g, []).append(r)
    keys = ['band_r', 'rho_word', 'sharpness', 'width_ms', 'centre', 'lag_ms']
    print(f"{'variant':28s} {'n':>2s} " + ' '.join(f'{k:>10s}' for k in keys))
    for g in sorted(groups):
        rs = groups[g]
        cells = []
        for k in keys:
            v = np.array([r[k] for r in rs])
            cells.append(f'{v.mean():6.3f}' + (f'±{v.std():.2f}' if len(v) > 1 else '     ') if 'ms' not in k
                         else f'{v.mean():6.0f}' + (f'±{v.std():<3.0f}' if len(v) > 1 else '    '))
        print(f'{g:28s} {len(rs):2d} ' + ' '.join(f'{c:>10s}' for c in cells))


if __name__ == '__main__':
    main()
