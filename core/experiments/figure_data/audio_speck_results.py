"""
Put the audio Speck SNN into the AuditoryDecoding pipeline as decoder 'speck_snn', in the same files the other
decoders write, so the fidelity table, the decoder comparison figure and the video pick it up:

    <pipeline>/speck_snn_predict_{train,valid,test}.npy
    <date dir>/results/speck_snn_regression_metrics.csv       (train/valid/test mean rho, as regression_metrics)

    python experiments/figure_data/audio_speck_results.py --audio-root .../pipeline_data/Neil/aSTG \
        --ckpt 07_31_13=RUN/model.pt [--ckpt DATE=CKPT ...] [--seeds DATE=CKPT1,CKPT2,CKPT3 ...]

--ckpt: the checkpoint written to the pipeline (the pipeline's other decoders use seed 1337). --seeds: optional extra
checkpoints whose test r is summarized per day (mean, SD) in <audio-root>/speck_snn_seeds.csv.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from snndec import data, evaluate as ev  # noqa: E402
from snndec.convert import load  # noqa: E402


def per_band(y, p):
    return [float(np.corrcoef(y[:, i], p[:, i])[0, 1]) for i in range(y.shape[1])]


def scores(path, pipeline):
    model, info = load(path)
    model.eval()
    return ev.evaluate_audio(model, pipeline, 'recording', info.get('channels'), ev.recording_file(info))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio-root', required=True)
    p.add_argument('--ckpt', nargs='+', default=[])
    p.add_argument('--seeds', nargs='*', default=[])
    a = p.parse_args()
    for item in a.ckpt:
        date, path = item.split('=', 1)
        pipe = f'{a.audio_root}/{date}/pipeline'
        metrics, preds = scores(path, pipe)
        row = {}
        for s in ('train', 'valid', 'test'):
            np.save(f'{pipe}/speck_snn_predict_{s}.npy', preds[s].astype(np.float64))
            rho = per_band(data.audio_windows(pipe, s)[1], preds[s])
            row[f'{s}_mean_rho'], row[f'{s}_mean_rho_list'] = metrics[f'r_{s}'], json.dumps(rho)
        row['name'] = 'speck_snn_regression_metrics'
        os.makedirs(f'{a.audio_root}/{date}/results', exist_ok=True)
        pd.DataFrame([row]).to_csv(f'{a.audio_root}/{date}/results/speck_snn_regression_metrics.csv')
        print(f"{date}: speck_snn r train {metrics['r_train']:.3f} valid {metrics['r_valid']:.3f} "
              f"test {metrics['r_test']:.3f} -> {pipe}")
    rows = []
    for item in a.seeds:
        date, paths = item.split('=', 1)
        for path in paths.split(','):
            m, _ = scores(path, f'{a.audio_root}/{date}/pipeline')
            rows.append({'date': date, 'ckpt': path, **m})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f'{a.audio_root}/speck_snn_seeds.csv', index=False)
        print(df.groupby('date')[['r_valid', 'r_test']].agg(['mean', 'std', 'count']).round(3).to_string())


if __name__ == '__main__':
    main()
