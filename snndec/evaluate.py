"""
Evaluate any checkpoint (unified or original) on its experiment's data, with each experiment's own metric.

    python -m snndec.evaluate motor  CKPT --data SESSION_DIR --scalers velocity_scalers.json
    python -m snndec.evaluate audio  CKPT --data PIPELINE_DIR [--mode window|stream|recording]
    python -m snndec.evaluate speech CKPT --data SNNDATA_DIR [--partition test]

Prints the metric per split; --save-predictions writes them (.npy per split) to compare with the originals'.
Audio modes: 'window' (each window from rest, its last step: the LSNN's window mode), 'stream' (the windows' frames as
one recording read without resets: the LSNN's stream mode, on the normalized window features) and 'recording'
(features.npy spike counts as one recording: the Speck network).
"""

import argparse
import os

import numpy as np
import torch

from . import data, tasks
from .convert import load


@torch.no_grad()
def predict_motor(model, trials):
    return [model(torch.as_tensor(t.x)[None])[0].numpy() for t in trials]


def evaluate_motor(model, session_dir, scalers):
    scale = data.velocity_scaler(session_dir, scalers)
    out = {}
    for split in ('train', 'test'):
        trials = data.motor_split(session_dir, split, scalers)
        preds = predict_motor(model, trials)
        sq = [(data.unscale_velocity(p, *scale) - data.unscale_velocity(t.y, *scale)) ** 2 for p, t in zip(preds, trials)]
        # train_snn.py: sqrt of the mean over each batch, averaged over batches (one trial per batch at test)
        out[split] = float(np.mean([np.sqrt(s.mean()) for s in sq]))
        out[f'{split}_predictions'] = preds
    return {'rmse_' + k: v for k, v in out.items() if not k.endswith('predictions')}, out


@torch.no_grad()
def predict_audio(model, pipeline_dir, mode, channels=None, features='features.npy', past=None):
    preds = {}
    if mode == 'window':
        for s in ('train', 'valid', 'test'):
            X, _ = data.audio_windows(pipeline_dir, s, past)
            preds[s] = np.concatenate([model(torch.as_tensor(X[i:i + 1024]))[:, -1].numpy()
                                       for i in range(0, len(X), 1024)])
        return preds
    if mode == 'stream':
        Xs = [data.audio_windows(pipeline_dir, s)[0] for s in ('train', 'valid', 'test')]
        X = np.concatenate(Xs)
        rec = np.concatenate([X[:, 0], X[-1, 1:]])
        W, n = X.shape[1], [len(a) for a in Xs]
    else:
        rec, W, nd = data.audio_recording(pipeline_dir, features)
        if channels is not None:                                       # the channels the Speck network kept
            rec = rec[:, channels]
        n = [nd['train'], nd['valid'], nd['test']]
    y = model(torch.as_tensor(rec)[None])[0].numpy()[W - 1:]           # window i -> its last frame
    i = 0
    for s, k in zip(('train', 'valid', 'test'), n):
        preds[s], i = y[i:i + k], i + k
    return preds


def evaluate_audio(model, pipeline_dir, mode, channels=None, features='features.npy', past=None):
    preds = predict_audio(model, pipeline_dir, mode, channels, features, past)
    metrics = {f'r_{s}': tasks.mean_correlation(data.audio_windows(pipeline_dir, s)[1], p) for s, p in preds.items()}
    return metrics, preds


def recording_file(info):
    """The per-frame input file an audio Speck network was trained on (its config's data.features)."""
    return (info.get('config') or {}).get('data', {}).get('features', 'features.npy')


@torch.no_grad()
def evaluate_speech(model, data_dir, sessions, partition='test'):
    trials = data.speech_split(data_dir, sessions, partition)
    logits = []
    for t in trials:
        y = model(torch.as_tensor(t.x)[None], t.day)[0]
        logits.append(y[:int(model.output_lengths(torch.tensor(len(t.x))))].numpy())
    return {f'per_{partition}': tasks.phoneme_error_rate(logits, [t.y for t in trials])}, {partition: logits}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', choices=['motor', 'audio', 'speech'])
    p.add_argument('ckpt')
    p.add_argument('--data', required=True)
    p.add_argument('--scalers', help='motor: velocity_scalers.json')
    p.add_argument('--mode', default=None, help='audio: window | stream | recording (default from the checkpoint)')
    p.add_argument('--partition', default='test', help='speech partition')
    p.add_argument('--save-predictions', default=None)
    a = p.parse_args(argv)

    model, info = load(a.ckpt)
    model.eval()
    if a.experiment == 'motor':
        metrics, preds = evaluate_motor(model, a.data, a.scalers)
    elif a.experiment == 'audio':
        cfg = info.get('config') or {}
        streamed = cfg.get('train', {}).get('batches') == 'chunks'           # an LSNN trained streaming
        mode = a.mode or ('recording' if model.spec['kind'] == 'speck' or cfg.get('data', {}).get('input') == 'counts'
                          else 'stream' if streamed else info.get('mode', 'window'))
        metrics, preds = evaluate_audio(model, a.data, mode, info.get('channels'), recording_file(info),
                                        (info.get('config') or {}).get('data', {}).get('past'))
    else:
        metrics, preds = evaluate_speech(model, a.data, info['sessions'], a.partition)
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')
    if a.save_predictions:
        os.makedirs(a.save_predictions, exist_ok=True)
        for k, v in preds.items():
            np.save(os.path.join(a.save_predictions, f'{k}.npy'), np.asarray(v, dtype=object) if isinstance(v, list)
                    else v, allow_pickle=True)
    return metrics


if __name__ == '__main__':
    main()
