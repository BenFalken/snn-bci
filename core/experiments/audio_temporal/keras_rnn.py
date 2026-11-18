"""
The audio GRU and LSTM of the AuditoryDecoding pipeline (modules/decoders/keras/{gru,lstm}_rnn: one 1,024-unit layer
over the 17-frame window, Dense to 128 bands, MSE, RMSprop, early stopping on validation loss with patience 5, best
weights restored), retrained to test the frame-to-frame loss term fairly against the spiking decoders.

Both the control (--deriv 0) and the variant (--deriv 1) train on PAIRS of neighbouring windows (i, i + 1) through
one shared network, so they differ only in the loss:
    MSE(window i) + MSE(window i + 1) + deriv * MSE((out_{i+1} - out_i) - (y_{i+1} - y_i))
Each window appears in about two pairs per epoch; the epoch ceiling (100) and batch size (32 pairs) follow the
pipeline's defaults.

    python experiments/audio_temporal/keras_rnn.py --cell gru --deriv 1 --seed 1337 --out runs/audio_temporal/K_gru_deriv_s1337

Writes <out>/predict_test.npy (aligned with window_targets_test.npy), history.json and done.
"""

import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[3]))
from paths import AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)
import argparse
import json
import os
from pathlib import Path

import numpy as np

PIPE = AUDITORY_DATA / 'Neil/aSTG/ALLWORDS_VERIFIED_DAYNORM/pipeline'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', choices=['gru', 'lstm'], required=True)
    p.add_argument('--deriv', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--units', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(a.seed)
    from tensorflow.keras import layers, Model, Sequential

    X = {s: np.load(PIPE / f'window_features_{s}.npy').astype(np.float32) for s in ('train', 'valid', 'test')}
    Y = {s: np.load(PIPE / f'window_targets_{s}.npy').astype(np.float32) for s in ('train', 'valid', 'test')}
    T, C, B = X['train'].shape[1], X['train'].shape[2], Y['train'].shape[1]

    cell = layers.GRU if a.cell == 'gru' else layers.LSTM
    base = Sequential([layers.Input((T, C)), cell(a.units), layers.Dense(B)])
    ia, ib = layers.Input((T, C)), layers.Input((T, C))
    pair = Model([ia, ib], layers.Concatenate()([base(ia), base(ib)]))
    w = float(a.deriv)

    def loss(yt, yp):
        ta, tb, pa, pb = yt[:, :B], yt[:, B:], yp[:, :B], yp[:, B:]
        mse = tf.reduce_mean(tf.square(pa - ta)) + tf.reduce_mean(tf.square(pb - tb))
        return mse + w * tf.reduce_mean(tf.square((pb - pa) - (tb - ta)))

    pair.compile(loss=loss, optimizer='rmsprop')
    pairs = lambda s: ([X[s][:-1], X[s][1:]], np.concatenate([Y[s][:-1], Y[s][1:]], 1))
    (xa, ya), (xv, yv) = pairs('train'), pairs('valid')
    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, restore_best_weights=True,
                                          verbose=1)
    h = pair.fit(xa, ya, validation_data=(xv, yv), epochs=a.epochs, batch_size=a.batch, callbacks=[cb], verbose=2,
                 shuffle=True)
    os.makedirs(a.out, exist_ok=True)
    np.save(Path(a.out) / 'predict_test.npy', base.predict(X['test'], batch_size=256, verbose=0))
    json.dump({k: [float(v) for v in vs] for k, vs in h.history.items()}, open(Path(a.out) / 'history.json', 'w'))
    Path(a.out, 'done').touch()


if __name__ == '__main__':
    main()
