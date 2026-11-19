"""
Channel lag tuning for audio and speech, as motor/snn_training/channel_lag_tuning.py does for motor: how each input
channel's activity relates in time to what the decoder reconstructs.

For every channel, a linear encoding model
    rate(t) = b0 + b1 z1(t + tau) + b2 z2(t + tau) + b3 z3(t + tau)
is fitted by least squares at every offset tau in -600..600 ms and scored by R^2 (tau > 0: the channel's activity
precedes the target; tau < 0: it follows it). Three regressors, as motor's (vx, vy, speed), so the R^2 scales match:
    audio   z = the top 3 principal components of the heard sound's log-mel spectrogram (128 bands, 40 ms frames;
            components fitted on the same training portion). Per recording day, the first 80% of the continuous
            stream (the decoders' training share; never their test data); every one of the 96 channels.
    speech  z = the top 3 principal components of the attempted phonemes, as a one-hot code per 20 ms bin (39
            phonemes, the word boundary, and rest before the first phoneme / after the last). Phoneme timing comes
            from forced alignment to the final LSNN's outputs on the training sentences (speech/scripts/
            export_alignments.py, run on Oscar): its outputs' own latency shifts every offset by about the same
            amount, so the spread of offsets is meaningful, the absolute centre less so. Every one of the 256 input
            features (threshold crossings and spike power of 128 electrodes), per session, training sentences only.
Rates are smoothed by a 40 ms Gaussian, as motor's. Motor and audio rates and regressors are high-passed by removing a
centred 5 s moving average (both filters symmetric: no lag shift), so slow covariation cannot tune a channel at every
offset; speech features are already normalized per block and its sentences last only a few seconds, so each
sentence's mean is removed instead, and fits use only time points whose offset partner lies in the same sentence.
A channel counts as tuned at peak R^2 >= 0.02, as motor's. Output: <workspace>/lag_tuning/<experiment>/
lag_tuning_<session>.json, in motor's format.

    python3 collect_lag_tuning.py audio | motor | speech ALIGNMENTS.npz [--out-name speech]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from repo_paths import AUDITORY_DATA, WORKSPACE  # noqa: E402  (the repository's paths.yaml)

OUT = WORKSPACE / "lag_tuning"
MAX_LAG_MS, SMOOTH_MS, MIN_R2 = 600.0, 40.0, 0.02
HIGHPASS_S = 5.0   # a centred moving average this long is removed from rates and regressors (no lag shift)
AUDIO_DAYS = {"Neil": ["07_19_11", "07_20_10", "07_31_13", "08_03_10", "08_06_13"],
              "Gomez": ["04_04_15", "04_05", "04_09_10", "04_12_13"]}
AUDIO_BIN_MS, AUDIO_TRAIN_FRAC = 40.0, 0.8


def lag_curves(rates, Z, lags):
    """(C, n_lags) R^2 of rate(t) ~ 1 + Z(t + lag), lags in bins (> 0: rate leads). rates (C, T), Z (T, k)."""
    C, T = rates.shape
    pad = int(max(abs(l) for l in lags))
    idx = np.arange(pad, T - pad)
    y = rates[:, idx].T
    y_c = y - y.mean(0)
    ss_tot = (y_c ** 2).sum(0)
    out = np.zeros((C, len(lags)))
    for j, lag in enumerate(lags):
        X = np.column_stack([np.ones(len(idx)), Z[idx + lag]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out[:, j] = np.where(ss_tot > 1e-9, 1 - (resid ** 2).sum(0) / np.maximum(ss_tot, 1e-9), 0.0)
    return out


def lag_curves_segmented(rates, Z, seg, lags):
    """lag_curves for data in segments (sentences): at each lag, only the time points t whose t + lag lies in the
    same segment. seg (T,) segment id per time point. The fit runs on cross products with a 0/1 weight per time
    point, so the rates are never copied per lag."""
    C, T = rates.shape
    out = np.zeros((C, len(lags)))
    t = np.arange(T)
    sq = rates * rates
    for j, lag in enumerate(lags):
        k = np.clip(t + lag, 0, T - 1)
        w = ((t + lag >= 0) & (t + lag < T) & (seg[k] == seg)).astype(float)
        X = np.column_stack([np.ones(T), Z[k]]) * w[:, None]       # rows outside the segment weigh zero
        n = w.sum()
        Xty = rates @ X                                             # (C, k + 1)
        beta = np.linalg.solve(X.T @ X, Xty.T)                      # (k + 1, C)
        yty = sq @ w
        ysum = rates @ w
        ss_res = yty - (beta.T * Xty).sum(1)
        ss_tot = yty - ysum ** 2 / n
        out[:, j] = np.where(ss_tot > 1e-9, 1 - ss_res / np.maximum(ss_tot, 1e-9), 0.0)
    return out


def summarize(curve, lags_ms):
    """motor's per-channel summary (channel_lag_tuning.summarize), unchanged."""
    j = int(np.argmax(curve))
    peak = float(curve[j])
    half = curve >= peak / 2
    causal = lags_ms >= 0
    jc = int(np.argmax(np.where(causal, curve, -np.inf)))
    return {"peak_r2": peak, "peak_lag_ms": float(lags_ms[j]),
            "width_ms": float(lags_ms[half].max() - lags_ms[half].min()) if peak > 0 else 0.0,
            "best_causal_delay_ms": float(lags_ms[jc]), "best_causal_r2": float(curve[jc]),
            "r2_at_0": float(curve[np.argmin(np.abs(lags_ms))])}


def top_components(Y, k=3):
    Yc = Y - Y.mean(0)
    _, s, vt = np.linalg.svd(Yc, full_matrices=False)
    return Yc @ vt[:k].T, (s[:k] ** 2 / (s ** 2).sum())


def write(exp, session, rates, Z, bin_ms, settings, seg=None):
    m = int(round(MAX_LAG_MS / bin_ms))
    lags = np.arange(-m, m + 1)
    lags_ms = lags * bin_ms
    if seg is not None:                      # sentences: smooth and centre each one on its own
        rates = rates.copy(); Z = Z.copy()
        for g in np.unique(seg):
            i = seg == g
            if SMOOTH_MS > 0:
                rates[:, i] = gaussian_filter1d(rates[:, i], SMOOTH_MS / bin_ms, axis=1)
            rates[:, i] -= rates[:, i].mean(1, keepdims=True)
            Z[i] -= Z[i].mean(0)
        curves = lag_curves_segmented(rates, Z, seg, lags)
        return _save(exp, session, curves, lags_ms, bin_ms, dict(settings, highpass="per-sentence mean"))
    if SMOOTH_MS > 0:
        rates = gaussian_filter1d(rates, SMOOTH_MS / bin_ms, axis=1)
    if HIGHPASS_S > 0:                       # slow covariation (drift, minutes of silence) would tune at every offset
        w = int(round(HIGHPASS_S * 1000 / bin_ms)) | 1
        rates = rates - uniform_filter1d(rates, w, axis=1, mode="nearest")
        Z = Z - uniform_filter1d(Z, w, axis=0, mode="nearest")
    curves = lag_curves(rates, Z, lags)
    return _save(exp, session, curves, lags_ms, bin_ms, dict(settings, highpass_s=HIGHPASS_S))


def _save(exp, session, curves, lags_ms, bin_ms, settings):
    stats = [summarize(c, lags_ms) for c in curves]
    (OUT / exp).mkdir(parents=True, exist_ok=True)
    s = dict(settings, bin_ms=bin_ms, smooth_ms=SMOOTH_MS, max_lag_ms=MAX_LAG_MS, min_r2=MIN_R2)
    json.dump({"session": session, "lags_ms": lags_ms.tolist(), "settings": s,
               "channels": [dict(st, channel=c, curve=curves[c].round(5).tolist()) for c, st in enumerate(stats)]},
              open(OUT / exp / f"lag_tuning_{session}.json", "w"))
    tuned = [st for st in stats if st["peak_r2"] >= MIN_R2]
    pk = np.array([st["peak_lag_ms"] for st in tuned])
    print(f"{exp} {session}: {len(tuned)}/{len(stats)} tuned"
          + (f", peak offset median {np.median(pk):.0f} ms (IQR {np.percentile(pk, 25):.0f} to "
             f"{np.percentile(pk, 75):.0f})" if len(pk) else ""), flush=True)


def audio():
    for monkey, days in AUDIO_DAYS.items():
        for day in days:
            P = AUDITORY_DATA / monkey / "aSTG" / day / "pipeline"
            X = np.load(P / "features.npy").astype(float)             # (T, 96) spike counts per 40 ms bin
            Y = np.load(P / "targets.npy").astype(float)              # (T, 128) log-mel frames of the heard sound
            n = int(len(X) * AUDIO_TRAIN_FRAC)                        # training portion only
            Z, var = top_components(Y[:n])
            write("audio", f"{monkey}_{day}", X[:n].T, Z, AUDIO_BIN_MS,
                  dict(source=str(P), train_frac=AUDIO_TRAIN_FRAC, regressors="top 3 log-mel PCs",
                       pc_variance=var.round(4).tolist()))


def speech(alignments, out_name="speech"):
    """Per session: the training sentences' features (snnData train.npz, the decoders' own) and their aligned
    phonemes (export_alignments.py) -> one-hot phoneme code per 20 ms bin -> its top 3 components."""
    from repo_paths import WORKSPACE as W
    A = np.load(alignments, allow_pickle=True)
    data = W / "speech_rebuild/derived/snnData"
    blank, stride = int(A["blank"]), int(A["frame_stride"])
    REST = blank                                   # the one-hot's extra class: before the first / after the last phoneme
    sess = A["sessions"]
    for s in dict.fromkeys(sess):
        z = np.load(data / s / "train.npz")
        F = z["features"]                           # read once (each z[...] access decompresses the whole array)
        lens = z["featureLens"]; starts = np.concatenate([[0], np.cumsum(lens)])
        X, codes, seg = [], [], []
        for i in np.where(sess == s)[0]:
            t = int(A["trial_idx"][i]); n = int(lens[t])
            assert int(A["input_len"][i]) == n, f"{s} trial {t}: alignment and snnData disagree on its length"
            path = A["path"][A["frame_off"][i]:A["frame_off"][i + 1]]
            lab = np.full(n, REST, dtype=int)
            on = [k for k in range(len(path)) if path[k] != blank and (k == 0 or path[k] != path[k - 1])]
            for j, k in enumerate(on):              # each phoneme from its first frame to the next phoneme's
                end = on[j + 1] if j + 1 < len(on) else max(kk for kk in range(len(path)) if path[kk] != blank) + 1
                lab[k * stride:min(n, end * stride)] = path[k]
            X.append(F[starts[t]:starts[t] + n]); codes.append(lab); seg.append(np.full(n, t))
        X, codes, seg = np.concatenate(X).astype(float), np.concatenate(codes), np.concatenate(seg)
        Y = np.eye(blank + 1)[codes]                # 40 tokens + rest
        Zc, var = top_components(Y)
        write(out_name, s, X.T, Zc, 20.0, dict(source=str(alignments), regressors="top 3 PCs of the one-hot phonemes",
                                              pc_variance=var.round(4).tolist(), sentences=int(len(np.unique(seg)))),
              seg=seg)


def motor():
    """motor's analysis (channel_lag_tuning.py: Indy, 20 ms bins, first 88%, regressors vx, vy, speed) through this
    file's fit, so the three experiments share every step (incl. the high-pass)."""
    import glob
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motor" / "snn_training"))
    from channel_lag_tuning import load_session
    for path in sorted(glob.glob(str(WORKSPACE / "spike_bmi_lsnn_data/processed/bmi/indy/mua/indy_*.h5"))):
        rates, v = load_session(path, 20.0, 0.0, 0.88)                 # smoothing is applied in write()
        Z = np.column_stack([v, np.linalg.norm(v, axis=1)])
        write("motor", Path(path).stem, rates, Z, 20.0, dict(source=path, train_frac=0.88, regressors="vx, vy, speed"))


if __name__ == "__main__":
    if sys.argv[1] == "speech":
        speech(sys.argv[2], *(sys.argv[4:5] if "--out-name" in sys.argv else []))
    else:
        {"audio": audio, "motor": motor}[sys.argv[1]]()
