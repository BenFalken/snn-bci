"""
How each input channel's activity relates to hand velocity in time: lead, lag and strength.

For every multi-unit channel of a session, a linear velocity tuning model
    rate(t) = b0 + bx * vx(t + tau) + by * vy(t + tau) + bs * speed(t + tau)
(direction and speed, as in Moran & Schwartz 1999) is fitted by least squares at
each offset tau from -max_lag to +max_lag, and scored by R^2. tau > 0: the
channel's activity precedes the velocity it relates to (it leads movement, like a
motor command), so a causal decoder uses it best delayed by about tau; tau < 0:
it follows movement (feedback). Per channel:
    peak_r2      the best R^2 over offsets (how strongly the channel locks to velocity)
    peak_lag_ms  the offset of that peak (> 0 leads, < 0 lags)
    width_ms     the span of offsets with R^2 >= half the peak
    best_causal_delay_ms  the offset >= 0 with the best R^2: the delay a decoder can use

Only the first --train-frac of each session is used (the decoders' test split is
the last 10%), so delays chosen from this never see test data.

Spikes are counted in --bin-ms bins and smoothed by a Gaussian of --smooth-ms
(symmetric, so smoothing shifts no lag); velocity is averaged in the same bins.

Inputs: preprocessing_training/process_data.py outputs (mua_trains, task_time,
task_data with vel_x, vel_y in columns 2-3). Outputs in --out-dir:
    lag_tuning_<session>.json   per channel: peak_r2, peak_lag_ms, width_ms,
                                best_causal_delay_ms, and the R^2 curve
    lag_tuning_summary.json     per session and pooled: lead/lag counts, lag quantiles
    lag_tuning.png              curves sorted by peak lag, peak-lag histogram, strength vs lag

    python snn_training/channel_lag_tuning.py --processed .../processed/bmi/indy/mua/*.h5 --out-dir .../lag_tuning
"""

import argparse
import json
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_session(path, bin_ms, smooth_ms, train_frac):
    with h5py.File(path, "r") as h:
        t = h["task_time"][()]
        vel = h["task_data"][()][:, 2:4]
        trains = [np.asarray(s) for s in h["mua_trains"][()]]
    n = int(len(t) * train_frac)                                   # training portion only
    t, vel = t[:n], vel[:n]
    step = int(round(bin_ms / 4.0))                                  # task samples are 4 ms apart
    nb = len(t) // step
    edges = t[0] + np.arange(nb + 1) * step * 0.004
    v = vel[:nb * step].reshape(nb, step, 2).mean(axis=1)
    rates = np.stack([np.histogram(s, edges)[0] for s in trains]).astype(float)       # (C, nb)
    if smooth_ms > 0:
        rates = gaussian_filter1d(rates, smooth_ms / bin_ms, axis=1)
    return rates, v


def lag_curves(rates, v, lags):
    """(C, n_lags) R^2 of the tuning fit with velocity shifted by each lag (in bins; > 0: rate leads)."""
    C, T = rates.shape
    speed = np.linalg.norm(v, axis=1)
    out = np.zeros((C, len(lags)))
    pad = max(abs(int(l)) for l in lags)
    idx = np.arange(pad, T - pad)
    y = rates[:, idx].T                                              # (n, C)
    y_c = y - y.mean(0)
    ss_tot = (y_c ** 2).sum(0)
    for j, lag in enumerate(lags):
        k = idx + lag                                                # velocity at t + lag
        X = np.column_stack([np.ones(len(idx)), v[k, 0], v[k, 1], speed[k]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out[:, j] = np.where(ss_tot > 1e-9, 1 - (resid ** 2).sum(0) / np.maximum(ss_tot, 1e-9), 0.0)   # silent: 0
    return out


def summarize(curve, lags_ms):
    j = int(np.argmax(curve))
    peak = float(curve[j])
    half = curve >= peak / 2
    causal = lags_ms >= 0
    jc = int(np.argmax(np.where(causal, curve, -np.inf)))
    return {"peak_r2": peak, "peak_lag_ms": float(lags_ms[j]),
            "width_ms": float(lags_ms[half].max() - lags_ms[half].min()) if peak > 0 else 0.0,
            "best_causal_delay_ms": float(lags_ms[jc]), "best_causal_r2": float(curve[jc]),
            "r2_at_0": float(curve[np.argmin(np.abs(lags_ms))])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", nargs="+", required=True, help="process_data.py .h5 files, one per session")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bin-ms", type=float, default=20.0)
    p.add_argument("--smooth-ms", type=float, default=40.0)
    p.add_argument("--max-lag-ms", type=float, default=600.0)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--min-r2", type=float, default=0.02, help="channels below this peak R^2 count as untuned")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    m = int(round(args.max_lag_ms / args.bin_ms))
    lags = np.arange(-m, m + 1)
    lags_ms = lags * args.bin_ms

    sessions, curves_all, stats_all = {}, [], []
    for path in sorted(args.processed):
        s = os.path.basename(path).replace(".h5", "")
        rates, v = load_session(path, args.bin_ms, args.smooth_ms, args.train_frac)
        curves = lag_curves(rates, v, lags)
        stats = [summarize(c, lags_ms) for c in curves]
        with open(os.path.join(args.out_dir, f"lag_tuning_{s}.json"), "w") as f:
            json.dump({"session": s, "lags_ms": lags_ms.tolist(), "settings": vars(args) | {"processed": path},
                       "channels": [dict(st, channel=c, curve=curves[c].round(5).tolist())
                                    for c, st in enumerate(stats)]}, f)
        tuned = [st for st in stats if st["peak_r2"] >= args.min_r2]
        pk = np.array([st["peak_lag_ms"] for st in tuned])
        sessions[s] = {"channels": len(stats), "tuned": len(tuned),
                       "leading (peak lag > 0)": int((pk > 0).sum()), "lagging (< 0)": int((pk < 0).sum()),
                       "peak_lag_ms_quartiles": np.percentile(pk, [25, 50, 75]).tolist() if len(pk) else None,
                       "best_causal_delay_ms_quartiles": np.percentile(
                           [st["best_causal_delay_ms"] for st in tuned], [25, 50, 75]).tolist() if tuned else None,
                       "median_peak_r2": float(np.median([st["peak_r2"] for st in tuned])) if tuned else None}
        print(s, sessions[s], flush=True)
        curves_all.append(curves)
        stats_all.append(stats)
    with open(os.path.join(args.out_dir, "lag_tuning_summary.json"), "w") as f:
        json.dump(sessions, f, indent=2)
    plot(curves_all, stats_all, list(sessions), lags_ms, args)


def plot(curves_all, stats_all, names, lags_ms, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Palatino", "serif"], "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for i, (curves, stats, s) in enumerate(zip(curves_all, stats_all, names)):
        tuned = [c for c, st in enumerate(stats) if st["peak_r2"] >= args.min_r2]
        order = sorted(tuned, key=lambda c: stats[c]["peak_lag_ms"])
        norm = curves[order] / np.maximum(curves[order].max(1, keepdims=True), 1e-9)
        ax = axes[i, 0]
        ax.imshow(norm, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                  extent=[lags_ms[0], lags_ms[-1], len(order), 0])
        ax.axvline(0, color="#C0392B", lw=1.2)
        ax.set_ylabel(f"{s}\nchannel (by peak lag)")
        ax.set_title("R² vs offset, each channel scaled to its peak" if i == 0 else "")
        pk = np.array([stats[c]["peak_lag_ms"] for c in tuned])
        ax = axes[i, 1]
        ax.hist(pk, bins=np.arange(lags_ms[0], lags_ms[-1] + args.bin_ms, 2 * args.bin_ms), color="0.35")
        ax.axvline(0, color="#C0392B", lw=1.2)
        ax.set_title("peak offset of each tuned channel" if i == 0 else "")
        ax = axes[i, 2]
        ax.scatter(pk, [stats[c]["peak_r2"] for c in tuned], s=14, color="0.2")
        ax.axvline(0, color="#C0392B", lw=1.2)
        ax.set_title("strength (peak R²) vs peak offset" if i == 0 else "")
    for ax in axes[-1]:
        ax.set_xlabel("offset (ms): > 0 channel leads velocity, < 0 lags")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "lag_tuning.png"), dpi=150, facecolor="white")
    print(f"wrote {args.out_dir}/lag_tuning.png")


if __name__ == "__main__":
    main()
