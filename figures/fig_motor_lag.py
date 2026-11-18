"""
Motor figure: M1 activity leads hand velocity, and the Speck SNN's delay taps sit across that lead.

Data: spike_bmi snn_training/channel_lag_tuning.py on all 36 Indy sessions (first 88% of each session only, so never
the decoders' test data; 20 ms bins, 40 ms Gaussian smoothing). For each multi-unit channel, the velocity tuning model
    rate(t) = b0 + bx vx(t + tau) + by vy(t + tau) + bs speed(t + tau)        (Moran & Schwartz 1999)
is fitted at every offset tau in -600..600 ms and scored by R^2; tau > 0 means the channel's activity precedes the
velocity it encodes. A channel counts as tuned at peak R^2 >= 0.02.

    a  one session: every tuned channel's R^2 against offset, each scaled to its own peak, sorted by peak offset
    b  all 36 sessions: where each tuned channel's tuning peaks (2,975 channels)
    c  all 36 sessions: the median tuning curve (scaled to each channel's peak) with its interquartile band; the
       Speck SNN's taps (0, 40, 100, 200 ms) marked, with the share of the peak each one sees

    python fig_motor_lag.py      -> figures/fig_motor_lag.{pdf,svg,png}
"""

from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).parent
LAG = WORKSPACE / "spike_bmi_lsnn_data/lag_tuning_all"
EXAMPLE = "indy_20160921_01"
TAPS = [0, 40, 100, 200]


def load():
    out = {}
    for f in sorted(glob.glob(str(LAG / "lag_tuning_indy_*.json"))):
        d = json.load(open(f))
        thr = d["settings"]["min_r2"]
        out[d["session"]] = (np.array(d["lags_ms"]), [c for c in d["channels"] if c["peak_r2"] >= thr])
    return out


def tap_marks(ax, ymax=None, label=True):
    for t in TAPS:
        ax.axvline(t, color=style.COLOR["speck"], lw=0.8, ls="--", zorder=1)
    if label:
        ax.text(TAPS[-1] + 8, (ymax if ymax is not None else ax.get_ylim()[1]) * 0.97, "Speck taps",
                color=style.COLOR["speck"], fontsize=6.5, va="top")


def main():
    data = load()
    lags, _ = next(iter(data.values()))
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 60 * style.MM),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1], wspace=0.42))

    # a: one session's channels
    ax = axes[0]
    _, ch = data[EXAMPLE]
    ch = sorted(ch, key=lambda c: c["peak_lag_ms"])
    img = np.array([np.array(c["curve"]) / c["peak_r2"] for c in ch])
    ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
              extent=[lags[0], lags[-1], len(ch), 0])
    ax.axvline(0, color=style.MUTED, lw=0.6)
    ax.grid(False)
    ax.set_xlim(-400, 400)
    ax.set_xlabel("offset (ms; > 0: activity leads)")
    ax.set_ylabel("tuned channels, by peak offset")
    ax.set_title(f"One session ({len(ch)} channels)", loc="left")

    # b: peak offsets, all sessions
    ax = axes[1]
    peaks = np.array([c["peak_lag_ms"] for _, chs in data.values() for c in chs])
    bins = np.arange(-410, 411, 20)
    ax.hist(peaks, bins=bins, color=style.MUTED, alpha=0.8)
    tap_marks(ax, label=False)
    lead = peaks[peaks > 0]
    q1, med, q3 = np.percentile(lead, [25, 50, 75])
    ax.set_xlim(-400, 400)
    ax.set_xlabel("peak offset (ms)")
    ax.set_ylabel("channels")
    ax.set_title(f"All 36 sessions ({len(peaks):,} channels)", loc="left")
    ax.text(0.03, 0.97, f"{100 * (peaks > 0).mean():.0f}% lead\nmedian {med:.0f} ms\nIQR {q1:.0f}–{q3:.0f} ms",
            transform=ax.transAxes, va="top", fontsize=6.5, bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # c: median normalized curve, taps
    ax = axes[2]
    allc = np.array([np.array(c["curve"]) / c["peak_r2"] for _, chs in data.values() for c in chs])
    lo, mid, hi = np.percentile(allc, [25, 50, 75], axis=0)
    ax.fill_between(lags, lo, hi, color=style.MUTED, alpha=0.25, lw=0)
    ax.plot(lags, mid, color=style.INK, lw=1.2)
    tap_marks(ax, label=False)
    ax.text(0.02, 0.97, "dashed: Speck taps\n(share of the peak)", transform=ax.transAxes, va="top", fontsize=6.5,
            color=style.COLOR["speck"], bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=5)
    for t in TAPS:
        v = float(np.interp(t, lags, mid))
        ax.plot(t, v, "o", color=style.COLOR["speck"], ms=3.5, zorder=4)
        ax.text(t - 8, v, f"{v:.2f}", ha="right", va="center", fontsize=6, color=style.COLOR["speck"],
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.set_xlim(-400, 400)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("offset (ms)")
    ax.set_ylabel("R² / channel's peak")
    ax.set_title("Median tuning curve", loc="left")

    for ax, k in zip(axes, "abc"):
        style.panel_label(ax, k, x=-0.22)
    style.save(fig, HERE / "figures" / "fig_motor_lag")
    print(f"wrote figures/fig_motor_lag; {len(peaks)} tuned channels, {100 * (peaks > 0).mean():.1f}% lead, "
          f"median lead {med:.0f} ms (IQR {q1:.0f}-{q3:.0f})")


if __name__ == "__main__":
    main()
