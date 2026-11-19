"""
Supplementary figure: when each task's input channels carry their information, for all three experiments: the motor
lag figure (fig_motor_lag.py) repeated for audio and speech, with one pipeline (collect_lag_tuning.py) for all rows.

Rows: motor (Indy, 36 sessions), audio (monkeys N and G, 9 recording days), speech (T12, its training sessions).
    a  one session: every tuned channel's R^2 against offset, each scaled to its own peak, sorted by peak offset
    b  all sessions: where each tuned channel's tuning peaks, with the share of channels tuned and the spread
    c  all sessions: the median tuning curve (scaled to each channel's peak) and its interquartile band, with the
       Speck SNN's input delays marked and the share of the peak each one sees
Offset > 0: the channel's activity precedes what is decoded. A channel is tuned at peak R^2 >= 0.02.

    python3 fig_lag_tuning.py [--speech-dir speech]     -> figures/fig_lag_tuning.{pdf,svg,png}
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)

HERE = Path(__file__).parent
LAG = WORKSPACE / "lag_tuning"
XLIM = 1200
ROWS = {  # experiment: (title, example session, the Speck SNN's input delays in ms, as offsets)
    "motor": ("Motor (I): hand velocity", "indy_20160921_01", [0, 40, 100, 200]),
    "audio": ("Audio: heard sound", "Neil_07_31_13", [320, 160, 80, 40, 0, -40, -80, -160, -320]),
    "speech": ("Speech: attempted phonemes", "t12.2022.06.14", [0, 80, 260, 500, 300, 600, 1200]),
}


def load(exp_dir):
    out = {}
    for f in sorted(glob.glob(str(LAG / exp_dir / "lag_tuning_*.json"))):
        d = json.load(open(f))
        thr = d["settings"]["min_r2"]
        out[d["session"]] = (np.array(d["lags_ms"]), [c for c in d["channels"] if c["peak_r2"] >= thr],
                             len(d["channels"]))
    return out


def taps(ax, ms, values=None):
    for t in ms:
        if abs(t) <= XLIM:
            ax.axvline(t, color=style.COLOR["speck"], lw=0.7, ls="--", zorder=1)
    off = [t for t in ms if abs(t) > XLIM]
    if off:
        ax.annotate(f"tap at {', '.join(f'{t:,}' for t in off)} ms off scale", xy=(1, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=5.5, color=style.COLOR["speck"])


def row(axes, exp, data, title, example, delays, draft):
    lags = next(iter(data.values()))[0]
    bin_ms = float(lags[1] - lags[0])
    # a: one session
    ax = axes[0]
    ex = example if example in data else next(iter(data))
    _, ch, n_ch = data[ex]
    ch = sorted(ch, key=lambda c: c["peak_lag_ms"])
    if ch:
        img = np.array([np.array(c["curve"]) / c["peak_r2"] for c in ch])
        ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  extent=[lags[0] - bin_ms / 2, lags[-1] + bin_ms / 2, len(ch), 0])
    ax.axvline(0, color=style.MUTED, lw=0.6)
    ax.grid(False)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylabel(f"{title}\n\ntuned channels, by peak")
    ax.set_title(f"One session: {len(ch)} of {n_ch} tuned", loc="left")
    # b: peak offsets, all sessions
    ax = axes[1]
    peaks = np.array([c["peak_lag_ms"] for _, chs, _ in data.values() for c in chs])
    n_all = sum(n for *_, n in data.values())
    w = max(bin_ms, 20.0) * (1 if bin_ms >= 40 else 2)                 # 40 ms histogram bins everywhere
    edges = np.arange(-XLIM - w / 2, XLIM + w, w)
    ax.hist(peaks, bins=edges, color=style.MUTED, alpha=0.8)
    taps(ax, delays)
    q1, med, q3 = np.percentile(peaks, [25, 50, 75])
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylabel("channels")
    ax.set_title(f"All {len(data)} session{'s' if len(data) != 1 else ''}", loc="left")
    ax.text(0.03, 0.97, f"{100 * len(peaks) / n_all:.0f}% of {n_all:,} tuned\nmedian {med:+.0f} ms\n"
            f"IQR {q1:+.0f} to {q3:+.0f} ms", transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    # c: median normalized curve, the Speck SNN's delays
    ax = axes[2]
    allc = np.array([np.array(c["curve"]) / c["peak_r2"] for _, chs, _ in data.values() for c in chs])
    lo, mid, hi = np.percentile(allc, [25, 50, 75], axis=0)
    ax.fill_between(lags, lo, hi, color=style.MUTED, alpha=0.25, lw=0)
    ax.plot(lags, mid, color=style.INK, lw=1.2)
    taps(ax, delays)
    shown = sorted(t for t in delays if abs(t) <= XLIM)
    for t in shown:
        v = float(np.interp(t, lags, mid))
        ax.plot(t, v, "o", color=style.COLOR["speck"], ms=3, zorder=4)
        if len(shown) <= 5:
            ax.text(t - 10, v, f"{v:.2f}", ha="right", va="center", fontsize=5.5, color=style.COLOR["speck"],
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("R² / channel's peak")
    ax.set_title("Median tuning curve", loc="left")
    if draft:
        for a in axes:
            a.text(0.5, 0.5, draft, transform=a.transAxes, ha="center", va="center", fontsize=11, color="#c0392b",
                   alpha=0.35, rotation=20, zorder=10)
    return dict(tuned=len(peaks), channels=n_all, median=med, iqr=(q1, q3), taps_seen={
        t: round(float(np.interp(t, lags, mid)), 2) for t in shown})


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--speech-dir", default="speech", help="lag_tuning/<this> for the speech row")
    pa.add_argument("--draft", default=None, help="watermark text for the speech row")
    a = pa.parse_args()
    style.apply()
    fig, axes = plt.subplots(3, 3, figsize=(style.DOUBLE, 150 * style.MM),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1], wspace=0.45, hspace=0.55))
    summary = {}
    for i, (exp, (title, example, delays)) in enumerate(ROWS.items()):
        data = load(a.speech_dir if exp == "speech" else exp)
        if not data:
            for ax in axes[i]:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", color=style.MUTED)
                ax.set_axis_off()
            continue
        summary[exp] = row(axes[i], exp, data, title, example, delays, a.draft if exp == "speech" else None)
    for ax in axes[-1]:
        ax.set_xlabel("offset (ms; > 0: activity leads)")
    for ax, k in zip(axes.flat, "abcdefghi"):
        style.panel_label(ax, k, x=-0.24)
    style.save(fig, HERE / "figures" / "fig_lag_tuning")
    for exp, s in summary.items():
        print(f"{exp:7s} {s['tuned']:,}/{s['channels']:,} tuned ({100 * s['tuned'] / s['channels']:.0f}%), median "
              f"{s['median']:+.0f} ms, IQR {s['iqr'][0]:+.0f} to {s['iqr'][1]:+.0f} | Speck delays see {s['taps_seen']}")


if __name__ == "__main__":
    main()
