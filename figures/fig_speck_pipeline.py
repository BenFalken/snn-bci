"""
How a Speck SNN decodes, step by step, drawn with real motor data (indy 2016-04-11, test split): the host bins and
binarizes the spikes, keeps the last 200 ms of them in a ring buffer, and at every 4 ms step reads four columns of it
(now, 40, 100 and 200 ms ago) into one 384-long input. Its events enter the chip's first core; the cores run
integrate-and-fire layers (no leak, no recurrence); the last layer's spikes return to the host, which smooths and
decodes them.

    a  the pipeline: recording -> ring buffer with its delay taps -> chip input -> chip layers -> host readout
    b  the same step in time: each tap is the recording shifted by its delay

    python3 fig_speck_pipeline.py      -> figures/fig_speck_pipeline.{pdf,svg,png}
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import style

HERE = os.path.dirname(os.path.abspath(__file__))
SNNDEC = os.path.dirname(HERE)                               # the repository root (snndec)
SESSION = os.path.expanduser("~/Downloads/spike_bmi_lsnn_data/snn_datasets/bmi/indy/mua_8_group/indy_20160411_01")
SCALERS = os.path.expanduser("~/Downloads/spike_bmi_lsnn/main/snn_training/velocity_scalers.json")
STEP_MS = 4
TAPS = [0, 10, 25, 50]                                      # steps back: 0, 40, 100, 200 ms
TAP_COLORS = ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]
BOX = "#6a5acd"
EDGE = "#4a4f5c"
HOST_BG = "#f4f4f6"
CHIP_BG = "#f1eefb"


def spikes():
    sys.path.insert(0, SNNDEC)
    from snndec import data
    x = data.motor_split(SESSION, "test", SCALERS)[0].x
    x = (x > 0).astype(float)                                   # the motor Speck input: counts clipped to 1
    t0 = 2000 + int(np.argmax(np.convolve(x[2000:6000].sum(1), np.ones(250), "valid")))   # a busy second
    return x[t0:t0 + 250]                                       # 1 s, 250 steps


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=EDGE, zorder=5, **kw))


def region(ax, x0, x1, y0, y1, label, fc, ec=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15", fc=fc,
                                ec=ec or "none", lw=1.0, ls=ls, zorder=0))
    ax.text((x0 + x1) / 2, y1 - 0.08, label, ha="center", va="top", fontsize=7, fontweight="bold",
            color=ec or style.MUTED)


def panel_pipeline(ax, x):
    now = len(x) - 1
    buf = x[now - TAPS[-1]:now + 1].T                          # 96 channels x 51 steps, oldest left
    # 1. the recording
    ax.imshow(1 - x.T, extent=[0.3, 3.1, 0.6, 3.6], aspect="auto", cmap="gray", vmin=0, vmax=1, zorder=2)
    ax.add_patch(Rectangle((0.3 + 2.8 * (now - TAPS[-1]) / len(x), 0.6), 2.8 * (TAPS[-1] + 1) / len(x), 3.0,
                           fill=False, ec=TAP_COLORS[0], lw=1.0, zorder=3))
    ax.text(1.7, 0.42, "96 channels, binned at 4 ms,\nbinarized (spike or not)", ha="center", va="top", fontsize=6)
    ax.text(1.7, 3.75, "recording (1 s)", ha="center", va="bottom", fontsize=6.5)
    arrow(ax, (3.2, 2.1), (3.75, 2.1))
    # 2. the ring buffer and its taps
    bx0, bx1, by0, by1 = 3.85, 6.45, 0.6, 3.6
    ax.imshow(1 - buf, extent=[bx0, bx1, by0, by1], aspect="auto", cmap="gray", vmin=0, vmax=1, zorder=2)
    w = (bx1 - bx0) / buf.shape[1]
    for d, c in zip(TAPS, TAP_COLORS):
        xc = bx1 - (d + 1) * w
        ax.add_patch(Rectangle((xc, by0), w, by1 - by0, fill=False, ec=c, lw=1.6, zorder=4))
        ax.text(xc + w / 2, by1 + 0.06, f"{d * STEP_MS}", ha="center", va="bottom", fontsize=5.5, color=c)
    ax.text((bx0 + bx1) / 2 - 0.15, by1 + 0.3, "ring buffer, last 200 ms (taps)", ha="center", va="bottom", fontsize=6.5)
    ax.text((bx0 + bx1) / 2, 0.42, "96 × 51 bits = 612 bytes;\none column written per step", ha="center", va="top",
            fontsize=6)
    arrow(ax, (6.55, 2.1), (7.05, 2.1))
    # 3. the chip input: the four tapped columns side by side
    iy0, ih = 0.6, 3.0 / 4
    for k, (d, c) in enumerate(zip(TAPS, TAP_COLORS)):
        col = buf[:, buf.shape[1] - 1 - d]
        y0 = iy0 + (3 - k) * ih
        ax.imshow(1 - col[:, None], extent=[7.15, 7.45, y0 + 0.03, y0 + ih - 0.03], aspect="auto", cmap="gray",
                  vmin=0, vmax=1, zorder=2)
        ax.add_patch(Rectangle((7.15, y0 + 0.03), 0.3, ih - 0.06, fill=False, ec=c, lw=1.2, zorder=3))
    ax.text(7.3, 3.75, "384\ninput lines", ha="center", va="bottom", fontsize=6.5)
    # core 1 holds a weight for every input line and neuron: that memory is the input budget
    ax.add_patch(FancyBboxPatch((7.06, 0.52), 1.86, 3.18, boxstyle="round,pad=0,rounding_size=0.1", fill=False,
                                ec=BOX, lw=0.9, ls=(0, (2, 2)), zorder=1))
    ax.text(7.99, 0.42, "core 1: 384 × 128 weights\n(≤ 65,536: the input budget)", ha="center", va="top",
            fontsize=5, color=BOX, zorder=6)
    arrow(ax, (7.55, 2.1), (8.15, 2.1))
    # 4. the chip
    cores = [(128, "Hidden 1"), (256, "Hidden 2"), (256, "Hidden 3"), (36, "Output\n2 × 18")]
    for k, (n, lab) in enumerate(cores):
        x0 = 8.3 + k * 0.78
        h = 0.9 + 1.6 * np.log2(n / 32) / 3
        ax.add_patch(FancyBboxPatch((x0, 2.1 - h / 2), 0.55, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec=BOX, lw=1.0, zorder=3))
        ax.text(x0 + 0.275, 2.1, str(n), ha="center", va="center", fontsize=7, fontweight="bold", zorder=4)
        ax.text(x0 + 0.275, 2.1 - h / 2 - 0.08, lab, ha="center", va="top", fontsize=5.5, zorder=4)
        if k:
            arrow(ax, (x0 - 0.22, 2.1), (x0 - 0.02, 2.1))
    ax.text(10.3, 0.42, "one core per layer; integrate-\nand-fire, no leak, hard reset",
            ha="center", va="top", fontsize=5.5)
    arrow(ax, (11.2, 2.1), (11.65, 2.1))
    # 5. the host readout
    ax.add_patch(FancyBboxPatch((11.75, 1.45), 1.5, 1.3, boxstyle="round,pad=0,rounding_size=0.1", fc="white",
                                ec=EDGE, lw=0.9, zorder=3))
    ax.text(12.5, 2.1, "spike rates:\n2 EMAs, ~18 ms\neach; then the\nrate-weighted\nmean per cohort", ha="center",
            va="center", fontsize=5.5, zorder=4, linespacing=1.1)
    ax.text(12.5, 0.42, "velocity x, y\nevery 4 ms", ha="center", va="top", fontsize=6)
    # where each part runs
    region(ax, 0.1, 6.58, -0.15, 4.75, "HOST", HOST_BG)
    region(ax, 6.68, 11.5, -0.15, 4.75, "SPECK2F CHIP", CHIP_BG, BOX, ls=(0, (4, 3)))
    region(ax, 11.6, 13.4, -0.15, 4.75, "HOST", HOST_BG)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.2, 4.8)
    ax.axis("off")


def panel_taps(ax, x):
    """Three channels over 400 ms: the recording, and what each tap hands the chip at every step (the same train,
    shifted by its delay); the dashed line is one step ('now') and the dots what the chip receives then."""
    ch = np.argsort(x.sum(0))[-3:][::-1]
    t = np.arange(len(x)) * STEP_MS
    now = len(x) - 1
    span = slice(now - 100, now + 1)
    for r, c in enumerate(ch):
        for k, (d, col) in enumerate(zip(TAPS, TAP_COLORS)):
            y = -(r * 5.6 + k * 1.2)
            sh = np.zeros(len(x))
            sh[d:] = x[:len(x) - d, c]
            tt = t[span][sh[span] > 0]
            ax.vlines(tt, y, y + 0.85, color=col, lw=1.0)
            if sh[now] > 0:
                ax.plot(t[now], y + 0.42, "o", ms=3.5, color=col, mec="white", mew=0.5, zorder=4)
            if r == 0:
                ax.text(t[span][0] - 6, y + 0.42, f"{d * STEP_MS} ms", ha="right", va="center", fontsize=5.5, color=col)
        ax.text(t[span][-1] + 8, -(r * 5.6 + 1.5 * 1.2) + 0.42, f"channel {c + 1}", ha="left", va="center",
                fontsize=6)
    ax.axvline(t[now], color=EDGE, ls="--", lw=0.7)
    ax.text(t[now], 1.1, "now", ha="center", va="bottom", fontsize=6)
    ax.set_xlim(t[span][0] - 60, t[span][-1] + 70)
    ax.set_yticks([])
    ax.set_xticks(t[span][::25] , [f"{(v - t[now]):.0f}" for v in t[span][::25]])
    ax.set_xlabel("time relative to now (ms)")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(False)


def main():
    style.apply()
    x = spikes()
    fig = plt.figure(figsize=(style.DOUBLE, 125 * style.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.28)
    a = fig.add_subplot(gs[0])
    panel_pipeline(a, x)
    style.panel_label(a, "a", x=-0.01, y=0.98)
    b = fig.add_subplot(gs[1])
    panel_taps(b, x)
    style.panel_label(b, "b", x=-0.01, y=1.02)
    b.set_title("Each tap is the recording delayed: at every step the chip sees the same channel at four ages",
                fontsize=7, fontweight="normal", loc="left", x=0.06)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    style.save(fig, os.path.join(HERE, "figures", "fig_speck_pipeline"))
    print("wrote figures/fig_speck_pipeline")


if __name__ == "__main__":
    main()
