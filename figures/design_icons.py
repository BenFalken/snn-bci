"""
Small explanatory pictures for the design figures (fig_lsnn_design.py, fig_speck_design.py), drawn in mm on an axes
whose y points down. Each takes (ax, x, y, w, h) = the box it fills.
"""
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

import style

INK, MUTED = style.INK, style.MUTED
FAINT = "#9a9ea8"
LSNN = style.COLOR["lsnn"]
LSNN_L = "#f6c9ab"
PLAIN = "#8a8f9c"
HOST = "#2a7f62"


def t(ax, x, y, s, size=4.6, color=MUTED, ha="left", va="top", weight="normal", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=9, linespacing=1.2, **kw)


def arr(ax, p0, p1, color=FAINT, lw=0.6, rad=0.0, head=2.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head * 0.4},head_width={head * 0.25}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=7, shrinkA=0,
                                 shrinkB=0))


def loop_arrow(ax, cx, cy, r=1.6, color=LSNN, lw=0.9):
    """A clear circular arrow (recurrence), centred at (cx, cy)."""
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=200, theta2=520 - 40, color=color, lw=lw, zorder=8))
    a = np.deg2rad(480)
    x0, y0 = cx + r * np.cos(a), cy - r * np.sin(a)
    ax.add_patch(FancyArrowPatch((x0 - 0.05, y0 - 0.6), (x0 + 0.35, y0 + 0.35),
                                 arrowstyle="-|>,head_length=1.0,head_width=0.7", color=color, lw=lw, zorder=8))


# ---------------------------------------------------------------- the LSNN's core
def recurrent_ring(ax, x, y, w, h, n=8, frac=0.9, seed=0):
    """A recurrent layer: neurons on a ring, each feeding several others (chords); adaptive ones filled orange."""
    cx, cy, r = x + w / 2, y + h / 2, min(w, h) / 2 - 1.2
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pts = [(cx + r * np.cos(a), cy - r * np.sin(a)) for a in ang]
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in rng.choice([k for k in range(n) if k != i], 2, replace=False):
            arr(ax, pts[i], pts[j], color="#e8b394", lw=0.45, rad=0.15, head=1.6)
    for i, p in enumerate(pts):
        adaptive = i != n - 1 if frac > 0 else False
        ax.add_patch(Circle(p, 0.9, fc=LSNN if adaptive else "white", ec=LSNN if adaptive else PLAIN, lw=0.7,
                            zorder=8))


def alif_grid(ax, x, y, w, h, n=10, frac=0.9, seed=1):
    """n x n neurons: frac adaptive (orange), the rest with a fixed threshold (grey outline)."""
    rng = np.random.default_rng(seed)
    plain = set(rng.choice(n * n, int(round(n * n * (1 - frac))), replace=False))
    d = min(w, h) / n
    for k in range(n * n):
        i, j = divmod(k, n)
        ax.add_patch(Circle((x + (j + 0.5) * d, y + (i + 0.5) * d), d * 0.36,
                            fc="white" if k in plain else LSNN, ec=PLAIN if k in plain else LSNN, lw=0.5, zorder=8))


def threshold_trace(ax, x, y, w, h):
    """Spikes (ticks) and two thresholds: fixed (grey dashed) and adaptive (orange: rises at each spike, relaxes)."""
    T = 200
    spikes = [30, 40, 48, 55, 120, 128]
    th = np.ones(T)
    b = 0.0
    for k in range(T):
        b *= np.exp(-1 / 45)
        if k in spikes:
            b += 0.35
        th[k] = 1 + b
    X = lambda k: x + w * k / T
    Y = lambda v: y + h - (v - 0.6) / 1.8 * h
    ax.plot([X(0), X(T)], [Y(1), Y(1)], color=PLAIN, lw=0.6, ls=(0, (2, 1.4)), zorder=8)
    ax.plot([X(k) for k in range(T)], [Y(v) for v in th], color=LSNN, lw=0.9, zorder=8)
    for s in spikes:
        ax.plot([X(s)] * 2, [y + h, y + h - 1.2], color=INK, lw=0.6, zorder=8)


def filter_kernels(ax, x, y, w, h, taus=(0.02, 0.1, 0.4, 1.3)):
    """Synaptic filters: the response to one input spike, for time constants 20 ms to 1.3 s (unit gain)."""
    tt = np.linspace(0, 1.5, 120)
    for k, tau in enumerate(taus):
        v = np.exp(-tt / tau)
        ax.plot(x + w * tt / 1.5, y + h - h * 0.92 * v, color=LSNN, lw=0.8, alpha=0.35 + 0.65 * k / (len(taus) - 1),
                zorder=8)
    ax.plot([x, x + w], [y + h, y + h], color=FAINT, lw=0.5, zorder=7)


# ---------------------------------------------------------------- readouts
def smooth_cascade(ax, x, y, w, h, stages=2, color=LSNN, out="velocity", trace=True):
    """Spikes -> leaky stage(s) -> a smooth output trace."""
    rng = np.random.default_rng(4)
    ys = y + h * 0.5
    for s in np.flatnonzero(rng.random(18) < 0.45):
        ax.plot([x + s * 0.5] * 2, [ys - 1.2, ys + 1.2], color=INK, lw=0.5, zorder=8)
    xx = x + 10.5
    arr(ax, (x + 9.0, ys), (xx, ys))
    for k in range(stages):
        tt = np.linspace(0, 1, 30)
        ax.add_patch(Rectangle((xx, ys - 2.4), 5.5, 4.8, fc="white", ec=color, lw=0.6, zorder=8))
        ax.plot(xx + 0.6 + 4.3 * tt, ys + 1.8 - 3.6 * np.exp(-tt * 3), color=color, lw=0.7, zorder=9)
        xx += 5.5
        arr(ax, (xx, ys), (xx + 1.6, ys))
        xx += 1.6
    if not trace:
        return xx
    tt = np.linspace(0, 1, 60)
    v = 0.6 * np.sin(2 * np.pi * 1.2 * tt) + 0.3 * np.sin(2 * np.pi * 2.6 * tt + 1)
    ax.plot(xx + 0.4 + (w - (xx - x) - 0.8) * tt, ys - 1.9 * v, color=INK, lw=0.9, zorder=8)
    t(ax, xx + 0.4, ys + 2.6, out, size=4.4)
    return xx


def spectrogram(ax, x, y, w, h, seed=2, cmap_color=(0.0, 0.45, 0.70)):
    rng = np.random.default_rng(seed)
    nb, nt = 12, 18
    base = np.convolve(rng.random(nt), np.ones(3) / 3, mode="same")
    for j in range(nt):
        for i in range(nb):
            v = np.clip(base[j] * np.exp(-((i - 3 - 4 * base[j]) ** 2) / 8) + 0.1 * rng.random(), 0, 1)
            ax.add_patch(Rectangle((x + j * w / nt, y + h - (i + 1) * h / nb), w / nt, h / nb,
                                   fc=(*cmap_color, 0.12 + 0.85 * v), ec="none", zorder=8))


def phoneme_scores(ax, x, y, w, h, color=LSNN, seq="M · AY · T R EH", seed=3):
    rng = np.random.default_rng(seed)
    n, m = 7, 14
    hot = rng.integers(0, n, m)
    for j in range(m):
        for i in range(n):
            ax.add_patch(Rectangle((x + j * w / m, y + i * (h - 2.6) / n), w / m * 0.9, (h - 2.6) / n * 0.85,
                                   fc=color if i == hot[j] and j % 3 != 1 else "#eef0f3", ec="none", zorder=8))
    t(ax, x, y + h - 2.0, seq, size=4.3, color=INK, family="monospace")


def population(ax, x, y, w, h, color=style.COLOR["speck"]):
    """Two rows of 18 output neurons (one row per axis), each with a preferred velocity; firing (fill) weights them;
    the velocity is their rate-weighted mean (arrow)."""
    rng = np.random.default_rng(6)
    for r, (lab, peak) in enumerate((("x", 0.65), ("y", 0.3))):
        yy = y + 1.5 + r * h / 2
        rates = np.exp(-((np.linspace(0, 1, 18) - peak) ** 2) / 0.02) * (0.7 + 0.3 * rng.random(18))
        for k, v in enumerate(rates):
            ax.add_patch(Circle((x + 3 + k * (w - 6) / 17, yy), 0.62, fc=color, ec=color, lw=0.4, alpha=0.1 + 0.9 * v,
                                zorder=8))
        px = x + 3 + peak * (w - 6)
        ax.plot([px, px], [yy - 1.4, yy + 1.4], color=INK, lw=0.8, zorder=9)
        t(ax, x, yy, lab, size=4.4, color=INK, va="center")
    t(ax, x + w / 2, y + h + 0.2, "preferred velocity  →  rate-weighted mean", size=4.2, ha="center")


# ---------------------------------------------------------------- training
def loso(ax, x, y, w, h, color=LSNN):
    """Pre-train on all other sessions, then fine-tune on the session itself."""
    bw, gap = 2.4, 0.6
    for k in range(6):
        ax.add_patch(Rectangle((x + k * (bw + gap), y + 1.0), bw, 4.0, fc="#d9dbe1" if k != 3 else "white",
                               ec=FAINT if k != 3 else color, lw=0.6, ls="-" if k != 3 else (0, (1.5, 1)), zorder=8))
    t(ax, x, y + 5.6, "other sessions", size=4.2)
    xx = x + 6 * (bw + gap) + 0.4
    arr(ax, (xx, y + 3), (xx + 3, y + 3))
    t(ax, xx + 3.4, y + 1.0, "pre-train", size=4.4, color=INK, weight="bold")
    arr(ax, (xx + 13.2, y + 3), (xx + 16.2, y + 3))
    ax.add_patch(Rectangle((xx + 16.6, y + 1.0), bw, 4.0, fc=LSNN_L, ec=color, lw=0.7, zorder=8))
    t(ax, xx + 19.8, y + 1.0, "fine-tune on\nthis session", size=4.4, color=INK, weight="bold")


def noise_dropout(ax, x, y, w, h):
    """Training noise: the input with white noise and a constant offset; dropout: some units silenced each step."""
    tt = np.linspace(0, 1, 50)
    v = 0.5 * np.sin(2 * np.pi * 1.5 * tt)
    rng = np.random.default_rng(3)
    ax.plot(x + 11 * tt, y + 3.2 - 1.6 * v, color=FAINT, lw=0.8, zorder=8)
    ax.plot(x + 11 * tt, y + 2.4 - 1.6 * (v + 0.25 * rng.standard_normal(50)), color=LSNN, lw=0.6, zorder=8)
    t(ax, x, y + 5.6, "noise + offset", size=4.2)
    gx = x + 15
    for k in range(12):
        i, j = divmod(k, 4)
        dead = k in (1, 6, 11)
        ax.add_patch(Circle((gx + j * 2.0, y + 1.4 + i * 1.7), 0.6, fc="white" if dead else LSNN,
                            ec=FAINT if dead else LSNN, lw=0.5, zorder=8))
        if dead:
            ax.plot([gx + j * 2.0 - 0.6, gx + j * 2.0 + 0.6], [y + 0.8 + i * 1.7, y + 2.0 + i * 1.7], color=INK,
                    lw=0.5, zorder=9)
    t(ax, gx, y + 5.6, "dropout", size=4.2)


def day_gains(ax, x, y, w, h, days=8, color=LSNN):
    """A learned gain per channel and day (bars), in front of one shared network."""
    rng = np.random.default_rng(7)
    for d in range(days):
        xx = x + d * 2.4
        for c in range(4):
            g = 0.5 + rng.random()
            ax.add_patch(Rectangle((xx + c * 0.5, y + 4.6 - 2.2 * g), 0.4, 2.2 * g, fc=color, ec="none",
                                   alpha=0.45 + 0.4 * (c % 2), zorder=8))
    t(ax, x, y + 5.4, "day 1 … 19: a gain per channel", size=4.2)
    arr(ax, (x + days * 2.4 + 0.3, y + 3.0), (x + days * 2.4 + 3.0, y + 3.0))
    ax.add_patch(Rectangle((x + days * 2.4 + 3.3, y + 1.2), 3.6, 3.6, fc=LSNN_L, ec=color, lw=0.6, zorder=8))
    t(ax, x + days * 2.4 + 7.6, y + 1.3, "one shared\nnetwork", size=4.3, color=INK)


def dial(ax, x, y, r=2.0, color="#6a5acd"):
    """A dial (a tuned setting), centred at (x, y)."""
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.8, zorder=8))
    for a in np.linspace(np.pi * 1.15, -np.pi * 0.15, 6):
        ax.plot([x + r * np.cos(a), x + 1.25 * r * np.cos(a)], [y - r * np.sin(a), y - 1.25 * r * np.sin(a)],
                color=color, lw=0.45, zorder=8)
    ax.plot([x, x + 0.8 * r * np.cos(np.pi * 0.35)], [y, y - 0.8 * r * np.sin(np.pi * 0.35)], color=color, lw=1.0,
            zorder=9)


def weights(ax, x, y, w, h, color=HOST, seed=0):
    rng = np.random.default_rng(seed)
    n, m = 6, 5
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x + j * w / m, y + i * h / n), w / m * 0.86, h / n * 0.8, fc=color, ec="none",
                                   alpha=0.25 + 0.7 * rng.random(), zorder=8))


def traces(ax, x, y, w, h, color=MUTED):
    """Spikes -> leaky traces at several time constants (host memory)."""
    rng = np.random.default_rng(9)
    for s in np.flatnonzero(rng.random(12) < 0.5):
        ax.plot([x + s * 0.5] * 2, [y + h / 2 - 1.2, y + h / 2 + 1.2], color=INK, lw=0.5, zorder=8)
    xx = x + 6.5
    arr(ax, (x + 6.0, y + h / 2), (xx + 1.2, y + h / 2))
    tt = np.linspace(0, 1, 30)
    for k, tau in enumerate((0.08, 0.25, 0.7)):
        ax.plot(xx + 1.6 + 6 * tt, y + h / 2 + 2.0 - 3.6 * np.exp(-tt / tau), color=color, lw=0.7,
                alpha=0.4 + 0.3 * k, zorder=8)
    return xx + 8.2


def self_loop(ax, cx, top, color=LSNN, lw=0.9, width=1.2, height=2.6, head=1.0):
    """Recurrence as a self-loop: an arrow that leaves the top of a layer (drawn as a column whose top edge is at
    y = top, in an axis whose y grows downwards) and curves back into it, its head pointing down into the layer."""
    from matplotlib.path import Path
    p = Path([(cx + width / 2, top), (cx + width * 1.6, top - height), (cx - width * 1.6, top - height),
              (cx - width / 2, top)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(FancyArrowPatch(path=p, arrowstyle=f"-|>,head_length={head},head_width={head * 0.7}", color=color,
                                 lw=lw, zorder=8, shrinkA=0, shrinkB=0))


def read_arrow(ax, p0, p1, color=FAINT, lw=0.6, rad=0.0, head=0.8):
    """A readout arrow from a layer to the readout node."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head},head_width={head * 0.6}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=6, shrinkA=0,
                                 shrinkB=0))
