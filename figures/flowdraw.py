"""
Small drawing kit for flow diagrams inside a figure, in millimetres with y pointing down (fig_threshold.py,
fig_refit.py). region(fig, ...) returns an axes whose data coordinates are mm from that region's top-left corner.
"""
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import style

INK, MUTED = style.INK, style.MUTED
FAINT = "#9a9ea8"
CHIP = "#6a5acd"
CHIP_T = "#eeedf9"
HOST = "#2a7f62"
HOST_T = "#e6f3ee"
REF = "#5b6170"
REF_T = "#f1f2f5"
BAD = "#b8423b"


def region(fig, Wmm, Hmm, x, y, w, h):
    ax = fig.add_axes([x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.7, r=1.6, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def txt(ax, x, y, s, size=5.8, color=INK, weight="normal", ha="left", va="top", style_="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style_,
                   zorder=z, linespacing=1.25, **kw)


def arrow(ax, p0, p1, color=FAINT, lw=0.8, rad=0.0, head=3.5, z=3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head * 0.4},head_width={head * 0.25}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=z, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def lock(ax, x, y, s=2.4, color=CHIP):
    """A padlock, its body's top-left at (x, y), s mm wide."""
    ax.add_patch(Rectangle((x, y + 0.45 * s), s, 0.62 * s, fc=color, ec="none", zorder=7))
    ax.add_patch(Arc((x + s / 2, y + 0.45 * s), 0.62 * s, 0.9 * s, theta1=180, theta2=360, color=color,
                     lw=1.1, zorder=7))                      # y points down: the shackle is the lower half-arc
    ax.add_patch(Circle((x + s / 2, y + 0.72 * s), 0.09 * s, fc="white", ec="none", zorder=8))


def raster(ax, x, y, w, h, rows=6, seed=0, color=INK, density=0.18):
    rng = np.random.default_rng(seed)
    for r in range(rows):
        yy = y + (r + 0.5) * h / rows
        for t in np.flatnonzero(rng.random(int(w * 2)) < density):
            ax.plot([x + t / 2] * 2, [yy - 0.32 * h / rows, yy + 0.32 * h / rows], color=color, lw=0.5, zorder=6)


def trace(ax, x, y, w, h, seed=0, color=INK, lw=0.9, noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 120)
    v = 0.5 * np.sin(2 * np.pi * 1.7 * t + 0.6) + 0.3 * np.sin(2 * np.pi * 4.1 * t)
    v = v + noise * rng.standard_normal(len(t)).cumsum() / 12
    v = (v - v.min()) / (v.max() - v.min())
    ax.plot(x + w * t, y + h * (1 - v), color=color, lw=lw, zorder=6)


def columns(ax, x, y, w, h, sizes, fc, ec, labels=True, size=4.6):
    """Layers as rounded columns (height ~ log size) joined by thin lines, inside x..x+w, centred in y..y+h."""
    cx = np.linspace(x + 1.0, x + w - 1.0, len(sizes))
    yc = y + h / 2
    for i, (c, n) in enumerate(zip(cx, sizes)):
        hh = min(h - 2.5, 1.6 + 1.6 * np.log2(max(n, 32) / 32))
        if i:
            ax.plot([cx[i - 1] + 0.7, c - 0.7], [yc, yc], color=FAINT, lw=0.5, zorder=5)
        box(ax, c - 0.7, yc - hh / 2, 1.4, hh, fc=fc, ec=ec, lw=0.5, r=0.5, z=6)
        if labels:
            txt(ax, c, yc + hh / 2 + 0.3, str(n), size=size, color=MUTED, ha="center")
    return cx, yc


def step_tag(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 1.9, fc=color, ec="none", zorder=8))
    ax.text(x, y + 0.08, str(n), fontsize=6.4, color="white", fontweight="bold", ha="center", va="center", zorder=9)
