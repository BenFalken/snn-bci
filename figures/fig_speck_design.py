"""
How each task shaped its Speck SNN: the design process, read top to bottom (the LSNN's: fig_lsnn_design.py).

    origin       a spiking decoder
    demands      what each task asks of a decoder, numbered (1-3 per task)
    LSNN         one shared network for every task, and the few settings each task changes
    Speck SNN    each task's answer to the same decisions, and the deployed network on the chip's nine cores

Each design choice carries the number of the task property that motivated it. Settings are the final ones (the
snn-decoders configs, the design documents; speech = round 20's V2+C: three networks sharing one chip). Core placement
is sinabs make_config's for each deployed network; a core's fill = inputs x 2^ceil(log2 outputs) / its memory.

    python3 fig_speck_design.py      -> figures/fig_speck_design.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import style
import design_icons as di

HERE = Path(__file__).parent

INK, MUTED = style.INK, style.MUTED
FAINT = "#9a9ea8"
LSNN, SPECK = style.COLOR["lsnn"], style.COLOR["speck"]
SPECK_B = "#9c6b00"                              # speech's second network
SPECK_C = "#d9a640"                              # speech's third network (V2+C)
SPECK_D = "#a87300"                              # Speck text accent
LSNN_T, SPECK_T = "#fcefe7", "#fdf6e6"
CARD, CARD_E = "white", "#d8dbe2"
DEM_T = "#f3f5f8"
CHIP = "#6a5acd"
HOST = "#5b6170"
BADGE = "#3d4352"

W, H = 183.0, 177.0
GUT = 19.0
GAP = 3.0
CW = (W - GUT - 2 * GAP - 1.0) / 3
COLX = [GUT + i * (CW + GAP) for i in range(3)]
CORE_KB = [16, 16, 16, 32, 32, 64, 64, 16, 16]   # Speck2f weight memory per core (K weights)


def X(i):
    return COLX[i]


# ---------------------------------------------------------------- primitives (mm, y downwards)
def box(ax, x, y, w, h, fc, ec=None, lw=0.6, r=1.6, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc,
                                lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, size=6.0, color=INK, weight="normal", ha="left", va="top", style_="normal", z=6):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style_,
                   zorder=z, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=FAINT, lw=0.7, head=4, z=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head * 0.35},"
                                 f"head_width={head * 0.22}", color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def badge(ax, x, y, n, color=BADGE):
    """The number of a task property, as a small filled disc (x, y = its centre)."""
    ax.add_patch(Circle((x, y), 1.15, fc=color, ec="none", zorder=7))
    ax.text(x, y + 0.05, str(n), fontsize=4.6, color="white", fontweight="bold", ha="center", va="center",
            zorder=8)


def items(ax, x, y, lines, dot, size=5.8, step=3.0, extra=0.6):
    """Lines that start with the number of the property they answer (or a plain dot)."""
    for n, s in lines:
        if n:
            badge(ax, x + 1.0, y + 1.1, n)
        else:
            ax.add_patch(Circle((x + 1.0, y + 1.1), 0.42, color=dot, zorder=5, lw=0))
        txt(ax, x + 2.9, y, s, size=size)
        y += step * (s.count("\n") + 1) + extra
    return y


# ---------------------------------------------------------------- mini visuals
def budget_bars(ax, x, y, w, nets, h=2.2, gap=0.6):
    """The 512-input budget per network: one segment per delay tap."""
    for k, (n_ch, taps, color, lab) in enumerate(nets):
        yy = y + k * (h + gap)
        box(ax, x, yy, w, h, "white", ec=FAINT, lw=0.5, r=0.5, z=3)
        for t in range(taps):
            ax.add_patch(Rectangle((x + w * n_ch * t / 512, yy), w * n_ch / 512, h, fc=color, ec="white", lw=0.6,
                                   zorder=4, alpha=0.9))
        if lab:
            txt(ax, x + w + 0.8, yy + h / 2, lab, size=4.9, color=MUTED, va="center")


def raster(ax, x, y, w, h, frac, seed, color=SPECK, nx=34, ny=6):
    """One step of input lines; filled cells carry an event (the input's density)."""
    rng = np.random.default_rng(seed)
    on = rng.random((ny, nx)) < frac
    dx, dy = w / nx, h / ny
    for j in range(ny):
        for i in range(nx):
            ax.add_patch(Rectangle((x + i * dx + 0.12, y + j * dy + 0.12), dx - 0.24, dy - 0.24,
                                   fc=color if on[j, i] else "#eceef2", ec="none", zorder=4))


T0, T1 = -1.25, 0.38


def tline(ax, x, y, w, taps, shade=None, traces=None, ahead=None, color=SPECK):
    """One time axis for all tasks (s, relative to the output): delay taps (dots), host traces (fading bar), the
    task's own window (shaded)."""
    f = lambda t: x + w * (t - T0) / (T1 - T0)
    ax.plot([f(T0), f(T1)], [y, y], color=FAINT, lw=0.6, zorder=3)
    for t, s in ((-1.2, "−1.2 s"), (-0.8, "−0.8"), (-0.4, "−0.4"), (0.0, "0")):
        ax.plot([f(t)] * 2, [y, y + 0.7], color=FAINT, lw=0.5, zorder=3)
        txt(ax, f(t), y + 1.0, s, size=4.8, color=MUTED, ha="center")
    ax.plot([f(0)] * 2, [y - 3.2, y + 0.7], color=INK, lw=0.7, zorder=4)
    if shade:
        a, b, lab = shade
        ax.add_patch(Rectangle((f(a), y - 2.6), f(b) - f(a), 2.6, fc="#e3e6ec", ec="none", zorder=2))
        txt(ax, (f(a) + f(b)) / 2, y - 2.9, lab, size=4.8, color=MUTED, ha="center", va="bottom")
    if ahead:
        ax.add_patch(Rectangle((f(0), y - 2.6), f(ahead) - f(0), 2.6, fc="#e3e6ec", ec="none", zorder=2))
        txt(ax, f(ahead), y - 2.9, "look-ahead", size=4.8, color=MUTED, ha="right", va="bottom")
    if traces:
        n = 40
        for k in range(n):
            t = -traces * k / n
            ax.add_patch(Rectangle((f(t - traces / n), y - 2.2), w * (traces / n) / (T1 - T0), 1.2, fc=HOST,
                                   ec="none", alpha=0.55 * (1 - k / n) + 0.05, zorder=3))
        txt(ax, f(-traces), y - 2.6, "host traces", size=4.8, color=HOST, ha="left", va="bottom")
    for t in taps:
        ax.add_patch(Circle((f(t), y), 0.62, fc=color, ec="white", lw=0.4, zorder=6))


def topology(ax, x, y, w, h, nets, n_in, read="last"):
    """The deployed network(s) as columns (height ~ size): the input lines (white), then each layer on the chip; then
    the host readout (green), with an arrow from every layer it reads: the last layer ('last') or every layer ('all')."""
    sub = h / len(nets)
    wc = w - 5.0                                       # the chip's columns; the host readout to their right
    hx = x + w - 1.4
    ends = []
    for k, (layers, color) in enumerate(nets):
        yc = y + k * sub + sub / 2 - 0.8
        cols = [n_in[k]] + layers
        cx = np.linspace(x + 1.0, x + wc - 1.0, len(cols))
        for i, (c, n) in enumerate(zip(cx, cols)):
            hh = min(sub - 3.2, 1.4 + 1.6 * np.log2(max(n, 32) / 32))
            if i:
                ax.plot([cx[i - 1] + 0.7, c - 0.7], [yc, yc], color=FAINT, lw=0.5, zorder=4)
            box(ax, c - 0.7, yc - hh / 2, 1.4, hh, "white" if i == 0 else color, ec=FAINT if i == 0 else color,
                lw=0.5, r=0.5, z=5)
            txt(ax, c, yc + hh / 2 + 0.35, str(n), size=4.5, color=MUTED, ha="center")
            if i and (read == "all" or i == len(cols) - 1):
                ends.append((k, c, yc, hh, i == len(cols) - 1))
    # the host readout: one node, every read layer's spikes into it
    top, bot = y + 0.6, y + h - 2.4
    ym = (top + bot) / 2
    hh_r = min(bot - top, 4.0 + 1.2 * len(nets))
    box(ax, hx - 0.8, ym - hh_r / 2, 1.6, hh_r, di.HOST, ec=di.HOST, lw=0.5, r=0.5, z=5)
    txt(ax, hx, ym + hh_r / 2 + 0.35, "host", size=4.5, color=di.HOST, ha="center")
    if read == "last":                                 # the last layer: straight into the host
        for k, c, yc, hh, last in ends:
            di.read_arrow(ax, (c + 0.7, yc), (hx - 0.8, ym + (yc - ym) * 0.35), color=di.HOST, lw=0.5)
    else:                                              # every layer: a tick up from each into a bus above the network
        for k in sorted({e[0] for e in ends}):
            es = [e for e in ends if e[0] == k]
            yb = min(yc - hh / 2 for _, _, yc, hh, _ in es) - 0.7
            for _, c, yc, hh, _ in es:
                ax.plot([c, c], [yc - hh / 2, yb], color=di.HOST, lw=0.5, zorder=6)
            ax.plot([es[0][1], hx - 1.6], [yb, yb], color=di.HOST, lw=0.5, zorder=6)
            di.read_arrow(ax, (hx - 1.6, yb), (hx - 0.8, ym + (yb - ym) * 0.45), color=di.HOST, lw=0.5)


def core_map(ax, x, y, placement, s=4.4, g=0.7):
    """Speck2f's nine cores by index (3 x 3), each labelled with its weight memory; a used core is filled by the
    share of that memory its layer takes, coloured by network, the layer named inside."""
    box(ax, x - 1.0, y - 1.0, 3 * s + 2 * g + 2.0, 3 * s + 2 * g + 2.0, "white", ec=CHIP, lw=0.7, r=1.0,
        ls=(0, (3, 2)), z=3)
    for c in range(9):
        cx, cy = x + (c % 3) * (s + g), y + (c // 3) * (s + g)
        ax.add_patch(Rectangle((cx, cy), s, s, fc="#f1f2f5", ec="#cfd3db", lw=0.4, zorder=4))
        if c in placement:
            lab, frac, color = placement[c]
            ax.add_patch(Rectangle((cx, cy + s * (1 - frac)), s, s * frac, fc=color, ec="none", zorder=5))
            ax.text(cx + s / 2, cy + s * 0.64, lab, fontsize=4.3, color="white", fontweight="bold",
                    ha="center", va="center", zorder=6)
        ax.text(cx + s / 2, cy + 0.25, f"{CORE_KB[c]}K", fontsize=3.2,
                color="white" if c in placement and placement[c][1] > 0.85 else MUTED, ha="center", va="top",
                zorder=6)
    ax.text(x + (3 * s + 2 * g) / 2, y + 3 * s + 2 * g + 1.4, "Speck2f, 9 cores", fontsize=4.5, color=CHIP,
            ha="center", va="top", zorder=6)


def lsnn_icon(ax, x, y, width_n, h=7.5):
    hh = 2.2 + 1.5 * np.log2(width_n / 128)
    for i in range(2):
        cx = x + 1.2 + i * 4.2
        box(ax, cx - 0.9, y + h - hh, 1.8, hh, "white", ec=LSNN, lw=0.6, r=0.7, z=5)
        ax.add_patch(FancyArrowPatch((cx - 0.5, y + h - hh - 0.1), (cx + 0.5, y + h - hh - 0.1),
                                     connectionstyle="arc3,rad=-1.4", arrowstyle="-|>,head_length=0.8,head_width=0.5",
                                     color=LSNN, lw=0.6, zorder=6))
    ax.plot([x + 2.1, x + 4.5], [y + h - 1.0] * 2, color=FAINT, lw=0.5, zorder=4)


def mem_bar(ax, x, y, w, val, vmax, label, color=LSNN):
    box(ax, x, y, w, 1.9, "#f1e3da", r=0.5, z=3)
    box(ax, x, y, max(w * val / vmax, 0.8), 1.9, color, r=0.5, z=4)
    txt(ax, x + max(w * val / vmax, 0.8) + 1.0, y - 0.25, label, size=5.6, weight="bold")


def use(n_in, n_out, core):
    """A layer's share of its core's weight memory."""
    return min(1.0, n_in * 2 ** int(np.ceil(np.log2(n_out))) / (CORE_KB[core] * 1024))


# ---------------------------------------------------------------- content
TASKS = [
    ("Motor (I, II)", "Hand velocity every 4 ms, 96 channels of M1",
     ["M1 leads the hand by ~160 ms", "Velocity is smooth, and coded by\nbroadly tuned populations",
      "Tuning drifts between sessions"]),
    ("Audio", "A 128-band spectrum every 40 ms,\n96 channels of auditory cortex",
     ["Responses follow the sound within\ntens of milliseconds", "Spikes in at 4 ms, a graded\n"
      "spectrum out every 40 ms", "Responses shift from day to day"]),
    ("Speech", "Phonemes every 80 ms, 256 features\nof speech motor cortex",
     ["A phoneme unfolds over ~100 ms,\na word over several hundred", "256 features, the densest input of all",
      "19 recording days, one decoder"]),
]

LSNN_CARDS = [
    (256, [(0, "2 × 256 neurons, a step every 4 ms"), (2, "Two smoothing readout stages"),
           (3, "Pre-trained on the other sessions,\nthen fine-tuned: the largest gain\n(RMSE 37.3 → 32.0 mm/s)")],
     5, "recurrence +5%"),
    (256, [(1, "Reads real 4 ms spike times:\nten steps per 40 ms output"),
           (2, "A leaky readout to 128 bands"),
           (3, "4 ms spikes, not 40 ms counts:\nlevel with the LSTM and GRU")],
     5, "4 ms spikes +5%"),
    (1024, [(1, "2 × 1,024: width and recurrence\ncarry the memory a word needs"),
            (0, "An output every 80 ms, CTC"),
            (3, "A learned gain per channel and day,\nall 19 days trained together")],
     53, "recurrence +53%"),
]

C1, C2, C3 = SPECK, SPECK_B, SPECK_C
SPECK_COLS = [
    dict(btitle="96 channels × 4 taps = 384", bn=0, bars=[(96, 4, C1, "")],
         frac=0.062, etxt="spikes as 0/1 per 4 ms\n~24 events per step (6%)", en=0,
         taps=[0, -0.04, -0.10, -0.20], shade=(-0.20, -0.12, "M1 lead"), traces=None, ahead=None,
         mtxt="four taps, 0 to 200 ms, span the lead", mn=1,
         nets=[([128, 256, 256, 36], C1)], n_in=[384],
         cores={5: ("L1", use(384, 128, 5), C1), 3: ("L2", use(128, 256, 3), C1), 6: ("L3", use(256, 256, 6), C1),
                0: ("out", use(256, 36, 0), C1)},
         ctxt="four layers on four cores; the last holds\n36 output neurons, 18 per axis",
         rtxt="decoded on the chip: a population\nvector per axis, smoothed 2 × 18 ms", rn=2,
         cal="× 1 to 1.5", cal2="Jenkins × 2 to 3", cal3="readout refit:\nJenkins", extra=(3, "one network per session")),
    dict(btitle="48 channels × 9 taps = 432", bn=0, bars=[(48, 9, C1, "")],
         frac=0.10, etxt="real spikes, one event per channel\nper 4 ms step · ~42 per step (10%)", en=2,
         taps=[0.32, 0.16, 0.08, 0.04, 0, -0.04, -0.08, -0.16, -0.32], shade=None, traces=None, ahead=0.32,
         mtxt="the reference ±320 ms window,\ntaps dense at the output frame", mn=1,
         nets=[([128, 256, 256], C1)], n_in=[432],
         cores={5: ("L1", use(432, 128, 5), C1), 3: ("L2", use(128, 256, 3), C1), 6: ("L3", use(256, 256, 6), C1)},
         ctxt="three layers on three cores; the graded\noutput is decoded on the host",
         rtxt="Hidden 3 smoothed once, mapped\nlinearly to 128 bands", rn=2,
         cal="× 2 to 3", cal2="", cal3="readout refit", extra=(3, "no per-day normalization")),
    dict(btitle="All 256 features × 4 taps as two\nnetworks, + the top 64 to 1.2 s", bn=2,
         bars=[(128, 4, C1, "A: top 128"), (128, 4, C2, "B: other 128"), (64, 4, C3, "C: top 64, long taps")],
         frac=0.26, etxt="per-day gain, event if z ≥ 0.5\n~330 per step (26%), the densest", en=2,
         taps=[0, -0.08, -0.26, -0.30, -0.50, -0.60, -1.20], shade=None, traces=1.2, ahead=None,
         mtxt="taps to 0.5 s (C: 0.3, 0.6, 1.2 s); host\ntraces of the chip’s spikes reach 1.2 s", mn=1,
         nets=[([128, 256, 64], C1), ([128, 256, 64], C2), ([64, 256, 64], C3)], n_in=[512, 512, 256],
         cores={5: ("A1", use(512, 128, 5), C1), 3: ("A2", use(128, 256, 3), C1), 0: ("A3", use(256, 64, 0), C1),
                6: ("B1", use(512, 128, 6), C2), 4: ("B2", use(128, 256, 4), C2), 1: ("B3", use(256, 64, 1), C2),
                2: ("C1", use(256, 64, 2), C3), 7: ("C2", use(64, 256, 7), C3), 8: ("C3", use(256, 64, 8), C3)},
         ctxt="three networks side by side, meeting only\nin the host readout; all nine cores", read="all",
         rtxt="every layer of all three networks with\ntraces → 41 classes every 80 ms → CTC", rn=1,
         cal="× 2.5", cal2="", cal3="readout refit", extra=(3, "a learned gain per channel and day")),
]


# ---------------------------------------------------------------- figure
def draw_header(ax):
    """The origin node and the numbered task demands (shared by the Speck and LSNN design figures); returns the
    demands row's bottom (mm)."""
    # ---- origin
    ow, oh = 92, 12.0
    ox, oy = GUT + (W - GUT - 1.5) / 2 - ow / 2, 1.2
    box(ax, ox, oy, ow, oh, "white", ec=INK, lw=0.8, r=2.2, z=3)
    for k in range(5):
        yy = oy + 2.4 + k * 1.65
        for t in np.random.default_rng(k + 3).random(5) * 9:
            ax.plot([ox + 3 + t] * 2, [yy, yy + 1.0], color=INK, lw=0.6, zorder=4)
    arrow(ax, ox + 13.5, oy + oh / 2, ox + 17.3, oy + oh / 2, color=INK, lw=0.6, head=3)
    ax.add_patch(Circle((ox + 20.4, oy + oh / 2), 2.1, fc="white", ec=INK, lw=0.7, zorder=4))
    ax.plot([ox + 19.3, ox + 20.4, ox + 20.4, ox + 21.5], [oy + oh / 2 + 0.9, oy + oh / 2 + 0.9,
            oy + oh / 2 - 0.9, oy + oh / 2 - 0.9], color=INK, lw=0.5, zorder=5)
    arrow(ax, ox + 23.0, oy + oh / 2, ox + 26.8, oy + oh / 2, color=INK, lw=0.6, head=3)
    tt = np.linspace(0, 1, 60)
    ax.plot(ox + 27.6 + 7 * tt, oy + oh / 2 - 1.8 * np.sin(2 * np.pi * 1.3 * tt) * np.exp(-tt), color=INK,
            lw=0.7, zorder=4)
    txt(ax, ox + 38.0, oy + oh / 2, "A spiking decoder", size=8.4, weight="bold", va="center")

    # ---- demands
    dy, dh = 20.0, 22.5
    txt(ax, 1.2, dy + 1.2, "What the\ntask\ndemands", size=6.3, weight="bold")
    bus = oy + oh + 2.2
    ax.plot([ox + ow / 2] * 2, [oy + oh, bus], color=FAINT, lw=0.7, zorder=2)
    ax.plot([X(0) + CW / 2, X(2) + CW / 2], [bus, bus], color=FAINT, lw=0.7, zorder=2)
    for i, (name, what, facts) in enumerate(TASKS):
        x = X(i)
        arrow(ax, x + CW / 2, bus, x + CW / 2, dy - 0.3)
        box(ax, x, dy, CW, dh, DEM_T, r=1.8, z=2)
        txt(ax, x + 2.2, dy + 1.8, name, size=7.8, weight="bold")
        items(ax, x + 2.0, dy + 6.2, [(k + 1, f) for k, f in enumerate(facts)], INK, size=5.8)

    return dy + dh


def main():
    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]   # true bold, arrow glyphs
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # the task demands are drawn once, at the top of the LSNN figure; the badges here refer to them
    # ---- Speck band
    sy = 1.0
    rows, y = {}, sy + 9.0
    for k, h in (("bud", 10.0), ("enc", 8.4), ("mem", 12.6), ("chip", 21.5), ("read", 15.5), ("cal", 5.0),
                 ("day", 3.0)):
        rows[k] = (y, h)
        y += h + 1.6
    sh = y + 0.4 - sy
    box(ax, 0.5, sy, W - 1.0, sh, SPECK_T, r=2.5, z=0)
    txt(ax, 1.2, sy + 2.2, "Speck SNN", size=8.2, weight="bold", color=SPECK_D)
    for i, (name, _, _) in enumerate(TASKS):          # the task names (the demands are in the LSNN figure)
        txt(ax, X(i) + 2.2, sy + 2.4, name, size=7.2, weight="bold")
    gl = dict(bud=("Input budget", ""), enc=("Encoding", ""), mem=("Memory", ""), chip=("On the chip", ""),
              read=("Host readout", ""), cal=("Calibration", ""), day=("Across days", ""))
    for k, (a, b) in gl.items():
        y0 = rows[k][0]
        txt(ax, 1.2, y0 + 0.2, a, size=6.0, weight="bold")
        if b:
            txt(ax, 1.2, y0 + 3.1, b, size=5.1, color=MUTED)
    top, bot = rows["bud"][0] - 2.4, rows["day"][0] + rows["day"][1] + 1.2
    for i, s in enumerate(SPECK_COLS):
        x = X(i)
        box(ax, x, top, CW, bot - top, CARD, ec=CARD_E, lw=0.5, r=1.6, z=1)
        for k in ("enc", "mem", "chip", "read", "cal", "day"):
            ax.plot([x + 2.0, x + CW - 2.0], [rows[k][0] - 1.0] * 2, color="#eceef2", lw=0.5, zorder=2)
        # budget
        y0 = rows["bud"][0]
        hx = x + 2.0
        if s["bn"]:
            badge(ax, hx + 1.0, y0 + 1.1, s["bn"])
            hx += 2.9
        txt(ax, hx, y0, s["btitle"], size=5.9, weight="bold")
        two = len(s["bars"]) >= 2
        budget_bars(ax, x + 2.0, y0 + (5.6 if two else 3.6), CW - (19.0 if two else 4.0), s["bars"],
                    h=(0.95 if len(s["bars"]) > 2 else 1.5) if two else 2.6, gap=0.85 if len(s["bars"]) > 2 else 0.6)
        if not two:
            txt(ax, x + CW - 2.0, y0 + 6.8, "512", size=4.8, color=MUTED, ha="right")
        # encoding
        y0 = rows["enc"][0]
        raster(ax, x + 2.0, y0 + 0.3, 15.0, 6.0, s["frac"], seed=10 + i)
        if s["en"]:
            badge(ax, x + 19.4, y0 + 1.4, s["en"])
        txt(ax, x + 21.3, y0 + 0.3, s["etxt"], size=5.6)
        # memory
        y0 = rows["mem"][0]
        tline(ax, x + 2.0, y0 + 4.3, CW - 4.0, s["taps"], shade=s["shade"], traces=s["traces"], ahead=s["ahead"])
        badge(ax, x + 3.0, y0 + 8.9, s["mn"])
        txt(ax, x + 4.9, y0 + 7.8, s["mtxt"], size=5.5)
        # on the chip: the network, and where it sits
        y0, h0 = rows["chip"]
        topology(ax, x + 1.6, y0 + 0.2, 26.0, h0 - 2.4, s["nets"], s["n_in"], s.get("read", "last"))
        core_map(ax, x + CW - 18.6, y0 + 1.2, s["cores"])
        # readout: a picture, then what it does
        y0 = rows["read"][0]
        if i == 0:
            di.population(ax, x + 2.0, y0, CW - 4.0, 6.5)
        elif i == 1:
            xe = di.smooth_cascade(ax, x + 2.0, y0, 20.0, 8.0, stages=1, color=SPECK_D, trace=False)
            di.weights(ax, xe + 0.6, y0 + 1.4, 6.0, 5.5)
            di.arr(ax, (xe + 7.2, y0 + 4.0), (xe + 9.4, y0 + 4.0))
            di.spectrogram(ax, xe + 9.8, y0 + 0.6, CW - (xe + 9.8 - x) - 2.0, 7.0)
        else:
            xe = di.traces(ax, x + 2.0, y0 + 0.4, 18.0, 7.0)
            di.arr(ax, (xe, y0 + 4.0), (xe + 2.0, y0 + 4.0))
            di.weights(ax, xe + 2.4, y0 + 1.4, 6.0, 5.5)
            di.arr(ax, (xe + 9.0, y0 + 4.0), (xe + 11.0, y0 + 4.0))
            di.phoneme_scores(ax, xe + 11.4, y0 + 0.2, CW - (xe + 11.4 - x) - 2.0, 9.0, color=SPECK_D)
        badge(ax, x + 3.0, y0 + 10.6, s["rn"])
        txt(ax, x + 4.9, y0 + 9.5, s["rtxt"], size=5.4)
        # calibration: both ends, in software, once
        y0 = rows["cal"][0]
        di.dial(ax, x + 4.0, y0 + 2.6, r=1.9)
        txt(ax, x + 7.2, y0 + 0.3, "layer 1 " + s["cal"], size=6.2, weight="bold", color=SPECK_D)
        di.weights(ax, x + CW / 2 + 3.0, y0 + 0.4, 4.4, 4.2, seed=2)
        txt(ax, x + CW / 2 + 8.4, y0 + 0.3, s["cal3"], size=5.6, weight="bold", color="#2a7f62")
        # across days / sessions
        y0 = rows["day"][0]
        n3, t3 = s["extra"]
        badge(ax, x + 3.0, y0 + 1.1, n3)
        txt(ax, x + 4.9, y0, t3, size=5.6)

    # ---- the rule
    end = sy + sh
    Hn = end + 1.5                                       # the figure ends with the band (no header, no closing note)
    fig.set_size_inches(W * style.MM, Hn * style.MM)
    ax.set_ylim(Hn, 0)
    print("layout bottom (mm):", round(Hn, 1))
    style.save(fig, HERE / "figures" / "fig_speck_design")


if __name__ == "__main__":
    main()
