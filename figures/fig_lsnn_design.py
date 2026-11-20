"""
How each task shaped its LSNN, the counterpart of fig_speck_design.py: the same origin and numbered task demands, then
one shared network (drawn: recurrent layers, the 90/10 adaptive split, the adaptive threshold, the synaptic filters)
and each task's answer to the same decisions, with a picture for each: input, network, memory, readout, training.

Settings from the snn-decoders configs (motor_lsnn, audio_lsnn, speech_lsnn) and the design documents.

    python3 fig_lsnn_design.py      -> figures/fig_lsnn_design.{pdf,svg,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style
import design_icons as di
from fig_speck_design import CARD, CARD_E, CW, FAINT, GUT, INK, LSNN, MUTED, T0, T1, W, X, arrow, box, draw_header, \
    items, txt

HERE = Path(__file__).parent
LSNN_T = "#fcefe7"
LSNN_D = "#b84a00"
H = 183.0


def topology(ax, x, y, w, h, n_in, hidden, n_out, lab):
    """Input -> two recurrent layers (a loop over each) -> output neurons reading both layers."""
    cols = [n_in] + hidden + [n_out]
    cx = np.linspace(x + 1.5, x + w - 1.5, 4)
    yc = y + h / 2 + 2.0
    hs = [min(h - 7.0, 1.6 + 1.6 * np.log2(max(n, 32) / 32)) for n in cols]
    for i, (c, n, hh) in enumerate(zip(cx, cols, hs)):
        if i:
            if i < 3:                                  # feedforward; the output's inputs are the read arrows below
                ax.plot([cx[i - 1] + 0.8, c - 0.8], [yc, yc], color=FAINT, lw=0.5, zorder=4)
        fc, ec = (("white", FAINT) if i == 0 else (LSNN, LSNN) if i in (1, 2) else ("#5b6170", "#5b6170"))
        box(ax, c - 0.8, yc - hh / 2, 1.6, hh, fc, ec=ec, lw=0.5, r=0.5, z=5)
        if i in (1, 2):
            di.self_loop(ax, c, yc - hh / 2, width=1.3, height=2.8)      # recurrence: back into the same layer
        if i == 0:
            txt(ax, c, yc + hh / 2 + 0.4, f"{n:,}", size=4.6, color=MUTED, ha="center")
        elif i < 3:                                    # beside the column, clear of the layer-1 read line below it
            txt(ax, c + 1.0, yc + hh / 2 - 1.6, f"{n:,}", size=4.6, color=MUTED, ha="left")
        else:                                          # the output's label beside it, clear of the read arrows
            txt(ax, c + 1.2, yc - 0.9, f"{n}\n{lab}", size=4.6, color=MUTED, ha="left")
    # the readout reads BOTH layers: layer 2 straight in, layer 1 by an arrow passing under layer 2
    di.read_arrow(ax, (cx[2] + 0.8, yc), (cx[3] - 0.8, yc), color=LSNN, lw=0.7)
    yb = yc + max(hs[1], hs[2]) / 2 + 2.0              # under the layer-size labels
    ax.plot([cx[1], cx[1], cx[3]], [yc + hs[1] / 2, yb, yb], color=LSNN, lw=0.7, zorder=6, solid_joinstyle="round")
    di.read_arrow(ax, (cx[3], yb), (cx[3], yc + hs[3] / 2), color=LSNN, lw=0.7)


def mem_line(ax, x, y, w, shade=None, window=None, word=None):
    f = lambda t: x + w * (t - T0) / (T1 - T0)
    ax.plot([f(T0), f(T1)], [y, y], color=FAINT, lw=0.6, zorder=3)
    for t, s in ((-1.2, "−1.2 s"), (-0.8, "−0.8"), (-0.4, "−0.4"), (0.0, "0")):
        ax.plot([f(t)] * 2, [y, y + 0.7], color=FAINT, lw=0.5, zorder=3)
        txt(ax, f(t), y + 1.0, s, size=4.8, color=MUTED, ha="center")
    ax.plot([f(0)] * 2, [y - 3.4, y + 0.7], color=INK, lw=0.7, zorder=4)
    reach = 1.3 if window is None else -window[0]
    n = 40
    for k in range(n):
        tt = -reach * k / n
        ax.add_patch(Rectangle((f(tt - reach / n), y - 1.9), w * (reach / n) / (T1 - T0), 1.3, fc=LSNN, ec="none",
                               alpha=0.65 * (1 - k / n) + 0.08, zorder=3))
    if window is None:
        txt(ax, f(-reach), y - 2.3, "inside the network", size=4.7, color=LSNN_D, va="bottom")
    else:
        txt(ax, f(-reach) - 0.8, y - 2.4, "memory ends\nat the window", size=4.5, color=LSNN_D, ha="right",
            va="center")
        ax.plot([f(window[0])] * 2, [y - 3.0, y + 0.6], color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=4)
        ax.add_patch(Rectangle((f(0), y - 1.9), f(window[1]) - f(0), 1.3, fc="#e3e6ec", ec="none", zorder=2))
        txt(ax, f(window[1]), y - 2.3, "look-ahead", size=4.7, color=MUTED, ha="right", va="bottom")
    if shade:
        a, b, lab = shade
        ax.add_patch(Rectangle((f(a), y - 3.4), f(b) - f(a), 1.3, fc="#e3e6ec", ec="none", zorder=2))
        txt(ax, (f(a) + f(b)) / 2, y - 3.6, lab, size=4.7, color=MUTED, ha="center", va="bottom")
    if word:
        ax.plot([f(-word), f(-word), f(0)], [y - 3.0, y - 3.4, y - 3.4], color=MUTED, lw=0.6, zorder=4)
        txt(ax, f(-word / 2), y - 3.6, "a word", size=4.7, color=MUTED, ha="center", va="bottom")


def core_box(ax, x, y, w, h):
    box(ax, x, y, w, h, "white", ec=LSNN, lw=0.7, r=1.8, z=2)
    txt(ax, x + 2.5, y + 1.6, "One network for every task", size=7.2, weight="bold")
    panels = [("recurrent layers", "each neuron\nfeeds the others\nat the next step"),
              ("90% adaptive, 10% not", "a fixed random\n90% of each\nlayer adapts"),
              ("adaptive threshold", "rises at each\nspike, relaxes\nover 500 ms"),
              ("synaptic filters", "each input\nsmoothed over\n20 ms to 1.3 s")]
    pw = (w - 6) / 4
    for k, (head, sub) in enumerate(panels):
        px = x + 3 + k * pw
        txt(ax, px, y + 6.0, head, size=5.6, weight="bold", color=LSNN_D)
        gx, gy, gw, gh = px + 0.5, y + 9.4, (h - 11.0) if k < 2 else pw * 0.5, h - 11.0
        if k == 0:
            di.recurrent_ring(ax, gx, gy, gw, gh)
        elif k == 1:
            di.alif_grid(ax, gx, gy, gw, gh)
        elif k == 2:
            di.threshold_trace(ax, gx, gy + 0.6, gw, gh - 1.2)
            txt(ax, gx + gw * 0.62, gy - 0.2, "adaptive", size=4.2, color=LSNN_D)
            txt(ax, gx + gw * 0.72, gy + gh * 0.66, "fixed", size=4.2, color=MUTED)
        else:
            di.filter_kernels(ax, gx, gy + 0.6, gw, gh - 1.4)
            txt(ax, gx + 2.2, gy + gh - 3.2, "20 ms", size=4.2, color=LSNN_D, va="bottom")
            txt(ax, gx + gw, gy + gh * 0.42, "1.3 s", size=4.2, color=LSNN_D, ha="right", va="bottom")
        txt(ax, gx + gw + 1.6, gy + 1.0, sub, size=4.8, color=INK)


COLS = [
    dict(inp=[(0, "96 channels, spikes as 0/1,\na step every 4 ms")], topo=(96, [256, 256], 2, "velocity"),
         net=[(0, "2 × 256")], mem=dict(shade=(-0.20, -0.12, "M1 lead")),
         mtxt=(1, "filters and recurrence span the lead"),
         read=("velocity", 2), rtxt=(2, "both layers → two leaky stages\n→ a smooth velocity every 4 ms"),
         train="loso", ttxt=(3, "pre-trained on the other sessions, then\nfine-tuned per session (37.3 → 32.0 mm/s)")),
    dict(inp=[(3, "96 channels, counts z-scored per day:\nthe largest gain of any change")],
         topo=(96, [256, 256], 128, "bands"), net=[(0, "2 × 256")], mem=dict(window=(-0.32, 0.32)),
         mtxt=(1, "the reference 17-frame window, one\nstep per 40 ms frame, read from rest"),
         read=("spectrum", 1), rtxt=(2, "both layers → one leaky stage\n→ 128 graded bands per frame"),
         train="noise", ttxt=(0, "squared error on the windows, with noise,\noffset and dropout; chosen on validation")),
    dict(inp=[(2, "256 features (crossings and spike-band\npower), 20 ms steps")],
         topo=(256, [1024, 1024], 41, "classes"), net=[(1, "2 × 1,024:\nwidth matters\nhere")], mem=dict(word=0.5),
         mtxt=(1, "recurrence and filters to 1.3 s\ncarry a word"),
         read=("phonemes", 1), rtxt=(1, "both layers → one leaky stage → 41\nclasses every 80 ms → CTC"),
         train="days", ttxt=(3, "CTC, all 19 days together; noise, offset\nand dropout as audio")),
]


def main():
    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    end_demands = draw_header(ax)

    sy = end_demands + 4.0
    cy, ch = sy + 2.5, 26.0
    rows, y = {}, cy + ch + 5.5
    for k, h in (("in", 7.0), ("net", 22.0), ("mem", 11.6), ("read", 15.5), ("train", 15.0)):
        rows[k] = (y, h)
        y += h + 1.6
    sh = y + 0.4 - sy
    box(ax, 0.5, sy, W - 1.0, sh, LSNN_T, ec=LSNN_T, r=2.5, z=0)
    txt(ax, 1.2, sy + 2.2, "LSNN", size=8.2, weight="bold", color=LSNN)
    core_box(ax, GUT, cy, W - GUT - 1.5, ch)
    labels = {"in": ("Input", ""), "net": ("Network", ""), "mem": ("Memory", ""), "read": ("Readout", ""),
              "train": ("Training", "")}
    for k, (a, b) in labels.items():
        y0 = rows[k][0]
        txt(ax, 1.2, y0 + 0.2, a, size=6.0, weight="bold")
        if b:
            txt(ax, 1.2, y0 + 3.1, b, size=5.1, color=MUTED)
    top, bot = rows["in"][0] - 2.4, rows["train"][0] + rows["train"][1] + 1.0
    for i, c in enumerate(COLS):
        x = X(i)
        arrow(ax, x + CW / 2, cy + ch, x + CW / 2, top - 0.3, color=LSNN)
        box(ax, x, top, CW, bot - top, CARD, ec=CARD_E, lw=0.5, r=1.6, z=1)
        for k in ("net", "mem", "read", "train"):
            ax.plot([x + 2.0, x + CW - 2.0], [rows[k][0] - 1.0] * 2, color="#eceef2", lw=0.5, zorder=2)
        items(ax, x + 2.0, rows["in"][0], c["inp"], LSNN, size=5.6)
        y0, h0 = rows["net"]
        topology(ax, x + 2.0, y0, 28, h0, *c["topo"])
        items(ax, x + 32.0, y0 + 8.0, c["net"], LSNN, size=5.6)
        y0 = rows["mem"][0]
        mem_line(ax, x + 2.0, y0 + 4.6, CW - 4.0, **c["mem"])
        items(ax, x + 2.0, y0 + 7.3, [c["mtxt"]], LSNN, size=5.3)
        y0 = rows["read"][0]
        kind, stages = c["read"]
        if kind == "velocity":
            di.smooth_cascade(ax, x + 2.0, y0, CW - 4.0, 8.0, stages=2, out="velocity")
        elif kind == "spectrum":
            xe = di.smooth_cascade(ax, x + 2.0, y0, 22.0, 8.0, stages=1, trace=False)
            di.spectrogram(ax, xe + 0.8, y0 + 0.6, CW - (xe + 0.8 - x) - 2.0, 7.0)
            txt(ax, xe + 0.8, y0 + 7.9, "128 bands, per frame", size=4.4, color=MUTED)
        else:
            xe = di.smooth_cascade(ax, x + 2.0, y0, 22.0, 8.0, stages=1, trace=False)
            di.phoneme_scores(ax, xe + 0.8, y0 + 0.3, CW - (xe + 0.8 - x) - 2.0, 9.0)
        items(ax, x + 2.0, y0 + 9.6, [c["rtxt"]], LSNN, size=5.3)
        y0 = rows["train"][0]
        if c["train"] == "loso":
            di.loso(ax, x + 2.0, y0, CW - 4.0, 7.0)
        elif c["train"] == "noise":
            di.noise_dropout(ax, x + 2.0, y0, CW - 4.0, 7.0)
        else:
            di.day_gains(ax, x + 2.0, y0, CW - 4.0, 7.0)
        items(ax, x + 2.0, y0 + 8.6, [c["ttxt"]], LSNN, size=5.3)
    end = sy + sh
    Hn = end + 1.5                                       # the figure ends with the band (no closing note)
    fig.set_size_inches(W * style.MM, Hn * style.MM)
    ax.set_ylim(Hn, 0)
    print("layout bottom (mm):", round(Hn, 1))
    style.save(fig, HERE / "figures" / "fig_lsnn_design")


if __name__ == "__main__":
    main()
