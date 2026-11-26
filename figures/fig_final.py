"""
The final results figures, from data/final_fidelity.csv and data/final_efficiency.csv (collect_final.py):

    a  fidelity: every session (motor, speech) or held-out word (audio) of every decoder, a dot each, the box its
       middle half and the line its median; the Speck SNN as scored on the chip, its software median as a hollow
       diamond beside it
    b  compute time per output against the time between outputs (the real-time budget): every CPU decoder on one
       Speck laptop thread; the Speck SNN on the chip (the host loop's wall time, its waits included)
    c  measured energy per output: CPU decoders by RAPL, idle subtracted; the Speck SNN as the chip's own (solid,
       devkit power monitor) plus the host's share (hatched: input buffer, delays and readout on the laptop CPU)

    python fig_final.py            paper figure (figures/fig_final.*) and one slide figure per experiment
                                   (figures/slides/final_<experiment>.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slide_style
import style
from fig_efficiency import fmt_uj

HERE = Path(__file__).parent
SETS = {  # experiment: decoders, metric, axis label, higher is better, groups (marker), unit, efficiency rows
    "motor": (["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], "cc", "velocity correlation", True,
              {"indy": "o", "loco": "s"}, "session", "motor"),
    "jenkins": (["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], "cc", "velocity correlation", True,
                {"jenkins": "^"}, "session", "motor_jenkins"),
    "audio": (["kf", "wf", "dense_nn", "gru", "lstm", "lsnn", "speck"], "rho_word", "spectral match within the word",
              True, {"word": "o"}, "word", "audio"),
    "speech": (["paper_rnn", "lsnn", "speck"], "per", "phoneme error rate", False, {"t12": "o"},
               "session", "speech"),
}
GROUP_NAME = {"indy": "monkey I", "loco": "monkey L", "jenkins": "monkey J"}


def load():
    return pd.read_csv(HERE / "data" / "final_fidelity.csv"), pd.read_csv(HERE / "data" / "final_efficiency.csv")


def tick(d, big):
    lab = style.LABEL[d]
    return lab.replace(" ", "\n", 1) if (big and " " in lab) else style.SHORT[d].replace("\n", " ") if not big else lab


# decoders shown apart from the rest: references we could only re-implement from their papers (no code; not the code
# base the other decoders share), drawn after a divider with dashed outlines and hollow points
SET_APART = {"aegru"}


def fidelity(ax, exp, fid, big):
    decs, metric, ylabel, higher, groups, unit, _ = SETS[exp]
    f = fid[(fid.experiment == exp) & (fid.metric == metric)]
    rng = np.random.default_rng(0)
    s_dot, lw, fs = (22, 2.2, 12) if big else (5, 1.0, 5.5)
    for i, d in enumerate(decs):
        v = f[f.decoder == d].value.to_numpy()
        hl = d in style.HIGHLIGHT
        apart = d in SET_APART
        ls = (0, (3, 2)) if apart else "-"
        lmx = d.endswith("_lm")                         # a decoder with the language model: filled box, beside it
        ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=matplotlib.colors.to_rgba(style.COLOR[d], 0.22) if lmx else "none",
                                 edgecolor=style.COLOR[d], linewidth=lw if hl else lw * 0.55,
                                 linestyle=ls),
                   medianprops=dict(color=style.COLOR[d], linewidth=lw * 1.2),
                   whiskerprops=dict(color=style.COLOR[d], linewidth=lw * 0.55, linestyle=ls),
                   capprops=dict(color=style.COLOR[d], linewidth=lw * 0.55))
        for g, mk in groups.items():
            vg = f[(f.decoder == d) & (f.group == g)].value.to_numpy()
            if apart:
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(vg)), vg, s=s_dot, marker=mk, facecolor="none",
                           edgecolor=style.COLOR[d], alpha=0.7, linewidths=0.5, zorder=3)
            else:
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(vg)), vg, s=s_dot, marker=mk, color=style.COLOR[d], alpha=0.55,
                           linewidths=0, zorder=3)
        ax.text(i, 1.01, f"{np.median(v):.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=fs, fontweight="bold" if hl else "normal")
        if d == "speck":
            sw = f[f.decoder == "speck_float"].value
            if len(sw):
                ax.scatter(i + 0.42, np.median(sw), marker="D", s=s_dot * 2.6, facecolor="white",
                           edgecolor=style.COLOR[d], linewidths=lw * 0.7, zorder=4)
                if i + 1 < len(decs) and decs[i + 1].endswith("_lm"):     # its 5-gram box sits there: the caption
                    continue                                                # names the diamond
                ax.annotate("software", (i + 0.42, np.median(sw)), xytext=(4 if big else 2, 0),
                            textcoords="offset points", va="center", fontsize=fs * 0.85, color=style.MUTED)
    ax.set_xticks(range(len(decs)))
    ax.set_xticklabels([tick(d, big) if big else style.SHORT[d] for d in decs], fontsize=None if big else 5)
    for t, d in zip(ax.get_xticklabels(), decs):
        if d in style.HIGHLIGHT:
            t.set_fontweight("bold")
    ax.set_xlim(-0.6, len(decs) - 0.1)
    first_apart = next((i for i, d in enumerate(decs) if d in SET_APART), None)
    if first_apart is not None:                      # the divider, and what sets those decoders apart
        ax.axvline(first_apart - 0.5, color=style.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.text(first_apart + (len(decs) - first_apart - 1) / 2, 0.035, "re-implemented\nfrom its paper",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=fs * 0.8, color=style.MUTED,
                style="italic")
    ax.set_ylabel(ylabel + ("" if higher else " (lower is better)"))
    if not higher:
        ax.invert_yaxis()
    if len(groups) > 1:
        h = [plt.Line2D([], [], marker=mk, ls="", color=style.MUTED, ms=7 if big else 3) for mk in groups.values()]
        n = {g: f[(f.group == g) & (f.decoder == decs[0])].shape[0] for g in groups}
        ax.legend(h, [f"{GROUP_NAME[g]} ({n[g]})" for g in groups], loc="upper left" if big else "lower left", frameon=False,
                  fontsize=11 if big else 5.5, handletextpad=0.2)


def latency(ax, exp, eff, big):
    decs, rows = SETS[exp][0], SETS[exp][6]
    e = eff[eff.experiment == rows].set_index("decoder").reindex(decs)
    iv = e.interval_ms.dropna().iloc[0]
    fs = 11 if big else 5.5
    for i, d in enumerate(decs):
        v = e.loc[d, "latency_ms"]
        share = v / iv
        ax.barh(i, share, 0.66, color=style.COLOR[d], zorder=3)
        lab = f"{v:.2g} ms" if v >= 0.01 else f"{v * 1e3:.2g} µs"
        inside = share > 0.75
        ax.text(share - 0.02 if inside else share + 0.02, i, lab + (" on the chip" if d == "speck" else ""),
                va="center", ha="right" if inside else "left", fontsize=fs, color="white" if inside else style.INK)
    ax.axvline(1, color=style.MUTED, ls="--", lw=1.4 if big else 0.7)
    ax.text(1.02, len(decs) - 0.6, f"budget\n{iv:g} ms", ha="left", va="bottom", fontsize=fs, color=style.MUTED)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_yticks(range(len(decs)))
    ax.set_yticklabels([style.LABEL[d] if big else style.SHORT[d].replace("\n", " ") for d in decs])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("time per output, share of the time between outputs" if big else "time per output / budget")


def energy(ax, exp, eff, big):
    decs, rows = SETS[exp][0], SETS[exp][6]
    e = eff[eff.experiment == rows].set_index("decoder").reindex(decs)
    fs = 11 if big else 5.5
    for i, d in enumerate(decs):
        if d == "speck":
            c, h = e.loc[d, "chip_uj"], e.loc[d, "host_uj"]
            ax.barh(i, c, 0.66, color=style.COLOR[d], zorder=3)
            ax.barh(i, h, 0.66, left=c, facecolor="white", edgecolor=style.COLOR[d], hatch="///",
                    linewidth=1.0 if big else 0.5, zorder=3)
            ax.text((c + h) * 1.5, i, f"chip {fmt_uj(c)} + host {fmt_uj(h)}", va="center", fontsize=fs)
        else:
            v = e.loc[d, "energy_uj"]
            ax.barh(i, v, 0.66, color=style.COLOR[d], zorder=3)
            ax.text(v * 1.5, i, fmt_uj(v), va="center", fontsize=fs)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e7)
    ax.set_yticks(range(len(decs)))
    ax.set_yticklabels([style.LABEL[d] if big else style.SHORT[d].replace("\n", " ") for d in decs])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("measured energy per output (µJ, log scale)" if big else "energy per output (µJ)")


def slide(exp, fid, eff):
    slide_style.apply()
    fig = plt.figure(figsize=(slide_style.W, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.1], wspace=0.5, left=0.06, right=0.97, top=0.86, bottom=0.14)
    titles = ["Fidelity, every " + SETS[exp][5] + "; median above", "Fast enough for real time?",
              "Energy per output, measured"]
    for k, (fn, t) in enumerate(zip((fidelity, latency, energy), titles)):
        ax = fig.add_subplot(gs[k])
        fn(ax, exp, fid if fn is fidelity else eff, True)
        ax.set_title(t, loc="left", fontsize=14, fontweight="normal", color=style.MUTED, pad=30 if k == 0 else 10)
    slide_style.save(fig, "final_" + exp)


def paper(fid, eff):
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 2.3 * len(SETS)))
    gs = fig.add_gridspec(len(SETS), 3, hspace=0.75, wspace=0.62, width_ratios=[1.25, 1, 1.1])
    for r, exp in enumerate(SETS):
        for k, fn in enumerate((fidelity, latency, energy)):
            ax = fig.add_subplot(gs[r, k])
            fn(ax, exp, fid if fn is fidelity else eff, False)
            if k == 0:
                ax.set_title(style.EXPERIMENTS[exp], pad=9)
            style.panel_label(ax, "abcdefghijkl"[3 * r + k], x=-0.3 if k == 0 else -0.32)
    style.save(fig, HERE / "figures" / "fig_final")
    print("wrote figures/fig_final.{pdf,svg,png}")


if __name__ == "__main__":
    fid, eff = load()
    for exp in SETS:
        slide(exp, fid, eff)
    paper(fid, eff)
