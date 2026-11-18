"""
Why the Speck SNN falls short on Jenkins, from the final laptop runs (final_v2/<subject>/combined_metrics.json and
speck_diagnosis.json; spike_bmi inference/diagnose_speck.py compares the chip's output spikes with the 8-bit network's
on the same test steps):

    a  every decoder on Jenkins with both arrays (192 channels) and with the M1 array alone (96): velocity correlation
       per session; every decoder loses with M1 alone, in every session
    b  Jenkins M1, where the Speck SNN's error comes from: the network in PyTorch, with 8-bit weights, the chip's
       spikes shifted back by their best-fitting delay, and the chip as run; the CPU decoders as reference lines
    c  the chip's cost (chip RMSE over PyTorch RMSE) against the chip's output spikes per step, every session of
       Indy, Loco and Jenkins: the sparse sessions lose 2-3%, Jenkins 34-40%

    python fig_jenkins_chip.py      -> figures/fig_jenkins_chip.{pdf,svg,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).parent
RUNS = Path.home() / "Downloads/final_v2"
DECS = ["kf", "wf", "lstm", "qrnn", "snn", "speck"]
NAME = {"snn": "lsnn"}


def cc(v):
    return (v["cc_x"] + v["cc_y"]) / 2


def load(subj):
    return (json.load(open(RUNS / subj / "combined_metrics.json")),
            json.load(open(RUNS / subj / "speck_diagnosis.json"))["sessions"])


def panel_channels(ax):
    a, _ = load("jenkins")
    b, _ = load("jenkins_m1")
    S = sorted(set(a) & set(b))
    for i, d in enumerate(DECS):
        c = style.COLOR[NAME.get(d, d)]
        ya, yb = [cc(a[s][d]) for s in S], [cc(b[s][d]) for s in S]
        for u, v in zip(ya, yb):
            ax.plot([i - 0.17, i + 0.17], [u, v], color=c, lw=0.8, alpha=0.6)
        ax.scatter([i - 0.17] * len(ya), ya, s=14, facecolor="white", edgecolor=c, lw=1, zorder=3)
        ax.scatter([i + 0.17] * len(yb), yb, s=14, color=c, zorder=3)
    ax.set_xticks(range(len(DECS)))
    ax.set_xticklabels([style.SHORT[NAME.get(d, d)] for d in DECS], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("velocity correlation")
    ax.set_title("Jenkins: 192 against 96 channels", loc="left")
    ax.scatter([], [], s=14, facecolor="white", edgecolor=style.MUTED, label="192 channels")
    ax.scatter([], [], s=14, color=style.MUTED, label="96, M1")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2)


def panel_sources(ax):
    m, diag = load("jenkins_m1")
    S = sorted(diag)
    pt = np.mean([diag[s]["pytorch"]["rmse"] for s in S])
    q = np.mean([diag[s]["quantized"]["rmse"] for s in S])
    lag = np.mean([diag[s]["chip_vs_quantized"]["rmse_lag_corrected"] for s in S])
    chip = np.mean([diag[s]["speck"]["rmse"] for s in S])
    bars = [("PyTorch", pt), ("8-bit", q), ("chip,\nshifted", lag), ("chip", chip)]
    col = style.COLOR["speck"]
    for i, (lab, v) in enumerate(bars):
        ax.bar(i, v, color=col, alpha=0.35 + 0.65 * (i == len(bars) - 1), width=0.65)
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    for d, ls in (("lstm", "--"), ("qrnn", ":"), ("snn", "-")):
        v = np.mean([m[s][d]["rmse"] for s in S])
        ax.axhline(v, color=style.COLOR[NAME.get(d, d)], ls=ls, lw=0.9, label=f"{style.SHORT[NAME.get(d, d)]} {v:.0f}")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.5)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, 95)
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=6.5, handlelength=1.8)
    ax.set_ylabel("RMSE, mm/s (mean of 4 sessions)")
    ax.set_title("Jenkins M1: Speck error by stage", loc="left")


def panel_density(ax):
    marks = {"indy": ("o", "Indy"), "loco": ("s", "Loco"), "jenkins_m1": ("^", "Jenkins M1"),
             "jenkins": ("v", "Jenkins, 192")}
    for subj, (mk, lab) in marks.items():
        _, diag = load(subj)
        x = [v["speck"]["output_spikes_per_step"] for v in diag.values() if "speck" in v]
        y = [100 * (v["speck"]["rmse"] / v["pytorch"]["rmse"] - 1) for v in diag.values() if "speck" in v]
        hollow = subj == "jenkins"
        ax.scatter(x, y, marker=mk, s=16, label=lab, color=style.COLOR["speck"] if not hollow else "white",
                   edgecolor=style.COLOR["speck"] if hollow else "none", lw=1, alpha=0.85)
    ax.axhline(0, color=style.MUTED, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("chip output spikes per 4 ms step")
    ax.set_ylabel("chip RMSE over PyTorch RMSE (%)")
    ax.set_title("Chip cost against activity", loc="left")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.1)


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 66 * style.MM),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=0.5))
    for ax, fn, k in zip(axes, (panel_channels, panel_sources, panel_density), "abc"):
        fn(ax)
        style.panel_label(ax, k, x=-0.2)
    style.save(fig, HERE / "figures" / "fig_jenkins_chip")
    print("wrote figures/fig_jenkins_chip.{pdf,svg,png}")


if __name__ == "__main__":
    main()
