"""
Figure: energy and latency across the three experiments (data/efficiency.csv, data/fidelity.csv).

    a-c  energy per output, estimated from operation counts (45 nm, on-chip memory; Kalman filters at their steady
         state); the Speck SNNs also measured on the chip by its power monitor (diamonds; static power included)
    d-f  compute time per output, every decoder on the Speck laptop's CPU, one thread (snn-decoders
         chip/latency_all.py); the Speck SNNs on the chip (the host loop's wall time, its waits included),
         against the time between outputs (dashed: the real-time budget)
    g-i  fidelity against estimated energy per output (motor: monkey I sessions; audio: single days, the ones the
         Speck SNN was trained on). The Speck SNNs (diamonds): fidelity and energy both on the chip (audio: layer 1
         threshold x4, chip/audio_chip.py)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).parent
COLS = [("motor", ["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], "cc", "velocity correlation", True),
        ("audio", ["kf", "wf", "dense_nn", "lstm", "gru", "lsnn", "speck"], "rho", "mel-spectrogram correlation", True),
        ("speech", ["lstm", "paper_rnn", "lsnn"], "per", "phoneme error rate", False)]


def fmt_uj(v):
    return f"{v * 1e3:.0f} nJ" if v < 1 else f"{v:.0f} µJ" if v < 1e3 else f"{v / 1e3:.1f} mJ" if v < 1e6 else f"{v / 1e6:.2f} J"


def main():
    style.apply()
    eff = pd.read_csv(HERE / "data" / "efficiency.csv")
    fid = pd.read_csv(HERE / "data" / "fidelity.csv")
    fig = plt.figure(figsize=(style.DOUBLE, 6.6))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.42, width_ratios=[6, 7, 4.2])
    for col, (exp, decs, metric, mlabel, higher) in enumerate(COLS):
        e = eff[eff.experiment == exp].set_index("decoder").reindex(decs)
        x = np.arange(len(decs))
        # energy per output
        ax = fig.add_subplot(gs[0, col])
        for i, d in enumerate(decs):
            v = e.loc[d, "energy_uj"]
            if np.isfinite(v):
                ax.bar(i, v, 0.62, color=style.COLOR[d], edgecolor="none")
                ax.text(i, v * 1.5, fmt_uj(v), ha="center", fontsize=5.5, color=style.INK)
            # measured: only the Speck chip's own power monitor (the CPU's RAPL counted the whole package under Keras)
            mv = e.loc[d, "measured_uj"] if e.loc[d].get("measured_by") == "Speck chip" else np.nan
            if np.isfinite(mv):
                ax.scatter(i, mv, marker="D", s=14, facecolor="white", edgecolor=style.COLOR[d], linewidths=1, zorder=4)
                if d == "speck":
                    ax.text(i, mv * 1.6, f"{fmt_uj(mv)}\nchip", ha="center", fontsize=5.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 5e6)
        ax.set_xticks(x)
        ax.set_xticklabels([style.SHORT[d] for d in decs], fontsize=5.5 if len(decs) >= 6 else None)
        ax.set_ylabel("energy per output (µJ)")
        ax.set_title(style.EXPERIMENTS[exp], pad=10)
        iv = e["interval_ms"].dropna().iloc[0]
        ax.text(0, 1.0, f"one output every {iv:g} ms" + ("; diamond: measured on the chip" if exp != "speech" else ""),
                transform=ax.transAxes, fontsize=6, color=style.MUTED, va="bottom")
        style.panel_label(ax, "abc"[col])
        # compute time per output vs the real-time budget
        ax = fig.add_subplot(gs[1, col])
        for i, d in enumerate(decs):
            v = e.loc[d, "latency_ms"]
            if not np.isfinite(v):
                ax.text(i, 6e-4, "not\nmeasured", ha="center", va="bottom", fontsize=5.5, color=style.MUTED)
                continue
            ax.bar(i, v, 0.62, color=style.COLOR[d], edgecolor="none")
            ax.text(i, v * 1.5, f"{v:.3g} ms", ha="center", fontsize=5.5)
        ax.axhline(iv, color=style.MUTED, ls="--", lw=0.8)
        ax.text(-0.45, iv * 1.25, f"{iv:g} ms between outputs", ha="left", fontsize=5.5, color=style.MUTED)
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 3e2)
        ax.set_xticks(x)
        ax.set_xticklabels([style.SHORT[d] for d in decs], fontsize=5.5 if len(decs) >= 6 else None)
        ax.set_ylabel("compute time per output (ms)")
        style.panel_label(ax, "def"[col])
        # fidelity vs energy
        ax = fig.add_subplot(gs[2, col])
        f = fid[(fid.experiment == exp) & (fid.metric == metric)]
        if exp == "motor":
            f = f[f.group == "indy"]
        if exp == "audio":
            f = f[f.group == "day"]                    # the sessions every audio decoder has
        for d in decs:
            hollow = e.loc[d].get("measured_by") == "Speck chip"     # the Speck SNNs: scored and measured on the chip
            en = e.loc[d, "measured_uj"] if hollow else e.loc[d, "energy_uj"]
            y = f[f.decoder == d].value.mean()
            ax.scatter(en, y, s=34 if d in style.HIGHLIGHT else 22, zorder=3, linewidths=1,
                       facecolor="white" if hollow else style.COLOR[d], edgecolor=style.COLOR[d],
                       marker="D" if hollow else "o")
            ax.annotate(style.SHORT[d], (en, y), xytext=(4, 3), textcoords="offset points", fontsize=6,
                        fontweight="bold" if d in style.HIGHLIGHT else "normal")
        ax.set_xscale("log")
        ax.set_xlabel("energy per output (µJ)")
        ax.set_ylabel(mlabel + ("\n(axis flipped: better is up)" if not higher else ""))
        if not higher:
            ax.invert_yaxis()
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 2, hi * 6)                   # room for the labels at both ends
        y0, y1 = ax.get_ylim()
        pad = (y1 - y0) * 0.08
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.grid(True, axis="both")
        style.panel_label(ax, "ghi"[col])
    style.save(fig, HERE / "figures" / "fig_efficiency")
    print("wrote figures/fig_efficiency.{pdf,svg,png}")


if __name__ == "__main__":
    main()
