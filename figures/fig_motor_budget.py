"""
Motor figure: the chip's weight memory caps how many inputs the network can take, and within that cap, channels are
worth more than delay taps.

    a  Speck2f's nine cores by weight memory (two of 65,536, two of 32,768, five of 16,384), and how much of each
       core size the motor Speck network's four layers use: 384 x 128, 128 x 256, 256 x 256 and 256 x 36 weights
    b  the widest first layer that fits a 65,536-weight core against the number of chip inputs (channels x taps);
       the deployed networks of each experiment, and Jenkins with both arrays and four taps (768 inputs), which
       would leave a first layer of 85 neurons
    c  the motor channels-against-taps sweep at a fixed 480 inputs (speck_memory, 8 Indy sessions x 3 seeds, test
       RMSE, readout of the last hidden layer): fewer channels with longer history decode worse at every step;
       96 channels with no taps (96 inputs) for reference

    python fig_motor_budget.py      -> figures/fig_motor_budget.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).parent
CORES = [65536, 65536, 32768, 32768, 16384, 16384, 16384, 16384, 16384]
MOTOR_LAYERS = [("layer 1", 384 * 128, 65536), ("layer 2", 128 * 256, 32768), ("layer 3", 256 * 256, 65536),
                ("output", 256 * 36, 16384)]


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 60 * style.MM),
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1.1], wspace=0.6))

    # a: layers in cores
    ax = axes[0]
    for i, (name, need, cap) in enumerate(MOTOR_LAYERS):
        ax.barh(i, cap, color="#eeeeee", edgecolor=style.MUTED, lw=0.6, height=0.6)
        ax.barh(i, need, color=style.COLOR["speck"], height=0.6)
        ax.text(cap * 1.03, i, f"{need:,} of {cap:,}\n({100 * need / cap:.0f}%)", va="center", fontsize=6)
    ax.set_yticks(range(len(MOTOR_LAYERS)))
    ax.set_yticklabels([m[0] for m in MOTOR_LAYERS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100000)
    ax.set_xticks([0, 16384, 32768, 65536])
    ax.set_xticklabels(["0", "16k", "32k", "64k"])
    ax.set_xlabel("weights (core capacity in grey)")
    ax.set_title("Motor network in the chip's cores", loc="left")
    ax.text(0.0, -0.27, "cores: 2 × 64k, 2 × 32k, 5 × 16k weights", transform=ax.transAxes, fontsize=6,
            color=style.MUTED)

    # b: first-layer width against inputs
    ax = axes[1]
    n = np.arange(64, 1025)
    ax.plot(n, 65536 / n, color=style.INK, lw=1.2)
    ax.fill_between(n, 65536 / n, 600, color="#f3d6d6", lw=0)
    ax.text(640, 470, "does not fit", fontsize=6.5, color="#c0392b", ha="center")
    pts = [(384, "motor, 96 × 4"), (432, "audio, 48 × 9"), (512, "speech, 128 × 4")]
    for x, lab in pts:
        ax.plot(x, 128, "o", color=style.COLOR["speck"], ms=4, zorder=4)
    ax.text(150, 95, "motor, audio, speech:\n384, 432, 512 inputs,\n128 wide", fontsize=6, ha="left", va="top")
    ax.plot(768, 128, "x", color="#c0392b", ms=5, mew=1.5, zorder=4)
    ax.plot(768, 65536 / 768, "o", color="#c0392b", ms=3, zorder=4)
    ax.text(860, 260, "Jenkins 192 × 4:\n768 inputs,\n≤ 85 wide", fontsize=6, color="#c0392b", ha="center",
            va="bottom")
    ax.set_xlim(64, 1024)
    ax.set_ylim(0, 600)
    ax.set_xlabel("chip inputs (channels × taps)")
    ax.set_ylabel("widest first layer")
    ax.set_title("One core bounds layer 1", loc="left")

    # c: channels against taps
    ax = axes[2]
    d = pd.read_csv(HERE / "data" / "results.csv")
    m = d[(d.experiment == "motor") & (d.metric == "rmse_test") & (d.readout == "hidden")]
    g = m.groupby(["channels", "n_delays", "inputs"]).value
    rows = [(c, k, i, v.mean(), v.std() / np.sqrt(len(v))) for (c, k, i), v in g]
    budget = sorted([r for r in rows if r[2] == 480], key=lambda r: r[0])
    xs = [r[0] for r in budget]
    ax.errorbar(xs, [r[3] for r in budget], yerr=[r[4] for r in budget], fmt="o-", color=style.COLOR["speck"],
                ms=3.5, lw=1.2, capsize=2, label="480 inputs")
    for c, k, _, v, _ in budget:
        ax.text(c * 1.07, v + 0.6, f"{c} × {k}", fontsize=6, ha="left", va="bottom")
    ref = [r for r in rows if r[2] == 96]
    if ref:
        ax.errorbar([88], [ref[0][3]], yerr=[ref[0][4]], fmt="s", color=style.MUTED, ms=3.5, capsize=2,
                    label="96 × 1, no taps")
    ax.set_xscale("log", base=2)
    ax.set_xlim(6.5, 140)
    ax.set_xticks([8, 16, 24, 32, 48, 96])
    ax.set_xticklabels(["8", "16", "24", "32", "48", "96"])
    ax.minorticks_off()
    ax.set_xlabel("channels (taps fill the rest of 480)")
    ax.set_ylabel("velocity RMSE per axis (mm/s)")
    ax.set_title("Channels beat taps", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    for ax, k in zip(axes, "abc"):
        style.panel_label(ax, k, x=-0.28)
    style.save(fig, HERE / "figures" / "fig_motor_budget")
    print("wrote figures/fig_motor_budget; sweep at 480 inputs: " +
          ", ".join(f"{c}x{k} {v:.1f}" for c, k, _, v, _ in budget) + (f"; 96x1 {ref[0][3]:.1f}" if ref else ""))


if __name__ == "__main__":
    main()
