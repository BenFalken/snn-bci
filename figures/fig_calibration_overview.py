"""
Overview of calibration: one trained network, tuned for the Speck2f chip from both ends, before deployment.

    centre       the network before deployment, run in the chip simulator: input events -> the chip network (layer 1's
                 threshold has a dial) -> spikes -> the host readout -> output
    input end    reference = the PyTorch model's own output (no labels): the simulated chip's output is matched to it,
                 and the best-agreeing layer-1 threshold factor is kept
    output end   reference = the true targets (phonemes, spectrum, velocity): the readout's output is compared with
                 them and only the readout's weights are retrained
    bottom       together: the deployable configuration, the real chip with its factor, read by the refitted readout;
                 both set in the simulator on training data, the network's trained weights unchanged

    python fig_calibration_overview.py      -> figures/fig_calibration_overview.{pdf,svg,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import style
from calibration_panels import scores_icon, weights_icon
from flowdraw import (CHIP, CHIP_T, FAINT, HOST, HOST_T, INK, MUTED, REF, REF_T, arrow, box, columns, lock, raster,
                      region, trace, txt)

HERE = Path(__file__).parent


def dial(ax, x, y, r=2.6, color=CHIP):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.9, zorder=8))
    for a in np.linspace(np.pi * 1.15, -np.pi * 0.15, 6):
        ax.plot([x + r * np.cos(a), x + 1.25 * r * np.cos(a)], [y - r * np.sin(a), y - 1.25 * r * np.sin(a)],
                color=color, lw=0.5, zorder=8)
    ax.plot([x, x + 0.8 * r * np.cos(np.pi * 0.35)], [y, y - 0.8 * r * np.sin(np.pi * 0.35)], color=color, lw=1.1,
            zorder=9)


def main():
    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    W, H = 183.0, 118.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = region(fig, W, H, 0, 0, W, H)

    # ---------------------------------------------------------------- centre: the network before deployment
    cy = 52
    txt(ax, 74, cy - 12.5, "before deployment: the trained network in the chip simulator", size=6.0,
        weight="bold", ha="center")
    box(ax, 3, cy - 9, 177, 26, fc="#fafafb", ec="#d8dbe2", lw=0.6, r=2.2, z=0)
    box(ax, 6, cy - 5, 16, 18, ec=FAINT)
    raster(ax, 8, cy - 3, 12, 11, rows=6, seed=2)
    txt(ax, 14, cy + 9.3, "input events", size=4.6, color=MUTED, ha="center")
    arrow(ax, (22, cy + 4), (27, cy + 4))
    box(ax, 27, cy - 6, 60, 20, fc=CHIP_T, ec=CHIP, ls=(0, (3, 2)))
    lock(ax, 29.5, cy - 4.4, s=2.6)
    txt(ax, 34, cy - 4.3, "chip network (simulated)", size=5.6, weight="bold")
    txt(ax, 34, cy - 1.0, "weights as trained", size=4.7, color=MUTED)
    cx, yc = columns(ax, 38, cy + 2, 46, 11, [128, 256, 64], fc=CHIP, ec=CHIP, labels=False)
    dial(ax, cx[0] - 6.0, cy + 6.6, r=2.2)
    ax.plot([cx[0] - 3.8, cx[0] - 0.8], [cy + 6.6, cy + 7.2], color=CHIP, lw=0.6, zorder=7)
    txt(ax, cx[0] - 8.0, cy + 9.9, "layer 1 threshold", size=4.3, color=CHIP, ha="left", weight="bold")
    arrow(ax, (87, cy + 4), (92, cy + 4), color=CHIP)
    raster(ax, 92.5, cy - 1.5, 8, 11, rows=6, seed=7, color=CHIP, density=0.3)
    txt(ax, 96.5, cy + 10.4, "spikes", size=4.6, color=CHIP, ha="center")
    arrow(ax, (101, cy + 4), (106, cy + 4), color=CHIP)
    box(ax, 106, cy - 6, 44, 20, fc=HOST_T, ec=HOST, lw=1.0)
    txt(ax, 147.5, cy - 4.3, "host readout", size=5.6, weight="bold", color=HOST, ha="right")
    weights_icon(ax, 110, cy + 0.5, 9, 10)
    txt(ax, 114.5, cy + 11.2, "weights", size=4.5, color=HOST, ha="center")
    arrow(ax, (120, cy + 5.5), (124, cy + 5.5), color=HOST)
    scores_icon(ax, 125, cy + 0.5, 13, 10)
    txt(ax, 131.5, cy + 11.2, "output", size=4.5, color=HOST, ha="center")
    arrow(ax, (150, cy + 4), (155, cy + 4))
    box(ax, 155, cy - 4, 22, 16, ec=INK)
    txt(ax, 166, cy - 2.6, "its output", size=5.4, weight="bold", ha="center")
    trace(ax, 157.5, cy + 2.4, 17, 7, seed=1, color=CHIP, noise=0.5)

    # ---------------------------------------------------------------- input end: reference = the PyTorch model
    box(ax, 3, 3, 84, 31, fc="white", ec=CHIP, lw=0.9, r=2.2)
    ax.add_patch(Circle((8, 8.2), 2.4, fc=CHIP, ec="none", zorder=8))
    ax.text(8, 8.3, "1", fontsize=7, color="white", fontweight="bold", ha="center", va="center", zorder=9)
    txt(ax, 12, 5.6, "INPUT END: layer 1's threshold", size=6.8, weight="bold", color=CHIP)
    txt(ax, 12, 9.6, "reference: the PyTorch model's own output · no labels", size=5.0, color=INK)
    box(ax, 6, 14, 26, 17, fc=REF_T, ec=REF)
    txt(ax, 19, 15.2, "PyTorch model\n(as trained)", size=4.9, ha="center", weight="bold")
    columns(ax, 9, 21.0, 20, 9, [128, 256, 64], fc="#c9ccd4", ec=REF, labels=False)
    arrow(ax, (32, 22.5), (36, 22.5), color=REF)
    box(ax, 36, 14, 23, 17, ec=INK)
    trace(ax, 38, 16.5, 19, 9.5, seed=1, color=REF, lw=1.1)
    trace(ax, 38, 16.5, 19, 9.5, seed=1, color=CHIP, lw=0.8, noise=0.5)
    txt(ax, 47.5, 27.0, "its output vs\nthe chip's", size=4.4, ha="center", color=MUTED)
    arrow(ax, (59, 22.5), (63, 22.5))
    txt(ax, 63.5, 14.8, "agreement r for\neach factor k;\nkeep the best", size=5.0, color=INK)
    txt(ax, 63.5, 25.2, "one number,\nno training", size=4.8, color=CHIP, weight="bold")
    arrow(ax, (cx[0] - 6.0, 34), (cx[0] - 6.0, cy + 4.2), color=CHIP, lw=1.3, head=4.5)

    # ---------------------------------------------------------------- output end: reference = the true targets
    box(ax, 96, 3, 84, 31, fc="white", ec=HOST, lw=0.9, r=2.2)
    ax.add_patch(Circle((101, 8.2), 2.4, fc=HOST, ec="none", zorder=8))
    ax.text(101, 8.3, "2", fontsize=7, color="white", fontweight="bold", ha="center", va="center", zorder=9)
    txt(ax, 105, 5.6, "OUTPUT END: the host readout", size=6.8, weight="bold", color=HOST)
    txt(ax, 105, 9.6, "reference: the true targets of the training data", size=5.0, color=INK)
    box(ax, 99, 14, 33, 17, fc="white", ec=INK)
    txt(ax, 115.5, 15.2, "true targets", size=4.9, ha="center", weight="bold")
    txt(ax, 115.5, 19.0, "speech: phonemes\naudio: the spectrum\nmotor: hand velocity", size=4.5, ha="center",
        color=MUTED)
    arrow(ax, (132, 22.5), (136, 22.5))
    txt(ax, 136.5, 14.8, "error between the\nreadout's output and\nthe truth; adjust only\nthe readout's weights",
        size=5.0, color=INK)
    txt(ax, 136.5, 27.5, "a small retraining", size=4.8, color=HOST, weight="bold")
    arrow(ax, (114.5, 34), (114.5, cy - 0.2), color=HOST, lw=1.3, head=4.5)

    # ---------------------------------------------------------------- together: the deployable configuration
    by = 84
    arrow(ax, (57, cy + 17.5), (80, by - 1), color=CHIP, lw=1.0)
    arrow(ax, (128, cy + 17.5), (104, by - 1), color=HOST, lw=1.0)
    box(ax, 46, by, 92, 29, fc="white", ec=INK, lw=1.0, r=2.4)
    txt(ax, W / 2 + 0.5, by + 1.6, "together: a deployable configuration", size=7, weight="bold", ha="center")
    box(ax, 50, by + 7.5, 38, 15, fc=CHIP_T, ec=CHIP)
    txt(ax, 69, by + 8.7, "the real Speck2f chip", size=5.2, weight="bold", ha="center")
    columns(ax, 52, by + 13, 26, 9, [128, 256, 64], fc=CHIP, ec=CHIP, labels=False)
    dial(ax, 82.5, by + 17, r=2.1)
    txt(ax, 82.5, by + 20.6, "× k", size=4.8, color=CHIP, ha="center", weight="bold")
    arrow(ax, (88, by + 15), (95, by + 15), color=CHIP)
    box(ax, 95, by + 7.5, 39, 15, fc=HOST_T, ec=HOST)
    txt(ax, 114.5, by + 8.7, "the refitted host readout", size=5.2, weight="bold", ha="center", color=HOST)
    weights_icon(ax, 101, by + 12.5, 8, 8.5, seed=3)
    arrow(ax, (110, by + 16.8), (114, by + 16.8), color=HOST)
    scores_icon(ax, 115, by + 12.5, 13, 8.5, seed=4)
    txt(ax, W / 2 + 0.5, by + 24.4, "both set in the simulator, on training data, before deployment;\n"
                                    "the network's trained weights never change", size=5.0, color=MUTED, ha="center")

    style.save(fig, HERE / "figures" / "fig_calibration_overview")
    print("wrote figures/fig_calibration_overview.{pdf,svg,png}")


if __name__ == "__main__":
    main()
