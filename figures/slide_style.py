"""
Slide versions of the paper figures: the same data and colors as style.py, at slide size (drawn for a 1664 px wide
area at 100 dpi) with larger text, one experiment or one question per figure. Written to figures/slides/.
"""

import os

import matplotlib.pyplot as plt

import style

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "slides")
W = 16.6                      # inches at 100 dpi = the slide's 1664 px content width


def apply():
    style.apply()
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 1.0,
                         "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 4,
                         "ytick.major.size": 4, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
                         "savefig.dpi": 100})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), facecolor="white", dpi=100)
    plt.close(fig)
    print("wrote figures/slides/" + name + ".png")
