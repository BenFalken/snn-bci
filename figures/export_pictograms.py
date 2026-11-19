"""
Each pictogram as its own editable SVG (every part a named group), for editing outside the figure:
    python figures/export_pictograms.py   -> figures/figures/pictograms/{monkey_reach,monkey_rest,person,
                                             brain_macaque,brain_human}.svg (and .pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pictograms as pg

OUT = Path(__file__).resolve().parent / "figures" / "pictograms"
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def one(name, draw, lim=((-0.8, 0.8), (-1.0, 0.9))):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    draw(ax)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", transparent=True)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
one("monkey_reach", lambda ax: pg.monkey(ax, -0.25, 0, 0.9, reach_to=(0.75, 0.05), name="monkey"),
    ((-0.95, 0.85), (-1.0, 0.65)))
one("monkey_rest", lambda ax: pg.monkey(ax, 0, 0, 0.9, name="monkey"), ((-0.75, 0.6), (-1.0, 0.65)))
one("person", lambda ax: pg.person(ax, 0, 0, 1.0, name="person", brain_site="6v", site_color="#AA4499"),
    ((-0.45, 0.4), (-1.0, 0.8)))
one("brain_macaque", lambda ax: pg.brain(ax, 0, 0, 1.0, human=False, name="brain"), ((-1.0, 1.0), (-0.9, 0.7)))
one("brain_human", lambda ax: pg.brain(ax, 0, 0, 1.0, human=True, name="brain"), ((-1.0, 1.0), (-1.0, 0.95)))
print("wrote", OUT)
