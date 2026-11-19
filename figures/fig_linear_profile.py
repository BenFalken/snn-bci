"""
Figure: how much input history and how many channels each task needs, asked of a linear decoder (collect_linear_
profile.py), and where the Speck SNN departs from it (the chip-budget sweep, data/results.csv).

Rows: motor (8 Indy sessions; RMSE, mm/s, axis flipped so up is better), audio (4 Neil days; mean correlation over the
128 bands), speech (6 T12 sessions; phoneme identity, frame accuracy, chance dotted). Means over units, band +-1 SEM.
Each unit's change, so differences between recording days do not hide each one's trend:
    a  memory: change from no history against the span of the taps (40 ms apart; all channels); dot: 90% of the gain
    b  breadth: change from all channels against the number of channels (best first; 5 taps)
    c  budget: change from the first split, for the linear decoder (solid) and the Speck SNN of the chip-budget sweep
       (dashed; same metric, same units), at the same splits of the chip's inputs, channels x delays

    python3 fig_linear_profile.py      -> figures/fig_linear_profile.{pdf,svg,png}
"""
import collections
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)

HERE = Path(__file__).parent
LP = WORKSPACE / "linear_profile"
ROWS = {  # experiment: (title, placements (name, label), metric label, higher is better)
    "motor": ("Motor (I): hand velocity", [(None, "causal taps")], "RMSE (mm/s)", False),
    "audio": ("Audio: heard sound", [("centred", "taps around the sound"), ("sweep", "the sweep's taps")],
              "correlation (r)", True),
    "speech": ("Speech: phoneme identity", [("causal", "past taps"), ("centred", "taps both sides")],
               "frame accuracy", True),
}
LINE = {0: style.INK, 1: style.MUTED}


def stack(L, place, kind, base=None):
    """Mean and SEM over units; with base (a key), each unit's change from its own value there."""
    pick = (lambda r: r) if place is None else (lambda r: r[place])
    keys = list(pick(next(iter(L.values())))[kind])
    M = np.array([[pick(r)[kind][k] for k in keys] for r in L.values()])
    if base is not None:
        M = M - M[:, [keys.index(base)]]
    return keys, M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))


def chip_budget(exp, keys, base):
    """The chip sweep per unit (mean over seeds), as each unit's change from its value at `base`; mean, SEM."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(HERE / "data" / "results.csv")):
        if r["experiment"] == exp and r["readout"] == "hidden":
            g[r["unit"]][f"{r['channels']}x{r['n_delays']}"].append(float(r["value"]))
    units = [u for u in g if all(k in g[u] for k in keys)]
    if not units:
        return None
    M = np.array([[np.mean(g[u][k]) for k in keys] for u in units])
    M = M - M[:, [keys.index(base)]]
    return M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))


def band(ax, x, m, se, color, label=None, ls="-"):
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, lw=1.2, ls=ls, marker="o", ms=2.5, label=label)


def main():
    style.apply()
    fig, axes = plt.subplots(3, 3, figsize=(style.DOUBLE, 150 * style.MM),
                             gridspec_kw=dict(wspace=0.55, hspace=0.6))
    for i, (exp, (title, places, ylab, up)) in enumerate(ROWS.items()):
        f = LP / f"{exp}.json"
        if not f.is_file():
            for ax in axes[i]:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", color=style.MUTED)
                ax.set_axis_off()
            continue
        L = json.load(open(f))
        sign = 1 if up else -1
        # a: memory, change from no history
        ax = axes[i, 0]
        for j, (place, lab) in enumerate(places):
            keys, m, se = stack(L, place, "memory", base=list(stack(L, place, "memory")[0])[0])
            span = np.array([(int(k) - 1) * 40 for k in keys])
            band(ax, span, m, se, LINE[j], lab)
            gain = m * sign
            if j == 0 and gain.max() > 0:
                if gain[-1] >= gain.max() - 1e-12 and gain[-1] - gain[-2] > 0.1 * gain.max():
                    ax.annotate(f"still rising at {span[-1]:,} ms", (span[-1], m[-1]), xytext=(-4, 6),
                                textcoords="offset points", fontsize=6, color=style.COLOR["speck"], ha="right")
                else:
                    k = int(np.argmax(gain >= 0.9 * gain.max()))
                    ax.plot(span[k], m[k], "o", color=style.COLOR["speck"], ms=5, zorder=5)
                    ax.annotate(f"90% by {span[k]} ms", (span[k], m[k]), xytext=(6, -10 if up else 8),
                                textcoords="offset points", fontsize=6, color=style.COLOR["speck"])
        ax.axhline(0, color=style.MUTED, lw=0.6)
        ax.set_ylabel(f"{title}\n\nchange in {ylab}")
        ax.set_title("Memory: from no history", loc="left")
        if len(places) > 1:
            ax.legend(fontsize=5.5, frameon=False, loc="lower right" if up else "upper right")
        # b: breadth, change from all channels
        ax = axes[i, 1]
        place = places[0][0]
        keys = stack(L, place, "breadth")[0]
        keys, m, se = stack(L, place, "breadth", base=keys[-1])
        C = np.array([int(k) for k in keys])
        band(ax, C, m, se, LINE[0])
        ax.axhline(0, color=style.MUTED, lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(C, [str(c) for c in C], fontsize=6)
        ax.minorticks_off()
        ax.set_title(f"Breadth: from all {C[-1]}", loc="left")
        # c: budget split, linear vs the chip sweep, both as change from the first split
        ax = axes[i, 2]
        keys = stack(L, place, "budget")[0]
        keys, m, se = stack(L, place, "budget", base=keys[0])
        x = np.arange(len(keys))
        band(ax, x, m, se, style.INK, "linear")
        chip = chip_budget(exp, keys, keys[0])
        if chip is not None:
            band(ax, x, chip[0], chip[1], style.COLOR["speck"], "Speck (chip sweep)", ls="--")
        else:
            ax.text(0.98, 0.95, "no chip sweep result", transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                    color=style.MUTED)
        ax.axhline(0, color=style.MUTED, lw=0.6)
        ax.set_xticks(x, [k.replace("x", "×") for k in keys], rotation=45, fontsize=6)
        ax.set_title(f"Budget: from {keys[0].replace('x', '×')}", loc="left")
        ax.legend(fontsize=5.5, frameon=False, loc={"motor": "lower left", "audio": "lower left"}.get(exp, "upper left"))
        for a in axes[i]:
            if not up:
                a.invert_yaxis()
    axes[-1, 0].set_xlabel("span of the taps (ms)")
    axes[-1, 1].set_xlabel("channels (best first)")
    axes[-1, 2].set_xlabel("channels × delays")
    for ax, k in zip(axes.flat, "abcdefghi"):
        style.panel_label(ax, k, x=-0.3)
    style.save(fig, HERE / "figures" / "fig_linear_profile")
    print("wrote figures/fig_linear_profile")


if __name__ == "__main__":
    main()
