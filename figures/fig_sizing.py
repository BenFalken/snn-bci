"""
Figure: sizing a decoder from the raw recordings. Every task spreads its information over its own span of time and its
own share of the electrodes; the Speck SNN covers each span with host delays; and channels trade against time both in
the information the recordings hold and in the chip's input budget.

    A  one electrode at a time (collect_lag_tuning.py): the median tuning curve of every tuned channel, each scaled to
       its own peak, against the offset between its activity and the decoded quantity (> 0: activity first)
    B  the whole array (collect_linear_profile.py): how much a linear decoder gains from a longer window of the past,
       each unit as the fraction of its own largest gain, then averaged (motor: the full sessions, R^2); open dots:
       90% of the curve's plateau; bars: the span each task's Speck SNN covers with its delays
    C  adding electrodes (5 taps, best first): the fraction of the full array's gain over the smallest set, against
       the fraction of electrodes used; ticks: the share each Speck SNN reads
    D  the same input budget split into channels x delays (the chip-budget sweep's splits, motor and audio): the
       linear decoder (solid) and the Speck SNN (dashed), relative to the all-channel split, against the memory span
       the split buys

    python3 fig_sizing.py      -> figures/fig_sizing.{pdf,svg,png}
"""
import collections
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)

HERE = Path(__file__).parent
LAG, LP = WORKSPACE / "lag_tuning", WORKSPACE / "linear_profile"
TASK = {  # name: (label, colour, linear-profile placement, higher is better)
    "motor": ("Motor", "#332288", None, False),
    "audio": ("Audio", "#117733", "centred", True),
    "speech": ("Speech", "#AA4499", "causal", True),
}
SPECK_SPAN = {"motor": [200], "audio": [640], "speech": [500, 1200]}         # ms covered by the delays
SPECK_SHARE = {"motor": 96 / 96, "audio": 48 / 96, "speech": 256 / 256}       # electrodes (features) read
N_CH = {"motor": 96, "audio": 96, "speech": 256}
SPLITS = ["96x5", "48x10", "32x15", "24x20", "16x30", "8x60"]


def lag_curves(exp):
    allc, lags = [], None
    for f in glob.glob(str(LAG / exp / "lag_tuning_*.json")):
        d = json.load(open(f))
        lags = np.array(d["lags_ms"])
        allc += [np.array(c["curve"]) / c["peak_r2"] for c in d["channels"] if c["peak_r2"] >= d["settings"]["min_r2"]]
    return lags, np.percentile(np.array(allc), [25, 50, 75], axis=0)


def profile(exp, kind):
    """(keys, units x keys). Motor's memory and breadth come from the full sessions (motor_full.json: the sweep's 42 s
    training sets let longer windows overfit); its budget split from the sweep's own data, to compare with the chip."""
    full = exp == "motor" and kind in ("memory", "breadth")
    L = json.load(open(LP / ("motor_full.json" if full else f"{exp}.json")))
    place = None if full else TASK[exp][2]
    pick = (lambda r: r) if place is None else (lambda r: r[place])
    keys = sorted(pick(next(iter(L.values())))[kind], key=lambda k: int(k) if k.isdigit() else k)
    M = np.array([[pick(r)[kind][k] for k in keys] for r in L.values()])
    return keys, M


def fraction_of_gain(M, up, i0, i1=None):
    """Each unit's curve as the fraction of its own gain from column i0 to column i1 (default: its best column),
    then the mean over units and its SEM."""
    S = M if up else -M
    top = S.max(1, keepdims=True) if i1 is None else S[:, [i1]]
    F = (S - S[:, [i0]]) / (top - S[:, [i0]])
    return F.mean(0), F.std(0, ddof=1) / np.sqrt(len(F))


def chip_split(exp):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(HERE / "data" / "results.csv")):
        if r["experiment"] == exp and r["readout"] == "hidden":
            g[r["unit"]][f"{r['channels']}x{r['n_delays']}"].append(float(r["value"]))
    return {u: {k: float(np.mean(v)) for k, v in d.items()} for u, d in g.items()}


def main():
    style.apply()
    fig, ax = plt.subplots(2, 2, figsize=(style.DOUBLE, 125 * style.MM), gridspec_kw=dict(wspace=0.32, hspace=0.5))
    A, B, C, D = ax.flat

    # A: single electrodes
    for exp, (lab, col, *_) in TASK.items():
        lags, (lo, mid, hi) = lag_curves(exp)
        A.fill_between(lags, lo, hi, color=col, alpha=0.12, lw=0)
        A.plot(lags, mid, color=col, lw=1.4, label=lab)
    A.axvline(0, color=style.MUTED, lw=0.6)
    A.set_xlim(-1000, 1200)
    A.set_ylim(0, 1.05)
    A.set_xlabel("offset (ms; > 0: activity first)")
    A.set_ylabel("tuning / channel's peak")
    A.set_title("One electrode at a time", loc="left")
    A.legend(fontsize=6.5, frameon=False, loc="upper left")

    # B: memory of the whole array
    for exp, (lab, col, place, up) in TASK.items():
        keys, M = profile(exp, "memory")
        span = np.array([(int(k) - 1) * 40 for k in keys])
        f, se = fraction_of_gain(M, True if exp == "motor" else up, 0)      # motor_full is scored as R^2
        B.fill_between(span, f - se, f + se, color=col, alpha=0.12, lw=0)
        B.plot(span, f, color=col, lw=1.4, marker="o", ms=2.5)
        k = int(np.argmax(f >= 0.9 * f.max()))                              # 90% of this curve's plateau
        B.plot(span[k], f[k], "o", mfc="white", mec=col, ms=5, mew=1.2, zorder=5)
        for s in SPECK_SPAN[exp]:
            B.plot([s, s], [-0.13, -0.05], color=col, lw=2.2, solid_capstyle="butt", clip_on=False)
    B.plot([], [], "o", mfc="white", mec=style.MUTED, ms=5, mew=1.2, label="90% of the plateau")
    B.set_ylim(-0.15, 1.12)
    B.set_xlim(-20, 1220)
    B.set_xlabel("span of the past the decoder sees (ms)")
    B.set_ylabel("fraction of the gain from history")
    B.set_title("The array: how far back it must look", loc="left")
    B.plot([], [], color=style.MUTED, lw=2.2, label="bars: Speck SNN's delay span")
    B.legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.06))

    # C: breadth
    for exp, (lab, col, place, up) in TASK.items():
        keys, M = profile(exp, "breadth")
        share = np.array([int(k) for k in keys]) / N_CH[exp]
        f, se = fraction_of_gain(M, True if exp == "motor" else up, 0, len(keys) - 1)
        C.fill_between(share, f - se, f + se, color=col, alpha=0.12, lw=0)
        C.plot(share, f, color=col, lw=1.4, marker="o", ms=2.5)
        nudge = {"motor": 0.93, "audio": 1.0, "speech": 1.0}[exp]          # motor and speech both read all
        C.plot([SPECK_SHARE[exp] * nudge] * 2, [-0.13, -0.05], color=col, lw=2.2, solid_capstyle="butt", clip_on=False)
    C.set_xscale("log", base=2)
    C.set_xticks([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1], ["1/32", "1/16", "1/8", "1/4", "1/2", "all"])
    C.minorticks_off()
    C.set_ylim(-0.15, 1.12)
    C.set_xlabel("share of the electrodes, best first")
    C.set_ylabel("fraction of the full array's gain")
    C.set_title("Adding electrodes: repeat or add?", loc="left")
    C.plot([], [], color=style.MUTED, lw=2.2, label="bars: share the Speck SNN reads")
    C.legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.06))

    # D: the budget split, information vs chip
    spans = np.array([(int(s.split("x")[1]) - 1) * 40 for s in SPLITS])
    pos = np.arange(len(SPLITS))                                              # evenly spaced splits
    for exp in ("motor", "audio"):
        lab, col, place, up = TASK[exp]
        keys, M = profile(exp, "budget")
        Ml = np.array([[row[keys.index(s)] for s in SPLITS] for row in M])
        rel = lambda X: X / X[:, :1] if up else X[:, :1] / X                    # 1 = as good as all channels
        for X, ls, name in ((Ml, "-", "linear"),
                            (np.array([[u[s] for s in SPLITS] for u in chip_split(exp).values()]), "--", "Speck")):
            R = rel(X)
            m, se = R.mean(0), R.std(0, ddof=1) / np.sqrt(len(R))
            D.fill_between(pos, m - se, m + se, color=col, alpha=0.12, lw=0)
            D.plot(pos, m, color=col, lw=1.4, ls=ls, marker="o", ms=2.5, label=f"{lab}, {name}")
    D.set_xticks(pos, [f"{s.replace('x', '×')}\n{t:,} ms" for s, t in zip(SPLITS, spans)], fontsize=5.5)
    D.axhline(1, color=style.MUTED, lw=0.6)
    D.set_ylim(0, 1.15)
    D.set_xlabel("channels × delays at a fixed input budget (memory span)")
    D.set_ylabel("performance relative to all channels")
    D.set_title("Trading channels for time", loc="left")
    D.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)

    for a, k in zip(ax.flat, "abcd"):
        style.panel_label(a, k, x=-0.14)
    style.save(fig, HERE / "figures" / "fig_sizing")
    print("wrote figures/fig_sizing")


if __name__ == "__main__":
    main()
