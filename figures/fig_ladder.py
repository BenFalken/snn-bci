"""
The design ladders: how each change to a spiking decoder moved its error, one change per step, each step keeping the
ones before. Adaptation appears only as a fork off the trunk, never as a trunk step.

(a) Motor, LSNN: 36 sessions of monkey I, mean per-axis test RMSE (data/motor_ladder.csv, from
    oscar_dump_motor_ladder.sh; step definitions in motor/sbatch_scripts/snn_config.sh, ladder_* and lsnn_ladder_*).
(b) Motor, Speck SNN: the chip's branch from the same start (ladder_chip, then speck_delay).
(c) Speech: 96-epoch runs on all 19 sessions, phoneme error on the test split as computed in training (seed 42 on the
    trunk; three seeds per fork). Values from the final speech ladder (speech/scripts/configs ladder96/fork96/
    nodelay96; runs L0_orig, L1_width, S2_rec, S3_out80, S4_readall, C_nodelay96; forks R_readall96/_s43/_s44 and
    C_lsnn_nodelay96, Q_lsnn_s43/_s44).

    python figures/fig_ladder.py          -> figures/figures/fig_ladder.{pdf,svg,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent

MOTOR_TRUNK = [("start", "Speck-style\nSNN"), ("filters", "+ synaptic\nfilters"), ("soft", "+ soft\nreset"),
               ("lsnn_neurons", "+ LSNN\nneurons"), ("recurrence", "+ recur-\nrence"), ("width", "+ width\n2 × 256")]
MOTOR_FORK = [("adaptation", "+ adapt-\nation"), ("pretrained", "+ pre-\ntraining")]
MOTOR_CHIP = [("start", "Speck-style\nSNN"), ("chip", "chip's\nshape"), ("delays", "+ input\ndelays")]

SPEECH_TRUNK = [("Original SNN\n(feedforward,\n512-256-128)", 0.900), ("+ width\n(2 × 1,024)", 0.7226),
                ("+ recurrence", 0.3423), ("+ outputs\nevery 80 ms", 0.2636),
                ("+ readout from\nboth layers", 0.2639), ("+ trained synaptic\ntime constants", 0.2627)]
SPEECH_FORK = {"delaynet": ("+ input delays\n(DelayNet)", [0.2449, 0.2492, 0.2504]),
               "lsnn": ("+ adaptation\n(LSNN)", [0.2499, 0.2446, 0.2450])}
SPEECH_REFS = [(0.208, "LSTM, same input and training: 0.208"), (0.188, "Paper RNN (5 × 1,024): 0.188")]

TRUNK_C = style.INK
FORK_C = style.COLOR["lsnn"]
CHIP_C = style.COLOR["speck"]
DELAY_C = style.COLOR["delaynet"]


def load_motor():
    import csv
    rows = list(csv.DictReader(open(HERE / "data" / "motor_ladder.csv")))
    cols = {k: np.array([float(r[k]) for r in rows if r[k]]) for k in rows[0] if k != "session"}
    n = {len(v) for v in cols.values()}
    assert len(n) == 1, f"sessions differ between steps: {n}"
    return cols, n.pop()


def mean_ci(v):
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def draw_steps(ax, xs, ms, cis, color, labels_y=True, fmt="{:.1f}", dy=0.6, lw=1.4, ms_size=4.5):
    ax.plot(xs, ms, "-", color=color, lw=lw, zorder=2)
    ax.errorbar(xs, ms, yerr=cis, fmt="o", color=color, ms=ms_size, lw=0.8, capsize=1.5, zorder=3)
    if labels_y:
        for x, m, c in zip(xs, ms, cis):
            ax.text(x, m + c + dy, fmt.format(m), ha="center", va="bottom", fontsize=6.3, color=color)


def main():
    style.apply()
    cols, n = load_motor()
    fig = plt.figure(figsize=(style.DOUBLE, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.55, 1], height_ratios=[1, 1.05], hspace=0.62, wspace=0.22)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # (a) motor LSNN ladder
    t = [mean_ci(cols[k]) for k, _ in MOTOR_TRUNK]
    f = [mean_ci(cols[k]) for k, _ in MOTOR_FORK]
    xt = np.arange(len(t)); xf = np.arange(len(t), len(t) + len(f))
    draw_steps(a, xt, [m for m, _ in t], [e for _, e in t], TRUNK_C)
    a.plot([xt[-1], xf[0]], [t[-1][0], f[0][0]], "-", color=FORK_C, lw=1.4, zorder=2)
    draw_steps(a, xf, [m for m, _ in f], [e for _, e in f], FORK_C)
    a.set_xticks(list(xt) + list(xf))
    a.set_xticklabels([l for _, l in MOTOR_TRUNK] + [l for _, l in MOTOR_FORK], fontsize=6)
    for tick in a.get_xticklabels()[len(t):]:
        tick.set_color(FORK_C)
    a.axvspan(xf[0] - 0.5, xf[-1] + 0.5, color=FORK_C, alpha=0.06, lw=0)
    a.text(xf.mean(), 46.2, "LSNN fork", color=FORK_C, ha="center", fontsize=6.5, fontweight="bold")
    a.set_ylim(28, 47.5); a.set_xlim(-0.5, xf[-1] + 0.5)
    a.set_ylabel("per-axis RMSE (mm/s)")
    a.set_title("Motor, LSNN", pad=8)
    style.panel_label(a, "a", x=-0.09)

    # (b) motor Speck branch from the same start, with the software filters step as a reference
    s = [mean_ci(cols[k]) for k, _ in MOTOR_CHIP]
    xs = np.arange(len(s))
    draw_steps(b, xs, [m for m, _ in s], [e for _, e in s], CHIP_C)
    b.axhline(t[1][0], color=style.MUTED, ls="--", lw=0.8, zorder=1)
    b.text(-0.42, t[1][0] - 0.3, f"synaptic filters (a): {t[1][0]:.1f}", ha="left", va="top",
           fontsize=5.8, color=style.MUTED)
    b.set_xticks(xs); b.set_xticklabels([l for _, l in MOTOR_CHIP], fontsize=6)
    b.set_ylim(28, 47.5); b.set_xlim(-0.5, xs[-1] + 0.5)
    b.set_title("Motor, Speck SNN", pad=8)
    style.panel_label(b, "b", x=-0.2)

    # (c) speech ladder with its fork
    ys = [p for _, p in SPEECH_TRUNK]; xt = np.arange(len(ys))
    c.plot(xt, ys, "-", color=TRUNK_C, lw=1.4, zorder=2)
    c.scatter(xt, ys, s=18, color=TRUNK_C, zorder=3)
    for x, y in zip(xt, ys):
        steep = 0 < x < 3                      # labels beside the steep part of the curve, off the line
        c.text(x + (0.12 if steep else 0), y + (0.015 if steep else 0.03), f"{y:.3f}",
               ha="left" if steep else "center", va="bottom", fontsize=6.3)
    end = len(ys)
    for (key, (label, seeds)), x, color in zip(SPEECH_FORK.items(), (end, end + 1), (DELAY_C, FORK_C)):
        m = float(np.mean(seeds))
        c.plot([xt[-1], x], [ys[-1], m], "-", color=color, lw=1.4, zorder=2)
        c.scatter(x + np.array([-0.09, 0, 0.09]), seeds, s=9, color=color, edgecolor="white", lw=0.3, zorder=4)
        c.plot([x - 0.2, x + 0.2], [m, m], color=color, lw=1.6, zorder=3)
        c.text(x, max(seeds) + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=6.3, color=color, fontweight="bold")
    c.axvspan(end - 0.5, end + 1.5, color=FORK_C, alpha=0.06, lw=0)
    c.text(end + 0.5, 0.86, "the fork: one memory\nmechanism, either way", ha="center", fontsize=6.3, color=style.INK)
    for y, lab in SPEECH_REFS:
        c.axhline(y, color=style.MUTED, ls="--" if y > 0.2 else ":", lw=0.8, zorder=1)
        up = y > 0.2
        c.text(-0.4, y + (0.006 if up else -0.008), lab, ha="left", va="bottom" if up else "top", fontsize=5.8,
               color=style.MUTED)
    c.set_xticks(list(xt) + [end, end + 1])
    c.set_xticklabels([l for l, _ in SPEECH_TRUNK] + [v[0] for v in SPEECH_FORK.values()], fontsize=5.8)
    c.get_xticklabels()[end].set_color(DELAY_C); c.get_xticklabels()[end + 1].set_color(FORK_C)
    c.set_ylim(0.12, 0.98); c.set_xlim(-0.5, end + 1.5)
    c.set_ylabel("phoneme error (test split)")
    c.set_title("Speech, recurrent integrate-and-fire network", pad=8)
    style.panel_label(c, "c", x=-0.045)
    (HERE / "figures").mkdir(exist_ok=True)
    style.save(fig, HERE / "figures" / "fig_ladder")
    print(f"motor sessions: {n}; trunk {[round(m, 2) for m, _ in t]}; fork {[round(m, 2) for m, _ in f]}; "
          f"chip {[round(m, 2) for m, _ in s]}")


if __name__ == "__main__":
    main()
