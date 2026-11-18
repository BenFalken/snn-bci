"""
Slide figures for the four-experiment deck, from data/four_fidelity.csv (collect_four.py) and
data/final_efficiency.csv (collect_final.py):

    four_fidelity.png        one row, four columns: Motor (I), Motor (II), Audio, Speech; each in its own metric,
                             one dot per evaluation unit (session, session, word, sentence), median above
    four_cost.png            two rows, four columns: time per output against the real-time budget; measured energy
    four_<experiment>.png    one experiment's three panels (fidelity with every reference decoder, time, energy),
                             as fig_final.slide

A column whose data do not exist yet (Motor (II) until the jenkins_m1 laptop run) is drawn as a labelled
placeholder, never filled with other data.

    python fig_four.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import fig_final
import slide_style
import style
from fig_final import HERE

EXPS = ["motor1", "motor2", "audio", "speech"]
TITLE = {"motor1": "Motor (I): monkeys I and L", "motor2": "Motor (II): monkey J, M1 array",
         "audio": "Audio: heard words", "speech": "Speech: T12, attempted speech"}
UNIT = {"session": "session", "word": "word", "neil": "word", "gomez": "word", "sentence": "sentence", "indy": "session", "jenkins_m1": "session"}
# experiment: (all decoders, combined-figure decoders, metric, axis label, higher is better, groups, efficiency rows)
SPEC = {
    "motor1": (["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], ["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], "cc",
               "velocity correlation", True, {"indy": "o", "loco": "s"}, "motor"),
    "motor2": (["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], ["kf", "wf", "lstm", "qrnn", "lsnn", "speck"], "cc",
               "velocity correlation", True, {"jenkins_m1": "^"}, "motor2"),
    "audio": (["kf", "wf", "wiener_cascade", "dense_nn", "simple_rnn", "gru", "lstm", "lsnn", "speck"],
              ["kf", "wf", "gru", "lstm", "lsnn", "speck"], "band_r", "mel-band correlation (paper's metric)", True,
              {"neil": "o", "gomez": "s"}, "audio"),
    "speech": (["paper_rnn", "lsnn", "speck"], ["paper_rnn", "lsnn", "speck"], "per",
               "phoneme error rate", False, {"sentence": "o", "session": "o"}, "speech"),
}
fig_final.GROUP_NAME.update({"jenkins_m1": "monkey J", "neil": "monkey N", "gomez": "monkey G"})
PENDING = {"motor2": "Jenkins, 96-channel M1 array:\nruns in progress on Oscar"}


def use(exp, fid, combined):
    """Point fig_final's panel functions at one experiment's decoders, metric and unit."""
    alldec, comb, metric, label, higher, groups, rows = SPEC[exp]
    present = {g: m for g, m in groups.items() if g in set(fid[fid.experiment == exp].group)}
    unit = UNIT[next(iter(present))] if present else "session"
    fig_final.SETS[exp] = (comb if combined else alldec, metric, label, higher, present or groups, unit, rows)
    return unit


def has(exp, fid, eff=None):
    if eff is not None:
        return (eff.experiment == SPEC[exp][6]).any()
    return ((fid.experiment == exp) & (fid.metric == SPEC[exp][2])).any()


def pending(ax, exp):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (4, 4))); s.set_color(style.MUTED)
    ax.text(0.5, 0.5, PENDING[exp], ha="center", va="center", fontsize=13, color=style.MUTED,
            transform=ax.transAxes)


def short_ticks(ax, decs, axis="x"):
    labels = [style.SHORT[d] if axis == "x" else style.SHORT[d].replace("\n", " ") for d in decs]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)
    for t, d in zip(ax.get_xticklabels() if axis == "x" else ax.get_yticklabels(), decs):
        t.set_fontweight("bold" if d in style.HIGHLIGHT else "normal")


def combined_fidelity(fid):
    slide_style.apply()
    fig, axes = plt.subplots(1, 4, figsize=(slide_style.W, 6.2), gridspec_kw=dict(wspace=0.32))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.17)
    for ax, exp in zip(axes, EXPS):
        if not has(exp, fid):
            pending(ax, exp)
            ax.set_title(TITLE[exp], loc="left", fontsize=15, pad=30)
            continue
        unit = use(exp, fid, combined=True)
        fig_final.fidelity(ax, exp, fid, True)
        short_ticks(ax, fig_final.SETS[exp][0])
        ax.tick_params(axis="x", labelsize=11)
        for t in [t for t in ax.texts if t.get_text() == "software"]:
            t.remove()
        ax.set_title(f"{TITLE[exp]}\none dot per {unit}", loc="left", fontsize=15, pad=30, linespacing=1.5)
    fig.legend([plt.Line2D([], [], marker="D", ls="", mfc="white", mec=style.COLOR["speck"], mew=1.5, ms=8)],
               ["Speck SNN in software (median); filled dots and box: on the chip"], loc="lower left",
               bbox_to_anchor=(0.05, 0.0), frameon=False, fontsize=12)
    slide_style.save(fig, "four_fidelity")


def combined_cost(fid, eff):
    slide_style.apply()
    fig, axes = plt.subplots(2, 4, figsize=(slide_style.W, 7.6), gridspec_kw=dict(wspace=0.55, hspace=0.55))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.09)
    for c, exp in enumerate(EXPS):
        for r, fn in enumerate((fig_final.latency, fig_final.energy)):
            ax = axes[r, c]
            if not has(exp, fid, eff):
                pending(ax, exp)
            else:
                use(exp, fid, combined=True)
                fn(ax, exp, eff, True)
                short_ticks(ax, fig_final.SETS[exp][0], axis="y")
                ax.xaxis.label.set_size(11)
                for t in ax.texts:
                    t.set_fontsize(10)
            if r == 0:
                ax.set_title(TITLE[exp], loc="left", fontsize=15, pad=12)
    slide_style.save(fig, "four_cost")


def one(exp, fid, eff):
    if not has(exp, fid):
        slide_style.apply()
        fig, ax = plt.subplots(figsize=(slide_style.W, 6.4))
        pending(ax, exp)
        slide_style.save(fig, "four_" + exp)
        return
    use(exp, fid, combined=False)
    if not has(exp, fid, eff):                      # fidelity exists, time and energy not yet
        slide_style.apply()
        fig, ax = plt.subplots(figsize=(slide_style.W * 0.45, 6.4))
        fig_final.fidelity(ax, exp, fid, True)
        slide_style.save(fig, "four_" + exp)
        return
    save, tick = slide_style.save, fig_final.tick
    slide_style.save = lambda fig, name: save(fig, "four_" + exp)
    if len(fig_final.SETS[exp][0]) > 6:             # many decoders: short tick labels so they do not collide
        fig_final.tick = lambda d, big: style.SHORT[d]
    try:
        fig_final.slide(exp, fid, eff)
    finally:
        slide_style.save, fig_final.tick = save, tick


if __name__ == "__main__":
    fid = pd.read_csv(HERE / "data" / "four_fidelity.csv")
    eff = pd.read_csv(HERE / "data" / "final_efficiency.csv")
    combined_fidelity(fid)
    combined_cost(fid, eff)
    for exp in EXPS:
        one(exp, fid, eff)
