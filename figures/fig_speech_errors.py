"""
Speech: the chip network misses phonemes rather than mistaking them (data/speech_errors.json, collect_speech_errors.py).

    (a) substituted, deleted and inserted phonemes per true phoneme, greedy and with the phoneme 5-gram
    (b) deleted phonemes by place in the word (greedy)
    (c) the Speck SNN with the blank's log-probability lowered: deletions turn into substitutions and insertions

    python3 fig_speech_errors.py      -> figures/fig_speech_errors.{pdf,svg,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).parent
DEC = ["paper_rnn", "lsnn", "speck"]
NAME = {"paper_rnn": "Paper RNN", "lsnn": "LSNN", "speck": "Speck SNN\n(chip)"}
KIND = [("S", "substituted", 1.0, None), ("D", "deleted", 0.55, "////"), ("I", "inserted", 0.25, None)]


def main():
    style.apply()
    E = json.load(open(HERE / "data" / "speech_errors.json"))["decoders"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw=dict(width_ratios=[1.45, 1, 1], wspace=0.45))

    ax = axs[0]                                                           # (a) error types
    w, x = 0.34, np.arange(len(DEC))
    for i, d in enumerate(DEC):
        for j, (mode, xo) in enumerate([("greedy", -w / 2 - 0.03), ("lm", w / 2 + 0.03)]):
            bottom = 0
            for k, lab, a, h in KIND:
                v = E[d][mode][k]
                ax.bar(x[i] + xo, v, w, bottom=bottom, color=style.COLOR[d], alpha=a if mode == "greedy" else a * 0.75,
                       hatch=h, edgecolor="white", linewidth=0.4)
                bottom += v
            ax.text(x[i] + xo, bottom + 0.008, f"{bottom:.2f}", ha="center", va="bottom", fontsize=5.5)
            ax.text(x[i] + xo, -0.012, "greedy" if mode == "greedy" else "5-gram", ha="center", va="top",
                    fontsize=4.6, color=style.MUTED)
    ax.set_xticks(x, [NAME[d] for d in DEC])
    ax.tick_params(axis="x", pad=9, length=0)
    ax.set_ylabel("phoneme errors per true phoneme")
    ax.set_ylim(0, 0.42)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=a, hatch=h, ec="white", lw=0.4) for _, _, a, h in KIND]
    ax.legend(handles, [lab for _, lab, _, _ in KIND], loc="upper left", frameon=False, handlelength=1.2)
    ax.set_title("Speck misses phonemes")
    style.panel_label(ax, "a", x=-0.24)

    ax = axs[1]                                                           # (b) deletions by place in the word
    P = ["first", "middle", "last"]
    for d in DEC:
        y = [E[d]["greedy"]["position"][p]["D"] for p in P]
        ax.plot(range(3), y, "-o", color=style.COLOR[d], ms=3.5, lw=1.4, label=NAME[d].replace("\n", " "))
        ax.text(2.1, y[-1] + {"paper_rnn": -0.009, "lsnn": 0.009}.get(d, 0), f"{y[-1]:.2f}", va="center", fontsize=5.5, color=style.COLOR[d])
    ax.set_xticks(range(3), [f"{p}\n(n={E['speck']['greedy']['position'][p]['n']:,})" for p in P])
    ax.set_xlim(-0.3, 2.45)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("deleted (share of phonemes)")
    ax.set_xlabel("place in the word (greedy)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.1)
    ax.set_title("Most within words")
    style.panel_label(ax, "b", x=-0.27)

    ax = axs[2]                                                           # (c) blank penalty, Speck
    pen = json.load(open(HERE / "data" / "speech_errors.json"))["penalty"]
    sw = E["speck"]["blank"]["all"]
    c = style.COLOR["speck"]
    for k, lab, a, h in KIND:
        ax.plot(pen, [s[k] for s in sw], "-", color=c, alpha=max(a, 0.35), lw=1.2, ls=(0, (3, 1.5)) if k == "D" else "-")
        ax.text(pen[-1] + 0.15, sw[-1][k] + {"S": 0, "D": -0.012, "I": 0.012}[k], lab, va="center", fontsize=5.5, color=style.MUTED)
    ax.plot(pen, [s["per"] for s in sw], "-", color=style.INK, lw=1.6)
    ax.text(pen[-1] + 0.15, sw[-1]["per"], "phoneme\nerror", va="center", fontsize=5.5, color=style.INK)
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("blank log-probability lowered by")
    ax.set_ylabel("per true phoneme (Speck, greedy)")
    ax.set_title("Recovered, mostly wrong")
    style.panel_label(ax, "c", x=-0.27)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    style.save(fig, HERE / "figures" / "fig_speech_errors")


if __name__ == "__main__":
    main()
