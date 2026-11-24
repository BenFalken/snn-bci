"""
The final results, four figures, each with one panel per experiment (Motor I, Motor II, Audio, Speech) and every
decoder of that experiment:

    fig_results_fidelity           each decoder in the experiment's own metric, one dot per session (motor), held-out
                                   word (audio, the paper's mel-band correlation) or test sentence (speech, greedy
                                   phoneme error); the Speck SNN on the chip, its software median as a hollow diamond
    fig_results_latency            compute time per output against the time between outputs (one laptop CPU thread;
                                   the Speck SNN: the chip loop plus the host)
    fig_results_energy             ESTIMATED energy per output from operation counts (45 nm: 4.6 pJ per
                                   multiply-add (MAC), 0.9 pJ per add (ACC), 5 pJ per on-chip memory access): every
                                   decoder under one rule; the Speck SNN's ACCs on the chip plus its host's MACs
    fig_results_energy_measured    auxiliary: MEASURED energy per output on the laptop (CPU decoders by RAPL, idle
                                   subtracted; the Speck SNN as the chip's own power monitor (solid) plus its host's
                                   share by RAPL (hatched))

The final models: motor's Speck SNNs as deployed (Jenkins with its readout refitted in software), audio's P5 with its
readout refitted, speech's V2 (two networks on one chip) with its readout refitted.

    python collect_four.py && python collect_final.py && python collect_speck_ops.py && python fig_results.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig_final
import fig_four
import fig_tradeoff
import style
from fig_final import HERE

TITLE = {"motor1": "Motor (I): monkeys I and L", "motor2": "Motor (II): monkey J",
         "audio": "Audio: heard words", "speech": "Speech: attempted speech"}


# each experiment's reference: the best decoder of the paper it comes from (motor (II) has none of its own: its
# reference set is motor (I)'s), marked in every figure
SOTA = {"motor1": "qrnn", "motor2": "qrnn", "audio": "lstm", "speech": "paper_rnn"}
SOTA_NOTE = ("Shaded, in bold: each experiment's reference, the best decoder of the paper it comes from (QRNN, Ahmadi et "
             "al. 2021; LSTM, Heelan et al. 2019; RNN, Willett et al. 2023). Motor (II) uses the Motor (I) reference.")


def mark_sota(ax, exp):
    """Shade the reference decoder's column (vertical layouts) or bar (horizontal) and label it in bold as the
    reference (drawn as its own text: matplotlib rebuilds tick labels at draw time and drops their weight)."""
    d = SOTA[exp]
    names = {style.SHORT[d], style.SHORT[d].replace("\n", " "), style.LABEL[d]}
    for axis, span, vertical in ((ax.xaxis, ax.axvspan, True), (ax.yaxis, ax.axhspan, False)):
        labs = axis.get_ticklabels()
        texts = [t.get_text() for t in labs]
        hit = [k for k, t in enumerate(texts) if t in names]
        if not hit:
            continue
        k = hit[0]
        locs = list(axis.get_ticklocs())
        loc, size = locs[k], labs[k].get_fontsize()
        span(loc - 0.42, loc + 0.42, color="#eceef3", zorder=0, lw=0)
        label = texts[k].replace("\n", " ") + ("\nreference" if vertical else " (reference)")
        texts[k] = ""
        axis.set_ticks(locs)
        axis.set_ticklabels(texts)
        tick = axis.get_major_ticks()[k]
        off = tick.get_tick_padding() + tick.get_pad()
        if vertical:
            ax.annotate(label, xy=(loc, 0), xycoords=("data", "axes fraction"), xytext=(0, -off),
                        textcoords="offset points", ha="center", va="top", fontsize=size, fontweight="bold",
                        color=style.INK, annotation_clip=False)
        else:
            ax.annotate(label, xy=(0, loc), xycoords=("axes fraction", "data"), xytext=(-off, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=size, fontweight="bold",
                        color=style.INK, annotation_clip=False)
        return


def grid(name, draw, data, h=150):
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, h * style.MM))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42)
    for k, exp in enumerate(fig_four.EXPS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        draw(ax, exp, data)
        mark_sota(ax, exp)
        ax.set_title(TITLE[exp], loc="left", pad=8)
        style.panel_label(ax, "abcd"[k], x=-0.2)
    style.save(fig, HERE / "figures" / name)
    print(f"wrote figures/{name}.{{pdf,svg,png}}")


def estimate_panel(ax, exp, _):
    t = fig_tradeoff.table(exp)
    t = t.dropna(subset=["estimate"]).sort_values("estimate")
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        c = style.COLOR[r.decoder]
        ax.barh(i, r.estimate, color=c, height=0.62, alpha=1.0 if r.decoder in style.HIGHLIGHT else 0.85)
        v = r.estimate
        lab = f"{v:,.0f} µJ" if v >= 100 else f"{v:.3g} µJ"
        if r.decoder == "speck":
            lab += "  (chip adds + host multiply-adds)"
        ax.text(v * 1.25, i, lab, va="center", fontsize=5.6, color=style.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([style.SHORT[d].replace("\n", " ") for d in t.decoder], fontsize=6.2)   # as the other figures
    ax.set_xscale("log")
    lo, hi = t.estimate.min(), t.estimate.max()
    ax.set_xlim(lo / 3, hi * 60)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    iv = t.interval_ms.iloc[0]
    ax.set_xlabel(f"estimated energy per output (µJ), one output every {iv:g} ms")


def main():
    fid = pd.read_csv(HERE / "data" / "four_fidelity.csv")
    eff = pd.read_csv(HERE / "data" / "final_efficiency.csv")
    for exp in fig_four.EXPS:
        fig_four.use(exp, fid, combined=False)

    def with_use(fn, data):
        def draw(ax, exp, _):
            fig_four.use(exp, fid, combined=False)
            fn(ax, exp, data, False)
        return draw
    def fidelity_with_aegru(ax, exp, _):
        fig_four.use(exp, fid, combined=False)
        if exp == "motor1" and (fid.decoder == "aegru").any():      # the re-implemented state of the art: a reference
            decs = [d for d in fig_final.SETS[exp][0] if d != "aegru"] + ["aegru"]   # too, but set apart (no code)
            fig_final.SETS[exp] = (decs,) + tuple(fig_final.SETS[exp][1:])
        fig_final.fidelity(ax, exp, fid, False)
    grid("fig_results_fidelity", fidelity_with_aegru, fid, h=150)
    grid("fig_results_latency", with_use(fig_final.latency, eff), eff, h=140)
    grid("fig_results_energy", estimate_panel, None, h=140)
    grid("fig_results_energy_measured", with_use(fig_final.energy, eff), eff, h=140)


if __name__ == "__main__":
    main()
