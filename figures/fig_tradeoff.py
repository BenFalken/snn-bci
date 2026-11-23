"""
Fidelity against latency, sized by memory, and measured energy linked to its operation-count estimate, for the four
experiments (Motor (II) a placeholder until its runs exist).

    row 1  each decoder's median fidelity (data/four_fidelity.csv, the experiment's metric) against its compute time
           per output (data/final_efficiency.csv: one laptop CPU thread; Speck: the chip loop's wall time plus the
           host), log scale, the real-time budget dashed. Dot AREA grows with the square root of the decoder's weight
           memory as deployed: fp32 (4 bytes per parameter) for every CPU decoder, the Speck SNN's chip weights at
           8 bits plus its host readout at fp32
    row 2  energy per output: hollow = estimate from operation counts (45 nm: 4.6 pJ per multiply-add, 0.9 pJ per
           add, 5 pJ per on-chip memory access; collect_efficiency.py and collect_speck_ops.py), filled = measured
           (laptop CPU by RAPL, idle subtracted; Speck: the chip's own power monitor), a line joining the two, the
           measured/estimated ratio at the right. Speck's chip + laptop host total as a cross

    python fig_tradeoff.py   -> figures/slides/tradeoff_four.png and tradeoff_<experiment>.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slide_style
import style
from collect_efficiency import energy_uj
from fig_efficiency import fmt_uj

HERE = Path(__file__).parent
EXPS = ["motor1", "motor2", "audio", "speech"]
TITLE = {"motor1": "Motor (I)", "motor2": "Motor (II)", "audio": "Audio", "speech": "Speech"}
METRIC = {"motor1": ("cc", "velocity correlation", True), "motor2": ("cc", "velocity correlation", True),
          "audio": ("band_r", "mel-band correlation", True), "speech": ("per", "phoneme error rate", False)}
EFF_ROWS = {"motor1": "motor", "motor2": "motor2", "audio": "audio", "speech": "speech"}
DECODERS = {"motor1": ["kf", "wf", "lstm", "qrnn", "lsnn", "speck"],
            "motor2": ["kf", "wf", "lstm", "qrnn", "lsnn", "speck"],
            "audio": ["kf", "wf", "wiener_cascade", "dense_nn", "simple_rnn", "gru", "lstm", "lsnn", "speck"],
            "speech": ["paper_rnn", "lsnn", "speck"]}

# parameters at the trained sizes (counted from the modules final_bench.py builds; Speck from its checkpoints)
PARAMS = {
    "motor1": {"kf": 9864, "wf": 2882, "lstm": 238802, "qrnn": 232402, "lsnn": 222726},
    "audio": {"kf": 54272, "wf": 209024, "wiener_cascade": 209536, "dense_nn": 1803392, "simple_rnn": 1280128,
              "gru": 3577984, "lstm": 4726912, "lsnn": 287488},
    "speech": {"gru": 3980329, "lstm": 5293097, "paper_rnn": 53550121, "lsnn": 3498834},
}
PARAMS["motor2"] = PARAMS["motor1"]          # Jenkins's M1 array: the same 96-channel architectures as Motor (I)
# the Speck SNN: (chip weights at 8 bits, host readout parameters at fp32)
SPECK_PARAMS = {"motor1": (384 * 128 + 128 * 256 + 256 * 256 + 256 * 36, 0),
                "audio": (432 * 128 + 128 * 256 + 256 * 256, 256 * 128 + 128),
                "speech": (2 * (512 * 128 + 128 * 256 + 256 * 64) + 256 * 64 + 64 * 256 + 256 * 64,
                           1280 * 5 * 41 + 41)}   # V2+C: three networks, one chip
SPECK_PARAMS["motor2"] = SPECK_PARAMS["motor1"]   # checked: 384-128-256-256-36, as indy
# operation-count estimates the older table lacks (audio): Wiener cascade = Wiener + a cubic per band; simple RNN over
# the 17-frame window, as the LSTM and GRU rows are counted
def _fine_lsnn():
    """The audio LSNN on real 4 ms spikes (LS1): its op-count estimate (collect_fine_ops.py -> data/fine_ops.csv)."""
    f = pd.read_csv(HERE / "data" / "fine_ops.csv").set_index("decoder")
    return float(f.loc["LS1", "energy_uj"])


EXTRA_EST = {("audio", "lsnn"): _fine_lsnn(),
             ("audio", "wiener_cascade"): energy_uj(17 * 96 * 128 + 128 * 3, 0),
             ("audio", "simple_rnn"): energy_uj(17 * (96 * 1024 + 1024 * 1024) + 1024 * 128, 0)}


def memory_bytes(exp, d):
    if d == "speck":
        chip_w, host_w = SPECK_PARAMS[exp]
        return chip_w + 4 * host_w
    return 4 * PARAMS[exp][d]


def table(exp):
    fid = pd.read_csv(HERE / "data" / "four_fidelity.csv")
    eff = pd.read_csv(HERE / "data" / "final_efficiency.csv")
    est = pd.read_csv(HERE / "data" / "efficiency.csv")
    ops = pd.read_csv(HERE / "data" / "speck_ops.csv").set_index("experiment")
    metric = METRIC[exp][0]
    f = fid[(fid.experiment == exp) & (fid.metric == metric)]
    if f.empty:
        return None
    rows = []
    e = eff[eff.experiment == EFF_ROWS[exp]].set_index("decoder")
    s = est[est.experiment == ("motor" if exp == "motor2" else EFF_ROWS[exp])].set_index("decoder")
    for d in DECODERS[exp]:
        if d not in e.index or d not in set(f.decoder):
            continue
        r = dict(decoder=d, fidelity=f[f.decoder == d].value.median(), latency_ms=e.loc[d, "latency_ms"],
                 interval_ms=e.loc[d, "interval_ms"], mem=memory_bytes(exp, d))
        if d == "speck":
            r.update(measured=e.loc[d, "chip_uj"], total=e.loc[d, "chip_uj"] + e.loc[d, "host_uj"],
                     estimate=ops.loc[EFF_ROWS[exp], "energy_uj"])
        else:
            r.update(measured=e.loc[d, "energy_uj"], total=np.nan,
                     estimate=EXTRA_EST.get((exp, d), s.loc[d, "energy_uj"] if d in s.index else np.nan))
        rows.append(r)
    return pd.DataFrame(rows)


def size(mem):
    return 20 + 110 * np.log10(max(mem, 1e3) / 1e3)   # points^2; area grows with log10(bytes): 1 kB to 200 MB fits


# label offsets (points) where decoders crowd each other
OFFSET = {("motor1", "lstm"): (8, -14), ("motor1", "qrnn"): (8, 4), ("motor2", "lstm"): (8, -14),
          ("motor2", "qrnn"): (8, 4),
          ("audio", "gru"): (-34, 10), ("audio", "lstm"): (10, 6), ("audio", "lsnn"): (10, -14),
          ("audio", "simple_rnn"): (-12, -20), ("audio", "dense_nn"): (6, -18), ("audio", "wf"): (-24, -18),
          ("audio", "wiener_cascade"): (9, 2),
          ("speech", "gru"): (-38, -2), ("speech", "lstm"): (6, -22), ("speech", "paper_rnn"): (12, 8),
          ("speech", "lsnn"): (-14, -22)}


def fmt_mem(b):
    return f"{b / 1e6:.3g} MB" if b >= 1e6 else f"{b / 1e3:.3g} kB"


def bubbles(ax, exp, t, fs):
    metric, label, higher = METRIC[exp]
    for _, r in t.iterrows():
        c = style.COLOR[r.decoder]
        ax.scatter(r.latency_ms, r.fidelity, s=size(r.mem), color=c, alpha=0.8, edgecolor="white", linewidth=0.8,
                   zorder=3)
        ax.annotate(style.SHORT[r.decoder].replace("\n", " "), (r.latency_ms, r.fidelity),
                    xytext=OFFSET.get((exp, r.decoder), (7, 5)),
                    textcoords="offset points", fontsize=fs, fontweight="bold" if r.decoder in style.HIGHLIGHT else
                    "normal", color=style.INK)
    iv = t.interval_ms.iloc[0]
    ax.axvline(iv, color=style.MUTED, ls="--", lw=1.2)
    ax.text(iv * 1.15, 0.02, f"budget\n{iv:g} ms", ha="left", va="bottom", fontsize=fs - 1,
            color=style.MUTED, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(min(5e-4, t.latency_ms.min() / 3) if exp != "speech" else t.latency_ms.min() / 5, iv * 12)
    ax.set_xlabel("compute time per output (ms, log)")
    ax.set_ylabel(label + ("" if higher else " (lower is better)"))
    lo, hi = t.fidelity.min(), t.fidelity.max()
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2 * pad)
    if not higher:
        ax.invert_yaxis()


def energy(ax, exp, t, fs):
    t = t.reset_index(drop=True)
    for i, r in t.iterrows():
        c = style.COLOR[r.decoder]
        if np.isfinite(r.estimate):
            ax.plot([r.estimate, r.measured], [i, i], color=c, lw=2, alpha=0.6, zorder=2)
            ax.scatter(r.estimate, i, s=55, facecolor="white", edgecolor=c, linewidth=1.8, zorder=3)
            ax.text(4e8, i, f"×{r.measured / r.estimate:,.0f}", va="center", ha="right", fontsize=fs - 1,
                    color=style.MUTED)
        ax.scatter(r.measured, i, s=55, color=c, zorder=4)
        if np.isfinite(r.total):
            ax.scatter(r.total, i, s=60, marker="x", color=c, linewidth=2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 4e8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([style.SHORT[d].replace("\n", " ") for d in t.decoder])
    for lab, d in zip(ax.get_yticklabels(), t.decoder):
        lab.set_fontweight("bold" if d in style.HIGHLIGHT else "normal")
    ax.invert_yaxis()
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("energy per output (µJ, log)")


def legend_sizes(fig, y):
    hs = [plt.scatter([], [], s=size(b), color=style.MUTED, alpha=0.6) for b in (1e4, 1e5, 1e6, 1e7, 2e8)]
    fig.legend(hs, ["10 kB", "100 kB", "1 MB", "10 MB", "200 MB"], title="weight memory (dot area grows with log memory)",
               loc="lower left", bbox_to_anchor=(0.03, y), ncol=5, frameon=False, fontsize=11, title_fontsize=11,
               columnspacing=1.6, handletextpad=0.6)
    marks = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK, ms=8),
             plt.Line2D([], [], marker="o", ls="", color=style.INK, ms=8),
             plt.Line2D([], [], marker="x", ls="", color=style.INK, ms=8, mew=2)]
    fig.legend(marks, ["estimated from operation counts (45 nm)", "measured (CPU: RAPL; Speck: chip monitor)",
                       "Speck chip + laptop host"], loc="lower right", bbox_to_anchor=(0.99, y), ncol=3,
               frameon=False, fontsize=11)


def pending(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (4, 4))); s.set_color(style.MUTED)
    ax.text(0.5, 0.5, "Jenkins, M1 array:\nruns in progress", ha="center", va="center", fontsize=13,
            color=style.MUTED, transform=ax.transAxes)


def four():
    slide_style.apply()
    fig, axes = plt.subplots(2, 4, figsize=(slide_style.W, 8.0), gridspec_kw=dict(wspace=0.45, hspace=0.42,
                                                                               height_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.17)
    for c, exp in enumerate(EXPS):
        t = table(exp)
        if t is None:
            pending(axes[0, c]); pending(axes[1, c])
        else:
            bubbles(axes[0, c], exp, t, 10)
            energy(axes[1, c], exp, t, 10)
        axes[0, c].set_title(TITLE[exp], loc="left", fontsize=15, pad=8)
    legend_sizes(fig, 0.0)
    slide_style.save(fig, "tradeoff_four")


def one(exp):
    t = table(exp)
    if t is None:
        return
    slide_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(slide_style.W, 6.2), gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1]))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.25)
    bubbles(axes[0], exp, t, 12)
    energy(axes[1], exp, t, 12)
    axes[0].set_title("Fidelity against time per output; dot area = weight memory", loc="left", fontsize=14,
                      color=style.MUTED)
    axes[1].set_title("Energy per output: estimated (hollow) and measured (filled)", loc="left", fontsize=14,
                      color=style.MUTED)
    legend_sizes(fig, 0.0)
    slide_style.save(fig, "tradeoff_" + exp)


if __name__ == "__main__":
    four()
    for exp in EXPS:
        one(exp)
    for exp in EXPS:
        t = table(exp)
        if t is not None:
            print(exp); print(t.assign(mem=t.mem.map(fmt_mem)).round(4).to_string(index=False))
