"""
Shared panels for the calibration figures (fig_calibration_overview.py, fig_calibration_results.py): the event-by-event
mechanism, the layer-1 threshold sweeps, layer firing, per-neuron rates, the decoded example sentence, the refit gains,
and the readout icons. Data: data/calibration_speech.{npz,json} (calibration_data.py), final_results/l1_final (the
threshold sweeps of the deployed networks), speck_r19/results (speech V2 sweep and refit).
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import style

HERE = Path(__file__).parent
DL = WORKSPACE
L1 = DL / "final_results/l1_final"
R19 = DL / "speck_r19/results"
R20 = DL / "speck_r20/results"          # round 20 V2+C, the final speech network
C_MOTOR, C_AUDIO, C_SPEECH = style.COLOR["speck"], style.COLOR["lstm"], style.COLOR["lsnn"]
INK, MUTED, FAINT = style.INK, style.MUTED, "#9a9ea8"
STEP1, STEP2 = "#6a5acd", "#2a7f62"


def tag(ax, text, color, x=0.0, y=1.17):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=5.6, color="white", fontweight="bold", va="bottom",
            ha="left", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=color, ec="none"))


# ------------------------------------------------------------------------------------------------ a: the mechanism
EVENTS = [0.6, 0.6, 0.5, -0.6, -0.5]


def panel_mechanism(ax_t, ax_c):
    """One neuron, one step of five input events (threshold 1): training sums them (0.6, no spike); the chip adds
    them one at a time, fires at the second (1.2), resets to 0 and loses the 0.2 above threshold."""
    t = np.arange(1, len(EVENTS) + 1)
    for ax, title in ((ax_t, "Training: a step's input is summed, then compared"),
                      (ax_c, "Speck2f: each input event lands on its own")):
        ax.set_xlim(0.3, 6.6)
        ax.set_ylim(-1.15, 1.85)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.axhline(0.0, color="#e3e6ec", lw=0.6, zorder=0)
        ax.text(6.55, 1.06, "threshold", fontsize=4.8, color=MUTED, ha="right", va="bottom")
        for x, e in zip(t, EVENTS):
            ax.annotate("", xy=(x, -0.62 if e > 0 else -1.1), xytext=(x, -1.1 if e > 0 else -0.62),
                        arrowprops=dict(arrowstyle="-|>", color="#2a7f62" if e > 0 else "#b8423b", lw=0.7,
                                        mutation_scale=4.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_visible(False)
        ax.set_title(title, loc="left", fontsize=6.0, pad=2)
    # training: nothing until the step is summed
    ax_t.step([0.5, 5.6, 5.6, 6.4], [0, 0, sum(EVENTS), sum(EVENTS)], where="post", color=INK, lw=1.0)
    ax_t.text(5.75, sum(EVENTS) - 0.12, f"sum {sum(EVENTS):.1f}: no spike", fontsize=4.9, ha="left", va="top")
    # chip: event by event
    v, xs, ys = 0.0, [0.5], [0.0]
    for x, e in zip(t, EVENTS):
        xs.append(x)
        ys.append(v)
        v += e
        xs.append(x)
        ys.append(v)
        if v >= 1.0:
            ax_c.plot([x, x], [v + 0.08, v + 0.5], color=C_SPEECH, lw=1.5)
            ax_c.text(x + 0.15, v + 0.42, "a spike training never makes", fontsize=4.9, color=C_SPEECH,
                      va="center")
            ax_c.text(x + 0.15, v + 0.12, f"reset to 0: the {v - 1:.1f} above threshold is lost", fontsize=4.6,
                      color=MUTED, va="center")
            xs.append(x)
            ys.append(0.0)
            v = 0.0
    xs.append(6.4)
    ys.append(v)
    ax_c.plot(xs, ys, color=INK, lw=1.0)
    ax_c.text(0.0, -0.06, "input events below each trace: ▲ excitatory  ▼ inhibitory", transform=ax_c.transAxes,
              fontsize=4.6, color=MUTED, va="top")


# ------------------------------------------------------------------------------------------------ b, c: the sweep
def sweeps():
    out = []
    for f in sorted(L1.glob("jenkins_m1_*.json")):
        d = json.load(open(f))
        r = {float(k): v for k, v in d["results"].items() if k.replace(".", "", 1).isdigit()}
        out.append(("Jenkins", C_MOTOR, "-", float(d["chosen_scale"]),
                    {k: v["agreement_r"] for k, v in r.items()}, {k: d["rmse_trained"] / v["rmse"] for k, v in r.items()}))
    for f in sorted((DL / "final_results/audio_novel").glob("*_P5_*/sweep.json")):   # final audio: N and G, 3 seeds
        d = json.load(open(f))
        r = {float(k): v for k, v in d["results"].items()}
        out.append(("audio", C_AUDIO, "-", float(d["chosen_scale"]),
                    {k: v["agreement_r"] for k, v in r.items()}, {k: v["r"] / d["r_float"] for k, v in r.items()}))
    ev = json.load(open(R20 / "l1/00_V2C_1chip_3chains_events.json"))["results"]
    sw = json.load(open(R20 / "l1/00_V2C_1chip_3chains_torch.json"))["results"]["1"]["per"]
    r = {float(k): v for k, v in ev.items()}
    agree = {k: v["logit_corr_vs_trained"] for k, v in r.items()}
    out.append(("speech", C_SPEECH, "-", max(agree, key=agree.get), agree,
                {k: (1 - v["per"]) / (1 - sw) for k, v in r.items()}))
    return out


def panel_sweep(ax, which):
    for name, c, ls, chosen, agree, acc in sweeps():
        d = agree if which == "agree" else acc
        ks = sorted(d)
        ax.plot(ks, [d[k] for k in ks], ls=ls, color=c, lw=1.0 if name == "speech" else 0.8,
                alpha=1.0 if name in ("speech",) else 0.8)
        ax.plot(chosen, d[chosen], "o", color=c, ms=3.4, mec="white", mew=0.4, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("layer 1 threshold factor")
    if which == "agree":
        ax.set_ylim(0.55, 1.01)
        ax.set_ylabel("agreement with the trained network (r)")
        ax.set_title("Chosen without labels", loc="left")
        for lab, c, ls in (("Jenkins, 4 sessions", C_MOTOR, "-"), ("audio, 2 monkeys × 3 seeds", C_AUDIO, "-"),
                           ("speech V2+C", C_SPEECH, "-")):
            ax.plot([], [], ls=ls, color=c, label=lab)
        ax.plot([], [], "o", color=INK, ms=3, label="label-free choice")
        ax.legend(loc="lower right", frameon=False, fontsize=5.3, handlelength=1.6, borderaxespad=0.1)
    else:
        ax.set_ylim(0.3, 1.08)
        ax.axhline(1, color=FAINT, lw=0.5, ls=":")
        ax.set_ylabel("accuracy, share of software")
        ax.set_title("Checked against labels", loc="left")


# ------------------------------------------------------------------------------------------------ d: layers
def panel_layers(ax, meta):
    L = meta["layers"]
    x = np.arange(3)
    w = 0.26
    for j, (k, lab, c) in enumerate((("trained", "trained network", FAINT), ("x1", "chip, ×1", "#c9a5a2"),
                                     ("x2.5", "chip, ×2.5 (chosen)", C_SPEECH))):
        ax.bar(x + (j - 1) * w, L[k], w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["layer 1", "layer 2", "layer 3"])
    ax.set_ylabel("spikes per neuron\nper 20 ms bin")
    ax.set_title("What the factor does (V2)", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=5.3, handlelength=1.0)
    ax.text(0, L["x1"][0] + 0.004, "surge", fontsize=5.0, color="#9c5f5a", ha="center", va="bottom")


# ------------------------------------------------------------------------------------------------ e: the recipe
def panel_recipe(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    steps = [("1", "Train", "in PyTorch,\nas usual", "white", INK),
             ("2", "Freeze", "the chip network,\nwith its layer-1\nfactor", "#eeedf9", STEP1),
             ("3", "Simulate", "the chip, event by\nevent, on training\ndata; keep spikes", "#eeedf9", STEP1),
             ("4", "Refit", "only the host\nreadout, with the\ntraining labels", "#e6f3ee", STEP2),
             ("5", "Deploy", "the same chip,\nread by the\nrefitted readout", "#e6f3ee", STEP2)]
    w, gap = 17.6, 2.9
    for i, (n, head, body, fc, ec) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 9), w, 18, boxstyle="round,pad=0,rounding_size=2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 2.0, 24.6, n, fontsize=8, fontweight="bold", color=ec, va="top")
        ax.text(x + 6.2, 24.3, head, fontsize=6.4, fontweight="bold", color=INK, va="top")
        ax.text(x + 2.0, 17.4, body, fontsize=5.2, color=INK, va="top", linespacing=1.3)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.3, 18), (x - 0.3, 18), arrowstyle="-|>,head_length=2.2,"
                                         "head_width=1.4", color=FAINT, lw=0.8))
    ax.text(0, 4.0, "No test data anywhere; the chip's weights, thresholds and cores are never changed.",
            fontsize=5.3, color=MUTED, style="italic")


# ------------------------------------------------------------------------------------------------ f: neurons
def panel_neurons(ax, z):
    a, b, lay = z["rate_trained"], z["rate_chip"], z["neuron_layer"]
    lo = 1e-4
    for l, c in zip(range(3), ("#f2b38d", "#e07a45", "#a33f12")):
        m = lay == l
        ax.scatter(np.maximum(a[m], lo), np.maximum(b[m], lo), s=2.2, color=c, lw=0, alpha=0.7,
                   label=f"layer {l + 1}")
    ax.plot([lo, 3], [lo, 3], color=FAINT, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 3)
    ax.set_ylim(lo, 3)
    ax.set_xlabel("trained network (spikes/bin)")
    ax.set_ylabel("chip model, ×2.5")
    ok = (a > lo) & (b > lo)
    r = np.corrcoef(np.log(a[ok]), np.log(b[ok]))[0, 1]
    ax.text(0.04, 0.96, f"each dot a neuron\nlog-rates r = {r:.2f}", transform=ax.transAxes, fontsize=5.2,
            va="top", color=INK)
    ax.set_title("A consistent distortion", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=5.0, markerscale=2.5, handletextpad=0.2)


# ------------------------------------------------------------------------------------------------ g: example
def panel_example(axes, z, meta):
    ex = meta["example"]
    rows = (("ex_software", "software", "trained network, in software"),
            ("ex_chip", "chip", "real chip, trained readout"),
            ("ex_refit", "refit", "real chip, refitted readout"))
    for ax, (k, key, lab) in zip(axes, rows):
        p = z[k][:, :40].T                                  # phonemes only (blank dropped)
        ax.imshow(p, aspect="auto", cmap="Oranges", vmin=0, vmax=0.35, interpolation="nearest", origin="lower")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#d8dbe2")
            sp.set_linewidth(0.5)
        ax.text(0, 1.06, lab, transform=ax.transAxes, fontsize=5.6, fontweight="bold", va="bottom")
        ax.text(1, 1.06, f"PER {ex['per'][key]:.2f}", transform=ax.transAxes, fontsize=5.6, va="bottom", ha="right",
                color=C_SPEECH if key != "chip" else MUTED, fontweight="bold")
        dec = " ".join(ex["decoded"][key].split()).replace(" SIL", " ·").replace("SIL ", "· ").strip(" ·")
        ax.text(0, -0.06, dec, transform=ax.transAxes, fontsize=4.9, va="top", color=INK, family="monospace")
    truth = " ".join(ex["truth"].split()).replace(" SIL", " ·").strip(" ·")
    axes[0].text(0, 1.36, "truth  " + truth, transform=axes[0].transAxes, fontsize=4.9, va="bottom",
                 family="monospace", color=MUTED)
    axes[0].set_title("One test sentence (V2), three ways", loc="left", pad=17)


# ------------------------------------------------------------------------------------------------ h: gains
def panel_gains(ax):
    W1 = json.load(open(R19 / "W1/refit.json"))
    rows = [
        ("speech V2+C\n(real chip, 680 sentences)", (1 - 0.4952) / (1 - 0.4435), (1 - 0.4396) / (1 - 0.4435), C_SPEECH, True),
        # 2026-11-18 expanded cohorts (collect_audio_novel.py): mean within-word spectral match, 3 seeds, real chip
        ("audio, monkey N\n(real chip, 44 words)", 0.5977 / 0.6436, 0.6280 / 0.6436, C_AUDIO, True),
        ("audio, monkey G\n(real chip, 35 words)", 0.5997 / 0.6257, 0.6401 / 0.6257, C_AUDIO, True),
        ("Jenkins M1\n(real chip, 4 sessions)", 52.79 / 70.68, 52.79 / 66.65, C_MOTOR, True),
    ]
    for i, (lab, a, b, c, real) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.annotate("", xy=(100 * b, y), xytext=(100 * a, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, mutation_scale=7))
        ax.plot(100 * a, y, "o", ms=4, mfc="white", mec=c, mew=0.9)
        ax.plot(100 * b, y, "o", ms=4.2, color=c, mec="white", mew=0.4)
        ax.text(71, y + 0.2, lab.replace("\n", " "), fontsize=5.3, color=INK, va="bottom")
        ax.text(100 * b + 1.2, y, f"{100 * a:.0f}→{100 * b:.0f}%", fontsize=5.3, color=c, va="center")
    ax.set_yticks([])
    ax.axvline(100, color=FAINT, lw=0.6, ls=":")
    ax.set_xlim(70, 112)
    ax.set_ylim(-0.5, len(rows) - 0.25)
    ax.set_xlabel("chip accuracy, share of software (%)")
    ax.grid(axis="x")
    ax.set_title("What it gains", loc="left")
    ax.text(0.0, -0.3, "open: trained readout · filled: refitted.\nThe same refit on software spikes gains\n"
            f"speech only {100 * ((1 - 0.4779) / (1 - 0.4872) - 1):.0f}%: the gain is the chip's.",
            transform=ax.transAxes, fontsize=4.8, color=MUTED, va="top")


def weights_icon(ax, x, y, w, h, seed=0, color="#2a7f62"):
    rng = np.random.default_rng(seed)
    n, m = 7, 5
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x + j * w / m, y + i * h / n), w / m * 0.86, h / n * 0.8, fc=color, ec="none",
                                   alpha=0.25 + 0.7 * rng.random(), zorder=6))


def scores_icon(ax, x, y, w, h, seed=0, color="#2a7f62"):
    rng = np.random.default_rng(seed)
    n, m = 8, 12
    hot = rng.integers(0, n, m)
    for j in range(m):
        for i in range(n):
            ax.add_patch(Rectangle((x + j * w / m, y + i * h / n), w / m * 0.9, h / n * 0.85,
                                   fc=color if i == hot[j] else "#e9edf0", ec="none", zorder=6))


