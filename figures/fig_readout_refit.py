"""
Explanatory figure (talk): how the Speck SNN's host readout is refitted to the chip, step by step, with the rules that
keep it honest and what it gained on the real chip.
    python3 fig_readout_refit.py      -> figures/fig_readout_refit.{png,pdf}
Steps (chip/audio_refit.py, chip/jenkins_refit.py, speech/scripts/refit_readout.py):
  1 train in software; 2 freeze the chip's weights, choose layer 1's factor without labels; 3 run the TRAINING data
  through the chip model (event-by-event firing) and keep the spikes the readout reads; 4 refit only the readout to
  the true targets (same smoothing; ridge for audio and monkey J, CTC for speech), tuned on held-out training data;
  5 deploy: the chip runs its trained weights, the host applies the new readout live, test scored once.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
INK, MUTED, ACCENT, CHIP, HOST, OK = "#1F2430", "#5B6270", "#B4500E", "#ECE8F7", "#F1EEE6", "#2F6F4E"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def box(ax, x, y, w, h, fc, ec=None, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc,
                                ec=ec or fc, lw=lw, ls=ls, transform=ax.transAxes))


def arrow(ax, x0, y0, x1, y1, c=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, color=c, lw=2.4,
                                 transform=ax.transAxes))


def main():
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.03, 0.95, "Refitting the readout to the chip: five steps, training data only", fontsize=22,
            fontweight="bold", color=INK, va="top")
    ax.text(0.03, 0.905, "The chip’s weights never change. Only the readout on the host, which turns the chip’s output spikes "
            "into the decoded quantity, is refitted.", fontsize=13.5, color=MUTED, va="top")

    steps = [
        ("1", "Train in software", "The network and its readout are\ntrained together, each step’s\ninput summed before firing.",
         HOST, None),
        ("2", "Freeze the chip’s weights", "Copy them to the chip as they are.\nRaise layer 1’s threshold by a\nfactor "
         "chosen without labels.", CHIP, None),
        ("3", "Record the chip’s spikes", "Run the TRAINING data\nthrough the chip model (each\nevent fires on arrival). Keep "
         "the\nspikes the readout would see.", CHIP, None),
        ("4", "Refit only the readout", "A new map from the smoothed\nspikes to the true targets\n(speech: its time "
         "constants too),\ntuned on held-out training data.", HOST, ACCENT),
        ("5", "Deploy and score once", "The chip runs its trained weights;\nthe host applies the new readout\nlive. "
         "Test data scored once.", HOST, None),
    ]
    x0, w, gap, y, h = 0.03, 0.172, 0.022, 0.56, 0.28
    for i, (n, title, body, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, fc, ec, lw=2.2 if ec else 1.0)
        ax.text(x + 0.012, y + h - 0.018, n, fontsize=26, fontweight="bold", color=ec or MUTED, va="top")
        ax.text(x + 0.012, y + h - 0.085, title, fontsize=14, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.012, y + h - 0.13, body, fontsize=12, color=INK, va="top", linespacing=1.45)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.002, y + h / 2, x + w + gap - 0.002, y + h / 2)
    ax.text(x0 + 2 * (w + gap) + w / 2, y - 0.018, "on the chip model: software that\nreproduces the chip’s firing",
            fontsize=10.5, color=MUTED, ha="center", va="top", style="italic")
    ax.text(x0 + (w + gap) + w + gap / 2, y + h + 0.012, "chip side", fontsize=11, color="#6B5BB0", ha="center")

    # how step 4 is done in each task
    ty = 0.45
    ax.text(0.03, ty, "Step 4, per task", fontsize=14, fontweight="bold", color=INK, va="top")
    rows = [("Audio", "128 bands", "ridge regression", "penalty chosen on the validation utterances"),
            ("Motor (II)", "velocity", "ridge regression", "penalty and lags by cross-validation across training trials"),
            ("Speech", "41 phoneme classes", "CTC, Adam", "early stopping on every 10th training sentence"),
            ("Motor (I)", "velocity", "not refitted", "a refit did not improve it; the trained readout is kept")]
    cols = [0.03, 0.135, 0.255, 0.385]
    for j, head in enumerate(["task", "maps to", "fitted by", "tuned on"]):
        ax.text(cols[j], ty - 0.05, head, fontsize=11.5, color=MUTED, va="top")
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            ax.text(cols[j], ty - 0.09 - i * 0.045, v, fontsize=12.2, color=INK, va="top",
                    fontweight="bold" if j == 0 else "normal")

    # guardrails
    gx = 0.66
    box(ax, gx, 0.06, 0.31, 0.40, "#F6F4EE", "#D8D3C6")
    ax.text(gx + 0.015, 0.445, "What keeps it honest", fontsize=14, fontweight="bold", color=INK, va="top")
    rules = ["Never sees test data: fitted and tuned\non training data only.",
             "Never changes the chip: same weights;\nonly the host readout is refitted.",
             "Fitted to the chip model, which matches\nthe chip (audio r 0.652 vs 0.636).",
             "Control: the same refit on the software\nnetwork’s spikes gains speech only 2%."]
    for i, t in enumerate(rules):
        ax.text(gx + 0.015, 0.39 - i * 0.082, "✓", fontsize=14, color=OK, va="top", fontweight="bold",
                fontfamily="DejaVu Sans")
        ax.text(gx + 0.037, 0.39 - i * 0.082, t, fontsize=11.8, color=INK, va="top", linespacing=1.35)

    # result strip
    ax.text(0.03, 0.15, "On the real chip, as a share of the software network’s score (trained readout → refitted):",
            fontsize=13, color=INK, va="top", fontfamily="DejaVu Sans")
    res = [("speech", "91% → 101%"), ("audio N", "97% → 100%"), ("audio G", "97% → 101%"), ("monkey J", "81% → 82%")]
    for i, (k, v) in enumerate(res):
        ax.text(0.03 + i * 0.155, 0.10, k, fontsize=11.5, color=MUTED, va="top")
        ax.text(0.03 + i * 0.155, 0.07, v, fontsize=15, color=ACCENT, va="top", fontweight="bold", fontfamily="DejaVu Sans")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "figures" / f"fig_readout_refit.{ext}", facecolor="#FBFBF8")
    print("wrote figures/fig_readout_refit.{png,pdf}")


if __name__ == "__main__":
    main()
