"""
plot_decoder_comparison.py

Every decoder on every dataset in one figure, from the pipeline's own <decoder>_regression_metrics.csv files (mean
correlation over the 128 mel bands, Fisher-averaged, as regression_metrics computes it):

    a   test correlation, every dataset x every decoder (the best decoder of each dataset outlined)
        (b and c abbreviate: KF Kalman filter, WF Wiener filter, WC Wiener cascade, DNN dense NN, RNN simple RNN)
    b   mean test correlation per decoder over the single days and over the multi-day datasets
    c   (with --pretrained-dir) the five-word verified dataset with and without all-words pretraining

Test is the fair split to compare on: every neural decoder stops early on validation, and the LSNN's size and
regularization were chosen on 07_19_11's validation split.

Usage:
    python3 plot_decoder_comparison.py --subject Neil --region aSTG \
        [--pretrained-dir /path/to/pretrain_all/data] [--output pipeline_data/aggregate/analysis/decoder_comparison_all.png]
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DECODERS = [("kalman_filter", "Kalman\nfilter"), ("wiener_filter", "Wiener\nfilter"),
            ("wiener_cascade", "Wiener\ncascade"), ("dense_nn", "Dense\nNN"), ("simple_rnn", "Simple\nRNN"),
            ("gru_rnn", "GRU"), ("lstm_rnn", "LSTM"), ("lsnn", "LSNN")]
DAYS = [("07_19_11", "19 Jul 2011"), ("07_20_10", "20 Jul 2010"), ("07_31_13", "31 Jul 2013"),
        ("08_03_10", "3 Aug 2010"), ("08_06_13", "6 Aug 2013")]
MULTI = [("COMBINED_naive", "Combined, naive"),
         ("COMBINED_per_day_normalized_causal_clip5.0", "Combined, per-day normalized"),
         ("FIVEWORD_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13", "Five words"),
         ("FIVEWORD_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13", "Five words, verified"),
         ("ALLWORDS_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13", "All words, verified")]
TAG = {"kalman_filter": "raw", "wiener_filter": "flat_window", "wiener_cascade": "flat_window",
       "dense_nn": "flat_window", "simple_rnn": "window", "gru_rnn": "window", "lstm_rnn": "window", "lsnn": "window"}
SHORT = ["KF", "WF", "WC", "DNN", "RNN", "GRU", "LSTM", "LSNN"]      # panels b and c
RED = "#C0392B"


def rho(targets, predict):
    """regression_metrics' mean correlation: per band, Fisher-averaged."""
    r = np.array([np.corrcoef(targets[:, i], predict[:, i])[0, 1] for i in range(targets.shape[1])])
    return float(np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.9999999, 0.9999999)))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="Neil")
    p.add_argument("--region", default="aSTG")
    p.add_argument("--data-root", default="pipeline_data")
    p.add_argument("--pretrained-dir", default=None,
                   help="directory with out/<decoder>_<split>.npy predictions of the all-words-pretrained decoders "
                        "on the five-word verified dataset")
    p.add_argument("--output", default="pipeline_data/aggregate/analysis/decoder_comparison_all.png")
    a = p.parse_args()
    base = Path(a.data_root) / a.subject / a.region

    def score(date, dec, split):
        f = base / date / "results" / f"{dec}_regression_metrics.csv"
        return pd.read_csv(f)[f"{split}_mean_rho"][0] if f.is_file() else np.nan

    rows = DAYS + MULTI
    test = np.array([[score(d, k, "test") for k, _ in DECODERS] for d, _ in rows])
    valid = np.array([[score(d, k, "valid") for k, _ in DECODERS] for d, _ in rows])

    plt.rcParams.update({"font.family": "Palatino", "font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    pre = a.pretrained_dir and os.path.isdir(a.pretrained_dir)
    fig = plt.figure(figsize=(19, 7.6))
    gs = fig.add_gridspec(1, 3 if pre else 2, width_ratios=[1.55, 1, 1] if pre else [1.55, 1], wspace=0.32)

    # a: test correlation heatmap
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(test, cmap="Greys", vmin=0, vmax=0.9, aspect="auto")
    for i in range(len(rows)):
        best = np.nanargmax(test[i])
        for j in range(len(DECODERS)):
            v = test[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if v > 0.5 else "black", fontweight="bold" if j == best else "normal")
        ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.2))
    ax.axhline(len(DAYS) - 0.5, color="black", lw=1.5)
    ax.set_xticks(range(len(DECODERS)))
    ax.set_xticklabels([l for _, l in DECODERS], fontsize=12)
    ax.get_xticklabels()[-1].set_color(RED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for _, l in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("a   Test correlation, every dataset (best outlined)", loc="left", fontsize=14, fontweight="bold")

    # b: mean over single days and over multi-day datasets
    ax = fig.add_subplot(gs[1])
    x = np.arange(len(DECODERS))
    n = len(DAYS)
    for off, part, label, face in ((-0.2, test[:n], "single days (5)", "0.55"), (0.2, test[n:], "multi-day (5)", "white")):
        m = np.nanmean(part, 0)
        ax.bar(x + off, m, 0.38, color=face, edgecolor="black", label=label)
        for xi, v in zip(x + off, m):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=10, rotation=90)
    ax.patches[len(DECODERS) - 1].set_facecolor(RED)
    ax.patches[2 * len(DECODERS) - 1].set_edgecolor(RED)
    ax.patches[2 * len(DECODERS) - 1].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=12, rotation=45, ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[-1].set_color(RED)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("mean test correlation")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.set_title("b   Single days vs multi-day datasets", loc="left", fontsize=14, fontweight="bold")

    # c: five words verified, without and with all-words pretraining
    if pre:
        five = MULTI[3][0]
        P = base / five / "pipeline"
        orig, pret = [], []
        for k, _ in DECODERS:
            orig.append(score(five, k, "test"))
            f = Path(a.pretrained_dir) / "out" / f"{k}_test.npy"
            pret.append(rho(np.load(P / f"{TAG[k]}_targets_test.npy"), np.load(f)) if f.is_file() else np.nan)
        ax = fig.add_subplot(gs[2])
        ax.bar(x - 0.2, orig, 0.38, color="white", edgecolor="black", label="five-word data only")
        ax.bar(x + 0.2, pret, 0.38, color="0.35", edgecolor="black", label="pretrained on all words")
        ax.patches[2 * len(DECODERS) - 1].set_facecolor(RED)
        for xi, v in zip(x + 0.2, pret):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=10, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, fontsize=12, rotation=45, ha="right", rotation_mode="anchor")
        ax.get_xticklabels()[-1].set_color(RED)
        ax.set_ylim(0, 0.8)
        ax.set_ylabel("test correlation")
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        ax.set_title("c   Five words: all-words pretraining", loc="left", fontsize=14, fontweight="bold")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Wrote {out.with_suffix('.png')} and .svg")
    print("mean test, single days / multi-day:")
    for j, (k, _) in enumerate(DECODERS):
        print(f"  {k:15s} {np.nanmean(test[:n, j]):.3f} / {np.nanmean(test[n:, j]):.3f}   "
              f"(validation {np.nanmean(valid[:n, j]):.3f} / {np.nanmean(valid[n:, j]):.3f})")


if __name__ == "__main__":
    main()
