"""
analyze_results.py

Two modes:

  --mode single (default): analyze one (subject, region, date)'s decoder
      results -- reads <results>/*_regression_metrics.csv, produces
      comparison plots plus a combined labeled (A/B/C/D) dashboard, all
      saved under <results>/analysis/.

  --mode aggregate: scans every (subject, region, date) under --data-root
      that has completed results, and produces cross-date comparison
      plots (per-decoder-across-dates, individually and combined) plus a
      labeled aggregate dashboard, saved under <data_root>/aggregate/analysis/.

Usage:
    python3 analyze_results.py --mode single --subject Neil --region aSTG --date 07_19_11
    python3 analyze_results.py --mode aggregate --data-root pipeline_data
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_paths import PipelinePaths, discover_all_runs, DEFAULT_DATA_ROOT

DECODER_SPLIT_TAG = {
    "wiener_filter": "flat_window", "wiener_cascade": "flat_window", "dense_nn": "flat_window",
    "kalman_filter": "raw", "simple_rnn": "window", "gru_rnn": "window", "lstm_rnn": "window",
    "lsnn": "window",
}

PANEL_LABELS = "ABCDEFGH"


def _label_panel(ax, letter):
    """Standard scientific-figure panel labeling: bold letter in the
    top-left corner, outside the axes."""
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")


# ---------------------------------------------------------------------------
# Single-date mode
# ---------------------------------------------------------------------------
def load_metrics_for_date(paths):
    rows = []
    for name in DECODER_SPLIT_TAG:
        csv_path = paths.results / f"{name}_regression_metrics.csv"
        if not csv_path.exists():
            print(f"  (skipping {name} -- not found)")
            continue
        df = pd.read_csv(csv_path)
        df["decoder"] = name
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No regression_metrics.csv files found under {paths.results}. "
                            f"Run run_auditory_experiment.py for this date first.")
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(paths.analysis / "all_decoders_metrics.csv", index=False)
    return combined


def plot_bar_comparison(ax, df):
    df_sorted = df.sort_values("valid_mean_rho", ascending=False)
    x = np.arange(len(df_sorted))
    width = 0.25
    ax.bar(x - width, df_sorted["train_mean_rho"], width, label="Train")
    ax.bar(x, df_sorted["valid_mean_rho"], width, label="Valid")
    ax.bar(x + width, df_sorted["test_mean_rho"], width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["decoder"], rotation=30, ha="right")
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_title("Decoder comparison")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8)


def plot_train_vs_valid(ax, df):
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="Overfitting line")
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for (_, row), color in zip(df.iterrows(), colors):
        ax.scatter(row["train_mean_rho"], row["valid_mean_rho"], s=100, color=color, edgecolor="black", zorder=5)
        ax.annotate(row["decoder"], (row["train_mean_rho"], row["valid_mean_rho"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean training correlation")
    ax.set_ylabel("Mean validation correlation")
    ax.set_title("Train vs. valid (single run per algorithm)")
    ax.legend(fontsize=8)


def plot_per_band_correlation(ax, df):
    for _, row in df.iterrows():
        rho_list = json.loads(row["valid_mean_rho_list"])
        ax.plot(rho_list, label=row["decoder"], alpha=0.8)
    ax.set_xlabel("Mel band")
    ax.set_ylabel("Validation correlation")
    ax.set_title("Per-band validation correlation")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.8)


def plot_spectrogram_comparison_panel(ax_target, ax_pred, ax_diff, paths, decoder_name, n_frames=500):
    tag = DECODER_SPLIT_TAG[decoder_name]
    target_path = paths.pipeline / f"{tag}_targets_valid.npy"
    predict_path = paths.pipeline / f"{decoder_name}_predict_valid.npy"
    if not (target_path.exists() and predict_path.exists()):
        ax_target.text(0.5, 0.5, "missing data", ha="center")
        return
    target = np.load(target_path)
    predict = np.load(predict_path)
    n = min(n_frames, target.shape[0], predict.shape[0])

    ax_target.imshow(target[:n].T, aspect="auto", origin="lower", cmap="magma")
    ax_target.set_title("Target mel-spectrogram")
    ax_target.set_ylabel("Mel band")
    ax_target.set_xlabel("Time frame")

    ax_pred.imshow(predict[:n].T, aspect="auto", origin="lower", cmap="magma")
    ax_pred.set_title(f"{decoder_name} prediction")
    ax_pred.set_xlabel("Time frame")

    diff = predict[:n].T - target[:n].T
    vmax = np.abs(diff).max() if diff.size else 1.0
    im = ax_diff.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_diff.set_title("Difference (prediction - target)")
    ax_diff.set_xlabel("Time frame")
    plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)


def build_single_date_dashboard(paths, df):
    """One combined, labeled (A/B/C/D/E/F) figure summarizing this date's
    decoder comparison -- the single glance-able artifact for this run."""
    best = df.sort_values("valid_mean_rho", ascending=False).iloc[0]["decoder"]

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_bar_comparison(ax_a, df)
    _label_panel(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_train_vs_valid(ax_b, df)
    _label_panel(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_per_band_correlation(ax_c, df)
    _label_panel(ax_c, "C")

    # D and E: equal-sized squares side by side, same column slots as A/B.
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    # F: difference heatmap, lower-left (directly below D).
    ax_f = fig.add_subplot(gs[2, 0])

    plot_spectrogram_comparison_panel(ax_d, ax_e, ax_f, paths, best)
    _label_panel(ax_d, "D")
    _label_panel(ax_e, "E")
    _label_panel(ax_f, "F")

    fig.suptitle(f"{paths.label()} -- best decoder: {best}", fontsize=14)
    out_path = paths.analysis / "dashboard.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_single_mode(paths):
    print(f"Loading metrics for {paths.label()}...")
    df = load_metrics_for_date(paths)
    print(f"\nFound results for: {', '.join(df['decoder'])}\n")
    print(df[["decoder", "train_mean_rho", "valid_mean_rho", "test_mean_rho"]]
          .sort_values("valid_mean_rho", ascending=False).to_string(index=False))

    build_single_date_dashboard(paths, df)

    wavs = sorted(paths.results.glob("*_reconstructed.wav"))
    print("\nReconstructed audio for listening comparison:")
    for w in wavs:
        print(f"  {w}")


# ---------------------------------------------------------------------------
# Aggregate mode -- across every (subject, region, date) found
# ---------------------------------------------------------------------------
def load_all_dates(data_root, subject_filter=None, region_filter=None):
    runs = discover_all_runs(data_root)
    if subject_filter:
        runs = [r for r in runs if r.subject == subject_filter]
    if region_filter:
        runs = [r for r in runs if r.region == region_filter]
    if not runs:
        raise RuntimeError(f"No completed runs found under {data_root} "
                            f"(subject={subject_filter}, region={region_filter}).")

    rows = []
    for paths in runs:
        for name in DECODER_SPLIT_TAG:
            csv_path = paths.results / f"{name}_regression_metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["decoder"] = name
            df["subject"] = paths.subject
            df["region"] = paths.region
            df["date"] = paths.date
            df["run_label"] = paths.label()
            rows.append(df)

    if not rows:
        raise RuntimeError(f"Found run directories under {data_root} but no regression_metrics.csv files in them.")
    return pd.concat(rows, ignore_index=True)


def plot_decoder_across_dates_individual(ax, df, decoder, dates):
    sub = df[df["decoder"] == decoder].set_index("run_label").reindex(dates)
    x = np.arange(len(dates))
    ax.bar(x - 0.2, sub["train_mean_rho"], 0.2, label="Train")
    ax.bar(x, sub["valid_mean_rho"], 0.2, label="Valid")
    ax.bar(x + 0.2, sub["test_mean_rho"], 0.2, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.set_title(decoder)
    ax.axhline(0, color="gray", linewidth=0.6)


def plot_all_decoders_across_dates_grid(df, out_path):
    dates = sorted(df["run_label"].unique())
    decoders_present = [d for d in DECODER_SPLIT_TAG if d in df["decoder"].unique()]
    n = len(decoders_present)
    ncols = min(3, n)
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for i, decoder in enumerate(decoders_present):
        ax = axes[i // ncols][i % ncols]
        plot_decoder_across_dates_individual(ax, df, decoder, dates)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Each decoder's performance across dates", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_overlay(ax, df):
    dates = sorted(df["run_label"].unique())
    for decoder in DECODER_SPLIT_TAG:
        sub = df[df["decoder"] == decoder].set_index("run_label").reindex(dates)
        ax.plot(dates, sub["valid_mean_rho"], marker="o", label=decoder)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean validation correlation")
    ax.set_title("All decoders, overlaid, across dates")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.6)


def plot_decoder_distribution_boxplot(ax, df):
    decoders_present = [d for d in DECODER_SPLIT_TAG if d in df["decoder"].unique()]
    data = [df[df["decoder"] == d]["valid_mean_rho"].dropna().values for d in decoders_present]
    ax.boxplot(data, labels=decoders_present)
    ax.set_xticklabels(decoders_present, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean validation correlation")
    ax.set_title("Validation correlation distribution across dates")
    ax.axhline(0, color="gray", linewidth=0.6)


def plot_train_vs_valid_all_runs(ax, df):
    decoders_present = [d for d in DECODER_SPLIT_TAG if d in df["decoder"].unique()]
    colors = plt.cm.tab10(np.linspace(0, 1, len(decoders_present)))
    for decoder, color in zip(decoders_present, colors):
        sub = df[df["decoder"] == decoder]
        ax.scatter(sub["train_mean_rho"], sub["valid_mean_rho"], color=color, label=decoder, alpha=0.7, s=40)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean training correlation")
    ax.set_ylabel("Mean validation correlation")
    ax.set_title("Train vs. valid, every date + decoder")
    ax.legend(fontsize=7, ncol=2)


def plot_best_decoder_win_counts(ax, df):
    best_per_date = df.loc[df.groupby("run_label")["valid_mean_rho"].idxmax()]
    counts = best_per_date["decoder"].value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("# dates where this decoder had the best valid correlation")
    ax.set_title("Best-decoder win counts across dates")


def build_aggregate_dashboard(df, out_path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_combined_overlay(ax_a, df)
    _label_panel(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_decoder_distribution_boxplot(ax_b, df)
    _label_panel(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    plot_train_vs_valid_all_runs(ax_c, df)
    _label_panel(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_best_decoder_win_counts(ax_d, df)
    _label_panel(ax_d, "D")

    fig.suptitle(f"Cross-date decoder comparison ({df['run_label'].nunique()} dates)", fontsize=14)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_aggregate_mode(data_root, subject_filter, region_filter):
    df = load_all_dates(data_root, subject_filter, region_filter)
    agg_dir = Path(data_root) / "aggregate" / "analysis"
    agg_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(agg_dir / "all_dates_all_decoders_metrics.csv", index=False)
    print(f"Aggregated {df['run_label'].nunique()} date(s), {len(df)} decoder-runs total.\n")
    print(df.groupby("decoder")[["train_mean_rho", "valid_mean_rho", "test_mean_rho"]].mean()
          .sort_values("valid_mean_rho", ascending=False))

    plot_all_decoders_across_dates_grid(df, agg_dir / "per_decoder_across_dates.png")
    build_aggregate_dashboard(df, agg_dir / "aggregate_dashboard.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "aggregate"], default="single")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subject", default=None, help="Required for --mode single; optional filter for --mode aggregate")
    parser.add_argument("--region", default=None, help="Required for --mode single; optional filter for --mode aggregate")
    parser.add_argument("--date", default=None, help="Required for --mode single")
    args = parser.parse_args()

    if args.mode == "single":
        if not (args.subject and args.region and args.date):
            parser.error("--mode single requires --subject, --region, and --date")
        paths = PipelinePaths(args.data_root, args.subject, args.region, args.date)
        run_single_mode(paths)
    else:
        run_aggregate_mode(args.data_root, args.subject, args.region)


if __name__ == "__main__":
    main()
