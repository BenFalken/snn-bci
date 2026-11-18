"""
diagnose_cross_day_heterogeneity.py

Pre-flight diagnostic, run BEFORE combine_days.py. Examines each
successfully-processed day's per-channel neural firing rates and per-band
mel-spectrogram energy, looking for the kind of day-to-day differences that
would explain why naive cross-day concatenation previously collapsed
decoder performance for every decoder simultaneously.

Four panels, saved as one labeled dashboard:
  A. Per-channel mean firing rate, one row per day (heatmap) -- reveals
     channels active on some days and silent on others.
  B. Per-band mean mel energy, one row per day (heatmap) -- reveals
     systematic spectral/loudness differences between days.
  C. "Dead channel" matrix, per day -- a channel alive on one day but dead
     on another means that feature's meaning shifts across a naive
     concatenation.
  D. PCA projection of per-bin feature vectors, colored by day -- if days
     form visually separate clusters, that is direct evidence of a
     day-level domain shift.

Usage:
    python3 diagnose_cross_day_heterogeneity.py --subject Neil --region aSTG
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from pipeline_paths import discover_all_runs, DEFAULT_DATA_ROOT

PANEL_LABELS = "ABCD"


def _label_panel(ax, letter):
    ax.text(-0.1, 1.08, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")


def load_days(subject, region, data_root):
    all_runs = discover_all_runs(data_root)
    day_runs = [
        r for r in all_runs
        if r.subject == subject and r.region == region
        and not r.date.startswith("COMBINED")
    ]
    day_runs.sort(key=lambda r: r.date)
    if not day_runs:
        raise RuntimeError(f"No completed days found for {subject}/{region} under {data_root}")

    days = []
    for r in day_runs:
        features = np.load(r.pipeline / "features.npy")
        targets = np.load(r.pipeline / "targets.npy")
        days.append((r.date, features, targets))
    return days


def compute_firing_rate_matrix(days):
    """Rows = day, columns = channel, values = mean spike count / bin."""
    n_channels = days[0][1].shape[1]
    mat = np.zeros((len(days), n_channels))
    for i, (date, features, _) in enumerate(days):
        mat[i] = features.mean(axis=0)
    return mat


def compute_mel_energy_matrix(days):
    """Rows = day, columns = mel band, values = mean normalized mel dB."""
    n_bands = days[0][2].shape[1]
    mat = np.zeros((len(days), n_bands))
    for i, (date, _, targets) in enumerate(days):
        mat[i] = targets.mean(axis=0)
    return mat


def compute_dead_channel_matrix(days, rel_threshold=0.1):
    """Rows = day, columns = channel. 1 = alive, 0 = dead (std below
    rel_threshold * that day's median channel std -- adaptive per day,
    since overall activity level can differ day to day)."""
    n_channels = days[0][1].shape[1]
    mat = np.zeros((len(days), n_channels))
    for i, (date, features, _) in enumerate(days):
        stds = features.std(axis=0)
        threshold = rel_threshold * np.median(stds)
        mat[i] = (stds >= threshold).astype(float)
    return mat


def plot_pca_by_day(ax, days, n_samples_per_day=500, seed=1337):
    rng = np.random.default_rng(seed)
    all_points, all_day_idx = [], []
    for i, (date, features, _) in enumerate(days):
        n = features.shape[0]
        idx = rng.choice(n, size=min(n_samples_per_day, n), replace=False)
        all_points.append(features[idx])
        all_day_idx.extend([i] * len(idx))

    X = np.concatenate(all_points, axis=0)
    day_idx = np.array(all_day_idx)

    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    proj = X_centered @ Vt[:2].T  # project onto top 2 principal components

    colors = plt.cm.tab10(np.linspace(0, 1, len(days)))
    for i, (date, _, _) in enumerate(days):
        mask = day_idx == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, color=colors[i], label=date)

    explained = (S[:2] ** 2) / (S ** 2).sum()
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(f"Feature-space PCA, colored by day\n({n_samples_per_day} bins/day, subsampled)")
    ax.legend(fontsize=7, markerscale=2, loc="best")


def build_dashboard(days, subject, region, out_path):
    dates = [d[0] for d in days]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    firing_rate_mat = compute_firing_rate_matrix(days)
    im_a = ax_a.imshow(firing_rate_mat, aspect="auto", cmap="viridis")
    ax_a.set_yticks(range(len(dates)))
    ax_a.set_yticklabels(dates, fontsize=8)
    ax_a.set_xlabel("Channel")
    ax_a.set_title("Mean firing rate per channel, per day")
    fig.colorbar(im_a, ax=ax_a, label="Spike count / bin")
    _label_panel(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    mel_energy_mat = compute_mel_energy_matrix(days)
    im_b = ax_b.imshow(mel_energy_mat, aspect="auto", cmap="magma")
    ax_b.set_yticks(range(len(dates)))
    ax_b.set_yticklabels(dates, fontsize=8)
    ax_b.set_xlabel("Mel band")
    ax_b.set_title("Mean mel energy per band, per day")
    fig.colorbar(im_b, ax=ax_b, label="Normalized mel dB")
    _label_panel(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    dead_mat = compute_dead_channel_matrix(days)
    im_c = ax_c.imshow(dead_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_c.set_yticks(range(len(dates)))
    ax_c.set_yticklabels(dates, fontsize=8)
    ax_c.set_xlabel("Channel")
    ax_c.set_title("Channel alive (green) / dead (red), per day")
    _label_panel(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_pca_by_day(ax_d, days)
    _label_panel(ax_d, "D")

    fig.suptitle(f"Cross-day heterogeneity diagnostics -- {subject}/{region} ({len(days)} days)", fontsize=14)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table(days):
    print(f"\n{'Date':<15} {'Mean rate':>10} {'Std rate':>10} {'% dead ch.':>11} "
          f"{'Mean mel':>10} {'Std mel':>10}")
    print("-" * 70)
    for date, features, targets in days:
        stds = features.std(axis=0)
        threshold = 0.1 * np.median(stds)
        pct_dead = (stds < threshold).mean() * 100
        print(f"{date:<15} {features.mean():>10.3f} {features.std():>10.3f} "
              f"{pct_dead:>10.1f}% {targets.mean():>10.3f} {targets.std():>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    days = load_days(args.subject, args.region, args.data_root)
    print(f"Loaded {len(days)} day(s) for {args.subject}/{args.region}")

    print_summary_table(days)

    from pipeline_paths import PipelinePaths
    agg_dir = PipelinePaths(args.data_root, args.subject, args.region, days[0][0]).aggregate
    out_path = agg_dir / f"cross_day_heterogeneity_{args.subject}_{args.region}.png"
    build_dashboard(days, args.subject, args.region, out_path)

    print(
        "\nWhat to look for:"
        "\n  Panel A: horizontal banding or a day with a visibly different overall"
        "\n           color level suggests a global firing-rate shift that day."
        "\n  Panel B: same idea for acoustic content -- a day that's systematically"
        "\n           louder/quieter or spectrally different from the others."
        "\n  Panel C: columns that flip between green and red across days are"
        "\n           channels whose 'meaning' as a feature is inconsistent"
        "\n           across the timeline you'd be concatenating."
        "\n  Panel D: if each day's points form a separate, non-overlapping cluster,"
        "\n           that's the clearest possible evidence of a day-level domain"
        "\n           shift -- a decoder trained on the concatenated data would be"
        "\n           partly learning to distinguish days, not just neural content."
    )


if __name__ == "__main__":
    main()
