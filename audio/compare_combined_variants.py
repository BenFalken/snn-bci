"""
compare_combined_variants.py

Directly compares every COMBINED_* variant's decoder results side by side
-- the comparison run_combined_variants.sh cares about, distinct from
analyze_results.py's aggregate mode (which deliberately excludes combined
variants from the per-real-day cross-date view).

Usage:
    python3 compare_combined_variants.py --subject Neil --region aSTG
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_paths import discover_all_runs, DEFAULT_DATA_ROOT

DECODER_ORDER = [
    "wiener_filter", "wiener_cascade", "kalman_filter",
    "dense_nn", "simple_rnn", "gru_rnn", "lstm_rnn", "lsnn",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    all_runs = discover_all_runs(args.data_root, include_combined=True)
    variant_runs = [
        r for r in all_runs
        if r.subject == args.subject and r.region == args.region
        and r.date.startswith("COMBINED")
    ]
    if not variant_runs:
        print("No COMBINED_* variants with results found -- nothing to compare.")
        return

    rows = []
    for r in variant_runs:
        for name in DECODER_ORDER:
            csv_path = r.results / f"{name}_regression_metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["decoder"] = name
            df["variant"] = r.date
            rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    out_dir = Path(args.data_root) / args.subject / args.region
    combined.to_csv(out_dir / "combined_variants_comparison.csv", index=False)

    variants = sorted(combined["variant"].unique())
    decoders_present = [d for d in DECODER_ORDER if d in combined["decoder"].unique()]

    print("\nMean validation correlation, variant x decoder:")
    pivot = combined.pivot_table(index="variant", columns="decoder", values="valid_mean_rho")
    print(pivot.round(3).to_string())

    fig, ax = plt.subplots(figsize=(max(10, len(decoders_present) * 1.5), 6))
    x = np.arange(len(decoders_present))
    width = 0.8 / len(variants)

    for i, variant in enumerate(variants):
        sub = combined[combined["variant"] == variant].set_index("decoder").reindex(decoders_present)
        ax.bar(x + i * width - 0.4 + width / 2, sub["valid_mean_rho"], width, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels(decoders_present, rotation=30, ha="right")
    ax.set_ylabel("Mean validation correlation")
    ax.set_title(f"Combined-variant comparison -- {args.subject}/{args.region}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = out_dir / "combined_variants_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
