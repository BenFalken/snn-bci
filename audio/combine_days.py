"""
combine_days.py

Builds a combined multi-day dataset from every successfully-completed date
for a given (subject, region), saving it under a synthetic "date" folder so
the existing run_auditory_experiment.py / analyze_results.py machinery
works on it completely unchanged.

Three modes:
  --mode naive           Plain sequential concatenation of each day's raw
                          features.npy/targets.npy, in date order. This is
                          the exact approach that previously produced
                          near-zero validation/test correlation across
                          every decoder -- included here for a direct,
                          reproducible confirmation/refutation rather than
                          relying on memory of the earlier result.
  --mode per_day_normalized   Each day's features and targets are
                          standardized (zero mean, unit variance, per
                          channel/band) INDEPENDENTLY before concatenation,
                          using statistics computed from that day's ENTIRE
                          timeline (train + valid + test bins together).
                          This is an offline-only variant: it uses
                          information a causal, online system would not
                          have yet, so treat its results as a ceiling, not
                          a preview of deployable performance.
  --mode per_day_normalized_causal   Same idea, but the mean/std (and clip
                          bounds, if used) are computed from ONLY each
                          day's leading TRAIN_DECIMAL fraction -- exactly
                          mirroring the calibration pattern already used
                          elsewhere in this pipeline (per-session spike
                          threshold estimation in prepare_auditory_data.py,
                          and the global StandardScaler's train_decimal=0.8
                          fit in run_auditory_experiment.py). The frozen
                          per-day statistics are then applied to that
                          day's full timeline, including its own valid/test
                          bins. This variant is causal and deployable: a
                          real system could run an equivalent calibration
                          window at the start of each session and freeze
                          normalization from there.

Includes a fast, decoder-free diagnostic immediately after combining (same
logic as the earlier diagnose_split_energy.py), so a doomed combination is
visible in seconds rather than after hours of decoder training.

Usage:
    python3 combine_days.py --subject Neil --region aSTG --mode naive
    python3 combine_days.py --subject Neil --region aSTG --mode per_day_normalized --clip-std 5.0
    python3 combine_days.py --subject Neil --region aSTG --mode per_day_normalized_causal --clip-std 5.0
"""

import argparse

import numpy as np

from pipeline_paths import PipelinePaths, discover_all_runs

TRAIN_DECIMAL = 0.80
VALID_DECIMAL = 0.10


def load_day_arrays(paths):
    features = np.load(paths.pipeline / "features.npy")
    targets = np.load(paths.pipeline / "targets.npy")
    return features, targets


def zscore_per_day(arr, clip_std=None, causal=False, train_fraction=TRAIN_DECIMAL):
    """Standardizes `arr` (zero mean, unit variance, per channel/band).

    If causal=False (default): statistics are computed from the entire
    array -- offline-only, uses information a live system wouldn't have.

    If causal=True: statistics are computed from only the leading
    `train_fraction` of the array (matching the calibration-window pattern
    already used for spike-threshold estimation and the global scaler
    elsewhere in this pipeline), then frozen and applied to the full
    array. This is the deployable version.
    """
    if causal:
        n_train = int(arr.shape[0] * train_fraction)
        calib_segment = arr[:n_train]
    else:
        calib_segment = arr

    mean = calib_segment.mean(axis=0, keepdims=True)
    std = calib_segment.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid divide-by-zero for any dead channel/band

    z = (arr - mean) / std
    if clip_std is not None:
        n_clipped = np.sum(np.abs(z) > clip_std)
        if n_clipped > 0:
            pct = 100 * n_clipped / z.size
            print(f"    Clipping {n_clipped} values ({pct:.3f}%) beyond +/-{clip_std} std")
        z = np.clip(z, -clip_std, clip_std)
    return z


def quick_saturation_check(targets, label):
    """Same diagnostic as diagnose_split_energy.py, run inline so a doomed
    combination is caught before any decoder training starts."""
    n = targets.shape[0]
    train_end = int(TRAIN_DECIMAL * n)
    valid_end = train_end + int(VALID_DECIMAL * n)

    for split_name, split in [("train", targets[:train_end]),
                               ("valid", targets[train_end:valid_end]),
                               ("test", targets[valid_end:])]:
        energy = split.mean(axis=1)
        print(f"  [{label}] {split_name}: energy mean={energy.mean():.3f} "
              f"std={energy.std():.3f} (std near 0 = degenerate/saturated split)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default="pipeline_data")
    parser.add_argument(
        "--mode",
        choices=["naive", "per_day_normalized", "per_day_normalized_causal"],
        required=True,
    )
    parser.add_argument(
        "--clip-std", type=float, default=None,
        help="Used with --mode per_day_normalized or per_day_normalized_causal. "
             "Clips each day's z-scored features to +/- this many standard "
             "deviations before concatenation, so rare extreme bursts (seen in "
             "cross-day diagnostics) can't distort the pooled scale. Try 5.0 "
             "as a starting point.",
    )
    parser.add_argument("--exclude-dates", default="",
                         help="Comma-separated dates to leave out of the combination, e.g. a known-bad day")
    args = parser.parse_args()

    exclude = set(d.strip() for d in args.exclude_dates.split(",") if d.strip())

    all_runs = discover_all_runs(args.data_root)
    day_runs = [
        r for r in all_runs
        if r.subject == args.subject and r.region == args.region
        and not r.date.startswith("COMBINED")  # never combine an already-combined dataset
        and r.date not in exclude
    ]
    day_runs.sort(key=lambda r: r.date)

    if not day_runs:
        raise RuntimeError(f"No completed days found for {args.subject}/{args.region} under {args.data_root}")

    print(f"Combining {len(day_runs)} day(s) for {args.subject}/{args.region}, mode={args.mode}:")
    for r in day_runs:
        print(f"  {r.date}")

    all_features, all_targets = [], []
    for r in day_runs:
        features, targets = load_day_arrays(r)
        print(f"\n{r.date}: features {features.shape}, targets {targets.shape}")

        if args.mode == "per_day_normalized":
            features = zscore_per_day(features, clip_std=args.clip_std, causal=False)
            targets = zscore_per_day(targets, clip_std=args.clip_std, causal=False)
        elif args.mode == "per_day_normalized_causal":
            n_train = int(features.shape[0] * TRAIN_DECIMAL)
            print(f"    Calibration window: first {n_train} of {features.shape[0]} bins "
                  f"({TRAIN_DECIMAL*100:.0f}%) -- statistics frozen from this window only")
            features = zscore_per_day(features, clip_std=args.clip_std, causal=True)
            targets = zscore_per_day(targets, clip_std=args.clip_std, causal=True)

        all_features.append(features)
        all_targets.append(targets)

    combined_features = np.concatenate(all_features, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)

    output_date = f"COMBINED_{args.mode}"
    if args.mode in ("per_day_normalized", "per_day_normalized_causal") and args.clip_std is not None:
        output_date += f"_clip{args.clip_std}"
    out_paths = PipelinePaths(args.data_root, args.subject, args.region, output_date)

    np.save(out_paths.pipeline / "features.npy", combined_features)
    np.save(out_paths.pipeline / "targets.npy", combined_targets)

    print(f"\nCombined features.npy: {combined_features.shape}")
    print(f"Combined targets.npy:  {combined_targets.shape}")
    print(f"Saved to {out_paths.pipeline}")

    print(f"\nQuick pre-flight diagnostic (before any decoder training):")
    quick_saturation_check(combined_targets, output_date)
    if args.mode == "per_day_normalized":
        print(
            "\nNOTE: this mode used whole-day statistics (offline-only) -- "
            "treat results as a ceiling, not a preview of deployable, causal "
            "performance. Compare against per_day_normalized_causal for the "
            "online-realistic estimate."
        )
    elif args.mode == "per_day_normalized_causal":
        print(
            "\nThis mode used only each day's leading calibration window "
            "for normalization statistics -- results here are a realistic "
            "estimate of what a causal, online system could achieve."
        )
    print(
        "\nIf 'valid'/'test' std is near zero above, this combination is "
        "already showing the same degenerate pattern seen in the earlier "
        "naive cross-day attempt -- worth stopping here rather than "
        "spending hours training decoders on it."
    )

    print(f"\nNext steps:")
    print(f"  python3 run_auditory_experiment.py --subject {args.subject} --region {args.region} "
          f"--date {output_date} --all-decoders")
    print(f"  python3 analyze_results.py --mode single --subject {args.subject} --region {args.region} "
          f"--date {output_date}")


if __name__ == "__main__":
    main()
