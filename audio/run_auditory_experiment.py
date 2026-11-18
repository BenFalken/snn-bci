"""
run_auditory_experiment.py

Orchestration script for the mikko auditory decoding experiment. Mimics the
job graph in experiments/basic_auditory_experiment.py, but dispatches each
stage with a plain `docker run` call instead of the full Dockex platform.

Operates on exactly one (subject, region, date) combination, reading
features.npy/targets.npy from that date's pipeline/ folder (produced by
prepare_auditory_data.py) and writing regression_metrics.csv + reconstructed
.wav files to that date's results/ folder.

Requires all module images already built (see build_all_images.sh):
    sklearn_preprocess_test, train_valid_test_split_test,
    create_window_features_test, kalman_filter_test, wiener_filter_test,
    wiener_cascade_test, dense_nn_test, gru_rnn_test, lstm_rnn_test,
    simple_rnn_test, regression_metrics_test, mikko_test

Usage:
    python3 run_auditory_experiment.py \
        --subject Neil --region aSTG --date 07_19_11 \
        --decoders wiener_filter,dense_nn,lstm_rnn

    python3 run_auditory_experiment.py \
        --subject Neil --region aSTG --date 07_19_11 --all-decoders
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone

from pipeline_paths import PipelinePaths, add_identity_args

ALL_DECODERS = [
    "wiener_filter", "wiener_cascade", "kalman_filter",
    "dense_nn", "simple_rnn", "gru_rnn", "lstm_rnn", "lsnn",
]

# ---------------------------------------------------------------------------
# Experiment-wide constants (mirrors basic_auditory_experiment.py)
# ---------------------------------------------------------------------------
TRAIN_DECIMAL = 0.80
VALID_DECIMAL = 0.10
TEST_DECIMAL = 0.10

SAMPLES_BEFORE = 8
SAMPLES_AFTER = 8
USE_CURRENT_SAMPLE = True

RANDOM_SEED = 1337
SAMPLING_RATE = 30000.0

# The LSNN's settings (see run_decoder)
LSNN_UNITS = 256
LSNN_PARAMS = {"input_dropout": 0.2, "layer_dropout": 0.2, "white_noise_sd": 1.0, "constant_offset_sd": 0.2,
               "patience": 10}

# Must match the values used to generate targets.npy in prepare_auditory_data.py
NUM_MELS = 128
MEL_FRAME_SHIFT_MS = 40.0


# ---------------------------------------------------------------------------
# Helper: write a job config and run it in the given image
# ---------------------------------------------------------------------------
def run_module(image, config, config_name, pipeline_dir):
    config_path = pipeline_dir / config_name
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}\nRunning {image}  ({config_name})\n{'=' * 70}")

    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{pipeline_dir}:{pipeline_dir}", image, str(config_path)]
    )
    if result.returncode != 0:
        raise RuntimeError(f"{image} failed (exit code {result.returncode}) -- see output above.")


# ---------------------------------------------------------------------------
# Stage 1: standardize features and targets (fit on train fraction, then
# transform the FULL array)
# ---------------------------------------------------------------------------
def preprocess_stage(paths):
    p = lambda name: str(paths.pipeline / name)
    features_npy = paths.pipeline / "features.npy"
    targets_npy = paths.pipeline / "targets.npy"

    for label, x_npy, scaler_out in [
        ("features", features_npy, p("features_scaler.joblib")),
        ("targets", targets_npy, p("targets_scaler.joblib")),
    ]:
        run_module(
            "sklearn_preprocess_test",
            {
                "params": {
                    "class": "StandardScaler",
                    "kwargs": {"copy": True, "with_mean": True, "with_std": label == "features"},
                    "method": "fit",
                    "train_decimal": TRAIN_DECIMAL,
                    "save_model": True,
                    "random_seed": RANDOM_SEED,
                },
                "input_pathnames": {"X_npy": str(x_npy)},
                "output_pathnames": {"model_joblib": scaler_out},
            },
            f"job_{label}_preprocess_fit.json", paths.base,
        )
        run_module(
            "sklearn_preprocess_test",
            {
                "params": {"method": "transform", "save_model": False, "random_seed": RANDOM_SEED},
                "input_pathnames": {"X_npy": str(x_npy), "model_joblib": scaler_out},
                "output_pathnames": {"output_npy": p(f"{label}_scaled_full.npy")},
            },
            f"job_{label}_preprocess_transform.json", paths.base,
        )


# ---------------------------------------------------------------------------
# Stage 2: split the (unwindowed) scaled features/targets -- for Kalman filter
# ---------------------------------------------------------------------------
def raw_split_stage(paths):
    p = lambda name: str(paths.pipeline / name)
    run_module(
        "train_valid_test_split_test",
        {
            "params": {"train_decimal": TRAIN_DECIMAL, "valid_decimal": VALID_DECIMAL, "test_decimal": TEST_DECIMAL},
            "input_pathnames": {"features_npy": p("features_scaled_full.npy"), "targets_npy": p("targets_scaled_full.npy")},
            "output_pathnames": {
                "features_train_npy": p("raw_features_train.npy"), "targets_train_npy": p("raw_targets_train.npy"),
                "features_valid_npy": p("raw_features_valid.npy"), "targets_valid_npy": p("raw_targets_valid.npy"),
                "features_test_npy": p("raw_features_test.npy"), "targets_test_npy": p("raw_targets_test.npy"),
            },
        },
        "job_raw_split.json", paths.base,
    )


# ---------------------------------------------------------------------------
# Stage 3: window, then split both windowed (3D) and flattened (2D) versions
# ---------------------------------------------------------------------------
def window_stage(paths):
    p = lambda name: str(paths.pipeline / name)
    run_module(
        "create_window_features_test",
        {
            "params": {"samples_before": SAMPLES_BEFORE, "samples_after": SAMPLES_AFTER, "use_current_sample": USE_CURRENT_SAMPLE},
            "input_pathnames": {"features_npy": p("features_scaled_full.npy"), "targets_npy": p("targets_scaled_full.npy")},
            "output_pathnames": {
                "window_features_npy": p("window_features_full.npy"), "window_targets_npy": p("window_targets_full.npy"),
                "flat_window_features_npy": p("flat_window_features_full.npy"), "flat_window_targets_npy": p("flat_window_targets_full.npy"),
            },
        },
        "job_create_window_features.json", paths.base,
    )

    for tag, feat_key, targ_key in [
        ("window", "window_features_full.npy", "window_targets_full.npy"),
        ("flat_window", "flat_window_features_full.npy", "flat_window_targets_full.npy"),
    ]:
        run_module(
            "train_valid_test_split_test",
            {
                "params": {"train_decimal": TRAIN_DECIMAL, "valid_decimal": VALID_DECIMAL, "test_decimal": TEST_DECIMAL},
                "input_pathnames": {"features_npy": p(feat_key), "targets_npy": p(targ_key)},
                "output_pathnames": {
                    "features_train_npy": p(f"{tag}_features_train.npy"), "targets_train_npy": p(f"{tag}_targets_train.npy"),
                    "features_valid_npy": p(f"{tag}_features_valid.npy"), "targets_valid_npy": p(f"{tag}_targets_valid.npy"),
                    "features_test_npy": p(f"{tag}_features_test.npy"), "targets_test_npy": p(f"{tag}_targets_test.npy"),
                },
            },
            f"job_{tag}_split.json", paths.base,
        )


# ---------------------------------------------------------------------------
# Stage 4: decoders
# ---------------------------------------------------------------------------
DECODER_SPLIT_TAG = {
    "wiener_filter": "flat_window", "wiener_cascade": "flat_window", "dense_nn": "flat_window",
    "kalman_filter": "raw", "simple_rnn": "window", "gru_rnn": "window", "lstm_rnn": "window",
    "lsnn": "window",
}


def _decoder_io(paths, tag, name):
    p = lambda n: str(paths.pipeline / n)
    return {
        "input_pathnames": {
            "X_train_npy": p(f"{tag}_features_train.npy"), "y_train_npy": p(f"{tag}_targets_train.npy"),
            "X_valid_npy": p(f"{tag}_features_valid.npy"), "y_valid_npy": p(f"{tag}_targets_valid.npy"),
            "X_test_npy": p(f"{tag}_features_test.npy"), "y_test_npy": p(f"{tag}_targets_test.npy"),
        },
        "output_pathnames": {
            "predict_train_npy": p(f"{name}_predict_train.npy"), "predict_valid_npy": p(f"{name}_predict_valid.npy"),
            "predict_test_npy": p(f"{name}_predict_test.npy"),
        },
    }


def run_regression_metrics(paths, name, tag):
    p = lambda n: str(paths.pipeline / n)
    run_module(
        "regression_metrics_test",
        {
            "name": f"{name}_regression_metrics",
            "params": {},
            "input_pathnames": {
                "targets_train_npy": p(f"{tag}_targets_train.npy"), "predict_train_npy": p(f"{name}_predict_train.npy"),
                "targets_valid_npy": p(f"{tag}_targets_valid.npy"), "predict_valid_npy": p(f"{name}_predict_valid.npy"),
                "targets_test_npy": p(f"{tag}_targets_test.npy"), "predict_test_npy": p(f"{name}_predict_test.npy"),
            },
            "output_pathnames": {"csv": str(paths.results / f"{name}_regression_metrics.csv")},
        },
        f"job_{name}_regression_metrics.json", paths.base,
    )


def reconstruct_audio_from_predictions(paths, name):
    p = lambda n: str(paths.pipeline / n)
    run_module(
        "sklearn_preprocess_test",
        {
            "params": {"method": "inverse_transform", "save_model": False, "random_seed": RANDOM_SEED},
            "input_pathnames": {"X_npy": p(f"{name}_predict_valid.npy"), "model_joblib": p("targets_scaler.joblib")},
            "output_pathnames": {"output_npy": p(f"{name}_predict_valid_unscaled.npy")},
        },
        f"job_{name}_inverse_transform.json", paths.base,
    )
    run_module(
        "mikko_test",
        {
            "params": {
                "method": "inverse_mel_transform", "sampling_rate": SAMPLING_RATE, "ref_level_db": 20, "power": 1.5,
                "griffin_lim_iters": 60, "fft_size": 2048, "num_mels": NUM_MELS, "preemphasis": 0.97,
                "max_abs_value": 4, "min_level_db": -100, "frame_shift_ms": MEL_FRAME_SHIFT_MS,
                "random_seed": RANDOM_SEED, "save_transformer": False,
            },
            "input_pathnames": {"input": p(f"{name}_predict_valid_unscaled.npy")},
            "output_pathnames": {"audio_wav": str(paths.results / f"{name}_reconstructed.wav")},
        },
        f"job_{name}_inverse_mel_transform.json", paths.base,
    )


def reconstruct_true_audio(paths):
    """Upper-bound reference: the actual validation targets, round-tripped
    through the same inverse pipeline as every decoder's predictions."""
    p = lambda n: str(paths.pipeline / n)
    run_module(
        "sklearn_preprocess_test",
        {
            "params": {"method": "inverse_transform", "save_model": False, "random_seed": RANDOM_SEED},
            "input_pathnames": {"X_npy": p("flat_window_targets_valid.npy"), "model_joblib": p("targets_scaler.joblib")},
            "output_pathnames": {"output_npy": p("true_valid_unscaled.npy")},
        },
        "job_true_inverse_transform.json", paths.base,
    )
    run_module(
        "mikko_test",
        {
            "params": {
                "method": "inverse_mel_transform", "sampling_rate": SAMPLING_RATE, "ref_level_db": 20, "power": 1.5,
                "griffin_lim_iters": 60, "fft_size": 2048, "num_mels": NUM_MELS, "preemphasis": 0.97,
                "max_abs_value": 4, "min_level_db": -100, "frame_shift_ms": MEL_FRAME_SHIFT_MS,
                "random_seed": RANDOM_SEED, "save_transformer": False,
            },
            "input_pathnames": {"input": p("true_valid_unscaled.npy")},
            "output_pathnames": {"audio_wav": str(paths.results / "true_reconstructed.wav")},
        },
        "job_true_inverse_mel_transform.json", paths.base,
    )


def run_decoder(paths, name, epochs, units, degree=4, C=1.0):
    """Dispatches to the right image/params for `name`, then scores and
    reconstructs audio -- one unified entry point instead of seven
    near-duplicate functions."""
    tag = DECODER_SPLIT_TAG[name]
    io = _decoder_io(paths, tag, name)
    p = lambda n: str(paths.pipeline / n)

    if name == "wiener_filter":
        params = {"method": "fit_predict", "save_model": True, "random_seed": RANDOM_SEED}
        model_key, model_path, image = "model_joblib", p("wiener_filter_model.joblib"), "wiener_filter_test"
    elif name == "wiener_cascade":
        params = {"method": "fit_predict", "degree": degree, "save_model": True, "random_seed": RANDOM_SEED}
        model_key, model_path, image = "model_joblib", p("wiener_cascade_model.joblib"), "wiener_cascade_test"
    elif name == "kalman_filter":
        params = {"method": "fit_predict", "C": C, "save_model": True, "random_seed": RANDOM_SEED}
        model_key, model_path, image = "model_joblib", p("kalman_filter_model.joblib"), "kalman_filter_test"
    elif name in ("dense_nn", "simple_rnn", "gru_rnn", "lstm_rnn"):
        params = {
            "method": "fit_predict", "units": units, "epochs": epochs,
            "batch_size": None, "save_model": True, "random_seed": RANDOM_SEED,
        }
        model_key, model_path, image = "model_keras", p(f"{name}_model.keras"), f"{name}_test"
    elif name == "lsnn":
        # the recurrent spiking LSNN (modules/decoders/torch/lsnn): the same windows as the Keras RNNs, one
        # 40 ms frame per step. Its own size and regularization, the best of a sweep on 07_19_11: 2 x 256
        # neurons (not --units), dropout 0.2 on the input and between layers, training-only input noise,
        # early stopping after 10 epochs without a lower validation loss
        params = {
            "method": "fit_predict", "units": LSNN_UNITS, "epochs": epochs,
            "batch_size": None, "save_model": True, "random_seed": RANDOM_SEED, **LSNN_PARAMS,
        }
        model_key, model_path, image = "model_pt", p("lsnn_model.pt"), "lsnn_test"
    else:
        raise ValueError(f"Unknown decoder: {name}")

    run_module(
        image,
        {"params": params, **io, "output_pathnames": {**io["output_pathnames"], model_key: model_path}},
        f"job_{name}.json", paths.base,
    )
    run_regression_metrics(paths, name, tag)
    reconstruct_audio_from_predictions(paths, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_identity_args(parser)
    parser.add_argument("--decoders", default="wiener_filter",
                         help="Comma-separated decoder names to run, e.g. "
                              "'wiener_filter,dense_nn,lstm_rnn'. Ignored if --all-decoders is set.")
    parser.add_argument("--all-decoders", action="store_true", help=f"Run all of: {', '.join(ALL_DECODERS)}")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch ceiling for Keras decoders (EarlyStopping usually cuts this short)")
    parser.add_argument("--units", type=int, default=1024, help="Hidden units for Keras decoders")
    parser.add_argument("--wiener-cascade-degree", type=int, default=4)
    parser.add_argument("--kalman-C", type=float, default=1.0)
    parser.add_argument("--skip-preprocessing", action="store_true",
                         help="Skip stages 1-3 (assumes they already ran for this date -- useful when "
                              "re-running just one more decoder on an already-prepared date)")
    args = parser.parse_args()

    paths = PipelinePaths.from_args(args)
    decoders = ALL_DECODERS if args.all_decoders else [d.strip() for d in args.decoders.split(",") if d.strip()]
    unknown = set(decoders) - set(ALL_DECODERS)
    if unknown:
        raise ValueError(f"Unknown decoder(s): {unknown}. Valid options: {ALL_DECODERS}")

    print(f"Running experiment for {paths.label()}")
    print(f"Decoders: {decoders}")

    if not args.skip_preprocessing:
        preprocess_stage(paths)
        raw_split_stage(paths)
        window_stage(paths)
    else:
        print("--skip-preprocessing set: assuming stages 1-3 already ran for this date.")

    reconstruct_true_audio(paths)

    for name in decoders:
        run_decoder(paths, name, epochs=args.epochs, units=args.units,
                    degree=args.wiener_cascade_degree, C=args.kalman_C)

    # Record what actually ran, for later reference / automation bookkeeping.
    metadata = {
        "subject": paths.subject, "region": paths.region, "date": paths.date,
        "decoders_run": decoders, "epochs": args.epochs, "units": args.units,
        "random_seed": RANDOM_SEED, "num_mels": NUM_MELS,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(paths.results / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE -- {paths.label()}")
    print(f"Results: {paths.results}")
    print("=" * 70)


if __name__ == "__main__":
    main()
