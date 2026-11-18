"""
generate_word_prediction_audio.py

For each requested word, finds a real validation-set instance of it (from
a specific combined dataset, e.g. FIVEWORD_VERIFIED_...), and reconstructs
both the true audio (ground truth, from the actual target mel-spectrogram)
and the decoder's predicted audio (from that decoder's prediction) --
letting you listen to both back to back.

This requires precisely replaying how that word's epoch was originally
folded into the combined dataset's row ordering (matching
build_dataset_from_rows in review_word_epochs.py exactly), then accounting
for windowing trim (create_window_features drops `samples_before` rows off
the front) if the requested decoder uses windowed input, then checking
whether the resulting row actually falls inside the validation split --
only validation-split epochs have a real prediction to compare against.

Usage:
    python3 generate_word_prediction_audio.py \
        --subject Neil --region aSTG \
        --raw-base-dir "/Users/you/.../aSTG" \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv" \
        --dataset-date FIVEWORD_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13 \
        --decoder lstm_rnn \
        --words tree,good,north,cricket,program \
        --output-dir word_predictions
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pipeline_paths import PipelinePaths
from prepare_auditory_data import BIN_MS
from build_word_dataset import FIVE_WORD_SUBSET

SAMPLING_RATE = 30000.0
TRAIN_DECIMAL = 0.80
VALID_DECIMAL = 0.10
SAMPLES_BEFORE = 8
SAMPLES_AFTER = 8
NUM_MELS = 128
MEL_FRAME_SHIFT_MS = 40.0
RANDOM_SEED = 1337

DECODER_SPLIT_TAG = {
    "wiener_filter": "flat_window", "wiener_cascade": "flat_window", "dense_nn": "flat_window",
    "kalman_filter": "raw", "simple_rnn": "window", "gru_rnn": "window", "lstm_rnn": "window",
    "lsnn": "window", "lsnn_pretrained": "window",
    # the same all-words pretraining for both (one seed each): the fair pretrained comparison
    "lstm_rnn_allwords_pretrained": "window", "lsnn_allwords_pretrained": "window",
}
SILENCE_S = 0.4


def load_relevant_rows(log_csv, dataset_date):
    """Replays the same filter build_dataset_from_rows used, based on
    which kind of combined dataset dataset_date names."""
    with open(log_csv) as f:
        all_rows = [r for r in csv.DictReader(f) if r.get("verified_word")]

    if "FIVEWORD" in dataset_date:
        return [r for r in all_rows if r["verified_word"] in FIVE_WORD_SUBSET]
    elif "ALLWORDS" in dataset_date:
        return all_rows
    raise ValueError(f"dataset_date '{dataset_date}' doesn't look like a FIVEWORD_* or "
                      f"ALLWORDS_* combined dataset -- can't infer which epochs it was built from.")


def compute_all_epoch_global_ranges(relevant_rows, args):
    """Exactly replays build_dataset_from_rows's iteration order to find
    every row's (global_start, global_end) bin position within the
    combined dataset's features.npy/targets.npy -- necessary because
    that's the only way to know where a specific word instance actually
    landed in the final concatenated array."""
    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))
    dates = sorted(set(r["date"] for r in relevant_rows))

    ranges = {}
    combined_position = 0

    for date in dates:
        paths = PipelinePaths(args.data_root, args.subject, args.region, date)
        sessions_this_date = sorted(set(r["session"] for r in relevant_rows if r["date"] == date))

        for session in sessions_this_date:
            audio_raw = np.load(paths.prep / f"{session}_audio_raw.npy").flatten()
            n_bins_session = len(audio_raw) // bin_samples

            session_rows = [r for r in relevant_rows if r["date"] == date and r["session"] == session]
            for r in session_rows:
                pad = int(args.pad_s * SAMPLING_RATE)
                start = max(0, int(r["start_sample"]) - pad)
                end = min(len(audio_raw), int(r["end_sample"]) + pad)
                bin_start_local = max(0, start // bin_samples)
                bin_end_local = min(n_bins_session, -(-end // bin_samples))
                length = bin_end_local - bin_start_local

                key = (r["date"], r["session"], int(r["epoch_idx"]))
                ranges[key] = (combined_position, combined_position + length)
                combined_position += length

    return ranges, combined_position


def find_validation_epoch(word, relevant_rows, ranges, combined_total, decoder):
    """Among all instances of `word`, finds one whose corresponding row
    (after accounting for windowing trim, if the decoder is windowed)
    falls inside the validation split. Returns (row, local_valid_start,
    local_valid_end) or None if no instance of this word landed in
    validation."""
    tag = DECODER_SPLIT_TAG[decoder]
    windowed = tag in ("window", "flat_window")
    offset = SAMPLES_BEFORE if windowed else 0
    effective_total = combined_total - (SAMPLES_BEFORE + SAMPLES_AFTER) if windowed else combined_total

    train_end = int(TRAIN_DECIMAL * effective_total)
    valid_end = train_end + int(VALID_DECIMAL * effective_total)

    candidates = [r for r in relevant_rows if r["verified_word"] == word]
    for r in candidates:
        key = (r["date"], r["session"], int(r["epoch_idx"]))
        if key not in ranges:
            continue
        g_start, g_end = ranges[key]

        w_start, w_end = g_start - offset, g_end - offset
        w_start, w_end = max(0, w_start), min(effective_total, w_end)
        if w_end <= w_start:
            continue  # entirely trimmed away by windowing

        overlap_start = max(w_start, train_end)
        overlap_end = min(w_end, valid_end)
        if overlap_end > overlap_start:
            # Also return the SAME overlap region mapped back into the
            # original (pre-windowing) index space -- needed so callers
            # slicing the raw features array (which was never windowed)
            # get a length-matched slice, not the epoch's full un-clipped
            # range. An epoch can be partially clipped here (straddling
            # the train/valid boundary, or the array's windowing-trimmed
            # edge), so the clipped length can be shorter than the full
            # epoch -- both slices must use that same clipped length.
            orig_overlap_start = overlap_start + offset
            orig_overlap_end = overlap_end + offset
            return r, overlap_start - train_end, overlap_end - train_end, orig_overlap_start, orig_overlap_end

    return None


def run_module(image, config, config_name, mount_dir):
    config_path = mount_dir / config_name
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{mount_dir}:{mount_dir}", image, str(config_path)]
    )
    if result.returncode != 0:
        raise RuntimeError(f"{image} failed for {config_name}")


def reconstruct_audio(mel_slice_scaled, paths, tag_name):
    """Inverse-transform (undo StandardScaler) then inverse-mel-transform
    (Griffin-Lim) a scaled mel-spectrogram slice into real audio."""
    scaled_npy = paths.pipeline / f"tmp_{tag_name}_scaled.npy"
    unscaled_npy = paths.pipeline / f"tmp_{tag_name}_unscaled.npy"
    wav_path = paths.results / f"{tag_name}.wav"

    np.save(scaled_npy, mel_slice_scaled)

    run_module(
        "sklearn_preprocess_test",
        {
            "params": {"method": "inverse_transform", "save_model": False, "random_seed": RANDOM_SEED},
            "input_pathnames": {"X_npy": str(scaled_npy), "model_joblib": str(paths.pipeline / "targets_scaler.joblib")},
            "output_pathnames": {"output_npy": str(unscaled_npy)},
        },
        f"job_{tag_name}_inverse_transform.json", paths.base,
    )

    run_module(
        "mikko_test",
        {
            "params": {
                "method": "inverse_mel_transform", "sampling_rate": SAMPLING_RATE,
                "ref_level_db": 20, "power": 1.5, "griffin_lim_iters": 60, "fft_size": 2048,
                "num_mels": NUM_MELS, "preemphasis": 0.97, "max_abs_value": 4, "min_level_db": -100,
                "frame_shift_ms": MEL_FRAME_SHIFT_MS, "random_seed": RANDOM_SEED, "save_transformer": False,
            },
            "input_pathnames": {"input": str(unscaled_npy)},
            "output_pathnames": {"audio_wav": str(wav_path)},
        },
        f"job_{tag_name}_inverse_mel_transform.json", paths.base,
    )

    audio = wavfile.read(wav_path)[1].astype(np.float32) / 32767.0
    mel_unscaled = np.load(unscaled_npy)
    return audio, mel_unscaled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default="pipeline_data")
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--dataset-date", required=True,
                         help="The combined dataset's pseudo-date, e.g. FIVEWORD_VERIFIED_...")
    parser.add_argument("--decoder", default="lstm_rnn")
    parser.add_argument("--words", required=True)
    parser.add_argument("--pad-s", type=float, default=0.3)
    parser.add_argument("--output-dir", default="word_predictions")
    args = parser.parse_args()

    paths = PipelinePaths(args.data_root, args.subject, args.region, args.dataset_date)
    tag = DECODER_SPLIT_TAG[args.decoder]

    targets_valid = np.load(paths.pipeline / f"{tag}_targets_valid.npy")
    predict_valid = np.load(paths.pipeline / f"{args.decoder}_predict_valid.npy")

    relevant_rows = load_relevant_rows(args.log_csv, args.dataset_date)
    ranges, combined_total = compute_all_epoch_global_ranges(relevant_rows, args)

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    battery = []
    for i, word in enumerate(words, start=1):
        result = find_validation_epoch(word, relevant_rows, ranges, combined_total, args.decoder)
        if result is None:
            print(f"{i:02d}: no validation-set instance of '{word}' found for decoder "
                  f"'{args.decoder}' -- skipping (try --dataset-date ALLWORDS_VERIFIED_... "
                  f"for more instances, or a different decoder).")
            continue

        row, local_start, local_end, _, _ = result
        print(f"{i:02d}: '{word}' -- {row['date']}/{row['session']}, "
              f"valid rows [{local_start}:{local_end}]")

        target_slice = targets_valid[local_start:local_end]
        predict_slice = predict_valid[local_start:local_end]

        gt_audio, gt_mel_unscaled = reconstruct_audio(target_slice, paths, f"{i:02d}_{word}_groundtruth")
        pred_audio, pred_mel_unscaled = reconstruct_audio(predict_slice, paths, f"{i:02d}_{word}_{args.decoder}")

        wavfile.write(out_dir / f"{i:02d}_{word}_groundtruth.wav", int(SAMPLING_RATE),
                      (gt_audio * 32767).astype(np.int16))
        wavfile.write(out_dir / f"{i:02d}_{word}_{args.decoder}.wav", int(SAMPLING_RATE),
                      (pred_audio * 32767).astype(np.int16))

        silence = np.zeros(int(SILENCE_S * SAMPLING_RATE), dtype=np.float32)
        battery.extend([gt_audio, silence, pred_audio, silence, silence])

    if battery:
        battery_audio = np.concatenate(battery)
        battery_path = out_dir / "word_prediction_battery.wav"
        wavfile.write(battery_path, int(SAMPLING_RATE), (battery_audio * 32767).astype(np.int16))
        print(f"\nSingle sequential file (groundtruth, prediction, per word, in order): {battery_path}")

    print(f"\nIndividual files saved to: {out_dir}/")


if __name__ == "__main__":
    main()
