"""
render_word_prediction_video.py

Final unification: for each word, renders two synced video sub-clips back
to back -- (1) ground-truth spectrogram + real neural activity, cursor
synced to the true reconstructed audio, then (2) the decoder's PREDICTED
spectrogram + the same real neural activity, cursor synced to the
decoder's reconstructed audio. Concatenates every word's pair, in order,
into one final video.

Reuses the exact epoch-to-validation-split mapping and audio
reconstruction from generate_word_prediction_audio.py -- this script does
not re-derive that logic, it imports it directly, so both scripts always
agree on which epoch instance and which validation rows a given word maps to.

Usage:
    python3 render_word_prediction_video.py \
        --subject Neil --region aSTG \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv" \
        --dataset-date FIVEWORD_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13 \
        --decoder lstm_rnn \
        --words tree,good,north,cricket,program \
        --output word_predictions.mp4
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.io import wavfile

from pipeline_paths import PipelinePaths
from generate_word_prediction_audio import (
    load_relevant_rows, compute_all_epoch_global_ranges, find_validation_epoch,
    reconstruct_audio, DECODER_SPLIT_TAG, SAMPLING_RATE,
)


def render_subclip(mel_slice, feat_slice, audio, title, out_path, fps):
    """Renders one spectrogram+activity+cursor clip, synced to `audio`'s
    ACTUAL length (not an assumed bin-count*frame-shift duration) --
    Griffin-Lim reconstruction doesn't guarantee an exact sample count, so
    the cursor timeline is built from the real audio duration to avoid
    silent drift between video and audio."""
    duration_s = len(audio) / SAMPLING_RATE
    n_bins = mel_slice.shape[0]
    bin_times = np.linspace(0, duration_s, n_bins, endpoint=False)

    fig, (ax_mel, ax_spike) = plt.subplots(2, 1, figsize=(10, 6))
    ax_mel.pcolormesh(bin_times, np.arange(mel_slice.shape[1]), mel_slice.T,
                       cmap="magma", shading="auto")
    ax_mel.set_ylabel("Mel band")
    ax_mel.set_title(title)

    ax_spike.pcolormesh(bin_times, np.arange(feat_slice.shape[1]), feat_slice.T,
                         cmap="viridis", shading="auto")
    ax_spike.set_ylabel("Channel")
    ax_spike.set_xlabel("Time (s)")

    cursor_mel = ax_mel.axvline(0, color="cyan", linewidth=2)
    cursor_spike = ax_spike.axvline(0, color="cyan", linewidth=2)
    plt.tight_layout()

    n_frames = max(1, int(duration_s * fps))

    def update(frame_idx):
        t = frame_idx / fps
        cursor_mel.set_xdata([t, t])
        cursor_spike.set_xdata([t, t])
        return cursor_mel, cursor_spike

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    writer = FFMpegWriter(fps=fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

    silent_path = out_path.with_suffix(".silent.mp4")
    anim.save(str(silent_path), writer=writer)
    plt.close(fig)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wavfile.write(tmp_wav.name, int(SAMPLING_RATE), (audio * 32767).astype(np.int16))
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(silent_path), "-i", tmp_wav.name,
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(result.stderr.decode())
            raise RuntimeError(f"ffmpeg audio mux failed for {out_path}")

    Path(tmp_wav.name).unlink(missing_ok=True)
    silent_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default="pipeline_data")
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--dataset-date", required=True)
    parser.add_argument("--decoder", default="lstm_rnn")
    parser.add_argument("--words", required=True)
    parser.add_argument("--pad-s", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="word_predictions.mp4")
    args = parser.parse_args()

    paths = PipelinePaths(args.data_root, args.subject, args.region, args.dataset_date)
    tag = DECODER_SPLIT_TAG[args.decoder]

    targets_valid = np.load(paths.pipeline / f"{tag}_targets_valid.npy")
    predict_valid = np.load(paths.pipeline / f"{args.decoder}_predict_valid.npy")
    features_combined = np.load(paths.pipeline / "features.npy")  # pre-windowing, raw spike counts

    relevant_rows = load_relevant_rows(args.log_csv, args.dataset_date)
    ranges, combined_total = compute_all_epoch_global_ranges(relevant_rows, args)

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    tmp_dir = Path(tempfile.mkdtemp(prefix="word_prediction_video_"))
    subclip_paths = []

    for i, word in enumerate(words, start=1):
        result = find_validation_epoch(word, relevant_rows, ranges, combined_total, args.decoder)
        if result is None:
            print(f"{i:02d}: no validation-set instance of '{word}' -- skipping")
            continue

        row, local_start, local_end, orig_overlap_start, orig_overlap_end = result
        feat_slice = features_combined[orig_overlap_start:orig_overlap_end]

        target_slice = targets_valid[local_start:local_end]
        predict_slice = predict_valid[local_start:local_end]

        print(f"{i:02d}: '{word}' -- {row['date']}/{row['session']}, "
              f"valid rows [{local_start}:{local_end}]")

        gt_audio, gt_mel = reconstruct_audio(target_slice, paths, f"{i:02d}_{word}_groundtruth")
        pred_audio, pred_mel = reconstruct_audio(predict_slice, paths, f"{i:02d}_{word}_{args.decoder}")

        gt_path = tmp_dir / f"{i:02d}a_{word}_groundtruth.mp4"
        pred_path = tmp_dir / f"{i:02d}b_{word}_{args.decoder}.mp4"

        render_subclip(gt_mel, feat_slice, gt_audio, f"Ground truth: '{word}'", gt_path, args.fps)
        render_subclip(pred_mel, feat_slice, pred_audio,
                        f"{args.decoder} prediction: '{word}'", pred_path, args.fps)

        subclip_paths.extend([gt_path, pred_path])

    if not subclip_paths:
        print("No words rendered -- nothing to concatenate.")
        return

    concat_list = tmp_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in subclip_paths:
            f.write(f"file '{p.resolve()}'\n")

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", args.output],
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stderr.decode())
        raise RuntimeError("ffmpeg concatenation failed")

    print(f"\nFinal video: {args.output}")


if __name__ == "__main__":
    main()
