"""
render_word_trial_video.py

Renders a video of labeled word trials: two vertically stacked panels
(mel-spectrogram on top, binned spike-count activity on bottom) with a
moving vertical time cursor synced to real audio playback, so the cursor
crossing the spectrogram's bright region visually lines up with the word
actually playing. One video segment per trial, concatenated into a single
continuous video that shifts to a new word each time.

Design notes:
  - Uses the ALREADY-BINNED spike counts (features.npy) for the activity
    panel, not a true point-process raster -- this reuses data already
    computed by prepare_auditory_data.py with zero extra reprocessing.
    A true raster (individual spike dots) would require re-running
    bandpass filtering + threshold detection on raw .ns5 data for every
    trial; ask if you want that traded in for slower rendering.
  - Two-step render: matplotlib renders a silent video per trial via
    FFMpegWriter, then ffmpeg muxes that trial's real audio clip onto it,
    then all per-trial clips are concatenated with ffmpeg's concat
    demuxer. All trials use identical video encoding settings so the
    final stream-copy concatenation is valid.

Prerequisites: ffmpeg on PATH (already required for Whisper transcription
earlier in this pipeline).

Usage:
    # One example of each of the 5 words
    python3 render_word_trial_video.py \
        --subject Neil --region aSTG \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv" \
        --words tree,good,north,cricket,program --n-per-word 1 \
        --output word_trials.mp4
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.io import wavfile

from pipeline_paths import PipelinePaths, DEFAULT_DATA_ROOT
from prepare_auditory_data import BIN_MS

SAMPLING_RATE = 30000.0


def load_epochs(log_csv, words, n_per_word):
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))

    selected = []
    for word in words:
        matches = [r for r in rows if r.get("verified_word") == word]
        if not matches:
            print(f"  WARNING: no verified epochs found for '{word}', skipping")
            continue
        selected.extend(matches[:n_per_word])
    return selected


def get_bin_offset(data_root, subject, region, date, session, session_order):
    """Cumulative bin offset for `session` within that date's concatenated
    features.npy/targets.npy, matching the same math used when the day's
    dataset was originally built."""
    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))
    paths = PipelinePaths(data_root, subject, region, date)
    offset = 0
    for s in session_order:
        if s == session:
            return offset, paths
        audio_raw = np.load(paths.prep / f"{s}_audio_raw.npy").flatten()
        offset += len(audio_raw) // bin_samples
    raise ValueError(f"Session {session} not found in session_order for {date}")


def render_trial(row, args, out_path):
    date, session = row["date"], row["session"]
    word = row["verified_word"]
    pad = int(args.pad_s * SAMPLING_RATE)
    start = max(0, int(row["start_sample"]) - pad)
    end = int(row["end_sample"]) + pad

    paths = PipelinePaths(args.data_root, args.subject, args.region, date)
    audio_raw = np.load(paths.prep / f"{session}_audio_raw.npy").flatten()
    end = min(len(audio_raw), end)
    clip_audio = audio_raw[start:end]
    duration_s = len(clip_audio) / SAMPLING_RATE

    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))

    # All sessions for this date, in the same sorted order used when the
    # day's dataset was built, to compute this session's cumulative bin
    # offset within the day's concatenated features.npy/targets.npy.
    all_sessions = sorted(p.stem for p in Path(args.raw_base_dir, date).glob("*.ns5"))
    all_sessions = [f"{date}_{s}" for s in all_sessions]
    offset = 0
    session_n_bins = None
    for s in all_sessions:
        s_audio = np.load(paths.prep / f"{s}_audio_raw.npy").flatten()
        n_bins_s = len(s_audio) // bin_samples
        if s == session:
            session_n_bins = n_bins_s
            break
        offset += n_bins_s

    features_day = np.load(paths.pipeline / "features.npy")
    targets_day = np.load(paths.pipeline / "targets.npy")
    bin_start = offset + max(0, start // bin_samples)
    bin_end = offset + min(session_n_bins, -(-end // bin_samples))
    mel_slice = targets_day[bin_start:bin_end]
    feat_slice = features_day[bin_start:bin_end]

    n_bins = mel_slice.shape[0]
    bin_times = np.arange(n_bins) * BIN_MS / 1000.0

    fig, (ax_mel, ax_spike) = plt.subplots(2, 1, figsize=(10, 6))
    ax_mel.pcolormesh(bin_times, np.arange(mel_slice.shape[1]), mel_slice.T,
                       cmap="magma", shading="auto")
    ax_mel.set_ylabel("Mel band")
    ax_mel.set_title(f"'{word}'  ({date} / {session})")

    ax_spike.pcolormesh(bin_times, np.arange(feat_slice.shape[1]), feat_slice.T,
                         cmap="viridis", shading="auto")
    ax_spike.set_ylabel("Channel")
    ax_spike.set_xlabel("Time (s)")

    cursor_mel = ax_mel.axvline(0, color="cyan", linewidth=2)
    cursor_spike = ax_spike.axvline(0, color="cyan", linewidth=2)
    plt.tight_layout()

    n_frames = max(1, int(duration_s * args.fps))

    def update(frame_idx):
        t = frame_idx / args.fps
        cursor_mel.set_xdata([t, t])
        cursor_spike.set_xdata([t, t])
        return cursor_mel, cursor_spike

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    writer = FFMpegWriter(fps=args.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

    silent_path = out_path.with_suffix(".silent.mp4")
    anim.save(str(silent_path), writer=writer)
    plt.close(fig)

    # Mux the real audio onto the silent video.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        peak = np.max(np.abs(clip_audio))
        clip_i16 = (clip_audio / (peak + 1e-8) * 32767).astype(np.int16)
        wavfile.write(tmp_wav.name, int(SAMPLING_RATE), clip_i16)

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
    print(f"  Rendered: {out_path} ({duration_s:.2f}s, '{word}')")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-base-dir", required=True,
                         help="Same directory used by prepare_auditory_data.py -- needed to "
                              "reconstruct each session's bin offset within its day's arrays.")
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--words", required=True, help="Comma-separated words to include")
    parser.add_argument("--n-per-word", type=int, default=1)
    parser.add_argument("--pad-s", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="word_trials.mp4")
    args = parser.parse_args()

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    epochs = load_epochs(args.log_csv, words, args.n_per_word)
    print(f"Rendering {len(epochs)} trial(s)...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="word_trial_render_"))
    trial_paths = []
    for i, row in enumerate(epochs):
        trial_path = tmp_dir / f"trial_{i:03d}_{row['verified_word']}.mp4"
        render_trial(row, args, trial_path)
        trial_paths.append(trial_path)

    concat_list = tmp_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in trial_paths:
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
