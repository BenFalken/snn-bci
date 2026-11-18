"""
review_word_epochs.py

Generalized version of review_five_word_epochs.py -- a single human review
pass against the full 20-word vocabulary, producing TWO cleaned datasets
from one set of listening decisions: FIVEWORD_VERIFIED (the paper's main
condition) and TWENTYWORD_VERIFIED (their generalization condition). No
epoch is ever listened to twice for this purpose.

Reads the log produced by build_word_dataset.py, plays each clip from a
single reusable temp file (not accumulated on disk) via macOS's `afplay`,
shows Whisper's guess as a hint, and asks you to confirm, correct with any
of the 20 valid words, or discard (for noise, macaque calls, or any other
non-vocabulary sound). Fully resumable -- progress saves after every clip.

Usage:
    python3 review_word_epochs.py \
        --subject Neil --region aSTG \
        --log-csv "pipeline_data/Neil/aSTG/word_transcription_log.csv"
"""

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pipeline_paths import PipelinePaths, DEFAULT_DATA_ROOT
from build_word_dataset import TWENTY_WORD_VOCABULARY, FIVE_WORD_SUBSET

SAMPLING_RATE = 30000.0
SILENCE_GAP_S = 0.2
PLAYBACK_TMP = Path(tempfile.gettempdir()) / "word_review_playback.wav"


def play_clip(clip, fs=SAMPLING_RATE):
    peak = np.max(np.abs(clip))
    clip_i16 = (clip / (peak + 1e-8) * 32767).astype(np.int16)
    wavfile.write(PLAYBACK_TMP, int(fs), clip_i16)
    result = subprocess.run(["afplay", str(PLAYBACK_TMP)], capture_output=True)
    if result.returncode != 0:
        print("  (afplay failed -- if you're not on macOS, swap this for `aplay`/`paplay`)")


def load_progress(output_csv):
    reviewed = {}
    if output_csv.exists():
        with open(output_csv) as f:
            for row in csv.DictReader(f):
                key = (row["date"], row["session"], int(row["epoch_idx"]))
                reviewed[key] = row
    return reviewed


def prompt_for_label(transcription, guess, score):
    """Returns (verified_word, novel_word). At most one is non-empty:
    verified_word is set for an accept/correction against the known
    20-word vocabulary; novel_word is set for an explicitly-flagged
    out-of-vocabulary real word (n:<word>); both empty means discard."""
    guess_display = guess if guess else "(no confident guess)"
    while True:
        response = input(
            f"    Whisper: '{transcription}' -> guess: {guess_display} (score={score})\n"
            f"    [ENTER=correct, type one of the 20 words=wrong/correction, "
            f"'n:<word>'=real word not in the 20 (e.g. n:gloop), "
            f"'r'=replay, 'x'=discard (noise/macaque call), 'q'=quit & save] > "
        ).strip().lower()

        if response == "q":
            return None, None
        if response == "r":
            return "REPLAY", None
        if response == "":
            if guess:
                return guess, ""
            print("    No guess to accept -- type a word, 'n:<word>', or 'x' to discard.")
            continue
        if response == "x":
            return "", ""

        if response.startswith("n:"):
            novel = re.sub(r"[^a-z]", "", response[2:])
            if not novel:
                print("    Give a word after 'n:', e.g. n:gloop")
                continue
            return "", novel

        tokens = re.sub(r"[^a-z\s]", "", response).split()
        if tokens and tokens[0] in TWENTY_WORD_VOCABULARY:
            return tokens[0], ""

        print(f"    Not one of the 20 known words. Use 'n:<word>' if it's a real word "
              f"outside the list, or 'x' to discard.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--review-mode", choices=["all", "rejected_only"], default="all")
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    if args.review_mode == "rejected_only":
        rows = [r for r in rows if r["accepted"].lower() != "true"]

    output_csv = log_path.parent / "word_human_verified_log.csv"
    reviewed = load_progress(output_csv)
    print(f"{len(rows)} epochs to review, {len(reviewed)} already reviewed in a previous session.")

    fieldnames = ["date", "session", "epoch_idx", "start_sample", "end_sample",
                  "transcription", "whisper_guess", "score", "verified_word", "novel_word"]
    if not output_csv.exists():
        with open(output_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    audio_cache = {}
    quit_early = False

    for row in rows:
        key = (row["date"], row["session"], int(row["epoch_idx"]))
        if key in reviewed:
            continue

        cache_key = (row["date"], row["session"])
        if cache_key not in audio_cache:
            paths = PipelinePaths(args.data_root, args.subject, args.region, row["date"])
            audio_path = paths.prep / f"{row['session']}_audio_raw.npy"
            audio_cache[cache_key] = np.load(audio_path).flatten()
        audio_raw = audio_cache[cache_key]

        pad = int(args.pad_s * SAMPLING_RATE)
        start = max(0, int(row["start_sample"]) - pad)
        end = min(len(audio_raw), int(row["end_sample"]) + pad)
        clip = audio_raw[start:end]

        print(f"\n[{row['date']} / {row['session']} / epoch {row['epoch_idx']}]")
        while True:
            play_clip(clip)
            verified_word, novel_word = prompt_for_label(row["transcription"], row["matched_word"], row["score"])
            if verified_word == "REPLAY":
                continue
            break

        if verified_word is None:
            quit_early = True
            break

        with open(output_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "date": row["date"], "session": row["session"], "epoch_idx": row["epoch_idx"],
                "start_sample": row["start_sample"], "end_sample": row["end_sample"],
                "transcription": row["transcription"], "whisper_guess": row["matched_word"],
                "score": row["score"], "verified_word": verified_word, "novel_word": novel_word,
            })
        reviewed[key] = {"verified_word": verified_word, "novel_word": novel_word}

    PLAYBACK_TMP.unlink(missing_ok=True)

    if quit_early:
        print(f"\nStopped early. Progress saved to {output_csv} -- rerun this same command to resume.")
        return

    print(f"\nReview complete. Building both datasets from {output_csv}...")
    build_both_datasets(args, output_csv)


def build_dataset_from_rows(args, rows, output_date):
    if not rows:
        print(f"  No matching rows for {output_date} -- skipping.")
        return

    from prepare_auditory_data import BIN_MS
    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))

    dates = sorted(set(r["date"] for r in rows))
    all_features, all_targets, all_audio = [], [], []
    match_counts = {}

    for date in dates:
        paths = PipelinePaths(args.data_root, args.subject, args.region, date)
        features_day = np.load(paths.pipeline / "features.npy")
        targets_day = np.load(paths.pipeline / "targets.npy")

        sessions_this_date = sorted(set(r["session"] for r in rows if r["date"] == date))
        bin_offset = 0
        for session in sessions_this_date:
            audio_raw = np.load(paths.prep / f"{session}_audio_raw.npy").flatten()
            n_bins_session = len(audio_raw) // bin_samples

            session_rows = [r for r in rows if r["date"] == date and r["session"] == session]
            for r in session_rows:
                pad = int(args.pad_s * SAMPLING_RATE)
                start = max(0, int(r["start_sample"]) - pad)
                end = min(len(audio_raw), int(r["end_sample"]) + pad)

                bin_start_local = max(0, start // bin_samples)
                bin_end_local = min(n_bins_session, -(-end // bin_samples))
                global_start = bin_offset + bin_start_local
                global_end = bin_offset + bin_end_local

                all_features.append(features_day[global_start:global_end])
                all_targets.append(targets_day[global_start:global_end])
                all_audio.append(audio_raw[start:end])
                all_audio.append(np.zeros(int(SILENCE_GAP_S * SAMPLING_RATE)))
                match_counts[r["verified_word"]] = match_counts.get(r["verified_word"], 0) + 1

            bin_offset += n_bins_session

    combined_features = np.concatenate(all_features, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)
    combined_audio = np.concatenate(all_audio, axis=0)

    out_paths = PipelinePaths(args.data_root, args.subject, args.region, output_date)
    np.save(out_paths.pipeline / "features.npy", combined_features)
    np.save(out_paths.pipeline / "targets.npy", combined_targets)
    wavfile.write(out_paths.results / "combined.wav", int(SAMPLING_RATE),
                  (combined_audio * 32767).astype(np.int16))

    print(f"\n{output_date}:")
    print(f"  features.npy: {combined_features.shape}  ({combined_features.shape[0]*0.04:.1f}s)")
    for word in sorted(match_counts):
        print(f"    {word}: {match_counts[word]}")
    print(f"  Saved to: {out_paths.pipeline}")


def build_both_datasets(args, verified_csv_path):
    with open(verified_csv_path) as f:
        all_rows = list(csv.DictReader(f))

    all_verified = [r for r in all_rows if r["verified_word"]]

    anomalies = [r for r in all_verified if r["verified_word"] not in TWENTY_WORD_VOCABULARY]
    if anomalies:
        print(f"\nWARNING: dropping {len(anomalies)} row(s) whose verified_word isn't actually "
              f"in the 20-word vocabulary (likely leftover from before the review tool's "
              f"vocabulary was tightened) -- these will NOT be included in either dataset:")
        for r in anomalies:
            print(f"    '{r['verified_word']}' (date={r['date']}, session={r['session']}, "
                  f"epoch={r['epoch_idx']}) -- run fix_verified_word_schema.py to migrate "
                  f"this into novel_word properly")
        all_verified = [r for r in all_verified if r["verified_word"] in TWENTY_WORD_VOCABULARY]
    novel_rows = [r for r in all_rows if r.get("novel_word")]

    if not all_verified and not novel_rows:
        print("No verified matches or novel words -- nothing to build.")
        return

    if novel_rows:
        novel_counts = {}
        for r in novel_rows:
            novel_counts[r["novel_word"]] = novel_counts.get(r["novel_word"], 0) + 1
        print(f"\n{len(novel_rows)} epoch(s) flagged as novel words NOT in the paper's 20-word "
              f"list -- logged for reference only, NOT included in either dataset below:")
        for word, count in sorted(novel_counts.items()):
            print(f"    '{word}': {count} occurrence(s)")

    if not all_verified:
        print("\nNo known-vocabulary matches to build datasets from.")
        return

    dates = sorted(set(r["date"] for r in all_verified))
    date_tag = "_".join(dates) if len(dates) <= 5 else f"{len(dates)}days"

    five_word_rows = [r for r in all_verified if r["verified_word"] in FIVE_WORD_SUBSET]
    build_dataset_from_rows(args, five_word_rows, f"FIVEWORD_VERIFIED_{date_tag}")
    build_dataset_from_rows(args, all_verified, f"ALLWORDS_VERIFIED_{date_tag}")

    print(f"\nNext steps, for each dataset:")
    for tag in [f"FIVEWORD_VERIFIED_{date_tag}", f"ALLWORDS_VERIFIED_{date_tag}"]:
        print(f"  python3 run_auditory_experiment.py --subject {args.subject} --region {args.region} "
              f"--date {tag} --all-decoders")
        print(f"  python3 analyze_results.py --mode single --subject {args.subject} --region {args.region} "
              f"--date {tag}")


if __name__ == "__main__":
    main()
