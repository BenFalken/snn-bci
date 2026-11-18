"""
build_five_word_dataset.py

Reconstructs the paper's main 5-word decoding condition (Heelan et al. 2019:
"tree", "good", "north", "cricket", "program") from already-processed daily
data, by:

  1. Re-detecting speech epochs from each session's ainp8 channel (the same
     QC-only mask used in prepare_auditory_data.py -- still never used to
     build features/targets directly, only to locate candidate word events).
  2. Transcribing each epoch's audio with Whisper and fuzzy-matching the
     result against the 5-word vocabulary.
  3. Keeping only the neural feature / mel-target bins, and the audio clips,
     for epochs that matched -- discarding everything else (macaque calls,
     unmatched/misheard epochs, inter-word silence).
  4. Concatenating the retained epochs into a new combined dataset, saved
     under a synthetic "date" (matching combine_days.py's convention), so
     the existing run_auditory_experiment.py / analyze_results.py machinery
     works on it unchanged. Also writes one combined WAV file of just the
     retained word clips.

IMPORTANT: Whisper is built for continuous speech; isolated single-word
clips are a harder case and some misrecognition is expected. Every epoch's
transcription (matched or not) is logged to a CSV -- inspect it before
trusting the filtered dataset for training.

Prerequisites (host machine):
    pip install openai-whisper
    macOS: brew install ffmpeg   |   Linux: apt install ffmpeg

Usage:
    python3 build_five_word_dataset.py \
        --subject Neil --region aSTG \
        --raw-base-dir "/Users/you/.../aSTG" \
        --dates 07_19_11,07_20_10,07_31_13,08_03_10,08_06_13
"""

import argparse
import csv
import difflib
import re
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pipeline_paths import PipelinePaths
from prepare_auditory_data import load_session, extract_speech_epochs, BIN_MS

VOCABULARY = ["tree", "good", "north", "cricket", "program"]
SAMPLING_RATE = 30000.0  # confirmed uniform across sessions via check_session_sampling_rates.py
SILENCE_GAP_S = 0.2      # inserted between concatenated clips in the output WAV, for listenability


def clean_text(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9\s]", "", text)


def best_vocabulary_match(transcription, vocabulary=VOCABULARY):
    """Fuzzy-matches every word in the transcription against the vocabulary,
    returns (best_word, best_score) using difflib's ratio (0-1)."""
    tokens = clean_text(transcription).split()
    if not tokens:
        return None, 0.0

    best_word, best_score = None, 0.0
    for token in tokens:
        for vocab_word in vocabulary:
            score = difflib.SequenceMatcher(None, token, vocab_word).ratio()
            if score > best_score:
                best_word, best_score = vocab_word, score
    return best_word, best_score


def transcribe_clip(model, clip, fs, initial_prompt):
    """Whisper's array API expects 16kHz; writing to a temp WAV and letting
    Whisper's own ffmpeg-based loader resample is the more robust, better-
    tested path than manual resampling."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        peak = np.max(np.abs(clip))
        clip_i16 = (clip / (peak + 1e-8) * 32767).astype(np.int16)
        wavfile.write(tmp.name, int(fs), clip_i16)
        result = model.transcribe(tmp.name, initial_prompt=initial_prompt, fp16=False)
    return result["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default="pipeline_data")
    parser.add_argument("--raw-base-dir", required=True,
                         help="Same directory passed to prepare_auditory_data.py -- "
                              "needed to re-read each session's ainp8 channel.")
    parser.add_argument("--dates", required=True,
                         help="Comma-separated dates already processed by prepare_auditory_data.py")
    parser.add_argument("--whisper-model", default="base.en",
                         help="Whisper model size. tiny.en/base.en are fast; small.en/medium.en "
                              "are slower but more accurate.")
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                         help="Minimum fuzzy-match score (0-1) to accept a transcription as a vocabulary word")
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    import whisper
    print(f"Loading Whisper model '{args.whisper_model}'...")
    model = whisper.load_model(args.whisper_model)
    initial_prompt = " ".join(VOCABULARY)

    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))

    all_matched_features, all_matched_targets, all_matched_audio = [], [], []
    log_rows = []
    match_counts = {w: 0 for w in VOCABULARY}

    for date in [d.strip() for d in args.dates.split(",") if d.strip()]:
        paths = PipelinePaths(args.data_root, args.subject, args.region, date)
        features_day = np.load(paths.pipeline / "features.npy")
        targets_day = np.load(paths.pipeline / "targets.npy")

        sessions = sorted(Path(args.raw_base_dir, date).glob("*.ns5"))
        bin_offset = 0

        for session_path in sessions:
            session_tag = f"{date}_{session_path.stem}"
            audio_raw_path = paths.prep / f"{session_tag}_audio_raw.npy"
            if not audio_raw_path.exists():
                print(f"  WARNING: {audio_raw_path} not found, skipping session {session_tag}")
                continue
            audio_raw = np.load(audio_raw_path).flatten()

            print(f"\n{session_tag}: re-detecting speech epochs...")
            _, ainp8, _, fs = load_session(session_path)
            epochs = extract_speech_epochs(ainp8)
            print(f"  {len(epochs)} candidate epochs")

            n_bins_session = len(audio_raw) // bin_samples

            for i, (start_sample, end_sample) in enumerate(epochs):
                pad = int(args.pad_s * SAMPLING_RATE)
                clip_start = max(0, start_sample - pad)
                clip_end = min(len(audio_raw), end_sample + pad)
                clip = audio_raw[clip_start:clip_end]

                transcription = transcribe_clip(model, clip, SAMPLING_RATE, initial_prompt)
                matched_word, score = best_vocabulary_match(transcription)
                is_match = matched_word is not None and score >= args.similarity_threshold

                log_rows.append({
                    "date": date, "session": session_tag, "epoch_idx": i,
                    "start_sample": start_sample, "end_sample": end_sample,
                    "transcription": transcription.strip(),
                    "matched_word": matched_word if is_match else "",
                    "score": round(score, 3), "accepted": is_match,
                })

                if is_match:
                    match_counts[matched_word] += 1
                    bin_start_local = max(0, clip_start // bin_samples)
                    bin_end_local = min(n_bins_session, -(-clip_end // bin_samples))
                    global_start = bin_offset + bin_start_local
                    global_end = bin_offset + bin_end_local

                    all_matched_features.append(features_day[global_start:global_end])
                    all_matched_targets.append(targets_day[global_start:global_end])
                    all_matched_audio.append(clip)
                    all_matched_audio.append(np.zeros(int(SILENCE_GAP_S * SAMPLING_RATE), dtype=clip.dtype))

                    print(f"  epoch {i}: '{transcription.strip()}' -> {matched_word} (score={score:.2f}) [KEPT]")
                else:
                    label = matched_word or "no match"
                    print(f"  epoch {i}: '{transcription.strip()}' -> {label} (score={score:.2f}) [discarded]")

            bin_offset += n_bins_session

    if not all_matched_features:
        print("\nNo epochs matched the vocabulary -- nothing to save. Check the log for what Whisper heard.")
        return

    combined_features = np.concatenate(all_matched_features, axis=0)
    combined_targets = np.concatenate(all_matched_targets, axis=0)
    combined_audio = np.concatenate(all_matched_audio, axis=0)

    output_date = "FIVEWORD_" + "_".join(args.dates.split(","))
    if len(output_date) > 60:  # keep the pseudo-date folder name manageable for many dates
        output_date = f"FIVEWORD_{len(args.dates.split(','))}days"
    out_paths = PipelinePaths(args.data_root, args.subject, args.region, output_date)

    np.save(out_paths.pipeline / "features.npy", combined_features)
    np.save(out_paths.pipeline / "targets.npy", combined_targets)

    wav_path = out_paths.results / "five_word_combined.wav"
    wavfile.write(wav_path, int(SAMPLING_RATE), (combined_audio * 32767).astype(np.int16))

    log_path = out_paths.results / "five_word_transcription_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    n_total = len(log_rows)
    n_accepted = sum(1 for r in log_rows if r["accepted"])
    print(f"\n{'=' * 60}")
    print(f"Total epochs examined: {n_total}")
    print(f"Accepted (matched vocabulary): {n_accepted} ({100*n_accepted/n_total:.1f}%)")
    print("Per-word counts:")
    for word, count in match_counts.items():
        print(f"  {word}: {count}")
    print(f"\nCombined features.npy: {combined_features.shape}")
    print(f"Combined targets.npy:  {combined_targets.shape}")
    print(f"Saved to: {out_paths.pipeline}")
    print(f"Combined WAV: {wav_path}")
    print(f"Full transcription log (INSPECT THIS): {log_path}")
    print(
        f"\nIMPORTANT: review {log_path.name} before trusting this dataset -- "
        f"Whisper on isolated single words can misrecognize or hallucinate. "
        f"Look especially at low-score accepted matches and any word whose "
        f"count looks implausibly high or low relative to the others."
    )
    print(f"\nNext steps:")
    print(f"  python3 run_auditory_experiment.py --subject {args.subject} --region {args.region} "
          f"--date {output_date} --all-decoders")
    print(f"  python3 analyze_results.py --mode single --subject {args.subject} --region {args.region} "
          f"--date {output_date}")


if __name__ == "__main__":
    main()
