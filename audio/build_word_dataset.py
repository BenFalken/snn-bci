"""
build_word_dataset.py

Generalized version of build_five_word_dataset.py -- detects and
transcribes speech epochs against the FULL 20-word vocabulary used
anywhere in Heelan et al. (2019) (their 17-word training set plus 3
single-trial-only generalization words: "two", "cool", "victory"), so a
single pass can support both the paper's main 5-word condition and their
17/20-word generalization condition.

Note: "macaque" is itself one of the 17 vocabulary WORDS (the TTS system
says it as a stimulus) -- distinct from an actual macaque call (animal
vocalization), which is not a word and should be discarded during review.

Usage:
    python3 build_word_dataset.py \
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

from pipeline_paths import PipelinePaths, DEFAULT_DATA_ROOT
from prepare_auditory_data import load_session, extract_speech_epochs, BIN_MS

TWENTY_WORD_VOCABULARY = [
    "program", "macaque", "laboratory", "quality", "good", "tree", "moo",
    "apple", "banana", "cricket", "zoo", "fuel", "reflection", "north",
    "sequence", "window", "error", "two", "cool", "victory",
]
FIVE_WORD_SUBSET = {"tree", "good", "north", "cricket", "program"}

SAMPLING_RATE = 30000.0
SILENCE_GAP_S = 0.2


def clean_text(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9\s]", "", text)


def best_vocabulary_match(transcription, vocabulary=TWENTY_WORD_VOCABULARY):
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
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-base-dir", required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    import whisper
    print(f"Loading Whisper model '{args.whisper_model}'...")
    model = whisper.load_model(args.whisper_model)
    initial_prompt = " ".join(TWENTY_WORD_VOCABULARY)

    bin_samples = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))
    log_rows = []

    for date in [d.strip() for d in args.dates.split(",") if d.strip()]:
        paths = PipelinePaths(args.data_root, args.subject, args.region, date)
        sessions = sorted(Path(args.raw_base_dir, date).glob("*.ns5"))

        for session_path in sessions:
            session_tag = f"{date}_{session_path.stem}"
            audio_raw_path = paths.prep / f"{session_tag}_audio_raw.npy"
            if not audio_raw_path.exists():
                print(f"  WARNING: {audio_raw_path} not found, skipping {session_tag}")
                continue
            audio_raw = np.load(audio_raw_path).flatten()

            print(f"\n{session_tag}: re-detecting speech epochs...")
            _, ainp8, _, fs = load_session(session_path)
            epochs = extract_speech_epochs(ainp8)
            print(f"  {len(epochs)} candidate epochs")

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
                    "in_five_word_subset": is_match and matched_word in FIVE_WORD_SUBSET,
                })

                label = matched_word if is_match else (matched_word or "no match")
                tag = " [KEPT, 5-word]" if is_match and matched_word in FIVE_WORD_SUBSET else \
                      " [KEPT, 20-word only]" if is_match else " [discarded]"
                print(f"  epoch {i}: '{transcription.strip()}' -> {label} (score={score:.2f}){tag}")

    if not log_rows:
        print("\nNo epochs found across the given dates.")
        return

    out_dir = Path(args.data_root) / args.subject / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "word_transcription_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    n_total = len(log_rows)
    n_five = sum(1 for r in log_rows if r["in_five_word_subset"])
    n_twenty = sum(1 for r in log_rows if r["accepted"])
    print(f"\n{'=' * 60}")
    print(f"Total epochs examined: {n_total}")
    print(f"Matched somewhere in the 20-word vocabulary: {n_twenty} ({100*n_twenty/n_total:.1f}%)")
    print(f"  Of those, in the 5-word subset: {n_five}")
    print(f"\nLog saved to: {log_path}")
    print(f"\nNext step -- review it with review_word_epochs.py before trusting either dataset:")
    print(f"  python3 review_word_epochs.py --subject {args.subject} --region {args.region} "
          f"--log-csv {log_path}")


if __name__ == "__main__":
    main()
