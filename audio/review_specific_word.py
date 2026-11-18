"""
review_specific_word.py

Targeted re-review of every epoch currently labeled as a specific word --
useful for checking a suspiciously overrepresented word (e.g. "good") for
a hidden merge with a similar-sounding but distinct word (e.g. "goop").

Resumable by construction: once a row's label changes away from the target
word, it naturally drops out of future runs with the same --word, since the
filter is just "verified_word == args.word" each time. Rows you explicitly
re-confirm as correct will show up again if you quit and rerun -- a quick
ENTER to re-confirm is low-friction, so this wasn't worth a separate
tracking column for what's meant to be a one-off targeted check.

Usage:
    python3 review_specific_word.py \
        --subject Neil --region aSTG --word good \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv"
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from pipeline_paths import PipelinePaths, DEFAULT_DATA_ROOT
from review_word_epochs import TWENTY_WORD_VOCABULARY, play_clip, PLAYBACK_TMP, SAMPLING_RATE


def prompt_for_recheck(word, guess, score):
    while True:
        response = input(
            f"    Currently labeled: '{word}'  |  Whisper originally guessed: "
            f"'{guess}' (score={score})\n"
            f"    [ENTER=keep as '{word}', type a different known word, "
            f"'n:<word>'=novel word (e.g. n:goop), 'r'=replay, 'x'=discard, 'q'=quit & save] > "
        ).strip().lower()

        if response == "q":
            return None
        if response == "r":
            return "REPLAY"
        if response == "":
            return (word, "")
        if response == "x":
            return ("", "")
        if response.startswith("n:"):
            novel = re.sub(r"[^a-z]", "", response[2:])
            if not novel:
                print("    Give a word after 'n:', e.g. n:goop")
                continue
            return ("", novel)

        tokens = re.sub(r"[^a-z\s]", "", response).split()
        if tokens and tokens[0] in TWENTY_WORD_VOCABULARY:
            return (tokens[0], "")
        print(f"    Not one of the 20 known words. Use 'n:<word>' for a real word "
              f"outside the list, or 'x' to discard.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--word", required=True, help="Word to re-review, e.g. 'good'")
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    targets = [r for r in rows if r.get("verified_word") == args.word]
    print(f"{len(targets)} epoch(s) currently labeled '{args.word}' to re-check.")
    if not targets:
        return

    audio_cache = {}
    quit_early = False
    n_changed = 0

    for target_row in targets:
        key = (target_row["date"], target_row["session"], int(target_row["epoch_idx"]))
        cache_key = (target_row["date"], target_row["session"])
        if cache_key not in audio_cache:
            paths = PipelinePaths(args.data_root, args.subject, args.region, target_row["date"])
            audio_path = paths.prep / f"{target_row['session']}_audio_raw.npy"
            audio_cache[cache_key] = np.load(audio_path).flatten()
        audio_raw = audio_cache[cache_key]

        pad = int(args.pad_s * SAMPLING_RATE)
        start = max(0, int(target_row["start_sample"]) - pad)
        end = min(len(audio_raw), int(target_row["end_sample"]) + pad)
        clip = audio_raw[start:end]

        print(f"\n[{target_row['date']} / {target_row['session']} / epoch {target_row['epoch_idx']}]")
        while True:
            play_clip(clip)
            decision = prompt_for_recheck(args.word, target_row["whisper_guess"], target_row["score"])
            if decision == "REPLAY":
                continue
            break

        if decision is None:
            quit_early = True
            break

        new_verified, new_novel = decision
        if new_verified != args.word or new_novel:
            n_changed += 1

        for r in rows:
            if (r["date"], r["session"], int(r["epoch_idx"])) == key:
                r["verified_word"] = new_verified
                r["novel_word"] = new_novel
                break

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    PLAYBACK_TMP.unlink(missing_ok=True)

    if quit_early:
        remaining = sum(1 for r in rows if r.get("verified_word") == args.word)
        print(f"\nStopped early. {remaining} epoch(s) still labeled '{args.word}' "
              f"(includes both unreviewed and re-confirmed ones) -- rerun to continue.")
        return

    print(f"\nDone. Changed {n_changed} of {len(targets)} epoch(s).")
    print(f"Rebuild both datasets to pick up any changes:")
    print(f'  python3 -c "')
    print(f"from review_word_epochs import build_both_datasets")
    print(f"import argparse")
    print(f"args = argparse.Namespace(subject='{args.subject}', region='{args.region}', "
          f"data_root='{args.data_root}', pad_s={args.pad_s})")
    print(f"build_both_datasets(args, '{log_path}')")
    print(f'"')


if __name__ == "__main__":
    main()
