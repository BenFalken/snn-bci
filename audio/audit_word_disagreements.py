"""
audit_word_disagreements.py

Second-pass review, targeted only at epochs where you and Whisper's
original guess disagreed -- exactly where an "autopilot" mistake during
the first review pass would hide. Does NOT re-surface genuine silence/noise
epochs (Whisper had no guess there in the first place, so there's nothing
to have disagreed with).

A disagreement is any already-reviewed epoch where Whisper's original
guess was non-empty, but your final decision was either:
  - discarded ('x'),
  - corrected to a DIFFERENT known word, or
  - flagged as a novel word ('n:...').

For each flagged epoch, replays the clip and shows both Whisper's original
guess and your previous decision side by side, then lets you keep your
previous call, accept Whisper's original guess instead, or enter a fresh
correction. Updates word_human_verified_log.csv in place (marking each
audited row so it won't be re-flagged on a future audit run once you've
explicitly confirmed it either way), and rebuilds both datasets from the
corrected log when the audit completes.

Usage:
    python3 audit_word_disagreements.py \
        --subject Neil --region aSTG \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv"
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from pipeline_paths import PipelinePaths
from review_word_epochs import (
    TWENTY_WORD_VOCABULARY, play_clip, PLAYBACK_TMP, SAMPLING_RATE,
    build_both_datasets,
)


def find_disagreements(rows):
    flagged = []
    for row in rows:
        if row.get("audit_resolved") == "true":
            continue
        guess = row.get("whisper_guess", "")
        if not guess:
            continue  # nothing to have disagreed with
        verified = row.get("verified_word", "")
        novel = row.get("novel_word", "")
        if verified == "" or novel or (verified and verified != guess):
            flagged.append(row)
    return flagged


def prompt_for_audit_decision(guess, score, previous_verified, previous_novel):
    if previous_novel:
        previous_display = f"novel word '{previous_novel}'"
    elif previous_verified:
        previous_display = f"'{previous_verified}'"
    else:
        previous_display = "DISCARDED (noise/other)"

    while True:
        response = input(
            f"    Whisper originally guessed: '{guess}' (score={score})\n"
            f"    You previously marked this as: {previous_display}\n"
            f"    [ENTER=accept Whisper's guess ('{guess}'), 'k'=keep my previous decision, "
            f"type a different known word, 'n:<word>'=novel word, "
            f"'r'=replay, 'x'=discard, 'q'=quit & save] > "
        ).strip().lower()

        if response == "q":
            return None
        if response == "r":
            return "REPLAY"
        if response == "k":
            return "KEEP"
        if response == "":
            return (guess, "")
        if response == "x":
            return ("", "")
        if response.startswith("n:"):
            import re
            novel = re.sub(r"[^a-z]", "", response[2:])
            if not novel:
                print("    Give a word after 'n:', e.g. n:gloop")
                continue
            return ("", novel)

        import re
        tokens = re.sub(r"[^a-z\s]", "", response).split()
        if tokens and tokens[0] in TWENTY_WORD_VOCABULARY:
            return (tokens[0], "")
        print(f"    Not one of the 20 known words. Use 'n:<word>' for a real word "
              f"outside the list, 'k' to keep your previous decision, or 'x' to discard.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-root", default="pipeline_data")
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "audit_resolved" not in fieldnames:
        fieldnames.append("audit_resolved")
        for r in rows:
            r["audit_resolved"] = ""

    disagreements = find_disagreements(rows)
    print(f"{len(disagreements)} disagreement(s) to audit "
          f"(epochs where you and Whisper's original guess didn't match).")
    if not disagreements:
        print("Nothing to audit -- either none exist, or all have already been audited.")
        return

    audio_cache = {}
    quit_early = False

    for target_row in disagreements:
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
            decision = prompt_for_audit_decision(
                target_row["whisper_guess"], target_row["score"],
                target_row.get("verified_word", ""), target_row.get("novel_word", ""),
            )
            if decision == "REPLAY":
                continue
            break

        if decision is None:
            quit_early = True
            break

        if decision == "KEEP":
            new_verified, new_novel = target_row.get("verified_word", ""), target_row.get("novel_word", "")
        else:
            new_verified, new_novel = decision

        for r in rows:
            if (r["date"], r["session"], int(r["epoch_idx"])) == key:
                r["verified_word"] = new_verified
                r["novel_word"] = new_novel
                r["audit_resolved"] = "true"
                break

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    PLAYBACK_TMP.unlink(missing_ok=True)

    if quit_early:
        remaining = len(find_disagreements(rows))
        print(f"\nStopped early. {remaining} disagreement(s) still unaudited -- "
              f"rerun this same command to resume.")
        return

    print("\nAudit complete. Rebuilding both datasets from the corrected log...")
    build_both_datasets(args, log_path)


if __name__ == "__main__":
    main()
