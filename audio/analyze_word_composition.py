"""
analyze_word_composition.py

Quick summary of what's actually in a word_human_verified_log.csv --
counts and duration per known vocabulary word, per novel word, discard
rate, and a per-date breakdown. No dataset building, just a fast read on
composition before deciding what to do next.

Usage:
    python3 analyze_word_composition.py \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv"
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from build_word_dataset import TWENTY_WORD_VOCABULARY

SAMPLING_RATE = 30000.0
FIVE_WORD_SUBSET = {"tree", "good", "north", "cricket", "program"}


def clip_duration_s(row, pad_s):
    start = max(0, int(row["start_sample"]) - int(pad_s * SAMPLING_RATE))
    end = int(row["end_sample"]) + int(pad_s * SAMPLING_RATE)
    return (end - start) / SAMPLING_RATE


def print_table(counter, total_duration_by_word, title):
    if not counter:
        print(f"  (none)")
        return
    width = max(len(w) for w in counter) + 2
    for word, count in counter.most_common():
        dur = total_duration_by_word.get(word, 0.0)
        marker = " *" if word in FIVE_WORD_SUBSET else ""
        print(f"  {word:<{width}} {count:>4}   {dur:>7.1f}s{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--pad-s", type=float, default=0.3)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    discarded = [r for r in rows if not r.get("verified_word") and not r.get("novel_word")]
    known = [r for r in rows if r.get("verified_word")]
    novel = [r for r in rows if r.get("novel_word")]

    anomalies = [r for r in known if r["verified_word"] not in TWENTY_WORD_VOCABULARY]
    if anomalies:
        print("=" * 60)
        print(f"WARNING: {len(anomalies)} row(s) have a verified_word that isn't actually "
              f"in the 20-word vocabulary -- likely leftover from before the review tool's "
              f"vocabulary was tightened. Run fix_verified_word_schema.py before trusting "
              f"the counts below.")
        for r in anomalies:
            print(f"    '{r['verified_word']}' (date={r['date']}, session={r['session']}, "
                  f"epoch={r['epoch_idx']})")
        print()

    known_counts = Counter(r["verified_word"] for r in known)
    novel_counts = Counter(r["novel_word"] for r in novel)

    known_duration = {}
    for r in known:
        w = r["verified_word"]
        known_duration[w] = known_duration.get(w, 0.0) + clip_duration_s(r, args.pad_s)

    novel_duration = {}
    for r in novel:
        w = r["novel_word"]
        novel_duration[w] = novel_duration.get(w, 0.0) + clip_duration_s(r, args.pad_s)

    five_word_total = sum(known_counts[w] for w in known_counts if w in FIVE_WORD_SUBSET)
    five_word_duration = sum(known_duration[w] for w in known_duration if w in FIVE_WORD_SUBSET)
    known_total_duration = sum(known_duration.values())
    novel_total_duration = sum(novel_duration.values())

    print("=" * 60)
    print(f"Total epochs reviewed: {total}")
    print(f"  Discarded (noise/other):    {len(discarded):>4}  ({100*len(discarded)/total:.1f}%)")
    print(f"  Known vocabulary matches:   {len(known):>4}  ({100*len(known)/total:.1f}%)")
    print(f"  Flagged as novel words:     {len(novel):>4}  ({100*len(novel)/total:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Known vocabulary ({len(known)} epochs, {known_total_duration:.1f}s total)")
    print(f"  ('*' marks the 5-word subset)")
    print("-" * 60)
    print_table(known_counts, known_duration, "known")
    print(f"\n  5-word subset only: {five_word_total} epochs, {five_word_duration:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"Novel (out-of-paper-vocabulary) words ({len(novel)} epochs, {novel_total_duration:.1f}s total)")
    print("-" * 60)
    print_table(novel_counts, novel_duration, "novel")

    print(f"\n{'=' * 60}")
    print("Per-date breakdown")
    print("-" * 60)
    dates = sorted(set(r["date"] for r in rows))
    for date in dates:
        date_rows = [r for r in rows if r["date"] == date]
        date_known = sum(1 for r in date_rows if r.get("verified_word"))
        date_novel = sum(1 for r in date_rows if r.get("novel_word"))
        date_discarded = len(date_rows) - date_known - date_novel
        print(f"  {date:<15} total={len(date_rows):>4}  known={date_known:>4}  "
              f"novel={date_novel:>4}  discarded={date_discarded:>4}")


if __name__ == "__main__":
    main()
