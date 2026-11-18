"""
fix_verified_word_schema.py

One-time cleanup: migrates any verified_word entries that aren't actually
in the 20-word vocabulary (leftover from before the review tool's
open-vocabulary phase was tightened) into novel_word, where they belong.

Usage:
    python3 fix_verified_word_schema.py \
        --log-csv "pipeline_data/Neil/aSTG/word_human_verified_log.csv"
"""

import argparse
import csv
from pathlib import Path

from build_word_dataset import TWENTY_WORD_VOCABULARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-csv", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    n_fixed = 0
    for r in rows:
        vw = r.get("verified_word", "")
        if vw and vw not in TWENTY_WORD_VOCABULARY:
            print(f"  Migrating '{vw}' (date={r['date']}, session={r['session']}, "
                  f"epoch={r['epoch_idx']}): verified_word -> novel_word")
            r["novel_word"] = vw
            r["verified_word"] = ""
            n_fixed += 1

    with open(log_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFixed {n_fixed} row(s).")


if __name__ == "__main__":
    main()
