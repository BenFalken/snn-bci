"""
Listen to each labelled word epoch whose automatic transcription heard a DIFFERENT stimulus word, and keep or correct
its label. Answers are written into word_human_verified_log.csv (a dated backup is made first) and the
words-plus-novel cohort is rebuilt.

    python3 ~/Downloads/word_label_check/review.py            (all unaudited collisions; --all adds the audited ones)
    python3 ~/Downloads/word_label_check/review.py --oo       EVERY epoch labelled with an "oo" word (good, goop, zoo,
                                                              cool, moo, bloop), grouped by label; "goo" and "blue" are valid labels

For each clip (it plays twice: the word is spoken twice per epoch):
    Enter           keep the label
    a word          relabel (one of the 20 vocabulary words, or bloop / pop / goop)
    r               replay
    x               exclude the epoch from both cohorts (clears its label)
    q               stop here (answers so far are kept)
"""
import argparse
import csv
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import AUDITORY_REPO, AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)

import numpy as np

REPO = AUDITORY_REPO
B = AUDITORY_DATA / "Neil/aSTG"
LOG = B / "word_human_verified_log.csv"
VOCAB = ['error', 'good', 'macaque', 'zoo', 'program', 'tree', 'reflection', 'victory', 'two', 'apple', 'north',
         'banana', 'quality', 'laboratory', 'cricket', 'cool', 'window', 'sequence', 'fuel', 'moo']
NOVEL = ['bloop', 'pop', 'goop', 'goo', 'blue']


def words(s):
    return set(re.sub(r'[^a-z ]', '', (s or '').lower()).split())


def play(row, tmp):
    f = B / row["date"] / "prep" / f"{row['session']}_audio_raw.npy"
    a = np.load(f, mmap_mode="r")[int(row["start_sample"]):int(row["end_sample"])].astype(np.float32)
    a = a / (np.abs(a).max() + 1e-9) * 0.9
    w = Path(tmp) / "clip.wav"
    with wave.open(str(w), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(30000)
        h.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["afplay", str(w)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also the epochs your earlier audit already resolved")
    ap.add_argument("--oo", action="store_true", help='every epoch labelled with an "oo" word, flagged or not')
    ap.add_argument("--start", type=int, default=1, help="resume at this clip number")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(LOG)))
    fields = list(rows[0].keys())
    todo = []
    for i, r in enumerate(rows):
        label = r["verified_word"] or r["novel_word"]
        if not label:
            continue
        heard = (words(r["transcription"]) | words(r["whisper_guess"])) & set(VOCAB + NOVEL) - {label}
        if a.oo:
            if "oo" in label:
                todo.append((i, label, sorted(heard)[0] if heard else (r["transcription"] or "-")))
        elif heard and (a.all or r["audit_resolved"].lower() != "true"):
            todo.append((i, label, sorted(heard)[0]))
    if a.oo:
        todo.sort(key=lambda x: (x[1], rows[x[0]]["date"], int(rows[x[0]]["epoch_idx"])))
    todo = todo[a.start - 1:]
    print(f"{len(todo)} epochs to review" + (f" (from clip {a.start})" if a.start > 1 else "") + ". Enter = keep, a word = relabel, r = replay, x = exclude, q = stop.\n")
    changes = {}
    tmp = tempfile.mkdtemp()
    for n, (i, label, heard) in enumerate(todo, 1):
        r = rows[i]
        print(f"[{n}/{len(todo)}] {r['date']} {r['session']} epoch {r['epoch_idx']}: labelled '{label}', "
              f"transcription heard '{heard}'")
        play(r, tmp)
        while True:
            ans = input("   keep / word / r / x / q > ").strip().lower()
            if ans == "r":
                play(r, tmp)
                continue
            if ans == "q":
                break
            if ans in ("", "x") or ans in VOCAB + NOVEL:
                break
            print(f"   '{ans}' is not a stimulus word ({', '.join(VOCAB + NOVEL)})")
        if ans == "q":
            break
        if ans and ans != label:
            changes[i] = ans
            print(f"   -> {'excluded' if ans == 'x' else ans}")
    if not changes:
        print("\nno changes: every label kept.")
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        shutil.copy(LOG, LOG.with_name(f"word_human_verified_log.backup_{stamp}.csv"))
        for i, new in changes.items():
            r = rows[i]
            r["verified_word"] = new if new in VOCAB else ""
            r["novel_word"] = new if new in NOVEL else ""
            r["audit_resolved"] = "true"
        with open(LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(changes)} label(s) changed in {LOG.name} (backup: word_human_verified_log.backup_{stamp}.csv)")
    print("rebuilding ALLWORDS_NOVEL_DAYNORM ...")
    subprocess.run([sys.executable, "build_word_dataset_daynorm.py", "--subject", "Neil", "--region", "aSTG",
                    "--include-novel", "--keep-split-from",
                    "pipeline_data/Neil/aSTG/ALLWORDS_VERIFIED_DAYNORM/prep/epochs.csv",
                    "--name", "ALLWORDS_NOVEL_DAYNORM"], cwd=REPO, check=True)
    with open(Path(__file__).with_name("review_result.txt"), "w") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} reviewed {len(todo)} epochs, changed {len(changes)}\n")
        for i, new in changes.items():
            r = rows[i]
            f.write(f"{r['date']} {r['session']} epoch {r['epoch_idx']} -> {new}\n")
    print("done. Tell Claude when you're finished.")


if __name__ == "__main__":
    main()
