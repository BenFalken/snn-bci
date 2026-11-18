"""
Monkey Gomez (aSTG): the listening pass over Whisper's transcription of every detected sound epoch, producing
pipeline_data/Gomez/aSTG/word_human_verified_log.csv in the same format as Neil's, then Gomez's cohort
ALLWORDS_NOVEL_DAYNORM (the 20 words plus bloop / pop / goop, day-normalized, 80/10/10 by utterance).

    python3 ~/Downloads/word_label_check/gomez_review.py              the shortlist (see below), resumable
    python3 ~/Downloads/word_label_check/gomez_review.py --confident  also listen to the confident matches
    python3 ~/Downloads/word_label_check/gomez_review.py --empty      also the epochs Whisper heard nothing in

Queue: (1) every epoch whose transcription is not empty but is not a clean match (uncertain matches, near-misses,
anything that may be bloop / pop / goop), (2) with --confident, the clean matches (Whisper heard exactly a vocabulary
word). Without --confident the clean matches are accepted as Whisper heard them. Empty transcriptions (silence,
static, macaque calls) are left out unless --empty.

For each clip:
    Enter    accept the suggested word (shown in brackets; no suggestion = discard)
    a word   label it (the 20 vocabulary words, or bloop / pop / goop)
    x        discard (not a stimulus word: noise, a call, a fragment)
    r        replay
    q        stop; every answer is saved as you go, and the next run resumes where you stopped
"""
import argparse
import csv
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import AUDIO, AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)

import numpy as np

REPO = AUDIO                     # the cohort builder runs from the repository's audio code
B = AUDITORY_DATA / "Gomez/aSTG"
SRC = B / "word_transcription_log.csv"
OUT = B / "word_human_verified_log.csv"
VOCAB = ['error', 'good', 'macaque', 'zoo', 'program', 'tree', 'reflection', 'victory', 'two', 'apple', 'north',
         'banana', 'quality', 'laboratory', 'cricket', 'cool', 'window', 'sequence', 'fuel', 'moo']
NOVEL = ['bloop', 'pop', 'goop']
FIELDS = ["date", "session", "epoch_idx", "start_sample", "end_sample", "transcription", "whisper_guess", "score",
          "verified_word", "novel_word", "audit_resolved"]


def play(r, tmp):
    a = np.load(B / r["date"] / "prep" / f"{r['session']}_audio_raw.npy", mmap_mode="r")
    a = a[int(r["start_sample"]):int(r["end_sample"])].astype(np.float32)
    a = a / (np.abs(a).max() + 1e-9) * 0.9
    w = Path(tmp) / "clip.wav"
    with wave.open(str(w), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(30000)
        h.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["afplay", str(w)])


def clean(r):
    """Whisper heard exactly one vocabulary word (said once or twice), nothing else."""
    w = set(r["transcription"].lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").split())
    return r["accepted"] == "True" and float(r["score"] or 0) >= 0.99 and w == {r["matched_word"]}


def save(rows):
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confident", action="store_true")
    ap.add_argument("--empty", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    src = list(csv.DictReader(open(SRC)))
    done = {}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            done[(r["date"], r["session"], r["epoch_idx"])] = r
    rows, todo = [], []
    for s in src:
        k = (s["date"], s["session"], s["epoch_idx"])
        r = done.get(k) or {"date": s["date"], "session": s["session"], "epoch_idx": s["epoch_idx"],
                            "start_sample": s["start_sample"], "end_sample": s["end_sample"],
                            "transcription": s["transcription"], "whisper_guess": s["matched_word"],
                            "score": s["score"], "verified_word": "", "novel_word": "", "audit_resolved": ""}
        rows.append(r)
        if r["audit_resolved"] == "true":                       # already listened to
            continue
        empty = not s["transcription"].strip()
        if clean(s) and not a.confident:
            r["verified_word"] = s["matched_word"]               # accepted as heard, not listened to
            continue
        if empty and not a.empty:
            continue
        todo.append((len(rows) - 1, s["matched_word"] if s["accepted"] == "True" else ""))
    save(rows)
    print(f"{len(todo)} clips to listen to. Enter = accept [suggestion], a word = label, x = discard, r = replay, "
          f"q = stop (saved as you go).\n")
    tmp = tempfile.mkdtemp()
    for n, (i, sug) in enumerate(todo, 1):
        r = rows[i]
        print(f"[{n}/{len(todo)}] {r['date']} {r['session'][-10:]} epoch {r['epoch_idx']}: Whisper heard "
              f"'{r['transcription'].strip() or '(nothing)'}'  [{sug or 'discard'}]")
        play(r, tmp)
        while True:
            ans = input("   > ").strip().lower()
            if ans == "r":
                play(r, tmp)
                continue
            if ans in ("q", "x", "") or ans in VOCAB + NOVEL:
                break
            print(f"   '{ans}' is not a stimulus word ({', '.join(VOCAB + NOVEL)})")
        if ans == "q":
            break
        word = sug if ans == "" else ("" if ans == "x" else ans)
        r["verified_word"] = word if word in VOCAB else ""
        r["novel_word"] = word if word in NOVEL else ""
        r["audit_resolved"] = "true"
        save(rows)
    left = sum(1 for i, _ in todo if rows[i]["audit_resolved"] != "true")
    n_word = sum(1 for r in rows if r["verified_word"] or r["novel_word"])
    print(f"\nsaved {OUT.name}: {n_word} labelled epochs" + (f"; {left} clips still to go (run again to resume)"
                                                           if left else ""))
    if left or a.no_build:
        return
    print("building Gomez's ALLWORDS_NOVEL_DAYNORM ...")
    subprocess.run([sys.executable, "build_word_dataset_daynorm.py", "--subject", "Gomez", "--region", "aSTG",
                    "--include-novel", "--name", "ALLWORDS_NOVEL_DAYNORM"], cwd=REPO, check=True)
    print("done. Tell Claude when you're finished.")


if __name__ == "__main__":
    main()
