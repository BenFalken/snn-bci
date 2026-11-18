"""
build_word_dataset_daynorm.py

Builds ALLWORDS_VERIFIED_DAYNORM: every human-verified word epoch (word_human_verified_log.csv, the 20-word
vocabulary; macaque calls, chirps, novel sounds and unlabelled clips are left out), with

  - per-day normalization of the neural features: each day's features are z-scored per channel with statistics from
    that day's TRAIN epochs only (frozen, then applied to its valid/test epochs too, so it stays deployable), and
    clipped at +/-5 SD, as combine_days.py --mode per_day_normalized_causal. The mel targets are NOT normalized: the
    stimuli are the same synthetic words every day, so their spectrograms already share one scale, and the decoded
    audio stays invertible.
  - a shuffled, word-stratified split by whole epochs: per word, 10% of its epochs to test and 10% to validation
    (at least one each), drawn at random (seed 1337) across days, the rest to train. The epochs are written train,
    then valid, then test, and a few frames of silent padding are trimmed from the ends of epochs so
    that the pipeline's 80/10/10 split by position falls exactly on the epoch boundaries (no epoch is shared by two
    splits).

Written under pipeline_data/<subject>/<region>/ALLWORDS_VERIFIED_DAYNORM/:
  pipeline/features.npy          per-day z-scored, clipped features (every pipeline decoder and the LSNN)
  pipeline/features_counts_daynorm.npy   spike counts divided by each day's train-epoch SD per channel, never
                                 negative (the Speck SNN, which takes events; a host could apply this gain)
  pipeline/targets.npy           mel targets as recorded
  prep/epochs.csv                one row per epoch: split, word, date, session, epoch_idx, first and last frame

Usage:
    python3 build_word_dataset_daynorm.py --subject Neil --region aSTG
then
    python3 run_auditory_experiment.py --subject Neil --region aSTG --date ALLWORDS_VERIFIED_DAYNORM --all-decoders
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from build_word_dataset import TWENTY_WORD_VOCABULARY
from pipeline_paths import PipelinePaths, DEFAULT_DATA_ROOT
from prepare_auditory_data import BIN_MS

SAMPLING_RATE = 30000.0
TRAIN_DECIMAL, VALID_DECIMAL = 0.80, 0.10
SAMPLES_BEFORE = SAMPLES_AFTER = 8
NAME = "ALLWORDS_VERIFIED_DAYNORM"
MIN_HELD_OUT = 10       # a word with fewer epochs (e.g. the novel 'goo', 'blue') goes to train only


def epochs_from_log(args):
    """Every verified epoch as (row, day features, day targets, first frame, end frame), padded by --pad-s."""
    rows = []
    for r in csv.DictReader(open(f"{args.data_root}/{args.subject}/{args.region}/word_human_verified_log.csv")):
        if r["verified_word"] in TWENTY_WORD_VOCABULARY:
            rows.append(r)
        elif args.include_novel and r.get("novel_word"):          # the novel words (bloop, pop, goop): same
            r = dict(r, verified_word=r["novel_word"])            # speaker, volume and cadence as the vocabulary
            rows.append(r)
    bs = int(round(BIN_MS / 1000.0 * SAMPLING_RATE))
    pad = int(args.pad_s * SAMPLING_RATE)
    days, out = {}, []
    for date in sorted({r["date"] for r in rows}):
        p = PipelinePaths(args.data_root, args.subject, args.region, date)
        offset, at = {}, 0
        for f in sorted(p.prep.glob("*_audio_raw.npy")):
            s = f.name[:-len("_audio_raw.npy")]
            offset[s] = (at, np.load(f, mmap_mode="r").size)
            at += offset[s][1] // bs
        days[date] = (np.load(p.pipeline / "features.npy"), np.load(p.pipeline / "targets.npy"), offset)
    for r in rows:
        feats, targs, offset = days[r["date"]]
        base, n_samples = offset[r["session"]]
        a = max(0, int(r["start_sample"]) - pad) // bs
        b = -(-min(n_samples, int(r["end_sample"]) + pad) // bs)
        out.append((r, base + a, base + b))
    return out, days


def stratified_split(epochs, seed, keep=None):
    """keep: {(date, session, epoch_idx): split} of an earlier cohort; those epochs keep their split (so its test
    utterances stay the test set) and only the other epochs are split by the rule below."""
    keep = keep or {}
    by_word = defaultdict(list)
    split = {"train": [], "valid": [], "test": []}
    for e in epochs:
        k = (e[0]["date"], e[0]["session"], str(e[0]["epoch_idx"]))
        if k in keep:
            split[keep[k]].append(e)
        else:
            by_word[e[0]["verified_word"]].append(e)
    rng = np.random.default_rng(seed)
    for word in sorted(by_word):
        es = by_word[word]
        order = rng.permutation(len(es))
        if len(es) < MIN_HELD_OUT:                    # too few to hold any out: train on them all, score none
            split["train"] += es
            continue
        k = max(1, int(round(0.1 * len(es))))
        split["test"] += [es[i] for i in order[:k]]
        split["valid"] += [es[i] for i in order[k:2 * k]]
        split["train"] += [es[i] for i in order[2 * k:]]
    for s in split:                                   # shuffle the order within each split, across words and days
        split[s] = [split[s][i] for i in rng.permutation(len(split[s]))]
    return split


def keep_only(split, keep):
    """Held-out splits that must stay as they are (an earlier cohort's split is being kept): none can give epochs."""
    return ("valid", "test") if keep else ()


def aligned_lengths(l_tr, l_va, l_te, slack):
    """Train, valid and test frame totals, each at most its slack below the given one, for which the pipeline's split
    by position (window i -> frame i + SAMPLES_BEFORE; int(0.8 n) train and int(0.1 n) valid windows) starts valid
    and test exactly at an epoch boundary. Prefers the smallest total trim."""
    best = None
    for te in range(l_te, l_te - slack["test"] - 1, -1):
        for va in range(l_va, l_va - slack["valid"] - 1, -1):
            for tr in range(l_tr, max(l_tr - slack["train"], l_tr - 400) - 1, -1):
                n = tr + va + te - SAMPLES_BEFORE - SAMPLES_AFTER
                if int(TRAIN_DECIMAL * n) + SAMPLES_BEFORE == tr and int(VALID_DECIMAL * n) == va:
                    cut = (l_tr - tr) + (l_va - va) + (l_te - te)
                    if best is None or cut < best[0]:
                        best = (cut, tr, va, te)
                    break
    if best is None:
        raise RuntimeError("no trim within the padding aligns the split; widen --pad-s")
    return best[1:]


def trim(spans, n, pad_frames):
    """Remove n frames in total from the trailing padding of the epochs (one at a time, round robin, at most
    pad_frames from each), so every word stays whole."""
    spans = [list(x) for x in spans]
    cut = [0] * len(spans)
    i = 0
    while n > 0:
        if cut[i] < pad_frames and spans[i][2] - spans[i][1] > 2 * pad_frames:
            spans[i][2] -= 1
            cut[i] += 1
            n -= 1
        i = (i + 1) % len(spans)
    return [tuple(x) for x in spans]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--pad-s", type=float, default=0.3)
    ap.add_argument("--clip-std", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--include-novel", action="store_true", help="add the novel words (bloop, pop, goop)")
    ap.add_argument("--keep-split-from", default=None, help="an earlier cohort's prep/epochs.csv: its epochs keep "
                    "their split")
    ap.add_argument("--name", default=NAME)
    args = ap.parse_args()

    epochs, days = epochs_from_log(args)
    keep = {}
    if args.keep_split_from:
        keep = {(r["date"], r["session"], str(r["epoch_idx"])): r["split"]
                for r in csv.DictReader(open(args.keep_split_from))}
    split = stratified_split(epochs, args.seed, keep)
    if keep:
        kept = sum(1 for s in split for e in split[s] if (e[0]["date"], e[0]["session"], str(e[0]["epoch_idx"])) in keep)
        print(f"kept the split of {kept} of {len(keep)} epochs from {args.keep_split_from}")
    pad_frames = int(args.pad_s * 1000 / BIN_MS) - 1                # padding frames that are surely not the word
    while True:
        length = {s: sum(b - a for _, a, b in split[s]) for s in split}
        try:
            target = aligned_lengths(length["train"], length["valid"], length["test"],
                                     {s: pad_frames * len(split[s]) for s in split})
            break
        except RuntimeError:
            # the held-out share is above 20% (words with 13 to 15 epochs hold out 2 + 2): move one held-out epoch
            # to train, from the larger held-out split, of a word that still has 2 there (the most frequent such)
            held = [s for s in ("valid", "test") if s not in keep_only(split, keep)]
            s = max(held, key=lambda h: length[h])
            count = {}
            for e in split[s]:
                count[e[0]["verified_word"]] = count.get(e[0]["verified_word"], 0) + 1
            total = {}
            for x in split.values():
                for e in x:
                    total[e[0]["verified_word"]] = total.get(e[0]["verified_word"], 0) + 1
            cand = [w for w, c in count.items() if c >= 2]
            if not cand:
                raise
            w = max(sorted(cand), key=lambda w: total[w])
            i = max(i for i, e in enumerate(split[s]) if e[0]["verified_word"] == w)
            split["train"].append(split[s].pop(i))
            print(f"moved one held-out '{w}' epoch from {s} to train to bring the held-out share to 20%")
    for s, want in zip(("train", "valid", "test"), target):
        split[s] = trim(split[s], length[s] - want, pad_frames)
    print("trimmed padding frames to align the split: " +
          ", ".join(f"{s} {length[s] - w}" for s, w in zip(("train", "valid", "test"), target)))

    # per-day statistics from each day's train epochs only
    stats = {}
    for date, (feats, _, _) in days.items():
        x = np.concatenate([feats[a:b] for r, a, b in split["train"] if r["date"] == date])
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        stats[date] = (mu, sd)

    F, C, T, table, at = [], [], [], [], 0
    for s in ("train", "valid", "test"):
        for r, a, b in split[s]:
            feats, targs, _ = days[r["date"]]
            mu, sd = stats[r["date"]]
            F.append(np.clip((feats[a:b] - mu) / sd, -args.clip_std, args.clip_std))
            C.append(feats[a:b] / sd)
            T.append(targs[a:b])
            table.append(dict(split=s, word=r["verified_word"], date=r["date"], session=r["session"],
                              epoch_idx=r["epoch_idx"], frame_start=at, frame_end=at + b - a))
            at += b - a
    F, C, T = (np.concatenate(v).astype(np.float32) for v in (F, C, T))

    out = PipelinePaths(args.data_root, args.subject, args.region, args.name)
    np.save(out.pipeline / "features.npy", F)
    np.save(out.pipeline / "features_counts_daynorm.npy", C)
    np.save(out.pipeline / "targets.npy", T)
    with open(out.prep / "epochs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    # where the pipeline's split by position falls (window i targets frame i + SAMPLES_BEFORE)
    n_win = len(F) - SAMPLES_BEFORE - SAMPLES_AFTER
    tr = int(TRAIN_DECIMAL * n_win)
    va = tr + int(VALID_DECIMAL * n_win)
    print(f"{args.name}: {len(F)} frames ({len(F) * BIN_MS / 1000 / 60:.1f} min), {len(table)} epochs, "
          f"{len({t['word'] for t in table})} words, {len(days)} days")
    for s in ("train", "valid", "test"):
        rows = [t for t in table if t["split"] == s]
        print(f"  {s:5s}: {len(rows):3d} epochs, frames {rows[0]['frame_start']}-{rows[-1]['frame_end']}, "
              f"days {sorted({t['date'] for t in rows})}")
    for name, frame in (("train/valid", tr + SAMPLES_BEFORE), ("valid/test", va + SAMPLES_BEFORE)):
        t = next(t for t in table if t["frame_start"] <= frame < t["frame_end"])
        ok = frame == t["frame_start"]
        print(f"  pipeline {name} boundary at frame {frame}: " + ("exactly at the start of" if ok else "INSIDE") +
              f" the {t['split']} epoch '{t['word']}' ({t['date']})")
        assert ok, "the split does not fall on an epoch boundary"


if __name__ == "__main__":
    main()
