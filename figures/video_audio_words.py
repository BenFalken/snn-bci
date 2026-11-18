"""
Video: heard words from the word-only dataset (ALLWORDS_VERIFIED_DAYNORM: the human-verified word epochs of all five
days, per-day normalized, split by whole epochs at random), decoded by the best ANN, the LSNN and the Speck SNN. Every
clip is a test utterance (a new recording of a word the decoders trained on): one instance of each word in the test split (prep/epochs.csv), in a fixed random order,
each played as the true sound and then once per decoder, at real time, with each decoder's correlation with the
truth so far (the renderer of video_audio.py).

A word epoch holds the spoken word twice; the clip is the 1.6 s around the epoch's loudest moment.

    python3 video_audio_words.py [--ann lstm_rnn] [--words 8] [--output figures/video_audio_words.mp4]
    python3 video_audio_words.py --all --within --output figures/video_audio_words_all.mp4
        every test word, GRU, LSTM, LSNN and Speck lanes, each scored by spectral match within the word's
        sound (silence excluded, as the fidelity figure's panel c), on the clip around the epoch's first utterance

The Speck lane is the hard-reset words network as it ran ON THE CHIP (layer 1 x4; --speck-seed picks which of its
three seeds, default s11, the median one on the chip per word), from final_results/audio_preds/words_P5_<seed>/chip.npy.

Needs Docker running (the pipeline's sklearn_preprocess_test and mikko_test images) and ffmpeg.
"""

from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import collect_fidelity as cf
import video_audio as va                              # changes directory to the AuditoryDecoding repo
from video_audio import g, PipelinePaths, START, HERE, style

DATASET = "ALLWORDS_VERIFIED_DAYNORM"
ANN = {"lstm_rnn": "lstm", "gru_rnn": "gru", "simple_rnn": "simple_rnn", "dense_nn": "dense_nn"}
CLIP = 40                                             # frames (40 ms) per clip


SYLLABLES = {"two": 1, "zoo": 1, "moo": 1, "cool": 1, "good": 1, "tree": 1, "fuel": 1, "north": 1,
             "apple": 2, "window": 2, "error": 2, "cricket": 2, "program": 2, "macaque": 2,
             "banana": 3, "sequence": 3, "quality": 3, "victory": 3, "reflection": 3, "laboratory": 4,
             "bloop": 1, "pop": 1, "goop": 1}


def test_words(paths, n, seed=0, every=False):
    """(word, date, first row, end row) in the test window rows: one instance per word, n words in random order; with
    every, all test instances, short words first (by syllables, then alphabetically)."""
    rows = [r for r in csv.DictReader(open(paths.prep / "epochs.csv")) if r["split"] == "test"]
    base = min(int(r["frame_start"]) for r in rows) - g.SAMPLES_BEFORE      # window i targets frame i + 8
    n_test = len(np.load(paths.pipeline / "window_targets_test.npy", mmap_mode="r"))
    rng = np.random.default_rng(seed)
    first = {}
    for i in rng.permutation(len(rows)):
        r = rows[i]
        a, b = int(r["frame_start"]) - g.SAMPLES_BEFORE - base, int(r["frame_end"]) - g.SAMPLES_BEFORE - base
        if (every or r["word"] not in first) and 0 <= a and b <= n_test:
            first[(r["word"], i) if every else r["word"]] = (r["word"], r["date"], a, b)
    if every:
        return sorted(first.values(), key=lambda w: (SYLLABLES.get(w[0], 9), w[0], w[1], w[2]))
    return list(first.values())[:n]


def first_utterance(truth, a, b, length):
    """The clip of `length` rows around the epoch's first utterance (an epoch holds the word twice), and its sound
    frames (collect_fidelity.sound_frames on the whole epoch, so the same mask as the figure)."""
    m = cf.sound_frames(truth[a:b].mean(1))
    on = a + int(np.argmax(m))
    s = min(max(a, on - 8), b - length)
    return s, s + length, m[s - a:s - a + length]


def loudest(truth, a, b, length):
    e = np.convolve(truth[a:b].mean(1), np.ones(5) / 5, "same")
    c = a + int(np.argmax(e))
    s = min(max(a, c - length // 2), b - length)
    return s, s + length


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ann", default="gru_rnn", choices=sorted(ANN))
    pa.add_argument("--words", type=int, default=8)
    pa.add_argument("--all", action="store_true", help="every test instance of every word, short words first")
    pa.add_argument("--within", action="store_true",
                    help="GRU and LSTM lanes, each scored by spectral match within the word's sound (fidelity figure panel c)")
    pa.add_argument("--output", default=str(HERE / "figures" / "video_audio_words.mp4"))
    pa.add_argument("--speck-seed", default="s11", choices=["s11", "s1337", "s99"],
                    help="which seed of the words Speck SNN's chip run plays in the Speck lane")
    pa.add_argument("--subject", default="Neil", help="Neil or Gomez (2026-11-18: the expanded cohorts)")
    pa.add_argument("--dataset", default=DATASET)
    pa.add_argument("--speck-preds", default=None, help="the Speck lane's test predictions (default: the 20-word "
                    "cohort's; for ALLWORDS_NOVEL_DAYNORM, results_audio_novel/preds_<subject>_<seed>/.../chip_refit.npy)")
    a = pa.parse_args()
    a.output = str(Path(a.output) if Path(a.output).is_absolute() else START / a.output)
    style.apply()
    style.LABEL["speck"] = "Speck"
    style.LABEL["gru"] = "GRU"
    style.LABEL["lsnn"] = "LSNN"
    paths = PipelinePaths("pipeline_data", a.subject, "aSTG", a.dataset)
    lanes = ([("gru", "gru_rnn"), ("lstm", "lstm_rnn")] if a.within else [(ANN[a.ann], a.ann)]) + \
        [("lsnn", "lsnn"), ("speck", "speck_snn")]
    targets = np.load(paths.pipeline / "window_targets_test.npy")
    preds = {n: np.load(paths.pipeline / f"{dec}_predict_test.npy") for n, dec in lanes if n != "speck"}
    # the chip's Hidden 3 spikes read by the readout refitted to the chip's firing on the training stream (the final
    # audio Speck SNN; chip/audio_refit.py + audio_apply_refit.py)
    if a.speck_preds is None and a.dataset == "ALLWORDS_NOVEL_DAYNORM":
        a.speck_preds = (WORKSPACE / f"results_audio_novel/preds_{a.subject}_{a.speck_seed}/"
                         f"{a.subject}_P5_{a.speck_seed}/chip_refit.npy")
    preds["speck"] = np.load(a.speck_preds or WORKSPACE / f"final_results/audio_preds/words_P5_{a.speck_seed}/chip_refit.npy")
    assert len(preds["speck"]) == len(targets), (len(preds["speck"]), len(targets))
    tmp = tempfile.mkdtemp(prefix="paper_audio_words_video_")
    clips = []
    for i, (word, date, wa, wb) in enumerate(test_words(paths, a.words, every=a.all)):
        if a.within:
            s, e, mask = first_utterance(targets, wa, wb, CLIP)
            scores = {n: cf.spectral_r(targets[s:e][mask], preds[n][s:e][mask]) for n, _ in lanes}
        else:
            (s, e), scores = loudest(targets, wa, wb, CLIP), None
        print(f"'{word}' ({date}): test rows {s}:{e}")
        audios, mels = {}, {}
        audios["truth"], truth_mel = g.reconstruct_audio(targets[s:e], paths, f"paper_words_{i:02d}_{word}_truth")
        for n, _ in lanes:
            audios[n], mels[n] = g.reconstruct_audio(preds[n][s:e], paths, f"paper_words_{i:02d}_{word}_{n}")
        m = min(len(truth_mel), *[len(v) for v in mels.values()])
        title = (f"Heard word: ‘{word}’   (monkey {a.subject[0]}, test utterance, day {date.replace('_', '/')}"
                 + (", a novel word)" if word in ("bloop", "pop", "goop") else ")"))
        clips += va.render_word(word, {k: v[:m] for k, v in mels.items()}, truth_mel[:m], audios, tmp, i, title,
                                scores=scores, score_label="spectral match within the word's sound" if scores else None)
        if scores:
            print("   " + "  ".join(f"{n} {v:.2f}" for n, v in scores.items()))
    va.assemble(clips, a.output, tmp)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
