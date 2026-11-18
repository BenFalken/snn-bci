"""
Showcase video (for slides and for sharing; NOT the headline result): for each of the 23 heard words, the one test
utterance that the decoders reconstructed best, from either monkey, played as the true sound and then by the paper's
GRU, the paper's LSTM, the LSNN and Speck on the chip (P5, readout refitted, seed s11), each with its spectral match
within the word's sound. The full, unselected results are video_audio_words_final_{Neil,Gomez}.mp4.

Selection: every test utterance of both monkeys' expanded cohorts (ALLWORDS_NOVEL_DAYNORM) is scored on the clip shown
(1.6 s around its first utterance, sound frames only) by each of the four decoders; per word, the utterance with the
highest mean over the four decoders is kept, so no single decoder is favoured. The video says on screen that each word
is its best-decoded utterance.

    python3 video_audio_showcase.py [--output figures/video_audio_showcase.mp4]
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import argparse
import tempfile
from pathlib import Path

import numpy as np

import collect_fidelity as cf
import video_audio as va                              # changes directory to the AuditoryDecoding repo
import video_audio_words as vw
from video_audio import g, PipelinePaths, START, style

DATASET = "ALLWORDS_NOVEL_DAYNORM"
LANES = [("gru", "gru_rnn"), ("lstm", "lstm_rnn"), ("lsnn", "lsnn"), ("speck", None)]
SEED = "s11"


def cohort(subject):
    paths = PipelinePaths("pipeline_data", subject, "aSTG", DATASET)
    y = np.load(paths.pipeline / "window_targets_test.npy")
    preds = {n: np.load(paths.pipeline / f"{dec}_predict_test.npy") for n, dec in LANES if dec}
    preds["speck"] = np.load(WORKSPACE / f"results_audio_novel/preds_{subject}_{SEED}/"
                             f"{subject}_P5_{SEED}/chip_refit.npy")
    assert all(len(p) == len(y) for p in preds.values())
    return paths, y, preds


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--output", default="figures/video_audio_showcase.mp4")
    a = pa.parse_args()
    a.output = str(Path(a.output) if Path(a.output).is_absolute() else START / a.output)
    style.apply()
    style.LABEL.update(speck="Speck", gru="GRU", lstm="LSTM", lsnn="LSNN")
    data = {m: cohort(m) for m in ("Neil", "Gomez")}
    best = {}                                          # word -> (mean score, subject, date, s, e, mask, scores)
    for m, (paths, y, preds) in data.items():
        for word, date, wa, wb in vw.test_words(paths, 0, every=True):
            s, e, mask = vw.first_utterance(y, wa, wb, vw.CLIP)
            if mask.sum() < 3:
                continue
            sc = {n: cf.spectral_r(y[s:e][mask], preds[n][s:e][mask]) for n, _ in LANES}
            mean = float(np.mean(list(sc.values())))
            if word not in best or mean > best[word][0]:
                best[word] = (mean, m, date, s, e, mask, sc)
    order = sorted(best, key=lambda w: (vw.SYLLABLES.get(w, 9), w))
    print(f"{len(order)} words, one utterance each:")
    tmp = tempfile.mkdtemp(prefix="paper_audio_showcase_")
    clips = []
    for i, word in enumerate(order):
        mean, m, date, s, e, mask, sc = best[word]
        paths, y, preds = data[m]
        print(f"  {word:12s} monkey {m[0]}, {date}, rows {s}:{e}, mean {mean:.2f} | "
              + "  ".join(f"{n} {v:.2f}" for n, v in sc.items()))
        audios, mels = {}, {}
        audios["truth"], truth_mel = g.reconstruct_audio(y[s:e], paths, f"showcase_{i:02d}_{word}_truth")
        for n, _ in LANES:
            audios[n], mels[n] = g.reconstruct_audio(preds[n][s:e], paths, f"showcase_{i:02d}_{word}_{n}")
        k = min(len(truth_mel), *[len(v) for v in mels.values()])
        novel = word in ("bloop", "pop", "goop")
        title = (f"Heard word: ‘{word}’   (monkey {m[0]}, day {date.replace('_', '/')}"
                 + (", a novel word" if novel else "") + "; the best-decoded utterance of this word)")
        clips += va.render_word(word, {n: v[:k] for n, v in mels.items()}, truth_mel[:k], audios, tmp, i, title,
                                scores=sc, score_label="spectral match within the word's sound")
    va.assemble(clips, a.output, tmp)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
