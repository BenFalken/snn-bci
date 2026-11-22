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
    python3 video_audio_showcase.py --fine        (-> figures/video_audio_showcase_4ms.mp4; the original is kept)
        EXPLORATORY (the spike-timing pilot, audio/speck/runs_fine_timing.txt): the same utterances, chosen by the same
        four decoders, with two more lanes: the LSNN reading real 4 ms spikes (LS1) and the Speck SNN on real 4 ms
        spikes (FT1; the chip model, readout refitted, until it has run on the chip: --speck4-preds). Each 4 ms lane is
        its network's MEDIAN seed for that monkey by the paper's metric on test (median per-word band r; with 3 seeds,
        the middle one), shown on screen; --fine-seed s11 forces one seed instead
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import collect_fidelity as cf
import video_audio as va                              # the repository's audio pipeline (changes directory to the data root)
import video_audio_words as vw
from video_audio import g, PipelinePaths, START, style

DATASET = "ALLWORDS_NOVEL_DAYNORM"
LANES = [("gru", "gru_rnn"), ("lstm", "lstm_rnn"), ("lsnn", "lsnn"), ("speck", None)]
SEED = "s11"


FINE = {"lsnn4": "LS1_fine", "speck4": "FT1_spikes"}
SEEDS_SHOWN = {}                                      # subject -> {lane: seed} for the 4 ms lanes    # the pilot's runs (final_results/audio_fine)


def median_seed(subject, run, pred_file):
    """The seed whose median per-word band r on test is the middle one of the run's finished seeds."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio" / "speck"))
    import fine_timing_summary as fs
    y, spans = fs.words(subject, "test")
    F = WORKSPACE / "final_results" / "audio_fine"
    sc = {s: fs.per_word(y, spans, np.load(F / f"{subject}_{run}_{s}" / pred_file))
          for s in ("s1337", "s11", "s99") if (F / f"{subject}_{run}_{s}" / pred_file).is_file()}
    order = sorted(sc, key=sc.get)
    pick = order[(len(order) - 1) // 2]
    print(f"  {subject} {run}: seeds {', '.join(f'{k} {v:.3f}' for k, v in sc.items())} -> {pick}")
    return pick


def cohort(subject, fine=False, speck4=None, fine_seed=None):
    paths = PipelinePaths(str(va.AUDITORY_DATA), subject, "aSTG", DATASET)
    y = np.load(paths.pipeline / "window_targets_test.npy")
    preds = {n: np.load(paths.pipeline / f"{dec}_predict_test.npy") for n, dec in LANES if dec}
    preds["speck"] = np.load(WORKSPACE / f"results_audio_novel/preds_{subject}_{SEED}/"
                             f"{subject}_P5_{SEED}/chip_refit.npy")
    if fine:
        F = WORKSPACE / "final_results" / "audio_fine"
        files = {"lsnn4": "preds/test.npy", "speck4": "refit/preds_event_refit.npy"}
        seeds = {n: fine_seed or median_seed(subject, FINE[n], files[n]) for n in FINE}
        preds["lsnn4"] = np.load(F / f"{subject}_{FINE['lsnn4']}_{seeds['lsnn4']}" / files["lsnn4"])
        preds["speck4"] = np.load(speck4.format(subject=subject, seed=seeds["speck4"]) if speck4 else
                                  F / f"{subject}_{FINE['speck4']}_{seeds['speck4']}" / files["speck4"])
        SEEDS_SHOWN[subject] = seeds
    assert all(len(p) == len(y) for p in preds.values())
    return paths, y, preds


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--output", default=None)
    pa.add_argument("--fine", action="store_true", help="add the 4 ms LSNN and Speck lanes (the selection is unchanged)")
    pa.add_argument("--speck4-preds", default=None, help="the 4 ms Speck lane's test predictions, with {subject} "
                    "and {seed} (default: its chip model)")
    pa.add_argument("--fine-seed", default=None, choices=["s1337", "s11", "s99"],
                    help="one seed for both 4 ms lanes (default: each network's median seed per monkey)")
    a = pa.parse_args()
    a.output = a.output or ("figures/video_audio_showcase_4ms.mp4" if a.fine else "figures/video_audio_showcase.mp4")
    a.output = str(Path(a.output) if Path(a.output).is_absolute() else START / a.output)
    style.apply()
    style.LABEL.update(speck="Speck", gru="GRU", lstm="LSTM", lsnn="LSNN", lsnn4="LSNN\n4 ms",
                       speck4="Speck\n4 ms" + ("" if a.speck4_preds else "\n(chip model)"))
    style.COLOR.update(lsnn4="#8B2E00", speck4="#A86F00")
    data = {m: cohort(m, a.fine, a.speck4_preds, a.fine_seed) for m in ("Neil", "Gomez")}
    order_lanes = [n for n, _ in LANES]
    if a.fine:                                         # each 4 ms lane right after its 40 ms counterpart
        order_lanes = ["gru", "lstm", "lsnn", "lsnn4", "speck", "speck4"]
    best = {}                                          # word -> (mean score, subject, date, s, e, mask, scores)
    for m, (paths, y, preds) in data.items():
        for word, date, wa, wb in vw.test_words(paths, 0, every=True):
            s, e, mask = vw.first_utterance(y, wa, wb, vw.CLIP)
            if mask.sum() < 3:
                continue
            sc = {n: cf.spectral_r(y[s:e][mask], preds[n][s:e][mask]) for n, _ in LANES}
            mean = float(np.mean(list(sc.values())))   # the selection: the original four decoders only
            sc = {n: sc[n] if n in sc else cf.spectral_r(y[s:e][mask], preds[n][s:e][mask]) for n in order_lanes}
            if word not in best or mean > best[word][0]:
                best[word] = (mean, m, date, s, e, mask, sc)
    tag = "4ms" if a.fine else ""                  # the intermediate files apart from the original's
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
        audios["truth"], truth_mel = g.reconstruct_audio(y[s:e], paths, f"showcase{tag}_{i:02d}_{word}_truth")
        for n in order_lanes:
            audios[n], mels[n] = g.reconstruct_audio(preds[n][s:e], paths, f"showcase{tag}_{i:02d}_{word}_{n}")
        k = min(len(truth_mel), *[len(v) for v in mels.values()])
        novel = word in ("bloop", "pop", "goop")
        title = (f"Heard word: ‘{word}’   (monkey {m[0]}, day {date.replace('_', '/')}"
                 + (", a novel word" if novel else "") + "; the best-decoded utterance of this word"
                 + (f"; 4 ms lanes: LSNN {SEEDS_SHOWN[m]['lsnn4']}, Speck {SEEDS_SHOWN[m]['speck4']}, the median seeds"
                    if a.fine else "") + ")")
        clips += va.render_word(word, {n: v[:k] for n, v in mels.items()}, truth_mel[:k], audios, tmp, i, title,
                                scores=sc, score_label="spectral match within the word's sound")
    va.assemble(clips, a.output, tmp)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
