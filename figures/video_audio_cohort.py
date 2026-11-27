"""
Condensed audio video (for presentations): every multi-syllable heard word once, from a REPRESENTATIVE set of test
utterances rather than the best-decoded ones. Lanes: the true sound, then the paper's GRU and LSTM, the LSNN and the
Speck SNN on the chip (both spiking decoders on 4 ms spikes, the final networks: LS1 and FS3, each at its median seed
per monkey, as videos 3 and 4).

Selection: among the multi-syllable words' test utterances of both monkeys (ALLWORDS_NOVEL_DAYNORM), one utterance per
word is chosen, all combinations searched, so that each lane's MEAN score over the chosen clips comes as close as
possible to that decoder's reported median over all 79 test utterances (data/four_fidelity.csv, audio band_r: the
paper's numbers). Scores are the paper's metric (band r: each mel band's correlation over the utterance epoch,
Fisher-averaged), computed for the predictions heard (the median seed). Ties are broken towards medians that also
match and towards both monkeys equally. The clip played is 1.6 s around the epoch's first utterance, as in videos 3-4.

    python3 video_audio_cohort.py [--output figures/video_audio_cohort.mp4] [--dry-run]

Needs Docker running (the pipeline's images, for the sound) and ffmpeg.
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import argparse
import csv
import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import video_audio as va                              # the repository's audio pipeline (changes directory to the data root)
import video_audio_words as vw
import video_audio_showcase as sh
from collect_fidelity import fisher_mean_r
from video_audio import g, START, style

HERE = Path(__file__).parent
LANES = ["gru", "lstm", "lsnn", "speck"]
MULTI = sorted(w for w, s in vw.SYLLABLES.items() if s >= 2)


def candidates(data):
    out = []
    for m, (paths, y, preds) in data.items():
        rows = [r for r in csv.DictReader(open(paths.prep / "epochs.csv")) if r["split"] == "test"]
        base = min(int(r["frame_start"]) for r in rows) - g.SAMPLES_BEFORE
        for r in rows:
            a, b = int(r["frame_start"]) - g.SAMPLES_BEFORE - base, int(r["frame_end"]) - g.SAMPLES_BEFORE - base
            if r["word"] not in MULTI or a < 0 or b > len(y):
                continue
            ok = np.all([np.isfinite(preds[n][a:b, 0]) for n in LANES], 0)
            if ok.sum() < 3:
                continue
            sc = {n: fisher_mean_r(y[a:b][ok], preds[n][a:b][ok]) for n in LANES}
            out.append(dict(word=r["word"], monkey=m, date=r["date"], a=a, b=b, uid=f"{r['word']}_{r['date']}_{r['epoch_idx']}",
                            **sc))
    return pd.DataFrame(out)


def select(c, target):
    words = sorted(c.word.unique())
    opts = [c.index[c.word == w].tolist() for w in words]
    S = c[LANES].to_numpy()
    t = np.array([target[n] for n in LANES])
    combos = np.array(list(itertools.product(*opts)), dtype=np.int32)          # (n_combos, n_words)
    mean = np.zeros((len(combos), len(LANES)))
    for j in range(combos.shape[1]):
        mean += S[combos[:, j]]
    mean /= combos.shape[1]
    err = ((mean - t) ** 2).sum(1)
    top = np.argsort(err)[:5000]                                            # near-equal on the means: refine
    med = np.median(S[combos[top]], axis=1)                                 # (top, n_words, lanes) -> (top, lanes)
    neil = (c.monkey.to_numpy()[combos[top]] == "Neil").sum(1)
    score = err[top] + 0.25 * ((med - t) ** 2).sum(1) + 1e-4 * np.abs(neil - combos.shape[1] / 2)
    best = combos[top[np.argmin(score)]]
    print(f"{len(combos):,} combinations of one utterance per word ({len(words)} words)")
    return c.loc[best]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--output", default=str(HERE / "figures" / "video_audio_cohort.mp4"))
    pa.add_argument("--dry-run", action="store_true", help="choose and print the clips only")
    a = pa.parse_args()
    a.output = str(Path(a.output) if Path(a.output).is_absolute() else START / a.output)
    f = pd.read_csv(HERE / "data" / "four_fidelity.csv")
    f = f[(f.experiment == "audio") & (f.metric == "band_r")]
    target = f.groupby("decoder").value.median().to_dict()
    print("reported medians (79 utterances):", {n: round(target[n], 3) for n in LANES})
    data = {m: sh.cohort_final(m) for m in ("Neil", "Gomez")}
    c = candidates(data)
    pop = c[LANES].mean()
    print(f"{len(c)} multi-syllable test utterances; their mean (median seed): "
          + ", ".join(f"{n} {pop[n]:.3f}" for n in LANES))
    sel = select(c, target).sort_values(["word"], key=lambda s: s.map(lambda w: (vw.SYLLABLES[w], w)))
    print("chosen:")
    for _, r in sel.iterrows():
        print(f"  {r.word:11s} monkey {r.monkey[0]} {r.date}  " + "  ".join(f"{n} {r[n]:.2f}" for n in LANES))
    print("  MEAN       " + "  ".join(f"{n} {sel[n].mean():.3f} (reported {target[n]:.3f})" for n in LANES))
    print("  MEDIAN     " + "  ".join(f"{n} {sel[n].median():.3f}" for n in LANES))
    print(f"  monkeys: {dict(sel.monkey.value_counts())}")
    sel.to_csv(Path(a.output).with_suffix(".csv"), index=False)
    if a.dry_run:
        return
    style.apply()
    style.LABEL.update(speck="Speck (chip)", gru="GRU", lstm="LSTM", lsnn="LSNN")
    tmp = tempfile.mkdtemp(prefix="paper_audio_cohort_")
    clips = []
    for i, (_, r) in enumerate(sel.iterrows()):
        paths, y, preds = data[r.monkey]
        s, e, _ = vw.first_utterance(y, r.a, r.b, vw.CLIP)
        audios, mels = {}, {}
        audios["truth"], truth_mel = g.reconstruct_audio(y[s:e], paths, f"cohort_{i:02d}_{r.word}_truth")
        for n in LANES:
            audios[n], mels[n] = g.reconstruct_audio(preds[n][s:e], paths, f"cohort_{i:02d}_{r.word}_{n}")
        k = min(len(truth_mel), *[len(v) for v in mels.values()])
        title = f"Heard word: ‘{r.word}’   (monkey {r.monkey[0]}, day {r.date.replace('_', '/')})"
        clips += va.render_word(r.word, {n: v[:k] for n, v in mels.items()}, truth_mel[:k], audios, tmp, i, title,
                                scores={n: r[n] for n in LANES},
                                score_label="band r over the utterance (the paper's metric)")
    va.assemble(clips, a.output, tmp)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
