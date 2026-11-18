"""
Video: the audio decoders against the ground truth, one heard word at a time (the five-word verified dataset, its
validation split, the pipeline's predictions of every core decoder: Kalman filter, Wiener filter, LSTM, LSNN).

Per word, one pass per lane, each at real time: the true sound first, then each decoder's sound (its predicted mel
spectrogram, unscaled and inverted by the pipeline's own Griffin-Lim). A cursor sweeps the word; next to each decoder its
correlation with the truth so far (Pearson r over every band and frame up to the cursor) grows as the cursor moves and
keeps its final value. The playing lane is outlined in its color.

    python3 video_audio.py [--words tree,good,north,cricket,program] [--output figures/video_audio.mp4]

Needs Docker running (the pipeline's sklearn_preprocess_test and mikko_test images) and ffmpeg.
"""

from repo_paths import AUDIO, AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)
import argparse
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.io import wavfile

import style

HERE = Path(__file__).parent
REPO = AUDIO                                       # the repository's audio pipeline code
sys.path.insert(0, str(REPO))
import os
START = Path.cwd()                                    # output paths are relative to where the script was started
os.chdir(AUDITORY_DATA.parent)                     # relative pipeline_data/ paths inside the pipeline resolve
import generate_word_prediction_audio as g          # noqa: E402  (the pipeline's own word mapping and audio inversion)
from pipeline_paths import PipelinePaths              # noqa: E402

DATASET = "FIVEWORD_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13"
LANES = [("kf", "kalman_filter"), ("wf", "wiener_filter"), ("lstm", "lstm_rnn"), ("lsnn", "lsnn")]
FPS = 25


def valid_instance(word, rows, ranges, total):
    """The first instance of `word` lying wholly inside the validation split, for every decoder's split (the raw
    split of the Kalman filter has slightly different boundaries from the windowed one). Returns
    {decoder: (start, end) in that decoder's validation rows} and the windowed (start, end), or None."""
    out = {}
    for r in [r for r in rows if r["verified_word"] == word]:
        key = (r["date"], r["session"], int(r["epoch_idx"]))
        if key not in ranges:
            continue
        a, b = ranges[key]
        ok = True
        for name, dec in LANES:
            tag = g.DECODER_SPLIT_TAG[dec]
            off = g.SAMPLES_BEFORE if tag != "raw" else 0
            eff = total - (g.SAMPLES_BEFORE + g.SAMPLES_AFTER) if tag != "raw" else total
            tr = int(g.TRAIN_DECIMAL * eff); va = tr + int(g.VALID_DECIMAL * eff)
            s, e = a - off, b - off
            if not (tr <= s and e <= va):
                ok = False
                break
            out[name] = (s - tr, e - tr)
        if ok:
            return r, out
    return None


def running_r(pred, truth):
    """Pearson r over all bands and frames up to each frame (NaN for the first few)."""
    out = np.full(len(truth), np.nan)
    for t in range(4, len(truth) + 1):
        a, b = pred[:t].ravel(), truth[:t].ravel()
        if a.std() > 0 and b.std() > 0:
            out[t - 1] = np.corrcoef(a, b)[0, 1]
    return out


def render_word(word, lanes_mel, truth_mel, audios, tmp, idx, title=None, scores=None, score_label=None):
    """One clip per pass (truth, then each decoder in lanes_mel's order); returns (silent video, audio) per pass,
    the audio padded to exactly the video's length, for assemble(). title replaces the heading 'Heard word: ...'. scores (decoder -> value) replaces the running correlation
    with one fixed score per decoder, shown from its pass on, under the heading score_label."""
    names = ["truth"] + list(lanes_mel)
    n_frames = truth_mel.shape[0]
    r_final = {n: running_r(lanes_mel[n], truth_mel) for n in lanes_mel}
    vmin, vmax = np.percentile(truth_mel, [2, 99.5])
    clips = []
    for p, playing in enumerate(names):
        audio = audios[playing]
        dur = len(audio) / g.SAMPLING_RATE
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(len(names), 2, width_ratios=[1, 0.13], hspace=0.18, wspace=0.03,
                              left=0.09, right=0.97, top=0.88, bottom=0.08)
        fig.text(0.09, 0.95, title or f"Heard word: ‘{word}’", fontsize=20, fontweight="bold")
        fig.text(0.09, 0.915, "now playing: " + ("the true sound" if playing == "truth" else
                                                 f"{style.LABEL[playing]} reconstruction".replace("\n", " ")),
                 fontsize=13, color=style.MUTED if playing == "truth" else style.COLOR[playing])
        fig.text(0.97, 0.915, score_label or "correlation with truth so far", fontsize=11, color=style.MUTED,
                 ha="right")
        cursors, rtexts = [], {}
        for i, n in enumerate(names):
            ax = fig.add_subplot(gs[i, 0])
            mel = truth_mel if n == "truth" else lanes_mel[n]
            ax.imshow(mel.T, origin="lower", aspect="auto", cmap="Greys", vmin=vmin, vmax=vmax,
                      extent=[0, dur, 0, mel.shape[1]])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(n == playing)
                s.set_color("#1a1a1a" if n == "truth" else style.COLOR.get(n, "#1a1a1a"))
                s.set_linewidth(3)
            ax.set_ylabel("Truth" if n == "truth" else style.LABEL[n], rotation=0, ha="right", va="center",
                          fontsize=13, fontweight="bold" if n in style.HIGHLIGHT else "normal",
                          color="#1a1a1a" if n == "truth" else style.COLOR[n])
            if i < len(names) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("time (s)", fontsize=12)
                ax.tick_params(labelsize=11)
            cursors.append(ax.axvline(0, color="#d62728" if False else "#1a1a1a", lw=2))
            tx = fig.add_subplot(gs[i, 1])
            tx.axis("off")
            if n != "truth":
                rtexts[n] = tx.text(0.05, 0.5, "", fontsize=16, va="center", color=style.COLOR[n],
                                    fontweight="bold" if n in style.HIGHLIGHT else "normal")
        order = names[1:]
        done = set(order[:max(p - 1, 0)]) if playing != "truth" else set()
        silent = Path(tmp) / f"w{idx}_{p}_silent.mp4"
        writer = FFMpegWriter(fps=FPS, bitrate=3000)
        n_vid = max(int(np.ceil(dur * FPS)), 1)             # exactly n_vid frames; the audio is padded to match
        with writer.saving(fig, str(silent), dpi=100):
            for k in range(n_vid):
                t = k / FPS
                for c in cursors:
                    c.set_xdata([t, t])
                fi = min(int(t / dur * n_frames), n_frames - 1)
                for n, txt in rtexts.items():
                    if n in done or (playing == "truth"):
                        v = (scores[n] if scores else r_final[n][-1]) if n in done else np.nan
                    elif n == playing:
                        v = scores[n] if scores else r_final[n][fi]
                    else:
                        v = np.nan
                    txt.set_text("" if np.isnan(v) else f"r = {v:.2f}")
                writer.grab_frame()
        plt.close(fig)
        n_samp = int(round(n_vid / FPS * g.SAMPLING_RATE))
        clips.append((silent, np.pad(np.asarray(audio, dtype=np.float64), (0, max(n_samp - len(audio), 0)))[:n_samp]))
    return clips


def assemble(parts, output, tmp):
    """Join render_word's passes into one video: the silent clips end to end (each exactly its frame count) and one
    continuous soundtrack built from the passes' audio, each padded to its clip's length, muxed once. Joining
    clips that each carry their own AAC track lets the audio drift from the picture by the encoder's padding at
    every join (about 190 joins in the all-words video)."""
    lst = Path(tmp) / "list.txt"
    lst.write_text("".join(f"file '{v}'\n" for v, _ in parts))
    video = Path(tmp) / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video)], check=True)
    wav = Path(tmp) / "audio.wav"
    wavfile.write(wav, int(g.SAMPLING_RATE),
                  (np.clip(np.concatenate([a for _, a in parts]), -1, 1) * 32767).astype(np.int16))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map",
                    "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(output)], check=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--words", default="tree,good,north,cricket,program")
    pa.add_argument("--output", default=str(HERE / "figures" / "video_audio.mp4"))
    a = pa.parse_args()
    a.output = str(Path(a.output) if Path(a.output).is_absolute() else START / a.output)
    style.apply()
    paths = PipelinePaths(str(AUDITORY_DATA), "Neil", "aSTG", DATASET)
    args = types.SimpleNamespace(data_root=str(AUDITORY_DATA), subject="Neil", region="aSTG", pad_s=0.3)
    rows = g.load_relevant_rows(str(AUDITORY_DATA / "Neil/aSTG/word_human_verified_log.csv"), DATASET)
    ranges, total = g.compute_all_epoch_global_ranges(rows, args)
    preds = {n: np.load(paths.pipeline / f"{dec}_predict_valid.npy") for n, dec in LANES}
    targets = np.load(paths.pipeline / "window_targets_valid.npy")
    tmp = tempfile.mkdtemp(prefix="paper_audio_video_")
    clips = []
    for i, word in enumerate(a.words.split(",")):
        found = valid_instance(word, rows, ranges, total)
        if found is None:
            print(f"'{word}': no instance wholly inside the validation split, skipped")
            continue
        r, sl = found
        ws, we = sl["lstm"]
        print(f"'{word}': {r['date']}/{r['session']} epoch {r['epoch_idx']}, validation rows {ws}:{we}")
        audios, mels = {}, {}
        audios["truth"], truth_mel = g.reconstruct_audio(targets[ws:we], paths, f"paper_{i:02d}_{word}_truth")
        for n, dec in LANES:
            s, e = sl[n]
            audios[n], mels[n] = g.reconstruct_audio(preds[n][s:e], paths, f"paper_{i:02d}_{word}_{n}")
            m = min(len(mels[n]), len(truth_mel))
            mels[n] = mels[n][:m]
        m = min(len(truth_mel), *[len(v) for v in mels.values()])
        truth_mel = truth_mel[:m]
        mels = {k: v[:m] for k, v in mels.items()}
        clips += render_word(word, mels, truth_mel, audios, tmp, i)
    assemble(clips, a.output, tmp)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
