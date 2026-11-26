"""
Video: attempted speech (T12) decoded by the best ANN (the paper's RNN, lowest phoneme error) and the LSNN, on the
test sentences exported by the speechBCI viewer (viewer_final: the final LSNN as 'SNN'), and the speech Speck SNN
(round 20 V2+C: three networks on one chip, V2 plus the top 64 features at 0-1.2 s) as it ran ON THE CHIP, its spikes read by the
readout refitted on training sentences (final_results/speech_v2c/chip_refit_logits.npz, the same test trials, 80 ms
frames).

No language model: every decoder is shown and scored by its phonemes alone (greedy CTC), as in the results.
Per sentence, one pass per lane at real time, as in the audio video: first the true sentence, then each decoder.
The neural features (threshold crossings and spike-band power, 256 channels) scroll under a cursor; the playing
lane reveals its phonemes (truth: aligned by the RNN; decoders: greedy CTC segments), and a neural voice speaks
them as a phoneme-decoding BCI would hear its output: each run of phonemes between silences is one decoded word,
spoken from its phonemes alone (Piper, offline, raw phoneme input: no spelling, no language model) at the time it
was decoded and fitted to the span it was decoded over (tts_phonemes.py). The true sentence goes through the same
voice from its aligned phonemes, so a perfect decoder would sound exactly like it. Lanes already played stay with
their phoneme error.

    python3 video_speech.py [--trials DIR] [--chip-logits NPZ] [--output figures/video_speech.mp4]

Needs ffmpeg, the Piper environment (.tts-venv with piper-tts) and a Piper voice (tts_voices/*.onnx). T12's
attempted speech made no usable sound, so the voice is synthetic. (macOS `say` ignores its phoneme mode on this
system and read the phoneme names aloud; the earlier version is video_speech.py.say.)
"""

from repo_paths import WORKSPACE, TTS_ENV, TTS_VOICES  # noqa: E402  (the repository's paths.yaml)
import argparse
import base64
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from scipy.io import wavfile

import style

HERE = Path(__file__).parent
LANES = [("paper_rnn", "RNN"), ("lsnn", "SNN")]          # (style key, name in the export)
FPS = 25
HOLD_S = 2.0                                              # the finished sentence stays this long


def heat(d):
    h = d["heat"]
    return np.frombuffer(base64.b64decode(h["data"]), np.uint8).reshape(h["rows"], h["cols"]).astype(float)


def strip(ax, segs, color, dur):
    """Phoneme blocks (silence left blank), labelled when wide enough."""
    for k, (ph, a, b) in enumerate(segs):
        if ph == "SIL":
            continue
        ax.add_patch(Rectangle((a / 1e3, 0.08), (b - a) / 1e3, 0.84, facecolor=color, alpha=0.35 + 0.3 * (k % 2),
                               edgecolor="white", lw=1))
        if (b - a) >= 70:
            ax.text((a + b) / 2e3, 0.5, ph, ha="center", va="center", fontsize=8, color=style.INK)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


ARPA_TO_SAY = {"AA": "AA", "AE": "AE", "AH": "AX", "AO": "AO", "AW": "AW", "AY": "AY", "B": "b", "CH": "C", "D": "d",
               "DH": "D", "EH": "EH", "ER": "UXr", "EY": "EY", "F": "f", "G": "g", "HH": "h", "IH": "IH", "IY": "IY",
               "JH": "J", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "N", "OW": "OW", "OY": "OY", "P": "p",
               "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "T", "UH": "UH", "UW": "UW", "V": "v", "W": "w",
               "Y": "y", "Z": "z", "ZH": "Z"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def phoneme_runs(segs):
    """A decoder's greedy phonemes as `say` phoneme strings, one per run between silences, each at its start (ms)."""
    runs, cur, at = [], [], None
    for ph, a, _ in list(segs) + [("SIL", None, None)]:
        if ph in ARPA_TO_SAY:
            at = a if not cur else at
            cur.append(ARPA_TO_SAY[ph] + ("1" if ph in VOWELS else ""))
        elif cur:
            runs.append(("[[inpt PHON]] " + "".join(cur), at))
            cur = []
    return runs


TTS_PY = TTS_ENV / "bin" / "python"
TTS_VOICE = TTS_VOICES / "en_US-lessac-high.onnx"


def word_runs(segs):
    """Phonemes grouped into decoded words: runs between silences, each [phonemes, start ms, end ms]."""
    runs, cur = [], None
    for ph, a, b in list(segs) + [("SIL", None, None)]:
        if ph != "SIL" and ph in ARPA_TO_SAY:
            cur = [[ph], a, b] if cur is None else [cur[0] + [ph], cur[1], b]
        elif cur is not None:
            runs.append(cur)
            cur = None
    return runs


def speak_phonemes(runs, path):
    """The decoded words as speech (tts_phonemes.py in the Piper environment); returns the track's length in s."""
    if not TTS_PY.is_file() or not TTS_VOICE.is_file():
        raise SystemExit(f"needs {TTS_PY} (piper-tts) and {TTS_VOICE}; see tts_phonemes.py")
    rp = Path(str(path) + ".runs.json")
    rp.write_text(json.dumps(runs))
    out = subprocess.run([str(TTS_PY), str(HERE / "tts_phonemes.py"), str(TTS_VOICE), str(rp), str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[-1])


def chip_decoder(logits, names, blank=40):
    """The Speck SNN's lane from the chip's logits (80 ms frames): greedy CTC segments, as the viewer exports them."""
    top = logits.argmax(1)
    segs, prev = [], None
    for i, t in enumerate(top):
        name = "SIL" if names[t] == "SIL" else names[t]
        if t != blank and t != prev:
            segs.append([name, 80.0 * i, 80.0 * (i + 1)])
        elif t == prev and t != blank and segs:
            segs[-1][2] = 80.0 * (i + 1)
        prev = t
    return {"name": "SpeckChip", "segments": segs}


def ctc_align(logits, seq, blank=40):
    """Frame spans of a phoneme sequence in CTC logits (Viterbi forced alignment): [(class, first, last frame)]."""
    x = logits - logits.max(1, keepdims=True)
    lp = x - np.log(np.exp(x).sum(1, keepdims=True))
    ext = [blank]
    for c in seq:
        ext += [c, blank]
    T, S = len(lp), len(ext)
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), int)
    dp[0, 0] = lp[0, ext[0]]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s_ in range(S):
            cands = [(dp[t - 1, s_], s_)]
            if s_ >= 1:
                cands.append((dp[t - 1, s_ - 1], s_ - 1))
            if s_ >= 2 and ext[s_] != blank and ext[s_] != ext[s_ - 2]:
                cands.append((dp[t - 1, s_ - 2], s_ - 2))
            v, b = max(cands)
            dp[t, s_], bp[t, s_] = v + lp[t, ext[s_]], b
    s_ = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = [s_]
    for t in range(T - 1, 0, -1):
        s_ = bp[t, s_]
        path.append(s_)
    path = path[::-1]
    spans = {}
    for t, s_ in enumerate(path):
        if ext[s_] != blank:
            k = (s_ - 1) // 2
            a, _ = spans.get(k, (t, t))
            spans[k] = (a, t)
    return [(seq[k], spans[k][0], spans[k][1]) for k in sorted(spans)]


def lm_decoder(logits, names, lm, alpha, beta, name, beam=32, blank=40):
    """A lane from beam search with the phoneme n-gram (ngram_lm.py): the decoded phonemes, timed by forced alignment
    to the same logits (the search itself outputs no timing)."""
    from ngram_lm import beam_decode
    seq = beam_decode(logits.astype(np.float64), lm, alpha, beta, beam)
    segs = [[names[c], 80.0 * a, 80.0 * (b + 1)] for c, a, b in ctc_align(logits, seq, blank)]
    return {"name": name, "segments": segs}


def edit_distance(a, b):
    D = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        P, D[0] = D.copy(), i
        for j, y in enumerate(b, 1):
            D[j] = min(P[j] + 1, D[j - 1] + 1, P[j - 1] + (x != y))
    return int(D[-1])


def speak(words, path, voice="Samantha", rate=170):
    """A mono wav with each (word, ms) spoken by the system voice at its time (or right after the previous word, if
    that one is still speaking). Returns its length in seconds."""
    sr, track = 22050, np.zeros(0, np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (w, at) in enumerate(words):
            aiff, wav = Path(tmp) / f"{i}.aiff", Path(tmp) / f"{i}.wav"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), w], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", str(sr), str(wav)],
                           check=True)
            _, x = wavfile.read(wav)
            x = x.astype(np.float32) / 32768
            start = max(int(at / 1e3 * sr), len(track))
            track = np.concatenate([track, np.zeros(start - len(track), np.float32), x])
    wavfile.write(path, sr, (np.clip(track, -1, 1) * 32767).astype(np.int16))
    return len(track) / sr


def render(d, path, playing, audio_s):
    """One pass over a sentence to its own silent clip: lane `playing` (0 = the true sentence, then each decoder)
    reveals its phonemes and words under the cursor, lanes already played are shown whole, later lanes are hidden."""
    dur = d["nBins"] * d["binMs"] / 1e3
    decs = {x["name"]: x for x in d["decoders"]}
    lanes = [("truth", "#7a7a7a", "true\nsentence", d["truth"]["phonemes"], [(w, a) for w, a, _ in d["truth"]["words"]],
              None)]
    for key, name in LANES:
        x = decs[name]
        lanes.append((key, style.COLOR[key], style.LABEL[key],
                      x["segments"], [], f"phoneme error {x['per']:.0%}"))
    total = max(dur, audio_s) + HOLD_S
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(1 + 2 * len(lanes), 1, height_ratios=[2.0] + [0.5, 0.38] * len(lanes),
                          hspace=0.35, left=0.15, right=0.97, top=0.86, bottom=0.05)
    fig.text(0.15, 0.94, f"Attempted sentence: ‘{d['sentence']}’", fontsize=19, fontweight="bold")
    key_p, col_p = lanes[playing][0], lanes[playing][1]
    fig.text(0.15, 0.905, "now playing: " + ("the sentence T12 attempted (synthetic voice)" if playing == 0 else
                                            f"what the {style.LABEL[key_p].replace(chr(10), ' ')} decoded (synthetic voice)"),
             fontsize=12, color=col_p)
    fig.text(0.97, 0.905, f"T12, session {d['session'].split('.', 1)[1]}, test sentence {d['trial']}", fontsize=11,
             color=style.MUTED, ha="right")
    ax = fig.add_subplot(gs[0])
    H = heat(d)
    ax.imshow(H, aspect="auto", origin="lower", cmap="Greys", extent=[0, dur, 0, H.shape[0]],
              vmin=np.percentile(H, 2), vmax=np.percentile(H, 99.5))
    ax.set_yticks([64, 192], ["threshold\ncrossings", "spike-band\npower"], fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, dur)
    ax.grid(False)
    cursors = [ax.axvline(0, color=style.INK, lw=2)]
    live = None
    for i, (key, color, label, segs, ws, score) in enumerate(lanes):
        ax = fig.add_subplot(gs[1 + 2 * i])
        strip(ax, segs, color, dur)
        ax.set_ylabel(label + ("" if i == 0 else "\nphonemes"), rotation=0, ha="right", va="center", fontsize=12,
                      color=color, fontweight="bold" if key in style.HIGHLIGHT else "normal")
        if i == playing:
            ax.add_patch(Rectangle((0, 0), dur, 1, fill=False, edgecolor=color, lw=3, clip_on=False, zorder=6))
        cover = ax.add_patch(Rectangle((0, 0), dur, 1, facecolor="white", edgecolor="none", zorder=5))
        cursors.append(ax.axvline(0, color=style.INK, lw=2, zorder=7))
        tx = fig.add_subplot(gs[2 + 2 * i])
        tx.axis("off")
        if i == 0:
            tx.text(-0.005, 0.5, "words", transform=tx.transAxes, ha="right", va="center", fontsize=11,
                    color=style.MUTED)
        wt = tx.text(0.0, 0.5, "", transform=tx.transAxes, va="center", fontsize=15, color=color)
        st = tx.text(1.0, 0.5, "", transform=tx.transAxes, va="center", ha="right", fontsize=12, color=style.MUTED)
        if i < playing:                                    # already played: shown whole
            cover.set_visible(False)
            wt.set_text(" ".join(w for w, _ in ws))
            st.set_text(score or "")
        elif i == playing:
            live = (cover, wt, st, ws, score)
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    with writer.saving(fig, str(path), dpi=100):
        cover, wt, st, ws, score = live
        n = max(int(round(total * FPS)), 1)                # exactly n frames; main() pads the audio to n / FPS
        for k in range(n):
            t = min(k / FPS, dur)
            for c in cursors:
                c.set_xdata([t, t])
            cover.set_x(t)
            cover.set_width(max(dur - t, 0))
            wt.set_text(" ".join(w for w, at in ws if at / 1e3 <= k / FPS))
            st.set_text(score if (score and k / FPS >= max(dur, audio_s)) else "")
            writer.grab_frame()
    plt.close(fig)
    return n / FPS


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--trials", default=str(WORKSPACE / "viewer_final"))
    pa.add_argument("--output", default=str(HERE / "figures" / "video_speech.mp4"))
    pa.add_argument("--chip-logits", default=str(WORKSPACE / "final_results/speech_v2c/chip_refit_logits.npz"),
                    help="the speech Speck SNN's logits from the devkit (speech_chip.py --mode chip): its lane is "
                         "added after the LSNN's when the file exists")
    pa.add_argument("--s3-logits", default=None, help="round 21 S3 (soft reset) test logits in the chip model "
                    "(S3_results/test_logits.npz): adds its lane, greedy as the others")
    pa.add_argument("--ngram", default=None, help="ngram_lm.py result json for those logits: adds S3 + the phoneme "
                    "n-gram (its order; each sentence at the alpha, beta chosen on the other half of the sentences)")
    pa.add_argument("--data-dir", default=str(WORKSPACE / "speech_rebuild/derived/snnData"))
    a = pa.parse_args()
    chip = np.load(a.chip_logits) if os.path.isfile(a.chip_logits) else None
    s3 = np.load(a.s3_logits) if a.s3_logits else None
    lm, lmj = None, None
    if a.ngram:
        import sys
        sys.path[:0] = [str(HERE.parent / "speech"), str(HERE.parent / "speech" / "scripts")]
        from snnDecoder.dataset import load_partitions
        from ngram_lm import KneserNey
        lmj = json.load(open(a.ngram))
        ss = sorted(x for x in os.listdir(a.data_dir) if x.startswith("t12"))
        test = {tuple(int(v) for v in t) for p in load_partitions(a.data_dir, ss, "test") for t in p.labels}
        lm = KneserNey([t for p in load_partitions(a.data_dir, ss, "train") for t in p.labels
                        if tuple(int(v) for v in t) not in test], lmj["order"])     # as ngram_lm.py: scored left out
    style.COLOR.update(speck_s3="#B07A00", speck_lm="#7A4E00")
    style.LABEL.update(speck_s3="Speck, soft reset\n(chip model)", speck_lm="Speck, soft reset\n+ phoneme 5-gram")
    style.HIGHLIGHT = set(style.HIGHLIGHT) | {"speck_s3", "speck_lm"}
    if chip is not None and ("speck", "SpeckChip") not in LANES:
        LANES.append(("speck", "SpeckChip"))
    style.apply()
    style.LABEL["speck"] = "Speck" if s3 is None else "Speck\n(on the chip)"
    if s3 is not None and ("speck_s3", "SpeckS3") not in LANES:
        LANES.append(("speck_s3", "SpeckS3"))
    if lm is not None and ("speck_lm", "SpeckS3LM") not in LANES:
        LANES.append(("speck_lm", "SpeckS3LM"))
    style.LABEL["lsnn"] = "LSNN"
    tmp = Path(tempfile.mkdtemp(prefix="paper_speech_video_"))
    clips = []
    for i, f in enumerate(sorted(glob.glob(os.path.join(a.trials, "trial_*.json")))):
        d = json.load(open(f))
        if chip is not None:
            key = f"{d['session']}/{d['trial']}"
            if key not in chip.files:
                raise SystemExit(f"{a.chip_logits} has no trial {key}")
            x = chip_decoder(chip[key], d["classNames"])
            truth = [p for p, _, _ in d["truth"]["phonemes"]]
            x["per"] = edit_distance([p for p, _, _ in x["segments"]], truth) / len(truth)
            d["decoders"].append(x)
        key = f"{d['session']}/{d['trial']}"
        truth = [p for p, _, _ in d["truth"]["phonemes"]]
        extra = []
        if s3 is not None:
            extra.append(dict(chip_decoder(s3[key], d["classNames"]), name="SpeckS3"))
        if lm is not None:
            al, be = lmj["chosen"][f"half{int(d['trial']) % 2}"]
            extra.append(lm_decoder(s3[key], d["classNames"], lm, al, be, "SpeckS3LM", lmj.get("beam", 32)))
        for x in extra:
            x["per"] = edit_distance([p for p, _, _ in x["segments"]], truth) / len(truth)
            d["decoders"].append(x)
        print(f"{os.path.basename(f)}: '{d['sentence']}'" + "".join(f" | {x['name']} {x['per']:.0%}" for x in extra))
        for p in range(1 + len(LANES)):
            x = {x["name"]: x for x in d["decoders"]}
            segs = d["truth"]["phonemes"] if p == 0 else x[LANES[p - 1][1]]["segments"]
            wav, silent, clip = tmp / f"s{i}_{p}.wav", tmp / f"s{i}_{p}_silent.mp4", tmp / f"s{i}_{p}.mp4"
            audio_s = speak_phonemes(word_runs(segs), wav)
            total = render(d, silent, p, audio_s)
            sr, x = wavfile.read(wav)
            n_samp = int(round(total * sr))
            clips.append((silent, sr, np.pad(x, (0, max(n_samp - len(x), 0)))[:n_samp]))
    # one continuous soundtrack, each pass's audio padded to exactly its clip's length, muxed once: clips that each
    # carry their own AAC track drift from the picture by the encoder's padding at every join
    (tmp / "list.txt").write_text("".join(f"file '{v}'\n" for v, _, _ in clips))
    video, track = tmp / "video.mp4", tmp / "audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", str(video)], check=True)
    wavfile.write(track, clips[0][1], np.concatenate([x for _, _, x in clips]))
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(track), "-map", "0:v", "-map",
                    "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", a.output], check=True)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
