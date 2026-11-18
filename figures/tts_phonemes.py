"""
Speak decoded phonemes as speech, for video_speech.py. Run with the Piper environment's Python (.tts-venv), which has
the piper-tts package; video_speech.py calls it as a subprocess.

A decoder's output is a stream of phonemes in time, with silences between words, as a phoneme decoder in a speech BCI
emits it. Each run of phonemes between silences (a decoded "word") is turned from ARPAbet into IPA, synthesized as one
word by a neural voice (Piper, offline; its phoneme input [[ ... ]] skips any spelling or language model, so exactly
the decoded phonemes are spoken, coarticulated as in natural speech), and placed at the time the decoder produced it,
time-stretched (pitch kept, ffmpeg atempo) to the span it was decoded over so the voice follows the cursor. The
stretch is kept between 0.6x and 1.6x of the voice's natural pace so it stays intelligible: a span too long ends in
silence, a word too long for its span runs on, and the next word waits for it.

    .tts-venv/bin/python tts_phonemes.py VOICE.onnx RUNS.json OUT.wav

RUNS.json: [[["TH", "IH", "NG", "K"], start_ms, end_ms], ...]. Prints the track's length in seconds.
"""

import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

# ARPAbet (the decoders' 39 phonemes) -> the espeak-style IPA the Piper voices are trained on (General American)
ARPA_TO_IPA = {"AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d",
               "DH": "ð", "EH": "ɛ", "ER": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː",
               "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p",
               "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v", "W": "w",
               "Y": "j", "Z": "z", "ZH": "ʒ"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SLOWEST, FASTEST = 0.6, 1.6          # stretch limits, as a speed factor on the voice's natural pace


def to_ipa(phones):
    """One decoded word as IPA: primary stress on its first vowel, unstressed AH as schwa."""
    out, stressed = [], False
    for p in phones:
        if p not in ARPA_TO_IPA:
            continue
        if p in VOWELS and not stressed:
            out.append("ˈ" + ARPA_TO_IPA[p])
            stressed = True
        elif p == "AH":
            out.append("ə")
        else:
            out.append(ARPA_TO_IPA[p])
    return "".join(out)


def synth(voice, ipa):
    """Piper's audio for one word given as raw phonemes."""
    from piper import SynthesisConfig
    chunks = list(voice.synthesize(f"[[ {ipa} ]]", syn_config=SynthesisConfig()))
    if not chunks:
        return np.zeros(0, np.float32), voice.config.sample_rate
    return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate


def stretch(x, sr, speed):
    """Time-stretch by `speed` (>1 faster), pitch kept: ffmpeg atempo."""
    if abs(speed - 1) < 0.03 or len(x) == 0:
        return x
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.wav"), os.path.join(tmp, "b.wav")
        write(a, sr, x)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-filter:a", f"atempo={speed:.4f}", b],
                       check=True)
        return read(b)[1]


def write(path, sr, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def read(path):
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        return sr, np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768


def main():
    from piper import PiperVoice
    model, runs_path, out = sys.argv[1:4]
    voice = PiperVoice.load(model)
    sr = voice.config.sample_rate
    track = np.zeros(0, np.float32)
    for phones, a, b in json.load(open(runs_path)):
        ipa = to_ipa(phones)
        if not ipa:
            continue
        x, sr = synth(voice, ipa)
        x = np.trim_zeros(x, "fb") if np.any(x) else x
        span = max(b - a, 1) / 1e3
        natural = len(x) / sr
        x = stretch(x, sr, float(np.clip(natural / span, SLOWEST, FASTEST)))
        start = max(int(a / 1e3 * sr), len(track))
        track = np.concatenate([track, np.zeros(start - len(track), np.float32), x])
    write(out, sr, track)
    print(f"{len(track) / sr:.3f}")


if __name__ == "__main__":
    main()
