"""
One fidelity table for the three experiments: data/fidelity.csv, one row per (experiment, group, session, decoder,
metric), each experiment in its native metric:

    motor    velocity correlation (cc, mean of x and y) and RMSE (deck measure, per axis), per session; groups indy, loco, jenkins
             (spike_bmi results/final/<subject>/combined_metrics.json: every decoder on the laptop, Speck on the chip)
    audio    mel-spectrogram correlation on the test split, per recording day (and per multi-day dataset, group 'multi')
             (AuditoryDecoding pipeline_data/.../results/<decoder>_regression_metrics.csv)
    speech   phoneme error (per) and word error with the language model (wer), per session
             (speechBCI figures/inference_final/decoderComparison_final.json)
"""

from repo_paths import WORKSPACE, AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MOTOR = WORKSPACE / "results/final"
AUDIO = AUDITORY_DATA / "Neil/aSTG"
SPEECH = WORKSPACE / "speechBCI-main/figures/inference_final/decoderComparison_final.json"

MOTOR_NAMES = {"kf": "kf", "wf": "wf", "lstm": "lstm", "qrnn": "qrnn", "snn": "lsnn", "speck": "speck"}
AUDIO_NAMES = {"kalman_filter": "kf", "wiener_filter": "wf", "lstm_rnn": "lstm", "lsnn": "lsnn", "gru_rnn": "gru",
               "dense_nn": "dense_nn", "simple_rnn": "simple_rnn", "wiener_cascade": "wiener_cascade", "speck_snn": "speck"}
AUDIO_DAYS = ["07_19_11", "07_20_10", "07_31_13", "08_03_10", "08_06_13"]
AUDIO_MULTI = ["COMBINED_naive", "COMBINED_per_day_normalized_causal_clip5.0",
               "FIVEWORD_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13",
               "FIVEWORD_VERIFIED_07_19_11_07_20_10_07_31_13_08_03_10_08_06_13",
               "ALLWORDS_VERIFIED_DAYNORM"]
SPEECH_NAMES = {"Kalman": "kf", "Wiener": "wf", "LSTM96": "lstm", "LSNN": "lsnn", "GRU96": "gru",
                "RNN": "paper_rnn", "DelayNet": "delaynet"}


WORDS = "ALLWORDS_VERIFIED_DAYNORM"
SAMPLES_BEFORE = 8                                    # window row i targets frame i + 8


def sound_frames(loudness):
    """The frames of one word epoch where the word sounds: band-mean loudness above its floor (20th percentile) by a
    quarter of the way to its peak (98th). An epoch is ~4.6 s holding the word twice; this keeps both utterances."""
    lo, hi = np.percentile(loudness, 20), np.percentile(loudness, 98)
    return loudness > lo + 0.25 * (hi - lo)


def fisher_mean_r(y, p):
    rs = [np.corrcoef(y[:, j], p[:, j])[0, 1] for j in range(y.shape[1]) if y[:, j].std() > 0 and p[:, j].std() > 0]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


def spectral_r(y, p):
    """Spectral match: at each frame, the correlation of the predicted with the true 128-band spectrum; Fisher mean
    over the frames. Stable on the few frames of one word, unlike a per-band correlation over time."""
    rs = [np.corrcoef(a, b)[0, 1] for a, b in zip(y, p) if a.std() > 0 and b.std() > 0]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


def word_rows():
    """Spectral match per held-out word epoch of the verified-word cohort: each decoder's test prediction scored only
    on the frames where the word sounds (silence and padding excluded), one data point per epoch."""
    pipe = AUDIO / WORDS / "pipeline"
    eps = [r for r in csv.DictReader(open(AUDIO / WORDS / "prep" / "epochs.csv")) if r["split"] == "test"]
    base = min(int(r["frame_start"]) for r in eps) - SAMPLES_BEFORE
    y = np.load(pipe / "window_targets_test.npy")
    preds = {d: np.load(pipe / f"{k}_predict_test.npy") for k, d in AUDIO_NAMES.items()
             if (pipe / f"{k}_predict_test.npy").is_file()}
    rows = []
    for r in eps:
        a, b = int(r["frame_start"]) - SAMPLES_BEFORE - base, int(r["frame_end"]) - SAMPLES_BEFORE - base
        if a < 0 or b > len(y):
            continue
        m = sound_frames(y[a:b].mean(1))
        for d, p in preds.items():
            rows.append(("audio", "word", f"{r['word']}_{r['date']}_{r['epoch_idx']}", d, "rho_word",
                         spectral_r(y[a:b][m], p[a:b][m])))
    return rows


def main():
    rows = []
    for subj in ("indy", "loco", "jenkins"):
        if not (MOTOR / subj / "combined_metrics.json").is_file():     # jenkins until its inference is pulled
            continue
        m = json.load(open(MOTOR / subj / "combined_metrics.json"))
        for session, decs in m.items():
            for k, v in decs.items():
                if k not in MOTOR_NAMES or not v:
                    continue
                d = MOTOR_NAMES[k]
                rows.append(("motor", subj, session, d, "cc", (v["cc_x"] + v["cc_y"]) / 2))
                rows.append(("motor", subj, session, d, "rmse", v["rmse"]))
    for group, dates in (("day", AUDIO_DAYS), ("multi", AUDIO_MULTI)):
        for date in dates:
            for k, d in AUDIO_NAMES.items():
                f = AUDIO / date / "results" / f"{k}_regression_metrics.csv"
                if f.is_file():
                    x = pd.read_csv(f)
                    rows.append(("audio", group, date, d, "rho", float(x.test_mean_rho[0])))
                    rows.append(("audio", group, date, d, "rho_valid", float(x.valid_mean_rho[0])))
    chip = HERE / "data" / "laptop" / "audio_chip_l1x4.json"       # audio Speck on the devkit (layer 1 threshold x4)
    if chip.is_file():
        C = json.load(open(chip))
        for i, row in enumerate(rows):
            e, g, date, d, metric, v = row
            if e == "audio" and d == "speck" and metric == "rho" and "r_chip" in C.get(date, {}):
                rows[i] = (e, g, date, d, metric, C[date]["r_chip"])         # as motor: Speck scored on the chip
                rows.append((e, g, date, "speck_float", metric, v))
    rows += word_rows()
    s = json.load(open(SPEECH))
    for k, d in SPEECH_NAMES.items():
        for session, v in s[k]["per_session"].items():
            rows.append(("speech", "t12", session, d, "per", v))
        for session, v in (s[k].get("wer_session") or {}).items():
            rows.append(("speech", "t12", session, d, "wer", v))
    # the speech Speck SNN (round 8, F3): its own decoder-comparison run (same test trials and language model)
    speck = SPEECH.parent / "decoderComparison_speck.json"
    if speck.is_file():
        S = json.load(open(speck))
        if "Speck" in S:
            for session, v in S["Speck"]["per_session"].items():
                rows.append(("speech", "t12", session, "speck", "per", v))
            for session, v in (S["Speck"].get("wer_session") or {}).items():
                rows.append(("speech", "t12", session, "speck", "wer", v))
    chip = HERE / "data" / "laptop" / "speech_chip.json"        # speech Speck on the devkit (speech_chip.py)
    if chip.is_file():
        C = json.load(open(chip))
        per = next(iter(C["results"].values()))["per_session"]   # greedy phoneme error, no language model
        for i, row in enumerate(rows):
            e, g, session, d, metric, v = row
            if e == "speech" and d == "speck" and metric == "per" and session in per:
                rows[i] = (e, g, session, d, metric, per[session])     # as motor and audio: scored on the chip
                rows.append((e, g, session, "speck_float", metric, v))
        have = {r[2] for r in rows if r[0] == "speech" and r[3] == "speck" and r[4] == "per"}
        tested = {r[2] for r in rows if r[0] == "speech" and r[3] == "lsnn" and r[4] == "per"}
        for session in sorted(tested - have):                     # no comparison run yet: the chip's sessions
            if session in per:
                rows.append(("speech", "t12", session, "speck", "per", per[session]))
    df = pd.DataFrame(rows, columns=["experiment", "group", "session", "decoder", "metric", "value"])
    os.makedirs(HERE / "data", exist_ok=True)
    df.to_csv(HERE / "data" / "fidelity.csv", index=False)
    print(df.groupby(["experiment", "metric", "decoder"]).value.agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()


def align_raw(pipe, pred):
    """The Kalman filter is fitted on the pipeline's frame-by-frame ('raw') split, whose test part starts a few frames
    after the window split's (window i is centred on frame i + 8; both splits take 80/10/10 of their own length).
    Returns its test predictions on the window split's test frames (NaN where it has none), at the offset where the
    two splits' own test targets coincide (checked: correlation 1)."""
    pipe = str(pipe)
    yw = np.load(f"{pipe}/window_targets_test.npy").mean(1)
    yr = np.load(f"{pipe}/raw_targets_test.npy").mean(1)

    def corr(o):
        j = np.arange(len(yw))
        k = j + o
        m = (k >= 0) & (k < len(yr))
        return np.corrcoef(yw[j[m]], yr[k[m]])[0, 1]
    o = max(range(-24, 25), key=corr)
    assert corr(o) > 0.999, f"raw/window test targets do not line up ({corr(o):.4f} at offset {o})"
    out = np.full((len(yw), pred.shape[1]), np.nan, dtype=float)
    j = np.arange(len(yw))
    k = j + o
    m = (k >= 0) & (k < len(pred))
    out[j[m]] = pred[k[m]]
    return out
