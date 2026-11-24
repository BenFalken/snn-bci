"""
Fidelity for the four-experiment figures (fig_four.py), one data point per evaluation unit:

    motor1   Motor (I), NeuroBench primate reaching (monkeys I and L, 96-channel M1): velocity correlation and RMSE per
             session, exactly as collect_final.py (final_v2/<subj>/combined_metrics.json)
    motor2   Motor (II), Jenkins on its 96-channel M1 array (subject jenkins_m1): the same decoders and metrics per
             session, from final_v2/jenkins_m1/combined_metrics.json once the laptop run has made it (absent until then)
    audio    per held-out WORD epoch: band_r = the paper's metric (Heelan et al. 2019), each of the 128 mel bands'
             Pearson r between decoded and true over the epoch, Fisher-z averaged; rho_word (our frame-wise spectral
             match on sound frames, collect_final.py) kept as a second metric. The Speck SNN is the mean of its 3 seeds
    speech   per test SENTENCE: greedy phoneme error (no language model) = that sentence's edit distance / its phoneme
             count, from final_results/perTrialPER_final_s3.json (perTrialPER_final.json from speech/analysis/perTrialPER.py on Oscar, plus
             the Speck SNN V2 on the chip, readout refitted, and in software, scored on the Mac on the same sentences); per session,
             as before, while that file is absent (group 'session' instead of 'sentence')

    python collect_four.py      -> data/four_fidelity.csv (efficiency stays data/final_efficiency.csv)
"""

import csv
import json

import numpy as np
import pandas as pd

from collect_fidelity import align_raw, fisher_mean_r, sound_frames, spectral_r
from collect_final import (AUDIO, AUDIO_NAMES, AUDIO_SEEDS, FINAL, HERE, MOTOR, MOTOR_NAMES, SAMPLES_BEFORE, WORDS,
                           speech_rows)

PER_TRIAL = FINAL / "perTrialPER_final_v2c.json"  # perTrialPER_final.json + V2+C (final, round 20) on the same 600 sentences
PER_TRIAL_NAMES = {"RNN": "paper_rnn", "GRU96": "gru", "LSTM96": "lstm", "LSNN": "lsnn", "SpeckV2C": "speck_float",
                   "SpeckChipV2C": "speck"}
JENKINS_REFIT = FINAL / "jenkins_refit"         # the Jenkins Speck SNN's readout refitted in software (jenkins_refit.py)
JENKINS_VAL = FINAL.parent / "results_deploy_jenkins_val"   # the validation-selected Jenkins Speck SNNs on the devkit
SPECK_VAL = FINAL.parent / "results_snn_bci_laptop_final" / "motor" / "final_v3"   # Motor (I) Speck, validation-selected


VAL = MOTOR.parent / "final_v3"           # validation-selected checkpoints (train_snn.py --val-frac), where scored


def motor_rows(exp, subjects):
    rows = []
    for subj in subjects:
        path = MOTOR / subj / "combined_metrics.json"
        if not path.is_file():
            print(f"[{exp}] no {path}: left out")
            continue
        val = VAL / subj / "combined_metrics.json"
        val = json.load(open(val)) if val.is_file() else {}
        spk = SPECK_VAL / subj / "combined_metrics.json"   # the validation-selected Speck SNNs on the devkit (laptop)
        spk = json.load(open(spk)) if spk.is_file() else {}
        gru = MOTOR / subj / "gru_metrics.json"           # the re-implemented GRUs (motor/gru_reference/score_gru.py)
        gru = json.load(open(gru)) if gru.is_file() else {}
        for session, g in gru.items():
            if "aegru" in g:
                m = g["aegru"]["seed_mean"]
                rows.append((exp, subj, session, "aegru", "cc", (m["cc_x"] + m["cc_y"]) / 2))
                rows.append((exp, subj, session, "aegru", "rmse", m["rmse"]))
        for session, decs in json.load(open(path)).items():
            if "snn" in val.get(session, {}):              # the LSNN: its validation-selected checkpoint
                decs = dict(decs, snn=val[session]["snn"])
            if "speck" in spk.get(session, {}):            # the Speck SNN: its validation-selected checkpoint, on the chip
                decs = dict(decs, speck=spk[session]["speck"])
            for k, v in decs.items():
                if k in MOTOR_NAMES and v:
                    cc, rm = (v["cc_x"] + v["cc_y"]) / 2, v["rmse"]
                    rf = JENKINS_REFIT / f"{session}.json"
                    rv = JENKINS_VAL / f"{session}.json"
                    if k == "speck" and subj == "jenkins_m1" and rv.is_file():     # validation-selected, on the chip,
                        b = json.load(open(rv))["refitted"]                          # its refitted readout live
                        cc, rm = (b["cc_x"] + b["cc_y"]) / 2, b["rmse"]
                    elif k == "speck" and subj == "jenkins_m1" and rf.is_file():   # the final, refitted readout
                        r = json.load(open(rf))
                        b = r[r["chosen"]]
                        cc, rm = (b["cc_x"] + b["cc_y"]) / 2, b["rmse_chip_test"]
                    rows.append((exp, subj, session, MOTOR_NAMES[k], "cc", cc))
                    rows.append((exp, subj, session, MOTOR_NAMES[k], "rmse", rm))
    return rows


def audio_rows():
    """2026-11-18: the expanded cohorts, both monkeys (collect_audio_novel.py -> data/audio_novel_fidelity.csv): every
    test utterance, vocabulary and novel words; group = the monkey. The earlier Neil-only rows: audio_rows_neil20."""
    t = pd.read_csv(HERE / "data" / "audio_novel_fidelity.csv")
    t = t[t.decoder != "speck_chip_trained"]
    rows = []
    for r in t.itertuples():
        g = r.monkey.lower()
        rows.append(("audio", g, r.utterance, r.decoder, "band_r", r.band_r))
        rows.append(("audio", g, r.utterance, r.decoder, "rho_word", r.rho_word))
    return rows


def audio_rows_neil20():
    pipe = AUDIO / WORDS / "pipeline"
    eps = [r for r in csv.DictReader(open(AUDIO / WORDS / "prep" / "epochs.csv")) if r["split"] == "test"]
    base = min(int(r["frame_start"]) for r in eps) - SAMPLES_BEFORE
    y = np.load(pipe / "window_targets_test.npy")
    preds = {d: [np.load(pipe / f"{k}_predict_test.npy")] for k, d in AUDIO_NAMES.items()
             if (pipe / f"{k}_predict_test.npy").is_file()}
    if "kf" in preds:                               # the Kalman filter's frame split, put on the window split's frames
        preds["kf"] = [align_raw(pipe, preds["kf"][0])]
    for name, kind in (("speck", "chip_refit"), ("speck_float", "float")):
        preds[name] = [np.load(FINAL / f"audio_preds/words_P5_{s}/{kind}.npy") for s in AUDIO_SEEDS]
    rows = []
    for r in eps:
        a, b = int(r["frame_start"]) - SAMPLES_BEFORE - base, int(r["frame_end"]) - SAMPLES_BEFORE - base
        if a < 0 or b > len(y):
            continue
        word = f"{r['word']}_{r['date']}_{r['epoch_idx']}"
        m = sound_frames(y[a:b].mean(1))
        for d, ps in preds.items():
            ok = np.isfinite(ps[0][a:b, 0])
            if ok.sum() < 3:
                continue
            rows.append(("audio", "word", word, d, "band_r",
                         np.mean([fisher_mean_r(y[a:b][ok], p[a:b][ok]) for p in ps])))
            rows.append(("audio", "word", word, d, "rho_word",
                         np.mean([spectral_r(y[a:b][m & ok], p[a:b][m & ok]) for p in ps])))
    return rows


def speech_sentence_rows():
    if not PER_TRIAL.is_file():
        print(f"[speech] no {PER_TRIAL}: per-session phoneme error for now")
        return [("speech", "session", s, d, "per", v) for _, _, s, d, _, v in speech_rows()]
    t = json.load(open(PER_TRIAL))
    n = np.asarray(t["phonemes"], float)
    rows = []
    for k, d in PER_TRIAL_NAMES.items():
        if k in t:
            for key, e, ni in zip(t["keys"], t[k]["edits"], n):
                rows.append(("speech", "sentence", key, d, "per", e / ni))
    return rows


def main():
    rows = (motor_rows("motor1", ("indy", "loco")) + motor_rows("motor2", ("jenkins_m1",)) + audio_rows()
            + speech_sentence_rows())
    fid = pd.DataFrame(rows, columns=["experiment", "group", "session", "decoder", "metric", "value"])
    fid.to_csv(HERE / "data" / "four_fidelity.csv", index=False)
    pd.set_option("display.width", 200)
    print(fid.groupby(["experiment", "group", "metric", "decoder"]).value.agg(["count", "mean", "median"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
