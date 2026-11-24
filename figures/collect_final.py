"""
The final results tables, from the final runs only: data/final_fidelity.csv and data/final_efficiency.csv.
(collect_fidelity.py / collect_efficiency.py hold the earlier, estimate-based tables and are left as they were.)

Fidelity, one row per (experiment, group, session, decoder, metric), each experiment in its native metric:

    motor    velocity correlation (cc, mean of x and y) and RMSE, per session; groups indy, loco
    jenkins  the same for monkey J (192 channels, its own experiment: denser input and a different Speck SNN)
             (final_v2/<subject>/combined_metrics.json: every decoder, the Speck SNN scored on the chip at a 3 ms wait
             with each session's layer 1 factor, chosen without labels on training data)
    audio    spectral match within each held-out word (rho_word: sound frames only, one point per word epoch) of the
             verified-word cohort; the Speck SNN is the hard-reset words network on the chip (layer 1 x4), averaged
             over its 3 seeds per word, with the same network in software as speck_float
    speech   phoneme error per session, greedy, no language model (decoderComparison_final.json); the Speck SNN is
             round 17's S3 (top 128 channels x 4 delays, chosen 2026-11-16 for the 4-delay structure the other
             Speck SNNs share) on the chip at layer 1 x2.5 (speech_speck/results/S3/chip.json), S3 in software as
             speck_float (torch_test.json). Round 16's R3 is no longer used.
             Kalman and Wiener filters are left out of speech: they do not decode phonemes and are no reference.

Efficiency, one row per (experiment, decoder), every CPU decoder measured in ONE run on the Speck laptop
(chip/final_bench.py: one thread, one output at a time, PyTorch/NumPy, Intel RAPL package
energy minus the idle power measured before and after):

    latency_ms     median compute time per output
    energy_uj      energy per output (CPU decoders: RAPL, idle subtracted; Speck SNN: chip + host, below)
    chip_uj        the Speck chip's own energy per output, from the devkit's power monitor
    host_uj        the host's share for the Speck SNN (input buffer, delays, readout) under RAPL
    interval_ms    the time between outputs: the real-time budget

    The Kalman filter is timed at its steady state (kf; the full recursion, kf_full, is kept as a column): its gain
    converges within the first second, so a deployed filter runs x <- A x + K (z - H x).
"""

from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_fidelity import AUDIO, SAMPLES_BEFORE, WORDS, sound_frames, spectral_r

HERE = Path(__file__).parent
DL = WORKSPACE
MOTOR = DL / "final_v2"
FINAL = DL / "final_results"
SPEECH = DL / "speechBCI-main/figures/inference_final/decoderComparison_final.json"
BENCH = DL / "results_final_batch/final_bench.json"
SPEECH_S3 = DL / "final_results/speech_v2c"         # round 19 V2 (two networks, one chip, readout refitted): the final speech Speck SNN
SPEECH_CHIP = SPEECH_S3 / "chip.json"

MOTOR_NAMES = {"kf": "kf", "wf": "wf", "lstm": "lstm", "qrnn": "qrnn", "snn": "lsnn", "speck": "speck"}
AUDIO_NAMES = {"kalman_filter": "kf", "wiener_filter": "wf", "lstm_rnn": "lstm", "lsnn": "lsnn", "gru_rnn": "gru",
               "dense_nn": "dense_nn", "simple_rnn": "simple_rnn", "wiener_cascade": "wiener_cascade"}
AUDIO_SEEDS = ["s11", "s1337", "s99"]
SPEECH_NAMES = {"LSTM96": "lstm", "LSNN": "lsnn", "GRU96": "gru", "RNN": "paper_rnn", "DelayNet": "delaynet"}
BENCH_NAMES = {"kf": "kf", "wf": "wf", "lstm": "lstm", "qrnn": "qrnn", "lsnn": "lsnn", "gru": "gru",
               "dense_nn": "dense_nn", "simple_rnn": "simple_rnn", "wiener_cascade": "wiener_cascade",
               "paper_rnn": "paper_rnn", "kalman": "kf", "wiener": "wf"}


def motor_rows():
    rows = []
    for subj in ("indy", "loco", "jenkins"):
        m = json.load(open(MOTOR / subj / "combined_metrics.json"))
        for session, decs in m.items():
            for k, v in decs.items():
                if k in MOTOR_NAMES and v:
                    exp = "jenkins" if subj == "jenkins" else "motor"
                    rows.append((exp, subj, session, MOTOR_NAMES[k], "cc", (v["cc_x"] + v["cc_y"]) / 2))
                    rows.append((exp, subj, session, MOTOR_NAMES[k], "rmse", v["rmse"]))
    return rows


def audio_rows():
    """Spectral match per held-out word epoch (sound frames only), as collect_fidelity.word_rows, plus the Speck SNN
    on the chip and in software, its 3 seeds averaged per word."""
    pipe = AUDIO / WORDS / "pipeline"
    eps = [r for r in csv.DictReader(open(AUDIO / WORDS / "prep" / "epochs.csv")) if r["split"] == "test"]
    base = min(int(r["frame_start"]) for r in eps) - SAMPLES_BEFORE
    y = np.load(pipe / "window_targets_test.npy")
    preds = {d: [np.load(pipe / f"{k}_predict_test.npy")] for k, d in AUDIO_NAMES.items()
             if (pipe / f"{k}_predict_test.npy").is_file()}
    for name, kind in (("speck", "chip_refit"), ("speck_float", "float")):
        preds[name] = [np.load(FINAL / f"audio_preds/words_P5_{s}/{kind}.npy") for s in AUDIO_SEEDS]
    rows = []
    for r in eps:
        a, b = int(r["frame_start"]) - SAMPLES_BEFORE - base, int(r["frame_end"]) - SAMPLES_BEFORE - base
        if a < 0 or b > len(y):
            continue
        m = sound_frames(y[a:b].mean(1))
        for d, ps in preds.items():
            v = np.mean([spectral_r(y[a:b][m], p[a:b][m]) for p in ps])
            rows.append(("audio", "word", f"{r['word']}_{r['date']}_{r['epoch_idx']}", d, "rho_word", v))
    return rows


def speech_rows():
    s = json.load(open(SPEECH))
    rows = [("speech", "t12", session, d, "per", v)
            for k, d in SPEECH_NAMES.items() for session, v in s[k]["per_session"].items()]
    chip = next(iter(json.load(open(SPEECH_S3 / "chip_refit.json"))["results"].values()))["per_session"]
    soft = next(iter(json.load(open(SPEECH_S3 / "torch_test.json"))["results"].values()))["per_session"]
    rows += [("speech", "t12", session, "speck", "per", v) for session, v in chip.items()]
    rows += [("speech", "t12", session, "speck_float", "per", v) for session, v in soft.items() if session in chip]
    return rows


def efficiency():
    B = json.load(open(BENCH))["results"]
    # the final speech network's own host share (laptop, RAPL): V2+C once measured, else V2's (2026-11-17)
    v2c = DL / "results_bench_v2c/speech_V2C_host.json"
    v2 = v2c if v2c.is_file() else DL / "results_speck_deploy/bench/speech_V2_host.json"
    if v2.is_file():
        B["speech/speck_host"] = json.load(open(v2))["results"]["speech/speck_host"]
    # 2026-11-23: the audio decoders re-measured in one session, twice (chip/final_bench.py --only audio --seconds 10 on the
    # Speck laptop, incl. the 4 ms networks), averaged; the paper's audio LSNN and Speck host are now the 4 ms ones
    runs = sorted(DL.glob("*/bench_audio_all_run*.json")) + sorted(DL.glob("*/results/bench_audio_all_run*.json"))
    if runs:
        R = [json.load(open(f))["results"] for f in runs]
        B = {k: v for k, v in B.items() if not k.startswith("audio/")}
        for key in R[0]:
            if all(key in r for r in R):
                B[key] = {**R[0][key], **{q: float(np.mean([r[key][q] for r in R]))
                                          for q in ("median_ms", "p95_ms", "energy_uj", "energy_raw_uj", "power_w")}}
        for new, old in (("audio/lsnn4", "audio/lsnn"), ("audio/speck_host4", "audio/speck_host")):
            if new in B:
                B[old.replace("audio/", "audio_40/")] = B.pop(old)       # the 40 ms versions, kept aside (not plotted)
                B[old] = B.pop(new)
        B = {k: v for k, v in B.items() if not k.startswith("audio_40/")}
        print(f"audio latency/energy: mean of {len(runs)} benchmark runs ({', '.join(f.name for f in runs)})")
    out = []
    for key, r in B.items():
        exp, dec = key.split("/")
        if dec in ("kf_full", "speck_host"):
            continue
        if exp == "speech" and dec in ("kalman", "wiener"):
            continue
        out.append(dict(experiment=exp, decoder=BENCH_NAMES[dec], latency_ms=r["median_ms"],
                        latency_p95_ms=r["p95_ms"], energy_uj=r["energy_uj"], interval_ms=r["interval_ms"],
                        measured_by="laptop CPU (RAPL, idle subtracted)",
                        kf_full_ms=B.get(f"{exp}/kf_full", {}).get("median_ms") if dec == "kf" else np.nan))
    # the Speck SNN: the chip (devkit power monitor; time per output = the host loop's wall time, waits included)
    # plus the host's share under RAPL
    chips = {}
    for subj, exp in (("indy", "motor"), ("loco", "motor_loco"), ("jenkins", "motor_jenkins"),
                      ("jenkins_m1", "motor2")):
        val = DL / "results_snn_bci_laptop_final/motor/final_v3" / subj / "efficiency_summary.json"   # validation-selected
        e = json.load(open(val if val.is_file() else MOTOR / subj / "efficiency_summary.json"))["laptop"]["speck"]
        chips[exp] = (1e3 * e["latency_s"], 1e6 * e["energy_j"], 1e3 * e["power_w"])
    # 2026-11-23: the audio Speck SNN on real 4 ms spikes (FT1) on the devkit, both monkeys, 3 seeds each
    a = [v for f in sorted((DL / "results_deploy_fine/audio").glob("*_FT1_*.json")) for v in json.load(open(f)).values()]
    chips["audio"] = (np.mean([v["latency_output_ms"] for v in a]), np.mean([v["energy_output_uj"] for v in a]),
                      np.mean([v["power_mw"] for v in a]))
    # 2026-11-23: Motor (II)'s validation-selected Speck SNNs on the devkit (results_deploy_jenkins_val): the chip loop's
    # time per 4 ms step, and its mean power x that time
    jv = [json.load(open(f)) for f in sorted((DL / "results_deploy_jenkins_val").glob("sub-Jenkins_*.json"))]
    if jv:
        ms = float(np.mean([j["ms_per_step"] for j in jv]))
        mw = float(np.mean([j["power_mw"] for j in jv]))
        chips["motor2"] = (ms, mw * ms, mw)
    c = json.load(open(SPEECH_CHIP))["chip"]
    # the same rule as every other decoder: energy while computing the output = the chip's mean power over its loop x
    # the loop time per output (4 bins); not power x the whole 80 ms interval (speech_chip.py's
    # energy_uj_per_output_80ms), which billed speech alone for idle time
    chips["speech"] = (4 * c["ms_per_bin"], c["power_mw"] * 4 * c["ms_per_bin"], c["power_mw"])
    for exp, (lat, chip_uj, mw) in chips.items():
        bench_exp = "motor" if exp in ("motor_loco", "motor2") else exp
        host = B[f"{bench_exp}/speck_host"]
        out.append(dict(experiment=exp, decoder="speck", latency_ms=lat + host["median_ms"], chip_ms=lat,
                        energy_uj=chip_uj + host["energy_uj"], chip_uj=chip_uj, host_uj=host["energy_uj"],
                        chip_mw=mw, interval_ms=host["interval_ms"], measured_by="Speck chip + host CPU (RAPL)"))
    df = pd.DataFrame(out)
    # loco and Jenkins's M1 array (motor2) use the indy-sized CPU decoders (same architectures, same 96 channels);
    # their Speck chips are their own runs
    same = df[(df.experiment == "motor") & (df.decoder != "speck")]
    df = pd.concat([df, same.assign(experiment="motor_loco"), same.assign(experiment="motor2")], ignore_index=True)
    df["power_mw"] = df.energy_uj / df.interval_ms
    return df


def main():
    rows = motor_rows() + audio_rows() + speech_rows()
    fid = pd.DataFrame(rows, columns=["experiment", "group", "session", "decoder", "metric", "value"])
    fid.to_csv(HERE / "data" / "final_fidelity.csv", index=False)
    eff = efficiency()
    eff.to_csv(HERE / "data" / "final_efficiency.csv", index=False)
    pd.set_option("display.width", 200)
    print(fid.groupby(["experiment", "group", "metric", "decoder"]).value.agg(["count", "mean", "median"])
          .round(3).to_string())
    print(eff[["experiment", "decoder", "latency_ms", "interval_ms", "energy_uj", "chip_uj", "host_uj", "power_mw"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
