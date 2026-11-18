"""
One efficiency table for the three experiments: data/efficiency.csv, one row per (experiment, decoder):

    energy_uj      estimated energy per output (operation counts x 45 nm costs, Horowitz 2014: 4.6 pJ per 32-bit
                   multiply-add, 0.9 pJ per add, 5 pJ per on-chip memory access)
    power_mw       energy_uj / output interval
    latency_ms     compute time per output, one CPU thread
    interval_ms    time between outputs (the real-time budget)
    measured_uj, measured_mw, measured_by   real measurements where they exist (motor: laptop RAPL, Speck's monitor)

    motor    spike_bmi results/final/indy: op estimates per session (memory: 4 accesses per multiply-add, 3 per add),
             laptop latency and RAPL energy, the Speck SNN measured on the chip. The Kalman filter is counted at its
             steady state (as speech's is): spike_bmi's count charged the full covariance recursion every step

    Kalman filters everywhere: steady state, x <- A x + K (z - H x), s^2 + 2 c s multiply-adds per output. Its gain
    converges within a second of the recording's start, so recomputing it every step (P, and a c x c inverse) only adds
    arithmetic that changes nothing.
    audio    computed here with motor's rules, per 40 ms output from a 17-frame window: Wiener filter (1632 -> 128),
             Kalman filter (128 states, 96 channels, steady state), LSTM, GRU (1 x 1024) and dense NN (1024), the
             pipeline's, LSNN
             (2 x 256, the pipeline's, spike counts from running each day's trained model on its test windows);
             latency timed here (NumPy / PyTorch, one thread, Apple M-series CPU)
    speech   speechBCI efficiency_final.json: its operation counts per 80 ms output, with motor's energy rule (its own
             memory model, kept as energy_uj_speech_model, counts weight reads and state accesses); timed on one Oscar
             CPU thread
"""

from repo_paths import WORKSPACE, AUDIO, AUDITORY_DATA, MIKKO, ROOT, CHIP, SNN_RUNS  # noqa: E402  (the repository's paths.yaml)
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
MOTOR = WORKSPACE / "results/final/indy"
AUDIO_REPO = AUDIO
SNNDEC = ROOT
AUDIO_DAYS = ["07_19_11", "07_20_10", "07_31_13", "08_03_10", "08_06_13"]
SPEECH = WORKSPACE / "speechBCI-main/figures/inference_final/efficiency_final.json"
SPEECH_ROWS = {"Kalman filter per session": "kf", "Wiener filter (160 ms)": "wf", "LSTM 2x1024, 96 ep": "lstm",
               "Model B, LSNN, final (C_lsnn_nodelay96)": "lsnn", "GRU 2x1024 out80, same input, 96 ep": "gru",
               "QRNN 2x1024, 96 ep": "qrnn", "Paper RNN": "paper_rnn", "Model A, DelayNet (R_readall96)": "delaynet"}
MAC_PJ, ACC_PJ, MEM_PJ = 4.6, 0.9, 5.0
KF_MOTOR_STATES = 6


def energy_uj(macs, accs):
    """Motor's rule: MAC = 3 loads + 1 store, ACC = 2 loads + 1 store, all on-chip."""
    return (macs * MAC_PJ + accs * ACC_PJ + (4 * macs + 3 * accs) * MEM_PJ) * 1e-6


def timeit(fn, n=200):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def audio_rows():
    torch.set_num_threads(1)
    W, C, S, U = 17, 96, 128, 1024                    # window, channels, mel bands, LSTM units
    rows = []
    # Wiener filter: every window value to every band
    macs = W * C * S
    Wm, x = np.random.randn(W * C, S).astype(np.float32), np.random.randn(1, W * C).astype(np.float32)
    rows.append(dict(decoder="wf", macs=macs, accs=0, latency_ms=timeit(lambda: x @ Wm, 2000)))
    # Kalman filter, steady state (s = 128 states, c = 96 channels): the gain converges within the first second and is
    # then fixed, so a deployed filter runs x <- A x + K (z - H x): s^2 + 2 c s multiply-adds per output
    s, c = S, C
    macs = s * s + 2 * c * s
    A, H, K = (np.random.randn(*sh).astype(np.float32) for sh in ((s, s), (c, s), (s, c)))
    z, st = np.random.randn(c).astype(np.float32), np.zeros(s, np.float32)
    def kf_step():
        sm = A @ st
        return sm + K @ (z - H @ sm)
    rows.append(dict(decoder="kf", macs=macs, accs=0, latency_ms=timeit(kf_step, 2000)))
    # LSTM: per step 4 x units gates from inputs and state, then Dense to the bands (dense z-scored input)
    macs = W * (C * 4 * U + U * 4 * U) + U * S
    lstm, dense = torch.nn.LSTM(C, U, batch_first=True), torch.nn.Linear(U, S)
    xw = torch.randn(1, W, C)
    with torch.no_grad():
        lat = timeit(lambda: dense(lstm(xw)[0][:, -1]), 100)
    rows.append(dict(decoder="lstm", macs=macs, accs=0, latency_ms=lat))
    # GRU (1 x 1024, the pipeline's): per window step 3 gates from inputs and state, then Dense to the bands
    macs = W * 3 * U * (C + U) + U * S
    gru = torch.nn.GRU(C, U, batch_first=True)
    with torch.no_grad():
        lat = timeit(lambda: dense(gru(xw)[0][:, -1]), 100)
    rows.append(dict(decoder="gru", macs=macs, accs=0, latency_ms=lat))
    # dense NN (the pipeline's): the flattened window (17 x 96) -> 1024 ReLU -> 128 bands
    macs = W * C * U + U * S
    mlp = torch.nn.Sequential(torch.nn.Linear(W * C, U), torch.nn.ReLU(), torch.nn.Linear(U, S))
    xf = torch.randn(1, W * C)
    with torch.no_grad():
        lat = timeit(lambda: mlp(xf), 1000)
    rows.append(dict(decoder="dense_nn", macs=macs, accs=0, latency_ms=lat))
    # LSNN: the pipeline's model on each day's test windows; dense first layer as multiply-adds, every spike-driven
    # synapse (between layers, recurrent, readout) as an add, neuron updates (filter, adaptation, readout leak) as MACs
    sys.path.insert(0, str(AUDIO_REPO)); sys.path.insert(0, str(MIKKO)); sys.path.insert(0, str(MIKKO / "modules/decoders/torch/lsnn"))
    import lsnn as L
    per_day = []
    for day in AUDIO_DAYS:
        P_ = AUDITORY_DATA / f"Neil/aSTG/{day}/pipeline"
        ck = torch.load(P_ / "lsnn_model.pt", map_location="cpu", weights_only=False)
        net = L.LSNNNet(ck["n_inputs"], ck["n_outputs"], **ck["net_kwargs"]); net.load_state_dict(ck["state_dict"]); net.eval()
        X = torch.as_tensor(np.load(P_ / "window_features_test.npy")[:600], dtype=torch.float32)
        counts = [0.0 for _ in net.spiking]
        steps = []
        for li, layer in enumerate(net.spiking):
            orig = layer.step
            def wrapped(current, st, _o=orig, _i=li):
                s_ = _o(current, st); counts[_i] += float(s_.sum()); return s_
            layer.step = wrapped
        with torch.no_grad():
            net(X)
        n = X.shape[0]
        widths = [l.width for l in net.spiking]
        spk = [cnt / n for cnt in counts]                             # spikes per window, per layer
        n_adapt = [float(l.adapt_mask.sum()) for l in net.spiking]
        macs = W * C * widths[0] + W * (sum(widths) + 2 * sum(n_adapt) + 2 * S)
        accs = spk[0] * widths[1] + sum(sp * w for sp, w in zip(spk, widths)) + sum(spk) * S
        per_day.append((macs, accs))
        with torch.no_grad():
            lat = timeit(lambda: net(X[:1]), 50)
    macs, accs = np.mean(per_day, 0)
    rows.append(dict(decoder="lsnn", macs=macs, accs=accs, latency_ms=lat))
    rows.append(audio_speck_row())
    out = []
    for r in rows:
        e = energy_uj(r["macs"], r["accs"])
        out.append(dict(experiment="audio", decoder=r["decoder"], energy_uj=e, interval_ms=40.0, power_mw=e / 40.0,
                        latency_ms=r["latency_ms"], macs=r["macs"], accs=r["accs"]))
    return out


def audio_speck_row():
    """The Speck SNN (snn-decoders, binB, seed 1337; its whole-recording mode, as scored), estimated, not measured: every input event and spike on the
    chip drives one add per synapse it reaches (fan-out = the next layer's width); the host reads the last hidden
    layer (EMA traces, then a linear map to the bands, as multiply-adds). Spike counts from each day's test frames.
    Its latency is not measured (no chip run for audio)."""
    sys.path.insert(0, str(SNNDEC))
    from snndec import data
    from snndec.convert import load
    per_day = []
    for day in AUDIO_DAYS:
        model, info = load(SNN_RUNS / f"speck_audio_binB/{day}_r13_binB_s1337/model.pt")
        model.eval()
        rec, _, n = data.audio_recording(str(AUDITORY_DATA / f"Neil/aSTG/{day}/pipeline"))
        rec = rec[-n["test"]:]                                             # the recording as scored, test part
        if info.get("channels") is not None:
            rec = rec[:, list(info["channels"])]
        with torch.no_grad():
            model(torch.as_tensor(rec)[None])
        frames = len(rec)                                                  # input frames = outputs
        widths = [s.shape[-1] for s in model.layer_spikes]
        ev = float(model.input_events.sum()) / frames                     # input events per frame
        spk = [float(s.sum()) / frames for s in model.layer_spikes]
        accs = ev * widths[0] + sum(sp * w for sp, w in zip(spk[:-1], widths[1:]))
        R = model.spec["readout"]
        macs = R.get("stages", 1) * widths[-1] * model.substeps + widths[-1] * model.spec["n_outputs"]
        per_day.append((macs, accs))
    macs, accs = np.mean(per_day, 0)
    return dict(decoder="speck", macs=macs, accs=accs, latency_ms=np.nan)


def motor_rows():
    m = json.load(open(MOTOR / "combined_metrics.json"))
    eff = next(iter(json.load(open(MOTOR / "efficiency_summary.json")).values()))
    names = {"kf": "kf", "wf": "wf", "lstm": "lstm", "qrnn": "qrnn", "snn": "lsnn", "speck": "speck"}
    out = []
    for k, d in names.items():
        per = [v[k]["op_estimate"]["per_sample"] for v in m.values() if v.get(k) and v[k].get("op_estimate")]
        e = 1e6 * np.mean([p["energy_total_j_low"] for p in per]) if per else np.nan
        if k == "kf":
            # spike_bmi counted the full recursion (covariance update and a 96 x 96 inverse every 4 ms: ~1.0 M MACs);
            # the gain is within 4% of its steady state after 400 ms, so a deployed filter runs the steady-state
            # update, s^2 + 2 c s with s = 6 states (position, velocity, acceleration in x and y), c = 96 channels
            per = [{"effective_macs": KF_MOTOR_STATES ** 2 + 2 * 96 * KF_MOTOR_STATES, "effective_accs": 0.0}]
            e = energy_uj(per[0]["effective_macs"], 0)
        r = eff.get(k, {})
        out.append(dict(experiment="motor", decoder=d, energy_uj=e, interval_ms=4.0, power_mw=e / 4.0,
                        latency_ms=1e3 * r.get("latency_s", np.nan),
                        macs=np.mean([p["effective_macs"] for p in per]) if per else np.nan,
                        accs=np.mean([p["effective_accs"] for p in per]) if per else np.nan,
                        measured_uj=1e6 * r.get("energy_j", np.nan), measured_mw=1e3 * r.get("power_w", np.nan),
                        measured_by={"rapl": "laptop CPU (RAPL)", "chip_power_monitor": "Speck chip"}.get(r.get("energy_method"), "")))
    return out


def speech_rows():
    e = json.load(open(SPEECH))
    out = []
    for r in e["rows"]:
        d = SPEECH_ROWS.get(r["label"])
        if not d:
            continue
        # speech's own operation counts per output, with motor's energy rule (so all three share one accounting)
        mac, acc = r["ops"]["per_frame"]["mac"], r["ops"]["per_frame"]["acc"]
        uj = energy_uj(mac, acc)
        iv = 1e3 * r["latency"]["output_interval_s"]
        out.append(dict(experiment="speech", decoder=d, energy_uj=uj, interval_ms=iv, power_mw=uj / iv,
                        macs=mac, accs=acc, energy_uj_speech_model=r["ops"]["energy"]["sram_fp32"]["uj_per_frame"],
                        latency_ms=1e3 * r["timing"]["s_per_output"], lookahead_ms=1e3 * r["latency"]["lookahead_s"]))
    # the speech Speck SNN (round 8, F3): operation counts from its decoder-comparison run, per 20 ms input bin,
    # four bins per 80 ms output
    speck = SPEECH.parent / "decoderComparison_speck.json"
    if speck.is_file():
        S = json.load(open(speck)).get("Speck", {})
        ps = (S.get("op_estimate") or {}).get("per_sample")
        if ps:
            mac, acc = 4 * ps["effective_macs"], 4 * ps["effective_accs"]
            uj = energy_uj(mac, acc)
            out.append(dict(experiment="speech", decoder="speck", energy_uj=uj, interval_ms=80.0, power_mw=uj / 80.0,
                            macs=mac, accs=acc, latency_ms=np.nan, lookahead_ms=0.0))
    return out


LAPTOP = HERE / "data" / "laptop"                     # the Speck laptop's results (snn-decoders chip)


def laptop_overrides(df):
    """Latency of every decoder from one machine, the Speck laptop's CPU (latency.json: one thread, one output at a
    time, PyTorch/NumPy), where it was timed; the audio Speck SNN's chip latency, power and energy per output
    (audio_chip_l1x4.json: layer 1 threshold x4, mean over the five recording days)."""
    df["latency_machine"] = df.experiment.map({"motor": "earlier laptop run (Keras)", "audio": "this Mac",
                                               "speech": "Oscar CPU"})
    lat = LAPTOP / "latency.json"
    if lat.is_file():
        L = json.load(open(lat))
        for key, r in L["results"].items():
            exp, dec = key.split("/")
            sel = (df.experiment == exp) & (df.decoder == dec)
            df.loc[sel, "latency_ms"] = r["median_ms"]
            df.loc[sel, "latency_machine"] = "Speck laptop CPU"
    chip = LAPTOP / "audio_chip_l1x4.json"
    if chip.is_file():
        days = [r for k, r in json.load(open(chip)).items() if not k.startswith("ALLWORDS") and "r_chip" in r]
        sel = (df.experiment == "audio") & (df.decoder == "speck")
        df.loc[sel, "latency_ms"] = np.mean([r["latency_output_ms"] for r in days])
        df.loc[sel, "measured_uj"] = np.mean([r["energy_output_uj"] for r in days])
        df.loc[sel, "measured_mw"] = np.mean([r["power_mw"] for r in days])
        df.loc[sel, "measured_by"] = "Speck chip"
        df.loc[sel, "latency_machine"] = "Speck chip"
    df.loc[(df.experiment == "motor") & (df.decoder == "speck"), "latency_machine"] = "Speck chip"
    chip = LAPTOP / "speech_chip.json"                 # speechBCI speech/scripts/speech_chip.py --mode chip
    if chip.is_file():
        C = json.load(open(chip))["chip"]
        sel = (df.experiment == "speech") & (df.decoder == "speck")
        if not sel.any():                             # no comparison run yet: the chip's measurements only
            df = pd.concat([df, pd.DataFrame([dict(experiment="speech", decoder="speck", interval_ms=80.0)])],
                           ignore_index=True)
            sel = (df.experiment == "speech") & (df.decoder == "speck")
        df.loc[sel, "latency_ms"] = 4 * C["ms_per_bin"]                   # four 20 ms bins per 80 ms output
        df.loc[sel, "measured_uj"] = C["energy_uj_per_output_80ms"]
        df.loc[sel, "measured_mw"] = C["power_mw"]
        df.loc[sel, "measured_by"] = "Speck chip"
        df.loc[sel, "latency_machine"] = "Speck chip"
    return df


def main():
    df = laptop_overrides(pd.DataFrame(motor_rows() + audio_rows() + speech_rows()))
    (HERE / "data").mkdir(exist_ok=True)
    df.to_csv(HERE / "data" / "efficiency.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["experiment", "decoder", "energy_uj", "power_mw", "latency_ms", "latency_machine", "measured_uj", "measured_mw"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
