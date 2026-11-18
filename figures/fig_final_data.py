"""
Final figure: properties of the data that explain how the decoders came out.

    a  audio, per recording day: how many of the 96 channels shift their mean count by more than one training standard
       deviation between the training part of the day and its validation and test parts (the day's frames in order,
       80/10/10, as the pipeline splits them)
    b  audio, per day: every decoder's test score in the pipeline's own metric (mean over the 128 bands of the
       correlation between decoded and true spectrum over the test stream); the days whose test or validation part
       shifts are the days every decoder, or the decoders selected on validation, do worst
    c  motor, per monkey: how fast the hand moves (speed distribution over the held-out segments, as the share of time
       above each speed); Jenkins makes brief, fast reaches between long holds
    d  motor, per monkey: each decoder's RMSE divided by the standard deviation of the hand's velocity in that
       session (median over sessions), which puts the three monkeys on one scale
    e  how much each task's LSNN gains from memory: motor, adding recurrence (per-axis RMSE, 36 Indy sessions);
       audio, tripling the window of history from 8 to 24 frames (r, 3 seeds); speech, adding recurrence (phoneme
       error). The Speck SNN cannot have either form of memory on the chip

    python fig_final_data.py      -> figures/fig_final_data.{pdf,svg,png}
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).parent
DL = Path.home() / "Downloads"
AUD = Path.home() / "Documents/GitHub/AuditoryDecoding/pipeline_data/Neil/aSTG"
DAYS = ["07_19_11", "07_20_10", "07_31_13", "08_03_10", "08_06_13"]
AUD_DECS = [("kalman_filter", "kf"), ("wiener_filter", "wf"), ("gru_rnn", "gru"), ("lstm_rnn", "lstm"),
            ("lsnn", "lsnn"), ("speck_snn", "speck")]
SUBJ = {"indy": ("bmi/indy", "Indy"), "loco": ("bmi/loco", "Loco"), "jenkins_m1": ("hkm/jenkins_m1", "Jenkins M1")}
MOTOR_DECS = [("kf", "kf"), ("wf", "wf"), ("lstm", "lstm"), ("qrnn", "qrnn"), ("snn", "lsnn"), ("speck", "speck")]
LADDER = HERE / "data" / "motor_ladder.csv"


def stream_r(p, y):
    return np.nanmean([np.corrcoef(p[:, b], y[:, b])[0, 1] for b in range(y.shape[1]) if y[:, b].std() > 0])


def audio_shift():
    rows = {}
    for d in DAYS:
        pipe = AUD / d / "pipeline"
        X = np.load(pipe / "features.npy").astype(float)
        n = {s: np.load(pipe / f"window_targets_{s}.npy", mmap_mode="r").shape[0] for s in ("train", "valid", "test")}
        W = len(X) - sum(n.values()) + 1
        a, b = n["train"], n["train"] + n["valid"]
        tr, va, te = X[:a + W - 1], X[a:b + W - 1], X[b:]
        mu, sd = tr.mean(0), tr.std(0) + 1e-9
        ok = tr.std(0) > 0
        rows[d] = dict(valid=int((np.abs((va.mean(0) - mu) / sd)[ok] > 1).sum()),
                       test=int((np.abs((te.mean(0) - mu) / sd)[ok] > 1).sum()))
        targets = [np.load(pipe / f"{t}_targets_test.npy") for t in ("window", "raw", "flat_window")
                   if (pipe / f"{t}_targets_test.npy").is_file()]
        rows[d]["scores"] = {}
        for k, lab in AUD_DECS:
            if (pipe / f"{k}_predict_test.npy").is_file():
                p_ = np.load(pipe / f"{k}_predict_test.npy")
                y = next((t for t in targets if len(t) == len(p_)), None)   # each decoder's own split (KF: raw)
                if y is not None:
                    rows[d]["scores"][lab] = stream_r(p_, y)
    return rows


def motor_stats():
    out = {}
    for subj, (path, _) in SUBJ.items():
        cm = json.load(open(DL / f"final_v2/{subj}/combined_metrics.json"))
        speeds, nrmse = [], {lab: [] for _, lab in MOTOR_DECS}
        for sess, decs in cm.items():
            f = DL / f"speck_per_session/data/dataset/{path}/mua/{sess}_binning.h5"
            if not f.is_file():
                continue
            with h5py.File(f, "r") as h:
                v = h["y_task"][:, 2:4]
            speeds.append(np.linalg.norm(v, axis=1))
            sd = np.sqrt(v.var(0).sum())
            for k, lab in MOTOR_DECS:
                if decs.get(k):
                    nrmse[lab].append(decs[k]["rmse"] / sd)
        out[subj] = dict(speed=np.concatenate(speeds), nrmse={k: float(np.median(v)) for k, v in nrmse.items() if v},
                         n=len(speeds))
    return out


def main():
    A = audio_shift()
    M = motor_stats()
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 125 * style.MM))
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.4)
    axa, axb = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    axc, axd, axe = fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:])

    x = np.arange(len(DAYS))
    axa.bar(x - 0.18, [A[d]["valid"] for d in DAYS], 0.36, color=style.COLOR["lstm"], label="validation part")
    axa.bar(x + 0.18, [A[d]["test"] for d in DAYS], 0.36, color="#c0392b", label="test part")
    axa.set_xticks(x)
    axa.set_xticklabels([d.replace("_", "/") for d in DAYS], fontsize=6.5)
    axa.set_ylabel("channels shifted > 1 SD")
    axa.set_title("Audio: drift within each day", loc="left")
    axa.legend(loc="upper center", frameon=False, fontsize=6.5)

    for _, lab in AUD_DECS:
        ys = [A[d]["scores"].get(lab, np.nan) for d in DAYS]
        axb.plot(x, ys, "o-", color=style.COLOR[lab], ms=3, lw=1.1 if lab in style.HIGHLIGHT else 0.8,
                 label=style.SHORT[lab])
    axb.set_xticks(x)
    axb.set_xticklabels([d.replace("_", "/") for d in DAYS], fontsize=6.5)
    axb.set_ylabel("test r (pipeline metric)")
    axb.set_title("Audio: every decoder, per day", loc="left")
    axb.set_ylim(0.05, 0.98)
    axb.legend(loc="upper left", ncol=6, frameon=False, fontsize=6, columnspacing=0.6, handlelength=1.2)

    grid = np.logspace(0, 3.3, 200)
    for subj, (_, lab) in SUBJ.items():
        s = M[subj]["speed"]
        axc.plot(grid, [(s > g).mean() for g in grid], lw=1.2, label=f"{lab} ({M[subj]['n']})",
                 color={"indy": style.COLOR["speck"], "loco": style.COLOR["wf"], "jenkins_m1": style.COLOR["lstm"]}[subj])
    axc.set_xscale("log"); axc.set_yscale("log")
    axc.set_ylim(1e-3, 1.05)
    axc.set_xlabel("hand speed (mm/s)")
    axc.set_ylabel("share of time faster")
    axc.set_title("How each monkey moves", loc="left")
    axc.legend(loc="lower left", frameon=False, fontsize=6)

    subs = list(SUBJ)
    w = 0.8 / len(MOTOR_DECS)
    for i, (_, lab) in enumerate(MOTOR_DECS):
        axd.bar(np.arange(len(subs)) + (i - (len(MOTOR_DECS) - 1) / 2) * w,
                [M[s]["nrmse"].get(lab, np.nan) for s in subs], w, color=style.COLOR[lab], label=style.SHORT[lab])
    axd.axhline(1, color=style.MUTED, lw=0.6, ls=":")
    axd.set_ylim(0, 1.32)
    axd.set_xticks(range(len(subs)))
    axd.set_xticklabels([SUBJ[s][1] for s in subs], fontsize=6.5)
    axd.set_ylabel("RMSE / velocity SD")
    axd.set_title("Error on one scale", loc="left")
    axd.legend(loc="upper left", ncol=3, frameon=False, fontsize=5.5, columnspacing=0.6, handlelength=1)

    import pandas as pd
    lad = pd.read_csv(LADDER).mean(numeric_only=True)
    aud = json.load(open(Path.home() / "Documents/GitHub/snn-decoders/runs/audio_final_scores.json"))
    r8 = np.mean([v["r_test"] for k, v in aud.items() if k.startswith("L0_base")])
    r24 = np.mean([v["r_test"] for k, v in aud.items() if k.startswith("L2_past24")])
    gains = [("motor\n(recurrence)", 100 * (1 - lad["recurrence"] / lad["lsnn_neurons"])),
             ("audio\n(3x history)", 100 * (r24 / r8 - 1)),
             ("speech\n(recurrence)", 100 * (1 - 0.34 / 0.72))]
    axe.bar(range(3), [g[1] for g in gains], color=[style.COLOR["speck"], style.COLOR["lstm"], style.COLOR["lsnn"]])
    for i, g in enumerate(gains):
        axe.text(i, g[1] + 1.2, f"{g[1]:.0f}%", ha="center", fontsize=6.5)
    axe.set_xticks(range(3))
    axe.set_xticklabels([g[0] for g in gains], fontsize=6)
    axe.set_ylim(0, 62)
    axe.set_ylabel("LSNN improvement from memory (%)")
    axe.set_title("How much memory pays", loc="left")

    for ax, k in zip([axa, axb, axc, axd, axe], "abcde"):
        style.panel_label(ax, k, x=-0.3 if k in "cde" else -0.15)
    style.save(fig, HERE / "figures" / "fig_final_data")
    print("audio shifted channels (valid/test):", {d: (A[d]["valid"], A[d]["test"]) for d in DAYS})
    print("audio test r per day:", {d: {k: round(v, 3) for k, v in A[d]["scores"].items()} for d in DAYS})
    print("motor nRMSE:", {s: {k: round(v, 2) for k, v in M[s]["nrmse"].items()} for s in subs})
    print("motor speed p50/p99 and share moving > 50 mm/s:",
          {s: (round(float(np.median(M[s]["speed"])), 1), round(float(np.percentile(M[s]["speed"], 99)), 0),
               round(float((M[s]["speed"] > 50).mean()), 3)) for s in subs})
    print("memory gains %:", [(g[0].split()[0], round(g[1], 1)) for g in gains])


if __name__ == "__main__":
    main()
