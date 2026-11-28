"""
Video: hand reaching decoded from motor cortex, in real time, by four tracks: the true hand, the source paper's best
decoder (QRNN, Ahmadi et al. 2021), the LSNN and Speck (the Speck SNN as it ran on the chip; Jenkins with its readout
refitted in software). The networks are the paper's final ones: validation-selected checkpoints, and for monkeys I and L
the soft-reset Speck SNN.

    left    the 2D workspace: each track's position, integrated from its velocity (the true one from the true
            velocity), every decoder re-anchored to the true hand at the start of each reach so that drift does not
            hide the reach; a crosshair marks the current position, a fading trail the last 1.5 s of the path
    right   x and y velocity, the last 3 s scrolling, the same four colours; each decoder's correlation with the true
            velocity so far in the legend

Data: final_v3/<subject>/sessions/<session>_arrays.npz (the validation-selected laptop run's test segment, 4 ms steps:
y_true = velocity in mm/s, y_pos = hand position in mm, pred_qrnn, pred_snn = the LSNN); the Speck lane:
  indy, loco   results_motor_soft/motor/final_v3_soft/<subject>/sessions/<session>_arrays.npz pred_speck (the soft-reset
               networks on the devkit; that run's segment starts 14 steps earlier, so it is aligned on its end and its
               y_true checked against final_v3's)
  jenkins_m1   --speck-pred <session>_arrays.npz from the soft-reset devkit run (motor_chip.py; refitted readout, as
               deployed), aligned trial by trial; without it, final_results/jenkins_refit/<session>_pred.npy, the
               EARLIER test-selected network, and the video says so.

    python3 video_motor.py --subject indy --session indy_20161013_03 --seconds 45 --output figures/video_motor1.mp4
    python3 video_motor.py --subject jenkins_m1 --session sub-Jenkins_ses-20090916_behavior+ecephys \
        --output figures/video_motor2.mp4
"""
from repo_paths import WORKSPACE  # noqa: E402  (the repository's paths.yaml)
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import style

HERE = Path(__file__).parent
DL = WORKSPACE
DT = 0.004
FPS = 30
TRAIL_S, WIN_S = 1.5, 3.0
TRACKS = [("truth", "Ground truth", "#1a1a1a"), ("qrnn", "QRNN (paper's best)", style.COLOR["qrnn"]),
          ("lsnn", "LSNN", style.COLOR["lsnn"]), ("speck", "Speck", style.COLOR["speck"])]


def reach_starts(v, min_gap=0.4, max_gap=2.0):
    """The start of each reach: the dips in the hand's (smoothed, 100 ms) speed between movements, at least
    min_gap s apart; at least one every max_gap s, so a decoder's drift never builds up for long."""
    sp = np.convolve(np.linalg.norm(v, axis=1), np.ones(25) / 25, mode="same")
    peak = np.percentile(sp, 90)
    cand = [t for t in range(1, len(sp) - 1) if sp[t] <= sp[t - 1] and sp[t] < sp[t + 1] and sp[t] < 0.35 * peak]
    starts, last = [0], 0
    for t in cand:
        if (t - last) * DT >= min_gap:
            starts.append(t)
            last = t
    out = [0]
    for t in starts[1:] + [len(v)]:
        while (t - out[-1]) * DT > max_gap:
            out.append(out[-1] + int(max_gap / DT))
        if t < len(v):
            out.append(t)
    return np.array(sorted(set(out)))


def integrate(v, p_true, anchors):
    p = np.zeros_like(p_true)
    edges = list(anchors) + [len(v)]
    for a, b in zip(edges[:-1], edges[1:]):
        p[a:b] = p_true[a] + np.vstack([[0, 0], np.cumsum(v[a:b - 1] * DT, 0)])
    return p


def align_trials(c, z, key):
    """The chip run's predictions on the final arrays' steps: each test trial's run starts some steps earlier (the
    final pipeline drops a trial's first steps), found by matching the true velocity trial by trial."""
    lens = c["trial_lens"]
    cs = np.r_[0, np.cumsum(lens)]
    ts = list(z["trial_starts"]) + [len(z["y_true"])]
    assert len(ts) - 1 == len(lens), (len(ts) - 1, len(lens))
    out = []
    for i in range(len(lens)):
        seg = z["y_true"][ts[i]:ts[i + 1]]
        n = len(seg)
        o = next((o for o in range(lens[i] - n + 1)
                  if np.allclose(c["y_true"][cs[i] + o:cs[i] + o + n], seg, atol=1e-2)), None)
        assert o is not None, f"trial {i}: no match"
        out.append(c[key][cs[i] + o:cs[i] + o + n])
    return np.concatenate(out).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="indy")
    ap.add_argument("--session", default="indy_20161013_03")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--start", type=float, default=None, help="s into the test segment (default: the most active)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--speck-pred", default=None, help="the Speck lane's predictions (npy, the test segment's steps)")
    a = ap.parse_args()
    z = np.load(DL / f"final_v3/{a.subject}/sessions/{a.session}_arrays.npz")
    v = {"truth": z["y_true"].astype(float), "qrnn": z["pred_qrnn"].astype(float), "lsnn": z["pred_snn"].astype(float)}
    n_all, provisional = len(v["truth"]), False
    if a.speck_pred and a.speck_pred.endswith(".npz"):        # motor_chip.py's <session>_arrays.npz: every test step
        v["speck"] = align_trials(np.load(a.speck_pred), z, "pred_refit" if a.subject == "jenkins_m1" else "pred_trained")
    elif a.speck_pred:
        v["speck"] = np.load(a.speck_pred).astype(float)
    elif a.subject in ("indy", "loco"):
        s = np.load(DL / f"results_motor_soft/motor/final_v3_soft/{a.subject}/sessions/{a.session}_arrays.npz")
        assert np.allclose(s["y_true"][-n_all:], z["y_true"]), "the soft-reset run's test segment does not match"
        v["speck"] = s["pred_speck"][-n_all:].astype(float)
    else:
        v["speck"] = np.load(DL / f"final_results/jenkins_refit/{a.session}_pred.npy").astype(float)
        provisional = True
        print("WARNING: Speck lane = the earlier test-selected Jenkins network (no predictions from the final run yet)")
    assert len(v["speck"]) == n_all, (len(v["speck"]), n_all)
    pos_true = z["y_pos"].astype(float)
    n = int(a.seconds / DT)
    if a.start is None:                              # the most active stretch
        sp = np.linalg.norm(v["truth"], axis=1)
        cs = np.concatenate([[0], np.cumsum(sp)])
        s0 = int(np.argmax(cs[n:] - cs[:-n])) if len(sp) > n else 0
    else:
        s0 = int(a.start / DT)
    sl = slice(s0, s0 + n)
    v = {k: x[sl] for k, x in v.items()}
    pos_true = pos_true[sl]
    # where trials are stitched together (hkm) the true hand jumps back to the centre: a clean restart there
    jumps = np.flatnonzero(np.linalg.norm(np.diff(pos_true, axis=0), axis=1) > 5.0) + 1
    anchors = np.array(sorted(set(reach_starts(v["truth"])) | set(jumps)))
    pos = {k: (pos_true if k == "truth" else integrate(x, pos_true, anchors)) for k, x in v.items()}
    anchor_set = np.array(anchors)

    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.04, 0.08, 0.5, 0.80])
    axv = [fig.add_axes([0.61, 0.49, 0.37, 0.36]), fig.add_axes([0.61, 0.08, 0.37, 0.36])]
    lo, hi = pos_true.min(0) - 25, pos_true.max(0) + 25              # the true hand's workspace
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#d0d3da")
    ax.set_title("Hand position on the screen (mm), integrated from each track's velocity", fontsize=14, loc="left")
    ax.text(0.0, -0.045, "decoders re-anchored to the true hand at the start of each reach (at least every 2 s); trails: last 1.5 s",
            transform=ax.transAxes, fontsize=11, color=style.MUTED)
    vmax = np.percentile(np.abs(np.concatenate([x for x in v.values()])), 99.5) * 1.1
    for k, axx in enumerate(axv):
        axx.set_ylim(-vmax, vmax)
        axx.set_xlim(-WIN_S, 0)
        axx.set_ylabel(f"{'xy'[k]} velocity (mm/s)", fontsize=12)
        axx.tick_params(labelsize=10)
        axx.axvline(0, color=style.MUTED, lw=0.8)
    axv[1].set_xlabel("time (s), now at 0", fontsize=12)
    title = {"indy": "Motor (I): monkey I, NeuroBench reaching", "jenkins_m1": "Motor (II): monkey J, M1 array"}
    fig.text(0.04, 0.95, title.get(a.subject, a.subject), fontsize=20, fontweight="bold")
    note = ("Speck: an earlier, test-selected network (provisional)" if provisional else
            "Speck: soft reset, on the chip")
    fig.text(0.04, 0.915, f"{a.session}, held-out test segment, decoded in real time (4 ms steps). {note}",
             fontsize=12, color=style.MUTED)

    trails, marks, vlines = {}, {}, {}
    for k, lab, col in TRACKS:
        lw = 2.6 if k == "truth" else 2.0
        trails[k] = LineCollection([], linewidths=3.2 if k == "truth" else 2.4, zorder=3 if k == "truth" else 4,
                                   capstyle="round")
        ax.add_collection(trails[k])
        marks[k] = ax.plot([], [], marker="+", ms=26 if k == "truth" else 22, mew=3.0 if k == "truth" else 2.4,
                           color=col, zorder=6, ls="none")[0]
        vlines[k] = [axx.plot([], [], color=col, lw=lw if k == "truth" else 1.5, zorder=5 if k == "truth" else 4)[0]
                     for axx in axv]
    xs = [0.61, 0.69, 0.832, 0.915]
    leg = [fig.text(xs[i], 0.905, "", fontsize=13, color=col, fontweight="bold") for i, (_, _, col) in
           enumerate(TRACKS)]
    out = Path(a.output or HERE / "figures" / f"video_motor_{a.subject}.mp4")
    W, H = fig.canvas.get_width_height()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf",
                           "20", str(out)], stdin=subprocess.PIPE)
    nt, ntr, nw = n, int(TRAIL_S / DT), int(WIN_S / DT)
    tt = np.arange(nw) * DT - WIN_S
    for f in range(int(a.seconds * FPS)):
        t = min(int(f / FPS / DT), nt - 1)
        a0 = max(0, t - ntr)
        past = jumps[jumps <= t]
        if len(past):                                    # trails start afresh after a trial boundary
            a0 = max(a0, int(past[-1]))
        for i, (k, lab, col) in enumerate(TRACKS):
            p = pos[k][a0:t + 1]
            if len(p) > 1:
                segs = np.stack([p[:-1], p[1:]], 1)
                rgba = np.tile(matplotlib.colors.to_rgba(col), (len(segs), 1))
                rgba[:, 3] = np.linspace(0.0, 0.9, len(segs)) ** 1.6
                if k != "truth":                           # no line across a re-anchoring jump
                    rgba[np.isin(np.arange(a0 + 1, t + 1), anchor_set), 3] = 0.0
                trails[k].set_segments(segs)
                trails[k].set_color(rgba)
            marks[k].set_data([pos[k][t, 0]], [pos[k][t, 1]])
            w0 = max(0, t - nw + 1)
            seg = v[k][w0:t + 1]
            for d in range(2):
                vlines[k][d].set_data(tt[nw - len(seg):], seg[:, d])
            if k == "truth":
                leg[i].set_text(lab)
            else:
                m = slice(0, t + 1)
                rr = np.nanmean([np.corrcoef(v[k][m, d], v["truth"][m, d])[0, 1] for d in range(2)]) if t > 50 else np.nan
                leg[i].set_text(f"{lab}  r={rr:.2f}" if np.isfinite(rr) else lab)
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
