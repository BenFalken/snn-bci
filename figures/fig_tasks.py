"""
Figure: the four experiments as the subject experiences them, where the recording array sits, and what is recorded
and decoded (schematic pictograms, drawn here, after the tasks of the source studies; the bottom row is real training
data).

Columns: Motor (I), monkeys I and L reaching to random targets on a grid (O'Doherty et al.); Motor (II), monkey J
reaching from the centre to a cued target (Churchland et al.); Audio, monkeys N and G hearing words (Heelan et al.);
Speech, participant T12 attempting to speak a cued sentence (Willett et al.).
    top     the task          middle  the array (96 channels, or 128 electrodes)          bottom  recorded -> decoded

    python3 fig_tasks.py      -> figures/fig_tasks.{pdf,svg,png}
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

import pictograms as pg
import style

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import AUDITORY_DATA, MOTOR, MOTOR_DATA, WORKSPACE  # noqa: E402
from snndec import data  # noqa: E402

HERE = Path(__file__).parent
INK, MUTED, LIGHT = style.INK, style.MUTED, "#d9dbe0"
COL = {"motor1": "#332288", "motor2": "#5b4fa8", "audio": "#117733", "speech": "#AA4499"}
SKIN, FUR = "#f1d3bd", "#b98d68"


# ------------------------------------------------------------------------------------------------------- pictograms
def blob(ax, pts, **kw):
    """A smooth closed shape through pts (Catmull-Rom through cubic Beziers)."""
    p = np.asarray(pts, float)
    n = len(p)
    verts, codes = [p[0]], [MPath.MOVETO]
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        verts += [p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2]
        codes += [MPath.CURVE4] * 3
    ax.add_patch(PathPatch(MPath(verts, codes), **kw))


def monkey(ax, x, y, s=1.0, arm_to=None):
    pg.monkey(ax, x, y, s, reach_to=arm_to)


def person(ax, x, y, s=1.0):
    pg.person(ax, x, y, s)


def screen(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc="#20242c", ec=INK,
                                lw=0.6, zorder=1))


def brain(ax, x, y, s, human, site, color, label):
    """A lateral view of the left hemisphere (front to the right) with the array's site marked."""
    mx, my = pg.brain(ax, x, y, s, human=human)(site)
    ax.add_patch(Rectangle((mx - .07 * s, my - .07 * s), .14 * s, .14 * s, fc=color, ec="white", lw=0.6, zorder=8))
    ax.text(x, y - (0.98 if human else 0.88) * s, label, ha="center", va="top", fontsize=6.2, color=INK)


def speaker(ax, x, y, s):
    ax.add_patch(Polygon([(x, y - .1 * s), (x + .12 * s, y - .1 * s), (x + .26 * s, y - .25 * s),
                          (x + .26 * s, y + .25 * s), (x + .12 * s, y + .1 * s), (x, y + .1 * s)],
                         fc="#5c6370", ec=INK, lw=0.6, zorder=3))
    for r in (.2, .32, .44):
        ax.add_patch(Arc((x + .26 * s, y), r * 2 * s, r * 2 * s, theta1=140, theta2=220, color=COL["audio"], lw=0.9,
                         zorder=3))


# ----------------------------------------------------------------------------------------------------- the panels
def task_motor1(ax):
    screen(ax, 1.05, -0.35, 1.25, 1.25)
    for i in range(8):
        for j in range(8):
            ax.add_patch(Circle((1.13 + i * .155, -0.27 + j * .155), .018, fc="#4a505c", ec="none", zorder=2))
    path = np.array([(1.3, 0.05), (1.5, 0.4), (1.85, 0.55), (2.05, 0.2)])
    ax.plot(path[:, 0], path[:, 1], color="#9fb4ff", lw=0.8, ls=":", zorder=3)
    ax.add_patch(Circle((2.05, 0.2), .06, fc="#ffd24d", ec="none", zorder=4))                      # target
    monkey(ax, 0.35, 0.05, 0.95, arm_to=(1.25, 0.08))
    ax.text(1.1, -1.0, "reach to a random target", ha="center", va="top", fontsize=6.2, color=INK)


def task_motor2(ax):
    screen(ax, 1.05, -0.35, 1.25, 1.25)
    cx, cy = 1.68, 0.28
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.add_patch(Circle((cx + .42 * np.cos(a), cy + .42 * np.sin(a)), .04, fc="#4a505c", ec="none", zorder=2))
    ax.add_patch(Circle((cx + .42, cy), .06, fc="#ffd24d", ec="none", zorder=4))
    ax.add_patch(Circle((cx, cy), .045, fc="#9fb4ff", ec="none", zorder=4))
    ax.add_patch(FancyArrowPatch((cx + .06, cy), (cx + .34, cy), arrowstyle="-|>", mutation_scale=6, color="#9fb4ff",
                                 lw=0.8, zorder=4))
    monkey(ax, 0.35, 0.05, 0.95, arm_to=(1.25, 0.1))
    ax.text(1.1, -1.0, "reach from the center to a cued target", ha="center", va="top", fontsize=6.2, color=INK)


def task_audio(ax):
    speaker(ax, 1.55, 0.3, 1.2)
    ax.add_patch(FancyBboxPatch((1.75, 0.62), 0.55, 0.26, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white",
                                ec=COL["audio"], lw=0.8, zorder=3))
    ax.text(2.02, 0.75, "“tree”", ha="center", va="center", fontsize=7, color=COL["audio"], zorder=4)
    monkey(ax, 0.55, 0.05, 0.95)
    ax.text(1.1, -1.0, "hear a spoken word", ha="center", va="top", fontsize=6.2, color=INK)


def task_speech(ax, sentence):
    screen(ax, 0.98, 0.05, 1.36, 0.6)
    words = sentence.split()
    ax.text(1.66, 0.35, " ".join(words[:3]) + "\n" + " ".join(words[3:]), ha="center", va="center", fontsize=5.4,
            color="white", zorder=3, linespacing=1.3)
    person(ax, 0.45, 0.0, 0.95)
    ax.add_patch(FancyBboxPatch((0.9, -0.32), 0.62, 0.22, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white",
                                ec=COL["speech"], lw=0.8, zorder=3))
    ax.text(1.21, -0.21, "attempts", ha="center", va="center", fontsize=5.6, color=COL["speech"], style="italic",
            zorder=4)
    ax.text(1.1, -1.0, "attempt to speak a cued sentence", ha="center", va="top", fontsize=6.2, color=INK)


def raster(ax, X, color, dt_s):
    t, c = np.nonzero(X > 0)
    ax.scatter(t * dt_s, c, s=0.25, color=color, marker="|", linewidths=0.4, rasterized=True)
    ax.set_xlim(0, len(X) * dt_s)
    ax.set_ylim(-1, X.shape[1])
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def recorded(axes, kind):
    """(raster axis, output axis): a short stretch of real training data and its decoded target."""
    ar, ao = axes
    S = str(MOTOR / "snn_training/velocity_scalers.json")
    if kind in ("motor1", "motor2"):
        n = 500                                                                       # 2 s at 4 ms
        if kind == "motor1":                    # the full recording (its trials here are short): training portion
            sys.path.insert(0, str(MOTOR / "snn_training"))
            from channel_lag_tuning import load_session
            rates, vv = load_session(str(WORKSPACE / "spike_bmi_lsnn_data/processed/bmi/indy/mua/indy_20160921_01.h5"),
                                     4.0, 0.0, 0.88)
            X, V = rates.T, vv
        else:
            sd = MOTOR_DATA / "snn_datasets/hkm/jenkins_m1/mua/sub-Jenkins_ses-20090916_behavior+ecephys"
            t = max(data.motor_split(str(sd), "train", S), key=lambda q: len(q.x))
            X, V = t.x, data.unscale_velocity(t.y, *data.velocity_scaler(str(sd), S))
        sp = np.convolve(np.linalg.norm(V, axis=1), np.ones(n) / n, "same")
        c = int(np.argmax(sp[n // 2: len(sp) - n // 2])) + n // 2                     # the busiest 2 s
        a = max(0, min(len(X) - n, c - n // 2))
        X, v = X[a:a + n], V[a:a + n]
        raster(ar, X, COL[kind], 0.004)
        tt = np.arange(n) * 0.004
        ao.plot(tt, v[:, 0], color=INK, lw=0.9, label="x")
        ao.plot(tt, v[:, 1], color=MUTED, lw=0.9, label="y")
        ao.set_xlim(0, n * 0.004)
        ao.set_ylabel("velocity", fontsize=5.6, labelpad=1)
        ao.legend(fontsize=5, frameon=False, loc="upper right", handlelength=1, borderaxespad=0)
        return f"96 channels, every 4 ms", "hand velocity"
    if kind == "audio":
        P = AUDITORY_DATA / "Neil/aSTG/07_31_13/pipeline"
        F, Y = np.load(P / "features.npy"), np.load(P / "targets.npy")
        e = Y.mean(1)
        # a 2 s stretch (training part) where the sound starts from near-silence: quiet 0.6 s, then loud 0.8 s
        m = int(len(e) * .8) - 50
        on = np.array([e[i + 18:i + 38].mean() - e[i:i + 15].mean() for i in range(m)])
        a = int(np.argmax(on)); b = a + 50
        ar.imshow(F[a:b].T, aspect="auto", origin="lower", cmap="Greens", interpolation="nearest",
                  extent=[0, (b - a) * 0.04, 0, F.shape[1]], vmax=np.percentile(F[a:b], 99))
        ar.set_yticks([]); ar.set_xticks([])
        for sp_ in ("top", "right", "left"):
            ar.spines[sp_].set_visible(False)
        ao.imshow(Y[a:b].T, aspect="auto", origin="lower", cmap="magma", extent=[0, (b - a) * 0.04, 0, 128])
        ao.set_ylabel("mel band", fontsize=5.6, labelpad=1)
        return "96 channels, every 40 ms", "the sound's spectrum"
    z = np.load(WORKSPACE / "speech_rebuild/derived/snnData/t12.2022.06.14/train.npz")
    n = int(z["featureLens"][0])
    X = z["features"][:n, :128]                                                         # threshold crossings
    raster(ar, (X > 1.0).astype(float), COL[kind], 0.02)
    lab = z["labels"][: int(z["labelLens"][0])]
    PH = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
          'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '|']
    words, cur = [], []
    for i in lab:
        if int(i) == 39:                                                                  # word boundary
            words.append(cur); cur = []
        else:
            cur.append(PH[int(i)])
    words.append(cur)
    words = [w for w in words if w]
    third = (len(words) + 2) // 3
    line = lambda ws: "  ·  ".join(" ".join(w) for w in ws)
    ao.set_xlim(0, n * 0.02)
    ao.set_ylim(0, 1)
    ao.set_yticks([])
    ao.text(0.01, 0.5, "\n".join(line(words[i:i + third]) for i in range(0, len(words), third)),
            transform=ao.transAxes, fontsize=4.8, va="center", ha="left", color=INK, linespacing=1.35,
            family="DejaVu Sans Mono")
    ao.set_ylabel("phonemes", fontsize=5.6, labelpad=1)
    ao.set_xticks([])
    ao.set_xlabel("")
    ao.spines["bottom"].set_visible(False)
    return "128 electrodes (crossings), every 20 ms", "the attempted phonemes (no timing)"


def main():
    style.apply()
    z = np.load(WORKSPACE / "speech_rebuild/derived/snnData/t12.2022.06.14/train.npz")
    sentence = "".join(chr(c) for c in z["transcriptions"][0] if c > 0)
    cols = [("motor1", "Motor (I)", "monkeys I, L", task_motor1, False, "M1", "M1 array, 96 channels"),
            ("motor2", "Motor (II)", "monkey J", task_motor2, False, "M1", "M1 array, 96 channels"),
            ("audio", "Audio", "monkeys N, G", task_audio, False, "aSTG", "aSTG array, 96 channels"),
            ("speech", "Speech", "participant T12", lambda ax: task_speech(ax, sentence), True, "6v",
             "area 6v, 128 electrodes")]
    W, H = 183.0, 112.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    A = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    cw = W / 4
    for i, (kind, name, who, draw, human, mark, alabel) in enumerate(cols):
        x0 = i * cw
        fig.text((x0 + cw / 2) / W, 1 - 2 / H, name, ha="center", va="top", fontsize=8.5, fontweight="bold",
                 color=COL[kind])
        fig.text((x0 + cw / 2) / W, 1 - 7 / H, who, ha="center", va="top", fontsize=6.2, color=MUTED)
        top = A(x0 + 2, 11, cw - 4, 34)
        top.set_xlim(-0.45, 2.45)
        top.set_ylim(-1.2, 1.05)
        draw(top)
        mid = A(x0 + 9, 47, cw - 18, 24)
        mid.set_xlim(-1.15, 1.15)
        mid.set_ylim(-1.0, 0.95)
        brain(mid, 0, 0.08, 0.80, human, mark, COL[kind], alabel)
        for a in (top, mid):
            a.set_aspect("equal")
            a.axis("off")
        ar, ao = A(x0 + 7, 75, cw - 10, 14), A(x0 + 7, 92, cw - 10, 11)
        rec, out = recorded((ar, ao), kind)
        ar.set_title(f"recorded: {rec}", fontsize=5.8, loc="left", pad=1.5)
        ao.set_title(f"decoded: {out}", fontsize=5.8, loc="left", pad=1.5)
        ao.tick_params(labelsize=5)
        if kind != "speech":
            ao.set_xlabel("s", fontsize=5.4, labelpad=0)
        for sp in ("top", "right"):
            ao.spines[sp].set_visible(False)
    style.save(fig, HERE / "figures" / "fig_tasks")
    print("wrote figures/fig_tasks")


if __name__ == "__main__":
    main()
