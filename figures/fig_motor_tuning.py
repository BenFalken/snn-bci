"""
Motor figure: M1 channels are broadly tuned to the direction of hand movement, and velocity is spread across the
population, so a decoder gains from every channel it reads (the reason the Speck SNN spends its input budget on
channels, and decodes with a population of output neurons).

Data: the Indy sessions' held-out segments (motor_data; about 80 s each, 4 ms bins), rates smoothed by 40 ms and
paired with the velocity 160 ms later (the median lead, fig_motor_lag). Per channel, rate = b0 + bx vx + by vy is
fitted by least squares; the preferred direction is atan2(by, bx), the tuning strength its R^2 (tuned: R^2 >= 0.02).
Channels with no activity in a segment are left out.

    a  four strongly tuned channels of one session: mean rate (z-scored, centred) in 30-degree bins of movement
       direction, while the hand moves faster than its median speed, with a cosine fitted to the bins
    b  that session's tuned channels' preferred directions: together they cover every direction
    c  every channel of all 36 sessions: the share of velocity's variance it explains on its own at its best lead
       (fig_motor_lag's data: the first 88% of each session, 20 ms bins); most explain little, so a decoder must pool
       them. How much pooling more channels gains under the chip's input budget is in fig_motor_budget.

    python fig_motor_tuning.py      -> figures/fig_motor_tuning.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import motor_data as md
import style

HERE = Path(__file__).parent
EXAMPLE = None                       # chosen in pick_example(): typical spread of preferred directions
LEAD_MS = 160.0
MIN_R2 = 0.02
SIZES = [1, 2, 4, 8, 16, 32, 64, 96]


def active(X):
    return X.std(0) > 1e-9


def fit(Z, V):
    """Per channel: (bx, by) and R^2 of rate ~ b0 + bx vx + by vy (Z: z-scored rates of active channels)."""
    A = np.column_stack([np.ones(len(V)), V])
    B, *_ = np.linalg.lstsq(A, Z, rcond=None)
    r2 = 1 - (Z - A @ B).var(0) / Z.var(0)
    return B[1:].T, r2


def ridge_corr(X, V, cols, lam=10.0):
    n = len(V) // 2
    mu, sd = X[:n, cols].mean(0), X[:n, cols].std(0) + 1e-9
    A = np.column_stack([np.ones(len(V)), (X[:, cols] - mu) / sd])
    W = np.linalg.solve(A[:n].T @ A[:n] + lam * np.eye(A.shape[1]), A[:n].T @ V[:n])
    P = A[n:] @ W
    return np.mean([np.corrcoef(P[:, i], V[n:, i])[0, 1] for i in range(2)])


def pd_spread(session):
    """(tuned channels, resultant length of their preferred directions: 0 = uniform, 1 = all alike)."""
    X, V = md.lead(*md.load(session), LEAD_MS)
    X = X[:, active(X)]
    b, r2 = fit((X - X.mean(0)) / X.std(0), V)
    pd = np.arctan2(b[:, 1], b[:, 0])[r2 >= MIN_R2]
    return len(pd), (abs(np.exp(1j * pd).mean()) if len(pd) else np.nan)


def pick_example():
    """A session with many tuned channels (at least 40) whose directions are spread as in the median session."""
    rows = [(s, *pd_spread(s)) for s in md.sessions()]
    med = np.nanmedian([r for _, n, r in rows if n >= 5])
    cand = [(abs(r - med), s) for s, n, r in rows if n >= 40]
    return min(cand)[1], med


def main():
    global EXAMPLE
    EXAMPLE, med_spread = pick_example()
    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 62 * style.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1.2], wspace=0.5)

    X, V = md.lead(*md.load(EXAMPLE), LEAD_MS)
    X = X[:, active(X)]
    Z = (X - X.mean(0)) / X.std(0)
    b, r2 = fit(Z, V)
    pd = np.arctan2(b[:, 1], b[:, 0])
    tuned = np.flatnonzero(r2 >= MIN_R2)

    # a: centred direction tuning of four strong channels with different preferred directions
    ax = fig.add_subplot(gs[0])
    speed = np.linalg.norm(V, axis=1)
    moving = speed > np.median(speed)
    ang = np.arctan2(V[moving, 1], V[moving, 0])
    edges = np.linspace(-np.pi, np.pi, 13)
    centres = (edges[:-1] + edges[1:]) / 2
    strong = tuned[np.argsort(-r2[tuned])][:20]
    picks = []
    for target in np.linspace(-np.pi, np.pi, 4, endpoint=False):
        order = strong[np.argsort(np.abs(np.angle(np.exp(1j * (pd[strong] - target)))))]
        picks.append(next(c for c in order if c not in picks))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(picks)))
    th = np.linspace(-np.pi, np.pi, 200)
    for c, col in zip(picks, cols):
        m = np.array([Z[moving, c][(ang >= lo) & (ang < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
        m -= m.mean()
        A = np.column_stack([np.cos(centres), np.sin(centres)])
        (p, q), *_ = np.linalg.lstsq(A, m, rcond=None)
        ax.plot(np.degrees(centres), m, "o", color=col, ms=3)
        ax.plot(np.degrees(th), p * np.cos(th) + q * np.sin(th), color=col, lw=1)
    ax.axhline(0, color=style.MUTED, lw=0.5)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("movement direction (°)")
    ax.set_ylabel("rate, centred (z)")
    ax.set_title("Direction tuning, four channels", loc="left")

    # b: preferred directions
    ax = fig.add_subplot(gs[1], projection="polar")
    hist, e = np.histogram(pd[tuned], bins=12, range=(-np.pi, np.pi))
    ax.bar((e[:-1] + e[1:]) / 2, hist, width=2 * np.pi / 12, color=style.MUTED, alpha=0.8, edgecolor="white", lw=0.5)
    for c, col in zip(picks, cols):
        ax.plot([pd[c], pd[c]], [0, hist.max() * 1.1], color=col, lw=1.2)
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6.5)
    spread = abs(np.exp(1j * pd[tuned]).mean())
    ax.set_title(f"Preferred directions,\n{len(tuned)} tuned channels", loc="left", pad=14, fontsize=7.5)
    ax.text(0.5, -0.2, f"clustering {spread:.2f} (median session {med_spread:.2f};\n0 = uniform, 1 = all alike)",
            transform=ax.transAxes, ha="center", va="top", fontsize=6)

    # c: how much of velocity one channel explains, all sessions (lag-tuning data, first 88% of each session)
    ax = fig.add_subplot(gs[2])
    import glob, json
    peaks = []
    for f in sorted(glob.glob(str(Path.home() / "Downloads/spike_bmi_lsnn_data/lag_tuning_all/lag_tuning_indy_*.json"))):
        peaks += [c["peak_r2"] for c in json.load(open(f))["channels"]]
    peaks = np.array(peaks)
    ax.hist(peaks, bins=np.linspace(0, 0.6, 31), color=style.MUTED, alpha=0.85)
    med = np.median(peaks)
    ax.axvline(med, color=style.INK, lw=0.9, ls="--")
    ax.text(med + 0.01, ax.get_ylim()[1] * 0.95, f"median {med:.2f}", fontsize=6.5, va="top")
    ax.text(0.97, 0.6, f"{len(peaks):,} channels, 36 sessions\n{100 * (peaks >= 0.3).mean():.0f}% above 0.3",
            transform=ax.transAxes, ha="right", fontsize=6.5)
    ax.set_xlabel("velocity R² of one channel (best lead)")
    ax.set_ylabel("channels")
    ax.set_title("No channel suffices alone", loc="left")
    curves = None

    for a_, k_ in zip(fig.axes, "abc"):
        style.panel_label(a_, k_, x=-0.25 if k_ != "b" else -0.12)
    style.save(fig, HERE / "figures" / "fig_motor_tuning")
    print(f"wrote figures/fig_motor_tuning; example {EXAMPLE}: {len(tuned)}/{len(r2)} tuned; single-channel R2 median "
          f"{med:.3f}, share >= 0.3: {(peaks >= 0.3).mean():.3f}")


if __name__ == "__main__":
    main()
