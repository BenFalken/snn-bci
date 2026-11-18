"""
Motor figure: what each electrode records drifts from session to session, so a decoder fixed on one day degrades on
the next. This is why the Speck SNN is trained per session and the LSNN is pre-trained on the other sessions and then
fine-tuned on each.

Data: the 36 Indy sessions' held-out segments (motor_data; about 80 s each, 4 ms bins), rates smoothed by 40 ms and
paired with the velocity 160 ms later. Per session and channel: the tuning weights (bx, by) of rate ~ b0 + bx vx +
by vy on z-scored rates, the preferred direction atan2(by, bx), and the mean rate. Sessions span June 2016 to
January 2017 (630 pairs).

    a  per pair of sessions, against the days between them: how far the preferred directions of channels tuned in
       both sessions moved (median absolute change; 90 degrees = unrelated)
    b  per pair: correlation of the 96 channels' tuning weights, and of their mean rates, between the two sessions
    c  a linear velocity readout (ridge) fitted on one session's first half and scored on another session's second
       half, as a share of its score on its own session's second half; 1 = transfers fully

    python fig_motor_drift.py      -> figures/fig_motor_drift.{pdf,svg,png}
"""

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import motor_data as md
import style

HERE = Path(__file__).parent
LEAD_MS = 160.0
MIN_R2 = 0.02
LAM = 10.0


def per_session(s):
    X, V = md.lead(*md.load(s), LEAD_MS)
    sd = X.std(0)
    ok = sd > 1e-9
    Z = np.zeros_like(X)
    Z[:, ok] = (X[:, ok] - X[:, ok].mean(0)) / sd[ok]
    A = np.column_stack([np.ones(len(V)), V])
    B, *_ = np.linalg.lstsq(A, Z, rcond=None)
    r2 = np.zeros(X.shape[1])
    r2[ok] = 1 - (Z[:, ok] - A @ B[:, ok]).var(0) / Z[:, ok].var(0)
    n = len(V) // 2
    mu, s_ = X[:n].mean(0), X[:n].std(0) + 1e-9
    F = np.column_stack([np.ones(len(V)), (X - mu) / s_])
    W = np.linalg.solve(F[:n].T @ F[:n] + LAM * np.eye(F.shape[1]), F[:n].T @ V[:n])
    return dict(b=B[1:].T, r2=r2, rate=X.mean(0), X=X, V=V, n=n, W=W, mu=mu, sd=s_, date=md.date(s))


def readout_corr(W, mu, sd, X, V):
    P = np.column_stack([np.ones(len(V)), (X - mu) / sd]) @ W
    return np.mean([np.corrcoef(P[:, i], V[:, i])[0, 1] for i in range(2)])


def binned(ax, x, y, color, label=None, edges=(0, 1, 7, 30, 60, 120, 240)):
    xb, yb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if np.isfinite(y[m]).sum() >= 3:
            xb.append(np.median(x[m]) + 1); yb.append(np.nanmedian(y[m]))
    ax.plot(xb, yb, "o-", color=color, ms=3.5, lw=1.4, label=label, zorder=4)


def main():
    S = md.sessions()
    D = {s: per_session(s) for s in S}
    same = {s: readout_corr(D[s]["W"], D[s]["mu"], D[s]["sd"], D[s]["X"][D[s]["n"]:], D[s]["V"][D[s]["n"]:])
            for s in S}
    days, dpd, rtune, rrate, transfer = [], [], [], [], []
    for a, b in permutations(S, 2):
        A, B = D[a], D[b]
        dd = abs((B["date"] - A["date"]).days)
        both = (A["r2"] >= MIN_R2) & (B["r2"] >= MIN_R2)
        if a < b:
            days.append(dd)
            pa, pb = np.arctan2(A["b"][both, 1], A["b"][both, 0]), np.arctan2(B["b"][both, 1], B["b"][both, 0])
            dpd.append(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (pa - pb)))))) if both.sum() >= 5 else np.nan)
            rtune.append(np.corrcoef(A["b"].ravel(), B["b"].ravel())[0, 1])
            rrate.append(np.corrcoef(A["rate"], B["rate"])[0, 1])
        own = same[b]
        transfer.append((dd, readout_corr(A["W"], A["mu"], A["sd"], B["X"][B["n"]:], B["V"][B["n"]:]) / own))
    days, dpd, rtune, rrate = map(np.array, (days, dpd, rtune, rrate))
    tdays, tval = map(np.array, zip(*transfer))

    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 58 * style.MM), gridspec_kw=dict(wspace=0.45))
    dot = dict(s=5, alpha=0.25, lw=0)

    ax = axes[0]
    ax.scatter(days + 1, dpd, color=style.MUTED, **dot)
    binned(ax, days, dpd, style.INK)
    ax.axhline(90, color=style.MUTED, ls=":", lw=0.8)
    ax.text(250, 86, "unrelated", fontsize=6.5, ha="right", va="top", color=style.MUTED)
    ax.set_ylim(0, 100)
    ax.set_xlabel("days between sessions")
    ax.set_ylabel("preferred-direction change (°)")
    ax.set_title("Tuning moves between sessions", loc="left")

    ax = axes[1]
    ax.scatter(days + 1, rtune, color=style.COLOR["lsnn"], **dot)
    ax.scatter(days + 1, rrate, color=style.COLOR["lstm"], **dot)
    binned(ax, days, rtune, style.COLOR["lsnn"], "tuning weights")
    binned(ax, days, rrate, style.COLOR["lstm"], "mean rates")
    ax.axhline(0, color=style.MUTED, lw=0.6)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel("days between sessions")
    ax.set_ylabel("correlation across channels")
    ax.set_title("Which channel carries what", loc="left")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=6.5)

    ax = axes[2]
    ax.scatter(tdays + 1, tval, color=style.COLOR["speck"], **dot)
    binned(ax, tdays, tval, style.COLOR["speck"])
    ax.axhline(1, color=style.MUTED, ls=":", lw=0.8)
    ax.axhline(0, color=style.MUTED, lw=0.6)
    ax.set_ylim(-0.5, 1.2)
    ax.set_xlabel("days between sessions")
    ax.set_ylabel("score on the other session / own")
    ax.set_title("A fixed decoder does not carry over", loc="left")

    for ax, k in zip(axes, "abc"):
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 8, 31, 101])
        ax.set_xticklabels(["0", "1", "7", "30", "100"])
        ax.set_xlim(0.8, 260)
        ax.minorticks_off()
        style.panel_label(ax, k, x=-0.25)
    style.save(fig, HERE / "figures" / "fig_motor_drift")
    near = days <= 7
    print(f"wrote figures/fig_motor_drift; {len(days)} pairs. Within a week / beyond 30 days (medians): PD change "
          f"{np.nanmedian(dpd[near]):.0f} / {np.nanmedian(dpd[days > 30]):.0f} deg; tuning-weight r "
          f"{np.median(rtune[near]):.2f} / {np.median(rtune[days > 30]):.2f}; rate r {np.median(rrate[near]):.2f} / "
          f"{np.median(rrate[days > 30]):.2f}; transfer {np.median(tval[tdays <= 7]):.2f} / "
          f"{np.median(tval[tdays > 30]):.2f}; own-session readout r median {np.median(list(same.values())):.2f}")


if __name__ == "__main__":
    main()
