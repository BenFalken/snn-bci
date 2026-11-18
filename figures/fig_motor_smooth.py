"""
Motor figure: hand velocity changes slowly against the 4 ms output step, so both decoders can smooth their output
over tens of milliseconds without losing the movement, and doing so removes the step-to-step noise of spikes.

Data: the Indy sessions' held-out segments (motor_data; 4 ms bins). Velocity spectra by Welch's method (4 s windows).

    a  ten seconds of hand velocity (x) of one session, with the 4 ms output steps of a short stretch marked
    b  every session's cumulative share of velocity power against frequency (median and interquartile band); the
       frequency below which 95% of the power lies is marked. The dataset's velocity is derived from finger
       position, so part of its smoothness above ~10 Hz comes from that derivation
    c  the gain of each readout's smoothing against frequency, over the same axis: the Speck SNN's two EMA stages
       (decay 0.8 per 4 ms step, about 18 ms each) and the LSNN's two leaky stages at their 40 ms starting time
       constant (trained per session from there); both pass the band that holds velocity and cut what lies above

    python fig_motor_smooth.py      -> figures/fig_motor_smooth.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import motor_data as md
import style

HERE = Path(__file__).parent
FS = 1000.0 / md.STEP_MS                     # 250 Hz
EXAMPLE = "indy_20161017_02"


def two_stage_gain(f, decay):
    """|H(f)| of two first-order low-pass stages y <- a y + (1 - a) x in series, at FS."""
    z = np.exp(-2j * np.pi * f / FS)
    h = (1 - decay) / (1 - decay * z)
    return np.abs(h) ** 2


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 58 * style.MM),
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1], wspace=0.45))

    # a: a velocity trace with 4 ms steps
    ax = axes[0]
    _, V = md.load(EXAMPLE, smooth_ms=0)
    t = np.arange(len(V)) / FS
    w = (t >= 20) & (t < 30)
    ax.plot(t[w], V[w, 0], color=style.INK, lw=0.9)
    ins = ax.inset_axes([0.58, 0.62, 0.38, 0.33])
    z = (t >= 24.0) & (t < 24.2)
    ins.step(t[z], V[z, 0], where="post", color=style.COLOR["speck"], lw=0.8)
    ins.plot(t[z], V[z, 0], ".", color=style.COLOR["speck"], ms=2)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("200 ms: 50 steps", fontsize=6, pad=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hand velocity x (mm/s)")
    ax.set_title("Hand velocity, one session", loc="left")

    # b: spectra
    ax = axes[1]
    specs = []
    for s in md.sessions():
        _, Vs = md.load(s, smooth_ms=0)
        f, p = welch(Vs - Vs.mean(0), fs=FS, nperseg=int(4 * FS), axis=0)
        p = p.sum(1)
        specs.append(p / p.sum())
    specs = np.array(specs)
    cum = np.cumsum(specs, axis=1)
    lo, mid, hi = np.percentile(cum, [25, 50, 75], axis=0)
    ax.fill_between(f[1:], lo[1:], hi[1:], color=style.MUTED, alpha=0.3, lw=0)
    ax.plot(f[1:], mid[1:], color=style.INK, lw=1.2)
    f95 = np.array([f[np.argmax(c >= 0.95)] for c in cum])
    ax.axhline(0.95, color=style.MUTED, lw=0.6, ls=":")
    ax.axvline(np.median(f95), color=style.COLOR["lstm"], ls="--", lw=0.8)
    ax.text(np.median(f95) * 1.15, 0.3, f"95% of power\nbelow {np.median(f95):.1f} Hz", fontsize=6.5,
            color=style.COLOR["lstm"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(f[1], FS / 2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("cumulative share of power")
    ax.set_title(f"Velocity power, {len(specs)} sessions", loc="left")

    # c: readout gains
    ax = axes[2]
    ff = np.logspace(np.log10(f[1]), np.log10(FS / 2), 400)
    lsnn_decay = np.exp(-md.STEP_MS / 40.0)
    ax.plot(ff, two_stage_gain(ff, 0.8), color=style.COLOR["speck"], lw=1.2, label="Speck: 2 × EMA, ~18 ms")
    ax.plot(ff, two_stage_gain(ff, lsnn_decay), color=style.COLOR["lsnn"], lw=1.2, label="LSNN: 2 × leaky, 40 ms")
    ax.axvspan(f[1], np.median(f95), color=style.COLOR["lstm"], alpha=0.12, lw=0)
    ax.text(f[1] * 1.3, 0.45, "velocity\nband", fontsize=6.5, color=style.COLOR["lstm"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(f[1], FS / 2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("readout gain")
    ax.set_title("What each readout passes", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=6.5)
    for name, d in (("Speck", 0.8), ("LSNN", lsnn_decay)):
        print(f"{name}: gain {two_stage_gain(np.median(f95), d):.2f} at {np.median(f95):.1f} Hz, "
              f"half-power at {ff[np.argmax(two_stage_gain(ff, d) < 0.5)]:.1f} Hz")

    for ax, k in zip(axes, "abc"):
        style.panel_label(ax, k, x=-0.25)
    style.save(fig, HERE / "figures" / "fig_motor_smooth")
    print(f"wrote figures/fig_motor_smooth; 95% of velocity power below {np.median(f95):.2f} Hz "
          f"(IQR {np.percentile(f95, 25):.2f}-{np.percentile(f95, 75):.2f})")


if __name__ == "__main__":
    main()
