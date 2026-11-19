"""
Figure: why the chip network is calibrated, and the two calibrations (condenses fig_motor_events,
fig_calibration_overview and fig_calibration_results into one figure).

  Why: the chip fires event by event (monkey J, M1 array, session 2009-09-16, one training reach; data the evaluation
  never uses)
    a  each layer's firing over time: the trained network (each step's input summed before any neuron fires), the
       chip at the trained thresholds (x1) and with layer 1's threshold raised by the label-free factor (x3)
    b  what each one decodes: the hand's velocity, the trained network, the chip at x1 (the saturated output
       averages its preferred directions out) and at x3
  The two calibrations, set once on training data
    c  input end: agreement of the chip model's output with the trained network's own output against the factor, for
       every deployed network that needed it; dots: the label-free choice
    d  the same sweeps scored against the labels afterwards (no part of the choice)
    e  output end: the chip's accuracy as a share of software with the trained readout and the refitted one, on the
       real chip

    python3 fig_calibration.py [--recompute]      -> figures/fig_calibration.{pdf,svg,png}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import calibration_panels as cp
import style
from style import INK, MUTED

HERE = Path(__file__).parent
CACHE = HERE / "data" / "calibration_motor_time.json"
RED = "#c0392b"


def compute():
    """Each layer's spikes per neuron per step over one Jenkins training reach, three ways, and the decoded velocity."""
    import torch
    import fig_motor_events as me
    torch.set_num_threads(4)
    subj_path, sess = me.CASES["jenkins_m1"]
    k = me.factor("jenkins_m1", sess)
    model, _ = me.load(str(me.PS / f"data/snn_checkpoints/{subj_path}/per_session_delay_binarized/{sess}/best_model_weights.pth"))
    model.eval()
    sdir = str(me.PS / f"data/snn_datasets/{subj_path}/mua/{sess}")
    scale = me.data.velocity_scaler(sdir, me.SCALERS)
    t = me.data.motor_split(sdir, "train", me.SCALERS)[0]
    with torch.no_grad():
        y_ref = model(torch.as_tensor(t.x)[None])[0].numpy()
        ev = me.chip.chip_events(model, model.input_events[0].numpy())
        layers = {"trained": [s[0].numpy().mean(1).tolist() for s in model.layer_spikes]}
        vel = {"truth": me.data.unscale_velocity(t.y, *scale)[:, 0].tolist(),
               "trained": me.data.unscale_velocity(y_ref, *scale)[:, 0].tolist()}
        for s in (1.0, k):
            net = me.chip.discretized(me.chip.linear_stack(model, s), model.host.n_out)
            cnt = me.layer_counts(net, ev, s)
            assert np.array_equal(cnt[-1], me.chip.run_events(net, ev, (s,))), "layer_counts differs from run_events"
            layers[f"x{s:g}"] = [c.mean(1).tolist() for c in cnt]
            vel[f"x{s:g}"] = me.data.unscale_velocity(me.decode(model, cnt[-1]), *scale)[:, 0].tolist()
        # every neuron's spikes, for the rasters, and the readout's view (its smoothed output population) each step
        full = {"trained": [s[0].numpy() for s in model.layer_spikes]}
        for s in (1.0, k):
            net = me.chip.discretized(me.chip.linear_stack(model, s), model.host.n_out)
            full[f"x{s:g}"] = me.layer_counts(net, ev, s)
        pops = {}
        for key, cnt in full.items():
            st, us = model.readout.init_state(1, "cpu"), []
            ro = model.readout
            d = torch.sigmoid(ro.decay_raw)
            for r in cnt[-1]:
                u = torch.as_tensor(r, dtype=torch.float32)[None]
                for j in range(ro.stages):
                    st["e"][j] = u if st["e"][j] is None else d * st["e"][j] + u
                    u = st["e"][j]
                us.append((u * (1 - d) ** ro.stages if ro.normalize else u)[0].numpy())
            pops[key] = np.array(us)                                      # (T, 36): what the population head reads
        np.savez_compressed(CACHE.with_suffix(".npz"),
                            **{f"{key}_L{i}": np.clip(c, 0, 255).astype(np.uint8) for key, cs in full.items()
                               for i, c in enumerate(cs)},
                            **{f"{key}_pop": v for key, v in pops.items()},
                            positions=model.readout.positions.numpy(), scale=np.array(scale, dtype=float))
    return dict(session=sess, factor=k, widths=list(model.widths), layers=layers, velocity=vel,
                input_events_per_step=float(ev.sum() / len(ev)))


def smooth(x, n=10):                                   # 40 ms box (10 steps of 4 ms), for display only
    return np.convolve(np.asarray(x), np.ones(n) / n, mode="same")


def tint(color):
    """White to `color`: the shading of one condition's rasters."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("t", ["#ffffff", color])


def main():
    if CACHE.is_file() and CACHE.with_suffix(".npz").is_file() and "--recompute" not in sys.argv:
        R = json.load(open(CACHE))
    else:
        R = compute()
        json.dump(R, open(CACHE, "w"))
    Z = np.load(CACHE.with_suffix(".npz"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snndec import data
    k = R["factor"]
    kk = f"x{k:g}"
    KEYS = ("trained", "x1", kk)
    LAB = {"trained": "as trained", "x1": "chip, ×1", kk: f"chip, ×{k:g}"}
    LONG = {"trained": "as trained (each step's input summed first)",
            "x1": "on the chip, ×1 (each input event fires at once)", kk: f"on the chip, layer 1 threshold ×{k:g}"}
    COL = {"trained": style.MUTED, "x1": RED, kk: cp.STEP2}          # green: not the Jenkins orange of d-e

    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]   # the arrow glyph in f
    W, H = 183.0, 178.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    A = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def band(y, text, color):
        fig.text(3 / W, 1 - y / H, text, fontsize=6.6, color="white", fontweight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc=color, ec="none"))

    band(2, f"WHY · the chip fires event by event: monkey J, {R['input_events_per_step']:.0f} input events per 4 ms "
            f"step, one training reach", RED)
    # the shared legend, overhead
    for i, key in enumerate(KEYS):
        x = 8 + i * 62
        fig.add_artist(plt.Line2D([x / W, (x + 5) / W], [1 - 10.5 / H] * 2, color=COL[key], lw=2.2))
        fig.text((x + 6.5) / W, 1 - 10.5 / H, LONG[key], fontsize=5.8, va="center", color=INK)

    v = np.array(R["velocity"]["truth"])
    tt = np.arange(len(v)) * 0.004
    p = int(np.argmax(np.abs(v)))
    win = (tt >= tt[p] - 0.5) & (tt <= tt[p] + 0.5)
    tw = tt[win] - tt[p]

    # a: the network, layer by layer: every neuron a row, its spikes per 4 ms step shaded
    names = [f"layer {i + 1}" for i in range(len(R["widths"]) - 1)] + ["output"]
    heights = [10, 13, 13, 7]                                         # mm per layer strip (output on top)
    top0, gap, colw = 17, 3.2, 52
    axa = {}
    for c, key in enumerate(KEYS):
        x0 = 22 + c * (colw + 6)
        y = top0
        for li in reversed(range(len(names))):                       # output at the top, layer 1 at the bottom
            ax = A(x0, y, colw, heights[li])
            X = Z[f"{key}_L{li}"][win].T.astype(float)
            if li < len(names) - 1:                                   # hidden layers: a spike or none
                X, vmax = (X > 0).astype(float), 1.0
            else:                                                     # output: its count, on this version's own range
                vmax = max(np.percentile(X, 99), 1e-9)
            ax.imshow(X, aspect="auto", cmap=tint(COL[key] if key != "trained" else "#3b3f47"), vmin=0, vmax=vmax,
                      interpolation="nearest", extent=[tw[0], tw[-1], X.shape[0], 0], rasterized=True)
            ax.axvline(0, color="#000000", lw=0.4, ls=":", alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#b8bcc4"); sp.set_linewidth(0.5)
            if c == 0:
                ax.text(-0.04, 0.5, f"{names[li]}\n({R['widths'][li]})", transform=ax.transAxes, ha="right",
                        va="center", fontsize=5.6, color=INK)
            if li < len(names) - 1:                                   # activity flows up into the next layer
                ax.annotate("", xy=(0.5, 1.0 + 1.9 / heights[li] * 1.3), xytext=(0.5, 1.0), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=COL[key], lw=0.9, mutation_scale=6))
            axa[(c, li)] = ax
            y += heights[li] + gap
        axa[(c, len(names) - 1)].set_title(LAB[key], loc="center", fontsize=7, color=COL[key], fontweight="bold",
                                           pad=2.5)
        bot = axa[(c, 0)]
        bot.set_xticks([-0.4, 0, 0.4]); bot.tick_params(labelsize=5.2, length=2)
        bot.set_xlabel("time from peak hand speed (s)", fontsize=5.8, labelpad=1)

    # b: what the readout sees at peak speed: the x-group's 18 output neurons, by preferred velocity
    pos = Z["positions"]
    pref = data.unscale_velocity(pos[:, None].repeat(2, 1), *Z["scale"])[:, 0]
    n = len(pos)
    yb = 82
    for r, key in enumerate(KEYS):
        ax = A(20, yb + r * 12.5, 62, 9.5)
        g = Z[f"{key}_pop"][p].reshape(2, -1)[0]
        ax.bar(pref, g / max(g.max(), 1e-12), width=(pref[1] - pref[0]) * 0.8, color=COL[key])
        dec = R["velocity"][key][p]
        hi = pref[-1] + 60
        if v[p] <= hi:
            ax.axvline(v[p], color=INK, lw=0.9, ls="--")
        hand_off = v[p] > hi
        ax.plot([dec], [1.18], marker="v", color=COL[key], ms=5, clip_on=False, mec="white", mew=0.4)
        ax.set_xlim(pref[0] - 60, pref[-1] + 60)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.tick_params(labelsize=5.2, length=2)
        ax.text(1.01, 0.5, LAB[key], transform=ax.transAxes, fontsize=5.8, color=COL[key], va="center")
        if r < len(KEYS) - 1:
            ax.set_xticklabels([])
        if r == 0:
            ax.set_title("What the readout sees at peak speed", loc="left", fontsize=7, pad=9)
    ax.set_xlabel("each neuron's preferred velocity x (mm/s)", fontsize=5.8, labelpad=1)
    axb0 = fig.axes[-3]
    axb0.text(0.0, 1.0, "▼ decoded: the weighted mean of the neurons' preferred velocities  ·  the hand: "
              f"{v[p]:,.0f} mm/s, beyond the range", transform=axb0.transAxes, fontsize=5.0, color=MUTED, va="bottom")

    # c: what each decodes
    c_ax = A(122, 82, 58, 34)
    c_ax.plot(tw, v[win], color=INK, lw=1.4, label="hand")
    for key in KEYS:
        c_ax.plot(tw, np.array(R["velocity"][key])[win], color=COL[key], lw=1.0)
    c_ax.axvline(0, color="#000000", lw=0.4, ls=":", alpha=0.6)
    c_ax.set_xlim(-0.5, 0.5)
    c_ax.set_xlabel("time from peak hand speed (s)", fontsize=6)
    c_ax.set_ylabel("velocity x (mm/s)", fontsize=6)
    c_ax.tick_params(labelsize=5.5)
    c_ax.set_title("What each one decodes", loc="left", fontsize=7)
    c_ax.legend(fontsize=5.5, frameon=False, loc="upper left")

    band(126, "THE TWO CALIBRATIONS · set once on training data; the chip's weights never change", cp.STEP1)
    d, e, f = A(18, 137, 42, 32), A(76, 137, 42, 32), A(131, 137, 50, 32)
    cp.panel_sweep(d, "agree")
    cp.panel_sweep(e, "acc")
    cp.panel_gains(f)
    d.set_title("Input end: threshold, no labels", loc="left")
    e.set_title("Checked against labels", loc="left")
    f.set_title("Output end: readout refit", loc="left")
    for ax, l, x, y in ((axa[(0, len(names) - 1)], "a", -0.36, 1.45), (axb0, "b", -0.25, 1.55),
                        (c_ax, "c", -0.24, 1.06), (d, "d", -0.38, 1.05), (e, "e", -0.34, 1.05), (f, "f", -0.08, 1.05)):
        ax.text(x, y, l, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    style.save(fig, HERE / "figures" / "fig_calibration")
    print(f"wrote figures/fig_calibration; monkey J {R['session']}, x{k:g}, "
          f"{R['input_events_per_step']:.0f} events/step; decoded at peak: " +
          ", ".join(f"{LAB[kk_]} {R['velocity'][kk_][p]:.0f}" for kk_ in KEYS) + f" (hand {v[p]:.0f}) mm/s")


if __name__ == "__main__":
    main()
