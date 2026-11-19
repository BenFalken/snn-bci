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
    return dict(session=sess, factor=k, widths=list(model.widths), layers=layers, velocity=vel,
                input_events_per_step=float(ev.sum() / len(ev)))


def smooth(x, n=10):                                   # 40 ms box (10 steps of 4 ms), for display only
    return np.convolve(np.asarray(x), np.ones(n) / n, mode="same")


def main():
    if CACHE.is_file() and "--recompute" not in sys.argv:
        R = json.load(open(CACHE))
    else:
        R = compute()
        json.dump(R, open(CACHE, "w"))
    k = R["factor"]
    kk = f"x{k:g}"
    LAB = {"trained": "as trained", "x1": "chip, ×1", kk: f"chip, ×{k:g}"}
    COL = {"trained": style.MUTED, "x1": RED, kk: cp.STEP2}          # green: not the Jenkins orange of c-d

    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]   # the arrow glyph in e
    W, H = 183.0, 128.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    A = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def band(y, text, color):
        fig.text(3 / W, 1 - y / H, text, fontsize=6.6, color="white", fontweight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc=color, ec="none"))

    band(2, f"WHY · the chip fires event by event: monkey J, {R['input_events_per_step']:.0f} input events per 4 ms "
            f"step, one training reach", RED)
    # the reach window
    v = np.array(R["velocity"]["truth"])
    tt = np.arange(len(v)) * 0.004
    pk = tt[int(np.argmax(np.abs(v)))]
    win = (tt >= pk - 0.5) & (tt <= pk + 0.5)
    tw = tt[win] - pk

    # a: each layer over time
    names = [f"layer {i + 1}" for i in range(len(R["widths"]) - 1)] + ["output"]
    axl = [A(20, 13 + i * 12.5, 70, 10) for i in range(len(names))]
    for i, ax in enumerate(axl):
        for key in ("trained", "x1", kk):
            ax.plot(tw, np.maximum(smooth(R["layers"][key][i])[win], 1e-3), color=COL[key],
                    lw=1.0 if key != "trained" else 1.2, label=LAB[key])
        ax.set_xlim(-0.5, 0.5)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.text(1.01, 0.5, f"{names[i]}\n({R['widths'][i]})", transform=ax.transAxes, fontsize=5.8, va="center")
        ax.tick_params(labelsize=5.5)
        if i < len(axl) - 1:
            ax.set_xticklabels([])
    axl[0].set_title("The surge, layer by layer", loc="left")
    axl[-1].set_xlabel("time from peak hand speed (s)")
    fig.text(9 / W, 1 - 39 / H, "spikes per neuron per step (log)", rotation=90, fontsize=6.5, va="center", ha="center")

    # b: what each decodes
    b = A(112, 13, 68, 47)
    b.plot(tw, v[win], color=style.INK, lw=1.4, label="hand velocity")
    for key in ("trained", "x1", kk):
        b.plot(tw, np.array(R["velocity"][key])[win], color=COL[key], lw=1.0, label=LAB[key])
    b.set_xlim(-0.5, 0.5)
    b.set_xlabel("time from peak hand speed (s)")
    b.set_ylabel("velocity x (mm/s)")
    b.set_title("What each one decodes", loc="left")
    b.legend(fontsize=5.8, frameon=False, loc="lower left")

    band(71, "THE TWO CALIBRATIONS · set once on training data; the chip's weights never change", cp.STEP1)
    c, d, e = A(18, 84, 42, 34), A(76, 84, 42, 34), A(131, 84, 50, 34)
    cp.panel_sweep(c, "agree")
    cp.panel_sweep(d, "acc")
    cp.panel_gains(e)
    c.set_title("Input end: threshold, no labels", loc="left")
    d.set_title("Checked against labels", loc="left")
    e.set_title("Output end: readout refit", loc="left")
    for ax, l, x, y in ((axl[0], "a", -0.2, 1.25), (b, "b", -0.17, 1.04), (c, "c", -0.38, 1.05), (d, "d", -0.34, 1.05),
                        (e, "e", -0.08, 1.05)):
        ax.text(x, y, l, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    style.save(fig, HERE / "figures" / "fig_calibration")
    print(f"wrote figures/fig_calibration; monkey J {R['session']}, x{k:g}, "
          f"{R['input_events_per_step']:.0f} events/step")


if __name__ == "__main__":
    main()
