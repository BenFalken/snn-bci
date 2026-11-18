"""
Motor figure: the chip fires on each input event as it arrives, not on a step's summed input as training does. With
dense activity this over-drives the network layer by layer; a label-free rise of layer 1's threshold removes the
surge. Shown on a sparse session (Indy) and a dense one (Jenkins M1).

Computed here, on training trials only (data the evaluation never uses), with the final per-session Speck networks:
    trained     the network as trained (PyTorch: each step's input summed before any neuron fires)
    chip, x1    the chip's event-by-event firing in software (snn-decoders snndec.chip.run_events, which matches the
                devkit: audio r 0.714 model against 0.713 chip), at the trained thresholds
    chip, xk    the same with layer 1's threshold raised by the session's label-free factor (speck_l1_factors.txt)
run_events returns only the last layer; layer_counts below repeats its arithmetic and keeps every layer's spikes,
and is checked against it.

    a  spikes per neuron per 4 ms step in each layer, Indy (one session)
    b  the same for Jenkins M1, session 09-16: the surge at x1 and its removal at x3
    c  Jenkins 09-16, one training trial: hand velocity (x) and what each version of the network decodes

    python fig_motor_events.py      -> figures/fig_motor_events.{pdf,svg,png}
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import style

HERE = Path(__file__).parent
SNNDEC = Path.home() / "Documents/GitHub/snn-decoders"
sys.path.insert(0, str(SNNDEC))
sys.path.insert(0, str(SNNDEC / "experiments/chip"))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402
from motor_l1_sweep import decode  # noqa: E402

PS = Path.home() / "Downloads/speck_per_session"
SCALERS = str(PS / "main/snn_training/velocity_scalers.json")
CASES = {"indy": ("bmi/indy", "indy_20160921_01"),
         "jenkins_m1": ("hkm/jenkins_m1", "sub-Jenkins_ses-20090916_behavior+ecephys")}
N_TRIALS = 3
CACHE = HERE / "data" / "motor_events.json"


def factor(subj, sess):
    for line in open(PS / "speck_l1_factors.txt"):
        p = line.split()
        if len(p) == 3 and p[0] == subj and p[1] == sess:
            return float(p[2])
    raise KeyError((subj, sess))


def layer_counts(network, events, k1):
    """snndec.chip.run_events with every layer's spike counts kept: (list of (T, n_l) arrays)."""
    seq = chip.quantized_layers(network)
    W = [m.weight.detach().numpy()[:, :, 0, 0] for m in seq if hasattr(m, "weight")]
    neurons = [m for m in seq if not hasattr(m, "weight")]
    scale = [k1] + [1.0] * (len(W) - 1)
    thr = [float(n.spike_threshold) * k for n, k in zip(neurons, scale)]
    floor = [float(n.min_v_mem) * k if n.min_v_mem is not None else -np.inf for n, k in zip(neurons, scale)]
    v = [np.zeros(w.shape[0]) for w in W]
    out = [np.zeros((len(events), w.shape[0])) for w in W]
    for t, row in enumerate(events):
        incoming = np.flatnonzero(row)
        for li, w in enumerate(W):
            fired = []
            for i in incoming:
                v[li] += w[:, i]
                hot = np.flatnonzero(v[li] >= thr[li])          # hard reset, as every deployed network
                if len(hot):
                    fired.extend(hot.tolist())
                    v[li][hot] = 0.0
                np.maximum(v[li], floor[li], out=v[li])
            np.add.at(out[li][t], fired, 1)
            incoming = fired
    return out


@torch.no_grad()
def run_case(name):
    subj_path, sess = CASES[name]
    k = factor(name, sess)
    model, _ = load(str(PS / f"data/snn_checkpoints/{subj_path}/per_session_delay_binarized/{sess}/best_model_weights.pth"))
    model.eval()
    sdir = str(PS / f"data/snn_datasets/{subj_path}/mua/{sess}")
    scale = data.velocity_scaler(sdir, SCALERS)
    trials = data.motor_split(sdir, "train", SCALERS)[:N_TRIALS]
    nets = {1.0: chip.discretized(chip.linear_stack(model, 1.0), model.host.n_out)}
    if k != 1.0:
        nets[k] = chip.discretized(chip.linear_stack(model, k), model.host.n_out)
    rates = {"trained": None, **{f"x{s:g}": None for s in nets}}
    steps, in_events, trace = 0, 0.0, None
    for ti, t in enumerate(trials):
        y_ref = model(torch.as_tensor(t.x)[None])[0].numpy()
        ev = chip.chip_events(model, model.input_events[0].numpy())
        steps += len(ev)
        in_events += float(ev.sum())
        tr = np.array([float(s[0].sum()) for s in model.layer_spikes])
        rates["trained"] = tr if rates["trained"] is None else rates["trained"] + tr
        outs = {}
        for s, net in nets.items():
            cnt = layer_counts(net, ev, s)
            if ti == 0:                                          # agree with snndec's own simulator
                assert np.array_equal(cnt[-1], chip.run_events(net, ev, (s,))), "layer_counts differs from run_events"
            tot = np.array([c.sum() for c in cnt])
            key = f"x{s:g}"
            rates[key] = tot if rates[key] is None else rates[key] + tot
            outs[key] = data.unscale_velocity(decode(model, cnt[-1]), *scale)[:, 0]
        if ti == 0:
            trace = dict(truth=data.unscale_velocity(t.y, *scale)[:, 0].tolist(),
                         trained=data.unscale_velocity(y_ref, *scale)[:, 0].tolist(),
                         **{kk: vv.tolist() for kk, vv in outs.items()})
    widths = list(model.widths)
    res = dict(session=sess, factor=k, steps=steps, input_events_per_step=in_events / steps, widths=widths,
               per_neuron={kk: (np.array(v) / steps / np.array(widths)).tolist() for kk, v in rates.items()},
               trace=trace)
    print(f"{name}: x{k:g}; input events/step {in_events / steps:.1f}; " + "; ".join(
        f"{kk} " + " ".join(f"{x:.3f}" for x in vv) for kk, vv in res["per_neuron"].items()))
    return res


def bars(ax, res, title):
    names = [n for n in ("trained", "x1", f"x{res['factor']:g}") if n in res["per_neuron"]]
    names = list(dict.fromkeys(names))
    labels = {"trained": "as trained", "x1": "chip, ×1", f"x{res['factor']:g}": f"chip, ×{res['factor']:g}"}
    colors = {"trained": style.MUTED, "x1": "#c0392b", f"x{res['factor']:g}": style.COLOR["speck"]}
    L = len(res["widths"])
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(L) + (i - (len(names) - 1) / 2) * w, res["per_neuron"][n], width=w, color=colors[n],
               label=labels[n])
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"layer {i + 1}\n({n})" for i, n in enumerate(res["widths"][:-1])] + [f"output\n({res['widths'][-1]})"],
                       fontsize=6)
    ax.set_yscale("log")
    ax.set_ylabel("spikes per neuron per step")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)


def main():
    if CACHE.is_file() and "--recompute" not in sys.argv:
        R = json.load(open(CACHE))
    else:
        torch.set_num_threads(4)
        R = {n: run_case(n) for n in CASES}
        json.dump(R, open(CACHE, "w"))
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 62 * style.MM),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.42))
    bars(axes[0], R["indy"], f"Indy, {R['indy']['input_events_per_step']:.0f} input events/step")
    bars(axes[1], R["jenkins_m1"], f"Jenkins M1, {R['jenkins_m1']['input_events_per_step']:.0f} events/step")
    ymin = min(min(v) for r in R.values() for v in r["per_neuron"].values())
    ymax = max(max(v) for r in R.values() for v in r["per_neuron"].values())
    for ax in axes[:2]:
        ax.set_ylim(ymin / 2, ymax * 8)

    ax = axes[2]
    tr = R["jenkins_m1"]["trace"]
    k = R["jenkins_m1"]["factor"]
    t = np.arange(len(tr["truth"])) * 0.004
    pk = t[int(np.argmax(np.abs(tr["truth"])))]
    w = (t >= pk - 0.5) & (t <= pk + 0.5)
    ax.plot(t[w], np.array(tr["truth"])[w], color=style.INK, lw=1.2, label="hand velocity")
    ax.plot(t[w], np.array(tr["trained"])[w], color=style.MUTED, lw=0.9, label="as trained")
    ax.plot(t[w], np.array(tr["x1"])[w], color="#c0392b", lw=0.9, label="chip, ×1")
    ax.plot(t[w], np.array(tr[f"x{k:g}"])[w], color=style.COLOR["speck"], lw=0.9, label=f"chip, ×{k:g}")
    ax.set_xlabel("time in trial (s), around one reach")
    ax.set_ylabel("velocity x (mm/s)")
    ax.set_title("Jenkins: what each decodes", loc="left")
    ax.legend(loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=6.5)
    for ax, kk in zip(axes, "abc"):
        style.panel_label(ax, kk, x=-0.3, y=1.1)
    style.save(fig, HERE / "figures" / "fig_motor_events")
    print("wrote figures/fig_motor_events")


if __name__ == "__main__":
    main()
