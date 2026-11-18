"""
The Speck SNN with an input delay bank, against the per-session Speck SNN, on one session.

Speck runs feedforward IAF layers only: no recurrence, no adaptation, no synaptic
filters, so the per-session SNN has no memory beyond its membranes. The delay
bank (train_snn.py --input-delays, models/model_bmi.py delay_bank()) gives it
some without changing the chip's network: off-chip, every input channel is also
sent with fixed delays, and the first layer sees each delayed copy as an input
channel of its own, so every neuron has one synapse (weight) per channel and
delay. On the chip the copies are input events like any other.

The chip stores every first-layer weight, so the delays cost first-layer width:
with 128 -> 64 -> 36 behind it, Speck2f maps a first layer of 512 for 96 inputs,
256 for 192 (2 delays) and 128 for 384 (4 delays). Configurations:

    per_session         96 -> 256 -> 128 -> 64      the per-session Speck SNN (snn_config.sh)
    per_session_narrow  96 -> 128 -> 128 -> 64      control: the 4-delay network's widths, no delays
    delay2              2 x 96 -> 256 -> 128 -> 64  delays 0, 100 ms; the per-session widths
    delay4              4 x 96 -> 128 -> 128 -> 64  delays 0, 40, 100, 200 ms
    delay4_short        4 x 96 -> 128 -> 128 -> 64  delays 0, 8, 16, 32 ms   (not run by default)
    delay4_mid          4 x 96 -> 128 -> 128 -> 64  delays 0, 16, 32, 64 ms  (not run by default)
    mix_sum             96 -> 256 -> 128 -> 64      delays 0, 8, 16, 32 ms, each channel's copies added
                                                    off-chip: 0-4 events per channel per step (not by default)
    mix_learned         96 -> 256 -> 128 -> 64      the same with a trained gain per channel and delay,
                                                    rounded to whole events (not run by default)
  Long delays (0-200 ms) and wider networks (not run by default):
    wide                96 -> 512 -> 128 -> 64      control: the widest first layer, no delays
    mix_sum_long        96 -> 256 -> 128 -> 64      delays 0, 40, 100, 200 ms added off-chip
    mix_sum_long_wide   96 -> 512 -> 128 -> 64      the same, widest first layer
    delay2_wide         2 x 96 -> 256 -> 256 -> 128 delays 0, 100 ms, wider later layers
    delay2_random       2 x 96 -> 256 -> 256 -> 128 each channel's second delay drawn from 4-200 ms
    delay4_wide         4 x 96 -> 128 -> 256 -> 256 delays 0, 40, 100, 200 ms, wider later layers

All: hard reset, thresholds 1.0, 2 EMA stages, no tau_syn, binarized input, the
same training as compare_lsnn.py. For each, besides the PyTorch test metrics:
whether sinabs maps it onto Speck2f, and the test error of the quantized network
(8-bit weights, integer thresholds) emulated on the host one step at a time --
the network the chip holds, though the chip itself updates per input event
(see inference/speck.py), which a devkit run measures.

Outputs in --out-dir: speck_delays.json, .md (and train_snn.py's directories).
    python snn_training/compare_speck_delays.py --data-path .../indy_20160411_01 --out-dir .../indy_20160411_01 --device cpu
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "inference")]
from compare_lsnn import evaluate, train  # noqa: E402
from dataset import CustomDataset  # noqa: E402
from train_snn import STEP_MS, build_model, unscale_velocity  # noqa: E402

THR = ["--spike-thresholds", "1.0", "1.0", "1.0", "1.0"]
CONFIGS = {
    "per_session": ["--hidden-dims", "256", "128", "64"] + THR,
    "per_session_narrow": ["--hidden-dims", "128", "128", "64"] + THR,
    "delay2": ["--hidden-dims", "256", "128", "64", "--input-delays", "0", "25"] + THR,
    "delay4": ["--hidden-dims", "128", "128", "64", "--input-delays", "0", "10", "25", "50"] + THR,
    # short delays, on the scale of the pooled SNN's synaptic filters (tau_syn 2-16 steps = 8-64 ms)
    "delay4_short": ["--hidden-dims", "128", "128", "64", "--input-delays", "0", "2", "4", "8"] + THR,
    "delay4_mid": ["--hidden-dims", "128", "128", "64", "--input-delays", "0", "4", "8", "16"] + THR,
}
# the per-session widths, the delayed copies folded back into the 96 channels off-chip (event counts per channel)
CONFIGS["mix_sum"] = ["--hidden-dims", "256", "128", "64", "--input-delays", "0", "2", "4", "8", "--input-mix", "sum"] + THR
CONFIGS["mix_learned"] = ["--hidden-dims", "256", "128", "64", "--input-delays", "0", "2", "4", "8",
                          "--input-mix", "learned"] + THR
# long delays (0-200 ms), other arrangements and wider networks (all map onto Speck2f; not run by default)
LONG = ["--input-delays", "0", "10", "25", "50"]
CONFIGS["wide"] = ["--hidden-dims", "512", "128", "64"] + THR                                   # control: width only
CONFIGS["mix_sum_long"] = ["--hidden-dims", "256", "128", "64"] + LONG + ["--input-mix", "sum"] + THR
CONFIGS["mix_sum_long_wide"] = ["--hidden-dims", "512", "128", "64"] + LONG + ["--input-mix", "sum"] + THR
CONFIGS["delay2_wide"] = ["--hidden-dims", "256", "256", "128", "--input-delays", "0", "25"] + THR
CONFIGS["delay2_random"] = ["--hidden-dims", "256", "256", "128", "--input-delays", "0", "50",
                            "--input-delays-random", "true"] + THR
CONFIGS["delay4_wide"] = ["--hidden-dims", "128", "256", "256"] + LONG + THR
# the Oscar speck_delay variants (sbatch_scripts/snn_config.sh), for local checks
WIDE4 = ["--hidden-dims", "128", "256", "256"] + THR
CONFIGS["delay4_wide_long"] = WIDE4 + ["--input-delays", "0", "25", "75", "150"]
CONFIGS["delay4_wide_xlong"] = WIDE4 + ["--input-delays", "0", "50", "150", "300"]
CONFIGS["delay4_wide_random"] = WIDE4 + LONG + ["--input-delays-random", "true"]
# from the channels' measured timing (channel_lag_tuning.py, make_input_plans.py; plans under --plan-root)
CONFIGS["delay4_wide_shift"] = WIDE4 + ["--input-delays", "0", "20", "40", "60"]          # 0, 80, 160, 240 ms
# learnable delays, all 96 channels, starting from 0/40/100/200 ms, within 0-300 ms (the cohort's window:
# channel_lag_tuning.py on 36 sessions, 50% of a channel's peak information between 40 and 280 ms)
CONFIGS["delay4_wide_learn"] = WIDE4 + LONG + ["--learn-input-delays", "channel", "--input-delay-max", "75"]
CONFIGS["delay4_wide_learn_shared"] = WIDE4 + LONG + ["--learn-input-delays", "shared", "--input-delay-max", "75"]
CONFIGS["plan_prune5"] = WIDE4 + ["--input-plan-dir", "{plan_root}/prune5"]
CONFIGS["plan_tuned"] = WIDE4 + ["--input-plan-dir", "{plan_root}/tuned"]
DEFAULT_CONFIGS = ["per_session", "per_session_narrow", "delay2", "delay4"]
SNN_FLAGS = ["--arch", "snn", "--neuron-type", "iaf", "--reset-type", "hard", "--spike-fn", "multi",
             "--min-vmem", "-1", "--use-iaf-squeeze", "--temporal-decay-stages", "2"]


def load(ckpt_dir):
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model_weights.pth"), map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    module, model = build_model(args, "bmi", ckpt["input_shape"][0], torch.device("cpu"))
    model(torch.zeros(1, 1, ckpt["input_shape"][0]))
    module.load_model_weights(model, ckpt["model_state_dict"], neuron_type=args.neuron_type, source_description=ckpt_dir)
    return model.eval(), args


def speck_check(ckpt_dir):
    """Map onto Speck2f, then the quantized network's test error, emulated on the host."""
    import speck
    model, args = load(ckpt_dir)
    seq = speck.flatten_snn(model).eval()
    n_in = model.layers[0].in_features
    out = {"chip_inputs": n_in}
    try:
        net = speck.discretize(seq, n_in)
        net.make_config(device="speck2fdevkit")
        order = net.chip_layers_ordering
        out["maps_to_speck2f"], out["cores"] = True, [order[k] for k in sorted(order)] if isinstance(order, dict) else list(order)
    except Exception as e:                                    # sinabs raises when a layer fits no core
        out["maps_to_speck2f"], out["map_error"] = False, str(e)[:200]
        return out
    quant = speck.discretized_sequential(net)
    test = CustomDataset(args.data_path, train=False, experiment="bmi", binarize=args.binarize_input)
    scale = (args.velocity_lo, args.velocity_hi, args.velocity_margin)
    errs, chip_events, raw_events = [], 0.0, 0.0
    for i in range(len(test)):
        _, x, v = test[i]
        spikes = x.numpy().T                                   # (C, T), already binarized
        chip_in = speck.delay_input(model, spikes)
        speck.cross_check_flattened(model, seq, spikes)
        counts = speck.run_discretized(quant, chip_in)
        pred = unscale_velocity(speck.decode_spike_counts(model, counts), *scale)
        errs.append(pred - unscale_velocity(v.numpy(), *scale))
        chip_events, raw_events = chip_events + chip_in.sum(), raw_events + spikes.sum()
    err = np.concatenate(errs)
    out["quantized_rmse"] = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    out["quantized_rmse_per_axis"] = [float(np.sqrt((err[:, k] ** 2).mean())) for k in range(2)]
    out["input_events_per_step"] = float(chip_events / len(err))
    out["input_event_factor"] = float(chip_events / max(raw_events, 1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler-patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-data-min", type=float, default=None)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--plan-root", default=os.path.expanduser("~/Downloads/spike_bmi_lsnn_data/input_plans"),
                   help="make_input_plans.py --out-dir, for the plan_* configs")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    out_json = os.path.join(args.out_dir, "speck_delays.json")
    results = json.load(open(out_json)) if os.path.isfile(out_json) else {}   # earlier configs kept
    for name in args.configs:
        ckpt_dir = train(f"snn_{name}", SNN_FLAGS + [f.format(plan_root=args.plan_root) for f in CONFIGS[name]], args)
        results[name], _, _ = evaluate(ckpt_dir, torch.device(args.device))
        results[name]["input_delays_ms"] = [d * STEP_MS for d in (getattr(load(ckpt_dir)[1], "input_delays", None) or [0])]
        results[name]["speck"] = speck_check(ckpt_dir)
        print(f"[{name}] rmse {results[name]['rmse']:.1f}, speck {results[name]['speck']}", flush=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    rows = [("Input delays (ms)", lambda r: ", ".join(f"{d:g}" for d in r["input_delays_ms"])),
            ("Network", lambda r: f"{r['speck']['chip_inputs']} -> " + " -> ".join(map(str, r["hidden_dims"])) + " -> 36"),
            ("Parameters", lambda r: f"{r['parameters']:,}"),
            ("Maps onto Speck2f", lambda r: "yes" if r["speck"]["maps_to_speck2f"] else "no"),
            ("Velocity error (RMS, PyTorch)", lambda r: f"{r['rmse']:.1f}"),
            ("Velocity error (RMS, quantized)", lambda r: f"{r['speck'].get('quantized_rmse', float('nan')):.1f}"),
            ("R² (x, y)", lambda r: f"{r['r2_xy'][0]:.3f}, {r['r2_xy'][1]:.3f}"),
            ("Chip input events per step", lambda r: f"{r['speck'].get('input_events_per_step', float('nan')):.1f}"),
            ("Spikes per neuron per step", lambda r: ", ".join(f"{s:.3f}" for s in r["spikes_per_neuron_per_step"])),
            ("Epochs (best)", lambda r: str(r["epochs_trained"]))]
    names = list(results)
    lines = [f"# Speck SNN with input delays: {os.path.basename(os.path.normpath(args.data_path))}", "",
             f"Test split: {results[names[0]]['test_seconds']:.0f} s, seed {args.seed}.", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {lab} | " + " | ".join(fn(results[n]) for n in names) + " |" for lab, fn in rows]
    with open(os.path.join(args.out_dir, "speck_delays.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
