"""
Train the spiking phoneme decoder with CTC on the speech sessions.

Data: --data-dir holds {session}/{train,test}.npz, exported from the RNN's
TFRecords by speech/analysis/exportTFRecordsToNpz.py (see dataset.py). One
model is trained on every session in --sessions, each session with its own
input layer (as the RNN); every batch holds trials of one session.

Loss: CTC over per-timestep log-softmax outputs (blank = last class, as the
RNN), so no per-bin phoneme labels are needed. Every trial is an
independent example: neuron state is reset at the start of each batch.
Each epoch is evaluated on --val-partition (default 'test', as the RNN's
validation): CTC loss and the greedy-decoding phoneme error rate (PER),
which selects the best model (--selection-metric).

Augmentation, as the RNN (speech_release_baseline.yaml): white noise
(--white-noise-sd) and a per-trial constant offset (--constant-offset-sd).
--smooth-sd applies the RNN's Gaussian input smoothing (smoothKernelSD=2 in
the RNN); it is off by default because the kernel is centred, i.e. it reads
~6 bins (120 ms) into the future, which a causal real-time decoder cannot.

Synaptic time constant: --tau-syn is off by default. With values, every
spiking layer low-pass filters its input current with a trained time
constant (in 20 ms timesteps) initialized to that value, or to its own
value given one per spiking layer (e.g. 2 4 8 16).

Spike-rate penalty (--spike-rate-weight, off by default): adds weight x sum
over spiking layers of max(0, rate - target), the rate in spikes per neuron
per timestep. Training loss only.

Pretrain -> fine-tune (scripts/run_snn_pretrain.sbatch, run_snn_finetune.sbatch):
pretrain with --reset-type hard, then fine-tune from it with
--init-weights-from and --reset-type soft (the reset has no weights, so the
pretrained weights load unchanged). A fine-tuning run keeps the checkpoint's
session list (its input layers); --sessions then picks which of them to
train on.

Outputs in --checkpoint-dir, with exp_name = speech_{neuron_type}_{reset_type}:
    best_model_weights.pth                   best --selection-metric so far
    checkpoint_{exp_name}_epoch{N}.pth       every --checkpoint-interval epochs
    final_model_{exp_name}.pth               written when training ends
    training_curves_{exp_name}.png, summary_{exp_name}.txt

Usage:
    python -m snnDecoder.train --config speech/snnDecoder/configs/speech_pooled.yaml \\
        --data-dir $BASE/derived/snnData --checkpoint-dir $BASE/derived/snns/pretrain --reset-type hard
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sinabs
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .delays import clamp_positions, delay_summary, set_delay_sigma
from .dataset import BASELINE_SESSIONS, SessionBatches, augment, load_partitions, smooth_inputs
from .model import N_CLASSES, create_model
from .utils import (EarlyStopping, edit_distance, greedy_decode, load_checkpoint,
                    plot_training_curves, save_checkpoint)


def str2bool(value):
    return str(value).lower() in ("1", "true", "yes", "y")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config; its values apply unless the flag is given explicitly")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", type=str, default="ff", choices=["ff", "rsnn", "native"],
                       help="ff: feedforward SNN_Speech (model.py); rsnn: recurrent, RNN-shaped (rsnn.py); "
                            "native: spike-native 20 ms recurrent network on raw counts (rsnn_native.py)")
    model.add_argument("--input-channels", type=int, nargs=2, default=None, metavar=("FROM", "TO"),
                       help="native: use channels [FROM, TO) of the data (e.g. 0 128 = tx1 of derived/snnData)")
    model.add_argument("--input-channel-file", type=str, default=None,
                       help="native: read only the channels listed in this file (whitespace-separated indices, "
                            "e.g. from scripts/rank_channels.py); the list is stored with the run's arguments")
    model.add_argument("--day-input", type=str, default="gain", choices=["gain", "linear", "none"],
                       help="native: per-day input stage: per-channel gain (spike-native), dense linear, or none")
    model.add_argument("--feedback", type=str2bool, default=False,
                       help="native: top-down input from the last layer's previous spikes into layer 1")
    model.add_argument("--out-tau-init", type=float, default=2.0,
                       help="native: initial time constant of the leaky output neurons, in 20 ms steps")
    model.add_argument("--out-stages", type=int, default=1,
                       help="native: leaky output stages in a row per class (2 = a smoother, second-order output)")
    model.add_argument("--out-tau-train", type=str2bool, default=True,
                       help="native: train one output time constant per class")
    model.add_argument("--rec-delays", type=int, nargs="+", default=None,
                       help="rsnn/native: per-neuron delays of the recurrent connections, in steps (e.g. 0 4 5); "
                            "each neuron gets one, in the proportions of --rec-delay-fractions")
    model.add_argument("--rec-delay-fractions", type=float, nargs="+", default=None,
                       help="share of neurons with each --rec-delays value (default equal)")
    model.add_argument("--syn-delay-in", type=int, default=0,
                       help="native: heterogeneous synaptic delays on layer 1's input synapses, each drawn from 0..D steps")
    model.add_argument("--syn-delay-rec", type=int, nargs="+", default=[0],
                       help="native: heterogeneous synaptic delays on the recurrent synapses, each drawn from 0..D steps; "
                            "one D for every layer, or one per layer (e.g. growing with depth)")
    model.add_argument("--syn-delay-ff", type=int, default=0,
                       help="native: heterogeneous synaptic delays on the connections between spiking layers, 0..D steps")
    model.add_argument("--adaptive-fraction", type=float, default=0.0,
                       help="native: mixed population (LSNN, Bellec et al. 2018): this fraction of each layer's neurons "
                            "has threshold adaptation (--tau-adapt, --adapt-scale), the rest none; use with "
                            "--neuron-type iaf or lif")
    model.add_argument("--seadlif-tau-u", type=float, nargs=2, default=[1.0, 25.0], metavar=("LO", "HI"),
                       help="native, --neuron-type seadlif: range of the membrane time constants, in steps")
    model.add_argument("--seadlif-tau-w", type=float, nargs=2, default=[1.5, 25.0], metavar=("LO", "HI"),
                       help="native, --neuron-type seadlif: range of the adaptation time constants, in steps")
    model.add_argument("--seadlif-a-max", type=float, default=1.0,
                       help="native, --neuron-type seadlif: largest subthreshold coupling a (adaptation from the membrane)")
    model.add_argument("--seadlif-b-max", type=float, default=1.0,
                       help="native, --neuron-type seadlif: largest spike-triggered adaptation b")
    model.add_argument("--seadlif-lr-mult", type=float, default=0.1,
                       help="native, --neuron-type seadlif: learning-rate multiple of its neuron parameters")
    model.add_argument("--branches", type=int, default=1,
                       help="native: K feedforward chains side by side (block-diagonal layers), joined at the readout")
    model.add_argument("--branch-split", type=str, default="delay", choices=["delay", "level", "channel"],
                       help="native, --branches > 1: how layer 1's input lines are shared out between the chains")
    model.add_argument("--branch-spec", type=str, default=None,
                       help="native: JSON file of chains of unequal size, a list of {channels, delays, dims} "
                            "(rsnn_native branch_groups); overrides --branches")
    model.add_argument("--readout-all", type=str2bool, default=False,
                       help="native: the output neurons read every spiking layer, not only the last (Bellec et al. 2020)")
    model.add_argument("--learn-delays", type=str2bool, default=False,
                       help="native: the per-synapse delays (--syn-delay-in/-rec/-ff) are learned, starting from the "
                            "fixed random ones (Hammouamri et al., ICLR 2024; snnDecoder/delays.py)")
    model.add_argument("--delay-sigma-frac", type=float, default=0.25,
                       help="learned delays: initial Gaussian width, as a fraction of each group's max delay (min 1 step)")
    model.add_argument("--delay-hard-from", type=float, default=0.75,
                       help="learned delays: fraction of the epochs after which delays are rounded and frozen")
    model.add_argument("--delay-lr-mult", type=float, default=1.0,
                       help="learned delays: learning-rate multiple for the delay positions (no weight decay)")
    model.add_argument("--syn-delay-in-mult", type=int, default=1,
                       help="native: input synapses per input-neuron pair, each with its own weight and random delay "
                            "(needs --syn-delay-in)")
    model.add_argument("--rec-no-self", type=str2bool, default=False,
                       help="rsnn/native: no recurrent self-connections (diagonal held at 0)")
    model.add_argument("--output-stride", type=int, default=1,
                       help="native: one output per this many 20 ms steps (4 = the RNN's 80 ms CTC frames)")
    model.add_argument("--tbptt", type=int, default=0,
                       help="native: truncated backpropagation through time, cutting gradients every K steps (0 = off)")
    model.add_argument("--tau-syn-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                       help="rsnn/native: per-neuron trainable synaptic time constants, log-spaced over [LO, HI] steps")
    model.add_argument("--input-window", type=int, default=32,
                       help="rsnn: bins stacked per step (the RNN's kernel, 32 = 640 ms)")
    model.add_argument("--input-stride", type=int, default=4,
                       help="rsnn: bins between steps (the RNN's stride, 4 = 80 ms)")
    model.add_argument("--input-layer", type=str, default="linear", choices=["linear", "gain"],
                       help="rsnn: per-session input layer, the RNN's Linear + softsign, or one gain per channel")
    model.add_argument("--input-encoding", type=str, default="none",
                       choices=["none", "ternary", "delta", "sigma_delta", "binary", "levels"],
                       help="rsnn: turn each bin's features into signed event counts (frontend.py), so the first "
                            "layer adds a weight per event instead of multiplying every input")
    model.add_argument("--encoding-threshold", type=float, default=0.25,
                       help="rsnn: feature change (delta) or value (ternary, sigma_delta) per event")
    model.add_argument("--encoding-max", type=int, default=3, help="rsnn: most events per channel per bin")
    model.add_argument("--encoding-levels", type=float, nargs="+", default=None,
                       help="native, --input-encoding levels: one on/off input line per channel and level, on when "
                            "the feature is at or above it (a thermometer code; the chip's input lines multiply)")
    model.add_argument("--input-delays", type=int, nargs="+", default=None,
                       help="rsnn: synaptic delays of the first layer, in frames (each with its own weights)")
    model.add_argument("--input-traces", type=float, nargs="+", default=None,
                       help="rsnn: time constants (frames; 0 = none) of the first layer's synaptic currents, "
                            "one weight set each")
    model.add_argument("--recurrent", type=str2bool, default=True,
                       help="rsnn: recurrent connections within each layer")
    model.add_argument("--rec-init-scale", type=float, default=1.0,
                       help="rsnn: scale of the recurrent weights' initialization (< 1: weaker initial feedback)")
    model.add_argument("--rec-init", type=str, default="default", choices=["default", "orthogonal"],
                       help="rsnn: recurrent weight init (orthogonal, as the RNN's GRU)")
    model.add_argument("--learn-threshold", type=str, default="none", choices=["none", "layer", "neuron"],
                       help="rsnn: train the base spike threshold, one per layer or one per neuron "
                            "(starting at --spike-thresholds)")
    model.add_argument("--threshold-lr-mult", type=float, default=1.0,
                       help="learning rate of learned thresholds (--learn-threshold) as a multiple of --lr")
    model.add_argument("--substeps", type=int, default=1,
                       help="rsnn: spiking steps per output frame (the frame's input held throughout; "
                            "time constants keep their length in real time)")
    model.add_argument("--input-substeps", type=int, default=1,
                       help="native chip network: K chip steps per 20 ms bin, each bin's count spread over them "
                            "as single events (at most K); delays and time constants stay in bins")
    model.add_argument("--readout-traces", type=float, nargs="+", default=None,
                       help="native: the readout also reads leaky traces of the spikes, one per time constant "
                            "(20 ms steps; 0 = the spikes themselves)")
    model.add_argument("--readout-source", type=str, default="spikes", choices=["spikes", "vmem"],
                       help="rsnn: the readout reads the last layer's spikes, or its membrane potential")
    model.add_argument("--tau-mem-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                       help="rsnn: per-neuron trainable membrane time constants, log-spaced over [LO, HI] "
                            "frames (gives every neuron a leak)")
    model.add_argument("--tau-adapt-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                       help="rsnn ALIF: per-neuron trainable adaptation time constants over [LO, HI] frames")
    model.add_argument("--tau-lr-mult", type=float, default=1.0,
                       help="learning rate of the trained per-neuron time constants as a multiple of --lr")
    model.add_argument("--surrogate-width", type=float, default=None,
                       help="grad_width of the sinabs surrogate gradient (default: sinabs' own, 0.5)")
    model.add_argument("--layer-dropout", type=float, default=0.4,
                       help="rsnn: dropout on each spiking layer's input (the RNN's 0.4)")
    model.add_argument("--neuron-type", type=str, default="iaf", choices=["iaf", "lif", "alif", "seadlif"])
    model.add_argument("--reset-type", type=str, default="hard", choices=["hard", "soft"])
    model.add_argument("--final-layer-reset-type", type=str, default=None, choices=["hard", "soft"],
                       help="Reset for the last spiking layer (default: same as --reset-type)")
    model.add_argument("--tau-mem", type=float, default=12.0, help="LIF membrane time constant")
    model.add_argument("--norm-input", type=str2bool, default=True, help="LIF/ALIF only")
    model.add_argument("--tau-adapt", type=float, default=25.0,
                       help="ALIF: time constant of the threshold adaptation, in timesteps")
    model.add_argument("--adapt-scale", type=float, default=1.8,
                       help="ALIF: threshold rise per unit of adaptation (threshold = b0 + scale * b)")
    model.add_argument("--tau-syn", type=float, nargs="+", default=None,
                       help="Initial synaptic time constant, trained per layer: one value for "
                            "every spiking layer or one per layer (default: none)")
    model.add_argument("--norm-syn", type=str2bool, default=True,
                       help="Scale each layer's input so its synaptic stage has unit gain, "
                            "whatever its tau_syn (see model.py); false = sinabs' behaviour")
    model.add_argument("--tau-syn-spread", type=float, default=1.0,
                       help="> 1: per-neuron tau_syn, log-spaced over [tau/F, tau*F] around each layer's tau_syn")
    model.add_argument("--hidden-dims", type=int, nargs="+", default=None,
                       help="Hidden spiking layer widths (default: 512 256 128)")
    model.add_argument("--readout-width", type=int, default=128,
                       help="Width of the last spiking layer, read out by the EMA + linear readout")
    model.add_argument("--spike-thresholds", type=float, nargs="+", default=None,
                       help="One threshold per spiking layer: len(hidden_dims) + 1")
    model.add_argument("--spike-fn", type=str, default="multi", choices=["multi", "single", "max"],
                       help="multi: any number of spikes per step (sinabs MultiSpike); single: at most one "
                            "(SingleSpike); max: at most --max-spikes (MaxSpike)")
    model.add_argument("--max-spikes", type=int, default=2, help="--spike-fn max: spikes per neuron per step")
    model.add_argument("--min-vmem", type=float, default=None, help="Lower bound on membrane potential")
    model.add_argument("--max-vmem", type=float, default=None, help="native: upper bound on membrane potential")
    model.add_argument("--weight-init", type=str, default="kaiming", choices=["kaiming", "xavier"])
    model.add_argument("--surrogate-grad", type=str, default="periodic_exponential",
                       choices=["periodic_exponential", "single_exponential", "gaussian",
                                "multi_gaussian", "heaviside", "ramp"],
                       help="ramp: the wide surrogate of Balafrej et al. (arXiv 2310.19067), 1 from one threshold "
                            "below firing upwards")
    model.add_argument("--last-layer-reset", action="store_true",
                       help="Reset the last spiking layer's membrane before each timestep")
    model.add_argument("--temporal-decay-init", type=float, default=0.8,
                       help="Initial decay of the readout's EMA smoothing")
    model.add_argument("--learnable-temporal-decay", type=str2bool, default=True)
    model.add_argument("--temporal-decay-stages", type=int, default=1,
                       help="Number of cascaded EMA stages in the readout")
    model.add_argument("--input-dropout", type=float, default=0.2,
                       help="Dropout after the per-session input layer (the RNN's inputNetwork uses 0.2)")
    model.add_argument("--use-exodus", type=str2bool, default=False,
                       help="Use sinabs-exodus CUDA kernels (forces soft reset)")

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", type=str, required=True,
                      help="Directory of {session}/{partition}.npz (exportTFRecordsToNpz.py)")
    data.add_argument("--sessions", type=str, nargs="+", default=None,
                      help="Sessions to train on (default: the RNN baseline's 19 sessions)")
    data.add_argument("--val-partition", type=str, default="test")
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--white-noise-sd", type=float, default=1.0)
    data.add_argument("--constant-offset-sd", type=float, default=0.2)
    data.add_argument("--smooth-sd", type=float, default=0.0,
                      help="RNN-style Gaussian input smoothing in bins (non-causal; 0 = off)")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--adam-eps", type=float, default=0.1,
                       help="--recipe rnn: Adam's epsilon (the paper RNN's 0.1; where a weight's gradient RMS is well "
                            "below it, its updates are plain SGD at lr / eps; scripts/adam_scale.py measures this)")
    train.add_argument("--weight-decay", type=float, default=0.001)
    train.add_argument("--recipe", type=str, default="default", choices=["default", "rnn"],
                       help="default: AdamW + ReduceLROnPlateau on the selection metric; rnn: the paper "
                            "RNN's recipe -- Adam with epsilon 0.1 and L2 (--weight-decay) added to the "
                            "gradient, learning rate decayed linearly to 0 over all batches of --epochs "
                            "(use with e.g. --lr 0.02 --grad-clip 10)")
    train.add_argument("--teacher-dir", type=str, default=None,
                       help="distill from a trained RNN's per-frame logits (speech/analysis/exportTeacherLogits.py); "
                            "needs a student with the RNN's frames (--arch rsnn, window 32, stride 4)")
    train.add_argument("--distill-weight", type=float, default=0.0,
                       help="weight of the distillation loss (KL from the teacher's to the student's softmax at "
                            "--distill-temp, x temp^2, summed over frames and averaged over trials like CTC)")
    train.add_argument("--distill-temp", type=float, default=2.0)
    train.add_argument("--teacher-offset", type=int, default=0,
                       help="student frame j learns from teacher frame j - offset: 7 for a native student (output at "
                            "the end of each 4-bin frame) and the RNN teacher (32-bin windows, stride 4), so both "
                            "frames end on the same bin")
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--input-event-weight", type=float, default=0.0,
                       help="rsnn with --input-encoding: penalty weight on input events per channel per bin "
                            "above --input-event-target")
    train.add_argument("--input-event-target", type=float, default=0.1)
    train.add_argument("--branching-weight", type=float, default=0.0,
                       help="weight of the branching-factor loss (Balafrej et al., arXiv 2310.19067): per layer, the "
                            "squared change of mean spikes per neuron from one step to the next, summed over each "
                            "sentence's steps and averaged over the batch; keeps activity steady instead of low")
    train.add_argument("--spike-rate-weight", type=float, default=0.0,
                       help="Spike-rate penalty (0 = off), training loss only")
    train.add_argument("--spike-rate-target", type=float, nargs="+", default=[0.1],
                       help="Rate each spiking layer may reach before the penalty applies: one "
                            "value for every layer or one per layer")
    train.add_argument("--selection-metric", type=str, default="per", choices=["per", "loss"],
                       help="Validation metric for the best model, LR schedule and early stopping")
    train.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    train.add_argument("--scheduler-patience", type=int, default=10,
                       help="Epochs without improvement before the LR is reduced")
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument("--lr-factor", type=float, default=0.1)
    train.add_argument("--max-lr-reductions", type=int, default=3,
                       help="With --disable-early-stopping: stop after this many LR reductions")
    train.add_argument("--disable-early-stopping", action="store_true")

    ckpt = parser.add_argument_group("checkpoints")
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ckpt.add_argument("--checkpoint-interval", type=int, default=10)
    ckpt.add_argument("--resume", type=str, default=None,
                      help="Continue an interrupted run (restores optimizer and epoch)")
    ckpt.add_argument("--init-weights-from", type=str, default=None,
                      help="Load only model weights from this checkpoint (fine-tuning); the "
                           "architecture flags must match it")

    log = parser.add_argument_group("logging")
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="speech-snn")
    log.add_argument("--wandb-entity", type=str, default=None)
    log.add_argument("--wandb-name", type=str, default=None)
    log.add_argument("--wandb-mode", type=str, default="online",
                     choices=["online", "offline", "disabled"])
    log.add_argument("--device", type=str, default="cuda")
    log.add_argument("--seed", type=int, default=42)
    log.add_argument("--verbose", action="store_true")
    log.add_argument("--log-interval", type=int, default=50, help="Batches between log lines")
    args = parser.parse_args(argv)
    return merge_config_args(args, sys.argv[1:] if argv is None else argv)


def merge_config_args(args, argv):
    """Apply --config values to args, except for flags given explicitly on
    the command line. Config sections are flattened: {training: {epochs: 5}}
    sets args.epochs; keys with no matching argument are ignored."""
    if not args.config:
        return args
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    explicit = {a[2:].split("=")[0].replace("-", "_") for a in argv if a.startswith("--")}
    items = []
    for key, value in config.items():
        items.extend(value.items() if isinstance(value, dict) else [(key, value)])
    for key, value in items:
        attr = key.replace("-", "_")
        if attr not in explicit and hasattr(args, attr):
            setattr(args, attr, value)
    return args


def model_kwargs(args):
    """create_model() arguments from a run's (saved) args dict."""
    tau_syn = args.get("tau_syn")
    return dict(
        weight_init=args.get("weight_init", "kaiming"),
        num_input_channels=args.get("num_input_channels", 256),
        n_classes=args.get("n_classes", N_CLASSES),
        hidden_dims=args.get("hidden_dims"),
        readout_width=args.get("readout_width", 128),
        neuron_type=args.get("neuron_type", "iaf"),
        tau_mem=args.get("tau_mem", 12.0),
        norm_input=args.get("norm_input", True),
        reset_type=args.get("reset_type", "hard"),
        final_layer_reset_type=args.get("final_layer_reset_type"),
        surrogate_grad=({"type": args.get("surrogate_grad", "periodic_exponential"),
                         "grad_width": args["surrogate_width"]} if args.get("surrogate_width")
                        else args.get("surrogate_grad", "periodic_exponential")),
        use_exodus=args.get("use_exodus", False),
        min_vmem=args.get("min_vmem"),
        spike_fn=(sinabs.activation.MultiSpike if args.get("spike_fn", "multi") == "multi"
                  else sinabs.activation.SingleSpike if args.get("spike_fn") == "single"
                  else sinabs.activation.MaxSpike(max_num_spikes_per_bin=int(args.get("max_spikes", 2)))),
        spike_thresholds=args.get("spike_thresholds"),
        tau_syn=tau_syn[0] if tau_syn and len(tau_syn) == 1 else tau_syn,
        last_layer_reset=args.get("last_layer_reset", False),
        temporal_decay_init=args.get("temporal_decay_init", 0.8),
        learnable_temporal_decay=args.get("learnable_temporal_decay", True),
        temporal_decay_stages=args.get("temporal_decay_stages", 1),
        input_dropout=args.get("input_dropout", 0.2),
        tau_adapt=args.get("tau_adapt", 25.0),
        tau_syn_spread=args.get("tau_syn_spread", 1.0),
        arch=args.get("arch", "ff"),
        input_window=args.get("input_window", 32),
        input_stride=args.get("input_stride", 4),
        recurrent=args.get("recurrent", True),
        layer_dropout=args.get("layer_dropout", 0.4),
        rec_init_scale=args.get("rec_init_scale", 1.0),
        rec_init=args.get("rec_init", "default"),
        input_channels=args.get("input_channels"),
        input_channel_list=args.get("input_channel_list"),
        day_input=args.get("day_input", "gain"),
        feedback=args.get("feedback", False),
        out_tau_init=args.get("out_tau_init", 2.0),
        out_tau_train=args.get("out_tau_train", True),
        tau_syn_range=args.get("tau_syn_range"),
        tbptt=args.get("tbptt", 0),
        max_vmem=args.get("max_vmem"),
        learn_threshold=args.get("learn_threshold", "none"),
        substeps=args.get("substeps", 1),
        readout_source=args.get("readout_source", "spikes"),
        tau_mem_range=args.get("tau_mem_range"),
        tau_adapt_range=args.get("tau_adapt_range"),
        adapt_scale=args.get("adapt_scale", 1.8),
        input_layer=args.get("input_layer", "linear"),
        input_encoding=args.get("input_encoding", "none"),
        encoding_threshold=args.get("encoding_threshold", 0.25),
        encoding_max=args.get("encoding_max", 3),
        **({"encoding_levels": args["encoding_levels"]} if args.get("encoding_levels") else {}),
        input_delays=args.get("input_delays"),
        input_traces=args.get("input_traces"),
        output_stride=args.get("output_stride", 1),
        rec_delays=args.get("rec_delays"),
        rec_delay_fractions=args.get("rec_delay_fractions"),
        rec_no_self=args.get("rec_no_self", False),
        syn_delay_in=args.get("syn_delay_in", 0),
        syn_delay_rec=args.get("syn_delay_rec", 0),
        syn_delay_ff=args.get("syn_delay_ff", 0),
        syn_delay_in_mult=args.get("syn_delay_in_mult", 1),
        learn_delays=args.get("learn_delays", False),
        readout_all=args.get("readout_all", False),
        readout_traces=args.get("readout_traces"),
        input_substeps=args.get("input_substeps", 1),
        branches=args.get("branches", 1) or 1,
        branch_split=args.get("branch_split", "delay"),
        branch_groups=args.get("branch_groups"),
        adaptive_fraction=args.get("adaptive_fraction", 0.0),
        seadlif=dict(tau_u=args.get("seadlif_tau_u", [1.0, 25.0]), tau_w=args.get("seadlif_tau_w", [1.5, 25.0]),
                     a_max=args.get("seadlif_a_max", 1.0), b_max=args.get("seadlif_b_max", 1.0)),
        out_stages=args.get("out_stages", 1),
        # Checkpoints from before norm_syn existed were trained without it.
        norm_syn=args.get("norm_syn", False),
    )


def spike_rate_penalty(model, n_steps, targets):
    """(sum over spiking layers of max(0, rate - target), [rate per layer]),
    rate = the layer's spikes per neuron per valid timestep in the last
    forward pass (n_steps = valid timesteps summed over the batch)."""
    rates = [spikes / (n_steps * width) for spikes, width in zip(model.layer_spikes, model.layer_widths)]
    penalty = sum(torch.relu(rate - target) for rate, target in zip(rates, targets))
    return penalty, [float(rate.detach()) for rate in rates]


def branching_loss(model):
    """Sum over spiking layers of: per sentence, sum over valid steps t >= 1 of (r[t] - r[t-1])^2, r = the
    layer's mean spikes per neuron at step t; averaged over the batch."""
    m = model.step_mask
    both = m[:, 1:] * m[:, :-1]
    total = 0.0
    for r in model.layer_step_rates:
        total = total + (((r[:, 1:] - r[:, :-1]) ** 2) * both).sum(1).mean()
    return total


def ctc_loss(logits, batch, blank, output_lengths=None):
    """Mean over the batch of each sequence's CTC loss (as the RNN).
    output_lengths: valid logit frames per trial (default: batch['lengths'])."""
    log_probs = logits.log_softmax(-1).transpose(0, 1)      # (T, N, K)
    lengths = batch['lengths'] if output_lengths is None else output_lengths
    losses = F.ctc_loss(log_probs, batch['targets'], lengths, batch['target_lengths'],
                        blank=blank, reduction='none', zero_infinity=True)
    return losses.mean()


def distill_loss(logits, teacher, lengths, temp, offset=0):
    """KL(teacher || student) of the softmax at temperature temp, x temp^2, summed over each trial's valid frames
    and averaged over the batch. offset: student frame j against teacher frame j - offset (lengths: the student's)."""
    if offset:
        logits, lengths = logits[:, offset:], (lengths - offset).clamp(min=0)
    T = min(logits.shape[1], teacher.shape[1])
    mask = (torch.arange(T, device=logits.device)[None, :] < lengths[:, None]).float()
    log_s = F.log_softmax(logits[:, :T] / temp, -1)
    log_t = F.log_softmax(teacher[:, :T] / temp, -1)
    kl = (log_t.exp() * (log_t - log_s)).sum(-1)
    return (kl * mask).sum(1).mean() * temp ** 2


def run_epoch(model, batches, device, args, optimizer=None, spike_rate_targets=None, desc=""):
    """One pass over `batches`; trains if an optimizer is given, else
    evaluates (CTC loss + greedy PER). Returns a metrics dict."""
    training = optimizer is not None
    model.train(training)
    blank = model.n_classes - 1
    sums = {"loss": 0.0, "ctc_loss": 0.0, "spike_rate_penalty": 0.0, "distill_loss": 0.0}
    layer_spikes, n_steps, n_neurons_steps, total_spikes = None, 0, 0, 0.0
    edit, n_phones = 0, 0
    skipped = 0

    for batch_idx, batch in enumerate(batches):
        for k in ("x", "lengths", "targets", "target_lengths", "teacher", "teacher_lengths"):
            if k in batch:
                batch[k] = batch[k].to(device)
        x = batch["x"]
        if training:
            x = augment(x, args.white_noise_sd, args.constant_offset_sd)
        x = smooth_inputs(x, args.smooth_sd)
        out_lengths = model.output_lengths(batch["lengths"])     # logit frames (stacking: fewer)
        steps = int(out_lengths.sum())

        with torch.set_grad_enabled(training):
            logits = model(x, batch["day_idx"], lengths=batch["lengths"])
            loss = pred_loss = ctc_loss(logits, batch, blank, out_lengths)
            rate_penalty, _ = spike_rate_penalty(model, steps, spike_rate_targets or [0.0])
            if training and args.spike_rate_weight > 0:
                loss = loss + args.spike_rate_weight * rate_penalty
            if training and getattr(args, "branching_weight", 0.0) > 0:
                branching = branching_loss(model)
                loss = loss + args.branching_weight * branching
                sums["branching_loss"] = sums.get("branching_loss", 0.0) + float(branching.detach())
            event_rate = None
            if getattr(getattr(model, "encoder", None), "active", False):
                event_rate = model.input_events / max(model.input_event_slots, 1.0)
                if training and getattr(args, "input_event_weight", 0.0) > 0:
                    # events per channel per bin, differentiable through the straight-through encoder
                    soft = model.input_events_soft / max(model.input_event_slots, 1.0)
                    loss = loss + args.input_event_weight * torch.relu(soft - args.input_event_target)
            if training and args.distill_weight > 0 and "teacher" in batch:
                off = getattr(args, "teacher_offset", 0)
                kd = distill_loss(logits, batch["teacher"],
                                  torch.minimum(out_lengths, batch["teacher_lengths"] + off), args.distill_temp, off)
                loss = loss + args.distill_weight * kd
                sums["distill_loss"] += kd.item()

        if training:
            optimizer.zero_grad()
            loss.backward()
            # as the paper RNN's training: an update whose gradients are not all finite (an overflow
            # through time) is skipped rather than applied
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                clamp_positions(model)
            else:
                optimizer.zero_grad()
                skipped += 1
            if args.recipe == "rnn":
                # linear decay to 0 over all batches of the run, per batch (as the RNN's PolynomialDecay)
                args.global_step += 1
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * group.get("lr_mult", 1.0) * max(0.0, 1.0 - args.global_step / args.total_steps)

        sums["loss"] += loss.item()
        sums["ctc_loss"] += pred_loss.item()
        sums["spike_rate_penalty"] += float(rate_penalty)
        spikes = [float(s.detach()) for s in model.layer_spikes]
        layer_spikes = spikes if layer_spikes is None else [a + b for a, b in zip(layer_spikes, spikes)]
        total_spikes += float(model.total_spikes.detach())
        n_steps += steps
        if event_rate is not None:
            sums["input_events"] = sums.get("input_events", 0.0) + float(model.input_events)
            sums["input_event_slots"] = sums.get("input_event_slots", 0.0) + model.input_event_slots

        if not training:
            logits_np = logits.detach().cpu().numpy()
            targets = np.split(batch["targets"].cpu().numpy(), np.cumsum(batch["target_lengths"].cpu().numpy())[:-1])
            for b, (length, target) in enumerate(zip(out_lengths.cpu().numpy(), targets)):
                decoded = greedy_decode(logits_np[b, :length], blank)
                edit += edit_distance(decoded, target)
                n_phones += len(target)
        if training and args.verbose and (batch_idx + 1) % args.log_interval == 0:
            print(f"  {desc} [{batch_idx + 1}/{len(batches)}] loss={loss.item():.4f} "
                  f"spikes/neuron/step={total_spikes / (n_steps * model.total_neuron_units):.4f}")

    events = sums.pop("input_events", None)
    slots = sums.pop("input_event_slots", None)
    metrics = {k: v / len(batches) for k, v in sums.items()}
    if events is not None:
        metrics["input_event_rate"] = events / max(slots, 1.0)
    metrics["skipped_updates"] = skipped
    metrics["layer_rates"] = [s / (n_steps * w) for s, w in zip(layer_spikes, model.layer_widths)]
    metrics["spike_rate"] = total_spikes / (n_steps * model.total_neuron_units)
    if not training:
        metrics["per"] = edit / max(n_phones, 1)
    else:
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics


def main(argv=None):
    args = parse_arguments(argv)
    if args.resume and args.init_weights_from:
        raise ValueError("--resume and --init-weights-from are mutually exclusive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Session list = the model's input layers. A fine-tuning run keeps the
    # pretrained checkpoint's list; --sessions picks which of them to train on.
    source = None
    if args.init_weights_from or args.resume:
        source = torch.load(args.init_weights_from or args.resume, map_location="cpu", weights_only=False)
        model_sessions = list(source["sessions"])
    else:
        model_sessions = list(args.sessions or BASELINE_SESSIONS)
    train_sessions = list(args.sessions or model_sessions)
    unknown = [s for s in train_sessions if s not in model_sessions]
    if unknown:
        raise ValueError(f"Sessions {unknown} have no input layer in {args.init_weights_from or args.resume} "
                         f"(it was trained on {model_sessions})")
    day_indices = [model_sessions.index(s) for s in train_sessions]
    args.model_sessions = model_sessions

    exp_name = f"speech_{'rsnn_' if args.arch == 'rsnn' else ''}{args.neuron_type}_{args.reset_type}"
    if args.neuron_type in ("lif", "alif"):
        exp_name += f"_tau{args.tau_mem}"
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_name or exp_name, config=vars(args), mode=args.wandb_mode)

    print(f"Experiment: {exp_name} | device: {device}")
    print(f"Training on {len(train_sessions)} of the model's {len(model_sessions)} sessions")
    if args.arch == "rsnn":
        print(f"Recurrent SNN: {'recurrent' if args.recurrent else 'feedforward'} layers {args.hidden_dims} | "
              f"input window {args.input_window} bins, stride {args.input_stride} | layer dropout {args.layer_dropout}")
    print(f"Hidden dims: {args.hidden_dims} + readout {args.readout_width} | thresholds: "
          f"{args.spike_thresholds} | tau_syn: {args.tau_syn}"
          f"{f' (per-neuron spread x{args.tau_syn_spread:g})' if args.tau_syn_spread > 1 else ''} | decay stages: {args.temporal_decay_stages}")
    print(f"Batch size: {args.batch_size} | lr: {args.lr} | epochs: {args.epochs} | augmentation: "
          f"noise {args.white_noise_sd}, offset {args.constant_offset_sd} | smoothing sd {args.smooth_sd}")

    train_parts = load_partitions(args.data_dir, train_sessions, "train", day_indices,
                                  teacher_dir=args.teacher_dir if args.distill_weight > 0 else None)
    if args.distill_weight > 0:
        print(f"Distilling from {args.teacher_dir}: weight {args.distill_weight}, temperature {args.distill_temp}")
    val_parts = load_partitions(args.data_dir, train_sessions, args.val_partition, day_indices)
    train_batches = SessionBatches(train_parts, args.batch_size, shuffle=True, seed=args.seed)
    val_batches = SessionBatches(val_parts, args.batch_size, shuffle=False)
    args.num_input_channels = int(train_parts[0].features[0].shape[1])
    if args.input_channel_file and not getattr(args, "input_channel_list", None):
        with open(args.input_channel_file) as f:
            args.input_channel_list = [int(c) for c in f.read().split()]
        print(f"Input channels: {len(args.input_channel_list)} from {args.input_channel_file}")
    if getattr(args, "branch_spec", None) and not getattr(args, "branch_groups", None):
        with open(args.branch_spec) as f:
            args.branch_groups = json.load(f)
        print(f"Branch groups from {args.branch_spec}: " + "; ".join(
            f"{len(g['channels'])} features x delays {g['delays']} -> {g['dims']}" for g in args.branch_groups))
    print(f"Data: {train_batches.n_trials} train trials, {val_batches.n_trials} "
          f"{args.val_partition} trials")

    model = create_model(n_sessions=len(model_sessions), **model_kwargs(vars(args))).to(device)
    if args.init_weights_from:
        from .model import load_model_weights
        load_model_weights(model, source["model_state_dict"], source_description=args.init_weights_from)
        print(f"Initialized weights from {args.init_weights_from}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.verbose:
        print(model)

    # learned thresholds (rsnn --learn-threshold) may learn at their own multiple of the learning rate
    thr_params = [p for n, p in model.named_parameters() if n.endswith("log_threshold")]
    tau_params = [p for n, p in model.named_parameters() if ".log_tau_" in n]
    delay_params = [p for n, p in model.named_parameters() if n.endswith("_delay.pos")]
    sead_params = [p for n, p in model.named_parameters() if ".sead_" in n]
    other_params = [p for n, p in model.named_parameters()
                    if not n.endswith("log_threshold") and ".log_tau_" not in n and not n.endswith("_delay.pos")
                    and ".sead_" not in n]
    param_groups = [{"params": other_params, "lr_mult": 1.0}]
    if thr_params:
        param_groups.append({"params": thr_params, "lr_mult": args.threshold_lr_mult,
                             "lr": args.lr * args.threshold_lr_mult})
    if tau_params:
        param_groups.append({"params": tau_params, "lr_mult": args.tau_lr_mult, "lr": args.lr * args.tau_lr_mult})
    if delay_params:
        # their own Adam epsilon: delay gradients are ~500x smaller than weight gradients, and the RNN recipe's
        # eps 0.1 would scale their updates down ~1e5-fold (the first learned-delay run, R_learn, never moved)
        param_groups.append({"params": delay_params, "lr_mult": args.delay_lr_mult,
                             "lr": args.lr * args.delay_lr_mult, "weight_decay": 0.0, "eps": 1e-8})
    if sead_params:
        # SE-adLIF time constants and couplings (values in [0, 1]): their own Adam epsilon for the same reason as
        # the delays, and no weight decay, which would pull them toward the bottom of their ranges
        param_groups.append({"params": sead_params, "lr_mult": args.seadlif_lr_mult,
                             "lr": args.lr * args.seadlif_lr_mult, "weight_decay": 0.0, "eps": 1e-8})
    if args.recipe == "rnn":
        # the paper RNN (NeuralSequenceDecoder): Adam, epsilon 0.1, L2 in the loss (= coupled weight decay)
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=args.adam_eps,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                  patience=args.scheduler_patience, min_lr=args.min_lr)
    early_stopping = None if args.disable_early_stopping else EarlyStopping(patience=args.patience)

    start_epoch, best_score, lr_reduction_count = 0, float("inf"), 0
    history = {"train_loss": [], "test_loss": [], "test_per": [], "lr": []}
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", float("inf"))
        history = checkpoint.get("history", history)
        lr_reduction_count = checkpoint.get("lr_reduction_count", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        if start_epoch >= args.epochs:
            raise ValueError(f"Checkpoint is already at epoch {start_epoch} >= --epochs "
                             f"{args.epochs}; raise --epochs to continue training.")

    def checkpoint_info(**extra):
        return {"best_score": best_score, "selection_metric": args.selection_metric,
                "history": history, "lr_reduction_count": lr_reduction_count,
                "args": vars(args), "sessions": model_sessions, **extra}

    n_spiking = len(model.layer_widths)
    if len(args.spike_rate_target) not in (1, n_spiking):
        raise ValueError(f"--spike-rate-target takes 1 or {n_spiking} values (one per spiking "
                         f"layer), got {len(args.spike_rate_target)}")
    spike_rate_targets = (args.spike_rate_target * n_spiking if len(args.spike_rate_target) == 1
                          else list(args.spike_rate_target))

    args.total_steps = max(1, args.epochs * len(train_batches))
    args.global_step = start_epoch * len(train_batches)
    if args.recipe == "rnn":
        for group in optimizer.param_groups:
            group["lr"] = args.lr * group.get("lr_mult", 1.0) * max(0.0, 1.0 - args.global_step / args.total_steps)
        print(f"RNN recipe: Adam (eps {args.adam_eps}, L2 {args.weight_decay}), lr {args.lr} -> 0 linearly over "
              f"{args.total_steps} batches, grad clip {args.grad_clip}")
    training_start = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        delays_hard = True
        if delay_params:
            delays_hard = set_delay_sigma(model, epoch, args.epochs, args.delay_sigma_frac, args.delay_hard_from)
        train_m = run_epoch(model, train_batches, device, args, optimizer=optimizer,
                            spike_rate_targets=spike_rate_targets, desc=f"Epoch {epoch + 1}")
        with torch.no_grad():
            val_m = run_epoch(model, val_batches, device, args, spike_rate_targets=spike_rate_targets)
        history["train_loss"].append(train_m["loss"])
        history["test_loss"].append(val_m["ctc_loss"])
        history["test_per"].append(val_m["per"])
        history.setdefault("test_spike_rate", []).append(val_m["spike_rate"])
        score = val_m["per"] if args.selection_metric == "per" else val_m["ctc_loss"]

        old_lr = optimizer.param_groups[0]["lr"]
        if args.recipe != "rnn":
            scheduler.step(score)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            lr_reduction_count += 1
            print(f"Learning rate reduced {old_lr:.6f} -> {new_lr:.6f} (#{lr_reduction_count})")
        history["lr"].append(new_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | train loss {train_m['loss']:.4f} | "
              f"{args.val_partition} loss {val_m['ctc_loss']:.4f} | {args.val_partition} PER "
              f"{val_m['per']:.4f} | spike rate {val_m['spike_rate']:.4f} | lr {train_m['learning_rate']:.6f}")
        print(f"  spikes per neuron per timestep by layer: {args.val_partition} "
              f"[{' '.join(f'{r:.3f}' for r in val_m['layer_rates'])}], train "
              f"[{' '.join(f'{r:.3f}' for r in train_m['layer_rates'])}]")
        if delay_params:
            print(f"  learned delays: {delay_summary(model)}")
        if "branching_loss" in train_m:
            print(f"  branching-factor loss (train, per batch): {train_m['branching_loss']:.4f}")
        if "input_event_rate" in val_m:
            print(f"  input events per channel per 20 ms bin: {args.val_partition} {val_m['input_event_rate']:.4f}, "
                  f"train {train_m['input_event_rate']:.4f}")
        if getattr(model, "tau_summary", None) and model.tau_summary():
            print(f"  trained time constants (frames), mean [min, max]: {model.tau_summary()}")
        if getattr(model, "threshold_summary", None) and model.threshold_summary():
            print(f"  learned thresholds by layer, mean [min, max]: {model.threshold_summary()}")
        if train_m.get("skipped_updates"):
            print(f"  skipped {train_m['skipped_updates']} of {len(train_batches)} updates (non-finite gradients)")
        if args.distill_weight > 0:
            print(f"  train CTC loss {train_m['ctc_loss']:.4f} | distillation loss {train_m['distill_loss']:.4f}")
        if args.wandb:
            wandb.log({"epoch": epoch + 1, "epoch_time": epoch_time,
                       **{f"train/{k}": v for k, v in train_m.items() if k != "layer_rates"},
                       **{f"val/{k}": v for k, v in val_m.items() if k != "layer_rates"}}, step=epoch)

        if score < best_score and delays_hard:      # learned delays: only models with whole-step delays
            best_score = score
            save_checkpoint(model, optimizer, epoch, val_m["ctc_loss"],
                            os.path.join(args.checkpoint_dir, "best_model_weights.pth"),
                            additional_info=checkpoint_info(is_best=True, val_metrics=val_m))
            print(f"Saved best model ({args.selection_metric} {best_score:.4f})")
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_m["ctc_loss"],
                            os.path.join(args.checkpoint_dir, f"checkpoint_{exp_name}_epoch{epoch + 1}.pth"),
                            additional_info=checkpoint_info())

        if args.disable_early_stopping:
            if new_lr <= args.min_lr and lr_reduction_count >= args.max_lr_reductions:
                print(f"Reached minimum LR after {lr_reduction_count} reductions; stopping.")
                break
        elif early_stopping(score):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - training_start
    print(f"\nTraining finished in {training_time / 60:.1f} min; best {args.selection_metric} {best_score:.4f}")
    save_checkpoint(model, optimizer, epoch, history["test_loss"][-1] if history["test_loss"] else float("nan"),
                    os.path.join(args.checkpoint_dir, f"final_model_{exp_name}.pth"),
                    additional_info=checkpoint_info(is_final=True))
    if history["train_loss"]:
        plot_training_curves(history["train_loss"], history["test_loss"],
                             os.path.join(args.checkpoint_dir, f"training_curves_{exp_name}.png"),
                             title=f"Training Progress - {exp_name}", lr_history=history["lr"],
                             test_per=history["test_per"])
    with open(os.path.join(args.checkpoint_dir, f"summary_{exp_name}.txt"), "w") as f:
        f.write(f"Experiment: {exp_name}\nFinal epoch: {epoch + 1}\n"
                f"Best {args.selection_metric}: {best_score:.5f}\nTraining time: {training_time / 60:.1f} minutes\n"
                f"Arguments: {vars(args)}\n")
    if args.wandb:
        wandb.run.summary.update({"best_score": best_score, "total_training_time": training_time})
        wandb.finish()
    print(f"Results saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
