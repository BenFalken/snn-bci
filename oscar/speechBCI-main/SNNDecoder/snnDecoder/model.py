"""
Feedforward spiking phoneme decoder, forked from spike_bmi_main's model_hkm.py
(velocity decoder) for this repo's speech data.

Architecture, one 20 ms bin per timestep:
    per-session input layer  Linear(256, 256) + softsign + dropout, non-spiking,
                             identity-initialized -- the same per-day layer the
                             RNN uses (NeuralDecoder inputNetwork) to absorb
                             day-to-day drift in the recordings
    spiking stack            Linear (no bias) -> IAF/LIF neuron per layer,
                             widths hidden_dims + [readout_width]
                             (default 512 -> 256 -> 128 -> 128), each layer with
                             its own optional trained tau_syn
    readout                  cascaded EMA of the last spiking layer's spikes
                             (temporal_decay_stages, one shared trained decay,
                             each stage with unit gain, so the readout sees a
                             smoothed spike rate) -> Linear(readout_width, n_classes)

The output is one row of logits per timestep, in the RNN's class order
(0-38 phonemes of PHONE_DEF_SIL, 39 SIL, 40 CTC blank), so the RNN's CTC
language-model decoders (lmDecoderUtils) work on it unchanged.

Every operation is causal, so forward() can run a whole (padded) batch of
sequences layer by layer -- identical to stepping one timestep at a time,
which step() does for streaming / real-time use. Padding at the end of a
sequence cannot affect its earlier outputs; pass `lengths` so padded
timesteps are left out of the spike and op counts.

tau_syn (IAF and LIF): adds a synaptic-current stage before the membrane,
input -> i_syn (decaying at tau_syn) -> v_mem, with tau_syn in timesteps
(20 ms here, vs 4 ms in the velocity decoder: the same physical time
constants are 5x smaller numbers here).

norm_syn (default True): sinabs updates i_syn = alpha * (i_syn + input),
alpha = exp(-1 / tau_syn), so a layer's steady-state gain is
alpha / (1 - alpha) -- 0.09 at tau_syn 0.4, 1.5 at 2, 15.5 at 16 -- and its
time constant also sets how hard it drives the next layer against a fixed
threshold. norm_syn scales each layer's input by (1 - alpha) / alpha (from
its trained tau_syn), making the synaptic stage a unit-gain low-pass filter,
so tau_syn sets only the timescale. One scalar per layer, so deployed it
folds into that layer's weights (not counted as extra ops). False keeps
sinabs' (and the velocity decoder's) behaviour.

neuron_type 'alif' (adaptive LIF, sinabs ALIF): a LIF neuron whose threshold
rises with its own recent spiking, threshold = b0 + adapt_scale * b, with
b a low-pass (tau_adapt timesteps) of its spikes. The adaptation limits
runaway firing and adds a slower state per neuron (spike-frequency
adaptation). A very large tau_mem with norm_input False makes it an
adaptive IAF.

tau_syn_spread F > 1: each neuron gets its own trained tau_syn, log-spaced
over [tau / F, tau * F] around its layer's tau_syn -- a bank of timescales
per layer instead of one shared time constant.

temporal_decay_stages 0: no EMA, the readout sees the last spiking layer's
spikes of the current timestep only.

norm_input (LIF/ALIF only): whether input is divided by tau_mem before
accumulating (sinabs' default True); with True, the division compounds with
depth, which can silence deep networks at large tau_mem.

count_ops (forward() only, off during training): accumulates effective
operation counts (MAC, ACC, elementwise) over the valid timesteps, in
self.op_counts, for energy.op_energy_estimate.finalize_snn_ops().
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import sinabs
import sinabs.activation
import sinabs.layers as sl
from sinabs.activation import (
    Gaussian,
    Heaviside,
    MultiGaussian,
    PeriodicExponential,
    SingleExponential,
)

# ---------------------------------------------------------------------------
# Optional Exodus acceleration (CUDA kernels for sinabs layers; training only)
# ---------------------------------------------------------------------------
try:
    import sinabs.exodus.layers as exodus_layers

    EXODUS_AVAILABLE = True
except ImportError:
    EXODUS_AVAILABLE = False

N_CLASSES = 41  # 39 phonemes + SIL + CTC blank (last)


class NeuronType(Enum):
    IAF = "iaf"
    LIF = "lif"
    ALIF = "alif"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def get_reset_fn(reset_type: str, force_soft: bool = False):
    """Return a sinabs reset function. reset_type: 'hard' or 'soft'.
    force_soft: always return soft reset (required by Exodus)."""
    if force_soft or reset_type == "soft":
        return sinabs.activation.MembraneSubtract()
    if reset_type == "hard":
        return sinabs.activation.MembraneReset()
    raise ValueError(f"Unknown reset_type {reset_type!r}. Choose 'hard' or 'soft'.")


class RampSurrogate:
    """Wide surrogate derivative of Balafrej, Alibart & Rouat (arXiv 2310.19067, DelayNet):
    min(max(u - u_th + 1, 0), 1), with u in units of the threshold here -- 1 at and above the threshold, falling
    linearly to 0 one threshold below it. The gradient passes whenever the neuron is anywhere near firing."""

    def __call__(self, v_mem, spike_threshold):
        return torch.clamp((v_mem - spike_threshold) / spike_threshold + 1.0, 0.0, 1.0)


def get_surrogate_grad_fn(surrogate_type: Union[str, dict, Callable]) -> Callable:
    """Return an instantiated sinabs surrogate-gradient function."""
    if callable(surrogate_type):
        return surrogate_type

    if isinstance(surrogate_type, dict):
        fn_type = surrogate_type.get("type", "periodic_exponential")
        params = {k: v for k, v in surrogate_type.items() if k != "type"}
    else:
        fn_type = surrogate_type
        params = {}

    _map = {
        "periodic_exponential": PeriodicExponential,
        "single_exponential": SingleExponential,
        "gaussian": Gaussian,
        "multi_gaussian": MultiGaussian,
        "heaviside": Heaviside,
        "ramp": RampSurrogate,
    }
    if fn_type not in _map:
        raise ValueError(f"Unknown surrogate gradient {fn_type!r}. Available: {list(_map)}")
    return _map[fn_type](**params)


class ALIFKeepSyn(sl.ALIF):
    """sinabs ALIF whose synaptic current carries over between calls.

    sinabs 3.1's ALIF.forward writes v_mem, b and spike_threshold back after a
    call but not i_syn, so with tau_syn every call restarts the synaptic
    current from its stale value: a whole sequence in one call is right, but
    stepping it (step(), chunked streaming) is not. Same forward, plus i_syn.
    Also accepts a per-neuron tau_syn tensor (sinabs' ALIF tests `if tau_syn`,
    which fails for tensors with more than one value)."""

    def __init__(self, *args, tau_syn=None, **kwargs):
        per_neuron = torch.is_tensor(tau_syn) and tau_syn.numel() > 1
        super().__init__(*args, tau_syn=1.0 if per_neuron else tau_syn, **kwargs)
        if per_neuron:
            self.tau_syn = nn.Parameter(tau_syn.detach().clone().float())

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        from sinabs.layers import functional
        batch_size, time_steps, *trailing_dim = input_data.shape
        if not self.is_state_initialised() or not self.state_has_shape((batch_size, *trailing_dim)):
            self.init_state_with_shape((batch_size, *trailing_dim))
        spikes, state, recordings = functional.alif_forward(
            input_data=input_data, alpha_mem=self.alpha_mem_calculated, alpha_adapt=self.alpha_adapt_calculated,
            alpha_syn=self.alpha_syn_calculated, adapt_scale=self.adapt_scale, state=dict(self.named_buffers()),
            spike_fn=self.spike_fn, reset_fn=self.reset_fn, surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem, b0=self.b0, norm_input=self.norm_input, record_states=self.record_states)
        self.b = state["b"]
        self.v_mem = state["v_mem"]
        self.spike_threshold = state["spike_threshold"]
        if "i_syn" in state:
            self.i_syn = state["i_syn"]
        self.recordings = recordings
        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes


def build_neuron(neuron_type: NeuronType, spike_threshold: float, reset_type: str,
                 surrogate_grad, tau_mem: float, min_vmem: Optional[float], spike_fn,
                 tau_syn: Optional[float], norm_input: bool, use_exodus: bool,
                 tau_adapt: float = 25.0, adapt_scale: float = 1.8) -> nn.Module:
    """One sinabs spiking layer taking (N, T, C) input."""
    use_exodus = use_exodus and EXODUS_AVAILABLE and neuron_type != NeuronType.ALIF
    common = dict(spike_threshold=spike_threshold, min_v_mem=min_vmem,
                  surrogate_grad_fn=get_surrogate_grad_fn(surrogate_grad),
                  reset_fn=get_reset_fn(reset_type, force_soft=use_exodus),
                  spike_fn=spike_fn, tau_syn=tau_syn)
    if neuron_type == NeuronType.ALIF:
        # threshold = b0 + adapt_scale * b; sinabs takes the base threshold b0 from spike_threshold
        return ALIFKeepSyn(tau_mem=tau_mem, tau_adapt=tau_adapt, adapt_scale=adapt_scale,
                       norm_input=norm_input, **common)
    if neuron_type == NeuronType.LIF:
        cls = exodus_layers.LIF if use_exodus else sl.LIF
        return cls(tau_mem=tau_mem, norm_input=norm_input, **common)
    cls = exodus_layers.IAF if use_exodus else sl.IAF
    return cls(**common)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SNN_Speech(nn.Module):
    """Feedforward spiking network for streaming phoneme (CTC) decoding."""

    def __init__(
        self,
        n_sessions: int,
        num_input_channels: int = 256,
        n_classes: int = N_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        readout_width: int = 128,
        neuron_type: NeuronType = NeuronType.IAF,
        tau_mem: float = 1.0,
        reset_type: str = "hard",
        final_layer_reset_type: Optional[str] = None,
        surrogate_grad: Union[str, dict, Callable] = "periodic_exponential",
        use_exodus: bool = False,
        min_vmem: Optional[float] = None,
        spike_fn: Optional[Callable] = None,
        spike_thresholds: Optional[List[float]] = None,
        tau_syn: Optional[Union[float, List[float]]] = None,
        norm_input: bool = True,
        last_layer_reset: bool = False,
        temporal_decay_init: float = 0.8,
        learnable_temporal_decay: bool = True,
        temporal_decay_stages: int = 1,
        input_dropout: float = 0.2,
        norm_syn: bool = True,
        tau_syn_spread: float = 1.0,
        tau_adapt: float = 25.0,
        adapt_scale: float = 1.8,
    ):
        super().__init__()
        self.norm_syn = norm_syn
        spike_fn = spike_fn or sinabs.activation.MultiSpike
        out_rst = final_layer_reset_type or reset_type
        self.last_layer_reset = last_layer_reset

        # Per-session input layers, identity-initialized like the RNN's.
        self.day_layers = nn.ModuleList()
        for _ in range(n_sessions):
            layer = nn.Linear(num_input_channels, num_input_channels)
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.day_layers.append(layer)
        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)

        hidden_dims = list(hidden_dims) if hidden_dims is not None else [512, 256, 128]
        layer_widths = [num_input_channels] + hidden_dims + [readout_width]
        n_layers = len(layer_widths) - 1

        thresholds = spike_thresholds or ([1.0] * n_layers)
        if len(thresholds) != n_layers:
            raise ValueError(
                f"spike_thresholds must have exactly {n_layers} values (one per spiking layer, "
                f"given hidden_dims={hidden_dims} + readout_width -> {n_layers} layers), "
                f"got {len(thresholds)}")

        # tau_syn: one initial value for every spiking layer, or one per layer.
        taus = list(tau_syn) if isinstance(tau_syn, (list, tuple)) else [tau_syn] * n_layers
        taus = [None if t is None else float(t) for t in taus]
        if len(taus) != n_layers:
            raise ValueError(f"tau_syn must have 1 or {n_layers} values (one per layer), got {len(taus)}")
        self.has_tau_syn = [t is not None for t in taus]
        if tau_syn_spread > 1:
            # per-neuron time constants, log-spaced around each layer's tau_syn
            taus = [None if t is None else torch.logspace(math.log10(t / tau_syn_spread),
                                                          math.log10(t * tau_syn_spread), width)
                    for t, width in zip(taus, layer_widths[1:])]
        self.neuron_type = neuron_type

        self.linears = nn.ModuleList()
        self.neurons = nn.ModuleList()
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            self.linears.append(nn.Linear(layer_widths[i], layer_widths[i + 1], bias=False))
            self.neurons.append(build_neuron(
                neuron_type, thresholds[i], out_rst if is_last else reset_type, surrogate_grad,
                tau_mem, min_vmem, spike_fn, taus[i], norm_input, use_exodus, tau_adapt, adapt_scale))
        self.layer_widths = list(layer_widths[1:])   # neurons per spiking layer
        self.total_neuron_units = sum(self.layer_widths)

        # EMA readout: bounded, learnable scalar decay shared by every stage.
        # With >1 stage the cascade forms a gamma/Erlang-shaped kernel that
        # weights a range of recent timesteps near-equally before tapering.
        if temporal_decay_stages < 0:
            raise ValueError(f"temporal_decay_stages must be >= 0, got {temporal_decay_stages}")
        self.temporal_decay_stages = temporal_decay_stages
        init_logit = torch.logit(torch.tensor(float(temporal_decay_init)))
        self._temporal_decay_raw = nn.Parameter(init_logit, requires_grad=learnable_temporal_decay)

        self.readout = nn.Linear(readout_width, n_classes)
        self.n_classes = n_classes
        self._stream_ema = None
        self.frame_kernel = self.frame_stride = 1   # one output per input bin

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Output frames per trial: one per input bin."""
        return lengths

    @property
    def temporal_decay(self) -> float:
        """Current EMA decay in (0, 1)."""
        return torch.sigmoid(self._temporal_decay_raw).item()

    def _day_layer(self, x: torch.Tensor, day_idx: int) -> torch.Tensor:
        return self.day_dropout(self.day_activation(self.day_layers[day_idx](x)))

    def _synaptic_input(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """Layer i's Linear output, scaled to unit synaptic gain (norm_syn)."""
        tau = getattr(self.neurons[i], 'tau_syn', None)
        if not self.norm_syn or tau is None:
            return x
        alpha = torch.exp(-1.0 / tau)
        return x * ((1 - alpha) / alpha)

    def _neuron(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if self.last_layer_reset and i == len(self.neurons) - 1:
            # Reset the output layer's membrane before every timestep.
            steps = []
            for t in range(x.shape[1]):
                self.neurons[i].reset_states()
                steps.append(self.neurons[i](x[:, t:t + 1]))
            return torch.cat(steps, dim=1)
        return self.neurons[i](x)

    def forward(self, x: torch.Tensor, day_idx: int, lengths: Optional[torch.Tensor] = None,
                count_ops: bool = False, reset_state: bool = True) -> torch.Tensor:
        """
        Run the network over a batch of sequences from one session.

        Parameters
        ----------
        x : (N, T, num_input_channels), batch-major
        day_idx : index of the session's input layer
        lengths : (N,) valid timesteps per sequence (default: all T); padded
            timesteps are excluded from the spike and op counts
        count_ops : also accumulate op counts into self.op_counts
            ({'mac', 'acc', 'elementwise'}, summed over valid timesteps and
            sequences; n_samples for finalize_snn_ops() is lengths.sum())
        reset_state : reset neuron state first (True: every sequence starts
            from rest, as in training). False carries the state of the
            previous call in, e.g. to decode a stream in chunks.

        Returns
        -------
        logits : (N, T, n_classes)

        Also sets self.layer_spikes (spikes per spiking layer over the valid
        timesteps, with gradients, for the spike-rate penalty) and
        self.total_spikes.
        """
        N, T = x.shape[0], x.shape[1]
        if reset_state:
            sinabs.utils.reset_states(self)
        if lengths is None:
            mask = torch.ones(N, T, 1, device=x.device)
        else:
            mask = (torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None]).float()[..., None]

        op_mac, op_acc, op_elementwise = 0, 0.0, 0
        n_valid = int(mask.sum().item()) if count_ops else 0
        if count_ops:
            # Input layer: continuous input, MAC per nonzero input x output; softsign.
            op_mac += int(torch.count_nonzero(x * mask).item()) * self.day_layers[day_idx].out_features
            op_elementwise += n_valid * self.day_layers[day_idx].out_features
        x = self._day_layer(x, day_idx)

        self.layer_spikes = []
        for i, (linear, units) in enumerate(zip(self.linears, self.layer_widths)):
            if count_ops:
                if i == 0:
                    # First Linear: continuous input-layer output -> MAC per nonzero.
                    op_mac += int(torch.count_nonzero(x * mask).item()) * units
                else:
                    # Later Linears: spike input -> one ACC per spike (MultiSpike
                    # can emit several per timestep, each a separate accumulate).
                    op_acc += float((x * mask).sum().item()) * units
            x = self._neuron(i, self._synaptic_input(i, linear(x)))
            # Spike counts are non-negative; clamp defensively before
            # accumulating so numerical instability cannot corrupt the
            # spike-rate penalty (--min-vmem should prevent it anyway).
            self.layer_spikes.append((x * mask).sum().clamp(min=0))
            if count_ops:
                if self.has_tau_syn[i]:
                    # i_syn[t] = decay * i_syn[t-1] + I_in[t]: a multiply per
                    # neuron per timestep, whatever the spike activity.
                    op_mac += n_valid * units
                # Membrane update + threshold compare + reset.
                op_elementwise += 2 * n_valid * units
                if self.neuron_type != NeuronType.IAF:
                    op_mac += n_valid * units          # membrane leak
                if self.neuron_type == NeuronType.ALIF:
                    op_mac += 2 * n_valid * units      # adaptation decay + threshold update
        self.total_spikes = sum(self.layer_spikes)

        # Cascaded EMA over time: stage 0 fed by the spikes, each later stage
        # by the previous stage's output. Each stage has unit gain,
        # s += (1 - decay) * (x - s), unlike the velocity decoder's
        # s = decay * s + x (whose population-vector ratio cancelled the gain
        # of 1 / (1 - decay) per stage; a softmax readout would not).
        decay = torch.sigmoid(self._temporal_decay_raw)
        for _ in range(self.temporal_decay_stages):
            state = torch.zeros(N, x.shape[-1], device=x.device)
            smoothed = []
            for t in range(T):
                state = state + (1 - decay) * (x[:, t] - state)
                smoothed.append(state)
            x = torch.stack(smoothed, dim=1)
        if count_ops:
            # One multiply-add (and a subtraction) per element per stage.
            op_mac += self.temporal_decay_stages * n_valid * x.shape[-1]
            op_elementwise += self.temporal_decay_stages * n_valid * x.shape[-1]
            op_mac += int(torch.count_nonzero(x * mask).item()) * self.n_classes
            self.op_counts = {'mac': op_mac, 'acc': op_acc, 'elementwise': op_elementwise}

        return self.readout(x)

    # -----------------------------------------------------------------------
    # Streaming: one timestep per call, state carried between calls
    # -----------------------------------------------------------------------

    def reset_stream(self):
        """Start a new stream (e.g. a new sentence) from rest."""
        sinabs.utils.reset_states(self)
        self._stream_ema = None

    def step(self, x: torch.Tensor, day_idx: int) -> torch.Tensor:
        """One timestep: x (N, num_input_channels) -> logits (N, n_classes).
        Equivalent to forward() over the same stream (call reset_stream()
        first), one bin at a time, as a real-time decoder would run."""
        x = self._day_layer(x, day_idx)[:, None, :]
        for i, linear in enumerate(self.linears):
            x = self._neuron(i, self._synaptic_input(i, linear(x)))
        x = x[:, 0]
        decay = torch.sigmoid(self._temporal_decay_raw)
        if self._stream_ema is None:
            self._stream_ema = [torch.zeros_like(x) for _ in range(self.temporal_decay_stages)]
        for s in range(self.temporal_decay_stages):
            self._stream_ema[s] = self._stream_ema[s] + (1 - decay) * (x - self._stream_ema[s])
            x = self._stream_ema[s]
        return self.readout(x)


def create_model(
    n_sessions: int,
    weight_init: Optional[str] = "kaiming",
    num_input_channels: int = 256,
    n_classes: int = N_CLASSES,
    hidden_dims: Optional[List[int]] = None,
    readout_width: int = 128,
    neuron_type: str = "iaf",
    arch: str = "ff",
    **kwargs,
) -> nn.Module:
    """Build an SNN_Speech. weight_init ('kaiming', 'xavier' or None) applies
    to the spiking stack's Linears and the readout; the per-session input
    layers stay identity-initialized."""
    if arch == "rsnn":
        # recurrent, RNN-shaped network (rsnn.py); builds its own layers
        from .rsnn import SNN_Recurrent
        return SNN_Recurrent(n_sessions=n_sessions, num_input_channels=num_input_channels,
                             n_classes=n_classes, hidden_dims=hidden_dims, neuron_type=neuron_type.lower(),
                             **kwargs)
    if arch == "native":
        # spike-native 20 ms recurrent network on raw counts (rsnn_native.py)
        from .rsnn_native import SNN_Native
        return SNN_Native(n_sessions=n_sessions, num_input_channels=num_input_channels, n_classes=n_classes,
                          hidden_dims=hidden_dims, neuron_type=neuron_type.lower(), **kwargs)
    if arch != "ff":
        raise ValueError(f"arch must be 'ff', 'rsnn' or 'native', got {arch!r}")
    try:
        neuron_enum = NeuronType(neuron_type.lower())
    except ValueError:
        raise ValueError(f"neuron_type must be one of {[t.value for t in NeuronType]}, "
                         f"got {neuron_type!r}")

    model = SNN_Speech(n_sessions=n_sessions, num_input_channels=num_input_channels,
                       n_classes=n_classes, hidden_dims=hidden_dims, readout_width=readout_width,
                       neuron_type=neuron_enum, **kwargs)
    for m in list(model.linears) + [model.readout]:
        if weight_init == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif weight_init == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
    nn.init.zeros_(model.readout.bias)
    return model


# Neuron state buffers, lazily shaped by sinabs on the first forward pass.
STATE_SUFFIXES = ('.v_mem', '.i_syn', '.b', '.spike_threshold')   # .b, .spike_threshold: ALIF


def load_model_weights(model: SNN_Speech, state_dict: dict, source_description: str = "checkpoint") -> None:
    """Load only weights/parameters into `model`, skipping sinabs' neuron
    state buffers (v_mem/i_syn), whose shape depends on the last batch run.

    Raises RuntimeError on anything that looks like a real architecture
    mismatch (a missing or unexpected parameter) rather than silently
    loading a partially-wrong model."""
    filtered = {k: v for k, v in state_dict.items() if not k.endswith(STATE_SUFFIXES)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if not k.endswith(STATE_SUFFIXES)]
    if missing:
        raise RuntimeError(f"{source_description}: missing key(s) {missing} -- the architecture "
                           f"flags do not match the checkpoint")
    if unexpected:
        raise RuntimeError(f"{source_description}: unexpected key(s) {unexpected} -- the "
                           f"architecture flags do not match the checkpoint")
