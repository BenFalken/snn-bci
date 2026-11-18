import os
os.environ["PYTHONHASHSEED"] = "0"
import sys
import math
import random

import numpy as np
import torch
import torch.nn as nn

from base_dockex.BaseModel import BaseModel


class MaxSpike(torch.autograd.Function):
    """Spike count per step: floor(v / threshold), clipped to [0, max_spikes].
    Backward: a single-exponential surrogate around the threshold."""

    @staticmethod
    def forward(ctx, v, threshold, max_spikes, width):
        ctx.save_for_backward(v, threshold)
        ctx.width = width
        return torch.clamp(torch.floor(v / threshold), 0, max_spikes)

    @staticmethod
    def backward(ctx, grad_out):
        v, threshold = ctx.saved_tensors
        sg = torch.exp(-torch.abs(v - threshold) / ctx.width) / threshold
        return grad_out * sg, None, None, None


class LSNNLayer(nn.Module):
    """One recurrent layer of integrate-and-fire neurons (no leak, soft reset), a
    fraction of them adaptive, each with its own trained synaptic time constant."""

    def __init__(self, n_in, n_out, tau_syn_steps, adaptive_fraction, tau_adapt_steps, adapt_scale,
                 max_spikes, rec_init_scale, max_vmem, min_vmem, seed):
        super().__init__()
        self.width = n_out
        self.ff = nn.Linear(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight)
        self.rec = nn.Linear(n_out, n_out, bias=False)
        nn.init.orthogonal_(self.rec.weight)
        with torch.no_grad():
            self.rec.weight.mul_(rec_init_scale)
        lo, hi = tau_syn_steps
        self.log_tau_syn = nn.Parameter(torch.log(torch.logspace(math.log10(lo), math.log10(hi), n_out)))
        g = torch.Generator().manual_seed(seed)
        mask = torch.zeros(n_out)
        mask[torch.randperm(n_out, generator=g)[:int(round(adaptive_fraction * n_out))]] = 1.0
        self.register_buffer("adapt_mask", mask)
        self.alpha_adapt = math.exp(-1.0 / tau_adapt_steps)
        self.adapt_scale, self.max_spikes = adapt_scale, max_spikes
        self.max_vmem, self.min_vmem = max_vmem, min_vmem

    def init_state(self, n, device):
        z = lambda: torch.zeros(n, self.width, device=device)
        return {"i": z(), "v": z(), "b": z(), "s": z()}

    def step(self, current, st):
        current = current + self.rec(st["s"])
        alpha = torch.exp(-1.0 / self.log_tau_syn.exp())
        st["i"] = alpha * st["i"] + (1 - alpha) * current          # unit-gain synaptic filter
        v = st["v"] + st["i"]                                       # integrate-and-fire, no leak
        thr = 1.0 + self.adapt_scale * st["b"] * self.adapt_mask
        s = MaxSpike.apply(v, thr, self.max_spikes, 0.5)
        v = v - s * thr                                             # soft reset
        st["v"] = torch.clamp(v, min=self.min_vmem, max=self.max_vmem)
        st["b"] = self.alpha_adapt * st["b"] + (1 - self.alpha_adapt) * s
        st["s"] = s
        return s


class LSNNNet(nn.Module):
    """Recurrent spiking layers, then leaky non-spiking output neurons reading every
    layer's spikes; the output neurons' potential at the window's last step is the
    prediction, as the Keras RNNs' last hidden state feeds their Dense layer."""

    def __init__(self, n_inputs, n_outputs, hidden_dims, step_ms, adaptive_fraction, tau_adapt_ms,
                 adapt_scale, tau_syn_ms, out_tau_ms, max_spikes, rec_init_scale, max_vmem, min_vmem,
                 input_dropout=0.0, layer_dropout=0.0):
        super().__init__()
        # dropout, training only: on the input features, and on the spikes each layer passes on (to the next
        # layer and to the output neurons), as the speech LSNN
        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        dims = [n_inputs] + list(hidden_dims)
        self.spiking = nn.ModuleList([
            LSNNLayer(dims[i], dims[i + 1], [t / step_ms for t in tau_syn_ms], adaptive_fraction,
                      tau_adapt_ms / step_ms, adapt_scale, max_spikes, rec_init_scale, max_vmem, min_vmem,
                      seed=5 + i)
            for i in range(len(hidden_dims))])
        self.readout = nn.Linear(sum(hidden_dims), n_outputs)
        nn.init.kaiming_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.out_log_tau = nn.Parameter(torch.full((n_outputs,), math.log(out_tau_ms / step_ms)))

    def forward(self, x, state=None, all_steps=False):
        """x: (N, T, C) -> ((N, n_outputs) prediction at the last step, or (N, T, n_outputs) with all_steps; mean
        spikes per neuron per step). state: carried over from the previous call (streaming); after the call,
        self.state holds the state to carry on with."""
        N, T, _ = x.shape
        a_out = torch.exp(-1.0 / self.out_log_tau.exp())
        if state is None:
            states = [layer.init_state(N, x.device) for layer in self.spiking]
            y = self.readout.bias.expand(N, -1)
        else:
            states, y = state
        drive = self.spiking[0].ff(self.input_dropout(x))            # first layer's input, all steps at once
        spikes = x.new_zeros(())
        outs = []
        for t in range(T):
            current, layer_out = drive[:, t], []
            for li, (layer, st) in enumerate(zip(self.spiking, states)):
                if li > 0:
                    current = layer.ff(self.layer_dropout(s))
                s = layer.step(current, st)
                layer_out.append(s)
                spikes = spikes + s.sum()
            y = a_out * y + (1 - a_out) * self.readout(self.layer_dropout(torch.cat(layer_out, -1)))
            if all_steps:
                outs.append(y)
        self.state = (states, y)
        rate = spikes / (N * T * sum(layer.width for layer in self.spiking))
        return (torch.stack(outs, 1) if all_steps else y), rate


class LSNN(BaseModel):
    def __init__(self, input_args):
        super().__init__(input_args)
        p = lambda k, d: self.params[k] if self.params[k] is not None else d
        self.units = self.params['units']
        self.n_layers = p('n_layers', 2)
        self.epochs = p('epochs', 2048)
        self.batch_size = p('batch_size', 64)
        self.lr = p('learning_rate', 1e-3)
        self.patience = p('patience', 5)
        self.spike_rate_weight = p('spike_rate_weight', 0.0)
        self.spike_rate_target = p('spike_rate_target', 0.1)
        self.net_kwargs = dict(
            hidden_dims=[self.units] * self.n_layers,
            step_ms=p('step_ms', 40.0),
            adaptive_fraction=p('adaptive_fraction', 0.9),
            tau_adapt_ms=p('tau_adapt_ms', 500.0),
            adapt_scale=p('adapt_scale', 1.8),
            tau_syn_ms=tuple(p('tau_syn_ms', [20.0, 1300.0])),
            out_tau_ms=p('out_tau_ms', 40.0),
            max_spikes=p('max_spikes', 3),
            rec_init_scale=p('rec_init_scale', 0.5),
            max_vmem=p('max_vmem', 50.0),
            min_vmem=p('min_vmem', -1.0),
            input_dropout=p('input_dropout', 0.0),
            layer_dropout=p('layer_dropout', 0.0),
        )
        # mode 'window': each window read from a fresh state, the prediction at its last step (as the Keras RNNs).
        # 'stream': the windows' frames put back into one continuous recording, read without resets; window i's
        # prediction is read at its last frame, so the same look-ahead, plus all the past. Trained on random
        # stream_chunk_steps stretches of the training portion, the first stream_burn_in_steps of each not scored.
        self.mode = p('mode', 'window')
        if self.mode not in ('window', 'stream'):
            raise ValueError(f"mode must be 'window' or 'stream', got {self.mode!r}")
        self.chunk_steps = p('stream_chunk_steps', 150)
        self.burn_in = p('stream_burn_in_steps', 25)
        self.coverage = p('stream_coverage', 4)               # each epoch's chunks cover the frames this many times
        self.stream_batch = p('stream_batch_size', 16)
        # training-only input noise, as the speech decoders: white noise on every value, and a constant offset per
        # window and channel
        self.white_noise_sd = p('white_noise_sd', 0.0)
        self.constant_offset_sd = p('constant_offset_sd', 0.0)
        self.device = torch.device(p('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    def set_random_seeds(self):
        super().set_random_seeds()
        torch.manual_seed(self.random_seed)

    def instantiate_model(self):
        self.model = LSNNNet(self.X_train.shape[2], self.y_train.shape[1], **self.net_kwargs).to(self.device)
        if self.input_pathnames['init_model_pt'] is not None:
            # fine-tuning: start from a saved model's weights (same inputs, outputs and network)
            ck = torch.load(self.input_pathnames['init_model_pt'], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck['state_dict'])
            print(f"Starting from the weights in {self.input_pathnames['init_model_pt']}")
        n = sum(q.numel() for q in self.model.parameters())
        print(f'LSNN: {self.n_layers} x {self.units} recurrent spiking layers, '
              f'{self.net_kwargs["adaptive_fraction"]:.0%} adaptive, {n:,} parameters, on {self.device}')

    def _tensor(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def _loss(self, X, Y):
        """Mean squared error over a whole split, in batches, without gradients."""
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                pred, _ = self.model(self._tensor(X[i:i + 1024]))
                total += float(((pred - self._tensor(Y[i:i + 1024])) ** 2).sum())
        return total / Y.size

    @staticmethod
    def _stream(windows):
        """(n, W, C) windows of consecutive frames (window i = frames i..i+W-1) -> the (n + W - 1, C) recording."""
        return np.concatenate([windows[:, 0], windows[-1, 1:]], 0)

    def _run_stream(self, x, piece=2000):
        """One pass over a (T, C) recording without resets, no gradients: (T, n_outputs) outputs, every step."""
        self.model.eval()
        out, state = [], None
        with torch.no_grad():
            for i in range(0, len(x), piece):
                y, _ = self.model(self._tensor(x[None, i:i + piece]), state=state, all_steps=True)
                state = self.model.state
                out.append(y[0].cpu().numpy())
        return np.concatenate(out, 0)

    def _fit_stream(self):
        """Training in stream mode: random stretches of the training recording, each from a fresh state; the
        output at step f is window f - lag's prediction (lag = window length - 1), scored after the burn-in."""
        print(f'Fitting model (stream: {self.chunk_steps}-step stretches, first {self.burn_in} not scored)')
        lag = self.X_train.shape[1] - 1
        n_tr = len(self.X_train)
        xs = self._stream(self.X_train)                                  # frames 0 .. n_tr + lag - 1
        ys = np.concatenate([np.zeros((lag, self.y_train.shape[1])), self.y_train], 0)   # target of step f
        L = min(self.chunk_steps, len(xs))
        has_valid = self.X_valid is not None and self.y_valid is not None
        if has_valid:
            x_tv = self._stream(np.concatenate([self.X_train, self.X_valid], 0))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, waited = math.inf, None, 0
        n_chunks = max(1, int(round(self.coverage * len(xs) / L)))
        for epoch in range(self.epochs):
            self.model.train()
            run_loss, run_rate, n_batches = 0.0, 0.0, 0
            starts = np.random.randint(0, len(xs) - L + 1, n_chunks)
            for i in range(0, n_chunks, self.stream_batch):
                s = starts[i:i + self.stream_batch]
                steps = s[:, None] + np.arange(L)[None]                       # (B, L) recording steps
                x = self._tensor(xs[steps])
                if self.white_noise_sd > 0:
                    x = x + self.white_noise_sd * torch.randn_like(x)
                if self.constant_offset_sd > 0:
                    x = x + self.constant_offset_sd * torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
                mask = (steps >= lag) & (np.arange(L)[None] >= self.burn_in)
                if not mask.any():
                    continue
                pred, rate = self.model(x, all_steps=True)
                m = torch.as_tensor(mask, device=self.device)
                loss = ((pred - self._tensor(ys[steps])) ** 2)[m].mean()
                total = loss + self.spike_rate_weight * (rate - self.spike_rate_target).clamp(min=0) ** 2
                opt.zero_grad()
                total.backward()
                if not all(torch.isfinite(q.grad).all() for q in self.model.parameters() if q.grad is not None):
                    continue
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                opt.step()
                run_loss, run_rate, n_batches = run_loss + float(loss), run_rate + float(rate), n_batches + 1
            if has_valid:
                out = self._run_stream(x_tv)[n_tr + lag:]                     # validation windows' predictions
                val = float(((out - self.y_valid) ** 2).mean())
            else:
                val = run_loss / max(n_batches, 1)
            print(f'Epoch {epoch + 1}/{self.epochs} - loss: {run_loss / max(n_batches, 1):.4f} - '
                  f'val_loss: {val:.4f} - spikes/neuron/step: {run_rate / max(n_batches, 1):.3f}', flush=True)
            if val < best - 1e-4:
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    print(f'Early stopping; restoring the weights of the best epoch (val_loss {best:.4f})')
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit(self):
        if self.mode == 'stream':
            return self._fit_stream()
        print('Fitting model')
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        has_valid = self.X_valid is not None and self.y_valid is not None
        best, best_state, waited = math.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = np.random.permutation(len(self.X_train))
            run_loss, run_rate, n_batches = 0.0, 0.0, 0
            for i in range(0, len(order), self.batch_size):
                idx = np.sort(order[i:i + self.batch_size])
                x = self._tensor(self.X_train[idx])
                if self.white_noise_sd > 0:
                    x = x + self.white_noise_sd * torch.randn_like(x)
                if self.constant_offset_sd > 0:
                    x = x + self.constant_offset_sd * torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
                pred, rate = self.model(x)
                loss = ((pred - self._tensor(self.y_train[idx])) ** 2).mean()
                total = loss + self.spike_rate_weight * (rate - self.spike_rate_target).clamp(min=0) ** 2
                opt.zero_grad()
                total.backward()
                if not all(torch.isfinite(q.grad).all() for q in self.model.parameters() if q.grad is not None):
                    continue                                        # skip a non-finite update
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                opt.step()
                run_loss, run_rate, n_batches = run_loss + float(loss), run_rate + float(rate), n_batches + 1
            val = self._loss(self.X_valid, self.y_valid) if has_valid else run_loss / max(n_batches, 1)
            print(f'Epoch {epoch + 1}/{self.epochs} - loss: {run_loss / max(n_batches, 1):.4f} - '
                  f'val_loss: {val:.4f} - spikes/neuron/step: {run_rate / max(n_batches, 1):.3f}', flush=True)
            if val < best - 1e-4:                                   # as the Keras decoders' EarlyStopping
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    print(f'Early stopping; restoring the weights of the best epoch (val_loss {best:.4f})')
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _predict(self, X):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                out.append(self.model(self._tensor(X[i:i + 1024]))[0].cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def predict(self):
        print('Generating predictions')
        if self.mode == 'stream':
            # one pass over the whole recording (train, then validation, then test), the state carried throughout
            parts = [a for a in (self.X_train, self.X_valid, self.X_test) if a is not None]
            lag = parts[0].shape[1] - 1
            out = self._run_stream(self._stream(np.concatenate(parts, 0)))[lag:].astype(np.float64)
            i = 0
            for name, a in (('train', self.X_train), ('valid', self.X_valid), ('test', self.X_test)):
                if a is not None:
                    setattr(self, f'predict_{name}', out[i:i + len(a)])
                    i += len(a)
            return
        if self.X_train is not None:
            self.predict_train = self._predict(self.X_train)
        if self.X_valid is not None:
            self.predict_valid = self._predict(self.X_valid)
        if self.X_test is not None:
            self.predict_test = self._predict(self.X_test)

    def save(self):
        if self.method == 'predict':
            print('Model was loaded from file; not saving it again')
            return
        print('Saving model')
        torch.save({'state_dict': self.model.state_dict(), 'net_kwargs': self.net_kwargs, 'mode': self.mode,
                    'n_inputs': self.X_train.shape[2], 'n_outputs': self.y_train.shape[1]},
                   self.output_pathnames['model_pt'])

    def load(self):
        print('Loading model')
        ck = torch.load(self.input_pathnames['model_pt'], map_location=self.device, weights_only=False)
        self.model = LSNNNet(ck['n_inputs'], ck['n_outputs'], **ck['net_kwargs']).to(self.device)
        self.model.eval()
        self.mode = ck.get('mode', 'window')
        self.model.load_state_dict(ck['state_dict'])


if __name__ == '__main__':
    LSNN(sys.argv).run()
