"""
The AEGRU of Wang, Wang & Liu (BioCAS 2024, arXiv:2410.22283) and its plain-GRU ablation, re-implemented from the
paper (no code was released).

Inference path (paper Fig. 1): FC_upstream (C_in -> 32) + batch norm -> GRU (32) over the N sub-windows -> FC (32 -> 2),
read at the last sub-window. Training adds the auxiliary branch: log sigma^2 = FC2(ReLU(FC1(x'))) (C_in -> 168 -> 32),
f ~ N(mu, sigma^2) with mu from FC_upstream, and a reconstruction r = exp(FC3(f)) of the sub-window spike counts,
scored by Poisson NLL. Loss = w_v * MSE(velocity) + w_x * PoissonNLL.

Choices the paper leaves open (documented in README.md): the GRU is fed the sampled f in training and mu at inference;
w_v = 1 and w_x, the batch size and velocity z-scoring are set here.
The plain GRU ("vanilla GRU" in the paper's Table III) is the GRU layer and the output FC alone, fed x' directly.
"""
import torch
from torch import nn


class AEGRU(nn.Module):
    def __init__(self, n_in, n_latent=32, n_hidden=32, n_var_hidden=168):
        super().__init__()
        self.fc_up = nn.Linear(n_in, n_latent)
        self.bn = nn.BatchNorm1d(n_latent)
        self.gru = nn.GRU(n_latent, n_hidden, batch_first=True)
        self.fc_out = nn.Linear(n_hidden, 2)
        self.fc1 = nn.Linear(n_in, n_var_hidden)      # auxiliary, training only
        self.fc2 = nn.Linear(n_var_hidden, n_latent)
        self.fc3 = nn.Linear(n_latent, n_in)

    def _mu(self, x):                                  # x: (B, N, C_in)
        mu = self.fc_up(x)
        return self.bn(mu.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        """Returns (velocity (B, 2), log_rate (B, N, C_in) or None)."""
        mu = self._mu(x)
        if not self.training:
            h, _ = self.gru(mu)
            return self.fc_out(h[:, -1]), None
        logvar = self.fc2(torch.relu(self.fc1(x)))
        f = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        h, _ = self.gru(f)
        return self.fc_out(h[:, -1]), self.fc3(f)

    def inference_params(self):
        """Parameters on the inference path (the paper's 45.5 kB footprint counts these)."""
        mods = (self.fc_up, self.bn, self.gru, self.fc_out)
        return sum(p.numel() for m in mods for p in m.parameters())


class PlainGRU(nn.Module):
    def __init__(self, n_in, n_hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, n_hidden, batch_first=True)
        self.fc_out = nn.Linear(n_hidden, 2)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc_out(h[:, -1]), None

    def inference_params(self):
        return sum(p.numel() for p in self.parameters())


def macs_per_output(model, n_in, n_steps):
    """Dense multiply-accumulates per decoded output (no sparsity), for the op estimate."""
    gru = model.gru
    H, I = gru.hidden_size, gru.input_size
    m = n_steps * 3 * (I * H + H * H)                  # three gates, input and recurrent
    if isinstance(model, AEGRU):
        m += n_steps * n_in * model.fc_up.out_features
    return m + H * 2
