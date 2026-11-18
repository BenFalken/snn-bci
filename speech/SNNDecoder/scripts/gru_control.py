"""
Control for SNN training: a causal GRU trained by the SNN's own pipeline (same
.npz data, per-session input layers, augmentation, CTC loss, greedy PER), so a
failure to learn can be pinned on the SNN or on the pipeline/data:

    GRU learns, SNN does not   -> the SNN's architecture or settings
    neither learns             -> data, augmentation or training loop

    python gru_control.py --data-dir $BASE_DIR/derived/snnData --sessions t12.2022.06.02 ... --epochs 25
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from snnDecoder.dataset import SessionBatches, augment, load_partitions, smooth_inputs
from snnDecoder.train import ctc_loss
from snnDecoder.utils import edit_distance, greedy_decode

BLANK = 40


class GRUControl(torch.nn.Module):
    def __init__(self, n_sessions, n_in=256, hidden=512, layers=2, n_classes=41):
        super().__init__()
        self.day_layers = torch.nn.ModuleList([torch.nn.Linear(n_in, n_in) for _ in range(n_sessions)])
        for layer in self.day_layers:
            torch.nn.init.eye_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.gru = torch.nn.GRU(n_in, hidden, layers, batch_first=True)
        self.out = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, day_idx):
        return self.out(self.gru(F.softsign(self.day_layers[day_idx](x)))[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--sessions', nargs='+', required=True)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--white-noise-sd', type=float, default=1.0)
    p.add_argument('--constant-offset-sd', type=float, default=0.2)
    p.add_argument('--smooth-sd', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = SessionBatches(load_partitions(args.data_dir, args.sessions, 'train'), args.batch_size, True, args.seed)
    test = SessionBatches(load_partitions(args.data_dir, args.sessions, 'test'), args.batch_size, False)
    model = GRUControl(len(args.sessions)).to(device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-3)
    print(f'GRU control | device {device} | {train.n_trials} train, {test.n_trials} test trials | '
          f'noise {args.white_noise_sd}, offset {args.constant_offset_sd}, smoothing sd {args.smooth_sd}')

    for epoch in range(1, args.epochs + 1):
        t0, model_loss, n = time.time(), 0.0, 0
        model.train()
        for b in train:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            x = smooth_inputs(augment(b['x'], args.white_noise_sd, args.constant_offset_sd), args.smooth_sd)
            loss = ctc_loss(model(x, b['day_idx']), b, BLANK)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model_loss, n = model_loss + loss.item(), n + 1

        model.eval()
        edit = phones = 0
        with torch.no_grad():
            for b in test:
                logits = model(smooth_inputs(b['x'].to(device), args.smooth_sd), b['day_idx']).cpu().numpy()
                targets = np.split(b['targets'].numpy(), np.cumsum(b['target_lengths'].numpy())[:-1])
                for i, (length, target) in enumerate(zip(b['lengths'].numpy(), targets)):
                    edit += edit_distance(greedy_decode(logits[i, :length], BLANK), target)
                    phones += len(target)
        print(f'Epoch {epoch}/{args.epochs} ({time.time() - t0:.1f}s) | train loss {model_loss / n:.4f} | '
              f'test PER {edit / phones:.4f}', flush=True)


if __name__ == '__main__':
    main()
