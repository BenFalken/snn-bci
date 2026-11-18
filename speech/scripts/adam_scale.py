#How much of Adam's step size does epsilon decide? The RNN recipe uses Adam with eps 0.1 (the paper RNN's); an
#update is lr * m / (sqrt(v) + eps), so where a weight's gradient RMS sqrt(v) is well below eps the update is plain
#SGD at lr / eps and Adam's per-weight scaling is gone. Reads the optimizer state saved in the training checkpoints
#(every 5 epochs and at the end) and prints, per parameter tensor, the median gradient RMS and the share of the
#denominator that is eps (1.0: all eps, i.e. SGD; 0.0: pure Adam).
#
#   python adam_scale.py $B/derived/snns/readall96/*_R_readall96/checkpoint_*_epoch5.pth \
#                        $B/derived/snns/readall96/*_R_readall96/final_model_*.pth
import argparse

import torch


def names_for_state(model_sd, groups):
    """Optimizer state ids -> parameter names. The optimizer's groups are train.py's: weights, then learned
    thresholds, time constants and delay positions, each in named_parameters order; buffers in the state dict have
    no optimizer state and are skipped by matching shapes in order."""
    keys = list(model_sd)
    kinds = [lambda k: not k.endswith('log_threshold') and '.log_tau_' not in k and not k.endswith('_delay.pos'),
             lambda k: k.endswith('log_threshold'), lambda k: '.log_tau_' in k, lambda k: k.endswith('_delay.pos')]
    out = {}
    present = [kd for kd in kinds if any(kd(k) for k in keys)]
    for g, kind in zip(groups, present):
        cand = [k for k in keys if kind(k)]
        i = 0
        for pid, shape in g:
            while i < len(cand) and tuple(model_sd[cand[i]].shape) != shape:
                i += 1                                   # a buffer (no gradient) of another shape
            out[pid] = cand[i] if i < len(cand) else f'param {pid}'
            i += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    a = p.parse_args()
    for path in a.checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        osd = ck['optimizer_state_dict']
        st = osd['state']
        groups = [[(pid, tuple(st[pid]['exp_avg'].shape)) for pid in g['params'] if pid in st]
                  for g in osd['param_groups']]
        names = names_for_state(ck['model_state_dict'], groups)
        print(f'\n{path}\n  epoch {ck.get("epoch", -1) + 1}')
        print(f'  {"parameter":46s} {"shape":>16s} {"lr":>9s} {"eps":>7s} {"median RMS":>11s} {"90th pct":>9s} '
              f'{"eps share":>9s}')
        for g in osd['param_groups']:
            b2, eps = g['betas'][1], g['eps']
            for pid in g['params']:
                if pid not in st:
                    continue
                s = st[pid]
                step = float(s['step'])
                rms = (s['exp_avg_sq'] / (1 - b2 ** step)).sqrt().flatten().float()
                share = (eps / (rms + eps)).median()
                q = torch.quantile(rms[torch.randperm(len(rms))[:1_000_000]], torch.tensor([0.5, 0.9]))
                name = names.get(pid, 'param %d' % pid)
                print(f'  {name[:46]:46s} {str(tuple(s["exp_avg"].shape)):>16s} {g["lr"]:9.2e} {eps:7.0e} '
                      f'{q[0]:11.2e} {q[1]:9.2e} {share:9.3f}')


if __name__ == '__main__':
    main()
