"""
Table of a sweep's runs (run_snn_sweep.sbatch, run_snn_diagnostics.sbatch),
best test PER first; also written to <sweep dir>/summary.csv. Works while the
sweep is still running (uses each run's newest checkpoint).

    python summarize_sweep.py $BASE_DIR/derived/snns/sweep
"""

import csv
import glob
import os
import sys

import torch


def newest_history(run_dir):
    """(history, finished) from the run's checkpoint with the longest history."""
    best, finished = None, False
    for path in glob.glob(os.path.join(run_dir, '*.pth')):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:  # a checkpoint being written right now
            print(f'  [skip] {path}: {e}')
            continue
        history = ckpt.get('history')
        if history and (best is None or len(history['test_per']) > len(best['test_per'])):
            best = history
        finished = finished or os.path.basename(path).startswith('final_model_')
    return best, finished


def main(sweep_dir):
    rows = []
    for run_dir in sorted(d for d in glob.glob(os.path.join(sweep_dir, '*')) if os.path.isdir(d)):
        history, finished = newest_history(run_dir)
        flags_file = os.path.join(run_dir, 'flags.txt')
        flags = open(flags_file).read().strip() if os.path.isfile(flags_file) else ''
        if not history or not history['test_per']:
            rows.append({'run': os.path.basename(run_dir), 'status': 'no epochs yet', 'flags': flags})
            continue
        per, loss = history['test_per'], history['test_loss']
        rates = history.get('test_spike_rate', [])
        best = min(range(len(per)), key=per.__getitem__)
        rows.append({
            'run': os.path.basename(run_dir),
            'status': 'done' if finished else 'running',
            'epochs': len(per),
            'best_per': per[best], 'best_epoch': best + 1,
            'last_per': per[-1],
            'best_test_loss': min(loss), 'last_test_loss': loss[-1],
            'last_train_loss': history['train_loss'][-1],
            'last_spike_rate': rates[-1] if rates else float('nan'),
            'flags': flags,
        })

    rows.sort(key=lambda r: r.get('best_per', 9.0))
    print(f"{'run':32s} {'status':8s} {'ep':>3s} {'bestPER':>8s} {'@ep':>4s} {'lastPER':>8s} "
          f"{'testLoss':>9s} {'rate':>6s}  flags")
    for r in rows:
        if 'best_per' not in r:
            print(f"{r['run']:32s} {r['status']}")
            continue
        print(f"{r['run']:32s} {r['status']:8s} {r['epochs']:3d} {r['best_per']:8.3f} {r['best_epoch']:4d} "
              f"{r['last_per']:8.3f} {r['last_test_loss']:9.2f} {r['last_spike_rate']:6.3f}  {r['flags']}")

    out = os.path.join(sweep_dir, 'summary.csv')
    fields = ['run', 'status', 'epochs', 'best_per', 'best_epoch', 'last_per', 'best_test_loss',
              'last_test_loss', 'last_train_loss', 'last_spike_rate', 'flags']
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
