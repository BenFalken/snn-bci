"""
Report on SNN training runs from their Slurm logs (snn_logs/*.out): sweep,
diagnostics, pretrain and fine-tune jobs, including runs that were cancelled
or failed. Logs of the same run (resubmissions, resumes) are merged; a later
log's epochs replace an earlier one's.

    python report_runs.py snn_logs --ckpt-root $BASE_DIR/derived/snns --out snn_report

writes snn_report.md (overview table + one section per run) and
snn_report.json (everything, per epoch). With --ckpt-root it also reports
where a restart of each unfinished sweep run would resume.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

EPOCH_RE = re.compile(
    r'^Epoch (\d+)/(\d+) \(([\d.]+)s\) \| train loss ([-\d.naninf]+)'
    r'(?: \| \w+ loss ([-\d.naninf]+))? \| \w+ PER ([-\d.naninf]+)'
    r'(?: \| spike rate ([-\d.naninf]+))?(?: \| lr ([-\d.e]+))?')
LAYER_RE = re.compile(r'spikes per neuron per timestep by layer: \w+ \[([^\]]*)\], train \[([^\]]*)\]')
HEADER_RE = re.compile(r'^\[(\d+)\] ([^:]+): (.*)$')
PROLOG_RE = re.compile(r'^##\s+(Job ID|Job Name|Nodelist|Job Started)\s*:\s*(.*)$')
ERROR_MARKERS = ('Traceback', 'Error', 'CANCELLED', 'TIME LIMIT', 'out of memory', 'oom-kill', 'Killed')

LEARNING_PER, MARGINAL_PER = 0.85, 0.95


def floats(text):
    return [float(v) for v in text.replace(',', ' ').split()]


def parse_log(path):
    log = {'file': os.path.basename(path), 'epochs': {}, 'errors': [], 'lr_reductions': [],
           'resumed_from': None, 'finished': False, 'header': None, 'stage': None}
    lines = open(path, errors='replace').read().splitlines()
    last_epoch, skip_until = None, -1
    for i, line in enumerate(lines):
        s = line.strip()
        if i < skip_until:
            continue
        m = PROLOG_RE.match(s)
        if m:
            log[m.group(1).lower().replace(' ', '_')] = m.group(2).strip()
            continue
        m = HEADER_RE.match(s)
        if m and log['header'] is None:
            log['header'] = {'index': int(m.group(1)), 'name': m.group(2).strip(), 'rest': m.group(3)}
            continue
        if s.startswith('Pretraining') or s.startswith('Fine-tuning'):
            log['stage'] = s
        m = EPOCH_RE.match(s)
        if m:
            e = int(m.group(1))
            val = lambda g: float(m.group(g)) if m.group(g) is not None else None
            log['epochs'][e] = {'epoch': e, 'of': int(m.group(2)), 'time_s': val(3), 'train_loss': val(4),
                                'test_loss': val(5), 'test_per': val(6), 'spike_rate': val(7), 'lr': val(8)}
            log['planned_epochs'] = int(m.group(2))
            last_epoch = e
            continue
        m = LAYER_RE.search(s)
        if m and last_epoch is not None:
            log['epochs'][last_epoch]['layer_rates_test'] = floats(m.group(1))
            log['epochs'][last_epoch]['layer_rates_train'] = floats(m.group(2))
            continue
        if s.startswith('[resume]') or s.startswith('Resumed from'):
            log['resumed_from'] = s
        elif s.startswith('Learning rate reduced'):
            log['lr_reductions'].append({'after_epoch': last_epoch, 'line': s})
        elif s.startswith('Training finished') or s.startswith('[skip] finished'):
            log['finished'] = True
        elif s.startswith('Early stopping'):
            log['early_stopped'] = s
        if any(k in s for k in ERROR_MARKERS) and not s.startswith('Epoch'):
            if s.startswith('Traceback'):
                # through the exception line (the first unindented line after the traceback)
                end = next((j for j in range(i + 1, len(lines)) if lines[j] and not lines[j].startswith(' ')), i)
                context, skip_until = lines[i:end + 1], end + 1
            else:
                context = [line]
            log['errors'].append('\n'.join(context[-12:]).strip())
    if log.get('planned_epochs') and log['epochs'] and max(log['epochs']) >= log['planned_epochs']:
        log['finished'] = True   # the GRU control prints no end marker
    return log


def run_key(log):
    kind = log['file'].split('_')[1] if log['file'].count('_') >= 2 else 'job'   # SWEEP, DIAG, PRETRAIN, ...
    if log['header']:
        return f"{kind}/{log['header']['index']:02d}_{log['header']['name']}"
    m = re.match(r'(.+?)_(\d+)(?:_(\d+))?\.out$', log['file'])
    task = f"_{m.group(3)}" if m and m.group(3) else ''
    return f"{log.get('job_name') or kind}{task}"


def job_order(log):
    m = re.search(r'_(\d+)(?:_\d+)?\.out$', log['file'])
    return int(m.group(1)) if m else 0


def slope(values):
    if len(values) < 3:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def resume_point(ckpt_root, key):
    """Epoch a restart resumes from (newest periodic checkpoint) for sweep/diagnostics runs."""
    kind, _, name = key.partition('/')
    if kind == 'RUN' and '/' in name:
        # run_snn_configs.sbatch: "<NN>_<group>/<run>" -> <ckpt_root>/<group>/<NN>_<run>
        idx, _, rest = name.partition('_')
        group, _, run = rest.partition('/')
        sub, name = group, f'{idx}_{run}'
    else:
        sub = {'SWEEP': 'sweep', 'DIAG': 'diagnostics', 'ABL': 'ablation', 'C2': 'c2'}.get(kind)
    if not sub or not name:
        return None
    run_dir = os.path.join(ckpt_root, sub, name)
    if kind == 'DIAG':
        run_dir = os.path.join(ckpt_root, sub, name.split('_', 1)[1])
    ckpts = glob.glob(os.path.join(run_dir, 'checkpoint_*_epoch*.pth'))
    epochs = [int(re.search(r'epoch(\d+)\.pth$', c).group(1)) for c in ckpts]
    finished = bool(glob.glob(os.path.join(run_dir, 'final_model_*.pth')))
    flags_file = os.path.join(run_dir, 'flags.txt')
    return {'run_dir': run_dir, 'finished_checkpoint': finished,
            'resume_epoch': max(epochs) if epochs else 0,
            'flags': open(flags_file).read().strip() if os.path.isfile(flags_file) else None}


def summarize(key, logs, ckpt_root):
    logs = sorted(logs, key=job_order)
    epochs = {}
    for log in logs:
        epochs.update(log['epochs'])
    ep = [epochs[e] for e in sorted(epochs)]
    last = logs[-1]
    errors = [err for log in logs for err in log['errors']]
    finished = any(log['finished'] for log in logs)
    if finished:
        status = 'completed'
    elif any('TIME LIMIT' in e for e in last['errors']):
        status = 'time limit'
    elif any('CANCELLED' in e for e in last['errors']):
        status = 'cancelled'
    elif any(e.startswith('Traceback') for e in last['errors']):
        status = 'failed'
    else:
        status = 'incomplete (running or killed)'

    run = {'run': key, 'status': status, 'logs': [l['file'] for l in logs],
           'header': (last['header'] or {}).get('rest') or last.get('stage'),
           'node': last.get('nodelist'), 'started': logs[0].get('job_started'),
           'planned_epochs': max((l.get('planned_epochs') or 0) for l in logs) or None,
           'epochs_done': len(ep), 'resumed': [l['resumed_from'] for l in logs if l['resumed_from']],
           'lr_reductions': [r for l in logs for r in l['lr_reductions']],
           'errors': errors[-5:], 'per_epoch': ep}
    if ep:
        per = [e['test_per'] for e in ep]
        best = int(np.argmin(per))
        run.update({
            'best_per': per[best], 'best_epoch': ep[best]['epoch'], 'last_per': per[-1],
            'per_at': {n: ep[n - 1]['test_per'] for n in (1, 5, 10, 20, 30, 40) if n <= len(ep)},
            'per_slope_last5': slope(per[-5:]),
            'first_epoch_per_below_0.9': next((e['epoch'] for e in ep if e['test_per'] < 0.9), None),
            'first_epoch_per_below_0.7': next((e['epoch'] for e in ep if e['test_per'] < 0.7), None),
            'last_train_loss': ep[-1]['train_loss'], 'last_test_loss': ep[-1]['test_loss'],
            'best_test_loss': min((e['test_loss'] for e in ep if e['test_loss'] is not None), default=None),
            'last_spike_rate': ep[-1]['spike_rate'],
            'first_spike_rate': ep[0]['spike_rate'],
            'last_layer_rates_test': ep[-1].get('layer_rates_test'),
            'mean_epoch_s': float(np.mean([e['time_s'] for e in ep])),
        })
        run['verdict'] = ('learning' if run['best_per'] < LEARNING_PER else
                          'marginal' if run['best_per'] < MARGINAL_PER else
                          'stuck' if len(ep) >= 5 else 'too early')
        if run['first_spike_rate'] and run['last_spike_rate']:
            run['spike_rate_trend'] = ('rising' if run['last_spike_rate'] > 1.5 * run['first_spike_rate'] else
                                       'falling' if run['last_spike_rate'] < run['first_spike_rate'] / 1.5 else 'stable')
        if run['planned_epochs'] and status != 'completed':
            run['eta_remaining_min'] = (run['planned_epochs'] - len(ep)) * run['mean_epoch_s'] / 60
    else:
        run['verdict'] = 'no epochs'
    if ckpt_root:
        rp = resume_point(ckpt_root, key)
        if rp:
            run['restart'] = rp
            if rp['flags'] is not None:
                run['flags'] = rp['flags']
            if status != 'completed' and ep:
                run['restart']['epochs_lost'] = len(ep) - rp['resume_epoch']
    return run


def fmt(v, spec='.3f'):
    return '-' if v is None else format(v, spec)


def write_markdown(runs, path):
    out = ['# SNN training runs', '',
           f'{len(runs)} runs; verdict: learning = best test PER < {LEARNING_PER}, '
           f'marginal < {MARGINAL_PER}, else stuck (too early: under 5 epochs).', '',
           '| run | status | verdict | epochs | best PER (ep) | last PER | PER slope/ep (last 5) | '
           'test loss last | spike rate first→last | s/epoch | restart resumes at |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        restart = r.get('restart')
        resume = '-' if not restart or r['status'] == 'completed' else (
            f"epoch {restart['resume_epoch'] + 1} (loses {restart.get('epochs_lost', 0)})"
            if restart['resume_epoch'] else 'from scratch')
        out.append(
            f"| {r['run']} | {r['status']} | {r['verdict']} | {r['epochs_done']}/{r['planned_epochs'] or '?'} | "
            f"{fmt(r.get('best_per'))} ({r.get('best_epoch', '-')}) | {fmt(r.get('last_per'))} | "
            f"{fmt(r.get('per_slope_last5'), '+.4f')} | {fmt(r.get('last_test_loss'), '.2f')} | "
            f"{fmt(r.get('first_spike_rate'))}→{fmt(r.get('last_spike_rate'))} | "
            f"{fmt(r.get('mean_epoch_s'), '.0f')} | {resume} |")
    for r in runs:
        out += ['', f"## {r['run']}", '',
                f"- status: {r['status']}, verdict: {r['verdict']}, logs: {', '.join(r['logs'])}",
                f"- config: {r.get('flags') or r.get('header') or '-'}"]
        if r.get('per_at'):
            out.append('- PER at epoch: ' + ', '.join(f'{k}: {v:.3f}' for k, v in r['per_at'].items()))
            out.append(f"- first epoch with PER < 0.9: {r['first_epoch_per_below_0.9']}, < 0.7: "
                       f"{r['first_epoch_per_below_0.7']}")
            out.append(f"- spike rate trend: {r.get('spike_rate_trend', '-')}; last test rates by layer: "
                       f"{r.get('last_layer_rates_test')}")
        if r['lr_reductions']:
            out.append('- ' + '; '.join(f"after epoch {x['after_epoch']}: {x['line']}" for x in r['lr_reductions']))
        if r['resumed']:
            out.append('- ' + '; '.join(r['resumed']))
        if r.get('eta_remaining_min') is not None:
            out.append(f"- remaining epochs need ~{r['eta_remaining_min']:.0f} min")
        for err in r['errors']:
            out += ['- error:', '```', err, '```']
        if r['per_epoch']:
            out += ['', '| epoch | s | train loss | test loss | test PER | spike rate | lr |', '|---|---|---|---|---|---|---|']
            out += [f"| {e['epoch']} | {fmt(e['time_s'], '.0f')} | {fmt(e['train_loss'], '.2f')} | "
                    f"{fmt(e['test_loss'], '.2f')} | {fmt(e['test_per'], '.4f')} | {fmt(e['spike_rate'], '.3f')} | "
                    f"{fmt(e['lr'], '.1e')} |" for e in r['per_epoch']]
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log_dir', help='folder with the Slurm .out logs (snn_logs)')
    p.add_argument('--ckpt-root', default=None, help='$BASE_DIR/derived/snns, for flags and restart points')
    p.add_argument('--pattern', default='*.out', help="log file glob, e.g. 'SNN_SWEEP_*.out'")
    p.add_argument('--out', default='snn_report', help='writes <out>.md and <out>.json')
    args = p.parse_args()

    groups = {}
    for path in sorted(glob.glob(os.path.join(args.log_dir, args.pattern))):
        log = parse_log(path)
        groups.setdefault(run_key(log), []).append(log)
    runs = [summarize(k, v, args.ckpt_root) for k, v in groups.items()]
    runs.sort(key=lambda r: (r['run'].split('/')[0], r.get('best_per', 9.0)))

    write_markdown(runs, args.out + '.md')
    with open(args.out + '.json', 'w') as f:
        json.dump(runs, f, indent=1)
    print(f"{'run':40s} {'status':12s} {'verdict':9s} {'ep':>7s} {'bestPER':>8s} {'lastPER':>8s} {'rate':>6s}")
    for r in runs:
        print(f"{r['run']:40s} {r['status'][:12]:12s} {r['verdict']:9s} "
              f"{str(r['epochs_done']) + '/' + str(r['planned_epochs'] or '?'):>7s} "
              f"{fmt(r.get('best_per')):>8s} {fmt(r.get('last_per')):>8s} {fmt(r.get('last_spike_rate')):>6s}")
    print(f"\nsaved {args.out}.md and {args.out}.json")


if __name__ == '__main__':
    main()
