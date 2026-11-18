"""
Train and evaluate one run of a speck_memory plan (plan.py), then measure its memory:

    python chip/memory_sweep/run.py RUNS_DIR/plan_motor_A.jsonl INDEX [INDEX ...] --root RUNS_DIR

For each run: the experiment's Speck config (configs/<config>.yaml) with the run's channels, delays, readout and seed;
trained with snndec.train (the experiment's own recipe); evaluated on the test split with the experiment's metric;
the silence response (snndec.memory) on test data; the chip mapping checked. Writes RUNS_DIR/<out>/best.pt and
result.json; a run whose result.json exists is skipped.
"""

import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from snndec import data, evaluate as ev  # noqa: E402
from snndec.convert import load  # noqa: E402
from snndec.memory import sample_segments, silence_response  # noqa: E402
from snndec.rank import rank_channels  # noqa: E402
from snndec.train import train  # noqa: E402

N_OUT = {'motor': 2, 'audio': 128, 'speech': 41}
# the silence response: steps before t0, and steps after it (input steps): ~1 s and 3-4 s
MEMORY = {'motor': (250, 750), 'audio': (25, 75), 'speech': (25, 200)}
K = 256                                                                     # moments silenced per run
# Chip fit depends only on the shapes (inputs, layer widths), not the weights. Every shape the sweep builds was mapped
# with sinabs' DynapcnnNetwork for the speck2fdevkit on 2026-11-05 (sinabs 3.1.0, samna 0.48): all fit. Batch jobs
# use this table instead of the mapper, whose samna import tried to pip-install samna into the shared env on Oscar.
# SNNDEC_CHECK_CHIP=1 runs the mapper anyway.
CHIP_VERIFIED = {(n_in, tuple([128, 256, 256] + out)) for n_in in (96, 480, 128, 512) for out in ([], [36], [128], [41])}


def build_config(row, root):
    cfg = yaml.safe_load(open(f"{REPO}/snndec/configs/{row['config']}.yaml"))
    m, exp = cfg['model'], row['experiment']
    cfg['train']['seed'] = row['seed']
    inp = m.setdefault('input', {})
    for k in ('n_channels', 'n_delays', 'delay_max'):                     # audio: set explicitly below
        inp.pop(k, None)
    inp['delays'] = row['delays']
    cmax = {'motor': 96, 'audio': 96, 'speech': 128}[exp]
    if row['channels'] < cmax:
        src, info = load(f"{root}/{row['rank_from']}/best.pt")
        inp['channels'] = rank_channels(src)[:row['channels']]
    elif exp == 'speech':
        inp['channels'] = list(range(128))                               # the 128 threshold-crossing features
    else:
        inp.pop('channels', None)
    n = N_OUT[exp]
    if row['readout'] == 'hidden':
        m['layers']['hidden'] = [128, 256, 256]
        if exp == 'motor':
            m['readout'] = dict(type='ema', read='last', decay=0.8, stages=2, head='linear', normalize=True,
                                bias_init='neutral')              # starts at zero velocity, as the population readout
    else:                                                                   # a spiking output layer on the chip
        m['layers']['hidden'] = [128, 256, 256, 36 if exp == 'motor' else n]
        if exp == 'audio':
            m['readout'] = dict(type='ema', read='last', decay=0.8, stages=2, head='affine', normalize=True)
        elif exp == 'speech':
            m['readout'] = dict(type='leaky', read='last', tau_ms=40, stages=1, init='zero', head='affine')
    return cfg


@torch.no_grad()
def memory(model, row, cfg):
    exp, (burn, length) = row['experiment'], MEMORY[row['experiment']]
    if exp == 'motor':
        x = data.motor_split(row['data'], 'test', row['scalers'])[0].x
        r = silence_response(model, sample_segments(x, K, burn, length), burn)
    elif exp == 'audio':
        rec, W, n = data.audio_recording(row['data'])
        r = silence_response(model, sample_segments(rec[n['train'] + n['valid']:], K, burn, length), burn)
    else:
        trials = [t for t in data.speech_split(row['data'], cfg['data']['sessions'], 'test') if len(t.x) >= burn + length]
        rng, rs, ws = np.random.default_rng(0), [], []
        for day in sorted({t.day for t in trials}):
            xs = [t.x for t in trials if t.day == day][:K // 16]
            segs = np.stack([x[s:s + burn + length] for x in xs for s in [rng.integers(0, len(x) - burn - length + 1)]])
            rs.append(silence_response(model, segs, burn, day))
            ws.append(len(segs))
        r = {k: np.average(np.stack([q[k] for q in rs]), axis=0, weights=ws) for k in rs[0]}
    out_ms = model.out_every * model.dt
    return {k: v.tolist() for k, v in r.items()}, (np.arange(len(r['kernel'])) * out_ms).tolist()


def run_one(row, root, device, overrides=()):
    out = f"{root}/{row['out']}"
    if os.path.isfile(f'{out}/result.json'):
        print(f"[skip] {row['out']}: done")
        return
    os.makedirs(out, exist_ok=True)
    cfg = build_config(row, root)
    for kv in overrides:                                                    # e.g. train.epochs=1 (tests)
        k, v = kv.split('=', 1)
        d = cfg
        *path, last = k.split('.')
        for q in path:
            d = d[q]
        d[last] = yaml.safe_load(v)
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
    t0 = time.time()
    model = train(cfg, SimpleNamespace(data=[row['data']], scalers=row['scalers'], out=out, init=None, device=device))
    model.eval()
    if row['experiment'] == 'motor':
        metrics, _ = ev.evaluate_motor(model, row['data'], row['scalers'])
    elif row['experiment'] == 'audio':
        metrics, _ = ev.evaluate_audio(model, row['data'], 'recording')
    else:
        metrics, _ = ev.evaluate_speech(model, row['data'], cfg['data']['sessions'], 'test')
    resp, lag_ms = memory(model, row, cfg)
    shape = (model.host.n_out, tuple(model.widths))
    if os.environ.get('SNNDEC_CHECK_CHIP') == '1':
        try:
            chip = bool(model.check_chip())
        except Exception as e:                                              # noqa: BLE001
            chip = f'no: {str(e).splitlines()[0][:80]}'
    else:
        chip = True if shape in CHIP_VERIFIED else f'not checked: {shape}'
    res = {**row, 'metrics': metrics, 'silence_response': resp, 'lag_ms': lag_ms, 'chip_maps': chip,
           'channels_used': model.host.channels, 'chip_inputs': model.host.n_out, 'widths': model.widths,
           'parameters': sum(p.numel() for p in model.parameters()),
           'buffer_values': model.host.n_in * max(model.host.delays),       # host memory: past samples held
           'train_minutes': (time.time() - t0) / 60}
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print(f"[done] {row['out']}: {metrics} | chip {chip} | {res['train_minutes']:.0f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('plan')
    p.add_argument('index', type=int, nargs='+')
    p.add_argument('--root', required=True)
    p.add_argument('--device', default=None)
    p.add_argument('--set', nargs='*', default=[], help='config overrides, e.g. train.epochs=1 (tests only)')
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.plan)]
    for i in a.index:
        if i < len(rows):
            run_one(rows[i], a.root, a.device, a.set)


if __name__ == '__main__':
    main()
