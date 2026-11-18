"""
The runs behind the Speck memory figure, for motor, audio and speech: one line per training run (JSON lines), in two
phases.

On Speck the first layer's input is the chip's widest point (480-512 inputs for a 128-wide first layer), and the host
fills it with channels x input delays. So channels trade against memory: halving the channels doubles the delays, and
with a fixed spacing between delays (40 ms in every experiment) that doubles how far back the chip sees.

    phase A  all channels (unit = session / day / the pooled speech sessions, x seeds):
               hid_D1     no delays, the host reads the last hidden layer (the "synaptic skip")
               out_D1     no delays, a spiking output layer on the chip, its spikes decoded on the host
               out_Dmin   the budget's delays with all channels, output-layer readout
               hid_Dmin   the budget's delays with all channels, hidden readout (also the sweep's first point)
    phase B  fewer channels, more delays (hidden readout), channels = the top ones of the unit's hid_Dmin seed-0
             model (snndec.rank: first-layer weight norm), so phase B needs phase A's results

    budget   motor 480 (96 channels x 5 ... 8 x 60), audio 480 (96 x 5 ... 8 x 60), speech 512 (128 x 4 ... 8 x 64)

    python experiments/speck_memory/plan.py --out RUNS_DIR --motor-root .../snn_datasets/bmi/indy/mua_8_group \
        --scalers .../velocity_scalers.json --audio-root .../pipeline_data/Neil/aSTG --speech-data .../derived/snnData

writes RUNS_DIR/plan_<experiment>_<A|B>.jsonl.
"""

import argparse
import glob
import json
import os

SPACING_MS = 40.0
SEEDS = [0, 1, 2]
GRID = {'motor': (96, [96, 48, 32, 24, 16, 8], 480, 4.0, 'motor_speck'),
        'audio': (96, [96, 48, 32, 24, 16, 8], 480, 40.0, 'audio_speck'),
        'speech': (128, [128, 64, 32, 16, 8], 512, 20.0, 'speech_speck')}
AUDIO_DAYS = ['07_31_13', '08_03_10', '07_20_10', '08_06_13']


def variants(exp):
    """(name, phase, channels, delays, readout) of one unit."""
    cmax, cs, budget, step_ms, _ = GRID[exp]
    dmin = budget // cmax
    out = [('hid_D1', 'A', cmax, 1, 'hidden'), ('out_D1', 'A', cmax, 1, 'output'),
           (f'out_D{dmin}', 'A', cmax, dmin, 'output'), (f'hid_D{dmin}', 'A', cmax, dmin, 'hidden')]
    out += [(f'hid_C{c}_D{budget // c}', 'B', c, budget // c, 'hidden') for c in cs if c < cmax]
    return out


def units(exp, a):
    if exp == 'motor':
        sessions = sorted(os.path.basename(d) for d in glob.glob(f'{a.motor_root}/*') if os.path.isdir(f'{d}/test'))
        k = a.motor_sessions
        pick = [sessions[round(i * (len(sessions) - 1) / (k - 1))] for i in range(k)] if len(sessions) > k else sessions
        return [(s, f'{a.motor_root}/{s}') for s in pick]
    if exp == 'audio':
        return [(d, f'{a.audio_root}/{d}/pipeline') for d in AUDIO_DAYS]
    return [('pooled', a.speech_data)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--motor-root')
    p.add_argument('--motor-sessions', type=int, default=8)
    p.add_argument('--scalers')
    p.add_argument('--audio-root')
    p.add_argument('--speech-data')
    p.add_argument('--experiments', nargs='+', default=['motor', 'audio', 'speech'])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for exp in a.experiments:
        cmax, _, budget, step_ms, config = GRID[exp]
        sp = int(round(SPACING_MS / step_ms))                      # the spacing in input steps
        rows = {'A': [], 'B': []}
        for unit, path in units(exp, a):
            for name, phase, c, d, readout in variants(exp):
                for seed in SEEDS:
                    rows[phase].append(dict(
                        experiment=exp, unit=unit, data=path, scalers=a.scalers, config=config, variant=name,
                        readout=readout, channels=c, n_delays=d, delays=[k * sp for k in range(d)], seed=seed,
                        spacing_ms=SPACING_MS, span_ms=(d - 1) * SPACING_MS, inputs=c * d,
                        rank_from=None if phase == 'A' else f'{exp}/{unit}/hid_D{budget // cmax}_s0',
                        out=f'{exp}/{unit}/{name}_s{seed}'))
        for phase, r in rows.items():
            path = f'{a.out}/plan_{exp}_{phase}.jsonl'
            with open(path, 'w') as f:
                f.writelines(json.dumps(x) + '\n' for x in r)
            print(f'{path}: {len(r)} runs')


if __name__ == '__main__':
    main()
