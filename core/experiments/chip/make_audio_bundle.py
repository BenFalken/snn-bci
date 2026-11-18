"""
Build the batch for the Speck laptop: a small self-contained folder with the snndec code and configs, the latency
harness (every decoder of the three experiments, no data needed), the audio chip runner, and per audio dataset only
what the run reads (the per-frame chip input, the test targets, the split lengths) plus its checkpoint. Copy the
folder to the laptop, plug in the devkit, and run `bash run.sh`.

    python experiments/chip/make_audio_bundle.py --out ~/Downloads/speck_laptop_batch1
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from snndec.convert import load  # noqa: E402
from snndec.evaluate import recording_file  # noqa: E402

AUDIO = os.path.expanduser('~/Documents/GitHub/AuditoryDecoding/pipeline_data/Neil/aSTG')
SETS = {d: f'{REPO}/runs/speck_audio_binB/{d}_r13_binB_s1337/model.pt'
        for d in ('07_19_11', '07_20_10', '07_31_13', '08_03_10', '08_06_13')}
SETS['ALLWORDS_VERIFIED_DAYNORM'] = f'{REPO}/runs/audio_speck_words_s1337/best.pt'

RUN_SH = '''#!/bin/bash
# On the Speck laptop. Two parts, results in results/ (send that folder back):
#   1. latency.json      every decoder of motor, audio and speech timed on this CPU, one thread (no devkit needed)
#   2. audio_chip_l1x4.json   the audio Speck SNN on the devkit: 5 days and the verified words, scored float,
#                        emulated, event-simulated and on the chip with layer 1's threshold x4 (the fix for the
#                        chip's event-by-event firing), with the chip's latency and measured power (resumable)
#   bash run.sh check      packages and devkit
#   bash run.sh            both parts (about 25 min); close other programs first, the timings use one core
#   bash run.sh latency    part 1 only
#   bash run.sh chip       part 2 only
#   bash run.sh nochip     part 1, then part 2 without the devkit
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ "${1:-}" = check ]; then
    $PY -c "import torch, sinabs, numpy, yaml; print('torch', torch.__version__, '| sinabs', sinabs.__version__)"
    $PY -c "import samna; print('samna', samna.__version__); print('devices:', [d.device_type_name for d in samna.device.get_unopened_devices()])"
    exit 0
fi
mkdir -p results
[ "${1:-}" = chip ] || OMP_NUM_THREADS=1 $PY -W ignore code/experiments/chip/latency_all.py --out results/latency.json
[ "${1:-}" = latency ] && exit 0
ARGS=(); for d in data/*/; do ARGS+=("${d}pipeline=${d}model.pt"); done
$PY -W ignore code/experiments/chip/audio_chip.py --data "${ARGS[@]}" --out results/audio_chip_l1x4.json --layer1-scale 4 \\
    $([ "${1:-}" = nochip ] && echo --nochip)
echo "Done: send back the results/ folder."
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    code = f'{out}/code'
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(f'{REPO}/snndec', f'{code}/snndec', ignore=shutil.ignore_patterns('__pycache__'))
    os.makedirs(f'{code}/experiments/chip', exist_ok=True)
    for f in ('audio_chip.py', 'latency_all.py'):
        shutil.copy(f'{REPO}/experiments/chip/{f}', f'{code}/experiments/chip/')
    shutil.copytree(f'{REPO}/configs', f'{code}/configs')
    for name, ckpt in SETS.items():
        src, dst = f'{AUDIO}/{name}/pipeline', f'{out}/data/{name}/pipeline'
        os.makedirs(dst, exist_ok=True)
        _, info = load(ckpt)
        feats = recording_file(info)
        shutil.copy(f'{src}/{feats}', dst)
        shutil.copy(f'{src}/window_targets_test.npy', dst)
        n = {s: len(np.load(f'{src}/window_targets_{s}.npy', mmap_mode='r')) for s in ('train', 'valid', 'test')}
        W = int(np.load(f'{src}/window_features_train.npy', mmap_mode='r').shape[1])
        json.dump({'n': n, 'W': W}, open(f'{dst}/lengths.json', 'w'))
        shutil.copy(ckpt, f'{out}/data/{name}/model.pt')
    with open(f'{out}/run.sh', 'w') as f:
        f.write(RUN_SH)
    total = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(out) for x in fs)
    print(f'wrote {out}: {len(SETS)} datasets, {total / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
