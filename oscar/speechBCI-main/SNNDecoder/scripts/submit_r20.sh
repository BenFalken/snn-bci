#!/bin/bash
# Round 20 on Oscar, from SNNDecoder/scripts:  bash submit_r20.sh
# Writes configs/speck_r20_V2C.json (the three chains) from round 17's ranking, then submits V2C (1 GPU).
set -eo pipefail
B=/users/$USER/scratch/speechBCI_dryad
mkdir -p snn_logs
for f in configs/channels/r3/top128.txt configs/channels/r3/top64.txt configs/channels/r3/all256_top128first.txt; do
    [ -f "$f" ] || { echo "ERROR: $f missing" >&2; exit 1; }
done
source ~/miniconda3/etc/profile.d/conda.sh; conda activate speechBCI
python - <<'PY'
import json
rd = lambda f: [int(x) for x in open(f).read().split()]
top, top64, order = rd('configs/channels/r3/top128.txt'), rd('configs/channels/r3/top64.txt'), \
    rd('configs/channels/r3/all256_top128first.txt')
assert len(top) == 128 and len(top64) == 64 and set(top64) <= set(top), 'top64 must be a subset of top128'
assert order[:128] == top, 'all256_top128first.txt must start with top128.txt'
spec = [dict(channels=top, delays=[0, 4, 13, 25], dims=[128, 256, 64]),
        dict(channels=order[128:], delays=[0, 4, 13, 25], dims=[128, 256, 64]),
        dict(channels=top64, delays=[0, 15, 30, 60], dims=[64, 256, 64])]
json.dump(spec, open('configs/speck_r20_V2C.json', 'w'))
print('wrote configs/speck_r20_V2C.json: chains of', [len(g['channels']) * len(g['delays']) for g in spec], 'inputs')
PY
R=$(sbatch --parsable --export=ALL,BASE_DIR=$B,RUNS=configs/speck_r20.txt,PACK=1,SESSIONS=all,EPOCHS=96 \
    --array=0-0 run_snn_configs.sbatch)
echo "round 20 job $R (log snn_logs/SNN_RUN_${R}_0.out); checkpoint $B/derived/snns/speck_r20/00_V2C_1chip_3chains/"
