#!/bin/bash
# EXPLORATORY, take 2 of the "next-generation Speck" speech networks (configs/speck_ng2.txt), on Oscar, from
# speech/scripts:   bash submit_speck_ng2.sh
# Writes the branch specs (more chains of V2C's per-core shape), then trains both networks, one GPU each, in parallel.
# Checkpoints: $BASE_DIR/derived/snns/speck_ng2/<NN>_<name>/. A run that hits the 24 h limit resumes when re-run.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
B=$SNNBCI_OSCAR_SCRATCH
mkdir -p snn_logs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate speechBCI
python - <<'PY'
import json
rd = lambda f: [int(x) for x in open(f).read().split()]
top, top64, order = rd('configs/channels/r3/top128.txt'), rd('configs/channels/r3/top64.txt'), \
    rd('configs/channels/r3/all256_top128first.txt')
assert order[:128] == top and set(top64) <= set(top)
CH = {'A': top, 'B': order[128:], 'C': top64}
D1, D2, D3, D4, CL, CL2 = [0, 4, 13, 25], [2, 8, 18, 35], [1, 6, 10, 20], [3, 15, 30, 45], [0, 15, 30, 60], [45, 60, 75, 90]
LAYOUTS = {'NG2b_5chains': [('A', D1), ('B', D1), ('A', D2), ('B', D2), ('C', CL)],
           'NG4b_10chains': [('A', D1), ('B', D1), ('A', D2), ('B', D2), ('A', D3), ('B', D3), ('A', D4), ('B', D4),
                             ('C', CL), ('C', CL2)]}
for name, lay in LAYOUTS.items():
    spec = [dict(channels=CH[c], delays=d, dims=[64, 256, 64] if c == 'C' else [128, 256, 64]) for c, d in lay]
    json.dump(spec, open(f'configs/speck_ng2_{name}.json', 'w'))
    print(f'wrote configs/speck_ng2_{name}.json: {len(spec)} chains of',
          [len(g['channels']) * len(g['delays']) for g in spec], 'inputs')
PY
R=$(sbatch --parsable --export=ALL,BASE_DIR=$B,RUNS=configs/speck_ng2.txt,PACK=1,SESSIONS=all,EPOCHS=96 \
    --array=0-1%2 run_snn_configs.sbatch)
echo "next-generation Speck, take 2: job $R (logs snn_logs/SNN_RUN_${R}_{0,1}.out); checkpoints $B/derived/snns/speck_ng2/"
echo "progress:  grep -h '^Epoch' snn_logs/SNN_RUN_${R}_0.out | tail -3     (V2C's log metric ended at test PER 0.4435)"
