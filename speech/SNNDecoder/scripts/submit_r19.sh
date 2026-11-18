#!/bin/bash
# Round 19 on Oscar, from SNNDecoder/scripts:  bash submit_r19.sh
# 1. rebuilds and verifies derived/snnData (rebuild_snndata.sbatch), unless it already verifies
# 2. writes configs/channels/r3/all256_top128first.txt from top128.txt
# 3. submits round 19 (4 runs, PACK=2, 2 GPUs), held until the rebuild succeeds
set -eo pipefail
B=/users/$USER/scratch/speechBCI_dryad
mkdir -p snn_logs
[ -f configs/channels/r3/top128.txt ] || { echo "ERROR: configs/channels/r3/top128.txt missing (round 17's ranking)" >&2; exit 1; }
python3 - <<'PY'
top = [int(x) for x in open('configs/channels/r3/top128.txt').read().split()]
assert len(top) == 128 and len(set(top)) == 128
rest = [c for c in range(256) if c not in set(top)]
open('configs/channels/r3/all256_top128first.txt', 'w').write('\n'.join(map(str, top + rest)) + '\n')
print('wrote configs/channels/r3/all256_top128first.txt: top 128, then', len(rest), 'others')
PY
DEP=()
source ~/miniconda3/etc/profile.d/conda.sh; conda activate speechBCI
if python verify_snndata.py $B/derived/snnData >/dev/null 2>&1; then
    echo "snnData already verifies: no rebuild"
else
    J=$(sbatch --parsable --export=ALL,BASE_DIR=$B${TF_ENV:+,TF_ENV=$TF_ENV} rebuild_snndata.sbatch)
    echo "rebuild job $J (log snn_logs/SNNDATA_$J.out)"
    DEP=(--dependency=afterok:$J)
fi
R=$(sbatch --parsable "${DEP[@]}" --export=ALL,BASE_DIR=$B,RUNS=configs/speck_r19.txt,PACK=2,SESSIONS=all,EPOCHS=96 \
    --array=0-1 run_snn_configs.sbatch)
echo "round 19 job $R (array 0-1; logs snn_logs/SNN_RUN_${R}_task*.out); checkpoints $B/derived/snns/speck_r19/"
