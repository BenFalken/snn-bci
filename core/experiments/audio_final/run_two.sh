#!/bin/bash
# 2026-11-18: audio Speck as two networks on one chip, 3 variants x 3 seeds x 2 monkeys; then the label-free layer-1
# sweep and the readout refit for each, a summary against P5, and the devkit bundle. From the repo root:
#   bash experiments/audio_final/run_two.sh > runs/audio_two.log 2>&1
AD=$HOME/Documents/GitHub/AuditoryDecoding/pipeline_data
SD=$(cd "$(dirname "$0")/../.." && pwd); cd "$SD"
OUT=$HOME/Downloads/final_results/audio_two; mkdir -p $OUT
train() {   # $1 = run list, $2 = parallel runs per monkey, $3 = threads each
  for m in Neil Gomez; do
    DATA=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline ROOT=runs/audio_two_$m P=$2 THREADS=$3 \
      LISTS=$1 bash experiments/audio_final/run_local.sh &
  done
  wait
}
echo "[wave 1 start: 3 variants, seed 1337] $(date +%H:%M)"
train experiments/audio_final/runs_speck_two_w1.txt 3 2
# the variant with the best validation score (best r_valid in training, mean over the two monkeys): no test data
V=$(python3 - <<'PY'
import re, glob, numpy as np
sc = {}
for v in ('A_blocks', 'B_interleave', 'C_tapsplit'):
    r = []
    for f in glob.glob(f'runs/audio_two_*/{v}_s1337/train.log'):
        x = [float(m) for m in re.findall(r'r_valid ([0-9.]+)', open(f).read())]
        if x: r.append(max(x))
    if len(r) == 2: sc[v] = float(np.mean(r))
print(max(sc, key=sc.get) if sc else 'none'); open('runs/audio_two_choice.txt', 'w').write(repr(sc))
PY
)
echo "[wave 1 done] $(date +%H:%M): validation r per variant $(cat runs/audio_two_choice.txt) -> $V"
[ "$V" != none ] || { echo "no variant trained: stopping"; exit 1; }
grep "^${V}_s\(11\|99\) " experiments/audio_final/runs_speck_two.txt > experiments/audio_final/runs_speck_two_w2.txt
echo "[wave 2 start: $V, seeds 11 and 99] $(date +%H:%M)"
train experiments/audio_final/runs_speck_two_w2.txt 2 3
echo "[train done] $(date +%H:%M)"
echo "$V" > $OUT/chosen_on_validation.txt
export OMP_NUM_THREADS=2
cal() {
  m=$1; r=$2; P=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline; C=runs/audio_two_$m/$r/best.pt; N=$OUT/${m}_$r
  [ -f $C ] || { echo "[skip] $m $r: no checkpoint"; return; }
  mkdir -p $N
  python3 -W ignore experiments/chip/audio_l1_sweep.py $P $C --scales 1 1.5 2 3 4 6 --out $N/sweep.json > $N/sweep.log 2>&1
  k=$(python3 -c "import json; print(json.load(open('$N/sweep.json'))['chosen_scale'])")
  python3 -W ignore experiments/chip/audio_refit.py --pipeline $P --ckpt $C --scale $k --out $N/refit > $N/refit.log 2>&1
  echo "[calibrated] $m $r x$k $(date +%H:%M)"
}
export -f cal; export AD OUT
for m in Neil Gomez; do for r in $(ls runs/audio_two_$m); do echo "$m $r"; done; done | xargs -P 6 -L 1 bash -c 'cal $0 $1'
echo "[calibration done] $(date +%H:%M)"
python3 experiments/audio_final/summarize_two.py
bash experiments/audio_final/make_two_bundle.sh
echo "ALL DONE $(date +%H:%M)"
