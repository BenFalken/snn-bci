#!/bin/bash
# Every audio temporal-precision run on this Mac, then the scores. From the repo root:
#   bash experiments/audio_temporal/run_all.sh > runs/audio_temporal.log 2>&1
set -o pipefail
cd "$(dirname "$0")/../.."
E=experiments/audio_temporal; mkdir -p runs/audio_temporal
( for cell in gru lstm; do for w in 0 1; do
    n=K_${cell}_$([ $w = 0 ] && echo ctl || echo deriv)_s1337; o=runs/audio_temporal/$n
    [ -f $o/done ] && continue; mkdir -p $o; echo "[start] $n $(date +%H:%M)"
    python3 $E/keras_rnn.py --cell $cell --deriv $w --seed 1337 --threads 4 --out $o > $o/train.log 2>&1
    echo "[done] $n $(date +%H:%M): $(grep -c val_loss $o/train.log) epochs"
done; done ) &
ROOT=runs/audio_temporal P=5 THREADS=2 LISTS="$E/runs_lsnn.txt $E/runs_speck.txt" bash experiments/audio_final/run_local.sh
wait
python3 $E/score_temporal.py --out runs/audio_temporal_scores.json --jobs 6
echo "ALL DONE $(date +%H:%M)"
