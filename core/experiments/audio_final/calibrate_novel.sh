#!/bin/bash
# Label-free layer-1 sweep (validation) then readout refit, for every P5 network of both monkeys (expanded cohorts).
#   bash experiments/audio_final/calibrate_novel.sh > $SNNBCI_SNN_RUNS/audio_novel_calib.log 2>&1
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
AD=$SNNBCI_AUDITORY_DATA
OUT=$SNNBCI_WORKSPACE/final_results/audio_novel; mkdir -p $OUT
export OMP_NUM_THREADS=3
one() {
  m=$1; s=$2; P=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline; C=$SNNBCI_SNN_RUNS/audio_novel_$m/P5_centre_ema1_deriv_$s/best.pt
  N=$OUT/${m}_P5_$s; mkdir -p $N
  python3 -W ignore experiments/chip/audio_l1_sweep.py $P $C --scales 1 1.5 2 3 4 6 --out $N/sweep.json > $N/sweep.log 2>&1
  k=$(python3 -c "import json; print(json.load(open('$N/sweep.json'))['chosen_scale'])")
  echo "[sweep] $m $s chose x$k $(date +%H:%M)"
  python3 -W ignore experiments/chip/audio_refit.py --pipeline $P --ckpt $C --scale $k --out $N/refit > $N/refit.log 2>&1
  echo "[refit] $m $s done $(date +%H:%M): $(tail -1 $N/refit.log)"
}
export -f one; export AD OUT
printf "%s\n" "Neil s11" "Neil s1337" "Neil s99" "Gomez s11" "Gomez s1337" "Gomez s99" | xargs -P 3 -L 1 bash -c 'one $0 $1'
echo "CALIBRATION DONE $(date +%H:%M)"
