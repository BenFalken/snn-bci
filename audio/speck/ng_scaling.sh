#!/bin/bash
# EXPLORATORY: trains the variants in runs_speck_ng.txt for both monkeys, then calibrates each exactly as P5 was
# (calibrate_novel.sh: label-free layer-1 sweep on validation, then the readout refit to the chip model's spikes).
# Writes only new folders; nothing of P5 or the final results is touched:
#   networks     $SNNBCI_SNN_RUNS/audio_ng_<monkey>/<variant>_<seed>/
#   calibration  $SNNBCI_WORKSPACE/final_results/audio_ng/<monkey>_<variant>_<seed>/
# Then prints the VALIDATION comparison with P5 (ng_summary.py; test scores only with --show-test).
#   bash audio/speck/explore_96ch.sh > $SNNBCI_SNN_RUNS/audio_explore.log 2>&1      (from the repository root)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -o pipefail
SD=$(cd "$(dirname "$0")/../.." && pwd); cd "$SD"
AD=$SNNBCI_AUDITORY_DATA
OUT=$SNNBCI_WORKSPACE/final_results/audio_ng; mkdir -p "$OUT"
LIST=audio/speck/runs_speck_ng.txt
for m in Neil Gomez; do
  echo "[train start] $m $(date +%H:%M)"
  DATA=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline ROOT=$SNNBCI_SNN_RUNS/audio_ng_$m P=${P:-4} THREADS=${THREADS:-3} \
      LISTS=$LIST bash audio/speck/run_local.sh
  echo "[train done] $m $(date +%H:%M)"
done
export OMP_NUM_THREADS=3
cal() {
  m=$1; r=$2; P=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline; C=$SNNBCI_SNN_RUNS/audio_ng_$m/$r/best.pt
  N=$OUT/${m}_$r; mkdir -p "$N"
  [ -f "$N/refit/refit.json" ] && { echo "[skip] $m $r"; return 0; }
  python3 -W ignore chip/audio_l1_sweep.py "$P" "$C" --scales 1 1.5 2 3 4 6 --out "$N/sweep.json" > "$N/sweep.log" 2>&1
  k=$(python3 -c "import json; print(json.load(open('$N/sweep.json'))['chosen_scale'])")
  echo "[sweep] $m $r chose x$k $(date +%H:%M)"
  python3 -W ignore chip/audio_refit.py --pipeline "$P" --ckpt "$C" --scale "$k" --out "$N/refit" > "$N/refit.log" 2>&1
  echo "[refit] $m $r done $(date +%H:%M)"
}
export -f cal; export AD OUT SNNBCI_SNN_RUNS
for m in Neil Gomez; do grep -v '^[[:space:]]*\(#\|$\)' $LIST | cut -d'|' -f1 | xargs -n1 echo "$m"; done \
    | xargs -P 3 -L 1 bash -c 'cal $0 $1'
python3 audio/speck/ng_summary.py
echo "NEXT-GEN DONE $(date +%H:%M)"
