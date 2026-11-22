#!/bin/bash
# EXPLORATORY: the spike-timing pilot (runs_fine_timing.txt), both monkeys, then the Speck variants calibrated exactly as
# P5 was (label-free layer-1 sweep on validation, then the readout refit), then the VALIDATION comparison
# (fine_timing_summary.py). Writes only new folders; nothing of P5, the LSNN or the final results is touched:
#   networks     $SNNBCI_SNN_RUNS/audio_fine_<monkey>/<variant>_<seed>/
#   calibration  $SNNBCI_WORKSPACE/final_results/audio_fine/<monkey>_<variant>_<seed>/
# Needs each cohort's 4 ms inputs (once per monkey, from the repository root):
#   python3 audio/prepare_fine_bins.py --subject <M> --region aSTG --date <day> --raw-base-dir <raw>/<M>/Complex/aSTG
#   (cd audio && python3 build_word_dataset_daynorm.py --subject <M> --region aSTG --include-novel \
#        --name ALLWORDS_NOVEL_DAYNORM --fine [--keep-split-from <ALLWORDS_VERIFIED_DAYNORM>/prep/epochs.csv])   (Neil: with it)
#   bash audio/speck/fine_timing.sh > $SNNBCI_SNN_RUNS/audio_fine.log 2>&1      (from the repository root)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -o pipefail
SD=$(cd "$(dirname "$0")/../.." && pwd); cd "$SD"
AD=$SNNBCI_AUDITORY_DATA
OUT=$SNNBCI_WORKSPACE/final_results/audio_fine; mkdir -p "$OUT"
LIST=audio/speck/runs_fine_timing.txt
for m in ${MONKEYS:-Neil Gomez}; do
  for f in features_4ms_counts.npy features_4ms_counts_daynorm.npy; do
    [ -f "$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline/$f" ] || { echo "missing $m $f: build the 4 ms inputs first"; exit 1; }
  done
done
for m in ${MONKEYS:-Neil Gomez}; do
  echo "[train start] $m $(date +%H:%M)"
  DATA=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline ROOT=$SNNBCI_SNN_RUNS/audio_fine_$m P=${P:-4} THREADS=${THREADS:-3} \
      LISTS=$LIST bash audio/speck/run_local.sh
  echo "[train done] $m $(date +%H:%M)"
done
export OMP_NUM_THREADS=3
cal() {
  m=$1; r=$2; P=$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline; C=$SNNBCI_SNN_RUNS/audio_fine_$m/$r/best.pt
  N=$OUT/${m}_$r; mkdir -p "$N"
  [ -f "$N/refit/refit.json" ] && { echo "[skip] $m $r"; return 0; }
  [ -f "$C" ] || { echo "[missing] $C"; return 0; }
  python3 -W ignore chip/audio_l1_sweep.py "$P" "$C" --scales 1 1.5 2 3 4 6 --out "$N/sweep.json" > "$N/sweep.log" 2>&1
  k=$(python3 -c "import json; print(json.load(open('$N/sweep.json'))['chosen_scale'])")
  echo "[sweep] $m $r chose x$k $(date +%H:%M)"
  python3 -W ignore chip/audio_refit.py --pipeline "$P" --ckpt "$C" --scale "$k" --out "$N/refit" > "$N/refit.log" 2>&1
  echo "[refit] $m $r done $(date +%H:%M)"
}
export -f cal; export AD OUT SNNBCI_SNN_RUNS
for m in ${MONKEYS:-Neil Gomez}; do grep -v '^[[:space:]]*\(#\|$\)' $LIST | cut -d'|' -f1 | grep -E '^FT' | xargs -n1 echo "$m"; done \
    | xargs -P 3 -L 1 bash -c 'cal $0 $1'
for m in ${MONKEYS:-Neil Gomez}; do                           # the LSNNs, scored streaming over the whole recording
  for r in $(grep -v '^[[:space:]]*\(#\|$\)' $LIST | cut -d'|' -f1 | grep -E '^LS'); do
    C=$SNNBCI_SNN_RUNS/audio_fine_$m/$r/best.pt; N=$OUT/${m}_$r; mkdir -p "$N"
    [ -f "$C" ] && [ ! -s "$N/eval.txt" ] && python3 -W ignore -m snndec.evaluate audio "$C" \
        --data "$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline" --save-predictions "$N/preds" > "$N/eval.txt" 2>&1
  done
done
# P5's refit again, with its own chosen factor, into audio_fine/<monkey>_P5_<seed> (its original folder is left as it
# is): the refit now also saves validation predictions, which the per-word metric needs; its scores must reproduce
for m in ${MONKEYS:-Neil Gomez}; do
  for s in s1337 s11; do
    N=$OUT/${m}_P5_$s; J=$SNNBCI_WORKSPACE/final_results/audio_novel/${m}_P5_$s/refit/refit.json
    [ -f "$N/refit/preds_event_refit_valid.npy" ] && continue
    k=$(python3 -c "import json; print(json.load(open('$J'))['scale'])")
    mkdir -p "$N"; python3 -W ignore chip/audio_refit.py --pipeline "$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline" \
        --ckpt "$SNNBCI_SNN_RUNS/audio_novel_$m/P5_centre_ema1_deriv_$s/best.pt" --scale "$k" --out "$N/refit" > "$N/refit.log" 2>&1
    echo "[P5 refit] $m $s x$k $(date +%H:%M)"
  done
done
python3 audio/speck/fine_timing_summary.py
echo "FINE TIMING DONE $(date +%H:%M)"
