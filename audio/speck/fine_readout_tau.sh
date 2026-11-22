#!/bin/bash
# EXPLORATORY (the spike-timing pilot): a refit-only check of the Speck SNN's readout smoothing. The readout's EMA
# (trained time constant ~150 ms per chip step) is refitted with its time constant x 0.5, 0.25 and 0.125, reusing each
# network's simulated chip-model spikes (no retraining, no new simulation); FT1 (4 ms spikes) and P5 (the control).
# Writes final_results/audio_fine/<monkey>_<run>_<seed>/refit_tau<f>/ beside each refit/ (left as it is). Then
# fine_readout_tau_summary.py decides on validation (rule in its docstring).
#   bash audio/speck/fine_readout_tau.sh        (from the repository root; finished folders are skipped)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"
set -o pipefail
SD=$(cd "$(dirname "$0")/../.." && pwd); cd "$SD"
OUT=$SNNBCI_WORKSPACE/final_results/audio_fine; AD=$SNNBCI_AUDITORY_DATA
export OMP_NUM_THREADS=2
one() {
  m=$1; run=$2; s=$3; f=$4
  R=$OUT/${m}_${run}_$s; N=$R/refit_tau$f
  [ -f "$R/refit/refit.json" ] || { echo "[not calibrated] $m $run $s"; return 0; }
  [ -f "$N/refit.json" ] && return 0
  case $run in P5) C=$SNNBCI_SNN_RUNS/audio_novel_$m/P5_centre_ema1_deriv_$s/best.pt ;;
               *)  C=$SNNBCI_SNN_RUNS/audio_fine_$m/${run}_$s/best.pt ;; esac
  k=$(python3 -c "import json; print(json.load(open('$R/refit/refit.json'))['scale'])")
  python3 -W ignore chip/audio_refit.py --pipeline "$AD/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline" --ckpt "$C" --scale "$k" \
      --tau-factor "$f" --spikes-from "$R/refit" --out "$N" > "$N.log" 2>&1 && echo "[refit] $m $run $s x$f $(date +%H:%M)"
}
export -f one; export OUT AD SNNBCI_SNN_RUNS
for m in Neil Gomez; do for run in FT1_spikes P5; do for s in s1337 s11 s99; do for f in 0.5 0.25 0.125; do
  echo "$m $run $s $f"; done; done; done; done | xargs -P 4 -L 1 bash -c 'one $0 $1 $2 $3'
python3 audio/speck/fine_readout_tau_summary.py
