#!/bin/bash
# Motor (II): Jenkins's M1 array alone (96 channels, the standard Speck SNN with input delays, as indy and loco). Every
# decoder on every session: Kalman, Wiener, LSTM, QRNN and the LSNN on this CPU, the Speck SNN on the devkit at each
# session's own layer-1 threshold factor (chosen without labels on 20 TRAINING trials: snn-decoders
# experiments/chip/motor_l1_sweep.py; the "jenkins_m1" lines of speck_l1_factors.txt), 3 ms per chip step.
#
#   bash jenkins_m1_final.sh                 (devkit plugged in; results/final/jenkins_m1/, then the summary)
#   bash jenkins_m1_final.sh 20090916        one session
# Finished sessions are skipped on a rerun.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSIONS=("$@"); [ ${#SESSIONS[@]} -eq 0 ] && SESSIONS=(20090912 20090916 20090918 20090923)
C="$HERE/data/snn_checkpoints/hkm/jenkins_m1"
OUT="$HERE/results/final/jenkins_m1"
for d in "$C/lsnn_loso_finetuned_binarized" "$C/per_session_delay_binarized" "$HERE/data/results/model_cache/hkm/jenkins_m1"; do
    [ -d "$d" ] || { echo "Missing $d: run SUBJ=jenkins_m1 bash fetch_jenkins.sh first" >&2; exit 1; }
done
bash "$HERE/run_speck.sh" check
cd "$HERE/main/sbatch_scripts"
for SESS in "${SESSIONS[@]}"; do
    F=$(awk -v s="$SESS" '$1 == "jenkins_m1" && $2 ~ s {print $3}' "$HERE/speck_l1_factors.txt")
    [ -n "$F" ] || { echo "no jenkins_m1 factor for $SESS in speck_l1_factors.txt" >&2; exit 1; }
    S=$(ls "$C/per_session_delay_binarized" | grep "$SESS")
    ONE="$HERE/results/final/_lsnn_m1_$SESS"; rm -rf "$ONE"; mkdir -p "$ONE"
    cp -r "$C/lsnn_loso_finetuned_binarized/$S" "$ONE/$S"           # sessions = those with an 'snn' checkpoint
    echo; echo "################ $S: every decoder, Speck layer 1 threshold x$F ################"
    SPECK_H1_SCALE=$F DATA_ROOT="$HERE/data" bash run_inference.sbatch --local hkm jenkins_m1 \
        DECODERS=kf,wf,lstm,qrnn,snn,speck \
        SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$C/per_session_delay_binarized" \
        RESULTS_DIR="$OUT" SNN_LABEL="LSNN" SPECK_WAIT_TIME="${SPECK_WAIT_TIME:-0.003}"
done
python3 - "$OUT" <<'PY'
import json, os, sys
f = os.path.join(sys.argv[1], 'combined_metrics.json')
if os.path.isfile(f):
    print(json.dumps(json.load(open(f)), indent=1)[:4000])
d = os.path.join(sys.argv[1], 'speck_diagnosis.json')
if os.path.isfile(d):
    for s, v in json.load(open(d))['sessions'].items():
        if 'speck' in v:
            print(f"{s[16:24]}: chip RMSE {v['speck']['rmse']:.2f} (PyTorch {v['pytorch']['rmse']:.2f}), "
                  f"{v['speck']['output_spikes_per_step']:.0f} output spikes per step")
PY
echo "Send back $OUT"
