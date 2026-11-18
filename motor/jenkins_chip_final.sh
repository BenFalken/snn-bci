#!/bin/bash
# Jenkins's Speck networks on the devkit, each session at its own layer-1 threshold factor, chosen without labels
# (closest match to the trained network in the chip's event-by-event firing, simulated in software; snn-decoders
# experiments/chip/motor_l1_sweep.py, on 20 TRAINING trials per session): 09-12 x2, 09-16 x4, 09-18 x3, 09-23 x3. The x2 used before suited only 09-12
# (Jenkins fires ~300 to 900 output spikes per step, against indy's ~8).
#
#   bash jenkins_chip_final.sh          (devkit plugged in; results/final/jenkins_l1/, then the summary)
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
C="$HERE/data/snn_checkpoints/hkm/jenkins"
OUT="$HERE/results/final/jenkins_l1"
cd "$HERE/main/sbatch_scripts"
for cfg in "20090912 2" "20090916 4" "20090918 3" "20090923 3"; do
    set -- $cfg
    S=$(ls "$C/per_session_delay2_binarized" | grep "$1")
    ONE="$HERE/results/final/_lsnn_one_$1"; rm -rf "$ONE"; mkdir -p "$ONE"
    cp -r "$C/lsnn_loso_finetuned_binarized/$S" "$ONE/$S"           # sessions = those with an 'snn' checkpoint
    echo; echo "################ $S: layer 1 threshold x$2 ################"
    SPECK_H1_SCALE=$2 DATA_ROOT="$HERE/data" bash run_inference.sbatch --local hkm jenkins \
        DECODERS=speck SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$C/per_session_delay2_binarized" \
        RESULTS_DIR="$OUT" SNN_LABEL="LSNN" SPECK_WAIT_TIME=0.003
done
python3 - "$OUT" <<'EOF'
import json, os, sys
d = os.path.join(sys.argv[1], 'speck_diagnosis.json')
if os.path.isfile(d):
    for s, v in json.load(open(d))['sessions'].items():
        if 'speck' in v:
            print(f"{s[16:24]}: chip RMSE {v['speck']['rmse']:.2f} (PyTorch {v['pytorch']['rmse']:.2f}), "
                  f"{v['speck']['output_spikes_per_step']:.0f} output spikes per step")
EOF
echo "Send back $OUT"
