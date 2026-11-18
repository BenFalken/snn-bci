#!/bin/bash
# The final motor batch on this laptop, devkit plugged in: every decoder (kf, wf, lstm, qrnn, LSNN, Speck on the chip)
# on every session of indy, loco and jenkins, from the cached models and each session's test split.
#
# Speck runs with hidden layer 1's threshold scaled per session (speck_l1_factors.txt). The chip applies a step's
# input events one at a time, where training summed the step first, so layer 1 over-fires. Each factor was chosen
# without labels, on 20 TRAINING trials per session: the scale whose event-by-event output in software agrees best
# (correlation) with the trained network's own output (chip/motor_l1_sweep.py). Test trials
# play no part in the choice. Every network is hard reset.
#
#   bash final_all.sh [subject ...]       default: indy loco jenkins
#
# Results: results/final_v2/<subject>/ (combined_metrics.json, efficiency_summary.json, speck_diagnosis.json).
# Finished sessions are skipped on a rerun.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco jenkins)
FACTORS="$HERE/speck_l1_factors.txt"
bash "$HERE/run_speck.sh" check
cd "$HERE/main/sbatch_scripts"
for SUBJ in "${SUBJECTS[@]}"; do
    EXP=bmi; SPECK_DIR=per_session_delay_binarized
    [ "$SUBJ" = jenkins ] && { EXP=hkm; SPECK_DIR=per_session_delay2_binarized; }
    C="$SNNBCI_MOTOR_DATA/snn_checkpoints/$EXP/$SUBJ"
    OUT="$SNNBCI_MOTOR_RESULTS/final_v2/$SUBJ"
    # one pass per factor, over the sessions that use it (sessions = those with an LSNN checkpoint)
    for F in $(awk -v s="$SUBJ" '$1 == s {print $3}' "$FACTORS" | sort -u); do
        ONE="$SNNBCI_MOTOR_RESULTS/final_v2/_lsnn_${SUBJ}_x$F"; rm -rf "$ONE"; mkdir -p "$ONE"
        for S in $(awk -v s="$SUBJ" -v f="$F" '$1 == s && $3 == f {print $2}' "$FACTORS"); do
            cp -r "$C/lsnn_loso_finetuned_binarized/$S" "$ONE/$S"
        done
        echo; echo "################ $SUBJ, layer 1 threshold x$F: $(ls "$ONE" | wc -l | tr -d ' ') sessions ################"
        SPECK_H1_SCALE=$F DATA_ROOT="$SNNBCI_MOTOR_DATA" bash run_inference.sbatch --local "$EXP" "$SUBJ" \
            DECODERS=kf,wf,lstm,qrnn,snn,speck \
            SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$C/$SPECK_DIR" \
            RESULTS_DIR="$OUT" SNN_LABEL="LSNN" SPECK_WAIT_TIME="${SPECK_WAIT_TIME:-0.003}"
    done
done
echo
python3 - "$SNNBCI_MOTOR_RESULTS/final_v2" "${SUBJECTS[@]}" <<'EOF'
import json, os, sys
root, subjects = sys.argv[1], sys.argv[2:]
for subj in subjects:
    f = os.path.join(root, subj, "efficiency_summary.json")
    if not os.path.isfile(f):
        print(f"  {subj}: no results"); continue
    for machine, decs in json.load(open(f)).items():
        print(f"  {subj} ({machine}):")
        for name, r in decs.items():
            e = f"{r['energy_j'] * 1e6:10.2f} uJ ({r['energy_method']})" if r.get('energy_j') is not None else "n/a"
            print(f"    {name:6s} RMSE {r['rmse']:6.2f} | {r['latency_s'] * 1e3:8.3f} ms/sample | {e}")
EOF
echo "Send back $SNNBCI_MOTOR_RESULTS/final_v2"
