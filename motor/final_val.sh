#!/bin/bash
# The validation-selected motor batch on the Speck laptop, devkit plugged in: every decoder (kf, wf, lstm, qrnn, LSNN,
# Speck on the chip) on every session of indy, loco and jenkins_m1 (the paper's Motor II), with the checkpoints chosen
# on a validation split (train_snn.py --val-frac; motor/sbatch_scripts/submit_val_reselect.sh) and each Speck
# network's layer-1 threshold factor re-chosen for its new checkpoint, label-free, on 20 training trials
# (chip/motor_l1_sweep.py; speck_l1_factors_val.txt). Every network is hard reset.
#
#   VAL_ROOT=/path/to/motor_val bash final_val.sh [subject ...]      default: indy loco jenkins_m1
#   DECODERS=speck ...   only the Speck SNN on the chip (the others already scored elsewhere)
#
# VAL_ROOT holds lsnn_loso_finetuned_binarized_val/<session>/ and per_session_delay_binarized_val/<session>/ for every
# subject (default: $SNNBCI_MOTOR_DATA/motor_val). Results: results/final_v3/<subject>/. Finished sessions are skipped.
#
# Other Speck networks (defaults = the final hard-reset ones above):
#   SPECK_DIR=per_session_delay_soft_binarized_val FACTORS=speck_l1_factors_val_soft.txt OUT_NAME=final_v3_soft \
#   DECODERS=speck VAL_ROOT=... bash final_val.sh indy loco
# With DECODERS=speck the LSNN folder is not needed: the sessions are read from the Speck folder.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco jenkins_m1)
FACTORS="${FACTORS:-$HERE/speck_l1_factors_val.txt}"; case "$FACTORS" in /*) ;; *) FACTORS="$HERE/$FACTORS" ;; esac
SPECK_DIR="${SPECK_DIR:-per_session_delay_binarized_val}"; OUT_NAME="${OUT_NAME:-final_v3}"
VAL_ROOT="${VAL_ROOT:-$SNNBCI_MOTOR_DATA/motor_val}"
case ",${DECODERS:-speck}," in *,speck,*) bash "$HERE/run_speck.sh" check ;; esac   # the devkit, if Speck runs
cd "$HERE/sbatch_scripts"
for SUBJ in "${SUBJECTS[@]}"; do
    EXP=bmi; [ "$SUBJ" = jenkins_m1 ] && EXP=hkm
    OUT="$SNNBCI_MOTOR_RESULTS/$OUT_NAME/$SUBJ"
    # one pass per factor, over the sessions that use it (sessions = those with an LSNN checkpoint)
    for F in $(awk -v s="$SUBJ" '$1 == s {print $3}' "$FACTORS" | sort -u); do
        # this pass's sessions only, for both networks: the chip diagnosis at the end of each pass walks the Speck
        # folder it is given, so a folder holding other subjects' or other factors' sessions would derail it
        ONE="$SNNBCI_MOTOR_RESULTS/$OUT_NAME/_lsnn_${SUBJ}_x$F"; rm -rf "$ONE"; mkdir -p "$ONE"
        SPK="$SNNBCI_MOTOR_RESULTS/$OUT_NAME/_speck_${SUBJ}_x$F"; rm -rf "$SPK"; mkdir -p "$SPK"
        for S in $(awk -v s="$SUBJ" -v f="$F" '$1 == s && $3 == f {print $2}' "$FACTORS"); do
            if [ "${DECODERS:-}" = speck ]; then cp -r "$VAL_ROOT/$SPECK_DIR/$S" "$ONE/$S"   # sessions only
            else cp -r "$VAL_ROOT/lsnn_loso_finetuned_binarized_val/$S" "$ONE/$S"; fi
            cp -r "$VAL_ROOT/$SPECK_DIR/$S" "$SPK/$S"
        done
        echo; echo "################ $SUBJ, layer 1 threshold x$F: $(ls "$ONE" | wc -l | tr -d ' ') sessions ################"
        SPECK_H1_SCALE=$F DATA_ROOT="$SNNBCI_MOTOR_DATA" bash run_inference.sbatch --local "$EXP" "$SUBJ" \
            DECODERS="${DECODERS:-kf,wf,lstm,qrnn,snn,speck}" \
            SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$SPK" \
            RESULTS_DIR="$OUT" SNN_LABEL="LSNN" SPECK_WAIT_TIME="${SPECK_WAIT_TIME:-0.003}"
    done
done
echo
python3 - "$SNNBCI_MOTOR_RESULTS/$OUT_NAME" "${SUBJECTS[@]}" <<'EOF'
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
echo "Send back $SNNBCI_MOTOR_RESULTS/$OUT_NAME"
