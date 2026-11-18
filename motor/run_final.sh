#!/bin/bash
# The final comparison on this laptop: every decoder on every session, from the cached models and test data.
#   kf, wf, lstm, qrnn   the conventional decoders (results/model_cache)
#   snn                  the LSNN, pretrained leave-one-session-out and fine-tuned (lsnn_loso_finetuned_binarized)
#   speck                the Speck SNN with input delays, trained per session, run on the Speck2f devkit
#                        (per_session_delay_binarized; 3 ms per step on the chip)
# Accuracy on each session's test split, single-sample latency and measured energy (RAPL on this CPU, the
# chip's power monitor for Speck), the report figures, and the Speck diagnosis.
#
#   bash run_final.sh [subject ...]        default: indy loco; jenkins (hkm) after fetch_jenkins.sh
#
# Results: results/final/<subject>/ (combined_metrics.json, efficiency_summary.json, decoder_*.png,
# speck_diagnosis.json). Finished sessions are skipped on a rerun. VENV=/path activates a virtualenv.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco)
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
bash "$HERE/run_speck.sh" check
cd "$HERE/main/sbatch_scripts"
for SUBJ in "${SUBJECTS[@]}"; do
    EXP=bmi; case "$SUBJ" in jenkins*) EXP=hkm ;; esac  # hkm: chip layer 1 threshold x2 (run_inference.sbatch)
    C="$SNNBCI_MOTOR_DATA/snn_checkpoints/$EXP/$SUBJ"
    # Speck checkpoints: jenkins's 192 channels x 4 delays (768 inputs) do not map onto the chip; its 2-delay
    # version (0, 100 ms; 384 inputs, as indy's 96 x 4) does. jenkins_m1 (the M1 array, 96 channels) uses the
    # standard 4-delay network, as indy and loco
    SPECK_DIR=per_session_delay_binarized; [ "$SUBJ" = jenkins ] && SPECK_DIR=per_session_delay2_binarized
    for d in "$C/lsnn_loso_finetuned_binarized" "$C/$SPECK_DIR" "$SNNBCI_MOTOR_DATA/results/model_cache/$EXP/$SUBJ"; do
        [ -d "$d" ] || { echo "Missing $d: fetch it first (see README.txt)" >&2; exit 1; }
    done
    echo; echo "################ $SUBJ: every decoder ################"
    # sessions are those with an LSNN checkpoint; Speck runs at 3 ms per step (the delay bank's input needs it)
    DATA_ROOT="$SNNBCI_MOTOR_DATA" bash run_inference.sbatch --local "$EXP" "$SUBJ" \
        DECODERS=kf,wf,lstm,qrnn,snn,speck \
        SNN_CHECKPOINT_ROOT="$C/lsnn_loso_finetuned_binarized" \
        SPECK_CHECKPOINT_ROOT="$C/$SPECK_DIR" \
        RESULTS_DIR="$SNNBCI_MOTOR_RESULTS/final/$SUBJ" SNN_LABEL="LSNN" \
        SPECK_WAIT_TIME="${SPECK_WAIT_TIME:-0.003}"
done
echo
echo "Summary per decoder (RMSE: the deck's per-axis measure; latency; energy per sample, measured):"
python3 - "$SNNBCI_MOTOR_RESULTS/final" "${SUBJECTS[@]}" <<'EOF'
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
echo "Results: $SNNBCI_MOTOR_RESULTS/final"
