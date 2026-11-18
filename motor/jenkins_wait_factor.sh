#!/bin/bash
# Jenkins on the devkit at each session's final layer 1 factor, at longer waits per step: how much of the chip's gap
# to its software model is timing? (The software model of event-by-event firing, which has no timing, accounts for
# about 9 of the ~22 mm/s gap on the test trials; this measures the rest.) The 3 ms result is the final batch's
# (results/final_v2/jenkins), so it is not rerun. Waits above ~3 ms are slower than real time (a step is 4 ms of
# data): this explains the gap; it does not replace the 3 ms results.
#
#   bash jenkins_wait_factor.sh                  all four sessions, 10 ms (about 1.5 h per session)
#   WAITS="10 20" bash jenkins_wait_factor.sh 20090916 20090918
#   DIAG=1 ...                                   also the chip-vs-8-bit diagnosis (spike ratio, lag): runs the chip
#                                                a second time, so about twice as long
# Only the named session runs on the chip (its own Speck checkpoint folder). Results: results/wait_factor/
# <session>_w<ms>/; finished ones are skipped; a table at the end, with the 3 ms results alongside.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSIONS=("$@"); [ ${#SESSIONS[@]} -eq 0 ] && SESSIONS=(20090912 20090916 20090918 20090923)
WAITS="${WAITS:-10}"
C="$SNNBCI_MOTOR_DATA/snn_checkpoints/hkm/jenkins"
R="$SNNBCI_MOTOR_RESULTS/wait_factor"
bash "$HERE/run_speck.sh" check
cd "$HERE/main/sbatch_scripts"
for SESS in "${SESSIONS[@]}"; do
    F=$(awk -v s="$SESS" '$1 == "jenkins" && $2 ~ s {print $3}' "$HERE/speck_l1_factors.txt")
    [ -n "$F" ] || { echo "no factor for $SESS in speck_l1_factors.txt"; exit 1; }
    S=$(ls "$C/per_session_delay2_binarized" | grep "$SESS")
    ONE="$R/_lsnn_$SESS"; SP="$R/_speck_$SESS"
    rm -rf "$ONE" "$SP"; mkdir -p "$ONE" "$SP"
    cp -r "$C/lsnn_loso_finetuned_binarized/$S" "$ONE/$S"
    cp -r "$C/per_session_delay2_binarized/$S" "$SP/$S"
    for W in $WAITS; do
        T="$R/${SESS}_w$W"
        [ -f "$T/sessions/$S.json" ] && { echo "skip $SESS w$W (done)"; continue; }
        echo; echo "################ $SESS: layer 1 x$F, wait $W ms ($(date +%H:%M)) ################"
        SPIKE_DENSITY="${DIAG:-0}" SPECK_H1_SCALE=$F DATA_ROOT="$SNNBCI_MOTOR_DATA" bash run_inference.sbatch --local hkm jenkins \
            DECODERS=speck SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$SP" \
            RESULTS_DIR="$T" SNN_LABEL="LSNN" SPECK_WAIT_TIME=$(python3 -c "print($W/1000)")
        [ -f "$T/sessions/$S.json" ] || echo "WARNING: $SESS w$W wrote no sessions/$S.json (see the errors above)"
    done
done
python3 - "$R" "$SNNBCI_MOTOR_RESULTS/final_v2/jenkins/sessions" <<'PY'
import glob, json, os, sys
def row(f):
    d = json.load(open(f))
    return d["full"]["metrics"]["speck"]["rmse"], d["profiles"]["laptop"]["decoders"]["speck"]["latency_s"] * 1e3
print("\nsession    wait  Speck RMSE  ms/step")
for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.json"))):
    s = os.path.basename(f)[:-5]; tag = s[16:24]
    try:
        r, l = row(f); print(f"{tag}   3 ms  {r:10.1f}  {l:7.2f}   (final batch)")
    except Exception: pass
    for g in sorted(glob.glob(os.path.join(sys.argv[1], f"{tag}_w*", "sessions", s + ".json")),
                    key=lambda p: int(p.split("_w")[1].split("/")[0])):
        r, l = row(g); w = g.split("_w")[1].split("/")[0]
        print(f"{tag}  {w:>2s} ms  {r:10.1f}  {l:7.2f}")
PY
echo "Send back $R (the sessions/ folders are what matter)"
