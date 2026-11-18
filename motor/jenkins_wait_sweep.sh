#!/bin/bash
# Jenkins's Speck network on the devkit, one session, at several chip wait settings: is its poor chip RMSE a
# timing problem? (speck_diagnosis.json: the chip's output lags the 8-bit model by ~4 steps and has 32-46% fewer
# spikes; Jenkins's network fires ~600 output spikes per step against indy's 8, and hkm steps are 4 ms.)
#
#   bash jenkins_wait_sweep.sh                   session 2009-09-16 (chip 132.6 vs PyTorch 42.8)
#   SESSION=20090912 bash jenkins_wait_sweep.sh
#
# Settings (tag: fixed wait per step, then settle = keep reading until the output is quiet that long):
#   w3   3 ms (as run_final.sh)      w10  10 ms      w20  20 ms      w40  40 ms
#   s2   3 ms + 2 ms quiet (<= 50 ms per step)       s5   3 ms + 5 ms quiet (<= 50 ms per step)
# Results: results/wait_sweep/<tag>/; a summary table at the end (RMSE and ms per step).
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-20090916}"
C="$HERE/data/snn_checkpoints/hkm/jenkins"
S=$(ls "$C/per_session_delay2_binarized" | grep "$SESSION")
[ -n "$S" ] || { echo "no session matching $SESSION"; exit 1; }
ONE="$HERE/results/wait_sweep/_lsnn_one"; rm -rf "$ONE"; mkdir -p "$ONE"
cp -r "$C/lsnn_loso_finetuned_binarized/$S" "$ONE/$S"          # sessions = those with an 'snn' checkpoint
cd "$HERE/main/sbatch_scripts"
for cfg in "w3 0.003 0" "w10 0.010 0" "w20 0.020 0" "w40 0.040 0" "s2 0.003 0.002" "s5 0.003 0.005"; do
    set -- $cfg
    echo; echo "################ $S: $1 (wait $2 s, settle $3 s) ################"
    SPECK_SETTLE_S=$3 DATA_ROOT="$HERE/data" bash run_inference.sbatch --local hkm jenkins \
        DECODERS=speck SNN_CHECKPOINT_ROOT="$ONE" SPECK_CHECKPOINT_ROOT="$C/per_session_delay2_binarized" \
        RESULTS_DIR="$HERE/results/wait_sweep/$1" SNN_LABEL="LSNN" SPECK_WAIT_TIME="$2"
done
python3 - "$HERE/results/wait_sweep" <<'EOF'
import glob, json, os, sys
print("\ntag   RMSE    ms/step   (PyTorch delay2 on this session: see speck_diagnosis.json 'pytorch')")
for tag in ["w3", "w10", "w20", "w40", "s2", "s5"]:
    f = os.path.join(sys.argv[1], tag, "efficiency_summary.json")
    if not os.path.isfile(f):
        print(f"{tag:5s} no result"); continue
    for machine, decs in json.load(open(f)).items():
        r = decs.get("speck")
        if r:
            print(f"{tag:5s} {r['rmse']:6.2f} {r['latency_s'] * 1e3:8.2f}")
    d = os.path.join(sys.argv[1], tag, "speck_diagnosis.json")
    if os.path.isfile(d):
        for s, v in json.load(open(d))["sessions"].items():
            c = v.get("chip_vs_quantized", {})
            print(f"      spikes chip/8-bit {c.get('spike_ratio', float('nan')):.2f}, best lag {c.get('best_lag')}, "
                  f"step corr at lag 0 {c.get('step_corr_lag0', float('nan')):.3f}")
EOF
