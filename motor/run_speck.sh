#!/bin/bash
# The Speck SNN with input delays, trained per session on Oscar (SPECK_MODEL=speck_delay), on every
# session in this folder: in PyTorch and on the Speck2f devkit.
#   network: 96 channels x 4 copies (now, 40, 100, 200 ms back; built on the host) = 384 chip inputs
#            -> 128 -> 256 -> 256 -> 36 spiking outputs (IAF, hard reset), decoded on the host
#
#   bash run_speck.sh check                    the Python packages and the devkit only
#   bash run_speck.sh nochip [subject ...]     PyTorch only (no devkit needed)
#   bash run_speck.sh [chip] [subject ...]     PyTorch and the chip (devkit connected); default: indy loco
#
# Results: results/<nochip|chip>/<subject>/: sessions/<session>.json, combined_metrics.json,
# efficiency_summary.json, the figures, and with the chip speck_diagnosis.json (PyTorch vs 8-bit
# quantized vs chip) and every session's layer activity. Finished sessions are skipped on a rerun, so an
# interrupted run resumes. SPECK_WAIT_TIME (default 0.003 s) is the chip's time per step before its
# output is read; VENV=/path activates a Python virtualenv.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-chip}"; shift || true
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(indy loco)
CKPT_DIR=per_session_delay_binarized

if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
python3 - "$MODE" <<'EOF'
import importlib.util, sys
need = ["numpy", "torch", "sinabs", "h5py", "scipy", "sklearn", "matplotlib", "tensorflow"]
if sys.argv[1] in ("chip", "check"):
    need += ["samna"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"Missing Python packages: {missing} (activate the Speck environment, or set VENV=...)")
print("Python packages: ok")
if sys.argv[1] in ("chip", "check"):
    try:
        import samna
        devices = [d.device_type_name for d in samna.device.get_unopened_devices()]
    except Exception as e:                      # the device listing differs between samna versions
        print(f"Could not list devices ({e}); the run will try to open speck2fdevkit:0 itself")
    else:
        print("Devices found:", devices or "none")
        if not any("speck2f" in d.lower() for d in devices):
            sys.exit("No Speck2f devkit found: plug it in (or run: bash run_speck.sh nochip)")
EOF
[ "$MODE" = "check" ] && exit 0

case "$MODE" in
    nochip) DECODERS=snn ;;
    chip)   DECODERS=snn,speck ;;
    *) echo "usage: bash run_speck.sh [check|nochip|chip] [subject ...]" >&2; exit 1 ;;
esac
cd "$HERE/main/sbatch_scripts"
for SUBJ in "${SUBJECTS[@]}"; do
    CKPTS="$HERE/data/snn_checkpoints/bmi/$SUBJ/$CKPT_DIR"
    [ -d "$CKPTS" ] || { echo "No checkpoints for $SUBJ under $CKPTS" >&2; exit 1; }
    echo; echo "################ $SUBJ: $(ls "$CKPTS" | wc -l | tr -d ' ') sessions ($MODE) ################"
    # 3 ms per step on the chip: the delay bank's ~4x input events need it (at 1 ms the chip's RMSE was 1.2 higher)
    DATA_ROOT="$HERE/data" bash run_inference.sbatch --local bmi "$SUBJ" \
        DECODERS="$DECODERS" SNN_CHECKPOINT_ROOT="$CKPTS" SPECK_CHECKPOINT_ROOT="$CKPTS" \
        RESULTS_DIR="$HERE/results/$MODE/$SUBJ" SNN_LABEL="Speck SNN (PyTorch)" \
        SPECK_WAIT_TIME="${SPECK_WAIT_TIME:-0.003}"
done
echo
echo "Summary (RMSE, the deck's per-axis measure; chip time per step, power, energy per sample):"
python3 - "$HERE/results/$MODE" "${SUBJECTS[@]}" <<'EOF'
import json, os, sys
import numpy as np
root, subjects = sys.argv[1], sys.argv[2:]
for subj in subjects:
    f = os.path.join(root, subj, "combined_metrics.json")
    if not os.path.isfile(f):
        print(f"  {subj}: no results"); continue
    m = json.load(open(f))
    snn = [v['snn']['rmse'] for v in m.values() if v.get('snn')]
    line = f"  {subj}: {len(m)} sessions | PyTorch {np.mean(snn):.2f}"
    chip = [v['speck']['rmse'] for v in m.values() if v.get('speck')]
    if chip:
        line += f" | chip {np.mean(chip):.2f} (chip - PyTorch {np.mean(chip) - np.mean(snn):+.2f})"
        e = json.load(open(os.path.join(root, subj, "efficiency_summary.json")))
        sp = next((v['speck'] for v in e.values() if 'speck' in v), None)
        if sp:
            line += (f" | {sp['latency_s'] * 1e3:.2f} ms/step, {sp['power_w'] * 1e3:.2f} mW, "
                     f"{sp['energy_j'] * 1e6:.2f} uJ/sample")
    print(line)
EOF
echo "Results: $HERE/results/$MODE"
