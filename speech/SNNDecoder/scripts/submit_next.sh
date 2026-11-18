#!/bin/bash
# Submits the next round in one go, from this directory on Oscar:
#   bash submit_next.sh
#
#   GPU (queued in this order; the 2-GPU limit runs two at a time)
#     GRU_2x896       GRU reference of the 2 x 2048 SNN's size (rerun: the first try lacked BASE_DIR)
#     GRU_TX1         GRU reference on raw counts, for the spike-native SNN (rerun, same reason)
#     rsnn_window     SNN 2 x 1280, target 0.1, input windows 4 / 8 / 14 bins (3 runs on one GPU)
#     GRU_5x512_k14   the paper's real-time RNN shape
#     GRU_2x512_k4    GRU twin of the 4-bin-window SNN
#     rsnn_width      SNN with the SR training at 2 x 512, 2 x 896, 5 x 1024 (3 runs on one GPU)
#   CPU (scripts/efficiency.py; no GPU needed)
#     EFF_now         as soon as the running 80 ms round (rsnn_srs2) has finished: everything trained so far
#     EFF_all         after every job above and the native round: everything
#
# Jobs it waits for: SRS2_JOB (default 7515256) and NATIVE_JOB (default 7515268); a job that has already left
# the queue is not waited for.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -euo pipefail
B=${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}
SRS2_JOB=${SRS2_JOB:-7515256}
NATIVE_JOB=${NATIVE_JOB:-7515268}
cd "$(dirname "$0")"
mkdir -p snn_logs
for f in run_gru_ref.sbatch run_efficiency.sbatch efficiency.py configs/rsnn_window.txt configs/rsnn_width.txt \
         configs/efficiency_runs.txt; do
    [ -f "$f" ] || { echo "ERROR: $f missing here; rsync scripts/ from your Mac first" >&2; exit 1; }
done
grep -q "gru_stack_5x512_k14" ann_controls.py || { echo "ERROR: ann_controls.py is the old version; rsync first" >&2; exit 1; }

in_queue() { [ -n "$(squeue -h -j "$1" 2>/dev/null)" ]; }
gru() {   # gru <job name> <exports>
    sbatch --parsable -J "$1" --export=ALL,BASE_DIR=$B,"$2" run_gru_ref.sbatch
}
snn_runs() {   # snn_runs <run file>: all its runs packed on one GPU
    local n
    n=$(grep -cv '^[[:space:]]*\(#\|$\)' "$1")
    sbatch --parsable --cpus-per-task=4 --export=ALL,BASE_DIR=$B,RUNS=$1,PACK=$n,SESSIONS=all,EPOCHS=96 \
        --array=0-0 run_snn_configs.sbatch
}

J_GRU896=$(gru GRU_2x896 MODEL=gru_stack_2x896)
J_GRUTX1=$(gru GRU_TX1 MODEL=gru_1024,TAG=_tx1,DATA=snnDataTx1,EPOCHS=48)
J_WIN=$(snn_runs configs/rsnn_window.txt)
J_GRU514=$(gru GRU_5x512_k14 MODEL=gru_stack_5x512_k14)
J_GRU2K4=$(gru GRU_2x512_k4 MODEL=gru_stack_2x512_k4)
J_WIDTH=$(snn_runs configs/rsnn_width.txt)
echo "GPU jobs: GRU_2x896 $J_GRU896 | GRU_TX1 $J_GRUTX1 | rsnn_window $J_WIN | GRU_5x512_k14 $J_GRU514 |" \
     "GRU_2x512_k4 $J_GRU2K4 | rsnn_width $J_WIDTH"

DEP_NOW=""
in_queue "$SRS2_JOB" && DEP_NOW="--dependency=afterany:$SRS2_JOB"
# shellcheck disable=SC2086  # DEP_NOW is empty or one flag
J_EFF_NOW=$(sbatch --parsable -J EFF_now $DEP_NOW \
    --export=ALL,BASE_DIR=$B,OUT=$B/derived/efficiency/efficiency_now run_efficiency.sbatch)

DEPS="$J_GRU896:$J_GRUTX1:$J_WIN:$J_GRU514:$J_GRU2K4:$J_WIDTH:$J_EFF_NOW"
in_queue "$NATIVE_JOB" && DEPS="$DEPS:$NATIVE_JOB"
J_EFF_ALL=$(sbatch --parsable -J EFF_all -t 24:00:00 --dependency=afterany:$DEPS \
    --export=ALL,BASE_DIR=$B,OUT=$B/derived/efficiency/efficiency_all run_efficiency.sbatch)
echo "CPU jobs: EFF_now $J_EFF_NOW (${DEP_NOW:-starts now}) | EFF_all $J_EFF_ALL (after everything)"
echo
echo "Results: $B/derived/efficiency/efficiency_now.md (then efficiency_all.md), logs snn_logs/EFFICIENCY_<id>.out"
echo "Check in a minute that nothing failed at start:  sacct -X -j ${J_GRU896},${J_GRUTX1} --format=JobID,JobName%15,State"
squeue -u "$USER" -o "%.12i %.16j %.3t %.10M %R"
