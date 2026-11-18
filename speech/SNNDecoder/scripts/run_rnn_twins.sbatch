#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -J RNN_TWIN
#SBATCH -o snn_logs/RNN_TWIN_%j_task.out
#SBATCH -e snn_logs/RNN_TWIN_%j_task.out
#
# ANN twins of the recurrent SNN, trained by this pipeline (ann_controls.py) on
# all 19 baseline sessions with the RNN's augmentation (noise 1.0, offset 0.2),
# 60 epochs, all three at once on one GPU:
#   0 gru_stack_2x512           the twin of R2_all19 (same front end and size, GRU units)
#   1 gru_stack_5x1024          the paper RNN's shape, causal
#   2 gru_stack_5x1024_smoothed the paper RNN's shape with its input smoothing (sd 2): against
#                               the cached RNN's own 0.188 test PER, the cost of this pipeline
# Read together: twin vs SNN = the cost of spiking at that size; twin vs cached RNN = the
# cost of this training pipeline.
#
# Submit from this directory:
#   mkdir -p snn_logs
#   sbatch --export=ALL,BASE_DIR=$SNNBCI_OSCAR_SCRATCH run_rnn_twins.sbatch
# RECIPE=rnn trains with the paper RNN's recipe (Adam eps 0.1, lr 0.02 -> 0 linearly, L2 1e-5,
# clip 10; ann_controls.py --recipe rnn); its runs are named <model>_rnnrecipe. EPOCHS (default 60;
# 96 = the RNN's 10,000 batches on these 19 sessions).
# Progress:  for f in snn_logs/RNN_TWIN_<job>_[0-9].out; do grep -m1 "^\[" $f | cut -c1-40; grep "^Epoch" $f | tail -1; done
# History:   $BASE_DIR/derived/snns/controls/<name>.json
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
source ./snn_config.sh
activate_env
export OMP_NUM_THREADS=1

SESSIONS19=(t12.2022.04.28 t12.2022.05.05 t12.2022.05.17 t12.2022.05.19 t12.2022.05.24 t12.2022.05.26
            t12.2022.06.02 t12.2022.06.07 t12.2022.06.14 t12.2022.06.16 t12.2022.06.21 t12.2022.06.28
            t12.2022.07.05 t12.2022.07.14 t12.2022.07.21 t12.2022.07.27 t12.2022.08.02 t12.2022.08.11
            t12.2022.08.13)
COMMON=(--data-dir "$SNN_DATA_DIR" --out-dir "${SNN_CKPT_ROOT}/controls" --sessions "${SESSIONS19[@]}"
        --epochs "${EPOCHS:-60}" --white-noise-sd 1.0 --constant-offset-sd 0.2 --recipe "${RECIPE:-default}")
SUFFIX=""
[ "${RECIPE:-default}" = "rnn" ] && SUFFIX="_rnnrecipe"
job=${SLURM_JOB_ID:-local}

"$PYTHON" ann_controls.py --model gru_stack_2x512 --index 0 --tag "$SUFFIX" "${COMMON[@]}" > "snn_logs/RNN_TWIN_${job}_0.out" 2>&1 &
p0=$!
"$PYTHON" ann_controls.py --model gru_stack_5x1024 --index 1 --tag "$SUFFIX" "${COMMON[@]}" > "snn_logs/RNN_TWIN_${job}_1.out" 2>&1 &
p1=$!
"$PYTHON" ann_controls.py --model gru_stack_5x1024 --index 2 --smooth-sd 2 --tag "_smoothed$SUFFIX" "${COMMON[@]}" \
    > "snn_logs/RNN_TWIN_${job}_2.out" 2>&1 &
p2=$!
status=0
for p in $p0 $p1 $p2; do wait $p || status=1; done
exit $status
