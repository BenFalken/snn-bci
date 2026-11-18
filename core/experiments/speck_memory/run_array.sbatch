#!/bin/bash
#SBATCH -J SPECK_MEM
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o experiments/speck_memory/logs/SPECK_MEM_%A_%a.out
#
# One array task per PACK lines of a speck_memory plan (plan.py). Submit from the repo root:
#
#   mkdir -p experiments/speck_memory/logs
#   P=$ROOT/plan_motor_A.jsonl; N=$(wc -l < $P)
#   sbatch --export=ALL,PLAN=$P,ROOT=$ROOT --array=0-$((N - 1)) experiments/speck_memory/run_array.sbatch
#   (speech, on GPUs, two runs per GPU:)
#   sbatch -p gpu --gres=gpu:1 -c 8 --mem=48G --export=ALL,PLAN=$P,ROOT=$ROOT,PACK=2 \
#          --array=0-$(((N + 1) / 2 - 1)) experiments/speck_memory/run_array.sbatch
#
# ENV_SCRIPT (default: the speech SNNDecoder's snn_config.sh) provides activate_env (torch, sinabs, pyyaml).
# Finished runs (result.json present) are skipped, so a resubmitted array only redoes what is missing.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
: "${PLAN:?set PLAN}" "${ROOT:?set ROOT}"
ENV_SCRIPT="${ENV_SCRIPT:-$SNNBCI_ROOT/speech/SNNDecoder/scripts/snn_config.sh}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
# snn_config.sh requires BASE_DIR (the speech data root) and exits a batch job without it
export BASE_DIR="${BASE_DIR:-$SNNBCI_OSCAR_SCRATCH}"
cd "$(dirname "$ENV_SCRIPT")" && source "$ENV_SCRIPT" && activate_env && cd "$REPO"
PACK=${PACK:-1}
task=${SLURM_ARRAY_TASK_ID:-0}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-4} / PACK > 0 ? ${SLURM_CPUS_PER_TASK:-4} / PACK : 1 ))
echo "task $task of $PLAN on $(hostname): lines $((task * PACK))..$((task * PACK + PACK - 1)), $OMP_NUM_THREADS threads each"
pids=()
for (( i = task * PACK; i < task * PACK + PACK; i++ )); do
    "${PYTHON:-python}" experiments/speck_memory/run.py "$PLAN" "$i" --root "$ROOT" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
