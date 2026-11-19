#!/bin/bash
# The GRU reference decoders on Oscar, from motor/gru_reference:   bash submit_gru.sh
#   step 1: their protocol (NeuroBench split, loco at 192 channels), 6 sessions x {aegru, gru} x 5 seeds
#   step 2: our protocol (last 10% test, 96 M1 channels), 45 sessions x {aegru, gru} x 3 seeds
# Results: $SNNBCI_OSCAR_MOTOR_DATA/results/gru_reference/{neurobench,ours}/
source "$(cd "$(dirname "$0")" && pwd)/../../snnbci_env.sh"
set -eo pipefail
mkdir -p gru_logs
R=$SNNBCI_OSCAR_MOTOR_DATA/results/gru_reference
N1=$(( $(grep -cv '^\s*$' sessions_neurobench.txt) * 2 - 1 )); N2=$(( $(grep -cv '^\s*$' sessions_ours.txt) * 2 - 1 ))
J1=$(sbatch --parsable --array=0-$N1 --export=ALL,PROTOCOL=neurobench,SESSIONS=sessions_neurobench.txt,SEEDS="0 1 2 3 4",OUT=$R/neurobench run_gru_array.sbatch)
J2=$(sbatch --parsable --array=0-$N2%40 --export=ALL,PROTOCOL=ours,SESSIONS=sessions_ours.txt,SEEDS="0 1 2",OUT=$R/ours run_gru_array.sbatch)
echo "step 1 (NeuroBench protocol): job $J1   step 2 (our protocol): job $J2   logs gru_logs/"
echo "when done:  scp -r $SNNBCI_OSCAR_USER@ssh.ccv.brown.edu:$R ~/Downloads/"
