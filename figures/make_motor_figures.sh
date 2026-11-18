#!/bin/bash
# Every figure of the motor design document (design_docs/02_motor), regenerated from the data, then copied into
# the document's figures/ folder.   bash make_motor_figures.sh   [--recompute: rerun the chip model in fig_motor_events]
set -e
cd "$(dirname "$0")"
for f in fig_topology fig_motor_lag fig_motor_tuning fig_motor_smooth fig_motor_drift fig_motor_budget fig_motor_ladder \
         fig_jenkins_chip; do
    echo "== $f"; python3 $f.py 2>&1 | grep -v "stringData\|timestamp\|Warning\|r2 = " | tail -2
done
echo "== fig_motor_events"; python3 fig_motor_events.py "$@" 2>&1 | tail -1
D=~/Downloads/design_docs/02_motor/figures; mkdir -p $D
cp figures/fig_topology_motor.pdf figures/fig_motor_{lag,tuning,smooth,drift,budget,ladder,events}.pdf figures/fig_jenkins_chip.pdf $D/
echo "copied to $D"
