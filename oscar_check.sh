#!/bin/bash
# Status of the open Oscar runs (2026-11-26), from anywhere on Oscar:  bash ~/snn-bci/oscar_check.sh
#   queue        your jobs, and the state of every task of the motor soft-reset arrays and speech R2
#   motor        speck_delay_soft (soft reset, floor -3) against the final hard-reset Speck SNN (speck_delay), per
#                subject: sessions finished, and the mean best VALIDATION loss of the sessions both have finished
#                (training's software network; the decision also needs the chip model). Test loss is not shown.
#   speech R2    host-loop feedback + soft reset: its last epoch, beside S3 at the same epoch
source "$(dirname "$0")/snnbci_env.sh" >/dev/null 2>&1
MOTOR=${MOTOR_JOBS:-"9644507 9644508 9644509"}; R2=${R2_JOB:-9583732_5}
D="${DATA_ROOT:-$SNNBCI_OSCAR_MOTOR_DATA}/snn_checkpoints"
echo "== queue"; squeue -u "$USER" -o "%.18i %.9P %.10j %.2t %.10M %R" 2>/dev/null
for j in $MOTOR; do
  sacct -j "$j" -X -n -o State 2>/dev/null | sort | uniq -c | awk -v j="$j" 'BEGIN{printf "array %s:", j} {printf " %s %s", $1, $2} END{print ""}'
done
echo; echo "== motor: soft reset + floor -3 vs hard reset (best validation loss, lower is better)"
for s in bmi/indy bmi/loco hkm/jenkins_m1; do
  python3 - "$D/$s" <<'PY'
import sys, glob, os, re
root = sys.argv[1]
def best(d):
    out = {}
    for f in glob.glob(f"{root}/{d}/*/summary_*.txt"):
        m = re.search(r"Best val loss: ([0-9.]+)", open(f).read())
        if m: out[os.path.basename(os.path.dirname(f))] = float(m.group(1))
    return out
h, s = best("per_session_delay_binarized_val"), best("per_session_delay_soft_binarized_val")
both = sorted(set(h) & set(s))
name = root.rstrip("/").split("/")[-1]
line = f"{name:11s} soft finished {len(s):2d} of {len(h):2d}"
if both:
    import statistics as st
    d = [s[k] - h[k] for k in both]
    line += (f" | {len(both)} paired: soft {st.mean(s[k] for k in both):.2f} vs hard {st.mean(h[k] for k in both):.2f}"
             f" (soft lower in {sum(x < 0 for x in d)} of {len(d)})")
print(line)
PY
done
echo; echo "== speech R2 (feedback + soft reset) vs S3, test PER at the same epoch (training's metric)"
L=~/snn-bci/speech/scripts/snn_logs
per() { grep "^Epoch" "$1" 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="PER") print $(i+1)}'; }
n=$(per "$L/SNN_RUN_$R2.out" | wc -l)
if [ "$n" -gt 0 ]; then
  echo "R2 epoch $n: PER $(per "$L/SNN_RUN_$R2.out" | tail -1) | S3 at epoch $n: $(per "$L/SNN_RUN_9583732_2.out" | sed -n "${n}p") (S3 final 0.3927)"
else
  echo "R2: no epochs yet"
fi
