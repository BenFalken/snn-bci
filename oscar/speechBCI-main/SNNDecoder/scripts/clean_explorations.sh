#!/bin/bash
# Remove the checkpoints of exploratory speech runs, keeping only what the final deck rests on:
#   final LSNN (both-layer readout)   C_lsnn_nodelay96, Q_lsnn_s43, Q_lsnn_s44
#   DelayNet, the fork's other branch R_readall96, R_readall_s43, R_readall_s44
#   the ladder on the final path      L0_orig, L1_width, S2_rec, S3_out80, S4_readall, C_nodelay96; side check R1_rec
#   the references                    everything in snns/controls (GRU, LSTM, QRNN, Wiener, Kalman)
# Training logs (snn_logs/) are not touched, so every PER in the deck stays traceable.
# Run directories written to in the last 2 hours are skipped (a run may still be training).
#   bash clean_explorations.sh            dry run: lists what would go, with sizes
#   bash clean_explorations.sh --delete   deletes it
SNNS="${BASE_DIR:-/users/$USER/scratch/speechBCI_dryad}/derived/snns"
KEEP="C_lsnn_nodelay96 Q_lsnn_s43 Q_lsnn_s44 R_readall96 R_readall_s43 R_readall_s44
      L0_orig L1_width S2_rec S3_out80 S4_readall C_nodelay96 R1_rec"
cd "$SNNS" || exit 1
kept=() gone=() other=()
for d in */*/; do
  d=${d%/}; group=${d%%/*}; run=${d#*/}
  [ "$group" = controls ] && continue
  case "$run" in *_*) ;; *) other+=("$d"); continue ;; esac          # not a <prefix>_<name> run directory: left alone
  keep=0; for n in $KEEP; do case "$run" in *_"$n") keep=1 ;; esac; done
  [ $keep = 1 ] && { kept+=("$d"); continue; }
  if [ -n "$(find "$d" -newermt '2 hours ago' -print -quit)" ]; then echo "skip (written in the last 2 h): $d"; continue; fi
  gone+=("$d")
done
echo "== keep (${#kept[@]})"; printf '  %s\n' "${kept[@]}"
[ ${#other[@]} -gt 0 ] && { echo "== left alone, not a run directory (${#other[@]})"; printf '  %s\n' "${other[@]}"; }
echo "== remove (${#gone[@]})"
[ ${#gone[@]} -eq 0 ] && exit 0
du -sh "${gone[@]}" | sed 's/^/  /'
du -shc "${gone[@]}" | tail -1 | sed 's/^/  /'
missing=$(for n in $KEEP; do ls -d */*_"$n" >/dev/null 2>&1 || printf '%s ' "$n"; done)
[ -n "$missing" ] && echo "!! no checkpoint found for: $missing"
if [ "$1" = "--delete" ]; then
  rm -rf "${gone[@]}"
  find . -mindepth 1 -maxdepth 1 -type d -empty -delete            # groups left empty
  echo "deleted ${#gone[@]} run directories"
else
  echo "dry run: nothing deleted (rerun with --delete)"
fi
