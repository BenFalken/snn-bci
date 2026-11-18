#!/bin/bash
# The deck's result figures and videos for the final model (LSNN reading both layers, C_lsnn_nodelay96), on the Mac:
#   1. copies from Oscar the decoder comparison (run_compare.sbatch), the efficiency report (efficiency_final) and the
#      exported trials (plotTrials.sbatch, viewer_final)
#   2. draws the main figures (paper RNN, LSTM, LSNN, Kalman, Wiener) and the appendix ones (adding GRU and DelayNet),
#      sorted by phoneme error, into figures/inference_final/{main,all}_{fidelity,latency,energy}.png
#   3. renders one video per trial plus all_trials.mp4, and one still per trial, into figures/viewer/{videos,stills}_final
#      (needs ffmpeg on the PATH)
#   bash make_final_figures.sh            # SKIP_SYNC=1 to reuse what was copied before
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OSCAR="${OSCAR:-$USER@ssh.ccv.brown.edu}"
DERIVED="${DERIVED:-$SNNBCI_OSCAR_SCRATCH/derived}"
FIG="$HERE/../figures/inference_final"
VIEW="${VIEW:-$SNNBCI_WORKSPACE/viewer_final}"
mkdir -p "$FIG" "$VIEW"

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  echo "=== copying from Oscar"
  rsync -av "$OSCAR:$DERIVED/decoderComparison_final.json" "$OSCAR:$DERIVED/decoderComparison_final_sentences.json" \
            "$OSCAR:$DERIVED/efficiency/efficiency_final.json" "$OSCAR:$DERIVED/efficiency/efficiency_final.md" "$FIG/"
  rsync -av --delete "$OSCAR:$DERIVED/viewer_final/" "$VIEW/"          # --delete: no trials left from earlier exports
fi

# the efficiency rows each decoder of the comparison is drawn with (configs/efficiency_final.txt labels)
MATCH=(--match "LSNN=Model B, LSNN, final (C_lsnn_nodelay96)" --match "DelayNet=Model A, DelayNet (R_readall96)"
       --match "GRU96=GRU 2x1024 out80, same input, 96 ep" --match "LSTM96=LSTM 2x1024, 96 ep"
       --match "Wiener=Wiener filter (160 ms)" --match "Kalman=Kalman filter per session" --match "RNN=Paper RNN"
       --label "LSNN=LSNN" --label "GRU96=GRU" --label "LSTM96=LSTM")
cd "$HERE"
echo "=== main figures"
MPLBACKEND=Agg python3 plotDecoderFigures.py --compare "$FIG/decoderComparison_final.json" \
  --efficiency "$FIG/efficiency_final.json" --out "$FIG/main" "${MATCH[@]}" --sort per \
  --decoders RNN LSTM96 LSNN Kalman Wiener
echo "=== appendix figures"
MPLBACKEND=Agg python3 plotDecoderFigures.py --compare "$FIG/decoderComparison_final.json" \
  --efficiency "$FIG/efficiency_final.json" --out "$FIG/all" "${MATCH[@]}" --sort per \
  --decoders RNN LSTM96 GRU96 DelayNet LSNN Kalman Wiener

echo "=== videos and stills"
cd "$HERE/trialViewer"
python3 make_video.py "$VIEW"/*.json --outDir ../../figures/viewer/videos_final --lanes RNN LSTM SNN Kalman Wiener --combine
python3 make_video.py "$VIEW"/*.json --outDir ../../figures/viewer/stills_final --lanes RNN LSTM SNN Kalman Wiener --still
echo "=== done"
ls -la "$FIG"/main_*.png "$FIG"/all_*.png ../../figures/viewer/videos_final ../../figures/viewer/stills_final
