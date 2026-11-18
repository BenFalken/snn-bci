#!/bin/bash
# 2026-11-17 audio retraining on the expanded cohorts (20 words + bloop/pop/goop), both monkeys:
#   (1) every pipeline decoder incl. the LSNN (AuditoryDecoding, Docker), Neil then Gomez
#   (2) the Speck SNN P5, 3 seeds per monkey (snn-decoders), each starting once its monkey's windows exist
#   bash experiments/audio_final/run_novel.sh > runs/audio_novel.log 2>&1      (from the snn-decoders root)
AD=$HOME/Documents/GitHub/AuditoryDecoding
SD=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$SD/runs"
( cd "$AD" && for m in Neil Gomez; do
    echo "[docker start] $m $(date +%H:%M)"
    python3 run_auditory_experiment.py --subject $m --region aSTG --date ALLWORDS_NOVEL_DAYNORM --all-decoders \
        > pipeline_data/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/run_all.log 2>&1
    echo "[docker done] $m $(date +%H:%M) exit $?"
  done ) &
for m in Neil Gomez; do
  P=$AD/pipeline_data/$m/aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline
  until [ -f $P/window_targets_test.npy ] && [ -f $P/window_features_train.npy ]; do sleep 60; done
  sleep 120     # let the split finish writing
  echo "[speck start] $m $(date +%H:%M)"
  ( cd "$SD" && DATA=$P ROOT=runs/audio_novel_$m P=3 THREADS=3 LISTS=experiments/audio_final/runs_speck_P5.txt \
      bash experiments/audio_final/run_local.sh )
  echo "[speck done] $m $(date +%H:%M)"
done
wait
echo "ALL DONE $(date +%H:%M)"
