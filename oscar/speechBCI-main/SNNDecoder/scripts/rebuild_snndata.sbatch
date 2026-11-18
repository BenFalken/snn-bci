#!/bin/bash
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -J SNNDATA
#SBATCH -o snn_logs/SNNDATA_%j.out
#SBATCH -e snn_logs/SNNDATA_%j.out
#
# Rebuilds derived/snnData (emptied on Oscar 2026-11-15 16:58, with derived/tfRecords) from the release's raw data,
# then checks it against fingerprints of the copy the earlier rounds trained on (verify_snndata.py). Submit from this
# directory:
#   sbatch --export=ALL,BASE_DIR=/users/$USER/scratch/speechBCI_dryad rebuild_snndata.sbatch
# Source, first that exists:
#   1. $BASE_DIR/derived.tar.gz (the Dryad release's derived folder): its tfRecords are extracted
#   2. $BASE_DIR/competitionData/{train,test,competitionHoldOut}/*.mat: tfRecords rebuilt as rnn_step1 does
# then AnalysisExamples/exportTFRecordsToNpz.py -> derived/snnData. Needs TensorFlow (and for 2, g2p_en and the
# NeuralDecoder package): TF_ENV names the conda env that has them (default: the SNN env, speechBCI).
# Old empty folders are moved aside (derived/snnData.empty_<date>), never deleted.
set -eo pipefail
B="${BASE_DIR:?set BASE_DIR}"
HERE=$(pwd)
AE="$(cd ../.. && pwd)/AnalysisExamples"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${TF_ENV:-speechBCI}"
PY=$(command -v python)
"$PY" -c "import tensorflow as tf; print('tensorflow', tf.__version__)" || {
    echo "ERROR: no TensorFlow in conda env ${TF_ENV:-speechBCI}; resubmit with --export=ALL,BASE_DIR=$B,TF_ENV=<env with tensorflow>" >&2; exit 1; }
STAMP=$(date +%Y%m%d_%H%M)
for d in tfRecords snnData; do
    if [ -d "$B/derived/$d" ] && [ -z "$(find "$B/derived/$d" -type f -size +0 -print -quit)" ]; then
        mv "$B/derived/$d" "$B/derived/$d.empty_$STAMP"; echo "moved empty derived/$d aside"
    fi
done
if [ -z "$(find "$B/derived/tfRecords" -name '*.tfrecord' -size +0 -print -quit 2>/dev/null)" ]; then
    if [ -f "$B/derived.tar.gz" ]; then
        echo "== extracting tfRecords from derived.tar.gz"
        mkdir -p "$B/tmp_derived_$STAMP"
        tar -xzf "$B/derived.tar.gz" -C "$B/tmp_derived_$STAMP" --wildcards '*tfRecords*'
        src=$(find "$B/tmp_derived_$STAMP" -type d -name tfRecords | head -1)
        [ -n "$src" ] || { echo "ERROR: no tfRecords folder inside derived.tar.gz" >&2; exit 1; }
        mkdir -p "$B/derived"; mv "$src" "$B/derived/tfRecords"; rm -rf "$B/tmp_derived_$STAMP"
    elif ls "$B"/competitionData/train/*.mat >/dev/null 2>&1; then
        echo "== rebuilding tfRecords from competitionData"
        cd "$AE"
        "$PY" - "$B" <<'EOF'
import sys
from makeTFRecordsFromSession import makeTFRecordsFromCompetitionFiles
from getSpeechSessionBlocks import getSpeechSessionBlocks
b = sys.argv[1]
for s in getSpeechSessionBlocks():
    print(s[0], flush=True)
    makeTFRecordsFromCompetitionFiles(s[0], b + '/competitionData', b + '/derived/tfRecords/' + s[0])
EOF
        cd "$HERE"
    else
        echo "ERROR: neither $B/derived.tar.gz nor $B/competitionData/train/*.mat exists: the raw data must be fetched again" >&2
        exit 1
    fi
fi
echo "== exporting to snnData"
"$PY" "$AE/exportTFRecordsToNpz.py" --tfRecordDir "$B/derived/tfRecords" --outDir "$B/derived/snnData"
echo "== verifying"
"$PY" verify_snndata.py "$B/derived/snnData"
echo "SNNDATA REBUILT AND VERIFIED"
