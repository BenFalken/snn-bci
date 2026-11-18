#!/bin/bash
set -eo pipefail
#
# Checks the data, sets up the env, exports the SNN data, runs a 1-epoch smoke
# test, then (only with SUBMIT=1) submits SNN pretraining + fine-tuning to Slurm.
#
#   ./run_all.sh 2>&1 | tee run_all.log                  # everything up to the smoke test
#   SUBMIT=1 SKIP_SMOKE=1 ./run_all.sh 2>&1 | tee -a run_all.log   # then submit training
#
# Run with no env active (conda deactivate / deactivate); the script activates its own.

############################################################
# 0. Paths: edit BASE_DIR (or pass it: BASE_DIR=/path ./run_all.sh)
############################################################
export REPO_DIR=/users/$USER/speechBCI-main
export BASE_DIR="${BASE_DIR:-/users/$USER/scratch/speechBCI_dryad}"   # folder holding competitionData/, derived/, languageModel/
export CONDA_ENV=speechBCI
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"                         # Oscar's GPU partition

[ -d "$BASE_DIR/competitionData" ] || { echo "ERROR: BASE_DIR=$BASE_DIR has no competitionData/; edit BASE_DIR at the top"; exit 1; }
[ -d "$REPO_DIR/SNNDecoder" ] || { echo "ERROR: REPO_DIR=$REPO_DIR has no SNNDecoder/"; exit 1; }

############################################################
# 1. Check the data
############################################################
cd "$BASE_DIR"
echo "--- unextracted Dryad archives (extract with: tar -xzf <file>)"
ls -lh *.tar.gz 2>/dev/null || echo "none"

echo "--- competitionData (.mat files per partition; expect 24 train sessions)"
for p in train test competitionHoldOut; do
  printf "%-20s %s files\n" "$p" "$(ls competitionData/$p/*.mat 2>/dev/null | wc -l)"
done

echo "--- derived/tfRecords (sessions with train/ and test/ records)"
if [ -d derived/tfRecords ]; then
  for s in derived/tfRecords/*/; do
    printf "%-18s train:%s test:%s holdout:%s\n" "$(basename "$s")" \
      "$(ls "$s"/train/*.tfrecord 2>/dev/null | wc -l)" \
      "$(ls "$s"/test/*.tfrecord 2>/dev/null | wc -l)" \
      "$(ls "$s"/competitionHoldOut/*.tfrecord 2>/dev/null | wc -l)"
  done
else
  echo "MISSING: extract derived.tar.gz, or build them in step 3b"
fi

echo "--- language model (needed only for rnn_step3)"
ls languageModel/TLG.fst languageModel/words.txt 2>/dev/null || echo "MISSING: extract languageModel.tar.gz"

echo "--- GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no nvidia-smi (CPU/login node)"

############################################################
# 2. Environment (created once; later runs reuse it)
############################################################
source ~/miniconda3/etc/profile.d/conda.sh
# test the env folder rather than `conda env list | grep -q`: with pipefail, grep -q
# closing the pipe early can make that test fail and `conda create -y` recreate the env
if [ ! -d "$(conda info --base)/envs/$CONDA_ENV" ]; then
  conda create -y -n "$CONDA_ENV" python=3.9
fi
conda activate "$CONDA_ENV"

pip install -e "$REPO_DIR/NeuralDecoder"   # TF 2.7, torch 1.13.1, numpy 1.25, g2p_en, ...
pip install -e "$REPO_DIR/SNNDecoder"      # sinabs, pyyaml, psutil
pip install "protobuf==3.20.3"             # TF 2.7 fails to import with protobuf >= 4

# Sanity check: every library imports (GPUs: [] / CUDA: False is expected on a login node)
python - <<'EOF'
import tensorflow as tf, torch, sinabs, numpy, snnDecoder, neuralDecoder
print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices('GPU'))
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("sinabs", sinabs.__version__, "| numpy", numpy.__version__)
EOF

############################################################
# 3. Data for the SNN
############################################################
# 3b. ONLY if derived/tfRecords is missing: build it from competitionData
#     (the same code as rnn_step1; g2p_en downloads NLTK data on first use, so this needs internet)
if [ ! -d "$BASE_DIR/derived/tfRecords" ]; then
  cd "$REPO_DIR/AnalysisExamples"
  python - <<EOF
from makeTFRecordsFromSession import makeTFRecordsFromCompetitionFiles
from getSpeechSessionBlocks import getSpeechSessionBlocks
base = "$BASE_DIR"
for s in getSpeechSessionBlocks():
    makeTFRecordsFromCompetitionFiles(s[0], base + '/competitionData', base + '/derived/tfRecords/' + s[0])
EOF
fi

# 3c. Export the TFRecords to .npz (identical features/labels for both decoders; skips existing files)
cd "$REPO_DIR/AnalysisExamples"
python exportTFRecordsToNpz.py --tfRecordDir "$BASE_DIR/derived/tfRecords" --outDir "$BASE_DIR/derived/snnData"

# The 19 training sessions should each now have train.npz and test.npz
python - <<EOF
import numpy as np, os
from snnDecoder.dataset import BASELINE_SESSIONS
d = "$BASE_DIR/derived/snnData"
missing = 0
for s in BASELINE_SESSIONS:
    row = []
    for p in ['train', 'test']:
        f = os.path.join(d, s, p + '.npz')
        if os.path.isfile(f):
            z = np.load(f)
            row.append(f"{p}: {len(z['featureLens'])} trials, {z['features'].shape[1]} feats")
        else:
            row.append(f"{p}: MISSING")
            missing += 1
    print(s, ' | '.join(row))
if missing:
    raise SystemExit(f"{missing} partitions missing from {d}")
EOF

############################################################
# 4. Fix the RNN config paths (only needed for training the RNN in rnn_step2)
############################################################
sed -i "s#/oak/stanford/groups/henderj/fwillett/speechPaperRelease_08_20#$BASE_DIR#g" \
  "$REPO_DIR/NeuralDecoder/neuralDecoder/configs/dataset/speech_release_baseline.yaml"

############################################################
# 5. Smoke test: 1 epoch on 1 session, checks the whole training path
#    (CPU on a login node: several minutes; SKIP_SMOKE=1 skips it)
############################################################
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  cd "$REPO_DIR/SNNDecoder"
  python -m snnDecoder.train \
    --config snnDecoder/configs/speech_pooled.yaml \
    --data-dir "$BASE_DIR/derived/snnData" \
    --sessions t12.2022.06.02 \
    --checkpoint-dir "$BASE_DIR/derived/snns/smoke_test" \
    --reset-type hard --epochs 1 --verbose
fi

[ "${SUBMIT:-0}" = "1" ] || { echo "Smoke test done. Check the output above, then rerun with SUBMIT=1 SKIP_SMOKE=1 to submit training."; exit 0; }

############################################################
# 6. Full training: pretrain (hard reset) then fine-tune (soft reset)
#    the fine-tune job starts only if pretraining succeeds
############################################################
conda deactivate   # the jobs activate the env themselves (snn_config.sh: activate_env)
cd "$REPO_DIR/SNNDecoder/scripts"
mkdir -p snn_logs
PRE=$(sbatch --parsable -p "$SBATCH_PARTITION" --export=ALL,BASE_DIR="$BASE_DIR",CONDA_ENV="$CONDA_ENV" run_snn_pretrain.sbatch)
echo "pretrain job: $PRE"
FT=$(sbatch --parsable -p "$SBATCH_PARTITION" --dependency=afterok:"$PRE" --export=ALL,BASE_DIR="$BASE_DIR",CONDA_ENV="$CONDA_ENV" run_snn_finetune.sbatch)
echo "fine-tune job: $FT (waits for $PRE)"
squeue -u "$USER"
echo "follow progress with: tail -f $REPO_DIR/SNNDecoder/scripts/snn_logs/SNN_PRETRAIN_${PRE}.out"
