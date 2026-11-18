#!/bin/bash
#
# run_all.sh — drives the full auditory decoding pipeline across every
# (subject, region, date) combination configured below, then produces the
# cross-date aggregate dashboard.
#
# Usage:
#     bash run_all.sh
#     bash run_all.sh --decoders wiener_filter,dense_nn,lstm_rnn --epochs 50
#     bash run_all.sh --force   # re-run dates even if already completed
#
# Edit the RUNS array below to add subjects/regions — dates within each
# raw_base_dir are discovered automatically (one subfolder per date,
# containing .ns5 files).

source "$(cd "$(dirname "$0")" && pwd)/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -uo pipefail  # NOT -e: one date's failure shouldn't kill the whole batch

# ---------------------------------------------------------------------------
# CONFIG — edit this list to add subjects/regions.
# Format: "subject|region|raw_base_dir"
# raw_base_dir must be the directory containing one subfolder per date
# (e.g. raw_base_dir/07_19_11/*.ns5).
# ---------------------------------------------------------------------------
RUNS=(
    "Neil|aSTG|$SNNBCI_RAW_AUDITORY/Neil/Complex/aSTG"
    # "OtherSubject|PSTG|/path/to/other/subject/pstg/data"
)

# ---------------------------------------------------------------------------
# Defaults — override with flags below
# ---------------------------------------------------------------------------
DATA_ROOT="$SNNBCI_AUDITORY_DATA"
DECODERS="wiener_filter,wiener_cascade,kalman_filter,dense_nn,simple_rnn,gru_rnn,lstm_rnn,lsnn"
EPOCHS=100
UNITS=1024
FORCE=false
N_QC_TRIALS=5

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --decoders) DECODERS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --units) UNITS="$2"; shift 2 ;;
        --n-qc-trials) N_QC_TRIALS="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LOG_DIR="run_all_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"

echo "Logging full output to $LOG_FILE"
echo "Data root: $DATA_ROOT"
echo "Decoders:  $DECODERS"
echo "Epochs:    $EPOCHS   Units: $UNITS"
echo "Force re-run: $FORCE"
echo ""

exec > >(tee -a "$LOG_FILE") 2>&1

SUCCEEDED=()
FAILED=()
SKIPPED=()

# ---------------------------------------------------------------------------
# Main loop: subject/region -> auto-discovered dates -> full 2-stage pipeline
# ---------------------------------------------------------------------------
for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r SUBJECT REGION RAW_BASE_DIR <<< "$run_spec"

    if [[ ! -d "$RAW_BASE_DIR" ]]; then
        echo "WARNING: raw_base_dir does not exist, skipping: $RAW_BASE_DIR"
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "Subject: $SUBJECT   Region: $REGION"
    echo "Raw data: $RAW_BASE_DIR"
    echo "======================================================================"

    for date_dir in "$RAW_BASE_DIR"/*/; do
        DATE=$(basename "$date_dir")

        # Only treat this subfolder as a real date if it actually contains
        # .ns5 files -- skips .DS_Store-type junk and empty folders.
        if ! ls "$date_dir"*.ns5 >/dev/null 2>&1; then
            continue
        fi

        RUN_LABEL="$SUBJECT/$REGION/$DATE"
        METADATA_FILE="$DATA_ROOT/$SUBJECT/$REGION/$DATE/results/run_metadata.json"
        SKIP=false
        if [[ "$FORCE" == false && -f "$METADATA_FILE" ]]; then
            # Verify the metadata actually covers every decoder we're about to
            # request -- a file left over from an earlier partial/smoke-test
            # run would otherwise satisfy an existence-only check and skip
            # decoders that were never actually trained.
            python3 -c "
import json, sys
meta = json.load(open('$METADATA_FILE'))
requested = set('$DECODERS'.split(','))
done = set(meta.get('decoders_run', []))
sys.exit(0 if requested <= done else 1)
"
            if [[ $? -eq 0 ]]; then
                SKIP=true
            fi
        fi

        if [[ "$SKIP" == true ]]; then
            echo ""
            echo "--- $RUN_LABEL already completed with all requested decoders (found $METADATA_FILE), skipping. Use --force to redo. ---"
            SKIPPED+=("$RUN_LABEL")
            continue
        fi

        echo ""
        echo "----------------------------------------------------------------------"
        echo "Processing $RUN_LABEL"
        echo "----------------------------------------------------------------------"

        python3 prepare_auditory_data.py \
            --subject "$SUBJECT" --region "$REGION" --date "$DATE" \
            --raw-base-dir "$RAW_BASE_DIR" --data-root "$DATA_ROOT" \
            --n-qc-trials "$N_QC_TRIALS"
        PREP_STATUS=$?

        if [[ $PREP_STATUS -ne 0 ]]; then
            echo "FAILED (prepare_auditory_data.py): $RUN_LABEL"
            FAILED+=("$RUN_LABEL (prep)")
            continue
        fi

        python3 run_auditory_experiment.py \
            --subject "$SUBJECT" --region "$REGION" --date "$DATE" \
            --data-root "$DATA_ROOT" --decoders "$DECODERS" \
            --epochs "$EPOCHS" --units "$UNITS"
        EXPERIMENT_STATUS=$?

        if [[ $EXPERIMENT_STATUS -ne 0 ]]; then
            echo "FAILED (run_auditory_experiment.py): $RUN_LABEL"
            FAILED+=("$RUN_LABEL (experiment)")
            continue
        fi

        # Per-date analysis, so results are viewable immediately without
        # waiting for the whole batch to finish.
        python3 analyze_results.py --mode single \
            --subject "$SUBJECT" --region "$REGION" --date "$DATE" \
            --data-root "$DATA_ROOT"

        SUCCEEDED+=("$RUN_LABEL")
    done
done

# ---------------------------------------------------------------------------
# Final aggregate across every date that succeeded
# ---------------------------------------------------------------------------
# Disable nounset here: older bash (macOS ships 3.2 by default, from long
# before the 4.4 fix) treats "${ARRAY[@]}" on a genuinely EMPTY array as an
# unbound-variable error under `set -u`, even though the array was properly
# declared. We're past the point where catching a real unset-variable typo
# matters, so it's safe to relax for the reporting section below.
set +u

echo ""
echo "======================================================================"
echo "BATCH SUMMARY"
echo "======================================================================"
echo "Succeeded (${#SUCCEEDED[@]}):"
printf '  %s\n' "${SUCCEEDED[@]}"
echo ""
echo "Skipped, already done (${#SKIPPED[@]}):"
printf '  %s\n' "${SKIPPED[@]}"
echo ""
echo "Failed (${#FAILED[@]}):"
printf '  %s\n' "${FAILED[@]}"

if [[ ${#SUCCEEDED[@]} -gt 0 || ${#SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo "======================================================================"
    echo "Running cross-date aggregate analysis"
    echo "======================================================================"
    python3 analyze_results.py --mode aggregate --data-root "$DATA_ROOT"
    echo ""
    echo "Aggregate dashboard: $DATA_ROOT/aggregate/analysis/aggregate_dashboard.png"
else
    echo ""
    echo "No completed dates -- skipping aggregate analysis."
fi

echo ""
echo "Full log saved to: $LOG_FILE"
