#!/bin/bash
#
# run_combined_variants.sh -- runs the full decoder suite on each combined
# multi-day dataset variant considered methodologically valid, back to back.
#
# Two variants are run by default:
#   naive                        Plain sequential concatenation, matching
#                                 the original paper's own method exactly.
#                                 Simplest, most faithful baseline.
#   per_day_normalized_causal    Per-day z-scoring + outlier clipping,
#                                 statistics frozen from each day's leading
#                                 80% calibration window only. Fully causal
#                                 and deployable -- see design_justification
#                                 .tex for why this is the recommended
#                                 production variant.
#
# The whole-day (non-causal) per_day_normalized variant is intentionally
# NOT run here -- it was found to add no benefit over the causal version
# once tested head to head, so it's kept available in combine_days.py for
# reference/reproducibility but isn't part of the standard pipeline.
#
# Usage:
#     bash run_combined_variants.sh
#     bash run_combined_variants.sh --force

source "$(cd "$(dirname "$0")" && pwd)/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
set -uo pipefail

SUBJECT="Neil"
REGION="aSTG"
DATA_ROOT="$SNNBCI_AUDITORY_DATA"
DECODERS="wiener_filter,wiener_cascade,kalman_filter,dense_nn,simple_rnn,gru_rnn,lstm_rnn,lsnn"
EPOCHS=100
UNITS=1024
CLIP_STD=5.0
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject) SUBJECT="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --decoders) DECODERS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --units) UNITS="$2"; shift 2 ;;
        --clip-std) CLIP_STD="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LOG_DIR="run_combined_variants_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_combined_variants_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Each entry: "combine_days.py --mode value|extra combine_days.py args|resulting date name"
VARIANTS=(
    "naive||COMBINED_naive"
    "per_day_normalized_causal|--clip-std ${CLIP_STD}|COMBINED_per_day_normalized_causal_clip${CLIP_STD}"
)

SUCCEEDED=()
FAILED=()

for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r MODE EXTRA_ARGS OUTPUT_DATE <<< "$variant"

    echo ""
    echo "======================================================================"
    echo "Variant: $MODE  (output date: $OUTPUT_DATE)"
    echo "======================================================================"

    METADATA_FILE="$DATA_ROOT/$SUBJECT/$REGION/$OUTPUT_DATE/results/run_metadata.json"
    SKIP=false
    if [[ "$FORCE" == false && -f "$METADATA_FILE" ]]; then
        # Don't just check the file exists -- verify it actually covers every
        # decoder we're about to request. A metadata file left over from an
        # earlier single-decoder smoke test (e.g. --decoders wiener_filter)
        # would otherwise satisfy this check and silently skip the other six.
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
        echo "Already completed with all requested decoders (found $METADATA_FILE), skipping. Use --force to redo."
        SUCCEEDED+=("$OUTPUT_DATE (skipped, already done)")
        continue
    fi

    python3 combine_days.py --subject "$SUBJECT" --region "$REGION" \
        --data-root "$DATA_ROOT" --mode "$MODE" $EXTRA_ARGS
    if [[ $? -ne 0 ]]; then
        echo "FAILED: combine_days.py for $MODE"
        FAILED+=("$OUTPUT_DATE (combine)")
        continue
    fi

    python3 run_auditory_experiment.py --subject "$SUBJECT" --region "$REGION" \
        --date "$OUTPUT_DATE" --data-root "$DATA_ROOT" \
        --decoders "$DECODERS" --epochs "$EPOCHS" --units "$UNITS"
    if [[ $? -ne 0 ]]; then
        echo "FAILED: run_auditory_experiment.py for $OUTPUT_DATE"
        FAILED+=("$OUTPUT_DATE (experiment)")
        continue
    fi

    python3 analyze_results.py --mode single --subject "$SUBJECT" --region "$REGION" \
        --date "$OUTPUT_DATE" --data-root "$DATA_ROOT"

    SUCCEEDED+=("$OUTPUT_DATE")
done

set +u
echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Succeeded (${#SUCCEEDED[@]}):"
printf '  %s\n' "${SUCCEEDED[@]}"
echo ""
echo "Failed (${#FAILED[@]}):"
printf '  %s\n' "${FAILED[@]}"

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo ""
    echo "======================================================================"
    echo "Comparing combined variants directly"
    echo "======================================================================"
    python3 compare_combined_variants.py --subject "$SUBJECT" --region "$REGION" --data-root "$DATA_ROOT"
fi
