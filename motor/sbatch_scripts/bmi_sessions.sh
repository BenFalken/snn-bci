# bmi_sessions <subject>: that subject's session ids, sorted, one per line: the
# raw .mat files under $DATA_ROOT/raw/bmi/<subject> plus sessions/bmi_<subject>.txt,
# so a session whose raw file is missing (e.g. purged from scratch) still gets
# an array task, which downloads it again (single_subject_pipeline.py).
# Source from sbatch_scripts/. Every step tolerates a missing folder or file:
# callers run under set -e -o pipefail, which bash carries into <(...) and
# subshells, so one failing find would otherwise end the listing silently.
bmi_sessions() {
    {
        { find "${DATA_ROOT}/raw/bmi/$1" -mindepth 1 -maxdepth 1 -name '*.mat' 2>/dev/null || true; } \
            | sed 's|.*/||; s/\.mat$//'
        if [ -f "sessions/bmi_$1.txt" ]; then grep -v '^[[:space:]]*\(#\|$\)' "sessions/bmi_$1.txt" || true; fi
    } | sort -u
}
