"""
pipeline_paths.py

Shared folder-structure and CLI-identity conventions for the auditory
decoding pipeline. Every other script (prepare_auditory_data.py,
run_auditory_experiment.py, analyze_results.py) imports this module rather
than each inventing its own paths — that's what keeps them from silently
drifting out of sync with each other.

Folder layout:

    <data_root>/
        <subject>/
            <region>/
                <date>/
                    prep/       per-session audio_raw.npy, mel_target.npy,
                                mel_config.json (intermediate outputs of
                                prepare_auditory_data.py)
                    qc/         unified, x-aligned QC figures — one PNG per
                                candidate trial, per session
                    pipeline/   every Docker job config, every intermediate
                                / prediction .npy, every trained model —
                                the working directory run_auditory_experiment.py
                                mounts into containers
                    results/    regression_metrics.csv per decoder, plus:
                        analysis/   per-date comparison plots
        aggregate/
            analysis/   cross-date / cross-subject comparison plots,
                        built by analyze_results.py --mode aggregate

Every script should construct a PipelinePaths object from the same
--subject/--region/--date/--data-root CLI arguments (added via
add_identity_args) so that a given (subject, region, date) always resolves
to the same folders no matter which script touches it.
"""

import argparse
import sys
from pathlib import Path

# The data root: auditory_data in the repository's paths.yaml; ./pipeline_data where paths.py is absent (e.g. inside
# a decoder's Docker container, which only ever receives absolute paths).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import AUDITORY_DATA, RAW_AUDITORY, AUDITORY_REPO  # noqa: E402
    DEFAULT_DATA_ROOT = str(AUDITORY_DATA)
except Exception:
    AUDITORY_DATA = RAW_AUDITORY = AUDITORY_REPO = None
    DEFAULT_DATA_ROOT = "pipeline_data"


def add_identity_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the --subject/--region/--date/--data-root arguments shared by
    every pipeline script. Call this from each script's own argparse setup."""
    parser.add_argument(
        "--subject", required=True,
        help="Subject identifier, e.g. 'Neil'",
    )
    parser.add_argument(
        "--region", required=True,
        help="Brain region label, e.g. 'aSTG', 'STG', 'PSTG', 'CPB' — free-form, "
             "just used as a folder name, not validated against a fixed list.",
    )
    parser.add_argument(
        "--date", required=True,
        help="Recording date / session-day folder name, e.g. '07_19_11'. "
             "Only .ns5 files inside this one date folder are processed — "
             "sessions are concatenated within a date, never across dates.",
    )
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT,
        help="Root directory under which all subject/region/date output "
             "folders are created. Default: auditory_data in paths.yaml",
    )
    return parser


class PipelinePaths:
    """Resolves and creates the full folder tree for one
    (subject, region, date) combination."""

    def __init__(self, data_root, subject, region, date):
        self.subject = subject
        self.region = region
        self.date = date

        self.data_root = Path(data_root).resolve()
        self.base = self.data_root / subject / region / date

        self.prep = self.base / "prep"
        self.qc = self.base / "qc"
        self.pipeline = self.base / "pipeline"
        self.results = self.base / "results"
        self.analysis = self.results / "analysis"

        self.aggregate = self.data_root / "aggregate" / "analysis"

        for d in (self.prep, self.qc, self.pipeline, self.results,
                  self.analysis, self.aggregate):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args):
        return cls(args.data_root, args.subject, args.region, args.date)

    def label(self):
        """Short human-readable identity string, used in plot titles/legends."""
        return f"{self.subject}/{self.region}/{self.date}"

    def __repr__(self):
        return f"PipelinePaths({self.label()}, base={self.base})"


def discover_all_runs(data_root, include_combined=False):
    """Walks <data_root>/*/*/*/results/ looking for completed runs (any
    directory containing at least one *_regression_metrics.csv). Returns a
    list of PipelinePaths, one per (subject, region, date) found — used by
    analyze_results.py's aggregate mode to find every date that's been run
    without the caller having to enumerate them manually.

    By default, synthetic "dates" produced by combine_days.py (named
    COMBINED_<mode>[...]) are excluded — they represent a pooled multi-day
    dataset, not a comparable single recording day, and mixing them into a
    chronological cross-date comparison would distort it. Pass
    include_combined=True to include them anyway (e.g. for a script that
    specifically wants to inspect combined-variant results)."""
    data_root = Path(data_root)
    found = []
    for csv_path in sorted(data_root.glob("*/*/*/results/*_regression_metrics.csv")):
        results_dir = csv_path.parent
        date_dir = results_dir.parent
        region_dir = date_dir.parent
        subject_dir = region_dir.parent
        key = (subject_dir.name, region_dir.name, date_dir.name)
        if not include_combined and key[2].startswith("COMBINED"):
            continue
        if key not in [(p.subject, p.region, p.date) for p in found]:
            found.append(PipelinePaths(data_root, *key))
    return found
