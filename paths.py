"""
The repository's paths, in one place: the roots in paths.yaml (overridden key by key by paths.local.yaml, if present)
and the code folders of the repository itself.

    from paths import WORKSPACE, AUDITORY_DATA      # e.g. WORKSPACE / "final_results"
    python3 paths.py                                # print every root and whether it exists
    eval "$(python3 paths.py --sh)"                 # the same roots as shell variables (SNNBCI_WORKSPACE, ...)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load():
    cfg = {}
    for name in ("paths.yaml", "paths.local.yaml"):
        f = ROOT / name
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():           # flat key: value pairs; no YAML library needed
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                if v.strip():
                    cfg[k.strip()] = v.strip()
    return {k: Path(os.path.expanduser(v)) for k, v in cfg.items()}


_CFG = _load()
WORKSPACE = _CFG["workspace"]
AUDITORY_REPO = _CFG["auditory_repo"]
AUDITORY_DATA = _CFG["auditory_data"]
SNN_RUNS = _CFG["snn_runs"]
RAW_AUDITORY = _CFG["raw_auditory"]
MOTOR_DATA = _CFG["motor_data"]
MOTOR_RESULTS = _CFG["motor_results"]
MOTOR_LSNN_DATA = _CFG["motor_lsnn_data"]
OSCAR_MOTOR_DATA = _CFG["oscar_motor_data"]
OSCAR_HOME = _CFG["oscar_home"]
OSCAR_SCRATCH = _CFG["oscar_scratch"]
TTS_ENV = _CFG["tts_env"]
TTS_VOICES = _CFG["tts_voices"]

# code, inside this repository
CORE = ROOT / "core"                       # the shared LSNN / Speck library (snndec) and the chip tools
SPEECH_SNN = ROOT / "speech" / "SNNDecoder"
MOTOR = ROOT / "motor"
AUDIO = ROOT / "audio"
FIGURES = ROOT / "figures"

ALL = dict(ROOT=ROOT, WORKSPACE=WORKSPACE, AUDITORY_REPO=AUDITORY_REPO, AUDITORY_DATA=AUDITORY_DATA, SNN_RUNS=SNN_RUNS,
           RAW_AUDITORY=RAW_AUDITORY, MOTOR_DATA=MOTOR_DATA, MOTOR_RESULTS=MOTOR_RESULTS,
           MOTOR_LSNN_DATA=MOTOR_LSNN_DATA, OSCAR_MOTOR_DATA=OSCAR_MOTOR_DATA, OSCAR_HOME=OSCAR_HOME, OSCAR_SCRATCH=OSCAR_SCRATCH, TTS_ENV=TTS_ENV,
           TTS_VOICES=TTS_VOICES, CORE=CORE,
           SPEECH_SNN=SPEECH_SNN, MOTOR=MOTOR, AUDIO=AUDIO, FIGURES=FIGURES)

if __name__ == "__main__":
    if "--sh" in sys.argv:
        for k, v in ALL.items():
            print(f'export SNNBCI_{k}="{v}"')
    else:
        for k, v in ALL.items():
            print(f"{k:14s} {'ok     ' if v.exists() else 'missing'} {v}")
