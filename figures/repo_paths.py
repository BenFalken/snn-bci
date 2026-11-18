"""Makes the repository's paths.py importable from the figure scripts: from repo_paths import WORKSPACE, ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403
