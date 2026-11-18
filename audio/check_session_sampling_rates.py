"""
Reads only the sampling rate metadata from each .ns5 session (fast — does
not load full data) to check whether sessions actually share a consistent
sampling rate, or whether some differ in a way that would corrupt
STFT/mel-spectrogram computation without showing up in raw waveform
statistics (std/min/max look the same regardless of what rate you assume).

Usage:
    python3 check_session_sampling_rates.py
"""

from pathlib import Path

from neo.io import BlackrockIO
from pipeline_paths import RAW_AUDITORY  # noqa: E402  (paths.yaml)

BASE_DIR = RAW_AUDITORY / "Neil/Complex/aSTG"

skips = {".DS_Store", "desktop.ini"}
days = sorted(d for d in BASE_DIR.iterdir() if d.name not in skips)

print(f"{'Session':<35} {'fs (Hz)':>15}")
print("-" * 52)

rates_seen = set()
for day in days:
    for session_path in sorted(day.glob("*.ns5")):
        session_tag = f"{day.name}_{session_path.stem}"
        try:
            reader = BlackrockIO(filename=str(session_path))
            blk = reader.read_block()

            audio_sig = None
            for asig in blk.segments[0].analogsignals:
                if asig.shape[1] < 90:
                    audio_sig = asig
                    break

            fs = float(audio_sig.sampling_rate.rescale("Hz").magnitude)
            rates_seen.add(fs)
            print(f"{session_tag:<35} {fs:>15.2f}")
        except Exception as e:
            print(f"{session_tag:<35} {'ERROR: ' + str(e):>15}")

print(f"\nDistinct sampling rates seen: {sorted(rates_seen)}")
if len(rates_seen) > 1:
    print("MISMATCH FOUND — not every session shares the same sampling rate. "
          "This would corrupt STFT/mel computation for whichever sessions "
          "differ from the assumed rate, without affecting raw waveform "
          "statistics at all.")
else:
    print("All sessions report the same sampling rate — this is not the cause.")