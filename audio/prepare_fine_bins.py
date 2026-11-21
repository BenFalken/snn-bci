"""
Threshold crossings of one recording day at 4 ms, aligned to the 40 ms frames of features.npy: features_4ms.npy,
shape (n_frames * 10, n_channels), where rows 10*i .. 10*i + 9 are the ten 4 ms sub-bins of 40 ms frame i.

It reuses prepare_auditory_data.py's own steps (load_session, the elliptic band-pass, the MAD threshold fitted on the
same training fraction, negative crossings), so the crossings are identical; only the bin width changes. Each
session's frame count was trimmed to its mel target when features.npy was built; that trim is recovered by matching
the recomputed 40 ms counts against features.npy, and the result is checked: the ten sub-bins of every frame must sum
exactly to that frame's count on every channel, or nothing is written.

    python3 audio/prepare_fine_bins.py --subject Neil --region aSTG --date 07_19_11 \
        --raw-base-dir /path/to/Auditory_cortex/Neil/Complex/aSTG
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_auditory_data as pad                       # noqa: E402
from pipeline_paths import PipelinePaths, add_identity_args  # noqa: E402

SUB = 10                                                  # 4 ms sub-bins per 40 ms frame


def fine_counts(path):
    """(counts at 40 ms (n40, C), counts at 4 ms (n40 * SUB, C)) for one .ns5 session."""
    _, _, spikes, fs = pad.load_session(path)
    frame = int(round(pad.BIN_MS / 1000.0 * fs))
    sub = frame // SUB
    assert sub * SUB == frame, f"a 40 ms frame ({frame} samples) does not split into {SUB} equal sub-bins"
    n40 = spikes.shape[0] // frame
    edges = np.arange(0, n40 * frame + 1, sub)
    fine = np.zeros((n40 * SUB, spikes.shape[1]), np.uint8)
    for ch in range(spikes.shape[1]):
        f = pad.bandpass_elliptic(spikes[:, ch].astype(np.float64), fs)
        thr = pad.compute_negative_thresholds(f[:, None])[0]
        x = pad.detect_negative_crossings(f, thr)
        fine[:, ch] = np.minimum(np.histogram(x, bins=edges)[0], 255)
    coarse = fine.reshape(n40, SUB, -1).sum(1)
    return coarse, fine


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_identity_args(p)
    p.add_argument("--raw-base-dir", required=True)
    a = p.parse_args()
    paths = PipelinePaths.from_args(a)
    ref = np.load(paths.pipeline / "features.npy")
    sessions = sorted((Path(a.raw_base_dir) / a.date).glob("*.ns5"))
    print(f"{paths.label()}: {len(sessions)} sessions; features.npy {ref.shape}")
    pieces, offset = [], 0
    computed = []
    for s in list(sessions):
        print(f"  {s.name} ...", flush=True)
        try:                                              # as prepare_auditory_data.py: a session that fails to load
            computed.append(fine_counts(s))               # was left out of features.npy, so it is left out here too
        except Exception as e:
            print(f"    FAILED ({type(e).__name__}: {e}); skipped, as when features.npy was built")
            sessions.remove(s)
    for k, (coarse, fine) in enumerate(computed):
        last = k == len(computed) - 1
        n = None
        for d in range(0, 4):                             # the trim to the mel target: at most a few frames
            m = len(coarse) - d
            if offset + m > len(ref) or not np.array_equal(coarse[:m], ref[offset:offset + m]):
                continue
            if last:
                if offset + m == len(ref):
                    n = m
                    break
            else:
                nxt = computed[k + 1][0]
                if np.array_equal(nxt[:5], ref[offset + m:offset + m + 5]):
                    n = m
                    break
        if n is None:
            raise SystemExit(f"session {sessions[k].name}: recomputed 40 ms counts do not match features.npy "
                             f"at offset {offset}; nothing written")
        pieces.append(fine[:n * SUB])
        print(f"    {n} frames (trimmed {len(coarse) - n})")
        offset += n
    fine = np.concatenate(pieces)
    assert offset == len(ref) and np.array_equal(fine.reshape(len(ref), SUB, -1).sum(1), ref), "check failed"
    out = paths.pipeline / "features_4ms.npy"
    np.save(out, fine)
    print(f"wrote {out} {fine.shape}: every frame's ten 4 ms sub-bins sum to its 40 ms count on every channel")


if __name__ == "__main__":
    main()
