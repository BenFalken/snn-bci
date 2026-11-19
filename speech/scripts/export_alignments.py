"""
Phoneme timing for the speech lag analysis (figures/collect_lag_tuning.py speech): runs a trained speech SNN over the
TRAINING sentences of its sessions and force-aligns each sentence's known phonemes to its frame-by-frame outputs
(CTC Viterbi: the most likely path that emits exactly the true phoneme sequence). Only the alignments are written,
not the logits, so the file is small; the lag analysis then reads the same training sentences from snnData itself.

    python export_alignments.py --ckpt CKPT --data-dir <baseDir>/derived/snnData --out alignments.npz [--sessions S ..]

alignments.npz:
    path        (sum frames,) int16   per output frame: the aligned token (phoneme id 0-39, or blank = n_classes - 1)
    frame_off   (N + 1,)              trial i's frames are path[frame_off[i]:frame_off[i+1]]
    labels      (sum labels,) int16 + label_off (N + 1,): each trial's true phoneme sequence
    sessions    (N,) str; trial_idx (N,): the trial's index in its session's train.npz; input_len (N,): its bins
    frame_stride, frame_kernel, blank, greedy_per (the SNN's own phoneme error on these sentences, a sanity check)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snnDecoder.inference import load_snn_model, snn_inference  # noqa: E402


def ctc_viterbi(logp, labels, blank):
    """Most likely CTC path for `labels` through log-probs logp (T, C): (T,) token per frame (blank or phoneme)."""
    T = len(logp)
    ext = [blank]
    for l in labels:
        ext += [int(l), blank]
    S = len(ext)
    NEG = -1e30
    a = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int32)
    a[0, 0] = logp[0, ext[0]]
    if S > 1:
        a[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(a[t - 1, s], s)]
            if s >= 1:
                cands.append((a[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                cands.append((a[t - 1, s - 2], s - 2))
            v, b = max(cands)
            a[t, s] = v + logp[t, ext[s]]
            bp[t, s] = b
    s = S - 1 if S == 1 or a[T - 1, S - 1] >= a[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = ext[s]
        s = bp[t, s]
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", default="train")
    ap.add_argument("--sessions", nargs="*", default=None, help="default: every session the checkpoint has")
    a = ap.parse_args()
    _, ck = load_snn_model(a.ckpt)
    sessions = a.sessions or list(ck["sessions"])
    sessions = [s for s in sessions if os.path.isfile(os.path.join(a.data_dir, s, f"{a.partition}.npz"))]
    out = snn_inference(a.ckpt, a.data_dir, sessions, a.partition, count_ops=False)
    blank = out["logits"].shape[2] - 1
    paths, labels, trial_idx, seen = [], [], [], {}
    for i in range(len(out["logitLengths"])):
        n = int(out["logitLengths"][i])
        lg = out["logits"][i, :n].astype(np.float64)
        logp = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)          # log-softmax
        lab = out["trueSeqs"][i, :int(out["trueSeqLengths"][i])]
        if len(lab) * 2 + 1 > 2 * n + 1:                                    # more phonemes than frames: cannot align
            paths.append(np.full(n, blank)); labels.append(lab)
        else:
            paths.append(ctc_viterbi(logp, lab, blank)); labels.append(lab)
        s = out["sessions"][i]
        trial_idx.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  aligned {i + 1}/{len(out['logitLengths'])}", flush=True)
    off = np.concatenate([[0], np.cumsum([len(p) for p in paths])])
    loff = np.concatenate([[0], np.cumsum([len(l) for l in labels])])
    np.savez_compressed(a.out, path=np.concatenate(paths).astype(np.int16), frame_off=off,
                        labels=np.concatenate(labels).astype(np.int16), label_off=loff,
                        sessions=np.array(out["sessions"]), trial_idx=np.array(trial_idx),
                        input_len=out["inputLengths"], frame_stride=out["frameStride"],
                        frame_kernel=out["frameKernel"], blank=blank, greedy_per=out["cer"], ckpt=a.ckpt)
    print(f"wrote {a.out}: {len(paths)} sentences from {len(sessions)} sessions; frame stride {out['frameStride']}, "
          f"kernel {out['frameKernel']}; the SNN's greedy phoneme error on them {out['cer']:.4f}")


if __name__ == "__main__":
    main()
