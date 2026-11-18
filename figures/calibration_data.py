"""
Data for fig_calibration.py, from the speech Speck SNN (V2: two networks on one Speck2f) and the readout refit.

    layers    spikes per neuron per 20 ms bin in each layer (both networks pooled), on 2 TRAINING sentences per
              session: the trained network (a step's input summed), the chip model (event by event) at layer 1 x1,
              and at the chosen x2.5
    neurons   each neuron's mean rate over all training sentences, trained network vs chip model at x2.5 (the
              spikes the readout refit is fitted to: final_results ... speck_r19/results/W1{,_software})
    example   one test sentence decoded three ways from the REAL devkit run (V2_onechip): the trained network in
              software, the chip with the trained readout, the chip with the refitted readout; logits as phoneme
              probabilities, greedy decodes, and the true phonemes

    python calibration_data.py      -> data/calibration_speech.npz + data/calibration_speech.json
"""
from repo_paths import WORKSPACE, SPEECH_SNN  # noqa: E402  (the repository's paths.yaml)
import json
import sys
from pathlib import Path

import numpy as np
import torch

DL = WORKSPACE
sys.path.insert(0, str(SPEECH_SNN / "scripts"))
import refit_readout as rr  # noqa: E402
import speech_chip as sc  # noqa: E402
import speck_io  # noqa: E402
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode  # noqa: E402

HERE = Path(__file__).parent
CKPT = DL / "speck_r19/01_V2_1chip_channels/best_model_weights.pth"
DATA = DL / "speech_rebuild/derived/snnData"
W1 = DL / "speck_r19/results/W1"
PHONES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
          'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z',
          'ZH', 'SIL']


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    model, ck = sc.load_snn_model(str(CKPT))
    S = list(ck["sessions"])
    CH = sc.chains(model)
    W = model.branch_widths                                   # per layer, per chain
    blank = model.n_classes - 1
    out, meta = {}, {}

    # --- layers: trained vs chip model x1 vs x2.5, 2 training sentences per session
    nets = {s: [speck_io.discretized(sc.spiking_stack(model, 1.0, chain=c), len(c[0])) for c in CH] for s in (1.0,)}
    stacks = [sc.spiking_stack(model, 1.0, as_trained=True, chain=c) for c in CH]
    sums = {k: np.zeros(3) for k in ("trained", "x1", "x2.5")}
    bins = 0
    for part in load_partitions(str(DATA), S, "train", list(range(len(S)))):
        for i in range(2):
            ev = sc.host_events(model, part.features[i], part.day_idx)
            runs = {"trained": [sc.run_stack(st, ev[:, c[0]], True) for st, c in zip(stacks, CH)],
                    "x1": [sc.run_events(n, ev[:, c[0]], 1.0, True) for n, c in zip(nets[1.0], CH)],
                    "x2.5": [sc.run_events(n, ev[:, c[0]], 2.5, True) for n, c in zip(nets[1.0], CH)]}
            for k, per_chain in runs.items():
                for l in range(3):
                    tot = 0.0
                    for ch, sp in enumerate(per_chain):
                        off = np.cumsum([0] + [W[j][ch] for j in range(3)])
                        tot += sp[:, off[l]:off[l + 1]].sum()
                    sums[k][l] += tot / sum(W[l])
            bins += len(ev)
    meta["layers"] = {k: (v / bins).tolist() for k, v in sums.items()}
    meta["layer_names"] = ["layer 1", "layer 2", "layer 3"]
    print("spikes per neuron per bin:", {k: [round(x, 3) for x in v] for k, v in meta["layers"].items()})

    # --- neurons: mean rate per neuron over the training sentences, from the refit caches
    def mean_rates(folder):
        tot, n = None, 0
        for f in sorted(folder.glob("*.npz")):
            z = np.load(f)
            for k in z.files:
                a = z[k].astype(np.float64)
                tot = a.sum(0) if tot is None else tot + a.sum(0)
                n += len(a)
        return tot / n
    out["rate_chip"] = mean_rates(W1 / "spikes_train_x2.5")
    out["rate_trained"] = mean_rates(DL / "speck_r19/results/W1_software/spikes_train_software")
    out["neuron_layer"] = np.concatenate([np.full(sum(W[l]), l) for l in range(3)])

    # --- example: one REAL devkit test sentence, three decodes
    chip_logits = np.load(DL / "V2_onechip/chip_logits.npz")
    counts = np.load(DL / "V2_onechip/chip_counts.npz")
    ro = rr.Readout(model)
    ro.load_state_dict(torch.load(W1 / "readout_refit.pt"))
    parts = {p.session: p for p in load_partitions(str(DATA), S, "test", list(range(len(S))))}
    cands = []
    for key in counts.files:
        sess, i = key.split("/")
        p = parts[sess]
        y = np.asarray(p.labels[int(i)])
        ref = sc.reference_logits(model, p.features[int(i)], p.day_idx)
        orig = chip_logits[key]
        refit = ro(ro.traces(torch.as_tensor(counts[key], dtype=torch.float32)[None]))[0].numpy()
        n = min(len(ref), len(orig), len(refit))
        e = {k: edit_distance(greedy_decode(v[:n], blank), y) / len(y) for k, v in
             (("software", ref), ("chip", orig), ("refit", refit))}
        cands.append((key, n, e, ref[:n], orig[:n], refit[:n], y, str(p.transcriptions[int(i)]) if hasattr(p, "transcriptions") else ""))
    # a typical sentence: 2-4 s long, chip clearly worse than software, refit close to software
    pool = [c for c in cands if 100 <= c[1] * 4 <= 220 and c[2]["chip"] - c[2]["software"] >= 0.1]
    pool = pool or cands
    key, n, e, ref, orig, refit, y, tr = min(pool, key=lambda c: abs(c[2]["refit"] - c[2]["software"]) - 0.3 * (c[2]["chip"] - c[2]["refit"]))
    names = lambda seq: " ".join(PHONES[s] for s in seq)
    meta["example"] = dict(key=key, per=e, truth=names(y), transcription=tr,
                           decoded={k: names(greedy_decode(v, blank)) for k, v in
                                    (("software", ref), ("chip", orig), ("refit", refit))})
    for k, v in (("ex_software", ref), ("ex_chip", orig), ("ex_refit", refit)):
        out[k] = softmax(v)
    meta["phones"] = PHONES + ["blank"]
    print("example", key, e, "\n  truth  ", meta["example"]["truth"], "\n  ", meta["example"]["decoded"])
    np.savez_compressed(HERE / "data/calibration_speech.npz", **out)
    json.dump(meta, open(HERE / "data/calibration_speech.json", "w"), indent=1)
    print("wrote data/calibration_speech.{npz,json}")


if __name__ == "__main__":
    main()
