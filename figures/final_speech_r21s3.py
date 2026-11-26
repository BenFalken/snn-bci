"""
The final speech Speck SNN, 2026-11-26: round 21 S3 (V2+C's three networks on one Speck2f, cores 5-3-0, 6-4-1 and 2-7-8,
retrained with soft reset and a membrane floor of -3; layer-1 threshold x1.5, chosen without labels; readout refitted
to the chip model's firing on training sentences and applied live on the devkit) into the results tables, scored like
every other speech decoder. final_speech_v2c.py (round 20 V2+C, hard reset: the final network until then) is kept.

    per sentence   on the 600 test sentences of perTrialPER_final_s3.json (every other decoder's): S3 in software and
                   on the REAL chip (results_s3_full devkit run, 680 sentences) read by the refitted readout
                   -> final_results/perTrialPER_final_r21s3.json (the s3 file plus SpeckR21S3 / SpeckChipR21S3)
    per session    the same, all 19 sessions -> final_results/speech_r21s3/{torch_test,chip_refit}.json ("results" ->
                   per_session, the layout collect_final.py reads); chip.json = the devkit run's power and timing
    operations     per 80 ms output, as final_speech_v2c.py -> data/speck_ops_speech_r21s3.json
    logits         the refitted readout on the real chip's spike counts, every sentence -> final_results/speech_r21s3/
                   chip_refit_logits.npz (the speech video's Speck lane, the n-gram)

(The file perTrialPER_final_s3.json predates round 21: its "s3" is an older name, unrelated to this S3.)

    python3 final_speech_r21s3.py
"""
from repo_paths import WORKSPACE, SPEECH_SNN  # noqa: E402  (the repository's paths.yaml)
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

DL = WORKSPACE
HERE = Path(__file__).parent
sys.path.insert(0, str(SPEECH_SNN / "scripts"))
sys.path.insert(0, str(HERE))
import refit_readout as rr  # noqa: E402
import speech_chip as sc  # noqa: E402
from collect_efficiency import energy_uj  # noqa: E402
from snnDecoder.dataset import load_partitions  # noqa: E402
from snnDecoder.utils import edit_distance, greedy_decode  # noqa: E402

CKPT = DL / "speck_r21/02_S3_soft_floor3/best_model_weights.pth"
DATA = DL / "speech_rebuild/derived/snnData"
CHIP = DL / "results_s3_full"
REFIT = DL / "speck_r21/S3_results/W1/readout_refit.pt"
FINAL = DL / "final_results"
OUT = FINAL / "speech_r21s3"


@torch.no_grad()
def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    model, ck = sc.load_snn_model(str(CKPT))
    S = list(ck["sessions"])
    CH = sc.chains(model)
    stacks = [sc.spiking_stack(model, 1.0, as_trained=True, chain=c) for c in CH]
    W = [[m.out_features for m in st if isinstance(m, torch.nn.Linear)] for st in stacks]
    n_read = sum(sum(w) for w in W)                                # every layer of every network read by the host
    print("networks (layer widths):", W, "| host reads", n_read, "neurons")
    blank = model.n_classes - 1
    ro = rr.Readout(model)
    ro.load_state_dict(torch.load(REFIT))
    counts = np.load(CHIP / "chip_counts_full.npz")
    parts = load_partitions(str(DATA), S, "test", list(range(len(S))))
    edits = {"sw": {}, "chip": {}}
    lens, refit_logits = {}, {}
    acc = ev_n = l1 = l2 = 0.0
    bins = 0
    for p in parts:
        for i in range(len(p)):
            key = f"{p.session}/{i}"
            y = np.asarray(p.labels[i])
            ref = sc.reference_logits(model, p.features[i], p.day_idx)
            lg = ro(ro.traces(torch.as_tensor(counts[key], dtype=torch.float32)[None]))[0].numpy()
            refit_logits[key] = lg.astype(np.float32)
            edits["sw"][key] = edit_distance(greedy_decode(ref, blank), y)
            edits["chip"][key] = edit_distance(greedy_decode(lg[:len(ref)], blank), y)
            lens[key] = len(y)
            ev = sc.host_events(model, p.features[i], p.day_idx)
            for st, c, w in zip(stacks, CH, W):
                sp = sc.run_stack(st, ev[:, c[0]], True)          # (T, w0 + w1 + w2) of this network
                acc += ev[:, c[0]].sum() * w[0] + sp[:, :w[0]].sum() * w[1] + sp[:, w[0]:w[0] + w[1]].sum() * w[2]
            bins += len(ev)
        print(p.session, flush=True)

    # per session (all 19) in collect_final's layout
    for kind, fname in (("sw", "torch_test.json"), ("chip", "chip_refit.json")):
        per = {}
        for s in S:
            ks = [k for k in lens if k.startswith(s + "/")]
            per[s] = sum(edits[kind][k] for k in ks) / sum(lens[k] for k in ks)
        tot = sum(edits[kind].values()) / sum(lens.values())
        json.dump({"source": "final_speech_r21s3.py", "results": {"1": {"per": tot, "per_session": per}}},
                  open(OUT / fname, "w"), indent=1)
        print(kind, "PER", round(tot, 4))
    shutil.copy(CHIP / "chip_full.json", OUT / "chip.json")
    np.savez_compressed(OUT / "chip_refit_logits.npz", **refit_logits)

    # per sentence, on every other decoder's 600 sentences
    t = json.load(open(FINAL / "perTrialPER_final_s3.json"))
    for name, kind, src in (("SpeckR21S3", "sw", "round 21 S3 (soft reset, floor -3) in software (torch)"),
                            ("SpeckChipR21S3", "chip", "round 21 S3 on the real chip, readout refitted on training sentences")):
        e = [float(edits[kind][k]) for k in t["keys"]]
        t[name] = {"per": sum(e) / sum(t["phonemes"]), "edits": e, "source": src}
        print(name, "600 sentences: PER", round(sum(e) / sum(t["phonemes"]), 4))
    json.dump(t, open(FINAL / "perTrialPER_final_r21s3.json", "w"))

    # operations per 80 ms output
    acc_bin = acc / bins
    mac_bin = 256 + 4 * n_read + n_read * 5 * 41 + 2 * 41
    m, a = 4 * mac_bin, 4 * acc_bin
    ops = dict(experiment="speech", accs=float(a), macs=float(m), energy_uj=float(energy_uj(m, a)), n=len(S),
               note="round 21 S3 (V2+C's three networks on one chip, soft reset, floor -3), trained network's spikes on "
                    "the 680 test sentences; per 80 ms output = 4 bins")
    json.dump(ops, open(HERE / "data/speck_ops_speech_r21s3.json", "w"), indent=1)
    print("ops per output:", {k: round(v, 1) if isinstance(v, float) else v for k, v in ops.items()})


if __name__ == "__main__":
    main()
