"""
EXPLORATORY (the audio spike-timing pilot, audio/speck/runs_fine_timing.txt): operation counts per 40 ms output for the
pilot's networks, under the paper's 45 nm rule (collect_efficiency.energy_uj), each counted on its test stream (with
audio_chip.BURN frames before it, as scored), averaged over monkeys N and G and the seeds present.

    Speck SNN  as collect_speck_ops.audio: one add per synaptic event (input events x layer 1's width, each layer's
               spikes x the next layer's), the host's EMA stage(s) on Hidden 3 at every chip step and its linear map
               per output as multiply-adds.                                       P5 (40 ms counts) and FT1 (4 ms spikes)
    LSNN       as collect_efficiency's audio LSNN rule, streaming: the first layer as one multiply-add per non-zero
               input value and neuron (dense at 40 ms; sparse 4 ms counts, so few), every spike-driven synapse
               (to the next layer, recurrent, to the readout) as an add, and per model step the neuron updates (one
               for the synaptic filter, two per adaptive neuron) and the readout's leak (two per output) as
               multiply-adds.                                                     LS0 (40 ms) and LS1 (4 ms, 10 steps)

The paper's audio LSNN reads every 17-frame window from rest (17 steps per output), so its estimate (data/efficiency.csv)
is listed for reference; LS0 is the same network reading the stream once.

    python collect_fine_ops.py      -> data/fine_ops.csv
"""
from repo_paths import WORKSPACE, AUDITORY_DATA, SNN_RUNS, ROOT, CHIP  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from collect_efficiency import energy_uj

HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHIP))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402
from snndec.evaluate import recording_file  # noqa: E402
from collect_speck_ops import synaptic_adds  # noqa: E402

RUNS = {"P5": ("audio_novel", "P5_centre_ema1_deriv"), "FT1": ("audio_fine", "FT1_spikes"), "FS3": ("audio_fine", "FS3_soft_floor3"),
        "LS0": ("audio_fine", "LS0_stream40"), "LS1": ("audio_fine", "LS1_fine")}
SEEDS = ("s1337", "s11", "s99")


def test_stream(model, info, monkey):
    from audio_chip import BURN
    pipe = AUDITORY_DATA / monkey / "aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline"
    cfg = info.get("config") or {}
    if model.spec["kind"] == "lsnn" and cfg.get("data", {}).get("input") != "counts":    # LS0: the window features
        Xs = [data.audio_windows(str(pipe), s)[0] for s in ("train", "valid", "test")]
        X = np.concatenate(Xs)
        rec, W, n = np.concatenate([X[:, 0], X[-1, 1:]]), X.shape[1], dict(zip(("train", "valid"), map(len, Xs)))
    else:
        rec, W, n = data.audio_recording(str(pipe), recording_file(info))
    first = n["train"] + n["valid"] + W - 1
    return rec[max(0, first - BURN):]


@torch.no_grad()
def count(name, monkey, seed):
    folder, pat = RUNS[name]
    f = SNN_RUNS / f"{folder}_{monkey}" / f"{pat}_{seed}" / "best.pt"
    if not f.is_file() or (folder == "audio_fine" and not (f.parent / "done").is_file()):   # finished runs only
        return None
    model, info = load(str(f))
    model.eval()
    rec = test_stream(model, info, monkey)
    model(torch.as_tensor(rec)[None])
    K = model.substeps
    if model.spec["kind"] == "speck":
        ev = chip.chip_events(model, model.input_events[0].numpy())
        a, _, _ = synaptic_adds(model, ev)
        h = model.layers[-1].width
        macs = model.out_every * int(model.spec["readout"].get("stages", 2)) * h + h * model.spec["n_outputs"]
        return a * model.out_every, macs
    # LSNN, per model step, then x K per output
    widths = [l.width for l in model.layers]
    u = model.input_events[0]                                          # (T * K, n_in): what layer 1 reads
    nz = float((u != 0).sum(-1).float().mean())                        # non-zero inputs per model step
    spk = [float(s[0].sum(-1).mean()) for s in model.layer_spikes]     # spikes per model step, per layer
    S = model.spec["n_outputs"]
    n_adapt = [float(l.adapt_mask.sum()) for l in model.layers]
    macs = nz * widths[0] + sum(widths) + 2 * sum(n_adapt) + 2 * S
    accs = sum(sp * w for sp, w in zip(spk[:-1], widths[1:]))          # to the next layer
    accs += sum(sp * w for sp, w in zip(spk, widths))                  # recurrent
    accs += sum(spk) * S                                               # to the readout (reads every layer)
    return accs * K, macs * K


def main():
    rows = []
    for name in RUNS:
        got = [c for m in ("Neil", "Gomez") for s in SEEDS if (c := count(name, m, s)) is not None]
        if not got:
            continue
        a, m = np.mean(got, 0)
        rows.append(dict(decoder=name, accs=a, macs=m, energy_uj=energy_uj(m, a), n=len(got)))
        print(f"{name}: {a:,.0f} adds, {m:,.0f} multiply-adds per output -> {energy_uj(m, a):.2f} uJ  (n={len(got)})")
    est = pd.read_csv(HERE / "data" / "efficiency.csv")
    ref = est[(est.experiment == "audio") & (est.decoder == "lsnn")].iloc[0]
    rows.append(dict(decoder="lsnn_paper (17-frame windows, data/efficiency.csv)", accs=ref.accs, macs=ref.macs,
                     energy_uj=ref.energy_uj, n=np.nan))
    print(f"paper's LSNN (windows): {ref.energy_uj:.2f} uJ")
    pd.DataFrame(rows).to_csv(HERE / "data" / "fine_ops.csv", index=False)
    print("wrote", HERE / "data" / "fine_ops.csv")


if __name__ == "__main__":
    main()
