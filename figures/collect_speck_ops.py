"""
Operation counts for the FINAL Speck SNNs, under the same 45 nm rule as every other decoder's estimate
(collect_efficiency.energy_uj: 4.6 pJ per multiply-add + 4 memory accesses, 0.9 pJ per add + 3, 5 pJ per access):

    chip  one add (ACC) per synaptic event: every input event fans out to layer 1's neurons, every layer's spike to
          the next layer's; spike counts from the trained network on each test split (snndec model.layer_spikes)
    host  the readout's multiply-adds (MAC): motor two EMA stages on the 36 output neurons and the population decode;
          audio the EMA stage(s) on Hidden 3 (one for the sharpened network) at every chip step and the linear map to 128 bands per output

    motor   indy, every local session (speck_per_session data), 20 training trials each (the test splits live on the
            Speck laptop; spike rates, not accuracy, are what is counted)
    audio   the hard-reset words network, its 3 seeds (speck_final_batch2), test stream
    speech  round 19 V2: op count of the trained network's spikes on the test sentences (per 20 ms bin, x4 per 80 ms
            output; data/speck_ops_speech_v2.json, from final_speech_v2.py)

    python collect_speck_ops.py      -> data/speck_ops.csv (experiment, macs, accs, energy_uj per output, notes)
"""

from repo_paths import WORKSPACE, AUDITORY_DATA, SNN_RUNS, ROOT, CHIP  # noqa: E402  (the repository's paths.yaml)
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from collect_efficiency import energy_uj

HERE = Path(__file__).parent
SNNDEC = ROOT
sys.path.insert(0, str(SNNDEC))
sys.path.insert(0, str(CHIP))
from snndec import chip, data  # noqa: E402
from snndec.convert import load  # noqa: E402

MOTOR = WORKSPACE / "speck_per_session"
AUDIO = WORKSPACE / "speck_final_batch2/audio"
SPEECH_S3 = WORKSPACE / "final_results/decoderComparison_speckS3.json"


def synaptic_adds(model, events):
    """Adds per model step: input events x layer 1's width, each layer's spikes x the next layer's width."""
    widths = [l.width for l in model.layers]
    acc = events.sum(1).mean() * widths[0]
    for s, w in zip(model.layer_spikes[:-1], widths[1:]):
        acc += float(s[0].sum(-1).mean()) * w
    return acc, [float(s[0].sum(-1).mean()) for s in model.layer_spikes], float(events.sum(1).mean())


@torch.no_grad()
def motor(exp="motor", subject="bmi/indy"):
    """exp 'motor' = indy (Motor I); 'motor2' = Jenkins's M1 array (subject 'hkm/jenkins_m1', Motor II)."""
    scalers = MOTOR / "main/snn_training/velocity_scalers.json"
    ck = MOTOR / f"data/snn_checkpoints/{subject}/per_session_delay_binarized"
    rows = []
    for sess in sorted(os.listdir(ck)):
        sdir = MOTOR / f"data/snn_datasets/{subject}/mua" / sess
        if not (ck / sess / "best_model_weights.pth").is_file() or not sdir.is_dir():
            continue
        model, _ = load(str(ck / sess / "best_model_weights.pth"))
        model.eval()
        accs, steps, rates = 0.0, 0, None
        for t in data.motor_split(str(sdir), "train", str(scalers))[:20]:
            model(torch.as_tensor(t.x)[None])
            ev = chip.chip_events(model, model.input_events[0].numpy())
            a, r, e = synaptic_adds(model, ev)
            accs += a * len(ev); steps += len(ev)
            rates = np.array(r + [e]) * len(ev) + (0 if rates is None else rates)
        n_out = model.layers[-1].width
        macs = 2 * n_out + 2 * n_out                             # two EMA stages, then x and y population decode
        rows.append(dict(session=sess, accs=accs / steps, macs=macs, rates=list(rates / steps)))
        print(f"motor {sess}: {accs / steps:,.0f} adds per step; spikes per step {np.round(rates / steps, 1)}")
    a, m = np.mean([r["accs"] for r in rows]), rows[0]["macs"]
    return dict(experiment=exp, accs=a, macs=m, energy_uj=energy_uj(m, a), n=len(rows),
                note=f"{subject.split('/')[-1]}, 20 training trials per session (test trials are on the Speck laptop); per 4 ms output = per chip step")


@torch.no_grad()
def audio():
    from audio_chip import BURN
    from snndec.evaluate import recording_file
    rows = []
    # 2026-11-18: the final audio networks, P5 on the expanded cohorts of both monkeys, 3 seeds each
    AD = AUDITORY_DATA
    SD = SNN_RUNS
    for monkey, seed in [(m, s) for m in ("Neil", "Gomez") for s in ("s11", "s1337", "s99")]:
        # 2026-11-23: the audio Speck SNN on real 4 ms spikes (FT1; audio/speck/runs_fine_timing.txt)
        model, info = load(str(SD / f"audio_fine_{monkey}/FT1_spikes_{seed}/best.pt"))
        model.eval()
        pipe = AD / monkey / "aSTG/ALLWORDS_NOVEL_DAYNORM/pipeline"
        rec, W, n = data.audio_recording(str(pipe), recording_file(info))
        if info.get("channels") is not None:
            rec = rec[:, list(info["channels"])]
        first = n["train"] + n["valid"] + W - 1
        model(torch.as_tensor(rec[max(0, first - BURN):])[None])
        ev = chip.chip_events(model, model.input_events[0].numpy())
        a, r, e = synaptic_adds(model, ev)
        h = model.layers[-1].width
        macs = model.out_every * int(model.spec["readout"].get("stages", 2)) * h + h * model.spec["n_outputs"]
        rows.append((a * model.out_every, macs))
        print(f"audio {monkey} {seed}: {a * model.out_every:,.0f} adds per output; per chip step: input events {e:.1f}, "
              f"spikes {np.round(r, 1)}")
    a, m = np.mean(rows, 0)
    return dict(experiment="audio", accs=a, macs=m, energy_uj=energy_uj(m, a), n=len(rows),
                note="FT1 (real 4 ms spikes) on the expanded cohorts, monkeys N and G x 3 seeds, test stream; per 40 ms output = 10 chip steps")


def speech():
    """Round 19 V2, counted by final_speech_v2.py (its two networks fan out only within themselves)."""
    return json.load(open(HERE / "data" / "speck_ops_speech_v2c.json"))   # round 20 V2+C (final)


if __name__ == "__main__":
    torch.set_num_threads(4)
    df = pd.DataFrame([motor(), motor("motor2", "hkm/jenkins_m1"), audio(), speech()])
    df.to_csv(HERE / "data" / "speck_ops.csv", index=False)
    print(df.to_string(index=False))
