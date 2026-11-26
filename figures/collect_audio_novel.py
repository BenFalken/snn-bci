"""
Audio, expanded cohorts (2026-11-18): every decoder per held-out word for both monkeys, in the reference paper's
metric (band_r: each mel band's correlation over the word's frames, Fisher-averaged) and our within-sound spectral
match (rho_word). Cohorts: pipeline_data/<Monkey>/aSTG/ALLWORDS_NOVEL_DAYNORM (Neil: 20 words + bloop/pop/goop, the
39 vocabulary test utterances as before; Gomez: a second monkey, the same 23 words). Speck = P5, 3 seeds averaged
per word, on the real chip with the readout refitted to the chip model's spikes (results_audio_novel, devkit run of
speck_audio_novel.zip); speck_float = the same networks in software; speck_chip_trained = chip with the trained readout.

2026-11-23, the spike-timing re-orientation (audio/speck/runs_fine_timing.txt): the paper's audio spiking decoders now
read real 4 ms threshold crossings. lsnn = LS1 (the LSNN streaming at 4 ms; 3 seeds averaged per word), speck = FT1 on
the real chip (real 4 ms spikes, no per-day normalization; readout refitted to the chip model's spikes; devkit run of
speck_deploy_fine.zip -> results_deploy_fine), speck_float / speck_chip_trained its software network / trained readout.
The 40 ms versions stay as the comparison: lsnn_40 (the pipeline's windowed LSNN, as before), lsnn_40stream (LS0, the
same streaming LSNN at 40 ms, the 4 ms run's control), speck_40 / speck_40_float (P5 on the chip / in software).
2026-11-26: speck = FS3 (FT1 with soft reset and a membrane floor of -3; collect_final.AUDIO_SPECK_*), devkit run of
speck_deploy_fine_soft.zip -> results_fine_soft.
    python collect_audio_novel.py   -> data/audio_novel_fidelity.csv
"""
from repo_paths import WORKSPACE, AUDITORY_DATA  # noqa: E402  (the repository's paths.yaml)
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from collect_final import AUDIO_NAMES, AUDIO_SPECK_CHIP, AUDIO_SPECK_TAG, SAMPLES_BEFORE
from collect_fidelity import align_raw, fisher_mean_r, sound_frames, spectral_r

HERE = Path(__file__).parent
AD = AUDITORY_DATA
CHIP = WORKSPACE / "results_audio_novel"           # P5 (40 ms counts) on the devkit
CHIP4 = AUDIO_SPECK_CHIP                             # the audio Speck SNN (FS3 since 2026-11-26; FT1 before) on the devkit
FINE = WORKSPACE / "final_results" / "audio_fine"    # the pilot's LSNN predictions (snndec.evaluate, streaming)
SEEDS = ("s11", "s1337", "s99")
NOVEL = {"bloop", "pop", "goop"}
KINDS = {"chip_refit": ("speck", "speck_40"), "float": ("speck_float", "speck_40_float"),
         "chip": ("speck_chip_trained", None)}


def rows_for(monkey):
    base_dir = AD / monkey / "aSTG" / "ALLWORDS_NOVEL_DAYNORM"
    pipe = base_dir / "pipeline"
    eps = [r for r in csv.DictReader(open(base_dir / "prep" / "epochs.csv")) if r["split"] == "test"]
    base = min(int(r["frame_start"]) for r in eps) - SAMPLES_BEFORE
    y = np.load(pipe / "window_targets_test.npy")
    preds = {d: [np.load(pipe / f"{k}_predict_test.npy")] for k, d in AUDIO_NAMES.items()
             if (pipe / f"{k}_predict_test.npy").is_file()}
    if "kf" in preds:                               # the Kalman filter's frame split, put on the window split's frames
        preds["kf"] = [align_raw(pipe, preds["kf"][0])]
    if "lsnn" in preds:                              # the pipeline's windowed LSNN at 40 ms: now the comparison
        preds["lsnn_40"] = preds.pop("lsnn")
    for kind, (new, old) in KINDS.items():
        preds[new] = [np.load(CHIP4 / f"preds_{monkey}_{AUDIO_SPECK_TAG}_{s}/{monkey}/{kind}.npy") for s in SEEDS]
        if old:
            preds[old] = [np.load(CHIP / f"preds_{monkey}_{s}/{monkey}_P5_{s}/{kind}.npy") for s in SEEDS]
    preds["lsnn"] = [np.load(FINE / f"{monkey}_LS1_fine_{s}/preds/test.npy") for s in SEEDS]
    preds["lsnn_40stream"] = [np.load(FINE / f"{monkey}_LS0_stream40_{s}/preds/test.npy") for s in SEEDS]
    for d, ps in preds.items():
        assert all(len(p) == len(y) for p in ps), (monkey, d, [len(p) for p in ps], len(y))
    out = []
    for r in eps:
        a, b = int(r["frame_start"]) - SAMPLES_BEFORE - base, int(r["frame_end"]) - SAMPLES_BEFORE - base
        if a < 0 or b > len(y):
            continue
        m = sound_frames(y[a:b].mean(1))
        uid = f"{r['word']}_{r['date']}_{r['epoch_idx']}"
        for d, ps in preds.items():
            ok = np.isfinite(ps[0][a:b, 0])
            if ok.sum() < 3:
                continue
            out.append((monkey, uid, r["word"], r["word"] in NOVEL, d,
                        np.mean([fisher_mean_r(y[a:b][ok], p[a:b][ok]) for p in ps]),
                        np.mean([spectral_r(y[a:b][m & ok], p[a:b][m & ok]) for p in ps])))
    return out


def main():
    df = pd.DataFrame(rows_for("Neil") + rows_for("Gomez"),
                      columns=["monkey", "utterance", "word", "novel", "decoder", "band_r", "rho_word"])
    df.to_csv(HERE / "data" / "audio_novel_fidelity.csv", index=False)
    pd.set_option("display.width", 200)
    for (mk, nv), g in df.groupby(["monkey", "novel"]):
        print(f"\n{mk}, {'novel' if nv else 'vocabulary'} words ({g.utterance.nunique()} utterances): median band_r")
        print(g.groupby("decoder").band_r.median().sort_values(ascending=False).round(3).to_string())
    print("\nall test utterances, median band_r")
    print(df.pivot_table(index="decoder", columns="monkey", values="band_r", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
