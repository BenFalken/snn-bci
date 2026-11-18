"""
Why the two calibrations are used: their effect, measured.

  Input end, layer 1's threshold (reference: the trained network's own output, no labels)
    a  what the factor does to the firing (speech V2, training sentences): the trained network, the chip model at x1
       (layer 1 surges, the layers after it starve) and at the chosen x2.5
    b  agreement with the trained network against the factor, for every deployed network (Jenkins M1, 4 sessions;
       audio P5, 3 seeds; speech V2); the label-free choice marked
    c  the same sweeps scored against the labels afterwards: the choice lands at or next to the most accurate factor

  Output end, the readout refit (reference: the true targets)
    d  why it can work: each neuron's mean rate, trained network vs chip model (speech V2); a consistent distortion
    e  one test sentence from the real devkit run, decoded three ways
    f  what it gains: the chip's accuracy as a share of software, before and after (speech: real chip; audio: chip
       model; Jenkins: refit on the chip model, scored on the real chip), with the software-spike control

    python calibration_data.py && python fig_calibration_results.py   -> figures/fig_calibration_results.{pdf,svg,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from calibration_panels import (STEP1, STEP2, panel_example, panel_gains, panel_layers, panel_neurons, panel_sweep)

HERE = Path(__file__).parent


def main():
    style.apply()
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    z = np.load(HERE / "data/calibration_speech.npz")
    meta = json.load(open(HERE / "data/calibration_speech.json"))
    W, H = 183.0, 132.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    A = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def band(y, text, color):
        fig.text(3 / W, 1 - y / H, text, fontsize=6.6, color="white", fontweight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc=color, ec="none"))

    band(2, "INPUT END · layer 1's threshold, chosen against the trained network's own output (no labels)", STEP1)
    a, b, c = A(18, 14, 42, 34), A(80, 14, 42, 34), A(140, 14, 40, 34)
    panel_layers(a, meta)
    panel_sweep(b, "agree")
    panel_sweep(c, "acc")
    a.set_title("What the factor does (V2)", loc="left")
    b.set_title("Chosen without labels", loc="left")
    c.set_title("Checked against labels", loc="left")

    band(66, "OUTPUT END · the host readout, refitted to the chip's firing against the true targets", STEP2)
    d = A(18, 80, 34, 40)
    g = [A(64, 82 + k * 16.0, 58, 9.5) for k in range(3)]
    f = A(131, 79, 50, 31)
    panel_neurons(d, z)
    panel_example(g, z, meta)
    panel_gains(f)
    d.set_title("Why it can work (V2)", loc="left")
    for ax, k, x, y in ((a, "a", -0.36, 1.05), (b, "b", -0.42, 1.05), (c, "c", -0.32, 1.05), (d, "d", -0.5, 1.04),
                        (g[0], "e", -0.08, 1.78), (f, "f", -0.08, 1.04)):
        ax.text(x, y, k, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    style.save(fig, HERE / "figures" / "fig_calibration_results")
    print("wrote figures/fig_calibration_results.{pdf,svg,png}")


if __name__ == "__main__":
    main()
