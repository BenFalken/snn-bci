"""
Paired comparisons behind the fidelity claims (Figure 6): for each pair, the per-unit difference (session, utterance
or sentence), its median, the number of units on which the first decoder is better, and a two-sided Wilcoxon
signed-rank test. Motor (II) has four sessions, too few for a test; it gets sign counts only.

    python figures/stats_tests.py   -> figures/data/stats_tests.json, figures/data/stats_tests.tex (table rows)

Input: data/four_fidelity.csv (one row per unit, decoder and metric). Higher is better for cc; lower for rmse and
per, so 'better' is signed per metric. Rerun after the fidelity data are refreshed (e.g. the validation-selected
motor checkpoints).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
LOWER_BETTER = {"rmse", "per"}
COMPARISONS = [  # (experiment, metric, group or None for all, decoder a, decoder b)
    ("motor1", "cc", None, "lsnn", "qrnn"), ("motor1", "cc", None, "lsnn", "lstm"),
    ("motor1", "cc", None, "speck", "lstm"), ("motor1", "cc", None, "speck", "wf"),
    ("motor1", "cc", None, "speck", "kf"),
    ("motor1", "cc", None, "lsnn", "aegru"),      # set apart: re-implemented from its paper, outside the shared code
    ("motor2", "cc", None, "lsnn", "qrnn"), ("motor2", "cc", None, "speck", "lstm"),
    ("audio", "band_r", None, "lsnn", "wf"), ("audio", "band_r", None, "lsnn", "lstm"), ("audio", "band_r", None, "lsnn", "gru"),
    ("audio", "band_r", None, "speck", "wf"), ("audio", "band_r", None, "speck", "speck_float"),
    ("audio", "band_r", None, "lsnn", "lsnn_40stream"), ("audio", "band_r", None, "speck", "speck_40"),   # 4 ms vs 40 ms
    ("speech", "per", None, "lsnn", "paper_rnn"), ("speech", "per", None, "lsnn", "lstm"),
    ("speech", "per", None, "speck", "speck_float"),
    # with the phoneme 5-gram (collect_speech_lm.py), the speech headline since 2026-11-27
    ("speech", "per", None, "lsnn_lm", "paper_rnn_lm"), ("speech", "per", None, "lsnn_lm", "lsnn"),
    ("speech", "per", None, "speck_lm", "speck"),
]
LABEL = {"aegru": "autoencoder GRU (re-impl.)", "lsnn": "LSNN", "qrnn": "QRNN", "lstm": "LSTM", "speck": "Speck (chip)", "wf": "Wiener filter",
         "kf": "Kalman filter", "gru": "GRU", "speck_float": "Speck (software)", "paper_rnn": "paper RNN",
         "lsnn_40stream": "LSNN at 40 ms", "speck_40": "Speck at 40 ms (chip)",
         "lsnn_lm": "LSNN + 5-gram", "paper_rnn_lm": "paper RNN + 5-gram", "speck_lm": "Speck (chip) + 5-gram"}
EXP = {"motor1": "Motor (I)", "motor2": "Motor (II)", "audio": "Audio", "speech": "Speech"}


def main():
    vals = defaultdict(dict)                       # (exp, metric, decoder) -> {unit: value}
    for r in csv.DictReader(open(HERE / "data" / "four_fidelity.csv")):
        vals[(r["experiment"], r["metric"], r["decoder"])][(r["group"], r["session"])] = float(r["value"])
    out, rows = [], []
    for exp, metric, group, a, b in COMPARISONS:
        va, vb = vals[(exp, metric, a)], vals[(exp, metric, b)]
        units = sorted(u for u in va if u in vb and (group is None or u[0] == group))
        d = np.array([va[u] - vb[u] for u in units])
        if metric in LOWER_BETTER:
            d = -d                                  # positive = a is better
        res = {"experiment": exp, "metric": metric, "a": a, "b": b, "n": len(d),
               "median_difference": float(np.median(d)), "a_better": int((d > 0).sum())}
        if len(d) >= 6:
            w = wilcoxon(d)
            res.update({"W": float(w.statistic), "p": float(w.pvalue)})
        out.append(res)
        p = res.get("p")
        ptxt = "--" if p is None else ("$<10^{-%d}$" % int(np.floor(-np.log10(p))) if p < 1e-3 else f"{p:.3f}")
        rows.append(f"{EXP[exp]} & {LABEL[a]} vs {LABEL[b]} & { {"cc": "r", "band_r": "r", "per": "PER"}[metric] } & "
                    f"{len(d)} & {res['median_difference']:+.3f} & {res['a_better']} of {len(d)} & {ptxt} \\\\")
        print(f"{exp:7s} {a:>6s} vs {b:<12s} n={len(d):3d}  median diff {res['median_difference']:+.4f}  "
              f"better {res['a_better']:3d}/{len(d)}  p={p if p is not None else '--'}")
    json.dump(out, open(HERE / "data" / "stats_tests.json", "w"), indent=1)
    open(HERE / "data" / "stats_tests.tex", "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
