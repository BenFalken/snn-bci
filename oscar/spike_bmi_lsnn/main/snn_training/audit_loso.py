"""
Audit the LOSO fine-tuned LSNN, session by session, while the array is still running.

For every session whose LSNN fine-tuning has finished (a final_model_*.pth):
  provenance  the fine-tuned checkpoint trained on that session's own dataset; it
              started from that session's own LOSO pretrained checkpoint; and the
              pretraining pool held no file of that session (the pool is symlinks,
              so each is resolved and checked against the session's directory)
  results     its test-split metrics (compare_lsnn.evaluate), next to the per-session
              Speck SNN's and, where its checkpoints still exist, the earlier SNN's
              LOSO fine-tuned model, on the same test split

Prints a table and the means over the sessions both models have; --out writes JSON.
    python snn_training/audit_loso.py --data-root $DATA_ROOT --subject indy [--out audit_indy.json]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_lsnn import evaluate  # noqa: E402

MODELS = {"lsnn": "lsnn_loso_finetuned_binarized", "speck": "per_session_binarized",
          "snn_loso": "loso_finetuned_binarized_soft"}


def provenance(base, session, ckpt_args):
    """(check name, passed) for one fine-tuned LSNN checkpoint."""
    checks = [("trained on its own session", os.path.basename(os.path.normpath(ckpt_args["data_path"])) == session)]
    init = ckpt_args.get("init_weights_from") or ""
    checks.append(("started from its own LOSO pretrain",
                   f"lsnn_loso_pretrained_binarized/{session}/" in init if init else None))   # None: resumed run
    pre = os.path.join(base, "lsnn_loso_pretrained_binarized", session, "best_model_weights.pth")
    if os.path.isfile(pre):
        pool = torch.load(pre, map_location="cpu", weights_only=False)["args"]["data_path"]
        own = os.path.realpath(ckpt_args["data_path"])
        files = [os.path.realpath(f) for f in glob.glob(os.path.join(pool, "*", "*"))]
        checks.append(("pretraining pool excludes it", bool(files) and not any(f.startswith(own + os.sep) for f in files)))
        checks.append(("pool size (files)", len(files)))
    else:
        checks.append(("pretraining pool excludes it", None))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/users/$USER/scratch/$USER/data"))
    p.add_argument("--experiment", default="bmi")
    p.add_argument("--subject", default="indy")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(2)
    base = os.path.join(args.data_root, "snn_checkpoints", args.experiment, args.subject)
    roots = {m: os.path.join(base, d) for m, d in MODELS.items()}
    sessions = sorted(s for s in os.listdir(roots["lsnn"])
                      if glob.glob(os.path.join(roots["lsnn"], s, "final_model_*.pth")))
    print(f"{len(sessions)} sessions with a finished LSNN fine-tune\n")

    report = {}
    for s in sessions:
        ck = torch.load(os.path.join(roots["lsnn"], s, "best_model_weights.pth"), map_location="cpu",
                        weights_only=False)
        rep = {"checks": provenance(base, s, ck["args"]), "best_epoch": int(ck["epoch"]) + 1}
        for m, r in roots.items():
            if os.path.isfile(os.path.join(r, s, "best_model_weights.pth")):
                res, _, _ = evaluate(os.path.join(r, s), torch.device("cpu"))
                rep[m] = {k: res[k] for k in ("rmse", "rmse_per_axis", "r2_mean", "epochs_trained")}
        report[s] = rep
        line = "  ".join(f"{m} {rep[m]['rmse']:.1f} (axis {np.mean(rep[m]['rmse_per_axis']):.1f}, R2 {rep[m]['r2_mean']:.3f})"
                         for m in MODELS if m in rep)
        bad = [c for c, ok in rep["checks"] if ok is False]
        print(f"{s}: {line}  | best epoch {rep['best_epoch']} | checks: {'FAILED ' + ', '.join(bad) if bad else 'ok'}"
              f" {[(c, ok) for c, ok in rep['checks'] if ok is None]}", flush=True)

    print("\nMeans over sessions both models have (velocity-vector RMS; per-axis RMS; R2), and LSNN wins:")
    for m in ("speck", "snn_loso"):
        both = [s for s in sessions if m in report[s] and "lsnn" in report[s]]
        if not both:
            print(f"  lsnn vs {m}: no sessions with both")
            continue
        f = lambda k, key: np.mean([report[s][k][key] if key != "rmse_per_axis" else np.mean(report[s][k][key])
                                    for s in both])
        wins = sum(report[s]["lsnn"]["rmse"] < report[s][m]["rmse"] for s in both)
        print(f"  {len(both)} sessions: lsnn {f('lsnn', 'rmse'):.1f} / {f('lsnn', 'rmse_per_axis'):.1f} / "
              f"{f('lsnn', 'r2_mean'):.3f}   {m} {f(m, 'rmse'):.1f} / {f(m, 'rmse_per_axis'):.1f} / "
              f"{f(m, 'r2_mean'):.3f}   LSNN lower error on {wins}/{len(both)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
