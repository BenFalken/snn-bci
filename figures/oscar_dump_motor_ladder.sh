# Run on Oscar; prints data/motor_ladder.csv for fig_motor_ladder.py (indy, best per-axis test loss per session).
source "$(cd "$(dirname "$0")" && pwd)/../snnbci_env.sh"   # paths.yaml -> $SNNBCI_*
cd "$SNNBCI_MOTOR/main/sbatch_scripts" && source ./snn_config.sh >/dev/null 2>&1; activate_env >/dev/null 2>&1
python3 - <<'PY'
import glob, os, torch
B = os.environ["SNNBCI_OSCAR_MOTOR_DATA"] + "/snn_checkpoints/bmi/indy"
COLS = [("start", "per_session_binarized"), ("filters", "per_session_ladder_filters_binarized"),
        ("soft", "per_session_ladder_filters_soft_binarized"), ("lsnn_neurons", "per_session_lsnn_ladder_soft_binarized"),
        ("recurrence", "per_session_lsnn_ladder_rec_binarized"), ("width", "per_session_lsnn_ladder_width_binarized"),
        ("adaptation", "per_session_lsnn_ladder_adapt_binarized"), ("pretrained", "lsnn_loso_finetuned_binarized"),
        ("chip", "per_session_ladder_chip_binarized"), ("delays", "per_session_delay_binarized")]
def get(d):
    f = os.path.join(d, "best_model_weights.pth")
    if not (os.path.isfile(f) and glob.glob(os.path.join(d, "final_model_*.pth"))):
        return ""
    return f'{torch.load(f, map_location="cpu", weights_only=False)["best_loss"]:.4f}'
sessions = sorted(os.path.basename(d) for d in glob.glob(f"{B}/per_session_binarized/*"))
print("session," + ",".join(k for k, _ in COLS))
for s in sessions:
    print(s + "," + ",".join(get(f"{B}/{f}/{s}") for _, f in COLS))
PY
