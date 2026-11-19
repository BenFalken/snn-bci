"""
Linear memory and breadth profiles: how much input history and how many channels a task needs, asked of a linear
decoder before any spiking network is trained. Ridge regression (penalty chosen on validation data, inputs
standardized on training data) from C channels x D delay taps, 40 ms apart, to the experiment's own target, scored with
the experiment's own metric on its test split. Three curves per experiment:
    memory   all channels, D = 1 .. (span up to 1.2 s)
    breadth  the best C channels (ranked by the weight norm of the all-channel, 5-tap fit), 5 taps
    budget   the chip-budget sweep's splits at 480 inputs: 96 x 5, 48 x 10, 32 x 15, 24 x 20, 16 x 30, 8 x 60
Same units, splits and metrics as the chip sweep (chip/memory_sweep; figures/data/results.csv), so the profile can be
checked against it:
    motor  the sweep's 8 Indy sessions (mua_8_group, 4 ms steps): each tap is the 40 ms of spike counts ending d*40 ms
           before the decoded step, as the chip's delays; target hand velocity; RMSE (mm/s) averaged over test trials
           (snndec.evaluate.evaluate_motor). Validation: the last 4 of the 20 training trials.
    audio  the sweep's 4 Neil days (40 ms frames): target the window's centre frame's spectrogram; mean correlation
           over the 128 bands (snndec.tasks.mean_correlation). Two tap placements: 'sweep' = taps back from the
           window's END, as the chip sweep (the newest input 320 ms after the target); 'centred' = taps around the
           target, the placement a designer would choose (P5's).
    speech 6 of the final LSNN's 19 sessions, spread over time; training sentences only (the alignments cover
           them: speech/scripts/export_alignments.py), split by sentence order 70 / 15 / 15 into training,
           validation and test. Target: which phoneme is attempted in each 20 ms bin (39 phonemes; rest and the word
           boundary excluded, so the score is about content, not speaking vs silence), from forced alignment to
           the LSNN's outputs. Ridge onto the one-hot code, the largest output wins; score: frame accuracy (chance:
           always the most common phoneme). Taps 40 ms apart over the 256 features (threshold crossings and spike
           power of 128 electrodes), zeros outside the sentence. Two placements: 'causal' = the bin and the past (a
           real-time decoder's constraint); 'centred' = both sides (the alignment carries the LSNN's latency).
           Budget: the speech sweep's 512 inputs: 256 x 2, 128 x 4, 64 x 8, 32 x 16, 16 x 32.
Writes <workspace>/linear_profile/<experiment>.json.

    python3 collect_linear_profile.py motor | motor-full | audio | speech | speech-extend
    (motor-full: motor's memory and breadth from the full sessions; speech-extend: speech spans to 1.2 s)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")           # light enough to share the machine with training
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import AUDITORY_DATA, MOTOR, MOTOR_DATA, WORKSPACE  # noqa: E402
from snndec import data, tasks  # noqa: E402

OUT = WORKSPACE / "linear_profile"
ALPHAS = np.logspace(-1, 5, 13)
MEMORY_D = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 31]          # taps: span (D - 1) x 40 ms, up to 1.2 s
BREADTH_C = [4, 8, 16, 24, 32, 48, 64, 96]
BUDGET = [(96, 5), (48, 10), (32, 15), (24, 20), (16, 30), (8, 60)]
MOTOR_UNITS = ["indy_20160407_02", "indy_20160426_01", "indy_20160915_01", "indy_20160930_02", "indy_20161011_03",
               "indy_20161025_04", "indy_20161212_02", "indy_20170131_02"]
AUDIO_DAYS = ["07_20_10", "07_31_13", "08_03_10", "08_06_13"]


class Ridge:
    """Ridge fits for many penalties at once (one eigendecomposition of the standardized Gram matrix)."""

    def __init__(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ymu = Y.mean(0)
        Z = (X - self.mu) / self.sd
        self.evals, self.evecs = np.linalg.eigh(Z.T @ Z)
        self.zty = self.evecs.T @ (Z.T @ (Y - self.ymu))

    def coef(self, alpha):
        return self.evecs @ (self.zty / (self.evals + alpha)[:, None])

    def predict(self, X, alpha):
        return ((X - self.mu) / self.sd) @ self.coef(alpha) + self.ymu


def fit_score(Xtr, Ytr, Xva, Yva, Xte, score):
    """Penalty chosen on validation (score: higher is better), then the test predictions and the chosen fit."""
    m = Ridge(Xtr, Ytr)
    best = max(ALPHAS, key=lambda a: score(Yva, m.predict(Xva, a)))
    return m.predict(Xte, best), m, best


# ------------------------------------------------------------------------------------------------------------ motor
def motor_taps(x, channels, D):
    """(T, C*D) per 4 ms step: tap d = spike counts in the 40 ms ending d*40 ms before the step (zeros before start)."""
    x = x[:, channels]
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)          # c[t] = sum x[:t]
    T = len(x)
    cols = []
    for d in range(D):
        end = np.arange(T) + 1 - 10 * d                                         # window (end - 10, end]
        hi, lo = np.clip(end, 0, T), np.clip(end - 10, 0, T)
        cols.append(c[hi] - c[lo])
    return np.hstack(cols).astype(np.float64)


def motor_unit(session):
    sd = str(MOTOR_DATA / "snn_datasets/bmi/indy/mua_8_group" / session)
    scalers = str(MOTOR / "snn_training/velocity_scalers.json")
    scale = data.velocity_scaler(sd, scalers)
    train, test = data.motor_split(sd, "train", scalers), data.motor_split(sd, "test", scalers)
    tr, va = train[:-4], train[-4:]
    unscale = lambda v: data.unscale_velocity(v, *scale)

    def rmse(trials, preds):                     # evaluate_motor: sqrt of each trial's mean square, averaged
        return float(np.mean([np.sqrt(((unscale(p) - unscale(t.y)) ** 2).mean()) for t, p in zip(trials, preds)]))

    def run(channels, D):
        X = lambda ts: np.vstack([motor_taps(t.x, channels, D) for t in ts])
        Y = lambda ts: np.vstack([t.y for t in ts])
        neg = lambda y, p: -np.sqrt(((unscale(p) - unscale(y)) ** 2).mean())
        P, m, a = fit_score(X(tr), Y(tr), X(va), Y(va), X(test), neg)
        n = np.cumsum([0] + [len(t.x) for t in test])
        return rmse(test, [P[n[i]:n[i + 1]] for i in range(len(test))]), m

    allc = list(range(96))
    res = {"memory": {}, "breadth": {}, "budget": {}}
    for D in MEMORY_D:
        res["memory"][D], m5 = run(allc, D)
        if D == 5:
            w = np.linalg.norm(m5.coef(ALPHAS[len(ALPHAS) // 2]).reshape(5, 96, -1), axis=(0, 2))
            rank = [int(i) for i in np.argsort(-w)]
    for C in BREADTH_C:
        res["breadth"][C] = run(rank[:C], 5)[0]
    for C, D in BUDGET:
        res["budget"][f"{C}x{D}"] = run(rank[:C], D)[0]
    return res


def motor_full(session, Ds=MEMORY_D, Cs=BREADTH_C):
    """Motor's memory and breadth curves from the session's full recording (the motor lag analysis's data: the first
    88% of the session, 20 ms bins), not the sweep's 42 s training sets, on which longer windows overfit. Split in
    time: 75% training, 12.5% validation, 12.5% test (all within the decoders' own training portion). Taps: 40 ms of
    counts (two bins) ending d*40 ms before the decoded bin. Score: R^2 of hand velocity (mean of x and y)."""
    sys.path.insert(0, str(MOTOR / "snn_training"))
    from channel_lag_tuning import load_session
    rates, v = load_session(str(WORKSPACE / "spike_bmi_lsnn_data/processed/bmi/indy/mua" / f"{session}.h5"),
                            20.0, 0.0, 0.88)
    x = rates.T                                                                   # (T, 96) counts per 20 ms
    T = len(x)
    a1, a2 = int(T * 0.75), int(T * 0.875)
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)

    def taps(channels, D):
        cols = []
        for d in range(D):
            end = np.arange(T) + 1 - 2 * d
            hi, lo = np.clip(end, 0, T), np.clip(end - 2, 0, T)
            cols.append((c[hi] - c[lo])[:, channels])
        return np.hstack(cols)

    def r2(y, p):
        return float(np.mean(1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))

    def run(channels, D):
        X = taps(channels, D)
        P, m, a = fit_score(X[:a1], v[:a1], X[a1:a2], v[a1:a2], X[a2:], r2)
        return r2(v[a2:], P), m

    res = {"memory": {}, "breadth": {}}
    allc = list(range(x.shape[1]))
    for D in Ds:
        res["memory"][D], m5 = run(allc, D)
        if D == 5:
            w = np.linalg.norm(m5.coef(ALPHAS[len(ALPHAS) // 2]).reshape(5, len(allc), -1), axis=(0, 2))
            rank = [int(i) for i in np.argsort(-w)]
    for C in Cs:
        res["breadth"][C] = run(rank[:C], 5)[0]
    return res


# ------------------------------------------------------------------------------------------------------------ audio
def audio_taps(raw, channels, offsets, k0, n):
    """(n, C*len(offsets)): for window k = k0 .. k0+n-1 (frames k .. k+16, target its centre k+8), the frames
    k + 8 + o for each offset o (frames; > 0 after the target), zeros outside the recording."""
    raw = raw[:, channels]
    k = np.arange(k0, k0 + n) + 8
    cols = []
    for o in offsets:
        i = k + o
        ok = (i >= 0) & (i < len(raw))
        col = np.zeros((n, raw.shape[1]))
        col[ok] = raw[i[ok]]
        cols.append(col)
    return np.hstack(cols)


def audio_unit(day):
    P = str(AUDITORY_DATA / "Neil/aSTG" / day / "pipeline")
    raw, W, n = data.audio_recording(P)
    raw = raw.astype(np.float64)
    Y = {s: data.audio_windows(P, s)[1].astype(np.float64) for s in ("train", "valid", "test")}
    start = {"train": 0, "valid": n["train"], "test": n["train"] + n["valid"]}
    score = lambda y, p: tasks.mean_correlation(y, p)

    def run(channels, offsets):
        X = {s: audio_taps(raw, channels, offsets, start[s], n[s]) for s in Y}
        Pt, m, a = fit_score(X["train"], Y["train"], X["valid"], Y["valid"], X["test"], score)
        return float(score(Y["test"], Pt)), m

    place = {"sweep": lambda D: [8 - d for d in range(D)],                    # back from the window's end
             "centred": lambda D: sorted({o for d in range(D) for o in ((d + 1) // 2 * (1 if d % 2 else -1),)})}
    allc = list(range(raw.shape[1]))
    res = {}
    for name, off in place.items():
        r = {"memory": {}, "breadth": {}, "budget": {}}
        for D in MEMORY_D:
            r["memory"][D], m5 = run(allc, off(D))
            if D == 5:
                w = np.linalg.norm(m5.coef(ALPHAS[len(ALPHAS) // 2]).reshape(5, len(allc), -1), axis=(0, 2))
                rank = [int(i) for i in np.argsort(-w)]
        for C in BREADTH_C:
            r["breadth"][C] = run(rank[:C], off(5))[0]
        for C, D in BUDGET:
            r["budget"][f"{C}x{D}"] = run(rank[:C], off(D))[0]
        res[name] = r
    return res


# ----------------------------------------------------------------------------------------------------------- speech
SPEECH_UNITS = ["t12.2022.04.28", "t12.2022.05.24", "t12.2022.06.14", "t12.2022.07.05", "t12.2022.07.27",
                "t12.2022.08.13"]
SPEECH_MEMORY_D = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]                 # up to 600 ms (4,096 inputs at 256 features)
SPEECH_BREADTH_C = [8, 16, 32, 64, 128, 256]
SPEECH_BUDGET = [(256, 2), (128, 4), (64, 8), (32, 16), (16, 32)]


def speech_sentences(session):
    """[(features (n, 256), label per bin)] of the session's aligned training sentences, in file order; label =
    phoneme id 0-38, 39 = word boundary, 40 = rest (before the first / after the last phoneme)."""
    A = np.load(WORKSPACE / "lag_tuning/alignments_lsnn_train.npz", allow_pickle=True)
    blank, stride = int(A["blank"]), int(A["frame_stride"])
    z = np.load(WORKSPACE / "speech_rebuild/derived/snnData" / session / "train.npz")
    F, lens = z["features"], z["featureLens"]
    starts = np.concatenate([[0], np.cumsum(lens)])
    out = []
    for i in np.where(A["sessions"] == session)[0]:
        t = int(A["trial_idx"][i]); n = int(lens[t])
        path = A["path"][A["frame_off"][i]:A["frame_off"][i + 1]]
        lab = np.full(n, blank, dtype=int)
        on = [k for k in range(len(path)) if path[k] != blank and (k == 0 or path[k] != path[k - 1])]
        for j, k in enumerate(on):
            end = on[j + 1] if j + 1 < len(on) else max(kk for kk in range(len(path)) if path[kk] != blank) + 1
            lab[k * stride:min(n, end * stride)] = path[k]
        out.append((F[starts[t]:starts[t] + n].astype(np.float64), lab))
    return out


def speech_taps(x, channels, offsets):
    """(n, C*len(offsets)): bin t's feature at t + o (bins), zeros outside the sentence."""
    x = x[:, channels]
    n = len(x)
    cols = []
    for o in offsets:
        col = np.zeros_like(x)
        lo, hi = max(0, -o), min(n, n - o)
        col[lo:hi] = x[lo + o:hi + o]
        cols.append(col)
    return np.hstack(cols)


def speech_unit(session):
    sents = speech_sentences(session)
    k1, k2 = int(len(sents) * 0.70), int(len(sents) * 0.85)
    parts = {"train": sents[:k1], "valid": sents[k1:k2], "test": sents[k2:]}
    PHON, NCLS = 39, 39
    acc = lambda y, p: float((p.argmax(1) == y.argmax(1)).mean())

    def build(sl, channels, offsets):
        X = np.vstack([speech_taps(x, channels, offsets) for x, _ in sl])
        lab = np.concatenate([l for _, l in sl])
        keep = lab < PHON                                               # phoneme bins only
        return X[keep], np.eye(NCLS)[lab[keep]]

    def run(channels, offsets):
        X = {s: build(v, channels, offsets) for s, v in parts.items()}
        P, m, a = fit_score(X["train"][0], X["train"][1], X["valid"][0], X["valid"][1], X["test"][0], acc)
        return acc(X["test"][1], P), m

    place = {"causal": lambda D: [-2 * d for d in range(D)],                   # 40 ms = 2 bins, back from the bin
             "centred": lambda D: sorted({(d + 1) // 2 * (2 if d % 2 else -2) for d in range(D)})}
    test_lab = np.concatenate([l for _, l in parts["test"]])
    test_lab = test_lab[test_lab < PHON]
    res = {"chance": float(np.bincount(test_lab).max() / len(test_lab)), "test_bins": int(len(test_lab))}
    allc = list(range(256))
    for name, off in place.items():
        r = {"memory": {}, "breadth": {}, "budget": {}}
        for D in SPEECH_MEMORY_D:
            r["memory"][D], m5 = run(allc, off(D))
            if D == 5:
                w = np.linalg.norm(m5.coef(ALPHAS[len(ALPHAS) // 2]).reshape(5, 256, -1), axis=(0, 2))
                rank = [int(i) for i in np.argsort(-w)]
        for C in SPEECH_BREADTH_C:
            r["breadth"][C] = run(rank[:C], off(5))[0]
        for C, D in SPEECH_BUDGET:
            r["budget"][f"{C}x{D}"] = run(rank[:C], off(D))[0]
        res[name] = r
    return res


class StreamRidge(Ridge):
    """Ridge from cross-products accumulated block by block (the input matrix is never held whole): the same fit as
    Ridge for inputs too large for memory."""

    def __init__(self, blocks):
        n, sx, sxx, sy, sxy = 0, None, None, None, None
        for X, Y in blocks:
            if sx is None:
                sx, sxx = np.zeros(X.shape[1]), np.zeros((X.shape[1], X.shape[1]))
                sy, sxy = np.zeros(Y.shape[1]), np.zeros((X.shape[1], Y.shape[1]))
            n += len(X); sx += X.sum(0); sxx += X.T @ X; sy += Y.sum(0); sxy += X.T @ Y
        self.mu, self.ymu = sx / n, sy / n
        self.sd = np.sqrt(np.maximum(np.diag(sxx) / n - self.mu ** 2, 0)) + 1e-6
        ztz = (sxx - n * np.outer(self.mu, self.mu)) / np.outer(self.sd, self.sd)
        zty = (sxy - n * np.outer(self.mu, self.ymu)) / self.sd[:, None]
        self.evals, self.evecs = np.linalg.eigh(ztz)
        self.zty = self.evecs.T @ zty


def speech_memory_extend(session, Ds=(20, 24, 31), place="causal"):
    """The speech memory curve at longer spans (all 256 features), fitted from streamed cross-products."""
    sents = speech_sentences(session)
    k1, k2 = int(len(sents) * 0.70), int(len(sents) * 0.85)
    parts = {"train": sents[:k1], "valid": sents[k1:k2], "test": sents[k2:]}
    PHON = 39
    out = {}
    for D in Ds:
        off = [-2 * d for d in range(D)] if place == "causal" else \
            sorted({(d + 1) // 2 * (2 if d % 2 else -2) for d in range(D)})

        def blocks(sl):
            for x, lab in sl:
                keep = lab < PHON
                if keep.any():
                    yield speech_taps(x, list(range(256)), off)[keep], np.eye(PHON)[lab[keep]]
        m = StreamRidge(blocks(parts["train"]))

        def accuracy(sl, a):
            hit = tot = 0
            for X, Y in blocks(sl):
                hit += int((m.predict(X, a).argmax(1) == Y.argmax(1)).sum()); tot += len(X)
            return hit / tot
        best = max(ALPHAS, key=lambda a: accuracy(parts["valid"], a))
        out[D] = accuracy(parts["test"], best)
        print(f"  {session} D={D}: {out[D]:.4f}", flush=True)
    return out


def main():
    if sys.argv[1] == "speech-extend":                   # add longer spans to speech.json's causal memory curve
        f = OUT / "speech.json"
        res = json.load(open(f))
        for u in res:
            for D, v in speech_memory_extend(u).items():
                res[u]["causal"]["memory"][str(D)] = v
            json.dump(res, open(f, "w"), indent=1)
        print("speech memory extended")
        return
    exp = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if exp == "motor-full":
        res = {}
        for u in MOTOR_UNITS:
            if not (WORKSPACE / "spike_bmi_lsnn_data/processed/bmi/indy/mua" / f"{u}.h5").is_file():
                print(f"motor-full {u}: no full recording here, skipped", flush=True)
                continue
            res[u] = motor_full(u)
            json.dump(res, open(OUT / "motor_full.json", "w"), indent=1)
            print(f"motor-full {u}: R2 no history {res[u]['memory'][1]:.3f}, 1.2 s {res[u]['memory'][31]:.3f}", flush=True)
        return
    units = {"motor": MOTOR_UNITS, "audio": AUDIO_DAYS, "speech": SPEECH_UNITS}[exp]
    fn = {"motor": motor_unit, "audio": audio_unit, "speech": speech_unit}[exp]
    res = {}
    for u in units:
        res[u] = fn(u)
        json.dump(res, open(OUT / f"{exp}.json", "w"), indent=1)
        print(f"{exp} {u} done", flush=True)


if __name__ == "__main__":
    main()
