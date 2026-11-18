#writes one made-up trial in the export format of plotExampleTrial.py --exportJson, so the viewer can be built and
#checked without the data; build_viewer.py marks it as an example. Real trials replace it.
#
#   python make_example.py --out example_trial.json
import argparse
import base64
import json

import numpy as np

CLASS_NAMES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
               'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
               'Z', 'ZH', 'SIL', 'blank']
BIN = 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='example_trial.json')
    a = p.parse_args()
    rng = np.random.default_rng(3)
    words = [('they', ['DH', 'EY']), ('were', ['W', 'ER']), ('okay', ['OW', 'K', 'EY'])]
    seq, wspans, t = [('SIL', 0, 420)], [], 420
    for w, phs in words:
        w0 = t
        for ph in phs:
            d = int(rng.integers(90, 170))
            seq.append((ph, t, t + d))
            t += d
        wspans.append([w, w0, t])
        seq.append(('SIL', t, t + int(rng.integers(120, 260))))
        t = seq[-1][2]
    T = (t + 600) // BIN
    seq[-1] = ('SIL', seq[-1][1], T * BIN)
    # neural activity: each phoneme drives its own random pattern, smoothed in time, plus noise
    pat = rng.standard_normal((len(CLASS_NAMES), 256))
    heat = rng.standard_normal((256, T)) * 0.6
    for ph, s, e in seq:
        if ph != 'SIL':
            heat[:, s // BIN:e // BIN] += 1.4 * pat[CLASS_NAMES.index(ph)][:, None]
    k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
    heat = np.apply_along_axis(lambda r: np.convolve(r, k / k.sum(), 'same'), 1, heat)
    heat = (heat - heat.mean(1, keepdims=True)) / (heat.std(1, keepdims=True) + 1e-8)
    heat8 = np.clip((heat + 3) / 6 * 255, 0, 255).astype(np.uint8)

    def decoder(name, kernel, stride, lag, flip, sentence, wer, inp):
        n = T // stride if name != 'RNN' else (T - kernel) // stride + 1
        ends = [(k * stride + kernel) * BIN for k in range(n)]
        starts = [e - stride * BIN for e in ends]
        top, conf = [], []
        for e in ends:
            truth = next((ph for ph, s, x in seq if s <= e - lag < x), 'SIL')
            c = CLASS_NAMES.index(truth)
            if rng.random() < flip:
                c = int(rng.integers(0, 39))
            top.append(c)
            conf.append(round(float(rng.uniform(0.35, 0.95)), 3))
        segs, s0 = [], 0
        for i in range(1, n + 1):
            if i == n or top[i] != top[s0]:
                segs.append([CLASS_NAMES[top[s0]], starts[s0], ends[i - 1]])
                s0 = i
        wtimes = []
        for j, w in enumerate(sentence.split()):
            wtimes.append([w, wspans[min(j, len(wspans) - 1)][2] + lag + 160])
        decoded = [x[0] for x in segs if x[0] != 'blank']
        truth_seq = [x[0] for x in seq]
        dp = list(range(len(truth_seq) + 1))                   # edit distance, as the real PER
        for i in range(1, len(decoded) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(truth_seq) + 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (decoded[i - 1] != truth_seq[j - 1]))
        per = round(dp[-1] / len(truth_seq), 3)
        return {'name': name, 'frameMs': ends, 'frameStartMs': starts, 'top': top, 'conf': conf, 'segments': segs,
                'per': per, 'kernelBins': kernel, 'strideBins': stride, 'input': inp,
                'lm': {'sentence': sentence, 'wer': wer, 'words': wtimes}}

    taps = [[c, int(rng.integers(0, 31)), round(float(rng.uniform(0.1, 1)), 3)] for c in range(256) for _ in range(4)]
    decs = [decoder('RNN', 32, 4, -40, 0.12, 'they were okay', 0.0, {'kind': 'window', 'ms': 640, 'lookaheadMs': 80}),
            decoder('SNN', 4, 4, 60, 0.22, 'they were okay', 0.0,
                    {'kind': 'taps', 'binMs': 20, 'neuron': 117, 'taps': taps, 'maxDelayBins': 30}),
            decoder('Wiener', 8, 4, 20, 0.62, 'the work a', 1.0, {'kind': 'window', 'ms': 160, 'lookaheadMs': 0}),
            decoder('Kalman', 4, 4, 40, 0.6, 'they work', 0.667, {'kind': 'frame', 'ms': 80, 'recurrent': True})]
    out = {'session': 'example', 'partition': 'test', 'trial': 0, 'sentence': 'they were okay', 'example': True,
           'binMs': BIN, 'nBins': int(T), 'classNames': CLASS_NAMES,
           'heat': {'rows': 256, 'cols': int(T), 'data': base64.b64encode(heat8.tobytes()).decode('ascii'),
                    'groups': [['threshold crossings', 0, 128], ['spike-band power', 128, 256]]},
           'truth': {'phonemes': [[p, s, e] for p, s, e in seq], 'words': wspans, 'alignedBy': 'RNN'},
           'decoders': decs}
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
