#plots sentence trials from competitionData as time-aligned panels: the neural features, the target (phonemes and
#words), and every available decoder's per-frame output (intended to be run between rnn_step2 and rnn_step3)
#
#example usage:
#   python plotExampleTrial.py --baseDir /path/to/speechPaperRelease_08_20 --session t12.2022.06.02 --trials 0 1 2
#   python plotExampleTrial.py --baseDir /path/to/speechPaperRelease_08_20 --session t12.2022.06.02 --trials 0 1 2 \
#       --lmDir /path/to/speechPaperRelease_08_20/languageModel \
#       --snnCkpt /path/to/speechPaperRelease_08_20/derived/snns/finetune_soft/best_model_weights.pth
#
#panels, all on the same time axis (s from the start of the trial, 20 ms bins):
#   neural features         area 6v tx1 + spikePow and area 44 tx1 heatmaps, 6v mean rate
#   target                  the true phonemes and words. The dataset has no phoneme timing, so each decoder's output
#                           is CTC force-aligned to the true phoneme sequence to estimate when each phoneme was
#                           attempted (one row per decoder); with no decoder they are only spread evenly (order only)
#   per decoder             phoneme probabilities per output frame, the greedy (argmax) decoding, and, with --lmDir,
#                           the language model's streaming hypothesis, each word placed at the frame from which the
#                           decoder's best path kept it until the end of the trial
#
#decoders: the RNN (--ckptDir, by default the cached derived/rnns/baselineRelease from derived.tar.gz) emits a frame
#every 80 ms, over the last 640 ms of input (and ~120 ms of lookahead from its input smoothing); the SNN (--snnCkpt)
#emits one every 20 ms bin, causally (the native SNN, --output-stride 4, and the GRU control --gruJson: one every 80
#ms, causally). A frame is drawn over the input bins it adds to the decoder's window.
#
#the language model needs the compiled lm_decoder module (LanguageModelDecoder/runtime/server/x86: python setup.py
#install) and lots of memory: ~60 GB for the 3-gram languageModel, far more for languageModel_5gram, so run it on a
#compute node. It is loaded once and reused for every trial and decoder.
import argparse
import os
import re
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BIN_SEC = 0.02 #each time bin is 20 ms
N_6V_CHANS = 128 #columns 0-127 = area 6v, 128-255 = area 44

PHONE_DEF_SIL = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]
SIL_IDX = PHONE_DEF_SIL.index('SIL') #39; decoder outputs have 41 classes, the CTC blank last (40)
CLASS_NAMES = PHONE_DEF_SIL + ['blank']
MODEL_COLORS = {'RNN': 'tab:orange', 'SNN': 'tab:green', 'GRU': 'tab:blue', 'LSTM': 'tab:purple', 'QRNN': 'tab:brown',
                'Wiener': 'tab:gray', 'Kalman': 'tab:olive'}
_EXTRA_COLORS = ['tab:red', 'tab:pink', 'tab:cyan', 'black']

def modelColor(name):
    #a fixed colour per known decoder; any other name gets the next unused one
    if name not in MODEL_COLORS:
        MODEL_COLORS[name] = _EXTRA_COLORS[sum(1 for k in MODEL_COLORS if k not in
                                               ('RNN', 'SNN', 'GRU', 'LSTM', 'QRNN', 'Wiener', 'Kalman')) % len(_EXTRA_COLORS)]
    return MODEL_COLORS[name]

def loadSession(dataPath):
    dat = scipy.io.loadmat(dataPath)
    print('Fields in ' + dataPath + ':')
    for k in dat.keys():
        if not k.startswith('__'):
            print('   ' + k + ': ' + str(dat[k].shape) + ' ' + str(dat[k].dtype))
    return dat

def getTrial(dat, trialIdx):
    nTrials = dat['sentenceText'].shape[0]
    if trialIdx >= nTrials:
        raise ValueError(f'trial {trialIdx} out of range, this file has {nTrials} trials')

    trial = {}
    trial['sentence'] = str(dat['sentenceText'][trialIdx]).strip()
    trial['block'] = int(np.squeeze(dat['blockIdx'])[trialIdx])
    trial['tx1'] = dat['tx1'][0, trialIdx].astype(np.float32)
    trial['spikePow'] = dat['spikePow'][0, trialIdx].astype(np.float32)
    trial['nTrials'] = nTrials
    return trial

def cleanSentence(sentence):
    #same text cleaning as makeTFRecordsFromSession.convertToTFRecord (the TFRecords' transcription)
    text = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    return text.replace('--', '').lower()

def sentenceToPhonemes(sentence):
    #same g2p conversion as makeTFRecordsFromSession.convertToTFRecord, as phoneme IDs
    from g2p_en import G2p
    text = cleanSentence(sentence)
    if len(text) == 0:
        return [SIL_IDX]

    phonemes = []
    for p in G2p()(text):
        if p == ' ':
            phonemes.append('SIL')
        p = re.sub(r'[0-9]', '', p)
        if re.match(r'[A-Z]+', p):
            phonemes.append(p)
    phonemes.append('SIL')
    return [PHONE_DEF_SIL.index(p) for p in phonemes]

def asciiToText(codes):
    codes = np.asarray(codes)
    return ''.join([chr(c) for c in codes[codes > 0]])

def getRNNOutputs(ckptDir, baseDir, session, partition):
    #runs the trained RNN over every trial in this session/partition (as in rnn_step3)
    from omegaconf import OmegaConf
    import tensorflow as tf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    #the cached RNN's args.yaml keeps the original authors' outputDir (/oak/stanford/...); inference only reads from it
    args['outputDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None
    args['testDir'] = partition

    sessions = list(args['dataset']['sessions'])
    if session not in sessions:
        raise ValueError(f'{session} was not one of the sessions this RNN was trained on: {sessions}')
    sessIdx = sessions.index(session)
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0
        args['dataset']['dataDir'][x] = baseDir + '/derived/tfRecords'
    args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)
    out = nsd.inference()

    model = args['model']
    stack = model.get('stack_kwargs', None)
    kernelSize = stack['kernel_size'] if stack is not None else 1
    stride = stack['strides'] if stack is not None else 1
    stride *= model.get('subsampleFactor', 1)
    return packOutputs('RNN', out, kernelSize, stride)

def getSNNOutputs(snnCkpt, snnDataDir, session, partition, name='SNN'):
    #runs the trained SNN over every trial in this session/partition (as in rnn_step3)
    from snnDecoder.inference import snn_inference
    out = snn_inference(snnCkpt, snnDataDir, [session], partition, device='cpu', count_ops=False)
    #feedforward SNN: one frame per bin; the RNN-shaped recurrent SNN: the RNN's 32-bin, stride-4 frames; the native
    #SNN (--output-stride 4) steps every bin and emits at the end of every 4th, so its frame covers those 4 bins
    stride = out.get('frameStride', 1)
    return packOutputs(name, out, max(out.get('frameKernel', 1), stride), stride)

def getGRUOutputs(gruJson, snnDataDir, session, partition, name='GRU'):
    #a GRU control from speech/scripts/ann_controls.py, on the same .npz data as the SNN
    from snnDecoder.inference import gru_inference
    out = gru_inference(gruJson, snnDataDir, [session], partition)
    packed = packOutputs(name, out, out['frameKernel'], out['frameStride'])
    packed['inputMs'] = out.get('inputMs')
    #Wiener/Kalman filters are fitted to frame-wise labels, not with CTC: their output has no blank-separated
    #spikes, so a CTC forced alignment to it is meaningless
    packed['frameWise'] = 'recipe' in out and out['recipe'] == 'closed form'
    return packed

def getSavedOutputs(npzPath, snnDataDir, session, partition, name, kernelSize, stride):
    #logits computed elsewhere (the Speck chip's, from snnDecoder/scripts/speech_chip.py --save-logits; keyed
    #<session>/<trial index in the partition>), with the true phonemes and sentence from the .npz data
    from snnDecoder.dataset import load_partitions
    z = np.load(npzPath)
    part = load_partitions(snnDataDir, [session], partition, required=True)[0]
    trials = []
    for i in range(len(part)):
        k = f'{session}/{i}'
        trials.append({'logits': z[k] if k in z else np.zeros((1, len(CLASS_NAMES)), np.float32),
                       'trueSeq': np.asarray(part.labels[i], np.int32),
                       'transcription': asciiToText(part.transcriptions[i])})
    return {'name': name, 'trials': trials, 'kernelSize': kernelSize, 'stride': stride}

def packOutputs(name, out, kernelSize, stride):
    #TFRecords and the .npz export are written in .mat trial order and are not shuffled at inference, so trial
    #indices line up with competitionData
    trials = []
    for i in range(len(out['logits'])):
        trueSeq = out['trueSeqs'][i]
        trials.append({'logits': out['logits'][i, :int(out['logitLengths'][i]), :],
                       'trueSeq': trueSeq[trueSeq >= 0].astype(np.int32),
                       'transcription': asciiToText(out['transcriptions'][i])})
    return {'name': name, 'trials': trials, 'kernelSize': kernelSize, 'stride': stride}

def frameEdges(model, nFrames):
    #output frame k sees input bins [k*stride, k*stride+kernelSize); it is drawn over the last stride bins of that
    #window, i.e. from when its newest input arrives
    k = np.arange(nFrames)
    t1 = (k*model['stride'] + model['kernelSize']) * BIN_SEC
    return t1 - model['stride']*BIN_SEC, t1

def logSoftmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=1, keepdims=True))

def editDistance(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j-1] + 1, prev + (a[i-1] != b[j-1]))
    return int(d[len(b)])

def forceAlign(logProbs, trueSeq):
    #returns a list of (phonemeIdx, startFrame, endFrame) in decoder output frames, or None if alignment fails;
    #each phoneme runs from the frame it is first emitted to the next phoneme's (so it includes its repeats and the
    #blanks after it). Standard CTC Viterbi over blank-interleaved labels: a phoneme may hold for several frames
    #(the torchaudio-tutorial trellis used before only let blanks fill the time, which squeezed phonemes whose
    #decoder output repeats for many frames, e.g. long SIL runs, into single frames)
    T, L = logProbs.shape[0], len(trueSeq)
    blank = logProbs.shape[1] - 1
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = trueSeq
    S, neg = len(ext), -1e30
    dp = np.full((T, S), neg)
    back = np.zeros((T, S), dtype=np.int8)                       # 0 stay, 1 from s-1, 2 from s-2 (skip a blank)
    dp[0, :2] = logProbs[0, ext[:2]]
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        prev = dp[t - 1]
        c = np.stack([prev, np.concatenate([[neg], prev[:-1]]),
                      np.where(skip, np.concatenate([[neg, neg], prev[:-2]]), neg)])
        back[t] = c.argmax(0)
        dp[t] = c.max(0) + logProbs[t, ext]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if dp[-1, s] <= neg / 2:
        return None                                              # too few frames for the sequence
    state = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        state[t] = s
        s -= int(back[t, s])
    starts = [int(np.flatnonzero(state == 2 * i + 1)[0]) for i in range(L)]
    ends = starts[1:] + [T]
    return [(int(trueSeq[i]), starts[i], ends[i]) for i in range(L)]

def greedySegments(logits):
    #runs of the same argmax class, without blank: (phonemeIdx, startFrame, endFrame)
    best = np.argmax(logits, axis=1)
    segments, start = [], 0
    for k in range(1, len(best) + 1):
        if k == len(best) or best[k] != best[start]:
            if best[start] != len(CLASS_NAMES) - 1:
                segments.append((int(best[start]), start, k))
            start = k
    return segments

def greedyDecode(logits):
    #CTC collapse of the argmax path: merge repeats, drop blanks
    best = np.argmax(logits, axis=1)
    return [int(c) for i, c in enumerate(best) if c != len(CLASS_NAMES) - 1 and (i == 0 or c != best[i-1])]

def wordSpans(segments, words):
    #groups phoneme segments into words at the SIL after each word: (word, startFrame, endFrame)
    spans, cur = [], []
    for seg in segments:
        if seg[0] == SIL_IDX:
            if cur:
                spans.append((cur[0][1], cur[-1][2]))
            cur = []
        else:
            cur.append(seg)
    if cur:
        spans.append((cur[0][1], cur[-1][2]))
    if len(spans) != len(words):
        words = ['?'] * len(spans) #g2p split the sentence differently from its spaces
    return [(w, s, e) for w, (s, e) in zip(words, spans)]

def buildLM(lmDir, acousticScale, beam):
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
    print('Loading language model from ' + lmDir + ' (this can take a while)')
    return lmDecoderUtils.build_lm_decoder(lmDir, acoustic_scale=acousticScale, nbest=1, beam=beam)

def lmStream(decoder, logits, blankPenalty, rescore):
    #feeds the decoder one output frame at a time, as it would run online; after every frame it holds the best
    #partial hypothesis. Returns (words of the partial hypothesis after each frame, final sentence)
    import lm_decoder
    from neuralDecoder.utils.lmDecoderUtils import rearrange_speech_logits

    logits = np.ascontiguousarray(rearrange_speech_logits(logits[None], has_sil=True)[0], dtype=np.float32)
    logPriors = np.zeros([1, logits.shape[1]], dtype=np.float32)
    decoder.Reset()
    partial = []
    for k in range(logits.shape[0]):
        lm_decoder.DecodeNumpy(decoder, logits[k:k+1], logPriors, blankPenalty)
        res = decoder.result()
        partial.append(res[0].sentence.split() if len(res) > 0 else [])
    decoder.FinishDecoding()
    if rescore:
        decoder.Rescore()
    res = decoder.result()
    final = res[0].sentence.strip() if len(res) > 0 else ''
    decoder.Reset()
    return partial, final

def stableWordFrames(partial, finalWords):
    #frame from which each final word stayed in the partial hypothesis (with every word before it) until the end of
    #the trial; None if it only appeared when decoding was finalized
    lcp = []
    for hyp in partial:
        n = 0
        while n < min(len(hyp), len(finalWords)) and hyp[n] == finalWords[n]:
            n += 1
        lcp.append(n)
    suffixMin = np.minimum.accumulate(np.array(lcp[::-1]))[::-1] if lcp else np.array([])
    frames = []
    for i in range(len(finalWords)):
        ok = np.nonzero(suffixMin > i)[0]
        frames.append(int(ok[0]) if len(ok) > 0 else None)
    return frames

def countRevisions(partial):
    #frames where the hypothesis changed other than by appending words
    return sum(1 for a, b in zip(partial[:-1], partial[1:]) if b[:len(a)] != a)

def zscoreChans(x, smoothSD):
    x = gaussian_filter1d(x, smoothSD, axis=0) if smoothSD > 0 else x
    return (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-8)

def drawSegments(ax, segments, y0, y1, color, labelFn, fontsize=6):
    #segments: (label, t0, t1) in seconds
    for label, t0, t1 in segments:
        isSil = label == 'SIL'
        ax.axvspan(t0, t1, ymin=y0, ymax=y1, color='0.85' if isSil else color, alpha=0.6, linewidth=0)
        if not isSil:
            ax.text((t0+t1)/2, (y0+y1)/2, labelFn(label), ha='center', va='center', fontsize=fontsize,
                    transform=ax.get_xaxis_transform(), clip_on=True)

def snnInputTaps(snnCkpt):
    #the SNN's real input delays for one first-layer neuron (the one with the largest total input weight):
    #[(channel, delay in bins, |weight| scaled to 0..1)], 4 synapses per channel; None without delayed inputs
    import torch
    from snnDecoder.inference import load_snn_model
    model, _ = load_snn_model(snnCkpt)
    if not getattr(model, 'syn_delay_in', 0):
        return None
    w0 = model.layers[0].ff.weight.detach().abs()                          # (neurons, channels)
    groups = [(w0, model.in_delay.delays() if hasattr(model, 'in_delay') else model.in_syn_masks.float().argmax(0))]
    if getattr(model, 'in_mult', 1) > 1:
        extra_w = model.in_extra.detach().abs()                             # (M - 1, neurons, channels)
        extra_d = model.in_extra_delay.delays() if hasattr(model, 'in_extra_delay') else \
            model.in_extra_masks.float().argmax(1)
        groups += [(extra_w[k], extra_d[k]) for k in range(extra_w.shape[0])]
    j = int(sum(w.sum(1) for w, _ in groups).argmax())
    wmax = float(max(w[j].max() for w, _ in groups))
    taps = []
    for w, d in groups:
        for c in range(w.shape[1]):
            taps.append((c, int(d[j, c]), round(float(w[j, c]) / wmax, 3)))
    return {'neuron': j, 'taps': taps, 'maxDelayBins': int(model.syn_delay_in)}

def exportTrial(path, session, partition, trialIdx, trial, transcription, trueSeq, models, smoothSD, snnTaps):
    #everything the interactive trial viewer (trialViewer/) needs, as one JSON file; times in ms from trial start
    import base64, json
    heat = np.concatenate([zscoreChans(trial['tx1'][:, :N_6V_CHANS], smoothSD),
                           zscoreChans(trial['spikePow'][:, :N_6V_CHANS], smoothSD)], axis=1).T   # (256, T)
    heat8 = np.clip((heat + 3) / 6 * 255, 0, 255).astype(np.uint8)
    ms = lambda sec: round(float(sec) * 1000, 1)
    truthModel = next((m for m in models if m['name'] == 'RNN' and m['alignment']), None) or \
        next((m for m in models if m['alignment']), None)
    truth = {'phonemes': [], 'words': [], 'alignedBy': truthModel['name'] if truthModel else None}
    if truthModel is not None:
        nf = truthModel['logits'].shape[0]
        t0, t1 = frameEdges(truthModel, nf)
        end = lambda e: t1[min(max(e, 1), nf) - 1]                          # segment ends are exclusive frame indices
        segs = [(p, s, e) for p, s, e in truthModel['alignment']]
        truth['phonemes'] = [[CLASS_NAMES[p], ms(t0[min(s, nf - 1)]), ms(end(e))] for p, s, e in segs]
        truth['words'] = [[w, ms(t0[min(s, nf - 1)]), ms(end(e))] for w, s, e in wordSpans(segs, transcription.split())]
    decs = []
    for m in models:
        n = m['logits'].shape[0]
        t0, t1 = frameEdges(m, n)
        lp = logSoftmax(m['logits'])
        top = np.argmax(lp, axis=1)
        conf = np.exp(lp[np.arange(n), top])
        decoded = greedyDecode(m['logits'])
        d = {'name': m['name'], 'frameMs': [ms(a) for a in t1], 'frameStartMs': [ms(a) for a in t0],
             'top': [int(c) for c in top], 'conf': [round(float(c), 3) for c in conf],
             'segments': [[CLASS_NAMES[p], ms(t0[s]), ms(t1[e - 1])] for p, s, e in greedySegments(m['logits'])],
             'per': editDistance(decoded, list(trueSeq)) / max(1, len(trueSeq)) if len(trueSeq) else None,
             'kernelBins': int(m['kernelSize']), 'strideBins': int(m['stride'])}
        if m['lm'] is not None:
            partial, final = m['lm']
            words = final.split()
            frames = stableWordFrames(partial, words)
            d['lm'] = {'sentence': final, 'wer': editDistance(words, transcription.split()) / max(1, len(transcription.split())),
                       'words': [[w, ms(t1[f]) if f is not None else ms(t1[-1])] for w, f in zip(words, frames)]}
        if m['name'] == 'RNN':
            d['input'] = {'kind': 'window', 'ms': m['kernelSize'] * BIN_SEC * 1000, 'lookaheadMs': 80}
        elif m['name'] == 'SNN' and snnTaps is not None:
            d['input'] = {'kind': 'taps', 'binMs': BIN_SEC * 1000, **snnTaps}
        elif m['name'].lower().startswith('kalman'):
            d['input'] = {'kind': 'frame', 'ms': m['kernelSize'] * BIN_SEC * 1000, 'recurrent': True}
        elif m.get('inputMs') and m['inputMs'] > m['stride'] * BIN_SEC * 1000:
            d['input'] = {'kind': 'window', 'ms': m['inputMs'], 'lookaheadMs': 0}
        elif m['kernelSize'] > m['stride']:
            d['input'] = {'kind': 'window', 'ms': m['kernelSize'] * BIN_SEC * 1000, 'lookaheadMs': 0}
        else:
            d['input'] = {'kind': 'frame', 'ms': m['stride'] * BIN_SEC * 1000, 'recurrent': True}
        decs.append(d)
    out = {'session': session, 'partition': partition, 'trial': trialIdx, 'sentence': transcription,
           'binMs': BIN_SEC * 1000, 'nBins': int(heat.shape[1]), 'classNames': CLASS_NAMES,
           'heat': {'rows': int(heat8.shape[0]), 'cols': int(heat8.shape[1]),
                    'data': base64.b64encode(heat8.tobytes()).decode('ascii'),
                    'groups': [['threshold crossings', 0, N_6V_CHANS], ['spike-band power', N_6V_CHANS, 2 * N_6V_CHANS]]},
           'truth': truth, 'decoders': decs}
    with open(path, 'w') as f:
        json.dump(out, f)
    print('exported ' + path)

def plotTrial(trial, target, models, title, outFile, smoothSD):
    nBins = trial['tx1'].shape[0]
    tEnd = nBins * BIN_SEC
    extent = [0, tEnd, N_6V_CHANS, 0]

    panels = [('tx1_6v', 3), ('spikePow_6v', 3), ('tx1_44', 2), ('popRate', 1.2), ('target', 0.6 + 0.6*max(1, len(models)))]
    for m in models:
        panels += [(m['name'] + '_probs', 3), (m['name'] + '_greedy', 0.5)]
        if m['lm'] is not None:
            panels += [(m['name'] + '_lm', 0.9)]
    heights = [h for _, h in panels]

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 1.1*sum(heights) + 1.5), sharex=True,
                             gridspec_kw={'height_ratios': heights})
    ax = dict(zip([p for p, _ in panels], axes))

    tx6v = trial['tx1'][:, :N_6V_CHANS]
    ax['tx1_6v'].imshow(zscoreChans(tx6v, smoothSD).T, aspect='auto', extent=extent, cmap='RdBu_r', vmin=-3, vmax=3, interpolation='none')
    ax['tx1_6v'].set_ylabel('Area 6v\nthreshold crossings\n(channel, z)')

    ax['spikePow_6v'].imshow(zscoreChans(trial['spikePow'][:, :N_6V_CHANS], smoothSD).T, aspect='auto', extent=extent, cmap='RdBu_r', vmin=-3, vmax=3, interpolation='none')
    ax['spikePow_6v'].set_ylabel('Area 6v\nspike band power\n(channel, z)')

    tx44 = trial['tx1'][:, N_6V_CHANS:]
    if tx44.shape[1] > 0:
        ax['tx1_44'].imshow(zscoreChans(tx44, smoothSD).T, aspect='auto', extent=[0, tEnd, tx44.shape[1], 0], cmap='RdBu_r', vmin=-3, vmax=3, interpolation='none')
    ax['tx1_44'].set_ylabel('Area 44\nthreshold crossings\n(not used by decoders)')

    timeAxis = (np.arange(nBins) + 0.5) * BIN_SEC
    popRate = gaussian_filter1d(np.mean(tx6v, axis=1), max(smoothSD, 1)) / BIN_SEC
    ax['popRate'].plot(timeAxis, popRate, 'k', linewidth=1)
    ax['popRate'].set_ylabel('6v mean\nrate (Hz)')

    #target: words on top, then one phoneme row per aligner
    axT = ax['target']
    axT.set_yticks([])
    words = target['transcription'].split()
    phoneName = lambda p: PHONE_DEF_SIL[p] if isinstance(p, (int, np.integer)) else p
    aligners = [m for m in models if m['alignment'] is not None]
    rowLabels = []
    alignModels = [m for m in models if not m.get('frameWise')] or models   # aligned rows: CTC decoders only
    nRows = 1 + max(1, len(alignModels))
    rowH = 1.0 / nRows
    if len(alignModels) == 0:
        #no decoder: phonemes in order, spread evenly over the trial (not their timing)
        edges = np.linspace(0, tEnd, len(target['phonemes']) + 1)
        segs = [(p, edges[i], edges[i+1]) for i, p in enumerate(target['phonemes'])]
        drawSegments(axT, [(phoneName(p), t0, t1) for p, t0, t1 in segs], 0, rowH, 'tab:gray', str)
        rowLabels.append('phonemes, in order\n(untimed: evenly spread)')
        wordSegs = wordSpans([(p, i, i+1) for i, (p, _, _) in enumerate(segs)], words)
        drawSegments(axT, [(w, edges[s], edges[e]) for w, s, e in wordSegs], 1 - rowH, 1, 'tab:gray', str, fontsize=7)
        rowLabels.append('words (untimed)')
    else:
        for r, m in enumerate(alignModels):
            y0 = r * rowH
            if m['alignment'] is None:
                axT.text(0.5, y0 + rowH/2, m['name'] + ': CTC forced alignment failed for this trial',
                         transform=axT.transAxes, ha='center', va='center', fontsize=7)
            else:
                f0, f1 = frameEdges(m, m['logits'].shape[0])
                drawSegments(axT, [(phoneName(p), f0[s], f1[e-1]) for p, s, e in m['alignment']], y0, y0 + rowH,
                             modelColor(m['name']), str)
            rowLabels.append('phonemes, aligned\nby ' + m['name'])
        #words from the first decoder that aligned
        if aligners:
            m = aligners[0]
            f0, f1 = frameEdges(m, m['logits'].shape[0])
            wordSegs = wordSpans(m['alignment'], words)
            drawSegments(axT, [(w, f0[s], f1[e-1]) for w, s, e in wordSegs], 1 - rowH, 1, 'tab:blue', str, fontsize=7)
            for _, s, _ in wordSegs:
                for a in axes:
                    a.axvline(f0[s], color='k', linewidth=0.4, alpha=0.4)
        rowLabels.append('words' + (' (' + aligners[0]['name'] + ')' if aligners else ''))
    axT.set_ylim([0, 1])
    axT.set_yticks((np.arange(nRows) + 0.5) * rowH)
    axT.set_yticklabels(rowLabels, fontsize=6)
    axT.set_ylabel('target', rotation=0, ha='right', va='center')

    for m in models:
        nFrames = m['logits'].shape[0]
        f0, f1 = frameEdges(m, nFrames)
        probs = np.exp(logSoftmax(m['logits']))
        axP = ax[m['name'] + '_probs']
        #frames are contiguous and equally spaced, so one image spans them all
        axP.imshow(probs.T, aspect='auto', extent=[f0[0], f1[-1], len(CLASS_NAMES), 0], cmap='magma', vmin=0, vmax=1, interpolation='none')
        axP.set_yticks(np.arange(len(CLASS_NAMES)) + 0.5)
        axP.set_yticklabels(CLASS_NAMES, fontsize=4)
        axP.set_ylabel(f"{m['name']} phoneme\nprobability\n({m['stride']*BIN_SEC*1000:.0f} ms frames)")

        axG = ax[m['name'] + '_greedy']
        axG.set_yticks([])
        drawSegments(axG, [(phoneName(p), f0[s], f1[e-1]) for p, s, e in greedySegments(m['logits'])], 0, 1,
                     modelColor(m['name']), str)
        axG.set_ylabel(m['name'] + ' greedy\n(argmax)', rotation=0, ha='right', va='center', fontsize=8)

        if m['lm'] is not None:
            partial, final = m['lm']
            finalWords = final.split()
            axL = ax[m['name'] + '_lm']
            axL.set_yticks([])
            axL.set_ylim([0, 1])
            for i, (w, k) in enumerate(zip(finalWords, stableWordFrames(partial, finalWords))):
                t = f1[k] if k is not None else f1[-1]
                y = 0.25 + 0.5 * (i % 2)
                axL.axvline(t, color=modelColor(m['name']), linewidth=0.8)
                axL.text(t, y, ' ' + w + ('*' if k is None else ''), ha='left', va='center', fontsize=7)
            axL.set_ylabel(m['name'] + ' + LM\n(word kept\nfrom here)', rotation=0, ha='right', va='center', fontsize=8)
            axL.text(1.0, 1.02, f"{countRevisions(partial)} revisions while streaming; * = only after finalizing",
                     transform=axL.transAxes, ha='right', va='bottom', fontsize=6, color='0.4')

    axes[-1].set_xlim([0, tEnd])
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=9, ha='left', x=0.01)
    titleInches = 0.14 * (title.count('\n') + 1)
    fig.tight_layout(rect=[0, 0, 1, 1 - titleInches / fig.get_figheight()])
    fig.savefig(outFile, dpi=200)
    plt.close(fig)
    print('Saved ' + outFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot competitionData sentence trials as time-aligned panels.')
    parser.add_argument('--baseDir', required=True, help='folder containing competitionData/ and derived/')
    parser.add_argument('--session', default='t12.2022.06.02')
    parser.add_argument('--partition', default='test', choices=['train', 'test', 'competitionHoldOut'])
    parser.add_argument('--trials', type=int, nargs='+', default=[0], help='trial indices within the session/partition file')
    parser.add_argument('--ckptDir', default='auto', help="trained RNN folder; 'auto' = <baseDir>/derived/rnns/baselineRelease if it exists, 'none' = no RNN")
    parser.add_argument('--snnCkpt', default=None, help='trained SNN checkpoint (.pth) from rnn_step2b (optional)')
    parser.add_argument('--snnName', default='SNN', help='label of the SNN in the figure')
    parser.add_argument('--gruJson', default=None, help='a GRU control (speech/scripts/ann_controls.py): its .json, '
                        'weights in the .pt next to it (optional)')
    parser.add_argument('--gruName', default='GRU', help='label of the GRU in the figure')
    parser.add_argument('--exportJson', default=None, help='folder for one JSON per trial for the interactive '
                        'trial viewer (trialViewer/build_viewer.py)')
    parser.add_argument('--control', action='append', default=[],
                        help='NAME=path.json: any control from ann_controls.py (GRU, LSTM, QRNN, Wiener or Kalman filter), '
                             'its weights in the .pt next to it; repeat for several, drawn in this order')
    parser.add_argument('--savedLogits', action='append', default=[],
                        help='NAME=logits.npz: a decoder from saved logits (keyed <session>/<trial>), e.g. the Speck '
                             'chip\'s from speech_chip.py --save-logits; its frames are --savedStride bins')
    parser.add_argument('--savedStride', type=int, default=4, help='bins per frame of --savedLogits (the SNN\'s 80 ms)')
    parser.add_argument('--snnDataDir', default=None, help='SNN .npz data (default <baseDir>/derived/snnData)')
    parser.add_argument('--lmDir', default=None, help='language model folder, e.g. <baseDir>/languageModel (optional; needs ~60 GB RAM)')
    parser.add_argument('--lmRescore', action='store_true', help='rescore with G_no_prune.fst, if the LM folder has one')
    parser.add_argument('--acousticScale', type=float, default=0.8, help='as in rnn_step3')
    parser.add_argument('--beam', type=float, default=18, help='as in rnn_step3')
    parser.add_argument('--blankPenalty', type=float, default=np.log(2), help='as in rnn_step3 (RNN)')
    parser.add_argument('--snnBlankPenalty', type=float, default=np.log(2), help='as in rnn_step3 (SNN; needs tuning)')
    parser.add_argument('--smoothSD', type=float, default=2.0, help='gaussian smoothing of neural heatmaps, in 20 ms bins')
    parser.add_argument('--outDir', default='.', help='output folder for trial_<session>_<partition>_<trial>.png')
    args = parser.parse_args()

    dat = loadSession(os.path.join(args.baseDir, 'competitionData', args.partition, args.session + '.mat'))

    ckptDir = args.ckptDir
    if ckptDir == 'auto':
        ckptDir = os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
        if not os.path.isfile(os.path.join(ckptDir, 'args.yaml')):
            print('No cached RNN at ' + ckptDir + ' (extract derived.tar.gz, or pass --ckptDir); plotting without it')
            ckptDir = None
    elif ckptDir == 'none':
        ckptDir = None

    #each decoder runs once over the whole session/partition
    decoders = []
    if ckptDir is not None:
        decoders.append((getRNNOutputs(ckptDir, args.baseDir, args.session, args.partition), args.blankPenalty))
    if args.snnCkpt is not None:
        snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
        decoders.append((getSNNOutputs(args.snnCkpt, snnDataDir, args.session, args.partition, args.snnName),
                         args.snnBlankPenalty))
    if args.gruJson is not None:
        snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
        decoders.append((getGRUOutputs(args.gruJson, snnDataDir, args.session, args.partition, args.gruName),
                         args.snnBlankPenalty))
    for spec in args.control:
        name, path = spec.split('=', 1)
        snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
        decoders.append((getGRUOutputs(path, snnDataDir, args.session, args.partition, name), args.snnBlankPenalty))
    for spec in args.savedLogits:
        name, path = spec.split('=', 1)
        snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
        decoders.append((getSavedOutputs(path, snnDataDir, args.session, args.partition, name, args.savedStride,
                                         args.savedStride), args.snnBlankPenalty))
    lm = buildLM(args.lmDir, args.acousticScale, args.beam) if args.lmDir is not None else None
    snnTaps = snnInputTaps(args.snnCkpt) if (args.exportJson and args.snnCkpt) else None

    os.makedirs(args.outDir, exist_ok=True)
    for trialIdx in args.trials:
        trial = getTrial(dat, trialIdx)
        transcription = cleanSentence(trial['sentence'])
        title = f"{args.session}  {args.partition}  trial {trialIdx}/{trial['nTrials']}  block {trial['block']}  ({trial['tx1'].shape[0]} bins = {trial['tx1'].shape[0]*BIN_SEC:.2f} s)\n"
        title += 'Sentence: "' + trial['sentence'] + '"\n'

        models = []
        for dec, blankPenalty in decoders:
            out = dec['trials'][trialIdx]
            if out['transcription'] != transcription:
                print(f"WARNING: {dec['name']} trial {trialIdx} is \"{out['transcription']}\", the .mat file has \"{transcription}\"")
            m = {'name': dec['name'], 'kernelSize': dec['kernelSize'], 'stride': dec['stride'], 'logits': out['logits']}
            m['frameWise'] = dec.get('frameWise', False)
            m['inputMs'] = dec.get('inputMs')
            m['alignment'] = forceAlign(logSoftmax(out['logits']), out['trueSeq']) \
                if len(out['trueSeq']) > 0 and not m['frameWise'] else None
            m['lm'] = lmStream(lm, out['logits'], blankPenalty, args.lmRescore) if lm is not None else None
            models.append(m)

        if decoders:
            trueSeq = list(decoders[0][0]['trials'][trialIdx]['trueSeq'])
        else:
            try:
                trueSeq = sentenceToPhonemes(trial['sentence'])
            except ImportError:
                trueSeq = []
        title += 'Phonemes: ' + ' '.join([PHONE_DEF_SIL[p] for p in trueSeq]) + '\n'
        for m in models:
            decoded = greedyDecode(m['logits'])
            line = f"{m['name']} greedy: " + ' '.join([PHONE_DEF_SIL[p] for p in decoded])
            if len(trueSeq) > 0:
                line += f"   (PER {editDistance(decoded, trueSeq)/len(trueSeq):.0%})"
            if m['lm'] is not None:
                line += f"\n{m['name']} + LM: \"{m['lm'][1]}\"   (WER {editDistance(m['lm'][1].split(), transcription.split())/max(1, len(transcription.split())):.0%})"
            title += line + '\n'

        target = {'phonemes': trueSeq, 'transcription': transcription}
        outFile = os.path.join(args.outDir, f'trial_{args.session}_{args.partition}_{trialIdx}.png')
        plotTrial(trial, target, models, title.strip(), outFile, args.smoothSD)
        if args.exportJson:
            os.makedirs(args.exportJson, exist_ok=True)
            exportTrial(os.path.join(args.exportJson, f'trial_{args.session}_{args.partition}_{trialIdx}.json'),
                        args.session, args.partition, trialIdx, trial, transcription, trueSeq, models, args.smoothSD,
                        snnTaps)
