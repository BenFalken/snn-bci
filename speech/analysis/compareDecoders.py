#head-to-head of the paper's RNN and one or more trained SNNs (SNNDecoder), as rnn_step3 does for one SNN, but as a
#script for batch jobs: the same test trials, the same 3-gram language model, then
#   greedy phoneme error rate (PER) and word error rate with the LM (WER, with 95% bootstrap intervals), pooled and
#   per session; op-count energy estimates per 20 ms bin; measured streaming latency / energy per bin (one session)
#
#   python compareDecoders.py --baseDir /path/to/speechPaperRelease_08_20 \
#       --snn R2r_alif=/path/to/best_model_weights.pth --snn R2r_iaf=/path/to/best_model_weights.pth \
#       --gru GRU96=/path/to/derived/snns/controls/gru_1024_out80_full96.json
#
#needs the compiled lm_decoder and ~60 GB of memory for the LM. Evaluates the RNN's evaluation sessions of rnn_step3
#(its session list, indices 4-18: the 15 sessions with a competitionHoldOut partition), on --partition (test).
#Writes <out>.json (all numbers) and prints the side-by-side table.
from snnDecoder.energy.energy_meter import EnergyMeter   #first: pins the math libraries' thread counts
import argparse
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""   #CPU, as rnn_step3

def asciiToText(codes):
    codes = np.asarray(codes)
    end = np.argwhere(codes == 0)
    return ''.join([chr(c) for c in codes[:end[0, 0]]]) if len(end) else ''.join([chr(c) for c in codes])

def lmDecodeAll(decoder, out, blankPenalty):
    #decodes every trial's logits with the LM (as lmDecoderUtils.cer_with_lm_decoder, without its notebook progress bar)
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
    logits = lmDecoderUtils.rearrange_speech_logits(out['logits'], has_sil=True)
    decoded = []
    for i in range(len(logits)):
        decoder.Reset()
        decoded.append(lmDecoderUtils.lm_decode(decoder, logits[i, :int(out['logitLengths'][i])],
                                                blankPenalty=blankPenalty).strip())
    return decoded

def wordErrors(decoded, true):
    #per trial: (word edit distance, number of true words), on lowercased words without punctuation
    import re
    norm = lambda t: re.sub(r"[^a-z' ]", ' ', t.lower()).split()
    out = []
    for d, t in zip(decoded, true):
        a, b = norm(d), norm(t)
        row = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            prev, row[0] = row[0], i
            for j in range(1, len(b) + 1):
                cur = min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev, row[j] = row[j], cur
        out.append((row[len(b)], len(b)))
    return out

def perSession(editDistances, trueLengths, sessions):
    result = {}
    for s in dict.fromkeys(sessions):
        idx = [i for i, x in enumerate(sessions) if x == s]
        result[s] = float(np.sum(np.asarray(editDistances)[idx]) / np.sum(np.asarray(trueLengths)[idx]))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper RNN vs trained SNNs on the same trials and language model.')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--snn', action='append', default=[], help='NAME=checkpoint.pth (repeat for several SNNs)')
    parser.add_argument('--gru', action='append', default=[],
                        help='NAME=controls/<model>.json: a GRU control from speech/scripts/ann_controls.py, its '
                             'weights in the .pt next to it (repeat for several)')
    parser.add_argument('--rnnDir', default=None, help='default <baseDir>/derived/rnns/baselineRelease')
    parser.add_argument('--snnDataDir', default=None, help='default <baseDir>/derived/snnData')
    parser.add_argument('--lmDir', default=None, help='default <baseDir>/languageModel (3-gram); none: no language '
                        'model (greedy phoneme error only, the paper\'s metric; no compiled lm_decoder needed)')
    parser.add_argument('--partition', default='test')
    parser.add_argument('--acousticScale', type=float, default=0.8, help='as rnn_step3')
    parser.add_argument('--beam', type=float, default=18, help='as rnn_step3')
    parser.add_argument('--blankPenalty', type=float, default=np.log(2), help='as rnn_step3, for the RNN')
    parser.add_argument('--snnBlankPenalty', type=float, default=None,
                        help='default: the RNN\'s (the recurrent SNN emits the RNN\'s 80 ms frames; a 20 ms-frame SNN needs tuning)')
    parser.add_argument('--profileBins', type=int, default=1000, help='bins timed for streaming latency/energy; 0 = skip')
    parser.add_argument('--out', default=None, help='default <baseDir>/derived/decoderComparison')
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import tensorflow as tf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
    useLM = (args.lmDir or '').lower() != 'none'
    if useLM:
        import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
        from neuralDecoder.utils.lmDecoderUtils import _cer_and_wer as cerAndWer
    from snnDecoder.inference import snn_inference, load_snn_model, gru_inference
    from snnDecoder.dataset import load_partitions
    from snnDecoder.energy.op_energy_estimate import estimate_ops_rnn
    from snnDecoder.energy.profiling import profile_rnn, profile_snn
    from snnDecoder.compare import print_comparison, save_comparison, rnn_lookahead_s, snn_lookahead_s

    rnnDir = args.rnnDir or os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
    snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
    lmDir = args.lmDir or os.path.join(args.baseDir, 'languageModel')
    outPath = args.out or os.path.join(args.baseDir, 'derived', 'decoderComparison')
    snnBlankPenalty = args.blankPenalty if args.snnBlankPenalty is None else args.snnBlankPenalty
    snns = [s.split('=', 1) for s in args.snn]
    grus = [s.split('=', 1) for s in args.gru]

    #--- the RNN, as rnn_step3 (its evaluation sessions 4-18)
    rnnArgs = OmegaConf.load(os.path.join(rnnDir, 'args.yaml'))
    rnnArgs['loadDir'] = rnnDir
    rnnArgs['outputDir'] = rnnDir #the cached args.yaml keeps the authors' /oak/stanford/... outputDir
    rnnArgs['mode'] = 'infer'
    rnnArgs['loadCheckpointIdx'] = None
    rnnArgs['testDir'] = args.partition
    for x in range(len(rnnArgs['dataset']['datasetProbabilityVal'])):
        rnnArgs['dataset']['datasetProbabilityVal'][x] = 0.0
    for sessIdx in range(4, 19):
        rnnArgs['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        rnnArgs['dataset']['dataDir'][sessIdx] = args.baseDir + '/derived/tfRecords'
    evalSessions = [rnnArgs['dataset']['sessions'][i] for i in range(4, 19)]
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(rnnArgs)
    rnnOut = nsd.inference()
    trueSentences = [asciiToText(rnnOut['transcriptions'][x, :]) for x in range(rnnOut['transcriptions'].shape[0])]
    print(f'RNN: {len(trueSentences)} trials from {len(evalSessions)} sessions, greedy PER {rnnOut["cer"]:.4f}', flush=True)

    def withLM(out, penalty):
        #(LM-decoded sentences, LM phoneme error, word error), or Nones without a language model
        if not useLM:
            return None, None, None
        decoded = lmDecodeAll(lm, out, penalty)
        cer, wer = cerAndWer(decoded, trueSentences, outputType='speech_sil', returnCI=True)
        return decoded, cer, wer

    lm = None
    if useLM:
        print('Loading language model from ' + lmDir + ' (this can take a while)', flush=True)
        lm = lmDecoderUtils.build_lm_decoder(lmDir, acoustic_scale=args.acousticScale, nbest=1, beam=args.beam)

    rnnDecoded, rnnCer, rnnWer = withLM(rnnOut, args.blankPenalty)
    print(f'RNN WER {rnnWer}', flush=True)

    entries = {'RNN': {'per': float(rnnOut['cer']), 'wer': rnnWer, 'lm_per': rnnCer,
                       'lookahead_s': rnn_lookahead_s(rnnArgs)}}
    sentences = {'true': trueSentences, 'RNN': rnnDecoded}
    trialSessions = None

    #--- each SNN on the same trials
    snnModels = {}
    for name, ckpt in snns:
        out = snn_inference(ckpt, snnDataDir, evalSessions, args.partition)
        snnTrue = [asciiToText(out['transcriptions'][x, :]) for x in range(out['transcriptions'].shape[0])]
        assert snnTrue == trueSentences, f'{name} was evaluated on different trials than the RNN'
        trialSessions = trialSessions or out['sessions']
        decoded, cer, wer = withLM(out, snnBlankPenalty)
        model, checkpoint = load_snn_model(ckpt)
        snnModels[name] = (model, checkpoint)
        entries[name] = {'per': float(out['cer']), 'wer': wer, 'lm_per': cer, 'op_estimate': out['op_estimate'],
                         'lookahead_s': snn_lookahead_s(model.smooth_sd), 'checkpoint': ckpt,
                         'frames': f"{out['frameKernel']} bins, every {out['frameStride']}",
                         'per_session': perSession(out['editDistances'], out['trueSeqLengths'], out['sessions'])}
        sentences[name] = decoded
        print(f'{name}: greedy PER {out["cer"]:.4f}, WER {wer}', flush=True)

    #--- each GRU control on the same trials (the same language model and blank penalty as the SNNs)
    for name, path in grus:
        out = gru_inference(path, snnDataDir, evalSessions, args.partition)
        gTrue = [asciiToText(out['transcriptions'][x, :]) for x in range(out['transcriptions'].shape[0])]
        assert gTrue == trueSentences, f'{name} was evaluated on different trials than the RNN'
        trialSessions = trialSessions or out['sessions']
        decoded, cer, wer = withLM(out, snnBlankPenalty)
        entries[name] = {'per': float(out['cer']), 'wer': wer, 'lm_per': cer, 'checkpoint': path,
                         'lookahead_s': snn_lookahead_s(out['smooth_sd']),
                         'per_session': perSession(out['editDistances'], out['trueSeqLengths'], out['sessions'])}
        sentences[name] = decoded
        print(f'{name}: greedy PER {out["cer"]:.4f}, WER {wer}', flush=True)

    #RNN op estimate over the same trials' bin counts, and its per-session PER
    if trialSessions is not None:
        entries['RNN']['op_estimate'] = estimate_ops_rnn(rnnArgs, out['inputLengths'])
        entries['RNN']['per_session'] = perSession(rnnOut['editDistances'], rnnOut['trueSeqLengths'], trialSessions)
        #per-session word error with the LM (for paired comparisons across sessions), every decoder
        for name in (entries if useLM else []):
            we = wordErrors(sentences[name], trueSentences)
            entries[name]['wer_session'] = perSession([e for e, _ in we], [n for _, n in we], trialSessions)
        sentences['sessions'] = list(trialSessions)

    #save what is done so far: the timing below takes a while and should not be able to lose the decoding
    save_comparison(entries, outPath + '.json')
    if useLM:
        save_comparison(sentences, outPath + '_sentences.json')
    print(f'saved {outPath}.json (error rates; timing follows)', flush=True)

    #--- measured streaming latency / energy, on the last evaluation session's test sentences
    if args.profileBins > 0 and snns:
        profSession = evalSessions[-1]
        rnnLayerIdx = rnnArgs['dataset']['datasetToLayerMap'][list(rnnArgs['dataset']['sessions']).index(profSession)]
        for name, (model, checkpoint) in snnModels.items():
            part = load_partitions(snnDataDir, [profSession], 'test', [checkpoint['sessions'].index(profSession)])[0]
            try:
                entries[name]['profile'] = profile_snn(model, [(f, part.day_idx) for f in part.features],
                                                       n_bins=args.profileBins, energy_meter_cls=EnergyMeter)
            except Exception as e:
                print(f'WARNING: timing {name} failed ({e!r}); its error rates are saved', flush=True)
        try:
            entries['RNN']['profile'] = profile_rnn(nsd, [(f, rnnLayerIdx) for f in part.features],
                                                    n_bins=args.profileBins, energy_meter_cls=EnergyMeter)
        except Exception as e:
            print(f'WARNING: timing the RNN failed ({e!r}); its error rates are saved', flush=True)

    print()
    print_comparison(entries)
    if trialSessions is not None:
        print('\nGreedy PER per session:')
        names = list(entries)
        print(f"{'session':16s}" + ''.join(f'{n:>16s}' for n in names))
        for s in dict.fromkeys(trialSessions):
            print(f'{s:16s}' + ''.join(f"{entries[n]['per_session'][s]:16.3f}" for n in names))
        print('\nWER (+ LM) per session:')
        print(f"{'session':16s}" + ''.join(f'{n:>16s}' for n in names))
        for s in dict.fromkeys(trialSessions):
            print(f'{s:16s}' + ''.join(f"{entries[n]['wer_session'][s]:16.3f}" for n in names))
    save_comparison(entries, outPath + '.json')
    if useLM:
        save_comparison(sentences, outPath + '_sentences.json')
