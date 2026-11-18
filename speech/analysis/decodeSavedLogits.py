#word error with the 3-gram language model for logits saved elsewhere (the Speck chip's, from
#snnDecoder/scripts/speech_chip.py --save-logits), on the same evaluation sessions and test trials as compareDecoders.py
#(the RNN's sessions 4-18), so the numbers sit beside its table. One LM load per --setting.
#
#   python decodeSavedLogits.py --baseDir /path/to/speechBCI_dryad --logits chip_logits.npz --name SpeckChip \
#       --setting default=0.8,0.693 --setting tuned=1.6,3.47 --out /path/name
#
#--setting NAME=acousticScale,blankPenalty. Writes <out>.json (PER, WER with 95% intervals, per session) and
#<out>_sentences.json (every decoded sentence, keyed <session>/<trial>).
import argparse
import json
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LM-decode saved logits (<session>/<trial> -> [T, classes]).')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--logits', required=True, help='.npz keyed <session>/<trial index in the partition>')
    parser.add_argument('--name', default='SpeckChip')
    parser.add_argument('--setting', action='append', default=[], help='NAME=acousticScale,blankPenalty (repeatable)')
    parser.add_argument('--partition', default='test')
    parser.add_argument('--beam', type=float, default=18)
    parser.add_argument('--rnnDir', default=None, help='default <baseDir>/derived/rnns/baselineRelease (session list)')
    parser.add_argument('--snnDataDir', default=None, help='default <baseDir>/derived/snnData')
    parser.add_argument('--lmDir', default=None, help='default <baseDir>/languageModel (3-gram)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
    from neuralDecoder.utils.lmDecoderUtils import _cer_and_wer as cerAndWer
    from snnDecoder.dataset import load_partitions
    from snnDecoder.utils import greedy_decode, edit_distance
    from compareDecoders import asciiToText, wordErrors, perSession

    rnnDir = args.rnnDir or os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
    snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
    lmDir = args.lmDir or os.path.join(args.baseDir, 'languageModel')
    settings = [(n, *map(float, v.split(','))) for n, v in (s.split('=', 1) for s in args.setting)] \
        or [('default', 0.8, float(np.log(2)))]

    #the same trials, in the same order, as compareDecoders.py's SNN entries
    evalSessions = list(OmegaConf.load(os.path.join(rnnDir, 'args.yaml'))['dataset']['sessions'])[4:19]
    z = np.load(args.logits)
    logits, lengths, trueSentences, sessions, keys, edits, trueLens = [], [], [], [], [], [], []
    for part in load_partitions(snnDataDir, evalSessions, args.partition):
        for i in range(len(part)):
            k = f'{part.session}/{i}'
            if k not in z:
                raise KeyError(f'{args.logits} has no {k}: it must hold every {args.partition} trial')
            l = z[k]
            logits.append(l); lengths.append(len(l)); keys.append(k); sessions.append(part.session)
            trueSentences.append(asciiToText(part.transcriptions[i]))
            edits.append(edit_distance(greedy_decode(l, l.shape[1] - 1), part.labels[i])); trueLens.append(len(part.labels[i]))
    padded = np.zeros((len(logits), max(lengths), logits[0].shape[1]), dtype=np.float32)
    for i, l in enumerate(logits):
        padded[i, :len(l)] = l
    rearranged = lmDecoderUtils.rearrange_speech_logits(padded, has_sil=True)
    per = float(np.sum(edits) / np.sum(trueLens))
    print(f'{args.name}: {len(keys)} trials from {len(evalSessions)} sessions, greedy PER {per:.4f}', flush=True)

    result = {args.name: {'per': per, 'per_session': perSession(edits, trueLens, sessions), 'logits': args.logits,
                          'settings': {}}}
    sentences = {'keys': keys, 'true': trueSentences}
    for name, acoustic, blank in settings:
        print(f'Loading language model (acoustic scale {acoustic}) from {lmDir}', flush=True)
        lm = lmDecoderUtils.build_lm_decoder(lmDir, acoustic_scale=acoustic, nbest=1, beam=args.beam)
        decoded = []
        for i in range(len(rearranged)):
            lm.Reset()
            decoded.append(lmDecoderUtils.lm_decode(lm, rearranged[i, :lengths[i]], blankPenalty=blank).strip())
        cer, wer = cerAndWer(decoded, trueSentences, outputType='speech_sil', returnCI=True)
        we = wordErrors(decoded, trueSentences)
        result[args.name]['settings'][name] = {'acousticScale': acoustic, 'blankPenalty': blank, 'wer': wer,
                                               'lm_per': cer,
                                               'wer_session': perSession([e for e, _ in we], [n for _, n in we], sessions)}
        sentences[name] = decoded
        print(f'{args.name} [{name}: acoustic {acoustic}, blank {blank:.3f}]: WER {wer}', flush=True)
        del lm

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out + '.json', 'w'), indent=1, default=float)
    json.dump(sentences, open(args.out + '_sentences.json', 'w'), indent=1)
    print(f'wrote {args.out}.json and {args.out}_sentences.json')
