#Greedy phoneme error per test SENTENCE (no language model) for the paper RNN, any trained SNNs and any saved logits
#(the Speck chip's), on the same evaluation sessions and trials as compareDecoders.py (the RNN's sessions 4-18).
#compareDecoders.py keeps only per-session PER; this keeps each sentence's edit distance and length, so the figures
#can show one point per sentence. Nothing is trained and no language model is loaded.
#
#   python perTrialPER.py --baseDir /path/to/speechBCI_dryad --gru GRU96=/path/controls/gru_1024_out80_full96.json \
#       --snn LSNN=/path/final_model_speech_iaf_soft.pth --snn SpeckR3=/path/best_model_weights.pth \
#       --logits SpeckChip=/path/chip_logits.npz --out /path/perTrialPER.json
#
#Writes {"keys": [<session>/<trial>], "sessions": [...], "true": [sentence], "phonemes": [n],
#        "<NAME>": {"per": pooled PER, "edits": [per-sentence edit distance]}}; per-sentence PER = edits / phonemes.
import argparse
import json
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy PER per test sentence, every decoder on the same trials.')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--snn', action='append', default=[], help='NAME=checkpoint.pth (repeatable)')
    parser.add_argument('--gru', action='append', default=[], help='NAME=controls/<model>.json, as compareDecoders.py (repeatable)')
    parser.add_argument('--logits', action='append', default=[], help='NAME=logits.npz keyed <session>/<trial> (repeatable)')
    parser.add_argument('--noRNN', action='store_true', help='skip the paper RNN (TensorFlow)')
    parser.add_argument('--rnnDir', default=None, help='default <baseDir>/derived/rnns/baselineRelease')
    parser.add_argument('--snnDataDir', default=None, help='default <baseDir>/derived/snnData')
    parser.add_argument('--partition', default='test')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from compareDecoders import asciiToText
    from snnDecoder.dataset import load_partitions
    from snnDecoder.utils import greedy_decode, edit_distance

    rnnDir = args.rnnDir or os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
    snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
    rnnArgs = OmegaConf.load(os.path.join(rnnDir, 'args.yaml'))
    evalSessions = [rnnArgs['dataset']['sessions'][i] for i in range(4, 19)]

    #the reference trial order: every test trial of the evaluation sessions, as decodeSavedLogits.py
    keys, sessions, true, labels = [], [], [], []
    for part in load_partitions(snnDataDir, evalSessions, args.partition):
        for i in range(len(part)):
            keys.append(f'{part.session}/{i}'); sessions.append(part.session)
            true.append(asciiToText(part.transcriptions[i])); labels.append(part.labels[i])
    result = {'keys': keys, 'sessions': sessions, 'true': true, 'phonemes': [int(len(l)) for l in labels]}
    nPh = np.asarray(result['phonemes'])

    def add(name, edits):
        edits = [int(e) for e in edits]
        assert len(edits) == len(keys), f'{name}: {len(edits)} trials, expected {len(keys)}'
        result[name] = {'per': float(np.sum(edits) / np.sum(nPh)), 'edits': edits}
        print(f'{name}: {len(edits)} sentences, greedy PER {result[name]["per"]:.4f}', flush=True)

    if not args.noRNN:
        import tensorflow as tf
        from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
        rnnArgs['loadDir'] = rnnDir
        rnnArgs['outputDir'] = rnnDir
        rnnArgs['mode'] = 'infer'
        rnnArgs['loadCheckpointIdx'] = None
        rnnArgs['testDir'] = args.partition
        for x in range(len(rnnArgs['dataset']['datasetProbabilityVal'])):
            rnnArgs['dataset']['datasetProbabilityVal'][x] = 0.0
        for sessIdx in range(4, 19):
            rnnArgs['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
            rnnArgs['dataset']['dataDir'][sessIdx] = args.baseDir + '/derived/tfRecords'
        tf.compat.v1.reset_default_graph()
        rnnOut = NeuralSequenceDecoder(rnnArgs).inference()
        rnnTrue = [asciiToText(rnnOut['transcriptions'][x, :]) for x in range(rnnOut['transcriptions'].shape[0])]
        assert rnnTrue == true, 'the RNN was evaluated on different trials (or in a different order)'
        assert list(map(int, rnnOut['trueSeqLengths'])) == result['phonemes'], 'RNN phoneme counts differ'
        add('RNN', rnnOut['editDistances'])

    from snnDecoder.inference import snn_inference
    for name, ckpt in (s.split('=', 1) for s in args.snn):
        out = snn_inference(ckpt, snnDataDir, evalSessions, args.partition)
        sTrue = [asciiToText(out['transcriptions'][x, :]) for x in range(out['transcriptions'].shape[0])]
        assert sTrue == true, f'{name} was evaluated on different trials (or in a different order)'
        add(name, out['editDistances'])

    from snnDecoder.inference import gru_inference
    for name, path in (s.split('=', 1) for s in args.gru):
        out = gru_inference(path, snnDataDir, evalSessions, args.partition)
        gTrue = [asciiToText(out['transcriptions'][x, :]) for x in range(out['transcriptions'].shape[0])]
        assert gTrue == true, f'{name} was evaluated on different trials (or in a different order)'
        add(name, out['editDistances'])

    for name, path in (s.split('=', 1) for s in args.logits):
        z = np.load(path)
        missing = [k for k in keys if k not in z]
        assert not missing, f'{path} lacks {len(missing)} trials, e.g. {missing[:3]}'
        add(name, [edit_distance(greedy_decode(z[k], z[k].shape[1] - 1), lab) for k, lab in zip(keys, labels)])

    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f'saved {args.out}')
