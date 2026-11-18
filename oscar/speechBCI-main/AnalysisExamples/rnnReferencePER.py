#greedy (no language model) phoneme error rate of a trained RNN, per session, on the same trials the SNN reports its
#test PER on, so the two can be compared directly. Also pools it over the session sets of the SNN diagnostics and sweep.
#
#   python rnnReferencePER.py --baseDir /path/to/speechPaperRelease_08_20
#
#by default uses the cached RNN from derived.tar.gz (derived/rnns/baselineRelease); writes rnnReferencePER.json
import argparse
import json
import os
import numpy as np

SESSION_SETS = {
    'diagnostics (4 sessions)': ['t12.2022.06.02', 't12.2022.06.07', 't12.2022.06.14', 't12.2022.06.16'],
    'sweep (8 sessions)': ['t12.2022.05.05', 't12.2022.05.24', 't12.2022.06.07', 't12.2022.06.16',
                           't12.2022.06.28', 't12.2022.07.14', 't12.2022.07.27', 't12.2022.08.11'],
}

def sessionPER(ckptDir, baseDir, session, partition):
    from omegaconf import OmegaConf
    import tensorflow as tf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['outputDir'] = ckptDir #the cached args.yaml keeps the authors' /oak/stanford/... outputDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None
    args['testDir'] = partition
    sessions = list(args['dataset']['sessions'])
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0
        args['dataset']['dataDir'][x] = baseDir + '/derived/tfRecords'
    args['dataset']['datasetProbabilityVal'][sessions.index(session)] = 1.0

    tf.compat.v1.reset_default_graph()
    out = NeuralSequenceDecoder(args).inference()
    return float(np.sum(out['editDistances'])), float(np.sum(out['trueSeqLengths'])), len(out['editDistances'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy test PER of the RNN per session.')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--ckptDir', default=None, help='default <baseDir>/derived/rnns/baselineRelease')
    parser.add_argument('--partition', default='test')
    parser.add_argument('--sessions', nargs='+', default=None, help='default: the 19 sessions the SNN trains on')
    parser.add_argument('--out', default='rnnReferencePER.json')
    args = parser.parse_args()

    ckptDir = args.ckptDir or os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
    if args.sessions is None:
        from snnDecoder.dataset import BASELINE_SESSIONS
        args.sessions = BASELINE_SESSIONS

    perSession = {}
    for s in args.sessions:
        err, n, nTrials = sessionPER(ckptDir, args.baseDir, s, args.partition)
        perSession[s] = {'edits': err, 'phonemes': n, 'trials': nTrials, 'per': err / n}
        print(f'{s}: PER {err/n:.4f} ({nTrials} trials)', flush=True)

    pooled = {}
    for name, sess in list(SESSION_SETS.items()) + [(f'all ({len(args.sessions)} sessions)', args.sessions)]:
        sess = [s for s in sess if s in perSession]
        if sess:
            pooled[name] = sum(perSession[s]['edits'] for s in sess) / sum(perSession[s]['phonemes'] for s in sess)
    print('\nRNN greedy ' + args.partition + ' PER, pooled:')
    for name, per in pooled.items():
        print(f'   {name:28s} {per:.4f}')
    with open(args.out, 'w') as f:
        json.dump({'ckptDir': ckptDir, 'partition': args.partition, 'perSession': perSession, 'pooled': pooled}, f, indent=1)
    print('saved ' + args.out)
