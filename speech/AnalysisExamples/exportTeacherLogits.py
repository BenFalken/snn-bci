#exports a trained RNN's per-frame logits for every trial, as distillation targets for the SNN (SNNDecoder train.py
#--teacher-dir). The recurrent SNN (rsnn.py) emits exactly the RNN's frames (32-bin stacking, stride 4), so frame k of
#the teacher and of the student see the same bins. Run in the TensorFlow env:
#
#   python exportTeacherLogits.py --baseDir /path/to/speechPaperRelease_08_20
#
#writes <outDir>/<session>/<partition>.npz (default outDir: <baseDir>/derived/teacherLogits/baselineRelease):
#   logits          (sum F_i, 41) float16  every trial's frames concatenated, in the RNN's class order (blank last)
#   frameLens       (N,)                   F_i per trial
#and checks each trial against the SNN data (<baseDir>/derived/snnData): same transcription, and
#F_i = (T_i - 32) // 4 + 1 for its T_i bins.
import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a trained RNN\'s per-frame logits for SNN distillation.')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--rnnDir', default=None, help='default <baseDir>/derived/rnns/baselineRelease')
    parser.add_argument('--snnDataDir', default=None, help='default <baseDir>/derived/snnData')
    parser.add_argument('--outDir', default=None, help='default <baseDir>/derived/teacherLogits/baselineRelease')
    parser.add_argument('--partitions', nargs='+', default=['train', 'test'])
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import tensorflow as tf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
    from snnDecoder.dataset import BASELINE_SESSIONS

    rnnDir = args.rnnDir or os.path.join(args.baseDir, 'derived', 'rnns', 'baselineRelease')
    snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
    outDir = args.outDir or os.path.join(args.baseDir, 'derived', 'teacherLogits', os.path.basename(rnnDir.rstrip('/')))

    for partition in args.partitions:
        rnnArgs = OmegaConf.load(os.path.join(rnnDir, 'args.yaml'))
        rnnArgs['loadDir'] = rnnDir
        rnnArgs['outputDir'] = rnnDir #the cached args.yaml keeps the authors' /oak/stanford/... outputDir
        rnnArgs['mode'] = 'infer'
        rnnArgs['loadCheckpointIdx'] = None
        rnnArgs['testDir'] = partition
        rnnSessions = list(rnnArgs['dataset']['sessions'])
        sessions = [s for s in BASELINE_SESSIONS if s in rnnSessions and os.path.isfile(os.path.join(snnDataDir, s, partition + '.npz'))]
        for x in range(len(rnnArgs['dataset']['datasetProbabilityVal'])):
            rnnArgs['dataset']['datasetProbabilityVal'][x] = 0.0
            rnnArgs['dataset']['dataDir'][x] = args.baseDir + '/derived/tfRecords'
        for s in sessions:
            rnnArgs['dataset']['datasetProbabilityVal'][rnnSessions.index(s)] = 1.0

        tf.compat.v1.reset_default_graph()
        nsd = NeuralSequenceDecoder(rnnArgs)
        out = nsd.inference()   #trials in session order (the args' order), each session in file order
        print(f'{partition}: {len(out["logitLengths"])} trials from {len(sessions)} sessions, greedy PER {out["cer"]:.4f}', flush=True)

        start = 0
        for s in [x for x in rnnSessions if x in sessions]:
            with np.load(os.path.join(snnDataDir, s, partition + '.npz')) as f:
                featureLens, transcriptions = f['featureLens'], f['transcriptions']
            n = len(featureLens)
            idx = range(start, start + n)
            start += n
            frameLens = np.array([int(out['logitLengths'][i]) for i in idx])
            expected = np.maximum((featureLens - 32) // 4 + 1, 0)
            assert np.array_equal(frameLens, expected), f'{s} {partition}: frame counts differ from the SNN data'
            assert np.array_equal(out['transcriptions'][list(idx)][:, :transcriptions.shape[1]], transcriptions), \
                f'{s} {partition}: trials differ from the SNN data'
            logits = np.concatenate([out['logits'][i, :frameLens[k]] for k, i in enumerate(idx)]).astype(np.float16)
            os.makedirs(os.path.join(outDir, s), exist_ok=True)
            np.savez(os.path.join(outDir, s, partition + '.npz'), logits=logits, frameLens=frameLens)
            print(f'  {s}: {n} trials, {int(frameLens.sum())} frames', flush=True)
        assert start == len(out['logitLengths']), f'{partition}: {len(out["logitLengths"]) - start} unmatched trials'
    print('saved ' + outDir)
