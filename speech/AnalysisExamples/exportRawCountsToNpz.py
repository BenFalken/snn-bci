#exports the raw threshold-crossing counts of the 128 area-6v electrodes (tx1, columns 0-127 of competitionData),
#20 ms bins, as .npz for the spike-native SNN (SNNDecoder --arch native), trial for trial aligned with the RNN's data:
#labels and transcriptions are copied from the existing SNN export (derived/snnData, exportTFRecordsToNpz.py), and
#each trial is checked against it -- the same length, and the same values once z-scored per block as the RNN's
#pipeline (makeTFRecordsFromSession.py) does.
#
#   python exportRawCountsToNpz.py --baseDir /path/to/speechPaperRelease_08_20
#
#writes <baseDir>/derived/snnDataTx1/<session>/<partition>.npz in the format of SNNDecoder/snnDecoder/dataset.py,
#features = raw counts (float32 holding integers), 128 columns
import argparse
import os
import numpy as np
import scipy.io

N_6V = 128

def blockZscore(trials, blockIdx):
    #as makeTFRecordsFromSession.makeTFRecordsFromCompetitionFiles: per block, over all its trials' bins
    out = [None] * len(trials)
    for b in np.unique(blockIdx):
        idx = np.argwhere(blockIdx == b)[:, 0]
        feats = np.concatenate([trials[i] for i in range(idx[0], idx[-1] + 1)], axis=0)
        mean, std = feats.mean(axis=0, keepdims=True), feats.std(axis=0, keepdims=True)
        for i in idx:
            out[i] = (trials[i] - mean) / (std + 1e-8)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export raw area-6v threshold-crossing counts for the spike-native SNN.')
    parser.add_argument('--baseDir', required=True)
    parser.add_argument('--snnDataDir', default=None, help='the existing export to align with (default <baseDir>/derived/snnData)')
    parser.add_argument('--outDir', default=None, help='default <baseDir>/derived/snnDataTx1')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    snnDataDir = args.snnDataDir or os.path.join(args.baseDir, 'derived', 'snnData')
    outDir = args.outDir or os.path.join(args.baseDir, 'derived', 'snnDataTx1')
    for session in sorted(os.listdir(snnDataDir)):
        for partition in ['train', 'test', 'competitionHoldOut']:
            ref = os.path.join(snnDataDir, session, partition + '.npz')
            mat = os.path.join(args.baseDir, 'competitionData', partition, session + '.mat')
            out = os.path.join(outDir, session, partition + '.npz')
            if not (os.path.isfile(ref) and os.path.isfile(mat)):
                continue
            if os.path.isfile(out) and not args.overwrite:
                print('[skip] ' + out + ' exists')
                continue
            dat = scipy.io.loadmat(mat)
            raw = [dat['tx1'][0, i][:, :N_6V].astype(np.float32) for i in range(dat['tx1'].shape[1])]
            assert all(np.array_equal(r, np.round(r)) for r in raw), f'{mat}: tx1 is not whole counts'
            with np.load(ref) as f:
                refFeatures, refLens = f['features'], f['featureLens']
                labels, labelLens, transcriptions = f['labels'], f['labelLens'], f['transcriptions']
            assert len(raw) == len(refLens), f'{session} {partition}: {len(raw)} trials, the SNN data has {len(refLens)}'
            assert np.array_equal([len(r) for r in raw], refLens), f'{session} {partition}: trial lengths differ'
            zs = np.concatenate(blockZscore(raw, np.squeeze(dat['blockIdx'])), axis=0)
            err = np.abs(zs - refFeatures[:, :N_6V]).max()
            assert err < 1e-3, f'{session} {partition}: z-scored counts differ from the SNN data by {err:.2e}'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, features=np.concatenate(raw, axis=0), featureLens=refLens, labels=labels,
                     labelLens=labelLens, transcriptions=transcriptions)
            allCounts = np.concatenate(raw, axis=0)
            print(f'{out}: {len(raw)} trials, mean count {allCounts.mean():.3f} per channel per bin, '
                  f'max {int(allCounts.max())}, checked (max deviation {err:.1e})', flush=True)
    print('saved ' + outDir)
