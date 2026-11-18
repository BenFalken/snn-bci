#exports the RNN's TFRecords (rnn_step1, or derived.tar.gz) to .npz files for the SNN decoder (SNNDecoder), so both
#decoders train and test on identical, already block-normalized features and phoneme labels. Run in the TensorFlow env.
#
#   python exportTFRecordsToNpz.py --tfRecordDir <baseDir>/derived/tfRecords --outDir <baseDir>/derived/snnData
#
#writes <outDir>/<session>/<partition>.npz for every session/partition folder with a .tfrecord in it
#(format: speech/snnDecoder/dataset.py)
import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

N_INPUT_FEATURES = 256
MAX_SEQ_ELEMENTS = 500

def readPartition(folder):
    files = sorted([str(x) for x in Path(folder).glob('*.tfrecord')])
    features = {
        'inputFeatures': tf.io.FixedLenSequenceFeature([N_INPUT_FEATURES], tf.float32, allow_missing=True),
        'seqClassIDs': tf.io.FixedLenFeature((MAX_SEQ_ELEMENTS), tf.int64),
        'nTimeSteps': tf.io.FixedLenFeature((), tf.int64),
        'nSeqElements': tf.io.FixedLenFeature((), tf.int64),
        'transcription': tf.io.FixedLenFeature((MAX_SEQ_ELEMENTS), tf.int64),
    }
    feats, labels, transcriptions = [], [], []
    for record in tf.data.TFRecordDataset(files):
        dat = tf.io.parse_single_example(record, features)
        nSteps = int(dat['nTimeSteps'])
        nSeq = int(dat['nSeqElements'])
        feats.append(dat['inputFeatures'].numpy()[:nSteps].astype(np.float32))
        #seqClassIDs are phoneme index + 1 (0 = padding); the RNN subtracts 1 before its CTC loss
        labels.append(dat['seqClassIDs'].numpy()[:nSeq].astype(np.int32) - 1)
        transcriptions.append(dat['transcription'].numpy().astype(np.int32))
    return feats, labels, transcriptions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TFRecords to .npz for SNNDecoder.')
    parser.add_argument('--tfRecordDir', required=True, help='derived/tfRecords (one folder per session)')
    parser.add_argument('--outDir', required=True, help='e.g. derived/snnData')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    for sessionDir in sorted(Path(args.tfRecordDir).iterdir()):
        if not sessionDir.is_dir():
            continue
        for partDir in sorted(sessionDir.iterdir()):
            if not partDir.is_dir() or not any(partDir.glob('*.tfrecord')):
                continue
            outFile = os.path.join(args.outDir, sessionDir.name, partDir.name + '.npz')
            if os.path.isfile(outFile) and not args.overwrite:
                print('[skip] ' + outFile + ' exists')
                continue
            feats, labels, transcriptions = readPartition(partDir)
            if len(feats) == 0:
                continue
            #labels are kept exactly as in the TFRecords, also for competitionHoldOut (redacted sentences)
            os.makedirs(os.path.dirname(outFile), exist_ok=True)
            np.savez(outFile,
                     features=np.concatenate(feats, axis=0),
                     featureLens=np.array([len(f) for f in feats]),
                     labels=np.concatenate(labels).astype(np.int32),
                     labelLens=np.array([len(l) for l in labels]),
                     transcriptions=np.stack(transcriptions))
            print(f'{outFile}: {len(feats)} trials, {sum(len(f) for f in feats)} bins')
