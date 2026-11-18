"""
Run a trained SNN over a partition of sessions and return its output in the
same format as NeuralSequenceDecoder.inference(), so rnn_step3's language
model decoding (lmDecoderUtils.cer_with_lm_decoder) and error rates run on
it unchanged:

    logits          (N, T_max, 41)  per 20 ms bin, zero-padded
    logitLengths    (N,)            valid bins per trial
    decodedSeqs     (N, 500)        greedy CTC decoding, -1 padded
    editDistances   (N,)            greedy decoding vs the true phonemes
    trueSeqs        (N, 500)        true phoneme IDs 0-39, -1 padded
    trueSeqLengths  (N,)
    transcriptions  (N, 500)        ASCII, zero-padded
    cer             phoneme error rate of the greedy decoding (sum of edit
                    distances / sum of true lengths), None without labels

plus 'sessions' (each trial's session), 'inputLengths' (20 ms bins), 'frameKernel' /
'frameStride' (bins per output frame and between frames: 1/1, or 32/4 for the
RNN-shaped recurrent SNN) and 'op_estimate'
(energy.op_energy_estimate.finalize_snn_ops() over every trial).

Trials are in session order, each session's in file order -- the RNN's order
for the same session list, which rnn_step3 checks via the transcriptions.
"""

import numpy as np
import torch

from .dataset import SessionBatches, load_partitions, smooth_inputs
from .energy.op_energy_estimate import finalize_snn_ops
from .model import create_model, load_model_weights
from .train import model_kwargs
from .utils import edit_distance, greedy_decode


def load_snn_model(checkpoint_path, device='cpu'):
    """Rebuild a train.py checkpoint from its saved args, in eval mode.
    Returns (model, checkpoint); checkpoint['sessions'] is the model's
    session list (input layer i = sessions[i])."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = dict(checkpoint['args'])
    args['weight_init'] = None
    model = create_model(n_sessions=len(checkpoint['sessions']), **model_kwargs(args))
    load_model_weights(model, checkpoint['model_state_dict'], source_description=checkpoint_path)
    model.smooth_sd = float(args.get('smooth_sd') or 0.0)
    return model.to(device).eval(), checkpoint


def snn_inference(checkpoint_path, data_dir, sessions, partition, device='cpu', batch_size=32,
                  max_seq_elements=500, count_ops=True):
    """Evaluate the SNN on `partition` of `sessions` (names; each must be one
    of the checkpoint's sessions). See the module docstring for the output."""
    model, checkpoint = load_snn_model(checkpoint_path, device)
    model_sessions = list(checkpoint['sessions'])
    unknown = [s for s in sessions if s not in model_sessions]
    if unknown:
        raise ValueError(f"{checkpoint_path} has no input layer for {unknown}")
    parts = load_partitions(data_dir, sessions, partition, [model_sessions.index(s) for s in sessions])
    blank = model.n_classes - 1

    logits, lengths, true_seqs, transcriptions, trial_sessions, input_lengths = [], [], [], [], [], []
    ops = {'mac': 0, 'acc': 0.0, 'elementwise': 0}
    with torch.no_grad():
        for part in parts:
            for batch in SessionBatches([part], batch_size, shuffle=False):
                x = smooth_inputs(batch['x'].to(device), model.smooth_sd)
                out = model(x, part.day_idx, lengths=batch['lengths'], count_ops=count_ops).cpu().numpy()
                out_lengths = model.output_lengths(batch['lengths']).cpu().numpy()
                if count_ops:
                    for k in ops:
                        ops[k] += model.op_counts[k]
                for b, i in enumerate(batch['trial_idx']):
                    n = int(out_lengths[b])
                    logits.append(out[b, :n])
                    lengths.append(n)
                    input_lengths.append(int(batch['lengths'][b]))
                    true_seqs.append(part.labels[i])
                    transcriptions.append(part.transcriptions[i])
                    trial_sessions.append(part.session)

    n_trials, t_max = len(logits), max(lengths)
    padded = np.zeros((n_trials, t_max, model.n_classes), dtype=np.float32)
    decoded_seqs = -np.ones((n_trials, max_seq_elements), dtype=np.int32)
    true_padded = -np.ones((n_trials, max_seq_elements), dtype=np.int32)
    edit = np.zeros(n_trials)
    for i, (l, target) in enumerate(zip(logits, true_seqs)):
        padded[i, :len(l)] = l
        decoded = greedy_decode(l, blank)[:max_seq_elements]
        decoded_seqs[i, :len(decoded)] = decoded
        true_padded[i, :len(target)] = target
        edit[i] = edit_distance(decoded, target)
    true_lengths = np.array([len(t) for t in true_seqs])

    return {
        'logits': padded,
        'logitLengths': np.array(lengths),
        'decodedSeqs': decoded_seqs,
        'editDistances': edit,
        'trueSeqs': true_padded,
        'trueSeqLengths': true_lengths,
        'transcriptions': np.stack(transcriptions),
        'cer': float(edit.sum() / true_lengths.sum()) if true_lengths.sum() > 0 else None,
        'sessions': trial_sessions,
        'inputLengths': np.array(input_lengths),
        'frameKernel': model.frame_kernel, 'frameStride': model.frame_stride,
        'op_estimate': finalize_snn_ops(ops['mac'], ops['acc'], ops['elementwise'],
                                        n_samples=int(np.sum(input_lengths))) if count_ops else None,
    }


def gru_inference(json_path, data_dir, sessions, partition, device='cpu', batch_size=32, max_seq_elements=500):
    """A GRU control trained by scripts/ann_controls.py (its <name>.json with the weights in <name>.pt next to it)
    on `partition` of `sessions`, in the same output format as snn_inference (op_estimate None), so the language
    model and error rates run on it unchanged."""
    import json
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
    import ann_controls
    info = json.load(open(json_path))
    args = info['args']
    model_sessions = list(args['sessions'])
    unknown = [s for s in sessions if s not in model_sessions]
    if unknown:
        raise ValueError(f"{json_path} has no input layer for {unknown}")
    parts = load_partitions(data_dir, sessions, partition, [model_sessions.index(s) for s in sessions])
    ann_controls.N_IN = int(parts[0].features[0].shape[1])
    model = ann_controls.MODELS[args['model']](len(model_sessions))
    model.load_state_dict(torch.load(os.path.splitext(json_path)[0] + '.pt', map_location='cpu'))
    model = model.to(device).eval()
    smooth_sd = float(args.get('smooth_sd') or 0.0)
    n_classes, blank = ann_controls.N_CLASSES, ann_controls.BLANK

    logits, lengths, true_seqs, transcriptions, trial_sessions, input_lengths = [], [], [], [], [], []
    with torch.no_grad():
        for part in parts:
            for batch in SessionBatches([part], batch_size, shuffle=False):
                x = smooth_inputs(batch['x'].to(device), smooth_sd)
                out = model(x, part.day_idx).cpu().numpy()
                n_out = ann_controls.out_lengths(model, batch['lengths']).cpu().numpy()
                for b, i in enumerate(batch['trial_idx']):
                    n = int(n_out[b])
                    logits.append(out[b, :n])
                    lengths.append(n)
                    input_lengths.append(int(batch['lengths'][b]))
                    true_seqs.append(part.labels[i])
                    transcriptions.append(part.transcriptions[i])
                    trial_sessions.append(part.session)

    n_trials, t_max = len(logits), max(lengths)
    padded = np.zeros((n_trials, t_max, n_classes), dtype=np.float32)
    decoded_seqs = -np.ones((n_trials, max_seq_elements), dtype=np.int32)
    true_padded = -np.ones((n_trials, max_seq_elements), dtype=np.int32)
    edit = np.zeros(n_trials)
    for i, (l, target) in enumerate(zip(logits, true_seqs)):
        padded[i, :len(l)] = l
        decoded = greedy_decode(l, blank)[:max_seq_elements]
        decoded_seqs[i, :len(decoded)] = decoded
        true_padded[i, :len(target)] = target
        edit[i] = edit_distance(decoded, target)
    true_lengths = np.array([len(t) for t in true_seqs])
    return {'logits': padded, 'logitLengths': np.array(lengths), 'decodedSeqs': decoded_seqs,
            'editDistances': edit, 'trueSeqs': true_padded, 'trueSeqLengths': true_lengths,
            'transcriptions': np.stack(transcriptions),
            'cer': float(edit.sum() / true_lengths.sum()) if true_lengths.sum() > 0 else None,
            'sessions': trial_sessions, 'inputLengths': np.array(input_lengths),
            # frames: GRUNet emits at the end of every out_stride bins (kernel = stride); StackGRU reads a
            # window-bin stack every stride bins
            # (LSTM, QRNN, Wiener, Kalman controls declare frame_kernel / frame_stride themselves)
            # (the Wiener filter's window ends with each 80 ms block, so for timing its kernel is the stride; its
            # 160 ms of history is reported as inputMs)
            'frameKernel': (getattr(model, 'frame_stride', None) if hasattr(model, 'frames') else None)
                           or getattr(model, 'frame_kernel', None) or getattr(model, 'window', None)
                           or getattr(model, 'out_stride', 1),
            'inputMs': (getattr(model, 'frame_kernel', None) or 1) * 20,
            'recipe': args.get('recipe'),
            'frameStride': getattr(model, 'frame_stride', None) or getattr(model, 'stride', None)
                           or getattr(model, 'out_stride', 1), 'op_estimate': None,
            'smooth_sd': smooth_sd}
