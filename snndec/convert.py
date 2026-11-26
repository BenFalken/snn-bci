"""
Loading the three projects' checkpoints into SNNDecoder, unchanged (tests/test_parity.py checks the outputs match the
original code's).

    load(path)          any checkpoint: a unified one, or one of the originals (detected from its contents)
    from_motor_lsnn     spike_bmi models/model_bmi_lsnn.py (train_snn.py --arch lsnn)
    from_motor_speck    spike_bmi models/model_bmi.py SNN_Speck (the Speck SNN, with or without input delays)
    from_audio_lsnn     AuditoryDecoding modules/decoders/torch/lsnn
    from_audio_speck    AuditoryDecoding modules/decoders/torch/speck_snn
    from_speech         speechBCI SNNDecoder rsnn_native.py (--arch native: the LSNN and the Speck screens)

Each returns (model, info): info holds what the task needs beyond the network (motor: the velocity scaling's
subject; audio: the window mode; speech: the sessions, in day-index order).
Options the final models never used are refused rather than approximated.
"""

import torch

from .model import SNNDecoder


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _set(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('channel_index')]
    if missing or unexpected:
        raise KeyError(f'checkpoint does not match the model: missing {missing}, unexpected {unexpected}')
    return model


def _layers(sd, prefix, n):
    """The original layers' tensors under our names."""
    out = {}
    for i in range(n):
        for k in ('ff.weight', 'rec.weight', 'log_tau_syn', 'adapt_mask'):
            if f'{prefix}.{i}.{k}' in sd and sd[f'{prefix}.{i}.{k}'] is not None:
                out[f'layers.{i}.{k}'] = sd[f'{prefix}.{i}.{k}']
    return out


# ------------------------------------------------------------------------------------------------------ motor
def from_motor_lsnn(path):
    ck = _load(path)
    sd, a = ck['model_state_dict'], ck['args']
    n = sum(1 for k in sd if k.startswith('spiking.') and k.endswith('.ff.weight'))
    hidden = [sd[f'spiking.{i}.ff.weight'].shape[0] for i in range(n)]
    spec = dict(kind='lsnn', step_ms=4.0, n_channels=sd['spiking.0.ff.weight'].shape[1], n_outputs=2, out_every=1,
                input=dict(encoding='clip' if a.get('binarize_input', True) else 'none'),
                layers=dict(hidden=hidden, recurrent='spiking.0.rec.weight' in sd, reset=a.get('lsnn_reset', 'soft'),
                            tau_syn_ms=list(a.get('lsnn_tau_syn_ms', [20.0, 1300.0])),
                            adaptive_fraction=a.get('lsnn_adaptive_fraction', 0.9),
                            tau_adapt_ms=a.get('lsnn_tau_adapt_ms', 500.0), adapt_scale=a.get('lsnn_adapt_scale', 1.8),
                            max_spikes=a.get('lsnn_max_spikes', 3), min_vmem=-1.0, max_vmem=50.0,
                            rec_init_scale=a.get('lsnn_rec_init_scale', 0.5), surrogate='lsnn'),
                readout=dict(type='leaky', read='all' if a.get('lsnn_readout_all') else 'last',
                             tau_ms=a.get('lsnn_out_tau_ms', 40.0), stages=sd['out_log_tau'].shape[0], init='bias'))
    model = SNNDecoder(spec)
    new = _layers(sd, 'spiking', n)
    new.update({'readout.linear.weight': sd['readout.weight'], 'readout.linear.bias': sd['readout.bias'],
                'readout.log_tau': sd['out_log_tau']})
    return _set(model, new), {'experiment': ck.get('experiment'), 'source': 'motor_lsnn', 'args': a}


def from_motor_speck(path):
    ck = _load(path)
    sd, a = ck['model_state_dict'], ck['args']
    for opt in ('input_delays_random', 'input_mix', 'learn_input_delays', 'input_plan_dir', 'recurrent'):
        if a.get(opt):
            raise NotImplementedError(f'motor Speck SNN option {opt} is not part of the unified model')
    if a.get('tau_syn') or a.get('neuron_type', 'iaf') != 'iaf' or a.get('final_layer_reset_type') not in (None, a.get('reset_type', 'hard')):
        raise NotImplementedError('only the chip configuration (IAF, one reset for every layer, no synaptic filters) converts')
    lin = sorted(int(k.split('.')[1]) for k in sd if k.startswith('layers.') and k.endswith('.weight')
                 and sd[k].dim() == 2)
    hidden = [sd[f'layers.{i}.weight'].shape[0] for i in lin]
    thr = [float(sd[f'layers.{i + 1}.neuron.spike_threshold']) for i in lin]
    delays = a.get('input_delays') or [0]
    n_ch = sd[f'layers.{lin[0]}.weight'].shape[1] // len(delays)
    spec = dict(kind='speck', step_ms=4.0, n_channels=n_ch, n_outputs=2, out_every=1,
                input=dict(encoding='clip' if a.get('binarize_input', True) else 'none', delays=delays),
                layers=dict(hidden=hidden, reset=a.get('reset_type', 'hard'), thresholds=thr, max_spikes=None,
                            min_vmem=a.get('min_vmem'), surrogate=a.get('surrogate_grad', 'periodic_exponential')),
                readout=dict(type='ema', read='last', decay=0.8, stages=a.get('temporal_decay_stages', 1),
                             head='population', normalize=False, neutral=float(sd['neutral_scaled_pred'])))
    model = SNNDecoder(spec)
    new = {f'layers.{j}.ff.weight': sd[f'layers.{i}.weight'] for j, i in enumerate(lin)}
    new.update({'readout.decay_raw': sd['_temporal_decay_raw'], 'readout.positions': sd['positions'],
                'readout.neutral': sd['neutral_scaled_pred']})
    return _set(model, new), {'experiment': ck.get('experiment'), 'source': 'motor_speck', 'args': a}


# ------------------------------------------------------------------------------------------------------ audio
def from_audio_lsnn(path):
    ck = _load(path)
    sd, k = ck['state_dict'], ck['net_kwargs']
    n = len(k['hidden_dims'])
    spec = dict(kind='lsnn', step_ms=k['step_ms'], n_channels=ck['n_inputs'], n_outputs=ck['n_outputs'], out_every=1,
                input=dict(dropout=k.get('input_dropout', 0.0)),
                layers=dict(hidden=list(k['hidden_dims']), recurrent=True, reset='soft',
                            tau_syn_ms=list(k['tau_syn_ms']), adaptive_fraction=k['adaptive_fraction'],
                            tau_adapt_ms=k['tau_adapt_ms'], adapt_scale=k['adapt_scale'], max_spikes=k['max_spikes'],
                            min_vmem=k['min_vmem'], max_vmem=k['max_vmem'], rec_init_scale=k['rec_init_scale'],
                            surrogate='lsnn', layer_dropout=k.get('layer_dropout', 0.0)),
                readout=dict(type='leaky', read='all', tau_ms=k['out_tau_ms'], stages=1, init='bias',
                             dropout=k.get('layer_dropout', 0.0)))
    model = SNNDecoder(spec)
    new = _layers(sd, 'spiking', n)
    new.update({'readout.linear.weight': sd['readout.weight'], 'readout.linear.bias': sd['readout.bias'],
                'readout.log_tau': sd['out_log_tau'][None]})
    return _set(model, new), {'source': 'audio_lsnn', 'mode': ck.get('mode', 'window')}


def from_audio_speck(path):
    ck = _load(path)
    sd = ck['state_dict']
    readout = ck.get('readout', 'band')
    if readout == 'pushpull':
        raise NotImplementedError("audio Speck readout 'pushpull' is not part of the unified model")
    lin = sorted(int(k.split('.')[1]) for k in sd if k.startswith('linears.'))
    hidden = [sd[f'linears.{i}.weight'].shape[0] for i in lin]
    K = int(ck.get('substeps', 1))
    binarize = bool(ck.get('binarize', False))
    enc = 'spread' if K > 1 else ('clip' if binarize else 'none')
    spec = dict(kind='speck', step_ms=40.0, n_channels=ck['n_inputs'], n_outputs=ck['n_outputs'], out_every=1,
                input=dict(encoding=enc, substeps=K, binarize=binarize, delays=list(ck['delays']),
                           input_scale=sd.get('input_scale')),
                layers=dict(hidden=hidden, reset='hard', thresholds=ck.get('thresholds') or 1.0, max_spikes=None,
                            min_vmem=-1.0, surrogate='single_exponential'),
                readout=dict(type='ema', read='last', decay=ck.get('ema_decay', 0.3), stages=ck.get('ema_stages', 2),
                             per_neuron=ck.get('ema_per_neuron', False),
                             head='affine' if readout == 'band' else 'linear', normalize=True))
    model = SNNDecoder(spec)
    new = {f'layers.{j}.ff.weight': sd[f'linears.{i}.weight'] for j, i in enumerate(lin)}
    new['readout.decay_raw'] = sd['decay_raw']
    if readout in ('linear', 'hidden'):
        new['readout.linear.weight'], new['readout.linear.bias'] = sd['host_linear.weight'], sd['host_linear.bias']
    else:
        new['readout.gain'], new['readout.offset'] = sd['gain'], sd['offset']
    if sd.get('input_scale') is not None:
        new['host.input_scale'] = sd['input_scale']
    return _set(model, new), {'source': 'audio_speck', 'channels': ck.get('channels')}


# ----------------------------------------------------------------------------------------------------- speech
_SPEECH_REFUSED = ('feedback', 'syn_delay_in', 'syn_delay_rec', 'syn_delay_ff', 'rec_delays', 'learn_delays',
                   'tau_mem_range', 'tau_adapt_range', 'input_traces', 'rec_no_self')


def from_speech(path):
    ck = _load(path)
    sd, a = ck['model_state_dict'], ck['args']
    if a.get('arch') != 'native':
        raise NotImplementedError(f"speech arch {a.get('arch')!r}: only --arch native converts")
    on = lambda v: bool(v) and (max(v) > 0 if isinstance(v, (list, tuple)) else bool(v))
    bad = [o for o in _SPEECH_REFUSED if on(a.get(o))]
    if a.get('neuron_type', 'iaf') not in ('iaf', 'alif') or a.get('readout_source', 'spikes') != 'spikes' \
            or a.get('learn_threshold', 'none') != 'none' or a.get('day_input', 'gain') != 'gain' \
            or a.get('input_encoding', 'none') not in ('none', 'binary') or not a.get('norm_syn', False) and \
            a.get('tau_syn_range') and min(a['tau_syn_range']) > 0:
        bad.append('neuron/readout/encoding settings')
    if bad:
        raise NotImplementedError(f'speech options not part of the unified model: {bad}')
    n = len(a['hidden_dims'])
    K = int(a.get('input_substeps', 1))
    sessions = ck.get('sessions') or a.get('model_sessions')
    if a.get('input_channel_list'):
        channels = list(a['input_channel_list'])
    elif a.get('input_channels'):
        channels = list(range(*a['input_channels']))
    else:
        channels = None
    if K > 1:
        inp = dict(encoding='spread', substeps=K, binarize=True, threshold=a.get('encoding_threshold', 1.0))
    elif a.get('input_encoding') == 'binary':
        inp = dict(encoding='threshold', threshold=a.get('encoding_threshold', 0.25))
    else:
        inp = dict(encoding='none')
    inp.update(channels=channels, dropout=a.get('input_dropout', 0.2), delays=a.get('input_delays'))
    tsr = a.get('tau_syn_range')
    alif = a.get('neuron_type') == 'alif'
    spec = dict(kind='lsnn' if (a.get('recurrent', True) or alif or a.get('adaptive_fraction')) else 'speck',
                step_ms=20.0, n_channels=a['num_input_channels'], n_outputs=sd['readout.bias'].shape[0],
                n_days=sd['day_gain'].shape[0], out_every=a.get('output_stride', 1), input=inp,
                layers=dict(hidden=list(a['hidden_dims']), recurrent=a.get('recurrent', True),
                            reset=a.get('reset_type', 'soft'),
                            tau_syn_ms=[t * 20.0 for t in tsr] if tsr and min(tsr) > 0 else None,
                            adaptive_fraction=1.0 if alif else a.get('adaptive_fraction', 0.0),
                            tau_adapt_ms=a.get('tau_adapt', 24.0) * 20.0, adapt_scale=a.get('adapt_scale', 1.8),
                            thresholds=a.get('spike_thresholds') or [1.0], max_spikes=a.get('max_spikes')
                            if a.get('spike_fn') == 'max' else None, min_vmem=a.get('min_vmem'),
                            max_vmem=a.get('max_vmem'), rec_init_scale=a.get('rec_init_scale', 1.0), adapt_seed=5,
                            surrogate=a.get('surrogate_grad', 'single_exponential'),
                            layer_dropout=a.get('layer_dropout', 0.2)),
                readout=dict(type='leaky', read='all' if a.get('readout_all') else 'last',
                             tau_ms=a.get('out_tau_init', 2.0) * 20.0, stages=a.get('out_stages', 1), init='zero',
                             traces_ms=[t * 20.0 for t in a['readout_traces']] if a.get('readout_traces') else None))
    th = spec['layers']['thresholds']
    spec['layers']['thresholds'] = th * n if len(th) == 1 else th
    model = SNNDecoder(spec)
    new = _layers(sd, 'layers', n)
    if alif:                                                   # every neuron adapts
        for i in range(n):
            new[f'layers.{i}.adapt_mask'] = torch.ones(a['hidden_dims'][i])
    S = model.readout.log_tau.shape[0]
    new.update({'readout.linear.weight': sd['readout.weight'], 'readout.linear.bias': sd['readout.bias'],
                'readout.log_tau': sd['out_log_tau'][None].expand(S, -1).clone(), 'host.day_gain': sd['day_gain']})
    return _set(model, new), {'source': 'speech', 'sessions': sessions}


# ------------------------------------------------------------------------------------------------- dispatch
def save(model: SNNDecoder, path: str, info: dict = None):
    torch.save({'format': 'snndec/1', 'spec': model.spec, 'state_dict': model.state_dict(), 'info': info or {}}, path)


def load(path):
    """A checkpoint of any of the four kinds -> (SNNDecoder, info)."""
    ck = _load(path)
    if ck.get('format', '').startswith('snndec'):
        return _set(SNNDecoder(ck['spec']), ck['state_dict']), ck.get('info', {})
    if 'net_kwargs' in ck:
        return from_audio_lsnn(path)
    if 'linears.0.weight' in ck.get('state_dict', {}):
        return from_audio_speck(path)
    sd = ck.get('model_state_dict', {})
    if 'spiking.0.ff.weight' in sd:
        return from_motor_lsnn(path)
    if '_temporal_decay_raw' in sd:
        return from_motor_speck(path)
    if 'day_gain' in sd and 'layers.0.ff.weight' in sd:
        return from_speech(path)
    raise ValueError(f'{path}: not a checkpoint this package knows')
