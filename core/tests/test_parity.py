"""
Parity with the original code: each project's model (its own class, its own forward) and the unified model loaded
from the same checkpoint give the same outputs on the same input.

    python tests/test_parity.py            (paths below; a case is skipped when its code or checkpoint is missing)

Checkpoints: real ones where they are on this machine, else the original class with random weights, saved in its
own format. Also checks streaming (step) against forward, and the chip mapping of every Speck network.
"""

import os
import sys
import tempfile
import warnings
import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[2]))
from paths import WORKSPACE, AUDITORY_REPO, SPEECH_SNN  # noqa: E402  (the repository's paths.yaml)

import numpy as np
import torch

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snndec import convert  # noqa: E402

HOME = os.path.expanduser('~')
MOTOR = f'{WORKSPACE}/spike_bmi_lsnn/main'
MOTOR_CK = f'{WORKSPACE}/spike_bmi_lsnn_data/lsnn_vs_snn/indy_20160411_01'
AUDIO = str(AUDITORY_REPO)
AUDIO_LSNN_CK = f'{AUDIO}/pipeline_data/Neil/aSTG/07_31_13/pipeline/lsnn_model.pt'
AUDIO_SPECK_CK = ('/private/tmp/claude-501/-Users-benjaminfalkenburg-Downloads-speechBCI-main/'
                  '2cd331d6-6514-4a76-8f8b-ca9427b209cc/scratchpad/speck_audio/07_31_13_r13_binB_s1337/model.pt')
SPEECH = str(SPEECH_SNN)
results = []


def report(name, ref, new, tol=1e-4, extra=''):
    err = float((ref - new).abs().max())
    ok = err <= tol * max(1.0, float(ref.abs().max()))
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} max |diff| {err:.2e}{extra}")


def stream_check(name, model, x, day=None):
    model.eval()
    with torch.no_grad():
        full = model(x, day)
        model.reset_stream()
        outs = [o for t in range(x.shape[1]) if (o := model.step(x[:, t], day)) is not None]
    report(f'{name}: step() == forward()', full, torch.stack(outs, 1))


def chip_check(name, model):
    try:
        model.check_chip()
        print(f'PASS  {name}: maps onto Speck2f')
        results.append(True)
    except Exception as e:
        print(f'FAIL  {name}: does not map onto Speck2f ({str(e).splitlines()[0][:70]})')
        results.append(False)


def motor():
    if not os.path.isdir(MOTOR):
        return print('skip  motor (no spike_bmi checkout)')
    sys.path[:0] = [MOTOR, f'{MOTOR}/inference', f'{MOTOR}/snn_training']
    from decoder_eval import load_snn_model
    torch.manual_seed(0)
    x = (torch.rand(1, 300, 96) < 0.15).float()                       # binarized counts, 4 ms bins, 1.2 s
    for sub, kind in (('lsnn_final', 'LSNN'), ('snn_delay4_wide', 'Speck (delays 0-200 ms)')):
        path = f'{MOTOR_CK}/{sub}/best_model_weights.pth'
        if not os.path.isfile(path):
            print(f'skip  motor {kind} ({path} missing)')
            continue
        ref_model = load_snn_model(path, 'bmi')[0].eval()
        model, _ = convert.load(path)
        model.eval()
        with torch.no_grad():
            ref = ref_model(x.transpose(0, 1))[0].transpose(0, 1)
            new = model(x)
        report(f'motor {kind}', ref, new)
        stream_check(f'motor {kind}', model, x[:, :120])
        if model.spec['kind'] == 'speck':
            chip_check(f'motor {kind}', model)


def audio():
    if not os.path.isdir(AUDIO):
        return print('skip  audio (no AuditoryDecoding checkout)')
    sys.path[:0] = [AUDIO, f'{AUDIO}/modules/decoders/torch/lsnn', f'{AUDIO}/modules/decoders/torch/speck_snn']
    import lsnn as A
    import speck_snn as S
    torch.manual_seed(1)
    if os.path.isfile(AUDIO_LSNN_CK):
        ck = torch.load(AUDIO_LSNN_CK, map_location='cpu', weights_only=False)
        ref_model = A.LSNNNet(ck['n_inputs'], ck['n_outputs'], **ck['net_kwargs']).eval()
        ref_model.load_state_dict(ck['state_dict'])
        model, _ = convert.load(AUDIO_LSNN_CK)
        model.eval()
        x = torch.randn(4, 17, ck['n_inputs'])                          # 4 windows of 17 frames
        with torch.no_grad():
            report('audio LSNN (07_31_13)', ref_model(x, all_steps=True)[0], model(x))
        stream_check('audio LSNN', model, x[:1])
    if os.path.isfile(AUDIO_SPECK_CK):
        ck = torch.load(AUDIO_SPECK_CK, map_location='cpu', weights_only=False)
        K = ck['substeps']
        ref_model = S.SpeckSNN(ck['n_inputs'], ck['n_outputs'], ck['hidden'], [d * K for d in ck['delays']],
                               ck['ema_decay'] ** (1 / K), ck['readout'], ck.get('thresholds'),
                               ck['state_dict'].get('input_scale'), ck['ema_stages'], ck['ema_per_neuron']).eval()
        ref_model.load_state_dict({k: v for k, v in ck['state_dict'].items() if not k.endswith('v_mem')}, strict=False)
        model, _ = convert.load(AUDIO_SPECK_CK)
        model.eval()
        counts = torch.poisson(torch.full((1, 60, ck['n_inputs']), 0.8))
        dec = S.SpeckSNNDecoder.__new__(S.SpeckSNNDecoder)                # its spread(), without a pipeline
        dec.substeps, dec.binarize = K, ck['binarize']
        with torch.no_grad():
            ref = ref_model(torch.as_tensor(dec.spread(counts.numpy())))[0][:, K - 1::K]
            new = model(counts)
        report(f'audio Speck (binB, {K} sub-steps)', ref, new)
        stream_check('audio Speck', model, counts[:, :30])
        chip_check('audio Speck', model)


def speech():
    if not os.path.isdir(SPEECH):
        return print('skip  speech (no SNNDecoder checkout)')
    sys.path.insert(0, SPEECH)
    from snnDecoder.train import model_kwargs, parse_arguments
    from snnDecoder.model import create_model
    base = ['--config', f'{SPEECH}/snnDecoder/configs/speech_native.yaml', '--data-dir', 'unused']
    cases = {
        'speech LSNN (final, Model B)': base + ['--neuron-type', 'iaf', '--adaptive-fraction', '0.9', '--tau-adapt',
                                                '25', '--readout-all', 'true', '--output-stride', '4',
                                                '--hidden-dims', '64', '48'],
        'speech Speck (R2: z > 0.5, 64 ch x 8 delays)': base + [
            '--neuron-type', 'iaf', '--reset-type', 'hard', '--recurrent', 'false', '--tau-syn-range', '0', '0',
            '--hidden-dims', '128', '256', '256', '--input-encoding', 'binary', '--encoding-threshold', '0.5',
            '--input-delays', '0', '2', '4', '7', '10', '13', '19', '25', '--output-stride', '4'],
        'speech Speck (R3: counts over 4 sub-steps)': base + [
            '--neuron-type', 'iaf', '--reset-type', 'hard', '--recurrent', 'false', '--tau-syn-range', '0', '0',
            '--hidden-dims', '128', '256', '256', '--input-substeps', '4', '--encoding-threshold', '1',
            '--input-delays', '0', '4', '13', '25', '--output-stride', '4', '--readout-traces', '0', '5', '15'],
    }
    for name, argv in cases.items():
        args = parse_arguments(argv)
        args.num_input_channels = 256 if 'LSNN' in name else 128
        if 'R2' in name:
            args.input_channel_list = sorted(np.random.default_rng(0).choice(128, 64, replace=False).tolist())
        torch.manual_seed(2)
        ref_model = create_model(n_sessions=3, **model_kwargs(vars(args))).eval()
        with torch.no_grad():                                          # trained-looking day gains and time constants
            ref_model.day_gain.uniform_(0.7, 1.3)
            ref_model.out_log_tau.add_(torch.rand_like(ref_model.out_log_tau))
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            torch.save({'model_state_dict': ref_model.state_dict(), 'args': vars(args)}, f.name)
            model, _ = convert.load(f.name)
        model.eval()
        x = torch.randn(2, 80, args.num_input_channels) if 'R3' not in name else \
            torch.poisson(torch.full((2, 80, args.num_input_channels), 0.6))
        with torch.no_grad():
            report(name, ref_model(x, 1), model(x, 1))
        stream_check(name, model, x[:1, :24], day=1)
        if model.spec['kind'] == 'speck':
            chip_check(name, model)


if __name__ == '__main__':
    motor()
    audio()
    speech()
    print(f'\n{sum(results)}/{len(results)} checks passed')
    sys.exit(0 if all(results) else 1)
