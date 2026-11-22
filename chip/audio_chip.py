"""
The audio Speck SNN on a Speck2f devkit: every checkpoint given is run on its dataset's test split three ways and
scored the same way (mel-spectrogram r, as the pipeline):

    float      the trained network in PyTorch
    emulated   the chip's network (8-bit weights, integer thresholds) stepped in software, a step's input summed
               before firing, as in training
    event_sim  the chip's event-by-event dynamics in software (snndec.chip.run_events), with --layer1-scale
    chip       the devkit (skipped with --nochip), with --layer1-scale: Hidden 1-3 on the chip, the host buffer
               and readout on the host

--layer1-scale multiplies layer 1's threshold on the chip: the chip fires on each input event, and dense input then
loses charge at early hard resets; x4 was chosen on the validation splits with run_events (mean r 0.605 there).

The host turns the recording into chip input exactly as in training (channels kept, counts spread over 10 steps of
4 ms per 40 ms frame and binarized, 10 delays); the chip returns Hidden 3's spikes; the host's readout (two EMAs,
then the linear map to the 128 bands) decodes them, one output per frame. The stream starts BURN frames before the
test split, so the network is not scored from rest.

    python chip/audio_chip.py --data DIR=CKPT [DIR=CKPT ...] [--nochip] [--wait 0.003] --out results.json

Per run: r (float, emulated, chip), and for the chip: latency per chip step and per 40 ms output (the loop's wall
time, its wait included), mean power, energy per output (power x latency), input events per output.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snndec import chip, data, tasks  # noqa: E402
from snndec.convert import load  # noqa: E402
from snndec.evaluate import recording_file  # noqa: E402

BURN = 25                                               # frames streamed before the test split (1 s)


@torch.no_grad()
def decode(model, spikes):
    """Hidden 3's spikes (T, n), one row per model step -> one output per input frame (the host readout)."""
    st = model.readout.init_state(1, 'cpu')
    outs = [model.readout.step(torch.as_tensor(s, dtype=torch.float32)[None], st)[0] for s in spikes]
    return torch.stack(outs)[model.out_every - 1::model.out_every].numpy()


def refit_decode(model, spikes, readout):
    """Hidden 3's spikes per chip step, decoded by a readout refitted to the chip's firing on the training stream
    (audio_refit.py -> readout_event.npz: the same smoothing, a refitted linear map), one output per frame."""
    from audio_refit import features
    r = np.load(readout)
    return features(model, np.asarray(spikes), r['decay'] if 'decay' in r.files else None) @ r['W'] + r['b']


@torch.no_grad()
def run_one(pipeline, ckpt, use_chip, wait, devkit, layer1_scale=1.0, save=None, readout=None):
    model, info = load(ckpt)
    model.eval()
    rec, W, n = data.audio_recording(pipeline, recording_file(info))
    if info.get('channels') is not None:                                  # older checkpoints keep them in info
        rec = rec[:, list(info['channels'])]
    y_test = np.load(f'{pipeline}/window_targets_test.npy').astype(np.float32)
    first = n['train'] + n['valid'] + W - 1                               # the frame of test row 0
    start = max(0, first - BURN)
    seg = torch.as_tensor(rec[start:])[None]
    y_float = model(seg)[0].numpy()
    events = chip.chip_events(model, model.input_events[0].numpy())       # (frames x 10, chip inputs), 0/1
    sel = slice(first - start, first - start + len(y_test))
    out = {'ckpt': ckpt, 'data': pipeline, 'test_outputs': len(y_test), 'chip_inputs': events.shape[1],
           'steps_per_output': model.out_every, 'r_float': tasks.mean_correlation(y_test, y_float[sel])}
    width = model.layers[-1].width
    out['r_emulated'] = tasks.mean_correlation(y_test, decode(model, chip.simulate(model, events, 1.0, 'software'))[sel])
    out['layer1_scale'] = layer1_scale
    out['networks'] = model.branches['n'] if model.branches else 1
    h3_sim = chip.simulate(model, events, layer1_scale)
    y_sim = decode(model, h3_sim)
    out['r_event_sim'] = tasks.mean_correlation(y_test, y_sim[sel])
    preds = {'float': y_float[sel], 'event_sim': y_sim[sel]}
    if readout:
        preds['event_sim_refit'] = refit_decode(model, h3_sim, readout)[sel]
        out['r_event_sim_refit'] = tasks.mean_correlation(y_test, preds['event_sim_refit'])
    if use_chip:
        stacks = chip.branch_stacks(model, layer1_scale)
        if len(stacks) == 1:
            dev = chip.Devkit(stacks[0][0], events.shape[1], devkit=devkit, wait_time=wait)
            dev.reset()
            counts, loop_s, power_w, n_events = dev.run(events, width)
        else:                                     # several networks, one chip (MultiDevkit); used-neuron reset
            chip.RESET_USED_ONLY = True
            dev = chip.MultiDevkit([q for q, _, _ in stacks], [len(l) for _, l, _ in stacks], devkit=devkit,
                                   wait_time=wait)
            dev.reset()
            parts, loop_s, power_w, n_events = dev.run([events[:, l] for _, l, _ in stacks])
            counts = np.zeros((len(events), width))
            for (_, _, last), c in zip(stacks, parts):
                counts[:, last] = c
            out['cores'] = dev.cores
        dev.close()
        steps = len(events)
        preds['chip'] = decode(model, counts)[sel]
        if readout:                                  # the deployed readout, live
            preds['chip_refit'] = refit_decode(model, counts, readout)[sel]
            out['r_chip_refit'] = tasks.mean_correlation(y_test, preds['chip_refit'])
        if save:                                 # Hidden 3 per chip step, to decode with a refitted readout
            os.makedirs(save, exist_ok=True)
            np.savez_compressed(os.path.join(save, 'chip_counts.npz'), h3=np.minimum(counts, 255).astype(np.uint8),
                                first_frame=first - start, n_test=len(y_test))
        out.update(r_chip=tasks.mean_correlation(y_test, preds['chip']),
                   wait_s=wait, latency_step_ms=1e3 * loop_s / steps,
                   latency_output_ms=1e3 * loop_s / steps * model.out_every, power_mw=1e3 * power_w,
                   energy_output_uj=1e6 * power_w * loop_s / steps * model.out_every,
                   input_events_per_output=n_events / steps * model.out_every)
    for k in [k for k in ('event_sim', 'chip') if k in preds]:       # agreement with the trained network over time
        q = len(y_test) // 6
        out[f'agreement_by_sixth_{k}'] = [float(np.corrcoef(preds[k][i * q:(i + 1) * q].ravel(),
                                                            preds['float'][i * q:(i + 1) * q].ravel())[0, 1])
                                          for i in range(6)]
    if save:
        os.makedirs(save, exist_ok=True)
        for k, v in preds.items():
            np.save(os.path.join(save, f'{k}.npy'), v.astype(np.float32))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', required=True,
                   help='PIPELINE_DIR=CHECKPOINT[=READOUT]: READOUT = a refitted readout (readout_event.npz), applied live')
    p.add_argument('--nochip', action='store_true')
    p.add_argument('--wait', type=float, default=0.003, help='seconds the host waits per chip step')
    p.add_argument('--devkit', default='speck2fdevkit:0')
    p.add_argument('--layer1-scale', type=float, default=1.0, help="layer 1's threshold on the chip, x this")
    p.add_argument('--out', required=True)
    p.add_argument('--save-preds', default=None, help='folder: <dataset>/{float,event_sim,chip}.npy test predictions')
    a = p.parse_args()
    results = json.load(open(a.out)) if os.path.isfile(a.out) else {}
    for item in a.data:
        pipeline, ckpt, *ro = item.split('=')
        ro = ro[0] if ro else None
        name = os.path.basename(os.path.dirname(pipeline.rstrip('/')))
        if name in results and (a.nochip or 'r_chip' in results[name]):
            print(f'{name}: done, skipped')
            continue
        t0 = time.time()
        results[name] = run_one(pipeline, ckpt, not a.nochip, a.wait, a.devkit, a.layer1_scale,
                                os.path.join(a.save_preds, name) if a.save_preds else None, readout=ro)
        r = results[name]
        print(f"{name}: r float {r['r_float']:.3f}  emulated {r['r_emulated']:.3f}  "
              f"event-sim x{r['layer1_scale']:g} {r['r_event_sim']:.3f}" +
              (f"  chip {r['r_chip']:.3f} | {r['latency_output_ms']:.1f} ms and {r['energy_output_uj']:.1f} uJ "
               f"per output at {r['power_mw']:.2f} mW" if 'r_chip' in r else '') +
              (f"  | refitted readout: event-sim {r['r_event_sim_refit']:.3f}" if 'r_event_sim_refit' in r else '') +
              (f", chip {r['r_chip_refit']:.3f}" if 'r_chip_refit' in r else '') + f"  ({time.time() - t0:.0f} s)")
        json.dump(results, open(a.out, 'w'), indent=1)                  # after every run: resumable
    print('wrote', a.out)


if __name__ == '__main__':
    main()
