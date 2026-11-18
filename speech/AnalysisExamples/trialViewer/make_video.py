#renders exported trials (plotExampleTrial.py --exportJson) as mp4s: one pass per decoder, ground truth first, a cursor
#sweeping the trial while that decoder's phonemes are revealed and voiced, and the language model's words appearing
#when the decoder settles on them. Earlier passes stay on screen (faded) for comparison. The phoneme synthesizer is the
#viewer's (viewer_template.html), ported to numpy; needs ffmpeg on the PATH.
#
#   python make_video.py ~/Downloads/viewer_trials/*.json --outDir ../../figures/viewer/videos --combine
import argparse
import base64
import os
import subprocess
import wave

import numpy as np
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SR = 44100
LABELS = {'truth': 'Ground truth', 'RNN': 'Paper RNN', 'SNN': 'LSNN (spiking)', 'Wiener': 'Wiener filter', 'Kalman': 'Kalman filter',
          'GRU': 'GRU', 'LSTM': 'LSTM'}
#the decoder colours of plotDecoderFigures.py (FAMILY_COLORS), so a decoder looks the same in the figures and the videos
COLORS = {'truth': '#7A7A7A', 'RNN': '#262626', 'SNN': '#C0392B', 'Wiener': '#D4AC2B', 'Kalman': '#8A6512', 'GRU': '#9DBBE3',
          'LSTM': '#2F5FA7'}
F0 = {'truth': 112, 'RNN': 138, 'SNN': 164, 'Wiener': 190, 'Kalman': 214, 'GRU': 150, 'LSTM': 126}
INK, MUTED, RULE, SOFT, PAPER, CURSOR = '#14171C', '#5B6270', '#D8DCE3', '#ECEEF2', '#FFFFFF', '#D1273A'

#same table as the viewer
PH = {
    'AA': {'v': [730, 1090, 2440]}, 'AE': {'v': [660, 1720, 2410]}, 'AH': {'v': [640, 1190, 2390]},
    'AO': {'v': [570, 840, 2410]}, 'AW': {'v': [730, 1090, 2440], 'to': [300, 870, 2240]},
    'AY': {'v': [730, 1090, 2440], 'to': [270, 2290, 3010]}, 'EH': {'v': [530, 1840, 2480]},
    'ER': {'v': [490, 1350, 1690]}, 'EY': {'v': [530, 1840, 2480], 'to': [270, 2290, 3010]},
    'IH': {'v': [390, 1990, 2550]}, 'IY': {'v': [270, 2290, 3010]}, 'OW': {'v': [570, 840, 2410], 'to': [300, 870, 2240]},
    'OY': {'v': [570, 840, 2410], 'to': [270, 2290, 3010]}, 'UH': {'v': [440, 1020, 2240]}, 'UW': {'v': [300, 870, 2240]},
    'M': {'nasal': [250, 1100, 2200]}, 'N': {'nasal': [250, 1600, 2600]}, 'NG': {'nasal': [250, 2000, 2800]},
    'L': {'v': [360, 1300, 2900], 'amp': .7}, 'R': {'v': [420, 1300, 1600], 'amp': .7},
    'W': {'v': [300, 610, 2200], 'amp': .6}, 'Y': {'v': [280, 2250, 2900], 'amp': .6},
    'F': {'noise': [3800, 1.5], 'amp': .6}, 'TH': {'noise': [4200, 1.5], 'amp': .5}, 'S': {'noise': [4800, 3]},
    'SH': {'noise': [2600, 2.5]}, 'HH': {'noise': [1400, 1.2], 'amp': .5},
    'V': {'noise': [3800, 1.5], 'amp': .6, 'buzz': .35}, 'DH': {'noise': [4200, 1.5], 'amp': .5, 'buzz': .35},
    'Z': {'noise': [4800, 3], 'buzz': .35}, 'ZH': {'noise': [2600, 2.5], 'buzz': .35},
    'P': {'burst': 700}, 'T': {'burst': 3200}, 'K': {'burst': 1800}, 'B': {'burst': 700, 'buzz': .45},
    'D': {'burst': 3000, 'buzz': .45}, 'G': {'burst': 1800, 'buzz': .45},
    'CH': {'burst': 3200, 'noise': [2600, 2.5]}, 'JH': {'burst': 3000, 'noise': [2600, 2.5], 'buzz': .35}}


# ---------- synthesizer (Web Audio biquads: RBJ cookbook) ----------
def bandpass(f, q):
    w = 2 * np.pi * f / SR
    al = np.sin(w) / (2 * q)
    return np.array([al, 0, -al]), np.array([1 + al, -2 * np.cos(w), 1 - al])


def lowpass(f, q=0.7071):
    w = 2 * np.pi * f / SR
    al, c = np.sin(w) / (2 * q), np.cos(w)
    return np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]), np.array([1 + al, -2 * c, 1 - al])


def env(n, dur, amp):
    t = np.arange(n) / SR
    att, rel = min(0.015, dur / 4), 0.03
    e = np.minimum(1, t / att) * np.clip((dur - t) / rel, 0, 1)
    return amp * e


def voiced(dur, f0, F, to, amp, nasal=False):
    n = int(dur * SR)
    f = np.linspace(f0, f0 * 0.94, n)
    saw = 2 * ((np.cumsum(f) / SR) % 1) - 1
    out = np.zeros(n)
    gains = [1, .2, .1] if nasal else [1, .55, .3]
    blk = int(0.005 * SR)
    for k, fk in enumerate(F):
        y, zi = np.zeros(n), np.zeros(2)
        for s in range(0, n, blk):               # formant glides: coefficients updated every 5 ms
            fr = fk if to is None else fk + (to[k] - fk) * s / n
            b, a = bandpass(fr, fr / [70, 100, 140][k])
            y[s:s + blk], zi = lfilter(b / a[0], a / a[0], saw[s:s + blk], zi=zi)
        out += gains[k] * 2.2 * y
    return out * env(n, dur, amp)


def hiss(dur, freq, q, amp, rng):
    n = int(dur * SR)
    b, a = bandpass(freq, q)
    y = lfilter(b, a, rng.uniform(-1, 1, n))
    b, a = lowpass(5000)
    return lfilter(b, a, y) * env(n, dur, amp)


def phoneme(ph, dur, f0, rng):
    p = PH.get(ph)
    out = np.zeros(int((dur + 0.1) * SR))
    if p is None:
        return out

    def add(x, at=0.0):
        i = int(at * SR)
        out[i:i + len(x)] += x[:len(out) - i]
    a = p.get('amp', 1)
    if 'v' in p:
        add(voiced(dur, f0, p['v'], p.get('to'), 0.16 * a))
    if 'nasal' in p:
        add(voiced(dur, f0, p['nasal'], None, 0.14, nasal=True))
    if 'burst' in p:
        add(hiss(0.025, p['burst'], 1.5, 0.12, rng))
    t0 = 0.03 if 'burst' in p else 0
    d = max(0.05, dur - 0.03) if 'burst' in p else dur
    if 'noise' in p:
        add(hiss(d, p['noise'][0], p['noise'][1], 0.07 * a, rng), t0)
    if 'buzz' in p:
        add(voiced(min(d, 0.07) if 'burst' in p else d, f0, [500, 1500, 2500], None, 0.12 * p['buzz']), t0)
    return out * 0.5


# ---------- one trial ----------
def lanes_of(D):
    out = [{'key': 'truth', 'segs': D['truth']['phonemes'], 'words': [(w, s, e) for w, s, e in D['truth']['words']],
            'lm': None, 'per': None, 'wer': None}]
    for d in D['decoders']:
        out.append({'key': d['name'], 'segs': [s for s in d['segments'] if s[0] != 'blank'], 'words': None,
                    'lm': d.get('lm'), 'per': d.get('per'), 'wer': (d.get('lm') or {}).get('wer')})
    return out


def render_trial(D, out_path, speed, fps, hold, tmp, still=False):
    dur = D['nBins'] * D['binMs']
    lanes = lanes_of(D)
    rng = np.random.default_rng(0)
    passS = dur / 1000 / speed
    segS = hold + passS + hold                                  # per pass: still, sweep, still
    total = segS * len(lanes)
    audio = np.zeros(int((total + 1) * SR))
    for i, L in enumerate([] if still else lanes):
        base = i * segS + hold
        for ph, s, e in L['segs']:
            if ph == 'SIL':
                continue
            sec = min(0.4, max(0.07, (e - s) / 1000 / speed))
            x = phoneme(ph, sec, F0.get(L['key'], 150), rng)
            k = int((base + s / 1000 / speed) * SR)
            audio[k:k + len(x)] += x[:len(audio) - k]

    # ---- figure ----
    W, H = 1920, 1080
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)
    plt.rcParams.update({'font.family': ['Palatino', 'DejaVu Sans']})          # the deck's face
    fig.text(0.04, 0.955, f'“{D["sentence"]}”', fontsize=26, color=INK, weight='bold', va='center')
    fig.text(0.04, 0.915, f'{D["session"]} · {D["partition"]} trial {D["trial"]} · {dur / 1000:.2f} s · '
             f'played at {speed:g}× speed', fontsize=14, color=MUTED, va='center')
    passTxt = fig.text(0.96, 0.955, '', fontsize=24, ha='right', va='center', weight='bold')

    axH = fig.add_axes([0.16, 0.70, 0.80, 0.17])
    heat = np.frombuffer(base64.b64decode(D['heat']['data']), np.uint8).reshape(D['heat']['rows'], D['heat']['cols'])
    axH.imshow(heat, aspect='auto', cmap='RdBu_r', extent=[0, dur, heat.shape[0], 0], interpolation='nearest')
    axH.set_xticks([])
    gr = D['heat'].get('groups') or []
    axH.set_yticks([(a + b) / 2 for _, a, b in gr])
    axH.set_yticklabels([g[0] for g in gr], fontsize=11, color=MUTED)
    axH.tick_params(length=0)
    axH.set_ylabel('area 6v', fontsize=13, color=INK, rotation=0, ha='right', va='center', labelpad=14)
    for s in axH.spines.values():
        s.set_visible(False)

    n = len(lanes)
    axL = fig.add_axes([0.16, 0.15, 0.80, 0.53])
    axL.set_xlim(0, dur)
    axL.set_ylim(n, 0)
    axL.axis('off')
    masks, fades, wordArts, labels = [], [], [], []
    for i, L in enumerate(lanes):
        col = COLORS.get(L['key'], COLORS['GRU'])
        y = i
        axL.add_patch(Rectangle((0, y + 0.04), dur, 0.92, color=SOFT, lw=0, zorder=0))
        for ph, s, e in L['segs']:
            sil = ph == 'SIL'
            axL.add_patch(Rectangle((s, y + 0.10), e - s, 0.42, facecolor=RULE if sil else col, lw=0.6,
                                    edgecolor=PAPER, zorder=2))
            if not sil and (e - s) >= 60:
                axL.text((s + e) / 2, y + 0.31, ph, ha='center', va='center', fontsize=10 if e - s < 120 else 12,
                         color='white', zorder=3, clip_on=True)
        arts = []
        if L['words']:
            for w, s, e in L['words']:
                axL.add_patch(Rectangle((s, y + 0.60), e - s, 0.30, facecolor='white', edgecolor=col, lw=1.2, zorder=2))
                axL.text((s + e) / 2, y + 0.75, w, ha='center', va='center', fontsize=14, color=INK, zorder=3)
        elif L['lm']:
            groups = []                                  # words kept at the same moment share one label
            for w, t in L['lm']['words']:
                if groups and abs(groups[-1][1] - t) < 1:
                    groups[-1][0] += ' ' + w
                else:
                    groups.append([w, t])
            for j, (w, t) in enumerate(groups):
                t = min(t, dur - 1)
                ln = axL.plot([t, t], [y + 0.58, y + 0.94], color=col, lw=1.5, zorder=6)[0]
                late = t > 0.95 * dur                    # kept only at the end: label to the left of its line
                tx = axL.text(t - 25 if late else t + 25, y + (0.70 if j % 2 == 0 else 0.86), w,
                              ha='right' if late else 'left', va='center', fontsize=14,
                              color=INK, zorder=6, clip_on=False)
                arts.append((t, ln, tx))
        wordArts.append(arts)
        m = Rectangle((0, y + 0.04), dur, 0.92, facecolor=PAPER, lw=0, zorder=5)
        f = Rectangle((0, y + 0.04), dur, 0.92, facecolor=PAPER, alpha=0.5, lw=0, zorder=5)
        axL.add_patch(m)
        axL.add_patch(f)
        masks.append(m)
        fades.append(f)
        sub = []
        if L['per'] is not None:
            sub.append(f'PER {100 * L["per"]:.0f}%')
        if L['wer'] is not None:
            sub.append(f'WER {100 * L["wer"]:.0f}%')
        lab = axL.text(-dur * 0.012, y + 0.5, LABELS.get(L['key'], L['key']) + ('\n' + ' · '.join(sub) if sub else ''),
                       ha='right', va='center', fontsize=15, color=col, clip_on=False, linespacing=1.4)
        labels.append(lab)
    foot = fig.text(0.96, 0.06, 'each lane: decoded phonemes (grey: silence), then the words of the language model\n'
             'at the moment it keeps them  ·  earlier passes stay, faded', ha='right', va='center', fontsize=12,
             color=MUTED, linespacing=1.5)
    for t in range(0, int(dur) + 1, 1000):
        axL.text(t, n + 0.08, f'{t // 1000} s', ha='center', va='top', fontsize=12, color=MUTED)
    curH = axH.axvline(0, color=CURSOR, lw=2.5)
    curL = axL.axvline(0, color=CURSOR, lw=2.5, zorder=7)
    sentTxt = fig.text(0.16, 0.06, '', fontsize=24, color=INK, va='center')
    if still:                                                    # every lane in full, no cursor: one image
        for j in range(n):
            masks[j].set_visible(False); fades[j].set_visible(False)
        curH.set_visible(False); curL.set_visible(False)
        fig.texts[1].set_text(f'{D["session"]} · {D["partition"]} trial {D["trial"]} · {dur / 1000:.2f} s')
        foot.set_text('each lane: decoded phonemes (grey: silence), then the words of the language model\n'
                      'at the moment it keeps them')
        fig.savefig(out_path, dpi=100, facecolor=PAPER)
        plt.close(fig)
        print('wrote', out_path)
        return

    # ---- frames ----
    wav = os.path.join(tmp, 'a.wav')
    nFrames = int(round(total * fps))
    audio = audio[:int(nFrames / fps * SR)]
    peak = np.abs(audio).max()
    pcm = (audio / max(peak, 1e-9) * 0.7 * 32767).astype(np.int16)
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}',
                           '-r', str(fps), '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                           '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-shortest', out_path],
                          stdin=subprocess.PIPE)
    for k in range(nFrames):
        t = k / fps
        i = min(int(t // segS), n - 1)
        local = t - i * segS - hold
        cur = float(np.clip(local * speed * 1000, 0, dur))
        L = lanes[i]
        col = COLORS.get(L['key'], COLORS['GRU'])
        passTxt.set_text(LABELS.get(L['key'], L['key']))
        passTxt.set_color(col)
        for j in range(n):
            if j < i:                                            # done: shown, faded
                masks[j].set_visible(False); fades[j].set_visible(True); labels[j].set_alpha(0.55)
                for _, ln, tx in wordArts[j]:
                    ln.set_visible(True); tx.set_visible(True); tx.set_alpha(0.55); ln.set_alpha(0.55)
            elif j > i:                                          # to come: hidden
                masks[j].set_visible(True); masks[j].set_x(0); masks[j].set_width(dur); fades[j].set_visible(False)
                labels[j].set_alpha(0.35)
                for _, ln, tx in wordArts[j]:
                    ln.set_visible(False); tx.set_visible(False)
            else:                                                # this pass: revealed up to the cursor
                masks[j].set_visible(True); masks[j].set_x(cur); masks[j].set_width(dur - cur + 1)
                fades[j].set_visible(False); labels[j].set_alpha(1)
                for tw, ln, tx in wordArts[j]:
                    on = tw <= cur
                    ln.set_visible(on); tx.set_visible(on); tx.set_alpha(1); ln.set_alpha(1)
        if L['key'] == 'truth':
            said = [w for w, s, e in L['words'] if s <= cur]
        else:
            said = [w for w, tw in (L['lm'] or {'words': []})['words'] if min(tw, dur - 1) <= cur]
        sentTxt.set_text(' '.join(said))
        sentTxt.set_color(col)
        curH.set_xdata([cur, cur])
        curL.set_xdata([cur, cur])
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    print('wrote', out_path, f'({total:.0f} s)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trials', nargs='+', help='exported trial JSON files')
    p.add_argument('--outDir', default='videos')
    p.add_argument('--speed', type=float, default=0.5, help='playback speed (0.5 = half real time)')
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--hold', type=float, default=1.0, help='seconds of still before and after each pass')
    p.add_argument('--combine', action='store_true', help='also join the trials into all_trials.mp4')
    p.add_argument('--lanes', nargs='+', default=None, help='decoders to show, in this order (e.g. RNN LSTM SNN); ground truth always first')
    p.add_argument('--still', action='store_true', help='one PNG per trial with every lane shown, instead of a video')
    a = p.parse_args()
    import json
    import tempfile
    os.makedirs(a.outDir, exist_ok=True)
    outs = []
    for f in a.trials:
        D = json.load(open(f))
        if a.lanes:                                         # those decoders, in the order given
            by_name = {d['name']: d for d in D['decoders']}
            D['decoders'] = [by_name[n] for n in a.lanes if n in by_name]
        out = os.path.join(a.outDir, f'trial_{D["session"]}_{D["partition"]}_{D["trial"]}.' + ('png' if a.still else 'mp4'))
        with tempfile.TemporaryDirectory() as tmp:
            render_trial(D, out, a.speed, a.fps, a.hold, tmp, still=a.still)
        outs.append(out)
    if a.combine and len(outs) > 1:
        lst = os.path.join(a.outDir, 'list.txt')
        open(lst, 'w').write(''.join(f"file '{os.path.abspath(o)}'\n" for o in outs))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                        os.path.join(a.outDir, 'all_trials.mp4')], check=True)
        os.remove(lst)
        print('wrote', os.path.join(a.outDir, 'all_trials.mp4'))


if __name__ == '__main__':
    main()
