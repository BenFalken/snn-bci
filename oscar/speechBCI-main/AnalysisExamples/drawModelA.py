#figure: model A, the spike-native delay model -- topology, neuron dynamics with a synaptic delay, and the delayed
#input synapses -- as a standalone SVG (vector, editable in Illustrator / Inkscape / Keynote).
#   python drawModelA.py --out modelA_figure.svg
#The neuron traces in panel B are simulated with the model's own update (rsnn.SpikingLayer): synaptic filter
#i <- a i + (1 - a) x, integrate-and-fire without leak v <- v + i, at most 3 spikes per step, soft reset v <- v - n theta.
import argparse
import math

INK, QUIET, GRID = '#1d1d1f', '#5f6368', '#c9ccd1'
PURPLE, RED, BLUE, ORANGE = '#6a4fc4', '#d93025', '#2f5fa7', '#c0661e'
FILL = '#eef0f4'


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=14, weight=400, fill=INK, anchor='start', italic=False):
        st = ' font-style="italic"' if italic else ''
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                 f'text-anchor="{anchor}"{st}>{s}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK, w=1.5, dash=None, marker=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{marker})"' if marker else ''
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{w}"{d}{m}/>')

    def path(self, d, stroke=INK, w=1.5, fill='none', dash=None, marker=None, opacity=1.0):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{marker})"' if marker else ''
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{w}" fill="{fill}" opacity="{opacity}"{da}{m}/>')

    def rect(self, x, y, w, h, rx=8, stroke=INK, fill='none', sw=1.5, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" stroke="{stroke}" '
                 f'stroke-width="{sw}" fill="{fill}"{da}/>')

    def circle(self, cx, cy, r, stroke=INK, fill='white', sw=1.4):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"/>')

    def save(self, path):
        defs = ''.join(f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                       f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>'
                       for n, c in (('arr', INK), ('arrp', PURPLE), ('arrb', BLUE), ('arrq', QUIET)))
        body = '\n'.join(self.parts)
        open(path, 'w').write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" font-family="Helvetica, Arial, sans-serif"><defs>{defs}</defs>'
            f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>')


def capsule(svg, cx, top, h, label, count, sub=None, stroke=INK, fill=FILL, circ=INK, sub_fill=QUIET):
    """A layer as a capsule of neurons; returns the y positions of its drawn neurons (top three, bottom three)."""
    w = 40
    svg.rect(cx - w / 2, top, w, h, rx=20, stroke=stroke, fill=fill)
    n_each = 3 if h > 200 else 2
    ys = []
    for i in range(n_each):
        ys += [top + 22 + i * 24, top + h - 22 - i * 24]
    for y in ys:
        svg.circle(cx, y, 8, stroke=circ)
    svg.text(cx, top + h / 2 + 5, '⋮', size=18, fill=QUIET, anchor='middle')
    svg.text(cx, top - 12, count, size=17, weight=700, anchor='middle')
    svg.text(cx, top + h + 22, label, size=14, fill=INK, anchor='middle')
    if sub:
        svg.text(cx, top + h + 40, sub, size=12, fill=sub_fill, anchor='middle', weight=700 if sub_fill != QUIET else 400)
    return sorted(ys)


def delayed_links(svg, x1, ys1, x2, ys2, pairs):
    """Dashed purple arrows between neurons of two layers: delayed synapses."""
    for i, j in pairs:
        svg.line(x1 + 9, ys1[i], x2 - 11, ys2[j], stroke=PURPLE, w=1.3, dash='5 4', marker='arrp')


def fan(svg, x1, t1, b1, x2, t2, b2, color='#9aa0a6', opacity=0.22, r=20):
    """Shaded connection between two capsules, bounded by the lines tangent to their rounded ends."""
    def tangent(ca, cb, up):
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        n = math.hypot(dx, dy)
        nx, ny = (dy / n, -dx / n) if up else (-dy / n, dx / n)
        return (ca[0] + r * nx, ca[1] + r * ny), (cb[0] + r * nx, cb[1] + r * ny)
    (p1, p2) = tangent((x1, t1 + r), (x2, t2 + r), True)
    (q1, q2) = tangent((x1, b1 - r), (x2, b2 - r), False)
    svg.path(f'M{p1[0]:.1f} {p1[1]:.1f} L{p2[0]:.1f} {p2[1]:.1f} L{x2:.1f} {t2 + r:.1f} L{x2:.1f} {b2 - r:.1f} '
             f'L{q2[0]:.1f} {q2[1]:.1f} L{q1[0]:.1f} {q1[1]:.1f} L{x1:.1f} {b1 - r:.1f} L{x1:.1f} {t1 + r:.1f} Z',
             stroke='none', fill=color, opacity=opacity)


def panel_a(svg, y0):
    svg.text(40, y0, 'A', size=24, weight=700)
    svg.text(72, y0, 'Network topology', size=19, weight=700)
    mid = y0 + 260
    cols = {'in': 140, 'h1': 340, 'h2': 520, 'h3': 700, 'out': 900}
    hs = {'in': 250, 'h1': 330, 'h2': 330, 'h3': 330, 'out': 210}
    top = {k: mid - v / 2 for k, v in hs.items()}
    bot = {k: mid + v / 2 for k, v in hs.items()}
    # dashed box: the spiking core
    svg.rect(cols['h1'] - 70, y0 + 54, cols['h3'] - cols['h1'] + 160, 450, rx=12, stroke=PURPLE, dash='7 6', sw=1.5)
    svg.text((cols['h1'] + cols['h3']) / 2 + 10, y0 + 42, 'spiking core: 3 × 1,024 recurrent IAF neurons', size=15,
             weight=700, anchor='middle', fill=PURPLE)
    # connection fans
    fan(svg, cols['in'], top['in'], bot['in'], cols['h1'], top['h1'], bot['h1'])
    for a, b in (('h1', 'h2'), ('h2', 'h3')):
        fan(svg, cols[a], top[a], bot[a], cols[b], top[b], bot[b])
    fan(svg, cols['h3'], top['h3'], bot['h3'], cols['out'], top['out'], bot['out'], color=BLUE, opacity=0.12)
    ys_in = capsule(svg, cols['in'], top['in'], hs['in'], 'Input', '256', 'one 20 ms bin per step')
    ys = {}
    for k, n, rng in (('h1', 'Hidden 1', 'recurrent 0–40 ms'), ('h2', 'Hidden 2', 'recurrent 0–100 ms'),
                      ('h3', 'Hidden 3', 'recurrent 0–200 ms')):
        ys[k] = capsule(svg, cols[k], top[k], hs[k], n, '1,024', rng, sub_fill=PURPLE)
    capsule(svg, cols['out'], top['out'], hs['out'], '41 leaky outputs', '41', 'never spike', stroke=BLUE,
            fill='#eef3fb', circ=BLUE)
    # delayed synapses, drawn dashed: input -> layer 1 (several per pair), between layers
    delayed_links(svg, cols['in'], ys_in, cols['h1'], ys['h1'], [(0, 1), (0, 4), (2, 0), (3, 5), (5, 2), (4, 3)])
    for a, b in (('h1', 'h2'), ('h2', 'h3')):
        delayed_links(svg, cols[a], ys[a], cols[b], ys[b], [(0, 2), (2, 4), (4, 1), (5, 3)])
    # recurrent self-loops, dashed: delayed recurrent synapses
    for k in ('h1', 'h2', 'h3'):
        cx, ty = cols[k], top[k] + 30
        svg.path(f'M{cx + 20} {ty} C{cx + 62} {ty - 14}, {cx + 62} {ty + 66}, {cx + 22} {ty + 54}', stroke=PURPLE,
                 w=1.8, dash='5 4', marker='arrp')
    # legend for the dashed lines
    ly = y0 + 534
    svg.line(cols['in'] - 20, ly - 4, cols['in'] + 30, ly - 4, stroke=PURPLE, w=1.6, dash='5 4', marker='arrp')
    svg.text(cols['in'] + 40, ly, 'delayed synapses', size=13, fill=PURPLE)
    # outputs read every 80 ms -> phonemes -> language model -> words
    ox = cols['out'] + 22
    svg.line(ox, mid, ox + 62, mid, marker='arr', w=1.6)
    svg.text(ox + 31, mid - 12, 'every', size=12, fill=QUIET, anchor='middle')
    svg.text(ox + 31, mid + 22, '80 ms', size=12, fill=QUIET, anchor='middle', weight=700)
    bx = ox + 66
    svg.rect(bx, mid - 44, 170, 88, rx=10, stroke=BLUE, fill='#eef3fb')
    svg.text(bx + 85, mid - 14, 'phoneme', size=14, anchor='middle', weight=700)
    svg.text(bx + 85, mid + 4, 'probabilities', size=14, anchor='middle', weight=700)
    svg.text(bx + 85, mid + 26, '39 + silence + blank', size=12, fill=QUIET, anchor='middle')
    lx = bx + 170 + 46
    svg.line(bx + 172, mid, lx - 4, mid, marker='arr', w=1.6)
    svg.rect(lx, mid - 44, 170, 88, rx=10, stroke=INK, fill='#f4f2ec')
    svg.text(lx + 85, mid - 14, '3-gram', size=14, anchor='middle', weight=700)
    svg.text(lx + 85, mid + 4, 'language model', size=14, anchor='middle', weight=700)
    svg.text(lx + 85, mid + 26, '125,000 words', size=12, fill=QUIET, anchor='middle')
    wx = lx + 170 + 46
    svg.line(lx + 172, mid, wx - 4, mid, marker='arr', w=1.6)
    svg.rect(wx, mid - 44, 150, 88, rx=10, stroke=INK, fill='white')
    svg.text(wx + 75, mid - 4, 'decoded', size=14, anchor='middle', weight=700)
    svg.text(wx + 75, mid + 16, 'sentence', size=14, anchor='middle', weight=700)


def panel_a_final(svg, y0):
    """The final model: 2 recurrent IAF layers, delayed synapses only on the input, outputs reading both layers."""
    svg.text(40, y0, 'A', size=24, weight=700)
    svg.text(72, y0, 'Network topology', size=19, weight=700)
    mid = y0 + 270
    cols = {'in': 140, 'h1': 380, 'h2': 600, 'out': 860}
    hs = {'in': 250, 'h1': 330, 'h2': 330, 'out': 210}
    top = {k: mid - v / 2 for k, v in hs.items()}
    bot = {k: mid + v / 2 for k, v in hs.items()}
    svg.rect(cols['h1'] - 70, y0 + 70, cols['h2'] - cols['h1'] + 160, 465, rx=12, stroke=PURPLE, dash='7 6', sw=1.5)
    svg.text((cols['h1'] + cols['h2']) / 2 + 10, y0 + 58, 'spiking core: 2 × 1,024 recurrent IAF neurons', size=15,
             weight=700, anchor='middle', fill=PURPLE)
    fan(svg, cols['in'], top['in'], bot['in'], cols['h1'], top['h1'], bot['h1'])
    fan(svg, cols['h1'], top['h1'], bot['h1'], cols['h2'], top['h2'], bot['h2'])
    fan(svg, cols['h2'], top['h2'], bot['h2'], cols['out'], top['out'], bot['out'], color=BLUE, opacity=0.12)
    ys_in = capsule(svg, cols['in'], top['in'], hs['in'], 'Input', '256', 'one 20 ms bin per step')
    ys = {k: capsule(svg, cols[k], top[k], hs[k], n, '1,024', 'filters 20 ms–1.3 s')
          for k, n in (('h1', 'Hidden 1'), ('h2', 'Hidden 2'))}
    capsule(svg, cols['out'], top['out'], hs['out'], '41 leaky outputs', '41', 'never spike', stroke=BLUE,
            fill='#eef3fb', circ=BLUE)
    # delayed synapses only on the input: several per input-neuron pair
    delayed_links(svg, cols['in'], ys_in, cols['h1'], ys['h1'], [(0, 1), (0, 4), (2, 0), (3, 5), (5, 2), (4, 3)])
    # recurrent connections within each layer (no delays)
    for k in ('h1', 'h2'):
        cx, ty = cols[k], top[k] + 30
        svg.path(f'M{cx + 20} {ty} C{cx + 62} {ty - 14}, {cx + 62} {ty + 66}, {cx + 22} {ty + 54}', stroke=INK,
                 w=1.8, marker='arr')
        svg.text(cx + 26, ty + 80, 'recurrent', size=11, fill=QUIET)
    # layer 1 also feeds the outputs directly
    x1, y1 = cols['h1'] + 20, bot['h1'] - 22
    x2, y2 = cols['out'] - 24, mid + 48
    dip = bot['h1'] + 92
    svg.path(f'M{x1} {y1} C{x1 + 40} {dip}, {x2 - 40} {dip + 10}, {x2} {y2}', stroke=BLUE, w=2.0, marker='arrb')
    svg.text(cols['h2'] + 150, mid + 236, 'layer 1 also feeds the outputs', size=12, fill=BLUE, weight=700)
    ly = y0 + 556
    svg.line(cols['in'] - 20, ly - 4, cols['in'] + 30, ly - 4, stroke=PURPLE, w=1.6, dash='5 4', marker='arrp')
    svg.text(cols['in'] + 40, ly, 'delayed synapses: 4 per input–neuron pair, 0–600 ms', size=13, fill=PURPLE)
    ox = cols['out'] + 22
    svg.line(ox, mid, ox + 62, mid, marker='arr', w=1.6)
    svg.text(ox + 31, mid - 12, 'every', size=12, fill=QUIET, anchor='middle')
    svg.text(ox + 31, mid + 22, '80 ms', size=12, fill=QUIET, anchor='middle', weight=700)
    bx = ox + 66
    svg.rect(bx, mid - 44, 170, 88, rx=10, stroke=BLUE, fill='#eef3fb')
    svg.text(bx + 85, mid - 14, 'phoneme', size=14, anchor='middle', weight=700)
    svg.text(bx + 85, mid + 4, 'probabilities', size=14, anchor='middle', weight=700)
    svg.text(bx + 85, mid + 26, '39 + silence + blank', size=12, fill=QUIET, anchor='middle')
    lx = bx + 170 + 46
    svg.line(bx + 172, mid, lx - 4, mid, marker='arr', w=1.6)
    svg.rect(lx, mid - 44, 170, 88, rx=10, stroke=INK, fill='#f4f2ec')
    svg.text(lx + 85, mid - 14, '3-gram', size=14, anchor='middle', weight=700)
    svg.text(lx + 85, mid + 4, 'language model', size=14, anchor='middle', weight=700)
    svg.text(lx + 85, mid + 26, '125,000 words', size=12, fill=QUIET, anchor='middle')
    wx = lx + 170 + 46
    svg.line(lx + 172, mid, wx - 4, mid, marker='arr', w=1.6)
    svg.rect(wx, mid - 44, 150, 88, rx=10, stroke=INK, fill='white')
    svg.text(wx + 75, mid - 4, 'decoded', size=14, anchor='middle', weight=700)
    svg.text(wx + 75, mid + 16, 'sentence', size=14, anchor='middle', weight=700)


def simulate(T=40):
    """One IAF neuron (no leak, soft reset, at most 3 spikes per step) driven by two synapses: synapse 1 delivers
    immediately, synapse 2 the same presynaptic spikes 4 steps (80 ms) later."""
    pre = [0] * T
    for t in (2, 3, 9, 10, 11, 20, 28, 29):
        pre[t] = 1
    d = 4
    arr2 = [pre[t - d] if t >= d else 0 for t in range(T)]
    a = math.exp(-1 / 3.0)                 # synaptic time constant: 3 steps = 60 ms
    i_syn, v, cur, vm, out = 0.0, 0.0, [], [], []
    w1, w2, theta = 0.55, 0.45, 1.0
    for t in range(T):
        x = w1 * pre[t] + w2 * arr2[t]
        i_syn = a * i_syn + (1 - a) * x
        v += i_syn
        n = min(3, int(v // theta)) if v >= theta else 0
        vm.append((v, n))
        v -= n * theta
        cur.append(i_syn)
        out.append(n)
    return pre, arr2, cur, vm, out, d


def panel_b(svg, y0):
    svg.text(40, y0, 'B', size=24, weight=700)
    svg.text(72, y0, 'Signal cascade through a single neuron (one step per 20 ms bin)', size=19, weight=700)
    pre, arr2, cur, vm, out, d = simulate()
    T = len(pre)
    x0, x1 = 300, 880
    X = lambda t: x0 + (x1 - x0) * t / T
    rows = [('1', 'Spikes sent', None), ('2', 'Arrival through a', 'synapse with delay d'), ('3', 'Synaptic current', 'each arriving spike adds'),
            ('4', 'Membrane potential', 'integrate-and-fire, no leak'), ('5', 'Output spikes', None)]
    ys = [y0 + 70, y0 + 150, y0 + 240, y0 + 350, y0 + 440]
    for (num, lab, sub), y in zip(rows, ys):
        svg.circle(58, y - 5, 13, stroke=INK, fill=INK)
        svg.text(58, y, num, size=13, weight=700, fill='white', anchor='middle')
        svg.text(80, y, lab, size=15, weight=700)
        if sub:
            svg.text(80, y + 18, sub, size=12, fill=QUIET)
        svg.line(x0, y + 8, x1, y + 8, stroke=GRID, w=1.2)
    # 1 spikes sent / 2 delayed arrival
    for t in range(T):
        if pre[t]:
            svg.line(X(t), ys[0] + 8, X(t), ys[0] - 20, w=2.6)
        if arr2[t]:
            svg.line(X(t), ys[1] + 8, X(t), ys[1] - 20, stroke=PURPLE, w=2.6)
    t0 = 2
    svg.line(X(t0), ys[0] + 14, X(t0), ys[1] - 26, stroke=GRID, w=1, dash='3 3')
    svg.line(X(t0) + 2, ys[1] - 28, X(t0 + d) - 2, ys[1] - 28, stroke=PURPLE, w=1.4, marker='arrp')
    svg.text((X(t0) + X(t0 + d)) / 2, ys[1] - 34, f'd = {d * 20} ms', size=12, fill=PURPLE, anchor='middle',
             italic=True)
    svg.text(x1, ys[1] + 30, 'the neuron also receives the undelayed copy (row 1) through another synapse', size=11,
             fill=QUIET, anchor='end')
    # 3 synaptic current (step-held per 20 ms bin)
    cmax = max(cur) or 1
    pts = []
    for t in range(T):
        yv = ys[2] + 8 - 50 * cur[t] / cmax
        pts.append(f'{X(t):.1f} {yv:.1f} L{X(t + 1):.1f} {yv:.1f}')
    svg.path('M' + ' L'.join(pts), stroke=PURPLE, w=2.2)
    svg.text(80, ys[2] + 34, 'current, which then fades', size=12, fill=QUIET)
    # 4 membrane potential with threshold and soft resets
    th_y = ys[3] - 52
    vmax = 1.0
    base_y = ys[3] + 8
    Y = lambda v: base_y - (base_y - th_y) * min(v, 1.6) / vmax
    svg.line(x0, th_y, x1, th_y, stroke=RED, w=1.6, dash='6 5')
    svg.text(x1 + 10, th_y + 5, 'θ', size=16, fill=RED, italic=True)
    d_path, prev_after = [], 0.0
    first_reset = None
    for t in range(T):
        v_pre, n = vm[t]
        d_path.append(f'{X(t):.1f} {Y(prev_after):.1f} L{X(t):.1f} {Y(v_pre):.1f}')
        after = v_pre - n
        if n:
            svg.line(X(t + 0.5), Y(v_pre), X(t + 0.5), Y(after), stroke=INK, w=1.2, dash='2 2')
            if first_reset is None:
                first_reset = (X(t + 0.5), Y(v_pre))
            svg.line(X(t + 0.5), ys[3] + 12, X(t + 0.5), ys[4] - 22, stroke=GRID, w=1, dash='3 3')
        d_path.append(f'{X(t + 1):.1f} {Y(v_pre if not n else after):.1f}')
        prev_after = after
    svg.path('M' + ' L'.join(d_path), stroke=INK, w=2.0)
    if first_reset:
        svg.text(first_reset[0] - 6, th_y - 16, 'soft reset: V ← V − nθ (remainder kept)', size=12,
                 fill=QUIET)
    # 5 output spikes (count per step, at most 3)
    for t in range(T):
        if out[t]:
            svg.line(X(t + 0.5), ys[4] + 8, X(t + 0.5), ys[4] + 8 - 9 * out[t] - 4, w=2.6)
            if out[t] > 1:
                svg.text(X(t + 0.5), ys[4] - 30, f'×{out[t]}', size=11, fill=QUIET, anchor='middle')
    # time axis
    ya = ys[4] + 40
    svg.line(x0, ya, x1 + 10, ya, w=1.5, marker='arr')
    svg.text(x1 + 10, ya + 22, 't (one step = one 20 ms bin)', size=12, fill=QUIET, anchor='end')
    for t in range(0, T + 1, 4):
        svg.line(X(t), ya - 4, X(t), ya + 4, stroke=QUIET, w=1)
    svg.text(x0, ya + 22, 'output read every 4 steps (80 ms)', size=12, fill=BLUE)


def panel_c(svg, y0):
    x0 = 1010
    svg.text(x0 - 30, y0, 'C', size=24, weight=700)
    svg.text(x0 + 2, y0, 'Delayed input synapses', size=19, weight=700)
    svg.text(x0 + 2, y0 + 24, 'memory without a window', size=14, fill=QUIET)
    # one input channel over the last 30 bins (600 ms)
    bx0, bx1, by = x0 + 10, x0 + 560, y0 + 110
    nb = 30
    BX = lambda b: bx0 + (bx1 - bx0) * b / nb
    for b in range(nb):
        svg.rect(BX(b) + 1, by - 14, (bx1 - bx0) / nb - 2, 28, rx=2, stroke=GRID, fill='white', sw=1)
    svg.text(bx0, by - 44, '600 ms ago', size=11, fill=QUIET)
    svg.text(bx1, by - 44, 'now (bin t)', size=11, fill=QUIET, anchor='end')
    svg.text(bx0 - 12, by + 5, 'input', size=12, fill=INK, anchor='end')
    svg.text(bx0 - 12, by + 20, 'channel', size=12, fill=INK, anchor='end')
    # the neuron and its 4 synapses from this channel, each with its own delay
    nx, ny = (bx0 + bx1) / 2 + 120, y0 + 330
    svg.circle(nx, ny, 26, stroke=INK, fill=FILL, sw=1.8)
    svg.text(nx, ny + 5, 'j', size=16, weight=700, anchor='middle', italic=True)
    svg.text(nx + 40, ny + 5, 'a layer-1 neuron', size=13, fill=INK)
    taps = [(29, 'w₁, 0 ms'), (25, 'w₂, 80 ms'), (16, 'w₃, 260 ms'), (4, 'w₄, 500 ms')]
    for b, lab in taps:
        cx = BX(b) + (bx1 - bx0) / nb / 2
        svg.rect(BX(b) + 1, by - 14, (bx1 - bx0) / nb - 2, 28, rx=2, stroke=PURPLE, fill='#efeafb', sw=1.8)
        svg.path(f'M{cx:.1f} {by + 16} C{cx:.1f} {by + 120}, {nx:.1f} {ny - 120}, {nx:.1f} {ny - 28}',
                 stroke=PURPLE, w=1.6, marker='arrp', opacity=0.9)
        svg.text(cx, by - 22, lab, size=11, fill=PURPLE, anchor='middle', weight=700)
    lines = ['Each input–neuron pair has 4 synapses, each reaching',
             'a different point back in time (0–600 ms).']
    for i, s in enumerate(lines):
        svg.text(x0 + 2, ny + 80 + 20 * i, s, size=14, fill=INK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='model_final_figure.svg')
    p.add_argument('--design', default='final', choices=['final', 'modelA'],
                   help='final: 2 layers, delays on the input only, outputs reading both layers; modelA: 3 layers')
    a = p.parse_args()
    svg = Svg(1600, 1210)
    (panel_a_final if a.design == 'final' else panel_a)(svg, 46)
    svg.line(40, 615, 1560, 615, stroke=GRID, w=1.2)
    panel_b(svg, 640)
    svg.line(965, 625, 965, 1180, stroke=GRID, w=1.2)
    panel_c(svg, 640)
    svg.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
