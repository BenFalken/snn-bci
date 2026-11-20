"""
Pictograms for the task figure, in a flat illustration style: a seated macaque and a person in side profile (facing
right), and lateral views of the macaque and human left hemisphere (front to the right).

Every shape is a smooth closed curve through hand-placed points (Catmull-Rom through cubic Beziers), drawn in a local
frame scaled by s and placed at (x, y). Only silhouettes carry a thin outline; parts inside a silhouette are flat
colour, so nothing reads as drawn on top.

Editable output: every part gets an SVG id (matplotlib's gid), "<name>_<part>", e.g. monkey1_head, monkey1_eye,
person_hair, brain_speech_cortex. In the figure's .svg each part is its own group, so it can be selected, recoloured,
moved or replaced in Illustrator, Inkscape or Figma.
"""
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.path import Path as MPath

LINE = "#4a3a2e"
FUR, FUR_SHADE, FUR_BELLY = "#9a6a3f", "#835833", "#a8794d"
FACE, FACE_SHADE, EAR, EAR_IN, HAND = "#f3d6b3", "#e9c39c", "#e7bf98", "#d29b76", "#e2b58e"
SKIN, SKIN_SHADE, HAIR, HAIR_LIGHT = "#f2d8c4", "#e3bea5", "#3b2c24", "#5a4436"
SHIRT, SHIRT_SHADE, LIP = "#8ea2c6", "#7a8fb4", "#c98a7c"
BRAIN, CEREB, SULCUS, BRAIN_LINE = "#f2e1df", "#e8d0cd", "#b48f8c", "#5a4a48"


def _bezier(pts, closed=True):
    p = np.asarray(pts, float)
    n = len(p)
    verts, codes = [p[0]], [MPath.MOVETO]
    for i in (range(n) if closed else range(n - 1)):
        p0 = p[i - 1] if (closed or i > 0) else p[i]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if (closed or i + 2 < n) else p2
        verts += [p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2]
        codes += [MPath.CURVE4] * 3
    return MPath(verts, codes)


class Frame:
    """Local drawing frame: point (u, v) -> (x + s*u, y + s*v). Every patch gets the SVG id prefix_part."""

    def __init__(self, ax, x, y, s, z=2, prefix="part", alpha=1.0):
        self.ax, self.x, self.y, self.s, self.z, self.prefix, self.alpha = ax, x, y, s, z, prefix, alpha

    def P(self, pts):
        return [(self.x + u * self.s, self.y + v * self.s) for u, v in pts]

    def _add(self, patch, part):
        patch.set_gid(f"{self.prefix}_{part}")
        self.ax.add_patch(patch)
        return patch

    def shape(self, part, pts, fc, ec="none", lw=0.4, dz=0, closed=True, alpha=None):
        return self._add(PathPatch(_bezier(self.P(pts), closed), fc=fc if closed else "none", ec=ec, lw=lw,
                                   zorder=self.z + dz, capstyle="round", joinstyle="round",
                                   alpha=self.alpha if alpha is None else alpha), part)

    def line(self, part, pts, color, lw=0.5, dz=0, alpha=None):
        return self.shape(part, pts, "none", ec=color, lw=lw, dz=dz, closed=False, alpha=alpha)

    def ellipse(self, part, c, w, h, fc, ec="none", lw=0.4, dz=0, angle=0, alpha=None):
        (cx, cy), = self.P([c])
        return self._add(Ellipse((cx, cy), w * self.s, h * self.s, angle=angle, fc=fc, ec=ec, lw=lw,
                                 zorder=self.z + dz, alpha=self.alpha if alpha is None else alpha), part)


def _limb_outline(joints, widths):
    J = np.asarray(joints, float)
    left, right = [], []
    for i, (p, w) in enumerate(zip(J, widths)):
        d = J[min(i + 1, len(J) - 1)] - J[max(i - 1, 0)]
        nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
        left.append(p + w * nrm)
        right.append(p - w * nrm)
    return [tuple(q) for q in left + right[::-1]]


# ------------------------------------------------------------------------------------------------------------ macaque
# The macaque follows the user's mockup (2026-11-19): no outlines, flat warm tans, thin darker accent strokes only at
# the face's edge, cheek, jaw and brow, a spiky crest swept back, a large glossy eye with two highlights, a concave
# forehead-to-snout profile with a mouth notch, a half-round ear cut by a straight edge with a dark lower rim, and
# partly angular forms. Coordinates are in the mockup's pixel space (x right, y down), mapped to the local frame.
M_FUR, M_FACE, M_EAR, M_ACCENT, M_RIM, M_FUR_DARK = "#d09a7b", "#e3c2ab", "#d9aa8f", "#bf8a6b", "#4a3a30", "#bb8565"
_K = 1.55 / 1100                                       # mockup pixels -> local units


def _px(pts, dx=0.0, dy=0.0):
    return [((u - 300 + dx) * _K, (420 - v - dy) * _K) for u, v in pts]


def monkey(ax, x, y, s=1.0, reach_to=None, z=2, name="monkey"):
    """A slender seated macaque in side profile, facing right, in the mockup's flat style. reach_to: the fingertip
    in data coordinates (the near arm reaching forward), or None (the near arm resting on the knee)."""
    from matplotlib.patches import Polygon
    f = Frame(ax, x, y, s, z, prefix=name)

    def poly(part, pts, fc, dz):
        return f._add(Polygon(f.P(pts), closed=True, fc=fc, ec="none", zorder=z + dz), part)

    acc = lambda part, pts, lw=0.9, color=M_ACCENT, dz=3.0: f.line(part, _px(pts), color, lw=lw * s, dz=dz)
    # tail, far leg, body, near leg, foot
    f.line("tail", _px([(110, 1060), (0, 1112), (-110, 1100), (-185, 1035), (-205, 950)]), M_FUR, lw=3.2 * s, dz=0)
    f.shape("leg_far", _px([(150, 960), (330, 930), (470, 960), (500, 1040), (420, 1100), (160, 1105)]),
            M_FUR_DARK, dz=0.5)
    poly("body", _px([(150, 545), (185, 515), (430, 580), (452, 690), (418, 830), (405, 960), (350, 1062), (170, 1092),
                      (108, 1010), (112, 880), (148, 720)]), M_FUR, dz=1)
    acc("chest", [(448, 700), (424, 820), (410, 940)], lw=0.8, dz=1.2)
    f.shape("leg_near", _px([(95, 975), (260, 905), (455, 890), (525, 935), (515, 1005), (330, 1055),
                             (120, 1090)]), M_FUR, dz=1.5)
    acc("thigh", [(215, 945), (345, 905), (470, 900), (518, 942), (510, 995)], lw=0.8, dz=1.6)
    f.shape("foot", _px([(430, 1060), (560, 1062), (650, 1090), (640, 1118), (440, 1118)]), M_FACE, dz=1.7)
    # head: fur with the crest, then the face, ear, eye and accents (the mockup's outline, traced)
    poly("head", _px([(165, 553), (185, 483), (160, 445), (85, 355), (72, 290), (90, 230), (130, 170), (160, 135),
                      (215, 58), (195, 112), (290, 32), (258, 82), (370, 25), (330, 68), (478, 28), (420, 62),
                      (552, 47), (480, 72), (605, 82), (510, 100), (540, 130), (575, 220), (620, 300), (625, 350),
                      (620, 470), (595, 503), (540, 500), (495, 515), (462, 545), (447, 580), (440, 625),
                      (300, 590)]), M_FUR, dz=2.5)
    f.shape("face", _px([(380, 225), (400, 170), (440, 125), (490, 100), (530, 115), (540, 135), (545, 190),
                         (570, 250), (610, 285), (655, 300), (662, 340), (655, 372), (632, 395), (620, 430),
                         (622, 470), (605, 498), (560, 500), (510, 490), (475, 465), (458, 425), (462, 385),
                         (480, 352), (440, 342), (400, 310), (383, 265)]), M_FACE, dz=2.6)
    acc("face_edge", [(378, 228), (382, 282), (412, 324), (472, 352)])
    acc("cheek", [(476, 358), (456, 405), (471, 463)])
    acc("jaw", [(442, 598), (462, 545), (520, 506), (592, 500)])
    acc("brow", [(488, 146), (456, 152), (441, 182), (437, 236)], lw=0.8)
    th = np.radians(np.linspace(78, 312, 30))
    ear = [(245 + 62 * np.cos(t), 340 - 62 * np.sin(t)) for t in th]
    poly("ear", _px(ear), M_EAR, dz=2.7)
    rim = [(245 + 63 * np.cos(t), 340 - 63 * np.sin(t)) for t in np.radians(np.linspace(178, 305, 16))]
    f.line("ear_rim", _px(rim), M_RIM, lw=0.6 * s, dz=2.75)
    (ex, ey), = _px([(486, 212)])
    f.ellipse("eye", (ex, ey), 80 * _K, 118 * _K, "black", dz=2.8)
    (hx1, hy1), (hx2, hy2) = _px([(501, 190), (474, 235)])
    f.ellipse("eye_highlight", (hx1, hy1), 34 * _K, 36 * _K, "white", dz=2.85)
    f.ellipse("eye_highlight_small", (hx2, hy2), 17 * _K, 18 * _K, "white", dz=2.85)
    # the near arm: slender, angular, with a light hand
    sh = np.array(_px([(330, 690)])[0])
    if reach_to is None:
        joints = [sh, np.array(_px([(400, 845)])[0]), np.array(_px([(520, 925)])[0])]
        hand_dir = np.array([1.0, -0.45])
    else:
        tip = (np.asarray(reach_to, float) - (x, y)) / s
        v = tip - sh
        L = np.hypot(*v)
        u = v / L
        nrm = np.array([u[1], -u[0]])
        wrist = sh + u * (L - 0.085)
        elbow = sh + u * (L - 0.085) * 0.5 + nrm * 0.05
        joints, hand_dir = [sh, elbow, wrist], u
    arm = _limb_outline(joints, [0.045, 0.034, 0.026])
    poly("arm", arm, M_FUR, dz=4)
    J = np.asarray(joints)
    edge = arm[:len(joints)]                                  # the arm's upper/back edge, as in the mockup's accents
    f.line("arm_edge", [tuple(np.asarray(edge[0]) * 0.6 + np.asarray(edge[1]) * 0.4)] + edge[1:], M_ACCENT,
           lw=0.8 * s, dz=4.1)
    w = J[-1]
    hd = hand_dir / np.hypot(*hand_dir)
    hn = np.array([-hd[1], hd[0]])
    hand = [w + 0.028 * hn, w + 0.045 * hd + 0.03 * hn, w + 0.06 * hd + 0.012 * hn, w + 0.09 * hd + 0.008 * hn,
            w + 0.092 * hd - 0.006 * hn, w + 0.05 * hd - 0.024 * hn, w - 0.026 * hn]
    f.shape("hand", [tuple(p) for p in hand], M_FACE, dz=4.5)


# ------------------------------------------------------------------------------------------------------------- person
def person(ax, x, y, s=1.0, z=2, name="person", alpha=0.92, brain_site=None, site_color=None):
    """A slender person in side profile, facing right, head and shoulders, drawn slightly translucent. With
    brain_site (e.g. '6v'), a faint brain shows through the head with that site marked in site_color."""
    f = Frame(ax, x, y, s, z, prefix=name, alpha=alpha)
    f.shape("torso", [(-0.30, -0.95), (-0.30, -0.40), (-0.21, -0.10), (-0.07, -0.01), (0.07, -0.02),
                      (0.18, -0.12), (0.24, -0.40), (0.25, -0.95)], SHIRT, ec=LINE, dz=1)
    f.shape("shirt_shade", [(-0.30, -0.95), (-0.30, -0.42), (-0.24, -0.20), (-0.18, -0.45), (-0.17, -0.95)],
            SHIRT_SHADE, dz=1.1)
    f.shape("collar", [(-0.09, 0.00), (0.00, -0.09), (0.08, -0.01), (0.00, -0.04)], SHIRT_SHADE, dz=1.2)
    f.shape("neck", [(-0.05, -0.03), (-0.06, 0.13), (0.05, 0.13), (0.045, -0.04)], SKIN_SHADE, ec=LINE, dz=1.5)
    head = [(-0.14, 0.10), (-0.23, 0.24), (-0.255, 0.42), (-0.20, 0.60), (-0.05, 0.69), (0.11, 0.67),
            (0.20, 0.58), (0.215, 0.49), (0.212, 0.455), (0.252, 0.385), (0.246, 0.362), (0.212, 0.352),
            (0.222, 0.322), (0.212, 0.300), (0.219, 0.280), (0.204, 0.235), (0.180, 0.195), (0.12, 0.165),
            (0.05, 0.155), (-0.02, 0.12)]
    f.shape("head", head, SKIN, ec=LINE, dz=2)
    if brain_site is not None:
        site = brain(ax, x + 0.0 * s, y + 0.47 * s, 0.155 * s, human=True, z=z + 2.1, name=f"{name}_brain",
                     alpha=0.45, folds=False)(brain_site)
        if site_color is not None:
            sq = 0.022 * s
            r = Rectangle((site[0] - sq, site[1] - sq), 2 * sq, 2 * sq, fc=site_color, ec="white", lw=0.4,
                          zorder=z + 2.6)
            r.set_gid(f"{name}_array")
            ax.add_patch(r)
    f.shape("hair", [(-0.262, 0.36), (-0.25, 0.55), (-0.15, 0.69), (0.02, 0.755), (0.16, 0.72), (0.25, 0.62),
                     (0.235, 0.55), (0.16, 0.585), (0.07, 0.57), (-0.02, 0.60), (-0.10, 0.54), (-0.16, 0.44),
                     (-0.20, 0.30), (-0.245, 0.26)], HAIR, dz=2.5, alpha=1.0)
    f.shape("hair_light", [(-0.12, 0.66), (0.02, 0.72), (0.14, 0.69), (0.04, 0.685), (-0.06, 0.64)], HAIR_LIGHT,
            dz=2.55, alpha=1.0)
    f.ellipse("ear", (-0.055, 0.37), 0.075, 0.115, SKIN_SHADE, ec=LINE, lw=0.35, dz=2.6, alpha=1.0)
    f.line("brow", [(0.140, 0.468), (0.172, 0.478), (0.198, 0.466)], HAIR, lw=0.7 * s, dz=2.8, alpha=1.0)
    f.ellipse("eye", (0.172, 0.432), 0.026, 0.030, "#2a2a2a", dz=2.8, alpha=1.0)
    f.ellipse("eye_highlight", (0.177, 0.438), 0.009, 0.009, "white", dz=2.85, alpha=1.0)
    f.line("lips", [(0.198, 0.262), (0.214, 0.266)], LIP, lw=0.7, dz=2.8, alpha=1.0)


# -------------------------------------------------------------------------------------------------------------- brain
def _wiggle(pts, amp, n=40, waves=3.0, seed=0):
    p = np.asarray(pts, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, seg[-1], n)
    q = np.c_[np.interp(t, seg, p[:, 0]), np.interp(t, seg, p[:, 1])]
    d = np.gradient(q, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]] / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9)
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    return q + nrm * (amp * np.sin(2 * np.pi * waves * t / seg[-1] + phase))[:, None]


def brain(ax, x, y, s=1.0, human=False, z=2, name="brain", alpha=1.0, folds=True):
    """A lateral view of the left hemisphere, front to the right: cortex, main sulci, cerebellum, brainstem.
    Returns a function mapping a site name ('M1', 'aSTG', '6v') to data coordinates."""
    f = Frame(ax, x, y, s, z, prefix=name, alpha=alpha)
    if human:
        cortex = [(-0.95, 0.02), (-0.84, 0.46), (-0.48, 0.78), (0.05, 0.88), (0.55, 0.74), (0.86, 0.40),
                  (0.93, 0.08), (0.78, -0.12), (0.55, -0.18), (0.48, -0.40), (0.10, -0.55), (-0.40, -0.47),
                  (-0.80, -0.28)]
        cereb = [(-0.82, -0.32), (-0.55, -0.30), (-0.25, -0.46), (-0.32, -0.66), (-0.62, -0.68), (-0.85, -0.52)]
        stem = [(-0.18, -0.46), (0.02, -0.48), (0.00, -0.92), (-0.14, -0.92)]
    else:
        cortex = [(-0.95, 0.02), (-0.72, 0.40), (-0.20, 0.60), (0.40, 0.52), (0.80, 0.30), (0.95, 0.10),
                  (0.86, -0.04), (0.58, -0.10), (0.46, -0.30), (0.10, -0.46), (-0.38, -0.44), (-0.80, -0.26)]
        cereb = [(-0.80, -0.26), (-0.50, -0.28), (-0.26, -0.42), (-0.34, -0.60), (-0.62, -0.62), (-0.82, -0.46)]
        stem = [(-0.20, -0.42), (-0.02, -0.44), (-0.05, -0.82), (-0.18, -0.82)]
    lw = 0.55 if folds else 0.4
    f.shape("brainstem", stem, CEREB, ec=BRAIN_LINE, lw=lw, dz=0)
    f.shape("cerebellum", cereb, CEREB, ec=BRAIN_LINE, lw=lw, dz=0.5)
    c = np.asarray(cereb)
    (x0, y0), (x1, y1) = c.min(0), c.max(0)
    if folds:
        for i, k in enumerate((0.3, 0.5, 0.7)):
            yy = y1 - k * (y1 - y0)
            f.line(f"folium{i}", [(x0 + 0.12 * (x1 - x0), yy + 0.02), (x0 + 0.5 * (x1 - x0), yy - 0.035),
                                  (x0 + 0.86 * (x1 - x0), yy - 0.005)], SULCUS, lw=0.45, dz=0.6)
    f.shape("cortex", cortex, BRAIN, ec=BRAIN_LINE, lw=lw + 0.1, dz=1)
    count = [0]

    def sul(label, pts, amp, lw_=0.8, w=3.0):
        count[0] += 1
        f.line(f"sulcus_{label}", [tuple(q) for q in _wiggle(pts, amp, waves=w, seed=count[0])], SULCUS,
               lw=lw_ * (1 if folds else 0.6), dz=1.2)

    if human:
        sul("central", [(0.00, 0.87), (0.05, 0.50), (0.15, 0.12)], 0.012, 0.9, 3)
        sul("lateral", [(0.50, -0.14), (0.15, 0.00), (-0.30, 0.16)], 0.006, 1.0, 2)
        if folds:
            sul("precentral", [(0.20, 0.82), (0.25, 0.48), (0.33, 0.12)], 0.010, 0.7, 3)
            sul("postcentral", [(-0.15, 0.85), (-0.10, 0.50), (0.00, 0.14)], 0.010, 0.7, 3)
            sul("superior_temporal", [(0.40, -0.28), (0.0, -0.16), (-0.42, 0.0)], 0.010, 0.75, 3)
            sul("superior_frontal", [(0.27, 0.62), (0.55, 0.58), (0.80, 0.40)], 0.010, 0.65, 2)
            sul("inferior_frontal", [(0.33, 0.30), (0.58, 0.27), (0.84, 0.14)], 0.010, 0.65, 2)
            sul("intraparietal", [(-0.13, 0.50), (-0.42, 0.46), (-0.72, 0.26)], 0.010, 0.65, 2)
        sites = {"6v": (0.30, 0.12), "M1": (0.13, 0.60)}
    else:
        sul("central", [(-0.08, 0.59), (-0.01, 0.32), (0.10, 0.02)], 0.010, 0.9, 2)
        sul("lateral", [(0.50, -0.12), (0.10, 0.00), (-0.32, 0.14)], 0.006, 1.0, 2)
        sul("superior_temporal", [(0.40, -0.24), (0.0, -0.14), (-0.45, 0.04), (-0.55, 0.16)], 0.010, 0.75, 2)
        sul("arcuate", [(0.32, 0.48), (0.40, 0.24), (0.32, 0.02)], 0.006, 0.75, 1)
        sul("principal", [(0.50, 0.24), (0.70, 0.20), (0.86, 0.14)], 0.006, 0.75, 1)
        sul("intraparietal", [(-0.14, 0.40), (-0.40, 0.30), (-0.58, 0.12)], 0.008, 0.65, 2)
        sul("lunate", [(-0.66, 0.36), (-0.72, 0.10), (-0.70, -0.14)], 0.008, 0.65, 2)
        sites = {"M1": (0.08, 0.40), "aSTG": (0.30, -0.13)}
    return lambda site: f.P([sites[site]])[0]


# The macaque is now constructed from anatomy (macaque.py: proportions -> skeleton -> volumes -> silhouette, in the
# mockup's flat style); it replaces the hand-placed monkey above.
from macaque import monkey  # noqa: E402,F401,F811
