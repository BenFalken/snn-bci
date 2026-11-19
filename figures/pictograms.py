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
def monkey(ax, x, y, s=1.0, reach_to=None, z=2, name="monkey"):
    """A slender seated macaque in side profile, facing right. reach_to: the fingertip in data coordinates (the
    near arm reaching forward), or None (the near arm resting on the knee)."""
    f = Frame(ax, x, y, s, z, prefix=name)
    tail = [(-0.30, -0.86), (-0.48, -0.96), (-0.64, -0.92), (-0.70, -0.84)]
    f.line("tail", tail, LINE, lw=1.9 * s, dz=0)
    f.line("tail_fill", tail, FUR_SHADE, lw=1.3 * s, dz=0.05)
    f.shape("leg_far", [(-0.10, -0.62), (0.18, -0.58), (0.34, -0.80), (0.28, -0.92), (-0.12, -0.92)], FUR_SHADE,
            ec=LINE, dz=0.5)
    f.shape("body", [(-0.30, -0.90), (-0.33, -0.60), (-0.26, -0.30), (-0.10, -0.06), (0.06, 0.04), (0.18, -0.02),
                     (0.17, -0.26), (0.10, -0.55), (0.10, -0.86), (-0.06, -0.93)], FUR, ec=LINE, dz=1)
    f.shape("belly", [(0.02, -0.20), (0.13, -0.24), (0.09, -0.52), (0.04, -0.70), (-0.02, -0.50)], FUR_BELLY, dz=1.1)
    f.shape("leg_near", [(-0.20, -0.66), (0.02, -0.52), (0.26, -0.52), (0.36, -0.62), (0.32, -0.80),
                         (0.20, -0.90), (-0.18, -0.90)], FUR, ec=LINE, dz=1.5)
    f.shape("foot", [(0.16, -0.86), (0.36, -0.84), (0.47, -0.89), (0.40, -0.935), (0.16, -0.935)], HAND,
            ec=LINE, dz=1.6)
    f.shape("neck", [(-0.04, -0.04), (0.04, 0.10), (0.20, 0.13), (0.30, 0.06), (0.26, -0.04), (0.14, -0.08)],
            FUR, ec=LINE, dz=2.4)
    hx, hy = 0.21, 0.19                                                   # head: round skull, flat face, large ear
    H = lambda pts: [(u + hx, v + hy) for u, v in pts]
    f.shape("head", H([(-0.20, 0.02), (-0.16, 0.16), (-0.04, 0.23), (0.10, 0.21), (0.20, 0.12), (0.22, -0.02),
                       (0.15, -0.14), (0.02, -0.19), (-0.12, -0.15)]), FUR, ec=LINE, dz=2.5)
    f.shape("face", H([(0.02, 0.13), (0.09, 0.16), (0.16, 0.13), (0.205, 0.05), (0.215, -0.04), (0.18, -0.11),
                       (0.10, -0.14), (0.04, -0.10), (0.00, -0.01)]), FACE, dz=2.6)
    f.shape("face_shade", H([(0.02, -0.02), (0.08, -0.06), (0.15, -0.06), (0.19, -0.09), (0.12, -0.135),
                             (0.05, -0.11)]), FACE_SHADE, dz=2.65)
    f.ellipse("ear", (hx - 0.10, hy + 0.01), 0.10, 0.13, EAR, ec=LINE, dz=2.7)
    f.ellipse("ear_inner", (hx - 0.095, hy + 0.01), 0.05, 0.075, EAR_IN, dz=2.75)
    f.ellipse("eye", (hx + 0.115, hy + 0.035), 0.050, 0.056, "#2a1d16", dz=2.8)
    f.ellipse("eye_highlight", (hx + 0.125, hy + 0.05), 0.016, 0.016, "white", dz=2.85)
    f.line("brow", H([(0.075, 0.085), (0.115, 0.10), (0.155, 0.085)]), FACE_SHADE, lw=0.8 * s, dz=2.82)
    f.ellipse("nose", (hx + 0.195, hy - 0.035), 0.018, 0.013, "#8a5d45", dz=2.85)
    f.line("mouth", H([(0.19, -0.085), (0.165, -0.095), (0.14, -0.088)]), "#a77558", lw=0.5, dz=2.85)
    sh = np.array([0.08, -0.10])                                          # the near arm, slender, small hand
    if reach_to is None:
        joints = [sh, sh + (0.10, -0.24), sh + (0.24, -0.40)]
        hand_dir = np.array([1.0, -0.35])
    else:
        tip = (np.asarray(reach_to, float) - (x, y)) / s
        v = tip - sh
        L = np.hypot(*v)
        u = v / L
        nrm = np.array([u[1], -u[0]])
        wrist = sh + u * (L - 0.10)
        elbow = sh + u * (L - 0.10) * 0.5 + nrm * 0.07
        joints, hand_dir = [sh, elbow, wrist], u
    f.shape("arm", _limb_outline(joints, [0.055, 0.042, 0.030]), FUR, ec=LINE, dz=4)
    w = np.asarray(joints[-1], float)
    hd = hand_dir / np.hypot(*hand_dir)
    hn = np.array([-hd[1], hd[0]])
    hand = [w + 0.032 * hn, w + 0.05 * hd + 0.034 * hn, w + 0.075 * hd + 0.014 * hn, w + 0.105 * hd + 0.008 * hn,
            w + 0.108 * hd - 0.006 * hn, w + 0.06 * hd - 0.028 * hn, w - 0.028 * hn]
    f.shape("hand", [tuple(p) for p in hand], HAND, ec=LINE, dz=4.5)


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
