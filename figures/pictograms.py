"""
Pictograms for the task figure: a seated macaque and a person in side profile (facing right), and lateral views of
the macaque and human left hemisphere (front to the right) with their main sulci, cerebellum and brainstem.
Every shape is a smooth closed curve through hand-placed points (Catmull-Rom through cubic Beziers), in a local frame
scaled by s and placed at (x, y).
"""
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Polygon
from matplotlib.path import Path as MPath

INK = "#1a1a1a"
FUR, FUR_DARK, FUR_LIGHT = "#a8784e", "#8a5f3b", "#bf9268"
FACE, FACE_DARK = "#ecd0b4", "#d9ad8c"
SKIN, SKIN_DARK, HAIR, SHIRT, SHIRT_DARK = "#f1d3bd", "#ddb397", "#4e3b2c", "#7a8fb0", "#6a7ea0"
BRAIN, BRAIN_SHADE, SULCUS, CEREB = "#f2e2e0", "#e6cfcc", "#b48f8c", "#e9d3d0"


def _bezier(pts, closed=True):
    p = np.asarray(pts, float)
    n = len(p)
    verts, codes = [p[0]], [MPath.MOVETO]
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = p[i - 1] if (closed or i > 0) else p[i]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if (closed or i + 2 < n) else p2
        verts += [p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2]
        codes += [MPath.CURVE4] * 3
    return MPath(verts, codes)


class Frame:
    """Local drawing frame: point (u, v) -> (x + s*u, y + s*v), mirrored when flip."""
    def __init__(self, ax, x, y, s, z=2):
        self.ax, self.x, self.y, self.s, self.z = ax, x, y, s, z

    def P(self, pts):
        return [(self.x + u * self.s, self.y + v * self.s) for u, v in pts]

    def shape(self, pts, fc, ec=INK, lw=0.6, dz=0, closed=True):
        self.ax.add_patch(PathPatch(_bezier(self.P(pts), closed), fc=fc if closed else "none", ec=ec, lw=lw,
                                    zorder=self.z + dz, capstyle="round", joinstyle="round"))

    def line(self, pts, color=INK, lw=0.6, dz=0):
        self.shape(pts, "none", ec=color, lw=lw, dz=dz, closed=False)

    def ellipse(self, c, w, h, fc, ec="none", lw=0.5, dz=0, angle=0):
        (cx, cy), = self.P([c])
        self.ax.add_patch(Ellipse((cx, cy), w * self.s, h * self.s, angle=angle, fc=fc, ec=ec, lw=lw,
                                  zorder=self.z + dz))


def _limb(f, joints, widths, fc, dz):
    """A tapered limb along joints (shoulder, elbow, wrist) with the given half-widths, as one smooth outline."""
    J = np.asarray(joints, float)
    left, right = [], []
    for i, (p, w) in enumerate(zip(J, widths)):
        d = (J[min(i + 1, len(J) - 1)] - J[max(i - 1, 0)])
        nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
        left.append(p + w * nrm); right.append(p - w * nrm)
    f.shape(left + right[::-1], fc, dz=dz)


def monkey(ax, x, y, s=1.0, reach_to=None, z=2):
    """A seated macaque in side profile, facing right. reach_to: the fingertip in data coordinates (an arm
    reaching forward), or None (the near arm resting on the lap)."""
    f = Frame(ax, x, y, s, z)
    f.line([(-0.38, -0.86), (-0.55, -0.95), (-0.72, -0.9)], color=INK, lw=2.2 * s, dz=0)           # tail
    f.line([(-0.38, -0.86), (-0.55, -0.95), (-0.72, -0.9)], color=FUR_DARK, lw=1.4 * s, dz=0.1)
    f.shape([(-0.40, -0.92), (-0.47, -0.55), (-0.38, -0.18), (-0.18, 0.04), (0.06, 0.10), (0.22, -0.02),
             (0.26, -0.32), (0.34, -0.58), (0.50, -0.74), (0.48, -0.93), (0.05, -0.98)], FUR, dz=1)  # body
    f.shape([(-0.30, -0.60), (0.05, -0.52), (0.40, -0.60), (0.56, -0.76), (0.42, -0.92), (-0.25, -0.93)],
            "#9a6c45", lw=0.45, dz=1.5)                                                             # haunch
    f.shape([(0.30, -0.90), (0.52, -0.86), (0.66, -0.92), (0.50, -0.98), (0.30, -0.98)], FACE_DARK, lw=0.5,
            dz=1.6)                                                                                  # foot
    hx, hy = 0.10, -0.07                                                                             # head offset
    H = lambda pts: [(u + hx, v + hy) for u, v in pts]
    f.shape(H([(-0.14, 0.30), (-0.10, 0.50), (0.08, 0.60), (0.28, 0.54), (0.38, 0.40), (0.36, 0.20),
               (0.22, 0.08), (0.02, 0.08), (-0.12, 0.16)]), FUR_LIGHT, dz=2.5)                       # head
    f.ellipse((0.02 + hx, 0.34 + hy), 0.075, 0.095, FACE_DARK, ec=INK, lw=0.4, dz=2.6)               # ear
    f.ellipse((0.025 + hx, 0.34 + hy), 0.032, 0.048, "#c79373", dz=2.7)
    f.shape(H([(0.17, 0.44), (0.31, 0.47), (0.40, 0.38), (0.47, 0.29), (0.535, 0.24), (0.535, 0.17),
               (0.48, 0.115), (0.36, 0.10), (0.24, 0.13), (0.16, 0.26)]), FACE, ec=INK, lw=0.4, dz=3)  # face
    f.line(H([(0.22, 0.425), (0.30, 0.455), (0.38, 0.42)]), color=FUR_DARK, lw=1.1 * s, dz=3.2)      # brow
    f.ellipse((0.315 + hx, 0.38 + hy), 0.045, 0.035, INK, dz=3.3)                                    # eye
    f.ellipse((0.52 + hx, 0.215 + hy), 0.022, 0.016, "#7a5a46", dz=3.3)                              # nostril
    f.line(H([(0.505, 0.155), (0.46, 0.14), (0.40, 0.15)]), color="#8a6650", lw=0.5, dz=3.3)         # mouth
    sh = np.array([0.13, -0.12])                                                                     # shoulder
    if reach_to is None:
        joints = [sh, sh + (0.10, -0.32), sh + (0.32, -0.50)]
        hand_dir = np.array([1.0, -0.15])
    else:
        tip = (np.asarray(reach_to, float) - (x, y)) / s
        v = tip - sh
        L = np.hypot(*v)
        u = v / L
        nrm = np.array([u[1], -u[0]])                                                                # elbow sags
        wrist = sh + u * (L - 0.13)
        elbow = sh + u * (L - 0.13) * 0.5 + nrm * 0.08
        joints, hand_dir = [sh, elbow, wrist], u
    _limb(f, joints, [0.085, 0.065, 0.045], FUR, dz=4)
    w = np.asarray(joints[-1], float); hd = hand_dir / np.hypot(*hand_dir); hn = np.array([-hd[1], hd[0]])
    palm = [w + 0.045 * hn, w + 0.07 * hd + 0.05 * hn, w + 0.09 * hd + 0.02 * hn, w + 0.13 * hd + 0.012 * hn,
            w + 0.135 * hd - 0.01 * hn, w + 0.08 * hd - 0.035 * hn, w - 0.04 * hn]                    # pointing hand
    f.shape([tuple(p) for p in palm], FACE_DARK, lw=0.5, dz=4.5)


def person(ax, x, y, s=1.0, z=2):
    """A seated person, head and shoulders in side profile, facing right."""
    f = Frame(ax, x, y, s, z)
    f.shape([(-0.46, -0.95), (-0.44, -0.30), (-0.30, -0.06), (-0.10, 0.02), (0.10, 0.00), (0.24, -0.10),
             (0.34, -0.36), (0.38, -0.95)], SHIRT, dz=1)                                             # torso
    f.line([(-0.12, -0.02), (-0.01, -0.09), (0.11, -0.02)], color=SHIRT_DARK, lw=0.7, dz=1.7)       # collar
    f.shape([(-0.07, -0.04), (-0.08, 0.14), (0.06, 0.14), (0.07, -0.03)], SKIN, lw=0.5, dz=1.5)      # neck
    f.shape([(-0.13, 0.10), (-0.22, 0.24), (-0.25, 0.40), (-0.19, 0.57), (-0.04, 0.66), (0.12, 0.64),
             (0.21, 0.55), (0.23, 0.46), (0.225, 0.43), (0.27, 0.36), (0.265, 0.335), (0.225, 0.325),
             (0.235, 0.29), (0.225, 0.265), (0.232, 0.24), (0.215, 0.20), (0.20, 0.16), (0.14, 0.13),
             (0.06, 0.12)], SKIN, dz=2)                                                              # head profile
    f.shape([(-0.255, 0.38), (-0.21, 0.58), (-0.05, 0.69), (0.12, 0.675), (0.22, 0.58), (0.205, 0.54),
             (0.10, 0.575), (-0.02, 0.56), (-0.11, 0.50), (-0.17, 0.42), (-0.22, 0.33)], HAIR, lw=0.5,
            dz=2.5)                                                                                  # hair
    f.ellipse((-0.035, 0.37), 0.075, 0.11, SKIN, ec=INK, lw=0.45, dz=2.6)                            # ear
    f.line([(-0.045, 0.40), (-0.015, 0.385), (-0.03, 0.345)], color=SKIN_DARK, lw=0.5, dz=2.7)
    f.line([(0.135, 0.455), (0.175, 0.465), (0.205, 0.452)], color=HAIR, lw=0.9 * s, dz=2.8)        # brow
    f.ellipse((0.175, 0.425), 0.032, 0.022, INK, dz=2.8)                                             # eye
    f.line([(0.20, 0.255), (0.225, 0.258)], color="#b07a68", lw=0.6, dz=2.8)                       # lips


def _wiggle(pts, amp, n=40, waves=3.0, seed=0):
    """A polyline through pts, resampled to n points with a small sideways wave (gyral folding)."""
    p = np.asarray(pts, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, seg[-1], n)
    q = np.c_[np.interp(t, seg, p[:, 0]), np.interp(t, seg, p[:, 1])]
    d = np.gradient(q, axis=0); nrm = np.c_[-d[:, 1], d[:, 0]] / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9)
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    return q + nrm * (amp * np.sin(2 * np.pi * waves * t / seg[-1] + phase))[:, None]


def brain(ax, x, y, s=1.0, human=False, z=2):
    """A lateral view of the left hemisphere, front to the right: cortex, main sulci, cerebellum, brainstem.
    Returns a function mapping a named site ('M1', 'aSTG', '6v') to data coordinates."""
    f = Frame(ax, x, y, s, z)
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
    f.shape(stem, CEREB, lw=0.6, dz=0)
    f.shape(cereb, CEREB, lw=0.6, dz=0.5)
    c = np.asarray(cereb)
    (x0, y0), (x1, y1) = c.min(0), c.max(0)
    for k in (0.3, 0.5, 0.7):                                                                        # folia
        yy = y1 - k * (y1 - y0)
        f.line([(x0 + 0.12 * (x1 - x0), yy + 0.02), (x0 + 0.5 * (x1 - x0), yy - 0.035),
                (x0 + 0.86 * (x1 - x0), yy - 0.005)], color=SULCUS, lw=0.5, dz=0.6)
    f.shape(cortex, BRAIN, lw=0.7, dz=1)
    sul = lambda pts, amp, lw=0.9, w=3.0, seed=0: f.line([tuple(q) for q in _wiggle(pts, amp, waves=w, seed=seed)],
                                                          color=SULCUS, lw=lw, dz=1.2)
    if human:
        central = [(0.00, 0.87), (0.05, 0.50), (0.15, 0.12)]
        lateral = [(0.50, -0.14), (0.15, 0.00), (-0.30, 0.16)]
        sul(central, 0.012, 1.0, 3, 1)
        sul([(0.20, 0.82), (0.25, 0.48), (0.33, 0.12)], 0.010, 0.75, 3, 2)                           # precentral
        sul([(-0.15, 0.85), (-0.10, 0.50), (0.00, 0.14)], 0.010, 0.75, 3, 3)                         # postcentral
        sul(lateral, 0.006, 1.1, 2, 4)
        sul([(0.40, -0.28), (0.0, -0.16), (-0.42, 0.0)], 0.010, 0.8, 3, 5)                           # STS
        sul([(0.27, 0.62), (0.55, 0.58), (0.80, 0.40)], 0.010, 0.7, 2, 7)                            # SFS
        sul([(0.33, 0.30), (0.58, 0.27), (0.84, 0.14)], 0.010, 0.7, 2, 8)                            # IFS
        sul([(-0.13, 0.50), (-0.42, 0.46), (-0.72, 0.26)], 0.010, 0.7, 2, 9)                         # intraparietal
        sites = {"6v": (0.30, 0.12), "M1": (0.13, 0.60)}
    else:
        central = [(-0.08, 0.59), (-0.01, 0.32), (0.10, 0.02)]
        lateral = [(0.50, -0.12), (0.10, 0.00), (-0.32, 0.14)]
        sul(central, 0.010, 1.0, 2, 1)
        sul(lateral, 0.006, 1.1, 2, 2)
        sul([(0.40, -0.24), (0.0, -0.14), (-0.45, 0.04), (-0.55, 0.16)], 0.010, 0.8, 2, 3)          # STS
        sul([(0.32, 0.48), (0.40, 0.24), (0.32, 0.02)], 0.006, 0.8, 1, 4)                            # arcuate
        sul([(0.50, 0.24), (0.70, 0.20), (0.86, 0.14)], 0.006, 0.8, 1, 5)                            # principal
        sul([(-0.14, 0.40), (-0.40, 0.30), (-0.58, 0.12)], 0.008, 0.7, 2, 6)                         # intraparietal
        sul([(-0.66, 0.36), (-0.72, 0.10), (-0.70, -0.14)], 0.008, 0.7, 2, 7)                        # lunate
        sites = {"M1": (0.08, 0.40), "aSTG": (0.30, -0.13)}
    return lambda name: f.P([sites[name]])[0]
