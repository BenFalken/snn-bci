"""
A seated macaque in side profile, built the way an illustrator constructs a figure, in four steps:

  1. PROPORTIONS  from rhesus macaque anatomy, in units of head height H (crown to chin): seated height ~4.2 H,
                  trunk (hip to shoulder) ~2.6 H, upper arm and forearm ~1.15 H each, hand ~0.6 H, thigh ~1.3 H,
                  shank ~1.25 H, foot ~1.1 H, tail ~1.8 H (rhesus: tail ~0.4 of body length). The head is about a
                  quarter of the seated height, so it reads as an animal rather than a cartoon.
  2. SKELETON     a seated pose as joint positions: hips low, trunk inclined forward, shoulders hunched, short neck
                  carrying the head forward, knees drawn up in front of the chest, feet flat; the near arm reaching
                  (to reach_to) or resting on the knee.
  3. STENCIL      simple volumes around the bones: ellipses for pelvis, ribcage and shoulder mass, tapered capsules
                  for neck and limbs. The body silhouette is the union of the trunk volumes (shapely), smoothed.
  4. STYLE        the user's mockup (2026-11-19): flat warm tans, no outlines; a thin darker accent stroke only where a
                  near limb overlaps the body (computed: the limb's edge clipped to the inside of the body), plus the
                  mockup's head (crest, glossy eye, concave snout, half-round ear) scaled to 1 H.

Every part is a named SVG group (<name>_<part>), so the figure stays editable.
"""
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Polygon
from matplotlib.path import Path as MPath
from shapely.geometry import LineString, Point, Polygon as SPolygon
from shapely.ops import unary_union
from shapely import affinity

FUR, FUR_FAR, FACE, EAR, ACCENT, RIM = "#d09a7b", "#bb8565", "#e3c2ab", "#d9aa8f", "#b98264", "#4a3a30"

# ------------------------------------------------------------------------------------------------ 1. proportions (H)
HUMERUS, RADIUS, HAND = 1.15, 1.15, 0.60
FEMUR, TIBIA, FOOT = 1.30, 1.25, 0.78

# ------------------------------------------------------------------------------------- 2. the seated pose (joints, H)
SEAT_Y, GROUND_Y = 0.0, -0.12
HIP = np.array([0.10, 0.40])
SHOULDER = np.array([0.55, 2.70])
NECK = np.array([0.95, 3.05])
HEAD_C = np.array([1.30, 3.45])            # centre of the 1 H head
KNEE = HIP + FEMUR * np.array([np.cos(np.radians(38)), np.sin(np.radians(38))])
ANKLE = KNEE + TIBIA * np.array([np.cos(np.radians(-82)), np.sin(np.radians(-82))])
ANKLE[1] = max(ANKLE[1], GROUND_Y + 0.13)
TOE = ANKLE + FOOT * np.array([np.cos(np.radians(-6)), np.sin(np.radians(-6))]) + np.array([0, -0.08])


def _capsule(a, b, ra, rb, n=24):
    """A tapered capsule from a (radius ra) to b (radius rb), as a shapely polygon."""
    return unary_union([Point(a).buffer(ra, n), Point(b).buffer(rb, n),
                        SPolygon(_side(a, b, ra, rb))])


def _side(a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.hypot(*(b - a))
    nrm = np.array([-d[1], d[0]])
    return [tuple(a + ra * nrm), tuple(b + rb * nrm), tuple(b - rb * nrm), tuple(a - ra * nrm)]


def _ellipse(c, w, h, angle=0):
    return affinity.rotate(affinity.scale(Point(c).buffer(1, 48), w / 2, h / 2), angle, origin=tuple(c))


def _smooth(g, r=0.08):
    return g.buffer(r, 32).buffer(-r, 32)


def _arm(reach_to_H):
    """(shoulder, elbow, wrist, hand direction) for a resting arm, or one reaching to reach_to_H (fingertip, H)."""
    if reach_to_H is None:
        elbow = SHOULDER + HUMERUS * np.array([np.cos(np.radians(-70)), np.sin(np.radians(-70))])
        wrist = elbow + RADIUS * np.array([np.cos(np.radians(-12)), np.sin(np.radians(-12))])
        return SHOULDER, elbow, wrist, np.array([np.cos(np.radians(-35)), np.sin(np.radians(-35))])
    tip = np.asarray(reach_to_H, float)
    v = tip - SHOULDER
    L = np.hypot(*v)
    u = v / L
    reach = min(L - HAND * 0.8, HUMERUS + RADIUS - 0.02)        # wrist, short of the fingertip by most of the hand
    wrist = SHOULDER + u * reach
    # two-bone inverse kinematics: elbow below the shoulder-wrist line
    half = reach / 2
    sag = np.sqrt(max(HUMERUS ** 2 - half ** 2, 0.0))
    nrm = np.array([u[1], -u[0]])
    elbow = SHOULDER + u * half + nrm * sag
    return SHOULDER, elbow, wrist, u


# ---------------------------------------------------------------------------------------------------- the mockup head
# The user's mockup in its own pixel frame (x right, y down); head height = skull top (y 100) to chin (y 503).
_HEAD_PX_H = 403.0
_HEAD_PX_C = np.array([380.0, 300.0])
HEAD_FUR = [(200, 500), (160, 445), (85, 355), (72, 290), (90, 230), (130, 170), (160, 135), (215, 58), (195, 112),
            (290, 32), (258, 82), (370, 25), (330, 68), (478, 28), (420, 62), (552, 47), (480, 72), (605, 82),
            (510, 100), (540, 130), (575, 220), (620, 300), (625, 350), (620, 470), (595, 503), (540, 500),
            (495, 515), (462, 545), (445, 575), (320, 560)]
HEAD_FACE = [(380, 225), (400, 170), (440, 125), (490, 100), (530, 115), (540, 135), (545, 190), (570, 250),
             (610, 285), (655, 300), (662, 340), (655, 372), (632, 395), (620, 430), (622, 470), (605, 498),
             (560, 500), (510, 490), (475, 465), (458, 425), (462, 385), (480, 352), (440, 342), (400, 310),
             (383, 265)]
HEAD_ACCENTS = {"face_edge": [(378, 228), (382, 282), (412, 324), (472, 352)],
                "cheek": [(476, 358), (456, 405), (471, 463)],
                "jaw": [(447, 570), (465, 540), (520, 506), (592, 500)],
                "brow": [(488, 146), (456, 152), (441, 182), (437, 236)]}


def _head_H(pts):
    p = (np.asarray(pts, float) - _HEAD_PX_C) / _HEAD_PX_H
    return [tuple(HEAD_C + np.array([q[0], -q[1]])) for q in p]


# ----------------------------------------------------------------------------------------------------------- drawing
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


def monkey(ax, x, y, s=1.0, reach_to=None, z=2, name="monkey"):
    """Draw the macaque. The local frame (as the other pictograms): about 1.6 s tall, seat near y - 0.95 s.
    reach_to: the fingertip in data coordinates, or None (the near arm resting on the knee)."""
    L = 1.6 / 4.2 * s                                            # H -> data units
    X0 = np.array([x - 0.75 * L, y - 0.95 * s])                 # data position of the H-frame origin (the seat)
    to_data = lambda P: [tuple(X0 + np.asarray(p, float) * L) for p in P]
    to_H = lambda p: (np.asarray(p, float) - X0) / L

    def add(patch, part):
        patch.set_gid(f"{name}_{part}")
        ax.add_patch(patch)

    def fill(part, geom, fc, dz):
        for g in getattr(geom, "geoms", [geom]):
            add(Polygon(to_data(list(g.exterior.coords)), closed=True, fc=fc, ec="none", zorder=z + dz), part)

    def stroke(part, coords, color=ACCENT, lw=0.75, dz=0.0, closed=False):
        add(PathPatch(MPath(to_data(coords)), fc="none", ec=color, lw=lw * s, zorder=z + dz, capstyle="round",
                      joinstyle="round"), part)

    def accent_inside(part, limb, body, dz, away_from=None):
        """The limb's outline where it lies inside the body: the edge that same-coloured parts would otherwise hide
        (not within 0.45 H of away_from, the joint where the limb grows out of the body)."""
        region = body.buffer(-0.02)
        if away_from is not None:
            region = region.difference(Point(away_from).buffer(0.45, 32))
        edge = limb.exterior.intersection(region)
        for g in getattr(edge, "geoms", [edge]):
            if g.length > 0.15:
                stroke(part, list(g.coords), dz=dz)

    # 3. stencil: the trunk as a union of volumes
    pelvis = _ellipse((-0.05, 0.50), 1.30, 1.05, 10)
    abdomen = _capsule((0.0, 0.70), (0.25, 1.75), 0.50, 0.55)
    ribcage = _ellipse((0.35, 2.10), 1.20, 1.55, -18)
    shoulder_mass = Point(SHOULDER).buffer(0.46, 32)
    neck = _capsule(SHOULDER + (0.05, 0.10), NECK, 0.40, 0.30)
    throat = _capsule(SHOULDER + (0.25, -0.05), HEAD_C + (-0.05, -0.42), 0.36, 0.26)   # fills under the jaw
    body = _smooth(unary_union([pelvis, abdomen, ribcage, shoulder_mass, neck, throat]), 0.10)

    thigh = _capsule(HIP, KNEE, 0.44, 0.27)
    shank = _capsule(KNEE, ANKLE, 0.24, 0.14)
    foot = _capsule(ANKLE + (0.0, -0.06), TOE + (0.0, 0.03), 0.13, 0.09)
    leg_near = _smooth(unary_union([thigh, shank]), 0.05)
    leg_far = affinity.translate(leg_near, -0.22, 0.06)
    foot_far = affinity.translate(foot, -0.20, 0.0)

    sh, el, wr, hd = _arm(None if reach_to is None else to_H(reach_to))
    arm = _smooth(unary_union([_capsule(sh, el, 0.23, 0.17), _capsule(el, wr, 0.16, 0.11)]), 0.03)
    hn = np.array([-hd[1], hd[0]])
    hand = SPolygon([tuple(wr + 0.10 * hn), tuple(wr + 0.30 * hd + 0.11 * hn), tuple(wr + 0.48 * hd + 0.05 * hn),
                     tuple(wr + HAND * hd + 0.02 * hn), tuple(wr + HAND * hd - 0.03 * hn),
                     tuple(wr + 0.30 * hd - 0.10 * hn), tuple(wr - 0.09 * hn)]).buffer(0.03, 16)

    # tail: a tapering curve from the rump, down and back, curling up
    tail_pts = np.array([(-0.55, 0.45), (-0.95, 0.22), (-1.30, 0.40), (-1.42, 0.85)])
    tail = unary_union([_capsule(a, b, ra, rb) for a, b, ra, rb in
                        zip(tail_pts[:-1], tail_pts[1:], (0.11, 0.09, 0.075), (0.09, 0.075, 0.06))])

    # 4. draw, back to front
    fill("tail", tail, FUR, 0.0)
    fill("leg_far", leg_far, FUR_FAR, 0.2)
    fill("foot_far", foot_far, FUR_FAR, 0.25)
    fill("body", body, FUR, 1.0)
    fill("leg_near", leg_near, FUR, 1.5)
    accent_inside("leg_near_edge", leg_near, body, 1.6, away_from=HIP)
    fill("foot", foot, FACE, 1.7)
    add(Polygon(_head_poly := to_data(_head_H(HEAD_FUR)), closed=True, fc=FUR, ec="none", zorder=z + 2.5),
        "head")
    add(PathPatch(_bezier(to_data(_head_H(HEAD_FACE))), fc=FACE, ec="none", zorder=z + 2.6), "face")
    for part, pts in HEAD_ACCENTS.items():
        add(PathPatch(_bezier(to_data(_head_H(pts)), closed=False), fc="none", ec=ACCENT, lw=0.7 * s,
                      zorder=z + 3, capstyle="round"), part)
    th = np.radians(np.linspace(78, 312, 30))
    ear = [(245 + 62 * np.cos(t), 340 - 62 * np.sin(t)) for t in th]
    add(Polygon(to_data(_head_H(ear)), closed=True, fc=EAR, ec="none", zorder=z + 2.7), "ear")
    rim = [(245 + 63 * np.cos(t), 340 - 63 * np.sin(t)) for t in np.radians(np.linspace(178, 305, 16))]
    add(PathPatch(_bezier(to_data(_head_H(rim)), closed=False), fc="none", ec=RIM, lw=0.5 * s, zorder=z + 2.75),
        "ear_rim")
    (ex, ey), = to_data(_head_H([(486, 212)]))
    add(Ellipse((ex, ey), 80 / _HEAD_PX_H * L, 118 / _HEAD_PX_H * L, fc="black", ec="none", zorder=z + 2.8), "eye")
    for part, (px, py, r) in {"eye_highlight": (501, 190, 17), "eye_highlight_small": (474, 235, 8.5)}.items():
        (hx, hy), = to_data(_head_H([(px, py)]))
        add(Ellipse((hx, hy), 2 * r / _HEAD_PX_H * L, 2.1 * r / _HEAD_PX_H * L, fc="white", ec="none",
                    zorder=z + 2.85), part)
    fill("arm", arm, FUR, 4.0)
    accent_inside("arm_edge", arm, unary_union([body, leg_near]), 4.1, away_from=SHOULDER)
    fill("hand", hand, FACE, 4.2)
