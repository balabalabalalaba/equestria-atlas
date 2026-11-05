"""The hero and the NPC cast — modular paper-doll pixel characters.

    from engine.character import draw_hero, draw_npc, hero_anchor, npc_anchor
    draw_hero(c, 120, 150, 'walk', f, facing='right', outfit='casual', backpack=True)
    hx, hy = hero_anchor('drink', f, 'right', 'hand')
    draw_iced_glass(c, 120 + hx, 150 + hy + 4)

(x, y) is the bottom-centre of the feet: y is the floor line, the soles' bottom
row is y-1. The hero is 32 px tall standing. See docs/CHARACTER.md for the pose
table, anchors, speeds and the seated / bed geometry.

How it works: every frame is composed in *label space* (one letter per
material: hair, skin, top, pants, shoes...) from hand-drawn ASCII parts (heads,
torsos, hands, shoes, accessories) and procedurally drawn limbs (2-bone IK, 2-3
px thick, shaded light-from-upper-left). Where a part overlaps another, an
interior contour line in the darker shade of the front part is drawn; finally a
1 px 'ink' silhouette outline is added and the labels are mapped to the colours
of the outfit / NPC. Every frame is built once and cached; a draw call is a
single masked blit (~0.02-0.05 ms).
"""
import math
from functools import lru_cache

import numpy as np

from .core import col, FRAMES_PER_BEAT
from .sprite import Sprite

# =====================================================================================
# HERO COLOURS — edit here to re-colour the hero (palette names, '#hex' or (r,g,b)).
# Everything about the hero's look is built from this dict + _OUTFIT_PAL below.
# =====================================================================================
HERO_COLORS = {
    'hair_dk':  'hair',        # darkest hair (shadow side, partings, inner lines)
    'hair':     'hair_mid',    # main hair colour
    'hair_hi':  'hair_hi',     # hair highlight (light from the upper-left)
    'skin_dk':  'skin_dk',     # skin contour lines (ear, jaw, fingers)
    'skin_sh':  'skin_sh',     # skin shadow
    'skin':     'skin',        # skin base
    'skin_hi':  'skin_hi',     # skin highlight
    'eye':      'ink',         # pupil
    'eye_hi':   'white',       # catch-light
    'blush':    'salmon',      # cheeks
    'mouth':    'skin_dk',     # closed mouth line
    'mouth_in': 'maroon',      # open mouth
}

OUTFITS = ('pajamas', 'casual', 'gym', 'lab', 'apron')

# =====================================================================================
# Labels: one letter per material. Parts are authored with these letters and every
# outfit / NPC maps them to colours.
# =====================================================================================
_LABELS = (
    '.',                        # transparent
    'o',                        # ink outline
    'h', 'H', 'j',              # hair: dark / mid / highlight
    'k', 'S', 's', 'l',         # skin: line / shade / base / highlight
    'e', 'w', 'b', 'm', 'n',    # eye, eye catch-light, blush, mouth line, mouth inside
    'd', 'T', 't', 'u',         # top: line / shade / base / highlight
    'a', 'A',                   # top accent (stripes, drawstrings, collar) + its shade
    'D', 'P', 'p', 'q',         # pants: line / shade / base / highlight
    'F', 'f', 'g', 'i',         # shoes: shade / base / sole / sock-or-lace accent
    'x', 'R', 'r', 'Q',         # apron: line / shade / base / highlight
    'Y', 'y', 'z',              # goggles: frame+strap / lens / glint
    'v', 'C', 'c', 'K', 'V',    # backpack: line / shade / base / highlight / strap
    'G', 'E',                   # glasses: frame / lens glint
    'B',                        # name badge
    'X', 'Z',                   # misc accent (sweatband, cap, hair tie) + its shade
)
_C = {ch: i for i, ch in enumerate(_LABELS)}

# far-side limbs are drawn one shade darker
_FAR = {'t': 'T', 'u': 't', 'T': 'd', 'a': 'A', 's': 'S', 'l': 's', 'S': 'k', 'p': 'P',
        'q': 'p', 'P': 'D', 'f': 'F', 'i': 'F', 'g': 'F', 'H': 'h', 'j': 'H', 'c': 'C',
        'K': 'c', 'C': 'v', 'r': 'R', 'Q': 'r', 'R': 'x', 'X': 'Z', 'V': 'v'}
_FAR_LUT = np.arange(256, dtype=np.uint8)
for _a, _b in _FAR.items():
    _FAR_LUT[_C[_a]] = _C[_b]

# the line (darkest) shade of each material: used for interior contour lines
_LINE_LUT = np.zeros(256, np.uint8)
for _grp, _ln in (('tuTd', 'd'), ('aA', 'd'), ('sSlkbmnew', 'k'), ('pPqD', 'D'), ('fFgi', 'F'),
                  ('hHj', 'h'), ('rRxQ', 'x'), ('cCKvV', 'v'), ('yYz', 'Y'), ('GE', 'G'),
                  ('XZ', 'Z'), ('B', 'd'), ('o', 'o')):
    for _ch in _grp:
        _LINE_LUT[_C[_ch]] = _C[_ln]


def _art(s):
    """ASCII art (label letters) -> uint8 label array. '.' and ' ' are transparent."""
    lines = [ln.strip() for ln in s.strip('\n').split('\n')]
    lines = [ln for ln in lines if ln]
    w = max(len(ln) for ln in lines)
    a = np.zeros((len(lines), w), np.uint8)
    for y, ln in enumerate(lines):
        for x, ch in enumerate(ln):
            if ch not in '. ':
                a[y, x] = _C[ch]
    return a


# =====================================================================================
# Composition canvas (label space). Origin = the floor point under the character.
# =====================================================================================
_CW, _CH = 80, 68
_OX, _OY = 40, 58          # origin pixel inside the work canvas (covers y -58..+9)


def _ring(m):
    """4-connected 1 px ring around mask m."""
    g = np.zeros_like(m)
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[1:, :] |= m[:-1, :]
    g[:-1, :] |= m[1:, :]
    return g & ~m


def _adjacent_label(lay):
    """For every pixel, the label of a 4-neighbour layer pixel (0 if none)."""
    out = np.zeros_like(lay)
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        sh = np.zeros_like(lay)
        ys = slice(max(0, dy), _CH + min(0, dy))
        yd = slice(max(0, -dy), _CH + min(0, -dy))
        xs = slice(max(0, dx), _CW + min(0, dx))
        xd = slice(max(0, -dx), _CW + min(0, -dx))
        sh[yd, xd] = lay[ys, xs]
        sel = (out == 0) & (sh > 0)
        out[sel] = sh[sel]
    return out


class _Comp:
    """Label-space compositor: layers are added back to front."""
    __slots__ = ('L', 'anchors')

    def __init__(self):
        self.L = np.zeros((_CH, _CW), np.uint8)
        self.anchors = {}

    @staticmethod
    def layer():
        return np.zeros((_CH, _CW), np.uint8)

    def add(self, lay, line=True, far=False):
        """Composite a layer on top. Where its 1 px ring overlaps what is already
        there, draw an interior contour line in the layer's own darkest shade."""
        if far:
            lay = _FAR_LUT[lay]
        m = lay > 0
        if not m.any():
            return
        if line:
            ring = _ring(m) & (self.L > 0) & (self.L != _C['o'])
            if ring.any():
                src = _adjacent_label(lay)
                self.L[ring] = _LINE_LUT[src[ring]]
        self.L[m] = lay[m]


def _put(lay, art, x, y, flip=False):
    """Paste label art with its top-left at origin-space (x, y)."""
    if art is None:
        return
    if flip:
        art = art[:, ::-1]
    h, w = art.shape
    X, Y = int(x) + _OX, int(y) + _OY
    x0, y0 = max(0, X), max(0, Y)
    x1, y1 = min(_CW, X + w), min(_CH, Y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = art[y0 - Y:y1 - Y, x0 - X:x1 - X]
    m = sub > 0
    lay[y0:y1, x0:x1][m] = sub[m]


def _px(lay, x, y, lab):
    X, Y = int(x) + _OX, int(y) + _OY
    if 0 <= X < _CW and 0 <= Y < _CH:
        lay[Y, X] = _C[lab]


def _bres(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def _seg_mask(p0, p1, w):
    """Mask of a thick segment. w=2: 2x2 brush, top-left on the line.
    w=3: 3x3 brush centred on the line (rounded ends)."""
    m = np.zeros((_CH, _CW), bool)
    for x, y in _bres(p0[0], p0[1], p1[0], p1[1]):
        X, Y = x + _OX, y + _OY
        if w == 1:
            m[Y, X] = True
        elif w == 2:
            m[Y:Y + 2, X:X + 2] = True
        else:
            m[Y - 1:Y + 2, X - 1:X + 2] = True
    return m


def _ik(a, b, l1, l2, hint):
    """2-bone IK. Returns the joint between a and b. `hint` = (hx, hy) direction the
    joint should bulge towards; the solution closest to it is used."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ax + hint[0] * l1, ay + hint[1] * l1)
    if d >= l1 + l2 - 1e-6:
        t = l1 / (l1 + l2)
        return (ax + dx * t, ay + dy * t)
    d = max(d, abs(l1 - l2) + 1e-3)
    ca = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    ang = math.acos(max(-1.0, min(1.0, ca)))
    base = math.atan2(dy, dx)
    best, bs = None, -1e9
    for sgn in (1, -1):
        t = base + sgn * ang
        j = (ax + l1 * math.cos(t), ay + l1 * math.sin(t))
        s = (j[0] - (ax + bx) / 2) * hint[0] + (j[1] - (ay + by) / 2) * hint[1]
        if s > bs:
            best, bs = j, s
    return best


def _ri(p):
    return (int(round(p[0])), int(round(p[1])))


_HINTS = {'back': (-1, 0.15), 'fwd': (1, 0.1), 'down': (0, 1), 'up': (0, -1),
          'out': (1, 0), 'in': (-1, 0), 'bfwd': (0.4, 1), 'bback': (-0.4, 1)}


# =====================================================================================
# PARTS — hand-drawn label art. Side-view art faces RIGHT (left = mirrored).
# Heads: 14 x 12 boxes (side) / 13 x 12 (front, back); the neck anchor is the
# column NECK[view] on the bottom row.
# =====================================================================================
_NECK = {'side': 6, 'front': 6, 'back': 6}

# --- hair styles (hair + skin in one piece; face features are added by code)
_HEADS = {}
# per-style overrides of face positions (e.g. a style without a visible ear)
_FACE_OVR = {}
# per-(style, view) head-box metadata: (xoff, ydown). xoff = extra columns on the LEFT of
# the standard box (ponytails, buns); ydown = rows BELOW the neck row (hair hanging down).
# Extra rows on top (tall hair, beanies) need no metadata: the art is simply taller.
_HEAD_META = {}
_HEADS['hero'] = {
    'side': _art('''
        ....hh.hHh....
        ..hhHjhjjHHh..
        .hHjjjjHHHHHh.
        hHjjHHHHHHHHHh
        hHjHHHHHHHHHjh
        hHHHHHHHHHHjsH
        hHHHHHHhHlssS.
        hhHHHhSssssss.
        hHHHhlSsssssss
        .hHhSSsssssss.
        ..hhkSsssssS..
        ....kSSsssS...
    '''),
    'front': _art('''
        ....hHjHh....
        ..hHjjjjHHh..
        .hHjjHHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHjHHh
        hHHHhHHHhHHHh
        hHhslshHsslHh
        hHsssssssssHh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    '''),
    'back': _art('''
        ....hHjHh....
        ..hHjjjjHHh..
        .hHjjHHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHhHHhHHh.
        ...hSsssSh...
        ....SsssS....
    '''),
}


# --- NPC hair styles. Same 14x12 (side) / 13x12 (front, back) face grid as the hero;
# bigger hair adds rows on top, columns on the left (xoff) or rows below the neck
# (ydown) — see _HEAD_META. X/Z/B in hair art = hat / hair-tie colours (per NPC).
_HEADS['curly'] = {    # big messy curls (sci_a, ginger + beard; shopper_a, white perm)
    'side': _art("""
        ...hh.hHh.hh...
        ..hHjhHjjhHjh..
        .hHjjHjjHHjHHh.
        hHjjHHHjHHHHjHh
        hjHHjHHHHjHHHHh
        hHHjHHHHjHHHjHh
        hHjHHHHjHHHHHjh
        hHHjHHHHhHlsjs.
        hHjHHHhSssssss.
        hhHHHhlSsssssss
        .hHHhSSsssssss.
        ..hhhkSsssssS..
        .....kSSsssS...
    """),
    'front': _art("""
        ...hh.hHh.hh...
        ..hHjhHjjhHjh..
        .hHjjHjjHHjHHh.
        hHjjHHjHHjHHjHh
        hjHHjHHjHHHjHHh
        hHHjHHHHHjHHHjh
        hHjHHhHjHhHjHHh
        hhHhslshHsslHhh
        hhHsssssssssHhh
        .hSsssssssssSh.
        ..Sssssssssss..
        ...Sssssssss...
        ....SSsssSS....
    """),
    'back': _art("""
        ...hh.hHh.hh...
        ..hHjhHjjhHjh..
        .hHjjHjjHHjHHh.
        hHjjHHjHHjHHjHh
        hjHHjHHjHHHjHHh
        hHHjHHHHHjHHHjh
        hHjHHjHHjHHjHHh
        hHHjHHHHHHjHHHh
        hSHHHjHHHjHHHSh
        .sHHHHHHHHHHHs.
        ..hHHhHHhHHHh..
        ....hSsssSh....
        .....SsssS.....
    """),
}
_HEAD_META.update({('curly', 'side'): (1, 0), ('curly', 'front'): (1, 0), ('curly', 'back'): (1, 0)})

_HEADS['ponytail'] = {   # sleek long hair, low ponytail with a tie (sci_b)
    'side': _art("""
        ......hhhHh.....
        ....hhHHjjjHh...
        ...hHHjjjHHHHh..
        ..hHHjjHHHHHHHh.
        ..hHHHHHHHHHHHHh
        ..hHHHHHHHHHHHjH
        ..hHHHHHHhHlsHs.
        ..hhHHHhSssssss.
        .XXHHHhlSsssssss
        hHhhHhSSsssssss.
        hHh.hhkSsssssS..
        hHh...kSSsssS...
        hHh.............
        .hh.............
        .h..............
    """),
    'front': _art("""
        ....hhHhh....
        ..hHHjhjjHh..
        .hHjjHhHHjHh.
        hHjHHHhHHHHHh
        hHjHHhshHHHHh
        hHHHhsssshHHh
        hHHslsssslHHh
        hHsssssssssHh
        hHsssssssssHh
        hhSsssssssShh
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        ....hhHhh....
        ..hHHjjjjHh..
        .hHjjjHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        SHHHHHhHHHHHS
        sHHHHXXXHHHHs
        .hHHHhHhHHHh.
        ...hShHhSh...
        ....SHHHS....
        .....hHh.....
        .....hHh.....
        .....hHh.....
        ......h......
    """),
}
_HEAD_META.update({('ponytail', 'side'): (2, 3), ('ponytail', 'back'): (0, 4)})

_HEADS['balding'] = {    # grey horseshoe of hair, shiny dome (sci_c)
    'side': _art("""
        ..............
        .....sllss....
        ...sllllsss...
        ..sllssssssS..
        .hlssssssssss.
        hHjssssssssssS
        hHHjsssssshhs.
        hHHHhSsssssss.
        hHHjhlSsssssss
        .hHhSSsssssss.
        ..hhkSsssssS..
        ....kSSsssS...
    """),
    'front': _art("""
        .............
        ....sslss....
        ..sllllssss..
        .slllsssssss.
        .slssssssssS.
        hjssssssssssh
        hHshhssshhsHh
        hHsssssssssHh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .............
        ....sslss....
        ..sllllssss..
        .slllsssssss.
        .slssssssssS.
        hjsssssssssSh
        hHjsssssssHHh
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHhHHhHHh.
        ...hSsssSh...
        ....SsssS....
    """),
}

_HEADS['bob'] = {        # chin-length bob with a straight fringe (student_a)
    'side': _art("""
        ....hhhHh.....
        ..hhHjjjjHh...
        .hHjjjHHHHHh..
        hHjjHHHHHHHHh.
        hHjHHHHHHHHHHh
        hHjHHHHHHHHHHh
        hHHHHHHHHHHhl.
        hHHHHHHHhSsss.
        hHjHHHHHhSssss
        hHHHHHHhSssss.
        .hHHHHhhkssS..
        ..hhhhhhkSsS..
    """),
    'front': _art("""
        ....hhHhh....
        ..hHjjjjHHh..
        .hHjjHHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHHHHh
        hHjHHHHHHHHHh
        hHHHhHHHhHHHh
        hHHsssssssHHh
        hHsssssssssHh
        hHSsssssssSHh
        hhhSsssssShhh
        ...SSsssSS...
    """),
    'back': _art("""
        ....hhHhh....
        ..hHjjjjHHh..
        .hHjjHHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        hhhhSsssShhhh
        ....SsssS....
    """),
}
_FACE_OVR[('bob', 'side')] = {'ear': ()}

_HEADS['beanie'] = {     # knitted beanie with a ribbed fold (student_b); X/Z/B = hat
    'side': _art("""
        .....ZXXZ.....
        ...ZXBBXXZ....
        ..ZXBXXXXXZ...
        .ZXBXXXXXXXZ..
        .ZXXXXXXXXXXZ.
        ZXXXXXXXXXXXXZ
        ZXZXZXZXZXZXZX
        hHhHHHHhSlsss.
        hhHHHhSssssss.
        hHHHhlSsssssss
        .hHhSSsssssss.
        ..hhkSsssssS..
        ....kSSsssS...
    """),
    'front': _art("""
        .....ZXZ.....
        ...ZXXBXXZ...
        ..ZXBBXXXXZ..
        .ZXBXXXXXXXZ.
        .ZXXXXXXXXXZ.
        ZXXXXXXXXXXXZ
        ZXZXZXZXZXZXZ
        hHsslssslssHh
        hHsssssssssHh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .....ZXZ.....
        ...ZXXBXXZ...
        ..ZXBBXXXXZ..
        .ZXBXXXXXXXZ.
        .ZXXXXXXXXXZ.
        ZXXXXXXXXXXXZ
        ZXZXZXZXZXZXZ
        hHHHHHHHHHHHh
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHhHHhHHh.
        ...hSsssSh...
        ....SsssS....
    """),
}

_HEADS['buzz'] = {       # buzz cut (gym_a)
    'side': _art("""
        ..............
        .....hhhhh....
        ...hhHHjjHh...
        ..hHHjjjHHHh..
        .hHHjjHHHHHHh.
        .hHHHHHHHHHHs.
        .hHHHHHHHhsss.
        .hHHHhSssssss.
        .hHHhlSsssssss
        ..hhSSsssssss.
        ...SkSsssssS..
        ....kSSsssS...
    """),
    'front': _art("""
        .............
        ....hhhhh....
        ..hhHjjjHHh..
        .hHjjHHHHHHh.
        .hHjHHHHHHHh.
        hHHHHHHHHHHHh
        hSshhssshhsSh
        hSsssssssssSh
        SSsssssssssSS
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .............
        ....hhhhh....
        ..hhHjjjHHh..
        .hHjjHHHHHHh.
        .hHjHHHHHHHh.
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        SHHHHHHHHHHHS
        SsHHHHHHHHHsS
        .SsHHHHHHHsS.
        ...SsssssS...
        ....SsssS....
    """),
}

_HEADS['high_pony'] = {  # hair pulled up into a high ponytail with a tie (gym_b)
    'side': _art("""
        .....XX..........
        .hhhhXXhhhHh.....
        hHjjhhHjjjjjHh...
        hHHhhHjjjHHHHHh..
        hHh.hHjjHHHHHHHh.
        hH..hHjHHHHHHHHHh
        hHh.hHHHHHHHHHHjH
        .hH.hHHHHHHhHlsHs.
        .hh.hhHHHhSssssss.
        ..h.hHHHhlSsssssss
        ....hhHhSSsssssss.
        ......hhkSsssssS..
        ........kSSsssS...
    """),
    'front': _art("""
        .....hXh.....
        ....hhXhh....
        ..hHjjjjHHh..
        .hHjjHHHHHHh.
        hHjjHHHHHHHHh
        hHjHHHHHHHHHh
        hHHHsllsssHHh
        hHsllssssssHh
        hHsssssssssHh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .....hXh.....
        ....hhXhh....
        ..hHjjhjHHh..
        .hHjjHhHHHHh.
        hHjjHHhHHHHHh
        hHjHHHhHHHHHh
        hHHHHHhHHHHHh
        hHHHHhHhHHHHh
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHhHHhHHh.
        ...hSsssSh...
        ....SsssS....
    """),
}
_HEAD_META.update({('high_pony', 'side'): (4, 0)})

_HEADS['bun'] = {        # hair pulled back into a bun (cashier)
    'side': _art("""
        .hhHh...........
        hHjjHhhhhHh.....
        hHjHHhHjjjjHh...
        .hhhhHjjHHHHHh..
        ..hHjjHHHHHHHHh.
        ..hHjHHHHHHHHHHh
        ..hHHHHHHHHHHHjH
        ..hHHHHHHhHlsHs.
        ..hhHHHhSssssss.
        ..hHHHhlSsssssss
        ...hHhSSsssssss.
        ....hhkSsssssS..
        ......kSSsssS...
    """),
    'front': _art("""
        .....hHh.....
        ....hhjhh....
        ..hHjjhjHHh..
        .hHjjHhHHHHh.
        hHjjHHhHHHHHh
        hHjHHhshHHHHh
        hHHHhsssshHHh
        hHHsllssssHHh
        hHsssssssssHh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .....hHh.....
        ...hhHjjHh...
        ..hHjjjHHHh..
        .hHjHHHHHHHh.
        hHjjhhhhhhHHh
        hHjHHHHHHHHHh
        hHHHHHHHHHHHh
        hHHHHHHHHHHHh
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHHHHHHHh.
        ...hSsssSh...
        ....SsssS....
    """),
}
_HEAD_META.update({('bun', 'side'): (2, 0)})

_HEADS['cap'] = {        # baseball cap, brim forward (shopper_b); X/Z/B = cap
    'side': _art("""
        ................
        ....ZXXXXZ......
        ..ZXBBXXXXXZ....
        .ZXBXXXXXXXXZ...
        .ZXXXXXXXXXXXZ..
        hZZZZZZZZZZZZXXZ
        hHHHHHHhSssls...
        hhHHHhSssssss...
        hHHHhlSsssssss..
        .hHhSSsssssss...
        ..hhkSsssssS....
        ....kSSsssS.....
    """),
    'front': _art("""
        .............
        ....ZXXXZ....
        ..ZXBBXXXXZ..
        .ZXBXXXXXXXZ.
        .ZXXXXXXXXXZ.
        ZXXXXXXXXXXXZ
        ZZZZZZZZZZZZZ
        hSsssssssssSh
        hSsssssssssSh
        .Sssssssssss.
        ..Sssssssss..
        ...SSsssSS...
    """),
    'back': _art("""
        .............
        ....ZXXXZ....
        ..ZXBBXXXXZ..
        .ZXBXXXXXXXZ.
        .ZXXXXXXXXXZ.
        ZXXXZhhhZXXXZ
        ZZZZZZZZZZZZZ
        SHHHHHHHHHHHS
        sHHHHHHHHHHHs
        .hHHHhHHhHHh.
        ...hSsssSh...
        ....SsssS....
    """),
}

# face feature positions in head-box coordinates
#   side: eye = top-left of a 2x2 eye box, mouth = mouth pixel, blush, ear pixels
_FACE = {
    'side': dict(eye=(10, 7), mouth=(11, 10), blush=((9, 9),), ear=((5, 8), (6, 8), (5, 9))),
    'front': dict(eyes=((3, 7), (8, 7)), mouth=(6, 10), blush=((2, 9), (10, 9))),
}


def _hmeta(style, view):
    return _HEAD_META.get((style, view), (0, 0))


@lru_cache(maxsize=None)
def _face_pos(style, view):
    """Face feature positions for a head style, in that style's art coordinates."""
    F = dict(_FACE['side' if view == 'side' else 'front'])
    F.update(_FACE_OVR.get((style, view), {}))
    xo, yd = _hmeta(style, view)
    yt = _HEADS[style][view].shape[0] - 12 - yd
    out = {'xoff': xo, 'ytop': yt}
    for k, v in F.items():
        if k in ('eye', 'mouth'):
            out[k] = (v[0] + xo, v[1] + yt)
        else:
            out[k] = tuple((p[0] + xo, p[1] + yt) for p in v)
    return out


def _set(a, x, y, lab):
    if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        a[y, x] = _C[lab]


def _get(a, x, y):
    if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        return _LABELS[a[y, x]]
    return '.'


# --- eyes & mouths -------------------------------------------------------------------
def _side_eye(a, x, y, st):
    E = lambda xx, yy, l: _set(a, xx, yy, l)  # noqa: E731
    if st == 'open':
        E(x, y, 'e'); E(x + 1, y, 'w'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st in ('blink', 'closed'):
        E(x, y + 1, 'k'); E(x + 1, y + 1, 'k')
    elif st == 'joy':
        E(x - 1, y + 1, 'k'); E(x, y, 'e'); E(x + 1, y, 'e'); E(x + 2, y + 1, 'k')
    elif st == 'half':
        E(x, y, 'k'); E(x + 1, y, 'k'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'wide':
        E(x, y - 1, 'e'); E(x + 1, y - 1, 'w'); E(x, y, 'e'); E(x + 1, y, 'e')
        E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'focus':
        E(x - 1, y - 1, 'h'); E(x, y - 1, 'h'); E(x + 1, y - 1, 'h')
        E(x, y, 'e'); E(x + 1, y, 'w'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'down':
        E(x, y + 1, 'e'); E(x + 1, y + 1, 'e'); E(x, y, 'k'); E(x + 1, y, 'k')
    elif st == 'up':
        E(x, y - 1, 'e'); E(x + 1, y - 1, 'w'); E(x, y, 'e'); E(x + 1, y, 'e')


def _side_mouth(a, x, y, st):
    M = lambda xx, yy, l: _set(a, xx, yy, l)  # noqa: E731
    if st == 'smile':
        M(x, y, 'm'); M(x - 1, y - 1, 'm')
    elif st == 'flat':
        M(x, y, 'm'); M(x - 1, y, 'm')
    elif st in ('open', 'talk'):
        M(x, y, 'n'); M(x - 1, y, 'm')
    elif st == 'grin':
        M(x, y, 'n'); M(x - 1, y, 'n'); M(x - 2, y - 1, 'm')
    elif st == 'O':
        M(x, y, 'n'); M(x, y + 1, 'n'); M(x - 1, y, 'm')
    elif st == 'yawn':
        M(x, y - 1, 'n'); M(x, y, 'n'); M(x, y + 1, 'n'); M(x - 1, y, 'n'); M(x - 1, y + 1, 'm')
    elif st == 'chew':
        M(x, y, 'm'); M(x - 2, y, 'S'); M(x - 2, y - 1, 'l')
    elif st == 'frown':
        M(x, y, 'm'); M(x - 1, y + 1, 'm')
    elif st == 'sip':
        M(x, y, 'm')


def _front_eye(a, x, y, st, left):
    E = lambda xx, yy, l: _set(a, xx, yy, l)  # noqa: E731
    # catch-light on the upper-left of each pupil (light from the upper-left)
    if st == 'open':
        E(x, y, 'w'); E(x + 1, y, 'e'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st in ('blink', 'closed'):
        E(x, y + 1, 'k'); E(x + 1, y + 1, 'k')
    elif st == 'joy':
        E(x, y, 'e'); E(x + 1, y, 'e'); E(x - 1, y + 1, 'k'); E(x + 2, y + 1, 'k')
    elif st == 'half':
        E(x, y, 'k'); E(x + 1, y, 'k'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'wide':
        E(x, y - 1, 'w'); E(x + 1, y - 1, 'e'); E(x, y, 'e'); E(x + 1, y, 'e')
        E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'focus':
        if left:
            E(x - 1, y - 1, 'h'); E(x, y - 1, 'h'); E(x + 1, y - 1, 'h')
        else:
            E(x, y - 1, 'h'); E(x + 1, y - 1, 'h'); E(x + 2, y - 1, 'h')
        E(x, y, 'w'); E(x + 1, y, 'e'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'down':
        E(x, y, 'k'); E(x + 1, y, 'k'); E(x, y + 1, 'e'); E(x + 1, y + 1, 'e')
    elif st == 'up':
        E(x, y - 1, 'w'); E(x + 1, y - 1, 'e'); E(x, y, 'e'); E(x + 1, y, 'e')


def _front_mouth(a, x, y, st):
    M = lambda xx, yy, l: _set(a, xx, yy, l)  # noqa: E731
    if st == 'smile':
        M(x - 1, y - 1, 'm'); M(x, y, 'm'); M(x + 1, y - 1, 'm')
    elif st == 'flat':
        M(x, y, 'm')
    elif st in ('open', 'talk'):
        M(x, y, 'n'); M(x - 1, y - 1, 'm'); M(x + 1, y - 1, 'm')
    elif st == 'grin':
        M(x - 1, y - 1, 'n'); M(x, y - 1, 'n'); M(x + 1, y - 1, 'n'); M(x, y, 'n')
    elif st == 'O':
        M(x, y - 1, 'n'); M(x, y, 'n')
    elif st == 'yawn':
        M(x - 1, y - 1, 'n'); M(x, y - 1, 'n'); M(x + 1, y - 1, 'n')
        M(x - 1, y, 'n'); M(x, y, 'n'); M(x + 1, y, 'n')
    elif st == 'chew':
        M(x - 1, y, 'm'); M(x, y, 'm'); M(x + 1, y - 1, 'm'); M(x + 2, y - 1, 'l')
    elif st == 'frown':
        M(x - 1, y, 'm'); M(x, y - 1, 'm'); M(x + 1, y, 'm')
    elif st == 'sip':
        M(x, y, 'm')


# =====================================================================================
# TORSOS: 10 rows (y = hip-9 .. hip), side art 8 wide (x = -4..+3, faces right),
# front/back art 9 wide (x = -4..+4). Bottom two rows are the waist (pants).
# =====================================================================================
_TORSO = {
    'side': _art('''
        ..uttt..
        .uttttt.
        .utttttT
        .utttttT
        .utttttT
        .utttttT
        .utttttT
        .TTTTTTT
        .qpppppP
        .qpppppP
    '''),
    'front': _art('''
        ..utttT..
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .TTTTTTTT
        .qppppppP
        .qpppDppP
    '''),
    'back': _art('''
        ..utttT..
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .TTTTTTTT
        .qppppppP
        .qpppDppP
    '''),
}


def _torso_art(top, view):
    """Torso label art for a top style (hoodie, pj, tee, tank, ...)."""
    return _torso_cached(top, view)


_COAT = {
    'side': _art('''
        ..uttt..
        .uttttA.
        .utttttA
        .utttttT
        .uttttta
        .utttttT
        .uttttta
        .dTTTTTd
        .utttttT
        .uttttta
        .utttttT
        .uttttTT
        .TTTTTTd
    '''),
    'front': _art('''
        ..usssT..
        .uttdttTT
        .uttdatTT
        .uttdttTT
        .uttdatTT
        .uttdttTT
        .uttdatTT
        .dTTdTTTd
        .uttdttTT
        .uttdatTT
        .uttdttTT
        .uttdttTT
        .TTTdTTTd
    '''),
    'back': _art('''
        ..utttT..
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .uttttttT
        .dTTTTTTd
        .uttttttT
        .uttttttT
        .utttdttT
        .utttdttT
        .TTTTdTTd
    '''),
}


@lru_cache(maxsize=None)
def _torso_cached(top, view):
    if top == 'coat':
        return _COAT[view]
    a = _TORSO[view].copy()
    S = lambda x, y, l: _set(a, x, y, l)  # noqa: E731
    if view == 'side':
        if top == 'hoodie':
            # hood bunched at the back of the neck, drawstrings, kangaroo pocket
            S(0, 0, 'T'); S(1, 0, 'u'); S(0, 1, 'T'); S(0, 2, 'T'); S(1, 1, 'u')
            S(6, 1, 'a'); S(6, 2, 'a'); S(6, 3, 'A'); S(7, 2, 'T')
            S(5, 5, 'T'); S(6, 5, 'T'); S(7, 5, 'T'); S(5, 6, 'T')
            for x in range(1, 8):
                S(x, 7, 'T')
            S(1, 7, 'd')
        elif top == 'pj':
            S(5, 0, 'a'); S(6, 1, 'a'); S(5, 1, 'A')
            for y in range(2, 7):
                S(2, y, 'a'); S(5, y, 'a')
            S(7, 3, 'A'); S(7, 5, 'A')
            for x in range(1, 8):
                S(x, 7, 'T')
        elif top in ('tee', 'polo', 'tank', 'sport'):
            if top == 'polo':
                S(5, 0, 'a'); S(6, 1, 'a'); S(4, 0, 'a'); S(6, 2, 'A')
            elif top == 'tank':
                S(2, 0, 's'); S(3, 0, 'u'); S(4, 0, 'u'); S(5, 0, 's'); S(1, 1, 's'); S(6, 1, 's')
            elif top == 'sport':
                S(5, 0, 'a'); S(1, 3, 'a'); S(1, 4, 'a'); S(1, 5, 'a')
            else:
                S(5, 0, 'T'); S(6, 1, 'T')
            for x in range(1, 8):
                S(x, 7, 'T')
        elif top == 'sweater':
            S(4, 0, 'a'); S(5, 0, 'a'); S(6, 1, 'A')
            for y in (2, 4, 6):
                S(2, y, 'T')
            for x in range(1, 8):
                S(x, 7, 'd')
        elif top == 'cardigan':
            S(5, 0, 'a'); S(6, 1, 'a'); S(7, 2, 'a'); S(7, 3, 'a'); S(7, 4, 'a'); S(7, 5, 'a')
            S(7, 6, 'a'); S(6, 2, 'T'); S(6, 5, 'd')
            for x in range(1, 8):
                S(x, 7, 'T')
        elif top == 'jacket':
            S(1, 0, 'T'); S(5, 0, 'a'); S(6, 1, 'a'); S(7, 2, 'd'); S(7, 3, 'd')
            S(5, 4, 'T'); S(6, 4, 'T')
            for x in range(1, 8):
                S(x, 7, 'a')
        else:  # longsleeve
            S(5, 0, 'T')
            for x in range(1, 8):
                S(x, 7, 'T')
    else:
        front = view == 'front'
        if top == 'hoodie':
            if front:
                S(3, 0, 'd'); S(4, 0, 'd'); S(5, 0, 'd')
                S(3, 1, 'a'); S(5, 1, 'a'); S(3, 2, 'a'); S(5, 2, 'A')
                for x in range(2, 7):
                    S(x, 5, 'T')
                S(2, 6, 'T'); S(6, 6, 'T')
            else:
                for x in range(2, 7):
                    S(x, 0, 'T')
                S(3, 1, 'T'); S(4, 1, 'T'); S(5, 1, 'T'); S(4, 2, 'd'); S(3, 2, 'T'); S(5, 2, 'T')
                S(1, 0, 'u'); S(7, 0, 'T')
            for x in range(1, 9):
                S(x, 7, 'T')
        elif top == 'pj':
            if front:
                S(3, 0, 'a'); S(5, 0, 'a'); S(4, 1, 'A'); S(2, 1, 'a'); S(6, 1, 'a')
                for y in range(2, 7):
                    S(4, y, 'd') if y % 2 else S(4, y, 'T')
            for y in range(2 if front else 1, 7):
                S(2, y, 'a'); S(6, y, 'a')
            for x in range(1, 9):
                S(x, 7, 'T')
        elif top in ('tee', 'polo', 'tank', 'sport'):
            if top == 'polo':
                if front:
                    S(2, 0, 'a'); S(3, 0, 'a'); S(5, 0, 'a'); S(6, 0, 'a'); S(4, 0, 'T')
                    S(3, 1, 'a'); S(5, 1, 'a'); S(4, 1, 'd'); S(4, 2, 'd')
                else:
                    for x in range(2, 7):
                        S(x, 0, 'a')
            elif top == 'tank':
                S(2, 0, 's'); S(3, 0, 's'); S(4, 0, 's'); S(5, 0, 's'); S(6, 0, 's')
                S(1, 1, 's'); S(7, 1, 's'); S(1, 2, 's'); S(7, 2, 's')
                if front:
                    S(3, 1, 's'); S(4, 1, 'S'); S(5, 1, 's')
                    S(2, 1, 'u'); S(6, 1, 'T')
            elif top == 'sport':
                if front:
                    S(3, 0, 's'); S(4, 0, 's'); S(5, 0, 's'); S(4, 1, 'S')
                for y in range(1, 7):
                    S(1, y, 'a'); S(8, y, 'A')
            else:
                if front:
                    S(3, 0, 'T'); S(4, 0, 's'); S(5, 0, 'T'); S(4, 1, 'T')
            for x in range(1, 9):
                S(x, 7, 'T')
        elif top == 'sweater':
            if front:
                S(3, 0, 'a'); S(4, 0, 'a'); S(5, 0, 'a'); S(4, 1, 'A')
                for x in (2, 4, 6):
                    S(x, 3, 'T'); S(x + 1, 4, 'T')
            for x in range(1, 9):
                S(x, 7, 'd')
        elif top == 'cardigan':
            if front:
                for y in range(0, 7):
                    S(3, y, 'a'); S(4, y, 'a' if y < 2 else 'A'); S(5, y, 'a')
                S(4, 0, 's'); S(3, 0, 'a')
                S(2, 3, 'd'); S(6, 3, 'd'); S(2, 5, 'd'); S(6, 5, 'd')
            for x in range(1, 9):
                S(x, 7, 'T')
        elif top == 'jacket':
            if front:
                S(3, 0, 'a'); S(5, 0, 'a'); S(4, 0, 'A')
                for y in range(1, 7):
                    S(4, y, 'd')
                S(2, 4, 'T'); S(6, 4, 'T')
            for x in range(1, 9):
                S(x, 7, 'a')
        else:
            if front:
                S(3, 0, 'T'); S(4, 0, 's'); S(5, 0, 'T')
            for x in range(1, 9):
                S(x, 7, 'T')
    return a


# --- hands (side art points RIGHT; top-left goes on the hand point) -------------------
_HANDS = {
    'fist':  _art('''
        sl
        sS
    '''),
    'open': _art('''
        ls.
        sss
        .S.
    '''),
    'point': _art('''
        slss
        sS..
    '''),
    'up': _art('''
        .l
        ss
        sS
    '''),
    'thumb': _art('''
        l.
        ss
        sS
    '''),
    'grip': _art('''
        ls
        SS
    '''),
}

# --- shoes: side view facing right; top-left at (ankle_x - 1, ankle_y + 1) -----------
_SHOES_SIDE = {
    'flat': _art('''
        fffi.
        ggggg
    '''),
    'toe': _art('''
        ff...
        ffff.
        ..ggg
    '''),
    'heel': _art('''
        ..ff
        ffff
        ggg.
    '''),
    'tip': _art('''
        ff..
        fff.
        .gg.
    '''),
    'lift': _art('''
        fffi
        gggg
    '''),
}
_SHOES_FRONT = _art('''
    fif
    ggg
''')
_SHOES_BACK = _art('''
    fff
    ggg
''')
_FEET_SIDE = {   # bare feet / socks without shoes (same anchors)
    'flat': _art('''
        sss..
        sSSSS
    '''),
}


# =====================================================================================
# LOOKS — material colours + which parts a character wears
# =====================================================================================
def _hero_pal():
    H = HERO_COLORS
    return {'o': 'ink', 'h': H['hair_dk'], 'H': H['hair'], 'j': H['hair_hi'],
            'k': H['skin_dk'], 'S': H['skin_sh'], 's': H['skin'], 'l': H['skin_hi'],
            'e': H['eye'], 'w': H['eye_hi'], 'b': H['blush'], 'm': H['mouth'], 'n': H['mouth_in']}


_CASUAL = {'d': '#083f47', 'T': 'deep_teal', 't': 'teal', 'u': 'aqua',
           'a': 'paper', 'A': 'stone',
           'D': '#23243c', 'P': 'navy', 'p': 'indigo', 'q': '#5d6094',
           'F': 'stone', 'f': 'white', 'g': 'mist', 'i': 'teal'}
_OUTFIT_PAL = {
    'pajamas': {'d': 'purple', 'T': 'violet', 't': 'lavender', 'u': '#c2a7f7',
                'a': 'lilac', 'A': '#cf93d9',
                'D': 'purple', 'P': 'violet', 'p': 'lavender', 'q': '#c2a7f7',
                'F': 'rose', 'f': 'pink', 'g': 'raspberry', 'i': 'blush'},
    'casual': _CASUAL,
    'gym': {'d': 'brick', 'T': 'vermilion', 't': 'coral', 'u': 'peach',
            'a': 'paper', 'A': 'stone',
            'D': '#202326', 'P': 'charcoal', 'p': '#474d52', 'q': '#62696e',
            'F': 'stone', 'f': 'white', 'g': 'mist', 'i': 'paper',
            'X': 'paper', 'Z': 'stone'},
    'lab': dict(_CASUAL, Y='#5a3a1c', y='amber', z='butter'),
    'apron': dict(_CASUAL, x='#8a5a12', R='#d29a1c', r='gold', Q='butter'),
}
_BACKPACK_PAL = {'v': '#5a2a24', 'C': 'rust', 'c': 'copper', 'K': 'clay', 'V': '#6e3a2a'}

# body: top style, sleeve length, legs ('long'/'shorts'), feet ('sneaker'/'slipper'/'bare')
_OUTFIT_PARTS = {
    'pajamas': dict(top='pj', sleeves='long', legs='long', feet='slipper', acc=()),
    'casual': dict(top='hoodie', sleeves='long', legs='long', feet='sneaker', acc=()),
    'gym': dict(top='tee', sleeves='short', legs='shorts', feet='sneaker', acc=('sweatband', 'socks')),
    'lab': dict(top='hoodie', sleeves='long', legs='long', feet='sneaker', acc=('goggles',)),
    'apron': dict(top='hoodie', sleeves='long', legs='long', feet='sneaker', acc=('apron',)),
}


class _Look:
    """Everything needed to build frames for one character look."""
    __slots__ = ('key', 'hair', 'top', 'sleeves', 'legs', 'feet', 'acc', 'lut', 'tall', 'blink')

    def __init__(self, key, hair, parts, pal):
        self.key = key
        self.hair = hair
        self.top = parts['top']
        self.sleeves = parts['sleeves']
        self.legs = parts['legs']
        self.feet = parts['feet']
        self.acc = tuple(parts.get('acc', ()))
        self.tall = int(parts.get('tall', 0))
        self.blink = int(parts.get('blink', 0))     # blink phase offset (frames)
        lut = np.zeros((256, 4), np.uint8)
        for ch, c in pal.items():
            if ch in _C:
                lut[_C[ch], :3] = col(c)
                lut[_C[ch], 3] = 255
        self.lut = lut


@lru_cache(maxsize=None)
def _hero_look(outfit, backpack):
    if outfit not in _OUTFIT_PARTS:
        raise ValueError(f'unknown outfit {outfit!r}; choose from {OUTFITS}')
    pal = dict(_hero_pal())
    pal.update(_OUTFIT_PAL[outfit])
    parts = dict(_OUTFIT_PARTS[outfit])
    if backpack:
        pal.update(_BACKPACK_PAL)
        parts['acc'] = tuple(parts['acc']) + ('backpack',)
    return _Look(('hero', outfit, backpack), 'hero', parts, pal)


# =====================================================================================
# HEAD builder (cached): hair style + face features + accessories
# =====================================================================================
_EXPR = {   # expr -> (eyes, mouth, blush)
    None: ('open', 'smile', True), 'happy': ('open', 'smile', True),
    'joy': ('joy', 'grin', True), 'sleepy': ('half', 'flat', False),
    'focus': ('focus', 'flat', False), 'surprised': ('wide', 'O', False),
    'chew': ('open', 'chew', True), 'tired': ('half', 'frown', False),
}
EXPRESSIONS = tuple(k for k in _EXPR if k)


@lru_cache(maxsize=None)
def _head(style, view, eyes, mouth, blush, acc):
    """Head label art; side view faces right."""
    a = _HEADS[style][view].copy()
    F = _face_pos(style, view)
    if view == 'side':
        ex, ey = F['eye']
        for (x, y) in F['ear']:
            _set(a, x, y, 'S' if (x, y) != F['ear'][-1] else 'k')
        if blush:
            for bx, by in F['blush']:
                _set(a, bx, by, 'b')
        _side_eye(a, ex, ey, eyes)
        _side_mouth(a, *F['mouth'], mouth)
    elif view == 'front':
        for i, (ex, ey) in enumerate(F['eyes']):
            _front_eye(a, ex, ey, eyes, i == 0)
        if blush:
            for bx, by in F['blush']:
                _set(a, bx, by, 'b')
        _front_mouth(a, *F['mouth'], mouth)
    for item in acc:
        fn = _HEAD_ACC.get(item)
        if fn:
            fn(a, view, F)
    return a


def _acc_goggles(a, view, F):
    if view == 'side':
        ex, ey = F['eye']
        for x in range(0, ex - 1):
            if _get(a, x, ey) in 'hHj' or _get(a, x, ey + 1) in 'hHj':
                _set(a, x, ey, 'Y')
        _set(a, ex - 1, ey - 1, 'Y'); _set(a, ex - 1, ey, 'Y'); _set(a, ex - 1, ey + 1, 'Y')
        _set(a, ex - 1, ey + 2, 'Y')
        for x in range(ex, ex + 3):
            _set(a, x, ey - 1, 'Y'); _set(a, x, ey + 2, 'Y')
            _set(a, x, ey, 'y'); _set(a, x, ey + 1, 'y')
        _set(a, ex, ey, 'z')
        _set(a, ex + 1, ey, 'e') if False else None
        _set(a, ex + 1, ey + 1, 'Y')
        _set(a, ex + 3, ey, 'Y'); _set(a, ex + 3, ey + 1, 'Y')
    elif view == 'front':
        (lx, ly), (rx, ry) = F['eyes']
        for x in range(lx - 1, rx + 3):
            _set(a, x, ly - 1, 'Y'); _set(a, x, ly + 2, 'Y')
        for x in range(lx - 1, rx + 3):
            for y in (ly, ly + 1):
                _set(a, x, y, 'y')
        _set(a, lx - 2, ly, 'Y'); _set(a, lx - 2, ly + 1, 'Y'); _set(a, rx + 3, ly, 'Y'); _set(a, rx + 3, ly + 1, 'Y')
        _set(a, 6, ly, 'Y'); _set(a, 6, ly + 1, 'Y')
        _set(a, lx, ly, 'z'); _set(a, rx, ly, 'z')
        _set(a, lx + 1, ly + 1, 'Y'); _set(a, rx + 1, ly + 1, 'Y')
    else:
        for x in range(0, a.shape[1]):
            if _get(a, x, 7 + F['ytop']) in 'hHjS':
                _set(a, x, 7 + F['ytop'], 'Y')


def _acc_sweatband(a, view, F):
    if view == 'side':
        row = 4 + F['ytop']
        for x in range(0, a.shape[1]):
            if _get(a, x, row) in 'hHj':
                _set(a, x, row, 'X')
            if _get(a, x, row + 1) in 'hHjslS':
                _set(a, x, row + 1, 'Z' if x < 3 else 'X')
    else:
        row = 4 + F['ytop']
        for x in range(0, a.shape[1]):
            if _get(a, x, row) in 'hHj':
                _set(a, x, row, 'X')
            if _get(a, x, row + 1) in 'hHj':
                _set(a, x, row + 1, 'Z')


def _acc_glasses(a, view, F):
    """Thin frames: dark top rims + temple arms ('G'), pale glass glints ('E')."""
    if view == 'side':
        ex, ey = F['eye']
        ax = F['ear'][0][0] + 2 if F['ear'] else ex - 4
        for x in range(ax, ex):
            _set(a, x, ey, 'G')
        for x in range(ex, ex + 3):
            _set(a, x, ey - 1, 'G')
        _set(a, ex + 2, ey, 'E'); _set(a, ex + 2, ey + 1, 'G')
    elif view == 'front':
        for x, y in F['eyes']:
            for xx in range(x - 1, x + 3):
                _set(a, xx, y - 1, 'G')
            _set(a, x - 1, y, 'E'); _set(a, x + 2, y + 1, 'E')
            _set(a, x - 1, y + 1, 'G'); _set(a, x + 2, y, 'G')
        (lx, ly), (rx, ry) = F['eyes']
        for xx in range(lx + 3, rx - 1):
            _set(a, xx, ly - 1, 'G')
        _set(a, lx - 2, ly, 'G'); _set(a, rx + 3, ly, 'G')


_SKINLAB = 'sSlkb'


def _acc_beard(a, view, F):
    """Full beard + moustache in the hair colour (mouth stays as a dark line)."""
    def hairify(x, y, lab='H'):
        if _get(a, x, y) in _SKINLAB + 'm':
            _set(a, x, y, lab)
    if view == 'side':
        mx, my = F['mouth']
        ex0 = F['ear'][0][0] if F['ear'] else mx - 6
        for y in range(my - 1, a.shape[0]):
            for x in range(ex0 + 1, a.shape[1]):
                if y == my - 1 and not (x <= ex0 + 2 or x >= mx - 1):
                    continue
                if y == a.shape[0] - 1 and x <= ex0 + 2:
                    continue            # keep the neck
                hairify(x, y, 'h' if (y == a.shape[0] - 1 or x == ex0 + 1) else
                        ('j' if (x + y) % 4 == 0 and y > my else 'H'))
        _set(a, mx, my, 'h' if _get(a, mx, my) != 'n' else 'n')
        _set(a, mx - 1, my, 'h')
        _set(a, mx - 1, my - 1, 'j')
    elif view == 'front':
        mx, my = F['mouth']
        w = a.shape[1]
        for y in range(my - 2, a.shape[0]):
            for x in range(w):
                edge = x <= 2 or x >= w - 3
                if y == my - 2 and not edge:
                    continue
                if y == my - 1 and not (edge or mx - 2 <= x <= mx + 2):
                    continue
                if y == a.shape[0] - 1 and not (mx - 2 <= x <= mx + 2):
                    continue
                hairify(x, y, 'h' if y == a.shape[0] - 1 else ('j' if (x * 3 + y) % 5 == 0 else 'H'))
        _set(a, mx, my, 'n' if _get(a, mx, my) == 'n' else 'h')
        _set(a, mx - 1, my, 'h'); _set(a, mx + 1, my, 'h')


_HEAD_ACC = {'goggles': _acc_goggles, 'sweatband': _acc_sweatband, 'glasses': _acc_glasses,
             'beard': _acc_beard}


# =====================================================================================
# LIMBS
# =====================================================================================
def _paint_limb(lay, m, base, shade, hi=None):
    """Fill a limb mask with base colour, shade on right & bottom edges and an optional
    highlight on the left edge (light from the upper-left)."""
    right = m & ~np.roll(m, -1, axis=1)
    bottom = m & ~np.roll(m, -1, axis=0)
    lay[m] = _C[base]
    if hi is not None:
        left = m & ~np.roll(m, 1, axis=1)
        lay[left & ~right] = _C[hi]
    lay[(right | bottom) & m] = _C[shade]


def _hint(h):
    return _HINTS[h] if isinstance(h, str) else h


def _arm(comp, lk, sh, spec, far=False, flip_hand=False, line=True, la=4.2, lb=4.2):
    """spec = (hand, kind, hint[, elbow]). The hand point is the top-left of the
    2x2 hand. Returns (hand, elbow)."""
    hand, kind, hint = spec[0], spec[1], spec[2]
    sh, hand = _ri(sh), _ri(hand)
    if len(spec) > 3 and spec[3] is not None:
        el = _ri(spec[3])
    else:
        el = _ri(_ik(sh, hand, la, lb, _hint(hint)))
    lay = comp.layer()
    mu = _seg_mask(sh, el, 2)
    mf = _seg_mask(el, hand, 2)
    sl = lk.sleeves
    if sl == 'long':
        _paint_limb(lay, mu | mf, 't', 'T')
        cuff = mf & _ring(_seg_mask(hand, hand, 2))
        lay[cuff] = _C['T']
    elif sl == 'short':
        _paint_limb(lay, mu | mf, 's', 'S')
        _paint_limb(lay, mu & _disc(sh, 2.3), 't', 'T')
    else:
        _paint_limb(lay, mu | mf, 's', 'S')
    if kind != 'none':
        _put(lay, _HANDS.get(kind, _HANDS['fist']), hand[0], hand[1], flip=flip_hand)
    comp.add(lay, line=line, far=far)
    return hand, el


@lru_cache(maxsize=4096)
def _disc_cached(cx, cy, r):
    yy, xx = np.mgrid[0:_CH, 0:_CW]
    return ((xx - (cx + _OX + 0.5)) ** 2 + (yy - (cy + _OY + 0.5)) ** 2) <= r * r


def _disc(c, r):
    return _disc_cached(int(c[0]), int(c[1]), float(r))


def _shoe_art(lk, foot, view):
    if view == 'front':
        art = _SHOES_FRONT
    elif view == 'back':
        art = _SHOES_BACK
    elif lk.feet == 'bare':
        art = _FEET_SIDE['flat']
    else:
        art = _SHOES_SIDE.get(foot, _SHOES_SIDE['flat'])
    if lk.feet == 'slipper':
        art = art.copy()
        art[art == _C['i']] = _C['f']
    return art


def _leg(comp, lk, root, spec, far=False, view='side', l1=4.2, l2=4.2):
    """spec = (ankle, foot, hint[, knee])."""
    ankle, foot = spec[0], spec[1]
    hint = spec[2] if len(spec) > 2 else ('fwd' if view == 'side' else 'out')
    root, ankle = _ri(root), _ri(ankle)
    if len(spec) > 3 and spec[3] is not None:
        kn = _ri(spec[3])
    else:
        kn = _ri(_ik(root, ankle, l1, l2, _hint(hint)))
    lay = comp.layer()
    mt = _seg_mask(root, kn, 3)
    ms = _seg_mask(kn, ankle, 3)
    if lk.legs == 'shorts':
        _paint_limb(lay, ms & ~mt, 's', 'S', 'l')
        _paint_limb(lay, mt, 'p', 'P', 'q')
    else:
        _paint_limb(lay, mt | ms, 'p', 'P', 'q')
        if lk.top == 'pj':
            lay[ms & ~np.roll(ms, -1, axis=0) & ~mt] = _C['a']
    if 'socks' in lk.acc:
        lay[ms & ~np.roll(ms, -1, axis=0) & ~mt] = _C['i']
    art = _shoe_art(lk, foot, view)
    if view == 'side':
        if foot == 'up':
            _put(lay, art, ankle[0] + 1, ankle[1] - 3)
        else:
            # shoes hang under the ankle; the 3-row tilted shoes (heel / toe / tip) need
            # the ankle lifted by >= 1 px for the sole to touch the floor line
            _put(lay, art, ankle[0] - 1, ankle[1] + 1)
    else:
        _put(lay, art, ankle[0] - 1, ankle[1] + 1)
    comp.add(lay, far=far)
    return kn


def _sheared(art, lean):
    """Shear torso art so that its top row is offset by `lean` px (bottom fixed)."""
    h, w = art.shape
    if lean == 0:
        return art, 0
    shifts = [int(round(lean * (h - 1 - r) / 9.0)) for r in range(h)]
    mn, mx = min(0, min(shifts)), max(0, max(shifts))
    out = np.zeros((h, w + mx - mn), np.uint8)
    for r in range(h):
        s = shifts[r] - mn
        out[r, s:s + w] = art[r]
    return out, mn


def _rotate_labels(L, deg, pivot):
    """Rotate a label canvas by `deg` (positive = clockwise on screen) about the
    origin-space pivot, nearest neighbour."""
    if deg % 360 == 0:
        return L
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    px, py = pivot[0] + _OX, pivot[1] + _OY
    yy, xx = np.mgrid[0:_CH, 0:_CW]
    dx, dy = xx - px, yy - py
    sx = np.rint(px + c * dx + s * dy).astype(int)
    sy = np.rint(py - s * dx + c * dy).astype(int)
    ok = (sx >= 0) & (sx < _CW) & (sy >= 0) & (sy < _CH)
    out = np.zeros_like(L)
    out[ok] = L[sy[ok], sx[ok]]
    return out


def _mirror(L):
    """Mirror a label canvas about the origin column."""
    out = np.zeros_like(L)
    src = L[:, ::-1]
    # column c of the result = column (2*_OX - c) of L
    shift = 2 * _OX - (_CW - 1)
    if shift >= 0:
        out[:, shift:] = src[:, :_CW - shift]
    else:
        out[:, :_CW + shift] = src[:, -shift:]
    return out


# =====================================================================================
# ACCESSORY LAYERS (apron, backpack)
# =====================================================================================
_APRON_SIDE = _art('''
    ..x.....
    ...x....
    ....rQ..
    ....rrrR
    ....rrrR
    xxxRRRRx
    ....rrrR
    ....rrrR
    ....rrrR
    ....rrrR
    ....rrrR
    .....RRR
''')   # top-left at (hip_x - 4, hip_y - 9) like the torso; bottom hangs over the thighs
_APRON_FRONT = _art('''
    .x.....x.
    ..x...x..
    ..QrrrR..
    ..rrrrR..
    ..rrrrR..
    xxRRRRRxx
    .rQrrrrR.
    .rrrrrrR.
    .rrRRRrR.
    .rrrrrrR.
    .rrrrrrR.
    .RRRRRRR.
''')
_APRON_BACK = _art('''
    .x.....x.
    ..x...x..
    ...x.x...
    ....x....
    ...x.x...
    xxxRxRxxx
    ...R.R...
''')
_PACK_SIDE = _art('''
    .vvvv.
    vKcccv
    vKcccC
    vccccC
    vKcccC
    vcvvcC
    vcKccC
    vccccC
    vCCCCC
    .vvvv.
''')   # top-left at (shoulder_x - 9, hip_y - 9)
_PACK_BACK = _art('''
    .vvvvvvv.
    vKcccccCv
    vKcccccCv
    vccvvvcCv
    vcKKcccCv
    vccccccCv
    vcvvvvcCv
    vcKcccCCv
    vCCCCCCCv
    .vvvvvvv.
''')


# =====================================================================================
# RIG RENDERERS
# =====================================================================================
def _face(R, expr):
    eyes, mouth, blush = _EXPR.get(expr if expr is not None else R.get('expr'), _EXPR[None])
    if R.get('eyes'):
        eyes = R['eyes']
    if R.get('mouth'):
        mouth = R['mouth']
    if R.get('blink') and eyes not in ('joy', 'closed', 'blink'):
        eyes = 'blink'
    return eyes, mouth, blush


def _head_layer(comp, lk, view, nx, ny, R, expr, add=True):
    eyes, mouth, blush = _face(R, expr)
    acc = tuple(a for a in lk.acc if a in _HEAD_ACC)
    h = _head(lk.hair, view, eyes, mouth, blush, acc)
    lay = comp.layer()
    xo, yd = _hmeta(lk.hair, view)
    hx0, hy0 = nx - _NECK[view] - xo, ny - (h.shape[0] - 1 - yd)
    _put(lay, h, hx0, hy0)
    if add:
        comp.add(lay)
    return hx0, hy0, lay


def _render_side(comp, lk, R, expr, legs=True, upper=True):
    hx, hy = R['hip']
    lean = R.get('lean', 0)
    br = R.get('breath', 0)
    sh_lean = int(round(lean * 0.8))
    shy = hy - 8 + br + R.get('shy', 0)
    shn = (hx + sh_lean - 1, shy)
    shf = (hx + sh_lean, shy)
    an, af = R['arm_n'], R['arm_f']
    lroot = R.get('lroot', 2)
    llen = R.get('llen', (4.2, 4.2))
    hf = hn = (0, 0)
    if upper:
        hf, _ = _arm(comp, lk, shf, af, far=True)
    if legs:
        ln, lf = R['leg_n'], R['leg_f']
        _leg(comp, lk, (hx - 1, hy - lroot), lf, far=True, l1=llen[0], l2=llen[1])
        _leg(comp, lk, (hx, hy - lroot), ln, l1=llen[0], l2=llen[1])
    if not upper:
        return
    # torso
    lay = comp.layer()
    t = _torso_art(lk.top, 'side')[br:]
    t, mn = _sheared(t, lean)
    _put(lay, t, hx - 4 + mn, hy - 9 + br)
    if 'badge' in lk.acc:
        bx = hx + 1 + int(round(lean * 0.4))
        _px(lay, bx, hy - 6 + br, 'B'); _px(lay, bx + 1, hy - 6 + br, 'B')
    comp.add(lay)
    if 'apron' in lk.acc:
        lay = comp.layer()
        ap, mn = _sheared(_APRON_SIDE[br:], lean)
        _put(lay, ap, hx - 4 + mn, hy - 9 + br)
        comp.add(lay)
    if 'backpack' in lk.acc:
        lay = comp.layer()
        _put(lay, _PACK_SIDE, hx + sh_lean - 9, hy - 9 + br + R.get('pack_dy', 0))
        comp.add(lay)
        lay = comp.layer()
        for k, y in enumerate(range(hy - 9 + br, hy - 4 + br)):
            _px(lay, hx + sh_lean - 3 + min(k, 2), y, 'V')
        comp.add(lay, line=False)
    # head
    hd = R.get('head', (0, 0))
    nx, ny = hx + lean + hd[0], hy - 10 + br + hd[1]
    hx0, hy0, _ = _head_layer(comp, lk, 'side', nx, ny, R, expr)
    # near arm (in front of everything)
    hn, _ = _arm(comp, lk, shn, an)
    F = _face_pos(lk.hair, 'side')
    comp.anchors.update(
        hand=(hn[0] + 1, hn[1] + 1), hand2=(hf[0] + 1, hf[1] + 1),
        head=(nx, hy0 - 1), eye=(hx0 + F['eye'][0], hy0 + F['eye'][1] + 1),
        mouth=(hx0 + F['mouth'][0], hy0 + F['mouth'][1]), hip=(hx, hy),
        chest=(hx + sh_lean + 1, hy - 7 + br))


def _render_frontback(comp, lk, R, expr, back=False):
    hx, hy = R['hip']
    br = R.get('breath', 0)
    view = 'back' if back else 'front'
    # arm_n = screen-right arm, arm_f = screen-left arm
    an, af = R['arm_n'], R['arm_f']
    shr = (hx + 4, hy - 8 + br)
    shl = (hx - 5, hy - 8 + br)
    ln, lf = R['leg_n'], R['leg_f']
    lroot = R.get('lroot', 2)
    _leg(comp, lk, (hx + 2, hy - lroot), ln, view=view)
    _leg(comp, lk, (hx - 2, hy - lroot), lf, view=view)
    arms_behind = back and R.get('arms_behind', True)
    if arms_behind:
        hr, _ = _arm(comp, lk, shr, an, line=False)
        hl, _ = _arm(comp, lk, shl, af, flip_hand=True, line=False)
    lay = comp.layer()
    _put(lay, _torso_art(lk.top, view)[br:], hx - 4, hy - 9 + br)
    if 'badge' in lk.acc and not back:
        _px(lay, hx + 2, hy - 6 + br, 'B'); _px(lay, hx + 3, hy - 6 + br, 'B')
    comp.add(lay)
    if 'apron' in lk.acc:
        lay = comp.layer()
        _put(lay, (_APRON_BACK if back else _APRON_FRONT)[br:], hx - 4, hy - 9 + br)
        comp.add(lay, line=not back)
    if 'backpack' in lk.acc:
        lay = comp.layer()
        if back:
            _put(lay, _PACK_BACK, hx - 4, hy - 9 + br)
        else:
            for y in range(hy - 9 + br, hy - 3 + br):
                _px(lay, hx - 3, y, 'V'); _px(lay, hx + 3, y, 'V')
        comp.add(lay)
    hd = R.get('head', (0, 0))
    nx, ny = hx + hd[0], hy - 10 + br + hd[1]
    hx0, hy0, _ = _head_layer(comp, lk, view, nx, ny, R, expr)
    if not arms_behind:
        hl, _ = _arm(comp, lk, shl, af, flip_hand=True)
        hr, _ = _arm(comp, lk, shr, an)
    F = _face_pos(lk.hair, view)
    comp.anchors.update(
        hand=(hr[0] + 1, hr[1] + 1), hand2=(hl[0], hl[1] + 1), head=(nx, hy0 - 1),
        eye=(hx0 + F['eyes'][1][0], hy0 + F['eyes'][1][1] + 1),
        mouth=(hx0 + F['mouth'][0], hy0 + F['mouth'][1]), hip=(hx, hy),
        chest=(hx, hy - 7 + br))


def _render_bend(comp, lk, R, expr):
    """Side view with the upper body rotated about the hips by R['bend'] degrees
    (negative = leaning back). Legs are drawn unrotated."""
    _render_side(comp, lk, R, expr, legs=True, upper=False)
    up = _Comp()
    _render_side(up, lk, R, expr, legs=False, upper=True)
    deg = R['bend']
    hx, hy = R['hip']
    piv = (hx, hy - 1)
    lay = _rotate_labels(up.L, deg, piv)
    comp.add(lay)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    for k, (ax, ay) in up.anchors.items():
        dx, dy = ax - piv[0], ay - piv[1]
        comp.anchors[k] = (int(round(piv[0] + c * dx - s * dy)), int(round(piv[1] + s * dx + c * dy)))


def _render_lie(comp, lk, R, expr):
    """Lying on the back, head to the LEFT, face up. Built as a standing side pose
    rotated 90 deg counter-clockwise. Origin: x = hips, y = mattress top row."""
    st = _Comp()
    S = dict(R)
    S['view'] = 'side'
    _render_side(st, lk, S, expr)
    ys, xs = np.nonzero(st.L)
    sx, sy = xs - _OX, ys - _OY
    lx = sy + 10                    # standing hip row (-10) -> x = 0
    ly = -sx - 5                    # body's back (x=-3) -> row -2 (outline at -1)
    ok = (lx + _OX >= 0) & (lx + _OX < _CW) & (ly + _OY >= 0) & (ly + _OY < _CH)
    comp.L[ly[ok] + _OY, lx[ok] + _OX] = st.L[ys[ok], xs[ok]]
    for k, (ax, ay) in st.anchors.items():
        comp.anchors[k] = (ay + 10, -ax - 5)


def _stretch(comp, row, n):
    """Make the upper body n px taller (n < 0: shorter) by repeating / removing the
    label row at origin-space y=row; everything above moves up by n."""
    L = comp.L
    r = row + _OY
    out = np.zeros_like(L)
    if n > 0:
        out[:r + 1 - n] = L[n:r + 1]
        out[r + 1 - n:r + 1] = L[r]
        out[r + 1:] = L[r + 1:]
    else:
        k = -n
        out[k:r + 1] = L[:r + 1 - k]
        out[r + 1:] = L[r + 1:]
    comp.L = out
    comp.anchors = {k2: (x, y - n if y <= row else y) for k2, (x, y) in comp.anchors.items()}


def _render(lk, R, expr):
    comp = _Comp()
    v = R['view']
    if v == 'side':
        if R.get('bend'):
            _render_bend(comp, lk, R, expr)
        else:
            _render_side(comp, lk, R, expr)
    elif v == 'front':
        _render_frontback(comp, lk, R, expr)
    elif v == 'back':
        _render_frontback(comp, lk, R, expr, back=True)
    elif v == 'lie':
        _render_lie(comp, lk, R, expr)
    if lk.tall and v in ('side', 'front', 'back') and not R.get('bend'):
        _stretch(comp, R['hip'][1] - 5, lk.tall)
    if R.get('mirror'):
        comp.L = _mirror(comp.L)
        comp.anchors = {k: (-x, y) for k, (x, y) in comp.anchors.items()}
    dy = R.get('dy', 0)
    if dy:
        comp.L = np.roll(comp.L, dy, axis=0)
        comp.anchors = {k: (x, y + dy) for k, (x, y) in comp.anchors.items()}
    # nothing may sink below the floor / mattress line: the outline ring goes on row -1
    comp.L[_OY - 1:] = 0
    m = comp.L > 0
    comp.L[_ring(m)] = _C['o']
    for k, v2 in R.get('extra', {}).items():
        comp.anchors[k] = v2
    return comp


# =====================================================================================
# POSES — each pose function returns a rig dict for drawing index i.
# Coordinates are relative to the floor point (x, y) of the character; side
# poses face right (left = mirrored). Arms: (hand, hand_kind, elbow_hint[, elbow]);
# legs: (ankle, foot_sprite, knee_hint[, knee]). The hand point is the top-left
# of the 2x2 hand; the ankle point is the centre of the shin's last row.
# =====================================================================================
def _rig(view='side', hip=(0, -10), **kw):
    R = dict(view=view, hip=hip)
    if view == 'side':
        hy = hip[1]
        R.update(arm_n=((hip[0] - 1, hy), 'fist', 'back'), arm_f=((hip[0], hy), 'fist', 'back'),
                 leg_n=((1, -4), 'flat'), leg_f=((-1, -4), 'flat'))
    else:
        hy = hip[1]
        R.update(arm_n=((hip[0] + 5, hy), 'fist', 'out'), arm_f=((hip[0] - 6, hy), 'fist', 'in'),
                 leg_n=((2, -4), 'flat', 'out'), leg_f=((-2, -4), 'flat', 'in'))
    R.update(kw)
    return R


def _breath(i, n=30):
    """1 px breathing: shoulders/head drop for the second half of the cycle."""
    return 1 if (i % n) >= n // 2 else 0


# ---------------------------------------------------------------- locomotion
# walk: 12 drawings x 3 frames = 36 frames = 2 steps (one step per beat).
# Body advances _WALK_X[i] px within the cycle (18 px per cycle); planted feet
# stay fixed on the ground when the scene moves the character with travel().
_WALK_X = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]
_WALK_FOOT = [  # near foot: (x rel. to hip, lift, sprite)
    (4, 1, 'heel'), (3, 0, 'flat'), (1, 0, 'flat'), (0, 0, 'flat'), (-2, 0, 'flat'),
    (-3, 0, 'flat'), (-5, 1, 'toe'), (-4, 2, 'tip'), (-2, 3, 'lift'), (1, 3, 'lift'),
    (3, 2, 'lift'), (4, 1, 'heel')]
_WALK_HIP = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
_WALK_ARM = [-3, -3, -2, -1, 0, 2, 3, 3, 2, 1, 0, -2]


def _walk_legs(i, xoff=0):
    nf = _WALK_FOOT[i % 12]
    ff = _WALK_FOOT[(i + 6) % 12]
    ln = ((nf[0] + xoff, -4 - nf[1]), nf[2])
    lf = ((ff[0] - 1 + xoff, -4 - ff[1]), ff[2])
    return ln, lf


def _pose_walk(i):
    dip = _WALK_HIP[i]
    ln, lf = _walk_legs(i)
    sw = _WALK_ARM[i]
    hy = -10 + dip
    return _rig(hip=(0, hy), leg_n=ln, leg_f=lf,
                arm_n=((-1 + sw, hy - (abs(sw) >= 3)), 'fist', 'back'),
                arm_f=((0 - sw, hy - (abs(sw) >= 3)), 'fist', 'back'))


def _pose_carry(i):
    dip = _WALK_HIP[i]
    ln, lf = _walk_legs(i)
    hy = -10 + dip
    return _rig(hip=(0, hy), leg_n=ln, leg_f=lf,
                arm_n=((4, hy - 6), 'grip', 'down'), arm_f=((5, hy - 7), 'grip', 'down'))


def _pose_carry_idle(i):
    br = _breath(i)
    return _rig(breath=br, arm_n=((4, -16 + br), 'grip', 'down'), arm_f=((5, -17 + br), 'grip', 'down'))


def _pose_push(i):
    dip = _WALK_HIP[i]
    ln, lf = _walk_legs(i, xoff=-1)
    hy = -10 + dip
    return _rig(hip=(0, hy), leg_n=ln, leg_f=lf, lean=1,
                arm_n=((7, -18), 'grip', 'down'), arm_f=((8, -18), 'grip', 'down'))


# run: 6 drawings x 3 frames = 18 frames (2 steps per beat), 6 px per drawing
_RUN_FOOT = [(3, 0, 'flat'), (-3, 1, 'toe'), (-6, 3, 'tip'), (-3, 5, 'lift'), (2, 4, 'lift'),
             (5, 2, 'heel')]
_RUN_HIP = [-10, -11, -12, -10, -11, -12]
_RUN_ARM = [(-4, -14), (-2, -13), (1, -14), (4, -16), (2, -15), (-1, -14)]
_DASH_FOOT = [(4, 0, 'flat'), (-5, 1, 'toe'), (-8, 3, 'tip'), (-4, 6, 'lift'), (3, 5, 'lift'),
              (6, 2, 'heel')]


def _pose_run(i, dash=False):
    tab = _DASH_FOOT if dash else _RUN_FOOT
    nf, ff = tab[i % 6], tab[(i + 3) % 6]
    hy = _RUN_HIP[i % 6] + (1 if dash else 0)
    lean = 3 if dash else 1
    ln = ((nf[0], -4 - nf[1]), nf[2])
    lf = ((ff[0] - 1, -4 - ff[1]), ff[2])
    a, b = _RUN_ARM[i % 6], _RUN_ARM[(i + 3) % 6]
    k = 1.5 if dash else 1.0
    R = _rig(hip=(0, hy), leg_n=ln, leg_f=lf, lean=lean, llen=(4.4, 4.4),
             arm_n=((int(a[0] * k) + lean, hy + (a[1] + 10)), 'fist', 'down'),
             arm_f=((int(b[0] * k) + lean, hy + (b[1] + 10)), 'fist', 'down'))
    if dash:
        R['head'] = (1, 1)
        R['expr'] = 'focus'
    return R


def _pose_dash(i):
    return _pose_run(i, dash=True)


# ---------------------------------------------------------------- standing actions
def _pose_idle(i):
    br = _breath(i)
    return _rig(breath=br, arm_n=((-1, -10 + br), 'fist', 'back'), arm_f=((0, -10 + br), 'fist', 'back'))


def _pose_drink(i, seated=False):
    hy = -12 if seated else -10
    base = (7, hy - 6) if seated else (2, hy - 1)
    mouth = (5, hy - 12)
    path = [base, base, (4, hy - 8), mouth, mouth, mouth, mouth, mouth, mouth, (4, hy - 8), base, base]
    hand = path[i]
    sip = 3 <= i <= 8
    R = _rig(hip=(0, hy), arm_n=(hand, 'grip', 'down' if not sip else 'bback'),
             arm_f=(((8, hy - 6) if seated else (0, hy)), 'fist', 'back'))
    if sip:
        R.update(head=(-1, 0) if 4 <= i <= 7 else (0, 0), eyes='closed' if 4 <= i <= 7 else None,
                 mouth='sip')
    if seated:
        R.update(_SEAT_LEGS)
    return R


def _pose_reach(i):
    path = [(3, -15), (6, -17), (9, -19), (10, -19)]
    kinds = ['fist', 'fist', 'open', 'open']
    return _rig(lean=1 if i else 0, arm_n=(path[i], kinds[i], 'down'),
                arm_f=((-2, -11), 'fist', 'back'), leg_n=((2, -4), 'flat'), leg_f=((-2, -4), 'flat'))


_TALK_HAND = [(3, -15), (4, -16), (5, -17), (5, -17), (4, -16), (3, -15), (2, -14), (3, -15),
              (5, -16), (6, -17), (5, -16), (3, -15)]
_TALK_MOUTH = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]


def _pose_talk(i):
    R = _rig(arm_n=(_TALK_HAND[i], 'open', 'down'), arm_f=((0, -10), 'fist', 'back'),
             head=(0, 0 if i % 4 < 2 else 0))
    if _TALK_MOUTH[i]:
        R['mouth'] = 'talk'
    if i in (2, 3, 8, 9):
        R['head'] = (0, -0)
    return R


def _pose_point(i):
    """Arm straight forward under the chin (keeps the face readable); pokes on the beat."""
    hand = (9, -20) if i == 0 else (8, -20)
    R = _rig(lean=1, arm_n=(hand, 'point', 'down'), arm_f=((0, -10), 'fist', 'back'))
    if i in (0, 1):
        R['mouth'] = 'talk'
    return R


def _pose_think(i):
    tap = (i % 6) in (0, 1)
    R = _rig(arm_n=((4, -21 if tap else -20), 'fist', 'down'),
             arm_f=((2, -15), 'fist', 'down'), eyes='up', mouth='flat',
             breath=0)
    return R


_PICK = [  # (hip_y, lean, hand, kind)
    (-10, 0, (1, -11), 'fist'), (-8, 2, (3, -7), 'open'), (-6, 3, (5, -4), 'open'),
    (-6, 3, (5, -4), 'grip'), (-8, 2, (4, -9), 'grip'), (-10, 0, (3, -14), 'grip')]


def _pose_pickup(i):
    hy, lean, hand, kind = _PICK[i]
    return _rig(hip=(0, hy), lean=lean, head=(1 if lean > 2 else 0, 0), eyes='down' if lean else None,
                arm_n=(hand, kind, 'back'), arm_f=((hand[0] - 1, hand[1] - 1), 'fist', 'back'),
                leg_n=((2, -4), 'flat'), leg_f=((-1, -4), 'flat'))


_SWEEP_DX = [0, 1, 2, 3, 3, 2, 1, 0, -1, -2, -2, -1]


def _pose_sweep(i):
    d = _SWEEP_DX[i]
    n = (2 + d, -17)
    fh = (5 + d, -12)
    # broom handle line through both hands, down to the floor
    gx, gy = n[0] + 1, n[1] + 1
    hx2, hy2 = fh[0] + 1, fh[1] + 1
    t = (-1 - gy) / max(1, (hy2 - gy))
    bx = int(round(gx + (hx2 - gx) * t))
    return _rig(lean=1, arm_n=(n, 'grip', 'down'), arm_f=(fh, 'grip', 'down'), eyes='down',
                leg_n=((2, -4), 'flat'), leg_f=((-2, -4), 'flat'), extra={'broom': (bx, -1)})


def _pose_adjust(i):
    twist = i % 4
    kind = ('grip', 'fist', 'grip', 'thumb')[twist]
    return _rig(lean=2, head=(1, 1), eyes='down', expr='focus',
                arm_n=((8, -19), kind, 'down'), arm_f=((7, -17), 'fist', 'down'),
                leg_n=((2, -4), 'flat'), leg_f=((-1, -4), 'flat'))


_CURL = [(2, -18), (4, -16), (4, -12), (1, -9), (4, -12), (4, -16)]


def _pose_curl(i):
    return _rig(arm_n=(_CURL[i], 'grip', 'down', (0, -13)), arm_f=((0, -10), 'grip', 'back'),
                expr='focus' if i in (5, 0) else None)


_SQUAT_HY = [-10, -9, -8, -7, -6, -6, -6, -7, -8, -9, -10, -10]
_SQUAT_LEAN = [0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0, 0]


def _pose_squat(i):
    """Barbell back squat: hips sit back, knees forward, torso leans to keep balance."""
    hy = _SQUAT_HY[i]
    d = -10 - hy                       # depth 0..4
    hx = -((d + 1) // 2)               # hips travel back
    lean = _SQUAT_LEAN[i] + (1 if d >= 3 else 0)
    sy = hy - 8
    return _rig(hip=(hx, hy), lean=lean, arm_n=((hx + lean - 2, sy - 2), 'grip', 'bback'),
                arm_f=((hx + lean - 1, sy - 2), 'grip', 'bback'),
                leg_n=((2, -4), 'flat'), leg_f=((-1, -4), 'flat'),
                expr='focus' if hy <= -7 else None)


_STRETCH = [((1, -16), (1, -15)), ((0, -26), (2, -25)), ((-1, -34), (1, -33)), ((-1, -35), (1, -34)),
            ((-1, -35), (1, -34)), ((0, -35), (2, -34)), ((-1, -35), (1, -34)), ((-1, -35), (1, -34)),
            ((-1, -34), (1, -33)), ((0, -26), (2, -25)), ((1, -14), (0, -13)), ((-1, -10), (0, -10))]


def _pose_stretch(i):
    a, b = _STRETCH[i]
    up = 2 <= i <= 8
    R = _rig(hip=(0, -11 if up else -10), arm_n=(a, 'fist', 'back'), arm_f=(b, 'fist', 'back'),
             lean=-1 if up else 0)
    if up:
        R.update(eyes='joy', mouth='O', leg_n=((1, -5), 'tip'), leg_f=((-1, -5), 'tip'))
    return R


_CHEER = [  # (hip_y, dy, near hand, far hand, feet). Side view: the near arm goes up
    # behind the face (x <= 0), the far arm forward-up behind the head (hand peeks out).
    (-9, 0, (-2, -11), (-1, -11), 'flat'), (-10, -2, (0, -27), (3, -27), 'tip'),
    (-11, -5, (-1, -35), (5, -34), 'tip'), (-11, -6, (-1, -35), (5, -34), 'tip'),
    (-11, -3, (0, -34), (5, -33), 'tip'), (-9, 0, (0, -27), (3, -27), 'flat')]


def _pose_cheer(i):
    hy, dy, a, b, ft = _CHEER[i]
    lift = 0 if ft == 'flat' else 1
    k = 'up' if a[1] < -20 else 'fist'
    return _rig(hip=(0, hy), dy=dy, expr='joy', arm_n=(a, k, 'back'), arm_f=(b, k, 'down'),
                leg_n=((1, -4 - lift), ft), leg_f=((-1, -4 - lift), ft))


_WAVE = [(0, -34), (1, -35), (2, -35), (3, -34), (2, -35), (1, -35)]   # hand beside the head


def _pose_wave(i):
    return _rig(arm_n=(_WAVE[i], 'up', 'down'), arm_f=((0, -10), 'fist', 'back'),
                mouth='grin' if i in (1, 2) else None)


def _pose_highfive(i):
    path = [(4, -19), (3, -27), (3, -35), (3, -35), (3, -35), (3, -34)]
    return _rig(lean=1, arm_n=(path[i], 'up', 'down'), arm_f=((0, -10), 'fist', 'back'), expr='joy',
                leg_n=((3, -4), 'flat'), leg_f=((-2, -4), 'flat'))


def _pose_thumbsup(i):
    br = _breath(i, 12)
    return _rig(breath=br, arm_n=((4, -17 + br), 'thumb', 'down'), arm_f=((0, -10 + br), 'fist', 'back'),
                mouth='grin')


def _pose_tired(i):
    br = 1 if (i % 30) >= 15 else 0
    return _rig(lean=2, head=(1, 1 + br), expr='tired', breath=0,
                arm_n=((1, -9), 'fist', 'down'), arm_f=((2, -9), 'fist', 'down'))


# ---------------------------------------------------------------- seated
_SEAT_LEGS = dict(hip=(0, -12), lroot=1, llen=(5.0, 9.0),
                  leg_n=((5, -4), 'flat', 'up', (5, -13)), leg_f=((4, -4), 'flat', 'up', (4, -13)))


def _seat(**kw):
    R = _rig()
    R.update(_SEAT_LEGS)
    R.update(arm_n=((3, -15), 'fist', 'back'), arm_f=((4, -15), 'fist', 'back'))
    R.update(kw)
    return R


def _pose_sit(i):
    br = _breath(i)
    return _seat(breath=br, arm_n=((3, -15), 'fist', 'back'), arm_f=((4, -15), 'fist', 'back'))


_TYPE_N = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
_TYPE_F = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]


def _pose_sit_type(i):
    return _seat(lean=1, arm_n=((7, -18 - _TYPE_N[i]), 'fist', 'down'),
                 arm_f=((8, -18 - _TYPE_F[i]), 'fist', 'down'))


def _pose_sit_eat(i):
    mouth = (5, -24)
    path = [(7, -18), (7, -19), (6, -21), mouth, mouth, (6, -21), (7, -19), (7, -18), (7, -18),
            (7, -18), (7, -18), (7, -18)]
    R = _seat(arm_n=(path[i], 'grip', 'down'), arm_f=((8, -18), 'fist', 'down'))
    if i == 3:
        R['mouth'] = 'O'
    elif i >= 4:
        R['mouth'] = 'chew' if i % 2 == 0 else 'flat'
    return R


def _pose_sit_drink(i):
    return _pose_drink(i, seated=True)


def _pose_sit_stretch(i):
    a, b = _STRETCH[i]
    up = 2 <= i <= 8
    R = _seat(arm_n=((a[0], a[1] - 2), 'fist', 'back'), arm_f=((b[0], b[1] - 2), 'fist', 'back'),
              lean=-1 if up else 0)
    if up:
        R.update(eyes='joy', mouth='O')
    if i >= 10:
        R.update(arm_n=((3, -15), 'fist', 'back'), arm_f=((4, -15), 'fist', 'back'))
    return R


# ---------------------------------------------------------------- front view
def _F(**kw):
    return _rig('front', **kw)


def _pose_f_idle(i):
    br = _breath(i)
    return _F(breath=br, arm_n=((5, -10 + br), 'fist', 'out'), arm_f=((-6, -10 + br), 'fist', 'in'))


_FWAVE = [(8, -27), (9, -28), (10, -27), (9, -26), (8, -27), (7, -28)]


def _pose_f_wave(i):
    return _F(arm_n=(_FWAVE[i], 'up', 'out'), arm_f=((-6, -10), 'fist', 'in'),
              mouth='grin' if i in (1, 2) else None)


def _pose_f_cheer(i):
    hy, dy, a, b, ft = _CHEER[i]
    lift = 0 if ft == 'flat' else 1
    if i == 0:
        hr, hl = (6, -11), (-7, -11)
    elif i in (1, 5):
        hr, hl = (8, -22), (-9, -22)
    else:
        hr, hl = (9, -28), (-10, -28)
    return _F(hip=(0, hy), dy=dy, expr='joy', arm_n=(hr, 'up', 'out'), arm_f=(hl, 'up', 'in'),
              leg_n=((2, -4 - lift), 'flat', 'out'), leg_f=((-2, -4 - lift), 'flat', 'in'))


def _pose_f_stretch(i):
    up = 2 <= i <= 8
    if i in (0, 11):
        hr, hl = (5, -10), (-6, -10)
    elif i in (1, 9):
        hr, hl = (7, -22), (-8, -22)
    elif i == 10:
        hr, hl = (6, -14), (-7, -14)
    else:
        w = 1 if i in (4, 6) else 0
        hr, hl = (2 + w, -34), (-3 - w, -34)
    R = _F(hip=(0, -11 if up else -10), arm_n=(hr, 'fist', 'out'), arm_f=(hl, 'fist', 'in'))
    if up:
        R.update(eyes='joy', mouth='O')
    return R


def _pose_f_yawn(i):
    if 2 <= i <= 8:
        R = _F(arm_n=((2, -22), 'open', 'out'), arm_f=((-9, -25), 'fist', 'in'),
               eyes='closed', mouth='yawn')
    elif i in (1, 9):
        R = _F(arm_n=((5, -16), 'fist', 'out'), arm_f=((-8, -17), 'fist', 'in'), eyes='half')
    else:
        R = _F(eyes='half' if i >= 10 else None)
    return R


def _pose_f_drink(i):
    path = [(5, -12), (5, -12), (4, -17), (2, -21), (2, -21), (2, -21), (2, -21), (2, -21),
            (2, -21), (4, -17), (5, -12), (5, -12)]
    R = _F(arm_n=(path[i], 'grip', 'out'))
    if 3 <= i <= 8:
        R.update(eyes='closed' if 4 <= i <= 7 else None, mouth='sip')
    return R


_FTALK = [(5, -14), (6, -16), (7, -17), (7, -17), (6, -16), (5, -14), (5, -13), (5, -14),
          (7, -16), (8, -17), (7, -16), (5, -14)]


def _pose_f_talk(i):
    R = _F(arm_n=(_FTALK[i], 'up', 'out'))
    if _TALK_MOUTH[i]:
        R['mouth'] = 'talk'
    return R


def _pose_f_squat(i):
    hy = _SQUAT_HY[i]
    d = -10 - hy
    sy = hy - 8
    return _F(hip=(0, hy), arm_n=((6, sy - 2), 'grip', 'down'), arm_f=((-7, sy - 2), 'grip', 'down'),
              leg_n=((3, -4), 'flat', 'out'), leg_f=((-3, -4), 'flat', 'in'),
              expr='focus' if d >= 3 else None)


def _pose_f_curl(i):
    hr, hl = [(5, -17), (6, -15), (6, -12), (5, -9), (6, -12), (6, -15)][i], None
    hl = (-1 - hr[0], hr[1])
    return _F(arm_n=(hr, 'grip', 'down', (5, -13)), arm_f=(hl, 'grip', 'down', (-6, -13)),
              expr='focus' if i in (5, 0) else None)


def _pose_f_thumbsup(i):
    br = _breath(i, 12)
    return _F(breath=br, arm_n=((7, -18 + br), 'thumb', 'out'), mouth='grin')


def _pose_spin(i):
    arms = dict(arm_n=((6, -11), 'open', 'out'), arm_f=((-7, -11), 'open', 'in'))
    if i == 0:
        return _F(expr='joy', **arms)
    if i == 2:
        return _rig('back', **arms)
    R = _rig(arm_n=((3, -12), 'open', 'down'), arm_f=((-3, -12), 'open', 'down'), expr='joy')
    if i == 3:
        R['mirror'] = True
    return R


# ---------------------------------------------------------------- back view
def _B(**kw):
    return _rig('back', **kw)


def _pose_b_idle(i):
    br = _breath(i)
    return _B(breath=br, arm_n=((5, -10 + br), 'fist', 'out'), arm_f=((-6, -10 + br), 'fist', 'in'))


_STIR = [(6, -19), (7, -19), (7, -20), (6, -21), (5, -21), (5, -20)]


def _pose_b_cook(i):
    return _B(head=(0, 1), arm_n=(_STIR[i], 'grip', 'out'), arm_f=((-8, -18), 'grip', 'in'))


_CHOP = [(5, -18), (5, -18), (5, -20), (5, -23), (5, -23), (5, -21)]


def _pose_b_chop(i):
    return _B(head=(0, 1), arm_n=(_CHOP[i], 'grip', 'out'), arm_f=((-6, -18), 'fist', 'in'))


def _pose_b_work(i):
    a = [(5, -18), (5, -19), (4, -18), (5, -18), (6, -19), (5, -18)][i % 6]
    b = [(-6, -18), (-6, -18), (-7, -19), (-6, -18), (-6, -18), (-5, -19)][(i + 2) % 6]
    return _B(head=(0, 1), arm_n=(a, 'fist', 'out'), arm_f=(b, 'fist', 'in'))


def _pose_b_reach_up(i):
    path = [(6, -21), (6, -28), (5, -33), (5, -35), (5, -35), (5, -34)]
    up = i >= 2
    return _B(hip=(0, -11 if up else -10), arm_n=(path[i], 'open', 'out'),
              leg_n=((2, -4), 'flat', 'out'), leg_f=((-2, -4), 'flat', 'in'),
              arms_behind=False)


# ---------------------------------------------------------------- bed
def _pose_sleep(i):
    br = 1 if 6 <= (i % 24) < 18 else 0
    return dict(view='lie', hip=(0, -10), breath=0, eyes='closed', mouth='flat', head=(6, 0),
                arm_n=((0, -11), 'fist', 'back'), arm_f=((1, -11), 'fist', 'back'),
                leg_n=((1, -4), 'flat'), leg_f=((0, -4), 'flat'), lean=br, sleep_breath=br)


_TAP = [(1, -11), (1, -12), (3, -24), (4, -32), (4, -36), (5, -37), (2, -37), (5, -37), (2, -37),
        (3, -31), (1, -19), (1, -11)]   # standing coords; lying: -x is up, so x 5 -> 2 taps down


def _pose_tap(i):
    """Asleep, the far arm reaches back over the pillow (to a phone on the nightstand at
    the head side) and taps twice: on f 18 and f 24."""
    R = _pose_sleep(0)
    R['arm_f'] = (_TAP[i], 'point' if 4 <= i <= 8 else 'fist', 'back')
    if 5 <= i <= 8:
        R['eyes'] = 'half' if i in (6, 8) else 'closed'
        R['mouth'] = 'frown' if i in (6, 8) else 'flat'
    return R


_WAKE_SEQ = (
    # (bend, eyes, mouth, near hand, far hand)
    (90, 'closed', 'flat', None, None), (90, 'closed', 'flat', None, None),
    (90, 'half', 'flat', None, None), (60, 'half', 'flat', None, None),
    (30, 'half', 'flat', None, None), (0, 'half', 'flat', (4, -5), (5, -5)),
    (0, 'half', 'flat', (4, -5), (5, -5)), (0, 'closed', 'O', (0, -19), (2, -19)),
    (0, 'closed', 'yawn', (-1, -27), (1, -27)), (0, 'closed', 'yawn', (-1, -28), (2, -28)),
    (0, 'closed', 'yawn', (0, -28), (2, -27)), (0, 'closed', 'yawn', (-1, -28), (2, -28)),
    (0, 'closed', 'yawn', (0, -28), (2, -27)), (0, 'closed', 'O', (-1, -27), (1, -27)),
    (0, 'half', 'flat', (1, -16), (2, -15)), (0, 'half', 'flat', (4, -5), (5, -5)),
    (0, 'half', 'flat', (4, -15), (5, -5)), (0, 'closed', 'flat', (4, -15), (5, -5)),
    (0, 'half', 'flat', (4, -15), (5, -5)), (0, None, 'smile', (4, -5), (5, -5)),
    (0, None, 'smile', (4, -5), (5, -5)), (0, None, 'smile', (4, -5), (5, -5)),
    (0, None, 'grin', (4, -5), (5, -5)), (0, None, 'smile', (4, -5), (5, -5)))


def _pose_wake(i):
    bend, eyes, mouth, a, b = _WAKE_SEQ[i]
    if bend == 90:
        R = _pose_sleep(0)
        R.update(eyes=eyes, mouth=mouth)
        return R
    R = _rig(hip=(0, -2), lroot=1, eyes=eyes, mouth=mouth, expr='sleepy',
             leg_n=((10, -3), 'up', 'up', (5, -3)), leg_f=((9, -4), 'up', 'up', (4, -4)),
             arm_n=(a or (4, -5), 'fist', 'down'), arm_f=(b or (5, -5), 'fist', 'down'))
    if bend:
        R['bend'] = -bend
        R['arm_n'] = ((-1, -2), 'fist', 'back')
        R['arm_f'] = ((0, -2), 'fist', 'back')
    if eyes is None:
        R['expr'] = None
    return R


# =====================================================================================
# POSE TABLE: (view, pose) -> (function, drawings, frames per drawing, loops)
# =====================================================================================
_POSE_FN = {
    # side
    ('side', 'idle'): (_pose_idle, 30, 3, True),
    ('side', 'walk'): (_pose_walk, 12, 3, True),
    ('side', 'run'): (_pose_run, 6, 3, True),
    ('side', 'dash'): (_pose_dash, 6, 3, True),
    ('side', 'carry'): (_pose_carry, 12, 3, True),
    ('side', 'carry_idle'): (_pose_carry_idle, 30, 3, True),
    ('side', 'push'): (_pose_push, 12, 3, True),
    ('side', 'sit'): (_pose_sit, 30, 3, True),
    ('side', 'sit_type'): (_pose_sit_type, 12, 3, True),
    ('side', 'sit_eat'): (_pose_sit_eat, 12, 3, True),
    ('side', 'sit_drink'): (_pose_sit_drink, 12, 3, True),
    ('side', 'sit_stretch'): (_pose_sit_stretch, 12, 3, True),
    ('side', 'drink'): (_pose_drink, 12, 3, True),
    ('side', 'reach'): (_pose_reach, 4, 3, False),
    ('side', 'talk'): (_pose_talk, 12, 3, True),
    ('side', 'point'): (_pose_point, 6, 3, True),
    ('side', 'think'): (_pose_think, 12, 3, True),
    ('side', 'pickup'): (_pose_pickup, 6, 3, True),
    ('side', 'sweep'): (_pose_sweep, 12, 3, True),
    ('side', 'adjust'): (_pose_adjust, 12, 3, True),
    ('side', 'curl'): (_pose_curl, 6, 3, True),
    ('side', 'squat'): (_pose_squat, 12, 3, True),
    ('side', 'stretch'): (_pose_stretch, 12, 3, True),
    ('side', 'cheer'): (_pose_cheer, 6, 3, True),
    ('side', 'wave'): (_pose_wave, 6, 3, True),
    ('side', 'highfive'): (_pose_highfive, 6, 3, False),
    ('side', 'thumbsup'): (_pose_thumbsup, 12, 3, True),
    ('side', 'tired'): (_pose_tired, 30, 3, True),
    # front
    ('front', 'idle'): (_pose_f_idle, 30, 3, True),
    ('front', 'wave'): (_pose_f_wave, 6, 3, True),
    ('front', 'cheer'): (_pose_f_cheer, 6, 3, True),
    ('front', 'stretch'): (_pose_f_stretch, 12, 3, True),
    ('front', 'yawn'): (_pose_f_yawn, 12, 3, True),
    ('front', 'drink'): (_pose_f_drink, 12, 3, True),
    ('front', 'talk'): (_pose_f_talk, 12, 3, True),
    ('front', 'squat'): (_pose_f_squat, 12, 3, True),
    ('front', 'curl'): (_pose_f_curl, 6, 3, True),
    ('front', 'thumbsup'): (_pose_f_thumbsup, 12, 3, True),
    ('front', 'spin'): (_pose_spin, 4, 2, True),
    # back
    ('back', 'idle'): (_pose_b_idle, 30, 3, True),
    ('back', 'cook'): (_pose_b_cook, 6, 3, True),
    ('back', 'chop'): (_pose_b_chop, 6, 3, True),
    ('back', 'work'): (_pose_b_work, 12, 3, True),
    ('back', 'reach_up'): (_pose_b_reach_up, 6, 3, False),
    # bed (side view; 'right' = head on the LEFT)
    ('bed', 'sleep'): (_pose_sleep, 24, 3, True),
    ('bed', 'wake'): (_pose_wake, 24, 3, False),
    ('bed', 'tap'): (_pose_tap, 12, 3, False),
}
_BED_POSES = ('sleep', 'wake', 'tap')

_DESC = {
    'idle': 'standing, 1 px breathing (90-frame loop) + blink every 3 s',
    'walk': 'walk, 2 steps per cycle, a foot lands on every beat (f % 18 == 0)',
    'run': 'jog/run, 2 steps per beat, flight phase',
    'dash': 'fast run, strong lean (add speed lines in the scene)',
    'carry': 'walk holding something in front of the chest with both hands',
    'carry_idle': 'stand holding something in front of the chest',
    'push': 'walk pushing a handle at y-17 (trolley) with both hands',
    'sit': 'seated on a chair, hands on the lap, breathing',
    'sit_type': 'seated typing, hands on a desk (top y-16)',
    'sit_eat': 'seated, fork from plate (y-17) to mouth and chew; one bite per 36 frames',
    'sit_drink': 'seated sip: cup up to the mouth, eyes closed, back down',
    'sit_stretch': 'seated stretch, arms overhead (end of the work day)',
    'drink': 'standing sip, cup up to the mouth and down',
    'reach': 'one-shot: arm extends forward at counter height (y-18), then holds',
    'talk': 'gesturing hand, mouth moving',
    'point': 'arm straight forward at chest height (whiteboard / screen), pokes on every beat',
    'think': 'hand on chin, looking up, finger taps',
    'pickup': 'squat down to the floor and back up (one per beat); hand at floor on f 6-11',
    'sweep': 'holding a broom (handle through hand & hand2), sweeping; anchor "broom" = floor point',
    'adjust': 'hand at table height turning a small knob (optics), looking down',
    'curl': 'dumbbell curl with the near hand, top of the rep on the beat',
    'squat': 'barbell squat (bar across the shoulders at the hands); deepest f 12-20 (beat f 18)',
    'stretch': 'arms overhead on tiptoes, eyes closed',
    'cheer': 'jump with arms up, one jump per beat (airborne f 3-14, peak f 9)',
    'wave': 'raised hand waving (side: beside the head so the face stays visible)',
    'highfive': 'one-shot: hand raised high (palm up beside the head), then holds',
    'thumbsup': 'thumbs up, grinning',
    'tired': 'slouched, head down, slow breathing',
    'yawn': 'front: yawn with hand at the mouth, other arm stretching',
    'spin': 'front->side->back->side turn (outfit-change gag), 8-frame loop',
    'cook': 'back view: stirring a pan at counter height (one circle per beat)',
    'chop': 'back view: knife up/down, the chop lands on the beat',
    'work': 'back view: hands busy at bench/table height',
    'reach_up': 'back view, one-shot: reach up to a shelf/cupboard on tiptoe, then holds',
    'sleep': 'bed: lying on the back, head on the LEFT, eyes closed, 1 breath per bar',
    'wake': 'bed, one-shot over one bar: wakes, sits up, stretches + yawns, smiles',
    'tap': 'bed, one-shot: asleep, reaches back over the pillow and taps (alarm) on f 18 & 24',
}


def _facings_of(pose):
    fac = []
    if ('side', pose) in _POSE_FN or ('bed', pose) in _POSE_FN:
        fac += ['right', 'left']
    if ('front', pose) in _POSE_FN:
        fac.append('front')
    if ('back', pose) in _POSE_FN:
        fac.append('back')
    return tuple(fac)


def _build_poses():
    out = {}
    for (view, name), (fn, n, k, loop) in _POSE_FN.items():
        if name in out:
            continue
        out[name] = {'facings': _facings_of(name), 'frames': n, 'frame_len': k, 'loop': loop,
                     'cycle': n * k, 'desc': _DESC.get(name, '')}
    return out


POSES = _build_poses()

WALK_SPEED = _WALK_X and 18 / 36.0     # 0.5 px per frame
RUN_SPEED = 36 / 18.0                  # 2 px per frame
DASH_SPEED = 54 / 18.0                 # 3 px per frame
_TRAVEL = {'walk': (_WALK_X, 18, 12, 3), 'carry': (_WALK_X, 18, 12, 3), 'push': (_WALK_X, 18, 12, 3),
           'run': ([0, 6, 12, 18, 24, 30], 36, 6, 3), 'dash': ([0, 9, 18, 27, 36, 45], 54, 6, 3)}


def travel(pose, f):
    """Integer distance (px) a walking/running character covers by local frame f,
    locked to the drawings so planted feet never slide. Use x = x0 + travel('walk', f)
    (or x0 - travel(...) when facing left). 0 for poses that don't travel."""
    t = _TRAVEL.get(pose)
    if t is None:
        return 0
    tab, cyc, n, k = t
    i = int(f) // k
    return (i // n) * cyc + tab[i % n]


SPEEDS = {'walk': WALK_SPEED, 'carry': WALK_SPEED, 'push': WALK_SPEED, 'run': RUN_SPEED,
          'dash': DASH_SPEED}


# =====================================================================================
# NPCs
# =====================================================================================
_SHOES = {   # F shade, f base, g sole, i lace/sock accent
    'white':  {'F': 'stone', 'f': 'white', 'g': 'mist', 'i': 'grey'},
    'brown':  {'F': '#4a2e22', 'f': '#70452f', 'g': '#2e2224', 'i': '#8c5a3c'},
    'black':  {'F': '#15151a', 'f': '#2c2a32', 'g': '#101014', 'i': '#45434e'},
    'boots':  {'F': '#2a2026', 'f': '#4a3a40', 'g': '#1a141a', 'i': '#6a5058'},
}
_PANTS = {   # D line, P shade, p base, q highlight
    'khaki':    {'D': '#5c4b35', 'P': '#8a7552', 'p': '#b39a6c', 'q': '#cbb488'},
    'black':    {'D': '#17151c', 'P': '#28252f', 'p': '#3a3644', 'q': '#544e5e'},
    'grey':     {'D': '#34343c', 'P': '#4e4e58', 'p': '#6a6a76', 'q': '#868694'},
    'charcoal': {'D': '#202326', 'P': 'charcoal', 'p': '#474d52', 'q': '#62696e'},
    'olive':    {'D': '#3e3626', 'P': '#5e5238', 'p': '#7d6e4a', 'q': '#9a8a62'},
    'denim':    {'D': '#24345a', 'P': '#34487a', 'p': '#4d65b4', 'q': '#7088c8'},
    'plum':     {'D': '#2e2634', 'P': '#463a4e', 'p': '#5e5068', 'q': '#7a6a84'},
}

NPCS = {
    'sci_a':     'experimentalist: tall, curly ginger hair + beard, green sweater, chinos',
    'sci_b':     'theorist: long black hair in a low ponytail, plum-purple top, black jeans, boots',
    'sci_c':     'senior professor: older, grey horseshoe hair, glasses, brown cardigan',
    'student_a': 'student: dark-brown bob, sky-blue raincoat, mustard backpack',
    'student_b': 'student: raspberry beanie, denim jacket, olive chinos, sage backpack',
    'gym_a':     'gym-goer: tall, buzz cut, sky-blue tank top, black shorts, red trainers',
    'gym_b':     'gym-goer: blonde high ponytail, hot-pink sports top, leggings',
    'cashier':   'New World cashier: hair bun, red polo with a white name badge',
    'shopper_a': 'shopper: short, older, white curly perm, long rose coat',
    'shopper_b': 'shopper: red baseball cap, olive jacket, light-blue jeans',
}
# who -> look. parts: top (hoodie pj tee polo tank sport sweater cardigan jacket coat
# longsleeve), sleeves (long/short/none), legs (long/shorts), feet, acc, tall (+-px).
_NPC_DEF = {
    'sci_a': dict(
        hair='curly', hairc='ginger', skin='fair', shoes='brown', pants='khaki',
        parts=dict(top='sweater', sleeves='long', legs='long', feet='sneaker', acc=('beard',), tall=2),
        pal={'d': '#0d3b30', 'T': 'forest', 't': 'green', 'u': '#41a86f', 'a': 'paper', 'A': 'stone'}),
    'sci_b': dict(
        hair='ponytail', hairc='black', skin='tan', shoes='boots', pants='black',
        parts=dict(top='longsleeve', sleeves='long', legs='long', feet='sneaker', acc=(), tall=0),
        pal={'d': '#3b1d4a', 'T': '#5c3070', 't': '#7d4690', 'u': '#a066b0', 'X': 'lilac', 'Z': 'violet'}),
    'sci_c': dict(
        hair='balding', hairc='grey', skin='light', shoes='brown', pants='grey',
        parts=dict(top='cardigan', sleeves='long', legs='long', feet='sneaker', acc=('glasses',), tall=-1),
        pal={'d': '#3d2618', 'T': '#5c3b25', 't': '#7a5236', 'u': '#9c7048', 'a': '#d8e4ec', 'A': '#a9bccb',
             'G': '#4a3a42', 'E': '#d6eef4'}),
    'student_a': dict(
        hair='bob', hairc='brown', skin='tan', shoes='white', pants='black',
        parts=dict(top='jacket', sleeves='long', legs='long', feet='sneaker', acc=('backpack',), tall=-1),
        pal={'d': '#24427a', 'T': 'blue', 't': 'sky', 'u': 'light_sky', 'a': 'paper', 'A': 'stone',
             'i': 'sky', 'v': '#7a5410', 'C': '#c08a18', 'c': 'gold', 'K': 'butter', 'V': '#8a5a12'}),
    'student_b': dict(
        hair='beanie', hairc='black', skin='deep', shoes='white', pants='olive',
        parts=dict(top='jacket', sleeves='long', legs='long', feet='sneaker', acc=('backpack',), tall=1),
        pal={'d': '#1f2f52', 'T': '#30487a', 't': '#4a6aa6', 'u': '#7898cc', 'a': 'paper', 'A': 'stone',
             'X': 'raspberry', 'Z': 'berry', 'B': 'rose', 'i': 'red',
             'v': '#24302a', 'C': 'slate', 'c': 'sage', 'K': 'pale_sage', 'V': '#2c3a30'}),
    'gym_a': dict(
        hair='buzz', hairc='black', skin='deep', shoes='white', pants='charcoal',
        parts=dict(top='tank', sleeves='none', legs='shorts', feet='sneaker', acc=('socks',), tall=1),
        pal={'d': '#1f4f80', 'T': 'blue', 't': 'sky', 'u': 'light_sky', 'f': 'red', 'F': 'brick',
             'g': 'paper', 'i': 'white'}),
    'gym_b': dict(
        hair='high_pony', hairc='blonde', skin='fair', shoes='white', pants='black',
        parts=dict(top='sport', sleeves='none', legs='long', feet='sneaker', acc=(), tall=0),
        pal={'d': '#7a1e3c', 'T': 'magenta', 't': 'hot_pink', 'u': 'pink', 'a': 'white', 'A': 'stone',
             'X': 'hot_pink', 'Z': 'magenta', 'i': 'hot_pink'}),
    'cashier': dict(
        hair='bun', hairc='black', skin='brown', shoes='black', pants='black',
        parts=dict(top='polo', sleeves='short', legs='long', feet='sneaker', acc=('badge',), tall=0),
        pal={'d': '#5a1414', 'T': 'brick', 't': 'red', 'u': '#f26a5a', 'a': 'crimson', 'A': 'maroon',
             'B': 'white'}),
    'shopper_a': dict(
        hair='curly', hairc='white', skin='brown', shoes='brown', pants='plum',
        parts=dict(top='coat', sleeves='long', legs='long', feet='sneaker', acc=(), tall=-2),
        pal={'d': '#6e2a3e', 'T': 'raspberry', 't': 'rose', 'u': 'pink', 'a': 'blush', 'A': 'raspberry'}),
    'shopper_b': dict(
        hair='cap', hairc='brown', skin='light', shoes='white', pants='denim',
        parts=dict(top='jacket', sleeves='long', legs='long', feet='sneaker', acc=(), tall=0),
        pal={'d': 'dark_olive', 'T': 'olive', 't': '#8a8a46', 'u': 'moss', 'a': 'khaki', 'A': 'pale_sage',
             'X': 'red', 'Z': 'brick', 'B': 'coral', 'i': 'slate'}),
}


@lru_cache(maxsize=None)
def _npc_look(who):
    if who not in _NPC_DEF:
        raise ValueError(f'unknown NPC {who!r}; choose from {tuple(NPCS)}')
    d = _NPC_DEF[who]
    pal = {'o': 'ink', 'e': 'ink', 'w': 'white', 'n': 'maroon', 'B': 'white', 'G': 'ink', 'E': 'white'}
    pal.update(_SKIN[d['skin']])
    pal.update(_HAIRC[d['hairc']])
    pal['m'] = pal['k']
    pal.update({'X': pal['H'], 'Z': pal['h']})
    pal.update(_PANTS[d['pants']])
    pal.update(_SHOES[d['shoes']])
    pal.update(_BACKPACK_PAL)
    p = d['pal']
    pal.update({'a': p.get('t'), 'A': p.get('T')})       # accent defaults to the top colour
    pal.update(p)
    parts = dict(d['parts'])
    parts.setdefault('blink', 11 + 23 * list(_NPC_DEF).index(who))   # nobody blinks in sync
    return _Look(('npc', who), d['hair'], parts, pal)


_SKIN = {
    'fair':  {'k': '#b06a5a', 'S': '#d99a82', 's': '#f4c9a8', 'l': '#fde4cd', 'b': 'pink'},
    'light': {'k': 'skin_dk', 'S': 'skin_sh', 's': 'skin', 'l': 'skin_hi', 'b': 'salmon'},
    'tan':   {'k': '#7c4636', 'S': '#a96b4c', 's': '#c98f66', 'l': '#e2b086', 'b': '#d97a62'},
    'brown': {'k': '#56301f', 'S': '#7d4a33', 's': '#a06848', 'l': '#bf875f', 'b': '#b85a4a'},
    'deep':  {'k': '#3a2018', 'S': '#5a3326', 's': '#7a4a34', 'l': '#99654a', 'b': '#8f4a3c'},
}
_HAIRC = {
    'dark':   {'h': 'hair', 'H': 'hair_mid', 'j': 'hair_hi'},
    'black':  {'h': '#1c1620', 'H': '#2c2430', 'j': '#4a3f52'},
    'ginger': {'h': '#7a2e1c', 'H': 'copper', 'j': 'clay'},
    'grey':   {'h': '#6c6474', 'H': '#a29cab', 'j': '#d6d2dc'},
    'brown':  {'h': '#3a2418', 'H': '#5c3b25', 'j': '#82593a'},
    'blonde': {'h': '#a8742e', 'H': '#dcac4e', 'j': '#f4d983'},
    'white':  {'h': '#8c869a', 'H': '#c9c5d3', 'j': '#f0eef5'},
}


# =====================================================================================
# FRAME CACHE + API
# =====================================================================================
def _view_of(facing, pose):
    if pose in _BED_POSES:
        return 'bed'
    return 'side' if facing in ('right', 'left') else facing


def _blink(f):
    return (f % 90) in (84, 85, 86)


@lru_cache(maxsize=None)
def _build(look, pose, view, i, expr, blink):
    fn, n, k, loop = _POSE_FN[(view, pose)]
    R = fn(i)
    if blink:
        R['blink'] = True
    comp = _render(look, R, expr)
    L = comp.L
    ys, xs = np.nonzero(L)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = look.lut[L[y0:y1, x0:x1]]
    spr = Sprite(np.ascontiguousarray(rgba))
    return spr, int(x0 - _OX), int(y0 - _OY), dict(comp.anchors)


def _frame_index(view, pose, f):
    fn, n, k, loop = _POSE_FN[(view, pose)]
    i = f // k
    return (i % n) if loop else min(max(i, 0), n - 1)


def _check(pose, facing):
    if facing not in ('right', 'left', 'front', 'back'):
        raise ValueError(f"facing must be 'right', 'left', 'front' or 'back', not {facing!r}")
    view = _view_of(facing, pose)
    if view == 'bed' and facing in ('front', 'back'):
        raise ValueError(f"bed pose {pose!r} supports facing 'right' (head on the left) or 'left'")
    if (view, pose) not in _POSE_FN:
        if pose not in POSES:
            raise ValueError(f'unknown pose {pose!r}; choose from {sorted(POSES)}')
        raise ValueError(f'pose {pose!r} has no {facing!r} facing; it supports {POSES[pose]["facings"]}')
    return view


def _get_frame(look, pose, f, facing, expr):
    view = _check(pose, facing)
    f = int(f)
    i = _frame_index(view, pose, f)
    if expr is not None and expr not in _EXPR:
        raise ValueError(f'unknown expr {expr!r}; choose from {EXPRESSIONS}')
    return _build(look, pose, view, i, expr, _blink(f + look.blink) and pose not in ('sleep', 'tap'))


def _draw(c, x, y, look, pose, f, facing, expr):
    spr, ox, oy, _ = _get_frame(look, pose, f, facing, expr)
    if facing == 'left':
        c.blit(spr.flipped(), x - (ox + spr.w - 1), y + oy)
    else:
        c.blit(spr, x + ox, y + oy)


def _anchor(look, pose, f, facing, name):
    spr, ox, oy, anc = _get_frame(look, pose, f, facing, None)
    if name not in anc:
        raise ValueError(f'unknown anchor {name!r}; available: {sorted(anc)}')
    dx, dy = anc[name]
    return (-dx if facing == 'left' else dx, dy)


def _sprite(look, pose, f, facing, expr):
    spr, ox, oy, _ = _get_frame(look, pose, f, facing, expr)
    if facing == 'left':
        return spr.flipped(), -(ox + spr.w - 1), oy
    return spr, ox, oy


def draw_hero(c, x, y, pose='idle', f=0, facing='right', outfit='casual', backpack=False, expr=None):
    """Draw the hero. (x, y) = bottom-centre of the feet (y = floor line); seated
    poses: the floor point under the hips; bed poses: x = hips, y = mattress top."""
    _draw(c, int(x), int(y), _hero_look(outfit, bool(backpack)), pose, f, facing, expr)


def draw_npc(c, x, y, who, pose='idle', f=0, facing='right', expr=None):
    """Draw an NPC (see NPCS) with the same pose system and anchors as the hero."""
    _draw(c, int(x), int(y), _npc_look(who), pose, f, facing, expr)


def hero_anchor(pose, f=0, facing='right', name='hand', outfit='casual'):
    """(dx, dy) offset from the draw point (x, y) of a body point on this frame:
    'hand' (front/near hand), 'hand2' (far hand), 'head' (top of head), 'mouth',
    'eye', 'hip', 'chest' (+ pose extras such as 'broom')."""
    return _anchor(_hero_look(outfit, False), pose, f, facing, name)


def npc_anchor(who, pose, f=0, facing='right', name='hand'):
    return _anchor(_npc_look(who), pose, f, facing, name)


def hero_sprite(pose='idle', f=0, facing='right', outfit='casual', backpack=False, expr=None):
    """(Sprite, ox, oy): the frame as a Sprite whose top-left goes at (x+ox, y+oy).
    Use for silhouettes (.tinted()), outlines, reflections..."""
    return _sprite(_hero_look(outfit, bool(backpack)), pose, f, facing, expr)


def npc_sprite(who, pose='idle', f=0, facing='right', expr=None):
    return _sprite(_npc_look(who), pose, f, facing, expr)


def pose_frame(pose, f, facing='right'):
    """Drawing index used for local frame f (handy to sync props/SFX to drawings)."""
    return _frame_index(_view_of(facing, pose), pose, int(f))


def npc_poses(who):
    """{pose: facings} supported by NPC `who` (NPCs share the hero's pose system)."""
    _npc_look(who)
    return {p: d['facings'] for p, d in POSES.items()}
