"""Big title lettering for the film's title / end cards (owned by the intro director).

A hand-made 13-px pixel serif (A-Z, 0-9, . , ! ' - & : /; anything else = a space) rendered as gold
letters with a royal-purple outline and a deep-purple extruded shadow, plus a parchment cartouche
plate with scroll rolls. All functions draw on a Canvas and are pure functions of their arguments.

    from engine.title import draw_title, title_size, draw_cartouche, draw_subtitle, draw_title_card

    draw_title(c, 160, 12, 'EQUESTRIA')                       # static, centred on x=160, cap top y=12
    draw_title(c, 160, 12, 'EQUESTRIA', f=f, t0=144, step=9)  # letter k pops in at t0 + 9k (2-frame
                                                              # white flash, drops `drop` px, bounces)
    draw_title(..., shine=u)                                  # u 0..1: a diagonal glint sweeps across
    draw_title(..., scale=2)                                  # integer nearest upscale (26-px caps)
    title_size('EQUESTRIA') -> (w, h)                         # box incl. outline + extrusion (114, 17)
    draw_cartouche(c, cx, y, w, h, t=1.0)                     # parchment plate, t 0..1 unrolls it
                                                              # from the centre (rolls slide apart)
    draw_subtitle(c, cx, y, 'A PIXEL WORLD ATLAS', f, t0)     # 5x7 caps, typed on 2 f/char, centred,
                                                              # diamond ornaments when complete
    draw_title_card(c, f, cx=160, y=6, t0=0, title=..., subtitle=..., step=9, sub_t0=None,
                    shine_t0=None, shine_every=288, dy=0, drop=7, style='gold')
        # the intro look in one call: the plate unrolls over the 8 frames before t0, letters pop in
        # from t0 every `step` frames, the subtitle types on from sub_t0, a glint sweeps the title
        # from shine_t0, corner twinkles once the title has landed. Plate ~155 x 35 px for EQUESTRIA.
        # Static card: pass t0 well before f (e.g. t0=f-200). Returns the plate rect (x0, y0, w, h).

Styles (style=): 'gold' (default: cream/butter/gold/amber ramp, royal outline, eggplant extrusion),
'silver' (moonlit white/periwinkle, indigo outline: night end card), 'rose' (cotton/bubblegum, plum).
"""
import math
from functools import lru_cache

import numpy as np

from .core import W, H, Canvas, col, mix, seg, clamp, ease_out_back, ease_out_bounce, ease_in, hash01
from .font import text, text_width

CAP = 13          # cap height of the big font (rows 13-14 are descenders, e.g. the tail of Q)

_G = {
    'A': ['......#......',
          '.....###.....',
          '.....###.....',
          '....#####....',
          '....#.###....',
          '...##.###....',
          '...#...###...',
          '..##...###...',
          '..#########..',
          '.##.....###..',
          '.#.......###.',
          '.#.......###.',
          '###.....#####'],
    'B': ['#########...',
          '.###...###..',
          '.###....###.',
          '.###....###.',
          '.###...###..',
          '.#######....',
          '.###...###..',
          '.###....###.',
          '.###.....###',
          '.###.....###',
          '.###....###.',
          '.###...###..',
          '#########...'],
    'C': ['....#####.#.',
          '..###...###.',
          '.###.....##.',
          '.###......#.',
          '###.........',
          '###.........',
          '###.........',
          '###.........',
          '###.........',
          '.###......#.',
          '.###.....##.',
          '..###...##..',
          '....#####...'],
    'D': ['########....',
          '.###...###..',
          '.###....###.',
          '.###.....###',
          '.###.....###',
          '.###.....###',
          '.###.....###',
          '.###.....###',
          '.###.....###',
          '.###.....###',
          '.###....###.',
          '.###...###..',
          '########....'],
    'E': ['###########',
          '.###.....##',
          '.###......#',
          '.###.......',
          '.###...#...',
          '.###...#...',
          '.########..',
          '.###...#...',
          '.###...#...',
          '.###.......',
          '.###......#',
          '.###.....##',
          '###########'],
    'F': ['###########',
          '.###.....##',
          '.###......#',
          '.###.......',
          '.###...#...',
          '.###...#...',
          '.########..',
          '.###...#...',
          '.###...#...',
          '.###.......',
          '.###.......',
          '.###.......',
          '#####......'],
    'G': ['....#####.#.',
          '..###...###.',
          '.###.....##.',
          '.###......#.',
          '###.........',
          '###.........',
          '###...######',
          '###.....###.',
          '###.....###.',
          '.###....###.',
          '.###....###.',
          '..###..####.',
          '....####..#.'],
    'H': ['#####...#####',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '.###########.',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '.###.....###.',
          '#####...#####'],
    'I': ['#####',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '.###.',
          '#####'],
    'J': ['...#####',
          '....###.',
          '....###.',
          '....###.',
          '....###.',
          '....###.',
          '....###.',
          '....###.',
          '....###.',
          '##..###.',
          '##..###.',
          '.##.##..',
          '..###...'],
    'K': ['#####..#####',
          '.###....##..',
          '.###...##...',
          '.###..##....',
          '.###.##.....',
          '.#####......',
          '.######.....',
          '.###.###....',
          '.###..###...',
          '.###...###..',
          '.###....###.',
          '.###.....###',
          '#####...####'],
    'L': ['#####......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###......#',
          '.###.....##',
          '.###....###',
          '###########'],
    'M': ['####.......####',
          '.###.......###.',
          '.####.....####.',
          '.####.....####.',
          '.##.##...##.##.',
          '.##.##...##.##.',
          '.##..##.##..##.',
          '.##..##.##..##.',
          '.##...###...##.',
          '.##...###...##.',
          '.##....#....##.',
          '.##.........##.',
          '####.......####'],
    'N': ['####.....####',
          '.####.....##.',
          '.#####....##.',
          '.##.###...##.',
          '.##..###..##.',
          '.##..###..##.',
          '.##...###.##.',
          '.##...###.##.',
          '.##....#####.',
          '.##.....####.',
          '.##......###.',
          '.##.......##.',
          '####.......#.'],
    'O': ['....#####....',
          '..###...###..',
          '.###.....###.',
          '.###.....###.',
          '###.......###',
          '###.......###',
          '###.......###',
          '###.......###',
          '###.......###',
          '.###.....###.',
          '.###.....###.',
          '..###...###..',
          '....#####....'],
    'P': ['#########..',
          '.###...###.',
          '.###....###',
          '.###....###',
          '.###....###',
          '.###...###.',
          '.#######...',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '.###.......',
          '#####......'],
    'Q': ['....#####....',
          '..###...###..',
          '.###.....###.',
          '.###.....###.',
          '###.......###',
          '###.......###',
          '###.......###',
          '###.......###',
          '###.......###',
          '.###.....###.',
          '.###.....###.',
          '..###...###..',
          '....######...',
          '.......###...',
          '........####.'],
    'R': ['#########....',
          '.###...###...',
          '.###....###..',
          '.###....###..',
          '.###....###..',
          '.###...###...',
          '.#######.....',
          '.###.###.....',
          '.###..###....',
          '.###..###....',
          '.###...###...',
          '.###...###...',
          '#####...#####'],
    'S': ['...######.#',
          '.###....###',
          '###......##',
          '###.......#',
          '.###.......',
          '..#####....',
          '....#####..',
          '.......###.',
          '........###',
          '#.......###',
          '##......###',
          '###....###.',
          '#.######...'],
    'T': ['#############',
          '##...###...##',
          '#....###....#',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '.....###.....',
          '....#####....'],
    'U': ['#####...####.',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '.###.....##..',
          '..###...##...',
          '...#####.....'],
    'V': ['#####...####',
          '.###.....##.',
          '.###.....##.',
          '..###....#..',
          '..###...##..',
          '..###...#...',
          '...###..#...',
          '...###.##...',
          '....####....',
          '....###.....',
          '....###.....',
          '.....#......',
          '.....#......'],
    'W': ['####...#####...###',
          '.###....###.....#.',
          '.###....###.....#.',
          '..###...####...##.',
          '..###...####...#..',
          '..###..##.###..#..',
          '...###.#..###.##..',
          '...###.#...####...',
          '...#####...####...',
          '....###.....###...',
          '....###.....##....',
          '.....#.......#....',
          '.....#.......#....'],
    'X': ['#####...####',
          '.###.....#..',
          '..###...#...',
          '..###...#...',
          '...###.#....',
          '....###.....',
          '....###.....',
          '....####....',
          '...#.###....',
          '...#..###...',
          '..#...###...',
          '.#.....###..',
          '####..######'],
    'Y': ['#####...####',
          '.###.....#..',
          '..###...#...',
          '..###...#...',
          '...###.#....',
          '....####....',
          '.....###....',
          '.....###....',
          '.....###....',
          '.....###....',
          '.....###....',
          '.....###....',
          '....#####...'],
    'Z': ['###########',
          '##.....###.',
          '#.....###..',
          '.....###...',
          '.....###...',
          '....###....',
          '...###.....',
          '...###.....',
          '..###......',
          '.###......#',
          '.###.....##',
          '###.....###',
          '###########'],
    '0': ['...#####...', '.###...###.', '.###...###.', '###.....###', '###.....###', '###.....###',
          '###.....###', '###.....###', '###.....###', '###.....###', '.###...###.', '.###...###.',
          '...#####...'],
    '1': ['...###.', '..####.', '.#####.', '...###.', '...###.', '...###.', '...###.', '...###.',
          '...###.', '...###.', '...###.', '...###.', '.#######'],
    '2': ['..######..', '.##....###', '##.....###', '.......###', '......###.', '.....###..',
          '....###...', '...###....', '..###.....', '.###.....#', '###.....##', '##########',
          '##########'],
    '3': ['.#######..', '##....###.', '#......###', '.......###', '.......###', '......###.',
          '...#####..', '......###.', '.......###', '.......###', '#......###', '##....###.',
          '.#######..'],
    '4': ['......###..', '.....####..', '....#####..', '...##.###..', '..##..###..', '.##...###..',
          '##....###..', '###########', '......###..', '......###..', '......###..', '......###..',
          '....#######'],
    '5': ['##########', '###......#', '###.......', '###.......', '########..', '......###.',
          '.......###', '.......###', '.......###', '#......###', '##.....###', '.##...###.',
          '..#####...'],
    '6': ['...######..', '..###...##.', '.###.......', '###........', '###........', '###.#####..',
          '#####...###', '###.....###', '###.....###', '###.....###', '.###....###', '..###..###.',
          '...#####...'],
    '7': ['##########', '#......###', '.......###', '......###.', '......###.', '.....###..',
          '.....###..', '....###...', '....###...', '...###....', '...###....', '...###....',
          '..#####...'],
    '8': ['...#####...', '.###...###.', '.###...###.', '.###...###.', '..###.###..', '...#####...',
          '.###...###.', '###.....###', '###.....###', '###.....###', '###.....###', '.###...###.',
          '...#####...'],
    '9': ['...#####...', '.###...###.', '###.....###', '###.....###', '###.....###', '.###...####',
          '..#####.###', '........###', '........###', '.......###.', '##....###..', '.##..###...',
          '..#####....'],
    ':': ['...'] * 3 + ['.#.', '###', '.#.'] + ['...'] * 3 + ['.#.', '###', '.#.', '...'],
    '/': ['.......##', '......###', '......##.', '.....###.', '.....##..', '....###..', '....##...',
          '...###...', '...##....', '..###....', '..##.....', '.###.....', '##.......'],
    '.': ['...', '...', '...', '...', '...', '...', '...', '...', '...', '...',
          '.#.', '###', '.#.'],
    ',': ['...', '...', '...', '...', '...', '...', '...', '...', '...', '...',
          '.#.', '###', '.##', '.#.', '#..'],
    '!': ['.#.', '###', '###', '###', '###', '###', '.#.', '.#.', '.#.', '...', '.#.', '###', '.#.'],
    "'": ['##', '##', '.#', '#.'] + ['..'] * 9,
    '-': ['.....'] * 6 + ['#####'] + ['.....'] * 6,
    '&': ['...###.....',
          '..##.##....',
          '..##.##....',
          '..##.##....',
          '...###.....',
          '..####..###',
          '.##.###..#.',
          '##...###.#.',
          '##....####.',
          '##.....##..',
          '.##...####.',
          '..####..###',
          '...........'],
}
_SPACE_W = 6


@lru_cache(maxsize=None)
def _mask(ch):
    rows = _G.get(ch)
    if rows is None:
        return None
    w = max(len(r) for r in rows)
    m = np.zeros((len(rows), w), bool)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            m[y, x] = c == '#'
    return m


# ---------------------------------------------------------------- styles
# ramp: fill colours top -> bottom over the 13 cap rows; hi: lit left/top edges; lo: right/bottom edges
STYLES = {
    'gold': dict(ramp=['cream', 'butter', 'butter', 'gold', 'gold', 'gold', 'butter', 'gold', 'gold',
                       'gold', 'amber', 'amber', 'amber'],
                 hi='cream', lo='amber', lo2='copper', outline='royal', shadow='#2e1f55',
                 shadow2='#45293f', flash='white'),
    'silver': dict(ramp=['white', 'white', 'baby_blue', 'baby_blue', 'cloud', 'cloud', 'white',
                         'cloud', 'cloud_dk', 'cloud_dk', 'periwinkle', 'periwinkle', 'periwinkle'],
                   hi='white', lo='periwinkle', lo2='lavender', outline='indigo', shadow='#1b1934',
                   shadow2='navy', flash='white'),
    'rose': dict(ramp=['white', 'cotton', 'cotton', 'bubblegum', 'bubblegum', 'bubblegum', 'cotton',
                       'bubblegum', 'bubblegum', 'pink', 'hot_pink', 'hot_pink', 'hot_pink'],
                 hi='white', lo='hot_pink', lo2='rose', outline='plum', shadow='#3a1030',
                 shadow2='eggplant', flash='white'),
}
_PAD = 1          # outline width
_EXT = 2          # extrusion depth (shadow rows below)


@lru_cache(maxsize=None)
def _letter(ch, style='gold', flash=False):
    """RGBA sprite of one letter incl. outline (1 px) and extrusion; origin = (-_PAD, -_PAD)."""
    m = _mask(ch)
    st = STYLES[style]
    gh, gw = m.shape
    h, w = gh + 2 * _PAD + _EXT, gw + 2 * _PAD + 1
    full = np.zeros((h, w), bool)
    full[_PAD:_PAD + gh, _PAD:_PAD + gw] = m
    # 8-way outline
    pad = np.pad(full, 1)
    dil = np.zeros_like(full)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    outl = dil & ~full
    rgba = np.zeros((h, w, 4), np.uint8)
    # extrusion: the outlined shape pushed down 1..EXT rows (and 1 px right on the last row)
    for k in range(_EXT, 0, -1):
        sh = np.zeros_like(dil)
        sh[k:, :] = dil[:h - k, :]
        if k == _EXT:
            sh2 = np.zeros_like(sh)
            sh2[:, 1:] = sh[:, :-1]
            sh |= sh2
        sel = sh & ~dil
        rgba[sel, :3] = col(st['shadow'] if k == _EXT else st['shadow2'])
        rgba[sel, 3] = 255
    rgba[outl, :3] = col(st['outline'])
    rgba[outl, 3] = 255
    # fill: row ramp + bevel (lit top/left edges, shaded right/bottom edges)
    p = np.pad(full, 1)
    up = p[:-2, 1:-1]
    dn = p[2:, 1:-1]
    lf = p[1:-1, :-2]
    rt = p[1:-1, 2:]
    for y in range(h):
        gy = y - _PAD
        if not (0 <= gy < len(st['ramp']) + 2):
            continue
        rc = col(st['ramp'][min(gy, len(st['ramp']) - 1)])
        row = full[y]
        rgba[y, row, :3] = rc
        rgba[y, row, 3] = 255
    hi = full & (~lf | ~up)
    lo = full & ~rt & lf
    lo2 = full & ~dn & up & (np.arange(h)[:, None] > _PAD + 6)
    rgba[hi, :3] = col(st['hi'])
    rgba[lo, :3] = col(st['lo'])
    rgba[lo2 & ~hi, :3] = col(st['lo2'])
    if flash:
        rgba[full, :3] = col(st['flash'])
        rgba[outl, :3] = col('lilac')
    return rgba


def _layout(s, spacing=1):
    """[(ch, x_offset)] and the total advance width of the glyph run (without outline)."""
    out = []
    x = 0
    for ch in s:
        if ch == ' ':
            x += _SPACE_W
            continue
        m = _mask(ch)
        if m is None:            # unknown characters render as a space
            x += _SPACE_W
            continue
        out.append((ch, x))
        x += m.shape[1] + spacing
    return out, max(0, x - spacing)


def title_size(s, spacing=1, scale=1):
    """(w, h) of the rendered title incl. outline and extrusion, at the given integer scale."""
    _, w = _layout(s.upper(), spacing)
    return (w + 2 * _PAD + 1) * scale, (CAP + 2 * _PAD + _EXT) * scale


def _blit_rgba(c, rgba, x, y, scale=1):
    if scale > 1:
        rgba = np.repeat(np.repeat(rgba, scale, 0), scale, 1)
    sh, sw = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(c.w, x + sw), min(c.h, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    m = sub[..., 3] >= 128
    c.a[y0:y1, x0:x1][m] = sub[..., :3][m]


def draw_title(c, x, y, s, f=None, t0=None, step=9, align='center', scale=1, spacing=1, style='gold',
               drop=10, shine=None, hide_after=None):
    """Big pixel-serif title. (x, y): x per `align` ('left' | 'center' | 'right'), y = top of the caps.

    f, t0: animate - letter k (spaces skipped) pops in at frame t0 + k*step: 2 frames of white flash,
    dropping from `drop` px above with a bounce (lands ~6 frames after its start). Without t0 every
    letter is drawn at rest. shine: None or 0..1, a diagonal glint band sweeping left -> right.
    Returns (x0, y0, w, h) of the whole title box (outline + extrusion included)."""
    s = s.upper()
    items, run_w = _layout(s, spacing)
    tw, th = title_size(s, spacing, scale)
    if align == 'center':
        x0 = int(x) - tw // 2
    elif align == 'right':
        x0 = int(x) - tw
    else:
        x0 = int(x)
    y0 = int(y) - _PAD * scale
    for k, (ch, ox) in enumerate(items):
        dy = 0
        flash = False
        if t0 is not None and f is not None:
            ts = t0 + k * step
            if f < ts:
                continue
            u = f - ts
            dy = -int(round(drop * (1 - ease_out_bounce(seg(u, 0, 7)))))
            flash = u < 2
        spr = _letter(ch, style, flash)
        _blit_rgba(c, spr, x0 + ox * scale, y0 + dy * scale, scale)
    if shine is not None and 0 < shine < 1:
        _shine(c, x0, y0, tw, th, shine, style)
    return x0, y0, tw, th


def _shine(c, x0, y0, w, h, u, style):
    """Diagonal glint band (3 px) over the bright fill pixels of the title box."""
    st = STYLES[style]
    bright = {col(n) for n in st['ramp'] + [st['hi'], st['lo'], st['lo2']]}
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    sub = c.a[y0c:y1c, x0c:x1c]
    yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
    cxp = x0 - h + (w + 2 * h) * u
    d = (xx - x0) + (yy - y0) * 0.6 - (cxp - x0)
    band = (d >= 0) & (d < 4)
    core = (d >= 1) & (d < 3)
    keys = (sub[..., 0].astype(np.int32) << 16) | (sub[..., 1].astype(np.int32) << 8) | sub[..., 2]
    bk = np.array([(r << 16) | (g << 8) | b for (r, g, b) in bright], np.int32)
    isfill = np.isin(keys, bk)
    sub[band & isfill] = col('cream')
    sub[core & isfill] = col('white')


# ---------------------------------------------------------------- cartouche (parchment plate)
_ROLL = ['plum_brown', 'clay', 'sand', 'cream', 'paper', 'tan', 'clay', 'rust']   # across a roll
_ROLL_W = len(_ROLL)


@lru_cache(maxsize=16)
def _plate(w, h):
    """RGBA of the flat parchment plate (without rolls), w x h."""
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 3] = 255
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.array(col('paper'), np.float32)
    warm = np.array(col('blush'), np.float32)
    lite = np.array(col('cream'), np.float32)
    t = yy / max(1, h - 1)
    rgb = base[None, None] * (1 - t[..., None] * 0.6) + warm[None, None] * (t[..., None] * 0.6)
    # paper fibres: sparse hashed flecks
    fl = ((xx * 7 + yy * 13) % 23 == 0) | ((xx * 5 + yy * 3) % 37 == 0)
    rgb[fl] = rgb[fl] * 0.93
    top = yy < 2
    rgb[top] = lite
    a[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    # quantise to a 3-step ramp so it stays pixel-art (paper / blush mix steps)
    q = np.array([col('cream'), col('paper'), mix('paper', 'blush', 0.5), col('blush')], np.int32)
    flat = a[..., :3].reshape(-1, 3).astype(np.int32)
    idx = ((flat[:, None, :] - q[None]) ** 2).sum(-1).argmin(1)
    a[..., :3] = q[idx].reshape(h, w, 3).astype(np.uint8)
    # frame: dark rim, gold rule, inner brass hairline with corner diamonds
    rim = col('plum_brown')
    a[0, :, :3] = rim
    a[-1, :, :3] = rim
    a[:, 0, :3] = rim
    a[:, -1, :3] = rim
    a[1, 1:-1, :3] = col('cream')
    a[-2, 1:-1, :3] = col('tan')
    g = col('brass')
    gl = col('gold')
    a[2, 2:-2, :3] = gl
    a[-3, 2:-2, :3] = g
    a[2:-2, 2, :3] = gl
    a[2:-2, -3, :3] = g
    for (cx, cy) in ((4, 4), (w - 5, 4), (4, h - 5), (w - 5, h - 5)):
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            a[cy + dy, cx + dx, :3] = g
        a[cy, cx, :3] = col('hot_pink')
    return a


@lru_cache(maxsize=8)
def _roll(h):
    """RGBA of a vertical scroll roll, height h (curl at top and bottom)."""
    a = np.zeros((h, _ROLL_W, 4), np.uint8)
    for x, cname in enumerate(_ROLL):
        a[1:-1, x, :3] = col(cname)
        a[1:-1, x, 3] = 255
    # rounded caps with the spiral end visible
    for x in range(1, _ROLL_W - 1):
        for yy in (0, h - 1):
            a[yy, x, :3] = col('plum_brown')
            a[yy, x, 3] = 255
    for yy in (1, h - 2):
        a[yy, 0, 3] = 0
        a[yy, -1, 3] = 0
    for yy in (2, 3, h - 4, h - 3):
        a[yy, 2:_ROLL_W - 2, :3] = col('tan')
    a[2, 3:5, :3] = col('rust')
    a[h - 3, 3:5, :3] = col('rust')
    a[1:-1, 0, :3] = col('plum_brown')
    a[1:-1, -1, :3] = col('plum_brown')
    # tiny wood knobs top/bottom
    return a


def cartouche_rect(cx, y, w, h):
    """(x0, y0, w, h) of the plate interior (between the rolls)."""
    return cx - w // 2 + _ROLL_W - 2, y, w - 2 * (_ROLL_W - 2), h


def draw_cartouche(c, cx, y, w, h, t=1.0, shadow=True):
    """Parchment plate centred on cx with top y, total width w (incl. both rolls) and height h.
    t 0..1: unroll from the centre (the rolls slide apart, revealing the plate)."""
    t = clamp(t)
    if t <= 0:
        return
    pw = w - 2 * (_ROLL_W - 2)
    vis = max(0, int(round(pw * t)))
    x_left = cx - vis // 2
    if shadow:
        c.blend(x_left - 1, y + 2, vis + 4, h + 2, '#120c20', 0.45)
    if vis > 0:
        plate = _plate(pw, h)
        off = (pw - vis) // 2
        sub = plate[:, off:off + vis]
        x0, x1 = max(0, x_left), min(c.w, x_left + vis)
        y0, y1 = max(0, y), min(c.h, y + h)
        if x1 > x0 and y1 > y0:
            c.a[y0:y1, x0:x1] = sub[y0 - y:y1 - y, x0 - x_left:x1 - x_left, :3]
    roll = _roll(h + 4)
    _blit_rgba(c, roll, x_left - _ROLL_W + 2, y - 2)
    _blit_rgba(c, roll[:, ::-1], x_left + vis - 2, y - 2)


def draw_subtitle(c, cx, y, s, f=None, t0=None, rate=2, color='royal', shade_color='lilac',
                  ornament='gold'):
    """Small caps line (5x7 font, 1 px letter-spacing) centred on cx, typed on from t0 at `rate`
    frames per character (all shown if t0 is None), flanked by tiny diamond ornaments."""
    s = s.upper()
    n = len(s) if t0 is None or f is None else int(clamp((f - t0) / rate + 1, 0, len(s)))
    if n <= 0:
        return
    tw = text_width(s, 'normal', spacing=1)
    x0 = cx - tw // 2
    shown = s[:n]
    text(c, x0, y + 1, shown, shade_color, spacing=1)
    text(c, x0, y, shown, color, spacing=1)
    if n < len(s) and n > 0:
        cx_ = x0 + text_width(shown, 'normal', spacing=1) + 1
        c.rect(cx_, y + 6, 4, 1, ornament)
    if n == len(s) and ornament:
        for sx in (x0 - 7, x0 + tw + 5):
            for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                c.px(sx + dx, y + 3 + dy, ornament)
            c.px(sx, y + 3, 'hot_pink')


def draw_title_card(c, f, cx=160, y=6, t0=0, title='EQUESTRIA', subtitle='A PIXEL WORLD ATLAS',
                    step=9, sub_t0=None, unroll=8, style='gold', glints=True, shine_t0=None,
                    shine_every=288, pad_x=8, dy=0, drop=7):
    """Cartouche + big title + subtitle, animated from t0 (the plate unrolls over `unroll` frames
    ending at t0; letter k pops in at t0 + k*step; the subtitle types on from sub_t0 (default: when
    the title is half done); a glint sweeps the title from shine_t0 (default: after the last letter)
    and repeats every `shine_every` frames). Pass t0 <= f - 200 for a fully static card.
    dy: extra vertical offset (e.g. to slide the card away). Returns the plate rect (x0, y0, w, h)."""
    y = y + dy
    tw, th = title_size(title)
    sw = text_width(subtitle.upper(), 'normal', spacing=1) + 16 if subtitle else 0
    inner = max(tw, sw) + 2 * pad_x
    w = inner + 2 * (_ROLL_W - 2)
    h = 4 + th + (11 if subtitle else 0) + 3
    k = clamp((f - (t0 - unroll)) / max(1, unroll))
    draw_cartouche(c, cx, y, w, h, ease_out_back(k, 1.2) if k < 1 else 1.0)
    if k < 1:
        return cx - w // 2, y, w, h
    n_letters = len([ch for ch in title if ch != ' '])
    end = t0 + (n_letters - 1) * step + 8
    sh = None
    s0 = end if shine_t0 is None else shine_t0
    if f >= s0:
        u = ((f - s0) % shine_every) / 16.0
        sh = u if u < 1 else None
    draw_title(c, cx, y + 5, title, f=f, t0=t0, step=step, style=style, shine=sh, drop=drop)
    if subtitle:
        st0 = (t0 + (n_letters // 2) * step) if sub_t0 is None else sub_t0
        draw_subtitle(c, cx, y + 5 + th, subtitle, f=f, t0=st0)
    if glints and f >= end:
        for i, (gx, gy) in enumerate(((cx - w // 2 + 8, y + 4), (cx + w // 2 - 9, y + h - 5),
                                      (cx + w // 2 - 9, y + 4), (cx - w // 2 + 8, y + h - 5))):
            _twinkle(c, gx, gy, f + i * 17, 'white')
    return cx - w // 2, y, w, h


def _twinkle(c, x, y, f, color='white', period=48):
    t = (f % period) / period
    s = int(round(math.sin(t * math.pi) * 2.4))
    if s <= 0:
        return
    c.px(x, y, color)
    for k in range(1, s + 1):
        cc = color if k < s else 'gold'
        c.px(x + k, y, cc)
        c.px(x - k, y, cc)
        c.px(x, y + k, cc)
        c.px(x, y - k, cc)
