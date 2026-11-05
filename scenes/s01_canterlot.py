"""1 · CANTERLOT — the royal capital at sunrise (5 bars, 360 f, HUD 05:30 -> 06:30).

Beat sheet (local frames)
  Bar 1   0– 71  world-map travel shot (engine.loc_worldmap.travel_frame), iris closes 64–71.
  Bar 2  72–143  SHOT A "the city on the mountain", blue hour. Iris opens on the castle (72–79).
                 Camera starts low in the valley (Ponyville roofs, Saddle Lake, dark trees) and
                 eases a tilt UP the waterfalls to the castle (layer ranges: fg 200, valley 150,
                 city 70, far range 30, sky 30 px). Windows and terrace lamps lit, mist boils at the
                 falls' feet, stars fade, wispy clouds drift. Callout CANTERLOT CASTLE f 90.
  Bar 3 144–215  SUNRISE (hit `sunrise` 144): the sun breaks the far ridge right of the mountain
                 (flash + rays). A gold rim light sweeps the city from the top-right down-left over
                 ~1 bar (indexed-material city: dawn palette -> sun palette with rim edges and long
                 cast shadows), sky cross-fades blue hour -> sunrise, sun & moon banners unfurl on
                 beats, windows switch off one by one behind the sweep, stained glass glints.
                 Callout CANTERLOT TOWER f 144. Atlas card 100–200.
  Bar 4 216–287  SHOT B "royal gardens" (cut on the bar line): hedge-maze terrace with geometric
                 topiary, fountains, urns and obelisks under the castle; camera pans right to the
                 mountain railway corkscrewing round a rock spur into a tunnel. The pink-and-white
                 train puffs out of the tunnel on the beat f 234. Callouts ROYAL GARDENS 216,
                 MOUNTAIN RAILWAY 234.
  Bar 5 288–359  SHOT C "the journey begins": airship docks on a cliff ledge, docked zeppelin, the
                 explorer's balloon ('l') casts off on beat f 306 and rises, shrinking with distance
                 (l -> m -> s -> xs), while the dock drops away and the camera pulls back to the
                 city glittering in morning light. Callout AIRSHIP DOCKS 288. Toast from 302.
Light: blue hour (cool, lamps/windows warm) -> low gold sun from the right (east) -> morning.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, balloon_size, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'canterlot'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

SUNRISE = 144
CUT_B = 216
CUT_C = 324

# =====================================================================================
# Indexed materials. Every static layer is drawn as (material id, depth z); colours for the
# three lights ('dawn' blue hour, 'sun' first gold light, 'morn' morning) come from this table.
# In 'sun'/'morn' the renderer adds cast shadows (nearer things to the right darken farther
# things to their left) and rim light on edges facing the sun.
# =====================================================================================
#  name        dawn        sun          morn        rim: 0 none, 1 right edges, 2 right+top
_MATS = [
    ('W_OUT',   '#34315a', '#6c4e76', '#5e5478', 1),
    ('W_DK',    '#2a2848', '#4e3a62', '#4a4270', 0),
    ('W_SH',    '#545488', '#a286ac', '#a49cc2', 1),
    ('W',       '#7072a8', '#e4c2c2', '#dcdaec', 1),
    ('W_LT',    '#8a8ec2', '#f8dac4', '#f0eef4', 1),
    ('W_HI',    '#a8aeda', '#fff2da', '#ffffff', 2),
    ('T_SH',    '#463e62', 'copper',  'copper',  1),
    ('T',       '#68607a', 'gold',    'gold',    2),
    ('T_HI',    '#8a8084', 'butter',  'butter',  2),
    ('R_OUT',   '#1c1a3a', '#2e1f58', '#32246a', 1),
    ('R_SH',    '#2a2856', '#46307e', '#46339a', 1),
    ('R',       '#3a3672', 'royal',   'royal',   2),
    ('R_LT',    '#4c4a8e', 'royal_lt', 'royal_lt', 2),
    ('R_HI',    '#6260aa', '#e2a2d8', 'lavender', 2),
    ('G_SH',    '#443e5e', 'rust',    'copper',  1),
    ('G',       '#5e5872', 'amber',   'amber',   2),
    ('G_LT',    '#7a7280', 'gold',    'gold',    2),
    ('G_HI',    '#968c88', 'butter',  'butter',  2),
    ('WIN',     '#20203c', '#3c3460', '#3e4a7e', 0),
    ('WIN_HI',  '#38385e', '#f2b48c', '#a8c8f0', 0),
    ('SG_A',    '#5a4a9a', 'royal_lt', 'royal_lt', 0),
    ('SG_B',    '#9a4a8a', 'magic',   'magic',   0),
    ('SG_C',    '#b08a48', 'gold',    'gold',    0),
    ('SG_D',    '#4a6ab0', 'crystal', 'crystal', 0),
    ('ROCK_DK', '#211d36', '#34284a', '#3c3454', 0),
    ('ROCK_SH', '#2e2a4a', '#4c3a62', '#564b72', 1),
    ('ROCK',    '#3e3860', '#6c5680', '#786c94', 1),
    ('ROCK_LT', '#504a76', '#a07a92', '#988eb2', 2),
    ('ROCK_HI', '#645e8c', '#e2a88e', '#bab0ce', 2),
    ('SNOW_SH', '#6a6ca2', '#c48ab4', '#c4cceb', 2),
    ('SNOW',    '#9698c8', '#ffd2c2', '#ffffff', 2),
    ('GR_DK',   '#182a3e', '#24403e', '#1a5246', 0),
    ('GR',      '#223e4c', '#3c6646', '#2a8a5e', 1),
    ('GR_LT',   '#2e525a', '#86a24e', '#78c462', 2),
    ('PAVE',    '#5a5a8c', '#e6bea6', '#d6d0e0', 2),
    ('WD_DK',   '#2a2236', '#4a2a36', '#523640', 0),
    ('WD',      '#3c2e48', '#8e4a3a', '#8c5a48', 1),
    ('WD_LT',   '#503e58', '#cd683d', '#b67a58', 2),
    ('FA',      '#3a3a6e', '#9a84bc', '#9fb2e2', 0),
    ('FA_LT',   '#46468a', '#b896c4', '#bccbf0', 2),
    ('FB',      '#2c2a5a', '#6e5c9a', '#8092c8', 0),
    ('FB_LT',   '#383870', '#90729e', '#a0b2e0', 2),
    ('FB_SH',   '#242250', '#584a86', '#6c7eb6', 0),
    ('POOL',    '#3a4886', '#9ab0e0', '#8fd3ff', 0),
    ('POOL_HI', '#5868a8', '#ffe2b8', '#e8ffff', 0),
    ('IRON',    '#24203a', '#3e2e4e', '#3a3450', 0),
    ('PINK',    '#7a5a8a', 'bubblegum', 'bubblegum', 1),
    ('PINK_SH', '#5a4070', 'rose',    'rose',    0),
    ('PINK_LT', '#9a7aa8', 'cotton',  'cotton',  2),
    ('CREAM',   '#8c8cb8', 'cream',   'cream',   2),
    ('MAG',     '#5a3060', 'magenta', 'magenta', 1),
    ('FLW_A',   '#6a4a7a', 'hot_pink', 'hot_pink', 0),
    ('FLW_B',   '#7a7aa8', 'white',   'white',   0),
    ('FLW_C',   '#5a4a8a', 'lavender', 'lavender', 0),
    ('FLW_D',   '#8a7a6a', 'gold',    'gold',    0),
    ('GRAVEL',  '#6a6890', '#f0d2b4', '#e8e0d4', 0),
    ('GRAVEL_SH', '#4e4c78', '#b89ab0', '#bab2c4', 0),
    ('HEDGE_DK', '#16283a', '#1c4a40', '#165a4c', 0),
    ('HEDGE',   '#203a48', '#2f7050', '#239063', 1),
    ('HEDGE_LT', '#2c4e58', '#7aa648', '#5cc06a', 2),
    ('HEDGE_HI', '#3a6066', '#c8d860', '#a8e070', 2),
]
MID = {m[0]: i + 1 for i, m in enumerate(_MATS)}
_N_M = 256


def _luts():
    out = {}
    for k, mode in enumerate(('dawn', 'sun', 'morn')):
        lut = np.zeros((_N_M, 3), np.uint8)
        sh = np.zeros((_N_M, 3), np.uint8)
        rim = np.zeros((_N_M, 3), np.uint8)
        for i, m in enumerate(_MATS):
            c = col(m[1 + k])
            lut[i + 1] = c
            if mode == 'sun':
                sh[i + 1] = mix(c, '#4c3c80', 0.48)
                rim[i + 1] = mix(c, 'butter', 0.62)
            elif mode == 'morn':
                sh[i + 1] = mix(c, '#5c5c9c', 0.32)
                rim[i + 1] = mix(c, 'white', 0.5)
            else:
                sh[i + 1] = c
                rim[i + 1] = c
        out[mode] = (lut, sh, rim)
    return out


_LUT = _luts()
_RIMF = np.zeros(_N_M, np.uint8)
for _i, _m in enumerate(_MATS):
    _RIMF[_i + 1] = _m[4]
_ARCH = np.zeros(_N_M, np.uint8)
for _i, _m in enumerate(_MATS):
    if _m[0].split('_')[0] in ('W', 'T', 'R', 'G', 'SNOW'):
        _ARCH[_i + 1] = 1
# special rim tints
_LUT['sun'][2][MID['FA']] = col('peach')
_LUT['sun'][2][MID['FA_LT']] = col('#ffd0a8')
_LUT['sun'][2][MID['FB_LT']] = col('#ffb898')
_LUT['sun'][2][MID['W_OUT']] = col('gold')
_LUT['sun'][2][MID['R_OUT']] = col('#f0a0c8')


class Ix:
    """Index canvas: draws (material, depth) pairs with the ordinary Canvas primitives."""

    def __init__(self, w, h):
        self.c = Canvas(w, h)
        self.z = 1

    def k(self, m):
        return (MID[m], self.z, 0)

    def rect(self, x, y, w, h, m):
        self.c.rect(x, y, w, h, self.k(m))

    def px(self, x, y, m):
        self.c.px(x, y, self.k(m))

    def hline(self, x0, x1, y, m):
        self.c.hline(x0, x1, y, self.k(m))

    def vline(self, x, y0, y1, m):
        self.c.vline(x, y0, y1, self.k(m))

    def poly(self, pts, m):
        self.c.poly(pts, self.k(m))

    def ellipse(self, cx, cy, rx, ry, m):
        self.c.ellipse(cx, cy, rx, ry, self.k(m))

    def line(self, x0, y0, x1, y1, m):
        self.c.line(x0, y0, x1, y1, self.k(m))

    def frame(self, x, y, w, h, m):
        self.c.frame(x, y, w, h, self.k(m))

    @property
    def mat(self):
        return self.c.a[:, :, 0]

    @property
    def zz(self):
        return self.c.a[:, :, 1]

    def recolor_where(self, sel, m):
        self.c.a[:, :, 0][sel] = MID[m]


def _finish(ix, shadow_len=12, slope=3, extra=None):
    """Bake shadow / rim masks. Returns the layer dict."""
    mat = ix.mat.copy()
    z = ix.zz.astype(np.int16)
    h, w = z.shape
    op = mat > 0
    sh = np.zeros((h, w), bool)
    for d in range(1, shadow_len + 1):
        dy = d // slope
        s = np.zeros((h, w), np.int16)
        s[dy:, :w - d] = z[:h - dy, d:]
        sh |= s > z + 1
    sh &= op
    zr = np.zeros_like(z)
    zr[:, :-1] = z[:, 1:]
    zu = np.zeros_like(z)
    zu[1:] = z[:-1]
    rf = _RIMF[mat]
    rim = op & (((rf >= 1) & (zr < z)) | ((rf >= 2) & (zu < z))) & ~sh
    L = dict(mat=mat, z=z, sh=sh, rim=rim, alpha=op)
    if extra:
        L.update(extra)
    return L


def _rgb(L, mode):
    key = ('_rgb', mode)
    if key in L:
        return L[key]
    lut, sh, rim = _LUT[mode]
    mat = L['mat']
    rgb = lut[mat]
    if mode != 'dawn':
        rgb[L['sh']] = sh[mat[L['sh']]]
        rgb[L['rim']] = rim[mat[L['rim']]]
    L[key] = rgb
    return rgb


def _draw_layer(c, L, x, y, mode='dawn', mode_b=None, mask=None, band=None):
    """Composite an indexed layer at screen (x, y). With mode_b+mask (screen-sized bool), pixels
    under the mask use mode_b; `band` (screen bool) gets the rim colour of mode_b (sweep front)."""
    al = L['alpha']
    lh, lw = al.shape
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + lw), min(H, y + lh)
    if x1 <= x0 or y1 <= y0:
        return
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    a = al[sl]
    src = _rgb(L, mode)[sl]
    if mode_b is not None and mask is not None:
        m = mask[y0:y1, x0:x1]
        if m.any():
            src = np.where(m[:, :, None], _rgb(L, mode_b)[sl], src)
            if band is not None:
                b = band[y0:y1, x0:x1] & (_ARCH[L['mat'][sl]] > 0)
                if b.any():
                    src = src.copy()
                    src[b] = _LUT[mode_b][2][L['mat'][sl][b]]
    dst = c.a[y0:y1, x0:x1]
    dst[a] = src[a]


# =====================================================================================
# Architecture kit (index space). Light comes from the right (east).
# =====================================================================================
def _cyl(u):
    return 'SH' if u < 0.24 else ('' if u < 0.55 else ('LT' if u < 0.84 else 'HI'))


def _m(base, part):
    return base + ('_' + part if part else '')


def cone(ix, cx, base_y, half, h, kind='R', flare=True):
    """Conical roof: apex at (cx, base_y - h), base half width `half` (pixels each side of cx)."""
    for k in range(h):
        y = base_y - h + k
        t = (k + 1) / h
        hw = half * (t ** 1.15 if flare else t)
        x0, x1 = int(round(cx - hw)), int(round(cx + hw))
        for x in range(x0, x1 + 1):
            u = (x - x0 + 0.5) / max(1, x1 - x0 + 1)
            part = _cyl(u)
            if kind == 'G':
                part = {'SH': 'SH', '': '', 'LT': 'LT', 'HI': 'HI'}[part]
            ix.px(x, y, _m(kind, part))
        # tile courses
        if k % 3 == 2 and x1 - x0 >= 4:
            for x in range(x0 + 1, x1, 2):
                if (x - x0) / max(1, x1 - x0) < 0.6:
                    ix.px(x, y, _m(kind, 'SH'))
        ix.px(x0, y, _m(kind, 'SH') if kind == 'G' else 'R_OUT')
    ix.hline(int(round(cx - half)), int(round(cx + half)), base_y, 'T')
    ix.px(int(round(cx + half)), base_y, 'T_HI')
    ix.px(int(round(cx - half)), base_y, 'T_SH')


def finial(ix, cx, apex_y, needle=4, flags=None, z_flag=None):
    ix.px(cx, apex_y, 'T')
    ix.vline(cx, apex_y - needle, apex_y - 1, 'T_SH')
    ix.px(cx, apex_y - 2, 'T_HI')
    if flags is not None:
        flags.append((cx, apex_y - needle))


def tower(ix, cx, top, bot, w, roof='R', roof_h=14, wins=None, flags=None, band=True,
          balcony=None, win_rows=None, needle=4, dome=False):
    x0 = cx - w // 2
    for i in range(w):
        ix.vline(x0 + i, top, bot, _m('W', _cyl((i + 0.5) / w)))
    ix.vline(x0, top, bot, 'W_OUT')
    ix.vline(x0 + w - 1, top, bot, 'W_OUT')
    # stone courses on the shadow side
    for y in range(top + 5, bot, 6):
        ix.hline(x0 + 1, x0 + max(1, w // 3), y, 'W_SH')
    if band:
        for i in range(w + 2):
            u = (i + 0.5) / (w + 2)
            ix.px(x0 - 1 + i, top, _m('T', {'SH': 'SH', '': '', 'LT': '', 'HI': 'HI'}[_cyl(u)]))
            ix.px(x0 - 1 + i, top + 1, 'T_SH' if u < 0.5 else 'T')
        for x in range(x0, x0 + w, 2):
            ix.px(x, top + 2, 'W_SH')
    if balcony is not None:
        by = balcony
        ix.hline(x0 - 2, x0 + w + 1, by, 'W_HI')
        ix.hline(x0 - 2, x0 + w + 1, by + 1, 'W_SH')
        for x in range(x0 - 2, x0 + w + 2, 2):
            ix.px(x, by - 1, 'W_LT')
        ix.px(x0 + w + 1, by - 1, 'W_HI')
        ix.px(x0 - 2, by + 1, 'W_OUT')
        for x in range(x0 - 1, x0 + w + 1, 3):
            ix.px(x, by + 2, 'W_SH')
    # windows (arched slits) recorded for dynamic lighting
    if win_rows:
        ww = 2 if w >= 9 else 1
        wx = cx - (1 if ww == 2 else 0)
        for wy in win_rows:
            wh = 4 if ww == 2 else 3
            ix.rect(wx, wy, ww, wh, 'WIN')
            ix.hline(wx - 1, wx + ww, wy - 1, 'T_SH')
            ix.hline(wx, wx + ww - 1, wy + wh, 'W_HI')
            if wins is not None:
                wins.append((wx, wy, ww, wh))
    if dome:
        rx = w // 2 + 1
        for x in range(cx - rx, cx + rx + 1):
            u = (x - (cx - rx) + 0.5) / (2 * rx + 1)
            hh = int(round(roof_h * math.sqrt(max(0.0, 1 - ((x - cx) / (rx + 0.5)) ** 2))))
            ix.vline(x, top - hh, top - 1, _m('G', _cyl(u)))
        finial(ix, cx, top - roof_h - 1, needle, flags)
        return
    if roof:
        cone(ix, cx, top - 1, w // 2 + 1.5, roof_h, roof)
        finial(ix, cx, top - 1 - roof_h, needle, flags)


def house(ix, x, y_bot, w, h, roof='R', wins=None, roof_h=None, door=True):
    """Small townhouse; (x, y_bot) = bottom-left."""
    top = y_bot - h
    for i in range(w):
        u = (i + 0.5) / w
        ix.vline(x + i, top, y_bot, 'W_SH' if u < 0.18 else ('W' if u < 0.7 else 'W_LT'))
    ix.vline(x, top, y_bot, 'W_OUT')
    ix.vline(x + w - 1, top, y_bot, 'W_LT')
    rh = roof_h or max(4, w // 2)
    # pitched roof with overhang
    ix.poly([(x - 1, top + 0.5), (x + w + 0.5, top + 0.5), (x + w / 2, top - rh)], _m(roof, ''))
    ix.poly([(x + w / 2, top - rh), (x + w + 0.5, top + 0.5), (x + w * 0.62, top + 0.5)], _m(roof, 'LT'))
    ix.line(x - 1, top, x + w // 2, top - rh, _m(roof, 'SH'))
    ix.hline(x - 1, x + w, top + 1, 'T_SH')
    ix.px(x + w // 2, top - rh - 1, 'T')
    # windows
    wx, wy = x + 2, top + 3
    if wy + 2 < y_bot - 1:
        ix.rect(wx, wy, 2, 3, 'WIN')
        ix.hline(wx, wx + 1, wy + 3, 'W_HI')
        if wins is not None:
            wins.append((wx, wy, 2, 3))
    if door and h >= 9:
        dx = x + w - 4
        ix.rect(dx, y_bot - 3, 2, 3, 'W_DK')
        ix.px(dx, y_bot - 4, 'W_SH')


def tree(ix, cx, by, r, seed=0):
    """Curly round-canopy tree; (cx, by) = trunk base."""
    ix.vline(cx, by - r, by, 'WD')
    blobs = [(0, -r - 1, r)] + [((hash01(seed, i) - 0.5) * r * 1.4, -r - 1 + (hash01(seed, i, 2) - 0.3) * r,
                                 r * (0.55 + 0.3 * hash01(seed, i, 3))) for i in range(3)]
    for dx, dy, rr in blobs:
        ix.ellipse(cx + dx, by + dy, rr, rr * 0.85, 'GR')
    for dx, dy, rr in blobs:
        ix.ellipse(cx + dx + rr * 0.35, by + dy - rr * 0.3, rr * 0.55, rr * 0.45, 'GR_LT')
    for dx, dy, rr in blobs:
        ix.ellipse(cx + dx - rr * 0.45, by + dy + rr * 0.35, rr * 0.45, rr * 0.35, 'GR_DK')


def _noise(h, w, cy, cx, seed):
    from scipy import ndimage
    g = rng(seed).random((h // cy + 4, w // cx + 4))
    return ndimage.zoom(g, (cy, cx), order=3)[:h, :w]


_RAMP = ('ROCK_DK', 'ROCK_SH', 'ROCK', 'ROCK_LT', 'ROCK_HI')


def rock_tex(ix, sel, seed=0, dark=None, strata=11, green=0.0, k=0.95, bias=0.0):
    """Paint crag texture into the selected pixels: vertical-ish noise facets lit from the right,
    wavy strata ledges (dark line, lit lip above) and optional grass tufts on the ledges.
    dark: optional float array subtracted from the light level (shadow side)."""
    from scipy import ndimage
    h, w = sel.shape
    hf = (0.60 * _noise(h, w, 30, 10, seed) + 0.30 * _noise(h, w, 12, 12, seed + 1)
          + 0.10 * _noise(h, w, 5, 4, seed + 2))
    d = np.zeros((h, w))
    d[:, 1:-1] = hf[:, :-2] - hf[:, 2:]
    lvl = 2.0 + bias + d * (k / max(1e-6, d.std()))
    if dark is not None:
        lvl -= dark
    yy, xx = np.mgrid[0:h, 0:w]
    wav = (_noise(h, w, 8, 14, seed + 5) * 6).astype(int)
    brk = _noise(h, w, 6, 9, seed + 6) > 0.42
    ph = (yy + wav) % strata
    lvl = np.where(brk & (ph == 0), np.minimum(lvl, 0.6), lvl)
    lvl = np.where(brk & (ph == strata - 1), np.maximum(lvl, 3.0) - (dark if dark is not None else 0) * 0.5, lvl)
    q = np.clip(np.round(lvl), 0, 4).astype(np.uint8)
    q = ndimage.median_filter(q, size=(3, 2))
    ids = np.array([MID[m] for m in _RAMP], np.uint8)
    ix.mat[sel] = ids[q[sel]]
    if green > 0:
        g = sel & brk & (ph == strata - 2) & (_noise(h, w, 5, 5, seed + 9) > 1 - green)
        ix.mat[g] = MID['GR']
        g2 = g & (np.roll(g, 1, axis=1) == 0)
        ix.mat[g2] = MID['GR_LT']


# =====================================================================================
# SHOT A layers
# =====================================================================================
CITY_H = 232
FALLS = [  # x, top, width, bottom (city-layer coords)
    (101, 107, 4, 214), (145, 143, 6, 216), (165, 143, 5, 216), (204, 144, 7, 216),
    (240, 142, 5, 216), (257, 138, 5, 216),
]


@lru_cache(maxsize=None)
def _city():
    ix = Ix(W, CITY_H)
    wins, flags, lamps, banners, roses = [], [], [], [], []
    # --- the mountain (z1)
    ix.z = 1
    mount = [(-2, 112), (8, 100), (16, 92), (24, 85), (30, 80), (36, 71), (42, 63), (48, 59),
             (53, 52), (57, 47), (60, 43), (63, 45), (66, 38), (71, 31), (76, 24), (80, 19),
             (84, 13), (88, 10), (91, 6), (94, 3), (97, 1), (100, 4), (103, 8), (106, 11),
             (110, 18), (115, 24), (119, 28), (124, 36), (132, 44), (140, 48), (150, 53),
             (166, 58), (190, 64), (212, 74), (232, 88), (248, 102), (258, 114), (264, 126),
             (268, 140), (264, 160), (270, 184), (266, 204), (272, CITY_H), (-2, CITY_H)]
    ix.poly(mount, 'ROCK')
    yy, xx = np.mgrid[0:CITY_H, 0:W]
    # shadow side: left of a jittered main ridge, plus the overhang shadow under the city shelf
    ridge = np.interp(yy[:, 0], [0, 30, 62, 100, 150, CITY_H], [97, 89, 81, 75, 70, 66])
    ridge = ridge + (_noise(CITY_H, 1, 6, 1, 77)[:, 0] - 0.5) * 6
    dark = np.where(xx < ridge[:, None], 1.15, 0.0) + np.clip((ridge[:, None] - 30 - xx) / 120, 0, 0.5)
    over = (yy >= 128) & (yy <= 156) & (xx > 122) & (xx < 266)
    dark = dark + np.where(over, 1.3 * (1 - np.clip((yy - 136) / 22, 0, 1)), 0)
    rock_tex(ix, ix.mat == MID['ROCK'], seed=3, dark=dark, green=0.25)
    # snow cap with alpenglow
    ix.poly([(97, 0), (89, 9), (86, 14), (90, 13), (93, 16), (96, 12), (99, 15), (103, 11),
             (107, 14), (100, 3)], 'SNOW')
    ix.poly([(97, 1), (89, 9), (86, 14), (90, 13), (93, 16), (95, 10)], 'SNOW_SH')
    # mountain road zig-zagging up the left slope + little trees
    road = [(4, 206), (60, 192), (14, 176), (66, 160), (26, 142), (72, 128), (40, 114), (90, 101)]
    for (xa, ya), (xb, yb) in zip(road, road[1:]):
        ix.line(xa, ya, xb, yb, 'GRAVEL_SH')
        ix.line(xa, ya + 1, xb, yb + 1, 'ROCK_DK')
    for i, (tx, ty) in enumerate([(10, 120), (20, 112), (34, 100), (47, 88), (6, 150), (52, 140),
                                  (30, 166), (58, 178), (16, 196), (44, 208), (62, 116)]):
        tree(ix, tx, ty, 2 + i % 2, 60 + i)
    # --- ledges (z2): upper ledge and the great city shelf with a rugged overhang
    ix.z = 2
    ix.poly([(90, 100), (141, 100), (138, 104), (130, 106), (124, 109), (116, 107), (108, 110),
             (100, 106), (92, 104)], 'ROCK_SH')
    ix.hline(90, 141, 100, 'GR_LT')
    ix.hline(91, 140, 101, 'GR')
    shelf = [(120, 127), (268, 127), (267, 131), (262, 134), (258, 139), (252, 136), (246, 141),
             (240, 138), (234, 143), (226, 139), (218, 142), (210, 140), (202, 145), (196, 140),
             (188, 143), (182, 141), (176, 145), (170, 140), (162, 143), (156, 140), (150, 144),
             (142, 139), (134, 137), (126, 133)]
    ix.poly(shelf, 'ROCK_SH')
    ix.hline(120, 268, 127, 'PAVE')
    ix.hline(121, 267, 128, 'ROCK_HI')
    ix.hline(122, 266, 129, 'ROCK_LT')
    sm = (ix.zz == 2) & (ix.mat == MID['ROCK_SH']) & (yy >= 131)
    rock_tex(ix, sm, seed=21, bias=-0.9, k=0.7)
    # openings under the shelf the falls pour from (dark arches)
    for fx_, ft, fw, fb in FALLS[1:]:
        ix.z = 2
        ix.rect(fx_ - 1, ft - 3, fw + 2, 3, 'ROCK_DK')
        ix.hline(fx_, fx_ + fw - 1, ft - 4, 'ROCK_DK')
    ix.rect(100, 104, 5, 3, 'ROCK_DK')

    # --- upper city on the slope (z3)
    ix.z = 3
    tree(ix, 132, 100, 4, 3)
    tower(ix, 113, 76, 100, 10, wins=wins, flags=flags, roof_h=7, dome=True, needle=3,
          win_rows=[84, 92], balcony=82)
    house(ix, 124, 100, 10, 12, 'R', wins)
    tower(ix, 140, 70, 104, 8, roof='R', roof_h=14, wins=wins, flags=flags, win_rows=[78, 90])
    house(ix, 146, 104, 9, 10, 'G', wins)
    tree(ix, 103, 100, 4, 7)
    # slender back spires behind the hall
    tower(ix, 160, 50, 100, 6, roof='R', roof_h=16, wins=wins, flags=flags, win_rows=[58, 70])
    tower(ix, 212, 52, 100, 6, roof='R', roof_h=15, wins=wins, flags=flags, win_rows=[60, 72])
    tower(ix, 199, 60, 100, 5, roof='G', roof_h=11, flags=flags, band=True)
    # --- Canterlot Tower (z4): slender, tall, violet cone with gold rings
    ix.z = 4
    tower(ix, 238, 50, 126, 9, roof='R', roof_h=24, wins=wins, flags=flags, balcony=58,
          win_rows=[64, 80, 96, 108], needle=5)
    for ry in (72, 90):
        ix.hline(234, 242, ry, 'T')
        ix.px(242, ry, 'T_HI')
        ix.px(234, ry, 'T_SH')
    tower(ix, 255, 100, 126, 7, roof='G', roof_h=10, wins=wins, flags=flags, win_rows=[106, 114])
    # --- castle (z5..z7)
    ix.z = 5
    hx0, hx1, ht, hb = 164, 209, 82, 124
    for x in range(hx0, hx1):
        u = (x - hx0 + 0.5) / (hx1 - hx0)
        ix.vline(x, ht, hb, 'W_SH' if u < 0.06 else ('W' if u < 0.9 else 'W_LT'))
    ix.vline(hx0, ht, hb, 'W_OUT')
    ix.vline(hx1 - 1, ht, hb, 'W_HI')
    # pilasters
    for x in (171, 179, 194, 202):
        ix.vline(x, ht + 2, hb, 'W_LT')
        ix.vline(x - 1, ht + 2, hb, 'W_SH')
    # roof (mansard) with gold crest
    ix.poly([(162, 82.5), (211, 82.5), (204, 71), (169, 71)], 'R')
    ix.poly([(196, 71), (204, 71), (211, 82.5), (200, 82.5)], 'R_LT')
    ix.poly([(162, 82.5), (169, 71), (172, 71), (167, 82.5)], 'R_SH')
    for x in range(171, 203, 4):
        ix.vline(x, 73, 81, 'R_SH')
    ix.hline(169, 204, 70, 'T')
    for x in range(170, 204, 3):
        ix.px(x, 69, 'T_HI')
    ix.hline(162, 211, 83, 'T')
    ix.hline(162, 211, 84, 'T_SH')
    ix.hline(164, 208, 96, 'T_SH')
    # tall arched windows
    for wx in (167, 174, 197, 204):
        ix.rect(wx, 100, 3, 10, 'WIN')
        ix.hline(wx, wx + 2, 99, 'WIN')
        ix.px(wx + 1, 98, 'WIN')
        ix.hline(wx - 1, wx + 3, 110, 'T_SH')
        ix.px(wx + 1, 102, 'WIN_HI')
        wins.append((wx, 99, 3, 11))
    # rose window (stained glass: abstract sun / moon / stars)
    rx, ry = 186, 91
    ix.ellipse(rx, ry, 5, 5, 'T')
    ix.ellipse(rx, ry, 4, 4, 'SG_A')
    for k in range(8):
        a = k * math.pi / 4
        ix.px(rx + round(math.cos(a) * 3), ry + round(math.sin(a) * 3), 'SG_B' if k % 2 else 'SG_D')
    ix.ellipse(rx, ry, 1, 1, 'SG_C')
    ix.px(rx, ry, 'T_HI')
    roses.append((rx, ry, 5))
    # grand doorway
    ix.rect(182, 108, 9, 16, 'T')
    ix.rect(183, 110, 7, 14, 'W_DK')
    ix.hline(184, 188, 109, 'W_DK')
    ix.rect(184, 112, 5, 12, 'WIN')
    wins.append((184, 112, 5, 12))
    ix.px(186, 107, 'T_HI')
    # side towers
    ix.z = 6
    tower(ix, 157, 56, 124, 10, roof='R', roof_h=20, wins=wins, flags=flags, balcony=66,
          win_rows=[72, 86, 100])
    tower(ix, 217, 62, 124, 10, roof='R', roof_h=18, wins=wins, flags=flags, balcony=72,
          win_rows=[78, 92, 104])
    # main keep: tallest, gold cone
    ix.z = 7
    tower(ix, 186, 46, 82, 14, roof='G', roof_h=24, wins=wins, flags=flags, balcony=58,
          win_rows=[50, 64], needle=5)
    # corner turrets on the hall
    ix.z = 8
    tower(ix, 166, 76, 96, 6, roof='R', roof_h=10, flags=flags, band=True)
    tower(ix, 207, 76, 96, 6, roof='R', roof_h=10, flags=flags, band=True)
    # --- curtain wall / front terrace (z9)
    ix.z = 9
    wx0, wx1, wt, wb = 124, 264, 117, 127
    for x in range(wx0, wx1):
        ix.vline(x, wt, wb, 'W')
    ix.hline(wx0, wx1 - 1, wt, 'T')
    ix.hline(wx0, wx1 - 1, wt + 1, 'W_LT')
    for x in range(wx0, wx1, 4):           # merlons
        ix.rect(x, wt - 2, 2, 2, 'W_LT')
        ix.px(x + 1, wt - 2, 'W_HI')
        ix.px(x, wt - 1, 'W_SH')
    for x in range(wx0 + 4, wx1 - 4, 8):   # arcade
        ix.rect(x, wt + 4, 4, 6, 'W_DK')
        ix.hline(x + 1, x + 2, wt + 3, 'W_DK')
        ix.vline(x + 4, wt + 3, wb, 'W_SH')
    ix.hline(wx0, wx1 - 1, wb, 'W_SH')
    ix.vline(wx0, wt - 2, wb, 'W_OUT')
    ix.vline(wx1 - 1, wt - 2, wb, 'W_HI')
    # gatehouse on the left
    tower(ix, 129, 98, 127, 10, roof='R', roof_h=12, wins=wins, flags=flags, win_rows=[104, 112])
    # lamps along the wall top
    for lx in (138, 150, 176, 196, 226, 248, 262):
        ix.vline(lx, wt - 5, wt - 3, 'IRON')
        lamps.append((lx, wt - 6))
    # banners hung on the wall and the hall facade (sun / moon alternating)
    for i, (bx, by, bl, bw) in enumerate([(174, 85, 12, 7), (192, 85, 12, 7), (220, 118, 8, 5),
                                          (232, 118, 8, 5), (154, 118, 8, 5), (143, 118, 8, 5)]):
        banners.append((bx, by, i % 2, i, bl, bw))
    # --- lower tier on the cliff (z4, below the shelf)
    ix.z = 4
    ix.poly([(174, 160), (232, 160), (228, 164), (178, 164)], 'ROCK')
    ix.hline(174, 232, 160, 'ROCK_LT')
    house(ix, 178, 160, 11, 10, 'R', wins)
    house(ix, 214, 160, 12, 9, 'G', wins)
    house(ix, 190, 160, 8, 7, 'R', wins, door=False)
    tree(ix, 227, 160, 3, 11)
    lamps.append((176, 153))
    lamps.append((232, 153))
    # balloon dock: wooden platform jutting from the cliff edge
    ix.z = 10
    ix.rect(262, 150, 18, 2, 'WD')
    ix.hline(262, 279, 150, 'WD_LT')
    ix.line(264, 152, 270, 158, 'WD_DK')
    ix.line(270, 152, 276, 158, 'WD_DK')
    ix.vline(279, 144, 149, 'WD')
    ix.px(279, 143, 'T')
    lamps.append((262, 146))
    ix.vline(262, 147, 149, 'IRON')
    # --- valley foot: talus + plunge pools (z1)
    ix.z = 1
    rng_ = rng(5)
    for k in range(40):
        bx = int(rng_.uniform(0, 300))
        by = int(rng_.uniform(205, 228))
        r_ = int(rng_.uniform(2, 6))
        ix.ellipse(bx, by, r_ + 1, r_ * 0.7, 'ROCK')
        ix.ellipse(bx + 1, by - 1, r_ * 0.6, r_ * 0.35, 'ROCK_LT')
    ix.rect(90, 216, 190, 6, 'POOL')
    ix.poly([(80, 222), (290, 222), (300, 232), (70, 232)], 'GR_DK')
    L = _finish(ix, shadow_len=14, slope=3)
    # window off-times: after the sweep reaches them (+ jitter), so they go out one by one
    offs = []
    for (x, y, w, h) in wins:
        sc = _score(x, y)
        tsw = _sweep_time(sc)
        offs.append(int(tsw + 8 + hash01(x, y, 3) * 26))
    L.update(wins=wins, offs=offs, flags=flags, lamps=lamps, banners=banners, roses=roses)
    return L


# --- sunrise sweep (screen space, shot A camera is still from f 142 on)
_YY, _XX = np.mgrid[0:H, 0:W]
_BAY = bayer(H, W)


def _score(x, y):
    return x + 1.25 * (H - y)


_SCORE = _score(_XX, _YY).astype(np.float32) + _BAY * 18
_SW0, _SW1 = 150, 210


def _sweep_th(f):
    t = seg(f, SUNRISE, _SW1)
    return lerp(420, -30, 0.55 * t + 0.45 * smooth(t))


def _sweep_time(sc):
    for f in range(SUNRISE, _SW1 + 1):
        if _sweep_th(f) < sc + 5:
            return f
    return _SW1


def _sun_mask(f):
    if f < SUNRISE:
        return None, None
    th = _sweep_th(f)
    m = _SCORE > th
    band = m & (_SCORE < th + 8) if f < _SW1 else None
    return m, band


@lru_cache(maxsize=None)
def _far():
    """Far ranges behind the right of the mountain (layer 320x110, ridge tops ~ rows 18-40)."""
    ix = Ix(W, 110)
    ix.z = 1
    pts = [(-2, 40)]
    for x in range(0, W + 8, 8):
        y = 30 + 10 * math.sin(x * 0.045 + 1.0) + 6 * math.sin(x * 0.13)
        if 280 < x < 320:
            y += 6
        pts.append((x, y))
    pts += [(W + 2, 110), (-2, 110)]
    ix.poly(pts, 'FA')
    ix.z = 2
    pts = [(-2, 52)]
    for x in range(0, W + 6, 6):
        y = 44 + 9 * math.sin(x * 0.06 + 2.3) + 4 * math.sin(x * 0.21)
        if 288 < x < 316:
            y += 3
        pts.append((x, y))
    pts += [(W + 2, 110), (-2, 110)]
    ix.poly(pts, 'FB')
    m = ix.mat
    # ridge facets: light on the right-facing slopes
    for y in range(110):
        row = m[y]
    zz = ix.zz
    for z, base, lt in ((1, 'FA', 'FA_LT'), (2, 'FB', 'FB_LT')):
        sel = zz == z
        top = np.argmax(sel, axis=0)
        for x in range(1, W - 1):
            if top[x + 1] > top[x - 1] + 0:  # slope falls to the right -> faces the sun
                y0 = top[x]
                ix.c.a[y0:y0 + 3 + (x % 3), x, 0] = np.where(ix.c.a[y0:y0 + 3 + (x % 3), x, 1] == z, MID[lt],
                                                             ix.c.a[y0:y0 + 3 + (x % 3), x, 0])
    sel = (zz == 2) & ((np.arange(110)[:, None] > 64))
    ix.mat[sel] = MID['FB_SH']
    return _finish(ix, shadow_len=0)


@lru_cache(maxsize=None)
def _valley():
    """Valley floor seen at the start of the tilt: Ponyville roofs, Saddle Lake, meadows."""
    ix = Ix(W, 80)
    wins = []
    ix.z = 1
    ix.poly([(-2, 18), (40, 10), (90, 14), (140, 20), (200, 16), (260, 22), (322, 18), (322, 80), (-2, 80)], 'GR_DK')
    ix.poly([(-2, 22), (60, 16), (120, 24), (180, 20), (240, 26), (322, 24), (322, 80), (-2, 80)], 'GR')
    # Saddle Lake
    ix.poly([(196, 34), (250, 30), (312, 31), (322, 33), (322, 44), (300, 46), (230, 45), (200, 40)], 'POOL')
    ix.hline(206, 296, 33, 'POOL_HI')
    # river from the falls
    ix.poly([(150, 20), (162, 20), (170, 28), (190, 34), (200, 38), (196, 40), (180, 36), (160, 30)], 'POOL')
    ix.z = 2
    # Ponyville (tiny thatched cottages)
    for i, (hx, hb, hw) in enumerate([(30, 34, 7), (39, 33, 6), (48, 35, 8), (58, 33, 6), (67, 35, 7),
                                     (76, 33, 6), (86, 35, 6)]):
        ix.rect(hx, hb - 4, hw, 4, 'CREAM')
        ix.vline(hx, hb - 4, hb - 1, 'W_SH')
        ix.poly([(hx - 1, hb - 3.5), (hx + hw + 0.5, hb - 3.5), (hx + hw / 2, hb - 8)], 'WD_LT')
        ix.line(hx - 1, hb - 4, hx + hw // 2, hb - 8, 'WD')
        wins.append((hx + 2 + (i % 2), hb - 3, 1, 1))
    tower(ix, 56, 26, 33, 4, roof='R', roof_h=6, band=False)
    for i, tx in enumerate((12, 22, 98, 110, 124, 138, 176, 190, 252, 268, 286, 302)):
        tree(ix, tx, 30 + (i % 3) * 3, 3 + (i % 2), 40 + i)
    # meadow texture
    sel = ix.mat == MID['GR']
    yy, xx = np.mgrid[0:80, 0:W]
    ix.mat[sel & (((xx * 3 + yy * 7) % 11) == 0)] = MID['GR_LT']
    L = _finish(ix, shadow_len=0)
    L['wins'] = wins
    return L


@lru_cache(maxsize=None)
def _fore():
    """Foreground framing: dark curly trees in the bottom corners + grass."""
    ix = Ix(W, 70)
    ix.z = 1
    ix.poly([(-2, 40), (60, 46), (120, 52), (200, 54), (260, 46), (322, 38), (322, 70), (-2, 70)], 'GR_DK')
    for cx, cy, r in ((8, 26, 14), (26, 34, 11), (-4, 44, 12), (300, 22, 13), (318, 30, 12), (282, 36, 10)):
        ix.ellipse(cx, cy, r, r * 0.9, 'HEDGE_DK')
        ix.ellipse(cx + r * 0.3, cy - r * 0.35, r * 0.5, r * 0.4, 'GR_DK')
    for x in range(0, W, 3):
        h = 2 + int(hash01(x, 9) * 4)
        ix.vline(x, 52 - h + int(hash01(x, 4) * 6), 60, 'HEDGE_DK')
    return _finish(ix, shadow_len=0)


# =====================================================================================
# Shot A dynamic pieces
# =====================================================================================
_BLUE = ['#1b1934', '#232448', '#2e3062', '#3f4280', '#5c5c9c', '#8a72aa', '#c48cac']
_SUNR = ['#4d65b4', '#6a7fcc', '#8f9de8', '#c4aee6', '#f4b8c4', '#ffcc9c', '#ffe4a0']
_MORN = ['#4d8fe0', '#5ea4ec', '#7cc0f8', '#a4d8ff', '#c4e6ff', '#e0f2ff', '#fff4dc']


def _sky(c, oy, t, morn=0.0, hy=136):
    if morn > 0:
        cols = [mix(a, b, morn) for a, b in zip(_SUNR, _MORN)]
    else:
        cols = [mix(a, b, t) for a, b in zip(_BLUE, _SUNR)]
    c.gradient(0, oy - 20, W, hy - oy + 20, cols, soft=0.7)
    c.rect(0, 0, W, max(0, oy - 20), cols[0])
    c.rect(0, hy, W, H - hy, cols[-1])


def _stars(c, f, oy, fade):
    for i in range(80):
        x = int(hash01(11, i, 1) * W)
        y = int(hash01(11, i, 2) * 100) + oy
        vis = hash01(11, i, 5) * 0.8 + (100 - (y - oy)) / 500
        if vis < fade:
            continue
        ph = hash01(11, i, 3)
        on = 0.5 + 0.5 * math.sin(2 * math.pi * (f / (40 + 50 * ph) + ph))
        if on < 0.2:
            continue
        cc = ('white', 'light_sky', 'lilac')[i % 3]
        if hash01(11, i, 4) > 0.93 and on > 0.6 and fade < 0.4:
            c.px(x, y, 'white')
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c.px(x + dx, y + dy, mix(cc, '#2e3062', 0.5))
        else:
            c.px(x, y, cc if on > 0.6 else mix(cc, '#2e3062', 0.5))


_CLOUDS = [(10, 38, 50, 0), (140, 20, 64, 1), (240, 52, 44, 2), (84, 68, 36, 3), (-50, 86, 54, 4)]


def _wisps(c, f, oy, t, drift=0.07, morn=0.0, clouds=_CLOUDS):
    top = mix('#3a3a72', '#f6d2e0', t)
    low = mix('#7a5c94', '#ffc890', t)
    sh = mix('#2e3062', '#c4a0d0', t)
    if morn > 0:
        top, low, sh = mix(top, 'white', morn), mix(low, 'cream', morn), mix(sh, '#c8d8f4', morn)
    for (x0, y0, w, s) in clouds:
        x = int((x0 + f * drift * (1 + s * 0.15)) % (W + 140)) - 70
        y = y0 + oy
        for k in range(4):
            ex = x + int(hash01(s, k) * w * 0.6)
            ew = int(w * (0.22 + 0.22 * hash01(s, k, 2)))
            ey = y + int(hash01(s, k, 3) * 3) - (k == 1) * 2
            c.ellipse(ex + ew // 2, ey + 1, ew // 2, 1.5, sh)
            c.ellipse(ex + ew // 2, ey, ew // 2, 1.5, top)
            c.hline(ex + 2, ex + ew - 2, ey + 2, low)


@lru_cache(maxsize=None)
def _fall_tex(fw, seed):
    r = rng(seed)
    P = 64
    tex = np.zeros((P, fw), np.int8)
    for k in range(fw):
        y = int(r.integers(0, 8))
        while y < P + 8:
            L = int(r.integers(3, 10))
            v = int(r.choice([1, 2, 2, 2, 3]))
            tex[(np.arange(y, y + L)) % P, k] = v
            y += L
    return tex


_FALL_PAL = {
    'dawn': np.array([col(c) for c in ('#2e3a78', '#4a62a6', '#7894cc', '#b0c8ec', '#d8e4f6')], np.uint8),
    'sun': np.array([col(c) for c in ('#5a70b8', '#8cb4e8', '#c8e4ff', '#fff0d8', 'white')], np.uint8),
    'morn': np.array([col(c) for c in ('crystal', '#86c4f4', 'crystal_lt', 'ice', 'white')], np.uint8),
}


def _falls(c, f, ox, oy, mode='dawn', mask=None, falls=FALLS, speed=2):
    """Animated waterfalls: streak texture scrolling down, foam lip on top, splay at the foot."""
    for i, (fx_, ft, fw, fb) in enumerate(falls):
        x0 = fx_ + ox
        y0, y1 = ft + oy, fb + oy
        ya, yb = max(0, y0), min(H, y1)
        xa, xb = max(0, x0), min(W, x0 + fw)
        if yb <= ya or xb <= xa:
            continue
        tex = _fall_tex(fw, i + 1)
        rows = (np.arange(ya, yb) - oy - f * speed) % tex.shape[0]
        lv = tex[rows].astype(np.int16)
        lv[:, 0] -= 1
        lv[:, -1] += 1 if fw > 3 else 0
        top = np.arange(ya, yb) - y0
        lv[top < 2] = 3
        lv[top < 1] = 4
        lv = np.clip(lv, 0, 4)[:, xa - x0:xb - x0]
        md = mode
        if mask is not None and mask[min(H - 1, max(0, ya + 2)), xa]:
            md = 'sun'
        c.a[ya:yb, xa:xb] = _FALL_PAL[md][lv]
        # foot splay
        for k in range(1, 4):
            yy = y1 - 4 + k
            if 0 <= yy < H:
                c.hline(x0 - k, x0 + fw - 1 + k, yy, _FALL_PAL[md][3 if (f // 3 + k) % 2 else 4])


def _mist(c, f, cx, cy, w, t=0.0, seed=0, morn=False):
    cols = ('#7a84bc', '#a0aad8', '#c8d0ee') if not morn else ('#c4d8f0', '#e2ecff', 'white')
    if t > 0 and not morn:
        cols = tuple(mix(a, b, t) for a, b in zip(cols, ('#d8b4cc', '#f6d4d4', '#fff0e0')))
    for k in range(10):
        ph = (f * 0.7 + k * 9 + hash01(seed, k) * 40) % 44
        r = 2.0 + ph / 8
        x = cx + int((hash01(seed, k, 1) - 0.5) * w + (ph / 44) * (k % 3 - 1) * 9)
        y = cy - 1 - int(ph / 4)
        cc = cols[2] if ph < 14 else (cols[1] if ph < 28 else cols[0])
        if ph < 36:
            c.ellipse(x, y, r, r * 0.6, cc)
            if ph > 20:
                c.dither(x - int(r), y - int(r * 0.6), int(2 * r) + 1, int(1.2 * r) + 1, cols[0], 0.5)


_SUNB = Sprite.from_ascii('''
    ...gggg...
    ..gbbbbg..
    .gbbwwbbg.
    gbbwwwwbbg
    gbbwwwwbbg
    gbbbwwbbbg
    gbbbbbbbbg
    .gbbbbbbg.
    ..gbbbbg..
    ...gggg...
''', {'g': 'gold', 'b': 'butter', 'w': 'white'})


def _flag(c, x, y, f, k, colr):
    """Pennant at a spire tip (x, y = top of the needle), flutters."""
    ph = (f // 4 + k) % 3
    c.px(x + 1, y, colr)
    c.px(x + 2, y + (1 if ph == 1 else 0), colr)
    if ph != 2:
        c.px(x + 3, y + (ph == 0), mix(colr, 'ink', 0.25))
    c.px(x + 1, y + 1, mix(colr, 'ink', 0.3))


def _banner(c, x, y, f, kind, i, dawn, unfurl_t, length=10, bw=5):
    """Hanging sun / moon banner; (x, y) = rod. kind 0 = sun (cream/gold), 1 = moon (navy/lilac)."""
    field, edge, emb, hi = (('cream', 'gold', 'amber', 'butter') if kind == 0 else
                            ('#38388a', 'gold', 'lilac', 'white'))
    k = 0.45 if dawn else 0.0
    tint = '#2e2f62'
    fc, ec, mc, hc = (mix(v, tint, k) for v in (field, edge, emb, hi))
    sc = mix(field, '#1c1a3a', 0.25 + k * 0.5)
    r1 = bw - 1
    c.hline(x - 1, x + bw, y, mix('gold', tint, k))
    c.px(x - 1, y - 1, ec)
    c.px(x + bw, y - 1, ec)
    L = int(round(length * unfurl_t))
    if L <= 1:
        c.rect(x, y + 1, bw, 2, sc)
        c.hline(x, x + r1, y + 1, fc)
        c.px(x + bw // 2, y + 3, ec)
        return
    for r in range(L):
        sway = int(round(math.sin(f / 10.0 + i + r * 0.25) * 0.9)) if r > length // 2 else 0
        c.hline(x + sway, x + r1 + sway, y + 1 + r, fc)
        c.px(x + sway, y + 1 + r, ec)
        c.px(x + r1 + sway, y + 1 + r, ec)
        c.px(x + r1 - 1 + sway, y + 1 + r, sc if r > 1 else fc)
    sway = int(round(math.sin(f / 10.0 + i + L * 0.25) * 0.9)) if L > length // 2 else 0
    ty = y + 1 + L
    for xx in range(bw):
        if xx != bw // 2:
            c.px(x + xx + sway, ty, ec if xx in (0, r1) else fc)
    c.px(x + bw // 2 + sway, ty - 1, ec)
    if L >= 7:
        ex, ey = x + bw // 2, y + 4
        if kind == 0:      # sun: disc with rays
            c.rect(ex - 1, ey, 3, 3, mc)
            c.px(ex, ey + 1, hc)
            c.px(ex, ey - 1, mc)
            c.px(ex, ey + 3, mc)
            if bw >= 7:
                c.px(ex - 2, ey + 1, mc)
                c.px(ex + 2, ey + 1, mc)
        else:              # moon: crescent (+ a star on wide banners)
            c.rect(ex - 1, ey, 2, 3, mc)
            c.px(ex + 1, ey - 1, mc)
            c.px(ex + 1, ey + 3, mc)
            c.px(ex - 1, ey + 1, hc)
            if bw >= 7:
                c.px(ex + 2, ey + 1, hc)


def _lit_window(c, x, y, w, h, f, i):
    flick = hash01(i, f // 5) > 0.94
    c.rect(x, y, w, h, 'amber' if flick else 'gold')
    c.px(x + w // 2, y + h - 1 - h // 3, 'butter')


@lru_cache(maxsize=None)
def _ridge_y(x):
    a = _far()['alpha'][:, x]
    return int(np.argmax(a))


def _tilt(f):
    return cruise(seg(f, 74, 141))


def _shot_a(f):
    tilt = _tilt(f)
    tsun = smooth(seg(f, SUNRISE - 12, 204))            # sky cross-fade
    oy_city = int(round(-70 + 70 * tilt))
    oy_sky = int(round(-30 + 30 * tilt))
    oy_far = 92 + oy_sky
    oy_val = int(round(118 + 122 * tilt))
    oy_fg = int(round(116 + 150 * tilt))
    c = Canvas()
    _sky(c, oy_sky, tsun, hy=126 + oy_sky)
    _stars(c, f, oy_sky, clamp(seg(f, 84, 176) * 1.1))
    # horizon pre-glow, then the sun breaks the ridge exactly on the hit
    sx = 300
    ry = _ridge_y(sx) + oy_far
    pre = seg(f, 90, SUNRISE)
    fx.glow(c, sx, ry + 4, 80, '#ff8a74', 0.08 + 0.20 * pre)
    if f >= SUNRISE:
        sy = int(ry + 3 - (f - SUNRISE) * 0.30)
        flash = 1 - seg(f, SUNRISE, SUNRISE + 14)
        fx.glow(c, sx, sy, 50, 'gold', 0.30 + 0.25 * flash)
        fx.glow(c, sx, sy, 18, 'butter', 0.45 + 0.4 * flash)
        _SUNB.draw(c, sx, sy + 5, anchor='bottom')
    _wisps(c, f, oy_sky, tsun)
    m, band = _sun_mask(f)
    _draw_layer(c, _far(), 0, oy_far, 'dawn', 'sun', m, band)
    if SUNRISE <= f < SUNRISE + 44:                     # burst rays over the far ridge
        k = 1 - seg(f, SUNRISE, SUNRISE + 44)
        for a in range(7):
            ang = math.pi * (0.70 + a * 0.10) + f * 0.003
            x2 = sx + int(math.cos(ang) * 150)
            y2 = ry + int(math.sin(ang) * 150)
            fx.light_beam(c, ((sx - 1, ry), (sx + 1, ry)), ((x2 - 5, y2), (x2 + 5, y2)), 'butter', 0.10 * k)
    L = _city()
    _draw_layer(c, L, 0, oy_city, 'dawn', 'sun', m, band)
    _falls(c, f, 0, oy_city, 'dawn', m)
    for i, (fx_, ft, fw, fb) in enumerate(FALLS):
        _mist(c, f, fx_ + fw // 2, fb + oy_city, 12 + fw * 2, tsun, seed=i)
    for k in range(6):                                   # plunge-pool shimmer
        x = 96 + int((hash01(k, 3) * 180 + f * 0.5 * (1 if k % 2 else -1)) % 180)
        c.hline(x, x + 3, 218 + oy_city, '#8a9ee0')
    # lit windows switch off one by one behind the sweep
    for i, ((x, y, w, h), off) in enumerate(zip(L['wins'], L['offs'])):
        if f < off:
            _lit_window(c, x, y + oy_city, w, h, f, i)
            if h > 5:
                fx.glow(c, x + w // 2, y + oy_city + h // 2, 6, 'amber', 0.16)
    # stained glass: glows at dawn, glints as the light reaches it
    for (rx, ry_, r) in L['roses']:
        y = ry_ + oy_city
        if f < SUNRISE + 30:
            fx.glow(c, rx, y, 9, 'magic', 0.20 * (1 - seg(f, SUNRISE, SUNRISE + 30)))
        if m is not None and m[min(H - 1, max(0, y)), rx]:
            fx.sparkle(c, rx + 3, y - 3, f, 'white', period=22, size=3)
            fx.sparkle(c, rx - 3, y + 2, f + 11, 'butter', period=22, size=2)
    # terrace lamps
    for i, (lx, ly) in enumerate(L['lamps']):
        y = ly + oy_city
        on = f < 184 + int(hash01(i, 7) * 22)
        c.px(lx, y, 'butter' if on else 'brass')
        c.px(lx, y - 1, '#3a3048')
        if on:
            fx.glow(c, lx, y, 6, 'amber', 0.30 if hash01(i, f // 4) > 0.15 else 0.22)
    # pennants on every spire
    for i, (px_, py) in enumerate(L['flags']):
        lit = m is not None and m[min(H - 1, max(0, py + oy_city)), min(W - 1, px_)]
        _flag(c, px_, py + oy_city, f, i, ('gold' if i % 2 else 'magic') if lit else
              ('#6e6474' if i % 2 else '#5a3a7a'))
    # banners: rolled at dawn, unfurl on the half-beats after the light reaches them
    for (bx, by, kind, i, bl, bw) in L['banners']:
        tsw = _sweep_time(_score(bx, by + 4))
        t0 = max(SUNRISE + 9, ((tsw + 8) // 9 + 1) * 9)
        u = ease_out_bounce(seg(f, t0, t0 + 10))
        lit = m is not None and m[min(H - 1, max(0, by + oy_city)), bx]
        _banner(c, bx, by + oy_city - 1, f, kind, i, not lit, u, bl, bw)
    # balloon moored at the cliff dock
    byl = 149 + oy_city
    c.line(275, byl - 4, 279, 143 + oy_city, '#6a5a6a' if f < SUNRISE else 'tan')
    draw_balloon(c, 272, byl, f, 's')
    # valley + foreground (leave the frame during the tilt)
    if oy_val < H:
        V = _valley()
        _draw_layer(c, V, 0, oy_val, 'dawn')
        for i, (x, y, w, h) in enumerate(V['wins']):
            if hash01(i, f // 12) > 0.12:
                c.px(x, y + oy_val, 'gold')
        for k in range(6):                               # Saddle Lake shimmer
            x = 210 + int((hash01(k, 8) * 90 + f * 0.4) % 90)
            c.hline(x, x + 2, oy_val + 35 + k % 3 * 3, '#c48cac' if k % 2 else '#8a72aa')
        for k in range(3):                               # chimney smoke in Ponyville
            fx.steam(c, 34 + k * 22, oy_val + 24, f, '#5c5c8c', n=3, height=10, seed=k)
    if oy_fg < H:
        _draw_layer(c, _fore(), 0, oy_fg, 'dawn')
    return c


# =====================================================================================
# SHOT B: royal gardens & the mountain railway (bar 4)
# =====================================================================================
def hedge_wall(ix, x0, x1, ytop, h):
    """Maze hedge seen from a raised 3/4 view: lit top, leafy front, dark foot."""
    ix.rect(x0, ytop, x1 - x0 + 1, 2, 'HEDGE_LT')
    ix.hline(x0 + (x1 - x0) * 2 // 3, x1, ytop, 'HEDGE_HI')
    ix.rect(x0, ytop + 2, x1 - x0 + 1, h - 2, 'HEDGE')
    ix.hline(x0, x1, ytop + h - 1, 'HEDGE_DK')
    ix.vline(x0, ytop + 1, ytop + h - 1, 'HEDGE_DK')
    ix.vline(x1, ytop, ytop + h - 2, 'HEDGE_LT')
    for x in range(x0 + 1, x1):
        for y in range(ytop + 2, ytop + h - 1):
            v = hash01(x, y, 17)
            if v > 0.82:
                ix.px(x, y, 'HEDGE_LT')
            elif v < 0.12:
                ix.px(x, y, 'HEDGE_DK')


def topi_cone(ix, cx, by, h, half, tiers=3):
    ix.vline(cx, by - 2, by, 'WD')
    for t in range(tiers):
        tb = by - 2 - t * (h // (tiers + 1))
        th = h // tiers + 2
        hw = half * (1 - t * 0.28)
        for k in range(th):
            y = tb - th + k
            w_ = hw * (k + 1) / th
            x0, x1 = int(round(cx - w_)), int(round(cx + w_))
            ix.hline(x0, x1, y, 'HEDGE')
            ix.hline(max(x0, cx), x1, y, 'HEDGE_LT')
            ix.px(x1, y, 'HEDGE_HI')
            ix.px(x0, y, 'HEDGE_DK')
        ix.hline(int(cx - hw), int(cx + hw), tb, 'HEDGE_DK')


def topi_spiral(ix, cx, by, h, half):
    ix.rect(cx - 2, by - 3, 5, 3, 'W')
    ix.hline(cx - 2, cx + 2, by - 3, 'W_HI')
    ix.px(cx - 2, by - 2, 'W_SH')
    for k in range(h):
        y = by - 3 - h + k
        w_ = half * (k + 1) / h
        x0, x1 = int(round(cx - w_)), int(round(cx + w_))
        for x in range(x0, x1 + 1):
            u = (x - x0 + 0.5) / max(1, x1 - x0 + 1)
            band = (y + int(u * 6)) % 5 == 0
            ix.px(x, y, 'HEDGE_DK' if band else ('HEDGE' if u < 0.5 else ('HEDGE_LT' if u < 0.85 else 'HEDGE_HI')))


def topi_ball(ix, cx, by, r, stem=4, balls=1):
    ix.rect(cx - 2, by - 3, 5, 3, 'W')
    ix.hline(cx - 2, cx + 2, by - 3, 'W_HI')
    ix.vline(cx, by - 3 - stem, by - 3, 'WD')
    y = by - 3 - stem - r
    for b in range(balls):
        rr = r - b
        ix.ellipse(cx, y, rr, rr, 'HEDGE')
        ix.ellipse(cx + rr // 3, y - rr // 3, max(1, rr * 0.6), max(1, rr * 0.55), 'HEDGE_LT')
        ix.px(cx + rr // 2, y - rr // 2, 'HEDGE_HI')
        ix.ellipse(cx - rr // 2, y + rr // 2, max(0.5, rr * 0.35), max(0.5, rr * 0.3), 'HEDGE_DK')
        y -= 2 * rr + 1
        if b < balls - 1:
            ix.vline(cx, y + rr, y + rr + 1, 'WD')


def topi_cube(ix, x, by, s):
    ix.rect(x, by - s, s, s, 'HEDGE')
    ix.rect(x, by - s, s, 2, 'HEDGE_LT')
    ix.vline(x + s - 1, by - s, by - 1, 'HEDGE_LT')
    ix.hline(x, x + s - 1, by - 1, 'HEDGE_DK')
    ix.vline(x, by - s + 1, by - 1, 'HEDGE_DK')


def urn(ix, cx, by, flw='FLW_A'):
    ix.rect(cx - 2, by - 4, 5, 4, 'W')
    ix.vline(cx + 2, by - 4, by - 1, 'W_HI')
    ix.vline(cx - 2, by - 4, by - 1, 'W_SH')
    ix.hline(cx - 3, cx + 3, by - 5, 'W_LT')
    ix.hline(cx - 1, cx + 1, by - 6, 'W')
    ix.ellipse(cx, by - 9, 3, 2.5, 'W')
    ix.ellipse(cx + 1, by - 10, 1.5, 1.5, 'W_HI')
    ix.px(cx - 2, by - 8, 'W_SH')
    ix.hline(cx - 3, cx + 3, by - 12, 'T')
    ix.px(cx + 3, by - 12, 'T_HI')
    for k in range(5):
        ix.px(cx - 2 + k, by - 13 - (k % 2), flw if k % 2 else 'HEDGE_LT')
    ix.px(cx, by - 15, flw)


def obelisk(ix, cx, by, h):
    ix.rect(cx - 3, by - 3, 7, 3, 'W')
    ix.hline(cx - 3, cx + 3, by - 3, 'W_HI')
    ix.vline(cx + 3, by - 3, by - 1, 'W_LT')
    for k in range(h):
        y = by - 3 - h + k
        hw = 1 + (k * 1.3) / h
        x0, x1 = int(round(cx - hw)), int(round(cx + hw))
        ix.hline(x0, x1, y, 'W')
        ix.px(x0, y, 'W_SH')
        ix.px(x1, y, 'W_HI')
    ix.px(cx, by - 3 - h - 1, 'T')
    ix.hline(cx - 1, cx + 1, by - 3 - h, 'T')
    ix.px(cx + 1, by - 3 - h, 'T_HI')
    ix.px(cx, by - 3 - h // 2, 'T')


def fountain(ix, cx, by, r):
    """Tiered fountain; returns (top jet anchor, upper bowl centre)."""
    ix.ellipse(cx, by - 2, r, 3, 'W_LT')
    ix.ellipse(cx, by - 3, r - 1, 2, 'POOL')
    ix.hline(cx - r + 2, cx + r - 3, by - 4, 'POOL_HI')
    ix.rect(cx - r, by - 2, 2 * r + 1, 2, 'W')
    ix.hline(cx - r, cx + r, by, 'W_SH')
    ix.vline(cx + r, by - 2, by, 'W_HI')
    top = max(6, r - 1)
    ix.rect(cx - 1, by - top, 3, top - 3, 'W')
    ix.vline(cx + 1, by - top, by - 4, 'W_HI')
    ix.ellipse(cx, by - top, max(2, r // 2), 1.5, 'W_LT')
    ix.ellipse(cx, by - top - 1, max(1, r // 2 - 1), 1, 'POOL')
    ix.hline(cx - r // 2, cx + r // 2, by - top + 1, 'T')
    ix.rect(cx, by - top - 6, 1, 5, 'W')
    ix.ellipse(cx, by - top - 6, 1.5, 1, 'W_LT')
    ix.px(cx, by - top - 8, 'T')
    return (cx, by - top - 8), (cx, by - top - 1)


@lru_cache(maxsize=None)
def _castle_b():
    """Castle wall and towers rising behind the gardens (parallax 0.6); ends in a corner tower."""
    ix = Ix(224, 118)
    flags, banners = [], []
    ix.z = 1
    tower(ix, 122, 12, 117, 16, roof='G', roof_h=28, flags=flags, balcony=28, win_rows=[36, 54],
          needle=5)
    tower(ix, 64, 40, 117, 9, roof='R', roof_h=18, flags=flags, win_rows=[48, 62])
    tower(ix, 174, 36, 117, 9, roof='R', roof_h=18, flags=flags, win_rows=[46, 60])
    ix.z = 2
    wx1 = 200
    for x in range(0, wx1):
        ix.vline(x, 64, 117, 'W')
    ix.hline(0, wx1 - 1, 64, 'T')
    ix.hline(0, wx1 - 1, 65, 'W_LT')
    for x in range(0, wx1, 4):
        ix.rect(x, 62, 2, 2, 'W_LT')
        ix.px(x + 1, 62, 'W_HI')
    for x in range(6, wx1 - 8, 14):      # tall arched windows + pilasters
        if 80 <= x <= 110:
            continue
        ix.rect(x, 74, 4, 14, 'WIN')
        ix.hline(x + 1, x + 2, 73, 'WIN')
        ix.hline(x - 1, x + 4, 88, 'W_HI')
        ix.px(x + 2, 76, 'WIN_HI')
        ix.vline(x + 9, 66, 117, 'W_LT')
        ix.vline(x + 8, 66, 117, 'W_SH')
    ix.hline(0, wx1 - 1, 96, 'T_SH')
    for x in range(0, wx1, 6):
        ix.rect(x + 1, 100, 3, 8, 'W_DK')
    ix.z = 3
    tower(ix, 30, 28, 117, 14, roof='R', roof_h=24, flags=flags, balcony=42, win_rows=[50, 68],
          needle=5)
    tower(ix, 150, 50, 117, 10, roof='G', roof_h=16, flags=flags, win_rows=[58, 72])
    tower(ix, 204, 40, 117, 18, roof='R', roof_h=26, flags=flags, balcony=54, win_rows=[46, 62, 78],
          needle=5)
    # chapel facade with stained glass
    ix.rect(84, 70, 22, 47, 'W')
    ix.vline(105, 70, 116, 'W_HI')
    ix.vline(84, 70, 116, 'W_OUT')
    ix.poly([(82, 70.5), (108, 70.5), (95, 56)], 'R')
    ix.poly([(95, 56), (108, 70.5), (99, 70.5)], 'R_LT')
    ix.hline(82, 108, 70, 'T')
    ix.ellipse(95, 82, 5, 5, 'T')
    ix.ellipse(95, 82, 4, 4, 'SG_A')
    for k in range(8):
        a = k * math.pi / 4
        ix.px(95 + round(math.cos(a) * 3), 82 + round(math.sin(a) * 3), 'SG_B' if k % 2 else 'SG_D')
    ix.ellipse(95, 82, 1, 1, 'SG_C')
    ix.rect(92, 96, 7, 20, 'T')
    ix.rect(93, 98, 5, 19, 'WIN')
    for i, bx in enumerate((14, 44, 136, 162, 186)):
        banners.append((bx, 67, i % 2, i, 14, 6))
    L = _finish(ix, shadow_len=12, slope=3)
    L.update(flags=flags, banners=banners, roses=[(95, 82)])
    return L


TRACK_C, TRACK_R, TRACK_Y0, TRACK_PITCH = 238, 27, 80, 26
PORTAL_X = 300


def _helix(phi):
    return (TRACK_C + TRACK_R * math.cos(phi), TRACK_Y0 + TRACK_PITCH * phi / (2 * math.pi))


@lru_cache(maxsize=None)
def _rail():
    """Rock spur with the corkscrew railway, viaduct and tunnel portal in the cliff (parallax 0.85)."""
    ix = Ix(372, 180)
    yy, xx = np.mgrid[0:180, 0:372]
    # back passes of the helix (behind the spur)
    ix.z = 1
    for turn in range(5):
        for k in range(120):
            phi = math.pi + turn * 2 * math.pi + k * math.pi / 120
            x, y = _helix(phi)
            ix.px(round(x), round(y), 'IRON')
            ix.px(round(x), round(y) + 1, 'WD_DK')
            ix.px(round(x), round(y) + 2, 'ROCK_SH')
    # the spur
    ix.z = 2
    spur = [(212, 180), (214, 140), (211, 110), (216, 84), (222, 70), (232, 64), (244, 63), (254, 68),
            (262, 82), (264, 110), (261, 140), (266, 180)]
    ix.poly(spur, 'ROCK')
    sel = (ix.zz == 2) & (ix.mat == MID['ROCK'])
    rock_tex(ix, sel, seed=31, dark=np.clip((240 - xx) / 22, 0, 1.2), strata=9, green=0.3)
    tree(ix, 236, 64, 4, 3)
    tree(ix, 247, 64, 3, 5)
    # main cliff with the tunnel portal
    ix.z = 3
    cliff = [(372, 0), (300, 0), (294, 20), (298, 40), (288, 58), (286, 90), (292, 120), (286, 150),
             (290, 180), (372, 180)]
    ix.poly(cliff, 'ROCK')
    sel = (ix.zz == 3) & (ix.mat == MID['ROCK'])
    rock_tex(ix, sel, seed=41, dark=np.clip((300 - xx) / 30, 0, 0.6), strata=10, green=0.3)
    tree(ix, 306, 26, 4, 8)
    tree(ix, 318, 20, 3, 9)
    # viaduct from the portal to the spur
    ix.z = 4
    ix.rect(258, 81, 44, 3, 'W')
    ix.hline(258, 301, 81, 'W_HI')
    for ax in (262, 274, 286):
        ix.rect(ax, 84, 3, 22, 'W')
        ix.vline(ax + 2, 84, 105, 'W_HI')
        ix.vline(ax, 84, 105, 'W_SH')
    ix.hline(258, 301, 80, 'IRON')
    for x in range(258, 302, 2):
        ix.px(x, 81, 'WD')
    # front passes (on carved ledges round the spur)
    ix.z = 5
    for turn in range(5):
        for k in range(121):
            phi = turn * 2 * math.pi + k * math.pi / 120
            x, y = _helix(phi)
            x, y = round(x), round(y)
            ix.px(x, y - 1, 'IRON')
            ix.px(x, y, 'WD' if k % 3 else 'WD_DK')
            ix.px(x, y + 1, 'ROCK_LT')
            ix.px(x, y + 2, 'ROCK_SH')
            if k % 14 == 7 and 15 < k < 105:
                ix.vline(x, y + 3, y + 5, 'WD_DK')
    # portal
    ix.z = 6
    px0 = PORTAL_X - 8
    ix.rect(px0, 66, 17, 16, 'W')
    ix.ellipse(PORTAL_X, 67, 8, 5, 'W')
    ix.rect(px0 + 3, 70, 11, 11, 'ROCK_DK')
    ix.ellipse(PORTAL_X, 70, 5, 3, 'ROCK_DK')
    ix.vline(px0, 66, 81, 'W_SH')
    ix.vline(px0 + 16, 66, 81, 'W_HI')
    ix.hline(PORTAL_X - 1, PORTAL_X + 1, 62, 'T')
    ix.hline(PORTAL_X - 2, PORTAL_X + 2, 63, 'T')
    ix.px(PORTAL_X + 2, 63, 'T_HI')
    ix.hline(px0 - 1, px0 + 17, 81, 'W_SH')
    ix.hline(px0 + 3, px0 + 13, 80, 'IRON')
    return _finish(ix, shadow_len=10, slope=3)


@lru_cache(maxsize=None)
def _garden():
    """The terrace: promenade, hedge maze, topiary, fountains, urns, obelisks, gazebo (parallax 1)."""
    ix = Ix(300, 180)
    jets = []
    yy, xx = np.mgrid[0:180, 0:300]
    ix.z = 1
    # terrace ground (gravel) + retaining wall at the right edge
    ix.rect(0, 112, 240, 68, 'GR_LT')
    sel = (ix.mat == MID['GR_LT'])
    ix.mat[sel & (((yy - 112) // 3) % 2 == 1)] = MID['HEDGE_LT']      # mowing stripes
    for (yt, h) in ((121, 4), (133, 5), (146, 6), (162, 7)):           # gravel walks
        ix.rect(0, yt + h, 236, 3, 'GRAVEL')
        ix.hline(0, 235, yt + h + 2, 'GRAVEL_SH')
    ix.ellipse(125, 158, 26, 7, 'GRAVEL')
    ix.ellipse(72, 143, 12, 4, 'GRAVEL')
    ix.ellipse(186, 143, 12, 4, 'GRAVEL')
    sel = (ix.mat == MID['GRAVEL'])
    ix.mat[sel & (((xx * 7 + yy * 13) % 17) == 0)] = MID['GRAVEL_SH']
    for x in range(236, 246):
        ix.vline(x, 112 + (x - 236), 180, 'W' if x < 243 else 'W_LT')
    ix.vline(245, 121, 180, 'W_HI')
    for y in range(124, 180, 6):
        ix.hline(237, 245, y, 'W_SH')
    # promenade + balustrade at the back
    ix.rect(0, 108, 240, 6, 'PAVE')
    ix.hline(0, 239, 113, 'GRAVEL_SH')
    ix.z = 2
    ix.hline(0, 244, 101, 'W_HI')
    ix.hline(0, 244, 102, 'W_LT')
    for x in range(1, 244, 3):
        ix.vline(x, 103, 107, 'W')
        ix.px(x + 1, 104, 'W_SH')
    ix.hline(0, 244, 107, 'W_SH')
    for x in range(0, 245, 30):
        ix.rect(x, 98, 4, 10, 'W')
        ix.vline(x + 3, 98, 107, 'W_HI')
        ix.hline(x - 1, x + 4, 97, 'T')
    # lawns inside the maze cells
    ix.z = 1
    for (x0, y0, w_) in ((8, 129, 30), (176, 141, 40), (20, 155, 22), (180, 172, 40), (52, 172, 30)):
        for k in range(w_ // 3):
            ix.px(x0 + k * 3, y0, ('FLW_A', 'FLW_B', 'FLW_C', 'FLW_D')[k % 4])
    # flower beds in the maze plaza
    for k in range(40):
        ix.px(104 + int(hash01(k, 1) * 42), 160 + int(hash01(k, 2) * 5),
              ('FLW_A', 'FLW_B', 'FLW_C', 'FLW_D')[k % 4])
    # promenade topiary, obelisks, urns (back row)
    ix.z = 3
    for bx in (16, 46, 180, 210):
        topi_ball(ix, bx, 114, 4, stem=5, balls=2 if bx in (46, 180) else 1)
    obelisk(ix, 78, 114, 18)
    obelisk(ix, 150, 114, 18)
    urn(ix, 100, 114, 'FLW_A')
    urn(ix, 128, 114, 'FLW_C')
    # gazebo at the terrace corner
    gx = 222
    ix.rect(gx - 13, 110, 27, 3, 'W_LT')
    ix.hline(gx - 13, gx + 13, 110, 'W_HI')
    for cxx in (gx - 11, gx - 5, gx + 1, gx + 7, gx + 12):
        ix.vline(cxx, 88, 109, 'W' if cxx < gx + 7 else 'W_HI')
        if cxx < gx:
            ix.vline(cxx - 1, 88, 109, 'W_SH')
    ix.rect(gx - 13, 85, 27, 3, 'T')
    ix.hline(gx - 13, gx + 13, 85, 'T_HI')
    rx = 14
    for x in range(gx - rx, gx + rx + 1):
        u = (x - (gx - rx) + 0.5) / (2 * rx + 1)
        hh = int(round(12 * math.sqrt(max(0.0, 1 - ((x - gx) / (rx + 0.5)) ** 2))))
        ix.vline(x, 85 - hh, 84, _m('R', _cyl(u)))
    for x in range(gx - rx + 3, gx + rx - 2, 5):
        hh = int(round(12 * math.sqrt(max(0.0, 1 - ((x - gx) / (rx + 0.5)) ** 2))))
        ix.vline(x, 85 - hh + 1, 84, 'T_SH')
    finial(ix, gx, 72, 4)
    # the maze, back to front
    rows = [
        (121, 4, [(0, 40), (52, 96), (150, 196), (206, 234)]),
        (133, 5, [(10, 70), (82, 100), (150, 162), (172, 228)]),
        (146, 6, [(0, 30), (44, 94), (156, 200), (212, 234)]),
        (162, 7, [(14, 64), (76, 102), (148, 176), (190, 234)]),
    ]
    links = {0: [40, 96, 196], 1: [70, 172, 222], 2: [30, 94, 156, 200]}
    for r, (yt, h, segs) in enumerate(rows):
        ix.z = 4 + r * 2
        for (a, b) in segs:
            hedge_wall(ix, a, b, yt, h)
        if r in links:
            ny = rows[r + 1][0]
            ix.z = 5 + r * 2
            for lx in links[r]:
                for y in range(yt + h - 1, ny + 2):
                    ix.hline(lx - 1, lx + 2, y, 'HEDGE')
                    ix.px(lx - 1, y, 'HEDGE_DK')
                    ix.px(lx + 2, y, 'HEDGE_LT')
                ix.hline(lx - 1, lx + 2, yt + h - 1, 'HEDGE_LT')
        ix.z = 5 + r * 2
        if r == 0:
            for tx in (52, 96, 150, 196):
                topi_cone(ix, tx, 122, 10, 4, tiers=2)
        if r == 1:
            topi_cube(ix, 10, 134, 6)
            topi_cube(ix, 222, 134, 6)
        if r == 2:
            topi_spiral(ix, 104, 152, 16, 5)
            topi_spiral(ix, 146, 152, 16, 5)
            urn(ix, 36, 152, 'FLW_D')
            urn(ix, 206, 152, 'FLW_A')
    # central fountain and two small ones
    ix.z = 9
    j1, j2 = fountain(ix, 125, 160, 14)
    jets.append((125, 160, 14, j1, j2))
    ix.z = 7
    j1, j2 = fountain(ix, 72, 144, 7)
    jets.append((72, 144, 7, j1, j2))
    j1, j2 = fountain(ix, 186, 144, 7)
    jets.append((186, 144, 7, j1, j2))
    L = _finish(ix, shadow_len=10, slope=2)
    L['jets'] = jets
    return L


_FG_HEDGES = ((0, 46), (78, 150), (190, 250), (290, 360), (380, 400))


@lru_cache(maxsize=None)
def _garden_fg():
    """Foreground framing: clipped hedge blocks, a big urn, flowers (parallax 1.35)."""
    ix = Ix(400, 40)
    ix.z = 1
    for (x0, x1) in _FG_HEDGES:
        ix.rect(x0, 20, x1 - x0, 20, 'HEDGE')
        ix.rect(x0, 18, x1 - x0, 3, 'HEDGE_LT')
        ix.hline(x0 + (x1 - x0) // 2, x1 - 1, 18, 'HEDGE_HI')
        ix.vline(x1 - 1, 18, 39, 'HEDGE_LT')
        ix.vline(x0, 19, 39, 'HEDGE_DK')
        for x in range(x0 + 1, x1 - 1):
            for y in range(22, 40):
                v = hash01(x, y, 23)
                if v > 0.8:
                    ix.px(x, y, 'HEDGE_LT')
                elif v < 0.18:
                    ix.px(x, y, 'HEDGE_DK')
    for k in range(70):
        x = int(hash01(k, 31) * 400)
        if any(a - 2 <= x <= b + 2 for a, b in _FG_HEDGES):
            continue
        ix.px(x, 30 + int(hash01(k, 32) * 8), ('FLW_A', 'FLW_B', 'FLW_C', 'FLW_D')[k % 4])
        ix.px(x, 31 + int(hash01(k, 32) * 8), 'HEDGE')
    ix.z = 2
    cx = 62                                    # big urn on a pedestal
    ix.rect(cx - 6, 24, 13, 16, 'W')
    ix.vline(cx + 6, 24, 39, 'W_HI')
    ix.vline(cx - 6, 24, 39, 'W_SH')
    ix.hline(cx - 7, cx + 7, 23, 'T')
    ix.ellipse(cx, 14, 8, 6, 'W')
    ix.ellipse(cx + 3, 12, 3, 3, 'W_HI')
    ix.ellipse(cx - 5, 16, 2, 3, 'W_SH')
    ix.hline(cx - 8, cx + 8, 7, 'T')
    ix.hline(cx - 2, cx + 8, 7, 'T_HI')
    ix.rect(cx - 2, 19, 5, 4, 'W')
    for k in range(12):
        ix.ellipse(cx - 7 + k * 1.3, 4 - (k % 3), 1.5, 1.5, ('FLW_A', 'HEDGE_LT', 'FLW_B')[k % 3])
    topi_cone(ix, 330, 22, 24, 8, tiers=3)
    return _finish(ix, shadow_len=0)


_ENGINE = Sprite.from_ascii('''
    ..SS.............
    .SSSS.....RRRRRRR
    ..SS......RwwRwwR
    .LPPPPPPPkPwwPwwP
    kPWWPPWWPkPPPPPPP
    cPPPPPPPPkPPPPPPP
    GkkkkkkkkkkkkkkkG
    ..OO..OO....OO.OO
''', {'S': 'plum', 'R': 'magenta', 'w': 'cream', 'P': 'bubblegum', 'W': 'white', 'k': 'raspberry',
      'G': 'gold', 'O': 'navy', 'L': 'butter', 'c': 'gold'})
_CAR = Sprite.from_ascii('''
    .CCCCCCCCCCCC.
    CCCCCCCCCCCCCC
    PwwPPwwPPwwPPP
    PwwPPwwPPwwPPP
    PWWWWWWWWWWWWP
    PPPPPPPPPPPPPP
    kkkkkkkkkkkkkk
    .OO.......OO..
''', {'C': 'white', 'w': 'cream', 'P': 'bubblegum', 'W': 'white', 'k': 'raspberry', 'O': 'navy'})
TRAIN_T0 = 252


def _train_path(s):
    """Railway-layer position (x, y of the rail) at distance s past the portal mouth."""
    if s <= 38:
        return PORTAL_X - s, 80.0
    phi = (s - 38) / TRACK_R
    return _helix(phi)


def _train_s(f):
    return (f - TRAIN_T0) * 1.05 + 3


def _draw_train(c, f, ox):
    """Train in screen space; parts still inside the tunnel are clipped at the portal."""
    if f < TRAIN_T0 - 2:
        return
    se = _train_s(f)
    clip_x = int(PORTAL_X - 5 + ox)
    over = Canvas(W, H, (1, 2, 3))
    for s_, spr, half in ((se, _ENGINE, 8), (se - 17, _CAR, 7), (se - 32, _CAR, 7), (se - 47, _CAR, 7)):
        x, y = _train_path(s_ + half)
        x2, y2 = _train_path(s_)
        cx = int(round((x + x2) / 2 + ox))
        cy = int(round(max(y, y2)))
        spr.draw(over, cx, cy, anchor='bottom')
    m = (over.a != np.array((1, 2, 3), np.uint8)).any(axis=2)
    m[:, max(0, clip_x):] = False
    c.a[m] = over.a[m]


def _steam_puffs(c, f, ox):
    """Puffs on every half-beat from the stack + a burst from the tunnel mouth on the hit."""
    out = []
    for k in range(9):
        t = TRAIN_T0 + k * 9
        if f < t:
            break
        age = f - t
        if age > 34:
            continue
        x, y = _train_path(_train_s(t) + 3)
        r = 1.5 + age * 0.16
        px = x + ox + 2 + age * 0.35
        py = y - 10 - age * 0.55
        colr = 'white' if age < 12 else ('cloud' if age < 22 else 'cloud_dk')
        out.append((px, py, r, colr))
    for (px, py, r, colr) in sorted(out, key=lambda q: -q[2]):
        c.ellipse(int(px), int(py), r, r * 0.8, colr)
        c.ellipse(int(px + r * 0.3), int(py - r * 0.3), max(0.5, r * 0.5), max(0.5, r * 0.4), 'white')
    if TRAIN_T0 - 1 <= f < TRAIN_T0 + 18:
        age = f - TRAIN_T0 + 1
        for k in range(5):
            r = 2 + age * (0.25 + k * 0.05)
            bx = PORTAL_X + ox - 4 - age * (0.5 + 0.2 * k)
            by = 74 - age * (0.3 + 0.15 * k) + k * 2
            c.ellipse(int(bx), int(by), r, r * 0.7, 'cloud' if k % 2 else 'white')


def _jet(c, f, x, y, h, r, k):
    """Fountain jet: a rising column and arcing droplets."""
    for i in range(h):
        c.px(x, y - i + 1, 'white' if (f // 2 + i) % 3 else 'ice')
    for d in (-1, 1):
        for i in range(6):
            ph = ((f * 0.5 + i * 2.2 + k) % 12) / 12
            px = x + d * int(ph * r)
            py = y - h + int((ph * 2 - 1) ** 2 * h * 0.9) - int(h * 0.1)
            c.px(px, py, 'white' if i % 2 else 'crystal_lt')


def _cam_b(f):
    return 48 * cruise(seg(f, CUT_B - 4, CUT_C + 8))


def _shot_b(f):
    cam = _cam_b(f)
    c = Canvas()
    _sky(c, 0, 1.0, morn=0.35, hy=118)
    fx.glow(c, 340, 70, 150, 'gold', 0.20)
    _wisps(c, f, -14, 1.0, drift=0.09, morn=0.3)
    _draw_layer(c, _far(), int(round(-cam * 0.25)), 66, 'sun')
    R = _rail()
    rox = int(round(-cam * 0.7)) + 8
    _draw_layer(c, R, rox, 0, 'sun')
    _draw_train(c, f, rox)
    _steam_puffs(c, f, rox)
    for k in range(6):                         # mist drifting in the valley below the spur
        x = int((k * 64 + f * 0.25) % 380) - 40
        c.ellipse(x, 152 + (k % 3) * 6, 18, 3, 'cloud_dk' if k % 2 else 'cloud')
        c.ellipse(x + 6, 151 + (k % 3) * 6, 9, 2, 'cloud')
    CB = _castle_b()
    cbx = int(round(-cam * 0.6))
    _draw_layer(c, CB, cbx, -2, 'sun')
    for i, (px_, py) in enumerate(CB['flags']):
        _flag(c, px_ + cbx, py - 2, f, i, 'gold' if i % 2 else 'magic')
    for (bx, by, kind, i, bl, bw) in CB['banners']:
        _banner(c, bx + cbx, by - 3, f, kind, i, False, 1.0, bl, bw)
    for (rx, ry) in CB['roses']:
        fx.sparkle(c, rx + cbx + 2, ry - 4, f, 'white', period=30, size=2)
    G = _garden()
    gx = int(round(-cam))
    _draw_layer(c, G, gx, 0, 'sun')
    for k, (fx0, fy, r, j1, j2) in enumerate(G['jets']):
        _jet(c, f, j1[0] + gx, j1[1], 6 if r > 10 else 4, r * 0.6, k * 3)
        for i in range(3):                     # ripples in the basin
            ph = (f + i * 7 + k * 5) % 21
            x0 = fx0 + gx - r // 2 + ph // 3
            c.hline(x0 - 2, x0, fy - 3, 'crystal_lt')
        fx.sparkle(c, fx0 + gx + 3, fy - 4, f + k * 7, 'white', period=26, size=1)
    for k in range(14):                        # petals and motes drifting in the low sun
        x = int((hash01(k, 41) * 360 + f * (0.5 + hash01(k, 42) * 0.4)) % 360) - 20
        y = int((hash01(k, 43) * 120 + math.sin(f / 9 + k) * 4 + f * 0.15) % 140) + 30
        c.px(x, y, ('cotton', 'butter', 'white')[k % 3])
    _draw_layer(c, _garden_fg(), int(round(-cam * 1.35)) - 10, 142, 'sun')
    return c


# =====================================================================================
# SHOT C: the airship docks, the balloon casts off, pull back to the city (bar 5)
# =====================================================================================
CAST_OFF = 360
MAST_X, DECK_Y, PIER_X = 30, 148, 252


@lru_cache(maxsize=None)
def _dock():
    """Foreground: cliff ledge with the airship dock, mooring mast and the docked zeppelin."""
    ix = Ix(W, 180)
    yy, xx = np.mgrid[0:180, 0:W]
    props = {}
    # rock ledge
    ix.z = 1
    ix.poly([(-2, 132), (14, 128), (30, 134), (60, 138), (90, 144), (120, 150), (150, 152),
             (170, 158), (178, 170), (174, 180), (-2, 180)], 'ROCK')
    rock_tex(ix, ix.mat == MID['ROCK'], seed=51, dark=0.6 + np.clip((150 - xx) / 150, 0, 0.6), strata=8,
             green=0.5)
    for i, (tx, ty) in enumerate([(6, 130), (18, 129)]):
        tree(ix, tx, ty, 5 - i, 70 + i)
    # dock deck with struts and railing
    ix.z = 3
    ix.rect(12, DECK_Y, 264, 2, 'WD_LT')
    ix.rect(12, DECK_Y + 2, 264, 3, 'WD')
    ix.hline(12, 275, DECK_Y + 5, 'WD_DK')
    for x in range(14, 276, 6):
        ix.vline(x, DECK_Y + 2, DECK_Y + 4, 'WD_DK')
    for x0 in (180, 206, 232, 258):
        ix.line(x0, DECK_Y + 6, x0 - 20, DECK_Y + 34, 'WD_DK')
        ix.line(x0 + 1, DECK_Y + 6, x0 - 19, DECK_Y + 34, 'WD')
        ix.vline(x0 + 8, DECK_Y + 6, DECK_Y + 34, 'WD_DK')
    for x in range(44, 276, 8):
        ix.vline(x, DECK_Y - 6, DECK_Y - 1, 'WD')
    ix.hline(44, 275, DECK_Y - 6, 'WD_LT')
    ix.hline(44, 275, DECK_Y - 3, 'WD_DK')
    # mooring mast: tapered lattice with a gold ring on top
    ix.z = 4
    for y in range(56, DECK_Y):
        t = (y - 56) / (DECK_Y - 56)
        t = t * 0.8
        hw = int(round(2 + 6 * t))
        ix.px(MAST_X - hw, y, 'IRON')
        ix.px(MAST_X + hw, y, 'WD_LT' if hw > 2 else 'IRON')
        if (y // 6) % 2 == 0:
            xx0 = MAST_X - hw + ((y % 6) * hw * 2) // 6
            ix.px(xx0, y, 'IRON')
        else:
            xx0 = MAST_X + hw - ((y % 6) * hw * 2) // 6
            ix.px(xx0, y, 'IRON')
        if y % 12 == 0:
            ix.hline(MAST_X - hw, MAST_X + hw, y, 'IRON')
    ix.rect(MAST_X - 3, 52, 7, 4, 'T')
    ix.hline(MAST_X - 3, MAST_X + 3, 52, 'T_HI')
    ix.px(MAST_X + 3, 54, 'T_HI')
    ix.vline(MAST_X, 46, 51, 'T_SH')
    props['mast_top'] = (MAST_X, 46)
    props['beacon'] = (MAST_X, 45)
    # the zeppelin (nose at the mast)
    ix.z = 5
    zx0, zx1, zcy, zr = 36, 196, 80, 18
    for x in range(zx0, zx1):
        u = (x - zx0) / (zx1 - zx0)
        hh = zr * math.sqrt(max(0.0, 1 - ((u - 0.45) / 0.55) ** 2)) * (1 if u < 0.45 else (1 - 0.25 * (u - 0.45)))
        hh = max(2, int(round(hh)))
        for y in range(zcy - hh, zcy + hh + 1):
            v = (y - (zcy - hh)) / max(1, 2 * hh)
            m = 'W_HI' if v < 0.22 else ('W_LT' if v < 0.5 else ('W' if v < 0.78 else 'W_SH'))
            ix.px(x, y, m)
        ix.px(x, zcy - hh, 'W_OUT' if u > 0.85 else 'W_HI')
        ix.px(x, zcy + hh, 'W_OUT')
        if x % 12 == 0 and 0.05 < u < 0.92:      # envelope gores
            for y in range(zcy - hh + 2, zcy + hh - 1):
                if ix.mat[y, x] in (MID['W'], MID['W_LT']):
                    ix.px(x, y, 'W_SH' if y > zcy else 'W')
        if 0.04 < u < 0.97:                        # royal stripe with gold piping
            ix.px(x, zcy + 1, 'R')
            ix.px(x, zcy + 2, 'R_SH')
            ix.px(x, zcy, 'T')
            ix.px(x, zcy + 3, 'T_SH')
    ix.vline(zx0, zcy - 2, zcy + 2, 'T')
    # tail fins
    fb = zx1 - 24
    ix.poly([(fb, zcy - 11), (fb + 24, zcy - 25), (fb + 29, zcy - 25), (fb + 22, zcy - 7)], 'R')
    ix.poly([(fb, zcy + 11), (fb + 24, zcy + 25), (fb + 29, zcy + 25), (fb + 22, zcy + 8)], 'R_SH')
    ix.line(fb, zcy - 11, fb + 24, zcy - 25, 'T')
    ix.line(fb, zcy + 11, fb + 24, zcy + 25, 'T')
    ix.poly([(fb + 10, zcy - 2), (fb + 31, zcy - 3), (fb + 31, zcy + 4), (fb + 10, zcy + 3)], 'R_LT')
    ix.vline(fb + 31, zcy - 3, zcy + 4, 'T')
    # gold sun emblem
    ex, ey = zx0 + 60, zcy - 8
    ix.ellipse(ex, ey, 3, 3, 'T')
    ix.ellipse(ex, ey, 1.5, 1.5, 'T_HI')
    for (dx, dy) in ((0, -5), (0, 5), (-5, 0), (5, 0), (-4, -4), (4, -4), (-4, 4), (4, 4)):
        ix.px(ex + dx, ey + dy, 'T')
    # gondola
    ix.z = 6
    gx0, gx1, gy = zx0 + 52, zx0 + 100, zcy + 21
    ix.line(gx0 + 4, zcy + 12, gx0 + 2, gy, 'IRON')
    ix.line(gx1 - 4, zcy + 12, gx1 - 2, gy, 'IRON')
    ix.rect(gx0, gy, gx1 - gx0, 8, 'W')
    ix.hline(gx0, gx1 - 1, gy, 'T')
    ix.hline(gx0, gx1 - 1, gy + 7, 'W_SH')
    ix.vline(gx1 - 1, gy, gy + 7, 'W_HI')
    ix.vline(gx0, gy, gy + 7, 'W_OUT')
    for wx in range(gx0 + 3, gx1 - 4, 5):
        ix.rect(wx, gy + 2, 3, 3, 'WIN')
        ix.px(wx + 2, gy + 2, 'WIN_HI')
    ix.poly([(gx0 - 4, gy + 8), (gx1 + 3, gy + 8), (gx1 - 2, gy + 11), (gx0, gy + 11)], 'WD')
    ix.hline(gx0 - 3, gx1 + 2, gy + 8, 'WD_LT')
    props['prop'] = (gx1 + 4, gy + 3)
    # mooring lines from hull to deck
    ix.z = 7
    for (ax, bx) in ((zx0 + 26, zx0 + 34), (zx0 + 112, zx0 + 122), (zx0 + 140, zx0 + 152)):
        hh = 14 if ax < zx0 + 130 else 9
        ix.line(ax, zcy + hh, bx, DECK_Y - 6, 'WD_LT')
    # cargo: crates, barrels, a ticket kiosk with an airship icon sign
    ix.z = 8
    for (cx_, w_, h_) in ((60, 8, 7), (68, 7, 6), (63, 6, 5)):
        y0 = DECK_Y - h_ if cx_ != 63 else DECK_Y - 12
        ix.rect(cx_, y0, w_, h_, 'WD')
        ix.frame(cx_, y0, w_, h_, 'WD_DK')
        ix.hline(cx_, cx_ + w_ - 1, y0, 'WD_LT')
        ix.line(cx_, y0, cx_ + w_ - 1, y0 + h_ - 1, 'WD_DK')
    for bx_ in (150, 156):
        ix.rect(bx_, DECK_Y - 6, 5, 6, 'WD_LT')
        ix.hline(bx_, bx_ + 4, DECK_Y - 5, 'IRON')
        ix.hline(bx_, bx_ + 4, DECK_Y - 2, 'IRON')
        ix.vline(bx_ + 4, DECK_Y - 6, DECK_Y - 1, 'WD')
    kx = 204
    ix.rect(kx, DECK_Y - 14, 14, 14, 'W')
    ix.vline(kx + 13, DECK_Y - 14, DECK_Y - 1, 'W_HI')
    ix.vline(kx, DECK_Y - 14, DECK_Y - 1, 'W_SH')
    ix.rect(kx + 3, DECK_Y - 10, 8, 5, 'WIN')
    ix.hline(kx + 2, kx + 11, DECK_Y - 5, 'T')
    ix.poly([(kx - 2, DECK_Y - 13.5), (kx + 15.5, DECK_Y - 13.5), (kx + 7, DECK_Y - 22)], 'R')
    ix.poly([(kx + 7, DECK_Y - 22), (kx + 15.5, DECK_Y - 13.5), (kx + 10, DECK_Y - 13.5)], 'R_LT')
    ix.px(kx + 7, DECK_Y - 23, 'T')
    ix.vline(kx - 6, DECK_Y - 22, DECK_Y - 1, 'IRON')        # sign post with an airship icon
    ix.rect(kx - 13, DECK_Y - 22, 8, 6, 'WD_LT')
    ix.frame(kx - 13, DECK_Y - 22, 8, 6, 'WD_DK')
    ix.hline(kx - 11, kx - 8, DECK_Y - 20, 'R')
    ix.hline(kx - 12, kx - 7, DECK_Y - 19, 'R')
    ix.px(kx - 9, DECK_Y - 18, 'IRON')
    # lamp posts
    lamps = []
    for lx in (48, 176, 228, 274):
        ix.vline(lx, DECK_Y - 12, DECK_Y - 1, 'IRON')
        ix.hline(lx - 1, lx + 1, DECK_Y - 13, 'IRON')
        lamps.append((lx, DECK_Y - 12))
    # balloon pier: bollards
    ix.rect(PIER_X + 12, DECK_Y - 4, 3, 4, 'IRON')
    ix.rect(PIER_X - 15, DECK_Y - 4, 3, 4, 'IRON')
    ix.px(PIER_X + 13, DECK_Y - 5, 'T')
    ix.px(PIER_X - 14, DECK_Y - 5, 'T')
    L = _finish(ix, shadow_len=8, slope=2)
    L.update(props=props, lamps=lamps)
    return L


def _cam_c(f):
    return cruise(seg(f, 348, 440))


_BSIZES = (('l', 0), ('m', 384), ('s', 400), ('xs', 414))


def _balloon_c(f, dy):
    """Screen (x, y_bottom, size) of the explorer's balloon in shot C."""
    x0, yb0 = PIER_X, DECK_Y + dy
    if f < CAST_OFF:
        return x0, yb0, 'l'
    size = 'l'
    for nm, t in _BSIZES:
        if f >= t:
            size = nm
    t = seg(f, CAST_OFF, 424)
    # centre of the envelope travels on an eased arc away and up toward the morning sun
    _, hl = balloon_size('l')
    cy0 = (DECK_Y - hl / 2)
    cx = lerp(x0, 262, cruise(t, 0.3)) + math.sin(f / 13.0) * 1.2 * t
    cy = lerp(cy0, 44, ease_in(t) * 0.25 + cruise(t, 0.3) * 0.75)
    _, h = balloon_size(size)
    return int(round(cx)), int(round(cy + h / 2)), size


def _shot_c(f):
    t = _cam_c(f)
    oy_city = int(round(-30 + 30 * t))
    oy_far = 92 - int(round(12 * (1 - t)))
    dy = int(round(136 * t))
    c = Canvas()
    _sky(c, int(round(-10 + 10 * t)), 1.0, morn=1.0, hy=126)
    sx, sy = 300, 86
    fx.glow(c, sx, sy, 70, 'butter', 0.22)
    fx.glow(c, sx, sy, 20, 'white', 0.35)
    _SUNB.draw(c, sx, sy + 5, anchor='bottom')
    _wisps(c, f, -6, 1.0, drift=0.08, morn=1.0)
    _draw_layer(c, _far(), 0, oy_far, 'morn')
    L = _city()
    _draw_layer(c, L, 0, oy_city, 'morn')
    _falls(c, f, 0, oy_city, 'morn')
    for i, (fx_, ft, fw, fb) in enumerate(FALLS):
        _mist(c, f, fx_ + fw // 2, fb + oy_city, 12 + fw * 2, seed=i, morn=True)
    for i, (px_, py) in enumerate(L['flags']):
        _flag(c, px_, py + oy_city, f, i, 'gold' if i % 2 else 'magic')
    for (bx, by, kind, i, bl, bw) in L['banners']:
        _banner(c, bx, by + oy_city - 1, f, kind, i, False, 1.0, bl, bw)
    # the city glitters: sparkles hop across the gold tips, more as we pull back
    n = 3 + int(6 * t)
    fl = L['flags']
    for k in range(n):
        j = int(hash01(k, f // 10, 5) * len(fl))
        px_, py = fl[j]
        fx.sparkle(c, px_, py + oy_city + 3, f + k * 4, ('white', 'butter')[k % 2], period=20, size=2)
    for (rx, ry, r) in L['roses']:
        fx.sparkle(c, rx + 2, ry + oy_city - 2, f, 'white', period=24, size=3)
    # morning haze between the city and the near dock (eases as the dock leaves)
    c.blend(0, 0, W, H, '#d4ecff', 0.20 * (1 - 0.6 * t))
    # foreground dock (drops away as the camera rises with the balloon)
    D = _dock()
    if dy < H:
        _draw_layer(c, D, 0, dy, 'morn')
        bx_, by_ = D['props']['beacon']
        if (f // 9) % 2 == 0:
            c.px(bx_, by_ + dy, 'hot_pink')
            fx.glow(c, bx_, by_ + dy, 5, 'magic', 0.35)
        px_, py = D['props']['prop']
        ph = (f // 2) % 3
        c.vline(px_, py + dy - (2 if ph == 0 else 1), py + dy + (2 if ph == 2 else 1), 'steel_hi')
        c.px(px_ - 1, py + dy, 'brass')
        for i, (lx, ly) in enumerate(D['lamps']):
            c.px(lx, ly + dy, 'butter')
            c.px(lx, ly + dy - 1, 'brass')
        for k in range(3):                     # pennants on the kiosk and mast
            pass
        _flag(c, MAST_X, 41 + dy, f, 1, 'magic')
        _flag(c, 211, DECK_Y - 28 + dy, f, 2, 'gold')
    # the balloon
    bx, byb, size = _balloon_c(f, dy)
    if f < CAST_OFF:
        # mooring ropes to the bollards
        c.line(bx - 3, byb - 3, PIER_X - 14, DECK_Y - 5 + dy, 'tan')
        c.line(bx + 3, byb - 3, PIER_X + 13, DECK_Y - 5 + dy, 'tan')
    elif f < CAST_OFF + 10:
        k = f - CAST_OFF
        c.line(PIER_X + 13, DECK_Y - 5 + dy, PIER_X + 15 + k, DECK_Y - 9 + dy + k // 2, 'tan')
        c.line(PIER_X - 14, DECK_Y - 5 + dy, PIER_X - 16 - k, DECK_Y - 9 + dy + k // 2, 'tan')
        for i in range(6):
            fx.sparkle(c, PIER_X - 10 + i * 4, DECK_Y - 3 + dy - k, f + i * 3, 'white', period=10, size=1)
    draw_balloon(c, bx, byb, f, size, flame=CAST_OFF <= f < 396, bob=f < CAST_OFF)
    return c


# =====================================================================================
def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT_B:
        c = _shot_a(f)
        oy = int(round(-70 + 70 * _tilt(f)))
        callout(c, 186, 91 + oy, 'CANTERLOT CASTLE', f, 90, 164, side='right', length=36)
        callout(c, 238, 60 + oy, 'CANTERLOT TOWER', f, 126, 206, side='right', length=18)
        atlas_card(c, f, 100, 206, 'ROYAL CAPITAL', ['ON MT. CANTERHORN', 'CASTLE - GARDENS - FALLS'])
        if f < 80:
            iris(c, 1 - seg(f, 72, 79), 186, 24)
        if SUNRISE <= f < SUNRISE + 10:         # crash: an additive burst of light from the sun
            k = 1 - seg(f, SUNRISE, SUNRISE + 10)
            fx.glow(c, 300, _ridge_y(300) + 92, 210, '#ffd8a0', 0.34 * k, rings=6)
        return c
    if f < CUT_C:
        c = _shot_b(f)
        cam = _cam_b(f)
        callout(c, 125 - int(round(cam)), 140, 'ROYAL GARDENS', f, CUT_B + 2, CUT_B + 78, side='up', length=16)
        rox = int(round(-cam * 0.7)) + 8
        callout(c, PORTAL_X + rox, 61, 'MOUNTAIN RAILWAY', f, 240, 312, side='up', length=12)
        return c
    c = _shot_c(f)
    dy = int(round(136 * _cam_c(f)))
    mx, my = _dock()['props']['mast_top']
    callout(c, mx, my + dy, 'AIRSHIP DOCKS', f, CUT_C + 2, 400, side='up', length=10)
    return c


SFX = [
    (80, 'waterfall', 0.35),                  # the falls after the reveal
    (90, 'blip', 0.35),                       # CANTERLOT CASTLE tag
    (126, 'blip', 0.30),                      # CANTERLOT TOWER tag
    (144, 'bell', 0.55),                      # sunrise: the castle bell under the crash
    (146, 'shimmer', 0.45),                   # gold light sweeping the towers
    (165, 'sparkle', 0.40),                   # stained glass catches the light
    (216, 'fountain', 0.40),                  # dissolve to the royal gardens
    (252, 'whistle', 0.60),                   # train out of the tunnel on the beat
    (254, 'steam', 0.40),                     # steam burst from the tunnel mouth
    (261, 'chug', 0.45),
    (279, 'chug', 0.40),
    (297, 'chug', 0.32),
    (324, 'creak', 0.35),                     # glide to the airship docks
    (360, 'puff', 0.50),                      # balloon casts off on the bar line
    (360, 'whoosh', 0.40, {'align': 'hit'}),
    (380, 'wind', 0.30, {'variant': 0}),      # rising into the morning air
    (396, 'sparkle', 0.35),                   # the city glitters
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'dissolve', {}), (324, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
