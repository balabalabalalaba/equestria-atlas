"""Static world-map texture builder (owned by the map builder).

build(clearings) -> dict of numpy arrays:
    rgb    (MH, MW, 3) uint8  the painted map (terrain, water, rivers, rails, forests, mountains)
    wdist  (MH, MW)    uint8  distance of sea/lake pixels to land (0 on land), for animated foam
    cls    (MH, MW)    uint8  biome class per pixel (CLASSES index; 0 = water)
    peaks  (N, 5)      int32  mountain placements (x, y_base, w, h, scheme index) for the overview
    glint  (M, 3)      int32  sparse water glint seeds (x, y, phase)
The result is cached on disk in out/cache/worldmap_<hash>.npz (hash of this module, the data, the
sprite generators and the clearings), written atomically so parallel workers can share it.
"""
import hashlib
import math
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import worldmap_data as D
from . import worldmap_art as A
from .core import col

MW, MH = 1536, 1608
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, 'out', 'cache')

CLASSES = ['water', 'meadow', 'meadow_lt', 'tundra', 'snow', 'taiga', 'crag', 'farm', 'arid', 'desert',
           'barren', 'mauve', 'badlands', 'jungle', 'swamp', 'dunes', 'cloudsea', 'swamp_dark', 'heath',
           'basalt', 'olive', 'volcanic', 'storm', 'ridge', 'sandstone', 'grey', 'caves', 'forest_floor']
CI = {n: i for i, n in enumerate(CLASSES)}

# 4-step ramps dark -> light (+ coast cliff colour)
RAMPS = {
    'water': ('#2c5aa0', '#2c5aa0', '#2c5aa0', '#2c5aa0', '#2c5aa0'),
    'meadow': ('#357f45', '#4c9e4c', '#6dbb55', '#97d266', '#7a5a3e'),
    'meadow_lt': ('#5b9a3c', '#7bb848', '#9dd056', '#c3e374', '#8a6a3e'),
    'tundra': ('#7e9c88', '#a2bca6', '#c9d9cb', '#e8f0ea', '#7d8ea8'),
    'snow': ('#a3b4da', '#c2d1ee', '#e0eafa', '#fbfdff', '#8496c4'),
    'taiga': ('#6f8f7c', '#8eac96', '#b4cbb8', '#dce8de', '#6d7f96'),
    'crag': ('#5e625c', '#7a7d74', '#989a8e', '#b8b8aa', '#4a4c46'),
    'farm': ('#4c9e4c', '#6dbb55', '#8bc95e', '#a8d872', '#7a5a3e'),
    'arid': ('#b67e4a', '#cb985c', '#ddb070', '#ecc88a', '#9a643a'),
    'desert': ('#d39a3e', '#e6b252', '#f3ca68', '#fce08c', '#b57a34'),
    'barren': ('#716b6a', '#888180', '#a19a96', '#bab2ac', '#5e5858'),
    'mauve': ('#865a72', '#a2708a', '#be90a2', '#d8b0bc', '#6a4458'),
    'badlands': ('#a8522e', '#c86a3a', '#dc874a', '#eba662', '#843e26'),
    'jungle': ('#2a6a3a', '#3b8842', '#57a44c', '#7cbd5a', '#6a4a30'),
    'swamp': ('#40593e', '#557346', '#6d8a54', '#8ba266', '#4a3e2e'),
    'dunes': ('#d6b47c', '#e6c993', '#f2dcae', '#faecca', '#c09a64'),
    'cloudsea': ('#6c6359', '#8a7f72', '#a99e8d', '#cbc0ab', '#5a524a'),
    'swamp_dark': ('#2e3c35', '#415248', '#5a6c56', '#778868', '#2a2a26'),
    'heath': ('#7a9858', '#95ae6a', '#b2c682', '#cddb9c', '#6a5a3e'),
    'basalt': ('#3a3842', '#524f5c', '#6d6a78', '#8b8896', '#2a2830'),
    'olive': ('#4d5838', '#636e48', '#7e885a', '#9ca474', '#4a4030'),
    'volcanic': ('#2a2426', '#3c3236', '#524449', '#6a595b', '#1e1a1c'),
    'storm': ('#5a3a34', '#784c42', '#93634e', '#ab7c60', '#3e2822'),
    'ridge': ('#9a9286', '#b2aa9c', '#cac2b2', '#e0dac8', '#7e766a'),
    'sandstone': ('#ad875c', '#c6a072', '#d9ba8b', '#eacfa4', '#8a6a46'),
    'grey': ('#6a706f', '#858d8a', '#a1a8a3', '#c0c6bf', '#545a58'),
    'caves': ('#d0ab72', '#e0c08a', '#eed4a4', '#f8e6c2', '#b08a58'),
    'forest_floor': ('#2c6a3c', '#3a8042', '#4f964a', '#68ac56', '#5a4a30'),
}
_RAMP_ARR = np.array([[col(c) for c in RAMPS[n][:4]] for n in CLASSES], np.uint8)
_CLIFF_ARR = np.array([col(RAMPS[n][4]) for n in CLASSES], np.uint8)

WATER = {  # by distance band
    'foam': '#eafcff', 'coast': '#9adcea', 'shallow2': '#6fbfe0', 'shallow': '#549fd4',
    'mid': '#3f7fc2', 'deep': '#2f62aa', 'deeper': '#2a539a',
}
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16


def bayer_full(h, w, ox=0, oy=0):
    yy = (np.arange(h) + oy) % 4
    xx = (np.arange(w) + ox) % 4
    return BAYER[yy[:, None], xx[None, :]]


# ------------------------------------------------------------------ geometry helpers
def P(pt):
    return pt[0] * MW, pt[1] * MH


def catmull(pts, closed=True, n=8):
    pts = [(float(p[0]), float(p[1])) for p in pts]
    out = []
    m = len(pts)
    rng_ = range(m) if closed else range(m - 1)
    for i in rng_:
        p0 = pts[(i - 1) % m] if closed or i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[(i + 1) % m]
        p3 = pts[(i + 2) % m] if closed or i + 2 < m else pts[-1]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    if not closed:
        out.append(pts[-1])
    return out


def polyline_px(pts_n, step=1.0):
    """Smooth open polyline (normalised pts) -> dense (N, 2) float array of px points ~step apart."""
    pp = [P(p) for p in pts_n]
    sm = catmull(pp, closed=False, n=10) if len(pp) > 2 else pp
    sm = np.array(sm, np.float64)
    seglen = np.hypot(*np.diff(sm, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seglen)])
    L = s[-1]
    ts = np.arange(0, L + 1e-6, step)
    x = np.interp(ts, s, sm[:, 0])
    y = np.interp(ts, s, sm[:, 1])
    return np.stack([x, y], 1), ts


def raster(pts_px, shape=(MH, MW)):
    im = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts_px], fill=1)
    return np.asarray(im, np.uint8).astype(bool)


def fbm(shape, seed, scales, weights=None):
    h, w = shape
    rg = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    tot = 0.0
    for i, s in enumerate(scales):
        amp = weights[i] if weights else 1.0 / (len(scales) - i) ** 0.0 * (s / scales[0]) ** 0.6
        g = rg.standard_normal((h // s + 3, w // s + 3)).astype(np.float32)
        up = ndimage.zoom(g, s, order=1, prefilter=False)[:h, :w]
        out += amp * up
        tot += amp
    out /= tot
    return out / (out.std() + 1e-6)


def noisy_mask(pts_n, amp, noise, closed=True, smooth=True):
    pts = [P(p) for p in pts_n]
    if smooth:
        pts = catmull(pts, closed=True, n=6)
    m = raster(pts)
    if amp <= 0:
        return m
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = int(amp * 3 + 6)
    x0, x1 = max(0, int(min(xs)) - pad), min(MW, int(max(xs)) + pad)
    y0, y1 = max(0, int(min(ys)) - pad), min(MH, int(max(ys)) + pad)
    sub = m[y0:y1, x0:x1]
    sdf = ndimage.distance_transform_edt(~sub) - ndimage.distance_transform_edt(sub)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = (sdf + noise[y0:y1, x0:x1] * amp) < 0
    return out


def disc_stamp(mask, x, y, r):
    x0, x1 = int(x - r - 1), int(x + r + 2)
    y0, y1 = int(y - r - 1), int(y + r + 2)
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    if X1 <= X0 or Y1 <= Y0:
        return
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    mask[Y0:Y1, X0:X1] |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r


def stamp(rgb, spr, x, y):
    """Blit RGBA sprite with top-left (x, y) into rgb (clipped)."""
    h, w = spr.shape[:2]
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(rgb.shape[1], x + w), min(rgb.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = spr[Y0 - y:Y1 - y, X0 - x:X1 - x]
    m = sub[:, :, 3] > 0
    rgb[Y0:Y1, X0:X1][m] = sub[:, :, :3][m]


# ------------------------------------------------------------------ build
def source_hash(extra=''):
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ('worldmap_data.py', 'worldmap_art.py', 'worldmap_build.py'):
        with open(os.path.join(here, fn), 'rb') as fh:
            h.update(fh.read())
    h.update(extra.encode())
    return h.hexdigest()[:12]


def load_or_build(clearings=()):
    key = source_hash(repr(sorted(clearings)))
    path = os.path.join(CACHE_DIR, f'worldmap_{key}.npz')
    if os.path.exists(path):
        try:
            z = np.load(path)
            return {k: z[k] for k in z.files}
        except Exception:
            pass
    res = build(clearings)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, **res)
    os.replace(tmp, path)
    for fn in os.listdir(CACHE_DIR):          # drop stale versions
        if fn.startswith('worldmap_') and fn.endswith('.npz') and fn != os.path.basename(path) \
                and '.tmp' not in fn:
            try:
                os.remove(os.path.join(CACHE_DIR, fn))
            except OSError:
                pass
    return res


def build(clearings=()):
    rg = np.random.default_rng(1234)
    shp = (MH, MW)
    n_coast = fbm(shp, 11, (64, 32, 16, 8, 4), (1.0, 0.7, 0.5, 0.3, 0.15))
    n_bio = fbm(shp, 12, (48, 24, 12, 6), (1.0, 0.6, 0.35, 0.2))

    # ---------------- land
    land = noisy_mask(D.MAIN, 7, n_coast)
    land |= noisy_mask(D.DRAGON_ISLE, 7, n_coast)
    for k, poly in D.ISLANDS.items():
        land |= noisy_mask(poly, 3 if len(poly) > 5 else 1.5, n_coast)
    cls = np.where(land, CI['meadow'], CI['water']).astype(np.uint8)
    for name, poly, amp in D.BIOMES:
        m = noisy_mask(poly, amp, n_bio) & land
        cls[m] = CI[name]
    # lakes
    lake = np.zeros(shp, bool)
    for k, poly in D.LAKES.items():
        lake |= noisy_mask(poly, 2, n_coast)
    land &= ~lake
    cls[lake] = CI['water']

    # forests mask (for the floor colour) and tree placements
    forest_any = np.zeros(shp, bool)
    forest_masks = []
    for kind, poly, sp, keep in D.FORESTS:
        m = noisy_mask(poly, 6, n_bio) & land
        forest_masks.append((kind, m, sp, keep))
        if kind in ('dark', 'round', 'mixed', 'olive', 'jungle', 'violet', 'autumn', 'pine_snow', 'bamboo',
                    'mushroom'):
            forest_any |= m

    # ---------------- distance fields
    dl = ndimage.distance_transform_edt(land).astype(np.float32)       # land px -> distance to water
    dw = ndimage.distance_transform_edt(~land).astype(np.float32)      # water px -> distance to land

    # ---------------- height & shading
    elev = np.clip(dl / 60.0, 0, 1) * 0.6 + fbm(shp, 13, (96, 48, 24, 12), (1.0, 0.6, 0.4, 0.2)) * 0.35
    rough = np.isin(cls, [CI[n] for n in ('crag', 'mauve', 'snow', 'barren', 'grey', 'olive', 'volcanic',
                                         'storm', 'tundra', 'taiga', 'ridge', 'sandstone')])
    elev += rough * fbm(shp, 14, (16, 8, 4), (1.0, 0.6, 0.3)) * 0.18
    gx = np.zeros_like(elev)
    gy = np.zeros_like(elev)
    gx[:, 1:-1] = elev[:, 2:] - elev[:, :-2]
    gy[1:-1, :] = elev[2:, :] - elev[:-2, :]
    shade = (gx + gy) * 14.0
    patch = fbm(shp, 15, (40, 20), (1.0, 0.5))
    fine = fbm(shp, 16, (3, 2), (1.0, 0.5))
    v = 1.65 + np.clip(shade, -1.4, 1.4) + patch * 0.45 + fine * 0.18
    # calm snowfields: compress the tonal range towards the light end
    snowy = np.isin(cls, [CI['snow'], CI['tundra'], CI['taiga']])
    v = np.where(snowy, 2.25 + (v - 1.65) * 0.42, v)
    # dunes: broad crescents (windward lit, steep shaded lee, crisp crest)
    dun = np.isin(cls, [CI['dunes'], CI['desert'], CI['caves']])
    ys_, xs_ = np.nonzero(dun)
    u = (xs_ * 0.18 + ys_ * 0.98) / 21.0 + n_bio[ys_, xs_] * 0.55 + np.sin(xs_ / 37.0) * 0.35
    fr = u - np.floor(u)
    dv = np.where(fr < 0.74, 1.9 + fr * 1.25, np.where(fr < 0.79, 3.4, np.where(fr < 0.83, 0.6, 1.2)))
    v[ys_, xs_] = dv + patch[ys_, xs_] * 0.18
    # forest floor darker
    cls_draw = cls.copy()
    cls_draw[forest_any & np.isin(cls, [CI['meadow'], CI['meadow_lt']])] = CI['forest_floor']
    v = np.where(forest_any, v - 0.4, v)
    thr = bayer_full(MH, MW)
    vi = np.floor(v)
    idx = (vi + ((v - vi) > (0.5 + (thr - 0.5) * 0.7))).astype(np.int32)
    idx = np.clip(idx, 0, 3)
    rgb = _RAMP_ARR[cls_draw, idx]

    # ---------------- texture details per biome
    _farmland(rgb, cls, rg)
    _tufts(rgb, cls_draw, dl, rg)
    _swamp(rgb, cls, rg)
    _special_ground(rgb, cls, rg, n_bio)

    # ---------------- coasts
    beach_n = n_bio * 0.8
    temperate = np.isin(cls, [CI[n] for n in ('meadow', 'meadow_lt', 'farm', 'jungle', 'heath', 'desert',
                                             'arid', 'dunes', 'forest_floor', 'caves', 'olive')])
    cold = np.isin(cls, [CI[n] for n in ('snow', 'tundra', 'taiga', 'crag', 'grey')])
    beach = land & (dl <= 2.6 + beach_n) & ~lake
    sand1, sand2 = np.array(col('#f6dfa4'), np.uint8), np.array(col('#fdf0c8'), np.uint8)
    bm = beach & temperate
    rgb[bm] = np.where(((dl[bm] <= 1.5))[:, None], sand2, sand1)
    rock1 = np.array(col('#c6d2e2'), np.uint8)
    rgb[beach & cold & (dl <= 1.5)] = rock1
    # cliff lip on south-facing shores (3/4 depth): land pixel with water 1-2 px below
    wat = ~land
    below1 = np.zeros_like(wat)
    below1[:-1] = wat[1:]
    below2 = np.zeros_like(wat)
    below2[:-2] = wat[2:]
    lip = land & (below1 | below2) & ~np.isin(cls, [CI['swamp'], CI['swamp_dark']])
    cliff = _CLIFF_ARR[cls]
    rgb[lip & below1] = cliff[lip & below1]
    rgb[lip & below2 & ~below1] = (cliff[lip & below2 & ~below1].astype(np.int16) * 13 // 10).clip(0, 255)

    # ---------------- water ramps
    dwn = dw + fbm(shp, 17, (32, 16, 8), (1.0, 0.5, 0.25)) * 2.2
    wcol = np.zeros((MH, MW, 3), np.uint8)
    bands = [(2.0, 'coast'), (5.5, 'shallow2'), (11, 'shallow'), (24, 'mid'), (52, 'deep'), (1e9, 'deeper')]
    prev = 0
    tw = bayer_full(MH, MW, 1, 2)
    wsel = np.full(shp, len(bands) - 1, np.int32)
    for i in range(len(bands) - 1, -1, -1):
        lim = bands[i][0]
        soft = max(1.0, lim * 0.18)
        wsel = np.where(dwn < lim - soft * (tw - 0.5) * 2, i, wsel)
    wpal = np.array([col(WATER[b[1]]) for b in bands], np.uint8)
    wcol = wpal[wsel]
    # lakes are lighter
    wcol[lake & (dw > 2)] = np.array(col('#5fb2dc'), np.uint8)
    wcol[lake & (dw > 5)] = np.array(col('#4f9fd2'), np.uint8)
    # icy seas north & south: lighter, more teal
    yy = np.arange(MH)[:, None]
    ice_t = np.clip(np.maximum((0.1 * MH - yy) / (0.06 * MH), (yy - 0.84 * MH) / (0.1 * MH)), 0, 1)
    ice_t = np.broadcast_to(ice_t, shp) + fbm(shp, 18, (32, 16), (1.0, 0.5)) * 0.12
    icy = (ice_t * 0.85 > bayer_full(MH, MW, 2, 1)) & (dw > 3)
    wcol[icy] = (wcol[icy].astype(np.int16) * 2 // 3 + np.array(col('#a6e0ee'), np.int16) // 3).astype(np.uint8)
    # shadow under shores (water just below land)
    above = np.zeros_like(land)
    above[1:] = land[:-1]
    above2 = np.zeros_like(land)
    above2[2:] = land[:-2]
    sh = ~land & (above | above2)
    wcol[sh] = (wcol[sh].astype(np.int16) * 4 // 5).astype(np.uint8)
    # static wave marks (little '~' crests) in open water
    _wave_marks(wcol, ~land & ~lake & (dw > 9), rg)
    rgb[~land] = wcol[~land]

    # ---------------- rivers, roads, rails
    river = _rivers(rgb, land)
    gorge = _gorges(rgb, land, river)
    occupied = river.copy() | gorge
    roadm = _roads(rgb, land)
    railm = _rails(rgb, land | river, river | ~land)
    occupied |= ndimage.binary_dilation(roadm | railm, iterations=2)

    # ---------------- sprites (y-sorted): trees, mountains, mesas, spires, ice, cloud puffs
    clear = np.zeros(shp, bool)
    for (cx, cy, rx, ry) in clearings:
        x0, x1 = int(max(0, cx - rx)), int(min(MW, cx + rx))
        y0, y1 = int(max(0, cy - ry)), int(min(MH, cy + ry))
        yy2, xx2 = np.mgrid[y0:y1, x0:x1]
        clear[y0:y1, x0:x1] |= ((xx2 - cx) / max(1, rx)) ** 2 + ((yy2 - cy) / max(1, ry)) ** 2 <= 1
    stamps = []          # (y_base, order, x_left, y_top, sprite)
    peaks = []
    _place_trees(stamps, forest_masks, occupied | clear | lake | (dl < 2.5), rg, rgb)
    _place_mountains(stamps, peaks, rg)
    _place_misc(stamps, land, dw, rg)
    stamps.sort(key=lambda s: (s[0], s[1]))
    for _, _, x, y, spr in stamps:
        stamp(rgb, spr, x, y)
    _ice_slide(rgb)
    _neighagra_basin(rgb)

    # ---------------- outputs
    wd = np.where(land, 0, np.clip(dw, 0, 255)).astype(np.uint8)
    wd[lake] = np.clip(dw[lake], 0, 255).astype(np.uint8)
    cand = np.argwhere((~land) & (dw > 7) & (dw < 160))
    pick = rg.choice(len(cand), size=min(len(cand), 5200), replace=False)
    g = cand[pick]
    glint = np.stack([g[:, 1], g[:, 0], rg.integers(0, 997, len(g))], 1).astype(np.int32)
    return dict(rgb=rgb, wdist=wd, cls=cls, peaks=np.array(peaks, np.int32).reshape(-1, 5),
                glint=glint, river=river, forest=forest_any)


# ------------------------------------------------------------------ detail passes
def _farmland(rgb, cls, rg):
    farm = cls == CI['farm']
    pts = np.argwhere(farm)
    if len(pts) == 0:
        return
    from scipy.spatial import cKDTree
    ys, xs = pts[:, 0], pts[:, 1]
    sx = np.arange(xs.min(), xs.max() + 14, 13)
    sy = np.arange(ys.min(), ys.max() + 11, 10)
    gxx, gyy = np.meshgrid(sx, sy)
    seeds = np.stack([gxx.ravel() + rg.uniform(-5, 5, gxx.size), gyy.ravel() + rg.uniform(-4, 4, gyy.size)], 1)
    tree = cKDTree(seeds)
    _, cell = tree.query(np.stack([xs, ys], 1))
    crops = [('#e8c25a', '#d4a43e'), ('#9ad064', '#7ab84e'), ('#5aa64c', '#468e40'), ('#b98a58', '#9a6e44'),
             ('#c9d86a', '#aebe52'), ('#f2d06a', '#e0b44a'), ('#7cc46a', '#5ea852'), ('#e8c25a', '#d4a43e')]
    ctype = rg.integers(0, len(crops), len(seeds))
    orient = rg.integers(0, 3, len(seeds))
    c0 = np.array([col(c[0]) for c in crops], np.uint8)
    c1 = np.array([col(c[1]) for c in crops], np.uint8)
    o = orient[cell]
    stripe = np.where(o == 0, ys % 3 == 0, np.where(o == 1, xs % 3 == 0, (xs + ys) % 3 == 0))
    t = ctype[cell]
    out = np.where(stripe[:, None], c1[t], c0[t])
    # hedgerows: cell boundaries
    cmap = np.full(cls.shape, -1, np.int64)
    cmap[ys, xs] = cell
    b = np.zeros(cls.shape, bool)
    b[:, 1:] |= (cmap[:, 1:] != cmap[:, :-1]) & (cmap[:, :-1] >= 0)
    b[1:, :] |= (cmap[1:, :] != cmap[:-1, :]) & (cmap[:-1, :] >= 0)
    rgb[ys, xs] = out
    hb = b & farm
    rgb[hb] = np.array(col('#3f7f3e'), np.uint8)
    # small hedge bushes
    hy, hx = np.nonzero(hb)
    sel = rg.random(len(hy)) < 0.12
    rgb[hy[sel], hx[sel]] = np.array(col('#2d6a36'), np.uint8)


def _tufts(rgb, cls_draw, dl, rg):
    for name, dark, light, sp, keep in (('meadow', '#357f45', '#a8dc72', 6, 0.45),
                                        ('meadow_lt', '#5b9a3c', '#d4ec88', 6, 0.45),
                                        ('forest_floor', '#24583a', '#5a9e50', 5, 0.3),
                                        ('jungle', '#2a6a3a', '#86c860', 6, 0.4),
                                        ('heath', '#6a8a4c', '#dce6a8', 6, 0.4),
                                        ('tundra', '#8eaa98', '#ffffff', 7, 0.3),
                                        ('olive', '#3e4830', '#a4ac78', 6, 0.35),
                                        ('swamp', '#34482f', '#9aae72', 5, 0.4),
                                        ('arid', '#a06c40', '#f2d29a', 8, 0.3),
                                        ('taiga', '#647e70', '#f0f6f2', 7, 0.3)):
        m = cls_draw == CI[name]
        ys = np.arange(2, MH - 2, sp)
        xs = np.arange(2, MW - 2, sp)
        gx, gy = np.meshgrid(xs, ys)
        gx = (gx + rg.integers(-2, 3, gx.shape)).clip(1, MW - 2)
        gy = (gy + rg.integers(-2, 3, gy.shape)).clip(1, MH - 2)
        ok = m[gy, gx] & (rg.random(gx.shape) < keep) & (dl[gy, gx] > 4)
        x, y = gx[ok], gy[ok]
        d = np.array(col(dark), np.uint8)
        l = np.array(col(light), np.uint8)
        rgb[y, x - 1] = d
        rgb[y, x + 1] = d
        rgb[y - 1, x] = l
        rgb[y, x] = d
    # flowers on light meadows
    m = cls_draw == CI['meadow_lt']
    ys, xs = np.nonzero(m)
    sel = rg.random(len(ys)) < 0.006
    fc = np.array([col(c) for c in ('#ffb0d8', '#ffffff', '#fff27a', '#ff8fb0', '#c8a8ff')], np.uint8)
    rgb[ys[sel], xs[sel]] = fc[rg.integers(0, len(fc), sel.sum())]
    m = cls_draw == CI['meadow']
    ys, xs = np.nonzero(m)
    sel = rg.random(len(ys)) < 0.002
    rgb[ys[sel], xs[sel]] = fc[rg.integers(0, len(fc), sel.sum())]


def _dunes(rgb, cls, noise):
    for name, crest, trough, fx, fy in (('dunes', '#fff6dc', '#c9a46c', 0.035, 0.16),
                                        ('desert', '#fff0b0', '#c88e3a', 0.03, 0.14),
                                        ('caves', '#fff4dc', '#c09a64', 0.03, 0.15)):
        m = cls == CI[name]
        ys, xs = np.nonzero(m)
        ph = xs * fx + ys * fy + noise[ys, xs] * 1.8 + np.sin(xs * 0.02) * 1.2
        f = ph - np.floor(ph)
        rgb[ys[(f > 0.0) & (f < 0.09)], xs[(f > 0.0) & (f < 0.09)]] = np.array(col(crest), np.uint8)
        sel = (f > 0.09) & (f < 0.2)
        rgb[ys[sel], xs[sel]] = np.array(col(trough), np.uint8)


def _swamp(rgb, cls, rg):
    n = fbm(cls.shape, 21, (12, 6, 3), (1.0, 0.5, 0.25))
    for name, pool, rim in (('swamp', '#4d6e64', '#6f9488'), ('swamp_dark', '#3a4e4a', '#566e66')):
        m = cls == CI[name]
        pm = m & (n > 0.75)
        rgb[pm] = np.array(col(pool), np.uint8)
        e = pm & ~ndimage.binary_erosion(pm)
        rgb[e] = np.array(col(rim), np.uint8)
        # reeds
        ys, xs = np.nonzero(m & ~pm)
        sel = rg.random(len(ys)) < 0.03
        rgb[ys[sel], xs[sel]] = np.array(col('#2a3a26'), np.uint8)
        y2 = (ys[sel] - 1).clip(0)
        rgb[y2, xs[sel]] = np.array(col('#8aa060'), np.uint8)


def _special_ground(rgb, cls, rg, noise):
    # badlands cracks / strata
    m = cls == CI['badlands']
    ys, xs = np.nonzero(m)
    ph = ys * 0.25 + noise[ys, xs] * 2.0
    f = ph - np.floor(ph)
    sel = f < 0.12
    rgb[ys[sel], xs[sel]] = np.array(col('#b45a32'), np.uint8)
    # caves of conundrum: holes
    m = cls == CI['caves']
    pts = np.argwhere(m)
    if len(pts):
        sel = pts[rg.random(len(pts)) < 0.0035]
        for y, x in sel:
            r = rg.integers(1, 3)
            for dy in range(-1, 2):
                for dx in range(-r, r + 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < MH and 0 <= xx < MW:
                        rgb[yy, xx] = col('#4a3424') if dy <= 0 else col('#f8e6c2')
    # basalt hex columns
    m = cls == CI['basalt']
    ys, xs = np.nonzero(m)
    hx = (xs + (ys // 4 % 2) * 3) % 6
    hy = ys % 4
    edge = (hx == 0) | (hy == 0)
    edge = edge & ((xs * 3 + ys * 5) % 7 != 0)
    rgb[ys[edge], xs[edge]] = (rgb[ys[edge], xs[edge]].astype(np.int16) * 4 // 5).astype(np.uint8)
    lit = (hx == 1) & (hy == 1)
    rgb[ys[lit], xs[lit]] = np.array(col('#8b8896'), np.uint8)
    # volcanic lava cracks
    m = cls == CI['volcanic']
    n2 = fbm(cls.shape, 23, (10, 5), (1.0, 0.5))
    lv = m & (np.abs(n2) < 0.06)
    rgb[lv] = np.array(col('#c8401e'), np.uint8)
    # snow sparkle + sastrugi
    m = cls == CI['snow']
    ys, xs = np.nonzero(m)
    sel = rg.random(len(ys)) < 0.004
    rgb[ys[sel], xs[sel]] = np.array(col('#ffffff'), np.uint8)
    # barren rock farm: rows of rocks
    m = cls == CI['barren']
    ys, xs = np.nonzero(m)
    sel = (ys % 6 == 0) & (xs % 4 == 0)
    rgb[ys[sel], xs[sel]] = np.array(col('#5a5454'), np.uint8)
    sel2 = (ys % 6 == 5) & (xs % 4 == 0)
    rgb[ys[sel2], xs[sel2]] = np.array(col('#c8c0ba'), np.uint8)
    # cloud sea base: billowy noise bands
    m = cls == CI['cloudsea']
    ys, xs = np.nonzero(m)
    f = noise[ys, xs]
    rgb[ys[f > 0.6], xs[f > 0.6]] = np.array(col('#d8ccb6'), np.uint8)


def _wave_marks(wcol, m, rg):
    ys = np.arange(4, MH - 4, 9)
    xs = np.arange(4, MW - 6, 13)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + rg.integers(-5, 6, gx.shape)
    gy = gy + rg.integers(-3, 4, gy.shape)
    ok = m[gy.clip(0, MH - 1), gx.clip(0, MW - 4)] & (rg.random(gx.shape) < 0.35)
    x, y = gx[ok], gy[ok]
    base = wcol[y, x].astype(np.int16)
    hi = np.clip(base + 26, 0, 255).astype(np.uint8)
    lo = np.clip(base - 14, 0, 255).astype(np.uint8)
    wcol[y, x] = hi
    wcol[y, x + 1] = hi
    wcol[y - 1, x + 2] = hi
    wcol[y + 1, x + 2] = lo
    wcol[y + 1, x] = lo


def _rivers(rgb, land):
    river = np.zeros((MH, MW), bool)
    bank = np.zeros((MH, MW), bool)
    for pts, w0, w1 in D.RIVERS:
        line, ts = polyline_px(pts, 0.7)
        n = len(line)
        for i, (x, y) in enumerate(line):
            t = i / max(1, n - 1)
            r = (w0 + (w1 - w0) * t) / 2
            disc_stamp(bank, x + 0.6, y + 0.9, r + 1.0)
            disc_stamp(river, x, y, r)
    edge = bank & ~river
    land_edge = edge & land
    rgb[land_edge] = (rgb[land_edge].astype(np.int16) * 6 // 10).astype(np.uint8)
    rv = river & land
    rgb[rv] = np.array(col('#56aedc'), np.uint8)
    # lighter centre line + sparkle
    core = ndimage.binary_erosion(rv, iterations=1)
    rgb[core] = np.array(col('#6cc2e6'), np.uint8)
    yy, xx = np.nonzero(core)
    sel = ((xx * 7 + yy * 13) % 23) == 0
    rgb[yy[sel], xx[sel]] = np.array(col('#d8f6ff'), np.uint8)
    # left/top bank highlight
    hl = np.zeros_like(rv)
    hl[1:, 1:] = rv[:-1, :-1] & ~rv[1:, 1:]
    return river & land


def _gorges(rgb, land, river):
    allm = np.zeros((MH, MW), bool)
    for pts, w in D.GORGES:
        line, _ = polyline_px(pts, 0.7)
        rim = np.zeros((MH, MW), bool)
        core = np.zeros((MH, MW), bool)
        for x, y in line:
            disc_stamp(rim, x, y, w / 2 + 2.6)
            disc_stamp(core, x, y + 0.6, max(1.2, w / 2 - 1.5))
        rim &= land
        core &= land
        wall = rim & ~core & ~river
        # walls: upper (north) side lit, lower side in shadow
        up = np.zeros_like(core)
        up[:-3] = core[3:]
        lit = wall & ~up
        rgb[wall] = np.array(col('#8c7e6e'), np.uint8)
        rgb[lit & ~core] = np.array(col('#c4a882'), np.uint8)
        dry = core & ~river
        if dry.any():
            rgb[dry] = np.array(col('#3a2e2a'), np.uint8)
            e = dry & ~ndimage.binary_erosion(dry)
            rgb[e] = np.array(col('#5a4a40'), np.uint8)
        allm |= rim
    return allm


def _ice_slide(rgb):
    line, _ = polyline_px(D.ICE_SLIDE, 0.5)
    m = np.zeros((MH, MW), bool)
    e = np.zeros((MH, MW), bool)
    for x, y in line:
        disc_stamp(e, x + 0.5, y + 0.8, 2.6)
        disc_stamp(m, x, y, 1.6)
    rgb[e & ~m] = np.array(col('#7aa4d0'), np.uint8)
    rgb[m] = np.array(col('#eef8ff'), np.uint8)


def _roads(rgb, land):
    m = np.zeros((MH, MW), bool)
    for pts in D.ROADS:
        line, ts = polyline_px(pts, 1.0)
        for i, (x, y) in enumerate(line):
            xi, yi = int(x), int(y)
            if not (0 <= xi < MW - 1 and 0 <= yi < MH - 1) or not land[yi, xi]:
                continue
            if i % 4 < 2:
                rgb[yi, xi] = col('#ead6a2')
                rgb[yi + 1, xi] = col('#a4845a')
            m[yi, xi] = True
    return m


TIE = np.array(col('#6b4a36'), np.uint8)
RAIL = np.array(col('#3a3040'), np.uint8)


def _rails(rgb, land, water):
    m = np.zeros((MH, MW), bool)
    for pts in D.RAILS:
        tunnel_segs = [i for i, p in enumerate(pts) if len(p) > 2 and p[2] == 'T']
        clean = [(p[0], p[1]) for p in pts]
        line, ts = polyline_px(clean, 0.5)
        # tunnel ranges in arc length: map control points to nearest line index
        tun = np.zeros(len(line), bool)
        cps = np.array([P(p) for p in clean])
        cidx = [int(np.argmin(np.hypot(line[:, 0] - cx, line[:, 1] - cy))) for cx, cy in cps]
        for k in tunnel_segs:
            tun[cidx[k]:cidx[k + 1] + 1] = True
        d = np.gradient(line, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        nrm /= np.maximum(1e-6, np.hypot(nrm[:, 0], nrm[:, 1]))[:, None]
        on_water = np.array([water[int(y) % MH, int(x) % MW] for x, y in line])
        # bridge decks first
        for i, (x, y) in enumerate(line):
            if on_water[i] and not tun[i]:
                for s in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
                    xx, yy = int(round(x + nrm[i, 0] * s)), int(round(y + nrm[i, 1] * s))
                    if 0 <= xx < MW and 0 <= yy < MH:
                        rgb[yy, xx] = col('#c8b088') if abs(s) < 2 else col('#8a6a4a')
                if i % 12 == 0:
                    for k in (1, 2):
                        yy, xx = int(y) + 2 + k, int(x)
                        if 0 <= xx < MW and 0 <= yy < MH:
                            rgb[yy, xx] = col('#5a4a44')
        for i, (x, y) in enumerate(line):
            if tun[i]:
                continue
            if i % 6 == 0:
                for s in (-1.5, -0.5, 0.5, 1.5):
                    xx, yy = int(round(x + nrm[i, 0] * s)), int(round(y + nrm[i, 1] * s))
                    if 0 <= xx < MW and 0 <= yy < MH:
                        rgb[yy, xx] = TIE
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < MW and 0 <= yi < MH:
                rgb[yi, xi] = RAIL
                m[yi, xi] = True
        # tunnel portals
        for k in tunnel_segs:
            for j in (cidx[k], cidx[k + 1]):
                x, y = line[j]
                for dx in range(-2, 3):
                    for dy in range(-3, 1):
                        xx, yy = int(x) + dx, int(y) + dy
                        if abs(dx) == 2 or dy == -3:
                            c = '#9a948c'
                        else:
                            c = '#1e1a22'
                        if abs(dx) == 2 and dy == -3:
                            continue
                        if 0 <= xx < MW and 0 <= yy < MH:
                            rgb[yy, xx] = col(c)
    return m


def _place_trees(stamps, forest_masks, blocked, rg, rgb):
    mixes = {
        'round': ['round', 'round2'], 'autumn': ['autumn', 'autumn2', 'autumn3', 'autumn'],
        'mixed': ['round', 'pine', 'round2', 'dark'], 'dark': ['dark'], 'violet': ['violet', 'violet', 'dark'],
        'pine': ['pine'], 'pine_snow': ['pine_snow', 'pine_snow', 'pine'], 'olive': ['olive', 'olive', 'pine'],
        'palm': ['palm'], 'jungle': ['jungle', 'jungle', 'round2'], 'mushroom': ['mushroom'],
        'bamboo': ['bamboo'], 'dead': ['dead'], 'orchard': ['orchard'], 'cherry': ['cherry'],
    }
    shadow = np.zeros((MH, MW), bool)
    order = 0
    for kind, m, sp, keep in forest_masks:
        ys0, xs0 = np.nonzero(m)
        if len(ys0) == 0:
            continue
        y0, y1, x0, x1 = ys0.min(), ys0.max(), xs0.min(), xs0.max()
        rowsp = sp * (0.8 if kind not in ('orchard', 'cherry') else 1.0)
        gy = np.arange(y0, y1 + 1, rowsp)
        gx = np.arange(x0, x1 + 1, sp)
        X, Y = np.meshgrid(gx, gy)
        if kind in ('orchard', 'cherry'):
            X = X + (np.arange(len(gy))[:, None] % 2) * 0
        else:
            X = X + (np.arange(len(gy))[:, None] % 2) * sp / 2 + rg.uniform(-sp * 0.35, sp * 0.35, X.shape)
            Y = Y + rg.uniform(-sp * 0.3, sp * 0.3, Y.shape)
        X = X.astype(int).clip(0, MW - 1)
        Y = Y.astype(int).clip(0, MH - 1)
        ok = m[Y, X] & ~blocked[Y, X] & (rg.random(X.shape) < keep)
        mix = mixes[kind]
        for x, y in zip(X[ok], Y[ok]):
            k = mix[int(rg.integers(0, len(mix)))]
            spr = A.tree_sprite(k, int(rg.integers(0, 6)))
            h, w = spr.shape[:2]
            stamps.append((y, order, x - w // 2, y - h + 1, spr))
            # ground shadow to the bottom-right
            r = max(2, w // 2 - 1)
            disc_stamp(shadow, x + 2, y, r)
        order += 1
    # darken ground under shadows (subtle)
    rgb[shadow] = (rgb[shadow].astype(np.int16) * 3 // 4).astype(np.uint8)


SCHEME_IDX = {k: i for i, k in enumerate(A.MTN_SCHEMES)}


def _place_mountains(stamps, peaks, rg):
    for scheme, line_n, pw, ph, spacing, rows, jit, seed in D.RANGES:
        r2 = np.random.default_rng(seed)
        line, ts = polyline_px(line_n, 1.0)
        L = ts[-1] if len(ts) else 0
        for row in range(rows - 1, -1, -1):
            k = 1.0 + 0.18 * row
            s = r2.uniform(0, spacing * 0.5)
            while s <= L:
                i = min(len(line) - 1, int(s))
                x, y = line[i]
                x += r2.uniform(-1, 1) * jit * MW
                y += r2.uniform(-1, 1) * jit * MH * 0.5 - row * ph * 0.42
                w = int(pw * k * r2.uniform(0.72, 1.25))
                h = int(ph * k * r2.uniform(0.75, 1.2))
                sd = int(r2.integers(0, 1 << 20))
                spr = A.mountain_sprite(w, h, scheme, sd)
                stamps.append((int(y), 0, int(x) - spr.shape[1] // 2, int(y) - spr.shape[0] + 3, spr))
                peaks.append((int(x), int(y), w, h, SCHEME_IDX[scheme]))
                s += spacing * r2.uniform(0.7, 1.25) * k
    for scheme, xn, yn, w, h, seed in D.PEAKS:
        x, y = xn * MW, yn * MH
        spr = A.mountain_sprite(w, h, scheme, seed)
        stamps.append((int(y), 0, int(x) - spr.shape[1] // 2, int(y) - spr.shape[0] + 3, spr))
        peaks.append((int(x), int(y), w, h, SCHEME_IDX[scheme]))


def _place_misc(stamps, land, dw, rg):
    for i, (kind, xn, yn, w, h) in enumerate(D.SPIRES):
        spr = A.spire_sprite(kind, w, h, i)
        x, y = int(xn * MW), int(yn * MH)
        stamps.append((y, 1, x - spr.shape[1] // 2, y - spr.shape[0] + 2, spr))
    for i, (scheme, xn, yn, w, h, td) in enumerate(D.MESAS):
        spr = A.mesa_sprite(w, h, td, scheme, i)
        x, y = int(xn * MW), int(yn * MH)
        stamps.append((y, 1, x - spr.shape[1] // 2, y - spr.shape[0] + 2, spr))
    # Great Iceberg Barrier floes
    r3 = np.random.default_rng(77)
    for y in range(int(0.865 * MH), MH + 20, 13):
        t = (y / MH - 0.865) / 0.135
        for x in range(-10, MW + 10, 15):
            xx = x + int(r3.integers(-6, 7))
            yy = y + int(r3.integers(-5, 6))
            # density grows south; thinner in the middle south (open water around Mt Aris)
            edge = 0.9 + 0.1 * math.sin(xx / MW * 9.0 + 1.3)
            start = 0.885 + 0.05 * math.exp(-((xx / MW - 0.55) / 0.2) ** 2)
            if yy / MH < start * edge + 0.06 * (1 - edge):
                continue
            dens = min(1.0, (yy / MH - start) / 0.06)
            if r3.random() > dens * 0.85:
                continue
            if 0 <= yy < MH and 0 <= xx < MW and dw[yy, xx] < 6:
                continue
            r = int(4 + r3.integers(0, 6) + 8 * dens * r3.random())
            spr = A.floe_sprite(r, int(r3.integers(0, 4000)))
            stamps.append((yy, 2, xx - spr.shape[1] // 2, yy - spr.shape[0] // 2, spr))
    # arctic icebergs (NE sea)
    for k in range(70):
        xx = int(r3.uniform(0.5, 1.0) * MW)
        yy = int(r3.uniform(0.0, 0.07) * MH)
        if dw[yy, xx] < 5:
            continue
        w = int(r3.integers(5, 12))
        spr = A.berg_sprite(w, int(w * r3.uniform(0.8, 1.4)), k)
        stamps.append((yy, 2, xx - spr.shape[1] // 2, yy - spr.shape[0] + 2, spr))
    # Sea of Clouds puffs
    ramp = ('#f6efde', '#dcd0b8', '#b7aa92', '#8f8371', '#776c5e')
    cs_poly = [P(p) for p in D.BIOMES[[b[0] for b in D.BIOMES].index('cloudsea')][1]]
    cs = raster(catmull(cs_poly, True, 6))
    for y in range(int(0.705 * MH), int(0.8 * MH), 8):
        for x in range(int(0.26 * MW), int(0.73 * MW), 13):
            xx = x + int(r3.integers(-6, 7)) + (y // 8 % 2) * 6
            yy = y + int(r3.integers(-3, 4))
            if not cs[min(MH - 1, yy), min(MW - 1, xx)] or r3.random() < 0.1:
                continue
            spr = A.puff_sprite(int(r3.integers(7, 13)), ramp, int(r3.integers(0, 50)))
            stamps.append((yy, 0, xx - spr.shape[1] // 2, yy - spr.shape[0] + 2, spr))
    # Ponhenge stone circle (tiny standing stones at Foal Mountain's foot)
    cx, cy = 0.507 * MW, 0.318 * MH
    for k in range(9):
        a = k / 9 * 2 * math.pi
        x = int(cx + math.cos(a) * 7)
        y = int(cy + math.sin(a) * 4)
        s = np.zeros((4, 2, 4), np.uint8)
        s[:, 0, :3] = col('#d8d4cc')
        s[:, 1, :3] = col('#8a8680')
        s[:, :, 3] = 255
        stamps.append((y, 3, x, y - 3, s))


def _neighagra_basin(rgb):
    x, y = D.NEIGHAGRA[0] * MW, D.NEIGHAGRA[1] * MH
    m = np.zeros((MH, MW), bool)
    yy, xx = np.mgrid[int(y) - 4:int(y) + 14, int(x) - 16:int(x) + 17]
    e = ((xx - x) / 11.0) ** 2 + ((yy - (y + 6)) / 5.0) ** 2 <= 1
    m[yy[e], xx[e]] = True
    rgb[m] = np.array(col('#6cc2e6'), np.uint8)
    edge = m & ~ndimage.binary_erosion(m)
    rgb[edge] = np.array(col('#4f9fd2'), np.uint8)
