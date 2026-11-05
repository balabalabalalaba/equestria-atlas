"""HEARTLAND WONDERS: Neighagra Falls, Foal Mountain & Ponhenge, Hollow Shades
(5 bars of scene incl. the travel bar + 1 extra flight bar outside; HUD clock 10:30 -> 11:15, late-morning sun high in the upper left).

Beat sheet (SCENE frames; bar = 72, beat = 18; the 2-bar map flight is added outside by
engine/regionfx.long_flight, film = scene + 72 for scene >= 72). N = 360. Three postcards (108 / 90 /
90 frames = 3.6 / 3.0 / 3.0 s), changed by 24-frame cloud sweeps (engine/shotfx.smooth_cuts) centred on
the beats CUT2 = 180 and CUT3 = 270 (music hits cut2 / cut3). Every camera uses shotfx.cruise and keeps drifting through the sweeps.
  0-71     World-map travel shot (`travel_frame`). Iris closes 64-71.
  72-179   NEIGHAGRA FALLS. REVEAL 72: iris opens 72-79 on the heart of the horseshoe.
           Side/3-quarter view from downstream: the river above with rapids races into a broad
           horseshoe lip; dozens of animated falling-water columns with sliding foam bands drop into a
           thundering white mist that billows up; a rainbow arcs in the spray. Gorge walls with green
           ledges (left in shade, right sunlit), the churning lower river, a wooded plateau and hazy
           hills with a faint Canterhorn beyond. A sightseeing zeppelin cruises right->left, the
           explorer's balloon ('m') drifts over the left bank, an empty barrel hops toward the edge ON
           THE BEATS (90, 108, 126, 144), teeters on the lip, tips over ON BEAT 162 and tumbles into
           the mist (splash SFX 170). Foreground: the viewing platform (deck, railings, coin telescope, lamp, pennant).
           Camera: pan right 40 px over 72-190. Callout NEIGHAGRA FALLS 82-160. Card 100-208.
  180-269  FOAL MOUNTAIN & PONHENGE. A big blue-teal massif with a pale summit and a thin cascade;
           Rambling Rock Ridge's pale tilted slabs hazy beyond; meadows where the Friendship Express
           (no driver visible) chugs into a tunnel at the mountain foot. On a grassy knoll, the
           Ponhenge stone circle (trilithons, a fallen stone, an altar) with long shadows; carved
           runes glow and PULSE ON EVERY BEAT with a ring ripple. Wind combs the tall foreground grass,
           cloud shadows slide over the slopes. Camera: pan right 40 px over 172-278 (parallax
           0.15-1.5). Balloon 's' over the mountain. Callouts FOAL MOUNTAIN 184-256, PONHENGE 188-260.
  270-359  HOLLOW SHADES. A forest-buried village of dark blue-violet houses with steep crooked
           pointed roofs, abandoned and misty even at noon: cool shade, dim god-rays through a canopy
           gap (the balloon drifts in it), a shutter swinging, a creaking moon sign, cold lanterns, the
           stone Well of Shade breathing a faint cold glow and curls of mist, creeping fog bands,
           drifting leaves. Camera: tilt down 28 px + pan 18 px over 262-372 (still moving at the
           end). Callouts HOLLOW SHADES 279-351, WELL OF SHADE 282-354. Toast from 302 (top centre
           kept clear). Last 8 frames: diamond wipe.
Ambient: water columns, foam bands, rapids, mist, spray, rainbow shimmer, clouds, zeppelin + prop,
barrel, pennants, lower-river foam | grass wind, rune pulses, sparkles, cloud shadows, train + steam,
mountain cascade | fog bands, god-rays, motes, leaves, shutters, sign, lanterns, well glow, balloon.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'heartland'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
CUT2, CUT3 = 180, 270                     # postcard changes (cloud sweeps; music hits cut2/cut3)

KEY = (1, 2, 3)
B = FRAMES_PER_BEAT
LIGHT = np.array([-0.55, -0.70, 0.45])
LIGHT = LIGHT / np.linalg.norm(LIGHT)       # key light: high sun, upper left, slightly in front


# ================================================================== generic helpers
def C(c):
    return np.array(col(c), np.uint8)


def new(w, h=H):
    return Canvas(w, h, KEY)


def to_layer(cv):
    a = cv.a
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    return a, m


def comp(c, layer, ox, oy=0):
    """Composite a cached (rgb, mask) layer so that layer pixel (ox, oy) lands on screen (0, 0)."""
    a, m = layer
    ox, oy = int(ox), int(oy)
    lh, lw = a.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(lw, ox + W), min(lh, oy + H)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - ox, y0 - oy
    dst = c.a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    np.copyto(dst, a[y0:y1, x0:x1], where=m[y0:y1, x0:x1, None])


def vnoise(h, w, cell, seed):
    """Smooth value noise (h, w) in 0..1."""
    R = rng(seed)
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = R.random((gh, gw))
    yy = np.arange(h) / cell
    xx = np.arange(w) / cell
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    fy = yy - y0
    fx = xx - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c_ = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None]
    bot = c_ + (d - c_) * fx[None]
    return top + (bot - top) * fy[:, None]


def fbm(h, w, cell, seed, octaves=3):
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += vnoise(h, w, max(1.0, cell / (2 ** o)), seed + 31 * o) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def poly_mask(h, w, pts):
    """Boolean mask of a polygon (even-odd), vectorised over rows."""
    m = np.zeros((h, w), bool)
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    ys = np.arange(h) + 0.5
    xs = np.arange(w) + 0.5
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if y0 == y1:
            continue
        lo, hi = min(y0, y1), max(y0, y1)
        rows = np.nonzero((ys >= lo) & (ys < hi))[0]
        if len(rows) == 0:
            continue
        xi = x0 + (ys[rows] - y0) * (x1 - x0) / (y1 - y0)
        m[rows] ^= xs[None, :] < xi[:, None]
    return m


def ramp_idx(t, n, x0, y0, h, w, dith=0.7):
    """Quantise t (0..1 array) into n ramp levels with ordered dither."""
    thr = bayer(h, w, x0, y0)
    return np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)


def blobs(cv, bl, ramp, outline=None, light=LIGHT, dith=0.6, rim=None, bias=0.0, crease=True):
    """Shaded union of ellipsoid blobs [(cx, cy, rx, ry, z)] (front-most surface wins), quantised to
    `ramp` (dark -> light) with ordered dither, darker creases between clumps and an outline."""
    ramp_a = np.array([col(r) for r in ramp], np.uint8)
    n = len(ramp)
    x0 = max(0, int(math.floor(min(b[0] - b[2] for b in bl))) - 1)
    x1 = min(cv.w, int(math.ceil(max(b[0] + b[2] for b in bl))) + 2)
    y0 = max(0, int(math.floor(min(b[1] - b[3] for b in bl))) - 1)
    y1 = min(cv.h, int(math.ceil(max(b[1] + b[3] for b in bl))) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    xx += 0.5
    yy += 0.5
    best = np.full(xx.shape, -1e9, np.float32)
    NX = np.zeros(xx.shape, np.float32)
    NY = np.zeros(xx.shape, np.float32)
    NZ = np.zeros(xx.shape, np.float32)
    EDGE = np.zeros(xx.shape, bool)
    for (cx, cy, rx, ry, z) in bl:
        u = (xx - cx) / rx
        v = (yy - cy) / ry
        d2 = u * u + v * v
        ins = d2 <= 1.0
        zz = np.sqrt(np.clip(1 - d2, 0, 1))
        hgt = z + zz * min(rx, ry)
        upd = ins & (hgt > best)
        best[upd] = hgt[upd]
        NX[upd] = u[upd]
        NY[upd] = v[upd]
        NZ[upd] = zz[upd]
        r = max(1.0, min(rx, ry))
        EDGE[upd] = d2[upd] > ((r - 1.0) / r) ** 2
    m = best > -1e8
    lam = NX * light[0] + NY * light[1] + NZ * light[2]
    t = np.clip((lam + 1) / 2 + bias, 0, 0.999)
    lvl = ramp_idx(t, n, x0, y0, y1 - y0, x1 - x0, dith)
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    bound = m & ~inner
    if crease:
        away = (NX * light[0] + NY * light[1]) < 0.15
        lvl = np.where(EDGE & ~bound & away, np.maximum(lvl - 1, 0), lvl)
    out = ramp_a[lvl]
    if outline is not None:
        out[bound] = C(outline)
        if rim is not None:
            lit = bound & ((NX * light[0] + NY * light[1]) > 0.5)
            out[lit] = C(rim)
    reg = cv.a[y0:y1, x0:x1]
    reg[m] = out[m]
    return x0, y0, m


def canopy(cx, cy, r, seed, squash=0.86):
    """Blob list for a round, curly tree canopy."""
    R = rng(seed)
    bl = [(cx, cy, r * 0.82, r * 0.82 * squash, 0.0)]
    n = 5 + int(r / 4)
    a0 = R.random() * 6.283
    for i in range(n):
        a = a0 + 6.283 * i / n + (R.random() - 0.5) * 0.5
        rr = r * (0.36 + 0.14 * R.random())
        d = r - rr * 0.9
        px = cx + math.cos(a) * d
        py = cy + math.sin(a) * d * squash
        bl.append((px, py, rr, rr * squash, (py - cy) / r * rr * 0.9 + rr * 0.15))
    return bl


def tree(cv, x, base, r, seed, ramp, out, trunk=('brew', 'brew_mid'), trunk_h=None):
    th = trunk_h if trunk_h is not None else max(2, int(r * 0.7))
    tw = max(1, int(r / 4))
    cv.rect(x - tw // 2, base - th - int(r * 0.6), tw + 1, th + int(r * 0.6), trunk[0])
    if tw >= 2:
        cv.vline(x - tw // 2, base - th - int(r * 0.6), base - 1, trunk[1])
    blobs(cv, canopy(x, base - th - r * 0.75, r, seed), ramp, outline=out)


def disc(c, cx, cy, r, color, level=1.0, ox=0, oy=0):
    """Filled circle drawn with ordered-dither coverage `level` (fades puffs out)."""
    x0, y0 = int(cx - r - 1), int(cy - r - 1)
    x1, y1 = int(cx + r + 2), int(cy + r + 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(c.w, x1), min(c.h, y1)
    if X1 <= X0 or Y1 <= Y0 or level <= 0:
        return
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    m = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    if level < 1:
        m &= bayer(Y1 - Y0, X1 - X0, X0 + ox, Y0 + oy) < level
    c.a[Y0:Y1, X0:X1][m] = col(color)


def darken(c, mask, mul, y0=0, x0=0):
    """Multiply the pixels under a boolean mask (placed at x0, y0) by mul (r, g, b)."""
    h, w = mask.shape
    reg = c.a[y0:y0 + h, x0:x0 + w]
    sub = reg[mask].astype(np.float32) * np.array(mul, np.float32)
    reg[mask] = np.clip(sub, 0, 255).astype(np.uint8)


def flash(c, f, f0):
    """2-frame white flash on a hard cut."""
    if f == f0:
        fx.fade(c, 'white', 0.82)
    elif f == f0 + 1:
        fx.fade(c, 'white', 0.38)


def cam_ease(f, f0, f1, a, b):
    return int(round(lerp(a, b, ease_in_out(seg(f, f0, f1)))))


def cloud_sprite(w, seed, ramp, out=None):
    """A puffy shaded cloud as an RGBA array (w wide)."""
    h = int(w * 0.5) + 6
    cv = new(w + 4, h)
    R = rng(seed)
    bl = []
    n = max(3, w // 7)
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi)
        r = max(3.0, w * (0.10 + 0.12 * bump) * (0.85 + 0.3 * R.random()))
        bl.append((2 + u * w, h - 3 - r * 0.85, r, r * 0.85, bump * 2))
    bl.append((2 + w / 2, h - 3 - w * 0.06, w * 0.45, w * 0.08, 0))
    blobs(cv, bl, ramp, outline=out, light=np.array([-0.4, -0.8, 0.45]), dith=0.5, crease=False)
    a, m = to_layer(cv)
    rgba = np.zeros(a.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = a
    rgba[..., 3] = m * 255
    return rgba


# ================================================================== SHOT A: NEIGHAGRA FALLS
MWA = 372                     # main layer width
CXA = 186                     # horseshoe centre (main-layer x)
AHW = 86                      # horseshoe half width
HZA = 64                      # far forest line
RIV0 = 72                     # where the upper river appears out of the forest
LIP0, LIPK = 84, 18           # lip arch: y = LIP0 + LIPK * u^2
SKY_A = ['#3f86d6', '#4d9be6', '#62aded', '#83c8f6', '#ade0fb', '#d6f1f7']
FAR_A = ['#86b4d2', '#9cc5da', '#b3d6e0']
HAZE_A = '#c4e2ea'
LEAF_A = [mix('forest', 'green', 0.25), 'green', mix('green', 'leaf', 0.45), 'leaf',
          mix('leaf', 'pale_leaf', 0.55)]
LEAF_A_OUT = mix('forest', 'deep_teal', 0.35)
ROCK_LIT = ['#7a5f55', '#94786a', '#ad9078', '#c6aa8a', '#dfcaa6']
ROCK_SH = ['#3d3448', '#4b4156', '#5c4f62', '#6e5f6c', '#827078']
WATER = ['#12606e', '#1f8a8f', '#45b3b5', '#8fd9da', '#c8f1f2', '#ffffff']
MIST = ['#aebedb', '#cad6ec', '#e2eaf8', '#f3f7fd', '#ffffff']


def lipy(x):
    u = (x - CXA) / AHW
    return LIP0 + LIPK * u * u


def fallh(x):
    u = (x - CXA) / AHW
    return 50 + 8 * u * u


def bank_x(y, side):
    """River bank x at row y (side -1 left, +1 right): the river widens toward the lip."""
    s = clamp((y - RIV0) / (lipy(CXA + side * AHW) - RIV0))
    far = CXA - 22 if side < 0 else CXA + 26
    near = CXA + side * AHW
    return far + (near - far) * s ** 1.25


def rim_y(x):
    """Gorge rim (top of the side walls) beyond the horseshoe arms."""
    L = CXA - AHW
    R = CXA + AHW
    if x < L:
        return lipy(L) + (L - x) * 0.12
    return lipy(R) + (x - R) * 0.10


def water_edge(y, side):
    """x of the lower river's edge at row y (the foot of the side walls)."""
    x0 = CXA + side * (AHW - 2)
    y0 = LIP0 + LIPK + 50 + 8
    return x0 + side * max(0.0, y - y0) * 1.5


@lru_cache(maxsize=1)
def _sky_a():
    c = Canvas()
    fx.sky(c, SKY_A, 0, 92)
    fx.glow(c, 18, -26, 100, 'butter', 0.28, rings=6)
    fx.glow(c, 18, -26, 54, 'white', 0.20, rings=4)
    return c


@lru_cache(maxsize=None)
def _cloud(w, seed, tone=0):
    ramp = [mix('#a9b9dc', 'lavender', 0.10), '#c9d6ee', '#e6eefc', 'white', '#fffbe8']
    if tone:
        ramp = [mix(r, '#9cc6e6', 0.45) for r in ramp]
    return cloud_sprite(w, seed, ramp, out=mix('#a9b9dc', 'indigo', 0.12))


def draw_clouds(c, f, specs, ox=0):
    for (x0, y, w, seed, spd, tone) in specs:
        spr = _cloud(w, seed, tone)
        span = W + w + 40
        x = int((x0 - ox + f * spd) % span) - w - 20
        c.blit(spr, x, y)


@lru_cache(maxsize=1)
def _far_a():
    """Hazy hills beyond the plateau (parallax 0.15) + a faint Canterhorn far to the west."""
    w = W + 30
    cv = new(w)
    xs = np.arange(w)
    # distant mountain with a white castle glint
    m0 = mix(FAR_A[0], 'indigo', 0.10)
    cv.poly([(40, 60), (80, 26), (92, 34), (126, 60)], m0)
    cv.poly([(80, 26), (92, 34), (126, 60), (88, 60)], mix(FAR_A[0], 'indigo', 0.22))
    cv.poly([(75, 31), (80, 26), (86, 30), (82, 32)], '#eef6fb')
    for k, (tx, th) in enumerate(((77, 4), (80, 7), (83, 5), (86, 3))):
        cv.vline(tx, 29 - th, 29, '#f4f2fa')
        cv.px(tx, 28 - th, 'gold' if k == 1 else '#c8b8f0')
    h1 = 52 + 4 * np.sin(xs / 23.0 + 1) + 3 * np.sin(xs / 9.0) + 2 * np.sin(xs / 51.0)
    h2 = 58 + 3 * np.sin(xs / 17.0 + 2) + 2 * np.sin(xs / 7.0 + 1)
    for x in range(w):
        cv.vline(x, int(h1[x]), 72, FAR_A[0])
        cv.px(x, int(h1[x]), mix(FAR_A[0], 'white', 0.35))
    for x in range(w):
        cv.vline(x, int(h2[x]), 72, FAR_A[1])
    for i in range(46):
        x = int(hash01(7, i) * w)
        y = int(h2[min(w - 1, x)])
        r = 1 + int(hash01(8, i) * 2)
        cv.ellipse(x, y, r + 1, r, FAR_A[1])
        cv.px(x - 1, y - r, mix(FAR_A[1], 'white', 0.3))
    return to_layer(cv)


def _rock_fill(cv, mask, ramp, seed, light_bias=0.0, ledge_cols=None, slope=0.0, cols_w=0.0):
    """Stratified rock texture inside mask: tilted strata bands, fbm mottling, vertical joints,
    cracks, and grassy ledges on top of some strata."""
    h, w = mask.shape
    n1 = fbm(h, w, 14, seed)
    n2 = vnoise(h, w, 3, seed + 5)
    yy, xx = np.mgrid[0:h, 0:w]
    sy = yy + xx * slope + n1 * 7
    band = np.floor(sy / 6.0)
    bt = (band * 0.6180339) % 1.0
    within = (sy / 6.0) % 1.0
    t = 0.30 + 0.32 * bt + 0.25 * (n1 - 0.5) + 0.12 * (n2 - 0.5) + light_bias
    t = np.where(within < 0.20, t + 0.20, t)
    t = np.where(within > 0.80, t - 0.22, t)
    if cols_w:
        jx = (xx + (n1 * 9).astype(int)) % 9
        t = np.where(jx == 0, t - 0.18, t)
        t = np.where(jx == 1, t + 0.08, t)
    t = np.clip(t, 0, 0.999)
    lvl = ramp_idx(t, len(ramp), 0, 0, h, w, 0.8)
    ra = np.array([col(r) for r in ramp], np.uint8)
    out = ra[lvl]
    crack = (vnoise(h, w, 2.5, seed + 9) > 0.87) & (within > 0.3)
    out[crack] = ra[0]
    if ledge_cols is not None:
        lc = np.array([col(r) for r in ledge_cols], np.uint8)
        ledge = (within < 0.20) & (vnoise(h, w, 8, seed + 11) > 0.50) & (bt > 0.3)
        lv = np.clip((n2 * len(lc)).astype(int), 0, len(lc) - 1)
        out[ledge] = lc[lv[ledge]]
        hang = np.zeros_like(ledge)
        hang[1:] = ledge[:-1] & (n2[1:] > 0.58)
        hang2 = np.zeros_like(ledge)
        hang2[1:] = hang[:-1] & (n2[1:] > 0.75)
        out[(hang | hang2) & ~ledge] = lc[0]
    cv.a[mask] = out[mask]


@lru_cache(maxsize=1)
def _main_a():
    """Static main layer: plateau + forest, river banks, upper river base, gorge walls,
    lower river base (all at parallax 1.0)."""
    cv = new(MWA)
    hh = H
    yy, xx = np.mgrid[0:hh, 0:MWA]
    xr = np.arange(MWA)
    rim = np.array([rim_y(x) if (x < CXA - AHW or x > CXA + AHW) else lipy(x) for x in xr])
    lipa = np.array([lipy(x) for x in xr])
    lb = np.array([bank_x(y, -1) for y in range(hh)])
    rb = np.array([bank_x(y, 1) for y in range(hh)])
    nz = fbm(hh, MWA, 10, 21)
    haze = np.array(col(HAZE_A), np.float32)
    # ---- plateau (meadow under the forest), from the horizon down to the rims
    plat = (yy >= HZA) & (yy < rim[None, :])
    depth = np.clip((yy - HZA) / 44.0, 0, 1)
    grass = [mix('#3d8f62', '#8fbfb0', 0.25), mix('green', '#8fbfb0', 0.15), mix('green', 'leaf', 0.35),
             mix('leaf', 'pale_leaf', 0.25), mix('pale_leaf', 'cream', 0.25)]
    t = 0.30 + 0.40 * depth + 0.30 * (nz - 0.5)
    lvl = ramp_idx(np.clip(t, 0, 0.999), 5, 0, 0, hh, MWA, 0.9)
    ga = np.array([col(g) for g in grass], np.uint8)[lvl].astype(np.float32)
    ga = ga + (haze - ga) * ((1 - depth)[..., None] ** 2 * 0.5)
    cv.a[plat] = ga.astype(np.uint8)[plat]
    # ---- upper river (between the banks, above the lip)
    river = (yy >= RIV0) & (xx + 0.5 >= lb[:, None]) & (xx + 0.5 <= rb[:, None]) & (yy < lipa[None, :])
    rd = np.clip((yy - RIV0) / (LIP0 + LIPK - RIV0), 0, 1)
    rt = 0.30 + 0.50 * rd + 0.10 * (nz - 0.5)
    rcols = ['#2c6f93', '#3a87ab', '#4f9fc2', '#79bfd8', '#a9dceb']
    rl = ramp_idx(np.clip(rt, 0, 0.999), 5, 0, 0, hh, MWA, 0.8)
    ra = np.array([col(r) for r in rcols], np.uint8)[rl].astype(np.float32)
    ra = ra + (haze - ra) * ((1 - rd)[..., None] ** 2 * 0.45)
    cv.a[river] = ra.astype(np.uint8)[river]
    for y in range(RIV0, int(LIP0 + LIPK) + 1):
        for side in (-1, 1):
            bx = int(round(bank_x(y, side)))
            if y < lipy(bx):
                cv.px(bx, y, mix('sand', 'cream', 0.3))
                cv.px(bx + side, y, mix('pale_sage', 'sand', 0.3))
                cv.px(bx - side, y, '#2c6f93')
    # rocks in the rapids
    for i in range(10):
        y = RIV0 + 3 + int(hash01(i, 61) * 11)
        x = int(lerp(bank_x(y, -1) + 4, bank_x(y, 1) - 4, hash01(i, 62)))
        if y < lipy(x) - 3:
            cv.rect(x, y, 2 + (i % 2), 1, mix('night', 'deep_teal', 0.3))
            cv.px(x, y - 1, 'stone')
    # ---- lower river base (in the gorge)
    lower = poly_mask(hh, MWA, [(water_edge(150, -1), 150), (CXA, 128), (water_edge(150, 1), 150),
                                (water_edge(181, 1), 181), (water_edge(181, -1), 181)])
    lt = np.clip(0.25 + (yy - 140) / 70 + 0.2 * (nz - 0.5), 0, 0.999)
    lc = [mix('deep_teal', 'teal', 0.3), 'teal', mix('teal', 'aqua', 0.45), mix('aqua', 'mint_cream', 0.3)]
    ll = ramp_idx(lt, 4, 0, 0, hh, MWA, 0.8)
    cv.a[lower] = np.array([col(r) for r in lc], np.uint8)[ll][lower]
    # ---- gorge walls (left in shade, right sunlit)
    we_l = np.array([water_edge(y, -1) for y in range(hh)])
    we_r = np.array([water_edge(y, 1) for y in range(hh)])
    walls_l = (xx < CXA - AHW + 3) & (yy >= rim[None, :]) & (xx + 0.5 < we_l[:, None])
    walls_r = (xx > CXA + AHW - 3) & (yy >= rim[None, :]) & (xx + 0.5 > we_r[:, None])
    ledge = [mix('green', 'forest', 0.3), 'green', mix('green', 'leaf', 0.5), 'leaf']
    _rock_fill(cv, walls_l, ROCK_SH, 41, light_bias=0.0,
               ledge_cols=[mix(c_, 'indigo', 0.30) for c_ in ledge], slope=-0.12, cols_w=1)
    _rock_fill(cv, walls_r, ROCK_LIT, 43, light_bias=0.04, ledge_cols=ledge, slope=0.10, cols_w=1)
    # ambient occlusion under the rim lip + wet dark foot
    for x in range(MWA):
        if CXA - AHW + 3 <= x <= CXA + AHW - 3:
            continue
        ry = int(rim[x])
        side = -1 if x < CXA else 1
        cv.px(x, ry, mix('green', 'leaf', 0.3) if side > 0 else mix('green', 'indigo', 0.25))
        cv.px(x, ry + 1, mix('dark_olive', 'green', 0.3))
        if hash01(x, 7) > 0.45:
            cv.px(x, ry + 2, mix('dark_olive', 'green', 0.2))
        if hash01(x, 8) > 0.8:
            cv.px(x, ry + 3, mix('dark_olive', 'green', 0.1))
    for side in (-1, 1):
        for y in range(int(LIP0 + LIPK + 50), 181):
            x = int(round(water_edge(y, side)))
            cv.px(x, y, mix('ink', 'deep_teal', 0.35))
            cv.px(x - side, y, mix('night', 'deep_teal', 0.3))
            cv.px(x - 2 * side, y, mix('night', 'deep_teal', 0.15) if hash01(x, y) > 0.4 else cv.a[y, x - 2 * side])
        for i in range(6):
            y = int(LIP0 + LIPK + 54 + i * 4.5 + hash01(side, i) * 2)
            x = int(water_edge(y, side)) + side * 1
            r = 2 + i * 0.6
            blobs(cv, [(x, y, r + 1, r * 0.7, 0)], ROCK_SH[1:] if side < 0 else ROCK_LIT[:4],
                  outline=mix('ink', 'deep_teal', 0.3))
    # ---- trees: far forest band, then river-bank and rim trees (bigger when nearer)
    trees = []
    for i in range(90):
        x = int(hash01(1, i) * MWA)
        y = HZA + 1 + int(hash01(2, i) * 5)
        if bank_x(RIV0, -1) - 2 < x < bank_x(RIV0, 1) + 2 and y > RIV0 - 8:
            continue
        trees.append((x, y, 2.2 + hash01(3, i) * 1.6, 100 + i))
    for side in (-1, 1):
        for i in range(8):
            y = RIV0 + 1 + i * 3 + int(hash01(side, i, 4) * 2)
            bx = bank_x(y, side)
            x = int(bx + side * (4 + hash01(side, i) * 5 + (y - RIV0) * 0.4))
            trees.append((x, y, 3 + (y - HZA) * 0.12, 200 + i + 20 * side))
        for i in range(16):
            x = int(CXA + side * (AHW + 6 + hash01(side, i, 8) * 110))
            if not (0 <= x < MWA):
                continue
            ry = rim_y(x)
            y = int(HZA + 6 + hash01(side, i, 9) * (ry - HZA - 8))
            trees.append((x, y, 3 + (y - HZA) * 0.14, 300 + i + 40 * side))
    for side in (-1, 1):
        for i in range(9):
            x = int(CXA + side * (AHW + 4 + i * 13 + hash01(side, i, 5) * 6))
            if not (0 <= x < MWA):
                continue
            trees.append((x, int(rim_y(x)) - 1, 5.5 + 2.5 * hash01(side, i, 6), 400 + i + 10 * side))
    trees.sort(key=lambda t: t[1])
    for (x, y, r, sd) in trees:
        d = clamp((y - HZA) / 40)
        ramp = [mix(c_, HAZE_A, (1 - d) * 0.55) for c_ in LEAF_A]
        out = mix(LEAF_A_OUT, '#8fbcc6', (1 - d) * 0.6)
        tree(cv, x, y, r, sd, ramp, out, trunk=(mix('brew', '#8fbcc6', 1 - d), mix('brew_mid', '#a9d0d6', 1 - d)),
             trunk_h=max(1, int(r * 0.35)))
    # little lookout tower on the right rim (pastel, conical roof, pennant)
    gx, gy = CXA + AHW + 34, int(rim_y(CXA + AHW + 34)) - 2
    cv.rect(gx - 4, gy - 12, 9, 12, 'cream')
    cv.rect(gx + 2, gy - 12, 3, 12, mix('cream', 'tan', 0.35))
    cv.rect(gx - 1, gy - 5, 2, 5, 'plum_brown')
    cv.px(gx - 1, gy - 6, 'plum_brown')
    cv.rect(gx - 2, gy - 10, 2, 2, 'periwinkle')
    cv.hline(gx - 5, gx + 5, gy - 13, 'tan')
    cv.poly([(gx - 6, gy - 13), (gx, gy - 22), (gx + 6, gy - 13)], 'periwinkle')
    cv.poly([(gx, gy - 22), (gx + 6, gy - 13), (gx + 1, gy - 13)], mix('periwinkle', 'royal', 0.45))
    cv.px(gx, gy - 23, 'gold')
    return to_layer(cv)


# ---- curtain of falling water
@lru_cache(maxsize=1)
def _curtain():
    xs, ys, dys, hs = [], [], [], []
    for x in range(CXA - AHW, CXA + AHW + 1):
        top = int(round(lipy(x)))
        fh = int(round(fallh(x)))
        for dy in range(fh):
            xs.append(x)
            ys.append(top + dy)
            dys.append(dy)
            hs.append(fh)
    xs = np.array(xs)
    ys = np.array(ys)
    dys = np.array(dys, np.float32)
    hs = np.array(hs, np.float32)
    xr = np.arange(MWA)
    # ropes of water: smooth brightness noise across x, 2-5 px wide
    rope = np.array([hash01(x // 3, 5) * 0.6 + hash01(x // 5, 6) * 0.4 for x in xr], np.float32)
    ph = np.array([hash01(x // 2, 3) * 40 for x in xr], np.float32)
    sp = np.array([3.0 + 1.2 * hash01(x // 2, 4) for x in xr], np.float32)
    u = (xs - CXA) / AHW
    lit = np.clip(0.62 + 0.38 * u, 0.25, 1.0)                 # left arm faces away from the sun
    lit = np.where(np.abs(u) < 0.45, np.maximum(lit, 0.85), lit)
    rel = dys / hs
    thr = bayer(1, 4)[0]
    dith = (thr[(xs + ys) % 4] - 0.5).astype(np.float32)
    edge = np.abs(u) > 0.86                                  # arm tips seen edge-on
    gap = np.zeros(len(xs), np.int8)                         # rocks splitting the flow at the lip
    for rx, gl in LIP_ROCKS:
        g = (np.abs(xs - rx) <= 1) & (dys < gl)
        gap[g] = 1
        gap[g & (dys < 2)] = 2
    return xs, ys, dys, rel, ph[xs], sp[xs], rope[xs], lit.astype(np.float32), dith, edge, gap


LIP_ROCKS = ((CXA - 33, 7), (CXA + 24, 5), (CXA + 61, 8))


_WCOLS = np.array([col(w) for w in WATER], np.uint8)
_WCOLS_SH = np.array([col(mix(w, '#7a74b8', 0.32)) for w in WATER], np.uint8)


def draw_curtain(c, f, cam):
    xs, ys, dys, rel, ph, sp, rope, lit, dith, edge, gap = _curtain()
    sx = xs - cam
    ok = (sx >= 0) & (sx < W)
    v = dys - f * sp + ph
    streak = ((v / 7.0 + rope * 3).astype(int) % 4 == 0).astype(np.float32)
    t = 0.40 + 0.22 * rope + 0.42 * rel + 0.14 * streak
    band = ((dys - f * 2.4 + ph * 0.3) % 15.0) < 1.6
    t = np.where(band & (rel > 0.15), t + 0.22, t)
    # green glassy lip, then the water breaks white
    t = np.where(dys < 2, 0.28 + 0.12 * rope, t)
    t = np.where((dys >= 2) & (dys < 5), 0.40 + 0.2 * rope + 0.1 * streak, t)
    lvl = np.clip((t * 6 + dith * 0.7).astype(int), 0, 5)
    lvl = np.where(edge & (rel < 0.6), np.maximum(lvl - 1, 1), lvl)
    lvl = np.where(gap == 1, np.minimum(lvl, 1), lvl)
    lvl = np.where((rope < 0.22) & (rel < 0.55) & (dys >= 5), np.maximum(lvl - 1, 1), lvl)
    shaded = lit + dith * 0.3 < 0.7
    out = np.where(shaded[:, None], _WCOLS_SH[lvl], _WCOLS[lvl])
    out[gap == 2] = C('#4a4a5c')
    c.a[ys[ok], sx[ok]] = out[ok]
    # bright rolling edge on the lip
    lx = np.arange(CXA - AHW, CXA + AHW + 1)
    tops = np.array([int(round(lipy(x))) for x in lx])
    s = lx - cam
    m = (s >= 0) & (s < W)
    glint = ((lx + f // 3) % 9 < 2)
    cl = np.where(glint[:, None], C('white')[None], C('#7fd6c9')[None])
    c.a[tops[m], s[m]] = cl[m]


# ---- upper river rapids (horizontal crest dashes sliding toward the lip)
@lru_cache(maxsize=1)
def _river_uv():
    pts = []
    for y in range(RIV0, int(LIP0 + LIPK) + 1):
        xl, xr = bank_x(y, -1), bank_x(y, 1)
        for x in range(int(math.ceil(xl)) + 1, int(xr)):
            if y < lipy(x) - 0.5:
                pts.append((x, y))
    p = np.array(pts)
    xs, ys = p[:, 0], p[:, 1]
    z = 240.0 / (ys - RIV0 + 5.0)
    xl = np.array([bank_x(y, -1) for y in ys])
    xr = np.array([bank_x(y, 1) for y in ys])
    u = (xs - xl) / np.maximum(1, xr - xl)
    near = np.clip((ys - RIV0) / (LIP0 - RIV0), 0, 1)
    gap = np.array([lipy(x) for x in xs]) - ys
    return xs, ys, z, u, near, gap


def _dash_tex():
    R = rng(606)
    t = R.random((97, 64))
    k = np.ones(7) / 7.0
    t = np.array([np.convolve(np.concatenate([r[-3:], r, r[:3]]), k, 'valid') for r in t])
    t = (t - t.min()) / (t.max() - t.min())
    return t


_TEX = _dash_tex()


def draw_rapids(c, f, cam):
    xs, ys, z, u, near, gap = _river_uv()
    sx = xs - cam
    ok = (sx >= 0) & (sx < W)
    tv = (z * 1.6 - f * 0.42).astype(int) % 97
    tu = (u * 63 + z * 0.3).astype(int) % 64
    n = _TEX[tv, tu]
    thr = 0.70 - 0.16 * near
    thr = np.where(gap < 3, 0.40, thr)
    foam = ok & (n > thr)
    hi = n > thr + 0.07
    c.a[ys[foam], sx[foam]] = np.where(hi[foam][:, None], C('white')[None], C('#bfe6f2')[None])
    # darker accelerating sheet right above the lip
    sheet = ok & (gap >= 1) & (gap < 2.5) & ~foam
    c.a[ys[sheet], sx[sheet]] = C('#246e8c')


_BARREL = Sprite.from_ascii('''
    .bbb.
    bBLBb
    hhhhh
    bBLBb
    hhhhh
    .bbb.
''', {'b': 'brew_mid', 'B': 'latte', 'L': 'crema', 'h': 'steel'})


def barrel_pos(f):
    """Barrel on the upper river: hops downstream on beats 90/108/126/144, tips over the lip on 162."""
    hops = [(72, (CXA - 12, 76)), (90, (CXA - 19, 78)), (108, (CXA - 26, 80)), (126, (CXA - 33, 83)),
            (144, (CXA - 40, 86))]
    x, y = hops[0][1]
    lift = 0.0
    for i in range(1, len(hops)):
        fb, pb = hops[i]
        fa, pa = hops[i - 1]
        if f >= fb + 8:
            x, y = pb
        elif f >= fb:
            t = (f - fb) / 8.0
            x, y = lerp(pa[0], pb[0], t), lerp(pa[1], pb[1], t)
            lift = math.sin(t * math.pi) * 3
            break
        else:
            x, y = pa
            break
    tip = 0
    if f >= 162:
        t = (f - 162) / 8.0
        y = 88 + t * t * 40
        x += t * 3
        tip = 1
    return x, y - lift, tip


def draw_barrel(c, f, cam):
    if f > 170:
        return
    x, y, tip = barrel_pos(f)
    sx, sy = int(round(x)) - cam, int(round(y))
    if tip:
        rot = Sprite(np.ascontiguousarray(np.rot90(_BARREL.rgba, k=1 + (f // 2) % 4)))
        c.blit(rot, sx - 3, sy - 3)
        return
    bob = (f // 6) % 2
    c.hline(sx - 3, sx + 3, sy + 1, '#d8f2f8')
    c.px(sx - 4 - (f // 3) % 2, sy + 1, 'white')
    c.px(sx + 4 + (f // 3) % 2, sy + 1, 'white')
    _BARREL.draw(c, sx, sy + 1 - bob, anchor='bottom')


# ---- lower river foam
@lru_cache(maxsize=1)
def _lower_uv():
    m = poly_mask(H, MWA, [(water_edge(150, -1), 150), (CXA, 128), (water_edge(150, 1), 150),
                           (water_edge(181, 1), 181), (water_edge(181, -1), 181)])
    ys, xs = np.nonzero(m)
    z = 300.0 / (ys - 96.0)
    u = (xs - CXA) / (ys - 96.0)
    return xs, ys, z, u


def draw_lower(c, f, cam):
    xs, ys, z, u = _lower_uv()
    sx = xs - cam
    ok = (sx >= 0) & (sx < W)
    tv = (z * 6.0 - f * 0.75).astype(int) % 97
    tu = (u * 90 + 200).astype(int) % 64
    n = _TEX[tv, tu]
    churn = np.clip(1 - (ys - 140) / 30.0, 0, 1)
    thr = 0.76 - 0.30 * churn
    foam = ok & (n > thr)
    c.a[ys[foam], sx[foam]] = np.where((n[foam] > thr[foam] + 0.06)[:, None], C('white')[None],
                                       C('#a9e4dc')[None])


# ---- mist
@lru_cache(maxsize=None)
def _puff(r, seed, tone=0):
    s = 2 * r + 6
    cv = new(s, s)
    R = rng(seed * 31 + r)
    bl = [(r + 3, r + 3.5, r, r * 0.92, 0)]
    for i in range(4):
        a = -math.pi / 2 + (R.random() - 0.5) * 2.4
        rr = r * (0.45 + 0.15 * R.random())
        bl.append((r + 3 + math.cos(a) * (r - rr), r + 3.5 + math.sin(a) * (r - rr) * 0.9, rr, rr * 0.9, r * 0.3))
    ramp = MIST if not tone else [mix(m, '#9fb0d8', 0.35) for m in MIST]
    blobs(cv, bl, ramp, light=np.array([-0.45, -0.75, 0.5]), dith=0.55, crease=False)
    a, m = to_layer(cv)
    return a.copy(), m


def draw_puff(c, x, y, r, seed, level=1.0, tone=0):
    a, m = _puff(int(r), seed % 5, tone)
    h, w = m.shape
    x0, y0 = int(x) - w // 2, int(y) - h // 2
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    mm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    if level < 1:
        mm = mm & (bayer(Y1 - Y0, X1 - X0, X0, Y0) < level)
    np.copyto(c.a[Y0:Y1, X0:X1], a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0], where=mm[..., None])


def base_y(x_main):
    return lipy(x_main) + fallh(x_main)


def draw_mist(c, f, cam):
    """Thundering mist: a churning bank over the plunge + puffs that rise, swell and thin out."""
    # rising puffs (back first)
    for i in range(18):
        P = 84 + int(hash01(i, 1) * 40)
        t = ((f + hash01(i, 2) * P) % P) / P
        u = (hash01(i, 3) - 0.5) * 1.5
        x0 = CXA + u * AHW
        y0 = base_y(x0) - 10
        rise = 26 + 30 * hash01(i, 4) * (1 - abs(u)) ** 2
        y = y0 - t * rise
        x = x0 + math.sin(t * 3 + i) * 3 + t * 10 * u - cam
        r = 5 + 7 * t + 3 * hash01(i, 5)
        if t > 0.86:
            continue
        lvl = 1.0 if t < 0.62 else 0.5
        draw_puff(c, x, y, r, i, lvl, tone=1 if u < -0.45 else 0)
    # churning bank along the foot of the curtain
    for i in range(17):
        u = -1.0 + i / 8.0
        xm = CXA + u * (AHW - 4)
        yb = base_y(xm) - 6 + int(round(math.sin(f * 0.13 + i * 1.7) * 2))
        r = 11 + 4 * hash01(i, 12) - 2 * abs(u)
        x = xm + math.sin(f * 0.07 + i) * 2 - cam
        draw_puff(c, x, yb, r, i + 7, 1.0, tone=1 if u < -0.5 else 0)
    # spray droplets flung outward
    for i in range(26):
        P = 18 + int(hash01(i, 8) * 14)
        t = ((f + hash01(i, 9) * P) % P) / P
        u = hash01(i, 10) * 2 - 1
        xm = CXA + u * AHW * 0.9
        x = xm + u * t * 22 - cam
        y = base_y(xm) - 12 - math.sin(t * math.pi) * 16 + t * 10
        c.px(int(x), int(y), 'white' if t < 0.6 else '#d6e6f4')


def draw_rainbow(c, f, cam):
    """Rainbow arc in the spray, gently breathing."""
    cx, cy, r0 = CXA - cam + 14, 158, 58
    bands = ['red', 'orange', 'gold', 'leaf', 'sky', 'violet']
    x0, x1 = max(0, cx - r0 - 2), min(W, cx + r0 + 3)
    y0, y1 = max(0, cy - r0 - 2), min(H, 146)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
    ang = np.arctan2(cy - yy - 0.5, xx + 0.5 - cx)
    breathe = 0.34 + 0.07 * math.sin(f * 0.11)
    fade_end = np.clip(np.sin(ang) * 1.5, 0, 1) * np.clip((146 - yy) / 22.0, 0, 1)
    reg = c.a[y0:y1, x0:x1]
    for k, bc in enumerate(bands):
        m = (d >= r0 - k - 1) & (d < r0 - k) & (ang > 0.15) & (ang < 2.99)
        a = (breathe * fade_end)[m][:, None]
        src = reg[m].astype(np.float32)
        reg[m] = np.clip(src + (np.array(col(bc), np.float32) - src) * a, 0, 255).astype(np.uint8)


# ---- foreground viewing platform
_VIEWER = Sprite.from_ascii('''
    ..oooooo.
    .oPPPPPPo
    ogpppppPo
    oGpppppPo
    .oppppppo
    ..oooooo.
    ....oSo..
    ....oSo..
    ....oSo..
    ....oSo..
    ....oSo..
    ....oSo..
    ....oSo..
    ....oSo..
    ...oSSSo.
    ..oSSSSSo
''', {'o': mix('ink', 'royal', 0.3), 'P': 'baby_blue', 'p': 'periwinkle', 'g': 'glass_hi',
      'G': 'crystal', 'S': 'steel'})


@lru_cache(maxsize=1)
def _fg_a():
    """Viewing platform in the bottom-right foreground (parallax 1.35)."""
    w = W + 70
    cv = new(w)
    x0 = 216
    cv.poly([(x0 - 8, 181), (x0 + 4, 167), (w, 167), (w, 181)], 'brew_mid')
    for i in range(14):
        y = 167 + i
        cv.hline(x0 + 4 - int(i * 0.9), w, y, 'latte' if i % 3 else 'brew_mid')
    for i in range(30):
        cv.vline(x0 + 9 + i * 8 - 0, 168, 180, mix('brew_mid', 'brew', 0.5))
    cv.hline(x0 + 4, w, 167, 'crema')
    for i in range(16):
        x = x0 + 4 + i * 10
        cv.rect(x, 150, 3, 17, 'brew')
        cv.rect(x, 150, 1, 17, 'crema')
        cv.rect(x - 1, 148, 5, 2, 'latte')
        cv.hline(x - 1, x + 3, 148, 'crema')
        for k in (4, 7):
            cv.vline(x + k, 154, 166, mix('latte', 'brew_mid', 0.35))
    cv.rect(x0 + 2, 151, w, 2, 'latte')
    cv.hline(x0 + 2, w, 151, 'crema')
    cv.rect(x0 + 2, 160, w, 1, 'brew_mid')
    _VIEWER.draw(cv, x0 + 44, 167, anchor='bottom')
    cv.rect(x0 + 72, 143, 20, 5, 'brick')
    cv.hline(x0 + 72, x0 + 91, 143, 'coral')
    for i in range(10):
        fxp = x0 + 73 + i * 2
        cv.px(fxp, 141 - (i % 2), ('hot_pink', 'gold', 'white', 'lilac')[i % 4])
        cv.px(fxp, 142, 'green')
        cv.px(fxp + 1, 142, 'leaf')
    lx = x0 + 106
    cv.rect(lx, 108, 2, 59, 'navy')
    cv.rect(lx, 108, 1, 59, 'indigo')
    cv.rect(lx - 3, 107, 8, 2, 'navy')
    cv.rect(lx - 2, 101, 6, 6, 'butter')
    cv.frame(lx - 2, 101, 6, 6, 'navy')
    cv.px(lx - 1, 102, 'white')
    cv.poly([(lx - 3, 101), (lx + 1, 97), (lx + 5, 101)], 'navy')
    cv.px(lx + 1, 96, 'gold')
    # hanging flower basket
    cv.px(lx + 4, 109, 'grey')
    cv.ellipse(lx + 5, 113, 3, 2, 'brew_mid')
    for dx in (-2, 0, 2):
        cv.px(lx + 5 + dx, 111, 'hot_pink' if dx else 'gold')
    for (bx, by, br, sd) in ((x0 - 14, 178, 10, 7), (x0 - 2, 176, 8, 8), (x0 + 8, 180, 6, 9)):
        blobs(cv, canopy(bx, by, br, sd), LEAF_A, outline=LEAF_A_OUT)
    for i in range(9):
        px = x0 - 20 + int(hash01(i, 71) * 30)
        py = 168 + int(hash01(i, 72) * 10)
        cv.px(px, py, ('hot_pink', 'white', 'gold')[i % 3])
    return to_layer(cv)


def draw_pennant(c, f, x, y):
    """Pennant flag waving from a pole top at (x, y) (screen)."""
    c.vline(x, y - 1, y + 9, 'navy')
    for i in range(11):
        wv = int(round(math.sin(f * 0.35 - i * 0.7) * (i / 6.0)))
        hh = max(1, 5 - i // 2)
        cc = 'hot_pink' if (i // 3) % 2 == 0 else 'lavender'
        c.vline(x + 1 + i, y + wv, y + wv + hh - 1, cc)


@lru_cache(maxsize=1)
def _zeppelin():
    cv = new(48, 22)
    blobs(cv, [(24, 8, 21, 6.5, 0), (8, 8, 7, 5, -1), (40, 8, 6, 5, -1)],
          ['#b8a8d8', mix('cotton', 'lavender', 0.4), 'cotton', '#fff0f8', 'white'],
          outline=mix('plum', 'indigo', 0.4), light=np.array([-0.4, -0.8, 0.45]), crease=False)
    for x in range(6, 43):
        cv.px(x, 9, 'lavender')
        cv.px(x, 10, mix('lavender', 'violet', 0.4))
    cv.px(24, 5, 'gold')
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv.px(24 + dx, 5 + dy, 'gold')
    cv.poly([(40, 3), (46, 0), (46, 6), (42, 7)], 'hot_pink')
    cv.poly([(40, 13), (46, 16), (46, 10), (42, 9)], mix('hot_pink', 'rose', 0.5))
    cv.hline(41, 46, 8, 'rose')
    cv.round_rect(16, 15, 16, 5, 'cream', r=1)
    cv.hline(17, 30, 15, 'white')
    for i in range(5):
        cv.rect(18 + i * 3, 16, 2, 2, 'periwinkle')
        cv.px(18 + i * 3, 16, 'glass_hi')
    cv.hline(16, 31, 19, mix('tan', 'lavender', 0.3))
    cv.vline(19, 13, 15, 'grey')
    cv.vline(29, 13, 15, 'grey')
    a, m = to_layer(cv)
    rgba = np.zeros(a.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = a
    rgba[..., 3] = m * 255
    return rgba


def draw_zeppelin(c, f, x, y):
    c.blit(_zeppelin(), x, y)
    px, py = x + 33, y + 17
    if (f // 2) % 2:
        c.vline(px, py - 2, py + 2, 'stone')
    else:
        c.hline(px - 1, px + 1, py, 'stone')


def cam_a(f):
    return int(round(lerp(4, 44, cruise(seg(f, 72, CUT2 + 10)))))


def render_a(f):
    cam = cam_a(f)
    c = _sky_a().copy()
    draw_clouds(c, f, [(30, 12, 46, 1, 0.10, 0), (150, 38, 30, 2, 0.14, 1), (250, 6, 58, 3, 0.08, 0),
                       (330, 42, 24, 4, 0.16, 1)], ox=cam * 0.1)
    comp(c, _far_a(), int(cam * 0.15))
    zx = int(lerp(270, 186, seg(f, 72, CUT2))) - int(cam * 0.25)
    zy = 24 + int(round(math.sin(f * 0.07)))
    draw_zeppelin(c, f, zx, zy)
    comp(c, _main_a(), cam)
    draw_rapids(c, f, cam)
    draw_lower(c, f, cam)
    draw_curtain(c, f, cam)
    draw_barrel(c, f, cam)
    draw_mist(c, f, cam)
    draw_rainbow(c, f, cam)
    bx = int(lerp(56, 90, seg(f, 72, CUT2))) - int(cam * 0.6)
    draw_balloon(c, bx, 72, f, size='m')
    fcam = int(cam * 1.35)
    comp(c, _fg_a(), fcam)
    draw_pennant(c, f, 216 + 106 + 1 - fcam, 88)
    return c


# ================================================================== SHOT B: FOAL MOUNTAIN & PONHENGE
SKY_B = ['#3878cc', '#478fe0', '#5ea8ec', '#84c5f4', '#b5e0f7', '#dcf1f1']
HAZE_B = '#b7d9e6'
MT = ['#1d4560', '#255676', '#2c6a88', '#36809a', '#4f9aaa', '#73b6b8', '#9dd0c8']
SNOW = ['#8ea8cc', '#aec4de', '#cddcee', '#e6f0f8', '#ffffff']
PEAKS = ((98, 10, 0.92), (50, 42, 0.95), (150, 30, 0.85), (196, 56, 0.95), (10, 62, 0.9))
MW_M = W + 24
GRASS_B = [mix('forest', 'green', 0.35), mix('green', 'forest', 0.1), mix('green', 'leaf', 0.35), 'leaf',
           mix('leaf', 'pale_leaf', 0.6)]
STONE = ['#454a63', '#5b627b', '#767f95', '#949db0', '#b9c0cb', '#d6dade']
STONE_OUT = '#33354a'
RUNE_DIM = '#3d4466'
KNOLL_C = (226, 118)          # stone circle centre (knoll layer coords)


def vnoise2(h, w, cy, cx, seed):
    """Anisotropic value noise: cell cy tall, cx wide."""
    R = rng(seed)
    gh, gw = int(h / cy) + 3, int(w / cx) + 3
    g = R.random((gh, gw))
    yy = np.arange(h) / cy
    xx = np.arange(w) / cx
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    fy = yy - y0
    fx = xx - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c_ = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None]
    bot = c_ + (d - c_) * fx[None]
    return top + (bot - top) * fy[:, None]


@lru_cache(maxsize=1)
def _sky_b():
    c = Canvas()
    fx.sky(c, SKY_B, 0, 110)
    fx.glow(c, 30, -30, 110, 'butter', 0.26, rings=6)
    fx.glow(c, 30, -30, 56, 'white', 0.18, rings=4)
    return c


@lru_cache(maxsize=1)
def _ridge_b():
    """Rambling Rock Ridge: long pale slabs tilted up to the east, hazy, beyond the meadows."""
    w = W + 12
    cv = new(w)
    slabs = [(196, 110, 62, 26), (236, 110, 54, 20), (262, 111, 66, 30), (300, 111, 40, 16)]
    top_c = [mix('#d9c9b0', 'lavender', 0.12), '#e6d9c2', '#f3ead8', '#fff8ea']
    face_c = ['#9e93b4', '#ada2c0', '#bdb3cc']
    for k, (x0, base, sw, sh) in enumerate(slabs):
        tip = (x0 + sw, base - sh)
        cv.poly([(x0, base), tip, (x0 + sw + 1, base)], top_c[1])
        # strata parallel to the dip, lighter toward the crest
        for j in range(1, sh, 3):
            ya = base - j
            xa = x0 + sw * (j / sh)
            cv.line(int(xa), ya, int(x0 + sw), int(base - sh + max(0, j - sh) ), top_c[0])
            cv.hline(int(xa) + 1, int(xa) + 4, ya, top_c[2])
        cv.line(x0, base, int(tip[0]), int(tip[1]), top_c[3])
        # steep broken face on the east side (in shade)
        face = [tip, (x0 + sw + 4, base - sh + 5), (x0 + sw + 6, base - sh + 12), (x0 + sw + 8, base),
                (x0 + sw, base)]
        cv.poly(face, face_c[0])
        for j in range(0, sh, 4):
            cv.hline(x0 + sw + 1, x0 + sw + 5, base - sh + 3 + j, face_c[1])
        cv.vline(x0 + sw + 1, base - sh + 2, base - 1, face_c[2])
        # scree at the foot
        for j in range(6):
            cv.px(x0 + sw + 2 + j * 2, base - (j % 2), face_c[1])
    cv.rect(190, 108, w - 190, 6, mix(HAZE_B, '#cfd9c4', 0.4))
    a, m = to_layer(cv)
    a = a.astype(np.float32)
    a[m] = a[m] + (np.array(col(HAZE_B), np.float32) - a[m]) * 0.28
    return a.astype(np.uint8), m


@lru_cache(maxsize=1)
def _mountain_b():
    """Foal Mountain: blue-teal massif, union of cones with gullies, lit from the upper left."""
    w, h = MW_M, 150
    cv = new(w, h)
    xs = np.arange(w)
    tops = np.stack([py + np.abs(xs - px) * s for (px, py, s) in PEAKS])
    idx = np.argmin(tops, 0)
    jag = (vnoise(1, w, 5, 77)[0] - 0.5) * 4 + (vnoise(1, w, 2, 78)[0] - 0.5) * 2
    top = tops.min(0) + jag
    pkx = np.array([p[0] for p in PEAKS])[idx]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = yy >= top[None, :]
    side = np.where(xx < pkx[None, :], -1.0, 1.0)
    # gullies running down the slopes: anisotropic noise sheared along the slope
    big = 0.65 * vnoise2(h, w + 240, 22, 5.5, 81) + 0.35 * vnoise2(h, w + 240, 9, 2.6, 82)
    shear = (xx + 120 + side * (yy - top[None, :]) * 0.55).astype(int)
    shear = np.clip(shear, 1, w + 238)
    g = big[yy, shear]
    gx = big[yy, np.clip(shear + 1, 0, w + 239)] - big[yy, np.clip(shear - 1, 0, w + 239)]
    depth = np.clip((yy - top[None, :]) / 80.0, 0, 1)
    t = 0.55 - 0.26 * side - 1.5 * gx + 0.08 * (g - 0.5)
    t += 0.10 * (1 - np.clip((yy - top[None, :]) / 6.0, 0, 1)) * (side < 0)   # lit ridge crests
    t -= 0.16 * np.clip((yy - 70) / 50.0, 0, 1)
    t = np.clip(t, 0, 0.999)
    lvl = ramp_idx(t, len(MT), 0, 0, h, w, 0.5)
    out = np.array([col(m) for m in MT], np.uint8)[lvl]
    # pale summit snow + snow streaks in gullies
    sn = 30 + (vnoise(1, w, 6, 91)[0] - 0.5) * 12 + np.abs(xs - 98) * 0.12
    snow = inside & ((yy < sn[None, :]) | ((yy < sn[None, :] + 12) & (g > 0.64) & (np.abs(xx - 98) < 60)))
    slv = ramp_idx(np.clip(t + 0.05, 0, 0.999), len(SNOW), 0, 0, h, w, 0.75)
    out[snow] = np.array([col(m) for m in SNOW], np.uint8)[slv][snow]
    # dark rock bands (cliffs)
    band = inside & ~snow & (vnoise2(h, w, 2, 4, 93) > 0.84) & (yy > 40) & (side > 0)
    out[band] = C(MT[0])
    # atmospheric haze: valley mist at the foot
    out = out.astype(np.float32)
    hz = np.clip((yy - 78) / 40.0, 0, 1)[..., None] * 0.55 + 0.10
    out = out + (np.array(col(HAZE_B), np.float32) - out) * hz
    out = out.astype(np.uint8)
    # silhouette edge
    edge = inside & ~np.vstack([np.zeros((1, w), bool), inside[:-1]])
    out[edge & (side < 0)] = C('#bfe4e0')
    out[edge & (side > 0)] = C(MT[3])
    cv.a[inside] = out[inside]
    return to_layer(cv)


def draw_cascade(c, f, ox):
    """Thin waterfall ribbon down a gully of the mountain."""
    pts = [(122, 46), (123, 52), (125, 58), (126, 64), (128, 70), (129, 78), (131, 86)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        for y in range(y0, y1):
            x = int(lerp(x0, x1, (y - y0) / (y1 - y0))) - ox
            on = (y - f // 2) % 5
            c.px(x, y, 'white' if on < 2 else '#cfeaf2')
    c.px(131 - ox - 1, 86, '#e6f4fa')
    c.px(131 - ox + 1, 86, '#e6f4fa')
    if (f // 4) % 2:
        c.px(131 - ox, 85, 'white')


@lru_cache(maxsize=1)
def _mid_b():
    """Meadows at the mountain foot (parallax 0.55): forest belt, fields, hedges, railway + tunnel."""
    w = W + 24
    cv = new(w)
    yy, xx = np.mgrid[0:H, 0:w]
    xs = np.arange(w)
    gtop = 104 + 3 * np.sin(xs / 31.0) + 2 * np.sin(xs / 13.0 + 1)
    land = yy >= gtop[None, :]
    nz = fbm(H, w, 9, 51)
    d = np.clip((yy - 100) / 40.0, 0, 1)
    t = 0.35 + 0.35 * d + 0.3 * (nz - 0.5)
    fields = [mix(g, HAZE_B, 0.30) for g in GRASS_B]
    lvl = ramp_idx(np.clip(t, 0, 0.999), 5, 0, 0, H, w, 0.9)
    cv.a[land] = np.array([col(g) for g in fields], np.uint8)[lvl][land]
    # field patches with hedgerows
    for i in range(7):
        x0 = 120 + i * 30 + int(hash01(i, 3) * 10)
        y0 = 108 + int(hash01(i, 4) * 10)
        fw, fh = 22 + int(hash01(i, 5) * 10), 6 + int(hash01(i, 6) * 4)
        tone = ('#c9d878', '#a9cf7a', '#d9cf8a')[i % 3]
        cv.poly([(x0, y0 + fh), (x0 + 4, y0), (x0 + fw, y0), (x0 + fw - 3, y0 + fh)], mix(tone, HAZE_B, 0.25))
        for k in range(0, fw, 3):
            cv.px(x0 + 4 + k, y0, mix('forest', HAZE_B, 0.3))
    # forest belt along the mountain foot
    for i in range(70):
        x = int(-6 + i * 3.6 + hash01(i, 11) * 3)
        if x > 236:
            break
        y = int(gtop[min(w - 1, max(0, x))]) + 2 + int(hash01(i, 12) * 6)
        r = 3 + hash01(i, 13) * 2.5
        ramp = [mix(c_, HAZE_B, 0.30) for c_ in ['deep_teal', mix('forest', 'deep_teal', 0.4), 'forest',
                                                   mix('forest', 'green', 0.5), 'sage']]
        tree(cv, x, y, r, 500 + i, ramp, mix('deep_teal', 'navy', 0.3), trunk=('brew', 'brew_mid'), trunk_h=1)
    # tunnel portal in the forested foothill
    px, py = 62, 120
    cv.rect(px - 8, py - 13, 18, 14, mix('stone', 'tan', 0.25))
    for k in range(6):
        cv.hline(px - 8, px + 9, py - 13 + k * 3, mix('stone', 'grey', 0.35))
    for k in range(5):
        cv.vline(px - 8 + k * 4 + (k % 2) * 2, py - 12, py, mix('stone', 'grey', 0.25))
    cv.hline(px - 9, px + 10, py - 14, 'paper')
    cv.round_rect(px - 5, py - 10, 11, 11, 'ink', r=2)
    cv.rect(px - 5, py - 6, 1, 7, 'night')
    cv.px(px, py - 11, 'paper')
    for i in range(4):
        tree(cv, px - 10 + i * 7, py - 13 - (i % 2), 3.5, 650 + i,
             [mix(c_, HAZE_B, 0.30) for c_ in ['deep_teal', mix('forest', 'deep_teal', 0.4), 'forest',
                                               mix('forest', 'green', 0.5), 'sage']],
             mix('deep_teal', 'navy', 0.3), trunk_h=1)
    # railway: ballast, sleepers, rails from the portal to the right edge
    for x in range(px - 4, w):
        y = int(py + (x - px) * 0.02 + 1.5 * math.sin((x - px) / 40.0))
        cv.px(x, y, mix('tan', HAZE_B, 0.2))
        cv.px(x, y - 1, 'steel_dk' if x % 3 == 0 else 'steel')
        cv.px(x, y - 2, 'steel_hi' if x >= px + 6 else 'steel')
    for i in range(16):
        x = px + 14 + i * 16
        y = int(py + (x - px) * 0.02 + 1.5 * math.sin((x - px) / 40.0))
        cv.vline(x, y - 10, y - 1, 'brew_mid')
        cv.hline(x - 1, x + 1, y - 9, 'brew')
    for i in range(30):
        x = 140 + int(hash01(i, 21) * 170)
        y = int(gtop[min(w - 1, x)]) + 6 + int(hash01(i, 22) * 16)
        if abs(y - (py + (x - px) * 0.02)) < 4:
            continue
        tree(cv, x, y, 2.5 + hash01(i, 23) * 2, 600 + i, [mix(c_, HAZE_B, 0.22) for c_ in LEAF_A],
             mix(LEAF_A_OUT, HAZE_B, 0.2), trunk_h=1)
    return to_layer(cv)


def rail_y(x_mid):
    px, py = 62, 120
    return int(py + (x_mid - px) * 0.02 + 1.5 * math.sin((x_mid - px) / 40.0))


@lru_cache(maxsize=1)
def _train():
    cv = new(64, 16)
    # loco (front at left)
    cv.round_rect(1, 6, 15, 7, 'royal_lt', r=1)
    cv.hline(2, 14, 6, 'lilac')
    cv.rect(10, 2, 6, 11, 'royal')
    cv.rect(11, 4, 3, 3, 'baby_blue')
    cv.rect(9, 1, 8, 2, 'hot_pink')
    cv.rect(3, 2, 3, 4, 'navy')
    cv.rect(2, 1, 5, 1, 'gold')
    cv.hline(1, 15, 10, 'gold')
    cv.poly([(0, 13), (-1, 13), (1, 10)], 'gold')
    for wx in (4, 9, 13):
        cv.rect(wx - 1, 12, 3, 3, 'ink')
        cv.px(wx, 13, 'grey')
    # carriages
    for k in range(3):
        x0 = 18 + k * 15
        cv.round_rect(x0, 4, 14, 9, 'cream', r=1)
        cv.rect(x0, 3, 14, 2, ('hot_pink', 'mint', 'gold')[k])
        cv.hline(x0, x0 + 13, 11, ('rose', 'teal', 'amber')[k])
        for j in range(3):
            cv.rect(x0 + 2 + j * 4, 6, 2, 3, 'periwinkle')
            cv.px(x0 + 2 + j * 4, 6, 'glass_hi')
        for wx in (x0 + 3, x0 + 10):
            cv.rect(wx - 1, 12, 3, 3, 'ink')
            cv.px(wx, 13, 'grey')
        cv.px(x0 - 1, 10, 'steel')
    a, m = to_layer(cv)
    rgba = np.zeros(a.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = a
    rgba[..., 3] = m * 255
    return rgba


def draw_train(c, f, ox):
    """Friendship Express chugging left into the mountain tunnel (clipped at the portal)."""
    xm = 122 - (f - CUT2) * 1.15
    portal = 62 - 5 - ox
    tx = int(xm) - ox
    ty = rail_y(int(xm) + 20) - 15
    spr = _train()
    clipped = spr.copy()
    cut = portal - tx
    if cut > 0:
        clipped[:, :min(cut, clipped.shape[1])] = 0
    c.blit(clipped, tx, ty)
    # steam puffs trailing back
    for i in range(6):
        age = ((f // 2) + i * 3) % 18
        sx = tx + 4 + age * 1.5 + i
        sy = ty - 1 - age * 0.6
        if sx <= portal + 2:
            continue
        r = 1 + age // 6
        cc = 'white' if age < 9 else '#dfe9f2'
        if age < 15:
            disc(c, sx, sy, r, cc, 1.0 if age < 10 else 0.5)


def _stone_list():
    """Ponhenge stones: (depth s, x, base_y, height, width, kind, i) in knoll coords, back to front."""
    cx, cy = KNOLL_C
    out = []
    n = 12
    for i in range(n):
        th = 2 * math.pi * i / n + 0.26
        s = math.sin(th)
        x = cx + math.cos(th) * 50
        base = cy + 1 + s * 11
        hgt = int(18 + 5 * s + hash01(i, 31) * 3)
        wid = int(round(8 + 1.5 * s))
        kind = 'stand'
        if i == 4:
            kind = 'fallen'
        if i == 8:
            kind = 'stump'
        out.append((s, x, base, hgt, wid, kind, i))
    out.sort()
    return out


RUNES = [
    ['.#.', '#.#', '.#.', '#.#', '.#.'],
    ['#.#', '.#.', '###', '.#.', '#.#'],
    ['.##', '#..', '#..', '#..', '.##'],
    ['#..', '##.', '#.#', '##.', '#..'],
    ['.#.', '###', '.#.', '.#.', '###'],
    ['#.#', '#.#', '###', '.#.', '.#.'],
]


def _stone(cv, x, base, hgt, wid, seed, lean=0):
    """A weathered standing stone: lit left face, shaded right face, rough rounded top."""
    R = rng(seed)
    x0 = int(round(x - wid / 2))
    top = int(base - hgt)
    nt = max(3, wid // 3)
    for yy in range(top, int(base) + 1):
        k = (yy - top) / max(1, hgt)
        dx = int(round(lean * (1 - k)))
        inset = 2 if yy == top else (1 if yy == top + 1 else 0)
        taper = 1 if k < 0.25 else 0
        xa, xb = x0 + dx + inset + taper, x0 + dx + wid - 1 - inset
        cv.hline(xa, xb, yy, STONE[2])
        cv.px(xa, yy, STONE[4])
        cv.px(xa + 1, yy, STONE[4] if k < 0.8 else STONE[3])
        cv.px(xa + 2, yy, STONE[3])
        cv.hline(xb - nt + 1, xb, yy, STONE[1])
        cv.px(xb, yy, STONE[0])
        cv.px(xa - 1, yy, STONE_OUT)
        cv.px(xb + 1, yy, STONE_OUT)
        if yy > base - 3:
            cv.hline(xa + 2, xb - 1, yy, STONE[1])
    cv.hline(x0 + lean + 2, x0 + lean + wid - 3, top - 1, STONE_OUT)
    cv.hline(x0 + lean + 2, x0 + lean + wid - 4, top, STONE[5])
    cv.px(x0 + lean + 1, top, STONE_OUT)
    cv.px(x0 + lean + wid - 2, top, STONE_OUT)
    # pits, lichen and a crack
    for k in range(5):
        lx = x0 + 2 + int(R.random() * (wid - 4))
        ly = top + 3 + int(R.random() * (hgt - 6))
        cv.px(lx, ly, (STONE[1], 'pale_sage', STONE[1], 'khaki', 'moss')[k])
    cx_ = x0 + 2 + int(R.random() * (wid - 4))
    for yy in range(top + 2, top + 2 + int(R.random() * 5) + 2):
        cv.px(cx_ + (yy % 2), yy, STONE[1])
    for k in range(5):
        gx = x0 - 1 + int(R.random() * (wid + 2))
        cv.px(gx, int(base), 'leaf')
        cv.px(gx, int(base) - 1, mix('green', 'leaf', 0.5))


@lru_cache(maxsize=1)
def _knoll_b():
    """The grassy knoll with the Ponhenge stone circle (static part, parallax 1.0)."""
    w = W + 44
    cv = new(w)
    yy, xx = np.mgrid[0:H, 0:w]
    xs = np.arange(w)
    cx, cy = KNOLL_C
    u = (xs - cx) / 100.0
    gtop = 136 - 24 * np.exp(-(u * u) ** 1.6 * 3.0) + 1.5 * np.sin(xs / 17.0)
    land = yy >= gtop[None, :]
    nz = fbm(H, w, 8, 61)
    slope = np.gradient(gtop)
    near = np.clip(1 - (yy - gtop[None, :]) / 22.0, 0, 1)
    t = 0.52 - 1.6 * slope[None, :] * near + 0.25 * (nz - 0.5) - 0.12 * np.clip((yy - 150) / 30.0, 0, 1)
    t = np.where(yy - gtop[None, :] < 1.5, t + 0.25, t)
    lvl = ramp_idx(np.clip(t, 0, 0.999), 5, 0, 0, H, w, 0.85)
    cv.a[land] = np.array([col(g) for g in GRASS_B], np.uint8)[lvl][land]
    # grass tufts along the knoll's top edge
    for x in range(w):
        if hash01(x, 13) > 0.6:
            cv.px(x, int(gtop[x]) - 1, mix('leaf', 'pale_leaf', 0.5))
    # winding path up the knoll to the circle
    for i in range(70):
        k = i / 69.0
        px = lerp(20, cx - 26, k) + math.sin(k * 3.2) * 24
        py = lerp(181, cy + 10, k ** 0.75)
        pw = lerp(10, 2, k)
        cv.hline(int(px - pw), int(px + pw), int(py), mix('tan', 'sand', 0.35))
        cv.px(int(px - pw), int(py), mix('tan', 'latte', 0.4))
        if i % 3 == 0:
            cv.px(int(px + pw * 0.3), int(py), mix('sand', 'cream', 0.4))
    # the henge: a low circular bank + ditch around the ring
    ring_o = ((xx - cx) / 64.0) ** 2 + ((yy - cy - 2) / 15.0) ** 2
    bank = land & (ring_o < 1) & (ring_o > 0.80)
    ditch = land & (ring_o <= 0.80) & (ring_o > 0.70)
    darken(cv, ditch & (yy > cy - 2), (0.80, 0.86, 0.92))
    lit_bank = bank & (yy < cy + 2)
    cv.a[lit_bank & (bayer(H, w) < 0.6)] = C(mix('leaf', 'pale_leaf', 0.5))
    darken(cv, bank & (yy >= cy + 2) & (xx > cx), (0.86, 0.9, 0.95))
    # long shadows (sun upper-left -> shadows fall to the right)
    stones = _stone_list()
    shadow = np.zeros((H, w), bool)
    tri_x = cx + 4
    items = [(x, base, hgt if kind == 'stand' else 6, wid) for (s, x, base, hgt, wid, kind, i) in stones
             if kind != 'fallen']
    items += [(tri_x - 9, cy - 5, 30, 8), (tri_x + 9, cy - 5, 30, 8)]
    for (x, base, hh, wid) in items:
        L = int(hh * 1.5)
        shadow |= poly_mask(H, w, [(x - wid / 2 + 1, base), (x + wid / 2, base),
                                   (x + wid / 2 + L, base + 3), (x - wid / 2 + 2 + L, base + 3)])
    shadow &= land
    darken(cv, shadow & (bayer(H, w) < 0.9), (0.62, 0.70, 0.90))
    # back half of the ring, then the central trilithon + altar, then the front half
    for (s, x, base, hgt, wid, kind, i) in stones:
        if s >= -0.05:
            break
        _draw_stone_kind(cv, s, x, base, hgt, wid, kind, i)
    _stone(cv, tri_x - 9, cy - 5, 30, 8, 900)
    _stone(cv, tri_x + 9, cy - 5, 30, 8, 901)
    _lintel(cv, tri_x - 15, tri_x + 15, cy - 39)
    cv.round_rect(cx - 9, cy + 1, 19, 5, STONE[3], r=1)
    cv.hline(cx - 8, cx + 9, cy + 1, STONE[5])
    cv.hline(cx - 8, cx + 9, cy + 5, STONE[1])
    cv.frame(cx - 10, cy, 21, 7, STONE_OUT)
    for (s, x, base, hgt, wid, kind, i) in stones:
        if s < -0.05:
            continue
        _draw_stone_kind(cv, s, x, base, hgt, wid, kind, i)
    st = {i: (x, base, hgt, wid) for (s, x, base, hgt, wid, kind, i) in stones}
    for a_, b_ in ((1, 2), (10, 11)):
        xa, ba, ha, wa = st[a_]
        xb, bb, hb, wb = st[b_]
        ytop = int(min(ba - ha, bb - hb)) - 3
        _lintel(cv, int(min(xa, xb) - wa / 2), int(max(xa, xb) + wb / 2), ytop)
    return to_layer(cv)


def _lintel(cv, x0, x1, ytop):
    cv.rect(x0, ytop, x1 - x0 + 1, 4, STONE[3])
    cv.hline(x0, x1, ytop, STONE[5])
    cv.hline(x0, x0 + 3, ytop + 1, STONE[4])
    cv.hline(x0, x1, ytop + 3, STONE[1])
    cv.frame(x0 - 1, ytop - 1, x1 - x0 + 3, 6, STONE_OUT)


def _draw_stone_kind(cv, s, x, base, hgt, wid, kind, i):
    if kind == 'stand':
        _stone(cv, x, base, hgt, wid, 700 + i, lean=(1 if i == 3 else 0))
    elif kind == 'stump':
        _stone(cv, x, base, 7, wid, 700 + i)
    else:            # fallen stone lying in the grass
        x0 = int(x - 11)
        b = int(base) + 1
        cv.round_rect(x0, b - 6, 22, 7, STONE[2], r=1)
        cv.hline(x0 + 1, x0 + 20, b - 6, STONE[5])
        cv.hline(x0 + 1, x0 + 20, b - 5, STONE[4])
        cv.hline(x0 + 1, x0 + 20, b, STONE[1])
        cv.frame(x0 - 1, b - 7, 24, 9, STONE_OUT)
        for k in range(6):
            cv.px(x0 + 1 + k * 4, b + 1, 'leaf')
            cv.px(x0 + 2 + k * 4, b, 'green')


def rune_spots():
    """(x, y, glyph) of the carved runes (knoll coords): front-facing stones + the trilithon."""
    out = []
    cx, cy = KNOLL_C
    for (s, x, base, hgt, wid, kind, i) in _stone_list():
        if kind != 'stand' or s < -0.35:
            continue
        out.append((int(round(x - 1)), int(base - hgt * 0.62), i % len(RUNES)))
    out.append((cx + 4 - 10, cy - 24, 4))
    out.append((cx + 4 + 8, cy - 24, 0))
    return out


def rune_pulse(f):
    """1 on each beat, decaying (runes pulse with the music)."""
    ph = (f - 144) % B
    return math.exp(-ph / 5.0)


def draw_runes(c, f, ox):
    k = rune_pulse(f)
    base = 0.55 + 0.45 * k
    ccol = mix('#2f7f8a', 'ice', base)
    hot = mix('ice', 'white', k)
    cx, cy = KNOLL_C
    for (x, y, g) in rune_spots():
        sx = x - ox
        if not (-4 < sx < W + 4):
            continue
        fx.glow(c, sx + 1, y + 2, 9, 'turquoise', 0.20 + 0.50 * k, rings=4)
        for r, row in enumerate(RUNES[g]):
            for q, ch in enumerate(row):
                if ch == '#':
                    c.px(sx + q + 1, y + r + 1, STONE[0])
        for r, row in enumerate(RUNES[g]):
            for q, ch in enumerate(row):
                if ch == '#':
                    c.px(sx + q, y + r, hot if (r + q + f // 6) % 5 == 0 and k > 0.4 else ccol)
    # magic ring rippling out from the altar on each beat
    ph = (f - 144) % B
    if ph < 14:
        rx = 6 + ph * 3.4
        lv = 1.0 if ph < 7 else 0.5
        sx = cx - ox
        m = np.zeros((1,), bool)
        for a in range(0, 360, 3):
            px = sx + math.cos(math.radians(a)) * rx
            py = cy + 4 + math.sin(math.radians(a)) * rx * 0.2
            if bayer(1, 1, int(px), int(py))[0, 0] < lv:
                c.px(int(px), int(py), 'ice' if ph < 7 else 'turquoise')
    # floating motes of light over the circle
    for i in range(10):
        P = 50 + int(hash01(i, 41) * 30)
        t = ((f + hash01(i, 42) * P) % P) / P
        x = cx - 40 + hash01(i, 43) * 80 + math.sin(t * 6 + i) * 3 - ox
        y = cy - 2 - t * 30
        if t < 0.85:
            c.px(int(x), int(y), 'ice' if (i + f // 5) % 3 else 'white')


# ---- foreground grass (vectorised blades)
@lru_cache(maxsize=1)
def _blades_b():
    R = rng(707)
    n = 170
    w = W + 80
    roots = np.sort(R.random(n) * w)
    hts = 10 + R.random(n) * 18
    edge = np.abs((roots - w / 2) / (w / 2))
    hts = hts * (0.7 + 0.6 * edge)                  # taller toward the frame edges
    ph = R.random(n) * 6.283
    tone = R.integers(0, 4, n)
    by = 181 - (R.random(n) * 6).astype(int)
    samples = 30
    k = np.linspace(0, 1, samples)
    return roots, hts, ph, tone, by, k


BLADE_C = np.array([col(c_) for c_ in (mix('forest', 'green', 0.4), 'green', mix('green', 'leaf', 0.5),
                                       'leaf', 'pale_leaf')], np.uint8)


def draw_grass_fg(c, f, ox, flowers=True):
    roots, hts, ph, tone, by, k = _blades_b()
    gust = 1.5 + 1.2 * math.sin(f * 0.09)
    sway = np.sin(f * 0.16 + ph + roots * 0.05) * 1.6 + gust            # tip offset in px
    X = (roots - ox)[:, None] + (sway[:, None] + 2.0) * (k[None, :] ** 2) * (hts[:, None] / 18.0)
    Y = by[:, None] - hts[:, None] * k[None, :]
    xi = np.round(X).astype(int)
    yi = np.round(Y).astype(int)
    lv = np.clip(tone[:, None] + (k[None, :] > 0.55).astype(int) - (k[None, :] < 0.2).astype(int), 0, 4)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    c.a[yi[ok], xi[ok]] = BLADE_C[lv[ok]]
    if flowers:
        for i in range(0, len(roots), 9):
            fxp, fyp = xi[i, -1], yi[i, -1]
            if 0 <= fxp < W and 0 <= fyp < H:
                cc = ('hot_pink', 'white', 'gold', 'lilac')[(i // 9) % 4]
                c.px(fxp, fyp, cc)
                c.px(fxp + 1, fyp, cc)
                c.px(fxp, fyp - 1, cc)
                c.px(fxp + 1, fyp - 1, 'butter' if cc != 'gold' else 'white')


@lru_cache(maxsize=1)
def _fg_b():
    """Foreground ground strip + lichen boulder (parallax 1.5) under the grass blades."""
    w = W + 80
    cv = new(w)
    xs = np.arange(w)
    top = 172 + 3 * np.sin(xs / 19.0) + 2 * np.sin(xs / 7.0)
    for x in range(w):
        cv.vline(x, int(top[x]), 180, mix('forest', 'green', 0.3))
        cv.px(x, int(top[x]), 'green')
    bx = 300
    blobs(cv, [(bx, 168, 16, 10, 0), (bx + 12, 172, 10, 7, -2), (bx - 12, 173, 8, 6, -2)],
          STONE, outline=STONE_OUT)
    for i in range(14):
        cv.px(bx - 12 + int(hash01(i, 51) * 26), 160 + int(hash01(i, 52) * 10),
              ('pale_sage', 'khaki', 'moss')[i % 3])
    return to_layer(cv)


def cloud_shadows(c, f, mt_mask_screen):
    """Soft shadows of drifting clouds sliding over the land."""
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W), bool)
    for (x0, y0, rx, ry, spd) in ((-60, 92, 70, 16, 0.9), (150, 140, 90, 20, 1.1), (60, 60, 46, 12, 0.7)):
        cx = x0 + (f - CUT2) * spd
        d = ((xx - cx) / rx) ** 2 + ((yy - y0) / ry) ** 2
        m |= (d < 1) & (bayer(H, W) < np.clip((1 - d) * 2.5, 0, 1))
    land = mt_mask_screen | (yy >= 104)
    darken(c, m & land, (0.80, 0.85, 0.94))


def cam_b(f):
    return int(round(lerp(0, 40, cruise(seg(f, CUT2 - 8, CUT3 + 8)))))


def render_b(f):
    cam = cam_b(f)
    c = _sky_b().copy()
    draw_clouds(c, f, [(40, 10, 52, 5, 0.12, 0), (190, 30, 34, 6, 0.16, 1), (290, 4, 44, 7, 0.10, 0)],
                ox=cam * 0.1)
    comp(c, _ridge_b(), int(cam * 0.15))
    mo = int(cam * 0.3)
    mt = _mountain_b()
    comp(c, mt, mo)
    mt_screen = np.zeros((H, W), bool)
    mm = mt[1][:H, mo:mo + W]
    mt_screen[:mm.shape[0], :mm.shape[1]] = mm
    draw_cascade(c, f, mo)
    draw_balloon(c, int(lerp(164, 190, seg(f, CUT2, CUT3))) - int(cam * 0.35), 50, f, size='s')
    mid = int(cam * 0.55)
    comp(c, _mid_b(), mid)
    draw_train(c, f, mid)
    comp(c, _knoll_b(), cam)
    cloud_shadows(c, f, mt_screen)
    draw_runes(c, f, cam)
    fo = int(cam * 1.5)
    comp(c, _fg_b(), fo)
    draw_grass_fg(c, f, fo)
    return c


# ================================================================== SHOT C: HOLLOW SHADES
HC = 212                      # layer height (camera tilts down over the extra rows)
GYC = 182                     # village ground line (layer coords)
SKY_C = ['#9aa4da', '#b4bce6', '#ccd3f0', '#dfe4f6']
FAR_C = ['#2e2c58', '#3a386a', '#4a487c', '#5e5c90']
MID_C = ['#19152e', '#221d3c', '#2d274c', '#3a335e']
WALL = ['#262650', '#33356a', '#414684', '#5258a0', '#6a70b8']
ROOF = ['#241b3e', '#33275a', '#45347a', '#5b4596', '#7660b0']
TIMBER = '#1f1a36'
WIN_DARK = '#17142a'
GROUND_C = ['#221e3a', '#2b2748', '#353158', '#423d68']
MOSS_C = ['#2f4a5a', '#3a5f6a', '#4f7a7c', '#6a9690']
STONE_C = ['#3a3a5e', '#4b4c74', '#5f618a', '#7a7ca2', '#9a9cc0']
FOG = '#8e92c6'
GLOW = 'crystal_lt'


@lru_cache(maxsize=1)
def _sky_c():
    """Deep-forest backdrop: dim violet depth, misty toward the ground, a bright canopy gap."""
    cv = Canvas(W, HC)
    fx.sky(cv, ['#2a2850', '#353362', '#46467a', '#5c5e92', '#7476a8'], 0, HC)
    yy, xx = np.mgrid[0:HC, 0:W]
    d = ((xx - 196) / 78.0) ** 2 + ((yy + 6) / 62.0) ** 2
    lvl = np.clip((1 - d) * 4, 0, 4)
    thr = bayer(HC, W)
    q = np.clip(np.floor(lvl + (thr - 0.5) * 0.9), 0, 4).astype(int)
    ramp = np.array([col(c_) for c_ in ('#000000', SKY_C[0], SKY_C[1], SKY_C[2], SKY_C[3])], np.uint8)
    m = q > 0
    cv.a[m] = ramp[q][m]
    fx.glow(cv, 196, 4, 60, 'white', 0.22, rings=5)
    return cv


def _trunk(cv, x, top, base, wdt, cols, seed, roots=True):
    """Tall tree trunk with bark grooves (cols dark->light), flared roots."""
    R = rng(seed)
    for y in range(int(top), int(base) + 1):
        flare = 0
        if roots and y > base - 8:
            flare = int((y - (base - 8)) ** 1.4 / 3)
        x0 = int(x - wdt / 2 - flare)
        x1 = int(x + wdt / 2 + flare)
        cv.hline(x0, x1, y, cols[1])
        cv.hline(x0, x0 + max(1, wdt // 4), y, cols[2])
        cv.hline(x1 - max(1, wdt // 3), x1, y, cols[0])
    for k in range(max(2, wdt // 3)):
        gx = int(x - wdt / 2 + 2 + R.random() * max(1, wdt - 3))
        y0 = int(top + R.random() * (base - top) * 0.5)
        L = int(8 + R.random() * 30)
        for y in range(y0, min(int(base), y0 + L)):
            cv.px(gx + ((y // 7) % 2), y, cols[0])


@lru_cache(maxsize=1)
def _far_c():
    """Far misty forest (pan 0.3, tilt 0.4): pale violet trunks + round canopies."""
    w = W + 20
    cv = new(w, HC)
    for i in range(15):
        x = int(i * 23 + hash01(i, 1) * 14) - 6
        top = 30 + int(hash01(i, 2) * 30)
        _trunk(cv, x, top, GYC - 6, 3 + int(hash01(i, 3) * 4), FAR_C[1:], 900 + i, roots=False)
    for i in range(34):
        x = int(i * 10 + hash01(i, 4) * 8) - 4
        y = 18 + int(hash01(i, 5) * 40)
        r = 9 + hash01(i, 6) * 8
        if 140 < x < 250 and y < 52:
            continue
        blobs(cv, canopy(x, y, r, 950 + i), [FAR_C[1], FAR_C[2], FAR_C[3], mix(FAR_C[3], 'white', 0.25)],
              outline=FAR_C[0])
    cv.rect(0, GYC - 10, w, HC - GYC + 10, FAR_C[1])
    a, m = to_layer(cv)
    a = a.astype(np.float32)
    yy = np.arange(HC)[:, None, None]
    k = 0.10 + 0.40 * np.clip((yy - 70) / 110.0, 0, 1)
    a = a + (np.array(col(FOG), np.float32) - a) * k
    return a.astype(np.uint8), m


@lru_cache(maxsize=1)
def _mid_c():
    """Mid forest (pan 0.6, tilt 0.7): dark trunks + the canopy roof with a gap above the village."""
    w = W + 30
    cv = new(w, HC)
    for i, x in enumerate((6, 44, 70, 118, 146, 244, 270, 302, 334)):
        _trunk(cv, x + int(hash01(i, 11) * 6), 0, GYC - 2, 5 + int(hash01(i, 12) * 8), MID_C, 1000 + i)
    for i in range(46):
        x = int(i * 8 + hash01(i, 13) * 8) - 6
        if 140 < x < 252:
            y = int(-16 + hash01(i, 14) * 8)
        else:
            y = int(4 + hash01(i, 14) * 34)
        r = 10 + hash01(i, 15) * 9
        blobs(cv, canopy(x, y, r, 1100 + i), [MID_C[0], MID_C[1], MID_C[2], MID_C[3], '#6f69a6'],
              outline='#211c3a')
    # hanging moss strands under the canopy
    for i in range(30):
        x = int(hash01(i, 16) * w)
        if 146 < x < 246:
            continue
        y0 = 30 + int(hash01(i, 17) * 20)
        for y in range(y0, y0 + 6 + int(hash01(i, 18) * 12)):
            cv.px(x + ((y - y0) // 5) % 2, y, MOSS_C[1] if y % 3 else MOSS_C[2])
    cv.rect(0, GYC - 4, w, HC - GYC + 4, MID_C[0])
    return to_layer(cv)


def _house(cv, x, w, wall_h, roof_h, lean=0, seed=0, dormer=False, chimney=False, spire=False):
    """A dark blue-violet cottage with a steep pointed roof (side view); x = left wall."""
    R = rng(seed)
    gy = GYC
    y0 = gy - wall_h
    # walls: lit left, shaded right, plank/stone texture
    cv.rect(x, y0, w, wall_h, WALL[2])
    cv.rect(x, y0, 2, wall_h, WALL[3])
    cv.rect(x + w - 3, y0, 3, wall_h, WALL[1])
    for yy in range(y0 + 3, gy, 4):
        for xx in range(x + 1 + (yy // 4) % 2 * 3, x + w - 1, 6):
            cv.hline(xx, xx + 3, yy, WALL[1])
    cv.rect(x, gy - 4, w, 4, WALL[0])
    # half timbering
    cv.rect(x, y0, w, 2, TIMBER)
    cv.vline(x, y0, gy - 1, TIMBER)
    cv.vline(x + w - 1, y0, gy - 1, TIMBER)
    # roof (steep, slightly crooked tip)
    ov = 3
    tipx = x + w // 2 + lean
    tipy = y0 - roof_h
    left = (x - ov, y0 + 1)
    right = (x + w + ov - 1, y0 + 1)
    cv.poly([left, (tipx, tipy), right], ROOF[2])
    cv.poly([(tipx, tipy), right, (tipx + (right[0] - tipx) // 3, y0 + 1)], ROOF[1])
    # shingle rows
    for yy in range(tipy + 4, y0 + 1, 3):
        k = (yy - tipy) / (y0 + 1 - tipy)
        xa = int(lerp(tipx, left[0], k)) + 1
        xb = int(lerp(tipx, right[0], k)) - 1
        for xx in range(xa + (yy // 3) % 2 * 2, xb, 4):
            cv.hline(xx, min(xb, xx + 2), yy, ROOF[0] if xx > tipx else mix(ROOF[2], ROOF[0], 0.6))
        cv.px(xa, yy, ROOF[3])
    cv.line(left[0], left[1], tipx, tipy, ROOF[3])
    cv.line(tipx, tipy, right[0], right[1], ROOF[0])
    cv.hline(left[0], right[0], y0 + 1, ROOF[0])
    cv.hline(left[0] + 1, right[0] - 1, y0 + 2, '#1a1530')
    # crooked finial
    cv.px(tipx, tipy - 1, ROOF[3])
    cv.px(tipx - 1, tipy - 2, ROOF[3])
    cv.px(tipx - 2, tipy - 3, ROOF[4])
    if spire:
        cv.vline(tipx, tipy - 6, tipy - 1, 'steel_dk')
        cv.px(tipx, tipy - 7, 'steel')
    if chimney:
        cx_ = x + w - 8
        cy_ = int(lerp(y0, tipy, 0.55))
        cv.rect(cx_, cy_ - 8, 5, 12, STONE_C[1])
        cv.rect(cx_, cy_ - 8, 1, 12, STONE_C[3])
        cv.rect(cx_ - 1, cy_ - 9, 7, 2, STONE_C[2])
        cv.px(cx_ + 2, cy_ - 4, STONE_C[0])
    if dormer:
        dx = x + 5
        dy = int(lerp(y0, tipy, 0.35))
        cv.rect(dx, dy, 9, 8, WALL[2])
        cv.poly([(dx - 2, dy), (dx + 4, dy - 7), (dx + 10, dy)], ROOF[2])
        cv.line(dx - 2, dy, dx + 4, dy - 7, ROOF[3])
        cv.rect(dx + 2, dy + 2, 5, 5, WIN_DARK)
        cv.vline(dx + 4, dy + 2, dy + 6, TIMBER)
        cv.hline(dx + 2, dx + 6, dy + 4, TIMBER)
    return y0, tipx, tipy


def _window(cv, x, y, w, h, boarded=False, lit=0.0):
    cv.rect(x - 1, y - 1, w + 2, h + 2, TIMBER)
    cv.rect(x, y, w, h, mix(WIN_DARK, GLOW, lit) if lit else WIN_DARK)
    if lit:
        cv.rect(x, y + h - 2, w, 2, mix(WIN_DARK, GLOW, lit * 0.6))
    cv.vline(x + w // 2, y, y + h - 1, TIMBER)
    cv.hline(x, x + w - 1, y + h // 2, TIMBER)
    cv.hline(x - 1, x + w, y + h + 1, STONE_C[3])
    if boarded:
        cv.line(x - 1, y + 1, x + w, y + h - 2, '#5a4a6a')
        cv.line(x - 1, y + h - 2, x + w, y + 1, '#4a3c5c')


def _door(cv, x, w, h):
    gy = GYC
    cv.rect(x - 1, gy - h - 1, w + 2, h + 1, TIMBER)
    cv.round_rect(x, gy - h, w, h, '#3a2a48', r=2)
    for k in range(1, w, 3):
        cv.vline(x + k, gy - h + 2, gy - 1, '#2a2038')
    cv.px(x + w - 2, gy - h // 2, 'steel')
    cv.hline(x - 1, x + w, gy, STONE_C[2])


HOUSES = {}


@lru_cache(maxsize=1)
def _village_c():
    """Village layer (pan 1.0, tilt 1.0): ground, houses, fences, path, well base, lantern posts."""
    w = W + 24
    cv = new(w, HC)
    yy, xx = np.mgrid[0:HC, 0:w]
    nz = fbm(HC, w, 7, 71)
    ground = yy >= GYC
    t = 0.30 + 0.35 * (nz - 0.5) + 0.25 * np.clip((yy - GYC) / 30.0, 0, 1)
    lvl = ramp_idx(np.clip(t, 0, 0.999), 4, 0, 0, HC, w, 0.9)
    cv.a[ground] = np.array([col(g) for g in GROUND_C], np.uint8)[lvl][ground]
    moss = ground & (vnoise(HC, w, 3, 72) > 0.70)
    mv = np.clip((nz * 3).astype(int), 0, 2)
    cv.a[moss] = np.array([col(mix(g, GROUND_C[2], 0.45)) for g in MOSS_C], np.uint8)[mv][moss]
    # cobble path to the well
    for y in range(GYC + 2, HC):
        k = (y - GYC) / (HC - GYC)
        xc = 186 + (k ** 1.3) * -40
        hw = 5 + k * 22
        for x in range(int(xc - hw), int(xc + hw)):
            if hash01(x // 3, y // 2, 5) > 0.25:
                cv.px(x, y, STONE_C[1] if (x // 3 + y // 2) % 2 else STONE_C[2])
    # house 1: tall narrow, crooked roof leaning left
    y1, t1x, t1y = _house(cv, 12, 36, 46, 58, lean=-5, seed=1)
    _door(cv, 16, 9, 17)
    _window(cv, 32, y1 + 9, 7, 9)                                 # (shutter animated)
    _window(cv, 18, y1 + 4, 5, 6, boarded=True)
    # house 2: wide, dormer + chimney, porch
    y2, t2x, t2y = _house(cv, 70, 60, 42, 54, lean=3, seed=2, dormer=True, chimney=True)
    _door(cv, 110, 10, 18)
    _window(cv, 76, y2 + 10, 9, 10, boarded=True)
    _window(cv, 94, y2 + 22, 7, 8)
    cv.poly([(104, y2 + 17), (128, y2 + 17), (131, y2 + 21), (101, y2 + 21)], ROOF[1])
    cv.hline(104, 128, y2 + 17, ROOF[3])
    cv.vline(104, y2 + 21, GYC - 1, TIMBER)
    cv.vline(127, y2 + 21, GYC - 1, TIMBER)
    for k in range(3):
        cv.rect(102 + k * 2, GYC - 2 - k * 2, 28 - k * 4, 2, '#3a2a48')
    # house 3: tower with a needle spire and a faint cold window
    y3, t3x, t3y = _house(cv, 228, 26, 68, 60, lean=2, seed=3, spire=True)
    _door(cv, 235, 9, 16)
    _window(cv, 236, y3 + 10, 7, 8, lit=0.45)
    _window(cv, 238, y3 + 36, 5, 6, boarded=True)
    cv.rect(225, y3 + 26, 32, 2, TIMBER)
    for k in range(7):
        cv.vline(226 + k * 5, y3 + 21, y3 + 25, TIMBER)
    cv.hline(225, 256, y3 + 21, TIMBER)
    # house 4: low cottage on the right
    y4, t4x, t4y = _house(cv, 278, 44, 26, 36, lean=-2, seed=4, chimney=True)
    _door(cv, 286, 9, 15)
    _window(cv, 304, y4 + 8, 7, 7, boarded=True)
    HOUSES.update(h1=(t1x, t1y, y1), h2=(t2x, t2y, y2), h3=(t3x, t3y, y3), h4=(t4x, t4y, y4))
    # broken picket fences
    for (fx0, fx1) in ((50, 68), (134, 158), (206, 224), (258, 276)):
        for x in range(fx0, fx1, 4):
            if hash01(x, 81) < 0.18:
                continue
            hgt = 7 + int(hash01(x, 82) * 3)
            tilt = 1 if hash01(x, 83) > 0.7 else 0
            for yy_ in range(GYC - hgt, GYC + 1):
                cv.px(x + (tilt if yy_ < GYC - hgt // 2 else 0), yy_, '#4a3f62')
            cv.px(x, GYC - hgt - 1, '#5c5078')
        cv.hline(fx0, fx1, GYC - 5, '#3e3456')
    # ivy on house walls
    for (ix, iy, ih) in ((12, GYC - 34, 34), (126, GYC - 30, 28), (250, GYC - 56, 50)):
        for y in range(iy, iy + ih):
            cv.px(ix + (y // 3) % 3, y, MOSS_C[(y // 2) % 3])
            if y % 4 == 0:
                cv.px(ix + 2 + (y // 3) % 2, y, MOSS_C[3])
    # grass tufts + mushrooms along the ground line
    for i in range(60):
        x = int(hash01(i, 84) * w)
        cv.px(x, GYC - 1, MOSS_C[2])
        cv.px(x + 1, GYC - 2, MOSS_C[3])
        cv.px(x - 1, GYC - 1, MOSS_C[1])
    return to_layer(cv)


WELL_X = 182                  # well centre (village layer)
WELL_Y = GYC + 10             # well base (slightly in front)


def draw_well(c, f, ox, oy):
    """The Well of Shade: stone ring, little pointed roof, crank and swaying bucket, cold glow."""
    x, y = WELL_X - ox, WELL_Y - oy
    R, WH = 17, 14
    k = 0.55 + 0.45 * math.sin(f * 0.09)
    fx.glow(c, x, y - WH - 2, 40, 'periwinkle', 0.22 + 0.16 * k, rings=5)
    fx.glow(c, x, y - WH - 1, 16, GLOW, 0.22 + 0.22 * k, rings=3)
    # back rim + glowing shaft
    c.ellipse(x, y - WH, R, 4, STONE_C[2])
    c.ellipse(x, y - WH, R - 2, 3, mix('#1c1a38', GLOW, 0.25 + 0.25 * k))
    c.ellipse(x, y - WH, R - 6, 1, mix('#1c1a38', GLOW, 0.55 + 0.40 * k))
    # posts + roof
    for px_ in (x - R + 2, x + R - 3):
        c.rect(px_, y - WH - 26, 2, 26, '#3a2c48')
        c.vline(px_, y - WH - 26, y - WH - 1, '#5a4868')
    rt = y - WH - 42
    c.poly([(x - R - 4, y - WH - 25), (x, rt), (x + R + 4, y - WH - 25)], ROOF[2])
    c.poly([(x, rt), (x + R + 4, y - WH - 25), (x + 5, y - WH - 25)], ROOF[1])
    for yy in range(rt + 4, y - WH - 25, 3):
        kk = (yy - rt) / (y - WH - 25 - rt)
        xa, xb = int(x - (R + 4) * kk) + 1, int(x + (R + 4) * kk) - 1
        for xx in range(xa + (yy // 3) % 2 * 2, xb, 4):
            c.hline(xx, min(xb, xx + 2), yy, ROOF[0])
    c.line(x - R - 4, y - WH - 25, x, rt, ROOF[4])
    c.hline(x - R - 4, x + R + 4, y - WH - 25, ROOF[0])
    c.hline(x - R - 3, x + R + 3, y - WH - 24, '#1a1530')
    c.px(x, rt - 1, ROOF[3])
    c.px(x - 1, rt - 2, ROOF[4])
    # crank axle + rope + swaying bucket
    c.hline(x - R + 3, x + R - 3, y - WH - 19, '#5a4868')
    c.vline(x + R - 1, y - WH - 19, y - WH - 15, 'steel')
    c.hline(x + R - 1, x + R + 1, y - WH - 15, 'steel')
    sw = int(round(math.sin(f * 0.11) * 1.4))
    for yy in range(y - WH - 18, y - WH - 6):
        c.px(x + int(sw * (yy - (y - WH - 18)) / 12), yy, 'tan')
    c.rect(x - 3 + sw, y - WH - 6, 7, 5, '#4a3a58')
    c.hline(x - 3 + sw, x + 3 + sw, y - WH - 6, 'steel')
    c.hline(x - 3 + sw, x + 3 + sw, y - WH - 3, '#3a2c48')
    # front wall: stone blocks, lit left
    c.rect(x - R, y - WH, 2 * R + 1, WH, STONE_C[2])
    for row in range(4):
        yy = y - WH + 1 + row * 4
        off = (row % 2) * 3
        for bx in range(x - R + off, x + R + 1, 6):
            c.hline(bx, min(x + R, bx + 4), yy, STONE_C[3] if bx < x - 6 else STONE_C[2])
            c.px(min(x + R, bx + 5), yy + 1, STONE_C[0])
            c.hline(bx, min(x + R, bx + 5), yy + 3, STONE_C[1])
    c.vline(x - R, y - WH, y - 1, STONE_C[4])
    c.rect(x + R - 4, y - WH, 5, WH, STONE_C[1])
    c.hline(x - R - 1, x + R + 1, y - WH, STONE_C[4])
    c.hline(x - R, x + R, y - WH - 1, mix(STONE_C[4], GLOW, 0.3 + 0.3 * k))
    c.hline(x - R, x + R, y, STONE_C[0])
    for i in range(6):
        c.px(x - R + 2 + i * 6, y - 3 + (i % 2), MOSS_C[2])
        c.px(x - R + 3 + i * 6, y - 2, MOSS_C[1])
    # cold mist curling out of the well
    for i in range(4):
        fx.steam(c, x - 9 + i * 6, y - WH - 2, f + i * 7, color=mix('periwinkle', 'white', 0.45), n=3,
                 height=22, spread=4, seed=40 + i, speed=0.25)
    # glowing mushrooms at the foot
    for i, (mx, my) in enumerate(((-R - 4, 0), (-R - 7, 1), (R + 3, 1), (R + 6, 0))):
        g = 0.5 + 0.5 * math.sin(f * 0.13 + i * 1.7)
        cc = mix('#5a6aa8', 'ice', g)
        c.px(x + mx, y + my - 1, 'cream')
        c.hline(x + mx - 1, x + mx + 1, y + my - 2, cc)
        c.px(x + mx, y + my - 3, cc)


def draw_shutter(c, f, ox, oy):
    """Loose shutter on house 1 swinging in the draught."""
    wx, wy = 32 - ox, HOUSES['h1'][2] + 9 - oy
    ph = math.sin(f * 0.13) * 0.9 + math.sin(f * 0.051) * 0.4
    ww = int(round(abs(math.cos(0.4 + ph)) * 5))
    hx = wx + 8                        # hinge on the window's right side
    if ww <= 0:
        c.vline(hx, wy - 1, wy + 8, '#5a4a6a')
        return
    c.rect(hx, wy - 1, ww, 10, '#4a3a5e')
    c.vline(hx + ww - 1, wy - 1, wy + 8, '#5e4e74')
    for yy in (wy + 1, wy + 4, wy + 7):
        c.hline(hx, hx + ww - 1, yy, '#3a2c4c')


def draw_sign(c, f, ox, oy):
    """Hanging inn sign (crescent-moon icon) creaking on its bracket on house 2."""
    bx, by = 132 - ox, HOUSES['h2'][2] + 6 - oy
    c.hline(bx - 1, bx + 12, by, '#2a2038')
    sw = int(round(math.sin(f * 0.10 + 0.6) * 1.5))
    x0 = bx + 2 + sw
    c.line(bx + 3, by + 1, x0 + 1, by + 4, 'grey')
    c.line(bx + 10, by + 1, x0 + 8, by + 4, 'grey')
    c.rect(x0, by + 4, 10, 8, '#4a3a58')
    c.frame(x0, by + 4, 10, 8, '#2a2038')
    c.hline(x0 + 1, x0 + 8, by + 5, '#5e4e74')
    # crescent moon icon
    for (dx, dy) in ((4, 6), (3, 7), (3, 8), (4, 9), (5, 9), (5, 6)):
        c.px(x0 + dx, by + dy, 'butter')
    c.px(x0 + 7, by + 6, 'cream')


def draw_lanterns(c, f, ox, oy):
    for i, (lx, ly) in enumerate(((58, GYC), (212, GYC), (266, GYC))):
        x, y = lx - ox, ly - oy
        c.rect(x, y - 22, 2, 22, '#2a2440')
        c.vline(x, y - 22, y - 1, '#3a3456')
        c.hline(x - 1, x + 4, y - 22, '#2a2440')
        c.vline(x + 4, y - 22, y - 20, 'grey')
        fl = hash01(i, f // 3)
        on = 0.35 + 0.35 * fl if i != 1 else 0.15 + 0.1 * fl
        c.rect(x + 3, y - 19, 3, 4, mix('#3a3a6a', 'ice', on))
        c.frame(x + 2, y - 20, 5, 6, '#2a2440')
        fx.glow(c, x + 4, y - 17, 7, 'periwinkle', 0.12 + 0.18 * on, rings=3)


@lru_cache(maxsize=1)
def _fg_c():
    """Foreground (pan 1.4, tilt 1.25): huge dark trunks framing the view, ferns, roots."""
    w = W + 40
    cv = new(w, HC + 30)
    cols = ['#120f22', '#1b1730', '#262040', '#332b52']
    _trunk(cv, 4, 0, HC + 28, 20, cols, 2001)
    _trunk(cv, 338, 0, HC + 28, 24, cols, 2002)
    # canopy masses in the top corners
    for i in range(8):
        x = (i * 10 - 10) if i < 4 else (300 + (i - 4) * 12)
        y = int(-6 + hash01(i, 91) * 12)
        blobs(cv, canopy(x, y, 13 + hash01(i, 92) * 5, 2100 + i), ['#120f22', '#1b1730', '#262040', '#332b52'],
              outline='#0e0b1a')
    # ferns at the feet
    for (fx0, base) in ((24, HC + 6), (318, HC + 8), (40, HC + 12)):
        for k in range(7):
            ang = -2.6 + k * 0.36
            L = 12 + (3 - abs(k - 3)) * 3
            for s_ in range(L):
                px_ = fx0 + int(math.cos(ang) * s_)
                py_ = base + int(math.sin(ang) * s_ + (s_ * s_) * 0.03)
                cv.px(px_, py_, '#1f3340' if s_ < L - 3 else '#2c4a54')
                if s_ % 3 == 0:
                    cv.px(px_, py_ - 1, '#2c4a54')
    return to_layer(cv)


@lru_cache(maxsize=None)
def _fog_tex(seed):
    return vnoise(64, W * 2, 18, seed) * 0.7 + vnoise(64, W * 2, 6, seed + 1) * 0.3


def draw_fog(c, f, y0, h, seed, spd, dens, color):
    """Creeping fog band: noise texture scrolled sideways, quantised alpha blend (3 soft steps)."""
    y0 = int(y0)
    Y0, Y1 = max(0, y0), min(H, y0 + h)
    if Y1 <= Y0:
        return
    tex = _fog_tex(seed)
    off = int(f * spd) % W
    rows = (np.arange(Y0, Y1) - y0) * 64 // h
    sl = tex[rows][:, off:off + W]
    prof = np.sin(np.clip((np.arange(Y0, Y1) - y0 + 0.5) / h, 0, 1) * math.pi)[:, None]
    lvl = np.clip((sl - 0.30) * 2.0, 0, 1) * prof * dens
    thr = bayer(Y1 - Y0, W, 0, Y0)
    q = np.floor(lvl * 3 + (thr - 0.5) * 0.5) / 3.0 * 0.6
    q = np.clip(q, 0, 0.6)[..., None]
    reg = c.a[Y0:Y1].astype(np.float32)
    reg += (np.array(col(color), np.float32) - reg) * q
    c.a[Y0:Y1] = reg.astype(np.uint8)


@lru_cache(maxsize=1)
def _vig_c():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H * 0.45) / (H / 2)) ** 2) / 1.3
    q = np.clip(np.floor((d - 0.55) / 0.15), 0, 3)
    return (1 - 0.12 * q)[..., None].astype(np.float32)


def draw_leaves(c, f):
    for i in range(10):
        P = 90 + int(hash01(i, 61) * 60)
        t = ((f + hash01(i, 62) * P) % P) / P
        x = int(hash01(i, 63) * W + math.sin(t * 9 + i) * 6 + t * 20)
        y = int(-4 + t * (H + 8))
        cc = ('#5a4a8a', '#6f5aa8', '#3e5a6a')[i % 3]
        c.px(x, y, cc)
        if (f // 4 + i) % 2:
            c.px(x + 1, y, cc)
        else:
            c.px(x, y + 1, cc)


def cam_c(f):
    t = cruise(seg(f, CUT3 - 8, N + 12))
    return int(round(lerp(0, 18, t))), int(round(lerp(0, 28, t)))


def render_c(f):
    cx_, cy_ = cam_c(f)
    c = Canvas()
    comp(c, to_layer(_sky_c()) if False else (_sky_c().a, np.ones((HC, W), bool)), 0, int(cy_ * 0.2))
    comp(c, _far_c(), int(cx_ * 0.3), int(cy_ * 0.4))
    balloon_y = 46 - int(cy_ * 0.4)
    draw_balloon(c, int(lerp(196, 214, seg(f, CUT3, N))) - int(cx_ * 0.3), balloon_y, f, size='s')
    draw_fog(c, f, 112 - int(cy_ * 0.4), 50, 11, 0.25, 0.9, '#8a8cc0')
    comp(c, _mid_c(), int(cx_ * 0.6), int(cy_ * 0.7))
    # dim god-rays through the canopy gap
    for i, (tx, bx, wd) in enumerate(((150, 196, 7), (176, 238, 9), (202, 276, 6))):
        s_ = 0.09 + 0.04 * math.sin(f * 0.06 + i * 2)
        ox = int(cx_ * 0.8)
        fx.light_beam(c, ((tx - ox, 0), (tx + wd - ox, 0)), ((bx - ox - 6, H), (bx + wd + 6 - ox, H)),
                      '#c8d0ff', s_)
    comp(c, _village_c(), cx_, cy_)
    draw_shutter(c, f, cx_, cy_)
    draw_sign(c, f, cx_, cy_)
    draw_lanterns(c, f, cx_, cy_)
    draw_fog(c, f, GYC - 30 - cy_, 34, 13, -0.35, 0.8, FOG)
    draw_well(c, f, cx_, cy_)
    fx.dust_motes(c, f, (120, 30, 140, 120), n=14, color='#d6dcff', seed=5)
    draw_fog(c, f, GYC - 12 - cy_, 36, 17, 0.5, 1.0, '#a8acd8')
    comp(c, _fg_c(), int(cx_ * 1.4), int(cy_ * 1.25))
    draw_leaves(c, f)
    c.a[:] = (c.a.astype(np.float32) * _vig_c()).astype(np.uint8)
    return c


# ================================================================== render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT2:
        c = render_a(f)
        cam = cam_a(f)
        callout(c, CXA - cam + 2, LIP0 - 1, 'NEIGHAGRA FALLS', f, 82, 160, side='up', length=22)
        if f < 80:
            iris(c, 1 - seg(f, 72, 79), CXA - cam, 112)
    elif f < CUT3:
        c = render_b(f)
        cam = cam_b(f)
        mo = int(cam * 0.3)
        callout(c, PEAKS[0][0] - mo + 1, PEAKS[0][1] + 6, 'FOAL MOUNTAIN', f, 184, 256, side='right', length=16)
        callout(c, KNOLL_C[0] + 4 - cam, KNOLL_C[1] - 35, 'PONHENGE', f, 188, 260, side='up', length=12)
    else:
        c = render_c(f)
        cx_, cy_ = cam_c(f)
        _village_c()
        t2x, t2y, _ = HOUSES['h2']
        callout(c, t2x - cx_, t2y - cy_ + 2, 'HOLLOW SHADES', f, 279, 351, side='right', length=14)
        callout(c, WELL_X - cx_, WELL_Y - 57 - cy_, 'WELL OF SHADE', f, 282, 354, side='up', length=8)
    atlas_card(c, f, 100, 208, 'HEARTLAND', ['FALLS - STONES - SHADES', 'EAST OF CANTERLOT'])
    return c


SFX = [
    (76, 'waterfall', 0.55),       # the falls roar in as the iris opens
    (90, 'horn', 0.20),            # distant zeppelin horn
    (114, 'waterfall', 0.50),
    (148, 'waterfall', 0.35),
    (170, 'splash', 0.40),         # the empty barrel drops into the mist
    (186, 'wind', 0.50),           # wind over the meadow (after the cut2 crash at 180)
    (192, 'chug', 0.22),           # Friendship Express heading for the tunnel
    (228, 'chug', 0.18),
    (240, 'wind', 0.35),
    (276, 'wind', 0.45),           # cold draught through Hollow Shades (after the cut3 crash at 270)
    (292, 'creak', 0.45),          # loose shutter
    (324, 'creak', 0.35),          # hanging sign
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(CUT2, 'cloud', {}), (CUT3, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
