"""SWEET APPLE ACRES: the apple farm (4 bars, 288 frames, HUD clock 08:30 -> 09:15).

Beat sheet (local frames; bar = 72, beat = 18):
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71.
  Bar 2  72-143   REVEAL (72): iris opens 72-79 on the BIG RED BARN's hay loft. Warm golden-green
                  morning, sun low upper-left; Canterlot's mountain on the far horizon, puffy clouds
                  drifting, rolling hills striped with orchard rows. Eased pan right along the orchard
                  rows (round apple trees heavy with red apples, dappled shade, ladder, crates) toward the
                  barn. ON EVERY BEAT (72..162) an apple drops from the orchard tree and lands in the
                  basket exactly on the beat, bouncing once ('pop'). The balloon drifts over the orchard.
                  Callouts SWEET APPLE ACRES (farm sign) 90-162, BIG RED BARN 108-180. Card 100-200.
  Bar 3 144-215   The pan carries on (velocity-continuous) past the clubhouse tree (treehouse, rope
                  ladder, pennant flapping), the laundry line, the two-storey farmhouse (porch, empty
                  rocking chair rocking in the breeze, flower boxes, chimney smoke), the cider press
                  (handle turning, cider dripping into a barrel), barrels, the apple cart and the windmill
                  pump turning over its trough; split-rail fences. Callout CLUBHOUSE 144-216.
  Bar 4 216-287   `zap` HIT (216): CUT with a rainbow flash to the zap-apple grove (dark twisted trunks,
                  striped apples) under a little storm cloud: rainbow lightning strikes at 216, smaller
                  crackles on beats 234/252/270. Slow pan right: the right edge darkens into the
                  Everfree treeline; Fluttershy's cottage (grass-and-thatch roof, round windows, empty
                  birdhouses) with a little footbridge over a stream at the forest edge.
                  Callouts ZAP APPLE TREES 220-279, FLUTTERSHY'S COTTAGE 238-281. Toast from 230.
Layers (parallax): sky + sun (0), clouds (0.04-0.1 + drift), Canterhorn range with Canterlot (0.08),
far hills (0.2), mid hills with orchard rows (0.42), back orchard (0.7), main farm (1.0), foreground (1.3).
Ambient: clouds, beat apples, falling leaves, grass sway, balloon, chimney smoke, rocking chair, laundry,
windmill, cider drips, pennant, Canterlot falls shimmer, pollen motes, zap crackles, stream ripples.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'sweetapple'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
GY = 150                      # main ground line (buildings stand here)
SUN = (34, 58)                # screen position of the morning sun
_L = np.array([-0.55, -0.72, 0.42])
_L = _L / np.linalg.norm(_L)  # key light: upper-left (morning sun)


def C(c):
    return np.array(col(c), np.uint8)


def hz(c, a, to=None):
    """Atmospheric haze: mix colour toward the horizon haze."""
    return mix(c, to or HAZE, a)


# ------------------------------------------------------------------ palette
HAZE = col('#d8ecdf')
SKY_COLS = ['#4b97e0', '#5fa9ec', '#7fc0f4', '#a0d6f6', '#c3e7f1', '#e4f1dc', '#fbf0c8']

LEAF = [mix('forest', 'green', 0.30), mix('green', 'forest', 0.10), mix('green', 'leaf', 0.45), 'leaf',
        mix('leaf', 'pale_leaf', 0.60)]
LEAF_OUT = mix('forest', 'dark_olive', 0.45)
TRUNK = ('brew', 'brew_mid', 'latte', 'crema')
GRASS = [mix('green', 'forest', 0.25), 'green', mix('green', 'leaf', 0.5), 'leaf', mix('leaf', 'pale_leaf', 0.5)]
G_TOP = mix('leaf', 'pale_leaf', 0.35)
G_MID = mix('leaf', 'green', 0.35)
G_LOW = mix('green', 'leaf', 0.25)
PATH = (mix('tan', 'latte', 0.4), 'tan', mix('sand', 'tan', 0.45), mix('sand', 'cream', 0.35))
WOOD = ('brew', 'brew_mid', 'latte', 'crema')
RED = (mix('maroon', 'brick', 0.3), 'brick', mix('red', 'brick', 0.35), mix('red', 'coral', 0.25))
TRIM = ('paper', mix('paper', 'stone', 0.45), mix('stone', 'grey', 0.4))
ROOF = ('maroon', 'wine', mix('rust', 'wine', 0.35), mix('rust', 'copper', 0.25))
APPLE = ('maroon', 'crimson', 'red', 'salmon')
CLOUD = [mix('#a9b8de', 'lavender', 0.15), '#c7d5f0', '#e8effc', 'white', '#fff4d8']


# ------------------------------------------------------------------ layer helpers
def new(w, h=H):
    return Canvas(w, h, KEY)


def to_layer(cv):
    a = cv.a
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    return a, m


def to_rgba(cv):
    a, m = to_layer(cv)
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = m * 255
    return out


def comp(c, layer, ox, oy=0):
    """Composite a cached (rgb, mask) layer at integer offset ox (layer x shown at screen 0)."""
    a, m = layer
    ox = int(ox)
    lw = a.shape[1]
    x0 = max(0, ox)
    x1 = min(lw, ox + W)
    if x1 <= x0:
        return
    sx0 = x0 - ox
    sl = a[:, x0:x1]
    ms = m[:, x0:x1]
    dst = c.a[:, sx0:sx0 + (x1 - x0)]
    np.copyto(dst, sl, where=ms[..., None])


def balls(cv, blobs, ramp, outline=None, dith=0.55, crease=True, light=_L, bias=0.0, gain=1.0,
          rim=None):
    """Shaded union of ellipsoid blobs [(cx, cy, rx, ry, z)]: the frontmost surface wins.
    Pseudo-3D sphere lighting quantised to `ramp` (dark->light) with ordered dither, darker creases
    where a clump overlaps another, and an outline around the union. Returns (x0, y0, mask, lvl)."""
    ramp_a = np.array([col(r) for r in ramp], np.uint8)
    n = len(ramp)
    x0 = max(0, int(math.floor(min(b[0] - b[2] for b in blobs))) - 1)
    x1 = min(cv.w, int(math.ceil(max(b[0] + b[2] for b in blobs))) + 2)
    y0 = max(0, int(math.floor(min(b[1] - b[3] for b in blobs))) - 1)
    y1 = min(cv.h, int(math.ceil(max(b[1] + b[3] for b in blobs))) + 2)
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
    for (cx, cy, rx, ry, z) in blobs:
        u = (xx - cx) / rx
        v = (yy - cy) / ry
        d2 = u * u + v * v
        ins = d2 <= 1.0
        zz = np.sqrt(np.clip(1 - d2, 0, 1))
        h = z + zz * min(rx, ry)
        upd = ins & (h > best)
        best[upd] = h[upd]
        NX[upd] = u[upd]
        NY[upd] = v[upd]
        NZ[upd] = zz[upd]
        r = min(rx, ry)
        EDGE[upd] = d2[upd] > ((r - 1.0) / r) ** 2
    m = best > -1e8
    lam = NX * light[0] + NY * light[1] + NZ * light[2]
    t = np.clip((lam * gain + 1) / 2 + bias, 0, 0.999)
    thr = bayer(y1 - y0, x1 - x0, x0, y0)
    lvl = np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    bound = m & ~inner
    if crease:
        away = (NX * light[0] + NY * light[1]) < 0.15
        cr = EDGE & ~bound & away
        lvl = np.where(cr, np.maximum(lvl - 1, 0), lvl)
    out = ramp_a[lvl]
    if outline is not None:
        oc = C(outline)
        out[bound] = oc
        if rim is not None:     # lit outline on the top-left of the silhouette
            lit = bound & ((NX * light[0] + NY * light[1]) > 0.55)
            out[lit] = C(rim)
    reg = cv.a[y0:y1, x0:x1]
    reg[m] = out[m]
    return x0, y0, m, lvl


def canopy_blobs(cx, cy, r, seed, squash=0.88):
    R = rng(seed)
    blobs = [(cx, cy, r * 0.84, r * 0.84 * squash, 0.0)]
    n = 6 + int(r / 4)
    a0 = R.random() * 6.283
    for i in range(n):
        a = a0 + 6.283 * i / n + (R.random() - 0.5) * 0.45
        rr = r * (0.34 + 0.14 * R.random())
        dist = r - rr * 0.92
        px = cx + math.cos(a) * dist
        py = cy + math.sin(a) * dist * squash
        z = (py - cy) / r * rr * 0.9 + rr * 0.15
        blobs.append((px, py, rr, rr * squash, z))
    for j in range(2 + int(r / 7)):
        a = R.random() * 6.283
        d = (0.2 + 0.4 * R.random()) * r
        rr = r * (0.28 + 0.1 * R.random())
        px = cx + math.cos(a) * d
        py = cy + math.sin(a) * d * squash + r * 0.12
        blobs.append((px, py, rr, rr * squash, r * 0.84 * 0.6 + (py - cy) * 0.3))
    return blobs


def apple_px(cv, x, y, lit=True, big=True, cols=APPLE):
    """Apple: 4x4 (big) or 2x2. (x, y) top-left."""
    d, s, m, h = (C(c) for c in cols)
    if not lit:
        d, s, m, h = C(mix(cols[0], 'ink', 0.3)), d, s, m
    a = cv.a
    if big:
        pts = [((1, 0), s), ((2, 0), s), ((0, 1), m), ((1, 1), h), ((2, 1), m), ((3, 1), s),
               ((0, 2), m), ((1, 2), m), ((2, 2), m), ((3, 2), d), ((1, 3), s), ((2, 3), d)]
    else:
        pts = [((0, 0), h), ((1, 0), m), ((0, 1), m), ((1, 1), d)]
    for (dx, dy), cc in pts:
        X, Y = x + dx, y + dy
        if 0 <= X < cv.w and 0 <= Y < cv.h:
            a[Y, X] = cc


def tree(cv, x, base, r, seed, ramp=LEAF, out=LEAF_OUT, trunk=TRUNK, apples=1.0, big_apples=True,
         trunk_h=None, tw=None, apple_cols=APPLE, apple_fn=None, squash=0.88):
    """A curly round-canopy fruit tree standing at (x, base)."""
    R = rng(seed + 7)
    tw = tw or max(3, int(r / 4.5))
    trunk_h = trunk_h if trunk_h is not None else int(r * 1.15)
    cy = base - trunk_h - int(r * 0.55)
    # trunk + fork
    o, sh, mid, hi = (C(c) for c in trunk)
    for y in range(cy, base + 1):
        t = (y - cy) / max(1, base - cy)
        hw = tw / 2 + t * 0.8 + max(0, y - (base - 3)) * 0.9
        xl = int(round(x - hw))
        xr = int(round(x + hw)) - 1
        cv.hline(xl, xr, y, tuple(mid))
        cv.px(xl + 1, y, tuple(hi))
        cv.px(xr - 1, y, tuple(sh))
        if xr - xl > 4:
            cv.px(xr - 2, y, tuple(sh))
        cv.px(xl, y, tuple(o))
        cv.px(xr, y, tuple(o))
    for k in range(3):        # bark marks
        bx = x + int((R.random() - 0.5) * tw * 0.6)
        by = cy + int(r * 0.6) + int(R.random() * max(1, base - cy - r * 0.6))
        cv.vline(bx, by, by + 2, tuple(sh))
    # branches into the canopy
    for sx, sy in ((-1, -1), (1, -1)):
        cv.line(x, cy + int(r * 0.7), x + sx * int(r * 0.45), cy + int(r * 0.2), tuple(sh))
        cv.line(x + 1, cy + int(r * 0.7), x + 1 + sx * int(r * 0.45), cy + int(r * 0.2), tuple(mid))
    res = balls(cv, canopy_blobs(x, cy, r, seed, squash), ramp, outline=out, rim=None)
    if res and apples > 0:
        x0, y0, m, lvl = res
        pad = np.pad(m, 3)
        inner = (pad[6:, 3:-3] & pad[:-6, 3:-3] & pad[3:-3, 6:] & pad[3:-3, :-6])
        cand = np.argwhere(inner)
        if len(cand):
            k = int(apples * r * r / 26)
            idx = R.permutation(len(cand))
            placed = []
            sp = 5 if big_apples else 3
            for i in idx:
                yy, xx = cand[i]
                if yy < r * 0.25 and R.random() < 0.6:
                    continue
                if any(abs(yy - py) < sp and abs(xx - px) < sp for py, px in placed):
                    continue
                placed.append((yy, xx))
                (apple_fn or apple_px)(cv, x0 + xx - 1, y0 + yy - 1, lit=lvl[yy, xx] >= 2, big=big_apples,
                                       cols=apple_cols)
                if len(placed) >= k:
                    break
    return cy


def tree_shadow(cv, x, y, r, dark, light=None, seed=0):
    """Dappled ground shadow (offset right: morning sun from the left)."""
    cv.ellipse(x + int(r * 0.35), y, int(r * 1.05), max(2, int(r * 0.16)), dark)
    if light is not None:
        R = rng(seed)
        for i in range(int(r / 2)):
            dx = int((R.random() - 0.5) * r * 1.6) + int(r * 0.35)
            dy = int((R.random() - 0.5) * r * 0.2)
            cv.px(x + dx, y + dy, light)
            cv.px(x + dx + 1, y + dy, light)


def poly_mask(pts, x0, y0, w, h):
    t = Canvas(w, h, (0, 0, 0))
    t.poly([(x - x0, y - y0) for x, y in pts], (255, 255, 255))
    return t.a[..., 0] > 0


def wavy(x, seed, amps=((5, 37.0), (3, 13.0), (1.5, 5.3))):
    return sum(a * math.sin(x / p + seed * 1.7 + i) for i, (a, p) in enumerate(amps))


# ------------------------------------------------------------------ sky
@lru_cache(maxsize=1)
def _sky():
    c = Canvas()
    c.gradient(0, 0, W, 140, SKY_COLS, soft=0.75)
    c.rect(0, 140, W, H - 140, SKY_COLS[-1])
    sx, sy = SUN
    fx.glow(c, sx, sy, 110, '#fff2b0', 0.16, rings=6)
    fx.glow(c, sx, sy, 46, '#fff6cc', 0.30, rings=5)
    fx.glow(c, sx, sy, 16, 'white', 0.45, rings=3)
    c.circle(sx, sy, 7, '#fff3c0')
    c.circle(sx, sy, 6, '#fffbe6')
    c.circle(sx - 1, sy - 1, 3, 'white')
    # soft morning sun rays fanning down-right
    for i, (ang, wdt, st) in enumerate(((0.32, 14, 0.06), (0.55, 20, 0.05), (0.80, 12, 0.05))):
        ex = sx + math.cos(ang) * 340
        ey = sy + math.sin(ang) * 340
        fx.light_beam(c, ((sx - 2, sy), (sx + 2, sy)), ((ex - wdt, ey), (ex + wdt, ey)), '#fff3c4', st)
    return c


@lru_cache(maxsize=None)
def _cloud(w, seed):
    h = int(w * 0.48) + 6
    cv = new(w + 10, h)
    R = rng(seed)
    blobs = []
    n = max(3, w // 9)
    base = h - 3
    for i in range(n):
        u = (i + 0.5) / n
        r = max(3.0, (0.38 + 0.62 * math.sin(u * math.pi)) * w * 0.23 * (0.8 + 0.4 * R.random()))
        cx = 5 + u * w
        cy = base - r * 0.75
        blobs.append((cx, cy, r, r * 0.92, R.random() * 2 + (1 - abs(u - 0.5)) * 2))
    blobs.append((5 + w / 2, base - 2, w / 2, 3.5, -3))
    balls(cv, blobs, CLOUD, outline=None, dith=0.7, light=np.array([-0.6, -0.6, 0.5]) / 0.98)
    cv.a[base + 1:] = KEY
    # flat soft shadow line at the base
    a, m = to_layer(cv)
    row = m[base]
    cv.a[base][row] = col(CLOUD[0])
    return to_rgba(cv)


CLOUDS = [  # x, y, width, seed, parallax, drift px/frame
    (-10, 14, 46, 1, 0.05, 0.10),
    (96, 30, 30, 2, 0.06, 0.12),
    (170, 8, 58, 3, 0.07, 0.08),
    (262, 36, 34, 4, 0.05, 0.11),
    (330, 20, 42, 5, 0.08, 0.09),
    (40, 46, 22, 6, 0.10, 0.14),
]


def draw_clouds(c, f, cam, tint=0.0):
    span = W + 140
    for (x, y, w, s, p, v) in CLOUDS:
        spr = _cloud(w, s)
        sx = int(x + v * f - cam * p)
        sx = (sx + 70) % span - 70
        c.blit(spr, sx, y)


# ------------------------------------------------------------------ far mountains + Canterlot (0.08)
MT_P = 0.08
MT_W = 560
CANT_X, CANT_Y = 160, 56       # Canterlot ledge on the mountain (layer coords)


@lru_cache(maxsize=1)
def _mountains():
    cv = new(MT_W)
    lit = hz('#b3bde6', 0.25)
    shd = hz('#8f9bd0', 0.25)
    snow = '#f4f6ff'
    snow_sh = '#ccd4f2'
    far = hz('#c3cdee', 0.45)
    # back range (paler)
    pts = [(0, 112)]
    for x in range(0, MT_W + 1, 4):
        y = 92 + 8 * math.sin(x / 41.0 + 1.3) + 5 * math.sin(x / 17.0) + 2 * math.sin(x / 6.1)
        pts.append((x, y))
    pts.append((MT_W, 112))
    cv.poly(pts, far)

    def mountain(px, py, hw, seed, cap=True):
        R = rng(seed)
        left, right = [], []
        for i in range(0, 41):
            t = i / 40
            jx = (R.random() - 0.5) * 3
            left.append((px - hw * t + jx, py + (112 - py) * t ** 0.9 + R.random() * 2))
            right.append((px + hw * t + jx, py + (112 - py) * t ** 0.85 + R.random() * 2))
        poly = left[::-1] + right
        cv.poly(poly, lit)
        # shaded right face: from the peak down a crooked spine
        spine = [(px + (R.random() - 0.5) * 2 + t * hw * 0.25, py + (112 - py) * t) for t in np.linspace(0, 1, 12)]
        cv.poly(spine + right[::-1], shd)
        if cap:
            cy = py + (112 - py) * 0.22
            capl = [p for p in left if p[1] <= cy]
            capr = [p for p in right if p[1] <= cy]
            jag = []
            xa, xb = capl[-1][0] if capl else px - 4, capr[-1][0] if capr else px + 4
            for k in range(9):
                u = k / 8
                jag.append((xb + (xa - xb) * u, cy + (3 if k % 2 else -1) + R.random() * 2))
            cv.poly(capl[::-1] + capr + jag, snow)
            sp = [p for p in spine if p[1] <= cy + 3]
            cv.poly(sp + capr[::-1], snow_sh)
    mountain(330, 50, 80, 5)
    mountain(470, 40, 90, 11)
    mountain(40, 60, 64, 3)
    mountain(400, 64, 50, 9, cap=False)
    mountain(150, 20, 100, 7)      # Mt. Canterhorn
    # Canterlot built out from the east face on a rocky spur: white towers, purple cones, gold tips
    x0, y0 = CANT_X, CANT_Y
    # the shoulder of the mountain the city is built on (same rock as the range, so it reads as one)
    cv.poly([(x0 - 32, y0 + 1), (x0 + 28, y0 + 1), (x0 + 44, y0 + 20), (x0 + 70, 112), (x0 - 52, 112),
             (x0 - 40, y0 + 18)], lit)
    cv.poly([(x0 + 4, y0 + 1), (x0 + 28, y0 + 1), (x0 + 44, y0 + 20), (x0 + 70, 112), (x0 + 14, 112),
             (x0 + 8, y0 + 30)], shd)
    cv.hline(x0 - 31, x0 + 27, y0 + 1, hz('#c9d0f0', 0.1))
    cv.hline(x0 - 31, x0 + 27, y0 + 2, hz('#9aa5d8', 0.2))
    for k in range(6):                                 # cliff strata under the city
        yy = y0 + 5 + k * 4
        cv.line(x0 - 30 + k * 3, yy, x0 - 14 + k * 2, yy + 1, hz('#9aa5d8', 0.25))
        cv.line(x0 + 12 + k, yy, x0 + 26 + k * 2, yy + 2, hz('#7d89c4', 0.25))
    wall = '#fbf8ff'
    wall_sh = '#cfc6ee'
    roof = 'royal_lt'
    roof_sh = mix('royal', 'royal_lt', 0.3)
    gold = 'gold'
    for dx, h_, w_ in ((-28, 3, 6), (-21, 5, 5), (-15, 4, 4), (-10, 6, 6), (4, 5, 6), (11, 4, 5), (17, 6, 6),
                       (23, 3, 4)):
        cv.rect(x0 + dx, y0 + 1 - h_, w_, h_, wall)
        cv.vline(x0 + dx + w_ - 1, y0 + 1 - h_, y0, wall_sh)
        cv.hline(x0 + dx, x0 + dx + w_ - 1, y0 - h_, roof_sh)
    towers = [(-24, 7, 2), (-17, 11, 3), (-11, 8, 2), (-6, 15, 3), (-1, 22, 4), (5, 13, 3), (10, 9, 2),
              (15, 12, 3), (21, 7, 2)]
    for dx, th, tw in towers:
        tx = x0 + dx
        cv.rect(tx, y0 - th, tw, th, wall)
        cv.vline(tx + tw - 1, y0 + 1 - th, y0, wall_sh)
        ch = 4 + tw
        cv.tri((tx - 1, y0 - th), (tx + tw, y0 - th), (tx + (tw - 1) / 2, y0 - th - ch), roof)
        cv.tri((tx + (tw - 1) / 2, y0 - th - ch), (tx + tw, y0 - th), (tx + (tw - 1) / 2 + 0.5, y0 - th), roof_sh)
        cv.px(int(tx + (tw - 1) / 2 + 0.5), y0 - 1 - th - ch, gold)
        if th > 9:
            cv.px(tx + tw // 2, y0 - th + 3, hz('royal', 0.2))
    cv.circle(x0 + 2, y0 - 3, 3, 'gold')
    cv.px(x0 + 1, y0 - 5, 'butter')
    cv.rect(x0 - 1, y0 - 2, 7, 3, wall)
    # mist at the base of the range
    a = cv.a
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    for y in range(96, 113):
        t = (y - 96) / 16
        thr = bayer(1, MT_W, 0, y)[0]
        sel = m[y] & (thr < t * 0.85)
        a[y][sel] = col(hz(HAZE, 0.0))
    return to_layer(cv)


def canterlot_falls(c, ox, f):
    """Tiny shimmering waterfalls under Canterlot's ledge (screen space)."""
    for i, (dx, y0, ln) in enumerate(((-16, 9, 20), (-2, 21, 24), (12, 13, 18))):
        x = CANT_X + dx - ox
        for y in range(CANT_Y + y0, CANT_Y + y0 + ln):
            k = (y - f // 2 + i * 2) % 5
            c.px(x, y, '#eef8ff' if k else '#a8c8f0')
        for k in range(3):        # mist puff at the foot
            if (f // 4 + k + i) % 3:
                c.px(x - 1 + k, CANT_Y + y0 + ln, '#dde6fb')


# ------------------------------------------------------------------ far hills (0.2)
FH_P = 0.2
FH_W = 480


@lru_cache(maxsize=1)
def _farhills():
    cv = new(FH_W)
    base = hz('#7fb07a', 0.55)
    lit = hz('#9cc77e', 0.55)
    dk = hz('#5f9466', 0.55)
    tree_c = hz('#4f8a5a', 0.5)
    pts = [(0, 140)]
    for x in range(0, FH_W + 1, 2):
        pts.append((x, 103 + wavy(x, 2, ((5, 47.0), (3, 19.0), (1, 7.0)))))
    pts.append((FH_W, 140))
    cv.poly(pts, base)
    # lit tops, patchwork field lines, dotted tree rows
    for x in range(FH_W):
        top = int(103 + wavy(x, 2, ((5, 47.0), (3, 19.0), (1, 7.0))))
        cv.px(x, top, lit)
        cv.px(x, top + 1, lit if x % 3 else base)
        for k, off in enumerate((6, 13, 21)):
            y = top + off + int(2 * math.sin(x / 23.0 + k))
            if (x // 3 + k) % 2 == 0:
                cv.px(x, y, tree_c)
            if (x + k * 5) % 41 < 18:
                cv.px(x, y + 2, dk)
    # Ponyville in the distance (tiny roofs) and the crystal-tree castle glint
    px0 = 236
    for i, (dx, w, rc) in enumerate(((0, 3, 'rose'), (5, 2, 'sand'), (9, 3, 'lavender'), (14, 2, 'rose'),
                                     (18, 3, 'sand'))):
        top = int(103 + wavy(px0 + dx, 2, ((5, 47.0), (3, 19.0), (1, 7.0)))) + 2
        cv.rect(px0 + dx, top - 2, w, 2, hz('cream', 0.5))
        cv.tri((px0 + dx - 1, top - 2), (px0 + dx + w, top - 2), (px0 + dx + (w - 1) / 2, top - 4), hz(rc, 0.5))
    top = int(103 + wavy(px0 + 25, 2, ((5, 47.0), (3, 19.0), (1, 7.0)))) + 2
    cv.rect(px0 + 25, top - 7, 2, 7, hz('violet', 0.5))
    cv.rect(px0 + 23, top - 5, 6, 1, hz('crystal', 0.5))
    cv.px(px0 + 26, top - 9, hz('crystal_pk', 0.4))
    return to_layer(cv)


# ------------------------------------------------------------------ mid hills with orchard rows (0.42)
MH_P = 0.42
MH_W = 660


def _mh_top(x):
    return 116 + wavy(x, 5, ((6, 53.0), (3, 23.0), (1, 9.0)))


@lru_cache(maxsize=1)
def _midhills():
    cv = new(MH_W)
    base = hz(mix('green', 'leaf', 0.5), 0.35)
    lit = hz(mix('leaf', 'pale_leaf', 0.5), 0.3)
    row_sh = hz(mix('green', 'forest', 0.2), 0.35)
    pts = [(0, 150)] + [(x, _mh_top(x)) for x in range(0, MH_W + 1, 2)] + [(MH_W, 150)]
    cv.poly(pts, base)
    cv.gradient(0, 120, MH_W, 30, [base, hz('green', 0.3)], soft=0.8)
    cv.poly([(0, 0), (0, 0)], base)
    # re-cut the top contour (gradient overwrote above it)
    for x in range(MH_W):
        t = int(_mh_top(x))
        cv.a[:t, x] = KEY
        cv.px(x, t, lit)
        cv.px(x, t + 1, lit)
    ramp = [hz('forest', 0.35), hz('green', 0.32), hz(mix('green', 'leaf', 0.5), 0.3), hz('leaf', 0.3)]
    dark = [hz('#1a3236', 0.3), hz('#22463f', 0.3), hz('#2f5e4c', 0.3), hz('#3f7458', 0.3)]
    R = rng(11)
    for row, off in enumerate((4, 10, 17, 25)):
        rr = 2.5 + row * 0.7
        step = 7 + row * 2
        x = R.random() * step
        while x < MH_W:
            top = _mh_top(x)
            y = top + off + 2 * math.sin(x / 31.0 + row)
            forest = x > 500 + row * 6
            rmp = dark if forest else ramp
            r2 = rr * (1.5 if forest else 1.0)
            cv.ellipse(x + rr * 0.6, y + rr * 0.7, rr * 1.4, 1, row_sh)
            balls(cv, [(x, y - (r2 - rr), r2, r2 * 0.9, 0)], rmp, outline=hz('forest', 0.3) if not forest
                  else hz('#132a2c', 0.3), dith=0.4, crease=False)
            if not forest and R.random() < 0.7:
                cv.px(int(x - 1 + R.random() * 2), int(y + R.random() * 2 - 1), hz('red', 0.3))
            x += (step if not forest else step * 0.6) + R.random() * 2
    # fence line along a contour
    for x in range(0, MH_W):
        y = int(_mh_top(x) + 31 + 2 * math.sin(x / 31.0 + 9))
        if y < 150:
            cv.px(x, y, hz('latte', 0.35))
            if x % 6 == 0:
                cv.vline(x, y - 2, y, hz('brew_mid', 0.35))
    return to_layer(cv)


# ------------------------------------------------------------------ back orchard (0.7)
BO_P = 0.7
BO_W = 860
BO_FOREST = 640          # layer x where the Everfree begins


def _bo_top(x):
    return 136 + wavy(x, 8, ((2, 41.0), (1, 15.0), (0.5, 6.0)))


def gnarled_tree(cv, x, base, h, r, seed, ramp, out, trunk_c, haze=0.0, vines=True):
    """Everfree-style tree: twisted trunk with roots and crooked bare branches, drooping clumps, vines."""
    R = rng(seed)
    tk = [hz(t, haze, 'ink') if haze else t for t in trunk_c]
    w = max(3, int(r / 3.2))
    top = base - h
    ph = R.random() * 6
    lean = (R.random() - 0.5) * r * 0.5
    xs = {}
    for y in range(base, top - 1, -1):
        t = (base - y) / h
        xx = x + math.sin(t * 4.2 + ph) * r * 0.2 + lean * t
        xs[y] = xx
        hw = w / 2 + (1 - t) ** 2 * 2 + max(0, 4 - (base - y)) * 1.0
        xl, xr = int(round(xx - hw)), int(round(xx + hw))
        cv.hline(xl, xr, y, tk[1])
        cv.px(xl, y, tk[2])
        cv.px(xl + 1, y, tk[2] if (y // 3) % 2 else tk[1])
        cv.px(xr, y, tk[0])
        if (y * 7 + xl) % 5 == 0:
            cv.px(xl + 2, y, tk[0])
    for k in (-1, 1):                                  # roots
        cv.line(x + k * w / 2, base - 3, x + k * (w / 2 + 7), base + 1, tk[1])
        cv.line(x + k * w / 2, base - 1, x + k * (w / 2 + 4), base + 2, tk[0])
    for k in range(3):                                 # crooked bare branches
        t = 0.55 + 0.13 * k
        y = int(base - h * t)
        xx = xs.get(y, x)
        side = -1 if (k + seed) % 2 else 1
        mx, my = xx + side * r * 0.5, y - r * 0.25 + R.random() * 4
        ex, ey = xx + side * r * (1.0 + 0.4 * R.random()), y - r * 0.55
        cv.line(xx, y, mx, my, tk[1])
        cv.line(xx, y + 1, mx, my + 1, tk[0])
        cv.line(mx, my, ex, ey, tk[0])
        cv.line(ex, ey, ex + side * 3, ey - 3, tk[0])
        cv.line(mx, my, mx + side * 2, my - 5, tk[0])
    xt = xs.get(top, x)
    blobs = []
    for i, (dx, dy, sc) in enumerate(((-0.62, 0.15, 0.62), (0.6, 0.05, 0.6), (0.0, -0.3, 0.7),
                                      ((R.random() - 0.5) * 0.8, 0.3, 0.5))):
        blobs += canopy_blobs(xt + dx * r, top - r * 0.1 + dy * r, r * sc, seed * 7 + i, squash=0.68)
    res = balls(cv, blobs, ramp, outline=out, dith=0.5)
    if vines and res:
        x0, y0, m, lvl = res
        cols = np.nonzero(m.any(axis=0))[0]
        for j in range(int(r / 3)):
            cx_ = cols[int(R.random() * len(cols))]
            rows = np.nonzero(m[:, cx_])[0]
            yb = y0 + rows[-1]
            L = 3 + int(R.random() * 10)
            cv.vline(x0 + cx_, yb, yb + L, ramp[1])
            cv.px(x0 + cx_, yb + L, ramp[2])


@lru_cache(maxsize=1)
def _backorchard():
    cv = new(BO_W)
    ramp = [hz(c, 0.22) for c in LEAF]
    out = hz(LEAF_OUT, 0.22)
    gtop = hz(G_TOP, 0.25)
    gmid = hz(G_MID, 0.22)
    for x in range(BO_W):
        t = int(_bo_top(x))
        cv.vline(x, t, H - 1, gmid)
        cv.px(x, t, gtop)
    cv.gradient(0, 140, BO_W, 12, [gmid, hz(G_LOW, 0.2)], soft=0.8)
    for x in range(BO_W):
        t = int(_bo_top(x))
        cv.a[:t, x] = KEY
        cv.px(x, t, gtop)
    R = rng(21)
    x = 6.0
    i = 0
    while x < BO_FOREST - 20:
        r = 9 + R.random() * 4
        base = int(_bo_top(x)) + 1
        tree_shadow(cv, int(x), base, r, hz(G_LOW, 0.15), seed=i)
        tree(cv, int(x), base, r, 100 + i, ramp=ramp, out=out,
             trunk=tuple(hz(t, 0.22) for t in TRUNK), apples=0.9, big_apples=False,
             apple_cols=tuple(hz(a, 0.18) for a in APPLE))
        x += 24 + R.random() * 10
        i += 1
    # Everfree edge: dark gnarled trees
    fr = [hz(c, 0.15, 'ink') for c in ('#173a37', '#1f4d45', '#2b6352', '#3d7a5c')]
    x = BO_FOREST - 10
    k = 0
    while x < BO_W + 20:
        base = int(_bo_top(x)) + 2
        gnarled_tree(cv, int(x), base, 30 + int(R.random() * 14), 13 + R.random() * 6, 300 + k, fr,
                     '#10272a', ('#1c1a26', '#2e2633', '#463a48'))
        x += 13 + R.random() * 8
        k += 1
    return to_layer(cv)


# ------------------------------------------------------------------ main farm layer (1.0)
WM = 1100
BEAT_TREE = (150, 19)          # x, canopy radius of the beat-apple tree
BASKET = (163, GY)             # basket centre-bottom
SIGN = (182, GY)
BARN_X = 198
CLUB_X = 334
LINE_X = (352, 380)            # laundry line: pole x, farmhouse porch post x
HOUSE_X = 384
HOUSE_W = 76
PRESS_X = 478
BARRELS_X = 512
CART_X = 542
MILL_X = 576
ZAP = ((704, 16, 1), (738, 19, 2), (774, 17, 3), (806, 15, 4))     # x, r, seed
STORM = (756, 50)
STREAM_X = 852
COTTAGE_X = 884
FOREST_X = 918


def _ground_top(x):
    return 143 + int(round(wavy(x, 3, ((1.5, 29.0), (1, 11.0), (0.5, 5.0)))))


def ground(cv):
    R = rng(5)
    for x in range(WM):
        t = _ground_top(x)
        cv.vline(x, t, H - 1, G_MID)
    cv.gradient(0, 141, WM, 40, [G_TOP, G_MID, G_LOW, mix(G_LOW, 'green', 0.5)], soft=0.8)
    for x in range(WM):
        cv.a[:_ground_top(x), x] = KEY
        cv.px(x, _ground_top(x), mix(G_TOP, 'pale_leaf', 0.4))
    # grass texture: tiny tufts and flowers
    for i in range(1400):
        x = int(R.random() * WM)
        y = int(147 + R.random() * 33)
        dk = mix(G_LOW, 'forest', 0.35) if y > 160 else mix(G_MID, 'green', 0.5)
        lt = mix(G_TOP, 'pale_leaf', 0.5)
        if R.random() < 0.5:
            cv.px(x, y, dk)
            cv.px(x + 1, y - 1, dk)
            cv.px(x + 2, y, dk)
        else:
            cv.px(x, y, lt)
    # the farm lane
    for x in range(WM):
        yt = 154 + int(round(1.5 * math.sin(x / 47.0) + 0.8 * math.sin(x / 13.0)))
        yb = yt + 7 + int(round(math.sin(x / 19.0 + 2)))
        cv.vline(x, yt, yb, PATH[1])
        cv.px(x, yt, PATH[3])
        cv.px(x, yt + 1, PATH[2])
        cv.px(x, yb, PATH[0])
        if x % 7 != 0:
            cv.px(x, yt + 3, PATH[2])      # wheel ruts (lit rims)
            cv.px(x, yt + 5, PATH[0])
    for i in range(160):
        x = int(R.random() * WM)
        y = 156 + int(R.random() * 5)
        cv.px(x, y, PATH[0] if R.random() < 0.6 else PATH[3])


def fence(cv, x0, x1, base, seed=0, h=12):
    """Split-rail fence: posts every 14 px, two rough rails."""
    R = rng(seed)
    o, sh, mid, hi = (C(c) for c in WOOD)
    x = x0
    posts = []
    while x <= x1:
        posts.append(x)
        x += 14
    for i, px in enumerate(posts):
        if i + 1 < len(posts):
            nx = posts[i + 1]
            for k, ry in enumerate((base - h + 3, base - h + 8)):
                sag = 1 if (k + i) % 2 else 0
                cv.line(px, ry, nx, ry + sag, tuple(mid))
                cv.line(px, ry + 1, nx, ry + 1 + sag, tuple(sh))
                cv.px(px + 3 + int(R.random() * 6), ry, tuple(hi))
    for px in posts:
        cv.rect(px - 1, base - h, 3, h + 1, tuple(mid))
        cv.vline(px - 1, base - h, base, tuple(hi))
        cv.vline(px + 1, base - h, base, tuple(sh))
        cv.hline(px - 1, px + 1, base - h, tuple(hi))
        cv.px(px + 2, base, tuple(mix(G_LOW, 'forest', 0.4)))


def basket(cv, x, base, full=0):
    """Woven apple basket, (x, base) = bottom centre. full: apples heaped (0..3 rows)."""
    w = 14
    x0 = x - w // 2
    top = base - 8
    for i, y in enumerate(range(top, base)):
        inset = 1 if y >= base - 2 else 0
        cv.hline(x0 + inset, x0 + w - 1 - inset, y, 'clay' if i % 2 else 'sand')
    for k in range(x0 + 1, x0 + w - 1, 3):
        cv.vline(k, top + 1, base - 2, 'copper')
    cv.hline(x0 - 1, x0 + w, top, 'copper')
    cv.hline(x0 - 1, x0 + w, top - 1, 'sand')
    cv.vline(x0 + w - 2, top, base - 1, 'rust')
    cv.hline(x0 + 1, x0 + w - 2, base - 1, 'rust')
    # handle
    cv.line(x0 + 2, top - 1, x0 + 4, top - 5, 'copper')
    cv.hline(x0 + 4, x0 + w - 5, top - 6, 'copper')
    cv.line(x0 + w - 5, top - 5, x0 + w - 3, top - 1, 'copper')
    return x0, top


def ladder(cv, x, base, h, lean=6):
    for k in (0, 5):
        cv.line(x + k, base, x + k + lean, base - h, 'latte')
    for y in range(base - 3, base - h, -4):
        t = (base - y) / h
        xa = x + int(lean * t)
        cv.hline(xa, xa + 5, y, 'crema')


def crate(cv, x, base, apples=True):
    cv.rect(x, base - 7, 12, 7, 'latte')
    cv.hline(x, x + 11, base - 7, 'crema')
    cv.hline(x, x + 11, base - 4, 'brew_mid')
    cv.vline(x, base - 7, base - 1, 'crema')
    cv.vline(x + 11, base - 7, base - 1, 'brew_mid')
    cv.hline(x, x + 11, base - 1, 'brew')
    if apples:
        for i in range(3):
            apple_px(cv, x + 1 + i * 4, base - 10, big=True)


def farm_sign(cv, x, base):
    """Ranch sign: post + arm with a hanging board showing an apple icon (no words)."""
    cv.rect(x - 1, base - 30, 3, 31, 'latte')
    cv.vline(x - 1, base - 30, base, 'crema')
    cv.vline(x + 1, base - 30, base, 'brew_mid')
    cv.rect(x - 2, base - 31, 16, 2, 'latte')
    cv.hline(x - 2, x + 13, base - 31, 'crema')
    cv.hline(x - 2, x + 13, base - 29, 'brew_mid')
    for cx in (x + 3, x + 11):
        cv.vline(cx, base - 29, base - 27, 'grey')
    bx, by = x + 1, base - 27
    cv.round_rect(bx, by, 14, 12, 'brew', r=1)
    cv.rect(bx + 1, by + 1, 12, 10, 'cream')
    cv.hline(bx + 1, bx + 12, by + 1, 'paper')
    cv.vline(bx + 12, by + 1, by + 10, 'tan')
    # apple icon
    cv.ellipse(bx + 6, by + 6, 3, 3, 'red')
    cv.rect(bx + 4, by + 9, 5, 1, 'crimson')
    cv.px(bx + 4, by + 5, 'salmon')
    cv.px(bx + 6, by + 2, 'brew_mid')
    cv.px(bx + 7, by + 2, 'leaf')
    cv.px(bx + 8, by + 1, 'leaf')


# ---- the BIG RED BARN
BARN_FW, BARN_WH = 72, 40
BARN_D = (26, -9)


def barn(cv, bx, gy=GY):
    FW, WH = BARN_FW, BARN_WH
    Dx, Dy = BARN_D
    ey = gy - WH
    kx, ky = 10, ey - 20
    ry = ey - 40
    xl, xr = bx, bx + FW
    xc = bx + FW // 2
    r_out, r_sh, r_mid, r_hi = RED
    # side wall (in shade)
    side = [(xr, gy + 1), (xr, ey), (xr + Dx, ey + Dy), (xr + Dx, gy + Dy + 1)]
    cv.poly(side, mix(r_sh, 'maroon', 0.25))
    for k in range(3, Dx, 4):
        y0 = ey + Dy * k / Dx
        cv.line(xr + k, y0 + 2, xr + k, gy + Dy * k / Dx, mix(r_sh, 'maroon', 0.55))
    # shadow under the eave
    cv.poly([(xr, ey), (xr + Dx, ey + Dy), (xr + Dx, ey + Dy + 3), (xr, ey + 3)], mix('maroon', 'ink', 0.2))
    # side windows (parallelograms)
    for u in (0.3, 0.72):
        wx = xr + Dx * u
        wy = ey + Dy * u + 10
        cv.poly([(wx - 3, wy), (wx + 4, wy - 2.4), (wx + 4, wy + 7.6), (wx - 3, wy + 10)], TRIM[2])
        cv.poly([(wx - 2, wy + 1), (wx + 3, wy - 0.7), (wx + 3, wy + 6.6), (wx - 2, wy + 8.3)],
                mix('navy', 'sky', 0.35))
        cv.px(int(wx), int(wy + 3), mix('light_sky', 'navy', 0.3))
    # stone foundation on the side
    cv.poly([(xr, gy - 2), (xr + Dx, gy + Dy - 2), (xr + Dx, gy + Dy + 1), (xr, gy + 1)], 'grey')
    cv.line(xr, gy - 2, xr + Dx, gy + Dy - 2, 'stone')
    # roof planes
    roof_lo = [(xr + 1, ey + 1), (xr - kx, ky), (xr - kx + Dx, ky + Dy), (xr + 1 + Dx, ey + 1 + Dy)]
    roof_up = [(xr - kx, ky), (xc, ry), (xc + Dx, ry + Dy), (xr - kx + Dx, ky + Dy)]
    cv.poly(roof_lo, ROOF[1])
    cv.poly(roof_up, ROOF[2])
    # shingle rows parallel to the ridge
    for i in range(1, 9):
        t = i / 9
        ax, ay = lerp(xr - kx, xc, t), lerp(ky, ry, t)
        cv.line(ax + 1, ay, ax + Dx, ay + Dy, ROOF[1] if i % 2 else mix(ROOF[2], ROOF[3], 0.5))
        for s in range(3, Dx, 5):
            cv.px(ax + s + (i % 2) * 2, ay + Dy * s / Dx + 1, ROOF[1])
    for i in range(1, 5):
        t = i / 5
        ax, ay = lerp(xr + 1, xr - kx, t), lerp(ey + 1, ky, t)
        cv.line(ax, ay, ax + Dx, ay + Dy, ROOF[0])
    # ridge cap (lit) and eave gutter trim
    cv.line(xc, ry - 1, xc + Dx, ry + Dy - 1, ROOF[3])
    cv.line(xc, ry, xc + Dx, ry + Dy, ROOF[3])
    cv.line(xr - kx, ky, xr - kx + Dx, ky + Dy, ROOF[0])
    cv.line(xr + 1, ey + 1, xr + 1 + Dx, ey + 1 + Dy, TRIM[1])
    cv.line(xr + Dx, ey + Dy, xr + Dx, gy + Dy, mix(r_sh, 'maroon', 0.6))
    # facade
    fac = [(xl, gy + 1), (xl, ey), (xl + kx, ky), (xc, ry), (xr - kx, ky), (xr, ey), (xr, gy + 1)]
    cv.poly(fac, r_mid)
    # boards + light gradient (warmer on the sunlit left), clipped to the facade
    fw, fh = FW + 1, gy - ry + 3
    msk = poly_mask(fac, xl, ry - 1, fw, fh)
    tmp = Canvas(fw, fh, r_mid)
    for x in range(3, fw, 4):
        tmp.vline(x, 0, fh, r_sh)
    tmp.dither(0, 0, FW // 3, fh, r_hi, 0.35, ox=xl, oy=ry - 1)
    tmp.dither(fw - 18, 0, 18, fh, r_sh, 0.4, ox=xl, oy=ry - 1)
    reg = cv.a[ry - 1:ry - 1 + fh, xl:xl + fw]
    reg[msk] = tmp.a[msk]
    # stone foundation
    cv.rect(xl, gy - 2, FW + 1, 3, 'grey')
    cv.hline(xl, xr, gy - 2, 'stone')
    for x in range(xl + 2, xr, 5):
        cv.px(x, gy - 1, 'shadow')
    # white trims: corner boards, eave band, fascia along the gambrel
    cv.rect(xl, ey, 2, gy - ey - 2, TRIM[0])
    cv.rect(xr - 1, ey, 2, gy - ey - 2, TRIM[1])
    cv.hline(xl, xr, ey, TRIM[0])
    cv.hline(xl, xr, ey + 1, r_sh)
    edges = [((xl - 2, ey + 1), (xl + kx - 1, ky)), ((xl + kx - 1, ky), (xc, ry - 1)),
             ((xc, ry - 1), (xr - kx + 1, ky)), ((xr - kx + 1, ky), (xr + 2, ey + 1))]
    for (ax, ay), (bx2, by2) in edges:
        cv.line(ax, ay, bx2, by2, TRIM[0])
        cv.line(ax, ay + 1, bx2, by2 + 1, TRIM[0])
        cv.line(ax, ay + 2, bx2, by2 + 2, r_sh)
    # big doors with white X braces
    dw, dh = 30, 34
    dx0, dy0 = xc - dw // 2, gy - dh
    cv.rect(dx0 - 2, dy0 - 3, dw + 4, dh + 1, TRIM[0])        # door frame
    cv.rect(dx0 - 2, dy0 - 3, dw + 4, 1, TRIM[0])
    cv.rect(dx0 + dw, dy0 - 2, 2, dh - 1, TRIM[1])
    for leaf in range(2):
        lx = dx0 + leaf * (dw // 2)
        lw = dw // 2
        cv.rect(lx, dy0, lw, dh - 2, r_mid)
        for x in range(lx + 2, lx + lw, 3):
            cv.vline(x, dy0, gy - 3, r_sh)
        cv.frame(lx, dy0, lw, dh - 2, TRIM[0])
        cv.frame(lx + 1, dy0 + 1, lw - 2, dh - 4, TRIM[0])
        cv.line(lx + 2, dy0 + 2, lx + lw - 3, gy - 5, TRIM[0])
        cv.line(lx + 3, dy0 + 2, lx + lw - 3, gy - 6, TRIM[0])
        cv.line(lx + lw - 3, dy0 + 2, lx + 2, gy - 5, TRIM[0])
        cv.line(lx + lw - 4, dy0 + 2, lx + 2, gy - 6, TRIM[0])
        cv.vline(lx + lw - 1, dy0, gy - 3, TRIM[1])
    cv.vline(xc, dy0, gy - 3, 'maroon')
    cv.rect(xc - 3, dy0 + 15, 2, 3, 'steel')                  # latch
    cv.rect(xc + 2, dy0 + 15, 2, 3, 'steel')
    cv.hline(dx0 - 3, dx0 + dw + 2, dy0 - 4, r_sh)           # header shadow
    # windows either side of the doors
    for wx in (xl + 6, xr - 15):
        cv.rect(wx - 1, ey + 7, 11, 12, TRIM[0])
        cv.rect(wx + 1, ey + 9, 7, 8, mix('navy', 'sky', 0.45))
        cv.rect(wx + 1, ey + 9, 7, 3, mix('sky', 'light_sky', 0.4))
        cv.px(wx + 2, ey + 10, 'glass_hi')
        cv.vline(wx + 4, ey + 9, ey + 16, TRIM[0])
        cv.hline(wx + 1, wx + 7, ey + 12, TRIM[0])
        cv.hline(wx - 2, wx + 10, ey + 19, TRIM[1])
        # flower box under each window
        cv.rect(wx - 1, ey + 20, 11, 3, 'brew_mid')
        cv.hline(wx - 1, wx + 9, ey + 20, 'latte')
        for k in range(0, 11, 2):
            cv.px(wx - 1 + k, ey + 19, ('gold', 'pink', 'white')[k % 3])
            cv.px(wx + k, ey + 19, 'green')
    # hay loft: open door, hay bale peeking out, hood + pulley beam
    lw_, lh = 16, 15
    lx0, ly0 = xc - lw_ // 2, ky - 4
    cv.rect(lx0 - 2, ly0 - 2, lw_ + 4, lh + 3, TRIM[0])
    cv.rect(lx0, ly0, lw_, lh, 'brew_dk')
    cv.rect(lx0, ly0, lw_, 3, mix('brew_dk', 'ink', 0.4))
    hb = (lx0 + 1, ly0 + lh - 8, lw_ - 1, 8)
    cv.rect(hb[0], hb[1], hb[2], hb[3], 'sand')
    cv.hline(hb[0], hb[0] + hb[2] - 1, hb[1], 'butter')
    cv.rect(hb[0], hb[1] + hb[3] - 2, hb[2], 2, 'amber')
    for k in range(hb[0] + 1, hb[0] + hb[2], 3):
        cv.vline(k, hb[1] + 1, hb[1] + 5, 'gold')
    cv.vline(hb[0] + 4, hb[1], hb[1] + hb[3] - 1, 'clay')    # twine
    cv.vline(hb[0] + 10, hb[1], hb[1] + hb[3] - 1, 'clay')
    for k, (sx, sy) in enumerate(((-1, -1), (2, -2), (6, -1), (11, -2), (14, -1))):
        cv.px(hb[0] + sx, hb[1] + sy, 'butter')
    cv.hline(lx0 - 2, lx0 + lw_ + 1, ly0 + lh + 1, TRIM[0])
    cv.hline(lx0 - 3, lx0 + lw_ + 2, ly0 + lh + 2, TRIM[1])
    # pulley beam under the ridge
    cv.rect(xc - 1, ry + 3, 3, 3, 'brew_mid')
    cv.px(xc - 1, ry + 3, 'latte')
    cv.px(xc, ry + 6, 'steel_hi')
    cv.vline(xc, ry + 7, ly0 - 3, 'tan')
    # round vent with an apple-red glint above? keep a small round window in the side gable
    # cupola on the ridge (no weathervane)
    cx_, cy_ = xc + Dx // 2, ry + Dy // 2
    cv.rect(cx_ - 5, cy_ - 9, 11, 10, TRIM[0])
    cv.rect(cx_ + 3, cy_ - 9, 3, 10, TRIM[1])
    for y in range(cy_ - 7, cy_ - 1, 2):
        cv.hline(cx_ - 3, cx_ + 2, y, mix('brew_dk', 'maroon', 0.4))
    cv.tri((cx_ - 8, cy_ - 9), (cx_ + 8, cy_ - 9), (cx_, cy_ - 16), ROOF[3])
    cv.tri((cx_, cy_ - 16), (cx_ + 8, cy_ - 9), (cx_ + 1, cy_ - 9), ROOF[1])
    cv.hline(cx_ - 8, cx_ + 8, cy_ - 9, TRIM[0])
    cv.px(cx_, cy_ - 17, 'gold')
    return xc


# ---- bar 3: clubhouse tree, farmhouse, cider press, barrels, cart, windmill
CLUB_R = 30
CLUB_PY = 106                  # treehouse platform


def clubhouse(cv, x, gy=GY):
    tree(cv, x, gy, CLUB_R, 77, apples=0.45, trunk_h=38, tw=9)
    py = CLUB_PY
    o, sh, mid, hi = (tuple(C(c)) for c in WOOD)
    cv.line(x - 16, py + 2, x - 4, py + 13, sh)
    cv.line(x - 15, py + 2, x - 3, py + 13, mid)
    cv.line(x + 16, py + 2, x + 4, py + 13, sh)
    # house: plank walls, gable roof, round window, rounded door
    hx0, hw, hh = x - 14, 28, 19
    hy0 = py - hh
    cv.rect(hx0, hy0, hw, hh, 'latte')
    for y in range(hy0 + 2, py, 3):
        cv.hline(hx0, hx0 + hw - 1, y, 'brew_mid')
        cv.hline(hx0, hx0 + hw - 1, y - 1, 'crema' if y < hy0 + 8 else 'latte')
    cv.rect(hx0 + hw - 4, hy0, 4, hh, mix('latte', 'brew_mid', 0.5))
    cv.rect(hx0 + 17, hy0 + 11, 6, 4, 'clay')           # patch board
    cv.px(hx0 + 17, hy0 + 11, 'steel_hi')
    cv.px(hx0 + 22, hy0 + 14, 'steel_hi')
    cv.vline(hx0, hy0, py - 1, 'brew')
    cv.vline(hx0 + hw - 1, hy0, py - 1, 'brew')
    ax, ay = x - 0.5, hy0 - 13
    cv.tri((hx0 - 4, hy0 + 1), (hx0 + hw + 3, hy0 + 1), (ax, ay), 'rose')
    cv.tri((ax, ay), (hx0 + hw + 3, hy0 + 1), (ax + 1, hy0 + 1), 'raspberry')
    for k in range(3, 13, 3):
        yy = ay + k
        half = (hw / 2 + 3.5) * k / 13.5
        cv.hline(int(ax - half + 1), int(ax + half), int(yy), mix('raspberry', 'berry', 0.5))
    cv.line(hx0 - 4, hy0 + 1, ax, ay, 'paper')
    cv.line(ax, ay, hx0 + hw + 3, hy0 + 1, TRIM[1])
    cv.hline(hx0 - 4, hx0 + hw + 3, hy0 + 1, 'brew')
    cv.hline(hx0, hx0 + hw - 1, hy0 + 2, 'brew_mid')
    cv.circle(ax, ay + 6, 2, 'paper')
    cv.px(int(ax), ay + 6, 'gold')                     # little star badge on the gable
    # door
    dx0 = hx0 + 4
    cv.rect(dx0, py - 12, 7, 12, 'brew_mid')
    cv.hline(dx0 + 1, dx0 + 5, py - 13, 'brew_mid')
    cv.vline(dx0 + 3, py - 12, py - 1, 'brew')
    cv.px(dx0 + 5, py - 6, 'gold')
    cv.vline(dx0 - 1, py - 12, py - 1, 'crema')
    # round window
    wx, wy = hx0 + 19, hy0 + 6
    cv.circle(wx, wy, 3, 'crema')
    cv.circle(wx, wy, 2, mix('sky', 'navy', 0.3))
    cv.px(wx - 1, wy - 1, 'light_sky')
    cv.px(wx, wy - 2, 'glass_hi')
    cv.hline(wx - 2, wx + 2, wy, 'crema')
    cv.vline(wx, wy - 2, wy + 2, 'crema')
    # platform + front railing
    cv.rect(x - 20, py, 40, 3, 'latte')
    cv.hline(x - 20, x + 19, py, 'crema')
    cv.hline(x - 20, x + 19, py + 2, 'brew_mid')
    for k in range(x - 20, x + 20, 3):
        cv.px(k, py + 1, 'brew_mid')
    cv.hline(x - 20, x - 8, py - 6, 'crema')
    cv.hline(x + 9, x + 19, py - 6, 'crema')
    for k in list(range(x - 20, x - 7, 3)) + list(range(x + 10, x + 20, 3)):
        cv.vline(k, py - 5, py - 1, 'latte')
    # flag pole (the pennant itself flaps, drawn per frame)
    cv.vline(int(ax), int(ay) - 13, int(ay) - 1, 'grey')
    cv.px(int(ax), int(ay) - 14, 'gold')
    # leaf clumps in front of the platform ends: the house sits IN the tree
    balls(cv, canopy_blobs(x - 24, py + 1, 8, 5), LEAF, outline=LEAF_OUT)
    balls(cv, canopy_blobs(x + 25, py - 3, 9, 6), LEAF, outline=LEAF_OUT)
    for (axp, ayp) in ((x - 27, py - 1), (x + 22, py - 6), (x + 27, py)):
        apple_px(cv, axp, ayp)
    # rope ladder
    for sx in (x + 7, x + 11):
        cv.vline(sx, py + 3, gy - 1, 'tan')
    for y in range(py + 6, gy - 1, 4):
        cv.hline(x + 7, x + 11, y, 'latte')
    return int(ax), int(ay)


HW = (mix('brick', 'maroon', 0.35), mix('salmon', 'brick', 0.6), mix('salmon', 'red', 0.35),
      mix('salmon', 'peach', 0.45))
HROOF = ('slate', mix('sage', 'slate', 0.45), 'sage', mix('sage', 'pale_sage', 0.55))


def window(cv, wx, wy, w=10, h=12, shutters=True, box=False, curtain='pink'):
    cv.rect(wx - 1, wy - 1, w + 2, h + 2, TRIM[0])
    cv.rect(wx, wy, w, h, mix('navy', 'sky', 0.35))
    cv.rect(wx, wy, w, h // 2, mix('sky', 'light_sky', 0.35))
    cv.line(wx, wy + h - 3, wx + 3, wy, mix('light_sky', 'white', 0.4))
    cv.px(wx + 1, wy + 1, 'glass_hi')
    cv.vline(wx, wy, wy + h - 1, curtain)
    cv.vline(wx + w - 1, wy, wy + h - 1, curtain)
    cv.vline(wx + w // 2, wy, wy + h - 1, TRIM[0])
    cv.hline(wx, wx + w - 1, wy + h // 2, TRIM[0])
    cv.hline(wx - 2, wx + w + 1, wy + h + 1, TRIM[1])
    if shutters:
        for sx in (wx - 4, wx + w + 1):
            cv.rect(sx, wy - 1, 3, h + 2, HROOF[2])
            for y in range(wy, wy + h, 2):
                cv.hline(sx, sx + 2, y, HROOF[1])
            cv.vline(sx, wy - 1, wy + h, HROOF[3])
    if box:
        cv.rect(wx - 1, wy + h + 2, w + 2, 3, 'brew_mid')
        cv.hline(wx - 1, wx + w, wy + h + 2, 'latte')
        for k in range(0, w + 2):
            cv.px(wx - 1 + k, wy + h + 1, ('green', 'leaf')[k % 2])
            if k % 2 == 0:
                cv.px(wx - 1 + k, wy + h, ('hot_pink', 'gold', 'white', 'pink')[(k // 2) % 4])


def farmhouse(cv, x0, gy=GY):
    FW = HOUSE_W
    x1 = x0 + FW
    Dx, Dy = 14, -5
    ey, fl, RH = gy - 52, gy - 27, 22
    out, sh, mid, hi = HW
    apy = ey - RH + Dy / 2
    apex = (x1 + Dx / 2, apy)
    sidec = mix(sh, 'maroon', 0.3)
    cv.poly([(x1, gy + 1), (x1, ey), (x1 + Dx, ey + Dy), (x1 + Dx, gy + Dy + 1)], sidec)
    cv.poly([(x1, ey), (x1 + Dx, ey + Dy), apex], sidec)
    for yy in range(ey + 2, gy, 3):
        cv.line(x1, yy, x1 + Dx, yy + Dy, mix(sidec, 'maroon', 0.4))
    cv.poly([(x1 + 4, ey + 9), (x1 + 9, ey + 7), (x1 + 9, ey + 16), (x1 + 4, ey + 18)], TRIM[2])
    cv.poly([(x1 + 5, ey + 10), (x1 + 8, ey + 9), (x1 + 8, ey + 15), (x1 + 5, ey + 16)], mix('navy', 'sky', 0.3))
    cv.line(apex[0], apex[1], x1 + Dx + 1, ey + Dy + 1, TRIM[1])
    cv.poly([(x1, gy - 2), (x1 + Dx, gy + Dy - 2), (x1 + Dx, gy + Dy + 1), (x1, gy + 1)], 'grey')
    # front wall: clapboards, sunlit left
    cv.rect(x0, ey, FW, gy - ey, mid)
    for yy in range(ey + 2, gy, 3):
        cv.hline(x0, x1 - 1, yy, sh)
    cv.dither(x0, ey, 22, gy - ey, hi, 0.3)
    cv.dither(x1 - 14, ey, 14, gy - ey, sh, 0.35)
    cv.rect(x0, ey, 2, gy - ey, TRIM[0])
    cv.rect(x1 - 2, ey, 2, gy - ey, TRIM[1])
    cv.hline(x0, x1 - 1, fl, TRIM[0])
    cv.hline(x0, x1 - 1, fl + 1, sh)
    # roof front plane
    rq = [(x0 - 4, ey + 1), (x1 + 3, ey + 1), (x1 + 3 + Dx / 2, apy), (x0 - 4 + Dx / 2, apy)]
    cv.poly(rq, HROOF[2])
    for i, yy in enumerate(range(int(apy) + 2, ey + 1, 3)):
        t = (yy - apy) / (ey + 1 - apy)
        xa = x0 - 4 + Dx / 2 * (1 - t)
        xb = x1 + 3 + Dx / 2 * (1 - t)
        cv.hline(int(xa) + 1, int(xb) - 1, yy, HROOF[1])
        for k in range(int(xa) + 2 + (i % 2) * 2, int(xb) - 1, 4):
            cv.px(k, yy - 1, HROOF[1])
        if yy < apy + 8:
            cv.dither(int(xa) + 1, yy - 2, int(xb - xa) - 1, 2, HROOF[3], 0.5)
    cv.line(x0 - 4 + Dx / 2, apy, x1 + 3 + Dx / 2, apy, HROOF[3])
    cv.line(x0 - 4 + Dx / 2, apy - 1, x1 + 3 + Dx / 2, apy - 1, HROOF[0])
    cv.line(x0 - 4, ey + 1, x0 - 4 + Dx / 2, apy, TRIM[0])
    cv.line(x1 + 3, ey + 1, x1 + 3 + Dx / 2, apy, TRIM[1])
    cv.hline(x0 - 4, x1 + 3, ey + 1, TRIM[0])
    cv.hline(x0 - 4, x1 + 3, ey + 2, HROOF[0])
    cv.hline(x0, x1 - 1, ey + 3, out)
    # chimney (smoke drawn per frame)
    chx = x0 + 58
    cv.rect(chx, int(apy) - 7, 7, 18, 'brick')
    for yy in range(int(apy) - 6, int(apy) + 11, 2):
        cv.hline(chx, chx + 6, yy, 'maroon')
        cv.px(chx + 2 + (yy % 4), yy - 1, 'maroon')
    cv.vline(chx, int(apy) - 7, int(apy) + 10, 'coral')
    cv.rect(chx - 1, int(apy) - 9, 9, 2, 'grey')
    cv.hline(chx - 1, chx + 7, int(apy) - 9, 'stone')
    # dormer with a round window
    dxc = x0 + 38
    cv.rect(dxc - 7, ey - 13, 15, 13, mid)
    for yy in range(ey - 11, ey, 3):
        cv.hline(dxc - 7, dxc + 7, yy, sh)
    cv.rect(dxc + 5, ey - 13, 3, 13, sh)
    cv.circle(dxc, ey - 7, 3, TRIM[0])
    cv.circle(dxc, ey - 7, 2, mix('navy', 'sky', 0.35))
    cv.px(dxc - 1, ey - 8, 'glass_hi')
    cv.tri((dxc - 10, ey - 12), (dxc + 10, ey - 12), (dxc, ey - 22), HROOF[2])
    cv.tri((dxc, ey - 22), (dxc + 10, ey - 12), (dxc + 1, ey - 12), HROOF[1])
    cv.line(dxc - 10, ey - 12, dxc, ey - 22, TRIM[0])
    cv.line(dxc, ey - 22, dxc + 10, ey - 12, TRIM[0])
    cv.hline(dxc - 7, dxc + 7, ey - 11, out)
    # upper floor windows with flower boxes
    for wx in (x0 + 8, x0 + 33, x0 + 58):
        window(cv, wx, ey + 5, box=True)
    # porch roof band + posts
    cv.poly([(x0 - 7, fl + 4), (x1 + 4, fl + 4), (x1 + 7, fl - 1), (x0 - 4, fl - 1)], HROOF[2])
    cv.hline(x0 - 5, x1 + 5, fl, HROOF[3])
    cv.hline(x0 - 6, x1 + 5, fl + 2, HROOF[1])
    cv.rect(x0 - 7, fl + 4, FW + 12, 2, TRIM[0])
    cv.hline(x0 - 7, x1 + 4, fl + 6, TRIM[1])
    cv.dither(x0, fl + 6, FW, 5, sh, 0.55)                   # shade under the porch roof
    # ground floor: door + windows
    window(cv, x0 + 8, fl + 10, h=11)
    window(cv, x0 + 54, fl + 10, h=11)
    dx0 = x0 + 29
    cv.rect(dx0 - 2, fl + 7, 14, gy - fl - 9, TRIM[0])
    cv.rect(dx0, fl + 9, 10, gy - fl - 11, 'brew_mid')
    cv.rect(dx0 + 2, fl + 11, 6, 5, mix('navy', 'sky', 0.4))
    cv.px(dx0 + 2, fl + 11, 'glass_hi')
    cv.hline(dx0 + 1, dx0 + 8, fl + 18, 'brew')
    cv.vline(dx0 + 9, fl + 9, gy - 3, 'brew')
    cv.px(dx0 + 8, fl + 15, 'gold')
    # deck, lattice skirt, steps
    cv.rect(x0 - 7, gy - 5, FW + 12, 2, 'latte')
    cv.hline(x0 - 7, x1 + 4, gy - 5, 'crema')
    cv.rect(x0 - 7, gy - 3, FW + 12, 4, 'brew_mid')
    for k in range(x0 - 7, x1 + 5, 2):
        cv.px(k, gy - 2 + (k // 2) % 2, 'crema')
    for i in range(3):
        cv.rect(x0 - 15 + i * 3, gy - 2 - i * 1, 9 - i * 3, 2, 'latte')
        cv.hline(x0 - 15 + i * 3, x0 - 8, gy - 2 - i, 'crema')
    for px in (x0 - 5, x0 + 22, x0 + 48, x1 + 2):
        cv.rect(px, fl + 6, 2, gy - fl - 11, TRIM[0])
        cv.vline(px + 1, fl + 6, gy - 6, TRIM[1])
    # left bay railing (the right bay's railing is drawn per frame, in front of the rocking chair)
    cv.hline(x0 - 4, x0 + 21, gy - 15, TRIM[0])
    cv.hline(x0 - 4, x0 + 21, gy - 7, TRIM[1])
    for k in range(x0 - 2, x0 + 21, 3):
        cv.vline(k, gy - 14, gy - 8, TRIM[1])
    # potted plant + lantern by the door
    cv.rect(x0 + 43, gy - 10, 4, 4, 'clay')
    cv.hline(x0 + 43, x0 + 46, gy - 10, 'sand')
    for k, (dx, dy) in enumerate(((0, -12), (1, -13), (2, -12), (3, -13), (1, -11), (2, -14))):
        cv.px(x0 + 43 + dx, gy + dy, ('green', 'leaf')[k % 2])
    cv.px(x0 + 44, gy - 15, 'hot_pink')
    cv.rect(dx0 + 13, fl + 10, 3, 4, 'brass')
    cv.px(dx0 + 14, fl + 11, 'butter')
    cv.px(dx0 + 14, fl + 9, 'grey')


def chair_rail(c, x0, f):
    """Empty rocking chair rocking in the breeze + the porch railing in front of it (screen space)."""
    cx, by = x0 + 62, GY - 5
    a = 0.16 * math.sin(2 * math.pi * f / 36.0)
    ca, sa = math.cos(a), math.sin(a)

    def P(px, py):          # local -> screen, rotate about the rocker's contact point
        return cx + px * ca - py * sa, by + px * sa + py * ca

    def L(p, q, cc):
        (x0_, y0_), (x1_, y1_) = P(*p), P(*q)
        c.line(int(round(x0_)), int(round(y0_)), int(round(x1_)), int(round(y1_)), cc)
    # rocker (curved runner)
    pts = [(-8, -2), (-5, -1), (-2, 0), (2, 0), (5, -1), (8, -2)]
    for p, q in zip(pts, pts[1:]):
        L(p, q, 'brew_mid')
    L((-5, -1), (-5, -8), 'brew_mid')
    L((4, -1), (4, -8), 'brew_mid')
    L((-6, -8), (5, -8), 'latte')
    L((-6, -9), (5, -9), 'crema')
    L((4, -9), (6, -21), 'latte')
    L((5, -9), (7, -21), 'brew_mid')
    L((-1, -9), (1, -21), 'latte')
    L((-2, -9), (0, -21), 'brew_mid')
    L((2, -9), (3, -20), 'brew_mid')
    L((1, -21), (7, -21), 'crema')
    L((2, -14), (6, -14), 'latte')
    L((-5, -13), (4, -13), 'latte')
    L((-5, -13), (-5, -9), 'latte')
    # railing for the right bay, in front of the chair
    xa, xb = x0 + 50, x0 + HOUSE_W + 1
    c.hline(xa, xb, GY - 15, TRIM[0])
    c.hline(xa, xb, GY - 7, TRIM[1])
    for k in range(xa + 1, xb, 3):
        c.vline(k, GY - 14, GY - 8, TRIM[1])


def cider_press(cv, x, gy=GY):
    o, sh, mid, hi = WOOD
    # crate of apples feeding the press
    crate(cv, x - 14, gy)
    crate(cv, x - 12, gy - 7)
    # table
    cv.rect(x, 133, 28, 3, mid)
    cv.hline(x, x + 27, 133, hi)
    cv.hline(x, x + 27, 135, sh)
    for lx in (x + 2, x + 24):
        cv.rect(lx, 136, 2, gy - 136, sh)
        cv.vline(lx, 136, gy - 1, mid)
    cv.line(x + 3, 146, x + 24, 141, sh)
    # frame posts + crossbeam
    for px in (x + 4, x + 21):
        cv.rect(px, 104, 3, 29, mid)
        cv.vline(px, 104, 132, hi)
        cv.vline(px + 2, 104, 132, sh)
    cv.rect(x + 2, 102, 24, 4, mid)
    cv.hline(x + 2, x + 25, 102, hi)
    cv.hline(x + 2, x + 25, 105, o)
    # screw + pressing plate
    cv.rect(x + 13, 106, 2, 11, 'steel')
    cv.vline(x + 13, 106, 116, 'steel_hi')
    cv.rect(x + 7, 117, 14, 2, sh)
    cv.hline(x + 7, x + 20, 117, mid)
    # slatted tub with apple mash
    cv.rect(x + 7, 119, 14, 13, mid)
    for k in range(x + 8, x + 21, 3):
        cv.vline(k, 119, 131, sh)
    cv.vline(x + 7, 119, 131, hi)
    for hy in (121, 129):
        cv.hline(x + 7, x + 20, hy, 'steel_dk')
        cv.px(x + 8, hy, 'steel_hi')
    # tray + spout
    cv.rect(x + 5, 131, 19, 2, 'brew')
    cv.rect(x + 24, 131, 6, 2, mid)
    cv.hline(x + 24, x + 29, 131, hi)
    cv.px(x + 29, 133, 'amber')
    # bucket catching cider
    bx = x + 26
    cv.rect(bx, gy - 7, 8, 7, 'steel')
    cv.vline(bx, gy - 7, gy - 1, 'steel_hi')
    cv.vline(bx + 7, gy - 7, gy - 1, 'steel_dk')
    cv.hline(bx, bx + 7, gy - 7, 'steel_hi')
    cv.hline(bx + 1, bx + 6, gy - 6, 'amber')
    cv.hline(bx + 1, bx + 6, gy - 5, 'clay')


def press_dynamic(c, x, f):
    """Turning T-handle on top of the screw + cider drips (screen space)."""
    a = f * 2 * math.pi / 72.0
    L = int(round(9 * math.cos(a)))
    cx, cy = x + 13, 100
    c.vline(cx, cy, cy + 1, 'steel_dk')
    c.vline(cx + 1, cy, cy + 1, 'steel')
    c.hline(cx - abs(L), cx + 1 + abs(L), cy - 1, 'latte' if math.sin(a) > 0 else 'brew_mid')
    c.hline(cx - abs(L), cx + 1 + abs(L), cy - 2, 'crema' if math.sin(a) > 0 else 'latte')
    for e in (cx - abs(L) - 1, cx + abs(L) + 2):
        c.px(e, cy - 2, 'brew')
        c.px(e, cy - 1, 'brew')
    # screw thread marks move as it turns
    for k in range(3):
        y = 107 + (k * 4 + f // 3) % 10
        c.px(x + 14, y, 'steel_dk')
    # drips: spout (x+29, 133) -> bucket surface (GY-6)
    for k in range(3):
        t = ((f + k * 4) % 12) / 12.0
        y = 134 + int((GY - 7 - 134) * t * t)
        if t < 0.9:
            c.px(x + 29, y, 'amber' if k % 2 else 'gold')
    if (f // 3) % 3 == 0:
        c.px(x + 29, GY - 7, 'butter')
        c.px(x + 28, GY - 7, 'gold')


def barrel(cv, x, base, w=12, h=14, emblem=False):
    o, sh, mid, hi = WOOD
    for y in range(base - h, base):
        t = (y - (base - h)) / (h - 1)
        hw = w / 2 * (0.84 + 0.16 * math.sin(math.pi * t))
        xl, xr = int(round(x - hw)), int(round(x + hw)) - 1
        cv.hline(xl, xr, y, mid)
        cv.px(xl + 1, y, hi)
        cv.px(xl + 2, y, hi)
        cv.px(xr - 1, y, sh)
        cv.px(xr - 2, y, sh)
        cv.px(xl, y, o)
        cv.px(xr, y, o)
        for k in range(xl + 4, xr - 2, 3):
            cv.px(k, y, sh)
    for t in (0.18, 0.8):
        y = int(base - h + t * (h - 1))
        hw = w / 2 * (0.84 + 0.16 * math.sin(math.pi * t))
        cv.hline(int(round(x - hw)), int(round(x + hw)) - 1, y, 'steel_dk')
        cv.px(int(round(x - hw)) + 1, y, 'steel_hi')
    cv.hline(int(x - w / 2 * 0.84) + 1, int(x + w / 2 * 0.84) - 2, base - h, hi)
    if emblem:
        cx, cy = int(x), base - h // 2
        cv.ellipse(cx, cy, 2, 2, 'red')
        cv.px(cx - 1, cy - 1, 'salmon')
        cv.px(cx, cy - 3, 'leaf')


def apple_cart(cv, x, gy=GY):
    o, sh, mid, hi = WOOD
    # shafts resting on the ground (no animal: parked)
    cv.line(x + 2, 134, x - 14, gy - 1, sh)
    cv.line(x + 2, 135, x - 13, gy, o)
    # bed
    cv.rect(x, 126, 30, 10, mid)
    cv.hline(x, x + 29, 126, hi)
    cv.hline(x, x + 29, 130, sh)
    cv.hline(x, x + 29, 135, o)
    for k in (x, x + 10, x + 20, x + 29):
        cv.vline(k, 126, 135, sh)
    cv.vline(x, 126, 135, hi)
    # heaped apples
    for row, (y, n, off) in enumerate(((121, 7, 1), (118, 6, 3), (115, 4, 7), (112, 2, 11))):
        for i in range(n):
            apple_px(cv, x + off + i * 4, y, lit=row > 0 or i < 4)
    # rear leg + wheel
    cv.rect(x + 26, 136, 2, gy - 136, sh)
    wx, wy, r = x + 15, gy - 8, 8
    cv.circle(wx, wy, r, o, fill=False)
    cv.circle(wx, wy, r - 1, sh, fill=False)
    for k in range(8):
        a = k * math.pi / 4
        cv.line(wx, wy, wx + math.cos(a) * (r - 1), wy + math.sin(a) * (r - 1), mid)
    cv.circle(wx, wy, 1, o)
    cv.px(wx - 1, wy - 1, hi)
    for a in np.linspace(3.4, 4.6, 6):
        cv.px(int(round(wx + math.cos(a) * r)), int(round(wy + math.sin(a) * r)), hi)


def windmill(cv, x, gy=GY):
    o, sh, mid, hi = WOOD
    top = 98
    for (bx, tx, cc) in ((-6, -1, mix('latte', 'tan', 0.5)), (6, 1, mix('latte', 'tan', 0.5)),
                         (-9, -2, mid), (9, 2, sh)):
        cv.line(x + bx, gy, x + tx, top, cc)
    levels = [gy, 136, 123, 111, top]
    for ya, yb in zip(levels, levels[1:]):
        def lx(y):
            return x - 9 + 7 * (gy - y) / (gy - top)

        def rx(y):
            return x + 9 - 7 * (gy - y) / (gy - top)
        cv.line(lx(ya), ya, rx(yb), yb, sh)
        cv.line(rx(ya), ya, lx(yb), yb, sh)
        cv.hline(int(lx(yb)), int(rx(yb)), yb, mid)
    cv.rect(x - 3, top - 2, 7, 3, mid)
    cv.hline(x - 3, x + 3, top - 2, hi)
    # tail boom + vane
    cv.line(x + 1, top - 5, x + 15, top - 6, 'steel')
    cv.rect(x + 13, top - 12, 9, 9, 'paper')
    cv.rect(x + 13, top - 9, 9, 3, mix('red', 'brick', 0.4))
    cv.frame(x + 13, top - 12, 9, 9, TRIM[2])
    # pump rod, pump head, spout over the trough
    cv.vline(x, top + 1, gy - 12, 'steel_dk')
    cv.rect(x - 2, gy - 13, 5, 10, 'steel')
    cv.vline(x - 2, gy - 13, gy - 4, 'steel_hi')
    cv.vline(x + 2, gy - 13, gy - 4, 'steel_dk')
    cv.rect(x + 3, gy - 12, 5, 2, 'steel')
    cv.px(x + 7, gy - 10, 'steel_dk')
    cv.rect(x - 3, gy - 3, 7, 3, 'grey')
    # trough
    tx0 = x + 5
    cv.rect(tx0, gy - 8, 20, 8, mid)
    cv.hline(tx0, tx0 + 19, gy - 8, hi)
    cv.rect(tx0 + 1, gy - 7, 18, 2, mix('sky', 'teal', 0.3))
    cv.hline(tx0 + 1, tx0 + 18, gy - 4, sh)
    cv.vline(tx0, gy - 8, gy - 1, hi)
    cv.vline(tx0 + 19, gy - 8, gy - 1, sh)


def windmill_dynamic(c, x, f):
    """Spinning wheel (16 blades) + water from the pump (screen space)."""
    hx, hy = x, 93
    a0 = f * 0.11
    for k in range(16):
        a = a0 + k * 2 * math.pi / 16
        ca, sa = math.cos(a), math.sin(a)
        cc = 'paper' if k % 2 else mix('red', 'brick', 0.3)
        c.line(int(round(hx + ca * 3)), int(round(hy + sa * 3)), int(round(hx + ca * 13)),
               int(round(hy + sa * 13)), cc)
    for a in np.linspace(0, 2 * math.pi, 40, endpoint=False):
        c.px(int(round(hx + math.cos(a) * 13)), int(round(hy + math.sin(a) * 13)), 'steel_dk')
    c.circle(hx, hy, 2, 'steel_dk')
    c.px(hx - 1, hy - 1, 'steel_hi')
    # water pulse from the spout into the trough
    ph = (f % 24)
    if ph < 14:
        for y in range(GY - 10, GY - 6):
            c.px(x + 8, y, 'light_sky' if (y + f) % 3 else 'white')
        c.px(x + 7 + (f // 2) % 3, GY - 7, 'white')
    c.px(x + 10 + (f // 5) % 12, GY - 7, mix('light_sky', 'white', 0.5))


# ---- bar 4: zap-apple grove, storm cloud, stream + footbridge, Fluttershy's cottage, Everfree edge
ZAP_LEAF = ['#191630', '#22264a', '#2a3f58', '#33605e', '#4a8474']
ZAP_OUT = '#0f0d1e'
ZAP_TRUNK = ('#16121c', '#2a2232', '#433649', '#5d4c60')
ZAP_STRIPES = ('magic', 'butter', 'ice', 'lavender')
STORM_RAMP = ['#2c2a44', '#3d3a5a', '#575478', '#7a779c', '#9a98b8']


def zap_apple(cv, x, y, lit=True, big=True, cols=None):
    sh = 1.0 if lit else 0.7
    shape = ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3))
    for dx, dy in shape:
        cc = ZAP_STRIPES[(dx + dy) % 3] if dx != 3 else 'violet'
        if dy == 3:
            cc = mix(cc, 'purple', 0.4)
        X, Y = x + dx, y + dy
        if 0 <= X < cv.w and 0 <= Y < cv.h:
            cv.a[Y, X] = col(mix(cc, 'navy', 1 - sh)) if sh < 1 else col(cc)
    if 0 <= x + 1 < cv.w and 0 <= y < cv.h:
        cv.a[y, x + 1] = col('white')


def zap_canopy(i):
    x, r, s = ZAP[i]
    trunk_h = int(r * 1.15)
    return x, GY - trunk_h - int(r * 0.55), r


def storm_cloud(cv):
    sx, sy = STORM
    puffs = [(-30, -6, 11), (-14, -14, 15), (5, -18, 16), (22, -11, 13), (36, -4, 9), (-42, -2, 7),
             (-22, -2, 10), (14, -3, 11), (-4, -6, 12)]
    blobs = [(sx + dx, sy + dy, r, r * 0.88, 4 + (dy + 18) * 0.25 + i * 0.3) for i, (dx, dy, r) in enumerate(puffs)]
    blobs.append((sx - 2, sy - 2, 48, 5, -2))
    balls(cv, blobs, STORM_RAMP, outline='#1d1a30', dith=0.6, rim='#8f8cb0', gain=1.25)
    cv.a[sy + 2:sy + 14, sx - 64:sx + 66] = KEY
    for x in range(sx - 44, sx + 44):                 # ragged dark underside
        if hash01(x, 5) < 0.45:
            cv.px(x, sy + 1, '#1d1a30')


def _stream_geo(y):
    t = (y - 141) / (H - 141)
    cx = STREAM_X + 14 - 30 * t + 2 * math.sin(y / 6.0)
    hw = 2.5 + t * 15
    return cx, hw


def stream(cv):
    """A brook winding out of the woods toward the viewer, under the footbridge."""
    for y in range(141, H):
        t = (y - 141) / (H - 141)
        cx, hw = _stream_geo(y)
        xl, xr = int(round(cx - hw)), int(round(cx + hw))
        cv.hline(xl - 2, xr + 2, y, 'brew_mid')
        cv.px(xl - 2, y, mix('latte', 'green', 0.5))
        cv.px(xr + 2, y, mix('brew_mid', 'brew', 0.5))
        cv.hline(xl, xr, y, mix('sky', 'teal', 0.4 - 0.25 * t))
        cv.hline(xl, xl + 1 + int(t * 3), y, mix('teal', 'sky', 0.35))
        cv.hline(xr - 1 - int(t * 2), xr, y, mix('light_sky', 'sky', 0.4))
    R = rng(44)
    for i in range(16):
        y = 145 + int(R.random() * 35)
        cx, hw = _stream_geo(y)
        t = (y - 141) / (H - 141)
        side = -1 if i % 2 else 1
        x = int(cx + side * (hw + 2 + R.random() * 2))
        w = 3 + int(R.random() * 3 + t * 3)
        cv.round_rect(x - w // 2, y - 1, w, 3, 'grey', r=1)
        cv.hline(x - w // 2 + 1, x + w // 2 - 1, y - 1, 'stone')


def stream_dynamic(c, f, cam):
    for i in range(14):
        ph = (f * 0.45 + i * 2.9) % 40
        y = int(142 + ph)
        if y >= H:
            continue
        cx, hw = _stream_geo(y)
        off = int((hash01(i, 7) - 0.5) * hw * 1.3)
        ln = 1 + int((y - 141) / 12)
        X = int(cx) - cam + off
        c.hline(X, X + ln, y, 'white' if i % 3 else 'light_sky')


def footbridge(cv, x0=STREAM_X - 18, x1=STREAM_X + 18, base=155):
    o, sh, mid, hi = WOOD
    n = x1 - x0
    for x in range(x0, x1 + 1):
        u = (x - x0) / n
        y = base - int(round(6 * math.sin(math.pi * u)))
        cv.vline(x, y, y + 2, mid)
        cv.px(x, y, hi)
        cv.px(x, y + 2, o)
        if (x - x0) % 3 == 0:
            cv.px(x, y + 1, sh)
        if x in (x0, x0 + n // 2, x1) or x in (x0 + n // 4, x0 + 3 * n // 4):
            cv.vline(x, y - 7, y - 1, mid)
            cv.px(x, y - 8, hi)
        yr = y - 7
        cv.px(x, yr, hi)
        cv.px(x, yr + 1, sh)
        if x % 4 == 0:
            cv.vline(x, yr + 2, y - 1, sh)
    # posts into the water
    for x in (x0 + 5, x1 - 5):
        u = (x - x0) / n
        y = base - int(round(6 * math.sin(math.pi * u)))
        cv.vline(x, y + 3, base + 6, o)


def birdhouse(cv, x, y, pole=True, color='sand', roof='rose'):
    """Little EMPTY birdhouse: box, roof, round hole. (x, y) = box top-left."""
    if pole:
        cv.vline(x + 3, y + 7, GY - 1, 'brew_mid')
        cv.vline(x + 2, y + 7, GY - 1, 'latte')
    cv.rect(x, y + 2, 7, 6, color)
    cv.vline(x + 6, y + 2, y + 7, mix(color, 'brew_mid', 0.4))
    cv.tri((x - 1, y + 2), (x + 7, y + 2), (x + 3, y - 2), roof)
    cv.hline(x - 1, x + 7, y + 2, mix(roof, 'ink', 0.3))
    cv.px(x + 3, y + 4, 'brew_dk')
    cv.px(x + 3, y + 6, 'brew_mid')        # perch peg


def cottage(cv, x0, gy=GY):
    """Fluttershy's cottage: stone walls, round windows, thatch eaves under a grassy roof."""
    w = 50
    x1 = x0 + w
    wt = gy - 24
    cv.rect(x0, wt, w, 24, 'cream')
    R = rng(55)
    for i in range(46):                                    # stone texture
        sx = x0 + 1 + int(R.random() * (w - 4))
        sy = wt + 1 + int(R.random() * 21)
        cv.hline(sx, sx + 2 + int(R.random() * 2), sy, mix('tan', 'cream', 0.4))
        cv.px(sx, sy - 1, 'paper')
    cv.dither(x1 - 14, wt, 14, 24, mix('tan', 'cream', 0.3), 0.55)
    cv.vline(x1 - 1, wt, gy - 1, 'tan')
    cv.rect(x0, gy - 3, w, 3, mix('tan', 'grey', 0.4))
    # round windows
    for wx in (x0 + 10, x1 - 11):
        cv.circle(wx, wt + 10, 5, 'latte')
        cv.circle(wx, wt + 10, 4, mix('navy', 'sky', 0.35))
        cv.circle(wx - 1, wt + 9, 2, mix('sky', 'light_sky', 0.5))
        cv.px(wx - 2, wt + 8, 'glass_hi')
        cv.hline(wx - 4, wx + 4, wt + 10, 'latte')
        cv.vline(wx, wt + 6, wt + 14, 'latte')
        cv.rect(wx - 5, wt + 16, 11, 2, 'brew_mid')        # flower box
        for k in range(0, 11, 2):
            cv.px(wx - 5 + k, wt + 15, ('pink', 'gold', 'lilac', 'white')[(k // 2) % 4])
            cv.px(wx - 4 + k, wt + 15, 'leaf')
    # door (rounded top)
    dx = x0 + w // 2 - 5
    cv.rect(dx - 1, wt + 6, 12, 18, 'latte')
    cv.rect(dx, wt + 8, 10, 16, mix('clay', 'brew_mid', 0.5))
    cv.hline(dx + 1, dx + 8, wt + 7, mix('clay', 'brew_mid', 0.5))
    cv.hline(dx + 2, dx + 7, wt + 6, 'latte')
    for k in range(dx + 2, dx + 10, 3):
        cv.vline(k, wt + 9, gy - 2, 'brew_mid')
    cv.px(dx + 8, wt + 16, 'gold')
    cv.rect(dx - 2, gy - 1, 14, 2, 'grey')
    cv.hline(dx - 2, dx + 11, gy - 1, 'stone')
    # thatch-and-grass roof: grassy mound with a straw fringe at the eaves
    rcx, ry = x0 + w / 2, wt + 2
    blobs = [(rcx, ry, w / 2 + 9, 24, 0)]
    for i, (dx_, dy_, rr) in enumerate(((-18, -14, 10), (-4, -20, 11), (12, -16, 10), (24, -6, 8),
                                        (-27, -4, 8), (3, -8, 12))):
        blobs.append((rcx + dx_, ry + dy_, rr, rr * 0.85, 6 + i))
    tmp = new(w + 40, 60)
    ox, oy = x0 - 20, ry - 46
    balls(tmp, [(bx - ox, by - oy, a, b, z) for bx, by, a, b, z in blobs], GRASS,
          outline=mix('forest', 'dark_olive', 0.3), dith=0.5)
    tmp.a[46 + 1:] = KEY
    cv.blit(to_rgba(tmp), ox, oy)
    # straw fringe
    for x in range(x0 - 9, x1 + 10):
        u = (x - rcx) / (w / 2 + 9)
        if abs(u) > 1:
            continue
        yb = ry + 1 + int(round(2 * (1 - abs(u))))
        L = 3 + (x % 3 == 0)
        cv.vline(x, yb - 3, yb - 3 + L, 'sand' if x % 2 else 'clay')
        cv.px(x, yb - 3 + L, 'copper' if x % 4 == 0 else 'amber')
    # grass tufts and flowers on the roof
    for i in range(16):
        u = (R.random() - 0.5) * 1.6
        x = int(rcx + u * (w / 2 + 6))
        a_ = abs(u)
        ytop = int(ry - 20 * math.sqrt(max(0, 1 - a_ * a_)) - 2)
        cv.px(x, ytop, ('pink', 'gold', 'white', 'lilac')[i % 4])
        cv.px(x + 1, ytop + 1, 'leaf')
    # round attic window in the roof
    cv.circle(int(rcx), ry - 12, 3, 'latte')
    cv.circle(int(rcx), ry - 12, 2, mix('navy', 'sky', 0.35))
    cv.px(int(rcx) - 1, ry - 13, 'glass_hi')
    # little stone chimney
    cv.rect(x1 - 6, ry - 26, 5, 10, 'stone')
    cv.vline(x1 - 2, ry - 26, ry - 17, 'grey')
    cv.hline(x1 - 7, x1 - 1, ry - 27, 'grey')
    # empty birdhouses: on the wall and on poles
    birdhouse(cv, x0 + 2, wt - 1, pole=False, color='peach', roof='sage')
    birdhouse(cv, x1 + 8, gy - 30, color='sand', roof='rose')
    birdhouse(cv, x0 - 14, gy - 26, color='baby_blue', roof='lavender')
    birdhouse(cv, x1 + 18, gy - 22, color='cotton', roof='sage')
    return int(rcx), ry


FOREST_RAMP = ['#0f2226', '#163430', '#1e4a3d', '#2b5f48', '#3d7351']


FOREST_TRUNK = ('#0e0a14', '#21182a', '#3a2c40')


def everfree_edge(cv):
    R = rng(66)
    deep = ['#0b1418', '#101f22', '#16302e']
    # dense dark wall of canopy behind, with the shadowy forest interior under it
    x = FOREST_X + 6
    while x < WM + 20:
        r = 13 + R.random() * 9
        cy = 78 + R.random() * 22
        balls(cv, [(x, cy, r, r * 0.8, 0), (x + r * 0.6, cy + 6, r * 0.7, r * 0.6, 2)], deep, outline='#081012',
              dith=0.5)
        x += 10 + R.random() * 8
    for x in range(FOREST_X + 2, WM):
        k = min(1.0, (x - FOREST_X) / 30.0)
        y0 = int(96 - 8 * k + 3 * math.sin(x / 7.0))
        cv.vline(x, y0, GY - 1, mix('#1c2c2a', '#0d1719', k))
    for i in range(26):                                # far trunks in the gloom
        x = int(FOREST_X + 10 + R.random() * (WM - FOREST_X))
        cv.rect(x, 96, 2 + int(R.random() * 2), GY - 96, '#0a1214')
    # mid row + front row of gnarled trees
    trees = []
    x = FOREST_X - 8
    k = 0
    while x < WM + 30:
        trees.append((x, 58 + int(R.random() * 36), 15 + R.random() * 9, 500 + k))
        x += 14 + R.random() * 10
        k += 1
    for (x, h, r, s) in trees[1::2]:
        gnarled_tree(cv, int(x), GY - 3, h + 8, r * 0.9, s, ['#0e1a20', '#152a2c', '#1d3c36', '#28503f'],
                     '#081014', FOREST_TRUNK, vines=False)
    for (x, h, r, s) in trees[::2]:
        gnarled_tree(cv, int(x), GY + 1, h, r, s, FOREST_RAMP, '#0a1618', FOREST_TRUNK)
    # undergrowth: ferns and dark bushes, low mist
    for i in range(30):
        x = FOREST_X - 10 + R.random() * (WM - FOREST_X + 10)
        balls(cv, [(x, GY - 1, 5 + R.random() * 4, 4, 0)], FOREST_RAMP[1:], outline='#0a1618', dith=0.5)
    for i in range(22):
        x = FOREST_X - 10 + R.random() * (WM - FOREST_X)
        y = int(GY - 14 + R.random() * 14)
        L = int(8 + R.random() * 22)
        cv.hline(int(x) + 3, int(x) + L - 3, y, '#2c3d44')
        if L > 16:
            cv.hline(int(x) + 7, int(x) + L - 7, y, '#3b5058')


# ------------------------------------------------------------------ main layer build
@lru_cache(maxsize=1)
def _main():
    cv = new(WM)
    ground(cv)
    stream(cv)
    storm_cloud(cv)
    shadow_c = mix(G_MID, 'green', 0.75)
    dapple = mix(G_TOP, 'pale_leaf', 0.3)
    # back fences (behind everything on this plane)
    fence(cv, 0, 110, GY - 4, seed=1, h=11)
    fence(cv, 476, 690, GY - 5, seed=2, h=11)
    # orchard trees (main row) + their dappled shadows
    orchard = [(14, 18, 1), (58, 21, 2), (104, 17, 3), (BEAT_TREE[0], BEAT_TREE[1], 4),
               (622, 18, 5), (662, 15, 6)]
    for x, r, s in orchard:
        tree_shadow(cv, x, GY - 1, r, shadow_c, dapple, seed=s)
    ladder(cv, 86, GY, 34, lean=7)
    for x, r, s in orchard:
        tree(cv, x, GY, r, 10 + s, trunk_h=32 if s == 4 else None)
    crate(cv, 120, GY)
    crate(cv, 124, GY - 7)
    crate(cv, 36, GY)
    basket(cv, 76, GY)
    for i in range(5):
        apple_px(cv, 71 + i * 2 - (i // 3) * 3, GY - 10 - (i // 3) * 2)
    basket(cv, 640, GY)
    farm_sign(cv, SIGN[0], GY)
    # barn
    tree_shadow(cv, BARN_X + 50, GY, 40, shadow_c)
    barn(cv, BARN_X)
    cv.rect(BARN_X - 6, GY - 8, 10, 8, 'sand')
    cv.hline(BARN_X - 6, BARN_X + 3, GY - 8, 'butter')
    cv.rect(BARN_X - 6, GY - 2, 10, 2, 'amber')
    cv.vline(BARN_X - 2, GY - 8, GY - 1, 'clay')
    # clubhouse tree, laundry pole, farmhouse
    tree_shadow(cv, CLUB_X, GY - 1, CLUB_R, shadow_c, dapple, seed=9)
    clubhouse(cv, CLUB_X)
    px = LINE_X[0]
    cv.vline(px, 116, GY - 1, 'latte')
    cv.vline(px + 1, 116, GY - 1, 'brew_mid')
    cv.hline(px - 3, px + 3, 116, 'latte')
    farmhouse(cv, HOUSE_X)
    # cider press, barrels, cart, windmill
    cider_press(cv, PRESS_X)
    bx = BARRELS_X
    barrel(cv, bx + 6, GY, emblem=True)
    barrel(cv, bx + 18, GY)
    barrel(cv, bx + 12, GY - 14)
    apple_cart(cv, CART_X)
    windmill(cv, MILL_X)
    # zap-apple grove (dark trunks, striped apples)
    for x, r, sd in ZAP:
        tree_shadow(cv, x, GY - 1, r, mix(G_LOW, 'navy', 0.35), seed=sd)
    for x, r, sd in ZAP:
        tree(cv, x, GY, r, 900 + sd, ramp=ZAP_LEAF, out=ZAP_OUT, trunk=ZAP_TRUNK, apples=0.8,
             apple_fn=zap_apple)
    # Everfree edge, then the cottage and the footbridge in front of it
    everfree_edge(cv)
    cottage(cv, COTTAGE_X)
    footbridge(cv)
    for i in range(12):                     # reeds along the brook
        x = STREAM_X + (-24 if i % 2 else 22) + int(hash01(i, 3) * 6)
        y = GY + 2 + int(hash01(i, 4) * 10)
        cv.line(x, y, x + (i % 3) - 1, y - 7 - i % 4, mix('green', 'olive', 0.4))
        cv.px(x + (i % 3) - 1, y - 8 - i % 4, 'brew_mid')
    return to_layer(cv)


# ------------------------------------------------------------------ foreground (1.3)
FG_P = 1.3
FG_W = 1320


def _fg_top(x):
    return 170 + wavy(x, 13, ((3, 31.0), (2, 11.0), (1, 4.0)))


@lru_cache(maxsize=1)
def _foreground():
    cv = new(FG_W)
    dk = mix('forest', 'green', 0.35)
    mid = mix('green', 'forest', 0.1)
    lt = mix('green', 'leaf', 0.45)
    for x in range(FG_W):
        t = int(_fg_top(x))
        cv.vline(x, t, H - 1, mid)
        cv.px(x, t, lt)
        cv.px(x, t + 1, lt if x % 3 else mid)
        if x % 2:
            cv.vline(x, t + 6, H - 1, dk)
    R = rng(31)
    for i in range(260):
        x = int(R.random() * FG_W)
        t = int(_fg_top(x))
        h = 3 + int(R.random() * 5)
        cv.line(x, t + 2, x - 1 + int(R.random() * 3), t - h, lt if R.random() < 0.5 else mid)
    for i in range(70):       # wildflowers
        x = int(R.random() * FG_W)
        t = int(_fg_top(x))
        cc = ('pink', 'gold', 'white', 'lilac', 'peach')[i % 5]
        cv.vline(x, t - 3, t + 1, dk)
        cv.px(x, t - 4, cc)
        cv.px(x - 1, t - 3, cc)
        cv.px(x + 1, t - 3, cc)
        cv.px(x, t - 3, 'gold' if cc != 'gold' else 'amber')
    for i in range(16):       # leafy clumps poking up from the bank
        x = 40 + i * 80 + R.random() * 40
        t = int(_fg_top(x))
        dark = x > 1080
        rmp = (FOREST_RAMP[1:] if dark else [mix('forest', 'green', 0.4), 'green', mix('green', 'leaf', 0.5), 'leaf'])
        r = 5 + R.random() * 4
        balls(cv, [(x, t + 2, r, r * 0.75, 0), (x + r, t + 3, r * 0.7, r * 0.55, 1)], rmp,
              outline=mix('forest', 'dark_olive', 0.5), dith=0.5)
        if not dark:
            for k in range(3):
                cv.px(int(x - r / 2 + k * r / 2), int(t - r * 0.4 + (k % 2) * 2), ('pink', 'white', 'gold')[(i + k) % 3])
    for i in range(18):       # rocks
        x = int(R.random() * FG_W)
        t = int(_fg_top(x))
        w = 4 + int(R.random() * 5)
        cv.round_rect(x, t + 1, w, 4, 'grey', r=1)
        cv.hline(x + 1, x + w - 2, t + 1, 'stone')
        cv.hline(x + 1, x + w - 2, t + 4, 'shadow')
    return to_layer(cv)


# ------------------------------------------------------------------ camera
def _hermite(f, f0, f1, p0, p1, v0, v1):
    T = f1 - f0
    t = clamp((f - f0) / T)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1


def cam_x(f):
    if f < 144:
        return _hermite(f, 72, 144, 0, 112, 0.0, 2.0)
    if f < 216:
        return _hermite(f, 144, 216, 112, 274, 2.0, 0.6)
    return _hermite(f, 216, 287, 676, 742, 1.2, 0.6)


# ------------------------------------------------------------------ dynamic bits
BEAT_BASKETS = (                # basket world x, beats that land in it, hanging apple x offsets/heights
    (170, (72, 90, 108), ((-8, 118), (-11, 120), (-5, 116))),
    (314, (126, 144, 162), ((-3, 117), (1, 120), (-6, 121))),
)


def beat_apples(c, f, cam):
    """Apples drop from the orchard tree / the clubhouse tree and land in a basket exactly ON each
    beat, bouncing once; the baskets fill up."""
    for bwx, beats, hangs in BEAT_BASKETS:
        bx = bwx - cam
        if not -20 < bx < W + 20:
            continue
        x0, top = basket(c, bx, GY)
        nland = sum(1 for tb in beats if f >= tb + 7)
        pile = [(2, -3), (6, -3), (10, -3), (4, -5), (8, -5), (6, -7)]
        for i in range(min(nland, len(pile))):
            dx, dy = pile[i]
            apple_px(c, x0 + dx - 1, top + dy)
        land_y = top - 5
        for k, tb in enumerate(beats):
            hx, hang_y = hangs[k]
            sx = bx + hx
            t0 = tb - 9                         # detach 9 frames before the beat
            if f < t0 - 6:
                apple_px(c, sx - 1, hang_y)     # still hanging on the tree
            elif f < t0:                        # wiggle (anticipation)
                apple_px(c, sx - 1 + ((f // 2) % 2), hang_y)
            elif f < tb:                        # free fall, lands exactly on the beat
                t = (f - t0) / 9
                apple_px(c, sx - 1, int(hang_y + (land_y - hang_y) * t * t))
            elif f < tb + 7:                    # one bounce
                t = (f - tb) / 7
                lx = sx - 1 + int(round((x0 + 5 - sx) * t))
                apple_px(c, lx, land_y - int(round(5 * 4 * t * (1 - t))))
                if f - tb < 2:                  # little landing puff
                    for d in (-3, 5):
                        c.px(sx + d, land_y + 3, 'paper')
                    c.px(sx + 1, land_y - 2, 'white')


def falling_leaves(c, f, n=7, seed=0, cols=('leaf', 'pale_leaf', 'gold')):
    for i in range(n):
        sp = 0.35 + hash01(seed, i, 1) * 0.3
        x = (hash01(seed, i, 2) * 360 + f * (0.5 + hash01(seed, i, 3) * 0.4)
             + math.sin(f / (11 + i) + i) * 6) % 360 - 20
        y = (hash01(seed, i, 4) * 200 + f * sp) % 200 - 10
        ph = (f // 4 + i) % 3
        cc = cols[i % len(cols)]
        X, Y = int(x), int(y)
        if ph == 0:
            c.px(X, Y, cc)
            c.px(X + 1, Y, cc)
        elif ph == 1:
            c.px(X, Y, cc)
            c.px(X + 1, Y + 1, mix(cc, 'green', 0.4))
        else:
            c.px(X, Y, mix(cc, 'green', 0.3))


def grass_sway(c, f, ox, seed=3):
    """Tall foreground grass blades swaying in the breeze (screen space)."""
    R = rng(seed)
    xs = R.random(90) * FG_W
    hs = 6 + (R.random(90) * 8).astype(int)
    for i in range(90):
        sx = int(xs[i] - ox)
        if -4 <= sx < W + 4:
            t = int(_fg_top(xs[i]))
            sw = math.sin(f / 13.0 + xs[i] / 23.0)
            tip = int(round(sw * 2))
            cc = mix('green', 'leaf', 0.3 + 0.3 * (i % 3) / 2)
            c.line(sx, t + 3, sx + tip, t - hs[i], cc)
            c.px(sx + tip, t - hs[i], mix('leaf', 'pale_leaf', 0.5))


def pennant(c, x, y, f):
    """Clubhouse pennant flapping from the pole top (x, y)."""
    for i in range(10):
        h = int(round(5 * (1 - i / 10.0)))
        dy = int(round(math.sin(f / 4.0 - i * 0.6) * 1.4 * i / 9.0))
        for j in range(h):
            yy = y + 1 + dy + j - h // 2 + 2
            cc = 'hot_pink' if j != h // 2 or i > 6 else 'gold'
            if i == 0:
                cc = 'magenta'
            c.px(x + 1 + i, yy, cc)


def laundry(c, f, cam):
    xa, xb = LINE_X[0] - cam, LINE_X[1] - 2 - cam
    ya, yb = 116, GY - 29

    def ly(x):
        u = (x - xa) / max(1, xb - xa)
        return int(round(ya + (yb - ya) * u + 3 * math.sin(math.pi * u)))
    for x in range(xa, xb + 1):
        c.px(x, ly(x), 'stone')
    for ox, w, h, cc, sh in ((3, 7, 10, 'paper', 'mist'), (12, 6, 8, 'pink', 'rose'),
                             (20, 5, 7, 'baby_blue', 'periwinkle')):
        for i in range(w):
            x = xa + ox + i
            yt = ly(x) + 1
            ph = f / 5.0 - (ox + i) * 0.55
            hh = h + int(round(math.sin(ph)))
            sw = int(round(math.sin(ph * 0.8 + 1) * 1.2))
            c.vline(x, yt, yt + hh // 2, cc if i < w - 1 else sh)
            c.vline(x + sw, yt + hh // 2, yt + hh, cc if (i < w - 1 and (i + f // 6) % 4) else sh)
        c.px(xa + ox, ly(xa + ox), 'latte')
        c.px(xa + ox + w - 1, ly(xa + ox + w - 1), 'latte')


RB = ('red', 'orange', 'gold', 'leaf', 'sky', 'violet')


def bolt(c, x0, y0, x1, y1, seed, bands=6, branch=True):
    """Rainbow lightning: a jagged ribbon of rainbow bands with a white-hot core."""
    R = rng(seed)
    n = 6
    pts = [(x0, y0)]
    for i in range(1, n):
        t = i / n
        pts.append((lerp(x0, x1, t) + (R.random() - 0.5) * 12, lerp(y0, y1, t)))
    pts.append((x1, y1))
    cols = RB if bands == 6 else ('magic', 'gold', 'ice')
    for k, cc in enumerate(cols):
        off = k - len(cols) // 2
        for p, q in zip(pts, pts[1:]):
            c.line(p[0] + off, p[1], q[0] + off, q[1], cc)
    for p, q in zip(pts, pts[1:]):
        c.line(p[0], p[1], q[0], q[1], 'white')
    if branch:
        p = pts[2]
        q = (p[0] + (R.random() - 0.3) * 18, p[1] + 10 + R.random() * 6)
        for k, cc in enumerate(('magic', 'butter', 'ice')):
            c.line(p[0] + k - 1, p[1], q[0] + k - 1, q[1], cc)
        c.line(p[0], p[1], q[0], q[1], 'white')


def zap_fx(c, f, cam):
    """216: big rainbow strike + flash; crackles on beats; sparkles round the striped apples."""
    sx, sy = STORM[0] - cam, STORM[1] + 2
    cans = [zap_canopy(i) for i in range(len(ZAP))]
    # sparkles always (zap season magic)
    for i, (x, cy, r) in enumerate(cans):
        fx.sparkles(c, x - cam, cy, f, radius=r, n=4, colors=('white', 'magic', 'ice', 'butter'),
                    seed=40 + i, period=20)
    # (cut flash removed: the shot change is a glide transition now, engine.shotfx)
    big = 216 <= f < 226
    if big:
        tx, tcy, tr = cans[1]
        bolt(c, sx - 6, sy, tx - cam + 2, tcy - tr + 4, 7000 + f // 2)
        if f < 222:
            tx2, tcy2, tr2 = cans[2]
            bolt(c, sx + 18, sy, tx2 - cam, tcy2 - tr2 + 6, 7100 + f // 2, bands=3)
        fx.glow(c, tx - cam, tcy - tr + 6, 22, 'magic', 0.45 * (1 - (f - 216) / 10.0), rings=4)
    for tb, i in ((234, 0), (252, 3), (270, 2)):
        if tb <= f < tb + 5:
            x, cy, r = cans[i]
            bolt(c, sx + (i - 1.5) * 20, sy, x - cam, cy - r + 5, 7200 + tb + f // 2, bands=3, branch=False)
            fx.glow(c, x - cam, cy - r + 5, 12, 'magic', 0.35, rings=3)
    # little arcs hopping between apples during the zappy arpeggio
    if 216 <= f < 252 and (f // 3) % 2 == 0:
        i = (f // 6) % len(cans)
        x, cy, r = cans[i]
        R = rng(f // 6)
        ax, ay = x - cam + int((R.random() - 0.5) * r), cy + int((R.random() - 0.5) * r * 0.6)
        bx_, by_ = ax + int((R.random() - 0.5) * 14), ay + int((R.random() - 0.5) * 10)
        for k, cc in enumerate(('magic', 'butter', 'ice')):
            c.line(ax, ay + k - 1, (ax + bx_) // 2 + 2, (ay + by_) // 2 + k - 3, cc)
            c.line((ax + bx_) // 2 + 2, (ay + by_) // 2 + k - 3, bx_, by_ + k - 1, cc)


def everfree_dark(c, f, cam):
    """The right edge of the frame darkens into the Everfree (grows through bar 4)."""
    xs = np.arange(W, dtype=np.float32) + cam
    prof = np.clip((xs - 870) / 190.0, 0, 1) ** 1.2
    k = 0.30 + 0.32 * smooth(seg(f, 224, 282))
    m = k * prof
    a = c.a.astype(np.float32)
    tint = np.array(col('#241c38'), np.float32)
    a = a * (1 - m)[None, :, None] + tint[None, None, :] * (m * 0.85)[None, :, None]
    c.a[:] = np.clip(a, 0, 255).astype(np.uint8)


def smoke(c, x, y, f, seed=0, n=5, height=22, color='mist'):
    for i in range(n):
        period = 54
        t = ((f + i * period / n + hash01(seed, i) * 9) % period) / period
        py = y - int(t * height)
        px = x + int(math.sin(t * 4 + i) * 2 + t * 9)
        r = 1 + int(t * 2.5)
        if t < 0.92:
            cc = color if t < 0.55 else mix(color, 'light_sky', 0.5)
            c.circle(px, py, r, cc)
            if r > 1:
                c.px(px - 1, py - r + 1, 'white')


def balloon(c, f, cam):
    if f < 216:
        if f < 144:
            bx = 72 + 8 * seg(f, 72, 143)
            by = 67 - 5 * seg(f, 72, 143)
        else:                               # world-anchored while the camera pans on, rising a little
            bx = 80 - 0.3 * (cam - 112)
            by = 62 - 12 * smooth(seg(f, 146, 200))
        by = int(round(by + math.sin(f / 40.0) * 2))
        draw_balloon(c, int(bx), by, f, size='m')
    else:
        bx = 196 + 0.22 * (f - 216) - (cam - 676) * 0.35
        by = 44 + int(round(math.sin(f / 37.0) * 2))
        draw_balloon(c, int(bx), by, f, size='s')


# ------------------------------------------------------------------ render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    cam = int(round(cam_x(f)))
    c = _sky().copy()
    draw_clouds(c, f, cam)
    mo = int(round(cam * MT_P)) + 40 if f < 216 else 216 + int(round((cam - 676) * MT_P))
    comp(c, _mountains(), mo)
    canterlot_falls(c, mo, f)
    comp(c, _farhills(), int(round(cam * FH_P)))
    comp(c, _midhills(), int(round(cam * MH_P)))
    balloon(c, f, cam)
    comp(c, _backorchard(), int(round(cam * BO_P)))
    comp(c, _main(), cam)
    if f < 216:
        if f < 190:
            beat_apples(c, f, cam)
        px_, py_ = CLUB_X - cam, CLUB_PY - 19 - 13 - 13
        if -20 < px_ < W + 20:
            pennant(c, px_, py_, f)
            laundry(c, f, cam)
        hx = HOUSE_X - cam
        if -100 < hx < W:
            smoke(c, hx + 61, int(GY - 52 - 22 - 2.5) - 10, f, seed=1)
            chair_rail(c, hx, f)
        if PRESS_X - cam < W:
            press_dynamic(c, PRESS_X - cam, f)
        if MILL_X - cam - 16 < W:
            windmill_dynamic(c, MILL_X - cam, f)
    else:
        stream_dynamic(c, f, cam)
        smoke(c, COTTAGE_X + 50 - 4 - cam, GY - 24 + 2 - 28, f, seed=2, n=4, height=18)
    fo = int(round(cam * FG_P))
    comp(c, _foreground(), fo)
    grass_sway(c, f, fo)
    falling_leaves(c, f)
    if f < 216:
        fx.dust_motes(c, f, (0, 60, 170, 90), n=14, color='butter', seed=4)
    else:
        everfree_dark(c, f, cam)
        zap_fx(c, f, cam)

    # callouts + card (screen coords of world landmarks)
    callout(c, SIGN[0] + 8 - cam, GY - 28, 'SWEET APPLE ACRES', f, 90, 150, side='up', length=44)
    callout(c, BARN_X + BARN_FW + 20 - cam, GY - 68, 'BIG RED BARN', f, 108, 176, side='up', length=12)
    if f < 216:
        callout(c, CLUB_X - cam, CLUB_PY - 19 - 12, 'CLUBHOUSE', f, 158, 209, side='up', length=22)
    zx, zcy, zr = zap_canopy(2)
    callout(c, zx - cam, zcy + 4, 'ZAP APPLE TREES', f, 220, 279, side='down', length=30)
    callout(c, COTTAGE_X + 25 - cam, GY - 24 - 14, "FLUTTERSHY'S COTTAGE", f, 238, 281, side='up', length=22)
    atlas_card(c, f, 100, 200, 'APPLE FARM', ['ORCHARDS - BARN - CIDER', 'SOUTH OF PONYVILLE'])
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), BARN_X + 36 - cam, GY - 58)
    return c


SFX = [
    (80, 'wind', 0.3),
    (90, 'pop', 0.32), (108, 'pop', 0.32), (126, 'pop', 0.32), (144, 'pop', 0.28), (162, 'pop', 0.24),
    (172, 'creak', 0.22), (190, 'pour', 0.2), (208, 'creak', 0.18),
    (216, 'zap', 0.85), (234, 'sparkle', 0.3), (240, 'wind', 0.35), (252, 'zap', 0.35),
    (258, 'fountain', 0.18), (270, 'zap', 0.25),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
