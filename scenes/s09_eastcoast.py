"""EAST COAST: Fillydelphia, Baltimare, Horseshoe Bay (scene frames 0..359; HUD clock 13:15 -> 14:00).

Beat sheet in SCENE frames (bar = 72, beat = 18; the 2-bar map flight is added by regionfx.long_flight,
film = scene + 72 for scene >= 72). Early afternoon, sea breeze, bright sun upper right (shadows fall
lower left), fluffy cumulus drifting on the breeze. Cameras: slow constant-speed `cruise` pans.
  0-71     World-map travel shot (`travel_frame`).
  72-179   FILLYDELPHIA (3.6 s). REVEAL (72): iris opens 72-79 on the GIANT BRONZE BELL (cracked), hung
           in an open white pavilion on a brick plaza. Pan right 54 px: Georgian brick rows with white
           frames + shutters, the white clock spire, a domed hall, glass crystal towers behind, golden
           fields at the right edge, plaza fountain. The bell is pulled into a swing from 82; BELL HIT
           (108): it strikes at the top of its swing -> concentric sound-ring arcs, 1-px shake, glint;
           echo stroke 126; the swing dies away by 179. Callouts FILLYDELPHIA 90-162, THE BIG BELL
           108-180. Card 98-160.
  180      CLOUD transition (168-191) to BALTIMARE (180-269, 3.0 s): domed white civic hall with a
           colonnade on its rise, pale towers with blue cones, a gold-roofed tower, red-brick
           warehouses, dock cranes (one swings a crate), crates, two moored ships rocking, flags,
           harbour reflections + sun sparkles. Pan right 72 px. Horn on the cut. Callout BALTIMARE 186-258.
  270      GLIDE transition (262-277) to HORSESHOE BAY (270-359): round sandy bay, turquoise shallows,
           foam waves washing up the sand ON EVERY BEAT, a three-masted tall ship at anchor (sails
           furled, pennants), red-and-white lighthouse on a rocky point (beam sweeping), palms; the pan
           (still moving at the end) runs south into the HAYSEED SWAMPS: bayou stilt cabins under
           mossy cypresses, lanterns, stove smoke. Callouts HORSESHOE BAY 274-346, HAYSEED SWAMPS
           288-360. Toast from 302; diamond wipe 352-359.
Balloon: over the bell pavilion (m), near the cranes (s), above the tall ship (s).
Ambient: clouds, flags/pennants, chimney + stove smoke, tower glints, fountain, bell swing + rings,
water reflections/sparkles, rocking ships, crane, foam waves, lighthouse beam, palm sway, lanterns.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'eastcoast'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
GY = 150
SUN = (292, 31)
_L = np.array([0.55, -0.70, 0.45])
_L = _L / np.linalg.norm(_L)          # key light: upper right (early-afternoon sun)


def C(c):
    return np.array(col(c), np.uint8)


HAZE = col('#cfe5f3')


def hz(c, a, to=None):
    return mix(c, to or HAZE, a)


# ------------------------------------------------------------------ palette
SKY_COLS = ['#3a7fd2', '#4a90dc', '#5ea3e6', '#78b7ee', '#94caf3', '#b2dbf5', '#cde7f4', '#e2f0ef']
CLOUD_RAMP = ['#9fb1d9', '#bccbeb', '#d9e3f8', '#eef3ff', 'white']
LEAF = ['#1b5244', '#256d4c', '#348b53', '#55aa5a', '#8ccc6a', '#c2e57e']
LEAF_OUT = '#173c33'
TRUNK = ('#3b2a26', '#5e3f30', '#83593e', '#a8784f')
BRICK = ('#6a2c2a', '#8c3a30', '#a74838', '#c05c45', '#d67b5c')
BRICK_FAR = tuple(mix(b, '#d9b8c0', 0.35) for b in BRICK)
TRIM = ('#8f8aa8', '#c4c2d6', '#e9e8f2', 'white')
SLATE = ('#3a3248', '#4f4562', '#665c7c', '#857c9a', '#a49cb6')
GLASS = ('#3d5f9a', '#5f8fca', '#9cc8ee', '#e3f6ff')
SHUTTER = ('#1f4a40', '#2f6655', '#3f8068')
BRONZE = ('#2e1a12', '#4a2c1a', '#6a4024', '#8c5a2e', '#b07a3c', '#d6a458', '#f4d892')
PLAZA = ('#8e5146', '#a8604f', '#bd735d', '#cf8b70', '#e2b29a')
GRASS = ('#2a6a45', '#3a8a4c', '#5aab55', '#8ccb64', '#bfe27a')


# ------------------------------------------------------------------ layer helpers
def new(w, h=H):
    return Canvas(w, h, KEY)


def to_layer(cv):
    a = cv.a
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    return a, m


def comp(c, layer, ox, oy=0):
    """Composite a cached (rgb, mask) layer: layer x `ox` is shown at screen x 0 (oy likewise)."""
    a, m = layer
    ox, oy = int(ox), int(oy)
    lh, lw = a.shape[:2]
    x0, x1 = max(0, ox), min(lw, ox + W)
    y0, y1 = max(0, oy), min(lh, oy + H)
    if x1 <= x0 or y1 <= y0:
        return
    dst = c.a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    np.copyto(dst, a[y0:y1, x0:x1], where=m[y0:y1, x0:x1, None])


def hashgrid(h, w, x0, y0, seed):
    """Deterministic per-pixel noise 0..1 (vectorised)."""
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    v = (xx * 374761393 + yy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    v = ((v ^ (v >> 13)) * 1274126177) & 0xFFFFFFFF
    v = v ^ (v >> 16)
    return (v & 0xFFFF) / 65535.0


def balls(cv, blobs, ramp, outline=None, dith=0.55, crease=True, light=_L, bias=0.0, gain=1.0, rim=None):
    """Shaded union of ellipsoid blobs [(cx, cy, rx, ry, z)] (frontmost wins), lit by `light`,
    quantised to `ramp` (dark->light) with ordered dither, creases where clumps overlap, outline."""
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
        hh = z + zz * min(rx, ry)
        upd = ins & (hh > best)
        best[upd] = hh[upd]
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
        out[bound] = C(outline)
        if rim is not None:
            lit = bound & ((NX * light[0] + NY * light[1]) > 0.55)
            out[lit] = C(rim)
    reg = cv.a[y0:y1, x0:x1]
    reg[m] = out[m]
    return x0, y0, m, lvl


def canopy_blobs(cx, cy, r, seed, squash=0.86):
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


def tree(cv, x, base, r, seed, ramp=LEAF, out=LEAF_OUT, trunk=TRUNK, th=None, haze=0.0, squash=0.86):
    """Round curly-canopy tree standing on `base`."""
    if haze:
        ramp = [hz(c_, haze) for c_ in ramp]
        out = hz(out, haze)
        trunk = [hz(c_, haze) for c_ in trunk]
    th = th if th is not None else int(r * 0.9)
    cy = base - th - r * 0.55
    tw = max(2, int(r / 4))
    cv.rect(x - tw // 2, int(cy), tw, base - int(cy), trunk[1])
    cv.vline(x - tw // 2, int(cy), base - 1, trunk[0])
    if tw > 2:
        cv.vline(x + tw - tw // 2 - 1, int(cy), base - 1, trunk[2])
    cv.px(x - tw // 2 - 1, base - 1, trunk[0])
    cv.px(x + tw - tw // 2, base - 1, trunk[1])
    balls(cv, canopy_blobs(x, cy, r, seed, squash), ramp, outline=out, rim=ramp[-1])
    return cy


def tri_noise_rows(seed, n, lo, hi):
    R = rng(seed)
    return [lo + (hi - lo) * R.random() for _ in range(n)]


# ------------------------------------------------------------------ architecture kit
def brick_wall(cv, x, y, w, h, ramp=BRICK, seed=0, near=False):
    """Brick facade fill: base colour, running-bond courses, speckle; darker toward the ground."""
    x0, y0, x1, y1 = cv._clip(x, y, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    hh, ww = y1 - y0, x1 - x0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.array([col(c_) for c_ in ramp], np.uint8)
    nz = hashgrid(hh, ww, x0, y0, seed)
    idx = np.full((hh, ww), 2, int)
    ch = 3 if near else 2
    course = (yy - y) // ch
    mort = (yy - y) % ch == ch - 1
    bw = 6 if near else 4
    joint = ((xx - x + (course % 2) * (bw // 2)) % bw == 0) & ~mort
    idx[nz < 0.18] = 1
    idx[nz > 0.86] = 3
    if near:
        idx[mort] = 1
        idx[joint] = 1
        idx[mort & ((xx + course) % 5 == 0)] = 0
    else:
        idx[mort & ((xx + course) % 2 == 0)] = 1
        idx[joint & (nz < 0.55)] = 1
    # ambient occlusion near the bottom, light toward the top-right
    idx[(yy >= y + h - 3) & (bayer(hh, ww, x0, y0) < 0.5)] = 1
    idx[(yy >= y + h - 2)] = np.minimum(idx[(yy >= y + h - 2)], 1)
    cv.a[y0:y1, x0:x1] = rr[np.clip(idx, 0, len(ramp) - 1)]


def window(cv, x, y, w, h, frame=TRIM, glass=GLASS, shutter=SHUTTER, sill=True, muntin=True,
           arch=False, box=None):
    """Georgian sash window: white frame, glazing bars, sky-reflecting glass, shutters, sill."""
    cv.rect(x, y, w, h, frame[2])
    gx, gy_, gw, gh = x + 1, y + 1, w - 2, h - 2
    if gw <= 0 or gh <= 0:
        return
    cv.rect(gx, gy_, gw, gh, glass[1])
    cv.hline(gx, gx + gw - 1, gy_, glass[0])
    cv.vline(gx, gy_, gy_ + gh - 1, glass[0])
    # sky reflection: a lighter diagonal streak toward the upper right
    for k in range(min(gw, gh)):
        px_, py_ = gx + gw - 1 - k, gy_ + 1 + k
        if gx < px_ <= gx + gw - 1 and py_ < gy_ + gh:
            cv.px(px_, py_, glass[2])
    cv.px(gx + gw - 1, gy_ + gh - 1, glass[2])
    if muntin:
        cv.hline(gx, gx + gw - 1, gy_ + gh // 2, frame[2])
        if gw >= 5:
            cv.vline(gx + gw // 2, gy_, gy_ + gh - 1, frame[2])
        if gh >= 9:
            cv.hline(gx, gx + gw - 1, gy_ + gh // 4, frame[1])
            cv.hline(gx, gx + gw - 1, gy_ + 3 * gh // 4, frame[1])
    cv.vline(x, y, y + h - 1, frame[1])                 # frame shade on the left (sun at right)
    if arch:
        cv.px(x, y, KEY if False else frame[1])
        cv.hline(x + 1, x + w - 2, y - 1, frame[2])
        cv.px(x + w // 2, y - 2, frame[2])
        cv.px(x + w // 2, y - 1, frame[3])
    else:
        cv.hline(x, x + w - 1, y - 1, frame[2])          # lintel
        cv.px(x + w // 2, y - 1, frame[3])               # keystone
    if shutter is not None:
        sw = 2 if w >= 7 else 1
        cv.rect(x - sw, y, sw, h, shutter[1])
        cv.rect(x + w, y, sw, h, shutter[1])
        cv.vline(x - sw, y, y + h - 1, shutter[0])
        cv.vline(x + w + sw - 1, y, y + h - 1, shutter[2])
        if h >= 8:
            for yy_ in range(y + 1, y + h - 1, 2):
                cv.hline(x - sw, x - 1, yy_, shutter[0])
                cv.hline(x + w, x + w + sw - 1, yy_, shutter[0])
    if sill:
        cv.hline(x - 1, x + w, y + h, frame[3])
        cv.hline(x - 1, x + w, y + h + 1, frame[0])
    if box:
        cv.rect(x, y + h - 1, w, 2, box[0])
        for i in range(w):
            cv.px(x + i, y + h - 2, box[1 + (i % (len(box) - 1))] if i % 2 == 0 else LEAF[3])


def slate_roof(cv, x, y, w, h, ramp=SLATE, inset=None):
    """Front-facing pitched roof band (trapezoid), slate courses, ridge highlight."""
    inset = h if inset is None else inset
    for i in range(h):
        k = int(round(inset * (h - i) / h))
        cv.hline(x + k, x + w - 1 - k + (1 if k == 0 else 0), y + i, ramp[2] if i % 2 else ramp[1])
        cv.px(x + k, y + i, ramp[0])
        cv.px(x + w - 1 - k, y + i, ramp[3])
    cv.hline(x + inset, x + w - 1 - inset, y, ramp[3])


def chimney(cv, x, top, base, ramp=BRICK, trim=TRIM):
    cv.rect(x, top, 4, base - top, ramp[2])
    cv.vline(x, top, base - 1, ramp[1])
    cv.vline(x + 3, top, base - 1, ramp[3])
    cv.rect(x - 1, top, 6, 1, trim[2])
    cv.px(x - 1, top, trim[1])
    cv.px(x + 4, top, trim[3])


def door(cv, x, gy, w=5, h=9, color='#2f6655', trim=TRIM, fan=True):
    cv.rect(x - 1, gy - h - 2, w + 2, h + 2, trim[2])
    cv.vline(x - 1, gy - h - 2, gy - 1, trim[1])
    cv.rect(x, gy - h, w, h, color)
    cv.vline(x, gy - h, gy - 1, shade(color, 0.7))
    cv.px(x + w - 2, gy - h // 2, 'gold')
    if fan:
        cv.hline(x, x + w - 1, gy - h - 1, GLASS[2])
        cv.px(x + w // 2, gy - h - 1, trim[3])
        cv.hline(x + 1, x + w - 2, gy - h - 3, trim[2])
    cv.hline(x - 2, x + w + 1, gy - 1, trim[1])          # step


def flag(c, x, y, f, w=8, h=5, cols=('baby_blue', 'white'), emblem='gold', seed=0, pole=True):
    """Small flag on a pole; the cloth ripples in the sea breeze (blowing to the left = wind from the
    sea on the right)."""
    if pole:
        c.vline(x, y, y + h + 6, 'shadow')
        c.px(x, y - 1, 'gold')
    for i in range(w):
        ph = f * 0.42 + i * 0.9 + seed
        dy = int(round(math.sin(ph) * (0.4 + 0.18 * i)))
        top = y + dy
        cc = cols[0] if (i // 2) % 2 == 0 or len(cols) == 1 else cols[1]
        cc2 = shade(cols[0], 0.82) if math.sin(ph + 0.8) < -0.3 else cols[0]
        c.vline(x - 1 - i, top, top + h - 1, cc2)
        if len(cols) > 1:
            c.px(x - 1 - i, top + h // 2, cols[1])
    if emblem:
        c.px(x - 1 - w // 2, y + h // 2 + int(round(math.sin(f * 0.42 + w // 2 * 0.9 + seed) * 0.6)), emblem)


def smoke(c, x, y, f, seed=0, n=5, height=20, color='#eef2f6', drift=-0.5):
    """Chimney smoke puffs drifting left on the breeze."""
    for i in range(n):
        period = 54
        t = ((f + i * period / n + hash01(seed, i) * 9) % period) / period
        px_ = x + int(round(t * height * drift + math.sin(t * 5 + i) * 1.5))
        py_ = y - int(t * height)
        r = 1 if t < 0.35 else 2
        cc = color if t < 0.6 else mix(color, 'stone', 0.25)
        if t < 0.9:
            c.circle(px_, py_, r, cc)


# ------------------------------------------------------------------ sky + clouds
@lru_cache(maxsize=None)
def _sky(warm=0.0):
    c = Canvas()
    cols = [mix(s_, '#ffe9c8', warm * i / len(SKY_COLS)) for i, s_ in enumerate(SKY_COLS)]
    c.gradient(0, 0, W, 130, cols, soft=0.7)
    c.rect(0, 130, W, H - 130, cols[-1])
    fx.glow(c, SUN[0], SUN[1], 80, '#fff1cf', strength=0.28, rings=7)
    fx.glow(c, SUN[0], SUN[1], 26, '#fffbe6', strength=0.45, rings=4)
    c.circle(SUN[0], SUN[1], 7, '#fff6d6')
    c.circle(SUN[0], SUN[1], 6, '#fffdf2')
    c.circle(SUN[0] + 1, SUN[1] - 1, 3, 'white')
    return c


@lru_cache(maxsize=None)
def _cumulus(w, h, seed):
    cv = new(w + 6, h + 6)
    R = rng(seed)
    base = h + 2
    blobs = []
    n = max(3, w // 9)
    for i in range(n):
        u = (i + 0.5) / n
        r = h * (0.30 + 0.08 * R.random())
        blobs.append((3 + u * w, base - r * 0.75, r * 1.35, r * 0.8, 0.0))
    k = max(2, w // 12)
    for i in range(k):
        u = 0.18 + 0.64 * (i + 0.3 + 0.4 * R.random()) / k
        bump = math.sin(u * math.pi)
        r = h * (0.32 + 0.36 * bump) * (0.85 + 0.3 * R.random())
        blobs.append((3 + u * w, base - h * 0.30 - r * 0.62, r, r * 0.92, r * 0.35))
    balls(cv, blobs, CLOUD_RAMP, outline=None, dith=0.7, crease=True, bias=0.06)
    a, m = to_layer(cv)
    m[base + 1:] = False
    # flat, shaded underside
    rows = np.nonzero(m.any(axis=1))[0]
    if len(rows):
        yb = rows[-1]
        a[yb][m[yb]] = C(CLOUD_RAMP[0])
        a[yb - 1][m[yb - 1]] = C(CLOUD_RAMP[1])
    return a, m


def draw_clouds(c, f, cam, clouds, drift=0.12):
    for (x, y, w, h, seed, par) in clouds:
        sx = int(round(x - cam * par - f * drift * (0.6 + par * 3)))
        comp(c, _cumulus(w, h, seed), -sx, -y)


# ================================================================== SHOT A: FILLYDELPHIA
A_CAM = 54
BELL_X, BELL_PIVOT = 204, 92          # near-layer x of the bell, y of the yoke pivot
PAV_L, PAV_R = BELL_X - 38, BELL_X + 38
SPIRE_X = 262                          # mid-layer x of the clock tower
DOME_X = 125                           # mid-layer x of the domed hall
MID_P, GLASS_P, FIELD_P, FG_P = 0.55, 0.3, 0.12, 1.35
A_CLOUDS = [(18, 50, 54, 18, 11, 0.05), (120, 22, 74, 24, 12, 0.06), (236, 58, 46, 14, 13, 0.04),
            (318, 16, 60, 20, 14, 0.07)]


def a_cam(f):
    return int(round(A_CAM * cruise(seg(f, 72, 179))))


@lru_cache(maxsize=1)
def _a_fields():
    """Far rolling farmland with golden field patches (haze)."""
    w = 360
    cv = new(w)
    xs = np.arange(w)
    top = 104 + 5 * np.sin(xs / 37.0) + 3 * np.sin(xs / 13.0 + 1.0)
    top2 = 116 + 4 * np.sin(xs / 29.0 + 2.0) + 2 * np.sin(xs / 9.0)
    gold = [hz('#e8c45a', 0.35), hz('#f2d77a', 0.35), hz('#c9b04e', 0.35)]
    greens = [hz('#7fb860', 0.45), hz('#9ccc6a', 0.45)]
    for x in range(w):
        t = int(top[x])
        cv.vline(x, t, H - 1, hz('#8fbf68', 0.5))
        cv.px(x, t, hz('#b9dc84', 0.5))
        t2 = int(top2[x])
        cv.vline(x, t2, H - 1, hz('#86b862', 0.35))
        cv.px(x, t2, hz('#b0d47c', 0.35))
    # field patches with furrow stripes
    R = rng(5)
    for i in range(16):
        fx0 = int(R.random() * w)
        fw = 14 + int(R.random() * 22)
        band = 0 if R.random() < 0.5 else 1
        base = top if band == 0 else top2
        cc = gold[i % 3] if R.random() < 0.7 else greens[i % 2]
        for x in range(fx0, min(w, fx0 + fw)):
            t = int(base[x]) + 2
            hgt = 5 + (x - fx0) % 3 // 2
            for yy in range(t, t + hgt):
                cv.px(x, yy, cc if (yy + x // 3) % 3 else shade(cc, 0.9))
        cv.vline(fx0, int(base[fx0]) + 1, int(base[fx0]) + 7, hz(LEAF[1], 0.45))   # hedgerow
    # far farm: little barns + silos
    for (bx, by) in ((300, int(top2[300])), (330, int(top[330]))):
        cv.rect(bx, by - 4, 6, 4, hz('brick', 0.4))
        cv.tri((bx - 1, by - 4), (bx + 7, by - 4), (bx + 3, by - 7), hz('#8a4a3c', 0.4))
        cv.rect(bx + 8, by - 8, 3, 8, hz('stone', 0.3))
        cv.circle(bx + 9, by - 8, 1, hz('stone', 0.2))
    for i in range(8):
        tx = 20 + i * 47 + int(R.random() * 10)
        tree(cv, tx, int(top[min(w - 1, tx)]) + 3, 4 + int(R.random() * 2), 300 + i, th=2, haze=0.45)
    return to_layer(cv)


def glass_tower(cv, x, top, w, base, seed=0, haze=0.35, facet='pyramid'):
    """Modern crystal-glass tower: lit right face, shaded left face, mullion grid, faceted top."""
    g = [hz(c_, haze) for c_ in ('#3e6aa8', '#5b8fcf', '#80b4e8', '#b7dcf8', '#eaf8ff')]
    lw = max(3, w // 3)                       # left (shaded) face width
    cv.rect(x, top, lw, base - top, g[1])
    cv.rect(x + lw, top, w - lw, base - top, g[2])
    cv.vline(x, top, base - 1, g[0])
    cv.vline(x + w - 1, top, base - 1, g[3])
    for yy in range(top + 2, base, 3):        # floor lines
        cv.hline(x + 1, x + lw - 1, yy, g[0])
        cv.hline(x + lw, x + w - 2, yy, g[1])
    for xx in range(x + lw + 2, x + w - 1, 3):  # mullions
        cv.vline(xx, top, base - 1, mix(g[2], g[1], 0.5))
    # sky reflection: big diagonal light band
    for k in range(base - top):
        bx = x + lw + 1 + (k // 2) % (w - lw)
        if k % 23 < 9:
            cv.px(bx, top + k, g[3])
            cv.px(min(x + w - 2, bx + 1), top + k, g[3])
    # faceted crystal top
    if facet == 'pyramid':
        hgt = w // 2 + 4
        cv.tri((x, top), (x + lw, top), (x + lw, top - hgt), g[1])
        cv.tri((x + lw, top), (x + w, top), (x + lw, top - hgt), g[3])
        cv.line(x + lw, top - hgt, x + lw, top - 1, g[4])
        cv.px(x + lw, top - hgt - 1, g[4])
    else:
        hgt = w // 2 + 2
        cv.poly([(x, top), (x + w, top), (x + w, top - hgt), (x + w // 3, top - hgt + 2)], g[2])
        cv.line(x + w // 3, top - hgt + 2, x + w - 1, top - hgt, g[4])
    cv.hline(x, x + w - 1, top, g[4])


@lru_cache(maxsize=1)
def _a_glass():
    w = 360
    cv = new(w)
    glass_tower(cv, 22, 46, 15, 130, facet='slant', haze=0.42)
    glass_tower(cv, 92, 34, 18, 130, haze=0.38)
    glass_tower(cv, 113, 54, 12, 130, facet='slant', haze=0.45)
    glass_tower(cv, 286, 40, 20, 130, haze=0.38)
    glass_tower(cv, 310, 58, 13, 130, facet='slant', haze=0.45)
    # far pale skyline blocks at the tower feet
    R = rng(21)
    x = 0
    while x < w:
        bw = 8 + int(R.random() * 12)
        bt = 92 + int(R.random() * 14)
        cc = hz(BRICK[3], 0.55) if R.random() < 0.6 else hz('#e8e2dc', 0.4)
        cv.rect(x, bt, bw, 130 - bt, cc)
        cv.vline(x, bt, 129, shade(cc, 0.9))
        for wy in range(bt + 3, 128, 4):
            for wx in range(x + 2, x + bw - 1, 3):
                cv.px(wx, wy, hz('#e9f2fb', 0.3))
        x += bw
    return to_layer(cv)


def row_house(cv, x, gy, w, floors, seed, ramp=BRICK, door_col=None, roof=True, dormers=True,
              box=None, cupola=False):
    """Georgian brick row house: brick front, white cornice, sash windows with shutters, door with a
    fanlight, slate roof with dormers and chimneys. Returns the roof top y."""
    R = rng(seed)
    fh = 10
    top = gy - floors * fh - 4
    brick_wall(cv, x, top, w, gy - top, ramp, seed=seed)
    cv.vline(x, top, gy - 1, ramp[0])
    cv.vline(x + w - 1, top, gy - 1, ramp[3])
    # stone base course
    cv.rect(x, gy - 3, w, 3, TRIM[1])
    cv.hline(x, x + w - 1, gy - 3, TRIM[2])
    # string course between floors
    for k in range(1, floors):
        cv.hline(x, x + w - 1, gy - 3 - k * fh, TRIM[1])
    # windows
    nwin = max(1, (w - 2) // 8)
    sp = w / nwin
    door_i = nwin // 2 if door_col else -1
    for fl in range(floors):
        wy = gy - 3 - (fl + 1) * fh + 2
        for i in range(nwin):
            wx = int(x + sp * i + sp / 2 - 2)
            if fl == 0 and i == door_i:
                door(cv, wx, gy - 3, w=5, h=7, color=door_col)
                continue
            window(cv, wx, wy, 5, 7, shutter=SHUTTER if (seed + i) % 3 else None,
                   box=box if (fl == 1 and box) else None)
    # cornice with dentils
    cv.rect(x - 1, top, w + 2, 3, TRIM[2])
    cv.hline(x - 1, x + w, top, TRIM[3])
    for i in range(x, x + w, 2):
        cv.px(i, top + 2, TRIM[0])
    cv.hline(x, x + w - 1, top + 3, ramp[0])
    rt = top
    if roof:
        rh = 7
        slate_roof(cv, x, top - rh, w, rh, inset=4)
        rt = top - rh
        if dormers and w >= 20:
            for i in range(max(1, w // 18)):
                dx_ = int(x + (i + 0.5) * w / max(1, w // 18)) - 3
                cv.rect(dx_, top - 6, 6, 6, TRIM[2])
                cv.tri((dx_ - 1, top - 6), (dx_ + 6, top - 6), (dx_ + 3, top - 9), TRIM[2])
                cv.line(dx_ + 3, top - 9, dx_ + 6, top - 6, TRIM[3])
                cv.rect(dx_ + 1, top - 5, 4, 4, GLASS[1])
                cv.px(dx_ + 4, top - 5, GLASS[2])
                cv.vline(dx_, top - 6, top - 1, TRIM[1])
        for cx_ in (x + 2, x + w - 6):
            if R.random() < 0.85:
                chimney(cv, cx_, rt - 4, top, ramp)
    if cupola:
        cxp = x + w // 2
        cv.rect(cxp - 3, rt - 7, 7, 7, TRIM[2])
        cv.vline(cxp - 3, rt - 7, rt - 1, TRIM[1])
        cv.rect(cxp - 1, rt - 6, 3, 4, GLASS[0])
        cv.ellipse(cxp, rt - 8, 3, 2, '#7fc4b4')
        cv.px(cxp + 1, rt - 9, '#c4ecd8')
        cv.vline(cxp, rt - 13, rt - 10, 'shadow')
        cv.px(cxp, rt - 14, 'gold')
    return rt


def clock_spire(cv, x, gy):
    """Independence-hall style tower: brick shaft, big clock, white belfry tiers, cupola, gold finial.
    x = centre. Returns the finial top y."""
    w = 18
    tl = x - w // 2
    top = gy - 64
    brick_wall(cv, tl, top, w, gy - top, BRICK, seed=77)
    cv.vline(tl, top, gy - 1, BRICK[0])
    cv.vline(tl + w - 1, top, gy - 1, BRICK[4])
    # tall arched window + door
    window(cv, x - 3, gy - 30, 7, 12, shutter=None, arch=True)
    door(cv, x - 3, gy - 3, w=7, h=10, color='#e9eef7', fan=True)
    cv.rect(tl, gy - 3, w, 3, TRIM[1])
    # clock face on the brick
    cy = top + 10
    cv.rect(tl - 1, top - 2, w + 2, 3, TRIM[2])
    cv.hline(tl - 1, tl + w, top - 2, TRIM[3])
    cv.circle(x, cy, 6, TRIM[2])
    cv.circle(x, cy, 5, 'gold')
    cv.circle(x, cy, 4, '#fbf6e8')
    for a in range(12):
        ang = a * math.pi / 6
        cv.px(x + int(round(math.cos(ang) * 3.5)), cy + int(round(math.sin(ang) * 3.5)), 'shadow')
    cv.line(x, cy, x, cy - 3, 'ink')                 # hands at ~1:30
    cv.line(x, cy, x + 2, cy + 1, 'ink')
    cv.px(x, cy, 'gold')
    # white wooden belfry tier
    b0 = top - 2
    cv.rect(x - 7, b0 - 14, 14, 14, TRIM[2])
    cv.vline(x - 7, b0 - 14, b0 - 1, TRIM[1])
    cv.vline(x + 6, b0 - 14, b0 - 1, TRIM[3])
    for ax in (x - 5, x + 1):
        cv.rect(ax, b0 - 11, 4, 9, '#4b5578')
        cv.hline(ax, ax + 3, b0 - 11, TRIM[2])
        cv.px(ax, b0 - 11, TRIM[2])
        cv.px(ax + 3, b0 - 11, TRIM[2])
        cv.px(ax + 1, b0 - 10, '#68739a')
        cv.hline(ax, ax + 3, b0 - 4, TRIM[1])
    cv.rect(x - 8, b0 - 16, 16, 2, TRIM[2])
    cv.hline(x - 8, x + 7, b0 - 16, TRIM[3])
    for i in range(x - 7, x + 8, 2):           # balustrade urns
        cv.px(i, b0 - 17, TRIM[2])
    # octagonal second tier with round arches
    t0 = b0 - 16
    cv.rect(x - 5, t0 - 11, 10, 11, TRIM[2])
    cv.vline(x - 5, t0 - 11, t0 - 1, TRIM[1])
    cv.vline(x + 4, t0 - 11, t0 - 1, TRIM[3])
    for ax in (x - 3, x + 1):
        cv.rect(ax, t0 - 8, 2, 6, '#4b5578')
        cv.px(ax, t0 - 9, TRIM[1])
    cv.rect(x - 6, t0 - 12, 12, 1, TRIM[3])
    # cupola dome + spire
    d0 = t0 - 12
    cv.ellipse(x, d0, 4, 3, TRIM[2])
    cv.rect(x - 5, d0, 10, 1, TRIM[2])
    cv.px(x + 2, d0 - 2, 'white')
    cv.px(x - 3, d0 - 1, TRIM[1])
    cv.vline(x, d0 - 12, d0 - 3, TRIM[2])
    cv.vline(x + 1, d0 - 8, d0 - 3, TRIM[1])
    cv.circle(x, d0 - 9, 1, 'gold')
    cv.px(x, d0 - 13, 'gold')
    cv.px(x, d0 - 14, 'butter')
    return d0 - 14


def domed_hall(cv, x, gy, w=58):
    """White classical hall with a portico and a ribbed dome (lantern + gold finial). x = left."""
    cx = x + w // 2
    top = gy - 26
    cv.rect(x, top, w, gy - top, TRIM[2])
    cv.vline(x, top, gy - 1, TRIM[0])
    cv.vline(x + w - 1, top, gy - 1, TRIM[3])
    for i in range(5):                                      # ashlar courses
        cv.hline(x + 1, x + w - 2, top + 4 + i * 5, TRIM[1])
    for wx in range(x + 4, x + w - 4, 8):
        if abs(wx + 2 - cx) > 10:
            window(cv, wx, top + 6, 5, 9, shutter=None, arch=True, frame=TRIM)
    # portico: pediment + 4 columns
    px0 = cx - 13
    cv.rect(px0, top - 2, 26, gy - top + 2, TRIM[1])
    for i in range(4):
        colx = px0 + 2 + i * 7
        cv.rect(colx, top + 2, 3, gy - top - 5, TRIM[2])
        cv.vline(colx, top + 2, gy - 4, TRIM[1])
        cv.vline(colx + 2, top + 2, gy - 4, TRIM[3])
        cv.rect(colx - 1, top + 1, 5, 1, TRIM[3])
    cv.rect(px0 + 5, gy - 14, 3, 11, '#5b5f86')
    cv.rect(px0 + 12, gy - 16, 3, 13, '#5b5f86')
    cv.rect(px0 + 19, gy - 14, 3, 11, '#5b5f86')
    cv.rect(px0 - 1, top - 3, 28, 3, TRIM[2])
    cv.hline(px0 - 1, px0 + 26, top - 3, TRIM[3])
    cv.tri((px0 - 1, top - 3), (px0 + 27, top - 3), (cx, top - 11), TRIM[2])
    cv.line(cx, top - 11, px0 + 26, top - 3, TRIM[3])
    cv.line(cx, top - 11, px0 - 1, top - 3, TRIM[1])
    cv.circle(cx, top - 6, 1, 'gold')
    cv.rect(x - 1, gy - 2, w + 2, 2, TRIM[1])
    # drum + dome
    dt = top - 11
    cv.rect(cx - 12, dt - 6, 24, 8, TRIM[2])
    cv.vline(cx - 12, dt - 6, dt + 1, TRIM[1])
    for i in range(cx - 10, cx + 11, 3):
        cv.vline(i, dt - 5, dt, TRIM[1] if i < cx else TRIM[2])
        cv.vline(i + 1, dt - 5, dt, '#5b5f86' if (i - cx) % 6 == 1 else TRIM[2])
    cv.rect(cx - 13, dt - 7, 26, 1, TRIM[3])
    balls(cv, [(cx + 0.5, dt - 7, 12, 13, 0)], ('#9fa6c4', '#bfc5dc', '#dfe3ef', '#f5f6fb', 'white'),
          outline=TRIM[0], dith=0.6, crease=False)
    cv.rect(cx - 13, dt - 7, 26, 8, KEY) if False else None
    cv.rect(cx - 12, dt - 6, 24, 7, TRIM[2])               # re-cover the lower half of the ellipse
    for i in range(cx - 10, cx + 11, 3):
        cv.vline(i, dt - 5, dt, TRIM[1] if i < cx else TRIM[2])
        cv.vline(i + 1, dt - 5, dt, '#5b5f86' if (i - cx) % 6 == 1 else TRIM[2])
    cv.vline(cx - 12, dt - 6, dt + 1, TRIM[1])
    cv.rect(cx - 13, dt - 7, 26, 1, TRIM[3])
    for k in (-6, 0, 6):                                     # dome ribs
        for yy in range(dt - 19, dt - 7):
            r_ = (yy - (dt - 7)) / 13.0
            xx = cx + int(round(k * math.sqrt(max(0, 1 - r_ * r_))))
            if cv.a[yy, xx].tolist() != list(KEY):
                cv.px(xx, yy, '#c9cee2' if k <= 0 else '#e4e7f2')
    # lantern + finial
    lt = dt - 20
    cv.rect(cx - 2, lt - 5, 5, 6, TRIM[2])
    cv.vline(cx - 2, lt - 5, lt, TRIM[1])
    cv.px(cx, lt - 3, '#5b5f86')
    cv.ellipse(cx, lt - 6, 2, 1, TRIM[3])
    cv.vline(cx, lt - 10, lt - 7, 'gold')
    cv.px(cx, lt - 11, 'butter')
    return lt - 11


@lru_cache(maxsize=1)
def _a_mid():
    w = 380
    cv = new(w)
    gy = 136
    # left rows
    row_house(cv, -4, gy, 26, 3, 1, door_col='#2f6655', box=('#6a3c2c', 'hot_pink', 'pink'))
    row_house(cv, 22, gy, 30, 4, 3, door_col='#2a3a5a', cupola=True)
    row_house(cv, 52, gy, 22, 3, 2, door_col='#b83a46')
    row_house(cv, 74, gy, 20, 3, 4, door_col='#e9b949', box=('#6a3c2c', 'gold', 'butter'))
    domed_hall(cv, DOME_X - 31, gy, 62)
    row_house(cv, 156, gy, 26, 2, 5, door_col='#5b8fcf')
    row_house(cv, 182, gy, 24, 3, 6, door_col='#2f6655')
    row_house(cv, 206, gy, 30, 2, 7, door_col='#b83a46', box=('#6a3c2c', 'hot_pink', 'cotton'))
    row_house(cv, 236, gy, 16, 3, 8, door_col='#2a3a5a')
    # the clock tower hall: two low brick wings
    for wx_ in (SPIRE_X - 30, SPIRE_X + 8):
        brick_wall(cv, wx_, gy - 24, 22, 24, BRICK, seed=wx_)
        cv.rect(wx_ - 1, gy - 25, 24, 2, TRIM[2])
        slate_roof(cv, wx_, gy - 30, 22, 5, inset=3)
        for i in range(3):
            window(cv, wx_ + 3 + i * 7, gy - 19, 4, 8, shutter=None)
        cv.rect(wx_, gy - 3, 22, 3, TRIM[1])
    clock_spire(cv, SPIRE_X, gy)
    row_house(cv, 284, gy, 22, 3, 9, door_col='#2f6655')
    row_house(cv, 306, gy, 18, 2, 10, door_col='#e9b949')
    # second white steeple (church) at the edge of town
    sx = 330
    brick_wall(cv, sx - 6, gy - 40, 12, 40, BRICK, seed=91)
    cv.rect(sx - 7, gy - 42, 14, 2, TRIM[2])
    cv.rect(sx - 5, gy - 52, 10, 10, TRIM[2])
    cv.vline(sx - 5, gy - 52, gy - 43, TRIM[1])
    cv.rect(sx - 2, gy - 50, 4, 6, '#4b5578')
    cv.circle(sx, gy - 30, 3, '#fbf6e8')
    cv.px(sx, gy - 31, 'ink')
    cv.px(sx + 1, gy - 30, 'ink')
    cv.tri((sx - 5, gy - 52), (sx + 5, gy - 52), (sx, gy - 70), TRIM[2])
    cv.line(sx, gy - 70, sx + 4, gy - 53, TRIM[3])
    cv.line(sx, gy - 70, sx - 4, gy - 53, TRIM[1])
    cv.px(sx, gy - 71, 'gold')
    row_house(cv, 338, gy, 20, 2, 11, door_col='#b83a46', roof=True)
    # trees between rows
    for (tx, r, s) in ((358, 9, 31), (372, 7, 32)):
        tree(cv, tx, gy, r, s, th=6, haze=0.15)
    return to_layer(cv)


def bell_sprite_build():
    """The giant bronze bell + its wooden yoke, unrotated. Returns (rgba, pivot_x, pivot_y)."""
    Wd, Hd = 42, 42
    cv = new(Wd, Hd)
    cx = Wd // 2
    py = 3                                     # pivot row (yoke centre)
    # yoke: heavy dark oak beam with iron straps
    cv.rect(cx - 17, py - 2, 35, 5, '#5a3826')
    cv.hline(cx - 17, cx + 17, py - 2, '#8a5c3c')
    cv.hline(cx - 16, cx + 16, py + 2, '#3a2418')
    cv.hline(cx - 15, cx + 15, py, '#4a2e1f')
    cv.px(cx - 17, py - 2, KEY)
    cv.px(cx + 17, py - 2, KEY)
    for sx_ in (cx - 12, cx - 4, cx + 4, cx + 12):
        cv.vline(sx_, py - 2, py + 2, '#4a4e5c')
        cv.px(sx_, py - 1, '#9aa3b4')
    cv.rect(cx - 2, py + 3, 5, 2, '#4a4e5c')    # crown loops
    cv.hline(cx - 1, cx + 1, py + 3, '#7d8494')
    top = py + 5
    BH = 30
    tt = np.array([0, .05, .12, .25, .45, .62, .74, .84, .92, .97, 1.0])
    ww = np.array([4.0, 6.4, 8.2, 9.0, 9.4, 10.1, 11.2, 12.8, 14.4, 15.2, 15.2])
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    t = (yy + 0.5 - top) / BH
    hw = np.interp(t, tt, ww)
    u = (xx + 0.5 - cx - 0.5) / hw
    ins = (t >= 0) & (t <= 1) & (np.abs(u) <= 1)
    nz = np.sqrt(np.clip(1 - u * u, 0, 1))
    ny = np.interp(t, [0, .15, .5, .8, 1], [-.75, -.35, -.1, .25, .55])
    lam = u * _L[0] + ny * _L[1] + nz * _L[2]
    lvl = np.clip(np.floor((lam + 1) / 2 * 7 - 0.6 + (bayer(Hd, Wd) - 0.5) * 0.8), 0, 6).astype(int)
    rr = np.array([col(c_) for c_ in BRONZE], np.uint8)
    out = rr[lvl]
    # raised lip ring + inscription band (rows), darker mouth row
    tr = (yy + 0.5 - top) / BH
    band = ins & (np.abs(tr - 0.22) < 0.025)
    out[band] = rr[np.maximum(lvl[band] - 2, 0)]
    dots = ins & (np.abs(tr - 0.27) < 0.02) & ((xx.astype(int) % 2) == 0)
    out[dots] = rr[np.maximum(lvl[dots] - 1, 0)]
    lip = ins & (np.abs(tr - 0.89) < 0.03)
    out[lip] = rr[np.minimum(lvl[lip] + 1, 6)]
    lip2 = ins & (np.abs(tr - 0.93) < 0.02)
    out[lip2] = rr[np.maximum(lvl[lip2] - 2, 0)]
    a = cv.a
    a[ins] = out[ins]
    # outline
    pad = np.pad(ins, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    bound = ins & ~inner
    a[bound] = C('#2c1810')
    lit = bound & (u > 0.55) & (t < 0.9)
    a[lit] = C(BRONZE[5])
    # mouth: dark inner shadow on the last row
    yb = int(top + BH - 1)
    for x in range(Wd):
        if ins[yb, x]:
            a[yb, x] = C('#24140c')
    # the famous crack: a jagged dark line up from the lip, left of centre
    crack = [(cx - 3, yb - 1), (cx - 3, yb - 2), (cx - 4, yb - 3), (cx - 4, yb - 4), (cx - 3, yb - 5),
             (cx - 3, yb - 6), (cx - 4, yb - 7), (cx - 4, yb - 8), (cx - 3, yb - 9), (cx - 3, yb - 10),
             (cx - 2, yb - 11), (cx - 3, yb - 12)]
    for (x, y) in crack:
        a[y, x] = C('#1e100a')
    for (x, y) in crack[1:9:3]:
        a[y, x + 1] = C(BRONZE[1])
    # specular glints on the lit shoulder
    a[top + 4, cx + 4] = C(BRONZE[6])
    a[top + 5, cx + 4] = C(BRONZE[6])
    a[top + 12, cx + 6] = C(BRONZE[6])
    rgba = np.zeros((Hd, Wd, 4), np.uint8)
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    rgba[..., :3] = a
    rgba[..., 3] = m * 255
    return rgba, cx + 0.5, py + 0.5


_BELL = None


@lru_cache(maxsize=None)
def _bell_rot(deg):
    global _BELL
    if _BELL is None:
        _BELL = bell_sprite_build()
    spr, spx, spy = _BELL
    sh, sw = spr.shape[:2]
    S = 60
    out = np.zeros((S, S, 4), np.uint8)
    opx, opy = S / 2, 6.0
    th = math.radians(deg)
    cs, sn = math.cos(th), math.sin(th)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx = xx + 0.5 - opx
    dy = yy + 0.5 - opy
    lx = cs * dx + sn * dy
    ly = -sn * dx + cs * dy
    sx = np.floor(lx + spx).astype(int)
    sy = np.floor(ly + spy).astype(int)
    ok = (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)
    out[ok] = spr[sy[ok], sx[ok]]
    return out, opx, opy


def bell_angle(f):
    """Swing angle (degrees). Extremes on the beats: +max at 108 (the strike), -max at 90 / 126."""
    if f < 82:
        return 0.0
    env = 15.0 * smooth(seg(f, 82, 106))
    if f > 108:
        env = 15.0 * math.exp(-(f - 108) / 34.0)
    return env * math.sin(math.pi * (f - 99) / 18.0)


def draw_bell(c, x, py, f):
    deg = int(round(bell_angle(f)))
    spr, opx, opy = _bell_rot(deg)
    # clapper hangs plumb: peeks out of the mouth when the bell is tilted
    cl_y = py + 34
    cl_x = x + int(round(math.sin(math.radians(deg) * 0.35) * 2))
    c.vline(cl_x, py + 24, cl_y, '#3a3a44')
    c.circle(cl_x, cl_y + 1, 2, '#3e4250')
    c.px(cl_x + 1, cl_y, '#7d8494')
    c.blit(spr, x - int(opx), py - int(opy))


def sound_rings(c, cx, cy, f, t0, strength=1.0, n=3):
    """Expanding concentric sound arcs '(( ))' from a struck bell."""
    if f < t0 or f > t0 + 34:
        return
    x0, y0, x1, y1 = max(0, cx - 80), max(0, cy - 60), min(W, cx + 81), min(H, cy + 61)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dxx = xx - cx
    dyy = yy - cy
    d = np.sqrt(dxx * dxx + dyy * 1.35 * dyy * 1.35)
    side = np.abs(dxx) > np.abs(dyy) * 1.1
    thr = bayer(y1 - y0, x1 - x0, x0, y0)
    reg = c.a[y0:y1, x0:x1]
    for i in range(n):
        tt = f - t0 - i * 5
        if tt < 0:
            continue
        r = 21 + tt * 2.6
        a = (1 - tt / (32 - i * 3)) * strength
        if a <= 0:
            continue
        ring = (np.abs(d - r) < 1.0) & side & (thr < a + 0.25)
        ring2 = (np.abs(d - r + 2.0) < 0.75) & side & (thr < a * 0.8)
        reg[ring2] = C('#f9c22b')
        reg[ring] = C('#fffbe0')


def pavilion_back(cv):
    """Parts of the pavilion that sit behind the bell: ceiling shadow, back columns."""
    x0, x1 = PAV_L, PAV_R
    for bx in (x0 + 12, x1 - 16):                         # back columns in shade
        cv.rect(bx, 90, 4, 52, '#b9b5cf')
        cv.vline(bx, 90, 141, '#9a96b4')
        cv.rect(bx - 1, 89, 6, 2, '#c9c6db')
        cv.rect(bx - 1, 138, 6, 3, '#c9c6db')


def pavilion_front(cv):
    """Open white bell pavilion (podium, steps, corner columns, entablature, pediment, cupola)."""
    x0, x1 = PAV_L, PAV_R
    cx = BELL_X
    # podium + steps
    for i, (dy, ins) in enumerate(((142, 0), (145, -2), (148, -4))):
        cv.rect(x0 + ins, dy, x1 - x0 - 2 * ins, 3 if i < 2 else 2, TRIM[2])
        cv.hline(x0 + ins, x1 - 1 - ins, dy, TRIM[3])
        cv.vline(x0 + ins, dy, dy + 2, TRIM[1])
    cv.rect(x0 + 1, 140, x1 - x0 - 2, 2, TRIM[1])
    # low balustrade between the columns
    for bx in range(x0 + 8, x1 - 7, 3):
        cv.vline(bx, 134, 139, TRIM[2])
        cv.px(bx, 136, TRIM[3])
        cv.px(bx - 1, 137, TRIM[1]) if bx > x0 + 8 else None
    cv.hline(x0 + 6, x1 - 7, 133, TRIM[3])
    cv.hline(x0 + 6, x1 - 7, 134, TRIM[1])
    # front corner columns (fluted, Doric)
    for bx in (x0 + 2, x1 - 8):
        cv.rect(bx, 88, 6, 52, TRIM[2])
        cv.vline(bx, 88, 139, TRIM[0])
        cv.vline(bx + 1, 88, 139, TRIM[1])
        cv.vline(bx + 3, 88, 139, TRIM[1])
        cv.vline(bx + 4, 88, 139, TRIM[3])
        cv.vline(bx + 5, 88, 139, TRIM[2])
        cv.rect(bx - 1, 86, 8, 2, TRIM[2])
        cv.hline(bx - 1, bx + 6, 86, TRIM[3])
        cv.rect(bx - 1, 137, 8, 3, TRIM[2])
        cv.hline(bx - 1, bx + 6, 137, TRIM[3])
    # entablature
    cv.rect(x0 - 2, 78, x1 - x0 + 4, 9, TRIM[2])
    cv.hline(x0 - 2, x1 + 1, 78, TRIM[3])
    cv.rect(x0, 81, x1 - x0, 3, '#bfe3ff')                   # frieze in pastel blue with gold stars
    for sx_ in range(x0 + 4, x1 - 2, 8):
        cv.px(sx_, 82, 'gold')
    cv.hline(x0 - 2, x1 + 1, 86, TRIM[1])
    cv.hline(x0 - 3, x1 + 2, 77, TRIM[2])
    cv.hline(x0 - 2, x1 + 1, 87, '#6b6884')                 # ceiling shadow line
    # pediment
    cv.tri((x0 - 3, 77), (x1 + 3, 77), (cx, 62), TRIM[2])
    cv.line(cx, 62, x1 + 3, 77, TRIM[3])
    cv.line(cx, 62, x0 - 3, 77, TRIM[1])
    cv.tri((x0 + 6, 75), (x1 - 6, 75), (cx, 66), '#e2e0ec')
    # gold sun emblem in the tympanum
    cv.circle(cx, 71, 2, 'gold')
    cv.px(cx + 1, 70, 'butter')
    for (dx, dy) in ((-4, 0), (4, 0), (0, -3), (-3, -2), (3, -2)):
        cv.px(cx + dx, 71 + dy, 'amber')
    # cupola: little round temple with a dome
    cv.rect(cx - 6, 54, 12, 9, TRIM[2])
    cv.vline(cx - 6, 54, 62, TRIM[1])
    cv.vline(cx + 5, 54, 62, TRIM[3])
    for ax in (cx - 4, cx + 1):
        cv.rect(ax, 56, 3, 5, '#5b6894')
        cv.px(ax + 2, 56, '#7d8bb8')
    cv.hline(cx - 7, cx + 6, 53, TRIM[3])
    balls(cv, [(cx + 0.5, 53, 6, 6, 0)], ('#6aa79a', '#7fc4b4', '#a7e0cc', '#d4f4e6'),
          outline='#3f7a6e', dith=0.5, crease=False)
    cv.rect(cx - 7, 53, 14, 1, TRIM[3])
    cv.vline(cx, 41, 46, 'gold')
    cv.px(cx, 40, 'butter')
    cv.px(cx - 1, 42, 'amber')
    cv.px(cx + 1, 42, 'amber')
    # iron hangers for the yoke
    for hx in (cx - 12, cx + 12):
        cv.vline(hx, 87, 91, '#4a4e5c')
        cv.px(hx, 92, '#7d8494')


def lamp_post(cv, x, gy, h=26):
    cv.rect(x - 1, gy - 3, 3, 3, '#3a3248')
    cv.vline(x, gy - h, gy - 3, '#3a3248')
    cv.px(x + 1, gy - h + 4, '#5a5272')
    cv.rect(x - 2, gy - h - 6, 5, 6, '#fff3c4')
    cv.frame(x - 3, gy - h - 7, 7, 8, '#3a3248')
    cv.vline(x, gy - h - 6, gy - h - 1, '#3a3248')
    cv.px(x + 1, gy - h - 5, 'white')
    cv.hline(x - 2, x + 2, gy - h - 8, '#3a3248')
    cv.px(x, gy - h - 9, 'gold')


def bench(cv, x, gy):
    cv.rect(x, gy - 6, 12, 2, '#9b6a4c')
    cv.hline(x, x + 11, gy - 6, '#c8966a')
    cv.rect(x, gy - 10, 12, 1, '#9b6a4c')
    cv.rect(x, gy - 8, 12, 1, '#9b6a4c')
    for lx in (x + 1, x + 10):
        cv.vline(lx, gy - 10, gy - 1, '#3a3248')


def planter(cv, x, gy, w, flowers=('hot_pink', 'pink', 'gold', 'white'), seed=0):
    brick_wall(cv, x, gy - 5, w, 5, PLAZA, seed=seed, near=True)
    cv.hline(x, x + w - 1, gy - 5, TRIM[2])
    R = rng(seed)
    for i in range(w):
        h_ = 2 + int(R.random() * 3)
        cv.vline(x + i, gy - 5 - h_, gy - 6, GRASS[1] if i % 2 else GRASS[2])
        if R.random() < 0.55:
            cv.px(x + i, gy - 6 - h_, flowers[int(R.random() * len(flowers))])


def plaza(cv, x0, x1, gy=GY):
    """Brick-paved plaza from the ground line to the bottom (running bond, kerb at the front)."""
    brick_wall(cv, x0, gy, x1 - x0, H - gy, PLAZA, seed=4, near=True)
    cv.hline(x0, x1 - 1, gy, PLAZA[4])
    # pale granite banding (perspective courses) + a marble walkway to the pavilion steps
    for yy in (156, 163):
        cv.hline(x0, x1 - 1, yy, '#d9b9a6')
        cv.hline(x0, x1 - 1, yy + 1, PLAZA[0])
    wl, wr = BELL_X - 16, BELL_X + 16
    cv.poly([(wl, gy), (wr, gy), (wr + 9, 171), (wl - 9, 171)], '#e9e4ee')
    for yy in range(gy + 2, 171, 3):
        k = (yy - gy) * 9 // 21
        cv.hline(wl - k, wr + k - 1, yy, '#cfc8dc')
        for xx in range(wl - k + (yy // 3) % 2 * 3, wr + k, 6):
            cv.px(xx, yy - 1, '#cfc8dc')
    cv.line(wl, gy, wl - 9, 170, '#b7b0c8')
    cv.line(wr - 1, gy, wr + 8, 170, 'white')
    # kerb with little granite bollards
    cv.rect(x0, 171, x1 - x0, 2, '#d8d2dc')
    cv.hline(x0, x1 - 1, 171, 'white')
    for bx in range(x0 + 6, x1, 22):
        if wl - 12 < bx < wr + 12:
            continue
        cv.rect(bx, 167, 3, 4, '#c4c2d6')
        cv.px(bx + 2, 167, 'white')
        cv.vline(bx, 167, 170, '#8f8aa8')
    cv.rect(x0, 173, x1 - x0, H - 173, '#5a9a52')
    cv.hline(x0, x1 - 1, 173, '#3f7a46')
    R = rng(17)
    for i in range((x1 - x0) // 3):
        gx = x0 + int(R.random() * (x1 - x0))
        cv.px(gx, 174 + int(R.random() * 5), ('#8ccb64', '#3f7a46', '#bfe27a')[i % 3])


def big_building(cv, x, gy, w):
    """Near Georgian hall at the left (2.5 storeys)."""
    top = gy - 54
    brick_wall(cv, x, top, w, gy - top, BRICK, seed=40, near=True)
    cv.vline(x + w - 1, top, gy - 1, BRICK[4])
    cv.rect(x, gy - 4, w, 4, TRIM[1])
    cv.hline(x, x + w - 1, gy - 4, TRIM[2])
    cv.hline(x, x + w - 1, gy - 27, TRIM[1])
    cv.hline(x, x + w - 1, gy - 28, TRIM[2])
    for i in range(6):
        wx = x + 6 + i * 18
        if wx + 8 > x + w - 2:
            break
        window(cv, wx, top + 8, 8, 13)
        if i == 3:
            door(cv, wx - 1, gy - 4, w=10, h=14, color='#2f6655')
            cv.rect(wx - 3, gy - 22, 14, 2, TRIM[2])
        else:
            window(cv, wx, gy - 22, 8, 13, box=('#6a3c2c', 'hot_pink', 'cotton', 'gold'))
    # cornice + roof with dormers
    cv.rect(x - 2, top - 3, w + 4, 4, TRIM[2])
    cv.hline(x - 2, x + w + 1, top - 3, TRIM[3])
    for i in range(x - 1, x + w + 1, 2):
        cv.px(i, top, TRIM[0])
    slate_roof(cv, x - 1, top - 13, w + 2, 10, inset=6)
    for i in range(3):
        dx_ = x + 16 + i * 34
        cv.rect(dx_, top - 11, 9, 8, TRIM[2])
        cv.tri((dx_ - 1, top - 11), (dx_ + 9, top - 11), (dx_ + 4, top - 15), TRIM[2])
        cv.line(dx_ + 4, top - 15, dx_ + 9, top - 11, TRIM[3])
        cv.rect(dx_ + 2, top - 9, 5, 6, GLASS[1])
        cv.px(dx_ + 5, top - 9, GLASS[2])
        cv.vline(dx_ + 4, top - 9, top - 4, TRIM[2])
    chimney(cv, x + 6, top - 20, top - 6)
    chimney(cv, x + w - 12, top - 20, top - 6)
    return top


@lru_cache(maxsize=1)
def _a_near():
    w = 390
    cv = new(w)
    plaza(cv, 0, w)
    big_building(cv, -6, GY, 122)
    # cast shadows on the plaza (sun upper right -> shadows fall left)
    sh = '#9a5a4c'
    cv.poly([(PAV_L - 4, GY), (PAV_R - 30, GY), (PAV_R - 44, GY + 9), (PAV_L - 26, GY + 9)], sh)
    pavilion_back(cv)
    # trees, fountain area, lamps
    tree(cv, 140, GY - 2, 15, 51, th=12)
    cv.ellipse(132, GY + 4, 14, 3, sh)
    tree(cv, 270, GY - 2, 17, 52, th=13)
    cv.ellipse(260, GY + 4, 15, 3, sh)
    tree(cv, 356, GY - 2, 14, 53, th=11)
    cv.ellipse(348, GY + 4, 13, 3, sh)
    planter(cv, 92, GY, 22, seed=7)
    planter(cv, 132, GY, 18, seed=9)
    lamp_post(cv, 164, GY)
    lamp_post(cv, 246, GY)
    lamp_post(cv, 330, GY)
    bench(cv, 352, GY)
    # flagpole
    cv.vline(254, 76, GY - 1, '#c4c2d6')
    cv.vline(255, 76, GY - 1, '#8f8aa8')
    cv.rect(252, GY - 3, 6, 3, TRIM[1])
    cv.circle(254, 75, 1, 'gold')
    # low brick wall + hedge to the right (fields beyond)
    brick_wall(cv, 286, GY - 8, w - 286, 8, BRICK, seed=12, near=True)
    cv.rect(286, GY - 9, w - 286, 2, TRIM[2])
    cv.hline(286, w - 1, GY - 9, TRIM[3])
    pavilion_front(cv)
    return to_layer(cv)


def fountain_dyn(c, x, gy, f):
    """Small round plaza fountain in front of the tree (animated jets)."""
    c.rect(x - 14, gy - 6, 29, 6, TRIM[2])
    c.hline(x - 14, x + 14, gy - 6, 'white')
    c.vline(x - 14, gy - 6, gy - 1, TRIM[1])
    c.vline(x + 14, gy - 6, gy - 1, TRIM[3])
    c.hline(x - 13, x + 13, gy - 2, TRIM[1])
    c.hline(x - 13, x + 13, gy - 7, '#7fc0e8')
    c.hline(x - 13, x + 13, gy - 8, '#a9dcf6')
    c.rect(x - 1, gy - 15, 3, 8, TRIM[2])
    c.vline(x - 1, gy - 15, gy - 8, TRIM[1])
    c.rect(x - 4, gy - 16, 9, 2, TRIM[3])
    c.hline(x - 4, x + 4, gy - 14, TRIM[1])
    for i in range(14):
        ph = (f * 0.09 + i / 14.0) % 1.0
        side = -1 if i % 2 else 1
        dx = side * ph * 11
        dy = -8 * (1 - (2 * ph - 1) ** 2)
        c.px(x + int(round(dx)), gy - 18 + int(round(dy + ph * 10)), 'white' if ph < 0.45 else '#bfe3ff')
    for k in range(4):
        yy = gy - 18 - k - (f // 3 + k) % 2
        c.px(x, yy, 'white' if k < 3 else '#bfe3ff')
    for k in range(3):
        c.px(x - 10 + (f // 3 + k * 7) % 21, gy - 8, 'white')


@lru_cache(maxsize=1)
def _a_fg():
    w = 400
    cv = new(w)
    # wrought-iron railings with gold finials + hedges at the corners
    for (x0, x1) in ((0, 70), (330, 400)):
        cv.hline(x0, x1, 166, '#3a3248')
        cv.hline(x0, x1, 175, '#3a3248')
        for x in range(x0, x1, 4):
            cv.vline(x, 163, H - 1, '#3a3248')
            cv.px(x, 162, 'gold')
            cv.px(x + 1, 165, '#5a5272')
    R = rng(8)
    for (bx, r) in ((18, 9), (44, 8), (352, 9), (380, 10)):
        balls(cv, canopy_blobs(bx, 176, r, int(R.random() * 99), squash=0.7), LEAF, outline=LEAF_OUT,
              rim=LEAF[-1])
        for k in range(5):
            fx_ = bx - r + int(R.random() * 2 * r)
            fy_ = 170 + int(R.random() * 8)
            cv.px(fx_, fy_, ('hot_pink', 'white', 'gold', 'pink')[k % 4])
    return to_layer(cv)


def a_balloon(c, f, cam):
    bx = BELL_X + 10 - int(round(cam * 0.8)) + int(round(5 * seg(f, 72, 179)))
    by = 31 + int(round(math.sin(f / 31.0) * 1.5))
    draw_balloon(c, bx, by, f, size='m')


def tower_glints(c, f, go):
    """A sun glint sliding up the lit glass faces."""
    for (gx, top, w, ph) in ((92, 34, 18, 0), (286, 40, 20, 40), (22, 46, 15, 20)):
        t = ((f + ph) % 80) / 80.0
        yy = int(130 - t * (130 - top))
        x = gx + max(3, w // 3) + 1 - go
        if 0 <= x < W and t < 0.85:
            c.hline(x, x + w - max(3, w // 3) - 3, yy, '#f2fbff')
            c.px(x + (w - max(3, w // 3)) // 2, yy - 1, 'white')


def render_a(f):
    cam = a_cam(f)
    c = _sky().copy()
    draw_clouds(c, f, cam, A_CLOUDS)
    comp(c, _a_fields(), int(round(cam * FIELD_P)))
    go = int(round(cam * GLASS_P))
    comp(c, _a_glass(), go)
    tower_glints(c, f, go)
    mo = int(round(cam * MID_P))
    comp(c, _a_mid(), mo)
    # rooftop flags + chimney smoke on the mid layer
    flag(c, DOME_X + 22 - mo, 92, f, w=7, h=4, cols=('#bfe3ff', 'white'), seed=1)
    flag(c, SPIRE_X + 22 - mo, 98, f, w=6, h=4, cols=('#ff9fd0', 'white'), seed=2)
    smoke(c, 88 - mo, 92, f, seed=3, n=4, height=14)
    smoke(c, 160 - mo, 102, f, seed=4, n=4, height=12)
    smoke(c, 288 - mo, 98, f, seed=5, n=4, height=12)
    a_balloon(c, f, cam)
    near = _a_near()
    comp(c, near, cam)
    # the bell (drawn over the near layer, under the entablature)
    bx = BELL_X - cam
    draw_bell(c, bx, BELL_PIVOT, f)
    # re-draw the entablature strip over the yoke ends so the bell hangs *inside* the pavilion
    a, m = near
    xs0, xs1 = max(0, PAV_L - 3 - cam), min(W, PAV_R + 3 - cam)
    if xs1 > xs0:
        sub = a[77:88, xs0 + cam:xs1 + cam]
        ms = m[77:88, xs0 + cam:xs1 + cam]
        np.copyto(c.a[77:88, xs0:xs1], sub, where=ms[..., None])
    flag(c, 253 - cam, 78, f, w=12, h=7, cols=('#ff9fd0', 'white'), emblem='gold', seed=4, pole=False)
    fountain_dyn(c, 318 - cam, GY + 3, f)
    if f >= 108:
        sound_rings(c, bx, BELL_PIVOT + 16, f, 108, 1.0, 3)
    if f >= 126:
        sound_rings(c, bx, BELL_PIVOT + 16, f, 126, 0.55, 2)
    if 108 <= f <= 112:
        fx.sparkle(c, bx + 5, BELL_PIVOT + 12, f - 108, 'white', period=10, size=3)
    fo = int(round(cam * FG_P))
    comp(c, _a_fg(), fo)
    fx.dust_motes(c, f, (150, 50, 160, 90), n=10, color='#fff6cf', seed=4)
    # callouts + card
    callout(c, SPIRE_X - mo, 50, 'FILLYDELPHIA', f, 90, 162, side='up', length=14)
    callout(c, bx - 14, BELL_PIVOT + 10, 'THE BIG BELL', f, 108, 180, side='left', length=26)
    if 108 <= f <= 111:                       # little shake on the strike
        dy = (1, -1, 1, 0)[f - 108]
        if dy > 0:
            c.a[dy:] = c.a[:-dy].copy()
        elif dy < 0:
            c.a[:dy] = c.a[-dy:].copy()
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), BELL_X - cam, 106)
    return c


# ================================================================== SHOT B: BALTIMARE
B_CAM = 72
WL = 134                               # harbour waterline (quay edge)
B_FAR_P, B_MID_P, B_WF_P, B_FG_P = 0.15, 0.4, 0.7, 1.35
CIVIC_X = 150                          # mid-layer x of the civic dome
B_CLOUDS = [(10, 26, 62, 20, 21, 0.05), (130, 12, 50, 16, 22, 0.06), (214, 40, 70, 22, 23, 0.04),
            (330, 20, 44, 14, 24, 0.07)]
WATER = ('#1f527c', '#2a6592', '#3a7fae', '#5aa0c8', '#9fd2ec', '#e8f8ff')
CRANE = ('#4e1a1c', '#7e2a27', '#a83c33', '#d0604a')
WOOD = ('#3b2a26', '#5e3f30', '#83593e', '#a8784f', '#c9a070')
PASTEL_RAMPS = (
    ('#a0587a', '#c06d90', '#df8fae', '#f2b3c9', '#ffd6e6'),      # pink
    ('#3f8a78', '#5aa891', '#7fc4ad', '#a9e0c9', '#d4f4e6'),      # mint
    ('#5a76b8', '#7493d0', '#94b2e6', '#b9d2f6', '#ddeeff'),      # baby blue
    ('#b88a3c', '#d4a852', '#ecc46e', '#f8dd96', '#fff1c8'),      # butter
)


def b_cam(f):
    return int(round(B_CAM * cruise(seg(f, 180, 269))))


def pale_tower(cv, x, gy, w, h, cone='#5b8fcf', haze=0.0, finial='gold'):
    """Round pale stone tower (shaded left, lit right) with a blue cone roof."""
    st = [hz(c_, haze) for c_ in ('#b6b2c8', '#d6d3e2', '#efedf5', 'white')]
    cn = [hz(c_, haze) for c_ in (shade(cone, 0.7), cone, mix(cone, 'white', 0.35), mix(cone, 'white', 0.7))]
    top = gy - h
    cv.rect(x, top, w, h, st[1])
    cv.rect(x + w // 2, top, w - w // 2, h, st[2])
    cv.vline(x, top, gy - 1, st[0])
    cv.vline(x + w - 1, top, gy - 1, st[3])
    for yy in range(top + 6, gy - 4, 9):
        cv.rect(x + w // 2 - 1, yy, 2, 4, hz(GLASS[0], haze))
        cv.px(x + w // 2, yy, hz(GLASS[2], haze))
    cv.hline(x - 1, x + w, top, st[3])
    ch = int(w * 1.3) + 3
    cv.tri((x - 1, top), (x + w // 2, top), (x + w // 2, top - ch), cn[1])
    cv.tri((x + w // 2, top), (x + w + 1, top), (x + w // 2, top - ch), cn[2])
    cv.line(x + w // 2, top - ch, x + w, top - 1, cn[3])
    cv.line(x + w // 2, top - ch, x - 1, top - 1, cn[0])
    cv.vline(x + w // 2, top - ch - 3, top - ch - 1, hz(finial, haze))
    return top - ch - 3


def gold_tower(cv, x, gy, w=14, h=62):
    """Square stone tower with a clock and a gold pyramid roof (Baltimare's gold-roofed tower)."""
    top = gy - h
    cv.rect(x, top, w, h, TRIM[2])
    cv.vline(x, top, gy - 1, TRIM[1])
    cv.vline(x + w - 1, top, gy - 1, TRIM[3])
    for yy in range(top + 18, gy - 4, 7):
        window(cv, x + 3, yy, 3, 5, shutter=None, sill=False, muntin=False)
        window(cv, x + w - 6, yy, 3, 5, shutter=None, sill=False, muntin=False)
    cv.circle(x + w // 2, top + 9, 4, 'gold')
    cv.circle(x + w // 2, top + 9, 3, '#fbf6e8')
    cv.vline(x + w // 2, top + 7, top + 9, 'ink')
    cv.hline(x + w // 2, x + w // 2 + 2, top + 9, 'ink')
    cv.rect(x - 2, top - 2, w + 4, 3, TRIM[2])
    cv.hline(x - 2, x + w + 1, top - 2, TRIM[3])
    for i in range(x - 1, x + w + 1, 2):
        cv.px(i, top - 3, TRIM[2])
    gh = w + 6
    cv.tri((x - 1, top - 3), (x + w // 2, top - 3), (x + w // 2, top - 3 - gh), '#c98a1e')
    cv.tri((x + w // 2, top - 3), (x + w + 1, top - 3), (x + w // 2, top - 3 - gh), '#f9c22b')
    cv.line(x + w // 2, top - 3 - gh, x + w, top - 4, '#fff1a8')
    for yy in range(top - gh + 2, top - 3, 3):
        k = (yy - (top - 3 - gh)) * (w // 2 + 1) // gh
        cv.px(x + w // 2 - k + 1, yy, '#a8701a')
    cv.vline(x + w // 2, top - gh - 8, top - gh - 4, 'gold')
    cv.px(x + w // 2, top - gh - 9, 'butter')
    return top - gh - 9


def civic_hall(cv, cx, gy):
    """Baltimare's domed white civic hall: wings with colonnades, central portico, drum, big dome."""
    WHT = ('#a9a5c2', '#cfcde0', '#ecebf4', 'white')
    hw = 52
    top = gy - 30
    x0 = cx - hw
    dt = top - 17                       # drum base
    # dome first (the drum and the portico are drawn over its lower part)
    dome_c = dt - 12
    balls(cv, [(cx + 0.5, dome_c, 15, 15, 0)], ('#8e8bb0', '#aeabca', '#cfcde2', '#e9e8f3', 'white'),
          outline='#7d7a9e', dith=0.6, crease=False)
    for k in (-9, -3, 3, 9):                              # ribs
        for yy in range(dome_c - 15, dome_c):
            r_ = (yy - dome_c) / 15.0
            xx = cx + int(round(k * math.sqrt(max(0, 1 - r_ * r_))))
            if 0 <= xx < cv.w and tuple(cv.a[yy, xx]) != KEY:
                cv.px(xx, yy, '#b9b7d2' if k < 0 else '#f6f5fb')
    # drum with columns
    cv.rect(cx - 17, dt - 11, 34, 12, WHT[2])
    cv.vline(cx - 17, dt - 11, dt, WHT[0])
    cv.vline(cx + 16, dt - 11, dt, WHT[3])
    for i in range(cx - 15, cx + 16, 4):
        cv.rect(i + 1, dt - 9, 2, 7, '#7b7aa0' if i < cx else '#9493b6')
        cv.vline(i, dt - 10, dt - 1, WHT[3] if i > cx else WHT[2])
    cv.rect(cx - 18, dt - 12, 36, 2, WHT[3])
    cv.hline(cx - 18, cx + 17, dt - 11, WHT[1])
    # lantern + finial
    lt = dome_c - 15
    cv.rect(cx - 3, lt - 5, 7, 6, WHT[2])
    cv.vline(cx - 3, lt - 5, lt, WHT[1])
    cv.rect(cx - 1, lt - 4, 2, 4, '#5b6894')
    cv.ellipse(cx, lt - 6, 3, 2, WHT[3])
    cv.vline(cx, lt - 12, lt - 8, 'gold')
    cv.circle(cx, lt - 10, 1, 'gold')
    cv.px(cx, lt - 13, 'butter')
    # stairs + plinth
    for i in range(3):
        cv.rect(x0 - 3 + i * 2, gy - 2 - i * 2, 2 * hw + 6 - i * 4, 2, WHT[2])
        cv.hline(x0 - 3 + i * 2, x0 + 2 * hw + 2 - i * 2, gy - 2 - i * 2, WHT[3])
    gy2 = gy - 6
    cv.rect(x0, top, 2 * hw, gy2 - top, WHT[2])
    cv.vline(x0, top, gy2 - 1, WHT[0])
    cv.vline(x0 + 2 * hw - 1, top, gy2 - 1, WHT[3])
    # wing colonnades (deep shadowed loggia behind the columns)
    cv.rect(x0 + 3, top + 5, 2 * hw - 6, gy2 - top - 6, '#8d89aa')
    for wx in range(x0 + 5, x0 + 2 * hw - 5, 6):
        window(cv, wx + 1, top + 8, 3, 6, frame=WHT, shutter=None, sill=False, muntin=False)
    for colx in range(x0 + 3, x0 + 2 * hw - 3, 6):
        cv.rect(colx, top + 4, 3, gy2 - top - 4, WHT[2])
        cv.vline(colx, top + 4, gy2 - 1, WHT[1])
        cv.vline(colx + 2, top + 4, gy2 - 1, WHT[3])
        cv.rect(colx - 1, top + 3, 5, 1, WHT[3])
    cv.rect(x0 - 2, top - 2, 2 * hw + 4, 5, WHT[2])
    cv.hline(x0 - 2, x0 + 2 * hw + 1, top - 2, WHT[3])
    cv.hline(x0 - 2, x0 + 2 * hw + 1, top + 2, WHT[0])
    for i in range(x0 - 1, x0 + 2 * hw + 1, 2):
        cv.px(i, top + 1, WHT[1])
    for i in range(x0, x0 + 2 * hw, 4):                   # roof balustrade
        cv.vline(i, top - 5, top - 3, WHT[2])
    cv.hline(x0, x0 + 2 * hw - 1, top - 6, WHT[3])
    # central portico (pediment)
    pw = 30
    px0 = cx - pw // 2
    cv.rect(px0 - 2, top - 8, pw + 4, 6, WHT[2])
    cv.hline(px0 - 2, px0 + pw + 1, top - 8, WHT[3])
    cv.tri((px0 - 3, top - 8), (px0 + pw + 3, top - 8), (cx, top - 17), WHT[2])
    cv.line(cx, top - 17, px0 + pw + 3, top - 8, WHT[3])
    cv.line(cx, top - 17, px0 - 3, top - 8, WHT[1])
    cv.tri((px0 + 4, top - 10), (px0 + pw - 4, top - 10), (cx, top - 14), '#dddbe9')
    cv.circle(cx, top - 11, 1, 'gold')
    for colx in range(px0, px0 + pw, 5):
        cv.rect(colx, top - 2, 3, gy2 - top + 2, WHT[3])
        cv.vline(colx, top - 2, gy2 - 1, WHT[1])
    cv.rect(cx - 3, gy2 - 12, 7, 12, '#4b5578')
    cv.hline(cx - 3, cx + 3, gy2 - 12, '#68739a')
    return lt - 13


@lru_cache(maxsize=1)
def _b_far():
    w = 340
    cv = new(w)
    xs = np.arange(w)
    top = 106 + 6 * np.sin(xs / 41.0 + 0.5) + 3 * np.sin(xs / 17.0)
    for x in range(w):
        t = int(top[x])
        cv.vline(x, t, WL, hz('#7fae86', 0.6))
        cv.px(x, t, hz('#b9d8a0', 0.55))
    for i in range(10):
        tx = 8 + i * 34 + (i * 13) % 11
        tree(cv, tx, int(top[min(w - 1, tx)]) + 4, 4, 400 + i, th=2, haze=0.6)
    # far pale towers with blue cones
    for (x, h, ww_, cc) in ((30, 34, 6, '#5b8fcf'), (44, 26, 5, '#7aa7e0'), (250, 30, 6, '#5b8fcf'),
                            (262, 22, 5, '#8fb6e8'), (300, 28, 5, '#5b8fcf')):
        pale_tower(cv, x, int(top[x]) + 6, ww_, h, cone=cc, haze=0.5)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _b_mid():
    w = 360
    cv = new(w)
    xs = np.arange(w)
    # grassy rise under the civic hall
    hill = 118 - 14 * np.exp(-((xs - CIVIC_X) / 70.0) ** 2) + 1.5 * np.sin(xs / 7.0)
    for x in range(w):
        t = int(hill[x])
        cv.vline(x, t, WL, hz(GRASS[2], 0.25))
        cv.px(x, t, hz(GRASS[4], 0.25))
        if x % 3 == 0:
            cv.px(x, t + 1, hz(GRASS[3], 0.25))
    pale_tower(cv, 62, 112, 9, 54, cone='#4d7fc4', haze=0.12)
    pale_tower(cv, 78, 114, 7, 40, cone='#6f9fdc', haze=0.15)
    gold_tower(cv, 214, 112, 14, 64)
    pale_tower(cv, 240, 116, 8, 44, cone='#4d7fc4', haze=0.15)
    civic_hall(cv, CIVIC_X, 108)
    # trees on the rise
    for (tx, r, sd) in ((92, 8, 61), (104, 7, 62), (196, 8, 63), (206, 6, 64), (28, 7, 65), (282, 8, 66),
                        (300, 7, 67), (330, 9, 68)):
        tree(cv, tx, int(hill[tx]) + 2, r, sd, th=4, haze=0.15)
    # pastel painted row houses down the slope
    x = 0
    i = 0
    for (hx, hw_, fl) in ((0, 22, 3), (22, 18, 2), (40, 20, 3), (252, 20, 3), (272, 18, 2), (290, 22, 3),
                          (312, 20, 2), (332, 26, 3)):
        row_house(cv, hx, 124, hw_, fl, 200 + i, ramp=PASTEL_RAMPS[i % 4], door_col='#f4efe4',
                  dormers=False)
        i += 1
    return to_layer(cv)


ANCHOR = ['..x..', '.xxx.', '..x..', 'x.x.x', '.xxx.']
BARREL = ['.xxx.', 'x.x.x', 'xxxxx', 'x.x.x', '.xxx.']


def icon(cv, x, y, pat, color):
    for j, row in enumerate(pat):
        for i, ch in enumerate(row):
            if ch == 'x':
                cv.px(x + i, y + j, color)


def warehouse(cv, x, gy, w, h, seed, sign=ANCHOR, ramp=BRICK):
    """Red-brick harbour warehouse with a gable front, arched windows, hoist beam and an icon sign."""
    top = gy - h
    brick_wall(cv, x, top, w, h, ramp, seed=seed)
    cv.vline(x, top, gy - 1, ramp[0])
    cv.vline(x + w - 1, top, gy - 1, ramp[4])
    gh = w * 3 // 8
    apex = (x + w // 2, top - gh)
    # gable brick + white bargeboards + slate roof sliver
    for yy in range(top - gh, top):
        k = (yy - (top - gh)) * (w // 2) // gh
        brick_wall(cv, x + w // 2 - k, yy, 2 * k + 1, 1, ramp, seed=seed + 1)
    cv.line(apex[0], apex[1] - 2, x - 3, top + 1, SLATE[1])
    cv.line(apex[0], apex[1] - 2, x + w + 2, top + 1, SLATE[3])
    cv.line(apex[0], apex[1] - 1, x - 2, top + 1, TRIM[2])
    cv.line(apex[0], apex[1] - 1, x + w + 1, top + 1, TRIM[3])
    cv.circle(apex[0], top - gh // 2 + 1, 2, TRIM[2])
    cv.px(apex[0], top - gh // 2 + 1, GLASS[0])
    # hoist beam + pulley + rope
    cv.hline(apex[0], apex[0] + 5, apex[1] + 3, WOOD[1])
    cv.px(apex[0] + 5, apex[1] + 4, '#4a4e5c')
    cv.vline(apex[0] + 5, apex[1] + 5, top + 8 + seed % 6, '#c9a070')
    # floors of arched windows
    fl = max(2, (h - 12) // 11)
    nwin = max(2, (w - 4) // 7)
    for j in range(fl):
        wy = top + 4 + j * 11
        for i in range(nwin):
            wx = x + 3 + i * ((w - 6) // nwin) + 1
            window(cv, wx, wy, 4, 7, frame=TRIM, glass=('#28465a', '#3b6178', '#7fb0c8', '#d6f0ff'),
                   shutter=None, arch=True, muntin=False)
    # big loading doors on the ground floor
    dw = 10
    dx_ = x + w // 2 - dw // 2
    cv.rect(dx_ - 1, gy - 13, dw + 2, 13, TRIM[2])
    cv.rect(dx_, gy - 12, dw, 12, WOOD[2])
    for i in range(dx_, dx_ + dw, 2):
        cv.vline(i, gy - 12, gy - 1, WOOD[1])
    cv.vline(dx_ + dw // 2, gy - 12, gy - 1, WOOD[0])
    cv.line(dx_, gy - 12, dx_ + dw // 2 - 1, gy - 1, WOOD[3])
    # icon sign board
    sy = top + 4 + fl * 11 - 1
    sx = x + w // 2 - 5
    if sy + 7 < gy - 14:
        cv.rect(sx, sy, 11, 7, '#fff4d6')
        cv.frame(sx, sy, 11, 7, WOOD[1])
        icon(cv, sx + 3, sy + 1, sign, '#323353')


def crate(cv, x, base, w, h, color=None):
    cc = color or '#c9a070'
    dk, md, lt = shade(cc, 0.62), cc, mix(cc, 'white', 0.3)
    cv.rect(x, base - h, w, h, md)
    cv.frame(x, base - h, w, h, dk)
    cv.line(x, base - 1, x + w - 1, base - h, dk)
    cv.hline(x + 1, x + w - 2, base - h, lt)
    cv.vline(x + w - 1, base - h + 1, base - 2, lt)


def barrel(cv, x, base):
    cv.rect(x, base - 7, 6, 7, WOOD[2])
    cv.vline(x, base - 7, base - 1, WOOD[1])
    cv.vline(x + 4, base - 6, base - 2, WOOD[3])
    cv.hline(x, x + 5, base - 6, '#4a4e5c')
    cv.hline(x, x + 5, base - 2, '#4a4e5c')
    cv.hline(x + 1, x + 4, base - 8, WOOD[1])


def bollard(cv, x, base):
    cv.rect(x, base - 4, 4, 4, '#3a3248')
    cv.rect(x - 1, base - 5, 6, 2, '#4f4562')
    cv.px(x + 3, base - 5, '#857c9a')


def crane_static(cv, x, gy):
    """Gantry dock crane: two A-frame legs + machinery house (the jib is drawn per frame)."""
    for lx in (x, x + 20):
        cv.line(lx, gy, x + 10, gy - 34, CRANE[1])
        cv.line(lx + 1, gy, x + 11, gy - 34, CRANE[2])
    cv.line(x + 2, gy - 8, x + 19, gy - 8, CRANE[1])
    cv.line(x + 4, gy - 16, x + 17, gy - 16, CRANE[1])
    cv.line(x + 3, gy - 8, x + 16, gy - 16, CRANE[0])
    cv.line(x + 17, gy - 8, x + 5, gy - 16, CRANE[0])
    cv.line(x + 6, gy - 24, x + 15, gy - 24, CRANE[1])
    cv.rect(x - 1, gy - 2, 4, 2, '#3a3248')
    cv.rect(x + 19, gy - 2, 4, 2, '#3a3248')
    # machinery house
    hx, hy = x + 2, gy - 46
    cv.rect(hx, hy, 18, 12, CRANE[2])
    cv.vline(hx, hy, hy + 11, CRANE[1])
    cv.vline(hx + 17, hy, hy + 11, CRANE[3])
    cv.rect(hx - 1, hy - 2, 20, 2, '#4f4562')
    cv.hline(hx - 1, hx + 18, hy - 2, '#857c9a')
    cv.rect(hx + 11, hy + 3, 5, 4, GLASS[1])
    cv.px(hx + 14, hy + 3, GLASS[3])
    cv.rect(hx + 2, hy + 4, 6, 7, CRANE[1])          # counterweight door
    cv.rect(hx - 6, hy + 4, 7, 6, '#5a5272')          # counterweight block
    cv.hline(hx - 6, hx, hy + 4, '#857c9a')
    cv.rect(hx + 4, hy - 6, 2, 4, '#4f4562')          # exhaust stack


def lattice(c, x0, y0, x1, y1, w, cols):
    """Lattice boom from (x0, y0) to (x1, y1), w px deep (perpendicular, upward)."""
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = uy, -ux                       # normal pointing up-left
    if ny > 0:
        nx, ny = -nx, -ny
    ax, ay = x0 + nx * w, y0 + ny * w
    bx, by = x1 + nx * w * 0.4, y1 + ny * w * 0.4
    c.line(x0, y0, x1, y1, cols[1])
    c.line(ax, ay, bx, by, cols[2])
    n = int(L // 5)
    for i in range(n):
        t0, t1 = i / n, (i + 0.5) / n
        p = (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0)
        q = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
        r = (x0 + (x1 - x0) * (t0 + 1 / n), y0 + (y1 - y0) * (t0 + 1 / n))
        c.line(p[0], p[1], q[0], q[1], cols[0])
        c.line(q[0], q[1], r[0], r[1], cols[0])


def crane_dyn(c, x, gy, f, phase=0.0, load=True):
    """Luffing jib + hook with a swinging crate; x = screen x of the crane's left leg."""
    px_, py_ = x + 18, gy - 40
    ang = math.radians(28 + 9 * math.sin(f * 2 * math.pi / 144.0 + phase))
    L = 46
    tx = px_ + math.cos(ang) * L
    ty = py_ - math.sin(ang) * L
    lattice(c, px_, py_, tx, ty, 4, CRANE)
    c.line(x + 6, gy - 52, int(tx), int(ty), '#5a5272')       # stay cable from the A-frame top
    c.vline(x + 6, gy - 54, gy - 46, CRANE[1])
    tx, ty = int(round(tx)), int(round(ty))
    c.px(tx, ty, '#3a3248')
    if load:
        drop = 18 + int(round(8 * math.sin(f * 2 * math.pi / 96.0 + phase)))
        sw = int(round(math.sin(f * 2 * math.pi / 40.0 + phase) * 1.2))
        c.line(tx, ty + 1, tx + sw, ty + drop, '#4a4e5c')
        hx, hy = tx + sw, ty + drop
        c.px(hx, hy + 1, '#9aa3b4')
        c.line(hx, hy + 1, hx - 4, hy + 4, '#4a4e5c')
        c.line(hx, hy + 1, hx + 4, hy + 4, '#4a4e5c')
        crate(c, hx - 5, hy + 12, 11, 8, '#d4a852')
        c.px(hx - 3, hy + 7, '#323353')


@lru_cache(maxsize=1)
def _b_wf():
    """Waterfront: red-brick warehouses, quay wall, crane bases, crates, barrels, bollards."""
    w = 380
    cv = new(w)
    qy = 126
    warehouse(cv, -4, qy, 34, 40, 1, ANCHOR)
    warehouse(cv, 30, qy, 28, 34, 2, BARREL, ramp=('#5e2a2a', '#7e3630', '#9a4436', '#b4553f', '#cf7458'))
    warehouse(cv, 58, qy, 38, 44, 3, ANCHOR)
    warehouse(cv, 98, qy, 26, 30, 4, BARREL, ramp=('#5e2a2a', '#7e3630', '#9a4436', '#b4553f', '#cf7458'))
    warehouse(cv, 300, qy, 34, 38, 5, ANCHOR)
    warehouse(cv, 334, qy, 30, 32, 6, BARREL)
    crane_static(cv, 178, qy)
    crane_static(cv, 246, qy)
    # quay top + granite wall with mooring rings
    cv.rect(0, qy, w, 2, '#c9c2c8')
    cv.hline(0, w - 1, qy, '#e6e0e6')
    cv.rect(0, qy + 2, w, WL - qy - 2, '#8e8aa4')
    for x in range(0, w, 7):
        cv.vline(x + (3 if (x // 7) % 2 else 0), qy + 2, WL - 1, '#6e6a86')
    cv.hline(0, w - 1, qy + 4, '#6e6a86')
    cv.hline(0, w - 1, WL - 1, '#5a5272')
    for x in range(10, w, 30):
        cv.circle(x, qy + 5, 1, '#3a3248', fill=False)
    # crates, barrels, bollards on the quay
    R = rng(33)
    for (x, n) in ((130, 3), (150, 2), (206, 2), (276, 3), (366, 2)):
        for k in range(n):
            crate(cv, x + k * 8, qy, 8, 7, ('#c9a070', '#df8fae', '#7fc4ad', '#94b2e6', '#ecc46e')[(x + k) % 5])
        if n == 3:
            crate(cv, x + 4, qy - 7, 8, 7, '#c9a070')
    for x in (164, 170, 232, 290):
        barrel(cv, x, qy)
    for x in range(12, w, 44):
        bollard(cv, x, qy)
    # quay flagpole
    cv.vline(140, 84, qy - 1, '#c4c2d6')
    cv.vline(141, 84, qy - 1, '#8f8aa8')
    cv.px(140, 83, 'gold')
    return to_layer(cv)


def reflect(c, wl, y1, f, tint, k0=0.38, k1=0.62, amp=1.0, x0=0, x1=W, src=None):
    """Mirror the image above `wl` into rows wl..y1 (rippled, tinted toward `tint`)."""
    a = c.a if src is None else src
    rows = np.arange(wl, y1)
    n = len(rows)
    if n <= 0:
        return
    depth = (rows - wl) / max(1, n)
    srcr = np.clip(2 * wl - rows - 1 - (depth * 6).astype(int), 0, H - 1)
    dx = np.round(np.sin(rows * 1.3 + f * 0.33) * amp * (0.5 + depth)).astype(int)
    xs = np.clip(np.arange(x0, x1)[None, :] + dx[:, None], 0, W - 1)
    ref = a[srcr[:, None], xs].astype(np.float32)
    k = (k0 + (k1 - k0) * depth)[:, None, None]
    out = ref * (1 - k) + np.array(col(tint), np.float32) * k
    c.a[wl:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


def water_glints(c, wl, y1, f, cam, seed=0, sunx=SUN[0], cols=('#9fd2ec', '#e8f8ff')):
    """Rippling highlight dashes + twinkling sun sparkles on the water."""
    for i in range(70):
        yy = wl + 1 + int(hash01(seed, i, 1) ** 1.4 * (y1 - wl - 2))
        span = 2 + int(hash01(seed, i, 2) * 5)
        x = int((hash01(seed, i, 3) * 400 - cam * (0.6 + (yy - wl) / 60.0) + f * 0.3 * ((i % 3) - 1)) % 340) - 10
        if (f // 6 + i) % 4 == 0:
            continue
        c.hline(x, x + span, yy, cols[0])
    for i in range(18):
        x = sunx - 40 + int(hash01(seed, i, 7) * 70)
        yy = wl + 2 + int(hash01(seed, i, 8) * (y1 - wl - 4))
        t = ((f + hash01(seed, i, 9) * 30) % 30) / 30.0
        if t < 0.35:
            c.px(x, yy, 'white')
            if t < 0.15:
                c.px(x - 1, yy, cols[1])
                c.px(x + 1, yy, cols[1])


def sailing_ship(c, x, wl, f, L=70, masts=2, hull=('#2f3150', '#f4efe4', '#5b8fcf'), seed=0, yards=False,
                 scale=1.0, pennant=('#ff9fd0', '#bfe3ff', 'gold')):
    """Moored sailing ship (sails furled) rocking on the swell. x = stern, wl = waterline.
    Returns the screen bbox (x0, y0, x1, y1) of what was drawn."""
    ph = seed * 1.7
    bob = int(round(math.sin(f * 2 * math.pi / 60.0 + ph) * 1.0))
    lean = math.sin(f * 2 * math.pi / 60.0 + ph + 0.9) * 1.6
    wy = wl + bob
    hh = int(10 * scale)
    # hull: sheer line rising to bow (right) and stern (left)
    top = []
    for i in range(L + 1):
        u = i / L
        top.append(wy - hh - int(round(3 * scale * ((1 - u) ** 2 * 0.8 + u ** 3 * 1.6))))
    pts = [(x + i, top[i]) for i in range(L + 1)]
    bot = [(x + L - 4, wy), (x + 8, wy), (x + 1, wy - hh // 2)]
    c.poly(pts + [(x + L + 4, top[L] - 1)] + bot, hull[1])
    # dark bottom band + stripe + gunwale
    for i in range(1, L + 3):
        xx = x + i
        tp = top[min(i, L)]
        c.vline(xx, wy - 2, wy - 1, hull[0]) if 8 <= i <= L - 4 else None
        c.px(xx, tp, mix(hull[1], 'white', 0.5) if i < L + 1 else hull[1])
        c.px(xx, tp + 3, hull[2]) if i < L - 2 else None
        if hh > 9:
            c.px(xx, tp + 4, hull[2]) if i < L - 2 else None
    for i in range(8, L - 6, 7):                    # portholes / gunports
        c.px(x + i, top[i] + 6, '#2f3150')
    c.line(x + 1, wy - hh // 2, x + 8, wy - 1, hull[0])
    c.line(x + L - 4, wy - 1, x + L + 4, top[L] - 1, shade(hull[1], 0.8))
    c.vline(x + 1, top[0], wy - hh // 2, shade(hull[1], 0.75))
    # stern cabin + rail
    c.rect(x + 3, top[3] - 4, 12, 4, '#a8784f')
    c.hline(x + 3, x + 14, top[3] - 5, '#5e3f30')
    c.rect(x + 5, top[3] - 3, 2, 2, GLASS[2])
    c.rect(x + 9, top[3] - 3, 2, 2, GLASS[2])
    for i in range(16, L - 2, 3):
        c.px(x + i, top[i] - 1, '#83593e')
    c.hline(x + 16, x + L - 2, top[L // 2] - 2, '#83593e')
    # bowsprit
    bx0, by0 = x + L + 2, top[L] - 2
    c.line(bx0, by0, bx0 + int(14 * scale), by0 - int(5 * scale), '#5e3f30')
    # masts
    mxs = [x + int(L * (0.3 + 0.55 * k / max(1, masts - 1))) for k in range(masts)] if masts > 1 else [x + L // 2]
    mh = [int((40 + 8 * (k == masts // 2)) * scale) for k in range(masts)]
    tips = []
    for k, mx in enumerate(mxs):
        base = top[mx - x] - 1
        tx = mx + int(round(lean * mh[k] / 40.0))
        ty = base - mh[k]
        c.line(mx, base, tx, ty, '#5e3f30')
        c.line(mx + 1, base, tx + 1, ty, '#83593e')
        tips.append((tx, ty))
        # furled sails on yards / booms
        if yards:
            for j, yf in enumerate((0.3, 0.55, 0.78)):
                yy = int(base - mh[k] * yf)
                xm = mx + int(round(lean * mh[k] * yf / 40.0))
                half = int((11 - j * 3) * scale)
                c.hline(xm - half, xm + half, yy, '#5e3f30')
                c.hline(xm - half + 1, xm + half - 1, yy + 1, '#f4efe4')
                for q in range(xm - half + 2, xm + half - 1, 3):
                    c.px(q, yy + 1, '#d9cdb8')
                c.px(xm + half - 1, yy + 2, '#d9cdb8')
        else:
            yy = base - 7
            bl = int(19 * scale)
            for q in range(1, bl):
                by_ = yy + (2 * q) // bl
                xq = mx - q + int(round(lean * 7 / 40.0))
                thick = 3 if q < bl - 4 else 2
                for t_ in range(1, thick + 1):
                    c.px(xq, by_ - t_, '#f4efe4' if t_ < thick else '#fffaf0')
                c.px(xq, by_ - 1, '#d9cdb8')
                if q % 4 == 0:
                    c.vline(xq, by_ - thick, by_ - 1, '#b5443a')       # sail ties
                c.px(xq, by_, '#5e3f30')
            # furled jib along the forestay
            c.line(mx + 2, yy - int(mh[k] * 0.55), mx + 2 + int(10 * scale), yy - 2, '#e2d8c4')
        # pennant at the masthead (snaps in the breeze, blowing left)
        for q in range(int(7 * scale)):
            dy = int(round(math.sin(f * 0.5 + q * 0.9 + k) * 0.7 * (q / 4.0)))
            c.px(tx - 1 - q, ty + 1 + dy, pennant[k % 2] if q < 6 * scale else pennant[2])
            if q < 3 * scale:
                c.px(tx - 1 - q, ty + 2 + dy, pennant[k % 2])
    # rigging
    rg = '#8b7e86'
    if tips:
        c.line(tips[0][0], tips[0][1], x + 2, top[0] - 2, rg)
        c.line(tips[-1][0], tips[-1][1], bx0 + int(14 * scale), by0 - int(5 * scale), rg)
        for a_, b_ in zip(tips, tips[1:]):
            c.line(a_[0], a_[1], b_[0], b_[1], rg)
    return x - 2, min(t[1] for t in tips) - 2 if tips else top[0], x + L + int(16 * scale), wy


def ship_with_reflection(c, f, wl, tint, **kw):
    """Draw a ship onto a scratch layer, composite it, then mirror it below its own waterline."""
    tmp = Canvas(W, H, KEY)
    x0, y0, x1, y1 = sailing_ship(tmp, wl=wl, f=f, **kw)
    m = ~np.all(tmp.a == np.array(KEY, np.uint8), axis=2)
    np.copyto(c.a, tmp.a, where=m[..., None])
    wy = y1
    n = min(H - wy, max(0, (wy - y0)))
    if n <= 0:
        return
    rows = np.arange(wy, wy + n)
    srcr = np.clip(2 * wy - rows - 1, 0, H - 1)
    dx = np.round(np.sin(rows * 1.3 + f * 0.33) * (0.6 + (rows - wy) / 18.0)).astype(int)
    xs = np.clip(np.arange(W)[None, :] + dx[:, None], 0, W - 1)
    ref = tmp.a[srcr[:, None], xs].astype(np.float32)
    rm = m[srcr[:, None], xs] & (((rows[:, None] + np.arange(W)[None, :]) % 2 == 0) | (rows[:, None] - wy < 8))
    tt = np.array(col(tint), np.float32)
    blended = np.clip(ref * 0.5 + tt * 0.5, 0, 255).astype(np.uint8)
    reg = c.a[wy:wy + n]
    reg[rm] = blended[rm]


@lru_cache(maxsize=1)
def _b_fg():
    """Near dock: plank pier with pilings at the right, mooring posts with rope at the left."""
    w = 440
    cv = new(w)
    # left: pilings with rope
    for (x, top) in ((6, 150), (20, 156), (34, 160)):
        cv.rect(x, top, 6, H - top, WOOD[2])
        cv.vline(x, top, H - 1, WOOD[1])
        cv.vline(x + 5, top, H - 1, WOOD[4])
        cv.ellipse(x + 2, top, 3, 1, WOOD[3])
        cv.hline(x, x + 5, top + 6, '#4f8a76')
        cv.hline(x, x + 5, top + 7, '#3f7a66')
    cv.line(9, 156, 23, 160, '#d9c08a')
    cv.line(23, 162, 37, 165, '#d9c08a')
    # right: plank pier deck at the bottom with a lamp, bollard + coiled rope, crates
    px0 = 300
    cv.rect(px0, 162, w - px0, H - 162, WOOD[2])
    for yy in range(164, H, 4):
        cv.hline(px0, w - 1, yy, WOOD[1])
    for x in range(px0, w, 9):
        cv.vline(x + (yy % 3), 162, H - 1, WOOD[1])
    cv.hline(px0, w - 1, 162, WOOD[4])
    cv.hline(px0, w - 1, 163, WOOD[3])
    cv.vline(px0, 162, H - 1, WOOD[0])
    for x in range(px0 + 4, w, 24):
        cv.rect(x, 156, 5, 7, WOOD[1])
        cv.ellipse(x + 2, 156, 2, 1, WOOD[3])
    bollard(cv, px0 + 30, 162)
    cv.ellipse(px0 + 46, 160, 5, 2, '#c9a070')
    cv.ellipse(px0 + 46, 160, 3, 1, '#83593e')
    cv.ellipse(px0 + 46, 159, 1, 0, '#c9a070')
    crate(cv, px0 + 66, 162, 12, 10, '#c9a070')
    crate(cv, px0 + 78, 162, 10, 8, '#df8fae')
    crate(cv, px0 + 70, 152, 9, 8, '#7fc4ad')
    lamp_post(cv, px0 + 100, 162, h=30)
    return to_layer(cv)


def b_balloon(c, f, cam):
    bx = 236 - int(round(cam * 0.55)) + int(round(12 * seg(f, 180, 269)))
    by = 54 + int(round(math.sin(f / 29.0) * 1.5))
    draw_balloon(c, bx, by, f, size='s')


def render_b(f):
    cam = b_cam(f)
    c = _sky().copy()
    draw_clouds(c, f, cam, B_CLOUDS)
    comp(c, _b_far(), int(round(cam * B_FAR_P)))
    mo = int(round(cam * B_MID_P))
    comp(c, _b_mid(), mo)
    flag(c, CIVIC_X - 52 - mo + 2, 72, f, w=6, h=4, cols=('#bfe3ff', 'white'), seed=1)
    flag(c, CIVIC_X + 52 - mo, 72, f, w=6, h=4, cols=('#ff9fd0', 'white'), seed=2)
    b_balloon(c, f, cam)
    wo = int(round(cam * B_WF_P))
    comp(c, _b_wf(), wo)
    crane_dyn(c, 178 - wo, 126, f, 0.0)
    crane_dyn(c, 246 - wo, 126, f, 2.1, load=False)
    flag(c, 140 - wo, 84, f, w=14, h=8, cols=('#5b8fcf', 'white'), emblem='gold', seed=3, pole=False)
    for (x, y, s_) in ((13, 81, 4), (44, 86, 5), (77, 73, 6), (317, 82, 7), (349, 87, 8)):
        flag(c, x - wo, y - 6, f, w=5, h=3, cols=('#ff9fd0',) if s_ % 2 else ('#bfe3ff',), emblem=None,
             seed=s_)
    smoke(c, 256 - wo + 5, 74, f, seed=9, n=4, height=14, color='#d8dde6')
    reflect(c, WL, H, f, WATER[1], k0=0.48, k1=0.72, amp=1.2)
    water_glints(c, WL, H, f, cam, seed=5)
    # moored ships (near layer, rocking)
    ship_with_reflection(c, f, 154, WATER[1], x=150 - cam, L=84, masts=2, seed=1, scale=1.15,
                         hull=('#2f3150', '#f4efe4', '#5b8fcf'))
    ship_with_reflection(c, f, 146, WATER[1], x=300 - cam, L=54, masts=1, seed=2, scale=0.85,
                         hull=('#3a2a30', '#b5443a', '#f4efe4'), pennant=('#bfe3ff', '#ff9fd0', 'gold'))
    comp(c, _b_fg(), int(round(cam * B_FG_P)))
    callout(c, CIVIC_X - 15 - mo, 58, 'BALTIMARE', f, 186, 258, side='left', length=18)
    return c


# ================================================================== SHOT C: HORSESHOE BAY + HAYSEED SWAMPS
C_CAM0, C_CAM = 60, 110
HORIZON = 82
SWAMP_X0 = 290                          # main-layer x where the beach turns into the bayou
C_FAR_P, C_LH_P, C_SHIP_P, C_SWB_P, C_FG_P = 0.12, 0.4, 0.6, 0.75, 1.4
LH_X = 52                               # lighthouse x in its layer
CABINS = ((318, 0), (392, 1))           # main-layer x of the stilt cabins
C_CLOUDS = [(30, 30, 58, 18, 31, 0.05), (150, 14, 76, 24, 32, 0.06), (250, 44, 48, 15, 33, 0.04),
            (360, 22, 64, 20, 34, 0.07)]
SAND = ('#c29a62', '#d9b67c', '#ead098', '#f6e4b8', '#fff4da')
BAY = ('#2a6aa6', '#2d82b4', '#2f9cbc', '#3ab6c0', '#57cfc4', '#8fe6d4', '#c4f6e8')
PALM = ('#1a5a3a', '#2f7d45', '#4fa24c', '#8ccf5a', '#c2e98a')
MURK = ('#24463c', '#2f5e50', '#4a7e62', '#76a274')
MOSS = ('#7f9a80', '#9fb79a', '#c2d4b8')


def c_cam(f):
    return C_CAM0 + int(round(C_CAM * cruise(seg(f, 270, 370))))     # still moving at the end


def shore_y(x):
    x = np.asarray(x, np.float32)
    return 147 - 11 * np.exp(-((x - 170) / 110.0) ** 2) + 1.5 * np.sin(x / 19.0)


def spit_y(x):
    """Top of the sand spit where the beach dips under the bayou (organic, wavy edge)."""
    d = x - (SWAMP_X0 - 70)
    return int(112 + d * 1.2 + 0.012 * d * d + 4 * math.sin(x / 6.0) + 3 * math.sin(x / 13.0 + 1)
               + 2 * math.sin(x / 2.7))


@lru_cache(maxsize=1)
def _c_sea():
    """Screen-space bay water: deep blue at the horizon to turquoise shallows."""
    c = Canvas(W, H, KEY)
    c.gradient(0, HORIZON, W, H - HORIZON - 20, list(BAY), soft=0.7)
    c.rect(0, H - 20, W, 20, BAY[-1])
    c.hline(0, W - 1, HORIZON, '#5e9fd0')
    c.hline(0, W - 1, HORIZON + 1, '#3a7cb8')
    return to_layer(c)


@lru_cache(maxsize=1)
def _c_far():
    """The far rim of the horseshoe: a low sandy, palm-dotted shore curving across the bay."""
    w = 360
    cv = new(w)
    for x in range(110, w):
        u = (x - 110) / (w - 110)
        t = int(HORIZON + 1 + 9 * u ** 1.3)
        hgt = 2 + int(5 * u)
        cv.vline(x, t - hgt, t, hz('#7fb07a', 0.55))
        cv.px(x, t - hgt, hz('#b7d892', 0.5))
        cv.hline(x, x, t + 1, hz(SAND[3], 0.4))
        if x % 9 == 0:
            pt = t - hgt
            cv.vline(x, pt - 4, pt, hz('#8a7a6a', 0.5))
            for dx_ in (-2, -1, 1, 2):
                cv.px(x + dx_, pt - 4 + abs(dx_) // 2, hz(PALM[2], 0.5))
        if x % 23 == 5:
            cv.rect(x, t - hgt - 3, 3, 3, hz(('#ff9fd0', '#bfe3ff', '#fff4d6')[x % 3], 0.4))
            cv.px(x + 1, t - hgt - 4, hz('brick', 0.4))
    return to_layer(cv)


def palm_trunk(cv, x, base, h, lean, seed):
    """Curved ringed palm trunk; returns the crown point."""
    pts = []
    for i in range(h):
        u = i / h
        px_ = x + lean * u ** 1.6
        pts.append((px_, base - i))
    for i, (px_, py_) in enumerate(pts):
        ix = int(round(px_))
        wdt = 3 if i < h * 0.7 else 2
        cv.hline(ix - 1, ix - 1 + wdt - 1, int(py_), TRUNK[2] if (i // 2) % 2 else TRUNK[1])
        cv.px(ix - 1, int(py_), TRUNK[0])
        cv.px(ix - 1 + wdt - 1, int(py_), TRUNK[3] if i % 2 else TRUNK[2])
    cx, cy = int(round(pts[-1][0])), int(pts[-1][1])
    cv.circle(cx, cy, 2, '#6a4a30')
    cv.px(cx - 1, cy + 1, '#4a3a6a') if False else None
    for (dx_, dy_) in ((-1, 2), (1, 2), (0, 3)):        # coconuts
        cv.px(cx + dx_, cy + dy_, '#5e3f30')
    return cx, cy


def palm_fronds(c, cx, cy, f, size=14, seed=0):
    """Swaying palm crown drawn per frame (the breeze blows from the sea, to the left)."""
    sw = math.sin(f * 2 * math.pi / 54.0 + seed) * 0.08
    angs = (-2.75, -2.25, -1.65, -1.05, -0.45, 0.1, 2.95)
    for k, a0 in enumerate(angs):
        a = a0 + sw + 0.05 * math.sin(f * 0.21 + k)
        L = size * (0.85 + 0.25 * hash01(seed, k))
        droop = 0.55 + 0.2 * hash01(seed, k, 2)
        prev = None
        n = int(L)
        for i in range(n + 1):
            s = i / n
            x = cx + math.cos(a) * s * L
            y = cy + math.sin(a) * s * L + droop * (s ** 2) * L * 0.9
            ix, iy = int(round(x)), int(round(y))
            if prev:
                c.line(prev[0], prev[1], ix, iy, PALM[1])
            prev = (ix, iy)
            ll = int(round((1 - s) * 3.2 + 0.8))
            if i % 1 == 0 and 0 < i < n:
                lit = PALM[3] if math.cos(a) > -0.2 else PALM[2]
                for q in range(1, ll + 1):
                    c.px(ix + int(round(-math.sin(a) * q * 0.4)), iy + q, PALM[0] if q == ll else PALM[2])
                    c.px(ix + int(round(math.sin(a) * q * 0.3)), iy - q + 1 if q > 1 else iy - 1,
                         lit if q < ll else PALM[2])
        if prev:
            c.px(prev[0], prev[1], PALM[0])
    c.px(cx, cy - 1, PALM[4])


PALMS = ((40, 170, 58, 10, 1, 20), (96, 160, 46, -7, 2, 17), (146, 157, 34, 8, 3, 14), (226, 166, 50, -6, 4, 18),
         (198, 154, 30, 5, 6, 13))


@lru_cache(maxsize=1)
def _c_palms():
    w = 520
    cv = new(w)
    crowns = []
    for (x, base, h, lean, seed, size) in PALMS:
        crowns.append(palm_trunk(cv, x, base, h, lean, seed) + (seed, size))
        cv.ellipse(x - 8, base + 1, 7, 1, SAND[1])
    return to_layer(cv), tuple(crowns)


def rocks(cv, blobs, ramp=('#4f4562', '#665c7c', '#857c9a', '#a49cb6', '#c7c0d4'), out='#3a3248'):
    balls(cv, blobs, ramp, outline=out, dith=0.6, crease=True)


@lru_cache(maxsize=1)
def _c_lighthouse():
    """Rocky point with the red-and-white lighthouse + keeper's cottage."""
    w = 200
    cv = new(w)
    RR = ('#5a4a52', '#776470', '#9a8690', '#bba9ae', '#dccfcf')
    R = rng(71)
    bl = []
    for i in range(14):
        x = -6 + i * 8 + R.random() * 5
        y = 96 + abs(x - LH_X) * 0.12 + R.random() * 4
        r = 7 + R.random() * 6 - abs(x - LH_X) * 0.03
        bl.append((x, y, r * 1.2, r * 0.8, R.random() * 3))
    rocks(cv, bl, RR, '#43363e')
    for i in range(18):                                      # grass tufts on the rocks
        gx = int(R.random() * 100)
        gy_ = 90 + int(abs(gx - LH_X) * 0.12) + int(R.random() * 3)
        cv.vline(gx, gy_ - 2, gy_, GRASS[2])
        cv.px(gx + 1, gy_ - 1, GRASS[3])
    # keeper's cottage
    kx = LH_X + 10
    cv.rect(kx, 80, 18, 10, '#f4efe4')
    cv.vline(kx, 80, 89, '#cfc8bf')
    cv.tri((kx - 2, 80), (kx + 20, 80), (kx + 9, 73), '#b5443a')
    cv.line(kx + 9, 73, kx + 20, 80, '#d66a52')
    cv.rect(kx + 4, 83, 3, 3, GLASS[1])
    cv.rect(kx + 11, 83, 3, 7, '#5b8fcf')
    cv.rect(kx + 14, 74, 2, 4, '#9a8690')
    # tower: tapered, red/white bands, lit on the right
    base, top = 92, 44
    for y in range(top, base):
        u = (y - top) / (base - top)
        hw_ = 4 + u * 3.5
        x0, x1 = int(round(LH_X - hw_)), int(round(LH_X + hw_))
        band = ((y - top) // 8) % 2 == 0
        cc = ('#b5443a', '#d0604a', '#ee8a6e') if band else ('#cfc8d6', '#f1eef6', 'white')
        cv.hline(x0, x1, y, cc[1])
        cv.hline(LH_X + 1, x1, y, cc[2])
        cv.px(x0, y, shade(cc[1], 0.7))
        cv.px(x1, y, cc[2])
    cv.rect(LH_X - 1, base - 6, 3, 6, '#3a3248')
    for wy in (top + 12, top + 28):
        cv.rect(LH_X, wy, 2, 3, '#323353')
    # gallery + lantern room + cap
    cv.rect(LH_X - 7, top - 2, 15, 2, '#3a3248')
    for i in range(LH_X - 7, LH_X + 8, 2):
        cv.vline(i, top - 5, top - 3, '#3a3248')
    cv.hline(LH_X - 7, LH_X + 7, top - 6, '#3a3248')
    cv.rect(LH_X - 3, top - 12, 7, 7, '#fff3b0')
    cv.vline(LH_X - 3, top - 12, top - 6, '#3a3248')
    cv.vline(LH_X + 3, top - 12, top - 6, '#3a3248')
    cv.vline(LH_X, top - 12, top - 6, '#c9a24a')
    cv.tri((LH_X - 5, top - 12), (LH_X + 5, top - 12), (LH_X, top - 18), '#b5443a')
    cv.line(LH_X, top - 18, LH_X + 5, top - 12, '#ee8a6e')
    cv.vline(LH_X, top - 21, top - 19, 'gold')
    return to_layer(cv)


def lighthouse_beam(c, lx, ly, f):
    """Rotating beam: a pale wedge sweeping round; flares when it faces the camera."""
    phi = f * 2 * math.pi / 72.0
    reach = math.cos(phi)
    facing = math.sin(phi)
    if facing > 0.75:
        fx.glow(c, lx, ly, 14, '#fff6c0', strength=0.55 * (facing - 0.75) * 4, rings=4)
    L = int(abs(reach) * 120)
    if L < 6:
        return
    d = 1 if reach > 0 else -1
    x0, x1 = (lx, lx + d * L) if d > 0 else (lx - L, lx)
    x0, x1 = max(0, min(x0, x1)), min(W, max(x0, x1))
    if x1 <= x0:
        return
    yy, xx = np.mgrid[max(0, ly - 30):min(H, ly + 31), x0:x1]
    dist = np.abs(xx - lx) + 1
    spread = 1.5 + dist * 0.12
    m = np.abs(yy - ly - dist * 0.05) < spread
    thr = bayer(m.shape[0], m.shape[1], x0, max(0, ly - 30))
    strength = (1 - dist / (L + 1)) * 0.55 * (0.4 + 0.6 * (1 - abs(facing)))
    m &= thr < strength
    reg = c.a[max(0, ly - 30):min(H, ly + 31), x0:x1]
    reg[m] = (reg[m].astype(np.int32) * 1 + np.array(col('#fff8d0'), np.int32) * 2) // 3


def stilt_cabin(cv, x, gy, seed):
    """Bayou cabin on stilts: faded pastel plank walls, tin roof, porch, lit window, stove pipe."""
    wall = (('#4f8a86', '#6aa6a0', '#8cc2b8'), ('#b0707e', '#c98a96', '#e0aab2'))[seed % 2]
    fl = gy - 16                       # floor level
    for sx in range(x - 2, x + 30, 6):
        cv.vline(sx, fl, gy + 14, WOOD[1])
        cv.vline(sx + 1, fl, gy + 14, WOOD[2])
    cv.line(x - 2, fl + 2, x + 28, gy + 8, WOOD[0])
    cv.rect(x - 4, fl - 1, 36, 3, WOOD[2])
    cv.hline(x - 4, x + 31, fl - 1, WOOD[3])
    # walls
    cv.rect(x, fl - 16, 22, 15, wall[1])
    for yy in range(fl - 15, fl - 1, 3):
        cv.hline(x, x + 21, yy, wall[0])
    cv.vline(x, fl - 16, fl - 2, wall[0])
    cv.vline(x + 21, fl - 16, fl - 2, wall[2])
    cv.rect(x + 3, fl - 12, 6, 5, '#ffcf6a')
    cv.frame(x + 2, fl - 13, 8, 7, WOOD[3])
    cv.vline(x + 6, fl - 12, fl - 8, WOOD[2])
    cv.rect(x + 13, fl - 12, 5, 10, '#f57d4a' if seed % 2 == 0 else '#5b8fcf')
    cv.px(x + 16, fl - 7, 'gold')
    # porch with railing on the right
    for px_ in range(x + 22, x + 32, 3):
        cv.vline(px_, fl - 6, fl - 2, WOOD[3])
    cv.hline(x + 22, x + 31, fl - 6, WOOD[3])
    cv.vline(x + 31, fl - 18, fl - 2, WOOD[2])
    # tin roof (corrugated, a little rust)
    for i in range(7):
        cv.hline(x - 3 + i, x + 34 - i, fl - 17 - i, '#8f97a8' if i % 2 else '#aab2c2')
    for xx in range(x - 3, x + 35, 2):
        cv.vline(xx, fl - 17 - max(0, min(6, min(xx - x + 3, x + 34 - xx))), fl - 17, '#7d8494')
    cv.hline(x + 3, x + 9, fl - 20, '#b5703a')
    cv.vline(x + 18, fl - 30, fl - 20, '#4a4e5c')
    cv.px(x + 18, fl - 31, '#3a3248')
    # ladder down to a little rowboat
    for yy in range(fl + 1, gy + 4, 3):
        cv.hline(x + 33, x + 35, yy, WOOD[3])
    cv.vline(x + 33, fl, gy + 4, WOOD[2])
    cv.vline(x + 35, fl, gy + 4, WOOD[2])
    return (x + 26, fl - 9), (x + 18, fl - 31)


def cypress(cv, x, base, h, seed, haze=0.0):
    """Bald cypress: flared buttress trunk, flat layered canopy (moss drawn per frame)."""
    tr = [hz(c_, haze) for c_ in ('#3e3530', '#5a4a40', '#7a6656', '#9a8470')]
    for i in range(h):
        u = i / h
        hw_ = 1.5 + 5 * (1 - u) ** 4
        cv.hline(int(x - hw_), int(x + hw_), base - i, tr[1])
        cv.px(int(x - hw_), base - i, tr[0])
        cv.px(int(x + hw_), base - i, tr[2])
    R = rng(seed)
    blobs = []
    for k in range(4):
        cy = base - h + 6 + k * 7
        for j in range(3):
            bx = x + (R.random() - 0.5) * (14 + k * 8)
            r = 6 + R.random() * 4 + k
            blobs.append((bx, cy + R.random() * 3, r * 1.3, r * 0.55, R.random() * 2 + (3 - k)))
    ramp = [hz(c_, haze) for c_ in ('#1e3b2c', '#2c5236', '#3f6b3e', '#5e8a4a', '#86a95e')]
    balls(cv, blobs, ramp, outline=hz('#162c22', haze), dith=0.6, crease=True)
    for k in range(3):                                        # knees in the water
        kx = x - 10 + int(R.random() * 20)
        cv.vline(kx, base - 1 - int(R.random() * 3), base + 1, tr[1])
    return blobs


@lru_cache(maxsize=1)
def _c_swamp_back():
    """Hazy back rank of cypresses + undergrowth behind the bayou (par 0.75)."""
    w = 460
    cv = new(w)
    x0 = 284
    for x in range(x0, w):
        t = 96 - int(4 + 3 * math.sin(x / 7.0) + 2 * math.sin(x / 3.1))
        e = max(0, (x0 + 14 - x) * 2)
        cv.vline(x, t + e, 118, hz('#2c5236', 0.2))
        cv.px(x, t + e, hz('#4f7a46', 0.2))
    R = rng(81)
    for i in range(8):
        cypress(cv, x0 + 12 + i * 22 + int(R.random() * 8), 118, 52 + int(R.random() * 16), 500 + i, haze=0.2)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _c_main():
    """Beach (sand, rocks, rowboat, sandcastle) turning into the bayou (murky water, cypresses, cabins)."""
    w = 520
    cv = new(w)
    xs = np.arange(w)
    ys = shore_y(xs)
    sand_a = np.array([col(c_) for c_ in SAND], np.uint8)
    nz = hashgrid(H, w, 0, 0, 9)
    for x in range(min(w, SWAMP_X0 + 30)):
        t = int(ys[min(x, SWAMP_X0)])
        if x > SWAMP_X0 - 70:                      # the beach tapers into a spit toward the bayou
            t = max(t, spit_y(x))
        if t >= H:
            continue
        col_ = np.full(H - t, 2, int)
        col_[:3] = 1                                      # wet sand band
        r = nz[t:, x]
        col_[r < 0.12] = np.minimum(col_[r < 0.12], 1)
        col_[(r > 0.9) & (np.arange(H - t) > 3)] = 3
        cv.a[t:, x] = sand_a[col_]
    # sand ripples + dune grass toward the bottom
    for i in range(60):
        rx = int(hash01(3, i) * SWAMP_X0)
        ry = int(ys[rx]) + 8 + int(hash01(4, i) * 26)
        if ry < H and all(ry > spit_y(q) + 1 for q in (rx, rx + 5) if q > SWAMP_X0 - 70):
            cv.hline(rx, rx + 3, ry, SAND[3])
            cv.hline(rx + 1, rx + 4, ry + 1, SAND[1])
    # muddy, reedy edge along the spit
    for x in range(SWAMP_X0 - 90, SWAMP_X0 + 30):
        t = max(int(ys[min(x, SWAMP_X0)]), spit_y(x))
        if t >= H:
            continue
        u = clamp((x - (SWAMP_X0 - 90)) / 60.0)
        for yy in range(t, min(H, t + 3)):
            cv.px(x, yy, mix('#8a7a52', '#5a5a3a', u))
        if hash01(x, 7) < 0.35 + 0.4 * u:
            hgt = 3 + int(hash01(x, 8) * (5 + 6 * u))
            cv.vline(x, t - hgt, t, ('#4f7a3e', '#6a9a4a', '#8ab85a')[x % 3])
    # lily pads
    R = rng(91)
    for i in range(22):
        lx = SWAMP_X0 + 6 + int(R.random() * (w - SWAMP_X0 - 10))
        ly = 124 + int(R.random() * 50)
        cv.ellipse(lx, ly, 2 + (i % 2), 1, '#4f8a46')
        cv.px(lx + 1, ly - 1, '#8ccf5a')
        if i % 5 == 0:
            cv.px(lx, ly - 1, '#ff9fd0')
    # cabins on stilts + cypresses in front
    for (cx_, sd) in CABINS:
        stilt_cabin(cv, cx_, 132 + sd * 4, sd)
    for (tx, h, sd) in ((300, 96, 1), (376, 104, 2), (452, 92, 3), (500, 100, 4)):
        cypress(cv, tx, 140 + sd % 2 * 6, h, 600 + sd)
    # boardwalk between the cabins
    cv.rect(354, 118, 38, 2, WOOD[2])
    cv.hline(368, 409, 118, WOOD[3])
    for px_ in range(356, 392, 8):
        cv.vline(px_, 120, 142, WOOD[1])
    # rowboat tied at the first cabin
    bx, by = 296, 156
    cv.poly([(bx, by - 4), (bx + 22, by - 4), (bx + 18, by), (bx + 3, by)], '#a8784f')
    cv.hline(bx, bx + 22, by - 4, '#c9a070')
    cv.hline(bx + 3, bx + 18, by - 1, '#5e3f30')
    cv.line(bx + 22, by - 4, bx + 30, by - 18, '#d9c08a')
    # beach props: rocks, rowboat pulled up, sandcastle with a flag, driftwood
    rocks(cv, [(18, 150, 9, 5, 0), (28, 152, 6, 4, 1), (8, 153, 5, 3, 0)])
    rocks(cv, [(196, 146, 5, 3, 0), (203, 147, 3, 2, 1)])
    bx, by = 120, 166
    cv.poly([(bx, by - 6), (bx + 30, by - 6), (bx + 26, by), (bx + 4, by)], '#b5443a')
    cv.hline(bx, bx + 30, by - 6, '#f4efe4')
    cv.hline(bx + 1, bx + 29, by - 5, '#d0604a')
    cv.hline(bx + 4, bx + 26, by - 1, '#7e2a27')
    cv.rect(bx + 6, by - 8, 2, 2, WOOD[2])
    cv.line(bx + 8, by - 9, bx + 20, by - 12, WOOD[3])
    cv.ellipse(bx + 13, by + 1, 14, 1, SAND[1])
    sx, sy = 196, 172
    cv.rect(sx, sy - 6, 14, 6, SAND[3])
    cv.rect(sx + 3, sy - 10, 8, 4, SAND[3])
    for i in range(sx, sx + 14, 2):
        cv.px(i, sy - 7, SAND[3])
    cv.vline(sx, sy - 6, sy - 1, SAND[1])
    cv.rect(sx + 6, sy - 4, 2, 4, SAND[1])
    cv.vline(sx + 7, sy - 15, sy - 11, WOOD[1])
    cv.rect(sx + 8, sy - 15, 3, 2, '#ff9fd0')
    cv.line(60, 160, 76, 158, '#a8947a')
    cv.line(61, 161, 77, 159, '#7a6a58')
    return to_layer(cv)


def bayou_water(c, f, cam, src):
    """Murky green bayou water mirroring the cypresses + sky (`src`: the frame without the far ship);
    blends (dithered) into the turquoise bay."""
    sx0 = max(0, int(SWAMP_X0 - 110 - cam))
    if sx0 >= W:
        return
    y0 = 116
    old = c.a[y0:H, sx0:W].copy()
    reflect(c, 118, H, f, MURK[1], k0=0.45, k1=0.66, amp=0.8, x0=sx0, x1=W, src=src)
    c.hline(sx0, W - 1, 118, MURK[3])
    c.rect(sx0, y0, W - sx0, 2, MURK[0])
    for i in range(30):                                    # slow drifting ripple streaks
        yy = 121 + int(hash01(8, i, 1) * 56)
        x = sx0 + int((hash01(8, i, 2) * 300 + f * 0.15) % max(1, W - sx0))
        c.hline(x, x + 2 + i % 3, yy, MURK[2] if i % 2 else MURK[3])
    yy = np.arange(y0, H)[:, None]
    xp = np.arange(sx0, W)[None, :] + cam
    xb = SWAMP_X0 - 40 - (yy - y0) * 0.9
    p = np.clip((xp - xb) / 30.0 + 0.5, 0, 1)
    keep = bayer(H - y0, W - sx0, sx0, y0) >= p
    reg = c.a[y0:H, sx0:W]
    reg[keep] = old[keep]


def foam(c, f, cam):
    """Foam waves washing up the sand on every beat (rush in fast, slide back slowly)."""
    x0 = 0
    x1 = max(0, min(W, SWAMP_X0 - 26 - cam))
    if x1 <= x0:
        return
    xs = np.arange(x0, x1) + cam
    ys = shore_y(xs)
    ph = ((f - 216) % 18) / 18.0
    if ph < 0.3:
        adv = math.sin(ph / 0.3 * math.pi / 2)
    else:
        adv = math.cos((ph - 0.3) / 0.7 * math.pi / 2) ** 1.4
    wig = 1.5 * np.sin(xs / 7.0 + f * 0.05) + np.sin(xs / 3.0)
    front = ys + 1 + adv * (7 + wig)
    maxf = ys + 1 + 7 + wig
    ya, yb = 120, 165
    yy = np.arange(ya, yb)[:, None]
    thr = bayer(yb - ya, x1 - x0, x0, ya)
    reg = c.a[ya:yb, x0:x1]
    wet = (yy >= ys[None, :] + 1) & (yy < maxf[None, :]) & (yy > front[None, :] + 1) & (ph > 0.3)
    reg[wet] = C(SAND[1])
    sheet = (yy >= ys[None, :]) & (yy < front[None, :] - 1)
    reg[sheet & (thr < 0.75)] = C(BAY[5])
    reg[sheet & (thr >= 0.75)] = C(BAY[6])
    edge = (yy >= front[None, :] - 1.5) & (yy < front[None, :] + 0.5)
    reg[edge] = C('white')
    lace = (yy >= front[None, :] - 4) & (yy < front[None, :] - 1.5) & (thr < 0.35)
    reg[lace] = C('#eafffa')
    # incoming crest lines out on the shallows
    for k in range(2):
        q = (ph + k * 0.5) % 1.0
        cy = ys - 12 + q * 11
        crest = (np.abs(yy - cy[None, :]) < 0.6) & (yy < ys[None, :]) & (thr < 0.6 - 0.4 * q) & \
                ((np.arange(x0, x1)[None, :] + k * 5 + int(f * 0.4)) % 13 < 8)
        reg[crest] = C('white')


def c_glints(c, f, cam):
    """Sun path sparkles on the open bay + small chop dashes."""
    for i in range(40):
        yy = HORIZON + 3 + int(hash01(6, i, 1) ** 1.3 * 50)
        x = int((hash01(6, i, 2) * 380 - cam * (0.15 + (yy - HORIZON) / 90.0) + f * 0.25 * ((i % 3) - 1)) % 340) - 10
        if (f // 5 + i) % 3 == 0 or x + cam * 0 > SWAMP_X0 - 120 - cam + (yy - HORIZON):
            continue
        c.hline(x, x + 1 + int(hash01(6, i, 3) * 3), yy, BAY[5] if yy > 105 else '#6fb2dc')
    for i in range(22):
        x = SUN[0] - 18 + int(hash01(7, i, 1) * 36) + int((hash01(7, i, 4) - 0.5) * (i % 5) * 4)
        yy = HORIZON + 2 + int(hash01(7, i, 2) ** 1.6 * 34)
        t = ((f + hash01(7, i, 3) * 24) % 24) / 24.0
        if t < 0.4:
            c.px(x, yy, 'white')
            if t < 0.15:
                c.px(x - 1, yy, '#d8f6ff')
                c.px(x + 1, yy, '#d8f6ff')


def moss(c, f, cam):
    """Spanish moss strands swaying under the cypress canopies."""
    for (tx, h, sd) in ((300, 96, 1), (376, 104, 2), (452, 92, 3), (500, 100, 4)):
        base = 140 + sd % 2 * 6
        sx = tx - cam
        if sx < -30 or sx > W + 30:
            continue
        for k in range(7):
            mx = sx - 14 + int(hash01(sd, k) * 28)
            my = base - h + 12 + int(hash01(sd, k, 2) * 18)
            ln = 5 + int(hash01(sd, k, 3) * 9)
            for j in range(ln):
                dx = int(round(math.sin(f * 0.12 + k + j * 0.25) * j / 6.0))
                c.px(mx + dx, my + j, MOSS[j % 3] if j < ln - 1 else MOSS[0])


@lru_cache(maxsize=1)
def _c_fg():
    w = 560
    cv = new(w)
    R = rng(55)
    # beach grass tufts (left) and cattails/reeds (right, bayou)
    for i in range(26):
        gx = int(R.random() * 250)
        gy_ = 172 + int(R.random() * 8)
        for k in range(5):
            cv.line(gx, gy_, gx - 3 + k * 1.5, gy_ - 6 - int(R.random() * 5), GRASS[1 + k % 3])
    for i in range(40):
        rx = 420 + int(R.random() * 140)
        ry = 168 + int(R.random() * 12)
        hgt = 14 + int(R.random() * 14)
        cv.vline(rx, ry - hgt, ry, '#4f7a3e' if i % 2 else '#6a9a4a')
        if i % 3 == 0:
            cv.rect(rx - 1, ry - hgt - 1, 3, 5, '#6a4a30')
            cv.px(rx + 1, ry - hgt, '#9a7a50')
    return to_layer(cv)


def c_balloon(c, f, cam):
    so = cam * C_SHIP_P
    bx = 196 - int(round(so)) + int(round(10 * seg(f, 270, 359)))
    by = 38 + int(round(math.sin(f / 29.0) * 1.5))
    draw_balloon(c, bx, by, f, size='s')


def lanterns(c, f, cam):
    for (cx_, sd) in CABINS:
        lx = cx_ + 26 - cam
        ly = 132 + sd * 4 - 16 - 9
        if -10 < lx < W + 10:
            fl = 0.35 + 0.1 * math.sin(f * 0.9 + sd * 2) + 0.05 * hash01(sd, f // 2)
            fx.glow(c, lx, ly, 9, '#ffcf6a', strength=fl, rings=3)
            c.px(lx, ly - 2, '#3a3248')
            c.rect(lx - 1, ly - 1, 3, 3, '#ffe9a0')
            c.px(lx, ly, 'white')
        sx_ = cx_ + 18 - cam
        if -10 < sx_ < W + 10:
            smoke(c, sx_, 132 + sd * 4 - 16 - 32, f, seed=40 + sd, n=4, height=16, color='#c9d2cc')


def render_c(f):
    cam = c_cam(f)
    c = _sky(0.12).copy()
    draw_clouds(c, f, cam, C_CLOUDS)
    comp(c, _c_sea(), 0)
    comp(c, _c_far(), int(round(cam * C_FAR_P)))
    lo = int(round(cam * C_LH_P))
    comp(c, _c_lighthouse(), lo)
    lighthouse_beam(c, LH_X - lo, 44 - 9, f)
    snap = Canvas(arr=c.a.copy())                 # what the bayou mirrors (no far ship / balloon)
    so = int(round(cam * C_SHIP_P))
    ship_with_reflection(c, f, 114, BAY[2], x=150 - so, L=100, masts=3, seed=3, scale=1.25, yards=True,
                         hull=('#1f1a26', '#3e3050', '#f9c22b'), pennant=('#ff9fd0', '#bfe3ff', 'gold'))
    sx_ = 150 - so + 100 + 2
    c.line(sx_, 103, sx_ + 6, 114, '#4a4e5c')                  # anchor chain
    c_balloon(c, f, cam)
    comp(c, _c_swamp_back(), int(round(cam * C_SWB_P)))
    comp(snap, _c_swamp_back(), int(round(cam * C_SWB_P)))
    bayou_water(c, f, cam, snap.a)
    c_glints(c, f, cam)
    comp(c, _c_main(), cam)
    foam(c, f, cam)
    pl, crowns = _c_palms()
    comp(c, pl, cam)
    for (cx_, cy_, sd, size) in crowns:
        if -30 < cx_ - cam < W + 30:
            palm_fronds(c, cx_ - cam, cy_, f, size=size, seed=sd)
    moss(c, f, cam)
    lanterns(c, f, cam)
    comp(c, _c_fg(), int(round(cam * C_FG_P)))
    callout(c, 150 - so + 70, 118, 'HORSESHOE BAY', f, 274, 346, side='down', length=14)
    hx = CABINS[0][0] + 11 - cam
    callout(c, hx, 132 - 16 - 22, 'HAYSEED SWAMPS', f, 288, 360, side='up', length=18)
    return c


# ------------------------------------------------------------------ render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < 180:
        c = render_a(f)
    elif f < 270:
        c = render_b(f)
    else:
        c = render_c(f)
    atlas_card(c, f, 98, 160, 'COASTAL CITIES', ['BRICK - BELLS - HARBOURS', 'ON THE CELESTIAL SEA'])
    return c


SFX = [
    (80, 'wind', 0.25),                 # sea breeze over the plaza
    (88, 'creak', 0.2),                 # the bell's oak yoke starts to swing (music strikes at 108)
    (126, 'bell', 0.2),                 # echo stroke on the back-swing (smaller rings)
    (150, 'creak', 0.12),               # the yoke settling
    (180, 'horn', 0.5),                 # ship horn on the Baltimare cut
    (188, 'waves', 0.16),               # harbour lapping at the quay
    (206, 'creak', 0.16),               # crane jib / moored hulls
    (224, 'clank', 0.18),               # the hanging crate
    (242, 'wind', 0.22),                # flags snapping
    (269, 'waves', 0.34),               # breaker lands ~293 with a foam rush
    (280, 'creak', 0.14),               # tall ship at anchor
    (305, 'waves', 0.28),               # breaker lands ~329
    (316, 'wind', 0.2),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(180, 'cloud', {}), (270, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
