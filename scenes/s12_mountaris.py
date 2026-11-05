"""12 · mountaris — Mount Aris + Seaquestria, the sea kingdoms. 5 bars / 360 f, HUD 16:15 -> 17:00.

Beat sheet (SCENE frames; 18 f per beat, 72 per bar; N = 432; the 2-bar map flight is added outside
by engine/regionfx.long_flight; A major Mediterranean -> underwater):
  0-71      World-map travel shot (`travel_frame`). Iris closes 64-71 on dest_screen (204, 94).
  72-179    SHOT A "Sea crag" (3.6 s), warm late afternoon, sun low on the right, gold glitter path.
            Iris opens 72-79 on the crag. Slow cruise pan right (parallax: sky, far storm isle,
            far clouds, crag + harbour, sea with the crag's wobbling reflection, Basalt Beach
            columns in the foreground with foam and spray between them).
            Far left: the Storm King's black spiked castle (cyan cracks) under a lightning cloud,
            bolts at 94, 130, 166. Balloon 's' drifting by the summit.
            Callouts: BASALT BEACH 90-162, STORM KING'S REALM 108-174. Atlas card 100-208.
  180-287   SHOT B "Summit" (3.6 s; dissolve on the half-bar 180): Hippogriffia's restored palace,
            towers with teal cones, arcades, terraces, bougainvillea, cypresses, banners, fountain and
            the tall bronze-gold BUD DOME (glint sweeps every 2 beats). Balloon 'm' moored to a tower.
            Slow tilt up; 268-287 the camera swoops down the cliff toward the sea.
            Callout MOUNT ARIS 198-264.
  288-323   SHOT C "Dive" (quick plunge; `dive` hit on the bar line 288, 12-f dissolve): splash jets,
            the surface line sweeps up the screen, bubble plume; underwater god-rays, golden kelp,
            pearls in open shells, the mountain's submerged foot; the balloon wobbles on the surface.
  324-431   SHOT D "Seaquestria" (3.6 s + transition bar; dissolve on the half-bar 324): bioluminescent
            coral palace (magenta, violet, teal): conch spire, scallop-shell towers, pearl dome,
            anemone lanterns pulsing on beats, pearl strings, bubble streams, caustics. Slow cruise pan.
            Callout SEAQUESTRIA 342-414. Toast from N-58 = 374 (top-centre kept clear).
SFX: waves, wind, distant thunder after the flashes, shimmer on the dome glint, splash on `dive`,
     bubbles, crystal on the coral reveal.
"""
import math
from functools import lru_cache

import numpy as np
from scipy import ndimage

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'mountaris'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

# =====================================================================================  toolkit
KEY = (255, 0, 254)


def new(w, h):
    return Canvas(w, h, KEY)


def rgba(cv):
    """Canvas drawn on the KEY colour -> RGBA layer (hard alpha)."""
    a = cv.a
    m = ~((a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2]))
    out = np.zeros((cv.h, cv.w, 4), np.uint8)
    out[..., :3] = a
    out[..., 3] = m.astype(np.uint8) * 255
    return out


def C(name):
    return np.array(col(name), np.uint8)


def ramp_arr(ramp):
    return np.array([col(x) for x in ramp], np.uint8)


def vnoise(h, w, cell, seed):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng(seed).random((gh, gw)).astype(np.float32)
    ys = np.arange(h) / cell
    xs = np.arange(w) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = ys - y0
    fx = xs - x0
    fy = (fy * fy * (3 - 2 * fy))[:, None]
    fx = (fx * fx * (3 - 2 * fx))[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c_ = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c_ * (1 - fx) + d * fx) * fy


def fbm(h, w, cell, seed, octaves=3):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for i in range(octaves):
        out += vnoise(h, w, max(1.0, cell / 2 ** i), seed + 17 * i) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def pmask(w, h, pts):
    cv = Canvas(w, h, (0, 0, 0))
    cv.poly(pts, (255, 255, 255))
    return cv.a[..., 0] > 0


def emask(w, h, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / (rx + 0.5)) ** 2 + ((yy - cy) / (ry + 0.5)) ** 2 <= 1.0


def edge_of(m):
    return m & ~ndimage.binary_erosion(m)


def shade_blob(a, mask, ramp, light=(0.75, -0.65), rad=6, bias=0.0, noise=None, dith=0.8,
               outline=None, rim=None, rim_k=0.82, ox=0, oy=0, flat=0.0):
    """Paint `mask` on RGB array `a` with a rounded, lit, dithered colour ramp (dark -> light)."""
    if not mask.any():
        return None
    ramp = ramp_arr(ramp)
    n = len(ramp)
    d = ndimage.distance_transform_edt(mask)
    hgt = np.sqrt(np.minimum(d, rad) / rad)
    gy, gx = np.gradient(hgt)
    nx, ny = -gx * rad * 0.8, -gy * rad * 0.8
    lx, ly = light
    lz = 0.55
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lit = (nx * lx + ny * ly + lz) / (np.sqrt(nx * nx + ny * ny + 1) * ln)
    t = lit * (1 - flat) + flat * 0.5 + bias
    if noise is not None:
        t = t + noise
    t = np.clip(t, 0, 0.999)
    h, w = mask.shape
    thr = bayer(h, w, ox, oy)
    idx = np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)
    a[mask] = ramp[idx][mask]
    e = edge_of(mask)
    if rim is not None:
        a[e & (t > rim_k)] = col(rim)
        if outline is not None:
            a[e & (t <= rim_k)] = col(outline)
    elif outline is not None:
        a[e] = col(outline)
    return t


def put(c, xs, ys, color):
    """Set many pixels (numpy int arrays), clipped."""
    xs = np.asarray(xs, int)
    ys = np.asarray(ys, int)
    m = (xs >= 0) & (xs < c.w) & (ys >= 0) & (ys < c.h)
    c.a[ys[m], xs[m]] = col(color)


def add_light(c, x, y, w, h, mask, color, k):
    """Additive light where `mask` (h x w bool) is set, placed at (x, y)."""
    x0, y0, x1, y1 = c._clip(x, y, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = c.a[y0:y1, x0:x1].astype(np.int16)
    reg[m] += (np.array(col(color), np.int16) * k).astype(np.int16)
    c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def flash(c, f, f0, dur=4, color='white', k=0.6):
    if f0 <= f < f0 + dur:
        fx.fade(c, color, k * (1 - (f - f0) / dur))


def cam_ease(f, f0, f1, a, b, ease=ease_in_out):
    return int(round(lerp(a, b, ease(seg(f, f0, f1)))))


def cam_cruise(f, f0, f1, a, b):
    return int(round(lerp(a, b, cruise(seg(f, f0, f1)))))


CUT1, DIVE, CUT3 = 180, 288, 324          # shot changes (scene frames, on bar / half-bar lines)


def flag(c, x, y, f, color, length=5, seed=0, hi=None, pole=True, ph=2):
    """Small waving pennant from a pole top at (x, y) blowing to the right."""
    if pole:
        c.vline(x, y, y + ph, 'ink')
    for i in range(length):
        dy = int(round(math.sin(f * 0.35 - i * 0.9 + seed) * 0.8 * (i / max(1, length - 1))))
        hh = 2 if i < length - 1 else 1
        c.vline(x + 1 + i, y + dy, y + dy + hh - 1, color if (i + f // 4) % 3 else (hi or color))


def lantern(c, x, y, f, seed=0, glow=True):
    fl = 0.75 + 0.25 * math.sin(f * 0.9 + seed * 2.3) * math.sin(f * 0.37 + seed)
    if glow:
        fx.glow(c, x, y, 4, 'amber', 0.35 * fl, rings=2)
    c.px(x, y, 'butter' if fl > 0.7 else 'gold')


# =====================================================================================  SHOT A
HZ = 108                       # horizon (screen y)
SUN = (290, 66)
CRX = 120                      # crag layer world-x origin
P_CRAG = 0.5
P_FAR = 0.12
P_FG = 1.0
STAIR = [(156, 126), (232, 114), (172, 103), (236, 92), (176, 81), (232, 70), (184, 60), (204, 54)]
LANDINGS = STAIR[1:-1]
CAVES = [(164, 118), (246, 104), (204, 86), (226, 98), (200, 66)]


def cam_a(f):
    return cam_cruise(f, 72, CUT1 - 1, 0, 34)


@lru_cache(maxsize=1)
def _sky_a():
    c = Canvas()
    c.gradient(0, 0, W, HZ, ['#4a6cc4', '#5b86d6', '#6fa2e3', '#8fbfee', '#b9d6f0', '#f2dcc8', '#ffd2a6'],
               soft=0.7)
    # warm haze toward the sun
    fx.glow(c, SUN[0], SUN[1], 110, '#ff9a40', 0.22, rings=6)
    fx.glow(c, SUN[0], SUN[1], 46, '#ffd080', 0.30, rings=5)
    fx.glow(c, SUN[0], SUN[1], 18, 'butter', 0.45, rings=3)
    # cirrus streaks
    for i, (x, y, w) in enumerate([(150, 78, 60), (210, 88, 80), (40, 70, 50), (250, 50, 46), (120, 94, 70)]):
        for k in range(3):
            xx = x + int(hash01(i, k) * 14) - 7
            ww = int(w * (0.5 + 0.5 * hash01(i, k, 3)))
            yy = y + k * 2
            cc = mix(c.a[min(H - 1, yy), min(W - 1, max(0, xx))], 'cream', 0.45)
            c.hline(xx + 2, xx + ww - 2, yy, cc)
            c.dither(xx - 4, yy, 6, 1, cc, 0.5)
            c.dither(xx + ww - 2, yy, 6, 1, cc, 0.5)
    # sun disc
    c.circle(SUN[0], SUN[1], 9, '#ffe9a8')
    c.circle(SUN[0], SUN[1], 8, '#fff6d0')
    c.circle(SUN[0], SUN[1], 6, 'white')
    return c


@lru_cache(maxsize=None)
def _cloud(w, h, seed, warm=1.0):
    r = rng(seed)
    m = np.zeros((h, w), bool)
    n = max(3, w // 10)
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi)
        rx = int(w * (0.12 + 0.08 * r.random()))
        ry = int(h * (0.25 + 0.45 * bump) * (0.8 + 0.3 * r.random()))
        cx = int(u * w)
        cy = h - 3 - ry // 2
        m |= emask(w, h, cx, cy, rx, ry)
    m[h - 3:, :] = False
    m &= ~(np.arange(w)[None, :] < 1)
    cv = new(w, h)
    ramp = ['#8f8fcf', '#b3aee0', '#d8cdec', '#f6e9ee', '#fff7ee', 'white']
    yy = np.mgrid[0:h, 0:w][0]
    shade_blob(cv.a, m, ramp, light=(0.85, -0.5), rad=5, bias=-0.12 + (1 - yy / h) * 0.18,
               outline=None, rim='#ffe0a8' if warm else None, rim_k=0.86)
    return rgba(cv)


# ---- storm isle (far layer, p=P_FAR); layer origin world (0, STORM_OY)
STORM_OY = 24
SW_, SH_ = 150, HZ - STORM_OY + 2
SBASE = HZ - STORM_OY            # horizon in layer coords


@lru_cache(maxsize=1)
def _storm():
    w, h = SW_, SH_
    cv = new(w, h)
    a = cv.a
    m = np.zeros((h, w), bool)
    blobs = [(22, 14, 22, 7), (50, 10, 26, 8), (82, 11, 26, 8), (112, 15, 20, 6), (132, 18, 10, 4),
             (40, 24, 22, 11), (66, 22, 20, 12), (92, 26, 18, 10), (22, 28, 13, 8), (114, 28, 12, 7),
             (64, 34, 32, 8), (46, 36, 14, 6), (88, 37, 16, 6)]
    for (x, y, rx, ry) in blobs:
        m |= emask(w, h, x, y + 1, rx, ry)
    nz = (fbm(h, w, 7, 51) - 0.5) * 0.3
    yy = np.mgrid[0:h, 0:w][0]
    shade_blob(a, m, ['#15131c', '#1c1825', '#25202f', '#30283b', '#3e3149', '#52405a', '#6e5270'],
               light=(0.9, -0.45), rad=6, bias=-0.02 - (yy / 46.0) * 0.4, noise=nz, outline='#15131c',
               rim='#b07888', rim_k=0.82)
    stormmask = m.copy()
    # rain veil under the cloud base: slanted broken streaks
    for i in range(140):
        x = 18 + int(hash01(i, 21) * 100)
        y = 34 + int(hash01(i, 22) * (SBASE - 38))
        L = 3 + int(hash01(i, 23) * 6)
        for k in range(L):
            xx, yy_ = x + (y + k) // 4, y + k
            if 0 <= xx < w and yy_ < SBASE and not m[yy_, xx]:
                a[yy_, xx] = col('#7d7296') if (k + i) % 3 else col('#968cb0')
    # rocky island at the horizon
    hz = SBASE
    isl = pmask(w, h, [(30, hz + 1), (36, hz - 4), (44, hz - 6), (58, hz - 8), (90, hz - 7), (100, hz - 5),
                       (110, hz + 1)])
    shade_blob(a, isl, ['#3a2a2a', '#5a3e34', '#7a5644', '#9a765c'], light=(0.9, -0.4), rad=3,
               outline='#2a1e22')
    # castle: black, spiked
    k0, k1, k2, k3 = col('#17151f'), col('#2b2536'), col('#463a56'), col('#6a5070')
    yb = hz - 6
    cv.poly([(40, yb + 2), (46, yb - 10), (96, yb - 10), (102, yb + 2)], k0)
    for x in range(46, 96, 4):
        cv.tri((x, yb - 9), (x + 3, yb - 9), (x + 1, yb - 14), k0)
    cv.hline(46, 95, yb - 10, k1)

    def tower(x0, x1, ytop, tip):
        cv.rect(x0, ytop, x1 - x0 + 1, yb - ytop + 2, k0)
        cv.tri((x0 - 1, ytop + 1), (x1 + 2, ytop + 1), ((x0 + x1) // 2 + 1, ytop - tip), k0)
        cv.vline(x1, ytop, yb, k2)
        cv.vline(x1 - 1, ytop + 1, yb, k1)
        cv.line((x0 + x1) // 2 + 1, ytop - tip + 1, x1 + 1, ytop, k3)

    tower(50, 58, yb - 24, 12)
    tower(84, 92, yb - 22, 11)
    tower(64, 78, yb - 34, 18)
    for (xa, xb, xt, yt) in ((64, 68, 59, yb - 45), (74, 78, 83, yb - 45)):
        cv.tri((xa, yb - 33), (xb, yb - 33), (xt, yt), k0)
        cv.px(xt, yt, k3)
    for (x, y, dx, dy) in ((42, yb - 2, -9, -8), (100, yb - 3, 9, -9), (50, yb - 18, -6, -5), (92, yb - 16, 6, -6)):
        cv.tri((x, y), (x, y + 3), (x + dx, y + dy), k0)
    for (x, y) in ((70, yb - 28), (70, yb - 20), (54, yb - 16), (88, yb - 15), (60, yb - 4), (80, yb - 4),
                   (74, yb - 12)):
        cv.rect(x, y, 1, 2, '#1f6a78')
    return rgba(cv), stormmask


def _crack_list():
    yb = SBASE - 6
    return [((71, yb - 31), (70, yb - 28), (72, yb - 25), (70, yb - 22), (71, yb - 18)),
            ((54, yb - 21), (55, yb - 18), (53, yb - 14)), ((88, yb - 19), (87, yb - 16), (89, yb - 13)),
            ((58, yb - 6), (62, yb - 7), (65, yb - 5)), ((76, yb - 6), (80, yb - 8), (83, yb - 6)),
            ((66, yb - 13), (69, yb - 12), (72, yb - 14))]


CASTLE_CRACKS = _crack_list()


def _bolt_pts(seed, x0, y0, x1, y1, n=7):
    pts = [(x0, y0)]
    for i in range(1, n):
        t = i / n
        pts.append((int(lerp(x0, x1, t) + (hash01(seed, i) - 0.5) * 9), int(lerp(y0, y1, t))))
    pts.append((x1, y1))
    return pts


BOLTS = [(94, 3, (40, 40), (71, SBASE - 58)), (130, 3, (104, 38), (98, SBASE - 1)),
         (166, 2, (30, 40), (36, SBASE - 1))]


def draw_storm(c, f, cam):
    lay, smask = _storm()
    ox = int(round(-cam * P_FAR)) - 6
    oy = STORM_OY
    c.blit(lay, ox, oy)
    # rain streaks (animated, slanted)
    for i in range(70):
        x = 22 + int(hash01(i, 1) * 96)
        L = 3 + int(hash01(i, 2) * 3)
        y = 62 + int((hash01(i, 3) * 44 + f * 2.5) % 44)
        if y + L < HZ:
            c.line(x + ox + (y - 60) // 3, y, x + ox + (y + L - 60) // 3, y + L, '#8f86ae')
    lit = 0.0
    bolt = None
    for (f0, dur, p0, p1) in BOLTS:
        if f0 <= f < f0 + dur:
            lit = 1.0 - (f - f0) / dur * 0.6
            bolt = (f0, p0, p1)
    for f0 in (82, 112, 148):
        if f0 <= f < f0 + 2:
            lit = max(lit, 0.45)
    if lit > 0:
        add_light(c, ox, oy, smask.shape[1], smask.shape[0], smask, '#a080d0', 0.55 * lit)
    if bolt:
        f0, (x0, y0), (x1, y1) = bolt
        pts = _bolt_pts(f0, x0 + ox, y0 + oy, x1 + ox, y1 + oy)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            c.line(xa - 1, ya, xb - 1, yb, '#a8a0ff')
            c.line(xa + 1, ya, xb + 1, yb, '#d8d0ff')
            c.line(xa, ya, xb, yb, 'white')
        fx.glow(c, x1 + ox, y1 + oy, 12, '#c0b0ff', 0.45, rings=3)
    # glowing cyan cracks on the castle (pulse)
    on = 0.5 + 0.5 * math.sin(f * 0.21)
    cc = mix('#1f6a78', '#4fe8ff', on)
    for crack in CASTLE_CRACKS:
        for (xa, ya), (xb, yb) in zip(crack, crack[1:]):
            c.line(xa + ox, ya + oy, xb + ox, yb + oy, cc)
    if on > 0.6:
        fx.glow(c, 71 + ox, SBASE - 26 + oy, 7, '#4fe8ff', 0.22 * on, rings=2)


def storm_anchor(cam):
    return 71 + int(round(-cam * P_FAR)) - 6, STORM_OY + SBASE - 40


# ---- far islands silhouettes (hazy)
@lru_cache(maxsize=1)
def _far_isles():
    cv = new(W + 60, 12)
    for (x0, x1, hgt, seed) in ((128, 168, 5, 1), (150, 196, 3, 2), (270, 330, 6, 3), (316, 370, 3, 4)):
        pts = [(x0, 12)]
        n = 6
        for i in range(n + 1):
            t = i / n
            pts.append((int(lerp(x0, x1, t)), 12 - int(hgt * math.sin(t * math.pi) * (0.7 + 0.5 * hash01(seed, i)))))
        pts.append((x1, 12))
        cv.poly(pts, '#a9a6c8')
        cv.hline(x0 + 3, x1 - 3, 11, '#c9bfd2')
    return rgba(cv)


# ---- sea
SEA_RAMP = ['#b9cbe2', '#8fb6e0', '#5f9fd6', '#3d86c4', '#2d70ae', '#225c98', '#194a82']


@lru_cache(maxsize=1)
def _sea_base():
    c = Canvas(W, H - HZ)
    c.gradient(0, 0, W, H - HZ, SEA_RAMP, soft=0.8)
    fx.glow(c, SUN[0], 0, 120, '#ff9a40', 0.16, rings=5)
    fx.glow(c, SUN[0], 0, 50, '#ffc070', 0.18, rings=4)
    return c


@lru_cache(maxsize=1)
def _sea_pattern():
    """Per-row wave dash pattern (rows HZ..H-1) on a 640-wide loop: 1 = crest, 2 = trough."""
    rows = H - HZ
    pat = np.zeros((rows, 640), np.uint8)
    for r in range(rows):
        d = r / rows
        L = 1 + 7 * d
        gap = 10 + 52 * d
        x = int(hash01(r, 7) * 40)
        k = 0
        while x < 640:
            ll = max(1, int(L * (0.5 + 1.0 * hash01(r, k))))
            pat[r, x:x + ll] = 1
            if r + 1 < rows and d > 0.1:
                seg_ = pat[r + 1, x + 1:x + ll + 2]
                pat[r + 1, x + 1:x + ll + 2] = np.where(seg_ == 0, 2, seg_)
            x += ll + int(gap * (0.4 + 1.2 * hash01(r, k, 2)))
            k += 1
    return pat


@lru_cache(maxsize=1)
def _sea_rowdata():
    rows = H - HZ
    d = np.arange(rows) / rows
    par = 0.15 + 0.85 * d * d
    speed = 0.12 + 0.45 * d
    base = _sea_base().a
    mid = base[:, 150].astype(np.float32)
    crest = np.clip(mid + (255 - mid) * (0.35 + 0.3 * d[:, None]), 0, 255).astype(np.uint8)
    trough = np.clip(mid * (0.8 - 0.1 * d[:, None]), 0, 255).astype(np.uint8)
    hw = 1.5 + np.arange(rows) * 0.34
    return par, speed, crest, trough, hw


def draw_sea(c, f, cam, under=None):
    rows = H - HZ
    pat = _sea_pattern()
    par, speed, crest, trough, hw = _sea_rowdata()
    c.a[HZ:H] = _sea_base().a
    if under is not None:
        under(c, f, cam)
    out = c.a[HZ:H]
    shift = (np.round(cam * par + f * speed).astype(int) + np.arange(rows) * 37) % 640
    idx = (np.arange(W)[None, :] + shift[:, None]) % 640
    p = pat[np.arange(rows)[:, None], idx]
    cm = p == 1
    tm = p == 2
    out[tm] = trough[np.nonzero(tm)[0]]
    out[cm] = crest[np.nonzero(cm)[0]]
    # sun glitter path (follows the sun, not the camera)
    xx = np.arange(W)[None, :]
    dx = np.abs(xx - SUN[0])
    cone = dx < hw[:, None]
    out[cm & cone] = C('butter')
    out[tm & cone] = C('#e89a50')
    # flickering gold dashes in the cone
    for i in range(60):
        r = int(hash01(i, 3) ** 1.3 * (rows - 2)) + 1
        on = (f // 3 + int(hash01(i, 4) * 7)) % 7
        if on > 3:
            continue
        x = SUN[0] + int((hash01(i, 5) - 0.5) * 1.8 * hw[r])
        y = HZ + r
        L = 1 + int(r / 14)
        c.hline(x - L, x + L, y, 'white' if on == 1 else ('butter' if on == 2 else 'gold'))


# ---- the crag of Mount Aris (+ harbour + palace on top), layer origin world (CRX, 0)
CW, CH = 180, 136
CBASE = 133
PLAT = 53
CRAG_POLY = [(150, 134), (153, 120), (156, 108), (158, 96), (161, 86), (165, 76), (168, 68), (172, 60), (175, 53),
             (233, 53), (236, 60), (240, 68), (244, 78), (247, 88), (250, 98), (253, 110), (256, 120), (262, 134)]
BUTTRESS = [
    [(138, 135), (142, 122), (148, 112), (155, 110), (159, 120), (161, 135)],
    [(155, 110), (157, 94), (161, 84), (167, 82), (168, 96), (165, 110)],
    [(250, 135), (253, 118), (258, 108), (265, 110), (270, 122), (276, 135)],
    [(246, 102), (247, 86), (244, 74), (249, 72), (254, 86), (256, 104)],
]


def _rock_paint(a, m, seed, bias_x=0.55):
    h, w = m.shape
    ys, xs = np.nonzero(m)
    x0, x1 = xs.min(), xs.max()
    yy, xx = np.mgrid[0:h, 0:w]
    tx = (xx - x0) / max(1, x1 - x0)
    nz = fbm(h, w, 6, seed) - 0.5
    strata = np.sin(yy * 0.62 + nz * 7.0)
    bias = (tx - 0.5) * bias_x + nz * 0.35 - (strata > 0.86) * 0.22 + ((strata > 0.55) & (strata < 0.7)) * 0.10
    ramp = ['#3b2e45', '#55435a', '#76616e', '#9a8482', '#bea995', '#d9c6a8', '#efdcb8', '#ffe8c4']
    return shade_blob(a, m, ramp, light=(0.85, -0.45), rad=9, bias=bias - 0.02, outline='#33283d',
                      rim='#fff1d4', rim_k=0.9, dith=0.9)


@lru_cache(maxsize=1)
def _crag():
    cv = new(CW, CH)
    a = cv.a
    sh = lambda pts: [(x - CRX, y) for (x, y) in pts]
    m = pmask(CW, CH, sh(CRAG_POLY))
    _rock_paint(a, m, 11)
    allm = m.copy()
    for i, b in enumerate(BUTTRESS):
        bm = pmask(CW, CH, sh(b))
        _rock_paint(a, bm, 20 + i, bias_x=0.9)
        allm |= bm
    # vertical cracks
    for i in range(16):
        x = int(lerp(152, 260, hash01(i, 1))) - CRX
        y = int(lerp(62, 124, hash01(i, 2)))
        L = 4 + int(hash01(i, 3) * 8)
        for k in range(L):
            xx = x + (k // 3) * (1 if hash01(i, 4) > 0.5 else -1)
            if 0 <= xx < CW and y + k < CH and allm[y + k, xx]:
                a[y + k, xx] = mix(a[y + k, xx], '#3b2e45', 0.6)
    # stairway: white zigzag with shadow + railing posts
    for (xa, ya), (xb, yb) in zip(STAIR, STAIR[1:]):
        xa, xb = xa - CRX, xb - CRX
        cv.line(xa, ya + 2, xb, yb + 2, '#55435a')
        cv.line(xa, ya + 1, xb, yb + 1, '#c8bfb8')
        cv.line(xa, ya, xb, yb, 'paper')
        n = max(abs(xb - xa), 1)
        for k in range(0, n, 3):
            x = int(lerp(xa, xb, k / n))
            y = int(round(lerp(ya, yb, k / n)))
            cv.px(x, y - 1, '#fff6e0')
            if k % 6 == 0:
                cv.px(x, y + 1, '#9a8f8a')
    # shrubs on ledges (not on the stairs)
    stair_m = np.zeros((CH, CW), bool)
    for (xa, ya), (xb, yb) in zip(STAIR, STAIR[1:]):
        n = max(abs(xb - xa), 1)
        for k in range(n + 1):
            x = int(lerp(xa, xb, k / n)) - CRX
            y = int(round(lerp(ya, yb, k / n)))
            stair_m[max(0, y - 3):y + 4, max(0, x - 2):x + 3] = True
    for i in range(60):
        x = int(lerp(150, 262, hash01(i, 11))) - CRX
        y = int(lerp(60, 128, hash01(i, 12)))
        if not (0 <= x < CW and allm[y, x] and allm[y - 2, x] and not stair_m[y, x]):
            continue
        r = 1 + int(hash01(i, 13) * 2)
        cv.ellipse(x, y, r + 1, r, '#2f5a3e')
        cv.ellipse(x, y - 1, r, max(1, r - 1), '#4f8a4a')
        cv.px(x + 1, y - 1 - (r > 1), '#9ccf68')
        if hash01(i, 14) > 0.55:
            cv.px(x - 1, y - 1, 'hot_pink')
    # caves / carved arched windows (lit per frame)
    for (x, y) in CAVES:
        x -= CRX
        cv.rect(x - 2, y - 3, 5, 5, '#3b2e45')
        cv.rect(x - 1, y - 4, 3, 1, '#3b2e45')
        cv.rect(x - 1, y - 2, 3, 3, '#5b2a24')
        cv.hline(x - 2, x + 2, y + 2, '#efdcb8')
    for (x, y) in LANDINGS:
        x -= CRX
        cv.rect(x - 3, y - 1, 7, 3, 'paper')
        cv.hline(x - 3, x + 3, y + 2, '#76616e')
        cv.vline(x + 2, y - 4, y - 1, '#4a3a4a')
    # harbour quay at the crag foot (left)
    q0, q1 = 124 - CRX, 162 - CRX
    cv.rect(q0, 125, q1 - q0, 8, '#e8dcc4')
    cv.hline(q0, q1 - 1, 125, 'white')
    cv.rect(q0, 130, q1 - q0, 3, '#bfae96')
    for x in range(q0 + 3, q1 - 3, 8):
        cv.rect(x, 128, 4, 5, '#5a4658')
        cv.px(x + 1, 127, '#5a4658')
        cv.px(x + 2, 127, '#5a4658')
    cv.rect(q0 + 2, 119, 7, 6, 'paper')           # harbour house
    cv.vline(q0 + 8, 119, 124, '#d8c6ac')
    cv.tri((q0 + 1, 119), (q0 + 9, 119), (q0 + 5, 115), 'clay')
    cv.px(q0 + 4, 121, '#2d72ad')
    cv.rect(q0 + 11, 121, 4, 4, 'cream')
    cv.hline(q0 + 10, q0 + 15, 120, 'copper')
    # ---------------- palace on the plateau
    P = PLAT
    cv.rect(169 - CRX, P - 8, 70, 9, '#f4ead8')
    cv.hline(169 - CRX, 238 - CRX, P - 8, 'white')
    cv.rect(169 - CRX, P - 1, 70, 2, '#c9b8a0')
    for x in range(172, 236, 4):
        cv.rect(x - CRX, P - 6, 2, 4, '#4b5f7a')
        cv.px(x - CRX, P - 6, '#f4ead8')
    cv.vline(238 - CRX, P - 8, P, '#c9b8a0')
    for (x, w_, h_) in ((166, 9, 6), (231, 10, 7), (175, 7, 5), (223, 7, 5)):
        cv.rect(x - CRX, P - 8 - h_, w_, h_, '#fbf3e4')
        cv.vline(x - CRX + w_ - 1, P - 8 - h_, P - 9, '#d8c6ac')
        cv.rect(x - CRX - 1, P - 10 - h_, w_ + 2, 2, 'clay')
        cv.hline(x - CRX - 1, x - CRX + w_, P - 10 - h_, 'coral')
        cv.px(x - CRX + w_ // 2, P - 6 - h_, '#2d4f7a')
    for tx in (184, 224):
        x = tx - CRX
        cv.rect(x - 4, P - 27, 9, 19, '#fbf3e4')
        cv.vline(x + 4, P - 27, P - 9, '#d8c6ac')
        cv.vline(x + 3, P - 27, P - 9, '#ece0cc')
        cv.vline(x - 4, P - 27, P - 9, '#e6d6bd')
        for wy in (P - 23, P - 16):
            cv.rect(x - 1, wy, 2, 3, '#2d4f7a')
            cv.px(x - 1, wy, '#fbf3e4')
        cv.hline(x - 5, x + 5, P - 28, '#e3b24f')
        cv.tri((x - 5, P - 28), (x + 6, P - 28), (x, P - 38), '#2d7fb8')
        cv.tri((x + 1, P - 28), (x + 6, P - 28), (x + 1, P - 37), '#1f5f96')
        cv.line(x - 3, P - 30, x - 1, P - 34, '#6fb8e8')
        cv.px(x, P - 39, 'gold')
    dx = 204 - CRX
    cv.rect(dx - 9, P - 16, 19, 9, '#fbf3e4')
    cv.vline(dx + 9, P - 16, P - 8, '#d8c6ac')
    for x in range(dx - 7, dx + 8, 3):
        cv.rect(x, P - 14, 2, 4, '#2d4f7a')
        cv.px(x, P - 14, '#fbf3e4')
    cv.hline(dx - 10, dx + 10, P - 17, '#c99a3a')
    R, top = 9, P - 17
    Hd = 26
    dm = np.zeros((CH, CW), bool)
    for k in range(Hd):
        u = k / Hd
        r = R * (0.82 + 0.55 * math.sin(math.pi * u * 0.9)) * (1 - u) ** 0.75
        y = top - k
        dm[y, int(round(dx - r)):int(round(dx + r)) + 1] = True
    shade_blob(a, dm, ['#4a2618', '#7a3a1e', '#a8582a', '#cf8a32', '#e8b440', '#f9d25a', '#fff2a0'],
               light=(0.9, -0.3), rad=6, outline='#3e2014', rim='#fff8d0', rim_k=0.9)
    for k in range(2, Hd - 3):
        u = k / Hd
        r = R * (0.82 + 0.55 * math.sin(math.pi * u * 0.9)) * (1 - u) ** 0.75
        y = top - k
        for s in (-0.5, 0.3):
            x = int(round(dx + r * s))
            if dm[y, x]:
                a[y, x] = mix(a[y, x], '#5a2a14', 0.45)
    cv.vline(dx, top - Hd - 5, top - Hd + 1, '#e8b440')
    cv.px(dx, top - Hd - 6, 'butter')
    cv.px(dx - 1, top - Hd - 3, '#c99a3a')
    cv.px(dx + 1, top - Hd - 3, '#c99a3a')
    for x in (172, 193, 215, 236):
        x -= CRX
        cv.ellipse(x, P - 13, 1, 5, '#1f4a36')
        cv.vline(x + 1, P - 16, P - 10, '#3f7a4a')
        cv.px(x, P - 19, '#2f6040')
    for i in range(18):
        x = int(lerp(168, 239, hash01(i, 31))) - CRX
        cv.px(x, P - 9, 'hot_pink' if i % 3 else 'bubblegum')
        cv.px(x + 1, P - 9 + (i % 2), 'rose')
    for i in range(7):
        x = int(lerp(170, 236, hash01(i, 41))) - CRX
        L = 3 + int(hash01(i, 42) * 5)
        for k in range(L):
            cv.px(x + (k % 2), P - 8 + k, 'hot_pink' if k % 3 else 'magenta')
    return rgba(cv), allm


def draw_crag(c, f, cam):
    lay, _ = _crag()
    ox = CRX - int(round(cam * P_CRAG))
    c.blit(lay, ox, 0)
    sx = lambda x: x - int(round(cam * P_CRAG))
    for i, (x, y) in enumerate(LANDINGS):
        lantern(c, sx(x + 2), y - 5, f, seed=i)
    for i, x in enumerate((130, 144, 158)):
        c.vline(sx(x), 121, 124, '#4a3a4a')
        lantern(c, sx(x), 120, f, seed=10 + i)
    for i, (x, y) in enumerate(CAVES):
        k = 0.6 + 0.4 * math.sin(f * 0.5 + i * 1.7)
        cc = mix('#c8642a', '#ffb84a', k)
        c.rect(sx(x) - 1, y - 2, 3, 3, cc)
        c.px(sx(x), y - 3, cc)
    flag(c, sx(184), PLAT - 45, f, '#e3b24f', length=5, seed=0, ph=6)
    flag(c, sx(224), PLAT - 45, f, '#e3b24f', length=5, seed=1.3, ph=6)
    for i, (x, y) in enumerate(((183, PLAT - 23), (223, PLAT - 23), (198, PLAT - 14), (207, PLAT - 14),
                                (184, PLAT - 16))):
        if (f // 6 + i * 3) % 11 == 0:
            c.px(sx(x), y, 'butter')
    g = (f - 72) % 54
    if g < 10:
        yy = PLAT - 22 - g * 2
        c.px(sx(206), yy, 'white')
        c.px(sx(207), yy, '#fff2a0')
    for i, (x, clr) in enumerate(((116, 'hot_pink'), (126, 'aqua'))):
        b = int(round(math.sin(f * 0.18 + i * 2) * 0.6))
        y = 132 + b
        c.hline(sx(x), sx(x) + 6, y, clr)
        c.hline(sx(x) + 1, sx(x) + 5, y + 1, shade(clr, 0.6))
        c.px(sx(x) + 3, y - 1, 'paper')
    for i in range(28):
        x = int(lerp(126, 276, i / 27))
        ph = math.sin(f * 0.15 + i * 1.3)
        if ph > -0.3:
            c.hline(sx(x) - 1, sx(x) + 1 + int(ph * 2), CBASE + (i % 2), 'white' if ph > 0.4 else '#cfe7f7')


@lru_cache(maxsize=1)
def _crag_reflect():
    lay, _ = _crag()
    src = lay[CBASE - 34:CBASE + 1][::-1]
    seaC = np.array(col('#2a68a4'), np.float32)
    rgb = src[..., :3].astype(np.float32) * 0.5 + seaC * 0.5
    out = src.copy()
    out[..., :3] = np.clip(rgb * 0.82, 0, 255).astype(np.uint8)
    th = bayer(out.shape[0], out.shape[1])
    fade_ = (np.arange(out.shape[0]) / out.shape[0])[:, None]
    out[..., 3] = np.where((th > fade_ * 1.05 - 0.05) & (src[..., 3] > 0), 255, 0)
    return out


def draw_reflection(c, f, cam):
    ref = _crag_reflect()
    ox = CRX - int(round(cam * P_CRAG))
    for r in range(ref.shape[0]):
        y = CBASE + 1 + r
        if y >= H:
            break
        off = int(round(math.sin(r * 0.9 + f * 0.25) * (0.5 + r * 0.05)))
        row = ref[r]
        x0 = ox + off
        a0, a1 = max(0, x0), min(W, x0 + row.shape[0])
        if a1 <= a0:
            continue
        seg_ = row[a0 - x0:a1 - x0]
        m = seg_[:, 3] > 0
        c.a[y, a0:a1][m] = seg_[m, :3]


# ---- the sailing ship (no crew), p 0.7
@lru_cache(maxsize=1)
def _ship():
    cv = new(26, 24)
    cv.poly([(1, 17), (25, 17), (22, 22), (4, 22)], '#8a3a2a')
    cv.hline(2, 24, 17, '#c8642a')
    cv.hline(3, 23, 19, 'paper')
    cv.hline(4, 22, 22, '#5a2620')
    for x in (6, 10, 14, 18):
        cv.px(x, 20, '#3b2420')
    cv.vline(12, 2, 17, '#5a3a2a')
    cv.vline(19, 6, 17, '#5a3a2a')
    cv.poly([(4, 4), (11, 3), (11, 15), (3, 14)], 'cream')
    cv.poly([(8, 3), (11, 3), (11, 15), (8, 15)], '#f0dcc0')
    cv.hline(4, 10, 8, 'hot_pink')
    cv.hline(4, 10, 9, 'hot_pink')
    cv.poly([(13, 7), (18, 7), (18, 15), (13, 16)], 'cream')
    cv.hline(13, 17, 11, 'aqua')
    cv.tri((20, 7), (20, 16), (25, 16), '#fff6e0')
    return rgba(cv)


def draw_ship(c, f, cam):
    x = 96 + (f - 72) * 0.2 - cam * 0.7
    b = int(round(math.sin(f * 0.16) * 0.8))
    xi = int(round(x))
    y = 120 + b
    c.blit(_ship(), xi, y)
    flag(c, xi + 12, y + 1, f, 'hot_pink', length=4, pole=False)
    for i in range(4):
        L = 3 + i
        if (f // 4 + i) % 3:
            c.hline(xi - 2 - i * 5, xi - 2 - i * 5 + L, y + 23 + (i % 2), '#e6f2fa' if i < 2 else '#a8cbe6')


# ---- Basalt Beach: hexagonal columns in the foreground, p=P_FG; layer origin world (BX0, BY0)
BX0, BY0 = -30, 88
BWID = 440
WL_BACK, WL_BACK_L = 160, 166


def _columns():
    back, front = [], []
    x, i = 226, 0
    while x < 430:
        w = 12 + int(hash01(i, 1) * 6)
        if x < 300:
            top = int(148 + 4 * math.sin(i * 1.7) - hash01(i, 2) * 4)
        else:
            top = int(146 - (x - 300) * 0.42 - hash01(i, 2) * 9)
        back.append((x, w, top, WL_BACK))
        x += w - 1
        i += 1
    x = 296
    while x < 430:
        w = 14 + int(hash01(i, 4) * 7)
        top = int(170 - max(0, x - 310) * 0.33 - hash01(i, 5) * 8)
        front.append((x, w, top, 181))
        x += w + int(hash01(i, 6) * 6) - 2
        i += 1
    x = -26
    while x < 58:
        w = 9 + int(hash01(i, 7) * 5)
        top = int(146 + (x + 26) * 0.2 + hash01(i, 8) * 8)
        back.append((x, w, top, WL_BACK_L))
        x += w - 1
        i += 1
    x = -26
    while x < 34:
        w = 10 + int(hash01(i, 9) * 5)
        top = int(166 + hash01(i, 10) * 6)
        front.append((x, w, top, 181))
        x += w + 1
        i += 1
    return back, front


COL_BACK, COL_FRONT = _columns()
COLUMNS = COL_BACK + COL_FRONT


def _draw_column(cv, j, x, w, top, wl):
    th = 3 if w < 12 else 4
    s1 = x + max(2, w // 4)
    s2 = x + w - max(2, w // 4)
    body0 = top + th
    hh = wl - body0 + 2
    cv.rect(x, body0 - 1, s1 - x, hh, '#3a3448')
    cv.rect(s1, body0 - 1, s2 - s1, hh, '#5c566e')
    cv.rect(s2, body0 - 1, x + w - s2, hh, '#8a8198')
    cv.dither(s1 + 1, body0, s2 - s1 - 1, hh - 2, '#504a62', 0.22, ox=j)
    cv.dither(s2 + 1, body0, x + w - s2 - 2, hh - 2, '#9d93a8', 0.25, ox=j)
    cv.vline(s1, body0, wl, '#6e6882')
    cv.vline(s2, body0, wl, '#a49bb0')
    cv.vline(x + w - 1, body0, wl, '#e8b48a')
    cv.vline(x, body0, wl, '#262132')
    y = body0 + 5 + int(hash01(j, 1) * 6)
    while y < wl - 4:
        cv.hline(x + 1, x + w - 2, y, '#28233a')
        cv.hline(x + 1, s1 - 1, y + 1, '#4c4660')
        cv.hline(s1 + 1, s2 - 1, y + 1, '#78718c')
        cv.hline(s2 + 1, x + w - 2, y + 1, '#c0b4c4')
        y += 6 + int(hash01(j, y) * 8)
    for k in range(7):
        yy = wl - k
        cv.dither(x + 1, yy, w - 2, 1, '#2c4f4a' if k > 2 else '#1f2c36', 0.85 - k * 0.11, ox=j)
    for k in range(3):
        if hash01(j, k, 9) > 0.4:
            cv.px(x + 1 + int(hash01(j, k, 8) * (w - 2)), wl - 3 - int(hash01(j, k, 7) * 4), '#5f9a4a')
    # hexagonal top face
    cx_ = x + w // 2
    pts = [(x, top + th // 2), (s1, top), (s2, top), (x + w, top + th // 2), (s2, top + th), (s1, top + th)]
    cv.poly(pts, '#9a93ab')
    cv.poly([(s1, top), (s2, top), (x + w, top + th // 2), (cx_, top + th // 2)], '#c4bccf')
    cv.hline(s1, s2 - 1, top, '#ece4ee')
    cv.px(x + w - 2, top + 1, '#ffd8b0')
    cv.px(s2, top, '#fff0dc')
    cv.line(x, top + th // 2, s1, top + th, '#4e4862')
    cv.hline(s1, s2, top + th, '#4e4862')
    cv.line(s2, top + th, x + w - 1, top + th // 2 + 1, '#8a8198')
    cv.line(x, top + th // 2, s1, top, '#6e6882')
    if hash01(j, 5) > 0.6:
        cv.hline(s1 + 1, s2 - 2, top + th // 2, '#8fb6e0')
        cv.px(s2 - 2, top + th // 2, '#e6f2fa')


@lru_cache(maxsize=1)
def _basalt():
    hh = H - BY0 + 4
    cv = new(BWID, hh)
    for j, (x, w, top, wl) in enumerate(COL_BACK):
        _draw_column(cv, j, x - BX0, w, top - BY0, wl - BY0)
    back = rgba(cv)
    cv2 = new(BWID, hh)
    for j, (x, w, top, wl) in enumerate(COL_FRONT):
        _draw_column(cv2, 100 + j, x - BX0, w, top - BY0, wl - BY0)
    front = rgba(cv2)
    return back, front


def fg_shift(cam):
    return int(round(cam * P_FG))


def _foam(c, f, j, X, w, wl):
    surge = math.sin(f * 2 * math.pi / 72 + j * 0.9)
    lvl = int(round(surge * 1.5))
    y = wl + lvl
    c.hline(X - 1, X + w, y, 'white')
    c.hline(X - 3, X + w + 2, y + 1, '#d6ecf8')
    c.dither(X - 4, y + 2, w + 8, 1, '#d6ecf8', 0.5, ox=f // 3)
    for k in range(3):
        bx = X + int(hash01(j, k, f // 4) * w)
        c.px(bx, y - 1, 'white' if surge > 0 else '#d6ecf8')


def draw_basalt(c, f, cam):
    s = fg_shift(cam)
    back, front = _basalt()
    c.blit(back, BX0 - s, BY0)
    for j, (x, w, top, wl) in enumerate(COL_BACK):
        X = x - s
        if -16 < X < W + 4:
            _foam(c, f, j, X, w, wl)
    # spray bursts in the gaps, every other beat
    for burst in range(8):
        f0 = 76 + burst * 18
        if not (f0 <= f < f0 + 16):
            continue
        j = int(hash01(burst, 77) * len(COL_BACK))
        x, w, top, wl = COL_BACK[j]
        X = x - s
        t = f - f0
        for k in range(12):
            vx = (hash01(burst, k) - 0.5) * 1.8
            vy = -2.0 - hash01(burst, k, 2) * 1.8
            px = X + w // 2 + vx * t
            py = wl - 1 + vy * t + 0.16 * t * t
            if py < wl:
                c.px(int(px), int(py), 'white' if t < 9 else '#d6ecf8')
    c.blit(front, BX0 - s, BY0)
    for j, (x, w, top, wl) in enumerate(COL_FRONT):
        X = x - s
        if -16 < X < W + 4:
            _foam(c, f, 100 + j, X, w, 177)


def render_a(f):
    cam = cam_a(f)
    c = _sky_a().copy()
    for (x, y, w, h, s, p) in ((-10, 14, 84, 26, 3, 0.04), (104, 30, 60, 20, 5, 0.05), (170, 4, 100, 30, 7, 0.06),
                               (300, 36, 50, 16, 9, 0.05)):
        c.blit(_cloud(w, h, s), int(round(x - cam * p + (f - 72) * 0.08)), y)
    draw_storm(c, f, cam)
    c.blit(_far_isles(), -int(round(cam * 0.15)) - 20, HZ - 11)
    draw_sea(c, f, cam, under=draw_reflection)
    draw_crag(c, f, cam)
    bx = int(round(262 - cam * P_CRAG + (f - 72) * 0.07))
    draw_balloon(c, bx, 36, f, size='s')
    draw_ship(c, f, cam)
    draw_basalt(c, f, cam)
    fx.grade(c, mul=(1.03, 1.0, 0.95))
    s = fg_shift(cam)
    jx, jw, jt, _ = COL_BACK[7]
    callout(c, jx + jw // 2 - s, jt + 1, 'BASALT BEACH', f, 90, 162, side='up', length=12)
    ax, ay = storm_anchor(cam)
    callout(c, ax, ay, "STORM KING'S REALM", f, 108, 174, side='down', length=30)
    return c


# =====================================================================================  SHOT B
# world coords of the summit; screen = world - cam * parallax
HZB = 132                      # horizon (world, far layer p=0.25)
SUNB = (300, 100)
BW_, BH_ = 380, 280
DOME_X, DOME_BASE, DOME_H, DOME_R = 178, 106, 72, 26
WHITE_RAMP = ['#b8a088', '#d2bc9e', '#e8d8be', '#f6ecd8', '#fff8ea']
BLUE_RAMP = ['#173a6a', '#1f4f86', '#2a6aa8', '#3f8ccc', '#6fb4e6', '#b0e0fa']


def cam_b(f):
    cx = cam_cruise(f, CUT1, DIVE - 1, 0, 18)
    cy = cam_cruise(f, CUT1, DIVE - 26, 20, 0)
    if f >= 196:                                         # director pass: longer, gentler swoop
        cy = max(cy, int(round(90 * seg(f, DIVE - 20, DIVE) ** 2)))
    return cx, cy


def _cyl(cv, x0, y0, w, h, ramp=WHITE_RAMP, peak=0.72):
    """Vertical cylinder lit from the right."""
    r = ramp_arr(ramp)
    n = len(r)
    for i in range(w):
        t = (i + 0.5) / w
        v = 1 - abs(t - peak) / max(peak, 1 - peak)
        v = v * 0.85 + (0.15 if t > 0.5 else 0)
        if i == w - 1:
            v = 0.45
        k = int(clamp(v, 0, 0.999) * n)
        cv.rect(x0 + i, y0, 1, h, tuple(r[k]))
        if 0 < k < n and (i % 2):
            cv.dither(x0 + i, y0, 1, h, tuple(r[k - 1]), 0.25)


def _cone(cv, cx, ybase, rw, hgt, ramp=BLUE_RAMP):
    m = pmask(cv.w, cv.h, [(cx - rw - 1, ybase), (cx + rw + 1, ybase), (cx, ybase - hgt)])
    yy, xx = np.mgrid[0:cv.h, 0:cv.w]
    half = np.maximum(1, (ybase - yy) / hgt * 0 + rw * (1 - (ybase - yy) / hgt) + 1)
    t = clamp(0.5, 0, 1) + (xx - cx) / (2 * half) * 0.9
    t = t + ((ybase - yy) % 3 == 0) * -0.18
    r = ramp_arr(ramp)
    n = len(r)
    th = bayer(cv.h, cv.w)
    idx = np.clip(np.floor(np.clip(t, 0, 0.999) * n + (th - 0.5) * 0.7), 0, n - 1).astype(int)
    cv.a[m] = r[idx][m]
    cv.a[edge_of(m)] = col('#122a4e')
    cv.hline(cx - rw - 1, cx + rw + 1, ybase, '#e3b24f')
    cv.hline(cx - rw, cx + rw, ybase + 1, '#a8782a')


def _arch(cv, x, y, w, h, inner='#334866', frame=None, lit=False):
    """Round-topped opening, top-left (x, y)."""
    r = w // 2
    cv.rect(x, y + r, w, h - r, inner)
    cv.ellipse(x + r - (0 if w % 2 else 0.5), y + r, r - 0.5 + (0 if w % 2 else 0), r, inner)
    if lit:
        cv.rect(x + 1, y + h - max(2, h // 3), w - 2, max(2, h // 3), '#f0a040')
        cv.hline(x + 1, x + w - 2, y + h - max(2, h // 3), '#ffd070')
    else:
        cv.vline(x + w - 1, y + r, y + h - 1, mix(inner, 'white', 0.25))
    if frame:
        cv.hline(x - 1, x + w, y + h, frame)


def _tile_roof(cv, x0, x1, y, rows=3, overhang=1):
    for k in range(rows):
        yy = y + k
        cv.hline(x0 - overhang, x1 + overhang, yy, 'clay' if k < rows - 1 else '#a85a34')
        for x in range(x0 - overhang + (k % 2) * 2, x1 + overhang + 1, 4):
            cv.px(x, yy, '#c86e40' if k < rows - 1 else '#7a3a24')
    cv.hline(x0 - overhang, x1 + overhang, y, 'coral')


def _cypress(cv, x, ybase, hgt, w=3):
    m = np.zeros((cv.h, cv.w), bool)
    for k in range(hgt):
        u = k / hgt
        r = w * math.sin(math.pi * min(1, 0.15 + u * 0.95)) ** 0.7
        y = ybase - k
        if 0 <= y < cv.h:
            m[y, max(0, int(round(x - r))):int(round(x + r)) + 1] = True
    shade_blob(cv.a, m, ['#123a2c', '#1d4f38', '#2c6a44', '#46864e', '#7aa85a'], light=(0.9, -0.3), rad=2,
               noise=(fbm(cv.h, cv.w, 2, int(x)) - 0.5) * 0.5, outline='#0e2a22')


def _round_tree(cv, x, ybase, r, seed, fruit=None):
    cv.vline(x, ybase - r, ybase, '#5a3a2a')
    m = np.zeros((cv.h, cv.w), bool)
    for i in range(5):
        ax = x + int((hash01(seed, i) - 0.5) * r * 1.4)
        ay = ybase - r - int(hash01(seed, i, 2) * r * 0.9)
        m |= emask(cv.w, cv.h, ax, ay, r * 0.7, r * 0.6)
    shade_blob(cv.a, m, ['#1f4a30', '#2f6a3a', '#4a8a44', '#78b050', '#b4d870'], light=(0.9, -0.5), rad=3,
               noise=(fbm(cv.h, cv.w, 3, seed) - 0.5) * 0.4, outline='#163826')
    if fruit:
        ys, xs = np.nonzero(m)
        for i in range(len(xs) // 25):
            j = int(hash01(seed, i, 7) * len(xs))
            cv.px(xs[j], ys[j], fruit)


def _bougain(cv, x, y, n, seed, spread=6):
    for i in range(n):
        dx = int((hash01(seed, i) - 0.5) * spread * 2)
        dy = int(hash01(seed, i, 2) ** 1.5 * spread * 2)
        cc = ('hot_pink', 'magenta', 'bubblegum', 'rose')[int(hash01(seed, i, 3) * 4)]
        cv.px(x + dx, y + dy, cc)
        if hash01(seed, i, 4) > 0.5:
            cv.px(x + dx + 1, y + dy, '#2f6a3a')


@lru_cache(maxsize=1)
def _sky_b():
    c = Canvas(W, H + 12)
    c.gradient(0, 0, W, c.h, ['#4a6cc4', '#5d88d8', '#78a8e6', '#a4c8ee', '#e8d8cc', '#ffd0a0', '#ffb880'],
               soft=0.4)
    fx.glow(c, SUNB[0], SUNB[1], 120, '#ff9a40', 0.22, rings=6)
    fx.glow(c, SUNB[0], SUNB[1], 40, '#ffd080', 0.35, rings=4)
    c.circle(SUNB[0], SUNB[1], 10, '#ffe9a8')
    c.circle(SUNB[0], SUNB[1], 9, '#fff6d0')
    c.circle(SUNB[0], SUNB[1], 7, 'white')
    return c


@lru_cache(maxsize=1)
def _far_b():
    """Distant sea below the horizon (p 0.25) incl. tiny storm isle on the left."""
    h = 100
    c = Canvas(W + 20, h)
    c.gradient(0, 0, c.w, h, ['#b6c6dc', '#8ab0d8', '#5c98cc', '#3c7cb8', '#2c66a2'], soft=0.8)
    fx.glow(c, SUNB[0], 0, 90, '#ff9a40', 0.18, rings=5)
    for r in range(2, h, 2):
        for k in range(int(6 + r * 0.6)):
            x = int(hash01(r, k) * c.w)
            L = 1 + int(r / 18)
            c.hline(x, x + L, r, mix(c.a[r, min(c.w - 1, x)], 'white', 0.35))
    return c


@lru_cache(maxsize=1)
def _palace():
    cv = new(BW_, BH_)
    a = cv.a
    T = 152                                     # main terrace level
    # --- rock summit + cliff below
    rock = pmask(BW_, BH_, [(0, T + 2), (BW_, T + 2), (BW_, BH_), (0, BH_)])
    rock &= ~pmask(BW_, BH_, [(0, 176), (24, 190), (52, 214), (74, 246), (84, BH_), (0, BH_)])
    rock &= ~pmask(BW_, BH_, [(BW_, 172), (352, 186), (324, 214), (306, 248), (296, BH_), (BW_, BH_)])
    _rock_paint(a, rock, 77, bias_x=0.5)
    # retaining wall under the terrace
    cv.rect(0, T, BW_, 10, '#e8d8be')
    for x in range(0, BW_, 6):
        cv.vline(x, T + 1, T + 9, '#d2bc9e')
    for y in (T + 3, T + 6):
        cv.hline(0, BW_ - 1, y, '#d2bc9e')
    cv.hline(0, BW_ - 1, T, '#fff8ea')
    cv.hline(0, BW_ - 1, T + 10, '#9a8482')
    # stairway top flights on the cliff + lanterns posts (lanterns per frame)
    for (xa, ya, xb, yb) in ((120, 260, 230, 236), (230, 236, 140, 212), (140, 212, 214, 190),
                             (214, 190, 170, 166)):
        cv.line(xa, ya + 2, xb, yb + 2, '#55435a')
        cv.line(xa, ya + 1, xb, yb + 1, '#c8bfb8')
        cv.line(xa, ya, xb, yb, 'paper')
        n = abs(xb - xa)
        for k in range(0, n, 3):
            x = int(lerp(xa, xb, k / n))
            y = int(round(lerp(ya, yb, k / n)))
            cv.px(x, y - 1, '#fff6e0')
            if k % 9 == 0:
                cv.vline(x, y - 3, y - 1, '#e8d8be')
    # shrubs on the cliff
    for i in range(40):
        x = int(hash01(i, 61) * BW_)
        y = int(lerp(T + 16, BH_ - 6, hash01(i, 62)))
        if not rock[y, x]:
            continue
        r = 2 + int(hash01(i, 63) * 3)
        cv.ellipse(x, y, r + 1, r, '#2f5a3e')
        cv.ellipse(x, y - 1, r, r - 1, '#4f8a4a')
        cv.px(x + 1, y - r, '#9ccf68')
        if hash01(i, 64) > 0.5:
            _bougain(cv, x, y - 1, 6, i, spread=2)
    # --- far-left turret with small teal dome
    _cyl(cv, 14, 100, 16, T - 100)
    cv.ellipse(22, 100, 9, 9, '#1f8a8a')
    shade_blob(a, emask(BW_, BH_, 22, 99, 8, 9) & (np.mgrid[0:BH_, 0:BW_][0] <= 99),
               ['#0b5e65', '#0b8a8f', '#2aa8a0', '#6ad0c0', '#b8f0e0'], light=(0.9, -0.4), rad=4, outline='#08404a')
    cv.hline(13, 30, 100, '#e3b24f')
    cv.vline(22, 86, 90, '#e3b24f')
    cv.px(22, 85, 'butter')
    for y in (110, 126):
        _arch(cv, 19, y, 5, 9, lit=(y == 126))
    # --- wings (left + right) with terracotta roofs
    for (x0, x1, top) in ((30, 72, 116), (276, 340, 112)):
        cv.rect(x0, top, x1 - x0, T - top, '#f6ecd8')
        cv.rect(x1 - 3, top, 3, T - top, '#e0ccb0')
        cv.dither(x0, top, 6, T - top, '#e0ccb0', 0.4)
        _tile_roof(cv, x0, x1, top - 3, rows=3, overhang=2)
        for x in range(x0 + 4, x1 - 6, 9):
            _arch(cv, x, top + 8, 5, 10, lit=hash01(x, 1) > 0.6)
            cv.hline(x - 1, x + 5, top + 18, '#d2bc9e')
            _arch(cv, x, top + 24, 5, 9)
        # balcony railing + flowers
        cv.hline(x0, x1 - 1, top + 21, '#c8b090')
        for x in range(x0, x1, 2):
            cv.px(x, top + 22, '#e8d8be')
        for i in range(10):
            _bougain(cv, x0 + 3 + i * (x1 - x0 - 6) // 10, top + 20, 4, x0 + i, spread=2)
    # pergola roof garden on the right wing
    for x in range(282, 336, 6):
        cv.vline(x, 100, 109, '#e8d8be')
    cv.hline(280, 337, 100, '#c8a070')
    for i in range(40):
        _bougain(cv, 282 + int(hash01(i, 71) * 54), 99, 1, i, spread=1)
    # --- towers (cylinders with blue cones)
    for (tx, top, w) in ((84, 74, 22), (262, 70, 22)):
        _cyl(cv, tx - w // 2, top, w, T - top)
        cv.hline(tx - w // 2 - 1, tx + w // 2, top, '#e3b24f')
        for y in range(top + 6, T - 10, 18):
            _arch(cv, tx - 3, y, 6, 10, lit=(y > top + 30 and hash01(tx, y) > 0.4))
        # balcony band
        cv.rect(tx - w // 2 - 2, top + 30, w + 4, 2, '#d2bc9e')
        cv.hline(tx - w // 2 - 2, tx + w // 2 + 1, top + 30, '#fff8ea')
        for x in range(tx - w // 2 - 1, tx + w // 2 + 1, 2):
            cv.vline(x, top + 26, top + 29, '#e8d8be')
        cv.hline(tx - w // 2 - 2, tx + w // 2 + 1, top + 25, '#c8b090')
        _bougain(cv, tx - w // 2, top + 32, 10, tx, spread=3)
        _cone(cv, tx, top - 1, w // 2 + 2, 34)
        cv.vline(tx, top - 42, top - 35, '#c99a3a')
        cv.px(tx, top - 43, 'butter')
    # --- main hall (under the dome)
    hx0, hx1, htop = 122, 236, 120
    cv.rect(hx0, htop, hx1 - hx0, T - htop, '#f6ecd8')
    cv.rect(hx1 - 4, htop, 4, T - htop, '#e0ccb0')
    cv.hline(hx0, hx1 - 1, htop, '#fff8ea')
    _tile_roof(cv, hx0, hx1, htop - 3, rows=2, overhang=2)
    for i, x in enumerate(range(hx0 + 6, hx1 - 8, 12)):
        big = abs(x + 4 - DOME_X) < 7
        if big:
            _arch(cv, x - 2, htop + 8, 12, 24, inner='#2a3a5a', lit=True)
            cv.rect(x - 1, htop + 20, 10, 12, '#7a4a2a')
            cv.vline(x + 4, htop + 20, htop + 31, '#5a3420')
            cv.px(x + 2, htop + 26, 'gold')
            cv.px(x + 6, htop + 26, 'gold')
        else:
            _arch(cv, x, htop + 10, 8, 18, inner='#334866', lit=hash01(i, 5) > 0.5)
        cv.vline(x - 2, htop + 6, T - 1, '#fff8ea')
        cv.vline(x - 1, htop + 6, T - 1, '#e0ccb0')
    cv.hline(hx0, hx1 - 1, htop + 6, '#d2bc9e')
    # --- drum
    dx0, dx1 = DOME_X - 30, DOME_X + 30
    dtop = DOME_BASE
    _cyl(cv, dx0, dtop, dx1 - dx0, htop - 3 - dtop)
    for x in range(dx0 + 3, dx1 - 4, 7):
        _arch(cv, x, dtop + 3, 4, 9, inner='#2a4a6a')
        cv.px(x + 1, dtop + 2, '#e3b24f')
    cv.hline(dx0 - 1, dx1, dtop, '#e3b24f')
    cv.hline(dx0 - 1, dx1, dtop + 1, '#a8782a')
    cv.hline(dx0, dx1 - 1, htop - 4, '#c8b090')
    # --- the bronze-gold bud dome
    dm = np.zeros((BH_, BW_), bool)
    prof = []
    for k in range(DOME_H):
        u = k / DOME_H
        r = DOME_R * (0.86 + 0.5 * math.sin(math.pi * u * 0.92)) * (1 - u) ** 0.8
        y = dtop - 1 - k
        prof.append((y, r))
        dm[y, int(round(DOME_X - r)):int(round(DOME_X + r)) + 1] = True
    yy, xx = np.mgrid[0:BH_, 0:BW_]
    nz = (fbm(BH_, BW_, 3, 5) - 0.5) * 0.12
    t = shade_blob(a, dm, ['#3e1e12', '#5e2e18', '#86441e', '#ae6626', '#d08e30', '#e8b440', '#f8d660',
                           '#fff0a0'], light=(0.95, -0.25), rad=14, noise=nz, outline='#2e160c',
                   rim='#fff8d8', rim_k=0.92)
    # petal ribs (embossed), curving with the profile
    for (y, r) in prof[3:-4]:
        for s_ in (-0.78, -0.42, 0.0, 0.42, 0.78):
            x = int(round(DOME_X + r * s_))
            if dm[y, x]:
                a[y, x] = mix(a[y, x], '#3e1e12', 0.55)
                if dm[y, x + 1]:
                    a[y, x + 1] = mix(a[y, x + 1], '#fff0a0', 0.35)
    # decorative ring + oculi near the base
    for (y, r) in prof[6:8]:
        cv.hline(int(DOME_X - r) + 1, int(DOME_X + r) - 1, y, '#5e2e18')
    for (y, r) in prof[7:8]:
        for s_ in (-0.62, -0.2, 0.2, 0.62):
            x = int(round(DOME_X + r * s_))
            cv.ellipse(x, y - 4, 1.5, 1.5, '#5e2e18')
            cv.px(x, y - 4, '#2a4a6a')
    # spire + sun finial
    tip = dtop - DOME_H
    cv.vline(DOME_X, tip - 14, tip + 1, '#e8b440')
    cv.vline(DOME_X + 1, tip - 10, tip + 1, '#a8782a')
    cv.ellipse(DOME_X, tip - 6, 2, 1, '#e8b440')
    cv.circle(DOME_X, tip - 17, 2, '#f8d660')
    for (ddx, ddy) in ((0, -4), (0, 4), (-4, 0), (4, 0), (-3, -3), (3, -3), (-3, 3), (3, 3)):
        cv.px(DOME_X + ddx, tip - 17 + ddy, '#e8b440')
    cv.px(DOME_X, tip - 17, 'white')
    # --- trees + cypresses on the terrace
    for (x, hgt) in ((40, 34), (112, 40), (246, 42), (350, 30), (6, 28)):
        _cypress(cv, x, T - 1, hgt, w=3 if hgt > 32 else 2)
    _round_tree(cv, 302, T - 1, 9, 3, fruit='orange')
    _round_tree(cv, 62, T - 1, 7, 4, fruit='amber')
    # flowers along the terrace edge + planters
    for i in range(70):
        x = int(hash01(i, 81) * BW_)
        cv.px(x, T - 1, ('hot_pink', 'bubblegum', 'magenta', 'butter')[i % 4])
        if i % 3 == 0:
            cv.px(x + 1, T - 2, '#3f7a4a')
    for x in (100, 150, 210, 228, 312):
        cv.rect(x - 3, T - 5, 7, 5, '#c86e40')
        cv.hline(x - 3, x + 3, T - 5, '#e89a60')
        _bougain(cv, x, T - 8, 12, x, spread=3)
    return rgba(cv), dm, t


@lru_cache(maxsize=1)
def _fg_b():
    """Foreground balustrade + big cypress + potted flowers (p 1.25), world origin (0, 150)."""
    w, h = 420, 120
    cv = new(w, h)
    y0 = 16                                     # balustrade top (world 166)
    cv.rect(0, y0, w, 4, '#f6ecd8')
    cv.hline(0, w - 1, y0, '#fff8ea')
    cv.hline(0, w - 1, y0 + 3, '#c8b090')
    for x in range(2, w, 7):
        cv.rect(x, y0 + 4, 4, 14, '#e8d8be')
        cv.rect(x + 1, y0 + 6, 2, 9, '#d2bc9e')
        cv.vline(x + 3, y0 + 4, y0 + 17, '#fff8ea')
        cv.hline(x, x + 3, y0 + 9, '#fff8ea')
    cv.rect(0, y0 + 18, w, 4, '#e0ccb0')
    cv.hline(0, w - 1, y0 + 18, '#fff8ea')
    cv.rect(0, y0 + 22, w, 3, '#c8b090')
    cv.hline(0, w - 1, y0 + 24, '#9a8482')
    for x in (30, 150, 270, 390):          # posts with pots
        cv.rect(x - 5, y0 - 2, 11, 24, '#f6ecd8')
        cv.vline(x + 5, y0 - 2, y0 + 21, '#d2bc9e')
        cv.hline(x - 5, x + 5, y0 - 2, '#fff8ea')
        cv.rect(x - 4, y0 - 9, 9, 7, '#c86e40')
        cv.hline(x - 4, x + 4, y0 - 9, '#e89a60')
        cv.vline(x + 4, y0 - 9, y0 - 3, '#a85a34')
        _bougain(cv, x, y0 - 14, 26, x, spread=5)
    # draping bougainvillea on the balustrade
    for i in range(10):
        x = int(hash01(i, 91) * w)
        for k in range(16):
            cv.px(x + int((hash01(i, k) - 0.5) * 8), y0 + int(hash01(i, k, 2) * 10),
                  ('hot_pink', 'magenta', 'bubblegum', '#2f6a3a')[k % 4])
    return rgba(cv)


@lru_cache(maxsize=1)
def _fg_cypress():
    cv = new(24, 120)
    _cypress(cv, 11, 119, 116, w=9)
    return rgba(cv)


BANNERS_B = [(140, 122), (216, 122)]
FLAGS_B = [(84, 74 - 44), (262, 70 - 44), (22, 85)]
LANTERNS_B = [(132, 148), (226, 148), (176, 220), (150, 210), (222, 190), (186, 172)]


def _fountain(c, f, x, y):
    c.rect(x - 7, y - 3, 15, 4, '#e8d8be')
    c.hline(x - 7, x + 7, y - 3, '#fff8ea')
    c.hline(x - 6, x + 6, y - 2, '#5fa0d8')
    c.rect(x - 1, y - 8, 3, 5, '#e8d8be')
    c.ellipse(x, y - 9, 3, 1, '#e8d8be')
    for i in range(10):
        t = ((f * 0.9 + i * 3.7) % 14) / 14
        side = -1 if i % 2 else 1
        px = x + side * t * 6
        py = y - 10 - 9 * t + 14 * t * t
        c.px(int(px), int(py), 'white' if t < 0.5 else '#a8d8f8')
    c.vline(x, y - 15 + (f // 3) % 2, y - 10, '#e6f6ff')


def render_b(f):
    cx, cy = cam_b(f)
    c = Canvas()
    sky = _sky_b()
    c.blit(sky, 0, -int(round(cy * 0.1)))
    for (x, y, w, h, s) in ((-6, 30, 90, 28, 13), (120, 14, 70, 22, 15), (236, 50, 70, 20, 17)):
        c.blit(_cloud(w, h, s), int(round(x - cx * 0.08 + (f - 144) * 0.1)), int(round(y - cy * 0.1)))
    hz = HZB - int(round(cy * 0.25))
    far = _far_b()
    c.blit(far, -int(round(cx * 0.25)), hz)
    # sun glitter on the far sea
    for i in range(30):
        r = int(hash01(i, 3) ** 1.4 * 60)
        if (f // 3 + i) % 5 > 2:
            continue
        x = SUNB[0] + int((hash01(i, 5) - 0.5) * (4 + r * 0.6))
        c.hline(x - 1, x + 1 + r // 20, hz + 1 + r, 'butter' if i % 3 else 'white')
    lay, dm, _ = _palace()
    ox, oy = -cx, -cy
    c.blit(lay, ox, oy)
    # dome glint sweep (every 2 beats a bright band crosses the dome left -> right)
    g = (f - CUT1) % 36
    if g < 14:
        bx = DOME_X - DOME_R + int(g / 13 * 2 * DOME_R)
        band = np.zeros_like(dm)
        band[:, max(0, bx - 1):bx + 2] = True
        band &= dm
        band[:, bx - 1] &= (np.arange(BH_) % 2 == 0)
        add_light(c, ox, oy, BW_, BH_, band, '#fff4c0', 0.55)
    fx.sparkle(c, DOME_X + ox, DOME_BASE - DOME_H - 17 + oy, f, 'white', period=36, size=3)
    fx.sparkle(c, DOME_X + 14 + ox, DOME_BASE - 40 + oy, f, 'butter', period=36, phase=0.5, size=2)
    # banners (long vertical pennants waving) on the hall facade
    for i, (x, y) in enumerate(BANNERS_B):
        X, Y = x + ox, y + oy
        c.hline(X - 1, X + 6, Y, '#a8782a')
        for k in range(22):
            sw = int(round(math.sin(f * 0.22 - k * 0.3 + i) * (k / 22) * 1.6))
            ww = 6 if k < 19 else 6 - (k - 18) * 2
            x0 = X + sw + (6 - ww) // 2
            c.hline(x0, x0 + ww - 1, Y + 1 + k, '#0b8a8f' if k % 10 else '#e3b24f')
            c.px(x0 + ww - 1, Y + 1 + k, '#0b5e65')
        c.circle(X + 3 + int(round(math.sin(f * 0.22 - 2.4 + i) * 0.5)), Y + 9, 2, '#f8d660')
        c.px(X + 3, Y + 9, '#fff0a0')
    for i, (x, y) in enumerate(FLAGS_B):
        flag(c, x + ox, y + oy, f, '#e3b24f' if i < 2 else '#0b8a8f', length=7, seed=i, ph=8)
    for i, (x, y) in enumerate(LANTERNS_B):
        c.vline(x + ox, y + oy + 1, y + oy + 4, '#4a3a4a')
        lantern(c, x + ox, y + oy, f, seed=i)
    _fountain(c, f, 196 + ox, 151 + oy)
    # windows catching the sun
    for i, (x, y) in enumerate(((90, 86), (268, 82), (300, 128), (52, 126))):
        if (f // 4 + i * 5) % 16 < 2:
            c.px(x + ox, y + oy, 'butter')
    # balloon moored to the left tower balcony
    bxw, byw = 40, 78
    bx, by = bxw + ox, byw + oy
    bob = int(round(math.sin(f * 2 * math.pi / 72.0) * 1.4))
    c.line(bx + 2, by + bob, 74 + ox, 104 + oy, '#8a6a4a')
    draw_balloon(c, bx, by, f, size='m')
    # foreground (p 1.25)
    fy = 150 - int(round(cy * 1.25))
    c.blit(_fg_cypress(), -int(round(cx * 1.25)) + 300, fy - 88)
    c.blit(_fg_b(), -int(round(cx * 1.25)) - 20, fy)
    fx.grade(c, mul=(1.04, 1.0, 0.94))
    callout(c, DOME_X + DOME_R - 2 + ox, DOME_BASE - 30 + oy, 'MOUNT ARIS', f, CUT1 + 18, DIVE - 24, side='right', length=14)
    return c


# =====================================================================================  underwater kit
CAUST_N = 16                    # caustic loop: 16 steps x 2 frames


@lru_cache(maxsize=1)
def _caustics():
    h, w = H + 40, W + 120
    n1 = fbm(h, w, 14, 901, octaves=2)
    n2 = fbm(h, w, 14, 902, octaves=2)
    out = []
    for i in range(CAUST_N):
        ph = 2 * math.pi * i / CAUST_N
        v = n1 * math.cos(ph) + n2 * math.sin(ph)
        v = v / (abs(math.cos(ph)) + abs(math.sin(ph)))
        m = (np.abs(v - 0.5) < 0.022) | (np.abs(v - 0.62) < 0.016) | (np.abs(v - 0.38) < 0.016)
        out.append(m)
    return out


def caustic_light(c, f, solid, y0, color='#c8fff4', k=0.35, ox=0, oy=0, fade_h=170):
    """Moving caustic network added on `solid` pixels below screen row y0, fading with depth."""
    m = _caustics()[(f // 2) % CAUST_N]
    ox = int(ox) % 100
    oy = int(oy) % 30
    sub = m[oy:oy + H, ox:ox + W] & solid
    if y0 > 0:
        sub[:max(0, min(H, y0))] = False
    rows = np.arange(H)
    fadev = np.clip(1 - (rows - y0) / fade_h, 0.15, 1.0)
    reg = c.a.astype(np.int16)
    add = (np.array(col(color), np.float32)[None, :] * (k * fadev)[:, None]).astype(np.int16)
    ys, xs = np.nonzero(sub)
    reg[ys, xs] += add[ys]
    c.a[:] = np.clip(reg, 0, 255).astype(np.uint8)


def blit_track(c, lay, x, y, solid):
    """Blit an RGBA layer and record its opaque pixels in `solid`."""
    x, y = int(x), int(y)
    sh, sw = lay.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = lay[y0 - y:y1 - y, x0 - x:x1 - x]
    m = sub[:, :, 3] >= 128
    c.a[y0:y1, x0:x1][m] = sub[:, :, :3][m]
    solid[y0:y1, x0:x1] |= m


def bubble(c, x, y, r, rim='#bff4ff', hi='white'):
    x, y = int(x), int(y)
    if r <= 0:
        c.px(x, y, hi)
    elif r == 1:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c.px(x + dx, y + dy, rim)
        c.px(x - 1, y - 1, hi)
    else:
        c.circle(x, y, r, rim, fill=False)
        c.px(x - r + 1, y - r + 1, hi)
        c.px(x - r + 2, y - r + 1, hi)


def bubble_stream(c, f, x, ybot, ytop, n=8, seed=0, speed=1.0, big=False):
    span = max(1, ybot - ytop)
    for i in range(n):
        t = ((f * speed * (0.8 + 0.4 * hash01(seed, i)) + hash01(seed, i, 1) * span) % span)
        y = ybot - t
        x_ = x + math.sin(t * 0.18 + i) * (1 + t / span * 2)
        r = 0 if hash01(seed, i, 2) < 0.4 else (1 if not big or hash01(seed, i, 3) < 0.6 else 2)
        bubble(c, x_, y, r)


def kelp(c, x, ybase, hgt, f, seed, cols=('#1f4a2a', '#3f7a32', '#6aa040', '#a8d060'), amp=4.0, lean=0.0,
         leaf_every=5, width=1):
    """Swaying kelp strand with alternating blades. All numpy, no per-pixel python loops beyond nodes."""
    k = np.arange(hgt)
    u = k / max(1, hgt - 1)
    sway = np.sin(f * 0.07 + seed * 1.7 - k * 0.11) * amp * u ** 1.3 + lean * u * hgt
    xs = np.round(x + sway).astype(int)
    ys = ybase - k
    put(c, xs, ys, cols[1])
    if width > 1:
        put(c, xs + 1, ys, cols[0])
    put(c, xs[::3] - 1, ys[::3], cols[2])
    # blades
    for j in range(2, hgt - 2, leaf_every):
        side = 1 if (j // leaf_every + seed) % 2 else -1
        L = 3 + int(4 * math.sin(math.pi * min(1, j / hgt + 0.15)))
        fl = math.sin(f * 0.09 + seed + j * 0.3)
        bx, by = xs[j], ys[j]
        pts_x = [bx + side * (q + 1) for q in range(L)]
        pts_y = [by - int((q + 1) * (0.6 + 0.3 * fl)) for q in range(L)]
        put(c, pts_x, pts_y, cols[2])
        put(c, pts_x[:-1], [p + 1 for p in pts_y[:-1]], cols[1])
        put(c, pts_x[1:2], pts_y[1:2], cols[3])
    put(c, xs[-1:], ys[-1:] - 1, cols[3])


# =====================================================================================  SHOT C
SURF0 = 104


def dive_D(f):
    """Camera depth below the start (0 at DIVE -> 98 at DIVE+13 -> 104 at the dissolve end)."""
    if f < DIVE + 13:
        return 98 * ease_out(seg(f, DIVE, DIVE + 13))
    return 98 + 6 * cruise(seg(f, DIVE + 13, CUT3 + 8))


def cam_c(f):
    return cam_cruise(f, DIVE + 4, CUT3 + 8, 0, 14)


WATER_C = ['#6fd8d4', '#3fb6c4', '#2896b8', '#1f7aa8', '#18629a', '#154e88', '#123c72', '#10305e']


@lru_cache(maxsize=1)
def _water_c():
    c = Canvas(W, 320)
    c.gradient(0, 0, W, 320, WATER_C, soft=0.7)
    return c


@lru_cache(maxsize=1)
def _far_c():
    w, h = W + 80, 170
    cv = new(w, h)
    a = cv.a
    yy = np.mgrid[0:h, 0:w][0]
    m = pmask(w, h, [(0, h), (0, 96), (20, 90), (40, 100), (60, 84), (76, 80), (92, 94), (118, 104), (150, 98),
                     (176, 106), (210, 100), (240, 92), (262, 96), (290, 104), (330, 96), (360, 88), (w, 92), (w, h)])
    m |= pmask(w, h, [(60, 84), (66, 50), (74, 40), (84, 44), (90, 60), (92, 94)])
    m |= pmask(w, h, [(232, 96), (236, 62), (246, 54), (254, 70), (258, 98)])
    arch = emask(w, h, 78, 82, 8, 14)
    m &= ~arch
    shade_blob(a, m, ['#1a5a86', '#1f6890', '#26789c', '#3088a8'], light=(0.3, -0.9), rad=6,
               bias=-(yy - 60) / 200, outline=None, dith=0.9)
    for i in range(16):
        x = int(hash01(i, 51) * w)
        hgt = 20 + int(hash01(i, 52) * 30)
        for k in range(hgt):
            xx = x + int(math.sin(k * 0.2 + i) * 2)
            yy_ = 100 - k + int(hash01(i, 53) * 10)
            if 0 <= xx < w and 0 <= yy_ < h:
                a[yy_, xx] = col('#2a7a90')
    # tiny far spires of Seaquestria (right, glowing)
    for (x, ht, cc) in ((W - 26, 30, '#6a3a9a'), (W - 16, 44, '#7a3aa0'), (W - 6, 26, '#5a3a8a'),
                        (W + 2, 36, '#6a3a9a'), (W + 10, 22, '#5a3a8a')):
        cv.tri((x - 3, 98), (x + 3, 98), (x, 98 - ht), cc)
        cv.px(x, 98 - ht, '#ff9ae0')
    return rgba(cv)


@lru_cache(maxsize=1)
def _mid_c():
    w, h = W + 100, 230
    cv = new(w, h)
    a = cv.a
    yy, xx = np.mgrid[0:h, 0:w]
    sea = 118
    sand = pmask(w, h, [(0, h), (0, sea + 4)] + [(x, sea + int(3 * math.sin(x * 0.05))) for x in range(0, w + 1, 10)] +
                 [(w, h)])
    a[sand] = col('#8aa89a')
    cv.dither(0, sea, w, h - sea, '#76988e', 0.35)
    for i in range(30):
        x = int(hash01(i, 71) * w)
        y = sea + 6 + int(hash01(i, 72) * 30)
        cv.hline(x, x + 6 + int(hash01(i, 73) * 8), y, '#9cb8a8')
    # rocks with coral tufts
    for i, (x, y, rx, ry) in enumerate(((30, sea, 22, 14), (130, sea + 2, 16, 10), (226, sea - 2, 26, 18),
                                        (330, sea + 2, 20, 12), (380, sea, 14, 9))):
        rm = emask(w, h, x, y, rx, ry) & (yy <= y + 4)
        shade_blob(a, rm, ['#1f4a6a', '#2a5e7e', '#3a748e', '#5a8ea0', '#86b0b8'], light=(0.3, -0.95), rad=5,
                   noise=(fbm(h, w, 4, 300 + i) - 0.5) * 0.4, outline='#163a58')
        for k in range(7):
            cx_ = x + int((hash01(i, k) - 0.5) * rx * 1.5)
            cy_ = y - int(ry * 0.85 * (1 - (abs(cx_ - x) / (rx + 1)) ** 2) ** 0.5) + 1
            cc = ('#e8607a', '#f08a40', '#b860c8', '#f0a0c0', '#e8d060', '#4fd1c5', '#ff7aa8')[(i * 3 + k) % 7]
            kind = (i + k) % 3
            if kind == 0:                       # branching coral
                for b in range(5):
                    bx = cx_ + int((hash01(i, k, b) - 0.5) * 8)
                    by = cy_ - 4 - int(hash01(i, k, b, 2) * 5)
                    cv.line(cx_, cy_, bx, by, shade(cc, 0.8))
                    cv.px(bx, by, mix(cc, 'white', 0.45))
            elif kind == 1:                     # brain / mound coral
                cv.ellipse(cx_, cy_ - 2, 3, 2, shade(cc, 0.75))
                cv.ellipse(cx_, cy_ - 3, 2, 1, cc)
                cv.px(cx_ + 1, cy_ - 4, mix(cc, 'white', 0.5))
            else:                               # tube coral
                for b in range(3):
                    bx = cx_ - 2 + b * 2
                    hh = 3 + int(hash01(i, k, b) * 4)
                    cv.vline(bx, cy_ - hh, cy_, shade(cc, 0.8))
                    cv.px(bx, cy_ - hh, mix(cc, 'white', 0.5))
    return rgba(cv)


@lru_cache(maxsize=1)
def _near_c():
    w, h = W + 120, 240
    cv = new(w, h)
    a = cv.a
    sea = 164
    sand = pmask(w, h, [(0, h)] + [(x, sea + int(2 * math.sin(x * 0.04 + 1))) for x in range(0, w + 1, 8)] +
                 [(w, h)])
    a[sand] = col('#d8c49a')
    yy, xx = np.mgrid[0:h, 0:w]
    rip = np.sin(xx * 0.32 + np.sin(yy * 0.5) * 2 + yy * 1.4)
    a[sand & (rip > 0.75)] = col('#f0e0b8')
    a[sand & (rip < -0.8)] = col('#b8a07a')
    cv.dither(0, sea, w, 3, '#b8c8a8', 0.5)
    # pebbles + shells
    for i in range(24):
        x = int(hash01(i, 81) * w)
        y = sea + 3 + int(hash01(i, 82) * 12)
        cv.ellipse(x, y, 1 + int(hash01(i, 83) * 2), 1, ('#8a7a6a', '#a89080', '#e8d0c0')[i % 3])
    # rocks with sea fans
    for i, (x, rx, ry) in enumerate(((26, 30, 24), (300, 22, 14), (410, 16, 10))):
        rm = emask(w, h, x, sea + 2, rx, ry) & (yy <= sea + 4)
        shade_blob(a, rm, ['#2a3a5a', '#3a5070', '#4e6884', '#6e8ea0', '#a0c0c8'], light=(0.3, -0.95), rad=5,
                   noise=(fbm(h, w, 4, 400 + i) - 0.5) * 0.4, outline='#1e2a44')
        fxx = x + rx // 3
        base = sea + 2 - ry
        for b in range(9):
            ang = -math.pi / 2 + (b - 4) * 0.22
            L = 10 + int(hash01(i, b) * 6)
            ex, ey = fxx + int(math.cos(ang) * L), base + int(math.sin(ang) * L)
            cv.line(fxx, base, ex, ey, '#e8506a')
            cv.px(ex, ey, '#ff9aa8')
        cv.line(fxx - 3, base - 6, fxx + 3, base - 6, '#c83a5a')
        cv.line(fxx - 5, base - 10, fxx + 5, base - 10, '#c83a5a')
        for k in range(6):
            cx_ = x - rx // 2 + int(hash01(i, k, 5) * rx)
            cy_ = sea + 3 - int(ry * (1 - ((cx_ - x) / (rx + 1)) ** 2) ** 0.5)
            cc = ('#ff7aa8', '#f08a40', '#b860c8', '#e8d060', '#4fd1c5', '#e8607a')[(i + k) % 6]
            if k % 2:
                cv.ellipse(cx_, cy_ - 2, 3, 2, shade(cc, 0.7))
                cv.ellipse(cx_, cy_ - 3, 2, 1, cc)
                cv.px(cx_ + 1, cy_ - 4, mix(cc, 'white', 0.5))
            else:
                for b in range(4):
                    hh = 3 + int(hash01(i, k, b) * 5)
                    cv.vline(cx_ - 3 + b * 2, cy_ - hh, cy_, shade(cc, 0.8))
                    cv.px(cx_ - 3 + b * 2, cy_ - hh, mix(cc, 'white', 0.5))
    return rgba(cv)


@lru_cache(maxsize=1)
def _sub_crag():
    """The submerged foot of Mount Aris (far layer), rising from the deep to above the surface."""
    w, h = 180, 240
    cv = new(w, h)
    a = cv.a
    yy, xx = np.mgrid[0:h, 0:w]
    pts = [(50, 0), (120, 0), (126, 40), (134, 80), (146, 120), (162, 160), (176, 196), (178, h),
           (0, h), (2, 190), (14, 150), (30, 110), (40, 70), (46, 30)]
    m = pmask(w, h, pts)
    nz = fbm(h, w, 6, 333) - 0.5
    strata = np.sin(yy * 0.5 + nz * 6)
    bias = -(yy / h) * 0.45 + nz * 0.3 - (strata > 0.85) * 0.2 + 0.1
    shade_blob(a, m, ['#1a6290', '#1e6c9a', '#2276a2', '#287fa8', '#2e88ae', '#3692b4'], light=(0.25, -0.95),
               rad=10, bias=bias, dith=0.9)
    for i in range(14):
        x = int(lerp(20, 160, hash01(i, 341)))
        y = int(lerp(40, 200, hash01(i, 342)))
        if m[y, x] and m[max(0, y - 3), x]:
            cc = ('#4a78aa', '#3a8a9e', '#5a72a8', '#4a8a8a')[i % 4]
            cv.ellipse(x, y, 2 + int(hash01(i, 343) * 3), 1 + int(hash01(i, 344) * 2), cc)
    return rgba(cv)


CLAMS = [(96, 168), (182, 172), (250, 169), (356, 171)]
KELP_MID = [(14, 120, 70), (34, 121, 52), (52, 119, 84), (96, 121, 60), (112, 120, 76), (150, 120, 92),
            (178, 121, 66), (196, 121, 48), (262, 118, 88), (280, 120, 58), (300, 120, 72), (350, 119, 94),
            (366, 121, 60), (392, 121, 64)]
KELP_NEAR = [(-4, 186, 170, 3), (8, 184, 140, 2), (136, 182, 120, 2), (322, 185, 160, 3), (338, 186, 130, 2),
             (420, 184, 150, 3)]


def draw_clam(c, x, y, f, seed):
    c.ellipse(x, y, 8, 3, '#5a3a6a')
    c.ellipse(x, y - 1, 7, 2, '#b088c0')
    c.hline(x - 6, x + 6, y - 2, '#e8c8e8')
    c.hline(x - 5, x + 5, y - 1, '#f4dcf4')
    # open top shell (fan) tilted back, ribs radiating
    for k in range(-7, 8):
        ex, ey = x + k, y - 9 - int((49 - k * k) ** 0.5 * 0.35)
        c.line(x + k // 3, y - 3, ex, ey, '#c8a0d8' if k % 2 else '#8a6a9a')
    for k in range(-7, 8):
        c.px(x + k, y - 9 - int((49 - k * k) ** 0.5 * 0.35), '#f4e0f4')
    c.hline(x - 6, x + 6, y - 3, '#6a4a7a')
    pulse = 0.5 + 0.5 * math.sin(f * 0.12 + seed)
    fx.glow(c, x, y - 4, 9, '#fff0ff', 0.25 + 0.2 * pulse, rings=3)
    c.circle(x, y - 4, 2, '#f0e8ff')
    c.px(x + 1, y - 3, '#c8b8e8')
    c.px(x - 1, y - 5, 'white')
    if (f + seed * 7) % 48 < 8:
        fx.sparkle(c, x + 1, y - 5, f, 'white', period=8, size=2)


def _surface_line(c, y, f):
    if y < -3 or y >= H:
        return
    xs = np.arange(W)
    off = np.round(np.sin(xs * 0.12 + f * 0.25) * 1.0 + np.sin(xs * 0.05 - f * 0.11)).astype(int)
    put(c, xs, y + off, 'white')
    put(c, xs, y + off + 1, '#c8fff8')
    m = (xs + f) % 7 < 3
    put(c, xs[m], y + off[m] + 2, '#9ef0f0')


@lru_cache(maxsize=1)
def _balloon_reflect_src():
    cv = new(24, 34)
    draw_balloon(cv, 12, 33, 0, size='m', bob=False)
    a = rgba(cv)[::-1]
    out = a.copy()
    rgb = a[..., :3].astype(np.float32)
    out[..., :3] = np.clip(rgb * 0.6 + np.array(col('#6fe0e8'), np.float32) * 0.4 + 10, 0, 255).astype(np.uint8)
    return out


def draw_balloon_reflection(c, f, x, ysurf):
    src = _balloon_reflect_src()
    h, w = src.shape[:2]
    th = bayer(h, w, 0, f // 3)
    for r in range(h):
        y = ysurf + 2 + r
        if y < 0 or y >= H:
            continue
        off = int(round(math.sin(r * 0.5 + f * 0.3) * (1 + r * 0.06)))
        row = src[r]
        x0 = x - w // 2 + off
        a0, a1 = max(0, x0), min(W, x0 + w)
        if a1 <= a0:
            continue
        seg_ = row[a0 - x0:a1 - x0]
        m = (seg_[:, 3] > 0) & (th[r, a0 - x0:a1 - x0] < 0.92 - r * 0.01)
        c.a[y, a0:a1][m] = seg_[m, :3]


def _above_water(c, f, surf, cx):
    """What is above the surface line (only rows < surf)."""
    if surf <= 0:
        return
    sky = _sky_b()
    hh = min(H, surf)
    c.a[:hh] = sky.a[60:60 + hh] if surf <= sky.h - 60 else sky.a[:hh]
    lay, _ = _crag()
    c.blit(lay[:CBASE + 1], 70 - cx, surf - CBASE - 1)
    # a thin strip of sea at the surface line (we are at water level)
    c.rect(0, surf - 2, W, 2, '#3d86c4')
    for i in range(20):
        x = int(hash01(i, 5) * W + f * 0.6) % W
        c.hline(x, x + 3, surf - 2, '#a8d0f0')


def draw_splash(c, f, surf, x0=164):
    t = f - DIVE
    if t < 0 or t > 28:
        return
    # white burst + mist at the impact point
    if t < 10:
        for i in range(18):
            ang = math.radians(-160 + 140 * hash01(i, 41))
            L = (8 + t * 3.5) * (0.5 + 0.6 * hash01(i, 42)) * (1 - t / 14)
            ex, ey = x0 + math.cos(ang) * L, surf + math.sin(ang) * L
            c.line(x0, surf - 1, int(ex), int(ey), 'white' if i % 3 else '#cfeeff')
            c.px(int(ex), int(ey) - 1, '#e8f8ff')
        c.dither(x0 - 10 - t, surf - 6 - t, 21 + 2 * t, 6 + t, '#e8f8ff', 0.35 * (1 - t / 10))
    # crown columns
    hgt = 40 * math.sin(math.pi * min(1, (t + 3) / 20))
    for k in range(-22, 23, 2):
        if abs(k) < 5:
            continue
        hh = int(hgt * (1 - abs(k) / 26) * (0.7 + 0.3 * hash01(k, 1)))
        if hh > 0:
            xx = x0 + k + (1 if k > 0 else -1) * t // 2
            c.vline(xx, surf - hh, surf - 1, 'white' if abs(k) % 4 else '#d8f4ff')
            c.px(xx, surf - hh - 1, '#d8f4ff')
    # droplets
    for i in range(90):
        tt = t + 2
        vx = (hash01(i, 1) - 0.5) * 6.0
        vy = -2.5 - hash01(i, 2) * 4.5
        px = x0 + vx * tt
        py = surf + vy * tt + 0.3 * tt * tt
        if py < surf - 1:
            c.px(int(px), int(py), 'white' if hash01(i, 3) > 0.3 else '#bfe8ff')
            if hash01(i, 4) > 0.6:
                c.px(int(px), int(py) + 1, '#bfe8ff')
    # foam ring on the surface
    r = 6 + t * 3
    for k in range(-r, r + 1):
        if abs(k) > r - 8 and 0 <= surf < H:
            c.px(x0 + k, surf, 'white')
            c.px(x0 + k, surf + 1, '#d8f4ff')
    # underwater bubble plume from the impact
    for i in range(50):
        st = hash01(i, 31) * 6
        tt = t - st
        if tt < 0:
            continue
        x = x0 + (hash01(i, 32) - 0.5) * (10 + tt * 4)
        y = surf + 4 + tt * (2.5 + hash01(i, 33) * 2) - 0.08 * tt * tt * 3
        if surf < y < H:
            bubble(c, x, y, 0 if hash01(i, 34) < 0.4 else (1 if hash01(i, 34) < 0.8 else 2))


def render_c(f):
    D = dive_D(f)
    surf = int(round(SURF0 - D))
    cx = cam_c(f)
    c = Canvas()
    solid = np.zeros((H, W), bool)
    # water (depth gradient anchored to the surface)
    wt = _water_c()
    y0 = max(0, surf)
    c.a[y0:] = wt.a[y0 - surf:y0 - surf + (H - y0)]
    # god rays from the surface
    for i in range(5):
        x = 10 + i * 70 - int(cx * 0.5) + int(math.sin(f * 0.03 + i * 1.9) * 8)
        wd = 4 + int(hash01(i, 9) * 6)
        k = 0.07 + 0.04 * math.sin(f * 0.05 + i * 2.3)
        fx.light_beam(c, ((x, surf + 2), (x + wd, surf + 2)), ((x + 40, surf + 160), (x + wd + 56, surf + 160)),
                      '#c8fff4', k)
    # far (p 0.4)
    fy = int(round(SURF0 - 30 - D * 0.4))
    fxo = -int(round(cx * 0.4)) - 10
    c.blit(_sub_crag(), fxo + 30, fy - 112)
    blit_track(c, _far_c(), fxo, fy, solid)
    pulse = 0.6 + 0.4 * math.sin(f * 2 * math.pi / 36)
    fx.glow(c, fxo + W - 4, fy + 86, 40, '#e46fb8', 0.28 * pulse, rings=4)
    fx.glow(c, fxo + W - 4, fy + 80, 18, '#4fd1c5', 0.2 * pulse, rings=3)
    # mid (p 0.7)
    my = int(round(SURF0 - 13 - D * 0.7))
    mxo = -int(round(cx * 0.7)) - 20
    blit_track(c, _mid_c(), mxo, my, solid)
    for i, (x, yb, hg) in enumerate(KELP_MID):
        kelp(c, x + mxo, yb + my, hg, f, i, cols=('#3a4a3a', '#5a6a3a', '#7a8a40', '#a8b060'), amp=3.0)
    # near (p 1.0)
    ny = int(round(SURF0 - D)) - 4
    nxo = -cx - 30
    blit_track(c, _near_c(), nxo, ny, solid)
    for i, (x, y) in enumerate(CLAMS):
        draw_clam(c, x + nxo, y + ny, f, i)
    bubble_stream(c, f, 210 + nxo, 166 + ny, surf + 4, n=10, seed=3, speed=1.1)
    bubble_stream(c, f, 70 + nxo, 166 + ny, surf + 4, n=8, seed=4, speed=0.9)
    caustic_light(c, f, solid, surf + 2, k=0.32, ox=cx * 0.7, oy=-D * 0.2)
    for i, (x, yb, hg, wd) in enumerate(KELP_NEAR):
        kelp(c, x + nxo, yb + ny, hg, f, 10 + i, cols=('#4a3a14', '#7a6a1e', '#b0982e', '#e8d060'), amp=5.0,
             width=wd, leaf_every=4)
    # surface from below + balloon reflection
    if surf < H:
        if surf > 0:
            _above_water(c, f, surf, cx)
        _surface_line(c, surf, f)
        if surf < 50:
            fx.light_rect(c, 0, surf + 2, W, 10, '#c8fff4', 0.18, fade='down')
            draw_balloon_reflection(c, f, 236 - int(cx * 0.3), surf)
    draw_splash(c, f, surf)
    # dive bubble cloud: the camera's own bubbles race up past the lens
    t = f - DIVE
    if 0 <= t < 34:
        for i in range(46):
            st = hash01(i, 21) * 14
            tt = t - st
            if tt < 0:
                continue
            x = 120 + (hash01(i, 22) - 0.5) * 220 + math.sin(tt * 0.4 + i) * 2
            y = 190 - tt * (7 + hash01(i, 23) * 6)
            if y < surf + 1:
                continue
            r = 0 if hash01(i, 24) < 0.3 else (1 if hash01(i, 24) < 0.7 else (2 if hash01(i, 24) < 0.92 else 3))
            bubble(c, x, y, r)
    if DIVE <= f < DIVE + 4:                                  # soft splash glow (director pass: was 0.55 white)
        c.blend(0, 0, W, H, 'white', 0.18 * (1 - (f - DIVE) / 4))
    return c


# =====================================================================================  SHOT D
MAG_RAMP = ['#2a1040', '#4a1658', '#7a1e78', '#a82a92', '#d04aaa', '#f07ac8', '#ffb8e8']
VIO_RAMP = ['#1e1450', '#33207a', '#4e2c9a', '#7444b8', '#a068d4', '#cc9aea', '#f0d0fa']
TEAL_RAMP = ['#08283e', '#0a3e52', '#0b5e65', '#11878a', '#2ab0a8', '#6ae0d0', '#c0fff0']
PEARL_RAMP = ['#2a2460', '#4a3e8a', '#7464b0', '#9e8ed2', '#c8baee', '#e8e0ff', '#ffffff']
CONCH_X, CONCH_BASE, CONCH_TOP = 206, 124, 18
DOME_D = (206, 152, 36)          # pearl dome centre-x, base y, radius
MIDW = W + 60


def cam_d(f):
    return cam_cruise(f, CUT3, N - 1, 0, 44)


def _quant(a, m, v, ramp, dith=0.8, ox=0, oy=0):
    r = ramp_arr(ramp)
    n = len(r)
    h, w = m.shape
    th = bayer(h, w, ox, oy)
    idx = np.clip(np.floor(np.clip(v, 0, 0.999) * n + (th - 0.5) * dith), 0, n - 1).astype(int)
    a[m] = r[idx][m]


def _spiral_cone(cv, cx, ybase, ytop, wbase, ramp, pitch=11.0, conc=0.85, outline='#1a0a2a', bias=0.0):
    h, w = cv.h, cv.w
    yy, xx = np.mgrid[0:h, 0:w]
    hh = ybase - ytop
    t = (ybase - yy) / hh
    r = wbase / 2 * np.clip(1 - t, 0, 1) ** conc
    m = (np.abs(xx - cx) <= r + 0.3) & (t >= 0) & (t <= 1)
    u = np.clip((xx - cx) / np.maximum(r, 0.6), -1, 1)
    th_ = np.arcsin(u)
    ph = (ybase - yy) / (pitch * (1 - 0.55 * np.clip(t, 0, 1))) + th_ / math.pi * 1.1
    fr = ph % 1.0
    v = 0.5 + 0.45 * np.cos(th_ - 0.6) - 0.2 + 0.22 * t + bias
    v = v - 0.32 * (fr < 0.16) + 0.22 * ((fr >= 0.16) & (fr < 0.24))
    _quant(cv.a, m, v, ramp)
    cv.a[edge_of(m)] = col(outline)
    return m


def _scallop(cv, cx, cy, R, ramp, nrib=11, outline='#1a0a3a'):
    h, w = cv.h, cv.w
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, yy - cy
    ang = np.arctan2(dy, dx)                      # upper half: -pi..0
    rr = np.sqrt(dx * dx + dy * dy)
    amin, amax = -math.pi + 0.35, -0.35
    s = (ang - amin) / (amax - amin)
    edge_r = R * (1 + 0.05 * np.cos(s * nrib * 2 * math.pi))
    m = (s >= 0) & (s <= 1) & (rr <= edge_r) & (dy <= 1)
    rib = np.cos(s * nrib * 2 * math.pi)
    v = 0.25 + 0.5 * (rr / R) + 0.22 * rib + 0.12 * (dx / R)
    _quant(cv.a, m, v, ramp)
    cv.a[edge_of(m)] = col(outline)
    # hinge ears
    cv.tri((cx - 7, cy + 1), (cx + 8, cy + 1), (cx, cy - 6), ramp[2])
    cv.hline(cx - 7, cx + 7, cy + 1, outline)
    return m


def _glow_window(cv, x, y, wdt, hgt, color):
    cv.rect(x, y + 1, wdt, hgt - 1, shade(color, 0.55))
    cv.hline(x + (1 if wdt > 2 else 0), x + wdt - 1 - (1 if wdt > 2 else 0), y, shade(color, 0.55))
    cv.rect(x + 1 if wdt > 2 else x, y + 2, max(1, wdt - 2), max(1, hgt - 3), color)


@lru_cache(maxsize=1)
def _water_d():
    c = Canvas()
    c.gradient(0, 0, W, H, ['#2a7ab0', '#1f5e9e', '#1c4688', '#20357a', '#24286a', '#22205a', '#1c1848'],
               soft=0.6)
    fx.glow(c, 180, 110, 150, '#a02a92', 0.18, rings=6)
    fx.glow(c, 180, 100, 70, '#e46fb8', 0.14, rings=4)
    return c


@lru_cache(maxsize=1)
def _far_d():
    w, h = W + 30, H
    cv = new(w, h)
    a = cv.a
    m = pmask(w, h, [(0, h), (0, 150)] + [(x, 144 + int(5 * math.sin(x * 0.06))) for x in range(0, w + 1, 12)] +
              [(w, h)])
    a[m] = col('#2a2468')
    for i, (x, hgt, wd) in enumerate(((20, 70, 14), (46, 50, 10), (84, 90, 16), (128, 60, 10), (262, 80, 14),
                                      (296, 58, 12), (322, 92, 16), (344, 50, 10), (150, 40, 8), (238, 44, 9))):
        _spiral_cone(cv, x, 150, 150 - hgt, wd, ['#241c5e', '#2c2270', '#36287e', '#40308c'], pitch=9,
                     outline='#1e1850')
        for k in range(3):
            yy = 150 - int(hgt * (0.2 + 0.25 * k))
            cv.px(x + (k % 2) - 1, yy, '#4fd1c5' if (i + k) % 2 else '#e46fb8')
    return rgba(cv)


@lru_cache(maxsize=1)
def _mid_d():
    w, h = MIDW, H
    cv = new(w, h)
    a = cv.a
    yy, xx = np.mgrid[0:h, 0:w]
    # coral reef terrace / seabed
    base = pmask(w, h, [(0, h), (0, 160), (40, 154), (90, 151), (150, 149), (260, 149), (320, 152), (w, 156),
                        (w, h)])
    shade_blob(a, base, VIO_RAMP[:5], light=(0.2, -1.0), rad=6, noise=(fbm(h, w, 4, 61) - 0.5) * 0.5,
               outline='#140e3a')
    # spiral coral spires (teal / violet)
    for (x, top, wd, ramp) in ((36, 60, 10, TEAL_RAMP), (140, 74, 9, VIO_RAMP), (276, 70, 9, TEAL_RAMP),
                               (346, 50, 11, VIO_RAMP), (170, 96, 7, TEAL_RAMP), (244, 92, 7, MAG_RAMP)):
        _spiral_cone(cv, x, 152, top, wd, ramp, pitch=8, conc=0.9)
    # scallop shell towers: coral columns topped by upright fans
    for (x, ytop, R, ramp) in ((112, 74, 22, VIO_RAMP), (300, 80, 20, MAG_RAMP)):
        col_m = (np.abs(xx - x) <= 6 - (yy - ytop) * 0.0) & (yy >= ytop) & (yy <= 152)
        col_m |= (np.abs(xx - x) <= 9) & (yy >= 136) & (yy <= 152)
        v = 0.55 + 0.35 * np.cos((xx - x) / 7.0 * 1.4 - 0.5) - 0.1 - ((yy % 9) == 0) * 0.3 + ((yy % 9) == 1) * 0.2
        _quant(a, col_m, v, TEAL_RAMP)
        a[edge_of(col_m)] = col('#08283e')
        for k in range(3):
            _glow_window(cv, x - 1, ytop + 12 + k * 15, 3, 6, '#7ff8ff')
        _scallop(cv, x, ytop + 2, R, ramp)
    # conch spire
    _spiral_cone(cv, CONCH_X, CONCH_BASE + 20, CONCH_TOP, 46, MAG_RAMP, pitch=13, conc=0.95)
    # pearl dome hall
    dx_, db, dr = DOME_D
    dm = emask(w, h, dx_, db, dr, int(dr * 0.85)) & (yy <= db)
    shade_blob(a, dm, PEARL_RAMP, light=(0.35, -0.95), rad=12, outline='#1e1850', rim='#c0fff0', rim_k=0.93)
    for s_ in (-0.66, -0.33, 0.0, 0.33, 0.66):        # dome ribs
        for k in range(int(dr * 0.85)):
            y = db - k
            half = dr * math.sqrt(max(0, 1 - (k / (dr * 0.85 + 0.5)) ** 2))
            x = int(round(dx_ + half * s_))
            if dm[y, x]:
                a[y, x] = mix(a[y, x], '#4a3e8a', 0.4)
    cv.hline(dx_ - dr + 2, dx_ + dr - 2, db - 8, '#7464b0')
    for k in range(-4, 5):                             # ring of windows
        x = dx_ + int(k * dr * 0.2)
        if k != 0:
            _glow_window(cv, x - 1, db - 16, 3, 6, '#7ff8ff')
    # grand gate
    cv.rect(dx_ - 7, db - 18, 15, 18, '#2a2460')
    cv.ellipse(dx_, db - 18, 7, 6, '#2a2460')
    cv.rect(dx_ - 5, db - 16, 11, 16, '#e46fb8')
    cv.ellipse(dx_, db - 16, 5, 4, '#e46fb8')
    cv.rect(dx_ - 4, db - 13, 9, 13, '#ff9ae0')
    cv.vline(dx_, db - 19, db - 1, '#a82a92')
    # coral arch bridges: dome <-> scallop towers
    for (xa, xb, yb_) in ((118, dx_ - dr + 6, 126), (dx_ + dr - 6, 294, 128)):
        for k in range(xa, xb + 1):
            t = (k - xa) / max(1, xb - xa)
            y = int(yb_ - 8 * math.sin(math.pi * t))
            cv.vline(k, y, y + 3, '#7444b8')
            cv.px(k, y, '#cc9aea')
            cv.px(k, y + 3, '#33207a')
        for k in range(xa + 4, xb - 2, 9):
            t = (k - xa) / max(1, xb - xa)
            y = int(yb_ - 8 * math.sin(math.pi * t))
            cv.vline(k, y + 4, y + 9, '#4e2c9a')
    # windows on the conch spire, following the spiral front
    for k in range(7):
        y = CONCH_BASE + 4 - k * 13
        tt = (CONCH_BASE + 20 - y) / (CONCH_BASE + 20 - CONCH_TOP)
        rr = 23 * (1 - tt) ** 0.95
        x = int(CONCH_X - rr * 0.35 + (k % 2) * rr * 0.6)
        if rr > 5:
            _glow_window(cv, x, y, 3 if rr > 10 else 2, 6 if rr > 10 else 4, '#7ff8ff')
    return rgba(cv), base


@lru_cache(maxsize=None)
def _branch_coral(seed, size, ramp_name):
    ramp = {'mag': MAG_RAMP, 'teal': TEAL_RAMP, 'orange': ['#4a1a1a', '#8a2e1e', '#c8502a', '#f08a40', '#ffc070',
                                                           '#fff0b0']}[ramp_name]
    w, h = size * 2 + 6, size + 6
    cv = new(w, h)

    def br(x, y, ang, L, depth, i):
        ex = x + math.cos(ang) * L
        ey = y + math.sin(ang) * L
        cc = ramp[min(len(ramp) - 1, 2 + (4 - depth))]
        cv.line(int(x), int(y), int(ex), int(ey), cc)
        cv.line(int(x) + 1, int(y), int(ex) + 1, int(ey), ramp[1])
        if depth > 0:
            for s_ in (-1, 1):
                br(ex, ey, ang + s_ * (0.35 + 0.3 * hash01(seed, i, s_)), L * (0.68 + 0.1 * hash01(seed, i)),
                   depth - 1, i * 2 + (s_ > 0))
        else:
            cv.px(int(ex), int(ey), ramp[-1])
            cv.px(int(ex), int(ey) - 1, ramp[-2])

    br(w / 2, h - 1, -math.pi / 2, size * 0.36, 4, 1)
    return rgba(cv)


@lru_cache(maxsize=1)
def _near_d():
    w, h = W + 90, H
    cv = new(w, h)
    a = cv.a
    yy, xx = np.mgrid[0:h, 0:w]
    sand = pmask(w, h, [(0, h), (0, 168)] + [(x, 166 + int(3 * math.sin(x * 0.05))) for x in range(0, w + 1, 10)] +
                 [(w, h)])
    rip = np.sin(yy * 0.85 + np.sin(xx * 0.07) * 2.5 + np.sin(xx * 0.19) * 0.8)
    a[sand] = col('#54468a')
    a[sand & (rip > 0.82)] = col('#7a6aa8')
    a[sand & (rip < -0.86)] = col('#3a2f6c')
    # coral rocks at the corners
    for i, (x, rx, ry) in enumerate(((10, 34, 26), (372, 36, 30), (180, 14, 7))):
        rm = emask(w, h, x, 172, rx, ry) & (yy <= 174)
        shade_blob(a, rm, ['#1e1450', '#2e1e66', '#45307e', '#5e4896', '#8068b0'], light=(0.2, -1.0), rad=6,
                   noise=(fbm(h, w, 4, 700 + i) - 0.5) * 0.5, outline='#140e3a')
    for (x, y, s_, seed, rn) in ((4, 152, 22, 1, 'mag'), (30, 156, 16, 2, 'teal'), (54, 166, 12, 3, 'orange'),
                                 (356, 150, 24, 4, 'teal'), (384, 148, 20, 5, 'mag'), (334, 164, 12, 6, 'orange'),
                                 (176, 168, 10, 7, 'mag')):
        lay = _branch_coral(seed, s_, rn)
        cv.blit(lay, x - lay.shape[1] // 2, y - lay.shape[0] + 3)
    # tube coral clusters on the sand
    for i, x in enumerate((96, 130, 236, 268, 300)):
        y = 172 + (i % 2) * 3
        cc = ('#4fd1c5', '#e46fb8', '#f08a40', '#b860c8', '#ffd070')[i]
        for b in range(5):
            hh = 3 + int(hash01(i, b, 3) * 6)
            xx_ = x - 4 + b * 2
            cv.vline(xx_, y - hh, y, shade(cc, 0.7))
            cv.px(xx_, y - hh, mix(cc, 'white', 0.55))
            cv.px(xx_ + 1, y - hh + 1, shade(cc, 0.45))
    return rgba(cv), sand


LANTERNS_D_MID = [(150, 149, '#e46fb8'), (176, 150, '#4fd1c5'), (238, 150, '#4fd1c5'), (262, 150, '#e46fb8'),
                  (70, 154, '#4fd1c5'), (330, 152, '#e46fb8')]
LANTERNS_D_NEAR = [(60, 160, '#e46fb8', 2), (312, 158, '#4fd1c5', 2), (150, 168, '#ffd070', 1)]
PEARL_STRINGS = [((112, 54), (CONCH_X - 10, 44), 14), ((CONCH_X + 10, 46), (300, 62), 14),
                 ((36, 62), (112, 58), 8), ((300, 62), (346, 52), 8)]
VENTS_D = [(122, 150), (196, 150), (286, 150), (60, 156)]


def beat_pulse(f):
    return (1 - (f % FRAMES_PER_BEAT) / FRAMES_PER_BEAT) ** 2


def anemone(c, x, y, f, color, size=1, seed=0):
    p = beat_pulse(f + seed * 4)
    hgt = 5 + size * 3
    c.vline(x, y - hgt, y, '#3a2a6a')
    c.vline(x + 1, y - hgt + 1, y, '#241c5e')
    bx, by = x, y - hgt
    r = 2 + size
    fx.glow(c, bx, by - 1, 10 + size * 5 + int(p * 5), color, 0.3 + 0.3 * p, rings=3)
    c.ellipse(bx, by, r, max(1, r - 1), shade(color, 0.7))
    c.ellipse(bx, by - 1, r - 1, max(1, r - 2), color)
    for k in range(-r - 1, r + 2):
        sway = int(round(math.sin(f * 0.15 + k + seed) * 1.0))
        L = 2 + size + int(p * 2)
        c.line(bx + k, by - 1, bx + k + sway + k // 2, by - 1 - L, mix(color, 'white', 0.35 + 0.3 * p))
    c.px(bx, by - 1, 'white')


def pearl_string(c, f, p0, p1, sag, ox, idx):
    (xa, ya), (xb, yb) = p0, p1
    n = int(abs(xb - xa) / 5)
    for k in range(n + 1):
        t = k / max(1, n)
        x = int(round(lerp(xa, xb, t))) + ox
        y = int(round(lerp(ya, yb, t) + sag * 4 * t * (1 - t)))
        if k < n:
            x2 = int(round(lerp(xa, xb, t + 0.5 / n))) + ox
            y2 = int(round(lerp(ya, yb, t + 0.5 / n) + sag * 4 * (t + 0.5 / n) * (1 - t - 0.5 / n)))
            c.px(x2, y2, '#4a3e8a')
        chase = ((f // 3) - k + idx * 5) % 12
        cc = 'white' if chase == 0 else ('#ffd8f4' if chase < 3 else '#c8baee')
        c.px(x, y, cc)
        if chase == 0:
            c.px(x, y + 1, '#ffd8f4')


def render_d(f):
    cx = cam_d(f)
    c = _water_d().copy()
    solid = np.zeros((H, W), bool)
    # surface far above + god rays + balloon wobbling on it
    _surface_line(c, 1, f)
    for i in range(4):
        x = 30 + i * 84 - int(cx * 0.3) + int(math.sin(f * 0.03 + i * 1.3) * 8)
        fx.light_beam(c, ((x, 3), (x + 6, 3)), ((x + 30, 150), (x + 64, 150)), '#a8f0ff', 0.06)
    draw_balloon_reflection(c, f, 244 - int(cx * 0.2), 0)
    # far city (p 0.3)
    fo = -int(round(cx * 0.3)) - 8
    blit_track(c, _far_d(), fo, -6, solid)
    for i in range(8):                               # twinkling far windows
        if (f // 4 + i * 3) % 10 < 4:
            x = fo + int(hash01(i, 91) * (W + 30))
            c.px(x, 110 + int(hash01(i, 92) * 30), '#4fd1c5' if i % 2 else '#e46fb8')
    # palace (p 0.65)
    mo = -int(round(cx * 0.65)) - 10
    lay, _ = _mid_d()
    blit_track(c, lay, mo, 0, solid)
    p = beat_pulse(f)
    # pearl finial on the conch: pulsing glow
    fx.glow(c, CONCH_X + mo, CONCH_TOP - 3, 14 + int(p * 6), '#ffd8f4', 0.3 + 0.3 * p, rings=4)
    c.circle(CONCH_X + mo, CONCH_TOP - 3, 2, '#fff0ff')
    c.px(CONCH_X + mo - 1, CONCH_TOP - 4, 'white')
    for (x, top) in ((36, 60), (140, 74), (276, 70), (346, 50), (170, 96), (244, 92)):
        fx.glow(c, x + mo, top, 5, '#7ff8ff' if x % 2 == 0 else '#ff9ae0', 0.3, rings=2)
        c.px(x + mo, top - 1, 'white')
    # window glow breathing
    fx.glow(c, DOME_D[0] + mo, DOME_D[1] - 10, 30, '#e46fb8', 0.10 + 0.1 * p, rings=3)
    for i, (a_, b_, sag) in enumerate(PEARL_STRINGS):
        pearl_string(c, f, a_, b_, sag, mo, i)
    for i, (x, y, cc) in enumerate(LANTERNS_D_MID):
        anemone(c, x + mo, y, f, cc, size=0, seed=i)
    for i, (x, y) in enumerate(VENTS_D):
        bubble_stream(c, f, x + mo, y, 4, n=7, seed=20 + i, speed=0.9)
    # caustics over the city
    caustic_light(c, f, solid, 0, color='#b8f8ff', k=0.22, ox=cx * 0.5, oy=0, fade_h=220)
    # near reef (p 1.0)
    no = -cx - 20
    nl, _ = _near_d()
    c.blit(nl, no, 0)
    for i, (x, y, cc, sz) in enumerate(LANTERNS_D_NEAR):
        anemone(c, x + no, y, f, cc, size=sz, seed=10 + i)
    for i, (x, yb, hg) in enumerate(((-6, 182, 150), (8, 182, 120), (402, 182, 160), (390, 182, 110))):
        kelp(c, x + no, yb, hg, f, 30 + i, cols=('#24183e', '#3a2a5e', '#5a4a8e', '#8a7ac8'), amp=5.0, width=2,
             leaf_every=4)
    # sea fans swaying in the foreground
    for i, (x, y, L) in enumerate(((84, 172, 16), (290, 170, 18))):
        sw = math.sin(f * 0.08 + i) * 2
        for b in range(9):
            ang = -math.pi / 2 + (b - 4) * 0.2
            ex = x + no + int(math.cos(ang) * L + sw * (b / 8))
            ey = y + int(math.sin(ang) * L)
            c.line(x + no, y, ex, ey, '#c83a8a')
            c.px(ex, ey, '#ff9ad8')
        for r in (6, 11):
            c.line(x + no - r // 2, y - r, x + no + r // 2 + int(sw), y - r, '#a02a7a')
    bubble_stream(c, f, 140 + no, 168, 0, n=8, seed=40, speed=1.2, big=True)
    clam_x = 236 + no
    draw_clam(c, clam_x, 175, f, 3)
    # drifting glow motes
    for i in range(18):
        t = (f * 0.4 + hash01(i, 1) * 180) % 180
        x = int(hash01(i, 2) * W + math.sin(t * 0.05 + i) * 6)
        y = int(180 - t)
        if (f // 5 + i) % 4:
            c.px(x, y, ('#7ff8ff', '#ff9ae0', '#fff0ff')[i % 3])
    callout(c, CONCH_X - 14 + mo, 70, 'SEAQUESTRIA', f, CUT3 + 18, CUT3 + 90, side='left', length=14)
    return c


def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT1:
        c = render_a(f)
        iris(c, 1 - seg(f, 72, 79), 204, 94)
    elif f < DIVE:
        c = render_b(f)
    elif f < CUT3:
        c = render_c(f)
    else:
        c = render_d(f)
    atlas_card(c, f, 100, 208, 'SEA KINGDOMS', ['PALACE ON A SEA CRAG', 'CORAL CITY BELOW'])
    return c


SFX = [
    (84, 'waves', 0.55, {'align': 'hit'}),
    (100, 'thunder', 0.4, {'variant': 0, 'pan': -0.6}),
    (108, 'wind', 0.35, {'variant': 0}),
    (138, 'thunder', 0.3, {'variant': 0, 'pan': -0.4}),
    (150, 'waves', 0.45, {'align': 'hit', 'pan': 0.5}),
    (198, 'wind', 0.35, {'variant': 0}),
    (216, 'shimmer', 0.4),
    (234, 'fountain', 0.3),
    (280, 'whoosh', 0.6, {'align': 'hit'}),
    (DIVE, 'splash', 1.0),
    (DIVE + 6, 'bubbles', 0.7),
    (CUT3, 'crystal', 0.6),
    (342, 'bubbles', 0.35),
    (360, 'crystal', 0.35),
    (396, 'bubbles', 0.3),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(CUT1, 'dissolve', {}), (DIVE, 'dissolve', {'dur': 12}), (CUT3, 'dissolve', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
