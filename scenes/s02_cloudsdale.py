"""CLOUDSDALE: the city in the sky (5 bars, 360 frames, HUD clock 06:30 -> 07:15).

Beat sheet (local frames; bar = 72, beat = 18):
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71.
  Bar 2  72-143   REVEAL (72): iris opens 72-79 on the main dome, which peeks out of a glowing sea of
                  clouds. The camera RISES (tilt over a tall layered world, 72-132): five cloud-bank
                  layers (gold rims, lilac shadows, each drifting = slow rolling) sweep down past the lens,
                  the cloud fog thins and Cloudsdale floats free in sunrise gold: cloud temples, the domed
                  rotunda, houses with round windows, the Colosseum, liquid-rainbow falls pouring from
                  the island's edges into the cloud sea. Sun low on the right (06:30 sunrise).
                  Callout CLOUDSDALE (dome) 90-170. Atlas card 100-200.
  Bar 3 144-215   Eased pan right across the city (120-214, parallax: back city 0.7, banks 1.1-1.45):
                  colonnades, cloud houses, the Colosseum with rainbow pennants fluttering; the balloon
                  drifts along above the roofs. Callout COLOSSEUM 144-210.
  Bar 4 216-287   CUT on the bar line: WEATHER FACTORY cross-section (dollhouse, 2 floors), slow drift.
                  Snowflake line (every flake a different crystal), lightning jars crackling + charger
                  arcs on the beat, rainbow vats bubbling in stripes, rain-cloud press (pistons strike on
                  every beat, pressed clouds stacked & dripping), steam vents, gauges, blinking lamps.
                  The balloon drifts past in the sky above the roof. Callout WEATHER FACTORY 222-282.
  Bar 5 288-359   `rainbow` HIT (288): cut to the rainbow falls close-up: prismatic spray burst and a
                  bow forming in the mist; the balloon (large) sails through the rainbow mist; Galloping
                  Gorge glimpsed far below through a cloud gap. Callout LIQUID RAINBOW 294-350.
                  Toast from 302 (top-centre kept clear). Last 8 frames: wipe.
Ambient: rolling cloud banks, falls flow + spray, pennants, window glints, sun shimmer, wisps,
         factory machines, steam, sparkles, the balloon.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'cloudsdale'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)

# ------------------------------------------------------------------ colours
RB_NAMES = ('red', 'orange', 'gold', 'leaf', 'sky', 'violet')
RB = np.array([col(n) for n in RB_NAMES], np.uint8)
RB_HI = np.array([mix(n, 'white', 0.5) for n in RB_NAMES], np.uint8)
RB_SH = np.array([shade(n, 0.78) for n in RB_NAMES], np.uint8)
RB_PAS = [mix(n, 'white', 0.55) for n in RB_NAMES]

CL_DEEP = mix('cloud_dk', 'lavender', 0.5)
CL_SH = mix('cloud_dk', 'lavender', 0.2)
CL_MID = col('cloud')
CL_LT = mix('white', 'cream', 0.6)
NEAR = (CL_DEEP, CL_SH, CL_MID, CL_LT)

A_OUT = mix('periwinkle', 'royal_lt', 0.55)
A_DEEP = mix('cloud_dk', 'periwinkle', 0.6)
A_SH = col('cloud_dk')
A_MID = col('cloud')
A_LT = col('white')
A_WARM = col('cream')
A_WIN = mix('periwinkle', 'royal', 0.35)
A_WIN2 = mix('periwinkle', 'royal', 0.65)
ARCH = (A_DEEP, A_SH, A_MID, A_LT)
G, G_HI, G_SH = col('gold'), col('butter'), col('amber')


def ramp_haze(ramp, haze, a):
    return tuple(mix(x, haze, a) for x in ramp)


# ------------------------------------------------------------------ helpers
def _new(w, h):
    return Canvas(w, h, color=KEY)


def _rgba(cv):
    a = cv.a
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = np.any(a != _KEYA, axis=2) * 255
    return out


_LV = np.array([0.62, -0.55, 0.56])
_LV = _LV / np.linalg.norm(_LV)
_LB = np.array([0.45, -0.8, 0.42])
_LB = _LB / np.linalg.norm(_LB)


def puff(cv, cx, cy, r, ramp, rim=None, thr=(-0.05, 0.33, 0.74), lv=_LV):
    """Shaded cloud ball (light from upper right); ramp = (deep, shadow, mid, light)."""
    x0, y0 = int(math.floor(cx - r - 1)), int(math.floor(cy - r - 1))
    x1, y1 = int(math.ceil(cx + r + 2)), int(math.ceil(cy + r + 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(cv.w, x1), min(cv.h, y1)
    if xb <= xa or yb <= ya:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb]
    dx = (xx + 0.5 - cx) / r
    dy = (yy + 0.5 - cy) / r
    d2 = dx * dx + dy * dy
    m = d2 <= 1.0
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    I = dx * lv[0] + dy * lv[1] + nz * lv[2]
    idx = np.digitize(I, thr)
    pal = np.array([col(x) for x in ramp], np.uint8)
    reg = cv.a[ya:yb, xa:xb]
    reg[m] = pal[idx[m]]
    if rim is not None and r >= 3:
        e = m & (d2 > (1 - 1.6 / r) ** 2) & (I > 0.42)
        reg[e] = col(rim)


def cloud_mass(cv, circles, ramp, rim=None, lv=_LV, thr=(-0.05, 0.33, 0.74), wrap=0, rim2=None):
    """Union of spheres (z-buffered) shaded as one cloud mass; clean lit silhouette with a sun rim."""
    if wrap:
        circles = [(x + o, y, r) for (x, y, r) in circles for o in (-wrap, 0, wrap)]
    circles = [(x, y, r) for (x, y, r) in circles if x + r >= -1 and x - r <= cv.w + 1]
    if not circles:
        return
    xa = max(0, int(math.floor(min(x - r for x, y, r in circles))) - 1)
    xb = min(cv.w, int(math.ceil(max(x + r for x, y, r in circles))) + 2)
    ya = max(0, int(math.floor(min(y - r for x, y, r in circles))) - 1)
    yb = min(cv.h, int(math.ceil(max(y + r for x, y, r in circles))) + 2)
    if xb <= xa or yb <= ya:
        return
    hh, ww = yb - ya, xb - xa
    zb = np.full((hh, ww), -1e9)
    nx = np.zeros((hh, ww))
    ny = np.zeros((hh, ww))
    nzz = np.zeros((hh, ww))
    for (cx, cy, r) in circles:
        x0, x1 = max(xa, int(cx - r - 1)), min(xb, int(cx + r + 2))
        y0, y1 = max(ya, int(cy - r - 1)), min(yb, int(cy + r + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = (xx + 0.5 - cx) / r
        dy = (yy + 0.5 - cy) / r
        d2 = dx * dx + dy * dy
        inside = d2 <= 1
        z = np.sqrt(np.clip(1 - d2, 0, 1)) * r
        sub = (slice(y0 - ya, y1 - ya), slice(x0 - xa, x1 - xa))
        upd = inside & (z > zb[sub])
        zb[sub][upd] = z[upd]
        nx[sub][upd] = dx[upd]
        ny[sub][upd] = dy[upd]
        nzz[sub][upd] = np.sqrt(np.clip(1 - d2[upd], 0, 1))
    m = zb > -1e8
    I = nx * lv[0] + ny * lv[1] + nzz * lv[2]
    idx = np.digitize(I, thr)
    pal = np.array([col(x) for x in ramp], np.uint8)
    reg = cv.a[ya:yb, xa:xb]
    reg[m] = pal[idx[m]]
    if rim is not None:
        up = np.zeros_like(m)
        up[1:] = m[:-1]
        rt = np.zeros_like(m)
        rt[:, :-1] = m[:, 1:]
        if wrap and xa == 0 and xb == cv.w:     # tileable strip: the right neighbour of the last column
            rt[:, -1] = m[:, 0]                 # is the first one (else a rim seam at the tile edge)
        edge = m & (~up | ~rt) & (I > 0.38)
        reg[edge] = col(rim)
        if rim2 is not None:
            e2 = m & ~up & (I > 0.7)
            reg[e2] = col(rim2)


def bank_strip(w, h, top, seed, ramp, rim, rmin, rmax, amp, rows=2, step=(0.9, 1.4), grow=0.25,
               rowstep=0.65, rim2=None, lv=_LB, rows_rim=False):
    """Tileable strip of cumulus banks: rows of cloud masses receding to the back."""
    cv = _new(w, h)
    R = rng(seed)
    yrow = top
    last = None
    for row in range(rows):
        k = 1 + grow * row
        circ = []
        x = R.uniform(0, rmax)
        while x < w:
            r = R.uniform(rmin, rmax) * k
            circ.append((x, yrow + r + R.uniform(0, amp * k), r))
            x += r * R.uniform(*step)
        lowest = max(cy for _, cy, _ in circ)
        cv.rect(0, int(lowest), w, h - int(lowest), ramp[0])
        cloud_mass(cv, circ, ramp, rim if row == 0 or rows_rim else None, wrap=w,
                   rim2=rim2 if row == 0 else None, lv=lv)
        yrow += rmax * rowstep * k
    return _rgba(cv)


def blit_wrap(c, arr, x, y):
    w = arr.shape[1]
    x = int(x) % w
    c.blit(arr, x - w, int(y))
    c.blit(arr, x, int(y))
    if x + w < W:
        c.blit(arr, x + w, int(y))


def veil(c, top, bot, color, y0=0, y1=H):
    """Dithered fog over rows y0..y1 with coverage going from top to bot."""
    y0, y1 = max(0, y0), min(H, y1)
    if y1 <= y0 or max(top, bot) <= 0:
        return
    t = np.linspace(top, bot, y1 - y0)[:, None]
    m = bayer(y1 - y0, W, 0, y0) < t
    c.a[y0:y1][m] = col(color)


def rainbow_fall(c, x, y0, y1, sw, f, speed=2.5, seed=0, lip=True, sheets=True):
    """Vertical liquid-rainbow curtain: 6 stripes of width sw, flowing down from y0 to y1."""
    w = 6 * sw
    xa, xb = max(0, x), min(W, x + w)
    ya, yb = max(0, y0), min(H, y1)
    if xb <= xa or yb <= ya:
        return
    R = rng(seed + 17)
    g = 1
    ng = w + 2
    ph = R.uniform(0, 60, ng)
    per = R.integers(18, 32, ng)
    ln = R.integers(3, 9, ng)
    xs = np.arange(xa - x, xb - x)
    gi = xs // g
    ys = np.arange(ya, yb)[:, None]
    stripe = np.minimum(xs // sw, 5)
    t = (ys - y0 - f * speed + ph[gi][None, :]) % per[gi][None, :]
    out = np.broadcast_to(RB[stripe][None, :, :], (yb - ya, xb - xa, 3)).copy()
    hi = t < ln[gi][None, :]
    sh = (t >= per[gi][None, :] * 0.6) & (t < per[gi][None, :] * 0.6 + 2)
    if sheets:      # chevron sheets sliding down the whole curtain
        sy = (ys - y0 - f * speed * 1.2 + np.abs(stripe - 2.5)[None, :] * 3) % 37
        hi |= sy < 1
    out[hi] = np.broadcast_to(RB_HI[stripe][None, :, :], out.shape)[hi]
    out[sh] = np.broadcast_to(RB_SH[stripe][None, :, :], out.shape)[sh]
    if sw >= 3:     # each stripe slightly rounded: dark left column
        edge = (xs % sw) == 0
        out[:, edge] = (out[:, edge].astype(np.int16) * 0.85).astype(np.uint8)
    if xa == x:
        out[:, 0] = RB_SH[0]
    c.a[ya:yb, xa:xb] = out
    if lip and 0 <= y0 < H:
        for i in range(6):
            c.hline(x + i * sw, x + i * sw + sw - 1, y0, tuple(RB_HI[i]))


def spray(c, cx, cy, w, f, seed=0, n=10, big=1.0):
    """Churning prismatic mist where a rainbow fall plunges into cloud."""
    if big < 1.5:
        for i in range(4):
            ox = int((i - 1.5) * w * 0.3)
            r = max(2, int((3 + 2 * hash01(seed, i)) * big + math.sin(f / 7 + i * 1.7) * big))
            c.ellipse(cx + ox, cy - r // 2, r + 1, r, 'white' if i % 2 == 0 else mix('white', 'cotton', 0.5))
    for i in range(n):
        P = 20 + int(hash01(seed, i, 3) * 10)
        t = ((f + hash01(seed, i) * P) % P) / P
        dirx = (hash01(seed, i, 1) - 0.5) * 2
        px = cx + int(dirx * w * 0.7 * t)
        py = cy - int(10 * big * t * (1 - t) * 4 * (0.5 + hash01(seed, i, 2)))
        c.px(px, py, RB_PAS[i % 6] if i % 3 else 'white')


# ------------------------------------------------------------------ city architecture
def column(cv, x, ytop, ybot):
    cv.rect(x, ytop + 2, 1, ybot - ytop - 3, A_SH)
    cv.rect(x + 1, ytop + 2, 1, ybot - ytop - 3, A_MID)
    cv.rect(x + 2, ytop + 2, 1, ybot - ytop - 3, A_LT)
    cv.rect(x - 1, ytop, 5, 2, A_MID)
    cv.hline(x - 1, x + 3, ytop, A_LT)
    cv.px(x - 1, ytop + 1, A_SH)
    cv.rect(x - 1, ybot - 1, 5, 1, A_MID)
    cv.px(x + 3, ybot - 1, A_LT)


def pediment(cv, x0, x1, y, ph, emblem=True):
    cx = (x0 + x1) // 2
    cv.poly([(x0, y + 1), (x1 + 1, y + 1), (cx + 0.5, y - ph)], A_MID)
    cv.poly([(x0 + 4, y), (x1 - 3, y), (cx + 0.5, y - ph + 3)], A_SH)
    cv.line(x0, y, cx, y - ph, A_LT)
    cv.line(cx, y - ph, x1, y, A_WARM)
    cv.line(cx + 1, y - ph + 1, x1 - 1, y, G_HI)
    if emblem and ph >= 6:
        ey = y - ph // 2 + 1
        cv.circle(cx, ey, 2, G)
        cv.px(cx, ey, G_HI)
        cv.px(cx + 1, ey - 1, G_HI)
    cv.px(cx, y - ph - 1, G)
    cv.px(x0, y - 1, G)
    cv.px(x1, y - 1, G)


def temple(cv, x, by, ncol, colh=18, step=6, emblem=True):
    w = ncol * step + 1
    # stylobate
    for k in range(3):
        yy = by - 6 + 2 * k
        cv.rect(x - 2 - 2 * k, yy, w + 4 + 4 * k, 2, A_MID)
        cv.hline(x - 2 - 2 * k, x + w + 1 + 2 * k, yy, A_LT)
        cv.px(x - 2 - 2 * k, yy + 1, A_SH)
    ytop = by - 6 - colh
    cv.rect(x, ytop, w, colh, A_DEEP)
    cv.dither(x, ytop, w, colh, A_SH, 0.25)
    dx = x + w // 2 - 3
    cv.rect(dx, by - 17, 7, 11, A_WIN2)
    cv.circle(dx + 3, by - 17, 3, A_WIN2)
    cv.px(dx + 5, by - 11, G)
    for i in range(ncol):
        column(cv, x + 1 + i * step, ytop, by - 6)
    # entablature
    y = ytop - 6
    cv.rect(x - 2, y, w + 4, 6, A_MID)
    cv.hline(x - 3, x + w + 2, y, A_LT)
    cv.hline(x - 2, x + w + 1, y + 1, A_SH)
    for k in range(x - 1, x + w + 1, 3):
        cv.rect(k, y + 2, 1, 2, A_SH)
    cv.hline(x - 2, x + w + 1, y + 4, A_LT)
    cv.hline(x - 2, x + w + 1, y + 5, A_SH)
    pediment(cv, x - 3, x + w + 2, y, max(5, w // 5), emblem)


def dome_shape(cv, cx, cy, r, ramp=ARCH, rim=G_HI, ribs=True):
    """Upper hemisphere centred (cx, cy) radius r with ribs."""
    yy, xx = np.mgrid[cy - r:cy + 1, cx - r - 1:cx + r + 2]
    dx = (xx + 0.5 - (cx + 0.5)) / r
    dy = (yy + 0.5 - (cy + 0.5)) / r
    d2 = dx * dx + dy * dy
    m = (d2 <= 1.0)
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    I = dx * _LV[0] + dy * _LV[1] + nz * _LV[2]
    idx = np.digitize(I, (-0.05, 0.33, 0.74))
    pal = np.array([col(x) for x in ramp], np.uint8)
    out = pal[idx]
    if ribs:
        hw = np.sqrt(np.clip(1 - dy * dy, 1e-6, 1))
        u = np.clip(dx / hw, -1, 1)
        ang = np.arcsin(u)
        k = ang / (math.pi / 9)
        rib = np.abs(k - np.round(k)) < 0.5 / (r * hw * 0.35 + 1)
        out[rib & m] = (out[rib & m].astype(int) * 0.88).astype(np.uint8)
    e = m & (d2 > (1 - 1.5 / r) ** 2) & (I > 0.45)
    out[e] = col(rim)
    ya, xa = cy - r, cx - r - 1
    reg = cv.a[ya:cy + 1, xa:cx + r + 2]
    reg[m] = out[m]


def drum(cv, cx, ytop, ybot, hw, ncol=9, win=True):
    xs = np.arange(cx - hw, cx + hw + 1)
    for x in xs:
        u = (x + 0.5 - (cx + 0.5)) / (hw + 0.5)
        I = u * 0.75 + math.sqrt(max(0.0, 1 - u * u)) * 0.5
        cc = A_DEEP if I < 0.0 else A_SH if I < 0.4 else A_MID if I < 0.8 else A_LT
        cv.vline(int(x), ytop, ybot, cc)
    th = np.linspace(-1.3, 1.3, ncol)
    pos = [int(round(cx + hw * 0.95 * math.sin(t))) for t in th]
    if win:
        for a, b in zip(pos, pos[1:]):
            m = (a + b) // 2
            ww = max(1, (b - a) // 2 - 1)
            wy = ytop + 4
            cv.rect(m - ww // 2, wy + 2, ww, ybot - ytop - 9, A_WIN)
            cv.circle(m, wy + 2, ww // 2, A_WIN) if ww >= 3 else None
            if ww >= 2:
                cv.px(m - ww // 2, wy + 3, A_OUT)
    for p in pos:
        cv.vline(p, ytop + 2, ybot - 1, A_LT if p > cx - hw // 3 else A_MID)
    cv.rect(cx - hw - 1, ytop, 2 * hw + 3, 2, A_MID)
    cv.hline(cx - hw - 1, cx + hw + 1, ytop, A_LT)
    cv.hline(cx - hw, cx + hw, ytop + 2, G)
    cv.rect(cx - hw - 1, ybot, 2 * hw + 3, 1, A_SH)


def rotunda(cv, cx, by):
    # podium steps
    for k in range(4):
        hw = 46 - 3 * k
        yy = by - 2 - 2 * k
        cv.rect(cx - hw, yy, 2 * hw + 1, 2, A_MID)
        cv.hline(cx - hw, cx + hw, yy, A_LT)
        cv.px(cx - hw, yy + 1, A_SH)
    base = by - 8
    # wings: colonnades with a flat roof + balustrade
    for side in (-1, 1):
        x0 = cx + (28 if side > 0 else -44)
        cv.rect(x0, base - 18, 17, 18, A_DEEP)
        for i in range(3):
            cv.rect(x0 + 2 + i * 5, base - 13, 3, 9, A_WIN)
            cv.circle(x0 + 3 + i * 5, base - 13, 1, A_WIN)
        for i in range(4):
            column(cv, x0 + i * 5, base - 18, base)
        cv.rect(x0 - 2, base - 22, 21, 4, A_MID)
        cv.hline(x0 - 2, x0 + 18, base - 22, A_LT)
        cv.hline(x0 - 2, x0 + 18, base - 19, A_SH)
        for i in range(0, 21, 2):
            cv.px(x0 - 2 + i, base - 23, A_MID)
        cv.hline(x0 - 2, x0 + 18, base - 24, A_LT)
    # portico in front of the drum
    drum(cv, cx, base - 34, base - 1, 24, 9)
    # portico
    px0 = cx - 12
    cv.rect(px0, base - 22, 25, 22, A_DEEP)
    cv.rect(cx - 3, base - 12, 7, 12, A_WIN2)
    cv.circle(cx, base - 12, 3, A_WIN2)
    cv.px(cx + 2, base - 6, G)
    for i in range(5):
        column(cv, px0 + 1 + i * 5 + (1 if i == 4 else 0), base - 22, base)
    cv.rect(px0 - 2, base - 27, 29, 5, A_MID)
    cv.hline(px0 - 3, px0 + 27, base - 27, A_LT)
    cv.hline(px0 - 2, px0 + 26, base - 23, A_SH)
    pediment(cv, px0 - 3, px0 + 27, base - 27, 7)
    # dome
    dome_shape(cv, cx, base - 35, 24)
    cv.hline(cx - 24, cx + 24, base - 35, G)
    # lantern
    ly = base - 59
    cv.rect(cx - 5, ly - 7, 11, 8, A_MID)
    for i in range(4):
        cv.vline(cx - 4 + i * 3, ly - 6, ly, A_LT if i > 1 else A_SH)
    cv.rect(cx - 3, ly - 5, 1, 4, A_WIN)
    cv.rect(cx, ly - 5, 1, 4, A_WIN)
    cv.rect(cx + 3, ly - 5, 1, 4, A_WIN)
    cv.hline(cx - 6, cx + 6, ly - 7, A_LT)
    dome_shape(cv, cx, ly - 8, 5, ribs=False, rim=G_HI)
    cv.vline(cx, ly - 17, ly - 13, G)
    cv.circle(cx, ly - 19, 2, G)
    cv.px(cx, ly - 19, G_HI)
    cv.px(cx + 1, ly - 20, G_HI)
    for dx_, dy_ in ((0, -4), (-3, -1), (3, -1), (0, 2)):
        pass


def house(cv, x, by, w, h, seed=0, roof=True):
    R = rng(seed)
    cv.round_rect(x, by - h, w, h, A_MID, r=2)
    cv.rect(x, by - h + 2, 2, h - 2, A_SH)
    cv.rect(x + w - 1, by - h + 2, 1, h - 3, A_LT)
    cv.hline(x + 1, x + w - 2, by - 1, A_SH)
    # round windows
    rows = max(1, (h - 10) // 9)
    for j in range(rows):
        wy = by - h + 6 + j * 9
        nwin = max(1, (w - 4) // 7)
        for i in range(nwin):
            wx = x + 4 + i * ((w - 6) // max(1, nwin)) + 1
            lit = R.uniform() < 0.3
            cv.circle(wx, wy, 2, A_OUT)
            cv.circle(wx, wy, 1, G_HI if lit else A_WIN)
            if not lit:
                cv.px(wx - 1, wy - 1, A_SH)
            cv.hline(wx - 2, wx + 2, wy + 3, A_LT)
    # round-top door
    dx = x + w // 2 - 2
    cv.rect(dx, by - 7, 5, 6, A_WIN2)
    cv.hline(dx + 1, dx + 3, by - 8, A_WIN2)
    cv.px(dx + 3, by - 4, G)
    if roof:
        n = max(2, w // 7)
        for i in range(n):
            u = (i + 0.5) / n
            r = 3 + int(R.uniform(0, 2.5)) + (2 if 0.3 < u < 0.7 else 0)
            puff(cv, x + u * w, by - h + 1, r, ARCH, rim=G_HI)


def colosseum(cv, x0, by, w=110, h=42):
    """Cloud stadium seen from slightly above: curved near wall with 3 tiers of arches and the seating
    bowl behind it. Returns the pennant pole tops."""
    cx = x0 + w / 2
    a = w / 2
    top = by - h
    CUR = 5

    def cu(x):
        u = (x + 0.5 - cx) / a
        return math.sqrt(max(0.0, 1 - u * u)), u

    seat_cols = [mix(n, 'cloud', 0.62) for n in RB_NAMES]
    # seating bowl + far rim
    for x in range(x0 + 2, x0 + w - 2):
        c_, u = cu(x)
        yf = top - 2 - int(round(9 * c_))
        yn = top + int(round(CUR * c_))
        sec = int((u + 1) * 6) % 6
        for y in range(yf + 2, yn):
            k = (y - yf) // 2
            cc = seat_cols[sec] if k % 2 == 0 else A_MID
            if (y - yf) % 2 == 1:
                cc = A_SH
            cv.px(x, y, cc)
        cv.px(x, yf, A_LT if u > -0.2 else A_MID)
        cv.px(x, yf + 1, A_MID)
    # near wall, cylindrical shading
    for x in range(x0, x0 + w):
        c_, u = cu(x)
        I = u * 0.7 + c_ * 0.55
        cc = A_DEEP if I < -0.05 else A_SH if I < 0.35 else A_MID if I < 0.75 else A_LT
        cv.vline(x, top + int(round(CUR * c_)), by - 1, cc)
    tiers = [(by - 15, by - 1, 9), (by - 29, by - 16, 8), (by - h + 2, by - 30, 5)]
    for t, (ya, yb, ah) in enumerate(tiers):
        n = 13 if t < 2 else 17
        for ang in np.linspace(-1.42, 1.42, n):
            xc = cx + a * 0.97 * math.sin(ang)
            c_, u = cu(int(xc))
            sh = int(round(CUR * c_ * (by - ya) / h))
            aw = max(1, int(round(4.2 * math.cos(ang))))
            xl = int(round(xc - aw / 2))
            if t < 2:
                cv.rect(xl, yb - ah + 1 + sh, aw, ah - sh, A_WIN)
                if aw >= 3:
                    cv.hline(xl + 1, xl + aw - 2, yb - ah + sh, A_WIN)
                    cv.px(xl, yb - ah + 1 + sh, A_OUT)
                if aw >= 2:
                    cv.vline(xl, yb - ah + 2 + sh, yb, mix(A_WIN, 'gold', 0.25))
            else:
                cv.rect(xl, ya + 1 + sh, max(1, aw - 1), ah - 2, A_WIN)
        for x in range(x0, x0 + w):
            c_, u = cu(x)
            sh = int(round(CUR * c_ * (by - ya) / h))
            cv.px(x, ya + sh, A_LT)
            cv.px(x, ya + sh + 1, A_SH)
    poles = []
    for x in range(x0, x0 + w):
        c_, u = cu(x)
        yt = top + int(round(CUR * c_))
        cv.px(x, yt, G)
        cv.px(x, yt + 1, A_LT)
    for ang in np.linspace(-1.25, 1.25, 7):
        px_ = int(round(cx + a * 0.95 * math.sin(ang)))
        c_, u = cu(px_)
        yt = top + int(round(CUR * c_))
        cv.vline(px_, yt - 9, yt, A_OUT)
        cv.px(px_, yt - 10, G)
        poles.append((px_, yt - 9))
    return poles


def tower(cv, x, by, w, h):
    cx = x + w // 2
    drum(cv, cx, by - h, by - 1, w // 2, ncol=5, win=False)
    for j in range(3):
        wy = by - h + 8 + j * 13
        cv.circle(cx + 1, wy, 2, A_OUT)
        cv.circle(cx + 1, wy, 1, A_WIN if j != 1 else G_HI)
    # conical cloud cap
    for k in range(10):
        hw = max(0, w // 2 + 2 - k * (w // 2 + 2) // 9)
        yy = by - h - 1 - k
        cv.hline(cx - hw, cx + hw, yy, A_MID)
        cv.px(cx - hw, yy, A_SH)
        cv.px(cx + hw, yy, G_HI)
        if hw > 1:
            cv.px(cx + hw - 1, yy, A_LT)
    cv.px(cx, by - h - 12, G)
    cv.hline(cx - 1, cx + 1, by - h - 13, G)
    cv.px(cx, by - h - 14, G_HI)
    cv.hline(cx - w // 2 - 2, cx + w // 2 + 2, by - h, G)


def arch_gate(cv, x, by, w=22, h=30):
    cv.rect(x, by - h, w, h, A_MID)
    cv.rect(x, by - h, 2, h, A_SH)
    cv.rect(x + w - 1, by - h, 1, h, A_LT)
    cv.rect(x + 6, by - h + 12, w - 12, h - 12, A_WIN)
    cv.circle(x + w // 2, by - h + 12, (w - 12) // 2, A_WIN)
    cv.rect(x - 1, by - h - 3, w + 2, 3, A_MID)
    cv.hline(x - 1, x + w, by - h - 3, A_LT)
    cv.hline(x - 1, x + w, by - h - 1, G)
    cv.circle(x + w // 2, by - h - 6, 2, G)
    cv.px(x + w // 2, by - h - 6, G_HI)


CITY_W, CITY_H, CITY_BY = 480, 140, 100
DOME = (165, 54)          # dome centre in city coords
LANTERN = (165, 31)
COLO = (320, 58)          # colosseum near-wall top-centre
FALLS = ((3, 103, True), (244, 116, False), (459, 103, True))   # (x, top y, in front?) width 18


@lru_cache(maxsize=1)
def _city_lips():
    cv = _new(CITY_W, CITY_H)
    for (x, y, front) in FALLS:
        if front:
            cloud_mass(cv, [(x + 3, y - 1, 5), (x + 9, y - 2, 6), (x + 15, y - 1, 5)], ARCH, rim=G, rim2=G_HI)
    return _rgba(cv)


@lru_cache(maxsize=1)
def _city():
    cv = _new(CITY_W, CITY_H)
    by = CITY_BY
    R = rng(7)
    # underside of the floating island (inverted dome of cloud)
    circ = []
    x = 6
    while x < CITY_W - 6:
        u = x / CITY_W
        r = R.uniform(7, 12) * (0.7 + 0.5 * math.sin(math.pi * u))
        cy = by + 2 + 14 * math.sin(math.pi * u) ** 0.7 + R.uniform(0, 4)
        circ.append((x, cy, r))
        x += r * R.uniform(0.8, 1.2)
    cloud_mass(cv, circ, (mix(CL_DEEP, 'lavender', 0.25), CL_DEEP, CL_SH, CL_MID), rim='peach')
    # back terrace row (raised) behind the main row
    for (hx, hw, hh) in ((46, 14, 16), (218, 15, 30), (236, 16, 36), (378, 14, 30), (432, 18, 34)):
        house(cv, hx, by - 6, hw, hh, seed=hx, roof=True)
    # main row
    temple(cv, 40, by, 7, colh=18)
    # victory column with sun
    column(cv, 98, 52, by - 2)
    cv.rect(97, 50, 5, 2, A_MID)
    cv.circle(99, 46, 3, G)
    cv.circle(99, 46, 1, G_HI)
    for dx_, dy_ in ((0, -5), (-5, 0), (5, 0), (-4, -4), (4, -4)):
        cv.px(99 + dx_, 46 + dy_, G)
    house(cv, 104, by, 13, 22, seed=3)
    rotunda(cv, DOME[0], by)
    house(cv, 214, by, 16, 26, seed=5)
    house(cv, 231, by, 20, 20, seed=6)
    house(cv, 252, by, 12, 30, seed=8)
    poles = colosseum(cv, 265, by, 110, 42)
    arch_gate(cv, 380, by, 22, 30)
    tower(cv, 405, by, 16, 58)
    house(cv, 424, by, 16, 22, seed=9)
    house(cv, 441, by, 14, 28, seed=10)
    # ground cloud row in front of the bases
    circ = []
    x = 8
    while x < CITY_W - 8:
        r = R.uniform(3, 6)
        circ.append((x, by + 2 + R.uniform(0, 2), r))
        x += r * R.uniform(1.0, 1.5)
    circ += [(12, by + 1, 9), (24, by + 3, 7), (468, by + 1, 9), (456, by + 3, 7)]
    cloud_mass(cv, circ, ARCH, rim=G, rim2=G_HI)
    return _rgba(cv), tuple(poles)


@lru_cache(maxsize=1)
def _backcity():
    cv = _new(480, 120)
    haze = mix('cotton', 'peach', 0.3)
    rp = ramp_haze(ARCH, haze, 0.45)
    R = rng(11)
    x = 10
    while x < 470:
        kind = R.integers(0, 4)
        h = int(R.uniform(20, 46))
        w = int(R.uniform(10, 22))
        b = 104
        if kind == 0:     # dome
            cv.rect(x, b - h + 8, w, h - 8, rp[2])
            cv.rect(x, b - h + 8, 2, h - 8, rp[1])
            dome_shape(cv, x + w // 2, b - h + 8, w // 2, ramp=rp, rim=mix(G_HI, haze, 0.3), ribs=False)
        elif kind == 1:   # tower with cap
            cv.rect(x, b - h, w // 2 + 2, h, rp[2])
            cv.rect(x, b - h, 1, h, rp[1])
            cv.tri((x - 1, b - h), (x + w // 2 + 2, b - h), (x + w // 4 + 1, b - h - 7), rp[3])
        elif kind == 2:   # colonnade block
            cv.rect(x, b - h // 2, w + 6, h // 2, rp[1])
            for i in range(0, w + 6, 3):
                cv.vline(x + i, b - h // 2 + 2, b, rp[3])
            cv.rect(x - 1, b - h // 2 - 2, w + 8, 2, rp[3])
        else:             # puffy cloud house
            cv.round_rect(x, b - h + 6, w, h - 6, rp[2], r=2)
            puff(cv, x + w / 2, b - h + 6, w / 2.6, rp, rim=mix(G_HI, haze, 0.3))
        for j in range(2, h - 10, 7):
            cv.px(x + w // 2, b - h + 10 + j, rp[0])
        x += w + int(R.uniform(2, 12))
    return _rgba(cv)


@lru_cache(maxsize=1)
def _skyA():
    cv = Canvas(W, 200)
    cv.gradient(0, 0, W, 110, ['blue', 'periwinkle', mix('periwinkle', 'cotton', 0.5), 'cotton',
                                'peach', mix('peach', 'sand', 0.6)], soft=0.7)
    cv.gradient(0, 110, W, 90, [mix('peach', 'sand', 0.6), 'sand'])
    return cv


@lru_cache(maxsize=None)
def _banksA():
    far1 = bank_strip(384, 100, 4, 21, ramp_haze(NEAR, mix('peach', 'cotton', 0.5), 0.5), 'butter', 3, 7, 3,
                      rows=3, grow=0.3, rowstep=0.7)
    far2 = bank_strip(400, 120, 4, 22, ramp_haze(NEAR, 'cotton', 0.3), 'gold', 5, 10, 5, rows=5,
                      grow=0.25, rowstep=0.75, rim2='butter')
    b3 = bank_strip(420, 120, 4, 23, ramp_haze(NEAR, 'cotton', 0.12), 'gold', 8, 14, 7, rows=3, rim2='butter')
    b4 = bank_strip(440, 150, 4, 24, NEAR, 'gold', 13, 20, 6, rows=2, rim2='butter', rowstep=0.9,
                    step=(1.0, 1.4))
    b5 = bank_strip(460, 190, 4, 25, NEAR, 'gold', 18, 28, 8, rows=3, rim2='butter', rowstep=1.0,
                    step=(1.0, 1.4))
    return far1, far2, b3, b4, b5


def _camA(f):
    rise = smooth(seg(f, 72, 134))
    Rv = 90.0 * (1 - rise)
    px = 150.0 * ease_in_out(seg(f, 118, 214))
    return px, Rv


def _balloonA(f, px):
    bx = 236 + 0.18 * (f - 72) + 0.6 * px
    by = 50 - 0.05 * (f - 72)
    return bx, by


def _pennants(c, poles, ox, oy, f):
    for i, (x, y) in enumerate(poles):
        x, y = x + ox, y + oy
        ph = (f // 4 + i * 2) % 4
        cc = RB_NAMES[i % 6]
        dk = shade(cc, 0.75)
        L = 6
        for k in range(L):
            wv = (0, 1, 1, 0)[(ph + k // 2) % 4] if k > 1 else 0
            hh = max(1, 3 - (k * 3) // L)
            c.vline(x + 1 + k, y + wv, y + wv + hh - 1, cc if k < L - 2 or ph % 2 else dk)
        c.px(x + 1, y + 2, dk)


def _scene_city(f):
    px, Rv = _camA(f)
    c = Canvas()
    c.blit(_skyA(), 0, -int(round(Rv * 0.05)))
    far1, far2, b3, b4, b5 = _banksA()
    # sun (low, right)
    sx, sy = 296 - int(round(px * 0.06)), 46 - int(round(Rv * 0.1))
    fx.glow(c, sx, sy, 90, 'gold', 0.28, rings=5)
    fx.glow(c, sx, sy, 34, 'butter', 0.45)
    c.circle(sx, sy, 10, 'butter')
    c.circle(sx, sy, 8, 'cream')
    c.circle(sx - 2, sy - 2, 4, 'white')
    # far cloud sea
    blit_wrap(c, far1, -int(round(px * 0.18 + f * 0.04)), 92 - int(round(Rv * 0.15)))
    # glints on the far sea under the sun
    for i in range(10):
        gx = sx - 40 + int(hash01(i, 1) * 80)
        gy = 95 + int(hash01(i, 2) * 8) - int(round(Rv * 0.15))
        if (f // 6 + i) % 5 == 0:
            c.hline(gx, gx + 2 + i % 3, gy, 'butter')
    blit_wrap(c, far2, -int(round(px * 0.3 + f * 0.07)), 104 - int(round(Rv * 0.2)))
    # back city
    c.blit(_backcity(), -int(round(px * 0.7)) - 10, -int(round(Rv * 0.27)) - 2)
    # falls behind the island + city
    cx0, cy0 = -int(round(px)), -int(round(Rv * 0.3))
    for i, (fxx, fyy, front) in enumerate(FALLS):
        if not front:
            rainbow_fall(c, fxx + cx0, fyy + cy0, 200, 3, f, speed=2.0, seed=i)
    city, poles = _city()
    c.blit(city, cx0, cy0)
    for i, (fxx, fyy, front) in enumerate(FALLS):
        if front:
            rainbow_fall(c, fxx + cx0, fyy + cy0, 200, 3, f, speed=2.0, seed=i)
    c.blit(_city_lips(), cx0, cy0)
    _pennants(c, poles, cx0, cy0, f)
    # window glints + finial sparkle
    for i, (gx, gy) in enumerate(((182, 68), (150, 70), (232, 86), (258, 80), (425, 72), (447, 82))):
        if (f + i * 13) % 48 < 4:
            fx.sparkle(c, gx + cx0, gy + cy0, f, 'white', period=8, size=2)
    fx.sparkle(c, LANTERN[0] + cx0, LANTERN[1] - 3 + cy0, f, 'white', period=36, size=3)
    fx.sparkle(c, 99 + cx0, 46 + cy0, f, 'butter', period=30, phase=0.5, size=2)
    # balloon
    bx, by = _balloonA(f, px)
    draw_balloon(c, int(round(bx)) + cx0, int(round(by)) + cy0, f, size='m', flame=(f // 36) % 3 == 0)
    # near cloud sea (rolling, sweeping past during the rise)
    y3 = 152 - int(round(Rv * 0.55))
    blit_wrap(c, b3, -int(round(px * 1.1 + f * 0.12)), y3)
    for i, (fxx, fyy, front) in enumerate(FALLS):
        spray(c, fxx + 9 + cx0, y3 + 6, 18, f, seed=i, n=12)
    blit_wrap(c, b4, -int(round(px * 1.25 + f * 0.2)), 166 - int(round(Rv * 0.9)))
    blit_wrap(c, b5, -int(round(px * 1.45 + f * 0.3)), 198 - int(round(Rv * 1.45)))
    # wisps rushing past (rise) / drifting
    for i in range(7):
        wy = int(hash01(i, 4) * 260) - 40 + int(round((90 - Rv) * 2.2))
        wy = (wy % 300) - 60
        wx = int((hash01(i, 5) * 400 - f * (0.4 + hash01(i, 6) * 0.4) - px * 1.6) % 420) - 50
        ln = 18 + int(hash01(i, 7) * 22)
        if wy < 150:
            c.dither(wx, wy, ln, 2, 'cream', 0.5)
            c.dither(wx + 4, wy - 1, ln - 8, 1, 'white', 0.5)
    # cloud fog at the start of the rise
    fog = 1 - seg(f, 76, 122)
    if fog > 0:
        veil(c, 0.1 * fog, 0.8 * fog, 'cream', 30, H)
    # callouts + card
    callout(c, LANTERN[0] + cx0 + 3, LANTERN[1] + cy0 - 1, 'CLOUDSDALE', f, 90, 168, side='right', length=16)
    callout(c, COLO[0] + cx0 + 14, COLO[1] + cy0 + 6, 'COLOSSEUM', f, 144, 208, side='up', length=22)
    atlas_card(c, f, 100, 196, 'SKY CITY', ['BUILT FROM CLOUDS', 'WHERE WEATHER IS MADE'])
    return c


# ------------------------------------------------------------------ weather factory (bar 4)
FLAKES_ASCII = [
    '''
    ...w...
    .w.w.w.
    ..www..
    wwwiwww
    ..www..
    .w.w.w.
    ...w...
    ''', '''
    .w...w.
    w.w.w.w
    .wwiww.
    ..iii..
    .wwiww.
    w.w.w.w
    .w...w.
    ''', '''
    ..www..
    .wiiiw.
    wiiwiiw
    wiwwwiw
    wiiwiiw
    .wiiiw.
    ..www..
    ''', '''
    w..w..w
    .w.w.w.
    ..www..
    wwwiwww
    ..www..
    .w.w.w.
    w..w..w
    ''', '''
    ..w.w..
    w.www.w
    .wwiww.
    wwi.iww
    .wwiww.
    w.www.w
    ..w.w..
    ''', '''
    ...i...
    .i.w.i.
    ..www..
    iwwiwwi
    ..www..
    .i.w.i.
    ...i...
    ''', '''
    .......
    ..wiw..
    .w.w.w.
    .iwwwi.
    .w.w.w.
    ..wiw..
    .......
    ''', '''
    ...w...
    ..wiw..
    w.wiw.w
    wiiwiiw
    w.wiw.w
    ..wiw..
    ...w...
    ''',
]
FLAKES = [Sprite.from_ascii(a, {'w': 'white', 'i': 'crystal_lt'}) for a in FLAKES_ASCII]

F_WALL = mix('periwinkle', 'royal', 0.42)
F_WALL_LT = mix('periwinkle', 'royal', 0.25)
F_WALL_DK = mix('periwinkle', 'royal', 0.62)
FW = 336


def _gauge(c, x, y, r, f, i):
    c.circle(x, y, r + 1, 'brass')
    c.circle(x, y, r, 'paper')
    c.px(x - r + 1, y, 'red')
    a = -2.4 + 1.6 * (0.5 + 0.5 * math.sin(f / (11 + i * 3) + i)) + 0.25 * ((f // 3 + i) % 2)
    ex, ey = x + int(round(math.cos(a - math.pi / 2 + 1.6) * (r - 1))), y + int(round(math.sin(a - math.pi / 2 + 1.6) * (r - 1)))
    c.line(x, y, ex, ey, 'crimson')
    c.px(x, y, 'ink')


@lru_cache(maxsize=1)
def _factory_bg():
    cv = Canvas(FW, H)
    cv.gradient(0, 0, FW, 60, ['sky', 'light_sky', 'baby_blue', 'cotton'], soft=0.7)
    cv.gradient(0, 60, FW, 120, ['cotton', 'peach'])
    # distant cloud tops
    b = bank_strip(FW, 80, 2, 31, ramp_haze(NEAR, 'cotton', 0.35), 'gold', 5, 10, 4)
    cv.blit(b, 0, 150)
    # building shell (cut face = cloud)
    X0, X1 = 10, 326
    cv.rect(X0, 26, X1 - X0, 150, A_MID)
    # interiors
    for (ya, yb) in ((36, 96), (104, 164)):
        cv.rect(X0 + 7, ya, X1 - X0 - 14, yb - ya, F_WALL)
        for x in range(X0 + 7, X1 - 7, 26):
            cv.vline(x, ya, yb - 1, F_WALL_DK)
            cv.vline(x + 1, ya, yb - 1, F_WALL_LT)
        cv.hline(X0 + 7, X1 - 8, ya + (yb - ya) * 2 // 3, F_WALL_LT)
        cv.rect(X0 + 7, ya + (yb - ya) * 2 // 3 + 1, X1 - X0 - 14, yb - ya - (yb - ya) * 2 // 3 - 1, F_WALL_DK)
        for x in range(X0 + 12, X1 - 8, 13):
            cv.px(x, ya + (yb - ya) * 2 // 3 + 3, F_WALL)
        cv.rect(X0 + 7, ya, X1 - X0 - 14, 3, F_WALL_DK)
        # floor (cloud slab top with gold trim)
        cv.hline(X0 + 7, X1 - 8, yb - 1, G)
    # back-wall round windows with sky
    for (wx, wy) in ((106, 52), (300, 50), (92, 118), (166, 116)):
        cv.circle(wx, wy, 8, A_MID)
        cv.circle(wx, wy, 7, A_OUT)
        cv.circle(wx, wy, 6, 'light_sky')
        cv.rect(wx - 6, wy + 1, 13, 6, 'baby_blue')
        cv.circle(wx, wy, 6, 'light_sky', fill=False) if False else None
        cv.vline(wx, wy - 6, wy + 6, A_MID)
        cv.hline(wx - 6, wx + 6, wy, A_MID)
        cv.px(wx - 3, wy - 3, 'white')
    # cut-face cloud texture on slabs and walls
    R = rng(33)
    def slab(x0, y0, x1, y1):
        cv.rect(x0, y0, x1 - x0, y1 - y0, A_MID)
        for k in range(int((x1 - x0) * (y1 - y0) / 30)):
            px_ = int(R.uniform(x0, x1)); py_ = int(R.uniform(y0, y1))
            cv.px(px_, py_, A_SH)
            cv.px(px_ + 1, py_ - 1, A_LT)
        cv.hline(x0, x1 - 1, y0, A_LT)
        cv.hline(x0, x1 - 1, y1 - 1, A_SH)
    slab(X0, 26, X1, 36)
    slab(X0, 96, X1, 104)
    slab(X0, 164, X1, 172)
    slab(X0, 26, X0 + 7, 172)
    slab(X1 - 7, 26, X1, 172)
    # roof puffs + chimneys + emblem pediment
    x = X0 - 2
    while x < X1 + 2:
        r = R.uniform(4, 7)
        puff(cv, x, 27, r, ARCH, rim=G_HI)
        x += r * R.uniform(1.0, 1.5)
    for chx in (60, 268):
        cv.rect(chx - 5, 8, 11, 20, A_MID)
        cv.rect(chx - 5, 8, 2, 20, A_SH)
        cv.rect(chx + 5, 8, 1, 20, A_LT)
        cv.hline(chx - 6, chx + 6, 8, A_LT)
        cv.rect(chx - 6, 9, 13, 2, G)
        cv.hline(chx - 6, chx + 6, 11, G_SH)
    pediment(cv, 140, 196, 26, 12, emblem=False)
    # emblem plaque: cloud + bolt
    cv.circle(168, 19, 6, G)
    cv.circle(168, 19, 5, A_WIN2)
    puff(cv, 167, 18, 3, ARCH)
    puff(cv, 170, 19, 2, ARCH)
    cv.line(168, 20, 167, 23, G_HI)
    cv.px(168, 22, G_HI)
    # bottom cloud puffs
    x = X0 - 6
    while x < X1 + 6:
        r = R.uniform(5, 9)
        puff(cv, x, 172 + R.uniform(0, 3), r, NEAR, rim='gold')
        x += r * R.uniform(1.0, 1.4)
    # ceiling pipes
    for py in (39, 107):
        cv.rect(X0 + 7, py, X1 - X0 - 14, 3, 'steel')
        cv.hline(X0 + 7, X1 - 8, py, 'steel_hi')
        cv.hline(X0 + 7, X1 - 8, py + 2, 'steel_dk')
        for jx in range(X0 + 30, X1 - 10, 46):
            cv.rect(jx, py - 1, 3, 5, 'brass')
            cv.px(jx + 2, py - 1, G_HI)
    # ---- floor 1 left: snow machine + conveyor frame
    cv.rect(20, 56, 30, 40, 'steel')
    cv.rect(20, 56, 2, 40, 'steel_dk')
    cv.rect(48, 56, 2, 40, 'steel_hi')
    cv.rect(24, 62, 22, 16, 'crystal_lt')
    cv.rect(24, 62, 22, 2, 'white')
    for k in range(5):
        cv.vline(26 + k * 4, 66, 76, 'crystal')
    cv.rect(24, 82, 22, 10, 'steel_dk')
    cv.px(28, 86, 'leaf')
    cv.px(32, 86, 'gold')
    cv.poly([(18, 44), (52, 44), (42, 56), (28, 56)], 'steel_hi')
    cv.hline(18, 52, 44, 'white')
    cv.poly([(24, 47), (46, 47), (40, 54), (30, 54)], 'steel_dk')
    for k in range(6):
        cv.vline(23 + k * 5, 92, 92 + 1 + k % 2, 'crystal_lt')   # icicles
    cv.rect(50, 77, 5, 4, 'steel_hi')
    # conveyor 1
    cv.rect(52, 84, 102, 3, 'steel_dk')
    for lx in (58, 100, 146):
        cv.rect(lx, 87, 2, 9, 'steel')
        cv.px(lx + 1, 87, 'steel_hi')
    cv.rect(52, 89, 102, 1, 'steel')
    # bin at the end
    cv.rect(156, 78, 16, 18, A_MID)
    cv.rect(156, 78, 2, 18, A_SH)
    cv.hline(155, 172, 78, A_LT)
    FLAKES[2].draw(cv, 161, 84) if hasattr(FLAKES[2], 'draw') else None
    # ---- floor 1 right: tesla charger + jar shelves
    cv.rect(180, 84, 22, 12, 'steel_dk')
    cv.rect(180, 84, 22, 2, 'steel')
    for k in range(12):
        cv.rect(186, 58 + k * 2, 10, 1, 'copper' if k % 2 else 'clay')
    cv.vline(186, 58, 82, 'rust')
    cv.circle(191, 52, 6, 'steel')
    cv.circle(190, 51, 4, 'steel_hi')
    cv.px(188, 49, 'white')
    for sy_ in (68, 93):
        cv.rect(208, sy_, 110, 3, A_MID)
        cv.hline(208, 317, sy_, A_LT)
        cv.hline(208, 317, sy_ + 2, A_SH)
        for bx_ in (210, 314):
            cv.rect(bx_, sy_ + 3, 2, 3, A_SH)
    # gauges board (back wall)
    cv.rect(226, 42, 60, 14, 'steel_dk')
    cv.rect(227, 43, 58, 12, 'steel')
    # ---- floor 2 left: vats frame
    cv.rect(18, 109, 150, 3, 'steel_dk')     # feed pipe
    cv.rect(18, 109, 150, 1, 'steel')
    # ---- floor 2 right: press frame + conveyor 2
    cv.rect(176, 146, 96, 3, 'steel_dk')
    cv.rect(176, 149, 96, 1, 'steel')
    for lx in (182, 230, 266):
        cv.rect(lx, 150, 2, 14, 'steel')
    # feed chute (rain clouds drop onto the belt)
    cv.poly([(174, 106), (204, 106), (196, 122), (182, 122)], 'steel')
    cv.hline(174, 204, 106, 'steel_hi')
    cv.rect(180, 109, 18, 9, mix('glass', F_WALL, 0.5))
    cv.rect(180, 112, 18, 6, 'grey')
    cv.dither(180, 111, 18, 3, 'stone', 0.5)
    cv.hline(180, 197, 117, mix('grey', 'indigo', 0.3))
    cv.vline(181, 110, 116, 'glass_hi')
    cv.rect(184, 122, 10, 3, 'steel_dk')
    for (px0) in (236,):
        cv.rect(px0 - 14, 110, 3, 36, 'steel')
        cv.rect(px0 + 13, 110, 3, 36, 'steel')
        cv.vline(px0 - 14, 110, 145, 'steel_dk')
        cv.vline(px0 + 15, 110, 145, 'steel_hi')
        cv.rect(px0 - 16, 108, 34, 6, 'brass')
        cv.hline(px0 - 16, px0 + 17, 108, G_HI)
        cv.hline(px0 - 16, px0 + 17, 113, G_SH)
    # pallet for stacked clouds
    cv.rect(272, 158, 46, 6, 'tan')
    cv.hline(272, 317, 158, 'sand')
    for k in range(272, 318, 8):
        cv.vline(k, 159, 163, 'clay')
    # drain grate
    cv.rect(272, 164, 46, 1, 'steel_dk')
    return cv


def _jar(c, x, y, f, i):
    """Glass jar 10x15 (top-left x,y) with a crackling bolt."""
    w, h = 10, 15
    on = hash01(i, f // 3) > 0.25
    c.rect(x + 1, y + 3, w - 2, h - 4, mix('glass', F_WALL, 0.55))
    if on:
        fx.glow(c, x + w // 2, y + h // 2 + 1, 10, 'butter', 0.35, rings=3)
        pts = [(x + w // 2, y + 4)]
        for k in range(1, 5):
            jx = x + 2 + int(hash01(i, k, f // 3) * (w - 4))
            pts.append((jx, y + 4 + k * 2 + 1))
        for a, b in zip(pts, pts[1:]):
            c.line(a[0], a[1], b[0], b[1], 'butter')
        for p in pts[1:-1]:
            c.px(p[0], p[1], 'white')
    else:
        c.px(x + w // 2, y + 8, 'gold')
    c.vline(x, y + 3, y + h - 2, 'glass')
    c.vline(x + w - 1, y + 3, y + h - 2, 'glass')
    c.hline(x + 1, x + w - 2, y + h - 1, 'glass')
    c.vline(x + 1, y + 4, y + 7, 'glass_hi')
    c.rect(x + 1, y, w - 2, 3, 'brass')
    c.hline(x + 1, x + w - 2, y, G_HI)
    c.px(x + w // 2, y - 1, 'steel_hi')


def _vat(c, x, y, f, i):
    """Glass vat 28 x 34 with rainbow layers, bubbles."""
    w, h = 28, 34
    top = y + 8
    xs = np.arange(x + 2, x + w - 2)
    ys = np.arange(top, y + h - 3)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    wav = np.round(np.sin(xx * 0.45 + f * 0.18 + i) * 1.0)
    band = np.clip(((yy - top + wav) // 4).astype(int), 0, 5)
    out = RB[band].copy()
    hi = ((xx + yy + f // 2) % 9 == 0)
    out[hi] = RB_HI[band][hi]
    c.a[top:y + h - 3, x + 2:x + w - 2] = out
    # surface wobble line
    for xx_ in range(x + 2, x + w - 2):
        dy = int(round(math.sin(xx_ * 0.45 + f * 0.18 + i)))
        c.px(xx_, top - 1 + max(0, dy), RB_HI[0])
    # bubbles
    for k in range(6):
        P = 26 + k * 3
        t = ((f + k * 11 + i * 7) % P) / P
        bx = x + 4 + int(hash01(i, k) * (w - 9)) + int(math.sin(t * 6 + k) * 1.2)
        byy = y + h - 5 - int(t * (h - 13))
        if t < 0.92:
            c.px(bx, byy, 'white')
            if k % 2 == 0:
                c.px(bx + 1, byy, 'white')
        else:
            c.px(bx - 1, top - 2, 'white')
            c.px(bx + 1, top - 2, 'white')
            c.px(bx, top - 3, 'white')
    # glass + frame
    c.vline(x + 2, top, y + h - 4, 'glass_hi')
    c.vline(x + 3, top + 2, top + 8, 'glass_hi')
    c.rect(x, y + 4, 2, h - 4, 'brass')
    c.rect(x + w - 2, y + 4, 2, h - 4, 'brass')
    c.vline(x + w - 1, y + 4, y + h - 1, G_SH)
    c.rect(x - 1, y + 2, w + 2, 3, 'copper')
    c.hline(x - 1, x + w, y + 2, 'clay')
    c.rect(x - 1, y + h - 3, w + 2, 3, 'copper')
    c.hline(x - 1, x + w, y + h - 3, 'clay')
    c.px(x + 4, y + h - 2, G_HI)
    c.px(x + w - 5, y + h - 2, G_HI)
    fx.steam(c, x + w // 2, top - 3, f, color=RB_PAS[(i * 2) % 6], n=3, height=10, spread=3, seed=i)


RAIN_RAMP = (mix('grey', 'indigo', 0.3), 'grey', mix('stone', 'cloud_dk', 0.4), mix('stone', 'cloud', 0.55))


@lru_cache(maxsize=None)
def _rain_sprite(v):
    cv = _new(22, 16)
    R = rng(70 + v)
    circ = [(11 + R.uniform(-1, 1), 10, 6), (6, 11, 4.5), (16, 11, 4.5),
            (8 + R.uniform(-1, 1), 7, 4 + R.uniform(0, 1)), (14, 6 + R.uniform(0, 1), 4.5)]
    cloud_mass(cv, circ, RAIN_RAMP, rim='cloud')
    cv.hline(3, 18, 15, KEY)
    return _rgba(cv)


@lru_cache(maxsize=None)
def _rain_squashed(v, h):
    s = _rain_sprite(v)
    rows = np.round(np.linspace(0, s.shape[0] - 1, h)).astype(int)
    return s[rows]


def _raincloud(c, x, y, f, k, squash=0.0):
    """Fluffy grey rain cloud (bottom-centre x,y); squash 0..1 -> pressed flat block."""
    h = int(round(16 - 9 * clamp(squash)))
    spr = _rain_squashed(k % 4, h)
    c.blit(spr, x - 11, y - h)
    if squash < 0.5 and (f // 4 + k) % 3 == 0:
        for j in range(3):
            c.px(x - 5 + j * 5 + (f // 4) % 2, y + 1 + (f + j * 2) % 4, 'light_sky')


def _scene_factory(f):
    lf = f - 216
    camx = int(round(14 * smooth(seg(lf, 0, 71))))
    big = _factory_bg()
    c = Canvas()
    c.blit(big, -camx, 0)
    o = -camx
    # chimneys vapour
    for k, chx in enumerate((60, 268)):
        for j in range(4):
            P = 40
            t = ((lf + j * 10 + k * 5) % P) / P
            r = 2 + int(t * 5)
            yy = 6 - int(t * 14)
            xx = chx + o + int(t * 10) + (j % 2)
            c.ellipse(xx, yy, r + 1, r, 'white' if j % 2 else 'cloud')
    # balloon in the sky
    draw_balloon(c, 92 + o + int(lf * 0.25), 20, f, size='s')
    # light shafts from round windows
    for (wx, wy) in ((106, 52), (300, 50), (92, 118), (166, 116)):
        fx.light_beam(c, ((wx - 4 + o, wy + 4), (wx + 4 + o, wy + 4)), ((wx - 2 + o, wy + 40), (wx + 16 + o, wy + 40)),
                      'butter', 0.12)
    # ---- snow line
    for k in range(7):
        sx = 50 + int((k * 14 + lf * 0.6) % 98)
        idx = (k + int((k * 14 + lf * 0.6) // 98) * 3) % len(FLAKES)
        FLAKES[idx].draw(c, sx + o, 77) if hasattr(FLAKES[idx], 'draw') else c.blit(FLAKES[idx], sx + o, 77)
    # dropping flake into bin
    t = (lf * 0.6 % 14) / 14
    c.blit(FLAKES[(lf // 23) % len(FLAKES)], 150 + o + int(t * 4), 77 + int(t * t * 8))
    # belt tread + rollers
    for x in range(52, 154):
        if (x + int(lf * 0.6)) % 4 == 0:
            c.px(x + o, 84, 'steel')
    for rx in (54, 152):
        c.circle(rx + o, 86, 3, 'steel')
        a = lf * 0.3
        c.px(rx + o + int(round(math.cos(a) * 2)), 86 + int(round(math.sin(a) * 2)), 'steel_hi')
        c.px(rx + o, 86, 'steel_dk')
    # cloud fed into the hopper
    puff(c, 35 + o, 44 + int(math.sin(lf / 6) * 1), 6, ARCH, rim=G_HI)
    fx.steam(c, 47 + o, 60, lf, color='crystal_lt', n=3, height=8, seed=4)
    # machine lamps
    for k in range(3):
        on = (lf // 6 + k) % 3 == 0
        c.px(36 + k * 3 + o, 86, ('leaf', 'gold', 'red')[k] if on else 'steel')
    # ---- lightning jars
    for row, sy_ in enumerate((68, 93)):
        for j in range(9):
            _jar(c, 212 + j * 11 + o, sy_ - 15, f, row * 9 + j)
    # charger arc on the beat
    ph = f % FRAMES_PER_BEAT
    if ph < 5:
        tgt = (212 + ((f // 18) % 9) * 11 + 5 + o, 55)
        pts = [(197 + o, 50)]
        n = 5
        for k in range(1, n):
            u = k / n
            pts.append((int(lerp(197 + o, tgt[0], u)), int(lerp(50, tgt[1], u) + (hash01(f, k) - 0.5) * 8)))
        pts.append(tgt)
        for a, b in zip(pts, pts[1:]):
            c.line(a[0], a[1], b[0], b[1], 'butter' if ph < 3 else 'gold')
        fx.glow(c, 191 + o, 52, 16, 'butter', 0.5)
        for p in pts[1:-1]:
            c.px(p[0], p[1] - 1, 'white')
    else:
        fx.glow(c, 191 + o, 52, 9, 'lavender', 0.3)
    # gauges + blinking lamps
    for k in range(3):
        _gauge(c, 236 + k * 16 + o, 49, 4, f, k)
    for k in range(4):
        on = (f // 9 + k * 2) % 4 < 2
        c.px(281 + o, 45 + k * 3, ('red', 'leaf', 'gold', 'sky')[k] if on else 'steel_dk')
    # ---- rainbow vats
    for i in range(3):
        _vat(c, 22 + i * 46 + o, 130, f, i)
    # feed pipe pouring into vat 1, glass pipes between vats
    rainbow_fall(c, 33 + o, 112, 138, 1, f, speed=1.5, seed=9, lip=False)
    for i in range(2):
        x0 = 50 + i * 46 + o
        c.rect(x0, 146, 18, 3, 'glass')
        for x in range(x0, x0 + 18):
            if (x - lf) % 5 < 2:
                c.px(x, 147, RB_NAMES[(x // 5 + lf // 5) % 6])
        c.rect(x0 + 7, 145, 4, 5, 'brass')
    for i in range(3):
        _gauge(c, 36 + i * 46 + o, 121, 3, f + 40, i + 3)
    # ---- cloud press
    phb = (f % FRAMES_PER_BEAT) / FRAMES_PER_BEAT
    step = f // FRAMES_PER_BEAT
    mv = ease_in_out(seg(phb, 0, 0.45))
    press = math.sin(math.pi * seg(phb, 0.5, 1.0))
    PX = 236
    for k in range(-1, 6):
        x = 188 + int(round((k + mv) * 24))
        if x > 262 or x < 186:
            continue
        slot = k + mv
        if slot < 1.5:
            sq = 0.0
        elif slot < 2.0 and k == 1 and mv >= 1:
            sq = 0.0
        else:
            sq = 1.0
        if k == 1 and mv >= 1:      # under the press right now
            sq = min(1.0, press * 1.4) if phb < 0.75 else 1.0
        _raincloud(c, x + o, 146, f, (step - k) % 7, sq)
    # pistons
    ptop = 114
    pl = 120 + int(round(press * 14))
    c.rect(PX - 2 + o, ptop, 5, pl - ptop, 'steel_hi')
    c.vline(PX - 2 + o, ptop, pl, 'steel')
    c.rect(PX - 9 + o, pl, 19, 4, 'steel')
    c.hline(PX - 9 + o, PX + 9 + o, pl, 'steel_hi')
    c.hline(PX - 9 + o, PX + 9 + o, pl + 3, 'steel_dk')
    if press > 0.95:
        fx.steam(c, PX - 10 + o, pl + 2, f, color='white', n=3, height=6, seed=5)
        fx.steam(c, PX + 10 + o, pl + 2, f, color='white', n=3, height=6, seed=6)
    fx.steam(c, PX + o + 14, 106, f, color='white', n=4, height=10, seed=7)
    # stacked pressed clouds dripping rain
    for r in range(4):
        for k in range(3 - (1 if r == 3 else 0)):
            _raincloud(c, 282 + k * 15 + (7 if r % 2 else 0) + o, 158 - r * 8, f, r * 3 + k, 1.0)
    for k in range(6):
        P = 14
        t = ((lf + k * 5) % P) / P
        dx = 276 + k * 7 + o
        c.px(dx, 159 + int(t * 6), 'light_sky')
    # hanging lamps
    for (lx, ly) in ((130, 42), (200, 42), (110, 110), (260, 110)):
        c.vline(lx + o, ly, ly + 5, 'steel_dk')
        c.rect(lx - 2 + o, ly + 5, 5, 2, 'brass')
        c.px(lx + o, ly + 7, 'butter')
        fx.glow(c, lx + o, ly + 8, 18, 'gold', 0.22 + 0.04 * math.sin(f / 5 + lx))
    callout(c, 168 + o, 13, 'WEATHER FACTORY', f, 222, 280, side='right', length=10)
    return c


# ------------------------------------------------------------------ rainbow falls close-up (bar 5)
@lru_cache(maxsize=1)
def _falls_bg():
    cv = Canvas(360, H)
    cv.gradient(0, 0, 360, 150, ['sky', 'light_sky', 'baby_blue', 'cotton', 'peach'], soft=0.7)
    cv.gradient(0, 150, 360, 30, ['peach', 'sand'])
    return cv


@lru_cache(maxsize=1)
def _gorge():
    """Galloping Gorge far below: plains split by a winding red-rock canyon with a river (hazed)."""
    cv = Canvas(200, 40)
    cv.gradient(0, 0, 200, 40, ['pale_sage', 'khaki'], soft=0.8)
    R = rng(51)
    for k in range(60):
        x, y = int(R.uniform(0, 200)), int(R.uniform(0, 40))
        cv.ellipse(x, y, int(R.uniform(2, 6)), 1, 'sage' if k % 3 else 'moss')
    for k in range(40):
        x, y = int(R.uniform(0, 200)), int(R.uniform(0, 40))
        cv.px(x, y, 'forest')
        cv.px(x + 1, y, 'green')
    for y in range(40):
        xc = 92 + 16 * math.sin(y / 9.0 + 0.6) + y * 0.4
        hw = 7 + 2 * math.sin(y / 5.0)
        x0, x1 = int(round(xc - hw)), int(round(xc + hw))
        for x in range(x0, x1 + 1):
            d = x - xc
            if abs(d) <= 1.2:
                cc = 'sky'
            elif d < 0:
                cc = ('brick', 'rust', 'brick', 'rust', 'maroon')[int(-d) % 5]
            else:
                cc = ('clay', 'sand', 'copper', 'clay', 'sand')[int(d) % 5]
            cv.px(x, y, cc)
        cv.px(x0 - 1, y, 'sand')
        cv.px(x1 + 1, y, 'tan')
        if y % 6 == 0:
            cv.px(int(round(xc)), y, 'white')
    a = cv.a.astype(np.float32)
    cv.a[:] = (a * 0.6 + np.array(col('baby_blue'), np.float32) * 0.4).astype(np.uint8)
    return cv


@lru_cache(maxsize=1)
def _falls_layers():
    # cloud sea with a gap (gorge seen through it)
    sea = bank_strip(400, 60, 2, 61, ramp_haze(NEAR, 'cotton', 0.15), 'gold', 6, 13, 6)
    yy, xx = np.mgrid[0:60, 0:400]
    hole = np.zeros((60, 400), bool)
    for (ex, ey, rx, ry) in ((66, 34, 26, 13), (96, 28, 30, 14), (124, 33, 22, 12), (84, 44, 30, 14),
                             (112, 46, 26, 12)):
        hole |= ((xx - ex) / rx) ** 2 + ((yy - ey) / ry) ** 2 < 1
    wall = np.zeros_like(hole)
    wall[:-3] = hole[3:]
    wall &= ~hole
    sea[wall, :3] = col(CL_DEEP)
    sea[wall, 3] = 255
    lip = np.zeros_like(hole)
    lip[:-1] = hole[1:]
    lip &= ~hole
    sea[hole, 3] = 0
    top_edge = np.zeros_like(hole)
    top_edge[1:] = wall[:-1] & ~wall[1:]
    sea[wall & ~np.roll(wall, 1, axis=0), :3] = col(CL_SH)
    below = np.zeros_like(hole)
    below[1:] = hole[:-1]
    below &= ~hole
    sea[below, :3] = col(CL_LT)
    # underside of the city (top of frame) with a cloud lip where the big fall spills
    cv = _new(400, 60)
    R = rng(62)
    cv.rect(140, 0, 260, 8, CL_DEEP)
    circ = []
    x = 132
    while x < 400:
        r = R.uniform(8, 14)
        circ.append((x, 6 + R.uniform(0, 7) + (5 if 200 < x < 300 else 0), r))
        x += r * R.uniform(0.8, 1.2)
    cloud_mass(cv, circ, (mix(CL_DEEP, 'lavender', 0.25), CL_DEEP, CL_SH, CL_MID), rim='peach')
    cloud_mass(cv, [(228, 22, 8), (240, 26, 9), (254, 27, 9), (268, 25, 8), (281, 21, 7)],
               ARCH, rim=G, rim2=G_HI)
    under = _rgba(cv)
    # small far islet with a thin fall
    cv2 = _new(80, 46)
    hz = 'cotton'
    temple(cv2, 22, 17, 4, colh=7, step=5, emblem=False)
    cloud_mass(cv2, [(18, 22, 7), (30, 20, 9), (43, 22, 8), (54, 24, 5)],
               ramp_haze(ARCH, hz, 0.25), rim=mix(G_HI, hz, 0.3))
    a = cv2.a
    m = np.any(a != _KEYA, axis=2)
    a[m] = (a[m].astype(np.float32) * 0.75 + np.array(col(hz), np.float32) * 0.25).astype(np.uint8)
    islet = _rgba(cv2)
    return sea, under, islet


MIST_RAMP = (mix('cotton', 'lavender', 0.35), mix('cotton', 'baby_blue', 0.3), mix('white', 'cotton', 0.35), 'white')


def mist_cluster(c, cx, cy, w, f, seed=0, n=10, rmax=13, rim='butter'):
    """Billowing mist mass (rolling puffs rise, swell and fade) centred (cx, cy), width w."""
    circ = []
    for i in range(n):
        P = 54 + int(hash01(seed, i) * 30)
        t = ((f + hash01(seed, i, 1) * P) % P) / P
        x = cx + (hash01(seed, i, 2) - 0.5) * w + math.sin(t * 3 + i) * 3
        y = cy - t * 12 + hash01(seed, i, 3) * 6
        r = rmax * (0.45 + 0.4 * math.sin(math.pi * t) + 0.2 * hash01(seed, i, 4))
        circ.append((x, y, r))
    for i in range(5):
        circ.append((cx + (i - 2) * w * 0.22, cy + 6, rmax * 0.9))
    cloud_mass(c, circ, MIST_RAMP, rim=rim)


def _scene_falls(f):
    lf = f - 288
    camx = 30 * smooth(seg(lf, 0, 80))
    c = Canvas()
    c.blit(_falls_bg(), -int(round(camx * 0.15)), 0)
    sea, under, islet = _falls_layers()
    # light rays from the low sun (upper right)
    for k in range(3):
        x0 = 250 + k * 30 - int(camx * 0.3)
        fx.light_beam(c, ((x0, 0), (x0 + 10, 0)), ((x0 - 120, 150), (x0 - 80, 150)), 'butter', 0.10)
    # gorge through the gap + the cloud sea
    gx = -int(round(camx * 0.35))
    c.blit(_gorge(), gx + 0, 146)
    blit_wrap(c, sea, gx - int(lf * 0.05), 132)
    # far islet + thin fall
    ix = 28 - int(round(camx * 0.5))
    rainbow_fall(c, ix + 33, 52, 140, 1, f, speed=1.2, seed=3, sheets=False)
    mist_cluster(c, ix + 36, 138, 16, f, seed=4, n=4, rmax=5)
    c.blit(islet, ix, 32)
    # the big rainbow curtain
    ox = -int(round(camx))
    FX0 = 196
    rainbow_fall(c, FX0 + ox, 26, 152, 9, f, speed=3.0, seed=12)
    # sheen + falling droplets at the edges
    for k in range(14):
        P = 18
        t = ((lf + k * 7) % P) / P
        side = k % 2
        ex = FX0 + ox + (-1 - int(t * 3) if side == 0 else 54 + int(t * 3))
        c.px(ex, 40 + (k * 9 + int(t * 30)) % 100, RB_PAS[0 if side == 0 else 5])
    c.blit(under, -int(round(camx)) - 30, 0)
    # mist bank at the plunge (rolling)
    mist_cluster(c, FX0 + 27 + ox, 144, 92, f, seed=5, n=12, rmax=15)
    spray(c, FX0 + 27 + ox, 136, 70, f, seed=5, n=30, big=2.0)
    # prismatic bow in the mist (appears on the hit)
    k = ease_out(seg(lf, 0, 6)) * (0.7 - 0.3 * seg(lf, 14, 70)) * (0.85 + 0.15 * math.sin(lf / 3))
    if k > 0:
        cxb, cyb, R0 = FX0 + 27 + ox, 160, 36
        y0b = cyb - R0 - 13
        yy, xx = np.mgrid[y0b:cyb, cxb - R0 - 13:cxb + R0 + 13]
        d = np.sqrt((xx - cxb) ** 2 + (yy - cyb) ** 2)
        band = np.floor((d - R0) / 2).astype(int)
        m = (band >= 0) & (band < 6)
        thr = bayer(yy.shape[0], yy.shape[1], xx[0, 0], yy[0, 0])
        m &= thr < k
        xs_ok = (xx >= 0) & (xx < W) & (yy >= 0)
        m &= xs_ok
        ys_, xs_ = yy[m], xx[m]
        c.a[ys_, xs_] = RB_HI[::-1][band[m]]
    # the balloon sails through the rainbow mist
    bt = seg(lf, 0, 71)
    bx = int(round(54 + 210 * bt)) - int(round(camx * 0.2))
    by = int(round(126 - 18 * smooth(bt)))
    draw_balloon(c, bx, by, f, size='l', flame=(f // 30) % 2 == 0)
    # mist veils the balloon when it is inside the spray
    if 160 < bx < 290:
        a = 0.45 * (1 - abs(bx - (FX0 + 27 + ox)) / 90)
        if a > 0:
            m = bayer(16, 30, bx - 15, by - 16) < a
            reg = c.a[by - 16:by, max(0, bx - 15):bx + 15]
            mm = m[:, :reg.shape[1]]
            reg[mm] = col('white')
    # spray burst on the hit
    for i in range(46):
        dt = lf - (i % 4)
        if dt < 0 or dt > 40:
            continue
        ang = -math.pi * (0.1 + 0.8 * hash01(i, 1))
        v = 1.6 + 2.6 * hash01(i, 2)
        x = FX0 + 27 + ox + math.cos(ang) * v * dt
        y = 140 + math.sin(ang) * v * dt + 0.09 * dt * dt
        cc = RB_PAS[i % 6] if i % 3 else 'white'
        c.px(int(x), int(y), cc)
        if dt < 10:
            c.px(int(x) + 1, int(y), cc)
    if lf < 30:
        fx.sparkles(c, FX0 + 27 + ox, 120, f, radius=40, n=10, colors=('white', 'butter', 'cotton'), seed=8, period=16)
    fx.sparkles(c, FX0 + 27 + ox, 100, f, radius=50, n=6, colors=('white', 'butter'), seed=9, period=34)
    callout(c, FX0 + ox + 4, 74, 'LIQUID RAINBOW', f, 294, 348, side='left', length=14)
    return c


# ------------------------------------------------------------------ main
def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < 216:
        c = _scene_city(f)
        if f < 80:
            px, Rv = _camA(f)
            iris(c, 1 - seg(f, 72, 79), DOME[0] - int(round(px)), DOME[1] - int(round(Rv * 0.3)))
        return c
    if f < 288:
        return _scene_factory(f)
    return _scene_falls(f)


SFX = [
    (80, 'wind', 0.5),
    (96, 'shimmer', 0.45),
    (126, 'waterfall', 0.35),
    (162, 'wind', 0.4),
    (216, 'hum', 0.6),
    (225, 'clank', 0.45),
    (234, 'zap', 0.4),
    (243, 'steam', 0.4),
    (252, 'bubbles', 0.4),
    (261, 'clank', 0.4),
    (270, 'zap', 0.35),
    (288, 'shimmer', 0.8),
    (289, 'waterfall', 0.6),
    (292, 'sparkle', 0.6),
    (324, 'whoosh', 0.35),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'zoom', {'cx': 165, 'cy': 78}), (288, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
