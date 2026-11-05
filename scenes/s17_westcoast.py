"""WEST COAST: Vanhoover, Tall Tale, the Smokey Mountains, Silver Shoals, Luna Bay.

5 bars in SCENE frames (N = 360; the 2-bar map flight is added by engine/regionfx.long_flight),
HUD clock 20:30 -> 21:00. A cozy rainy NIGHT on the north-west coast: cool blue-teal ambient, warm
window and lamp light as the key, wet reflections everywhere, a light slanted drizzle; at `moonrise`
(288) the clouds break and silver moonlight floods the sea. Shot changes at 180 and 270 are night
cloud passes (engine/shotfx.smooth_cuts, 24-frame windows). Cameras use shotfx.cruise.

Beat sheet (scene frames; bar = 72, beat = 18, 8th = 9):
  0-71     World-map travel shot (`travel_frame`); its iris closes on (204, 103).
  72-179   SHOT A  VANHOOVER (3.6 s). Iris opens 72-79 on the hero glass tower (204, 103). Pale
           periwinkle glass towers with lit window grids (windows switching, blinking spire lights)
           on a misty bay; green suspension bridge with necklace lights from the pine headland;
           white sail-roof pavilion; Unicorn Range teal silhouettes; low clouds lit from below.
           Big ferry crossing (horn 108), a little ferry far out; the balloon drifts over the bay,
           burner glowing (reflected). Wet seawall promenade with lamps and puddle glints. Slow
           cruise pan right to TALL TALE's warm lights. Callouts VANHOOVER 90-162, TALL TALE 99-171.
           Atlas card 100-206 (crosses the cloud pass).
  180-269  SHOT B  SMOKEY MOUNTAINS (3.0 s). Twin wooded peaks: wooden fortress with waving
           horseshoe banners (left), log cabins with chimney smoke (right); a valley of replanted
           saplings among stumps, creek, footbridge, lanterns; rain mist. Balloon drifting in the
           valley. From 236 the cloud deck silvers at the top right. Callout SMOKEY MOUNTAINS 186-258.
  270-359  SHOT C  SILVER SHOALS / LUNA BAY (3.0 s). Still raining, the moon hidden; at `moonrise`
           (288) the clouds part, the moon breaks out with a bloom, a silver path runs across the
           sea, moonlight rims everything, rain thins, stars appear. Pastel stilt houses, boardwalk,
           docks, bobbing sailboats, striped lighthouse with a sweeping beam; the balloon moored at
           the landing. Callouts SILVER SHOALS 279-351, LUNA BAY 288-352. Last 8 frames: wipe.
Ambient: drizzle (2 layers), rain rings, drifting clouds and mist, water shimmer and reflections,
         windows switching, blinking lights, ferries + wakes, lamp flicker, balloon bob + burner,
         chimney smoke, banners, creek flow, boats bobbing, lighthouse beam, moon glitter, stars.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.core import bayer
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'westcoast'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

T_A, T_B, T_C = 72, 180, 270
T_MOON = 288                 # `moonrise` hit (bar line inside shot C)
KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)


# ================================================================== generic helpers
def layer(w, h=H):
    return Canvas(w, h, KEY)


def to_rgba(cv):
    a = cv.a
    m = ~np.all(a == _KEYA, axis=2)
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = m.astype(np.uint8) * 255
    return out


def paste(c, rgba, x, y=0):
    c.blit(rgba, int(x), int(y))


def ri(rr, a, b):
    return int(rr.integers(a, b + 1))


def smooth_noise(h, w, seed, scale=(9.0, 31.0)):
    """Cheap smooth 2-D noise (sum of sines) in 0..1, deterministic."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = np.zeros((h, w), np.float32)
    amp = 1.0
    tot = 0.0
    for k in range(4):
        a1 = hash01(seed, k, 1) * 6.28
        a2 = hash01(seed, k, 2) * 6.28
        sy, sx = scale[0] / (1.7 ** k), scale[1] / (1.7 ** k)
        v += amp * (np.sin(xx / sx + a1 + np.sin(yy / (sy * 1.3) + a2) * 1.4) *
                    np.cos(yy / sy + a2 * 0.7 + np.sin(xx / (sx * 0.8)) * 0.9))
        tot += amp
        amp *= 0.55
    return (v / tot) * 0.5 + 0.5


@lru_cache(maxsize=None)
def mist_mask(h, w, seed, level):
    """Dithered soft mist blobs (bool mask) of size h x w, wraps horizontally."""
    n = smooth_noise(h, w, seed, (5.0, 23.0))
    yy = np.arange(h)[:, None]
    env = np.sin(np.pi * (yy + 0.5) / h) ** 0.8          # fade at the band's top/bottom edges
    v = n * env
    thr = bayer(h, w) * 0.9 + 0.05
    return (v * level * 2.2) > thr


def mist(c, y0, h, f, color, alpha, seed=0, level=0.5, speed=0.25, ox=0):
    """Drifting band of dithered mist (blend toward color)."""
    m = mist_mask(h, 640, seed, level)
    off = int(f * speed + ox) % 640
    idx = (np.arange(W) + off) % 640
    mm = m[:, idx]
    ys0, ys1 = max(0, y0), min(H, y0 + h)
    if ys1 <= ys0:
        return
    mm = mm[ys0 - y0:ys1 - y0]
    reg = c.a[ys0:ys1]
    cc = np.array(col(color), np.float32)
    sel = reg[mm].astype(np.float32)
    reg[mm] = np.clip(sel + (cc - sel) * alpha, 0, 255).astype(np.uint8)


def rain_layer(c, f, n, color, seed, slant=1, speed=6, length=3, region=(0, 0, W, H), shift=0):
    """Slanted drizzle: like fx.rain but drop columns are anchored to world shift (parallax)."""
    x0, y0, w, h = region
    cc = col(color)
    a = c.a
    for i in range(n):
        x = x0 + int((hash01(seed, i) * w + f * slant * speed / 4 - shift) % w)
        y = y0 + int((hash01(seed, i, 2) * h + f * speed) % h)
        for k in range(length):
            yy = y + k
            xx = x - (k * slant) // 2
            if 0 <= xx < W and 0 <= yy < H:
                a[yy, xx] = cc


def rain_rings(c, f, y0, y1, n, color, seed, shift=0):
    """Rain hitting water: tiny expanding rings (dot -> dash -> split)."""
    for i in range(n):
        per = 12 + int(hash01(seed, i, 7) * 8)
        ph = int(hash01(seed, i, 8) * per)
        k = (f + ph) // per
        t = (f + ph) % per
        x = int((hash01(seed, i, k) * (W + 40) - shift) % (W + 40)) - 20
        y = y0 + int(hash01(seed, i, k, 3) * (y1 - y0))
        if t == 0 or t == 1:
            c.px(x, y, color)
        elif t < 5:
            c.hline(x - 1, x + 1, y, color)
        elif t < 8:
            c.px(x - 2, y, color)
            c.px(x + 2, y, color)


def reflect(c, yw, y1, f, water_top, water_bot, k0=0.62, k1=0.22, stretch=0.72, wob=1.6,
            bright_boost=0.5, mask=None):
    """Mirror everything above the waterline yw into rows yw..y1-1 (rippled, stretched, tinted).

    water_top/bottom: water colours at the top and bottom of the band. Bright pixels (windows,
    lamps) keep more of their colour, so lights smear into long shimmering streaks."""
    rows = y1 - yw
    if rows <= 0:
        return
    ys = np.arange(yw, y1)
    d = ys - yw
    src = np.clip(yw - 1 - (d * stretch).astype(int), 0, yw - 1)
    amp = 0.4 + wob * d / rows
    ph = (f // 3) * 0.55
    off = np.round(np.sin(ys * 0.83 + ph) * amp + np.sin(ys * 2.17 - ph * 0.7) * 0.45 * amp).astype(int)
    xs = np.clip(np.arange(W)[None, :] + off[:, None], 0, W - 1)
    refl = c.a[src[:, None], xs].astype(np.float32)
    tt = (d / max(1, rows - 1))[:, None, None]
    water = np.array(col(water_top), np.float32) * (1 - tt) + np.array(col(water_bot), np.float32) * tt
    k = (k0 + (k1 - k0) * tt[:, :, 0])
    lum = refl.mean(axis=2)
    k = k * (1 + bright_boost * (lum > 150))
    # dark ripple gaps: some rows lose the reflection (broken water surface)
    gap = ((ys * 5 + (f // 4) * 3) % 7 == 0)[:, None] & (bayer(rows, W, 0, yw) < 0.55)
    k = np.where(gap, k * 0.35, k)
    k = np.clip(k, 0, 0.95)[:, :, None]
    out = water + (refl - water) * k
    out = np.clip(out, 0, 255).astype(np.uint8)
    if mask is None:
        c.a[yw:y1] = out
    else:
        reg = c.a[yw:y1]
        reg[mask] = out[mask]


def pine(cv, x, by, h, body, dark, hi, out=None, lean=0, tiers=None, trunk='#2a2026'):
    """Tiered conifer with base centre (x, by), height h (cached builders only)."""
    if out is None:
        out = shade(body, 0.6)
    tiers = tiers or max(2, h // 6)
    wmax = max(3.0, h * 0.42)
    tl = h / tiers
    rows = []
    for r in range(h):
        y = by - h + r
        t = r / max(1, h - 1)
        p = (r % tl) / tl
        hw = (0.12 + 0.88 * t) * wmax / 2 * (0.45 + 0.55 * p) + 0.3
        cx = x + lean * (1 - t)
        x0, x1 = int(round(cx - hw)), int(round(cx + hw))
        rows.append((y, x0, x1))
    for y, x0, x1 in rows:
        cv.hline(x0 - 1, x1 + 1, y, out)
    cv.vline(int(x), by - 2, by, trunk)
    for y, x0, x1 in rows:
        cv.hline(x0, x1, y, body)
        sw = max(1, (x1 - x0) * 2 // 5)
        cv.hline(x1 - sw + 1, x1, y, dark)
        cv.px(x0, y, hi)
        if x1 - x0 > 4:
            cv.px(x0 + 1, y, hi)


def ellipse_layer(cv, cx, cy, rx, ry, c):
    cv.ellipse(cx, cy, rx, ry, c)


# ================================================================== shared sky / cloud pieces
SKY_NIGHT = ['#0a0d1f', '#0e1328', '#121a32', '#16213b', '#1b2a45', '#22354f', '#2a4258']


@lru_cache(maxsize=None)
def cloud_deck(w, h, seed, base, mid, lit, rim, n=40, ry=(3, 7), rx=(10, 26), lit_from='below'):
    """Rain-cloud bank (RGBA): overlapping soft ellipses, darker tops, lit undersides."""
    cv = layer(w, h)
    rr = rng(seed)
    blobs = []
    for i in range(n):
        x = rr.uniform(0, w)
        y = rr.uniform(h * 0.15, h * 0.75)
        a = rr.uniform(*rx)
        b = rr.uniform(*ry)
        blobs.append((x, y, a, b))
    for dx in (-w, 0, w):           # draw wrapped copies so the bank tiles
        for x, y, a, b in blobs:
            if lit_from == 'below':
                cv.ellipse(x + dx, y + 1, a, b, lit)
            else:
                cv.ellipse(x + dx, y - 1, a, b, lit)
        for x, y, a, b in blobs:
            cv.ellipse(x + dx, y, a - 1, b - 0.5, mid)
        for x, y, a, b in blobs:
            cv.ellipse(x + dx - 2, y - 1, a - 4, b - 2, base)
    # rim highlights on the lit edge
    a = cv.a
    key = np.all(a == _KEYA, axis=2)
    if lit_from == 'below':
        edge = ~key[:-1] & key[1:]
        a[:-1][edge] = col(rim)
    else:
        edge = ~key[1:] & key[:-1]
        a[1:][edge] = col(rim)
    return to_rgba(cv)


def tile_paste(c, rgba, x, y):
    """Paste a horizontally tiling layer at offset x (wraps)."""
    w = rgba.shape[1]
    x = int(x) % w
    paste(c, rgba, x - w, y)
    paste(c, rgba, x, y)
    if x + w < W:
        paste(c, rgba, x + w, y)


# ================================================================== SHOT A: VANHOOVER
YW_A = 118                    # waterline of the bay (city base 117)
WA = 400                      # city layer width (parallax 0.6 of the camera)
HERO_X = 204                  # hero tower centre in city coordinates (screen x at f 72)

G_RIM, G_HI, G_MID, G_SH, G_DK, G_OUT = '#c6d8f0', '#a2b8da', '#7f95bd', '#62749e', '#4a587d', '#323d5e'
W_OFF, W_OFF2 = '#34416a', '#3d4b76'
W_WARM, W_WARM2, W_COOL, W_TEAL = '#ffd98a', '#ffc46a', '#d4ecff', '#9be8e4'

# (x, w, top, crown, lit fraction)
TOWERS_BACK = [(116, 14, 80), (140, 12, 72), (158, 14, 64), (188, 10, 62), (222, 12, 58), (238, 12, 70),
               (255, 10, 84)]
TOWERS = [
    (121, 12, 86, 'round', 0.45), (133, 14, 70, 'cone', 0.5), (149, 15, 60, 'step', 0.55),
    (166, 12, 78, 'dome', 0.5), (177, 15, 54, 'sail', 0.6), (195, 18, 38, 'hero', 0.62),
    (215, 14, 56, 'cone', 0.55), (231, 15, 68, 'dome', 0.5), (247, 12, 80, 'step', 0.45),
]


def _win_color(seed, frac):
    h = hash01(*seed)
    if h > frac:
        return W_OFF if hash01(*seed, 9) > 0.4 else W_OFF2
    g = hash01(*seed, 4)
    if g < 0.55:
        return W_WARM
    if g < 0.75:
        return W_WARM2
    if g < 0.92:
        return W_COOL
    return W_TEAL


def glass_tower(cv, x, w, top, base, crown, frac, seed, pal=None, toggles=None):
    rim, hi, mid, sh, dk, out = pal or (G_RIM, G_HI, G_MID, G_SH, G_DK, G_OUT)
    body_top = top
    # ---- crown
    if crown == 'dome':
        r = w // 2
        cv.ellipse(x + w / 2 - 0.5, top + 1, w / 2 - 0.5, r * 0.8, out)
        cv.ellipse(x + w / 2 - 0.5, top + 1, w / 2 - 1.5, r * 0.8 - 1, mid)
        cv.ellipse(x + w / 2 - 1.5, top, w / 2 - 3, r * 0.5, hi)
        cv.px(x + w // 2 - 2, top - int(r * 0.5) + 1, rim)
        cv.vline(x + w // 2, top - int(r * 0.8) - 3, top - int(r * 0.8), out)
        cv.px(x + w // 2, top - int(r * 0.8) - 4, W_WARM)
    elif crown == 'cone':
        hgt = int(w * 1.0)
        cv.tri((x - 1, top + 1), (x + w, top + 1), (x + w / 2 - 0.5, top - hgt), out)
        cv.tri((x, top + 1), (x + w - 1, top + 1), (x + w / 2 - 0.5, top - hgt + 1), sh)
        cv.tri((x, top + 1), (x + w / 2 - 0.5, top + 1), (x + w / 2 - 0.5, top - hgt + 1), hi)
        for k in range(2, hgt, 3):            # glass ribs on the cone
            yy = top - hgt + 1 + k
            span = int((w / 2) * k / hgt)
            cv.px(x + w // 2 - span // 2, yy, rim)
        cv.px(x + w // 2, top - hgt - 1, W_TEAL)
    elif crown == 'step':
        cv.rect(x + 3, top - 8, w - 6, 9, out)
        cv.rect(x + 4, top - 7, w - 8, 8, mid)
        cv.rect(x + 4, top - 7, 1, 8, rim)
        cv.rect(x + w - 6, top - 7, 2, 8, sh)
        cv.hline(x + 4, x + w - 5, top - 2, W_TEAL)
        cv.rect(x + 5, top - 13, w - 10, 6, out)
        cv.rect(x + 6, top - 12, w - 12, 5, hi)
        cv.hline(x + 6, x + w - 7, top - 10, W_WARM)
        cv.vline(x + w // 2, top - 18, top - 13, out)
    elif crown == 'sail':
        cv.poly([(x - 1, top + 6), (x + w, top - 4), (x + w, top + 8), (x - 1, top + 8)], out)
        cv.poly([(x, top + 6), (x + w - 1, top - 3), (x + w - 1, top + 8), (x, top + 8)], mid)
        cv.line(x, top + 5, x + w - 1, top - 3, rim)
        cv.line(x + 1, top + 7, x + w - 2, top - 1, W_TEAL)
        body_top = top + 8
    elif crown == 'round':
        cv.round_rect(x - 1, top - 3, w + 2, 6, out, r=2)
        cv.round_rect(x, top - 2, w, 5, hi, r=2)
        cv.hline(x + 2, x + w - 3, top - 2, rim)
    elif crown == 'hero':
        # tall slender lantern crown: glowing teal band, glass cap, needle spire
        cv.rect(x + 2, top - 10, w - 4, 11, out)
        cv.rect(x + 3, top - 9, w - 6, 10, mid)
        cv.rect(x + 3, top - 9, 1, 10, rim)
        cv.rect(x + w - 6, top - 9, 3, 10, sh)
        for yy in (top - 7, top - 4, top - 1):
            cv.hline(x + 4, x + w - 5, yy, W_TEAL)
        cv.ellipse(x + w / 2 - 0.5, top - 11, w / 2 - 2.5, 4, out)
        cv.ellipse(x + w / 2 - 0.5, top - 11, w / 2 - 3.5, 3, hi)
        cv.px(x + w // 2 - 2, top - 13, rim)
        cv.vline(x + w // 2, top - 26, top - 14, G_OUT)
        cv.vline(x + w // 2 - 1, top - 20, top - 15, G_HI)
    # ---- body
    cv.rect(x - 1, body_top, w + 2, base - body_top + 1, out)
    cv.rect(x, body_top, w, base - body_top + 1, mid)
    shw = max(2, w * 3 // 10)
    cv.rect(x + w - shw, body_top, shw, base - body_top + 1, sh)
    cv.vline(x, body_top, base, rim)
    cv.vline(x + 1, body_top, base, hi)
    # window grid (cells 3 x 3: 2x2 window + mullion)
    cy0 = body_top + 2
    for row, yy in enumerate(range(cy0, base - 2, 3)):
        for colx, xx in enumerate(range(x + 2, x + w - 2, 3)):
            cc = _win_color((seed, row, colx), frac)
            if xx + 1 >= x + w - shw:
                cc = mix(cc, sh, 0.35)
            cv.rect(xx, yy, 2, 2, cc)
            if toggles is not None and hash01(seed, row, colx, 77) < 0.05:
                toggles.append((xx, yy, seed * 131 + row * 17 + colx))
    # floor bands every 12 px
    for yy in range(body_top + 13, base - 4, 12):
        cv.hline(x, x + w - 1, yy, dk)
        cv.px(x, yy, hi)


@lru_cache(maxsize=None)
def city_a():
    """Vanhoover skyline (city coordinates, WA x 120): returns (rgba, toggle windows, blinkers)."""
    cv = layer(WA, YW_A)
    toggles = []
    blinkers = []
    base = YW_A - 1
    # ---- back row of hazier towers (behind the main towers)
    back = ('#9fb2d2', '#8296bd', '#6c80a8', '#58698f', '#4a587d', '#3a4668')
    for i, (x, w, top) in enumerate(TOWERS_BACK):
        glass_tower(cv, x, w, top, base, 'round' if i % 2 else 'dome', 0.32, 900 + i, back)
    # ---- main towers
    for i, (x, w, top, crown, frac) in enumerate(TOWERS):
        glass_tower(cv, x, w, top, base, crown, frac, 100 + i, toggles=toggles)
    hx = TOWERS[5][0] + TOWERS[5][2] * 0 + TOWERS[5][1] // 2
    blinkers.append((hx, TOWERS[5][2] - 27, 'red'))
    blinkers.append((TOWERS[2][0] + TOWERS[2][1] // 2, TOWERS[2][2] - 19, 'teal'))
    blinkers.append((TOWERS[6][0] + TOWERS[6][1] // 2, TOWERS[6][2] - 16, 'red'))
    # ---- low-rise waterfront row
    rr = rng(31)
    x = 112
    while x < 268:
        w = ri(rr, 7, 12)
        h = ri(rr, 7, 15)
        top = base - h
        body = rr.choice(['#5b6a8e', '#6a6f94', '#56708a', '#6d6488'])
        cv.rect(x - 1, top - 1, w + 2, h + 2, '#2c3452')
        cv.rect(x, top, w, h + 1, body)
        cv.rect(x + w - 3, top, 3, h + 1, shade(body, 0.8))
        cv.hline(x, x + w - 1, top, mix(body, 'white', 0.3))
        if rr.random() < 0.5:                      # little conical roof
            cv.tri((x - 1, top), (x + w, top), (x + w / 2 - 0.5, top - 5), '#3d3a5c')
            cv.tri((x, top), (x + w / 2 - 0.5, top), (x + w / 2 - 0.5, top - 4), '#57527a')
        for yy in range(top + 2, base - 1, 4):
            for xx in range(x + 1, x + w - 1, 3):
                if hash01(x, yy, xx) < 0.6:
                    cv.rect(xx, yy, 1, 2, W_WARM if hash01(xx, yy) < 0.7 else W_COOL)
        x += w + ri(rr, 0, 2)
    # ---- sail-roof pavilion on its pier (white tent peaks, lit inside)
    px0, px1 = 216, 262
    cv.rect(px0 - 2, base - 6, px1 - px0 + 4, 3, '#3b4766')
    cv.hline(px0 - 2, px1 + 1, base - 6, '#8a9bc0')
    for xx in range(px0, px1, 3):
        cv.px(xx, base - 5, W_WARM if xx % 2 else W_WARM2)
    for xx in range(px0, px1 + 1, 5):
        cv.vline(xx, base - 3, base, '#28304a')
    peaks = [(px0 + 4, 13), (px0 + 13, 17), (px0 + 22, 15), (px0 + 31, 18), (px0 + 40, 12)]
    for (cx_, hgt) in peaks:
        cv.tri((cx_ - 7, base - 6), (cx_ + 7, base - 6), (cx_, base - 6 - hgt), '#5a6a8e')
        cv.tri((cx_ - 6, base - 6), (cx_ + 6, base - 6), (cx_, base - 5 - hgt), '#e6eef8')
        cv.tri((cx_, base - 6), (cx_ + 6, base - 6), (cx_, base - 5 - hgt), '#b8c7de')
        cv.vline(cx_, base - 7 - hgt, base - 5 - hgt, '#8c9bbb')
        cv.line(cx_ - 4, base - 7, cx_ - 1, base - 3 - hgt, '#fff4d8')
    # ---- the green suspension bridge (left), deck y 104, towers x 38 / 92
    deck = 104
    for tx in (38, 92):
        cv.rect(tx - 1, 78, 4, base - 78 + 1, '#1f3a3c')
        cv.rect(tx, 78, 2, base - 78 + 1, '#3f6f68')
        cv.vline(tx, 78, base, '#5f9488')
        for yy in (82, 92, 101):
            cv.rect(tx - 2, yy, 6, 2, '#2c4f4c')
        cv.px(tx, 77, 'red')
        blinkers.append((tx, 76, 'red'))
    # cables (parabola main span + side spans) with suspenders
    cab = []
    for xx in range(-4, 125):
        if xx < 38:
            t = (xx + 4) / 42.0
            y = deck - 2 + (78 - (deck - 2)) * t ** 1.6
        elif xx <= 93:
            u = (xx - 38) / 55.0
            y = 78 + (deck - 4 - 78) * (1 - (2 * u - 1) ** 2)
        else:
            t = (124 - xx) / 31.0
            y = deck - 2 + (78 - (deck - 2)) * t ** 1.6
        cab.append((xx, int(round(y))))
    for xx, y in cab:
        if xx % 3 == 0 and y < deck - 1:
            cv.vline(xx, y + 1, deck - 1, '#33435e')
    for xx, y in cab:
        cv.px(xx, y, '#5f9488')
        if xx % 4 == 0:
            cv.px(xx, y, W_WARM if xx % 8 else '#fff4d0')
    cv.rect(-4, deck, 128, 3, '#253049')
    cv.hline(-4, 124, deck, '#5d6c92')
    for xx in range(-4, 124, 4):
        cv.px(xx, deck + 1, W_WARM2)
    for xx in (8, 64, 116):                        # deck piers
        cv.rect(xx, deck + 3, 3, base - deck - 2, '#2a3550')
    # ---- pine headland (left) and shore pines
    hill = [(-2, base + 1), (-2, 92), (10, 88), (24, 90), (40, 98), (54, 108), (66, 116), (70, base + 1)]
    cv.poly(hill, '#132826')
    rr = rng(7)
    for i in range(16):
        px = int(rr.uniform(-2, 64))
        gy = int(92 + (px + 2) * 0.36) if px > 20 else 90
        pine(cv, px, gy + ri(rr, 3, 8), ri(rr, 10, 19), '#1e3d3a', '#152d2d', '#2f5a52', '#0d1d20')
    for i in range(10):
        px = int(rr.uniform(262, 300))
        pine(cv, px, base + 1, ri(rr, 7, 14), '#1d3a39', '#15302f', '#2e554f', '#0f2124')
    # ---- coast to the right: hills to Tall Tale
    cv.poly([(262, base + 1), (262, 113), (290, 108), (305, 104), (322, 101), (340, 103), (360, 108),
             (385, 112), (400, 114), (400, base + 1)], '#1b3034')
    cv.poly([(300, base + 1), (306, 110), (320, 106), (334, 105), (350, 108), (366, 113), (372, base + 1)],
            '#20383a')
    # Tall Tale: a little hill town of warm windows and a tall clock tower
    tt = [(308, 108, 5), (314, 106, 4), (320, 104, 6), (329, 103, 5), (336, 105, 4), (343, 107, 5),
          (351, 110, 4), (358, 112, 4)]
    for (hx_, hy, hw) in tt:
        cv.rect(hx_, hy - 3, hw, 4, '#4f566f')
        cv.tri((hx_ - 1, hy - 3), (hx_ + hw, hy - 3), (hx_ + hw / 2 - 0.5, hy - 6), '#6b4a5a')
        cv.px(hx_ + 1, hy - 2, W_WARM)
        if hw > 4:
            cv.px(hx_ + 3, hy - 2, W_WARM2)
    cv.rect(325, 92, 3, 12, '#5a5f7a')                 # the tall tale clock tower
    cv.tri((324, 92), (329, 92), (326.5, 87), '#6b4a5a')
    cv.px(326, 95, W_WARM)
    blinkers.append((326, 86, 'gold'))
    for i in range(7):
        px = int(rr.uniform(296, 372))
        pine(cv, px, base + 1 - int(max(0, 9 - abs(px - 330) * 0.25)), ri(rr, 5, 9),
             '#223f3e', '#1a3333', '#335c56', '#132628')
    # shoreline lights (tiny pier lamps) along the water
    for xx in range(268, 372, 9):
        cv.px(xx, base - 1, W_WARM2)
    return to_rgba(cv), tuple(toggles), tuple(blinkers)


@lru_cache(maxsize=None)
def sky_a():
    cv = Canvas(W, YW_A)
    cv.gradient(0, 0, W, YW_A, SKY_NIGHT, soft=0.7)
    fx.glow(cv, 200, 112, 120, '#3a3a52', 0.55, rings=6)      # city glow on the low cloud
    fx.glow(cv, 60, 110, 60, '#2a3a50', 0.35, rings=4)
    return cv


@lru_cache(maxsize=None)
def range_a():
    """Unicorn Range behind Vanhoover: two teal ridges with faint snow, in rainy haze."""
    w = 360
    cv = layer(w, YW_A)
    pts1 = [(0, YW_A)]
    rr = rng(41)
    peaks = [(10, 76), (48, 64), (80, 72), (118, 58), (150, 70), (186, 62), (222, 74), (258, 66),
             (300, 78), (330, 70), (360, 80)]
    for i, (px, py) in enumerate(peaks):
        pts1.append((px, py))
        if i < len(peaks) - 1:
            nx, ny = peaks[i + 1]
            pts1.append(((px + nx) / 2, max(py, ny) + 10 + ri(rr, 0, 6)))
    pts1.append((w, YW_A))
    cv.poly(pts1, '#28495a')
    # snow caps + lit faces
    for (px, py) in peaks:
        cv.tri((px - 5, py + 6), (px + 5, py + 6), (px, py), '#5d8394')
        cv.tri((px - 3, py + 4), (px, py + 4), (px, py), '#7fa5b2')
        cv.tri((px, py), (px + 6, py + 8), (px + 2, py + 9), '#3a6070')
    for xx in range(0, w, 2):                     # haze dither at the ridge foot
        pass
    cv.dither(0, 96, w, 22, '#30566a', 0.5)
    # nearer ridge
    pts2 = [(0, YW_A), (0, 100), (30, 92), (60, 98), (96, 88), (130, 96), (170, 90), (210, 98),
            (250, 92), (290, 100), (330, 94), (360, 99), (360, YW_A)]
    cv.poly(pts2, '#1d3846')
    for i in range(0, w, 5):
        h0 = 92 + int(6 * math.sin(i * 0.07))
        cv.vline(i, h0 + 4, h0 + 6, '#24434f')
    return to_rgba(cv)


def ferry_sprite(size):
    """White harbour ferry (side view), facing left. size 'l' (46 px) or 's' (22 px)."""
    return _ferry(size)


@lru_cache(maxsize=None)
def _ferry(size):
    if size == 'l':
        w, h = 48, 16
        cv = layer(w, h)
        # hull
        cv.poly([(1, 10), (47, 10), (44, 15), (4, 15)], '#2c3a5c')
        cv.poly([(2, 10), (46, 10), (44, 14), (4, 14)], '#cbd8e8')
        cv.hline(3, 45, 12, '#2f4f8a')
        cv.hline(4, 44, 13, '#95a6c2')
        cv.hline(2, 46, 10, '#eef4fb')
        # car deck band with lit openings
        cv.rect(5, 7, 38, 3, '#9fb0c8')
        for xx in range(7, 42, 3):
            cv.rect(xx, 8, 2, 1, W_WARM2)
        # passenger deck with long window row
        cv.rect(8, 3, 31, 4, '#dfe8f3')
        cv.rect(36, 3, 3, 4, '#aebfd6')
        for xx in range(9, 37, 2):
            cv.px(xx, 4, W_WARM if (xx // 2) % 5 else W_COOL)
            cv.px(xx, 5, W_WARM)
        cv.hline(8, 38, 3, '#ffffff')
        # bridge + funnel + mast
        cv.rect(14, 1, 10, 2, '#e8eef6')
        for xx in range(15, 23, 2):
            cv.px(xx, 1, W_COOL)
        cv.rect(28, 0, 4, 3, '#5aa8b8')
        cv.hline(28, 31, 1, '#2a5b8a')
        cv.vline(12, 0, 2, '#7a88a8')
        cv.px(4, 9, 'red')
        cv.px(44, 9, '#5be08a')
    else:
        w, h = 22, 8
        cv = layer(w, h)
        cv.poly([(0, 5), (22, 5), (20, 8), (2, 8)], '#2c3a5c')
        cv.poly([(1, 5), (21, 5), (20, 7), (2, 7)], '#a9b8cf')
        cv.rect(4, 2, 14, 3, '#c3cfe0')
        for xx in range(5, 17, 2):
            cv.px(xx, 3, W_WARM)
        cv.rect(12, 0, 2, 2, '#5aa8b8')
        cv.px(1, 4, 'red')
    return to_rgba(cv)


def draw_ferry(c, x, yb, f, size, flip=False, wake_dir=1):
    """Ferry with reflection and wake. (x, yb) = left of hull at the waterline row."""
    spr = _ferry(size)
    h, w = spr.shape[:2]
    if flip:
        spr = spr[:, ::-1]
    x, yb = int(x), int(yb)
    top = yb - h + (2 if size == 'l' else 1)
    # reflection: flipped, darkened, rippled rows
    refl = spr[::-1].copy()
    refl[..., :3] = (refl[..., :3].astype(np.float32) * np.array([0.45, 0.5, 0.62]) + np.array([6, 12, 22])).astype(np.uint8)
    rows = refl.shape[0]
    for r in range(rows):
        if (r + f // 4) % 3 == 2:
            continue
        dx = int(round(math.sin(r * 1.1 + f * 0.2) * (0.5 + r * 0.12)))
        row = refl[r:r + 1]
        lit = row[..., :3].mean(axis=2) > 110
        row = row.copy()
        row[..., 3] = np.where((bayer(1, w, x, yb + r) < 0.8 - r * 0.04) | lit, row[..., 3], 0)
        c.blit(row, x + dx, yb + 1 + r)
    c.blit(spr, x, top)
    # wake: foam trailing behind (wake_dir=+1 -> wake to the right of the ferry)
    tail = x + w if wake_dir > 0 else x
    L = 40 if size == 'l' else 18
    for i in range(L):
        xx = tail + wake_dir * i
        spread = i // 7
        fade = 1 - i / L
        for sgn in (-1, 1):
            yy = yb + (1 if size == 'l' else 0) + sgn * spread // 2 + (1 if sgn > 0 else 0)
            if hash01(i, sgn, (f + i) // 3) < fade * 0.8:
                c.px(xx, yy, '#9fb6cf' if fade > 0.5 else '#6f87a6')
    # bow wave
    bow = x - 1 if wake_dir > 0 else x + w
    if (f // 3) % 2:
        c.px(bow, yb, '#c8d8ea')
    c.px(bow - wake_dir, yb + 1, '#8fa6c2')


A_LAMPS = (36, 278, 404)      # foreground lamp posts (fg coordinates, parallax 1.6)
FGW_A = 480


@lru_cache(maxsize=None)
def seawall_a():
    """Foreground promenade: railing, wet stone coping, paving, lamp posts, planters (RGBA)."""
    cv = layer(FGW_A, H)
    # coping at the water's edge
    cv.rect(0, 151, FGW_A, 5, '#3a4562')
    cv.hline(0, FGW_A - 1, 151, '#7b8bb2')
    cv.hline(0, FGW_A - 1, 152, '#56638a')
    cv.hline(0, FGW_A - 1, 155, '#262d45')
    for xx in range(0, FGW_A, 11):
        cv.vline(xx, 153, 155, '#2c3450')
    # paving: wet flagstones
    cv.gradient(0, 156, FGW_A, 24, ['#272e48', '#20263d', '#1a1f33'], soft=0.6)
    for r, yy in enumerate(range(160, H, 6)):
        cv.hline(0, FGW_A - 1, yy, '#161a2b')
        for xx in range((r * 7) % 14, FGW_A, 14):
            cv.vline(xx, yy - 5 if yy > 160 else 157, yy, '#161a2b')
            cv.hline(xx + 1, xx + 5, yy - 5 if yy > 160 else 157, '#323a58')
    # railing
    for xx in range(2, FGW_A, 12):
        cv.rect(xx, 140, 2, 12, '#1d2236')
        cv.vline(xx, 140, 151, '#4a557a')
        cv.rect(xx - 1, 139, 4, 1, '#2a3150')
    for yy in (141, 146):
        cv.hline(0, FGW_A - 1, yy, '#232a42')
        cv.hline(0, FGW_A - 1, yy - 1, '#56628a')
    # bollards with mooring chain
    for bx in (120, 196, 340, 452):
        cv.rect(bx, 146, 5, 6, '#1d2236')
        cv.rect(bx + 1, 145, 3, 1, '#58648a')
        cv.vline(bx + 1, 146, 150, '#45507a')
    # planters with flowers
    for px_ in (92, 224, 372):
        cv.rect(px_, 163, 18, 9, '#3c2f3e')
        cv.rect(px_ + 1, 164, 16, 7, '#5a4255')
        cv.hline(px_ + 1, px_ + 16, 164, '#7d5a70')
        cv.rect(px_ + 2, 172, 14, 1, '#141827')
        for k in range(8):
            fx_ = px_ + 2 + k * 2
            cv.vline(fx_, 159 - (k % 3), 163, '#2f5a46')
            cv.px(fx_, 158 - (k % 3), ['#ff9ec8', '#ffd4e6', '#c9a8ff', '#fff2c8'][k % 4])
            cv.px(fx_ + 1, 160 - (k % 2), '#3f7a5a')
    # bench
    bx = 310
    cv.rect(bx, 163, 22, 2, '#5a3f3a')
    cv.hline(bx, bx + 21, 163, '#8a6450')
    cv.rect(bx, 158, 22, 2, '#5a3f3a')
    cv.hline(bx, bx + 21, 158, '#8a6450')
    for lx in (bx + 2, bx + 18):
        cv.rect(lx, 165, 2, 6, '#1d2236')
        cv.rect(lx, 157, 1, 8, '#1d2236')
    # signpost with a ferry icon
    sx = 176
    cv.rect(sx, 150, 2, 22, '#1d2236')
    cv.vline(sx, 150, 171, '#4a557a')
    cv.round_rect(sx - 7, 142, 16, 10, '#2f6f8a', r=1)
    cv.rect(sx - 6, 143, 14, 8, '#e8f1f8')
    cv.poly([(sx - 4, 147), (sx + 6, 147), (sx + 5, 149), (sx - 3, 149)], '#2f4f8a')
    cv.rect(sx - 1, 145, 4, 2, '#2f4f8a')
    cv.px(sx + 3, 144, '#2f4f8a')
    # lamp posts
    for lx in A_LAMPS:
        cv.rect(lx - 3, 168, 7, 4, '#1a1f33')
        cv.rect(lx - 2, 165, 5, 3, '#252c46')
        cv.rect(lx - 1, 112, 3, 54, '#1d2236')
        cv.vline(lx - 1, 112, 166, '#4a557a')
        cv.hline(lx - 3, lx + 3, 128, '#2a3150')
        cv.rect(lx - 3, 103, 7, 1, '#2a3150')
        cv.rect(lx - 2, 101, 5, 2, '#3a4466')
        cv.px(lx, 100, '#3a4466')
        cv.rect(lx - 3, 104, 7, 7, '#2a3150')
        cv.rect(lx - 2, 105, 5, 5, '#ffe7a8')
        cv.vline(lx, 105, 109, '#fff6dc')
        cv.rect(lx - 2, 111, 5, 1, '#2a3150')
    return to_rgba(cv)


def cam_a(f):
    return 58 * cruise(seg(f, 76, 179))


def render_a(f):
    cx = cam_a(f)
    c = sky_a().copy()
    c = Canvas(arr=np.concatenate([c.a, np.zeros((H - YW_A, W, 3), np.uint8)], axis=0))
    # rain clouds drifting (two decks)
    tile_paste(c, cloud_deck(480, 40, 3, '#141a30', '#1b2340', '#2a3452', '#3c4766', 46), -f * 0.12 - cx * 0.2, -6)
    tile_paste(c, cloud_deck(420, 30, 5, '#1a2038', '#232c4a', '#343e5e', '#4a5576', 30), -f * 0.25 - cx * 0.3, 22)
    paste(c, range_a(), -int(cx * 0.35) - 10, 0)
    mist(c, 84, 30, f, '#3c5a6e', 0.35, seed=3, level=0.55, speed=0.18, ox=cx * 0.35)
    city, toggles, blinkers = city_a()
    ox = -int(round(cx * 0.6))
    paste(c, city, ox, 0)
    a = c.a
    for (xx, yy, s) in toggles:
        per = 54 + int(hash01(s, 1) * 72)
        on = hash01(s, 2, (f + int(hash01(s, 3) * per)) // per) > 0.45
        c.rect(xx + ox, yy, 2, 2, (W_WARM if hash01(s, 4) < 0.7 else W_COOL) if on else W_OFF)
    for i, (bx, by, bc) in enumerate(blinkers):
        if ((f + i * 11) // 12) % 3 != 2:
            cc = {'red': '#ff5a6a', 'teal': W_TEAL, 'gold': W_WARM}[bc]
            c.px(bx + ox, by, cc)
            if bc == 'red' and ((f + i * 11) // 12) % 3 == 0:
                c.px(bx + ox - 1, by, '#8a3048')
                c.px(bx + ox + 1, by, '#8a3048')
    # crown glow of the hero tower
    hx = TOWERS[5][0] + TOWERS[5][1] // 2 + ox
    fx.glow(c, hx, TOWERS[5][2] - 5, 9, '#3a8a9a', 0.25 + 0.05 * math.sin(f * 0.2), rings=3)
    # balloon drifting over the bay (burner glowing)
    bxp = int(round(62 + (f - 72) * 0.3 - cx * 0.8))
    draw_balloon(c, bxp, 76, f, 's', flame=True)
    fx.glow(c, bxp, 70, 5, '#ffb050', 0.18, rings=2)
    # mist at the far waterline
    mist(c, 100, 20, f, '#4a6a80', 0.28, seed=8, level=0.5, speed=0.35, ox=cx * 0.6)
    # ---- water
    reflect(c, YW_A, 152, f, '#1a2a42', '#0e1729', k0=0.42, k1=0.14, stretch=0.5, wob=2.6,
            bright_boost=1.0)
    mist(c, 112, 16, f, '#3d5872', 0.3, seed=9, level=0.45, speed=-0.3, ox=cx * 0.8)
    # far little ferry (left -> right) and the big ferry (right -> left)
    draw_ferry(c, 30 + (f - 72) * 0.45 - cx * 0.75, 124, f, 's', flip=True, wake_dir=-1)
    draw_ferry(c, 300 - (f - 72) * 0.85 - cx * 1.0, 137, f, 'l')
    rain_rings(c, f, 121, 151, 22, '#5f7a9c', 11, shift=cx)
    # ---- foreground promenade
    fo = -int(round(cx * 1.6))
    paste(c, seawall_a(), fo, 0)
    for i, lx in enumerate(A_LAMPS):
        sx = lx + fo
        if -30 < sx < W + 30:
            fl = 0.42 + 0.04 * math.sin(f * 0.37 + i * 2) + (0.03 if hash01(i, f // 5) > 0.8 else 0)
            fx.glow(c, sx, 107, 22, '#ffc070', fl * 0.5, rings=5)
            # light pool on the wet paving + streak reflection
            fx.glow(c, sx, 170, 16, '#ffb060', 0.22, rings=3, oy=0)
            for yy in range(158, 180, 2):
                w2 = 1 + (yy - 158) // 8
                dx = int(round(math.sin(yy * 0.9 + f * 0.25)))
                c.blend(sx - w2 + dx, yy, 2 * w2 + 1, 1, '#ffd890', 0.35)
            # lamp light reflected in the bay just behind the railing
            for yy in range(136, 151, 2):
                dx = int(round(math.sin(yy * 0.7 + (f // 3) * 0.6) * 2))
                c.blend(sx - 1 + dx, yy, 3, 1, '#ffcf80', 0.3)
    # ---- rain: far fine drops + near longer streaks; glints in the lamp light
    rain_layer(c, f, 70, '#4d6385', 21, slant=1, speed=5, length=2, shift=cx * 0.6)
    rain_layer(c, f, 26, '#8aa6c8', 22, slant=2, speed=8, length=4, shift=cx * 1.6)
    for i, lx in enumerate(A_LAMPS):
        sx = lx + fo
        if -10 < sx < W + 10:
            rain_layer(c, f, 6, '#ffe2a8', 40 + i, slant=2, speed=8, length=3, region=(sx - 14, 94, 28, 60))
    # splashes on the paving
    for i in range(10):
        k = (f + i * 5) // 6
        xx = int(hash01(i, k, 5) * W)
        yy = 158 + int(hash01(i, k, 6) * 20)
        if (f + i * 5) % 6 < 2:
            c.px(xx - 1, yy, '#6d84a8')
            c.px(xx + 1, yy, '#6d84a8')
    # ---- callouts
    hx = TOWERS[5][0] + TOWERS[5][1] // 2 + ox
    callout(c, hx, 22, 'VANHOOVER', f, 90, 162, side='right', length=16)
    callout(c, 326 + ox, 100, 'TALL TALE', f, 99, 171, side='up', length=14)
    return c


# ================================================================== SHOT B: SMOKEY MOUNTAINS
WB = 420
PK_Y = 8              # peaks layer y offset on screen
FOREST, FOREST_DK, FOREST_HI, FOREST_OUT = '#183a31', '#112a26', '#2c5a48', '#0b1c1b'
LOG, LOG_HI, LOG_SH, LOG_DK = '#6e4a35', '#94664a', '#4e3426', '#33221b'
ROOF, ROOF_HI, ROOF_SH = '#6b2f3a', '#93464c', '#4a2030'
B_SMOKE = []          # (x, y, seed) chimney tops in mid-layer coordinates (filled by peaks_b)


def smoke(c, x, y, f, seed=0, color='#6a7890', n=5, height=26, wind=0.45, ox=0):
    """Curling chimney smoke: puffs rise, grow, drift with the wind and thin out."""
    period = 72
    for i in range(n):
        t = ((f + i * period / n + seed * 5) % period) / period
        py = y - t * height
        px = x + t * height * wind + math.sin(t * 4.5 + seed) * (0.5 + 3.0 * t) + ox
        r = 0.7 + 2.4 * t
        cc = mix(color, '#232c44', t * 0.75)
        if t < 0.9:
            c.ellipse(px, py, r, max(0.5, r * 0.75), cc)


def log_cabin(cv, x, by, w, h, roof='#4b3a44', roof_hi='#6b5562', chimney_side=1, seed=0):
    """Rustic log cabin, base-left (x, by). Returns chimney top (x, y) and window centre."""
    top = by - h
    cv.rect(x - 1, top - 1, w + 2, h + 2, LOG_DK)
    for r, yy in enumerate(range(top, by + 1)):
        cv.hline(x, x + w - 1, yy, LOG if r % 2 == 0 else LOG_SH)
        cv.px(x - 1, yy, LOG_HI if r % 2 == 0 else LOG_DK)       # log ends
        cv.px(x + w, yy, LOG_HI if r % 2 == 0 else LOG_DK)
    cv.hline(x, x + w - 1, top, LOG_HI)
    # roof (gable seen from the long side: trapezoid with overhang)
    rh = max(4, h * 2 // 3)
    cv.poly([(x - 3, top + 1), (x + w + 2, top + 1), (x + w - 1, top - rh), (x, top - rh)], shade(roof, 0.6))
    cv.poly([(x - 2, top), (x + w + 1, top), (x + w - 1, top - rh + 1), (x + 1, top - rh + 1)], roof)
    for yy in range(top - rh + 2, top, 2):
        cv.hline(x, x + w - 1, yy, shade(roof, 0.82))
    cv.hline(x + 1, x + w - 2, top - rh + 1, roof_hi)
    # chimney
    chx = x + w - 4 if chimney_side > 0 else x + 2
    cv.rect(chx, top - rh - 3, 3, rh + 2, '#5c5f72')
    cv.px(chx, top - rh - 3, '#8a8ea4')
    cv.px(chx + 1, top - rh - 1, '#43465a')
    # window + door
    wx = x + 2 if chimney_side > 0 else x + w - 6
    cv.rect(wx - 1, top + 2, 5, 4, LOG_DK)
    cv.rect(wx, top + 3, 3, 2, W_WARM)
    cv.px(wx + 1, top + 3, '#fff2c8')
    cv.px(wx + 1, top + 4, W_WARM2)
    dx = x + w // 2 if chimney_side > 0 else x + 2
    cv.rect(dx, by - 4, 3, 5, '#2a1a16')
    cv.px(dx + 1, by - 3, '#6a4a30')
    return (chx + 1, top - rh - 4), (wx + 1, top + 3)


@lru_cache(maxsize=None)
def far_b():
    """Unicorn Range: far teal silhouettes (two ridges) with faint snow; RGBA (WB x H)."""
    cv = layer(WB, 140)
    rr = rng(77)
    peaks = [(-10, 74), (30, 52), (70, 66), (104, 46), (146, 60), (190, 42), (232, 58), (270, 48),
             (316, 62), (356, 50), (400, 64), (430, 56)]
    pts = [(-10, 140)]
    for i, (px, py) in enumerate(peaks):
        pts.append((px, py))
        if i < len(peaks) - 1:
            nx, ny = peaks[i + 1]
            pts.append(((px + nx) / 2, max(py, ny) + 12 + ri(rr, 0, 8)))
    pts.append((430, 140))
    cv.poly(pts, '#2c5868')
    for (px, py) in peaks:
        cv.tri((px - 6, py + 8), (px + 6, py + 8), (px, py), '#5b8696')
        cv.tri((px - 4, py + 6), (px, py + 6), (px, py), '#86aebb')
        cv.tri((px, py), (px + 8, py + 11), (px + 3, py + 12), '#3b6676')
    cv.dither(0, 84, WB, 30, '#356272', 0.45)
    pts2 = [(-10, 140), (-10, 92), (40, 80), (80, 90), (124, 74), (170, 86), (214, 76), (258, 88),
            (300, 78), (350, 90), (400, 80), (430, 88), (430, 140)]
    cv.poly(pts2, '#21475a')
    for (px, py) in pts2[2:-2]:
        cv.tri((px - 5, py + 6), (px + 5, py + 6), (px, py), '#2f5c6c')
    cv.dither(0, 98, WB, 40, '#28505f', 0.4)
    return to_rgba(cv)


def _forest_fill(cv, poly, seed, n, ylim, hrange=(5, 9)):
    """Fill a slope polygon with dark forest and a scatter of little pine silhouettes."""
    cv.poly(poly, FOREST_DK)
    xs = [p[0] for p in poly]
    rr = rng(seed)
    a = cv.a
    pts = []
    for i in range(n):
        x = int(rr.uniform(min(xs), max(xs)))
        y = int(rr.uniform(ylim[0], ylim[1]))
        if 0 <= x < cv.w and 0 <= y < cv.h and not np.all(a[y, x] == _KEYA):
            pts.append((y, x))
    for (y, x) in sorted(pts):
        h = ri(rr, *hrange)
        pine(cv, x, y + 1, h, FOREST, FOREST_DK, FOREST_HI, FOREST_OUT)


@lru_cache(maxsize=None)
def peaks_b():
    """The twin peaks: fortress (left) and log cabins (right), forested; RGBA + chimneys, lights."""
    cv = layer(WB, 160)
    smokes, lights, flags = [], [], []
    # ---- left peak (fortress)
    # rocky crown under the fortress (drawn first; the forest climbs over its foot)
    cv.poly([(44, 57), (148, 57), (156, 66), (150, 76), (130, 80), (112, 84), (92, 80), (72, 84),
             (54, 78), (38, 70)], '#2c3342')
    cv.poly([(46, 57), (146, 57), (150, 62), (132, 64), (112, 68), (92, 64), (72, 68), (56, 64),
             (44, 62)], '#3f4760')
    for (x0, y0) in [(52, 60), (78, 62), (100, 61), (122, 60), (140, 61)]:
        cv.line(x0, y0, x0 + 3, y0 + 5, '#232836')
        cv.hline(x0 - 3, x0, y0 - 1, '#5d6684')
    lp = [(-12, 160), (-12, 108), (8, 96), (24, 84), (38, 72), (52, 70), (70, 74), (92, 70),
          (112, 74), (134, 70), (150, 72), (162, 86), (176, 104), (192, 120), (206, 134), (218, 160)]
    _forest_fill(cv, lp, 5, 300, (66, 160))
    # ---- right peak (cabins)
    rp = [(172, 160), (190, 140), (212, 120), (234, 102), (256, 84), (276, 70), (294, 64), (310, 63),
          (328, 66), (348, 74), (370, 86), (400, 96), (424, 100), (424, 160)]
    _forest_fill(cv, rp, 6, 260, (63, 160))
    # clearings on the right peak for the cabins (meadow terraces)
    for (x0, y0, w) in [(276, 70, 34), (246, 92, 30), (306, 84, 30), (272, 108, 34), (326, 100, 28)]:
        cv.rect(x0 + 2, y0 + 1, w - 4, 2, '#1f3a2e')
        cv.hline(x0 + 3, x0 + w - 4, y0, '#355e45')
    # ---- fortress (left summit plateau y 54, x 58..134)
    fy = 57
    cv.rect(48, fy, 98, 2, '#5a6070')                       # stone footing
    cv.hline(48, 145, fy, '#7a8098')
    # main lodge behind the palisade
    lx0, lx1 = 72, 118
    cv.rect(lx0 - 1, 33, lx1 - lx0 + 2, fy - 33, LOG_DK)
    for r, yy in enumerate(range(34, fy)):
        cv.hline(lx0, lx1 - 1, yy, LOG if r % 2 == 0 else LOG_SH)
    cv.hline(lx0, lx1 - 1, 41, LOG_DK)
    for wx in range(lx0 + 3, lx1 - 3, 7):                    # two storeys of lit windows
        for wy in (35, 43):
            cv.rect(wx - 1, wy - 1, 5, 5, LOG_DK)
            lit = hash01(wx, wy) < 0.8
            cv.rect(wx, wy, 3, 3, W_WARM if lit else '#2a2030')
            if lit:
                cv.px(wx + 1, wy, '#fff2c8')
                cv.px(wx + 1, wy + 1, LOG_DK)
                lights.append((wx + 1, wy + 1))
    # big gable roof
    cv.poly([(lx0 - 5, 34), (lx1 + 4, 34), (lx1 - 6, 18), (lx0 + 6, 18)], ROOF_SH)
    cv.poly([(lx0 - 4, 33), (lx1 + 3, 33), (lx1 - 6, 19), (lx0 + 6, 19)], ROOF)
    for yy in range(21, 33, 2):
        cv.hline(lx0 - 2, lx1 + 1, yy, shade(ROOF, 0.85))
    cv.hline(lx0 + 6, lx1 - 6, 19, ROOF_HI)
    cv.line(lx0 - 4, 33, lx0 + 6, 19, ROOF_HI)
    # dormer with a round window
    cv.rect(92, 22, 7, 7, LOG_SH)
    cv.tri((90, 23), (100, 23), (95, 17), ROOF_SH)
    cv.tri((91, 22), (99, 22), (95, 18), ROOF_HI)
    cv.circle(95, 25, 1.5, W_WARM)
    lights.append((95, 25))
    for chx in (79, 109):
        cv.rect(chx, 12, 4, 10, '#55596c')
        cv.hline(chx - 1, chx + 4, 12, '#7a7e94')
        cv.vline(chx + 3, 13, 21, '#3e4152')
        smokes.append((chx + 2, 11))
    flags.append((95, 17, 8))           # lodge flag pole top (x, y, pole height)
    # palisade (pointed logs)
    for x in range(54, 138, 2):
        top = 45 + (1 if (x // 2) % 3 == 1 else 0)
        cv.rect(x, top + 1, 2, fy - top, LOG_SH)
        cv.px(x, top + 1, LOG_HI)
        cv.vline(x, top + 2, fy - 1, LOG)
        cv.px(x, top, LOG_DK)
        cv.px(x + 1, top, LOG_DK)
    cv.hline(54, 137, 49, LOG_DK)
    cv.hline(54, 137, 52, LOG_DK)
    # gate with lanterns
    cv.rect(91, 47, 10, 8, LOG_DK)
    cv.ellipse(95.5, 48, 4, 3, LOG_DK)
    cv.rect(92, 48, 8, 7, '#2a1812')
    cv.vline(95, 46, 54, '#3a2418')
    for lxx in (89, 102):
        cv.rect(lxx, 48, 2, 3, '#ffcf70')
        cv.px(lxx, 47, '#3a2a20')
        lights.append((lxx, 49))
    # watchtowers with conical roofs
    for tx in (52, 134):
        cv.rect(tx - 5, 30, 11, fy - 30, LOG_DK)
        for r, yy in enumerate(range(31, fy)):
            cv.hline(tx - 4, tx + 4, yy, LOG if r % 2 == 0 else LOG_SH)
        cv.hline(tx - 6, tx + 6, 30, LOG_HI)
        cv.rect(tx - 6, 29, 13, 1, LOG_DK)
        for k in range(-6, 7, 2):
            cv.vline(tx + k, 26, 29, LOG_SH)
        cv.hline(tx - 6, tx + 6, 26, LOG)
        cv.tri((tx - 8, 26), (tx + 8, 26), (tx, 13), ROOF_SH)
        cv.tri((tx - 7, 25), (tx + 7, 25), (tx, 14), ROOF)
        cv.tri((tx - 7, 25), (tx, 25), (tx, 14), ROOF_HI)
        cv.rect(tx - 1, 37, 3, 4, '#2a1a16')
        cv.rect(tx - 1, 38, 3, 2, W_WARM)
        lights.append((tx, 38))
        flags.append((tx, 13, 6))
    # ---- cabins on the right peak
    cabins = [(280, 70, 15, 8, 1), (306, 84, 13, 7, -1), (248, 92, 14, 8, 1), (276, 108, 15, 8, -1),
              (328, 100, 12, 7, 1)]
    for i, (x, by, w, h, side) in enumerate(cabins):
        roof = ['#5a3a3e', '#3f4a3a', '#5a4436', '#4a3a4e', '#3f4a3a'][i]
        ch, win = log_cabin(cv, x, by, w, h, roof, shade(roof, 1.35), side, seed=i)
        smokes.append(ch)
        lights.append(win)
        # porch lantern
        lx = x + w + 1 if side > 0 else x - 2
        cv.px(lx, by - 4, '#ffcf70')
        lights.append((lx, by - 4))
        # little fence
        for k in range(0, 10, 2):
            cv.vline(x - 10 + k if side > 0 else x + w + 2 + k, by - 1, by, '#5a4232')
        cv.hline(x - 10 if side > 0 else x + w + 2, (x - 2) if side > 0 else x + w + 10, by - 1, '#7a5a40')
    # zig-zag lantern path between cabins
    path = [(242, 128), (264, 112), (276, 108), (294, 96), (306, 84), (298, 76), (288, 70)]
    for (a0, b0), (a1, b1) in zip(path, path[1:]):
        cv.line(a0, b0 + 1, a1, b1 + 1, '#3a5a46')
    # stumps on the lower slopes near the valley (old logging)
    rr = rng(19)
    for i in range(14):
        sx = int(rr.uniform(140, 240))
        ytop = 160
        for yy in range(40, 160):
            if not np.all(cv.a[yy, sx] == _KEYA):
                ytop = yy
                break
        sy = ytop + ri(rr, 4, 20)
        cv.rect(sx, sy, 3, 2, '#5a3e2e')
        cv.hline(sx, sx + 2, sy, '#b08a64')
    return to_rgba(cv), tuple(smokes), tuple(lights), tuple(flags)


@lru_cache(maxsize=None)
def valley_b():
    """Valley floor: meadow with the creek, footbridge and the replanted sapling rows (RGBA)."""
    cv = layer(WB, H)
    cv.poly([(-10, H), (-10, 150), (60, 142), (130, 134), (180, 128), (220, 126), (260, 128),
             (320, 136), (380, 144), (430, 150), (430, H)], '#1e3a2c')
    cv.gradient(-10, 126, 440, H - 126, ['#24432f', '#1d3a2a', '#16301f'], soft=0.6)
    # re-cut the top silhouette (gradient filled a rect): restore transparency above the meadow line
    a = cv.a
    for x in range(WB):
        u = x / WB
        ytop = int(128 + 22 * (2 * u - 0.95) ** 2 * 0.9)
        a[:max(0, ytop), x] = _KEYA
        cv.px(x, ytop, '#3a6a48')
    # creek (from the far valley down to the front)
    pts_l, pts_r = [], []
    for yy in range(128, H + 1):
        t = (yy - 128) / (H - 128)
        cxw = 214 + math.sin(t * 4.2) * 18 * t + t * 30
        hw = 1 + t * 9
        pts_l.append((cxw - hw, yy))
        pts_r.append((cxw + hw, yy))
    cv.poly(pts_l + pts_r[::-1], '#1a2c46')
    for (x0, y0), (x1, y1) in zip(pts_l, pts_r):
        cv.px(x0, y0, '#4a6a5a')
        cv.px(x1, y1, '#2a4a3a')
    # footbridge across the creek at y 158
    bt = (158 - 128) / (H - 128)
    bcx = 214 + math.sin(bt * 4.2) * 18 * bt + bt * 30
    bx0, bx1 = int(bcx - 13), int(bcx + 13)
    for x in range(bx0, bx1 + 1):
        u = (x - bx0) / (bx1 - bx0)
        yy = int(round(158 - 4 * math.sin(u * math.pi)))
        cv.vline(x, yy, yy + 1, '#5a3e2e')
        cv.px(x, yy, '#94664a')
        if (x - bx0) % 4 == 0:
            cv.vline(x, yy - 4, yy - 1, '#4e3426')
        cv.px(x, yy - 4, '#7a5a40')
    for x in (bx0, bx1):
        cv.rect(x - 1, 157, 2, 6, '#3a281e')
    return to_rgba(cv), (bx0, bx1)


# saplings: (x, y, size) in valley coordinates, rows across the valley floor
@lru_cache(maxsize=None)
def saplings_b():
    out = []
    rows = [(132, 9, 1), (138, 10, 1), (145, 11, 2), (153, 12, 2), (162, 14, 3), (172, 16, 3)]
    for r, (y, sp, sz) in enumerate(rows):
        x = 100 + (r * 5) % sp
        while x < 340:
            t = (y - 128) / (H - 128)
            ccx = 214 + math.sin(t * 4.2) * 18 * t + t * 30
            if abs(x - ccx) > 4 + t * 12 and hash01(x, y) > 0.12:
                out.append((x, y, sz, hash01(x, y, 3)))
            x += sp
    return tuple(out)


def draw_sapling(c, x, y, sz, ph, f):
    sway = 1 if math.sin(f * 0.11 + ph * 6 + x * 0.05) > 0.55 else 0
    if hash01(int(ph * 1000)) < 0.15:            # an old stump instead
        c.rect(x - 1, y - 1, 3, 2, '#5a3e2e')
        c.hline(x - 1, x + 1, y - 1, '#b08a64')
        return
    th = 2 + sz
    c.vline(x, y - th, y, '#5a4232')
    if sz >= 2:
        c.vline(x + 2, y - th + 1, y, '#6a5040')                # support stake
    r = sz
    cx, cy = x + sway, y - th - r
    if r <= 1:
        c.rect(cx - 1, cy - 1, 3, 2, '#3e7a4a')
        c.px(cx - 1, cy - 1, '#5fa060')
    else:
        c.circle(cx, cy, r, '#2c5a3a')
        c.circle(cx - 0.5, cy - 0.5, r - 0.6, '#3e7a4a')
        c.px(cx - r // 2, cy - r // 2, '#6ab06a')
    c.px(x, y + 1, '#16301f')


@lru_cache(maxsize=None)
def fg_b():
    """Foreground framing: big dark pines, ferns, rocks, a rail fence and a sapling signpost."""
    w = 520
    cv = layer(w, H)
    rr = rng(55)
    # ground hump at the very bottom
    cv.poly([(-10, H), (-10, 168), (40, 164), (90, 170), (140, 172), (200, 174), (260, 172), (330, 168),
             (400, 166), (470, 170), (530, 168), (530, H)], '#0f2018')
    for x in range(-10, 530, 3):
        y0 = 168 + int(3 * math.sin(x * 0.05))
        cv.vline(x, y0 - ri(rr, 1, 4), y0, '#1a3a26')
    # rail fence
    for x in range(60, 250, 16):
        cv.rect(x, 160, 2, 12, '#3a281e')
        cv.px(x, 160, '#7a5a40')
    for yy in (162, 166):
        cv.hline(60, 236, yy, '#4e3426')
        cv.hline(60, 236, yy - 1, '#7a5a40')
    # rocks
    for (x, y, rw) in [(30, 170, 9), (150, 174, 7), (300, 170, 10), (440, 172, 8)]:
        cv.ellipse(x, y, rw, 4, '#2a3040')
        cv.ellipse(x - 1, y - 1, rw - 2, 2.5, '#3c4458')
        cv.hline(x - rw + 3, x, y - 3, '#5a6480')
    # ferns
    for i in range(18):
        fxx = int(rr.uniform(-5, 520))
        fy = 172 + ri(rr, -2, 3)
        for k in range(-3, 4):
            cv.line(fxx, fy, fxx + k * 2, fy - 5 + abs(k), '#24502f' if k % 2 else '#2f6a3a')
    # signpost with a sapling icon
    sx = 268
    cv.rect(sx, 150, 2, 22, '#3a281e')
    cv.vline(sx, 150, 171, '#7a5a40')
    cv.round_rect(sx - 7, 141, 16, 11, '#4e3426', r=1)
    cv.rect(sx - 6, 142, 14, 9, '#c8a878')
    cv.vline(sx + 1, 145, 149, '#5a3e2e')
    cv.circle(sx + 1, 144, 2, '#3e7a4a')
    cv.px(sx, 143, '#6ab06a')
    cv.hline(sx - 3, sx + 5, 149, '#6a4a30')
    # big framing pines (left edge, right edge, one entering later)
    for (px, h, wdt) in [(4, 190, 1.0), (470, 186, 1.0)]:
        by = 178
        tiers = 9
        for r in range(h):
            yy = by - h + r
            t = r / h
            p = (r % (h / tiers)) / (h / tiers)
            hw = (2 + 30 * t * (0.5 + 0.5 * p)) * wdt
            cv.hline(int(px - hw) - 1, int(px + hw) + 1, yy, FOREST_OUT)
        for r in range(h):
            yy = by - h + r
            t = r / h
            p = (r % (h / tiers)) / (h / tiers)
            hw = (2 + 30 * t * (0.5 + 0.5 * p)) * wdt
            cv.hline(int(px - hw), int(px + hw), yy, '#10251f')
            cv.hline(int(px - hw), int(px - hw) + 1, yy, '#1f3d34')
            if p > 0.75:
                cv.hline(int(px - hw) + 2, int(px + hw) - 2, yy, '#0c1a17')
        cv.rect(px - 2, by - 8, 5, 10, '#21160f')
    return to_rgba(cv)


def cam_b(f):
    return 36 * cruise(seg(f, 180, 269))


def draw_flag(c, x, y, pole, f, color='#c0405a', emblem=W_WARM, seed=0):
    """Pole with a banner waving in the wind (free edge to the right). (x, y) = pole top."""
    c.vline(x, y, y + pole, '#3a281e')
    c.px(x, y - 1, W_WARM)
    for k in range(7):
        off = int(round(math.sin(f * 0.28 - k * 0.9 + seed) * (k / 6.0) * 1.4))
        ln = 5 - (1 if k == 6 else 0)
        cc = color if k < 5 else shade(color, 0.8)
        c.vline(x + 1 + k, y + 1 + off, y + ln + off, cc)
    off = int(round(math.sin(f * 0.28 - 3 * 0.9 + seed) * 0.7))
    c.hline(x + 3, x + 5, y + 2 + off, emblem)      # horseshoe emblem (an arch, open at the bottom)
    c.px(x + 3, y + 3 + off, emblem)
    c.px(x + 5, y + 3 + off, emblem)


def render_b(f):
    cx = cam_b(f)
    c = Canvas(W, H)
    c.gradient(0, 0, W, 130, SKY_NIGHT, soft=0.7)
    moon_t = seg(f, 236, 269)
    if moon_t > 0:            # the moon is coming: cloud deck silvering at the top right
        fx.glow(c, 292, 0, 110, '#8090b8', 0.32 * smooth(moon_t), rings=6)
    tile_paste(c, cloud_deck(480, 40, 13, '#141a30', '#1b2340', '#283250', '#38445f', 46), -f * 0.12 - cx * 0.2, -8)
    if moon_t > 0:
        fx.glow(c, 292, 0, 80, '#a8b4dc', 0.26 * smooth(moon_t), rings=5)
    paste(c, far_b(), -int(cx * 0.3) - 20, 0)
    mist(c, 78, 26, f, '#4a7080', 0.3, seed=21, level=0.5, speed=0.2, ox=cx * 0.3)
    tile_paste(c, cloud_deck(420, 26, 15, '#1d2440', '#252f4c', '#34405e', '#48557a', 24), -f * 0.3 - cx * 0.4, 30)
    pk, smokes, lights, flags = peaks_b()
    po = -int(round(cx * 0.7)) + 6
    paste(c, pk, po, PK_Y)
    for i, (sx, sy) in enumerate(smokes):
        smoke(c, sx + po, sy + PK_Y, f, seed=i * 3 + 1, n=12, height=30 + (i % 3) * 5, color='#8a96ac')
    for i, (lx, ly) in enumerate(lights):
        fl = 0.16 + 0.04 * math.sin(f * 0.3 + i * 1.7)
        fx.glow(c, lx + po, ly + PK_Y, 5, '#ffb060', fl, rings=2)
    for i, (fxx, fyy, ph) in enumerate(flags):
        draw_flag(c, fxx + po, fyy + PK_Y - ph, ph, f, seed=i * 1.3)
    # balloon drifting through the valley
    bx = int(round(186 + (f - 180) * 0.26 - cx * 0.75))
    draw_balloon(c, bx, 104 + int(round(2 * math.sin(f * 0.05))), f, 's', flame=True)
    fx.glow(c, bx, 98, 5, '#ffb050', 0.18, rings=2)
    mist(c, 104, 30, f, '#5a7a88', 0.32, seed=23, level=0.55, speed=0.35, ox=cx * 0.75)
    # valley floor
    vl, (bx0, bx1) = valley_b()
    vo = -int(round(cx * 1.0)) - 14
    paste(c, vl, vo, 0)
    # creek flow highlights + lantern reflections
    for i in range(16):
        t = ((f * 0.9 + i * 13) % 52) / 52
        yy = 128 + t * (H - 128)
        tt = (yy - 128) / (H - 128)
        ccx = 214 + math.sin(tt * 4.2) * 18 * tt + tt * 30
        xx = ccx + (hash01(i, 3) - 0.5) * (2 + tt * 12)
        c.hline(int(xx + vo), int(xx + vo) + (1 if tt < 0.5 else 2), int(yy), '#4a6a90')
    for (x, y, sz, ph) in saplings_b():
        draw_sapling(c, x + vo, y, sz, ph, f)
    # path lanterns along the valley (on posts) with glow
    for i, (lx, ly) in enumerate([(150, 150), (bx0 - 4, 156), (bx1 + 6, 156), (290, 146)]):
        sx = lx + vo
        c.vline(sx, ly - 9, ly, '#3a281e')
        c.rect(sx - 1, ly - 12, 3, 3, '#ffcf70')
        c.px(sx, ly - 13, '#3a281e')
        fx.glow(c, sx, ly - 11, 10, '#ffb060', 0.3 + 0.04 * math.sin(f * 0.4 + i), rings=4)
    mist(c, 126, 22, f, '#5a7a88', 0.22, seed=25, level=0.45, speed=-0.25, ox=cx)
    # foreground framing
    paste(c, fg_b(), -int(round(cx * 1.35)), 0)
    # rain
    rain_layer(c, f, 70, '#4d6385', 31, slant=1, speed=5, length=2, shift=cx * 0.7)
    rain_layer(c, f, 24, '#8aa6c8', 32, slant=2, speed=8, length=4, shift=cx * 1.35)
    # callout: the valley between the twin peaks
    callout(c, 140 + po, 26 + PK_Y, 'SMOKEY MOUNTAINS', f, 186, 258, side='right', length=16)
    return c


# ================================================================== SHOT C: SILVER SHOALS / LUNA BAY
HZ = 96                      # ocean horizon
YW_C = 128                   # waterline of the stilt town / lighthouse rocks
MOON = (254, 40)
MOON_R = 11
ROCK, ROCK_SH, ROCK_HI, ROCK_MOON = '#2c3346', '#20263a', '#414b66', '#8e9cc4'
HOUSE_COLS = [('#5fb3b0', '#3f8a8c'), ('#f29a8e', '#c06a6a'), ('#ffd98a', '#d0a456'),
              ('#b8a2e8', '#8a72c0'), ('#9fe0b8', '#6ab08a'), ('#f6b6d6', '#c8809e')]


def moon_t(f):
    return smooth(seg(f, T_MOON, T_MOON + 24))


@lru_cache(maxsize=None)
def cloud_puff(seed, w, h, base='#1d2443', mid='#2a3456', lit='#4a5882', rim='#8d9cc8'):
    """A single soft rain-cloud puff (RGBA) with a lit underside and a silver lower rim."""
    cv = layer(w, h)
    rr = rng(seed)
    blobs = []
    n = max(4, w // 7)
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi)
        a = rr.uniform(6, 12) + 6 * bump
        b = (h * 0.22 + h * 0.25 * bump) * rr.uniform(0.8, 1.1)
        blobs.append((u * (w - 12) + 6, h - 2 - b, a, b))
    for x, y, a, b in blobs:
        cv.ellipse(x, y + 1, a, b, lit)
    for x, y, a, b in blobs:
        cv.ellipse(x, y, a - 1, b - 0.5, mid)
    for x, y, a, b in blobs:
        cv.ellipse(x - 2, y - 1, a - 4, b - 2, base)
    a_ = cv.a
    key = np.all(a_ == _KEYA, axis=2)
    edge = ~key[:-1] & key[1:]
    a_[:-1][edge] = col(rim)
    return to_rgba(cv)


# (x, y, w, h, seed) cloud puffs over Luna Bay at f 216 (they part from the moon)
C_PUFFS = [(-40, -10, 130, 34, 1), (70, -12, 140, 36, 2), (190, -10, 150, 38, 3),
           (-20, 16, 110, 30, 4), (80, 12, 120, 32, 5), (196, 10, 120, 36, 6), (290, 14, 100, 30, 7),
           (30, 38, 90, 22, 8), (140, 36, 96, 24, 9), (222, 30, 84, 28, 10), (300, 38, 70, 20, 11),
           (100, 52, 70, 16, 12), (250, 52, 60, 14, 13)]


def paste_dither(c, rgba, x, y, level, ox=0):
    """Paste an RGBA layer through an ordered-dither mask covering `level` of its pixels."""
    if level <= 0:
        return
    if level >= 1:
        return paste(c, rgba, x, y)
    h, w = rgba.shape[:2]
    m = (bayer(h, w, ox, 0) < level)
    sub = rgba.copy()
    sub[..., 3] = np.where(m, sub[..., 3], 0)
    paste(c, sub, x, y)


def stilt_house(cv, rim, x, wy, w, h, stilt, colors, roof, seed, lights, flags=None):
    """Pastel house on stilts over the water. (x, wy) = left post at the waterline.

    Draws into cv; the moon-rim highlights (right-hand edges) go into the `rim` layer."""
    body, body_sh = colors
    deck = wy - stilt
    # stilts + cross braces
    for px_ in range(x, x + w + 1, 4):
        cv.vline(px_, deck, wy + 2, '#3a2a24')
        rim.px(px_ + 1 if px_ + 1 <= x + w else px_, deck + 2, '#7a6a64')
    for px_ in range(x, x + w - 3, 4):
        cv.line(px_, deck + 2, px_ + 4, deck + stilt - 2, '#2a1e1a')
    # deck + railing
    cv.rect(x - 2, deck - 1, w + 5, 2, '#5a4436')
    cv.hline(x - 2, x + w + 2, deck - 1, '#8a6a52')
    for px_ in range(x - 2, x + w + 3, 2):
        cv.px(px_, deck - 3, '#5a4436')
    cv.hline(x - 2, x + w + 2, deck - 4, '#6a5040')
    # house body
    top = deck - 1 - h
    hx0, hx1 = x + 1, x + w - 1
    cv.rect(hx0 - 1, top - 1, hx1 - hx0 + 3, h + 1, shade(body_sh, 0.6))
    cv.rect(hx0, top, hx1 - hx0 + 1, h, body)
    cv.rect(hx1 - 2, top, 3, h, body_sh)
    for yy in range(top + 2, deck - 1, 3):                  # clapboard lines
        cv.hline(hx0, hx1, yy, mix(body, body_sh, 0.5))
    rim.vline(hx1 + 1, top, deck - 2, mix(body, '#e8f0ff', 0.6))
    # roof: steep gable (side view), or a conical cap
    rr = int(seed) % 2
    rc, rcs = roof
    if rr == 0:
        rh = (w + 4) // 2 + 2
        cv.tri((hx0 - 3, top + 1), (hx1 + 3, top + 1), ((hx0 + hx1) / 2, top - rh), shade(rcs, 0.7))
        cv.tri((hx0 - 2, top), (hx1 + 2, top), ((hx0 + hx1) / 2, top - rh + 1), rc)
        cv.tri(((hx0 + hx1) / 2, top), (hx1 + 2, top), ((hx0 + hx1) / 2, top - rh + 1), rcs)
        rim.line((hx0 + hx1) / 2 + 1, top - rh + 2, hx1 + 3, top, '#dfe8fb')
        apex = ((hx0 + hx1) // 2, top - rh)
    else:
        rh = w // 2 + 4
        cv.poly([(hx0 - 2, top + 1), (hx1 + 2, top + 1), (hx1, top - 2), ((hx0 + hx1) / 2 + 1, top - rh),
                 ((hx0 + hx1) / 2, top - rh), (hx0, top - 2)], shade(rcs, 0.7))
        cv.poly([(hx0 - 1, top), (hx1 + 1, top), (hx1 - 1, top - 2), ((hx0 + hx1) / 2, top - rh + 1),
                 (hx0 + 1, top - 2)], rc)
        cv.poly([((hx0 + hx1) / 2, top), (hx1 + 1, top), (hx1 - 1, top - 2), ((hx0 + hx1) / 2, top - rh + 1)], rcs)
        rim.line((hx0 + hx1) / 2 + 1, top - rh + 2, hx1 + 1, top - 1, '#dfe8fb')
        apex = ((hx0 + hx1) // 2, top - rh)
    cv.px(apex[0], apex[1] - 1, '#ffd98a')
    # round window (warm) + door
    wx = hx0 + 2
    wy0 = top + max(1, h // 2 - 2)
    cv.circle(wx + 1, wy0 + 1, 1.6, shade(body_sh, 0.55))
    cv.rect(wx, wy0, 3, 3, W_WARM)
    cv.px(wx + 1, wy0, '#fff2c8')
    lights.append((wx + 1, wy0 + 1))
    if w >= 11:
        dx = hx1 - 4
        cv.rect(dx, deck - 6, 3, 5, '#3a2a30')
        cv.px(dx + 1, deck - 5, '#ffcf70')
    # ladder down to the water
    lx = x + w + 2
    cv.vline(lx, deck, wy, '#4e3a30')
    cv.vline(lx + 2, deck, wy, '#4e3a30')
    for yy in range(deck + 1, wy, 2):
        cv.hline(lx, lx + 2, yy, '#6a5040')
    return apex


@lru_cache(maxsize=None)
def far_c():
    """Luna Bay: far snowy headland (left), low far cape (right), horizon haze. RGBA (360 x HZ+1)."""
    w = 360
    cv = layer(w, HZ + 1)
    cv.poly([(-10, HZ + 1), (-10, 66), (12, 62), (30, 66), (44, 60), (60, 68), (84, 74), (110, 84),
             (132, 92), (150, HZ + 1)], '#3a4a6e')
    for (px, py) in [(12, 62), (44, 60)]:
        cv.tri((px - 8, py + 7), (px + 9, py + 7), (px, py), '#a8b8d8')
        cv.tri((px, py), (px + 9, py + 7), (px + 3, py + 8), '#7d8db4')
    cv.dither(-10, 80, 160, 17, '#4a5a80', 0.5)
    cv.poly([(250, HZ + 1), (262, 90), (286, 86), (310, 88), (340, 84), (370, 88), (370, HZ + 1)], '#34436a')
    cv.dither(250, 90, 120, 7, '#42527a', 0.5)
    # tiny far lights (Luna Bay village on the far shore)
    for (lx, ly) in [(64, 79), (72, 81), (79, 83), (96, 88), (300, 89), (314, 88)]:
        cv.px(lx, ly, '#ffd98a')
    return to_rgba(cv)


@lru_cache(maxsize=None)
def town_c():
    """Silver Shoals: rocky hill with stilt houses, boardwalks, stairs + a separate moon-rim layer.

    Returns (rgba, rim_rgba, lights, apexes, lighthouse_lamp)."""
    w = 420
    cv = layer(w, YW_C + 4)
    rim = layer(w, YW_C + 4)
    lights = []
    # ---- the rocky hill (left): cliff faces, moss, pines, terraced houses
    hill = [(-10, YW_C + 2), (-10, 70), (6, 66), (22, 68), (36, 64), (52, 68), (68, 76), (84, 82),
            (100, 90), (114, 100), (126, 110), (134, 120), (140, YW_C + 2)]
    cv.poly(hill, ROCK)
    rr = rng(91)
    # cliff facets: lighter faces on the moon side (right), dark clefts
    for i in range(22):
        x = int(rr.uniform(-6, 118))
        y = int(rr.uniform(70, 106))
        if np.all(cv.a[y, max(0, x)] == _KEYA):
            continue
        fw, fh = ri(rr, 5, 11), ri(rr, 4, 8)
        cv.poly([(x, y + fh), (x + 2, y), (x + fw, y + 1), (x + fw + 1, y + fh)], ROCK_SH)
        cv.poly([(x + 1, y + fh - 1), (x + 3, y + 1), (x + fw - 1, y + 2), (x + fw - 1, y + fh - 1)], ROCK)
        cv.line(x + 3, y + 1, x + fw - 1, y + 2, ROCK_HI)
        rim.px(x + fw - 1, y + 2, ROCK_MOON)
    for i in range(18):                                     # moss / grass tufts
        x = int(rr.uniform(-6, 132))
        y = int(rr.uniform(70, YW_C - 4))
        if np.all(cv.a[y, max(0, x)] == _KEYA):
            continue
        cv.hline(x, x + ri(rr, 2, 5), y, '#2a4a3a')
        cv.px(x + 1, y - 1, '#3a6a4a')
    a_ = cv.a
    for x in range(0, 140):                                  # moonlit ridge line
        col_ = a_[:, x]
        filled = np.nonzero(~np.all(col_ == _KEYA, axis=1))[0]
        if len(filled):
            rim.px(x, filled[0], ROCK_MOON if x % 3 else '#b8c4e4')
            cv.px(x, filled[0], '#3a5048')
    for i in range(12):                                     # pines on the hilltop
        x = int(rr.uniform(-6, 70))
        filled = np.nonzero(~np.all(a_[:, max(0, x)] == _KEYA, axis=1))[0]
        y = filled[0] if len(filled) else 70
        pine(cv, x, y + 3, ri(rr, 7, 13), '#1e3d3a', '#152d2d', '#2f5a52', '#0d1d20')
    # zig-zag stairs with lanterns up the hill
    stairs = [(130, 116), (104, 108), (116, 98), (88, 90), (100, 82), (74, 76)]
    for (a0, b0), (a1, b1) in zip(stairs, stairs[1:]):
        cv.line(a0, b0, a1, b1, '#6a5446')
        cv.line(a0, b0 + 1, a1, b1 + 1, '#3a2c28')
    for (sx, sy) in stairs[1:]:
        cv.vline(sx, sy - 4, sy, '#3a2a24')
        cv.px(sx, sy - 5, '#ffcf70')
        lights.append((sx, sy - 5))
    # houses on the hill terraces (one window each, off-centre; a door on the other side)
    hill_houses = [(12, 68, 11, 8, 0), (40, 66, 10, 7, 3), (60, 78, 11, 8, 1), (84, 86, 10, 7, 2),
                   (104, 98, 10, 7, 5), (24, 84, 10, 7, 4), (48, 96, 11, 8, 2), (70, 104, 10, 7, 0)]
    for i, (x, by, hw, hh, ci) in enumerate(hill_houses):
        body, sh = HOUSE_COLS[ci]
        cv.rect(x - 2, by, hw + 4, 2, ROCK_SH)              # little stone terrace
        cv.hline(x - 2, x + hw + 1, by, ROCK_HI)
        cv.rect(x - 1, by - hh - 1, hw + 2, hh + 2, shade(sh, 0.55))
        cv.rect(x, by - hh, hw, hh, body)
        cv.rect(x + hw - 2, by - hh, 2, hh, sh)
        rh = hw // 2 + 3
        cv.tri((x - 2, by - hh), (x + hw + 1, by - hh), (x + hw / 2 - 0.5, by - hh - rh), '#3a3450')
        cv.tri((x + hw / 2 - 0.5, by - hh), (x + hw + 1, by - hh), (x + hw / 2 - 0.5, by - hh - rh), '#2c2840')
        rim.line(x + hw / 2 + 0.5, by - hh - rh + 1, x + hw + 1, by - hh, '#c8d4f0')
        rim.vline(x + hw, by - hh, by - 1, mix(body, '#e8f0ff', 0.6))
        wx = x + 1 if i % 2 == 0 else x + hw - 4
        cv.rect(wx, by - hh + 2, 2, 2, W_WARM)
        lights.append((wx, by - hh + 2))
        dx = x + hw - 4 if i % 2 == 0 else x + 1
        cv.rect(dx, by - 4, 2, 4, '#3a2a30')
    # ---- stilt houses over the water + boardwalk
    apexes = []
    stilts = [(120, 14, 11, 12, 0, ('#d86a7a', '#a84a5a')),
              (142, 13, 10, 16, 2, ('#4a8ab8', '#2f6a94')),
              (162, 16, 12, 10, 3, ('#e0a050', '#b07830')),
              (186, 12, 10, 14, 4, ('#7a5ab0', '#5a3e8a')),
              (204, 13, 9, 11, 1, ('#d86a7a', '#a84a5a'))]
    for i, (x, w2, h2, st, ci, roof) in enumerate(stilts):
        apexes.append(stilt_house(cv, rim, x, YW_C, w2, h2, st, HOUSE_COLS[ci], roof, i, lights))
    # boardwalk joining the decks, out to a little landing (balloon mooring) at x 222..250
    cv.rect(118, YW_C - 9, 122, 2, '#5a4436')
    cv.hline(118, 239, YW_C - 9, '#8a6a52')
    rim.hline(118, 239, YW_C - 9, '#b8a890')
    for px_ in range(118, 240, 5):
        cv.vline(px_, YW_C - 7, YW_C + 2, '#3a2a24')
    cv.rect(216, YW_C - 11, 24, 2, '#6a5040')                 # landing
    cv.hline(216, 239, YW_C - 11, '#9a7a5a')
    cv.vline(238, YW_C - 18, YW_C - 10, '#4e3a30')            # mooring post
    cv.px(238, YW_C - 19, '#9a7a5a')
    # string of lanterns along the boardwalk
    for k, px_ in enumerate(range(124, 224, 7)):
        y = YW_C - 13 + int(round(1.5 * math.sin((px_ - 124) / 7 * math.pi)))
        cv.px(px_, y, ['#ffcf70', '#ff9ec8', '#9be8e4'][k % 3])
    # ---- lighthouse on its rock (right)
    lx = 330
    cv.poly([(300, YW_C + 2), (306, 118), (318, 112), (338, 110), (352, 116), (362, YW_C + 2)], ROCK)
    cv.poly([(306, 120), (318, 114), (336, 112), (338, 116), (320, 118)], ROCK_HI)
    rim.line(318, 112, 352, 116, ROCK_MOON)
    for yy in range(76, 112):
        t = (yy - 76) / 36
        hw = int(3 + t * 3)
        stripe = ((yy - 76) // 6) % 2
        cv.hline(lx - hw - 1, lx + hw + 1, yy, '#3a3046')
        cv.hline(lx - hw, lx + hw, yy, '#f4f0f6' if stripe == 0 else '#ff8fb8')
        cv.hline(lx + hw - 1, lx + hw, yy, '#c8c0d0' if stripe == 0 else '#d0668e')
        rim.px(lx + hw + 1, yy, '#e8eeff')
    cv.rect(lx - 4, 69, 9, 7, '#3a3046')
    cv.rect(lx - 3, 70, 7, 5, '#fff2c0')
    cv.vline(lx, 70, 74, '#3a3046')
    cv.rect(lx - 5, 75, 11, 1, '#5a4a66')
    cv.tri((lx - 5, 69), (lx + 5, 69), (lx, 63), '#c8406a')
    cv.tri((lx, 69), (lx + 5, 69), (lx, 63), '#982e52')
    cv.px(lx, 62, '#ffd98a')
    cv.rect(lx - 1, 104, 3, 5, '#3a2a30')
    cv.px(lx, 93, '#ffd98a')
    # ---- pines along the right shore behind the lighthouse
    for i in range(7):
        px = int(rr.uniform(360, 418))
        pine(cv, px, YW_C + 1, ri(rr, 8, 15), '#1d3a39', '#15302f', '#2e554f', '#0f2124')
    cv.poly([(356, YW_C + 2), (364, 122), (390, 118), (420, 116), (420, YW_C + 2)], ROCK_SH)
    return to_rgba(cv), to_rgba(rim), tuple(lights), tuple(apexes), (lx, 72)


@lru_cache(maxsize=None)
def sailboat(seed, hull, sail_furled=True):
    """Small moored sailboat (RGBA), mast with a lantern. Bottom row = waterline."""
    w, h = 22, 34
    cv = layer(w, h)
    cv.poly([(0, 27), (21, 27), (18, 33), (3, 33)], shade(hull, 0.55))
    cv.poly([(1, 27), (20, 27), (18, 32), (3, 32)], hull)
    cv.hline(1, 20, 27, mix(hull, 'white', 0.5))
    cv.hline(3, 18, 31, shade(hull, 0.75))
    cv.rect(7, 24, 7, 3, '#e8e0d0')                 # little cabin
    cv.px(9, 25, W_WARM)
    cv.px(11, 25, W_WARM)
    cv.vline(10, 2, 26, '#5a4a40')                  # mast
    cv.hline(10, 19, 22, '#5a4a40')                 # boom
    if sail_furled:
        cv.rect(11, 21, 8, 1, '#e8eef6')
        cv.hline(11, 18, 20, '#c8d0de')
    cv.line(10, 2, 1, 26, '#4a4a5a')                # stays
    cv.line(10, 2, 20, 26, '#4a4a5a')
    cv.px(10, 1, '#ffcf70')
    return to_rgba(cv)


def draw_boat(c, rgba, x, wy, f, ph=0.0, refl_k=0.5):
    """Boat bobbing on the swell with its reflection. (x, wy) = left of the hull at the waterline."""
    bob = int(round(math.sin(f * 0.11 + ph) * 1.2))
    h, w = rgba.shape[:2]
    top = wy - h + 1 + bob - 2
    refl = rgba[::-1].copy()
    refl[..., :3] = (refl[..., :3].astype(np.float32) * np.array([0.42, 0.48, 0.62]) +
                     np.array([8, 14, 28])).astype(np.uint8)
    for r in range(0, h, 1):
        if (r + f // 4) % 3 == 2:
            continue
        dx = int(round(math.sin(r * 0.9 + f * 0.2 + ph) * (0.4 + r * 0.1)))
        row = refl[r:r + 1].copy()
        lit = row[..., :3].mean(axis=2) > 120
        row[..., 3] = np.where((bayer(1, w, int(x), wy + r) < 0.85 - r * 0.025) | lit, row[..., 3], 0)
        c.blit(row, int(x) + dx, wy + 1 + bob + r - 2 + 2)
    c.blit(rgba, int(x), top)


@lru_cache(maxsize=None)
def dock_c():
    """Foreground wooden dock (RGBA): planks, posts into the water, crates, rope, buoy, lantern post."""
    w = 460
    cv = layer(w, H)
    y0 = 156
    cv.rect(-10, y0, 210, 7, '#4e3a30')
    for x in range(-10, 200, 6):
        cv.vline(x, y0 + 1, y0 + 6, '#3a2a24')
        cv.hline(x + 1, x + 4, y0, '#8a6a52')
    cv.hline(-10, 199, y0 + 6, '#2a1e1a')
    for x in range(-6, 200, 18):                            # posts
        cv.rect(x, y0 + 7, 4, H - y0 - 7, '#2f2420')
        cv.vline(x, y0 + 7, H, '#5a4436')
        cv.rect(x - 1, y0 - 4, 6, 4, '#3a2a24')
        cv.hline(x - 1, x + 4, y0 - 4, '#7a5a44')
    # crates + rope coil + buoy + lantern post
    for (x, cw, chh) in [(24, 10, 8), (35, 8, 6), (28, 7, 6)]:
        yb = y0 if x != 28 else y0 - 8
        cv.rect(x, yb - chh, cw, chh, '#7a5a3a')
        cv.frame(x, yb - chh, cw, chh, '#4a3424')
        cv.line(x, yb - chh, x + cw - 1, yb - 1, '#4a3424')
        cv.hline(x + 1, x + cw - 2, yb - chh, '#a8845a')
    cv.ellipse(60, y0 - 2, 5, 2, '#a88a5a')
    cv.ellipse(60, y0 - 2, 3, 1, '#6a5038')
    cv.ellipse(84, y0 - 4, 3, 4, '#f4f0f6')
    cv.hline(81, 87, y0 - 5, '#ff6a8a')
    cv.hline(81, 87, y0 - 3, '#ff6a8a')
    cv.vline(140, y0 - 30, y0, '#2f2420')
    cv.vline(139, y0 - 30, y0, '#5a4436')
    cv.hline(137, 143, y0 - 30, '#3a2a24')
    cv.rect(141, y0 - 29, 3, 4, '#ffcf70')
    cv.px(142, y0 - 30, '#3a2a24')
    # rope from the bollard down into the water (to a boat)
    for k in range(14):
        cv.px(196 + k, y0 + 2 + int(k * k * 0.04), '#a88a5a')
    return to_rgba(cv)


def cam_c(f):
    return 30 * cruise(seg(f, 270, 359))


def draw_beam(c, lx, ly, f):
    """Lighthouse beam sweeping (projected rotation): a dithered additive wedge."""
    a = f * 0.06 + 1.2
    d = math.cos(a)
    L = int(150 * abs(d))
    if L > 4:
        sgn = 1 if d > 0 else -1
        x0, x1 = (lx, lx + sgn * L) if sgn > 0 else (lx + sgn * L, lx)
        x0, x1 = max(0, x0), min(W, x1)
        if x1 > x0:
            y0, y1 = max(0, ly - 16), min(H, ly + 17)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = np.abs(xx - lx).astype(np.float32)
            half = 0.6 + dist * 0.07
            dy = np.abs(yy - ly)
            inside = dy <= half
            inten = np.clip(1 - dist / max(1, L), 0, 1) ** 1.3 * (0.55 + 0.45 * abs(d))
            inten = inten * (1.15 - 0.6 * dy / np.maximum(half, 1))
            thr = bayer(y1 - y0, x1 - x0, x0, y0)
            m = inside & (thr < inten)
            reg = c.a[y0:y1, x0:x1].astype(np.float32)
            reg[m] += np.array([56, 52, 34], np.float32)
            core = inside & (dy <= np.maximum(0.6, half * 0.3)) & (dist < L * 0.65)
            reg[core] += np.array([40, 38, 24], np.float32)
            c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    flare = 1 - min(1, abs(d) * 3)
    fx.glow(c, lx, ly, 6 + int(8 * flare), '#ffe6a0', 0.35 + 0.4 * flare, rings=3)


def render_c(f):
    cx = cam_c(f)
    mt = moon_t(f)
    c = Canvas(W, H)
    sky = [mix(a, b, mt * 0.55) for a, b in zip(SKY_NIGHT, ['#0e1430', '#141c3e', '#1a2650', '#22325e',
                                                             '#2b3f6c', '#364d78', '#425a82'])]
    c.gradient(0, 0, W, HZ + 1, sky, soft=0.7)
    mx, my = MOON[0] - int(cx * 0.1), MOON[1]
    # stars in the cleared sky
    if mt > 0:
        fx.stars(c, f, seed=17, n=int(56 * mt), region=(0, 0, W, 80))
    # moon + halo (rises 4 px as it breaks out)
    rise = int(round(4 * (1 - smooth(seg(f, T_MOON, T_MOON + 20)))))
    my += rise
    fx.glow(c, mx, my, 44, '#7a8cc0', 0.22 + 0.2 * mt, rings=6)
    c.circle(mx, my, MOON_R + 1, '#9fb0d6')
    c.circle(mx, my, MOON_R, '#eef3fb')
    c.circle(mx + 2, my + 1, MOON_R - 2, '#dde6f4')
    for (dx, dy, r) in [(-3, -1, 2.6), (-1, 2, 2.2), (1, 0, 1.6), (4, 5, 0.8)]:   # one mare + a crater
        c.circle(mx + dx, my + dy, r, '#c9d3e7')
    c.px(mx - 6, my - 5, 'white')
    c.px(mx - 5, my - 6, 'white')
    # clouds parting: every puff slides away from the moon (eased), thinning the deck
    k = ease_out(seg(f, T_MOON - 4, T_MOON + 56)) * 110
    kf = ease_out(seg(f, T_MOON - 6, T_MOON + 8)) * 80      # the puffs over the moon part fast
    for (px, py, pw, ph, sd) in C_PUFFS:
        pc = px + pw / 2 - MOON[0]
        dirn = 1 if pc > 0 else -1
        amt = k * (0.5 + 0.9 * hash01(sd, 4)) + (kf * 1.2 if abs(pc) < 60 else 0)
        xx = int(round(px - f * 0.06 - cx * 0.2 + dirn * amt))
        yy = int(round(py - k * 0.05 * hash01(sd, 6) * 8))
        paste(c, cloud_puff(sd, pw, ph), xx, yy)
    pulse = seg(f, T_MOON, T_MOON + 18)
    if 0 < pulse < 1:                                    # moonrise bloom ring, over the clouds
        r = int(14 + 44 * ease_out(pulse))
        fx.glow(c, mx, my, r, '#c8d6ff', 0.4 * (1 - pulse), rings=4)
    # far headlands of Luna Bay
    paste(c, far_c(), -int(cx * 0.15) - 10, 0)
    # ---- open sea (horizon to the stilt-town waterline)
    c.gradient(0, HZ, W, YW_C - HZ, [mix('#1a2a4a', '#2a3c66', mt), mix('#14223e', '#1e2e52', mt),
                                     mix('#101c34', '#18264a', mt)], soft=0.6)
    for i in range(30):                                   # swell lines drifting toward shore
        t = ((f * 0.25 + i * 7.3) % 32) / 32
        y = HZ + 1 + int(t * t * (YW_C - HZ - 2))
        x = int((hash01(i, 3) * 360 - cx * 0.6) % 360) - 20
        ln = 2 + int(t * 6)
        c.hline(x, x + ln, y, mix('#24365a', '#4a5e8e', mt))
    # far sailboat on the bay, silhouetted on the moon path
    fbx = int(244 + (f - 270) * 0.12 - cx * 0.3)
    fby = 104 + int(round(math.sin(f * 0.1)))
    c.tri((fbx, fby - 1), (fbx + 6, fby - 1), (fbx + 3, fby - 10), '#c8d2e6')
    c.vline(fbx + 3, fby - 11, fby - 1, '#3a3a50')
    c.rect(fbx - 1, fby - 1, 9, 2, '#28304a')
    c.px(fbx + 3, fby - 12, '#ffcf70')
    # ---- Silver Shoals town, lighthouse
    town, rim, lights, apexes, (llx, lly) = town_c()
    to = -int(round(cx * 0.7)) - 6
    paste(c, town, to, 0)
    paste_dither(c, rim, to, 0, mt, ox=to)
    for i, (lx, ly) in enumerate(lights):
        fx.glow(c, lx + to, ly, 5, '#ffb060', 0.16 + 0.04 * math.sin(f * 0.3 + i), rings=2)
    # balloon moored at the end of the boardwalk landing (burner glowing)
    bx = 226 + to
    draw_balloon(c, bx, YW_C - 12, f, 'm', flame=True)
    c.line(bx + 4, YW_C - 15, 238 + to, YW_C - 18, '#a88a5a')
    fx.glow(c, bx, YW_C - 18, 7, '#ffb050', 0.22, rings=2)
    # ---- near water: reflection of everything above the waterline
    reflect(c, YW_C, H, f, mix('#16243e', '#22345a', mt), '#0c1428', k0=0.5, k1=0.18, stretch=0.6,
            wob=2.4, bright_boost=0.9)
    rain_rings(c, f, YW_C + 2, H - 2, int(20 * (1 - mt) + 4), '#5f7a9c', 51, shift=cx)
    # moon path: glittering dashes from the horizon toward the viewer
    reach = HZ + 2 + (H - HZ) * ease_out(seg(f, T_MOON, T_MOON + 22))
    pcx = mx
    for y in range(HZ + 1, int(reach)):
        if (y + f // 3) % 2 and y > HZ + 3:
            continue
        d = y - HZ
        wdt = 2 + d * 0.38
        n = 1 + int(d * 0.06)
        for j in range(n):
            hh = hash01(y, j, f // 3)
            xx = int(pcx + (hh - 0.5) * 2 * wdt)
            ln = 1 + int(hash01(y, j, 7) * (1 + d * 0.05))
            cc = '#ffffff' if hash01(y, j, f // 3, 2) > 0.6 else ('#cfdcf4' if d < 40 else '#9fb4dc')
            if mt < 1 and hash01(y, j, 5) > mt * 1.4:
                continue
            c.hline(xx, xx + ln - 1, y, cc)
    # sailboats moored near the dock (bobbing) with reflections
    draw_boat(c, sailboat(1, '#5fb3b0'), 92 - int(cx * 1.0), 150, f, 0.0)
    draw_boat(c, sailboat(2, '#f29a8e'), 270 - int(cx * 1.0), 146, f, 1.7)
    # lighthouse beam
    draw_beam(c, llx + to, lly, f)
    # ---- foreground dock
    paste(c, dock_c(), -int(round(cx * 1.3)) - 4, 0)
    dl = 141 - int(round(cx * 1.3)) - 4
    fx.glow(c, dl + 1, 129, 14, '#ffb060', 0.3 + 0.04 * math.sin(f * 0.4), rings=4)
    # rain thinning out after the moonrise
    nr = int(60 * (1 - mt) + 6)
    rain_layer(c, f, nr, '#5d7394', 61, slant=1, speed=5, length=2, shift=cx * 0.7)
    rain_layer(c, f, max(2, int(22 * (1 - mt))), '#9ab6d8', 62, slant=2, speed=8, length=4, shift=cx * 1.3)
    # silver flood: lift the whole frame slightly toward the moonlight
    if mt > 0:
        fx.grade(c, (1 + 0.06 * mt, 1 + 0.08 * mt, 1 + 0.12 * mt), (4 * mt, 6 * mt, 10 * mt))
    # callouts
    hx, hy = apexes[1]
    callout(c, hx + to, hy - 1, 'SILVER SHOALS', f, 279, 351, side='up', length=16)
    callout(c, 274 - int(cx * 0.15), HZ - 1, 'LUNA BAY', f, 288, 352, side='up', length=22)
    return c


# ================================================================== render
def render(f):
    if f < T_A:
        return travel_frame(f, ID)
    if f < T_B:
        c = render_a(f)
    elif f < T_C:
        c = render_b(f)
    else:
        c = render_c(f)
    atlas_card(c, f, 100, 206, 'NORTHWEST COAST', ['RAIN - PINES - HARBOURS'])
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), 204, 103)
    return c


SFX = [
    (76, 'rain', 0.45, {'variant': 1}),          # VANHOOVER: drizzle on the city roofs
    (90, 'waves', 0.3),                          # bay lapping at the seawall
    (108, 'horn', 0.6, {'pan': 0.3}),            # the big ferry sounds its horn
    (132, 'rain', 0.4, {'variant': 1}),
    (150, 'waves', 0.28),
    (182, 'wind', 0.4, {'variant': 0}),          # SMOKEY MOUNTAINS: wind in the pines
    (198, 'creak', 0.35, {'pan': -0.4}),         # fortress timbers / banner poles
    (216, 'rain', 0.38, {'variant': 0}),         # steady rain on the forest
    (234, 'wind', 0.32, {'variant': 0}),
    (272, 'rain', 0.36, {'variant': 0}),         # SILVER SHOALS: still raining before the moon
    (292, 'waves', 0.5),                         # the sea after the moonrise (288)
    (310, 'rain', 0.22, {'variant': 1}),         # last drips off the stilt-house roofs
    (324, 'creak', 0.3, {'pan': 0.4}),           # moored boats creaking on the swell
    (338, 'waves', 0.4),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(180, 'cloud', {}), (270, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
