"""GRIFFONSTONE: the griffon kingdom (4 bars, 288 frames, HUD clock 14:15 -> 15:00). E minor, rugged.

Beat sheet (local frames; bar = 72, beat = 18). Weather: overcast and WINDY all segment (wind blows
left -> right): layered grey cloud decks race, flat diffuse light, one sun-break shaft as key light.
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71 on dest_screen (204, 92).
  Bar 2  72-143   REVEAL (72): iris opens 72-79 on TROTTINGHAM's castle (204, 92). ACROSS THE STRAIT:
                  the Griffish Isle (rolling patchwork fields, hedgerows, drifting fog) with Tudor
                  half-timbered houses (red tile roofs, smoking chimneys blown sideways), an inn sign,
                  a church tower, the stone castle with a whipping flag, the harbour wall with a blinking
                  beacon and bobbing rowboats. A two-masted sailing ship heels in the wind on a whitecapped
                  slate sea; spray bursts on the foreground rocks (beats). Far right on the horizon: the
                  Griffonstone spire silhouette (teaser). Sun-break shaft sweeps slowly over the town.
                  Camera: slow eased pan right (parallax 7 layers). Balloon 's' fights the wind.
                  Callout TROTTINGHAM 81-137. Atlas card 100-176 (bottom-left, clear of the train).
  Bar 3 144-215   CUT on the bar line: GRIFFONSTONE. Opens at the gorge base: GRIFFONSTONE STATION by the
                  Guto River, the pink Friendship Express steaming (smoke blown flat), dark pines, olive
                  crags, the foot of the enormous grey rock spire (a dead giant tree of stone).
                  162-210: tall eased TILT UP the spire: gnarled rock branches with stone-and-plank houses,
                  rope ladders, walkways, the library tower, to the ruined castle on the crown with the
                  empty throne terrace and tattered banners. Sun-break shaft on the crown.
                  Callouts GRIFFONSTONE STATION 148-178, GRIFFONSTONE 200-232 (castle).
  Bar 4 216-287   GUST (216): banners whip, pines bend, debris blasts across, the balloon is shoved; the
                  camera is "blown" right and down (216-258) to the ABYSMAL ABYSS: the rickety rope bridge
                  sways hard (planks rattle; a loose plank snaps at 234 and tumbles into the mist), mist rolls
                  in the dark chasm. Callout ABYSMAL ABYSS 240-280. Toast from 230 (top-centre kept clear).
Ambient: racing cloud decks + scud, sweeping sun shaft, fog/mist banks, whitecaps + waves, ship heel +
         pennants, chimney smoke, flags/banners, lit windows flicker, beacon blink, rowboats bob,
         spray, wind-bent grass + pines, train steam, river flow, bridge sway, blowing leaves, balloon.
SFX: wind (soft + howl at the gust), waves, creak (ship, bridge, plank), whistle, steam, thunder (distant).
"""
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.shotfx import cruise
from engine import fx
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame, dest_screen

ID = 'griffonstone'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
BT = FRAMES_PER_BEAT
GUST = 216
KEY = (255, 0, 255)

# ------------------------------------------------------------------ palette (drab, overcast)
SKY = ['#565c66', '#646b74', '#79808a', '#8f979c', '#a6adad', '#b7bcb7']
DECK_NEAR = ('#40454e', '#4b5059', '#575d66', '#646a72')      # shade, body, light, rim
DECK_MID = ('#5d636c', '#6b727a', '#7c838a', '#c3c7c2')
DECK_FAR = ('#848b90', '#929a9d', '#a2a9aa', '#b8bdb9')
SEA = ['#536a6c', '#465d61', '#3e5357', '#364a4f', '#2f4247']
SEA_HI = '#6f8684'
FOAM = '#c9d2cc'
GRASS = ('#3e5543', '#4d6a4c', '#5f7d55', '#768f5f', '#90a46c')   # dark .. light
HEDGE = ('#2c3e33', '#385040')
PLASTER = ('#cfc6ae', '#e6dfca', '#f2ecdb')
TIMBER = ('#3f2d2a', '#563c33')
PLASTERS = [PLASTER, ('#d6c49a', '#e8d6a8', '#f1e3bb'), ('#d2bcb0', '#e4cfc3', '#efdfd5'),
            ('#c4c6b8', '#d9dccd', '#e9ebde')]
TILE = ('#6e2a27', '#8e3a2f', '#a8493a', '#c0634b')
STONE = ('#4e5155', '#666a6c', '#80858a', '#9da3a4', '#b9bebb')
WARM = ('#f5b14c', '#ffd77a')


# ------------------------------------------------------------------ helpers
def _layer(w, h):
    return Canvas(w, h, KEY)


def _rgba(cv):
    a = cv.a
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = (~np.all(a == np.array(KEY, np.uint8), axis=2)) * 255
    return out


def _tnoise(h, w, sigma, seed):
    """Tileable smooth noise 0..1."""
    r = np.random.RandomState(seed).rand(h, w)
    n = gaussian_filter(r, sigma, mode='wrap')
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def _wrap_slice(arr, x, w):
    """Columns x..x+w of a horizontally tileable array."""
    n = arr.shape[1]
    idx = (np.arange(w) + int(x)) % n
    return arr[:, idx]


def _blit_wrap(c, arr, x_off, y):
    c.blit(_wrap_slice(arr, x_off, W), 0, y)


def _ramp_px(c, x, y, cols, k):
    c.px(x, y, cols[max(0, min(len(cols) - 1, k))])


def gust_k(f):
    """Wind strength 0..1: steady breeze with swells, the big GUST at 216 (decays over ~2 bars)."""
    base = 0.45 + 0.15 * math.sin(f / 23.0) + 0.1 * math.sin(f / 9.0 + 1.3)
    if f >= GUST:
        t = (f - GUST) / 14.0
        base += 0.9 * math.exp(-max(0.0, (f - GUST - 4)) / 30.0) * min(1.0, t * 3)
    return base


# ------------------------------------------------------------------ cloud decks (shared by both shots)
@lru_cache(maxsize=None)
def _deck(kind):
    """Tileable (512 wide) overcast cloud deck strip, RGBA. Lobed puffy underside, streaky body."""
    TW = 512
    if kind == 'near':
        hh, base, rx, ry, cols, seed, step = 46, 26, (10, 20), (5, 9), DECK_NEAR, 11, 9
    elif kind == 'mid':
        hh, base, rx, ry, cols, seed, step = 40, 26, (8, 16), (4, 7), DECK_MID, 23, 8
    else:
        hh, base, rx, ry, cols, seed, step = 24, 14, (7, 14), (3, 5), DECK_FAR, 37, 7
    sh, body, lt, rim = cols
    cv = Canvas(TW * 3, hh, KEY)
    r = rng(seed)
    lobes = []
    x = 0
    while x < TW:
        a = rx[0] + r.random() * (rx[1] - rx[0])
        b = ry[0] + r.random() * (ry[1] - ry[0])
        yb = base + int(5 * math.sin(x / TW * 2 * math.pi * 3 + seed) + 3 * math.sin(x / TW * 2 * math.pi * 7))
        yb += int((r.random() - 0.5) * 6)
        lobes.append((x, yb, a, b))
        x += step + int(r.random() * step)
    # continuous body above the lobes
    cv.rect(0, 0, TW * 3, max(1, base - 8), body)
    lobes.sort(key=lambda L: L[1])
    for (lx, yb, a, b) in lobes:
        for k in (0, 1, 2):
            cx = lx + TW * k
            cv.ellipse(cx, yb - b, int(a), int(b), sh)
            cv.ellipse(cx, yb - b - 1, int(a), int(b), body)
            cv.ellipse(cx - int(a * 0.25), yb - b - int(b * 0.55), int(a * 0.55), max(1, int(b * 0.45)), lt)
    out = _rgba(cv)[:, TW:TW * 2].copy()
    # streaks of lighter / darker texture in the body (wind-combed)
    n = _tnoise(hh, TW, (1.0, 9.0), seed + 1)
    m = out[..., 3] > 0
    bodyc = np.array(col(body), np.uint8)
    isbody = m & np.all(out[..., :3] == bodyc, axis=2)
    thr = bayer(hh, TW)
    out[isbody & (n > 0.62 + thr * 0.12), :3] = col(lt)
    out[isbody & (n < 0.30 - thr * 0.1), :3] = col(sh)
    return out


def draw_sky(c, f, cam_x, gap_x, y_off=0, horizon=100, tilt=0):
    """Overcast sky with three racing cloud decks and a sun-break gap at screen x gap_x."""
    y0 = -tilt
    c.gradient(0, y0, W, horizon - y0, SKY, soft=0.7)
    if horizon < H:
        c.rect(0, horizon, W, H - horizon, SKY[-1])
    # sun glow behind the gap
    gy = 30 + y_off
    fx.glow(c, gap_x, gy, 34, '#fff3cf', strength=0.22, rings=5)
    fx.glow(c, gap_x, gy - 2, 16, '#fffbe8', strength=0.28, rings=4)
    w = gust_k(f)
    # far deck (low, slow)
    far = _deck('far')
    _blit_wrap(c, far, int(f * 0.35 + cam_x * 0.15), horizon - 34 + y_off // 2)
    # mid deck with the sun-break hole
    mid = _deck('mid')
    ms = _wrap_slice(mid, int(f * (0.7 + 0.25 * w) + cam_x * 0.25), W).copy()
    my = 12 + y_off
    hh = ms.shape[0]
    yy, xx = np.mgrid[0:hh, 0:W]
    rn = _tnoise(hh, 512, 2.0, 91)[:, (np.arange(W) + int(f * 0.4)) % 512]
    dx = (xx - gap_x) / 20.0
    dy = (yy + my - gy) / 10.0
    d = dx * dx / (1 + np.maximum(dy, 0) * 0.9) + np.minimum(dy, 0) ** 2 + (rn - 0.5) * 0.9
    hole = d < 1.0
    deck = ms[..., 3] > 0
    bt = bayer(hh, W)
    rim = (d >= 1.0) & (d < 1.15) & deck & (bt < 0.3)
    ms[hole, 3] = 0
    ms[rim, :3] = col(DECK_MID[3])
    rim2 = (d >= 1.0) & (d < 1.6) & deck & (bt < 0.5)
    ms[rim2, :3] = col(DECK_MID[2])
    c.blit(ms, 0, my)
    # near deck (top, fast)
    near = _deck('near')
    ns = _wrap_slice(near, int(f * (1.3 + 0.8 * w) + cam_x * 0.35), W).copy()
    ny = -14 + y_off
    hh = ns.shape[0]
    yy, xx = np.mgrid[0:hh, 0:W]
    dy = (yy + ny - gy + 6) / 14.0
    d = ((xx - gap_x) / 30.0) ** 2 / (1 + np.maximum(dy, 0)) + np.minimum(dy, 0) ** 2
    ns[d < 1.0, 3] = 0
    c.blit(ns, 0, ny)
    # light leaking around the break brightens the cloud edges
    fx.glow(c, gap_x, gy - 4, 46, '#fff6dc', strength=0.10, rings=4)
    # scud: small dark wisps racing under the decks
    for i in range(9):
        sp = 2.2 + hash01(i, 5) * 1.5 + w
        sx = int((hash01(i, 1) * 600 + f * sp - cam_x * 0.4) % 600) - 140
        sy = int(36 + hash01(i, 2) * 40) + y_off
        L = 8 + int(hash01(i, 3) * 16)
        c.hline(sx, sx + L, sy, DECK_NEAR[1])
        c.hline(sx + 3, sx + L - 4, sy - 1, DECK_NEAR[2])
        c.hline(sx + 5, sx + L + 3, sy + 1, mix(DECK_NEAR[1], SKY[3], 0.5))


def sun_shaft(c, gap_x, gy, ground_y, f, strength=0.15, slant=-0.35):
    sw = 9 + int(2 * math.sin(f / 17.0))
    top = ((gap_x - sw, gy + 6), (gap_x + sw, gy + 6))
    dy = ground_y - gy
    bx = gap_x + int(dy * slant)
    bottom = ((bx - sw - 16, ground_y), (bx + sw + 18, ground_y))
    fx.light_beam(c, top, bottom, '#fff1c4', strength=strength)
    fx.light_beam(c, ((gap_x - 3, gy + 6), (gap_x + 4, gy + 6)),
                  ((bx - 6, ground_y), (bx + 8, ground_y)), '#fffbe0', strength=strength * 0.8)


# ================================================================== BAR 2: TROTTINGHAM
TW_ISLE = 440        # world width of the isle layer
ISLE_TOP = 40        # isle layer starts at screen y 40
SEA_Y = 114          # isle waterline (screen y)
HORIZON = 101
PAN = 74             # camera travel over the bar


def _isle_ground(x):
    """Ground y (screen) of the isle at world x, or None (sea)."""
    if x < 52 or x > 404:
        return None
    pts = [(52, 114), (70, 109), (96, 104), (128, 100), (156, 97), (178, 91), (192, 86), (222, 86),
           (238, 90), (262, 97), (290, 99), (318, 95), (340, 98), (372, 106), (404, 114)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            t = t * t * (3 - 2 * t)
            return y0 + (y1 - y0) * t
    return None


def _tudor(cv, x, base, w, wall, roof, seed, lit=False, oy=0, sign=False, rot=False):
    """Half-timbered house: (x, base) bottom-left, w wide, wall height, roof height."""
    r = rng(seed)
    y0 = base - wall
    PLASTER = PLASTERS[seed % 4]
    # walls: ground storey (slightly narrower), jettied upper storey
    lw = w - 2
    cv.rect(x + 1, base - wall // 2, lw, wall // 2, PLASTER[1])
    cv.rect(x, y0, w, wall - wall // 2, PLASTER[2])
    cv.hline(x, x + w - 1, base - wall // 2, TIMBER[0])          # jetty beam
    cv.hline(x + 1, x + w - 2, base - wall // 2 + 1, mix(PLASTER[1], TIMBER[0], 0.35))  # jetty shadow
    # timber frame
    cv.vline(x, y0, base - wall // 2, TIMBER[0])
    cv.vline(x + w - 1, y0, base - wall // 2, TIMBER[0])
    cv.vline(x + 1, base - wall // 2, base - 1, TIMBER[0])
    cv.vline(x + w - 2, base - wall // 2, base - 1, TIMBER[0])
    cv.hline(x, x + w - 1, y0, TIMBER[0])
    for px_ in range(x + 4, x + w - 3, 5):
        cv.vline(px_, y0, base - wall // 2, TIMBER[1])
        cv.vline(px_, base - wall // 2, base - 1, TIMBER[1])
    if w >= 10:
        cv.line(x + 1, base - wall // 2 - 1, x + 3, y0 + 1, TIMBER[1])
        cv.line(x + w - 2, base - wall // 2 - 1, x + w - 4, y0 + 1, TIMBER[1])
    # right-side shade (light from upper-left)
    cv.vline(x + w - 2, y0 + 1, base - wall // 2 - 1, PLASTER[0])
    # windows: leaded, some lit
    wins = []
    for wx in range(x + 2, x + w - 3, 5):
        if r.random() < 0.8:
            wy = y0 + 2
            litw = lit and r.random() < 0.55
            cv.rect(wx, wy, 2, 2, WARM[0] if litw else '#3a4048')
            if litw:
                cv.px(wx, wy, WARM[1])
                wins.append((wx, wy))
            else:
                cv.px(wx, wy, '#6c7680')
    # door
    dx = x + 2 + int(r.random() * max(1, w - 6))
    dh = min(5, wall // 2 - 1)
    cv.rect(dx, base - dh, 3, dh, TIMBER[0])
    cv.px(dx + 1, base - dh, TIMBER[1])
    if wall // 2 >= 5 and w >= 9:
        lx = x + w - 5 if dx < x + w // 2 else x + 3
        litw = lit and r.random() < 0.7
        cv.rect(lx, base - wall // 2 + 2, 2, 2, WARM[0] if litw else '#3a4048')
        if litw:
            wins.append((lx, base - wall // 2 + 2))
    # steep red-tile gable roof (side view: a trapezoid, ridge along x)
    ry0 = y0 - roof
    for i in range(roof):
        yy = ry0 + i
        inset = max(0, (roof - 1 - i) // 2 - 1) if not rot else max(0, roof - 1 - i)
        xa, xb = x - 1 + inset, x + w + inset * 0 - inset
        k = 3 if i < 1 else (2 if i % 2 == 0 else 1)
        cv.hline(xa, xb, yy, TILE[k])
        if i % 2 == 1:
            for tx in range(xa + (i // 2) % 2, xb, 3):
                cv.px(tx, yy, TILE[0])
    cv.hline(x - 1, x + w, y0 - 1, TILE[0])
    cv.hline(x - 1 + max(0, (roof - 1) // 2 - 1), x + w - max(0, (roof - 1) // 2 - 1), ry0, TILE[3])
    # chimney
    chim = None
    if r.random() < 0.85:
        cx_ = x + 2 + int(r.random() * (w - 5))
        ch = roof + 3
        cv.rect(cx_, ry0 - 3, 3, ch - 2, '#7a3b30')
        cv.vline(cx_, ry0 - 3, ry0 + ch - 6, '#9a5240')
        cv.rect(cx_ - 1, ry0 - 4, 5, 1, '#5b2c26')
        chim = (cx_ + 1, ry0 - 5)
    if sign:
        sx = x + w
        cv.hline(sx, sx + 4, base - wall // 2 - 1, TIMBER[0])
        cv.rect(sx + 1, base - wall // 2, 5, 5, '#6b4a2e')
        cv.rect(sx + 2, base - wall // 2 + 1, 3, 3, '#c79a4a')      # mug icon
        cv.px(sx + 5, base - wall // 2 + 2, '#c79a4a')
        cv.px(sx + 3, base - wall // 2 + 1, '#f2ecdb')
    return chim, wins


def _castle(cv, cx, base):
    """Small stone castle on the hilltop. Returns flag pole top (x, y), window list."""
    s0, s1, s2, s3, s4 = STONE
    # curtain wall
    cv.rect(cx - 30, base - 12, 60, 12, s2)
    cv.hline(cx - 30, cx + 29, base - 12, s3)
    for mx in range(cx - 30, cx + 30, 4):
        cv.rect(mx, base - 15, 2, 3, s2)
        cv.px(mx, base - 15, s3)
    cv.rect(cx + 18, base - 11, 12, 11, s1)       # shaded right part
    # keep
    kx, kw, kh = cx - 9, 18, 30
    cv.rect(kx, base - kh, kw, kh, s2)
    cv.vline(kx, base - kh, base - 1, s3)
    cv.vline(kx + 1, base - kh, base - 1, s3)
    cv.rect(kx + kw - 5, base - kh, 5, kh, s1)
    cv.vline(kx + kw - 1, base - kh, base - 1, s0)
    for mx in range(kx - 1, kx + kw + 1, 4):
        cv.rect(mx, base - kh - 3, 3, 3, s2 if mx < kx + kw - 5 else s1)
        cv.px(mx, base - kh - 3, s3)
    cv.hline(kx - 1, kx + kw, base - kh, s0)
    # stone courses
    for yy in range(base - kh + 4, base - 2, 4):
        for xx in range(kx + 2 + (yy // 4) % 2 * 2, kx + kw - 1, 5):
            cv.px(xx, yy, s1 if xx < kx + kw - 5 else s0)
    # arrow slits + a lit window
    wins = []
    for yy in (base - kh + 6, base - kh + 15):
        cv.rect(kx + 5, yy, 1, 4, s0)
        cv.rect(kx + 11, yy, 1, 4, s0)
    cv.rect(kx + 7, base - kh + 9, 3, 4, '#3a3f46')
    cv.rect(kx + 7, base - kh + 8, 3, 1, s3)
    cv.rect(kx + 8, base - kh + 10, 1, 3, WARM[0])
    wins.append((kx + 8, base - kh + 10))
    # gate
    cv.rect(cx - 2, base - 7, 5, 7, '#2f3237')
    cv.px(cx - 2, base - 7, s2)
    cv.px(cx + 2, base - 7, s2)
    cv.hline(cx - 2, cx + 2, base - 8, s3)
    # round tower (left) with a conical slate roof
    tx = cx - 27
    cv.rect(tx, base - 24, 10, 24, s2)
    cv.vline(tx, base - 24, base - 1, s3)
    cv.rect(tx + 7, base - 24, 3, 24, s1)
    for i in range(10):
        yy = base - 25 - i
        half = 6 - (i * 6) // 10
        cv.hline(tx + 5 - half, tx + 4 + half, yy, '#4a5262' if i % 3 else '#5c6577')
    cv.px(tx + 4, base - 35, '#5c6577')
    cv.rect(tx + 4, base - 18, 2, 3, '#33373d')
    # small square tower right
    rx = cx + 20
    cv.rect(rx, base - 20, 8, 20, s1)
    cv.vline(rx, base - 20, base - 1, s2)
    for mx in range(rx - 1, rx + 9, 3):
        cv.rect(mx, base - 22, 2, 2, s1)
    cv.rect(rx + 3, base - 15, 1, 3, s0)
    return (kx + 3, base - kh - 3), wins


@lru_cache(maxsize=None)
def _isle_layer():
    """Main isle + town + castle (world coords: x 0..TW_ISLE, y ISLE_TOP..SEA_Y+2). RGBA + metadata."""
    hgt = SEA_Y + 3 - ISLE_TOP
    cv = _layer(TW_ISLE, hgt)
    oy = ISLE_TOP
    gy = np.full(TW_ISLE, 999.0)
    for x in range(TW_ISLE):
        g = _isle_ground(x)
        if g is not None:
            gy[x] = g
    # patchwork fields
    yy, xx = np.mgrid[0:hgt, 0:TW_ISLE]
    sy = yy + oy
    gcol = gy[None, :]
    inside = sy >= gcol
    d = sy - gcol
    band = (d // 7).astype(int)
    segid = ((xx + band * 13) // 33).astype(int)
    pid = (band * 31 + segid * 7) % 5
    slope = np.gradient(gy)
    slope = np.clip(np.nan_to_num(slope), -2, 2)[None, :]
    lit = (-slope) * 1.2 + 0.0
    k = np.clip(np.round(2 + lit + (pid - 2) * 0.45 - band * 0.12), 0, 4).astype(int)
    # dither between field tones
    thr = bayer(hgt, TW_ISLE)
    k2 = np.clip(k + ((pid % 2) * (thr > 0.6)).astype(int) * 0, 0, 4)
    gpal = np.array([col(cc) for cc in GRASS], np.uint8)
    rgb = gpal[k2]
    # olive / moss tinted fields
    tint = (pid == 4) & (thr < 0.5)
    rgb[tint] = col('#7c8a55')
    tint2 = (pid == 1) & (thr < 0.35)
    rgb[tint2] = col('#5b7550')
    # hedgerows (dark, bushy) along the contour bands and field edges
    hedge = (((d % 7) == 0) & (d > 2)) | ((((xx + band * 13) % 33) == 0) & (d > 2))
    rgb[hedge] = col(HEDGE[1])
    bush = hedge & (((xx * 7 + yy * 3) % 5) < 2)
    rgb[bush] = col(HEDGE[0])
    # top rim highlight
    rim = inside & (d < 1.2)
    rgb[rim] = col(GRASS[4])
    a = cv.a
    a[inside] = rgb[inside]
    chims, wins = [], []
    # shore rocks along the waterline
    r = rng(5)
    for x in range(52, 405, 3):
        g = _isle_ground(x)
        if g is None or g < SEA_Y - 8:
            continue
        h = 2 + int(r.random() * 3)
        cv.rect(x, SEA_Y - h - oy, 3 + int(r.random() * 3), h + 2, STONE[1])
        cv.px(x, SEA_Y - h - oy, STONE[3])
    # trees bending in the wind (round canopies, pushed right)
    for (tx, sd) in ((88, 1), (140, 2), (171, 3), (250, 4), (300, 5), (332, 6), (356, 7), (118, 8)):
        g = int(_isle_ground(tx)) - oy
        cv.vline(tx, g - 6, g, TIMBER[1])
        cv.px(tx + 1, g - 6, TIMBER[1])
        cv.ellipse(tx + 2, g - 9, 4, 3, HEDGE[0])
        cv.ellipse(tx + 2, g - 10, 4, 3, GRASS[1])
        cv.ellipse(tx + 1, g - 11, 2, 2, GRASS[2])
        cv.px(tx, g - 12, GRASS[3])
    # castle on the hilltop
    flag, cw = _castle(cv, 207, 88 - oy)
    wins += [(x, y + oy) for x, y in cw]
    # church with a square tower (right slope)
    chx, chb = 258, int(_isle_ground(262)) + 2 - oy
    cv.rect(chx, chb - 12, 22, 12, STONE[2])
    cv.rect(chx + 16, chb - 12, 6, 12, STONE[1])
    for i in range(6):
        cv.hline(chx - 1 + i, chx + 22 - i, chb - 13 - i, '#4a5262' if i % 2 else '#565f70')
    cv.rect(chx - 6, chb - 26, 8, 26, STONE[2])
    cv.vline(chx - 6, chb - 26, chb - 1, STONE[3])
    cv.rect(chx - 1, chb - 26, 3, 26, STONE[1])
    for mx in range(chx - 7, chx + 3, 3):
        cv.rect(mx, chb - 28, 2, 2, STONE[2])
    cv.rect(chx - 4, chb - 21, 2, 4, '#3a3f46')
    cv.px(chx - 4, chb - 22, STONE[3])
    for wx in range(chx + 3, chx + 15, 4):
        cv.rect(wx, chb - 9, 2, 4, '#3a3f46')
        cv.px(wx, chb - 10, STONE[3])
    wins.append((chx + 7, chb - 8 + oy))
    # Tudor houses: (x, w, wall, roof, seed, sign)
    houses = [(64, 14, 12, 7, 1, False), (80, 16, 13, 8, 2, False), (99, 12, 12, 7, 3, False),
              (114, 17, 14, 8, 4, True), (135, 13, 12, 7, 5, False), (152, 15, 13, 7, 6, False),
              (170, 11, 11, 6, 7, False), (238, 12, 11, 6, 8, False), (284, 15, 13, 7, 9, False),
              (303, 13, 12, 7, 10, False), (320, 16, 13, 8, 11, False), (340, 12, 11, 6, 12, False),
              (356, 14, 12, 7, 13, False), (91, 11, 10, 6, 14, False), (126, 12, 10, 6, 15, False)]
    # back row first (those drawn higher up the slope)
    houses.sort(key=lambda h: _isle_ground(h[0] + h[1] // 2))
    for (hx, w, wall, roof, sd, sign) in houses:
        g = max(_isle_ground(hx), _isle_ground(hx + w - 1))
        base = int(g) + 2 - oy
        if hx in (91, 126):
            base -= 6
        ch, wn = _tudor(cv, hx, base, w, wall, roof, sd, lit=True, sign=sign)
        if ch:
            chims.append((ch[0], ch[1] + oy))
        wins += [(x, y + oy) for x, y in wn]
    # front row: harbour-front and lower-slope houses (overlap the back row -> dense town)
    front = [(70, 13, 11, 6, 21), (85, 15, 12, 7, 22), (108, 12, 11, 6, 23), (122, 14, 12, 7, 24),
             (146, 13, 11, 6, 25), (163, 12, 11, 6, 26), (246, 14, 12, 7, 27), (268, 12, 11, 6, 28),
             (296, 15, 12, 7, 29), (318, 12, 11, 6, 30), (348, 14, 12, 7, 31)]
    for (hx, w, wall, roof, sd) in front:
        g = max(_isle_ground(hx), _isle_ground(hx + w - 1))
        base = min(SEA_Y - 1, int(g) + 11) - oy
        ch, wn = _tudor(cv, hx, base, w, wall, roof, sd, lit=True)
        if ch:
            chims.append((ch[0], ch[1] + oy))
        wins += [(x, y + oy) for x, y in wn]
        cv.hline(hx, hx + w - 1, base, HEDGE[1])
    # harbour wall (curving breakwater) + quay
    hw_y = SEA_Y - oy
    cv.rect(18, hw_y - 5, 46, 6, STONE[2])
    cv.hline(18, 63, hw_y - 5, STONE[4])
    cv.hline(18, 63, hw_y - 4, STONE[3])
    for xx in range(19, 63, 4):
        cv.vline(xx, hw_y - 3, hw_y, STONE[1])
    cv.rect(52, hw_y - 3, 14, 4, '#6b4a2e')     # wooden quay
    cv.hline(52, 65, hw_y - 3, '#8a6a44')
    for xx in (54, 58, 62):
        cv.vline(xx, hw_y - 3, hw_y + 2, '#4a3324')
    # beacon tower at the tip
    bx = 22
    cv.rect(bx - 3, hw_y - 22, 7, 17, PLASTER[2])
    cv.rect(bx + 1, hw_y - 22, 3, 17, PLASTER[0])
    for yy in range(hw_y - 19, hw_y - 6, 5):
        cv.rect(bx - 3, yy, 7, 2, TILE[1])
    cv.rect(bx - 2, hw_y - 26, 5, 4, '#3a3f46')
    cv.rect(bx - 4, hw_y - 22, 9, 1, '#2f3237')
    cv.tri((bx - 3, hw_y - 27), (bx + 3, hw_y - 27), (bx, hw_y - 30), TILE[0])
    beacon = (bx, hw_y - 24 + oy)
    return _rgba(cv), tuple(chims), tuple(wins), (flag[0], flag[1] + oy), beacon


@lru_cache(maxsize=None)
def _far_layers():
    """Far griffon mainland on the horizon (with the spire teaser) + low far isles. RGBA, width 520."""
    WF = 520
    cv = _layer(WF, 60)
    oy = HORIZON - 40
    far = '#8d9597'
    farsh = '#838b8f'
    # mainland crags (right part)
    xs = np.arange(WF)
    prof = 17 + 6 * np.sin(xs / 23.0) + 4 * np.sin(xs / 7.3 + 1) + 2 * np.sin(xs / 3.1)
    for x in range(230, WF):
        t = clamp((x - 230) / 50.0)
        top = int(40 - (40 - prof[x]) * t)
        cv.vline(x, top, 40, far if (x // 3) % 4 else farsh)
        cv.px(x, top, '#9aa2a3')
    # the spire silhouette (teaser) at x 300
    sx = 300
    for yy in range(2, 32):
        hw = 1 + (yy - 2) * 3 // 30
        cv.hline(sx - hw, sx + hw, yy, '#7f878b')
    cv.line(sx, 10, sx - 6, 5, '#7f878b')
    cv.line(sx, 14, sx + 7, 8, '#7f878b')
    cv.line(sx, 20, sx - 8, 15, '#7f878b')
    cv.line(sx + 1, 6, sx + 4, 2, '#7f878b')
    cv.rect(sx - 2, 0, 4, 3, '#7f878b')
    # far isles (left), paler
    for (x0, x1, hgt) in ((0, 70, 6), (120, 170, 3), (190, 230, 2)):
        for x in range(x0, x1):
            t = (x - x0) / (x1 - x0)
            hh = int(hgt * math.sin(t * math.pi) + 0.5)
            if hh > 0:
                cv.vline(x, 40 - hh, 40, '#8a9693')
    return _rgba(cv), oy


@lru_cache(maxsize=None)
def _sea_tex():
    """Tileable sea texture (rows from SEA top to bottom of frame)."""
    hh, TW = H - HORIZON, 512
    n = _tnoise(hh, TW, (0.6, 6.0), 77)
    yy = np.arange(hh)[:, None] / hh
    base = np.clip(yy * 4.2, 0, 4)
    k = np.clip(np.floor(base + (n - 0.5) * 1.6), 0, 4).astype(int)
    pal = np.array([col(cc) for cc in SEA], np.uint8)
    rgb = pal[k]
    # crest lines: thin light dashes, longer near the viewer
    crest = (n > 0.70) & (np.roll(n, 1, axis=0) <= 0.70)
    rgb[crest] = col(SEA_HI)
    return rgb


def _draw_sea(c, f, cam):
    tex = _sea_tex()
    hh = tex.shape[0]
    TW = tex.shape[1]
    rows = np.arange(hh)
    depth = rows / hh
    par = 0.3 + 1.1 * depth
    off = (cam * par + f * (0.4 + 1.2 * depth)).astype(int)
    idx = (np.arange(W)[None, :] + off[:, None]) % TW
    c.a[HORIZON:H] = tex[rows[:, None], idx]
    # far sea haze band
    c.dither(0, HORIZON, W, 4, '#7d8f8e', 0.5)
    c.hline(0, W - 1, HORIZON, '#93a3a0')


def _whitecaps(c, f, cam, y0=HORIZON + 3, y1=H, n=46, seed=3):
    for i in range(n):
        life = 22 + int(hash01(seed, i, 1) * 18)
        ph = int(hash01(seed, i, 2) * life)
        cyc = (f + ph) // life
        t = ((f + ph) % life) / life
        yy = y0 + int(hash01(seed, i, cyc, 3) ** 0.7 * (y1 - y0))
        depth = (yy - HORIZON) / (H - HORIZON)
        par = 0.3 + 1.1 * depth
        xx = int((hash01(seed, i, cyc, 4) * 420 - cam * par + t * 10 * (0.5 + depth)) % 420) - 50
        L = 1 + int(depth * 4 * math.sin(t * math.pi) + 0.5)
        if t < 0.15 or t > 0.85:
            c.px(xx, yy, SEA_HI)
            continue
        c.hline(xx - L, xx + L, yy, FOAM)
        c.px(xx, yy - 1, 'white')
        if depth > 0.5:
            c.hline(xx - L + 1, xx + L + 2, yy + 1, SEA_HI)


# ---- sailing ship (heels in the wind; cached by angle)
@lru_cache(maxsize=None)
def _ship_sprite(ang10):
    ang = ang10 / 10.0
    S = 64
    cv = Canvas(S, S, KEY)
    ox, oy = 26, 46          # waterline centre of the hull in the sprite
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def P(u, v):             # u along the hull (+ = bow, right), v up; heel rotates clockwise
        x = u * ca + v * sa
        y = -v * ca + u * sa * 0.25
        return (int(round(ox + x)), int(round(oy + y)))

    # hull
    hull = [P(-17, 4), P(15, 4), P(20, 6), P(14, -2), P(-14, -2), P(-18, 1)]
    cv.poly(hull, '#4a3324')
    cv.poly([P(-17, 4), P(15, 4), P(19, 6), P(16, 2), P(-17, 2)], '#6b4a2e')
    cv.line(*P(-17, 2), *P(16, 2), '#c9a25a')               # gilt stripe
    cv.line(*P(-17, 4), *P(15, 4), '#8a6a44')
    cv.line(*P(-14, -1), *P(13, -1), '#3a281e')
    for u in (-10, -4, 2, 8):
        cv.px(*P(u, 3), '#2b1f19')                           # gun ports
    cv.poly([P(-18, 6), P(-12, 6), P(-12, 4), P(-18, 4)], '#5a3f2a')   # stern castle
    cv.line(*P(-18, 7), *P(-12, 7), '#8a6a44')
    # masts
    mainb, maint = P(-2, 4), P(-2, 34)
    forb, fort = P(9, 4), P(9, 28)
    cv.line(*mainb, *maint, '#3a281e')
    cv.line(*forb, *fort, '#3a281e')
    cv.line(*P(20, 6), *P(28, 13), '#3a281e')               # bowsprit
    # sails (bellied to the right by the wind): two square sails per mast + jib
    sail, sail_sh, sail_hi = '#e9e2cf', '#c3bba5', '#fbf7ea'

    def square(u, v0, v1, wd, belly):
        pts = [P(u - wd, v1), P(u + wd, v1), P(u + wd + belly, (v0 + v1) / 2), P(u + wd, v0), P(u - wd, v0),
               P(u - wd + belly, (v0 + v1) / 2)]
        cv.poly(pts, sail)
        cv.poly([P(u + wd - 2, v1), P(u + wd, v1), P(u + wd + belly, (v0 + v1) / 2), P(u + wd, v0),
                 P(u + wd - 2, v0), P(u + wd - 2 + belly, (v0 + v1) / 2)], sail_sh)
        cv.line(*P(u - wd + 1, v1 - 1), *P(u + wd - 3, v1 - 1), sail_hi)
        cv.line(*P(u - wd - 1, v1), *P(u + wd + 1, v1), '#3a281e')          # yard

    square(-2, 8, 19, 7, 3)
    square(-2, 21, 31, 5, 2)
    square(9, 9, 18, 6, 3)
    square(9, 20, 27, 4, 2)
    cv.poly([P(10, 27), P(27, 12), P(15, 10)], sail)                 # jib
    cv.line(*P(10, 27), *P(27, 12), sail_sh)
    cv.poly([P(-3, 6), P(-3, 18), P(-15, 8)], sail_sh)                # spanker
    cv.line(*P(-3, 18), *P(-15, 8), '#3a281e')
    # rigging
    cv.line(*maint, *P(-18, 6), '#5a4a3e')
    cv.line(*fort, *P(27, 12), '#5a4a3e')
    cv.line(*maint, *fort, '#5a4a3e')
    return _rgba(cv), (ox, oy), (maint, fort)


def _draw_ship(c, f, sx, sy):
    w = gust_k(f)
    ang = 8 + 5 * w + 2.5 * math.sin(f / 13.0)
    a10 = int(round(ang)) * 10
    spr, (ox, oy), (mt, ft) = _ship_sprite(a10)
    bob = int(round(math.sin(f / 9.0) * 1.2))
    X, Y = sx - ox, sy - oy + bob
    c.blit(spr, X, Y)
    # pennants streaming right from the mast tops
    for (mx, my), L in ((mt, 9), (ft, 6)):
        for i in range(L):
            yy = Y + my + int(round(math.sin(f * 0.6 - i * 0.9) * (i / L) * 1.6))
            c.px(X + mx + 1 + i, yy, '#b8402f' if i < L - 2 else '#d9644a')
    # bow wave + wake, hull waterline foam
    for i in range(14):
        t = ((f * 0.9 + i * 3.1) % 14) / 14
        c.px(sx - 18 - int(t * 22), sy + 5 + int(hash01(i, 7) * 2) + bob, FOAM if t < 0.6 else SEA_HI)
    c.hline(sx - 18, sx + 18, sy + 6 + bob, SEA[2])
    c.hline(sx - 16, sx + 16, sy + 7 + bob, SEA[3])
    for i in range(5):
        k = (f // 3 + i) % 5
        c.px(sx + 19 + k, sy + 5 + bob - (k % 2), 'white')
        c.px(sx + 16 + k, sy + 6 + bob, FOAM)
    c.hline(sx - 18, sx + 19, sy + 5 + bob, FOAM)


# ---- foreground rocks with wind-bent grass + spray
@lru_cache(maxsize=None)
def _fg_rocks():
    WF = 520
    cv = _layer(WF, 50)
    r = rng(9)
    rock = ('#2a2f30', '#353d3b', '#454f49', '#5a6658', '#717d66')
    xs = np.arange(WF)
    top = (30 - 18 * np.exp(-((xs - 400) / 70.0) ** 2) - 10 * np.exp(-((xs - 470) / 30.0) ** 2)
           - 6 * np.exp(-((xs - 40) / 30.0) ** 2) + 2 * np.sin(xs / 5.0))
    top = np.where((xs > 120) & (xs < 300), 60, top)
    nz = _tnoise(50, WF, (3.0, 4.0), 19)
    thr = bayer(50, WF)
    for x in range(WF):
        t = int(top[x])
        if t >= 50:
            continue
        sl = top[min(WF - 1, x + 1)] - top[max(0, x - 1)]
        for y in range(t, 50):
            d = y - t
            v = 1.6 - d * 0.06 + (nz[y, x] - 0.5) * 2.6 - sl * 0.25 + (thr[y, x] - 0.5) * 0.6
            cv.px(x, y, rock[int(clamp(round(v), 0, 3))])
        cv.px(x, t, rock[3])
        cv.px(x, t + 1, rock[2])
    # facets / cracks
    for i in range(60):
        x = int(r.random() * WF)
        t = int(top[x])
        if t >= 48:
            continue
        L = 3 + int(r.random() * 8)
        cv.line(x, t + 2 + int(r.random() * 6), x + int(r.random() * 6) - 3, t + 2 + L, rock[0])
        cv.px(x - 1, t + 3, rock[2])
    # moss / olive turf caps
    for x in range(WF):
        t = int(top[x])
        if t < 44 and top[max(0, x - 1)] < 50 and abs(top[min(WF - 1, x + 1)] - top[max(0, x - 1)]) < 3:
            cv.px(x, t, '#6d7a45')
            cv.px(x, t - 1, '#55623a' if x % 3 else '#6d7a45')
    tufts = [(x, int(top[x])) for x in range(4, WF, 5) if top[x] < 44]
    return _rgba(cv), tuple(tufts)


def _grass_tuft(c, x, y, f, k, seed, colors=('#55623a', '#7d8a4c', '#9aa45e'), hgt=5):
    """Wind-bent grass tuft rooted at (x, y), bending right by wind strength k."""
    for j in range(3):
        h = hgt - j + int(hash01(seed, j) * 2)
        sway = k * (2.2 + 1.2 * math.sin(f * 0.45 + seed * 1.7 + j))
        for i in range(h):
            t = i / max(1, h)
            xx = x + j - 1 + int(round(sway * t * t))
            c.px(xx, y - i, colors[min(2, j)] if i < h - 1 else colors[2])


def _spray(c, f, x, y, t0, seed):
    dt = f - t0
    if dt < 0 or dt > 30:
        return
    for i in range(14):
        a = -math.pi / 2 + (hash01(seed, i) - 0.5) * 1.4
        v = 1.2 + hash01(seed, i, 2) * 1.6
        px_ = x + math.cos(a) * v * dt + 0.06 * dt * dt * (1 + gust_k(f))
        py_ = y + math.sin(a) * v * dt + 0.09 * dt * dt
        if dt > 20 and hash01(seed, i, dt) < 0.5:
            continue
        c.px(int(px_), int(py_), 'white' if dt < 12 else FOAM)


def _smoke(c, f, x, y, seed, n=7, k=1.0, color=('#9aa0a2', '#b5b9b7', '#c9ccc8')):
    """Chimney smoke blown flat to the right by the wind."""
    for i in range(n):
        life = 34
        t = ((f + i * life / n + hash01(seed, i) * 9) % life) / life
        px_ = x + int(t * (16 + 14 * k)) + int(math.sin(t * 6 + i) * 1.5)
        py_ = y - int(t * (6 - 3 * k)) - int(t * t * 2)
        cc = color[0] if t < 0.3 else (color[1] if t < 0.65 else color[2])
        if t > 0.8 and hash01(seed, i, f // 3) < 0.5:
            continue
        c.px(px_, py_, cc)
        if t < 0.6:
            c.px(px_ + 1, py_, cc)
            if t > 0.25:
                c.px(px_, py_ - 1, cc)


def _flag(c, f, x, y, L=9, hgt=5, colors=('#b8402f', '#e0b04a'), k=1.0, tatter=False):
    """Flag on a pole top (x, y), flying right, rippling with the wind."""
    for i in range(L):
        amp = (0.5 + 1.4 * k) * (i / L)
        dy = int(round(math.sin(f * (0.5 + 0.3 * k) - i * 0.8) * amp))
        hh = hgt - (1 if i > L - 3 else 0)
        if tatter and i > L - 4 and (i + f // 4) % 2:
            hh -= 2
        for j in range(max(1, hh)):
            cc = colors[0]
            if j == hgt // 2 and 2 <= i < L - 1:
                cc = colors[1]
            c.px(x + 1 + i, y + j + dy, cc)
        c.px(x + 1 + i, y + dy, mix(colors[0], 'white', 0.25))


def _trottingham(f):
    t = seg(f, 72, 143)
    cam = int(round(PAN * ease_in_out(t)))
    wk = gust_k(f)
    c = Canvas()
    gap_x = int(150 + (f - 72) * 0.42 - cam * 0.25)
    draw_sky(c, f, cam, gap_x, horizon=HORIZON)
    # far mainland + far isles
    far, foy = _far_layers()
    c.blit(far, -int(cam * 0.3), foy)
    _draw_sea(c, f, cam)
    # sun glint on the sea under the shaft
    bx = gap_x + int((SEA_Y - 30) * -0.35)
    for i in range(18):
        gx = bx - 20 + int(hash01(i, 1) * 44)
        gy = HORIZON + 2 + int(hash01(i, 2) * 22)
        if hash01(i, f // 4) < 0.5:
            c.hline(gx, gx + 1 + int(hash01(i, 3) * 3), gy, '#a9b8ae')
    # isle reflection (dark, broken) just below the waterline
    isle, chims, wins, flag, beacon = _isle_layer()
    ix = -cam
    # main isle
    c.blit(isle, ix, ISLE_TOP)
    for j in range(6):
        y = SEA_Y + 2 + j
        src = isle[SEA_Y - ISLE_TOP - 2 - j * 2]
        off = int(round(math.sin(f * 0.3 + j * 1.7) * 1.5))
        m = src[:, 3] > 0
        xs_ = np.nonzero(m)[0] + ix + off
        ok = (xs_ >= 0) & (xs_ < W) & ((xs_ + j + f // 2) % 3 != 0)
        if np.any(ok):
            row = c.a[y, xs_[ok]].astype(np.int16)
            c.a[y, xs_[ok]] = np.clip(row * 0.82 + np.array([8, 14, 14]), 0, 255).astype(np.uint8)
    # lit windows flicker
    for i, (wx, wy) in enumerate(wins):
        if hash01(i, f // 6) < 0.12:
            c.px(wx + ix, wy, WARM[0])
        else:
            c.px(wx + ix, wy, WARM[1] if hash01(i, 9) < 0.5 else WARM[0])
    # beacon blink (every beat)
    bxx, byy = beacon[0] + ix, beacon[1]
    if (f // 9) % 2 == 0:
        c.rect(bxx - 1, byy - 1, 3, 2, '#ffd77a')
        fx.glow(c, bxx, byy, 7, '#ffd77a', strength=0.35, rings=3)
    else:
        c.rect(bxx - 1, byy - 1, 3, 2, '#b07a32')
    # rowboats bobbing in the harbour
    for i, (rbx, rc) in enumerate(((34, '#7a3b30'), (44, '#3f5a6a'), (60, '#6b4a2e'))):
        yy = SEA_Y + 3 + int(round(math.sin(f / 8.0 + i * 2) * 1))
        xx = rbx + ix + int(round(math.sin(f / 13.0 + i) * 1))
        c.hline(xx - 3, xx + 3, yy, rc)
        c.hline(xx - 2, xx + 2, yy + 1, shade(rc, 0.7))
        c.px(xx - 3, yy - 1, rc)
        c.px(xx + 3, yy - 1, rc)
        c.hline(xx - 4, xx + 4, yy + 2, SEA[1])
    # chimney smoke streaming right
    for i, (sx_, sy_) in enumerate(chims):
        _smoke(c, f, sx_ + ix, sy_, 100 + i, n=6, k=wk)
    # castle flag
    _flag(c, f, flag[0] + ix, flag[1] - 8, L=10, hgt=5, k=wk)
    c.vline(flag[0] + ix, flag[1] - 9, flag[1], '#3a3f46')
    c.px(flag[0] + ix, flag[1] - 10, '#e0b04a')
    # fog banks drifting across the isle
    fog = _fog_tex()
    fs = _wrap_slice(fog, int(f * (0.5 + 0.4 * wk) + cam * 0.9), W)
    yy0 = 84
    _fog_blend(c, fs, yy0, '#b9c0bc', 0.6)
    # sun shaft (key light) over the town
    sun_shaft(c, gap_x, 30, SEA_Y + 4, f, strength=0.13)
    # sailing ship
    sx = int(132 + (f - 72) * 0.5) - int(cam * 1.15)
    _draw_ship(c, f, sx, 140)
    _whitecaps(c, f, cam)
    # balloon fighting the wind (sky parallax 0.6)
    bx_ = 268 - int(cam * 0.6) + int(round(math.sin(f / 11.0) * 2 * wk)) - int((f - 72) * 0.15)
    by_ = 62 + int(round(math.sin(f / 7.0) * 2))
    draw_balloon(c, bx_, by_, f, size='s', flame=(f // 18) % 2 == 0)
    # foreground rocks (parallax 1.6) + spray + grass
    rocks, tufts = _fg_rocks()
    fx_ = -int(cam * 1.6) - 60
    c.blit(rocks, fx_, H - 50)
    for i, (tx, ty) in enumerate(tufts):
        X = tx + fx_
        if -4 <= X < W + 4:
            _grass_tuft(c, X, H - 50 + ty, f, wk, i)
    for k in range(4):
        t0 = 72 + 9 + k * 2 * BT
        _spray(c, f, 302 + fx_, H - 26, t0, k)
        _spray(c, f, 116 + fx_, H - 24, t0 + BT, k + 10)
    # foam where the sea meets the rocks
    for i in range(10):
        ph = (f + i * 5) % 20
        for (ex, ey) in ((300, H - 22), (118, H - 20)):
            xx = ex + fx_ + (i - 5) + (1 if ph < 10 else 0) * (-1 if ex > 200 else 1)
            c.px(xx, ey + (i % 3), FOAM if ph < 14 else SEA_HI)
    return c


def _fog_blend(c, fs, y0, color, amount):
    """Soft fog: 3-level banded alpha with dithered band edges."""
    hh = fs.shape[0]
    y1 = min(H, y0 + hh)
    ys = max(0, y0)
    if y1 <= ys:
        return
    fs = fs[ys - y0:y1 - y0]
    lvl = np.clip(np.floor(fs * 3 + (bayer(y1 - ys, W, 0, ys) - 0.5) * 0.9), 0, 3) / 3.0 * amount
    reg = c.a[ys:y1].astype(np.float32)
    reg += (np.array(col(color), np.float32) - reg) * lvl[..., None]
    c.a[ys:y1] = reg.astype(np.uint8)


@lru_cache(maxsize=None)
def _fog_tex():
    hh, TW = 34, 512
    n = _tnoise(hh, TW, (2.5, 18.0), 5)
    yy = np.arange(hh)[:, None] / hh
    env = np.sin(yy * math.pi) ** 1.5
    return np.clip((n - 0.45) * 2.2, 0, 1) * env


# ================================================================== BARS 3-4: GRIFFONSTONE
GW, GH = 600, 440
ROCK = ('#2f2e34', '#3e3c41', '#504d4f', '#64605e', '#7a7570', '#918b83', '#a9a399', '#c2bcb0')
MOSS = ('#4e5130', '#646838', '#7d8245')
OLIVE = ('#36351f', '#4a4828', '#5e5c30', '#76743a', '#8f8d48')
PINE = ('#1d2a28', '#263631', '#31443b', '#3f5446', '#516652')
PLANK = ('#4a3426', '#634532', '#7c5a3e', '#97744f', '#b08f63')
ROOF = ('#3b2a24', '#4f3529', '#664432', '#7b5a3e', '#8d6e4c')
GSTONE = ('#47474c', '#5c5b5e', '#737170', '#8c8984', '#a7a39b')
BR_A = (294, 129)          # rope bridge anchors (deck level, world)
BR_B = (474, 127)
RIVER_Y = 424
CLIFF_X = 286

BRANCHES = [
    [(176, 342, 11), (140, 322, 8), (108, 300, 6), (90, 284, 3), (84, 272, 1)],
    [(124, 313, 4), (117, 298, 3), (114, 287, 1)],
    [(176, 252, 9), (144, 234, 7), (116, 216, 5), (100, 200, 3), (95, 190, 1)],
    [(178, 162, 7), (154, 147, 5), (134, 135, 3), (126, 123, 1)],
    [(206, 362, 10), (238, 344, 8), (262, 332, 5), (276, 320, 2), (280, 311, 1)],
    [(204, 268, 9), (234, 254, 7), (256, 240, 4), (264, 226, 1)],
    [(242, 251, 3), (250, 237, 2), (252, 229, 1)],
    [(200, 178, 9), (232, 158, 7), (262, 142, 6), (282, 136, 6), (296, 135, 5)],
    [(198, 106, 6), (218, 92, 4), (230, 82, 2), (234, 73, 1)],
    [(182, 120, 4), (170, 110, 3), (163, 101, 1)],
]


def _spire_cx(y):
    return 190 + 6 * math.sin(y / 55.0)


def _spire_hw(y):
    hw = 15 + (y - 66) * 0.055 + 2.4 * math.sin(y / 23.0) + 1.2 * math.sin(y / 7.0)
    hw += 58 * math.exp((y - GH) / 20.0)
    if y >= 60:
        hw += 36 * math.exp(-(y - 60) / 7.0)
    return hw


def _branch_path(b):
    """Dense samples (x, y, r) along a branch polyline."""
    out = []
    for (x0, y0, r0), (x1, y1, r1) in zip(b, b[1:]):
        L = max(1, int(math.hypot(x1 - x0, y1 - y0) * 2))
        for i in range(L):
            t = i / L
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r0 + (r1 - r0) * t))
    out.append(b[-1])
    return out


def _pine(cv, x, base, h, seed, pal=PINE, lean=1):
    """Layered conifer (static). (x, base) = trunk foot."""
    r = rng(seed)
    cv.vline(x, base - 3, base, PLANK[0])
    tiers = max(3, h // 4)
    for i in range(tiers):
        t = i / tiers
        ty = base - 2 - int(t * (h - 3))
        hw = int((1 - t) * h * 0.32) + 1
        sx = int(round(lean * t * 1.5))
        for j in range(4):
            yy = ty - j
            w2 = max(0, hw - j * (hw // 3 + 1) // 2)
            if w2 <= 0 and j > 0:
                break
            cv.hline(x - w2 + sx, x + w2 + sx, yy, pal[1])
            cv.hline(x - w2 + sx, x + sx, yy, pal[2])
            if w2 > 1:
                cv.px(x - w2 + sx, yy, pal[3])
            cv.px(x + w2 + sx, yy, pal[0])
        if r.random() < 0.6:
            cv.px(x - hw + sx + 1, ty + 1, pal[3])
    cv.px(x + int(round(lean * 1.5)), base - h, pal[2])


def _crag(cv, x, base, w, h, seed, pal=OLIVE):
    """Blocky olive crag with a grassy cap and grey rock faces."""
    r = rng(seed)
    top = base - h
    pts = [(x, base), (x + 1, top + 3), (x + 3, top), (x + w - 4, top + 1), (x + w - 1, top + 4), (x + w, base)]
    cv.poly(pts, GSTONE[1])
    cv.poly([(x + 1, base), (x + 2, top + 3), (x + 4, top + 1), (x + w // 2, top + 2), (x + w // 2 - 2, base)],
            GSTONE[2])
    cv.vline(x + 1, top + 3, base, GSTONE[3])
    for i in range(max(2, w // 6)):
        fx_ = x + 3 + int(r.random() * (w - 6))
        cv.vline(fx_, top + 3 + int(r.random() * 3), base, GSTONE[0])
        cv.vline(fx_ - 1, top + 4 + int(r.random() * 3), base - 2, GSTONE[2])
    cv.hline(x + 2, x + w - 3, top, pal[3])
    cv.hline(x + 1, x + w - 2, top + 1, pal[2])
    for xx in range(x + 1, x + w - 1):
        if r.random() < 0.5:
            cv.px(xx, top + 2, pal[1])
        if r.random() < 0.25:
            cv.px(xx, top - 1, pal[4])


def _gs_house(cv, x, base, w, wall, roof, seed, stilts=0):
    """Ramshackle stone-and-plank house. Returns (chimney top, [windows])."""
    r = rng(seed)
    y0 = base - wall
    sh = wall // 2
    # stilts / struts below
    if stilts > 0:
        for sx in (x + 1, x + w - 2):
            cv.vline(sx, base, base + stilts, PLANK[0])
        cv.line(x + 1, base + stilts, x + w // 2, base, PLANK[1])
        cv.line(x + w - 2, base + stilts, x + w // 2, base, PLANK[0])
    cv.hline(x - 1, x + w, base, PLANK[1])
    # stone ground storey
    cv.rect(x, base - sh, w, sh, GSTONE[2])
    for yy in range(base - sh, base, 2):
        for xx in range(x + (yy // 2) % 2 * 2, x + w, 4):
            cv.px(xx, yy, GSTONE[1])
    cv.vline(x, base - sh, base - 1, GSTONE[3])
    cv.rect(x + w - 2, base - sh, 2, sh, GSTONE[1])
    # plank upper storey (slightly overhanging, a bit crooked)
    lean = 1 if r.random() < 0.4 else 0
    for yy in range(y0, base - sh):
        off = lean if yy < y0 + (wall - sh) // 2 else 0
        cv.hline(x - 1 + off, x + w + off, yy, PLANK[2])
    for xx in range(x, x + w + 1, 3):
        cv.vline(xx, y0, base - sh - 1, PLANK[1])
    cv.vline(x - 1, y0, base - sh - 1, PLANK[3])
    cv.vline(x + w, y0, base - sh - 1, PLANK[0])
    cv.hline(x - 1, x + w, base - sh, PLANK[0])
    # patch board
    if w >= 10:
        px_ = x + 2 + int(r.random() * (w - 6))
        cv.rect(px_, y0 + 1, 3, 2, PLANK[4])
    wins = []
    # windows with shutters (one askew)
    nw = max(1, (w - 2) // 6)
    for i in range(nw):
        wx = x + 2 + i * 6 + (1 if w > 12 else 0)
        wy = y0 + 1 + (wall - sh - 4) // 2
        lit = r.random() < 0.6
        cv.rect(wx, wy, 3, 3, WARM[0] if lit else '#2f3035')
        if lit:
            cv.px(wx, wy, WARM[1])
            wins.append((wx + 1, wy + 1))
        cv.vline(wx - 1, wy, wy + 2, PLANK[1])
        if r.random() < 0.5:
            cv.line(wx + 3, wy, wx + 4, wy + 3, PLANK[1])
        else:
            cv.vline(wx + 3, wy, wy + 2, PLANK[1])
    # round-topped door
    dx = x + w // 2 - 1 + int((r.random() - 0.5) * (w // 3))
    dh = min(6, sh)
    cv.rect(dx, base - dh, 3, dh, PLANK[0])
    cv.px(dx + 1, base - dh - 1, PLANK[0])
    cv.px(dx + 2, base - dh // 2, PLANK[3])
    # sagging plank roof (lean-to gable), patched
    ry0 = y0 - roof
    for i in range(roof):
        yy = ry0 + i
        inset = max(0, roof - 1 - i) * 1
        xa, xb = x - 2 + inset, x + w + 1 - inset // 2
        cv.hline(xa, xb, yy, ROOF[2] if i % 2 == 0 else ROOF[1])
        if i == 0:
            cv.hline(xa, xb, yy, ROOF[3])
        for tx in range(xa + (i % 2) * 2, xb, 4):
            cv.px(tx, yy, ROOF[0])
    cv.hline(x - 2, x + w + 1, y0 - 1, ROOF[0])
    if r.random() < 0.7:
        pxx = x + int(r.random() * (w - 3))
        cv.rect(pxx, ry0 + 1, 3, 2, ROOF[4])
    chim = None
    if r.random() < 0.8:
        cx_ = x + w - 4 - int(r.random() * 3)
        cv.rect(cx_, ry0 - 3, 3, 4, GSTONE[1])
        cv.vline(cx_, ry0 - 3, ry0, GSTONE[3])
        cv.hline(cx_ - 1, cx_ + 3, ry0 - 4, GSTONE[0])
        chim = (cx_ + 1, ry0 - 5)
    return chim, wins


def _ruin(cv):
    """Ruined castle on the crown + empty throne terrace. Returns banner pole tops and windows."""
    s = GSTONE
    top = 62
    # main hall wall with broken battlements
    hx0, hx1 = 166, 224
    for x in range(hx0, hx1):
        h = 22 + int(3 * math.sin(x * 0.7)) - (8 if 190 <= x < 199 else 0) - (5 if 208 <= x < 213 else 0)
        cv.vline(x, top - h, top - 1, s[2])
        if (x - hx0) % 4 < 2 and not (190 <= x < 199):
            cv.vline(x, top - h - 3, top - h - 1, s[2])
            cv.px(x, top - h - 3, s[4])
        cv.px(x, top - h, s[3])
    cv.rect(hx1 - 9, top - 24, 9, 24, s[1])
    # stone courses
    for yy in range(top - 20, top - 1, 3):
        for xx in range(hx0 + 1 + (yy // 3) % 2 * 3, hx1 - 1, 6):
            cv.px(xx, yy, s[1])
    # big broken arch window (dark, open to the sky behind = gloomy)
    cv.rect(181, top - 16, 9, 12, '#2c2c31')
    cv.rect(182, top - 18, 7, 2, '#2c2c31')
    cv.rect(183, top - 19, 5, 1, '#2c2c31')
    cv.hline(181, 189, top - 4, s[3])
    cv.px(186, top - 13, '#4a4a52')
    # small lit window
    cv.rect(203, top - 13, 2, 4, WARM[0])
    wins = [(203, top - 12)]
    # left tall tower with a jagged broken top
    tx0, tx1 = 158, 171
    for x in range(tx0, tx1):
        h = 40 + [0, 2, 5, 3, 7, 1, 4, 8, 5, 2, 6, 3, 0][x - tx0]
        cv.vline(x, top - h, top - 1, s[2] if x < tx1 - 4 else s[1])
        cv.px(x, top - h, s[4] if x < tx1 - 4 else s[2])
    cv.vline(tx0, top - 40, top - 1, s[3])
    cv.vline(tx1 - 1, top - 34, top - 1, s[0])
    for yy in (top - 32, top - 22):
        cv.rect(163, yy, 2, 5, '#2c2c31')
        cv.px(163, yy - 1, s[3])
    for yy in range(top - 36, top - 2, 4):
        cv.px(160 + (yy // 4) % 3, yy, s[1])
    # right short tower
    rx0 = 222
    cv.rect(rx0, top - 30, 10, 30, s[1])
    cv.vline(rx0, top - 30, top - 1, s[2])
    cv.vline(rx0 + 9, top - 30, top - 1, s[0])
    for mx in (rx0, rx0 + 4, rx0 + 8):
        cv.rect(mx, top - 33, 2, 3, s[1])
    cv.rect(rx0 + 1, top - 33, 1, 1, s[3])
    cv.rect(rx0 + 4, top - 24, 2, 4, '#2c2c31')
    # rubble on the crown
    r = rng(44)
    for i in range(14):
        xx = 150 + int(r.random() * 92)
        cv.rect(xx, top - 2, 2 + int(r.random() * 2), 2, s[1 + int(r.random() * 3)])
    # throne terrace jutting out to the left on corbels
    ty = top - 2
    cv.rect(118, ty, 42, 3, s[2])
    cv.hline(118, 159, ty, s[4])
    cv.hline(118, 159, ty + 2, s[0])
    for cx_ in range(121, 158, 7):
        cv.tri((cx_, ty + 3), (cx_ + 4, ty + 3), (cx_ + 3, ty + 7), s[1])
        cv.px(cx_ + 1, ty + 3, s[3])
    # balustrade (some posts missing)
    for bx in range(119, 158, 3):
        if bx in (128, 143):
            continue
        cv.vline(bx, ty - 4, ty - 1, s[3])
    cv.hline(118, 125, ty - 5, s[3])
    cv.hline(131, 141, ty - 5, s[3])
    cv.hline(146, 159, ty - 5, s[3])
    # the empty throne: tall stone back with a shield emblem, steps
    thx = 138
    cv.rect(thx - 4, ty - 2, 11, 2, s[3])           # step
    cv.rect(thx - 3, ty - 4, 9, 2, s[2])
    cv.rect(thx - 3, ty - 18, 2, 14, s[2])           # back post L
    cv.rect(thx + 4, ty - 18, 2, 14, s[1])
    cv.rect(thx - 1, ty - 16, 5, 9, s[1])            # back panel
    cv.rect(thx - 3, ty - 9, 9, 2, s[3])             # seat
    cv.rect(thx - 4, ty - 12, 1, 5, s[3])            # arm rests
    cv.rect(thx + 6, ty - 12, 1, 5, s[2])
    cv.px(thx - 3, ty - 19, s[4])
    cv.px(thx + 5, ty - 19, s[3])
    # shield emblem on the back (dull gold, no creatures)
    cv.rect(thx, ty - 15, 3, 3, '#a8883e')
    cv.px(thx + 1, ty - 12, '#a8883e')
    cv.px(thx, ty - 15, '#c9a85a')
    # banner poles (tattered banners are dynamic)
    poles = [(123, ty - 22), (154, ty - 22)]
    for (px_, py_) in poles:
        cv.vline(px_, py_, ty - 1, PLANK[0])
        cv.px(px_, py_ - 1, '#a8883e')
    return poles, wins


def _train(cv, X, Y):
    """Compact Friendship Express facing right; (X, Y) = rail level at the rear of the last coach."""
    def wheel(cx, cy, r):
        cv.circle(cx, cy, r, 'eggplant')
        cv.circle(cx, cy, r - 1, 'steel_dk')
        cv.px(cx, cy, 'steel_hi')
    for k, (body, low, roofc) in enumerate((('cotton', 'bubblegum', 'periwinkle'),
                                             ('bubblegum', 'hot_pink', 'royal_lt'))):
        x0 = X + k * 30
        cv.rect(x0, Y - 21, 28, 17, 'berry')
        cv.rect(x0 + 1, Y - 20, 26, 9, body)
        cv.rect(x0 + 1, Y - 11, 26, 7, low)
        cv.hline(x0 + 1, x0 + 26, Y - 11, 'gold')
        for wx in range(x0 + 3, x0 + 25, 6):
            cv.rect(wx, Y - 18, 4, 5, 'raspberry')
            cv.rect(wx + 1, Y - 17, 2, 3, WARM[0])
            cv.px(wx + 1, Y - 17, WARM[1])
        cv.rect(x0 - 1, Y - 24, 30, 3, roofc)
        cv.hline(x0, x0 + 27, Y - 25, mix(roofc, 'white', .4))
        cv.hline(x0 - 1, x0 + 28, Y - 22, shade(roofc, .7))
        cv.rect(x0 + 2, Y - 4, 24, 2, 'eggplant')
        for wx in (x0 + 5, x0 + 9, x0 + 19, x0 + 23):
            wheel(wx, Y - 2, 2)
        cv.rect(x0 + 28, Y - 9, 2, 2, 'steel_dk')
    X2 = X + 60
    # tender
    cv.rect(X2, Y - 17, 13, 13, 'berry')
    cv.rect(X2 + 1, Y - 16, 11, 12, 'hot_pink')
    cv.hline(X2 + 1, X2 + 11, Y - 11, 'gold')
    for k in range(4):
        cv.circle(X2 + 3 + k * 2, Y - 18, 1, 'charcoal')
    for wx in (X2 + 3, X2 + 9):
        wheel(wx, Y - 2, 2)
    # locomotive cab + boiler
    L = X2 + 14
    cv.rect(L, Y - 25, 10, 21, 'berry')
    cv.rect(L + 1, Y - 24, 8, 20, 'hot_pink')
    cv.vline(L + 1, Y - 24, Y - 5, 'bubblegum')
    cv.rect(L + 2, Y - 22, 5, 5, 'berry')
    cv.rect(L + 3, Y - 21, 3, 3, WARM[0])
    cv.rect(L - 1, Y - 27, 12, 2, 'violet')
    cv.hline(L - 1, L + 10, Y - 28, 'lavender')
    for y in range(Y - 18, Y - 7):
        t = (y - (Y - 18)) / 10
        cc = 'cotton' if t < 0.15 else 'bubblegum' if t < 0.35 else 'hot_pink' if t < 0.7 else 'rose'
        cv.hline(L + 10, L + 26, y, cc)
    cv.hline(L + 10, L + 26, Y - 19, 'berry')
    for gx in (L + 14, L + 20):
        cv.vline(gx, Y - 18, Y - 8, 'gold')
    cv.rect(L + 26, Y - 19, 4, 12, 'plum')
    cv.vline(L + 27, Y - 18, Y - 8, 'berry')
    cv.px(L + 28, Y - 14, 'gold')
    cv.rect(L + 27, Y - 22, 3, 3, 'gold')
    cv.px(L + 28, Y - 21, 'butter')
    cv.rect(L + 21, Y - 26, 3, 7, 'raspberry')         # funnel
    cv.rect(L + 20, Y - 29, 5, 3, 'magenta')
    cv.hline(L + 20, L + 24, Y - 29, 'gold')
    cv.ellipse(L + 16, Y - 20, 2, 1, 'gold')
    cv.rect(L + 6, Y - 7, 26, 3, 'eggplant')
    cv.poly([(L + 30, Y - 7), (L + 33, Y), (L + 29, Y)], 'amber')
    for wx in (L + 9, L + 16, L + 23):
        wheel(wx, Y - 3, 3)
    cv.line(L + 9, Y - 4, L + 23, Y - 4, 'steel_hi')
    return (L + 22, Y - 30), (L + 4, Y - 3)


@lru_cache(maxsize=None)
def _gs_world():
    """Main layer (parallax 1): spire, houses, castle ruin, base crags, station, far pinnacle."""
    from scipy.ndimage import binary_erosion
    yy, xx = np.mgrid[0:GH, 0:GW]
    M = np.zeros((GH, GW), bool)
    cxs = np.array([_spire_cx(y) for y in range(GH)])
    hws = np.array([_spire_hw(y) if y >= 60 else -1 for y in range(GH)])
    trunk = (np.abs(xx - cxs[:, None]) <= hws[:, None]) & (yy >= 60)
    M |= trunk
    for b in BRANCHES:
        for (bx, by, br) in _branch_path(b):
            x0, x1 = int(bx - br - 1), int(bx + br + 2)
            y0, y1 = int(by - br - 1), int(by + br + 2)
            sub = (xx[y0:y1, x0:x1] - bx) ** 2 + (yy[y0:y1, x0:x1] - by) ** 2 <= br * br + 0.5
            M[y0:y1, x0:x1] |= sub
    M[:, CLIFF_X:] = False
    # shading from a blurred-mask normal + cylinder term + texture
    G = gaussian_filter(M.astype(np.float32), 2.6)
    gyy, gxx = np.gradient(G)
    nx, ny = -gxx * 7, -gyy * 7
    nn = np.sqrt(nx * nx + ny * ny + 1)
    Lx, Ly, Lz = -0.62, -0.5, 0.6
    ln = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    lum = (nx * Lx + ny * Ly + Lz) / nn / ln
    u = np.clip((xx - cxs[:, None]) / np.maximum(hws[:, None], 1), -1.5, 1.5)
    u = np.where(trunk, u, 0)
    blot = _tnoise(GH, GW, 2.5, 61)
    streak = _tnoise(GH, GW, (7.0, 0.7), 62)
    v = lum * 6.0 - u * 1.3 + (blot - 0.5) * 1.6 + 0.2 + (bayer(GH, GW) - 0.5) * 0.8
    k = np.clip(np.floor(v), 0, 7).astype(int)
    k = np.where(trunk & (streak < 0.22), np.maximum(k - 2, 0), k)
    k = np.where(trunk & (streak > 0.80), np.minimum(k + 1, 7), k)
    pal = np.array([col(c_) for c_ in ROCK], np.uint8)
    rgb = pal[k]
    moss = M & (ny < -0.25) & (blot > 0.45)
    rgb[moss] = col(MOSS[1])
    rgb[moss & (bayer(GH, GW) < 0.4)] = col(MOSS[2])
    edge = M & ~binary_erosion(M)
    rgb[edge] = col(ROCK[0])
    rgb[edge & (nx < -0.3) & (ny < 0.2)] = col(ROCK[2])
    cv = _layer(GW, GH)
    cv.a[M] = rgb[M]
    # branch fissures along the arms
    for b in BRANCHES:
        path = _branch_path(b)
        for i in range(0, len(path) - 2, 1):
            x0, y0, r0 = path[i]
            x1, y1, _ = path[i + 1]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1
            pxn, pyn = -dy / L, dx / L
            for off in (0.45, -0.1):
                if r0 < 3 or hash01(int(x0), int(y0), off) < 0.2:
                    continue
                X, Y = int(round(x0 + pxn * r0 * off)), int(round(y0 + pyn * r0 * off))
                if 0 <= X < CLIFF_X and M[Y, X]:
                    cv.px(X, Y, ROCK[2] if off > 0 else ROCK[5])
    # knot holes on the trunk ("dead tree")
    for (kx, ky, kr) in ((184, 302, 3), (199, 214, 3), (183, 132, 2), (196, 388, 4), (178, 248, 2)):
        cv.ellipse(kx, ky, kr, kr + 1, ROCK[1])
        cv.ellipse(kx, ky - 1, max(1, kr - 1), kr, ROCK[0])
        cv.hline(kx - kr + 1, kx + kr - 1, ky + kr + 1, ROCK[5])
    # long cracks
    r = rng(71)
    for i in range(7):
        y = 90 + int(r.random() * 300)
        x = int(_spire_cx(y) + (r.random() - 0.5) * _spire_hw(y) * 1.2)
        for j in range(18 + int(r.random() * 20)):
            if not (0 <= x < CLIFF_X and 0 <= y < GH) or not M[y, x]:
                break
            cv.px(x, y, ROCK[1])
            if x - 1 >= 0 and M[y, x - 1]:
                cv.px(x - 1, y, ROCK[5])
            y += 1
            x += int(round((r.random() - 0.5) * 1.6))
    meta = dict(chims=[], wins=[], ladders=[], lanterns=[])
    # castle ruin + throne terrace
    poles, wns = _ruin(cv)
    meta['poles'] = poles
    meta['wins'] += wns
    # houses on the branches: (x, base, w, wall, roof, seed, stilts)
    houses = [(98, 291, 14, 12, 5, 1, 4), (126, 309, 16, 13, 6, 2, 4), (134, 225, 15, 12, 5, 4, 4),
              (138, 140, 12, 11, 5, 5, 2), (228, 343, 16, 13, 6, 6, 3), (252, 330, 13, 11, 5, 7, 3),
              (222, 252, 15, 12, 5, 8, 3), (220, 160, 14, 12, 5, 9, 3), (160, 200, 12, 11, 5, 10, 0),
              (208, 300, 13, 12, 5, 11, 0), (162, 380, 14, 12, 6, 12, 0), (245, 145, 12, 10, 4, 13, 3)]
    for (hx, base, w, wall, roof, sd, st) in houses:
        if sd in (10, 11, 12):          # clinging to the trunk: plank platform on brackets
            cv.hline(hx - 2, hx + w + 1, base, PLANK[1])
            cv.hline(hx - 2, hx + w + 1, base + 1, PLANK[0])
            for bx in (hx, hx + w - 1):
                cv.line(bx, base + 1, bx + (3 if bx < hx + w // 2 else -3), base + 5, PLANK[0])
        ch, wn = _gs_house(cv, hx, base, w, wall, roof, sd, stilts=st)
        if ch:
            meta['chims'].append(ch)
        meta['wins'] += wn
    # library tower on the left arm (round, conical roof, lit window, book sign)
    lx, lb = 102, 214
    cv.rect(lx, lb - 26, 12, 26, GSTONE[2])
    cv.vline(lx, lb - 26, lb - 1, GSTONE[3])
    cv.rect(lx + 9, lb - 26, 3, 26, GSTONE[1])
    for yy_ in range(lb - 24, lb - 1, 3):
        for xx_ in range(lx + 1 + (yy_ // 3) % 2 * 2, lx + 9, 4):
            cv.px(xx_, yy_, GSTONE[1])
    for i in range(12):
        half = 8 - (i * 8) // 12
        cv.hline(lx + 6 - half, lx + 5 + half, lb - 27 - i, ROOF[2] if i % 2 else ROOF[1])
        cv.px(lx + 6 - half, lb - 27 - i, ROOF[3])
    cv.px(lx + 5, lb - 39, '#a8883e')
    cv.rect(lx + 4, lb - 20, 3, 5, WARM[0])
    cv.px(lx + 5, lb - 21, WARM[0])
    cv.px(lx + 4, lb - 20, WARM[1])
    meta['wins'].append((lx + 5, lb - 18))
    cv.rect(lx + 4, lb - 7, 4, 7, PLANK[0])
    # book sign (icon)
    cv.hline(lx + 12, lx + 15, lb - 13, PLANK[0])
    cv.rect(lx + 13, lb - 12, 6, 4, PLANK[1])
    cv.rect(lx + 14, lb - 11, 2, 2, '#e9e2cf')
    cv.rect(lx + 16, lb - 11, 2, 2, '#d9cfb6')
    cv.px(lx + 15, lb - 11, '#7a3b30')
    # scone stall on the right arm (striped awning, scone icon)
    sx0, sb = 238, 248
    cv.rect(sx0, sb - 6, 10, 6, PLANK[2])
    cv.hline(sx0, sx0 + 9, sb - 6, PLANK[4])
    for i in range(12):
        cv.vline(sx0 - 1 + i, sb - 12, sb - 10, '#a8463a' if (i // 2) % 2 else '#d9cfb6')
    cv.vline(sx0, sb - 10, sb - 6, PLANK[0])
    cv.vline(sx0 + 9, sb - 10, sb - 6, PLANK[0])
    for i in range(3):
        cv.rect(sx0 + 1 + i * 3, sb - 8, 2, 2, '#c99a5a')
        cv.px(sx0 + 1 + i * 3, sb - 8, '#e8c58a')
    # walkways (plank bridges with rope rails) + ladders + lanterns
    walks = [((150, 214), (172, 202)), ((236, 252), (256, 246)), ((234, 158), (292, 130)),
             ((112, 290), (176, 288))]
    for (a, b) in walks[:2]:
        n = max(2, int(abs(b[0] - a[0])))
        for i in range(n + 1):
            t = i / n
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t + math.sin(t * math.pi) * 2))
            cv.px(x, y, PLANK[3] if i % 3 else PLANK[1])
            cv.px(x, y + 1, PLANK[0])
            if i % 4 == 0:
                cv.vline(x, y - 4, y - 1, PLANK[1])
            cv.px(x, y - 4, '#8a7a5a')
    # plank path along the long right arm to the bridge platform
    path = _branch_path(BRANCHES[7])
    for (bx, by, br) in path[::2]:
        if bx < 240:
            continue
        X, Y = int(round(bx)), int(round(by - br))
        cv.px(X, Y - 1, PLANK[3])
        cv.px(X, Y, PLANK[1])
    # bridge platform at the arm's end
    ax, ay = BR_A
    cv.rect(ax - 12, ay, 14, 2, PLANK[2])
    cv.hline(ax - 12, ax + 1, ay, PLANK[4])
    cv.hline(ax - 12, ax + 1, ay + 2, PLANK[0])
    for px_ in (ax - 11, ax - 1):
        cv.rect(px_, ay - 10, 2, 10, PLANK[1])
        cv.vline(px_, ay - 10, ay - 1, PLANK[3])
        cv.hline(px_, px_ + 1, ay - 7, '#b9a57a')
        cv.hline(px_, px_ + 1, ay - 5, '#b9a57a')
    meta['lanterns'] += [(ax - 11, ay - 11), (230, 246), (152, 212), (98, 283)]
    meta['ladders'] += [(118, 216, 70), (226, 252, 82), (176, 140, 54), (252, 330, 40), (148, 309, 30)]
    # ---------- base: olive crags, pines, station, rails, river, cliff edge
    for (x, base, w, h, sd) in ((0, 418, 40, 34, 1), (34, 418, 30, 22, 2), (96, 410, 44, 30, 3),
                                (230, 412, 40, 26, 4), (254, 420, 32, 40, 5), (60, 418, 26, 14, 6),
                                (130, 400, 30, 18, 7), (266, 404, 20, 52, 8)):
        _crag(cv, x, base, w, h, sd)
    for (x, base, h, sd) in ((12, 386, 26, 1), (24, 392, 20, 2), (48, 398, 22, 3), (88, 382, 28, 4),
                             (104, 386, 18, 5), (118, 382, 24, 6), (144, 384, 16, 7), (240, 388, 22, 8),
                             (258, 382, 26, 9), (276, 354, 22, 10), (6, 400, 16, 11), (72, 404, 14, 12),
                             (268, 316, 14, 13), (90, 300, 12, 14), (132, 316, 11, 15)):
        _pine(cv, x, base, h, sd)
    # station: platform, house, water tower, rails, train
    cv.rect(0, 414, CLIFF_X - 4, 2, '#5c4636')                # rail bed
    for x in range(0, CLIFF_X - 6, 4):
        cv.rect(x, 415, 3, 2, PLANK[0])                         # sleepers
    cv.hline(0, CLIFF_X - 8, 414, 'steel')
    cv.hline(0, CLIFF_X - 8, 413, 'steel_hi')
    # platform (stone edge + planks) behind the rails
    cv.rect(150, 406, 124, 7, GSTONE[2])
    cv.hline(150, 273, 406, GSTONE[4])
    cv.hline(150, 273, 407, GSTONE[3])
    for x in range(152, 273, 6):
        cv.vline(x, 408, 412, GSTONE[1])
    # station house (stone, slate roof, clock cupola, chimney)
    hx, hb = 226, 406
    cv.rect(hx, hb - 22, 40, 22, GSTONE[2])
    cv.vline(hx, hb - 22, hb - 1, GSTONE[3])
    cv.rect(hx + 32, hb - 22, 8, 22, GSTONE[1])
    for yy_ in range(hb - 20, hb - 1, 3):
        for xx_ in range(hx + 1 + (yy_ // 3) % 2 * 3, hx + 38, 6):
            cv.px(xx_, yy_, GSTONE[1])
    for wx in (hx + 4, hx + 14, hx + 26):
        cv.rect(wx, hb - 16, 5, 7, '#3b3a40')
        cv.rect(wx + 1, hb - 15, 3, 5, WARM[0])
        cv.px(wx + 1, hb - 15, WARM[1])
        cv.hline(wx, wx + 4, hb - 17, GSTONE[4])
        meta['wins'].append((wx + 2, hb - 13))
    cv.rect(hx + 20, hb - 12, 4, 12, PLANK[0])
    for i in range(9):
        cv.hline(hx - 3 + i, hx + 42 - i, hb - 23 - i, '#454c5c' if i % 2 else '#566073')
    cv.hline(hx - 3, hx + 42, hb - 23, '#3a404d')
    cv.rect(hx + 16, hb - 38, 8, 7, GSTONE[3])
    cv.circle(hx + 20, hb - 35, 2, '#e9e2cf')
    cv.px(hx + 20, hb - 36, PLANK[0])
    cv.px(hx + 21, hb - 35, PLANK[0])
    cv.tri((hx + 15, hb - 38), (hx + 25, hb - 38), (hx + 20, hb - 43), '#454c5c')
    cv.rect(hx + 32, hb - 36, 3, 6, GSTONE[1])
    meta['chims'].append((hx + 33, hb - 37))
    # station sign with a train icon
    cv.rect(hx - 14, hb - 18, 11, 7, PLANK[1])
    cv.rect(hx - 13, hb - 17, 9, 5, '#e9e2cf')
    cv.rect(hx - 12, hb - 16, 5, 2, '#a8463a')
    cv.rect(hx - 8, hb - 17, 2, 1, '#a8463a')
    cv.px(hx - 11, hb - 13, PLANK[0])
    cv.px(hx - 8, hb - 13, PLANK[0])
    cv.vline(hx - 10, hb - 11, hb - 1, PLANK[0])
    # lamp posts on the platform
    for lx_ in (160, 206):
        cv.vline(lx_, hb - 16, hb - 1, '#3a3f46')
        cv.rect(lx_ - 1, hb - 19, 3, 3, '#3a3f46')
        meta['lanterns'].append((lx_, hb - 18))
    # buffer stop at the cliff edge
    cv.rect(CLIFF_X - 10, 407, 4, 7, '#7a3b30')
    cv.hline(CLIFF_X - 11, CLIFF_X - 6, 407, '#a8463a')
    # water tower
    wx_, wb = 62, 412
    for lx_ in (wx_, wx_ + 9):
        cv.vline(lx_, wb - 14, wb, PLANK[0])
    cv.line(wx_, wb - 2, wx_ + 9, wb - 12, PLANK[0])
    cv.rect(wx_ - 2, wb - 26, 14, 12, PLANK[2])
    for xx_ in range(wx_ - 2, wx_ + 12, 3):
        cv.vline(xx_, wb - 26, wb - 15, PLANK[1])
    cv.hline(wx_ - 2, wx_ + 11, wb - 22, '#3a3f46')
    cv.hline(wx_ - 2, wx_ + 11, wb - 17, '#3a3f46')
    cv.tri((wx_ - 3, wb - 26), (wx_ + 12, wb - 26), (wx_ + 4, wb - 31), ROOF[1])
    # the train
    funnel, cyl = _train(cv, 136, 413)
    meta['funnel'] = funnel
    meta['cyl'] = cyl
    # river embankment
    cv.rect(0, 417, CLIFF_X - 2, RIVER_Y - 417, GSTONE[1])
    cv.hline(0, CLIFF_X - 3, 417, GSTONE[3])
    for x in range(0, CLIFF_X - 2, 5):
        cv.px(x, 419 + (x // 5) % 3, GSTONE[0])
    # near cliff face into the abyss
    for y in range(300, GH):
        xe = CLIFF_X - 1 - int(2 * math.sin(y / 9.0) + (y - 300) * 0.02)
        cv.hline(xe - 3, xe, y, ROCK[1])
        cv.px(xe, y, ROCK[0])
        cv.px(xe - 3, y, ROCK[3])
    # ---------- far pinnacle (bridge's far anchor) rising from the abyss
    bx, by = BR_B
    for y in range(by - 2, GH):
        t = (y - by) / (GH - by)
        hw = 34 + 14 * t + 4 * math.sin(y / 11.0) + 2 * math.sin(y / 4.0)
        cx = 512 + 8 * math.sin(y / 40.0)
        x0, x1 = int(cx - hw), int(cx + hw)
        for x in range(max(0, x0), min(GW, x1)):
            uu = (x - x0) / max(1, x1 - x0)
            kk = 5 - int(uu * 4.5) + int((blot[y, x] - 0.5) * 2.5) - int(t * 1.5)
            if streak[y, x] < 0.22:
                kk -= 2
            cv.px(x, y, ROCK[int(clamp(kk, 0, 7))])
        cv.px(x0, y, ROCK[1])
        cv.px(x1 - 1, y, ROCK[0])
    cv.hline(bx - 2, 556, by - 2, OLIVE[3])
    cv.hline(bx - 2, 556, by - 1, OLIVE[2])
    for x in range(bx, 556):
        if hash01(x, 3) < 0.5:
            cv.px(x, by, OLIVE[1])
    for (x, h, sd) in ((500, 22, 21), (512, 28, 22), (526, 20, 23), (540, 26, 24), (552, 18, 25)):
        _pine(cv, x, by - 2, h, sd)
    for px_ in (bx, bx + 10):
        cv.rect(px_, by - 12, 2, 10, PLANK[1])
        cv.vline(px_, by - 12, by - 3, PLANK[3])
        cv.hline(px_, px_ + 1, by - 9, '#b9a57a')
    cv.rect(bx - 1, by - 2, 14, 2, PLANK[2])
    meta['lanterns'].append((bx + 10, by - 13))
    return _rgba(cv), meta


@lru_cache(maxsize=None)
def _gs_far():
    """Far ranges (parallax 0.25): misty blue-grey mountains over hazy lowland forest."""
    FW, FH = 420, 270
    cv = _layer(FW, FH)
    xs = np.arange(FW)
    ridge = 118 + 14 * np.sin(xs / 31.0) + 8 * np.sin(xs / 13.0 + 2) + 4 * np.abs(np.sin(xs / 5.0))
    ridge2 = 132 + 10 * np.sin(xs / 21.0 + 1) + 5 * np.sin(xs / 8.0)
    for x in range(FW):
        cv.vline(x, int(ridge[x]), FH - 1, '#8e979d')
        cv.px(x, int(ridge[x]), '#a3abaf')
        cv.vline(x, int(ridge2[x]), FH - 1, '#7f888c')
        cv.px(x, int(ridge2[x]), '#939b9d')
    cv.gradient(0, 150, FW, FH - 150, ['#7d8682', '#6f7a72', '#66705f'], soft=0.8)
    # far forest texture
    r = rng(3)
    for i in range(500):
        x = int(r.random() * FW)
        y = 152 + int(r.random() * (FH - 152))
        cv.px(x, y, '#5d6858')
        cv.px(x, y - 1, '#6c776a')
    return _rgba(cv)


@lru_cache(maxsize=None)
def _gs_mid():
    """Mid-far layer (parallax 0.5): stacked olive crag ridges with pine-covered tops (hazy at the back)."""
    MW, MH = 480, 340
    cv = _layer(MW, MH)
    ridges = [  # base y, amplitude, seed, (shade, body, light, rim), pine palette
        (140, 16, 1, ('#6c7368', '#767d70', '#848b7b', '#959b88'), ('#596356', '#616b5d', '#6a7465', '#747d6d', '#7e8776')),
        (172, 18, 2, ('#5a614f', '#646b57', '#727960', '#82886a'), ('#465045', '#4d584b', '#566152', '#606b5b', '#6a7563')),
        (208, 20, 3, ('#4a503e', '#545a44', '#62684c', '#747a55'), ('#333e36', '#3a463c', '#435045', '#4d5a4d', '#586556')),
        (248, 22, 4, ('#3c4132', '#464b38', '#53593f', '#666b48'), ('#26302b', '#2d3831', '#36423a', '#414e44', '#4c594e')),
        (290, 22, 5, ('#30352a', '#393e2f', '#454a35', '#575c3e'), ('#1e2724', '#25302b', '#2d3933', '#37443c', '#425046')),
    ]
    xs = np.arange(MW)
    for (by, amp, sd, pc, pp) in ridges:
        r = rng(sd + 50)
        ph = r.random() * 10
        top = (by - amp * (0.5 + 0.5 * np.sin(xs / (19.0 + sd * 3) + ph)) - 0.4 * amp * np.abs(np.sin(xs / 7.0 + ph * 2))
               - 3 * np.abs(np.sin(xs / 2.7 + sd)))
        # rock faces: vertical bands of shade/body/light by facet
        nz = _tnoise(MH, MW, (3.0, 2.5), 500 + sd)
        for x in range(MW):
            t0 = int(top[x])
            ys = np.arange(max(0, t0), MH)
            if len(ys) == 0:
                continue
            k = np.full(len(ys), 1)
            k = np.where(nz[ys, x] > 0.66, 2, k)
            k = np.where(nz[ys, x] < 0.30, 0, k)
            k = np.where(ys - t0 < 3, 2, k)
            cv.a[ys, x] = np.array([col(pc[i]) for i in range(3)], np.uint8)[k]
            cv.px(x, t0, pc[3])
        # pines along the ridge top
        for i in range(MW // 7):
            x = int(r.random() * MW)
            _pine(cv, x, int(top[x]) + 3 + int(r.random() * 4), 7 + int(r.random() * (6 + sd * 2)), sd * 1000 + i, pal=pp)
    return _rgba(cv)


@lru_cache(maxsize=None)
def _gs_abyss():
    """Far wall of the Abysmal Abyss (parallax 0.85), world-like coords; returns (rgba, x0, y0)."""
    X0, Y0, AW, AH = 140, 110, 430, 360
    cv = _layer(AW, AH)
    blot = _tnoise(AH, AW, (2.0, 3.0), 91)
    streak = _tnoise(AH, AW, (9.0, 0.8), 92)
    pal = np.array([col(c_) for c_ in ('#2d3037', '#363940', '#404349', '#4b4d51', '#58595c', '#676868',
                                        '#787876')], np.uint8)
    for x in range(AW):
        X = x + X0
        top = 152 + 5 * math.sin(X / 13.0) + 3 * math.sin(X / 4.3) + 2 * abs(math.sin(X / 2.1))
        t0 = int(top) - Y0
        if t0 >= AH:
            continue
        ys = np.arange(max(0, t0), AH)
        depth = (ys - t0) / 16.0
        v = 5.6 - np.minimum(depth, 3.3) + (blot[ys, x] - 0.5) * 2.4 + (bayer(AH, AW)[ys, x] - 0.5) * 0.9
        v = np.where(streak[ys, x] < 0.25, v - 1.5, v)
        k = np.clip(np.floor(v), 0, 6).astype(int)
        cv.a[ys, x] = pal[k]
        if 0 <= t0 < AH:
            cv.px(x, t0, '#6f7552')
            cv.px(x, t0 + 1, '#5a5f45')
    # ledges with tiny hazy pines
    r = rng(93)
    for i in range(44):
        X = 160 + int(r.random() * 390)
        Y = 166 + int(r.random() * (60 if i < 16 else 280))
        L = 6 + int(r.random() * 14)
        cv.hline(X - X0, X - X0 + L, Y - Y0, '#6a6b69')
        cv.hline(X - X0, X - X0 + L, Y - Y0 + 1, '#2c2e35')
        if r.random() < 0.6:
            _pine(cv, X - X0 + L // 2, Y - Y0, 6 + int(r.random() * 6), 300 + i,
                  pal=('#2a312f', '#333c38', '#3d4741', '#48524a', '#55605a'))
    for i in range(30):
        X = 150 + int(r.random() * 410)
        top = 150 + 5 * math.sin(X / 13.0) + 3 * math.sin(X / 4.3)
        _pine(cv, X - X0, int(top) - Y0 + 2, 7 + int(r.random() * 8), 400 + i,
              pal=('#39423c', '#414b43', '#4b564c', '#566158', '#626d63'))
    return _rgba(cv), X0, Y0


def _gs_cam(f):
    t1 = cruise(seg(f, 148, 214))                 # director pass: even tilt (~5 px/f), was a 16 px/f whip
    cx = 18 + 22 * t1 + 2 * seg(f, 144, 166)
    cy = 260 - 260 * t1
    if f >= GUST:
        t2 = cruise(seg(f, GUST, 262), 0.3)          # director pass: softer gust push (~5 px/f)
        cx += 150 * t2 + 0.33 * max(0, f - 258)
        cy += 30 * t2
    return int(round(cx)), int(round(cy))


@lru_cache(maxsize=None)
def _mist_tex(hh, seed):
    TW = 512
    n = _tnoise(hh, TW, (hh / 10.0, 16.0), seed)
    yy = np.arange(hh)[:, None] / hh
    env = np.sin(yy * math.pi) ** 1.2
    return np.clip((n - 0.35) * 2.0, 0, 1) * env


def _mist(c, f, y_scr, hh, xoff, x0, x1, speed, amount, seed, color='#b7bdb9'):
    """Drifting mist band at screen y_scr, horizontally limited to screen x0..x1 (soft edges)."""
    if y_scr >= H or y_scr + hh <= 0:
        return
    tex = _mist_tex(hh, seed)
    fs = _wrap_slice(tex, int(f * speed + xoff), W).copy()
    xs = np.arange(W)
    edge = np.clip(np.minimum(xs - x0, x1 - xs) / 24.0, 0, 1)
    fs *= edge[None, :]
    _fog_blend(c, fs, int(y_scr), color, amount)


@lru_cache(maxsize=None)
def _river_tex():
    hh, TW = GH - RIVER_Y, 512
    n = _tnoise(hh, TW, (0.7, 7.0), 31)
    pal = np.array([col(c_) for c_ in ('#3b4f50', '#465c5a', '#55695f', '#7a8c80', '#aab8ad')], np.uint8)
    yy = np.arange(hh)[:, None] / hh
    k = np.clip(np.floor(n * 4.2 + yy * 0.8 - 0.6), 0, 4).astype(int)
    return pal[k]


def _ladder(c, x, y, L, f, k, seed):
    sw = math.sin(f * 0.16 + seed * 1.9) * (0.8 + 2.2 * k)
    for i in range(L):
        t = i / L
        dx = int(round(sw * t ** 1.4))
        c.px(x + dx, y + i, '#8a7a5a')
        c.px(x + 3 + dx, y + i, '#6e5f45')
        if i % 3 == 2:
            c.hline(x + dx + 1, x + 2 + dx, y + i, PLANK[3])


def _lantern(c, x, y, f, k, seed):
    ang = math.sin(f * 0.21 + seed * 2.3) * (0.25 + 0.6 * k)
    lx = x + int(round(math.sin(ang) * 3))
    ly = y + 3
    c.line(x, y, lx, ly, '#3a3f46')
    c.rect(lx - 1, ly, 3, 3, '#3a3f46')
    lit = WARM[1] if hash01(seed, f // 4) > 0.15 else WARM[0]
    c.px(lx, ly + 1, lit)
    fx.glow(c, lx, ly + 1, 6, WARM[0], strength=0.28, rings=2)


def _puffs(c, f, x, y, seed, k, n=9, big=True):
    """Locomotive smoke: puffs streaming right with the wind, growing and fading."""
    for i in range(n):
        life = 40
        t = ((f + i * life / n) % life) / life
        px_ = x + int(t * (34 + 26 * k)) + int(math.sin(t * 5 + i) * 2)
        py_ = y - int(t * 10) - int(math.sqrt(t) * 6)
        r = 1 + int(t * (4 if big else 2))
        cc = '#e9e6dc' if t < 0.25 else ('#c9cac4' if t < 0.6 else '#a9aeab')
        if t > 0.75 and hash01(seed, i, f // 3) < 0.4:
            continue
        c.circle(px_, py_, r, mix(cc, '#7d8487', 0.25))
        c.circle(px_ - 1, py_ - 1, max(1, r - 1), cc)


def _fg_pine(c, x, base, h, f, k, seed):
    """Foreground pine silhouette (dark), swaying with the wind."""
    pal = ('#141b1b', '#1b2523', '#24302c')
    tiers = h // 5
    sway = (1.5 + 3.5 * k) * (0.7 + 0.3 * math.sin(f * 0.13 + seed))
    c.rect(x - 1, base - 6, 3, 7, '#1a1514')
    for i in range(tiers):
        t = i / tiers
        ty = base - 4 - int(t * (h - 4))
        hw = int((1 - t) * h * 0.3) + 2
        sx = int(round(sway * t * t))
        for j in range(5):
            w2 = hw - j * (hw // 4 + 1) // 2
            if w2 < 0:
                break
            c.hline(x - w2 + sx, x + w2 + sx + (1 if j < 2 else 0), ty - j, pal[0 if j < 2 else 1])
        c.px(x - hw + sx, ty, pal[2])
    c.vline(x + int(round(sway)), base - h - 2, base - h, pal[1])


FG = [  # foreground items in parallax-1.3 space: (kind, x, base, h)
    ('pine', -4, 520, 84), ('pine', 22, 528, 60), ('pine', 334, 524, 78), ('pine', 352, 530, 56),
    ('crag', 548, 214, 0), ('pine', 566, 212, 46), ('pine', 588, 214, 60), ('pine', 250, 226, 50),
]


def _bridge(c, f, ox, oy, k):
    """The rickety rope bridge, swaying; the loose plank tumbles after 234."""
    ax, ay = BR_A[0] - ox, BR_A[1] - oy
    bx, by = BR_B[0] - ox, BR_B[1] - oy
    n = 60
    A = 0.6 + 0.6 * k
    if f >= GUST:
        A += 5.5 * min(1.0, (f - GUST) / 4.0) * math.exp(-(f - GUST) / 48.0)
    loose = int(n * 0.56)
    pts = []
    for i in range(n + 1):
        t = i / n
        s = math.sin(math.pi * t)
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t + 15 * s + A * s * math.sin(f * 0.33) + A * 0.6 * s * math.sin(t * 9.0 - f * 0.47)
        tw = A * 0.55 * s * math.sin(f * 0.24 + t * 3.0)
        pts.append((int(round(x)), int(round(y)), tw))
    # far rope rail + far suspenders (behind the deck)
    for i in range(n):
        x0, y0, t0 = pts[i]
        x1, y1, t1_ = pts[i + 1]
        c.line(x0, y0 - 8 - int(round(t0)), x1, y1 - 8 - int(round(t1_)), '#5e5242')
        if i % 3 == 0:
            c.vline(x0, y0 - 8 - int(round(t0)), y0 - 1, '#54493b')
    # deck planks
    for i, (x, y, tw) in enumerate(pts):
        if hash01(i, 77) < 0.07 or (i == loose and f >= 234):
            continue
        dy = 0
        if A > 1.6 and hash01(i, f // 2) > 0.72:
            dy = -1
        th = 1 + (1 if abs(tw) > 1.2 else 0)
        cc = PLANK[3] if (i % 4) else PLANK[2]
        if i == loose and f >= GUST and (f // 2) % 2:
            dy -= 1
            cc = PLANK[4]
        c.rect(x, y + dy - th + 1, 2, th, cc)
        c.px(x, y + dy + 1, PLANK[0])
    # deck support ropes under the planks
    for i in range(n):
        x0, y0, _ = pts[i]
        x1, y1, _ = pts[i + 1]
        c.line(x0, y0 + 2, x1, y1 + 2, '#4a3e32')
    # near rope rail + suspenders
    for i in range(n):
        x0, y0, t0 = pts[i]
        x1, y1, t1_ = pts[i + 1]
        c.line(x0, y0 - 7 + int(round(t0 * 0.8)), x1, y1 - 7 + int(round(t1_ * 0.8)), '#b9a57a')
        if i % 3 == 1:
            c.vline(x0, y0 - 6 + int(round(t0 * 0.8)), y0 - 1, '#9a8a66')
    # the loose plank tumbling into the abyss (snaps at 234 with splinters)
    if f >= 234:
        dt = f - 234
        x0, y0, _ = pts[loose]
        if dt < 8:
            for j in range(5):
                c.px(x0 - 2 + j + dt * (j - 2) // 3, y0 + 1 + dt // 2 + (j % 2), PLANK[4] if dt < 4 else PLANK[2])
        px_ = x0 + int(dt * (0.8 + 0.6 * k))
        py_ = y0 + int(0.5 * 0.30 * dt * dt + 0.6 * dt)
        fr = (dt // 3) % 4
        lt, dk = PLANK[4], PLANK[1]
        if py_ < H + 4:
            if fr == 0:
                c.hline(px_ - 2, px_ + 2, py_, lt)
                c.hline(px_ - 2, px_ + 2, py_ + 1, dk)
            elif fr == 1:
                c.line(px_ - 2, py_ - 2, px_ + 1, py_ + 1, lt)
                c.line(px_ - 1, py_ - 2, px_ + 2, py_ + 1, dk)
            elif fr == 2:
                c.vline(px_, py_ - 2, py_ + 2, lt)
                c.vline(px_ + 1, py_ - 2, py_ + 2, dk)
            else:
                c.line(px_ - 2, py_ + 1, px_ + 1, py_ - 2, lt)
                c.line(px_ - 1, py_ + 1, px_ + 2, py_ - 2, dk)
    return pts


def _debris(c, f, k):
    """Leaves / twigs carried by the wind; a dense blast at the GUST."""
    n = 7
    if GUST - 2 <= f < GUST + 40:
        n = 7 + int(40 * math.exp(-(f - GUST) / 16.0))
    cols = ('#7d8245', '#8a6a44', '#a68e6b', '#5e5c30')
    for i in range(n):
        gust_p = i >= 7
        sp = (9 + 5 * hash01(i, 1)) if gust_p else (2.2 + 1.5 * hash01(i, 1) + k)
        if gust_p:
            x = int(-20 + (f - GUST + 2) * sp - hash01(i, 2) * 200)
            y = int(20 + hash01(i, 3) * 150 + math.sin(f * 0.5 + i) * 3)
        else:
            x = int((hash01(i, 2) * 400 + f * sp) % 400) - 40
            y = int(30 + hash01(i, 3) * 130 + math.sin(f * 0.21 + i) * 6)
        cc = cols[i % 4]
        c.px(x, y, cc)
        if (f // 2 + i) % 2:
            c.px(x + 1, y, cc)
        else:
            c.px(x, y + 1, mix(cc, 'ink', 0.3))
    if GUST <= f < GUST + 26:
        for i in range(14):
            sp = 13 + 6 * hash01(i, 8)
            x = int(-40 + (f - GUST) * sp - hash01(i, 9) * 120)
            y = int(14 + hash01(i, 10) * 160)
            L = 8 + int(hash01(i, 11) * 20)
            m = '#cfd4cf' if i % 2 else '#aeb4b1'
            c.dither(x, y, L, 1, m, 0.75)


def _griffonstone(f):
    cam_x, cam_y = _gs_cam(f)
    wk = gust_k(f)
    c = Canvas()
    yoff = -int(cam_y * 0.12)
    gap_x = int(203 - cam_x * 0.25 + (f - 144) * 0.1)
    draw_sky(c, f, cam_x, gap_x, y_off=yoff + 4, horizon=170 - int(cam_y * 0.25), tilt=int(cam_y * 0.1))
    c.blit(_gs_far(), -int(cam_x * 0.25), -int(cam_y * 0.25))
    _mist(c, f, 128 - int(cam_y * 0.3), 30, cam_x * 0.3, -40, W + 40, 0.5 + 0.3 * wk, 0.45, 41)
    c.blit(_gs_mid(), -int(cam_x * 0.5), -int(cam_y * 0.5))
    ab, aX0, aY0 = _gs_abyss()
    c.blit(ab, int(aX0 - cam_x * 0.85), int(aY0 - cam_y * 0.85))
    _mist(c, f, 300 - int(cam_y * 0.85), 56, cam_x * 0.85, 150 - cam_x * 0.85, 600 - cam_x * 0.85,
          0.5 + 0.4 * wk, 0.5, 45, color='#8f9696')
    _mist(c, f, 372 - int(cam_y * 0.85), 50, cam_x * 0.85, 150 - cam_x * 0.85, 600 - cam_x * 0.85,
          0.4 + 0.3 * wk, 0.45, 46, color='#878e8e')
    # abyss mist far (between the mid layer and the main layer)
    _mist(c, f, 172 - int(cam_y * 0.85), 60, cam_x * 0.85, 150 - cam_x * 0.85, 560 - cam_x * 0.85,
          0.7 + 0.5 * wk, 0.6, 42, color='#a9b0ae')
    world, meta = _gs_world()
    c.blit(world, -cam_x, -cam_y)
    ox, oy = cam_x, cam_y
    # river + waterfall into the abyss
    ry = RIVER_Y - oy
    if ry < H:
        tex = _river_tex()
        sl = _wrap_slice(tex, int(-f * 1.4), W)
        x1 = min(W, CLIFF_X - 2 - ox)
        if x1 > 0:
            c.a[ry:min(H, ry + tex.shape[0]), 0:x1] = sl[:min(H, ry + tex.shape[0]) - ry, :x1]
            c.hline(0, x1 - 1, ry, '#8fa196')
        wx = CLIFF_X - 3 - ox
        for i in range(12):
            for j in range(0, 60, 1):
                y = ry + j
                if y >= H:
                    break
                xx = wx + int(i * 0.9) + j // 14
                ph = (j * 3 - f * 7 + i * 13) % 23
                cc = '#e3e9e2' if ph < 5 else ('#aebcb5' if ph < 13 else '#7d918b')
                c.px(xx, y, cc)
        for i in range(8):
            ph = (f * 2 + i * 7) % 16
            c.px(wx + i - 2, ry - 1 + (ph // 8), 'white' if ph < 8 else FOAM)
    # sun-break shaft onto the crown
    sun_shaft(c, gap_x, 30 + yoff, H, f, strength=0.12, slant=-0.8)
    # dynamic details on the main layer
    for i, (wx_, wy_) in enumerate(meta['wins']):
        X, Y = wx_ - ox, wy_ - oy
        if 0 <= X < W and 0 <= Y < H and hash01(i, f // 5) < 0.14:
            c.px(X, Y, WARM[0])
    for i, (sx_, sy_) in enumerate(meta['chims']):
        X, Y = sx_ - ox, sy_ - oy
        if -40 < X < W and -10 < Y < H + 10:
            _smoke(c, f, X, Y, 200 + i, n=6, k=wk)
    for i, (lx, ly, L) in enumerate(meta['ladders']):
        X, Y = lx - ox, ly - oy
        if -10 < X < W and -L < Y < H:
            _ladder(c, X, Y, L, f, wk, i)
    for i, (lx, ly) in enumerate(meta['lanterns']):
        X, Y = lx - ox, ly - oy
        if -6 < X < W + 6 and -6 < Y < H:
            _lantern(c, X, Y, f, wk, i)
    for i, (px_, py_) in enumerate(meta['poles']):
        X, Y = px_ - ox, py_ - oy
        if -20 < X < W and -10 < Y < H:
            _flag(c, f + i * 7, X, Y + 1, L=10, hgt=7, colors=('#6e3a3a', '#b89a4a'), k=wk * 1.2, tatter=True)
    # Friendship Express steaming
    fxw, fyw = meta['funnel']
    X, Y = fxw - ox, fyw - oy
    if -20 < X < W and -20 < Y < H + 20:
        _puffs(c, f, X, Y, 5, wk)
        cx_, cy_ = meta['cyl']
        for t0 in (162, 198):
            dt = f - t0
            if 0 <= dt < 14:
                for j in range(6):
                    c.px(cx_ - ox - dt // 2 - j, cy_ - oy - (j % 3), 'white' if dt < 7 else FOAM)
                    c.px(cx_ - ox + 26 + dt // 2 + j, cy_ - oy - (j % 2), 'white' if dt < 7 else FOAM)
    # the rope bridge + near abyss mist
    _bridge(c, f, ox, oy, wk)
    _mist(c, f, 158 - int(cam_y * 0.95), 44, cam_x * 0.95, 120 - ox, 560 - ox, 0.9 + 0.6 * wk, 0.7,
          43, color='#c3c8c3')
    _mist(c, f, 380 - int(cam_y * 1.0), 70, cam_x * 1.0, CLIFF_X - ox - 6, 520 - ox, 0.6 + 0.4 * wk, 0.75,
          44, color='#bfc5c0')
    # balloon battling the wind
    shove = 0.0
    if f >= GUST:
        dt = f - GUST
        shove = 22 * (1 - math.exp(-dt / 5.0)) * math.exp(-dt / 38.0)
    bx = 250 - 0.4 * (cam_x - 40) + shove + math.sin(f / 9.0) * (1 + 2 * wk)
    by = (58 - 0.4 * (cam_y if f >= GUST else 0) - 12 * math.sin(math.pi * seg(f, 166, 214))
          + math.sin(f / 6.0) * 1.5 * wk + shove * 0.3)
    draw_balloon(c, int(round(bx)), int(round(by)), f, size='m', flame=(f // 12) % 3 == 0)
    # foreground (parallax 1.3)
    fox, foy = cam_x * 1.3, cam_y * 1.3
    for i, (kind, x, base, h) in enumerate(FG):
        X, B = int(x - fox), int(base - foy)
        if base < 300 and f < GUST:
            continue
        if kind == 'pine':
            if -30 < X < W + 30 and 0 < B < H + h:
                _fg_pine(c, X, B, h, f, wk, i)
        else:
            if -80 < X < W and B < H + 40:
                c.poly([(X, H), (X + 6, B - 6), (X + 30, B - 12), (X + 60, B - 8), (X + 90, B), (X + 90, H)],
                       '#1b2220')
                c.line(X + 6, B - 6, X + 30, B - 12, '#2e3a33')
                c.line(X + 30, B - 12, X + 60, B - 8, '#2e3a33')
    _debris(c, f, wk)
    return c


# ================================================================== render
def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < 144:
        c = _trottingham(f)
        cam = int(round(PAN * ease_in_out(seg(f, 72, 143))))
        callout(c, 229 - cam, 73, 'TROTTINGHAM', f, 81, 137, side='right', length=10)
        cx, cy = dest_screen(ID)
        iris(c, 1 - seg(f, 72, 79), cx, cy)
    else:
        c = _griffonstone(f)
        ox, oy = _gs_cam(f)
        callout(c, 246 - ox, 375 - oy, 'GRIFFONSTONE STATION', f, 148, 178, side='up', length=10)
        callout(c, 196 - ox, 44 - oy, 'GRIFFONSTONE', f, 200, 232, side='right', length=14)
        callout(c, 386 - ox, 182 - oy, 'ABYSMAL ABYSS', f, 240, 280, side='right', length=12)
    atlas_card(c, f, 100, 176, 'GRIFFON KINGDOM', ['CITY ON A ROCK SPIRE', 'ACROSS THE SEA'])
    return c


SFX = [
    (76, 'wind', 0.45, {'variant': 0}),          # sea breeze over the strait
    (81, 'waves', 0.55, {'align': 'hit'}),        # spray on the foreground rocks
    (92, 'creak', 0.35),                          # ship timbers
    (108, 'wind', 0.4, {'variant': 0}),
    (117, 'waves', 0.45, {'align': 'hit'}),
    (150, 'whistle', 0.55),                       # Friendship Express at Griffonstone Station
    (162, 'steam', 0.55),                         # cylinder hiss
    (176, 'wind', 0.45, {'variant': 0, 'pan': 0.3}),   # tilt up the spire
    (198, 'steam', 0.4),
    (216, 'wind', 0.85, {'variant': 1}),          # GUST: howling wind
    (222, 'creak', 0.75, {'pan': 0.4}),           # rope bridge strains
    (234, 'creak', 0.8, {'pan': 0.2}),            # the loose plank snaps
    (250, 'creak', 0.5, {'pan': 0.1}),
    (262, 'thunder', 0.35, {'variant': 0}),       # distant roll over the abyss
    (270, 'wind', 0.5, {'variant': 0}),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(144, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
