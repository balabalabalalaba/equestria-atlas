"""PONYVILLE: the country town (5 bars, 360 frames, HUD clock 07:30 -> 08:30).

Beat sheet (local frames; bar = 72, beat = 18):
  Bar 1   0-71    World-map travel shot (`travel_frame`); iris closes 64-71.
  Bar 2  72-143   REVEAL (72): iris opens 72-79 on the CASTLE OF FRIENDSHIP (crystal tree castle, its
                  star glinting on every beat). Fresh morning: low sun on the left (east), long soft
                  shadows to the right, dew sparkles, mist in the far fields, Canterlot with its
                  waterfalls on the violet mountain far behind, the Everfree's dark treeline on the
                  left. SHOT A = one eased pan right (72-215) to the SCHOOL OF FRIENDSHIP.
                  Callouts CASTLE OF FRIENDSHIP 72-138, SCHOOL OF FRIENDSHIP 108-180. Card 100-200.
  Bar 3 144-215   The pan reaches the market square: TOWN HALL (spire + flag), striped stalls with
                  apples / carrots / flowers, lantern strings, the clock tower (its hands show the HUD
                  time). A weather-schedule raincloud drizzles on the town-hall garden 156-197 and
                  drifts off on the beat at 198. Callout TOWN HALL 144-209.
  Bar 4 216-287   CUT on the bar line, SHOT B = one eased pan right (216-359): laundry flapping,
                  SUGARCUBE CORNER (gingerbread bakery, giant cupcake, oven steam) and CAROUSEL
                  BOUTIQUE (carousel roof, gem ornaments, diamond door).
                  Callouts SUGARCUBE CORNER 216-282, CAROUSEL BOUTIQUE 252-324.
  Bar 5 288-359   `train` HIT (288): the pink Friendship Express rolls in from the right, whistle
                  steam at 288 and 306 (the music's two whistle notes), chuffing on 8ths, stops at the
                  platform ~334 with a steam hiss. The stream under the rail bridge glitters, Saddle
                  Lake beyond; the balloon drifts over the thatched rooftops. Callout TRAIN STATION
                  288-346. Toast from 302 (top-centre kept clear). Last 8 frames: wipe.
Ambient: drifting clouds, Canterlot waterfalls, far-field mist, windmill, lake + stream glints,
         chimney smoke, banners + flags, crystal glints, lantern sway, laundry, clock hands, oven
         steam, drizzle, dew sparkles, train smoke + wheels, the balloon.
"""
import math
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'ponyville'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)
GY = 148                        # ground line: building bases
P_CAN, P_FAR, P_MID, P_BACK, P_FG = 0.1, 0.2, 0.4, 0.7, 1.3
MAIN_W = 1400
HAZE = col('#bfe6ff')

# ------------------------------------------------------------------ world layout (main layer x)
CASTLE_X = 236
SCHOOL_X0, SCHOOL_X1 = 298, 384
GARDEN_X0, GARDEN_X1 = 386, 414
HALL_X0, HALL_X1 = 418, 494
STALLS = [(516, 'apple', ('crimson', 'paper')), (540, 'carrot', ('aqua', 'paper')),
          (564, 'flower', ('hot_pink', 'paper'))]
CLOCK_X = 505
LAUNDRY_X0, LAUNDRY_X1 = 630, 656
SUGAR_X0, SUGAR_X1 = 704, 772
BOUT_X = 826
STATION_X0, STATION_X1 = 912, 962          # station house
PLATFORM_X0, PLATFORM_X1 = 898, 1104
STREAM_X0, STREAM_X1 = 1112, 1150
TRAIN_STOP = 912                            # loco front (world x) when stopped

# ------------------------------------------------------------------ camera
_CAM_A = PchipInterpolator([72, 110, 144, 180, 215], [0, 24, 112, 200, 296])
_CAM_B = PchipInterpolator([216, 252, 288, 330, 359], [606, 640, 712, 790, 820])


def cam_x(f):
    if f < 216:
        return int(round(float(_CAM_A(clamp(f, 72, 215)))))
    return int(round(float(_CAM_B(clamp(f, 216, 359)))))


# ------------------------------------------------------------------ helpers
_NZ = np.random.RandomState(1234).rand(1 << 16)


def nz(a, b=0, s=0):
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    return _NZ[((a * 73856093) ^ (b * 19349663) ^ (int(s) * 83492791)) & 0xFFFF]


def R(*cs):
    return np.array([col(c) for c in cs], np.uint8)


def hz(ramp, t, to=None):
    to = np.array(HAZE if to is None else col(to), np.float32)
    return np.clip(ramp.astype(np.float32) * (1 - t) + to * t + 0.5, 0, 255).astype(np.uint8)


def tup(a):
    return (int(a[0]), int(a[1]), int(a[2]))


def _new(w, h):
    return Canvas(w, h, KEY)


def _rgba(cv):
    a = cv.a
    m = np.any(a != _KEYA, axis=2)
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = m * 255
    return out


def blit_wrap(c, arr, x, y=0):
    w = arr.shape[1]
    x = int(x) % w
    c.blit(arr, -x, y)
    if w - x < W:
        c.blit(arr, w - x, y)


def paint(cv, x, y, mask, idx, ramp):
    h, w = mask.shape
    x0, y0, x1, y1 = cv._clip(x, y, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    i = np.clip(idx[y0 - y:y1 - y, x0 - x:x1 - x], 0, len(ramp) - 1)
    view = cv.a[y0:y1, x0:x1]
    view[m] = ramp[i[m]]


def qd(L, n, x0=0, y0=0, soft=0.7):
    h, w = L.shape
    thr = bayer(h, w, x0, y0)
    return np.clip(np.floor(L * (n - 1) + 0.5 + (thr - 0.5) * soft), 0, n - 1).astype(int)


def edge(m):
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def arch_mask(w, h):
    r = (w - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    cy = r
    return (yy >= cy) | ((xx - r) ** 2 + (yy - cy) ** 2 <= (r + 0.45) ** 2)


def cyl_L(w, peak=0.3, k=1.4):
    u = (np.arange(w) + 0.5) / w
    return np.clip(1 - np.abs(u - peak) * k, 0, 1)


def cylinder(cv, x, y, w, h, ramp, peak=0.3, outline=True):
    L = np.broadcast_to(cyl_L(w, peak)[None, :], (h, w))
    idx = 1 + qd(L, len(ramp) - 1, x, y)
    if outline:
        idx[:, 0] = 0
        idx[:, -1] = 0
    paint(cv, x, y, np.ones((h, w), bool), idx, ramp)


def cone(cv, cx, ytop, ybase, r, ramp, curve=1.5, peak=0.3, shingles=0, stripes=0, outline=True):
    h = ybase - ytop + 1
    w = 2 * r + 1
    yy, xx = np.mgrid[0:h, 0:w]
    t = (yy + 0.5) / h
    hw = r * t ** curve + 0.35
    dx = xx - r
    m = np.abs(dx) <= hw
    u = (dx / np.maximum(hw, 0.8) + 1) / 2
    L = 1 - np.abs(u - peak) * 1.35
    if shingles:
        row = yy // shingles
        L = L - (((yy % shingles) == shingles - 1) & ((xx + row * 2) % 4 != 0)) * 0.2
    if stripes:
        sg = np.floor(np.arcsin(np.clip(dx / np.maximum(hw, 0.8), -1, 1)) / math.pi * stripes * 2)
        L = L + ((sg.astype(int) % 2) * 2 - 1) * 0.12
    idx = 1 + qd(np.clip(L, 0, 1), len(ramp) - 1, cx - r, ytop)
    if outline:
        idx[edge(m)] = 0
    paint(cv, cx - r, ytop, m, idx, ramp)


def blob(cv, circles, ramp, outline=True, seed=0, tex=0.16, light=(0.42, 0.5)):
    xs0 = int(min(c[0] - c[2] for c in circles)) - 2
    ys0 = int(min(c[1] - c[2] for c in circles)) - 2
    xs1 = int(max(c[0] + c[2] for c in circles)) + 3
    ys1 = int(max(c[1] + c[2] for c in circles)) + 3
    w, h = xs1 - xs0, ys1 - ys0
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx + xs0
    Y = yy + ys0
    m = np.zeros((h, w), bool)
    L = np.zeros((h, w))
    for (bx, by, br) in sorted(circles, key=lambda c: c[1]):
        nx = (X - bx) / (br + 0.5)
        ny = (Y - by) / (br + 0.5)
        ins = nx * nx + ny * ny <= 1
        l = 0.62 - light[0] * nx - light[1] * ny
        L = np.where(ins, l, L)
        m |= ins
    gcx, gcy = (xs0 + xs1) / 2, (ys0 + ys1) / 2
    gr = max(w, h) / 2
    G = 0.6 - 0.4 * (X - gcx) / gr - 0.5 * (Y - gcy) / gr
    L = 0.55 * L + 0.45 * G
    if tex:
        L = L + (nz(X // 2, Y // 2, seed) - 0.5) * tex
    idx = 1 + qd(np.clip(L, 0, 1), len(ramp) - 1, xs0, ys0)
    if outline:
        idx[edge(m)] = 0
    paint(cv, xs0, ys0, m, idx, ramp)


def thatch(cv, x0, x1, yb, hr, seed=0, p=1.9, ramp=None, tiers=6, streaks=True, crown=True):
    """Rounded straw-thatch hood: eave row at yb-1, top at yb-hr. Lit from the left."""
    ramp = TH if ramp is None else ramp
    w = x1 - x0 + 1
    h = hr
    yy, xx = np.mgrid[0:h, 0:w]
    s = (h - 1 - yy + 0.5) / h
    hw = (w / 2) * np.clip(1 - s ** p, 0, 1) ** (1 / p)
    dx = xx - (w - 1) / 2
    m = np.abs(dx) <= hw
    a = dx / np.maximum(hw, 1)
    L = 0.6 - 0.34 * a + 0.12 * s
    if streaks:
        k = np.floor((a + 1) * w / 4.5).astype(int)
        L = L + (nz(k, yy // 9, seed) - 0.5) * 0.3
    r = h - 1 - yy
    wig = (nz(xx // 3, 7, seed) * 2).astype(int)
    tier = ((r + wig) % tiers == 0) & (r > 2) & (s < 0.8)
    lip = ((r + wig) % tiers == 1) & (r > 2) & (s < 0.8)
    L = L - tier * 0.32 + lip * 0.14
    if crown:
        L = L - (s > 0.8) * 0.22
    L = np.where(r <= 1, L - 0.38, L)
    idx = 1 + qd(np.clip(L, 0, 1), len(ramp) - 1, x0, yb - h, soft=0.5)
    idx[edge(m)] = 0
    paint(cv, x0, yb - h, m, idx, ramp)


def roof_y(x0, x1, yb, hr, x, p=1.9):
    """Top surface y of a thatch() roof at column x."""
    w = x1 - x0 + 1
    dx = abs(x - (x0 + (w - 1) / 2)) / (w / 2)
    if dx >= 1:
        return yb
    s = (1 - dx ** p) ** (1 / p)
    return int(round(yb - s * hr))


# ------------------------------------------------------------------ ramps
TH = R(mix('rust', 'wine', .3), mix('copper', 'clay', .45), 'clay', mix('sand', 'gold', .5),
       mix('gold', 'butter', .5), 'cream')
GR = R(mix('forest', 'ink', .35), 'forest', 'green', mix('green', 'leaf', .5), 'leaf', 'pale_leaf')
TIMB = R('plum_brown', 'mauve', mix('mauve', 'tan', .5))
WOOD = R(mix('rust', 'plum_brown', .5), 'rust', 'copper', 'clay')
STONE = R('shadow', 'grey', mix('stone', 'tan', .4), mix('paper', 'stone', .3), 'paper')
CRV = R('eggplant', 'purple', mix('purple', 'violet', .5), 'violet', mix('violet', 'lavender', .45),
        'lavender', mix('lavender', 'lilac', .55))
CRT = R(mix('violet', 'eggplant', .4), 'violet', mix('violet', 'lavender', .5), 'lavender',
        mix('lavender', 'lilac', .5), mix('lilac', 'white', .4))
CRR = R(mix('crystal_dk', 'navy', .4), 'crystal_dk', mix('teal', 'crystal_dk', .35),
        mix('aqua', 'crystal', .45), 'crystal', 'crystal_lt', mix('crystal_lt', 'white', .6))
STAR = R('plum', 'raspberry', 'magic', 'crystal_pk', mix('crystal_pk', 'white', .55), 'white')
SR = R(mix('royal', 'ink', .35), 'royal', mix('royal', 'royal_lt', .5), 'royal_lt', 'lavender',
       mix('lavender', 'lilac', .5))
BR = R(mix('indigo', 'ink', .3), 'indigo', 'blue', mix('blue', 'periwinkle', .5), 'periwinkle',
       mix('periwinkle', 'baby_blue', .6))
FR = R('raspberry', 'rose', 'pink', 'bubblegum', 'cotton', mix('cotton', 'white', .6))
GB = R('rust', mix('copper', 'rust', .35), 'copper', 'clay', mix('clay', 'sand', .5))
CUP = R('tan', mix('peach', 'clay', .3), 'peach', mix('peach', 'cream', .5), 'cream')
LANE = R(mix('plum_brown', 'mauve', .4), mix('mauve', 'tan', .45), mix('tan', 'sand', .3),
         mix('tan', 'sand', .55), mix('sand', 'cream', .6))
RED_TILE = R('maroon', 'brick', mix('brick', 'vermilion', .5), 'vermilion', 'coral', 'peach')
CL = R(mix('cloud_dk', 'periwinkle', .3), 'cloud_dk', 'cloud', 'white', mix('cream', 'white', .4))


def WR(base, sh, k=.38):
    return R(mix(base, 'plum_brown', .6), mix(base, sh, k), base, mix(base, 'white', .5))


WALL = {
    'cotton': WR('cotton', 'rose', .35),
    'cream': WR('cream', 'tan', .45),
    'mint': WR('mint_cream', 'sage', .32),
    'lilac': WR(mix('lilac', 'white', .4), 'violet', .3),
    'peach': WR(mix('peach', 'cream', .45), 'clay', .3),
    'butter': WR(mix('butter', 'cream', .55), 'sand', .45),
    'blue': WR(mix('baby_blue', 'white', .25), 'periwinkle', .45),
}
FLOWERS = ('hot_pink', 'red', 'gold', 'white', 'lilac', 'coral', 'bubblegum')


# ------------------------------------------------------------------ small parts
def win_round(cv, cx, cy, r=3, frame=None, box=True, seed=0, meta=None):
    frame = TIMB if frame is None else frame
    cv.circle(cx, cy, r + 1, tup(frame[0]))
    cv.circle(cx, cy, r, tup(frame[2]))
    cv.circle(cx, cy, r - 1, 'baby_blue')
    cv.rect(cx - r + 1, cy + 1, 2 * r - 1, r - 1, mix('baby_blue', 'periwinkle', .45))
    cv.circle(cx, cy, r - 1, 'baby_blue', fill=False) if False else None
    cv.hline(cx - r + 1, cx + r - 1, cy, tup(frame[1]))
    cv.vline(cx, cy - r + 1, cy + r - 1, tup(frame[1]))
    cv.px(cx - 1, cy - 2, 'white')
    cv.px(cx - 2, cy - 1, mix('baby_blue', 'white', .6))
    if box:
        flowerbox(cv, cx - r - 1, cy + r + 1, 2 * r + 3, seed)
    if meta is not None:
        meta.append((cx, cy, r))


def win_arch(cv, x, y, w, h, frame=None, glass='baby_blue', shutters=None, box=False, seed=0, meta=None):
    frame = TIMB if frame is None else frame
    m = arch_mask(w + 2, h + 1)
    paint(cv, x - 1, y - 1, m, np.zeros(m.shape, int), R(tup(frame[0])))
    m2 = arch_mask(w, h)
    idx = np.zeros(m2.shape, int)
    idx[h // 2:, :] = 1
    paint(cv, x, y, m2, idx, R(glass, mix(glass, 'periwinkle', .45)))
    cv.vline(x + w // 2, y + 1, y + h - 1, tup(frame[1]))
    cv.hline(x, x + w - 1, y + h // 2, tup(frame[1]))
    cv.px(x + 1, y + 2, 'white')
    if shutters:
        cv.rect(x - 3, y, 2, h, shutters)
        cv.rect(x + w + 1, y, 2, h, shade(shutters, .8))
        cv.hline(x - 3, x - 2, y + h // 2, shade(shutters, .75))
    if box:
        flowerbox(cv, x - 1, y + h, w + 2, seed)
    if meta is not None:
        meta.append((x + w // 2, y + h // 2, max(2, w // 2)))


def flowerbox(cv, x, y, w, seed=0):
    cv.rect(x, y, w, 2, 'rust')
    cv.hline(x, x + w - 1, y, 'copper')
    for i in range(w):
        if hash01(seed, i, 3) < 0.75:
            cv.px(x + i, y - 1, 'green' if hash01(seed, i) < 0.45 else FLOWERS[int(hash01(seed, i, 2) * 7)])
        if hash01(seed, i, 4) < 0.3:
            cv.px(x + i, y - 2, FLOWERS[int(hash01(seed, i, 5) * 7)])


def door(cv, x, yb, w=7, h=12, wood=None, knob='gold', frame=None):
    wood = WOOD if wood is None else wood
    frame = TIMB if frame is None else frame
    m = arch_mask(w + 2, h + 1)
    paint(cv, x - 1, yb - h - 1, m, np.zeros(m.shape, int), R(tup(frame[0])))
    m2 = arch_mask(w, h)
    xx = np.arange(w)[None, :].repeat(h, 0)
    idx = np.where(xx % 3 == 2, 0, np.where(xx < w // 2, 2, 1))
    paint(cv, x, yb - h, m2, idx, wood)
    cv.px(x + w - 2, yb - h // 2, knob)
    cv.rect(x - 1, yb, w + 2, 1, 'stone')


def picket(cv, x0, x1, yb, h=7, col_=None):
    c3 = col_ or ('grey', mix('stone', 'paper', .4), 'paper', 'white')
    cv.rect(x0, yb - h + 2, x1 - x0, 1, c3[1])
    cv.rect(x0, yb - 3, x1 - x0, 1, c3[1])
    for x in range(x0, x1 - 1, 3):
        cv.rect(x, yb - h + 1, 2, h - 1, c3[2])
        cv.vline(x, yb - h + 1, yb - 1, c3[3])
        cv.vline(x + 1, yb - h + 2, yb - 1, c3[1])
        cv.px(x, yb - h, c3[3])
        cv.px(x + 1, yb - h + 1, c3[2])


def tree(cv, x, yb, r, seed=0, ramp=None, th=None):
    ramp = GR if ramp is None else ramp
    th = th if th is not None else max(3, r // 2 + 2)
    cy = yb - th - r + 2
    cv.rect(x - 1, yb - th - 3, 3, th + 3, TIMB[1])
    cv.vline(x - 1, yb - th - 3, yb - 1, TIMB[2])
    cv.vline(x + 1, yb - th - 3, yb - 1, TIMB[0])
    circles = [(x, cy, r * 0.82)]
    n = 5 + r // 3
    for i in range(n):
        a = 2 * math.pi * (i / n + hash01(seed, i) * 0.12)
        rr = r * (0.48 + 0.14 * hash01(seed, i, 2))
        circles.append((x + math.cos(a) * r * 0.5, cy + math.sin(a) * r * 0.42, rr))
    blob(cv, circles, ramp, seed=seed)


def bush(cv, x, yb, w, h, seed=0, ramp=None, flowers=True):
    ramp = GR if ramp is None else ramp
    n = max(2, w // 5)
    circles = []
    for i in range(n):
        u = (i + 0.5) / n
        rr = h * (0.55 + 0.25 * math.sin(u * math.pi)) * (0.85 + 0.3 * hash01(seed, i))
        circles.append((x + u * w, yb - rr + 1, rr))
    blob(cv, circles, ramp, seed=seed)
    if flowers:
        for i in range(w // 3):
            fx_ = x + int(hash01(seed, i, 1) * w)
            fy = yb - 1 - int(hash01(seed, i, 2) * h * 1.2)
            if hash01(seed, i, 3) < 0.6 and cv.a[fy % cv.h, fx_ % cv.w].tolist() != list(KEY):
                cv.px(fx_, fy, FLOWERS[int(hash01(seed, i, 4) * 7)])


def lamp_post(cv, x, yb, h=22):
    cv.rect(x, yb - h, 2, h, 'night')
    cv.vline(x, yb - h, yb - 1, 'shadow')
    cv.rect(x - 1, yb - 2, 4, 2, 'night')
    cv.rect(x - 2, yb - h - 5, 6, 5, 'night')
    cv.rect(x - 1, yb - h - 4, 4, 3, mix('butter', 'cream', .4))
    cv.px(x - 1, yb - h - 4, 'white')
    cv.hline(x - 3, x + 4, yb - h - 6, 'night')
    cv.px(x, yb - h - 7, 'night')
    cv.px(x + 1, yb - h - 7, 'night')


def barrel(cv, x, yb):
    cylinder(cv, x, yb - 7, 6, 7, WOOD)
    cv.hline(x, x + 5, yb - 6, 'shadow')
    cv.hline(x, x + 5, yb - 2, 'shadow')


def crystal_star(cv, cx, cy, ro, ri, ramp=STAR):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        r = ro if i % 2 == 0 else ri
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    for i in range(10):
        p, q = pts[i], pts[(i + 1) % 10]
        am = -math.pi / 2 + (i + 0.5) * math.pi / 5
        Lv = 0.5 + 0.5 * (-math.cos(am) * 0.75 - math.sin(am) * 0.66)
        k = 1 + int(round(clamp(Lv) * (len(ramp) - 2)))
        cv.tri((cx + 0.5, cy + 0.5), (p[0] + 0.5, p[1] + 0.5), (q[0] + 0.5, q[1] + 0.5), tup(ramp[k]))
    for i in range(10):
        p, q = pts[i], pts[(i + 1) % 10]
        cv.line(p[0], p[1], q[0], q[1], tup(ramp[0]))


def small_star(cv, x, y, c='gold', c2=None):
    c2 = c2 or shade(c, .75)
    cv.px(x, y - 1, c)
    cv.hline(x - 1, x + 1, y, c)
    cv.px(x - 1, y + 1, c2)
    cv.px(x + 1, y + 1, c2)


def gem(cv, x, y, c='crystal'):
    cv.px(x, y - 1, mix(c, 'white', .6))
    cv.px(x - 1, y, c)
    cv.px(x, y, mix(c, 'white', .3))
    cv.px(x + 1, y, shade(c, .7))
    cv.px(x, y + 1, shade(c, .6))


ICONS = {
    'apple': Sprite.from_ascii('''
        ..g..
        .rrr.
        rwrrr
        rrrrd
        .rdd.
    ''', {'g': 'green', 'r': 'red', 'w': 'white', 'd': 'crimson'}),
    'carrot': Sprite.from_ascii('''
        .g.g
        ..g.
        .oo.
        .oo.
        ..o.
    ''', {'g': 'leaf', 'o': 'orange'}),
    'flower': Sprite.from_ascii('''
        .p.
        pyp
        .p.
        .g.
        gg.
    ''', {'p': 'hot_pink', 'y': 'gold', 'g': 'green'}),
    'cupcake': Sprite.from_ascii('''
        ..r..
        .ppp.
        ppwpp
        .ccc.
        .ccc.
    ''', {'r': 'red', 'p': 'bubblegum', 'w': 'white', 'c': 'sand'}),
    'train': Sprite.from_ascii('''
        p......
        p.pppp.
        ppppppp
        ppppppp
        .o.o.o.
    ''', {'p': 'hot_pink', 'o': 'ink'}),
    'gem': Sprite.from_ascii('''
        .lll.
        lcccd
        .ccd.
        ..d..
    ''', {'l': 'crystal_lt', 'c': 'crystal', 'd': 'crystal_dk'}),
}


def signboard(cv, x, y, icon, board='paper'):
    s = ICONS[icon]
    cv.rect(x, y, s.w + 4, s.h + 4, 'rust')
    cv.rect(x + 1, y + 1, s.w + 2, s.h + 2, board)
    s.draw(cv, x + 2, y + 2)


# ------------------------------------------------------------------ cottage
def cottage(cv, x, w, wh, rh, wall='cream', seed=0, chim=True, dormer=False, meta=None,
            timber=True, ov=4, door_at=None, upper=True):
    rw = WALL[wall]
    yt = GY - wh
    cv.rect(x, yt, w, wh, tup(rw[0]))
    cv.rect(x + 1, yt, w - 2, wh - 1, tup(rw[2]))
    for i in range(w // 2):
        if hash01(seed, i, 9) < 0.3:
            cv.px(x + 2 + int(hash01(seed, i, 8) * (w - 4)), yt + 4 + int(hash01(seed, i, 7) * (wh - 8)),
                  tup(rw[1]))
    cv.rect(x + 1, yt, 1, wh - 1, tup(rw[3]))
    cv.rect(x + w - 3, yt, 2, wh - 1, tup(rw[1]))
    cv.rect(x, GY - 3, w, 3, 'grey')
    cv.hline(x, x + w - 1, GY - 3, mix('stone', 'paper', .3))
    for i in range(x + 2, x + w - 1, 4):
        cv.px(i, GY - 2, 'shadow')
    if timber:
        my = yt + wh // 2 - 1
        for xx in (x + 1, x + w - 3):
            cv.rect(xx, yt + 3, 2, wh - 6, tup(TIMB[1]))
            cv.vline(xx, yt + 3, GY - 4, tup(TIMB[2]))
        if wh >= 20:
            cv.rect(x + 1, my, w - 2, 2, tup(TIMB[1]))
            cv.hline(x + 1, x + w - 2, my, tup(TIMB[2]))
            cv.line(x + 3, my - 1, x + 7, yt + 5, tup(TIMB[1]))
            cv.line(x + w - 4, my - 1, x + w - 8, yt + 5, tup(TIMB[1]))
    dxp = x + (w - 7) // 2 if door_at is None else x + door_at
    door(cv, dxp, GY - 1, 7, 12, wood=WOOD if hash01(seed, 1) < 0.5 else
         R(mix('royal', 'plum_brown', .4), 'royal_lt', 'lavender', 'lilac'))
    wins = []
    for wx in (x + 5, x + w - 6):
        if abs(wx - (dxp + 3)) >= 8:
            win_round(cv, wx, GY - 9, 3, seed=seed + wx, meta=meta)
    if upper and wh >= 22:
        uy = yt + 7
        if w >= 30:
            for wx in (x + w // 2 - 6, x + w // 2 + 6):
                win_round(cv, wx, uy, 2, seed=seed + wx, box=False, meta=meta)
        else:
            win_round(cv, x + w // 2, uy, 2, seed=seed + 3, box=False, meta=meta)
    rx0, rx1 = x - ov, x + w - 1 + ov
    yb = yt + 4
    if chim:
        chx = x + int(w * (0.68 + 0.12 * hash01(seed, 5)))
        ry = roof_y(rx0, rx1, yb, rh, chx)
        top = ry - 5
        cv.rect(chx, top, 4, yb - top, tup(STONE[2]))
        cv.vline(chx, top, yb - 1, tup(STONE[3]))
        cv.vline(chx + 3, top, yb - 1, tup(STONE[1]))
        cv.rect(chx - 1, top - 1, 6, 2, tup(STONE[1]))
        cv.hline(chx - 1, chx + 4, top - 1, tup(STONE[3]))
        if meta is not None:
            meta.append(('smoke', chx + 2, top - 2, seed))
    thatch(cv, rx0, rx1, yb, rh, seed)
    if chim:
        ry = roof_y(rx0, rx1, yb, rh, chx)
        top = ry - 5
        cv.rect(chx, top, 4, max(1, ry - top + 1), tup(STONE[2]))
        cv.vline(chx, top, ry, tup(STONE[3]))
        cv.vline(chx + 3, top, ry, tup(STONE[1]))
        cv.rect(chx - 1, top - 1, 6, 2, tup(STONE[1]))
        cv.hline(chx - 1, chx + 4, top - 1, tup(STONE[3]))
        cv.hline(chx, chx + 3, ry, tup(TH[1]))
    cv.hline(x + 1, x + w - 2, yb, tup(rw[1]))
    if dormer and rh >= 16:
        dcx = x + w // 2 - 1
        dy = yb - rh // 2
        thatch(cv, dcx - 6, dcx + 6, dy - 2, 5, seed + 1, tiers=99, streaks=False, crown=False)
        win_round(cv, dcx, dy + 1, 2, box=False, meta=meta)


# ------------------------------------------------------------------ CASTLE OF FRIENDSHIP
CASTLE_TURRETS = [(-46, 98, 12, 13, 16), (-28, 72, 10, 13, 17), (30, 68, 10, 13, 17), (48, 94, 12, 13, 16)]


def stroke(cv, p0, p1, p2, r0, r1, ramp, n=26):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append((x, y, r0 + (r1 - r0) * t))
    for x, y, r in pts:
        cv.circle(x, y, r, tup(ramp[0]))
    for x, y, r in pts:
        cv.circle(x, y, max(0.5, r - 1), tup(ramp[2]))
    for x, y, r in pts:
        if r > 2:
            cv.circle(x - 0.5, y - r * 0.35, max(0.5, r * 0.45), tup(ramp[4]))
    for x, y, r in pts[::3]:
        if r > 2.5:
            cv.px(x - 1, y - r + 1, tup(ramp[6]))


def castle(cv, meta):
    cx = CASTLE_X
    # roots
    for sd in (-1, 1):
        for (x0, y0, x1, lt) in ((14, 124, 56, 0), (16, 134, 38, 1), (10, 138, 26, 2)):
            pts = [(cx + sd * (x0 + (x1 - x0) * t), GY - (GY - y0) * (1 - t) ** 2)
                   for t in np.linspace(0, 1, 16)]
            body = CRV[3] if sd < 0 else CRV[2]
            cv.poly(pts + [(cx + sd * x0, GY)], tup(body))
            for (a, b) in zip(pts, pts[1:]):
                cv.line(a[0], a[1] + 1, b[0], b[1] + 1, tup(CRV[5] if sd < 0 else CRV[4]))
                cv.line(a[0], a[1], b[0], b[1], tup(CRV[0]))
            mid = [(cx + sd * (x0 + (x1 - x0) * t), GY - (GY - y0) * 0.55 * (1 - t) ** 2)
                   for t in np.linspace(0, 0.8, 8)]
            for (a, b) in zip(mid, mid[1:]):
                cv.line(a[0], a[1], b[0], b[1], tup(CRV[1] if sd > 0 else CRV[2]))
    # branches (behind the trunk)
    for (dx, by, w, h, rh) in CASTLE_TURRETS:
        sd = 1 if dx > 0 else -1
        p0 = (cx + sd * 6, by + 24 if abs(dx) > 40 else by + 18)
        p1 = (cx + dx * 0.55, by + 13)
        p2 = (cx + dx, by + 3)
        stroke(cv, p0, p1, p2, 5.4, 3.0, CRV)
    # trunk
    ytop = 46
    bins = [0, .1, .27, .46, .64, .83, 1.01]
    fcol = [6, 5, 4, 3, 2, 1]
    for y in range(ytop, GY):
        t = (y - ytop) / (GY - ytop)
        hw = 10 + 3 * t + 10 * t ** 5
        x0 = int(round(cx - hw))
        x1 = int(round(cx + hw))
        wd = x1 - x0 + 1
        for x in range(x0, x1 + 1):
            u = (x - x0 + 0.5) / wd
            k = next(i for i in range(6) if u < bins[i + 1])
            cc = fcol[k]
            if k > 0 and (x - x0) == int(bins[k] * wd):
                cc = min(6, cc + 2)
            for yc in (68, 92, 118):
                if y - yc == int(abs(x - cx) * 0.45):
                    cc = 6 if x < cx else max(1, cc - 1)
            cv.px(x, y, tup(CRV[cc]))
        cv.px(x0, y, tup(CRV[0]))
        cv.px(x1, y, tup(CRV[0]))
    # balcony collar where the branches split
    cv.rect(cx - 15, 86, 31, 3, tup(CRT[4]))
    cv.hline(cx - 15, cx + 15, 86, tup(CRT[5]))
    cv.hline(cx - 15, cx + 15, 89, tup(CRV[0]))
    cv.hline(cx - 15, cx + 15, 82, 'gold')
    for x in range(cx - 15, cx + 16, 3):
        cv.vline(x, 83, 85, 'gold')
    cv.hline(cx - 13, cx + 13, 56, 'gold')
    for x in range(cx - 12, cx + 13, 3):
        cv.vline(x, 57, 58, 'gold')
    # turrets
    for (dx, by, w, h, rh) in CASTLE_TURRETS:
        tx = cx + dx
        cv.tri((tx - w // 2 - 1, by - 1), (tx + w // 2 + 1, by - 1), (tx, by + 5), tup(CRV[2]))
        cv.line(tx - w // 2 - 1, by - 1, tx, by + 5, tup(CRV[0]))
        cv.line(tx + w // 2 + 1, by - 1, tx, by + 5, tup(CRV[0]))
        cylinder(cv, tx - w // 2, by - h, w, h, CRT)
        cv.hline(tx - w // 2, tx + w // 2 - 1, by - h, 'gold')
        cv.hline(tx - w // 2, tx + w // 2 - 1, by - 1, tup(CRV[1]))
        cv.circle(tx, by - h // 2, 3, 'gold')
        cv.circle(tx, by - h // 2, 2, 'crystal')
        cv.px(tx - 1, by - h // 2 - 1, 'crystal_lt')
        cv.px(tx, by - h // 2, mix('crystal_lt', 'white', .5))
        meta.append((tx, by - h // 2, 2))
        cone(cv, tx, by - h - rh, by - h + 1, w // 2 + 2, CRR, curve=1.35)
        cv.vline(tx, by - h - rh - 4, by - h - rh, tup(CRR[5]))
        cv.px(tx, by - h - rh - 5, 'white')
        cv.hline(tx - w // 2 - 2, tx + w // 2 + 2, by - h + 1, tup(CRR[1]))
    # bead strings
    for (ax, ay, bx2, by2) in ((cx - 44, 86, cx - 12, 92), (cx + 12, 90, cx + 46, 82)):
        for i in range(0, 21, 2):
            t = i / 20
            x = ax + (bx2 - ax) * t
            y = ay + (by2 - ay) * t + math.sin(t * math.pi) * 6
            cv.px(x, y, 'crystal_lt' if i % 4 else 'crystal_pk')
    # windows on the trunk
    win_arch(cv, cx - 4, 96, 8, 13, frame=CRV, glass='crystal_lt', meta=meta)
    win_arch(cv, cx - 12, 102, 4, 9, frame=CRV, glass='crystal', meta=meta)
    win_arch(cv, cx + 8, 102, 4, 9, frame=CRV, glass='crystal', meta=meta)
    win_arch(cv, cx - 3, 62, 6, 10, frame=CRV, glass='crystal_lt', meta=meta)
    cv.circle(cx, 76, 3, tup(CRV[0]))
    cv.circle(cx, 76, 2, 'crystal_lt')
    small_star(cv, cx, 76, 'magic')
    # door
    m = arch_mask(16, 24)
    paint(cv, cx - 8, GY - 24, m, np.zeros(m.shape, int), R(tup(CRT[5])))
    m = arch_mask(14, 22)
    paint(cv, cx - 7, GY - 22, m, np.zeros(m.shape, int), R(tup(CRV[0])))
    m = arch_mask(12, 21)
    xx = np.arange(12)[None, :].repeat(21, 0)
    paint(cv, cx - 6, GY - 21, m, (xx >= 6).astype(int) + (xx % 6 == 2), R('royal', 'royal_lt', 'crystal'))
    cv.vline(cx, GY - 18, GY - 1, tup(CRV[0]))
    cv.px(cx - 2, GY - 10, 'gold')
    cv.px(cx + 1, GY - 10, 'gold')
    small_star(cv, cx, GY - 27, 'magic', 'plum')
    cv.rect(cx - 11, GY - 2, 22, 2, tup(CRT[4]))
    cv.hline(cx - 11, cx + 10, GY - 2, tup(CRT[5]))
    # crown roof + star
    cone(cv, cx, 22, 48, 14, CRR, curve=1.3)
    cv.hline(cx - 14, cx + 14, 48, tup(CRR[1]))
    cv.hline(cx - 13, cx + 13, 49, 'gold')
    crystal_star(cv, cx, 15, 8, 3.4)
    meta.append(('star', cx, 15))


# ------------------------------------------------------------------ SCHOOL OF FRIENDSHIP
def school(cv, meta):
    x0, x1 = SCHOOL_X0, SCHOOL_X1
    cx = (x0 + x1) // 2
    wl = WALL['cream']
    # wings
    for (wx, ww) in ((x0 + 10, 22), (x1 - 10 - 22, 22)):
        cv.rect(wx, 112, ww, GY - 112, tup(wl[0]))
        cv.rect(wx + 1, 112, ww - 2, GY - 113, tup(wl[2]))
        cv.rect(wx + ww - 3, 112, 2, GY - 113, tup(wl[1]))
        cv.rect(wx, GY - 4, ww, 4, tup(STONE[2]))
        cv.hline(wx, wx + ww - 1, GY - 4, tup(STONE[3]))
        cv.hline(wx, wx + ww - 1, 126, 'lilac')
        for i in range(2):
            win_arch(cv, wx + 4 + i * 9, 129, 5, 11, frame=SR, glass='baby_blue', meta=meta)
            win_round(cv, wx + 6 + i * 9, 118, 2, frame=SR, box=False, meta=meta)
        pts = [(wx - 3, 113), (wx + ww + 2, 113), (wx + ww - 4, 100), (wx + 3, 100)]
        cv.poly(pts, tup(SR[3]))
        for y in range(100, 113):
            if (y - 100) % 3 == 2:
                cv.hline(int(wx - 3 + (113 - y) * 0.45), int(wx + ww + 2 - (113 - y) * 0.45), y, tup(SR[2]))
        for (a, b) in zip(pts, pts[1:] + pts[:1]):
            cv.line(a[0], a[1], b[0], b[1], tup(SR[0]))
        cv.line(wx + ww - 4, 100, wx + ww + 2, 113, tup(SR[1]))
        small_star(cv, wx + ww // 2, 107, 'gold')
    # central block
    bx0, bx1 = cx - 20, cx + 20
    cv.rect(bx0, 94, bx1 - bx0, GY - 94, tup(wl[0]))
    cv.rect(bx0 + 1, 94, bx1 - bx0 - 2, GY - 95, tup(wl[2]))
    cv.vline(bx0 + 1, 94, GY - 2, tup(wl[3]))
    cv.rect(bx1 - 4, 94, 3, GY - 95, tup(wl[1]))
    for px_ in (bx0 + 2, bx1 - 5):
        cv.rect(px_, 98, 2, GY - 98, mix('lilac', 'white', .3))
        cv.vline(px_ + 1, 98, GY - 1, 'lilac')
    cv.rect(bx0, GY - 4, bx1 - bx0, 4, tup(STONE[2]))
    cv.hline(bx0, bx1 - 1, GY - 4, tup(STONE[3]))
    cone(cv, cx, 58, 96, 23, SR, curve=1.0, shingles=3)
    cv.hline(cx - 23, cx + 23, 96, tup(SR[0]))
    cv.hline(cx - 22, cx + 22, 95, 'gold')
    # rose window with star
    cv.circle(cx, 110, 8, tup(SR[0]))
    cv.circle(cx, 110, 7, 'gold')
    cv.circle(cx, 110, 6, mix('periwinkle', 'baby_blue', .4))
    cv.circle(cx, 110, 4, 'periwinkle')
    crystal_star(cv, cx, 110, 4, 1.8, R('plum', 'magenta', 'magic', 'bubblegum', 'cotton', 'white'))
    for a in range(8):
        cv.px(cx + round(math.cos(a * math.pi / 4) * 6), 110 + round(math.sin(a * math.pi / 4) * 6), 'gold')
    meta.append((cx, 110, 6))
    # gable window in roof
    win_round(cv, cx, 80, 3, frame=SR, box=False, meta=meta)
    # door
    m = arch_mask(16, 22)
    paint(cv, cx - 8, GY - 22, m, np.zeros(m.shape, int), R('gold'))
    m = arch_mask(14, 21)
    xx = np.arange(14)[None, :].repeat(21, 0)
    paint(cv, cx - 7, GY - 21, m, (xx >= 7).astype(int), R(mix('royal', 'royal_lt', .3), 'royal'))
    cv.vline(cx, GY - 17, GY - 1, tup(SR[0]))
    small_star(cv, cx - 3, GY - 12, 'gold')
    small_star(cv, cx + 3, GY - 12, 'gold')
    cv.rect(cx - 11, GY - 2, 22, 2, 'paper')
    cv.hline(cx - 11, cx + 10, GY - 2, 'white')
    # belfry on the roof
    cylinder(cv, cx - 4, 50, 9, 10, WALL['cream'])
    cv.rect(cx - 2, 53, 5, 5, mix('royal', 'ink', .3))
    cv.px(cx, 55, 'gold')
    cone(cv, cx, 40, 50, 6, SR, curve=1.2)
    small_star(cv, cx, 37, 'gold')
    meta.append(('flag', cx + 1, 31, 'school'))
    cv.vline(cx, 30, 36, 'steel_hi')
    # turrets
    for tx in (x0, x1 - 13):
        cylinder(cv, tx, 90, 13, GY - 90, WALL['cream'])
        cv.rect(tx, GY - 4, 13, 4, tup(STONE[2]))
        cv.hline(tx, tx + 12, 98, 'lilac')
        cv.hline(tx, tx + 12, 124, 'lilac')
        win_arch(cv, tx + 5, 103, 3, 8, frame=SR, meta=meta)
        win_arch(cv, tx + 5, 128, 3, 8, frame=SR, meta=meta)
        cone(cv, tx + 6, 60, 91, 9, SR, curve=1.45, shingles=3)
        cv.hline(tx - 3, tx + 15, 91, tup(SR[0]))
        cv.hline(tx - 2, tx + 14, 90, 'gold')
        small_star(cv, tx + 6, 57, 'gold')
        cv.vline(tx + 6, 54, 56, 'gold')
        meta.append(('banner', tx + 4, 99, 'school'))


# ------------------------------------------------------------------ TOWN HALL
def town_hall(cv, meta):
    x0, x1 = HALL_X0, HALL_X1
    cx = (x0 + x1) // 2
    wl = WALL['butter']
    cv.rect(x0, 110, x1 - x0, GY - 110, tup(wl[0]))
    cv.rect(x0 + 1, 110, x1 - x0 - 2, GY - 111, tup(wl[2]))
    cv.vline(x0 + 1, 110, GY - 2, tup(wl[3]))
    cv.rect(x1 - 4, 110, 3, GY - 111, tup(wl[1]))
    for px_ in range(x0 + 1, x1 - 1, 15):
        cv.rect(px_, 112, 2, GY - 115, tup(TIMB[1]))
        cv.vline(px_, 112, GY - 4, tup(TIMB[2]))
    cv.rect(x0 + 1, 127, x1 - x0 - 2, 2, tup(TIMB[1]))
    cv.hline(x0 + 1, x1 - 2, 127, tup(TIMB[2]))
    cv.rect(x0, GY - 4, x1 - x0, 4, tup(STONE[2]))
    cv.hline(x0, x1 - 1, GY - 4, tup(STONE[3]))
    for i, wx in enumerate((x0 + 6, x0 + 20, x1 - 25, x1 - 11)):
        win_arch(cv, wx, 133, 5, 9, shutters='sky' if i % 2 else 'aqua', box=True, seed=wx, meta=meta)
        win_round(cv, wx + 2, 119, 3, box=False, meta=meta)
    # door + pillars + balcony
    door(cv, cx - 7, GY - 1, 14, 19)
    cv.vline(cx, GY - 16, GY - 2, tup(WOOD[0]))
    for px_ in (cx - 11, cx + 9):
        cv.rect(px_, 124, 3, GY - 125, 'paper')
        cv.vline(px_ + 2, 124, GY - 2, 'stone')
        cv.rect(px_ - 1, 123, 5, 2, 'white')
    cv.rect(cx - 13, 121, 27, 2, 'paper')
    cv.hline(cx - 13, cx + 13, 120, 'gold')
    for x in range(cx - 13, cx + 14, 2):
        cv.vline(x, 116, 119, 'gold')
    cv.hline(cx - 13, cx + 13, 115, 'gold')
    cv.rect(cx - 4, 111, 9, 9, tup(WOOD[0]))
    cv.rect(cx - 3, 112, 7, 8, 'baby_blue')
    cv.vline(cx, 112, 119, tup(WOOD[1]))
    cv.px(cx - 2, 113, 'white')
    # heart plaque
    cv.px(cx - 1, 108, 'hot_pink'); cv.px(cx + 1, 108, 'hot_pink')
    cv.hline(cx - 2, cx + 2, 109, 'hot_pink'); cv.hline(cx - 1, cx + 1, 110, 'rose')
    # roof + tower + spire
    thatch(cv, x0 - 6, x1 + 5, 114, 23, seed=41, p=2.8, ramp=RED_TILE, streaks=False, tiers=3, crown=False)
    for xr in range(x0 - 4, x1 + 4, 4):
        cv.px(xr, roof_y(x0 - 6, x1 + 5, 114, 23, xr, p=2.8) + 1, 'blush')
    cv.hline(x0 - 6, x1 + 5, 113, 'paper')
    cv.hline(x0 + 1, x1 - 2, 114, tup(wl[1]))
    for dxr in (x0 + 12, x1 - 12):
        thatch(cv, dxr - 6, dxr + 6, 102, 6, seed=dxr, tiers=99, streaks=False, crown=False, ramp=RED_TILE)
        win_round(cv, dxr, 104, 2, box=False, meta=meta)
    tx0 = cx - 11
    cv.rect(tx0, 68, 22, 28, tup(wl[0]))
    cv.rect(tx0 + 1, 68, 20, 28, tup(wl[2]))
    cv.vline(tx0 + 1, 68, 95, tup(wl[3]))
    cv.rect(tx0 + 17, 68, 4, 28, tup(wl[1]))
    for px_ in (tx0 + 1, tx0 + 19):
        cv.rect(px_, 70, 2, 26, tup(TIMB[1]))
    cv.rect(tx0 - 1, 66, 24, 3, 'paper')
    cv.hline(tx0 - 1, tx0 + 22, 68, 'stone')
    cv.circle(cx, 80, 5, 'gold')
    cv.circle(cx, 80, 4, 'baby_blue')
    cv.rect(cx - 3, 81, 7, 3, mix('baby_blue', 'periwinkle', .5))
    cv.hline(cx - 4, cx + 4, 80, 'gold')
    cv.vline(cx, 76, 84, 'gold')
    cv.px(cx - 2, 78, 'white')
    meta.append((cx, 80, 4))
    cv.rect(tx0 - 2, 92, 26, 2, tup(TIMB[1]))
    thatch(cv, tx0 - 4, tx0 + 25, 100, 8, seed=42, tiers=99, crown=False, ramp=RED_TILE, streaks=False)
    cone(cv, cx, 36, 66, 14, RED_TILE, curve=1.25, shingles=3)
    cv.hline(cx - 14, cx + 14, 66, 'paper')
    cv.rect(cx - 1, 36, 3, 4, 'gold')
    cv.px(cx, 35, 'butter')
    cv.vline(cx, 24, 35, 'steel_hi')
    cv.px(cx, 23, 'gold')
    meta.append(('flag', cx + 1, 25, 'hall'))


def clock_tower(cv, meta):
    cx = CLOCK_X
    x0 = cx - 7
    cylinder(cv, x0, 62, 14, GY - 62, STONE, peak=0.25)
    for y in range(66, GY - 3, 4):
        off = 2 if (y // 4) % 2 else 0
        for x in range(x0 + 1 + off, x0 + 13, 5):
            cv.px(x, y, tup(STONE[1]))
    cv.rect(x0 - 1, GY - 8, 16, 8, tup(STONE[2]))
    cv.hline(x0 - 1, x0 + 14, GY - 8, tup(STONE[4]))
    cv.vline(x0 + 14, GY - 8, GY - 1, tup(STONE[1]))
    door(cv, cx - 3, GY - 1, 6, 10)
    win_arch(cv, cx - 1, 104, 2, 7, frame=STONE)
    win_arch(cv, cx - 1, 116, 2, 7, frame=STONE)
    cv.circle(cx, 80, 6, 'gold')
    cv.circle(cx, 80, 5, 'paper')
    for a in range(12):
        cv.px(cx + round(math.sin(a * math.pi / 6) * 4), 80 - round(math.cos(a * math.pi / 6) * 4),
              'shadow' if a % 3 else 'ink')
    cv.rect(x0 - 1, 60, 16, 3, tup(STONE[3]))
    cv.hline(x0 - 1, x0 + 14, 62, tup(STONE[1]))
    cv.rect(x0 + 2, 50, 10, 10, tup(STONE[2]))
    m = arch_mask(6, 8)
    paint(cv, cx - 3, 51, m, np.zeros(m.shape, int), R('night'))
    cv.circle(cx, 56, 2, 'gold')
    cv.hline(cx - 2, cx + 2, 58, 'amber')
    cv.px(cx - 1, 55, 'butter')
    cv.rect(x0 + 1, 48, 12, 2, tup(STONE[3]))
    cone(cv, cx, 30, 48, 9, BR, curve=1.3, shingles=3)
    cv.px(cx, 28, 'gold')
    cv.px(cx, 29, 'gold')


# ------------------------------------------------------------------ market
def stall(cv, x, kind, cols, meta):
    w = 22
    for px_ in (x + 1, x + w - 3):
        cv.rect(px_, 118, 2, GY - 118, tup(WOOD[1]))
        cv.vline(px_, 118, GY - 1, tup(WOOD[3]))
    cv.rect(x, 135, w, GY - 135, tup(WOOD[0]))
    for y in range(136, GY, 3):
        cv.rect(x + 1, y, w - 2, 2, tup(WOOD[2]))
        cv.hline(x + 1, x + w - 2, y, tup(WOOD[3]))
    cv.rect(x - 1, 134, w + 2, 2, tup(WOOD[3]))
    # goods
    for i in range(w // 3):
        gx = x + 1 + i * 3
        if kind == 'apple':
            for (ox, oy) in ((0, 0), (1, -1), (2, 0)):
                cv.px(gx + ox, 133 + oy, 'red' if hash01(i, ox) < 0.7 else 'gold')
            cv.px(gx + 1, 132, 'white') if i % 3 == 0 else None
            cv.px(gx, 131, 'red')
        elif kind == 'carrot':
            cv.vline(gx, 131, 133, 'orange')
            cv.vline(gx + 1, 132, 133, 'amber')
            cv.px(gx, 130, 'leaf')
            cv.px(gx + 1, 129, 'green')
        else:
            cv.rect(gx, 131, 2, 3, tup(WOOD[2]))
            cv.px(gx, 129, FLOWERS[i % 7])
            cv.px(gx + 1, 128, FLOWERS[(i + 3) % 7])
            cv.px(gx + 1, 130, 'green')
    # awning (stripes, sloped) + scallops
    for y in range(114, 122):
        sl = (y - 114) // 3
        for i in range(-2, w + 2):
            c0 = cols[0] if ((i + 30) // 3) % 2 == 0 else cols[1]
            cv.px(x + i - sl, y, c0 if y < 120 else shade(c0, .82))
    cv.hline(x - 2, x + w + 1, 113, shade(cols[0], .6))
    for i in range(-4, w + 1, 3):
        c0 = cols[0] if ((i + 32) // 3) % 2 == 0 else cols[1]
        cv.hline(x + i, x + i + 2, 122, shade(c0, .82))
        cv.px(x + i + 1, 123, shade(c0, .7))
    signboard(cv, x + w // 2 - 4, 124, kind)
    meta.append(('stall', x, kind))


def garden(cv, meta):
    x0, x1 = GARDEN_X0, GARDEN_X1
    cv.rect(x0, GY - 3, x1 - x0, 3, mix('rust', 'plum_brown', .4))
    cv.hline(x0, x1 - 1, GY - 3, 'rust')
    for i, x in enumerate(range(x0 + 2, x1 - 2, 6)):
        h = 18 + int(hash01(i, 77) * 6)
        cv.vline(x, GY - h, GY - 3, 'green')
        cv.px(x + 1, GY - h // 2, 'leaf')
        cv.px(x - 1, GY - h // 2 - 3, 'leaf')
        cv.circle(x, GY - h - 1, 2, 'gold')
        cv.px(x, GY - h - 1, 'rust')
        cv.px(x - 1, GY - h - 2, 'butter')
    for x in range(x0 + 4, x1 - 2, 6):
        blob(cv, [(x, GY - 6, 2.2), (x + 1, GY - 7, 1.6)], GR, seed=x, tex=0)
    for i in range(x1 - x0):
        if hash01(i, 5) < 0.5:
            cv.px(x0 + i, GY - 4 - int(hash01(i, 6) * 3), FLOWERS[int(hash01(i, 7) * 7)])
    picket(cv, x0 - 1, x1 + 1, GY, 8)
    # watering can
    cv.rect(x1 - 7, GY - 5, 4, 4, 'steel')
    cv.line(x1 - 3, GY - 4, x1 - 1, GY - 6, 'steel')


def laundry_posts(cv):
    for x in (LAUNDRY_X0, LAUNDRY_X1):
        cv.rect(x, 122, 2, GY - 122, tup(WOOD[1]))
        cv.vline(x, 122, GY - 1, tup(WOOD[3]))
        cv.hline(x - 1, x + 2, 122, tup(WOOD[0]))


# ------------------------------------------------------------------ SUGARCUBE CORNER
def sugarcube(cv, meta):
    x0, x1 = SUGAR_X0, SUGAR_X1
    cx = (x0 + x1) // 2
    cv.rect(x0, 102, x1 - x0, GY - 102, tup(GB[0]))
    cv.rect(x0 + 1, 102, x1 - x0 - 2, GY - 103, tup(GB[2]))
    cv.vline(x0 + 1, 102, GY - 2, tup(GB[4]))
    cv.vline(x0 + 2, 102, GY - 2, tup(GB[3]))
    cv.rect(x1 - 5, 102, 4, GY - 103, tup(GB[1]))
    for i in range(60):
        cv.px(x0 + 3 + int(hash01(i, 1) * (x1 - x0 - 8)), 104 + int(hash01(i, 2) * 40), tup(GB[1]))
    # icing zigzags
    for yz in (124, 102):
        for x in range(x0, x1):
            cv.px(x, yz + (1 if (x // 2) % 2 else 0), 'white')
            if (x // 2) % 2:
                cv.px(x, yz + 2, 'paper')
    # candy-cane posts
    for px_ in (x0 - 2, x1):
        for y in range(106, GY):
            cv.px(px_, y, 'red' if ((y + px_) // 2) % 2 else 'white')
            cv.px(px_ + 1, y, 'crimson' if ((y + px_ + 1) // 2) % 2 else 'paper')
        cv.px(px_ + (1 if px_ < cx else 0), 105, 'red')
        cv.px(px_ + (2 if px_ < cx else -1), 104, 'white')
        cv.px(px_ + (3 if px_ < cx else -2), 105, 'red')
    # display window with cakes
    cv.circle(cx - 15, 135, 8, 'white')
    cv.circle(cx - 15, 135, 7, 'bubblegum')
    cv.circle(cx - 15, 135, 6, mix('baby_blue', 'cream', .4))
    cv.hline(cx - 21, cx - 9, 138, 'paper')
    for i, cc in enumerate(('bubblegum', 'cream', 'lilac', 'peach')):
        bx = cx - 20 + i * 3
        cv.rect(bx, 135, 3, 3, cc)
        cv.px(bx + 1, 134, 'red' if i % 2 else 'white')
    cv.px(cx - 18, 131, 'white')
    cv.px(cx - 17, 130, 'white')
    meta.append((cx - 15, 135, 6))
    # door
    m = arch_mask(11, 16)
    paint(cv, cx + 3, GY - 16, m, np.zeros(m.shape, int), R('white'))
    m = arch_mask(9, 15)
    paint(cv, cx + 4, GY - 15, m, np.zeros(m.shape, int), R('raspberry'))
    cv.rect(cx + 5, GY - 11, 7, 5, 'baby_blue')
    cv.px(cx + 10, GY - 6, 'gold')
    cv.rect(cx + 2, GY - 1, 13, 1, 'paper')
    signboard(cv, cx + 15, 128, 'cupcake', board='cotton')
    # awning over display window
    for y in range(120, 124):
        for x in range(cx - 25, cx - 4):
            cv.px(x - (y - 120) // 2, y, 'hot_pink' if (x // 2) % 2 else 'white')
    for x in range(cx - 27, cx - 5, 2):
        cv.px(x, 124, 'hot_pink')
    # upper windows with icing frames
    for wx in (cx - 16, cx, cx + 16):
        cv.circle(wx, 112, 4, 'white')
        cv.circle(wx, 112, 3, 'bubblegum')
        cv.circle(wx, 112, 2, 'baby_blue')
        cv.px(wx - 1, 111, 'white')
        meta.append((wx, 112, 2))
    # frosting roof
    rx0, rx1, yb, rh = x0 - 6, x1 + 5, 106, 25
    chx = x1 - 12
    ry = roof_y(rx0, rx1, yb, rh, chx)
    thatch(cv, rx0, rx1, yb, rh, seed=7, ramp=FR, streaks=False, tiers=7, crown=False)
    for i in range(0, rx1 - rx0, 2):
        x = rx0 + i
        top = roof_y(rx0, rx1, yb, rh, x)
        if hash01(i, 11) < 0.45 and top < yb - 2:
            cv.px(x, top + 2 + int(hash01(i, 12) * (yb - top - 4)), FLOWERS[int(hash01(i, 13) * 7)])
    for x in range(rx0 + 1, rx1):
        dl = 1 + int(hash01(x, 21) * 3) if hash01(x, 22) < 0.5 else 0
        cv.vline(x, yb - 2, yb - 1 + dl, 'white')
        if dl:
            cv.px(x, yb + dl, 'paper')
    # oven chimney
    cv.rect(chx, ry - 8, 6, 10, 'brick')
    cv.vline(chx, ry - 8, ry + 1, 'coral')
    cv.vline(chx + 5, ry - 8, ry + 1, 'maroon')
    cv.rect(chx - 1, ry - 10, 8, 3, 'white')
    cv.px(chx - 1, ry - 7, 'white'); cv.px(chx + 6, ry - 7, 'white')
    meta.append(('steam', chx + 3, ry - 11, 0))
    # the giant cupcake
    top = roof_y(rx0, rx1, yb, rh, cx)
    by = top + 3
    pts = [(cx - 10, by), (cx + 10, by), (cx + 13, by - 13), (cx - 13, by - 13)]
    cv.poly(pts, tup(CUP[2]))
    for i, x in enumerate(range(cx - 12, cx + 13)):
        u = (x - (cx - 12)) / 25
        cc = CUP[4] if u < 0.25 else (CUP[3] if u < 0.55 else (CUP[2] if u < 0.8 else CUP[1]))
        cv.vline(x, by - 13, by - 1, tup(cc))
        if x % 3 == 0:
            cv.vline(x, by - 12, by - 1, tup(CUP[1]))
    cv.poly(pts[:2] + [(cx + 13, by + 1)], (0, 0, 0)) if False else None
    for (a, b) in zip(pts, pts[1:] + pts[:1]):
        cv.line(a[0], a[1], b[0], b[1], tup(CUP[0]))
    for (fy, rx, ry2) in ((by - 16, 15, 5), (by - 21, 12, 4), (by - 26, 8, 3), (by - 30, 4, 2)):
        cv.ellipse(cx, fy, rx + 1, ry2 + 1, tup(FR[0]))
    for (fy, rx, ry2) in ((by - 16, 15, 5), (by - 21, 12, 4), (by - 26, 8, 3), (by - 30, 4, 2)):
        cv.ellipse(cx, fy, rx, ry2, tup(FR[3]))
        cv.ellipse(cx - rx // 3, fy - 1, rx // 2, max(1, ry2 - 2), tup(FR[4]))
        cv.hline(cx + rx // 3, cx + rx - 1, fy + ry2 - 1, tup(FR[2]))
        cv.px(cx - rx // 2, fy - ry2 + 1, tup(FR[5]))
    for i in range(14):
        sx = cx - 12 + int(hash01(i, 31) * 24)
        sy = by - 32 + int(hash01(i, 32) * 18)
        if cv.a[sy, sx].tolist() in (list(FR[3]), list(FR[4])):
            cv.px(sx, sy, ('gold', 'aqua', 'white', 'lavender', 'leaf')[i % 5])
    cv.circle(cx + 1, by - 35, 3, 'crimson')
    cv.circle(cx + 1, by - 35, 2, 'red')
    cv.px(cx, by - 36, 'white')
    cv.line(cx + 2, by - 38, cx + 4, by - 42, 'green')
    meta.append(('cherry', cx + 1, by - 35))


# ------------------------------------------------------------------ CAROUSEL BOUTIQUE
WB = R(mix('lilac', 'plum_brown', .55), mix('lilac', 'violet', .2), mix('lilac', 'white', .45),
       mix('lilac', 'white', .7), 'white')


def boutique(cv, meta):
    cx = BOUT_X
    r = 26
    x0 = cx - r
    cylinder(cv, x0, 104, 2 * r, GY - 104, WB, peak=0.3)
    for k in range(-3, 4):
        th = k * 0.42
        px_ = int(round(cx + math.sin(th) * (r - 2)))
        cv.rect(px_ - 1, 108, 3, GY - 112, 'lavender')
        cv.vline(px_ - 1, 108, GY - 5, 'lilac')
        cv.vline(px_ + 1, 108, GY - 5, 'violet')
        cv.rect(px_ - 2, 107, 5, 2, 'gold')
        gem(cv, px_, 116, ('crystal', 'crystal_pk', 'aqua')[k % 3])
    for k in (-2, -1, 1, 2):
        th = (k - 0.5 * (1 if k > 0 else -1)) * 0.42
        wx = int(round(cx + math.sin(th) * (r - 2)))
        win_arch(cv, wx - 2, 124, 4, 11, frame=R('gold', 'amber', 'gold'), glass='periwinkle', meta=meta)
        cv.px(wx - 2, 126, 'bubblegum'); cv.px(wx + 1, 126, 'bubblegum')
    # door with diamond motif
    m = arch_mask(14, 22)
    paint(cv, cx - 7, GY - 22, m, np.zeros(m.shape, int), R('gold'))
    m = arch_mask(12, 21)
    xx = np.arange(12)[None, :].repeat(21, 0)
    paint(cv, cx - 6, GY - 21, m, (xx >= 6).astype(int), R('periwinkle', 'royal_lt'))
    ICONS['gem'].draw(cv, cx - 2, GY - 17)
    gem(cv, cx - 3, GY - 10, 'crystal')
    gem(cv, cx + 3, GY - 10, 'crystal')
    cv.rect(cx - 9, GY - 2, 18, 2, 'paper')
    cv.rect(x0, GY - 4, 2 * r, 4, tup(STONE[2]))
    cv.hline(x0, x0 + 2 * r - 1, GY - 4, tup(STONE[3]))
    # valance
    for x in range(x0 - 3, x0 + 2 * r + 3):
        k = (x - x0 + 30) // 4
        cv.vline(x, 103, 105, 'bubblegum' if k % 2 else 'gold')
        if (x - x0 + 30) % 4 in (1, 2):
            cv.px(x, 106, 'bubblegum' if k % 2 else 'amber')
    # big carousel roof
    cone(cv, cx, 70, 103, r + 6, BR, curve=1.25, stripes=6)
    cv.hline(cx - r - 6, cx + r + 6, 103, tup(BR[0]))
    cv.hline(cx - r - 5, cx + r + 5, 102, 'gold')
    # upper drum + cone + spire
    cylinder(cv, cx - 9, 60, 18, 12, WB)
    cv.hline(cx - 9, cx + 8, 60, 'gold')
    for i, wx in enumerate((cx - 5, cx, cx + 5)):
        cv.circle(wx, 66, 1, ('crystal', 'crystal_pk', 'aqua')[i])
    cone(cv, cx, 42, 61, 12, BR, curve=1.3, stripes=4)
    cv.hline(cx - 12, cx + 12, 61, 'gold')
    cv.vline(cx, 34, 41, 'gold')
    cv.vline(cx + 1, 36, 41, 'amber')
    cv.circle(cx, 37, 1, 'gold')
    ICONS['gem'].draw(cv, cx - 2, 29)
    meta.append(('gem', cx, 30))
    meta.append(('pennant', cx + 1, 34))


# ------------------------------------------------------------------ station
def station(cv, meta):
    # platform
    cv.rect(PLATFORM_X0, GY - 4, PLATFORM_X1 - PLATFORM_X0, 4, 'grey')
    cv.hline(PLATFORM_X0, PLATFORM_X1 - 1, GY - 4, 'paper')
    cv.hline(PLATFORM_X0, PLATFORM_X1 - 1, GY - 3, 'stone')
    for x in range(PLATFORM_X0, PLATFORM_X1, 7):
        cv.vline(x, GY - 2, GY - 1, 'shadow')
    x0, x1 = STATION_X0, STATION_X1
    cx = (x0 + x1) // 2
    yb = GY - 4
    wl = WALL['cream']
    cv.rect(x0, yb - 30, x1 - x0, 30, tup(wl[0]))
    cv.rect(x0 + 1, yb - 30, x1 - x0 - 2, 29, tup(wl[2]))
    cv.vline(x0 + 1, yb - 30, yb - 1, tup(wl[3]))
    cv.rect(x1 - 4, yb - 30, 3, 29, tup(wl[1]))
    cv.rect(x0 + 1, yb - 14, x1 - x0 - 2, 2, 'hot_pink')
    for px_ in (x0 + 1, x1 - 3, cx - 1):
        cv.rect(px_, yb - 29, 2, 28, tup(TIMB[1]))
    door(cv, cx + 4, yb - 1, 8, 14, wood=R('raspberry', 'hot_pink', 'bubblegum', 'cotton'))
    win_arch(cv, x0 + 6, yb - 24, 8, 9, shutters='hot_pink', box=True, seed=3, meta=meta)
    cv.rect(x0 + 5, yb - 12, 10, 1, 'rust')
    signboard(cv, cx - 5, yb - 37, 'train', board='paper')
    cv.circle(cx + 9, yb - 33, 3, 'gold')
    cv.circle(cx + 9, yb - 33, 2, 'paper')
    cv.px(cx + 9, yb - 34, 'ink'); cv.px(cx + 10, yb - 33, 'ink')
    # roof: pink tiled hip with a little gable
    rt = yb - 30
    pts = [(x0 - 5, rt + 2), (x1 + 4, rt + 2), (x1 - 4, rt - 14), (x0 + 3, rt - 14)]
    cv.poly(pts, 'rose')
    for y in range(rt - 14, rt + 2):
        if (y - rt) % 3 == 0:
            k = (rt + 2 - y) * 0.5
            cv.hline(int(x0 - 5 + k), int(x1 + 4 - k), y, 'raspberry')
        for x in range(x0 - 4, x0 + 10):
            pass
    cv.line(x1 - 4, rt - 14, x1 + 4, rt + 2, 'raspberry')
    cv.line(x0 + 3, rt - 14, x0 - 5, rt + 2, 'pink')
    for (a, b) in zip(pts, pts[1:] + pts[:1]):
        cv.line(a[0], a[1], b[0], b[1], 'berry')
    cv.hline(x0 + 3, x1 - 4, rt - 15, 'gold')
    cv.rect(x1 - 14, rt - 22, 4, 9, tup(STONE[2]))
    cv.rect(x1 - 15, rt - 23, 6, 2, tup(STONE[1]))
    meta.append(('smoke', x1 - 12, rt - 24, 77))
    # canopy over the platform
    c0, c1 = x1, x1 + 62
    for px_ in range(c0 + 4, c1, 18):
        cv.rect(px_, yb - 32, 2, 32, 'navy')
        cv.vline(px_, yb - 32, yb - 1, 'indigo')
        cv.line(px_ - 4, yb - 32, px_, yb - 28, 'navy')
        cv.line(px_ + 5, yb - 32, px_ + 1, yb - 28, 'navy')
    for y in range(yb - 38, yb - 32):
        for x in range(c0 - 2, c1 + 3):
            cv.px(x, y, 'hot_pink' if (x // 3) % 2 else 'cotton')
    cv.hline(c0 - 3, c1 + 3, yb - 39, 'berry')
    for x in range(c0 - 2, c1 + 3):
        if x % 3 != 1:
            cv.px(x, yb - 32, 'cotton' if (x // 3) % 2 else 'hot_pink')
        if x % 3 == 0:
            cv.px(x, yb - 31, 'paper')
    for lx in (c0 + 13, c0 + 31, c0 + 49):
        cv.vline(lx, yb - 31, yb - 29, 'night')
        cv.rect(lx - 1, yb - 28, 3, 3, 'gold')
        cv.px(lx, yb - 27, 'butter')
    # bench, churns, luggage
    cv.rect(c0 + 20, yb - 6, 12, 2, tup(WOOD[2]))
    cv.vline(c0 + 21, yb - 5, yb - 1, tup(WOOD[0]))
    cv.vline(c0 + 30, yb - 5, yb - 1, tup(WOOD[0]))
    cv.rect(c0 + 20, yb - 10, 12, 2, tup(WOOD[1]))
    for i in range(2):
        cylinder(cv, c0 + 38 + i * 5, yb - 8, 4, 8, R('shadow', 'steel', 'steel_hi', 'white'))
    cv.rect(c0 + 50, yb - 7, 8, 7, tup(WOOD[2]))
    cv.rect(c0 + 51, yb - 11, 6, 4, 'coral')
    cv.hline(c0 + 50, c0 + 57, yb - 4, tup(WOOD[0]))
    barrel(cv, x0 - 8, yb)
    bush(cv, x0 - 16, yb, 8, 6, seed=91)
    # water tower
    wt = c1 + 14
    for lx in (wt + 2, wt + 15):
        cv.rect(lx, 112, 2, GY - 112, tup(WOOD[0]))
        cv.vline(lx, 112, GY - 1, tup(WOOD[2]))
    cv.line(wt + 3, 140, wt + 15, 116, tup(WOOD[0]))
    cv.line(wt + 3, 116, wt + 15, 140, tup(WOOD[0]))
    cylinder(cv, wt, 96, 19, 17, WOOD)
    for y in (99, 106, 111):
        cv.hline(wt, wt + 18, y, 'shadow')
    thatch(cv, wt - 2, wt + 20, 97, 7, seed=5, ramp=R('maroon', 'brick', 'vermilion', 'coral', 'peach', 'blush'),
           streaks=False, tiers=99, crown=False)
    cv.rect(wt - 4, 106, 5, 2, 'steel_dk')
    cv.vline(wt - 4, 108, 112, 'steel_dk')
    # signal
    sx = wt + 34
    cv.rect(sx, 104, 2, GY - 104, 'steel_dk')
    cv.rect(sx + 2, 106, 8, 2, 'red')
    cv.px(sx + 8, 106, 'white')
    cv.circle(sx + 1, 102, 2, 'night')
    cv.px(sx + 1, 102, 'leaf')
    meta.append(('signal', sx + 1, 102))
    # buffer stop
    bx = PLATFORM_X0 - 12
    cv.rect(bx, 141, 4, 10, tup(WOOD[0]))
    cv.rect(bx + 1, 141, 2, 10, tup(WOOD[2]))
    for y in range(140, 146):
        cv.hline(bx + 3, bx + 8, y, 'red' if (y // 2) % 2 else 'white')
    cv.rect(bx + 8, 141, 2, 2, 'steel')
    cv.rect(bx + 8, 144, 2, 2, 'steel')


def track(cv, x0, x1):
    cv.rect(x0, 150, x1 - x0, 8, mix('mauve', 'tan', .45))
    for x in range(x0, x1):
        for y in range(150, 158):
            if hash01(x, y, 3) < 0.25:
                cv.px(x, y, 'mauve' if hash01(x, y) < 0.6 else mix('tan', 'cream', .5))
    for x in range(x0 + 1, x1, 6):
        cv.rect(x, 152, 4, 2, mix('plum_brown', 'rust', .3))
        cv.hline(x, x + 3, 152, 'mauve')
    cv.hline(x0, x1 - 1, 150, 'steel_hi')
    cv.hline(x0, x1 - 1, 151, 'steel_dk')


# ------------------------------------------------------------------ ground
def lane(cv, x0, x1):
    cv.rect(x0, 148, x1 - x0, 10, tup(LANE[1]))
    rows = ((148, 3, 5), (152, 3, 6), (156, 2, 7))
    for ri, (y, h, w) in enumerate(rows):
        x = x0 - 6 + ri * 2
        i = 0
        while x < x1:
            ww = w + int(hash01(ri, i, 1) * 2)
            k = 2 + (1 if hash01(ri, i, 2) < 0.4 else 0)
            cv.round_rect(x, y, ww, h, tup(LANE[k]), r=1)
            cv.hline(x + 1, x + ww - 3, y, tup(LANE[4]))
            if h > 2:
                cv.hline(x + 2, x + ww - 2, y + h - 1, tup(LANE[1]))
            x += ww + 1
            i += 1
    cv.hline(x0, x1 - 1, 158, tup(GR[2]))
    cv.rect(x0, 159, x1 - x0, 8, tup(GR[3]))
    for x in range(x0, x1):
        hgt = int(hash01(x, 41) * 3)
        cv.vline(x, 158 - hgt, 158, tup(GR[3] if hgt < 2 else GR[4]))
        if hash01(x, 42) < 0.06:
            cv.px(x, 157 - hgt, FLOWERS[int(hash01(x, 43) * 7)])


# ------------------------------------------------------------------ MAIN LAYER
@lru_cache(maxsize=1)
def _main():
    meta = []
    gnd = _new(MAIN_W, H)
    # yard strip behind the lane
    gnd.rect(0, 142, MAIN_W, 6, tup(GR[3]))
    gnd.hline(0, MAIN_W - 1, 142, tup(GR[4]))
    for x in range(MAIN_W):
        if hash01(x, 51) < 0.3:
            gnd.px(x, 141, tup(GR[4]))
    lane(gnd, 0, PLATFORM_X0 - 14)
    track(gnd, PLATFORM_X0 - 14, MAIN_W)
    gnd.rect(PLATFORM_X0 - 14, 158, MAIN_W - PLATFORM_X0 + 14, 9, tup(GR[3]))
    gnd.hline(PLATFORM_X0 - 14, MAIN_W - 1, 158, tup(GR[2]))
    # stream through the yard and under the rail bridge
    sx0, sx1 = STREAM_X0, STREAM_X1
    gnd.rect(sx0, 140, sx1 - sx0, 27, 'sky')
    gnd.rect(sx0 + 2, 140, sx1 - sx0 - 4, 2, 'light_sky')
    for y in range(142, 167, 2):
        for x in range(sx0 + 1, sx1 - 1):
            if hash01(x, y, 61) < 0.18:
                gnd.px(x, y, 'light_sky')
    gnd.vline(sx0, 140, 166, tup(GR[1]))
    gnd.vline(sx1 - 1, 140, 166, tup(GR[2]))

    b = _new(MAIN_W, H)
    # background trees between buildings
    for (tx, r) in ((20, 14), (62, 11), (118, 16), (160, 12), (395, 9), (512, 10), (612, 13),
                    (690, 11), (790, 9), (884, 11), (1070, 12), (1175, 15), (1230, 11), (1300, 16), (1360, 12)):
        tree(b, tx, GY - 2, r, seed=tx)
    # castle grounds (left)
    for (x, w_, h_) in ((4, 18, 7), (40, 14, 6), (84, 20, 8), (140, 16, 7)):
        bush(b, x, GY, w_, h_, seed=x)
    lamp_post(b, 108, GY, 20)
    castle(b, meta)
    school(b, meta)
    town_hall(b, meta)
    garden(b, meta)
    for (sx, kind, cols) in STALLS:
        stall(b, sx, kind, cols, meta)
    clock_tower(b, meta)
    cottage(b, 594, 32, 26, 22, 'cotton', seed=11, meta=meta, dormer=True)
    laundry_posts(b)
    cottage(b, 662, 30, 22, 19, 'mint', seed=12, meta=meta)
    sugarcube(b, meta)
    bush(b, 776, GY, 12, 7, seed=33)
    lamp_post(b, 790, GY, 20)
    boutique(b, meta)
    cottage(b, 860, 30, 24, 20, 'lilac', seed=13, meta=meta)
    station(b, meta)
    # beyond the stream: fields with a cottage
    cottage(b, 1180, 34, 24, 21, 'peach', seed=14, meta=meta, dormer=True)
    cottage(b, 1262, 28, 20, 18, 'blue', seed=15, meta=meta)
    # rail bridge over the stream
    b.rect(sx0 - 2, 148, sx1 - sx0 + 4, 3, tup(STONE[2]))
    b.hline(sx0 - 2, sx1 + 1, 148, tup(STONE[4]))
    b.hline(sx0 - 2, sx1 + 1, 150, 'steel_hi')
    m = np.zeros((10, sx1 - sx0 + 4), bool)
    yy, xx = np.mgrid[0:10, 0:sx1 - sx0 + 4]
    rr = (sx1 - sx0) / 2
    m = ((xx - rr - 2) / (rr + 0.5)) ** 2 + ((yy - 10) / 9.5) ** 2 > 1
    paint(b, sx0 - 2, 151, m, np.zeros(m.shape, int) + ((xx < 4) | (xx > sx1 - sx0 - 1)).astype(int),
          R(tup(STONE[2]), tup(STONE[1])))
    for x in range(sx0 - 2, sx1 + 2, 4):
        b.px(x, 153, tup(STONE[1]))
    # front props on the lane edge
    for (x, kind) in ((176, 'lamp'), (276, 'lamp'), (470, 'barrel'), (585, 'barrel'), (650, 'mail'),
                      (700, 'barrel'), (852, 'mail'), (1166, 'lamp')):
        if kind == 'lamp':
            lamp_post(b, x, GY, 22)
        elif kind == 'barrel':
            barrel(b, x, GY)
        else:
            b.rect(x, GY - 9, 2, 9, tup(WOOD[1]))
            b.rect(x - 2, GY - 13, 6, 5, 'sky')
            b.hline(x - 2, x + 3, GY - 13, 'light_sky')
            b.px(x + 3, GY - 15, 'red')
            b.px(x + 3, GY - 14, 'red')

    # long morning shadows (sun low on the left): cast to the right along the ground
    bm = np.any(b.a != _KEYA, axis=2)
    hgt = np.where(bm[:GY].any(0), GY - np.argmax(bm[:GY], axis=0), 0)
    run = np.zeros(MAIN_W)
    cur = 0.0
    for x in range(MAIN_W):
        cur = max(cur - 1.0, hgt[x] * 0.55)
        run[x] = cur
    thr = bayer(12, MAIN_W, 0, 0)
    reg = gnd.a[140:152].astype(np.float32)
    lv = np.clip(run / 10.0, 0, 1)[None, :]
    rowk = np.linspace(1.0, 0.55, 12)[:, None]
    sh = (thr < lv * rowk) & (run[None, :] > 0)
    tint = np.array([0.74, 0.76, 0.9], np.float32)
    reg[sh] = reg[sh] * tint
    gnd.a[140:152] = reg.astype(np.uint8)
    gnd.blit(_rgba(b), 0, 0)
    return _rgba(gnd), tuple(meta)


# ------------------------------------------------------------------ BACK VILLAGE (0.7)
@lru_cache(maxsize=1)
def _back():
    w = 920
    cv = _new(w, H)
    cv.rect(0, 136, w, 14, tup(GR[3]))
    cv.hline(0, w - 1, 136, tup(GR[4]))
    meta = []
    x = 6
    i = 0
    while x < w - 30:
        r = hash01(i, 101)
        if r < 0.55:
            ww = 20 + int(hash01(i, 102) * 10)
            wh = 13 + int(hash01(i, 103) * 5)
            wall = ('cotton', 'cream', 'mint', 'lilac', 'peach', 'butter')[int(hash01(i, 104) * 6)]
            cv_ = cv
            # small cottage: shift ground to 141
            rw = WALL[wall]
            yb = 141
            yt = yb - wh
            cv_.rect(x, yt, ww, wh, tup(rw[0]))
            cv_.rect(x + 1, yt, ww - 2, wh - 1, tup(rw[2]))
            cv_.rect(x + ww - 3, yt, 2, wh - 1, tup(rw[1]))
            cv_.rect(x + ww // 2 - 2, yb - 7, 4, 7, tup(WOOD[1]))
            for wx in (x + 4, x + ww - 5):
                cv_.circle(wx, yb - 7, 2, tup(TIMB[0]))
                cv_.circle(wx, yb - 7, 1, 'baby_blue')
            rh = 11 + int(hash01(i, 105) * 5)
            chx = x + int(ww * 0.7)
            if hash01(i, 106) < 0.6:
                ry = roof_y(x - 3, x + ww + 2, yt + 3, rh, chx)
                cv_.rect(chx, ry - 4, 3, 6, tup(STONE[2]))
                cv_.rect(chx - 1, ry - 5, 5, 1, tup(STONE[1]))
                meta.append(('smoke', chx + 1, ry - 6, i))
            thatch(cv_, x - 3, x + ww + 2, yt + 3, rh, seed=200 + i, tiers=5)
            x += ww + 6 + int(hash01(i, 107) * 8)
        else:
            r_ = 7 + int(hash01(i, 108) * 6)
            tree(cv, x + r_, 140, r_, seed=300 + i)
            x += 2 * r_ + 2
        i += 1
    arr = _rgba(cv)
    a = arr[..., :3].astype(np.float32)
    a = a * 0.84 + np.array(HAZE, np.float32) * 0.16
    arr[..., :3] = np.clip(a + 0.5, 0, 255).astype(np.uint8)
    return arr, tuple(meta)


# ------------------------------------------------------------------ MID HILLS (0.4)
MILL_X, MILL_Y = 190, 104
LAKE_X0, LAKE_X1 = 470, 700


@lru_cache(maxsize=1)
def _mid():
    w = 720
    cv = _new(w, H)
    xs = np.arange(w)
    GM = hz(GR, 0.32)
    # back hills
    yb_ = 112 + 4 * np.sin(xs / 41.0 + 0.7) + 2.5 * np.sin(xs / 19.0)
    for x in range(w):
        cv.vline(x, int(yb_[x]), 159, tup(GM[3]))
        cv.px(x, int(yb_[x]), tup(GM[4]))
    # front rolling hills with fields
    hills = [(40, 120, 70), (150, 116, 80), (270, 121, 70), (380, 118, 85), (540, 131, 120), (690, 122, 70)]
    yf = np.full(w, 200.0)
    for (hx, hy, hw) in hills:
        yf = np.minimum(yf, hy + ((xs - hx) / hw) ** 2 * 26)
    lake = (xs >= LAKE_X0) & (xs < LAKE_X1)
    fields = hz(R('leaf', 'pale_leaf', mix('lime', 'sand', .35), mix('leaf', 'green', .4), 'moss'), 0.3)
    for x in range(w):
        y0 = int(yf[x])
        for y in range(y0, 160):
            k = int(nz((x + (y - y0) * 2) // 26, (y - y0) // 7, 5) * 5)
            cv.px(x, y, tup(fields[k]))
        cv.px(x, y0, tup(GM[5]))
        cv.px(x, y0 + 1, tup(GM[4]))
    # hedgerows
    for x in range(0, w, 26):
        for y in range(int(yf[min(w - 1, x)]) + 1, 160, 1):
            xx = x + (y - int(yf[min(w - 1, x)])) // 2
            if 0 <= xx < w:
                cv.px(xx, y, tup(GM[2]))
        for k in range(4):
            tx = x + int(hash01(x, k) * 20)
            if tx < w:
                ty = int(yf[tx]) + 2 + int(hash01(x, k, 2) * 14)
                blob(cv, [(tx, ty, 2 + hash01(x, k, 3) * 1.5)], GM, seed=x + k, tex=0)
    # Saddle Lake
    for x in range(LAKE_X0, LAKE_X1):
        sh = 125 + int(2 * math.sin((x - LAKE_X0) / 23.0))
        for y in range(sh, 160):
            t = (y - sh) / 20
            cc = mix('light_sky', 'sky', t * 0.6)
            cv.px(x, y, cc)
        cv.px(x, sh, mix('baby_blue', 'white', .3))
        cv.vline(x, sh + 1, sh + 2, mix(tup(GM[2]), 'sky', .5))
    for i in range(80):
        x = LAKE_X0 + int(hash01(i, 401) * (LAKE_X1 - LAKE_X0))
        y = 129 + int(hash01(i, 402) * 22)
        cv.hline(x, x + 2 + int(hash01(i, 403) * 4), y, mix('baby_blue', 'white', .4))
    # near shore covering the lake bottom
    for x in range(LAKE_X0 - 20, LAKE_X1 + 20):
        if 0 <= x < w:
            y0 = 146 + int(3 * math.sin(x / 13.0)) + int(2 * math.sin(x / 5.0))
            cv.vline(x, y0, 159, tup(GM[3]))
            cv.px(x, y0, tup(GM[4]))
    # jetty + rowboat
    cv.rect(LAKE_X0 + 60, 141, 18, 2, tup(hz(WOOD, .3)[2]))
    for px_ in range(LAKE_X0 + 61, LAKE_X0 + 78, 5):
        cv.vline(px_, 143, 146, tup(hz(WOOD, .3)[0]))
    cv.poly([(LAKE_X0 + 90, 139), (LAKE_X0 + 101, 139), (LAKE_X0 + 99, 142), (LAKE_X0 + 92, 142)],
            tup(hz(R('rust'), .3)[0]))
    cv.hline(LAKE_X0 + 91, LAKE_X0 + 100, 139, mix('clay', HAZE, .3))
    # Everfree treeline (dark, left edge)
    EF = R(mix('slate', 'ink', .4), mix('forest', 'slate', .4), 'slate', mix('forest', 'sage', .3),
           mix('sage', 'deep_teal', .3), 'sage')
    EF = hz(EF, 0.12)
    for i in range(28):
        tx = int(i * 6 + hash01(i, 501) * 4) - 6
        fade = clamp(1 - tx / 170)
        if fade <= 0:
            continue
        r = 5 + hash01(i, 502) * 5 * fade
        ty = 104 + (1 - fade) * 14 + hash01(i, 503) * 4
        blob(cv, [(tx, ty, r), (tx + 3, ty + 3, r * 0.8)], EF, seed=i, tex=0.2)
    for x in range(0, 150):
        top = 112 + int(x * 0.09)
        cv.vline(x, top, 130 + int(x * 0.06), tup(EF[2]))
    # trees & tiny cottages on the hills
    for i in range(26):
        tx = 160 + int(hash01(i, 601) * (w - 170))
        if LAKE_X0 - 6 < tx < LAKE_X1 + 6:
            continue
        ty = int(yf[tx]) + 1
        r = 2 + hash01(i, 602) * 3
        cv.vline(tx, ty - 2, ty, tup(hz(TIMB, .3)[1]))
        blob(cv, [(tx, ty - 2 - r, r), (tx + 1, ty - 1 - r, r * 0.7)], GM, seed=i, tex=0)
    for i in range(9):
        hx = 200 + int(hash01(i, 701) * (w - 220))
        if LAKE_X0 - 10 < hx < LAKE_X1 + 4:
            continue
        hy = int(yf[hx]) + 6
        wl = hz(WALL[('cream', 'cotton', 'mint', 'lilac')[i % 4]], .3)
        cv.rect(hx, hy - 4, 7, 4, tup(wl[2]))
        cv.px(hx + 3, hy - 2, tup(wl[0]))
        thatch(cv, hx - 1, hx + 7, hy - 3, 4, seed=i, ramp=hz(TH, .3), tiers=99, crown=False)
    # windmill
    mx, my = MILL_X, int(yf[MILL_X]) + 2
    for y in range(my - 16, my):
        t = (y - (my - 16)) / 16
        hw = 2 + t * 2
        cv.hline(int(mx - hw), int(mx + hw), y, mix('cream', HAZE, .25))
        cv.px(int(mx + hw), y, mix('tan', HAZE, .25))
    cv.rect(mx - 1, my - 4, 2, 4, mix('rust', HAZE, .3))
    thatch(cv, mx - 3, mx + 3, my - 15, 5, seed=3, ramp=hz(TH, .25), tiers=99, crown=False)
    return _rgba(cv), int(yf[MILL_X]) + 2 - 17


# ------------------------------------------------------------------ FAR HILLS (0.2)
@lru_cache(maxsize=1)
def _far():
    w = 500
    cv = _new(w, H)
    xs = np.arange(w)
    r1 = 100 + 5 * np.sin(xs / 37.0 + 1) + 3 * np.sin(xs / 13.0) + 2 * np.sin(xs / 7.0 + 2)
    r2 = 108 + 4 * np.sin(xs / 29.0 + 2.5) + 2 * np.sin(xs / 11.0)
    c1 = hz(R('sage', 'pale_sage', 'khaki'), 0.62)
    c2 = hz(R('green', 'sage', 'pale_sage'), 0.48)
    for x in range(w):
        cv.vline(x, int(r1[x]), 150, tup(c1[1]))
        cv.px(x, int(r1[x]), tup(c1[2]))
        if hash01(x, 7) < 0.35:
            cv.px(x, int(r1[x]) - 1, tup(c1[1]))
    for x in range(w):
        cv.vline(x, int(r2[x]), 150, tup(c2[1]))
        cv.px(x, int(r2[x]), tup(c2[2]))
        if hash01(x, 8) < 0.5:
            cv.px(x, int(r2[x]) + 2 + int(hash01(x, 9) * 4), tup(c2[0]))
    return _rgba(cv)


# ------------------------------------------------------------------ CANTERLOT (0.1)
MT_X = 262
CITY_BASE = 68
FALLS = ((-48, 72, 98), (-31, 72, 104), (-16, 72, 94))


@lru_cache(maxsize=1)
def _canterlot():
    w = 420
    cv = _new(w, H)
    xs = np.arange(w, dtype=float)
    peaks = [(MT_X, 16, 1.25), (MT_X + 36, 40, 1.0), (MT_X - 40, 54, 0.9), (MT_X + 80, 70, 0.8),
             (MT_X - 95, 82, 0.75)]
    ridge = np.full(w, 200.0)
    for (px_, py, sl) in peaks:
        ridge = np.minimum(ridge, py + np.abs(xs - px_) * sl)
    ridge = ridge + nz(xs.astype(int) // 3, 1, 9) * 2
    # ledge for the city
    ledge = (xs > MT_X - 52) & (xs < MT_X - 8)
    MR = R(mix('purple', 'royal', .3), 'violet', mix('violet', 'lavender', .5), 'lavender', 'lilac')
    for x in range(w):
        y0 = int(ridge[x])
        if y0 >= 120:
            continue
        slope = ridge[min(w - 1, x + 2)] - ridge[max(0, x - 2)]
        lit = slope < 0
        for y in range(y0, 120):
            depth = (y - y0)
            v = (2 if lit else 1) + (1 if (lit and depth < 2) else 0)
            if MT_X - 64 < x < MT_X and CITY_BASE - 34 < y < CITY_BASE + 8:
                v = min(v, 1)
            if MT_X - 62 < x < MT_X - 4 and CITY_BASE <= y < CITY_BASE + 4 + int(nz(x // 2, 0, 4) * 3):
                v = 0
            if depth > 3 and nz(x // 7, y // 7, 3) < 0.55:
                if lit and (5 * x + 4 * y) % 61 < 3:
                    v -= 1
                elif not lit and (5 * x - 4 * y) % 53 < 3:
                    v -= 1
            k = int(clamp(v, 0, 4))
            cc = MR[k]
            haze = 0.36 + 0.2 * clamp((y - 30) / 80)
            cv.px(x, y, mix(tup(cc), HAZE, haze))
        cv.px(x, y0, mix('white', HAZE, .35) if lit else mix(tup(MR[2]), HAZE, .4))
    # the city on the ledge
    base = CITY_BASE
    WH, WS = mix('white', HAZE, .12), mix('cloud_dk', HAZE, .15)
    cv.hline(MT_X - 62, MT_X - 4, base, mix('lilac', HAZE, .3))
    TW = [(-31, 26, 5, 'gold', 8), (-37, 19, 4, 'gold', 6), (-25, 17, 3, 'royal_lt', 6),
          (-44, 12, 3, 'royal_lt', 5), (-20, 11, 4, 'gold', 5), (-49, 8, 4, 'gold', 4),
          (-13, 9, 3, 'royal_lt', 4), (-56, 5, 4, 'royal_lt', 3), (-8, 5, 4, 'gold', 3)]
    for (dx, th, tw, rc, rh) in TW:
        tx = MT_X + dx
        cv.rect(tx, base - th, tw, th, WH)
        cv.vline(tx + tw - 1, base - th, base - 1, WS)
        rcc = mix(rc, HAZE, .1)
        for k in range(rh):
            hw = (tw / 2 + 0.5) * (k + 1) / rh
            cv.hline(int(round(tx + tw / 2 - hw)), int(round(tx + tw / 2 + hw - 1)), base - th - rh + k, rcc)
        cv.px(int(tx + tw / 2), base - th - rh - 1, rcc)
        if th > 8:
            for wy in range(base - th + 3, base - 3, 4):
                cv.px(tx + tw // 2, wy, mix('periwinkle', HAZE, .25))
    # terrace walls and houses with violet roofs
    for (dx, w_) in ((-58, 10), (-46, 8), (-17, 10), (-6, 6)):
        cv.rect(MT_X + dx, base - 4, w_, 4, WH)
        cv.hline(MT_X + dx, MT_X + dx + w_ - 1, base - 5, mix('royal_lt', HAZE, .15))
        cv.px(MT_X + dx + 2, base - 2, mix('periwinkle', HAZE, .2))
    for x in range(MT_X - 60, MT_X - 4, 2):
        cv.px(x, base - 1, WS)
    return _rgba(cv)


# ------------------------------------------------------------------ SKY + CLOUDS + MIST
SUN = (34, 62)


@lru_cache(maxsize=1)
def _sky():
    c = Canvas(W, H)
    c.gradient(0, 0, W, 132, ['#4d9be6', '#62aef0', '#86cbfb', '#b4e2ff', '#e2f4ff'], soft=0.6)
    c.rect(0, 132, W, H - 132, '#e2f4ff')
    fx.glow(c, SUN[0], SUN[1], 150, 'peach', strength=0.26, rings=6)
    fx.glow(c, SUN[0], SUN[1], 34, 'gold', strength=0.22, rings=4)
    c.circle(SUN[0], SUN[1], 7, 'gold')
    c.circle(SUN[0], SUN[1], 6, 'butter')
    c.circle(SUN[0], SUN[1], 4, mix('butter', 'white', .6))
    c.circle(SUN[0] - 1, SUN[1] - 1, 2, 'white')
    return c


@lru_cache(maxsize=1)
def _clouds():
    w = 640
    cv = _new(w, 90)
    specs = [(30, 34, 46), (150, 18, 30), (250, 44, 62), (370, 22, 38), (470, 50, 54), (560, 28, 34)]
    for i, (x, y, cw) in enumerate(specs):
        circles = []
        n = max(3, cw // 9)
        for k in range(n):
            u = (k + 0.5) / n
            r = cw * (0.13 + 0.12 * math.sin(u * math.pi)) * (0.85 + 0.3 * hash01(i, k))
            circles.append((x + u * cw, y - r * 0.6, r))
        circles.append((x + cw * 0.5, y - cw * 0.12, cw * 0.2))
        blob(cv, circles, CL, outline=False, seed=i, tex=0.05, light=(0.5, 0.45))
        cv.a[int(y) + 1:int(y) + 8, max(0, x - 4):min(w, x + cw + 4)][
            np.all(cv.a[int(y) + 1:int(y) + 8, max(0, x - 4):min(w, x + cw + 4)] != _KEYA, axis=2)] = _KEYA
    return _rgba(cv)


@lru_cache(maxsize=1)
def _mist():
    w = 640
    cv = _new(w, 30)
    for i in range(18):
        x = int(hash01(i, 801) * w)
        y = 8 + int(hash01(i, 802) * 16)
        lw = 40 + int(hash01(i, 803) * 70)
        for dy in range(-3, 4):
            lv = 0.55 * (1 - abs(dy) / 4)
            for xx in range(x - lw // 2, x + lw // 2):
                tt = 1 - abs(xx - x) / (lw / 2)
                if bayer(1, 1, xx, y + dy)[0, 0] < lv * tt * 1.3:
                    cv.px(xx % w, y + dy, mix('white', HAZE, .2))
    return _rgba(cv)


# ------------------------------------------------------------------ FOREGROUND (1.3)
FG_W = 1460


@lru_cache(maxsize=1)
def _fg():
    cv = _new(FG_W, H)
    FG = R(mix('forest', 'ink', .3), 'forest', 'green', mix('green', 'mint', .3), 'leaf', 'pale_leaf')
    xs = np.arange(FG_W)
    top = 166 + 2.0 * np.sin(xs / 23.0) + 1.5 * np.sin(xs / 9.0 + 1)
    for x in range(FG_W):
        y0 = int(top[x])
        cv.vline(x, y0, H - 1, tup(FG[2]))
        cv.vline(x, y0 + 6, H - 1, tup(FG[1]))
        cv.px(x, y0, tup(FG[4]))
        for k in range(3):
            if hash01(x, k, 901) < 0.35:
                yy = y0 + 2 + int(hash01(x, k, 902) * 12)
                cv.px(x, yy, tup(FG[3]))
    # tall grass tufts
    for i in range(240):
        x = int(hash01(i, 911) * FG_W)
        y0 = int(top[x % FG_W])
        h = 3 + int(hash01(i, 912) * 5)
        lean = 1 if hash01(i, 913) < 0.5 else -1
        cv.line(x, y0 + 1, x + lean, y0 - h, tup(FG[3]))
        cv.px(x + lean, y0 - h, tup(FG[5]))
        cv.line(x + 2, y0 + 1, x + 2, y0 - h + 2, tup(FG[2]))
    # wildflowers
    for i in range(180):
        x = int(hash01(i, 921) * FG_W)
        y = int(top[x % FG_W]) - 1 - int(hash01(i, 922) * 4)
        cc = FLOWERS[int(hash01(i, 923) * 7)]
        cv.vline(x, y + 1, y + 3, tup(FG[2]))
        cv.px(x, y, cc)
        if hash01(i, 924) < 0.5:
            cv.px(x + 1, y, cc)
            cv.px(x, y - 1, mix(cc, 'white', .4))
    # picket fence runs
    for (a, b2) in ((60, 150), (330, 420), (610, 700), (880, 950), (1180, 1250)):
        picket(cv, a, b2, 171, 12)
    # bushes
    for i, x in enumerate((10, 240, 520, 800, 1080, 1360)):
        bush(cv, x, H + 2, 34, 13, seed=900 + i, ramp=FG)
    # fg stream + footbridge
    s0, s1 = 1300, 1384
    for x in range(s0, s1):
        y0 = int(top[x]) + 1
        cv.vline(x, y0, H - 1, 'sky')
        cv.px(x, y0, 'light_sky')
    for x in range(s0, s1):
        for y in range(168, H):
            if hash01(x, y, 931) < 0.12:
                cv.px(x, y, 'light_sky')
    cv.vline(s0, 160, H - 1, tup(FG[1]))
    cv.vline(s1 - 1, 160, H - 1, tup(FG[1]))
    for x in range(s0 - 6, s1 + 6):
        t = (x - (s0 - 6)) / (s1 - s0 + 12)
        y = 163 - int(math.sin(t * math.pi) * 5)
        cv.rect(x, y, 1, 3, tup(WOOD[2]))
        cv.px(x, y, tup(WOOD[3]))
        cv.px(x, y + 2, tup(WOOD[0]))
        cv.px(x, y - 6, tup(WOOD[1]))
        if (x - s0) % 7 == 0:
            cv.vline(x, y - 6, y, tup(WOOD[0]))
    return _rgba(cv), top.astype(int)


# ------------------------------------------------------------------ dynamic pieces
def puff(c, x, y, r, color, alpha=1.0):
    x, y = int(x), int(y)
    r = max(1, int(r))
    x0, y0, x1, y1 = c._clip(x - r, y - r, 2 * r + 1, 2 * r + 1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = ((xx - x) ** 2 + (yy - y) ** 2) / (r + 0.5) ** 2
    m = (d <= 1) & (bayer(y1 - y0, x1 - x0, x0, y0) < alpha * (1.2 - 0.55 * d))
    c.a[y0:y1, x0:x1][m] = col(color)


def smoke(c, x, y, f, seed=0, n=5, rise=20, drift=7, big=1.0):
    for i in range(n):
        period = 64
        t = ((f + i * period / n + hash01(seed, i) * period) % period) / period
        px_ = x + int(t * drift + math.sin(t * 5 + i) * 1.2)
        py = y - int(t * rise)
        r = (0.8 + t * 2.6) * big
        a = 0.95 - t * 0.85
        puff(c, px_, py, r, mix('white', 'cloud', min(1.0, t * 1.4)), a)
        if t < 0.3:
            c.px(px_, py, 'white')


def flag(c, x, y, f, w=8, h=5, cols=('hot_pink', 'rose'), emblem='gold', phase=0.0):
    c.vline(x - 1, y - 1, y + h + 1, 'steel_hi')
    for i in range(w):
        dy = int(round(math.sin(f * 0.32 - i * 0.8 + phase) * (i / w) * 1.6))
        sl = math.cos(f * 0.32 - i * 0.8 + phase)
        cc = cols[0] if sl > -0.3 else cols[1]
        c.vline(x + i, y + dy, y + dy + h - 1, cc)
        if i == w // 2 - 1 and emblem:
            c.px(x + i, y + dy + h // 2, emblem)


def banner(c, x, y, f, w=5, h=13, body='royal_lt', dark='royal', emblem='magic', phase=0.0):
    c.hline(x - 1, x + w, y, 'gold')
    for j in range(h):
        dx = int(round(math.sin(f * 0.11 + phase - j * 0.3) * (j / h) * 1.5))
        xa = x + dx
        last = j >= h - 2
        for i in range(w):
            if last and (i == w // 2 or (j == h - 1 and i in (w // 2 - 1, w // 2 + 1))):
                continue
            cc = body if i < w - 1 else dark
            if i == 0:
                cc = mix(body, 'white', .3)
            c.px(xa + i, y + 1 + j, cc)
        if j == h // 2 - 1:
            small_star(c, xa + w // 2, y + 1 + j, emblem, shade(emblem, .7))


def raincloud(c, f, cam):
    if f < 140 or f > 216:
        return
    wx = (GARDEN_X0 + GARDEN_X1) // 2
    y = 92
    if f < 152:
        t = ease_out(seg(f, 140, 152))
        wx -= int((1 - t) * 40)
    if f >= 198:
        t = ease_in(seg(f, 198, 216))
        wx += int(t * 90)
        y -= int(t * 26)
    wx += int(round(math.sin(f * 0.08) * 1.2))
    sx = wx - cam
    if sx < -30 or sx > W + 30:
        return
    raining = 154 <= f < 202
    if raining:
        for i in range(26):
            dx = int(hash01(i, 1001) * 26) - 13
            ph = int(hash01(i, 1002) * 60)
            yy = y + 7 + (ph + f * 4) % 50
            if yy + 3 < GY - 3:
                c.vline(sx + dx, yy, yy + 3, 'sky')
                c.px(sx + dx, yy, 'baby_blue')
        for i in range(5):
            if (f + i * 3) % 6 < 3:
                c.px(sx - 10 + int(hash01(i, f // 6, 1003) * 20), GY - 4 - int(hash01(i, 1004) * 20), 'white')
    RC = R(mix('grey', 'indigo', .35), 'grey', mix('stone', 'grey', .35), 'stone', mix('stone', 'white', .5))
    sq = 1 if (raining and (f // 9) % 2) else 0
    circles = [(sx - 11, y + 2, 5), (sx - 4, y - 2, 7), (sx + 5, y, 6.5), (sx + 12, y + 3, 4.5)]
    blob(c, circles, RC, seed=5, tex=0.05)
    c.hline(sx - 14, sx + 15, y + 7 - sq, tup(RC[1]))
    c.hline(sx - 13, sx + 14, y + 8 - sq, tup(RC[0]))


def laundry(c, f, cam):
    x0, x1 = LAUNDRY_X0 + 1, LAUNDRY_X1
    if x1 - cam < -5 or x0 - cam > W + 5:
        return
    pts = []
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        pts.append((x - cam, 123 + int(round(math.sin(t * math.pi) * 3))))
    for (a, b) in zip(pts, pts[1:]):
        c.line(a[0], a[1], b[0], b[1], 'paper')
    items = [(2, 5, 7, 'sky', 'shirt'), (9, 4, 9, 'hot_pink', 'dress'), (15, 6, 6, 'gold', 'sheet'),
             (22, 3, 5, 'mint', 'sock')]
    for i, (ox, w, h, cc, kind) in enumerate(items):
        xa = x0 + ox - cam
        ya = pts[min(len(pts) - 1, ox)][1] + 1
        for j in range(h):
            dx = int(round(math.sin(f * 0.25 + i * 1.7 - j * 0.5) * (j / h) * 2.0 + (j / h) * 1.2))
            ww = w + (1 if kind == 'dress' and j > h // 2 else 0)
            c.hline(xa + dx, xa + dx + ww - 1, ya + j, cc)
            c.px(xa + dx + ww - 1, ya + j, shade(cc, .78))
            if j == 0:
                c.px(xa + dx, ya - 1, 'rust')
                c.px(xa + dx + ww - 1, ya - 1, 'rust')
        if kind == 'shirt':
            c.px(xa - 1, ya + 1, cc)
            c.px(xa + w, ya + 1, cc)


def lanterns(c, f, cam):
    strings = [(HALL_X1 - 4, 113, CLOCK_X - 7, 110), (STALLS[0][0] + 1, 116, STALLS[2][0] + 20, 116)]
    for si, (ax, ay, bx2, by2) in enumerate(strings):
        if bx2 - cam < 0 or ax - cam > W:
            continue
        prev = None
        n = int(bx2 - ax)
        for k in range(0, n + 1):
            t = k / n
            x = ax + (bx2 - ax) * t - cam
            y = ay + (by2 - ay) * t + math.sin(t * math.pi) * 5
            p = (int(x), int(y))
            if prev:
                c.line(prev[0], prev[1], p[0], p[1], 'night')
            prev = p
        for k in range(1, 6):
            t = k / 6
            x = int(ax + (bx2 - ax) * t - cam)
            y = int(ay + (by2 - ay) * t + math.sin(t * math.pi) * 5)
            sw = int(round(math.sin(f * 0.15 + k + si) * 0.8))
            cc = ('coral', 'gold', 'bubblegum', 'turquoise', 'lavender')[(k + si) % 5]
            c.vline(x, y + 1, y + 1, 'night')
            c.rect(x - 1 + sw, y + 2, 3, 3, cc)
            c.px(x - 1 + sw, y + 2, mix(cc, 'white', .5))
            c.hline(x - 1 + sw, x + 1 + sw, y + 5, shade(cc, .6))


def clock_hands(c, f, cam):
    cx = CLOCK_X - cam
    if cx < -10 or cx > W + 10:
        return
    mins = 30 + 60 * (f + FLIGHT_SHIFT) / (N + FLIGHT_SHIFT - 1)
    am = 2 * math.pi * (mins / 60)
    ah = 2 * math.pi * ((7 + mins / 60) / 12)
    c.line(cx, 80, cx + round(math.sin(am) * 4), 80 - round(math.cos(am) * 4), 'ink')
    c.line(cx, 80, cx + round(math.sin(ah) * 2.4), 80 - round(math.cos(ah) * 2.4), 'brick')
    c.px(cx, 80, 'gold')
    if 198 <= f < 222:          # the bell strikes eight (sfx 'bell' at 198)
        sw = int(round(math.sin((f - 198) * 0.55) * 1.6 * (1 - (f - 198) / 24)))
        c.rect(cx - 2, 54, 5, 5, 'night')
        c.circle(cx + sw, 56, 2, 'gold')
        c.hline(cx - 2 + sw, cx + 2 + sw, 58, 'amber')
        c.px(cx - 1 + sw, 55, 'butter')
        if f < 210 and (f - 198) % 6 < 3:
            for (dx, dy) in ((-5, -2), (5, -2), (-6, 1), (6, 1)):
                c.px(cx + dx, 56 + dy, 'white')


def castle_fx(c, f, cam):
    cx = CASTLE_X - cam
    if cx < -70 or cx > W + 70:
        return
    # crystal light sweep over the trunk every bar
    ph = (f - 72) % 72
    if ph < 30:
        s = 150 - ph * 4
        y0, y1 = max(0, s - 30), min(GY, s + 12)
        x0, x1 = max(0, cx - 24), min(W, cx + 25)
        if y1 > y0 and x1 > x0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            band = (np.abs((yy - s) + (xx - cx) * 0.6) < 2)
            reg = c.a[y0:y1, x0:x1]
            cm = _castle_mask()[y0:y1, (x0 + cam - CASTLE_X + 60):(x1 + cam - CASTLE_X + 60)]
            mm = band & cm
            regf = reg.astype(np.float32)
            regf[mm] = regf[mm] * 0.5 + 255 * 0.5
            c.a[y0:y1, x0:x1] = regf.astype(np.uint8)
    # star glint on every beat
    b = f % 18
    if b < 10:
        fx.sparkle(c, cx + 5, 9, b, 'white', period=10, size=3)
    fx.sparkle(c, cx - 6, 18, f + 7, 'white', period=36, size=1)
    fx.sparkles(c, cx, 15, f, radius=14, n=4, colors=('white', 'crystal_pk', 'crystal_lt'), seed=3, period=40)


@lru_cache(maxsize=1)
def _castle_mask():
    arr, _ = _main()
    reg = arr[:, CASTLE_X - 60:CASTLE_X + 61, :3]
    cols = np.array([list(v) for v in CRV[1:]] + [list(v) for v in CRT[1:]], np.int16)
    m = np.zeros(reg.shape[:2], bool)
    for cc in cols:
        m |= np.all(reg == cc, axis=2)
    return m


# ------------------------------------------------------------------ train
def train_front(f):
    t = seg(f, 262, 334)
    return TRAIN_STOP + 122 * (1 - t) ** 2


def _wheel(c, x, y, r, ang, rim='crimson'):
    c.circle(x, y, r, 'ink')
    c.circle(x, y, r - 1, rim)
    if r >= 4:
        c.circle(x, y, r - 2, 'berry')
    for k in range(3 if r < 4 else 4):
        a = ang + k * math.pi / (3 if r < 4 else 4) * (2 if r < 4 else 1)
        c.line(x, y, x + round(math.cos(a) * (r - 1)), y + round(math.sin(a) * (r - 1)), 'raspberry')
    c.px(x, y, 'gold')


def draw_train(c, f, cam):
    fw = train_front(f)
    X = int(round(fw)) - cam
    if X > W + 5:
        return None
    dist = TRAIN_STOP + 122 - fw
    ang = -dist / 5.0
    Y = 150
    # coaches (drawn first, right to left)
    for k, (body, low, roofc) in enumerate((('cotton', 'bubblegum', 'periwinkle'),
                                             ('bubblegum', 'hot_pink', 'royal_lt'))):
        x0 = X + 70 + k * 38
        if x0 > W + 2:
            continue
        c.rect(x0, Y - 32, 36, 26, 'berry')
        c.rect(x0 + 1, Y - 31, 34, 13, body)
        c.rect(x0 + 1, Y - 18, 34, 11, low)
        c.hline(x0 + 1, x0 + 34, Y - 18, 'gold')
        c.hline(x0 + 1, x0 + 34, Y - 31, mix(body, 'white', .5))
        c.vline(x0 + 1, Y - 30, Y - 8, mix(body, 'white', .4))
        for wx in range(x0 + 4, x0 + 33, 8):
            c.rect(wx, Y - 28, 5, 7, 'raspberry')
            c.rect(wx + 1, Y - 27, 3, 5, 'baby_blue')
            c.px(wx + 1, Y - 27, 'white')
            c.rect(wx + 1, Y - 23, 3, 1, mix('baby_blue', 'periwinkle', .5))
        c.rect(x0 - 1, Y - 35, 38, 3, roofc)
        c.hline(x0, x0 + 35, Y - 36, mix(roofc, 'white', .4))
        c.hline(x0 - 1, x0 + 36, Y - 33, shade(roofc, .7))
        c.rect(x0 + 2, Y - 7, 32, 2, 'eggplant')
        for wx in (x0 + 6, x0 + 11, x0 + 25, x0 + 30):
            _wheel(c, wx, Y - 3, 3, ang * 1.6)
        c.rect(x0 - 3, Y - 12, 3, 2, 'steel_dk')
    # tender
    tx = X + 50
    c.rect(tx, Y - 26, 19, 20, 'berry')
    c.rect(tx + 1, Y - 25, 17, 18, 'hot_pink')
    c.hline(tx + 1, tx + 17, Y - 17, 'gold')
    c.vline(tx + 1, Y - 25, Y - 8, 'bubblegum')
    c.rect(tx + 15, Y - 25, 3, 18, 'rose')
    for k in range(5):
        c.circle(tx + 3 + k * 3, Y - 27, 2, 'night' if k % 2 else 'charcoal')
    c.rect(tx + 1, Y - 7, 17, 2, 'eggplant')
    for wx in (tx + 5, tx + 13):
        _wheel(c, wx, Y - 3, 3, ang * 1.6)
    c.rect(tx - 2, Y - 12, 2, 2, 'steel_dk')
    # locomotive
    c.rect(X + 6, Y - 10, 43, 4, 'eggplant')
    # cab
    c.rect(X + 34, Y - 38, 15, 30, 'berry')
    c.rect(X + 35, Y - 37, 13, 28, 'hot_pink')
    c.vline(X + 35, Y - 37, Y - 10, 'bubblegum')
    c.rect(X + 45, Y - 37, 3, 28, 'rose')
    c.rect(X + 37, Y - 33, 7, 8, 'berry')
    c.rect(X + 38, Y - 32, 5, 6, mix('eggplant', 'baby_blue', .25))
    c.px(X + 38, Y - 32, 'baby_blue')
    c.px(X + 39, Y - 31, 'white')
    c.hline(X + 35, X + 47, Y - 21, 'gold')
    c.rect(X + 32, Y - 41, 19, 3, 'violet')
    c.hline(X + 33, X + 50, Y - 42, 'lavender')
    c.hline(X + 32, X + 50, Y - 39, 'purple')
    # boiler
    for y in range(Y - 28, Y - 11):
        t = (y - (Y - 28)) / 16
        cc = ('cotton' if t < 0.1 else 'bubblegum' if t < 0.25 else 'hot_pink' if t < 0.62 else
              'rose' if t < 0.85 else 'raspberry')
        c.hline(X + 12, X + 34, y, cc)
    c.hline(X + 12, X + 34, Y - 29, 'berry')
    for gx in (X + 15, X + 22, X + 29):
        c.vline(gx, Y - 28, Y - 12, 'gold')
        c.vline(gx + 1, Y - 27, Y - 12, 'amber')
    # smokebox + lamp
    c.rect(X + 6, Y - 29, 7, 19, 'plum')
    c.rect(X + 7, Y - 28, 5, 17, 'berry')
    c.vline(X + 7, Y - 28, Y - 12, 'raspberry')
    c.circle(X + 9, Y - 20, 2, 'gold')
    c.px(X + 9, Y - 20, 'butter')
    c.rect(X + 6, Y - 33, 5, 4, 'gold')
    c.rect(X + 7, Y - 32, 3, 2, 'butter')
    # funnel
    c.rect(X + 15, Y - 40, 5, 11, 'raspberry')
    c.vline(X + 15, Y - 40, Y - 30, 'rose')
    c.rect(X + 13, Y - 45, 9, 5, 'magenta')
    c.hline(X + 13, X + 21, Y - 45, 'gold')
    c.vline(X + 13, Y - 44, Y - 41, 'rose')
    c.px(X + 20, Y - 44, 'plum')
    # dome + whistle
    c.ellipse(X + 26, Y - 30, 3, 2, 'gold')
    c.px(X + 25, Y - 31, 'butter')
    c.rect(X + 31, Y - 34, 2, 5, 'brass')
    c.px(X + 31, Y - 35, 'gold')
    # cowcatcher + buffer beam
    c.poly([(X - 1, Y), (X + 7, Y), (X + 7, Y - 8), (X + 4, Y - 8)], 'amber')
    for k in range(0, 8, 2):
        c.line(X + k, Y - 1, X + 5, Y - 7, 'gold')
    c.rect(X + 4, Y - 12, 4, 4, 'crimson')
    # cylinder + wheels + rods
    c.rect(X + 8, Y - 10, 8, 5, 'raspberry')
    c.hline(X + 8, X + 15, Y - 10, 'rose')
    _wheel(c, X + 10, Y - 3, 3, ang * 1.6)
    pins = []
    for wx in (X + 19, X + 29, X + 39):
        _wheel(c, wx, Y - 5, 5, ang)
        pins.append((wx + round(math.cos(ang) * 3), Y - 5 + round(math.sin(ang) * 3)))
    c.line(pins[0][0], pins[0][1], pins[2][0], pins[2][1], 'steel_hi')
    c.line(X + 14, Y - 7, pins[0][0], pins[0][1], 'steel')
    for p in pins:
        c.px(p[0], p[1], 'white')
    return X


def train_fx(c, f, cam, X):
    if X is None:
        return
    # chuffs from the funnel (8ths while rolling)
    for k in range(6):
        e = (f // 9 - k) * 9
        if e < 262 or e > 334:
            continue
        a = f - e
        if a > 46:
            continue
        ex = int(round(train_front(e))) + 17
        px_ = ex - cam + int(a * 0.35)
        py = 150 - 47 - int(a * 0.9)
        r = 2 + a * 0.14
        puff(c, px_, py, r, mix('white', 'cloud', min(1.0, a / 36)), 1.05 - a / 46)
        if a < 14:
            puff(c, px_ - 1, py - 1, r * 0.6, 'white', 1.0)
    if f > 334:
        smoke(c, X + 17, 150 - 47, f, seed=55, n=4, rise=26, drift=10, big=1.2)
        a = f - 334
        if a < 30:
            for s in (-1, 1):
                for k in range(4):
                    rr = 2 + a // 6 + k
                    px_ = X + 12 + s * (k * 5 + a // 2)
                    puff(c, px_, 147 - k, rr * 0.7 + 1, mix('white', 'cloud', k / 4), 1 - a / 30)
    # whistle steam (288 and 306, as in the music)
    for (t0, dur) in ((288, 6), (297, 16)):
        a = f - t0
        if 0 <= a < dur + 14:
            wx, wy = X + 31, 150 - 36
            hgt = min(a * 4 + 4, 16)
            al = 1.0 if a < dur else max(0.0, 1 - (a - dur) / 14)
            if a < dur:
                for j in range(hgt):
                    c.px(wx + j // 6, wy - j, 'white')
                    c.px(wx + 1 + j // 6, wy - j, 'white' if j < hgt - 3 else 'cloud')
            rr = min(6, 2 + a // 2)
            bx = wx + 3 + a // 2
            puff(c, bx, wy - hgt - 1, rr, 'white', al * 1.1)
            puff(c, bx + 4, wy - hgt + 2, rr - 1, 'cloud', al)
            puff(c, bx - 3, wy - hgt + 1, rr - 2, 'white', al)


# ------------------------------------------------------------------ shots
def _sx(wx, cam, p=1.0):
    return int(wx - int(round(cam * p)))


def render_diorama(f):
    cam = cam_x(f)
    c = _sky().copy()
    # clouds
    blit_wrap(c, _clouds(), int(round(cam * 0.05 + f * 0.12)), 0)
    # Canterlot + waterfalls
    oc = int(round(cam * P_CAN))
    c.blit(_canterlot(), -oc, 0)
    for i, (fdx, y0, y1) in enumerate(FALLS):
        x = MT_X + fdx - oc
        for y in range(y0 - 2, y1):
            k = (y - f + i * 5) % 7
            fade = (y - (y1 - 9)) / 9.0
            if fade > 0 and bayer(1, 1, x, y)[0, 0] < fade:
                continue
            wob = 1 if (y + i) % 9 < 2 else 0
            c.px(x + wob, y, 'white' if k < 4 else mix('baby_blue', 'white', .4))
            c.px(x + 1 + wob, y, mix('baby_blue', HAZE, .3) if k != 5 else 'white')
        for j in range(3):
            fx.sparkle(c, x + j * 2 - 2, y1 - 6 + j, f + j * 9 + i * 5, 'white', period=27, size=1)
    for i in range(3):
        fx.sparkle(c, MT_X - 30 + i * 9 - oc, 50 - i * 5, f + i * 11, 'white', period=44, size=1)
    # far hills + mist
    c.blit(_far(), -int(round(cam * P_FAR)), 0)
    blit_wrap(c, _mist(), int(round(cam * 0.3 + f * 0.2)), 100)
    # mid hills, windmill, lake glints
    om = int(round(cam * P_MID))
    marr, millty = _mid()
    c.blit(marr, -om, 0)
    mx, my = MILL_X - om, millty
    if -12 < mx < W + 12:
        a0 = f * 0.045
        for k in range(4):
            a = a0 + k * math.pi / 2
            ex, ey = mx + math.cos(a) * 9, my + math.sin(a) * 9
            c.line(mx, my, ex, ey, mix('tan', HAZE, .2))
            c.line(mx + math.cos(a + 0.3) * 3, my + math.sin(a + 0.3) * 3,
                   ex + math.cos(a + 0.25) * 1.5, ey + math.sin(a + 0.25) * 1.5, mix('paper', HAZE, .2))
        c.px(mx, my, mix('rust', HAZE, .2))
    for i in range(14):
        lx = LAKE_X0 + int(hash01(i, 1201) * (LAKE_X1 - LAKE_X0)) - om
        ly = 128 + int(hash01(i, 1202) * 16)
        if 0 <= lx < W and (f // 3 + i * 5) % 17 < 3:
            c.hline(lx, lx + 2, ly, 'white')
            fx.sparkle(c, lx + 1, ly, f + i * 7, 'white', period=24, size=1)
    blit_wrap(c, _mist(), int(round(cam * 0.5 + f * 0.3)) + 220, 122)
    # back village + smoke
    ob = int(round(cam * P_BACK))
    barr, bmeta = _back()
    c.blit(barr, -ob, 0)
    for (_, x, y, sd) in bmeta:
        sx = x - ob
        if -10 < sx < W + 10:
            smoke(c, sx, y, f, seed=sd, n=4, rise=16, drift=6, big=0.8)
    # main layer
    marr2, meta = _main()
    c.blit(marr2, -cam, 0)
    for m in meta:
        if isinstance(m[0], str):
            kind = m[0]
            sx = m[1] - cam
            if sx < -40 or sx > W + 40:
                continue
            if kind == 'smoke':
                smoke(c, sx, m[2], f, seed=m[3])
            elif kind == 'steam':
                fx.steam(c, sx, m[2], f, color='white', n=5, height=16, spread=3, seed=2)
                fx.steam(c, sx + 1, m[2], f + 9, color='cloud', n=3, height=12, spread=2, seed=3)
            elif kind == 'flag':
                flag(c, sx, m[2], f, w=8 if m[3] == 'hall' else 6, h=5 if m[3] == 'hall' else 4,
                     cols=('hot_pink', 'rose') if m[3] == 'hall' else ('magic', 'raspberry'),
                     emblem='gold', phase=hash01(m[1]))
            elif kind == 'banner':
                banner(c, sx, m[2], f, w=5, h=13, body='magic', dark='raspberry', emblem='gold',
                       phase=m[1] * 0.1)
            elif kind == 'gem':
                fx.sparkle(c, sx, m[2], f, 'white', period=30, size=2)
            elif kind == 'cherry':
                fx.sparkle(c, sx - 1, m[2] - 1, f + 13, 'white', period=48, size=1)
            elif kind == 'pennant':
                flag(c, sx, m[2], f + 5, w=6, h=3, cols=('bubblegum', 'rose'), emblem=None)
            elif kind == 'signal':
                if f >= 300:
                    c.px(sx, m[2], 'red')
            elif kind == 'star':
                pass
        else:
            wx, wy, wr = m
            sx = wx - cam
            if 0 <= sx < W and (f + int(hash01(wx, wy) * 240)) % 240 < 4:
                k = (f + int(hash01(wx, wy) * 240)) % 240
                c.px(sx - wr + 1 + k, wy - 1 + (k // 2), 'white')
                c.px(sx - wr + 2 + k, wy - 2 + (k // 2), 'white')
    castle_fx(c, f, cam)
    # castle banners
    for (dx, dy) in ((-34, 92), (26, 88)):
        sx = CASTLE_X + dx - cam
        if -10 < sx < W + 10:
            banner(c, sx, dy, f, w=5, h=12, body='royal_lt', dark='royal', emblem='magic', phase=dx * 0.2)
    lanterns(c, f, cam)
    clock_hands(c, f, cam)
    laundry(c, f, cam)
    raincloud(c, f, cam)
    X = draw_train(c, f, cam) if f >= 216 else None
    train_fx(c, f, cam, X)
    # stream glitter
    for i in range(8):
        gx = STREAM_X0 + 2 + int(hash01(i, f // 4, 1301) * (STREAM_X1 - STREAM_X0 - 4)) - cam
        gy = 141 + int(hash01(i, f // 4, 1302) * 24)
        if 0 <= gx < W and not (148 <= gy <= 152):
            c.px(gx, gy, 'white')
    # dew sparkles on the verge
    for i in range(30):
        dx = int(hash01(i, 1401) * MAIN_W) - cam
        if 0 <= dx < W:
            fx.sparkle(c, dx, 159 + int(hash01(i, 1402) * 6), f, 'white', period=40 + int(hash01(i, 1403) * 30),
                       phase=hash01(i, 1404), size=1)
    # balloon
    if f < 216:
        bx = 128 + int((f - 72) * 0.15)
        by = 72 - int((f - 72) * 0.04)
    else:
        bx = 250 - int((f - 216) * 1.33)
        by = 66 + int((f - 216) * 0.1)
    draw_balloon(c, bx, by, f, 'm', flame=(f // 36) % 4 == 0)
    # foreground
    ofg = int(round(cam * P_FG))
    farr, ftop = _fg()
    c.blit(farr, -ofg, 0)
    for i in range(40):
        gx = int(hash01(i, 1501) * FG_W)
        sx = gx - ofg
        if 0 <= sx < W:
            fx.sparkle(c, sx, int(ftop[gx]) + 1 + int(hash01(i, 1502) * 8), f, 'white',
                       period=34 + int(hash01(i, 1503) * 30), phase=hash01(i, 1504), size=1)
    for i in range(6):
        gx = 1302 + int(hash01(i, f // 3, 1601) * 80) - ofg
        if 0 <= gx < W:
            c.px(gx, 170 + int(hash01(i, f // 3, 1602) * 9), 'white')
    # warm morning light from the east
    fx.light_rect(c, 0, 0, 150, H, 'gold', strength=0.07, fade='right')
    return c


from engine.loc_worldmap import dest_screen
_IRIS = tuple(int(v) for v in dest_screen(ID))


def render(f):
    if f < 72:
        return travel_frame(f, ID)
    c = render_diorama(f)
    cam = cam_x(f)
    # callouts
    callout(c, CASTLE_X - 46 - cam, 86, 'CASTLE OF FRIENDSHIP', f, 72, 138, side='left', length=12)
    callout(c, (SCHOOL_X0 + SCHOOL_X1) // 2 - cam, 50, 'SCHOOL OF FRIENDSHIP', f, 108, 180, side='up', length=10)
    if f < 216:
        callout(c, HALL_X0 + 10 - cam, 136, 'TOWN HALL', f, 144, 209, side='down', length=12)
    else:
        sxc = (SUGAR_X0 + SUGAR_X1) // 2 - cam
        callout(c, sxc + 1, 50, 'SUGARCUBE CORNER', f, 216, 282, side='up', length=10)
        callout(c, BOUT_X + 14 - cam, 138, 'CAROUSEL BOUTIQUE', f, 252, 324, side='down', length=10)
        callout(c, (STATION_X0 + STATION_X1) // 2 - 5 - cam, 104, 'TRAIN STATION', f, 288, 346, side='up', length=22)
    atlas_card(c, f, 100, 200, 'COUNTRY TOWN', ['THATCH ROOFS - MARKET', 'BELOW CANTERLOT'])
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), *_IRIS)
    return c


SFX = [
    (80, 'wind', 0.35),
    (90, 'crystal', 0.45),
    (152, 'rain', 0.25),
    (198, 'bell', 0.45),
    (199, 'whoosh', 0.25),
    (222, 'chime', 0.3),
    (234, 'steam', 0.25),
    (262, 'chug', 0.45),
    (288, 'whistle', 0.85),
    (298, 'chug', 0.5),
    (334, 'steam', 0.6),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
