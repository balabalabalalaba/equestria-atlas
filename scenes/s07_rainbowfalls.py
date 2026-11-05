"""RAINBOW FALLS + OUR TOWN (4 bars, 288 frames, HUD clock 11:30 -> 12:00, bright midday).

Beat sheet (local frames; bar = 72, beat = 18). Key light: high noon sun, slightly upper-left
(lit tops and left faces, short shadows under eaves and ledges). Crisp blue sky, high cumulus.
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71.
  Bar 2  72-143   REVEAL (72): iris opens on the clifftop village (dest_screen 204,100): warm plank
                  houses with shingle roofs, flower boxes, bunting, the Games judges' tower and the
                  grandstand on a flat green mesa top; behind, the snow peaks of the Yaket Range.
                  Five rainbow waterfalls pour off the lip. The hoop course (floating cloud rings with
                  pennants) arcs over the village and dives down the cliff to the gorge.
                  The balloon threads the top rings and follows the course down the falls; the camera
                  eases DOWN along the falls (pan + tilt) to the stone rail trestle (arrives ~138).
                  Callout RAINBOW FALLS 86-144 (on the main fall). Atlas card 100-192.
                  Train enters from the right (~118) heading left, chugging (puffs on 8ths).
  Bar 3 144-215   `trestle` HIT (144): the Friendship Express is mid-span: whistle jet + big steam
                  burst; the balloon hovers above the deck in the steam. Short drift 138-160, then the
                  camera TILTS UP from the trestle (160-210), the balloon rising with its burner on
                  through the climbing rings, past the falls, up to the village and the sky.
                  Callouts RAIL TRESTLE 124-174, HOOP COURSE 156-209. Card out by 200.
  Bar 4 216-287   CUT on the bar line: OUR TOWN seen from a hill: a dusty valley among sandstone
                  hoodoos and mesas, identical cream houses with red roofs in two straight rows (the
                  "=" layout), the "=" banner on its pole, the big leafy tree at the end of the street,
                  the round vault door in the cliff; Starlight's Cave far off on grey crags. Slow pan
                  right; dust devils, cloud shadows, chimney smoke, banner flutter; small balloon.
                  Callout OUR TOWN 222-279. Toast from 230 (top-centre kept clear). Wipe 280-287.
Ambient: falls flow + lip foam + prismatic mist, river ripples, drifting cumulus, flapping flags,
         bunting and ring pennants, ring bob, chimney smoke, train + steam, window glints,
         sparkles, balloon; Our Town: dust devils, cloud shadows, banner, tree sway, smoke.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine import fx
from engine.fx import steam, sparkle
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'rainbowfalls'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)


def _C(c):
    return np.array(col(c), np.uint8)


# ------------------------------------------------------------------ layer helpers
def _layer(w, h):
    return Canvas(w, h, KEY)


def _finish(cv):
    a = cv.a
    return a, np.any(a != _KEYA, axis=2)


def _paste(c, layer, ox, oy):
    """Draw a (rgb, mask) layer so that layer pixel (ox, oy) lands on screen (0, 0)."""
    rgb, m = layer
    h, w = m.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + W), min(h, oy + H)
    if x1 <= x0 or y1 <= y0:
        return
    sm = m[y0:y1, x0:x1]
    dst = c.a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    dst[sm] = rgb[y0:y1, x0:x1][sm]


def _ramp(c, deep='plum_brown', light='cream'):
    """5-step ramp: deep, shade, base, light, highlight."""
    return [mix(c, deep, 0.55), mix(c, deep, 0.28), col(c), mix(c, light, 0.32), mix(c, 'white', 0.55)]


# ------------------------------------------------------------------ sky + cumulus
NOON = ['#3f5fb8', 'blue', 'sky', '#6db6f0', 'light_sky', '#c4ecff']


@lru_cache(maxsize=None)
def _sky_tall(h=420):
    c = Canvas(W, h)
    c.gradient(0, 0, W, h, NOON, soft=0.7)
    return c.a


def _cumulus(cv, x, y, w, seed, lit='white', mid='cloud', sh='cloud_dk', base_sh=None):
    """Crisp cumulus with a flat base. (x, y) = left end of the flat base line."""
    base_sh = base_sh or mix('cloud_dk', 'periwinkle', 0.35)
    r = rng(seed)
    n = max(3, w // 10)
    blobs = []
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi)
        rad = max(3, int(w * (0.10 + 0.13 * bump) * (0.8 + 0.4 * r.random())))
        cx = x + int(u * w)
        cy = y - int(rad * (0.55 + 0.6 * bump))
        blobs.append((cx, cy, rad))
    # top tower puffs
    for k in range(max(1, w // 40)):
        cx = x + int(w * (0.3 + 0.4 * r.random()))
        rad = int(w * (0.12 + 0.05 * r.random()))
        blobs.append((cx, y - int(w * 0.32) - rad // 2, rad))
    hgt = int(w * 0.75) + 6
    x0, y0 = x - 4, y - hgt
    ww, hh = w + 8, hgt + 1
    yy, xx = np.mgrid[y0:y0 + hh, x0:x0 + ww]
    inside = np.zeros((hh, ww), bool)
    shade_m = np.zeros((hh, ww), bool)
    for (cx, cy, rad) in blobs:
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        inside |= d <= rad * rad
        # shadow on the lower right of each puff (sun upper-left)
        shade_m |= ((xx - cx + rad * 0.45) ** 2 + (yy - cy + rad * 0.5) ** 2 > (rad * 1.02) ** 2) & (d <= rad * rad)
    inside &= yy <= y
    thr = bayer(hh, ww, x0, y0)
    out = np.zeros((hh, ww, 3), np.uint8)
    out[:] = _C(lit)
    out[shade_m] = _C(mid)
    lowz = (yy > y - 4)
    out[lowz & (thr < 0.5 + (yy - (y - 4)) * 0.18)] = _C(sh)
    out[yy >= y - 1] = _C(base_sh)
    # sub-cv write
    sub = cv.a[max(0, y0):y0 + hh, max(0, x0):x0 + ww]
    m = inside[max(0, -y0):, max(0, -x0):][:sub.shape[0], :sub.shape[1]]
    o = out[max(0, -y0):, max(0, -x0):][:sub.shape[0], :sub.shape[1]]
    sub[m] = o[m]


# ------------------------------------------------------------------ RAINBOW FALLS world
# main layer (k=1) coordinates; camera = top-left of the view in these coordinates.
MESA_TOP = 150
MESA_BOT = 280
RIVER_Y = 284
REF = (110.0, 128.0)
RB_NAMES = ('red', 'orange', 'gold', 'leaf', 'sky', 'violet')
RB_RAMP = np.array([[mix(n, 'plum', 0.30), col(n), mix(n, 'white', 0.38), mix(n, 'white', 0.75)]
                    for n in RB_NAMES], np.uint8)                       # 6 stripes x 4 levels
RB_PASTEL = [mix(n, 'white', 0.6) for n in RB_NAMES]
# falls: (x0, width, flow speed px/frame)
FALLS = ((132, 10, 2), (214, 12, 3), (286, 22, 3), (364, 14, 3), (440, 12, 2))
MAIN_FALL = FALLS[2]

CAM_KEYS = [(72, (40.0, 38.0)), (96, (58.0, 40.0)), (138, (116.0, 124.0)), (160, (132.0, 128.0)),
            (210, (212.0, 34.0)), (230, (218.0, 32.0))]


def cam_rf(f):
    return keyframes(f, CAM_KEYS, ease_in_out)


def off(cam, k):
    return (int(round(REF[0] + (cam[0] - REF[0]) * k)), int(round(REF[1] + (cam[1] - REF[1]) * k)))


def mesa_left(y):
    s = (y - MESA_TOP) // 13
    return int(110 - 0.26 * (y - MESA_TOP) - 3 * hash01(7, s) - (2 if y > 268 else 0))


def mesa_right(y):
    s = (y - MESA_TOP) // 11
    return int(476 + 0.32 * (y - MESA_TOP) + 4 * hash01(9, s))


ROCK = [mix(mix('sand', 'clay', 0.55), 'tan', 0.30),       # ochre
        mix(mix('salmon', 'rose', 0.35), 'tan', 0.35),      # rose
        mix('peach', 'tan', 0.30),                          # peach
        mix('copper', 'mauve', 0.40),                       # brown
        mix('blush', 'tan', 0.35),                          # cream
        mix('mauve', 'rose', 0.30)]                         # mauve
BAND_SEQ = [(4, 4), (0, 10), (1, 6), (2, 12), (3, 5), (0, 9), (4, 3), (1, 10), (5, 6), (2, 14),
            (3, 8), (0, 10), (1, 8), (5, 12), (2, 9), (3, 10), (0, 8), (5, 14), (3, 12)]
ROCK_RAMP = np.array([_ramp(c) for c in ROCK], np.uint8)   # 6 x 5 x 3


def _rock_texture(x0, y0, w, h, seed, band_seq=BAND_SEQ, ramp=ROCK_RAMP, colw=(40, 70), top=0):
    """Canyon-wall texture (numpy): strong horizontal strata (lit tops, shaded undersides), soft
    broad buttresses, a few long fissures. Returns (h, w, 3) uint8 and the band-top map."""
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    wob = (3.0 * np.sin(xx * 0.028 + seed) + 1.3 * np.sin(xx * 0.09 + seed * 2.1)
           + 0.6 * np.sin(xx * 0.27 + seed * 0.7))
    yd = (yy - top) + np.round(wob).astype(int)
    hs = np.array([b[1] for b in band_seq])
    edges = np.cumsum(hs)
    total = edges[-1]
    ydc = np.mod(yd, total)
    bi = np.clip(np.searchsorted(edges, ydc, side='right'), 0, len(band_seq) - 1)
    band_col = np.array([b[0] for b in band_seq])[bi]
    starts = np.concatenate([[0], edges[:-1]])
    pos_in = ydc - starts[bi]
    bh = hs[bi]
    t = (pos_in + 0.5) / bh
    hard = (bi % 2 == 0)
    lvl = np.full((h, w), 2.0)
    # band relief: lit tops, shaded undersides (protruding hard bands stronger)
    lvl += np.where(t < 0.22, 0.8, 0.0) + np.where(hard & (pos_in == 0), 0.9, 0.0)
    lvl -= np.where(t > 0.78, 0.7, 0.0) + np.where(hard & (pos_in == bh - 1), 0.8, 0.0)
    lvl -= np.where(~hard, 0.25, 0.0)
    # broad buttresses (sun upper-left): lit left flanks, shaded right flanks, dithered soft edges
    r = rng(seed)
    P = (colw[0] + colw[1]) / 2.0
    ph = np.cumsum(r.uniform(0.6, 1.4, w + 8))[:w] * 0 + 0
    s = np.sin(2 * np.pi * (xx + 6 * np.sin(yy * 0.02 + seed)) / P + 1.7 * np.sin(xx * 0.011 + seed))
    thr = bayer(h, w, x0, y0)
    lvl += np.where(s + (thr - 0.5) * 0.5 > 0.55, 0.9, 0.0)
    lvl -= np.where(s + (thr - 0.5) * 0.5 < -0.45, 0.9, 0.0)
    # lower part darker (reflected light fades), dithered
    depth = np.clip((yy - (top + h * 0.55)) / (h * 0.45 + 1), 0, 1)
    lvl -= np.where(thr < depth * 0.7, 1.0, 0.0)
    # speckle
    n = (np.sin(xx * 12.9898 + yy * 78.233 + seed) * 43758.5453) % 1.0
    lvl -= np.where(n < 0.025, 1.0, 0.0)
    lvl += np.where(n > 0.985, 1.0, 0.0)
    lv = np.clip(np.round(lvl), 0, 4).astype(int)
    # long fissures
    for k in range(int(w / 34)):
        fx_ = int(r.integers(0, w))
        fy0 = int(r.integers(0, max(1, h - 20)))
        fl = int(r.integers(12, 60))
        xcur = fx_
        for yy_ in range(fy0, min(h, fy0 + fl)):
            if r.random() < 0.18:
                xcur += int(r.integers(-1, 2))
            if 0 <= xcur < w - 1:
                lv[yy_, xcur] = 0
                lv[yy_, xcur + 1] = min(4, lv[yy_, xcur + 1] + 1)
    out = ramp[band_col % len(ramp), lv]
    return out, (pos_in == 0) & hard


def _wood_house(cv, x, base, w, wall_h, roof_h, wall, roof, gable=True, seed=0, two=False,
                chimney=False, shutters='teal', boxes=True, round_win=True):
    """Warm plank house in side view. x = left wall edge, base = ground row (bottom of walls)."""
    r = rng(seed)
    top = base - wall_h
    wr = _ramp(wall, deep='brew', light='cream')
    # walls: planks
    cv.rect(x, top, w, wall_h, wr[2])
    for yy in range(top + 2, base, 3):
        cv.hline(x, x + w - 1, yy, wr[1])
    for k in range(int(w * wall_h / 40)):
        jx = x + 2 + int(r.integers(0, max(1, w - 4)))
        jy = top + 3 * int(r.integers(0, max(1, wall_h // 3)))
        cv.vline(jx, jy, min(base - 1, jy + 1), wr[1])
    cv.vline(x, top, base - 1, wr[3])
    cv.vline(x + 1, top, base - 1, wr[3])
    cv.rect(x + w - 2, top, 2, wall_h, wr[1])
    cv.vline(x + w - 1, top, base - 1, wr[0])
    # corner posts + foundation
    cv.rect(x, base - 2, w, 2, 'grey')
    cv.hline(x, x + w - 1, base - 2, 'stone')
    if two:
        cv.hline(x, x + w - 1, top + wall_h // 2, wr[0])
        cv.hline(x, x + w - 1, top + wall_h // 2 + 1, wr[3])
    # roof
    rr = _ramp(roof, deep='ink', light='cream')
    ov = 2
    if gable:
        apex = (x + w // 2, top - roof_h)
        pts = [(x - ov, top + 1), (apex[0] + 0.5, apex[1]), (x + w + ov, top + 1)]
        cv.poly(pts, rr[2])
        # shingle rows
        for yy in range(top - roof_h + 2, top + 1, 2):
            t = (yy - (top - roof_h)) / roof_h
            hw = (w / 2 + ov) * t
            xa, xb = int(apex[0] - hw) + 1, int(apex[0] + hw)
            for xx in range(xa, xb, 3):
                cv.px(xx + (yy // 2) % 2, yy, rr[1])
        # lit left slope edge, shaded right slope
        cv.line(x - ov, top + 1, apex[0], apex[1], rr[4])
        cv.line(x - ov + 1, top + 1, apex[0], apex[1] + 1, rr[3])
        cv.line(apex[0] + 1, apex[1] + 1, x + w + ov, top + 1, rr[0])
        cv.hline(x - ov, x + w + ov, top + 1, rr[0])
        # gingerbread trim under the eaves (scallops)
        for xx in range(x, x + w, 2):
            cv.px(xx, top + 2, 'cream')
        cv.hline(x, x + w - 1, top + 3, wr[0])
        # attic round window
        if round_win and roof_h >= 9:
            cxw, cyw = apex[0], top - roof_h // 2 + 1
            cv.circle(cxw, cyw, 2, 'cream')
            cv.px(cxw, cyw, mix('baby_blue', 'sky', 0.4))
            cv.px(cxw - 1, cyw, 'baby_blue')
            cv.px(cxw, cyw - 1, 'white')
    else:
        # side roof (trapezoid)
        pts = [(x - ov - 1, top + 1), (x + 3, top - roof_h), (x + w - 3, top - roof_h), (x + w + ov + 1, top + 1)]
        cv.poly(pts, rr[2])
        for yy in range(top - roof_h + 2, top + 1, 2):
            for xx in range(x - ov, x + w + ov, 3):
                cv.px(xx + (yy // 2) % 2, yy, rr[1])
        cv.hline(x + 3, x + w - 4, top - roof_h, rr[4])
        cv.line(x - ov - 1, top + 1, x + 3, top - roof_h, rr[3])
        cv.line(x + w - 3, top - roof_h, x + w + ov + 1, top + 1, rr[0])
        cv.hline(x - ov - 1, x + w + ov + 1, top + 1, rr[0])
        for xx in range(x, x + w, 2):
            cv.px(xx, top + 2, 'cream')
        cv.hline(x, x + w - 1, top + 3, wr[0])
    if chimney:
        chx = x + w - 7
        chy = top - roof_h + (roof_h // 3 if gable else 0) - 4
        cv.rect(chx, chy, 4, 7, 'stone')
        cv.rect(chx + 2, chy, 2, 7, 'grey')
        cv.hline(chx - 1, chx + 4, chy, 'mist')
    # door (rounded top) at a deterministic spot
    dx = x + (w // 2 - 2 if w < 24 else 3 + int(r.integers(0, 3)))
    dh = 9
    cv.rect(dx - 1, base - dh - 1, 6, dh + 1, wr[0])
    cv.rect(dx, base - dh, 4, dh, 'brew_mid')
    cv.px(dx - 1, base - dh - 1, wr[2])
    cv.px(dx + 4, base - dh - 1, wr[2])
    cv.vline(dx, base - dh, base - 1, 'latte')
    cv.px(dx + 3, base - 5, 'gold')
    cv.hline(dx - 1, dx + 4, base, 'stone')
    # windows
    wins = []
    rows = [base - 8] if not two else [base - 8, top + 4]
    for wy in rows:
        for wx in range(x + 3, x + w - 5, 9):
            if wy == base - 8 and dx - 6 <= wx <= dx + 5:
                continue
            wins.append((wx, wy))
    for (wx, wy) in wins:
        cv.rect(wx - 1, wy - 1, 6, 7, 'cream')
        cv.rect(wx, wy, 4, 5, mix('baby_blue', 'sky', 0.25))
        cv.rect(wx, wy + 3, 4, 2, mix('sky', 'royal', 0.3))
        cv.px(wx, wy, 'white')
        cv.px(wx + 1, wy, 'baby_blue')
        cv.hline(wx, wx + 3, wy + 2, 'cream')
        cv.px(wx - 1, wy - 1, wr[2])
        cv.px(wx + 4, wy - 1, wr[2])
        if shutters:
            cv.vline(wx - 2, wy, wy + 4, shutters)
            cv.vline(wx + 5, wy, wy + 4, shade(shutters, 0.75))
        if boxes:
            cv.rect(wx - 1, wy + 5, 6, 2, 'latte')
            cv.hline(wx - 1, wx + 4, wy + 6, 'brew_mid')
            for k in range(6):
                cc = ('hot_pink', 'leaf', 'gold', 'magenta', 'leaf', 'bubblegum')[(k + seed) % 6]
                cv.px(wx - 1 + k, wy + 4, cc)
            cv.px(wx, wy + 3, 'green')
            cv.px(wx + 3, wy + 3, 'green')
    return wins


def _curly_tree(cv, x, base, rad, seed, dark='forest', mid='green', lit='leaf', hi='pale_leaf', trunk='brew_mid'):
    r = rng(seed)
    th = rad + 2
    cv.rect(x - 1, base - th, 3, th, trunk)
    cv.vline(x - 1, base - th, base - 1, mix(trunk, 'crema', 0.4))
    cy = base - th - rad + 2
    puffs = [(x, cy, rad)]
    for k in range(4):
        a = -math.pi * (0.1 + 0.8 * k / 3)
        puffs.append((x + int(math.cos(a) * rad * 0.75), cy + int(math.sin(a) * rad * 0.55) + 1,
                      max(2, int(rad * (0.55 + 0.15 * r.random())))))
    for (px_, py, pr) in puffs:
        cv.circle(px_ + 1, py + 1, pr, dark)
    for (px_, py, pr) in puffs:
        cv.circle(px_, py, pr, mid)
    for (px_, py, pr) in puffs:
        cv.circle(px_ - max(1, pr // 3), py - max(1, pr // 3), max(1, pr // 2), lit)
        cv.px(px_ - pr // 2, py - pr // 2 - 1, hi)
    # curls
    for k in range(3):
        a = r.random() * math.pi
        cv.px(x + int(math.cos(a) * rad * 0.6), cy + int(math.sin(a) * rad * 0.5), dark)


@lru_cache(maxsize=None)
def _rf_peaks():
    """Yaket Range (k=0.3): lavender-grey snow peaks, hazy toward the bottom."""
    cv = _layer(600, 330)
    haze = '#c4ecff'
    rock = [mix(mix('periwinkle', 'grey', 0.5), haze, a) for a in (0.15, 0.32)]
    rock_sh = [mix(mix('indigo', 'grey', 0.4), haze, a) for a in (0.15, 0.32)]
    snow, snow_sh = 'white', mix('cloud_dk', 'periwinkle', 0.25)
    rows = [  # (peak x, peak y, half width, depth index)
        (70, 168, 60, 1), (150, 150, 52, 1), (250, 160, 64, 1), (360, 146, 60, 1), (460, 158, 58, 1),
        (540, 150, 50, 1),
        (30, 150, 44, 0), (110, 132, 50, 0), (205, 140, 46, 0), (300, 124, 56, 0), (410, 134, 48, 0),
        (500, 128, 52, 0), (580, 142, 44, 0)]
    rows.sort(key=lambda t: -t[3])
    for (px_, py, hw, d) in rows:
        rr = rng(px_ * 7 + py)
        base = 250
        # jagged ridge outline
        left = [(px_ - hw, base)]
        right = [(px_ + hw, base)]
        steps = 6
        for i in range(1, steps):
            t = i / steps
            left.append((px_ - hw * (1 - t) + rr.integers(-3, 4), base - (base - py) * t + rr.integers(-3, 3)))
            right.append((px_ + hw * (1 - t) + rr.integers(-3, 4), base - (base - py) * t + rr.integers(-3, 3)))
        pts = left + [(px_, py)] + right[::-1]
        a = 1 - d
        cv.poly(pts, rock[d])
        # shaded right face
        rpts = [(px_, py)] + right[1:][::-1][::-1] if False else [(px_, py)] + list(reversed(right)) + [(px_ + 4, base)]
        cv.poly([(px_, py)] + list(reversed(right)) + [(px_ + 6, base)], rock_sh[d])
        # snow cap
        sl = int((base - py) * 0.38)
        cap = [(px_, py)]
        for i in range(1, 7):
            t = i / 7
            cap.append((px_ + hw * 0.42 * t * (1 + 0.25 * math.sin(i * 2.3)), py + sl * t + (3 if i % 2 else -2)))
        cap.append((px_ - hw * 0.42, py + sl))
        for i in range(6, 0, -1):
            t = i / 7
            cap.append((px_ - hw * 0.42 * t * (1 + 0.25 * math.cos(i * 1.7)), py + sl * t + (2 if i % 2 else -3)))
        cv.poly(cap, snow)
        # snow shade on the right half
        rc = [(px_, py)] + [p for p in cap[1:8]] + [(px_ + 1, py + sl)]
        cv.poly(rc, snow_sh)
        # snow streaks in gullies
        for k in range(4):
            gx = px_ + int((rr.random() - 0.5) * hw * 0.9)
            gy = py + sl + int(rr.integers(0, 8))
            cv.line(gx, gy, gx + int(rr.integers(-2, 3)), gy + int(rr.integers(4, 12)), mix(snow, rock[d], 0.35))
    # haze at the foot
    cv.dither(0, 205, 600, 20, mix(haze, 'white', 0.3), 0.25)
    cv.dither(0, 225, 600, 30, haze, 0.6)
    cv.rect(0, 238, 600, 92, KEY)
    return _finish(cv)


@lru_cache(maxsize=None)
def _rf_clouds():
    cv = _layer(760, 330)
    specs = [(20, 150, 70, 3), (130, 124, 54, 4), (230, 160, 92, 5), (370, 132, 64, 6), (470, 156, 80, 7),
             (600, 128, 58, 8), (680, 150, 72, 9)]
    for (x, y, w, s) in specs:
        _cumulus(cv, x, y, w, s)
    # thin high streaks
    for (x, y, w) in ((60, 104, 40), (300, 98, 52), (520, 106, 36)):
        cv.hline(x, x + w, y, mix('white', 'light_sky', 0.4))
        cv.hline(x + 6, x + w - 8, y - 1, 'white')
    return _finish(cv)


@lru_cache(maxsize=None)
def _rf_far():
    """Far rose mesas (k=0.55) with faint pastel falls; hazy."""
    cv = _layer(640, 360)
    haze = '#c4ecff'
    seq = [(4, 5), (2, 9), (1, 6), (0, 10), (2, 8), (1, 12), (0, 30)]
    ramp = np.array([[mix(c, haze, 0.52) for c in _ramp(rc)] for rc in ROCK], np.uint8)
    mesas = [(40, 150, 182), (180, 290, 192), (330, 420, 186), (430, 560, 178), (575, 640, 190)]
    for i, (xa, xb, top) in enumerate(mesas):
        bot = 330
        tex, _ = _rock_texture(xa - 10, top, xb - xa + 20, bot - top, 30 + i, seq, ramp, colw=(50, 90), top=top)
        for y in range(top, bot):
            g = y - top
            l = xa - int(g * 0.35) - (2 if (g // 7) % 2 else 0)
            r = xb + int(g * 0.35) + (2 if (g // 9) % 2 else 0)
            l, r = max(0, xa - 10, l), min(cv.w, xb + 9, r)
            cv.a[y, l:r] = tex[g, l - (xa - 10):r - (xa - 10)]
        cv.rect(xa, top - 2, xb - xa, 3, mix('green', haze, 0.5))
        cv.hline(xa + 1, xb - 2, top - 2, mix('leaf', haze, 0.45))
        # faint pastel falls
        for k in range(1 + (xb - xa) // 70):
            fx_ = xa + 16 + k * 44 + i * 5
            if fx_ + 4 >= xb:
                continue
            for j, cc in enumerate(RB_PASTEL[::2]):
                cv.rect(fx_ + j, top + 1, 1, 60 + i * 6, mix(cc, haze, 0.25))
    cv.dither(0, 250, 640, 40, haze, 0.35)
    return _finish(cv)


@lru_cache(maxsize=None)
def _rf_mid():
    """Mid buttes (k=0.75) left and right of the main mesa, green tops, curly trees."""
    cv = _layer(640, 380)
    haze = '#c4ecff'
    ramp = np.array([[mix(c, haze, 0.25) for c in _ramp(rc)] for rc in ROCK], np.uint8)
    seq = BAND_SEQ[3:] + BAND_SEQ[:3]
    for (xa, xb, top, s) in ((10, 120, 186, 41), (440, 560, 180, 42)):
        bot = 380
        tex, _ = _rock_texture(xa - 20, top, xb - xa + 40, bot - top, s, seq, ramp, colw=(40, 70), top=top)
        for y in range(top, bot):
            g = y - top
            l = xa - int(g * 0.22) - 3 * int(hash01(s, g // 10) * 2)
            r = xb + int(g * 0.22) + 3 * int(hash01(s, g // 12, 1) * 2)
            l, r = max(0, xa - 20, l), min(cv.w, xb + 19, r)
            cv.a[y, l:r] = tex[g, l - (xa - 20):r - (xa - 20)]
        cv.rect(xa - 1, top - 3, xb - xa + 2, 4, mix('green', haze, 0.25))
        cv.hline(xa, xb - 1, top - 3, mix('leaf', haze, 0.2))
        for k in range(5):
            tx = xa + 8 + int(hash01(s, k) * (xb - xa - 16))
            _curly_tree(cv, tx, top - 1, 3 + int(hash01(s, k, 2) * 3), s * 10 + k,
                        dark=mix('forest', haze, 0.3), mid=mix('green', haze, 0.25),
                        lit=mix('leaf', haze, 0.2), hi=mix('pale_leaf', haze, 0.2))
    # a thin far fall on the right butte
    for j, cc in enumerate(RB_NAMES):
        cv.rect(470 + j, 181, 1, 120, mix(cc, haze, 0.35))
    cv.dither(0, 300, 640, 80, haze, 0.25)
    return _finish(cv)


# ---- main mesa
def _near_fall(x, pad=2):
    return any(x0 - pad <= x < x0 + w + pad for (x0, w, _) in FALLS)


def _cliff_home(cv, x, y, seed, awning='sky'):
    """Cliff dwelling dug into the rock: arched door, round window, plank balcony with railing,
    striped awning, flower pots, a lantern. (x, y) = left end of the balcony floor."""
    r = rng(seed)
    # recess shadow + door
    cv.rect(x + 3, y - 12, 12, 12, mix('plum_brown', 'mauve', 0.3))
    cv.rect(x + 4, y - 11, 10, 11, mix('plum_brown', 'night', 0.3))
    cv.rect(x + 5, y - 9, 5, 9, 'latte')
    cv.rect(x + 5, y - 10, 5, 1, 'stone')
    cv.px(x + 5, y - 10, mix('plum_brown', 'night', 0.3))
    cv.px(x + 9, y - 10, mix('plum_brown', 'night', 0.3))
    cv.vline(x + 5, y - 9, y - 1, 'crema')
    cv.px(x + 8, y - 5, 'gold')
    cv.circle(x + 12, y - 7, 1, 'cream')
    cv.px(x + 12, y - 7, 'baby_blue')
    # stone arch frame
    for xx in range(x + 3, x + 15):
        cv.px(xx, y - 13, 'stone' if xx % 2 else 'mist')
    # awning (striped, scalloped)
    for k in range(13):
        cc = awning if k % 2 == 0 else 'white'
        cv.vline(x + 3 + k, y - 15, y - 13, cc)
        if k % 2 == 0:
            cv.px(x + 3 + k, y - 12, shade(awning, 0.8))
    # balcony floor + braces + railing
    cv.hline(x, x + 17, y, 'crema')
    cv.hline(x, x + 17, y + 1, 'brew_mid')
    cv.line(x + 2, y + 2, x + 6, y + 6, 'brew_mid')
    cv.line(x + 15, y + 2, x + 11, y + 6, 'brew_mid')
    for xx in range(x, x + 18, 3):
        cv.vline(xx, y - 3, y - 1, 'latte')
    cv.hline(x, x + 17, y - 4, 'crema')
    # pots + lantern + flag
    for k in range(2):
        px_ = x + 1 + k * 14
        cv.rect(px_, y - 2, 2, 2, 'clay')
        cv.px(px_, y - 3, ('hot_pink', 'gold')[k])
        cv.px(px_ + 1, y - 3, 'leaf')
    cv.vline(x + 16, y - 10, y - 5, 'night')
    cv.px(x + 16, y - 11, 'gold')


def _stairs(cv, pts):
    """Zig-zag wooden stairway with handrails along a polyline."""
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        n = max(abs(xb - xa), abs(yb - ya))
        for i in range(n + 1):
            t = i / n
            x = int(round(xa + (xb - xa) * t))
            y = int(round(ya + (yb - ya) * t))
            cv.px(x, y, 'crema')
            cv.px(x, y + 1, 'brew_mid')
            if i % 3 == 0:
                cv.px(x, y - 1, 'latte')
            if i % 6 == 0:
                cv.vline(x, y - 4, y - 1, 'brew_mid')
            cv.px(x, y - 4, 'latte')
        # landing
        cv.hline(xb - 3, xb + 3, yb + 1, 'crema')
        cv.hline(xb - 3, xb + 3, yb + 2, 'brew_mid')
        cv.vline(xb, yb + 3, yb + 6, 'brew_mid')


HOUSES = [  # x, w, wall_h, roof_h, wall, roof, kw
    (142, 22, 13, 10, 'crema', 'rust', dict(seed=1)),
    (170, 26, 22, 12, 'latte', 'brick', dict(seed=2, two=True, chimney=True)),
    (222, 24, 14, 10, 'clay', 'teal', dict(seed=4, shutters='cream')),
    (250, 32, 18, 14, 'crema', 'rust', dict(seed=5, chimney=True)),
    (318, 22, 15, 10, 'latte', 'plum', dict(seed=7, shutters='gold')),
    (388, 24, 22, 12, 'crema', 'brick', dict(seed=8, two=True, gable=False, chimney=True)),
    (416, 20, 14, 10, 'clay', 'teal', dict(seed=10, shutters='hot_pink')),
]


@lru_cache(maxsize=None)
def _rf_main():
    """The main mesa (k=1): rock face, green top, village, talus, pools, river bed."""
    WWm, WHm = 600, 330
    cv = _layer(WWm, WHm)
    # --- rock face
    tex, ledge = _rock_texture(0, MESA_TOP, WWm, WHm - MESA_TOP, 3, top=MESA_TOP - 2)
    for y in range(MESA_TOP, MESA_BOT + 6):
        l, r = mesa_left(y), mesa_right(y)
        cv.a[y, l:r] = tex[y - MESA_TOP, l:r]
        cv.a[y, l:l + 2] = ROCK_RAMP[4, 3]
        cv.a[y, r - 3:r] = ROCK_RAMP[3, 0]
    # shadow right under the overhanging lip
    for x in range(mesa_left(MESA_TOP), mesa_right(MESA_TOP)):
        d = 2 + int(hash01(x // 3, 31) * 3)
        for y in range(MESA_TOP + 2, MESA_TOP + 2 + d):
            cv.a[y, x] = (cv.a[y, x].astype(int) * 0.62 + np.array(col('plum_brown')) * 0.38).astype(np.uint8)
    # wet dark rock + moss around each fall
    for (x0, w, _) in FALLS:
        for y in range(MESA_TOP + 2, MESA_BOT):
            for dx in (-3, -2, -1, w, w + 1, w + 2):
                xx = x0 + dx
                if hash01(xx, y, 5) < (0.9 if dx in (-1, w) else 0.5):
                    cv.a[y, xx] = (cv.a[y, xx].astype(int) * 0.7 + np.array(col('plum_brown')) * 0.3).astype(np.uint8)
            if hash01(x0, y, 8) < 0.3:
                side = hash01(x0, y, 3) < 0.5
                xx = x0 - 2 - int(hash01(x0, y, 9) * 5) if side else x0 + w + 1 + int(hash01(x0, y, 9) * 5)
                cv.px(xx, y, 'green' if hash01(x0, y, 4) < 0.6 else 'leaf')
    # ledge shrubs on hard band tops
    r = rng(11)
    ys, xs = np.nonzero(ledge)
    for i in r.choice(len(ys), 140, replace=False):
        y, x = ys[i] + MESA_TOP, xs[i]
        if not (mesa_left(y) + 3 < x < mesa_right(y) - 4) or y > MESA_BOT - 4 or y < MESA_TOP + 8:
            continue
        if _near_fall(x, 3):
            continue
        k = r.random()
        if k < 0.5:
            cv.hline(x, x + 2, y - 1, 'sage')
            cv.px(x + 1, y - 2, 'leaf')
        elif k < 0.85:
            cv.circle(x, y - 2, 2, 'green')
            cv.px(x - 1, y - 3, 'leaf')
            cv.px(x + 1, y - 1, 'forest')
        else:
            _curly_tree(cv, x, y, 3, 700 + i)
    # --- cliff life: stairway, dwellings, vines
    _stairs(cv, [(160, 152), (204, 168), (156, 186), (204, 204), (156, 222), (204, 240), (168, 256)])
    for (hx, hy, s, aw) in ((236, 184, 1, 'sky'), (322, 206, 2, 'hot_pink'), (402, 178, 3, 'gold'),
                            (244, 240, 4, 'leaf'), (414, 232, 5, 'violet'), (112, 214, 6, 'red'),
                            (460, 200, 7, 'sky')):
        _cliff_home(cv, hx, hy, s, aw)
    l0, r0 = mesa_left(MESA_TOP), mesa_right(MESA_TOP)
    for x in range(l0 + 2, r0 - 2):
        if _near_fall(x, 3) or hash01(x, 51) > 0.22:
            continue
        ln = 3 + int(hash01(x, 52) * 12)
        for k in range(ln):
            xx = x + (1 if (k // 4) % 2 else 0)
            cv.px(xx, 153 + k, 'green' if k % 3 else 'sage')
        if hash01(x, 53) < 0.5:
            cv.px(x, 153 + ln, ('hot_pink', 'lavender', 'gold')[int(hash01(x, 54) * 3)])
    # --- plateau top: grass band + lip overhang
    cv.rect(l0 - 1, 142, r0 - l0 + 2, 9, 'green')
    for x in range(l0 - 1, r0 + 1):
        h = int(hash01(x, 1) * 3)
        cv.vline(x, 140 - (h if hash01(x, 7) < 0.3 else 0), 143, mix('leaf', 'green', 0.4))
        if hash01(x, 2) < 0.5:
            cv.px(x, 141 + int(hash01(x, 3) * 2), 'leaf')
        d = int(hash01(x, 4) * 4)
        cv.vline(x, 150, 151 + d, 'green' if d < 3 else 'forest')
        if hash01(x, 6) < 0.4:
            cv.px(x, 152 + d, 'forest')
    cv.hline(l0, r0, 150, 'forest')
    cv.hline(l0, r0, 148, mix('green', 'leaf', 0.3))
    for x in range(l0, r0):
        if hash01(x, 9) < 0.08:
            cv.px(x, 143 + int(hash01(x, 10) * 4), ('hot_pink', 'gold', 'white', 'lavender')[int(hash01(x, 11) * 4)])
    for x in range(l0 + 10, r0 - 10):
        cv.px(x, 146, mix('sand', 'cream', 0.45))
        if hash01(x, 12) < 0.6:
            cv.px(x, 147, mix('sand', 'tan', 0.3))
    # --- background trees on the plateau (behind houses)
    for k, tx in enumerate((124, 168, 205, 246, 280, 324, 352, 372, 410, 438, 466)):
        _curly_tree(cv, tx, 144, 5 + int(hash01(k, 77) * 4), 200 + k)
    # --- village
    for (x, w, wh, rh, wc, rc, kw) in HOUSES:
        _wood_house(cv, x, 146, w, wh, rh, wc, rc, **kw)
    # judges' lookout tower (wooden lattice + cabin + cone roof)
    tx0 = 202
    for yy in range(108, 146):
        cv.px(tx0, yy, 'brew_mid')
        cv.px(tx0 + 11, yy, 'brew')
    for yy in range(112, 146, 8):
        cv.line(tx0, yy, tx0 + 11, yy + 8, 'latte')
        cv.line(tx0 + 11, yy, tx0, yy + 8, 'brew_mid')
        cv.hline(tx0, tx0 + 11, yy, 'crema')
    cv.rect(tx0 - 2, 98, 16, 10, 'crema')
    cv.rect(tx0 - 2, 98, 2, 10, mix('crema', 'cream', 0.4))
    cv.rect(tx0 + 12, 98, 2, 10, 'latte')
    cv.rect(tx0 + 1, 100, 10, 4, mix('baby_blue', 'sky', 0.25))
    cv.px(tx0 + 1, 100, 'white')
    cv.vline(tx0 + 6, 100, 103, 'cream')
    cv.hline(tx0 - 3, tx0 + 14, 107, 'brew_mid')
    cv.hline(tx0 - 3, tx0 + 14, 108, 'latte')
    for xx in range(tx0 - 3, tx0 + 15, 2):
        cv.vline(xx, 105, 107, 'cream')
    cv.poly([(tx0 - 4, 98), (tx0 + 6, 86), (tx0 + 7, 86), (tx0 + 17, 98)], 'teal')
    cv.poly([(tx0 + 7, 86), (tx0 + 17, 98), (tx0 + 8, 98)], 'deep_teal')
    cv.line(tx0 - 4, 98, tx0 + 6, 86, mix('teal', 'white', 0.4))
    cv.hline(tx0 - 4, tx0 + 17, 98, 'deep_teal')
    cv.vline(tx0 + 6, 76, 86, 'stone')
    # arched footbridges over the streams of the main fall and fall 4
    for (bx0, bx1) in ((282, 312), (360, 382)):
        for xx in range(bx0, bx1 + 1):
            t = (xx - bx0) / (bx1 - bx0)
            yy = 143 - int(round(math.sin(t * math.pi) * 4))
            cv.vline(xx, yy, yy + 1, 'latte')
            cv.px(xx, yy - 1, 'crema')
            if xx % 4 == 0:
                cv.vline(xx, yy - 4, yy - 1, 'brew_mid')
        mid = (bx0 + bx1) // 2
        cv.line(bx0, 139, mid, 135, 'crema')
        cv.line(mid, 135, bx1, 139, 'crema')
    # flower garden + well between the bridges / by fall 4
    for k in range(14):
        gx = 342 + k
        cv.px(gx, 142 - (k % 3 == 0), ('hot_pink', 'gold', 'lavender', 'red')[k % 4])
    cv.rect(346, 136, 7, 6, 'stone')
    cv.hline(346, 352, 136, 'mist')
    cv.vline(346, 128, 135, 'brew_mid')
    cv.vline(352, 128, 135, 'brew_mid')
    cv.poly([(344, 129), (349, 125), (354, 129)], 'rust')
    # grandstand (Games trials): tiered bleachers with a striped awning
    gx = 440
    for i in range(5):
        cv.rect(gx + i * 5, 146 - (i + 1) * 4, 34 - i * 5, 4, 'latte' if i % 2 else 'crema')
        cv.hline(gx + i * 5, gx + 33, 146 - (i + 1) * 4, 'cream')
    for xx in range(gx, gx + 35, 5):
        cv.vline(xx, 126, 145, 'brew_mid')
    for k in range(8):
        cc = ('sky', 'white', 'gold', 'white', 'hot_pink', 'white', 'sky', 'white')[k]
        cv.rect(gx - 2 + k * 5, 120, 5, 4, cc)
        cv.px(gx - 2 + k * 5 + 2, 124, cc)
    cv.hline(gx - 2, gx + 37, 119, 'brew_mid')
    cv.hline(gx - 2, gx + 37, 124, mix('ink', 'sky', 0.3))
    # fence along the lip
    for x in range(l0 + 6, r0 - 6):
        if _near_fall(x, 2):
            continue
        if x % 5 == 0:
            cv.vline(x, 144, 148, 'latte')
            cv.px(x, 144, 'cream')
        cv.px(x, 145, 'crema')
    # lamp posts
    for lx in (164, 244, 314, 384, 436):
        cv.vline(lx, 134, 146, 'night')
        cv.rect(lx - 1, 131, 3, 3, 'gold')
        cv.px(lx, 132, 'butter')
        cv.hline(lx - 1, lx + 1, 130, 'night')
    # --- stream channels feeding the falls (rainbow strip on the top) + rock lip at the notch
    for (x0, w, _) in FALLS:
        for j in range(w):
            si = min(5, j * 6 // w)
            cv.vline(x0 + j, 146, 147, RB_RAMP[si, 1])
            cv.px(x0 + j, 146, RB_RAMP[si, 2])
        for d in range(3):
            cv.vline(x0 - 1 - d, 146 + d, 152, 'forest' if d == 0 else 'green')
            cv.vline(x0 + w + d, 146 + d, 152, 'forest' if d == 0 else 'green')
    # --- talus base, pools, river bed (behind the trestle)
    for x in range(0, WWm):
        tb = MESA_BOT - 6 + int(4 * math.sin(x * 0.07) + 3 * hash01(x // 5, 2))
        cv.vline(x, tb, RIVER_Y - 1, mix('mauve', 'tan', 0.4) if (x // 4 + tb) % 3 else mix('tan', 'peach', 0.4))
        if hash01(x, 13) < 0.25:
            cv.px(x, tb, 'green')
    for k in range(40):
        bx = int(hash01(k, 21) * WWm)
        by = MESA_BOT - 2 + int(hash01(k, 22) * 4)
        br = 2 + int(hash01(k, 23) * 3)
        cv.ellipse(bx, by, br + 1, br, mix('mauve', 'plum_brown', 0.3))
        cv.ellipse(bx - 1, by - 1, br, br - 1, mix('tan', 'peach', 0.3))
        cv.px(bx - br // 2, by - br + 1, 'blush')
    for k in range(26):
        bx = int(hash01(k, 31) * WWm)
        _curly_tree(cv, bx, RIVER_Y - 1, 3 + int(hash01(k, 32) * 3), 400 + k)
    cv.rect(0, RIVER_Y, WWm, WHm - RIVER_Y, mix('sky', 'teal', 0.45))
    for y in range(RIVER_Y, WHm):
        if (y - RIVER_Y) % 5 == 1:
            for x in range(0, WWm, 9):
                cv.hline(x + (y * 3) % 9, x + (y * 3) % 9 + 3, y, mix('sky', 'teal', 0.2))
    cv.hline(0, WWm - 1, RIVER_Y, mix('light_sky', 'white', 0.3))
    # rainbow reflections of the falls in the river (static part)
    for (x0, w, _) in FALLS:
        for j in range(w):
            si = min(5, j * 6 // w)
            for y in range(RIVER_Y + 1, RIVER_Y + 14):
                if (y + j) % 2 == 0 and hash01(x0 + j, y) < 0.8 - (y - RIVER_Y) * 0.05:
                    cv.px(x0 + j, y, mix(RB_NAMES[si], 'teal', 0.45))
    return _finish(cv)


# ---- falls (dynamic)
@lru_cache(maxsize=None)
def _fall_tex(w, seed, period=72):
    """Flow texture levels (period x w) in 0..3; tiled vertically."""
    r = rng(seed)
    lv = np.ones((period, w), np.int8)
    for j in range(w):
        for k in range(3):
            y0 = int(r.integers(0, period))
            ln = int(r.integers(5, 16))
            ys = (np.arange(y0, y0 + ln)) % period
            lv[ys, j] = 2
            if r.random() < 0.45:
                lv[ys[:max(2, ln // 3)], j] = 3
        y0 = int(r.integers(0, period))
        ys = (np.arange(y0, y0 + int(r.integers(4, 10)))) % period
        lv[ys, j] = 0
    lv[:, 0] = np.minimum(lv[:, 0], 1)
    lv[:, -1] = 0
    return lv


@lru_cache(maxsize=None)
def _stripe_idx(w):
    return np.minimum(5, (np.arange(w) * 6) // w)


def draw_falls(c, f, ox, oy):
    for i, (x0, w, spd) in enumerate(FALLS):
        sx = x0 - ox
        if sx > W or sx + w < 0:
            continue
        top, bot = 148, MESA_BOT + 1
        ya, yb = max(0, top - oy), min(H, bot - oy)
        if yb <= ya:
            continue
        lv = _fall_tex(w, 50 + i)
        P = lv.shape[0]
        rows = (np.arange(ya + oy, yb + oy) - f * spd) % P
        L = lv[rows].astype(int)
        img = RB_RAMP[_stripe_idx(w)[None, :], L]
        xa, xb = max(0, sx), min(W, sx + w)
        c.a[ya:yb, xa:xb] = img[:, xa - sx:xb - sx]
        # lip: glassy water curling over the edge, 1 px wider each side
        si = _stripe_idx(w)
        for ry, ext, lev in ((148, 0, 2), (149, 0, 2), (150, 0, 1)):
            yy = ry - oy
            if not 0 <= yy < H:
                continue
            for j in range(-ext, w + ext):
                xx = sx + j
                if 0 <= xx < W:
                    jj = min(w - 1, max(0, j))
                    lv_ = lev if 0 <= j < w else 1
                    if ry == 148 and (j + f // 3) % 5 == 0:
                        lv_ = 3
                    c.a[yy, xx] = RB_RAMP[si[jj], lv_]
        # spray flecks along the sides
        for k in range(10):
            yy = top + int(hash01(i, k, 1) * (bot - top))
            yy = top + (yy - top + f * spd) % (bot - top)
            side = -1 if k % 2 else w
            c.px(sx + side + (1 if side > 0 else -1) * int(hash01(i, k, f // 3) * 2), yy - oy,
                 'white' if (k + f // 2) % 3 else mix('white', RB_NAMES[k % 6], 0.4))


def draw_fall_mist(c, f, ox, oy):
    """Foam + prismatic mist at the foot of every fall."""
    for i, (x0, w, spd) in enumerate(FALLS):
        cx = x0 + w // 2 - ox
        by = MESA_BOT + 1 - oy
        if cx < -40 or cx > W + 40 or by < -20 or by > H + 30:
            continue
        rw = w // 2 + 6
        c.ellipse(cx, by + 1, rw + 3, 4, mix('cloud', 'sky', 0.3))
        c.ellipse(cx, by, rw + 1, 3, 'cloud')
        for k in range(7):
            ph = (f * 0.9 + k * 13 + i * 7) % 28
            t = ph / 28
            px_ = cx + int((hash01(i, k) - 0.5) * (rw * 2 + 6)) + int((t - 0.5) * 6 * (1 if k % 2 else -1))
            py = by - int(t * 16) - 1
            rr = int(1 + t * 4)
            colr = 'white' if t < 0.35 else mix('white', RB_PASTEL[(k + i) % 6], 0.6)
            if t < 0.75:
                c.circle(px_, py, rr, colr)
            else:
                c.dither(px_ - rr, py - rr, rr * 2 + 1, rr * 2 + 1, colr, 0.5, ox=f % 2)
        for k in range(4):
            c.px(cx + int((hash01(i, k, f // 2) - 0.5) * rw * 2), by - 2 - int(hash01(i, k, 3) * 4), 'white')
        sparkle(c, cx - rw + int(hash01(i, 9) * rw * 2), by - 8 - int(hash01(i, 8) * 6), f, 'white',
                period=26, phase=i * 0.23, size=2)
    # prismatic bow over the main fall's pool
    x0, w, _ = MAIN_FALL
    cx, cy = x0 + w // 2 - ox, MESA_BOT - 4 - oy
    for j, name in enumerate(RB_NAMES):
        R = 22 - j
        for a in range(0, 180, 3):
            if (a // 3 + j + f // 6) % 2:
                continue
            ang = math.radians(a)
            px_ = cx + int(round(math.cos(ang) * R * 1.3))
            py = cy - int(round(math.sin(ang) * R))
            if 0 <= px_ < W and 0 <= py < H:
                c.a[py, px_] = (c.a[py, px_].astype(int) + np.array(col(name))) // 2


def draw_river(c, f, ox, oy):
    y0 = RIVER_Y - oy
    if y0 >= H:
        return
    for k in range(60):
        wx = int(hash01(k, 41) * 700) - 50
        wy = RIVER_Y + 3 + int(hash01(k, 42) * 26)
        x = (wx + f * (1 + k % 2) // 2) % 660 - 30 - ox
        y = wy - oy
        if 0 <= y < H:
            ln = 2 + k % 4
            c.hline(x, x + ln, y, 'light_sky' if k % 3 else 'white')
    # rainbow sheen where the falls meet the river
    for i, (x0, w, _) in enumerate(FALLS):
        for j in range(3):
            y = RIVER_Y + 2 + j * 3 - oy
            x = x0 - ox + ((f // 3 + j * 5) % (w + 8)) - 4
            c.hline(x, x + 2, y, RB_PASTEL[(j * 2 + i) % 6])


# ---- trestle (k=1.15) + train
TRK = 1.15
RAIL_Y = 258


@lru_cache(maxsize=None)
def _rf_trestle():
    WWt, WHt = 640, 380
    cv = _layer(WWt, WHt)
    st = [mix('stone', 'tan', 0.35), mix('paper', 'tan', 0.25), mix('cream', 'paper', 0.4)]
    st_sh = mix('grey', 'mauve', 0.35)
    st_dk = mix('shadow', 'plum_brown', 0.4)
    top = RAIL_Y + 1
    # deck + spandrel wall
    cv.rect(0, top, WWt, WHt - top, st[1])
    # mortar courses
    for y in range(top + 3, WHt, 4):
        cv.hline(0, WWt - 1, y, st[0])
        for x in range((y // 4) % 2 * 4, WWt, 8):
            cv.px(x, y - 1, st[0])
            cv.px(x, y - 2, st[0])
    # big arches
    span, pier = 56, 12
    for k in range(-1, WWt // span + 2):
        ax = k * span + pier
        aw = span - pier
        cx = ax + aw / 2
        rad = aw / 2
        crown = top + 17
        for y in range(int(crown), WHt):
            if y < crown + rad:
                dy = crown + rad - y
                hw = math.sqrt(max(0, rad * rad - dy * dy))
            else:
                hw = rad
            xa, xb = int(round(cx - hw)), int(round(cx + hw))
            if xb > xa:
                cv.a[y, max(0, xa):max(0, xb)] = _KEYA
                # intrados shadow (right side of the opening is the lit pier face)
                if 0 <= xa < WWt:
                    cv.a[y, xa] = col(st_dk)
                if 0 <= xa + 1 < WWt:
                    cv.a[y, xa + 1] = col(st_sh)
                if 0 <= xb - 1 < WWt:
                    cv.a[y, xb - 1] = col(st[2])
        # voussoirs around the arch
        for a in range(0, 181, 9):
            ang = math.radians(a)
            px_ = int(round(cx + math.cos(ang) * (rad + 2)))
            py = int(round(crown + rad - math.sin(ang) * (rad + 2)))
            cv.px(px_, py, st[0])
            cv.px(int(round(cx + math.cos(ang) * (rad + 4))), int(round(crown + rad - math.sin(ang) * (rad + 4))), st[0])
        # keystone
        cv.rect(int(cx) - 1, int(crown) - 4, 3, 4, st[2])
        cv.hline(int(cx) - 1, int(cx) + 1, int(crown) - 4, 'white')
        # pier: lit left edge, shaded right
        px0 = k * span
        cv.vline(px0, top + 8, WHt - 1, st[2])
        cv.vline(px0 + pier - 1, top + 16, WHt - 1, st_sh)
        # pilaster + small oculus in the spandrel above the pier
        ocx = px0 + pier // 2
        cv.circle(ocx, top + 11, 2, st_dk)
        cv.px(ocx - 1, top + 10, st_sh)
        cv.px(ocx + 1, top + 12, st[2])
    # deck: coping, parapet posts, ballast + sleepers + rails
    cv.rect(0, top, WWt, 3, st[2])
    cv.hline(0, WWt - 1, top, 'white')
    cv.hline(0, WWt - 1, top + 3, st_sh)
    cv.hline(0, WWt - 1, top + 4, st_dk)
    cv.hline(0, WWt - 1, RAIL_Y - 1, 'steel_hi')
    cv.hline(0, WWt - 1, RAIL_Y, 'steel_dk')
    for x in range(0, WWt, 4):
        cv.px(x, RAIL_Y + 1, 'brew_mid')
        cv.px(x + 1, RAIL_Y + 1, 'latte')
    # lamp/telegraph posts every other pier
    for k in range(0, WWt // span + 1, 2):
        x = k * span + 6
        cv.vline(x, RAIL_Y - 16, RAIL_Y - 1, 'brew_mid')
        cv.hline(x - 3, x + 3, RAIL_Y - 14, 'brew_mid')
        cv.px(x - 3, RAIL_Y - 15, 'white')
        cv.px(x + 3, RAIL_Y - 15, 'white')
    # mossy streaks + flowers on the spandrels
    for k in range(120):
        x = int(hash01(k, 61) * WWt)
        y = top + 5 + int(hash01(k, 62) * 10)
        if np.all(cv.a[y, x] == _KEYA):
            continue
        cv.px(x, y, 'sage' if k % 3 else 'leaf')
    return _finish(cv)


def _wheel(c, x, y, r, ang, rim='steel_dk', hub='steel_hi'):
    c.circle(x, y, r, rim)
    c.circle(x, y, max(0, r - 1), 'ink' if r > 2 else rim)
    c.px(x, y, hub)
    c.px(x + round(math.cos(ang) * (r - 1)), y + round(math.sin(ang) * (r - 1)), hub)


TRAIN_SPEED = 3.0        # px / frame in trestle-layer x, travelling LEFT
TRAIN_LEN = 120
_TW, _TH = TRAIN_LEN + 6, 28


def train_front(f):
    """Front (left end) of the loco in trestle-layer x."""
    return TRAIN_A - (f - 144) * TRAIN_SPEED


@lru_cache(maxsize=16)
def _train_img(phase):
    """The Friendship Express drawn facing right (flipped when blitted); phase = wheel angle step."""
    c = Canvas(_TW, _TH, KEY)
    X, Y = _TW - 4, _TH - 2
    ang = phase * math.pi / 4
    coaches = (('cotton', 'bubblegum', 'periwinkle'), ('bubblegum', 'hot_pink', 'royal_lt'),
               ('cotton', 'lavender', 'periwinkle'))
    for k, (body, low, roofc) in enumerate(coaches):
        x0 = X - 46 - (k + 1) * 25
        c.rect(x0, Y - 17, 24, 13, 'berry')
        c.rect(x0 + 1, Y - 16, 22, 6, body)
        c.rect(x0 + 1, Y - 10, 22, 5, low)
        c.hline(x0 + 1, x0 + 22, Y - 10, 'gold')
        c.hline(x0 + 1, x0 + 22, Y - 16, mix(body, 'white', .5))
        for wx in range(x0 + 3, x0 + 21, 5):
            c.rect(wx, Y - 15, 3, 4, 'baby_blue')
            c.hline(wx, wx + 2, Y - 15, 'glass_hi')
            c.hline(wx, wx + 2, Y - 12, mix('baby_blue', 'periwinkle', 0.5))
        c.rect(x0 - 1, Y - 19, 26, 2, roofc)
        c.hline(x0, x0 + 23, Y - 20, mix(roofc, 'white', .4))
        c.hline(x0 + 1, x0 + 22, Y - 4, 'eggplant')
        for wx in (x0 + 4, x0 + 8, x0 + 16, x0 + 20):
            _wheel(c, wx, Y - 2, 2, ang * 1.5)
        c.rect(x0 + 24, Y - 7, 1, 2, 'steel_dk')
    tx = X - 45
    c.rect(tx, Y - 14, 13, 11, 'berry')
    c.rect(tx + 1, Y - 13, 11, 9, 'hot_pink')
    c.hline(tx + 1, tx + 11, Y - 9, 'gold')
    c.vline(tx + 1, Y - 13, Y - 5, 'bubblegum')
    for k in range(4):
        c.circle(tx + 2 + k * 3, Y - 15, 1, 'night' if k % 2 else 'charcoal')
    for wx in (tx + 3, tx + 9):
        _wheel(c, wx, Y - 2, 2, ang * 1.5)
    cx0 = X - 32
    c.rect(cx0, Y - 21, 9, 17, 'berry')
    c.rect(cx0 + 1, Y - 20, 7, 15, 'hot_pink')
    c.vline(cx0 + 7, Y - 20, Y - 6, 'bubblegum')
    c.rect(cx0 + 3, Y - 18, 4, 4, mix('eggplant', 'baby_blue', .3))
    c.px(cx0 + 6, Y - 18, 'baby_blue')
    c.hline(cx0 + 1, cx0 + 7, Y - 11, 'gold')
    c.rect(cx0 - 1, Y - 23, 11, 2, 'violet')
    c.hline(cx0 - 1, cx0 + 9, Y - 24, 'lavender')
    for y in range(Y - 15, Y - 5):
        t = (y - (Y - 15)) / 9
        cc = ('cotton' if t < 0.12 else 'bubblegum' if t < 0.3 else 'hot_pink' if t < 0.65 else
              'rose' if t < 0.86 else 'raspberry')
        c.hline(X - 23, X - 7, y, cc)
    c.hline(X - 23, X - 7, Y - 16, 'berry')
    for gx in (X - 20, X - 14):
        c.vline(gx, Y - 15, Y - 6, 'gold')
    c.rect(X - 7, Y - 16, 5, 11, 'plum')
    c.rect(X - 6, Y - 15, 3, 9, 'berry')
    c.px(X - 4, Y - 11, 'gold')
    c.rect(X - 6, Y - 19, 3, 3, 'gold')
    c.px(X - 5, Y - 18, 'butter')
    c.rect(X - 12, Y - 22, 3, 6, 'raspberry')
    c.vline(X - 10, Y - 22, Y - 17, 'rose')
    c.rect(X - 13, Y - 25, 5, 3, 'magenta')
    c.hline(X - 13, X - 9, Y - 25, 'gold')
    c.ellipse(X - 17, Y - 16, 2, 1, 'gold')
    c.rect(X - 22, Y - 18, 1, 2, 'brass')
    c.rect(X - 30, Y - 5, 28, 2, 'eggplant')
    c.poly([(X - 3, Y - 5), (X - 3, Y), (X + 3, Y)], 'amber')
    c.line(X - 2, Y - 1, X, Y - 3, 'gold')
    c.rect(X - 7, Y - 6, 4, 2, 'raspberry')
    _wheel(c, X - 6, Y - 2, 2, ang * 1.5)
    pins = []
    for wx in (X - 25, X - 18, X - 11):
        _wheel(c, wx, Y - 3, 3, ang)
        pins.append((wx + round(math.cos(ang) * 2), Y - 3 + round(math.sin(ang) * 2)))
    c.line(pins[0][0], pins[0][1], pins[2][0], pins[2][1], 'steel_hi')
    a = c.a[:, ::-1].copy()
    return a, np.any(a != _KEYA, axis=2)


def draw_train(c, f, ox, oy):
    """Returns the loco front screen x (or None when off-screen)."""
    Xs = int(round(train_front(f))) - ox
    if Xs > W + 4 or Xs + TRAIN_LEN < -4:
        return None
    phase = int(train_front(f) // 2) % 8
    img = _train_img(phase)
    _paste(c, img, -(Xs - 3), -((RAIL_Y - 1 - oy) - (_TH - 2)))
    return Xs


def _puff(c, x, y, r, a_fade=0.0):
    """Solid cartoon steam puff (shadow lower-right, highlight upper-left)."""
    r = int(round(r))
    if r < 1:
        return
    c.circle(x + 1, y + 1, r, mix('cloud_dk', 'periwinkle', 0.25))
    c.circle(x, y, r, 'white' if a_fade < 0.5 else 'cloud')
    if r >= 3:
        c.circle(x - r // 3, y - r // 3, max(1, r // 2), 'white')


def draw_steam(c, f, ox, oy, X):
    """Chuffs from the funnel on 8ths (left hanging in the air), whistle jet + burst at the hit."""
    for k in range(6):
        e = (f // 9 - k) * 9
        a = f - e
        if a < 0 or a > 46 or e < 110 or e > 214:
            continue
        ex = int(round(train_front(e))) + 11 - ox
        ey = RAIL_Y - 28 - oy
        px_ = ex + int(a * 0.45)
        py = ey - int(a * 0.6)
        r = 2 + min(a, 32) * 0.14
        if a > 32:
            r *= 1 - (a - 32) / 15
        _puff(c, px_, py, r, a / 46)
        if a > 6 and r > 2:
            _puff(c, px_ + int(r), py + 2, r * 0.6, a / 46)
    if X is None:
        return
    a = f - 144
    wx, wy = X + 22, RAIL_Y - 20 - oy
    if 0 <= a < 34:
        for k in range(7):
            rr = (1.5 + k * 0.45 + a * 0.12) * (1 - max(0, a - 22) / 14)
            _puff(c, wx + k * 3 + a // 2 + (k % 2) * 2, wy - 4 - k * 4 - a // 2 + (k % 3), rr, a / 34)
    if 0 <= a < 6 or 9 <= a < 26:
        c.vline(wx, wy - 5, wy - 1, 'white')
        c.vline(wx + 1, wy - 8, wy - 4, mix('white', 'cloud', 0.4))
        c.px(wx, wy - 9, 'white')


# ---- foreground (k=1.4)
FGK = 1.4


@lru_cache(maxsize=None)
def _rf_fore():
    cv = _layer(680, 380)
    # grassy rocky knoll bottom-right with flowers and a curly tree
    for x in range(330, 680):
        t = (x - 330) / 350
        top = int(300 - 26 * math.sin(min(1.0, t * 1.6) * math.pi / 2) + 3 * math.sin(x * 0.2))
        cv.vline(x, top, 379, 'green')
        cv.vline(x, top + 3, 379, mix('green', 'forest', 0.5))
        cv.vline(x, top, top + 1, 'leaf')
        if hash01(x, 71) < 0.35:
            cv.vline(x, top - 2, top - 1, 'leaf' if hash01(x, 72) < 0.5 else 'green')
        if hash01(x, 73) < 0.1:
            cv.px(x, top - 1 - int(hash01(x, 74) * 3), ('hot_pink', 'gold', 'white', 'lavender', 'red')[int(hash01(x, 75) * 5)])
    for (bx, by, br) in ((378, 296, 3), (436, 286, 4), (488, 282, 3)):
        cv.ellipse(bx + 1, by + 1, br + 1, br - 1, mix('mauve', 'plum_brown', 0.5))
        cv.ellipse(bx, by, br + 1, br - 1, mix('tan', 'peach', 0.35))
        cv.px(bx - 1, by - br + 2, mix('blush', 'white', 0.3))
    for k, (bx, by, br) in enumerate(((400, 292, 6), (414, 290, 5), (462, 284, 7), (500, 280, 5), (560, 276, 8))):
        cv.circle(bx + 1, by + 1, br, 'forest')
        cv.circle(bx, by, br, 'green')
        cv.circle(bx - br // 3, by - br // 3, br // 2, 'leaf')
        for j in range(3):
            cv.px(bx - br // 2 + j * br // 2, by - br + 1 + (j % 2), ('hot_pink', 'white', 'gold')[(j + k) % 3])
    _curly_tree(cv, 520, 276, 16, 901)
    _curly_tree(cv, 452, 280, 9, 902)
    return _finish(cv)


# ---- balloon path (world, k=1; basket bottom) = the flight-trial course line
BAL_KEYS = [(56, (196.0, 74.0)), (68, (226.0, 66.0)), (80, (256.0, 64.0)), (92, (284.0, 74.0)),
            (104, (296.0, 100.0)), (118, (298.0, 140.0)), (132, (300.0, 176.0)), (146, (310.0, 200.0)),
            (158, (334.0, 208.0)), (170, (360.0, 196.0)), (182, (370.0, 164.0)), (194, (374.0, 126.0)),
            (206, (378.0, 92.0)), (218, (386.0, 66.0)), (232, (396.0, 46.0))]


def _catmull(f, keys):
    fs = [k[0] for k in keys]
    if f <= fs[0]:
        return keys[0][1]
    if f >= fs[-1]:
        return keys[-1][1]
    i = max(0, min(len(fs) - 2, int(np.searchsorted(fs, f, side='right')) - 1))
    p0 = keys[max(0, i - 1)][1]
    p1, p2 = keys[i][1], keys[i + 1][1]
    p3 = keys[min(len(keys) - 1, i + 2)][1]
    u = (f - fs[i]) / (fs[i + 1] - fs[i])
    u2, u3 = u * u, u * u * u
    return tuple(0.5 * ((2 * b) + (-a + c_) * u + (2 * a - 5 * b + 4 * c_ - d) * u2 + (-a + 3 * b - 3 * c_ + d) * u3)
                 for a, b, c_, d in zip(p0, p1, p2, p3))


def balloon_pos(f):
    return _catmull(f, BAL_KEYS)


RING_R = 12
RING_FRAMES = (78, 91, 106, 138, 178, 192, 206)
EXTRA_RINGS = ((118.0, 100.0, -20.0), (164.0, 84.0, -24.0))     # start of the course (left)


def _ring_list():
    out = list(EXTRA_RINGS)
    for fr in RING_FRAMES:
        x, y = balloon_pos(fr)
        x1, y1 = balloon_pos(fr - 1)
        x2, y2 = balloon_pos(fr + 1)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        out.append((x, y - 10, ang))          # ring centre = balloon centre
    return out


RINGS = _ring_list()


def _course_dots():
    pts = []
    last = None
    for x, y, _ in EXTRA_RINGS[:1]:
        last = (x, y)
        pts.append(last)
    for i in range(0, 2 * (232 - 56)):
        fr = 56 + i * 0.5
        x, y = balloon_pos(fr)
        p = (x, y - 10)
        if last is None or (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= 36:
            pts.append(p)
            last = p
    # lead-in from the extra rings
    lead = []
    (ax, ay, _), (bx, by, _) = EXTRA_RINGS
    for k in range(1, 8):
        t = k / 8
        lead.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return [(int(round(x)), int(round(y))) for x, y in lead + pts]


COURSE = _course_dots()


@lru_cache(maxsize=None)
def _ring_sprites(ang10):
    """(back, front) RGBA sprites of a puffy cloud hoop whose plane is perpendicular to the travel
    direction `ang10`/10 degrees."""
    ang = ang10 / 10.0
    S = 2 * RING_R + 12
    back = np.zeros((S, S, 4), np.uint8)
    front = np.zeros((S, S, 4), np.uint8)
    cx = cy = S // 2
    a = math.radians(ang)
    t = (math.cos(a), math.sin(a))
    nrm = (-t[1], t[0])
    pts = []
    for i in range(40):
        th = i * 2 * math.pi / 40
        u, v = math.cos(th), math.sin(th)
        pts.append((cx + nrm[0] * RING_R * u + t[0] * 4.5 * v, cy + nrm[1] * RING_R * u + t[1] * 4.5 * v, v, i))
    yy, xx = np.mgrid[0:S, 0:S]
    for (x, y, v, i) in sorted(pts, key=lambda p: p[2]):
        tgt = front if v > 0 else back
        rad = 2.3 + 0.7 * (i % 3 == 0)
        d = (xx - x) ** 2 + (yy - y) ** 2
        m = d <= rad * rad
        base = col('cloud') if v > 0 else mix('cloud_dk', 'periwinkle', 0.3)
        tgt[m, :3] = base
        tgt[m, 3] = 255
        lo = m & (yy > y + 0.6)
        tgt[lo, :3] = col('cloud_dk') if v > 0 else mix('cloud_dk', 'periwinkle', 0.55)
        hi = (d <= (rad - 1.2) ** 2) & (yy < y - 0.2) & (xx < x + 0.5)
        tgt[hi, :3] = col('white') if v > 0 else col('cloud')
    # trial bands: four short coloured wraps
    for j, i in enumerate((5, 15, 25, 35)):
        x, y, v, _ = pts[i]
        tgt = front if v > 0 else back
        cc = col(('gold', 'sky', 'gold', 'sky')[j]) if v > 0 else mix(('gold', 'sky', 'gold', 'sky')[j], 'periwinkle', 0.4)
        m = ((xx - x) ** 2 + (yy - y) ** 2 <= 2.2) 
        tgt[m & (tgt[:, :, 3] > 0), :3] = cc
    top = min(pts, key=lambda p: p[1])
    return back, front, (int(round(top[0])) - cx, int(round(top[1])) - cy)


def ring_pos(i, f):
    x, y, ang = RINGS[i]
    return int(round(x)), int(round(y + math.sin(f * 2 * math.pi / 90 + i * 1.3) * 1.2)), ang


def draw_course(c, f, ox, oy):
    """Twinkling gold dotted flight line linking the hoops (RPG route style)."""
    n = len(COURSE)
    for k, (x, y) in enumerate(COURSE):
        sx, sy = x - ox, y - oy
        if 0 <= sx < W and 0 <= sy < H:
            hot = (k - f // 2) % 9 == 0
            c.px(sx, sy, 'white' if hot else ('gold' if k % 2 else mix('gold', 'cream', 0.5)))


def draw_rings(c, f, ox, oy, part):
    for i in range(len(RINGS)):
        x, y, ang = ring_pos(i, f)
        sx, sy = x - ox, y - oy
        if sx < -30 or sx > W + 30 or sy < -30 or sy > H + 30:
            continue
        back, front, (tx, ty) = _ring_sprites(int(round(ang / 5.0)) * 50)
        S = back.shape[0]
        c.blit(back if part == 0 else front, sx - S // 2, sy - S // 2)
        if part == 1:
            px_, py = sx + tx, sy + ty - 2
            c.vline(px_, py - 6, py, 'brew_mid')
            c.px(px_, py - 7, 'gold')
            cc = ('red', 'sky', 'gold', 'hot_pink', 'leaf')[i % 5]
            ph = (f // 3 + i) % 4
            ln = (5, 6, 5, 4)[ph]
            for k in range(ln):
                hh = 2 if k < ln - 2 else 1
                dy = (0, 0, 1, 1, 0, -1)[(k + ph) % 6] if k > 1 else 0
                c.vline(px_ + 1 + k, py - 6 + dy, py - 6 + dy + hh - 1, cc if k < ln - 1 else shade(cc, 0.8))


# ---- flags + bunting on the village (dynamic)
FLAGS = [(208, 76, 'gold', 7), (183, 103, 'sky', 5), (266, 105, 'hot_pink', 5), (329, 112, 'gold', 4),
         (396, 103, 'sky', 5), (444, 111, 'gold', 4), (456, 111, 'sky', 4), (468, 111, 'hot_pink', 4)]
BUNTING = [((158, 128), (172, 126)), ((196, 126), (224, 132)), ((246, 132), (252, 128)),
           ((340, 132), (358, 137)), ((382, 137), (390, 127)), ((412, 127), (418, 132)), ((436, 133), (440, 124))]


@lru_cache(maxsize=None)
def _village_windows():
    cv = _layer(600, 200)
    out = []
    for (x, w, wh, rh, wc, rc, kw) in HOUSES:
        out += [(wx, wy) for (wx, wy) in _wood_house(cv, x, 146, w, wh, rh, wc, rc, **kw)]
    return out


def draw_village_life(c, f, ox, oy):
    for i, (x, y, cc, ln) in enumerate(FLAGS):
        sx, sy = x - ox, y - oy
        if sx < -10 or sx > W + 10 or sy < -10 or sy > H + 10:
            continue
        c.vline(sx, sy, sy + (8 if ln < 6 else 10), 'brew_mid')
        c.px(sx, sy - 1, 'gold')
        ph = (f // 3 + i * 2) % 4
        for k in range(ln):
            off_ = (0, 1, 1, 0, -1, -1)[(k + ph) % 6] if k > 0 else 0
            hh = 3 if ln >= 6 else 2
            hh = max(1, hh - (k * hh) // (ln + 1))
            c.vline(sx + 1 + k, sy + off_, sy + off_ + hh, cc if (k + ph) % 3 else mix(cc, 'white', 0.35))
    for j, ((xa, ya), (xb, yb)) in enumerate(BUNTING):
        n = max(2, abs(xb - xa) // 4)
        for k in range(n + 1):
            t = k / n
            x = int(round(xa + (xb - xa) * t)) - ox
            y = int(round(ya + (yb - ya) * t + math.sin(t * math.pi) * 3)) - oy
            c.px(x, y, 'cream')
            if 0 < k < n:
                cc = ('red', 'gold', 'sky', 'leaf', 'hot_pink', 'violet')[(k + j) % 6]
                sw = (k + f // 4 + j) % 5 == 0
                c.px(x, y + 1, cc)
                c.px(x + (1 if sw else 0), y + 2, cc)
    # chimney smoke (house 2, 5, 7)
    for (x, y, s) in ((191, 111, 1), (277, 113, 2), (407, 107, 3)):
        steam(c, x - ox, y - oy, f, color=mix('white', 'cloud', 0.3), n=4, height=16, spread=3, seed=s, speed=0.4)
    # window glints sweeping (sun catching the glass)
    for i, (x, y) in enumerate(_village_windows()):
        if (f + i * 13) % 61 < 3:
            c.px(x - ox, y - oy, 'white')
            c.px(x + 1 - ox, y + 1 - oy, 'white')


# ---- train timing anchor: the loco front sits at screen x ~214 on the hit frame
TRAIN_A = 150 + off(cam_rf(144), TRK)[0]


def render_rf(f):
    cam = cam_rf(f)
    c = Canvas()
    # sky
    sky = _sky_tall()
    oy = off(cam, 0.2)[1] - 80
    oy = max(0, min(sky.shape[0] - H, oy))
    c.a[:] = sky[oy:oy + H]
    # clouds (drifting right)
    ox, oy = off(cam, 0.12)
    _paste(c, _rf_clouds(), ox - int(f * 0.15) + 40, oy)
    _paste(c, _rf_peaks(), *off(cam, 0.3))
    _paste(c, _rf_far(), *off(cam, 0.55))
    _paste(c, _rf_mid(), *off(cam, 0.75))
    ox, oy = off(cam, 1.0)
    _paste(c, _rf_main(), ox, oy)
    draw_river(c, f, ox, oy)
    draw_falls(c, f, ox, oy)
    draw_fall_mist(c, f, ox, oy)
    draw_village_life(c, f, ox, oy)
    draw_course(c, f, ox, oy)
    draw_rings(c, f, ox, oy, 0)
    bx, by = balloon_pos(f)
    flame = 160 <= f <= 212 and (f // 6) % 3 != 2
    draw_balloon(c, int(round(bx)) - ox, int(round(by)) - oy, f, 'm', flame=flame)
    draw_rings(c, f, ox, oy, 1)
    tox, toy = off(cam, TRK)
    _paste(c, _rf_trestle(), tox, toy)
    X = draw_train(c, f, tox, toy)
    draw_steam(c, f, tox, toy, X)
    _paste(c, _rf_fore(), *off(cam, FGK))
    return c, cam


# ------------------------------------------------------------------ OUR TOWN (bar 4)
OT_SAND = mix('#d9c49a', 'sand', 0.12)
OT_SAND_LT = mix('#d9c49a', 'cream', 0.45)
OT_SAND_DK = col('#a4876a')
OT_ROOF = col('#b5523b')
OT_SKY = ['#3f5fb8', 'blue', 'sky', '#6db6f0', 'light_sky', mix('light_sky', 'blush', 0.55)]
OT_HAZE = mix('light_sky', 'blush', 0.5)
OT_STREET_Y = (110, 122)          # street band between the rows (world y)


def cam_ot(f):
    return ilerp(0, 40, ease_in_out(seg(f, 216, 287)))


@lru_cache(maxsize=None)
def _ot_sky():
    c = Canvas(W, 100)
    c.gradient(0, 0, W, 100, OT_SKY, soft=0.7)
    return c.a


@lru_cache(maxsize=None)
def _ot_clouds():
    cv = _layer(560, 120)
    for (x, y, w, s) in ((10, 40, 64, 21), (140, 26, 48, 22), (240, 44, 84, 23), (380, 30, 56, 24), (470, 46, 70, 25)):
        _cumulus(cv, x, y, w, s)
    return _finish(cv)


@lru_cache(maxsize=None)
def _ot_far():
    """k=0.25: pale far mesas on the horizon + Starlight's Cave on grey crags (left)."""
    cv = _layer(420, 180)
    hz = 80
    far = mix('#d9c49a', OT_HAZE, 0.55)
    far_sh = mix('#a4876a', OT_HAZE, 0.55)
    for (xa, xb, top) in ((60, 150, 66), (170, 250, 70), (262, 330, 64), (340, 410, 69)):
        cv.rect(xa, top, xb - xa, hz - top + 4, far)
        cv.rect(xb - 8, top + 2, 8, hz - top + 2, far_sh)
        cv.hline(xa, xb - 1, top, mix(far, 'white', 0.3))
        for y in range(top + 4, hz, 5):
            cv.hline(xa + 1, xb - 9, y, mix(far, far_sh, 0.5))
    # grey crags with the cave
    g1, g2, g3 = mix('stone', OT_HAZE, 0.35), mix('grey', OT_HAZE, 0.35), mix('shadow', OT_HAZE, 0.3)
    peaks = [(84, 84), (92, 58), (100, 62), (108, 42), (117, 50), (124, 32), (134, 44), (142, 36), (152, 54),
             (162, 48), (172, 66), (182, 84)]
    peaks = [(x, y - 4) for (x, y) in peaks]
    cv.poly(peaks + [(182, 90), (84, 90)], g2)
    for (px_, py) in peaks[1:-1]:
        cv.line(px_, py, px_ - 4, py + 18, g1)
        cv.line(px_ + 1, py + 1, px_ + 6, py + 20, g3)
    for (px_, py) in peaks[1:-1]:
        cv.px(px_, py, mix(g1, 'white', 0.5))
    # cave mouth (dark, with a lit rim, no figures)
    CX, CY = 130, 58
    cv.ellipse(CX, CY, 6, 6, mix('navy', 'shadow', 0.5))
    cv.rect(CX - 6, CY, 13, 9, mix('navy', 'shadow', 0.5))
    cv.ellipse(CX, CY + 1, 4, 5, mix('night', 'navy', 0.4))
    cv.rect(CX - 4, CY + 1, 9, 8, mix('night', 'navy', 0.4))
    for a in range(0, 181, 20):
        cv.px(CX + int(round(math.cos(math.radians(a)) * 7)), CY - int(round(math.sin(math.radians(a)) * 7)), g1)
    cv.line(CX - 9, CY + 9, CX - 30, CY + 24, mix(g1, 'white', 0.3))
    # rail line passing by the crags (thin)
    cv.hline(60, 220, 82, mix('brew_mid', OT_HAZE, 0.4))
    for x in range(60, 220, 3):
        cv.px(x, 83, mix('brew_mid', OT_HAZE, 0.55))
    for y in range(78, 92):
        t = (y - 78) / 14
        row = cv.a[y]
        m = np.any(row != _KEYA, axis=1)
        row[m] = (row[m] * (1 - t * 0.6) + np.array(col(OT_HAZE)) * t * 0.6).astype(np.uint8)
    cv.rect(0, 90, 420, 90, KEY)
    return _finish(cv)


def _hoodoo(cv, x, base, h, w, seed, haze=0.0):
    """Sandstone hoodoo: stacked rounded drums of tan/sand strata with a dark caprock."""
    cols = [mix(c_, OT_HAZE, haze) for c_ in (OT_SAND_LT, '#d9c49a', mix('clay', 'tan', 0.5), mix('peach', 'tan', 0.4))]
    sh = mix(OT_SAND_DK, OT_HAZE, haze)
    y = base
    k = 0
    r = rng(seed)
    while y > base - h:
        dh = int(r.integers(4, 8))
        ww = max(2, int(w * (0.55 + 0.45 * r.random())))
        cv.rect(x - ww // 2, y - dh, ww, dh, cols[k % 4])
        cv.vline(x - ww // 2, y - dh, y - 1, mix(cols[k % 4], 'white', 0.3))
        cv.rect(x + ww // 2 - 1, y - dh, 2, dh, sh)
        cv.hline(x - ww // 2, x + ww // 2, y - dh, mix(cols[k % 4], 'white', 0.25))
        y -= dh
        k += 1
    cw = w + 3
    cv.ellipse(x, y - 1, cw // 2, 2, mix('copper', 'plum_brown', 0.45 - haze * 0.2))
    cv.hline(x - cw // 2 + 1, x + cw // 2 - 2, y - 3, mix('clay', 'white', 0.2 + haze * 0.3))


@lru_cache(maxsize=None)
def _ot_mid():
    """k=0.5: sandstone mesas + hoodoos around the valley."""
    cv = _layer(520, 180)
    seq = [(4, 4), (0, 7), (2, 5), (1, 6), (0, 6), (2, 8), (3, 5), (0, 9), (2, 30)]
    ramp = np.array([[mix(c_, OT_HAZE, 0.22) for c_ in _ramp(rc)] for rc in
                     (mix('#d9c49a', 'sand', 0.3), mix('clay', 'peach', 0.4), mix('#d9c49a', 'cream', 0.3),
                      mix('#a4876a', 'clay', 0.3), mix('blush', 'tan', 0.3))], np.uint8)
    for i, (xa, xb, top) in enumerate(((-10, 56, 50), (190, 260, 60), (310, 380, 54), (430, 520, 48))):
        bot = 120
        tex, _ = _rock_texture(xa - 10, top, xb - xa + 20, bot - top, 60 + i, seq, ramp, colw=(30, 50), top=top)
        for y in range(top, bot):
            g = y - top
            l = xa - int(g * 0.5) - (1 if (g // 5) % 2 else 0)
            r = xb + int(g * 0.5) + (1 if (g // 6) % 2 else 0)
            l, r = max(0, xa - 10, l), min(cv.w, xb + 9, r)
            cv.a[y, l:r] = tex[g, l - (xa - 10):r - (xa - 10)]
        cv.hline(max(0, xa), xb - 1, top, mix('cream', OT_HAZE, 0.2))
    for (x, h, w, s) in ((82, 30, 7, 1), (94, 22, 6, 2), (172, 36, 8, 3), (282, 30, 6, 4), (294, 22, 5, 5),
                         (400, 36, 7, 6), (414, 28, 6, 7)):
        _hoodoo(cv, x, 104, h, w, s, haze=0.2)
    cv.dither(0, 96, 520, 14, OT_HAZE, 0.3)
    return _finish(cv)


@lru_cache(maxsize=None)
def _ot_vault():
    """k=0.85: the sandstone cliff with the round cave vault door (right), switchback path."""
    cv = _layer(460, 180)
    seq = [(0, 6), (2, 7), (1, 5), (3, 8), (0, 6), (2, 9), (1, 30)]
    ramp = np.array([_ramp(rc) for rc in (mix('#d9c49a', 'sand', 0.3), mix('clay', 'peach', 0.35),
                                          mix('#d9c49a', 'cream', 0.25), mix('#a4876a', 'clay', 0.25))], np.uint8)
    xa, xb, top = 226, 470, 64
    tex, _ = _rock_texture(xa - 30, top, xb - xa + 30, 120, 77, seq, ramp, colw=(26, 44), top=top)
    for y in range(top, 120):
        g = y - top
        l = xa - int(g * 0.9) + int(3 * math.sin(g * 0.4))
        l = max(xa - 30, l)
        cv.a[y, l:460] = tex[g, l - (xa - 30):460 - (xa - 30)]
        cv.a[y, l:l + 1] = col(mix('cream', 'sand', 0.3))
    cv.hline(xa, 459, top, 'cream')
    for k in range(13):
        _hoodoo(cv, xa + 14 + k * 17, top, 6 + int(hash01(k, 3) * 9), 5, 40 + k)
    # vault cave + round door
    vx, vy = 266, 82
    cv.ellipse(vx, vy, 11, 11, mix('plum_brown', 'night', 0.35))
    cv.rect(vx - 11, vy, 23, 12, mix('plum_brown', 'night', 0.35))
    cv.circle(vx, vy + 1, 8, 'steel_dk')
    cv.circle(vx, vy + 1, 7, 'steel')
    cv.circle(vx - 1, vy, 5, 'steel_hi')
    cv.circle(vx, vy + 1, 4, 'steel')
    for a in range(0, 360, 45):
        cv.px(vx + int(round(math.cos(math.radians(a)) * 6)), vy + 1 + int(round(math.sin(math.radians(a)) * 6)), 'steel_dk')
    # "=" emblem on the wheel
    cv.hline(vx - 2, vx + 2, vy, 'royal_lt')
    cv.hline(vx - 2, vx + 2, vy + 2, 'royal_lt')
    cv.rect(vx + 7, vy - 2, 3, 6, 'brass')
    # stone threshold + path down to the valley
    cv.rect(vx - 12, vy + 11, 25, 2, 'stone')
    pts = [(vx + 10, vy + 13), (vx + 34, vy + 18), (vx + 12, vy + 24), (vx + 40, vy + 30), (vx + 52, vy + 36)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv.line(x0, y0, x1, y1, OT_SAND_LT)
        cv.line(x0, y0 + 1, x1, y1 + 1, mix('#d9c49a', 'tan', 0.4))
    cv.dither(150, 112, 310, 8, OT_SAND, 0.5)
    return _finish(cv)


def _ot_house(cv, x, base, w, wall_h, roof_h, chimney=True):
    """Identical cream house with a red roof, seen from the hill (front wall + roof slope from above)."""
    cr = _ramp('cream', deep='tan', light='white')
    cv.rect(x, base - wall_h, w, wall_h, cr[2])
    cv.vline(x, base - wall_h, base - 1, cr[4])
    cv.rect(x + w - 2, base - wall_h, 2, wall_h, cr[1])
    cv.hline(x, x + w - 1, base - 1, cr[0])
    # door + two windows
    dx = x + w // 2 - 1
    cv.rect(dx, base - 5, 3, 4, mix('#a4876a', 'brew_mid', 0.4))
    cv.px(dx + 2, base - 3, 'gold')
    for wx in (x + 2, x + w - 5):
        cv.rect(wx, base - wall_h + 2, 3, 3, mix('baby_blue', 'sky', 0.3))
        cv.px(wx, base - wall_h + 2, 'white')
        cv.hline(wx, wx + 2, base - wall_h + 5, cr[0])
    # roof: trapezoid slope seen from above, shingle rows, lit ridge
    rr = [mix(OT_ROOF, 'maroon', 0.5), mix(OT_ROOF, 'maroon', 0.2), OT_ROOF, mix(OT_ROOF, 'coral', 0.4), mix(OT_ROOF, 'peach', 0.6)]
    yb = base - wall_h
    cv.poly([(x - 1, yb + 1), (x + 3, yb - roof_h), (x + w - 3, yb - roof_h), (x + w + 1, yb + 1)], rr[2])
    for yy in range(yb - roof_h + 2, yb + 1, 2):
        for xx in range(x, x + w, 3):
            cv.px(xx + (yy // 2) % 2, yy, rr[1])
    cv.hline(x + 3, x + w - 4, yb - roof_h, rr[4])
    cv.hline(x + 3, x + w - 4, yb - roof_h + 1, rr[3])
    cv.line(x - 1, yb + 1, x + 3, yb - roof_h, rr[3])
    cv.line(x + w - 3, yb - roof_h, x + w + 1, yb + 1, rr[0])
    cv.hline(x - 1, x + w + 1, yb + 1, rr[0])
    if chimney:
        cv.rect(x + w - 6, yb - roof_h - 3, 3, 4, cr[2])
        cv.vline(x + w - 4, yb - roof_h - 3, yb - roof_h, cr[1])
        cv.hline(x + w - 7, x + w - 3, yb - roof_h - 3, mix('#a4876a', 'tan', 0.5))
    # short noon shadow under the eaves / at the foot
    cv.hline(x, x + w - 1, yb + 2, cr[1])
    cv.hline(x + 1, x + w + 1, base, mix(OT_SAND_DK, OT_SAND, 0.4))


OT_BACK = [(78 + i * 21, 108) for i in range(9)]      # back row (x, base)
OT_FRONT = [(70 + i * 23, 136) for i in range(9)]     # front row
OT_TREE = (284, 124)
OT_POLE = (56, 124)


@lru_cache(maxsize=None)
def _ot_town():
    """k=1: the valley floor, the two rows, the street, fences, rail end, the big tree trunk."""
    cv = _layer(400, 180)
    # valley floor (dusty), with texture
    cv.gradient(0, 92, 400, 88, [mix(OT_SAND, OT_HAZE, 0.25), OT_SAND, mix(OT_SAND, OT_SAND_DK, 0.25)], soft=0.8)
    for k in range(700):
        x = int(hash01(k, 1) * 400)
        y = 94 + int(hash01(k, 2) * 86)
        cc = OT_SAND_DK if k % 3 else OT_SAND_LT
        cv.px(x, y, cc)
        if k % 5 == 0:
            cv.px(x + 1, y, cc)
    # sagebrush + small rocks
    for k in range(70):
        x = int(hash01(k, 11) * 400)
        y = 96 + int(hash01(k, 12) * 80)
        if OT_STREET_Y[0] - 14 < y < OT_STREET_Y[1] + 16 and 50 < x < 300:
            continue
        cv.ellipse(x, y, 2, 1, mix('sage', 'pale_sage', 0.4))
        cv.px(x - 1, y - 1, 'pale_sage')
        cv.px(x + 1, y + 1, mix('sage', 'olive', 0.4))
    # the street (lighter, packed earth) + ruts
    y0, y1 = OT_STREET_Y
    cv.rect(46, y0, 256, y1 - y0, OT_SAND_LT)
    cv.hline(46, 301, y0, mix(OT_SAND_LT, OT_SAND_DK, 0.3))
    for yy in (y0 + 4, y0 + 8):
        for x in range(50, 298, 2):
            cv.px(x + (yy % 2), yy, mix(OT_SAND_LT, OT_SAND, 0.6))
    # side paths from the doors to the street
    for (x, b) in OT_BACK:
        cv.rect(x + 7, b, 3, y0 - b, mix(OT_SAND_LT, OT_SAND, 0.4))
    # roads in from the station (left) and up to the vault (right)
    for x in range(14, 50):
        yy = int(126 - (x - 14) * 0.3)
        cv.hline(x, x, yy, OT_SAND_LT)
        cv.px(x, yy + 1, OT_SAND_LT)
    for x in range(300, 400):
        yy = int(116 - (x - 300) * 0.06)
        cv.px(x, yy, OT_SAND_LT)
        cv.px(x, yy + 1, mix(OT_SAND_LT, OT_SAND, 0.5))
    # vegetable plots behind the back row, fenced
    for k in range(4):
        gx = 84 + k * 48
        cv.rect(gx, 92, 34, 7, mix('#a4876a', 'brew_mid', 0.25))
        for yy in (93, 95, 97):
            for xx in range(gx + 1, gx + 33, 2):
                cv.px(xx, yy, 'leaf' if (xx + yy) % 4 else 'green')
        if k % 2:
            for xx in range(gx + 2, gx + 32, 5):
                cv.px(xx, 94, 'orange')
        cv.hline(gx - 1, gx + 34, 91, 'latte')
        for xx in range(gx - 1, gx + 35, 4):
            cv.vline(xx, 90, 92, 'brew_mid')
    # front gardens (flowers) before the front row
    for (x, b) in OT_FRONT:
        for k in range(6):
            cv.px(x + 1 + k * 3, b + 2, ('hot_pink', 'leaf', 'gold', 'leaf', 'lavender', 'leaf')[k])
            cv.px(x + 1 + k * 3, b + 3, 'green')

    # rows of identical houses
    for (x, b) in OT_BACK:
        _ot_house(cv, x, b, 16, 9, 5)
    # rail end of the line (left): track + buffer stop + tiny platform shelter
    for x in range(0, 44):
        cv.px(x, 128, 'steel')
        cv.px(x, 130, 'steel')
        if x % 3 == 0:
            cv.vline(x, 128, 131, 'brew_mid')
    cv.rect(42, 125, 3, 5, 'red')
    cv.hline(41, 45, 125, 'gold')
    cv.rect(8, 120, 22, 2, '#a4876a')
    cv.rect(8, 117, 22, 3, OT_ROOF)
    cv.vline(9, 120, 127, 'brew_mid')
    cv.vline(28, 120, 127, 'brew_mid')
    cv.rect(8, 126, 22, 2, 'stone')
    for (x, b) in OT_FRONT:
        _ot_house(cv, x, b, 18, 10, 6)
    # fences at the street ends
    for (xa, ya) in ((40, 104), (40, 136), (304, 104), (304, 136)):
        for x in range(xa, xa + 10, 3):
            cv.vline(x, ya - 3, ya, 'latte')
        cv.hline(xa, xa + 9, ya - 2, 'crema')
    # the bakery: one house has a tiny cupcake sign
    bx, bb = OT_FRONT[3]
    cv.vline(bx + 18, bb - 9, bb - 6, 'brew_mid')
    cv.rect(bx + 19, bb - 9, 5, 4, 'cream')
    cv.rect(bx + 20, bb - 7, 3, 2, 'clay')
    cv.hline(bx + 20, bx + 22, bb - 8, 'bubblegum')
    cv.px(bx + 21, bb - 9, 'red')
    # tree trunk + roots + shade under the big tree
    tx, tb = OT_TREE
    cv.ellipse(tx + 4, tb + 1, 18, 3, mix(OT_SAND, OT_SAND_DK, 0.6))
    cv.rect(tx - 2, tb - 22, 5, 22, 'brew_mid')
    cv.vline(tx - 2, tb - 22, tb - 1, 'latte')
    cv.vline(tx + 2, tb - 22, tb - 1, 'brew')
    cv.line(tx - 2, tb - 1, tx - 6, tb + 1, 'brew_mid')
    cv.line(tx + 2, tb - 1, tx + 6, tb + 1, 'brew')
    cv.line(tx, tb - 14, tx - 7, tb - 22, 'brew_mid')
    cv.line(tx + 1, tb - 16, tx + 8, tb - 24, 'brew_mid')
    # banner pole base
    px_, pb = OT_POLE
    cv.rect(px_ - 2, pb - 1, 5, 2, 'stone')
    return _finish(cv)


def _ot_tree_canopy(c, f, x, y):
    """Big leafy tree canopy at the end of the street; slow sway of the upper puffs + leaf glints."""
    sway = 1 if (f // 16) % 4 in (1, 2) else 0
    puffs = [(0, 0, 14, 0), (-12, 4, 9, 0), (12, 3, 10, 0), (-6, -9, 10, 1), (7, -10, 9, 1), (0, -16, 8, 1),
             (-15, -3, 7, 0), (16, -4, 7, 1)]
    for (dx, dy, r, up) in puffs:
        c.circle(x + dx + 1 + sway * up, y + dy + 2, r, 'forest')
    for (dx, dy, r, up) in puffs:
        c.circle(x + dx + sway * up, y + dy, r, 'green')
    for (dx, dy, r, up) in puffs:
        c.circle(x + dx - r // 3 + sway * up, y + dy - r // 3, max(2, r // 2), 'leaf')
        c.px(x + dx - r // 2 + sway * up, y + dy - r // 2, 'pale_leaf')
    for k in range(10):
        a = hash01(k, 5) * 6.283
        rr = hash01(k, 6) * 14
        if (f // 4 + k * 3) % 7 == 0:
            c.px(x + int(math.cos(a) * rr), y - 4 + int(math.sin(a) * rr * 0.8), 'pale_leaf')
        c.px(x + int(math.cos(a) * rr) + 1, y - 3 + int(math.sin(a) * rr * 0.8), 'forest')


def _ot_banner(c, f, x, base):
    """The '=' banner on its pole, waving."""
    top = base - 46
    c.vline(x, top, base - 1, 'brew_mid')
    c.vline(x + 1, top, base - 1, 'brew')
    c.px(x, top - 1, 'gold')
    c.px(x + 1, top - 1, 'gold')
    c.hline(x + 1, x + 18, top + 2, 'brew_mid')
    for j in range(17):
        wv = 1 if math.sin((f * 0.22) - j * 0.45) * (j / 16) > 0.45 else 0
        xx = x + 2 + j
        y0 = top + 3 + wv
        shade_ = math.sin((f * 0.22) - j * 0.45 + 1.2) < -0.6
        c.vline(xx, y0, y0 + 15, mix('cream', 'tan', 0.25) if shade_ else 'cream')
        c.px(xx, y0 + 15, mix('cream', 'tan', 0.5))
        if 3 <= j <= 13:
            for (yy, cc) in ((4, 'royal'), (5, 'royal'), (6, 'royal_lt'), (9, 'royal'), (10, 'royal'), (11, 'royal_lt')):
                c.px(xx, y0 + yy, cc)
        if j == 16:
            c.vline(xx, y0, y0 + 15, mix('cream', 'tan', 0.45))


def _ot_dust_devil(c, f, x, y, h, seed):
    for k in range(h):
        w = 1 + k * 4 // h
        ph = f * 0.6 + k * 0.9 + seed
        dx = int(round(math.sin(ph) * w))
        cc = OT_SAND_LT if k % 3 else mix(OT_SAND_LT, 'white', 0.4)
        if hash01(seed, k, f // 2) < 0.75:
            c.px(x + dx, y - k, cc)
        if k > h // 2 and hash01(seed, k, f // 2, 1) < 0.5:
            c.px(x - dx, y - k, mix(OT_SAND, OT_SAND_LT, 0.5))
    for k in range(4):
        c.px(x + int(math.sin(f * 0.5 + k * 1.7) * 4), y + (k % 2), OT_SAND_LT)


def _ot_cloud_shadows(c, f, ox):
    """Dithered cloud shadows drifting across the valley floor."""
    y0, y1 = 94, 180
    reg = c.a[y0:y1]
    yy, xx = np.mgrid[y0:y1, 0:W]
    m = np.zeros((y1 - y0, W), bool)
    for (cx0, cy, rx, ry) in ((60, 150, 40, 9), (260, 112, 34, 6), (420, 166, 48, 10)):
        cx = int((cx0 + f * 0.5) % 520) - 100 - ox
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        m |= (d <= 0.7) | ((d <= 1.0) & (bayer(y1 - y0, W, 0, y0) < 0.5))
    reg[m] = (reg[m].astype(np.uint16) * 222 // 256).astype(np.uint8)


@lru_cache(maxsize=None)
def _ot_fore():
    """k=1.35: the hill we watch from (bottom): sandy lip, strata, dry grass, desert flowers,
    a rock outcrop (left) and a trail sign with an '=' icon (right)."""
    cv = _layer(420, 180)
    lip = mix('#d9c49a', 'cream', 0.35)
    bands = [mix('#d9c49a', 'sand', 0.35), mix('clay', 'peach', 0.45), mix('#a4876a', 'clay', 0.3),
             mix('#d9c49a', 'tan', 0.4)]
    for x in range(420):
        top = int(163 - 9 * math.exp(-((x - 30) / 46.0) ** 2) - 7 * math.exp(-((x - 380) / 44.0) ** 2)
                  + 1.5 * math.sin(x * 0.13))
        for y in range(top, 180):
            g = y - top
            b = bands[((g + int(2 * math.sin(x * 0.05))) // 4) % 4]
            cc = b if (x + y) % 7 else mix(b, 'plum_brown', 0.2)
            cv.a[y, x] = col(cc)
        cv.vline(x, top, top + 1, lip)
        cv.px(x, top + 2, mix(lip, bands[0], 0.5))
        if hash01(x, 81) < 0.45:
            hgt = 1 + int(hash01(x, 82) * 4)
            cv.vline(x, top - hgt, top - 1, ('khaki', 'pale_sage', mix('sand', 'khaki', 0.5), 'moss')[int(hash01(x, 83) * 4)])
        if hash01(x, 84) < 0.05:
            cv.px(x, top - 2 - int(hash01(x, 85) * 3), ('gold', 'amber', 'white', 'coral')[int(hash01(x, 86) * 4)])
    # rock outcrop (left) with strata + lit top
    cv.poly([(0, 150), (10, 140), (26, 138), (40, 144), (52, 156), (58, 180), (0, 180)], mix('#a4876a', 'clay', 0.3))
    cv.poly([(0, 150), (10, 140), (26, 138), (30, 142), (12, 146), (0, 156)], mix('#d9c49a', 'cream', 0.3))
    for (ya, cc) in ((150, mix('clay', 'peach', 0.4)), (158, mix('#d9c49a', 'sand', 0.3)), (166, mix('#a4876a', 'plum_brown', 0.2))):
        for x in range(0, 56):
            yy = ya + int(math.sin(x * 0.2) * 1.2)
            if cv.a[yy, x].tolist() != list(KEY) and x < 50 + (yy - 150) * 0.3:
                cv.px(x, yy, cc)
    cv.line(40, 144, 54, 172, mix('#a4876a', 'plum_brown', 0.45))
    # small rocks
    for (bx, by, br) in ((70, 166, 3), (330, 163, 2), (352, 160, 4)):
        cv.ellipse(bx + 1, by + 1, br + 1, br - 1, mix('#a4876a', 'plum_brown', 0.5))
        cv.ellipse(bx, by, br + 1, br - 1, mix('#d9c49a', 'clay', 0.35))
        cv.px(bx - 1, by - br + 2, mix('cream', 'sand', 0.4))
    # trail sign with the '=' icon
    sx, sy = 300, 161
    cv.vline(sx, sy - 14, sy, 'brew_mid')
    cv.rect(sx - 6, sy - 15, 13, 7, 'latte')
    cv.rect(sx - 5, sy - 14, 11, 5, 'crema')
    cv.hline(sx - 3, sx + 3, sy - 13, 'royal')
    cv.hline(sx - 3, sx + 3, sy - 11, 'royal')
    cv.px(sx + 7, sy - 12, 'latte')
    # a round desert shrub
    for (bx, by) in ((228, 160), (96, 162)):
        cv.circle(bx + 1, by, 3, mix('sage', 'olive', 0.4))
        cv.circle(bx, by - 1, 3, 'sage')
        cv.px(bx - 1, by - 3, 'pale_sage')
        cv.px(bx + 1, by - 2, 'gold')
    return _finish(cv)


def render_ot(f):
    cx = cam_ot(f)
    c = Canvas()
    c.a[:100] = _ot_sky()
    _paste(c, _ot_clouds(), int(round(cx * 0.1)) - int(f * 0.2) + 60, 0)
    _paste(c, _ot_far(), int(round(cx * 0.25)) + 10, 0)
    _paste(c, _ot_mid(), int(round(cx * 0.5)) + 20, 0)
    _paste(c, _ot_vault(), int(round(cx * 0.85)) + 30, 0)
    ox = int(round(cx))
    _paste(c, _ot_town(), ox, 0)
    _ot_cloud_shadows(c, f, ox)
    # front row is in the town layer; dynamic dressing on top
    _ot_dust_devil(c, f, 150 + int((f - 216) * 0.45) - ox, 128, 14, 3)
    _ot_dust_devil(c, f, 336 - int((f - 216) * 0.3) - ox, 100, 10, 7)
    for (x, b) in OT_BACK[1::3]:
        steam(c, x + 11 - ox, b - 18, f, color=mix('white', 'cloud', 0.3), n=3, height=12, spread=2, seed=x, speed=0.35)
    for (x, b) in OT_FRONT[2::3]:
        steam(c, x + 13 - ox, b - 20, f, color=mix('white', 'cloud', 0.3), n=3, height=12, spread=2, seed=x, speed=0.35)
    _ot_tree_canopy(c, f, OT_TREE[0] - ox, OT_TREE[1] - 34)
    _ot_banner(c, f, OT_POLE[0] - ox, OT_POLE[1])
    # vault glint
    vx = 266 - int(round(cx * 0.85)) - 30
    fx.sparkle(c, vx - 3, 80, f, 'white', period=30, phase=0.2, size=2)
    # window glints
    for i, (x, b) in enumerate(OT_BACK + OT_FRONT):
        if (f + i * 7) % 40 < 2:
            c.px(x + 2 - ox, b - (7 if b < 120 else 8), 'white')
    # balloon, small, drifting over the town
    draw_balloon(c, 196 + int((f - 216) * 0.35) - ox, 64, f, 's')
    # foreground hill + tumbleweed
    _paste(c, _ot_fore(), int(round(cx * 1.35)) + 10, 0)
    tw = int((f - 216) * 1.1) - 20 - int(round(cx * 1.35)) + 60
    ty = 162 - abs(int(math.sin(f * 0.5) * 2))
    c.circle(tw, ty, 3, mix('#a4876a', 'olive', 0.3))
    c.circle(tw, ty, 2, mix('khaki', 'tan', 0.4))
    for k in range(5):
        a = f * 0.5 + k * 1.3
        c.px(tw + int(round(math.cos(a) * 2)), ty + int(round(math.sin(a) * 2)), '#a4876a')
    return c


# ------------------------------------------------------------------ callouts / card / render
def _callouts(c, f):
    if f < 216:
        cam = cam_rf(f)
        ox, oy = off(cam, 1.0)
        # RAINBOW FALLS: on the main fall, the marker slides along the water to stay mid-frame
        x0, w, _ = MAIN_FALL
        ay = int(clamp(cam[1] + 92, 160, 262))
        callout(c, x0 + w - 2 - ox, ay - oy, 'RAINBOW FALLS', f, 86, 144, side='right', length=14)
        # RAIL TRESTLE: on the parapet above a pier
        tox, toy = off(cam, TRK)
        px_ = 56 * ((tox + 250) // 56) + 6
        callout(c, px_ - tox, RAIL_Y + 6 - toy, 'RAIL TRESTLE', f, 124, 174, side='down', length=9)
        # HOOP COURSE: on the first climbing hoop
        rx, ry, _ = ring_pos(6, f)
        callout(c, rx + RING_R - 1 - ox, ry - oy, 'HOOP COURSE', f, 156, 209, side='right', length=12)
    else:
        ox = int(round(cam_ot(f)))
        callout(c, 168 - ox, OT_STREET_Y[0] - 8, 'OUR TOWN', f, 222, 279, side='up', length=26)


def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < 216:
        c, cam = render_rf(f)
        if f < 80:
            iris(c, 1 - seg(f, 72, 79), 204, 100)
    else:
        c = render_ot(f)
    _callouts(c, f)
    atlas_card(c, f, 100, 192, 'CLIFFTOP VILLAGE', ['RAINBOW WATERFALLS', 'GAMES TRIAL GROUNDS'])
    return c


SFX = [
    (76, 'waterfall', 0.55, {'pan': 0.2}),
    (90, 'wind', 0.35, {'variant': 0}),
    (108, 'chug', 0.45, {'pan': 0.6}),
    (126, 'chug', 0.55, {'pan': 0.3}),
    (136, 'waterfall', 0.5),
    (144, 'whistle', 0.85),
    (144, 'chug', 0.7),
    (162, 'chug', 0.55, {'pan': -0.3}),
    (166, 'wind', 0.4, {'variant': 0}),
    (180, 'chug', 0.4, {'pan': -0.6}),
    (196, 'waterfall', 0.4),
    (220, 'wind', 0.45, {'variant': 0}),
    (244, 'creak', 0.3, {'pan': -0.5}),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
