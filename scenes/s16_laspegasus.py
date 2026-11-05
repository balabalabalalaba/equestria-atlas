"""LAS PEGASUS: the resort city on the clouds (5 bars, 360 frames, HUD clock 19:30 -> 20:30).

Beat sheet (local frames; bar = 72, beat = 18, 8th = 9). Dusk -> neon night over the west coast.
  Bar 1   0-71    World-map travel shot (`travel_frame`); iris closes 64-71 on (204, 100).
  Bar 2  72-143   REVEAL: iris opens 72-79 on the GRAND TOWER (crown + gold star). SHOT A, dusk:
                  sunset afterglow on the sea horizon behind the city (backlit silhouettes with
                  peach rim light), first stars, a thin crescent moon; far left the coast hills
                  with the APPLEWOOD sign + gold star and the little studio town on the shore; the
                  hotel/casino towers stand on a fluffy cloud bank lit pink from below; back cloud
                  row; drifting foreground puffs; a tiny sailboat on the sea far below. Camera
                  almost still. Callouts LAS PEGASUS 90-143, APPLEWOOD 108-170. Atlas card 100-196.
  Bar 3 144-215   `neon` HIT (144): the Grand Tower lights on the downbeat, then the cascade runs
                  down the strip, ONE BUILDING PER 8TH (153 horseshoe casino, 162 turret hotel,
                  171 card-suit tower, 180 cherry slots, 189 dice casino, 198 top-hat theatre,
                  207 gold resort + Ferris wheel): neon flicker on the stab, glow halos, then the
                  window grids light bottom-up; marquee bulbs chase; the sky darkens to deep
                  violet; neon reflections bloom on the cloud bank; searchlights sweep up from
                  Applewood. Eased pan right following the cascade. Callout THE STRIP 171-215.
  Bar 4 216-287   CUT, SHOT B: FOUNTAIN PLAZA at night: dancing jets (new pattern on every beat,
                  lit from below in shifting colours), the Ferris wheel turning with coloured
                  gondolas, the WILD BLUE YONDER coaster with an empty car racing the loop, a
                  slot-reel JACKPOT sign that lands three stars at 252 and flashes. Pan right.
                  Callouts FOUNTAIN PLAZA 222-281, WILD BLUE YONDER 240-287.
  Bar 5 288-359   `fireworks` HIT (288): CUT back to the lit strip; shells burst on 288, 324, 342
                  (big) with smaller ones between: trails, sparkle falloff, light flashes on the
                  towers and clouds; the balloon drifts across and is silhouetted against the 324
                  burst; the camera tilts up/pulls back into the firework sky. Toast from 302
                  (top-centre kept clear). Last 8 frames under the wipe.
Ambient: twinkling stars, sea glitter, drifting clouds, sailboat, searchlights, neon flicker,
         chasing marquee bulbs, blinking signs, rolling dice, window flickers, fountain jets,
         Ferris wheel, coaster car, fireworks, balloon bob.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.core import bayer
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'laspegasus'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)
T_A, T_NEON, T_B, T_C = 72, 144, 216, 288
WA = 480            # shot A world width (city layer)
GROUND = 134        # building base line (cloud top)
HORIZ = 112         # sea horizon
SUNX = 262          # afterglow centre (screen x)
FW = 360            # far layer width
BW = 420            # back cloud layer width
PAR_FAR, PAR_BACK, PAR_FG = 0.2, 0.5, 1.4


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


def xpaste(c, ra, rb, x, y, t):
    """Paste layer ra (t=0) or rb (t=1), dither cross-fade in between (dither tied to the layer)."""
    x, y = int(x), int(y)
    if t <= 0.0:
        return paste(c, ra, x, y)
    if t >= 1.0:
        return paste(c, rb, x, y)
    h, w = ra.shape[:2]
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1, sy1 = min(w, W - x), min(h, H - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    t = round(t * 12) / 12.0
    a = ra[sy0:sy1, sx0:sx1]
    b = rb[sy0:sy1, sx0:sx1]
    ma, mb = a[..., 3] >= 128, b[..., 3] >= 128
    rgb = (a[..., :3].astype(np.float32) * (1 - t) + b[..., :3].astype(np.float32) * t + 0.5)
    rgb = rgb.astype(np.uint8)
    rgb[ma & ~mb] = a[..., :3][ma & ~mb]
    rgb[mb & ~ma] = b[..., :3][mb & ~ma]
    comp = np.zeros_like(a)
    comp[..., :3] = rgb
    comp[..., 3] = (ma | mb) * 255
    c.blit(comp, x + sx0, y + sy0)


def eased_cam(f, keys):
    return int(round(keyframes(f, keys, ease=smooth)))


def add_light(c, arr, k=1.0):
    """Additive uint8/int16 light image (same size as the canvas)."""
    if k <= 0:
        return
    if k == 1.0:
        s = c.a.astype(np.int16) + arr
    else:
        s = c.a.astype(np.int16) + (arr * k).astype(np.int16)
    np.clip(s, 0, 255, out=s)
    c.a[:] = s.astype(np.uint8)


# ================================================================== icons (neon signs)
ICONS = {
    'star7': """
...#...
..###..
#######
.#####.
..###..
.##.##.
.#...#.""",
    'star5': """
..#..
.###.
#####
.###.
.#.#.""",
    'heart': """
.#.#.
#####
#####
.###.
..#..""",
    'diamond': """
..#..
.###.
#####
.###.
..#..""",
    'spade': """
..#..
.###.
#####
#####
..#..
.###.""",
    'club': """
..#..
.###.
#.#.#
#####
..#..
.###.""",
    'horseshoe': """
##.....##
#+.....+#
##.....##
#+.....+#
##.....##
.##...##.
.###.###.
..#####..
...###...""",
    'cherries': """
.....++
....++.
...+..+
..+...+
.##..##
###.###
###.###
.#...#.""",
    'crown': """
#...#...#
##.###.##
#########
#+##+##+#
#########""",
    'tophat': """
..#####..
..#####..
..#####..
..#####..
..+++++..
#########
.#######.""",
    'bell': """
...#...
..###..
.#####.
.#####.
.#####.
#######
...#...""",
}


@lru_cache(maxsize=None)
def icon_pts(name):
    rows = ICONS[name].strip('\n').split('\n')
    p1, p2 = [], []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                p1.append((x, y))
            elif ch == '+':
                p2.append((x, y))
    return tuple(p1), tuple(p2), len(rows[0]), len(rows)


def icon(cv, name, x, y, c1, c2=None, s=1):
    p1, p2, w, h = icon_pts(name)
    c1 = col(c1)
    for dx, dy in p1:
        cv.rect(x + dx * s, y + dy * s, s, s, c1)
    if c2 is not None:
        c2 = col(c2)
        for dx, dy in p2:
            cv.rect(x + dx * s, y + dy * s, s, s, c2)


def icon_size(name, s=1):
    _, _, w, h = icon_pts(name)
    return w * s, h * s


# ================================================================== state colours
SIL = '#281c50'       # dusk backlit silhouette tint


def T(c, st, k=0.66):
    """Facade tone: true colour when floodlit (st 2), backlit silhouette otherwise."""
    return col(c) if st == 2 else mix(c, SIL, k)


def NE(c, st):
    """Neon tube: bright when on (st >= 1), dull glass tube otherwise."""
    return col(c) if st >= 1 else mix(c, '#4b3d70', 0.74)


WIN_LIT = ('butter', 'cream', '#ffd27a', '#ffc6e8', '#c8f4ff', '#ffe9a8')


def WC(st, h):
    if st == 2:
        if h < 0.07:
            return col('#5a3f7a')
        return col(WIN_LIT[int(h * 997) % len(WIN_LIT)])
    if h < 0.10:
        return col('#ffb35a')
    return col('#3b2e66') if h < 0.55 else col('#2b2350')


def windows(cv, x0, y0, nx, ny, gx, gy, ww, wh, st, seed, wall=None, arch=False, sill=None):
    for i in range(nx):
        for j in range(ny):
            h = hash01(seed, i, j)
            x, y = x0 + i * gx, y0 + j * gy
            cc = WC(st, h)
            cv.rect(x, y, ww, wh, cc)
            if arch and wall is not None and ww >= 2 and wh >= 3:
                cv.px(x, y, wall)
                cv.px(x + ww - 1, y, wall)
            if st == 2 and h >= 0.07 and wh >= 3 and ww >= 2:
                cv.px(x + (0 if not arch else 1), y + (0 if not arch else 1), mix(cc, 'white', 0.55))
            if sill is not None:
                cv.hline(x, x + ww - 1, y + wh, sill)


def flood(cv, x0, x1, ybot, h, color, strength=0.35):
    """Floodlight from below on the opaque pixels of a layer canvas (x0..x1 inclusive)."""
    y0 = max(0, ybot - h)
    reg = cv.a[y0:ybot, x0:x1 + 1]
    m = ~np.all(reg == _KEYA, axis=2)
    t = (np.arange(y0, ybot) - y0 + 0.5) / max(1, ybot - y0)
    thr = bayer(ybot - y0, x1 - x0 + 1, x0, y0)
    q = np.clip(np.floor(t[:, None] ** 1.6 * 4 + (thr - 0.5) * 0.9) / 4, 0, 1)
    add = np.array(col(color), np.float32)[None, None, :] * (q * strength)[..., None]
    out = np.clip(reg.astype(np.float32) + add, 0, 255).astype(np.uint8)
    reg[m] = out[m]


# ================================================================== buildings (shot A strip)
# Each builder draws into the world layer for state st (0 dusk, 1 neon on, 2 fully lit) and
# returns (glows, bulbs): glow sources (x, y, r, colour, strength) and marquee bulb points.

def b_main(cv, st):
    """THE GRAND TOWER (hero): pink-white shaft, gold crown, gold star on the spire."""
    wall, sh, hi = T('#f7d6ee', st), T('#e3a6d4', st), T('white', st)
    gold, gold_d = T('gold', st), T('#c98a1e', st)
    # podium
    cv.rect(182, 112, 45, 24, wall)
    cv.rect(219, 112, 8, 24, sh)
    cv.vline(182, 112, 135, hi)
    cv.rect(181, 110, 47, 2, gold)
    cv.hline(181, 227, 112, gold_d)
    for x in (185, 190, 213, 218, 223):
        cv.rect(x, 116, 3, 7, WC(st, hash01(9, x)))
        cv.px(x, 116, wall)
        cv.px(x + 2, 116, wall)
        cv.hline(x - 1, x + 3, 123, T('#c98ab8', st))
    for x in (185, 190, 213, 218, 223):
        cv.rect(x, 126, 3, 6, WC(st, hash01(8, x)))
    # grand entrance arch + marquee canopy
    cv.rect(198, 121, 13, 15, gold)
    cv.rect(199, 122, 11, 14, T('#5a2a6a', st))
    cv.px(198, 121, wall)
    cv.px(210, 121, wall)
    if st == 2:
        cv.rect(200, 124, 9, 11, '#ffd27a')
        cv.rect(201, 125, 7, 10, '#fff0c0')
        cv.vline(204, 124, 134, '#c98a1e')
    cv.rect(194, 118, 21, 3, T('#7a2a8a', st))
    cv.hline(194, 214, 117, NE('neon_pink', st))
    # shaft
    cv.rect(190, 50, 29, 60, wall)
    cv.rect(213, 50, 6, 60, sh)
    cv.vline(191, 50, 109, hi)
    windows(cv, 193, 54, 6, 11, 4, 5, 2, 3, st, 31, sill=T('#e7b8dc', st))
    for yy in (78, 96):
        cv.hline(190, 218, yy - 1, T('#d9a0cc', st))
    # neon edge tubes + setback band
    cv.vline(189, 50, 110, NE('neon_pink', st))
    cv.vline(219, 50, 110, NE('neon_pink', st))
    cv.rect(188, 48, 33, 2, gold)
    cv.hline(188, 220, 50, NE('neon_cyan', st))
    # upper shaft
    cv.rect(194, 38, 21, 10, wall)
    cv.rect(210, 38, 5, 10, sh)
    windows(cv, 196, 40, 5, 2, 4, 4, 2, 2, st, 33)
    # crown
    cv.rect(193, 34, 23, 4, gold)
    cv.hline(193, 215, 37, gold_d)
    for i, x in enumerate((193, 198, 204, 210, 215)):
        hgt = 7 if x == 204 else 5
        cv.vline(x, 34 - hgt, 34, gold)
        cv.vline(x + (1 if x < 204 else -1), 34 - hgt + 2, 34, gold)
        cv.px(x, 34 - hgt - 1, NE('neon_gold', st))
    for x, cc in ((196, 'neon_pink'), (201, 'neon_cyan'), (207, 'neon_cyan'), (212, 'neon_pink')):
        cv.px(x, 35, NE(cc, st))
    cv.hline(193, 215, 33, NE('neon_gold', st))
    # spire + star
    cv.vline(204, 22, 26, gold)
    icon(cv, 'star7', 201, 15, NE('neon_gold', st))
    if st == 2:
        flood(cv, 181, 227, 136, 90, '#ff6ad0', 0.28)
    glows = [(204, 18, 16, 'neon_gold', 0.55), (204, 33, 20, 'neon_gold', 0.30),
             (189, 70, 9, 'neon_pink', 0.35), (219, 70, 9, 'neon_pink', 0.35),
             (189, 100, 9, 'neon_pink', 0.35), (219, 100, 9, 'neon_pink', 0.35),
             (204, 118, 14, '#ffb0e0', 0.40), (204, 140, 34, 'neon_pink', 0.32)]
    bulbs = [(x, 121) for x in range(195, 215, 2)] + [(x, 111) for x in range(183, 228, 3)]
    return glows, bulbs


def b_horseshoe(cv, st):
    """Lucky Horseshoe casino: violet block, big gold neon horseshoe on posts."""
    wall, sh, hi = T('royal', st), T('#3e2a80', st), T('royal_lt', st)
    x0, x1, top = 106, 137, 98
    cv.rect(x0, top, x1 - x0 + 1, GROUND + 2 - top, wall)
    cv.rect(x1 - 4, top, 5, GROUND + 2 - top, sh)
    cv.vline(x0, top, GROUND + 1, hi)
    cv.rect(x0 - 1, top - 2, x1 - x0 + 3, 2, T('gold', st))
    windows(cv, x0 + 3, top + 4, 6, 3, 5, 7, 3, 4, st, 41, wall=wall, arch=True)
    cv.rect(x0 + 4, 122, 24, 3, T('#2a1f60', st))
    cv.hline(x0 + 4, x0 + 27, 121, NE('neon_cyan', st))
    cv.rect(x0 + 11, 125, 10, 11, T('#241c50', st))
    if st == 2:
        cv.rect(x0 + 12, 126, 8, 10, '#ffd27a')
        cv.vline(x0 + 16, 126, 135, '#c98a1e')
    for x in (x0 + 3, x0 + 25):
        cv.rect(x, 126, 3, 6, WC(st, hash01(4, x)))
    # sign posts + horseshoe
    for x in (114, 129):
        cv.vline(x, 88, top - 3, T('steel_dk', st))
    icon(cv, 'horseshoe', 113, 77, NE('neon_gold', st), NE('neon_pink', st), s=2)
    if st == 2:
        flood(cv, x0 - 1, x1 + 1, 136, 40, '#6af0ff', 0.25)
    glows = [(122, 85, 18, 'neon_gold', 0.45), (122, 121, 12, 'neon_cyan', 0.35),
             (122, 140, 22, 'neon_gold', 0.25)]
    bulbs = [(x, 96) for x in range(x0, x1 + 1, 3)]
    return glows, bulbs


def b_turrets(cv, st):
    """Fairy-tale hotel: pink body with two conical-roof turrets, neon stars on the cones."""
    wall, sh, hi = T('bubblegum', st), T('#d0679f', st), T('cotton', st)
    roof, roof_sh = T('royal_lt', st), T('royal', st)
    cv.rect(146, 84, 32, 52, wall)
    cv.rect(172, 84, 6, 52, sh)
    cv.vline(146, 84, 135, hi)
    cv.rect(145, 82, 34, 2, T('gold', st))
    windows(cv, 149, 88, 6, 6, 5, 7, 2, 4, st, 51, wall=wall, arch=True, sill=hi)
    for tx, tw, ttop in ((141, 11, 66), (172, 10, 70)):
        cv.rect(tx, ttop, tw, 136 - ttop, wall)
        cv.rect(tx + tw - 3, ttop, 3, 136 - ttop, sh)
        cv.vline(tx, ttop, 135, hi)
        windows(cv, tx + 3, ttop + 4, 1, 6, 5, 9, 3, 4, st, tx, wall=wall, arch=True)
        cx = tx + tw // 2
        cv.tri((tx - 2, ttop), (tx + tw + 1, ttop), (cx, ttop - 17), roof)
        cv.tri((cx, ttop), (tx + tw + 1, ttop), (cx, ttop - 17), roof_sh)
        cv.hline(tx - 2, tx + tw + 1, ttop, T('gold', st))
        cv.vline(cx, ttop - 22, ttop - 17, T('gold', st))
        cv.line(tx - 2, ttop - 1, cx, ttop - 17, NE('neon_cyan', st))
        cv.line(tx + tw + 1, ttop - 1, cx, ttop - 17, NE('neon_cyan', st))
        icon(cv, 'star5', cx - 2, ttop - 27, NE('neon_gold', st))
    # balcony + marquee
    cv.rect(152, 118, 20, 2, T('#7a2a6a', st))
    cv.hline(152, 171, 117, NE('neon_pink', st))
    cv.rect(156, 124, 12, 12, T('#4a1f5a', st))
    if st == 2:
        cv.rect(157, 125, 10, 11, '#ffd9f0')
        flood(cv, 139, 182, 136, 70, '#ff8ad8', 0.22)
    glows = [(146, 41, 10, 'neon_gold', 0.4), (177, 45, 10, 'neon_gold', 0.4),
             (146, 58, 12, 'neon_cyan', 0.25), (177, 62, 12, 'neon_cyan', 0.25),
             (162, 118, 12, 'neon_pink', 0.35), (162, 140, 24, 'magic', 0.25)]
    bulbs = [(x, 121) for x in range(152, 172, 2)]
    return glows, bulbs


def b_suits(cv, st):
    """Baby-blue tower with a domed top and a vertical card-suits blade sign."""
    wall, sh, hi = T('baby_blue', st), T('#86b6e6', st), T('white', st)
    cv.rect(232, 58, 26, 78, wall)
    cv.rect(252, 58, 6, 78, sh)
    cv.vline(232, 58, 135, hi)
    # dome
    cv.ellipse(245, 58, 13, 9, T('periwinkle', st))
    cv.ellipse(242, 55, 6, 4, T('#c6ccff', st))
    cv.rect(230, 58, 30, 2, T('gold', st))
    cv.vline(245, 40, 49, T('gold', st))
    cv.circle(245, 40, 1, NE('neon_cyan', st))
    cv.line(232, 57, 245, 49, NE('neon_cyan', st))
    cv.line(258, 57, 245, 49, NE('neon_cyan', st))
    windows(cv, 236, 63, 5, 13, 4, 5, 2, 3, st, 61, sill=hi)
    # blade sign hanging on the left side
    cv.rect(226, 66, 7, 42, T('#2a2050', st))
    cv.frame(226, 66, 7, 42, NE('neon_cyan', st))
    for i, (nm, cc) in enumerate((('spade', 'neon_cyan'), ('heart', 'neon_pink'),
                                  ('diamond', 'neon_pink'), ('club', 'neon_cyan'))):
        icon(cv, nm, 227, 68 + i * 10, NE(cc, st))
    cv.rect(236, 124, 14, 12, T('#3a5a9a', st))
    if st == 2:
        cv.rect(237, 125, 12, 11, '#d8f6ff')
        flood(cv, 226, 259, 136, 80, '#6af0ff', 0.22)
    glows = [(229, 76, 12, 'neon_cyan', 0.4), (229, 98, 12, 'neon_pink', 0.4),
             (245, 50, 14, 'neon_cyan', 0.35), (245, 140, 24, 'neon_cyan', 0.28)]
    bulbs = [(x, 122) for x in range(234, 256, 2)]
    return glows, bulbs


def b_cherries(cv, st):
    """Cherry-slots casino: mint facade, a giant slot-reel sign with cherries on the roof."""
    wall, sh, hi = T('#8fe0c8', st), T('#5fb8a8', st), T('mint_cream', st)
    cv.rect(264, 88, 32, 48, wall)
    cv.rect(290, 88, 6, 48, sh)
    cv.vline(264, 88, 135, hi)
    cv.rect(263, 86, 34, 2, T('gold', st))
    windows(cv, 267, 92, 6, 3, 5, 7, 3, 4, st, 71, wall=wall, arch=True)
    # roof sign: a slot machine with three reels
    cv.round_rect(266, 66, 28, 18, T('#c0303a', st), r=2)
    cv.rect(268, 69, 24, 11, T('#3a1028', st))
    for i in range(3):
        cv.rect(269 + i * 8, 70, 6, 9, T('cream', st))
    cv.frame(266, 66, 28, 18, NE('neon_gold', st))
    cv.vline(296, 70, 80, T('steel', st))
    cv.circle(296, 69, 1, NE('neon_pink', st))
    cv.vline(272, 84, 86, T('steel_dk', st))
    cv.vline(288, 84, 86, T('steel_dk', st))
    cv.rect(268, 120, 24, 3, T('#c0303a', st))
    cv.hline(268, 291, 119, NE('neon_pink', st))
    cv.rect(274, 124, 12, 12, T('#1f4a44', st))
    if st == 2:
        cv.rect(275, 125, 10, 11, '#fff0c0')
        flood(cv, 263, 297, 136, 50, '#ff4fd8', 0.2)
    glows = [(280, 74, 18, 'neon_gold', 0.35), (280, 74, 10, 'red', 0.3),
             (280, 120, 12, 'neon_pink', 0.35), (280, 140, 22, 'neon_pink', 0.25)]
    bulbs = [(x, 85) for x in range(265, 296, 2)]
    return glows, bulbs


def b_dice(cv, st):
    """Dice casino: white cube with pink trims and a giant die on the roof."""
    wall, sh, hi = T('#f4f0ff', st), T('#c8c0e8', st), T('white', st)
    cv.rect(301, 82, 32, 54, wall)
    cv.rect(327, 82, 6, 54, sh)
    cv.vline(301, 82, 135, hi)
    cv.rect(300, 80, 34, 2, T('hot_pink', st))
    for yy in (94, 108):
        cv.hline(301, 332, yy, T('#e0a0d0', st))
    windows(cv, 304, 84, 7, 7, 4, 7, 2, 4, st, 81, sill=T('#e0d8f8', st))
    # roof die (front face + top + side)
    cv.rect(309, 62, 16, 16, T('white', st))
    cv.poly([(309, 62), (313, 58), (329, 58), (325, 62)], T('#e8e0ff', st))
    cv.poly([(325, 62), (329, 58), (329, 74), (325, 78)], T('#b8b0d8', st))
    cv.frame(309, 62, 16, 16, NE('neon_pink', st))
    cv.vline(312, 78, 79, T('steel_dk', st))
    cv.vline(322, 78, 79, T('steel_dk', st))
    cv.rect(305, 122, 24, 2, T('hot_pink', st))
    cv.hline(305, 328, 121, NE('neon_cyan', st))
    cv.rect(311, 124, 12, 12, T('#5a4a8a', st))
    if st == 2:
        cv.rect(312, 125, 10, 11, '#ffe9a8')
        flood(cv, 300, 334, 136, 76, '#c08aff', 0.22)
    glows = [(317, 70, 16, 'neon_pink', 0.4), (317, 122, 12, 'neon_cyan', 0.35),
             (317, 140, 22, 'magic', 0.25)]
    bulbs = [(x, 81) for x in range(301, 334, 3)]
    return glows, bulbs


def b_theatre(cv, st):
    """Top-hat theatre (the show palace): plum facade, big marquee, neon top hat sign."""
    wall, sh, hi = T('#9a3a7a', st), T('plum', st), T('#c860a0', st)
    cv.rect(338, 92, 36, 44, wall)
    cv.rect(367, 92, 7, 44, sh)
    cv.vline(338, 92, 135, hi)
    cv.rect(337, 90, 38, 2, T('gold', st))
    for x in (341, 349, 357, 365):
        cv.rect(x, 94, 2, 20, T('#e8c060', st))
    windows(cv, 344, 96, 3, 2, 8, 9, 3, 6, st, 91, wall=wall, arch=True)
    # marquee box
    cv.rect(340, 114, 32, 8, T('cream', st))
    cv.rect(342, 116, 28, 4, T('#3a1030', st))
    cv.rect(346, 124, 20, 12, T('#3a1030', st))
    cv.rect(349, 124, 2, 12, T('#c0303a', st))
    cv.rect(361, 124, 2, 12, T('#c0303a', st))
    if st == 2:
        cv.rect(351, 125, 10, 11, '#ffe0a0')
    # top hat sign
    cv.vline(350, 84, 89, T('steel_dk', st))
    cv.vline(362, 84, 89, T('steel_dk', st))
    icon(cv, 'tophat', 347, 70, NE('#c8f8ff', st), NE('neon_pink', st), s=2)
    if st == 2:
        flood(cv, 337, 375, 136, 50, '#ffd05a', 0.22)
    glows = [(356, 77, 16, 'neon_cyan', 0.4), (356, 118, 16, 'neon_gold', 0.35),
             (356, 140, 22, 'neon_gold', 0.25)]
    bulbs = ([(x, 113) for x in range(340, 373, 2)] + [(x, 122) for x in range(340, 373, 2)]
             + [(339, y) for y in range(115, 122, 2)] + [(372, y) for y in range(115, 122, 2)])
    return glows, bulbs


def b_gold(cv, st):
    """Gaudy gold resort tower (columns, gold star) + the carnival Ferris wheel behind."""
    # Ferris wheel (behind, right)
    cx, cy, r = 446, 92, 30
    rim = NE('neon_pink', st) if st else T('#8a6aa0', st)
    cv.circle(cx, cy, r, rim, fill=False)
    cv.circle(cx, cy, r - 3, NE('neon_cyan', st) if st else T('#6a5a90', st), fill=False)
    for k in range(12):
        a = k * math.pi / 6
        cv.line(cx, cy, cx + int(round(math.cos(a) * (r - 1))), cy + int(round(math.sin(a) * (r - 1))),
                T('#b8a0d0', st, 0.7))
    for k in range(12):
        a = k * math.pi / 6 + 0.26
        gx, gy = cx + int(round(math.cos(a) * r)), cy + int(round(math.sin(a) * r))
        cc = ('neon_pink', 'neon_cyan', 'neon_gold', 'lavender')[k % 4]
        cv.rect(gx - 1, gy + 1, 3, 3, NE(cc, st) if st == 2 else T(cc, st))
    cv.line(cx, cy, cx - 16, GROUND + 1, T('#9a88b8', st))
    cv.line(cx, cy, cx + 16, GROUND + 1, T('#9a88b8', st))
    cv.circle(cx, cy, 2, T('gold', st))
    # gold tower
    wall, sh, hi = T('#ffe6a0', st), T('#e0b050', st), T('#fff6d8', st)
    cv.rect(382, 110, 40, 26, wall)
    cv.rect(414, 110, 8, 26, sh)
    cv.rect(388, 44, 28, 66, wall)
    cv.rect(410, 44, 6, 66, sh)
    cv.vline(388, 44, 135, hi)
    for x in range(383, 422, 5):
        cv.vline(x, 114, 135, T('#fff6d8', st))
        cv.vline(x + 1, 114, 135, T('#d8a040', st))
    cv.rect(381, 108, 42, 3, T('gold', st))
    windows(cv, 391, 48, 6, 11, 4, 5, 2, 3, st, 101, sill=hi)
    cv.rect(386, 40, 32, 4, T('gold', st))
    cv.hline(386, 417, 39, NE('neon_gold', st))
    cv.rect(392, 32, 20, 8, wall)
    cv.rect(406, 32, 6, 8, sh)
    cv.tri((390, 32), (414, 32), (402, 20), T('gold', st))
    cv.tri((402, 32), (414, 32), (402, 20), T('#c98a1e', st))
    icon(cv, 'star7', 399, 11, NE('neon_gold', st))
    cv.vline(402, 18, 19, T('gold', st))
    cv.vline(387, 44, 108, NE('neon_gold', st))
    cv.vline(416, 44, 108, NE('neon_gold', st))
    cv.rect(396, 122, 12, 14, T('#7a4a1a', st))
    if st == 2:
        cv.rect(397, 123, 10, 13, '#fff0c0')
        flood(cv, 380, 423, 136, 100, '#ffc04a', 0.18)
    glows = [(402, 14, 16, 'neon_gold', 0.5), (402, 40, 14, 'neon_gold', 0.3),
             (387, 80, 9, 'neon_gold', 0.25), (416, 80, 9, 'neon_gold', 0.25),
             (446, 92, 36, 'neon_pink', 0.18), (402, 140, 26, 'neon_gold', 0.28),
             (446, 140, 20, 'neon_cyan', 0.2)]
    bulbs = [(x, 109) for x in range(382, 423, 3)]
    return glows, bulbs


# group: (builder, x0, x1, top y, t_on)
GROUPS = (
    (b_main, 180, 228, 14, 144),
    (b_horseshoe, 104, 139, 76, 153),
    (b_turrets, 139, 180, 38, 162),
    (b_suits, 225, 261, 38, 171),
    (b_cherries, 262, 299, 64, 180),
    (b_dice, 299, 335, 56, 189),
    (b_theatre, 336, 376, 68, 198),
    (b_gold, 376, WA, 10, 207),
)


@lru_cache(maxsize=None)
def city_layers():
    """(rgba[3], glow int16 maps per group, bulbs per group, column->group index)."""
    out = []
    meta = None
    for st in (0, 1, 2):
        cv = layer(WA)
        ms = []
        for g in GROUPS:
            ms.append(g[0](cv, st))
        if st == 0:
            # backlit rim light on top edges (afterglow behind the city)
            a = cv.a
            op = ~np.all(a == _KEYA, axis=2)
            top = op.copy()
            top[1:] &= ~op[:-1]
            rim = np.array(col('#ff9a9a'), np.float32)
            a[top] = np.clip(a[top] * 0.45 + rim * 0.55, 0, 255).astype(np.uint8)
            right = op.copy()
            right[:, :-1] &= ~op[:, 1:]
            a[right & ~top] = np.clip(a[right & ~top] * 0.7 + rim * 0.3, 0, 255).astype(np.uint8)
        if st == 2:
            meta = ms
        out.append(to_rgba(cv))
    glows = []
    for (gl, _), g in zip(meta, GROUPS):
        gc = Canvas(WA, H, (0, 0, 0))
        for (x, y, r, cc, s) in gl:
            fx.glow(gc, x, y, r, cc, s, rings=4)
        glows.append(gc.a.astype(np.int16))
    bulbs = [np.array(b, np.int32) if b else np.zeros((0, 2), np.int32) for _, b in meta]
    gid = np.full(WA, -1, np.int32)
    for i, g in enumerate(GROUPS):
        gid[g[1]:g[2]] = i
    return out, glows, bulbs, gid


def group_state(f, i):
    """(state, wipe_row, glow_k) of group i at frame f during the cascade."""
    u = f - GROUPS[i][4]
    if u < 0:
        return 0, None, 0.0
    if u == 0:
        return 1, None, 1.4
    if u == 1:
        return 0, None, 0.0
    if u < 9:
        top = GROUPS[i][3]
        row = int(round(GROUND + 2 - (GROUND + 2 - top) * (u - 1) / 8.0))
        return 1, row, 0.8
    return 2, None, 1.0


def draw_city(c, f, sx, oy=0, night=False):
    """Composite the strip for camera world x sx (screen x = world x - sx)."""
    layers, glows, bulbs, gid = city_layers()
    x0 = max(0, sx)
    x1 = min(WA, sx + W)
    if x1 <= x0:
        return
    states = [(2, None, 1.0) if night else group_state(f, i) for i in range(len(GROUPS))]
    cols_g = gid[x0:x1]
    st_col = np.zeros(x1 - x0, np.int32)
    row_col = np.full(x1 - x0, 9999, np.int32)
    for i, (st, row, k) in enumerate(states):
        m = cols_g == i
        st_col[m] = st
        if row is not None:
            row_col[m] = row
    L0, L1, L2 = (l[:, x0:x1] for l in layers)
    yy = np.arange(H)[:, None]
    s2 = (st_col[None, :] == 2) | ((st_col[None, :] == 1) & (yy >= row_col[None, :]))
    s1 = (st_col[None, :] >= 1) & ~s2
    comp = np.where(s2[..., None], L2, np.where(s1[..., None], L1, L0))
    c.blit(comp, x0 - sx, oy)
    # neon glow halos (additive, include reflections on the cloud bank)
    acc = None
    for i, (st, row, k) in enumerate(states):
        if k <= 0:
            continue
        g = glows[i][:, x0:x1]
        g = g if k == 1.0 else (g * k).astype(np.int16)
        acc = g.copy() if acc is None else acc + g
    if acc is not None:
        full = np.zeros((H, W, 3), np.int16)
        ys0, ys1 = max(0, oy), min(H, H + oy)
        full[ys0:ys1, x0 - sx:x1 - sx] = acc[ys0 - oy:ys1 - oy]
        add_light(c, full)
    # chasing marquee bulbs + per-sign animation
    for i, (st, row, k) in enumerate(states):
        if st < 1:
            continue
        b = bulbs[i]
        if len(b):
            ph = (np.arange(len(b)) + f // 3) % 3
            for (bx, by), p in zip(b, ph):
                c.px(bx - sx, by + oy, 'white' if p == 0 else ('neon_gold' if p == 1 else '#b0702a'))
        sign_anim(c, f, i, st, sx, oy)


def sign_anim(c, f, i, st, sx, oy):
    """Per-frame bits on lit signs: blinks on the beat, rolling die, spinning slot reels."""
    bt = f // FRAMES_PER_BEAT
    fb = f % FRAMES_PER_BEAT
    if i == 0:          # grand tower star twinkle
        if fb < 4:
            fx.sparkle(c, 204 - sx + 3, 15 + oy, f, 'white', period=FRAMES_PER_BEAT, size=3)
    elif i == 1:        # horseshoe: pink nails chase
        if fb < 3:
            icon(c, 'horseshoe', 113 - sx, 77 + oy, 'butter', None, s=2)
    elif i == 4:        # slot reels spin, land on cherries on every beat
        for r in range(3):
            x = 269 + r * 8 - sx
            spin = fb < 6 + r * 3
            if spin:
                k = (f + r * 2) % 3
                c.rect(x, 70 + oy, 6, 9, 'cream')
                c.hline(x, x + 5, 71 + k * 3 + oy, '#c0303a')
                c.hline(x, x + 5, 72 + k * 3 + oy, '#ffb0b0')
            else:
                c.rect(x, 70 + oy, 6, 9, 'cream')
                c.circle(x + 1, 76 + oy, 1, '#e02040')
                c.circle(x + 4, 76 + oy, 1, '#e02040')
                c.line(x + 1, 74 + oy, x + 3, 71 + oy, 'green')
                c.line(x + 4, 74 + oy, x + 3, 71 + oy, 'green')
    elif i == 5:        # die rolls a new face on every beat
        n = 1 + int(hash01(bt, 5) * 6)
        x, y = 309 - sx, 62 + oy
        pips = {1: [(1, 1)], 2: [(0, 0), (2, 2)], 3: [(0, 0), (1, 1), (2, 2)],
                4: [(0, 0), (2, 0), (0, 2), (2, 2)], 5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
                6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)]}[n]
        cc = 'neon_pink' if fb > 1 else 'white'
        for px_, py_ in pips:
            c.rect(x + 3 + px_ * 4, y + 3 + py_ * 4, 2, 2, cc)
    elif i == 6:        # theatre marquee icons
        for k, nm in enumerate(('star5', 'heart', 'star5')):
            on = (bt + k) % 3 != 0 or fb > 9
            icon(c, nm, 345 + k * 9 - sx, 115 + oy, ('neon_gold', 'neon_pink', 'neon_gold')[k] if on else '#6a3050')
    elif i == 7:        # gold star sparkle + gondolas twinkle
        if (fb + 9) % FRAMES_PER_BEAT < 4:
            fx.sparkle(c, 402 - sx, 14 + oy, f + 9, 'white', period=FRAMES_PER_BEAT, size=3)


# ================================================================== sky, sea, far coast
@lru_cache(maxsize=None)
def backdrop(tq):
    """Sky + sea for night factor tq/12 (0 dusk afterglow .. 1 deep violet night)."""
    t = tq / 12.0
    c = Canvas()
    dusk = ['#2a1d58', '#3e2a72', '#663a86', '#a24b86', '#e06a8a', '#ff9a7a', '#ffc08a']
    night = ['#100c26', '#17113a', '#1f1648', '#2a1a4a', '#3a1e5c', '#4c2468', '#5e2a6e']
    fx.sky_blend(c, dusk, night, t, y0=0, y1=HORIZ)
    if t < 1:
        fx.glow(c, SUNX, HORIZ, 110, '#ff8a5a', 0.30 * (1 - t), rings=6)
        fx.glow(c, SUNX, HORIZ, 46, '#ffd08a', 0.35 * (1 - t), rings=4)
    # sea
    sd = ['#6a3a7a', '#4a2c6a', '#33245a', '#241c48', '#1c1838']
    sn = ['#3a1e58', '#2a1848', '#1e1440', '#161034', '#100c28']
    fx.sky_blend(c, sd, sn, t, y0=HORIZ, y1=H)
    if t < 1:
        fx.light_rect(c, SUNX - 26, HORIZ, 52, 60, '#ff9a6a', 0.35 * (1 - t), fade='down')
    c.hline(0, W - 1, HORIZ, mix('#ffb08a', '#6a3a8a', t))
    return c


def sea_glitter(c, f, t):
    """Twinkling horizontal dashes on the sea (afterglow column at dusk, scattered neon at night)."""
    for i in range(46):
        h1, h2, h3 = hash01(i, 1, 77), hash01(i, 2, 77), hash01(i, 3, 77)
        y = HORIZ + 2 + int(h2 ** 1.6 * 66)
        spread = 10 + (y - HORIZ) * 0.9
        x = int(SUNX + (h1 - 0.5) * 2 * spread) if i % 3 else int(h1 * W)
        on = math.sin(f * 0.21 + h3 * 6.28) > 0.2
        if not on:
            continue
        ln = 1 + int(h3 * 3)
        cc = mix('#ffd0a0', '#ff7ad8', t) if i % 3 else mix('#c89ad8', '#4ff6ff', t)
        c.hline(x, x + ln, y, cc)


@lru_cache(maxsize=None)
def far_layer(night):
    """Coast hills with the APPLEWOOD sign + star, the little studio town at the shore."""
    cv = layer(FW)
    ridge2 = [(0, 102), (24, 96), (60, 98), (100, 104), (140, 108), (180, 112), (180, 116), (0, 116)]
    cv.poly(ridge2, '#3a2a62' if night else '#7a5a9a')
    ridge = [(0, 88), (14, 84), (34, 78), (52, 74), (68, 75), (84, 80), (100, 90), (116, 102),
             (134, 112), (150, 118), (160, 122), (0, 122)]
    hill, hill_lt = ('#251c48', '#33285e') if night else ('#4a3470', '#7a4a86')
    cv.poly(ridge, hill)
    # lit right-facing slopes (afterglow from the right) / soft moon rim at night
    for i in range(len(ridge) - 3):
        (xa, ya), (xb, yb) = ridge[i], ridge[i + 1]
        if yb > ya:
            cv.line(xa, ya, xb, yb, hill_lt)
            cv.line(xa, ya + 1, xb, yb + 1, mix(hill_lt, hill, 0.5))
    # scrub texture
    for i in range(60):
        x = int(hash01(i, 3, 5) * 150)
        y0 = 80 + int(hash01(i, 4, 5) * 40)
        if cv.a[y0, x].tolist() == list(col(hill)):
            cv.px(x, y0, '#1c1640' if night else '#3e2a62')
    # shore + beach
    cv.rect(0, 122, 162, 2, '#c08a7a' if not night else '#4a3a5a')
    cv.rect(0, 124, 158, 1, '#8a5a7a' if not night else '#2e2448')
    # studio town (sound stages with curved roofs, water tower, palms)
    for k, (x, w, h) in enumerate(((4, 12, 6), (18, 9, 8), (30, 14, 5), (48, 10, 7), (62, 13, 6),
                                   (79, 8, 9), (90, 12, 5), (106, 10, 6), (120, 9, 4))):
        y = 122 - h
        body = '#3a2c5e' if night else '#5e4580'
        cv.rect(x, y, w, h, body)
        cv.hline(x + 1, x + w - 2, y - 1, body)
        cv.hline(x + 1, x + w - 2, y, '#4c3a72' if night else '#a87a9a')
        for j in range(w // 3):
            hh = hash01(k, j, 9)
            if night and hh < 0.75:
                cv.px(x + 1 + j * 3, y + 2, ('#ffd27a', '#ffc6e8', '#c8f4ff')[int(hh * 30) % 3])
            elif not night and hh < 0.2:
                cv.px(x + 1 + j * 3, y + 2, '#ffb35a')
    cv.vline(44, 112, 121, '#3a2c5e' if night else '#5e4580')
    cv.rect(42, 109, 5, 4, '#4c3a72' if night else '#7a5a8a')
    for x in (26, 74, 100):
        cv.vline(x, 114, 121, '#2a2048' if night else '#4a3470')
        cv.hline(x - 2, x + 2, 113, '#2a2048' if night else '#4a3470')
        cv.px(x - 3, 114, '#2a2048' if night else '#4a3470')
        cv.px(x + 3, 114, '#2a2048' if night else '#4a3470')
    # the APPLEWOOD sign: white letters stepped along the slope + a gold star at the end
    lt = 'white' if night else '#ffe6ea'
    sh = '#9a8ab8' if night else '#a87a9a'
    xs = 16
    for k, ch in enumerate('APPLEWOOD'):
        x = xs + k * 7
        yr = ridge_y(ridge, x + 3) + 4
        text(cv, x + 1, yr + 1, ch, sh, font='normal')
        text(cv, x, yr, ch, lt, font='normal')
    sx, sy = xs + 9 * 7 + 2, ridge_y(ridge, xs + 66) - 3
    icon(cv, 'star7', sx, sy, '#c98a1e')
    icon(cv, 'star7', sx - 1, sy - 1, 'gold')
    cv.px(sx + 2, sy + 1, 'butter')
    return to_rgba(cv)


def ridge_y(ridge, x):
    for (xa, ya), (xb, yb) in zip(ridge, ridge[1:]):
        if xa <= x <= xb and xb > xa:
            return int(round(ya + (yb - ya) * (x - xa) / (xb - xa)))
    return ridge[-1][1]


def _sh(m, dx, dy):
    """out[y, x] = m[y + dy, x + dx] (False outside)."""
    h, w = m.shape
    out = np.zeros_like(m)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    out[ys0:ys1, xs0:xs1] = m[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def circ_mask(shape, cx, cy, r):
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 0.5) ** 2


def cloud_mass(cv, circles, mid, hi, lo, rim, dk, front=()):
    """Union-shaded cumulus: top-left highlight, rim, under-lit bottom band; `front` puffs get
    an inner top arc + a soft shade line under them for cauliflower definition."""
    shape = (cv.h, cv.w)
    M = np.zeros(shape, bool)
    for cx, cy, r in circles:
        M |= circ_mask(shape, cx, cy, r)
    a = cv.a
    thr = bayer(cv.h, cv.w)
    a[M] = col(mid)
    a[M & ~_sh(M, 0, 6)] = mix(mid, lo, 0.45)
    a[M & ~_sh(M, 0, 3)] = col(lo)
    a[M & ~_sh(M, -1, -4)] = mix(mid, hi, 0.5)
    a[M & ~_sh(M, -1, -2)] = col(hi)
    a[M & ~_sh(M, 0, -1)] = col(rim)
    for cx, cy, r in front:
        cm = circ_mask(shape, cx, cy, r)
        under = M & ~cm & _sh(cm, 0, -1) & _sh(M, 0, 2)
        a[under] = col(dk)
        arc = cm & ~_sh(cm, -1, -2) & _sh(M, 0, -3)
        a[arc] = col(hi)
    return M


@lru_cache(maxsize=None)
def back_clouds(night):
    cv = layer(BW)
    mid, hi, lo, rim, dk = (('#3a2a66', '#56448c', '#a8409a', '#6e5aa8', '#30225a') if night else
                            ('#8e62a6', '#d2a2d4', '#ff8aa8', '#ffd0dc', '#7a5296'))
    circ, front = [], []
    for i in range(22):
        x = 140 + i * 13 + int(hash01(i, 1, 21) * 8)
        r = 8 + int(hash01(i, 2, 21) * 9)
        y = 124 - int(hash01(i, 3, 21) * 10)
        circ.append((x, y, r))
        if i % 3 == 1:
            front.append((x + 4, y + 6, max(5, r - 4)))
    circ += [(x, 132, 7) for x in range(140, BW, 8)]
    cloud_mass(cv, circ + front, mid, hi, lo, rim, dk, front)
    return to_rgba(cv)


@lru_cache(maxsize=None)
def front_bank(night):
    """The cloud bank the city stands on (world coords of the city layer)."""
    cv = layer(WA)
    mid, hi, lo, rim, dk = (('#523a86', '#7e62bc', '#d84ab6', '#a88ee0', '#3e2c6e') if night else
                            ('#a874b6', '#e6b8e0', '#ff9fd0', '#ffe2f0', '#8e5ea6'))
    circ, front = [], []
    # left cliff
    circ += [(106, 146, 12), (100, 138, 7), (110, 156, 9), (97, 150, 6)]
    front += [(108, 149, 9)]
    for i in range(34):
        x = 112 + i * 11 + int(hash01(i, 4, 31) * 6)
        r = 5 + int(hash01(i, 5, 31) * 6)
        circ.append((x, 137 + int(hash01(i, 6, 31) * 3), r))
        if i % 2 == 0:
            front.append((x + 5, 143 + int(hash01(i, 7, 31) * 3), r - 1))
    for i in range(32):
        x = 112 + i * 12 + int(hash01(i, 1, 31) * 6)
        r = 5 + int(hash01(i, 2, 31) * 4)
        circ.append((x, 150 + int(hash01(i, 3, 31) * 4), r))
    cloud_mass(cv, circ + front, mid, hi, lo, rim, dk, front)
    return to_rgba(cv)


@lru_cache(maxsize=None)
def fg_puffs(night):
    cv = layer(560, 40)
    mid, hi, lo, rim, dk = (('#3e2c6c', '#6a50a0', '#b03aa0', '#8a72c8', '#2e2058') if night else
                            ('#b882bc', '#f2c8e6', '#ff96c4', '#fff0f6', '#9a68ae'))
    circ, front = [], []
    for i in range(7):
        x = 20 + i * 80 + int(hash01(i, 1, 41) * 20)
        n = 2 + int(hash01(i, 2, 41) * 3)
        for j in range(n):
            r = 5 + int(hash01(i, j, 42) * 6)
            circ.append((x + j * 8, 34 - r // 2 + int(hash01(i, j, 43) * 3), r))
        front.append((x + 6, 38, 6))
    cloud_mass(cv, circ + front, mid, hi, lo, rim, dk, front)
    return to_rgba(cv)


# ================================================================== shot A: the strip at dusk -> neon
CAM_A = [(72, 0), (118, 8), (214, 150)]


def night_t(f):
    """Night factor: dusk 0 until the neon hit, then the sky darkens through bar 3."""
    return smooth(seg(f, 140, 212))


def moon(c, x, y, t):
    """Thin waxing crescent (lit on the lower right), 11 px."""
    fx.glow(c, x, y, 12, '#c8a0ff', 0.18 + 0.1 * t, rings=3)
    reg = c.a
    yy, xx = np.mgrid[y - 6:y + 7, x - 6:x + 7]
    a = (xx - x) ** 2 + (yy - y) ** 2 <= 30
    b = (xx - x + 2) ** 2 + (yy - y + 2) ** 2 <= 26
    m = a & ~b
    ys, xs = yy[m], xx[m]
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    reg[ys[ok], xs[ok]] = col('#fff4d6')


STREAKS = ((0, 30, 50), (96, 54, 64), (204, 24, 44), (282, 50, 58), (384, 36, 50), (474, 60, 46))


def streaks(c, f, cam, oy, t):
    """Thin stratus streaks: lit pink/gold from below at dusk, faint violet at night."""
    top = mix('#ffb0a8', '#3a2a66', t)
    bot = mix('#ff7a9a', '#4a2a70', t)
    core = mix('#c85a8a', '#2a1e52', t)
    for x0, y, w in STREAKS:
        x = int(x0 - cam * 0.12 + f * 0.05) % 560 - 120
        yy = y + oy
        c.hline(x + 4, x + w - 6, yy, top)
        c.hline(x, x + w, yy + 1, core)
        c.hline(x + 3, x + w - 2, yy + 2, bot)
        c.hline(x + w // 3, x + w - 10, yy - 1, top)


def searchlights(c, f, ox, t):
    """Two premiere searchlights sweeping up from the studio town (night only)."""
    if t <= 0.05:
        return
    for k, (bx, ph) in enumerate(((36, 0.0), (86, 2.1))):
        a = -math.pi / 2 + 0.55 * math.sin(f * 0.045 + ph)
        x0, y0 = bx + ox, 114
        L = 120
        ex, ey = x0 + math.cos(a) * L, y0 + math.sin(a) * L
        w = 9
        fx.light_beam(c, ((ex - w, ey), (ex + w, ey)), ((x0, y0), (x0 + 1, y0)),
                      '#a8b0ff', 0.12 * t)


def shot_strip(f, cam, oy=0, night=None, sky_t=None, balloon=True, sky_fx=None):
    t = night_t(f) if sky_t is None else sky_t
    nit = night if night is not None else False
    bd = backdrop(int(round(t * 12)))
    if oy:
        c = Canvas(W, H, bd.a[0, 0])
        c.blit(bd, 0, oy)
    else:
        c = bd.copy()
    nstar = int(18 + 70 * t)
    fx.stars(c, f, seed=16, n=nstar, region=(0, oy - 50, W, 146))
    moon(c, 292, 40 + oy // 3, t)
    streaks(c, f, cam, oy, t)
    sea_glitter(c, f, t)
    if sky_fx is not None:
        sky_fx(c)
    # far coast + Applewood
    fxo = -int(cam * PAR_FAR)
    xpaste(c, far_layer(False), far_layer(True), fxo, oy, t)
    searchlights(c, f, fxo, t)
    # Applewood sign floodlight at night
    if t > 0.3:
        fx.glow(c, fxo + 48, 82 + oy, 34, '#fff0d0', 0.10 * t, rings=4)
    # tiny sailboat on the sea far below (drifts left, lantern blinking)
    bx = 70 - int(f * 0.08) + fxo // 2
    by = 136 + oy
    c.hline(bx - 6, bx + 8, by + 2, mix('#c89ad8', '#6a5aa0', t))
    c.hline(bx - 4, bx + 4, by, '#8a4a5a')
    c.hline(bx - 3, bx + 3, by + 1, '#5a2e44')
    c.vline(bx, by - 8, by - 1, '#d8b090')
    c.tri((bx + 1, by - 8), (bx + 1, by - 2), (bx + 5, by - 2), mix('#ffd8e0', '#b8a0d8', t))
    c.tri((bx - 1, by - 6), (bx - 1, by - 2), (bx - 4, by - 2), mix('#ffb8c8', '#9a80c0', t))
    c.px(bx + 4, by - 1, 'butter' if (f // 12) % 2 else '#ffb35a')
    # back clouds
    bxo = -int(cam * PAR_BACK) - 20
    xpaste(c, back_clouds(False), back_clouds(True), bxo, oy, t)
    # balloon drifting among the towers (behind the city)
    if balloon:
        draw_balloon(c, 232 + int((f - 72) * 0.32) - int(cam * 0.72), 48 + oy, f, size='s',
                     flame=(f // 24) % 3 == 0)
    # the city
    draw_city(c, f, cam, oy, night=nit)
    xpaste(c, front_bank(False), front_bank(True), -cam, oy, t)
    # re-add neon reflections on the bank (glow maps were added before the bank was drawn)
    bank_glow(c, f, cam, oy, nit)
    # foreground puffs drift
    fgo = -int(cam * PAR_FG) + int(f * 0.12) - 120
    xpaste(c, fg_puffs(False), fg_puffs(True), fgo, 142 + oy, t)
    return c


@lru_cache(maxsize=None)
def bank_glow_maps():
    """Per-group neon light pooled on the cloud bank (the bank is drawn over the halos)."""
    _, _, _, _ = city_layers()
    out = []
    bank = front_bank(True)
    mask = (bank[..., 3] > 0)
    for (b, x0, x1, top, t_on) in GROUPS:
        gc = Canvas(WA, H, (0, 0, 0))
        cx = (x0 + x1) // 2
        cc = {b_main: 'neon_pink', b_horseshoe: 'neon_gold', b_turrets: 'magic', b_suits: 'neon_cyan',
              b_cherries: 'neon_pink', b_dice: 'lavender', b_theatre: 'neon_gold', b_gold: 'neon_gold'}[b]
        fx.glow(gc, cx, 140, max(18, (x1 - x0) // 2 + 6), cc, 0.42, rings=4)
        fx.glow(gc, cx, 138, 10, 'white', 0.18, rings=2)
        a = gc.a.astype(np.int16)
        a[~mask] = 0
        out.append(a)
    return out


def bank_glow(c, f, sx, oy, night):
    maps = bank_glow_maps()
    x0, x1 = max(0, sx), min(WA, sx + W)
    acc = None
    for i in range(len(GROUPS)):
        st, row, k = (2, None, 1.0) if night else group_state(f, i)
        if k <= 0:
            continue
        g = maps[i][:, x0:x1]
        g = g if k == 1.0 else (g * k).astype(np.int16)
        acc = g.copy() if acc is None else acc + g
    if acc is None:
        return
    full = np.zeros((H, W, 3), np.int16)
    ys0, ys1 = max(0, oy), min(H, H + oy)
    full[ys0:ys1, x0 - sx:x1 - sx] = acc[ys0 - oy:ys1 - oy]
    add_light(c, full)


def render_a(f):
    cam = eased_cam(f, CAM_A)
    c = shot_strip(f, cam)
    # callouts
    callout(c, 220 - cam, 36, 'LAS PEGASUS', f, 90, 143, side='right', length=10)
    callout(c, 84 - int(cam * PAR_FAR), 80, 'APPLEWOOD', f, 108, 170, side='up', length=10)
    sx = 317 - cam
    callout(c, sx, 80, 'THE STRIP', f, 171, 209, side='up', length=12)
    atlas_card(c, f, 100, 196, 'RESORT CITY', ['NEON ON THE CLOUDS', 'SHOWS ALL NIGHT'])
    return c


# ================================================================== shot B: fountain plaza
PW = 420
CAM_B = [(216, 0), (287, 60)]
PAR_MID, PAR_FAR_B, PAR_FG_B = 0.85, 0.4, 1.25
RIM_Y = 136           # fountain rim (jet nozzles)
FLOOR_B = 158         # plaza floor line in front of the basin
JET_COLS = ('neon_cyan', 'magic', 'neon_gold', 'lavender')


@lru_cache(maxsize=None)
def far_resort():
    """Distant towers behind the plaza: dark violet with lit window dots and neon tops."""
    cv = layer(400)
    x, i = 0, 0
    while x < 400:
        w = 12 + int(hash01(i, 1, 51) * 16)
        h = 36 + int(hash01(i, 2, 51) * 58)
        top = 130 - h
        body = mix('#2c2058', '#3c2c6c', hash01(i, 3, 51))
        cv.rect(x, top, w, 150 - top, body)
        cv.rect(x + w - 3, top, 3, 150 - top, shade(body, 0.82))
        for yy in range(top + 3, 140, 4):
            for xx in range(x + 2, x + w - 3, 3):
                h2 = hash01(i, xx, yy)
                if h2 < 0.5:
                    cv.px(xx, yy, mix(WIN_LIT[int(h2 * 97) % 6], body, 0.35))
        nc = ('neon_pink', 'neon_cyan', 'neon_gold', 'magic')[i % 4]
        cv.hline(x, x + w - 1, top, nc)
        if hash01(i, 4, 51) < 0.45:
            cv.vline(x + w // 2, top - 8, top - 1, '#5a4a8a')
            cv.px(x + w // 2, top - 9, nc)
        elif hash01(i, 6, 51) < 0.5:
            cv.round_rect(x + 2, top - 7, w - 4, 6, '#241a48', r=1)
            nm = ('star5', 'heart', 'diamond', 'bell')[i % 4]
            icon(cv, nm, x + w // 2 - 2, top - 7 + (0 if nm != 'bell' else -1), nc)
        x += w + 1 + int(hash01(i, 5, 51) * 5)
        i += 1
    return to_rgba(cv)


# ---- roller coaster path (mid-layer coordinates)
def _coaster_pts():
    pts = []
    def seg_line(a, b, n):
        for k in range(n):
            u = k / n
            pts.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    def bez(a, b, cc, n):
        for k in range(n):
            u = k / n
            x = (1 - u) ** 2 * a[0] + 2 * (1 - u) * u * cc[0] + u * u * b[0]
            y = (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * cc[1] + u * u * b[1]
            pts.append((x, y))
    seg_line((214, 132), (226, 126), 14)
    seg_line((226, 126), (240, 46), 80)                  # lift hill
    bez((240, 46), (258, 50), (247, 36), 24)            # crest
    bez((258, 50), (296, 130), (270, 128), 90)          # drop
    seg_line((296, 130), (312, 130), 16)
    cx0, cy, r = 312, 103, 27                           # the loop (slight spiral)
    n = 170
    for k in range(n):
        u = k / n
        th = math.pi / 2 - u * 2 * math.pi
        pts.append((cx0 + 8 * u + r * math.cos(th), cy + r * math.sin(th)))
    seg_line((320, 130), (350, 130), 30)
    bez((350, 130), (384, 130), (367, 66), 80)           # camelback
    bez((384, 130), (396, 134), (392, 134), 14)
    seg_line((396, 134), (214, 134), 182)               # return run to the station
    return np.array(pts, np.float32)


@lru_cache(maxsize=None)
def coaster_path():
    """Resampled path (1 px spacing) + frame->arc-length table (energy-based speed)."""
    p = _coaster_pts()
    d = np.sqrt(((np.roll(p, -1, 0) - p) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(d[:-1])])
    L = s[-1] + d[-1]
    ss = np.arange(0, L, 1.0)
    xs = np.interp(ss, np.append(s, L), np.append(p[:, 0], p[0, 0]))
    ys = np.interp(ss, np.append(s, L), np.append(p[:, 1], p[0, 1]))
    n = len(ss)
    lift0, lift1 = 14, 14 + 82
    v = np.where((np.arange(n) >= lift0) & (np.arange(n) < lift1), 0.9,
                 0.56 * np.sqrt(np.maximum(5.0, ys - 40.0)))
    v[-185:] = np.minimum(v[-185:], 1.6)               # brakes on the return run
    tt = np.concatenate([[0], np.cumsum(1.0 / v[:-1])])
    return xs, ys, tt, tt[-1] + 1.0 / v[-1]


def coaster_s(f):
    xs, ys, tt, T = coaster_path()
    tt = tt
    t = (f - 216 + tt[84] + 30.0) % T   # phase: crest just after the cut, loop ~244-270
    return float(np.interp(t, tt, np.arange(len(tt))))


def cpos(s):
    xs, ys, _, _ = coaster_path()
    n = len(xs)
    i = int(s) % n
    j = (i + 1) % n
    u = s - int(s)
    return xs[i] + (xs[j] - xs[i]) * u, ys[i] + (ys[j] - ys[i]) * u


@lru_cache(maxsize=None)
def coaster_layer():
    """Static track + lattice supports (mid layer)."""
    cv = layer(PW)
    xs, ys, _, _ = coaster_path()
    n = len(xs)
    # supports
    loop_top = (xs > 290) & (xs < 348) & (ys < 124)
    posts = []
    for x in range(216, 398, 9):
        m = (np.abs(xs - x) < 0.8) & ~loop_top
        if not m.any():
            continue
        y = int(round(ys[m].min()))
        if y >= 128:
            continue
        posts.append((x, y))
    for (xa, ya), (xb, yb) in zip(posts, posts[1:]):
        yt = max(ya, yb) + 4
        if yt < RIM_Y - 6:
            cv.line(xa, yt, xb, RIM_Y, '#2e2e66')
            cv.line(xb, yt, xa, RIM_Y, '#2e2e66')
            for yy in range(yt, RIM_Y, 12):
                cv.hline(xa, xb, yy, '#34346e')
    for x, y in posts:
        cv.vline(x, y + 2, RIM_Y + 2, '#3a3a7a')
        cv.vline(x + 1, y + 2, RIM_Y + 2, '#5a5aa0')
    for x in (296, 330):
        cv.vline(x, 104, RIM_Y + 2, '#3a3a7a')
        cv.vline(x + 1, 104, RIM_Y + 2, '#5a5aa0')
    cv.rect(206, 130, 24, 8, '#2a2a5a')
    cv.rect(206, 128, 24, 2, '#5a8ad8')
    # rails: two-tone with ties
    for k in range(n):
        x, y = xs[k], ys[k]
        x2, y2 = xs[(k + 1) % n], ys[(k + 1) % n]
        tx, ty = x2 - x, y2 - y
        nx, ny = ty, -tx
        cv.px(int(round(x + nx)), int(round(y + ny)), 'baby_blue')
        cv.px(int(round(x)), int(round(y)), '#4a72c8')
        if k % 3 == 0:
            cv.px(int(round(x - nx)), int(round(y - ny)), '#2e3a7a')
    return to_rgba(cv)


def coaster_dyn(c, f, ox):
    xs, ys, _, _ = coaster_path()
    n = len(xs)
    # chasing blue neon along the rail
    for k in range((f * 2) % 10, n, 10):
        x, y = int(round(xs[k])) + ox, int(round(ys[k]))
        c.px(x, y + 1, 'neon_cyan')
    # the empty car train (2 cars), oriented along the track
    s0 = coaster_s(f)
    for j in range(2):
        s = s0 - j * 7
        x, y = cpos(s)
        xa, ya = cpos(s - 2.5)
        xb, yb = cpos(s + 2.5)
        tx, ty = xb - xa, yb - ya
        L = math.hypot(tx, ty) or 1.0
        tx, ty = tx / L, ty / L
        nx, ny = ty, -tx
        X = x + ox
        def P(a, b):
            return (X + tx * a + nx * b, y + ty * a + ny * b)
        c.poly([P(-3.5, 0.5), P(3.5, 0.5), P(3.5, 4.0), P(-3.5, 4.0)], 'hot_pink')
        c.poly([P(-3.5, 3.0), P(3.5, 3.0), P(3.5, 4.0), P(-3.5, 4.0)], 'cotton')
        c.poly([P(-3.5, 0.5), P(3.5, 0.5), P(3.5, 1.4), P(-3.5, 1.4)], 'gold')
        c.px(*[int(round(v)) for v in P(-2, 4.6)], 'neon_cyan')
        c.px(*[int(round(v)) for v in P(1, 4.6)], 'neon_cyan')
        if j == 0:
            hx, hy = P(3.6, 2)
            c.px(int(round(hx)), int(round(hy)), 'butter')
            fx.glow(c, int(hx), int(hy), 6, 'neon_gold', 0.3, rings=2)


# ---- Ferris wheel
WCX, WCY, WR = 96, 72, 46
GOND = ('neon_pink', 'neon_cyan', 'neon_gold', 'lavender')


def ferris(c, f, ox):
    cx, cy, R = WCX + ox, WCY, WR
    ang = (f - 216) * 0.011
    for dx in (-24, 24):
        c.line(cx, cy, cx + dx, RIM_Y + 2, '#4a3c80')
        c.line(cx + (1 if dx > 0 else -1), cy, cx + dx + (1 if dx > 0 else -1), RIM_Y + 2, '#2e2660')
    c.line(cx - 15, cy + 30, cx + 15, cy + 30, '#3a3070')
    fx.glow(c, cx, cy, R + 8, 'magic', 0.10, rings=3)
    c.circle(cx, cy, R - 4, '#3e2e78', fill=False)
    for k in range(16):
        a = ang + k * math.pi / 8
        c.line(cx, cy, cx + math.cos(a) * (R - 1), cy + math.sin(a) * (R - 1), '#7a6ab0')
    c.circle(cx, cy, R, 'neon_pink', fill=False)
    # chase bulbs on the inner ring
    for k in range(40):
        a = ang + k * math.pi / 20
        p = (k + f // 2) % 4
        c.px(int(round(cx + math.cos(a) * (R - 4))), int(round(cy + math.sin(a) * (R - 4))),
             ('white', 'neon_cyan', '#3a6a9a', 'neon_cyan')[p])
    # gondolas (always hang upright), empty, lit inside
    for k in range(16):
        a = ang + (k + 0.5) * math.pi / 8
        gx, gy = int(round(cx + math.cos(a) * R)), int(round(cy + math.sin(a) * R))
        cc = GOND[k % 4]
        c.vline(gx, gy, gy + 1, '#9a8ac8')
        c.rect(gx - 3, gy + 2, 7, 1, shade(cc, 0.7))
        c.rect(gx - 3, gy + 3, 7, 4, cc)
        c.rect(gx - 2, gy + 4, 5, 2, '#fff0c0')
        c.hline(gx - 3, gx + 3, gy + 7, shade(cc, 0.6))
    c.circle(cx, cy, 4, '#c98a1e')
    c.circle(cx, cy, 3, 'gold')
    icon(c, 'star5', cx - 2, cy - 2, 'butter')


# ---- jackpot casino (mid layer)
REELS = ('star5', 'heart', 'bell', 'diamond', 'star5', 'heart', 'diamond', 'bell')
REEL_C = {'star5': 'neon_gold', 'heart': 'neon_pink', 'bell': 'gold', 'diamond': 'neon_cyan'}
REEL_STOP = (234, 243, 252)
JX, JY = 152, 64        # sign top-left (mid layer)


@lru_cache(maxsize=None)
def casino_layer():
    cv = layer(PW)
    x0, x1, top = 146, 208, 92
    cv.rect(x0, top, x1 - x0, RIM_Y + 4 - top, '#4a2a7a')
    cv.rect(x1 - 6, top, 6, RIM_Y + 4 - top, '#3a2066')
    cv.vline(x0, top, RIM_Y + 3, '#6a4aa0')
    cv.rect(x0 - 1, top - 2, x1 - x0 + 2, 2, 'gold')
    windows(cv, x0 + 4, top + 5, 9, 4, 6, 8, 3, 5, 2, 211, wall='#4a2a7a', arch=True)
    cv.hline(x0, x1 - 1, top + 38, 'neon_pink')
    # sign
    cv.vline(JX + 8, 88, top - 3, 'steel_dk')
    cv.vline(JX + 40, 88, top - 3, 'steel_dk')
    cv.round_rect(JX, JY, 50, 26, '#c98a1e', r=2)
    cv.round_rect(JX + 1, JY + 1, 48, 24, 'gold', r=2)
    cv.rect(JX + 3, JY + 3, 44, 20, '#3a1028')
    # lever
    cv.vline(JX + 51, JY + 6, JY + 16, 'steel_hi')
    cv.circle(JX + 52, JY + 5, 2, 'red')
    cv.px(JX + 51, JY + 4, 'salmon')
    return to_rgba(cv)


def reel_icon(f, r):
    """Icon index + sub-scroll of reel r at frame f."""
    stop = REEL_STOP[r]
    if f >= stop:
        return 0, 0                      # star
    sp = 3 if f < stop - 6 else 1 + (stop - f) // 3
    pos = (f - 216) * sp + r * 17
    return (pos // 12) % len(REELS), pos % 12


def jackpot(c, f, ox):
    jx, jy = JX + ox, JY
    for r in range(3):
        x = jx + 5 + r * 14
        c.rect(x, jy + 4, 12, 18, 'cream')
        c.hline(x, x + 11, jy + 4, '#d8c8b0')
        c.hline(x, x + 11, jy + 21, '#d8c8b0')
        idx, sub = reel_icon(f, r)
        for k in (-1, 0, 1):
            nm = REELS[(idx + k) % len(REELS)]
            yy = jy + 10 + k * 12 + sub
            if jy + 4 <= yy and yy + 5 <= jy + 22:
                icon(c, nm, x + 4, yy, REEL_C[nm])
        if f >= REEL_STOP[r] and f < REEL_STOP[r] + 3:
            c.rect(x, jy + 4, 12, 18, 'white')
            icon(c, 'star5', x + 4, jy + 10, 'gold')
    won = f >= REEL_STOP[2]
    # rim bulbs
    nb = 0
    pts = ([(jx + i, jy) for i in range(1, 50, 3)] + [(jx + 49, jy + i) for i in range(2, 25, 3)] +
           [(jx + 49 - i, jy + 25) for i in range(1, 50, 3)] + [(jx, jy + 25 - i) for i in range(2, 25, 3)])
    for k, (x, y) in enumerate(pts):
        if won:
            on = ((f // 3) % 2 == 0) == (k % 2 == 0)
        else:
            on = (k + f // 2) % 4 == 0
        c.px(x, y, 'white' if on else '#b0702a')
    if won:
        u = f - REEL_STOP[2]
        pulse = 1.0 if (f % FRAMES_PER_BEAT) < 6 else 0.5
        fx.glow(c, jx + 25, jy + 13, 40, 'neon_gold', 0.30 * pulse, rings=4)
        if u < 4:
            fx.glow(c, jx + 25, jy + 13, 70, 'white', 0.25 * (1 - u / 4), rings=3)
        # coins/sparkles spilling from the sign
        for k in range(14):
            h1, h2 = hash01(k, 9, 3), hash01(k, 8, 3)
            age = (u + int(h1 * 30)) % 30
            px_ = jx + 25 + int((h2 - 0.5) * 50) + int((h2 - 0.5) * age)
            py_ = jy + 26 + int(age * age * 0.06)
            if py_ < RIM_Y - 2:
                c.px(px_, py_, 'butter' if (age + k) % 4 else 'white')
    else:
        fx.glow(c, jx + 25, jy + 13, 30, 'neon_gold', 0.12, rings=3)


# ---- fountain jets
NJ = 11
def jet_x(i):
    return 56 + 28 * i


def jet_h(f, i):
    """Jet height: a new pattern jumps up on every beat, then settles."""
    b = (f - 216) // FRAMES_PER_BEAT
    fb = (f - 216) % FRAMES_PER_BEAT
    p = b % 4
    mid = (NJ - 1) / 2
    if p == 0:
        hh = 0.55 + 0.1 * (i % 2)
    elif p == 1:
        hh = 0.95 if i % 2 == 0 else 0.35
    elif p == 2:
        hh = 0.3 + 0.7 * (i / (NJ - 1))
    else:
        hh = 1.0 - 0.7 * abs(i - mid) / mid
    if b % 8 >= 4:
        hh = 1.25 - hh if p != 3 else hh
    k = 0.45 + 0.55 * ease_out(min(1.0, fb / 5.0)) - 0.12 * seg(fb, 8, 17)
    return max(6, int(86 * hh * k))


def jet_col(f, i):
    b = (f - 216) // FRAMES_PER_BEAT
    return JET_COLS[(i + b) % 4]


def fountain(c, f, ox):
    """Dancing jets: wobbling translucent-looking column (coloured from below, white core),
    a mist crown, an umbrella of falling droplets and foam at the base."""
    for i in range(NJ):
        x = jet_x(i) + ox
        if x < -16 or x > W + 16:
            continue
        h = jet_h(f, i)
        cc = col(jet_col(f, i))
        top = RIM_Y - h
        light = mix(cc, 'white', 0.6)
        fx.glow(c, x, RIM_Y - 2, 18, cc, 0.45, rings=3)
        fx.glow(c, x, top + 2, 7, light, 0.30, rings=2)
        for y in range(top + 1, RIM_Y):
            u = (y - top) / max(1, h)
            wob = int(hash01(i, y // 2, (f + y) // 3) * 3) - 1 if u > 0.15 else 0
            half = 0 if u < 0.12 else (1 if u < 0.7 else 2)
            edge = mix(cc, 'white', 0.25 + 0.5 * (1 - u))
            c.hline(x - half + min(0, wob), x + half + max(0, wob), y, edge)
            c.px(x, y, mix('white', cc, 0.2 * u))
        # crown
        c.hline(x - 2, x + 2, top, light)
        c.hline(x - 1, x + 1, top - 1, 'white')
        c.px(x, top - 2, light)
        # droplets: umbrella falling back into the basin
        g = 2.0 * (h + 6) / (24.0 * 24.0)
        for k in range(22):
            h1, h2 = hash01(i, k, 61), hash01(i, k, 62)
            age = (f + int(h1 * 24)) % 24
            vx = (0.15 + 0.45 * h2) * (1 if k % 2 else -1)
            dx = int(round(vx * age))
            dy = int(round(-0.9 * age + 0.5 * g * age * age + 0.9 * age * (age / 24.0)))
            yy = top + dy
            if yy < RIM_Y - 1:
                pc = light if age < 9 else (mix(cc, 'white', 0.3) if k % 3 else 'white')
                c.px(x + dx, yy, pc)
                if age > 4 and k % 2:
                    c.px(x + dx, yy - 1, mix(pc, '#2a1a4a', 0.5))
        # foam at the nozzle
        for k in range(5):
            fx_ = x - 4 + int(hash01(i, k, f // 2) * 9)
            c.px(fx_, RIM_Y - 1 - int(hash01(k, i, f // 3) * 2), 'white')


@lru_cache(maxsize=None)
def plaza_layer():
    """Basin rim + front wall + paved plaza floor (shot B, parallax 1)."""
    cv = layer(PW)
    # floor rows (side-view perspective bands)
    rows = ((FLOOR_B, 3), (FLOOR_B + 3, 4), (FLOOR_B + 7, 5), (FLOOR_B + 12, 6), (FLOOR_B + 18, 8))
    for k, (y, h) in enumerate(rows):
        cv.rect(0, y, PW, h, '#3e2c6a' if k % 2 == 0 else '#46337a')
        cv.hline(0, PW - 1, y, '#5a4690')
        step = 10 + k * 4
        off = (k * 7) % step
        for x in range(off, PW, step):
            cv.vline(x, y + 1, y + h - 1, '#2e2252')
    # basin front wall
    cv.rect(30, RIM_Y + 2, 340, FLOOR_B - RIM_Y - 1, '#7c5ea8')
    cv.rect(30, FLOOR_B - 3, 340, 3, '#7a5ca0')
    for x in range(36, 366, 28):
        cv.rect(x, RIM_Y + 6, 20, 9, '#8466ac')
        cv.frame(x, RIM_Y + 6, 20, 9, '#c4a8e4')
        cv.rect(x + 8, RIM_Y + 9, 4, 3, '#ffe9a8')
    cv.rect(28, RIM_Y - 1, 344, 3, 'cotton')
    cv.hline(28, 371, RIM_Y - 1, 'white')
    cv.hline(28, 371, RIM_Y + 2, 'gold')
    # end pedestals with gold urn
    for x in (24, 366):
        cv.rect(x, RIM_Y - 8, 10, FLOOR_B - RIM_Y + 8, '#d8c0ec')
        cv.rect(x + 8, RIM_Y - 8, 2, FLOOR_B - RIM_Y + 8, '#a888c8')
        cv.rect(x - 1, RIM_Y - 10, 12, 2, 'gold')
        cv.ellipse(x + 5, RIM_Y - 14, 4, 3, 'gold')
        cv.rect(x + 3, RIM_Y - 19, 5, 2, '#c98a1e')
    return to_rgba(cv)


@lru_cache(maxsize=None)
def plaza_fg():
    """Foreground: ornate lamp posts + palm planters (parallax 1.25)."""
    cv = layer(520)
    for x in (30, 250, 470):
        cv.rect(x - 4, 168, 9, 12, '#3a2a62')
        cv.rect(x - 3, 166, 7, 2, 'gold')
        cv.rect(x - 1, 112, 3, 54, '#2e2252')
        cv.vline(x - 1, 112, 165, '#4a3a7a')
        cv.hline(x - 7, x + 7, 118, '#2e2252')
        for dx in (-7, 0, 7):
            cv.rect(x + dx - 2, 108 if dx else 104, 5, 6, '#fff4d6')
            cv.rect(x + dx - 1, 107 if dx else 103, 3, 1, 'gold')
            cv.hline(x + dx - 2, x + dx + 2, 114 if dx else 110, '#c98a1e')
        cv.px(x, 101, 'gold')
    for x in (140, 360):
        cv.rect(x - 9, 166, 19, 14, '#c0408a')
        cv.rect(x - 10, 164, 21, 3, '#e070b0')
        cv.hline(x - 9, x + 9, 167, '#8a2a6a')
        for k in range(7):
            cv.px(x - 8 + k * 3, 163, ('neon_cyan', 'cotton', 'neon_gold')[k % 3])
    return to_rgba(cv)


def palm(c, x, f, k):
    """Palm tree in a planter (foreground), fronds sway."""
    sway = int(round(math.sin(f * 0.09 + k) * 1.2))
    for i in range(44):
        yy = 163 - i
        xx = x + int(math.sin(i * 0.05) * 3 + (sway if i > 30 else 0) * i / 44)
        c.rect(xx, yy, 3, 1, '#4a3050' if i % 4 else '#6a4a6a')
    tx, ty = x + 3 + sway, 119
    for j, (dx, dy) in enumerate(((-14, 4), (-10, 9), (12, 5), (9, 10), (-4, 11), (3, -6), (-8, -3), (8, -2))):
        ex, ey = tx + dx + (sway if j % 2 else 0), ty + dy
        c.line(tx, ty, ex, ey, '#1e6a5a')
        c.line(tx, ty + 1, ex, ey + 1, '#16504a')
        c.px(ex, ey + 1, '#2a8a6a')
    c.rect(tx - 1, ty - 1, 3, 3, '#4a3050')


def render_b(f):
    cam = eased_cam(f, CAM_B)
    c = backdrop(12).copy()
    fx.stars(c, f, seed=26, n=80, region=(0, 0, W, 90))
    moon(c, 292, 40, 1.0)
    fx.glow(c, 160, 128, 150, '#ff4fd8', 0.10, rings=5)
    paste(c, far_resort(), -int(cam * PAR_FAR_B) - 20, 0)
    fx.light_rect(c, 0, 112, W, 30, '#8a3aa8', 0.18, fade='up')
    mo = -int(cam * PAR_MID)
    draw_balloon(c, 196 + int((f - 216) * 0.25) - int(cam * 0.5), 44, f, size='m',
                 flame=(f // 18) % 4 == 0)
    ferris(c, f, mo)
    paste(c, casino_layer(), mo, 0)
    jackpot(c, f, mo)
    paste(c, coaster_layer(), mo, 0)
    coaster_dyn(c, f, mo)
    po = -cam
    # neon pooled on the plaza
    fountain(c, f, po)
    paste(c, plaza_layer(), po, 0)
    for i in range(NJ):
        x = jet_x(i) + po
        cc = jet_col(f, i)
        c.hline(x - 3, x + 3, RIM_Y, mix(cc, 'white', 0.4))
        fx.light_rect(c, x - 9, RIM_Y + 3, 19, 18, cc, 0.16, fade='down')
        fx.light_rect(c, x - 6, FLOOR_B, 13, 12, cc, 0.18, fade='down')
    fo = -int(cam * PAR_FG_B)
    paste(c, plaza_fg(), fo, 0)
    for k, x in enumerate((30, 250, 470)):
        fx.glow(c, x + fo, 108, 16, 'butter', 0.30, rings=3)
    palm(c, 136 + fo, f, 0)
    palm(c, 356 + fo, f, 2)
    xpaste(c, fg_puffs(True), fg_puffs(True), -int(cam * 1.4) - 60 + int((f - 216) * 0.2), 150, 1.0)
    callout(c, 196 + po, RIM_Y + 1, 'FOUNTAIN PLAZA', f, 222, 281, side='down', length=10)
    lx = 320 + mo
    callout(c, lx, 76, 'WILD BLUE YONDER', f, 240, 287, side='up', length=12)
    return c


# ================================================================== shot C: fireworks
CAM_C = [(288, 40), (359, 92)]
TILT_C = [(288, 0), (300, 2), (359, 40)]
# (t_burst, x, y, radius, c1, c2, n, kind)   x, y in sky screen coords at tilt 0
SHELLS = (
    (288, 112, 34, 44, 'neon_pink', 'magic', 96, 'peony'),
    (294, 262, 50, 20, 'neon_cyan', 'baby_blue', 40, 'ring'),
    (299, 46, 56, 26, 'neon_gold', 'amber', 50, 'willow'),
    (306, 228, 32, 26, 'lavender', 'neon_pink', 48, 'heart'),
    (313, 292, 44, 16, 'leaf', 'neon_cyan', 32, 'peony'),
    (318, 150, 50, 16, 'neon_cyan', 'white', 32, 'ring'),
    (324, 90, 16, 50, 'neon_gold', 'amber', 100, 'willow'),
    (333, 262, 30, 26, 'neon_cyan', 'white', 52, 'ring'),
    (342, 212, 24, 46, 'neon_pink', 'neon_gold', 100, 'peony'),
    (346, 50, 40, 18, 'magic', 'lavender', 36, 'peony'),
    (350, 288, 22, 22, 'neon_gold', 'neon_pink', 44, 'heart'),
)


def shell_pos(sh, a, i):
    """Position of particle i of shell sh at age a (sky coords). Inner core for big peonies."""
    t0, x, y, R, c1, c2, n, kind = sh
    a = max(0.0, a)
    D = R * (1 - math.exp(-a / 5.0))
    g = 0.010 * a * a * (2.4 if kind == 'willow' else 1.0)
    h1, h2 = hash01(t0, i, 1), hash01(t0, i, 2)
    th = 2 * math.pi * i / n + (h1 - 0.5) * 0.25
    if kind == 'heart':
        u = 2 * math.pi * i / n
        hx = 16 * math.sin(u) ** 3
        hy = -(13 * math.cos(u) - 5 * math.cos(2 * u) - 2 * math.cos(3 * u) - math.cos(4 * u))
        dx, dy = hx / 17.0, hy / 17.0 - 0.15
    elif kind == 'ring':
        dx, dy = math.cos(th), math.sin(th) * 0.75
    else:
        sp = (0.82 + 0.18 * h2) if i % 3 else (0.45 + 0.1 * h2)
        dx, dy = math.cos(th) * sp, math.sin(th) * sp
    return x + dx * D, y + dy * D + g


def fireworks(c, f, oy):
    """Shells: rising trail, white flash, coloured streaks, colour shift, twinkling falloff."""
    flashes = []
    dimk = col('#1a1030')
    for sh in SHELLS:
        t0, x, y, R, c1, c2, n, kind = sh
        a = f - t0
        yy0 = y + oy
        if -12 <= a < 0:
            u = (a + 12) / 12.0
            sy = int(lerp(150 + oy, yy0, ease_out(u)))
            sx = x + int(round(math.sin(u * 3)))
            c.px(sx, sy, 'white')
            c.px(sx, sy + 1, 'butter')
            for k in range(2, 9):
                if hash01(t0, f, k) > 0.35:
                    c.px(sx + (1 if hash01(t0, k, f) > 0.7 else 0), sy + k, mix('amber', '#3a2050', k / 9.0))
            continue
        life = 30 if kind != 'willow' else 44
        if a < 0 or a > life:
            continue
        if a < 10:
            flashes.append((x, yy0, R, c1, a))
        for i in range(n):
            core = kind == 'peony' and i % 3 == 0
            cA, cB = (c2, c1) if core else (c1, c2)
            if a < 3:
                cc = col('white')
            elif a < life * 0.45:
                cc = col(cA)
            elif a < life * 0.72:
                cc = col(cB) if hash01(t0, i, f) > 0.2 else col('white')
            else:
                if hash01(t0, i, f) < 0.5:
                    continue
                cc = col('butter') if kind == 'willow' else mix(cB, 'white', 0.4)
            px_, py_ = shell_pos(sh, a, i)
            ix, iy = int(round(px_)), int(round(py_ + oy))
            if a >= 1 and a < life * 0.5:
                # radial streak: dim tail, bright head
                back = 4.0 if (kind == 'willow' or R >= 40) else 2.5
                qx, qy = shell_pos(sh, a - back, i)
                c.line(qx, qy + oy, ix, iy, mix(cc, dimk, 0.55))
                qx, qy = shell_pos(sh, a - 1.2, i)
                c.line(qx, qy + oy, ix, iy, cc)
            elif a >= life * 0.5 and kind == 'willow':
                qx, qy = shell_pos(sh, a - 3, i)
                c.line(qx, qy + oy, ix, iy, mix('amber', dimk, 0.5))
            c.px(ix, iy, cc if a >= 4 else col('white'))
        if a < 5:
            c.circle(x, yy0, max(1, 4 - a), 'white')
            c.circle(x, yy0, max(0, 2 - a), 'white')
        elif a < 22:
            # thin smoke left behind
            c.dither(x - R // 3, yy0 - R // 4, 2 * R // 3, R // 2, '#3a2a58', 0.15 * (1 - (a - 5) / 17.0))
    return flashes


def balloon_silhouette(c, x, y, f, size, k, rim_col, rim_dx):
    """The shared balloon sprite drawn darkened (silhouette against a burst) with a rim light."""
    tmp = Canvas(40, 50, KEY)
    draw_balloon(tmp, 20, 46, f, size=size)
    a = tmp.a
    m = ~np.all(a == _KEYA, axis=2)
    dark = (a.astype(np.float32) * (1 - k) + np.array(col('#1c1234'), np.float32) * k).astype(np.uint8)
    edge = m & ~np.roll(m, -rim_dx, axis=1)
    dark[edge] = col(rim_col)
    x0, y0 = int(x) - 20, int(y) - 46
    reg_m = m
    out = np.zeros((50, 40, 4), np.uint8)
    out[..., :3] = dark
    out[..., 3] = reg_m * 255
    c.blit(out, x0, y0)


def render_c(f):
    cam = eased_cam(f, CAM_C)
    oy = eased_cam(f, TILT_C)
    box = []
    c = shot_strip(f, cam, oy=oy, night=True, sky_t=1.0, balloon=False,
                   sky_fx=lambda cc: box.extend(fireworks(cc, f, oy)))
    # burst light on the towers, clouds and sky
    for (x, y, R, cc, a) in box:
        k = 1 - a / 10.0
        fx.glow(c, x, y, int(R * 1.6), cc, 0.30 * k, rings=4)
        if R >= 40:
            fx.glow(c, x, y + 40, 170, cc, 0.16 * k, rings=4)
            if a < 3:
                c.blend(0, 0, W, H, mix(cc, 'white', 0.5), 0.10 * (3 - a) / 3.0)
    # the balloon crosses the sky and is silhouetted against the big 324 gold willow
    bx = 64 + int((f - 288) * 0.6)
    by = 46 + oy // 3
    a = f - 324
    if 0 <= a < 30:
        balloon_silhouette(c, bx, by, f, 'l', 0.85 - 0.3 * a / 30.0, 'butter', 1)
    else:
        draw_balloon(c, bx, by, f, size='l', flame=(f // 18) % 3 == 0)
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
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), 204, 100)
    return c


SFX = [
    (84, 'wind', 0.35, {'variant': 0}),
    (144, 'neon', 0.8),
    (162, 'neon', 0.4),
    (180, 'neon', 0.4),
    (198, 'neon', 0.45),
    (216, 'fountain', 0.55),
    (240, 'whoosh', 0.45, {'align': 'hit'}),
    (252, 'jackpot', 0.7),
    (270, 'fountain', 0.35),
    (288, 'firework', 0.9, {'align': 'hit', 'variant': 2}),
    (306, 'firework', 0.5, {'align': 'hit', 'variant': 0}),
    (324, 'firework', 0.75, {'align': 'hit', 'variant': 1}),
    (342, 'firework', 0.85, {'align': 'hit', 'variant': 2}),
    (351, 'firework', 0.45, {'align': 'hit', 'variant': 0}),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'dissolve', {}), (288, 'dissolve', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
