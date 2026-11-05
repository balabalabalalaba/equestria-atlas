"""MANEHATTAN: the island metropolis (scene frames 0-431, HUD clock 12:15 -> 13:00).

Scene frames (engine/regionfx.long_flight maps them to film frames; bar = 72, beat = 18). High noon:
the sun stands high, crisp short shadows fall straight down under every ledge, marquee and awning.
Four shots of 90 frames (3 s), slow even `cruise` camera pans, smooth cuts (dissolve, glide, glide).
  0-71      World-map travel (handled by the wrapper).
  72-161    REVEAL (72): iris opens 72-79 on the CRYSTALLER BUILDING (stepped sunburst crown, gold
            needle glinting on every beat). SHOT A: skyline from the harbour: far mainland + the
            torch-column islet (tiny distant silhouette), hazy far towers, the Art-Deco cluster
            (setbacks, ribbing, gold crowns, hundreds of windows, glints), quay with piers, sheds and
            swinging cranes, rippling reflections + sun glitter, ferries / tug / sailboat with wakes,
            buoy, foreground pilings. Callouts CRYSTALLER BUILDING 80-147, MANEHATTAN 90-147.
            Atlas card 100-196.
  162-251   dissolve -> SHOT B: BRIDLEWAY at street level: marquees (chasing bulbs, full flash on
            every beat), blade signs lighting icon by icon, billboard, posters, awnings, lamp
            posts, manhole steam, parked cab cart, the el train on its girder viaduct.
            Callout BRIDLEWAY 168-237.
  252-341   glide -> SHOT C: THE PARK: lake, rowboats, stone arch bridge + reflection, the giant
            rock, fountain jets pulsing on the beat, blossom tree, boathouse; Bronclyn brownstones
            (stoops, fire escapes, tanks, laundry) and towers behind. Callouts THE PARK 258-327,
            BRONCLYN 270-327.
  342-431   glide -> SHOT D: the RAIL BRIDGE to the mainland. `bridge` HIT on the bar line 360: the
            pink Friendship Express bursts in from the left with a whistle plume, tower emblems
            flash; second toot at 396. The balloon sails past the Crystaller spire; the camera pans
            with the train and cranes up (pull back). Callout RAIL BRIDGE 348-420. Toast from 374
            (top-centre kept clear). Last 8 frames under the wipe.
Ambient: drifting clouds, water sparkles + rippling reflections, ferries/tug/sailboat, wakes,
         smoke, window glints, flags, swinging cranes, buoy bob, marquee bulbs, neon blade signs,
         manhole steam, el train, fountain jets, rowboats bobbing, petals, laundry, train steam.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'manehattan'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)
HAZE = col('#c4e4f8')
SUN = (236, 9)

T_A, T_B, T_C, T_D = 72, 162, 252, 342      # cuts every 90 frames (3 s per shot)
D_HIT = 360                                 # `bridge` hit: bar line, the train bursts onto the bridge


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
    """Blit an RGBA layer with its top-left at screen (x, y)."""
    c.blit(rgba, int(x), int(y))


def hz(cc, t):
    return mix(cc, HAZE, t) if t > 0 else col(cc)


_NZ = rng(808).random((128, 1024)).astype(np.float32)


def eased_cam(f, keys):
    return int(round(keyframes(f, keys, ease=smooth)))


def cruise_cam(f, f0, f1, dist):
    """Slow, even camera travel (constant speed middle, short ramps)."""
    return int(round(dist * cruise(seg(f, f0, f1))))


# ================================================================== sky + clouds
@lru_cache(maxsize=None)
def sky_layer(y1=H):
    c = Canvas()
    c.gradient(0, 0, W, y1, ['#3a6cc4', '#4d86da', 'sky', '#6db6f0', 'light_sky', '#c8ecff'], soft=0.7)
    # high-noon sun: a white disc with stepped warm glow, partly above the frame
    fx.glow(c, SUN[0], SUN[1], 30, '#fff6c8', 0.35, rings=5)
    fx.glow(c, SUN[0], SUN[1], 14, 'white', 0.45, rings=3)
    c.circle(SUN[0], SUN[1], 6, '#fffbe6')
    c.circle(SUN[0], SUN[1], 4, 'white')
    return c


@lru_cache(maxsize=None)
def cloud_sprite(w, seed, tint=0.0):
    cv = layer(w + 6, max(5, w // 3) + 6)
    base = mix('white', HAZE, tint)
    fx.cloud(cv, 2, 2, w, base, mix(base, '#9ab4d8', 0.45), seed=seed)
    # light top rim
    return to_rgba(cv)


def draw_clouds(c, f, specs, ox=0):
    """specs: (x0, y, w, speed, seed); x scrolls slowly and wraps over 520 px."""
    for x0, y, w, sp, sd in specs:
        x = int((x0 + f * sp - ox) % 560) - 120
        paste(c, cloud_sprite(w, sd), x, y)


# ================================================================== architecture kit
FAC = {
    'teal':  ('#9fd0c8', '#e6f6ee', '#3f8f8a'),
    'beige': ('#eadbc0', '#fff6e0', '#b5654a'),
    'terra': ('#cf7c5c', '#f6caa0', '#7a3a2c'),
    'cream': ('#f6ecd4', '#ffffff', '#c9a24a'),
    'slate': ('#8fa3b8', '#d8e4f0', '#4f6f8f'),
    'sage':  ('#b8c8b8', '#eef6ee', '#5f8f8a'),
    'rose':  ('#ecb8aa', '#fff0e8', '#a85a4a'),
    'sand':  ('#e2c48e', '#fbe8c0', '#9a6a3a'),
    'blush': ('#f2cfd8', '#fff4f6', '#b06a82'),
}
GLASS = ('#34497a', '#4b6ea8', '#86b6e6', '#c6e6ff')


def win_cols(cv, x0, y0, w, h, wall, seed, haze=0.0, side=False, pitch=3):
    """Ribbed deco window field: 1-px light piers, 2x2 glass panes, 1-px spandrels."""
    wall = col(wall)
    pier = hz(mix(wall, 'white', 0.30 if not side else 0.05), haze)
    span = hz(shade(wall, 0.80 if not side else 0.66), haze)
    if w < 3 or h < 3:
        return
    cv.rect(x0, y0, w, h, span)
    for xx in range(x0, x0 + w, pitch):
        cv.vline(xx, y0, y0 + h - 1, pier)
    for xx in range(x0 + 1, x0 + w, pitch):
        reflect = hash01(seed, xx, 7)
        for yy in range(y0 + 1, y0 + h - 1, 3):
            r = hash01(seed, xx, yy)
            if side:
                g = GLASS[0] if r < 0.8 else GLASS[1]
            else:
                # sky reflection streaks run down whole columns; lower rows darker
                k = (yy - y0) / max(1, h)
                if reflect > 0.72 and r > 0.25:
                    g = GLASS[2] if r < 0.85 else GLASS[3]
                elif r < 0.10:
                    g = '#f2e6c8'          # blinds drawn
                elif r < 0.55 + 0.3 * k:
                    g = GLASS[0]
                else:
                    g = GLASS[1]
            g = hz(g, haze)
            ww = min(2, x0 + w - xx - 1) if not side else min(2, x0 + w - xx)
            if ww > 0:
                cv.rect(xx, yy, ww, 2, g)


def tower(cv, x, base, w, h, fac, crown='step', seed=0, tiers=2, haze=0.0, meta=None):
    """Art-Deco tower. (x, base) = bottom-left. Returns (crown_x, crown_y) of its top centre."""
    wall, trim, roof = (col(v) for v in FAC[fac])
    gold = hz('gold', haze)
    fr = {1: [1.0], 2: [0.74, 0.26], 3: [0.58, 0.26, 0.16]}[tiers]
    y = base
    tx, tw = x, w
    top = base
    for i, fz in enumerate(fr):
        th = max(5, int(round(h * fz)))
        top = y - th
        sw = max(2, tw // 5)
        cv.rect(tx, top, tw, th, hz(wall, haze))
        win_cols(cv, tx + 1, top + 3, tw - sw - 2, th - 4, wall, seed + i * 11, haze)
        win_cols(cv, tx + tw - sw, top + 3, sw, th - 4, shade(wall, 0.85), seed + i * 13, haze, side=True)
        cv.vline(tx, top, y - 1, hz(mix(wall, 'white', 0.45), haze))
        cv.rect(tx + tw - sw, top, 1, th, hz(shade(wall, 0.70), haze))
        # tier cap: trim ledge (overhangs 1 px), gold band, noon shadow straight down under it
        cv.hline(tx - 1, tx + tw, top, hz(trim, haze))
        cv.hline(tx, tx + tw - 1, top + 1, hz(shade(wall, 0.62), haze))
        if i > 0 or tiers == 1:
            cv.hline(tx, tx + tw - 1, top + 2, hz(shade(wall, 0.72), haze))
        y = top
        if i < len(fr) - 1:
            inset = max(2, tw // 6)
            tx, tw = tx + inset, tw - 2 * inset
    cx = tx + tw // 2
    # ---- crowns
    if crown == 'step':
        yy = top
        for k in range(3):
            ww = tw - 4 - 4 * k
            if ww < 3:
                break
            cv.rect(cx - ww // 2, yy - 2, ww, 2, hz(wall, haze))
            cv.hline(cx - ww // 2, cx - ww // 2 + ww - 1, yy - 2, hz(trim, haze))
            cv.px(cx + ww // 2 - 1, yy - 1, hz(shade(wall, 0.7), haze))
            yy -= 2
        cv.vline(cx, yy - 4, yy - 1, gold)
        cv.px(cx, yy - 5, hz('butter', haze))
        top = yy - 5
    elif crown == 'gold':
        cv.rect(tx + 1, top - 3, tw - 2, 3, gold)
        cv.hline(tx + 1, tx + tw - 2, top - 3, hz('butter', haze))
        for xx in range(tx + 2, tx + tw - 2, 3):
            cv.vline(xx, top - 6, top - 3, gold)
            cv.px(xx, top - 7, hz('butter', haze))
        cv.vline(cx, top - 11, top - 3, gold)
        cv.px(cx, top - 12, hz('white', haze))
        top -= 12
    elif crown == 'pyramid':
        ph = max(4, int(tw * 0.75))
        cv.tri((tx - 1, top), (tx + tw, top), (cx, top - ph), hz(roof, haze))
        cv.tri((cx, top), (tx + tw, top), (cx, top - ph), hz(shade(roof, 0.75), haze))
        for k in range(2, ph, 3):
            ww = int((tw + 1) * (1 - k / ph)) // 2
            cv.hline(cx - ww, cx + ww, top - k, hz(shade(roof, 0.85), haze))
        cv.vline(cx, top - ph - 3, top - ph, gold)
        cv.px(cx, top - ph - 4, hz('butter', haze))
        top -= ph + 4
    elif crown == 'dome':
        r = max(3, tw // 2 - 1)
        cv.ellipse(cx, top, r, r, hz('#5fb8a0', haze))
        cv.ellipse(cx + 1, top, r - 1, r - 1, hz('#3f9a86', haze))
        cv.rect(cx - r - 1, top, 2 * r + 3, 2, hz(trim, haze))
        cv.px(cx - r // 2, top - r + 1, hz('#bff0e0', haze))
        cv.vline(cx, top - r - 4, top - r, gold)
        cv.px(cx, top - r - 5, hz('butter', haze))
        top -= r + 5
    elif crown == 'tank':
        # rooftop wooden water tank on legs + a mast
        tx0 = tx + 2
        cv.vline(tx0, top - 3, top - 1, hz('#5e3c2c', haze))
        cv.vline(tx0 + 4, top - 3, top - 1, hz('#5e3c2c', haze))
        cv.rect(tx0, top - 8, 5, 5, hz('#9a6a48', haze))
        cv.vline(tx0 + 4, top - 8, top - 4, hz('#6e4a34', haze))
        cv.hline(tx0, tx0 + 4, top - 6, hz('#5e3c2c', haze))
        cv.tri((tx0 - 1, top - 8), (tx0 + 5, top - 8), (tx0 + 2, top - 11), hz('#6e4a34', haze))
        cv.vline(tx + tw - 3, top - 9, top - 1, hz('#7a8494', haze))
        if meta is not None:
            meta.setdefault('beacons', []).append((tx + tw - 3, top - 10))
        top -= 11
    elif crown == 'mast':
        cv.rect(cx - 2, top - 3, 5, 3, hz(trim, haze))
        cv.vline(cx, top - 14, top - 3, hz('#9aa4b4', haze))
        cv.hline(cx - 1, cx + 1, top - 8, hz('#9aa4b4', haze))
        if meta is not None:
            meta.setdefault('beacons', []).append((cx, top - 15))
        top -= 15
    elif crown == 'flag':
        cv.vline(cx, top - 10, top - 1, hz('#c8ccd4', haze))
        if meta is not None:
            meta.setdefault('flags', []).append((cx + 1, top - 10))
        top -= 10
    if meta is not None:
        meta.setdefault('tops', []).append((cx, top))
        # a few glinting windows for the noon sun
        for k in range(3):
            gx = x + 2 + int(hash01(seed, k, 3) * max(1, w - 6))
            gy = base - 6 - int(hash01(seed, k, 4) * max(1, h * 0.7))
            meta.setdefault('glints', []).append((gx, gy, hash01(seed, k, 5)))
    return cx, top


def crystaller(cv, cx, base, meta=None, haze=0.0):
    """The Crystaller Building: silver-cream shaft, stacked sunburst-arch crown, gold needle."""
    wall = col('#ece6d6')
    trim = col('white')
    steel = col('#c9ccd8')
    steel_dk = col('#8e93a8')
    gold = col('gold')
    H_ = lambda cc: hz(cc, haze)
    # podium
    pw = 30
    cv.rect(cx - pw // 2, base - 14, pw, 14, H_(wall))
    win_cols(cv, cx - pw // 2 + 1, base - 12, pw - 6, 11, wall, 901, haze)
    win_cols(cv, cx + pw // 2 - 5, base - 12, 5, 11, shade(wall, 0.85), 902, haze, side=True)
    cv.hline(cx - pw // 2 - 1, cx + pw // 2, base - 14, H_(trim))
    cv.hline(cx - pw // 2, cx + pw // 2 - 1, base - 13, H_(shade(wall, 0.62)))
    # shaft with corner setbacks
    sw_ = 22
    y0 = base - 14
    sh = 50
    cv.rect(cx - sw_ // 2, y0 - sh, sw_, sh, H_(wall))
    win_cols(cv, cx - sw_ // 2 + 1, y0 - sh + 3, sw_ - 6, sh - 4, wall, 903, haze)
    win_cols(cv, cx + sw_ // 2 - 4, y0 - sh + 3, 4, sh - 4, shade(wall, 0.85), 904, haze, side=True)
    cv.vline(cx - sw_ // 2, y0 - sh, y0 - 1, H_('white'))
    cv.vline(cx + sw_ // 2 - 4, y0 - sh, y0 - 1, H_(shade(wall, 0.7)))
    cv.hline(cx - sw_ // 2 - 1, cx + sw_ // 2, y0 - sh, H_(trim))
    cv.hline(cx - sw_ // 2, cx + sw_ // 2 - 1, y0 - sh + 1, H_(shade(wall, 0.62)))
    # deco corner ornaments (abstract steel fins with gold caps)
    for ex in (cx - sw_ // 2 - 1, cx + sw_ // 2):
        cv.rect(ex, y0 - sh - 2, 1, 4, H_(steel))
        cv.px(ex, y0 - sh - 3, H_(gold))
    # upper shaft
    uw = 16
    y1 = y0 - sh
    uh = 12
    cv.rect(cx - uw // 2, y1 - uh, uw, uh, H_(wall))
    win_cols(cv, cx - uw // 2 + 1, y1 - uh + 2, uw - 5, uh - 2, wall, 905, haze)
    win_cols(cv, cx + uw // 2 - 3, y1 - uh + 2, 3, uh - 2, shade(wall, 0.85), 906, haze, side=True)
    cv.vline(cx - uw // 2, y1 - uh, y1 - 1, H_('white'))
    cv.hline(cx - uw // 2 - 1, cx + uw // 2, y1 - uh, H_(gold))
    # sunburst crown: stacked arches getting narrower
    yb = y1 - uh
    arches = [(8, 7), (7, 6), (5, 6), (4, 5), (2, 4)]
    for k, (hw, ah) in enumerate(arches):
        # arch k: half-ellipse of half width hw, height ah, sitting on yb
        cv.rect(cx - hw, yb - ah + 2, 2 * hw, ah - 2, H_(steel))
        cv.ellipse(cx, yb - ah + 3, hw, 3, H_(steel))
        cv.rect(cx + hw - 2, yb - ah + 2, 2, ah - 2, H_(steel_dk))
        # sunburst triangular windows (dark glass with a gold-lit tip)
        for j in range(-hw + 2, hw - 1, 2):
            tx = cx + j
            cv.vline(tx, yb - ah + 3, yb - 1, H_('#3a4a72'))
            cv.px(tx, yb - ah + 3, H_('butter' if (j // 2 + k) % 2 == 0 else '#9ec8ee'))
        cv.hline(cx - hw + 1, cx + hw - 2, yb - ah + 1, H_('white'))
        yb -= ah - 1
    # needle spire
    tipy = yb - 16
    cv.rect(cx - 1, yb - 4, 2, 4, H_(steel))
    cv.vline(cx, tipy, yb - 1, H_(steel))
    cv.vline(cx - 1, tipy + 6, yb - 4, H_('white'))
    cv.px(cx, tipy - 1, H_('gold'))
    cv.px(cx, tipy - 2, H_('butter'))
    if meta is not None:
        meta['crystaller'] = (cx, tipy - 2, yb + 6)
        for k, gy in enumerate(range(y0 - sh + 6, base - 16, 9)):
            meta.setdefault('glints', []).append((cx - 6 + (k * 5) % 12, gy, hash01(77, k)))
    return tipy - 2


# ================================================================== water
_WROWS = {}


def water(c, f, wy, cols, refl=0.42, amp=1.0, speed=0.22, seed=0, src_y0=None, sun_x=SUN[0], ripple=1.0,
          glitter=46):
    """Harbour water from row wy to the bottom: rippling reflection of what is above + glints."""
    h2 = H - wy
    k = np.arange(h2)
    src = np.clip(wy - 1 - (k * 1.1).astype(int), 0, wy - 1)
    if src_y0 is not None:
        src = np.maximum(src, src_y0)
    off = np.round(np.sin(k * 0.45 + f * speed + seed) * (0.3 + k * 0.035 * amp)
                   + np.sin(k * 0.21 - f * speed * 0.6) * 0.5).astype(int)
    xs = np.clip(np.arange(W)[None, :] + off[:, None], 0, W - 1)
    rf = c.a[src[:, None], xs].astype(np.float32)
    key = (wy, h2, tuple(cols))
    if key not in _WROWS:
        base = Canvas(W, h2)
        base.gradient(0, 0, W, h2, cols, soft=0.8)
        _WROWS[key] = base.a.astype(np.float32)
    wc = _WROWS[key]
    rr = refl * (1 - 0.35 * k / h2)[:, None, None]
    out = wc * (1 - rr) + rf * rr
    # ripple highlights: horizontal dashes (longer near us) drifting slowly, plus darker troughs
    L = (2 + k // 9)[:, None]
    drift = (f * (0.25 + (k % 4) * 0.12)).astype(int)[:, None] * ((k % 2) * 2 - 1)[:, None]
    sx = ((np.arange(W)[None, :] + drift + (k * 37)[:, None]) // L) % 1024
    n = _NZ[((k * 7 + seed) % 128)[:, None], sx]
    dens = (0.05 + 0.07 * (k / h2))[:, None] * ripple
    hi = n < dens
    lo = (n > 1 - dens * 1.6) & (k[:, None] % 2 == 1)
    out[hi] = out[hi] * 0.35 + np.array((236, 250, 255), np.float32) * 0.65
    out[lo] = out[lo] * 0.80
    c.a[wy:] = np.clip(out, 0, 255).astype(np.uint8)
    # sun glitter: a column of twinkles under the sun + scattered ones
    for i in range(glitter):
        u = hash01(seed, i, 1)
        gx = int(sun_x + (u - 0.5) * (70 + 160 * hash01(seed, i, 4)))
        gy = wy + 2 + int(hash01(seed, i, 2) * (h2 - 4))
        ph = hash01(seed, i, 3)
        t = ((f / (14 + 10 * ph)) + ph) % 1.0
        if t < 0.35:
            s = 1 if t < 0.1 or t > 0.25 else 2
            c.px(gx, gy, 'white')
            if s == 2:
                for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
                    c.px(gx + dx, gy + dy, '#d8f4ff')


# ================================================================== SHOT A: skyline from the harbour
A_WY = 124             # waterline (screen)
A_P = dict(far=0.12, back=0.32, main=0.55, quay=0.78, boat=1.0, fg=1.45)
CRYS_X = 204           # Crystaller centre in the main layer (screen x at f 72)


def cam_a(f):
    return cruise_cam(f, 72, T_B - 1, 84)


@lru_cache(maxsize=None)
def a_far():
    """Mainland coast, the torch-column islet, distant hills."""
    cv = layer(380)
    # far hills of the mainland (very hazy)
    for x0, x1, hh, cc in ((0, 120, 9, '#a9cfe6'), (240, 380, 12, '#a4cbe4'), (90, 200, 5, '#b4d6ea')):
        for x in range(x0, x1):
            u = (x - x0) / (x1 - x0)
            yy = A_WY - 3 - int(hh * math.sin(u * math.pi) ** 0.7 * (0.8 + 0.2 * math.sin(x * 0.3)))
            cv.vline(x, yy, A_WY - 1, cc)
    # tiny distant towers on the mainland shores
    for i in range(14):
        x = 250 + int(hash01(5, i) * 120)
        hh = 3 + int(hash01(5, i, 2) * 6)
        cv.rect(x, A_WY - 3 - hh, 2 + (i % 2), hh, '#9cc2dc')
    # monument islet: rocky islet, star-shaped fort base, pedestal, an abstract torch column (patina
    # shaft, capital, gold bowl and flame; no figure: project rule, no character statues)
    ix = 24
    cv.ellipse(ix, A_WY - 1, 13, 2, '#8fb5a0')
    cv.rect(ix - 9, A_WY - 4, 18, 3, '#a9b9b0')
    cv.rect(ix - 4, A_WY - 9, 8, 5, '#b8c4be')
    cv.rect(ix - 2, A_WY - 11, 4, 2, '#a9b9b0')
    for x, cc in ((ix - 1, '#7fc0b0'), (ix, '#5fa898'), (ix + 1, '#4f9888')):   # shaft: lit left, shaded right
        cv.vline(x, A_WY - 18, A_WY - 12, cc)
    cv.hline(ix - 2, ix + 2, A_WY - 19, '#8fc8b8')                              # capital
    cv.hline(ix - 1, ix + 1, A_WY - 20, 'gold')                                 # torch bowl + flame
    cv.px(ix, A_WY - 21, 'butter')
    cv.px(ix, A_WY - 22, 'white')
    return to_rgba(cv)


@lru_cache(maxsize=None)
def a_back():
    """Hazy back row of towers (atmospheric perspective)."""
    cv = layer(400)
    meta = {}
    specs = [(104, 12, 50, 'slate', 'flag'), (118, 14, 70, 'sage', 'step'), (134, 10, 58, 'beige', 'mast'),
             (146, 16, 84, 'teal', 'pyramid'), (166, 12, 62, 'cream', 'gold'), (236, 14, 72, 'sage', 'gold'),
             (252, 12, 90, 'slate', 'step'), (266, 16, 66, 'cream', 'pyramid'), (286, 12, 56, 'teal', 'mast'),
             (302, 14, 48, 'beige', 'tank'), (318, 10, 40, 'slate', 'flag'), (332, 14, 32, 'sage', 'step')]
    for i, (x, w, h, fc, cr) in enumerate(specs):
        tower(cv, x, A_WY - 4, w, h, fc, cr, seed=300 + i, tiers=2 if h > 35 else 1, haze=0.55, meta=meta)
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def a_main():
    """The Art-Deco cluster with the Crystaller Building."""
    cv = layer(420)
    meta = {}
    base = A_WY - 6
    back = [  # x, w, h, facade, crown, tiers
        (92, 12, 22, 'sand', 'tank', 1), (104, 14, 34, 'teal', 'step', 2), (118, 16, 46, 'beige', 'gold', 2),
        (134, 12, 40, 'rose', 'mast', 2), (146, 20, 64, 'teal', 'step', 3), (166, 12, 50, 'slate', 'pyramid', 2),
        (178, 14, 74, 'terra', 'gold', 3), (218, 14, 56, 'sage', 'dome', 2), (232, 20, 80, 'teal', 'gold', 3),
        (252, 12, 52, 'blush', 'pyramid', 2), (264, 16, 66, 'beige', 'step', 3), (280, 12, 44, 'slate', 'mast', 2),
        (292, 18, 58, 'cream', 'gold', 3), (310, 14, 40, 'terra', 'step', 2), (324, 16, 50, 'sage', 'pyramid', 2),
        (340, 14, 34, 'beige', 'dome', 2), (354, 18, 42, 'slate', 'step', 2), (372, 12, 28, 'rose', 'tank', 1),
        (384, 16, 30, 'teal', 'flag', 1), (400, 12, 20, 'sand', 'tank', 1), (412, 14, 14, 'beige', 'flag', 1),
    ]
    for i, (x, w, h, fc, cr, t) in enumerate(back):
        tower(cv, x, base, w, h, fc, cr, seed=500 + i, tiers=t, haze=0.18, meta=meta)
    crystaller(cv, CRYS_X, base, meta=meta)
    front = [
        (88, 18, 12, 'terra', 'tank'), (104, 14, 18, 'beige', 'flag'), (120, 22, 14, 'rose', 'tank'),
        (140, 16, 24, 'cream', 'step'), (160, 20, 18, 'terra', 'flag'), (180, 10, 14, 'slate', 'tank'),
        (222, 18, 20, 'beige', 'tank'), (244, 22, 14, 'terra', 'flag'), (268, 16, 24, 'teal', 'step'),
        (288, 20, 18, 'beige', 'tank'), (312, 18, 12, 'terra', 'flag'), (332, 14, 20, 'cream', 'step'),
        (350, 20, 14, 'rose', 'tank'), (372, 18, 10, 'sand', 'flag'), (392, 20, 8, 'terra', 'tank'),
    ]
    for i, (x, w, h, fc, cr) in enumerate(front):
        tower(cv, x, base + 2, w, h, fc, cr, seed=700 + i, tiers=1, meta=meta)
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def a_quay():
    """Quay wall, sheds, piers with pilings, crane towers, crates, lamps, the green tip park."""
    cv = layer(440)
    meta = {'cranes': [], 'flags': [], 'lamps': []}
    wy = A_WY
    # quay wall
    Q0 = 92
    cv.rect(Q0, wy - 6, 440 - Q0, 6, '#b9ad98')
    cv.hline(Q0, 439, wy - 6, '#e4dac4')
    cv.hline(Q0, 439, wy - 1, '#7c705e')
    cv.rect(Q0, wy - 6, 2, 6, '#8c806c')
    for x in range(Q0 + 2, 440, 6):
        cv.vline(x + (2 if (x // 6) % 2 else 0), wy - 5, wy - 2, '#a69a86')
    # sheds / pier houses
    sheds = [(110, 30, 8, '#d7c3a0', '#3f8f8a'), (150, 44, 10, '#cdbb96', '#5f7f9a'),
             (240, 38, 9, '#e8d8b8', '#3f8f8a'), (318, 42, 8, '#d7c3a0', '#b5654a'), (392, 40, 9, '#e8d8b8', '#5f7f9a')]
    for x, w, h, wall, roof in sheds:
        cv.rect(x, wy - 6 - h, w, h, wall)
        cv.rect(x + w - 3, wy - 6 - h, 3, h, shade(wall, 0.8))
        # arched cargo doors
        for dx in range(3, w - 5, 7):
            cv.rect(x + dx, wy - 6 - h + 3, 4, h - 3, shade(wall, 0.55))
            cv.hline(x + dx, x + dx + 3, wy - 6 - h + 2, shade(wall, 0.7))
            cv.px(x + dx + 1, wy - 6 - h + 4, '#e9c46a')
        # gable roof
        cv.poly([(x - 1, wy - 6 - h), (x + w, wy - 6 - h), (x + w - 4, wy - 10 - h), (x + 3, wy - 10 - h)], roof)
        cv.hline(x + 3, x + w - 4, wy - 10 - h, mix(roof, 'white', 0.4))
        cv.hline(x - 1, x + w, wy - 6 - h + 1, shade(wall, 0.6))
        meta['flags'].append((x + w // 2 + 1, wy - 16 - h))
        cv.vline(x + w // 2, wy - 16 - h, wy - 11 - h, '#d8d4cc')
    # piers: wooden decks reaching toward us, with pilings into the water
    for x, w in ((100, 12), (190, 22), (270, 18), (350, 22)):
        cv.rect(x, wy - 2, w, 3, '#9a6a48')
        cv.hline(x, x + w - 1, wy - 2, '#c89a68')
        for px_ in range(x + 1, x + w, 4):
            cv.vline(px_, wy + 1, wy + 3, '#5e3c2c')
        cv.px(x + 2, wy - 3, '#5e3c2c')
        cv.px(x + w - 3, wy - 3, '#5e3c2c')
    # crates and barrels on the quay
    for i in range(18):
        x = Q0 + 4 + int(hash01(41, i) * 336)
        cc = ('#c86a4a', '#3f8f8a', '#e9b44a', '#5f7f9a', '#a85a82')[i % 5]
        cv.rect(x, wy - 9, 4, 3, cc)
        cv.hline(x, x + 3, wy - 9, mix(cc, 'white', 0.35))
        cv.px(x + 3, wy - 7, shade(cc, 0.7))
    # crane gantries (towers + cab); the booms are drawn per frame
    for x in (204, 300):
        cv.vline(x, wy - 30, wy - 7, '#b84a3a')
        cv.vline(x + 3, wy - 30, wy - 7, '#b84a3a')
        for yy in range(wy - 30, wy - 7, 3):
            cv.line(x, yy, x + 3, yy + 3, '#8a3428')
        cv.rect(x - 1, wy - 34, 6, 4, '#d8604a')
        cv.px(x + 3, wy - 33, '#bfe8ff')
        meta['cranes'].append((x + 2, wy - 33))
    # quay lamps
    for x in range(100, 440, 34):
        cv.vline(x, wy - 13, wy - 7, '#4a5a6a')
        cv.rect(x - 1, wy - 15, 3, 2, '#4a5a6a')
        cv.px(x, wy - 14, '#fff4c8')
    # island tip park: round trees
    for i, x in enumerate((404, 414, 426, 434)):
        r = 5 + (i % 2)
        cv.circle(x, wy - 9 - r // 2, r, '#3f9a5a')
        cv.circle(x - 1, wy - 10 - r // 2, r - 2, '#5fba6a')
        cv.px(x - 2, wy - 12 - r // 2, '#9fe08a')
    return to_rgba(cv), meta


FERRY = Sprite.from_ascii('''
    .......ss........
    .......sS........
    ...wwwwwwwwwww...
    ...wgwgwgwgwgw...
  .wwwwwwwwwwwwwwwww.
  .wGwGwGwGwGwGwGwGw.
  ooooooooooooooooooo
  .OOOOOOOOOOOOOOOOO.
  ..rrrrrrrrrrrrrrr..
''', {'w': 'white', 'g': '#5a7fb5', 'G': '#34497a', 'o': 'orange', 'O': '#e8742a', 'r': '#a8441e',
      's': '#f2c24a', 'S': '#c8963a'})

TUG = Sprite.from_ascii('''
    ...ss.....
    ...sS.....
    .wwwwww...
    .wGwGww...
  rrrrrrrrrrrr
  .RRRRRRRRRRk
  ..kkkkkkkk..
''', {'w': 'cream', 'G': '#34497a', 'r': '#d8443a', 'R': '#a8302a', 'k': '#3a3046',
      's': '#3a3046', 'S': '#6a6070'})

SAIL = Sprite.from_ascii('''
  ...w....
  ...ww...
  ...wwP..
  ...wwwP.
  ...wwwwP
  ...m....
  bbbbbbbb
  .BBBBBB.
''', {'w': 'white', 'P': '#d8e4f0', 'm': '#9a6a48', 'b': '#3f8f8a', 'B': '#2a6a6a'})


def boat(c, spr, x, y, f, flip=False, wake=True, seed=0):
    """A vessel on the water with a bob, a short dark reflection and a foamy wake."""
    bob = 1 if (f // 12 + seed) % 2 == 0 else 0
    x, y = int(x), int(y)
    w, h = spr.w, spr.h
    # reflection: flipped, darkened copy under the hull
    arr = spr.rgba[::-1].copy()
    m = arr[..., 3] > 0
    arr[..., :3][m] = (arr[..., :3][m].astype(np.float32) * 0.45 + np.array((40, 90, 140)) * 0.55).astype(np.uint8)
    c.blit(arr[: max(2, h // 2)], x, y + 1 + bob, flip=flip)
    spr.draw(c, x, y - h + 1 + bob, flip=flip)
    if wake:
        sx = x + (w if flip else 0)  # wake trails behind (flip = moving left -> wake to the right)
        for i in range(18):
            t = i / 18
            wx = sx + (i * 2 if flip else -i * 2)
            if hash01(seed, i, f // 3) < 0.85 - t * 0.6:
                c.px(wx, y + (i % 2), 'white')
            if i < 10 and hash01(seed, i, f // 3, 2) < 0.6 - t:
                c.px(wx, y + 1 + (i % 3) // 2, '#d8f4ff')
        # bow wave
        bx = x + (0 if flip else w - 1)
        c.px(bx + (-1 if flip else 1), y, 'white')
        c.px(bx + (-2 if flip else 2), y + 1, '#d8f4ff')


def smoke_puffs(c, x, y, f, n=5, drift=0.35, seed=0, color='#e4e8ee', dark='#b8bcc8'):
    for i in range(n):
        period = 40
        t = ((f + i * period / n + hash01(seed, i) * 5) % period) / period
        px = int(x + t * period * drift + math.sin(t * 6 + i) * 1.5)
        py = int(y - t * 16)
        r = 1 + int(t * 3)
        if t < 0.92:
            c.circle(px, py, r, dark)
            c.circle(px - (1 if r > 1 else 0), py - (1 if r > 1 else 0), max(1, r - 1), color)


def flag(c, x, y, f, color='#e83b3b', seed=0, w=4):
    ph = (f // 4 + seed) % 3
    for i in range(w):
        dy = (1 if (i + ph) % 3 == 0 else 0)
        c.vline(x + i, y + dy, y + 1 + dy, color if i < w - 1 else shade(color, 0.8))


def glint(c, x, y, f, ph, period=60):
    t = ((f / period) + ph) % 1.0
    if t < 0.07:
        c.px(x, y, 'white')
        c.px(x + 1, y, '#d8f4ff')
    elif t < 0.12:
        c.px(x, y, 'white')
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c.px(x + dx, y + dy, 'white')
        c.px(x + 2, y, '#d8f4ff')
        c.px(x - 2, y, '#d8f4ff')
        c.px(x, y - 2, '#d8f4ff')


A_CLOUDS = ((40, 26, 40, 0.10, 1), (190, 48, 30, 0.08, 2), (330, 18, 52, 0.12, 3), (470, 40, 34, 0.09, 4))


@lru_cache(maxsize=None)
def a_fg():
    """Foreground: dark pilings with rope at the bottom-right, a bit of dock planking."""
    cv = layer(500)
    for x0 in (360, 372, 470):
        cv.rect(x0, 146, 7, 40, '#6e4a34')
        cv.rect(x0 + 5, 146, 2, 40, '#4e3226')
        cv.vline(x0, 146, 179, '#9a6a48')
        cv.ellipse(x0 + 3, 146, 3, 1, '#c89a68')
        cv.hline(x0 + 1, x0 + 5, 146, '#e0b884')
        for yy in (156, 157, 160, 161):
            cv.hline(x0, x0 + 6, yy, '#d8c08a' if yy % 2 == 0 else '#b89a68')
    cv.rect(352, 166, 140, 6, '#8a5a3a')
    cv.hline(352, 491, 166, '#c08a5a')
    cv.hline(352, 491, 171, '#5e3c2c')
    for x in range(356, 492, 9):
        cv.vline(x, 167, 170, '#6e4a34')
    return to_rgba(cv)


def shot_a(f):
    X = cam_a(f)
    c = sky_layer().copy()
    draw_clouds(c, f, A_CLOUDS, ox=X * 0.05)
    far = a_far()
    paste(c, far, -int(X * A_P['far']))
    back, bmeta = a_back()
    paste(c, back, -int(X * A_P['back']))
    main, meta = a_main()
    mo = -int(X * A_P['main'])
    paste(c, main, mo)
    # beacons / flags / glints on the towers
    for (x, y) in meta.get('beacons', []):
        if (f // 9) % 4 == 0:
            c.px(x + mo, y, '#ff5a4a')
    for i, (x, y) in enumerate(meta.get('flags', [])):
        flag(c, x + mo, y, f, ('#e83b3b', '#3f6fc8', '#f2c24a', '#3f9a86')[i % 4], seed=i)
    for (x, y, ph) in meta.get('glints', []):
        glint(c, x + mo, y, f, ph, period=72)
    cx, ty, _ = meta['crystaller']
    if f % 18 < 6:                                   # the gold needle glints on every beat
        fx.sparkle(c, cx + mo, ty, f % 18, 'white', period=12, size=3)
    # quay
    qmeta = a_quay()[1]
    qo = -int(X * A_P['quay'])
    paste(c, a_quay()[0], qo)
    for i, (x, y) in enumerate(qmeta['flags']):
        flag(c, x + qo, y, f, ('#e83b3b', '#3f6fc8', '#f2c24a')[i % 3], seed=i + 3, w=3)
    for i, (x, y) in enumerate(qmeta['cranes']):
        a = math.radians(-12 + 10 * math.sin(f / 40 + i * 2.1)) + (math.pi if i % 2 else 0)
        L = 22
        ex, ey = x + qo + int(round(math.cos(a) * L)), y + int(round(math.sin(a) * L * 0.25)) - 2
        c.line(x + qo, y - 1, ex, ey, '#d8604a')
        c.line(x + qo, y - 3, ex, ey, '#b84a3a')
        bx = x + qo - int(round(math.cos(a) * 6))
        c.rect(bx - 1, y - 2, 3, 3, '#5a5a6a')
        hl = 8 + int(3 * math.sin(f / 23 + i))
        c.vline(ex, ey + 1, ey + hl, '#5a5a6a')
        c.rect(ex - 1, ey + hl, 3, 2, ('#3f8f8a', '#e9b44a', '#c86a4a')[i % 3])
    # water + reflections
    water(c, f, A_WY, ['#7cc0e8', '#4f9ad0', '#3a7cb8', '#2e66a4'], refl=0.40, seed=1, src_y0=10)
    # vessels (world x at boat layer)
    sail_x = 250 - (f - 72) * 0.25 - X * 0.85
    boat(c, SAIL, sail_x, A_WY + 4, f, flip=True, seed=5)
    fer2_x = 300 - (f - 72) * 0.55 - X * 0.9
    boat(c, FERRY.recolor({col('orange'): col('#3f8f8a'), col('#e8742a'): col('#2f7a7a')}), fer2_x, A_WY + 9, f,
         flip=True, seed=3)
    fer1_x = 96 + (f - 72) * 0.75 - X * 0.95
    boat(c, FERRY, fer1_x, A_WY + 16, f, seed=1)
    smoke_puffs(c, int(fer1_x) + 8, A_WY + 6, f, n=4, drift=-0.2, seed=2)
    tug_x = 150 + (f - 72) * 0.95 - X * 1.1
    boat(c, TUG, tug_x, A_WY + 38, f, seed=9)
    smoke_puffs(c, int(tug_x) + 4, A_WY + 30, f + 7, n=6, drift=-0.25, seed=4, color='#d8d8e0', dark='#9a96a8')
    # channel buoy (bobbing, with a blinking light)
    bx = 168 - int(X * 1.15)
    by = 158 + (1 if (f // 15) % 2 else 0)
    c.rect(bx - 2, by - 6, 5, 6, '#d8443a')
    c.rect(bx + 1, by - 6, 2, 6, '#a8302a')
    c.hline(bx - 2, bx + 2, by - 4, 'white')
    c.vline(bx, by - 10, by - 7, '#5a5a6a')
    c.px(bx, by - 11, '#ffe25a' if (f // 9) % 3 == 0 else '#b8a040')
    c.hline(bx - 4, bx + 4, by, '#2e66a4')
    c.hline(bx - 3, bx + 3, by + 1, '#d8f4ff' if (f // 6) % 2 else '#7cc0e8')
    # balloon drifting over the harbour, right of the Crystaller
    draw_balloon(c, 262 - int(X * 0.4) + int((f - 72) * 0.18), 62, f, size='s')
    # foreground pilings
    paste(c, a_fg(), -int(X * A_P['fg']))
    # labels
    cx, ty, cy = meta['crystaller']
    callout(c, cx + mo + 5, ty + 24, 'CRYSTALLER BUILDING', f, 80, 147, side='right', length=12)
    callout(c, 300 + mo, 92, 'MANEHATTAN', f, 90, 147, side='down', length=14)
    return c


# ================================================================== SHOT B: Bridleway at street level
B_P = dict(far=0.25, el=0.6, st=1.0, fg=1.4)
B_SW = 140          # sidewalk top = facade base
B_CURB = 152

ICONS = {
    'star': ('..#..', '.###.', '#####', '.###.', '.#.#.'),
    'note': ('..##.', '..#.#', '..#..', '.##..', '.##..'),
    'heart': ('.#.#.', '#####', '#####', '.###.', '..#..'),
    'hat': ('.###.', '.###.', '.###.', '.+++.', '#####'),
    'crown': ('#.#.#', '#####', '#####', '.....', '.....'),
    'spark': ('..#..', '..#..', '##+##', '..#..', '..#..'),
}


def icon(cv, name, x, y, color, hi='white', scale=1):
    for j, row in enumerate(ICONS[name]):
        for i, ch in enumerate(row):
            if ch != '.':
                cc = color if ch == '#' else hi
                if scale == 1:
                    cv.px(x + i, y + j, cc)
                else:
                    cv.rect(x + i * scale, y + j * scale, scale, scale, cc)


def cam_b(f):
    return cruise_cam(f, T_B, T_C - 1, 92)


@lru_cache(maxsize=None)
def b_far():
    cv = layer(420)
    meta = {}
    specs = [(10, 16, 70, 'slate', 'step', 2), (30, 12, 84, 'sage', 'pyramid', 3), (48, 18, 60, 'beige', 'gold', 2),
             (70, 14, 92, 'teal', 'step', 3), (90, 12, 66, 'cream', 'mast', 2), (106, 16, 78, 'slate', 'gold', 3),
             (176, 14, 70, 'sage', 'dome', 2), (194, 18, 96, 'teal', 'gold', 3), (216, 12, 72, 'beige', 'pyramid', 2),
             (232, 16, 82, 'slate', 'step', 3), (254, 12, 64, 'cream', 'mast', 2), (270, 18, 88, 'sage', 'gold', 3),
             (292, 14, 70, 'teal', 'pyramid', 2), (312, 16, 60, 'beige', 'step', 2), (334, 12, 76, 'slate', 'mast', 2),
             (352, 18, 64, 'cream', 'gold', 2), (376, 14, 70, 'sage', 'step', 2), (396, 16, 58, 'teal', 'tank', 2)]
    for i, (x, w, h, fc, cr, t) in enumerate(specs):
        tower(cv, x, 118, w, h, fc, cr, seed=1300 + i, tiers=t, haze=0.5, meta=meta)
    crystaller(cv, 150, 132, meta=meta, haze=0.38)
    return to_rgba(cv), meta


def girder(cv, x0, x1, y, h, c1, c2, c3):
    """Riveted steel girder with an X lattice web."""
    cv.rect(x0, y, x1 - x0, h, c2)
    cv.hline(x0, x1 - 1, y, c1)
    cv.hline(x0, x1 - 1, y + h - 1, c3)
    for x in range(x0, x1, h - 1):
        cv.line(x, y + 1, x + h - 3, y + h - 2, c3)
        cv.line(x + h - 3, y + 1, x, y + h - 2, c3)
        cv.vline(x, y + 1, y + h - 2, c1)


@lru_cache(maxsize=None)
def b_el():
    """Elevated railway viaduct behind the theatre row (deck y 52-59)."""
    cv = layer(420)
    c1, c2, c3 = col('#7aa898'), col('#4f7a6c'), col('#2f4a44')
    girder(cv, 0, 420, 52, 8, c1, c2, c3)
    cv.hline(0, 419, 51, '#8a8478')            # ballast / rail
    cv.hline(0, 419, 50, '#b8b4ac')
    for x in range(0, 420, 4):
        cv.px(x, 51, '#5e3c2c')
    for x in range(14, 420, 58):            # lattice columns down to the street
        cv.rect(x, 60, 4, 82, c2)
        cv.vline(x, 60, 141, c1)
        cv.vline(x + 3, 60, 141, c3)
        for yy in range(62, 140, 6):
            cv.line(x, yy, x + 3, yy + 3, c3)
        cv.rect(x - 2, 59, 8, 2, c1)
    return to_rgba(cv)


@lru_cache(maxsize=None)
def el_train():
    n = 4
    L = 30
    cv = layer(n * (L + 2) + 2, 16)
    for k in range(n):
        x = k * (L + 2)
        body = col('#3f9a7a') if k % 2 == 0 else col('#e07a5a')
        cv.round_rect(x, 2, L, 11, shade(body, 0.6), r=2)
        cv.rect(x + 1, 3, L - 2, 9, body)
        cv.hline(x + 2, x + L - 3, 2, '#8a96a6')
        cv.hline(x + 1, x + L - 2, 3, mix(body, 'white', 0.35))
        cv.rect(x + 1, 5, L - 2, 4, 'cream')
        for wx in range(x + 3, x + L - 4, 5):
            cv.rect(wx, 5, 3, 3, '#5a86b8')
            cv.px(wx, 5, '#c6e6ff')
        cv.rect(x + L // 2 - 1, 5, 3, 6, shade(body, 0.7))
        cv.hline(x + 1, x + L - 2, 11, shade(body, 0.7))
        for wx in (x + 5, x + L - 6):
            cv.circle(wx, 13, 2, '#3a3046')
            cv.px(wx, 13, '#8a8a9a')
        if k < n - 1:
            cv.rect(x + L, 7, 2, 2, '#3a3046')
    # headlight on the front (right end)
    cv.px(n * (L + 2) - 3, 9, '#fff4b0')
    return to_rgba(cv)


def theatre(cv, meta, x, w, top, wall, trim, accent, board, icons, seed, blade=None):
    wall, trim, accent = col(wall), col(trim), col(accent)
    sw = B_SW
    cv.rect(x, top, w, sw - top, wall)
    cv.rect(x + w - 3, top, 3, sw - top, shade(wall, 0.8))
    cv.vline(x, top, sw - 1, mix(wall, 'white', 0.4))
    # pilasters + arched windows on the upper storeys
    for px_ in range(x + 4, x + w - 4, 12):
        cv.rect(px_, top + 6, 2, sw - top - 40, mix(wall, 'white', 0.22))
        cv.vline(px_ + 2, top + 6, sw - 35, shade(wall, 0.85))
    for wy in range(top + 9, sw - 40, 12):
        for wx in range(x + 8, x + w - 8, 12):
            cv.rect(wx, wy, 5, 8, '#34497a')
            cv.hline(wx + 1, wx + 3, wy - 1, '#34497a')
            cv.px(wx + 1, wy, '#86b6e6')
            cv.vline(wx + 1, wy + 1, wy + 4, '#5a7fb5')
            cv.hline(wx, wx + 4, wy, shade(wall, 0.55))           # noon shadow under the lintel
            cv.hline(wx - 1, wx + 5, wy + 8, trim)                  # sill
            cv.px(wx + 2, wy - 2, accent)                           # keystone
    # cornice with dentils + gold parapet ornament, crisp shadow straight down
    cv.rect(x - 2, top, w + 4, 3, trim)
    cv.hline(x - 2, x + w + 1, top, 'white')
    for dx in range(0, w + 4, 2):
        cv.px(x - 2 + dx, top + 3, shade(trim, 0.7))
    cv.hline(x, x + w - 1, top + 4, shade(wall, 0.6))
    cx = x + w // 2
    cv.rect(cx - 10, top - 6, 20, 6, trim)
    for k in range(5):
        cv.line(cx, top - 1, cx - 8 + k * 4, top - 5, 'gold')
    cv.hline(cx - 10, cx + 9, top - 6, 'white')
    cv.circle(cx, top - 1, 2, 'gold')
    cv.px(cx, top - 2, 'butter')
    # ground floor: rusticated band, recessed lobby with glass doors, poster frames
    cv.rect(x, sw - 34, w, 34, shade(wall, 0.88))
    for yy in range(sw - 30, sw, 4):
        cv.hline(x, x + w - 1, yy, shade(wall, 0.78))
    ew = min(36, w - 28)
    ex = cx - ew // 2
    cv.rect(ex, sw - 20, ew, 20, '#3a3046')
    cv.rect(ex + 1, sw - 19, ew - 2, 2, '#5e4a52')
    for dx in range(ex + 2, ex + ew - 6, 7):
        cv.rect(dx, sw - 15, 5, 15, 'gold')
        cv.rect(dx + 1, sw - 14, 3, 14, '#4b6ea8')
        cv.vline(dx + 1, sw - 14, sw - 9, '#a8d4f5')
    cv.rect(ex - 1, sw - 21, ew + 2, 1, 'gold')
    for k, px_ in enumerate((x + 5, ex - 11, ex + ew + 3, x + w - 13)):
        if px_ < x + 3 or px_ + 8 > x + w - 2:
            continue
        cv.rect(px_, sw - 17, 8, 12, 'gold')
        bgc = ('#3a6cc4', '#c83a6a', '#2f7a7a', '#7a3a9a')[(k + seed) % 4]
        cv.rect(px_ + 1, sw - 16, 6, 10, bgc)
        icon(cv, icons[(k + seed) % len(icons)], px_ + 1, sw - 14, 'butter')
    # marquee (front face + underside), with its noon shadow on the sidewalk
    mx0, mx1 = x + 6, x + w - 6
    my = sw - 40
    cv.rect(mx0 + 2, my, mx1 - mx0 - 4, 1, 'gold')
    cv.rect(mx0, my + 1, mx1 - mx0, 14, 'gold')
    cv.rect(mx0 + 1, my + 3, mx1 - mx0 - 2, 10, shade(accent, 0.9))
    cv.rect(mx0 + 3, my + 4, mx1 - mx0 - 6, 8, board)
    nic = max(1, (mx1 - mx0 - 10) // 9)
    gx0 = (mx0 + mx1) // 2 - (nic * 9 - 4) // 2
    for i in range(nic):
        icon(cv, icons[(i + seed) % len(icons)], gx0 + i * 9, my + 5, accent if i % 2 else '#3a3046',
             hi='gold')
    cv.rect(mx0 + 1, my + 15, mx1 - mx0 - 2, 3, shade(accent, 0.55))
    cv.hline(mx0 + 1, mx1 - 2, my + 18, '#3a3046')
    for dx in range(mx0 + 4, mx1 - 3, 6):
        cv.px(dx, my + 17, '#fff4b0')
    for dx in range(mx0 + 1, mx1 - 1, 2):
        meta['bulbs'].append((dx, my + 1, len(meta['bulbs'])))
        meta['bulbs'].append((dx, my + 14, len(meta['bulbs'])))
    for dy in range(my + 3, my + 14, 2):
        meta['bulbs'].append((mx0, dy, len(meta['bulbs'])))
        meta['bulbs'].append((mx1 - 1, dy, len(meta['bulbs'])))
    meta['marquees'].append((mx0, mx1, my))
    # vertical blade sign
    if blade is not None:
        bx = blade
        by0, by1 = top + 8, my - 2
        cv.rect(bx, by0, 11, by1 - by0, 'gold')
        cv.rect(bx + 1, by0 + 1, 9, by1 - by0 - 2, accent)
        cv.rect(bx + 2, by0 + 2, 7, by1 - by0 - 4, shade(accent, 0.7))
        cv.tri((bx, by0), (bx + 10, by0), (bx + 5, by0 - 7), 'gold')
        cv.px(bx + 5, by0 - 4, 'white')
        cv.rect(bx + 3, by1, 5, 2, '#5a5a6a')
        slots = []
        for k, yy in enumerate(range(by0 + 4, by1 - 6, 8)):
            slots.append((bx + 3, yy, icons[(k + seed + 1) % len(icons)]))
        meta['blades'].append((bx, by0, by1, slots))
        for yy in range(by0 + 1, by1 - 1, 2):
            meta['bulbs'].append((bx, yy, len(meta['bulbs'])))
            meta['bulbs'].append((bx + 10, yy, len(meta['bulbs'])))
    return cx, my


def shopfront(cv, meta, x, w, top, wall, awn, sign_icon, seed, escape=True, tank=False):
    wall = col(wall)
    sw = B_SW
    cv.rect(x, top, w, sw - top, wall)
    # brick texture
    for yy in range(top + 2, sw - 22, 3):
        for xx in range(x + (yy // 3) % 2 * 2, x + w, 4):
            cv.px(xx, yy, shade(wall, 0.86))
    cv.vline(x, top, sw - 1, mix(wall, 'white', 0.3))
    cv.rect(x - 1, top, w + 2, 2, mix(wall, 'white', 0.5))
    cv.hline(x, x + w - 1, top + 2, shade(wall, 0.6))
    for wy in range(top + 6, sw - 28, 11):
        for wx in range(x + 4, x + w - 5, 8):
            cv.rect(wx, wy, 4, 7, '#34497a')
            cv.vline(wx, wy, wy + 6, '#5a7fb5')
            cv.px(wx + 1, wy + 1, '#a8d4f5')
            cv.hline(wx - 1, wx + 4, wy + 7, mix(wall, 'white', 0.5))
            cv.hline(wx, wx + 3, wy, shade(wall, 0.5))
            if hash01(seed, wx, wy) < 0.4:                     # flower boxes
                cv.hline(wx - 1, wx + 4, wy + 8, '#7a4a2a')
                for k in range(0, 5, 2):
                    cv.px(wx - 1 + k, wy + 7, ('#f04f78', '#f9c22b', '#ffffff')[k // 2])
    if escape:
        # fire escape: landings + zig-zag ladders (dark steel)
        fx0, fx1 = x + 2, x + w - 3
        for wy in range(top + 14, sw - 28, 11):
            cv.hline(fx0, fx1, wy, '#3a3046')
            for xx in range(fx0, fx1 + 1, 2):
                cv.px(xx, wy - 1, '#5a5060')
            cv.hline(fx0, fx1, wy - 4, '#5a5060')
            cv.line(fx1 - 6, wy, fx1 - 1, wy + 10, '#3a3046')
    if tank:
        tx = x + w // 2 - 3
        cv.vline(tx, top - 4, top - 1, '#5e3c2c')
        cv.vline(tx + 5, top - 4, top - 1, '#5e3c2c')
        cv.rect(tx, top - 11, 6, 7, '#9a6a48')
        cv.vline(tx + 5, top - 11, top - 5, '#6e4a34')
        cv.hline(tx, tx + 5, top - 8, '#5e3c2c')
        cv.tri((tx - 1, top - 11), (tx + 6, top - 11), (tx + 2, top - 15), '#6e4a34')
    # shop window + door
    cv.rect(x, sw - 24, w, 24, shade(wall, 0.75))
    cv.rect(x + 3, sw - 18, w - 14, 14, '#4b6ea8')
    cv.line(x + 4, sw - 17, x + 9, sw - 6, '#a8d4f5')
    cv.rect(x + 3, sw - 4, w - 14, 4, '#7a4a2a')
    cv.rect(x + w - 9, sw - 18, 6, 18, '#5e3c2c')
    cv.px(x + w - 5, sw - 9, 'gold')
    # hanging sign with an icon
    cv.rect(x + w // 2 - 4, sw - 33, 9, 8, 'cream')
    cv.frame(x + w // 2 - 4, sw - 33, 9, 8, '#5e3c2c')
    icon(cv, sign_icon, x + w // 2 - 2, sw - 32, '#c83a3a')
    # striped scalloped awning
    ay = sw - 24
    for xx in range(x - 1, x + w + 1):
        stripe = ((xx - x) // 3) % 2
        cc = col(awn) if stripe else col('white')
        cv.vline(xx, ay - 4, ay + 1, shade(cc, 0.92))
        if (xx - x) % 3 != 2:
            cv.px(xx, ay + 2, cc)
    cv.hline(x - 1, x + w, ay - 5, shade(awn, 0.7))
    meta['awnings'].append((x - 1, x + w + 1, ay + 2))


@lru_cache(maxsize=None)
def b_street():
    cv = layer(470)
    meta = {'bulbs': [], 'marquees': [], 'blades': [], 'awnings': [], 'manholes': [], 'lamps': [], 'banners': []}
    shopfront(cv, meta, -4, 42, 78, '#b5654a', '#c83a3a', 'note', 1, tank=True)
    theatre(cv, meta, 38, 98, 64, '#e8b4a8', '#fff0e8', '#c83a6a', '#fff4dc', ('star', 'note', 'heart'), 2,
            blade=44)
    shopfront(cv, meta, 138, 34, 84, '#eadbc0', '#2f8f8a', 'heart', 3, escape=False)
    theatre(cv, meta, 174, 114, 54, '#f6ecd4', '#ffffff', '#3a6cc4', '#fff4dc', ('star', 'hat', 'note', 'crown'), 0,
            blade=270)
    shopfront(cv, meta, 290, 24, 74, '#8fa3b8', '#f9c22b', 'spark', 5, tank=True)
    theatre(cv, meta, 316, 96, 60, '#9fd0c8', '#e6f6ee', '#a8447a', '#fff4dc', ('heart', 'note', 'star'), 1,
            blade=396)
    shopfront(cv, meta, 414, 56, 70, '#cf7c5c', '#3a6cc4', 'star', 7)
    # roof billboard over the grand theatre (scaffold + panel; bulbs animated)
    bx, by, bw, bh = 52, 34, 70, 22
    for lx in range(bx + 4, bx + bw - 2, 12):
        cv.vline(lx, by + bh, 61, '#5a5a6a')
        cv.line(lx, by + bh, lx + 6, 61, '#4a4a5a')
    cv.rect(bx, by, bw, bh, 'gold')
    cv.gradient(bx + 1, by + 1, bw - 2, bh - 2, ['#3a6cc4', '#7a5ac8', '#e0609a'], soft=0.8)
    icon(cv, 'star', bx + 5, by + 4, 'gold', hi='butter', scale=3)
    icon(cv, 'hat', bx + 26, by + 6, '#2e2a3a', hi='#e83b5a', scale=2)
    icon(cv, 'note', bx + 42, by + 5, 'white', scale=2)
    icon(cv, 'note', bx + 54, by + 9, 'cream', scale=2)
    meta['billboard'] = (bx, by, bw, bh)
    for dx in range(bx + 1, bx + bw - 1, 2):
        meta['bulbs'].append((dx, by - 1, len(meta['bulbs'])))
    # sidewalk + curb + street
    sw = B_SW
    cv.rect(-1, sw, 472, B_CURB - sw, '#cfcac4')
    for yy in (sw + 4, sw + 8):
        cv.hline(0, 469, yy, '#bcb6b0')
    for xx in range(0, 470, 9):
        cv.vline(xx, sw, B_CURB - 1, '#bcb6b0')
    cv.hline(0, 469, sw, '#a8a29c')
    # noon shadows straight down under marquees and awnings
    for mx0, mx1, my in meta['marquees']:
        cv.dither(mx0 + 1, sw, mx1 - mx0 - 2, 7, '#9a948e', 1.0)
        cv.dither(mx0 + 1, sw + 7, mx1 - mx0 - 2, 2, '#9a948e', 0.5)
    for ax0, ax1, ay in meta['awnings']:
        cv.dither(ax0 + 1, sw, ax1 - ax0 - 2, 4, '#a8a29c', 1.0)
    cv.rect(-1, B_CURB, 472, 2, '#e8e4de')
    cv.rect(-1, B_CURB + 2, 472, 2, '#8a8680')
    cv.rect(-1, B_CURB + 4, 472, H - B_CURB - 4, '#5d6474')
    cv.dither(-1, B_CURB + 4, 472, H - B_CURB - 4, '#666e80', 0.25)
    cv.dither(-1, B_CURB + 4, 472, 3, '#4a505e', 0.6)
    for xx in range(0, 470, 26):
        cv.rect(xx, 171, 12, 2, '#e8dcb0')
    for mxh in (70, 236, 392):
        cv.ellipse(mxh, 164, 6, 2, '#3a3e4a')
        cv.ellipse(mxh, 164, 5, 1, '#4e5464')
        cv.hline(mxh - 3, mxh + 3, 164, '#3a3e4a')
        meta['manholes'].append((mxh, 163))
    # lamp posts at the curb (deco, double globe)
    for lx in (26, 160, 300, 440):
        cv.rect(lx - 1, B_CURB - 2, 4, 2, '#2f4a44')
        cv.vline(lx, 108, B_CURB - 2, '#2f4a44')
        cv.vline(lx + 1, 108, B_CURB - 2, '#4f7a6c')
        cv.hline(lx - 4, lx + 5, 110, '#2f4a44')
        for gx in (lx - 4, lx + 5):
            cv.circle(gx, 113, 2, '#fff4dc')
            cv.px(gx - 1, 112, 'white')
        cv.circle(lx, 105, 2, '#fff4dc')
        cv.px(lx, 102, '#2f4a44')
        meta['banners'].append((lx + 2, 118))
    # hydrant, mailbox, newsstand
    cv.rect(118, 144, 4, 7, '#d8443a')
    cv.rect(117, 146, 6, 2, '#a8302a')
    cv.circle(119, 143, 2, '#d8443a')
    cv.rect(352, 140, 7, 11, '#3a6cc4')
    cv.ellipse(355, 140, 3, 2, '#3a6cc4')
    cv.hline(353, 357, 142, '#2a4c94')
    cv.rect(198, 136, 16, 15, '#3f8f6a')
    cv.rect(196, 133, 20, 3, '#2f6a50')
    for k, cc in enumerate(('#e83b3b', '#f9c22b', '#3a6cc4', '#ffffff')):
        cv.rect(200 + k * 3, 139, 2, 4, cc)
    # parked cab cart (yellow, chequer stripe, no driver, shafts down)
    kx = 330
    cv.rect(kx, 150, 26, 9, '#f2c24a')
    cv.round_rect(kx + 4, 140, 16, 11, '#f2c24a', r=2)
    cv.rect(kx + 6, 142, 6, 5, '#4b6ea8')
    cv.px(kx + 7, 143, '#c6e6ff')
    cv.rect(kx + 14, 142, 4, 5, '#4b6ea8')
    for xx in range(kx, kx + 26, 2):
        cv.px(xx, 154, '#3a3046')
        cv.px(xx + 1, 155, '#3a3046')
    cv.hline(kx, kx + 25, 150, '#fbe8a0')
    cv.line(kx + 26, 156, kx + 40, 166, '#7a4a2a')
    for wx in (kx + 5, kx + 20):
        cv.circle(wx, 160, 4, '#3a3046')
        cv.circle(wx, 160, 2, '#c8963a')
        cv.px(wx, 160, '#3a3046')
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def b_fg():
    cv = layer(520)
    # near curb and the near sidewalk edge
    cv.rect(0, 176, 520, 4, '#cfcac4')
    cv.hline(0, 519, 176, '#e8e4de')
    cv.hline(0, 519, 175, '#8a8680')
    # near lamp posts (dark, partial) + a traffic light pole
    for lx in (380, 505):
        cv.rect(lx, 70, 3, 106, '#253a36')
        cv.vline(lx, 70, 175, '#3f6058')
        cv.rect(lx - 6, 66, 15, 3, '#253a36')
        for gx in (lx - 6, lx + 8):
            cv.circle(gx, 72, 3, '#fff4dc')
            cv.px(gx - 1, 71, 'white')
    tx = 96
    cv.rect(tx, 92, 3, 84, '#3a3e4a')
    cv.vline(tx, 92, 175, '#5a6070')
    cv.rect(tx - 3, 78, 9, 16, '#f2c24a')
    cv.rect(tx - 2, 79, 7, 14, '#3a3e4a')
    return to_rgba(cv)


B_CLOUDS = ((60, 8, 44, 0.10, 11), (250, 22, 30, 0.08, 12), (420, 12, 38, 0.12, 13))


def bulb(c, x, y, lit, flash):
    if flash:
        c.px(x, y, 'white')
    elif lit:
        c.px(x, y, '#fff4b0')
    else:
        c.px(x, y, '#b8862a')


def shot_b(f):
    X = cam_b(f)
    c = sky_layer().copy()
    draw_clouds(c, f, B_CLOUDS, ox=X * 0.08)
    far, fmeta = b_far()
    fo = -int(X * B_P['far'])
    paste(c, far, fo)
    for (x, y, ph) in fmeta.get('glints', []):
        glint(c, x + fo, y, f, ph, period=80)
    for (x, y) in fmeta.get('beacons', []):
        if (f // 9) % 4 == 0:
            c.px(x + fo, y, '#ff7a6a')
    for i, (x, y) in enumerate(fmeta.get('flags', [])):
        flag(c, x + fo, y, f, '#c8d8ea', seed=i, w=3)
    cx, ty, _ = fmeta['crystaller']
    if f % 18 < 6:
        fx.sparkle(c, cx + fo, ty, f % 18, 'white', period=12, size=2)
    # balloon drifting high between the towers
    draw_balloon(c, 236 - int(X * 0.3) + int((f - T_B) * 0.15), 34, f, size='s')
    # elevated railway + the train
    eo = -int(X * B_P['el'])
    paste(c, b_el(), eo)
    tr = el_train()
    tx = -150 + int((f - T_B - 4) * 6.0)
    if -tr.shape[1] < tx + eo < W:
        paste(c, tr, tx + eo, 36)
        if (f // 2) % 5 == 0:                         # sparks on the third rail
            sx = tx + eo + 8 + int(hash01(f, 3) * 100)
            c.px(sx, 51, 'white')
            c.px(sx + 1, 50, '#fff4b0')
    # street layer
    st, meta = b_street()
    so = -int(X * B_P['st'])
    paste(c, st, so)
    step = f // 6
    flash = f % 18 < 2
    for (x, y, i) in meta['bulbs']:
        bulb(c, x + so, y, (i // 2 + step) % 3 != 0, flash)
    bi = (f // 18) % 6
    for k, (bx, by0, by1, slots) in enumerate(meta['blades']):
        for j, (ix, iy, nm) in enumerate(slots):
            on = j <= bi or (f // 9) % 2 == 0 and j == bi + 1
            icon(c, nm, ix + so, iy, '#ffe25a' if on else '#c8963a', hi='white' if on else '#e0c070')
            if on and j == bi:
                c.px(ix + 2 + so, iy + 2, 'white')
    for (lx, ly) in meta['banners']:
        # small hanging street banners flapping on the lamp posts
        ph = (f // 5 + lx) % 3
        cc = ('#c83a6a', '#3a6cc4', '#f9c22b', '#2f8f8a')[(lx // 7) % 4]
        c.rect(lx + so, ly, 4, 9, cc)
        c.px(lx + 3 + so + (1 if ph == 0 else 0), ly + 9, cc)
        c.px(lx + 1 + so, ly + 9, cc)
        c.px(lx + 2 + so, ly + 4, 'white')
    for i, (mx, my) in enumerate(meta['manholes']):
        smoke_puffs(c, mx + so, my, f + i * 13, n=5, drift=0.08, seed=20 + i, color='#f4f6fa', dark='#d0d6e0')
        fx.steam(c, mx + so + 2, my, f + 9, color='#ffffff', n=3, height=14, spread=3, seed=i + 7, speed=0.35)
    # foreground
    paste(c, b_fg(), -int(X * B_P['fg']))
    lx = 96 - int(X * B_P['fg'])
    ph = (f // 36) % 3
    for k, cc in enumerate(('#ff5a4a', '#ffd84a', '#5aff7a')):
        c.px(lx + 1, 81 + k * 4, cc if k == (2 - ph) else shade(cc, 0.35))
        c.px(lx + 2, 81 + k * 4, cc if k == (2 - ph) else shade(cc, 0.35))
    callout(c, 231 + so, 120, 'BRIDLEWAY', f, 168, 237, side='down', length=17)
    return c




# ================================================================== SHOT C: the park + Bronclyn
C_P = dict(far=0.2, brown=0.45, trees=0.72, bank=1.0, fg=1.38)
C_WY = 134
LAKE = ['#9ad8d0', '#6cbcb8', '#4a9ea4', '#357e90']


def cam_c(f):
    return cruise_cam(f, T_C, T_D - 1, 92)


def tree(cv, x, y, r, greens, seed=0):
    """Round-canopy park tree; (x, y) = trunk base."""
    dk, md, lt, hi = (col(g) for g in greens)
    cv.rect(x - 1, y - r - 2, 3, r + 2, '#6e4a34')
    cv.vline(x + 1, y - r - 2, y - 1, '#4e3226')
    cy = y - r - 4
    for k in range(5):
        a = hash01(seed, k) * 6.28
        rr = int(r * (0.55 + 0.25 * hash01(seed, k, 2)))
        ox, oy = int(math.cos(a) * r * 0.45), int(math.sin(a) * r * 0.35)
        cv.circle(x + ox, cy + oy + 1, rr, dk)
    cv.circle(x, cy, r, dk)
    cv.circle(x - 1, cy - 1, r - 1, md)
    cv.circle(x - max(1, r // 4), cy - max(1, r // 3), max(2, r // 2 + 1), lt)
    for k in range(r):
        hx = x - r // 2 + int(hash01(seed, k, 5) * r * 0.9) - r // 3
        hy = cy - r // 2 + int(hash01(seed, k, 6) * r * 0.8) - r // 4
        cv.px(hx, hy, hi)
        cv.px(x - r + 2 + int(hash01(seed, k, 7) * 2 * r * 0.8), cy + r // 3 + int(hash01(seed, k, 8) * r * 0.5),
              shade(dk, 0.85))


GREENS = [('#2f7a4a', '#3f9a5a', '#6cc06a', '#b0e88a'), ('#2a6a50', '#358a62', '#5aae78', '#9ad8a0'),
          ('#3f7a3a', '#5a9a42', '#86c05a', '#c8e888')]


def brownstone(cv, meta, x, w, base, h, wall, seed, escape=False, tank=False, bay=False):
    wall = col(wall)
    top = base - h
    cv.rect(x, top, w, h, wall)
    for yy in range(top + 3, base - 14, 3):
        for xx in range(x + (yy // 3) % 2 * 2, x + w, 4):
            cv.px(xx, yy, shade(wall, 0.9))
    cv.vline(x, top, base - 1, mix(wall, 'white', 0.2))
    cv.vline(x + w - 1, top, base - 1, shade(wall, 0.75))
    # cornice with brackets + its noon shadow
    cv.rect(x - 1, top - 3, w + 2, 3, '#e6d6bc')
    cv.hline(x - 1, x + w, top - 3, '#fff4e0')
    for bx in range(x, x + w, 3):
        cv.px(bx, top, '#c8b49a')
    cv.hline(x, x + w - 1, top + 1, shade(wall, 0.55))
    # rusticated parlour/basement levels
    cv.rect(x, base - 14, w, 14, shade(wall, 0.85))
    cv.hline(x, x + w - 1, base - 14, shade(wall, 1.15))
    # windows (tall, white frames, lintels)
    floors = list(range(top + 5, base - 20, 10))
    for fi, wy in enumerate(floors):
        for k, wx in enumerate((x + 3, x + w - 7) if not bay else (x + 2, x + w // 2 - 2, x + w - 6)):
            hh = 6 if fi else 5
            cv.rect(wx - 1, wy - 1, 6, hh + 2, '#f0e6d4')
            cv.rect(wx, wy, 4, hh, '#34497a')
            cv.vline(wx, wy, wy + hh - 1, '#5a7fb5')
            cv.px(wx + 1, wy, '#a8d4f5')
            cv.hline(wx - 1, wx + 4, wy - 2, '#e6d6bc')
            if hash01(seed, fi, k) < 0.35:
                cv.hline(wx - 1, wx + 4, wy + hh + 1, '#6e4a34')
                cv.px(wx, wy + hh, '#f04f78')
                cv.px(wx + 2, wy + hh, '#f9c22b')
                cv.px(wx + 3, wy + hh, '#5aae78')
    if bay:
        cv.vline(x + 1, top + 4, base - 15, mix(wall, 'white', 0.3))
        cv.vline(x + w - 2, top + 4, base - 15, shade(wall, 0.7))
    # stoop: steps up to the parlour door, with dark iron railings
    dx = x + w - 9
    cv.rect(dx, base - 24, 6, 10, '#5e3c2c')
    cv.rect(dx + 1, base - 23, 4, 9, '#7a4a2a')
    cv.ellipse(dx + 2, base - 24, 3, 2, '#f0e6d4')
    cv.px(dx + 2, base - 25, '#a8d4f5')
    cv.px(dx + 4, base - 19, 'gold')
    for k in range(7):
        sx = dx - 2 - k * 2
        cv.rect(sx, base - 14 + k * 2, 9, 2, '#c8b49a')
        cv.hline(sx, sx + 8, base - 14 + k * 2, '#e6d6bc')
    cv.line(dx - 2, base - 16, dx - 16, base - 2, '#3a3046')
    cv.line(dx - 2, base - 18, dx - 16, base - 4, '#3a3046')
    cv.vline(dx - 16, base - 4, base - 1, '#3a3046')
    # basement window under the stoop
    cv.rect(x + 2, base - 10, 4, 4, '#34497a')
    if escape:
        for wy in floors[1:]:
            cv.hline(x + 1, x + 9, wy + 7, '#3a3046')
            for xx in range(x + 1, x + 10, 2):
                cv.px(xx, wy + 6, '#5a5060')
            cv.hline(x + 1, x + 9, wy + 3, '#5a5060')
            cv.line(x + 8, wy + 7, x + 3, wy + 16, '#3a3046')
    if tank:
        tx = x + 3
        cv.vline(tx, top - 7, top - 4, '#5e3c2c')
        cv.vline(tx + 5, top - 7, top - 4, '#5e3c2c')
        cv.rect(tx, top - 13, 6, 6, '#9a6a48')
        cv.vline(tx + 5, top - 13, top - 8, '#6e4a34')
        cv.hline(tx, tx + 5, top - 10, '#5e3c2c')
        cv.tri((tx - 1, top - 13), (tx + 6, top - 13), (tx + 2, top - 16), '#6e4a34')
    else:
        cv.rect(x + w - 6, top - 7, 3, 4, '#8a4a3a')
        cv.hline(x + w - 7, x + w - 3, top - 7, '#6e3a2c')
        meta['chimneys'].append((x + w - 5, top - 8))
    meta['tops'].append((x, x + w, top))


@lru_cache(maxsize=None)
def c_far():
    cv = layer(380)
    meta = {}
    specs = [(10, 14, 60, 'slate', 'step', 2), (28, 12, 74, 'sage', 'pyramid', 3), (44, 16, 54, 'beige', 'gold', 2),
             (64, 12, 80, 'teal', 'step', 3), (80, 14, 62, 'cream', 'mast', 2), (98, 16, 70, 'slate', 'gold', 3),
             (118, 12, 58, 'beige', 'dome', 2), (240, 14, 66, 'sage', 'dome', 2), (258, 16, 84, 'teal', 'gold', 3),
             (278, 12, 64, 'beige', 'pyramid', 2), (294, 16, 74, 'slate', 'step', 3), (314, 12, 56, 'cream', 'mast', 2),
             (330, 18, 70, 'sage', 'gold', 3), (352, 14, 60, 'teal', 'pyramid', 2)]
    for i, (x, w, h, fc, cr, t) in enumerate(specs):
        tower(cv, x, 104, w, h, fc, cr, seed=2300 + i, tiers=t, haze=0.5, meta=meta)
    crystaller(cv, 196, 124, meta=meta, haze=0.4)
    for i, (x, w, h, fc, cr, t) in enumerate([(140, 16, 54, 'blush', 'step', 2), (160, 12, 66, 'sage', 'gold', 2),
                                              (214, 14, 60, 'cream', 'pyramid', 2)]):
        tower(cv, x, 108, w, h, fc, cr, seed=2400 + i, tiers=t, haze=0.45, meta=meta)
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def c_brown():
    cv = layer(420)
    meta = {'chimneys': [], 'tops': [], 'lines': []}
    walls = ['#8a5a48', '#a0654e', '#7a4a3c', '#b0705a', '#946048', '#9a5a4a']
    x = 2
    i = 0
    base = 120
    while x < 410:
        w = 22 + (i % 3) * 2
        h = 44 + int(hash01(61, i) * 8)
        brownstone(cv, meta, x, w, base, h, walls[i % len(walls)], 900 + i, escape=(i % 3 == 1),
                   tank=(i % 4 == 2), bay=(i % 3 == 2))
        x += w + 1
        i += 1
    # street + sidewalk in front of the row, then the low park wall
    cv.rect(0, base, 420, 3, '#cfcac4')
    cv.rect(0, base + 3, 420, 3, '#7a8090')
    cv.rect(0, base + 6, 420, 4, '#b8ad98')
    cv.hline(0, 419, base + 6, '#e0d6c4')
    for xx in range(0, 420, 7):
        cv.vline(xx, base + 7, base + 9, '#a09484')
    # laundry lines between neighbouring roof/escape levels
    tops = meta['tops']
    for k in (1, 4, 7, 10, 13):
        if k + 1 < len(tops):
            x0 = tops[k][1] - 4
            x1 = tops[k + 1][0] + 6
            meta['lines'].append((x0 - 6, x1 + 4, tops[k][2] + 18 + (k % 2) * 10))
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def c_trees():
    cv = layer(420)
    # back lawn
    cv.rect(0, 124, 420, 12, '#5aae58')
    cv.dither(0, 124, 420, 3, '#3f8a48', 0.5)
    trees = [(8, 128, 16, 0), (30, 131, 12, 1), (52, 127, 18, 2), (76, 130, 13, 0), (96, 131, 10, 1),
             (128, 131, 8, 2), (196, 131, 9, 0),
             (236, 130, 12, 2), (258, 127, 18, 0), (282, 130, 14, 1), (304, 128, 17, 2), (330, 131, 12, 0),
             (350, 127, 18, 1), (376, 130, 14, 2), (398, 128, 17, 0), (418, 131, 12, 1)]
    for i, (x, y, r, g) in enumerate(trees):
        tree(cv, x, y, r, GREENS[g], seed=40 + i)
    # small lamp posts along the path
    for lx in (108, 160, 214):
        cv.vline(lx, 118, 131, '#2f4a44')
        cv.circle(lx, 116, 1, '#fff4dc')
        cv.px(lx, 114, '#2f4a44')
    return to_rgba(cv)


@lru_cache(maxsize=None)
def c_bank():
    """Rock, fountain terrace, boathouse, blossom tree and the stone bank (p 1.0)."""
    cv = layer(460)
    meta = {}
    # stone bank / embankment edge
    cv.rect(0, C_WY - 5, 460, 5, '#b8ad98')
    cv.hline(0, 459, C_WY - 5, '#e0d6c4')
    cv.hline(0, 459, C_WY - 1, '#7c705e')
    for xx in range(0, 460, 8):
        cv.vline(xx + (4 if (xx // 8) % 2 else 0), C_WY - 4, C_WY - 2, '#a09484')
    # the giant rock (granite outcrop)
    pts = [(0, C_WY - 4), (4, C_WY - 22), (14, C_WY - 33), (30, C_WY - 38), (46, C_WY - 34), (58, C_WY - 24),
           (66, C_WY - 12), (70, C_WY - 4)]
    cv.poly(pts, '#8e8a96')
    cv.poly([(6, C_WY - 20), (15, C_WY - 31), (30, C_WY - 36), (44, C_WY - 32), (30, C_WY - 28), (16, C_WY - 22)],
            '#b4b0bc')
    cv.poly([(46, C_WY - 33), (57, C_WY - 23), (65, C_WY - 11), (69, C_WY - 4), (54, C_WY - 4), (48, C_WY - 18)],
            '#6e6a7a')
    cv.line(14, C_WY - 30, 28, C_WY - 35, '#dcd8e2')
    cv.line(20, C_WY - 18, 34, C_WY - 8, '#6e6a7a')
    cv.line(36, C_WY - 26, 42, C_WY - 12, '#6e6a7a')
    for k in range(10):
        mx = 6 + int(hash01(71, k) * 56)
        my = C_WY - 6 - int(hash01(71, k, 2) * 16)
        cv.px(mx, my, '#6a9a52')
        cv.px(mx + 1, my, '#86b85a')
    # fountain terrace: wide basin, pedestal, two bowls, a gold sun emblem on top
    fx0 = 104
    cv.rect(fx0 - 22, C_WY - 9, 44, 5, '#d8ccb4')
    cv.hline(fx0 - 22, fx0 + 21, C_WY - 9, '#f4ecdc')
    cv.rect(fx0 - 20, C_WY - 8, 40, 2, '#6cbcb8')
    cv.hline(fx0 - 22, fx0 + 21, C_WY - 5, '#a89c84')
    cv.rect(fx0 - 2, C_WY - 24, 4, 15, '#d8ccb4')
    cv.vline(fx0 + 1, C_WY - 24, C_WY - 10, '#b0a48c')
    cv.ellipse(fx0, C_WY - 24, 10, 2, '#e6dcc8')
    cv.hline(fx0 - 9, fx0 + 9, C_WY - 24, '#6cbcb8')
    cv.rect(fx0 - 1, C_WY - 34, 3, 10, '#d8ccb4')
    cv.ellipse(fx0, C_WY - 34, 5, 1, '#e6dcc8')
    cv.circle(fx0, C_WY - 39, 3, 'gold')
    cv.px(fx0 - 1, C_WY - 40, 'butter')
    for a in range(8):
        ang = a * math.pi / 4
        cv.px(fx0 + int(round(math.cos(ang) * 5)), C_WY - 39 + int(round(math.sin(ang) * 5)), 'gold')
    meta['fountain'] = (fx0, C_WY - 39, C_WY - 24, C_WY - 8)
    # terrace balustrade either side
    for x0 in (fx0 - 40, fx0 + 24):
        cv.rect(x0, C_WY - 12, 16, 7, '#e6dcc8')
        cv.hline(x0, x0 + 15, C_WY - 12, '#fff8ec')
        for xx in range(x0 + 1, x0 + 15, 2):
            cv.vline(xx, C_WY - 10, C_WY - 7, '#b0a48c')
    # boathouse pavilion (copper-green roof, arches) with a little dock
    bx = 330
    cv.rect(bx, C_WY - 22, 44, 17, '#f0e6d4')
    for ax in range(bx + 3, bx + 40, 8):
        cv.rect(ax, C_WY - 18, 5, 13, '#5a7fb5')
        cv.ellipse(ax + 2, C_WY - 18, 2, 2, '#5a7fb5')
        cv.vline(ax, C_WY - 17, C_WY - 6, '#a8d4f5')
    cv.poly([(bx - 3, C_WY - 22), (bx + 46, C_WY - 22), (bx + 40, C_WY - 31), (bx + 3, C_WY - 31)], '#4aa08a')
    cv.hline(bx + 3, bx + 40, C_WY - 31, '#8ad8c0')
    cv.hline(bx - 3, bx + 46, C_WY - 21, '#2f7a6a')
    cv.vline(bx + 22, C_WY - 38, C_WY - 31, '#d8d4cc')
    meta['flag'] = (bx + 23, C_WY - 38)
    cv.rect(bx - 12, C_WY - 3, 70, 3, '#9a6a48')
    cv.hline(bx - 12, bx + 57, C_WY - 3, '#c89a68')
    # blossom tree near the boathouse (petals animated)
    tree(cv, 296, C_WY - 3, 13, ('#c85a8a', '#f080b0', '#ffb0d0', '#ffe0ee'), seed=88)
    meta['blossom'] = (296, C_WY - 20)
    # benches + lamps on the bank
    for bxx in (150, 240):
        cv.rect(bxx, C_WY - 9, 10, 2, '#9a6a48')
        cv.hline(bxx, bxx + 9, C_WY - 12, '#9a6a48')
        cv.vline(bxx + 1, C_WY - 7, C_WY - 6, '#3a3046')
        cv.vline(bxx + 8, C_WY - 7, C_WY - 6, '#3a3046')
    for lx in (196, 284, 420):
        cv.vline(lx, C_WY - 20, C_WY - 6, '#2f4a44')
        cv.rect(lx - 1, C_WY - 23, 3, 3, '#fff4dc')
        cv.px(lx, C_WY - 24, '#2f4a44')
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def c_bridge():
    """Hump-backed stone arch bridge standing in the lake (side view); bottom row = waterline."""
    w, h = 124, 36
    cv = layer(w, h)
    stone, stone_lt, stone_dk = col('#dccfb6'), col('#f4ecdc'), col('#a89c84')
    cx = w // 2
    tops = []
    for x in range(w):
        top = 10 - int(round(6 * math.sin(math.pi * (x + 0.5) / w)))
        tops.append(top)
        cv.vline(x, top, h - 1, stone)
        cv.px(x, top, stone_lt)
        cv.px(x, top + 1, stone_dk)
        # balustrade
        cv.px(x, top - 5, stone_lt)
        cv.px(x, top - 4, stone)
        if x % 3 == 0:
            cv.vline(x, top - 3, top - 1, stone)
    # masonry courses
    for yy in range(14, h, 4):
        for xx in range(0, w, 6):
            cv.px(xx + (3 if (yy // 4) % 2 else 0), yy, mix(stone, stone_dk, 0.6))
    # voussoir ring, shadowed intrados, opening
    rx, ry = 36, 24
    cv.ellipse(cx, h, rx + 3, ry + 3, stone_lt)
    for k in range(0, 19):
        a = math.pi * k / 18
        cv.line(cx + int(round(math.cos(a) * rx)), h - int(round(math.sin(a) * ry)),
                cx + int(round(math.cos(a) * (rx + 3))), h - int(round(math.sin(a) * (ry + 3))), stone_dk)
    cv.ellipse(cx, h, rx, ry, '#6e6452')
    cv.ellipse(cx, h + 4, rx - 1, ry - 1, KEY)
    cv.rect(cx - 1, h - ry - 4, 3, 3, 'gold')
    cv.px(cx, h - ry - 3, 'butter')
    # spandrel medallions + end piers
    for mx in (12, w - 13):
        cv.circle(mx, 20, 3, stone_dk)
        cv.circle(mx, 20, 2, '#6e6452')
    for xx in (0, w - 4):
        cv.rect(xx, tops[min(w - 1, xx)] - 7, 4, h, stone_lt)
        cv.rect(xx + 3, tops[min(w - 1, xx)] - 7, 1, h, stone_dk)
        cv.rect(xx - 1, tops[min(w - 1, xx)] - 8, 6, 2, stone)
        cv.px(xx + 1, tops[min(w - 1, xx)] - 10, '#2f4a44')
        cv.px(xx + 1, tops[min(w - 1, xx)] - 11, '#fff4dc')
    return to_rgba(cv)


ROWBOAT = Sprite.from_ascii('''
  ...........o....
  ..........o.....
  w...wwwwwwwwwww.
  wwwwwwwwwwwwwwww
  .bbbbbbbbbbbbbb.
  ..BBBBBBBBBBBB..
''', {'w': '#e0b884', 'o': '#9a6a48', 'b': '#3a7cb8', 'B': '#2a5a8a'})


@lru_cache(maxsize=None)
def c_fg():
    cv = layer(520)
    # near lawn edge with tulip beds + a bench and a lamp, low along the bottom
    cv.rect(0, 172, 520, 8, '#4a9a4a')
    cv.hline(0, 519, 172, '#7ac060')
    cv.dither(0, 175, 520, 5, '#3a7a40', 0.4)
    cv.rect(0, 170, 520, 2, '#c8bca4')
    cv.hline(0, 519, 170, '#e6dcc8')
    for k in range(70):
        x = int(hash01(17, k) * 520)
        cc = ('#f04f78', '#f9c22b', '#ffffff', '#c86ad8', '#ff8a5a')[k % 5]
        cv.vline(x, 166, 170, '#3f8a3a')
        cv.rect(x - 1, 164, 3, 2, cc)
        cv.px(x, 163, cc)
    for lx in (20, 420):
        cv.rect(lx, 112, 3, 60, '#253a36')
        cv.vline(lx, 112, 171, '#3f6058')
        cv.rect(lx - 2, 104, 7, 8, '#253a36')
        cv.rect(lx - 1, 105, 5, 6, '#fff4dc')
        cv.px(lx, 106, 'white')
        cv.tri((lx - 3, 104), (lx + 5, 104), (lx + 1, 100), '#253a36')
    return to_rgba(cv)


C_CLOUDS = ((20, 14, 46, 0.09, 21), (200, 30, 34, 0.07, 22), (380, 10, 40, 0.10, 23))


def petals(c, f, x0, y0, n, seed, spread=90, fall=60):
    for i in range(n):
        period = 90 + int(hash01(seed, i) * 50)
        t = ((f + hash01(seed, i, 2) * period) % period) / period
        x = x0 + int(t * spread + math.sin(t * 9 + i) * 4 + hash01(seed, i, 3) * 20)
        y = y0 + int(t * fall + hash01(seed, i, 4) * 6)
        c.px(x, y, '#ffc0dc' if (f // 6 + i) % 3 else 'white')


def shot_c(f):
    X = cam_c(f)
    c = sky_layer().copy()
    draw_clouds(c, f, C_CLOUDS, ox=X * 0.06)
    far, fmeta = c_far()
    fo = -int(X * C_P['far'])
    paste(c, far, fo)
    for (x, y, ph) in fmeta.get('glints', []):
        glint(c, x + fo, y, f, ph, period=80)
    for i, (x, y) in enumerate(fmeta.get('flags', [])):
        flag(c, x + fo, y, f, '#c8d8ea', seed=i, w=3)
    cx, ty, _ = fmeta['crystaller']
    if f % 18 < 6:
        fx.sparkle(c, cx + fo, ty, f % 18, 'white', period=12, size=2)
    draw_balloon(c, 250 - int(X * 0.25) + int((f - T_C) * 0.16), 40, f, size='s')
    br, bmeta = c_brown()
    bo = -int(X * C_P['brown'])
    paste(c, br, bo)
    for k, (x0, x1, y) in enumerate(bmeta['lines']):
        c.hline(x0 + bo, x1 + bo, y, '#e6e0d8')
        for j, xx in enumerate(range(x0 + 2, x1 - 2, 4)):
            sw = 1 if (f // 6 + j + k) % 3 == 0 else 0
            cc = ('#f04f78', '#ffffff', '#3a7cb8', '#f9c22b', '#86c05a')[(j + k) % 5]
            c.rect(xx + bo + sw, y + 1, 2, 3 + (j % 2), cc)
    for i, (x, y) in enumerate(bmeta['chimneys']):
        if i % 2 == 0:
            fx.steam(c, x + bo, y, f, color='#e8ecf2', n=3, height=10, spread=2, seed=i, speed=0.25)
    paste(c, c_trees(), -int(X * C_P['trees']))
    bank, kmeta = c_bank()
    ko = -int(X * C_P['bank'])
    paste(c, bank, ko)
    # fountain jets (pulse on the beat), drawn before the water so they reflect
    fx0, fty, fby, fpy = kmeta['fountain']
    fx0 += ko
    pulse = 2 if f % 18 < 4 else 0
    jh = 7 + pulse
    for s in (-1, 1):
        for j in range(0, 7):
            u = j / 6
            xx = fx0 + s * int(round(u * 6))
            yy = fby - 1 - int(round(jh * (1 - (2 * u - 0.6) ** 2)))
            c.px(xx, yy, 'white' if (j + f // 2) % 3 else '#bfefff')
        for j in range(0, 9):
            u = j / 8
            xx = fx0 + s * int(round(5 + u * 9))
            yy = fby + int(round(u * u * 14)) - 1
            c.px(xx, yy, '#d8f6ff' if (j + f // 2) % 3 else 'white')
    c.vline(fx0, fty - 5 - pulse, fty - 4, 'white')
    for s in (-1, 1):
        for j in range(6):
            c.px(fx0 + s * (j + 2), fby - 10 - int(3 * (1 - (j / 5 - 0.2) ** 2)) + j * 2 // 2 + (j * j) // 6, '#e8fbff')
    for i in range(10):
        ph = (f + i * 7) % 14
        c.px(fx0 - 18 + int(hash01(9, i) * 36), fpy - (ph // 4), 'white' if ph < 7 else '#9ad8d0')
    fl = kmeta['flag']
    flag(c, fl[0] + ko, fl[1], f, '#e83b3b', seed=2)
    # water with reflections
    water(c, f, C_WY, LAKE, refl=0.56, seed=3, src_y0=40, sun_x=150, ripple=0.55, amp=0.6, glitter=24)
    # rowboat behind the bridge, the bridge, then boats in front
    rb2 = ROWBOAT.recolor({col('#3a7cb8'): col('#e8e4dc'), col('#2a5a8a'): col('#b8b0a4')})
    boat(c, rb2, 300 + int(4 * math.sin(f / 30)) - int(X * 1.05), C_WY + 6, f, wake=False, seed=2)
    bri = c_bridge()
    bx = 134 - int(X * 1.08)
    by = 149
    bh = bri.shape[0]
    # bridge reflection (wobbly, tinted)
    part = bri[::-1].copy()
    m = part[..., 3] > 0
    part[..., :3][m] = (part[..., :3][m].astype(np.float32) * 0.62 + np.array((60, 140, 150)) * 0.38).astype(np.uint8)
    for r in range(0, bh - 4):
        off = int(round(math.sin(r * 0.7 + f * 0.25) * (0.5 + r * 0.05)))
        row = part[r:r + 1]
        if (r + f // 4) % 6 != 0:
            c.blit(row, bx + off, by + 1 + r)
    paste(c, bri, bx, by - bh + 1)
    rb3 = ROWBOAT.recolor({col('#3a7cb8'): col('#d8443a'), col('#2a5a8a'): col('#a8302a')})
    boat(c, ROWBOAT, 84 + int(3 * math.sin(f / 24)) - int(X * 1.2), 157, f, wake=False, seed=1)
    boat(c, rb3, 290 + int(3 * math.sin(f / 27 + 1)) - int(X * 1.2), 160, f, flip=True, wake=False, seed=3)
    petals(c, f, 230 + ko, 104, 14, 5, spread=110, fall=64)
    paste(c, c_fg(), -int(X * C_P['fg']))
    petals(c, f, 120 - int(X * C_P['fg']), 100, 8, 9, spread=140, fall=74)
    callout(c, bx + 62, by - 31, 'THE PARK', f, 258, 327, side='up', length=12)
    callout(c, 268 + bo, 96, 'BRONCLYN', f, 270, 327, side='up', length=16)
    return c


# ================================================================== SHOT D: the rail bridge to the mainland
D_P = dict(far=0.1, back=0.22, city=0.3, bridge=1.0, fg=1.7)
D_WY = 142
D_DECK = 104
T1, T2 = 236, 476            # tower centres (world x, bridge layer)
TW = 22
TTOP = 26
AL, AR = 150, 588            # cable anchor points (left / right)


def cam_d(f):
    return cruise_cam(f, T_D, N - 1, 190)


def cam_dy(f):
    """Gentle crane up while pulling back (everything but the sky sinks a little)."""
    return cruise_cam(f, D_HIT, N - 1, 10)


def cable_y(x):
    if x <= T1:
        u = (x - AL) / (T1 - AL)
        return lerp(100, TTOP + 6, u) + 10 * 4 * u * (1 - u)
    if x <= T2:
        u = (x - T1) / (T2 - T1)
        return TTOP + 6 + (94 - TTOP - 6) * 4 * u * (1 - u)
    u = (x - T2) / (AR - T2)
    return lerp(TTOP + 6, 100, u) + 10 * 4 * u * (1 - u)


def stone_tower(cv, cx, top, base):
    st, lt, dk, dd = col('#dccfb6'), col('#f4ecdc'), col('#a89c84'), col('#6e6452')
    x0 = cx - TW // 2
    cv.rect(x0, top, TW, base - top, st)
    cv.rect(x0 + TW - 5, top, 5, base - top, dk)
    cv.vline(x0, top, base - 1, lt)
    for yy in range(top + 4, base, 5):
        for xx in range(x0 + 2, x0 + TW - 5, 6):
            cv.px(xx + (3 if (yy // 5) % 2 else 0), yy, mix(st, dk, 0.6))
    # twin pointed (gothic) arches where the deck passes through
    for ax in (x0 + 4, x0 + 12):
        cv.rect(ax, 72, 6, D_DECK - 72, dd)
        cv.tri((ax, 72), (ax + 5, 72), (ax + 2, 66), dd)
        cv.vline(ax, 72, D_DECK - 1, '#4e4638')
        cv.px(ax + 2, 65, lt)
    # cornice, upper arches, gold star emblem, flag
    cv.rect(x0 - 2, top, TW + 4, 3, lt)
    cv.hline(x0 - 2, x0 + TW + 1, top + 3, dk)
    cv.rect(x0 + 3, top + 8, 4, 10, dd)
    cv.rect(x0 + TW - 9, top + 8, 4, 10, dd)
    cv.circle(cx, top + 22, 3, 'gold')
    cv.px(cx - 1, top + 21, 'butter')
    cv.rect(x0 - 1, 112, TW + 2, 4, lt)
    cv.rect(x0 - 3, base - 8, TW + 6, 8, dk)
    cv.hline(x0 - 3, x0 + TW + 2, base - 8, lt)
    cv.vline(cx, top - 6, top - 1, '#d8d4cc')


@lru_cache(maxsize=None)
def d_bridge():
    cv = layer(700)
    meta = {}
    steel, steel_lt, steel_dk = col('#7a8494'), col('#b8c2d0'), col('#4a5262')
    # far bank of the mainland (right) and the island quay (left)
    for i, x in enumerate(range(586, 700, 9)):
        r = 8 + (i * 5) % 6
        cv.circle(x, D_WY - 8 - r // 2, r, '#3f8a4a')
        cv.circle(x - 1, D_WY - 9 - r // 2, r - 2, '#5aae58')
        cv.px(x - 3, D_WY - 12 - r // 2, '#9ad880')
    cv.rect(580, D_WY - 8, 120, 8, '#6a9a4a')
    cv.hline(580, 699, D_WY - 8, '#8ac060')
    cv.hline(580, 699, D_WY - 1, '#4a6a3a')
    # deck truss
    x0, x1 = 0, 660
    cv.rect(x0, D_DECK, x1 - x0, 9, steel)
    cv.hline(x0, x1 - 1, D_DECK, steel_lt)
    cv.hline(x0, x1 - 1, D_DECK + 8, steel_dk)
    for x in range(x0, x1, 8):
        cv.line(x, D_DECK + 1, x + 7, D_DECK + 7, steel_dk)
        cv.vline(x, D_DECK + 1, D_DECK + 7, steel_lt)
    cv.hline(x0, x1 - 1, D_DECK - 1, '#5e3c2c')
    for x in range(x0, x1, 3):
        cv.px(x, D_DECK - 1, '#8a6a4a')
    cv.hline(x0, x1 - 1, D_DECK - 2, '#9aa4b4')
    # suspender cables + main cable (drawn before the towers)
    for x in range(AL, AR + 1):
        y = int(round(cable_y(x)))
        if x % 4 == 0 and y < D_DECK - 2:
            cv.vline(x, y + 1, D_DECK - 3, '#9aa4b4')
    prev = None
    for x in range(AL, AR + 1):
        y = int(round(cable_y(x)))
        if prev is not None and abs(y - prev) > 1:
            cv.vline(x, min(y, prev), max(y, prev), '#4a5262')
        cv.px(x, y, '#4a5262')
        cv.px(x, y - 1, '#c8d0dc')
        prev = y
    # diagonal stays fanning from the tower tops
    for t in (T1, T2):
        for d in (24, 44, 64, 84):
            for s in (-1, 1):
                cv.line(t + s * 6, TTOP + 10, t + s * (d + 6), D_DECK - 3, '#aab4c2')
    stone_tower(cv, T1, TTOP, D_WY + 4)
    stone_tower(cv, T2, TTOP, D_WY + 4)
    # anchorages (big stone blocks with arches)
    for vx in range(0, 132, 22):             # approach viaduct arches from the island
        cv.rect(vx, D_DECK + 9, 22, D_WY + 4 - D_DECK - 9, '#d0c4ac')
        cv.ellipse(vx + 11, D_WY + 4, 7, 20, '#6e6452')
        cv.ellipse(vx + 11, D_WY + 6, 6, 19, '#8a8070')
        cv.vline(vx, D_DECK + 9, D_WY + 3, '#f4ecdc')
        cv.hline(vx, vx + 21, D_DECK + 9, '#a89c84')
    for ax, aw in ((138, 26), (566, 28)):
        cv.rect(ax, 94, aw, D_WY + 4 - 94, '#d0c4ac')
        cv.rect(ax + aw - 5, 94, 5, D_WY + 4 - 94, '#a89c84')
        cv.hline(ax - 1, ax + aw, 94, '#f4ecdc')
        cv.hline(ax, ax + aw - 1, 95, '#a89c84')
        for k in range(ax + 5, ax + aw - 8, 10):
            cv.rect(k, 118, 6, D_WY + 4 - 118, '#6e6452')
            cv.ellipse(k + 3, 118, 3, 3, '#6e6452')
        for yy in range(99, D_WY, 5):
            for xx in range(ax + 2, ax + aw - 5, 7):
                cv.px(xx + (3 if (yy // 5) % 2 else 0), yy, '#b8ac94')
    meta['flags'] = [(T1 + 1, TTOP - 9), (T2 + 1, TTOP - 9)]
    # city portal (stone gateway the train comes out of); also redrawn over the train
    pcv = layer(34, 40)
    pcv.rect(0, 4, 34, 36, '#c8bca4')
    pcv.rect(28, 4, 6, 36, '#a89c84')
    pcv.hline(0, 33, 4, '#f4ecdc')
    pcv.rect(-1, 0, 36, 4, '#e6dcc8')
    pcv.hline(-1, 34, 0, '#ffffff')
    pcv.rect(12, 8, 10, 6, '#6e6452')
    pcv.circle(17, 9, 3, 'gold')
    for yy in range(16, 40, 5):
        for xx in range(2, 28, 7):
            pcv.px(xx + (3 if (yy // 5) % 2 else 0), yy, '#b0a48c')
    meta['portal'] = to_rgba(pcv)
    return to_rgba(cv), meta


@lru_cache(maxsize=None)
def d_train():
    """The pink Friendship Express facing right (front at the right end). Bottom row = rail."""
    coaches = [('#f2a0c8', '#c86a98'), ('#c8a8f0', '#9a78c8'), ('#a0e0c8', '#68b098'), ('#a8d0f8', '#6a98c8')]
    L = 24
    n = len(coaches)
    w = n * (L + 2) + 12 + 30
    h = 18
    cv = layer(w, h)
    x = 0
    for body, dk in coaches:
        cv.round_rect(x, 5, L, 10, dk, r=1)
        cv.rect(x + 1, 6, L - 2, 8, body)
        cv.round_rect(x - 1, 3, L + 2, 3, '#f4ecf8', r=1)
        cv.hline(x, x + L - 1, 5, '#d8c8e0')
        cv.rect(x + 1, 7, L - 2, 3, 'cream')
        for wx in range(x + 2, x + L - 3, 4):
            cv.rect(wx, 7, 3, 3, '#5a86b8')
            cv.px(wx, 7, '#c6e6ff')
        cv.hline(x + 1, x + L - 2, 12, 'gold')
        for wx in (x + 4, x + L - 5):
            cv.circle(wx, 15, 2, '#5a4a6a')
            cv.px(wx, 15, '#c8b8d8')
        cv.rect(x + L, 11, 2, 2, '#5a4a6a')
        x += L + 2
    # tender
    cv.rect(x, 7, 12, 8, '#c8407a')
    cv.rect(x + 1, 6, 10, 2, '#5a4a52')
    cv.hline(x, x + 11, 12, 'gold')
    for wx in (x + 3, x + 8):
        cv.circle(wx, 15, 2, '#5a4a6a')
    x += 12
    # locomotive: cab at the left, boiler to the right, funnel near the front, cowcatcher
    cv.rect(x, 2, 9, 13, '#e8609a')
    cv.rect(x - 1, 1, 11, 2, '#9a78c8')
    cv.rect(x + 2, 4, 5, 4, '#c6e6ff')
    cv.px(x + 2, 4, 'white')
    cv.rect(x + 9, 6, 17, 8, '#f080b0')
    cv.hline(x + 9, x + 25, 6, '#ffc0dc')
    cv.hline(x + 9, x + 25, 13, '#c8407a')
    for bx in (x + 13, x + 19):
        cv.vline(bx, 6, 13, 'gold')
    cv.rect(x + 15, 3, 3, 3, 'gold')                  # dome
    cv.rect(x + 21, 0, 4, 6, '#5a4a52')               # funnel
    cv.rect(x + 20, 0, 6, 2, 'gold')
    cv.rect(x + 26, 8, 2, 4, 'gold')                  # headlamp housing
    cv.px(x + 27, 9, '#fff4b0')
    cv.tri((x + 25, 15), (x + 29, 15), (x + 25, 11), '#c83a3a')
    for wx, r in ((x + 4, 3), (x + 12, 3), (x + 19, 3), (x + 24, 2)):
        cv.circle(wx, 14, r, '#c83a3a')
        cv.circle(wx, 14, max(1, r - 2), '#ffc0dc')
        cv.px(wx, 14, '#5a4a52')
    cv.hline(x + 4, x + 19, 14, '#d8d4dc')
    return to_rgba(cv), x + 23, x + 3       # rgba, funnel x (local), whistle x (local)


@lru_cache(maxsize=None)
def d_city():
    """Left part of the shot-A skyline (with the Crystaller), cropped for the bridge shot."""
    rgba, meta = a_main()
    crop = rgba[:, :241].copy()
    return crop, meta


@lru_cache(maxsize=None)
def d_back():
    rgba, meta = a_back()
    return rgba[:, :250].copy(), meta


@lru_cache(maxsize=None)
def d_hills():
    """The mainland across the river: hazy blue ridge, green hills with tree clumps, a village."""
    cv = layer(420)
    for x in range(150, 420):
        u = x / 420
        ramp = min(1.0, (x - 150) / 60)
        y = D_WY - 1 - int(ramp * (21 + 10 * math.sin(u * 4.0 + 0.6) ** 2 + 4 * math.sin(u * 11)))
        cv.vline(x, y, D_WY - 1, '#a8cce4')
        cv.px(x, y, '#c8e2f2')
    for x in range(170, 420):
        u = x / 420
        ramp = min(1.0, (x - 170) / 40)
        y = D_WY - 1 - int(ramp * (9 + 8 * math.sin(u * 5.0 + 0.6) ** 2 + 3 * math.sin(u * 13)))
        cv.vline(x, y, D_WY - 1, '#7fb88a')
        cv.px(x, y, '#a8d8a0')
        cv.dither(x, y + 3, 1, D_WY - y - 3, '#6aa47a', 0.35)
    for i in range(26):
        x = 180 + int(hash01(19, i) * 236)
        y = D_WY - 6 - int(hash01(19, i, 2) * 5)
        r = 2 + int(hash01(19, i, 3) * 2)
        cv.circle(x, y, r, '#5a9a6a')
        cv.px(x - 1, y - r + 1, '#9ad09a')
    for i in range(12):
        x = 250 + int(hash01(9, i) * 150)
        y = D_WY - 12 - int(hash01(9, i, 2) * 6)
        cv.rect(x, y, 4, 4, '#f0e8d8')
        cv.tri((x - 1, y), (x + 4, y), (x + 1, y - 2), ('#c87a5a', '#7a8ac8', '#c8a05a')[i % 3])
        cv.px(x + 1, y + 2, '#5a86b8')
    cv.vline(330, D_WY - 26, D_WY - 14, '#f0e8d8')
    cv.px(330, D_WY - 27, 'gold')
    return to_rgba(cv)


D_CLOUDS = ((60, 20, 44, 0.10, 31), (260, 40, 30, 0.08, 32), (420, 14, 50, 0.11, 33))
TRAIN_X0 = 60                       # train front (world) at the hit: it bursts in from the left edge


def train_front(f):
    return TRAIN_X0 + 4.6 * (f - D_HIT)


def shot_d(f):
    X = cam_d(f)
    Y = cam_dy(f)
    c = sky_layer().copy()
    draw_clouds(c, f, D_CLOUDS, ox=X * 0.06)
    paste(c, d_hills(), -int(X * D_P['far']), Y)
    back, _ = d_back()
    paste(c, back, -int(X * D_P['back']) - 110, Y + 12)
    city, cmeta = d_city()
    co = -int(X * D_P['city']) - 100
    cy_ = Y + 18
    paste(c, city, co, cy_)
    quay = a_quay()[0]
    paste(c, quay[:, :238], co, cy_)
    c.rect(co + 236, A_WY - 6 + cy_, 2, 6, '#8c806c')
    cx, ty, _ = cmeta['crystaller']
    for (x, y, ph) in cmeta.get('glints', []):
        if x < CRYS_X + 64:
            glint(c, x + co, y + cy_, f, ph, period=72)
    for i, (x, y) in enumerate(cmeta.get('flags', [])):
        if x < CRYS_X + 60:
            flag(c, x + co, y + cy_, f, ('#e83b3b', '#3f6fc8', '#f2c24a', '#3f9a86')[i % 4], seed=i)
    if f % 18 < 6:
        fx.sparkle(c, cx + co, ty + cy_, f % 18, 'white', period=12, size=3)
    # the balloon sails past the Crystaller spire
    bxb = 52 + int((f - T_D) * 0.8) - int(X * D_P['city'])
    draw_balloon(c, bxb, 58 + cy_ - 16 - int((f - T_D) * 0.1), f, size='m', flame=(f // 24) % 2 == 0)
    # bridge
    bridge, meta = d_bridge()
    bo = -int(X * D_P['bridge'])
    byo = Y
    paste(c, bridge, bo, byo)
    for i, (x, y) in enumerate(meta['flags']):
        flag(c, x + bo, y + byo + 3, f, ('#f04f78', '#3f6fc8')[i], seed=i + 5, w=5)
    for t in (T1, T2):                                  # tower star emblems flash on the hit + beats
        k = f - D_HIT if f < D_HIT + 18 else f % 18
        if 0 <= k < 8:
            fx.sparkle(c, t + bo, TTOP + 22 + byo, k, 'white', period=16, size=4 if D_HIT <= f < D_HIT + 18 else 2)
    # train + steam
    tr, fun_x, whi_x = d_train()
    tw = tr.shape[1]
    fr = train_front(f)
    tx = int(round(fr)) - tw + bo
    ty_ = D_DECK - 2 - tr.shape[0] + 1 + byo
    # trailing steam plume: puffs stay where they were emitted and rise
    for j in range(16):
        te = f - j * 3 - (f % 3)
        if te < D_HIT - 24:
            break
        age = f - te
        sx = train_front(te) - tw + fun_x + bo - int(age * 0.25)
        sy = ty_ - int(age * 0.55) - 1
        r = 1 + min(4, age // 7)
        cc = '#ffffff' if age < 18 else ('#eef0f6' if age < 32 else '#d8dce8')
        if age < 46:
            c.circle(int(sx), sy + 1, r, '#c8ccd8')
            c.circle(int(sx), sy, r, cc)
    paste(c, tr, tx, ty_)
    # whistle plume on the BRIDGE hit and a second toot half a bar later
    for t0, big in ((D_HIT, 1.0), (D_HIT + 36, 0.6)):
        if t0 <= f < t0 + 22:
            k = f - t0
            wx = int(round(train_front(f))) - tw + whi_x + bo
            for i in range(7):
                age = k - i * 2
                if age < 0:
                    continue
                r = int((2 + min(4, age / 2)) * big + 0.5)
                px_ = wx - int(age * 0.7) - (i % 2) * 2
                py_ = ty_ - 4 - int(age * 1.4 * big) - i * 2
                if age < 20:
                    c.circle(px_, py_ + 1, r, '#c8ccd8')
                    c.circle(px_, py_, r, 'white' if age < 12 else '#eef0f6')
    # tower legs + portal in front of the train
    for t in (T1, T2):
        sub = bridge[D_DECK - 40:D_DECK + 1, t - TW // 2 - 1:t + TW // 2 + 1]
        paste(c, sub, t - TW // 2 - 1 + bo, D_DECK - 40 + byo)
    # river
    wy = D_WY + byo
    water(c, f, wy, ['#80c4ea', '#58a2d4', '#3e84bc', '#2f6aa6'], refl=0.42, seed=7, src_y0=20, sun_x=236)
    # vessels on the river
    boat(c, SAIL, 300 - (f - T_D) * 0.4 - int(X * 1.1), wy + 12, f, flip=True, seed=11)
    boat(c, FERRY, 150 + (f - T_D) * 0.5 - int(X * 1.2), wy + 30, f, seed=14)
    smoke_puffs(c, 150 + (f - T_D) * 0.5 - int(X * 1.2) + 8, wy + 20, f, n=4, drift=-0.2, seed=15)
    boat(c, TUG, 500 - (f - T_D) * 0.3 - int(X * 1.15), wy + 26, f, flip=True, seed=12)
    smoke_puffs(c, 500 - (f - T_D) * 0.3 - int(X * 1.15) + 5, wy + 18, f, n=5, drift=0.25, seed=13,
                color='#d8d8e0', dark='#9a96a8')
    paste(c, a_fg(), -110 - int(X * D_P['fg']), Y)
    callout(c, T1 + bo + TW // 2 + 1, 52 + byo, 'RAIL BRIDGE', f, 348, 420, side='right', length=12)
    return c


# ================================================================== render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < T_B:
        c = shot_a(f)
    elif f < T_C:
        c = shot_b(f)
    elif f < T_D:
        c = shot_c(f)
    else:
        c = shot_d(f)
    atlas_card(c, f, 100, 196, 'METROPOLIS', ['ISLAND OF SKYSCRAPERS', 'THEATRES - PIERS - PARK'])
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), CRYS_X, 70)
    return c


SFX = [
    (82, 'waves', 0.45),
    (108, 'horn', 0.55),
    (140, 'waves', 0.35),
    (162, 'neon', 0.55),
    (172, 'chug', 0.3),
    (196, 'steam', 0.35),
    (208, 'chug', 0.3),
    (252, 'fountain', 0.5),
    (288, 'fountain', 0.35),
    (306, 'creak', 0.3),
    (360, 'whistle', 0.85),
    (366, 'chug', 0.5),
    (396, 'whistle', 0.5),
    (402, 'chug', 0.45),
    (414, 'horn', 0.35),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(162, 'dissolve', {}), (252, 'glide', {'dir': -1}), (342, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
