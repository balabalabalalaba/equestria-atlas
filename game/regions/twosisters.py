"""The Castle of the Two Sisters as an explorable diorama (art: scenes/s20_twosisters.py, used read-only).

The film's finale shows three different views (zoom / dissolve between them), so the region has three
areas, flown through left to right with a cloud sweep between them:
  0  THE CHASM (shot A): midnight in the Everfree under the full moon, the ruined castle on its crag,
     the near cliff with the gnarled tree and the swaying rope bridge, mist rolling through the gorge.
     Camera 0..130: the far forest and the gorge's far wall are rebuilt wider with the scene's code,
     so the balloon can fly on east past the castle into the moonlit gorge.
  1  THE OLD THRONE HALL (shot B): the dollhouse cross-section of the ruined hall, moonbeams through
     the collapsed vault, torches, sun and moon banners, the two empty thrones under the rose window,
     the stairwell glowing cyan below. Camera 0..80 (the hall is 400 px).
  2  THE CAVE OF HARMONY (shot C): the film's tall strip as a vertical panorama (camera y 0..250): the
     castle on top under the moon, the cut-away rock with the stair shaft, the cave with the Tree of
     Harmony and its reflecting pool. The balloon climbs and sinks twice as fast here. Leaving the hall
     high up brings it into the sky by the castle; leaving it low (by the stairwell) straight down into
     the cave by the tree.
Each view is the scene's own render_A / render_B / render_C, bound (rebind) to the player's camera,
without the film's balloon (nor the burner glow of the balloon resting on the keep), callouts, card,
iris and cuts; their big layers blit through a faster masked copy and the moonbeams come from cached
layers (same pixels). Timed film beats on game clocks:
  - MIDNIGHT (film 288): the tree's flare, the gems pulsing on the bell strikes and the light running
    along the veins play the first time the balloon comes down to the tree (each visit), and again on
    SPACE; after it the castle's rose window brightens (the light has climbed up, film 380..396).
  - the shooting star (film 396..409) crosses the sky over the castle every 11 s (A, and C up top).
Things to set off with SPACE: the full moon blazes and a shower of shooting stars streaks the sky, a gust
swings the rope bridge and tears leaves off the gnarled tree, the hall's torches flare one after
another, the rose window blazes over the thrones of the sun and the moon, and the midnight bell rings at
the Tree of Harmony (the flare again, or one more strike while it still blazes). Three atlas stamps are
hidden: low in the gorge mist at the chasm's far east end, high in the broken vault of the hall, and
deep in the cave by the pool among the pink crystals.
No creatures anywhere (the edition's rule): stone, light, moon, stars and crystal only.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, Sprite, fx, seg, hash01, clamp, mix, col, rng
from engine.core import bayer
import scenes.s20_twosisters as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# ----------------------------------------------------------------------------- geometry
CAM_A = 130             # the film's far forest (340 px, p .25) and gorge wall (360, p .5) end at cam 80:
WF, WM = 360, 390       # both are rebuilt wider with the scene's own code (_far_wide, _mid_wide)
CAM_B = 80              # the hall is 400 px (p 1)
CAM_CY = sc.S_H - H     # the strip is 430 px tall: camera y 0..250
P = sc.A_P
AX, AY = sc.A_CASTLE                    # castle sprite origin in shot A (castle layer, p .8)
ROSE_A = (AX + sc.ROSE[0], AY + sc.ROSE[1])
MOON_A = sc.A_MOON
BRIDGE_TAG = ((46 + AX + 14) // 2, (129 + AY + 121) // 2 - 2)     # the film's mid_x at cam 0 (p ~.9)
SX, SY = sc.S_CASTLE                    # castle origin in the strip (shot C)
ROSE_C = (SX + sc.ROSE[0], SY + sc.ROSE[1])
HEART = sc.HEART
THRONES = ((252 + 10, 141 - 32), (280 + 10, 141 - 32))     # sun / moon emblem centres (hall px)
RX, RY, RR = sc.B_ROSE

LANDMARKS_A = [
    Landmark('ROPE BRIDGE', 94, (BRIDGE_TAG[0], BRIDGE_TAG[1], 'up', 14), p=0.9),
    # the film's leader is 28 px; 14 keeps the tag clear of the atlas card while the camera pans east
    Landmark('CASTLE OF THE TWO SISTERS', 214, (AX + 79, AY + 100, 'down', 14), p=P['castle']),
]
LANDMARKS_B = [Landmark('THE OLD THRONE HALL', 276, (276, 104, 'up', 10))]
LANDMARKS_C = [Landmark('TREE OF HARMONY', HEART[0], (226, 318, 'right', 10), y=340)]

SUNMOON = Sprite.from_ascii('''
    ...ooo...
    ..oYYbo..
    .oYyYbbo.
    oyYyYbBbo
    oYYyYbbbo
    oyYyYbBbo
    .oYyYbbo.
    ..oYYbo..
    ...ooo...
''', ui._ICON_PAL)

HOT_A = [Hotspot('moon', MOON_A[0], MOON_A[1], 'moon', cooldown=90, p=P['moon']),
         Hotspot('gust', 72, 152, 'wind', cooldown=80)]
HOT_B = [Hotspot('torches', sc.B_TORCH[1][0], sc.B_TORCH[1][1] - 4, 'lantern', cooldown=90),
         Hotspot('thrones', RX, 126, SUNMOON, cooldown=100)]
HOT_C = [Hotspot('tree', HEART[0], HEART[1], 'bell', cooldown=120)]
STAMPS_A = [Stamp('gorge', 434, 150)]
STAMPS_B = [Stamp('vault', 60, 12)]
STAMPS_C = [Stamp('pool', 288, 394)]

TINT_MOON = ('#7f84cc', 0.46)           # moonlit blue
TINT_HALL = ('#6c6aa8', 0.5)            # inside the ruined hall
TINT_TORCH = ('#e8b07c', 0.34)          # by a torch
TINT_ROCK = ('#4a4880', 0.62)           # inside the cut-away rock
TINT_CAVE = ('#b29ae8', 0.36)           # the tree's lilac light
TINT_FLARE = ('#efe4ff', 0.18)          # midnight

STAR_PERIOD = 330       # a shooting star crosses the sky every 11 s (A, and C near the castle)
FLARE_LEN = 90          # the midnight flare (bloom, ring, sparkles, 12 bell strikes on the gems)
BELL_RINGS = ((18, ('butter', 'gold')), (36, ('ice', 'lilac')))     # later strikes: (step, ring colours)
GUST = 75               # steps of the gust's swing
# the moon's shower: (start step, head x, head y, tail colours) on the sky layer (p .1), film-style streaks
_GOLD = ('white', 'butter', 'butter', 'gold', 'amber', mix('amber', 'royal', 0.5))
_ICE = ('white', 'ice', 'baby_blue', 'baby_blue', 'periwinkle', mix('periwinkle', 'royal', 0.5))
_PINK = ('white', mix('crystal_pk', 'white', 0.5), 'crystal_pk', 'lilac', 'lavender', mix('lilac', 'royal', 0.5))
GORGE_Y = 124           # sky hook: rows from here down are the misty gorge, never open sky
WISPS_A = ((70, 4, 380, 66, 0.12), (54, 7, 150, 30, 0.08), (90, 9, 40, 84, 0.1))     # render_A's wisps
WISPS_C = ((64, 4, 260, 74, 0.1), (50, 7, 60, 30, 0.07))                            # render_C's
SHOWER = ((0, 300, 6, _ICE), (5, 236, 2, _GOLD), (10, 352, 20, _PINK), (15, 196, 10, _ICE), (20, 300, 30, _GOLD),
          (25, 380, 4, _ICE), (30, 250, 16, _PINK), (36, 340, 34, _ICE), (42, 220, 0, _GOLD))
SHOWER_LEN = 20         # steps a streak takes to cross
MOON_FX = 72            # steps of the moon's blaze


def _noop(*_a, **_k):
    pass


# ----------------------------------------------------------------------------- shot A widened east
@lru_cache(maxsize=None)
def _far_wide():
    """sc.farA (far forest hills, hazy and moonlit) built WF px wide with the scene's code."""
    w = WF
    cv = sc._new(w, H)
    ramp = sc.haze(sc.LEAF_N, 0.55)
    r = rng(5)
    x = -6
    while x < w + 6:
        rr = int(r.integers(5, 10))
        yb = 108 + 8 * math.sin(x / 47.0) + 5 * math.sin(x / 19.0 + 1)
        sc.leafball(cv, x, yb, rr, ramp)
        x += rr * 1.3
    cv.rect(0, 112, w, H - 112, ramp[1])
    for k in range(40 * w // 340):
        sc.leafball(cv, r.uniform(0, w), r.uniform(114, 150), int(r.integers(4, 8)), ramp)
    rgba = sc._rgba(cv)
    sc.rimlight(rgba, mix(sc.MOONLIT, sc.FOGC, 0.4), 0.5, dirs=((0, -1), (1, 0)))
    return _register(rgba)


@lru_cache(maxsize=None)
def _mid_wide():
    """sc.midA (forest on the gorge's far rim + its far wall) built WM px wide with the scene's code."""
    w = WM
    cv = sc._new(w, H)
    pts = [(-2, 140), (40, 136), (90, 141), (150, 135), (210, 139), (270, 134), (330, 140), (370, 136),
           (w + 2, 139), (w + 2, H + 2), (-2, H + 2)]
    wall = sc.boulder_rock(w, H, pts, sc.haze(sc.ROCK_N, 0.45), 14, cell=(9, 5))
    cv.blit(sc._rgba(wall), 0, 0)
    ramp = sc.haze(sc.LEAF_N, 0.3)
    r = rng(8)
    for k in range(9 * w // 360):
        x = r.uniform(0, w)
        top = r.uniform(108, 122)
        tr = sc.haze(sc.BARK_N, 0.35)
        cv.rect(int(x) - 2, int(top), 4, 140 - int(top), tr[1])
        cv.vline(int(x) + 1, int(top), 140, tr[2])
        sc.canopy(cv, x, top, 13, 5, 7, 5, 9, ramp, 30 + k)
    x = -6
    while x < w + 6:
        rr = int(r.integers(5, 10))
        yb = 131 + 4 * math.sin(x / 31.0) + 3 * math.sin(x / 13.0)
        sc.leafball(cv, x, yb, rr, ramp)
        x += rr * 1.2
    rgba = sc._rgba(cv)
    sc.rimlight(rgba, mix(sc.MOONLIT, sc.FOGC, 0.25), 0.55, dirs=((0, -1), (1, 0)))
    return _register(rgba)


# ----------------------------------------------------------------------------- speed: fast blits
_FAST = {}


class _FC(Canvas):
    """A Canvas whose blit of the scene's big cached RGBA layers copies only their opaque bounding box
    with a precomputed mask (same pixels, several times faster); everything else is the engine's."""
    __slots__ = ()

    def copy(self):
        return _FC(arr=self.a.copy())

    def blit(self, src, x, y, flip=False):
        e = _FAST.get(id(src)) if isinstance(src, np.ndarray) else None
        if e is None or e[0] is not src or flip:
            return Canvas.blit(self, src, x, y, flip)
        _, rgb, m, r0, c0 = e
        x, y = int(round(x)) + c0, int(round(y)) + r0
        X0, Y0 = max(0, x), max(0, y)
        X1, Y1 = min(self.w, x + rgb.shape[1]), min(self.h, y + rgb.shape[0])
        if X1 > X0 and Y1 > Y0:
            np.copyto(self.a[Y0:Y1, X0:X1], rgb[Y0 - y:Y1 - y, X0 - x:X1 - x], where=m[Y0 - y:Y1 - y, X0 - x:X1 - x])


def _register(rgba):
    m = rgba[..., 3] >= 128
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    r0, r1, c0, c1 = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) if rows.size else (0, 0, 0, 0)
    _FAST[id(rgba)] = (rgba, np.ascontiguousarray(rgba[r0:r1, c0:c1, :3]), m[r0:r1, c0:c1, None], r0, c0)
    return rgba


@lru_cache(maxsize=None)
def _layers():
    """Register the scene's big cached layers for the fast blit (built once)."""
    for fn in (sc.farA, sc.midA, sc.cragA, sc.castle_rgba, sc.nearA, sc.fgA, sc.hallB, sc.strip_rgba,
               sc.fg_crystals):
        _register(fn())
    return True


_ALL_SKY = np.ones((H, W), bool)     # the hall's window sky: every pixel of the sky layer (the hall covers it)


@lru_cache(maxsize=None)
def _sky():
    _layers()
    return _FC(arr=sc.sky_bg().a.copy())


@lru_cache(maxsize=4)
def _chasm_sky(mxs):
    """render_A's gradient + full moon (no blaze) at moon x mxs, and the moon light it adds (int16), so a
    star under the halo gets the same light (same pixels up to the glows' rounding)."""
    mx, my, mr = sc.A_MOON
    c = sc.sky_bg().copy()
    if mxs is not None:                         # None: the clock says day, no painted moon
        sc.draw_moon(c, 0, mxs, my, mr)
    return c.a, c.a.astype(np.int16) - sc.sky_bg().a


@lru_cache(maxsize=None)
def _sky_s_only():
    """sc.sky_s() (the strip's sky layer) with its baked far forest inverted, so a sky hook never takes
    the forest for sky."""
    s = sc.sky_s().a
    g = Canvas(W, s.shape[0])
    g.gradient(0, 0, W, 200, sc.SKY_N, soft=0.7)
    g.rect(0, 200, W, s.shape[0] - 200, sc.SKY_N[-1])
    out = s.copy()
    m = (s != g.a).any(axis=2)
    out[m] = 255 - s[m]
    return Canvas(arr=out)


@lru_cache(maxsize=None)
def _vig_mask(strength):
    h, w = H, W
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
    lvl = np.clip((d - 0.55) / 0.45, 0, 1) * strength
    return bayer(h, w) < lvl


def _vignette(c, strength=0.35, color='ink'):
    """fx.vignette with its mask cached (same pixels)."""
    c.a[_vig_mask(strength)] = col(color)


class _Fx:
    """engine.fx for the rebound film functions: cached vignette; light beams can be switched off
    (drawn from a cached layer instead)."""
    def __init__(self, beams=True):
        self.vignette = _vignette
        if not beams:
            self.light_beam = _noop

    def __getattr__(self, k):
        return getattr(fx, k)


FX_A, FX_NOBEAM = _Fx(), _Fx(beams=False)
_FX_CLOCK = {}


def _fx_clock(nq, beams=True):
    """An _Fx whose vignette (the film's dark ink-navy dither in the corners) is 30 % strength by day,
    full at night (nq: daylight.night, 0.1 steps)."""
    key = (nq, beams)
    if key not in _FX_CLOCK:
        e = _Fx(beams)
        s_ = 0.3 + 0.7 * nq
        e.vignette = lambda c, strength=0.35, color='ink': _vignette(c, round(strength * s_, 3), color)
        _FX_CLOCK[key] = e
    return _FX_CLOCK[key]


def _beam_rows(acc, top, bottom, color, strength):
    """fx.light_beam's additive light into a float layer (dither anchored to the layer)."""
    (txl, ty), (txr, _) = top
    (bxl, by), (bxr, _) = bottom
    h, w = acc.shape[:2]
    cc = np.array(col(color), np.float32) * strength
    for y in range(max(0, int(ty)), min(h, int(by) + 1)):
        t = seg(y, ty, by)
        xl, xr = int(txl + (bxl - txl) * t), int(txr + (bxr - txr) * t)
        if xr < xl:
            xl, xr = xr, xl
        x0, x1 = max(0, xl), min(w, xr + 1)
        if x1 <= x0:
            continue
        m = bayer(1, x1 - x0, x0, y)[0] < (1 - t * 0.7)
        acc[y, x0:x1][m] += cc


@lru_cache(maxsize=None)
def _beams_b():
    """The hall's moonbeams (render_B's light_beam calls at k = 1), in hall px."""
    acc = np.zeros((H, sc.B_W + 60, 3), np.float32)
    mb = mix('baby_blue', 'cloud', 0.4)
    _beam_rows(acc, ((130, 50), (168, 50)), ((176, 150), (226, 150)), mb, 0.12)
    for (hx, hy, hw, hh) in sc.B_HOLES:
        _beam_rows(acc, ((hx + 4, hh), (hx + hw - 4, hh)), ((hx + 40, 150), (hx + hw + 50, 150)), mb, 0.12)
    for (wx, wy, ww, wh) in sc.B_WINDOWS:
        _beam_rows(acc, ((wx + 2, wy + 10), (wx + ww - 2, wy + 10)), ((wx + 26, 150), (wx + ww + 34, 150)),
                   'periwinkle', 0.07)
    return acc


@lru_cache(maxsize=None)
def _beam_c():
    """The strip's moonbeam down the fissure (render_C's light_beam), in strip px."""
    acc = np.zeros((sc.S_H, W, 3), np.float32)
    _beam_rows(acc, ((76, 262), (84, 262)), ((100, 402), (126, 402)), mix('baby_blue', 'cloud', 0.4), 0.12)
    return acc


def _add(c, acc, ox, oy, k=1.0):
    """Add a float light layer (its px (ox, oy) at the screen's top-left) to the canvas."""
    h, w = acc.shape[:2]
    y0, y1 = max(0, -oy), min(H, h - oy)
    x0, x1 = max(0, -ox), min(W, w - ox)
    if y1 <= y0 or x1 <= x0:
        return
    reg = c.a[y0:y1, x0:x1].astype(np.float32)
    reg += acc[y0 + oy:y1 + oy, x0 + ox:x1 + ox] * k
    c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def _banner(c, x, y, f, w, L, kind, seed, amp=1.6):
    """sc.draw_banner in one blit (each row shifted by its sway), amp: sway in px at the hem."""
    spr = sc.banner_rgba(w, L, kind)
    rows = np.arange(L)
    dx = np.round(np.sin(f / 21.0 + seed - rows * 0.07) * amp * (rows / L)).astype(int)
    pad = int(math.ceil(amp)) + 1
    cols = np.arange(w + 2 * pad)[None, :] - pad - dx[:, None]
    ok = (cols >= 0) & (cols < w)
    out = np.zeros((L, w + 2 * pad, 4), np.uint8)
    rr = np.broadcast_to(rows[:, None], cols.shape)
    out[ok] = spr[rr[ok], cols[ok]]
    Canvas.blit(c, out, x - pad, y)


def _meteor(c, k, hx, hy, dx=-110, dy=30):
    """The film's shooting star (render_C, 396..409) at progress k (0..1) from head (hx, hy)."""
    if not 0 <= k < 1:
        return
    x, y = int(hx + dx * k), int(hy + dy * k)
    tail = ('white', 'white', 'cloud', 'cloud', 'baby_blue', 'baby_blue', 'baby_blue', 'periwinkle',
            'periwinkle', 'periwinkle', mix('periwinkle', 'royal', 0.5), mix('periwinkle', 'royal', 0.5))
    s = dy / dx
    for j, cc in enumerate(tail[:int(6 + 6 * min(1.0, k * 3))]):
        c.px(x + j, y + int(round(j * s)), cc)
    if k < 0.85:
        c.px(x - 1, y, 'baby_blue')
        c.px(x, y + 1, 'baby_blue')


def _streak(c, k, hx, hy, cols, dx=-160, dy=50, L=32):
    """A bold shooting star at progress k (0..1): 2-px head, a tail that grows then fades."""
    if not 0 <= k < 1:
        return
    x, y = hx + dx * k, hy + dy * k
    n = int(L * min(1.0, k * 4) * (1 - max(0.0, k - 0.7) / 0.3 * 0.8))
    d = math.hypot(dx, dy)
    ux, uy = -dx / d, -dy / d
    for j in range(n, 0, -1):
        cc = cols[min(len(cols) - 1, j * len(cols) // max(1, n))]
        px, py = int(round(x + ux * j)), int(round(y + uy * j))
        c.px(px, py, cc)
        if j < n // 3:
            c.px(px, py - 1, cc)
    xi, yi = int(round(x)), int(round(y))
    c.rect(xi - 1, yi - 1, 2, 2, 'white')
    if k < 0.8:
        for ddx, ddy in ((-2, -1), (1, -1), (-1, -2), (-1, 1)):
            c.px(xi + ddx, yi + ddy, cols[1])


def _flame(c, x, y, f, k, seed):
    """A torch flaring up: a tall flickering teardrop of fire with its base at (x, y)."""
    hgt = int(6 + 13 * k)
    w = 1.5 + 2.2 * k
    for j in range(hgt):
        u = j / hgt
        hw = w * (1 - u ** 1.4) * (0.8 + 0.2 * math.sin(f * 0.9 + j * 0.7 + seed))
        lean = int(round(math.sin(f * 0.45 + j * 0.35 + seed * 2) * u * 2.2))
        xl, xr = int(round(x + lean - hw)), int(round(x + lean + hw))
        yy = y - 1 - j
        c.hline(xl, xr, yy, 'vermilion' if u > 0.75 else 'orange')
        if xr - xl >= 2:
            c.hline(xl + 1, xr - 1, yy, 'gold' if u > 0.45 else 'amber')
        if xr - xl >= 4 and u < 0.5:
            c.hline(xl + 2, xr - 2, yy, 'butter' if u > 0.15 else 'white')
    if (f + seed) % 3 == 0:
        c.px(x + int(round(math.sin(f + seed) * 2)), y - hgt - 2, 'gold')


def _ring(c, hx, hy, rad, fade, cols):
    """A dithered ring of light (the flare's ring, in other colours): cols = (outer, inner)."""
    y0, y1 = max(0, int(hy - rad - 3)), min(H, int(hy + rad + 3))
    x0, x1 = max(0, int(hx - rad - 3)), min(W, int(hx + rad + 3))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
    th = bayer(y1 - y0, x1 - x0, x0, y0)
    m = (np.abs(d - rad) < 1.0) & (th < fade)
    m2 = (np.abs(d - rad + 2) < 0.8) & (th < fade * 0.5)
    c.a[y0:y1, x0:x1][m2] = col(cols[1])
    c.a[y0:y1, x0:x1][m] = col(cols[0])


def _ease_extra(a, n, extra):
    """Extra frames after `a` steps of an eased speed-up adding `extra` over n steps (ends flat)."""
    a = min(max(a, 0), n)
    return extra / 2 * (1 - math.cos(math.pi * a / n))


def _tint_between(a, b, q):
    """A tint q (0..1, quantised to eighths: the compositor caches one sprite per tint) from a to b."""
    q = round(clamp(q) * 8) / 8
    return mix(a[0], b[0], q), round(a[1] + (b[1] - a[1]) * q, 3)


# ----------------------------------------------------------------------------- the world clock's lights
# The film painted the castle at midnight with every light on. Seen at another hour (game/daylight.py
# lifts the night scene for the day), each light of the castle is a unit that comes on at its own point
# of the windows' night ramp (nightwin.ramp: on through 19:20 .. 20:00, off through 05:30 .. 06:20):
# off, it shows dark glass / a dark doorway / the rose window's unlit stained glass, and the film's
# halo round it fades in only after it is on (no lamps, halos or glowing windows by day).
UNLIT_GLASS = mix('ink', 'navy', 0.3)          # the castle's own unlit windows
UNLIT_DOOR = mix('ink', 'wine', 0.25)          # the portal arch's dark
LIGHTS = (      # (name, ramp threshold, box (x, y, w, h) in castle px: its warm pixels are the light)
    ('rose', 0.04, (70, 47, 9, 19)),            # the rose window's sun half (lit from inside)
    ('portal', 0.12, (75, 108, 9, 15)),
    ('w1', 0.3, (38, 84, 3, 9)),
    ('w3', 0.46, (106, 100, 3, 7)),
    ('w2', 0.6, (52, 84, 3, 9)),
    ('w4', 0.76, (26, 66, 1, 5)),
)
HALOS = {(79, 112): 'portal', (39, 85): 'w1', (53, 85): 'w2', (107, 100): 'w3'}   # the film's amber glows
LANTERN_THR = 0.08              # the bridge lantern comes on first
HALO_FADE = 0.22                # ramp span over which a light's halo grows once it is on
DUSK_SKY = (19.9, 20.3)         # the clock's dusk sky gives way to the film's midnight sky (and its full
                                # moon) over these hours (daylight.sky_weight drops to 0 at 20.3: a pop)
MAGENTA = np.array((255, 0, 255), np.uint8)


def light_k(hour):
    """0 (day) .. 1 (night): the castle's lights' ramp at a clock hour."""
    return nightwin.ramp(hour)


def unit_on(k):
    """Which castle lights are on at ramp k (a tuple of bools, LIGHTS order)."""
    return tuple(k > thr for _n, thr, _b in LIGHTS)


def unit_fade(k, name):
    """0..1: how far the halo of light `name` (or 'lantern') has grown at ramp k."""
    thr = LANTERN_THR if name == 'lantern' else next(t for n, t, _b in LIGHTS if n == name)
    return clamp((k - thr) / HALO_FADE)


@lru_cache(maxsize=None)
def _units():
    """Each light's pixels in castle px: (ys, xs) of its warm (lit) pixels."""
    a = sc.castle_rgba()
    rgb = a[..., :3].astype(np.int16)
    warm = (a[..., 3] >= 128) & (rgb[..., 0] > rgb[..., 2] + 40)
    out = []
    for _n, _t, (x, y, w, h) in LIGHTS:
        m = np.zeros(warm.shape, bool)
        m[y:y + h, x:x + w] = True
        out.append(np.nonzero(m & warm))
    return tuple(out)


def _unlit(rgb, i):
    """Light i's pixels switched off."""
    if LIGHTS[i][0] == 'rose':              # stained glass with no light behind it: dull ochre and gold
        f = rgb.astype(np.float32)
        return (f * 0.42 + np.array(col(mix('navy', 'ink', 0.4)), np.float32) * 0.58).astype(np.uint8)
    return np.broadcast_to(np.array(col(UNLIT_DOOR if LIGHTS[i][0] == 'portal' else UNLIT_GLASS), np.uint8),
                           rgb.shape)


@lru_cache(maxsize=16)
def _castle_var(on):
    """castle_rgba with the lights that are not `on` switched off (registered for the fast blit)."""
    src = sc.castle_rgba()
    if all(on):
        return src
    a = src.copy()
    for i, (ys, xs) in enumerate(_units()):
        if not on[i]:
            a[ys, xs, :3] = _unlit(a[ys, xs, :3], i)
    return _register(a)


@lru_cache(maxsize=16)
def _strip_var(on):
    """strip_rgba (castle baked in at S_CASTLE) with the castle's lights that are not `on` switched off."""
    src = sc.strip_rgba()
    if all(on):
        return src
    a = src.copy()
    for i, (ys, xs) in enumerate(_units()):
        if not on[i]:
            yy, xx = ys + SY, xs + SX
            a[yy, xx, :3] = _unlit(a[yy, xx, :3], i)
    return _register(a)


def _lit_mask(on, x, y):
    """Screen mask of the castle's lit pixels with the sprite at (x, y) (None if none shows)."""
    m = None
    for i, (ys, xs) in enumerate(_units()):
        if not on[i]:
            continue
        sy, sx = ys + y, xs + x
        ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        if ok.any():
            if m is None:
                m = np.zeros((H, W), bool)
            m[sy[ok], sx[ok]] = True
    return m


def _lantern(c, x, y, f, seed, k):
    """sc.lantern lit by the clock: its halo grows with k (0: unlit, dark glass, no halo)."""
    fl = 0.85 + 0.15 * hash01(seed, f // 3)
    if k > 0.02:
        sc.add_light(c, x, y + 4, 14, 'amber', 0.2 * fl * k, rings=4)
        sc.add_light(c, x, y + 4, 5, 'butter', 0.25 * fl * k, rings=2)
    c.rect(x - 2, y + 1, 5, 6, mix('ink', 'navy', 0.3))
    if k > 0:
        c.rect(x - 1, y + 2, 3, 4, 'gold' if fl > 0.92 else 'amber')
        c.px(x, y + 3, 'butter')
    else:
        c.rect(x - 1, y + 2, 3, 4, mix('steel_dk', 'navy', 0.55))
        c.px(x - 1, y + 2, mix('steel_dk', 'cloud', 0.3))
    c.hline(x - 2, x + 2, y + 1, mix('steel_dk', 'navy', 0.2))
    c.px(x, y, mix('steel_dk', 'navy', 0.2))


def _stamp(a, rgba, x, y):
    """Magenta (never sky) over `a` wherever the registered layer `rgba` at (x, y) is solid."""
    e = _FAST.get(id(rgba))
    if e is None or e[0] is not rgba:
        e = (None, None, (rgba[..., 3] >= 128)[..., None], 0, 0)
    _, _rgb, m, r0, c0 = e
    x, y = int(x) + c0, int(y) + r0
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(a.shape[1], x + m.shape[1]), min(a.shape[0], y + m.shape[0])
    if X1 > X0 and Y1 > Y0:
        np.copyto(a[Y0:Y1, X0:X1], MAGENTA, where=m[Y0 - y:Y1 - y, X0 - x:X1 - x])


def _stamp_wisps(a, f, specs, ox=0, oy=0):
    """Magenta over the wisps (sc.draw_wisps' placement): they are clouds, never open sky."""
    for (w, seed, x0, y, sp) in specs:
        span = W + w + 40
        _stamp(a, sc.wisp_rgba(w, seed), int((x0 - f * sp - ox) % span) - w - 20, y - oy)


@lru_cache(maxsize=96)
def _wisp_lit(w, seed, fh, hq):
    """sc.wisp_rgba for the clock's hour hq: the film's night wisp at night; towards evening, day and
    morning a pale cloud streak in the clock's sky colours (white tops, shadow toward the sky's blue),
    painted so that the relight of the night scene (painted hour fh -> hq) shows those colours."""
    spr = sc.wisp_rgba(w, seed)
    d = clamp((1.0 - daylight.night(hq)) * 1.4)
    if d < 0.02:
        return spr
    cols = daylight._sky_cols(hq)
    top, hz = np.array(cols[0], np.float32), np.array(cols[-1], np.float32)
    lo = np.array(col('cloud_dk'), np.float32) * 0.6 + top * 0.4           # shadowed underside
    mid = np.array(col('cloud'), np.float32) * 0.75 + hz * 0.25
    hi = np.array(col('white'), np.float32) * 0.8 + hz * 0.2
    rgb = spr[..., :3].astype(np.float32)
    op = spr[..., 3] >= 128
    lum = rgb @ np.array([0.3, 0.55, 0.15], np.float32)
    l0, l1 = lum[op].min(), lum[op].max()
    u = np.clip((lum - l0) / max(1.0, l1 - l0), 0, 1)[..., None]
    tgt = np.where(u < 0.5, lo + (mid - lo) * (u * 2), mid + (hi - mid) * (u * 2 - 1))
    out = spr.copy()
    for v in np.unique(out[op][:, :3].reshape(-1, 3), axis=0):     # each painted colour: its relit target
        m = op & np.all(out[..., :3] == v, axis=2)
        t = tgt[m].mean(axis=0) * d + _relit(tuple(int(q) for q in v), fh, hq) * (1 - d)
        out[m, :3] = daylight.unlight(tuple(int(q) for q in np.clip(t, 0, 255)), fh, hq)
    return out


def _relit(rgb, fh, hq):
    """What the relight makes of the colour rgb painted at fh, at the clock's hour hq."""
    a = np.array([[rgb]], np.uint8)
    daylight.relight(a, fh, hq)
    return a[0, 0].astype(np.float32)


class TwoSisters(Region):
    SEG = 'twosisters'
    CARD = ('ANCIENT CASTLE', ['RUINS OF THE TWO SISTERS', 'TREE OF HARMONY BELOW'])
    OWN_MOON = True             # its big full moon (a hotspot): the clock's small one would show beside it

    def build_areas(self):
        return [
            Area(self.render_chasm, cam_x=(0, CAM_A), y_range=(30, 172), start=(96, 84),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=TINT_MOON, hotspots=HOT_A, stamps=STAMPS_A,
                 sky=self.sky_chasm),
            Area(self.render_hall, cam_x=(0, CAM_B), y_range=(24, 148), start=(40, 92),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=TINT_HALL, hotspots=HOT_B, stamps=STAMPS_B),
            Area(self.render_cave, cam_x=(0, 0), cam_y=(0, CAM_CY), y_range=(36, 410), start=(46, 84),
                 landmarks=LANDMARKS_C, balloon='m', balloon_tint=TINT_MOON, hotspots=HOT_C, stamps=STAMPS_C,
                 sky=self.sky_cave),
        ]

    def _fly(self, inp):
        super()._fly(inp)
        if self.ai == 2 and self.vy:            # the tall strip: the balloon rises and sinks twice as fast
            self.by = clamp(self.by + self.vy, *self.area.y_range)

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        self.flare_t0 = -10 ** 6        # step the midnight flare last started
        self.strike_t0 = -10 ** 6       # step of an extra bell strike (SPACE while the flare still blazes)
        self.flared = False             # played this visit to the cave
        self.seen_tree = False

    def on_enter(self):
        self.flared = False

    def on_area(self, ai):
        g = self.g
        if ai == 1:
            g.sfx('wind', 0.3)
        elif ai == 2:
            self.flared = self.seen_tree = False
            # in from the hall: from up in the vault into the sky by the castle, from low in the hall
            # (the stairwell glowing below) straight down into the cave, by the tree
            y0, y1 = self.area.y_range
            k = (self.by - y0) / (y1 - y0)
            self.by = 300 + (k - 0.55) / 0.45 * 100 if k >= 0.55 else 40 + k / 0.55 * 110
            self.cx, self.cy = self._cam_target()
            self.prev = self.snapshot()
            g.sfx('rumble', 0.3)

    def flare_age(self, t):
        return t - self.flare_t0

    def _flare(self, t):
        self.flare_t0 = t
        self.flared = True
        self.g.sfx('bell', 0.6)

    def on_interact(self, key, t):
        g = self.g
        if key == 'moon':
            g.sfx('star', 0.5)
        elif key == 'gust':
            g.sfx('wind', 0.5, 1)
        elif key == 'thrones':
            g.sfx('shimmer', 0.55)
        elif key == 'torches':
            g.sfx('whoosh', 0.5)
        elif key == 'tree':
            if 0 <= self.flare_age(t) < 50:           # the flare is still blazing: one more strike rings out
                self.strike_t0 = t
                g.sfx('bell', 0.5)
            else:
                self._flare(t)

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.switch is not None:
            return
        if self.ai == 0:
            a = t - self.spot_t.get('moon', -10 ** 6)
            if a == 12:
                g.sfx('shimmer', 0.35)
            a = t - self.spot_t.get('gust', -10 ** 6)
            if a in (8, 34):
                g.sfx('creak', 0.45 if a == 8 else 0.3)
            elif t % 150 == 40:
                g.sfx('creak', 0.3 * self.near_gain(BRIDGE_TAG[0] + 40))
            if t % STAR_PERIOD == 100:
                g.sfx('star', 0.25)
            if t % 420 == 300:
                g.sfx('wind', 0.22)
            self.area.balloon_tint = TINT_MOON
        elif self.ai == 1:
            a = t - self.spot_t.get('thrones', -10 ** 6)
            if a == 6:
                g.sfx('crystal', 0.45)
            a = t - self.spot_t.get('torches', -10 ** 6)
            if a == 16:
                g.sfx('whoosh', 0.35, 1)
            if t % 300 == 150:
                g.sfx('shimmer', 0.18)
            d = min(abs(self.bx - tx) for tx, _ in sc.B_TORCH)
            warm = clamp(1 - d / 60) * clamp(1 - abs(self.by - 100) / 70)
            if 0 <= t - self.spot_t.get('torches', -10 ** 6) < 60:
                warm = max(warm, 0.6)
            self.area.balloon_tint = _tint_between(TINT_HALL, TINT_TORCH, warm)
        elif self.ai == 2:
            if not self.seen_tree and self.by > 250:
                self.seen_tree = True
                if self.by < 300:                       # the tree comes into view (film 270)
                    g.sfx('crystal', 0.5)
            if not self.flared and self.by > 300:
                self._flare(t)
            a = self.flare_age(t)
            if a == 3:
                g.sfx('crystal', 0.5)
            elif a == 8:
                g.sfx('shimmer', 0.55)
            elif a in (18, 36):                          # the later bell strikes
                g.sfx('bell', 0.32 if a == 18 else 0.24)
            if t % STAR_PERIOD == 100 and self.cy < 30:
                g.sfx('star', 0.25)
            if t % 360 == 200 and self.by > 250:
                g.sfx('sparkle', 0.22)
            self.area.balloon_tint = self._tint_cave(t)

    def _tint_cave(self, t):
        by = self.by
        if by < sc.GROUND + 8:
            tint = TINT_MOON
        elif by < sc.CAVE_TOP + 20:
            tint = _tint_between(TINT_MOON, TINT_ROCK, (by - sc.GROUND - 8) / 40)
        else:
            tint = _tint_between(TINT_ROCK, TINT_CAVE, (by - sc.CAVE_TOP - 20) / 50)
        a = self.flare_age(t)
        if 0 <= a < 60 and by > sc.CAVE_TOP:
            tint = _tint_between(tint, TINT_FLARE, 1 - a / 60)
        return tint

    # ------------------------------------------------------------------ shooting stars
    def _sky_stars(self, c, t, ox, oy):
        """The film's shooting star, every STAR_PERIOD steps somewhere over the castle (sky px - ox/oy)."""
        n, a = divmod(t, STAR_PERIOD)
        if 100 <= a < 114:
            hx = 220 + int(hash01(n, 71) * 120)
            hy = 10 + int(hash01(n, 72) * 24)
            _meteor(c, (a - 100) / 14, hx - ox, hy - oy)

    def moon_up(self):
        """The painted full moon (and its halo over the castle) shows only while the world clock says
        night; by day the repainted sky has the sun (game/daylight.py)."""
        from .. import daylight
        return daylight.night(self.g.hour) > 0.4

    # ------------------------------------------------------------------ the world clock (game/daylight.py)
    _lk = None                  # (frame as rendered, its lit castle pixels, first underground row) for after_light

    def _clock(self):
        """This frame's lights by the world clock: ramp k (0 day .. 1 night), which castle lights are on,
        night() nk, the hour step the wisps are coloured for (None: as painted) and the film's fx with
        its dark vignette fading by day."""
        hour = self.g.hour
        k = light_k(hour)
        nk = daylight.night(hour)
        hq = None if abs(hour - self.HOUR) < 0.05 else daylight._q(hour)
        return dict(k=k, on=unit_on(k), nk=nk, hq=hq, fx=_fx_clock(round(nk, 1)))

    def light_tint(self):
        """Down in the rock and the cave (shown as painted at every hour, after_light) the balloon keeps
        the cave's own light, blended in over the first 30 px below the ground."""
        base = super().light_tint()
        if self.ai != 2:
            return base
        w = clamp((self.by - sc.GROUND - 8) / 30)
        return daylight.mix_tint(base, self.area.balloon_tint, w) if w > 0 else base

    def _wisps(self, c, f, specs, ox, oy, lk):
        """sc.draw_wisps in the clock's cloud colours (_wisp_lit); oy: the sky layer's parallax offset."""
        for (w, seed, x0, y, sp) in specs:
            spr = sc.wisp_rgba(w, seed) if lk['hq'] is None else _wisp_lit(w, seed, self.HOUR, lk['hq'])
            span = W + w + 40
            c.blit(spr, int((x0 - f * sp - ox) % span) - w - 20, y - oy)

    @staticmethod
    def _dusk_k(h):
        """0..1: how far the film's midnight sky has taken over from the clock's dusk sky (DUSK_SKY)."""
        k = clamp((h - DUSK_SKY[0]) / (DUSK_SKY[1] - DUSK_SKY[0])) if h >= 12 else 0.0
        return k * k * (3 - 2 * k)

    def _keep(self, c, mask, under=None):
        """At the end of an area render: the castle's lit pixels (mask) and the first underground row
        (under, area 2) get their painted colour back right after the clock's light (after_light)."""
        self._lk = (c.a.copy(), mask, under)

    def _world(self, t, cam=None, with_balloon=False):
        self._lk = None
        return super()._world(t, cam, with_balloon)

    def after_light(self, c, t):
        """The relight lifts the midnight scene for other hours and dims its warm lamps toward dark
        glass (daylight._lamps): the lights that are on (light_k) keep their painted colour, so at
        dusk and dawn the lit windows shine as at midnight; under the ground (the rock, the stair shaft,
        the cave and the Tree of Harmony, lit by the tree) everything stays as painted at every hour."""
        g, self._lk = self._lk, None
        if g is None:
            return
        pre, m, under = g
        h = self.g.hour
        u = self._dusk_k(h)
        if 0 < u < 1 and self.area.sky is not None and daylight.sky_weight(self.HOUR, h) > 0.01:
            ys, xs = np.nonzero(self._sky_memo[1])      # (this frame's open sky)
            px = pre[ys, xs][None].copy()
            daylight.relight(px, self.HOUR, h, t)
            cur = c.a[ys, xs].astype(np.float32)
            c.a[ys, xs] = (cur + (px[0] - cur) * u).astype(np.uint8)
        if under is not None and under < H:
            y0 = max(0, under)
            c.a[y0:] = pre[y0:]
        if m is not None:
            ys, xs = np.nonzero(m)
            p = pre[ys, xs].astype(np.int16)
            warm = p[:, 0] > p[:, 2] + 40               # still the lit glass (nothing drawn over it)
            c.a[ys[warm], xs[warm]] = pre[ys[warm], xs[warm]]

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_chasm(self, f, cx, cy):
        """render_A's sky alone at the player's camera: gradient, stars, the (blazing) moon, and the
        shooting stars / moon shower the region draws over the wisps. The wisps (clouds), the forests,
        the crag, the castle, the near cliff and the foreground are stamped out: their darkest navy is
        within the sky test's tolerance of the night sky (paint_sky speckled them with the clock's sky)."""
        t = self.g.t
        cam = cx
        ma = t - self.spot_t.get('moon', -10 ** 6)
        p = sc.A_P
        mx, my, mr = sc.A_MOON
        mxs = int(mx - cam * p['moon'])
        if 0 <= ma < MOON_FX:                       # the blaze: drawn in full
            c = sc.sky_bg().copy()
            sc.draw_stars(c, f, 3, 150, W + 20, 120, ox=int(cam * p['stars']), mask_moon=(mxs, my, mr))
            self._moon_a(c, ma, f, mxs, my, mr)
        else:                                       # cached gradient + moon, the stars lit by its glow
            base, delta = _chasm_sky(mxs if self.moon_up() else None)
            st = Canvas(arr=np.zeros((H, W, 3), np.uint8))
            sc.draw_stars(st, f, 3, 150, W + 20, 120, ox=int(cam * p['stars']), mask_moon=(mxs, my, mr))
            c = Canvas(arr=base.copy())
            s = st.a[:121]                          # the stars' rows
            m = (s[..., 0] | s[..., 1] | s[..., 2]) > 0
            c.a[:121][m] = np.clip(s[m] + delta[:121][m], 0, 255)
        self._sky_stars(c, t, int(cam * 0.1), 0)
        if 0 <= ma < MOON_FX:
            for (s0, x, y, cols) in SHOWER:
                _streak(c, (ma - s0) / SHOWER_LEN, x - int(cam * P['moon']) + 20, y, cols)
        _layers()
        _ga, _gk, gx = self._gust(t)
        _stamp_wisps(c.a, f + int(gx * 0.5), WISPS_A, int(cam * 0.15))
        cxs = int(AX - cam * P['castle'])
        for lay, x, y in ((_far_wide(), -int(cam * P['far']), 0), (_mid_wide(), -int(cam * P['mid']), 0),
                          (sc.cragA(), cxs - 40, 100), (sc.castle_rgba(), cxs, AY), (sc.nearA(), -cam, 0),
                          (sc.fgA(), -int(cam * P['fg']), 0)):
            _stamp(c.a, lay, x, y)
        c.a[GORGE_Y:] = 255 - c.a[GORGE_Y:]     # below the forest tops the gradient shows only down the gorge
        return c

    def sky_cave(self, f, cx, cy):
        """render_C's sky alone (the top of the strip, over the castle): sky_s's gradient without its
        baked far forest, stars, moon and the shooting star. The wisps, the strip (castle, crag, rock,
        cave) and the foreground crystals are stamped out (the strip's dark stone and rock seams are
        within the sky test's tolerance of the night sky)."""
        cam = cy
        c = Canvas(W, H, (255, 0, 255))
        c.blit(_sky_s_only(), 0, -int(cam * 0.5))
        mx, my, mr = sc.S_MOON
        msy = my - int(cam * 0.5)
        if msy + mr * 3 > 0:
            sc.draw_stars(c, f, 21, 120, W, 160, oy=int(cam * 0.5), mask_moon=(mx, msy, mr))
            if self.moon_up():
                sc.draw_moon(c, f, mx, msy, mr, halo=1.0)
        if cam < 60:
            self._sky_stars(c, self.g.t, 0, int(cam * 0.5))
        _layers()
        _stamp_wisps(c.a, f, WISPS_C, 0, int(cam * 0.5))
        _stamp(c.a, sc.strip_rgba(), 0, -cam)
        _stamp(c.a, sc.fg_crystals(), 0, 470 - int(cam * 1.2))
        return c

    # ------------------------------------------------------------------ area 0: the chasm
    def render_chasm(self, f, cx, cy):
        t = self.g.t
        cam = cx
        ga, gk, gx = self._gust(t)
        ma = t - self.spot_t.get('moon', -10 ** 6)
        lk = self._clock()
        k, on, nk = lk['k'], lk['on'], lk['nk']
        cxs = int(AX - cam * P['castle'])                  # the castle sprite's screen x (render_A's cx)
        lamp = []

        def wisps(c, f_, specs, ox=0):
            self._wisps(c, f_ + int(gx * 0.5), specs, ox, 0, lk)
            self._sky_stars(c, t, int(cam * 0.1), 0)
            if 0 <= ma < MOON_FX:
                for (s0, x, y, cols) in SHOWER:
                    _streak(c, (ma - s0) / SHOWER_LEN, x - int(cam * P['moon']) + 20, y, cols)

        def moon(c, f_, x, y, r, halo=1.0):
            if self.moon_up() or 0 <= ma < MOON_FX:    # by day only SPACE's blaze shows it
                self._moon_a(c, ma, f_, x, y, r, halo)

        def bridge(c, f_, a, b, sag=9.0):
            sc.rope_bridge(c, f_, a, b, sag + 6 * gk * math.sin(ga * 0.33))

        def vine(c, x, y0, L, f_, seed, ramp=sc.MOSS_N, amp=1.2):
            sc.sway_vine(c, x, y0, L, f_ + int(gx), seed, ramp, amp + 2.2 * gk)

        def lantern(c, x, y, f_, seed=0):
            x += int(round(2 * gk * math.sin(ga * 0.4)))
            _lantern(c, x, y, f_, seed, unit_fade(k, 'lantern') if k > LANTERN_THR else 0.0)
            if k > LANTERN_THR:
                lamp.append((x - 1, y + 2))

        def mist(c, f_, *a, **k_):
            sc.mist(c, f_ + int(gx * 2.5), *a, **k_)

        def motes(c, f_, region, n, seed, **k_):         # the film's motes, and more east of them (fewer,
            n = max(2, int(n * (0.3 + 0.7 * nk)))         # and without their glow, by day)
            k_['glow'] = nk > 0.5
            sc.motes(c, f_, region, n, seed, **k_)
            x0, y0, w, h = region
            sc.motes(c, f_, (x0 + w, y0, 160, h), n // 2, seed + 50, **k_)

        def add_light(c, x, y, r, color, kk, rings=4):  # the film's amber glows round the lit windows
            u = HALOS.get((x - cxs, y - AY))
            sc.add_light(c, x, y, r, color, kk * (1.0 if u is None else unit_fade(k, u)), rings)

        def window_glow(c, x, y, f_, kk=1.0):
            sc.window_glow(c, x, y, f_, kk * unit_fade(k, 'rose'))

        render = rebind(sc.render_A, camA=lambda _f: cam, sky_bg=_sky, draw_balloon=_noop, callout=_noop,
                        iris=_noop, fx=lk['fx'], draw_wisps=wisps, draw_moon=moon, rope_bridge=bridge,
                        sway_vine=vine, lantern=lantern, mist=mist, motes=motes, farA=_far_wide, midA=_mid_wide,
                        castle_rgba=lambda: _castle_var(on), add_light=add_light, window_glow=window_glow,
                        pennant=lambda c, x, y, f_, color='royal_lt': sc.pennant(c, x, y, f_ + int(gx), color))
        c = render(f)
        if 0 <= ga < GUST:
            self._gust_fx(c, t, ga, cam)
        m = _lit_mask(on, cxs, AY)
        for (lx, ly) in lamp:                           # the lantern's glass
            if m is None:
                m = np.zeros((H, W), bool)
            m[max(0, ly):max(0, ly + 4), max(0, lx):max(0, lx + 3)] = True
        self._keep(c, m)
        return c

    def _gust(self, t):
        """(steps since the gust, its strength 0..1..0, extra flapping frames)."""
        ga = t - self.spot_t.get('gust', -10 ** 6)
        gk = math.sin(math.pi * clamp(ga / GUST)) if 0 <= ga < GUST else 0.0
        return ga, gk, _ease_extra(ga, GUST, 70)

    def _moon_a(self, c, ma, f_, x, y, r, halo=1.0):
        """The chasm's full moon (sc.draw_moon), blazing for MOON_FX steps after SPACE at it."""
        k = 0.0
        if 0 <= ma < MOON_FX:
            k = min(1.0, ma / 5) * (1 - seg(ma, 10, MOON_FX))
        if k > 0.02:
            sc.add_light(c, x, y, int(r * 3.2), mix('royal', 'periwinkle', 0.4), 0.25 * k, rings=6)
        sc.draw_moon(c, f_, x, y, r, halo * (1 + 2.5 * k))
        if k > 0.02:
            sc.bloom(c, x, y, r * 2.2, 0.55 * k, mix('cloud', 'lilac', 0.35))
            sc.add_light(c, x, y, int(r * 1.3 + 6 * k), 'cloud', 0.22 * k, rings=4)
            self._moon_fx(c, ma, x, y, r)

    def _moon_fx(self, c, ma, x, y, r):
        """The full moon blazes: rings of light roll out from its rim and stars twinkle round it."""
        for j, cc in enumerate(('ice', 'lilac', 'baby_blue')):
            a = ma - j * 9
            if 0 <= a < 34:
                rad = r + 3 + a * 1.9
                fade = 1 - a / 34
                for i in range(int(rad * 3)):
                    ang = i / (rad * 3) * 6.283
                    if hash01(i, j, 5) < fade:
                        c.px(int(round(x + math.cos(ang) * rad)), int(round(y + math.sin(ang) * rad)), cc)
        for i in range(12):
            ang = hash01(i, 61) * 6.283
            rad = r + 8 + hash01(i, 62) * 26
            sx, sy = int(x + math.cos(ang) * rad), int(y + math.sin(ang) * rad * 0.9)
            ph = (ma + int(hash01(i, 63) * 16)) % 16
            if ma < MOON_FX - 8:
                fx.sparkle(c, sx, sy, ph, ('butter', 'white', 'crystal_pk', 'ice')[i % 4], period=16, size=3)

    def _gust_fx(self, c, t, ga, cam):
        """The gust: leaves torn from the gnarled tree blow east over the chasm, streaks of wind."""
        leaves = (sc.LEAF_N[3], sc.LEAF_N[4], 'sage', 'mint', 'gold', sc.MOSS_N[2])
        for i in range(48):
            a = ga - int(hash01(i, 31) * 30)                 # leaves come off over the first second
            if a < 0:
                continue
            x0 = 6 + hash01(i, 32) * 70 - cam
            y0 = 14 + hash01(i, 33) * 34
            vx = 2.2 + 2.6 * hash01(i, 34)
            px = x0 + vx * a
            py = y0 + a * (0.25 + 0.5 * hash01(i, 35)) + math.sin(a * 0.3 + i) * 4
            if px > W + 4 or py > H:
                continue
            cc = leaves[i % len(leaves)]
            c.rect(int(px), int(py), 2, 2 if (a // 3 + i) % 2 else 1, cc)
        for i in range(22):
            x = -60 + int((ga * 6 + hash01(i, 8) * 300) % 420)
            y = 40 + int(hash01(i, 9) * 130)
            if ga < GUST - 8 and (ga + i) % 5:
                L = 10 + int(hash01(i, 10) * 18)
                c.hline(x, x + L, y, mix('periwinkle', 'cloud', 0.4) if i % 3 else 'cloud')
                c.hline(x + L // 2, x + L + 4, y - 1, mix('periwinkle', 'indigo', 0.2))

    # ------------------------------------------------------------------ area 1: the old throne hall
    def render_hall(self, f, cx, cy):
        t = self.g.t
        cam = cx
        ta = t - self.spot_t.get('torches', -10 ** 6)
        ra = t - self.spot_t.get('thrones', -10 ** 6)
        rk = 0.0                                                # the rose window's blaze 0..1
        if 0 <= ra < 90:
            rk = min(1.0, ra / 6) * (1 - seg(ra, 30, 90))
        draught = _ease_extra(ta, 60, 50)

        def glow_window(c, x, y, f_, k=1.0):
            _add(c, _beams_b(), cam, 0, 0.85 + 0.15 * math.sin(f_ / 23.0))      # the moonbeams (cached)
            sc.window_glow(c, x, y, f_, k * (1 + 2.2 * rk))
            if rk > 0.02:
                self._rose_blaze(c, t, ra, rk, -cam)

        def flame(c, x, y, f_, seed):
            a = ta - seed * 10                          # the flare runs along the hall, torch by torch
            k = 0.0
            if 0 <= a < 70:
                k = min(1.0, a / 3) * (1 - seg(a, 20, 70))
            if k > 0.02:
                sc.add_light(c, x, y - 10, int(40 + 50 * k), 'orange', 0.16 * k, rings=6)
                sc.add_light(c, x, y - 8, int(14 + 14 * k), 'gold', 0.3 * k, rings=4)
                _flame(c, x, y + 1, f_, k, seed)
            else:
                sc.torch_flame(c, x, y, f_, seed)
            if 0 <= a < 56:
                ui.burst(c, t, self.spot_t['torches'] + seed * 10, x, y - 10, ('gold', 'amber', 'butter', 'orange'),
                         n=22, seed=40 + seed, power=0.75, life=50, gravity=-0.035)

        def banner(c, x, y, f_, w, L, kind, seed):
            _banner(c, x, y, f_ + int(draught), w, L, kind, seed, 1.6 + 1.8 * clamp(1 - abs(ta - 25) / 35))
            if rk > 0.05:
                bx = x + w // 2
                sc.add_light(c, bx, y + L // 3, 9 + int(4 * rk), 'gold' if kind == 'sun' else 'crystal',
                             0.35 * rk, rings=3)

        sky = self._hall_sky(f, cam)
        kw = {} if sky is None else dict(sky_bg=lambda: sky, draw_stars=_noop, draw_moon=_noop)
        render = rebind(sc.render_B, camB=lambda _f: cam, **{'sky_bg': _sky, **kw}, draw_balloon=_noop,
                        callout=_noop, fx=FX_NOBEAM, window_glow=glow_window, torch_flame=flame, draw_banner=banner)
        c = render(f)
        if 0 <= ra < 90:
            self._throne_fx(c, t, ra, rk, -cam)
        return c

    def _hall_sky(self, f, cam):
        """The sky seen through the hall's broken vault and windows for the world clock's hour. The hall
        is an interior (game/daylight.py INTERIORS: never re-lit, it keeps its torches and moonbeams), so
        the clock's sky is painted here: render_B's own night sky (gradient, stars, the full moon) with
        daylight.paint_sky's sky for the hour blended over it (sun by day, its moon and stars by night).
        None at the painted hour (the film's sky steps as they are)."""
        from .. import daylight
        hour = self.g.hour
        w = daylight.sky_weight(self.HOUR, hour) * (1 - self._dusk_k(hour))
        if w <= 0.01:
            return None
        c = _FC(arr=sc.sky_bg().a.copy())
        mxs = 120 - int(cam * 0.25)                         # (render_B's own sky steps)
        sc.draw_stars(c, f, 9, 120, W + 40, 120, ox=int(cam * 0.2), mask_moon=(mxs, 30, 24))
        sc.draw_moon(c, f, mxs, 30, 24, halo=0.9)
        daylight.paint_sky(c.a, _ALL_SKY, hour, w, cam, 0, -10, 130, self.g.t)
        return c

    def _rose_blaze(self, c, t, ra, rk, x0):
        """The rose window blazes: sun-gold and moon-cyan light pours down onto the thrones."""
        rx = x0 + RX
        sc.add_light(c, rx - 6, RY, int(RR * 2 + 14 * rk), 'gold', 0.3 * rk, rings=5)
        sc.add_light(c, rx + 6, RY, int(RR * 2 + 14 * rk), 'crystal', 0.3 * rk, rings=5)
        fx.light_beam(c, ((rx - RR + 2, RY + 8), (rx - 2, RY + 8)), ((rx - 30, 142), (rx - 4, 142)),
                      'gold', strength=0.22 * rk)
        fx.light_beam(c, ((rx + 2, RY + 8), (rx + RR - 2, RY + 8)), ((rx + 4, 142), (rx + 30, 142)),
                      'crystal', strength=0.22 * rk)

    def _throne_fx(self, c, t, ra, rk, x0):
        t0 = self.spot_t['thrones']
        (sx, sy), (mx, my) = THRONES
        for (ex, ey, cc, hi) in ((sx, sy, 'gold', 'butter'), (mx, my, 'crystal', 'crystal_lt')):
            sc.add_light(c, x0 + ex, ey, int(10 + 12 * rk), cc, 0.45 * rk, rings=4)
            fx.sparkle(c, x0 + ex, ey, ra, 'white', period=24, size=4)
        for (lx, cc) in ((-16, 'amber'), (14, 'crystal')):           # the dais floods with colour
            c.ellipse(x0 + RX + lx, 143, int(13 + 6 * rk), 2, mix(cc, 'indigo', 0.55 - 0.3 * rk))
            c.ellipse(x0 + RX + lx, 143, int(8 + 4 * rk), 1, mix(cc, 'white', 0.2 * rk))
        ui.burst(c, t, t0, x0 + RX - 6, RY, ('gold', 'butter', 'amber', 'white'), n=26, seed=51, power=0.9,
                 life=56, gravity=0.06)
        ui.burst(c, t, t0, x0 + RX + 6, RY, ('crystal_lt', 'ice', 'crystal', 'white'), n=26, seed=52, power=0.9,
                 life=56, gravity=0.06)
        if ra < 16:                                                    # a ring of light from the window
            r = 4 + ra * 2.4
            c.circle(x0 + RX, RY, int(r), 'butter' if ra % 4 < 2 else 'white', fill=False)

    # ------------------------------------------------------------------ area 2: the cave of harmony
    def render_cave(self, f, cx, cy):
        t = self.g.t
        cam = cy
        a = self.flare_age(t)
        mid = f - a if 0 <= a < 400 else f - 10 ** 4          # the film's MIDNIGHT on the flare clock
        wb = 0.5 * seg(a, 40, 56) * (1 - seg(a, 260, 330)) if a >= 0 else 0.0
        lk = self._clock()
        k, on = lk['k'], lk['on']
        oy = SY - cam                                   # the strip castle's screen y

        def wisps(c, f_, specs, ox=0):
            self._wisps(c, f_, specs, ox, int(cam * 0.5), lk)
            if cam < 60:
                self._sky_stars(c, t, 0, int(cam * 0.5))

        def glow_window(c, x, y, f_, kk=1.0):         # the rose window's halo (once its light is on)
            sc.window_glow(c, x, y, f_, (kk + wb * (0.8 + 0.2 * math.sin(f_ / 8.0))) * unit_fade(k, 'rose'))

        def add_light(c, x, y, r, color, kk, rings=4):  # the portal's amber glow (the cave's own: as painted)
            u = HALOS.get((x - SX, y - oy)) if color == 'amber' else None
            sc.add_light(c, x, y, r, color, kk * (1.0 if u is None else unit_fade(k, u)), rings)

        vein = rebind(sc.vein_pulse, MIDNIGHT=mid)
        flare0 = rebind(sc.flare, MIDNIGHT=mid)

        def flare(c, f_, hx, hy):
            flare0(c, f_, hx, hy)
            for (b0, cols) in BELL_RINGS:              # the later bell strikes ring out from the heart
                if 0 <= a - b0 < 30:
                    _ring(c, hx, hy, (a - b0) * 6.5, 1 - (a - b0) / 30, cols)
            sa = t - self.strike_t0
            if 0 <= sa < 34:                           # an extra strike: the heart blazes, a gold ring
                sc.bloom(c, hx, hy - 10, 60, 0.6 * (1 - sa / 34), mix('lilac', 'white', 0.6))
                _ring(c, hx, hy, sa * 6.5, 1 - sa / 34, ('butter', 'gold'))
                if sa >= 6:
                    _ring(c, hx, hy, (sa - 6) * 6.5, 1 - sa / 34, ('crystal_pk', 'lilac'))
        beam = _beam_c()

        fxc = _Fx(beams=False)
        fxc.vignette = lk['fx'].vignette

        def beam_then(c, *a_, **k_):            # the fissure's moonbeam, cached (drawn where the film does)
            _add(c, beam, 0, cam)
        fxc.light_beam = beam_then

        def moon(c, *a_, **k_):                 # the painted moon only while the clock says night
            if self.moon_up():
                sc.draw_moon(c, *a_, **k_)

        render = rebind(sc.render_C, cam_s=lambda _f: cam, Canvas=_FC, draw_balloon=_noop, callout=_noop,
                        draw_moon=moon,
                        fx=fxc, draw_wisps=wisps, window_glow=glow_window, vein_pulse=vein, flare=flare,
                        add_light=add_light, strip_rgba=lambda: _strip_var(on),
                        MIDNIGHT=mid, STAR0=10 ** 9, RISE0=10 ** 9, RISE1=10 ** 9)
        _layers()
        c = render(f)
        self._keep(c, _lit_mask(on, SX, oy), sc.GROUND + 1 - cam)
        return c
