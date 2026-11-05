"""The Dragon Lands as an explorable diorama (art: scenes/s11_dragonlands.py, used read-only).

The film shows three separate views joined by a zoom and a cloud, so the region has three areas, flown
through left to right with a cloud sweep between them (film order):
  0  THE VOLCANO ISLAND FROM THE SEA (shot A): the basalt cone with its glowing rivulets and plume over the
     jungle fringe, black-sand beaches and the rippling sea. The pan runs on past the film's framing to
     the island's eastern beach and the open sea, with the foreground sea stack sliding by.
  1  INSIDE THE VOLCANIC HEART (shot B): the caldera with the empty throne ledge, the lavafall, the lava
     river and the gem hoards.
  2  KIRIN GROVE (shot C): karst towers, the Lost Lagoon, the pavilion, the stream and the moon bridge.
The film's render switches content by shot and draws its timed beats inline, so each area re-composes
the scene's own cached layers and helpers itself, in the film's layer order, at the player's camera
(recompose style, see sweetapple.py). Timed beats on game clocks:
  - the ERUPTION (film 198..290: rumble, pool swell, lava fountain, ember shower, a warm glow) goes off
    by itself once per visit, shortly after you first fly into the caldera; after that only SPACE at the
    hot pool wakes it. No screen shake (the film's tremor/shake is left out: the world never moves) and
    the glow swells in over 6 steps at a third of the film's flash and eases out (comfort); the second
    smoke plume from the pool keeps billowing all the time so nothing pops when the eruption starts;
  - the little sailing ship (no crew) crosses the far horizon behind the island on a long loop instead
    of sailing off for good (the film draws it on top of everything; here it passes behind the island).
Things to set off with SPACE (film effects reused where they exist):
  - THE VOLCANO's crater (sea): flash and shock ring, a short lava fountain, lava bombs landing glowing on
    the flanks, an underlit ash billow, a bright front racing down the rivulets;
  - the sea stack (sea): a rogue wave throws a wall of spray up the rock, mist drifts off, and a little
    rainbow hangs in the spray (opposite the low sun);
  - the hot pool (caldera): wakes the ERUPTION now (rumble, pool swell, fountain, ember shower, glow);
  - the gem hoards (caldera): a golden flare, shafts of light and a fountain of gems and coins;
  - the lantern string (grove): a wave of light runs to the pavilion, the moon windows blaze and paper
    sky lanterns rise into the sky (wind chimes).
Three atlas stamps: up in the ash plume east of the crater, over the empty throne's spire, and low
under the moon bridge.
The world clock (game/daylight.py): every area starts from the clock's own sky (no painted low sun on the
wrong side, a sea that reflects the hour), and keeps its real lights itself (lava, fires, lit lanterns and
windows: marked by the renders, restored in after_light); the lanterns and windows follow the clock.
The film's balloon is never drawn (the base class draws the player's), nor its callouts, card or iris.
All four film tags sit on the 0.6 parallax layers (island, ledges, grove bank), hence Landmark(p=0.6).
Speed (~3-4 ms per frame): layers composited inside their opaque bounding box, the colour grades via a
lookup table (same pixels as the film's comp / fx.grade), the beach foam vectorised, every cached layer
built when the region loads. No creatures anywhere (landscape only, per the edition's rule).
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, fx, seg, lerp, hash01, col, Sprite
import scenes.s11_dragonlands as sc

from engine import timeline as TL

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# ----------------------------------------------------------------------------- camera limits
#   sea:   island 380 px (p .6) ends at cam 100, but it only holds the island (44..374) on a clear sky:
#          past it the open sea and the far islets carry on (sky, sea, waves, reflection are full-screen)
#   heart: wall 380 (p .3, -20) -> 133; mid 380 (p .6, -6) -> 90; fg 410 (p 1.3, -8) -> 63; the right crater
#          wall closes over the gem hoards past ~45 (the eruption does not shake the world)
#   grove: bank 400 (p .6, -10) -> 116; mid karsts 380 (p .35, -8) -> 148; lagoon 360 (p .2, -10) -> 150;
#          far karsts 360 (p .12, -10) -> 250; fg boulders split in a left and a right group (see _fg_parts)
SEA_CAM = 150
HEART_CAM = 40
GROVE_CAM = 110
PAINTED_H = TL.hour_at('dragonlands', TL.seg_frames('dragonlands') // 2)     # = Region.HOUR (15.63)

# ----------------------------------------------------------------------------- clocks
ERUPT_LEAD = 170        # eruption clock on arrival: calm, rumble at 198, fountain at 216 (film frames)
ERUPT_ERUPT_AT = 208    # SPACE at the pool: the clock jumps here (tremor on, fountain 8 steps later)
ERUPT_PERIOD = 432      # 24 beats (~14 s): all calm again by ~290; the clock then rests at ERUPT_CALM
ERUPT_CALM = ERUPT_LEAD + ERUPT_PERIOD      # (no automatic repeat: once per visit, then SPACE only)
FLASH_IN, FLASH_OUT, FLASH_K = 6, 20, 1 / 3     # eruption glow: ramp in, ease out, a third of the film's
SHIP_SPEED, SHIP_SPAN, SHIP_PHASE = 0.09, 560, 40

VOLC_X = 260            # world x (main layer) with the crater near the screen centre, for sound
GEM_X = 286
GROVE_X = 250

LANDMARKS_SEA = [       # shot A callouts (island layer, p 0.6: anchors in island px)
    Landmark('DRAGON LANDS', 104, (98, 112, 'down', 10), p=sc.A_PAR),
    Landmark('THE VOLCANO', 236, (sc.VX + 6, sc.VT + 1, 'right', 14), p=sc.A_PAR),
]
LANDMARKS_HEART = [     # shot B (mid ledges, p 0.6, offset -6)
    Landmark('GEM HOARDS', GEM_X, (276 - 6, 96, 'up', 14), p=0.6),
]
_PVX, _PVY = 196 - 10, sc.BANK - 62 + 2
LANDMARKS_GROVE = [     # shot C (grove bank, p 0.6, offset -10)
    Landmark('KIRIN GROVE', GROVE_X, (_PVX + 38, _PVY + 1, 'up', 12), p=0.6),
]


# ----------------------------------------------------------------------------- small helpers
def _h01(*xs):
    """engine.hash01 over numpy integer arrays (same values)."""
    h = np.full(np.broadcast(*[np.asarray(x) for x in xs]).shape, 2166136261, np.uint64)
    for x in xs:
        h ^= (np.asarray(x, np.int64) & 0xffffffff).astype(np.uint64)
        h = (h * np.uint64(16777619)) & np.uint64(0xffffffff)
        h ^= h >> np.uint64(13)
        h = (h * np.uint64(0x5bd1e995)) & np.uint64(0xffffffff)
        h ^= h >> np.uint64(15)
    return (h & np.uint64(0xffffff)).astype(np.float64) / float(0x1000000)


_RGB = {}


def _c(name):
    v = _RGB.get(name)
    if v is None:
        v = _RGB[name] = np.array(col(name), np.uint8)
    return v


def _nop(*_a, **_k):
    pass


_BBOX = {}


def _comp(c, layer, x, y=0):
    """The scene's comp() (copy a cached (rgb, mask) layer to screen (x, y)), clipped to the bounding
    box of the layer's opaque pixels (the scene's layers are mostly empty rows: ~3x faster)."""
    a, m = layer
    bb = _BBOX.get(id(m))
    if bb is None:
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if rows.size:
            bb = (m, rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
        else:
            bb = (m, 0, 0, 0, 0)
        _BBOX[id(m)] = bb
    _, r0, r1, c0, c1 = bb
    x, y = int(x) + c0, int(y) + r0
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(W, x + c1 - c0), min(H, y + r1 - r0)
    if X1 <= X0 or Y1 <= Y0:
        return
    sm = m[r0 + Y0 - y:r0 + Y1 - y, c0 + X0 - x:c0 + X1 - x]
    np.copyto(c.a[Y0:Y1, X0:X1], a[r0 + Y0 - y:r0 + Y1 - y, c0 + X0 - x:c0 + X1 - x], where=sm[:, :, None])


@lru_cache(maxsize=64)
def _grade_lut(mul, add):
    v = np.arange(256, dtype=np.float32)[None, :] * np.array(mul, np.float32)[:, None] + \
        np.array(add, np.float32)[:, None]
    return np.clip(v, 0, 255).astype(np.uint8)


_CH = np.arange(3)


def _grade(c, mul, add):
    """engine.fx.grade (same pixels) through a per-channel lookup table."""
    c.a[...] = _grade_lut(tuple(mul), tuple(add))[_CH, c.a]


_draw_bamboo = rebind(sc.draw_bamboo, comp=_comp)


# ============================================================================= area 0: the sea
@lru_cache(maxsize=1)
def _beach():
    """Static per-column data of the film's beach swash (island px 0..IW)."""
    xi = np.arange(sc.IW)
    s = xi // 9
    on = _h01(s, 3) < 0.7
    amp = 1 + _h01(s, 4) * 2.5
    wet = _h01(xi, 9) < 0.5
    mist = _h01(xi, 5) < 0.5
    return on, amp, wet, mist


def _beach_waves(c, f, ox):
    """The film's foam swashes running up the black sand and back (vectorised)."""
    ph = (f % 72) / 72
    run = math.sin(ph * math.pi) ** 2
    on, amp, wet, mist = _beach()
    x0, x1 = max(0, 44 + ox), min(W, 374 + ox)
    if x1 <= x0:
        return
    xs = np.arange(x0, x1)
    xi = xs - ox
    k = on[xi]
    xs, xi = xs[k], xi[k]
    wv = (run * amp[xi]).astype(int)
    white = _h01(xi, f // 6) < 0.8
    y = sc.WL - wv
    c.a[y[white], xs[white]] = _c('white')
    c.a[y[~white], xs[~white]] = _c('mist')
    w = wet[xi]
    c.a[sc.WL + 1, xs[w]] = _c('#7fb8b4')
    m2 = (wv >= 2) & mist[xi]
    c.a[y[m2] + 1, xs[m2]] = _c('mist')


def _stack_foam(c, f, fxo):
    x0, x1 = max(0, fxo + 2), min(W, fxo + 96)
    if x1 <= x0:
        return
    xs = np.arange(x0, x1)
    yb = H - 3 + np.round(np.sin(xs * 0.35 + f * 0.13) * 1.2).astype(int)
    a = _h01(xs, 2) < 0.8
    wht = _h01(xs, f // 5) < 0.75
    ok = a & (yb < H)
    c.a[yb[ok & wht], xs[ok & wht]] = _c('white')
    c.a[yb[ok & ~wht], xs[ok & ~wht]] = _c('mist')
    b = (_h01(xs, 3) < 0.4) & (yb - 1 < H)
    c.a[yb[b] - 1, xs[b]] = _c('mist')


def _ship(c, t, cam):
    """The film's little crewless sloop on the far horizon (p 0.1), sailing east on a long loop."""
    hor = sc.HOR
    shx = int((t * SHIP_SPEED + SHIP_PHASE) % SHIP_SPAN) - 60 - int(cam * 0.1)
    if not -10 < shx < W + 10:
        return
    c.rect(shx, hor - 1, 7, 2, '#5a3a40')
    c.hline(shx + 1, shx + 5, hor + 1, '#7a5a5c')
    c.vline(shx + 3, hor - 8, hor - 2, '#5a3a40')
    c.tri((shx + 4, hor - 8), (shx + 4, hor - 3), (shx + 7, hor - 3), '#f6dcc0')
    c.tri((shx + 2, hor - 7), (shx + 2, hor - 3), (shx - 1, hor - 3), '#e8c4a8')
    c.px(shx + 3, hor - 9, 'hot_pink')


@lru_cache(maxsize=1)
def _sea_cols():
    nrow = H - sc.WL - 1
    d = np.arange(nrow) / nrow
    seacol = np.array([np.array(col(sc.mix('#18606a', '#0d3846', t)), np.float32) for t in d])
    k = (0.50 + 0.38 * d)[:, None, None]
    return seacol[:, None, :] * k * np.array([0.92, 0.95, 1.0], np.float32), (1 - k) * np.array(
        [0.92, 0.95, 1.0], np.float32), d


# ----------------------------------------------------------------------------- the skies alone
# Area.sky hooks (game/daylight.py repaints the open sky for the world clock's hour): the film's sky as
# each render draws it first, with everything that is not sky stamped out in magenta (never sky): the
# far islets, sea and volcano baked into the scene's sky canvases, and the opaque scenery layers whose
# dark basalt / hazy karsts sit too close to the sky colours for the colour test alone.
_MAGENTA = np.array((255, 0, 255), np.uint8)


def _sky_only(full, y1, draw):
    """`full` (a scene sky canvas with scenery baked in) reduced to its sky: rows >= y1 and every pixel
    that differs from `draw` (the same sky steps without the scenery) become magenta."""
    from engine import Canvas
    c = Canvas()
    draw(c)
    a = full.a.copy()
    bad = np.any(a != c.a, axis=2)
    bad[y1:] = True
    a[bad] = _MAGENTA
    return a


def _draw_sky_a(c, sun=True):                         # sc._sky_a's sky steps (no islets, no sea)
    c.gradient(0, 0, W, sc.HOR + 1, sc.SKY_A, soft=0.7)
    if not sun:
        return
    fx.glow(c, 40, 62, 70, 'coral', 0.30, rings=6)
    fx.glow(c, 40, 62, 34, 'sand', 0.35, rings=5)
    sc.disc(c, 40, 62, 9, '#f6c79a')
    sc.disc(c, 40, 62, 8, '#fbd9a8')
    sc.disc(c, 39, 61, 6, 'cream')
    for yy, x0, x1 in ((60, 18, 66), (63, 24, 60), (66, 14, 70)):
        c.hline(x0, x1, yy, '#e4a77e')


def _draw_sky_c(c, sun=True):                         # sc._sky_c's sky steps (no volcano, no smoke)
    c.gradient(0, 0, W, sc.LAG0 + 2, sc.SKY_C, soft=0.7)
    if not sun:
        return
    fx.glow(c, 34, 30, 80, 'peach', 0.16, rings=6)
    fx.glow(c, 34, 30, 30, 'cream', 0.25, rings=4)
    sc.disc(c, 34, 30, 7, '#fff0cf')
    sc.disc(c, 33, 29, 5, 'cream')


@lru_cache(maxsize=1)
def _sky_a_only():
    return _sky_only(sc._sky_a(), sc.HOR, _draw_sky_a)


# The painted low sun (and its glow) fades out as the world clock's sky takes over (sky_weight):
# a half-blended sky would otherwise show a ghost of it far from the clock's own sun.
@lru_cache(maxsize=2)
def _sunless(which):
    """(scene sky canvas array, its sky-only hook array) with the painted sun and glow removed."""
    from engine import Canvas
    full, only, draw = ((sc._sky_a(), _sky_a_only(), _draw_sky_a) if which == 'a' else
                        (sc._sky_c(), _sky_c_only(), _draw_sky_c))
    c1, c0 = Canvas(), Canvas()
    draw(c1)
    draw(c0, sun=False)
    sunpx = np.all(full.a == c1.a, axis=2) & np.any(c1.a != c0.a, axis=2)
    a = full.a.copy()
    a[sunpx] = c0.a[sunpx]
    o = only.copy()
    keep = ~np.all(o == _MAGENTA, axis=2)
    o[keep] = a[keep]
    return a, o


@lru_cache(maxsize=1)
def _sky_b_only():
    a = sc._sky_b().a.copy()
    a[70:] = _MAGENTA                                    # the gradient covers rows 0..69 only
    return a


# ----------------------------------------------------------------------------- the world clock's sky
# The film painted the islands at 15:38 under a red ash sky with a LOW sun on the left (sea x 40, grove
# x 34) and its glitter on the sea: at and near the painted hour game/daylight.py keeps the painted sky, so
# the afternoon showed a sunset sky and a sun on the wrong side, and the sea (the reflection of the sky)
# kept the red horizon at every hour. So every area starts from the clock's own sky instead (paint_sky's
# gradient, sun or moon on the arc, stars), put into the painted light (the relight's inverse, so the
# relight brings it back to the clock's colours): the sea reflects the hour's sky, the glitter runs under
# the clock's sun or moon, and the ash strata take the hour's sky colours.
@lru_cache(maxsize=64)
def _inv_lut(fq, tq):
    """Per channel: the painted value that daylight.relight (painted fq -> clock tq) turns into each value."""
    if abs(fq - tq) < 0.05:
        return np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    lut = daylight._tables(fq, tq, False)[0]
    v = np.arange(256)
    return np.stack([np.clip(np.searchsorted(lut[ch], v), 0, 255) for ch in range(3)]).astype(np.uint8)


def _unlit(a, hour):
    """The RGB array a (the clock's colours) put into the painted light (in place)."""
    a[...] = _inv_lut(daylight._q(PAINTED_H), daylight._q(hour))[_CH, a]


def _clock_sky(which, hour, cx, t):
    """(scene sky canvas array, its sky-only hook array) with the open sky the clock's at `hour`, in the
    painted light (cached: 3-minute steps; the stars' parallax and twinkle only at night)."""
    n = daylight.night(hour) > 0.25
    return _clock_sky_q(which, daylight.sky_q(hour), int(cx * 0.06) if n else 0, (t // 7) % 11 if n else 0)


@lru_cache(maxsize=12)
def _clock_sky_q(which, hq, sx, tw):
    if which == 'b':
        full, only = sc._sky_b().a, _sky_b_only()
    else:
        full, only = _sunless(which)
    sky = ~np.all(only == _MAGENTA, axis=2)
    a = full.copy()
    daylight.paint_sky(a, sky, hq, 1.0, (sx + 0.5) / 0.06, 0, -10, 130, tw * 7)
    if which == 'a':                                     # the far islets and the far sea (sc._sky_a's)
        _far_sea(a, hq)                                  # in the clock's horizon colours
    _unlit(a, hq)
    o = only.copy()
    o[sky] = a[sky]
    return a, o


def _far_sea(a, hq):
    """sc._sky_a's hazy far islets and far sea, redrawn in the clock's sky colours at hq (the film's: a
    sunset horizon, salmon islets and a salmon sea strip at every hour)."""
    from engine import Canvas
    strip = daylight._sky_strip(hq, -10, 130)
    hzc = tuple(int(v) for v in strip[min(129, 104 + 10)].mean(axis=0))      # the sky behind the islets
    low = tuple(int(v) for v in strip[min(129, sc.HOR + 10)].mean(axis=0))   # ...and at the horizon
    c = Canvas(arr=a)
    hz, hz2 = sc.mix(hzc, 'mauve', 0.35), sc.mix(hzc, 'mauve', 0.15)
    HOR = sc.HOR
    c.poly([(0, HOR + 1), (0, 104), (6, 101), (14, 99), (20, 102), (30, HOR + 1)], hz)
    c.poly([(8, HOR + 1), (16, 104), (24, 106), (34, HOR + 1)], hz2)
    c.poly([(286, HOR + 1), (298, 100), (304, 97), (309, 99), (322, 106), (330, HOR + 1)], hz)
    c.poly([(302, 98), (304, 97), (306, 98)], sc.mix(hzc, 'white', 0.25))
    c.gradient(0, HOR, W, H - HOR, [sc.mix(low, '#3f5560', 0.08), sc.mix(low, '#3f5560', 0.35),
                                    sc.mix(low, '#3f5560', 0.65), '#3f5560', '#1f4a55', '#16404d', '#123844'],
               soft=0.6)
    c.hline(0, W - 1, HOR, sc.mix(low, 'white', 0.3))


_STRATA_COLS = (('#4a2338', '#7a3a48'), ('#5c2a3c', '#934a4c'), ('#7a3a44', '#b8644f'), ('#a24e48', '#d68662'),
                ('#c06a54', '#eaa070'))     # (body, lit edge) of the film's five ash strata (sc._strata)


@lru_cache(maxsize=4)
def _strata_q(hq):
    """The film's ash strata in the clock's sky colours (painted light): pale wisps on a day sky, dark
    streaks with glowing edges on a sunset sky (as painted), faint on the night."""
    cols = daylight._sky_cols(hq)
    hz = np.array(cols[-1], np.float32)
    strip = daylight._sky_strip(hq, -10, 130)
    n = daylight.night(hq)                               # (at night only a shade off the sky: no black
    kd = 0.8 + 0.13 * n                                  # bars across the moon)
    out = []
    for (lay, y, w), (dk, lt) in zip(sc._strata(), _STRATA_COLS):
        rgb, m = lay
        sky = strip[min(strip.shape[0] - 1, y + 12)].astype(np.float32).mean(axis=0)
        day = min(1.0, max(0.0, (float(sky @ np.array([0.3, 0.55, 0.15], np.float32)) - 110) / 60))
        body = (sky * kd + hz * 0.08) * (1 - day) + (sky * 0.55 + np.array(col('#e6eaf0')) * 0.45) * day
        edge = (sky * 0.45 + hz * 0.55) * (1 - day) + (sky * 0.3 + 255 * 0.7) * day
        new = np.clip(np.array([body, edge]), 0, 255).astype(np.uint8)[None]
        _unlit(new, hq)
        a = rgb.copy()
        a[np.all(rgb == col(dk), axis=2)] = new[0, 0]
        a[np.all(rgb == col(lt), axis=2)] = new[0, 1]
        out.append(((a, m), y, w))
    return out


def _draw_strata(c, f, cam, hour, par=0.15):
    """sc.draw_strata with the strata in the clock's colours."""
    for i, (lay, y, w) in enumerate(_strata_q(round(hour * 4) / 4)):
        sp = 0.06 + 0.035 * i
        x = -int(f * sp + cam * par * (0.5 + 0.2 * i)) % w
        _comp(c, lay, x - w, y)
        _comp(c, lay, x, y)


# ----------------------------------------------------------------------------- the world clock's lights
# The relight keeps small warm spots as painted at night (auto_glow): here that lit the gold studs of the
# moon bridge, flowers in the grass and the gold coins of the hoards like lamps, while the lava (long thin
# rivulets, the lake's dither as one big blob) and half the lanterns went dull. So the region keeps its
# own lights (auto_glow False): the renders mark the pixels of the real lights in a mask (lava, the crater,
# embers, the lava's reflections, the moon's and sun's glitter on the sea, braziers, and the lanterns and
# windows while lit) and after_light gives them back their painted colour. The windows and lanterns follow
# the clock (_lamp_k, each at its own point, _lv): dark by day with no halos, lit through the dusk.
LAMP_V = 0.9


def _lamp_k(hour):
    """0 (day) .. 1 (night): how far the lamps are on (just starting ~18:50, all on by ~19:35)."""
    return min(1.0, max(0.0, (daylight.night(hour) - 0.05) / 0.55))


def _lv(k, *seed):
    """Is one lamp (threshold hashed from seed) on at lamp level k?"""
    return k > hash01(*seed, 913) * LAMP_V


def _mark(glow, c, box, fn, *a, **kw):
    """fn(c, *a, **kw), adding the pixels it changed inside box (x0, y0, x1, y1) to the light mask glow."""
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if glow is None or x1 <= x0 or y1 <= y0:
        fn(c, *a, **kw)
        return
    before = c.a[y0:y1, x0:x1].copy()
    fn(c, *a, **kw)
    glow[y0:y1, x0:x1] |= np.any(c.a[y0:y1, x0:x1] != before, axis=2)


def _mark_px(glow, xs, ys):
    """Add the pixels (xs, ys) (arrays or ints) that lie on screen to the light mask."""
    if glow is None:
        return
    xs, ys = np.broadcast_arrays(np.atleast_1d(np.asarray(xs, int)), np.atleast_1d(np.asarray(ys, int)))
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    glow[ys[ok], xs[ok]] = True


def _bright(a):
    """Light-coloured pixels: warm (lava, flame, lantern paper) or bright (glitter, flashes): the noted
    lights' pixels that are still lights at the end of a render (not covered by nearer scenery)."""
    p = a.astype(np.int32)                               # (int16 sums would wrap: white would fail)
    lum = (p[..., 0] * 77 + p[..., 1] * 150 + p[..., 2] * 29) >> 8
    return ((p[..., 0] > 120) & (p[..., 0] - p[..., 2] > 50)) | (lum > 190)


_NOGLOW = type('NoGlow', (), {'glow': staticmethod(_nop)})
_lantern_dark = rebind(sc.draw_lantern, fx=_NOGLOW)       # the film's paper lantern without its halo


def _lantern(c, x, y, f, seed, size=1, on=True, glow=None):
    """A paper lantern: lit (its painted colours, marked as a light; no dithered halo) or unlit (dark
    paper where its flame shows)."""
    if on:
        _mark(glow, c, (x - 4, y, x + 5, y + 11), _lantern_dark, x, y, f, seed, size=size)
        return
    _lantern_dark(c, x, y, f, seed, size=size)
    x += int(round(math.sin(f * 2 * math.pi / 60 + seed) * 0.8))
    if size == 1:
        c.px(x, y + 2, '#a8402e')
        c.px(x, y + 3, '#8a3226')
    else:
        c.rect(x - 1, y + 3, 3, 4, '#b0442e')
        c.px(x, y + 4, '#8a3226')


# the pavilion's moon windows, upper window and door are painted lit: by day they are dark, at dusk each
# goes on at its own point of the ramp (lit: the painted colours)
_PAV_LIT = {col('#ffd88a'): '#6a4c3e', col('#fff0b8'): '#7e5e4a', col('#ffcf7a'): '#7a4a34'}


@lru_cache(maxsize=1)
def _pavilion_lights():
    """(unlit pavilion layer, [(ys, xs, painted colours) per light: left window, right window, door,
    upper window])."""
    rgb, m = sc._pavilion()
    out = rgb.copy()
    lit = np.zeros(m.shape, bool)
    for cl, dk in _PAV_LIT.items():
        k = m & np.all(rgb == cl, axis=2)
        out[k] = col(dk)
        lit |= k
    cx = rgb.shape[1] // 2
    ys, xs = np.nonzero(lit)
    groups = [(xs < cx - 6) & (ys > 30), (xs > cx + 6) & (ys > 30), (abs(xs - cx) <= 6) & (ys > 30), ys <= 30]
    return (out, m), [(ys[g], xs[g], rgb[ys[g], xs[g]]) for g in groups]


@lru_cache(maxsize=1)
def _sky_c_only():
    return _sky_only(sc._sky_c(), sc.LAG0 + 2, _draw_sky_c)


def _stamp(a, layers):
    """Magenta wherever the (rgb, mask) layers are opaque, each placed as _comp places it at (x, y)."""
    hit = np.zeros((H, W), bool)
    for (_rgb, m), x, y in layers:
        x, y = int(x), int(y)
        h, w = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            hit[y0:y1, x0:x1] |= m[y0 - y:y1 - y, x0 - x:x1 - x]
    np.copyto(a, _MAGENTA, where=hit[..., None])


def render_sea(f, cam, t, ev, hour, glow=None):
    """Shot A at camera `cam` (the film's render_a, with the ship on a loop and no balloon/tags) under
    the clock's sky at `hour`; glow: the light mask to mark the real lights in (or None)."""
    from engine import Canvas
    c = Canvas(arr=_clock_sky('a', hour, cam, t)[0].copy())
    _draw_strata(c, f, cam, hour)
    ox = -int(round(cam * sc.A_PAR))
    _ship(c, t, cam)
    sc.draw_plume(c, f, ox)
    lay, riv, palms = sc._island()
    _comp(c, lay, ox, 0)
    # crater glow flicker + lava pulses down the rivulets
    rx, ry, rs, rj = riv
    ph = (rs * 3.2 - f * 0.045 + rj * 0.37) % 1.0
    cols_ = np.where(ph[:, None] < 0.18, _c('lava_hi'), np.where(ph[:, None] < 0.55, _c('lava'), _c('#e0401a')))
    sx = rx + ox
    ok = (sx >= 0) & (sx < W)
    c.a[ry[ok], sx[ok]] = cols_[ok]
    _mark_px(glow, sx[ok], ry[ok])                       # (the lava glows at every hour)
    a = ev['crater']
    if 0 <= a < 64:                                      # crater burst: a bright front races down
        front = a / 26
        hot = ok & (rs <= front) & (rs >= front - 0.45)
        c.a[ry[hot], sx[hot]] = _c('butter') if a < 40 else _c('lava_hi')
    vx = sc.VX + ox
    fl = (f // 3) % 4
    c.hline(vx - 5 + fl % 2, vx + 4 - fl // 2, sc.VT + 1, 'butter' if fl == 0 else 'lava_hi')
    _mark_px(glow, np.arange(vx - 5 + fl % 2, vx + 5 - fl // 2), sc.VT + 1)
    fx.glow(c, vx, sc.VT - 2, 22, 'lava', 0.22 + 0.06 * math.sin(f * 0.3), rings=4)
    for k, (px_, pb, ph_, ln) in enumerate(palms):
        if -12 < px_ + ox < W + 12:
            sc.draw_palm(c, px_ + ox, pb, ph_, ln, f, k)
    for i in range(26):                                  # embers rising from the crater
        per = 70 + int(hash01(i, 3) * 50)
        tt = ((f + hash01(i, 1) * per) % per) / per
        x = vx + (hash01(i, 2) - 0.5) * 22 + tt * (30 + hash01(i, 4) * 40) + math.sin(tt * 8 + i) * 2
        y = sc.VT - 2 - tt * (40 + hash01(i, 5) * 50)
        if hash01(i, f // 4) < 0.85 - tt * 0.5:
            sc.ember(c, int(x), int(y), f, i)
            _mark_px(glow, int(x), int(y))
    for k, (vx_, vy_) in enumerate(((176, 70), (236, 78), (160, 88))):     # fumaroles on the flanks
        fx.steam(c, vx_ + ox, vy_, f, color='#c8b4b0', n=3, height=10, spread=2, seed=k, speed=0.25)
    # ---- sea: reflection of everything above the waterline, rippled + tinted
    WL = sc.WL
    nrow = H - WL - 1
    src_rows = WL - 1 - np.arange(nrow)
    rows = WL + 1 + np.arange(nrow)
    sea_add, sea_mul, d = _sea_cols()
    shift = np.round(np.sin(f * 0.16 + np.arange(nrow) * 0.9) * (1 + d * 2)).astype(int)
    xi = np.clip(np.arange(W)[None, :] + shift[:, None], 0, W - 1)
    refl = c.a[src_rows[:, None], xi].astype(np.float32)
    c.a[rows] = np.clip(refl * sea_mul + sea_add, 0, 255).astype(np.uint8)
    # wave dashes (far rows + near rows), parallax by depth
    HOR = sc.HOR
    tex = sc._wave_tex()
    yrows = np.arange(HOR + 1, H)
    dd = (yrows - HOR) / (H - HOR)
    offs = (f * (0.15 + dd * 0.5) + cam * (0.4 + dd * 1.0)).astype(int)
    xs = (np.arange(W)[None, :] + offs[:, None]) % tex.shape[1]
    tv = tex[(yrows - HOR)[:, None], xs]
    reg = c.a[HOR + 1:H]
    reg[tv == 1] = np.clip(reg[tv == 1].astype(np.int16) + 34, 0, 255).astype(np.uint8)
    reg[tv == 2] = (reg[tv == 2] * 0.75).astype(np.uint8)
    kind, bx, by = daylight._body_pos(hour)              # glitter path under the clock's sun or moon
    gk = 0.45 if kind == 'sun' else 0.35 * daylight.night(hour)
    gc = (('white', 'cream') if by < 40 else ('butter', '#f6c79a')) if kind == 'sun' else ('#e8ecff', '#b8c0e0')
    for i in range(40):                                  # (screen-fixed, as the sky)
        yq = HOR + 2 + int(hash01(i, 11) * (H - HOR - 4))
        spread = 3 + (yq - HOR) * 0.18
        xq = bx + int((hash01(i, 12) - 0.5) * 2 * spread)
        if hash01(i, f // 3) < gk:
            L = 1 + int((yq - HOR) / 20)
            c.hline(xq, xq + L, yq, gc[0] if hash01(i, 13) < 0.5 else gc[1])
            _mark_px(glow, np.arange(xq, xq + L + 1), yq)
    for j, xr in enumerate((180, 200, 222)):             # lava glow reflections below the rivulets
        for q in range(4):
            yq = WL + 3 + q * 4 + j
            if hash01(j, q, f // 4) < 0.6:
                x0 = xr + ox - 1 + int(hash01(j, q, f // 6) * 3)
                c.hline(x0, xr + ox + 1, yq, '#c0502e')
                _mark_px(glow, np.arange(x0, xr + ox + 2), yq)
    _beach_waves(c, f, ox)
    if 0 <= a < 96:
        _mark(glow, c, (0, 0, W, H), _crater_burst, a, sc.VX + ox)
    # foreground basalt sea stack with a leaning palm (parallax 1.25)
    fxo = 250 - int(round(cam * 1.25))
    if -100 < fxo < W:
        fy = H - 96 + 2
        _comp(c, sc._fg_a(), fxo, fy)
        sc.draw_big_palm(c, fxo + 40, fy + 30, 26, -7, f, 9)
        _stack_foam(c, f, fxo)
        dt = (f + 30) % 72
        if dt < 26:                                      # spray on the wave beat (once per bar)
            for q in range(26):
                a_ = -math.pi / 2 + (hash01(q, 1) - 0.5) * 1.8
                v = 0.9 + hash01(q, 2) * 0.9
                sx_ = fxo + 6 + hash01(q, 3) * 86
                x = sx_ + math.cos(a_) * v * dt
                y = H - 3 + math.sin(a_) * v * dt * 1.4 + 0.07 * dt * dt
                if y < H - 2 and hash01(q, dt // 3) < 1 - dt / 30:
                    c.px(int(x), int(y), 'white' if q % 3 else 'mist')
    if 0 <= ev['surf'] < 84:
        _surf(c, ev['surf'], fxo, f)
    _grade(c, (1.0, 0.96, 0.94), (6, 0, 0))            # warm key light: ash-red afternoon
    return c


# ============================================================================= area 1: the volcanic heart
def _swell(e):
    """The film's pool swell (0..1) at eruption-clock frame e (film frame numbers)."""
    if e < sc.ERUPT:
        return seg(e, sc.ERUPT - 20, sc.ERUPT) ** 2
    return max(0.0, 1 - seg(e, sc.ERUPT, sc.ERUPT + 20))


def _flash(e):
    """The eruption's warm flash at eruption frame e (0 = the fountain bursts): instead of the film's
    one-frame white-out, a smoothstep ramp over FLASH_IN steps up to FLASH_K of the film's strength,
    then a smooth ease out over FLASH_OUT steps (no frame-to-frame jump of more than ~0.02 in mean
    brightness; comfort for players sensitive to flashes)."""
    if e < 0 or e >= FLASH_IN + FLASH_OUT:
        return 0.0
    if e < FLASH_IN:
        u = (e + 1) / (FLASH_IN + 1)
    else:
        u = 1 - (e - FLASH_IN) / FLASH_OUT
    return FLASH_K * u * u * (3 - 2 * u)


def _pool_glow(c, e, rox, sy=0):
    """The red glow over the bursting pool (eruption frame e), swelling in with the flash."""
    if 0 <= e < 30:
        fx.glow(c, sc.POOL[0] + rox, sc.POOL[1] + sy - 40, 90, 'lava',
                0.35 * min(1.0, (e + 1) / FLASH_IN) * (1 - e / 30), rings=5)


def _heart_grade(c, ef):
    """The caldera's light over everything drawn: the warm grade (redder as the pool swells) and the
    eruption's flash. DragonLands.sky_b puts the same light on its sky hook, so game/daylight.py's sky
    mask keeps finding the open sky through the eruption (no snap from the clock's sky to the painted
    one and back)."""
    _grade(c, (1.0, 0.95, 0.92), (8 + int(18 * _swell(ef)), 0, 0))
    k = _flash(ef - sc.ERUPT)                            # eruption glow: swells in, eases out
    if k > 0.004:                                        # (fx.grade's pixels, through the lookup table)
        q = round(k, 3)
        _grade(c, (1 + 0.5 * q, 1 + 0.35 * q, 1 + 0.1 * q), (150 * q, 110 * q, 50 * q))


def _lava_px(a, y0, y1):
    """The lava pixels (hot orange-red) of rows y0..y1 of the RGB array a, as a full-frame mask."""
    m = np.zeros(a.shape[:2], bool)
    p = a[y0:y1].astype(np.int16)
    m[y0:y1] = (p[..., 0] > 110) & (p[..., 0] - p[..., 2] > 60) & (p[..., 0] > p[..., 1] + 30)
    return m


def render_heart(f, cam, ef, ev, t, hour, glow=None):
    """Shot B at camera `cam`: ambience on the free scene frame f, the eruption on its clock ef, under the
    clock's sky at `hour`; glow: the light mask to mark the real lights in (or None)."""
    from engine import Canvas
    sx, sy = 0, 0                                        # no world shake (comfort; the film's shake_b)
    e = ef - sc.ERUPT
    swell = _swell(ef)
    c = Canvas(arr=_clock_sky('b', hour, cam, t)[0].copy())
    pox = -int(cam * 0.1) + sx
    sc.draw_plume(c, f, pox, x0=150, y0=60, n=18, period=240, rise=70, drift=60, rmax=16, seed=3, hot=True)
    sc.draw_plume(c, f - sc.ERUPT, pox, x0=sc.POOL[0] - 30, y0=50, n=10, period=120, rise=60, drift=40,
                  rmax=18, seed=4, hot=True)
    wl, veins, rim, LF = sc._wall_b()
    wox = -int(cam * 0.3) - 20 + sx
    _comp(c, wl, wox, sy)
    for j, pts in enumerate(veins):                      # vein pulses
        n = len(pts)
        for i, (vx, vy) in enumerate(pts):
            ph = (i / n * 2 - f * 0.05 + j * 0.3) % 1.0
            if ph < 0.35:
                c.px(vx + wox, vy + sy, 'lava' if ph > 0.12 else 'lava_hi')
                _mark_px(glow, vx + wox, vy + sy)
    _mark(glow, c, (LF + wox - 8, 0, LF + wox + 9, sc.RIV0 + sy + 4), sc.draw_lavafall, f, LF + wox,
          rim[LF] + 6 + sy, sc.RIV0 + 2 + sy)
    fx.glow(c, LF + wox, sc.RIV0 + sy, 26, 'lava', 0.25, rings=4)
    ml, piles = sc._mid_b()
    mox = -int(cam * 0.6) - 6 + sx
    _comp(c, ml, mox, sy)
    for bx in (56 - 22, 56 + 22):
        _mark(glow, c, (bx + mox - 3, 60 + sy, bx + mox + 4, 71 + sy), sc.draw_brazier_flame, f, bx + mox, 69 + sy, bx)
    hk = 1 - _lamp_k(hour)                               # the throne fires' halo: by day only (at night a
    if hk > 0.05:                                        # darkened halo is a brown blotch)
        fx.glow(c, 56 + mox, 68 + sy, 30, 'amber', 0.12 * hk, rings=3)
    for i in range(14):                                  # gem sparkles
        (px_, py_, rw, rh) = piles[i % len(piles)]
        gx = int(px_ + (hash01(i, 21) - 0.5) * 1.6 * rw) + mox
        gy = int(py_ - hash01(i, 22) * rh * 0.8) + sy
        fx.sparkle(c, gx, gy, f, ('white', 'butter', 'crystal_lt', 'lilac')[i % 4], period=28 + i % 5 * 4,
                   phase=hash01(i, 23), size=2 if i % 3 else 3)
    # geyser on the left step: bursts on beats 1 and 3 into a fan of hot spray + a steam cloud
    gx, gy = 149 + mox, 116 + sy
    bp = f % 36
    if bp < 24:
        jh = int(24 * sc.ease_out(min(1, bp / 4))) if bp < 8 else int(24 * max(0, 1 - (bp - 8) / 16) ** 1.4)
        for yy in range(gy - jh, gy):
            tt = (gy - yy) / max(1, jh)
            ww = 1 + int(tt * tt * 6)
            wob = int(round(math.sin(yy * 0.6 + f * 0.9) * 0.6))
            for x in range(gx - ww + wob, gx + ww + wob + 1):
                if tt < 0.6 or hash01(x, yy, f // 2) < 1.3 - tt:
                    c.px(x, yy, '#f4c6a4' if (yy + x + f) % 3 else '#e0a184')
            c.vline(gx + wob, yy, yy, 'cream')
        for q in range(18):
            tt = ((bp + q * 1.3) % 14) / 14
            side = 1 if q % 2 else -1
            sx_ = gx + side * (2 + tt * (6 + q % 5 * 2))
            sy_ = gy - jh + tt * tt * (jh + 4) - math.sin(tt * math.pi) * 4
            if sy_ < gy:
                c.px(int(sx_), int(sy_), 'cream' if q % 3 else '#e0a184')
    sc.steam_puffs(c, f - 6, gx + 6, gy - 30, period=36, n=3, height=34, rmax=10, seed=11, drift=16, tone=6)
    rox = -cam + sx
    sc.draw_lava_river(c, f, rox, sy, swell)
    if ef < sc.ERUPT and swell > 0.05:                   # pool swell dome before the eruption
        dh = int(1 + swell * 5)
        c.ellipse(sc.POOL[0] + rox, sc.POOL[1] + sy - dh // 2, 8 + dh, dh, 'lava_hi')
        c.ellipse(sc.POOL[0] + rox, sc.POOL[1] + sy - dh // 2 - 1, 4 + dh, max(1, dh - 1), 'butter')
    y0s, y1s = 96, sc.RIV0                               # heat shimmer over the lava
    rows = np.arange(y0s, y1s)
    shift = np.round(np.sin(rows * 0.7 + f * 0.35) * 0.6 * (rows - y0s) / (y1s - y0s) * 1.6).astype(int)
    xi = np.clip(np.arange(W)[None, :] - shift[:, None], 0, W - 1)
    c.a[y0s:y1s] = c.a[rows[:, None], xi]
    fx.light_rect(c, 0, 92, W, sc.RIV0 - 92, 'lava', 0.10 + 0.08 * swell, fade='up')
    if e >= 0:                                           # the eruption
        _mark(glow, c, (0, 0, W, H), sc.draw_fountain, ef, sc.POOL[0] + rox, sc.POOL[1] + sy)
        _pool_glow(c, e, rox, sy)
    if 0 <= ev['gems'] < 72:
        _gem_shower(c, ev['gems'], mox, sy, piles, f)
    fl, vents, stop = sc._fg_b()
    fox = -int(cam * 1.3) - 8 + sx
    _comp(c, fl, fox, sy)
    for k, (vx, _) in enumerate(vents):                  # steam vents puffing on the beats
        top = stop[vx] + sy
        burst = ((f - k * 18) % 36) < 12
        fx.steam(c, vx + fox, top - 1, f + k * 5, color='#f4d6c0', n=7 if burst else 3,
                 height=26 if burst else 14, spread=3, seed=k, speed=0.6 if burst else 0.3)
        fx.steam(c, vx + fox + 1, top - 1, f + k * 7 + 3, color='#e0a184', n=4 if burst else 2,
                 height=18, spread=2, seed=k + 9, speed=0.45)
    for i in range(34):                                  # embers rising everywhere
        per = 80 + int(hash01(i, 3) * 60)
        tt = ((f + hash01(i, 1) * per) % per) / per
        x = hash01(i, 2) * W + math.sin(tt * 6 + i) * 4 - cam * 0.6
        y = sc.RIV0 - tt * 130
        if hash01(i, f // 4) < 0.85:
            sc.ember(c, int(x) % W, int(y), f, i)
            _mark_px(glow, int(x) % W, int(y))
    if 0 <= e < 72:                                      # ember shower raining down after the burst
        for i in range(34, 90):
            dt = e - int(hash01(i, 9) * 10)
            if dt < 0 or dt > 60:
                continue
            x = sc.POOL[0] + rox + (hash01(i, 2) - 0.5) * 300 * min(1, dt / 12)
            y = 10 + dt * (1.2 + hash01(i, 4)) + math.sin(dt * 0.3 + i) * 2 - 40 * max(0, 1 - dt / 8)
            if hash01(i, f // 4) < 0.85:
                sc.ember(c, int(x) % W, int(y), f, i)
                _mark_px(glow, int(x) % W, int(y))
    _heart_grade(c, ef)
    if glow is not None:                                 # the lava lake and the pool, where still in view
        glow |= _lava_px(c.a, max(0, sc.RIV0 + sy - 3), min(H, sc.RIV1 + sy + 2))
    return c


# ============================================================================= area 2: Kirin Grove
FG_SPLIT = 190          # fg boulders: left group (x < 190, framing the start) and right group (the end)
FG_R_SHIFT = int(round(1.3 * (GROVE_CAM - 30)))     # the right group frames the end as at film cam 30


@lru_cache(maxsize=1)
def _fg_parts():
    (a, m), lamp = sc._fg_c()
    ml, mr = m.copy(), m.copy()
    ml[:, FG_SPLIT:] = False
    mr[:, :FG_SPLIT] = False
    return (a, ml), (a, mr), lamp


def render_grove(f, cam, t, ev, lantern_ts, hour, glow=None):
    """Shot C at camera `cam` (the film's render_c without its balloon/tag; drifting mist and lagoon
    glints anchored to their layers; the foreground framing split so it frames both ends) under the
    clock's sky at `hour`, its windows and lanterns lit by the clock; glow: the light mask (or None)."""
    from engine import Canvas
    lk = _lamp_k(hour)
    c = Canvas(arr=_clock_sky('c', hour, cam, t)[0].copy())
    for i, (cy_, w_, sp) in enumerate(((22, 44, 0.08), (40, 30, 0.12), (12, 36, 0.05))):
        x = int(150 + i * 70 - f * sp - cam * 0.05) % 380 - 40
        fx.cloud(c, x, cy_, w_, color='#fff6e6', shade_color='#e8d6c8', seed=i + 3)
    _comp(c, sc._far_karsts(), -int(cam * 0.12) - 10, 0)
    lag, isl = sc._lagoon()
    lox = -int(cam * 0.2) - 10
    _comp(c, lag, lox, 0)
    for i in range(18):                                  # lagoon glints (on the lagoon layer)
        gx = int(hash01(i, 31) * 360 + lox) % 360 - 20
        gy = sc.LAG0 + 2 + int(hash01(i, 32) * (sc.LAG1 - sc.LAG0 - 3))
        if hash01(i, f // 5) < 0.4:
            c.hline(gx, gx + 1 + i % 2, gy, 'white')
    mox = -int(cam * 0.35) - 8
    _comp(c, sc._mid_karsts(), mox, 0)
    sc.draw_ribbon_fall(c, f, 40 + mox, 52, 118)
    for i in range(7):                                   # mist drifting along the karst feet
        x = int((i * 64 + f * 0.25 + mox) % 450) - 70
        y = 104 + (i % 3) * 4
        L = 50 + (i % 3) * 18
        c.blend(x, y, L, 3, '#eef6f0', 0.32)
        c.blend(x + 8, y - 1, L - 16, 1, '#eef6f0', 0.22)
        c.blend(x + 6, y + 3, L - 12, 1, '#eef6f0', 0.18)
    bk, top, sm, pl = sc._bank()
    box = -int(cam * 0.6) - 10
    _draw_bamboo(c, f, box, 0, 501, 26, 170, 300, 46, 124, sc.GW, phase=0.0, amp=2,
                   ramp=('#3f7a52', '#5f9a62', '#8cbc78', '#bfe0a0'))
    _draw_bamboo(c, f, box, 0, 502, 18, 0, 92, 50, 126, sc.GW, phase=1.3, amp=2,
                   ramp=('#3f7a52', '#5f9a62', '#8cbc78', '#bfe0a0'))
    _comp(c, bk, box, 0)
    pvx, pvy = 196 + box, sc.BANK - 62 + 2
    pav, lights = _pavilion_lights()
    _comp(c, pav, pvx, pvy)                              # dark windows; each lit by the clock
    for j, (ys, xs, rgb) in enumerate(lights):
        if _lv(lk, 30, j):
            X, Y = xs + pvx, ys + pvy
            ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
            c.a[Y[ok], X[ok]] = rgb[ok]
            _mark_px(glow, X[ok], Y[ok])
            if j < 2 and (f // 4) % 3 == 0:              # the lit moon windows flicker warm
                c.px(pvx + 38 + (16 if j else -16), pvy + 43, 'butter')
    for (lx, ly, sd) in ((pvx + 1, pvy + 30, 1), (pvx + 75, pvy + 30, 2), (pvx + 15, pvy + 12, 3),
                         (pvx + 61, pvy + 12, 4)):
        _lantern(c, lx, ly, f, sd, 1, _lv(lk, 31, sd), glow)
    sc.draw_stream(c, f, box, 0)
    sc.draw_bridge(c, box, 0)
    sc.draw_pond(c, f, box, pl)
    ax, ay = 148 + box, 122                              # lantern string bridge post -> pavilion eave
    bx_, by_ = pvx + 2, pvy + 30
    for k in range(0, 41):
        tt = k / 40
        c.px(int(lerp(ax, bx_, tt)), int(lerp(ay, by_, tt) + 6 * math.sin(tt * math.pi)), '#5a3a2e')
    for k in range(1, 5):
        tt = k / 5
        _lantern(c, int(lerp(ax, bx_, tt)), int(lerp(ay, by_, tt) + 6 * math.sin(tt * math.pi)) + 1, f, k, 1,
                 _lv(lk, 32, k), glow)
    c.vline(ax, ay, 140, '#7a4a2e')
    c.px(ax, ay - 1, 'gold')
    _lantern(c, ax + 2, ay + 1, f, 7, 2, _lv(lk, 33), glow)
    _lanterns(c, t, f, ev['lanterns'], lantern_ts, pvx, pvy, ax, ay, bx_, by_, lk, glow)
    for i in range(10):                                  # sparkles on the stream
        y = sc.STREAM_TOP + 10 + int(hash01(i, 41) * (H - sc.STREAM_TOP - 12))
        x = int(sc.stream_cx(y) + (hash01(i, 42) - 0.5) * sc.stream_hw(y) * 1.4) + box
        fx.sparkle(c, x, y, f, 'white', period=20 + i % 4 * 6, phase=hash01(i, 43), size=1)
    # foreground framing (p 1.3): boulders + stone lantern + bamboo on the left at the start, the right
    # group shifted so that it frames the far end of the pan
    fox = -int(cam * 1.3)
    fgl, fgr, (lx_, ly_) = _fg_parts()
    if fox - 10 + FG_SPLIT > 0:
        _comp(c, fgl, fox - 10, 0)
        if _lv(lk, 34):                                  # the stone lantern: lit by the clock, no halo
            c.rect(lx_ + fox - 10 - 1, ly_, 3, 3, 'butter' if (f // 5) % 4 else 'lava_hi')
            _mark_px(glow, np.arange(lx_ + fox - 11, lx_ + fox - 8).repeat(3), np.tile(np.arange(ly_, ly_ + 3), 3))
        else:
            c.rect(lx_ + fox - 10 - 1, ly_, 3, 3, '#4a4038')
        _draw_bamboo(c, f, fox, 0, 601, 6, -8, 22, -40, H, 380, phase=0.7, amp=3, canopy=False, thick=3)
    fxr = fox + FG_R_SHIFT
    if fxr + FG_SPLIT - 10 < W:
        _comp(c, fgr, fxr - 10, 0)
        _draw_bamboo(c, f, fxr, 0, 602, 6, 326, 356, -40, H, 380, phase=2.1, amp=3, canopy=False, thick=3)
    for i in range(14):                                  # falling leaves
        per = 120 + int(hash01(i, 51) * 80)
        tt = ((f + hash01(i, 52) * per) % per) / per
        x = int(hash01(i, 53) * W + tt * 40 + math.sin(tt * 12 + i) * 5)
        y = int(-4 + tt * (H + 8))
        c.px(x % W, y, 'leaf' if (f // 4 + i) % 2 else 'mint')
        c.px((x + 1) % W, y + ((f // 4 + i) % 2), 'green')
    _grade(c, (1.0, 0.99, 0.95), (6, 3, 0))
    return c


# ============================================================================= the region
# ============================================================================= SPACE events
VOLCANO_ICON = Sprite.from_ascii('''
    ..Y.y.Y..
    ...yry...
    ...oro...
    ..oNrNo..
    ..NNNrN..
    .oNNNrNo.
    .NNNNNrN.
    oNNNNNNNo
    ooooooooo
''', {'o': ui.INK, 'y': 'gold', 'Y': 'butter', 'r': 'lava', 'N': '#5a4048'})


def _bombs(c, a, x, y, n, seed, spread, vmin, vmax, g, life, cols, trail=None, stop=None, stagger=6,
           big=0.34):
    """Ballistic particles (lava bombs, gems, spray) flung up from (x, y), a steps after the start:
    each launches within `stagger` steps, flies an arc and stops at `stop(x, y)` or after `life`."""
    for i in range(n):
        dt = a - int(hash01(seed, i, 1) * stagger)
        if dt < 0 or dt > life:
            continue
        ang = -math.pi / 2 + (hash01(seed, i, 2) - 0.5) * spread
        v = vmin + (vmax - vmin) * hash01(seed, i, 3)
        vx, vy = math.cos(ang) * v, math.sin(ang) * v
        px, py = x + vx * dt, y + vy * dt + g * dt * dt
        if py > H + 2 or (stop is not None and dt > 3 and stop(px, py)):
            continue
        if trail is not None:
            c.line(int(px), int(py), int(px - vx * 1.5), int(py - (vy + 2 * g * dt) * 1.5), trail)
        s = 2 if (hash01(seed, i, 4) < big and dt < life * 0.7) else 1
        c.rect(int(px), int(py), s, s, cols[i % len(cols)])


def _crater_burst(c, a, vx):
    """THE VOLCANO clears its throat: a flash and a shock ring at the rim, a short lava fountain, lava
    bombs arcing out and landing glowing on the flanks, a billow of underlit ash."""
    vt = sc.VT
    if a < 34:
        k = 1 - a / 34
        fx.glow(c, vx, vt - 2, 34 + a, 'lava_hi', 0.6 * k, rings=5)
        if a < 6:
            fx.glow(c, vx, vt - 2, 16, 'white', 0.8 * (1 - a / 6), rings=3)
    if a < 20:                                           # shock ring racing out over the rim
        rx, ry = 8 + a * 2.6, 2 + a * 0.55
        n = int(rx * 2.2) + 8
        for q in range(n):
            ang = q / n * 2 * math.pi
            if (q + a) % 3 == 0:
                continue
            c.px(int(vx + math.cos(ang) * rx), int(vt - 3 + math.sin(ang) * ry),
                 'cream' if a < 10 else '#f6c79a')
    for q in range(7):                                   # the ash billow: underlit, then cooling grey
        age = a - 4 - q * 3
        if not 0 <= age < 90:
            continue
        r = 4 + min(age, 40) * 0.24 + (q % 3)
        x = vx + (q - 3) * 4 + age * (0.20 + 0.10 * (q % 3))
        y = vt - 6 - min(age, 30) * 0.7 - age * 0.12 - q
        tone = 0 if age < 14 else (1 if age < 44 else 2)
        lvl = 1.0 if age < 64 else 1 - (age - 64) / 26 * 0.85
        sc.draw_puff(c, int(x), int(y), int(r), tone, seed=q, level=lvl)
    if a < 30:                                           # a short lava fountain out of the crater
        hc = 24 * math.sin(min(1.0, a / 6) * math.pi / 2) * (1 if a < 12 else max(0.0, 1 - (a - 12) / 18))
        top = int(vt - hc)
        for yy in range(max(0, top), vt + 1):
            tt = (vt - yy) / max(1.0, hc)
            ww = 2 + int((1 - tt) * 2) + (1 if ((yy + a * 3) // 3) % 3 == 0 else 0)
            wob = int(round(math.sin(yy * 0.4 + a * 0.9) * tt))
            c.hline(vx + wob - ww, vx + wob + ww, yy, 'lava')
            c.hline(vx + wob - ww + 1, vx + wob + ww - 1, yy, 'lava_hi' if (yy + a) % 4 else 'butter')
        if hc > 3:
            c.ellipse(vx, top + 1, 4, 2, 'lava_hi')
            c.ellipse(vx, top, 2, 1, 'butter')

    def on_cone(x, y):                                   # a bomb stops where it lands on the flanks
        return (y > vt + 2 and abs(x - vx) < float(sc.cone_r(y)) - 1) or y > sc.WL - 4

    _bombs(c, a - 2, vx, vt - 3, 46, 3, 2.2, 1.3, 2.7, 0.08, 70, ('butter', 'lava_hi', 'lava_hi', 'amber'),
           trail='#e0401a', stop=on_cone, big=0.6)
    for i in range(16):                                  # where the bombs landed: glowing, then cooling
        sx_ = vx + int((hash01(i, 77) - 0.5) * 2 * 60)
        sy_ = int(vt + 8 + hash01(i, 78) * 60)
        d = a - 16 - int(hash01(i, 79) * 20)
        if 0 <= d < 60 and abs(sx_ - vx) < float(sc.cone_r(sy_)) - 3:
            c.rect(sx_, sy_, 2, 1, 'butter' if d < 8 else ('lava_hi' if d < 30 else 'lava'))
            if d < 30:
                fx.glow(c, sx_, sy_, 4, 'lava', 0.3 * (1 - d / 30), rings=2)


@lru_cache(maxsize=1)
def _rainbow_px():
    """Pixels of a small rainbow arc (dx, dy, band) around (0, 0), bands outer -> inner, 2 px each."""
    out = []
    for band in range(4):
        for w_ in range(2):
            r = 54 - band * 2 - w_
            for k in range(0, 361):
                ang = math.radians(180 + k / 2)
                out.append((int(round(math.cos(ang) * r)), int(round(math.sin(ang) * r * 0.85)), band))
    arr = np.array(sorted(set(out)), int)
    return arr[:, 0], arr[:, 1], arr[:, 2]


_RAINBOW = np.array([col(x) for x in ('coral', 'butter', 'mint', 'baby_blue')], np.float32)


@lru_cache(maxsize=1)
def _sheet_cols():
    """Spray sheet thrown up the sea stack (stack px -8..105): per-column jet heights (in jets of 2..4 px)
    and a static noise texture for its droplet-y body."""
    xs = np.arange(-8, 106)
    R = sc.rng(4141)
    prof = np.zeros(len(xs))
    i = 0
    while i < len(xs):
        w_ = 2 + int(R.random() * 3)
        prof[i:i + w_] = 0.45 + 0.55 * R.random()
        i += w_
    bell = np.clip(1 - ((xs - 48) / 62.0) ** 2, 0, 1) ** 0.5
    noise = R.random((48, (len(xs) + 1) // 2)).astype(np.float32)          # chunky 2x2 droplets
    noise = np.repeat(np.repeat(noise, 2, axis=0), 2, axis=1)[:, :len(xs)]
    return xs, prof * bell, noise


_SPRAY = np.array([col(x) for x in ('#9fd6d0', 'ice', 'mist', 'white')], np.uint8)


def _spray_sheet(c, a, fxo):
    """Jets of white water thrown up the face of the stack, dissolving into droplets after ~14 steps."""
    if not 0 <= a < 36:
        return
    hmax = 84 * math.sin(min(1.0, a / 10) * math.pi / 2)
    dens = 1.0 if a < 10 else max(0.0, 1 - (a - 10) / 20)
    xs, pr, noise = _sheet_cols()
    X = xs + fxo
    ok = (X >= 0) & (X < W)
    if not ok.any():
        return
    X, pr, nz = X[ok], pr[ok], noise[:, ok]
    h = np.maximum(1, (hmax * pr).astype(int))
    rows = np.arange(96)[:, None]
    frac = rows / h[None, :]
    nz = np.roll(nz, -2 * a, axis=0)                     # the droplets stream upward
    inside = rows < h[None, :]
    keep = inside & (nz < dens * (0.78 - 0.5 * frac))
    crown = inside & (rows >= h[None, :] - 2) & (dens > 0.4)
    keep |= crown
    lvl = np.clip((frac * 3.2 + 0.4).astype(int), 0, 3)
    lvl[crown] = 3
    yy = H - 3 - rows + np.zeros_like(h)[None, :]
    xx = np.broadcast_to(X[None, :], keep.shape)
    keep &= yy >= 0
    c.a[yy[keep], xx[keep]] = _SPRAY[lvl[keep]]


def _surf(c, a, fxo, f):
    """A rogue wave slams the sea stack: a foaming crest, a wall of spray up the rock face that falls
    back as droplets, drifting sea mist and a little rainbow (opposite the low sun) in the spray."""
    if a < 20:                                           # the crest rearing up along the foot
        hw = int(math.sin(a / 20 * math.pi) * 9)
        for x in range(max(0, fxo - 30), min(W, fxo + 126)):
            y = H - 3 - hw + int(math.sin(x * 0.3 + a * 0.7) * 1.5)
            c.px(x, y, 'white')
            if hash01(x, a // 2) < 0.7:
                c.px(x, y + 1, 'mist')
            if hash01(x, a // 2, 3) < 0.5:
                c.px(x, y + 2, '#9fd6d0')
    _spray_sheet(c, a - 4, fxo)
    for q in range(3):                                   # sea mist drifting off with the wind
        age = a - 12 - q * 4
        if not 0 <= age < 56:
            continue
        r = 4 + min(age, 24) * 0.18
        lvl = 0.6 if age < 30 else 0.6 - (age - 30) / 26 * 0.5
        sc.draw_puff(c, fxo + 24 + q * 24 + int(age * 0.5), int(H - 74 - (q % 2) * 6 - age * 0.2), int(r), 4,
                     seed=q, level=lvl)
    _bombs(c, a - 14, fxo + 48, H - 70, 60, 11, 2.6, 0.4, 1.6, 0.11, 36, ('white', 'white', 'mist', 'ice'),
           stagger=10, big=0.4)
    _bombs(c, a - 4, fxo + 48, H - 4, 40, 12, 1.3, 2.6, 4.6, 0.12, 44, ('white', 'mist', 'ice'), stagger=8)
    if 12 <= a < 84:                                     # the rainbow in the spray
        al = 0.55 * min(1.0, (a - 12) / 10) * min(1.0, (84 - a) / 22)
        dx, dy, band = _rainbow_px()
        xs, ys = fxo + 48 + dx, H - 4 + dy
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys, band = xs[ok], ys[ok], band[ok]
        v = c.a[ys, xs].astype(np.float32) * (1 - al) + _RAINBOW[band] * al
        c.a[ys, xs] = v.astype(np.uint8)


_GEMS = ('crimson', 'red', 'blue', 'crystal_lt', 'green', 'mint', 'purple', 'lilac', 'gold', 'butter', 'white')


def _gem_shower(c, a, mox, sy, piles, f):
    """The gem hoards answer: the big pile flares gold, shafts of light fan up out of it, a fountain of
    gems and coins leaps high and rains back onto the piles, and glitter twinkles all over the hoards."""
    k = 1 - a / 72
    (px_, py_, rw, rh) = piles[0]
    gx, gy = px_ + mox, py_ - rh + sy                    # top of the big pile
    up = min(1.0, a / 6)
    fx.glow(c, gx, gy + 4, 54, 'gold', 0.42 * k * up, rings=5)
    fx.glow(c, gx, gy + 2, 16, 'butter', 0.55 * k * up, rings=3)
    if a < 56:                                           # shafts of light fanning up out of the hoard
        kk = min(1.0, a / 8) * (1 - a / 56)
        for j in range(5):
            ang = -math.pi / 2 + (j - 2) * 0.32 + math.sin(a * 0.05 + j) * 0.04
            L = 70 + 8 * (j % 2)
            for q in range(6, L, 2):
                x = int(gx + math.cos(ang) * q)
                y = int(gy + 2 + math.sin(ang) * q)
                w_ = 1 + q // 22
                if 0 <= y < H and hash01(j, q, a // 3) < 0.85:
                    c.blend(x - w_, y, 2 * w_ + 1, 2, 'butter', 0.30 * kk * (1 - q / L))
    for j, (qx, qy, qw, qh) in enumerate(piles[:3]):     # the fountain of gems and coins
        top = qy - qh * 0.6 + sy
        _bombs(c, a - j * 5, qx + mox, top, 56 if j == 0 else 16, 20 + j, 1.6, 2.4 if j == 0 else 1.4,
               4.2 if j == 0 else 2.6, 0.12, 64, _GEMS, stop=lambda x, y, yy=qy + sy: y > yy - 3,
               stagger=18, big=0.92)
    for i in range(36):                                  # glitter all over the hoards
        (qx, qy, qw, qh) = piles[i % len(piles)]
        sx_ = int(qx + (hash01(i, 91) - 0.5) * 1.8 * qw) + mox
        sy_ = int(qy - hash01(i, 92) * qh * 1.1) + sy
        if hash01(i, 93) * 72 < 72 - a:
            fx.sparkle(c, sx_, sy_, f + i * 3, ('white', 'butter', 'crystal_lt', 'lilac')[i % 4], period=12,
                       size=3 if i % 3 == 0 else 2)


SKY_LANTERNS = 8
SKY_LIFE = 240          # steps a sky lantern is drawn (it has risen off the top of the screen by then)


def _sky_lantern(c, x, y, d, i, lk=0.0, glow=None):
    """A paper sky lantern (5 x 7) with its flame, glowing warm (its halo by day and dusk: lk, the clock's
    lamp level, fades it out; at night the lantern itself shines: marked in the light mask glow)."""
    fl = (0.30 + 0.08 * math.sin(d * 0.5 + i * 1.7)) * (1 - lk)
    if fl > 0.01:
        fx.glow(c, x, y + 3, 10, 'amber', fl, rings=3)
    _mark(glow, c, (x - 2, y, x + 3, y + 8), _sky_lantern_body, x, y, d, i)


def _sky_lantern_body(c, x, y, d, i):
    c.hline(x - 1, x + 1, y, 'brick')
    c.rect(x - 2, y + 1, 5, 5, 'vermilion')
    c.vline(x - 2, y + 1, y + 5, 'brick')
    c.rect(x - 1, y + 2, 3, 3, '#ffb070')
    c.px(x, y + 3, 'butter' if (d // 4 + i) % 3 else 'white')
    c.hline(x - 1, x + 1, y + 6, '#7a2a2a')
    c.px(x, y + 7, 'gold' if (d // 3) % 2 else 'lava_hi')


def _lanterns(c, t, f, a, starts, pvx, pvy, ax, ay, bx_, by_, lk=0.0, glow=None):
    """Kirin Grove lights up: a wave of light runs along the lantern string to the pavilion, every lantern
    and the moon windows blaze, and paper sky lanterns rise from the courtyard into the sky (the blazing
    lanterns, windows and sky lanterns marked in the light mask glow; not their halos)."""
    if 0 <= a < 60:
        # (x, top, size, seed) as the scene draws them, in the order the wave reaches them
        lan = [(ax + 2, ay + 1, 2, 7)]
        lan += [(int(lerp(ax, bx_, k / 5)), int(lerp(ay, by_, k / 5) + 6 * math.sin(k / 5 * math.pi)) + 1, 1, k)
                for k in range(1, 5)]
        lan += [(pvx + 1, pvy + 30, 1, 1), (pvx + 15, pvy + 12, 1, 3), (pvx + 61, pvy + 12, 1, 4),
                (pvx + 75, pvy + 30, 1, 2)]
        for j, (lx, ly, size, seed) in enumerate(lan):
            d = a - j * 3
            if d < 0:
                continue
            lx += int(round(math.sin(f * 2 * math.pi / 60 + seed) * 0.8))
            bw, bh = (3, 4) if size == 1 else (5, 6)
            x0, y0 = lx - bw // 2, ly + size
            if d < 6:
                c.rect(x0, y0, bw, bh, 'white')
                fx.sparkle(c, lx, y0 + bh // 2, d, 'white', period=12, size=4)
            elif d < 34:
                c.rect(x0, y0, bw, bh, 'butter')
                c.px(lx, y0 + bh // 2, 'white')
            elif d < 48:
                c.rect(x0 + 1 if bw > 3 else x0, y0 + 1, bw - (2 if bw > 3 else 0), bh - 2, '#ffd08a')
            if d < 48:
                _mark_px(glow, np.tile(np.arange(x0, x0 + bw), bh), np.arange(y0, y0 + bh).repeat(bw))
            e = min(1.0, d / 3) * max(0.0, 1 - max(0, d - 20) / 40)
            fx.glow(c, lx, y0 + bh // 2, 10, 'gold', 0.5 * e, rings=3)
        k = min(1.0, a / 8) * max(0.0, 1 - max(0, a - 30) / 30)
        for (wx, wy, r) in ((pvx + 38 - 16, pvy + 43, 5), (pvx + 38 + 16, pvy + 43, 5), (pvx + 38, pvy + 16, 2)):
            if k > 0.25:
                for (rr, cl) in ((r - 1, 'butter'), (max(1, r - 3), 'white')):
                    _mark(glow, c, (wx - r, wy - r, wx + r + 1, wy + r + 1), c.__class__.circle, wx, wy, rr, cl)
            fx.glow(c, wx, wy, 12 + r, 'gold', 0.55 * k, rings=3)
    for t0 in starts:                                    # sky lanterns (several launches may overlap)
        for i in range(SKY_LANTERNS):
            d = t - t0 - 6 - i * 5
            if not 0 <= d < SKY_LIFE:
                continue
            sp = 0.55 + 0.12 * hash01(i, 401)
            x = pvx + 38 + int((hash01(i, 402) - 0.5) * 64) + int(d * (0.08 + 0.10 * hash01(i, 403))
                                                                   + math.sin(d * 0.06 + i) * 2.5)
            y = pvy + 54 - int(d * sp)
            if y < -10 or not -6 < x < W + 6:
                continue
            _sky_lantern(c, x, y, d, i, lk, glow)


def _warm():
    """Build every cached layer up front (~0.3 s, while the region loads at the landing press) so the
    first sweep into the caldera or the grove doesn't stall (the grove layers alone take ~150 ms)."""
    sc._sky_a(), sc._strata(), sc._island(), sc._wave_tex(), sc._fg_a(), _beach(), _sea_cols()
    sc._sky_b(), sc._wall_b(), sc._mid_b(), sc._lava_tex(), sc._crusts(), sc._fg_b()
    sc._sky_c(), sc._far_karsts(), sc._lagoon(), sc._mid_karsts(), sc._bank(), sc._pavilion(), _fg_parts()
    _sunless("a"), _sunless("c"), _pavilion_lights()
    back = ('#3f7a52', '#5f9a62', '#8cbc78', '#bfe0a0')
    for s in range(-2, 3):                               # the bamboo clumps at every sway offset
        sc._bamboo(s, 501, 26, 170, 300, 46, 124, sc.GW, back, True, 2)
        sc._bamboo(s, 502, 18, 0, 92, 50, 126, sc.GW, back, True, 2)
    front = ('#2e7a3a', '#4fa04a', '#86c85a', '#c8e88a')
    for s in range(-3, 4):
        sc._bamboo(s, 601, 6, -8, 22, -40, H, 380, front, False, 3)
        sc._bamboo(s, 602, 6, 326, 356, -40, H, 380, front, False, 3)


# hotspots (world px of the thing; p = parallax of its layer) and hidden stamps (main-layer world px)
HOTSPOTS_SEA = [
    Hotspot('crater', sc.VX, sc.VT - 3, VOLCANO_ICON, cooldown=96, p=sc.A_PAR),    # THE VOLCANO's crater
    Hotspot('surf', 250 + 48, 112, 'splash', cooldown=84, p=1.25),                 # the sea stack
]
HOTSPOTS_HEART = [
    Hotspot('erupt', sc.POOL[0], sc.POOL[1] - 16, VOLCANO_ICON, cooldown=150),      # the hot pool
    Hotspot('gems', 247 - 6, 115, 'gem', cooldown=72, p=0.6),                       # the front gem pile
]
HOTSPOTS_GROVE = [
    Hotspot('lanterns', 163, 110, 'lantern', cooldown=75, p=0.6),                   # the lantern string
]
STAMPS_SEA = [Stamp('plume', 330, 22)]            # up high in the ash plume, east of the crater
STAMPS_HEART = [Stamp('throne', 50, 27)]          # over the empty throne's centre spire
STAMPS_GROVE = [Stamp('bridge', 108, 133)]        # low under the moon bridge, over the little falls

TINT_SEA = ('#ffd8c0', 0.14)                      # a warm volcanic haze (the sky is the clock's now)
TINT_HEART = ('#ff9a6a', 0.38)                    # lava glow from below
TINT_GROVE = ('#d6f0dc', 0.20)                    # jade light in the grove


class DragonLands(Region):
    SEG = 'dragonlands'
    CARD = ('DRAGON LANDS', ['VOLCANOES - LAVA - GEMS', 'KIRIN GROVE TO THE SOUTH'])

    def build_areas(self):
        return [
            Area(self.render_a, cam_x=(0, SEA_CAM), cam_y=(0, 0), y_range=(30, 146),
                 start=(110, 80), landmarks=LANDMARKS_SEA, balloon='m', balloon_tint=TINT_SEA,
                 hotspots=HOTSPOTS_SEA, stamps=STAMPS_SEA, sky=self.sky_a),
            Area(self.render_b, cam_x=(0, HEART_CAM), cam_y=(0, 0), y_range=(30, 128),
                 start=(70, 66), landmarks=LANDMARKS_HEART, balloon='m', balloon_tint=TINT_HEART,
                 hotspots=HOTSPOTS_HEART, stamps=STAMPS_HEART, sky=self.sky_b),
            Area(self.render_c, cam_x=(0, GROVE_CAM), cam_y=(0, 0), y_range=(28, 146),
                 start=(64, 62), landmarks=LANDMARKS_GROVE, balloon='m', balloon_tint=TINT_GROVE,
                 hotspots=HOTSPOTS_GROVE, stamps=STAMPS_GROVE, sky=self.sky_c),
        ]

    def warm(self):
        _warm()
        super().warm()

    # ------------------------------------------------------------------ clocks
    def init_state(self):
        self.area_t0 = 0            # step the current area was entered (ambient sound cadence)
        self.erupt_t0 = 0           # step the eruption clock was (re)started
        self.erupt_visit = -1       # the visit whose automatic eruption has played (once per visit)
        self.lantern_ts = ()        # recent lantern launches (sky lanterns of several may overlap)

    def on_area(self, ai):
        self.area_t0 = self.g.t
        if ai == 1 and self.erupt_visit != self.visits:  # first time in the caldera this visit: it erupts
            self.erupt_visit = self.visits               # shortly after arrival (rumble ~1 s, burst ~1.5 s)
            self.erupt_t0 = self.g.t
        elif self.erupt_f(self.g.t) < 196 or self.erupt_f(self.g.t) >= 282:
            self.erupt_t0 = -10 ** 6                     # later: calm until SPACE at the hot pool
        g = self.g
        if ai == 0:
            g.sfx('rumble', 0.4)                         # the volcano breathing as you land
        elif ai == 1:
            g.sfx('lava', 0.55, 0)                       # into the caldera: the lava river blorps
        else:
            g.sfx('gong', 0.4)                           # calm after the fire

    def erupt_f(self, t):
        return ERUPT_LEAD + min(t - self.erupt_t0, ERUPT_PERIOD)     # rests at ERUPT_CALM when done

    def ages(self, t):
        return {k: t - self.spot_t.get(k, -10 ** 6) for k in ('crater', 'surf', 'gems', 'lanterns')}

    # ------------------------------------------------------------------ SPACE
    def on_interact(self, key, t):
        g = self.g
        if key == 'crater':
            g.sfx('rumble', 0.6)
            g.sfx('lava', 0.55, 0)
        elif key == 'surf':
            g.sfx('waves', 0.65)
            g.sfx('splash', 0.5)
        elif key == 'erupt':
            ef = self.erupt_f(t)
            if ef < 196 or ef >= 282:                    # calm: wake it now (fountain 8 steps later)
                self.erupt_t0 = t - (ERUPT_ERUPT_AT - ERUPT_LEAD)
                g.sfx('rumble', 0.9)
            else:                                        # it's already going: just a louder rumble
                g.sfx('rumble', 0.5)
        elif key == 'gems':
            g.sfx('crystal', 0.55)
            g.sfx('clink', 0.45)
        elif key == 'lanterns':
            self.lantern_ts = tuple(x for x in self.lantern_ts if t - x < SKY_LIFE + 50) + (t,)
            g.sfx('chime', 0.55)
            g.sfx('shimmer', 0.3)

    # ------------------------------------------------------------------ sound
    def update(self, t):
        g = self.g
        k = t - self.area_t0
        f = self.scene_f(t)
        if self.ai == 0:
            if t - self.t_enter == 6:
                g.sfx('wind', 0.3)
            if f % 144 == 84:                            # surf breaking on the black sand (foam peak)
                g.sfx('waves', 0.45 * self.near_gain(170))
            if k % 220 == 110:
                g.sfx('lava', 0.25 * self.near_gain(VOLC_X), 0)      # the crater bubbling
            if k % 300 == 200:
                g.sfx('wind', 0.25, 0)
        elif self.ai == 1:
            ef = self.erupt_f(t)
            if ef == 198:
                g.sfx('rumble', 0.85)                    # the pool swells, tremor
            elif ef == sc.ERUPT:
                g.sfx('lava', 1.0, 1)                    # ERUPT: lava fountain boom
            elif ef == sc.ERUPT + 18:
                g.sfx('steam', 0.4)                      # vents hiss after the blast
            if f % 72 == 0 and not 190 <= ef < 250:
                g.sfx('steam', 0.3 * self.near_gain(155))    # the geyser on the beat
            if k % 160 == 80 and not 190 <= ef < 260:
                g.sfx('lava', 0.3, 0)
        else:
            if k % 75 == 4:
                g.sfx('waterfall', 0.32 if k < 80 else 0.24)  # the stream's little falls
            if k % 240 == 150:
                g.sfx('wind', 0.3, 0)                    # bamboo rustle

    # ------------------------------------------------------------------ skies alone (see _sky_only)
    # (the clock's sky, as the renders draw it first: _clock_sky)
    def sky_a(self, f, cx, cy):
        from engine import Canvas
        a = _clock_sky('a', self.g.hour, cx, self.g.t)[1].copy()
        _stamp(a, [(sc._island()[0], -int(round(cx * sc.A_PAR)), 0),
                   (sc._fg_a(), 250 - int(round(cx * 1.25)), H - 94)])
        return Canvas(arr=a)

    def sky_b(self, f, cx, cy):
        from engine import Canvas
        sx, sy = 0, 0                                    # (no eruption shake, see render_heart)
        ef = self.erupt_f(self.g.t)                      # the light render_heart puts on the sky too
        a = _clock_sky('b', self.g.hour, cx, self.g.t)[1].copy()
        top = Canvas(arr=a[:70])                         # (the sky's rows only, as _sky_b_only: same pixels)
        _pool_glow(top, ef - sc.ERUPT, -int(cx))
        _heart_grade(top, ef)
        _stamp(a, [(sc._wall_b()[0], -int(cx * 0.3) - 20 + sx, sy), (sc._mid_b()[0], -int(cx * 0.6) - 6 + sx, sy),
                   (sc._fg_b()[0], -int(cx * 1.3) - 8 + sx, sy)])
        return Canvas(arr=a)

    def sky_c(self, f, cx, cy):
        from engine import Canvas
        a = _clock_sky('c', self.g.hour, cx, self.g.t)[1].copy()
        box = -int(cx * 0.6) - 10
        _stamp(a, [(sc._far_karsts(), -int(cx * 0.12) - 10, 0), (sc._lagoon()[0], -int(cx * 0.2) - 10, 0),
                   (sc._mid_karsts(), -int(cx * 0.35) - 8, 0), (sc._bank()[0], box, 0),
                   (sc._pavilion(), 196 + box, sc.BANK - 60)])
        return Canvas(arr=a)

    # ------------------------------------------------------------------ renders
    def render_a(self, f, cx, cy):
        t = self.g.t
        return self._lit(render_sea(f, cx, t, self.ages(t), self.g.hour, self._new_glow()))

    def render_b(self, f, cx, cy):
        t = self.g.t
        return self._lit(render_heart(f, cx, self.erupt_f(t), self.ages(t), t, self.g.hour, self._new_glow()))

    def render_c(self, f, cx, cy):
        t = self.g.t
        return self._lit(render_grove(f, cx, t, self.ages(t), self.lantern_ts, self.g.hour, self._new_glow()))

    # ------------------------------------------------------------------ the clock's lights
    _glow = None
    _pre = None

    def auto_glow(self):
        return False                                     # the real lights are kept by mask (after_light)

    def _new_glow(self):
        self._glow = np.zeros((H, W), bool)
        return self._glow

    def _lit(self, c):
        """End of an area render: the marked lights that are still lights (nothing nearer drawn over them),
        and the painted picture, for after_light."""
        self._glow &= _bright(c.a)
        self._pre = c.a.copy()
        return c

    def after_light(self, c, t):
        """Right after the clock's light: the real lights (lava, crater, embers, glitter on the sea, fires,
        lit windows and lanterns) get their painted colour back."""
        m, pre, self._glow, self._pre = self._glow, self._pre, None, None
        if m is None or pre is None or abs(daylight._q(self.g.hour) - daylight._q(self.HOUR)) < 0.05:
            return
        memo = self._sky_memo                            # (never over the repainted open sky)
        if memo is not None and memo[0][:2] == (self.ai, t):
            m = m & ~memo[1]
        c.a[m] = pre[m]
