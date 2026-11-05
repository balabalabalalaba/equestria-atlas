"""Yakyakistan, the Frozen North and Mt. Everhoof as an explorable diorama (art: scenes/s19_yakyakistan.py,
used read-only).

The film shows three views (a zoom into the gates, then a cloud pass), so the region has three areas,
flown through left to right with a cloud sweep between them:
  0  THE LOG GATES (shot A): the palisade with its horned gate, the village roofs and watchtower behind
     it, the far range with Mt. Everhoof's distant pyramid. The palisade runs on east of the gate past
     a third turret, more torches, banners and village roofs, to a stand of big pines.
  1  INSIDE THE WALLS (shot B): huts, drying rugs, firewood and the Great Hall on the plateau, the
     glossy Ice Slide winding down the flank above the moonlit sea of clouds. Past the plateau the
     flank runs on east (more rock ledges, a few pines) over the cloud sea.
  2  THE FROZEN NORTH (shot C): the blue ice cliff and its glowing cave, the igloos, the aurora and
     Mt. Everhoof's colossal summit, with the film's tilt (the camera pans up to the summit). The
     drifts run on east to more igloos, snow lanterns and a cluster of ice spires.
The film's layers are only a little wider than the screen, so each is rebuilt wider by the film's own
builder (`_film`: a copy of the builder with its layer width and art lists swapped and its globals
rebound), with extra art drawn by the scene's own helpers (huts, turrets, pines, rocks, igloos, ice).
Every view is recomposed in the film's layer order with all its helpers running on the free scene clock
(snow, wind gusts, smoke, banners, pennants, torches, rugs, plume, spindrift, cave light, aurora). The
film's balloons are never drawn. The cloud sea sways to and fro instead of drifting out of its layer.
The balloon is lit by the moon (bluish), warm beside torches, braziers and lanterns, icy blue at the cave.
The world clock (game/daylight.py): the fires, lanterns and windows follow it (cold and dark by day, coming on
one by one through the dusk), with the layers' baked light pools; see "the world clock's lights" below.

Things to set off with SPACE: sound the great horn on the gate (a gust whips every banner and pennant,
the snow streams sideways, the palisade sheds its snow and the fires roar up); strike the Great Hall's
gong (its boom cracks the roof's snow slab: the film's avalanche, replayed, and the roof snows over
again a few seconds later); push a sled down the Ice Slide (it races down the curves, spraying ice, the
lanterns flaring as it passes); wake the crystals of the ice cave (a blaze of blue light and shards);
call up the aurora (curtains surge across the whole sky). Three atlas stamps are hidden: low behind the
big pines at the far end of the palisade, under the Ice Slide's lowest bend over the cloud sea, and at
the top of the world, above Mt. Everhoof's summit.
"""
import math
import types
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, Sprite, fx, hash01, clamp, mix, smooth
from engine.core import bayer
import scenes.s19_yakyakistan as sc

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind


def _film(fn, consts=None, **names):
    """A copy of one of the film's layer builders (unwrapped from its lru_cache) with some literal
    constants swapped (the layer width, its art lists) and some module globals rebound: the film's own
    code, run over a wider strip."""
    raw = getattr(fn, '__wrapped__', fn)
    code = raw.__code__
    if consts:
        def swap(v):
            for old, new in consts.items():
                if type(v) is type(old) and v == old:
                    return new
            return v
        code = code.replace(co_consts=tuple(swap(v) for v in code.co_consts))
    g = dict(raw.__globals__)
    g.update(names)
    return types.FunctionType(code, g, raw.__name__, raw.__defaults__, raw.__closure__)


def _over(base, top):
    """RGBA `top` over RGBA `base` (same size, hard alpha)."""
    out = base.copy()
    m = top[..., 3] >= 128
    out[m] = top[m]
    return out


def _under(base, bottom):
    """RGBA `bottom` behind RGBA `base` (same size, hard alpha)."""
    out = base.copy()
    m = (base[..., 3] < 128) & (bottom[..., 3] >= 128)
    out[m] = bottom[m]
    return out


def _vis(x, w=0, pad=24):
    return -w - pad < x < W + pad


def _ring(c, x, y, r, color, level=1.0, sy=1.0):
    """A 1-px ring (radius r, vertical squash sy), dithered away as level drops below 1."""
    if r < 1 or level <= 0:
        return
    ry = max(1.0, r * sy)
    x0, y0, x1, y1 = c._clip(int(x - r - 2), int(y - ry - 2), int(2 * r + 5), int(2 * ry + 5))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xx - x) / (r + 0.5)) ** 2 + ((yy - y) / (ry + 0.5)) ** 2)
    m = (d <= 1.0) & (d > 1.0 - 1.15 / max(r, 1.0))
    if level < 1:
        m &= bayer(y1 - y0, x1 - x0, x0, y0) < level
    c.a[y0:y1, x0:x1][m] = np.array(sc.col(color), np.uint8)


def _powder(c, x, y0, y1, level):
    """A curtain of snow powder pouring from y0 (thin tail) to y1 (thick head) at column x."""
    x0, ya, x1, yb = c._clip(x - 3, y0, 7, y1 - y0 + 1)
    if x1 <= x0 or yb <= ya:
        return
    yy, xx = np.mgrid[ya:yb, x0:x1]
    u = (yy - y0) / max(1.0, y1 - y0)
    wob = np.round(np.sin(yy * 0.45 + x) * 0.6)
    d = np.abs(xx - x - wob)
    thr = bayer(yb - ya, x1 - x0, x0, ya)
    m = (d <= 0.6 + 2.0 * u) & (thr < level * (0.2 + 0.7 * u))
    core = m & (d <= 0.6 + 0.9 * u)
    c.a[ya:yb, x0:x1][m & ~core] = np.array(sc.col('#b8c6f1'), np.uint8)
    c.a[ya:yb, x0:x1][core] = np.array(sc.col('#e4ebff'), np.uint8)


def _tint_between(a, b, k):
    k = clamp(round(k * 8) / 8)                 # eighths: the compositor caches one sprite per tint
    return (mix(a[0], b[0], k), round(a[1] + (b[1] - a[1]) * k, 3))


TINT_MOON = ('#7078c4', 0.42)
TINT_FIRE = ('#ffb27a', 0.3)
TINT_ICE = ('#6ab8f0', 0.36)
GX, GY = sc.GX, sc.GY

# ============================================================================= the world clock's lights
# The film painted the north at night, every fire and window lit and their light pools baked into the
# layers. game/daylight.py lifts that night to the clock's hour, and by day dims what looks like a lamp
# towards dark glass: that left orange dithered pools on the palisade and the snow, grey-blue flame shapes,
# half-lit windows at dusk (dimmed 58 % at 19:12) and blue-grey specks on every painted gold trim by day.
# So the lights here follow the clock themselves: one ramp (_lamp_k: 0 by day, on through the dusk, off
# through the dawn), every light (window, torch, brazier, lantern, snow lamp, igloo door) going on at its
# own point of it (_lv) and its halo fading in with it; the layers' baked pools fade with the ramp (_lit:
# the film's builder rerun without them, plus their light times the ramp) and their windows are dark glass
# while off; by day the torches and braziers stand cold. After the relight, what its lamp switch-off
# caught is put back as the plain relight would have it (after_light): those are painted trims now.
LAMP_V = 0.9            # the lights' thresholds spread over this much of the ramp
LAMP_FADE = 0.15        # ...and each one fades in over this much of it


def _lamp_k(hour):
    """0 (day) .. 1 (night): how far the lights are on (just starting ~18:50, all on by ~19:35; off again
    from ~05:40 to ~06:30)."""
    return clamp((daylight.night(hour) - 0.05) / 0.55)


def _lv(k, *seed):
    """0..1: one light's own level on the ramp k (its threshold hashed from seed)."""
    return clamp((k - hash01(*seed, 911) * LAMP_V) / LAMP_FADE)


def _nop(*_a, **_k):
    pass


def _cold_brazier(c, x, base, f, seed):
    """A brazier standing cold by day: the bowl and tripod, ash and charcoal, no flame."""
    _brazier_bowl(c, x, base, f, seed)
    c.hline(x - 4, x + 4, base - 13, '#2b2430')
    c.hline(x - 3, x + 3, base - 13, '#3d3440')
    c.px(x - 1, base - 14, '#4a4250')
    c.px(x + 2, base - 14, '#2b2430')


_brazier_bowl = rebind(sc.brazier, flame=_nop)
_cold_torch = rebind(sc.wall_torch, flame=_nop)


def _cold_lantern_px(c, x, y):
    """An unlit lantern's glass (dark, lit like the hour by the relight)."""
    c.px(x, y, '#3c4660')
    c.px(x, y + 1, '#30384e')


# film layers with baked light pools: the same builders rerun without bake_lights (the unlit layers)
@lru_cache(maxsize=1)
def _wall_ax_unlit():
    return _film(sc._wall_a, {352: WALL_AW}, TURRETS_A=TURRETS_AX, LIGHTS_A=LIGHTS_AX, bake_lights=_nop)()


@lru_cache(maxsize=1)
def _village_b_unlit():
    return _film(sc._village_b, bake_lights=_nop)()


@lru_cache(maxsize=1)
def _igloo_cx_unlit():
    return _film(sc._igloo_c, {340: IGL_CW, 320: 494}, IGLOOS=IGLOOS_X, SNOWLAMPS=SNOWLAMPS_X,
                 bake_lights=_nop)()


_LAYERS = {             # key -> (the film's lit layer, the unlit one) as RGBA getters
    'vil_a': (lambda: _village_ax()[0], lambda: _village_ax()[0]),
    'wall_a': (lambda: _wall_ax(), lambda: _wall_ax_unlit()),
    'vil_b': (lambda: sc._village_b()[0], lambda: _village_b_unlit()[0]),
    'igl_c': (lambda: _igloo_cx()[0], lambda: _igloo_cx_unlit()[0]),
}
_WARM = np.array([sc.col(x) for x in sc.WARM], np.uint8)            # the film's lit-glass ramp
_GLASS = np.array([sc.col(x) for x in ('#1d2132', '#262c40', '#30384e', '#3c4660', '#4e5a76')], np.uint8)


@lru_cache(maxsize=None)
def _layer_lights(key):
    """(unlit layer, baked light (int16 RGB), window labels, window thresholds, lit layer) of a layer: its
    windows are the film's warm glass (lit windows and doorways, turret and lantern rooms, the gate crack,
    the hall's oculus, igloo doors, snow lamps: exact ramp colours, the night grade leaves them alone)."""
    from scipy import ndimage
    lit, unlit = (g() for g in _LAYERS[key])
    rgb = unlit[..., :3]
    idx = np.full(rgb.shape[:2], -1, np.int8)
    for i, cw in enumerate(_WARM):
        idx[np.all(rgb == cw, axis=2)] = i
    win = (unlit[..., 3] >= 128) & (idx >= 0)
    lab, n = ndimage.label(win, structure=np.ones((3, 3)))
    v = np.full(n + 1, 2.0, np.float32)
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        v[i] = hash01(s_[1].start, s_[0].start, len(key), 912) * LAMP_V
    dark = unlit.copy()
    dark[win, :3] = _GLASS[idx[win]]
    glow = lit[..., :3].astype(np.int16) - unlit[..., :3].astype(np.int16)
    return dark, glow, lab, v, unlit, lit


@lru_cache(maxsize=24)
def _lit(key, q):
    """A layer at lamp level q/16: each window lit once the ramp passes its threshold (dark glass before),
    the baked light pools at q/16 of their strength (q 16: exactly the film's layer)."""
    dark, glow, lab, v, unlit, lit = _layer_lights(key)
    if q >= 16:
        return lit
    out = dark.copy()
    on = (v < q / 16.0)[lab] & (lab > 0)
    out[on] = unlit[on]
    if q > 0:
        out[..., :3] = np.clip(out[..., :3] + (glow * (q / 16.0)).astype(np.int16), 0, 255).astype(np.uint8)
    return out


def _scaled_glow(s):
    """sc.glow_a at s times its strength (the cave's light by the clock)."""
    def glow(a, cx, cy, r, color, k, rings=4, sx=1.0):
        if k * s > 0.004:
            sc.glow_a(a, cx, cy, r, color, k * s, rings=rings, sx=sx)
    return glow


@lru_cache(maxsize=9)
def _cave_light(q):
    """The film's cave_light with its halos at q/8 (the crystals themselves always show)."""
    return rebind(sc.cave_light, glow_a=_scaled_glow(q / 8.0))

# ============================================================================= area 0: the log gates
# film layers: far 330 (p .1), mid 340 (.3), village 350 (.7), wall 352 (.85), fg 372 (1.35): they end
# at camera ~36. Rebuilt for CAM_A: every layer must reach p * CAM_A + 320.
CAM_A = 250
FAR_AW, MID_AW, VIL_AW, WALL_AW, FG_AW = 360, 404, 512, 540, 680
_MTS_A = ((-6, 86, 40, 46, 12), (150, 80, 46, 40, 13), (226, 92, 40, 50, 14), (300, 78, 44, 46, 15),
          (118, 98, 30, 34, 16))
_MTS_AX = _MTS_A[:4] + ((378, 88, 42, 48, 17),) + _MTS_A[4:]
HUTS_AX = list(sc.HUTS_A) + [(356, 107, 26, 17, 18, 1), (390, 104, 36, 20, 24, 0), (436, 108, 24, 16, 16, 1),
                             (466, 105, 40, 20, 24, 0)]
TURRETS_AX = tuple(sc.TURRETS_A) + (478,)
LIGHTS_AX = list(sc.LIGHTS_A) + [(352, 112, 'torch'), (408, 112, 'torch'), (446, GY - 12, 'brazier'), (478, 87, 'lantern'),
                                 (520, 112, 'torch')]
# banners (wall x, rod y, seed, scheme, w) and pennant poles (wall x, seed, length): the film's + more
BANNERS_A = [(GX - 33, sc.POST_TOP + 3, 1, 0, 10), (GX + 23, sc.POST_TOP + 3, 2, 1, 10),
             (GX - 102, 106, 3, 2, 9), (GX + 92, 106, 4, 0, 9), (424, 106, 5, 1, 9), (500, 106, 6, 2, 9)]
PENNANTS_A = [(GX - 58, 11, 16), (GX + 56, 12, 15), (GX - 176, 13, 13), (GX + 166, 14, 14), (458, 15, 14),
              (528, 16, 11)]
FG_PINES_A = [(8, 124, 1810, 11), (34, 70, 1811, 8), (360, 92, 1812, 9), (492, 50, 1813, 7),
              (632, 120, 1814, 11), (662, 76, 1815, 8)]
FG_BUMPS_A = [(20, 30), (356, 26), (644, 30)]
FG_ROCKS_A = [(304, 170, 15, 8, 1820), (96, 175, 11, 6, 1821), (548, 173, 12, 6, 1824)]
HORN_XY = [(GX - 30, 46), (GX + 30, 46)]          # the crescent horn carvings on the gateposts (wall px)
HORN_T = 90             # the horn blast: gust, shed snow, roaring fires
WIND_E = 44             # px the falling snow is blown sideways by one blast (banked, never jumps back)


@lru_cache(maxsize=1)
def _far_ax():
    return _film(sc._far_a, {330: FAR_AW, _MTS_A: _MTS_AX})()


@lru_cache(maxsize=1)
def _mid_ax():
    return _film(sc._mid_a, {340: MID_AW, 110: 130})()


@lru_cache(maxsize=1)
def _village_ax():
    return _film(sc._village_a, {350: VIL_AW}, HUTS_A=HUTS_AX)()


@lru_cache(maxsize=1)
def _wall_ax():
    return _film(sc._wall_a, {352: WALL_AW}, TURRETS_A=TURRETS_AX, LIGHTS_A=LIGHTS_AX)()


@lru_cache(maxsize=1)
def _fg_ax():
    """The film's _fg_a (snow drifts, framing pines, rocks, the sign post) over a wider strip."""
    W_ = FG_AW
    lay = sc.Lay(W_, H)
    xs = np.arange(W_)
    gl = 169 + 4 * np.sin(xs / 31.0 + 0.5) + (sc.noise1(W_, 7, 1800) - 0.5) * 6
    for bx, bw in FG_BUMPS_A:
        gl = gl - 12 * np.exp(-((xs - bx) / float(bw)) ** 2)
    yy = np.arange(H)[:, None]
    m = yy >= gl[None, :]
    sl = np.gradient(gl)
    nz = sc.fbm(H, W_, 5, 1801, 3)
    t = 0.55 + np.clip(sl, -1.5, 1.5)[None, :] * 0.2 + (nz - 0.5) * 0.3 - (yy - gl[None, :]) * 0.025
    lay.put(0, 0, m, sc.M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.97
    for (px, h, seed, tiers) in FG_PINES_A:
        sc.pine(lay, px, 178, h, seed, tiers=tiers)
    for (rx, ry, rw, rh, s) in FG_ROCKS_A:
        rm = sc.emask(rw * 2 + 1, rh * 2 + 1, rw, rh, rw, rh)
        rm[rh + 2:] = False
        lay.put(rx - rw, ry - rh, rm, sc.M_ROCK, sc.blob_t(rm, rad=4))
    sc.snow_tops(lay, 286, 322, 2, 3, 1822, y_lo=150)
    sc.snow_tops(lay, 82, 110, 2, 2, 1823, y_lo=160)
    sc.snow_tops(lay, 532, 564, 2, 2, 1825, y_lo=160)
    cv = lay.canvas()
    sx, sy = 70, 170                                  # the film's sign post (mountain icon)
    cv.vline(sx, sy - 18, sy, '#2b1e2c')
    cv.vline(sx + 1, sy - 18, sy, '#4a2f35')
    cv.round_rect(sx - 8, sy - 22, 18, 10, '#3a2630', 1)
    cv.rect(sx - 7, sy - 21, 16, 8, '#5e3e3c')
    cv.tri((sx - 5, sy - 14), (sx, sy - 20), (sx + 5, sy - 14), '#c6d0f2')
    cv.tri((sx + 1, sy - 14), (sx + 4, sy - 18), (sx + 7, sy - 14), '#8fa1dc')
    cv.hline(sx - 7, sx + 8, sy - 23, '#e4ebff')
    cv.hline(sx - 6, sx + 7, sy - 24, '#b8c6f1')
    return sc.rgba(cv)


def _windy(e):
    """The film's wind-driven helpers with the gust raised by e (0..1): banners, pennants and smoke
    whip, flames lean."""
    if e <= 0.02:
        return dict(banner=sc.banner, pennant=sc.pennant, smoke=sc.smoke, wall_torch=sc.wall_torch,
                    brazier=sc.brazier)

    def gust(f_):
        g = sc.gust(f_)
        return g + (1 - g) * e

    flame = rebind(sc.flame, gust=gust)
    return dict(banner=rebind(sc.banner, gust=gust), pennant=rebind(sc.pennant, gust=gust),
                smoke=rebind(sc.smoke, gust=gust), wall_torch=rebind(sc.wall_torch, flame=flame),
                brazier=rebind(sc.brazier, flame=flame))


def _blow(a):
    """Sideways snow displacement after a steps of one horn blast (eased, ends flat)."""
    return WIND_E * smooth(a / 70.0)


# ============================================================================= area 1: inside the walls
# village (p 1) is open sky past its cornice (x 268) and the fg (1.3) drops below the screen past x 250;
# the slope (.85, 400 px) is rebuilt wider; far (.08, 340) ends at cam 250; the cloud sea (.2, 420)
# sways by up to SWAY px instead of drifting on.
CAM_B = 220
SLOPE_BW = 512
SWAY = 50
AVA = sc.AVA
GONG_T = 200            # gong -> avalanche (crack at +6, slide at +20) -> mound; the roof snows over again
REFILL = (140, 196)
SLED_T = 120
_LINE, _CUM = sc._slide_line()
SLIDE_L = float(_CUM[-1])
SLIDE_TAG = (int(_LINE[64][0]), int(_LINE[64][1]) - 1)
HALL_APEX = (int(round(sc.HALL_X + sc.HALL_W / 2 - 0.5)), sc.GB - sc.HALL_WH - sc.HALL_RH)
FLANK_PINES = [(418, 26, 2101), (440, 18, 2102), (474, 30, 2103), (500, 22, 2104)]


@lru_cache(maxsize=1)
def _slope_bx():
    slope, _ice = _film(sc._slope_b, {400: SLOPE_BW, 395: 500})()
    lay = sc.Lay(SLOPE_BW, H)
    xs = np.arange(SLOPE_BW)
    ridge = np.where(xs < 248, 140.0, 140 + (xs - 248) * 0.13)
    for (px, h, seed) in FLANK_PINES:
        sc.pine(lay, px, int(ridge[px]) + 6, h, seed, tiers=max(3, h // 6))
    return _over(slope, sc.rgba(lay.canvas()))


def _sled_at(s):
    """Slope-layer (x, y, facing) of a point s px down the ice slide."""
    s = clamp(s, 0, SLIDE_L - 0.01)
    i = int(np.searchsorted(_CUM, s, side='right')) - 1
    i = max(0, min(len(_LINE) - 2, i))
    p0, p1 = _LINE[i], _LINE[i + 1]
    k = (s - _CUM[i]) / max(1e-6, _CUM[i + 1] - _CUM[i])
    x = p0[0] + (p1[0] - p0[0]) * k
    y = p0[1] + (p1[1] - p0[1]) * k
    return int(round(x)), int(round(y)), (1 if p1[0] >= p0[0] else -1)


def _sled_s(a):
    """Distance down the slide a steps after the push (accelerating)."""
    return 0.4 * a + 0.075 * a * a


def draw_sled(c, x, y, d):
    """A little painted toboggan (no rider), runners on the ice at (x, y), front facing d."""
    def X(dx):
        return x + dx * d
    for dx in range(-6, 6):                              # runner with a curled front
        c.px(X(dx), y, '#cfd5f2' if dx % 4 == 1 else '#9ca2bf')
    c.px(X(6), y - 1, '#9ca2bf')
    c.px(X(7), y - 2, '#9ca2bf')
    c.px(X(7), y - 3, '#cfd5f2')
    c.px(X(6), y - 4, '#9ca2bf')
    for dx in (-4, 0, 4):                                # struts
        c.px(X(dx), y - 1, '#462b37')
    for dx in range(-6, 6):                              # painted deck (folk band)
        c.px(X(dx), y - 2, '#5e3a3d')
        c.px(X(dx), y - 3, ('#c23f55', '#e0b443', '#3f73cc')[(dx + 6) // 2 % 3])
        c.px(X(dx), y - 4, '#9a6a53')
    c.px(X(-6), y - 5, '#7a4f46')
    c.px(X(-6), y - 6, '#9a6a53')


# ============================================================================= area 2: the frozen north
# film layers (w x h, p for both axes): far 340 x 190 (.15), Everhoof 340 x 200 (.3), ice cliff 340 x 220
# (.6), igloos 340 x 230 (.85), drifts 360 x 244 (1.2): they end at cam ~23. Rebuilt for CAM_C; the tilt
# runs to TILT_C (the drifts end at cy 53).
CAM_C = 220
TILT_C = 36
FAR_CW, EVER_CW, MID_CW, IGL_CW, FG_CW = 372, 392, 460, 512, 592
_MTS_C = ((10, 84, 50, 44, 121), (300, 80, 46, 50, 122), (250, 100, 40, 40, 123), (60, 104, 40, 40, 124))
_MTS_CX = _MTS_C + ((366, 90, 44, 46, 125),)
IGLOOS_X = list(sc.IGLOOS) + [(352, 152, 14, 10), (412, 156, 19, 14), (466, 150, 10, 7)]
SNOWLAMPS_X = list(sc.SNOWLAMPS) + [(326, 158), (384, 157), (440, 159), (494, 157)]
_BOULDERS_C = ((-6, 196, 34, 30, 191), (322, 200, 36, 26, 192), (30, 206, 16, 12, 193))
_BOULDERS_CX = _BOULDERS_C + ((540, 198, 40, 30, 196), (512, 208, 18, 12, 197))
SPIRES_C = [(318, 140, 20, 34, 3101), (334, 140, 26, 52, 3102), (356, 140, 18, 30, 3103),
            (394, 140, 22, 40, 3104), (412, 140, 16, 24, 3105)]
POLES_C = [(84, 176, 41), (452, 178, 42)]        # folk marker poles in the drifts (fg px, pennant seed)
CAVE_T = 90
AURORA_T = 150


@lru_cache(maxsize=None)
def _aurora_full(w, seed):
    """A fuller curtain profile than the film's faint remnant (for the aurora the player calls up)."""
    a = sc.fbm(1, w, 22, seed, 3)[0]
    b = sc.fbm(1, w, 5, seed + 9, 2)[0]
    return np.clip((a - 0.18) * 2.0, 0, 1) * (0.5 + 0.5 * b)


_aurora_surge = rebind(sc.aurora, _aurora_prof=_aurora_full)


def _mountain_more(lay, ax, ay, *a, **k):
    """sc.mountain, plus one more peak east of the film's right-hand one (drawn first, behind it)."""
    if ax == 292:
        sc.mountain(lay, 382, 104, 178, 58, 64, 135, sc.M_MIDS, lit=(0.5, 0.86), shd=(0.12, 0.38), jag=3,
                    rock=0.14)
    return sc.mountain(lay, ax, ay, *a, **k)


@lru_cache(maxsize=1)
def _far_cx():
    return _film(sc._far_c, {340: FAR_CW, _MTS_C: _MTS_CX})()


@lru_cache(maxsize=1)
def _ever_cx():
    return _film(sc._ever_c, {340: EVER_CW}, mountain=_mountain_more)()


@lru_cache(maxsize=1)
def _mid_cx():
    base = _film(sc._mid_c, {340: MID_CW})()
    hh = H + 40
    lay = sc.Lay(MID_CW, hh)
    yb, xb = np.mgrid[0:hh, 0:MID_CW]
    for (bx, base_y, bw, bh, s) in SPIRES_C:               # ice spires behind the snow ridge
        pts = [(bx, base_y), (bx + bw * 0.18, base_y - bh * 0.55), (bx + bw * 0.46, base_y - bh),
               (bx + bw * 0.62, base_y - bh * 0.7), (bx + bw * 0.86, base_y - bh * 0.4), (bx + bw, base_y)]
        m = sc.pmask(MID_CW, hh, pts)
        tb = 0.32 + 0.38 * (xb > bx + bw * 0.46) + (sc.fbm(hh, MID_CW, 4, s, 2) - 0.5) * 0.3
        tb = tb - np.clip((yb - (base_y - bh)) / max(1, bh), 0, 1) * 0.15
        lay.put(0, 0, m, sc.M_ICE, tb)
        lay.put(0, 0, sc.edge_of(m) & (xb > bx + bw * 0.46), sc.M_ICE, 0.9)
    sc.snow_tops(lay, 310, 440, 2, 1, 3110, y_lo=90, y_hi=140)
    return _under(base, sc.rgba(lay.canvas()))


@lru_cache(maxsize=1)
def _igloo_cx():
    return _film(sc._igloo_c, {340: IGL_CW, 320: 494}, IGLOOS=IGLOOS_X, SNOWLAMPS=SNOWLAMPS_X)()


@lru_cache(maxsize=1)
def _fg_cx():
    return _film(sc._fg_c, {360: FG_CW, _BOULDERS_C: _BOULDERS_CX})()


# ============================================================================= landmarks, hotspots, stamps
LANDMARKS_A = [
    Landmark('YAKYAKISTAN', 70, (sc.TOWER_A[0] + 4, sc.TOWER_A[1] + 24, 'right', 12), p=sc.P_VIL),
    Landmark('LOG GATES', 214, (GX + 5, GY - 14, 'down', 22), p=sc.P_WALL),
]
LANDMARKS_B = [
    Landmark('GREAT HALL', HALL_APEX[0], (HALL_APEX[0] - 20, HALL_APEX[1] + 22, 'up', 30)),
    Landmark('ICE SLIDE', int((SLIDE_TAG[0] - 24) / 0.85), (SLIDE_TAG[0], SLIDE_TAG[1], 'up', 34), p=sc.P_BSLOPE),
]
LANDMARKS_C = [
    Landmark('MT. EVERHOOF', sc.EVER[0], (sc.EVER[0] + 5, sc.EVER[1] + 12, 'right', 30), p=sc.P_CEVER),
]
HOT_A = [Hotspot('horn', GX, 48, 'horn', r=46, cooldown=HORN_T + 10, p=sc.P_WALL)]
HOT_B = [Hotspot('gong', HALL_APEX[0] + 30, 90, 'bell', cooldown=GONG_T),
         Hotspot('sled', 262, 136, 'wind', cooldown=SLED_T, p=sc.P_BSLOPE)]
HOT_C = [Hotspot('cave', sc.CAVE[0] + 2, sc.CAVE[1] - 34, 'gem', cooldown=CAVE_T, p=sc.P_CMID),
         Hotspot('aurora', 362, 46, 'moon', cooldown=AURORA_T)]
STAMPS_A = [Stamp('pines', CAM_A + 298, 140)]
STAMPS_B = [Stamp('slide', 446, 160)]
STAMPS_C = [Stamp('summit', sc.EVER[0], 14)]


class Yakyakistan(Region):
    SEG = 'yakyakistan'
    CARD = ('YAKYAKISTAN', ['LOG GATES - ICE SLIDES', 'TOP OF THE WORLD'])

    def build_areas(self):
        return [
            Area(self.render_gates, cam_x=(0, CAM_A), y_range=(30, 156), start=(112, 64),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=TINT_MOON, hotspots=HOT_A, stamps=STAMPS_A,
                 sky=self.sky_gates),
            Area(self.render_inside, cam_x=(0, CAM_B), y_range=(28, 174), start=(36, 70),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=TINT_MOON, hotspots=HOT_B, stamps=STAMPS_B,
                 sky=self.sky_inside),
            Area(self.render_north, cam_x=(0, CAM_C), cam_y=(0, TILT_C), y_range=(26, 150), start=(44, 132),
                 landmarks=LANDMARKS_C, balloon='m', balloon_tint=TINT_MOON, hotspots=HOT_C, stamps=STAMPS_C,
                 sky=self.sky_north),
        ]

    # -------------------------------------------------------------- state and clocks
    def init_state(self):
        self.horn_t0 = None         # step of the last horn blast
        self.wind_bank = 0.0        # sideways snow displacement banked from earlier blasts

    def on_interact(self, key, t):
        g = self.g
        if key == 'horn':
            if self.horn_t0 is not None:
                self.wind_bank += _blow(t - self.horn_t0)
            self.horn_t0 = t
            g.sfx('horn', 0.7)
        elif key == 'gong':
            g.sfx('gong', 0.6)
        elif key == 'sled':
            g.sfx('whoosh', 0.45)
        elif key == 'cave':
            g.sfx('crystal', 0.6)
        elif key == 'aurora':
            g.sfx('shimmer', 0.55)

    def since(self, key, t):
        t0 = self.spot_t.get(key)
        return -1 if t0 is None else t - t0

    def update(self, t):
        g = self.g
        a = self.since('horn', t) if self.ai == 0 else -1
        if a == 6:
            g.sfx('wind', 0.45, 1)
        elif a == 16:
            g.sfx('avalanche', 0.18)                     # the palisade sheds its snow
        a = self.since('gong', t) if self.ai == 1 else -1
        if a == 6:
            g.sfx('rumble', 0.5)
        elif a == 20:
            g.sfx('avalanche', 0.9 * max(0.4, self.near_gain(HALL_APEX[0])))
        elif a == 25:
            g.sfx('thud', 0.6 * max(0.4, self.near_gain(HALL_APEX[0])))
        a = self.since('sled', t) if self.ai == 1 else -1
        if a in (14, 30):
            g.sfx('sweep', 0.25)
        elif a == 44:
            g.sfx('sparkle', 0.3)
        a = self.since('cave', t) if self.ai == 2 else -1
        if a == 12:
            g.sfx('shimmer', 0.3)
        k = (t - self.t_enter) % 420
        if k == 6:
            g.sfx('wind', 0.3, 1)
        elif k == 210:
            g.sfx('wind', 0.22, 0)
        elif k == 120 and self.ai == 0 and self.near_gain(GX) > 0.6:
            g.sfx('creak', 0.25)                         # the gate timbers groan in the gust
        elif k == 300 and self.ai == 2 and self.cx < 120:
            g.sfx('crystal', 0.2)
        self._tint()

    def _tint(self):
        """Moonlit; warm beside a fire or lantern; icy blue in the cave's light."""
        A = self.area
        sx, sy = self.bx - self.cx, self.by - 10 - self.cy
        near, ice = 0.0, 0.0
        if self.ai == 0:
            for (x, y, kind) in LIGHTS_AX:
                lx = x - self.cx * sc.P_WALL
                ly = GY - 18 if kind == 'brazier' else y
                near = max(near, clamp(1 - math.hypot(sx - lx, (sy - ly) * 1.2) / 44))
        elif self.ai == 1:
            for (lx, ly) in ((HALL_APEX[0] - 12, sc.GB - 25), (HALL_APEX[0] + 13, sc.GB - 25),
                             (185, sc.GB - 12), (236, sc.GB - 12)):
                near = max(near, clamp(1 - math.hypot(sx - (lx - self.cx), (sy - ly) * 1.2) / 40))
        else:
            cx_ = sc.CAVE[0] - self.cx * sc.P_CMID
            cy_ = sc.CAVE[1] - 9 - self.cy * sc.P_CMID
            ice = clamp(1 - math.hypot(sx - cx_, sy - cy_) / 60)
            if 0 <= self.since('cave', self.g.t) < CAVE_T:
                ice = min(1.0, ice * 1.6 + 0.2)
        if ice > near:
            A.balloon_tint = _tint_between(TINT_MOON, TINT_ICE, ice)
        else:
            A.balloon_tint = _tint_between(TINT_MOON, TINT_FIRE, near)

    # ============================================================== area 0: the log gates
    # the skies alone, as the renders draw them first (game/daylight.py repaints them for the clock);
    # every scenery layer is stamped out in magenta: at night the dark ranges and pines are close to
    # the night sky's colours and would otherwise be repainted as sky
    def sky_gates(self, f, cx, cy):
        c = sc._sky_a().copy()
        sc.stars(c, f, 1, 150, 110, avoid=((sc.MOON_A[0], sc.MOON_A[1], 22),))
        sc.aurora(c, f, 46, 50, k=0.75, seed=5)
        np.copyto(c.a, _MAGENTA, where=_scenery_gates(cx)[..., None])
        return c

    def sky_inside(self, f, cx, cy):
        c = sc._sky_b().copy()
        sc.stars(c, f, 2, 160, 110, avoid=((sc.MOON_B[0], sc.MOON_B[1], 20),))
        sc.aurora(c, f, 64, 44, k=0.55, seed=8, x_shift=120)
        sway = SWAY / 2 * (1 - math.cos(2 * math.pi * f / 2400.0))
        m = _scenery_inside(cx).copy()
        _or_shift(m, 'cloudsea', -int(round(cx * sc.P_BCLOUD + sway)), 0)
        np.copyto(c.a, _MAGENTA, where=m[..., None])
        return c

    def sky_north(self, f, cx, cy):
        c = Canvas(W, H)
        oy0 = -int(round(cy * sc.P_CSKY))
        c.blit(sc._sky_c(), 0, oy0)
        sc.stars(c, f, 3, 230, 120, avoid=((sc.MOON_C[0], sc.MOON_C[1] + oy0, 20),), dy=oy0)
        sc.aurora(c, f, 64 + oy0, 56, k=1.1, seed=11, x_shift=60)
        np.copyto(c.a, _MAGENTA, where=_scenery_north(cx, cy)[..., None])
        return c

    def render_gates(self, f, cx, cy):
        t = self.g.t
        cam = cx
        a = self.since('horn', t)
        e = 0.0
        if 0 <= a < HORN_T:
            e = clamp(a / 6.0) * (1 - smooth((a - 36) / 50.0))
        hw = _windy(e)
        blown = self.wind_bank + (_blow(t - self.horn_t0) if self.horn_t0 is not None else 0.0)
        snow = sc.snowfall
        if blown:
            snow = rebind(sc.snowfall, wind_x=lambda f_: sc.wind_x(f_) + blown)
        lk = _lamp_k(self.g.hour)                         # the clock's lights (see above)
        q = int(round(lk * 16))
        c = sc._sky_a().copy()
        sc.stars(c, f, 1, 150, 110, avoid=((sc.MOON_A[0], sc.MOON_A[1], 22),))
        sc.aurora(c, f, 46, 50, k=0.75, seed=5)
        c.blit(_far_ax(), -int(round(cam * sc.P_FAR)), 0)
        snow(c, f, 70, 101, vy=0.35, wk=0.5, color='#6e77b6')
        c.blit(_mid_ax(), -int(round(cam * sc.P_MID)), 0)
        ov = -int(round(cam * sc.P_VIL))
        _vil, chim = _village_ax()
        c.blit(_lit('vil_a', q), ov, 0)
        for i, (x, y) in enumerate(chim):
            if _vis(x + ov, 30):
                hw['smoke'](c, x + ov, y - 1, f, 40 + i, n=8, h=24)
        tx, ty = sc.TOWER_A
        if _vis(tx + ov):
            c.vline(tx + ov, ty - 9, ty - 3, '#2b2a40')
            hw['pennant'](c, tx + ov, ty - 9, f, 15, length=9)
            lv = _lv(lk, 1, 1)
            if lv > 0:
                sc.glow_a(c.a, tx + ov, ty + 11, 9, 'amber', 0.16 * sc.flicker(f, 3) * lv, rings=3)
        snow(c, f, 80, 102, vy=0.5, wk=0.8, color='#a3b2e4')
        ow = -int(round(cam * sc.P_WALL))
        c.blit(_lit('wall_a', q), ow, 0)
        for (bx, by, s, sch, w) in BANNERS_A:
            if _vis(bx + ow, w):
                hw['banner'](c, bx + ow, by, f, s, w=w, h=22, scheme=sch)
        for (px, s, L) in PENNANTS_A:
            x = px + ow
            if _vis(x, L):
                c.vline(x, 76, 99, '#1c1a2b')
                c.vline(x + 1, 78, 99, '#2b2a40')
                c.px(x, 75, '#e0b443')
                hw['pennant'](c, x, 76, f, s, length=L)
        for (x, y, kind) in LIGHTS_AX:
            s = x * 7 + y
            lx = x + ow
            if not _vis(lx, 0, 30):
                continue
            lv = _lv(lk, x, y)                            # this fire's own point on the clock's ramp
            if kind == 'torch':
                if lv > 0:
                    hw['wall_torch'](c, lx, y, f, s)
                    sc.torch_light(c, lx, y - 3, f, s, r=14 + int(8 * e), k=(0.22 + 0.25 * e) * lv)
                else:
                    _cold_torch(c, lx, y, f, s)
            elif kind == 'brazier':
                if lv > 0:
                    hw['brazier'](c, lx, GY + 2, f, s)
                    sc.torch_light(c, lx, y - 2, f, s, r=18 + int(10 * e), k=(0.25 + 0.3 * e) * lv)
                else:
                    _cold_brazier(c, lx, GY + 2, f, s)
            else:
                c.vline(lx, 82, 84, '#1c1a2b')
                c.rect(lx - 1, 85, 3, 4, '#2b2a40')
                if lv > 0:
                    c.px(lx, 86, 'butter' if sc.flicker(f, s) > 0.8 else 'gold')
                    c.px(lx, 87, 'gold')
                    sc.torch_light(c, lx, 87, f, s, r=8 + int(4 * e), k=(0.2 + 0.2 * e) * lv)
                else:
                    _cold_lantern_px(c, lx, 86)
        lv = _lv(lk, 2, 1)
        if lv > 0:
            sc.glow_a(c.a, GX + ow, 126, 8, 'amber', 0.1 * sc.flicker(f, 9) * lv, rings=3, sx=0.5)
        for i in range(26):                              # moonlit glints on the snow
            x = int(hash01(i, 71) * WALL_AW) + ow
            y = GY + 10 + int(hash01(i, 72) * 12)
            if (f // 3 + i * 5) % 26 < 3 and 0 <= x < W:
                c.px(x, y, 'white')
        if 0 <= a < HORN_T:
            self._horn_fx(c, f, t, a, ow, lk)
        snow(c, f, 60, 103, vy=0.8, wk=1.1, color='#e4ebff', size=2, alt='#b8c6f1')
        c.blit(_fg_ax(), -int(round(cam * sc.P_FG)), 0)
        snow(c, f, 26, 104, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
        self._pre = c.a.copy()                           # (for after_light)
        return c

    def _horn_fx(self, c, f, t, a, ow, lk=1.0):
        """The horn blast: sound rings from both horn carvings, the palisade shedding its snow (a wave
        running out from the gate), embers roaring up from the braziers (the lit ones: lk, the clock's
        lights)."""
        for (hx, hy) in HORN_XY:                         # the blast rolls out of both horns
            if a < 16:
                sc.glow_a(c.a, hx + ow, hy, 14, '#f7f4ff', 0.4 * (1 - a / 16.0), rings=3)
            for k in range(2):
                r = (a - k * 9) * 1.3 + 6
                if 6 <= r < 42:
                    lv = 0.9 - 0.8 * clamp((r - 14) / 28.0)
                    _ring(c, hx + ow, hy, r, '#f7f4ff' if k == 0 else '#aeabcb', lv, sy=0.75)
        for j in range(-22, 23):                         # powder pouring off the palisade tips
            wx = GX + j * 12 + (j % 2) * 3
            x = wx + ow
            if not _vis(x, 0, 10):
                continue
            if abs(j) <= 3:
                y0 = sc.POST_TOP - 6 if abs(j) >= 2 else sc.LINTEL_Y - 2
            else:
                y0 = 94 + int(hash01(j, 5) * 4)
            b = a - 8 - abs(j) * 1.4
            if not 0 <= b < 44:
                continue
            fade = 1.0 if b < 22 else (44 - b) / 22.0
            head = int(min(GY - 2, y0 + 0.1 * b * b))
            tail = int(min(GY - 2, y0 + 0.1 * max(0.0, b - 12) ** 2))
            if head - tail >= 2:
                _powder(c, x + int(b * 0.3), tail, head, fade)
            if y0 + 0.1 * b * b >= GY - 2:               # landing billow
                k = clamp((y0 + 0.1 * b * b - (GY - 2)) / 40.0)
                sc.puff(c, x + int(b * 0.3), GY - 2, 2 + 5 * k, fade * (1 - k) * 0.85)
            for q in range(2):
                py = int(y0 + 0.13 * b * b * (0.7 + 0.3 * hash01(j, q, 4)))
                if py < GY and (int(b) + q) % 3:
                    c.px(x + int((hash01(j, q, 3) - 0.5) * 8 + b * 0.3), py, 'white')
        for (x, y, kind) in LIGHTS_AX:                   # the fires roar up
            if kind == 'brazier' and _vis(x + ow) and _lv(lk, x, y) > 0:
                ui.burst(c, t, self.spot_t['horn'] + 2, x + ow, GY - 14, ('gold', 'amber', 'orange', 'butter'),
                         n=22, seed=x, power=0.7, life=40, gravity=-0.02)

    # ============================================================== area 1: inside the walls
    def render_inside(self, f, cx, cy):
        t = self.g.t
        cam = cx
        ga = self.since('gong', t)
        af = None
        if 0 <= ga < REFILL[0]:
            af = AVA - 20 + ga                           # crack from +6, slide on +20
        sx_, sy_ = 0, 0                                   # (the avalanche no longer shakes the village:
        # screen shake under the player's balloon is uncomfortable to watch; the snow slide tells it)
        hour = self.g.hour
        lk = _lamp_k(hour)                                # the clock's lights (see above)
        c = sc._sky_b().copy()
        sc.stars(c, f, 2, 160, 110, avoid=((sc.MOON_B[0], sc.MOON_B[1], 20),))
        sc.aurora(c, f, 64, 44, k=0.55, seed=8, x_shift=120)
        fo = -int(round(cam * sc.P_BFAR))
        c.blit(sc._far_b(), fo, 0)
        ce = (300 + fo, 112)                              # the Crystal Empire's light far away (at night)
        if lk > 0:
            sc.glow_a(c.a, ce[0], ce[1], 9, '#f39ce8', (0.25 + 0.08 * math.sin(f * 0.2)) * lk, rings=3)
            if lk > 0.5:
                fx.sparkle(c, ce[0], ce[1] - 2, f, '#b4e6ff', period=30, size=2)
        sway = SWAY / 2 * (1 - math.cos(2 * math.pi * f / 2400.0))
        c.blit(sc._cloudsea(), -int(round(cam * sc.P_BCLOUD + sway)), 0)
        mk = clamp((daylight.night(hour) - 0.5) / 0.3)  # the moon's reflection on the cloud sea, at night,
        if mk > 0:                                      # under the moon that shows: the painted one while
            mx = sc.MOON_B[0]                           # the painted sky is kept, else the clock's own
            if daylight.sky_weight(self.HOUR, hour) >= 0.5:
                kind, mx, _my = daylight._body_pos(hour)
                mx = mx if kind == 'moon' else None
            if mx is not None:
                sc.glow_a(c.a, mx, 126, 16, '#c9d0f2', 0.18 * mk, rings=4, sx=0.35)
        sc.snowfall(c, f, 60, 201, vy=0.35, wk=0.5, color='#6e77b6')
        os_ = -int(round(cam * sc.P_BSLOPE)) + sx_
        c.blit(_slope_bx(), os_, sy_)
        sa = self.since('sled', t)
        s_sled = _sled_s(sa) if 0 <= sa < SLED_T else None
        Lt = SLIDE_L
        for k in range(3):                               # moonlight glints running down the ice
            pos = (f * 2.2 + k * Lt / 3) % Lt
            i = min(int(np.searchsorted(_CUM, pos)), len(_LINE) - 1)
            gx_, gy_ = int(_LINE[i][0]) + os_, int(_LINE[i][1]) + sy_
            c.px(gx_, gy_, 'white')
            c.px(gx_ - 1, gy_, '#d4f2ff')
        for j, idx in enumerate((10, 40, 70, 100, 130)):  # lanterns along the slide
            if idx >= len(_LINE):
                continue
            lx, ly = int(_LINE[idx][0]) + os_ + 5, int(_LINE[idx][1]) + sy_ - 2
            if not _vis(lx):
                continue
            boost = 0.0
            if s_sled is not None:
                boost = clamp(1 - abs(_CUM[idx] - s_sled) / 40.0) + clamp(1 - abs(_CUM[idx] - s_sled + 30) / 90.0) * 0.4
            c.vline(lx, ly - 4, ly + 2, '#2b2a40')
            lv = max(_lv(lk, 3, j), min(1.0, boost))  # lit by the clock (flaring as the sled passes)
            if lv <= 0:
                _cold_lantern_px(c, lx, ly - 5)
                continue
            c.px(lx, ly - 5, 'butter' if sc.flicker(f, j) > 0.8 or boost > 0.5 else 'gold')
            sc.glow_a(c.a, lx, ly - 5, 7 + int(6 * boost), 'amber', (0.25 * sc.flicker(f, j) + 0.3 * boost) * lv,
                      rings=3)
        if s_sled is not None:
            self._sled_fx(c, f, sa, s_sled, os_, sy_)
        ov = -cam + sx_
        _vil, info = sc._village_b()
        c.blit(_lit('vil_b', int(round(lk * 16))), ov, sy_)
        for i, key in enumerate(('hutL', 'hutR')):
            ch = info[key]['chimney']
            if _vis(ch[0] + ov, 30):
                sc.smoke(c, ch[0] + ov, ch[1] + sy_, f, 70 + i, n=8, h=26)
        hc = info['hall_chim']
        if _vis(hc[0] + ov, 34):
            sc.smoke(c, hc[0] + ov, hc[1] + sy_, f, 77, n=9, h=32)
        ax, ay = info['hall_apex']
        flare = 0.0
        if 0 <= ga < 40:
            flare = 1 - ga / 40.0
        for (tx_, s) in ((ax - 12, 1), (ax + 13, 2)):
            if _vis(tx_ + ov):
                lv = _lv(lk, 4, s)
                if lv > 0:
                    sc.wall_torch(c, tx_ + ov, sc.GB - 22 + sy_, f, 60 + s)
                    sc.torch_light(c, tx_ + ov, sc.GB - 25 + sy_, f, 60 + s, r=14 + int(8 * flare),
                                   k=(0.22 + 0.25 * flare) * lv)
                else:
                    _cold_torch(c, tx_ + ov, sc.GB - 22 + sy_, f, 60 + s)
        for j, lx in enumerate((182, 233)):
            x_ = lx + 3 + ov
            if _vis(x_):
                c.vline(x_, sc.GB - 15 + sy_, sc.GB - 14 + sy_, '#1c1a2b')
                c.rect(x_ - 1, sc.GB - 13 + sy_, 3, 3, '#2b2a40')
                lv = _lv(lk, 5, j)
                if lv > 0:
                    c.px(x_, sc.GB - 12 + sy_, 'butter' if sc.flicker(f, j + 5) > 0.8 else 'gold')
                    sc.glow_a(c.a, x_, sc.GB - 12 + sy_, 8, 'amber', 0.2 * sc.flicker(f, j + 5) * lv, rings=3)
                else:
                    c.px(x_, sc.GB - 12 + sy_, '#3c4660')
        for (x, y, w, h, sch) in sc.RUGS_B:
            if _vis(x + ov, w):
                sc.rug(c, x + ov, y + sy_, f, x, w, h, sch)
        if _vis(243 + ov, 20):                           # flags on the slide arch
            sc.pennant(c, 237 + ov, sc.GB - 25 + sy_, f, 31, length=8)
            sc.pennant(c, 249 + ov, sc.GB - 25 + sy_, f, 32, length=8, cols=('#3f73cc', '#e0b443', '#c23f55'))
            c.vline(237 + ov, sc.GB - 25 + sy_, sc.GB - 21 + sy_, '#2b2a40')
            c.vline(249 + ov, sc.GB - 25 + sy_, sc.GB - 21 + sy_, '#2b2a40')
        if _vis(ax + ov, 80):
            self._avalanche(c, f, ga, af, ov, sy_)
        sc.snowfall(c, f, 70, 202, vy=0.6, wk=0.9, color='#b8c6f1')
        for i in range(14):
            x = int(hash01(i, 171) * 230) + ov
            y = sc.GB + 4 + int(hash01(i, 172) * 18)
            if (f // 3 + i * 5) % 26 < 3 and 0 <= x < W:
                c.px(x, y, 'white')
        c.blit(sc._fg_b(), -int(round(cam * sc.P_BFG)) + sx_, sy_)
        sc.snowfall(c, f, 26, 204, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
        self._pre = c.a.copy()                           # (for after_light)
        return c

    def _avalanche(self, c, f, ga, af, ov, oy):
        """The roof slab: resting, the film's avalanche (crack, slide, plume, pour-off, billow, mound),
        then the roof snowing over again while the mound settles away."""
        if 0 <= ga < 36:                                 # the gong's boom rolls out of the hall
            ax = HALL_APEX[0] + ov
            k0 = 1 - ga / 36.0
            sc.glow_a(c.a, ax, HALL_APEX[1] + 21 + oy, 14, '#ffd16a', 0.5 * k0, rings=3)   # the oculus flares
            sc.glow_a(c.a, ax, sc.GB - 14 + oy, 26, 'amber', 0.3 * k0, rings=4)
            for k in range(3):
                r = (ga - k * 6) * 1.7 + 6
                if 6 <= r < 56:
                    lv = 0.95 - 0.85 * clamp((r - 16) / 40.0)
                    _ring(c, ax, sc.GB - 14 + oy, r, '#ffd16a' if k == 0 else '#f7f4ff', lv, sy=0.8)
        if af is not None:
            sc.draw_avalanche(c, af, ov, oy)
            return
        k = 1.0
        if 0 <= ga < REFILL[1]:
            k = clamp((ga - REFILL[0]) / float(REFILL[1] - REFILL[0]))
        spr, x0, y0, ux, uy, ex, ey, p0, L = sc._slab_b()
        sx, sy = x0 + ov, y0 + oy
        cut = ex + ov + 3 - sx
        if cut > 0:
            arr = spr[:, :min(spr.shape[1], cut)]
            if k < 1:
                arr = arr.copy()
                thr = bayer(arr.shape[0], arr.shape[1], sx, sy)
                arr[..., 3] = np.where(thr < k, arr[..., 3], 0)
            c.blit(arr, sx, sy)
        if k < 1:                                        # the landed mound settles away
            lv = 1 - k
            mh = 6
            mx, gy = ex + ov + 5, sc.GB + oy
            sc.dith_ell(c, mx, gy - 1, 8 + mh // 2, mh, '#6a7cc2', lv)
            sc.dith_ell(c, mx - 1, gy - 2, 7 + mh // 2, max(1, mh - 1), '#b8c6f1', lv)
            sc.dith_ell(c, mx - 2, gy - 2 - mh // 2, 4, max(1, mh // 2), '#e4ebff', lv)

    def _sled_fx(self, c, f, a, s, ox, oy):
        """The sled racing down the slide, spraying ice; a puff where it was pushed off."""
        if a < 14:
            x0, y0, _ = _sled_at(0)
            sc.puff(c, x0 + ox + 1, y0 + oy - 3 - a * 0.3, 3 + a * 0.3, 1 - a / 14.0)
        trail = 64
        for q in range(0, trail, 2):                     # the ice glows in the sled's wake
            sb = s - 4 - q
            if 0 < sb < SLIDE_L:
                px, py, _ = _sled_at(sb)
                u = q / trail
                if hash01(q, f // 2, 7) > u:
                    c.px(px + ox, py + oy, 'white' if u < 0.35 else '#9ad6f6')
                    c.px(px + ox + 1, py + oy, '#d4f2ff' if u < 0.5 else '#62aee6')
        if s >= SLIDE_L:
            return
        x, y, d = _sled_at(s)
        sp = 0.4 + 0.15 * a                              # speed (px / step): the spray grows with it
        for q in range(14):                              # ice spray thrown up behind the runners
            sb = s - 2 - q * 1.6
            if sb <= 0:
                continue
            px, py, dq = _sled_at(sb)
            jx = int((hash01(q, f // 2, 1) - 0.5) * 3) - dq * int(q * 0.4)
            jy = -int(hash01(q, f // 2, 2) * (1 + q * 0.25) * min(2.5, sp * 0.5))
            if (q + f) % 3:
                cc = 'white' if q < 5 else '#b4e6ff'
                c.px(px + ox + jx, py + oy - 1 + jy, cc)
                if q < 6:
                    c.px(px + ox + jx + 1, py + oy - 1 + jy, cc)
        for q in range(3):                               # powder kicked up in its wake
            sb = s - 6 - q * 7
            if sb > 0:
                px, py, dq = _sled_at(sb)
                sc.puff(c, px + ox - dq * 2, py + oy - 2 - q * 1.5, 1.6 + q * 1.2 + min(2.0, sp * 0.2),
                        (0.75 - q * 0.22) * min(1.0, sp / 3.0), body='#d6def8', sh='#8f9dd6', hi='#ffffff')
        sc.glow_a(c.a, x + ox, y + oy - 3, 12, '#9ad6f6', 0.35, rings=3)
        draw_sled(c, x + ox, y + oy - 1, d)
        if a % 6 < 3:
            fx.sparkle(c, x + ox - d * 8, y + oy - 6, a, 'white', period=6, size=2)

    # ============================================================== area 2: the frozen north
    def render_north(self, f, cx, cy):
        t = self.g.t

        def off(p):
            return -int(round(cx * p)), -int(round(cy * p))

        lk = _lamp_k(self.g.hour)                         # the clock's lights (see above)
        c = Canvas(W, H)
        oy0 = off(sc.P_CSKY)[1]
        c.blit(sc._sky_c(), 0, oy0)
        sc.stars(c, f, 3, 230, 120, avoid=((sc.MOON_C[0], sc.MOON_C[1] + oy0, 20),), dy=oy0)
        ua = self.since('aurora', t)
        surge = 0.0
        if 0 <= ua < AURORA_T:
            surge = smooth(ua / 20.0) * (1 - smooth((ua - 90) / 60.0))
        painted_sky = daylight.sky_weight(self.HOUR, self.g.hour) < 0.5
        sc.aurora(c, f, 64 + oy0, 56, k=1.1 + 0.8 * surge * painted_sky, seed=11, x_shift=60)
        if surge > 0.01 and painted_sky:                 # great curtains surging across the whole sky (on
            self._surge(c, f, ua, surge, oy0)            # the clock's own sky: after_light)
        fo = off(sc.P_CFAR)
        c.blit(_far_cx(), *fo)
        eo = off(sc.P_CEVER)
        c.blit(_ever_cx(), *eo)
        sc.plume(c, f, sc.EVER[0] + eo[0], sc.EVER[1] + eo[1])
        sc.snowfall(c, f, 70, 301, vy=0.35, wk=0.5, color='#6e77b6')
        mo = off(sc.P_CMID)
        c.blit(_mid_cx(), *mo)
        if _vis(sc.CAVE[0] + mo[0], 0, 40):
            _cave_light(int(round(lk * 8)))(c, f, *mo)    # the cave's glow at night, its crystals always
            ca = self.since('cave', t)
            if 0 <= ca < CAVE_T:
                self._cave_fx(c, f, t, ca, mo)
        sc.snowfall(c, f, 70, 302, vy=0.5, wk=0.8, color='#a3b2e4')
        io = off(sc.P_CIGL)
        _ig, doors = _igloo_cx()
        c.blit(_lit('igl_c', int(round(lk * 16))), *io)
        for i, (dx, dy) in enumerate(doors):
            lv = _lv(lk, 6, i)
            if lv > 0 and _vis(dx + io[0]):
                sc.torch_light(c, dx + io[0], dy + io[1] - 1, f, 200 + i, r=7, k=0.16 * lv)
        for i, (x, base, r, h) in enumerate(IGLOOS_X):
            if r >= 10 and _vis(x + io[0], 20):
                sc.smoke(c, x + io[0] + 1, base - h - 1 + io[1], f, 210 + i, n=6, h=16, period=60)
        for i, (lx, ly) in enumerate(SNOWLAMPS_X):
            lv = _lv(lk, 7, i)
            if lv > 0 and _vis(lx + io[0]):
                c.px(lx + io[0], ly - 7 + io[1], 'butter' if sc.flicker(f, 220 + i) > 0.8 else 'gold')
                sc.glow_a(c.a, lx + io[0], ly - 6 + io[1], 6, 'amber', 0.18 * sc.flicker(f, 220 + i) * lv, rings=2)
        sc.snowfall(c, f, 60, 303, vy=0.8, wk=1.1, color='#e4ebff', size=2, alt='#b8c6f1')
        go = off(sc.P_CFG)
        fg, crests = _fg_cx()
        c.blit(fg, *go)
        sc.spindrift(c, f, [p for p in crests if _vis(p[0] + go[0], 20, 4)], *go)
        for (mx, my, s) in POLES_C:
            x = mx + go[0]
            if not _vis(x, 14):
                continue
            if mx != 84:                                 # the film's pole is baked into its drift layer
                y = my + go[1]
                c.vline(x, y - 26, y, '#2b1e2c')
                c.vline(x + 1, y - 26, y, '#5e3e3c')
                for k, cc in enumerate(('#3f73cc', '#e0b443', '#c23f55', '#3f73cc')):
                    c.rect(x, y - 22 + k * 3, 2, 2, cc)
                c.hline(x - 1, x + 2, y - 27, '#e4ebff')
            sc.pennant(c, x + 1, my - 26 + go[1], f, s, length=12)
        sc.snowfall(c, f, 26, 304, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
        self._pre = c.a.copy()                           # (for after_light)
        return c

    def _surge(self, c, f, ua, surge, oy0, k=1.0):
        """The aurora the player calls up: great curtains sweeping across the whole sky."""
        sweep = 160 * smooth(ua / float(AURORA_T))
        _aurora_surge(c, f, 8 + oy0, 96, k=2.1 * surge * k, seed=17, x_shift=sweep)
        _aurora_surge(c, f + 37, 30 + oy0, 70, k=1.4 * surge * k, seed=23, x_shift=200 - sweep * 0.7)

    # ============================================================== the clock's light, after the relight
    _pre = None

    def after_light(self, c, t):
        """Right after the clock's light (and sky): the relight's day lamp switch-off undone (the lights
        here are switched by the clock already, so what it caught are painted trims, glow specks and the
        gold of banners: they get the plain relight's colours), and the aurora the player calls up drawn
        on the clock's own sky when that has replaced the painted night (fainter by day)."""
        pre, self._pre = self._pre, None
        if pre is None or pre.shape != c.a.shape:
            return
        hour = self.g.hour
        memo = self._sky_memo                            # this frame's open sky (Region._world), if any
        sky = memo[1] if memo is not None and memo[0][:2] == (self.ai, t) else None
        fq, tq = daylight._q(self.HOUR), daylight._q(hour)
        off = daylight.night(self.HOUR) - daylight.night(hour)
        if off > 0.3 and abs(fq - tq) >= 0.05:
            # every pixel its lamp test could catch (the warm test alone: never snow, so the day snow left
            # it alone) gets the relight's plain colour: exact for those it did not catch, too
            p = pre.astype(np.int16)
            r_, g_, b_ = p[..., 0], p[..., 1], p[..., 2]
            m = (r_ > 200) & (g_ > 140) & (b_ < 175) & (r_ > b_ + 55)
            if sky is not None:
                m &= ~sky
            if m.any():
                lut = daylight._tables(fq, tq, True)[0]
                px = pre[m]
                c.a[m] = np.stack([lut[ch].take(px[:, ch]) for ch in range(3)], axis=1)
        if self.ai == 2 and sky is not None:
            ua = self.since('aurora', t)
            if 0 <= ua < AURORA_T and daylight.sky_weight(self.HOUR, hour) >= 0.5:
                surge = smooth(ua / 20.0) * (1 - smooth((ua - 90) / 60.0))
                if surge > 0.01:
                    tmp = Canvas(arr=c.a.copy())
                    self._surge(tmp, self.scene_f(t), ua, surge, -int(round(memo[0][3] * sc.P_CSKY)),
                                0.45 + 0.55 * daylight.night(hour))
                    np.copyto(c.a, tmp.a, where=sky[..., None])

    def _cave_fx(self, c, f, t, a, mo):
        """The cave's crystals wake: a blaze of blue light, a fan of light on the snow, shards flying."""
        x, y = sc.CAVE[0] + mo[0], sc.CAVE[1] - 9 + mo[1]
        k = smooth(a / 6.0) * (1 - smooth((a - 34) / 56.0))
        base = sc.CAVE[1] + mo[1]
        # the whole ice cliff lights up from inside, then the mouth blazes
        sc.glow_a(c.a, x + 4, y - 6, 62, '#2c5aa8', 0.55 * k, rings=6, sx=1.3)
        sc.glow_a(c.a, x, y, 30 + a // 4, '#62aee6', 0.6 * k, rings=5)
        sc.glow_a(c.a, x, y + 1, 13, '#ffffff', 0.7 * k, rings=3)
        for j in range(5):                               # shafts of light fanning out of the mouth
            ang = -math.pi / 2 + (j - 2) * 0.42 + 0.06 * math.sin(a * 0.15 + j)
            L = (22 + 26 * k) * (0.8 + 0.3 * hash01(j, 9))
            ex, ey = x + math.cos(ang) * L, y + math.sin(ang) * L
            for q in range(int(L)):
                u = q / L
                if hash01(j, q, a // 2) < k * (1 - u) * 0.9:
                    c.px(int(x + (ex - x) * u), int(y + (ey - y) * u), 'white' if u < 0.4 else '#9ad6f6')
        sc.glow_a(c.a, x + 6, base + 5, 34, '#62aee6', 0.4 * k, rings=4, sx=2.2)
        t0 = self.spot_t['cave']
        ui.burst(c, t, t0 + 2, x, y - 2, ('white', '#b4e6ff', '#8ad6f6', '#f39ce8'), n=40, seed=7, power=1.3,
                 life=56, gravity=0.04)
        ui.burst(c, t, t0 + 10, x, y - 4, ('white', '#d4f2ff', '#62aee6'), n=24, seed=8, power=0.9, life=44,
                 gravity=0.02)
        for j in range(14):                              # motes rising out of the mouth
            ph = (a * 1.3 + j * 7) % 70
            if ph < a + 1 and a < CAVE_T - 6:
                px = x + int(math.sin(j * 1.7 + a * 0.08) * (5 + j * 1.2))
                py = y - int(ph * 0.9)
                c.px(px, py, 'white' if j % 2 else '#b4e6ff')
                if j % 3 == 0:
                    c.px(px + 1, py, '#8ad6f6')


@lru_cache(maxsize=None)
def _opaque(key, pad):
    """A scenery layer's opaque-pixel mask (grown by pad px: layers that shake when the gong rings)."""
    rgba = {'far_ax': _far_ax, 'mid_ax': _mid_ax, 'vil_ax': lambda: _village_ax()[0], 'wall_ax': _wall_ax,
            'fg_ax': _fg_ax, 'far_b': sc._far_b, 'cloudsea': sc._cloudsea, 'slope_bx': _slope_bx,
            'vil_b': lambda: sc._village_b()[0], 'fg_b': sc._fg_b, 'far_cx': _far_cx, 'ever_cx': _ever_cx,
            'mid_cx': _mid_cx, 'igl_cx': lambda: _igloo_cx()[0], 'fg_cx': lambda: _fg_cx()[0]}[key]()
    m = rgba[..., 3] > 0
    if pad:
        from scipy import ndimage
        m = ndimage.binary_dilation(m, iterations=pad)
    return m


def _scenery(parts):
    """Union mask (H x W) of scenery layers [(key, x, y, pad)] at their screen offsets."""
    out = np.zeros((H, W), bool)
    for key, x, y, pad in parts:
        m = _opaque(key, pad)
        h, w = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] |= m[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


# the scenery masks of the sky hooks, per camera (cached: the hooks cost ~0.1 ms while flying)
@lru_cache(maxsize=16)
def _scenery_gates(cx):
    return _scenery([('far_ax', -int(round(cx * sc.P_FAR)), 0, 0), ('mid_ax', -int(round(cx * sc.P_MID)), 0, 0),
                     ('vil_ax', -int(round(cx * sc.P_VIL)), 0, 0), ('wall_ax', -int(round(cx * sc.P_WALL)), 0, 0),
                     ('fg_ax', -int(round(cx * sc.P_FG)), 0, 0)])


_MAGENTA = np.array((255, 0, 255), np.uint8)


def _or_shift(out, key, x, y):
    """OR one cached layer mask into `out` at (x, y) (a layer that drifts every frame)."""
    m = _opaque(key, 0)
    h, w = m.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] |= m[y0 - y:y1 - y, x0 - x:x1 - x]


@lru_cache(maxsize=16)
def _scenery_inside(cx):
    return _scenery([('far_b', -int(round(cx * sc.P_BFAR)), 0, 0),
                     ('slope_bx', -int(round(cx * sc.P_BSLOPE)), 0, 2), ('vil_b', -cx, 0, 2),
                     ('fg_b', -int(round(cx * sc.P_BFG)), 0, 2)])


@lru_cache(maxsize=16)
def _scenery_north(cx, cy):
    def off(p):
        return -int(round(cx * p)), -int(round(cy * p))
    return _scenery([('far_cx', *off(sc.P_CFAR), 0), ('ever_cx', *off(sc.P_CEVER), 0),
                     ('mid_cx', *off(sc.P_CMID), 0), ('igl_cx', *off(sc.P_CIGL), 0),
                     ('fg_cx', *off(sc.P_CFG), 0)])

