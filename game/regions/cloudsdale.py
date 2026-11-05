"""Cloudsdale as an explorable diorama (art: scenes/s02_cloudsdale.py, used read-only).

The film shows three different views, so the region has three areas, in film order:
  0  the sky city: the floating island over the cloud sea, with the film's vertical rise (fly low into
     the cloud banks, climb to see the whole city) and its pan across the dome and the Colosseum.
     The scene's `_scene_city` is rebound to the player's camera (no film balloon / tags / card / fog).
  1  the Weather Factory cross-section (the film's `_scene_factory` rebound: camera, no balloon / tag).
     Machines run on the scene frame, so the press strikes and the charger arcs stay on the beat.
  2  the liquid-rainbow falls close-up, re-composed here from the scene's layers and helpers so that
     the film balloon and its mist patch are gone, the cloud sea keeps its gap over Galloping Gorge,
     and the film's `rainbow` hit (spray burst + bright bow) plays when you arrive and on demand.

Interaction (SPACE): the dome's lantern blazes and rings light over the city; a rainbow arches over
the Colosseum while its pennants whip and streamers burst above it; the charger sends chain lightning
through every jar; the snow machine spurts a flurry of the film's snowflakes; the falls throw up a
geyser of rainbow drops with the bright bow (the film's `rainbow` hit). Three atlas stamps hide
under the island's far end, on the factory roof and in the sky below the city's underside past the falls.
"""
import math
from functools import lru_cache

from engine import W, fx, seg, ease_out, bayer, hash01, Sprite, col, mix, clamp
import numpy as np
from scipy import ndimage
import scenes.s02_cloudsdale as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# ------------------------------------------------------------------ area 0: the city
CITY_CAM_X = sc.CITY_W - W            # city + back city are 480 px wide: cam 0..160
CITY_CAM_Y = 27                       # the film's rise: Rv 90..0, city layer moves 0.3 Rv
CITY_LANDMARKS = [
    Landmark('CLOUDSDALE', sc.DOME[0], (sc.LANTERN[0] + 3, sc.LANTERN[1] - 1, 'right', 16)),
    Landmark('COLOSSEUM', sc.COLO[0], (sc.COLO[0] + 14, sc.COLO[1] + 6, 'up', 22)),
]

CITY_SPOTS = [
    Hotspot('lantern', sc.DOME[0], 66, 'spark', cooldown=80),          # bubble over the drum, clear of the tag
    Hotspot('banners', 292, 70, 'flag', cooldown=96),
]
CITY_STAMPS = [Stamp('underisland', 436, 119)]     # low under the island's far east end, above the falls
STREAMERS = (282, 296, 310, 330, 344, 358)         # rainbow streamers launched from the arena
ARCH_C = (sc.COLO[0], sc.COLO[1] - 1)              # the rainbow arches from the arena rim

# ------------------------------------------------------------------ area 1: the factory
FACT_CAM_X = sc.FW - W                # 16
FACT_LANDMARKS = [Landmark('WEATHER FACTORY', 168, (168, 13, 'right', 10))]
CHARGER = (191, 52)
JARS = [(217 + 11 * j, 61) for j in range(9)] + [(217 + 11 * j, 86) for j in range(8, -1, -1)]
HOPPER = (41, 50)
SNOW_ICON = Sprite.from_ascii('''
    ....B....
    .B..B..B.
    ..B.B.B..
    ...BbB...
    BBBBbBBBB
    ...BbB...
    ..B.B.B..
    .B..B..B.
    ....B....''', {'B': 'royal', 'b': 'baby_blue'})
FACT_SPOTS = [
    Hotspot('snow', HOPPER[0], 60, SNOW_ICON, cooldown=90),
    Hotspot('charger', CHARGER[0], CHARGER[1] + 2, 'bolt', cooldown=75),
]
FACT_STAMPS = [Stamp('roof', 84, 23)]              # up on the roof, beside the west chimney

# ------------------------------------------------------------------ area 2: the falls close-up
FX0 = 196
FALLS_CAM_X = 50                      # the city underside layer (400 px at -cam - 30) ends at cam 50
FALLS_LANDMARKS = [Landmark('LIQUID RAINBOW', FX0 + 27, (FX0 + 4, 74, 'left', 14))]
FALLS_SPOTS = [Hotspot('falls', FX0 + 27, 122, 'splash', cooldown=90)]
FALLS_HORIZON = 150                   # the falls' sky reaches the horizon here (sc._falls_bg's gradient)
FALLS_STAMPS = [Stamp('underside', 300, 58)]       # in the open sky east of the falls, below the city's
#                                                   underside (clear of the HUD counters at the top right)


def _nop(*_a, **_k):
    pass


def _mend(a):
    """The scene's tileable cloud strips (bank_strip) carry a 1-px vertical rim line in their last
    column, which shows as a gold seam when they wrap; repeat the column before it instead."""
    a = a.copy()
    a[:, -1] = a[:, -2]
    return a


def _arch(c, cx, cy, r0, sweep, k):
    """A six-band rainbow arching over (cx, cy) (inner radius r0), drawn from its left foot to the
    right as `sweep` goes 0..1, dithered in by k (0..1)."""
    R1 = r0 + 12
    x0, x1 = max(0, cx - R1), min(W, cx + R1 + 1)
    y0, y1 = max(0, cy - R1), min(sc.H, cy + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    band = np.floor((d - r0) / 2).astype(int)
    ang = np.arctan2(cy - yy, xx - cx)               # pi at the left foot, 0 at the right
    m = (band >= 0) & (band < 6) & (ang >= math.pi * (1 - sweep))
    if k < 1:
        m &= bayer(y1 - y0, x1 - x0, x0, y0) < k
    c.a[yy[m], xx[m]] = sc.RB[::-1][band[m]]
    hi = m & (band == 5) & ((xx + yy) % 3 == 0)      # a glint along the outer band
    c.a[yy[hi], xx[hi]] = sc.RB_HI[0]


@lru_cache(maxsize=1)
def _falls_layers():
    sea, under, islet = sc._falls_layers()
    return _mend(sea), under, islet


_NOT_SKY = np.array((255, 0, 255), np.uint8)     # sky hooks: a key no painted pixel is near


# ------------------------------------------------------------------ the clock's light on the cloud city
# The film painted Cloudsdale at sunrise: every cloud bank, puff and ledge has a gold / butter rim of
# low sunlight, a third of the round windows glow butter, gold glints sit on the far cloud sea under the
# sun on the right, the back city and the far sea are hazed peach. Here the rims are gold only when the
# clock's sun is low (sunrise, golden hour to sunset) and plain white sunlit tops by day (moonlit pale
# edges by night, not olive lines), the far haze is a pale sky haze by day, the glints follow the clock's
# sun, and by night the city's windows, doors and the dome's lantern light up one by one (kept shining
# with GlowKeep, the relight's own glow keeping is off: it would keep the gold finials, rim fragments and
# the vivid pennants glowing).
_G, _G_HI = np.array(sc.G, np.uint8), np.array(sc.G_HI, np.uint8)
_RIM_DAY = {tuple(sc.G): col(sc.CL_LT), tuple(sc.G_HI): col('white')}
_UNDER_DAY = mix(sc.CL_MID, 'white', 0.4)
_WIN = np.array(col(sc.A_WIN), np.uint8)
_DOOR = np.array(col(sc.A_WIN2), np.uint8)
_LIT = np.concatenate([nightwin.WARM, np.array([col('#f2b955'), col('#f0c26e')], np.uint8)])   # + door, far


def _goldness(hour, from_h):
    """0..1 (in thirds): how golden the low sun's rim light is at the clock's hour: full at sunrise while
    the painted sunrise sky (its sun low on the right, where the rims are) is kept, and from golden hour
    to sunset (the clock's sun low on the right again); none by day, at dawn or by night."""
    h = hour % 24
    morning = 1 - daylight.sky_weight(from_h, h) if 4.5 <= h <= 9.5 else 0.0
    g = max(morning, clamp((h - 16.4) / 1.0) * clamp((19.4 - h) / 0.6))
    return round(g * 3) / 3


def _rims(rgba, g, where=None, under=None):
    """A copy of a layer with its sun-rim pixels (exact gold / butter, `where` limits them) moved from
    the painted gold (g 1) to plain white sunlit tops (g 0); `under`: where the island's underside has
    its peach rim (the low sun's warm bounce: pale cloud light at g 0, not a muddy line by night)."""
    out = rgba.copy()
    if g >= 1:
        return out
    a = out[..., :3]
    for src, day, lim in ((sc.G, _RIM_DAY[tuple(sc.G)], where), (sc.G_HI, _RIM_DAY[tuple(sc.G_HI)], where),
                          (col('peach'), _UNDER_DAY, under)):
        if lim is False:
            continue
        m = np.all(a == np.array(src, np.uint8), axis=2)
        if lim is not None:
            m &= lim
        a[m] = np.array(mix(day, src, g), np.uint8)
    return out


@lru_cache(maxsize=1)
def _city_parts():
    """The city layer's lights and rims: (rim mask, painted-lit window mask, windows (ys, xs, v, kind))
    in city px. Windows: the drum's arched slits, the wings', the round house and tower windows (those
    painted lit butter at sunrise too), the Colosseum's arches, the doors and the lantern's slits; v =
    when it goes on (< 0: with the lamps), kind = its warm shade (3: a door)."""
    rgba = sc._city()[0]
    a, op = rgba[..., :3], rgba[..., 3] >= 128
    gold, butter = np.all(a == _G, axis=2) & op, np.all(a == _G_HI, axis=2) & op
    out = np.all(a == np.array(col(sc.A_OUT), np.uint8), axis=2) & op
    lab, n = ndimage.label(butter)
    litwin = np.zeros_like(butter)
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        if s_[0].stop - s_[0].start <= 3 and s_[1].stop - s_[1].start <= 3:
            y0, x0 = max(0, s_[0].start - 1), max(0, s_[1].start - 1)
            if out[y0:s_[0].stop + 1, x0:s_[1].stop + 1].sum() >= 4:   # a butter pane in a window ring
                litwin[s_] |= lab[s_] == i
    door = np.all(a == _DOOR, axis=2) & op
    rows = np.arange(a.shape[0])[:, None]
    rim = (butter & ~litwin) | (gold & (rows >= sc.CITY_BY - 6) & ~ndimage.binary_dilation(door, iterations=2))
    win = (np.all(a == _WIN, axis=2) & op) | litwin | door
    lab, n = ndimage.label(win)
    ys, xs, vs, ks = [], [], [], []
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        yy, xx = np.nonzero(lab[s_] == i)
        yy, xx = yy + s_[0].start, xx + s_[1].start
        x0, y0 = s_[1].start, s_[0].start
        if 380 <= x0 < 402 and s_[0].stop - y0 > 10:
            continue                                     # the arch gate's opening: open sky, not a window
        v = hash01(x0, y0, 501)
        if abs(x0 - sc.LANTERN[0]) <= 6 and abs(y0 - sc.LANTERN[1]) <= 8:
            v = -1.0                                     # the dome's lantern: lit with the lamps
        elif 265 <= x0 < 375:
            v *= 0.8                                     # the Colosseum's arches: most of them lit
        isdoor = door[yy[0], xx[0]]
        if isdoor and abs(x0 - sc.DOME[0]) <= 4:
            v = -1.0                                     # the rotunda's portico door
        ys.append(yy)
        xs.append(xx)
        vs.append(np.full(len(yy), v, np.float32))
        ks.append(np.full(len(yy), 3 if isdoor else int(hash01(x0, y0, 502) * 3)))
    return rim, litwin, (np.concatenate(ys), np.concatenate(xs), np.concatenate(vs), np.concatenate(ks))


@lru_cache(maxsize=4)
def _city_g(g):
    """The city layer at goldness g, its sunrise-lit windows as plain glass (lit again by night here)."""
    rgba, poles = sc._city()
    rim, litwin, _w = _city_parts()
    rows = np.arange(rgba.shape[0])[:, None] >= sc.CITY_BY
    out = _rims(rgba, g, rim, rows)
    out[litwin, :3] = _WIN
    return out, poles


@lru_cache(maxsize=4)
def _lips_g(g):
    return _rims(sc._city_lips(), g)


_BACK_HAZE = mix('cotton', 'peach', 0.3)                             # sc._backcity's sunrise haze ...
_FAR_HAZE = mix('peach', 'cotton', 0.5)                              # ... and the far cloud sea's (far1)
_DAY_HAZE = mix('cotton', 'baby_blue', 0.6)                          # the haze of a blue-sky day


def _rehaze(rgba, cols, g):
    """A copy of a layer with its painted hazy colours (pairs (painted, day) in `cols`) moved to their
    day versions as the sun's goldness g goes from 1 (as painted) to 0."""
    out = rgba.copy()
    if g >= 1:
        return out
    a = out[..., :3]
    m = [np.all(a == np.array(col(src), np.uint8), axis=2) for src, _day in cols]
    for (src, day), mm in zip(cols, m):
        a[mm] = np.array(mix(day, src, g), np.uint8)
    return out


def _haze_pairs(ramp, a, rim=None):
    """(painted, day) colour pairs of a ramp hazed by sc.ramp_haze at strength a (+ a hazed rim)."""
    pairs = list(zip(sc.ramp_haze(ramp, _BACK_HAZE, a), sc.ramp_haze(ramp, _DAY_HAZE, a)))
    if rim is not None:
        pairs.append((mix(rim, _BACK_HAZE, 0.3), mix('white', _DAY_HAZE, 0.3)))
    return pairs


@lru_cache(maxsize=4)
def _banks_g(g):
    """The cloud banks at goldness g (rims; the far sea's sunrise haze by day a pale sky haze)."""
    far1, *near = sc._banksA()
    pairs = list(zip(sc.ramp_haze(sc.NEAR, _FAR_HAZE, 0.5),
                     sc.ramp_haze(sc.NEAR, mix(_DAY_HAZE, 'cotton', 0.5), 0.5)))
    return tuple(_rims(_mend(a), g) for a in [_rehaze(far1, pairs, g)] + near)


@lru_cache(maxsize=4)
def _backcity_g(g):
    """The hazy back city at goldness g: its sunrise haze (peach, gold rims) by day a pale sky haze."""
    return _rehaze(sc._backcity(), _haze_pairs(sc.ARCH, 0.45, sc.G_HI), g)


@lru_cache(maxsize=1)
def _back_windows():
    """(ys, xs, v) of the hazy back city's tiny windows (lone dark-haze px of sc._backcity)."""
    rgba = sc._backcity()
    a, op = rgba[..., :3], rgba[..., 3] >= 128
    dark = np.all(a == np.array(sc.ramp_haze(sc.ARCH, _BACK_HAZE, 0.45)[0], np.uint8), axis=2)
    dark &= op
    p = np.pad(dark, 1)
    lone = dark & ~(p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    ys, xs = np.nonzero(lone)
    return ys, xs, np.array([hash01(x, y, 503) for y, x in zip(ys, xs)], np.float32)


def _lamp_share(hour):
    """0..1: the lamps' (the lantern's, the portico's) share on; they come on before the windows."""
    return clamp((daylight.night(hour) - 0.08) / 0.25)


def _city_lights(c, hour, g, cx0, cy0, bx0, by0, lit):
    """The city by night: its windows, doors and the lantern lit where nothing covers them (painted
    warm; `lit` collects them for GlowKeep)."""
    kw, kl = nightwin.ramp(hour), _lamp_share(hour)
    if kw <= 0 and kl <= 0:
        return
    ys, xs, v, kd = _city_parts()[2]
    src = _city_g(0)[0]
    for (yy, xx, vv, kk, ox, oy, layer, share) in (
            (ys, xs, v, kd, cx0, cy0, src, 0.75),
            (*_back_windows(), None, bx0, by0, _backcity_g(g), 0.5)):
        if kk is None:
            kk = np.full(len(yy), 4)
        on = np.where(vv < 0, kl > 0, vv < kw * share)
        sy, sx = yy + oy, xx + ox
        ok = on & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < sc.H)
        if not ok.any():
            continue
        seen = np.all(c.a[sy[ok], sx[ok]] == layer[yy[ok], xx[ok], :3], axis=1)
        sy, sx, k = sy[ok][seen], sx[ok][seen], kk[ok][seen]
        c.a[sy, sx] = _LIT[k]
        lit.mark(c, sy, sx)


@lru_cache(maxsize=64)
def _relit(rgb, from_h, hq):
    """The colour rgb (painted at from_h) under the clock's light at hq."""
    a = np.array([[rgb]], np.uint8)
    daylight.relight(a, from_h, hq, glow=False)
    return tuple(int(v) for v in a[0, 0])


@lru_cache(maxsize=1)
def _city_alphas():
    return sc._backcity()[..., 3] >= 128, sc._city()[0][..., 3] >= 128


def _occlude(c, al, x, y):
    """Sky hooks: mark a scenery layer's opaque pixels (al) as not-sky (the pale cloud architecture
    is within the sky test's tolerance of the hazy sky and the sun's glow behind it)."""
    lh, lw = al.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + lw), min(sc.H, y + lh)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0:y1, x0:x1], _NOT_SKY, where=al[y0 - y:y1 - y, x0 - x:x1 - x, None])


@lru_cache(maxsize=512)
def _sky_city(cx, cy):
    """_scene_city's sky (camera rebound to (cx, cy / 0.3)): gradient, then the low sun and its glows."""
    px, Rv = float(cx), cy / 0.3
    c = sc.Canvas()
    c.blit(sc._skyA(), 0, -int(round(Rv * 0.05)))
    sx, sy = 296 - int(round(px * 0.06)), 46 - int(round(Rv * 0.1))
    fx.glow(c, sx, sy, 90, 'gold', 0.28, rings=5)
    fx.glow(c, sx, sy, 34, 'butter', 0.45)
    c.circle(sx, sy, 10, 'butter')
    c.circle(sx, sy, 8, 'cream')
    c.circle(sx - 2, sy - 2, 4, 'white')
    back, city = _city_alphas()
    _occlude(c, back, -int(round(px * 0.7)) - 10, -int(round(Rv * 0.27)) - 2)
    _occlude(c, city, -int(round(px)), -int(round(Rv * 0.3)))
    return c


@lru_cache(maxsize=1)
def _factory_sky_layer():
    """The sky gradients of the scene's _factory_bg, without the cloud tops and the building."""
    cv = sc.Canvas(sc.FW, sc.H)
    cv.gradient(0, 0, sc.FW, 60, ['sky', 'light_sky', 'baby_blue', 'cotton'], soft=0.7)
    cv.gradient(0, 60, sc.FW, 120, ['cotton', 'peach'])
    return cv


@lru_cache(maxsize=64)
def _sky_factory(cx):
    c = sc.Canvas()
    c.blit(_factory_sky_layer(), -cx, 0)
    return c


@lru_cache(maxsize=128)
def _sky_falls(camx):
    c = sc.Canvas()
    c.blit(sc._falls_bg(), -int(round(camx * 0.15)), 0)
    for k in range(3):
        x0 = 250 + k * 30 - int(camx * 0.3)
        fx.light_beam(c, ((x0, 0), (x0 + 10, 0)), ((x0 - 120, 150), (x0 - 80, 150)), 'butter', 0.10)
    return c


class _Lit:
    """Collects the pixels that glow by themselves (the liquid rainbow, the city's lit windows, the
    lantern, the events' bursts of light): they keep shining at night (GlowKeep). Off by day (no cost)."""
    def __init__(self, on):
        self.on = on
        self.m = np.zeros((sc.H, W), bool) if on else None
        self.v = np.zeros((sc.H, W, 3), np.uint8) if on else None

    def fall(self, c, *a, **kw):
        """sc.rainbow_fall, its pixels collected."""
        if not self.on:
            return sc.rainbow_fall(c, *a, **kw)
        before = c.a.copy()
        sc.rainbow_fall(c, *a, **kw)
        self.drawn(c, before)
        return None

    def drawn(self, c, before):
        """Everything drawn since the copy `before` glows."""
        new = (c.a != before).any(axis=2)
        self.m |= new
        self.v[new] = c.a[new]

    def mark(self, c, yy, xx):
        self.m[yy, xx] = True
        self.v[yy, xx] = c.a[yy, xx]

    def mask(self, c):
        """The glowing pixels still showing (anything drawn over them since, cloud or mist, drops out)."""
        return self.m & (c.a == self.v).all(axis=2)


# ------------------------------------------------------------------ the falls, the glints, the wisps
_HZ = 'cotton'                                                       # the far islet's haze (sc._falls_layers)


def _islet_rim(rim):
    """The far islet's rim colour as sc._falls_layers paints it: mixed into the haze, then hazed again."""
    a = np.array(col(mix(rim, _HZ, 0.3)), np.float32) * 0.75 + np.array(col(_HZ), np.float32) * 0.25
    return a.astype(np.uint8)


@lru_cache(maxsize=4)
def _falls_g(g):
    """The falls close-up's layers (cloud sea, city underside, far islet) with their sun rims at
    goldness g (see _goldness)."""
    sea, under, islet = _falls_layers()
    islet = islet.copy()
    if g < 1:
        a = islet[..., :3]
        m = np.all(a == _islet_rim(sc.G_HI), axis=2)
        a[m] = np.array(mix(_islet_rim('white'), _islet_rim(sc.G_HI), g), np.uint8)
    return _rims(sea, g), _rims(under, g), islet


def _mist_rim(g):
    """The mist's rim colour (sc.mist_cluster paints it butter: the low sun)."""
    return mix('white', 'butter', g)


def _glints(c, hour, from_h, ry, f):
    """The film's sun glints on the far cloud sea (butter dashes under its painted sun) for the clock:
    under the clock's sun once its sky is the clock's, white by day, butter when the sun is low, none
    once it has set. ry: the far sea's row offset (the film's -Rv * 0.15)."""
    kind, bx, _by = daylight._body_pos(hour)
    if kind != 'sun' or daylight.night(hour) > 0.1:
        return
    cc = 'butter' if _goldness(hour, from_h) >= 0.5 else 'white'
    for i in range(10):
        if (f // 6 + i) % 5 == 0:
            gx = bx - 40 + int(hash01(i, 1) * 80)
            c.hline(gx, gx + 2 + i % 3, 95 + int(hash01(i, 2) * 8) + ry, cc)


class _CityCanvas(sc.Canvas):
    """_scene_city's canvas: once `wisps` is a list, its dithers (the drifting wisps, drawn last) are
    collected there instead of drawn (drawn again for the clock after its light: _wisps)."""
    wisps = None

    def dither(self, x, y, w, h, c, level, ox=0, oy=0):
        if self.wisps is None:
            return super().dither(x, y, w, h, c, level, ox, oy)
        self.wisps.append((x, y, w, h, c, level, ox, oy))
        return None


def _wisps(c, wisps, hour, from_h):
    """The drifting wisps over the frame after the clock's light and sky: the film's cream / white
    checker under the clock's light, fading out with the dusk (not pale checker bars on the night sky)."""
    k = clamp(1 - 1.2 * daylight.night(hour))
    hq = round(hour * 10) / 10.0                     # (the relight's own 6-minute steps)
    for (x, y, w, h, cc, level, ox, oy) in wisps:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(sc.H, y + h)
        if x1 <= x0 or y1 <= y0 or level <= 0:
            continue
        m = bayer(y1 - y0, x1 - x0, x0 + ox, y0 + oy) < level
        tgt = np.array(_relit(col(cc), from_h, hq), np.float32)
        sub = c.a[y0:y1, x0:x1]
        cur = sub[m].astype(np.float32)
        sub[m] = (cur + (tgt - cur) * k).astype(np.uint8)


def _resky(c, mask, hour, t, ix, iy, top, height):
    """The clock's sky again over the sky pixels, from a strip reaching down to the area's own horizon:
    game/daylight.paint_sky's strip ends 120 rows below the top of the highest view and repeats its
    last row below that, a flat band of horizon colour (peach at 16:00, gold at sunset) under a sky that
    is still blue or red just above it."""
    daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, top, height, t)


class _CityFx:
    """engine.fx for _scene_city by the clock: its sun glints on the windows and the column's gilt sun
    twinkle by day only; by night the lantern's twinkle is its light (butter, kept shining)."""
    def __init__(self, night, lit):
        self.night, self.lit = night, lit

    def __getattr__(self, name):
        return getattr(fx, name)

    def sparkle(self, c, x, y, f, color='white', period=24, phase=0.0, size=2):
        if period == 36:                                 # the dome's lantern
            if self.night > 0.2:
                before = c.a.copy()
                fx.sparkle(c, x, y, f, 'butter', period, phase, size)
                self.lit.drawn(c, before)
                return
        elif self.night > 0.2:                           # sunlight on glass and gilt: none at night
            return
        fx.sparkle(c, x, y, f, color, period, phase, size)


class Cloudsdale(GlowKeep, Region):
    SEG = 'cloudsdale'
    CARD = ('SKY CITY', ['BUILT FROM CLOUDS', 'WHERE WEATHER IS MADE'])

    def build_areas(self):
        return [
            Area(self.render_city, cam_x=(0, CITY_CAM_X), cam_y=(0, CITY_CAM_Y), y_range=(30, 134),
                 start=(110, 131), landmarks=CITY_LANDMARKS, balloon='m',
                 hotspots=CITY_SPOTS, stamps=CITY_STAMPS, sky=self.sky_city),
            # the factory is a cut-away: the balloon drifts past its open side (in front of the floors)
            Area(self.render_factory, cam_x=(0, FACT_CAM_X), cam_y=(0, 0), y_range=(34, 150),
                 start=(40, 90), landmarks=FACT_LANDMARKS, balloon='m',
                 hotspots=FACT_SPOTS, stamps=FACT_STAMPS, sky=self.sky_factory),
            Area(self.render_falls, cam_x=(0, FALLS_CAM_X), cam_y=(0, 0), y_range=(40, 132),
                 start=(60, 110), landmarks=FALLS_LANDMARKS, balloon='l',
                 hotspots=FALLS_SPOTS, stamps=FALLS_STAMPS, sky=self.sky_falls),
        ]

    # -------------------------------------------------------------- clocks + sound
    def init_state(self):
        self.hit_auto = None        # step of the falls' arrival burst (the film's `rainbow` hit)

    def on_area(self, ai):
        g = self.g
        if ai == 1:
            g.sfx('hum', 0.4)
        elif ai == 2:
            self.hit_auto = g.t + 30                 # once the cloud sweep has cleared
            g.sfx('waterfall', 0.45)

    def ago(self, key, t):
        """Steps since hotspot `key` was last set off (large if never)."""
        return t - self.spot_t.get(key, -10 ** 6)

    def hit_f(self, t):
        """Steps since the last falls burst (arrival or SPACE); large when there was none."""
        a = self.ago('falls', t)
        if self.hit_auto is not None and t >= self.hit_auto:
            a = min(a, t - self.hit_auto)
        return a

    def on_interact(self, key, t):
        g = self.g
        if key == 'lantern':
            g.sfx('chime', 0.55)
            g.sfx('shimmer', 0.35)
        elif key == 'banners':
            g.sfx('banner', 0.5)
            g.sfx('whoosh', 0.3)
        elif key == 'charger':
            g.sfx('zap', 0.65)
        elif key == 'snow':
            g.sfx('puff', 0.45)
            g.sfx('chime', 0.3)
        elif key == 'falls':
            g.sfx('splash', 0.6)
            g.sfx('shimmer', 0.5)

    def update(self, t):
        g = self.g
        f = self.scene_f(t)
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.ai == 0:
            if t % 400 == 120:
                g.sfx('waterfall', 0.2)
            if self.ago('banners', t) == 18:
                g.sfx('firework', 0.45)
            elif self.ago('banners', t) == 30:
                g.sfx('firework', 0.3)
        elif self.ai == 1:
            ph = f % 72
            if ph == 9:
                g.sfx('clank', 0.3)
            elif ph == 18:
                g.sfx('zap', 0.25)
            elif ph == 27 and (f // 72) % 2 == 0:
                g.sfx('steam', 0.25)
            elif ph == 36 and (f // 72) % 2 == 1:
                g.sfx('bubbles', 0.25)
            if t % 300 == 0:
                g.sfx('hum', 0.3)
            if self.ago('charger', t) in (12, 24):
                g.sfx('zap', 0.4)
            elif self.ago('charger', t) == 36:
                g.sfx('crack', 0.35)
        else:
            if self.hit_auto is not None and t == self.hit_auto:
                g.sfx('shimmer', 0.6)
                g.sfx('splash', 0.4)

    # -------------------------------------------------------------- sky hooks (game/daylight.py)
    # Each reproduces only the sky steps its render draws first (gradient, sun disc and glows, the
    # falls' light shafts), cached per camera.
    def sky_city(self, f, cx, cy):
        return _sky_city(int(cx), int(cy))

    def sky_factory(self, f, cx, cy):
        return _sky_factory(int(cx))

    def sky_falls(self, f, cx, cy):
        return _sky_falls(int(cx))

    # -------------------------------------------------------------- the clock's light
    # The relight's own glow keeping is off (it would keep the gold finials, rim fragments and the vivid
    # pennants shining like lamps): the real lights are collected by the area renders (_Lit) and kept
    # with GlowKeep. Some light is drawn after the clock's light and sky (the wisps, the falls' sky).
    _post = None
    _lighting = False               # inside _world: after_light will run on this render

    def auto_glow(self):
        return False

    def _later(self, fn):
        """Run fn(c) on the frame right after the clock's light and sky (Region.after_light)."""
        if self._post is None:
            self._post = []
        self._post.append(fn)

    def _world(self, t, cam=None, with_balloon=False):
        self._post = None
        self._lighting = True
        try:
            return super()._world(t, cam, with_balloon)
        finally:
            self._lighting = False

    def after_light(self, c, t):
        post, self._post = self._post, None
        if (self.ID, self.ai) in daylight.INTERIORS:
            return
        if self.ai == 2:
            self._sky_down_to(c, t, FALLS_HORIZON)
        self._moonlit(c, t)
        for fn in post or ():
            fn(c)
        super().after_light(c, t)                        # GlowKeep: the lights get their colour back

    NIGHT_DIM = 0.3                 # by night the cloud city is this much darker than the relight makes it

    def _moonlit(self, c, t):
        """By night the relight leaves the white cloud city a bright periwinkle mass (twice as bright as
        another city's night) that the lit windows hardly stand out from: the scenery (not the sky, not
        the lights) is darkened with the night."""
        k = self.NIGHT_DIM * daylight.night(self.g.hour)
        memo = self._sky_memo
        if k < 0.01 or memo is None or memo[0][:2] != (self.ai, t):
            return
        keep = memo[1] if self._glow is None else memo[1] | self._glow[1]
        m = ~keep
        c.a[m] = (c.a[m].astype(np.uint16) * int(round((1 - k) * 256)) >> 8).astype(np.uint8)

    def _sky_down_to(self, c, t, bottom):
        """Once the clock's sky is all there, repaint it reaching the area's horizon at world row
        `bottom` (see _resky). Right after the base _world's paint_sky: its sky mask is this frame's."""
        memo, hour = self._sky_memo, self.g.hour
        if memo is None or memo[0][:2] != (self.ai, t) or daylight.sky_weight(self.HOUR, hour) < 0.999:
            return
        (_ai, _t, ix, iy), mask = memo
        y0, y1 = self.area.cam_y
        if bottom - (y0 - 10) > y1 - y0 + 130 and mask.any():
            _resky(c, mask, hour, t, ix, iy, y0 - 10, bottom - (y0 - 10))

    def warm(self):
        for g in (0.0, 1 / 3, 2 / 3, 1.0):          # every rim goldness, built at landing: the clock's dusk
            _banks_g(g), _city_g(g), _lips_g(g), _backcity_g(g), _falls_g(g)   # would stall a frame on each
        super().warm()

    # -------------------------------------------------------------- area 0
    def render_city(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        night = daylight.night(hour)
        g = _goldness(hour, self.HOUR)
        flap = self.ago('banners', t) < 84
        banks = _banks_g(g)
        moved = daylight.sky_weight(self.HOUR, hour) >= 0.5      # the clock's sun, not the painted one
        state = {}

        def pennants(c, poles, ox, oy, _f):          # the pennants whip in the wind of the show
            sc._pennants(c, poles, ox, oy, f * 3 if flap else f)

        def blit_wrap(c, arr, x, y):
            if arr is banks[0] and moved:                # the glints go on the far sea right after it
                sc.blit_wrap(c, arr, x, y)
                state['sea'] = c.a.copy()
                return
            if arr is banks[1] and 'sea' in state:       # ...moved under the clock's sun
                np.copyto(c.a, state.pop('sea'))
                _glints(c, hour, self.HOUR, -int(round(cy / 0.3 * 0.15)), f)
            sc.blit_wrap(c, arr, x, y)
            if arr is banks[4] and self._lighting:
                c.wisps = []                             # the wisps come next: drawn after the light

        lit = _Lit(night > 0.05)
        render = rebind(sc._scene_city, _camA=lambda _f: (float(cx), cy / 0.3), _pennants=pennants,
                        _banksA=lambda: banks, _city=lambda: _city_g(g), _city_lips=lambda: _lips_g(g),
                        _backcity=lambda: _backcity_g(g), blit_wrap=blit_wrap, Canvas=_CityCanvas,
                        fx=_CityFx(night, lit), draw_balloon=_nop, callout=_nop, atlas_card=_nop, veil=_nop,
                        rainbow_fall=lit.fall)
        c = render(f)
        wisps, c.wisps = c.wisps, None
        if wisps:
            self._later(lambda cc: _wisps(cc, wisps, hour, self.HOUR))
        if lit.on:
            _city_lights(c, hour, g, -int(round(cx)), -int(round(cy)), -int(round(cx * 0.7)) - 10,
                         -int(round(cy * 0.9)) - 2, lit)
            before = c.a.copy()
            self._city_events(c, t, cx, cy)
            lit.drawn(c, before)                         # the lantern's blaze, the rainbow, the streamers
            self.keep_glow(c, lit.mask(c), 0.9)
        else:
            self._city_events(c, t, cx, cy)
        return c

    def _city_events(self, c, t, cx, cy):
        """Visible results of the city's SPACE events (reads state only)."""
        a = self.ago('lantern', t)
        if 0 <= a < 72:                                  # the dome's lantern blazes like a second sun
            lx, ly = sc.LANTERN[0] - cx, sc.LANTERN[1] - 3 - cy
            k = 1 - a / 72
            fx.glow(c, lx, ly + 18, 34, 'gold', 0.35 * k, rings=4)      # the dome catches the light
            fx.glow(c, lx, ly, 22 + a // 2, 'gold', 0.6 * k, rings=5)
            fx.glow(c, lx, ly, 10, 'white', 0.85 * k, rings=3)
            for j in range(3):                           # pulses of light ring out over the city
                r = (a - j * 10) * 1.3 + 5
                if 5 <= r < 52:
                    c.circle(lx, ly, int(r), ('butter', 'gold', 'white')[j], fill=False)
            if a < 40:                                   # turning rays
                ln = int(8 + min(a, 12) * 2.0 * (1 - seg(a, 28, 40)))
                for j in range(8):
                    ang = j * math.pi / 4 + a * 0.04
                    ex, ey = lx + int(round(math.cos(ang) * ln)), ly + int(round(math.sin(ang) * ln))
                    sx, sy = lx + int(round(math.cos(ang) * 4)), ly + int(round(math.sin(ang) * 4))
                    c.line(sx, sy, ex, ey, 'gold' if j % 2 else 'white')
            for j in range(22):                          # golden sparks shower down the dome
                u = a - (j % 6) * 3
                if not 0 <= u < 50:
                    continue
                dx = (hash01(j, 1) - 0.5) * 2 * (4 + u * 0.55)
                y = ly + 2 + u * (0.25 + 0.3 * hash01(j, 2)) + 0.012 * u * u
                if (u + j) % 5:
                    c.px(int(lx + dx), int(y), ('white', 'butter', 'gold')[j % 3])
            for i, (gx, gy) in enumerate(((150, 70), (182, 68), (232, 86), (258, 80), (99, 46))):
                u = a - 6 - i * 4                        # windows answer, one after another
                if 0 <= u < 24:
                    fx.sparkle(c, gx - cx, gy - cy, u, 'white', period=24, size=3)
        a = self.ago('banners', t)
        if 0 <= a < 84:                                  # a rainbow arches over the Colosseum ...
            sweep = ease_out(seg(a, 2, 20))
            k = 1 - seg(a, 56, 84)
            if sweep > 0 and k > 0:
                _arch(c, ARCH_C[0] - cx, ARCH_C[1] - cy, 34, sweep, k)
            t0 = t - a                                   # ... and streamers burst above it
            for i, x in enumerate(STREAMERS):
                s0 = i * 3 if i % 2 == 0 else 9 + i * 2
                u = a - s0
                if u < 0:
                    continue
                sx = x - cx + (1 if i % 2 else -1) * (u // 6 if u < 16 else 2)
                cc = sc.RB_NAMES[i]
                apex = 16 + (i * 7) % 12
                if u < 16:
                    y = int(round(56 - (56 - apex) * ease_out(u / 16))) - cy
                    c.rect(sx, y, 2, 8, cc)
                    c.rect(sx, y + 8, 2, 6, sc.RB_PAS[i])
                    c.rect(sx, y - 1, 2, 2, 'white')
                else:
                    ui.burst(c, t, t0 + s0 + 16, sx, apex - cy, colors=(cc, cc, 'white'),
                             n=22, seed=i * 5 + 1, power=0.85, life=46, gravity=0.04)
                    if u < 24:
                        fx.glow(c, sx, apex - cy, 12, cc, 0.6)

    # -------------------------------------------------------------- area 1
    def render_factory(self, f, cx, cy):
        render = rebind(sc._scene_factory, smooth=lambda _u: cx / 14.0,
                        draw_balloon=_nop, callout=_nop)
        c = render(f)
        self._factory_events(c, self.g.t, cx)
        return c

    def _factory_events(self, c, t, cx):
        a = self.ago('charger', t)
        if 0 <= a < 64:                                  # chain lightning through every jar
            ox = -cx
            k = 1 - seg(a, 40, 64)
            fx.glow(c, CHARGER[0] + ox, CHARGER[1], 26, 'butter', 0.7 * k, rings=4)
            fx.glow(c, CHARGER[0] + ox, CHARGER[1], 8, 'white', 0.9 * k, rings=2)
            n = min(len(JARS), a // 2 + 1)
            for j in range(n):                           # every jar the bolt reaches fills with light
                jx, jy = JARS[j][0] + ox, JARS[j][1]
                if k < 1 and hash01(j, a // 3) > k:
                    continue
                fx.glow(c, jx, jy, 9, 'gold', 0.45 * k, rings=3)
                c.rect(jx - 3, jy - 4, 7, 10, 'butter')
                pts = [(jx + int((hash01(j, a // 2, q) - 0.5) * 5), jy - 4 + q * 3) for q in range(4)]
                for p, q in zip(pts, pts[1:]):
                    c.line(p[0], p[1], q[0], q[1], 'white')
            if a < 44:                                   # the bolt head hops jar to jar
                pts = [(CHARGER[0] + 6, CHARGER[1] - 2)] + JARS[:n]
                lo = max(0, len(pts) - 5)
                for s0, (p, q) in enumerate(zip(pts[lo:], pts[lo + 1:])):
                    m1 = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2 - 3 + int((hash01(a // 2, s0) - 0.5) * 8))
                    for dy, cc in ((1, 'gold'), (0, 'white')):
                        c.line(p[0] + ox, p[1] + dy, m1[0] + ox, m1[1] + dy, cc)
                        c.line(m1[0] + ox, m1[1] + dy, q[0] + ox, q[1] + dy, cc)
                hx, hy = pts[-1]
                fx.glow(c, hx + ox, hy, 14, 'white', 0.6)
            if 34 <= a < 70:
                ui.burst(c, t, t - a + 34, JARS[-1][0] + ox, JARS[-1][1], colors=('butter', 'white', 'gold'),
                         n=16, seed=7, power=0.8, life=30, gravity=0.06)
        a = self.ago('snow', t)
        if 0 <= a < 100:                                 # the snow machine spurts a flurry of crystals
            ox = -cx
            if a < 24:                                   # a puff from the hopper
                for k in range(3):
                    r = 3 + (a - k * 4) * 0.3
                    if 0 <= a - k * 4 < 20:
                        c.circle(HOPPER[0] + ox + k * 3 - 3, HOPPER[1] - 8 - (a - k * 4), int(r),
                                 'white' if k % 2 == 0 else 'cloud')
            for i in range(22):
                u = a - (i % 11) * 2
                if u < 0:
                    continue
                vx = 0.7 + 1.3 * hash01(i, 11)
                vy = -0.8 - 0.9 * hash01(i, 12)
                x = 52 + vx * u + 2 * math.sin(u / 7 + i)
                y = 70 + vy * u + 0.035 * u * u
                if y > 93 or x > 200:
                    continue
                spr = sc.FLAKES[i % len(sc.FLAKES)]
                spr.draw(c, int(x) + ox - 3, int(y) - 3)

    # -------------------------------------------------------------- area 2
    def render_falls(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        hf = self.hit_f(t)
        lf = f - 288
        camx = cx
        g = _goldness(hour, self.HOUR)
        lit = _Lit(daylight.night(hour) > 0.05)
        c = sc.Canvas()
        c.blit(sc._falls_bg(), -int(round(camx * 0.15)), 0)
        sea, under, islet = _falls_g(g)
        for k in range(3):
            x0 = 250 + k * 30 - int(camx * 0.3)
            fx.light_beam(c, ((x0, 0), (x0 + 10, 0)), ((x0 - 120, 150), (x0 - 80, 150)), 'butter', 0.10)
        # gorge through the gap + the cloud sea (a gentle sway instead of the film's drift, so the gap
        # stays over the gorge)
        gx = -int(round(camx * 0.35))
        c.blit(sc._gorge(), gx, 146)
        sway = int(round(2 * math.sin(f / 90.0)))
        sc.blit_wrap(c, sea, gx - 2 + sway, 132)
        # far islet + thin fall
        ix = 28 - int(round(camx * 0.5))
        lit.fall(c, ix + 33, 52, 140, 1, f, speed=1.2, seed=3, sheets=False)
        sc.mist_cluster(c, ix + 36, 138, 16, f, seed=4, n=4, rmax=5, rim=_mist_rim(g))
        c.blit(islet, ix, 32)
        # the big rainbow curtain
        ox = -camx
        lit.fall(c, FX0 + ox, 26, 152, 9, f, speed=3.0, seed=12)
        for k in range(14):
            P = 18
            u = ((lf + k * 7) % P) / P
            side = k % 2
            ex = FX0 + ox + (-1 - int(u * 3) if side == 0 else 54 + int(u * 3))
            c.px(ex, 40 + (k * 9 + int(u * 30)) % 100, sc.RB_PAS[0 if side == 0 else 5])
        c.blit(under, -camx - 30, 0)
        sc.mist_cluster(c, FX0 + 27 + ox, 144, 92, f, seed=5, n=12, rmax=15, rim=_mist_rim(g))
        sc.spray(c, FX0 + 27 + ox, 136, 70, f, seed=5, n=30, big=2.0)
        # prismatic bow in the mist: brightens on every hit, then settles to the film's end strength
        k = (0.4 + 0.3 * ease_out(seg(hf, 0, 6)) * (1 - seg(hf, 14, 70))) * (0.85 + 0.15 * math.sin(f / 3))
        cxb, cyb, R0 = FX0 + 27 + ox, 160, 36
        y0b = cyb - R0 - 13
        yy, xx = np.mgrid[y0b:cyb, cxb - R0 - 13:cxb + R0 + 13]
        d = np.sqrt((xx - cxb) ** 2 + (yy - cyb) ** 2)
        band = np.floor((d - R0) / 2).astype(int)
        m = (band >= 0) & (band < 6)
        m &= bayer(yy.shape[0], yy.shape[1], xx[0, 0], yy[0, 0]) < k
        m &= (xx >= 0) & (xx < W) & (yy >= 0)
        c.a[yy[m], xx[m]] = sc.RB_HI[::-1][band[m]]
        if lit.on:
            lit.mark(c, yy[m], xx[m])
            before = c.a.copy()                          # the falls' magic spray and sparkles glow too
        # spray burst on the hit
        for i in range(46):
            dt = hf - (i % 4)
            if dt < 0 or dt > 40:
                continue
            ang = -math.pi * (0.1 + 0.8 * sc.hash01(i, 1))
            v = 1.6 + 2.6 * sc.hash01(i, 2)
            x = FX0 + 27 + ox + math.cos(ang) * v * dt
            y = 140 + math.sin(ang) * v * dt + 0.09 * dt * dt
            cc = sc.RB_PAS[i % 6] if i % 3 else 'white'
            c.px(int(x), int(y), cc)
            if dt < 10:
                c.px(int(x) + 1, int(y), cc)
        if hf < 60:                                      # a prismatic geyser out of the plunge
            if hf < 16:
                fx.glow(c, FX0 + 27 + ox, 132, 44, 'white', 0.55 * (1 - hf / 16), rings=4)
            for i in range(64):                          # chunky rainbow drops fountain up and rain back
                dt = hf - (i % 8)
                if not 0 <= dt < 52 or (dt > 36 and (i + dt) % 3 == 0):
                    continue
                ang = -math.pi * (0.22 + 0.56 * hash01(i, 21))
                v = 1.8 + 2.8 * hash01(i, 22)
                x = FX0 + 27 + ox + math.cos(ang) * v * dt
                y = 138 + math.sin(ang) * v * dt + 0.1 * dt * dt
                c.rect(int(x), int(y), 2, 2, sc.RB_NAMES[i % 6] if i % 4 else 'white')
        if hf < 30:
            fx.sparkles(c, FX0 + 27 + ox, 120, f, radius=40, n=10, colors=('white', 'butter', 'cotton'),
                        seed=8, period=16)
        fx.sparkles(c, FX0 + 27 + ox, 100, f, radius=50, n=6, colors=('white', 'butter'), seed=9, period=34)
        if lit.on:
            lit.drawn(c, before)
            self.keep_glow(c, lit.mask(c), 0.9)
        return c

