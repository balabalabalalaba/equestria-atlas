"""The Everfree Forest as an explorable diorama (art: scenes/s05_everfree.py, used read-only).

The film shows three separate views of the wild woods (glide cuts between them), so the region has three
areas, flown through left to right with a cloud sweep between them:
  1. UNDER THE CANOPY: the winding forest path (the film's shot A).
  2. ZECORA'S HUT in its clearing, with the poison joke beds (shot B).
  3. FROGGY BOTTOM BOGG and the gorge with the rope bridge (shot C).
Each area is the film's own shot renderer (`render_A/B/C`) bound to replacement globals (`rebind`): the
player drives the camera, the film's balloon and name tags are switched off (the base class draws the
player's balloon and the tags). Timed film beats run on game clocks:
  - the self-made GUST (film frame 108) blows through the canopy when the player calls it (SPACE at the
    canopy gap), and on its own now and then; the low mist keeps a continuous position (the film speeds
    it up via `f * speed`, which would jump on a free-running clock),
  - the GLIMPSE (film frame 288): SPACE at the castle parts the mist curtain, the ruined castle stands
    clear for a few seconds and the mist rolls back,
  - the little cloud raining by itself sways back and forth over the gorge rim instead of drifting away.
SPACE also makes the poison joke puff glittering spores, Zecora's cauldron bubble over and the
will-o'-wisps rise out of the bog. Three atlas stamps hide in the canopy gap, by the hut's chimney and
under a mangrove's prop roots.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, seg, smooth, hash01, mix, fx, col, clamp, bayer
from game import ui
import scenes.s05_everfree as sc

from .. import daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# camera limits (the largest camera with art on every layer):
#   A: near/canopy 480 px (p 1) -> 160; mid 420 (p .6) -> 166; fg 540 (p 1.35) -> 162; far 380 (p .3) -> 200
#   B: hut 380 (p 1) -> 60; far 340 / mid 360 (p .3 / .6) -> 66; fg 420 (p 1.3) -> 76
#   C: near 460 (p 1) -> 140; mid walls 430 (p .6) -> 183; far walls 400 (p .3) -> 266; ridge 340 (p .1) -> 200
CAM_A, CAM_B, CAM_C = 160, 60, 140

# The film's callouts, in world px. The three the film hangs below the ground line point up here
# instead (same spots), so the game's key-hint strip along the bottom edge never covers them.
LANDMARKS_A = [
    Landmark('EVERFREE FOREST', 156, (150 + 12, 90, 'right', 10)),
]
LANDMARKS_B = [
    Landmark("ZECORA'S HUT", sc.HUT_X, (sc.HUT_X - 46, 62, 'left', 10)),
    Landmark('POISON JOKE', sc.PJ_B[0][0] + 4, (sc.PJ_B[0][0] + 4, sc.GB - 9, 'up', 12)),
]
LANDMARKS_C = [
    Landmark('FROGGY BOTTOM BOGG', 150, (150, sc.WL + 2, 'up', 16)),
    Landmark('ROPE BRIDGE', 362, (362, 146, 'up', 22)),       # the bridge deck, mid-span
]

# ---- hotspots (SPACE) and hidden stamps, world px
GAP = (sc.GAPS_A[1][0], 28)                 # the sunlit near-canopy gap the gust blows through
JOKE = (sc.PJ_A[1][0] + 4, sc.PJ_A[1][1] - 4)   # the big poison-joke bed by the warning sign
CAUL = (sc.CAUL_X, sc.CAUL_Y)               # Zecora's cauldron (rim at y - 9)
WISP0 = sc.WISPS[0]                         # the leftmost will-o'-wisp over the bog
CASTLE = (sc.CASTLE_X, 92)                  # the ruined castle, far beyond the gorge (p 0.1)

HOTSPOTS_A = [Hotspot('gust', GAP[0], GAP[1], 'wind', cooldown=75),
              Hotspot('joke', JOKE[0], JOKE[1], 'spark', cooldown=90)]
HOTSPOTS_B = [Hotspot('brew', CAUL[0], CAUL[1] - 12, 'steam', cooldown=90)]
HOTSPOTS_C = [Hotspot('wisps', WISP0[0], WISP0[1], 'lantern', cooldown=100),
              Hotspot('castle', CASTLE[0], CASTLE[1], 'moon', cooldown=90, p=0.1)]
STAMPS_A = [Stamp('canopy', sc.GAPS_A[2][0], 38)]            # high in the far canopy gap
STAMPS_B = [Stamp('chimney', sc.HUT_X + 40, 33)]             # up by the chimney, in the smoke
STAMPS_C = [Stamp('roots', 196, 122)]                         # under the mangrove's prop-root arch

# ---- the gust (film: scene frame GUST = 108; ramps up over 10 frames, dies away 24..48)
GUST_LEN = 48
_GUST_CUM = np.concatenate([[0.0], np.cumsum([smooth(seg(a, 0, 10)) * (1 - smooth(seg(a, 24, 48)))
                                              for a in range(GUST_LEN)])])
_MIST_GUST = {5: (0.25, 1.2), 6: (0.5, 1.5)}      # render_A's mist(seed): (base speed, extra speed * gust)
GUST_AUTO = 540                                    # the forest makes its own gust after this many quiet steps

# ---- the glimpse (film: scene frame GLIMPSE = 288; the mist parts over 10 frames)
GLIMPSE_IN, GLIMPSE_OUT = 10, 60                   # frames to part / to close again
GLIMPSE_SHOW = 140                                 # steps the castle stands clear

# ---- the self-raining cloud: sways over the gorge rim (film: drifts 0..13 px right of x 217, p .3)
RAIN_MID, RAIN_SWAY, RAIN_PERIOD = 15, 15, 1200

# ---- interaction results (steps)
JOKE_LIFE, BREW_LIFE, WISP_LIFE = 84, 84, 96
SPORE_COLS = ('ice', 'baby_blue', 'white', 'crystal', 'light_sky', 'lavender')
BREW_COLS = ('lime', 'pale_leaf', 'green', 'mint', 'butter')
WISP_N = 12


def _noop(*_a, **_k):
    return None


class _FastCanvas(Canvas):
    """A Canvas whose blit of an RGBA layer uses np.copyto (same pixels, ~2x faster than the engine's
    masked assignment). The film's full-screen parallax layers are blitted ~10 times per frame."""
    __slots__ = ()

    def copy(self):
        return _FastCanvas(arr=self.a.copy())

    def blit(self, src, x, y, flip=False):
        if flip or not isinstance(src, np.ndarray) or src.ndim != 3 or src.shape[2] != 4:
            return Canvas.blit(self, src, x, y, flip)
        x, y = int(round(x)), int(round(y))
        sh, sw = src.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.a.shape[1], x + sw), min(self.a.shape[0], y + sh)
        if x1 <= x0 or y1 <= y0:
            return None
        sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
        np.copyto(self.a[y0:y1, x0:x1], sub[:, :, :3], where=sub[:, :, 3:] >= 128)
        return None


def _fast_bg(bg):
    """The scene's cached backdrop, wrapped as a _FastCanvas (the scene renders draw on its .copy())."""
    return lambda: _FastCanvas(arr=bg().a)


_BG_A, _BG_B, _BG_C = _fast_bg(sc.bgA), _fast_bg(sc.bgB), _fast_bg(sc.bgC)


# ======================================================================= sky hooks (game/daylight.py)
# Each backdrop (bgA/B/C) is the sky gradient on top and the under-canopy air (or the low marsh haze)
# below it; only the sky rows are given (the air is depth between the trees, not sky), the rest is a
# key colour no painted pixel is near. Screen-fixed, like the backdrops. The sun shafts that start in
# the sky (path, hut) are added as the render adds them, so they are repainted with the sky; over the
# bog the far ridge, the gorge walls, the castle and the mist curtain are keyed out (they are within
# the sky test's tolerance of the pale sky behind them).
_NOT_SKY = np.array((255, 0, 255), np.uint8)
SHAFT = mix('butter', 'pale_leaf', 0.35)


def _sky_rows(bg, rows):
    c = Canvas(arr=bg().a.copy())
    c.a[rows:] = _NOT_SKY
    return c


_SKY_A = _sky_rows(sc.bgA, 26)          # bgA: SKY 0..30, the air gradient from row 26
_SKY_B = _sky_rows(sc.bgB, 50)          # bgB: SKY 0..60, the air from row 50
_SKY_C = _sky_rows(sc.bgC, 96)          # bgC: sky 0..100, the marsh haze from row 96


def shaft_k(hour):
    """How strong the sun shafts through the canopy are at a clock hour: full while the clock's sun stands
    high, fading as it sinks (none from sunset on, none at night; they come back after sunrise)."""
    kind, _x, y = daylight._body_pos(hour)
    if kind != 'sun':
        return 0.0
    up = (70 - y) / 58.0                              # the sun's height on its arc, 0 (horizon) .. 1
    return clamp(min((up - 0.12) / 0.33, 1.0 - daylight.night(hour) * 1.25))


_PAINTED_SUN_X = daylight._body_pos(9.88)[1]       # the film's morning sun (left of centre)


def shaft_lean(hour):
    """Which way the shafts slant, from the clock's sun: 1 as painted (the morning sun on the left: down
    to the right), 0 at noon (straight down), negative in the afternoon (the sun on the right)."""
    kind, x, _y = daylight._body_pos(hour)
    if kind != 'sun':
        return 1.0
    return clamp((W / 2 - x) / (W / 2 - _PAINTED_SUN_X), -1.3, 1.3)


def _gust_k(age):
    """render_A's gust strength `age` steps after the gust started."""
    return smooth(seg(age, 0, 10)) * (1 - smooth(seg(age, 24, 48)))


def _shafts_A(c, f, cam, age, k, lean=1.0):
    gust = _gust_k(age)
    for i, (gx, gw) in enumerate(sc.GAPS_A):
        sx = gx - cam
        if sx < -140 or sx > W + 20:
            continue
        slope = 0.4 + 0.12 * math.sin(f / 46.0 + i * 2.1) + gust * 0.12
        sc.ray(c, sx - 2, 6, 170, gw * 0.6, gw * 1.6, slope * lean, SHAFT, 0.3 * k, fade=0.55)


def _shaft_B(c, f, cam, k, lean=1.0):
    slope = 0.66 + 0.06 * math.sin(f / 40.0)
    sc.ray(c, 182 - int(cam * 0.6), 0, 166, 22, 56, slope * lean, SHAFT, 0.26 * k)


def _key(c, rgba, x, y):
    al = rgba[..., 3] >= 128
    lh, lw = al.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + lw), min(H, y + lh)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0:y1, x0:x1], _NOT_SKY, where=al[y0 - y:y1 - y, x0 - x:x1 - x, None])


def _sky_C(f, cam, g):
    """render_C's sky: the gradient rows, with the far ridge, the castle (while the mist is parted), the
    mist curtain's pixels and the gorge walls keyed out (all pale and close to the sky)."""
    c = _SKY_C.copy()
    probe = Canvas(arr=np.zeros((H, W, 3), np.uint8))
    ccx = sc.CASTLE_X - int(cam * 0.1)
    yb = sc._YYf[28:118]
    hole = 1 - g * np.exp(-(((sc._XXf[28:118] - ccx) / (12 + 36 * g)) ** 2 + ((yb - 72) / 44.0) ** 2))
    sc.mist(probe, f, 28, 118, mix('mint_cream', 'baby_blue', 0.25), 0.3, 25, dens=1.0 + 0.9 * (1 - g),
            a1=0.55, a2=0.9, hole=hole)
    c.a[probe.a.any(axis=2)] = _NOT_SKY
    _key(c, sc.farC(), -int(cam * 0.1), 0)
    if g > 0:
        spr = sc.castle_sprite()
        x0, y0 = ccx - 30, 98 - 64
        sub = c.a[max(0, y0):y0 + 64, x0:x0 + 60]
        th = bayer(64, 60, x0, y0)[-sub.shape[0]:, :sub.shape[1]]
        sub[(spr[-sub.shape[0]:, :sub.shape[1], 3] > 0) & (th < g)] = _NOT_SKY
    _key(c, sc.wallsC(0), -int(cam * 0.3), 0)
    _key(c, sc.wallsC(1), -int(cam * 0.6), 0)
    return c


# ======================================================================= glow kept at night
# The relight's own keeping of small warm / magic spots (auto_glow) is off here: it kept the hut's painted
# masks, bells and bead curtain glowing like lamps. The real lights are marked by the area renders instead
# (GlowKeep): poison joke and glowshrooms, the will-o'-wisps, Zecora's doorway, window, lantern, fire and
# brew, and the SPACE effects' magic.
_YY, _XX = np.mgrid[0:H, 0:W]


def _disc_mask(x, y, r):
    return (_XX - x) ** 2 + (_YY - y) ** 2 <= r * r + r * 0.8


def _cool_glow(c, y0=0):
    """The bright cool magic hues (turquoise, ice, baby blue: poison joke, glowshrooms, wisps, spores)
    below row y0 (screen mask). Mint mist, sage and the painted greens are not."""
    a = c.a.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (np.maximum(g, b) > 170) & (b > r + 50) & (b >= g - 45)
    m[:y0] = False
    return m


def _bright_new(c, before, lum=150):
    """Pixels an effect just drew (differ from `before`) that are bright: its sparks, not its halo."""
    a = c.a.astype(np.int32)
    return (c.a != before).any(axis=2) & ((a[..., 0] * 77 + a[..., 1] * 150 + a[..., 2] * 29) >> 8 > lum)


def _door_mask(cam):
    dx, dy = sc.DOOR[0] - cam, sc.DOOR[1]
    m = _disc_mask(dx, dy, 9)
    m[dy:sc.GB, max(0, dx - 9):max(0, dx + 10)] = True
    return m


def _window_mask(cam):
    return _disc_mask(sc.WIN_B[0] - cam, sc.WIN_B[1], 5)


def _hut_glow(c, cam):
    """Zecora's hut by night: the lit doorway and round window, the cauldron's fire and its bright green
    brew and steam (screen mask)."""
    m = _door_mask(cam) | _window_mask(cam)
    x0, x1 = max(0, sc.CAUL_X - cam - 14), max(0, sc.CAUL_X - cam + 48)
    sub = c.a[0:sc.CAUL_Y - 8, x0:x1].astype(np.int16)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    m[0:sc.CAUL_Y - 8, x0:x1] |= (g > 150) & (g > r + 20) & (g > b + 20)     # brew-green steam and foam
    fx0, fx1 = max(0, sc.CAUL_X - cam - 9), max(0, sc.CAUL_X - cam + 10)
    sub = c.a[sc.GB - 12:sc.GB + 1, fx0:fx1].astype(np.int16)
    m[sc.GB - 12:sc.GB + 1, fx0:fx1] |= (sub[..., 0] > 220) & (sub[..., 1] > 80) & (sub[..., 2] < 140)  # fire
    return m


# Zecora's lantern: the film's sprite (lit glass) at night, plain smoky glass by day
LANTERN_DARK = sc.LANTERN.recolor({col('gold'): mix('copper', 'stone', 0.45),
                                   col('butter'): mix('khaki', 'stone', 0.35)})
_HUT_HALOS = {('amber', 16), ('gold', 11), ('amber', 14)}      # lantern, window, doorway: lamp light


# ======================================================================= the air under the sky
# Under each backdrop's sky rows is painted air: the haze between the trunks, the marsh haze over the
# bog, and over the bog the mist curtain veiling the sky. The clock repaints only the sky, so the painted
# air kept the morning's colour: a hard band under a blue noon, a pink dusk or a night sky, and the bog's
# veils glowed pale on the night. The pixels that are only backdrop (+ mist) take the clock's change of sky
# colour: the bog's haze and curtain fully, the forest air fading out over AIR_FADE rows under the sky.
SKY_END = (26, 50, 96)                  # the first air row of bgA / bgB / bgC
SKY_H = (30, 60, 100)                   # ...and the rows their sky gradient spans (from the top)
AIR_FADE = (16, 18, None)               # None: every veil pixel takes the whole change
VEIL_HUE = 0.4                          # ...of its brightness, and this much of its hue


@lru_cache(maxsize=32)
def _sky_shift(ai, from_h, hour10, hq):
    """(H, 3) float32: per screen row, the clock's sky (row mean) minus the relit painted sky, as much as
    paint_sky shows it (sky_weight); rows under the sky take the last sky row's."""
    hour = hour10 / 10.0
    n = SKY_END[ai]
    bg = (sc.bgA, sc.bgB, sc.bgC)[ai]().a
    painted = bg[:n].copy()
    daylight.relight(painted, from_h, hour, glow=False)
    strip = daylight._sky_strip(hq, 0, SKY_H[ai])                  # as EverfreeForest._resky paints it
    rows = np.minimum(np.arange(H), n - 1)
    clock = strip[rows].astype(np.float32).mean(axis=1)
    mine = painted.astype(np.float32).mean(axis=1)[np.minimum(np.arange(H), n - 1)]
    return (clock - mine) * daylight.sky_weight(from_h, hour)


def _sky_low(ai, from_h, hour10, hq):
    """(3,) float32: the sky's colour just above the air (its last row) as the clock shows it."""
    n = SKY_END[ai]
    bg = (sc.bgA, sc.bgB, sc.bgC)[ai]().a
    painted = bg[n - 1:n].copy()
    daylight.relight(painted, from_h, hour10 / 10.0, glow=False)
    return painted.astype(np.float32).mean(axis=1)[0] + _sky_shift(ai, from_h, hour10, hq)[n - 1]


def lamp_k(hour):
    """0 (day) .. 1: how far the hut's lamps are on (the world's lamps' ramp: on from about 19:10)."""
    from .. import nightwin
    return nightwin.ramp(hour)


def _lit_doorway(c, cam, k):
    """At night the doorway and the round window glow with the fire inside: their interior is lifted
    towards amber (top) .. butter (bottom), k of the way."""
    if k <= 0.01:
        return
    for m, top, bot in ((_door_mask(cam), mix('amber', 'rust', 0.35), col('butter')),
                        (_window_mask(cam), col('amber'), col('butter'))):
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        u = ((ys - ys.min()) / max(1, ys.max() - ys.min()))[:, None]
        warm = np.array(top, np.float32) * (1 - u) + np.array(bot, np.float32) * u
        old = c.a[ys, xs].astype(np.float32)
        lit = np.clip(old * 0.45 + warm * 0.75, 0, 255)
        c.a[ys, xs] = (old + (lit - old) * k).astype(np.uint8)


# ======================================================================= glow kept at night
class GlowKeep:
    """Region mixin: things that glow by themselves (a lit doorway, a magic brew, a rainbow) keep
    shining when the world clock darkens the place. The area's render marks them at its very end with
    `self.keep_glow(c, mask)`; right after the base `_world` has re-lit the frame (Region.after_light,
    before the stamps and hotspot effects are drawn over it), those pixels (where the render drew nothing
    over them since) get back most of their painted colour. The base relight keeps only small warm or
    magic-hued spots; big glowing shapes and greens need this."""
    _glow = None

    def keep_glow(self, c, mask, k=0.8):
        if mask is None or not mask.any():
            return
        if self._glow is None:
            self._glow = [c.a.copy(), mask.copy(), k]
        else:
            self._glow[0] = c.a.copy()
            self._glow[1] |= mask

    def _world(self, t, cam=None, with_balloon=False):
        self._glow = None                       # (nothing left over from a render outside _world)
        return super()._world(t, cam, with_balloon)

    def after_light(self, c, t):
        g, self._glow = self._glow, None
        if g is None or (self.ID, self.ai) in daylight.INTERIORS:
            return
        pre, mask, k = g
        hour = self.g.hour
        kk = k * clamp((daylight.night(hour) - daylight.night(self.HOUR)) * 1.5)
        if kk < 0.02:
            return
        ys, xs = np.nonzero(mask)
        y0, y1 = max(0, ys.min() - 4), min(H, ys.max() + 5)
        x0, x1 = max(0, xs.min() - 4), min(W, xs.max() + 5)
        p = pre[y0:y1, x0:x1]
        e = p.copy()
        daylight.relight(e, self.HOUR, hour, t, snow=self.ID in daylight.SNOWY,   # what the base made
                         glow=self.auto_glow())                                  # of these pixels
        sub = c.a[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1] & (sub == e).all(axis=2)          # ...and nothing drawn over them since
        if m.any():
            ef = e[m].astype(np.float32)
            sub[m] = (ef + (p[m].astype(np.float32) - ef) * kk).astype(np.uint8)


# ======================================================================= interaction effects (screen px)
def joke_fx(c, a, x, y):
    """The poison joke puffs: a cold blue flare, a pop of glitter and a column of spores drifting up."""
    k = 1 - a / JOKE_LIFE
    sc.add_glow(c, x, y - 4, 16 + min(18, a // 2), 'baby_blue', 0.55 * k)
    sc.add_glow(c, x, y - 14 - min(30, a // 2), 12, 'ice', 0.3 * k)
    if a < 12:
        sc.add_glow(c, x, y - 3, 10, 'white', 0.7 * (1 - a / 12))
    ui.burst(c, a, 0, x, y - 4, colors=SPORE_COLS, n=26, seed=31, power=0.5, life=42, gravity=0.025)
    for i in range(44):
        s = int(hash01(i, 3) * 20)
        age = a - s
        if age < 0 or age > JOKE_LIFE - s - 2:
            continue
        h = hash01(i, 4)
        spread = 4 + age * 0.22
        px = x + (hash01(i, 5) - 0.5) * 2 * spread + math.sin(age * 0.11 + i * 1.7) * (2 + age * 0.06)
        py = y - 4 - age * (0.55 + 0.85 * h)
        cc = SPORE_COLS[i % len(SPORE_COLS)]
        if (age + i) % 9 < 7:
            c.px(int(px), int(py), cc)
            if i % 3 == 0:
                c.px(int(px) + 1, int(py), cc)
        if i % 4 == 0 and (age // 3 + i) % 3 == 0:          # a few twinkle as little crosses
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c.px(int(px) + dx, int(py) + dy, 'white')
    if a < 64:
        fx.sparkles(c, x, y - 16, a, radius=20, n=8, colors=('white', 'ice', 'baby_blue'), seed=33, period=20)


def leaf_shower(c, a, f):
    """The gust tears leaves loose from the canopy: they tumble down across the screen, blown right."""
    for i in range(40):
        s = int(hash01(i, 41) * 16)
        age = a - s
        if not 0 <= age < 76:
            continue
        h = hash01(i, 42)
        x = -30 + hash01(i, 43) * (W + 20) + age * (1.3 + 1.6 * h) + math.sin(age * 0.2 + i) * 4
        y = 12 + hash01(i, 44) * 26 + age * (0.8 + 0.7 * hash01(i, 45)) + math.sin(age * 0.3 + i * 2) * 2
        if -4 < x < W + 4 and y < H:
            sc.big_leaf(c, x, y, f, i, sc.GUST_COLS[i % len(sc.GUST_COLS)])


def _gust_leaves(c, f, t0, n=56, seed=7):
    """The film's whirling gust stream, thickened with a second, later stream."""
    sc.gust_leaves(c, f, t0, n, seed)
    sc.gust_leaves(c, f, t0 + 7, 36, 19)


def brew_fx(c, a, x, y):
    """Zecora's cauldron bubbles over: foam swells over the rim and dribbles down, green drops fly, a big
    green plume rolls up and the clearing flashes lime."""
    k = 1 - a / BREW_LIFE
    sc.add_glow(c, x, y - 12, 28, 'lime', 0.42 * k)
    sc.add_glow(c, x, sc.GB - 3, 16, 'amber', 0.3 * k)
    rim = y - 10
    # the rolling plume: big two-tone puffs swelling as they rise, thinning out at the top
    for i in range(10):
        age = a - i * 4
        if not 0 <= age < 64:
            continue
        tt = age / 64
        px = x - 2 + int(tt * 26 + math.sin(tt * 5 + i * 1.3) * (3 + 5 * tt))
        py = rim - 4 - int(tt * 92)
        r = int(3 + tt * 7)
        body = mix('lime', 'pale_leaf', 0.3 + 0.5 * tt)
        if tt < 0.6:
            sc.disc(c, px, py, r, mix(body, 'green', 0.35))
            sc.disc(c, px - 1, py - 1, r - 1, body)
            sc.disc(c, px - 2, py - 2, max(0, r - 3), mix('pale_leaf', 'mint_cream', 0.5))
        else:
            lvl = 1 - (tt - 0.6) / 0.4
            for j in range(10):
                ang = hash01(i, j) * 6.283
                rr = r * hash01(i, j, 2)
                if hash01(i, j, 3) < lvl:
                    sc.disc(c, px + math.cos(ang) * rr, py + math.sin(ang) * rr, 1.5, body)
    # foam swelling over the rim, then sinking back
    fh = smooth(seg(a, 0, 8)) * (1 - smooth(seg(a, 44, 64)))
    if fh > 0.05:
        for i in range(8):
            fx_ = x - 12 + i * 3.4
            r = (2 + 2 * hash01(i, 7)) * fh + 0.5
            fy = rim - fh * (2 + 3 * math.sin(i * 0.9 + 0.4) ** 2) - 0.5 * math.sin(a * 0.5 + i)
            sc.disc(c, fx_, fy, r, mix('lime', 'green', 0.3))
            sc.disc(c, fx_ - 0.5, fy - 0.8, max(0.0, r - 1), 'lime')
            if r > 2:
                c.px(int(fx_ - 1), int(fy - r + 1), 'butter')
        for side in (-1, 1):                                  # overflow dribbling down the pot's bulge
            for j, (dx, L0) in enumerate(((13, 8), (11, 5))):
                L = int(min(a - 3 - j * 4, L0) * fh)
                if L > 0:
                    xx = x + side * dx
                    c.vline(xx, rim + 1, rim + L, 'lime')
                    c.px(xx, rim + L + 1, mix('lime', 'green', 0.5))
                    c.px(xx + side, rim + 1, mix('lime', 'green', 0.3))
    ui.burst(c, a, 0, x, rim - 3, colors=BREW_COLS, n=26, seed=21, power=0.75, life=46, gravity=0.07)


def wisps_fx(c, a, cam):
    """The bog stirs: rings spread on the water and a flock of will-o'-wisps rises out of it, circles
    up under the canopy and fades."""
    for i in range(WISP_N):
        s = int(hash01(i, 1) * 16)
        age = a - s
        if age < 0:
            continue
        x0 = 16 + hash01(i, 2) * 226
        if age < 24:                                         # rings where each one surfaces
            r = 2 + age // 3
            c.ellipse(int(x0 - cam), sc.WL + 4, r + 1, max(1, r // 2), mix('pale_leaf', 'turquoise', 0.4),
                      fill=False)
        life = WISP_LIFE - s
        if age >= life:
            continue
        rise = 1 - (1 - min(1.0, age / 50)) ** 2
        top = 60 + hash01(i, 3) * 50
        y = sc.WL + 2 - (sc.WL + 2 - top) * rise + 3 * math.sin(age * 0.2 + i)
        x = x0 - cam + (6 + 8 * rise) * math.sin(age * 0.13 + i * 2.1)
        k = min(1.0, age / 6) * (1 - seg(age, life - 22, life))
        if k <= 0.03:
            continue
        sc.add_glow(c, x, y, 11, 'turquoise', 0.42 * k)
        xi, yi = int(x), int(y)
        c.px(xi, yi, 'white')
        if k > 0.4:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c.px(xi + dx, yi + dy, 'ice')
        for tr in range(1, 4):                               # a short glittering tail
            if hash01(i, tr, a // 3) > 0.35:
                c.px(xi - int(round(math.cos(age * 0.13 + i * 2.1) * tr * 1.5)), yi + tr + 1, 'turquoise')


class EverfreeForest(GlowKeep, Region):
    SEG = 'everfree'
    CARD = ('WILD FOREST', ['WEATHER RUNS ITSELF', 'TREAD CAREFULLY'])

    def build_areas(self):
        return [
            Area(self.render_path, cam_x=(0, CAM_A), y_range=(40, 130), start=(92, 84),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=('#7d9a88', 0.4),
                 hotspots=HOTSPOTS_A, stamps=STAMPS_A, sky=self.sky_path),
            Area(self.render_hut, cam_x=(0, CAM_B), y_range=(40, 130), start=(64, 62),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=('#b2c2a6', 0.25),
                 hotspots=HOTSPOTS_B, stamps=STAMPS_B, sky=self.sky_hut),
            Area(self.render_bog, cam_x=(0, CAM_C), y_range=(40, 136), start=(110, 80),
                 landmarks=LANDMARKS_C, balloon='m', balloon_tint=('#9db2ae', 0.3),
                 hotspots=HOTSPOTS_C, stamps=STAMPS_C, sky=self.sky_bog),
        ]

    def init_state(self):
        self.glimpse = None         # [step, k at that step, opening?]: the castle glimpse
        self.gust_t0 = None         # step the latest gust started
        self.gust_base = 0.0        # gust strength summed over all earlier gusts (mist offset)

    def on_area(self, ai):
        t = self.g.t
        self.glimpse = None                                  # the mist has closed while you were away
        if ai == 0 and (self.gust_t0 is None or t - self.gust_t0 > GUST_LEN):
            self.gust_base = self.gust_integral(t)
            self.gust_t0 = t - GUST_AUTO + 210               # the forest's own gust, ~7 s after arriving

    # ------------------------------------------------------------ clocks
    def gust_age(self, t):
        return 10 ** 4 if self.gust_t0 is None else t - self.gust_t0

    def gust_integral(self, t):
        """Gust strength summed over all steps up to t (keeps the mist's position continuous)."""
        return self.gust_base + _GUST_CUM[int(np.clip(self.gust_age(t), 0, GUST_LEN))]

    def start_gust(self, t, gain=0.6):
        self.gust_base = self.gust_integral(t)
        self.gust_t0 = t
        self.g.sfx('wind', gain, 1)                          # howling gust

    def glimpse_k(self, t):
        """0..1 how far the mist curtain has parted (film: ease_out(seg(f, 286, 296)))."""
        if self.glimpse is None:
            return 0.0
        t0, k0, opening = self.glimpse
        if opening:
            return k0 + (1 - k0) * sc.ease_out(seg(t, t0, t0 + GLIMPSE_IN))
        return k0 * (1 - smooth(seg(t, t0, t0 + GLIMPSE_OUT)))

    def glimpse_since(self, t):
        """Steps since the curtain started parting, or -1 while it is closed / closing (castle glint)."""
        if self.glimpse is None or not self.glimpse[2] or t < self.glimpse[0]:
            return -1
        return t - self.glimpse[0]

    def spot_age(self, key, t):
        return t - self.spot_t.get(key, -10 ** 6)

    # ------------------------------------------------------------ events
    def on_interact(self, key, t):
        g = self.g
        if key == 'gust':
            g.sfx('whoosh', 0.45)
            self.start_gust(t, 0.65)
        elif key == 'joke':
            g.sfx('shimmer', 0.45)
            g.sfx('pop', 0.3, 1)
        elif key == 'brew':
            g.sfx('bubbles', 0.6)
            g.sfx('pop', 0.45, 2)
        elif key == 'wisps':
            g.sfx('bubbles', 0.45)
            g.sfx('crystal', 0.35)
        elif key == 'castle':
            self.glimpse = [t + 4, self.glimpse_k(t), True]  # the mist curtain parts
            g.sfx('wind', 0.45)

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.4)
        if self.ai == 0:
            if self.gust_age(t) >= GUST_AUTO:
                self.start_gust(t, 0.5 * self.near_gain(self.cx + 160))
            if self.spot_age('gust', t) == 28:
                g.sfx('shimmer', 0.2)                         # the film's poison-joke twinkle after the gust
        elif self.ai == 1:                                    # the cauldron (film: bubbles, steam, 3 pops)
            gain = self.near_gain(sc.CAUL_X)
            a = t % 216
            if a == 0:
                g.sfx('bubbles', 0.45 * gain)
            elif a in (36, 54, 72) and gain > 0.4:
                g.sfx('pop', (0.4 - (a - 36) / 360) * gain, (a // 18) % 4)
            elif a == 126:
                g.sfx('steam', 0.3 * gain)
            if self.spot_age('brew', t) == 14:
                g.sfx('steam', 0.45)
        elif self.ai == 2:
            if t % 192 == 0:
                g.sfx('bubbles', 0.35 * self.near_gain(150))
            if t % 108 == 54:
                g.sfx('creak', 0.45 * self.near_gain(362))
            gl = self.glimpse
            if gl is not None:
                if gl[2] and t == gl[0] and gl[1] < 0.5:
                    g.sfx('shimmer', 0.45)                    # the castle appears
                elif gl[2] and t - gl[0] > GLIMPSE_SHOW:
                    self.glimpse = [t, self.glimpse_k(t), False]   # ... and the mist rolls back
                elif not gl[2] and t - gl[0] > GLIMPSE_OUT:
                    self.glimpse = None

    # ------------------------------------------------------------ sky hooks + clock light
    def sky_path(self, f, cx, cy):
        k = shaft_k(self.g.hour)
        if k < 0.02:
            return _SKY_A
        c = _SKY_A.copy()
        _shafts_A(c, f, cx, self.gust_age(self.g.t), k, shaft_lean(self.g.hour))
        c.a[26:] = _NOT_SKY
        return c

    def sky_hut(self, f, cx, cy):
        k = shaft_k(self.g.hour)
        if k < 0.02:
            return _SKY_B
        c = _SKY_B.copy()
        _shaft_B(c, f, cx, k, shaft_lean(self.g.hour))
        c.a[50:] = _NOT_SKY
        return c

    def sky_bog(self, f, cx, cy):
        return _sky_C(f, cx, self.glimpse_k(self.g.t))

    def _shafts(self):
        """The scene's ray / ray_motes, by the clock's sun: strong while it stands high, slanting away
        from it (the painted morning slant in the morning, straight down at noon, the other way in the
        afternoon), fading as it sinks, none from sunset to sunrise."""
        k = shaft_k(self.g.hour)
        lean = shaft_lean(self.g.hour)

        def ray(c, *a, **kw):
            if k >= 0.02:
                a = list(a)
                a[5] *= lean                                  # slope
                a[7] *= k                                     # strength
                sc.ray(c, *a, **kw)

        def ray_motes(c, *a, **kw):
            if k >= 0.35:
                a = list(a)
                a[4] *= lean
                sc.ray_motes(c, *a, **kw)
        return ray, ray_motes

    def auto_glow(self):
        return False                # the real lights are kept by the area renders (GlowKeep masks)

    _veil = None                    # (area, mask) of the backdrop-only pixels the clock's sky tints

    def _world(self, t, cam=None, with_balloon=False):
        self._veil = None
        return super()._world(t, cam, with_balloon)

    def _resky(self, c, t):
        """The clock's sky again, its gradient spanning the backdrop's own sky rows (the base strip is 130
        rows tall: its horizon colours never reached these short skies, so a mid-sky blue or red sat on
        the pale air under the canopy)."""
        memo, hour = self._sky_memo, self.g.hour
        if memo is None or memo[0][:2] != (self.ai, t) or abs(hour - self.HOUR) < 0.05:
            return
        w = daylight.sky_weight(self.HOUR, hour)
        (_ai, _t, ix, iy), mask = memo
        if w > 0.01 and mask.any():
            daylight.paint_sky(c.a, mask, hour, w, ix, iy, 0, SKY_H[self.ai], t)

    def after_light(self, c, t):
        self._resky(c, t)
        g = self._glow
        super().after_light(c, t)                           # GlowKeep: the lights get their colour back
        v, self._veil = self._veil, None
        if v is None:
            return
        ai, mask = v
        hour = self.g.hour
        if abs(hour - self.HOUR) < 0.05 or ai != self.ai:
            return
        wt = mask if mask.dtype != bool else None           # a weight per pixel (the bog) or a mask
        mask = mask > 0
        if g is not None:
            mask = mask & ~g[1]
        sky = self._sky_memo[1] if self._sky_memo is not None else None
        if sky is not None:
            mask = mask & ~sky                              # (the open sky itself is the clock's already)
        n, fade = SKY_END[ai], AIR_FADE[ai]
        if fade is not None:
            mask[n + fade:] = False
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return
        key = (ai, self.HOUR, int(round(hour * 10)), daylight.sky_q(hour))
        px = c.a[ys, xs].astype(np.float32)
        if fade is None:                                    # veils: the sky's change shows through them
            d = _sky_shift(*key)
            lum = d.mean(axis=1, keepdims=True)             # (its brightness fully, its hue in part: a
            d = lum + (d - lum) * VEIL_HUE                  # sunset sky only warms the haze)
            px += d[ys] * (1.0 if wt is None else wt[ys, xs][:, None])
        else:                                               # the forest air melts into the sky's low
            k = (np.clip(1.0 - (ys - n + 0.5) / fade, 0, 1) ** 1.3 * 0.9)[:, None]   # colour under it
            px += (_sky_low(*key)[None] - px) * k
        c.a[ys, xs] = np.clip(px, 0, 255).astype(np.uint8)

    def _mark_air(self, c, bg, ai):
        """Mark the forest air under the sky (pixels still the backdrop's) for after_light."""
        if abs(self.g.hour - self.HOUR) < 0.05:
            return
        n = SKY_END[ai]
        m = np.zeros((H, W), bool)
        m[n:n + AIR_FADE[ai]] = (c.a[n:n + AIR_FADE[ai]] == bg[n:n + AIR_FADE[ai]]).all(axis=2)
        self._veil = (ai, m)

    def _glows(self, c, mask, extra=None):
        """Keep the area's real lights (mask) and an effect's sparks (extra) shining at night."""
        if extra is not None:
            mask = mask | extra
        if daylight.night(self.g.hour) > 0.02:
            self.keep_glow(c, mask, 0.95)

    def _halo(self, keep=()):
        """The scene's add_glow, by the clock: the lamps' halos (`keep`: their (colour, radius)) never
        (lamplight by day; at night only the lamp itself shines), the fire / brew / magic halos fading out
        as the lamps come on (at night a darkened halo is only a muddy smudge; the light itself is kept)."""
        k = 1.0 - lamp_k(self.g.hour)

        def add_glow(c, x, y, r, color, s):
            if (color, int(r)) in keep or k <= 0.02:
                return
            sc.add_glow(c, x, y, r, color, s * k)
        return add_glow

    # ------------------------------------------------------------ areas
    def render_path(self, f, cx, cy):
        t = self.g.t
        age = self.gust_age(t)
        gi = self.gust_integral(t)

        def mist(c, f_, y0, y1, color, speed, seed, **kw):
            base, k = _MIST_GUST.get(seed, (speed, 0.0))
            return sc.mist(c, f_, y0, y1, color, base, seed, off=k * gi, **kw)

        ray, motes = self._shafts()
        render = rebind(sc.render_A, bgA=_BG_A, camA=lambda _f: cx, draw_balloon=_noop, callout=_noop,
                        GUST=f - age, mist=mist, gust_leaves=_gust_leaves, ray=ray, ray_motes=motes,
                        add_glow=self._halo())
        c = render(f)
        if age < 92:
            leaf_shower(c, age, f)
        a = self.spot_age('joke', t)
        spark = None
        if 0 <= a < JOKE_LIFE:
            before = c.a.copy()
            joke_fx(c, a, JOKE[0] - cx, JOKE[1] + 4)
            spark = _bright_new(c, before)
        self._glows(c, _cool_glow(c, 60), spark)              # poison joke, glowshrooms, spores
        self._mark_air(c, sc.bgA().a, 0)
        return c

    def render_hut(self, f, cx, cy):
        ray, motes = self._shafts()
        lit = lamp_k(self.g.hour) > 0.05
        render = rebind(sc.render_B, bgB=_BG_B, camB=lambda _f: cx, draw_balloon=_noop, callout=_noop,
                        ray=ray, ray_motes=motes, add_glow=self._halo(_HUT_HALOS),
                        LANTERN=sc.LANTERN if lit else LANTERN_DARK)
        c = render(f)
        _lit_doorway(c, cx, lamp_k(self.g.hour))
        a = self.spot_age('brew', self.g.t)
        spark = None
        if 0 <= a < BREW_LIFE:
            before = c.a.copy()
            brew_fx(c, a, CAUL[0] - cx, CAUL[1])
            spark = _bright_new(c, before)
        cool = _cool_glow(c, 120)                          # poison joke (not the potion jars: paint)
        cool[128:143, max(0, 150 - cx):max(0, 250 - cx)] = False
        mask = _hut_glow(c, cx) | cool
        if lit:
            mask |= _lantern_mask(f, cx)
        self._glows(c, mask, spark)
        self._mark_air(c, sc.bgB().a, 1)
        return c

    def render_bog(self, f, cx, cy):
        t = self.g.t
        k = self.glimpse_k(t)
        since = self.glimpse_since(t)
        sway = RAIN_MID + RAIN_SWAY * math.sin(2 * math.pi * t / RAIN_PERIOD) + 0.5
        hour = self.g.hour
        shift = _sky_shift(2, self.HOUR, int(round(hour * 10)), daylight.sky_q(hour))
        veil = None                 # the backdrop with only the mists on it (for the veils, after_light)
        if abs(hour - self.HOUR) >= 0.05 and np.abs(shift).max() >= 3:
            veil = sc.bgC().a.copy()

        def mist(c_, *a, **kw):
            return _mist2(c_, veil, *a, **kw)

        render = rebind(sc.render_C, bgC=_BG_C, camC=lambda _f: cx, draw_balloon=_noop, callout=_noop,
                        ease_out=lambda _x: k,
                        GLIMPSE=f - since if since >= 0 else f + 1,
                        CUT2=f - sway / 0.12, mist=mist, add_glow=self._halo())
        c = render(f)
        a = self.spot_age('wisps', t)
        spark = None
        if 0 <= a < WISP_LIFE:
            before = c.a.copy()
            wisps_fx(c, a, cx)
            spark = _bright_new(c, before)
        self._glows(c, _wisp_mask(c, f, cx), spark)
        if veil is not None:
            # the veils for after_light: the haze and the mist curtain (backdrop + mist only) wholly, the
            # far ridge and gorge walls in part (aerial haze: the farther, the more)
            wm = np.zeros((H, W), np.float32)
            for lay, x, k in ((sc.farC(), -int(cx * 0.1), 0.8), (sc.wallsC(0), -int(cx * 0.3), 0.55),
                              (sc.wallsC(1), -int(cx * 0.6), 0.3), (sc.nearC(), -cx, 0.0),
                              (sc.lilyC(), -cx, 0.0), (sc.fgC(), -int(cx * 1.3), 0.0)):
                _put_alpha(wm, lay, x, k)
            wm[(c.a == veil).all(axis=2)] = 1.0
            self._veil = (2, wm)
        return c


def _mist2(c, veil, f, y0, y1, color, speed, seed, dens=1.0, a1=0.28, a2=0.5, off=0, hole=None):
    """sc.mist (same pixels) on the canvas c and, with the same masks, on the array `veil` (or None)."""
    if veil is None:
        return sc.mist(c, f, y0, y1, color, speed, seed, dens, a1, a2, off, hole)
    y0, y1 = max(0, y0), min(H, y1)
    h = y1 - y0
    if h <= 1:
        return
    nz = sc._noise(256, h, seed, max(1.0, h / 5.0), 11.0)
    xs = (np.arange(W) + int(off + f * speed)) % 256
    d = nz[:, xs]
    prof = np.sin(np.pi * (np.arange(h) + 0.5) / h) ** 0.7
    L = np.clip((d - 0.28) / 0.5, 0, 1) * prof[:, None] * dens
    if hole is not None:
        L = L * hole
    thr = bayer(h, W, 0, y0)
    m1 = thr < L
    m2 = thr < L - 0.45
    cc = np.array(col(color), np.float32)
    for a in (c.a, veil):
        reg = a[y0:y1].astype(np.float32)
        reg[m1] += (cc - reg[m1]) * a1
        reg[m2] += (cc - reg[m2]) * a2
        a[y0:y1] = reg.astype(np.uint8)


def _put_alpha(wm, rgba, x, k):
    """wm[pixels the layer (pasted at x, 0) covers] = k."""
    al = rgba[..., 3] >= 128
    x0, x1 = max(0, x), min(W, x + al.shape[1])
    if x1 > x0:
        h = min(H, al.shape[0])
        sub = wm[:h, x0:x1]
        sub[al[:h, x0 - x:x1 - x]] = k


def _lantern_mask(f, cam):
    """The lit glass of Zecora's swinging lantern (screen mask), as render_B draws it."""
    sw = int(round(math.sin(f / 22.0) * 1.2))
    spr = sc.LANTERN
    x0, y0 = sc.LANT_B[0] - cam + sw - spr.w // 2, sc.LANT_B[1] + 5
    rgb = spr.rgba[..., :3]
    glass = (rgb == np.array(col('gold'), np.uint8)).all(axis=2) | (rgb == np.array(col('butter'), np.uint8)).all(axis=2)
    m = np.zeros((H, W), bool)
    ys, xs = np.nonzero(glass & (spr.rgba[..., 3] > 0))
    ys, xs = ys + y0, xs + x0
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    m[ys[ok], xs[ok]] = True
    return m


def _wisp_mask(c, f, cam):
    """The will-o'-wisps' cores and glittering tails (screen mask), where render_C draws them."""
    m = np.zeros((H, W), bool)
    cool = _cool_glow(c, 0)
    for i, (wx, wy) in enumerate(sc.WISPS):
        x = wx - cam + int(round(6 * math.sin(f / 23.0 + i * 1.9)))
        y = wy + int(round(4 * math.sin(f / 17.0 + i * 2.6)))
        if not (3 <= x < W - 3 and 1 <= y < H - 6):
            continue
        m[y - 1:y + 2, x] = True
        m[y, x - 1:x + 2] = True
        m[y + 2:y + 5, x - 3:x + 4] |= cool[y + 2:y + 5, x - 3:x + 4]
    return m
