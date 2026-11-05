"""The Changeling Kingdom in the Badlands as an explorable diorama (art: scenes/s14_badlands.py, read-only).

The film shows two views joined by a dissolve, so the region has two areas, flown through left to right
with a cloud sweep between them:
  0  THE CRATER (shot A, golden hour): the hive mound in its striped-mesa crater, ringed by mauve peaks,
     with slime pools, dead trees and drifting haze. Every parallax layer has art up to camera 44 (the
     film pans 36), so the camera range is the film's framing; the balloon has the whole sky.
  1  THE CANYON AND THE HIVE'S FLANK (shot B): the Tartarus gate sealed in its shadowed canyon (west),
     the barren rocks the bloom did not reach, then the blooming hive's flank close up (vine arches with
     lanterns, round windows, crystal spires, the turquoise pool). The film pans west from the flank to
     the gate; here you arrive at the gate and fly east to the flank.
The film's render switches content by shot and runs its beats on film frames, so both areas re-compose
the scene's own cached layers and helpers at the player's camera (recompose style, see sweetapple.py),
without the film's balloon, callouts, card, iris or dissolve. Timed beats on game clocks:
  - THE TRANSFORM (film 118..206): the first landing starts with the dark hive. SPACE at the hive's
    heart (or flying on east past the hive) plays the heart's swell, a soft white bloom of light (ramped
    in and out, no tremor: comfort) and the rainbow ring sweeping out, turning chitin into crystal;
    afterwards SPACE sends another ring pulse over the bloom. The hive stays bloomed on later landings
    too (Region.lasting 'bloom', saved). The sky hook follows what is drawn (dark sky until the ring
    reaches it), so the world clock's sky never snaps while the hive transforms.
  - THE TARTARUS STING (film 236..290): the seams heat up, flare white-hot, the seal's shockwave rings
    out, dust and pebbles fall and the canyon shakes. It plays once by itself when you first reach the
    gate, then on SPACE.
  - the crater haze drifts on a seamless loop (the film's haze field only lasts its 4 bars).
Things to set off with SPACE: the hive's heart (transform / ring pulse), the big slime pool (a slime
geyser while dark, a turquoise fountain in bloom), the Tartarus seal (the sting) and the lantern arches
on the hive's flank (lanterns, windows and facets blaze, petals burst). Three atlas stamps are hidden:
up by the hive's tallest spire, low on the far side of the right-hand crater rocks, and low under the
second vine arch on the flank. No creatures anywhere (the edition's rule): landscape and light only.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, hash01, col, clamp, ease_in, ease_out, mix, shade
import scenes.s14_badlands as sc

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp
from .everfree import GlowKeep

# ----------------------------------------------------------------------------- camera limits
#   crater: near ground 358 px (p .85) -> 44; fg 373 (p 1.2) -> 44; mid 346 (p .5) -> 52; far mesas 336
#           (p .28) -> 57; peaks 332 (p .12) -> 100; sky + clouds are screen-wide
#   canyon: main 568 (p 1) -> 248; fg 645 (p 1.3) -> 250; mid 450 (p .5) -> 260; far 381 (p .22) -> 277;
#           sky 343 (p .08) -> 287
CAM_A = 44
CAM_B = 246
P_MID, P_NEAR, P_FG = sc.P_MID, sc.P_NEAR, sc.P_FG

HEART_M = (sc.HIVE_POS[0] + sc.HEART[0], sc.HIVE_POS[1] + sc.HEART[1])     # (204, 103) on the mid layer
POOL_N = sc.POOLS_A[0]                   # the big slime pool (near layer): (cx, cy, rx, ry)
POOL_JET = POOL_N[0] + 8                 # where its geyser / fountain rises (clear of the SPACE bubble)
JET_H = 54

# ----------------------------------------------------------------------------- clocks (film frames)
FORM_LEAD = 124          # transform clock at the press: swell + tremor, burst at T_FORM (144), ring to 206
FORM_END = 216
AUTO_FORM_X = 272        # flying east past the hive while it is still dark sets the transform off
FLASH_IN, FLASH_OUT = 6, 16     # the burst's white light: smoothstep in over 6 steps (from 3 before the
FLASH_PEAK = {'dark': 0.16, 'bloom': 0.10}  # burst), out over 16; peak blend (film: 0.45 / 0.3, 2 frames)
TART_LEAD = 8            # the seams heat up for 8 steps before the sting flares
TART_END = 90
AUTO_TART_DX = 90        # the first time the balloon comes this close to the gate, it stings by itself

# ----------------------------------------------------------------------------- landmarks
LANDMARKS_A = [          # shot A callouts (both on the mid layer, p .5: anchors in mid-layer px)
    Landmark('CHANGELING KINGDOM', 214, (sc.HIVE_POS[0] + 112, sc.HIVE_POS[1] + 16, 'left', 10), p=P_MID),
    Landmark('THE BADLANDS', 322, (314, 96, 'up', 14), p=P_MID),
]
LANDMARKS_B = [          # shot B (main layer = world px)
    Landmark('TARTARUS', sc.GX, (sc.GX, 20, 'down', 4)),
    Landmark('THE HIVE IN BLOOM', 462, (469, 62, 'up', 10)),
]

HOTSPOTS_A = [
    Hotspot('heart', HEART_M[0], HEART_M[1], 'heart', r=40, cooldown=100, p=P_MID),
    Hotspot('pool', POOL_N[0] - 18, POOL_N[1] - 3, 'splash', cooldown=75, p=P_NEAR),
]
HOTSPOTS_B = [
    Hotspot('tartarus', sc.EMB[0], sc.EMB[1] + 40, 'bolt', cooldown=100),
    Hotspot('lanterns', 420, 112, 'lantern', cooldown=80),
]
STAMPS_A = [Stamp('spire', 232, 22), Stamp('rocks', 340, 134)]
STAMPS_B = [Stamp('arch', 520, 134)]

ARCHES = [(420, 150, 15, 50), (520, 150, 13, 44)]          # the flank's vine arches (world px)
WINDOWS = [(470, 92, 6), (376, 112, 4), (556, 92, 5), (448, 54, 4), (500, 40, 3), (406, 70, 3)]

TINT_DARK = ('#a2a888', 0.45)            # the gloomy green-grey crater
TINT_GOLD = ('#ffcf9c', 0.28)            # golden hour over the bloom
TINT_CANYON = ('#8c7090', 0.42)          # the shadowed canyon at the gate


def _tint_between(a, b, q):
    """A balloon tint q (0..1, quantised to eighths: the compositor caches one sprite per tint) of the
    way from tint a to tint b."""
    q = round(clamp(q) * 8) / 8
    return mix(a[0], b[0], q), round(a[1] + (b[1] - a[1]) * q, 3)


# ----------------------------------------------------------------------------- small helpers
_BBOX = {}


def _comp(c, layer, x, y=0):
    """The scene's comp() (copy a cached (rgb, mask) layer to screen (x, y)), clipped to the bounding
    box of the layer's opaque pixels (same pixels, faster)."""
    a, m = layer
    bb = _BBOX.get(id(m))
    if bb is None:
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        bb = (m, rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) if rows.size else (m, 0, 0, 0, 0)
        _BBOX[id(m)] = bb
    _, r0, r1, c0, c1 = bb
    x, y = int(x) + c0, int(y) + r0
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(W, x + c1 - c0), min(H, y + r1 - r0)
    if X1 <= X0 or Y1 <= Y0:
        return
    sm = m[r0 + Y0 - y:r0 + Y1 - y, c0 + X0 - x:c0 + X1 - x]
    np.copyto(c.a[Y0:Y1, X0:X1], a[r0 + Y0 - y:r0 + Y1 - y, c0 + X0 - x:c0 + X1 - x], where=sm[:, :, None])


# ----------------------------------------------------------------------------- the world clock (game/daylight.py)
# The art is the film's golden hour (painted 18.13 h): its low sun on the LEFT, the bloom's windows, doors,
# tunnels and lanterns lit, amber halos round them, the clouds gold-rimmed. The relight keeps no painted spot
# glowing (Badlands.auto_glow): the lights are drawn by the clock (dark glass by day, lit one by one at dusk)
# and kept by mask (GlowKeep), with the magic ones (the heart, the crystal facets, Tartarus' lava seams and
# seal); no dithered halos; the clouds, the sunlight wash and the sky follow the clock.
def _ramp(x, a, b):
    """0 at a .. 1 at b, smooth."""
    u = clamp((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def _q(k, n=4):
    return round(k * n) / n


def _lamp_k(hour):
    """0 (day) .. 1 (night): how many of the lights are on. Each light has its own threshold (_thr): they
    come on one by one from 18.5 to 19.4 h (sundown into the dusk) and go out from 5.2 to 6.1 h."""
    h = hour % 24
    return clamp((h - 18.5) / 0.9) if h >= 12 else clamp((6.1 - h) / 0.9)


def _thr(x, y):
    """A light's threshold in 0.04..0.94 (lit once _lamp_k passes it)."""
    return 0.04 + 0.9 * hash01(int(x), int(y), 911)


def _day(hour):
    """0..1: full daylight, when the clouds keep no sunrise / golden-hour colour."""
    h = hour % 24
    return _ramp(h, 6.8, 8.0) * (1 - _ramp(h, 15.6, 17.0))


def _cloud_q(hour):
    """(day, night) weights in quarters for the clouds' colours (see _cloud_v)."""
    n = daylight.night(hour)
    return _q(_day(hour)), _q(clamp((n - 0.05) / 0.45))


@lru_cache(maxsize=None)
def _cloud_v(w, sd, state, dq, nq):
    """sc.cloud_sprite as the clock lights it: by day (dq of the way) the bloom's lilac / peach / gold rims
    turned to the day's white and blue-grey, the gloom's olive to a plain overcast grey; at night (nq) the
    colours greyed, so the relight dims them like the sky (no lavender clouds glowing on the night)."""
    a, m = sc.cloud_sprite(w, sd, state)
    if dq <= 0 and nq <= 0:
        return a, m
    f = a.astype(np.float32)
    lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
    if dq > 0:
        if state == 'bloom':
            tgt = np.stack([lum * 0.9 + 22, lum * 0.94 + 20, lum * 0.94 + 34], axis=-1)
        else:
            tgt = np.stack([lum * 1.05 + 8, lum * 1.07 + 10, lum * 1.1 + 18], axis=-1)
        f += (tgt - f) * dq * 0.92
    if nq > 0:
        f += (lum[..., None] * 0.86 - f) * 0.6 * nq
    return np.where(m[..., None], np.clip(f, 0, 255).astype(np.uint8), a), m


def _draw_clouds(c, f, state, specs, ox, q):
    """sc.draw_clouds with the clock's cloud colours (q: _cloud_q)."""
    for (x, y, w, sd, sp) in specs:
        lay = _cloud_v(w, sd, state, *q)
        span = W + w + 40
        _comp(c, lay, int((x + f * sp + ox) % span) - w - 20, y)


def _sun_wash(hour):
    """('left' | 'right', 0..1): the low sun's warm light washing over the crater from its side of the sky:
    the film's (from the left) in the morning's golden light, from the right in the evening's; none by day
    and by night."""
    h = hour % 24
    if h >= 12:
        return 'right', _q(_ramp(h, 16.4, 17.4) * (1 - _ramp(h, 18.8, 19.4)))
    return 'left', _q(_ramp(h, 5.8, 6.4) * (1 - _ramp(h, 7.4, 8.2)))


class _Lights:
    """The lights drawn this frame: the pixels of each mask as drawn, read back once the render is done
    (the mask GlowKeep keeps glowing at night, where nothing was drawn over them since)."""

    def __init__(self):
        self.parts = []

    def add_mask(self, c, x, y, m):
        """Record the pixels of the bool array m (placed at screen x, y) as lights."""
        h, w = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
            if mm.any():
                self.parts.append((x0, y0, x1, y1, c.a[y0:y1, x0:x1].copy(), mm))

    def add_changed(self, c, before, y0=0):
        """Record every pixel of rows y0.. that differs from `before` (a copy of those rows) as a light."""
        self.add_mask(c, 0, y0, np.any(c.a[y0:y0 + before.shape[0]] != before, axis=2))

    def mask(self, c):
        if not self.parts:
            return None
        m = np.zeros((H, W), bool)
        for (x0, y0, x1, y1, s, mm) in self.parts:
            m[y0:y1, x0:x1] |= mm & np.all(c.a[y0:y1, x0:x1] == s, axis=2)
        return m


def _vivid(a, m):
    """The crystal facets' bright magic-hued highlights in a layer (what the relight's _lights kept glowing
    at night as 'magic'): bright, saturated cyan / blue / violet / magenta, brighter than around them."""
    r, g, b = (a[..., k].astype(np.int32) for k in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = (mx > 200) & ((mx - mn) * 12 > mx * 5) & m
    v &= ~((r >= b) & (g >= b) & (r + 20 >= g))
    v &= ~((r > g) & (b * 5 < r * 4))
    v &= ~((g >= r) & (g > b + 25))
    lum = ((r * 77 + g * 150 + b * 29) >> 8).astype(np.int32)
    h, w = lum.shape
    cs = np.pad(np.pad(lum, 3, mode='edge').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    lift = lum * 49 - (cs[7:7 + h, 7:7 + w] - cs[:h, 7:7 + w] - cs[7:7 + h, :w] + cs[:h, :w])
    return v & (lift > 10 * 49)


def _recolor(a, box, mask, cmap):
    """In a[box] (x0, y0, x1, y1), the mask's pixels of the colours in cmap {name: rgb} turned to cmap's."""
    x0, y0, x1, y1 = box
    sub = a[y0:y1, x0:x1]
    for src, dst in cmap.items():
        hit = mask & np.all(sub == np.array(col(src), np.uint8), axis=2)
        sub[hit] = col(dst)


def _box(m):
    """(x0, y0, x1, y1) of a bool mask's True pixels."""
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


TUNNEL_DAY = {'amber': '#4e2e2a', 'butter': '#3a2228'}              # a tunnel / doorway in daylight: dark


@lru_cache(maxsize=1)
def _hive_lights():
    """The crater's bloomed mid layer by day (sc.mid_a('bloom') with its gold windows turned into crystal
    cavities like the other holes, its four base tunnels dark) and its lights in mid-layer px: [(x, y, mask,
    lit rgb)], the lit pixels to put back one by one at dusk; + the heart chamber's and the crystal facets'
    masks (magic: they glow at night)."""
    lit_a, m = sc.mid_a('bloom')
    a = lit_a.copy()
    g = sc.hive_geo()
    hx, hy = sc.HIVE_POS
    lights = []

    def add(hm, cmap, tg=None):
        x0, y0, x1, y1 = _box(hm)
        sub = hm[y0:y1, x0:x1]
        X0, Y0 = x0 + hx, y0 + hy
        box = (X0, Y0, X0 + x1 - x0, Y0 + y1 - y0)
        _recolor(a, box, sub, cmap)
        if tg is not None:                               # a crystal cavity, as sc.hive_sprite paints them
            reg = a[box[1]:box[3], box[0]:box[2]]
            reg[sub & ~sc.shift(hm, 0, 1)[y0:y1, x0:x1]] = col(sc.CRYSTAL[tg][0])
            reg[sub & ~sc.shift(hm, 0, -1)[y0:y1, x0:x1]] = col(sc.CRYSTAL[tg][3])
        lm = sub & np.any(lit_a[box[1]:box[3], box[0]:box[2]] != a[box[1]:box[3], box[0]:box[2]], axis=2)
        if lm.any():
            lights.append((box[0], box[1], lm, lit_a[box[1]:box[3], box[0]:box[2]].copy()))
    for i, (x, y, hw, hh) in enumerate(g['holes']):
        if hash01(41, i) > 0.3:
            continue                                     # (a crystal cavity already)
        hm = sc.arch_mask(sc.HH, sc.HW, x, y, hw, hh) & g['m']
        if hm.any():
            tg = int(g['TAG'][y, x])
            c2 = sc.CRYSTAL[tg][2]
            add(hm, {'gold': c2, 'butter': c2, 'amber': c2}, tg)
    for (x, y, hw, hh) in g['tunnels']:
        add(sc.arch_mask(sc.HH, sc.HW, x, y, hw, hh), TUNNEL_DAY)
    hxx, hyy = sc.HEART
    heart = np.zeros(m.shape, bool)
    hm = sc.arch_mask(sc.HH, sc.HW, hxx, hyy + 8, 8, 18)
    heart[hy:hy + sc.HH, hx:hx + sc.HW] = hm | (sc.shift(hm, 0, -1) & ~hm)
    hive = np.zeros(m.shape, bool)
    hive[hy:hy + sc.HH, hx:hx + sc.HW] = g['m']
    facets = _vivid(lit_a, hive & m) & ~heart
    for (x, y, lm, _rgb) in lights:
        facets[y:y + lm.shape[0], x:x + lm.shape[1]] &= ~lm
    return (a, m), tuple(lights), heart & m, facets


DARK_GLOW = (('#1f3324', '#24402a', '#2f5a32', '#3c7a3c'),          # the dark hive's openings: inside,
             ('#2f5a32', '#3c7a3c', '#4f9a40', '#5fb14a', '#6bd04b'))  # and the core, by glow strength


@lru_cache(maxsize=1)
def _dark_openings():
    """The dark hive's four base tunnels and its heart chamber: (x, y, inside, core) masks in hive px."""
    out = []
    g = sc.hive_geo()
    hx, hy = sc.HEART
    for (x, y, hw, hh) in list(g['tunnels']) + [(hx, hy + 8, 6, 16)]:
        inner = sc.arch_mask(sc.HH, sc.HW, x, y - 1, hw - 1, hh - 2)
        core = sc.arch_mask(sc.HH, sc.HW, x, y - 1, max(1, hw - 3), max(2, hh - 6))
        y0, y1 = np.nonzero(inner.any(axis=1))[0][[0, -1]]
        x0, x1 = np.nonzero(inner.any(axis=0))[0][[0, -1]]
        out.append((int(x0), int(y0), inner[y0:y1 + 1, x0:x1 + 1] & ~core[y0:y1 + 1, x0:x1 + 1],
                    core[y0:y1 + 1, x0:x1 + 1]))
    return tuple(out)


DOOR_DAY = {'amber': '#6a3430', 'gold': '#58282c', 'butter': '#482230'}   # an open doorway in daylight
PANE_DAY = {'cream': '#a898c8'}                                           # the round pane over it: glass
WIN_DAY = {'gold': '#6e5490', 'butter': '#9a88bc'}                        # a round window: violet glass


@lru_cache(maxsize=1)
def _flank_lights():
    """The canyon's main layer by day (sc.main_b with its two vine-arch doorways dark inside, their round
    panes and the six round windows glass) and its lights in layer px: [(x, y, mask, lit rgb)]; + the
    crystal facets' mask (magic: they glow at night)."""
    (lit_a, m), _pool, _lan, _gl = sc.main_b()
    a = lit_a.copy()
    lights = []

    def add(mask, cmap, keep=None):
        x0, y0, x1, y1 = _box(mask)
        sub = mask[y0:y1, x0:x1]
        rc = sub if keep is None else sub & ~keep[y0:y1, x0:x1]
        _recolor(a, (x0, y0, x1, y1), rc, cmap)
        lights.append((x0, y0, sub.copy(), lit_a[y0:y1, x0:x1].copy()))
    for (ax, ay, hw, hh) in ARCHES:
        inner = sc.arch_mask(H, sc.WB, ax, ay, hw, hh)
        door = np.zeros_like(inner)
        door[ay - 18:ay, ax - 4:ax + 5] = True           # the amber crystal door inside stays amber by day
        yy, xx = np.ogrid[0:H, 0:sc.WB]
        pane = (xx - ax) ** 2 + (yy - (ay - 28)) ** 2 <= 9
        add(inner | pane, dict(DOOR_DAY, **PANE_DAY), keep=door)
    for (wx, wy, r) in WINDOWS:
        yy, xx = np.ogrid[0:H, 0:sc.WB]
        add(((xx - wx) ** 2 + (yy - wy) ** 2 <= r * r + r) & m, WIN_DAY)
    hive = np.zeros(m.shape, bool)
    hive[:151, sc.HX0:] = True
    facets = _vivid(lit_a, hive & m)
    for (x, y, lm, _rgb) in lights:
        facets[y:y + lm.shape[0], x:x + lm.shape[1]] &= ~lm
    return (a, m), tuple(lights), facets


def _put_lights(c, lights, ox, oy, lk, rec):
    """Put back the lit pixels of the lights (layer px, the layer at screen ox, oy) the lamp ramp lk has
    switched on; recorded in rec. Returns the screen mask of what was lit (for reflections)."""
    on = np.zeros((H, W), bool) if lk > 0 else None
    for (x, y, lm, rgb) in lights:
        if lk <= _thr(x, y):
            continue
        h, w = lm.shape
        X, Y = x + ox, y + oy
        x0, y0, x1, y1 = max(0, X), max(0, Y), min(W, X + w), min(H, Y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        mm = lm[y0 - Y:y1 - Y, x0 - X:x1 - X]
        np.copyto(c.a[y0:y1, x0:x1], rgb[y0 - Y:y1 - Y, x0 - X:x1 - X], where=mm[..., None])
        rec.add_mask(c, x0, y0, mm)
        on[y0:y1, x0:x1] |= mm
    return on


def _env(e, n_in, n_out):
    """0..1..0: smoothstep up over n_in steps from e = 0, then smoothstep down over n_out steps."""
    if e < 0 or e >= n_in + n_out:
        return 0.0
    u = (e + 1) / (n_in + 1) if e < n_in else 1 - (e - n_in) / n_out
    return u * u * (3 - 2 * u)


def form_light(c, tf, hx, hy, before):
    """The transform's burst of light on c at transform clock tf (heart at screen (hx, hy)): a white glow
    on the heart and a soft white wash over everything, both swelling in over a few steps around the
    burst (T_FORM) and easing out (the film's 2-frame 0.45 white flash, softened for comfort). The sky
    hook gets the same light (Badlands.sky_crater), so the daylight sky mask holds through it."""
    e = tf - (sc.T_FORM - 3)
    g = _env(e, 5, 12)
    if g > 0.01:
        fx.glow(c, hx, hy, 40, 'white', round(0.6 * g, 3), rings=5)
    k = FLASH_PEAK[before] * _env(e, FLASH_IN, FLASH_OUT)
    if k > 0.003:
        c.blend(0, 0, W, H, 'white', round(k, 3))


def _vshift(c, s):
    """Whole-frame vertical shake by s px (edge rows repeated, so no layer edge ever shows)."""
    if s > 0:
        c.a[s:] = c.a[:-s].copy()
        c.a[:s] = c.a[s:s + 1]
    elif s < 0:
        c.a[:s] = c.a[-s:].copy()
        c.a[s:] = c.a[s - 1:s]


def _pvnoise(h, w, cy, cx, seed):
    """The scene's value noise, periodic in x (w must be a multiple of the cell size cx)."""
    R = sc.rng(seed)
    gh, gw = int(h / cy) + 3, int(round(w / cx))
    g = R.random((gh, gw))
    yy, xx = np.arange(h) / cy, np.arange(w) / cx
    y0, x0 = yy.astype(int), xx.astype(int)
    fy, fx_ = yy - y0, xx - x0
    fy = fy * fy * (3 - 2 * fy)
    fx_ = fx_ * fx_ * (3 - 2 * fx_)
    x1 = (x0 + 1) % gw
    x0 = x0 % gw
    a, b = g[y0][:, x0], g[y0][:, x1]
    c_, d = g[y0 + 1][:, x0], g[y0 + 1][:, x1]
    top = a + (b - a) * fx_[None]
    bot = c_ + (d - c_) * fx_[None]
    return top + (bot - top) * fy[:, None]


HZ_L = 480               # haze loop length (px)


@lru_cache(maxsize=1)
def _haze():
    """The film's stepped haze bands (haze_field), made seamless so they can drift forever."""
    out, amp, tot = np.zeros((H, HZ_L)), 1.0, 0.0
    for o in range(3):
        k = 2 ** o
        out += _pvnoise(H, HZ_L, max(1.0, 6 / k), max(1.0, 40 / k), 404 + 31 * o) * amp
        tot += amp
        amp *= 0.5
    n = out / tot
    yy = np.arange(H)[:, None]
    env = np.clip(1 - np.abs(yy - 96) / 46.0, 0, 1)
    k = np.clip((n - 0.45) * 1.6, 0, 1) * env * 0.42
    k = (np.round(k * 6) / 6).astype(np.float32)
    k = np.concatenate([k, k[:, :W]], axis=1)
    rows = np.flatnonzero(k.any(axis=1))
    return k[rows[0]:rows[-1] + 1, :, None], rows[0], rows[-1] + 1


_HAZE_C = np.array(col('#8b9273'), np.float32)


@lru_cache(maxsize=1)
def _ring_grid():
    """Distance (y stretched 1.1 like the film) and ring-colour angle around a centre at (W, H)."""
    yy, xx = np.mgrid[0:2 * H, 0:2 * W]
    d = np.hypot(xx - W, (yy - H) * 1.1).astype(np.float32)
    ang = (np.arctan2(yy - H, xx - W) / 6.283 * 8).astype(np.float32)
    return d, ang


_RING_RGB = sc.ramp_arr(sc.RING_COLS)
_WHITE = np.array(col('white'), np.uint8)


# ============================================================================= area 0: the crater
def compose_crater(f, state, cam, hour, rec, tf=None):
    """The film's compose_a at camera `cam` with ambience on the free frame f, lit by the clock `hour` (its
    lights into rec, a _Lights); tf = transform clock (film frames) for the heart's swell, or None."""
    c = Canvas(W, H)
    c.a[:124] = sc.sky_a(state).a
    _draw_clouds(c, f, state, sc.CLOUDS_A, -int(cam * 0.06), _cloud_q(hour))
    _comp(c, sc.peaks_a(state), -int(round(cam * sc.P_PEAK)))
    _comp(c, sc.farmesa_a(state), -int(round(cam * sc.P_FARM)))
    mx = -int(round(cam * P_MID))
    if state == 'bloom':                                 # the windows and tunnels lit one by one at dusk
        mid, lights, heart, facets = _hive_lights()
        _comp(c, mid, mx)
        _put_lights(c, lights, mx, 0, _lamp_k(hour), rec)
        rec.add_mask(c, mx, 0, heart)                    # (magic: the heart and the crystal facets glow
        rec.add_mask(c, mx, 0, facets)                   # at night)
    else:
        _comp(c, sc.mid_a(state), mx)
    nl, _top = sc.near_a(state)
    nx = -int(round(cam * P_NEAR))
    _comp(c, nl, nx)
    hx, hy = sc.HIVE_POS[0] + mx, sc.HIVE_POS[1]
    g = sc.hive_geo()
    s_holes = seg(tf, 120, 143) if tf is not None else 0.0
    s_heart = seg(tf, 118, 144) if tf is not None else 0.0
    if state == 'dark':
        before = c.a.copy()                              # the slime's green glows: kept at night
        for i, (x, y, hw, hh) in enumerate(g['holes']):          # faint green glows low in the holes
            ph = (f / 40.0 + hash01(3, i)) % 1.0
            k = min(1.0, 0.5 + 0.5 * math.sin(ph * 6.283) + s_holes * 0.8)
            if k > 0.45:
                c.px(hx + x, hy + y, '#3c7a3c' if k < 0.8 else '#5fb14a')
                if hw >= 2 and k > 0.7:
                    c.px(hx + x - 1, hy + y, '#2f5a32')
        # the tunnels and the heart chamber glow green inside their openings (the film's dithered halos
        # over the hive and the sand turned to muddy rings at night), brighter after dark
        nk = clamp(daylight.night(hour) * 1.5)
        hk = 0.35 + 0.15 * math.sin(f / 9.0) + 0.6 * ease_in(s_heart)
        for i, (x, y, inner, core) in enumerate(_dark_openings()):
            k = (0.5 + 0.5 * math.sin(f / 23.0 + i)) if i < 4 else hk
            k = k * (0.6 + 0.4 * nk) + 0.25 * nk
            for mm, cs in ((inner, DARK_GLOW[0]), (core, DARK_GLOW[1])):
                cc = cs[min(len(cs) - 1, int(k * len(cs)))]
                X0, Y0 = hx + x, hy + y
                ys, xs = np.nonzero(mm)
                ok = (xs + X0 >= 0) & (xs + X0 < W)
                c.a[ys[ok] + Y0, xs[ok] + X0] = col(cc)
        if s_heart > 0:                                  # the heart swells: its light spills out
            fx.glow(c, hx + sc.HEART[0], hy + sc.HEART[1], 9 + int(10 * s_heart), '#6bd04b',
                    round(0.6 * ease_in(s_heart), 3), rings=4)
        glows = np.any(c.a != before, axis=2)
        for i, (cx, cy, rx, ry) in enumerate(sc.POOLS_A):      # slime pools bubbling
            for k in range(3):
                per = 26 + k * 7
                t = ((f + int(hash01(i, k, 5) * per)) % per) / per
                bx = nx + cx + int((hash01(i, k, 6) - 0.5) * rx * 1.4)
                by = cy + int((hash01(i, k, 7) - 0.5) * ry)
                if t < 0.6:
                    c.circle(bx, by - int(t * 2), 0.5 + t * 2.2, sc.SLIME[3], fill=False)
                    c.px(bx - 1, by - int(t * 2) - 1, sc.SLIME[4])
                elif t < 0.7:
                    c.px(bx - 2, by - 2, sc.SLIME[4])
                    c.px(bx + 2, by - 3, sc.SLIME[4])
            fx.steam(c, nx + cx, cy - 2, f, color='#93a46a', n=2, height=12, spread=4, seed=i)
        hz, r0, r1 = _haze()                                    # drifting haze wisps (seamless loop)
        xo = (100 - int(f * 0.4) + int(cam * 0.5)) % HZ_L
        k = hz[:, xo:xo + W]
        reg = c.a[r0:r1]
        c.a[r0:r1] = (reg * (1 - k) + _HAZE_C * k).astype(np.uint8)
        rec.add_mask(c, 0, 0, glows)                    # (as the haze veils them)
    else:
        for i, (cx, cy, rx, ry) in enumerate(sc.POOLS_A):      # warm pools: glints + ripples
            for k in range(3):
                ph = (f + k * 11 + i * 5) % 30
                if ph < 12:
                    gx = nx + cx - rx // 2 + int(hash01(i, k, (f + k * 11) // 30) * rx)
                    c.px(gx, cy - 1 + k % 2, 'white')
            rr = ((f + i * 13) % 40) / 40.0
            c.ellipse(nx + cx + 4, cy, 2 + rr * (rx - 6), 0.5 + rr * (ry - 1), 'ice', fill=False)
        if s_heart > 0:                                         # the heart radiates as it swells
            fx.glow(c, hx + sc.HEART[0], hy + sc.HEART[1], 18 + int(12 * s_heart), 'magic',
                    round(0.45 * ease_in(s_heart), 3), rings=4)
        for i in range(10):                                     # facet glints catching the sun
            per = 34
            t0 = int(hash01(11, i) * per)
            if (f + t0) % per < 10:
                j = int(hash01(12, i, (f + t0) // per) * len(g['holes']))
                x, y = g['holes'][j][:2]
                fx.sparkle(c, hx + x - 3, hy + y - 3, f + t0, 'white', period=10, size=2)
        for i in range(18):                                     # drifting petals
            sp = 0.35 + hash01(13, i) * 0.3
            px_ = int((hash01(14, i) * (W + 40) + f * sp) % (W + 40)) - 20
            py_ = int((hash01(15, i) * 140 + f * 0.25 + math.sin(f / 12.0 + i) * 3) % 150) + 20
            c.px(px_, py_, ('bubblegum', 'cotton', 'lilac', 'cream')[i % 4])
        side, k = _sun_wash(hour)                               # the low sun's warm light, from its side
        if k > 0:
            fx.light_rect(c, 0 if side == 'left' else 160, 40, 160, 100, 'amber', 0.07 * k,
                          fade='right' if side == 'left' else 'left')
    _comp(c, sc.fg_a(state), -int(round(cam * P_FG)))
    return c


def _ring_band(tf, cx, cy):
    """The transform ring's rainbow band and white edge (ring_over) as a screen mask: magic light."""
    r = sc.ring_r(tf)
    D, _A = _ring_grid()
    d = D[H - cy:2 * H - cy, W - cx:2 * W - cx]
    return (d >= r - 5) & (d < r + 0.6)


def ring_over(c, bloom, f, tf, cx, cy):
    """The film's transform ring at radius ring_r(tf) around screen (cx, cy): `bloom` inside (None: c is
    already the bloom), the rainbow band with its white leading edge, sparkles riding it."""
    r = sc.ring_r(tf)
    D, A = _ring_grid()
    ys, xs = H - cy, W - cx
    d = D[ys:ys + H, xs:xs + W]
    if bloom is not None:
        np.copyto(c.a, bloom.a, where=(d < r)[:, :, None])
    band = (d >= r - 5) & (d < r - 1)
    if band.any():
        ang = ((A[ys:ys + H, xs:xs + W][band] + tf / 3.0) % 4).astype(int)
        c.a[band] = _RING_RGB[ang]
    c.a[(d >= r - 1) & (d < r + 0.6)] = _WHITE
    for i in range(22):
        a_ = hash01(21, i) * 6.283 + tf * 0.01
        rr = r - 3 - hash01(22, i) * 10
        fx.sparkle(c, int(cx + math.cos(a_) * rr), int(cy + math.sin(a_) * rr / 1.1), int(tf) + i * 3,
                   ('white', 'ice', 'cotton', 'butter')[i % 4], period=12, size=2)


def pool_burst(c, t, t0, cam, bloom):
    """SPACE at the big pool: a slime geyser with fat popping bubbles and fumes (dark), or a turquoise
    fountain with a sparkling crown (bloom); droplets rain back and rings spread over the pool."""
    a = t - t0
    if not 0 <= a < 72:
        return
    px, py = POOL_JET - int(round(cam * P_NEAR)), POOL_N[1] - 1
    if bloom:
        edge, body, core = '#1f6e80', 'ice', 'white'
        spray = ('white', 'ice', 'aqua', 'cotton', 'white')
    else:
        edge, body, core = 'dark_olive', '#8be04b', '#c6f27a'
        spray = ('#c6f27a', '#8be04b', '#7fb83a', '#c6f27a', 'olive')
    hj = int(JET_H * ease_out(min(1.0, a / 7))) if a < 12 else int(JET_H * max(0.0, 1 - (a - 12) / 28) ** 1.3)
    if bloom and hj > 4:
        fx.glow(c, px, py - hj // 2, 10 + hj // 3, 'ice', 0.3, rings=3)
    for yy in range(py - hj, py):                               # the jet, widening into a crown
        tt = (py - yy) / max(1, hj)
        if tt > 0.75 and hash01(yy, a // 2, 5) < (tt - 0.75) * 2.4:
            continue
        ww = 2 + int(tt * tt * 6) + (1 if tt < 0.12 else 0)
        wob = int(round(math.sin(yy * 0.5 + a * 0.9) * 0.7))
        x0, x1 = px - ww + wob, px + ww + wob
        c.hline(x0, x1, yy, body)
        c.px(x0, yy, edge)
        c.px(x1, yy, edge)
        c.px(px + wob - (1 if (yy + a) % 4 < 2 else 0), yy, core)
    if 0 < hj:                                                  # spray arcing off the crown
        for q in range(16):
            tt = ((a + q * 1.3) % 12) / 12
            side = 1 if q % 2 else -1
            sx_ = px + side * (3 + tt * (6 + q % 5 * 2))
            sy_ = py - hj + tt * tt * (hj * 0.6) - math.sin(tt * math.pi) * 5
            c.px(int(sx_), int(sy_), spray[q % 4])
    ui.burst(c, t, t0 + 7, px, py - JET_H, colors=spray, n=30, seed=7, power=0.75, life=48, gravity=0.07)
    if a < 22:                                                  # foam at the foot
        c.ellipse(px, py, 5 + a // 3, 2, core)
        c.ellipse(px, py, 3 + a // 4, 1, 'white' if bloom else spray[0])
    for k in range(3):                                          # rings spreading over the pool
        rr = 4 + (a - 6 - k * 9) * 1.0
        if 4 <= rr < POOL_N[2]:
            c.ellipse(px - 4, py + 1, rr, max(1.0, rr * 0.15), core if bloom else spray[0], fill=False)
    if not bloom:                                               # fat slime bubbles welling up and popping
        for i in range(7):
            tt = a - int(hash01(i, 71) * 24)
            if 0 <= tt < 26:
                bx = px - 4 + int((hash01(i, 72) - 0.5) * POOL_N[2] * 1.5)
                r_ = 1.5 + tt * 0.2
                if tt < 20:
                    c.circle(bx, py - int(r_), r_, spray[1], fill=False)
                    c.px(bx - 1, py - int(r_) - 1, spray[0])
                else:
                    for q in range(4):
                        c.px(bx + (-4, 4, -2, 2)[q], py - 4 - (tt - 20) - q % 2, spray[0])
        if a < 56:
            fx.steam(c, px + 2, py - 6, a + 400, color='#93a46a', n=7, height=34, spread=10, seed=9, speed=0.7)
    else:
        for i in range(10):
            fx.sparkle(c, px + int((hash01(i, 73) - 0.5) * 70), py - 8 - int(hash01(i, 74) * 44), a + i * 5,
                       ('white', 'ice', 'butter')[i % 3], period=14, size=2)


# ============================================================================= area 1: canyon + flank
CLOUDS_B = [(60, 14, 50, 21, 0.1), (190, 30, 40, 22, 0.08), (300, 8, 60, 23, 0.06)]


def render_canyon(f, cam, sting, lamps, t, hour, rec):
    """The film's render_b at camera `cam`: ambience on the free frame f, the Tartarus sting at age
    `sting` (steps after the flare; negative while the seams heat up; None: never), the lantern blaze
    at age `lamps` (or None), lit by the clock `hour` (its lights into rec, a _Lights). No balloon,
    callout or card."""
    a = sting
    c = Canvas(W, H)
    sk = sc.sky_b()
    sx = int(round(cam * sc.P_SKYB))
    c.a[:] = sk.a[:, sx:sx + W]
    _draw_clouds(c, f, 'bloom', CLOUDS_B, -int(round(cam * 0.12)), _cloud_q(hour))
    _comp(c, sc.far_b(), -int(round(cam * sc.P_FARB)))
    _comp(c, sc.mid_b(), -int(round(cam * sc.P_MIDB)))
    _ml, pool, lanterns, glints = sc.main_b()
    main, lights, facets = _flank_lights()
    mx = -int(round(cam))
    lk = _lamp_k(hour)
    _comp(c, main, mx)
    on = _put_lights(c, lights, mx, 0, lk, rec)          # doorways and windows lit one by one at dusk
    rec.add_mask(c, mx, 0, facets)                       # (magic: the crystal facets glow at night)
    # ---- pool reflection (screen space)
    X0, X1 = max(0, mx), min(W, mx + sc.WB)
    pm = pool[:, X0 - mx:X1 - mx]
    if pm.any():
        ys, xs = np.nonzero(pm)
        wl = 156
        src_y = np.clip(2 * wl - ys - 2, 0, H - 1)
        rip = (np.sin((ys * 1.7 + f * 0.35)) * 1.2).astype(int)
        src_x = np.clip(xs + X0 + rip, 0, W - 1)
        refl = c.a[src_y, src_x].astype(np.float32)
        refl = refl * 0.55 + np.array(col('teal'), np.float32) * 0.45
        c.a[ys, xs + X0] = np.clip(refl, 0, 255).astype(np.uint8)
        if on is not None:                               # the lit doorways' reflections shine too
            lit = on[src_y, src_x]
            if lit.any():
                rm = np.zeros((H, W), bool)
                rm[ys[lit], xs[lit] + X0] = True
                rec.add_mask(c, 0, 0, rm)
        for k in range(6):
            ph = (f + k * 7) % 28
            if ph < 14:
                gx = mx + 340 + int(hash01(k, 3, (f + k * 7) // 28) * 220)
                c.hline(gx, gx + 2 + k % 3, 160 + (k * 3) % 10, 'ice')
    # ---- hive life: lanterns (paper by day, lit one by one at dusk: no halos), glints, petals
    for i, (lx, ly, lc) in enumerate(lanterns):
        X = lx + mx
        if not -3 < X < W + 3:
            continue
        if lk > _thr(lx, ly):
            k = 0.6 + 0.4 * math.sin(f / 5.0 + i * 1.3)
            c.rect(X - 1, ly - 1, 3, 3, mix(lc, 'cream', 0.45))
            c.vline(X, ly - 1, ly, 'white' if k > 0.75 else 'cream')
            rec.add_mask(c, X - 1, ly - 1, np.ones((3, 3), bool))
        else:
            c.px(X, ly - 1, shade(lc, 0.85))
    for i, (gx, gy) in enumerate(glints):
        fx.sparkle(c, gx + mx, gy, f, 'white', period=26, phase=hash01(9, i), size=2)
    for i in range(16):
        sp = 0.3 + hash01(31, i) * 0.4
        px_ = int((hash01(32, i) * (W + 40) - f * sp) % (W + 40)) - 20 + mx + sc.PAN_B
        py_ = int((hash01(33, i) * 150 + f * 0.3 + math.sin(f / 10.0 + i) * 3) % 160) + 10
        if 0 <= px_ < W:
            c.px(px_, py_, ('bubblegum', 'cotton', 'lilac', 'cream')[i % 4])
    if lamps is not None:                                # (the blaze shines at night too)
        before = c.a.copy()
        lantern_blaze(c, t, lamps, mx, lanterns, glints)
        rec.add_changed(c, before)
    # ---- the gate: red seams, pulsing cracks, rune-ring seal, the sting
    gx = sc.GX + mx
    flare = (1 - seg(a, 0, 30)) ** 2 if a is not None and a >= 0 else 0.0
    if a is None:
        pre = 0.0
    elif a < 0:
        pre = 0.4 * seg(a, -TART_LEAD, 0)
    else:
        pre = 0.4 * (1 - seg(a, 30, 70))
    pulse = 0.5 + 0.5 * math.sin(f / 7.0)
    heat = clamp(0.35 + 0.3 * pulse + pre + flare)
    hot = mix('lava_dk', 'lava', heat) if heat < 0.7 else mix('lava', 'lava_hi', (heat - 0.7) / 0.3)
    core = 'butter' if flare > 0.3 else hot
    GF = sc.GFLOOR
    if gx - 90 < W and gx + 90 > 0:                      # (lava and magic: all of it shines at night;
        before = c.a.copy()                              # the red and magic halos only in the sting)
        if flare > 0.01:
            fx.glow(c, gx, GF, 50 + int(40 * flare), 'red', round(0.4 * flare, 3), rings=5)
        c.vline(gx, 38, GF - 3, hot)
        c.vline(gx, 48, GF - 12, core)
        c.vline(gx - 37, 38, GF - 3, mix(hot, 'lava_dk', 0.4))
        c.vline(gx + 37, 38, GF - 3, mix(hot, 'lava_dk', 0.4))
        c.hline(gx - 37, gx + 37, 37, mix(hot, 'lava_dk', 0.3))
        if flare > 0:
            c.vline(gx - 1, 40, GF - 3, mix('lava', 'lava_dk', 1 - flare))
            c.vline(gx + 1, 40, GF - 3, mix('lava', 'lava_dk', 1 - flare))
            fx.light_rect(c, gx - 70, GF - 4, 140, 16, 'lava', 0.35 * flare, fade='down')
        for j, pts in enumerate(sc.GATE_CRACKS):
            ck = clamp(0.25 + 0.3 * math.sin(f / 9.0 + j * 1.1) + pre + flare)
            cc = mix('lava_dk', 'red', ck / 0.7) if ck < 0.7 else mix('red', 'lava', (ck - 0.7) / 0.3)
            if flare > 0.7:
                cc = 'lava_hi'
            if flare > 0.2:
                mid_ = pts[len(pts) // 2]
                fx.glow(c, mid_[0] + mx, mid_[1], 8, 'red', 0.3 * flare, rings=3)
            for (a_, b_) in zip(pts[:-1], pts[1:]):
                c.line(a_[0] + mx, a_[1], b_[0] + mx, b_[1], cc)
        ex, ey = sc.EMB[0] + mx, sc.EMB[1]                      # the seal: turning rune ring + lock sigil
        if flare > 0.01:
            fx.glow(c, ex, ey, 26 + int(10 * flare), 'magic', round(0.3 * flare, 3), rings=4)
        for i in range(12):
            a_ = i / 12 * 6.283 + f * 0.012
            rx_, ry_ = ex + math.cos(a_) * 15, ey + math.sin(a_) * 15
            c.px(rx_, ry_, 'lilac' if (i + f // 6) % 4 else 'white')
            if i % 3 == 0:
                c.px(rx_ + math.cos(a_) * 1.5, ry_ + math.sin(a_) * 1.5, 'lavender')
        c.circle(ex, ey, 10, 'lavender', fill=False)
        for (dx, dy) in [(0, -8), (0, 8), (-8, 0), (8, 0)]:
            c.line(ex, ey, ex + dx, ey + dy, 'lilac')
        c.circle(ex, ey, 3, 'white')
        c.px(ex, ey - 1, 'magic')
        c.vline(ex, ey, ey + 2, 'magic')
        for i in range(10):                                     # embers rising from the seam
            per = 40 + int(hash01(51, i) * 30)
            tt = ((f + int(hash01(52, i) * per)) % per) / per
            exx = gx + int((hash01(53, i) - 0.5) * 70) + int(math.sin(f / 8.0 + i) * 2)
            eyy = GF - 4 - int(tt * 70)
            c.px(exx, eyy, 'lava_hi' if tt < 0.4 else ('lava' if tt < 0.7 else 'lava_dk'))
        rec.add_changed(c, before)
        if a is not None and 0 <= a < TART_END:
            tartarus_sting(c, a, gx, ex, ey)
    fx.dust_motes(c, f, (mx + 60, 110, 220, 50), n=14, color='#b0806c', seed=7)
    _comp(c, sc.fg_b(), -int(round(cam * sc.P_FGB)))
    return c


def tartarus_sting(c, t, gx, ex, ey):
    """The film's sting t steps after the flare: the seal's shockwave, dust curtains pouring from the
    lintel joints, pebbles falling, a dust cloud billowing at the foot."""
    GF = sc.GFLOOR
    if t < 14:
        rr = 20 + t * 4
        c.circle(ex, ey, rr, 'lilac' if t < 7 else 'lavender', fill=False)
        if t < 8:
            c.circle(ex, ey, rr - 2, 'magic', fill=False)
    for j, sx_ in enumerate((-44, -14, 18, 40)):
        tt = t - j * 2
        if 0 <= tt < 30:
            front = 30 + int(tt * 5.5)
            dens = 0.55 * (1 - tt / 30)
            c.dither(gx + sx_, 30, 2, min(front, GF) - 30, '#c8a890', dens)
            c.dither(gx + sx_ + 2, 30, 1, min(front, GF) - 34, '#8a7478', dens)
    for i in range(44):
        tt = t - int(hash01(61, i) * 12)
        if tt < 0:
            continue
        x0 = gx + int((hash01(62, i) - 0.5) * 120)
        y = 18 + int(hash01(63, i) * 16) + int(0.5 * 0.14 * tt * tt + tt * 0.5)
        if y < GF - 1:
            if i % 6 == 0:
                c.rect(x0, y, 2, 2, '#6c5a60')
                c.px(x0, y, '#b0806c')
            else:
                c.px(x0, y, '#c8a890')
                if tt > 4:
                    c.px(x0, y - 1, '#8a7478')
    k = seg(t, 4, 34)
    if 0 < k < 1:
        for i in range(14):
            xx = gx + int((hash01(71, i) - 0.5) * 120 * (0.5 + k))
            yy = GF - 2 - int(hash01(72, i) * 10 * k)
            c.dither(xx - 3, yy - 2, 7, 4, '#8a7478', 0.5 * (1 - k))


def lantern_blaze(c, t, a, mx, lanterns, glints):
    """SPACE at the arches: the lanterns light up one after the other along their vines, the doors pour
    warm light over the pool, the round windows blaze, the facets all glint and petals burst out."""
    if not 0 <= a < 80:
        return
    fade = 1 - seg(a, 34, 80)
    kk = ease_out(min(1.0, a / 5)) * fade
    for (ax, ay, hw, hh) in ARCHES:                             # light pouring out of the doors
        fx.glow(c, ax + mx, ay - hh // 3, hw + 18, 'gold', 0.45 * kk, rings=4)
        fx.light_beam(c, ((ax + mx - hw + 3, ay - 2), (ax + mx + hw - 3, ay - 2)),
                      ((ax + mx - hw - 16, ay + 24), (ax + mx + hw + 16, ay + 24)), 'butter', 0.4 * kk)
    for (wx, wy, r) in WINDOWS:
        fx.glow(c, wx + mx, wy, r + 10, 'butter', 0.6 * kk, rings=3)
        if kk > 0.4:
            c.circle(wx + mx, wy, max(1, r - 1), 'cream')
            c.circle(wx + mx, wy, max(1, r - 3), 'white')
    for i, (lx, ly, lc) in enumerate(lanterns):                 # a wave of light along the vines
        ki = ease_out(min(1.0, max(0.0, a - 2 * (i % 4)) / 4)) * fade
        if ki <= 0:
            continue
        fx.glow(c, lx + mx, ly, 10 + int(8 * ki), lc, 0.7 * ki, rings=3)
        if ki > 0.3:
            c.rect(lx + mx - 1, ly - 2, 3, 4, 'white' if (a + i) % 6 < 3 else 'cream')
            if a < 24:
                fx.sparkle(c, lx + mx, ly - 1, a + i * 3, 'white', period=12, size=3)
    if a < 48:
        for i, (gx, gy) in enumerate(glints):
            fx.sparkle(c, gx + mx, gy, a + i * 2, ('white', 'butter', 'ice')[i % 3], period=16, size=3)
    for j, (ax, ay, hw, hh) in enumerate(ARCHES):               # petals bursting out over the arches
        for q in range(2):
            ui.burst(c, t, t - a + 3 + q * 9, ax + mx, ay - hh + 4,
                     colors=('hot_pink', 'bubblegum', 'white', 'butter', 'lilac', 'cotton'),
                     n=24, seed=3 + j * 2 + q, power=0.9 - q * 0.2, life=60, gravity=0.03)


# ============================================================================= the region
_NOSKY = np.array((255, 0, 255), np.uint8)     # sky hooks: a colour no frame pixel has (scenery)


def _cover(c, layers):
    """Paint _NOSKY over `c` wherever the (rgb, mask) layers at screen x are solid."""
    hit = np.zeros((H, W), bool)
    for (_a, m), x in layers:
        x0, x1 = max(0, x), min(W, x + m.shape[1])
        h = min(H, m.shape[0])
        if x1 > x0:
            hit[:h, x0:x1] |= m[:h, x0 - x:x1 - x]
    np.copyto(c.a, _NOSKY, where=hit[..., None])
    return c


@lru_cache(maxsize=8)
def _crater_sky(state, cx):
    """Area 0's painted sky alone (gradient, low sun and its glow), as compose_crater draws it; the
    peaks, mesas, hive and crater rim blanked out (their hazy sunlit faces must not pass for sky)."""
    c = Canvas(W, H)
    c.a[:124] = sc.sky_a(state).a
    return _cover(c, [(sc.peaks_a(state), -int(round(cx * sc.P_PEAK))),
                      (sc.farmesa_a(state), -int(round(cx * sc.P_FARM))),
                      (sc.mid_a(state), -int(round(cx * P_MID))), (sc.near_a(state)[0], -int(round(cx * P_NEAR))),
                      (sc.fg_a(state), -int(round(cx * P_FG)))])


@lru_cache(maxsize=8)
def _canyon_sky(cx):
    """Area 1's painted sky alone (the panning sky strip with its sun), as render_canyon draws it; the
    peaks, canyon walls, hive flank and foreground blanked out."""
    c = Canvas(W, H)
    sx = int(round(cx * sc.P_SKYB))
    c.a[:] = sc.sky_b().a[:, sx:sx + W]
    return _cover(c, [(sc.far_b(), -int(round(cx * sc.P_FARB))), (sc.mid_b(), -int(round(cx * sc.P_MIDB))),
                      (sc.main_b()[0], -cx), (sc.fg_b(), -int(round(cx * sc.P_FGB)))])


def _warm():
    """Build every cached layer up front (while the region loads) so no area change stalls."""
    for st in ('dark', 'bloom'):
        sc.sky_a(st), sc.peaks_a(st), sc.farmesa_a(st), sc.mid_a(st), sc.near_a(st), sc.fg_a(st)
        for (x, y, w, sd, sp) in sc.CLOUDS_A:
            sc.cloud_sprite(w, sd, st)
    for (x, y, w, sd, sp) in CLOUDS_B:
        sc.cloud_sprite(w, sd, 'bloom')
    sc.hive_geo(), sc.sky_b(), sc.far_b(), sc.mid_b(), sc.main_b(), sc.fg_b()
    _haze(), _ring_grid(), _hive_lights(), _flank_lights(), _dark_openings()


class Badlands(GlowKeep, Region):
    SEG = 'badlands'
    _raw = None                 # the area's frame as rendered (for the sky repaint in after_light)
    CARDS = {False: ('CHANGELING KINGDOM', ['A HIVE IN THE BADLANDS', 'WAITING TO BLOOM']),
             True: ('CHANGELING KINGDOM', ['A HIVE IN THE BADLANDS', 'NOW IN FULL BLOOM'])}

    @property
    def CARD(self):
        """The atlas card: as the hive was when the player landed (the card shows on arrival; it never
        changes under the player's eyes), else (the journal) as it is saved."""
        b = getattr(self, 'card_bloom', None)
        return self.CARDS[bool('bloom' in self.lasting if b is None else b)]

    def auto_glow(self):
        """No painted spot glows by itself (sunlit sand and mesa rims, the pink crystal's edge, gold trims):
        the real lights are drawn by the clock and kept by mask (_Lights, GlowKeep)."""
        return False

    def after_light(self, c, t):
        """After the relight: the kept lights (GlowKeep), then the clock's sky over the whole open sky at
        every hour (the base keeps the painted golden sky, its low sun on the LEFT, while the clock's sky
        looks alike, ~18.0 to 18.3 h: the sun jumped from the clock's west to the painted east and back)."""
        super().after_light(c, t)
        raw, self._raw = self._raw, None
        if raw is None or self.area.sky is None:
            return
        hour = self.g.hour
        if daylight.sky_weight(self.HOUR, hour) >= 0.999:
            return                                       # (the base has painted it already)
        a, f, ix, iy = raw
        key = (self.ai, t, ix, iy)                       # the base's memo: once per step
        if self._sky_memo is None or self._sky_memo[0] != key:
            self._sky_memo = (key, daylight.sky_pixels(a, self.area.sky(f, ix, iy).a))
        mask = self._sky_memo[1]
        if mask.any():
            y0, y1 = self.area.cam_y
            daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, y1 - y0 + 130, t)

    def _done(self, c, f, cx, cy, rec):
        """End of an area render: its lights kept glowing at night, the frame kept for the sky repaint."""
        self.keep_glow(c, rec.mask(c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        return c

    def build_areas(self):
        _warm()
        return [
            Area(self.render_crater, cam_x=(0, CAM_A), cam_y=(0, 0), y_range=(20, 150), start=(96, 74),
                 landmarks=LANDMARKS_A, balloon='s', balloon_tint=TINT_DARK,
                 hotspots=HOTSPOTS_A, stamps=STAMPS_A, sky=self.sky_crater),
            Area(self.render_flank, cam_x=(0, CAM_B), cam_y=(0, 0), y_range=(26, 150), start=(58, 62),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=TINT_CANYON,
                 hotspots=HOTSPOTS_B, stamps=STAMPS_B, sky=self.sky_canyon),
        ]

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_crater(self, f, cx, cy):
        """The sky as render_crater draws it: while the first transform plays, the dark sky with the
        bloomed sky only inside the ring (its band stays scenery), and the burst's light on top."""
        tf = self.form_f(self.g.t)
        if tf is None:
            return _crater_sky('bloom' if self.bloom else 'dark', cx).copy()
        before = self.form_from
        r = sc.ring_r(tf)
        c = _crater_sky('bloom' if before == 'bloom' or r >= 230 else 'dark', cx).copy()
        hx, hy = HEART_M[0] - int(round(cx * P_MID)), HEART_M[1]
        if before == 'dark' and 0 <= r < 230:
            D, _A = _ring_grid()
            d = D[H - hy:2 * H - hy, W - hx:2 * W - hx]
            np.copyto(c.a, _crater_sky('bloom', cx).a, where=(d < r - 5)[:, :, None])
        nosky = np.all(c.a == _NOSKY, axis=2)
        form_light(c, tf, hx, hy, before)
        c.a[nosky] = _NOSKY
        return c

    def sky_canyon(self, f, cx, cy):
        return _canyon_sky(cx).copy()

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        self.bloom = False          # the hive has (or is) turning to crystal (stays: Region.lasting)
        self.form_t0 = None         # step the transform clock started (film frame FORM_LEAD)
        self.form_from = 'dark'     # 'dark': the first transform; 'bloom': a ring pulse over the bloom
        self.tart_t0 = None         # step the Tartarus sting's seams started heating up
        self.tart_auto = False      # the gate stung by itself once this visit
        self.area_t0 = 0

    def on_enter(self):
        # once the hive has bloomed it stays crystal on later landings (its SPACE then sends a ring pulse)
        self.bloom = 'bloom' in self.lasting or 'heart' in self.used
        self.form_t0 = None
        self.form_from = 'bloom' if self.bloom else 'dark'
        self.tart_t0 = None
        self.tart_auto = False
        self.areas[0].balloon_tint = TINT_GOLD if self.bloom else TINT_DARK
        self.card_bloom = self.bloom                        # the arrival card tells the hive as it is now

    def on_area(self, ai):
        self.area_t0 = self.g.t
        g = self.g
        if ai == 0:
            g.sfx('wind', 0.4)
        else:
            g.sfx('sparkle', 0.3)

    def form_f(self, t):
        """Transform clock in film frames, or None when no transform is playing."""
        if self.form_t0 is None or t < self.form_t0:
            return None
        tf = FORM_LEAD + (t - self.form_t0)
        return tf if tf < FORM_END else None

    def sting_age(self, t):
        if self.tart_t0 is None:
            return None
        a = t - self.tart_t0 - TART_LEAD
        return a if a < TART_END else None

    def _start_form(self, t):
        self.form_from = 'bloom' if self.bloom else 'dark'
        self.form_t0 = t
        self.bloom = True
        self.lasting.add('bloom')                           # saved: the hive stays bloomed (on_enter)
        self.spot_t['heart'] = t
        self.g.sfx('rumble', 0.5 if self.form_from == 'dark' else 0.3)

    def _start_sting(self, t):
        self.tart_t0 = t
        self.g.sfx('rumble', 0.3)

    # ------------------------------------------------------------------ events
    def on_interact(self, key, t):
        g = self.g
        if key == 'heart':
            self._start_form(t)
        elif key == 'pool':
            if self.bloom:
                g.sfx('splash', 0.5)
                g.sfx('fountain', 0.3)
            else:
                g.sfx('lava', 0.55)
                g.sfx('bubbles', 0.4)
        elif key == 'tartarus':
            self._start_sting(t)
        elif key == 'lanterns':
            g.sfx('chime', 0.5)
            g.sfx('sparkle', 0.4)

    def update(self, t):
        g = self.g
        k = t - self.area_t0
        if t - self.t_enter == 6:
            g.sfx('wind', 0.45)
        if self.ai == 0:
            if not self.bloom and self.form_t0 is None and self.bx > AUTO_FORM_X:
                self._start_form(t)
            tf = self.form_f(t)
            if tf is not None:
                if self.form_from == 'dark':
                    cue = {146: ('shimmer', 0.6), 158: ('crystal', 0.5), 176: ('sparkle', 0.4),
                           198: ('crystal', 0.35)}.get(tf)
                else:
                    cue = {146: ('shimmer', 0.45), 170: ('sparkle', 0.35)}.get(tf)
                if cue:
                    g.sfx(*cue)
            elif not self.bloom and k % 160 == 60:
                g.sfx('lava', 0.3 * self.near_gain(POOL_N[0]))          # slime pools blorping
            if k % 300 == 230:
                g.sfx('wind', 0.25, 0)
            self._tint_crater(t, tf)
        else:
            if not self.tart_auto and k > 40 and abs(self.bx - sc.GX) < AUTO_TART_DX:
                self.tart_auto = True
                self._start_sting(t)
            a = self.sting_age(t)
            if a == 2:
                g.sfx('crack', 0.5 * self.near_gain(sc.GX))
            elif a == 10:
                g.sfx('rumble', 0.4 * self.near_gain(sc.GX))
            if k % 260 == 120 and self.cx < 140:
                g.sfx('wind', 0.35, 0)                              # bleak canyon gust
            if k % 330 == 200 and self.cx > 160:
                g.sfx('sparkle', 0.2)                               # facets glinting on the flank
            self.areas[1].balloon_tint = _tint_between(TINT_CANYON, TINT_GOLD, (self.bx - 190) / 170)

    def _tint_crater(self, t, tf):
        """Light the balloon like the crater: gloomy until the ring has swept past it, then golden."""
        A = self.areas[0]
        if not self.bloom:
            A.balloon_tint = TINT_DARK
        elif tf is None or self.form_from == 'bloom':
            A.balloon_tint = TINT_GOLD
        else:
            hx = HEART_M[0] - int(round(self.cx * P_MID))
            d = math.hypot(self.bx - self.cx - hx, (self.by - 8 - HEART_M[1]) * 1.1)
            A.balloon_tint = TINT_GOLD if d < sc.ring_r(tf) else TINT_DARK

    # ------------------------------------------------------------------ renders
    def render_crater(self, f, cx, cy):
        t = self.g.t
        hour = self.g.hour
        rec = _Lights()
        tf = self.form_f(t)
        if tf is None:
            c = compose_crater(f, 'bloom' if self.bloom else 'dark', cx, hour, rec)
        else:
            before = self.form_from
            hx, hy = HEART_M[0] - int(round(cx * P_MID)), HEART_M[1]
            r = sc.ring_r(tf)
            if r < 0:
                c = compose_crater(f, before, cx, hour, rec, tf)
            elif r >= 230:
                c = compose_crater(f, 'bloom', cx, hour, rec)
            elif before == 'dark':
                c = compose_crater(f, 'dark', cx, hour, rec)
                ring_over(c, compose_crater(f, 'bloom', cx, hour, rec), f, tf, hx, hy)
            else:
                c = compose_crater(f, 'bloom', cx, hour, rec)
                ring_over(c, None, f, tf, hx, hy)
            if 0 <= r:                                           # the ring is magic: it shines at night
                rec.add_mask(c, 0, 0, _ring_band(tf, hx, hy))
            form_light(c, tf, hx, hy, before)                    # (no tremor: the world holds still)
        a = t - self.spot_t.get('pool', -999)
        if 0 <= a < 72:
            bloom = self.bloom
            if tf is not None and self.form_from == 'dark':      # has the ring reached the pool yet?
                dx = (POOL_JET - int(round(cx * P_NEAR))) - (HEART_M[0] - int(round(cx * P_MID)))
                bloom = math.hypot(dx, (POOL_N[1] - HEART_M[1]) * 1.1) < sc.ring_r(tf)
            pool_burst(c, t, self.spot_t['pool'], cx, bloom)
        return self._done(c, f, cx, cy, rec)

    def render_flank(self, f, cx, cy):
        t = self.g.t
        a = self.sting_age(t)
        la = t - self.spot_t.get('lanterns', -999)
        # (the sting no longer shakes the canyon: screen shake is uncomfortable to watch)
        rec = _Lights()
        c = render_canyon(f, cx, a, la if 0 <= la < 80 else None, t, self.g.hour, rec)
        return self._done(c, f, cx, cy, rec)
