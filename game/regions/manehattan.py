"""Manehattan as an explorable diorama (art: scenes/s08_manehattan.py, used read-only).

The film shows the island metropolis in four different views (dissolve / glide cuts), so the region has
four areas, flown through left to right with a cloud sweep between them, in film order:
  0  THE SKYLINE FROM THE HARBOUR (shot A): far mainland and the torch islet, the hazy back row, the
     Art-Deco cluster with the Crystaller Building, the quay with sheds and swinging cranes, the
     harbour with ferries, tug, sailboat and buoy, foreground pilings. The world here is the skyline
     layer (both film tags sit on it); the harbour layers in front move faster.
  1  BRIDLEWAY at street level (shot B): theatres with chasing marquee bulbs, blade signs, the el train
     on its viaduct, manhole steam, the parked cab cart (the film's `shot_b`, rebound).
  2  THE PARK and BRONCLYN (shot C): lake, stone arch bridge, fountain, boathouse, blossom tree, the
     brownstone row behind (the film's `shot_c`, rebound). The BRONCLYN tag rides on the brownstone
     layer and THE PARK tag on the bridge (Landmark.p), so both stay on what they name.
  3  THE RAIL BRIDGE to the mainland (shot D), with the film's gentle crane (a 10 px tilt).
Every area is drawn without the film's balloon, tags, card, iris and cut transitions. Timed film
beats run on game clocks: the boats sail on loops that wrap where no camera can see them, the el train
rattles across every 10 s, and the Friendship Express bursts onto the rail bridge on a bar line soon
after you arrive (whistle plume, flashing tower emblems, second toot), then comes again every 6 bars.

Things to set off with SPACE: the Crystaller Building's spire beacon (the sunburst crown lights up tier
by tier and a wave of gold runs down every window, rays and pulses from the needle, gold sparks), the
blue ferry moored at the quay (two horn blasts with steam puffs and sound arcs, then it casts off for a
lap of the harbour and comes back alongside), Bridleway's premiere show (every marquee runs a fast colour chase,
blade signs blaze, searchlights sweep, confetti pops from the marquees), the park fountain (a tall
jet with a spray crown and a little rainbow, reflected in the lake), and the Friendship Express (called
onto the bridge at once, or a toot of its whistle when it is already rolling). Three atlas stamps are
hidden: over the island's east end, under the stone arch bridge and up by the far bridge tower's flag.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, col, hash01, seg, smooth, ease_out, bayer, clamp
import scenes.s08_manehattan as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# ------------------------------------------------------------------ area 0: the harbour skyline
A_PM = sc.A_P['main']                  # world = skyline layer (film parallax .55)
# camera limits (skyline px): quay 440 px at .78/.55 -> 84; skyline 420 -> 100; back 400 (.32) -> 137
CAM_A = 84
A_XMAX = CAM_A / A_PM                  # largest film camera (boat-layer px)
_CRYS = sc.a_main()[1]['crystaller']   # (x, needle tip y, crown y)
QUAY_P = sc.A_P['quay'] / A_PM         # the quay layer's parallax in skyline px
FERRY_X = 168                          # the moored blue ferry (quay layer px), left of the pier at 190
FERRY_Y = sc.A_WY + 3
BEACON_LEN = 72
HORN_LEN = 168                         # two horn blasts, then a lap of the harbour and back to the quay
LAP0, LAP1 = 20, 160                   # the lap (steps after SPACE)
LAP_RX, LAP_RY = 64, 13                # its ellipse (quay px), out to the left and towards us

# ------------------------------------------------------------------ area 1: Bridleway
# street 470 (p 1) -> 150; fg 520 (p 1.4, the near curb runs the full width) -> 142; el 420 (.6) -> 166
CAM_B = 142
EL_PERIOD = 300                        # the el train crosses every 10 s (6 px / frame, ~95 frames)
EL_PHASE = 120
SHOW_LEN = 84                          # the premiere show (marquee chase, searchlights, confetti)
GRAND = (231, 48)                      # the grand theatre's parapet ornament (searchlights)
MARQUEES = [(44, 130), (180, 282), (322, 406)]   # theatre marquees (x0, x1); top y 100
MARQ_Y = sc.B_SW - 40

# ------------------------------------------------------------------ area 2: the park
# trees 420 (p .72) -> 138; bank 460 (p 1) -> 140; fg 520 (p 1.38) -> 144; brownstones 420 (.45) -> 222
CAM_C = 138
FOUNT = (104, sc.C_WY - 39, sc.C_WY - 24)   # fountain x, sun emblem y, top bowl y (bank layer = world)
FOUNT_LEN = 90

# ------------------------------------------------------------------ area 3: the rail bridge
# the deck truss ends at bridge x 660 -> 340; hills 420 (p .1) -> 1000
CAM_D = 340
CRANE_D = 10                           # the film's crane up: every layer but the sky sinks 0..10 px
TRAIN_PERIOD = 432                     # 6 bars between trains
TRAIN_LEAD = 16                        # frames before the hit the train starts (front just off screen)
TRAIN_SPEED = 4.6
TRAIN_GONE = 200                       # frames after the hit when the train + its plume are off screen
EXPRESS_SPOT = (356, 98)               # mid-span, on the rails


def _nop(*_a, **_k):
    return None


def _next_bar(t):
    """The first bar line at or after step t (scene frame 360 + t is on a bar when t % 72 == 0)."""
    return t + (-t) % 72


def _sail(x0, v, f, p, xmax, w, margin=44):
    """Boat-layer x of a vessel sailing at v px / frame (film position x0 at scene frame 320), wrapping
    over a span that is off screen for every film camera 0..xmax (screen x = x - xmax * p at worst).
    The margin covers the 36 px wake and the funnel smoke."""
    lo = -w - margin
    span = W + p * xmax + margin - lo
    return lo + (x0 + v * (f - 320) - lo) % span


@lru_cache(maxsize=1)
def _ferry_teal():
    return sc.FERRY.recolor({col('orange'): col('#3f8f8a'), col('#e8742a'): col('#2f7a7a')})


@lru_cache(maxsize=1)
def _ferry_blue():
    return sc.FERRY.recolor({col('orange'): col('#3f6fc8'), col('#e8742a'): col('#2f58a8')})


_MAGENTA = np.array((255, 0, 255), np.uint8)       # a key no painted pixel is near: never sky


def _stamp(c, rgba, x, y):
    """Stamp a scenery layer's opaque pixels (as sc.paste places it) out of a sky hook."""
    x, y = int(x), int(y)
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        m = rgba[y0 - y:y1 - y, x0 - x:x1 - x, 3] >= 128
        c.a[y0:y1, x0:x1][m] = _MAGENTA


# ------------------------------------------------------------------ the city by night
# The film painted Manehattan at noon. The relight's own glow keeping is off here (auto_glow): it would
# keep the gold finials, trims, the cab and vivid sky specks on the water shining like lamps. Instead every
# real light is drawn by the area render and collected in a _Lit, whose mask GlowKeep keeps shining after
# the clock's light: window panes (dark glass here, pale-wall panes via game/nightwin.py), lamp globes,
# marquee bulbs and lightboxes, beacons, the bridge's necklace, cabin windows, and their reflections in
# the water. Lamps come on first (from daylight.night() 0.08, one after another), the windows with
# nightwin's ramp (0.3 .. 0.7); GlowKeep fades them all in with the night.
LIT_SHARE = 0.6                         # share of the window panes lit (the city that never sleeps)
_LAMP_COLS = ('#fff4dc', '#fff4c8')
_WARM = np.array([col('#ffd466'), col('#ffc455'), col('#ffe07a')], np.uint8)
_ALL = np.concatenate([_WARM, np.array([col('#ffe27a'), col('#c8843a')], np.uint8)])   # panes, globe, rim
_MARK = np.array([(255, 219, 133), (255, 236, 176)], np.uint8)   # lit cabin windows (exact: found again)


def _lamp_share(hour):
    """0..1: the share of the lamps that are on (they come on through sunset's last minutes)."""
    return clamp((daylight.night(hour) - 0.08) / 0.25)


class _Lit:
    """The real lights of one frame: where they were drawn and their colour then; the mask keeps only
    those still showing at the end (nothing drawn over them since)."""
    def __init__(self):
        self.m = np.zeros((H, W), bool)
        self.v = np.zeros((H, W, 3), np.uint8)

    def add(self, c, m):
        if m is not None:
            self.m |= m
            self.v[m] = c.a[m]

    def px(self, c, ys, xs):
        ys, xs = np.asarray(ys), np.asarray(xs)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        self.m[ys[ok], xs[ok]] = True
        self.v[ys[ok], xs[ok]] = c.a[ys[ok], xs[ok]]

    def drawn(self, c, before):
        """Everything drawn since the copy `before` is light."""
        self.add(c, np.any(c.a != before, axis=2))

    def marks(self, c, y0, y1):
        """The lit cabin windows (the exact _MARK colours) in rows y0..y1."""
        a = c.a[y0:y1]
        m = np.zeros((H, W), bool)
        m[y0:y1] = np.all(a == _MARK[0], axis=2) | np.all(a == _MARK[1], axis=2)
        self.add(c, m)

    def mask(self, c):
        return self.m & np.all(c.a == self.v, axis=2)


@lru_cache(maxsize=16)
def _lights_of(name):
    """(ys, xs, kind, src rgb, v) of the lightable pixels of the scene layer sc.<name>(): kind 0..2 = a lit
    pane's warm shade, 3 = a lamp globe, 4 = its amber rim; v = when it goes on (0..1: a pane once
    nightwin's ramp passes v, a lamp once the lamp share passes v)."""
    from scipy import ndimage
    rgba = getattr(sc, name)()
    rgba = rgba[0] if isinstance(rgba, tuple) else rgba
    a = rgba[..., :3].astype(np.int32)
    op = rgba[..., 3] >= 128
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = (r * 77 + g * 150 + b * 29) >> 8
    glass = op & (b > r + 40) & (b > g + 15) & (lum < 130)
    wall = op & ~((b > r + 25) & (b > g + 5))
    lab, n = ndimage.label(glass)
    ys, xs, kind, vs = [], [], [], []
    if n:
        sl = ndimage.find_objects(lab)
        size = np.bincount(lab.ravel(), minlength=n + 1)
        for i, s_ in enumerate(sl, 1):
            hh, ww = s_[0].stop - s_[0].start, s_[1].stop - s_[1].start
            if not (3 <= size[i] <= 40 and hh >= 2 and ww <= 6):
                continue
            if hash01(s_[1].start, s_[0].start, 404) >= LIT_SHARE:
                continue
            if name == 'b_street' and s_[0].start >= sc.B_SW:  # the parked cab's windows: nobody inside
                continue
            if nightwin.pale_wall(lum, wall, s_):        # lit by game/nightwin.py instead
                continue
            # a lit pane must stand out from its 7x7 surroundings (or daylight dims it to a khaki block):
            # none on pale facades
            y0, y1 = max(0, s_[0].start - 3), s_[0].stop + 3
            x0, x1 = max(0, s_[1].start - 3), s_[1].stop + 3
            box = lum[y0:y1, x0:x1].astype(np.float32)
            box[(lab[y0:y1, x0:x1] == i)] = 215
            if 215 - box.mean() < 30:
                continue
            yy, xx = np.nonzero(op[s_])                  # the whole pane (its mullion / glint too)
            ys.append(yy + s_[0].start)
            xs.append(xx + s_[1].start)
            kind.append(np.full(len(yy), int(hash01(s_[1].start, s_[0].start, 405) * 3)))
            vs.append(np.full(len(yy), hash01(s_[1].start, s_[0].start, 406), np.float32))
    lamp = np.zeros_like(op)
    for cc in _LAMP_COLS:
        lamp |= op & np.all(rgba[..., :3] == np.array(col(cc), np.uint8), axis=2)
    lab, n = ndimage.label(lamp)
    if n:
        size = np.bincount(lab.ravel(), minlength=n + 1)
        keep = (size <= 40) & (np.arange(n + 1) > 0)
        glob = keep[lab]
        white = op & np.all(rgba[..., :3] == 255, axis=2)        # ...with the globe's white glint
        glob |= white & ndimage.binary_dilation(glob)
        # big globes get an amber rim (a lamp with a little shape, not a flat disc)
        big = glob & ndimage.binary_dilation((lab > 0) & (size[lab] > 12))
        rim = big & ~ndimage.binary_erosion(glob)
        yy, xx = np.nonzero(glob)
        ys.append(yy)
        xs.append(xx)
        kind.append(np.where(rim[yy, xx], 4, 3))
        # one switch-on time per lamp post (its globes are within a few px of each other)
        post, _np = ndimage.label(ndimage.binary_dilation(glob, iterations=5))
        pv = np.array([0.0] + [hash01(s_[1].start, s_[0].start, 409) for s_ in ndimage.find_objects(post)],
                      np.float32)
        vs.append(pv[post[yy, xx]])
    if not ys:
        z = np.zeros(0, int)
        return z, z, z, np.zeros((0, 3), np.uint8), np.zeros(0, np.float32)
    ys, xs, kind = np.concatenate(ys), np.concatenate(xs), np.concatenate(kind)
    return ys, xs, kind, rgba[ys, xs, :3].copy(), np.concatenate(vs)


def _night_lights(c, hour, layers, lit, ramps=None):
    """Light the panes and lamps of the given (layer name, x, y) placements as the night comes (lamps
    first, then the windows one by one), where nothing is drawn in front of them; they go into `lit`.
    ramps: (window ramp, lamp share) if already known."""
    kw, kl = ramps if ramps is not None else (nightwin.ramp(hour), _lamp_share(hour))
    if kw <= 0 and kl <= 0:
        return
    for name, x, y in layers:
        ys, xs, kind, src, v = _lights_of(name)
        if not len(ys):
            continue
        on = np.where(kind >= 3, v < kl, v < kw)
        sy, sx = ys + int(y), xs + int(x)
        ok = on & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        sy, sx, kd, sr = sy[ok], sx[ok], kind[ok], src[ok]
        seen = np.all(c.a[sy, sx] == sr, axis=1)          # nothing drawn in front of it
        sy, sx, kd = sy[seen], sx[seen], kd[seen]
        c.a[sy, sx] = _ALL[kd]
        lit.px(c, sy, sx)


# Window panes on pale walls (the Crystaller Building, the cream and beige towers): daylight can't keep a
# gold pane on a cream wall lit (it dims to khaki), so they are lit by game/nightwin.py and kept shining
# with GlowKeep; more of them as the night deepens, the Crystaller most of all.
AFTER_SHARE = 0.8                       # share of the pale-wall panes lit in full night
CRYS_SHARE = 1.0                        # ... on the Crystaller Building (every window)


def _pale_windows(layers):
    """game/nightwin.py placements for the pale-wall panes of the given (scene layer name, x, y)."""
    out = []
    for name, x, y in layers:
        boost = (sc.CRYS_X - 15, sc.CRYS_X + 15, CRYS_SHARE) if name == 'a_main' else None
        out.append(('mh_' + name, getattr(sc, name), x, y, AFTER_SHARE, boost, True))
    return out


# ------------------------------------------------------------------ water, clouds, haze by the clock
@lru_cache(maxsize=1)
def _sky_plain():
    """The film's noon sky gradient without its sun (sc.sky_layer's own gradient call)."""
    c = Canvas()
    c.gradient(0, 0, W, H, ['#3a6cc4', '#4d86da', 'sky', '#6db6f0', 'light_sky', '#c8ecff'], soft=0.7)
    return c.a


def _water(c, f, wy, cols, hour, from_h, **kw):
    """sc.water for the world clock: the sun glitter sits under the clock's sun (the moon's, fainter, at
    night) and, once the clock has moved the sun, the water no longer mirrors the film's painted noon sun
    (its disc and glow are swapped for plain sky while the water copies the rows above). Returns what
    _reflect needs: (wy, the frame right after the water, the water's wobble parameters)."""
    w = daylight.sky_weight(from_h, hour)
    _kind, bx, _by = daylight._body_pos(hour)
    kw['sun_x'] = int(round(kw.get('sun_x', sc.SUN[0]) * (1 - w) + bx * w))
    kw['glitter'] = int(round(kw.get('glitter', 46) * (1 - 0.6 * daylight.night(hour))))
    y1 = min(wy, 46)
    top = None
    if w >= 0.5:
        top = c.a[:y1].copy()
        m = np.all(top == sc.sky_layer().a[:y1], axis=2)
        c.a[:y1][m] = _sky_plain()[:y1][m]
    sc.water(c, f, wy, cols, **kw)
    if top is not None:
        c.a[:y1] = top
    return wy, c.a.copy(), kw.get('amp', 1.0), kw.get('speed', 0.22), kw.get('seed', 0), kw.get('src_y0')


def _reflect(c, lit, wat, f):
    """The lights mirrored in the water below them as broken, wobbling streaks (sc.water's own mirror
    rows and ripple wobble), dimmer with depth, only where the water still shows (nothing in front of it
    since it was drawn); they go into `lit` too."""
    if wat is None:
        return
    wy, snap, amp, speed, seed, src_y0 = wat
    m = lit.mask(c)
    if wy >= H or not m[:wy].any():
        return
    h2 = H - wy
    k = np.arange(h2)
    src = np.clip(wy - 1 - (k * 1.1).astype(int), 0, wy - 1)
    if src_y0 is not None:
        src = np.maximum(src, src_y0)
    off = np.round(np.sin(k * 0.45 + f * speed + seed) * (0.3 + k * 0.035 * amp)
                   + np.sin(k * 0.21 - f * speed * 0.6) * 0.5).astype(int)
    xs = np.clip(np.arange(W)[None, :] + off[:, None], 0, W - 1)
    hit = m[src[:, None], xs]
    hit &= ((k[:, None] + f // 8) % 2 == 0) & ((xs // 3 + k[:, None]) % 3 != 2)   # broken into ripple dashes
    hit &= np.all(c.a[wy:] == snap[wy:], axis=2)
    if not hit.any():
        return
    yy, xx = np.nonzero(hit)
    rgb = c.a[src[yy], xs[yy, xx]].astype(np.float32)
    dim = (0.72 - 0.32 * yy / h2)[:, None]
    c.a[wy + yy, xx] = (rgb * dim).astype(np.uint8)
    lit.px(c, wy + yy, xx)


def _clouds(hour):
    """sc.draw_clouds for the clock: by night the white clouds are painted duskier (the relight alone
    turns them into flat pale-lavender slabs brighter than every lamp)."""
    k = round(daylight.night(hour) * 0.4, 2)
    if k <= 0.02:
        return sc.draw_clouds

    def draw(c, f, specs, ox=0):
        for x0, y, w, sp, sd in specs:
            x = int((x0 + f * sp - ox) % 560) - 120
            sc.paste(c, _night_cloud(w, sd, k), x, y)
    return draw


@lru_cache(maxsize=64)
def _night_cloud(w, seed, k):
    a = sc.cloud_sprite(w, seed).copy()
    a[..., :3] = (a[..., :3].astype(np.float32) * (1 - k)).astype(np.uint8)
    return a


def _glint(hour):
    """sc.glint (sunlight flashing on glass): none at night."""
    return _nop if daylight.night(hour) > 0.2 else sc.glint


@lru_cache(maxsize=8)
def _sky_rows(hq, top, height):
    """Mean colour per row of the clock's sky strip (game/daylight.py) and of the film's noon sky."""
    return (daylight._sky_strip(hq, top, height).astype(np.float32).mean(axis=1),
            sc.sky_layer().a.astype(np.float32).mean(axis=1))


@lru_cache(maxsize=4)
def _haze_px(name, cols):
    """(ys, xs) of the haze-blue pixels (the colours `cols`) of the layer sc.<name>()."""
    rgba = getattr(sc, name)()
    rgba = rgba[0] if isinstance(rgba, tuple) else rgba
    m = np.zeros(rgba.shape[:2], bool)
    for cc in cols:
        m |= np.all(rgba[..., :3] == np.array(col(cc), np.uint8), axis=2)
    return np.nonzero(m & (rgba[..., 3] >= 128))


def _haze_of(c, name, cols, x, y):
    """The haze pixels of a far layer pasted at (x, y) still showing in c: a bool mask."""
    ys, xs = _haze_px(name, cols)
    sy, sx = ys + int(y), xs + int(x)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    m = np.zeros((H, W), bool)
    m[sy[ok], sx[ok]] = True
    return m


def _aerial(c, mask, painted, hour, from_h, iy, span):
    """Aerial perspective for the clock: the far hazy ridges (painted noon-pale against a noon horizon)
    take the colour of the clock's sky behind them, in the painted ratio to it, instead of staying a pale
    blue slab beside a gold, orange or night horizon. span: the clock's sky strip (Manehattan._sky_span)."""
    w = daylight.sky_weight(from_h, hour)
    if w < 0.01 or not mask.any():
        return
    top, height = span
    clock, noon = _sky_rows(daylight.sky_q(hour), top, height)
    ys, xs = np.nonzero(mask)
    sky = noon[ys] * (1 - w) + clock[np.clip(ys + iy - top, 0, height - 1)] * w
    ratio = np.minimum(painted[ys, xs].astype(np.float32) / np.maximum(noon[ys], 1), 1.1)
    cur = c.a[ys, xs].astype(np.float32)
    c.a[ys, xs] = np.clip(cur + (sky * ratio - cur) * (0.8 * w), 0, 255).astype(np.uint8)


_FAR_HAZE = ('#a9cfe6', '#a4cbe4', '#b4d6ea', '#9cc2dc')   # a_far: the mainland's hazy hills + far towers
_D_HAZE = ('#a8cce4', '#c8e2f2')                           # d_hills: the hazy ridge across the river


# ------------------------------------------------------------------ Bridleway's lights
@lru_cache(maxsize=1)
def _street_glass():
    """(ys, xs, v, rgb) of Bridleway's street-level glass (b_street px): the theatres' lobby doors (v 0:
    lit with the lamps) and the shop windows (v > 0: lit with the windows), and their lit colours."""
    from scipy import ndimage
    rgba, _meta = sc.b_street()
    a = rgba[..., :3]
    pane, line = np.array(col('#4b6ea8'), np.uint8), np.array(col('#a8d4f5'), np.uint8)
    glass = np.all(a == pane, axis=2) | np.all(a == line, axis=2)
    glass[:sc.B_SW - 20] = False
    glass[sc.B_SW:] = False
    lab, n = ndimage.label(glass)
    ys, xs, vs, cs = [], [], [], []
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        yy, xx = np.nonzero(lab[s_] == i)
        yy, xx = yy + s_[0].start, xx + s_[1].start
        door = s_[1].stop - s_[1].start <= 4
        v = 0.0 if door else 0.15 + 0.8 * hash01(s_[1].start, 412)
        hi = np.all(a[yy, xx] == line, axis=1)
        warm = (('#ffe6a0', '#fff6d8') if door else ('#ffd27e', '#ffe6b0'))
        rgb = np.where(hi[:, None], np.array(col(warm[1]), np.uint8), np.array(col(warm[0]), np.uint8))
        ys.append(yy)
        xs.append(xx)
        vs.append(np.full(len(yy), v, np.float32))
        cs.append(rgb)
    return (np.concatenate(ys), np.concatenate(xs), np.concatenate(vs), np.concatenate(cs),
            a[np.concatenate(ys), np.concatenate(xs)].copy())


@lru_cache(maxsize=1)
def _lightboxes():
    """(ys, xs) of Bridleway's lit signs (b_street px): the marquee boards, the lights under the
    marquees and the rooftop billboard's panel."""
    rgba, meta = sc.b_street()
    a = rgba[..., :3]
    m = np.zeros(a.shape[:2], bool)
    board = np.all(a == np.array(col('#fff4dc'), np.uint8), axis=2)
    under = np.all(a == np.array(col('#fff4b0'), np.uint8), axis=2)
    for mx0, mx1, my in meta['marquees']:
        m[my + 4:my + 12, mx0 + 3:mx1 - 3] |= board[my + 4:my + 12, mx0 + 3:mx1 - 3]
        m[my + 17, mx0:mx1] |= under[my + 17, mx0:mx1]
    bx, by, bw, bh = meta['billboard']
    m[by + 1:by + bh - 1, bx + 1:bx + bw - 1] |= rgba[by + 1:by + bh - 1, bx + 1:bx + bw - 1, 3] >= 128
    return np.nonzero(m)


def _lit_windows(rgba, extra=()):
    """A train's rgba with its windows (and `extra` colours) lit: the exact _MARK colours, found again
    in the frame after the paste."""
    tr = rgba.copy()
    for src, dst in (('#5a86b8', _MARK[0]), ('#c6e6ff', _MARK[1])) + tuple((e, _MARK[1]) for e in extra):
        m = np.all(tr[..., :3] == np.array(col(src), np.uint8), axis=2)
        tr[m, :3] = dst
    return tr


@lru_cache(maxsize=1)
def _el_train_lit():
    return _lit_windows(sc.el_train(), ('#fff4b0',))


@lru_cache(maxsize=1)
def _train_lit():
    return _lit_windows(sc.d_train()[0], ('#fff4b0',))


@lru_cache(maxsize=8)
def _boat_lit(spr):
    """A vessel sprite with its cabin windows lit (the exact _MARK colours)."""
    return spr.recolor({col('#5a7fb5'): tuple(int(v) for v in _MARK[1]),
                        col('#34497a'): tuple(int(v) for v in _MARK[0])})


@lru_cache(maxsize=4)
def _tops(name):
    """Per column of a scene layer (sc.<name>): its first opaque row (its height where a column is empty)."""
    rgba = getattr(sc, name)()
    m = (rgba[0] if isinstance(rgba, tuple) else rgba)[..., 3] >= 128
    return np.where(m.any(axis=0), m.argmax(axis=0), m.shape[0])


def _stamp_below(c, top, x):
    """Stamp everything from a layer's skyline (`top` per layer column) down: holes in a facade are not
    sky."""
    xs = np.arange(W) - int(x)
    ok = (xs >= 0) & (xs < len(top))
    t = np.full(W, H)
    t[ok] = top[xs[ok]]
    c.a[np.arange(H)[:, None] >= t[None, :]] = _MAGENTA


@lru_cache(maxsize=1)
def _d_quay():
    """The island quay, cropped where the film's bridge shot ends it (sliced once)."""
    return sc.a_quay()[0][:, :238].copy()


@lru_cache(maxsize=1)
def _crys_windows():
    """(ys, xs) of the Crystaller's shaft and podium window panes in the skyline layer (where nothing
    in front covers them): the building drawn alone, its glass picked out and kept where a_main agrees."""
    cv = sc.layer(420)
    sc.crystaller(cv, sc.CRYS_X, sc.A_WY - 6)
    solo = cv.a
    main = sc.a_main()[0][..., :3]
    glass = np.zeros(solo.shape[:2], bool)
    for g in sc.GLASS + ('#f2e6c8',):
        glass |= np.all(solo == np.array(col(g), np.uint8), axis=2)
    glass &= np.all(main == solo, axis=2)
    return np.nonzero(glass)


@lru_cache(maxsize=1)
def _crown():
    """The Crystaller's sunburst crown windows, bottom tier first: [[(x, y0, y1), ...], ...] (skyline
    px), recomputed the way scenes/s08_manehattan.crystaller lays them out."""
    cx, base = sc.CRYS_X, sc.A_WY - 6
    yb = base - 14 - 50 - 12
    tiers = []
    for hw, ah in [(8, 7), (7, 6), (5, 6), (4, 5), (2, 4)]:
        tiers.append([(cx + j, yb - ah + 3, yb - 1) for j in range(-hw + 2, hw - 1, 2)])
        yb -= ah - 1
    return tiers, yb


def _ferry_at(a):
    """(x, y, facing left, under way) of the blue ferry `a` steps after its horn: moored at the quay,
    or on its lap (out along the quay to the left, round towards us and back), eased in and out."""
    if not LAP0 <= a < LAP1:
        return FERRY_X, FERRY_Y, False, False

    def pos(s):
        th = -math.pi / 2 - 2 * math.pi * smooth(s)
        return FERRY_X + LAP_RX * math.cos(th), FERRY_Y + LAP_RY * (1 + math.sin(th))

    s = seg(a, LAP0, LAP1)
    x, y = pos(s)
    x2, _ = pos(min(1.0, s + 0.01))
    return int(round(x)), int(round(y)), x2 < x, True


def _env(a, n, rise=8, fall=20):
    """0..1 envelope of an n-step event: eased in over `rise`, out over the last `fall` steps."""
    return ease_out(seg(a, 0, rise)) * (1 - smooth(seg(a, n - fall, n)))


@lru_cache(maxsize=1)
def _bayer():
    return bayer(H, W)


def _beam(c, sx, sy, ang, length, width, strength):
    """A searchlight: a dithered additive cone from (sx, sy) at angle `ang` from straight up."""
    y0 = max(0, int(sy - length))
    if sy <= y0:
        return
    ys = np.arange(y0, int(sy))
    d = sy - ys                                         # distance up the beam
    cx = sx + np.tan(ang) * d
    hw = 1 + d / length * width / 2
    thr = _bayer()
    white = np.array((255, 252, 236), np.float32)
    for y, xc, h, dd in zip(ys, cx, hw, d):
        x0, x1 = max(0, int(xc - h)), min(W, int(xc + h) + 1)
        if x1 <= x0:
            continue
        k = 0.85 - 0.45 * dd / length
        m = thr[y, x0:x1] < k
        row = c.a[y, x0:x1].astype(np.float32)
        row[m] += white * strength
        c.a[y, x0:x1] = np.clip(row, 0, 255).astype(np.uint8)


NIGHT_LAYERS = ('a_back', 'a_main', 'a_quay', 'b_far', 'b_street', 'b_fg', 'c_far', 'c_brown', 'c_bank', 'c_trees',
                'c_fg', 'd_back', 'd_city')


def warm_night():
    """Build the night lights' caches (window and lamp pixels of every layer, the street glass, the
    lightboxes, the Crystaller's windows and crown, the pale-wall panes): at landing, not on the first
    frame of dusk (it stalled ~20 ms there)."""
    from engine import Canvas
    for name in NIGHT_LAYERS:
        _lights_of(name)
    _street_glass()
    _lightboxes()
    _crys_windows()
    _crown()
    for name in NIGHT_LAYERS:
        nightwin.paint(Canvas(), 23.0, 0, _pale_windows(((name, 0, 0),)))


class Manehattan(GlowKeep, Region):
    SEG = 'manehattan'
    CARD = ('METROPOLIS', ['ISLAND OF SKYSCRAPERS', 'THEATRES - PIERS - PARK'])

    def warm(self):
        warm_night()
        super().warm()

    def build_areas(self):
        harbour = [
            # the film points at the needle (y 25); 40 keeps the tag under the HUD's star / stamp counters
            Landmark('CRYSTALLER BUILDING', _CRYS[0], (_CRYS[0] + 5, _CRYS[1] + 39, 'right', 12)),
            Landmark('MANEHATTAN', 300, (300, 92, 'down', 14)),
        ]
        street = [Landmark('BRIDLEWAY', 231, (231, 120, 'down', 17))]
        park = [
            # the tag rides on the arch bridge (p 1.08), the proximity x is where the balloon meets it
            Landmark('THE PARK', 192, (134 + 62, 118, 'up', 12), p=1.08),
            # rides on the brownstone row (p .45); the film points at y 96, 84 keeps the leader on the
            # brownstones above every tree crown
            Landmark('BRONCLYN', 290, (268, 84, 'up', 14), p=sc.C_P['brown']),
        ]
        bridge = [Landmark('RAIL BRIDGE', sc.T1, (sc.T1 + sc.TW // 2 + 1, 52, 'right', 12))]
        return [
            Area(self.render_harbour, cam_x=(0, CAM_A), y_range=(30, 146), start=(56, 84),
                 landmarks=harbour, balloon='m',
                 hotspots=[Hotspot('beacon', _CRYS[0], 24, 'spark', cooldown=BEACON_LEN + 10),
                           Hotspot('horn', FERRY_X + 9, FERRY_Y - 9, 'horn', cooldown=HORN_LEN + 10,
                                   p=QUAY_P)],
                 stamps=[Stamp('east end', 378, 84)],
                 sky=self.sky_harbour),
            Area(self.render_street, cam_x=(0, CAM_B), y_range=(28, 140), start=(36, 70),
                 landmarks=street, balloon='m',
                 hotspots=[Hotspot('marquee', 231, MARQ_Y + 1, 'note', cooldown=SHOW_LEN + 6)],
                 sky=self.sky_street),
            Area(self.render_park, cam_x=(0, CAM_C), y_range=(28, 154), start=(36, 64),
                 landmarks=park, balloon='m',
                 hotspots=[Hotspot('fountain', FOUNT[0] + 24, sc.C_WY - 10, 'splash',
                                   cooldown=FOUNT_LEN + 6)],
                 stamps=[Stamp('under the arch', 192, 140)],
                 sky=self.sky_park),
            Area(self.render_bridge, cam_x=(0, CAM_D), cam_y=(-CRANE_D, 0), y_range=(16, 136),
                 start=(40, 70), landmarks=bridge, balloon='m',
                 hotspots=[Hotspot('express', EXPRESS_SPOT[0], EXPRESS_SPOT[1], 'whistle', cooldown=50)],
                 stamps=[Stamp('tower flag', sc.T2 + 16, 14)],
                 sky=self.sky_bridge),
        ]

    # -------------------------------------------------------------- clocks
    def init_state(self):
        self.train_hit = None       # step of the Friendship Express hit (front at train_x0, bursting in)
        self.train_x0 = sc.TRAIN_X0
        self.toot_t = None          # step of a player-asked whistle toot

    def train_k(self, t):
        """Steps since the train's hit (from -TRAIN_LEAD while it rolls in), None when no train runs."""
        if self.train_hit is None or not -TRAIN_LEAD <= t - self.train_hit <= TRAIN_GONE:
            return None
        return t - self.train_hit

    def train_busy(self, t):
        """A train is rolling or scheduled."""
        return self.train_hit is not None and t <= self.train_hit + TRAIN_GONE

    def train_front(self, k):
        return self.train_x0 + TRAIN_SPEED * k

    @staticmethod
    def el_phase(t):
        return (t + EL_PHASE) % EL_PERIOD

    # -------------------------------------------------------------- events + sound
    def on_area(self, ai):
        g, t = self.g, self.g.t
        if ai == 0:
            g.sfx('waves', 0.4, 0)
        elif ai == 1:
            g.sfx('neon', 0.5)
        elif ai == 2:
            g.sfx('fountain', 0.4)
        else:
            g.sfx('waves', 0.35, 1)
            if not self.train_busy(t):                  # bridge empty: a train on the next bar line
                self.train_hit, self.train_x0, self.toot_t = _next_bar(t + 30), sc.TRAIN_X0, None

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.ai == 3 and self.switch is None and not self.train_busy(t):
            last = self.train_hit if self.train_hit is not None else t
            self.train_hit = _next_bar(max(t + 30, last + TRAIN_PERIOD))
            self.train_x0, self.toot_t = sc.TRAIN_X0, None
        if self.switch is not None:
            return
        if self.ai == 0:
            ah = t - self.spot_t.get('horn', -999)
            if ah == 14:
                g.sfx('horn', 0.55)
            elif ah == LAP0:
                g.sfx('splash', 0.25)
            elif ah == LAP1 - 6:
                g.sfx('creak', 0.3, 1)                   # back alongside the quay
            if t % 360 == 40:
                g.sfx('waves', 0.3, (t // 360) % 2)
            if t % 600 == 250:
                g.sfx('horn', 0.3)
        elif self.ai == 1:
            if t % 432 == 216:
                g.sfx('neon', 0.25)
            ph = self.el_phase(t)
            if ph in (14, 50):
                g.sfx('chug', 0.3, (ph // 36) % 2)
            if t % 300 == 150:
                g.sfx('steam', 0.25)
        elif self.ai == 2:
            if t % 288 == 0:
                g.sfx('fountain', 0.3 * self.near_gain(FOUNT[0]))
            if t % 216 == 108:
                g.sfx('creak', 0.25 * self.near_gain(200), (t // 216) % 2)
        else:
            k = self.train_k(t)
            if k is not None:
                gain = self.near_gain(self.train_front(k) - 60)
                if k == 0:
                    g.sfx('whistle', 0.85 * max(0.4, gain))
                elif k == 36:
                    g.sfx('whistle', 0.5 * max(0.3, gain))
                elif k in (6, 42, 78):
                    g.sfx('chug', 0.5 * gain, (k // 36) % 2)
            if t % 540 == 414:
                g.sfx('horn', 0.3)

    def on_interact(self, key, t):
        g = self.g
        if key == 'beacon':
            g.sfx('shimmer', 0.55)
            g.sfx('chime', 0.35)
        elif key == 'horn':
            g.sfx('horn', 0.7)                           # the second blast and the casting off: update()
        elif key == 'marquee':
            g.sfx('neon', 0.6)
            g.sfx('chime', 0.4)
            g.sfx('pop', 0.35, 1)
        elif key == 'fountain':
            g.sfx('fountain', 0.6)
            g.sfx('splash', 0.35)
        elif key == 'express':
            k = self.train_k(t)
            # the train (or its trailing plume, ~230 px behind the front) is still in or left of the view
            if k is not None and self.train_front(k) - 230 <= self.cx + W:
                self.toot_t = t                          # a toot of its whistle
                g.sfx('whistle', 0.75)
            else:                                        # call it onto the bridge now, from just
                self.train_hit = t + TRAIN_LEAD          # off the left edge of the screen
                self.train_x0 = max(sc.TRAIN_X0, int(self.cx) + 60)
                self.toot_t = None
                g.sfx('chug', 0.45, 0)

    # -------------------------------------------------------------- the clock's light
    _post = None                    # light drawn after the clock's light (searchlights, aerial haze)

    def auto_glow(self):
        return False                # the real lights are kept by the area renders (_Lit, GlowKeep)

    def _later(self, fn):
        """Run fn(c) on the frame right after the clock's light and sky (Region.after_light)."""
        if self._post is None:
            self._post = []
        self._post.append(fn)

    def _world(self, t, cam=None, with_balloon=False):
        self._post = None
        return super()._world(t, cam, with_balloon)

    def after_light(self, c, t):
        self._resky(c, t)
        super().after_light(c, t)                        # GlowKeep: the lights get their colour back
        post, self._post = self._post, None
        for fn in post or ():
            fn(c)

    HORIZON = {3: sc.D_WY}          # areas whose open sky shows lower than the clock's sky strip reaches

    def _sky_span(self):
        """(top, height) in world rows of the clock's sky strip here: the base _world's (it ends 120 rows
        below the top of the highest view and game/daylight.paint_sky repeats its last row below that),
        or one reaching down to the area's horizon (the water line) where the sky shows that low."""
        y0, y1 = self.area.cam_y
        top, height = y0 - 10, y1 - y0 + 130
        bottom = self.HORIZON.get(self.ai)
        return (top, max(height, bottom - top)) if bottom is not None else (top, height)

    def _resky(self, c, t):
        """The rail bridge's sky under the deck was the strip's flat last row, a slab of horizon colour
        (peach at 16:00, gold and orange at sunset) under a sky still blue or red just above the deck:
        once the clock's sky is all there, it is painted again from a strip reaching the water line."""
        top, height = self._sky_span()
        y0, y1 = self.area.cam_y
        memo, hour = self._sky_memo, self.g.hour
        if (height == y1 - y0 + 130 or memo is None or memo[0][:2] != (self.ai, t)
                or daylight.sky_weight(self.HOUR, hour) < 0.999):
            return
        (_ai, _t, ix, iy), mask = memo
        if mask.any():
            daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, top, height, t)

    def _haze_later(self, c, mask, iy):
        """Aerial perspective for the far haze (mask) once the clock's light is on (see _aerial)."""
        if mask is not None and mask.any() and abs(self.g.hour - self.HOUR) >= 0.05:
            painted, hour, span = c.a.copy(), self.g.hour, self._sky_span()
            self._later(lambda cc: _aerial(cc, mask, painted, hour, self.HOUR, iy, span))

    # -------------------------------------------------------------- the sky alone
    # every view starts from the film's screen-fixed noon sky (gradient + sun disc and glow); clouds
    # are scenery (game/daylight.py repaints the sky for the world clock's hour). Each area stamps its
    # scenery layers out of it (pale far ridges, glass shop windows and light poles are close to the sky's
    # colours), and the water that mirrors the sky.
    def sky_noon(self, f, cx, cy):
        return sc.sky_layer().copy()

    def sky_harbour(self, f, cx, cy):
        X, P = cx / A_PM, sc.A_P
        c = sc.sky_layer().copy()
        for rgba, x in ((sc.a_far(), -int(X * P['far'])), (sc.a_back()[0], -int(X * P['back'])),
                        (sc.a_main()[0], -cx), (sc.a_quay()[0], -int(X * P['quay'])),
                        (sc.a_fg(), -int(X * P['fg']))):
            _stamp(c, rgba, x, 0)
        c.a[sc.A_WY:] = _MAGENTA
        return c

    def sky_street(self, f, cx, cy):
        X, P = cx, sc.B_P
        c = sc.sky_layer().copy()
        for rgba, x in ((sc.b_far()[0], -int(X * P['far'])), (sc.b_el(), -int(X * P['el'])),
                        (sc.b_street()[0], -int(X * P['st'])), (sc.b_fg(), -int(X * P['fg']))):
            _stamp(c, rgba, x, 0)
        _stamp_below(c, _tops('b_street'), -int(X * P['st']))    # its see-through shop windows too
        return c

    def sky_park(self, f, cx, cy):
        X, P = cx, sc.C_P
        c = sc.sky_layer().copy()
        for rgba, x in ((sc.c_far()[0], -int(X * P['far'])), (sc.c_brown()[0], -int(X * P['brown'])),
                        (sc.c_trees(), -int(X * P['trees'])), (sc.c_bank()[0], -int(X * P['bank'])),
                        (sc.c_fg(), -int(X * P['fg']))):
            _stamp(c, rgba, x, 0)
        _stamp_below(c, _tops('c_brown'), -int(X * P['brown']))    # the gaps between the brownstones
        c.a[sc.C_WY:] = _MAGENTA
        return c

    def sky_bridge(self, f, cx, cy):
        X, Y, P = cx, -cy, sc.D_P
        c = sc.sky_layer().copy()
        co = -int(X * P['city']) - 100
        for rgba, x, y in ((sc.d_hills(), -int(X * P['far']), Y), (sc.d_back()[0], -int(X * P['back']) - 110, Y + 12),
                           (sc.d_city()[0], co, Y + 18), (_d_quay(), co, Y + 18), (sc.d_bridge()[0], -X, Y),
                           (sc.a_fg(), -110 - int(X * P['fg']), Y)):
            _stamp(c, rgba, x, y)
        c.a[max(0, sc.D_WY + Y):] = _MAGENTA
        return c

    # -------------------------------------------------------------- area 0: the harbour skyline
    def render_harbour(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        X = cx / A_PM                       # the film's camera (boat-layer px)
        lit = _Lit()
        lamps = _lamp_share(hour)
        c = sc.sky_layer().copy()
        _clouds(hour)(c, f, sc.A_CLOUDS, ox=X * 0.05)
        fo = -int(X * sc.A_P['far'])
        sc.paste(c, sc.a_far(), fo)
        haze, haze_rgb = _haze_of(c, 'a_far', _FAR_HAZE, fo, 0), c.a.copy()
        if lamps > 0:                                    # the torch islet's flame
            lit.px(c, [sc.A_WY - 22, sc.A_WY - 21, sc.A_WY - 20], [24 + fo] * 3)
        back, _ = sc.a_back()
        sc.paste(c, back, -int(X * sc.A_P['back']))
        main, meta = sc.a_main()
        mo = -cx
        sc.paste(c, main, mo)
        for (x, y) in meta.get('beacons', []):
            if (f // 9) % 4 == 0:
                c.px(x + mo, y, '#ff5a4a')
                if lamps > 0:                            # the towers' red beacons
                    lit.px(c, [y], [x + mo])
        for i, (x, y) in enumerate(meta.get('flags', [])):
            sc.flag(c, x + mo, y, f, ('#e83b3b', '#3f6fc8', '#f2c24a', '#3f9a86')[i % 4], seed=i)
        glint = _glint(hour)
        for (x, y, ph) in meta.get('glints', []):
            glint(c, x + mo, y, f, ph, period=72)
        if f % 18 < 6:                                   # the gold needle glints on every beat (by night:
            before = c.a.copy()                          # the spire's light)
            fx.sparkle(c, _CRYS[0] + mo, _CRYS[1], f % 18, 'white' if lamps <= 0 else 'butter', period=12,
                       size=3)
            if lamps > 0:
                lit.drawn(c, before)
        a = t - self.spot_t.get('beacon', -999)
        night = nightwin.ramp(hour)
        if 0 <= a < BEACON_LEN or night > 0:
            before = c.a.copy()
            if 0 <= a < BEACON_LEN:
                self._beacon_crown(c, a, mo)
            else:                                          # night: the sunburst crown is lit, tier by tier
                self._night_crown(c, night, mo, f)
            if daylight.night(hour) > 0.1:                 # its gold shines at night too
                lit.drawn(c, before)
        # quay, flags and swinging cranes
        quay, qmeta = sc.a_quay()
        qo = -int(X * sc.A_P['quay'])
        sc.paste(c, quay, qo)
        for i, (x, y) in enumerate(qmeta['flags']):
            sc.flag(c, x + qo, y, f, ('#e83b3b', '#3f6fc8', '#f2c24a')[i % 3], seed=i + 3, w=3)
        for i, (x, y) in enumerate(qmeta['cranes']):
            ang = math.radians(-12 + 10 * math.sin(f / 40 + i * 2.1)) + (math.pi if i % 2 else 0)
            L = 22
            ex, ey = x + qo + int(round(math.cos(ang) * L)), y + int(round(math.sin(ang) * L * 0.25)) - 2
            c.line(x + qo, y - 1, ex, ey, '#d8604a')
            c.line(x + qo, y - 3, ex, ey, '#b84a3a')
            bx = x + qo - int(round(math.cos(ang) * 6))
            c.rect(bx - 1, y - 2, 3, 3, '#5a5a6a')
            hl = 8 + int(3 * math.sin(f / 23 + i))
            c.vline(ex, ey + 1, ey + hl, '#5a5a6a')
            c.rect(ex - 1, ey + hl, 3, 2, ('#3f8f8a', '#e9b44a', '#c86a4a')[i % 3])
        wy = sc.A_WY
        wat = _water(c, f, wy, ['#7cc0e8', '#4f9ad0', '#3a7cb8', '#2e66a4'], hour, self.HOUR, refl=0.40, seed=1,
                     src_y0=10)
        # the city lights go on after the water (it mirrors the painted city; _reflect adds their light)
        lay = (('a_back', -int(X * sc.A_P['back']), 0), ('a_main', mo, 0), ('a_quay', qo, 0))
        _night_lights(c, hour, lay, lit)
        lit.add(c, nightwin.paint(c, hour, t, _pale_windows(lay)))
        # the blue ferry moored at the quay (its horn is a hotspot: two blasts, then a harbour lap)
        boat = _boat_lit if lamps > 0 else (lambda s: s)
        ah = t - self.spot_t.get('horn', -999)
        horn = 0 <= ah < 48
        fxp, fyp, flip, moving = _ferry_at(ah)
        sc.boat(c, boat(_ferry_blue()), fxp + qo, fyp, f, flip=flip, wake=moving, seed=7)
        if moving:
            sc.smoke_puffs(c, fxp + qo + (9 if flip else 8), fyp - 10, f, n=4, drift=0.2 if flip else -0.2,
                           seed=8)
        # vessels: the film's courses, looping where no camera sees them
        sail_x = _sail(238, -0.25, f, 0.85, A_XMAX, sc.SAIL.w) - X * 0.85
        sc.boat(c, sc.SAIL, sail_x, wy + 4, f, flip=True, seed=5)
        fer2_x = _sail(274, -0.55, f, 0.9, A_XMAX, sc.FERRY.w) - X * 0.9
        sc.boat(c, boat(_ferry_teal()), fer2_x, wy + 9, f, flip=True, seed=3)
        fer1_x = _sail(132, 0.75, f, 0.95, A_XMAX, sc.FERRY.w) - X * 0.95
        sc.boat(c, boat(sc.FERRY), fer1_x, wy + 16, f, seed=1)
        sc.smoke_puffs(c, int(fer1_x) + 8, wy + 6, f, n=4, drift=-0.2, seed=2)
        tug_x = _sail(196, 0.95, f, 1.1, A_XMAX, sc.TUG.w) - X * 1.1
        sc.boat(c, boat(sc.TUG), tug_x, wy + 38, f, seed=9)
        sc.smoke_puffs(c, int(tug_x) + 4, wy + 30, f + 7, n=6, drift=-0.25, seed=4, color='#d8d8e0',
                       dark='#9a96a8')
        if lamps > 0:
            lit.marks(c, wy - 12, H)
        # channel buoy (bobbing, blinking)
        bx = 168 - int(X * 1.15)
        by = 158 + (1 if (f // 15) % 2 else 0)
        c.rect(bx - 2, by - 6, 5, 6, '#d8443a')
        c.rect(bx + 1, by - 6, 2, 6, '#a8302a')
        c.hline(bx - 2, bx + 2, by - 4, 'white')
        c.vline(bx, by - 10, by - 7, '#5a5a6a')
        on = (f // 9) % 3 == 0
        c.px(bx, by - 11, '#ffe25a' if on else '#b8a040')
        if on and lamps > 0:
            lit.px(c, [by - 11], [bx])
        c.hline(bx - 4, bx + 4, by, '#2e66a4')
        c.hline(bx - 3, bx + 3, by + 1, '#d8f4ff' if (f // 6) % 2 else '#7cc0e8')
        if horn:
            self._horn(c, ah, FERRY_X + qo + 9, FERRY_Y - 9)
        if 0 <= a < BEACON_LEN:                            # its light over the clock's own sky (drawn on the
            bx_, by_ = _CRYS[0] + mo, _CRYS[1] + 1        # painted noon sky, a night kept a blue blot)
            self._later(lambda cc: self._beacon_sky(cc, t, a, bx_, by_))
        _reflect(c, lit, wat, f)
        sc.paste(c, sc.a_fg(), -int(X * sc.A_P['fg']))
        if lamps > 0:
            self.keep_glow(c, lit.mask(c), 0.95)
        self._haze_later(c, haze & np.all(c.a == haze_rgb, axis=2), int(cy))
        return c

    @staticmethod
    def _night_crown(c, k, mo, f):
        """The Crystaller by night: its sunburst crown windows lit gold, from the bottom tier up as the
        night deepens (k 0..1), the needle tipped with a warm light; now and then a window glints."""
        tiers, ytop = _crown()
        n = int(round(k * len(tiers)))
        for j, wins in enumerate(tiers[:n]):
            for i, (x, y0, y1) in enumerate(wins):
                cc = 'white' if hash01(i, j, f // 24) < 0.06 else ('butter' if (i + j) % 2 else 'gold')
                c.vline(x + mo, y0, y1, cc)
        if k > 0.5:
            c.px(sc.CRYS_X + mo, _CRYS[1] + 1, 'butter')

    @staticmethod
    def _beacon_crown(c, a, mo):
        """The Crystaller lights up: the sunburst crown tier by tier, then a wave of gold runs down every
        window of the shaft; it goes out again from the foot up."""
        tiers, ytop = _crown()
        n = len(tiers)
        for k, wins in enumerate(tiers):
            if not 2 + 3 * k <= a < BEACON_LEN - 6 - 3 * (n - 1 - k):
                continue
            for j, (x, y0, y1) in enumerate(wins):
                cc = 'white' if (j + k + a // 3) % 3 == 0 else 'butter'
                c.vline(x + mo, y0, y1, cc)
        if 10 <= a < BEACON_LEN - 4:                   # the needle glows white along its length
            c.vline(sc.CRYS_X + mo, _CRYS[1] + 2, ytop - 1, 'white' if (a // 3) % 2 else 'butter')
        ys, xs = _crys_windows()
        if a < 10:
            return
        front = 40 + 4 * (a - 10)                      # the gold wave runs down the shaft ...
        foot = sc.A_WY - 6 - 6 * max(0, a - (BEACON_LEN - 22))   # ... and goes out from the foot up
        m = (ys < front) & (ys < foot) & (xs + mo >= 0) & (xs + mo < W)
        if not m.any():
            return
        y, x = ys[m], xs[m] + mo
        tw = ((x + y // 3 + a // 4) % 5 == 0) | (y >= front - 3)
        c.a[y, x] = np.where(tw[:, None], np.array(col('white'), np.uint8), np.array(col('butter'), np.uint8))

    @staticmethod
    def _beacon_sky(c, t, a, x, y):
        """Light from the needle tip: a halo round the crown, rays fanning down over the skyline, beacon
        pulses and a shower of gold sparks."""
        k = _env(a, BEACON_LEN, rise=6, fall=24)
        fx.glow(c, x, y + 34, 34, 'butter', 0.30 * k, rings=5)
        fx.glow(c, x, y, 14 + a // 5, 'butter', 0.65 * k, rings=4)
        fx.glow(c, x, y, 6, 'white', 0.8 * k, rings=2)
        for i in range(13):                            # rays (the upper half is above the frame)
            ang = math.pi * (i + 0.5) / 13 + 0.06 * math.sin(a / 7 + i)
            L = (22 + 40 * k) * (1.0 if i % 2 else 0.65)
            for r in range(6, int(L), 2):
                if hash01(i, r, a // 3) < 0.3 + 0.55 * k:
                    c.px(x + int(round(math.cos(ang) * r)), y + int(round(math.sin(ang) * r)),
                         'white' if r < L * 0.5 else 'butter')
        for j in range(3):                             # beacon pulses
            r = (a - j * 12) * 1.6 + 4
            if 4 <= r < 58:
                c.circle(x, y, int(r), 'white' if j % 2 == 0 else '#fff4b0', fill=False)
        ui.burst(c, t, t - a, x, y + 4, colors=('gold', 'white', 'butter'), n=34, seed=3, power=0.85,
                 life=64, gravity=0.05)
        if a >= 20:
            ui.burst(c, t, t - a + 20, x, y + 12, colors=('butter', 'white', 'hot_pink'), n=24, seed=4,
                     power=0.7, life=44, gravity=0.06)

    @staticmethod
    def _horn(c, a, x, y):
        """The ferry's horn, three blasts: a steam puff from the funnel and sound arcs rolling out with
        each, the puffs drifting off on the breeze."""
        for b in range(3):
            ab = a - b * 14                            # this blast's age
            if ab < 0:
                continue
            for i in range(3):                         # its puffs
                age = ab - i * 3
                if not 0 <= age < 44:
                    continue
                r = int(2 + min(4, age / 4)) - (i == 2)
                px_ = x - 2 - int(age * 0.8) - i * 3
                py_ = y - 3 - int(age * 0.9) - i * 2 + b * 2
                if age < 40 or (i + age) % 2:
                    c.circle(px_, py_ + 1, r, '#b8bcc8')
                    c.circle(px_, py_, r, 'white' if age < 26 else '#e4e8ee')
            if ab < 26:                                # sound arcs to the right of the funnel
                for j in range(2):
                    r = 5 + (ab - j * 5) * 1.1
                    if not 5 <= r < 26:
                        continue
                    for d in range(-50, 51, 7 if r < 14 else 5):
                        rad = math.radians(d)
                        c.px(x + 2 + int(round(math.cos(rad) * r)), y - 1 + int(round(math.sin(rad) * r)),
                             'white' if j == 0 else '#d8f4ff')
        if a < 46:                                     # propeller churn at the stern
            for i in range(12):
                if (i + a) % 3:
                    c.px(x - 10 - int(hash01(i, a // 2) * 10), y + 10 + (i % 3), 'white')

    # -------------------------------------------------------------- area 1: Bridleway
    def render_street(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        ph = self.el_phase(t)
        a = t - self.spot_t.get('marquee', -999)
        lit = _Lit()
        lamps, wins = _lamp_share(hour), nightwin.ramp(hour)
        on_night = lamps > 0
        pts = []                                         # (y, x, rgb) of the bulbs and blade icons lit
        kw = dict(cam_b=lambda _f: cx, draw_balloon=_nop, callout=_nop, T_B=f - 4 - ph, glint=_glint(hour),
                  draw_clouds=_clouds(hour))
        show = 0 <= a < SHOW_LEN
        hot = show and a < SHOW_LEN - 10
        if on_night:
            kw['el_train'] = _el_train_lit

        def bulb(c, x, y, on, flash):
            if not hot:
                sc.bulb(c, x, y, on, flash)
                if on_night and (on or flash) and 0 <= x < W and 0 <= y < H:
                    pts.append((y, x, tuple(c.a[y, x])))
                return
            if a % 9 < 2:
                cc = [(x, y, 'white')]
            else:
                ph = ((x + y) // 2 - a) % 4
                cc = [(x + dx, y + dy, '#fff4b0') for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))] if ph == 0 else []
                cc.append((x, y, ('white', '#ff9ad0', '#fff4b0', '#ffd84a')[ph]))   # the chase's head flares
            for xx, yy, k in cc:
                c.px(xx, yy, k)
                if on_night and 0 <= xx < W and 0 <= yy < H:
                    pts.append((yy, xx, tuple(c.a[yy, xx])))

        def icon(cv, nm, x, y, color, hi='white', scale=1):
            on = color == '#ffe25a'
            if show:
                on = (a // 4) % 2 == 0
                color, hi = ('white', '#ffe25a') if on else ('#ffe25a', 'white')
                on = True
            if not on_night or not on:
                return sc.icon(cv, nm, x, y, color, hi=hi, scale=scale)
            y0, x0 = max(0, y - 1), max(0, x - 1)
            before = cv.a[y0:y + 8 * scale, x0:x + 8 * scale].copy()
            sc.icon(cv, nm, x, y, color, hi=hi, scale=scale)
            yy, xx = np.nonzero(np.any(cv.a[y0:y + 8 * scale, x0:x + 8 * scale] != before, axis=2))
            pts.extend((y0 + j, x0 + i, tuple(cv.a[y0 + j, x0 + i])) for j, i in zip(yy, xx))

        kw.update(bulb=bulb, icon=icon)
        c = rebind(sc.shot_b, **kw)(f)
        if on_night:
            self._street_lights(c, f, cx, lit, pts, lamps, wins)
            lit.add(c, nightwin.paint(c, hour, t, _pale_windows(
                (('b_far', -int(cx * sc.B_P['far']), 0), ('b_street', -int(cx * sc.B_P['st']), 0)))))
        if show:
            before = c.a.copy()
            self._show(c, t, a, cx, day=not on_night)
            if on_night:
                lit.drawn(c, before)
            self._later(lambda cc: self._searchlights(cc, a, cx))
        if on_night:
            self.keep_glow(c, lit.mask(c), 0.95)
        return c

    @staticmethod
    def _street_lights(c, f, cx, lit, pts, lamps, wins):
        """Bridleway by night: the bulbs and blade icons lit by the film's chase (pts), the marquee boards,
        the lights under the marquees and the billboard glowing, lobby doors and then shop windows lit,
        window panes and lamp globes, the far towers' red beacons, the el train's windows and the traffic
        light, all collected in `lit`."""
        P = sc.B_P
        so = -int(cx * P['st'])
        if pts:
            ys, xs, rgb = zip(*pts)
            ok = np.all(c.a[list(ys), list(xs)] == np.array(rgb, np.uint8), axis=1)
            lit.px(c, np.array(ys)[ok], np.array(xs)[ok])
        rgba = sc.b_street()[0]
        ys, xs = _lightboxes()
        sx = xs + so
        ok = (sx >= 0) & (sx < W)
        ys, xs, sx = ys[ok], xs[ok], sx[ok]
        seen = np.all(c.a[ys, sx] == rgba[ys, xs, :3], axis=1)
        lit.px(c, ys[seen], sx[seen])
        gy, gx, gv, grgb, gsrc = _street_glass()
        sx = gx + so
        ok = (sx >= 0) & (sx < W) & np.where(gv == 0, lamps > 0, gv < wins)
        ok[ok] &= np.all(c.a[gy[ok], sx[ok]] == gsrc[ok], axis=1)
        c.a[gy[ok], sx[ok]] = grgb[ok]
        lit.px(c, gy[ok], sx[ok])
        _night_lights(c, None, (('b_far', -int(cx * P['far']), 0), ('b_street', so, 0),
                                ('b_fg', -int(cx * P['fg']), 0)), lit, (wins, lamps))
        fo = -int(cx * P['far'])
        if (f // 9) % 4 == 0:
            red = np.array(col('#ff7a6a'), np.uint8)
            for (x, y) in sc.b_far()[1].get('beacons', []):
                if 0 <= x + fo < W and np.all(c.a[y, x + fo] == red):
                    lit.px(c, [y], [x + fo])
        lit.marks(c, 30, 56)                             # the el train's windows
        lx, k = 96 - int(cx * P['fg']), 2 - (f // 36) % 3
        lit.px(c, [81 + k * 4] * 2, [lx + 1, lx + 2])     # the traffic light's lamp that is on

    @staticmethod
    def _searchlights(c, a, cam):
        """The premiere show's two searchlights from the grand theatre, swept over the frame after the
        clock's light and sky (on the painted noon sky they would show as noon-blue stripes at dusk)."""
        k = _env(a, SHOW_LEN, rise=6, fall=18)
        gx, gy = GRAND[0] - cam, GRAND[1]
        for s in (-1, 1):
            ang = s * 0.32 + 0.38 * math.sin(a / 10 + (0 if s < 0 else 2.2))
            _beam(c, gx + s * 6, gy, ang, 74, 20, 0.42 * k)

    @staticmethod
    def _show(c, t, a, cam, day=True):
        """Premiere show: the marquees glow (by day; at night their bulbs and boards blaze instead of a
        halo over the facades), confetti pops from every marquee on the beats (the searchlights:
        _searchlights)."""
        k = _env(a, SHOW_LEN, rise=6, fall=18)
        gx, gy = GRAND[0] - cam, GRAND[1]
        if day:
            fx.glow(c, gx, gy, 7, 'white', 0.6 * k, rings=2)
        beat = 1 - seg(a % 18, 0, 10)
        for i, (x0, x1) in enumerate(MARQUEES):
            mx = (x0 + x1) // 2 - cam
            if not -70 < mx < W + 70:
                continue
            if day:
                fx.glow(c, mx, MARQ_Y + 8, (x1 - x0) // 2 + 6, 'butter', (0.18 + 0.22 * beat) * k, rings=4)
            for j in range(4):                         # confetti pops from the marquee top on the beats
                ui.burst(c, t, t - a + j * 18 + i * 3, mx + (j % 2 * 2 - 1) * (x1 - x0) // 4, MARQ_Y - 1,
                         n=36, seed=11 + i * 5 + j, power=1.0, life=52, gravity=0.06,
                         colors=('hot_pink', 'butter', 'baby_blue', 'mint', 'white', 'lavender'))
            for j in range(3):                         # big twinkles on the bulbs
                ph = (a + i * 5 + j * 7) % 14
                fx.sparkle(c, x0 - cam + int(hash01(i, j, a // 14) * (x1 - x0)), MARQ_Y + (1 if j % 2 else 14),
                           ph, 'white', period=14, size=3)

    # -------------------------------------------------------------- area 2: the park
    def render_park(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        a = t - self.spot_t.get('fountain', -999)
        lit = _Lit()
        lamps = _lamp_share(hour)
        wat = []

        def water(c, f_, wy, cols, **kwargs):          # the show jets go in before the lake reflects them
            if 0 <= a < FOUNT_LEN:
                self._fountain(c, t, a, FOUNT[0] - cx)
            wat.append(_water(c, f_, wy, cols, hour, self.HOUR, **kwargs))

        kw = dict(cam_c=lambda _f: cx, draw_balloon=_nop, callout=_nop, water=water, glint=_glint(hour),
                  draw_clouds=_clouds(hour))
        c = rebind(sc.shot_c, **kw)(f)
        if lamps > 0:                                    # the park lamps, then Bronclyn's windows
            P = sc.C_P
            _night_lights(c, hour, (('c_far', -int(cx * P['far']), 0), ('c_brown', -int(cx * P['brown']), 0),
                                    ('c_trees', -int(cx * P['trees']), 0), ('c_bank', -int(cx * P['bank']), 0),
                                    ('c_fg', -int(cx * P['fg']), 0)), lit)
            lit.add(c, nightwin.paint(c, hour, t, _pale_windows(
                (('c_far', -int(cx * P['far']), 0), ('c_brown', -int(cx * P['brown']), 0)))))
            _reflect(c, lit, wat[0] if wat else None, f)
            self.keep_glow(c, lit.mask(c), 0.95)
        return c

    @staticmethod
    def _fountain(c, t, a, x):
        """The fountain's show: a tall central jet with a spray crown, high side arcs, mist and a little
        rainbow (drawn before the lake, so all of it is reflected)."""
        k = _env(a, FOUNT_LEN, rise=8, fall=24)
        fty, fby = FOUNT[1], FOUNT[2]
        hj = int(44 * k)
        top = fty - 4 - hj
        fx.glow(c, x, fby - 4, 22, 'white', 0.25 * k, rings=3)
        if k > 0.55 and a > 10:                        # a little rainbow in the spray
            for i, cc in enumerate(('#ff6a6a', '#ffc84a', '#7ad86a', '#6ab0ff', '#b08aff')):
                R = 27 - i
                for d in range(196, 345, 3):
                    rad = math.radians(d)
                    if (d // 3 + a // 2) % 7:
                        c.px(x + int(round(math.cos(rad) * R)), fby + 2 + int(round(math.sin(rad) * R)), cc)
        if hj > 1:                                     # the central jet: 3 px with a blue rim, shimmering
            for y in range(max(0, top), fty - 3):
                sh = (y + a) % 4
                c.px(x - 2, y, '#5ab0d0')
                c.px(x - 1, y, 'white' if sh else '#bfefff')
                c.px(x, y, '#d8f6ff' if sh == 1 else 'white')
                c.px(x + 1, y, 'white' if sh != 2 else '#bfefff')
                c.px(x + 2, y, '#5ab0d0')
            for s in (-1, 1):                          # spray crown falling back from the top
                for j in range(16):
                    u = j / 15
                    xx = x + s * int(round(2 + u * 18))
                    yy = top + int(round((u * u * 1.5 - u * 0.45) * hj * 0.75))
                    if (j + a // 2) % 4:
                        c.rect(xx, yy, 2 if j % 3 else 1, 2 if j % 2 else 1, 'white' if j % 2 else '#bfefff')
                        c.px(xx + s, yy + 2, '#5ab0d0')
        hs = int(18 * k)
        for s in (-1, 1):                              # the bowl jets arc higher and wider
            for j in range(12):
                u = j / 11
                xx = x + s * int(round(u * 14))
                yy = fby - 1 - int(round(hs * (1 - (2 * u - 0.5) ** 2)))
                c.rect(xx, yy, 1, 2, 'white' if (j + a // 2) % 3 else '#bfefff')
        for j in range(4):                             # droplet bursts while the jet is up
            ui.burst(c, t, t - a + 6 + j * 14, x, max(top, 2), colors=('white', '#bfefff', '#5ab0d0'), n=22,
                     seed=40 + j, power=0.75, life=40, gravity=0.09, size=2)

    # -------------------------------------------------------------- area 3: the rail bridge
    def render_bridge(self, f, cx, cy):
        t, hour = self.g.t, self.g.hour
        X, Y = cx, -cy
        P = sc.D_P
        lit = _Lit()
        lamps = _lamp_share(hour)
        c = sc.sky_layer().copy()
        _clouds(hour)(c, f, sc.D_CLOUDS, ox=X * 0.06)
        ho = -int(X * P['far'])
        sc.paste(c, sc.d_hills(), ho, Y)
        haze, haze_rgb = _haze_of(c, 'd_hills', _D_HAZE, ho, Y), c.a.copy()
        wins = nightwin.ramp(hour)
        for i in range(12):                              # the village across the river: windows lit
            if hash01(i, 414) < wins * 0.75:             # here and there (sc.d_hills' houses)
                x = 250 + int(hash01(9, i) * 150) + 1 + ho
                y = sc.D_WY - 12 - int(hash01(9, i, 2) * 6) + 2 + Y
                if 0 <= x < W and 0 <= y < H and tuple(c.a[y, x]) == col('#5a86b8'):
                    c.px(x, y, '#ffd466')
                    lit.px(c, [y], [x])
        back, _ = sc.d_back()
        sc.paste(c, back, -int(X * P['back']) - 110, Y + 12)
        city, cmeta = sc.d_city()
        co = -int(X * P['city']) - 100
        cy_ = Y + 18
        sc.paste(c, city, co, cy_)
        sc.paste(c, _d_quay(), co, cy_)
        c.rect(co + 236, sc.A_WY - 6 + cy_, 2, 6, '#8c806c')
        ccx, cty, _ = cmeta['crystaller']
        glint = _glint(hour)
        for (x, y, ph) in cmeta.get('glints', []):
            if x < sc.CRYS_X + 64:
                glint(c, x + co, y + cy_, f, ph, period=72)
        for i, (x, y) in enumerate(cmeta.get('flags', [])):
            if x < sc.CRYS_X + 60:
                sc.flag(c, x + co, y + cy_, f, ('#e83b3b', '#3f6fc8', '#f2c24a', '#3f9a86')[i % 4], seed=i)
        if f % 18 < 6:                                   # the needle's glint (by night: its light)
            before = c.a.copy()
            fx.sparkle(c, ccx + co, cty + cy_, f % 18, 'white' if lamps <= 0 else 'butter', period=12, size=3)
            if lamps > 0:
                lit.drawn(c, before)
        # bridge
        bridge, meta = sc.d_bridge()
        bo, byo = -X, Y
        sc.paste(c, bridge, bo, byo)
        for i, (x, y) in enumerate(meta['flags']):
            sc.flag(c, x + bo, y + byo + 3, f, ('#f04f78', '#3f6fc8')[i], seed=i + 5, w=5)
        if lamps > 0:                                    # the bridge lights up
            before = c.a.copy()
            self._bridge_lights(c, f, lamps, bo, byo)
            lit.drawn(c, before)
        k = self.train_k(t)
        kt = t - self.toot_t if self.toot_t is not None else -1
        before = c.a.copy()
        for tx in (sc.T1, sc.T2):                        # tower emblems: flash on the hit / a toot, else beats
            if k is not None and 0 <= k < 18 or 0 <= kt < 18:
                kk = k if k is not None and 0 <= k < 18 else kt
                if kk < 8:
                    fx.sparkle(c, tx + bo, sc.TTOP + 22 + byo, kk, 'white', period=16, size=4)
            elif f % 18 < 8:
                fx.sparkle(c, tx + bo, sc.TTOP + 22 + byo, f % 18, 'white', period=16, size=2)
        if lamps > 0:                                    # (by night the lit emblems flash)
            lit.drawn(c, before)
        if k is not None:
            self._train(c, k, bo, byo, lamps > 0)
        for tx in (sc.T1, sc.T2):                        # tower legs + portal in front of the train
            sub = bridge[sc.D_DECK - 40:sc.D_DECK + 1, tx - sc.TW // 2 - 1:tx + sc.TW // 2 + 1]
            sc.paste(c, sub, tx - sc.TW // 2 - 1 + bo, sc.D_DECK - 40 + byo)
        # river, the city's lights (after the water: it mirrors the painted city) and the vessels
        # (looping where no camera sees them)
        wy = sc.D_WY + byo
        wat = _water(c, f, wy, ['#80c4ea', '#58a2d4', '#3e84bc', '#2f6aa6'], hour, self.HOUR, refl=0.42, seed=7,
                     src_y0=20, sun_x=236)
        lay = (('d_back', -int(X * P['back']) - 110, Y + 12), ('d_city', co, Y + 18))
        _night_lights(c, hour, lay, lit)
        lit.add(c, nightwin.paint(c, hour, t, _pale_windows(lay)))
        boat = _boat_lit if lamps > 0 else (lambda s: s)
        sx = _sail(332, -0.4, f, 1.1, CAM_D, sc.SAIL.w) - int(X * 1.1)
        sc.boat(c, sc.SAIL, sx, wy + 12, f, flip=True, seed=11)
        fx_ = _sail(110, 0.5, f, 1.2, CAM_D, sc.FERRY.w) - int(X * 1.2)
        sc.boat(c, boat(sc.FERRY), fx_, wy + 30, f, seed=14)
        sc.smoke_puffs(c, fx_ + 8, wy + 20, f, n=4, drift=-0.2, seed=15)
        tx_ = _sail(524, -0.3, f, 1.15, CAM_D, sc.TUG.w) - int(X * 1.15)
        sc.boat(c, boat(sc.TUG), tx_, wy + 26, f, flip=True, seed=12)
        sc.smoke_puffs(c, tx_ + 5, wy + 18, f, n=5, drift=0.25, seed=13, color='#d8d8e0', dark='#9a96a8')
        if lamps > 0:
            lit.marks(c, 0, H)                           # cabin windows: the train's and the boats'
            _reflect(c, lit, wat, f)
        sc.paste(c, sc.a_fg(), -110 - int(X * P['fg']), Y)
        if lamps > 0:
            self.keep_glow(c, lit.mask(c), 0.95)
        self._haze_later(c, haze & np.all(c.a == haze_rgb, axis=2), int(cy))
        return c

    @staticmethod
    def _bridge_lights(c, f, k, bo, byo):
        """The bridge by night: a necklace of warm bulbs along the main cables, switched on from the
        towers outward as the night comes (k 0..1, the lamp share), one twinkling now and then; lamps
        along the deck; the towers' emblems lit (no halo: only the light itself shines); red lights
        blinking on the tower tops."""
        for x in range(sc.AL + 2, sc.AR - 1, 4):
            if min(abs(x - sc.T1), abs(x - sc.T2)) > 40 + 200 * k:
                continue
            sx, sy = x + bo, int(round(sc.cable_y(x))) + byo
            if 0 <= sx < W and 0 <= sy < H:
                tw = hash01(x, f // 8, 3) < 0.07
                c.px(sx, sy, 'white' if tw else ('butter' if (x // 4) % 2 else '#ffd27a'))
        dy = sc.D_DECK + byo
        for x in range(sc.AL + 10, sc.AR - 4, 16):          # deck lamps (none inside the towers)
            if min(abs(x - sc.T1), abs(x - sc.T2)) <= sc.TW // 2 + 1 or hash01(x, 413) > k * 1.2:
                continue
            sx = x + bo
            if 0 <= sx < W:
                c.vline(sx, dy - 6, dy - 3, '#3a3048')
                c.px(sx, dy - 7, '#ffe27a')
                c.px(sx - 1, dy - 7, '#c8843a')
                c.px(sx + 1, dy - 7, '#c8843a')
        for tx in (sc.T1, sc.T2):
            ex, ey = tx + bo, sc.TTOP + 22 + byo
            if -12 < ex < W + 12:
                c.circle(ex, ey, 2, 'butter')
                c.px(ex, ey, 'white')
                if (f // 18) % 2 == 0:                       # aviation lights on the top corners
                    for ox in (-sc.TW // 2 + 1, sc.TW // 2 - 2):
                        c.px(tx + ox + bo, sc.TTOP + 1 + byo, '#ff4a3a')

    def _train(self, c, k, bo, byo, night=False):
        """The Friendship Express, k steps after its hit (the film's shot D with f - D_HIT = k); by night
        its windows and headlamp lit."""
        front = self.train_front
        tr, fun_x, whi_x = sc.d_train()
        if night:
            tr = _train_lit()
        tw = tr.shape[1]
        ty_ = sc.D_DECK - 2 - tr.shape[0] + 1 + byo
        # trailing steam plume: puffs stay where they were emitted and rise
        for j in range(16):
            ke = k - j * 3 - (k % 3)
            if ke < -TRAIN_LEAD:
                break
            age = k - ke
            sx = front(ke) - tw + fun_x + bo - int(age * 0.25)
            sy = ty_ - int(age * 0.55) - 1
            r = 1 + min(4, age // 7)
            cc = '#ffffff' if age < 18 else ('#eef0f6' if age < 32 else '#d8dce8')
            if age < 46:
                c.circle(int(sx), sy + 1, r, '#c8ccd8')
                c.circle(int(sx), sy, r, cc)
        sc.paste(c, tr, int(round(front(k))) - tw + bo, ty_)
        # whistle plume on the hit, a second toot half a bar later, and any toot the player asked for
        toots = [(0, 1.0), (36, 0.6)]
        if self.toot_t is not None:
            toots.append((self.toot_t - self.train_hit, 1.0))
        for k0, big in toots:
            if k0 <= k < k0 + 22:
                kk = k - k0
                wx = int(round(front(k))) - tw + whi_x + bo
                for i in range(7):
                    age = kk - i * 2
                    if age < 0:
                        continue
                    r = int((2 + min(4, age / 2)) * big + 0.5)
                    px_ = wx - int(age * 0.7) - (i % 2) * 2
                    py_ = ty_ - 4 - int(age * 1.4 * big) - i * 2
                    if age < 20:
                        c.circle(px_, py_ + 1, r, '#c8ccd8')
                        c.circle(px_, py_, r, 'white' if age < 12 else '#eef0f6')
