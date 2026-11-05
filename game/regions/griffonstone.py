"""Griffonstone (+ Trottingham) as an explorable diorama (art: scenes/s10_griffonstone.py, used read-only).

The film shows two places in two shots, so the region has two areas, in film order (recompose style:
the scene's cached layers and animation helpers, in the film's layer order, without the film's balloon,
callouts, card, iris or shot change):

  area 0  TROTTINGHAM   across the strait: the Griffish Isle town, castle, harbour and beacon, the
                        two-master sailing the whitecapped sea, spray on the foreground rocks. The pan
                        runs on past the isle over open sea towards the griffon mainland and its spire.
  area 1  GRIFFONSTONE  the rock spire, panned AND tilted: Griffonstone Station and the Friendship
                        Express at the gorge base, the houses on the stone branches, the ruined castle
                        on the crown, and the rope bridge over the Abysmal Abyss to the far pinnacle.

Wind: the film's wind strength (steady breeze with swells) drives every wind helper, and its phases are
integrated over time (the film multiplies the scene frame by a wind-dependent speed, which only works for
a 4-bar shot). Timed beats on game clocks: in Griffonstone a GUST blows every GUST_PERIOD steps (banners
whip, pines bend, a debris blast, the rope bridge swings hard, the balloon gets a shove); on the first gust
the player watches, the loose plank snaps and tumbles into the abyss. The engine steams at the station
with a cylinder-hiss double puff and a whistle on a train clock. The ship sails the strait in a loop
(re-entering off screen) and the sea spray bursts on the rocks every beat, as in the film.

Camera: the Griffonstone tilt runs 1.3x ahead of the balloon (looks up at the crown near the top, down on
the station near the floor), and the balloon keeps its relative height through the cloud sweep between the
areas (skim the waves at Trottingham -> the gorge floor; fly high -> the crown).

SPACE (hotspots): the harbour beacon flares and sweeps the strait; the church bell rings; on the crown's
throne terrace you call the GUST; the Friendship Express whistles and blows off steam; lightning strikes
the far pinnacle (one bolt and one soft, ramped wash of light: no strobing flicker). Stamps: far out over the strait, above the castle's broken tower, deep in the chasm by
the river's fall.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, clamp, hash01, mix, col, bayer
import scenes.s10_griffonstone as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# ----------------------------------------------------------------------------- wind
GUST_PERIOD = 600       # one big gust every 20 s in Griffonstone
GUST_LEAD = 150         # steps from arriving in Griffonstone to the first gust
SNAP_S = 18             # the loose plank snaps this long into the gust (film 216 -> 234)


def _base_w(f):
    """The film's wind strength without the gust (gust_k for f < GUST)."""
    return 0.45 + 0.15 * math.sin(f / 23.0) + 0.1 * math.sin(f / 9.0 + 1.3)


def _base_int(f):
    """Integral of _base_w: the wind phase."""
    return 0.45 * f - 0.15 * 23.0 * math.cos(f / 23.0) - 0.1 * 9.0 * math.cos(f / 9.0 + 1.3)


def _gust_w(s):
    """The film's gust term, s frames after the gust (gust_k(GUST + s) - base)."""
    if s < 0:
        return 0.0
    return 0.9 * math.exp(-max(0.0, s - 4) / 30.0) * min(1.0, s / 14.0 * 3)


_GUST_INT = np.concatenate([[0.0], np.cumsum([_gust_w(s) for s in range(GUST_PERIOD)])])


# ----------------------------------------------------------------------------- fast compositing
_LAYERS = {
    'deck_far': lambda: sc._deck('far'),
    'deck_near': lambda: sc._deck('near'),
    'far_isles': lambda: sc._far_layers()[0],
    'isle': lambda: sc._isle_layer()[0],
    'rocks': lambda: sc._fg_rocks()[0],
    'gs_far': sc._gs_far,
    'gs_mid': sc._gs_mid,
    'abyss': lambda: sc._gs_abyss()[0],
    'world': lambda: _world_rgba(),
}


def _world_rgba():
    """The main layer, with the far pinnacle's one hard shading step (a straight band edge at y ~336 that
    the film's camera never reaches) broken up: a wavy, grainy transition over ~50 rows, so the deep
    abyss view shows rock, not a seam."""
    a = sc._gs_world()[0].copy()
    pal = np.array([col(c_) for c_ in sc.ROCK], np.uint8)
    lut = {tuple(int(v) for v in p): i for i, p in enumerate(pal)}
    by, gh = sc.BR_B[1], sc.GH
    step = next(y for y in range(by, gh) if int((y - by) / (gh - by) * 1.5) >= 1)
    wave = sc._tnoise(4, sc.GW, (1.0, 7.0), 71)[0]
    blot = sc._tnoise(60, sc.GW, (1.4, 2.4), 73)                 # clumpy, like the rock's own blots
    grain = (np.argsort(np.argsort(blot, axis=None)) / blot.size).reshape(blot.shape)
    for y in range(step - 30, step + 30):
        t = (y - by) / (gh - by)
        hw = 34 + 14 * t + 4 * math.sin(y / 11.0) + 2 * math.sin(y / 4.0)
        cx = 512 + 8 * math.sin(y / 40.0)
        for x in range(max(0, int(cx - hw) + 1), min(sc.GW, int(cx + hw) - 1)):
            k = lut.get(tuple(int(v) for v in a[y, x, :3]))
            if k is None:
                continue
            mid = step + (wave[x] - 0.5) * 26
            lower = grain[y - step + 30, x] < clamp((y - mid + 14) / 28.0)
            if lower and y < step:
                k -= 1
            elif not lower and y >= step:
                k += 1
            a[y, x, :3] = pal[max(0, min(7, k))]
    return a


@lru_cache(maxsize=None)
def _layer(name):
    """A scene RGBA layer split into (rgb, opaque mask), as Canvas.blit reads it (alpha >= 128)."""
    a = _LAYERS[name]()
    return np.ascontiguousarray(a[..., :3]), a[..., 3] >= 128


def _put(c, rgb, m, x, y):
    """Canvas.blit of an (rgb, mask) pair with its top-left at (x, y), ~4x faster (np.copyto)."""
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0:y1, x0:x1], rgb[y0 - y:y1 - y, x0 - x:x1 - x], where=m[y0 - y:y1 - y, x0 - x:x1 - x, None])


def _paste(c, name, x, y):
    _put(c, *_layer(name), int(x), int(y))


def _paste_wrap(c, name, x_off, y):
    """sc._blit_wrap: screen column 0 shows layer column x_off (the layer tiles horizontally)."""
    rgb, m = _layer(name)
    o = int(x_off) % m.shape[1]
    _put(c, rgb, m, -o, y)
    if m.shape[1] - o < W:
        _put(c, rgb, m, m.shape[1] - o, y)


def _beam(c, top, bottom, color, strength):
    """fx.light_beam in one numpy pass (same rows, edges, fade and dither)."""
    (txl, ty), (txr, _) = top
    (bxl, by), (bxr, _) = bottom
    y0, y1 = max(0, int(ty)), min(H, int(by) + 1)
    if y1 <= y0:
        return
    ys = np.arange(y0, y1)
    t = np.clip((ys - ty) / (by - ty), 0.0, 1.0) if by > ty else (ys >= by).astype(np.float64)
    xl = np.trunc(txl + (bxl - txl) * t).astype(int)
    xr = np.trunc(txr + (bxr - txr) * t).astype(int)
    lo = np.clip(np.minimum(xl, xr), 0, W)
    hi = np.clip(np.maximum(xl, xr) + 1, 0, W)
    x0, x1 = int(lo.min()), int(hi.max())
    if x1 <= x0:
        return
    xs = np.arange(x0, x1)[None, :]
    m = (xs >= lo[:, None]) & (xs < hi[:, None]) & (bayer(y1 - y0, x1 - x0, x0, y0) < (1 - t * 0.7)[:, None])
    reg = c.a[y0:y1, x0:x1]
    reg[m] = np.minimum(reg[m].astype(np.int16) + np.array(col(color), np.float32) * strength, 255).astype(np.uint8)


def _sun_shaft(c, gap_x, gy, ground_y, f, strength=0.15, slant=-0.35):
    """sc.sun_shaft with the vectorised beam."""
    sw = 9 + int(2 * math.sin(f / 17.0))
    bx = gap_x + int((ground_y - gy) * slant)
    _beam(c, ((gap_x - sw, gy + 6), (gap_x + sw, gy + 6)), ((bx - sw - 16, ground_y), (bx + sw + 18, ground_y)),
          '#fff1c4', strength)
    _beam(c, ((gap_x - 3, gy + 6), (gap_x + 4, gy + 6)), ((bx - 6, ground_y), (bx + 8, ground_y)),
          '#fffbe0', strength * 0.8)


# ----------------------------------------------------------------------------- the world clock's lights
# The film paints an overcast afternoon with half the windows lit (WARM glass), the lanterns lit with a
# dithered halo and the harbour beacon blinking. By the world clock: by day every window is dark glass and
# the lanterns and the beacon are off; from the dusk ramp on (nightwin.ramp) each window goes on at its own
# point (the film's lit ones first, then half of its dark ones), the lanterns' glass (no halo) and the
# beacon (a steady lamp, a flash every three beats, its broken reflection on the sea), and they keep their
# colour through the night's grade (GlowKeep), as do the beacon's sweep, the bell's rings and the lightning.
# The relight's own glow keeping is off: it kept painted warm spots (gold flag tips, the train's trim).
UNLIT = {'isle': ('#3a4048', '#6c7680'), 'world': ('#2f3035', '#2f3035')}   # dark glass: pane, glint
LAMP_OFF = '#7d8790'                                # the beacon's lamp glass by day


def _eq(rgb, c_):
    return np.all(rgb == np.array(col(c_), np.uint8), axis=-1)


@lru_cache(maxsize=None)
def _windows(name):
    """The windows of scene layer `name`: (ys, xs, v, lit rgb, unlit rgb) per pixel (layer px; v: the dusk
    ramp at which it goes on), and a layer-sized map of each window's v (nan: no window) for the film's
    flicker spots. The film's lit glass (WARM) is a window lit first; its dark glass (UNLIT, the pane's
    size) is one of the later ones, or stays dark."""
    from scipy import ndimage
    rgb, m = _layer(name)
    pane, glint = (np.array(col(c_), np.uint8) for c_ in UNLIT[name])
    w0, w1 = (np.array(col(c_), np.uint8) for c_ in sc.WARM)
    vmap = np.full(m.shape, np.nan, np.float32)
    out = []
    lit = m & (_eq(rgb, sc.WARM[0]) | _eq(rgb, sc.WARM[1]))
    lab, n = ndimage.label(lit, structure=np.ones((3, 3)))
    for i, s_ in enumerate(ndimage.find_objects(lab), 1):
        if s_[0].stop - s_[0].start > 6 or s_[1].stop - s_[1].start > 4:
            continue
        ys, xs = np.nonzero(lab[s_] == i)
        ys, xs = ys + s_[0].start, xs + s_[1].start
        v = hash01(int(xs[0]), int(ys[0]), 611) * 0.8
        off = np.where(_eq(rgb[ys, xs], sc.WARM[1])[:, None], glint, pane)
        out.append((ys, xs, v, rgb[ys, xs], off))
    dark = m & _eq(rgb, UNLIT[name][0])
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    size = 2 if name == 'isle' else 3
    for i, s_ in enumerate(ndimage.find_objects(lab), 1):
        if s_[0].stop - s_[0].start != size or s_[1].stop - s_[1].start != size:
            continue
        y0, x0 = s_[0].start, s_[1].start
        if hash01(x0, y0, 612) >= 0.5:                   # (half of them stay dark all night)
            continue
        ys, xs = (a.ravel() for a in np.mgrid[y0:y0 + size, x0:x0 + size])
        v = 0.45 + hash01(x0, y0, 613) * 0.5
        on = np.where(((ys == y0) & (xs == x0))[:, None], w1, w0)
        out.append((ys, xs, v, on, rgb[ys, xs]))
    for ys, xs, v, _on, _off in out:
        vmap[ys, xs] = v
    ys, xs = (np.concatenate([o[i] for o in out]) for i in range(2))
    v = np.concatenate([np.full(len(o[0]), o[2], np.float32) for o in out])
    on, off = (np.concatenate([o[i] for o in out]) for i in (3, 4))
    return ys, xs, v, on, off, vmap


def _put_windows(c, name, x, y, k, lit):
    """Switch the windows of layer `name` (just pasted at (x, y)) for the dusk ramp k: on in their warm
    glass, off in dark glass. The lit pixels' screen spots go into `lit` (for keep_glow)."""
    ys, xs, v, on, off, _vm = _windows(name)
    sy, sx = ys + int(y), xs + int(x)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if not ok.any():
        return
    sy, sx, v, on, off = sy[ok], sx[ok], v[ok], on[ok], off[ok]
    lit_ = v < k
    c.a[sy, sx] = np.where(lit_[:, None], on, off)
    if lit_.any():
        lit.append((sy[lit_], sx[lit_]))


def _win_on(name, lx, ly, k):
    """Is the window at layer px (lx, ly) on at the dusk ramp k? (the film's flicker spots)"""
    vm = _windows(name)[5]
    return 0 <= ly < vm.shape[0] and 0 <= lx < vm.shape[1] and vm[ly, lx] < k


def _lantern(c, x, y, f, k, seed, on, lit):
    """sc._lantern without its dithered halo: the lantern swings; when on, its glass row is lit (and goes
    into `lit`), else it is dark glass."""
    ang = math.sin(f * 0.21 + seed * 2.3) * (0.25 + 0.6 * k)
    lx = x + int(round(math.sin(ang) * 3))
    ly = y + 3
    c.line(x, y, lx, ly, '#3a3f46')
    c.rect(lx - 1, ly, 3, 3, '#3a3f46')
    if not on:
        c.px(lx, ly + 1, '#5e5a52')
        return
    c.hline(lx - 1, lx + 1, ly + 1, sc.WARM[0])
    c.px(lx, ly + 1, sc.WARM[1] if hash01(seed, f // 4) > 0.15 else '#fff1c4')
    xs = np.arange(lx - 1, lx + 2)
    ok = (xs >= 0) & (xs < W) & (0 <= ly + 1 < H)
    if ok.any():
        lit.append((np.full(int(ok.sum()), ly + 1), xs[ok]))


@lru_cache(maxsize=16)
def _unlit_strip(hq, top, span, fq, tq):
    """The clock's sky strip (daylight._sky_strip) as the colours to paint into the scene so that its
    relight from fq to tq shows it (daylight.unlight)."""
    return daylight.unlight_array(daylight._sky_strip(hq, top, span), fq, tq)   # (~3 ms; per colour: 70)


def _sun_x(hour):
    """Screen x of the clock's sun on daylight's arc (held at its ends outside the day)."""
    u = (min(max(hour % 24, 5.6), 19.2) - 5.6) / 13.6
    return W * (0.06 + 0.88 * u)


PAINTED_SUN_X = _sun_x(14.63)


def gap_shift(hour):
    """How far the overcast's bright gap sits from the film's (painted 14:38) spot: over the clock's sun."""
    return int(round(_sun_x(hour) - PAINTED_SUN_X))


def slant_k(hour):
    """The sun shafts' slant relative to the film's (afternoon sun right of the zenith: down to the left);
    a morning sun (left) slants them down to the right."""
    return clamp((_sun_x(hour) - W / 2) / (PAINTED_SUN_X - W / 2), -1.2, 1.2)


# ----------------------------------------------------------------------------- sky (shared)
_SKY_GRID = {}


def _grid(hh):
    g = _SKY_GRID.get(hh)
    if g is None:
        g = _SKY_GRID[hh] = np.mgrid[0:hh, 0:W]
    return g


@lru_cache(maxsize=None)
def _rn_tex():
    return sc._tnoise(sc._deck('mid').shape[0], 512, 2.0, 91)


def _sky(c, f, K, w, cam_x, gap_x, y_off=0, horizon=100, tilt=0):
    """sc.draw_sky with the decks' and scud's wind speeds integrated (phase K = integral of the wind)."""
    y0 = -tilt
    c.gradient(0, y0, W, horizon - y0, sc.SKY, soft=0.7)
    if horizon < H:
        c.rect(0, horizon, W, H - horizon, sc.SKY[-1])
    gy = 30 + y_off
    fx.glow(c, gap_x, gy, 34, '#fff3cf', strength=0.22, rings=5)
    fx.glow(c, gap_x, gy - 2, 16, '#fffbe8', strength=0.28, rings=4)
    _paste_wrap(c, 'deck_far', f * 0.35 + cam_x * 0.15, horizon - 34 + y_off // 2)
    ms = sc._wrap_slice(sc._deck('mid'), int(0.7 * f + 0.25 * K + cam_x * 0.25), W).copy()
    my = 12 + y_off
    hh = ms.shape[0]
    yy, xx = _grid(hh)
    rn = _rn_tex()[:, (np.arange(W) + int(f * 0.4)) % 512]
    dx = (xx - gap_x) / 20.0
    dy = (yy + my - gy) / 10.0
    d = dx * dx / (1 + np.maximum(dy, 0) * 0.9) + np.minimum(dy, 0) ** 2 + (rn - 0.5) * 0.9
    deck = ms[..., 3] > 0
    bt = bayer(hh, W)
    ms[(d >= 1.0) & (d < 1.15) & deck & (bt < 0.3), :3] = col(sc.DECK_MID[3])
    ms[(d >= 1.0) & (d < 1.6) & deck & (bt < 0.5), :3] = col(sc.DECK_MID[2])
    _put(c, ms[..., :3], (ms[..., 3] >= 128) & (d >= 1.0), 0, my)
    ns = sc._wrap_slice(sc._deck('near'), int(1.3 * f + 0.8 * K + cam_x * 0.35), W)
    ny = -14 + y_off
    hh = ns.shape[0]
    yy, xx = _grid(hh)
    dy = (yy + ny - gy + 6) / 14.0
    d = ((xx - gap_x) / 30.0) ** 2 / (1 + np.maximum(dy, 0)) + np.minimum(dy, 0) ** 2
    _put(c, ns[..., :3], (ns[..., 3] >= 128) & (d >= 1.0), 0, ny)
    fx.glow(c, gap_x, gy - 4, 46, '#fff6dc', strength=0.10, rings=4)
    for i in range(9):                                  # scud racing under the decks
        sx = int((hash01(i, 1) * 600 + f * (2.2 + hash01(i, 5) * 1.5) + K - cam_x * 0.4) % 600) - 140
        sy = int(36 + hash01(i, 2) * 40) + y_off
        L = 8 + int(hash01(i, 3) * 16)
        c.hline(sx, sx + L, sy, sc.DECK_NEAR[1])
        c.hline(sx + 3, sx + L - 4, sy - 1, sc.DECK_NEAR[2])
        c.hline(sx + 5, sx + L + 3, sy + 1, mix(sc.DECK_NEAR[1], sc.SKY[3], 0.5))


def _sky_alone(gap_x, y_off=0, horizon=100, tilt=0):
    return Canvas(arr=_sky_alone_a(gap_x, y_off, horizon, tilt).copy())


@lru_cache(maxsize=64)
def _sky_alone_a(gap_x, y_off, horizon, tilt):
    """The overcast backdrop of _sky alone (gradient + the light gap's glows, no cloud decks or scud):
    the area's sky hook (game/daylight.py repaints it for the world clock's hour)."""
    c = Canvas()
    y0 = -tilt
    c.gradient(0, y0, W, horizon - y0, sc.SKY, soft=0.7)
    if horizon < H:
        c.rect(0, horizon, W, H - horizon, sc.SKY[-1])
    gy = 30 + y_off
    fx.glow(c, gap_x, gy, 34, '#fff3cf', strength=0.22, rings=5)
    fx.glow(c, gap_x, gy - 2, 16, '#fffbe8', strength=0.28, rings=4)
    fx.glow(c, gap_x, gy - 4, 46, '#fff6dc', strength=0.10, rings=4)
    return c.a


_MAGENTA = np.array((255, 0, 255), np.uint8)


@lru_cache(maxsize=1)
def _bayer_full():
    return bayer(H, W)


def _repaint(hour):
    """0..1: how much of the overcast sky the clock repaints (none in the grey daytime 08:30-16:00, all of
    it from 17:00 to 07:30, ramped in between)."""
    h = hour % 24
    if 8.5 <= h <= 16.0:
        return 0.0
    if 7.5 < h < 8.5:
        return 8.5 - h
    if 16.0 < h < 17.0:
        return h - 16.0
    return 1.0


def _block(c, pastes, y_from=None):
    """Stamp scenery layers out of a sky hook (magenta: never sky), each placed as _paste places it:
    the grey rock is too close to the grey backdrop for the colour test alone."""
    hit = np.zeros((H, W), bool)
    for name, x, y in pastes:
        m = _layer(name)[1]
        x, y = int(x), int(y)
        h, w = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            hit[y0:y1, x0:x1] |= m[y0 - y:y1 - y, x0 - x:x1 - x]
    if y_from is not None:
        hit[max(0, y_from):] = True
    np.copyto(c.a, _MAGENTA, where=hit[..., None])
    return c


def _flag(c, f, K, x, y, k, kmul=1.0, **kw):
    """sc._flag with an integrated ripple phase: its sin(f * (0.5 + 0.3 k)) becomes 0.5 f + 0.3 K."""
    km = k * kmul
    fe = (0.5 * f + 0.3 * kmul * K) / (0.5 + 0.3 * km)
    sc._flag(c, fe, x, y, k=km, **kw)


def _mist(c, f, phase, y_scr, hh, xoff, x0, x1, amount, seed, color='#b7bdb9'):
    """sc._mist with an integrated drift (phase = its f * speed)."""
    sc._mist(c, 1, y_scr, hh, xoff, x0, x1, phase, amount, seed, color=color)


# ----------------------------------------------------------------------------- area 0: Trottingham
ISLE_CAM = 140          # past the isle (ends at 404) over open sea; the rocks (520, p 1.6) are tiled
ROCK_TW = 520           # the rock strip's ends are the same height: it tiles
SHIP_SPEED = 0.5        # the film's 0.5 px / frame (ship layer p 1.15)
SHIP_X0, SHIP_X1 = -70, 650     # ship layer x where a run starts / ends: off screen at any camera
SHIP_PERIOD = int((SHIP_X1 - SHIP_X0) / SHIP_SPEED)
SHIP_LEAD = int((132 - SHIP_X0) / SHIP_SPEED)   # on landing the ship is where the film shows it

LANDMARKS_ISLE = [      # the film's callout (isle layer = world)
    Landmark('TROTTINGHAM', 214, (229, 73, 'right', 10)),
]
_isle_meta = sc._isle_layer()
BEACON = _isle_meta[4]                              # harbour beacon lamp (world px)
CHURCH = (256, int(sc._isle_ground(262)) + 2 - 28)  # top of the church tower (belfry below)
HOTSPOTS_ISLE = [
    Hotspot('beacon', BEACON[0], BEACON[1] - 2, 'lantern', cooldown=90),
    Hotspot('bell', CHURCH[0], CHURCH[1] - 6, 'bell', cooldown=60),
]
STAMPS_ISLE = [Stamp('strait', 436, 94)]           # far out over the strait, past the isle

# ----------------------------------------------------------------------------- area 1: Griffonstone
GS_CAM_X = (0, 280)     # main layer 600 x 440 (p 1): x <= 280, y <= 260; mid 480 x 340 (p .5) is wider
GS_CAM_Y = (0, 260)
LOOSE = int(60 * 0.56)  # the bridge's loose plank (sc._bridge)
TRAIN_PERIOD = 330      # whistle, then the cylinder hiss twice (film 150, 162, 198)
TRAIN_LEAD = 60
_bx0, _bx1 = sc.BR_A[0], sc.BR_B[0]

_FALL_I, _FALL_J = (a.ravel() for a in np.mgrid[0:12, 0:60])      # the river's fall into the abyss
_FALL_X = (_FALL_I * 0.9).astype(int) + _FALL_J // 14
_FALL_COLS = np.array([col('#e3e9e2'), col('#aebcb5'), col('#7d918b')], np.uint8)
TILT_A, TILT_S = 105, 1.3  # camera tilt = (balloon y - TILT_A) * TILT_S: looks up at the crown, down at the gorge
TINT_OVERCAST = ('#a6adad', 0.12)    # flat grey daylight on the balloon (darker down in the chasm)
TOP_FG_CY = 30          # the bridge-level foreground (crag + pines) stops rising here (see render_spire)

_gs_meta = sc._gs_world()[1]
FUNNEL, CYL = _gs_meta['funnel'], _gs_meta['cyl']
DOME = (FUNNEL[0] - 6, FUNNEL[1] + 9)              # the engine's steam dome (its whistle)
TERRACE = (140, 44)                                # the empty throne terrace on the crown
STRIKE = (530, 106)                                # the pines on the far pinnacle
STORM_IN, STORM_OUT, STORM_PEAK = 5, 22, 0.20       # the strike's light: one soft wash (ramped in, eased out)


def storm_light(a):
    """Blend weight of the lightning's white-blue wash a steps after the strike: a smoothstep up over
    STORM_IN steps to STORM_PEAK, then a smoothstep down over STORM_OUT (one soft flash; the film's
    0.38 / 0.22 / 0.12 double flicker strobed, which is hard on flash-sensitive players)."""
    if a < 0 or a >= STORM_IN + STORM_OUT:
        return 0.0
    u = (a + 1) / (STORM_IN + 1) if a < STORM_IN else 1 - (a - STORM_IN) / STORM_OUT
    return round(STORM_PEAK * u * u * (3 - 2 * u), 3)
HOTSPOTS_GS = [
    Hotspot('gust', TERRACE[0], TERRACE[1], 'wind', cooldown=150),
    Hotspot('train', DOME[0] - 6, DOME[1] + 5, 'whistle', cooldown=90),
    Hotspot('storm', STRIKE[0], STRIKE[1], 'bolt', cooldown=90),
]
STAMPS_GS = [Stamp('crown', 216, 18),             # high above the castle's broken tower
             Stamp('abyss', 306, 372)]            # deep in the chasm, by the river's fall

LANDMARKS_GS = [        # the film's callouts (main layer = world); y bands keep each tag on screen
    Landmark('GRIFFONSTONE', 176, (196, 44, 'right', 14), y=45),
    Landmark('GRIFFONSTONE STATION', 170, (246, 375, 'up', 10), y=349),
    Landmark('ABYSMAL ABYSS', 392, (386, 182, 'right', 12), y=180),
]


class Griffonstone(GlowKeep, Region):
    SEG = 'griffonstone'
    CARD = ('GRIFFON KINGDOM', ['CITY ON A ROCK SPIRE', 'ACROSS THE SEA'])

    def build_areas(self):
        return [
            Area(self.render_isle, cam_x=(0, ISLE_CAM), cam_y=(0, 0), y_range=(22, 108),
                 start=(118, 62), landmarks=LANDMARKS_ISLE, balloon='s', balloon_tint=TINT_OVERCAST,
                 hotspots=HOTSPOTS_ISLE, stamps=STAMPS_ISLE, sky=self.sky_isle),
            Area(self.render_spire, cam_x=GS_CAM_X, cam_y=GS_CAM_Y, y_range=(28, 392),
                 start=(60, 360), landmarks=LANDMARKS_GS, balloon='m', balloon_tint=TINT_OVERCAST,
                 hotspots=HOTSPOTS_GS, stamps=STAMPS_GS, sky=self.sky_spire),
        ]

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        self.area_t0 = 0            # step the current area was entered
        self.ship_t0 = 0            # ship clock origin
        self.plank_f = None         # scene frame the loose plank snapped (None: still in the bridge)
        self.gust_t0 = 0            # step the gust schedule counts from (first gust)
        self.gust_acc = 0.0         # wind phase banked when the player calls a gust early

    def on_enter(self):
        self.ship_t0 = self.g.t - SHIP_LEAD
        self.plank_f = None         # a new visit: the bridge has been mended

    def on_area(self, ai):
        self.area_t0 = self.g.t
        self.gust_t0 = self.g.t + GUST_LEAD
        self.gust_acc = 0.0

    def on_interact(self, key, t):
        g = self.g
        if key == 'beacon':
            g.sfx('horn', 0.45)                         # fog horn, the lamp flares
            g.sfx('shimmer', 0.3)
        elif key == 'bell':
            g.sfx('bell', 0.6)
        elif key == 'gust':                             # call the gust now (update() blows it)
            d = t - self.gust_t0
            if d >= 0:
                n, s = divmod(d, GUST_PERIOD)
                self.gust_acc += n * _GUST_INT[-1] + _GUST_INT[s]
            self.gust_t0 = t
            g.sfx('whoosh', 0.5)
        elif key == 'train':
            g.sfx('whistle', 0.85)
            g.sfx('steam', 0.6)
        elif key == 'storm':
            g.sfx('zap', 0.55)
            g.sfx('thunder', 0.7, 1)

    def spot_a(self, key, t):
        """Steps since hotspot `key` was set off (a large number if never)."""
        return t - self.spot_t.get(key, -10 ** 6)

    def _cam_target(self):
        """In Griffonstone the tilt runs ahead of the balloon (x1.3): near the crown the view looks up at
        the castle (balloon low on screen), near the gorge floor it looks down on the station (balloon
        high on screen), so each end of the 440 px spire is in view from a comfortable height."""
        tx, ty = super()._cam_target()
        if self.ai == 1:
            ty = clamp((self.by - TILT_A) * TILT_S + self.look_y, *self.area.cam_y)
        return tx, ty

    def _switch(self, ai, from_right, dir):
        """Keep the balloon's relative height across the cloud sweep: skim the waves at Trottingham and
        you come out at the gorge floor by the station, fly high and you come out at the castle."""
        y0, y1 = self.area.y_range
        k = clamp((self.by - y0) / (y1 - y0))
        super()._switch(ai, from_right, dir)
        y0, y1 = self.area.y_range
        self.by = y0 + k * (y1 - y0)
        self.cx, self.cy = self._cam_target()
        self.prev = self.snapshot()

    def gust_s(self, t):
        """Steps since the last gust started (None before the first one)."""
        if self.ai != 1:
            return None
        d = t - self.gust_t0
        return None if d < 0 else d % GUST_PERIOD

    def wind(self, t, f):
        """(strength now, integrated phase) for this area."""
        w, K = _base_w(f), _base_int(f)
        if self.ai == 1:
            d = t - self.gust_t0
            if d >= 0:
                n, s = divmod(d, GUST_PERIOD)
                w += _gust_w(s)
                K += n * _GUST_INT[-1] + _GUST_INT[s]
            K += self.gust_acc
        return w, K

    def ship_x(self, t):
        return SHIP_X0 + SHIP_SPEED * ((t - self.ship_t0) % SHIP_PERIOD)

    def train_c(self, t):
        return (t - self.area_t0 + TRAIN_PERIOD - TRAIN_LEAD) % TRAIN_PERIOD

    def bridge_on_screen(self):
        return (self.cx - 20 < _bx1 and self.cx + W + 20 > _bx0 and
                self.cy - 30 < sc.BR_A[1] + 20 and self.cy + H > sc.BR_A[1])

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        f = self.scene_f(t)
        if self.ai == 0:
            if (f - 81) % 72 == 0:                      # spray bursting on the rocks (the film's 81, 117 ...)
                g.sfx('waves', 0.3)
            if (t - self.area_t0) % 150 == 40:
                g.sfx('wind', 0.22, 0)
            sx = self.ship_x(t) - 1.15 * self.cx        # ship timbers, while the ship is in view
            if (t - self.area_t0) % 120 == 60 and -40 < sx < W + 40:
                g.sfx('creak', 0.25)
            return
        deep = clamp((self.by - 200) / 180) if self.bx > sc.CLIFF_X else 0.0
        self.area.balloon_tint = (mix(TINT_OVERCAST[0], '#454850', deep), TINT_OVERCAST[1] + 0.22 * deep)
        s = self.gust_s(t)
        if s == 0:
            g.sfx('wind', 0.75, 1)
            self.vx += 1.2                      # the gust shoves the balloon
        elif s == 6 and self.bridge_on_screen():
            g.sfx('creak', 0.6)
        elif s == SNAP_S and self.plank_f is None and self.bridge_on_screen():
            self.plank_f = f
            g.sfx('creak', 0.75)
        elif s == 46:
            g.sfx('thunder', 0.25, 0)
        elif s == 300:
            g.sfx('wind', 0.35, 0)
        tc = self.train_c(t)
        gain = self.near_gain(210) * (1.0 if self.cy > 140 else 0.4)
        if tc == 0 and ((t - self.area_t0) // TRAIN_PERIOD) % 2 == 0:     # a whistle every other cycle
            g.sfx('whistle', 0.55 * max(0.3, gain))
        elif tc in (12, 48):
            g.sfx('steam', 0.5 * gain if tc == 12 else 0.4 * gain)

    # ------------------------------------------------------------------ the world clock's lights
    def auto_glow(self):
        """The relight keeps no painted spot glowing here (gold flag tips and the train's trim are paint):
        the windows, lanterns, beacon and the events' light are kept by mask (_keep_lights)."""
        return False

    def lamp_k(self):
        """0 (day) .. 1 (night): the dusk ramp the windows and lanterns go on by (nightwin.ramp)."""
        return nightwin.ramp(self.g.hour)

    def _keep_lights(self, c, lit, exact=()):
        """End of an area render: the lit pixels still showing warm light (a window under the fog banks
        too, never smoke, a sail or a pine drawn over it) and the light effects' pixels nothing covered
        since (exact: (ys, xs, rgb)) keep their colour through the night's grade (GlowKeep)."""
        if not lit and not exact:
            return
        m = np.zeros((H, W), bool)
        if lit:
            ys, xs = np.concatenate([a for a, _ in lit]), np.concatenate([b for _, b in lit])
            px = c.a[ys, xs].astype(np.int16)
            ok = (px[:, 0] > 120) & (px[:, 0] > px[:, 2] + 40)
            m[ys[ok], xs[ok]] = True
        for ys, xs, rgb in exact:
            ok = np.all(c.a[ys, xs] == rgb, axis=1)
            m[ys[ok], xs[ok]] = True
        self.keep_glow(c, m, 0.95)

    def _gap_isle(self, f, cam):
        """The overcast's bright gap over Trottingham (screen x): the film's drift, over the clock's sun."""
        return int(168 + 16 * math.sin(f / 330.0) - cam * 0.25) + gap_shift(self.g.hour)

    def _gap_spire(self, f, cam_x):
        return int(209 - cam_x * 0.25 + 8 * math.sin(f / 300.0)) + gap_shift(self.g.hour)

    @staticmethod
    def _beacon(c, f, x, y, lit):
        """The harbour beacon by night: a steady lamp, a flash every three beats (a cross of light swelling
        and shrinking) and its broken reflection on the sea; its pixels go into `lit`."""
        x0, y0, x1, y1 = max(0, x - 10), max(0, y - 8), min(W, x + 11), min(H, sc.SEA_Y + 18)
        if x1 <= x0 or y1 <= y0:
            return
        pre = c.a[y0:y1, x0:x1].copy()
        ph = f % 54
        flash = ph < 8
        c.rect(x - 1, y - 1, 3, 2, '#fff1c4' if flash else '#ffd77a')
        c.px(x, y - 1, 'white' if flash else '#fff1c4')
        if flash:
            L = 2 + 2 * min(ph, 7 - ph)
            c.hline(x - L, x + L, y - 1, '#ffe9a0')
            c.vline(x, y - 1 - L // 2, y + L // 2, '#ffe9a0')
            c.px(x, y - 1, 'white')
        for j in range(7):                               # its light broken on the waves below
            if (j + f // 6) % 3 == 2:
                continue
            yy = sc.SEA_Y + 2 + j * 2
            wob = int(round(math.sin(f * 0.3 + j * 1.9)))
            ww = max(1, 3 - j // 2) + (2 if flash and j < 3 else 0)
            c.hline(x - ww // 2 + wob, x + (ww - 1) // 2 + wob, yy, '#ffd77a' if j < 3 else '#d9a050')
        ys, xs = np.nonzero(np.any(c.a[y0:y1, x0:x1] != pre, axis=2))
        if len(ys):
            lit.append((ys + y0, xs + x0))

    def _night_band(self, c, gap_x, cy, y_off=0, horizon=100, tilt=0):
        """The grey bank under the far deck by night. The sky hook keeps it as scenery (no fiery blobs of
        the clock's dusk in its gaps), so the night's grade would leave its backdrop a flat band lighter
        than the night sky above, with a straight top edge. As the night deepens its backdrop (right after
        _sky, before anything is drawn over it) takes the clock's sky, painted in so that the relight
        shows it (daylight.unlight)."""
        hour = self.g.hour
        q = clamp((daylight.night(hour) - 0.3) / 0.5) * _repaint(hour)
        y0 = max(0, horizon - 28 + y_off // 2)
        if q < 0.02 or y0 >= H:
            return
        bd = _sky_alone_a(gap_x, y_off, horizon, tilt)[y0:]
        sub = c.a[y0:]
        m = np.all(sub == bd, axis=2)
        if not m.any():
            return
        top, span = self.area.cam_y[0] - 10, self.area.cam_y[1] - self.area.cam_y[0] + 130    # (Region._world's
        strip = _unlit_strip(daylight.sky_q(hour), top, span, daylight._q(self.HOUR), daylight._q(hour))
        tgt = strip[np.clip(np.arange(y0, H) + int(cy) - top, 0, span - 1)][m].astype(np.float32)
        cur = sub[m].astype(np.float32)
        sub[m] = (cur + (tgt - cur) * q).astype(np.uint8)

    # ------------------------------------------------------------------ the skies alone
    def sky_isle(self, f, cx, cy):
        cam = int(cx)
        gap_x = self._gap_isle(f, cam)
        c = _sky_alone(gap_x, horizon=sc.HORIZON)
        self._overcast(c, f, cam, gap_x, horizon=sc.HORIZON)
        return _block(c, [('far_isles', -int(cam * 0.3), sc._far_layers()[1]), ('isle', -cam, sc.ISLE_TOP)],
                      y_from=sc.HORIZON)

    def sky_spire(self, f, cx, cy):
        cam_x, cam_y = int(cx), int(cy)
        yoff = -int(cam_y * 0.12)
        gap_x = self._gap_spire(f, cam_x)
        kw = dict(y_off=yoff + 4, horizon=170 - int(cam_y * 0.25), tilt=int(cam_y * 0.1))
        c = _sky_alone(gap_x, **kw)
        self._overcast(c, f, cam_x, gap_x, **kw)
        k = storm_light(self.spot_a('storm', self.g.t))  # the render's lightning wash, so the sky mask
        if k > 0.003:                                    # holds through it (no snap to the painted sky)
            m = np.all(c.a == _MAGENTA, axis=2)
            c.blend(0, 0, W, H, '#e6ebff', k)
            c.a[m] = _MAGENTA
        _ab, aX0, aY0 = sc._gs_abyss()
        return _block(c, [('gs_far', -int(cam_x * 0.25), -int(cam_y * 0.25)),
                          ('gs_mid', -int(cam_x * 0.5), -int(cam_y * 0.5)),
                          ('abyss', int(aX0 - cam_x * 0.85), int(aY0 - cam_y * 0.85)),
                          ('world', -cam_x, -cam_y)])

    def _overcast(self, c, f, cam_x, gap_x, **kw):
        """Griffonstone's weather is a grey overcast: the cloud decks and scud are scenery (re-lit, never
        repainted), and in the grey daytime the painted backdrop stays too (the clock's clear sky only
        shows at dawn, dusk and night, dithered in around the change)."""
        p = _repaint(self.g.hour)
        if p < 1.0:
            c.a[_bayer_full() >= p] = _MAGENTA
        if p > 0.0:
            key = (self.ai, self.g.t, f, cam_x, gap_x, tuple(sorted(kw.items())))
            memo = getattr(self, '_clouds', None)
            if memo is not None and memo[0] == key:
                m = memo[1]
            else:                                        # (the render normally noted them just before)
                wk, K = self.wind(self.g.t, f)
                full = Canvas()
                _sky(full, f, K, wk, cam_x, gap_x, **kw)
                m = np.any(full.a != _sky_alone_a(gap_x, kw.get('y_off', 0), kw['horizon'], kw.get('tilt', 0)),
                           axis=2)
            c.a[m] = _MAGENTA
            # under the far deck's base the horizon shows only in ragged gaps (fiery blobs once the
            # clock's dusk is painted in them): the band down to the horizon stays a grey cloud bank
            c.a[max(0, kw['horizon'] - 28 + kw.get('y_off', 0) // 2):] = _MAGENTA

    def _note_clouds(self, c, f, cam_x, gap_x, **kw):
        """Remember which pixels of the sky _sky just drew are cloud decks and scud (for the sky hook,
        which the world draw calls right after the render): a cache, the render's output is unchanged."""
        if _repaint(self.g.hour) <= 0.0:
            return
        key = (self.ai, self.g.t, f, cam_x, gap_x, tuple(sorted(kw.items())))
        self._clouds = (key, np.any(c.a != _sky_alone_a(gap_x, kw.get('y_off', 0), kw['horizon'],
                                                         kw.get('tilt', 0)), axis=2))

    # ------------------------------------------------------------------ area 0: Trottingham
    def render_isle(self, f, cx, cy):
        t = self.g.t
        cam = int(cx)
        wk, K = self.wind(t, f)
        c = Canvas()
        gap_x = self._gap_isle(f, cam)
        sl = slant_k(self.g.hour)
        lk = self.lamp_k()
        lit, exact = [], []
        _sky(c, f, K, wk, cam, gap_x, horizon=sc.HORIZON)
        self._note_clouds(c, f, cam, gap_x, horizon=sc.HORIZON)
        self._night_band(c, gap_x, cy, horizon=sc.HORIZON)
        _paste(c, 'far_isles', -int(cam * 0.3), sc._far_layers()[1])
        sc._draw_sea(c, f, cam)
        bx = gap_x + int((sc.SEA_Y - 30) * -0.35 * sl)
        sun = 1.0 - _repaint(self.g.hour)                # the shaft only shows in the grey daytime
        for i in range(18):                              # sun glint under the shaft
            gx = bx - 20 + int(hash01(i, 1) * 44)
            gy = sc.HORIZON + 2 + int(hash01(i, 2) * 22)
            if hash01(i, f // 4) < 0.5 * sun:
                c.hline(gx, gx + 1 + int(hash01(i, 3) * 3), gy, '#a9b8ae')
        isle, chims, wins, flag, beacon = sc._isle_layer()
        ix = -cam
        _paste(c, 'isle', ix, sc.ISLE_TOP)
        _put_windows(c, 'isle', ix, sc.ISLE_TOP, lk, lit)  # the clock's windows (dark by day)
        for j in range(6):                               # broken reflection under the waterline
            y = sc.SEA_Y + 2 + j
            src = isle[sc.SEA_Y - sc.ISLE_TOP - 2 - j * 2]
            off = int(round(math.sin(f * 0.3 + j * 1.7) * 1.5))
            xs_ = np.nonzero(src[:, 3] > 0)[0] + ix + off
            ok = (xs_ >= 0) & (xs_ < W) & ((xs_ + j + f // 2) % 3 != 0)
            if np.any(ok):
                row = c.a[y, xs_[ok]].astype(np.int16)
                c.a[y, xs_[ok]] = np.clip(row * 0.82 + np.array([8, 14, 14]), 0, 255).astype(np.uint8)
        for i, (wx, wy) in enumerate(wins):              # lit windows flicker
            if not _win_on('isle', wx, wy - sc.ISLE_TOP, lk):
                continue
            if hash01(i, f // 6) < 0.12:
                c.px(wx + ix, wy, sc.WARM[0])
            else:
                c.px(wx + ix, wy, sc.WARM[1] if hash01(i, 9) < 0.5 else sc.WARM[0])
        bxx, byy = beacon[0] + ix, beacon[1]             # the beacon: dark glass by day, lit by night
        if lk > 0.05:
            self._beacon(c, f, bxx, byy, lit)
        else:
            c.rect(bxx - 1, byy - 1, 3, 2, LAMP_OFF)
        for i, (rbx, rc) in enumerate(((34, '#7a3b30'), (44, '#3f5a6a'), (60, '#6b4a2e'))):   # rowboats
            yy = sc.SEA_Y + 3 + int(round(math.sin(f / 8.0 + i * 2) * 1))
            xx = rbx + ix + int(round(math.sin(f / 13.0 + i) * 1))
            c.hline(xx - 3, xx + 3, yy, rc)
            c.hline(xx - 2, xx + 2, yy + 1, sc.shade(rc, 0.7))
            c.px(xx - 3, yy - 1, rc)
            c.px(xx + 3, yy - 1, rc)
            c.hline(xx - 4, xx + 4, yy + 2, sc.SEA[1])
        for i, (sx_, sy_) in enumerate(chims):           # chimney smoke streaming right
            sc._smoke(c, f, sx_ + ix, sy_, 100 + i, n=6, k=wk)
        _flag(c, f, K, flag[0] + ix, flag[1] - 8, wk, L=10, hgt=5)
        c.vline(flag[0] + ix, flag[1] - 9, flag[1], '#3a3f46')
        c.px(flag[0] + ix, flag[1] - 10, '#e0b04a')
        fs = sc._wrap_slice(sc._fog_tex(), int(0.5 * f + 0.4 * K + cam * 0.9), W)
        sc._fog_blend(c, fs, 84, '#b9c0bc', 0.6)          # fog banks over the isle
        if sun > 0.02:
            _sun_shaft(c, gap_x, 30, sc.SEA_Y + 4, f, strength=0.13 * sun, slant=-0.35 * sl)
        pre = None
        if daylight.night(self.g.hour) > 0.02 and (self.spot_a('beacon', t) < 84 or self.spot_a('bell', t) < 54):
            pre = c.a.copy()                             # (the beacon's sweep and the bell's rings: light)
        self._isle_events(c, t, cam)
        if pre is not None:
            ys, xs = np.nonzero(np.any(c.a != pre, axis=2))
            exact.append((ys, xs, c.a[ys, xs].copy()))
        sx = int(self.ship_x(t)) - int(cam * 1.15)        # the two-master sails the strait
        if -60 < sx < W + 60:
            sc._draw_ship(c, f, sx, 140)
        sc._whitecaps(c, f, cam)
        rocks, tufts = sc._fg_rocks()                     # foreground rocks (p 1.6), tiled
        fx0 = -int(cam * 1.6) - 60
        for k in range(2):
            ox = fx0 + k * ROCK_TW
            if ox >= W or ox + ROCK_TW <= 0:
                continue
            _paste(c, 'rocks', ox, H - 50)
            for i, (tx, ty) in enumerate(tufts):
                X = tx + ox
                if -4 <= X < W + 4:
                    sc._grass_tuft(c, X, H - 50 + ty, f, wk, i)
            t0 = f - (f - 81) % 36                       # spray bursts on the beat (film 81 + 2 beats)
            sc._spray(c, f, 302 + ox, H - 26, t0, (t0 // 36) % 4)
            t1 = f - (f - 99) % 36
            sc._spray(c, f, 116 + ox, H - 24, t1, (t1 // 36) % 4 + 10)
            for i in range(10):                          # foam where the sea meets the rocks
                ph = (f + i * 5) % 20
                for (ex, ey) in ((300, H - 22), (118, H - 20)):
                    xx = ex + ox + (i - 5) + (1 if ph < 10 else 0) * (-1 if ex > 200 else 1)
                    c.px(xx, ey + (i % 3), sc.FOAM if ph < 14 else sc.SEA_HI)
        self._keep_lights(c, lit, exact)
        return c

    # ------------------------------------------------------------------ area 1: Griffonstone
    def render_spire(self, f, cx, cy):
        t = self.g.t
        cam_x, cam_y = int(cx), int(cy)
        wk, K = self.wind(t, f)
        s = self.gust_s(t)
        c = Canvas()
        yoff = -int(cam_y * 0.12)
        gap_x = self._gap_spire(f, cam_x)
        lk = self.lamp_k()
        lit, exact = [], []
        _sky(c, f, K, wk, cam_x, gap_x, y_off=yoff + 4, horizon=170 - int(cam_y * 0.25), tilt=int(cam_y * 0.1))
        self._note_clouds(c, f, cam_x, gap_x, y_off=yoff + 4, horizon=170 - int(cam_y * 0.25),
                          tilt=int(cam_y * 0.1))
        self._night_band(c, gap_x, cy, y_off=yoff + 4, horizon=170 - int(cam_y * 0.25), tilt=int(cam_y * 0.1))
        _paste(c, 'gs_far', -int(cam_x * 0.25), -int(cam_y * 0.25))
        _mist(c, f, 0.5 * f + 0.3 * K, 128 - int(cam_y * 0.3), 30, cam_x * 0.3, -40, W + 40, 0.45, 41)
        _paste(c, 'gs_mid', -int(cam_x * 0.5), -int(cam_y * 0.5))
        _ab, aX0, aY0 = sc._gs_abyss()
        _paste(c, 'abyss', int(aX0 - cam_x * 0.85), int(aY0 - cam_y * 0.85))
        a0, a1 = 150 - cam_x * 0.85, 600 - cam_x * 0.85
        _mist(c, f, 0.5 * f + 0.4 * K, 300 - int(cam_y * 0.85), 56, cam_x * 0.85, a0, a1, 0.5, 45, color='#8f9696')
        _mist(c, f, 0.4 * f + 0.3 * K, 372 - int(cam_y * 0.85), 50, cam_x * 0.85, a0, a1, 0.45, 46, color='#878e8e')
        _mist(c, f, 0.7 * f + 0.5 * K, 172 - int(cam_y * 0.85), 60, cam_x * 0.85, a0, 560 - cam_x * 0.85, 0.6, 42,
              color='#a9b0ae')
        meta = sc._gs_world()[1]
        _paste(c, 'world', -cam_x, -cam_y)
        _put_windows(c, 'world', -cam_x, -cam_y, lk, lit)   # the clock's windows (dark by day)
        ox, oy = cam_x, cam_y
        ry = sc.RIVER_Y - oy                             # river + waterfall into the abyss
        if ry < H:
            tex = sc._river_tex()
            sl = sc._wrap_slice(tex, int(-f * 1.4), W)
            x1 = min(W, sc.CLIFF_X - 2 - ox)
            if x1 > 0:
                c.a[ry:min(H, ry + tex.shape[0]), 0:x1] = sl[:min(H, ry + tex.shape[0]) - ry, :x1]
                c.hline(0, x1 - 1, ry, '#8fa196')
            wx = sc.CLIFF_X - 3 - ox
            ys, xs = _FALL_J + ry, _FALL_X + wx
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            if ok.any():
                ph = (_FALL_J * 3 - f * 7 + _FALL_I * 13) % 23
                c.a[ys[ok], xs[ok]] = _FALL_COLS[(ph >= 5).astype(int) + (ph >= 13)][ok]
            for i in range(8):
                ph = (f * 2 + i * 7) % 16
                c.px(wx + i - 2, ry - 1 + (ph // 8), 'white' if ph < 8 else sc.FOAM)
        sun = 1.0 - _repaint(self.g.hour)                # the shaft only shows in the grey daytime
        if sun > 0.02:
            _sun_shaft(c, gap_x, 30 + yoff, H, f, strength=0.12 * sun, slant=-0.8 * slant_k(self.g.hour))
        for i, (wx_, wy_) in enumerate(meta['wins']):
            X, Y = wx_ - ox, wy_ - oy
            if 0 <= X < W and 0 <= Y < H and hash01(i, f // 5) < 0.14 and _win_on('world', wx_, wy_, lk):
                c.px(X, Y, sc.WARM[0])
        for i, (sx_, sy_) in enumerate(meta['chims']):
            X, Y = sx_ - ox, sy_ - oy
            if -40 < X < W and -10 < Y < H + 10:
                sc._smoke(c, f, X, Y, 200 + i, n=6, k=wk)
        for i, (lx, ly, L) in enumerate(meta['ladders']):
            X, Y = lx - ox, ly - oy
            if -10 < X < W and -L < Y < H:
                sc._ladder(c, X, Y, L, f, wk, i)
        for i, (lx, ly) in enumerate(meta['lanterns']):
            X, Y = lx - ox, ly - oy
            if -6 < X < W + 6 and -6 < Y < H:
                _lantern(c, X, Y, f, wk, i, hash01(i, 614) * 0.5 < lk, lit)
        for i, (px_, py_) in enumerate(meta['poles']):
            X, Y = px_ - ox, py_ - oy
            if -20 < X < W and -10 < Y < H:
                _flag(c, f + i * 7, K, X, Y + 1, wk, kmul=1.2, L=10, hgt=7, colors=('#6e3a3a', '#b89a4a'),
                      tatter=True)
        fxw, fyw = meta['funnel']                        # the Friendship Express steaming
        X, Y = fxw - ox, fyw - oy
        if -20 < X < W and -20 < Y < H + 20:
            sc._puffs(c, f, X, Y, 5, wk)
            cx_, cy_ = meta['cyl']
            tc = self.train_c(t)
            for t0 in (12, 48):
                dt = tc - t0
                if 0 <= dt < 14:
                    for j in range(6):
                        c.px(cx_ - ox - dt // 2 - j, cy_ - oy - (j % 3), 'white' if dt < 7 else sc.FOAM)
                        c.px(cx_ - ox + 26 + dt // 2 + j, cy_ - oy - (j % 2), 'white' if dt < 7 else sc.FOAM)
        self._bridge(c, f, ox, oy, wk, s)
        _mist(c, f, 0.9 * f + 0.6 * K, 158 - int(cam_y * 0.95), 44, cam_x * 0.95, 120 - ox, 560 - ox, 0.7, 43,
              color='#c3c8c3')
        _mist(c, f, 0.6 * f + 0.4 * K, 380 - int(cam_y * 1.0), 70, cam_x * 1.0, sc.CLIFF_X - ox - 6, 520 - ox,
              0.75, 44, color='#bfc5c0')
        # (the film's balloon is the player's: not drawn)
        pre = c.a.copy() if daylight.night(self.g.hour) > 0.02 and self.spot_a('storm', t) < 64 else None
        self._spire_events(c, t, ox, oy, wk)
        if pre is not None:                              # (the lightning: light)
            ys, xs = np.nonzero(np.any(c.a != pre, axis=2))
            exact.append((ys, xs, c.a[ys, xs].copy()))
        # foreground (p 1.3). The film shows the bridge-level crag and pines only at its last framing
        # (camera y ~30); below that they would rise as a tall dark pillar across the abyss, so they stop
        # rising there and stay along the bottom edge as a frame. The gorge-floor pines are true p 1.3.
        fox = cam_x * 1.3
        for i, (kind, x, base, h) in enumerate(sc.FG):
            foy = 1.3 * (cam_y if base >= 300 else min(cam_y, TOP_FG_CY))
            X, B = int(x - fox), int(base - foy)
            if kind == 'pine':
                if -30 < X < W + 30 and 0 < B < H + h:
                    sc._fg_pine(c, X, B, h, f, wk, i)
            elif -80 < X < W and B < H + 40:
                c.poly([(X, H), (X + 6, B - 6), (X + 30, B - 12), (X + 60, B - 8), (X + 90, B), (X + 90, H)],
                       '#1b2220')
                c.line(X + 6, B - 6, X + 30, B - 12, '#2e3a33')
                c.line(X + 30, B - 12, X + 60, B - 8, '#2e3a33')
        self._debris(c, f, K, wk, s)
        k = storm_light(self.spot_a('storm', t))         # the lightning's soft wash over everything
        if k > 0.003:
            c.blend(0, 0, W, H, '#e6ebff', k)
            exact = [(ys, xs, c.a[ys, xs].copy()) for ys, xs, _ in exact]
        self._keep_lights(c, lit, exact)
        return c

    # ------------------------------------------------------------------ SPACE events (read state only)
    def _isle_events(self, c, t, cam):
        a = self.spot_a('beacon', t)
        if a < 84:                                       # the harbour beacon flares and sweeps the strait
            k = min(1.0, a / 4) * (1 - seg(a, 54, 84))
            bx, by = BEACON[0] - cam, BEACON[1]
            ex = bx + 30 + int(a * 3.2)
            _beam(c, ((bx - 1, by), (bx + 2, by)), ((ex - 14 - a // 4, 176), (ex + 16 + a // 4, 176)), '#fff1c4',
                  0.30 * k)
            _beam(c, ((bx, by), (bx + 1, by)), ((ex - 4, 176), (ex + 6, 176)), '#fffbe0', 0.30 * k)
            fx.glow(c, bx, by, 12 + int(14 * k), '#ffd77a', 0.55 * k, rings=4)
            fx.glow(c, bx, by, 6, 'white', 0.8 * k, rings=2)
            c.rect(bx - 1, by - 1, 3, 2, 'white')
            if a < 16 or (a // 3) % 3:
                L = 3 + int(5 * k)
                c.hline(bx - L, bx + L, by, 'butter')
                c.vline(bx, by - L + 1, by + L - 1, 'butter')
                c.px(bx, by, 'white')
            for i in range(10):                          # glints on the waves under the beam
                gx = ex - 44 + int(hash01(i, a // 3, 11) * 60)
                gy = 128 + int(hash01(i, 12) * 40)
                if hash01(i, a // 2, 13) < 0.6 * k:
                    c.hline(gx, gx + 2, gy, 'white')
        a = self.spot_a('bell', t)
        if a < 54:                                       # the church bell rings out over the town
            bx, by = CHURCH[0] - cam - 2, CHURCH[1] + 9
            for k in range(3):
                r = (a - k * 9) * 1.1 + 3
                if 3 <= r < 36:
                    c.circle(bx, by, int(r), 'butter' if k == 0 else 'white', fill=False)
            if (a // 4) % 2 == 0:
                c.rect(bx - 1, by - 2, 2, 4, '#ffd77a')
            fx.glow(c, bx, by, 9, '#ffd77a', 0.35 * (1 - a / 54), rings=3)
            for j in range(3):
                fx.sparkle(c, bx - 10 + j * 10, by - 12 - (j % 2) * 5, a + j * 4, 'butter', period=18, size=2)

    def _spire_events(self, c, t, ox, oy, wk):
        a = self.spot_a('train', t)
        if a < 90:                                       # the Friendship Express whistles and blows off steam
            wx, wy = DOME[0] - ox, DOME[1] - oy - 2
            if a < 34 and (a < 10 or a >= 14):           # the whistle's jet (two blasts)
                c.vline(wx, wy - 6, wy - 1, 'white')
                c.vline(wx + 1, wy - 9, wy - 4, mix('white', 'cloud', 0.4))
                c.px(wx, wy - 10, 'white')
            for k in range(12):                          # its plume, blown right by the wind
                age = a - k * 3
                if 0 <= age < 44:
                    r = 1.5 + age * 0.13
                    px_ = wx + 1 + int(age * (0.5 + 0.4 * wk))
                    py_ = wy - 8 - int(age * 0.7)
                    cc = 'white' if age < 14 else ('#e9e6dc' if age < 28 else '#c9cac4')
                    if age < 36 or (age + k) % 2:
                        c.circle(px_, py_, r, cc)
            cx_, cy_ = CYL[0] - ox, CYL[1] - oy           # cylinder cocks: steam both ways along the platform
            for side, x0 in ((-1, cx_ - 2), (1, cx_ + 26)):
                for k in range(5):
                    age = a - k * 4
                    if 0 <= age < 40:
                        r = 2 + age * 0.22
                        px_ = x0 + side * int(3 + age * (1.3 - k * 0.12)) + (int(age * 0.3 * wk) if side > 0 else 0)
                        py_ = cy_ - 1 - int(age * 0.25) - k % 2
                        cc = 'white' if age < 12 else ('#e3e6df' if age < 26 else '#b9bdb8')
                        if age < 30 or (age + k) % 2:
                            c.ellipse(px_, py_, r, r * 0.65, cc)
        a = self.spot_a('gust', t)
        if a < 44:                                       # wind ribbons race past, dust and grit torn off the terrace
            tx, ty = TERRACE[0] - ox, TERRACE[1] - oy
            for i in range(9):
                head = -40 + int((a - hash01(i, 41) * 12) * (9 + 4 * hash01(i, 42)))
                y0 = ty - 30 + int(hash01(i, 43) * 70)
                L = 26 + int(hash01(i, 44) * 26)
                for j in range(0, L, 1):
                    x = head - j
                    if 0 <= x < W and (j < L * 0.6 or (x + y0) % 2):
                        y = y0 + int(round(2.5 * math.sin((x + i * 17) * 0.11)))
                        c.px(x, y, 'white' if j < 6 else '#cfd4cf')
            for i in range(26):
                age = a - int(hash01(i, 21) * 10)
                if age < 0:
                    continue
                v = 2.5 + hash01(i, 22) * 3.5
                px_ = tx - 16 + int(hash01(i, 23) * 32) + int(age * v)
                py_ = ty - 2 + int(hash01(i, 24) * 14 - 7) + int(math.sin(age * 0.3 + i) * 3) + age // 6
                cc = ('#a68e6b', '#cfd4cf', '#8a6a44', '#b9bebb')[i % 4]
                c.px(px_, py_, cc)
                if (age + i) % 3:
                    c.px(px_ - 1, py_, cc)
        a = self.spot_a('storm', t)
        if a < 64:                                       # lightning strikes the far pinnacle's pines
            sx, sy = STRIKE[0] - ox, STRIKE[1] - oy
            seed = self.spot_t['storm']
            if a < 10:                                   # one bolt (no flicker): white, then fading blue
                core = 'white' if a < 7 else '#c8d4ff'
                pts = [(sx, sy)]
                n = 9
                for i in range(1, n + 1):
                    yy = sy - (sy + 8) * i / n
                    xx = sx + int((hash01(seed, i, 31) - 0.5) * 22) + int(i * 1.5)
                    pts.append((xx, int(yy)))
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    c.line(x0 + 1, y0, x1 + 1, y1, '#c8d4ff' if a < 7 else '#9aa6d4')
                    c.line(x0, y0, x1, y1, core)
                bi = 3 + int(hash01(seed, 32) * 3)       # one fork
                fxp, fyp = pts[bi]
                for j in range(4):                       # (a short jagged branch, down and left)
                    nx_ = fxp - 2 - int(hash01(seed, j, 33) * 6) + (3 if j % 2 else 0)
                    ny_ = fyp + 4 + int(hash01(seed, j, 34) * 4)
                    c.line(fxp, fyp, nx_, ny_, '#dfe6ff' if a < 7 else '#9aa6d4')
                    fxp, fyp = nx_, ny_
            if a < 56:
                k = min(1.0, (a + 1) / 3) * (1 - a / 56)  # (swells in with the wash)
                fx.glow(c, sx, sy, 10 + int(12 * k), '#dfe6ff', 0.6 * k, rings=4)
                fx.glow(c, sx, sy, 5, 'white', 0.8 * k, rings=2)
                for j in range(5):                       # static crawling over the pinnacle's pines
                    fx.sparkle(c, sx - 18 + j * 9, sy + 2 - (j % 2) * 6, a + j * 5, 'baby_blue', period=14, size=2)
            ui.burst(c, t, self.spot_t['storm'], sx, sy - 2, colors=('white', 'butter', '#c8d4ff', 'gold'), n=26,
                     seed=seed % 97, power=0.9, life=44, gravity=0.06)

    def _bridge(self, c, f, ox, oy, k, s):
        """The film's rope bridge on the gust clock; the loose plank is drawn here until it snaps."""
        gust = f - s if s is not None else f + 10 ** 6
        pts = rebind(sc._bridge, GUST=gust)(c, f, ox, oy, k)
        x, y, tw = pts[LOOSE]
        pf = self.plank_f
        if pf is None or f < pf:                         # still in place (rattling in the gust)
            A = 0.6 + 0.6 * k
            if s is not None:
                A += 5.5 * min(1.0, s / 4.0) * math.exp(-s / 48.0)
            dy = -1 if A > 1.6 and hash01(LOOSE, f // 2) > 0.72 else 0
            th = 1 + (1 if abs(tw) > 1.2 else 0)
            cc = sc.PLANK[3] if LOOSE % 4 else sc.PLANK[2]
            if s is not None and s < 120 and (f // 2) % 2:
                dy -= 1
                cc = sc.PLANK[4]
            c.rect(x, y + dy - th + 1, 2, th, cc)
            c.px(x, y + dy + 1, sc.PLANK[0])
            return
        dt = f - pf                                      # snapped: splinters, then the tumble
        if dt > 140:
            return
        P = sc.PLANK
        if dt < 8:
            for j in range(5):
                c.px(x - 2 + j + dt * (j - 2) // 3, y + 1 + dt // 2 + (j % 2), P[4] if dt < 4 else P[2])
        px_ = x + int(dt * (0.8 + 0.6 * k))
        py_ = y + int(0.5 * 0.30 * dt * dt + 0.6 * dt)
        if py_ >= H + 4:
            return
        fr = (dt // 3) % 4
        if fr == 0:
            c.hline(px_ - 2, px_ + 2, py_, P[4])
            c.hline(px_ - 2, px_ + 2, py_ + 1, P[1])
        elif fr == 1:
            c.line(px_ - 2, py_ - 2, px_ + 1, py_ + 1, P[4])
            c.line(px_ - 1, py_ - 2, px_ + 2, py_ + 1, P[1])
        elif fr == 2:
            c.vline(px_, py_ - 2, py_ + 2, P[4])
            c.vline(px_ + 1, py_ - 2, py_ + 2, P[1])
        else:
            c.line(px_ - 2, py_ + 1, px_ + 1, py_ - 2, P[4])
            c.line(px_ - 1, py_ + 1, px_ + 2, py_ - 2, P[1])

    @staticmethod
    def _debris(c, f, K, k, s):
        """sc._debris on the gust clock, with the drift speed integrated (phase K)."""
        n = 7
        if s is not None and s < 40:
            n = 7 + int(40 * math.exp(-s / 16.0))
        cols = ('#7d8245', '#8a6a44', '#a68e6b', '#5e5c30')
        for i in range(n):
            if i >= 7:
                sp = 9 + 5 * hash01(i, 1)
                x = int(-20 + (s + 2) * sp - hash01(i, 2) * 200)
                y = int(20 + hash01(i, 3) * 150 + math.sin(f * 0.5 + i) * 3)
            else:
                x = int((hash01(i, 2) * 400 + f * (2.2 + 1.5 * hash01(i, 1)) + K) % 400) - 40
                y = int(30 + hash01(i, 3) * 130 + math.sin(f * 0.21 + i) * 6)
            cc = cols[i % 4]
            c.px(x, y, cc)
            if (f // 2 + i) % 2:
                c.px(x + 1, y, cc)
            else:
                c.px(x, y + 1, mix(cc, 'ink', 0.3))
        if s is not None and s < 26:
            for i in range(14):
                sp = 13 + 6 * hash01(i, 8)
                x = int(-40 + s * sp - hash01(i, 9) * 120)
                y = int(14 + hash01(i, 10) * 160)
                L = 8 + int(hash01(i, 11) * 20)
                c.dither(x, y, L, 1, '#cfd4cf' if i % 2 else '#aeb4b1', 0.75)
