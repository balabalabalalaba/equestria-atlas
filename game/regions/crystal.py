"""The Crystal Empire as an explorable diorama (art: scenes/s18_crystal.py, used read-only).

Night under the aurora. The film shows three different views (dissolves / a glide between them), so the
region has three areas, flown through left to right with a cloud sweep between them:
  0  THE CRYSTAL CITY (shots A and C): the wide view of the city in its rising rings of crystal houses
     round the slender palace, Mt. Everhoof's moonlit pyramid behind, the aurora overhead and the
     shimmering dome. The camera pans the whole city and tilts up into the aurora (the film's shot C).
  1  THE PALACE (shot B): a tall panorama of the Crystal Palace, from the tip of its spire (which the
     film never quite shows) down the shard balconies to the arches, the Crystal Heart turning on its
     pedestal and the plaza with its rainbow glints.
  2  THE CRYSTAL STATION (shot D): the faceted train shed and the gift shop's snow-globe window in
     front of the city; the shed is widened to the east (the scene's own builder on a wider canvas, plus
     an end pillar, a snow bank with crystal clusters, platform lamps and a signal) so the Friendship
     Express runs in on open track. The camera cranes from the platform up to the city, like the film.
The film's render is shot-gated, so every area re-composes the scene's own cached layers and helpers
at the player's camera (recompose style, see sweetapple.py), without the film's balloons, callouts,
card, iris or dissolves. Layers are blitted with masked copies clipped to their art, the aurora is the
scene's curtain model computed once per two steps on a strip a little wider than the screen, and the
big glows are cached, so the city stays well inside the frame budget.

Timed beats on game clocks:
  - THE HEART'S WAVE (film 216..340): the first landing starts with the city unlit, as the film opens.
    SPACE at the Heart (wide view or close-up) sends the wave: in the wide view the ring of light rolls
    out through every ring of houses, lighting each as it passes; the dome flashes when it reaches the
    rim; the aurora surges into rainbow. The city then stays lit, on later landings too (Region.lasting
    'lit', saved); later pulses send another ring.
  - THE FRIENDSHIP EXPRESS (film 342..400): it pulls in the first time you reach the station (chuffs,
    whistle, the stop hiss), leaves west on SPACE and comes back by itself (or when called).
Things to set off with SPACE: the Heart's wave (wide view), the aurora's rainbow surge (up in the sky),
the Heart's beat (close-up: flash, the ring over the plaza, the near houses light), the snow globe
(shaken: a blizzard swirls inside, flakes and sparkles burst out) and the train (whistle: depart or
call). Three atlas stamps are hidden: low among the crystal shards at the far east end of the city
view, up by the very tip of the palace spire, and under the far end of the train-shed canopy.
No creatures anywhere (the edition's rule): crystal, light, snow and the empty train only.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, hash01, clamp, mix, col, ease_in_out, Sprite
from engine.core import bayer
import scenes.s18_crystal as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# ----------------------------------------------------------------------------- camera limits
#   wide city: fg banks (580 px from -100, p 1.3) cover cam -77..123; outer ring (p .92) -108..173;
#              everything else is wider. Vertical: shot C's tilt (-34) up into the aurora.
#   palace:    layers 380 px from -30 (no x pan in the film); the near houses (p 1.18) cover cam -25..25;
#              the tilt runs from above the spire tip (d 160) down to the plaza front (d -10)
#   station:   the shed layer (p 1 here) widened by D_EXT; the city behind at cam / 1.1 (outer ring ok)
CAM_A = (-30, 123)
CAMY_A = (-34, 0)
CAM_B = (-25, 25)
D_TOP, D_BOT = 160, -10                  # palace tilt range (film d = B_TILT - camy)
D_EXT = 170
D_W = sc.DW_ + D_EXT                      # widened station layer (E-x -20 .. 510)
CAM_D = (-18, D_W - sc.DOX - W - 2)
CAMY_D = (-26, 50)

P_IN, P_MID, P_OUT, P_FG = sc.P_IN, sc.P_MID, sc.P_OUT, sc.P_FG
HX_IN = sc.ring_centre(P_IN)             # the palace / heart (inner layer px)
HY_IN = sc.PAL_BASE - int(9 * sc.PAL_S)

# ----------------------------------------------------------------------------- clocks
C_SPEED = sc.C_SPEED                     # the wave front (px / step on the inner ring's scale)
WAVE_END = 170                           # steps until the front has left every ring
AUR_LEN = 150                            # the aurora's rainbow surge
GLOBE_LEN = 90
TRAIN_LEN = 170                          # train: arrival run (steps)
TRAIN_IN = D_W - sc.DOX + 30             # nose x where an arriving train starts (off the east end)
DEPART_ACC = 0.03
TRAIN_GONE = -sc.DOX - 200               # nose x past which the whole train has left (west)
TRAIN_RETURN = 540

TINT_NIGHT = ('#7f84cc', 0.46)           # moonlit blue
TINT_PINK = ('#f2b6dc', 0.30)            # the Heart's pink light
TINT_LIT = ('#dcd2f6', 0.22)             # inside the lit city's glow
TINT_LAMP = ('#f2d4a6', 0.30)            # the station's lanterns

# ----------------------------------------------------------------------------- landmarks
LANDMARKS_A = [
    Landmark('MT. EVERHOOF', 96, (156, 44, 'left', 14), y=-10, p=sc.P_EVER),
    Landmark('CRYSTAL EMPIRE', 172, (round(sc.ring_centre(P_OUT)) + 95, 128, 'up', 22), y=110, p=P_OUT),
]
LANDMARKS_B = [
    Landmark('CRYSTAL PALACE', sc.B_HX, (sc.B_HX + 1, -14, 'right', 26), y=-40),
    Landmark('CRYSTAL HEART', sc.B_HX + 20, (sc.B_HX + 11, sc.B_HY - 4, 'right', 40), y=160),
]
LANDMARKS_D = [
    Landmark('CRYSTAL STATION', 90, (50, 128, 'right', 30)),
]

GLOBE = (34, 176)
TRAIN_SPOT = (sc.TRAIN_STOP + 40, sc.D_RAIL - 36)

_AUR_ICON = Sprite.from_ascii('''
    .g.....v.
    .gg..v.v.
    ggg.vvvv.
    gGgmvmvv.
    gGGmmmvv.
    .GGmmmv..
    ..G.m.v..
    .........
    .........''', {'g': '#5dfca4', 'G': 'turquoise', 'm': '#7a6cf0', 'v': '#9d4fd8'})
_GLOBE_ICON = Sprite.from_ascii('''
    ..ooooo..
    .obwbbbo.
    obwbwbbbo
    obbbbwbbo
    obwbbbbbo
    .obbbwbo.
    .nnnnnnn.
    .nNNNNNn.
    .........''', {'o': '#2a2440', 'b': 'baby_blue', 'w': 'white', 'n': 'copper', 'N': 'rust'})

HOTSPOTS_A = [
    Hotspot('wave', HX_IN, HY_IN - 2, 'heart', r=40, cooldown=110, p=P_IN),
    Hotspot('aurora', 92, 30, _AUR_ICON, r=44, cooldown=100, p=0.1),
]
HOTSPOTS_B = [
    Hotspot('heart', sc.B_HX - 2, sc.B_HY, 'heart', r=44, cooldown=60),
]
HOTSPOTS_D = [
    Hotspot('globe', GLOBE[0], GLOBE[1] - 15, _GLOBE_ICON, r=40, cooldown=70),
    Hotspot('train', TRAIN_SPOT[0], TRAIN_SPOT[1], 'whistle', r=44, cooldown=90),
]
STAMPS_A = [Stamp('shards', 414, 146)]
STAMPS_B = [Stamp('spire', sc.B_HX + 20, -112)]
STAMPS_D = [Stamp('shed', 334, 166)]


# ============================================================================= fast drawing helpers
_MASKS = {}


def _blit(c, rgba, x, y, cols=None):
    """Canvas.blit for a cached RGBA layer: masked copy clipped to the layer's art (and to screen
    columns cols=(x0, x1) if given). Same pixels, much faster than the boolean-index blit."""
    e = _MASKS.get(id(rgba))
    if e is None:
        m = rgba[..., 3] >= 128
        rows, cs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        bb = (rows[0], rows[-1] + 1, cs[0], cs[-1] + 1) if rows.size else (0, 0, 0, 0)
        e = (rgba, np.ascontiguousarray(rgba[..., :3]), m, bb)
        _MASKS[id(rgba)] = e
    _, rgb, m, (r0, r1, c0, c1) = e
    x, y = int(x), int(y)
    X0, X1 = max(0, x + c0), min(W, x + c1)
    if cols is not None:
        X0, X1 = max(X0, int(cols[0])), min(X1, int(cols[1]))
    Y0, Y1 = max(0, y + r0), min(H, y + r1)
    if X1 <= X0 or Y1 <= Y0:
        return
    np.copyto(c.a[Y0:Y1, X0:X1], rgb[Y0 - y:Y1 - y, X0 - x:X1 - x], where=m[Y0 - y:Y1 - y, X0 - x:X1 - x, None])


def _sat_add(c, add, x, y):
    """Saturating add of a uint8 (h, w, 3) array at screen (x, y), clipped."""
    h, w = add.shape[:2]
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    reg = c.a[Y0:Y1, X0:X1]
    a = add[Y0 - y:Y1 - y, X0 - x:X1 - x]
    np.add(reg, np.minimum(a, 255 - reg), out=reg)


@lru_cache(maxsize=512)
def _glow_arr(r, color, sq, rings, bx, by):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(xx * xx + yy * yy) / r
    lvl = np.clip(1 - d, 0, 1) * rings
    thr = bayer(2 * r + 1, 2 * r + 1, bx, by)
    q = np.clip(np.floor(lvl + (thr - 0.5) * 0.9) / rings, 0, 1)
    return np.clip(np.array(col(color), np.float32) * q[..., None] * (sq / 100.0), 0, 255).astype(np.uint8)


def _glow(c, cx, cy, r, color, strength=0.5, rings=4):
    """fx.glow with the light pattern cached (strength quantised to 1/100)."""
    cx, cy, r = int(cx), int(cy), max(1, int(r))
    if cx + r < 0 or cx - r >= W or cy + r < 0 or cy - r >= H:
        return
    sq = int(round(clamp(strength, 0, 3) * 100))
    if sq <= 0:
        return
    _sat_add(c, _glow_arr(r, color, sq, rings, (cx - r) % 4, (cy - r) % 4), cx - r, cy - r)


# ----------------------------------------------------------------------------- the world clock's lights
# The film paints the city at night with every window, door and lantern lit (gold / butter glass, cream /
# white upper panes). The relight alone left them half lit by day (only its brightest warm pixels pass
# the lamp test: khaki doors, gold sills, white flicker flashes) and half dimmed at 19:36 and 05:36, when
# the city should read lit. So the region switches them itself: every light (the gold / butter glass of
# the scene layers, plus the cream / white panes of their window rects and of the doors) goes on at its
# own point of the dusk ramp (nightwin.ramp) in its painted colour, and is dark crystal glass, lit like
# the hour, while off (after_light). The painted glows (palace, Heart, lamps, lanterns, snow globe) fade
# with the same ramp (lamp_k): none by day.
_GOLD = np.array([col('gold'), col('butter')], np.uint8)
_GLASS = np.array([col('cream'), col('white')], np.uint8)
UNLIT = (mix('crystal_dk', 'periwinkle', 0.3), mix('crystal_dk', 'navy', 0.3))   # off glass: upper, lower
STATION_WINS = ((60, 157, 4, 6), (67, 157, 4, 6), (84, 157, 4, 6), (91, 157, 4, 6), (4, 157, 4, 6))
_LIGHT_SRC = {      # key -> lit -> (rgba, window rects, rect offset to layer px)
    'in': lambda lit: (sc.inner_layer(lit)[0], sc.inner_layer(lit)[1], (sc.LM, sc.LT)),
    'mid': lambda lit: (sc.mid_layer(lit)[0], sc.mid_layer(lit)[1], (sc.LM, sc.LT)),
    'out': lambda lit: (sc.outer_layer(lit)[0], sc.outer_layer(lit)[1], (sc.LM, sc.LT)),
    'b2': lambda lit: (sc.b_back_layer(2, lit)[0], sc.b_back_layer(2, lit)[1], (sc.BOX, sc.BOY)),
    'b1': lambda lit: (sc.b_back_layer(1, lit)[0], sc.b_back_layer(1, lit)[1], (sc.BOX, sc.BOY)),
    'bpal': lambda lit: (sc.b_palace_layer(False)[0], sc.b_palace_layer(False)[1], (sc.BOX, sc.BOY)),
    'bnear': lambda lit: (sc.b_near_layer(lit)[0], sc.b_near_layer(lit)[1], (sc.BOX, sc.BOY)),
    'station': lambda lit: (station_layer(), STATION_WINS, (sc.DOX, sc.DOY)),
}


def _is_any(rgb, cols):
    return np.all(rgb == cols[0], axis=-1) | np.all(rgb == cols[1], axis=-1)


@lru_cache(maxsize=None)
def _light_set(key, lit=False):
    """A scene layer's lights in layer px: (ys, xs, threshold v, upper pane?). Gold / butter glass
    anywhere; cream / white panes inside the layer's window rects or touching gold glass (a door's upper
    panes), never big pale areas (the platform's cream edge). One threshold per window (its rect) or per
    lit blob (doors, lanterns)."""
    from scipy import ndimage
    rgba, wins, (dx, dy) = _LIGHT_SRC[key](lit)
    rgb, op = rgba[..., :3], rgba[..., 3] >= 128
    gold = op & _is_any(rgb, _GOLD)
    glass = op & _is_any(rgb, _GLASS)
    rect = np.full(op.shape, -1, np.int32)
    for i, (x, y, w, h) in enumerate(wins):
        rect[max(0, y + dy):y + dy + h, max(0, x + dx):x + dx + w] = i
    lab, n = ndimage.label(glass)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    touch = np.zeros(n + 1, bool)
    touch[np.unique(lab[ndimage.binary_dilation(gold) & glass])] = True
    touch[0] = False
    m = gold | (glass & ((rect >= 0) | (touch & (size <= 48))[lab]))
    blob = ndimage.label(m, structure=np.ones((3, 3)))[0]
    ys, xs = np.nonzero(m)
    gid = np.where(rect[ys, xs] >= 0, rect[ys, xs], 10 ** 5 + blob[ys, xs])
    ug, inv = np.unique(gid, return_inverse=True)
    v = np.array([hash01(int(g), len(key), 811) * 0.9 for g in ug], np.float32)[inv]
    return ys, xs, v, glass[ys, xs]


def _note(c, notes, key, lit, x, y, cols=None):
    """Note where a layer's lights are on the canvas right after the layer (and its window flicker) went
    on at (x, y): screen ys, xs, thresholds, upper pane?, the colours now. cols: only these columns."""
    if notes is None:
        return
    ys, xs, v, top = _light_set(key, lit)
    sy, sx = ys + int(y), xs + int(x)
    x0, x1 = (0, W) if cols is None else (max(0, int(cols[0])), min(W, int(cols[1])))
    ok = (sx >= x0) & (sx < x1) & (sy >= 0) & (sy < H)
    if ok.any():
        sy, sx = sy[ok], sx[ok]
        notes.append((sy, sx, v[ok], top[ok], c.a[sy, sx].copy()))


SKY_EASE = 0.6                  # hours over which the clock's sky hands over to the painted night sky


@lru_cache(maxsize=4)
def _sky_switch(fq):
    """The clock hours where the painted night sky is kept from (evening) and until (small hours):
    daylight.sky_weight steps between 1 and 0 within ~0.05 h there."""
    ev = next(h for h in np.arange(18.0, 24.0, 0.05) if daylight.sky_weight(fq, h) <= 0.01)
    mo = next(h for h in np.arange(0.0, 6.0, 0.05) if daylight.sky_weight(fq, h) > 0.01)
    return round(float(ev), 2), round(float(mo), 2)


DAY_FILL = (150, 165, 170)      # the daylight that fills the night-painted shadows (C4)
DAY_FILL_K = 0.3


@lru_cache(maxsize=64)
def _fill_col(hq):
    """DAY_FILL in the clock's grade (warm and dimmer towards the evening)."""
    from engine import loc_worldmap as LW
    a = np.array([[DAY_FILL]], np.uint8)
    LW._apply_grade(a, hq)
    return a[0, 0].astype(np.float32)


@lru_cache(maxsize=64)
def _unlit(fq, tq):
    """The off glass (upper, lower) lit like the clock's hour tq in a scene painted at fq."""
    a = np.array([[col(UNLIT[0]), col(UNLIT[1])]], np.uint8)
    daylight.relight(a, fq, tq, snow=True, glow=False)
    return a[0]


# ----------------------------------------------------------------------------- the aurora
AUR_PAD = 24
_AX = np.arange(-AUR_PAD, W + AUR_PAD, dtype=np.float32)[None, :]
_AY = np.arange(0, 130, dtype=np.float32)[:, None]
_LUT_U = np.linspace(0, 1, 64, dtype=np.float32)
_AUR_LUT = np.stack([np.interp(_LUT_U, sc._AUR_STOPS_U, sc._AUR_COLS[:, j]) for j in range(3)], -1).astype(np.float32)
_AUR_THR = bayer(130, W + 2 * AUR_PAD)


@lru_cache(maxsize=8)
def _aurora_strip(f, fq, kq):
    """The scene's draw_aurora (same curtain model) for screen columns -PAD..W+PAD at camera 0, as a
    uint8 additive (130, W + 2 PAD, 3) array + its row span. flare = fq / 20, k = kq / 20."""
    flare, k = fq / 20.0, kq / 20.0
    xs = _AX
    tot = np.zeros((130, xs.shape[1], 3), np.float32)
    y_lo, y_hi = 130, 0
    for i, (by, amp, ln, ph, sp, st, ef) in enumerate(sc._CURTAINS):
        t = f * sp
        yb = (by + amp * np.sin(xs * 0.021 + t * 0.035 + ph)
              + amp * 0.5 * np.sin(xs * 0.051 - t * 0.047 + ph * 1.7)
              + 2.0 * np.sin(xs * 0.13 + t * 0.09 + ph))
        env = np.clip(np.sin(xs * ef + ph * 0.7 + t * 0.006) * 1.6 + 0.9, 0, 1)
        ray = 0.55 + 0.45 * np.sin(xs * 0.55 + 2.4 * np.sin(xs * 0.07 + t * 0.05 + ph) + t * 0.04)
        ray = np.round(ray * 3) / 3
        L = ln * (0.55 + 0.45 * ray) * (1 + 0.5 * flare)
        r0 = max(0, int(np.floor((yb - L).min())) - 1)
        r1 = min(130, int(np.ceil((yb + 0.06 * L).max())) + 2)
        if r1 <= r0:
            continue
        y_lo, y_hi = min(y_lo, r0), max(y_hi, r1)
        u = (yb - _AY[r0:r1]) / L
        inside = (u >= -0.06) & (u <= 1.0)
        w = env * (0.45 + 0.55 * ray) * st
        inten = np.where(u < 0, 1 + u * 12, np.clip(1 - u, 0, 1) ** 1.3) * w
        inten = np.where(inside, np.clip(inten, 0, 1), 0)
        uc = np.clip(u, 0, 1)
        colr = _AUR_LUT[(uc * 63 + 0.5).astype(np.intp)]
        if flare > 0:
            hue = (xs / 260.0 + _AY[r0:r1] / 200.0 - f * 0.012 + i * 0.2) % 1.0
            rb = sc._rainbow(hue)
            colr = colr + (rb - colr) * min(1.0, flare)
        tot[r0:r1] += colr * inten[..., None]
    if y_hi <= y_lo:
        return None, 0
    tot = tot[y_lo:y_hi]
    gain = 0.5 * k * (1 + 0.6 * flare)
    mx = tot.max(-1)
    lvl = np.clip(np.floor(np.clip(mx / 255.0, 0, 1) * 5 * gain * 1.6 + (_AUR_THR[y_lo:y_hi] - 0.5) * 0.9) / 5.0, 0, 1)
    add = tot / np.maximum(1, mx)[..., None] * 255.0 * (lvl * 0.8)[..., None]
    return np.clip(add, 0, 255).astype(np.uint8), y_lo


def draw_aurora(c, f, camx=0.0, camy=0.0, flare=0.0, k=1.0):
    """Aurora curtains at camera (camx, camy): the strip is recomputed every second step (the curtains
    drift well under a pixel per step) and shifted by whole pixels with the camera."""
    add, y = aurora_add(f, camx, camy, flare, k)
    if add is not None:
        _sat_add(c, add, 0, y)


def aurora_add(f, camx=0.0, camy=0.0, flare=0.0, k=1.0):
    """draw_aurora's light: (uint8 additive (h, W, 3) array, its screen y) or (None, 0)."""
    fq = int(round(clamp(flare, 0, 1.5) * 20))
    kq = int(round(clamp(k, 0, 3) * 20))
    add, y_lo = _aurora_strip(int(f) - int(f) % 2, fq, kq)
    if add is None:
        return None, 0
    s = int(round(camx * 0.08))
    s = max(-AUR_PAD, min(AUR_PAD, s))
    return add[:, AUR_PAD + s:AUR_PAD + s + W], int(round(-camy * 0.12)) + y_lo


def moon_at(hour):
    """The moon's screen spot at the clock's hour (on daylight's arc, where the clock's sky draws it when
    it repaints the sky), or None by day: the painted night sky shows it there too, so it never jumps
    when the sky switches between the clock's and the painted one (dusk, the small hours)."""
    kind, x, y = daylight._body_pos(hour)
    return (x, y) if kind == 'moon' else None


# ----------------------------------------------------------------------------- the wide city
def heart_bounce(a, amp=0.6):
    """The Heart's scale bounce a steps after a beat (the film's heart_scale / shot C's hk)."""
    if a is None or a < 0:
        return 1.0
    t = a / 18.0
    return 1.0 + amp * math.exp(-2.6 * t) * math.cos(t * 7.5)


def city_sky(c, f, camx, camy, flare=0.0, aur_k=1.0, glow=1.0, moon=None):
    """The first steps of city(): the sky alone (gradient, stars, aurora, moon) and the palace's two big
    glows (glow: their share by the clock, as city() draws them), for the sky hooks; Mt. Everhoof and the
    far mountains are stamped in inverted (their dark foothills and dithered faces match the night sky)."""
    oy = max(0, min(80, int(round(40 + camy * 0.12))))
    c.a[:] = sc.sky_canvas().a[oy:oy + H]
    _stars18(c, f)
    draw_aurora(c, f, camx, camy, flare, aur_k)
    if moon is not None:
        sc.draw_moon(c, *moon)
    hx, hy0 = int(round(HX_IN - camx * P_IN)), -int(round(camy * P_IN))
    _glow(c, hx, sc.PAL_BASE + hy0 + 10, 110, 'magic', 0.1 * glow, rings=5)    # the palace's big glows
    _glow(c, hx, sc.PAL_BASE + hy0 - 40, 46, 'crystal', 0.14 * glow)           # lie over the sky too
    _hide(c, sc.everhoof_layer(), -sc.LM - int(round(camx * sc.P_EVER)), -sc.LT - int(round(camy * sc.P_EVER)))
    _hide(c, sc.ranges_layer(), -sc.LM - int(round(camx * sc.P_RANGE)), -sc.LT - int(round(camy * sc.P_RANGE)))
    y = max(0, _ranges_bottom() - sc.LT - int(round(camy * sc.P_RANGE)))
    c.a[y:] = (0, 255, 0)                                # below the ranges: the plateau, never sky (a
    # sentinel: inverted sky colours matched the pale ice highlights)


_INV = {}


@lru_cache(maxsize=1)
def _star_table18():
    """fx.stars(seed=18, n=70, region=(0, 0, W, 105), the city's colours) precomputed per star."""
    cols = ('white', 'light_sky', 'lilac', 'baby_blue')
    out = []
    for i in range(70):
        x, y = int(hash01(18, i, 1) * W), int(hash01(18, i, 2) * 105)
        ph = hash01(18, i, 3)
        cc = cols[i % len(cols)]
        out.append((x, y, ph, hash01(18, i, 4) > 0.92, col(cc), col(mix(cc, 'navy', 0.4)), col(mix(cc, 'navy', 0.5))))
    return out


def _stars18(c, f):
    """The same pixels as city()'s fx.stars call, without re-hashing every star each frame."""
    a, white = c.a, col('white')
    for (x, y, ph, big, cc, rim, dim) in _star_table18():
        on = 0.5 + 0.5 * math.sin(2 * math.pi * (f / (50 + 70 * ph) + ph))
        if on < 0.25:
            continue
        if big and on > 0.6:
            a[y, x] = white
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < W and 0 <= y + dy < H:
                    a[y + dy, x + dx] = rim
        else:
            a[y, x] = cc if on > 0.6 else dim


@lru_cache(maxsize=1)
def _ranges_bottom():
    """One past the last row of the ranges layer's art (layer px)."""
    return int(np.flatnonzero(sc.ranges_layer()[..., 3].max(axis=1) >= 128)[-1]) + 1


def _hide(c, rgba, x, y):
    """_blit a scenery layer's opaque pixels with inverted colours (sky hooks: never counted as sky)."""
    v = _INV.get(id(rgba))
    if v is None or v[0] is not rgba:
        v = _INV[id(rgba)] = (rgba, np.dstack([255 - rgba[..., :3], rgba[..., 3:]]))
    _blit(c, v[1], x, y)


def city(c, f, camx, camy, lit_r=None, lit_all=False, flare=0.0, aur_k=1.0, heart_k=1.0, heart_glow=0.0,
         fg=True, glow=1.0, notes=None, moon=None):
    """The scene's city_frame at camera (camx, camy) onto canvas c. lit_r: radius of the wave front
    (screen px from the heart; houses inside it are lit) or None; lit_all: every ring lit. glow: the
    share of the painted night glows (palace, Heart, lamps) by the clock; notes: a list the rings' lights
    are noted in (_note); moon: its screen spot (moon_at) or None. Returns (hx, hy0): the heart's screen
    x and the inner layer's y offset."""
    oy = max(0, min(80, int(round(40 + camy * 0.12))))
    c.a[:] = sc.sky_canvas().a[oy:oy + H]
    fx.stars(c, f, seed=18, n=70, region=(0, 0, W, 105), colors=('white', 'light_sky', 'lilac', 'baby_blue'))
    draw_aurora(c, f, camx, camy, flare, aur_k)
    if moon is not None:
        sc.draw_moon(c, *moon)

    def place(rgba, par, cols=None):
        _blit(c, rgba, -sc.LM - int(round(camx * par)), -sc.LT - int(round(camy * par)), cols)

    place(sc.everhoof_layer(), sc.P_EVER)
    sx, sy = 150 - int(round(camx * sc.P_EVER)), 12 - int(round(camy * sc.P_EVER))
    for i in range(10):                                  # spindrift off the summit
        t = ((f * 0.8 + i * 9) % 70) / 70.0
        if hash01(i, f // 4) < 1 - t:
            c.px(sx + int(t * 34), sy + int(t * 8 + math.sin(t * 9 + i) * 1.5), mix('cloud', 'periwinkle', 0.4 * t))
    place(sc.ranges_layer(), sc.P_RANGE)
    place(sc.plateau_layer(), sc.P_PLAT)
    sc.snow_glitter(c, f, 3, (0, 125, W + 40, 30), 26, camx, camy, sc.P_PLAT)
    hx = int(round(HX_IN - camx * P_IN))
    hy0 = -int(round(camy * P_IN))
    offs = {P_IN: 0.0, P_MID: 14.0, P_OUT: 30.0}
    for name, fn, par in (('in', sc.inner_layer, P_IN), ('mid', sc.mid_layer, P_MID), ('out', sc.outer_layer, P_OUT)):
        sxl, syl = -int(round(camx * par)), -int(round(camy * par))
        r0 = fn(False)
        wins = r0[1]
        pieces = [(False, None)]
        if lit_all:
            r1 = fn(True)
            place(r1[0], par)
            wins = r1[1]
            pieces = [(True, None)]
        elif lit_r is not None and lit_r - offs[par] > 0:
            r1 = fn(True)
            hxl = sc.ring_centre(par) - camx * par
            R = lit_r - offs[par]
            a0, a1 = int(math.ceil(hxl - R)), int(math.ceil(hxl + R))   # |x - hxl| < R
            place(r0[0], par, (0, a0))
            place(r0[0], par, (a1, W))
            place(r1[0], par, (a0, a1))
            pieces = [(False, (0, a0)), (False, (a1, W)), (True, (a0, a1))]
            if min(W, a1) - max(0, a0) > W // 2:
                wins = r1[1]
        else:
            place(r0[0], par)
        sc.win_flicker(c, wins, sxl, syl, f, {'in': 1, 'mid': 2, 'out': 3}[name])
        for lit_, cols in pieces:
            _note(c, notes, name, lit_, sxl - sc.LM, syl - sc.LT, cols)
        if name == 'in':
            info = r0[2]
            hy = sc.PAL_BASE + hy0 - int(9 * sc.PAL_S)
            hgk = max(glow, clamp((heart_glow - 0.25) / 0.75))          # (a pulse of the Heart: any hour)
            _glow(c, hx, hy, int(16 + 30 * heart_glow), 'magic',
                  (0.22 + 0.06 * math.sin(f * 0.21) + 0.4 * heart_glow) * hgk)
            sc.draw_heart(c, hx, hy + int(round(math.sin(f * 0.1))), max(3, int(round(4 * heart_k))), f,
                          lit=0.4 * heart_glow)
            for sgn in (-1, 1):
                px_ = hx + sgn * int(round(12.5 * sc.PAL_S))
                sc.pennant(c, px_, int(info['side_tip']) + hy0 + 2, f + sgn * 5, 1, 'crystal_pk' if sgn < 0 else 'ice')
    _glow(c, hx, sc.PAL_BASE + hy0 + 10, 110, 'magic', 0.1 * glow, rings=5)
    _glow(c, hx, sc.PAL_BASE + hy0 - 40, 46, 'crystal', 0.14 * glow)
    if fg:
        place(sc.fg_layer(), P_FG)
    return hx, hy0


def light_rings(c, hx, gy, radii, ry=0.24):
    """Rings of light rolling over the ground plane (the film's wave ellipses), radii [(r, strength)]."""
    for rr, kk in radii:
        if rr <= 2 or rr >= 420 or kk <= 0.01:
            continue
        n = int(min(1600, 40 + rr * 7))
        th = np.linspace(0, 2 * math.pi, n)
        for dy in (0, -1):
            xs = np.round(hx + rr * np.cos(th)).astype(int)
            ys = np.round(gy + dy + rr * ry * np.sin(th)).astype(int)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            if not ok.any():
                continue
            pts = np.unique(np.stack([ys[ok], xs[ok]]), axis=1)
            sc.add_light(c, (pts[0], pts[1]), 'cotton', kk * (0.8 if dy else 1.0))


# ----------------------------------------------------------------------------- the station
@lru_cache(maxsize=1)
def station_layer():
    """The scene's d_station_layer on a canvas widened east by D_EXT, plus the shed's end pillar, two
    crystal lamps on the open platform and a signal mast (drawn with the scene's own helpers)."""
    rgba = rebind(sc.d_station_layer.__wrapped__, DW_=D_W, dlayer=lambda: Canvas(D_W, sc.DH_, sc.KEY))()
    cv = Canvas(D_W, sc.DH_, sc.KEY)
    cv.a[:] = rgba[..., :3]
    cv.a[rgba[..., 3] < 128] = sc.KEY
    o = sc.OC(cv, sc.DOX, sc.DOY)
    rpb = sc.hue_ramp('blue', True)
    for x in range(356, D_W - sc.DOX):                  # a snow bank behind the open platform
        k = min(1.0, (x - 356) / 18.0)
        top = sc.D_PLAT - k * (9 + 3 * math.sin(x * 0.05) + 2 * math.sin(x * 0.13 + 1))
        o.vline(x, top, sc.D_PLAT - 1, mix('periwinkle', 'baby_blue', 0.3))
        o.vline(x, top + 4, sc.D_PLAT - 1, mix('crystal_dk', 'periwinkle', 0.45))
        o.px(x, top, 'cloud')
    for x0, hue, hh in ((372, 'pink', 12), (386, 'blue', 8), (424, 'teal', 14), (440, 'lilac', 9),
                        (470, 'pink', 10), (506, 'blue', 13)):
        sc.gem_shaft(o, x0, sc.D_PLAT - 6, sc.D_PLAT - 6 - hh, 5 if hh > 10 else 4, 4, 4, sc.hue_ramp(hue, True))
    x = 350                                              # end pillar under the last gable
    o.rect(x - 1, 146, 5, 52, mix('crystal_dk', 'ink', 0.2))
    o.rect(x, 146, 1, 52, rpb['md'])
    o.rect(x + 1, 146, 2, 52, rpb['hi'])
    o.rect(x - 2, 194, 7, 4, rpb['lt'])
    o.rect(x - 2, 144, 7, 3, rpb['lt'])
    for lx in (402, 458):                                # crystal lamps on the open platform
        sc.lamp(o, lx, sc.D_PLAT, 16)
    sx = 492                                             # signal mast with a crystal lamp head
    o.rect(sx - 1, sc.D_PLAT - 30, 3, 30, mix('crystal_dk', 'ink', 0.3))
    o.vline(sx, sc.D_PLAT - 30, sc.D_PLAT - 1, rpb['md'])
    o.rect(sx - 3, sc.D_PLAT - 40, 7, 11, mix('indigo', 'ink', 0.3))
    o.rect(sx - 2, sc.D_PLAT - 39, 5, 9, mix('indigo', 'navy', 0.2))
    o.rect(sx - 4, sc.D_PLAT - 2, 9, 2, rpb['lt'])
    for x0, hue in ((380, 'teal'), (430, 'pink'), (476, 'lilac')):     # crystal shrubs by the bank
        for dx, hh in ((-3, 6), (0, 10), (3, 7)):
            sc.gem_shaft(o, x0 + dx, sc.D_RAIL + 24, sc.D_RAIL + 24 - hh + 2, 3, 2, 3, sc.hue_ramp(hue, True),
                         glow=False)
    return sc.to_rgba(cv)


LANTERNS = [116 + k * 34 for k in range(7)]             # the shed's hanging lanterns (E-x)
PLATFORM_LAMPS = [(402, sc.D_PLAT - 16 - 2), (458, sc.D_PLAT - 16 - 2)]
SIGNAL = (492, sc.D_PLAT - 35)


def _depart_x(a):
    return sc.TRAIN_STOP - DEPART_ACC * max(0, a) ** 2


def _arrive_x(a, x0, n):
    q = clamp(a / n)
    return sc.TRAIN_STOP + (x0 - sc.TRAIN_STOP) * (1 - q) ** 2.2


def _run(x0):
    """Steps an arrival from nose x x0 takes (the film's run: ~2.6 px per step at the start)."""
    return int(clamp((x0 - sc.TRAIN_STOP) / 2.5, 60, TRAIN_LEN))


# ============================================================================= the region
class CrystalEmpire(Region):
    SEG = 'crystal'
    CARD = ('CRYSTAL EMPIRE', ['CITY OF CRYSTAL', 'HOME OF THE HEART'])

    def build_areas(self):
        return [
            Area(self.render_city, cam_x=CAM_A, cam_y=CAMY_A, y_range=(-14, 160), start=(112, 88),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=TINT_NIGHT,
                 hotspots=HOTSPOTS_A, stamps=STAMPS_A, sky=self.sky_city),
            Area(self.render_palace, cam_x=CAM_B, cam_y=(-D_TOP, -D_BOT), y_range=(-D_TOP + 30, 178),
                 start=(40, 40), landmarks=LANDMARKS_B, balloon='l', balloon_tint=TINT_NIGHT,
                 hotspots=HOTSPOTS_B, stamps=STAMPS_B, sky=self.sky_palace),
            Area(self.render_station, cam_x=CAM_D, cam_y=CAMY_D, y_range=(-4, 196), start=(40, 70),
                 landmarks=LANDMARKS_D, balloon='m', balloon_tint=TINT_NIGHT,
                 hotspots=HOTSPOTS_D, stamps=STAMPS_D, sky=self.sky_station),
        ]

    def warm(self):
        """Also build the lit layers (the wave lights them mid-flight) and the big glows' patterns."""
        super().warm()
        c = Canvas()
        for fn in (sc.inner_layer, sc.mid_layer, sc.outer_layer):
            _blit(c, fn(True)[0], -500, -500)
        for lay in (sc.b_back_layer(2, True), sc.b_back_layer(1, True), sc.b_near_layer(True)):
            _blit(c, lay[0], -500, -500)
        for r, color, s, rings in ((110, 'magic', 0.1, 5), (46, 'crystal', 0.14, 4), (70, 'crystal', 0.12, 4),
                                   (40, 'crystal_pk', 0.12, 4)):
            for bx in range(4):
                for by in range(4):
                    _glow_arr(r, color, int(round(s * 100)), rings, bx, by)

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        self.lit = False            # the wave has lit the whole city (stays lit: Region.lasting)
        self.lit_wave = None        # step of the wave that is lighting the city (unlit -> lit)
        self.train = None           # None (away) | ('arrive', start step, from x, run) | ('depart', start step)
        self.train_back = None      # step at which a departed train comes back by itself
        self.train_seen = False

    def on_enter(self):
        # once the Heart has lit the city it stays lit on later landings (its SPACE then sends a pulse)
        self.lit = 'lit' in self.lasting or bool({'wave', 'heart'} & self.used)
        self.lit_wave = None
        self.train = None
        self.train_back = None
        self.train_seen = False
        for k in ('wave', 'aurora', 'heart', 'globe', 'train'):
            self.spot_t.pop(k, None)

    def on_area(self, ai):
        g = self.g
        if ai == 0:
            g.sfx('wind', 0.3, 0)
        elif ai == 1:
            g.sfx('crystal', 0.35)
        else:
            g.sfx('wind', 0.25, 0)
            if not self.train_seen:
                self.train_seen = True
                self.train, self.train_back = ('arrive', g.t + 30, TRAIN_IN, TRAIN_LEN), None

    def age(self, key, t, life):
        a = t - self.spot_t.get(key, -10 ** 6)
        return a if 0 <= a < life else None

    def lit_k(self, t):
        """0..1: how lit the city is (ramps up while a wave lights it, so nothing pops at the end)."""
        if self.lit:
            return 1.0
        if self.lit_wave is not None:
            return clamp((t - self.lit_wave) / WAVE_END)
        a = self.age('heart', t, 40)
        return clamp(a / 36.0) if a is not None else 0.0

    def flare(self, t):
        """Aurora flare / brightness from the aurora surge and the Heart's waves."""
        fl, k = 0.0, 1.0 + 0.2 * self.lit_k(t)
        a = self.age('aurora', t, AUR_LEN)
        if a is not None:
            e = ease_in_out(seg(a, 0, 24)) * (1 - ease_in_out(seg(a, 80, AUR_LEN)))
            fl, k = max(fl, e), k + 0.45 * e
        for key in ('wave', 'heart'):
            a = self.age(key, t, 150)
            if a is not None:
                e = ease_in_out(seg(a, 0, 40)) * (1 - ease_in_out(seg(a, 90, 150)))
                fl, k = max(fl, 0.8 * e), k + 0.4 * math.exp(-a / 22.0)
        return fl, k

    def train_state(self, t):
        """(nose x in station px, moving, kind, age) or None when the train is away."""
        if self.train is None or t < self.train[1]:
            return None
        kind, t0 = self.train[:2]
        a = t - t0
        if kind == 'arrive':
            x0, n = self.train[2:]
            return _arrive_x(a, x0, n), a < n, kind, a, n
        return _depart_x(a), True, kind, a, 0

    # ------------------------------------------------------------------ events
    def on_interact(self, key, t):
        g = self.g
        if key == 'wave':
            if not self.lit and self.lit_wave is None:
                self.lit_wave = t
            g.sfx('crystal', 0.7)
            g.sfx('shimmer', 0.45)
        elif key == 'aurora':
            g.sfx('shimmer', 0.6)
            g.sfx('sparkle', 0.35)
        elif key == 'heart':
            g.sfx('crystal', 0.75)
            g.sfx('chime', 0.4)
        elif key == 'globe':
            g.sfx('chime', 0.45)
            g.sfx('sparkle', 0.35)
        elif key == 'train':
            st = self.train_state(t)
            if st is None:                                   # away: call it in (just past the screen's edge)
                x0 = max(sc.TRAIN_STOP + 90, int(self.cx) + W + 12)
                self.train, self.train_back = ('arrive', t, x0, _run(x0)), None
                g.sfx('whistle', 0.4)
            elif st[2] == 'arrive' and not st[1]:            # parked: off it goes, west
                self.train = ('depart', t)
                g.sfx('whistle', 0.8)
            else:
                g.sfx('tick', 0.3)

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        # the wave lights the city ring by ring; once it has left the outer ring the city stays lit
        if self.lit_wave is not None and t - self.lit_wave > WAVE_END:
            self.lit, self.lit_wave = True, None
            self.lasting.add('lit')
        a = self.age('heart', t, 999)
        if a is not None and a > 36 and not self.lit and self.lit_wave is None:
            self.lit = True                                  # the close-up beat lit the city too
            self.lasting.add('lit')
        if self.ai == 0:
            a = self.age('wave', t, WAVE_END)
            if a in (int(150 / C_SPEED),):
                g.sfx('shimmer', 0.4)                         # the dome flashes at the rim
            if (t - self.t_enter) % 420 == 200:
                g.sfx('shimmer', 0.18)
        elif self.ai == 1:
            if t % 150 == 75:
                g.sfx('sparkle', 0.15)
        # the train (runs whichever area you are in; heard only at the station)
        st = self.train_state(t)
        if st is not None:
            x, moving, kind, a, n = st
            gain = self.near_gain(x + 40) if self.ai == 2 else 0.0
            if kind == 'arrive':
                if a % 36 == 0 and a < n - 24 and gain > 0:
                    g.sfx('chug', 0.45 * gain, (a // 36) % 2)
                elif a == min(60, n // 2) + 6 and gain > 0 and n >= TRAIN_LEN:
                    g.sfx('whistle', 0.55 * max(0.4, gain))
                elif a == n - 6 and gain > 0:
                    g.sfx('steam', 0.5 * gain)
            else:
                if a % 30 == 10 and gain > 0:
                    g.sfx('chug', 0.45 * gain, (a // 30) % 2)
                if x < TRAIN_GONE:
                    self.train, self.train_back = None, t + TRAIN_RETURN
        elif self.train_back is not None and t >= self.train_back:
            self.train, self.train_back = ('arrive', t, TRAIN_IN, TRAIN_LEN), None
        self._tint(t)

    def _tint(self, t):
        """Light the balloon like where it is: moonlit blue, pink near the Heart, warm under the
        station lanterns, bright while a wave of light sweeps past it."""
        A = self.area
        ex, ey = self.envelope()
        sx, sy = ex - self.cx, ey - self.cy                  # envelope on screen
        if self.ai == 0:
            hx = HX_IN - self.cx * P_IN
            hy = HY_IN - self.cy * P_IN
            d = math.hypot(sx - hx, (sy - hy) * 1.2)
            q = clamp(1 - (d - 24) / 110)
            base = _between(TINT_NIGHT, TINT_LIT, 0.5 * self.lit_k(t))
            tint = _between(base, TINT_PINK, q)
            a = self.age('wave', t, WAVE_END)
            if a is not None and abs(math.hypot(sx - hx, (sy - hy) * 2.5) - a * C_SPEED) < 26:
                tint = TINT_LIT
        elif self.ai == 1:
            d = math.hypot(sx - (sc.B_HX - self.cx), (sy - (sc.B_HY - self.cy)) * 1.1)
            q = clamp(1 - (d - 30) / 120)
            base = _between(TINT_NIGHT, TINT_LIT, 0.15 + 0.25 * self.lit_k(t))
            tint = _between(base, TINT_PINK, q)
            a = self.age('heart', t, 40)
            if a is not None and a < 24:
                tint = TINT_LIT
        else:
            q = clamp(1 - abs(ey - 160) / 60) * clamp((ex - 70) / 30) * clamp((380 - ex) / 30)
            g = clamp(1 - math.hypot(ex - GLOBE[0], ey - GLOBE[1]) / 60)
            tint = _between(TINT_NIGHT, TINT_LAMP, max(q, g))
        A.balloon_tint = tint

    def dome_k(self):
        """The share of the dome's shimmer drawn in the render: full while the painted night sky stays,
        none when the sky is repainted (dawn, day, dusk): its light added over the painted night sky would
        stay behind as dark-cored streaks on the repainted sky. The rest goes on in after_light, over the
        clock's sky (_sky_lights)."""
        return 1.0 - daylight.sky_weight(self.HOUR, self.g.hour)

    def sky_glow_k(self):
        """0 (day) .. 1 (night): the aurora and the dome's shimmer by the clock, while the clock's own sky
        shows (twilight, the small hours, dawn): they fade in with the dusk as the windows go on (out with
        the dawn), so nothing pops on when the painted night sky takes over (20.3 h) or goes (2.9 h)."""
        return clamp((daylight.night(self.g.hour) - 0.5) / 0.5)

    def clock_sky_k(self):
        """0..1: how much of the clock's night sky is blended over the painted one while the painted sky
        is kept, just after it takes over (evening) and before it goes (small hours): the gradient eases
        across over SKY_EASE hours instead of switching within minutes."""
        h = self.g.hour % 24
        ev, mo = _sky_switch(daylight._q(self.HOUR))
        u = 1 - (h - ev) / SKY_EASE if ev <= h < ev + SKY_EASE else (
            (h - mo + SKY_EASE) / SKY_EASE if mo - SKY_EASE <= h < mo else 0.0)
        return u * u * (3 - 2 * u)

    def _stash_sky(self, c, f, cx, cy):
        """End of an area render: the frame as drawn, while the clock's sky eases over the painted one."""
        self._pre = (c.a.copy(), f, cx, cy) if self.clock_sky_k() > 0.01 else None

    def _sky_lights(self, c, w, mask, dome=True):
        """The aurora (on the open sky pixels `mask` only) and the dome's shimmer over the clock's sky
        (blend weight w), with the area render's own aurora / dome parameters (_aur, _dome)."""
        aur, dome = getattr(self, '_aur', None), getattr(self, '_dome', None) if dome else None
        k = self.sky_glow_k()
        if aur is not None and k * w > 0.02:
            add, y = aurora_add(*aur)
            y0, y1 = max(0, y), min(H, y + (0 if add is None else add.shape[0]))
            if y1 > y0:
                a = add[y0 - y:y1 - y].astype(np.int16) * int(round(k * w * 256)) >> 8
                a[~mask[y0:y1]] = 0
                reg = c.a[y0:y1]
                reg[...] = np.minimum(reg + a, 255)
        if dome is not None:
            f, hx, y, dk, flash, keep = dome
            s_ = (dk * k + flash) * w
            if s_ > 0.05:
                if keep is None:
                    sc.dome(c, f, hx, y, 150, 108, s_)
                else:                                    # (only where the city still shows)
                    e = c.copy()
                    sc.dome(e, f, hx, y, 150, 108, s_)
                    np.copyto(c.a, e.a, where=keep[..., None])

    # ------------------------------------------------------------------ the world clock's lights
    def lamp_k(self):
        """0 (day) .. 1 (night): the dusk ramp the windows go on by (nightwin.ramp), for the painted
        glows (palace, Heart, lamps, lanterns, snow globe), in tenths (the glow patterns are cached)."""
        return round(nightwin.ramp(self.g.hour) * 10) / 10

    def _lights_done(self, c, notes):
        """End of an area render: the noted lights nothing has covered since (only additive glow and light
        drawn over them, never a nearer layer, the train or the snow), for after_light."""
        self._lit = None
        if not notes:
            return
        sy, sx, v, top, snap = (np.concatenate([n[i] for n in notes]) for i in range(5))
        now = c.a[sy, sx].astype(np.int16)
        d = now - snap
        ok = np.all((d >= 0) & (d <= 80), axis=1)
        self._lit = (sy[ok], sx[ok], v[ok], top[ok], c.a[sy[ok], sx[ok]].copy())

    def after_light(self, c, t):
        """Right after the clock's light: the aurora and the dome over the clock's sky (_sky_lights); each
        window, door and lantern on (its painted colour) once the dusk ramp passes its threshold, dark
        crystal glass lit like the hour while off."""
        w = daylight.sky_weight(self.HOUR, self.g.hour)
        pre, self._pre = getattr(self, '_pre', None), None
        if w > 0.01 and self._sky_memo is not None:
            self._sky_lights(c, w, self._sky_memo[1])
        elif w <= 0.01 and pre is not None:          # the painted night sky, the clock's eased over it
            wb = self.clock_sky_k()
            a, f, ix, iy = pre
            A = self.area
            mask = daylight.sky_pixels(a, A.sky(f, ix, iy).a)
            daylight.paint_sky(c.a, mask, self.g.hour, wb, ix, iy, A.cam_y[0] - 10, A.cam_y[1] - A.cam_y[0] + 130, t)
            self._sky_lights(c, wb, mask, dome=False)
        self._aur = self._dome = None
        self._day_fill(c, w)
        L, self._lit = getattr(self, '_lit', None), None
        if L is None:
            return
        sy, sx, v, top, rgb = L
        hour = self.g.hour
        on = v < nightwin.ramp(hour)
        c.a[sy[on], sx[on]] = rgb[on]
        off = ~on
        if off.any():
            un = _unlit(daylight._q(self.HOUR), daylight._q(hour))
            c.a[sy[off], sx[off]] = np.where(top[off][:, None], un[0], un[1])

    def _day_fill(self, c, w):
        """By day the city was still lit like the night it was painted in: its dark bands (the pine
        forest under the ranges, the frozen lake, shaded walls) stayed navy under a noon sky, as the
        relight only lifts a snow scene softly. The daylight fills them: pixels darker than 140 move
        towards DAY_FILL (graded by the clock), more the darker they are; never the sky."""
        k = DAY_FILL_K * (1.0 - clamp(daylight.night(self.g.hour) * 2))    # (gone by mid-dusk)
        if k < 0.01:
            return
        f = c.a.astype(np.float32)
        lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
        wk = np.clip((140.0 - lum) * (k / 140.0), 0, None)
        if w > 0.01 and self._sky_memo is not None:
            wk[self._sky_memo[1]] = 0
        f += (_fill_col(daylight.sky_q(self.g.hour)) - f) * wk[..., None]
        c.a[...] = f.astype(np.uint8)

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_city(self, f, cx, cy):
        """render_city's sky alone: city()'s night gradient, stars, aurora and moon."""
        fl, ak = self.flare(self.g.t)
        c = Canvas()
        city_sky(c, f, cx, cy, fl, ak, self.lamp_k(), moon_at(self.g.hour))
        return c

    def sky_palace(self, f, cx, cy):
        """render_palace's sky alone: draw_sky (gradient + stars) and the aurora."""
        fl, ak = self.flare(self.g.t)
        d = -cy
        c = Canvas()
        sc.draw_sky(c, f, 0, int(round(-30 + d * 0.5)))
        draw_aurora(c, f, cx * 0.5, int(round(-d * 0.45 + 110)), fl, 1.15 * ak)
        moon = moon_at(self.g.hour)
        if moon is not None:
            sc.draw_moon(c, *moon)
        return c

    def sky_station(self, f, cx, cy):
        """render_station's sky alone: the city's sky on the station's crane."""
        t = self.g.t
        fl, ak = self.flare(t)
        c = Canvas()
        city_sky(c, f, cx / 1.1, cy * 0.62, max(fl, 0.75 * self.lit_k(t)), ak, self.lamp_k(), moon_at(self.g.hour))
        return c

    # ------------------------------------------------------------------ area 0: the crystal city
    def render_city(self, f, cx, cy):
        t = self.g.t
        a = self.age('wave', t, WAVE_END)
        lighting = self.lit_wave is not None and not self.lit
        lit_r = (t - self.lit_wave) * C_SPEED if lighting else None
        fl, ak = self.flare(t)
        hk = heart_bounce(a)
        lk = self.lit_k(t)
        hg = max(clamp(1 - a / 24.0) if a is not None else 0.0, 0.25 * lk)
        gk = self.lamp_k()
        c = Canvas()
        notes = []
        hx, hy0 = city(c, f, cx, cy, lit_r=lit_r, lit_all=self.lit, flare=fl, aur_k=ak, heart_k=hk, heart_glow=hg,
                       glow=gk, notes=notes, moon=moon_at(self.g.hour))
        self._aur = (f, cx, cy, fl, ak)
        gy = sc.PAL_BASE + hy0 + 24                       # ground level of the rings (screen)
        dk = 1.0 + 0.25 * lk                                 # the dome shimmers by night (by day it only
        flash = 2.2 * math.exp(-abs(a - 150 / C_SPEED) / 8.0) if a is not None else 0.0   # flashes when
        self._dome = (f, hx, sc.PAL_BASE + 36 + hy0, dk, flash, None)                     # the wave hits it)
        if (dk + flash) * self.dome_k() > 0.05:
            sc.dome(c, f, hx, sc.PAL_BASE + 36 + hy0, 150, 108, (dk + flash) * self.dome_k())
        if a is not None:
            r = a * C_SPEED
            light_rings(c, hx, gy, [(r, 0.55), (r - 5, 0.25)])
            for sgn in (-1, 1):                              # the wall of light where the front meets the houses
                x = int(hx + sgn * r)
                if 0 <= x < W and r < 300:
                    fx.light_rect(c, x - 2, gy - 40, 5, 40, 'cotton', strength=0.25, fade='up')
            if a < 4:
                fx.grade(c, add=(40 - 10 * a, 20 - 5 * a, 32 - 8 * a))
        cm = int(round(sc.ring_centre(P_MID) - cx * P_MID))
        ly = 134 - 8 - int(round(cy * P_MID))
        for lx in (-15, 15, -60, 60, -100, 100):
            _glow(c, cm + lx, ly, 5, 'magic', (0.25 + 0.05 * lk + 0.08 * math.sin(f * 0.3 + lx)) * gk, rings=2)
        sc.snowfall(c, f, n=14, seed=2, speed=0.35, camx=cx, camy=cy)
        a = self.age('aurora', t, AUR_LEN)
        if a is not None:                                    # sparkles shower out of the curtains
            for j in range(3):
                fx.sparkles(c, int(92 + 90 * j - cx * 0.1), int(40 + 12 * (j % 2) - cy * 0.1), f, radius=26 + 6 * j,
                            n=8, colors=('white', '#b8ffd0', 'crystal_lt', 'lilac'), seed=j + 3, period=24)
            ui.burst(c, t, self.spot_t['aurora'], int(92 - cx * 0.1), int(30 - cy * 0.1),
                     ('#b8ffd0', '#5dfca4', 'turquoise', '#9d4fd8', 'white'), n=30, seed=11, power=1.1, life=60,
                     gravity=0.03)
        aw = self.age('wave', t, 60)
        if aw is not None:
            ui.burst(c, t, self.spot_t['wave'], hx, sc.PAL_BASE + hy0 - 8, ('cotton', 'crystal_pk', 'white', 'crystal_lt'),
                     n=26, seed=7, power=0.9, life=50, gravity=0.02)
        self._stash_sky(c, f, cx, cy)
        self._lights_done(c, notes)
        return c

    # ------------------------------------------------------------------ area 1: the palace
    def render_palace(self, f, cx, cy):
        t = self.g.t
        d = -cy
        a = self.age('heart', t, 90)
        fl, ak = self.flare(t)
        c = Canvas()
        sc.draw_sky(c, f, 0, int(round(-30 + d * 0.5)))
        draw_aurora(c, f, cx * 0.5, int(round(-d * 0.45 + 110)), fl, 1.15 * ak)
        self._aur, self._dome = (f, cx * 0.5, int(round(-d * 0.45 + 110)), fl, 1.15 * ak), None
        moon = moon_at(self.g.hour)
        if moon is not None:
            sc.draw_moon(c, *moon)
        ring = a * 9 if a is not None else -1
        lit_all = self.lit
        bh = sc.B_HX
        gk = self.lamp_k()
        notes = []

        def lay(fn, par, key, lit_ok=True):
            r0 = fn(False)
            oy = int(round(d * par)) - sc.BOY
            ox = -sc.BOX - int(round(cx * par))
            pieces = [(False, None)]
            if lit_ok and lit_all:
                _blit(c, fn(True)[0], ox, oy)
                pieces = [(True, None)]
            elif lit_ok and ring > 0:
                hxl = bh - cx * par
                R = ring * (0.75 + 0.25 * par)
                a0, a1 = int(math.ceil(hxl - R)), int(math.ceil(hxl + R))
                _blit(c, r0[0], ox, oy, (0, a0))
                _blit(c, r0[0], ox, oy, (a1, W))
                _blit(c, fn(True)[0], ox, oy, (a0, a1))
                pieces = [(False, (0, a0)), (False, (a1, W)), (True, (a0, a1))]
            else:
                _blit(c, r0[0], ox, oy)
            sc.win_flicker(c, r0[1], ox + sc.BOX, oy + sc.BOY, f, int(par * 100))
            for lit_, cols in pieces:
                _note(c, notes, key, lit_, ox, oy, cols)
            return oy + sc.BOY

        lay(lambda lit: sc.b_back_layer(2, lit), sc.PB_BACK2, 'b2')
        lay(lambda lit: sc.b_back_layer(1, lit), sc.PB_BACK1, 'b1')
        poy = lay(sc.b_palace_layer, sc.PB_PAL, 'bpal', lit_ok=False)
        info = sc.b_palace_layer(False)[2]
        hx = bh - cx
        _glow(c, hx, int(info['top']) + poy - 60, 70, 'crystal', 0.12 * gk)
        for sgn in (-1, 1):
            sc.pennant(c, hx + sgn * int(round(12.5 * sc.B_S)), int(info['side_tip']) + poy + 4, f + sgn * 7, 2,
                       'crystal_pk' if sgn < 0 else 'ice')
        self._crown(c, f, hx, poy, t)
        hy = sc.B_HY + poy + int(round(math.sin(f * 2 * math.pi / 72) * 1.5))
        k = heart_bounce(a, 0.42)
        pulse = clamp(1 - a / 24.0) if a is not None else 0.0
        _glow(c, hx, hy, int(30 + 70 * (1 - pulse)) if pulse > 0 else 30, 'magic',
              (0.3 + 0.05 * math.sin(f * 0.35) + 0.06 * self.lit_k(t)) * gk + 0.5 * pulse, rings=5)
        _glow(c, hx, sc.B_Y0 + poy + 4, 40, 'crystal_pk', 0.12 * gk)
        sc.draw_heart(c, hx, hy, max(4, int(round(sc.B_HSIZE * k))), f, lit=0.3 * pulse)
        self._glints(c, f, hx, sc.B_Y0 + poy)
        lay(sc.b_near_layer, sc.PB_NEAR, 'bnear')
        noy = int(round(d * sc.PB_NEAR))
        nox = -int(round(cx * sc.PB_NEAR))
        for lx in (70, 250):
            _glow(c, lx + nox, 172 - 52 + noy, 12, 'magic', (0.3 + 0.06 * math.sin(f * 0.27 + lx)) * gk, rings=3)
        for lx, cc_ in ((70, 'ice'), (250, 'crystal_pk')):
            sc.pennant(c, lx + nox, 172 - 40 + noy, f + lx, 3, cc_)
        sc.snowfall(c, f, n=10, seed=9, speed=0.9, camx=cx * 1.4, camy=-d * 1.4)
        if a is not None:                                    # the heartbeat: flash, rings over the plaza
            if a < 4:
                fx.grade(c, add=(60 - 15 * a, 30 - 8 * a, 50 - 12 * a))
            light_rings(c, hx, sc.B_Y0 + poy + 6, [(ring, 0.6 * (1 - a / 30.0)), (ring - 4, 0.3 * (1 - a / 30.0))],
                        ry=0.22)
            if a < 20:
                _glow(c, hx, hy, 24 + int(a * 9), 'cotton', 0.3 * (1 - a / 20.0))
            ui.burst(c, t, self.spot_t['heart'], hx, hy, ('cotton', 'crystal_pk', 'white', 'lilac', 'crystal_lt'),
                     n=34, seed=5, power=1.2, life=56, gravity=0.03)
        self._stash_sky(c, f, cx, cy)
        self._lights_done(c, notes)
        return c

    def _crown(self, c, f, hx, poy, t):
        """The spire's tip: a slow glint (the film never shows the very top)."""
        ty = -135 + poy
        if -10 < ty < H:
            ph = (f % 90)
            if ph < 10:
                fx.sparkle(c, hx, ty + 2, ph, 'white', period=10, size=3)

    def _glints(self, c, f, hx, fy):
        """Refracted rainbow glints on the plaza + the spectrum fan under the Heart (the film's)."""
        step = f // 3
        for i in range(14):
            a_ = (i / 14.0) * 2 * math.pi + step * 0.07
            rx_ = 30 + 26 * hash01(i, 2)
            gx = hx + int(round(math.cos(a_) * rx_ * 1.6))
            gy = fy + 12 + int(round(math.sin(a_) * rx_ * 0.32))
            if gy <= fy + 1:
                continue
            rc = _RB[int(((i / 14.0 + step * 0.01) % 1.0) * 64) % 64]
            tw = (step + i * 5) % 9
            if tw < 6:
                c.rect(gx - 1, gy, 3, 1, mix(rc, 'white', 0.25))
                if tw in (2, 3):
                    c.px(gx, gy - 1, 'white')
                    c.px(gx, gy + 1, mix(rc, 'white', 0.5))
                    c.rect(gx - 2, gy, 5, 1, mix(rc, 'white', 0.4))
        for j in range(6):
            rc = _RB[int(j / 6.0 * 64)]
            x1 = hx - 14 + j * 6 + int(round(2 * math.sin(step * 0.07)))
            ys = np.arange(fy + 2, fy + 9)
            tt = (ys - fy - 2) / 7.0
            xs = np.round(hx - 2 + j * 0.8 + (x1 - (hx - 2 + j * 0.8)) * tt).astype(int)
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            sc.add_light(c, (ys[ok], xs[ok]), rc, 0.35)

    # ------------------------------------------------------------------ area 2: the station
    def render_station(self, f, cx, cy):
        t = self.g.t
        fl, ak = self.flare(t)
        ccx, ccy = cx / 1.1, cy * 0.62                       # the city behind (the film's crane)
        c = Canvas()
        lk = self.lit_k(t)
        lit_r = (t - self.lit_wave) * C_SPEED if self.lit_wave is not None and not self.lit else None
        gk = self.lamp_k()
        notes = []
        hx, hy0 = city(c, f, ccx, ccy, lit_r=lit_r, lit_all=self.lit, flare=max(fl, 0.75 * lk), aur_k=ak,
                       heart_glow=0.25 * lk, fg=False, glow=gk, notes=notes, moon=moon_at(self.g.hour))
        self._aur = (f, ccx, ccy, max(fl, 0.75 * lk), ak)
        dk = 1.0 + 0.2 * lk
        if dk * self.dome_k() > 0.05:
            sc.dome(c, f, hx, sc.PAL_BASE + 36 + hy0, 150, 108, dk * self.dome_k())
        behind = c.a.copy()                                  # (the city: where the dome may go on later)
        cm = int(round(sc.ring_centre(P_MID) - ccx * P_MID))
        for lx in (-15, 15, -60, 60, -100, 100):
            _glow(c, cm + lx, 134 - 8 + int(round(-ccy * P_MID)), 5, 'magic', 0.3 * gk, rings=2)
        ox, sy = -cx, -cy                                    # station px -> screen
        _blit(c, station_layer(), -sc.DOX + ox, sy - sc.DOY)
        _note(c, notes, 'station', False, -sc.DOX + ox, sy - sc.DOY)
        ag = self.age('globe', t, GLOBE_LEN)
        gx, gyy = GLOBE[0] + ox, GLOBE[1] + sy
        shake = 0
        gf = f
        if ag is not None:
            shake = [2, -2, 2, -1, 1, -1, 1, 0, 1, 0][ag] if ag < 10 else 0
            gf = f + int(_spin(ag, GLOBE_LEN, 160))         # the flurry whirls fast, then settles
            k = 1 - ag / GLOBE_LEN
            _glow(c, gx, gyy, 46, 'crystal_lt', 0.45 * k, rings=5)
        _glow(c, gx, gyy, 24, 'crystal_lt', (0.25 + 0.05 * math.sin(f * 0.2)) * gk, rings=4)
        sc.snow_globe(c, gx + shake, gyy, 13, gf)
        if ag is not None:
            self._blizzard(c, gx + shake, gyy, ag, f)
        for x in LANTERNS + [354]:
            _glow(c, x + 1 + ox, 160 + sy, 7, 'gold', 0.25 * gk, rings=2)
        for x, y in PLATFORM_LAMPS:
            _glow(c, x + ox, y + sy, 8, 'magic', (0.3 + 0.06 * math.sin(f * 0.27 + x)) * gk, rings=3)
        on = (f // 45) % 4 != 3
        c.rect(SIGNAL[0] - 1 + ox, SIGNAL[1] - 2 + sy, 3, 3, '#7dffb0' if on else mix('crystal_dk', 'navy', 0.3))
        if on:
            _glow(c, SIGNAL[0] + ox, SIGNAL[1] - 1 + sy, 6, '#5dfca4', 0.3 * gk, rings=2)
        st = self.train_state(t)
        if st is not None:
            self._train(c, f, st, ox, sy)
        sc.snowfall(c, f, n=46, seed=31, speed=0.5, camx=cx, camy=cy)
        sc.snowfall(c, f, n=10, seed=32, speed=1.0, camx=cx * 1.3, camy=cy * 1.3)
        self._dome = (f, hx, sc.PAL_BASE + 36 + hy0, dk, 0.0, np.all(c.a == behind, axis=2))
        self._stash_sky(c, f, cx, cy)
        self._lights_done(c, notes)
        return c

    def _train(self, c, f, st, ox, sy):
        x, moving, kind, a, n = st
        tx = int(round(x)) + ox
        if not -260 < tx < W + 20:
            return
        stx, sty = sc.draw_train(c, tx, sc.D_RAIL + sy, f, moving)
        _glow(c, tx + 4, sc.D_RAIL + sy - 15, 6, 'butter', 0.35 * self.lamp_k(), rings=2)
        spd = 1.0 if moving else 0.4
        for i in range(7):                                   # steam from the stack
            per = 26
            q = ((f * spd + i * per / 7.0) % per) / per
            px_ = stx + int(q * (14 if moving else 4)) + int(math.sin(q * 6 + i) * 2)
            py_ = sty - int(q * 22)
            r_ = 1 + int(q * 4)
            if hash01(i, int(f * spd) // 9) < 1 - q * 0.6:
                c.circle(px_, py_, r_, mix('white', 'cloud_dk', q))
                c.circle(px_ - 1, py_ - 1, max(1, r_ - 1), 'white')
        if kind == 'arrive' and n - 4 <= a < n + 20:     # the stop hiss
            q = (a - n + 4) / 24.0
            for k in range(6):
                r_ = int(2 + q * 7 + k % 3)
                c.circle(tx + 14 - int(q * 14) - k * 3, sc.D_RAIL + sy - 3 - int(q * 5), r_, mix('white', 'cloud', q))
        if kind == 'depart' and a < 24:                      # the start: a burst of steam at the wheels
            q = a / 24.0
            for k in range(6):
                r_ = int(2 + q * 6 + k % 3)
                c.circle(tx + 20 + int(q * 16) + k * 4, sc.D_RAIL + sy - 3 - int(q * 4), r_, mix('white', 'cloud', q))

    def _blizzard(self, c, gx, gy, a, f):
        """The shaken globe: a dense blizzard whirls inside, the little palace lights up, flakes and
        sparkles burst out and a flurry sweeps over the station."""
        t0 = self.g.t - a
        k = 1 - a / GLOBE_LEN
        r = 12
        for i in range(40 if a < 50 else 20):                # the blizzard inside the glass
            ang = i * 2.39 + a * (0.22 + 0.05 * (i % 4)) * (1.0 if i % 2 else -0.8)
            rr = (r - 2) * (0.25 + 0.75 * hash01(i, 7))
            x = int(round(gx + math.cos(ang) * rr))
            y = int(round(gy - 1 + math.sin(ang) * rr * 0.85))
            if (x - gx) ** 2 + (y - gy) ** 2 < (r - 1) ** 2:
                c.px(x, y, 'white' if i % 3 else 'crystal_lt')
        _glow(c, gx, gy - 2, 7, 'cotton', 0.5 * k, rings=2)   # the tiny palace inside glows
        for i in range(3):
            ui.burst(c, self.g.t, t0 + i * 10, gx, gy - 12, ('white', 'cloud', 'crystal_lt', 'periwinkle'),
                     n=26, seed=21 + i, power=1.0 - 0.15 * i, life=56, gravity=0.012)
        ui.burst(c, self.g.t, t0 + 4, gx, gy - 12, ('white', 'crystal_pk', 'gold', 'crystal_lt'), n=18, seed=25,
                 power=0.7, life=50, gravity=-0.01, size=1)
        if a < 60:
            for j in range(4):
                fx.sparkle(c, gx - 16 + 11 * j, gy - 16 - 6 * (j % 2), (a + j * 5) % 18, 'white', period=18, size=3)
        for i in range(64):                                  # a flurry blows out over the station
            b_ = a - int(hash01(41, i) * 30)
            if not 0 <= b_ < 60 or (b_ > 42 and (i + b_) % 3 == 0):
                continue
            sp = 1.6 + 3.4 * hash01(41, i, 2)
            ang = -1.0 + 1.15 * hash01(41, i, 3)
            x = int(round(gx + math.cos(ang) * sp * b_ + 2 * math.sin(b_ * 0.3 + i)))
            y = int(round(gy - 10 + math.sin(ang) * sp * b_ * 0.55 + 0.03 * b_ * b_))
            if i % 3 == 0:
                c.px(x, y, 'crystal_lt')
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.px(x + dx, y + dy, 'white')
            else:
                c.rect(x, y, 2 if i % 3 == 1 else 1, 1, 'white' if i % 2 else 'cloud')


_RB = [tuple(int(v) for v in sc._rainbow(np.array([h / 64.0]))[0]) for h in range(64)]


def _between(a, b, q):
    """A balloon tint q (0..1, quantised to eighths: the compositor caches one sprite per tint) of the
    way from tint a to tint b."""
    q = round(clamp(q) * 8) / 8
    if q <= 0:
        return a
    if q >= 1:
        return b
    return mix(a[0], b[0], q), round(a[1] + (b[1] - a[1]) * q, 3)


def _spin(a, n, extra):
    """Extra frames after `a` steps of a spin-up adding `extra` frames over n steps (smooth, ends flat)."""
    a = min(max(a, 0), n)
    return extra / 2 * (1 - math.cos(math.pi * a / n))
