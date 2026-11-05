"""Rainbow Falls (+ Our Town) as an explorable diorama (art: scenes/s07_rainbowfalls.py, used read-only).

The film shows two views: the clifftop village and its five rainbow falls, panned and tilted down to the
stone rail trestle (render_rf, a 2D camera with `off(cam, k)` parallax), then a cut to Our Town seen from
a hill (render_ot, a short horizontal pan). Each is one Area here, re-composed from the scene's cached
layers and animation helpers in the film's layer order (recompose style), without the film's balloon,
callouts, card or iris:

  area 0  RAINBOW FALLS  pan + tilt over the village, the falls, the hoop course, the trestle and river
  area 1  OUR TOWN       the valley of identical houses, reached by flying past the right edge

Timed beats on game clocks: the Friendship Express crosses the trestle (right to left, chuffing on 8ths,
whistle jet + steam burst mid-span at the film's HIT spot) soon after landing, then every TRAIN_PERIOD
steps. Endless drifts in the film (clouds, dust devils, the tumbleweed) wrap around off screen.

Things to set off with SPACE: the main fall bursts (a rainbow geyser at its lip, a surge down the water
and a rainbow arching from fall to fall), the hoop course lights up ring by ring along its dotted line,
the Friendship Express whistles across the trestle (called in from the right if it isn't on screen),
Our Town's '=' banner flaps in a gust (every identical house puffs one identical smoke ring at the same
moment), and its big tree bursts into pink blossom that drifts down the street.
Three atlas stamps are hidden: high above the judges' tower, low at the far east end of the gorge over
the foreground knoll, and by the old rail end at the west edge of Our Town.

The landing spot is lower than the film's opening shot (village + sky): from there one view holds the
village, the falls and the trestle the train is about to cross; climb for the sky and the hoops, sink
for the gorge and the river.
"""
import math

import numpy as np

from functools import lru_cache

from engine import W, H, Canvas, Sprite, fx, mix, col, hash01, bayer, ease_in_out, clamp
from engine.fx import steam
import scenes.s07_rainbowfalls as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp
from .everfree import GlowKeep

# ------------------------------------------------------------------ RAINBOW FALLS (area 0)
# camera = top-left of the view in main-layer (k=1) px. Limits (layer width / height, parallax k,
# off(cam, k) = REF + (cam - REF) * k): main 600x330 -> x <= 280, y <= 150; trestle 640x380 (k 1.15)
# -> x >= 15, x <= 292; foreground 680x380 (k 1.4) -> x <= 288. The film's path is (40..218, 32..128).
RF_CAM_X = (15, 280)
RF_CAM_Y = (16, 150)

TRAIN_START = 649          # loco front (trestle-layer x) when a run starts: off screen right at any camera
TRAIN_HIT = (TRAIN_START - sc.TRAIN_A) // 3     # run step of the film's trestle HIT (whistle + burst)
TRAIN_EMIT = 262           # last chuff (front well past the left edge)
TRAIN_PERIOD = 450         # one crossing every 15 s

MAIN_X = sc.MAIN_FALL[0] + sc.MAIN_FALL[1] // 2
_r6x, _r6y, _ = sc.RINGS[6]

RF_LANDMARKS = [           # the film's callouts, in world (main layer) px
    Landmark('RAINBOW FALLS', MAIN_X, (sc.MAIN_FALL[0] + sc.MAIN_FALL[1] - 2, 178, 'right', 14), y=170),
    # the trestle layer moves 1.15x the camera around REF (110, 128): anchor (146, 266) at REF
    Landmark('RAIL TRESTLE', 130, (163, 285, 'up', 16), y=290, p=1.15),
    Landmark('HOOP COURSE', int(round(_r6x)), (int(round(_r6x)) + sc.RING_R - 1, int(round(_r6y)), 'right', 12),
             y=165),
]

_IPAL = {'o': ui.INK, 'r': 'red', 'y': 'gold', 'b': 'sky', 'c': 'cloud', 'w': 'white', 'p': 'hot_pink',
         'P': 'cotton', 'G': 'green'}
ICON_RAINBOW = Sprite.from_ascii('''
    ...rrr...
    .rryyyrr.
    .ryybyyr.
    rybb.bbyr
    ryb...byr
    ryb...byr
    ryb...byr
    cw.....wc
    ccc...ccc''', _IPAL)
ICON_BLOSSOM = Sprite.from_ascii('''
    ..pp.pp..
    .pPPpPPp.
    .pPPPPPp.
    ..pPyPp..
    .pPPPPPp.
    .pPPpPPp.
    ..pp.pp..
    ....G....
    ...GG....''', _IPAL)

# hotspots (world px; the trestle one sits on the trestle layer: its x, y are REF-shifted like the tag)
LIP = (MAIN_X, sc.MESA_TOP - 1)                      # the main fall's lip
ARC = (296, sc.MESA_TOP - 2, 76, 64)                # rainbow from fall 2's lip to fall 4's lip: cx, base, rx, ry
TRESTLE_SPOT = (240, sc.RAIL_Y - 8 + 19)            # on the track, mid-gorge (trestle layer, p 1.15)
RF_HOTSPOTS = [
    Hotspot('hoops', int(round(sc.RINGS[0][0])), int(round(sc.RINGS[0][1])), 'spark', cooldown=110),
    Hotspot('train', TRESTLE_SPOT[0], TRESTLE_SPOT[1], 'whistle', cooldown=120, p=sc.TRK),
    Hotspot('falls', LIP[0], LIP[1], ICON_RAINBOW, cooldown=90),
]
RF_STAMPS = [Stamp('tower', 208, 46), Stamp('knoll', 568, 246)]

# ------------------------------------------------------------------ OUR TOWN (area 1)
# layers: town 400 (k 1) -> x <= 80; foreground 420 (k 1.35, +10) -> x <= 66; vault 460 (k .85, +30)
OT_CAM_X = (0, 66)
OT_LANDMARKS = [
    Landmark('OUR TOWN', 168, (168, sc.OT_STREET_Y[0] - 8, 'up', 26)),
]
BANNER = (sc.OT_POLE[0] + 10, sc.OT_POLE[1] - 36)    # the '=' banner's middle
CANOPY = (sc.OT_TREE[0], sc.OT_TREE[1] - 34)         # the big tree's canopy centre
OT_HOTSPOTS = [
    Hotspot('banner', BANNER[0], BANNER[1], 'flag', cooldown=80),
    Hotspot('tree', CANOPY[0], CANOPY[1] - 4, ICON_BLOSSOM, cooldown=100),
]
OT_STAMPS = [Stamp('railend', 20, 122)]
RAINBOW = ('red', 'orange', 'gold', 'leaf', 'sky', 'violet')


def _paste(c, layer, ox, oy):
    """sc._paste (layer pixel (ox, oy) lands on screen (0, 0)), ~4x faster with np.copyto."""
    rgb, m = layer
    h, w = m.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + W), min(h, oy + H)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0 - oy:y1 - oy, x0 - ox:x1 - ox], rgb[y0:y1, x0:x1], where=m[y0:y1, x0:x1, None])


def _stamp(c, layer, ox, oy):
    """Mark a layer's opaque pixels (placed as _paste places it) magenta in a sky hook: never sky."""
    rgb, m = layer
    h, w = m.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + W), min(h, oy + H)
    if x1 > x0 and y1 > y0:
        c.a[y0 - oy:y1 - oy, x0 - ox:x1 - ox][m[y0:y1, x0:x1]] = (255, 0, 255)


def _tree_canopy(c, f, x, y):
    """sc._ot_tree_canopy, minus its dark leaf dots that land on the light puffs: there a pair of them
    plus a third read as a little face (no creatures, no faces)."""
    sc._ot_tree_canopy(c, f, x, y)
    leaf, dark = np.array(col('leaf'), np.uint8), np.array(col('forest'), np.uint8)
    for k in range(10):                               # the film's dot positions
        a = hash01(k, 5) * 6.283
        rr = hash01(k, 6) * 14
        px, py = x + int(math.cos(a) * rr) + 1, y - 3 + int(math.sin(a) * rr * 0.8)
        if 1 <= px < W - 1 and 1 <= py < H - 1 and np.array_equal(c.a[py, px], dark):
            nb = c.a[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3)
            if np.all(nb == leaf, axis=1).sum() >= 3:
                c.a[py, px] = leaf


@lru_cache(maxsize=1)
def _ot_haze():
    """(the town sky's rows under the far mesas, the same toned down to a grey haze)."""
    a = sc._ot_sky()[OT_HAZE_Y:100]
    g = a.astype(np.float32).mean(axis=2, keepdims=True)
    return a, (a * 0.35 + g * 0.65 * 0.85).astype(np.uint8)


def _paste_wrap(c, layer, ox, oy, period):
    """_paste for a layer that drifts forever: it repeats every `period` px."""
    o = ox % period
    _paste(c, layer, o, oy)
    _paste(c, layer, o - period, oy)


@lru_cache(maxsize=None)
def _fore_rich():
    """The film's foreground knoll (k 1.4) with its flat inside textured: grass speckle, tufts, a few
    flowers and a darker dithered depth. The film never shows its lower part; the deep tilt here does."""
    rgb, m = sc._rf_fore()
    rgb = rgb.copy()
    flat = m & np.all(rgb == np.array(col(mix('green', 'forest', 0.5)), np.uint8), axis=2)
    h, w = m.shape
    top = np.where(m.any(axis=0), m.argmax(axis=0), h)
    depth = np.arange(h)[:, None] - top[None, :]
    n = np.random.default_rng(7).random((h, w))
    thr = bayer(h, w)
    deep = flat & (depth > 26) & (thr < np.clip((depth - 26) / 60, 0, 0.5))
    rgb[deep] = col(mix('forest', 'green', 0.2))
    rgb[flat & (n < 0.05)] = col('green')
    rgb[flat & (n >= 0.05) & (n < 0.085)] = col('forest')
    tuft = flat & (n > 0.988) & (depth > 4)
    ys, xs = np.nonzero(tuft)
    for y, x in zip(ys, xs):
        rgb[y - 2:y + 1, x] = col('leaf' if (x + y) % 2 else 'green')
        if x + 1 < w:
            rgb[y - 1:y + 1, x + 1] = col('green')
    fl = flat & (n > 0.9975) & (depth > 8)
    ys, xs = np.nonzero(fl)
    for k, (y, x) in enumerate(zip(ys, xs)):
        cc = col(('hot_pink', 'gold', 'white', 'lavender')[k % 4])
        rgb[y, max(0, x - 1):x + 2] = cc
        rgb[y - 1:y + 2, x] = cc
        rgb[y, x] = col('butter')
    return rgb, m


def _train_front(tc):
    """Loco front (trestle-layer x) at run step tc."""
    return TRAIN_START - tc * sc.TRAIN_SPEED


def _draw_train(c, tc, ox, oy):
    """The film's draw_train, driven by the run step (loco front from _train_front)."""
    front = _train_front(tc)
    Xs = int(round(front)) - ox
    if Xs > W + 4 or Xs + sc.TRAIN_LEN < -4:
        return None
    img = sc._train_img(int(front // 2) % 8)
    _paste(c, img, -(Xs - 3), -((sc.RAIL_Y - 1 - oy) - (sc._TH - 2)))
    return Xs


def _draw_steam(c, tc, ox, oy):
    """The film's draw_steam on the run clock: chuffs on 8ths from the funnel (left hanging in the air)
    for the whole crossing, and the whistle jet + big burst at the trestle HIT."""
    for k in range(6):
        e = (tc // 9 - k) * 9
        a = tc - e
        if a < 0 or a > 46 or e < 0 or e > TRAIN_EMIT:
            continue
        ex = int(round(_train_front(e))) + 11 - ox
        ey = sc.RAIL_Y - 28 - oy
        px_ = ex + int(a * 0.45)
        py = ey - int(a * 0.6)
        r = 2 + min(a, 32) * 0.14
        if a > 32:
            r *= 1 - (a - 32) / 15
        sc._puff(c, px_, py, r, a / 46)
        if a > 6 and r > 2:
            sc._puff(c, px_ + int(r), py + 2, r * 0.6, a / 46)
    _whistle_fx(c, tc - TRAIN_HIT, int(round(_train_front(tc))) - ox, oy)


def _whistle_fx(c, a, X, oy):
    """The film's trestle HIT at the loco (screen x X): whistle jet + a big steam burst, a in 0..33."""
    if not 0 <= a < 34:
        return
    wx, wy = X + 22, sc.RAIL_Y - 20 - oy
    for k in range(7):
        rr = (1.5 + k * 0.45 + a * 0.12) * (1 - max(0, a - 22) / 14)
        sc._puff(c, wx + k * 3 + a // 2 + (k % 2) * 2, wy - 4 - k * 4 - a // 2 + (k % 3), rr, a / 34)
    if 0 <= a < 6 or 9 <= a < 26:
        c.vline(wx, wy - 5, wy - 1, 'white')
        c.vline(wx + 1, wy - 8, wy - 4, mix('white', 'cloud', 0.4))
        c.px(wx, wy - 9, 'white')


# ------------------------------------------------------------------ SPACE events (area 0)
def _course_order():
    """The hoop course's dots in flying order (the film lists the lead-in before the first hoop) and
    the dot index at each hoop."""
    pts = sc.COURSE
    order = [pts[7]] + list(pts[:7]) + list(pts[8:])
    at = [int(np.argmin([(x - rx) ** 2 + (y - ry) ** 2 for x, y in order])) for (rx, ry, _) in sc.RINGS]
    return order, at


ORDER, RING_AT = _course_order()
HOOP_SPEED = 1.3            # course dots per step for the light running along the line
HOOP_END = int(len(ORDER) / HOOP_SPEED)


def _blit_mask(c, x, y, mask, rgb):
    """Copy rgb[mask] to the canvas with its top-left at screen (x, y), clipped."""
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    np.copyto(c.a[y0:y1, x0:x1], rgb[y0 - y:y1 - y, x0 - x:x1 - x], where=m[:, :, None])


@lru_cache(maxsize=None)
def _arc_art():
    """Rainbow arch from fall 2's lip to fall 4's lip, over the main fall: (x0, y0, band, angle, rgb)."""
    cx, base, rx, ry = ARC
    x0, y0 = cx - rx - 7, base - ry - 7
    yy, xx = np.mgrid[0:ry + 8, 0:2 * rx + 15]
    dx = (xx + x0 - cx).astype(float)
    dy = (base - (yy + y0)) * rx / ry
    d = np.sqrt(dx * dx + dy * dy)
    band = (d >= rx - 6) & (d < rx + 6) & (dy >= 0)
    stripe = np.clip(((rx + 6 - d) // 2).astype(int), 0, 5)
    lev = np.where(((rx + 6 - d) % 2) < 1, 2, 1)                  # two px per stripe: light over base
    rgb = sc.RB_RAMP[stripe, lev]
    rgb[band & (d >= rx + 5)] = col('white')                       # a glassy rim on top
    return x0, y0, band, np.arctan2(dy, dx), rgb


def _draw_arc(c, a, ox, oy):
    """The rainbow arch grows from left to right (a 4..24), holds, then dissolves (a 56..80)."""
    if not 4 <= a < 80:
        return
    x0, y0, band, ang, rgb = _arc_art()
    g = clamp((a - 4) / 20)
    show = band & (ang >= math.pi * (1 - ease_in_out(g)) - 0.02)
    fade = clamp((a - 56) / 24)
    if fade > 0:
        show = show & (bayer(*band.shape, x0, y0) >= fade)
    _blit_mask(c, x0 - ox, y0 - oy, show, rgb)
    if a < 60:                                                     # glints running along the arch
        cx, base, rx, ry = ARC
        for k in range(5):
            th = math.pi * (1 - ((a * 0.03 + k * 0.2) % 1.0) * g)
            fx.sparkle(c, int(cx + math.cos(th) * rx) - ox, int(base - math.sin(th) * ry) - oy,
                       a + k * 5, 'white', period=10, size=2)


def _draw_surge(c, f, a, ox, oy):
    """A surge of extra water runs down the main fall (a 0..26): the column swells 2 px on each side
    behind a frothy white crest; then the fall keeps glittering."""
    x0, w, spd = sc.MAIN_FALL
    front = sc.MESA_TOP + a * 6
    lv = sc._fall_tex(w, 52)
    if a < 28:
        for wy in range(max(sc.MESA_TOP, front - 34), min(sc.MESA_BOT + 1, front + 1)):
            yy = wy - oy
            if not 0 <= yy < H:
                continue
            age = front - wy
            row = lv[(wy - f * spd) % lv.shape[0]]
            if age < 7:                                            # the frothy crest
                e = 4 - age // 2
                for xx in range(x0 - e, x0 + w + e):
                    if hash01(xx, wy, a) < 0.85 - age * 0.06:
                        c.px(xx - ox, yy, 'white' if (xx + wy) % 3 else 'cloud')
            else:                                                  # the swollen column
                k = 2 if age < 26 else 1
                c.hline(x0 - k - ox, x0 - 1 - ox, yy, tuple(sc.RB_RAMP[0, int(row[1])]))
                c.hline(x0 + w - ox, x0 + w + k - 1 - ox, yy, tuple(sc.RB_RAMP[5, int(row[-2])]))
                if hash01(wy, a, 3) < 0.3:
                    c.px(x0 - k - 1 - ox if wy % 2 else x0 + w + k - ox, yy, 'white')
    if 10 <= a < 70:
        for k in range(8):
            y = sc.MESA_TOP + 8 + int(hash01(k, 51) * 124)
            fx.sparkle(c, x0 + int(hash01(k, 52) * w) - ox, y - oy, a + k * 4, 'white', period=16, size=2)


def _draw_geyser(c, a, ox, oy):
    """The main fall's lip shoots up a rainbow geyser that rains back down as coloured droplets."""
    if not 0 <= a < 62:
        return
    x, y = LIP[0] - ox, LIP[1] - oy
    if a < 26:                                                     # the jet: six 2-px stripes + foam cap
        h = int(52 * math.sin(math.pi * a / 26))
        for j in range(12):
            name = RAINBOW[j // 2]
            c.vline(x - 6 + j, y - h, y, name if j % 2 else mix(name, 'white', 0.35))
        if h > 4:
            for k, (dx, dy, r) in enumerate(((0, 0, 5), (-5, 3, 3), (5, 2, 3), (-2, -3, 3), (3, -4, 2))):
                sc._puff(c, x + dx, y - h + dy, r + (a // 6) % 2 * (k == 0))
    for i in range(52):
        ang = -math.pi / 2 + (hash01(i, 91) - 0.5) * 1.5
        v = 1.5 + 1.7 * hash01(i, 92)
        b = a - int(hash01(i, 93) * 16)
        if b < 0 or b > 46:
            continue
        px = x + math.cos(ang) * v * b
        py = y + math.sin(ang) * v * b + 0.085 * b * b
        if py > y + 44:
            continue
        sz = 3 if b < 12 and i % 3 == 0 else (2 if b < 32 else 1)
        c.rect(int(px), int(py), sz, sz, RAINBOW[i % 6] if b > 3 else 'white')


@lru_cache(maxsize=None)
def _lit_ring(ang10, hot):
    """The film's hoop sprites (back, front) recoloured gold (lit) or white-gold (flashing)."""
    ramp = np.array([col(n) for n in (('gold', 'butter', 'white', 'white') if hot else
                                      ('amber', 'gold', 'butter', 'white'))], np.uint8)
    out = []
    for spr in sc._ring_sprites(ang10)[:2]:
        g = spr.copy()
        m = g[..., 3] > 0
        lum = g[..., :3].astype(int).sum(axis=2)
        lo, hi = lum[m].min(), lum[m].max()
        idx = np.clip(((lum - lo) * 4) // max(1, hi - lo + 1), 0, 3)
        g[m, :3] = ramp[idx[m]]
        out.append(g)
    return tuple(out)


def _draw_hoops(c, f, a, ox, oy):
    """The hoop course lights up ring by ring: a glowing comet runs along the dotted line, every hoop
    flashes white-gold as it passes and stays gold; confetti at the end of the course."""
    if not 0 <= a < 110:
        return
    n = len(ORDER)
    head = a * HOOP_SPEED
    fade = clamp((a - 84) / 26)
    for j in range(min(n, int(head))):                             # the lit line
        if fade > 0 and hash01(j, 77) < fade:
            continue
        x, y = ORDER[j][0] - ox, ORDER[j][1] - oy
        c.rect(x, y, 2, 2, 'gold')
        c.px(x, y, 'white' if (j + a // 3) % 4 == 0 else 'butter')
    for i in range(len(sc.RINGS)):
        ai = int(RING_AT[i] / HOOP_SPEED)
        b = a - ai
        x, y, ang = sc.ring_pos(i, f)
        x, y = x - ox, y - oy
        if b < 0 or fade >= 1 or not (-30 < x < W + 30 and -30 < y < H + 30):
            continue
        if fade > 0 and hash01(i, a // 3) < fade:                  # the lights flicker out
            continue
        back, front = _lit_ring(int(round(ang / 5.0)) * 50, b < 8)
        S = back.shape[0]
        c.blit(back, x - S // 2, y - S // 2)
        c.blit(front, x - S // 2, y - S // 2)
        if b < 16:
            fx.glow(c, x, y, 20, 'gold', 0.6 * (1 - b / 16), rings=4)
            ui.burst(c, a, ai, x, y, colors=('white', 'gold', 'butter', 'hot_pink'), n=14, seed=40 + i,
                     power=0.75, life=26, gravity=0.03)
    if head < n + 4:                                               # the comet
        for k, cc in enumerate(('white', 'white', 'butter', 'gold', 'gold', 'hot_pink')):
            j = int(head) - k
            if 0 <= j < n:
                x, y = ORDER[j][0] - ox, ORDER[j][1] - oy
                sz = 3 if k == 0 else (2 if k < 4 else 1)
                c.rect(x - sz // 2, y - sz // 2, sz, sz, cc)
                if k == 0:
                    fx.glow(c, x, y, 8, 'white', 0.6, rings=3)
    x, y = ORDER[-1]
    fx.confetti(c, a, HOOP_END, x - ox, y - oy, n=30, seed=9)
    if 0 <= a - HOOP_END < 18:
        fx.glow(c, x - ox, y - oy, 18, 'white', 0.7 * (1 - (a - HOOP_END) / 18), rings=4)


# ------------------------------------------------------------------ SPACE events (area 1)
CHIMNEYS = [(x + 11, b - 18) for (x, b) in sc.OT_BACK] + [(x + 13, b - 20) for (x, b) in sc.OT_FRONT]


def _draw_gust(c, f, a, ox):
    """A gust through the '=' banner: wind streaks, a dust swirl at the pole, '=' confetti, and every
    identical house puffs one identical smoke ring at the same moment."""
    if not 0 <= a < 72:
        return
    px_, base = sc.OT_POLE
    if a < 40:
        for k in range(6):
            x = px_ - 50 + a * 8 + k * 11 - ox
            y = base - 52 + k * 7 + (k % 2)
            c.hline(x, x + 12 - k % 3 * 2, y, 'white' if k % 2 else mix('white', 'light_sky', 0.4))
            c.hline(x + 4, x + 9, y - 1, mix('white', 'light_sky', 0.6))
    if a < 50:
        sc._ot_dust_devil(c, f, px_ + 6 + int(a * 0.7) - ox, base, 16, 11)
    b = a - 10
    if 0 <= b < 40:                                                # all chimneys at once
        for (x, y) in CHIMNEYS:
            r = 1.5 + min(b, 26) * 0.1 - max(0, b - 26) * 0.28
            sc._puff(c, x - ox + int(b * 0.35), y - 2 - int(b * 0.6), r, b / 40)
    bx, by = BANNER[0] - ox, BANNER[1]
    ui.burst(c, a, 0, bx, by, colors=('royal', 'cream', 'gold', 'royal_lt', 'white'), n=30, seed=61,
             power=0.9, life=56, gravity=0.04)
    for k in range(6):                                             # '=' tiles flying off downwind
        b = a - k * 5
        if not 0 <= b < 60:
            continue
        x = bx + 6 + int(b * (1.0 + 0.25 * k) + math.sin(b * 0.25 + k) * 3)
        y = by - 6 + int((k % 3 - 1) * 6) - int(b * (0.35 + 0.08 * k)) + int(3 * math.sin(b * 0.3 + k))
        c.rect(x - 3, y - 2, 7, 6, 'cream')
        c.hline(x - 2, x + 2, y - 1, 'royal')
        c.hline(x - 2, x + 2, y + 1, 'royal')
        c.hline(x - 3, x + 3, y + 3, mix('cream', 'tan', 0.4))


@lru_cache(maxsize=None)
def _blossom_spots():
    """Points spread over the big tree's canopy puffs (offsets from its centre)."""
    puffs = ((0, 0, 14), (-12, 4, 9), (12, 3, 10), (-6, -9, 10), (7, -10, 9), (0, -16, 8), (-15, -3, 7),
             (16, -4, 7))
    pts = []
    for k in range(36):
        dx, dy, r = puffs[int(hash01(k, 1) * len(puffs))]
        ang, rr = hash01(k, 2) * 6.283, r * 0.85 * math.sqrt(hash01(k, 3))
        pts.append((dx + int(math.cos(ang) * rr), dy + int(math.sin(ang) * rr)))
    return pts


def _draw_blossoms(c, a, ox):
    """The big tree shakes and bursts into pink blossom; the petals then drift down the street."""
    if not 0 <= a < 100:
        return
    x, y = CANOPY[0] - ox, CANOPY[1]
    if a < 14:
        fx.glow(c, x, y, 22, 'cotton', 0.45 * (1 - a / 14), rings=3)
    for k, (dx, dy) in enumerate(_blossom_spots()):
        on, off = 2 + int(hash01(k, 7) * 12), 44 + int(hash01(k, 8) * 26)
        if a < on:
            continue
        cc = ('hot_pink', 'cotton', 'bubblegum', 'white')[k % 4]
        if a < off:                                                # open on the tree
            px, py = x + dx, y + dy
            if a - on < 3:
                c.rect(px, py, 2, 2, cc)
            else:
                big = k % 3 == 0
                c.rect(px - 1, py - 1, 3, 3, cc)
                if big:
                    c.hline(px - 2, px + 2, py, cc)
                    c.vline(px, py - 2, py + 2, cc)
                c.px(px, py, 'gold')
                c.px(px - 1, py - 1, mix(cc, 'white', 0.5))
        else:                                                      # a petal drifting east and down
            d = a - off
            px = x + dx + int(d * 0.7 + math.sin(d * 0.2 + k) * 3)
            py = y + dy + int(d * 0.55)
            if py < sc.OT_TREE[1] + 12:
                c.rect(px, py, 2, 1 + (d // 4 + k) % 2, cc)
    ui.burst(c, a, 0, x, y - 4, colors=('hot_pink', 'cotton', 'white', 'gold'), n=24, seed=71, power=0.75,
             life=36, gravity=0.03)


OT_HAZE_Y = 85                 # town view: the far mesas stand on y 84 (sc._ot_far)
OT_TOWN_RGBA = nightwin.from_mask(sc._ot_town)
GLOW_NIGHT = 0.2               # daylight.night() above which the rainbow falls glow by themselves

# ------------------------------------------------------------------ the lights at night
# The film paints the village at noon: its windows are daytime glass and nothing is lit. At dusk and night
# (the world clock) the real lights go on, drawn at the end of the area render and kept shining through the
# re-light by GlowKeep (game/regions/everfree.py): the village windows and attic windows, the judges' tower
# window, the cliff dwellings' lanterns and round windows, the clifftop lamp posts; in Our Town the houses'
# windows (game/nightwin.py). auto_glow is off: the re-light's own warm / vivid spot test would light the
# painted pennants, bunting, the striped poles, the tree's dots and the signs instead.
LAMP_POSTS = (164, 244, 314, 384, 436)                # sc._rf_main: gold lamp heads at (x-1..x+1, 131..133)
CLIFF_HOMES = ((236, 184), (322, 206), (402, 178), (244, 240), (414, 232), (112, 214), (460, 200))
LAMP_ON = ('#ffe9a0', '#fff6d8')                       # a lit lamp head: glass, its bright core
LANTERN_ON, ROUND_ON = '#ffd466', '#ffc455'


def lamp_ramp(hour):
    """0..1: the lamps and lanterns, on a little before the windows (from ~18:50) and off after them."""
    return clamp((daylight.night(hour) - 0.04) / 0.35)


@lru_cache(maxsize=1)
def _rf_lights():
    """The village's light spots in main-layer px: (ys, xs, colour index, threshold, lamp?). Colours:
    0..2 nightwin.WARM (windows), 3 LANTERN_ON, 4 ROUND_ON, 5 / 6 LAMP_ON. Windows have their own threshold
    (a share of them lit, as nightwin), lamps go on together with lamp_ramp."""
    pts = []                                         # (x, y, colour, threshold, lamp)

    def window(x0, y0, w, rows):
        v = hash01(x0, y0, 406) / nightwin.SHARE
        kd = int(hash01(x0, y0, 405) * 3)
        for yy in rows:
            for xx in range(x0, x0 + w):
                pts.append((xx, yy, kd, v, False))
    for (wx, wy) in sc._village_windows():           # 4 x 5 glass, a cream mullion across row 2
        window(wx, wy, 4, (wy, wy + 1, wy + 3, wy + 4))
    for (x, w, wh, rh, wc, rc, kw) in sc.HOUSES:     # the attic round windows (gable roofs)
        if kw.get('gable', True) and kw.get('round_win', True) and rh >= 9:
            cxw, cyw = x + w // 2, 146 - wh - rh // 2 + 1
            v = hash01(cxw, cyw, 406) / nightwin.SHARE
            pts += [(cxw, cyw, 4, v, False), (cxw - 1, cyw, 4, v, False), (cxw, cyw - 1, 4, v, False)]
    tx0 = 202                                        # the judges' tower cabin window (a mullion at +6)
    window(tx0 + 1, 100, 5, range(100, 104))
    window(tx0 + 7, 100, 4, range(100, 104))
    for k, (x, y) in enumerate(CLIFF_HOMES):        # dwellings: the balcony lantern, the round window
        pts.append((x + 16, y - 11, 3, 0.3 * hash01(k, 7), True))
        pts.append((x + 12, y - 7, 4, hash01(x, y, 406) / nightwin.SHARE, False))
    for k, lx in enumerate(LAMP_POSTS):
        for dy in range(3):
            for dx in (-1, 0, 1):
                pts.append((lx + dx, 131 + dy, 6 if (dx, dy) == (0, 1) else 5, 0.3 * hash01(k, 8), True))
    xs, ys, kd, v, lamp = (np.array(a) for a in zip(*pts))
    return ys.astype(int), xs.astype(int), kd.astype(int), v.astype(np.float32), lamp.astype(bool)


_RF_COLS = np.array([*nightwin.WARM, col(LANTERN_ON), col(ROUND_ON), col(LAMP_ON[0]), col(LAMP_ON[1])], np.uint8)


class RainbowFalls(GlowKeep, Region):
    SEG = 'rainbowfalls'
    CARD = ('CLIFFTOP VILLAGE', ['RAINBOW WATERFALLS', 'GAMES TRIAL GROUNDS'])

    def build_areas(self):
        return [
            Area(self.render_falls, cam_x=RF_CAM_X, cam_y=RF_CAM_Y, y_range=(44, 282), start=(64, 206),
                 landmarks=RF_LANDMARKS, hotspots=RF_HOTSPOTS, stamps=RF_STAMPS, balloon='m',
                 sky=self.sky_falls),
            Area(self.render_town, cam_x=OT_CAM_X, cam_y=(0, 0), y_range=(24, 132), start=(60, 64),
                 landmarks=OT_LANDMARKS, hotspots=OT_HOTSPOTS, stamps=OT_STAMPS, balloon='s',
                 sky=self.sky_town),
        ]

    def auto_glow(self):
        return False                # the real lights are drawn and kept by the region (see _rf_lights)

    def _village_lights(self, c, cx, cy):
        """The end of render_falls: light the village's windows, lanterns and lamps for the clock's hour where
        nothing is drawn in front of them, and keep them shining through the re-light (GlowKeep)."""
        hour = self.g.hour
        kw, kl = nightwin.ramp(hour), lamp_ramp(hour)
        if kl <= 0:
            return
        ys, xs, kd, v, lamp = _rf_lights()
        on = v < np.where(lamp, kl, kw)
        sy, sx = ys - cy, xs - cx
        on &= (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        ys, xs, sy, sx, kd = ys[on], xs[on], sy[on], sx[on], kd[on]
        rgb, _ = sc._rf_main()
        seen = np.all(c.a[sy, sx] == rgb[ys, xs], axis=1)     # nothing in front (flags, smoke, rings...)
        if not seen.any():
            return
        sy, sx = sy[seen], sx[seen]
        c.a[sy, sx] = _RF_COLS[kd[seen]]
        m = np.zeros((H, W), bool)
        m[sy, sx] = True
        self.keep_glow(c, m, 0.95)

    def init_state(self):
        self.train_t0 = None        # step the train runs are counted from
        self.train_old = None       # (step, run clock) of a run replaced by a called train: runs out of sight
        self.whistle_t0 = None      # step of a whistle the player asked for

    def on_enter(self):
        self.train_t0 = self.g.t + 20      # the balloon lands right above the trestle: here it comes
        self.train_old = self.whistle_t0 = None

    def warm(self):
        super().warm()
        _fore_rich()
        _arc_art()                              # the SPACE events' own art, so a first press never stalls
        _blossom_spots()
        for (_, _, ang) in sc.RINGS:
            for hot in (False, True):
                _lit_ring(int(round(ang / 5.0)) * 50, hot)

    def train_clock(self, t):
        if self.train_t0 is None or t < self.train_t0:
            return None
        return (t - self.train_t0) % TRAIN_PERIOD

    def old_clock(self, t):
        if self.train_old is None:
            return None
        tc = self.train_old[1] + t - self.train_old[0]
        return tc if tc <= TRAIN_EMIT + 50 else None

    def _call_train(self, t):
        """SPACE at the trestle: the loco whistles; if it isn't on screen, a train comes in from the
        right edge first (a train already out west keeps running out of sight)."""
        tox = int(round(sc.REF[0] + (self.cx - sc.REF[0]) * sc.TRK))      # the camera on the trestle layer
        tc = self.train_clock(t)
        X = _train_front(tc) - tox if tc is not None and tc <= TRAIN_EMIT else None
        if X is not None and -4 <= X <= 280:
            self.whistle_t0 = t
            return
        if X is not None and 280 < X <= W + 4:
            self.whistle_t0 = t + int(math.ceil((X - 280) / sc.TRAIN_SPEED))
            return
        if X is not None and X < -4:
            self.train_old = (t, tc)
        tc_new = int((TRAIN_START - tox - (W + 4)) / sc.TRAIN_SPEED)
        self.train_t0 = t - tc_new
        X = _train_front(tc_new) - tox
        self.whistle_t0 = t + int(math.ceil((X - 280) / sc.TRAIN_SPEED))
        self.g.sfx('chug', 0.6)

    def on_interact(self, key, t):
        g = self.g
        if key == 'train':
            self._call_train(t)
            if self.whistle_t0 == t:
                g.sfx('whistle', 0.85)
        elif key == 'falls':
            g.sfx('splash', 0.6)
            g.sfx('shimmer', 0.45)
        elif key == 'hoops':
            g.sfx('star', 0.5)
        elif key == 'banner':
            g.sfx('curtain', 0.55)
            g.sfx('wind', 0.35)
        elif key == 'tree':
            g.sfx('bag', 0.5)
            g.sfx('wind', 0.25)

    def update(self, t):
        g = self.g
        if self.ai == 0:
            tc = self.train_clock(t)
            if tc is not None and tc <= TRAIN_EMIT:
                fx_ = _train_front(tc) + 60 + 16.5 - 0.15 * self.cx     # train middle, world x
                low = min(1.0, max(0.25, (self.cy - 40) / 90))
                gain = self.near_gain(fx_) * low
                if tc % 36 == 0:
                    g.sfx('chug', 0.5 * gain)
                if tc == TRAIN_HIT and (self.whistle_t0 is None or abs(t - self.whistle_t0) > 45):
                    g.sfx('whistle', 0.85 * max(0.35, gain))
            if self.whistle_t0 is not None and t == self.whistle_t0 and t != self.spot_t.get('train'):
                g.sfx('whistle', 0.85)
            a = t - self.spot_t.get('hoops', -999)
            if 0 < a <= HOOP_END + 2:
                for i, j in enumerate(RING_AT):
                    if a == int(j / HOOP_SPEED):
                        g.sfx('pop', 0.3, i % 4)
                if a == HOOP_END:
                    g.sfx('sparkle', 0.5)
            if t - self.spot_t.get('falls', -999) == 24:
                g.sfx('splash', 0.35 * self.near_gain(MAIN_X))
            if (t - self.t_enter) % 60 == 6:
                g.sfx('waterfall', 0.28 + 0.2 * self.near_gain(MAIN_X))
        else:
            if t % 240 == 120:
                g.sfx('creak', 0.2 * self.near_gain(sc.OT_POLE[0]))      # the banner pole
            elif t % 240 == 0:
                g.sfx('wind', 0.22)                                      # dusty valley gusts
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)

    # -------------------------------------------------------------- the skies alone
    # as the renders draw them first (game/daylight.py repaints them for the world clock's hour)
    def sky_falls(self, f, cx, cy):
        c = Canvas()
        sky = sc._sky_tall()
        oy = max(0, min(sky.shape[0] - H, sc.off((cx, cy), 0.2)[1] - 80))
        c.a[:] = sky[oy:oy + H]
        return c

    def sky_town(self, f, cx, cy):
        c = Canvas(color=(255, 0, 255))                 # below the sky band: never matches the ground
        c.a[:100] = sc._ot_sky()
        # the pale far mesas are near the horizon's colour: stamp the scenery out (as render_town places it)
        for layer, x in ((sc._ot_far(), int(round(cx * 0.25)) + 10), (sc._ot_mid(), int(round(cx * 0.5)) + 20),
                         (sc._ot_vault(), int(round(cx * 0.85)) + 30), (sc._ot_town(), cx)):
            _stamp(c, layer, x, 0)
        c.a[OT_HAZE_Y:] = (255, 0, 255)                 # below the far mesas' feet: valley haze, not sky
        return c

    # -------------------------------------------------------------- area 0
    def render_falls(self, f, cx, cy):
        cam = (cx, cy)
        off = sc.off
        c = Canvas()
        sky = sc._sky_tall()
        oy = max(0, min(sky.shape[0] - H, off(cam, 0.2)[1] - 80))
        c.a[:] = sky[oy:oy + H]
        ox, oy = off(cam, 0.12)
        _paste_wrap(c, sc._rf_clouds(), ox - int(f * 0.15) + 40, oy, 800)
        _paste(c, sc._rf_peaks(), *off(cam, 0.3))
        _paste(c, sc._rf_far(), *off(cam, 0.55))
        _paste(c, sc._rf_mid(), *off(cam, 0.75))
        ox, oy = cx, cy
        _paste(c, sc._rf_main(), ox, oy)
        t = self.g.t
        a_falls = t - self.spot_t.get('falls', -999)
        glow = daylight.night(self.g.hour) > GLOW_NIGHT
        if glow:
            b0 = c.a.copy()
        sc.draw_river(c, f, ox, oy)
        sc.draw_falls(c, f, ox, oy)
        if 0 <= a_falls < 70:
            _draw_surge(c, f, a_falls, ox, oy)
        if glow:                                     # the rainbow water, as drawn (see _world)
            snap = c.a.copy()
            gm = np.any(snap != b0, axis=2)
        sc.draw_fall_mist(c, f, ox, oy)
        sc.draw_village_life(c, f, ox, oy)
        if daylight.night(self.g.hour) > 0.05:      # no sun glinting on the glass at dusk and night
            self._no_glints(c, f, ox, oy)
        _draw_arc(c, a_falls, ox, oy)
        sc.draw_course(c, f, ox, oy)
        sc.draw_rings(c, f, ox, oy, 0)
        sc.draw_rings(c, f, ox, oy, 1)          # (the film's balloon flies between the two halves)
        _draw_hoops(c, f, t - self.spot_t.get('hoops', -999), ox, oy)
        _draw_geyser(c, a_falls, ox, oy)
        if 22 <= a_falls < 64:                  # the surge lands in the plunge pool
            ui.burst(c, a_falls, 22, MAIN_X - ox, sc.MESA_BOT - oy, colors=('white', 'cloud', 'light_sky') +
                     tuple(sc.RB_PASTEL[::2]), n=30, seed=33, power=1.0, life=40, gravity=0.06)
        tox, toy = off(cam, sc.TRK)
        _paste(c, sc._rf_trestle(), tox, toy)
        for tc in (self.old_clock(t), self.train_clock(t)):
            if tc is not None:
                _draw_train(c, tc, tox, toy)
                _draw_steam(c, tc, tox, toy)
        tc = self.train_clock(t)
        if tc is not None and self.whistle_t0 is not None:
            _whistle_fx(c, t - self.whistle_t0, int(round(_train_front(tc))) - tox, toy)
        pt = self.spot_t.get('train')
        if pt is not None and self.whistle_t0 is not None and self.whistle_t0 > pt and 0 <= t - pt < self.whistle_t0 - pt + 4:
            a, y = t - pt, sc.RAIL_Y - 1 - toy                     # a called train: the rails start to sing
            for k in range(3):
                x = W - 4 - a * 9 - k * 37
                c.hline(x, x + 6, y, 'white')
                c.px(x + 3, y - 1, 'butter')
        _paste(c, _fore_rich(), *off(cam, sc.FGK))
        self._falls_glow = ((f, cx, cy), gm & np.all(c.a == snap, axis=2), c.a.copy()) if glow else None
        self._village_lights(c, cx, cy)
        return c

    @staticmethod
    def _no_glints(c, f, ox, oy):
        """Undo sc.draw_village_life's window glints (white pixels where the sun catches the glass)."""
        rgb, _ = sc._rf_main()
        white = np.array((255, 255, 255), np.uint8)
        for i, (x, y) in enumerate(sc._village_windows()):
            if (f + i * 13) % 61 < 3:
                for (px, py) in ((x, y), (x + 1, y + 1)):
                    sx, sy = px - ox, py - oy
                    if 0 <= sx < W and 0 <= sy < H and np.array_equal(c.a[sy, sx], white):
                        c.a[sy, sx] = rgb[py, px]

    def _world(self, t, cam=None, with_balloon=False):
        """The world draw, then the rainbow river and falls glow again at night: they shine by their own
        magic, but daylight.relight only keeps small bright lights (the falls are broad bands). Every
        falls pixel the overlays left alone (still exactly as the clock's relight made it) goes back most
        of the way to its painted colour."""
        c = super()._world(t, cam, with_balloon)
        memo = getattr(self, '_falls_glow', None)
        if memo is None or self.ai != 0:
            return c
        cx, cy = (self.cx, self.cy) if cam is None else cam
        if memo[0] != (self.scene_f(t), int(math.floor(cx)), int(math.floor(cy))):
            return c
        _, m, raw = memo
        relit = raw.copy()
        daylight.relight(relit, self.HOUR, self.g.hour, t, glow=self.auto_glow())
        m = m & np.all(c.a == relit, axis=2)
        if m.any():
            k = min(1.0, (daylight.night(self.g.hour) - GLOW_NIGHT) / 0.4) * 0.8
            lo, hi = relit[m].astype(np.float32), raw[m].astype(np.float32)
            c.a[m] = (lo + (hi - lo) * k).astype(np.uint8)
        return c

    def front(self, f, cx, cy):
        """The near half of every cloud hoop (and its pennant) goes in front of the balloon, so it flies
        THROUGH the hoops instead of over them (lit gold while the course is lit)."""
        if self.ai != 0:
            return None
        from ..compose import KEY
        c = Canvas(color=KEY)
        sc.draw_rings(c, f, cx, cy, 1)
        a = self.g.t - self.spot_t.get('hoops', -999)
        if 0 <= a < 84:
            for i in range(len(sc.RINGS)):
                if a >= int(RING_AT[i] / HOOP_SPEED):
                    x, y, ang = sc.ring_pos(i, f)
                    front = _lit_ring(int(round(ang / 5.0)) * 50, a - int(RING_AT[i] / HOOP_SPEED) < 8)[1]
                    S = front.shape[0]
                    c.blit(front, x - cx - S // 2, y - cy - S // 2)
        return c

    # -------------------------------------------------------------- area 1
    def render_town(self, f, cx, cy):
        c = Canvas()
        c.a[:100] = sc._ot_sky()
        _paste_wrap(c, sc._ot_clouds(), int(round(cx * 0.1)) - int(f * 0.2) + 60, 0, 620)
        _paste(c, sc._ot_far(), int(round(cx * 0.25)) + 10, 0)
        _paste(c, sc._ot_mid(), int(round(cx * 0.5)) + 20, 0)
        _paste(c, sc._ot_vault(), int(round(cx * 0.85)) + 30, 0)
        ox = cx
        _paste(c, sc._ot_town(), ox, 0)
        sc._ot_cloud_shadows(c, f, ox)
        # dust devils wander the valley floor and wrap around off screen
        sc._ot_dust_devil(c, f, -30 + int(f * 0.45 + 180) % 460 - ox, 128, 14, 3)
        sc._ot_dust_devil(c, f, 430 - int(f * 0.3 + 94) % 460 - ox, 100, 10, 7)
        for (x, b) in sc.OT_BACK[1::3]:
            steam(c, x + 11 - ox, b - 18, f, color=mix('white', 'cloud', 0.3), n=3, height=12, spread=2, seed=x,
                  speed=0.35)
        for (x, b) in sc.OT_FRONT[2::3]:
            steam(c, x + 13 - ox, b - 20, f, color=mix('white', 'cloud', 0.3), n=3, height=12, spread=2, seed=x,
                  speed=0.35)
        t = self.g.t
        a_tree = t - self.spot_t.get('tree', -999)
        shake = (1 if (a_tree // 2) % 2 else -1) if 0 <= a_tree < 20 else 0
        _tree_canopy(c, f, sc.OT_TREE[0] - ox + shake, sc.OT_TREE[1] - 34)
        a_gust = t - self.spot_t.get('banner', -999)
        fb = f
        if 0 <= a_gust < 70:                    # flap ~2x faster, ending exactly two waves ahead
            u = a_gust / 70
            fb = f + int(round(57 * (3 * u * u - 2 * u ** 3)))
        sc._ot_banner(c, fb, sc.OT_POLE[0] - ox, sc.OT_POLE[1])
        vx = 266 - int(round(cx * 0.85)) - 30
        sun = daylight.night(self.g.hour) <= 0.05            # the sun glints on the vault and the glass
        if sun:
            fx.sparkle(c, vx - 3, 80, f, 'white', period=30, phase=0.2, size=2)
        for i, (x, b) in enumerate(sc.OT_BACK + sc.OT_FRONT):
            if sun and (f + i * 7) % 40 < 2:
                c.px(x + 2 - ox, b - (7 if b < 120 else 8), 'white')
        _draw_gust(c, f, a_gust, ox)
        _draw_blossoms(c, a_tree, ox)
        fo = int(round(cx * 1.35)) + 10
        _paste(c, sc._ot_fore(), fo, 0)
        # the tumbleweed rolls along the hill and comes round again
        tw = -20 + int(f * 1.1) % 480 - fo
        ty = 162 - abs(int(math.sin(f * 0.5) * 2))
        c.circle(tw, ty, 3, mix('#a4876a', 'olive', 0.3))
        c.circle(tw, ty, 2, mix('khaki', 'tan', 0.4))
        for k in range(5):
            a = f * 0.5 + k * 1.3
            c.px(tw + int(round(math.cos(a) * 2)), ty + int(round(math.sin(a) * 2)), '#a4876a')
        w = daylight.sky_weight(self.HOUR, self.g.hour)
        if w > 0.01:
            # the noon sky-blue haze under the far mesas is valley haze, not sky (the clock's sky never repaints
            # it): away from noon it turns a grey haze that the re-light tints like the valley (else a noon-blue
            # band at sunset, a glowing blue one at night)
            band, (sky, haze) = c.a[OT_HAZE_Y:100], _ot_haze()
            m = np.all(band == sky, axis=2)
            if w < 0.99:
                haze = (sky * (1 - w) + haze * w).astype(np.uint8)
            np.copyto(band, haze, where=m[..., None])
        # at dusk and night the houses' windows light up (game/nightwin.py), kept through the re-light
        self.keep_glow(c, nightwin.paint(c, self.g.hour, self.g.t, [('ot_town', OT_TOWN_RGBA, -cx, 0)]), 0.95)
        return c
