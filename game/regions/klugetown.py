"""Klugetown as an explorable diorama (art: scenes/s13_klugetown.py, used read-only).

The film shows the desert trade town in three different views (zoom / glide cuts), so the region has
three areas, flown through left to right with a cloud sweep between them, in film order:
  0  BONE DRY DESERT and KLUGETOWN (shot A): hazy amber late afternoon, dunes, the bleached rock arch,
     the cobbled road and the town of stacked shacks on three rock pillars (crane, windmills, rope
     bridges, string lights, cargo lift, the tiny moored airship). The desert runs on east of the
     town (the film's layers rebuilt wider, more dunes, a second spire, rocks and cacti).
  1  THE BAZAAR at street level (shot B): awnings, lanterns, spice sacks, carpets, the junk pile with
     its cracked bell, the brazier, the creaky sign; a far pillar seen through the arch. The film's
     own builders run again with a longer street (more FACADES, STALLS and WALL_SIGNS in their lists);
     the film's bunting, drawn in screen space for its 48 px pan, is strung along the street instead.
  2  AIRSHIP DOCKS and the SEA OF CLOUDS (shot C): piers jutting from a pillar with the big patched
     airship moored; the brown-grey cloud sea with black peaks runs far out to the east (the film's
     peaks layer is rebuilt wider by its own generator, the cloud bands tile).
Each area is the film's own shot renderer (`render_a/b/c`) bound to replacement globals (`rebind`): the
player drives the camera; the film's balloon and name tags are switched off (the base class draws the
player's balloon and the tags; the tags on slower layers ride them with Landmark.p).

Timed film beats on game clocks:
  - the tumbleweed bounces across the desert every 12 s (near-layer world px, entering and leaving
    beyond every camera);
  - the AIRSHIP beat is the player's: SPACE at the moored airship casts it off (lines drop, steam
    burst, propellers spin up) and it sails east over the Sea of Clouds, past the end of the area.
    Out of sight it turns back; it comes down out of the sky to its berth by itself after a while
    (or at once on SPACE), and makes fast with a puff of steam.
Things to set off with SPACE: light up the town's windows and string lights (pillar 1), send a desert
gust from the east dunes (a dust devil crosses the view, sand streams, the windmills race, a tumbleweed
rolls), make the kettle on the lamp stall whistle, ring the junk pile's cracked bell, cast off / call
in the airship (a toot and a puff from its stack while it is under way).
Hidden stamps: high above the crane at the desert's east end, low between the lamp and junk stalls,
under the upper pier among its struts.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, fx, col, mix, shade, ease_out, ease_in_out, hash01, clamp, bayer
import scenes.s13_klugetown as sc

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# ------------------------------------------------------------------ area 0: desert + town (shot A)
# The film's layers end at camera 68 (near dunes 388 px, p 1). The desert runs on east of the town:
# the far / mid / near / fg layers are rebuilt wider by their own builders (so the film's dunes are no
# longer cut at the edge) and get more dunes, a second bleached spire, rocks and a cactus beyond, drawn
# with the scene's own dune / shade helpers. The mesas (.1) and the town (.42) end in open sky before
# their layer edges, so they need no more art.
CAM_A = 200
A_W = {k: W + int(math.ceil(CAM_A * p)) + 8 for k, p in sc.PAR_A.items()}
# colour ramps of the film's own layers (local to its builders in scenes/s13_klugetown.py)
FAR_DUNE = ('#dcab78', '#e3b783', '#e8c08c', '#f0cf9c', '#f5dcae', '#f9e8c4')
MID_DUNE = ('#c98d58', '#d69e66', '#e3b37c', '#efc78f', '#f7d9a6', '#fdebc4')
NEAR_DUNE = ('#b97a48', '#c98c56', '#d9a46c', '#e9bc82', '#f3d099', '#fbe4b8')
FG_DUNE = ('#9e6038', '#b27046', '#c58756', '#d89e68', '#e6b67c', '#f2cc96')
BONE = ['#8e7462', '#b39a80', '#cbb596', '#ddcaa9', '#ebdcbe', '#f6ecd6']
PEBBLE = ['#7e6452', '#a28a72', '#bfa98e', '#d6c4a6', '#e8dac0']
P_TOWN = sc.PAR_A['town']
P_MID = sc.PAR_A['mid']
TUMBLE_PERIOD = 360     # a tumbleweed every 12 s
TUMBLE_V = 4.5          # px / frame (the film's 308 px in 68 frames)
TUMBLE_X0, TUMBLE_X1 = -12, CAM_A + W + 12
LIGHTS_T = 120          # 'lights': the windows light up in a sweep and go out again
GUST_T = 110            # 'gust': a dust devil crosses, sand streams, the windmills race, a tumbleweed rolls
GUST_V = 3.6

# ------------------------------------------------------------------ area 1: the bazaar (shot B)
# the street is extended (film lists + more facades / stalls / signs); stall layer 610 (p 1) -> 286,
# walls 560 (.8) -> 300, fg (1.4) transparent past the crates
CAM_B = 286
B_WALL_W = W + int(math.ceil(CAM_B * sc.PAR_B['wall'])) + 12
B_STALL_W = W + CAM_B + 4
B_FG_W = 410
FACADES_X = list(sc.FACADES) + [
    (372, 428, 2, '#c99a4a', 'plaster'),
    (428, 478, 22, '#8a5236', 'planks'),
    (478, 536, 8, '#b86e3e', 'plaster'),
    (536, B_WALL_W + 4, 26, '#a84a3a', 'planks'),
]
STALLS_X = list(sc.STALLS) + [
    (386, 452, 91, '#3e7a8a', '#efe2c4', 'spice'),
    (462, 528, 88, '#d4a640', '#7a4a2e', 'carpet'),
    (540, 606, 93, '#b8483a', '#efe2c4', 'lamp'),
]
SIGNS_X = list(sc.WALL_SIGNS) + [(404, 44, 'spice'), (496, 50, 'carpet'), (566, 40, 'ship')]
WASH_X = [((388, 60), (440, 64), 5, 7), ((486, 66), (530, 62), 4, 9)]     # wall px
BUNT_H = 32             # rows holding the film's bunting (drawn in screen space in render_b)
BUNT_P = 1.2            # its parallax here: hanging a little in front of the stalls
BUNT_STEP, BUNT_N = 250, 3
SIGN_X = 126            # the creaky sign on the fg bracket (fg layer, p 1.4)
BELL = (sc.STALLS[3][0] + 46, sc.STALLS[3][2] + 16)    # the cracked bell on the junk stall (centre x, top)
KETTLE = (sc.STALLS[2][0] + 30, sc.COUNTER - 5)        # the kettle on the lamp stall's brazier

# ------------------------------------------------------------------ area 2: docks + sea of clouds (shot C)
# the dock (p 1) and the fg pier (1.35) are open sky / cloud sea right of x ~150; the cloud bands and
# the stratus tile; the peaks layer (p .06) is rebuilt PEAKS_W wide by the film's own generator.
CAM_C = 300
PEAKS_W = W + int(math.ceil(CAM_C * sc.PAR_C['peaks'])) + 8
SHIP_W = 128
BERTH_X = sc.SHIP_X0 + SHIP_W // 2
MOORED = 60 * 20000     # AIRSHIP frame while moored: a multiple of the ship's 60-frame bob
AWAY = 240              # steps out of sight before the airship comes back by itself
DESC = 120              # its descent to the berth
DESC_DX, DESC_DY = 70, 136      # it comes down from up here (off the top of the screen)
PUFF = 16               # the steam puff when it makes fast
SEA_X = 440


_NOSKY = np.array((255, 0, 255), np.uint8)     # sky hooks: a colour no frame pixel has (scenery)


@lru_cache(maxsize=8)
def _desert_sky(cx):
    """Area 0's painted sky alone, the dunes, mesas, town and rocks covering it blanked out (their pale
    sunlit ridges must not pass for sky). Static per camera: cached for the last few."""
    P = sc.PAR_A
    return Klugetown._cover(sc._sky_a().copy(), [
        (lay, -int(round(cx * P[k])), 0) for lay, k in (
            (sc._mesa_a(), 'mesa'), (_far_x(), 'far'), (sc._town_a()[0], 'town'), (_mid_x(), 'mid'),
            (_near_x(), 'near'), (_fg_ax(), 'fg'))])


@lru_cache(maxsize=8)
def _bazaar_sky(cx):
    """Area 1's painted sky alone (the strip over the alley and through the arches), with the far
    dunes, the facades, stalls and foreground blanked out: the pale ochre facades are close to the
    sky's warm horizon colour and must not be repainted as sky."""
    P = sc.PAR_B
    return Klugetown._cover(sc._sky_b().copy(), [
        (sc._far_b(), -int(round(cx * P['far'])), 0), (_wall_x()[0], -int(round(cx * P['wall'])), 0),
        (_stalls_x()[0], -int(round(cx * P['stall'])), 0), (_fg_x(), -int(round(cx * P['fg'])), 0)])


# ---- the world clock (game/daylight.py)
# The film paints Klugetown in the late golden hour: its low sun on the LEFT (the desert's sun disc and its
# crepuscular beams, the docks' sky glow, the bazaar's diagonal shafts), a third of the windows and every
# lamp lit. Here it follows the clock instead. The relight's own glow keeping is off (it would keep sunlit
# shack rims, the bazaar's jars, spices and brass pots glowing like lamps): GlowKeep keeps the REAL lights.
#   - the clock's sky, its sun on its arc, over the open sky at every hour (the painted left sun goes);
#   - low-sun light only in the golden hours, from the clock's sun: the desert's beams radiate from it, the
#     bazaar's shafts slant down from the left in the morning and from the right in the evening;
#   - by day the stratus and the cloud sea lose their sunset colours;
#   - the lamps come on one by one through the dusk (_lamp_k): the windows (the painted lit ones first,
#     then most of the dark ones), the string lights and bridge lamps, the lanterns (their glass: the
#     halos fade out as the night comes), the airship's portholes; the brazier's coals burn at every hour.
SHAFT = '#ffd890'                       # the film's low-sun light (its shafts) ...
BEAM = '#fff4d0'                        # ... and its crepuscular beams
UNLIT_BULB = '#a06a40'                  # the film's colour for a string-light bulb that is out
UNLIT_LAMP = '#8a6448'                  # a rope-bridge lamp by day
BULB_COLS = ('#ffd27a', '#ff9a6a', '#fff0a8', '#7ac0a0', '#e86a8a')
BULB_V = (0.22, 0.38, 0.55)             # lamp threshold of each string of lights (sc.LIGHTS_A)
BRIDGE_V = 0.3
COALS = ('#ff9a30', '#ffd27a', '#e85a20')
# the bazaar's shafts: the film's bands (slanting down to the right: the morning sun on the left) and the
# evening's (mirrored: the sun on the right), as (offset, width) on u = x -/+ 0.75 y
WALL_BANDS = ((-40, 26), (70, 18), (150, 30), (250, 22), (330, 26), (424, 24), (512, 28))
STALL_BANDS = ((-20, 30), (96, 22), (196, 34), (306, 26), (410, 26), (500, 32), (590, 24))


def _ramp(x, a, b):
    """0 at a .. 1 at b, smooth."""
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def _lamp_k(hour):
    """0 (day) .. 1 (night): how many of the lights are on. Each light has its threshold in 0..1: they come
    on one by one from 18.4 to 19.5 h and go out from 5.3 to 6.3 h."""
    h = hour % 24
    return clamp((h - 18.4) / 1.1) if h >= 12 else clamp((6.3 - h) / 1.0)


def _golden(hour):
    """0..1: the low sun's light (the bazaar's shafts, the desert's beams): morning 5.9 -> 8.2 h, evening
    16.2 -> 19.0 h; none by day or by night."""
    h = hour % 24
    if h >= 12:
        return _ramp(h, 16.2, 17.2) * (1 - _ramp(h, 18.4, 19.0))
    return _ramp(h, 5.9, 6.5) * (1 - _ramp(h, 7.4, 8.2))


def _day(hour):
    """0..1: full daylight, when the clouds keep no sunrise / sunset colour (8 to 15.6 h)."""
    h = hour % 24
    return _ramp(h, 6.8, 8.0) * (1 - _ramp(h, 15.6, 17.0))


def _halo(hour):
    """How much of a lit lantern's painted halo shows: all of it in the dusk, none at night (only the
    glass is light then: a halo darkened by the night is a muddy ring)."""
    return clamp(1 - 2.2 * daylight.night(hour))


def _q(k, n=4):
    return round(k * n) / n


def _glass(a, m, cols, tol=9):
    """Pixels of layer (a, m) within L1 tol of any of the colours."""
    f = a.astype(np.int16)
    hit = np.zeros(m.shape, bool)
    for cc in cols:
        hit |= np.abs(f - np.array(col(cc), np.int16)).sum(axis=2) < tol
    return hit & m


# the windows of the film's layers: lit glass, dark glass, the dark glass's marks (a top-left glint, or
# the bazaar's muntin cross), the window sizes (h, w), and what a dark window's marks become when it is lit
def _hz(c_):
    return mix(c_, sc.HAZE, 0.1)                 # the desert town layer's 0.1 haze


_WIN_SPECS = {
    'town': ([_hz(sc.WIN_LIT), _hz(sc.WIN_HI)], [_hz(sc.WIN_DK)], [_hz('#5a4050')], {(3, 3), (3, 2)},
             _hz(sc.WIN_LIT), _hz(sc.WIN_HI), 'glint'),
    'dock': ([sc.WIN_LIT, sc.WIN_HI], [sc.WIN_DK], ['#5a4050'], {(6, 6), (3, 2)},
             sc.WIN_LIT, sc.WIN_HI, 'glint'),
    'wall': ([sc.WIN_LIT, sc.WIN_HI], [sc.WIN_DK], ['#5a4050'], {(10, 7)}, sc.WIN_LIT, sc.WIN_HI, 'cross'),
}


@lru_cache(maxsize=3)
def _lamp_layer(key):
    """(dark layer, windows) of a layer the film paints with some of its windows lit ('town': the desert
    stacks, 'wall': the bazaar's facades without their shafts, 'dock': the docks' pillar). The dark layer
    has every window unlit (the lit ones turned to the film's dark glass, with its glint or muntin cross);
    windows = (ys, xs, lit colour, threshold) of every window's glass pixels in layer px: the painted lit
    windows keep their colours and come on first, ~60 % of the dark ones come on after them."""
    from scipy import ndimage
    a, m = {'town': lambda: sc._town_a()[0], 'wall': lambda: _wall_night()[0],
            'dock': lambda: sc._dock_c()[0]}[key]()
    lit_c, dark_c, mark_c, sizes, rgb, hi, marks = _WIN_SPECS[key]
    gl, gd, gm = _glass(a, m, lit_c), _glass(a, m, dark_c), _glass(a, m, mark_c)
    lab, n = ndimage.label(gl | gd | gm, structure=np.ones((3, 3)))
    hmax, wmax = max(h for h, _w in sizes), max(w for _h, w in sizes)
    b = a.copy()
    dk, mk = np.array(col(dark_c[0]), np.uint8), np.array(col(mark_c[0]), np.uint8)
    lit_rgb, hi_rgb = np.array(col(rgb), np.uint8), np.array(col(hi), np.uint8)
    YS, XS, RGB, V = [], [], [], []
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        if s_ is None:
            continue
        y0, x0 = s_[0].start, s_[1].start
        own = lab[s_] == i
        if (own.shape not in sizes or not own.all()) and not (     # a window cut by the layer's edge or
                own.shape[0] <= hmax and own.shape[1] <= wmax and own.sum() >= max(4, 0.6 * own.size)):
            continue                                               # partly behind a balcony rail
        if marks == 'cross':                     # the whole 7 x 10 window (glass seen between balusters)
            own = (gl | gd | gm)[y0:y0 + hmax, x0:x0 + wmax]
        yy, xx = np.nonzero(own)
        ys, xs = yy + y0, xx + x0
        h = hash01(x0, y0, 913)
        if gl[ys, xs].any():                     # painted lit: its own colours, among the first on
            out, v = a[ys, xs].copy(), 0.04 + 0.4 * h
            b[ys, xs] = dk                       # ... and dark glass in the dark layer
            if marks == 'glint':
                b[y0, x0] = mk
            else:
                b[ys[(yy == 4) | (xx == 3)], xs[(yy == 4) | (xx == 3)]] = mk
        else:
            sel = gd[ys, xs] | (gm[ys, xs] & (marks == 'glint'))
            ys, xs, yy, xx = ys[sel], xs[sel], yy[sel], xx[sel]
            out = np.repeat(lit_rgb[None], len(ys), 0)
            out[(yy == 0) & (xx == 0)] = hi_rgb
            v = 0.12 + h / 0.7
        YS.append(ys)
        XS.append(xs)
        RGB.append(out)
        V.append(np.full(len(ys), v, np.float32))
    win = (np.concatenate(YS), np.concatenate(XS), np.concatenate(RGB), np.concatenate(V))
    return (b, m), win


def _light_windows(c, win, k, ox, oy, rec):
    """Paint the windows of a layer just composited at (ox, oy) whose threshold the lamp ramp k has passed;
    their pixels and colours go to rec (to be kept lit: _kept)."""
    if k <= 0:
        return
    ys, xs, rgb, v = win
    sy, sx = ys + oy, xs + ox
    ok = (v < k) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    c.a[sy[ok], sx[ok]] = rgb[ok]
    rec.append((sy[ok], sx[ok], rgb[ok]))


def _rec_px(c, rec, pts, cols):
    """Add the screen pixels pts (list of (x, y)) that show one of the colours to rec."""
    if not pts:
        return
    xs, ys = (np.array(v) for v in zip(*pts))
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    if len(xs):
        px = c.a[ys, xs]
        hit = np.zeros(len(xs), bool)
        for cc in cols:
            hit |= np.all(px == np.array(col(cc), np.uint8), axis=1)
        rec.append((ys[hit], xs[hit], px[hit]))


def _kept(rec, c):
    """The mask of the recorded light pixels still showing their colours (nothing drawn over them)."""
    m = np.zeros((H, W), bool)
    for ys, xs, rgb in rec:
        if len(ys):
            s = np.all(c.a[ys, xs] == rgb, axis=1)
            m[ys[s], xs[s]] = True
    return m


class _LampFx:
    """engine.fx for the film's lantern and brazier halos, each scaled: a lantern's by `k` (set by the
    lantern drawn just before it: 0 while it is out, _halo when lit), the brazier's (#ff9a40) by `brazier`."""
    def __init__(self, brazier=1.0):
        self.k, self.brazier = 1.0, brazier

    def glow(self, c, x, y, r, color, strength=0.5, rings=4, ox=0, oy=0):
        k = self.brazier if color == '#ff9a40' else self.k
        if k > 0.02:
            fx.glow(c, x, y, r, color, strength=strength * k, rings=rings, ox=ox, oy=oy)

    def __getattr__(self, name):
        return getattr(fx, name)


def _lantern_fn(k, halo, lfx, rec):
    """sc.draw_lantern by the clock: out (unlit glass, no halo) until the lamp ramp k passes its threshold,
    then lit (its glass recorded in rec; the halo after it scaled by `halo`)."""
    def lantern(c, x, y, gc, f, seed, big=False):
        if k <= 0.08 + 0.8 * hash01(seed, 911):
            lfx.k = 0.0
            return sc.draw_lantern(c, x, y, mix(gc, '#5a4a52', 0.55), f, seed, big)
        lfx.k = halo
        x = sc.draw_lantern(c, x, y, gc, f, seed, big)
        pts = [(x - 1 + i, y + 1 + j) for i in range(3) for j in range(5)] if big else [(x, y + 1), (x, y + 2)]
        xs, ys = (np.array(v) for v in zip(*pts))
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        rec.append((ys[ok], xs[ok], c.a[ys[ok], xs[ok]].copy()))
        return x
    return lantern


def _flicker_fn(k):
    """hash01 for the film's shot renderers: a window flickering on (2 args) only once the lamps are lit;
    the desert's string-light bulbs (3 args: string, bulb, time) only past their string's threshold."""
    def h01(*a):
        if len(a) == 3:
            return hash01(*a) if k > BULB_V[a[0] % 3] else 0.0
        return hash01(*a) if k > 0.6 else 1.0
    return h01


def _flicker_px(rects, ox, oy=0):
    """Screen pixels of the film's flickering window rects ((x, y, w, h, seed), layer px)."""
    return [(x + ox + i, y + oy + j) for (x, y, w, h, _s) in rects for i in range(w) for j in range(h)]


@lru_cache(maxsize=4)
def _cloud_day(key, dq):
    """A cloud layer of the film (the stratus streaks, a band of the cloud sea) seen by day: its
    sunset-lit peach and amber turned to the day's white and blue-grey (dq of the way; the cloud sea keeps
    a little of its dusty brown)."""
    a, m = sc._stratus_a() if key == 's' else sc._cloud_band(key)
    if dq <= 0:
        return a, m
    f = a.astype(np.float32)
    lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
    tgt = np.stack([lum * 0.9 + 18, lum * 0.94 + 16, lum * 0.94 + 30], axis=-1)
    k = dq * (0.92 if key == 's' else 0.7)
    out = np.clip(f + (tgt - f) * k, 0, 255).astype(np.uint8)
    return np.where(m[..., None], out, a), m


def _ship_fn(on):
    """sc._ship with the big airship's portholes and cabin windows lit (on) or dark glass."""
    def ship(kind):
        return _ship_v(kind, on)
    return ship


@lru_cache(maxsize=4)
def _ship_v(kind, on):
    rgba, anchors = sc._ship(kind)
    if not kind:
        return rgba, anchors
    a = rgba.copy()
    for x in (46, 56, 66, 76, 86):              # the portholes (the film lights two)
        a[48, x, :3] = col('#ffd27a' if on else '#3a2531')
    for x in (61, 67):                          # the cabin's windows
        a[35:38, x:x + 3, :3] = col(sc.WIN_LIT if on else sc.WIN_DK)
    a[35, 61, :3] = col(sc.WIN_HI if on else '#5a4050')
    return a, anchors


@lru_cache(maxsize=1)
def _ship_lights():
    """(ys, xs, rgb) of the big airship's lights, sprite px."""
    on, off = _ship_v(1, True)[0], _ship_v(1, False)[0]
    ys, xs = np.nonzero(np.any(on[..., :3] != off[..., :3], axis=2))
    return ys, xs, on[ys, xs, :3].copy()


@lru_cache(maxsize=16)
def _beams(sx, sy, gq):
    """The film's crepuscular beams (_beams_a) radiating from the clock's sun at screen (sx, sy) instead of
    its painted left one (mirrored when the sun is on the right), thinned by gq (the golden light)."""
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - sx) * (1 if sx < W // 2 else -1)
    th = np.arctan2(yy - sy, dx)
    d = np.hypot(dx, yy - sy)
    band = (np.sin(th * 11 + 0.7) > 0.72) & (d > 13) & (d < 230) & (th > -0.5) & (th < 1.2)
    fade = np.clip(1 - d / 230.0, 0, 1)
    return band & (bayer(H, W) < (0.55 * fade + 0.1) * gq)


def _beam_at(hour):
    """The beams' mask at the clock's hour (None outside the golden hours or with the sun down)."""
    gq = _q(_golden(hour))
    kind, bx, by = daylight._body_pos(hour)
    if gq <= 0 or kind != 'sun':
        return None
    return _beams(bx // 4 * 4, by // 4 * 4, gq)


_NO_BEAM = np.zeros((H, W), bool)


@lru_cache(maxsize=1)
def _bulb_px():
    """Town-layer px of the string lights' bulbs (sc.render_a draws one every 4 px of sc.LIGHTS_A)."""
    out = []
    for (x0, y0), (x1, y1), sag in sc.LIGHTS_A:
        for i in range(2, x1 - x0 + 1, 4):
            u = i / max(1, x1 - x0)
            out.append((x0 + i, int(round(y0 + (y1 - y0) * u + sag * math.sin(math.pi * u))) + 1))
    return out


def _shaft_mask(shape, bands, side, ymin=0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    u = xx - yy * 0.75 if side == 'L' else xx + yy * 0.75 - 135
    sh = np.zeros(shape, bool)
    for (s0, sw) in bands:
        sh |= (u - s0 >= 0) & (u - s0 < sw)
    return sh & (yy > ymin)


@lru_cache(maxsize=4)
def _shafted(key, side, gq):
    """The bazaar's walls ('wall': the dark-window layer) or stalls ('stall') with the low sun's shafts at
    strength gq, slanting from the sun's side ('L' morning, 'R' evening)."""
    if key == 'wall':
        (a, m), _w = _lamp_layer('wall')
        bands, k, ymin = WALL_BANDS, 0.28, 0
    else:
        a, m = _stalls_night()[0]
        bands, k, ymin = STALL_BANDS, 0.22, 120
    if gq <= 0:
        return a, m
    lm = _shaft_mask(m.shape, bands, side, ymin) & m
    a = a.copy()
    v = a[lm].astype(np.float32)
    v += (np.array(col(SHAFT), np.float32) - v) * k * gq
    a[lm] = np.clip(v, 0, 255).astype(np.uint8)
    return a, m


def _unshaft(layer, bands, k, ymin=0):
    """Undo the low sun's light shafts (_shafts / the film's builders: v += (#ffd890 - v) * k on diagonal
    bands) on a layer."""
    a, m = layer
    a = a.copy()
    lm = _shaft_mask(m.shape, bands, 'L', ymin) & m
    v = a[lm].astype(np.float32)
    tgt = np.array(col(SHAFT), np.float32)
    v = (v - tgt * k) / (1 - k)
    a[lm] = np.clip(v, 0, 255).astype(np.uint8)
    return a, m


@lru_cache(maxsize=1)
def _wall_night():
    """The bazaar's facades without the film's shafts (nor the extended street's)."""
    lay, anim = _wall_x()
    return _unshaft(lay, WALL_BANDS, 0.28), anim


@lru_cache(maxsize=1)
def _stalls_night():
    lay, anim = _stalls_x()
    return _unshaft(lay, STALL_BANDS, 0.22, ymin=120), anim


def _nop(*_a, **_k):
    return None


def _shafts(layer, bands, k, ymin=0):
    """The film's low-sun light shafts (diagonal bands, as in _wall_b / _stalls_b) on more of a layer."""
    a, m = layer
    a = a.copy()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    shaft = np.zeros(m.shape, bool)
    for (s0, sw) in bands:
        u = xx - (yy * 0.75) - s0
        shaft |= (u >= 0) & (u < sw)
    lm = shaft & m & (yy > ymin)
    v = a[lm].astype(np.float32)
    v += (np.array(col('#ffd890'), np.float32) - v) * k
    a[lm] = np.clip(v, 0, 255).astype(np.uint8)
    return a, m


@lru_cache(maxsize=1)
def _dark_windows():
    """The unlit shack windows of the town layer (WIN_DK under the layer's 0.1 haze), one entry per
    window: (centre x, centre y, ys, xs) in town-layer px (4-connected groups of a few pixels)."""
    (a, m), _ = sc._town_a()
    dk = np.array(col(sc.WIN_DK), np.float32)
    dk += (np.array(col(sc.HAZE), np.float32) - dk) * 0.1
    left = set(zip(*np.nonzero((np.abs(a.astype(np.float32) - dk).sum(axis=2) < 8) & m)))
    out = []
    while left:
        stack, grp = [left.pop()], []
        while stack:
            y, x = stack.pop()
            grp.append((y, x))
            for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if q in left:
                    left.remove(q)
                    stack.append(q)
        ys, xs = np.array(grp).T
        out.append((int(xs.mean()), int(ys.mean()), ys, xs))
    return sorted(out, key=lambda w: w[0])


@lru_cache(maxsize=1)
def _lit_colours():
    h = sc.HAZE
    return np.array(col(mix(sc.WIN_LIT, h, 0.1)), np.uint8), np.array(col(mix(sc.WIN_HI, h, 0.1)), np.uint8)


def _cactus(cv, cx0, base, hgt):
    """The film's two-armed cactus (as in _near_a)."""
    g = ['#3f5a3a', '#4f7046', '#628a52', '#7ea864']
    cv.rect(cx0, base - hgt, 4, hgt, g[1])
    cv.vline(cx0, base - hgt + 1, base - 1, g[2])
    cv.vline(cx0 + 3, base - hgt + 1, base - 1, g[0])
    cv.hline(cx0 + 1, cx0 + 2, base - hgt - 1, g[1])
    cv.rect(cx0 - 4, base - hgt + 6, 4, 3, g[1])
    cv.rect(cx0 - 4, base - hgt + 2, 3, 5, g[1])
    cv.vline(cx0 - 4, base - hgt + 2, base - hgt + 8, g[2])
    cv.rect(cx0 + 4, base - hgt + 9, 3, 2, g[1])
    cv.rect(cx0 + 5, base - hgt + 5, 3, 5, g[1])
    cv.vline(cx0 + 7, base - hgt + 5, base - hgt + 9, g[0])
    for yy_ in range(base - hgt + 1, base, 3):
        cv.px(cx0 + 1, yy_, g[3])
    cv.px(cx0 + 1, base - hgt - 2, '#e86a8a')


def _rock(cv, rx, ry, rw, rh, seed):
    m = sc.poly_mask(H, cv.w, [(rx, ry + rh), (rx + 2, ry + 2), (rx + rw // 3, ry), (rx + rw - 3, ry + 1),
                               (rx + rw, ry + rh)])
    sc.shade_solid(cv, m, PEBBLE, outline='#6a5244', seed=seed)


def _more_far(cv):
    for i, (xa, ya, wl, wr) in enumerate([(356, 104, 54, 30), (330, 111, 40, 20), (392, 109, 40, 24)]):
        sc.dune(cv, xa, ya, wl, wr, FAR_DUNE, seed=20 + i, ripple=True, slant=0.5)


def _more_mid(cv):
    w = cv.w
    sc.dune(cv, 452, 125, 100, 60, MID_DUNE, seed=16, slant=0.55)
    spire = sc.poly_mask(H, w, [(398, 142), (401, 116), (405, 104), (410, 99), (416, 101), (418, 112),
                                (421, 142)])
    sc.shade_solid(cv, spire, BONE, outline='#7e6452', strata=(4, 0.14), seed=24, cracks=5)
    cap = sc.poly_mask(H, w, [(399, 102), (403, 95), (414, 92), (423, 96), (419, 101), (405, 104)])
    sc.shade_solid(cv, cap, BONE, outline='#7e6452', seed=25)
    sc.dune(cv, 392, 136, 70, 50, MID_DUNE, seed=17, slant=0.5)
    sc.dune(cv, 470, 134, 60, 50, MID_DUNE, seed=18, slant=0.5)


def _more_near(cv):
    sc.dune(cv, 476, 147, 90, 60, NEAR_DUNE, seed=34, slant=0.5)
    sc.dune(cv, 560, 151, 70, 50, NEAR_DUNE, seed=35, slant=0.5)
    _cactus(cv, 452, 156, 16)
    _cactus(cv, 548, 158, 10)
    for i, (rx, ry, rw, rh) in enumerate([(420, 160, 16, 8), (500, 166, 12, 7), (580, 158, 10, 6)]):
        _rock(cv, rx, ry, rw, rh, 80 + i)


def _more_fg(cv):
    sc.dune(cv, 600, 166, 80, 70, FG_DUNE, seed=42, slant=0.4)


def _wider(builder, key, more, hook):
    """The film's layer builder run at the wider width A_W[key], with `more` art added at its end
    (before the layer's haze when it has one, so the new art is hazed alike)."""
    if hook == 'haze_layer':
        def h(cv, color, k, *a, **kw):
            more(cv)
            sc.haze_layer(cv, color, k, *a, **kw)
    else:
        def h(cv):
            more(cv)
            return sc.to_layer(cv)
    return rebind(builder.__wrapped__, lw=lambda _p: A_W[key], **{hook: h})()


@lru_cache(maxsize=1)
def _far_x():
    return _wider(sc._far_a, 'far', _more_far, 'haze_layer')


@lru_cache(maxsize=1)
def _mid_x():
    return _wider(sc._mid_a, 'mid', _more_mid, 'haze_layer')


@lru_cache(maxsize=1)
def _near_x():
    return _wider(sc._near_a, 'near', _more_near, 'to_layer')


@lru_cache(maxsize=1)
def _fg_ax():
    return _wider(sc._fg_a, 'fg', _more_fg, 'to_layer')


@lru_cache(maxsize=1)
def _wall_x():
    lay, anim = rebind(sc._wall_b.__wrapped__, lwb=lambda _p: B_WALL_W, FACADES=FACADES_X,
                       WALL_SIGNS=SIGNS_X)()
    return _shafts(lay, ((424, 24), (512, 28)), 0.28), anim


def _crate_more(cv, x, yb, w, h, light=False):
    """sc.crate, plus the extended street's clutter right after the film's last street crate."""
    sc.crate(cv, x, yb, w, h, light)
    if (x, yb) == (306, sc.FLOOR + 4):
        sc.barrel(cv, 380, sc.FLOOR + 4, 9, 13)
        sc.crate(cv, 454, sc.FLOOR + 3, 10, 9)
        sc.barrel(cv, 531, sc.FLOOR + 5, 9, 12)
        sc.crate(cv, 541, sc.FLOOR + 6, 11, 8, light=True)


@lru_cache(maxsize=1)
def _stalls_x():
    lay, anim = rebind(sc._stalls_b.__wrapped__, lwb=lambda _p: B_STALL_W, STALLS=STALLS_X,
                       crate=_crate_more)()
    return _shafts(lay, ((410, 26), (500, 32), (590, 24)), 0.22, ymin=120), anim


@lru_cache(maxsize=1)
def _fg_x():
    return rebind(sc._fg_b.__wrapped__, lwb=lambda _p: B_FG_W)()


@lru_cache(maxsize=1)
def _wide_peaks():
    """The film's black peaks, generated by its own builder over a wider strip (same seed: the first
    327 px are identical, the rest continues the range)."""
    return rebind(sc._peaks_c.__wrapped__, lwc=lambda _p: PEAKS_W)()


def _ease_in_ext(t):
    """core.ease_in, continued past t = 1 at its end slope (the airship keeps sailing east)."""
    return t * t if t <= 1 else 1 + 2 * (t - 1)


def cast_offset(s):
    """(dx, dy) of the airship s steps after its cast-off: the film's path, continued east."""
    t = max(0.0, s / (sc.C1 - sc.AIRSHIP))
    return int(round(110 * _ease_in_ext(t) * 0.6 + 40 * t)), int(round(14 * ease_in_out(t)))


def _spin(a, extra, span):
    """Extra frames after a steps of a spin-up adding `extra` frames over `span` steps (smooth)."""
    a = min(max(a, 0), span)
    return extra / 2 * (1 - math.cos(math.pi * a / span))


class Klugetown(GlowKeep, Region):
    SEG = 'klugetown'
    CARD = ('TRADE TOWN', ['RUST - SCRAP - AIRSHIPS', 'BEYOND EQUESTRIA'])
    _raw = None                 # the area's frame as rendered (for the sky repaint in after_light)

    def auto_glow(self):
        """No painted spot glows by itself (sunlit shack rims, jars, spices, brass pots): the real lights
        are kept by mask (the area renders: keep_glow)."""
        return False

    def after_light(self, c, t):
        """After the relight: the kept lights (GlowKeep), then the clock's sky over the whole open sky at
        every hour (the base keeps the painted sky, its sun low on the LEFT, while the clock's sky looks
        alike, ~16.5 to ~18.3 h: the sun jumped sides), and the desert's beams in the sky, radiating from
        the clock's sun in the golden hours."""
        super().after_light(c, t)
        raw, self._raw = self._raw, None
        if raw is None or self.area.sky is None:
            return
        a, f, ix, iy = raw
        key = (self.ai, t, ix, iy)                       # the base's memo: once per step
        if self._sky_memo is None or self._sky_memo[0] != key:
            self._sky_memo = (key, daylight.sky_pixels(a, self.area.sky(f, ix, iy).a))
        mask = self._sky_memo[1]
        if not mask.any():
            return
        hour = self.g.hour
        if daylight.sky_weight(self.HOUR, hour) < 0.999:
            y0, y1 = self.area.cam_y
            span = y1 - y0 + 130                         # (as the base: to the area's horizon)
            if self.area.horizon is not None:
                span = max(span, self.area.horizon - (y0 - 10) + 4)
            daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, span, t)
        bm = _beam_at(hour) if self.ai == 0 else None
        if bm is not None:
            m = bm & mask
            reg = c.a[m].astype(np.int16)
            c.a[m] = np.clip(reg + ((np.array(col(BEAM), np.int16) - reg) * 3) // 20, 0, 255).astype(np.uint8)

    def build_areas(self):
        desert = [
            Landmark('BONE DRY DESERT', 64, (58, 92, 'down', 24), p=P_MID),
            Landmark('KLUGETOWN', 236, (215, 47, 'up', 19), p=P_TOWN),
        ]
        bazaar = [Landmark('THE BAZAAR', 196, (196, 92, 'up', 20))]
        docks = [
            Landmark('AIRSHIP DOCKS', 96, (86, sc.PIER2_Y + 2, 'down', 5)),
            Landmark('SEA OF CLOUDS', SEA_X, (SEA_X, 128, 'down', 10)),
        ]
        return [
            Area(self.render_desert, cam_x=(0, CAM_A), y_range=(30, 140), start=(176, 66),
                 landmarks=desert, balloon='m', balloon_tint=('#f0a868', 0.22),
                 hotspots=[Hotspot('lights', 134, 72, 'lantern', cooldown=LIGHTS_T, p=P_TOWN),
                           Hotspot('gust', 446, 130, 'wind', cooldown=GUST_T)],
                 stamps=[Stamp('sky', 412, 44)], sky=self.sky_desert),
            Area(self.render_bazaar, cam_x=(0, CAM_B), y_range=(30, 146), start=(40, 74),
                 landmarks=bazaar, balloon='m', balloon_tint=('#c9805a', 0.3),
                 hotspots=[Hotspot('kettle', KETTLE[0], KETTLE[1] - 6, 'steam', cooldown=90),
                           Hotspot('bell', BELL[0], BELL[1] + 6, 'bell', cooldown=75)],
                 stamps=[Stamp('stalls', 238, 128)], sky=self.sky_bazaar, horizon=150),
            Area(self.render_docks, cam_x=(0, CAM_C), y_range=(24, 140), start=(98, sc.PIER1_Y - 1),
                 landmarks=docks, balloon='m', balloon_tint=('#e0906a', 0.25),
                 hotspots=[Hotspot('airship', BERTH_X, 84, 'horn', r=52, cooldown=60)],
                 stamps=[Stamp('pier', 74, 122)], sky=self.sky_docks),
        ]

    # -------------------------------------------------------------- sky hooks (game/daylight.py)
    # The painted sky alone; where the film's solid scenery layers cover it (dunes, mesas, the town,
    # the peaks, the dock pillar), a colour no frame pixel has, so their pale sunlit ridges never pass
    # for sky.
    @staticmethod
    def _cover(c, layers):
        """Paint _NOSKY over `c` wherever the (rgb, mask) layers at (x, y) are solid."""
        hit = np.zeros((H, W), bool)
        for (_a, m), x, y in layers:
            x0, x1 = max(0, x), min(W, x + m.shape[1])
            y0, y1 = max(0, y), min(H, y + m.shape[0])
            if x1 > x0 and y1 > y0:
                hit[y0:y1, x0:x1] |= m[y0 - y:y1 - y, x0 - x:x1 - x]
        np.copyto(c.a, _NOSKY, where=hit[..., None])
        return c

    def sky_desert(self, f, cx, cy):
        return _desert_sky(cx).copy()

    def sky_bazaar(self, f, cx, cy):
        return _bazaar_sky(cx).copy()

    def sky_docks(self, f, cx, cy):
        lays = [(_wide_peaks(), -int(round(cx * sc.PAR_C['peaks'])), 0),
                (sc._dock_c()[0], -int(round(cx * sc.PAR_C['dock'])), 0)]
        for lvl, sp in ((0, 0.08), (1, 0.2), (2, 0.4)):  # the cloud sea (its pale tops are not sky)
            band = sc._cloud_band(lvl)
            ox = -int((cx * sc.PAR_C['cloud%d' % lvl] - f * sp) % 480)
            oy = int(round(math.sin(f / 30.0 + lvl) * (lvl * 0.6)))
            lays += [(band, ox, oy), (band, ox + 480, oy)]
        return self._cover(sc._sky_c().copy(), lays)

    # -------------------------------------------------------------- clocks
    def init_state(self):
        self.ship = ('moored', -10 ** 6)     # (mode, step): moored | cast | away | descend
        self.gust_x0 = 0.0                   # near-layer x the gust's tumbleweed starts from

    def on_area(self, ai):
        if ai != 2:
            self.ship = ('moored', -10 ** 6)  # back at its berth while you are elsewhere

    @staticmethod
    def tumble_x(t):
        """Near-layer x of the ambient tumbleweed (None between runs)."""
        x = TUMBLE_X0 + (t % TUMBLE_PERIOD) * TUMBLE_V
        return x if x <= TUMBLE_X1 else None

    def ship_state(self, f, t):
        """(AIRSHIP frame on the free-running scene clock f, dx, dy) of the airship at step t."""
        mode, t0 = self.ship
        if mode == 'cast' and t >= t0:
            dx, dy = cast_offset(t - t0)
            return self.scene_f(t0), dx, dy
        if mode in ('away', 'descend'):
            a = ((f - 1000) // 120) * 120        # long under way (props spinning); bob-phase safe
            if mode == 'away':
                return a, 4000, 0
            k = ease_out(min(1.0, (t - t0) / DESC))
            return a, int(round(DESC_DX * (1 - k))), int(round(DESC_DY * (1 - k)))
        return MOORED, 0, 0

    # -------------------------------------------------------------- interaction
    def on_interact(self, key, t):
        g = self.g
        if key == 'lights':
            g.sfx('chime', 0.4)
            g.sfx('sparkle', 0.3)
        elif key == 'gust':
            self.gust_x0 = self.cx - 14
            g.sfx('wind', 0.55, 1)
            g.sfx('creak', 0.25)
        elif key == 'bell':
            g.sfx('bell', 0.55)
            g.sfx('clank', 0.3)
        elif key == 'kettle':
            g.sfx('whistle', 0.35)
            g.sfx('steam', 0.4)
        elif key == 'airship':
            mode, t0 = self.ship
            g.sfx('horn', 0.5)
            if mode == 'moored':
                d = 2
                while (self.scene_f(t + d) - MOORED) % 30:      # cast off on a bob zero: no jump
                    d += 1
                self.ship = ('cast', t + d)
            elif mode == 'away':
                self.ship = ('descend', t)
                g.sfx('whoosh', 0.25)

    # -------------------------------------------------------------- events + sound
    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.switch is not None:
            return
        f = self.scene_f(t)
        if self.ai == 0:
            if t % 216 == 40:
                g.sfx('creak', 0.18)                 # windmills / rope bridges
            if t % 288 == 130:
                g.sfx('clank', 0.14)                 # the crane hook swinging
            if t % 360 == 250:
                g.sfx('steam', 0.12)                 # smokestacks
            if t % TUMBLE_PERIOD == 20:
                g.sfx('wind', 0.22)                  # the gust that rolls the tumbleweed
        elif self.ai == 1:
            if (f - sc.B0) % 72 == 9:
                g.sfx('creak', 0.3 * self.near_gain(SIGN_X + 0.4 * self.cx))
            if t % 300 == 120:
                g.sfx('clank', 0.18 * self.near_gain(275))    # the junk pile
            if t % 360 == 200:
                g.sfx('steam', 0.15 * self.near_gain(KETTLE[0]))
        else:
            mode, t0 = self.ship
            s = t - t0
            if mode == 'moored':
                if t % 240 == 100 and s > PUFF:
                    g.sfx('creak', 0.25 * self.near_gain(BERTH_X))   # hull straining at its lines
            elif mode == 'cast' and s >= 0:
                dx, _ = cast_offset(s)
                gain = self.near_gain(BERTH_X + dx)
                if s == 0:
                    g.sfx('steam', 0.6 * max(0.4, gain))
                elif s == 3:
                    g.sfx('creak', 0.5 * gain)
                elif s == 14:
                    g.sfx('whoosh', 0.25 * gain)
                if sc.SHIP_X0 + dx > self.cx + W + 8:
                    self.ship = ('away', t)          # out of sight past the right edge
            elif mode == 'away' and s >= AWAY:
                self.ship = ('descend', t)
                g.sfx('whoosh', 0.2 * self.near_gain(BERTH_X))
            elif mode == 'descend' and s >= DESC:
                self.ship = ('moored', t)            # makes fast with a puff of steam
                g.sfx('steam', 0.45 * self.near_gain(BERTH_X))
                g.sfx('clank', 0.35 * self.near_gain(BERTH_X))
            if t % 420 == 300:
                g.sfx('wind', 0.25)

    # -------------------------------------------------------------- area 0: desert + town
    def render_desert(self, f, cx, cy):
        t = self.g.t
        fg = _fg_ax()
        town = sc._town_a()[0]
        ox = -int(round(cx * P_TOWN))
        weeds = []
        tx = self.tumble_x(t)
        if tx is not None:
            weeds.append((tx, tx - TUMBLE_X0))
        tg = self.spot_t.get('gust')
        ga = None if tg is None else t - tg
        if ga is not None and 0 <= ga < GUST_T + 60:
            d = ga * GUST_V
            weeds.append((self.gust_x0 + 26 + d * 1.15, d + 77))     # rolls just ahead of the devil
        extra = 0.0 if tg is None else _spin(t - tg, 80, GUST_T)
        tl = self.spot_t.get('lights')
        la = None if tl is None else t - tl

        hour = self.g.hour
        k = _lamp_k(hour)                        # the clock's lights (see the world clock above)
        town_lay, town_win = _lamp_layer('town')
        rec = []                                 # the lit pixels, kept lit by GlowKeep
        bm = _beam_at(hour)                      # the low sun's beams, on the scenery behind the dunes
        if bm is not None:                       # (on the sky: after_light, over the clock's sky)
            bm = bm & np.all(_desert_sky(cx).a == _NOSKY, axis=2)
        dq = _q(_day(hour), 10)

        def comp(c, layer, x, y=0):
            if layer is town:                    # its windows unlit, lit one by one by the clock
                layer = town_lay
            if layer is fg:                      # tumbleweeds + the dust devil, between near and fg
                for (wx, d) in weeds:
                    ty = 160 - int(abs(math.sin(d / 308.0 * 9.5)) * 6)
                    sc.draw_tumbleweed(c, int(round(wx)) - cx, ty, int(d / 4.5))
                if ga is not None and 0 <= ga < GUST_T:
                    self._dust_devil(c, ga, int(round(self.gust_x0 + ga * GUST_V)) - cx, f)
            sc.comp(c, layer, x, y)
            if layer is town_lay:
                _light_windows(c, town_win, k, x, 0, rec)
                if la is not None and 0 <= la < LIGHTS_T:
                    self._windows(c, la, ox, rec)

        def windmill(c, x, y, r, f_, speed=0.22, seed=0):
            sc.draw_windmill(c, x, y, r, f_ + extra * (1 + (seed % 3) * 0.25), speed, seed)

        render = rebind(sc.render_a, cam_a=lambda _f: cx, draw_balloon=_nop, callout=_nop, comp=comp,
                        draw_windmill=windmill, _far_a=_far_x, _mid_a=_mid_x, _near_a=_near_x, _fg_a=_fg_ax,
                        hash01=_flicker_fn(k), _beams_a=lambda: _NO_BEAM if bm is None else bm,
                        _stratus_a=lambda: _cloud_day('s', dq))
        c = render(f)
        lamps = [((x0 + x1) // 2 + ox, int(round((y0 + y1) / 2 + sag)) - 4)
                 for (x0, y0), (x1, y1), sag in sc.BRIDGES]
        if k > BRIDGE_V:                         # the rope bridges' lamps
            _rec_px(c, rec, lamps, ('#ffd27a', '#e8a040'))
        else:
            for (x, y) in lamps:
                if 0 <= x < W and tuple(c.a[y, x]) in (col('#ffd27a'), col('#e8a040')):
                    c.a[y, x] = col(UNLIT_LAMP)
        if k > 0:                                # the string lights, a window flickering on
            _rec_px(c, rec, [(x + ox, y) for (x, y) in _bulb_px()], BULB_COLS)
            _rec_px(c, rec, _flicker_px(sc._town_a()[1].get('windows', []), ox), ('#e8a040',))
        if la is not None and 0 <= la < LIGHTS_T:
            self._string_lights(c, la, ox, rec)
        if ga is not None and 0 <= ga < GUST_T:
            k = math.sin(math.pi * ga / GUST_T)
            sc.sand_wisps(c, f * 2, (0, 100, W, 70), n=int(60 * k), seed=8, color='#c99a68', speed=3.2)
            sc.sand_wisps(c, f * 2, (0, 64, W, 50), n=int(30 * k), seed=9, color='#fdf0d0', speed=3.8)
        self.keep_glow(c, _kept(rec, c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        return c

    @staticmethod
    def _dust_devil(c, a, x, f):
        """'gust': a twisting column of sand travelling east over the near dunes (a dust skirt at its
        foot, banded swirls up the funnel, grains whirling round it)."""
        k = min(1.0, a / 14, (GUST_T - a) / 18)
        if k <= 0 or not -40 < x < W + 40:
            return
        hgt = int(64 * k)
        base, top = 158, 158 - hgt
        cols = ('#f4dcb0', '#ddb47e', '#c0925e', '#9c7048')
        for y in range(top, base):
            h = (base - y) / 64.0
            r = 3 + int((h ** 1.3) * 17 * k)
            sway = int(round(math.sin(y * 0.14 + a * 0.3) * (1 + h * 5)))
            dens = (0.95 - 0.35 * h) * k
            for xx in range(-r, r + 1):
                e = abs(xx) / (r + 1)
                if hash01(xx * 7 + y * 3, (a + y) // 2, 3) < dens * (1 - e * e * 0.8):
                    band = ((y * 2 + xx + a * 2) // 5) % 3
                    c.px(x + sway + xx, y, cols[3] if e > 0.85 else cols[band])
        for xx in range(-22, 23):                # the dust skirt
            for y in range(base - 5, base + 3):
                e = (xx / 22.0) ** 2 + ((y - base + 1) / 4.0) ** 2
                if e < 1 and hash01(xx, y, a // 3) < (1 - e) * 0.8 * k:
                    c.px(x + xx, y, cols[1] if (xx + y) % 3 else cols[2])
        for q in range(14):                      # grains whirling round the column
            ang = a * 0.45 + q * 0.45
            yy = base - 4 - int(hash01(q, 5) * (hgt - 6))
            rr = 6 + int((base - yy) * 0.3)
            c.px(x + int(math.cos(ang) * rr), yy + int(math.sin(ang) * 2), '#fbe8c4' if q % 2 else '#8a6444')

    @staticmethod
    def _windows(c, a, ox, rec):
        """'lights': the dark shack windows light up in a sweep from the left, with a warm bloom (the lit
        glass goes to rec: kept lit at night)."""
        from engine import fx
        lit, hi = _lit_colours()
        for (wx, wy, ys, xs) in _dark_windows():
            on0 = (wx - 110) * 0.22
            if not (on0 <= a < on0 + 62 + (wx * 7) % 13):
                continue
            sx = xs + ox
            if sx.max() < 0 or sx.min() >= W:
                continue
            ok = (sx >= 0) & (sx < W)
            c.a[ys[ok], sx[ok]] = lit
            c.a[ys.min(), min(W - 1, max(0, sx.min()))] = hi
            rec.append((ys[ok], sx[ok], c.a[ys[ok], sx[ok]].copy()))
            fx.glow(c, wx + ox, wy, 5, '#ffc861', strength=0.45 if a - on0 > 3 else 0.7, rings=2)

    @staticmethod
    def _string_lights(c, a, ox, rec):
        """'lights': every bulb on the strings and bridges lit, chasing colours, with a little bloom (the
        bulbs go to rec)."""
        from engine import fx
        if a >= 96:
            return
        bulbs = ('#ffd27a', '#ff9a6a', '#fff0a8', '#7ac0a0', '#e86a8a')
        pts = []
        for li, ((x0, y0), (x1, y1), sag) in enumerate(sc.LIGHTS_A):
            for i in range(2, x1 - x0 + 1, 4):
                u = i / max(1, x1 - x0)
                yy = int(round(y0 + (y1 - y0) * u + sag * math.sin(math.pi * u)))
                cc = bulbs[(i // 4 + li + a // 5) % 5]
                c.rect(x0 + i + ox, yy + 1, 2, 2, cc)
                pts += [(x0 + i + ox + q % 2, yy + 1 + q // 2) for q in range(4)]
                if (i // 4 + a // 5) % 2 == 0:
                    fx.glow(c, x0 + i + ox, yy + 1, 4, cc, strength=0.35, rings=2)
        xs, ys = (np.array(v) for v in zip(*pts))
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        rec.append((ys[ok], xs[ok], c.a[ys[ok], xs[ok]].copy()))
        ui.burst(c, a, 0, 205 + ox, 40, ('gold', 'butter', 'white'), n=26, seed=5, power=0.7, life=44,
                 gravity=0.03)

    # -------------------------------------------------------------- area 1: the bazaar
    def render_bazaar(self, f, cx, cy):
        t = self.g.t
        hour = self.g.hour
        k = _lamp_k(hour)                        # the clock's lights (see the world clock above)
        gq = _q(_golden(hour))                   # the low sun's shafts, from its side
        side = 'R' if hour % 24 >= 12 else 'L'
        wall_l, stalls = _shafted('wall', side, gq), _shafted('stall', side, gq)
        wall_anim, stall_anim = _wall_x()[1], _stalls_x()[1]
        win = _lamp_layer('wall')[1]
        wo = -int(round(cx * sc.PAR_B['wall']))
        fgl = _fg_x()
        snap = []
        rec = []                                 # the lit pixels, kept lit by GlowKeep
        halo = _halo(hour)
        lfx = _LampFx(brazier=max(_golden(hour), k) * halo)

        def comp(c, layer, x, y=0):
            if layer is stalls:                  # washing lines over the extended street
                for (p0, p1, n, seed) in WASH_X:
                    sc.draw_washing(c, f, (p0[0] + wo, p0[1]), (p1[0] + wo, p1[1]), n, seed=seed)
            sc.comp(c, layer, x, y)
            if layer is wall_l:                  # the facades' windows, lit one by one by the clock
                _light_windows(c, win, k, x, 0, rec)
            if layer is fgl:                     # the film's screen-space bunting comes next
                snap.append(c.a[:BUNT_H].copy())

        render = rebind(sc.render_b, cam_b=lambda _f: cx, draw_balloon=_nop, callout=_nop, comp=comp,
                        _wall_b=lambda: (wall_l, wall_anim), _stalls_b=lambda: (stalls, stall_anim),
                        _fg_b=_fg_x, WALL_SIGNS=SIGNS_X, fx=lfx, hash01=_flicker_fn(k),
                        draw_lantern=_lantern_fn(k, halo, lfx, rec))
        c = render(f)
        if k > 0:                                # a window flickering on
            _rec_px(c, rec, _flicker_px([(x, y, 3, 4, s_) for (x, y, _w, _h, s_) in wall_anim.get('win', [])],
                                        wo), (sc.WIN_LIT,))
        _rec_px(c, rec, [(296 - cx + i, 147) for i in range(-4, 5)] + [(295 - cx, 146)], COALS)   # the brazier
        c.a[:BUNT_H] = snap[0]                   # ... is replaced by bunting strung along the street
        self._bunting(c, f, -int(round(cx * BUNT_P)))
        tk = self.spot_t.get('kettle')
        if tk is not None and 0 <= t - tk < 90:
            a = t - tk
            kx, ky = KETTLE[0] + 7 - cx, KETTLE[1] - 5
            k = min(1.0, a / 8) * (1 - max(0, a - 60) / 30)
            for j in range(int(4 * k) + 1):
                sc.wisp(c, f * 2 + j * 7, kx + j - 1, ky, int(18 + 22 * k), seed=40 + j,
                        cols=('#ffffff', '#fbf6ea', '#f0e6d4', '#e2d6c2'), drift=6 + 3 * j, curl=1.8, width=2)
            ui.burst(c, t, tk, kx, ky - 2, ('white', 'cloud', 'cotton'), n=22, seed=11, power=0.55, life=46,
                     gravity=-0.03)
            if a < 40 and (a // 3) % 2 == 0:
                c.px(kx - 1, ky - 1, 'white')
        tb = self.spot_t.get('bell')
        if tb is not None and 0 <= t - tb < 70:
            a = t - tb
            bx, by = BELL[0] - cx, BELL[1] + 8
            for r0 in (0, 10, 20):
                r = 5 + (a - r0) * 0.9
                if 0 <= a - r0 < 34:
                    cc = 'butter' if a - r0 < 18 else '#e8c890'
                    c.circle(bx, by, int(r), cc, fill=False)
            if a < 30:
                sh = 1 if (a // 2) % 2 else -1     # the clapper strikes
                c.px(bx + sh * 3, by + 4, 'white')
                c.px(bx + sh * 4, by + 3, 'butter')
            ui.burst(c, t, tb, bx, by + 2, ('gold', 'butter', 'white'), n=18, seed=13, power=0.9, life=36)
        self.keep_glow(c, _kept(rec, c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        return c

    @staticmethod
    def _bunting(c, f, off):
        """The film's two rows of bunting across the top of the alley (render_b), strung every
        BUNT_STEP px along the street on a layer a little in front of the stalls."""
        cols_ = ['#b8483a', '#efe2c4', '#4e8c84', '#d4a640', '#7e4a5e']
        for k in range(BUNT_N):
            x0b = 56 + k * BUNT_STEP + off
            x1b = x0b + 206
            if x1b < 0 or x0b >= W:
                continue
            for (y0, y1, seedb) in ((6, 14, 1 + k), (18, 12, 2 + k)):
                for i in range(206):
                    x = x0b + i
                    if not 0 <= x < W + 4:
                        continue
                    u = i / 206
                    y = int(round(y0 + (y1 - y0) * u + 5 * math.sin(math.pi * u)))
                    c.px(x, y, '#4a3a34')
                    if i % 8 == 0 and 2 < i < 202:
                        cc = cols_[(i // 8 + seedb) % 5]
                        flap = (f // 4 + i // 8) % 2
                        c.rect(x, y + 1, 4, 2, cc)
                        c.rect(x + 1, y + 3, 2, 1 + flap, cc)
                        c.px(x + 3, y + 1, shade(col(cc), 0.8))

    # -------------------------------------------------------------- area 2: docks + sea of clouds
    def render_docks(self, f, cx, cy):
        t = self.g.t
        A, dx, dy = self.ship_state(f, t)

        def seg_(f_, a, b):
            return 0.0 if a == A else 1.0       # ship clock 0 (path from ease_*); film balloon 'gone'

        hour = self.g.hour
        k = _lamp_k(hour)                        # the clock's lights (see the world clock above)
        dock = sc._dock_c()
        dock_lay, dock_win = _lamp_layer('dock')
        rec = []                                 # the lit pixels, kept lit by GlowKeep
        lfx = _LampFx()
        dq = _q(_day(hour), 10)
        on = k > 0.35                            # the airship's portholes and cabin

        def comp(c, layer, x, y=0):
            if layer is dock[0]:                 # the pillar's windows unlit, lit one by one by the clock
                layer = dock_lay
            sc.comp(c, layer, x, y)
            if layer is dock_lay:
                _light_windows(c, dock_win, k, x, 0, rec)

        render = rebind(sc.render_c, cam_c=lambda _f: cx, draw_balloon=_nop, callout=_nop, comp=comp,
                        AIRSHIP=A, seg=seg_, ease_in=lambda _t: dx / 66.0, ease_in_out=lambda _t: dy / 14.0,
                        _peaks_c=_wide_peaks, _stratus_a=lambda: _cloud_day('s', dq),
                        _cloud_band=lambda lvl: _cloud_day(lvl, dq), _ship=_ship_fn(on), fx=lfx,
                        hash01=_flicker_fn(k), draw_lantern=_lantern_fn(k, _halo(hour), lfx, rec))
        c = render(f)
        bob = int(round(math.sin((f - A) * 2 * math.pi / 60)))
        shx, shy = sc.SHIP_X0 + dx - cx, sc.SHIP_Y0 - dy + bob
        if on:
            ys, xs, rgb = _ship_lights()
            sy, sx = ys + shy, xs + shx
            ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
            rec.append((sy[ok], sx[ok], rgb[ok]))
        if k > 0:                                # a window flickering on
            _rec_px(c, rec, _flicker_px(dock[1].get('windows', []), -cx), ('#e8a040',))
        mode, t0 = self.ship
        if mode == 'moored' and 0 <= t - t0 < PUFF:          # makes fast: steam from the gondola
            k = (t - t0) / PUFF
            for i in range(7):
                a_ = i * 0.9
                px_ = shx + 50 + int(math.cos(a_) * k * 22)
                py_ = shy + 46 + int(math.sin(a_) * k * 8) + int(k * 4)
                if (i + f) % 3 or k < 0.6:
                    c.circle(px_, py_, 2 + int(k * 6), '#f4ecdc' if k < 0.5 else '#dccdb8')
        th = self.spot_t.get('airship')
        if th is not None and 0 <= t - th < 40 and mode != 'away':     # the toot: steam from the stack
            ax, ay = sc._ship(1)[1]['stack']
            ui.burst(c, t, th, shx + ax, shy + ay - 2, ('white', 'cloud', '#e8dccb'), n=16, seed=21,
                     power=0.45, life=40, gravity=-0.04)
        self.keep_glow(c, _kept(rec, c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        return c
