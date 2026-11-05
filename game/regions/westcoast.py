"""Vanhoover and the west coast as an explorable diorama (art: scenes/s17_westcoast.py, used read-only).

A cozy rainy night on the north-west coast. The film shows three different views joined by night
cloud passes, so the region has three areas, flown through left to right with a cloud sweep between
them, in film order:
  0  VANHOOVER and TALL TALE (shot A): pale glass towers with lit window grids on the misty bay, the
     green suspension bridge from the pine headland, the sail-roof pavilion, Tall Tale's warm lights up
     the coast; the wet seawall promenade with its lamps in the foreground. The film pans 58 px; here
     the coast runs on east of Tall Tale (the film's city layer gets more wooded hills, a cove of
     cottages and shore lights; the Unicorn Range, the sky glow and the promenade are rebuilt wider
     with the scene's own code), so the balloon has the whole bay.
  1  THE SMOKEY MOUNTAINS (shot B): the twin peaks with the log fortress and the cabins, chimney
     smoke, waving banners, the valley of replanted saplings with its creek, footbridge and lanterns.
     The valley floor is rebuilt wider (the film's builder, longer) so the camera can run between
     the two big framing pines.
  2  SILVER SHOALS and LUNA BAY (shot C): the stilt town on its rocky hill, the boardwalk, moored
     sailboats, the striped lighthouse with its sweeping beam, Luna Bay out to the horizon.
The film's shot renderers draw their content per shot, on film frames, with the film's balloon (and
its burner glow and mooring rope), so every area re-composes the scene's own cached layers and
helpers at the player's camera (recompose style, see sweetapple.py / badlands.py), in the film's
layer order, without the film's balloon, callouts, card, iris or cloud passes.

Timed film beats on game clocks:
  - both ferries cross the bay on endless loops (the big one right to left in front, the little one
    far out the other way), each restarting beyond every camera;
  - the far sailboat drifts across Luna Bay on a loop; the rain clouds over Luna Bay sway instead of
    drifting off for good;
  - THE MOONRISE (film 280..344): every landing starts under the rain clouds. Flying out east to
    Luna Bay (or SPACE at the hidden moon) plays it: the clouds part, the moon breaks out with its
    bloom, the silver path runs across the sea, the rain thins and the stars come out. SPACE at the
    moon afterwards rolls the clouds back in (the drizzle returns), and SPACE again replays it.
Things to set off with SPACE: the hero glass tower lights every window of Vanhoover in a sweep (the
bay mirrors it); the big ferry (the bubble rides on it) sounds its horn with a steam plume and its
decks blazing; the valley lantern by the footbridge lights the lanterns of the Smokey Mountains in a
chain, from the valley up to the fortress and the cabins, and paper sky lanterns rise; the lighthouse
flashes and its beam spins a full turn; the moon (the moonrise, or the clouds rolling back).
Hidden stamps: low under the suspension bridge's deck, high above the fortress's west watchtower
flag, and down under the Silver Shoals boardwalk among the stilts.
The balloon is lit like the night (balloon_tint): bluish in the rain, warm beside the promenade lamps
and the valley lanterns, silver once the moon is out. No creatures anywhere (the edition's rule).
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, smooth, hash01, col, mix, shade, clamp, lerp, rng
from engine import ease_out, ease_in_out
import scenes.s17_westcoast as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# ================================================================== area 0: Vanhoover (shot A)
# layers: sky + glow (p .6 here), cloud decks (tile), Unicorn Range 360 (p .35, +10) -> cam 85,
# city 400 (p .6) -> 133, seawall 480 (p 1.6) -> 100. All three are rebuilt wider for CAM_A.
CAM_A = 200
P_RANGE, P_CITY, P_FAR_FERRY, P_FG = 0.35, 0.6, 0.75, 1.6
YW_A = sc.YW_A
CITY_W = W + int(math.ceil(CAM_A * P_CITY)) + 4
RANGE_W = 10 + W + int(math.ceil(CAM_A * P_RANGE)) + 4
FG_W = W + int(math.ceil(CAM_A * P_FG)) + 4
SKY_W = W + int(math.ceil(CAM_A * P_CITY)) + 2
LAMPS_A = sc.A_LAMPS + (526, 636)               # promenade lamps (fg px): the film's three + two more
HERO = sc.TOWERS[5]
HERO_X = HERO[0] + HERO[1] // 2                 # hero glass tower centre (city px) = 204
TOWER_SPOT = (HERO_X, 66)                       # SPACE here: Vanhoover lights up (on its body)
TALL_TALE_X = 326
TOWER_T = 110                                   # the window sweep: out 0..28, hold, fade 72..106
# ferries (world px of the hull's left end; the big one on the main layer, the little one p .75)
BIG_V, BIG_X0, BIG_X1 = 0.85, CAM_A + W + 4, -92        # right -> left, wake trailing right
BIG_P = int(math.ceil((BIG_X0 - BIG_X1) / BIG_V))
BIG_LAND = 470                                  # where the big ferry is when you land (enters soon)
SMALL_V, SMALL_X0, SMALL_X1 = 0.45, -44, int(math.ceil(CAM_A * P_FAR_FERRY)) + W + 2
SMALL_P = int(math.ceil((SMALL_X1 - SMALL_X0) / SMALL_V))
FERRY_YB = 137
FERRY_T = 80                                    # horn: plume + blazing decks

# ================================================================== area 1: Smokey Mountains (shot B)
# far range 420 (p .3, +20) -> 266; peaks 420 (p .7, -6) -> 151; valley 420 (p 1, +14) -> 86 (rebuilt
# wider here); fg 520 (p 1.35) -> 148
CAM_B = 148
P_FAR_B, P_PEAKS, P_FG_B = 0.3, 0.7, 1.35
VO = 14                                         # valley px = world px + 14
VB_W = VO + W + CAM_B + 6
LANTERN_T = 120                                 # the lantern chain's cooldown
LANTERN_LIFE = 290                              # its flares + the sky lanterns

# ================================================================== area 2: Silver Shoals / Luna Bay (shot C)
# far headlands 360 (p .15, +10) -> 200; town 420 (p .7, +6) -> 134; dock 460 (p 1.3, +4) is open
# water right of x 200; sky, sea and puffs are screen-wide
CAM_C = 134
P_FAR_C, P_TOWN, P_DOCK, P_PUFF, P_MOON = 0.15, 0.7, 1.3, 0.2, 0.1
TO = 6                                          # town px = screen px + 0.7 cam + 6
HZ, YW_C = sc.HZ, sc.YW_C
T_MOON = sc.T_MOON
RISE_F0 = 280                                   # the moonrise clock starts 8 frames before `moonrise`
RISE_END = 350
MOON_T = 40                                     # the moon hotspot's cooldown
CLOSE_T = 96                                    # the clouds rolling back in (eased out: fast, then settling)
RERISE_T = 60                                   # the clouds parting again from part-way
MOON_AUTO_X = 330                               # first time this far east: the moon rises by itself
LIGHT_T = 60                                    # lighthouse flash + one extra turn of the beam
FAR_BOAT_V, FAR_BOAT_X0, FAR_BOAT_X1 = 0.12, -14, 362    # the far sailboat (p .3 px)
FAR_BOAT_P = int(math.ceil((FAR_BOAT_X1 - FAR_BOAT_X0) / FAR_BOAT_V))

TINT_RAIN = ('#6a76b0', 0.5)
TINT_LAMP = ('#ffc890', 0.32)
TINT_MOON = ('#c8d2f2', 0.28)


_SPLIT = {}


def _split(rgba):
    """(rgb, hard-alpha mask) of a cached RGBA layer, kept for fast pasting."""
    v = _SPLIT.get(id(rgba))
    if v is None or v[0] is not rgba:
        v = (rgba, np.ascontiguousarray(rgba[..., :3]), rgba[..., 3:4] >= 128)
        _SPLIT[id(rgba)] = v
    return v[1], v[2]


def _paste(c, rgba, x, y=0):
    """sc.paste (Canvas.blit of an RGBA layer at int(x), int(y)), ~4x faster: masks are cached."""
    rgb, m = _split(rgba)
    x, y = int(x), int(y)
    sh, sw = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + sw), min(H, y + sh)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0:y1, x0:x1], rgb[y0 - y:y1 - y, x0 - x:x1 - x], where=m[y0 - y:y1 - y, x0 - x:x1 - x])


def _tile_paste(c, rgba, x, y):
    """sc.tile_paste with _paste."""
    w = rgba.shape[1]
    x = int(x) % w
    _paste(c, rgba, x - w, y)
    _paste(c, rgba, x, y)
    if x + w < W:
        _paste(c, rgba, x + w, y)


class _FerrySpot(Hotspot):
    """The big ferry's hotspot: it rides on the ferry's funnel, and hides while the ferry is out of sight
    beyond the ends of the bay (x is computed from the ferry's clock)."""

    def __init__(self, region, cooldown):
        super().__init__('ferry', 0, FERRY_YB - 16, 'horn', r=48, cooldown=cooldown)
        self.region = region

    @property
    def x(self):
        fxw = self.region.big_x(self.region.g.t) + 30
        return int(fxw) if -20 < fxw < CAM_A + W + 30 else -10 ** 4

    @x.setter
    def x(self, _v):
        pass


def _tint_between(a, b, k):
    k = clamp(round(k * 20) / 20)                # quantised: the balloon sprite cache stays small
    return (mix(a[0], b[0], k), round(lerp(a[1], b[1], k), 3))


def _no_sky(y):
    """A blank sky-hook canvas whose rows from y down can never match a frame (sentinel colour)."""
    c = Canvas(W, H)
    c.a[y:] = (255, 0, 255)
    return c


_INV = {}


def _hide(c, rgba, x, y):
    """Paste a scenery layer's opaque pixels into a sky hook with inverted colours (never sky)."""
    v = _INV.get(id(rgba))
    if v is None or v[0] is not rgba:
        v = _INV[id(rgba)] = (rgba, np.dstack([255 - rgba[..., :3], rgba[..., 3:]]))
    _paste(c, v[1], x, y)


def _tile_hide(c, rgba, x, y):
    """_tile_paste of a layer's inverted colours: tiled rain decks that must never count as sky."""
    w = rgba.shape[1]
    x = int(x) % w
    for xx in (x - w, x, x + w):
        if xx < W:
            _hide(c, rgba, xx, y)


# ------------------------------------------------------------------ the lights by the world clock
# The film paints the coast at night with every light on. By day the re-light (game/daylight.py) switches off
# whatever is warm and bright: here that is also the yellow and peach stilt-house walls, the pink hulls and the
# flowers, and at dusk it dims the lamps while it is still dark. So the region switches its own lights: the
# film's light colours in small spots of its layers (_spots), by the clock's ramps (WestCoast.ramps: windows on
# nightwin's ramp, lamps / lanterns / beacons a little earlier and later), each spot at its own threshold so they
# come on one by one; WestCoast.after_light undoes the re-light's switch-off, and halos, glows and the beam are
# drawn onto the re-lit picture (WestCoast._post), so none is left as a night-coloured blot on the clock's sky.
W_DARK = '#2a2030'                  # the film's unlit fortress window
GLASS_OFF = '#3a4262'               # an unlit lamp's glass
PAPER_OFF = '#7a5844'               # an unlit paper lantern
FERRY_OFF = '#6a7a98'               # a ferry's windows by day


def _city_table():
    """The city layer's light colours -> (unlit colour, lamp?) (the shaded tower windows too)."""
    t = {sc.W_WARM: (sc.W_OFF, False), sc.W_WARM2: (sc.W_OFF, False), sc.W_COOL: (sc.W_OFF2, False),
         sc.W_TEAL: (sc.W_OFF2, False), '#fff4d0': ('#5f9488', True)}
    for sh in (sc.G_SH, '#58698f'):
        for lc, (oc, _) in list(t.items())[:4]:
            t[mix(lc, sh, 0.35)] = (mix(oc, sh, 0.35), False)
    return t


SPOTS = {   # layer key -> (getter, {light colour: (unlit colour, lamp?)}, largest spot in px)
    'city': (lambda: _city_a()[0], _city_table, 20),
    'seawall': (lambda: _seawall_a(), lambda: {'#ffe7a8': (GLASS_OFF, True), '#fff6dc': (GLASS_OFF, True)}, 25),
    'peaks': (lambda: sc.peaks_b()[0], lambda: {sc.W_WARM: (W_DARK, False), sc.W_WARM2: (W_DARK, False),
                                               '#fff2c8': (W_DARK, False), '#ffcf70': (PAPER_OFF, True)}, 16),
    'town': (lambda: sc.town_c()[0], lambda: {sc.W_WARM: (sc.W_OFF, False), '#fff2c8': (sc.W_OFF, False),
                                             '#ffcf70': (PAPER_OFF, True), '#fff2c0': (GLASS_OFF, True)}, 16),
    'farc': (lambda: sc.far_c(), lambda: {'#ffd98a': ('#3a4a6e', True)}, 2),
    'dock': (lambda: sc.dock_c(), lambda: {'#ffcf70': (PAPER_OFF, True)}, 16),
}


@lru_cache(maxsize=None)
def _spots(key):
    """The light spots of a layer: (ys, xs, threshold, unlit rgb, lamp?) per pixel. A spot is a small group of
    light-coloured pixels (a window, a lamp's glass, a bulb); a big one is a painted wall, never a light. The
    city's bridge bulbs, deck lights and shore lights are lamps (on with the lamps), its windows windows."""
    from scipy import ndimage
    get, table, nmax = SPOTS[key]
    rgba = get()
    tab = {tuple(col(k)): v for k, v in table().items()}
    rgb = rgba[..., :3]
    q = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    qs = {(r << 16) | (g << 8) | b: v for (r, g, b), v in tab.items()}
    m = (rgba[..., 3] >= 128) & np.isin(q, list(qs))
    lab, n = ndimage.label(m, np.ones((3, 3)))
    ys, xs = np.nonzero(m)
    ls = lab[ys, xs]
    size = np.bincount(ls, minlength=n + 1)
    boxes = ndimage.find_objects(lab)
    ok = np.zeros(n + 1, bool)
    v = np.zeros(n + 1, np.float32)
    for i, s in enumerate(boxes, 1):
        hh, ww = s[0].stop - s[0].start, s[1].stop - s[1].start
        ok[i] = size[i] <= nmax and hh <= 6 and ww <= 16
        v[i] = hash01(s[1].start, s[0].start, 611)
    keep = ok[ls]
    ys, xs, ls = ys[keep], xs[keep], ls[keep]
    off = np.array([col(qs[int(k)][0]) for k in q[ys, xs]], np.uint8).reshape(-1, 3)
    lamp = np.array([qs[int(k)][1] for k in q[ys, xs]], bool)
    if key == 'city':                                   # bridge, deck and shore lights go on with the lamps
        lamp |= ((xs < 128) & (ys <= 106)) | (ys >= YW_A - 6) & (xs >= 262)
    return ys, xs, v[ls] * 0.94, off, lamp


def _switch(c, key, x, y, kw, kl):
    """After pasting layer `key` at (x, y): its lights not on yet at the window / lamp ramps kw / kl get their
    unlit colour."""
    if kw >= 1 and kl >= 1:
        return
    ys, xs, v, off, lamp = _spots(key)
    sel = v >= np.where(lamp, kl, kw)
    sy, sx = ys[sel] + int(y), xs[sel] + int(x)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    c.a[sy[ok], sx[ok]] = off[sel][ok]


def _mark(c, key, x, y):
    """In a sky hook: the layer's light spots never count as sky (an unlit window can come close to the
    inverted lit one _hide leaves there)."""
    ys, xs = _spots(key)[:2]
    sy, sx = ys + int(y), xs + int(x)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    c.a[sy[ok], sx[ok]] = (255, 0, 255)


@lru_cache(maxsize=2)
def _ferry_lights(size):
    a = sc._ferry(size)
    m = np.zeros(a.shape[:2], bool)
    for cc in (sc.W_WARM, sc.W_WARM2, sc.W_COOL):
        m |= np.all(a[..., :3] == np.array(col(cc), np.uint8), axis=2) & (a[..., 3] >= 128)
    return np.nonzero(m), a.shape[:2]


def _ferry_off(c, x, yb, size, flip=False):
    """A ferry's windows unlit (by day), as sc.draw_ferry placed the sprite."""
    (ys, xs), (h, w) = _ferry_lights(size)
    if flip:
        xs = w - 1 - xs
    sx, sy = xs + int(x), ys + int(yb) - h + (2 if size == 'l' else 1)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    c.a[sy[ok], sx[ok]] = col(FERRY_OFF)


# the rain decks (w, h, seed, colours, n): by day they are drawn grey (rain clouds in daylight, not night
# clouds lifted), mixed toward DAY_GREY by the world clock's daylight dk (0 night .. DK day)
DECK_A0 = (480, 40, 3, ('#141a30', '#1b2340', '#2a3452', '#3c4766'), 46)
DECK_A1 = (420, 30, 5, ('#1a2038', '#232c4a', '#343e5e', '#4a5576'), 30)
DECK_B0 = (480, 40, 13, ('#141a30', '#1b2340', '#283250', '#38445f'), 46)
DECK_B1 = (420, 26, 15, ('#1d2440', '#252f4c', '#34405e', '#48557a'), 24)
PUFF_C = ('#1d2443', '#2a3456', '#4a5882', '#8d9cc8')
DAY_GREY, DAY_MIX = '#727a8e', 0.58
DK = 8                                             # day_k steps (night 0 .. day DK)
# the water by day (as the re-lit frame should show it under the clock's overcast noon: a grey-blue bay,
# not the night's navy); Badlands-style: the region paints the colour daylight.unlight gives for it
DAY_SEA_A = ('#6d8aa6', '#4a6684')                 # Vanhoover's bay: reflection band top / bottom
DAY_SEA_C = ('#7894ae', '#6c88a3', '#607c98')      # Luna Bay's open sea, horizon -> waterline
DAY_REFL_C = ('#62809b', '#46627e')                # the water under the town and the boats
DAY_RIPPLE = '#8fa8c0'


def _day_cols(cs, dk):
    return cs if dk <= 0 else tuple(mix(c, DAY_GREY, DAY_MIX * dk / DK) for c in cs)


def _deck(spec, dk):
    w, h, seed, cs, n = spec
    return sc.cloud_deck(w, h, seed, *_day_cols(cs, dk), n)


def _puff(sd, pw, ph, dk):
    return sc.cloud_puff(sd, pw, ph, *_day_cols(PUFF_C, dk))


def _puffs(f, cx, P):
    """The Luna Bay rain puffs: (seed, w, h, x, y) at film frame f, camera cx, moon params P."""
    k, kf = P['k'], P['kf']
    sway = 10 * math.sin(f * 0.0021)            # the deck sways in place (the film's drifts away)
    for (px, py, pw, ph, sd) in sc.C_PUFFS:
        pc = px + pw / 2 - sc.MOON[0]
        dirn = 1 if pc > 0 else -1
        amt = k * (0.5 + 0.9 * hash01(sd, 4)) + (kf * 1.2 if abs(pc) < 60 else 0)
        yield (sd, pw, ph, int(round(px + sway - cx * P_PUFF + dirn * amt)),
               int(round(py - k * 0.05 * hash01(sd, 6) * 8)))


@lru_cache(maxsize=4)
def _sky_mountains(cx):
    """WestCoast.sky_mountains at camera x cx."""
    c = _no_sky(130)
    c.gradient(0, 0, W, 130, sc.SKY_NIGHT, soft=0.7)
    moon_t = 0.35 + 0.65 * cx / CAM_B
    fx.glow(c, 292, 0, 110, '#8090b8', 0.32 * smooth(moon_t), rings=6)
    fx.glow(c, 292, 0, 80, '#a8b4dc', 0.26 * smooth(moon_t), rings=5)
    _hide(c, sc.far_b(), -int(cx * P_FAR_B) - 20, 0)       # the dark ranges and peaks are sky-dark
    _hide(c, sc.peaks_b()[0], -int(round(cx * P_PEAKS)) + 6, sc.PK_Y)
    _mark(c, 'peaks', -int(round(cx * P_PEAKS)) + 6, sc.PK_Y)
    return c


@lru_cache(maxsize=2)
def _far_c(day):
    """sc.far_c; by day without the checker haze over the open sky (it showed as a grey rectangle on the
    day sky): a haze pixel whose left and right neighbours are both empty lies outside the land."""
    a = sc.far_c()
    if not day:
        return a
    a = a.copy()
    on = a[..., 3] >= 128
    left, right = np.zeros_like(on), np.zeros_like(on)
    left[:, 1:], right[:, :-1] = on[:, :-1], on[:, 1:]
    a[..., 3][on & ~left & ~right] = 0
    return a


@lru_cache(maxsize=4)
def _sky_shoals(mt, mx, my, pulse, fo, day=False):
    """render_shoals' sky without the stars: (rgb, the light added over the gradient as int16, where a
    star shows: not on the moon disc nor under the far headland / sea)."""
    c = _no_sky(HZ + 1)
    sky = [mix(a, b, mt * 0.55) for a, b in zip(sc.SKY_NIGHT, ['#0e1430', '#141c3e', '#1a2650', '#22325e',
                                                               '#2b3f6c', '#364d78', '#425a82'])]
    c.gradient(0, 0, W, HZ + 1, sky, soft=0.7)
    grad = c.a.astype(np.int16)
    fx.glow(c, mx, my, 44, '#7a8cc0', 0.22 + 0.2 * mt, rings=6)
    disc = Canvas(arr=np.zeros((H, W, 3), np.uint8))
    for cv in (c, disc):
        cv.circle(mx, my, sc.MOON_R + 1, '#9fb0d6')
        cv.circle(mx, my, sc.MOON_R, '#eef3fb')
        cv.circle(mx + 2, my + 1, sc.MOON_R - 2, '#dde6f4')
        for (dx, dy, r) in [(-3, -1, 2.6), (-1, 2, 2.2), (1, 0, 1.6), (4, 5, 0.8)]:
            cv.circle(mx + dx, my + dy, r, '#c9d3e7')
        cv.px(mx - 6, my - 5, 'white')
        cv.px(mx - 5, my - 6, 'white')
    if 0 < pulse < 1:
        r = int(14 + 44 * ease_out(pulse))
        fx.glow(c, mx, my, r, '#c8d6ff', 0.4 * (1 - pulse), rings=4)
    _hide(c, _far_c(day), fo, 0)                            # the far headland and cape
    _mark(c, 'farc', fo, 0)
    c.a[HZ:] = (255, 0, 255)                                 # the sea starts on the horizon row
    hid = np.zeros((H, W), bool)
    fm = _far_c(day)[..., 3] >= 128
    x0, x1 = max(0, fo), min(W, fo + fm.shape[1])
    if x1 > x0:
        hid[:fm.shape[0], x0:x1] = fm[:H, x0 - fo:x1 - fo]
    free = ~disc.a.any(axis=2) & ~hid
    return c.a, c.a.astype(np.int16) - grad, free


# ------------------------------------------------------------------ area 0 builders
@lru_cache(maxsize=1)
def _sky_a():
    """The film's night sky (sky_a), wider, its city glows riding the city's parallax."""
    cv = Canvas(SKY_W, YW_A)
    cv.gradient(0, 0, SKY_W, YW_A, sc.SKY_NIGHT, soft=0.7)
    fx.glow(cv, 214, 112, 120, '#3a3a52', 0.55, rings=6)      # city glow on the low cloud
    fx.glow(cv, 60, 110, 60, '#2a3a50', 0.35, rings=4)
    fx.glow(cv, 336, 112, 50, '#3a3442', 0.35, rings=4)        # Tall Tale's warm haze
    return cv


@lru_cache(maxsize=1)
def _range_a():
    """The Unicorn Range behind Vanhoover (the film's range_a), with two more peaks to the east."""
    w = RANGE_W
    cv = sc.layer(w, YW_A)
    pts1 = [(0, YW_A)]
    rr = rng(41)
    peaks = [(10, 76), (48, 64), (80, 72), (118, 58), (150, 70), (186, 62), (222, 74), (258, 66),
             (300, 78), (330, 70), (360, 80), (392, 66), (424, 74)]
    for i, (px, py) in enumerate(peaks):
        pts1.append((px, py))
        if i < len(peaks) - 1:
            nx, ny = peaks[i + 1]
            pts1.append(((px + nx) / 2, max(py, ny) + 10 + sc.ri(rr, 0, 6)))
    pts1.append((w + 30, YW_A))
    cv.poly(pts1, '#28495a')
    for (px, py) in peaks:
        cv.tri((px - 5, py + 6), (px + 5, py + 6), (px, py), '#5d8394')
        cv.tri((px - 3, py + 4), (px, py + 4), (px, py), '#7fa5b2')
        cv.tri((px, py), (px + 6, py + 8), (px + 2, py + 9), '#3a6070')
    cv.dither(0, 96, w, 22, '#30566a', 0.5)
    pts2 = [(0, YW_A), (0, 100), (30, 92), (60, 98), (96, 88), (130, 96), (170, 90), (210, 98),
            (250, 92), (290, 100), (330, 94), (360, 99), (396, 91), (430, 97), (w, YW_A)]
    cv.poly(pts2, '#1d3846')
    for i in range(0, w, 5):
        h0 = 92 + int(6 * math.sin(i * 0.07))
        cv.vline(i, h0 + 4, h0 + 6, '#24434f')
    return sc.to_rgba(cv)


@lru_cache(maxsize=1)
def _city_a():
    """The film's Vanhoover skyline (city_a) with the coast running on east of Tall Tale: wooded hills,
    a cove of cottages with warm windows, shore lights. Returns (rgba, toggles, blinkers)."""
    film, toggles, blinkers = sc.city_a()
    base = YW_A - 1
    cv = sc.layer(CITY_W, YW_A)
    cv.poly([(386, base + 1), (386, 113), (404, 113), (420, 110), (434, 105), (448, 102), (462, 104),
             (CITY_W + 2, 106), (CITY_W + 2, base + 1)], '#1b3034')
    cv.poly([(408, base + 1), (416, 112), (430, 108), (444, 106), (458, 108), (CITY_W + 2, 111),
             (CITY_W + 2, base + 1)], '#20383a')
    for (hx_, hy, hw) in [(414, 112, 4), (421, 110, 5), (431, 108, 4), (439, 107, 5)]:
        cv.rect(hx_, hy - 3, hw, 4, '#4f566f')
        cv.tri((hx_ - 1, hy - 3), (hx_ + hw, hy - 3), (hx_ + hw / 2 - 0.5, hy - 6), '#6b4a5a')
        cv.px(hx_ + 1, hy - 2, sc.W_WARM)
        if hw > 4:
            cv.px(hx_ + 3, hy - 2, sc.W_WARM2)
    rr = rng(617)
    for i in range(12):
        px = int(rr.uniform(392, CITY_W))
        if 410 < px < 446:
            continue
        sc.pine(cv, px, base + 1 - int(max(0, 8 - abs(px - 452) * 0.2)), sc.ri(rr, 6, 12),
             '#1d3a39', '#15302f', '#2e554f', '#0f2124')
    for xx in range(376, CITY_W, 9):
        cv.px(xx, base - 1, sc.W_WARM2)
    a = sc.to_rgba(cv)
    m = film[..., 3] > 0
    a[:, :sc.WA][m] = film[m]
    return a, toggles, blinkers


@lru_cache(maxsize=1)
def _dark_windows():
    """Unlit glass-tower windows of the city layer: (ys, xs, distance from the hero tower, colours)."""
    a = _city_a()[0]
    rgb = a[..., :3].astype(np.int16)
    m = a[..., 3] > 0
    targets = [sc.W_OFF, sc.W_OFF2]
    for sh in (sc.G_SH, '#58698f'):
        targets += [mix(sc.W_OFF, sh, 0.35), mix(sc.W_OFF2, sh, 0.35)]
    sel = np.zeros(m.shape, bool)
    for tc in targets:
        sel |= m & (np.abs(rgb - np.array(col(tc), np.int16)).sum(axis=2) <= 3)
    ys, xs = np.nonzero(sel)
    d = np.hypot(xs - HERO_X, (ys - TOWER_SPOT[1]) * 1.3)
    cell = (xs // 3) * 7919 + (ys // 3) * 104729
    h = (cell * 2654435761 % 1000) / 1000.0
    pal = np.array([col(sc.W_WARM), col(sc.W_WARM2), col(sc.W_COOL), col(sc.W_TEAL)], np.uint8)
    idx = np.where(h < 0.55, 0, np.where(h < 0.78, 1, np.where(h < 0.94, 2, 3)))
    return ys, xs, d, pal[idx], h


def _more_seawall(cv):
    """More promenade furniture east of the film's (the film's own bollard / planter / bench shapes)."""
    for bx in (566, 610):
        cv.rect(bx, 146, 5, 6, '#1d2236')
        cv.rect(bx + 1, 145, 3, 1, '#58648a')
        cv.vline(bx + 1, 146, 150, '#45507a')
    for px_ in (468, 586):
        cv.rect(px_, 163, 18, 9, '#3c2f3e')
        cv.rect(px_ + 1, 164, 16, 7, '#5a4255')
        cv.hline(px_ + 1, px_ + 16, 164, '#7d5a70')
        cv.rect(px_ + 2, 172, 14, 1, '#141827')
        for k in range(8):
            fx_ = px_ + 2 + k * 2
            cv.vline(fx_, 159 - (k % 3), 163, '#2f5a46')
            cv.px(fx_, 158 - (k % 3), ['#ff9ec8', '#ffd4e6', '#c9a8ff', '#fff2c8'][k % 4])
            cv.px(fx_ + 1, 160 - (k % 2), '#3f7a5a')
    bx = 560
    cv.rect(bx, 163, 22, 2, '#5a3f3a')
    cv.hline(bx, bx + 21, 163, '#8a6450')
    cv.rect(bx, 158, 22, 2, '#5a3f3a')
    cv.hline(bx, bx + 21, 158, '#8a6450')
    for lx in (bx + 2, bx + 18):
        cv.rect(lx, 165, 2, 6, '#1d2236')
        cv.rect(lx, 157, 1, 8, '#1d2236')
    return sc.to_rgba(cv)


@lru_cache(maxsize=1)
def _seawall_a():
    """The film's promenade (seawall_a) built FG_W wide, with two more lamps and more furniture."""
    return rebind(sc.seawall_a.__wrapped__, FGW_A=FG_W, A_LAMPS=LAMPS_A, to_rgba=_more_seawall)()


# ------------------------------------------------------------------ area 1 builders
def _creek_x(yy):
    t = (yy - 128) / (H - 128)
    return 214 + math.sin(t * 4.2) * 18 * t + t * 30


@lru_cache(maxsize=1)
def _valley_b():
    """The film's valley floor (valley_b) built VB_W wide: the meadow runs on east of the film's."""
    w = VB_W
    cv = sc.layer(w, H)
    cv.poly([(-10, H), (-10, 150), (60, 142), (130, 134), (180, 128), (220, 126), (260, 128),
             (320, 136), (380, 144), (430, 150), (w + 10, 151), (w + 10, H)], '#1e3a2c')
    cv.gradient(-10, 126, w + 20, H - 126, ['#24432f', '#1d3a2a', '#16301f'], soft=0.6)
    a = cv.a
    for x in range(w):
        if x <= sc.WB:
            u = x / sc.WB
            ytop = int(128 + 22 * (2 * u - 0.95) ** 2 * 0.9)
        else:
            ytop = int(149.8 + 2.2 * math.sin((x - sc.WB) * 0.045))
        a[:max(0, ytop), x] = sc._KEYA
        cv.px(x, ytop, '#3a6a48')
    pts_l, pts_r = [], []
    for yy in range(128, H + 1):
        t = (yy - 128) / (H - 128)
        cxw = _creek_x(yy)
        hw = 1 + t * 9
        pts_l.append((cxw - hw, yy))
        pts_r.append((cxw + hw, yy))
    cv.poly(pts_l + pts_r[::-1], '#1a2c46')
    for (x0, y0), (x1, y1) in zip(pts_l, pts_r):
        cv.px(x0, y0, '#4a6a5a')
        cv.px(x1, y1, '#2a4a3a')
    bcx = _creek_x(158)
    bx0, bx1 = int(bcx - 13), int(bcx + 13)
    for x in range(bx0, bx1 + 1):
        u = (x - bx0) / (bx1 - bx0)
        yy = int(round(158 - 4 * math.sin(u * math.pi)))
        cv.vline(x, yy, yy + 1, '#5a3e2e')
        cv.px(x, yy, '#94664a')
        if (x - bx0) % 4 == 0:
            cv.vline(x, yy - 4, yy - 1, '#4e3426')
        cv.px(x, yy - 4, '#7a5a40')
    for x in (bx0, bx1):
        cv.rect(x - 1, 157, 2, 6, '#3a281e')
    # a stile and a woodpile on the meadow east of the film's valley
    for (x, y) in ((402, 152), (470, 153)):
        cv.rect(x, y - 4, 9, 4, '#5a3e2e')
        for k in range(3):
            cv.circle(x + 1 + k * 3, y - 3, 1.2, '#94664a')
            cv.px(x + 1 + k * 3, y - 3, '#b08a64')
    return sc.to_rgba(cv), (bx0, bx1)


@lru_cache(maxsize=1)
def _lanterns_b():
    bx0, bx1 = _valley_b()[1]
    return ((150, 150), (bx0 - 4, 156), (bx1 + 6, 156), (290, 146), (372, 150), (452, 152))


@lru_cache(maxsize=1)
def _saplings_b():
    """The film's sapling rows (saplings_b), planted on across the wider valley."""
    out = []
    rows = [(132, 9, 1), (138, 10, 1), (145, 11, 2), (153, 12, 2), (162, 14, 3), (172, 16, 3)]
    for r, (y, sp, sz) in enumerate(rows):
        x = 100 + (r * 5) % sp
        while x < VB_W - 30:
            t = (y - 128) / (H - 128)
            ccx = _creek_x(y)
            ok = abs(x - ccx) > 4 + t * 12 and hash01(x, y) > 0.12
            if x > 340:                       # the new rows: thinner, and only on the meadow
                ytop = 150 + 2.2 * math.sin((x - sc.WB) * 0.045) if x > sc.WB else \
                    128 + 22 * (2 * x / sc.WB - 0.95) ** 2 * 0.9
                ok = ok and y > ytop + 6 and hash01(x, y, 11) > 0.3
            if ok:
                out.append((x, y, sz, hash01(x, y, 3)))
            x += sp
    return tuple(out)


SAP_PER, SAP_Y = 57, 120


@lru_cache(maxsize=None)
def _sapling_layer(ph):
    """Every sapling at sway phase ph (0..56; the film's sway, sin(f * 0.11), has a 57-frame period),
    drawn by the film's draw_sapling into one layer (valley px, rows SAP_Y..H): one paste per frame."""
    cv = sc.layer(VB_W, H - SAP_Y)
    fe = ph * 2 * math.pi / (SAP_PER * 0.11)
    for (x, y, sz, p) in _saplings_b():
        sc.draw_sapling(cv, x, y - SAP_Y, sz, p, fe)
    return sc.to_rgba(cv)


# ------------------------------------------------------------------ area 2 helpers
def _fm_params(fm):
    """The film's moonrise curves at film frame fm (render_c)."""
    return dict(mt=smooth(seg(fm, T_MOON, T_MOON + 24)),
                rise=int(round(4 * (1 - smooth(seg(fm, T_MOON, T_MOON + 20))))),
                k=ease_out(seg(fm, T_MOON - 4, T_MOON + 56)) * 110,
                kf=ease_out(seg(fm, T_MOON - 6, T_MOON + 8)) * 80,
                pulse=seg(fm, T_MOON, T_MOON + 18),
                reach=HZ + 2 + (H - HZ) * ease_out(seg(fm, T_MOON, T_MOON + 22)))


CLOUDED = _fm_params(RISE_F0)
CLEAR = _fm_params(RISE_END)


# ================================================================== the region
class WestCoast(Region):
    SEG = 'westcoast'
    CARD = ('NORTHWEST COAST', ['RAIN - PINES - HARBOURS'])

    def warm(self):
        for dk in range(DK + 1):                 # the rain decks and puffs at every daylight step, built at
            for spec in (DECK_A0, DECK_A1, DECK_B0, DECK_B1):       # landing: dusk changed the step
                _deck(spec, dk)                                      # and stalled ~15 ms each time
            for (_px, _py, pw, ph, sd) in sc.C_PUFFS:
                _puff(sd, pw, ph, dk)
        super().warm()

    def build_areas(self):
        hx, hy = sc.town_c()[3][1]                  # Silver Shoals: the blue stilt house's roof apex
        lantern = _lanterns_b()[1]
        self.lantern_spot = (lantern[0] - VO, lantern[1] - 13)
        van = [
            Landmark('VANHOOVER', 232, (HERO_X, 22, 'right', 16), p=P_CITY),
            Landmark('TALL TALE', 432, (TALL_TALE_X, 100, 'up', 14), p=P_CITY),
        ]
        smokey = [Landmark('SMOKEY MOUNTAINS', 176, (140 + 6, 26 + sc.PK_Y, 'right', 16), p=P_PEAKS)]
        shoals = [
            Landmark('SILVER SHOALS', 150, (hx - TO, hy - 1, 'up', 16), p=P_TOWN),
            Landmark('LUNA BAY', 336, (192, HZ - 1, 'up', 22), p=P_FAR_C),
        ]
        self.ferry_spot = _FerrySpot(self, FERRY_T)
        self.moon_spot = Hotspot('moon', sc.MOON[0], sc.MOON[1] + 2, 'moon', r=40, cooldown=MOON_T,
                                 p=P_MOON)
        return [
            Area(self.render_vanhoover, cam_x=(0, CAM_A), cam_y=(0, 0), y_range=(30, 148), start=(112, 82),
                 landmarks=van, balloon='m', balloon_tint=TINT_RAIN,
                 hotspots=[Hotspot('tower', TOWER_SPOT[0], TOWER_SPOT[1], 'lantern', r=40,
                                   cooldown=TOWER_T, p=P_CITY),
                           self.ferry_spot],
                 stamps=[Stamp('bridge', 84, 112)], sky=self.sky_vanhoover),
            Area(self.render_mountains, cam_x=(0, CAM_B), cam_y=(0, 0), y_range=(20, 168), start=(60, 96),
                 landmarks=smokey, balloon='m', balloon_tint=TINT_RAIN,
                 hotspots=[Hotspot('lanterns', self.lantern_spot[0], self.lantern_spot[1], 'lantern',
                                   cooldown=LANTERN_T)],
                 stamps=[Stamp('fortress', 58, 8)], sky=self.sky_mountains),
            Area(self.render_shoals, cam_x=(0, CAM_C), cam_y=(0, 0), y_range=(28, 146), start=(56, 70),
                 landmarks=shoals, balloon='m', balloon_tint=TINT_RAIN,
                 hotspots=[Hotspot('lighthouse', 330 - TO, 76, 'spark', r=40, cooldown=LIGHT_T, p=P_TOWN),
                           self.moon_spot],
                 stamps=[Stamp('boardwalk', 182, 124)], sky=self.sky_shoals),
        ]

    # ------------------------------------------------------------------ state
    def init_state(self):
        self.big_t0 = 0
        self.small_t0 = 0
        self.moon = (-1, -10 ** 6, CLOUDED)     # (direction, step, sky then): +1 the clouds part, -1 close
        self.moon_auto = False
        self.area_t0 = 0
        self.sap_warm = 0           # sapling sway layers built so far (one per step, ahead of area 1)
        self.lantern_ts = []        # steps the lantern chain was set off (still showing)

    def on_enter(self):
        t = self.g.t
        self.big_t0 = t - int((BIG_X0 - BIG_LAND) / BIG_V)
        self.small_t0 = t - int((30 - SMALL_X0) / SMALL_V)
        self.moon = (-1, -10 ** 6, CLOUDED)
        self.moon_auto = False
        self.moon_spot.icon = 'moon'

    def on_area(self, ai):
        self.area_t0 = self.g.t
        g = self.g
        if ai == 0:
            g.sfx('rain', 0.4, 1)
        elif ai == 1:
            g.sfx('wind', 0.4, 0)
        else:
            g.sfx('rain', 0.35, 0)

    # ---- clocks
    def big_x(self, t):
        return BIG_X0 - BIG_V * ((t - self.big_t0) % BIG_P)

    def small_x(self, t):
        return SMALL_X0 + SMALL_V * ((t - self.small_t0) % SMALL_P)

    def moon_fm(self, t):
        """Film frame of the moonrise when it plays from the clouded sky (the film's own curves), else None."""
        d, t0, P0 = self.moon
        if d > 0 and P0 is CLOUDED:
            return min(RISE_END, RISE_F0 + max(0, t - t0))
        return None

    def moon_params(self, t):
        """The sky over Luna Bay at step t. From the clouded sky the moonrise is the film's (its curves at
        film frames 280..350); the clouds closing, or parting again from part-way, blend from the sky as
        it was at the press to the clouded / clear sky, so nothing jumps when SPACE interrupts."""
        d, t0, P0 = self.moon
        fm = self.moon_fm(t)
        if fm is not None:
            return _fm_params(fm)
        if d > 0:
            u, P1 = ease_in_out(seg(t, t0, t0 + RERISE_T)), CLEAR
        else:
            u, P1 = ease_out(seg(t, t0, t0 + CLOSE_T)), CLOUDED
        if u >= 1:
            return P1
        out = {k: P0[k] + (P1[k] - P0[k]) * u for k in P0}
        out['rise'] = int(round(out['rise']))
        out['pulse'] = 0.0
        return out

    def clear_haze(self):
        """The far cape without its checker haze over the open sky: by day, and whenever the clock's sky is
        painted behind it (dusk, dawn: the haze's gaps showed the clock's pink as checker patches)."""
        return self.day_k() >= DK // 2 or daylight.sky_weight(self.HOUR, self.g.hour) > 0.3

    def day_k(self):
        """Daylight by the world clock in steps, 0 (night) .. DK (day): the rain decks turn grey by day (fine
        steps: each is a cached deck, and coarse ones popped between neighbouring hours)."""
        return int(round(DK * (1 - daylight.night(self.g.hour))))

    def ramps(self):
        """(windows, lamps) 0..1: how far the lights are on at the world clock's hour. Windows follow
        nightwin's ramp (19:20-19:40, off 05:40-06:00); street lamps, lanterns, beacons and the beam come on
        a little earlier (from 18:50) and go off later (by 06:20). All on at the painted hour."""
        n = daylight.night(self.g.hour)
        return nightwin.ramp(self.g.hour), clamp((n - 0.04) / 0.35)

    # ---- light on the re-lit picture (see _spots)
    _pre = _cmp = None
    _posts = ()

    def _world(self, t, cam=None, with_balloon=False):
        self._pre, self._posts = None, []          # (nothing left over from a render outside _world)
        return super()._world(t, cam, with_balloon)

    def _post(self, c, fn):
        """Draw fn(canvas) (a halo, a glow, the beam) onto the re-lit picture instead (after_light), only where
        the render drew nothing over this point since (its z-order is kept)."""
        if isinstance(self._posts, list):
            self._posts.append((c.a.copy(), fn))

    def _done(self, c, cmp=None):
        """The end of an area render: the frame as rendered, for after_light (cmp: the frame before a last
        overall grade, which _post compares against)."""
        self._pre = c.a.copy()
        self._cmp = self._pre if cmp is None else cmp
        return c

    def after_light(self, c, t):
        pre, cmp, posts = self._pre, self._cmp, self._posts
        self._pre, self._cmp, self._posts = None, None, []
        if pre is None or pre.shape != c.a.shape:
            return
        hour = self.g.hour
        off = daylight.night(self.HOUR) - daylight.night(hour)
        fq, tq = daylight._q(self.HOUR), daylight._q(hour)
        if off > 0.3 and abs(fq - tq) >= 0.05:
            # undo the re-light's switch-off of warm bright spots (daylight.relight / _finish): the region's
            # lights are switched by the clock in the render, and the rest are painted walls, hulls, flowers
            m = daylight._lamps(pre)
            if m.any():
                lut, desat, _ = daylight._tables(fq, tq, False)
                p = pre[m]
                out = np.stack([lut[ch].take(p[:, ch]) for ch in range(3)], axis=1)
                if desat > 0.01:
                    fl = out.astype(np.float32)
                    lum = fl @ np.array([0.3, 0.55, 0.15], np.float32)
                    fl += (lum[:, None] - fl) * desat
                    out = np.clip(fl, 0, 255).astype(np.uint8)
                o = out.astype(np.float32)
                dim = (o + (np.array((74, 80, 120), np.float32) - o) * min(1.0, off) * 0.85).astype(np.uint8)
                ys, xs = np.nonzero(m)
                same = np.all(c.a[ys, xs] == dim, axis=1)
                c.a[ys[same], xs[same]] = out[same]
        for before, fn in posts:
            tmp = Canvas(arr=c.a.copy())
            fn(tmp)
            m = np.all(cmp == before, axis=2) & np.any(tmp.a != c.a, axis=2)
            c.a[m] = tmp.a[m]

    def water(self, night, day):
        """A water colour for the world clock: the painted night colour mixed (by daylight) toward the colour
        game/daylight.py's re-light turns into `day` (daylight.unlight), so by day the sea is a grey-blue
        under the clock's sky instead of the night's navy (the scene is painted at night; the re-light
        alone leaves its water dark)."""
        from .. import daylight
        q = 1 - daylight.night(self.g.hour)                 # (smooth with the clock: no steps like day_k)
        if q <= 0.01:
            return night
        return mix(night, daylight.unlight(col(day), self.HOUR, self.g.hour), min(1.0, q))

    def moon_up(self):
        """The painted moon shows only while the world clock says night (game/daylight.py)."""
        from .. import daylight
        return daylight.night(self.g.hour) > 0.4

    def moon_open(self, t):
        return self.moon_params(t)['mt']

    # ------------------------------------------------------------------ events
    def on_interact(self, key, t):
        g = self.g
        if key == 'tower':
            g.sfx('neon', 0.5)
            g.sfx('chime', 0.3)
        elif key == 'ferry':
            g.sfx('horn', 0.75)
            g.sfx('steam', 0.3)
        elif key == 'lanterns':
            self.lantern_ts = [t0 for t0 in self.lantern_ts if t - t0 < LANTERN_LIFE] + [t]
            g.sfx('puff', 0.4)
            g.sfx('chime', 0.4)
        elif key == 'lighthouse':
            g.sfx('horn', 0.4)
            g.sfx('shimmer', 0.4)
        elif key == 'moon':
            if self.moon[0] > 0:                    # risen (or rising): the clouds roll back in
                self.moon = (-1, t, self.moon_params(t))
                g.sfx('wind', 0.4, 1)
                g.sfx('rain', 0.35, 0)
            else:
                self._rise(t)

    def _rise(self, t):
        self.moon = (1, t, self.moon_params(t))
        self.moon_auto = True
        self.g.sfx('whoosh', 0.3)

    def update(self, t):
        g = self.g
        k = t - self.area_t0
        if t - self.t_enter == 6:
            g.sfx('rain', 0.4, 1)
        if self.sap_warm < SAP_PER:
            _sapling_layer(self.sap_warm)
            self.sap_warm += 1
        self.moon_spot.icon = 'rain' if self.moon[0] > 0 else 'moon'
        if self.ai == 0:
            if k % 300 == 150:
                g.sfx('rain', 0.3, 1)
            if k % 420 == 330:
                g.sfx('waves', 0.25)
            bx = self.big_x(t)
            if int(bx) == 300 and int(self.big_x(t - 1)) != 300 and t - self.spot_t.get('ferry', -999) > 200:
                g.sfx('horn', 0.4 * self.near_gain(bx + 24))
            self._tint_vanhoover()
        elif self.ai == 1:
            if k % 330 == 200:
                g.sfx('wind', 0.3, 0)
            if k % 410 == 120:
                g.sfx('creak', 0.3 * self.near_gain(110), 0)
            if k % 300 == 60:
                g.sfx('rain', 0.3, 0)
            self._tint_mountains()
        else:
            if not self.moon_auto and self.moon[0] < 0 and self.bx > MOON_AUTO_X:
                self._rise(t)
            fm, fm1 = self.moon_fm(t), self.moon_fm(t - 1)
            if fm is not None and fm1 is not None:
                if fm1 < T_MOON <= fm:
                    g.sfx('shimmer', 0.55)
                elif fm1 < T_MOON + 4 <= fm:
                    g.sfx('waves', 0.5)
            mt = self.moon_open(t)
            if k % 280 == 140:
                g.sfx('waves', 0.25 + 0.2 * mt)
            if k % 300 == 40 and mt < 0.5:
                g.sfx('rain', 0.3, 0)
            if k % 400 == 260:
                g.sfx('creak', 0.28 * self.near_gain(270), 1)
            self.areas[2].balloon_tint = _tint_between(TINT_RAIN, TINT_MOON, mt)

    def _tint_vanhoover(self):
        """Bluish in the rain; warm when flying low beside a promenade lamp."""
        A = self.areas[0]
        sx = self.bx - self.cx
        near = 0.0
        for lx in LAMPS_A:
            d = abs(sx - (lx - self.cx * P_FG))
            near = max(near, clamp(1 - d / 40))
        near *= clamp((self.by - 96) / 40)
        A.balloon_tint = _tint_between(TINT_RAIN, TINT_LAMP, near)

    def _tint_mountains(self):
        A = self.areas[1]
        near = 0.0
        for (lx, ly) in _lanterns_b():
            d = math.hypot(self.bx - (lx - VO), (self.by - 10 - (ly - 11)) * 1.2)
            near = max(near, clamp(1 - d / 36))
        A.balloon_tint = _tint_between(TINT_RAIN, TINT_LAMP, near)

    # ================================================================== sky hooks (game/daylight.py)
    def sky_vanhoover(self, f, cx, cy):
        """render_vanhoover's sky alone: the night gradient with its city glows (the rain decks are
        clouds and stay); below the sky layer is masked out."""
        dk = self.day_k()
        c = _no_sky(YW_A)
        so = int(round(cx * P_CITY))
        c.a[:YW_A] = _sky_a().a[:, so:so + W]
        _tile_hide(c, _deck(DECK_A0, dk),
                   -f * 0.12 - cx * 0.2, -6)                    # the rain decks are sky-dark too (they
        _tile_hide(c, _deck(DECK_A1, dk),
                   -f * 0.25 - cx * 0.3, 22)                    # stay as clouds by day)
        _hide(c, _range_a(), -int(cx * P_RANGE) - 10, 0)       # the dark range and city are sky-dark
        _hide(c, _city_a()[0], -int(round(cx * P_CITY)), 0)
        _mark(c, 'city', -int(round(cx * P_CITY)), 0)
        return c

    def sky_mountains(self, f, cx, cy):
        """render_mountains' sky alone: the night gradient and the moon-side silvering glows (cached per
        camera), with the drifting rain decks stamped out."""
        dk = self.day_k()
        c = _sky_mountains(cx).copy()
        _tile_hide(c, _deck(DECK_B0, dk),
                   -f * 0.12 - cx * 0.2, -8)                    # the rain decks stay clouds by day
        _tile_hide(c, _deck(DECK_B1, dk),
                   -f * 0.3 - cx * 0.4, 30)
        return c

    def sky_shoals(self, f, cx, cy):
        """render_shoals' sky alone: gradient, stars, the moon with its glow (and the break-out pulse);
        the cloud puffs are stamped out. All but the stars and puffs is cached; the stars get the moon's
        light."""
        dk = self.day_k()
        P = self.moon_params(self.g.t)
        mt = P['mt']
        mx = sc.MOON[0] - int(round(cx * P_MOON))
        my = sc.MOON[1] + P['rise'] if self.moon_up() else -200      # by day the painted moon is not drawn
        base, delta, free = _sky_shoals(mt, mx, my, P["pulse"], -int(cx * P_FAR_C) - 10, self.clear_haze())
        c = Canvas(arr=base.copy())
        if mt > 0:
            st = Canvas(arr=np.zeros((H, W, 3), np.uint8))
            fx.stars(st, f, seed=17, n=int(56 * mt), region=(0, 0, W, 80))
            s = st.a[:81]
            m = ((s[..., 0] | s[..., 1] | s[..., 2]) > 0) & free[:81]
            c.a[:81][m] = np.clip(s[m] + delta[:81][m], 0, 255)
        for sd, pw, ph, xx, yy in _puffs(f, cx, P):           # the rain puffs stay clouds by day
            _hide(c, _puff(sd, pw, ph, dk), xx, yy)
        _hide(c, sc.town_c()[0], -int(round(cx * P_TOWN)) - TO, 0)   # the town's hill: its night-dark dither
        return c                                                    # matched the sky (pink patches at dawn)

    # ================================================================== area 0: Vanhoover
    def render_vanhoover(self, f, cx, cy):
        dk = self.day_k()
        t = self.g.t
        c = Canvas(W, H)
        so = int(round(cx * P_CITY))
        c.a[:YW_A] = _sky_a().a[:, so:so + W]
        _tile_paste(c, _deck(DECK_A0, dk),
                      -f * 0.12 - cx * 0.2, -6)
        _tile_paste(c, _deck(DECK_A1, dk),
                      -f * 0.25 - cx * 0.3, 22)
        _paste(c, _range_a(), -int(cx * P_RANGE) - 10, 0)
        sc.mist(c, 84, 30, f, '#3c5a6e', 0.35, seed=3, level=0.55, speed=0.18, ox=cx * 0.35)
        city, toggles, blinkers = _city_a()
        ox = -int(round(cx * P_CITY))
        kw, kl = self.ramps()
        a = t - self.spot_t.get('tower', -999)
        if 0 <= a < TOWER_T and kl > 0:             # the lit city warms the low clouds above it
            gk = 0.35 * smooth(seg(a, 0, 20)) * (1 - seg(a, 70, TOWER_T)) * kl
            self._post(c, lambda cv, x=HERO_X + ox, gk=gk: fx.glow(cv, x, 70, 110, '#6a5a48', gk, rings=6))
        _paste(c, city, ox, 0)
        _switch(c, 'city', ox, 0, kw, kl)          # by day and at dusk: its windows and bulbs unlit
        for (xx, yy, s) in toggles:
            per = 54 + int(hash01(s, 1) * 72)
            on = hash01(s, 2, (f + int(hash01(s, 3) * per)) // per) > 0.45 and hash01(s, 5) < kw
            c.rect(xx + ox, yy, 2, 2, (sc.W_WARM if hash01(s, 4) < 0.7 else sc.W_COOL) if on else sc.W_OFF)
        lit_all = 0 <= a < TOWER_T
        for i, (bx, by, bc) in enumerate(blinkers):
            if lit_all and a < 80 or ((f + i * 11) // 12) % 3 != 2 and kl > 0.3 + 0.4 * hash01(i, 8):
                cc = {'red': '#ff5a6a', 'teal': sc.W_TEAL, 'gold': sc.W_WARM}[bc]
                c.px(bx + ox, by, cc)
                if bc == 'red' and ((f + i * 11) // 12) % 3 == 0:
                    c.px(bx + ox - 1, by, '#8a3048')
                    c.px(bx + ox + 1, by, '#8a3048')
        if lit_all:
            self._city_lights(c, a, ox)
        hx = HERO_X + ox
        hk = (0.25 + 0.05 * math.sin(f * 0.2)) * kw
        k = 1 - seg(a, 60, TOWER_T) if lit_all else 0.0
        if hk > 0.01 or k > 0:                      # the hero crown's teal halo (and the SPACE flare)

            def crown(cv, hk=hk, k=k, r=12 + int(10 * smooth(seg(a, 0, 12)))):
                if hk > 0.01:
                    fx.glow(cv, hx, HERO[2] - 5, 9, '#3a8a9a', hk, rings=3)
                if k > 0:
                    fx.glow(cv, hx, HERO[2] - 6, r, '#9be8e4', 0.5 * k, rings=4)
            self._post(c, crown)
        if lit_all and (a // 4) % 2 == 0 and a < 80:
            c.px(hx, HERO[2] - 27, 'white')
            c.px(hx - 1, HERO[2] - 26, '#ffd0d8')
            c.px(hx + 1, HERO[2] - 26, '#ffd0d8')
        sc.mist(c, 100, 20, f, '#4a6a80', 0.28, seed=8, level=0.5, speed=0.35, ox=cx * 0.6)
        # ---- water
        sc.reflect(c, YW_A, 152, f, self.water('#1a2a42', DAY_SEA_A[0]), self.water('#0e1729', DAY_SEA_A[1]),
                   k0=0.42, k1=0.14, stretch=0.5, wob=2.6, bright_boost=1.0)
        sc.mist(c, 112, 16, f, '#3d5872', 0.3, seed=9, level=0.45, speed=-0.3, ox=cx * 0.8)
        xs = self.small_x(t) - cx * P_FAR_FERRY
        if -50 < xs < W + 30:
            sc.draw_ferry(c, xs, 124, f, 's', flip=True, wake_dir=-1)
            if kw < 0.5:
                _ferry_off(c, xs, 124, 's', flip=True)
        xb = self.big_x(t) - cx
        if -100 < xb < W + 4:
            sc.draw_ferry(c, xb, FERRY_YB, f, 'l')
            if kw < 0.5:
                _ferry_off(c, xb, FERRY_YB, 'l')
            if 0 <= t - self.spot_t.get('ferry', -999) < FERRY_T:
                self._post(c, lambda cv, xb=xb: self._ferry_horn(cv, t, xb))
        sc.rain_rings(c, f, 121, 151, 22, '#5f7a9c', 11, shift=cx)
        # ---- foreground promenade
        fo = -int(round(cx * P_FG))
        _paste(c, _seawall_a(), fo, 0)
        _switch(c, 'seawall', fo, 0, kw, kl)       # the lamps' glass unlit by day
        if kl > 0:                                  # their light: on the re-lit picture

            def lamps(cv, fo=fo, kl=kl):
                for i, lx in enumerate(LAMPS_A):
                    sx = lx + fo
                    if not -30 < sx < W + 30:
                        continue
                    fl = kl * 0.42 + 0.04 * math.sin(f * 0.37 + i * 2) + (0.03 if hash01(i, f // 5) > 0.8 else 0)
                    fx.glow(cv, sx, 107, 22, '#ffc070', fl * 0.5, rings=5)
                    fx.glow(cv, sx, 170, 16, '#ffb060', 0.22 * kl, rings=3, oy=0)
                    for yy in range(158, 180, 2):
                        w2 = 1 + (yy - 158) // 8
                        dx = int(round(math.sin(yy * 0.9 + f * 0.25)))
                        cv.blend(sx - w2 + dx, yy, 2 * w2 + 1, 1, '#ffd890', 0.35 * kl)
                    for yy in range(136, 151, 2):
                        dx = int(round(math.sin(yy * 0.7 + (f // 3) * 0.6) * 2))
                        cv.blend(sx - 1 + dx, yy, 3, 1, '#ffcf80', 0.3 * kl)
            self._post(c, lamps)
        # ---- rain: far fine drops + near longer streaks; glints in the lamp light
        sc.rain_layer(c, f, 70, '#4d6385', 21, slant=1, speed=5, length=2, shift=cx * 0.6)
        sc.rain_layer(c, f, 26, '#8aa6c8', 22, slant=2, speed=8, length=4, shift=cx * 1.6)
        for i, lx in enumerate(LAMPS_A):
            sx = lx + fo
            if -10 < sx < W + 10 and kl > 0.5:
                sc.rain_layer(c, f, 6, '#ffe2a8', 40 + i, slant=2, speed=8, length=3, region=(sx - 14, 94, 28, 60))
        for i in range(10):
            kk = (f + i * 5) // 6
            xx = int(hash01(i, kk, 5) * W)
            yy = 158 + int(hash01(i, kk, 6) * 20)
            if (f + i * 5) % 6 < 2:
                c.px(xx - 1, yy, '#6d84a8')
                c.px(xx + 1, yy, '#6d84a8')
        return self._done(c)

    @staticmethod
    def _city_lights(c, a, ox):
        """SPACE at the hero tower: every dark window lights up in a sweep out from it, holds, and the
        lights go out again one by one (drawn before the bay's reflection, so the water mirrors it)."""
        ys, xs, d, cols, h = _dark_windows()
        R = 330 * ease_out(seg(a, 0, 30))
        off = seg(a, 72, TOWER_T - 4)
        on = (d < R) & (h >= off)
        sx = xs + ox
        on &= (sx >= 0) & (sx < W)
        cc = cols[on].copy()
        front = (d[on] > R - 10) & (a < 32)
        cc[front] = col('white')
        c.a[ys[on], sx[on]] = cc

    def _ferry_horn(self, c, t, xb):
        a = t - self.spot_t.get('ferry', -999)
        if not 0 <= a < FERRY_T:
            return
        k = 1 - a / FERRY_T
        fxw = int(xb) + 30
        fx.glow(c, int(xb) + 24, FERRY_YB - 9, 30, '#ffd890', 0.45 * k * k, rings=4)
        for xx in range(int(xb) + 9, int(xb) + 37, 2):          # the decks blaze
            if a < 40 or (xx // 2 + a // 3) % 3:
                c.px(xx, FERRY_YB - 12, '#fff6d8')
        for j in range(5):                                       # steam puffs from the funnel
            age = a - j * 7
            if 0 <= age < 52:
                r = 1.5 + age * 0.085
                py = FERRY_YB - 16 - age * 0.95
                px = fxw + age * 0.36 + math.sin(age * 0.15 + j) * 1.2
                base = mix('#f4f8ff', '#7a88a4', age / 52)
                if age < 44 or (age + j) % 2:
                    c.circle(px, py, r, shade(base, 0.78))
                    c.circle(px - r * 0.3, py - r * 0.3, max(0.6, r * 0.7), base)
        for j in range(3):                                       # the horn's blast rings out
            r = (a - j * 8) * 1.3 + 4
            if 4 <= r < 44:
                c.circle(fxw - 2, FERRY_YB - 15, int(r), '#ffe6a0' if j == 0 else '#c8d8ea', fill=False)

    # ================================================================== area 1: Smokey Mountains
    def render_mountains(self, f, cx, cy):
        dk = self.day_k()
        t = self.g.t
        c = Canvas(W, H)
        c.gradient(0, 0, W, 130, sc.SKY_NIGHT, soft=0.7)
        moon_t = 0.35 + 0.65 * cx / CAM_B           # the cloud deck silvers toward the coast (east)
        fx.glow(c, 292, 0, 110, '#8090b8', 0.32 * smooth(moon_t), rings=6)
        _tile_paste(c, _deck(DECK_B0, dk),
                      -f * 0.12 - cx * 0.2, -8)
        fx.glow(c, 292, 0, 80, '#a8b4dc', 0.26 * smooth(moon_t), rings=5)
        _paste(c, sc.far_b(), -int(cx * P_FAR_B) - 20, 0)
        sc.mist(c, 78, 26, f, '#4a7080', 0.3, seed=21, level=0.5, speed=0.2, ox=cx * 0.3)
        _tile_paste(c, _deck(DECK_B1, dk),
                      -f * 0.3 - cx * 0.4, 30)
        pk, smokes, lights, flags = sc.peaks_b()
        po = -int(round(cx * P_PEAKS)) + 6
        _paste(c, pk, po, sc.PK_Y)
        kw, kl = self.ramps()
        _switch(c, 'peaks', po, sc.PK_Y, kw, kl)   # the fortress and cabin windows unlit by day
        for i, (sx, sy) in enumerate(smokes):
            if -40 < sx + po < W + 10:
                sc.smoke(c, sx + po, sy + sc.PK_Y, f, seed=i * 3 + 1, n=12, height=30 + (i % 3) * 5,
                         color='#8a96ac')
        if kw > 0:                                  # the windows' glow, on the re-lit picture

            def windows(cv, kw=kw):
                for i, (lx, ly) in enumerate(lights):
                    if -6 < lx + po < W + 6:
                        fx.glow(cv, lx + po, ly + sc.PK_Y, 5, '#ffb060', kw * (0.16 + 0.04 * math.sin(f * 0.3 + i * 1.7)),
                                rings=2)
            self._post(c, windows)
        for i, (fxx, fyy, ph) in enumerate(flags):
            sc.draw_flag(c, fxx + po, fyy + sc.PK_Y - ph, ph, f, seed=i * 1.3)
        sc.mist(c, 104, 30, f, '#5a7a88', 0.32, seed=23, level=0.55, speed=0.35, ox=cx * 0.75)
        vl, (bx0, bx1) = _valley_b()
        vo = -cx - VO
        _paste(c, vl, vo, 0)
        for i in range(16):
            tt_ = ((f * 0.9 + i * 13) % 52) / 52
            yy = 128 + tt_ * (H - 128)
            tt = (yy - 128) / (H - 128)
            xx = _creek_x(yy) + (hash01(i, 3) - 0.5) * (2 + tt * 12)
            c.hline(int(xx + vo), int(xx + vo) + (1 if tt < 0.5 else 2), int(yy), '#4a6a90')
        _paste(c, _sapling_layer(f % SAP_PER), vo, SAP_Y)
        ons = []
        for i, (lx, ly) in enumerate(_lanterns_b()):
            sx = lx + vo
            on = clamp((kl - 0.4 * hash01(i, 613)) / 0.4)        # the valley lanterns, one by one
            ons.append(on)
            if -12 < sx < W + 12:
                c.vline(sx, ly - 9, ly, '#3a281e')
                c.rect(sx - 1, ly - 12, 3, 3, mix(PAPER_OFF, '#ffcf70', on))
                c.px(sx, ly - 13, '#3a281e')
        if kl > 0:

            def lanterns(cv):
                for i, (lx, ly) in enumerate(_lanterns_b()):
                    sx = lx + vo
                    if -12 < sx < W + 12 and ons[i] > 0:
                        fx.glow(cv, sx, ly - 11, 10, '#ffb060', ons[i] * (0.3 + 0.04 * math.sin(f * 0.4 + i)), rings=4)
            self._post(c, lanterns)
        for t0 in self.lantern_ts:                  # (a retrigger keeps the earlier lanterns aloft)
            if 0 <= t - t0 < LANTERN_LIFE:          # (onto the re-lit picture: the sky lanterns rise into the sky)
                self._post(c, lambda cv, a=t - t0: self._lantern_chain(cv, a, vo, po, lights))
        sc.mist(c, 126, 22, f, '#5a7a88', 0.22, seed=25, level=0.45, speed=-0.25, ox=cx)
        _paste(c, sc.fg_b(), -int(round(cx * P_FG_B)), 0)
        sc.rain_layer(c, f, 70, '#4d6385', 31, slant=1, speed=5, length=2, shift=cx * 0.7)
        sc.rain_layer(c, f, 24, '#8aa6c8', 32, slant=2, speed=8, length=4, shift=cx * 1.35)
        return self._done(c)

    def _lantern_chain(self, c, a, vo, po, lights):
        """SPACE at the footbridge lantern: the lanterns flare one after another, out from it across the
        valley and up the slopes to the fortress and the cabins; paper sky lanterns rise from the valley."""
        hx = self.lantern_spot[0] + vo + VO          # the hotspot lantern, screen px
        pts = [(lx + vo, ly - 11, 0) for (lx, ly) in _lanterns_b()]
        pts += [(lx + po, ly + sc.PK_Y, 1) for (lx, ly) in lights]
        for i, (sx, sy, far) in enumerate(pts):
            if not -30 < sx < W + 30:
                continue
            age = a - abs(sx - hx) * 0.2 - abs(sy - 140) * 0.12
            if not 0 <= age < 80:
                continue
            k = min(1.0, age / 6) * (1 - seg(age, 30, 80))
            if far:
                fx.glow(c, sx, sy, 5 + int(7 * k), '#ffc070', 0.6 * k, rings=3)
                if age < 14:
                    c.px(sx, sy, 'white')
                    c.px(sx - 1, sy, '#fff2c8')
                    c.px(sx + 1, sy, '#fff2c8')
            else:
                fx.glow(c, sx, sy, 12 + int(14 * k), '#ffc070', 0.6 * k, rings=5)
                c.rect(sx - 1, sy - 1, 3, 3, 'white' if age < 20 else '#fff2c8')
                ui.burst(c, int(a), int(a - age), sx, sy - 1, colors=('butter', 'gold', 'white', 'peach'),
                         n=10, seed=i * 7, power=0.55, life=30, gravity=0.03, size=1)
        for j, (lx, ly) in enumerate(_lanterns_b()):         # sky lanterns rise and sway
            for m in range(3):
                sx0 = lx + vo + (m - 1) * 5
                age = a - abs(lx + vo - hx) * 0.2 - 8 - m * 12
                if not 0 <= age < 170:
                    continue
                py = ly - 14 - age * 0.5 - m * 2
                px = sx0 + math.sin(age * 0.05 + j * 1.7 + m) * 3 + age * 0.1
                fade = 1 - seg(age, 120, 170)
                if py < -6 or hash01(j, m, int(age) // 3) > fade:
                    continue
                ix, iy = int(px), int(py)
                fx.glow(c, ix, iy + 1, 6, '#ffb060', 0.35 * fade, rings=2)
                c.rect(ix - 1, iy - 1, 3, 4, '#ff9a50')
                c.rect(ix - 1, iy, 3, 2, '#ffd27a')
                c.px(ix, iy + 1, '#fff2c8')
                c.hline(ix - 1, ix + 1, iy - 2, '#c86a40')

    # ================================================================== area 2: Silver Shoals / Luna Bay
    def render_shoals(self, f, cx, cy):
        dk = self.day_k()
        t = self.g.t
        P = self.moon_params(t)
        mt = P['mt']
        c = Canvas(W, H)
        sky = [mix(a, b, mt * 0.55) for a, b in zip(sc.SKY_NIGHT, ['#0e1430', '#141c3e', '#1a2650', '#22325e',
                                                                   '#2b3f6c', '#364d78', '#425a82'])]
        c.gradient(0, 0, W, HZ + 1, sky, soft=0.7)
        mx, my = sc.MOON[0] - int(round(cx * P_MOON)), sc.MOON[1]
        if mt > 0:
            fx.stars(c, f, seed=17, n=int(56 * mt), region=(0, 0, W, 80))
        my += P['rise']
        if not self.moon_up():                      # the world clock says day: no painted moon (the
            my = -200                               # repainted sky has the sun; clouds stay)
        fx.glow(c, mx, my, 44, '#7a8cc0', 0.22 + 0.2 * mt, rings=6)
        c.circle(mx, my, sc.MOON_R + 1, '#9fb0d6')
        c.circle(mx, my, sc.MOON_R, '#eef3fb')
        c.circle(mx + 2, my + 1, sc.MOON_R - 2, '#dde6f4')
        for (dx, dy, r) in [(-3, -1, 2.6), (-1, 2, 2.2), (1, 0, 1.6), (4, 5, 0.8)]:
            c.circle(mx + dx, my + dy, r, '#c9d3e7')
        c.px(mx - 6, my - 5, 'white')
        c.px(mx - 5, my - 6, 'white')
        for sd, pw, ph, xx, yy in _puffs(f, cx, P):
            _paste(c, _puff(sd, pw, ph, dk), xx, yy)
        pulse = P['pulse']
        if 0 < pulse < 1:
            r = int(14 + 44 * ease_out(pulse))
            fx.glow(c, mx, my, r, '#c8d6ff', 0.4 * (1 - pulse), rings=4)
        _paste(c, _far_c(self.clear_haze()), -int(cx * P_FAR_C) - 10, 0)
        c.gradient(0, HZ, W, YW_C - HZ, [self.water(mix('#1a2a4a', '#2a3c66', mt), DAY_SEA_C[0]),
                                         self.water(mix('#14223e', '#1e2e52', mt), DAY_SEA_C[1]),
                                         self.water(mix('#101c34', '#18264a', mt), DAY_SEA_C[2])], soft=0.6)
        ripple = self.water(mix('#24365a', '#4a5e8e', mt), DAY_RIPPLE)
        for i in range(30):
            tt = ((f * 0.25 + i * 7.3) % 32) / 32
            y = HZ + 1 + int(tt * tt * (YW_C - HZ - 2))
            x = int((hash01(i, 3) * 360 - cx * 0.6) % 360) - 20
            ln = 2 + int(tt * 6)
            c.hline(x, x + ln, y, ripple)
        kw, kl = self.ramps()
        _switch(c, 'farc', -int(cx * P_FAR_C) - 10, 0, kw, kl)      # Luna Bay's far shore lights
        fbx = int(FAR_BOAT_X0 + FAR_BOAT_V * ((t + 900) % FAR_BOAT_P) - cx * 0.3)
        fby = 104 + int(round(math.sin(f * 0.1)))
        if -10 < fbx < W + 2:
            c.tri((fbx, fby - 1), (fbx + 6, fby - 1), (fbx + 3, fby - 10), '#c8d2e6')
            c.vline(fbx + 3, fby - 11, fby - 1, '#3a3a50')
            c.rect(fbx - 1, fby - 1, 9, 2, '#28304a')
            c.px(fbx + 3, fby - 12, '#ffcf70' if kl > 0.5 else '#3a3a50')
        # the moonlight: the painted moon's while the painted sky is kept (the night hours); once the clock's
        # sky replaces it (dusk, dawn, day) the moon path follows the clock's moon and the silver rims and the
        # moonlit grade go
        ws = daylight.sky_weight(self.HOUR, self.g.hour)
        mk = mt * (1 - ws)
        town, rim, lights, apexes, (llx, lly) = sc.town_c()
        to = -int(round(cx * P_TOWN)) - TO
        _paste(c, town, to, 0)
        _switch(c, 'town', to, 0, kw, kl)          # windows, stair / door / string lanterns, the lighthouse lamp
        if mk >= 1:
            _paste(c, rim, to, 0)
        elif mk > 0:
            sc.paste_dither(c, rim, to, 0, mk, ox=to)
        la = self._light_age(t)
        if kw > 0 or la is not None:                # the windows' glow (and the SPACE flash), on the re-lit picture

            def glows(cv, kw=kw):
                for i, (lx, ly) in enumerate(lights):
                    if -6 < lx + to < W + 6 and kw > 0:
                        fx.glow(cv, lx + to, ly, 5, '#ffb060', kw * (0.16 + 0.04 * math.sin(f * 0.3 + i)), rings=2)
                if la is not None:
                    self._lamp_flash(cv, la, llx + to, lly, full=False)
            self._post(c, glows)
        sc.reflect(c, YW_C, H, f, self.water(mix('#16243e', '#22345a', mt), DAY_REFL_C[0]),
                   self.water('#0c1428', DAY_REFL_C[1]), k0=0.5, k1=0.18, stretch=0.6, wob=2.4, bright_boost=0.9)
        sc.rain_rings(c, f, YW_C + 2, H - 2, int(20 * (1 - mt) + 4), '#5f7a9c', 51, shift=cx)
        if ws < 0.5:
            pcx, reach = mx, (P['reach'] if self.moon_up() else HZ + 1)
        else:
            kind, pcx, _ = daylight._body_pos(self.g.hour)
            reach = P['reach'] if kind == 'moon' and daylight.night(self.g.hour) > 0.4 else HZ + 1
        for y in range(HZ + 1, int(reach)):
            if (y + f // 3) % 2 and y > HZ + 3:
                continue
            d = y - HZ
            wdt = 2 + d * 0.38
            n = 1 + int(d * 0.06)
            for j in range(n):
                if mt < 1 and hash01(y, j, 5) > mt * 1.4:
                    continue
                hh = hash01(y, j, f // 3)
                xx = int(pcx + (hh - 0.5) * 2 * wdt)
                ln = 1 + int(hash01(y, j, 7) * (1 + d * 0.05))
                cc = '#ffffff' if hash01(y, j, f // 3, 2) > 0.6 else ('#cfdcf4' if d < 40 else '#9fb4dc')
                c.hline(xx, xx + ln - 1, y, cc)
        sc.draw_boat(c, sc.sailboat(1, '#5fb3b0'), 92 - cx, 150, f, 0.0)
        sc.draw_boat(c, sc.sailboat(2, '#f29a8e'), 270 - cx, 146, f, 1.7)
        if kl > 0 or la is not None:                # the beam sweeps across the re-lit sky
            self._post(c, lambda cv, kl=kl: self._lighthouse(cv, f, t, llx + to, lly, kl))
        dko = -int(round(cx * P_DOCK)) - 4
        _paste(c, sc.dock_c(), dko, 0)
        _switch(c, 'dock', dko, 0, kw, kl)
        dl = 141 + dko
        if -20 < dl < W + 20 and kl > 0:
            self._post(c, lambda cv, kl=kl: fx.glow(cv, dl + 1, 129, 14, '#ffb060',
                                                    kl * (0.3 + 0.04 * math.sin(f * 0.4)), rings=4))
        nr = int(60 * (1 - mt) + 6)
        sc.rain_layer(c, f, nr, '#5d7394', 61, slant=1, speed=5, length=2, shift=cx * 0.7)
        sc.rain_layer(c, f, max(2, int(22 * (1 - mt))), '#9ab6d8', 62, slant=2, speed=8, length=4,
                      shift=cx * 1.3)
        cmp = c.a.copy()
        if mk > 0:                                  # the moonlit night (only under the painted moon)
            fx.grade(c, (1 + 0.06 * mk, 1 + 0.08 * mk, 1 + 0.12 * mk), (4 * mk, 6 * mk, 10 * mk))
        return self._done(c, cmp)

    def _light_age(self, t):
        a = t - self.spot_t.get('lighthouse', -999)
        return a if 0 <= a < LIGHT_T else None

    @staticmethod
    def _lamp_flash(c, a, lx, ly, full=True):
        """The lamp's three flashes (full=False: just the light, drawn before the sea mirrors it)."""
        for p0 in (0, 14, 28):
            q = a - p0
            if not 0 <= q < 16:
                continue
            kk = 1 - q / 16
            fx.glow(c, lx, ly, 14 + q * 2, '#fff2c0', 0.85 * kk, rings=5)
            if not full:
                continue
            r = 6 + q * 3
            c.circle(lx, ly, r, '#ffe6a0' if q < 8 else '#c8b890', fill=False)
            L = 16 - q
            if L > 2:
                c.hline(lx - L, lx + L, ly, 'white')
                c.vline(lx, ly - L // 2, ly + L // 2, 'white')
                for k in range(2, L // 3 + 1):
                    for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        c.px(lx + sx_ * k, ly + sy_ * k, '#fff6d0')

    def _lighthouse(self, c, f, t, lx, ly, k=1.0):
        """The sweeping beam (k: its strength by the clock, none by day); SPACE: the lamp flashes three times
        and the beam spins one extra turn."""
        a = self._light_age(t)
        b = c.a.copy() if k < 1 else None
        if a is not None:
            extra = 2 * math.pi * ease_in_out(seg(a, 0, LIGHT_T - 6))
            fb = f + extra / 0.06
            for _ in range(2 if a < LIGHT_T - 10 else 1):
                sc.draw_beam(c, lx, ly, fb)
        elif k > 0:
            sc.draw_beam(c, lx, ly, f)
        if b is not None:
            m = np.any(c.a != b, axis=2)
            kk = max(k, 0.6) if a is not None else k
            c.a[m] = (b[m] + (c.a[m].astype(np.float32) - b[m]) * kk).astype(np.uint8)
        if a is not None:
            self._lamp_flash(c, a, lx, ly)
