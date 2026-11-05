"""Canterlot as an explorable diorama (art: scenes/s01_canterlot.py, used read-only).

The film shows Canterlot in three different views, so the region has three areas, in film order:
  0  THE CITY ON THE MOUNTAIN (shot A): a vertical panorama, from the valley (Ponyville roofs,
     Saddle Lake) up the waterfalls to the castle. Every layer is one screen wide, so the camera only
     tilts (the film's layer ranges). Blue hour on landing; the sunrise (sun breaking the far ridge,
     gold light sweeping the city, banners unfurling, windows going dark) is replayed on a game clock
     when the player rings the castle bell (or by itself after a while), anchored to the world so it
     stays put while the camera tilts.
  1  ROYAL GARDENS & MOUNTAIN RAILWAY (shot B): a short pan along the hedge-maze terrace to the rock
     spur. The pink-and-white train puffs out of the tunnel when you arrive and corkscrews all the
     way down the spur (hidden behind the rock on the back passes); more trains come when called.
  2  AIRSHIP DOCKS (shot C): a vertical panorama from the dock ledge (mast, docked zeppelin, pier)
     up to the city glittering in morning light. The film's own balloon (moored at the pier, then
     casting off) is never drawn: the player's balloon is the explorer's.
Every film-time ambient (falls, mist, flags, banners, fountains, lamps, beacon, propeller, clouds)
runs on the free scene frame; the scene's private layer builders and helpers are reused as they are.

Things to set off with SPACE: the bell in Canterlot Tower (rings in the dawn; once the sun is up, a
wave of gold light runs over the city), the waterfalls (they surge, spray and throw a rainbow over
the plunge pool), the royal fountains (all three jets shoot up), the mountain railway (calls the train
out of the tunnel, or makes it whistle and blow steam) and the docked airship (it slips its lines and
rises over the dock, and comes back down by itself or when called). Three atlas stamps are hidden:
beside the snowy summit of Mt. Canterhorn, under the railway viaduct, and high by a castle spire above
the docks.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, smooth, clamp, hash01, bayer, mix, col, ease_in_out, lerp
import scenes.s01_canterlot as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp
from .everfree import GlowKeep

# ----------------------------------------------------------------------------- area 0: the city
CITY_CY = 70            # shot A tilt: city layer -70..0 (valley + foreground run out below 70)
SUN_HOLD = 96           # story frame held in blue hour until the sunrise starts
SUN_END = 236           # story frame the sunrise holds at (sweep done, sun clear of the ridge)
SUN_TRIGGER_T = 420     # steps after landing before the sun rises by itself (the bell does it sooner)
# the world clock (game/daylight.py): the city's blue hour / sunrise follows it. Dawn: blue hour on
# landing, the sun rises by itself or on the bell. Day: the city lands already sunlit. Dusk: the sunrise
# plays backwards (the light leaves the city, windows and lamps come back on). Night: blue hour, and
# the bell only rings. The city is painted at HOUR_BLUE in the blue hour, HOUR_SUN once sunlit.
DAY_FROM, NIGHT_FROM = 6.6, 19.3
HOUR_BLUE, HOUR_SUN = 5.7, 7.0     # (5.7: night enough that the relight only dims the blue hour at night)
HOUR_GARDEN, HOUR_DOCK = 7.5, 8.5   # the gardens and the docks are painted in morning sun
TINT_BLUE = ('#6a64a8', 0.42)                                # the balloon in the blue hour
_BAYW = bayer(H + CITY_CY + 2, W).astype(np.float32) * 18     # world-anchored sweep dither
_YY, _XX = np.mgrid[0:H, 0:W]
_ONES = np.ones((H, W), bool)


def _th(s):
    """Sweep threshold (film's _sweep_th up to its end, then carried on over the valley)."""
    if s <= sc._SW1:
        return sc._sweep_th(s)
    return -30 - (s - sc._SW1) * 5.0


_SCORE0 = (_XX + 1.25 * (H - _YY)).astype(np.float32)


def _score_rows(cy):
    """The film's sweep score in screen space for a camera at city height cy (world anchored)."""
    return _SCORE0 - np.float32(1.25 * cy) + _BAYW[cy:cy + H]


def _sun_mask(s, cy):
    """(mask, band, all_lit): screen-space 'sun' mask of the sweep for story frame s."""
    if s < sc.SUNRISE:
        return None, None, False
    th = _th(s)
    if th < -125:
        return None, None, True
    sco = _score_rows(cy)
    m = sco > th
    band = m & (sco < th + 8)
    return m, band, False


BELL = (238, 44)        # the belfry of Canterlot Tower (its pennant tip is at (238, 20))
FALLS_SPOT = (184, 198)  # over the plunge pool, between the falls
POOL_Y = 218
RAINBOW = (182, 222, 76)                                     # centre x, centre y, outer radius
_RB_COLS = [np.array(col(c), np.float32) for c in ('red', 'orange', 'butter', 'mint', 'baby_blue', 'lavender')]

# ----------------------------------------------------------------------------- area 1: the gardens
GARDEN_CAM = 80         # rail layer (372 px, p 0.7, +8) ends at cam 85; garden + castle stay covered
TRAIN_LEAD = 222        # train clock on arrival (emerges ~1 s after the cloud sweep)
TRAIN_END = 1190        # train clock by which the whole train has corkscrewed below the screen
TRAIN_CLEAR = 340       # train clock by which its last car is out of the tunnel and away from it
TRAIN_AUTO = 900        # steps without a train in the gardens before one comes by itself
SPUR_X = 238            # helix centre (rail layer)
VIADUCT = (280 + 8, 88)  # mid viaduct in rail-layer px (+ the layer's offset), parallax 0.7


_CARS = ((sc._ENGINE, sc._ENGINE.flipped()), (sc._CAR, sc._CAR.flipped()))   # facing left / right


def _rail_x(cam):
    return int(round(-cam * 0.7)) + 8


# ----------------------------------------------------------------------------- area 2: the docks
DOCK_RISE = 136         # shot C: the dock drops 136 px as the camera rises to the city
SHIP_RISE = 58          # the airship lifts this far off its mast when cast off
SHIP_UP, SHIP_STAY = 64, 450                    # rise, time aloft before it comes back by itself
SHIP_LAND, SHIP_DOWN = 84, 96                    # descent, + the lines thrown back up
_ZX0, _ZCY = 36, 80     # zeppelin nose x and centre line (scene _dock)
LINES = [((_ZX0 + 26, _ZCY + 14), (_ZX0 + 34, sc.DECK_Y - 6)),
         ((_ZX0 + 112, _ZCY + 14), (_ZX0 + 122, sc.DECK_Y - 6)),
         ((_ZX0 + 140, _ZCY + 9), (_ZX0 + 152, sc.DECK_Y - 6))]
MOORING = (150, 126)    # where the middle mooring line meets the deck

LANDMARKS_CITY = [      # shot A callouts at the film's top-of-tilt framing (city layer = world)
    # (the camera tilts, so the tags are kept out of the HUD's top-right column: the castle's leader is
    # a little shorter than the film's and the tower's tag hangs on its left; y bands show them once
    # the camera has tilted up far enough to frame them)
    Landmark('CANTERLOT CASTLE', 186, (186, 91, 'right', 30), y=76),
    Landmark('CANTERLOT TOWER', 238, (238, 60, 'left', 12), y=44),
]
LANDMARKS_GARDEN = [    # shot B (garden layer = world; the portal sits on the 0.7 layer)
    Landmark('ROYAL GARDENS', 125, (125, 140, 'up', 16)),
    Landmark('MOUNTAIN RAILWAY', 330, (sc.PORTAL_X + 8, 61, 'up', 12), p=0.7),
]
_MAST = sc._dock()['props']['mast_top']
LANDMARKS_DOCK = [      # shot C (dock layer = world)
    Landmark('AIRSHIP DOCKS', 70, (_MAST[0], _MAST[1], 'up', 10), y=100),
]

HOTSPOTS_CITY = [
    Hotspot('bell', BELL[0], BELL[1], 'bell', cooldown=60),
    Hotspot('falls', FALLS_SPOT[0], FALLS_SPOT[1], 'splash', cooldown=100),
]
_JETS = sc._garden()['jets']
HOTSPOTS_GARDEN = [
    Hotspot('fountains', _JETS[2][3][0], _JETS[2][3][1] - 2, 'splash', cooldown=90),
    Hotspot('train', VIADUCT[0], VIADUCT[1], 'whistle', cooldown=60, p=0.7),
]
HOTSPOTS_DOCK = [
    Hotspot('airship', MOORING[0], MOORING[1], 'horn', cooldown=120),
]
STAMPS_CITY = [Stamp('summit', 128, 22)]        # in the sky beside the snowy peak of Mt. Canterhorn
STAMPS_GARDEN = [Stamp('viaduct', 300, 97)]     # under the railway viaduct, between its piers
STAMPS_DOCK = [Stamp('spire', 250, -112)]       # high above the docks, by a castle spire


def _shift_clouds(clouds, dx):
    return [(x0 + dx, y0, w, s) for (x0, y0, w, s) in clouds]


def _plume(c, f, x, y, h, spread, w, seed, k):
    """A boosted fountain jet: a w-px column h high and a crown of droplets arcing down both sides
    to the basin (y = the jet's foot)."""
    top = y - h
    for i in range(h):
        for dx in range(w):
            c.px(x - w // 2 + dx, y - i, 'white' if (f // 2 + i + dx) % 3 else 'ice')
    fx.glow(c, x, top, 7 + w, 'white', 0.22 * k)
    for d in (-1, 1):
        for i in range(16):
            ph = ((f * 0.9 + i * 1.5 + seed) % 24) / 24
            px = x + d * int(round(ph * spread))
            py = top + int(round(-5 * ph + (h + 5) * ph * ph))
            c.px(px, py, ('white', 'crystal_lt', 'ice', 'white')[(i + seed) % 4])
            if i % 3 == 0 and ph < 0.6:
                c.px(px + d, py, 'white')
    for i in range(3):                                   # the basin churns
        c.hline(x - int(spread * 0.7) + i * 2, x + int(spread * 0.7) - i * 2, y + 3 - i, 'crystal_lt' if i else 'white')


@lru_cache(maxsize=None)
def _dock_split():
    """The film's dock layer split in two: the dock (without the zeppelin and its mooring lines) and
    the zeppelin (hull, fins, gondola), so the airship can rise off its mast. The lines are drawn by
    hand (they drop and are thrown back up); pixels they had overwritten are patched."""
    D = sc._dock()
    z, al, mat = D['z'], D['alpha'], D['mat']
    lines = al & (z == 7)
    yy = np.arange(z.shape[0])[:, None]
    ship = al & ((z == 5) | (z == 6))
    ship_mat = mat.copy()
    patch = lines & (yy <= _ZCY + 16)                  # line starts that had cut into the hull
    ship |= patch
    ship_mat[patch] = sc.MID['W_SH']
    keep = lines & (yy >= sc.DECK_Y - 6)                 # line feet on the railing (same wood)
    base = dict(mat=mat, z=z, sh=D['sh'], rim=D['rim'], alpha=al & ~ship & ~(lines & ~keep))
    hull = dict(mat=ship_mat, z=z, sh=D['sh'], rim=D['rim'], alpha=ship)
    return base, hull


_ROPE = tuple(int(v) for v in sc._LUT['morn'][0][sc.MID['WD_LT']])


_NOT_SKY = np.array((255, 0, 255), np.uint8)     # sky hooks: a key no painted pixel is near


def _occlude(c, L, x, y):
    """Sky hooks: mark a scenery layer's opaque pixels as not-sky (the blue-hour rock and the pink
    castle are within the sky test's tolerance of the sky right behind them)."""
    al = L['alpha']
    lh, lw = al.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + lw), min(H, y + lh)
    if x1 > x0 and y1 > y0:
        np.copyto(c.a[y0:y1, x0:x1], _NOT_SKY, where=al[y0 - y:y1 - y, x0 - x:x1 - x, None])


def _sun_swell(s):
    """0..1..0: the extra glow round the film's sun as it clears the ridge, swelling in and out over about a
    second (the film flashed it on in one frame)."""
    return math.sin(math.pi * seg(s, sc.SUNRISE, sc.SUNRISE + 28))


@lru_cache(maxsize=256)
def _sky_city_base(cy, tsun):
    """The city's sky gradient (blue hour .. sunrise) with the far ridge and the city keyed out."""
    oy_sky = int(round(-cy * 30 / CITY_CY))
    c = Canvas()
    sc._sky(c, oy_sky, tsun, hy=126 + oy_sky)
    _occlude(c, sc._far(), 0, 92 + oy_sky)
    _occlude(c, sc._city(), 0, -cy)
    return c


@lru_cache(maxsize=256)
def _sky_city(f, cy, s, fade, s_sun=None):
    """render_city's sky at story frame s: gradient (cached with the scenery keyed out), stars, the
    pre-dawn glow on the ridge, then the sun with its glows once it has risen (at story frame s_sun: the
    evening sweep keeps the sun down, see Canterlot.sun_frame)."""
    s_sun = s if s_sun is None else s_sun
    oy_sky = int(round(-cy * 30 / CITY_CY))
    c = _sky_city_base(cy, smooth(seg(s, sc.SUNRISE - 12, 204))).copy()
    if fade < 1.0:
        sc._stars(c, f, oy_sky, fade)
    sx = 300
    ry = sc._ridge_y(sx) + 92 + oy_sky
    pre = seg(s_sun, 90, sc.SUNRISE)
    fx.glow(c, sx, ry + 4, 80, '#ff8a74', 0.08 + 0.20 * pre)
    if s_sun >= sc.SUNRISE:
        sy = int(ry + 3 - (s_sun - sc.SUNRISE) * 0.30)
        flash = _sun_swell(s_sun)
        fx.glow(c, sx, sy, 50, 'gold', (0.30 + 0.25 * flash) * seg(s_sun, sc.SUNRISE - 2, sc.SUNRISE + 12))
        fx.glow(c, sx, sy, 18, 'butter', (0.45 + 0.4 * flash) * seg(s_sun, sc.SUNRISE - 2, sc.SUNRISE + 12))
        sc._SUNB.draw(c, sx, sy + 5, anchor='bottom')
    return c


@lru_cache(maxsize=16)
def _sky_garden_raw(gx):
    """render_garden's sky (gradient and the low sun's glow, centred at x gx) before any scenery."""
    c = Canvas()
    sc._sky(c, 0, 1.0, morn=0.35, hy=118)
    fx.glow(c, gx, 70, 150, 'gold', 0.20)
    return c


@lru_cache(maxsize=128)
def _sky_garden(cam):
    c = _sky_garden_raw(340 - cam // 10).copy()
    _occlude(c, sc._far(), int(round(-cam * 0.25)), 66)
    _occlude(c, sc._rail(), _rail_x(cam), 0)
    _occlude(c, sc._castle_b(), int(round(-cam * 0.6)), -2)
    _occlude(c, sc._garden(), -cam, 0)
    c.a[SEAM_Y:] = _NOT_SKY     # (see _fill_seam)
    return c


SEAM_Y = 160


def _fill_seam(c, cam):
    """A seam between the rock spur and the hedges at the foot of the gardens shows the sky: a pale square
    by day, the clock's horizon colour (an orange square) at dusk, a pale one at night. Filled with the
    hedge just right of it."""
    sky = _sky_garden_raw(340 - cam // 10).a[SEAM_Y:]
    lo = c.a[SEAM_Y:]
    hole = (lo == sky).all(axis=2)
    hole[:, -1] = False
    for y in np.nonzero(hole.any(axis=1))[0]:
        xs = np.nonzero(hole[y])[0]
        lo[y, xs] = lo[y, xs[-1] + 1]


@lru_cache(maxsize=256)
def _sky_dock(dy, hz=1.0):
    tt = dy / DOCK_RISE
    c = Canvas()
    sc._sky(c, int(round(-10 + 10 * tt)), 1.0, morn=1.0, hy=126)
    sx, sy = 300, 86
    fx.glow(c, sx, sy, 70, 'butter', 0.22)
    fx.glow(c, sx, sy, 20, 'white', 0.35)
    sc._SUNB.draw(c, sx, sy + 5, anchor='bottom')
    if hz > 0:
        c.blend(0, 0, W, H, '#d4ecff', 0.20 * (1 - 0.6 * tt) * hz)    # the render's morning haze
    _occlude(c, sc._far(), 0, 92 - int(round(12 * (1 - tt))))
    _occlude(c, sc._city(), 0, int(round(-30 + 30 * tt)))
    return c


def _layer(get, mode):
    """A getter for an indexed scene layer drawn in `mode`, as RGBA for game/nightwin.py."""
    def f():
        L = get()
        return np.dstack([sc._rgb(L, mode), L['alpha'].astype(np.uint8) * 255])
    return f


# ----------------------------------------------------------------------------- the clock's light
# The relight keeps no painted spot glowing here (auto_glow False: the gold trims, domes, pennants and
# the morning rims are paint); every view keeps its REAL lights itself (GlowKeep's keep_glow, restored by
# Canterlot.after_light, ramped by the night): the windows (the film's glass materials WIN / WIN_HI: the
# tall hall windows are too big for game/nightwin.py's pane test), the film's own lit windows and lamps
# in the blue hour, the stained-glass roses, the pier lamps and the mast beacon. No halos: only the
# light itself is lit.
_LAYERS = {'city': sc._city, 'castle_b': sc._castle_b, 'dock': lambda: _dock_split()[0],
           'hull': lambda: _dock_split()[1]}
WIN_SHARE = {'city': 0.72, 'castle_b': 0.8, 'dock': 0.85, 'hull': 1.0}   # windows lit in full night
_WIN = (sc.MID['WIN'], sc.MID['WIN_HI'])
_SG = tuple(sc.MID[k] for k in ('SG_A', 'SG_B', 'SG_C', 'SG_D'))
_GLINT = np.array(col('#fff0b8'), np.uint8)            # a lit window's highlight (the glass's WIN_HI px)
NIGHT_GLOW = 1.5                                       # glow restored = clamp(night x this)
LAMP_OFF = '#cfc6b4'                                   # a pier lamp's glass by day (unlit)
_LAMP_COLS = np.array([col(c) for c in ('gold', 'amber', 'butter')], np.uint8)   # the film's lit windows, lamps


@lru_cache(maxsize=None)
def _panes(name):
    """(ys, xs, v, hi, pane) of a layer's window glass: its pixels, the night ramp at which each window
    goes on (one per window, hashed from its corner), whether the pixel is the glass's highlight, and
    the window's number."""
    from scipy import ndimage
    L = _LAYERS[name]()
    mat = L['mat']
    glass = L['alpha'] & np.isin(mat, _WIN)
    lab, n = ndimage.label(glass)
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    v = np.array([0.0] + [hash01(s[1].start, s[0].start, 406) for s in ndimage.find_objects(lab)],
                 np.float32) / WIN_SHARE[name]
    return ys, xs, v[ids], mat[ys, xs] == sc.MID['WIN_HI'], ids


def _windows(c, name, modes, x, y, hour):
    """Light the windows of layer `name` (drawn at x, y in one of `modes`) as the night deepens
    (game/nightwin.py's ramp), where the layer still shows (nothing drawn in front). Returns the mask."""
    k = nightwin.ramp(hour)
    if k <= 0:
        return None
    ys, xs, v, hi, ids = _panes(name)
    on = v < k
    ys, xs, hi, ids = ys[on], xs[on], hi[on], ids[on]
    sy, sx = ys + y, xs + x
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if not ok.any():
        return None
    ys, xs, hi, ids, sy, sx = ys[ok], xs[ok], hi[ok], ids[ok], sy[ok], sx[ok]
    L = _LAYERS[name]()
    px = c.a[sy, sx]
    seen = np.zeros(len(sy), bool)
    for mode in modes:
        seen |= (px == sc._rgb(L, mode)[ys, xs]).all(axis=1)
    if not seen.any():
        return None
    sy, sx, hi, ids = sy[seen], sx[seen], hi[seen], ids[seen]
    c.a[sy, sx] = np.where(hi[:, None], _GLINT, nightwin.WARM[ids % 3])
    m = np.zeros((H, W), bool)
    m[sy, sx] = True
    return m


def _roses(c, L, roses, x, y, mode):
    """Stained-glass roses lit from inside at night: their glass in its vivid sunlit colours (the blue
    hour paints them muted). Returns the mask."""
    m = np.zeros((H, W), bool)
    lut = sc._LUT['sun'][0]
    mat = L['mat']
    for ro in roses:
        rx, ry, r = ro if len(ro) == 3 else (ro[0], ro[1], 5)
        x0, y0, x1, y1 = rx - r, ry - r, rx + r + 1, ry + r + 1
        sub = mat[max(0, y0):y1, max(0, x0):x1]
        sg = np.isin(sub, _SG) & L['alpha'][max(0, y0):y1, max(0, x0):x1]
        yy, xx = np.nonzero(sg)
        yy, xx = yy + max(0, y0), xx + max(0, x0)
        sy, sx = yy + y, xx + x
        ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        yy, xx, sy, sx = yy[ok], xx[ok], sy[ok], sx[ok]
        seen = (c.a[sy, sx] == sc._rgb(L, mode)[yy, xx]).all(axis=1)
        c.a[sy[seen], sx[seen]] = lut[mat[yy[seen], xx[seen]]]
        m[sy[seen], sx[seen]] = True
    return m


def _haze(hour):
    """The docks' morning haze by the clock: full in the morning (as painted), a little by day, none at
    golden hour, dusk and night (it washed the night city out to a milky blue)."""
    h = hour % 24
    k = seg(h, 5.6, 7.0) * (1 - 0.65 * seg(h, 9.5, 12.0)) * (1 - seg(h, 16.0, 17.5))
    return round(k * 20) / 20


class Canterlot(GlowKeep, Region):
    SEG = 'canterlot'
    CARD = ('ROYAL CAPITAL', ['ON MT. CANTERHORN', 'CASTLE - GARDENS - FALLS'])

    def build_areas(self):
        return [
            Area(self.render_city, cam_x=(0, 0), cam_y=(0, CITY_CY), y_range=(30, 214),
                 start=(48, 188), landmarks=LANDMARKS_CITY, balloon='m', balloon_tint=TINT_BLUE,
                 hotspots=HOTSPOTS_CITY, stamps=STAMPS_CITY, sky=self.sky_city),
            Area(self.render_garden, cam_x=(0, GARDEN_CAM), cam_y=(0, 0), y_range=(34, 128),
                 start=(60, 70), landmarks=LANDMARKS_GARDEN, balloon='m',
                 hotspots=HOTSPOTS_GARDEN, stamps=STAMPS_GARDEN, sky=self.sky_garden),
            Area(self.render_dock, cam_x=(0, 0), cam_y=(-DOCK_RISE, 0), y_range=(-106, 136),
                 start=(70, 100), landmarks=LANDMARKS_DOCK, balloon='l',
                 hotspots=HOTSPOTS_DOCK, stamps=STAMPS_DOCK, sky=self.sky_dock),
        ]

    # ------------------------------------------------------------------ state + clocks
    # the hour each view is painted at (game/daylight.py re-lights from it to the world clock)
    @property
    def HOUR(self):
        ai = getattr(self, 'ai', 0)
        if ai == 1:
            return HOUR_GARDEN
        if ai == 2:
            return HOUR_DOCK
        if not hasattr(self, 'sun_go'):
            return HOUR_BLUE
        return HOUR_BLUE + (HOUR_SUN - HOUR_BLUE) * seg(self.view_s(self.g.t), sc.SUNRISE, SUN_END)

    @HOUR.setter
    def HOUR(self, v):              # the base sets the film's hour (06:00): the views above replace it
        self._film_hour = v

    def clock_night(self):
        h = self.g.hour % 24
        return h >= NIGHT_FROM or h < 5.0

    def clock_day(self):
        return DAY_FROM <= self.g.hour % 24 < NIGHT_FROM

    def init_state(self):
        self.dusk_go = None         # step the evening (the sunrise played backwards) started
        self.sun_go = None          # step the sunrise clock started (story frame SUN_HOLD)
        self.area_t0 = 0            # step the current area was entered (sound cadences)
        self.trains = []            # one start step per train on the line (its clock = TRAIN_LEAD)
        self.train_last = -10 ** 6  # step the last train was sent
        self.blast_t0 = None        # step the train last whistled and blew steam on request
        self.ship = None            # None (moored) | ('up' | 'down', step it started)

    def on_enter(self):
        self.sun_go = None          # landings start in the blue hour (dawn, night)...
        self.dusk_go = None
        if self.clock_day():        # ...or in full sun by day
            self.sun_go = self.g.t - 400
        self.ship = None
        self.areas[0].balloon_tint = TINT_BLUE

    def on_area(self, ai):
        self.area_t0 = self.g.t
        if ai == 1 and all(tf > 600 for tf in self.train_fs(self.g.t)):
            self.send_train(self.g.t)       # a train comes out of the tunnel as you arrive

    def warm(self):
        _dock_split()
        super().warm()

    def sun_s(self, t):
        if self.dusk_go is not None:                     # evening: the sweep runs backwards
            return max(SUN_HOLD, SUN_END - max(0, t - self.dusk_go))
        if self.sun_go is None or t < self.sun_go:
            return SUN_HOLD
        return min(SUN_END, SUN_HOLD + t - self.sun_go)

    def view_s(self, t):
        """The story frame the city is drawn at: sun_s, but always true to the world clock (update
        starts the evening and the morning on the clock's step; until it has, e.g. when the clock was
        set from outside, the city must not show the sunlit view at night nor the blue hour by day)."""
        if self.dusk_go is None and self.sun_go is not None:
            if self.clock_night():
                return SUN_HOLD
            if self.g.hour % 24 < 5.9 and self.sun_go != self.spot_t.get('bell', -999) - 10:
                return SUN_HOLD                         # (before 05:54 only the bell brings the sun up)
        if self.sun_go is None and self.dusk_go is None and self.clock_day():
            return SUN_END
        return self.sun_s(t)

    def train_fs(self, t):
        """Clocks (scene frames of the film's train) of every train on the line, newest last."""
        return [TRAIN_LEAD + t - t0 for t0 in self.trains if t >= t0]

    def send_train(self, t, tf=TRAIN_LEAD):
        self.trains.append(t - (tf - TRAIN_LEAD))
        self.train_last = t

    def ship_pose(self, t):
        """(dy of the airship, mooring-line extension 0..1, engines running)."""
        if self.ship is None:
            return 0, 1.0, False
        kind, t0 = self.ship
        a = t - t0
        bob = math.sin(t / 22.0) * 2.0
        if kind == 'up':
            dy = -SHIP_RISE * ease_in_out(seg(a, 4, SHIP_UP)) + bob * seg(a, 44, 84)
            return int(round(dy)), 1 - seg(a, 0, 10), True
        e = ease_in_out(seg(a, 0, SHIP_LAND))
        dy = -SHIP_RISE * (1 - e) + bob * (1 - seg(a, 0, 60))
        return int(round(dy)), seg(a, SHIP_LAND + 2, SHIP_DOWN), a < SHIP_LAND

    # ------------------------------------------------------------------ events
    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        night = self.clock_night()
        if night and self.dusk_go is None and self.sun_go is not None and self.sun_s(t) >= SUN_END:
            self.dusk_go = t if self.ai == 0 else t - 400     # dusk: the light leaves the city
        elif not night and self.dusk_go is not None and self.sun_s(t) <= SUN_HOLD:
            self.dusk_go = self.sun_go = None                  # a new dawn: blue hour again
        if self.sun_go is None and not night:
            if self.ai != 0:
                if self.g.hour % 24 >= 5.9:     # left the city before dawn: it's morning now (at the
                    self.sun_go = t - 400       # hour the city's own sunrise would have come)
            elif (t - self.t_enter > SUN_TRIGGER_T and self.g.hour % 24 >= 5.9) or self.clock_day():
                self.sun_go = t
        s = self.sun_s(t)
        k = seg(s, sc.SUNRISE, 200)
        self.areas[0].balloon_tint = None if k >= 1 else (TINT_BLUE[0], TINT_BLUE[1] * (1 - k))
        if self.ship is not None:
            kind, t0 = self.ship
            if kind == 'up' and t - t0 >= SHIP_STAY:
                self.ship = ('down', t)
                g.sfx('horn', 0.3 * (1 if self.ai == 2 else 0.2))
            elif kind == 'down' and t - t0 >= SHIP_DOWN:
                self.ship = None
                if self.ai == 2:
                    g.sfx('creak', 0.35)
        if self.ai == 0:
            ds = None if self.sun_go is None else t - self.sun_go + SUN_HOLD   # story frame now
            if ds == sc.SUNRISE:
                g.sfx('bell', 0.55)
            elif ds == sc.SUNRISE + 2:
                g.sfx('shimmer', 0.45)
            elif ds == 165:
                g.sfx('sparkle', 0.4)
            if (t - self.area_t0) % 60 == 10:
                g.sfx('waterfall', 0.14 + 0.12 * self.cy / CITY_CY)
        self.trains = [t0 for t0 in self.trains if TRAIN_LEAD + t - t0 < TRAIN_END]
        if self.ai == 1 and not self.trains and t - self.train_last > TRAIN_AUTO:
            self.send_train(t)
        if self.ai == 1:
            tfs = self.train_fs(t)
            gain = self.near_gain(SPUR_X + 40)
            for j, tf in enumerate(tfs):
                if tf == sc.TRAIN_T0:
                    g.sfx('whistle', 0.6 * max(0.4, gain))
                elif tf == sc.TRAIN_T0 + 2:
                    g.sfx('steam', 0.4 * gain)
                elif (j == len(tfs) - 1 and sc.TRAIN_T0 + 9 <= tf < sc.TRAIN_T0 + 400
                      and (tf - sc.TRAIN_T0 - 9) % 36 == 0):         # the newest train chugs
                    k = (tf - sc.TRAIN_T0 - 9) // 36
                    g.sfx('chug', 0.42 * gain * max(0.25, 1 - k * 0.08), k % 2)
            if (t - self.area_t0) % 90 == 20:
                g.sfx('fountain', 0.22 * self.near_gain(125))
        elif self.ai == 2:
            if (t - self.area_t0) % 96 == 30 and self.cy > -70 and self.ship is None:
                g.sfx('creak', 0.22, (t // 96) % 2)
            if (t - self.area_t0) % 150 == 75 and self.cy < -80:
                g.sfx('wind', 0.25, 0)

    def on_interact(self, key, t):
        g = self.g
        if key == 'bell':
            g.sfx('bell', 0.65)
            if self.sun_go is None and not self.clock_night():   # ring in the dawn: the sun breaks
                self.sun_go = t - 10                               # the ridge in ~1.3 s
            else:
                g.sfx('shimmer', 0.35)
        elif key == 'falls':
            g.sfx('waterfall', 0.55)
            g.sfx('splash', 0.45)
            g.sfx('sparkle', 0.3)
        elif key == 'fountains':
            g.sfx('fountain', 0.5)
            g.sfx('splash', 0.35, 1)
        elif key == 'train':
            tfs = self.train_fs(t)
            if not tfs or tfs[-1] >= TRAIN_CLEAR:          # the line is clear: a train comes out now
                self.send_train(t, sc.TRAIN_T0 - 6)
                g.sfx('steam', 0.3)
            elif tfs[-1] < sc.TRAIN_T0 - 6:                # one is still in the tunnel: hurry it out
                self.trains[-1] = t - (sc.TRAIN_T0 - 6 - TRAIN_LEAD)
                g.sfx('steam', 0.3)
            else:                                          # just out: whistle and a blast of steam
                self.blast_t0 = t
                g.sfx('whistle', 0.75)
                g.sfx('steam', 0.4)
        elif key == 'airship':
            if self.ship is None:                          # slip the lines: up she goes
                self.ship = ('up', t)
                g.sfx('horn', 0.6)
                g.sfx('whoosh', 0.3)
            elif self.ship[0] == 'up':                     # aloft: call her back to the mast
                self.ship = ('down', t)
                g.sfx('horn', 0.45)
            else:
                g.sfx('tick', 0.3)

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    # Each reproduces only the sky steps its render draws first (gradient, stars, sun glow and disc,
    # the docks' haze); the static part is cached per camera with the scenery layers keyed out.
    def sky_city(self, f, cx, cy):
        s = self.view_s(self.g.t)
        cy = int(clamp(cy, 0, CITY_CY))
        fade = clamp(seg(s, 84, 176) * 1.1)
        if fade >= 1.0:                     # no star can show (vis <= 1): the whole sky is cached
            return _sky_city(-1, cy, s, 1.0, self.sun_frame(s))
        return _sky_city.__wrapped__(f, cy, s, fade, self.sun_frame(s))

    def sun_frame(self, s):
        """The story frame the film's sun is drawn at: s, except in the evening sweep (the sunrise played
        backwards as the light leaves the city at dusk), where the clock's sun has already set: the film's
        rising sun, its flash and burst rays would set on the right and vanish in one frame at SUNRISE."""
        return s if self.dusk_go is None else min(s, sc.SUNRISE - 1)

    def sky_garden(self, f, cx, cy):
        return _sky_garden(int(clamp(cx, 0, GARDEN_CAM)))

    def sky_dock(self, f, cx, cy):
        return _sky_dock(int(clamp(-cy, 0, DOCK_RISE)), _haze(self.g.hour))

    # ------------------------------------------------------------------ the clock's light
    def auto_glow(self):
        return False                # every view keeps its real lights itself (see _windows)

    def _drawn(self, c, f, cx, cy):
        """(The area render notes its frame and camera: after_light repaints its sky. Where the view wants
        the clock's sky and the base keeps the painted one, its open sky is found here, before the
        clock's light: the sky test's colour step alone against the sky hook, a fraction of the cost
        of relighting the hook and running the full test after the light.)"""
        self._drawn_at = (self.ai, f, cx, cy)
        self._sky_pre = None
        hour = self.g.hour
        want = self.sky_want(hour)
        if want is not None and want >= 0.99 and daylight.sky_weight(self.HOUR, hour) < 0.99:
            hook = self.area.sky(f, cx, cy).a
            self._sky_pre = (self.ai, np.abs(c.a.astype(np.int16) - hook).sum(axis=2) < daylight.SKY_TOL)

    def sky_want(self, hour):
        """The clock's-sky weight a view wants where the base keeps (part of) the painted sky, or None
        for the base's own (daylight.sky_weight). The film's skies carry its sun low in the east (right)
        with its glow, glare and haze, in a pink morning tint: wrong for the clock's day, whose sun
        rises on the left. The gardens and the docks always take the clock's sky; the city keeps its
        painted sunrise only while the sunrise plays at dawn (fading to the clock's sky at its end),
        and takes the clock's sky once the painted sun stands over the ridge."""
        if self.ai != 0:
            return 1.0
        s = self.view_s(self.g.t)
        if s >= SUN_END:
            return 1.0
        h = hour % 24
        if self.dusk_go is None and s >= sc.SUNRISE - 12 and 5.6 <= h < 6.8:
            return seg(s, SUN_END - 36, SUN_END)
        return None

    def after_light(self, c, t):
        """After the clock's light: the view's sky as sky_want asks, then its own lights get their colour
        back (GlowKeep's keep_glow masks; restored by the clock's night alone, so the city's blue-hour
        windows shine as bright at 19:30 as at midnight)."""
        self._sky_fix(c, t)
        g, self._glow = self._glow, None
        if g is None or (self.ID, self.ai) in daylight.INTERIORS:
            return
        pre, mask, k = g
        hour = self.g.hour
        kk = k * clamp(daylight.night(hour) * NIGHT_GLOW)
        if kk < 0.02 or abs(daylight._q(hour) - daylight._q(self.HOUR)) < 0.05:
            return                                      # (unlit: the frame is as painted)
        ys, xs = np.nonzero(mask)
        y0, y1 = max(0, ys.min() - 4), min(H, ys.max() + 5)
        x0, x1 = max(0, xs.min() - 4), min(W, xs.max() + 5)
        p = pre[y0:y1, x0:x1]
        e = p.copy()
        daylight.relight(e, self.HOUR, hour, t, glow=False)                  # what the base made of them
        sub = c.a[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1] & (sub == e).all(axis=2)                     # ...and nothing drawn over
        if m.any():
            ef = e[m].astype(np.float32)
            sub[m] = (ef + (p[m].astype(np.float32) - ef) * kk).astype(np.uint8)

    def _sky_fix(self, c, t):
        A = self.area
        d = getattr(self, '_drawn_at', None)
        hour = self.g.hour
        want = self.sky_want(hour)
        if want is None or d is None or d[0] != self.ai:
            return
        w0 = daylight.sky_weight(self.HOUR, hour)
        if abs(want - w0) < 0.01:
            return
        _ai, f, ix, iy = d
        pre, self._sky_pre = getattr(self, '_sky_pre', None), None
        if want >= 0.99 and pre is not None and pre[0] == self.ai:     # the clock's sky over all of it
            y0, y1 = A.cam_y
            daylight.paint_sky(c.a, pre[1], hour, 1.0, ix, iy, y0 - 10, y1 - y0 + 130, t)
            return
        hook = A.sky(f, ix, iy).a.copy()
        key = (hook == _NOT_SKY).all(axis=2)
        daylight.relight(hook, self.HOUR, hour, t, glow=False)               # the painted sky, as lit now
        hook[key] = _NOT_SKY
        if w0 > 0.01 and self._sky_memo is not None:   # the base painted the clock's sky: back to the
            mask = self._sky_memo[1]                    # painted one where it did
            np.copyto(c.a, hook, where=mask[..., None])
        else:
            mask = daylight.sky_pixels(c.a, hook)
        if want > 0.01 and mask.any():
            y0, y1 = A.cam_y
            daylight.paint_sky(c.a, mask, hour, want, ix, iy, y0 - 10, y1 - y0 + 130, t)

    # ------------------------------------------------------------------ area 0: the city
    def render_city(self, f, cx, cy):
        t = self.g.t
        s = self.view_s(t)
        cy = int(clamp(cy, 0, CITY_CY))
        oy_city = -cy
        oy_sky = int(round(-cy * 30 / CITY_CY))
        oy_far = 92 + oy_sky
        oy_val = int(round(240 - cy * 122 / CITY_CY))
        oy_fg = int(round(266 - cy * 150 / CITY_CY))
        tsun = smooth(seg(s, sc.SUNRISE - 12, 204))
        m, band, lit_all = _sun_mask(s, cy)
        mode_a, mode_b = ('sun', None) if lit_all else ('dawn', 'sun')
        # the bell rung in full sun: a wave of gold light runs over the city
        ab = t - self.spot_t.get('bell', -999)
        wave = lit_all and 0 <= ab < 60 and self.sun_s(self.spot_t['bell']) >= SUN_END
        if wave:
            th = lerp(430, -140, seg(ab, 0, 54))
            sco = _score_rows(cy)
            m, band, mode_b = _ONES, (sco > th) & (sco < th + 12), 'sun'

        def lit(x, y):
            if lit_all:
                return True
            return m is not None and bool(m[min(H - 1, max(0, y)), min(W - 1, max(0, x))])

        c = Canvas()
        sc._sky(c, oy_sky, tsun, hy=126 + oy_sky)
        sc._stars(c, f, oy_sky, clamp(seg(s, 84, 176) * 1.1))
        sx = 300
        ry = sc._ridge_y(sx) + oy_far
        s_sun = self.sun_frame(s)
        pre = seg(s_sun, 90, sc.SUNRISE)
        fx.glow(c, sx, ry + 4, 80, '#ff8a74', 0.08 + 0.20 * pre)
        if s_sun >= sc.SUNRISE:
            sy = int(ry + 3 - (s_sun - sc.SUNRISE) * 0.30)
            flash = _sun_swell(s_sun)
            fx.glow(c, sx, sy, 50, 'gold', (0.30 + 0.25 * flash) * seg(s_sun, sc.SUNRISE - 2, sc.SUNRISE + 12))
            fx.glow(c, sx, sy, 18, 'butter', (0.45 + 0.4 * flash) * seg(s_sun, sc.SUNRISE - 2, sc.SUNRISE + 12))
            sc._SUNB.draw(c, sx, sy + 5, anchor='bottom')
        sc._wisps(c, f, oy_sky, tsun)
        sc._draw_layer(c, sc._far(), 0, oy_far, mode_a, mode_b, m, band)
        if sc.SUNRISE <= s_sun < sc.SUNRISE + 44:        # burst rays over the far ridge (fading in, not a flash)
            k = (1 - seg(s_sun, sc.SUNRISE, sc.SUNRISE + 44)) * seg(s_sun, sc.SUNRISE, sc.SUNRISE + 10)
            for a in range(7):
                ang = math.pi * (0.70 + a * 0.10) + s_sun * 0.003
                x2 = sx + int(math.cos(ang) * 150)
                y2 = ry + int(math.sin(ang) * 150)
                fx.light_beam(c, ((sx - 1, ry), (sx + 1, ry)), ((x2 - 5, y2), (x2 + 5, y2)), 'butter', 0.10 * k)
        L = sc._city()
        sc._draw_layer(c, L, 0, oy_city, mode_a, mode_b, m, band)
        if wave:                                         # the wave front glows gold on the stonework
            rows = L['alpha'][cy:cy + H]
            n = rows.shape[0]
            bm = band[:n] & rows
            if bm.any():
                sub = c.a[:n]
                sub[bm] = (sub[bm] * 0.5 + np.array(col('butter'), np.float32) * 0.5).astype(np.uint8)
        af = t - self.spot_t.get('falls', -999)
        surge = 0.0 if not 0 <= af < 96 else seg(af, 0, 8) * (1 - seg(af, 64, 96))
        sc._falls(c, f, 0, oy_city, mode_a, m)
        if surge > 0:                                    # the falls swell: wider, faster water
            sc._falls(c, f, 0, oy_city, mode_a, m, speed=4,
                      falls=[(x - 1, ft, fw + 2, fb) for (x, ft, fw, fb) in sc.FALLS])
        for i, (fx_, ft, fw, fb) in enumerate(sc.FALLS):
            sc._mist(c, f, fx_ + fw // 2, fb + oy_city, 12 + fw * 2, tsun, seed=i)
            if surge > 0.3:
                sc._mist(c, f + 17, fx_ + fw // 2, fb + oy_city - 2, int((20 + fw * 3) * surge), tsun, seed=i + 7)
        for k in range(6):                               # plunge-pool shimmer
            x = 96 + int((hash01(k, 3) * 180 + f * 0.5 * (1 if k % 2 else -1)) % 180)
            c.hline(x, x + 3, POOL_Y + oy_city, '#8a9ee0')
        # (the film's warm halos round its lit windows and lamps and the roses' magic glow are left
        # out: at the clock's night they only dimmed to murky rings over the stone; the lights
        # themselves are kept bright instead, see the end of this render)
        lamps = np.zeros((H, W), bool)
        for i, ((x, y, w, h), off) in enumerate(zip(L['wins'], L['offs'])):
            if s < off:                                  # windows go dark behind the sweep
                sc._lit_window(c, x, y + oy_city, w, h, f, i)
                lamps[max(0, y + oy_city):max(0, y + oy_city + h), x:x + w] = True
        for (rx, ry_, r) in L['roses']:                  # stained glass
            y = ry_ + oy_city
            if lit(rx, y):
                fx.sparkle(c, rx + 3, y - 3, f, 'white', period=22, size=3)
                fx.sparkle(c, rx - 3, y + 2, f + 11, 'butter', period=22, size=2)
            if wave and 6 <= ab < 46:
                fx.glow(c, rx, y, 12, 'magic', 0.4 * (1 - seg(ab, 6, 46)))
        for i, (lx, ly) in enumerate(L['lamps']):        # terrace lamps
            y = ly + oy_city
            on = s < 184 + int(hash01(i, 7) * 22)
            c.px(lx, y, 'butter' if on else 'brass')
            c.px(lx, y - 1, '#3a3048')
            if on and 0 <= y < H:
                lamps[y, lx] = True
        for i, (px_, py) in enumerate(L['flags']):       # pennants on every spire
            on = lit(px_, py + oy_city)
            sc._flag(c, px_, py + oy_city, f, i, ('gold' if i % 2 else 'magic') if on else
                     ('#6e6474' if i % 2 else '#5a3a7a'))
            if wave and band[min(H - 1, max(0, py + oy_city + 2)), min(W - 1, max(0, px_))]:
                fx.sparkle(c, px_ + 1, py + oy_city - 1, 2, 'white', period=8, size=3)   # glint as it passes
        for (bx, by, kind, i, bl, bw) in L['banners']:   # banners unfurl as the light reaches them
            tsw = sc._sweep_time(sc._score(bx, by + 4))
            t0 = max(sc.SUNRISE + 9, ((tsw + 8) // 9 + 1) * 9)
            u = sc.ease_out_bounce(seg(s, t0, t0 + 10))
            sc._banner(c, bx, by + oy_city - 1, f, kind, i, not lit(bx, by + oy_city), u, bl, bw)
        # (the film's moored balloon at the cliff dock is the player's: not drawn)
        if oy_val < H:                                   # valley + foreground
            V = sc._valley()
            sc._draw_layer(c, V, 0, oy_val, mode_a, mode_b, m, None)
            for i, (x, y, w, h) in enumerate(V['wins']):
                if hash01(i, f // 12) > 0.12 and s < 200 + int(hash01(i, 9) * 30):
                    c.px(x, y + oy_val, 'gold')
                    if 0 <= y + oy_val < H:
                        lamps[y + oy_val, x] = True
            for k in range(6):                           # Saddle Lake shimmer
                x = 210 + int((hash01(k, 8) * 90 + f * 0.4) % 90)
                c.hline(x, x + 2, oy_val + 35 + k % 3 * 3, '#c48cac' if k % 2 else '#8a72aa')
            for k in range(3):                           # chimney smoke in Ponyville
                fx.steam(c, 34 + k * 22, oy_val + 24, f, '#5c5c8c', n=3, height=10, seed=k)
        if oy_fg < H:
            sc._draw_layer(c, sc._fore(), 0, oy_fg, mode_a, mode_b, m, None)
        if sc.SUNRISE - 6 <= s_sun < sc.SUNRISE + 30:    # the sun's light swells over the ridge (the film's
            k = math.sin(math.pi * seg(s_sun, sc.SUNRISE - 6, sc.SUNRISE + 30))     # sudden burst was a flash)
            fx.glow(c, 300, sc._ridge_y(300) + oy_far, 210, '#ffd8a0', 0.14 * k, rings=6)
        self._bell_fx(c, t, oy_city)
        if surge > 0:
            self._falls_fx(c, t, af, surge, oy_city, s)
        self._drawn(c, f, 0, cy)
        hour = self.g.hour
        if daylight.night(hour) > 0.02:                 # the city's lights by night
            self.keep_glow(c, _windows(c, 'city', (mode_a, mode_b or mode_a), 0, oy_city, hour), 0.95)
            lamps &= (c.a[:, :, None, :] == _LAMP_COLS[None, None]).all(axis=3).any(axis=2)   # (still seen)
            self.keep_glow(c, lamps, 0.95)
            if s < sc.SUNRISE + 30:                      # the roses lit from inside (until the sun is up)
                self.keep_glow(c, _roses(c, L, L['roses'], 0, oy_city, mode_a), 0.9)
        return c

    def _bell_fx(self, c, t, oy):
        a = t - self.spot_t.get('bell', -999)
        if not 0 <= a < 54:
            return
        bx, by = BELL[0], BELL[1] + oy
        fx.glow(c, bx, by, 16, 'gold', 0.55 * (1 - a / 54))
        if a < 24 and (a // 3) % 2 == 0:                 # the bell swings
            c.circle(bx, by, 2, 'butter')
        for k in range(3):                               # the bell's peal rings out from the belfry
            r = (a - k * 9) * 1.3 + 4
            if 4 <= r < 36:
                c.circle(bx, by, int(r), 'butter' if k == 0 else 'white', fill=False)
        if a < 30:
            fx.sparkles(c, bx, by, t, radius=10, n=6, colors=('white', 'gold', 'butter'), seed=3, period=10)

    def _falls_fx(self, c, t, a, k, oy, s):
        """Surge: spray bursts at every fall's foot and a rainbow over the plunge pool."""
        t0 = t - a
        for i, (fx_, ft, fw, fb) in enumerate(sc.FALLS):
            ui.burst(c, t, t0 + i * 3, fx_ + fw // 2, fb + oy - 2, ('white', 'ice', 'crystal_lt', 'light_sky'),
                     n=14, seed=20 + i, power=0.55, life=44, gravity=0.06)
        rx, ry, r0 = RAINBOW
        ry += oy
        rev = seg(a, 4, 26)                              # the arc draws itself left to right
        al = 0.62 * k * (0.7 + 0.3 * smooth(seg(s, sc.SUNRISE, 200)))
        if al <= 0.02 or rev <= 0:
            return
        y0, y1 = max(0, ry - r0 - 1), min(H, POOL_Y + oy - 1, ry)
        x0, x1 = max(0, rx - r0 - 1), min(W, rx + r0 + 1)
        if y1 <= y0 or x1 <= x0:
            return
        dx = _XX[y0:y1, x0:x1] - rx
        dy = _YY[y0:y1, x0:x1] - ry
        d = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(-dy, -dx)                       # 0 at the left foot, pi at the right one
        show = ang <= math.pi * rev
        dst = c.a[y0:y1, x0:x1].astype(np.float32)
        for i, cc in enumerate(_RB_COLS):
            mm = show & (d >= r0 - 2 * (i + 1)) & (d < r0 - 2 * i)
            if mm.any():
                dst[mm] = dst[mm] * (1 - al) + cc * al
        c.a[y0:y1, x0:x1] = dst.astype(np.uint8)

    # ------------------------------------------------------------------ area 1: the gardens
    def _trains(self, c, trains, ox):
        """The film's train on the corkscrew, past the film's half turn: cars on the back passes go
        behind the rock spur (rail-layer depth), puffs too. trains = [(train clock, steps since a
        requested whistle or None)]; all trains share one back and one front overlay."""
        trains = [q for q in trains if q[0] >= sc.TRAIN_T0 - 2]
        if not trains:
            return
        R = sc._rail()
        z = R['z']
        clip_x = int(sc.PORTAL_X - 5 + ox)
        sent = (1, 2, 3)
        back = Canvas(W, H, sent)
        front = Canvas(W, H, sent)

        def behind_at(s_):
            return s_ > 38 and ((s_ - 38) / sc.TRACK_R) % (2 * math.pi) > math.pi

        for tf, blast in trains:
            se = sc._train_s(tf)
            for s_, spr, half in ((se, 0, 8), (se - 17, 1, 7), (se - 32, 1, 7), (se - 47, 1, 7)):
                behind = behind_at(s_ + half / 2)
                x, y = sc._train_path(s_ + half)
                x2, y2 = sc._train_path(s_)
                px = int(round((x + x2) / 2 + ox))
                py = int(round(max(y, y2)))
                if py - 10 > H:
                    continue
                _CARS[spr][behind].draw(back if behind else front, px, py, anchor='bottom')
            # stack puffs on every half-beat while the train runs (+ a blast when asked to whistle)
            out = []
            k0 = max(0, (tf - sc.TRAIN_T0 - 34) // 9)
            for k in range(k0, k0 + 6):
                te = sc.TRAIN_T0 + k * 9
                if tf < te:
                    break
                age = tf - te
                if age > 34:
                    continue
                ss = sc._train_s(te) + 3
                x, y = sc._train_path(ss)
                r = 1.5 + age * 0.16
                out.append((x + ox + 2 + age * 0.35, y - 10 - age * 0.55, r,
                            'white' if age < 12 else ('cloud' if age < 22 else 'cloud_dk'), behind_at(ss)))
            if blast is not None and 0 <= blast < 56:
                for j in range(0, min(blast, 26) + 1, 2):    # a puff every 2 steps for the first 26
                    age = blast - j
                    if age > 30:
                        continue
                    ss = sc._train_s(tf - age) + 3
                    x, y = sc._train_path(ss)
                    r = 2.0 + age * 0.24
                    out.append((x + ox + 1 + age * 0.45 + (j % 3 - 1), y - 11 - age * 1.1, r,
                                'white' if age < 14 else ('cloud' if age < 24 else 'cloud_dk'), False))
            for (px, py, r, colr, behind) in sorted(out, key=lambda q: -q[2]):
                if py > H + 6:
                    continue
                o = back if behind else front
                o.ellipse(int(px), int(py), r, r * 0.8, colr)
                o.ellipse(int(px + r * 0.3), int(py - r * 0.3), max(0.5, r * 0.5), max(0.5, r * 0.4), 'white')
            if sc.TRAIN_T0 - 1 <= tf < sc.TRAIN_T0 + 18:     # burst from the tunnel mouth
                age = tf - sc.TRAIN_T0 + 1
                for k in range(5):
                    r = 2 + age * (0.25 + k * 0.05)
                    bx = sc.PORTAL_X + ox - 4 - age * (0.5 + 0.2 * k)
                    by = 74 - age * (0.3 + 0.15 * k) + k * 2
                    c.ellipse(int(bx), int(by), r, r * 0.7, 'cloud' if k % 2 else 'white')
        sv = np.array(sent, np.uint8)
        mb = (back.a != sv).any(axis=2)
        if mb.any():                                    # hide behind the spur (depth >= 2)
            occ = np.zeros((H, W), bool)
            lh, lw = z.shape
            x0, x1 = max(0, ox), min(W, ox + lw)
            if x1 > x0:
                occ[:min(H, lh), x0:x1] = z[:min(H, lh), x0 - ox:x1 - ox] >= 2
            mb &= ~occ
            mb[:, max(0, clip_x):] = False
            c.a[mb] = back.a[mb]
        mf = (front.a != sv).any(axis=2)
        mf[:, max(0, clip_x):] = False
        c.a[mf] = front.a[mf]

    def render_garden(self, f, cx, cy):
        t = self.g.t
        cam = int(clamp(cx, 0, GARDEN_CAM))
        tfs = self.train_fs(t)
        c = Canvas()
        sc._sky(c, 0, 1.0, morn=0.35, hy=118)
        fx.glow(c, 340 - cam // 10, 70, 150, 'gold', 0.20)
        sc._wisps(c, f, -14, 1.0, drift=0.09, morn=0.3,
                  clouds=_shift_clouds(sc._CLOUDS, -cam * 0.1))
        sc._draw_layer(c, sc._far(), int(round(-cam * 0.25)), 66, 'sun')
        R = sc._rail()
        rox = _rail_x(cam)
        sc._draw_layer(c, R, rox, 0, 'sun')
        blast = None if self.blast_t0 is None else t - self.blast_t0
        self._trains(c, [(tf, blast if j == len(tfs) - 1 else None) for j, tf in enumerate(tfs)], rox)
        for k in range(6):                               # mist drifting in the valley below the spur
            x = int((k * 64 + f * 0.25 - cam * 0.7) % 380) - 40
            c.ellipse(x, 152 + (k % 3) * 6, 18, 3, 'cloud_dk' if k % 2 else 'cloud')
            c.ellipse(x + 6, 151 + (k % 3) * 6, 9, 2, 'cloud')
        CB = sc._castle_b()
        cbx = int(round(-cam * 0.6))
        sc._draw_layer(c, CB, cbx, -2, 'sun')
        for i, (px_, py) in enumerate(CB['flags']):
            sc._flag(c, px_ + cbx, py - 2, f, i, 'gold' if i % 2 else 'magic')
        for (bx, by, kind, i, bl, bw) in CB['banners']:
            sc._banner(c, bx + cbx, by - 3, f, kind, i, False, 1.0, bl, bw)
        for (rx, ry) in CB['roses']:
            fx.sparkle(c, rx + cbx + 2, ry - 4, f, 'white', period=30, size=2)
        G = sc._garden()
        gx = -cam
        sc._draw_layer(c, G, gx, 0, 'sun')
        af = t - self.spot_t.get('fountains', -999)
        boost = 0.0 if not 0 <= af < 84 else ease_in_out(seg(af, 0, 12)) * (1 - ease_in_out(seg(af, 56, 84)))
        for k, (fx0, fy, r, j1, j2) in enumerate(G['jets']):
            h0 = 6 if r > 10 else 4
            if boost > 0:                                # the jets shoot up in tall plumes
                h = h0 + int(round((26 if r > 10 else 16) * boost))
                _plume(c, f, j1[0] + gx, j1[1], h, (r * 1.3 + 4) * boost, 3 if r > 10 else 2, k * 5, boost)
                ui.burst(c, t, t - af + 12, j1[0] + gx, j1[1] - h, ('white', 'crystal_lt', 'ice', 'baby_blue'),
                         n=16, seed=30 + k, power=0.45, life=40, gravity=0.07)
            else:
                sc._jet(c, f, j1[0] + gx, j1[1], h0, r * 0.6, k * 3)
            for i in range(3):                           # ripples in the basin
                ph = (f + i * 7 + k * 5) % 21
                x0 = fx0 + gx - r // 2 + ph // 3
                c.hline(x0 - 2, x0, fy - 3, 'crystal_lt')
            fx.sparkle(c, fx0 + gx + 3, fy - 4, f + k * 7, 'white', period=26, size=1)
            if boost > 0.2:
                fx.sparkles(c, j1[0] + gx, j1[1] - 6, t, radius=int(6 + 6 * boost), n=4,
                            colors=('white', 'crystal_lt', 'butter'), seed=k, period=12)
        for k in range(14):                              # petals and motes drifting in the low sun
            x = int((hash01(k, 41) * 360 + f * (0.5 + hash01(k, 42) * 0.4) - cam) % 400) - 40
            y = int((hash01(k, 43) * 120 + math.sin(f / 9 + k) * 4 + f * 0.15) % 140) + 30
            c.px(x, y, ('cotton', 'butter', 'white')[k % 3])
        sc._draw_layer(c, sc._garden_fg(), int(round(-cam * 1.35)) - 10, 142, 'sun')
        _fill_seam(c, cam)
        self._drawn(c, f, cam, 0)
        hour = self.g.hour
        if daylight.night(hour) > 0.02:                 # the castle's windows and rose by night
            self.keep_glow(c, _windows(c, 'castle_b', ('sun',), cbx, -2, hour), 0.95)
            self.keep_glow(c, _roses(c, CB, [(rx, ry, 4) for (rx, ry) in CB['roses']], cbx, -2, 'sun'), 0.9)
        return c

    # ------------------------------------------------------------------ area 2: the docks
    def render_dock(self, f, cx, cy):
        t = self.g.t
        dy = int(clamp(-cy, 0, DOCK_RISE))
        tt = dy / DOCK_RISE
        oy_city = int(round(-30 + 30 * tt))
        oy_far = 92 - int(round(12 * (1 - tt)))
        c = Canvas()
        sc._sky(c, int(round(-10 + 10 * tt)), 1.0, morn=1.0, hy=126)
        sx, sy = 300, 86
        fx.glow(c, sx, sy, 70, 'butter', 0.22)
        fx.glow(c, sx, sy, 20, 'white', 0.35)
        sc._SUNB.draw(c, sx, sy + 5, anchor='bottom')
        sc._wisps(c, f, -6, 1.0, drift=0.08, morn=1.0)
        sc._draw_layer(c, sc._far(), 0, oy_far, 'morn')
        L = sc._city()
        sc._draw_layer(c, L, 0, oy_city, 'morn')
        sc._falls(c, f, 0, oy_city, 'morn')
        for i, (fx_, ft, fw, fb) in enumerate(sc.FALLS):
            sc._mist(c, f, fx_ + fw // 2, fb + oy_city, 12 + fw * 2, seed=i, morn=True)
        for i, (px_, py) in enumerate(L['flags']):
            sc._flag(c, px_, py + oy_city, f, i, 'gold' if i % 2 else 'magic')
        for (bx, by, kind, i, bl, bw) in L['banners']:
            sc._banner(c, bx, by + oy_city - 1, f, kind, i, False, 1.0, bl, bw)
        n = 3 + int(6 * tt)                              # the city glitters, more as we rise
        fl = L['flags']
        for k in range(n):
            j = int(hash01(k, f // 10, 5) * len(fl))
            px_, py = fl[j]
            fx.sparkle(c, px_, py + oy_city + 3, f + k * 4, ('white', 'butter')[k % 2], period=20, size=2)
        for (rx, ry, r) in L['roses']:
            fx.sparkle(c, rx + 2, ry + oy_city - 2, f, 'white', period=24, size=3)
        hz = _haze(self.g.hour)
        if hz > 0:                                       # morning haze (by the clock: _haze)
            c.blend(0, 0, W, H, '#d4ecff', 0.20 * (1 - 0.6 * tt) * hz)
        D = sc._dock()
        base, hull = _dock_split()
        sdy, ext, engines = self.ship_pose(t)
        lamps = np.zeros((H, W), bool)
        if dy < H:
            sc._draw_layer(c, base, 0, dy, 'morn')
            # the mast beacon blinks (faster with the engines running) and the pier lamps are lit by
            # night only; no halos (the film's dithered glows dimmed to murky rings at night)
            bx_, by_ = D['props']['beacon']
            lamp_on = nightwin.ramp(self.g.hour) > 0.1
            if (f // (4 if engines else 9)) % 2 == 0:
                c.px(bx_, by_ + dy, 'hot_pink')
                if lamp_on:                              # at night a small cross of light
                    for ddx, ddy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                        if ddx or ddy:
                            c.px(bx_ + ddx, by_ + dy + ddy, 'magic')
                        if 0 <= by_ + dy + ddy < H:
                            lamps[by_ + dy + ddy, bx_ + ddx] = True
            for i, (lx, ly) in enumerate(D['lamps']):
                c.px(lx, ly + dy, 'butter' if lamp_on else LAMP_OFF)
                c.px(lx, ly + dy - 1, 'brass')
                if lamp_on and 0 <= ly + dy < H:
                    lamps[ly + dy, lx] = True
            sc._flag(c, sc.MAST_X, 41 + dy, f, 1, 'magic')
            sc._flag(c, 211, sc.DECK_Y - 28 + dy, f, 2, 'gold')
        # the airship (moored, or risen off its mast)
        sc._draw_layer(c, hull, 0, dy + sdy, 'morn')
        px_, py = D['props']['prop']
        py += dy + sdy
        if engines:                                       # propeller a blur, exhaust puffs
            ph = f % 2
            c.vline(px_, py - 3 + ph, py + 3 - ph, 'steel_hi')
            c.vline(px_ + 1, py - 2, py + 2, 'steel')
            fx.steam(c, px_ + 4, py + 1, f, 'cloud', n=3, height=8, seed=4)
        else:
            ph = (f // 2) % 3
            c.vline(px_, py - (2 if ph == 0 else 1), py + (2 if ph == 2 else 1), 'steel_hi')
        c.px(px_ - 1, py, 'brass')
        for (ax, ay), (bx, by) in LINES:                  # mooring lines: cast off / thrown back up
            if ext <= 0.04:
                continue
            hx, hy = ax, ay + sdy
            ex = int(round(lerp(bx, hx, ext)))
            ey = int(round(lerp(by, hy, ext)))
            c.line(bx, by + dy, ex, ey + dy, _ROPE)
        if self.ship is not None:
            kind, t0 = self.ship
            a = t - t0
            if kind == 'up' and a < 16:                    # sparkles as the lines are slipped
                for i in range(8):
                    fx.sparkle(c, 64 + i * 16, sc.DECK_Y - 4 + dy - a // 2, f + i * 3, 'white', period=10, size=2)
                ui.burst(c, t, t0, MOORING[0], MOORING[1] + 10 + dy, ('white', 'butter', 'cotton'), n=14, seed=40,
                         power=0.5, life=30, gravity=0.04)
            if kind == 'down' and SHIP_UP <= a < SHIP_DOWN + 8:
                for i in range(3):
                    fx.sparkle(c, LINES[i][1][0], LINES[i][1][1] + dy, f + i * 4, 'white', period=10, size=2)
        # (the film's balloon at the pier is the player's: not drawn)
        self._drawn(c, f, 0, -dy)
        hour = self.g.hour
        if daylight.night(hour) > 0.02:                 # the city's, the dock's and the gondola's windows
            for name, x, y in (('city', 0, oy_city), ('dock', 0, dy), ('hull', 0, dy + sdy)):
                self.keep_glow(c, _windows(c, name, ('morn',), x, y, hour), 0.95)
            self.keep_glow(c, _roses(c, L, L['roses'], 0, oy_city, 'morn'), 0.9)
            self.keep_glow(c, lamps, 0.95)
        return c
