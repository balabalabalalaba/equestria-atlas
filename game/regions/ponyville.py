"""Ponyville as an explorable diorama (the reference region; see game/region.py for the base class).

The art is the film's scene, scenes/s03_ponyville.py, used read-only. Its `render_diorama(f)` takes
the camera, the balloon, the train and the raincloud from the film's timeline; here a copy of that
function is bound to replacement globals (`rebind`): the player drives the camera, the film's balloon
is switched off (the base class draws the player's), the Friendship Express rolls in the first time the
player heads east, a raincloud passes on a schedule and the clock tower keeps game time.

Things to set off with SPACE: the castle's star blazes, the raincloud comes over the town-hall garden,
the clock tower's bell rings, Sugarcube Corner's cupcake bursts into sprinkles, and the Friendship
Express backs out of the station east along its line (or is called back in). Three atlas stamps are hidden: above the castle,
low among the market stalls, and far east over the stream.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, seg, fx, col, mix, shade, clamp, hash01
import scenes.s03_ponyville as pv

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

CAM_MAX = 850           # the back / foreground parallax layers end at cam ~857

LANDMARKS = [           # the film's callouts (main layer = world px; this scene has no vertical camera)
    Landmark('CASTLE OF FRIENDSHIP', pv.CASTLE_X, (pv.CASTLE_X - 46, 86, 'left', 12)),
    Landmark('SCHOOL OF FRIENDSHIP', (pv.SCHOOL_X0 + pv.SCHOOL_X1) // 2,
             ((pv.SCHOOL_X0 + pv.SCHOOL_X1) // 2, 50, 'up', 10)),
    Landmark('TOWN HALL', (pv.HALL_X0 + pv.HALL_X1) // 2, (pv.HALL_X0 + 10, 136, 'down', 12)),
    Landmark('SUGARCUBE CORNER', (pv.SUGAR_X0 + pv.SUGAR_X1) // 2,
             ((pv.SUGAR_X0 + pv.SUGAR_X1) // 2 + 1, 50, 'up', 10)),
    Landmark('CAROUSEL BOUTIQUE', pv.BOUT_X + 14, (pv.BOUT_X + 14, 138, 'down', 10)),
    Landmark('TRAIN STATION', (pv.STATION_X0 + pv.STATION_X1) // 2,
             ((pv.STATION_X0 + pv.STATION_X1) // 2 - 5, 104, 'up', 22)),
]

GARDEN_X = (pv.GARDEN_X0 + pv.GARDEN_X1) // 2
STATION_X = (pv.STATION_X0 + pv.STATION_X1) // 2
STAR = (pv.CASTLE_X + 5, 9)          # the castle's crystal star (castle_fx glint)
BELFRY = (pv.CLOCK_X, 56)
CHERRY = (739, 49)                   # Sugarcube Corner's cupcake cherry
OVEN = (763, 75)                     # its oven chimney

HOTSPOTS = [
    Hotspot('star', STAR[0], STAR[1] + 4, 'spark'),
    Hotspot('rain', GARDEN_X, 112, 'rain', cooldown=120),
    Hotspot('bell', BELFRY[0], BELFRY[1], 'bell', cooldown=60),
    Hotspot('sprinkles', OVEN[0], OVEN[1], 'steam'),
    Hotspot('train', STATION_X, 112, 'whistle', cooldown=90),
]
STAMPS = [Stamp('castle', pv.CASTLE_X + 36, 24), Stamp('market', 552, 108),
          Stamp('stream', pv.STREAM_X0 + 18, 96)]

# the train keeps the film's 262..334 timing (chuffs, whistles, stop hiss) but starts further east, so
# it rolls in from off-screen even at the end of the town. The line ends at the station (the film lays
# rails only east of it), so departing, the train backs out east along its own track, the way it came.
TRAIN_RUN = 330
TRAIN_TRIGGER_X = 600   # balloon x that calls the train the first time
DEPART_ACC = 0.02       # backing out: front = stop + DEPART_ACC * frames^2
TRAIN_OUT = CAM_MAX + 320 + 8                   # front x past which no camera can see the train
TRAIN_RETURN = 600      # steps after leaving before it comes back by itself
RAIN_PERIOD = 1500      # a raincloud passes by itself every 50 s


def _train_front(f):
    t = seg(f, 262, 334)
    return pv.TRAIN_STOP + TRAIN_RUN * (1 - t) ** 2


def _depart_front(f):
    """Departing: from scene frame 400 on, the train backs out east, accelerating."""
    return pv.TRAIN_STOP + DEPART_ACC * max(0.0, f - 400) ** 2


_draw_train = rebind(pv.draw_train, train_front=_train_front)
_train_fx = rebind(pv.train_fx, train_front=_train_front)
_draw_train_dep = rebind(pv.draw_train, train_front=_depart_front)
_train_fx_dep = rebind(pv.train_fx, train_front=_depart_front)


def _clock_f(minutes):
    """Scene frame at which the film's clock tower shows 07:<minutes>."""
    return (minutes - 30) * (pv.N + pv.FLIGHT_SHIFT - 1) / 60 - pv.FLIGHT_SHIFT


def _no_balloon(*_a, **_k):
    pass


# ---- the town by the clock (game/daylight.py re-lights it to the world clock)
# The relight keeps no painted spot glowing here (auto_glow False: sunlit thatch speckles, gold trims,
# flowers and fruit are paint); the town keeps its REAL lights itself with GlowKeep, ramped by the night:
# the lit windows (_town_windows, game/nightwin.py's panes), the street-lamp heads (x, head top y), the station canopy
# lamps, the market's two lantern strings (as pv.lanterns hangs them, swaying), and the crystal glows of
# Twilight's castle (its star, gems and the light that sweeps up its trunk).
LAMPS = [(x, pv.GY - 22 - 4) for x in (176, 276, 1166)] + [(x, pv.GY - 20 - 4) for x in (108, 790)]
STATION_LAMPS = [(pv.STATION_X1 + d, pv.GY - 32) for d in (13, 31, 49)]   # (the canopy hangs at the platform, GY - 4)
STRINGS = [(pv.HALL_X1 - 4, 113, pv.CLOCK_X - 7, 110), (pv.STALLS[0][0] + 1, 116, pv.STALLS[2][0] + 20, 116)]
CASTLE_GLOW = (pv.CASTLE_X - 62, pv.CASTLE_X + 62, 108)    # Twilight's castle: its crystal glows (x0, x1, y1)
LAMP_GLASS = np.array([col(mix('butter', 'cream', .4)), col('white')], np.uint8)
STATION_GLASS = np.array([col('gold'), col('butter')], np.uint8)
STAR_BOX = (pv.CASTLE_X - 9, pv.CASTLE_X + 10, 6, 25)     # the castle's crystal star (castle(): r 8 at y 15)
STAR_COLS = np.array([col(c) for c in ('plum', 'raspberry', 'magic', 'crystal_pk', mix('crystal_pk', 'white', .55),
                                       'white')], np.uint8)
# Sugarcube Corner's round display window (pale glass, cakes on a shelf: no nightwin pane): (x, y, r)
SHOP = ((pv.SUGAR_X0 + pv.SUGAR_X1) // 2 - 15, 135, 6)
SHOP_GLASS = col(mix('baby_blue', 'cream', .4))
# distant Canterlot's tower and house windows (1 px, as pv._canterlot paints them; layer px, p 0.1)
_TW = [(-31, 26, 5), (-37, 19, 4), (-25, 17, 3), (-44, 12, 3), (-20, 11, 4)]
FAR_WINS = ([(pv.MT_X + dx + tw // 2, wy) for (dx, th, tw) in _TW
             for wy in range(pv.CITY_BASE - th + 3, pv.CITY_BASE - 3, 4)] +
            [(pv.MT_X + dx + 2, pv.CITY_BASE - 2) for dx in (-58, -46, -17, -6)])
FAR_GLASS = np.array([col(mix('periwinkle', pv.HAZE, .25)), col(mix('periwinkle', pv.HAZE, .2))], np.uint8)
# the lit windows (game/nightwin.py's panes of the back and main layers). Things its pane test takes
# for blue glass that are paint, in main-layer px (x0, y0, x1, y1): the castle's violet trunk streaks
# and balcony railing, the school's star-rose ring and window rims, the Carousel Boutique's lilac
# pillar stripes, the market's blue sign and the station bench (all glowed like lamps at night).
FALSE_PANES = ((239, 50, 240, 63), (222, 83, 236, 86), (194, 91, 195, 94), (240, 94, 241, 98),
               (240, 109, 241, 116), (339, 53, 344, 58), (333, 104, 350, 117), (320, 117, 321, 120),
               (361, 117, 362, 120), (808, 109, 845, 115), (648, 135, 654, 140), (984, 136, 986, 144))
# the School of Friendship's pale glass (arched facade windows, round windows: too big for the pane
# test); main-layer box. Its star rose (an emblem) stays unlit.
SCHOOL_BOX = (pv.SCHOOL_X0, 70, pv.SCHOOL_X1, 146)
SCHOOL_GLASS = np.array([(191, 227, 255), (174, 200, 255), (229, 244, 255)], np.uint8)
SCHOOL_SHARE = 0.85                                        # its windows lit in full night (the rest: nightwin's)
_NOT_SKY = np.array((255, 0, 255), np.uint8)               # sky hook: a key no painted pixel is near


def _in(c, y0, y1, x0, x1, cols):
    """Mask of the box's pixels (screen, clipped) whose colour is one of cols (nothing drawn over them)."""
    m = np.zeros((H, W), bool)
    y0, y1, x0, x1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
    if y1 > y0 and x1 > x0:
        sub = c.a[y0:y1, x0:x1]
        m[y0:y1, x0:x1] = (sub[:, :, None, :] == cols[None, None]).all(axis=3).any(axis=2)
    return m


@lru_cache(maxsize=2)
def _windows_of(key):
    """(ys, xs, v, shade, flip, window, painted rgb) of a layer's windows: nightwin's panes without the
    false ones (and, on the main layer, the school's glass), one window = its panes across the 1-px
    mullions together (nightwin lights every pane by itself: half-lit windows), with the night ramp at
    which it goes on and whether it is one that goes on now and then."""
    from scipy import ndimage
    get = pv._main if key == 'pv_main' else pv._back
    rgba = get()[0]
    glass = np.zeros(rgba.shape[:2], bool)
    ys, xs = nightwin.panes(key, get, exclude=FALSE_PANES if key == 'pv_main' else ())[:2]
    glass[ys, xs] = True
    if key == 'pv_main':
        x0, y0, x1, y1 = SCHOOL_BOX
        sub = rgba[y0:y1, x0:x1]
        glass[y0:y1, x0:x1] |= (sub[..., 3] >= 128) & (sub[..., :3][:, :, None, :] ==
                                                        SCHOOL_GLASS[None, None]).all(axis=3).any(axis=2)
    gap = ~glass                                       # a 1-px mullion: glass on both sides
    gap[:, 1:-1] &= glass[:, :-2] & glass[:, 2:]
    gap[1:-1, :] |= ~glass[1:-1, :] & glass[:-2, :] & glass[2:, :]
    gap[:, [0, -1]] = False
    gap[[0, -1], :] = False
    lab, _n = ndimage.label(glass | gap)
    ys, xs = np.nonzero(glass)
    win = lab[ys, xs].astype(np.int64)
    v, shade, flip = [9.0], [0], [False]
    for s_ in ndimage.find_objects(lab):
        px, py = s_[1].start, s_[0].start
        sh = SCHOOL_SHARE if key == 'pv_main' and SCHOOL_BOX[0] <= px < SCHOOL_BOX[2] else nightwin.SHARE
        v.append(hash01(px, py, 406) / sh)
        shade.append(int(hash01(px, py, 405) * 3))
        flip.append(hash01(px, py, 407) < nightwin.FLIP)
    return (ys, xs, np.array(v, np.float32)[win], np.array(shade)[win], np.array(flip)[win], win,
            rgba[ys, xs, :3].copy())


def _town_windows(c, hour, t, cam):
    """The cottages', the town hall's, the castle's, the school's and the shops' windows lit as the night
    deepens (nightwin's ramp and its now-and-then windows), where the layer still shows. Their mask."""
    k = nightwin.ramp(hour)
    if k <= 0:
        return None
    m = np.zeros((H, W), bool)
    ep = t // nightwin.FLIP_T
    for key, p in (('pv_back', pv.P_BACK), ('pv_main', 1.0)):
        ys, xs, v, shade, flip, win, src = _windows_of(key)
        sx = xs - int(round(cam * p))
        on = v < k
        on = np.where(flip, (k > 0.5) & (((win * 2654435761 + ep * 40503) >> 9) % 7 < 3), on)
        on &= (sx >= 0) & (sx < W)
        ys2, sx2 = ys[on], sx[on]
        seen = (c.a[ys2, sx2] == src[on]).all(axis=1)          # nothing drawn in front of it
        ys2, sx2 = ys2[seen], sx2[seen]
        c.a[ys2, sx2] = nightwin.WARM[shade[on][seen]]
        m[ys2, sx2] = True
    return m


def _night_windows(c, hour, cam):
    """Windows nightwin cannot find, lit with its ramp: Sugarcube Corner's display window (its pale glass
    turns warm, the cakes are lit) and distant Canterlot's tiny windows. Returns their mask."""
    k = nightwin.ramp(hour)
    if k <= 0:
        return None
    m = np.zeros((H, W), bool)
    x, y, r = SHOP
    sx = x - cam
    y0, y1, x0, x1 = y - r, y + r + 1, max(0, sx - r), min(W, sx + r + 1)
    if k > 0.15 and x1 > x0:                            # (in view: off the edge the box is empty)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - sx) ** 2 + (yy - y) ** 2 <= r * r
        sub = c.a[y0:y1, x0:x1]
        glass = disc & (sub == SHOP_GLASS).all(axis=2)
        sub[glass] = np.where((yy[glass] + xx[glass]) % 5 == 0, 1, 0)[:, None] * (nightwin.WARM[2] -
                                                                                    nightwin.WARM[1]) + nightwin.WARM[1]
        m[y0:y1, x0:x1] = disc
    oc = int(round(cam * pv.P_CAN))
    for i, (wx, wy) in enumerate(FAR_WINS):
        if hash01(wx, wy, 611) * 0.85 < k:              # most go on as the night deepens
            fxp = wx - oc
            if 0 <= fxp < W and (c.a[wy, fxp] == FAR_GLASS).all(axis=1).any():
                c.a[wy, fxp] = nightwin.WARM[i % 3]
                m[wy, fxp] = True
    return m


def _lights(c, f, cam):
    """The town's own lights in the frame (before the relight): lamp heads, canopy lamps, lanterns and
    the castle's crystal glows, exactly their pixels (no halo: only the light itself is lit)."""
    m = np.zeros((H, W), bool)
    sx = STAR_BOX[0] - cam
    m |= _in(c, STAR_BOX[2], STAR_BOX[3], sx, sx + STAR_BOX[1] - STAR_BOX[0], STAR_COLS)
    for (x, y) in LAMPS:
        sx = x - cam
        if -4 < sx < W + 4:
            m |= _in(c, y, y + 3, sx - 1, sx + 3, LAMP_GLASS)
    for (x, y) in STATION_LAMPS:
        sx = x - cam
        if -4 < sx < W + 4:
            m |= _in(c, y, y + 3, sx - 1, sx + 2, STATION_GLASS)
    for si, (ax, ay, bx2, by2) in enumerate(STRINGS):     # (pv.lanterns: 5 per string, swaying)
        for k in range(1, 6):
            u = k / 6
            x = int(ax + (bx2 - ax) * u - cam)
            if not -4 < x < W + 4:
                continue
            y = int(ay + (by2 - ay) * u + math.sin(u * math.pi) * 5)
            sw = int(round(math.sin(f * 0.15 + k + si) * 0.8))
            cc = ('coral', 'gold', 'bubblegum', 'turquoise', 'lavender')[(k + si) % 5]
            cols = np.array([col(cc), col(mix(cc, 'white', .5)), col(shade(cc, .6))], np.uint8)
            m |= _in(c, y + 2, y + 6, x - 1 + sw, x + 2 + sw, cols)
    x0, x1, y1 = max(0, CASTLE_GLOW[0] - cam), min(W, CASTLE_GLOW[1] - cam), CASTLE_GLOW[2]
    if x1 > x0:                                         # the castle's crystal: its own bright spots...
        m[:y1, x0:x1] |= daylight._lights(np.ascontiguousarray(c.a[:y1, x0:x1]))
        # (the light sweeping up the trunk, castle_fx, is not kept: restored to its painted near-white it
        # crossed the night trunk as a white neon slash; relit with the trunk it stays a soft shimmer)
    return m


class _FxNoSun:
    """engine.fx for the film's render, without its screen-anchored warm morning light (the town draws
    that light by the clock: _sun_light)."""
    def __getattr__(self, name):
        return getattr(fx, name)

    @staticmethod
    def light_rect(*_a, **_k):
        pass


_FX = _FxNoSun()


def _sun_light(c, hour):
    """The film's warm light from the low morning sun (the left 150 px, fading right), by the clock: as
    painted in the morning, from the right in the afternoon (the sun's arc), none at dusk or night."""
    km = seg(hour, 6.2, 7.2) * (1 - seg(hour, 10.0, 12.0))
    ka = seg(hour, 13.0, 15.0) * (1 - seg(hour, 17.4, 18.6))
    if km > 0.02:
        fx.light_rect(c, 0, 0, 150, H, 'gold', strength=0.07 * km, fade='right')
    if ka > 0.02:
        fx.light_rect(c, W - 150, 0, 150, H, 'gold', strength=0.07 * ka, fade='left')


@lru_cache(maxsize=1)
def _opaque():
    """(opaque mask, parallax) of every scenery layer (the film blits them at (-cam * p, 0))."""
    return [(arr[..., 3] >= 128, p) for arr, p in (
        (pv._canterlot(), pv.P_CAN), (pv._far(), pv.P_FAR), (pv._mid()[0], pv.P_MID),
        (pv._back()[0], pv.P_BACK), (pv._main()[0], 1.0), (pv._fg()[0], pv.P_FG))]


@lru_cache(maxsize=64)
def _sky_key(cam):
    """The pixels that are not sky at camera cam: every scenery layer's opaque pixels. The cottages'
    pale glass, the white pickets, the town hall's pillars, pale cobbles and the hazy far hills are all
    within the sky test's tolerance of the pale morning sky behind them (paint_sky took them for sky
    and painted the clock's sunset or night sky into them)."""
    m = np.zeros((H, W), bool)
    for op, p in _opaque():
        x = -int(round(cam * p))
        lh, lw = op.shape
        x0, x1, h = max(0, x), min(W, x + lw), min(H, lh)
        if x1 > x0:
            m[:h, x0:x1] |= op[:h, x0 - x:x1 - x]
    m[pv.GY + 2:] = True                # (the sky glimpsed through seams in the ground layers is not sky)
    return m


def _open_sky(a, cam):
    """The frame's open-sky pixels at camera cam: still the film's sky (daylight.sky_pixels' colour
    test alone: this sky has no enclosed holes to find) and no scenery layer there. (The morning mist's
    specks over the sky stay: white mist suits a morning sky.)"""
    y1 = pv.GY + 2
    m = np.zeros((H, W), bool)
    m[:y1] = np.abs(a[:y1].astype(np.int16) - pv._sky().a[:y1]).sum(axis=2) < daylight.SKY_TOL
    m &= ~_sky_key(int(clamp(cam, 0, CAM_MAX)))
    return m


class Ponyville(GlowKeep, Region):
    SEG = 'ponyville'
    CARD = ('COUNTRY TOWN', ['THATCH ROOFS - MARKET', 'BELOW CANTERLOT'])

    def build_areas(self):
        return [Area(self.render_town, cam_x=(0, CAM_MAX), start=(104, 78), landmarks=LANDMARKS,
                     hotspots=HOTSPOTS, stamps=STAMPS, sky=self.sky_town)]

    def sky_town(self, f, cx, cy):
        """The sky alone (game/daylight.py), the scenery keyed out. The drifting mist's specks over open
        sky are sky (the clock's sky replaces them: painted white specks on a dusk sky were wrong);
        over the hills they are scenery."""
        cam = int(clamp(cx, 0, CAM_MAX))
        c = pv._sky().copy()
        pv.blit_wrap(c, pv._mist(), int(round(cam * 0.3 + f * 0.2)), 100)
        pv.blit_wrap(c, pv._mist(), int(round(cam * 0.5 + f * 0.3)) + 220, 122)
        c.a[_sky_key(cam)] = _NOT_SKY
        return c

    def auto_glow(self):
        return False                # the town keeps its real lights itself (_lights, nightwin)

    def after_light(self, c, t):
        own, self._sky_own = self._sky_own, None
        if own is not None and own.any():               # the clock's sky where the base kept the painted
            y0, y1 = self.area.cam_y                    # one (its sun low on the left horizon, its glare)
            daylight.paint_sky(c.a, own, self.g.hour, 1.0, self._cam, 0, y0 - 10, y1 - y0 + 130, t)
        super().after_light(c, t)                       # (GlowKeep: the lamps and windows shine)
        self._draw_light_fx(c, t, self._cam)            # the star and the bell: light, never dimmed

    def init_state(self):
        self.train = None           # None | ('arrive' | 'depart', step it starts)
        self.train_back = None      # step at which a departed train comes back by itself
        self.rain_t0 = -10 ** 6     # step at which the current raincloud pass began (scene frame 140)
        self._cam = 0               # camera of the frame last drawn (its after-light effects)
        self._sky_own = None        # its open sky, where the clock's sky must still go (after_light)

    # ---- clocks (scene frames for the film's helpers)
    def train_f(self, t):
        """('arrive' | 'depart', scene frame) or None."""
        if self.train is None or t < self.train[1]:
            return None
        kind, t0 = self.train
        return kind, (262 if kind == 'arrive' else 400) + (t - t0)

    def rain_f(self, t):
        return 140 + (t - self.rain_t0)

    # ---- events
    def update(self, t):
        g = self.g
        if self.train is None and self.train_back is None and self.bx > TRAIN_TRIGGER_X:
            self.train = ('arrive', t + 40)
        if self.train is None and self.train_back is not None and t >= self.train_back:
            self.train, self.train_back = ('arrive', t), None
        tr = self.train_f(t)
        if tr is not None:
            kind, tf = tr
            if kind == 'arrive':
                gain = self.near_gain(STATION_X)
                if tf in (262, 298):
                    g.sfx('chug', 0.45 * gain, (tf // 36) % 2)
                elif tf == 288:
                    g.sfx('whistle', 0.8 * max(0.35, gain))
                elif tf == 334:
                    g.sfx('steam', 0.55 * gain)
            else:
                df = tf - 400
                if df % 36 == 12:
                    g.sfx('chug', 0.45 * self.near_gain(_depart_front(tf)), (df // 36) % 2)
                if _depart_front(tf) > TRAIN_OUT:
                    self.train, self.train_back = None, t + TRAIN_RETURN
        if t - self.rain_t0 >= RAIN_PERIOD:
            self.rain_t0 = t
        if self.rain_f(t) == 154:
            g.sfx('rain', 0.4 * self.near_gain(GARDEN_X), 1)
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)

    def on_interact(self, key, t):
        g = self.g
        if key == 'star':
            g.sfx('crystal', 0.55)
        elif key == 'rain':
            if not 140 <= self.rain_f(t) <= 216:      # call the raincloud unless it is already here
                self.rain_t0 = t
            g.sfx('wind', 0.35, 1)
        elif key == 'bell':
            g.sfx('bell', 0.6)
        elif key == 'sprinkles':
            g.sfx('pop', 0.45, 2)
            g.sfx('steam', 0.3)
        elif key == 'train':
            tr = self.train_f(t)
            if tr is None:                               # out of town (or not come yet): call it in
                self.train, self.train_back = ('arrive', t + 20), None
                g.sfx('whistle', 0.35)
            elif tr[0] == 'arrive' and tr[1] > 340:      # parked: all aboard, off it goes (at once)
                self.train = ('depart', t)
                g.sfx('whistle', 0.8)
            else:
                g.sfx('tick', 0.3)

    # ---- drawing
    def render_town(self, f, cx, cy):
        t = self.g.t
        tr = self.train_f(t)
        rf = self.rain_f(t)
        minutes = 40 + (t / 30) * 0.5          # the clock tower: one minute every two seconds

        def train(c, _f, cam):
            if tr is None:
                return None
            return (_draw_train if tr[0] == 'arrive' else _draw_train_dep)(c, tr[1], cam)

        def train_fx(c, _f, cam, X):
            if tr is not None:
                (_train_fx if tr[0] == 'arrive' else _train_fx_dep)(c, tr[1], cam, X)

        render = rebind(pv.render_diorama,
                        cam_x=lambda _f: cx,
                        draw_balloon=_no_balloon,
                        draw_train=train, train_fx=train_fx,
                        raincloud=lambda c, _f, cam: pv.raincloud(c, rf, cam),
                        clock_hands=lambda c, _f, cam: pv.clock_hands(c, _clock_f(minutes), cam),
                        fx=_FX)                         # (its morning light is drawn by the clock below)
        c = render(f)
        self._cam = cx
        self._draw_events(c, t, cx)
        hour = self.g.hour
        win = lit = None
        if daylight.night(hour) > 0.02:
            # the windows first: they are found by the layers' own colours, so nothing may be drawn
            # over them yet (the film's morning light over the left of the screen hid them before)
            win = _town_windows(c, hour, t, cx)
            more = _night_windows(c, hour, cx)
            if more is not None:
                win = more if win is None else win | more
            lit = _lights(c, f, cx)
        _sun_light(c, hour)
        self.keep_glow(c, win, 0.95)
        self.keep_glow(c, lit, 0.9)
        # the clock's sky always: near the painted hour (07:30-10:24) the base keeps the film's sky,
        # whose sun sits low on the left horizon, off the clock's arc (and jumps when the base switches)
        self._sky_own = None
        if daylight.sky_weight(self.HOUR, hour) < 0.99:
            self._sky_own = _open_sky(c.a, cx)
        return c

    def _draw_light_fx(self, c, t, cam):
        """The SPACE effects that are light (drawn after the clock's light, so they shine at night too)."""
        a = t - self.spot_t.get('star', -999)
        if 0 <= a < 60:                                       # the star blazes and showers sparkles
            sx = STAR[0] - cam
            k = 1 - a / 60
            fx.glow(c, sx, STAR[1] + 2, 26 + a // 3, 'crystal_pk', 0.75 * k, rings=5)
            fx.glow(c, sx, STAR[1] + 2, 10, 'white', 0.8 * k, rings=3)
            for j in range(16):                               # sparks falling down the crystal tree
                ang = j / 16 * 6.283 + 0.4
                rr = 4 + a * (0.9 + 0.5 * ((j * 7) % 5) / 5)
                px, py = sx + int(math.cos(ang) * rr), STAR[1] + 2 + int(math.sin(ang) * rr * 0.7 + a * a * 0.012)
                if (a + j) % 4:
                    c.px(px, py, ('white', 'crystal_lt', 'gold', 'crystal_pk')[j % 4])
            if a < 14:
                c.hline(sx - 14 + a // 2, sx + 14 - a // 2, STAR[1] + 2, 'white')
                c.vline(sx, STAR[1] - 12 + a // 2, STAR[1] + 16 - a // 2, 'white')
        a = t - self.spot_t.get('bell', -999)
        if 0 <= a < 48:                                       # the bell's sound rings out
            bx, by = BELFRY[0] - cam, BELFRY[1]
            for k in range(3):
                r = (a - k * 9) * 1.2 + 4
                if 4 <= r < 40:
                    c.circle(bx, by, int(r), 'butter' if k == 0 else 'white', fill=False)

    def _draw_events(self, c, t, cam):
        """Visible results of the other SPACE interactions (world-anchored; reads state only)."""
        a = t - self.spot_t.get('sprinkles', -999)
        if 0 <= a < 70:                                       # the oven puffs, sprinkles shower from the cupcake
            t0 = self.spot_t['sprinkles']
            ui.burst(c, t, t0, CHERRY[0] - cam, CHERRY[1], n=36, seed=5, power=0.9, life=60,
                     colors=('hot_pink', 'butter', 'baby_blue', 'mint', 'white', 'lavender'))
            for k in range(3):
                r = 3 + (a - k * 5) * 0.35
                if 0 <= a - k * 5 < 40:
                    c.circle(OVEN[0] - cam + k * 2, OVEN[1] - 4 - (a - k * 5), int(r), 'white')
