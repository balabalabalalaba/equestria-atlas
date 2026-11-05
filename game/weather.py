"""Weather: snow, rain and wind on the world map and in some places, and how the balloon takes it.

World map (world px zones; everything is a pure function of the step):
  snow   the Frozen North (y < 300) and the icy south (y > 1440): flakes drift down over the land.
  rain   four dark rain clouds drift in slow loops (over the Everfree / Ponyville country, the
         Manehattan-Baltimare coast, the west coast and the eastern forests), casting shadows; rain
         streaks fall under them.
  wind   windy zones (the Celestial Sea, the Bone Dry Desert with blowing sand, the Luna Ocean, the
         Cloudsdale highlands): streaks (or sand) race along, gusting slowly.
Places (`PLACES`): sand gusts (Appleloosa, Klugetown), breezes (Cloudsdale, Mount Aris, Baltimare),
drizzle (Griffonstone), and the snow / rain the scenes already draw (Yakyakistan, the Crystal Empire,
Vanhoover): there only the balloon's reaction is added.

The balloon reacts:
  wind  carries it (a drift added to its motion): fly with it for speed, against it to struggle; in a
        place, gusts come and go and push it sideways.
  rain  soaks it: the envelope darkens and drips (`wet` 0..1); flying out of the rain while soaked
        leaves a little rainbow behind.
  snow  settles on top of the envelope (`snow` 0..1); boosting (SHIFT) shakes it off in a puff.
The first meeting with each kind shows a toast (kept in the save). game/compose.py draws the wet and
snowy balloon (Frame.balloon wet= / snow=).
"""
import math

import numpy as np

from engine import W, H, fx, hash01, col, mix
from engine import loc_worldmap as LW

from . import ui, perks

SNOW_RATE, SNOW_MELT = 0.0035, 0.002
WET_RATE, DRY_RATE = 0.02, 0.0045
GUST_PERIOD, GUST_LEN = 330, 110

# rain clouds on the map: (centre x, centre y, loop rx, ry, phase)
RAIN_CLOUDS = [(600, 600, 90, 40, 0.0), (1050, 520, 70, 60, 1.7), (230, 520, 60, 90, 3.1), (1330, 650, 70, 70, 4.4)]
CLOUD_RX, CLOUD_RY = 34, 16                 # the ground area a rain cloud covers
CLOUD_UP = 58                               # a cloud floats this far above its ground point
# windy zones: (x0, y0, x1, y1, dir x, dir y, strength px/step, kind)
WIND_ZONES = [(880, 300, 1250, 700, 0.8, -1.0, 0.75, 'breeze'),
              (560, 990, 1000, 1170, -1.0, 0.1, 0.85, 'sand'),
              (0, 380, 200, 760, -0.7, 1.0, 0.7, 'breeze'),
              (400, 310, 560, 420, 1.0, 0.0, 0.55, 'breeze')]

# places: place id -> (kind, areas or None for all, draw it (False: the scene already does), strength)
PLACES = {
    'appleloosa': ('sand', None, True, 1.0),
    'klugetown': ('sand', (0, 2), True, 0.9),
    'cloudsdale': ('breeze', (0, 2), True, 0.8),
    'mount_aris': ('breeze', (0, 1), True, 0.7),
    'baltimare': ('breeze', None, True, 0.7),
    'griffonstone': ('rain', None, True, 0.6),
    'vanhoover': ('rain', None, False, 1.0),
    'yakyakistan': ('snow', None, False, 1.0),
    'crystal_empire': ('snow', None, False, 0.8),
}
MEET = {'snow': ('SNOW!', 'IT SETTLES ON YOUR BALLOON - SHIFT SHAKES IT OFF'),
        'rain': ('RAIN!', 'YOUR BALLOON GETS SOAKED - FLY OUT FOR A RAINBOW'),
        'wind': ('WIND!', 'IT CARRIES YOUR BALLOON - RIDE IT FOR SPEED')}


def _unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def cloud_pos(i, t):
    cx, cy, rx, ry, ph = RAIN_CLOUDS[i]
    return (cx + rx * math.sin(t * 0.0016 + ph), cy + ry * math.sin(t * 0.0021 + ph * 1.3))


def map_snow(x, y):
    """Snowfall 0..1 at a map ground point."""
    return max(min(1.0, (300 - y) / 70), min(1.0, (y - 1440) / 60), 0.0)


def map_rain(x, y, t):
    """Rain 0..1 at a map ground point (under a rain cloud)."""
    best = 0.0
    for i in range(len(RAIN_CLOUDS)):
        cx, cy = cloud_pos(i, t)
        d = ((x - cx) / CLOUD_RX) ** 2 + ((y - cy) / CLOUD_RY) ** 2
        if d < 1:
            best = max(best, min(1.0, (1 - d) * 2.5))
    return best


CALM_IN, CALM_OUT = 30, 70     # a harbour calm round a found place's icon: no wind within its landing reach (world.NEAR), full at 70 px


def map_wind(x, y, t, calm=()):
    """(wx, wy, kind) the wind pushes a balloon by at a map ground point (px per step). calm: the icon
    centres of the places found so far; the push fades out near one, so a balloon hovering over a
    place (its land plate up) is never blown off it."""
    wx = wy = 0.0
    kind = None
    for j, (x0, y0, x1, y1, dx, dy, s, k) in enumerate(WIND_ZONES):
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        edge = min(x - x0, x1 - x, y - y0, y1 - y)
        e = min(1.0, edge / 40)
        g = s * (0.55 + 0.45 * math.sin(t * 0.012 + j * 2.1)) * e
        ux, uy = _unit(dx, dy)
        wx += ux * g
        wy += uy * g
        kind = k
    if kind is not None and calm:
        d = min(math.hypot(x - px, y - py) for px, py in calm)
        f = min(1.0, max(0.0, (d - CALM_IN) / (CALM_OUT - CALM_IN)))
        wx, wy = wx * f, wy * f
    return wx, wy, kind


_ICON_XY = {}


def _found_icons(g):
    """The found places' icon centres as ground points: the landing reach is measured from the balloon's
    body, FOCUS_DY above its ground point, so the calm sits that far below the icon (cached per place)."""
    if not _ICON_XY:
        from .world import icon_centre, FOCUS_DY
        from engine import timeline as TL
        for lid in TL.TOUR:
            x, y = icon_centre(lid)
            _ICON_XY[lid] = (x, y + FOCUS_DY)
    return [_ICON_XY[lid] for lid in g.found if lid in _ICON_XY]


class Weather:
    def __init__(self, game):
        self.g = game
        self.wet = 0.0
        self.snow = 0.0
        self.shake_t = None             # step snow was shaken off (a puff of flakes)
        self.rainbow = None             # (step, world x, world y, mode id) of an after-rain rainbow
        self.met = set()                # kinds of weather met (toasts once)
        self.drift = (0.0, 0.0)         # the wind's push this step (map / place)
        self._in_rain = False

    def save(self):
        return dict(met=sorted(self.met))

    def load(self, d):
        self.met = {k for k in (d or {}).get('met', ()) if k in MEET}

    def _meet(self, kind):
        """The first rain / snow / wind: a toast says what it does. It is called every step under that
        weather, so while other toasts are up or waiting it simply waits for the lane to be free (and its
        toast is dropped if it still cannot show within WEATHER_TTL steps), so it never shows late."""
        if kind in self.met or self.g.toasts.busy:
            return
        self.met.add(kind)
        title, sub = MEET[kind]
        self.g.toasts.push(title, sub, _ICON[kind], ttl=WEATHER_TTL)   # (never late: it is news of now)

    def _settle(self, rain, snow, boost, x, y, where):
        """Wet / snow levels for one step under `rain` and `snow` (0..1)."""
        g = self.g
        if rain > 0.05:
            self.wet = min(1.0, self.wet + WET_RATE * rain)
            self._meet('rain')
        else:
            if self._in_rain and self.wet > 0.45:
                self.rainbow = (g.t, x, y, where)
                g.sfx('shimmer', 0.35)
            self.wet = max(0.0, self.wet - DRY_RATE)
        self._in_rain = rain > 0.05
        if snow > 0.05:
            self.snow = min(1.0, self.snow + SNOW_RATE * snow)
            self._meet('snow')
        else:
            self.snow = max(0.0, self.snow - SNOW_MELT)
        if boost and self.snow > 0.15:
            self.shake_t = g.t
            self.snow = 0.0
            g.sfx('puff', 0.35)

    # -------------------------------------------------------------- world map
    def map_step(self, m):
        """Called after the map balloon has moved: the wind for the next step (MapMode._fly adds it before
        the camera follows, so the balloon never jitters against the ground), rain, snow."""
        g = self.g
        if m.landing is not None:
            self.drift = (0.0, 0.0)
            return
        wx, wy, kind = map_wind(m.bx, m.by, g.t, _found_icons(g))
        if kind is not None and math.hypot(wx, wy) > 0.15:
            self._meet('wind')
        self.drift = (wx, wy) if m.takeoff is None and not perks.calm(g) else (0.0, 0.0)   # (next step's flight)
        self._settle(map_rain(m.bx, m.by, g.t), map_snow(m.bx, m.by), m.boost, m.bx, m.by, 'map')

    def map_draw_ground(self, c, t, x0, y0):
        """Rain-cloud shadows on the land (before the fog)."""
        a = c.a
        for i in range(len(RAIN_CLOUDS)):
            cx, cy = cloud_pos(i, t)
            sx, sy = int(cx) - x0, int(cy) - y0
            if not (-50 < sx < W + 50 and -30 < sy < H + 30):
                continue
            y_lo, y_hi = max(0, sy - CLOUD_RY), min(H, sy + CLOUD_RY + 1)
            x_lo, x_hi = max(0, sx - CLOUD_RX), min(W, sx + CLOUD_RX + 1)
            if y_hi <= y_lo or x_hi <= x_lo:
                continue
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            d = ((xx - sx) / CLOUD_RX) ** 2 + ((yy - sy) / CLOUD_RY) ** 2
            m = (d < 1) & (((xx + yy) % 2 == 0) | (d < 0.55))
            sub = a[y_lo:y_hi, x_lo:x_hi]
            sub[m] = (sub[m].astype(np.uint16) * 4 // 5).astype(np.uint8)

    def map_draw_sky(self, c, t, x0, y0, hour=12.0):
        """Snow, wind streaks, sand, rain clouds and their rain, the after-rain rainbow (over the fog). The
        flakes, streaks and rain are lit by the clock like the map under them (no white dashes at night);
        the clouds darken by themselves."""
        from . import daylight
        before = None if daylight.is_day(hour) else c.a.copy()
        cxv, cyv = x0 + W // 2, y0 + H // 2
        # snow: world-anchored flakes over the snowy bands
        if map_snow(cxv, y0) > 0 or map_snow(cxv, y0 + H) > 0:
            for i in range(90):                     # each flake lives in a 330 x 190 world tile
                fxw = hash01(i, 11) * 330 + t * 0.25 + 6 * math.sin(t * 0.05 + i)
                fyw = hash01(i, 12) * 190 + t * (0.45 + 0.3 * hash01(i, 13))
                px = int((fxw - x0) % 330) - 5
                py = int((fyw - y0) % 190) - 5
                s = map_snow(x0 + px, y0 + py)
                if s > hash01(i, 14) and 0 <= px < W - 1 and 0 <= py < H - 1:
                    c.px(px, py, 'white')
                    c.px(px + 1, py + 1, '#8f9ac0')            # a soft edge: seen on snowy ground too
                    if i % 5 == 0:
                        c.px(px + 1, py, 'white')
        # wind: streaks (or sand) on a world-anchored tile, drawn where a zone blows
        zones = [z for z in WIND_ZONES if not (z[2] < x0 or z[0] > x0 + W or z[3] < y0 or z[1] > y0 + H)]
        for j, (zx0, zy0, zx1, zy1, dx, dy, s, kind) in enumerate(zones):
            ux, uy = _unit(dx, dy)
            jj = WIND_ZONES.index((zx0, zy0, zx1, zy1, dx, dy, s, kind))
            g = 0.55 + 0.45 * math.sin(t * 0.012 + jj * 2.1)
            n = 34 if kind == 'breeze' else 110
            sp = s * 3.0
            for i in range(n):
                if hash01(jj, i, 4) > g + 0.15:
                    continue
                fxw = hash01(jj, i, 2) * 330 + ux * t * sp * (0.8 + 0.4 * hash01(jj, i, 5))
                fyw = hash01(jj, i, 3) * 190 + uy * t * sp * (0.8 + 0.4 * hash01(jj, i, 5))
                px = int((fxw - x0) % 330) - 5
                py = int((fyw - y0) % 190) - 5
                wx_, wy_ = x0 + px, y0 + py
                if not (0 <= px < W and 0 <= py < H and zx0 + 8 < wx_ < zx1 - 8 and zy0 + 8 < wy_ < zy1 - 8):
                    continue
                if kind == 'sand':
                    if i % 3 == 0:
                        _swoosh(c, px, py, ux, uy, 12 + i % 5, i + t * 0.15, ('paper', 'cream'))
                    else:
                        c.rect(px, py, 2, 1, 'brew' if i % 3 == 1 else 'clay')
                        c.px(px - int(round(ux * 2)), py - int(round(uy * 2)), 'tan')
                else:
                    _swoosh(c, px, py, ux, uy, 9 + i % 6, i + t * 0.15)
        # rain clouds and their rain (all the rain first: it is lit with the streaks, then the clouds)
        clouds = []
        for i in range(len(RAIN_CLOUDS)):
            cx, cy = cloud_pos(i, t)
            sx, sy = int(cx) - x0, int(cy) - y0
            if not (-60 < sx < W + 60 and -40 < sy < H + 90):
                continue
            top = sy - CLOUD_UP
            for k in range(26):                                   # rain streaks under the cloud
                rx = sx - CLOUD_RX + 4 + int(hash01(i, k, 5) * (2 * CLOUD_RX - 8))
                ry = top + 8 + int((hash01(i, k, 6) * 60 + t * 5) % (CLOUD_UP + CLOUD_RY - 6))
                c.line(rx, ry, rx - 1, ry + 3, 'light_sky')
            clouds.append((i, sx, top))
        if before is not None:
            daylight.dim_new(c.a, before, hour, 0.85)
        for i, sx, top in clouds:
            _rain_cloud(c, sx, top, t + i * 17, LW.grade_params(hour)[2])
        rb = self.rainbow
        if rb is not None and rb[3] == 'map' and 0 <= t - rb[0] < 100:
            _rainbow(c, int(rb[1]) - x0, int(rb[2]) - y0 - 30, t - rb[0])

    # -------------------------------------------------------------- places
    def place(self, r):
        w = PLACES.get(r.ID)
        if w is None:
            return None
        kind, areas, draw, s = w
        if areas is not None and r.ai not in areas:
            return None
        return kind, draw, s

    def gust(self, r, t):
        """(-1..1 signed gust strength) of a windy place at step t."""
        n = t // GUST_PERIOD
        a = t % GUST_PERIOD
        if a >= GUST_LEN:
            return 0.0
        d = 1 if hash01(sum(map(ord, r.ID)), n) > 0.45 else -1
        return d * math.sin(math.pi * a / GUST_LEN)

    def region_step(self, r):
        """After the balloon has moved in a place: gusts push it; rain / snow settle on it."""
        g = self.g
        w = self.place(r)
        rain = snow = 0.0
        self.drift = (0.0, 0.0)
        if w is not None:
            kind, _draw, s = w
            if kind in ('breeze', 'sand'):
                gs = self.gust(r, g.t) * s
                if abs(gs) > 0.1:
                    self._meet('wind')
                self.drift = (gs * 0.9, 0.0) if not perks.calm(g) else (0.0, 0.0)   # (next step's flight)
                if g.t % GUST_PERIOD == 2:
                    g.sfx('wind', 0.3 * s)
            elif kind == 'rain':
                rain = s
            elif kind == 'snow':
                snow = s
        self._settle(rain, snow, r.boost, r.bx, r.by, r.ID)

    def region_draw(self, r, c, t, ix, iy):
        w = self.place(r)
        if w is not None:
            kind, draw, s = w
            if kind in ('breeze', 'sand'):
                gs = self.gust(r, t) * s
                if abs(gs) > 0.05:
                    _gust(c, t, gs, kind, ix)
            elif kind == 'rain' and draw:
                fx.rain(c, t, n=int(70 * s), color=mix('light_sky', 'cloud', 0.4), seed=9, slant=1, speed=7,
                        length=3)
        rb = self.rainbow
        if rb is not None and rb[3] == r.ID and 0 <= t - rb[0] < 100:
            _rainbow(c, int(rb[1]) - ix, int(rb[2]) - iy - 34, t - rb[0])

    # -------------------------------------------------------------- the balloon
    def balloon_fx(self, c, t, sx, sy, up):
        """Drips under a wet balloon and the puff when snow is shaken off (screen px of the basket)."""
        if self.wet > 0.2:
            for i in range(3):
                p = (t * 0.35 + i * 7) % 12
                if hash01(i, int((t * 0.35 + i * 7) // 12)) < self.wet:
                    c.px(int(sx) - 3 + i * 3, int(sy) + int(p), 'light_sky')
        if self.shake_t is not None and 0 <= t - self.shake_t < 24:
            ui.burst(c, t, self.shake_t, int(sx), int(sy) - up, ('white', 'cloud', 'white'), n=26,
                     seed=self.shake_t % 97, power=0.8, life=24, gravity=0.04)


# ------------------------------------------------------------------ drawing helpers
def _rain_cloud(c, x, y, t, night=0.0):
    """A dark, puffy rain cloud (~64 x 22) centred on x, top at y (darker at night)."""
    bob = int(round(math.sin(t * 0.05) * 1))
    y += bob
    dk, md, hi = (mix(cc, '#232842', 0.55 * night) for cc in ('#4f5a7a', '#6f7a9a', '#8f9ab8'))
    for dx, dy, r in ((-22, 10, 9), (-8, 5, 12), (9, 6, 11), (23, 11, 8), (0, 12, 12)):
        c.circle(x + dx, y + dy + 1, r, dk)
    for dx, dy, r in ((-22, 10, 8), (-8, 5, 11), (9, 6, 10), (23, 11, 7), (0, 12, 11)):
        c.circle(x + dx, y + dy, r, md)
    for dx, dy, r in ((-10, 2, 6), (7, 3, 5)):
        c.circle(x + dx, y + dy, r, hi)
    c.hline(x - 28, x + 28, y + 19, dk)


def _rainbow(c, x, y, a):
    """A little rainbow arc that fades in and out after flying out of the rain."""
    k = min(1.0, a / 12) * (1 - max(0.0, (a - 70) / 30))
    if k <= 0:
        return
    for j, name in enumerate(('red', 'orange', 'butter', 'leaf', 'sky', 'violet')):
        r = 18 - j
        cc = np.array(col(name), np.float32)
        for q in range(0, 181, 3):
            ang = math.radians(q)
            px, py = int(round(x + math.cos(ang) * r)), int(round(y - math.sin(ang) * r * 0.7))
            if 0 <= px < W and 0 <= py < H:
                c.a[py, px] = (c.a[py, px] * (1 - 0.75 * k) + cc * 0.75 * k).astype(np.uint8)


def _gust(c, t, gs, kind, ix):
    """A gust across a place: streaks (or blowing sand) racing in its direction."""
    d = 1 if gs > 0 else -1
    a = abs(gs)
    n = int(30 * a) if kind == 'breeze' else int(110 * a)
    for i in range(n):
        y = int(hash01(i, 21) * (H - 20)) + 6
        x = int((hash01(i, 22) * (W + 80) + d * t * (5 + 3 * hash01(i, 23))) % (W + 80)) - 40
        if kind == 'sand':
            if i % 4 == 0:
                _swoosh(c, x, y, d, 0.0, 12 + i % 6, i + t * 0.2, ('cream', 'sand'))
            else:
                c.rect(x, y, 2, 1, 'tan' if i % 3 else 'clay')
                c.px(x - 2 * d, y, 'sand')
        elif i % 2 == 0:
            _swoosh(c, x, y, d, 0.0, 10 + i % 6, i + t * 0.2)


def _swoosh(c, x, y, ux, uy, ln, ph, cols=('white', 'cloud')):
    """A wind streak: a straight, fading dash along (ux, uy) (no curl or wave: nothing that could
    read as a bird)."""
    for q in range(ln):
        qx = int(round(x - ux * q))
        qy = int(round(y - uy * q))
        if (q + int(ph)) % 5 == 4 and q > ln // 2:           # breaks up towards its tail
            continue
        if 0 <= qx < W and 0 <= qy < H:
            c.px(qx, qy, cols[0] if q < ln // 2 else cols[1])

def _icon_snow(c, x, y):
    for px, py in ((4, 0), (4, 8), (0, 4), (8, 4), (1, 1), (7, 7), (1, 7), (7, 1), (4, 4), (4, 2), (4, 6), (2, 4),
                   (6, 4)):
        c.px(x + px, y + py, 'white' if (px, py) != (4, 4) else 'ice')


def _icon_rain(c, x, y):
    c.circle(x + 4, y + 3, 3, '#6f7a9a')
    c.rect(x + 1, y + 3, 7, 2, '#6f7a9a')
    for px in (2, 4, 6):
        c.vline(x + px, y + 6, y + 8, 'light_sky')


def _icon_wind(c, x, y):
    for yy, ln in ((2, 7), (4, 5), (6, 8)):
        c.hline(x, x + ln - 1, y + yy, 'white')
    c.px(x + 7, y + 1, 'white')
    c.px(x + 8, y + 7, 'white')


_ICON = {'snow': _icon_snow, 'rain': _icon_rain, 'wind': _icon_wind}
WEATHER_TTL = 150                   # steps a first-weather toast may wait for its turn
