"""The Heartland as an explorable diorama (art: scenes/s06_heartland.py, used read-only).

The film shows three postcards joined by cloud sweeps, so the region has three areas, in film order:
  0  NEIGHAGRA FALLS: the horseshoe falls seen from downstream, with the viewing platform in front.
  1  FOAL MOUNTAIN & PONHENGE: the blue massif, the meadows with the railway tunnel, the stone circle.
  2  HOLLOW SHADES: the misty forest village with the Well of Shade (the film's tilt down is a vertical
     camera here).
Each area is the film's own shot renderer (`render_a/b/c`) bound to replacement globals (`rebind`): the
player drives the camera and the film's balloon is switched off (the base class draws the player's).
Timed film beats run on game clocks:
  - the empty barrel bobs up on the upper river, hops toward the lip, tips over and drops into the mist
    (splash), then surfaces again a few seconds later;
  - the sightseeing zeppelin cruises right to left over the plateau and comes round again off screen;
  - the Friendship Express rolls in from the east and chugs into the tunnel each time you arrive at the
    meadows, then again TRAIN_GAP steps after each run;
  - the cloud shadows slide over the slopes and wrap around off screen (the film's run off for good).

Things to set off with SPACE:
  - BARREL (falls): a barrel bobs up by the lip, hops to the edge, teeters, tips over and plunges into
    the mist with a big splash of spray;
  - RAINBOW (falls): the bow in the spray blazes, a second bow appears outside it and a glint runs along it;
  - TRAIN (tunnel mouth): the Friendship Express bursts out of the tunnel whistling and steams off east
    (if it is already out in the meadows, it whistles and puffs a tall plume);
  - STONES (Ponhenge altar): every rune blazes, rings of light ripple out over the knoll and a pillar of
    light rises from the circle;
  - WELL (Well of Shade): the well's cold light surges up through the canopy gap, motes of light spiral
    up and the village lanterns flare gold one after another.
Three atlas stamps are hidden: low in the shaded gorge left of the falls, at the far end of the meadows
over Rambling Rock Ridge, and up in the dark canopy at the far left of Hollow Shades.

The balloon is the 'm' sprite everywhere (it flies in front of every layer here, unlike the film's distant
one). Hollow Shades draws its god-rays with a cached copy of engine.fx.light_beam (same pixels, ~25x
faster than the film's per-row loop).
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, fx, bayer, col, mix, seg, lerp, hash01
import scenes.s06_heartland as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# camera limits (the largest camera with art on every layer):
#   falls:  main 372 (p 1) -> 52; platform 390 (p 1.35) -> 51; far hills 350 (p .15) -> 200
#   foal:   meadows 344 (p .55) -> 43; knoll 364 (p 1) -> 44; mountain 344 / ridge 332 (p .3 / .15) -> 80;
#           grass strip 400 (p 1.5) -> 53
#   shades: village 344 x 212 (p 1) -> x 24, y 32; fg 360 x 242 (p 1.4 / 1.25) -> 28 / 49;
#           mid 350 x 212 (p .6 / .7) -> 50 / 45; far 340 x 212 (p .3 / .4) -> 66 / 80
FALLS_CAM = 51
FOAL_CAM = 43
SHADES_CAM_X, SHADES_CAM_Y = 24, 32

# ---- falls clocks
BARREL_PERIOD = 288     # 16 beats: surface (60..71), ride + hops (72..161), tip + drop (162..170), gone
BARREL_SURFACE = 60
ZEP_SPEED = 0.75        # px / step on its layer (the film: 84 px over 108 frames)
ZEP_X0, ZEP_SPAN = 344, 430     # layer x from off screen right (>= 332 at any camera) to off screen left
ZEP_PHASE = 150
ZEP_HORN_X = W - 20

# ---- meadow clocks (meadow-layer px, parallax .55)
TRAIN_X0 = 352          # westbound loco front at the start of a run: off screen at any camera
TRAIN_SPEED = 1.15
TRAIN_RUN = int((TRAIN_X0 + 10) / TRAIN_SPEED) + 2      # until the last carriage is in the tunnel
TRAIN_GAP = 300         # steps from the end of a run to the next westbound run
PORTAL_X = 57           # the tunnel mouth: draw_train clips the train left of it
OUT_V0, OUT_ACC = 0.9, 0.004                            # eastbound: front = PORTAL_X + v0 u + acc u^2
OUT_RUN = int((-OUT_V0 + math.sqrt(OUT_V0 ** 2 + 4 * OUT_ACC * (W + 66 - PORTAL_X))) / (2 * OUT_ACC)) + 2

_PEAK = sc.PEAKS[0]

# ---- hotspots (world px; the train's sits on the meadow layer)
BARREL_SPOT = (sc.CXA - 38, 83)              # the upper river, just above the lip
RAINBOW_SPOT = (sc.CXA + 14 + 40, 116)       # on the bow's right leg, in the spray
PORTAL_SPOT = (62, 112)                      # the tunnel mouth (meadow layer)
ALTAR = (sc.KNOLL_C[0], sc.KNOLL_C[1] - 2)   # the middle of the stone circle
LANTERNS = (58, 212, 266)                    # Hollow Shades lantern posts (village x)
LANTERN_SPOT = (LANTERNS[1] + 4, sc.GYC - 18)    # the dead lantern right of the well

# the player's barrel: surface, hop, hop, teeter, tip (scene px)
PB_SURF, PB_HOP, PB_LIP = (sc.CXA - 27, 80), (sc.CXA - 34, 83), (sc.CXA - 41, 86)
PB_TIP, PB_SPLASH = 18, 25                   # steps after SPACE: tips over the lip / hits the mist

BOW_BANDS = ['red', 'orange', 'gold', 'leaf', 'sky', 'violet']


def _nop(*_a, **_k):
    pass


def _env(a, rise, hold, fall):
    """0..1 envelope of an event at step a: ramps up over `rise`, holds until `hold`, fades by `fall`."""
    if a < 0 or a >= fall:
        return 0.0
    if a < rise:
        return a / rise
    if a < hold:
        return 1.0
    return 1 - (a - hold) / (fall - hold)


class Heartland(GlowKeep, Region):
    SEG = 'heartland'
    CARD = ('HEARTLAND', ['FALLS - STONES - SHADES', 'EAST OF CANTERLOT'])

    def build_areas(self):
        sc._village_c()                      # fills sc.HOUSES
        h2x, h2y, _ = sc.HOUSES['h2']
        falls = [Landmark('NEIGHAGRA FALLS', sc.CXA, (sc.CXA + 2, sc.LIP0 - 1, 'up', 26))]
        foal = [Landmark('FOAL MOUNTAIN', _PEAK[0] + 12, (_PEAK[0] + 1, _PEAK[1] + 6, 'down', 16), p=0.3),
                Landmark('PONHENGE', sc.KNOLL_C[0] + 4, (sc.KNOLL_C[0] + 4, sc.KNOLL_C[1] - 35, 'up', 12))]
        shades = [Landmark('HOLLOW SHADES', h2x, (h2x, h2y + 2, 'right', 14)),
                  Landmark('WELL OF SHADE', sc.WELL_X, (sc.WELL_X, sc.WELL_Y - 57, 'up', 8))]
        return [
            Area(self.render_falls, cam_x=(0, FALLS_CAM), cam_y=(0, 0), y_range=(30, 140),
                 start=(70, 76), landmarks=falls, balloon='m',
                 hotspots=[Hotspot('barrel', *BARREL_SPOT, 'splash', cooldown=90),
                           Hotspot('rainbow', *RAINBOW_SPOT, 'spark', cooldown=90)],
                 stamps=[Stamp('gorge', 24, 122)], sky=_sky_falls),
            Area(self.render_foal, cam_x=(0, FOAL_CAM), cam_y=(0, 0), y_range=(26, 150),
                 start=(50, 66), landmarks=foal, balloon='m',
                 hotspots=[Hotspot('train', *PORTAL_SPOT, 'whistle', cooldown=60, p=0.55),
                           Hotspot('stones', *ALTAR, 'gem', cooldown=80)],
                 stamps=[Stamp('ridge', 338, 64)], sky=_sky_foal),
            Area(self.render_shades, cam_x=(0, SHADES_CAM_X), cam_y=(0, SHADES_CAM_Y), y_range=(30, 172),
                 start=(60, 70), landmarks=shades, balloon='m', balloon_tint=('#8a86c8', 0.55),
                 hotspots=[Hotspot('well', *LANTERN_SPOT, 'lantern', cooldown=100)],
                 stamps=[Stamp('canopy', 20, 28)],
                 sky=lambda f, cx, cy: _sky_shades(f, cx, cy, self.rays())),
        ]

    # -------------------------------------------------------------- clocks
    def init_state(self):
        self.barrel_t0 = 0
        self.train = None           # None | ('west' | 'east', step the run starts)
        self.train_next = None      # step a westbound run starts by itself (None: when you next arrive)
        self.whistle_t = -10 ** 6
        self.sound_ai = None        # area whose entry sound was played last, and when
        self.sound_t = 0

    def on_enter(self):
        self.barrel_t0 = self.g.t + 10
        self.sound_ai = None

    def on_area(self, ai):
        if ai == 1 and self.train is None:           # the train comes each time you arrive
            self.train, self.train_next = ('west', self.g.t + 24), None

    def barrel_f(self, t):
        """The film's barrel frame (72..170 = ride, hops, tip, drop), from BARREL_SURFACE."""
        return BARREL_SURFACE + (t - self.barrel_t0) % BARREL_PERIOD

    @staticmethod
    def zep_x(t):
        return int(ZEP_X0 - ((t + ZEP_PHASE) * ZEP_SPEED) % ZEP_SPAN)

    def train_run(self, t):
        """(kind, steps into the run) of the train on the line, or None."""
        if self.train is None or t < self.train[1]:
            return None
        return self.train[0], t - self.train[1]

    def spot_a(self, key, t):
        return t - self.spot_t.get(key, -10 ** 6)

    # -------------------------------------------------------------- sound + triggers
    def update(self, t):
        g = self.g
        tr = self.train_run(t)
        if tr is not None and tr[1] > (TRAIN_RUN if tr[0] == 'west' else OUT_RUN):
            self.train, self.train_next = None, t + TRAIN_GAP
        if self.train is None and self.train_next is not None and t >= self.train_next:
            self.train, self.train_next = ('west', t), None
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.ai != self.sound_ai and self.switch is None:
            self.sound_ai, self.sound_t = self.ai, t
            if self.ai in (1, 2):
                g.sfx('wind', 0.45)                  # meadow wind / cold draught through the village
        k = t - self.sound_t
        if self.ai == 0:
            if k % 54 == 0:                          # the falls' roar, re-cued as the ambience loops
                g.sfx('waterfall', 0.5 if k == 0 else 0.32)
            if self.barrel_f(t) == 170:
                g.sfx('splash', 0.4)                 # the barrel drops into the mist
            if self.zep_x(t) <= ZEP_HORN_X < self.zep_x(t - 1):
                g.sfx('horn', 0.2)                   # the zeppelin comes into view
            if self.spot_a('barrel', t) == PB_SPLASH:
                g.sfx('splash', 0.75)
            if self.spot_a('rainbow', t) == 12:
                g.sfx('sparkle', 0.4)
        elif self.ai == 1:
            tr = self.train_run(t)
            if tr is not None and tr[1] % 36 == 0:
                kind, u = tr
                if kind == 'west':
                    xm = TRAIN_X0 - u * TRAIN_SPEED
                    if xm < W + 20:
                        g.sfx('chug', 0.22 * max(0.4, min(1.0, xm / 120)), (u // 36) % 2)
                else:
                    g.sfx('chug', 0.3 * max(0.35, 1 - u / OUT_RUN), (u // 36) % 2)
            if k % 240 == 120:
                g.sfx('wind', 0.25)
        else:
            if k % 150 == 40:
                g.sfx('creak', 0.4)                  # the loose shutter
            if k % 150 == 115:
                g.sfx('creak', 0.3, 1)               # the hanging moon sign
            if k % 240 == 200:
                g.sfx('wind', 0.3)
            a = self.spot_a('well', t)
            for i, lx in enumerate(LANTERNS):
                if a == _lantern_delay(lx):
                    g.sfx('clink', 0.3, i)           # a lantern flares

    def on_interact(self, key, t):
        g = self.g
        if key == 'barrel':
            g.sfx('pop', 0.45, 1)
            g.sfx('bubbles', 0.3)
        elif key == 'rainbow':
            g.sfx('shimmer', 0.6)
        elif key == 'stones':
            g.sfx('gong', 0.45)
            g.sfx('crystal', 0.4)
        elif key == 'well':
            g.sfx('shimmer', 0.5)
            g.sfx('crystal', 0.35)
        elif key == 'train':
            self.whistle_t = t
            tr = self.train_run(t)
            ox = int(self.cx * 0.55)
            if tr is not None and tr[0] == 'west':
                xm = TRAIN_X0 - tr[1] * TRAIN_SPEED
                hidden = xm - ox > W - 6 or xm + 64 < PORTAL_X + 2   # not in yet / all in the tunnel
            else:
                hidden = tr is None
            if hidden:                                # out of the tunnel it comes
                self.train, self.train_next = ('east', t), None
                g.sfx('steam', 0.45)
            g.sfx('whistle', 0.75)

    def day(self):
        """0 (night) .. 1 (day) by the world clock: sunlit things (the falls' rainbow, the god-rays through
        the Hollow Shades canopy) fade out at night."""
        return 1.0 - daylight.night(self.g.hour)

    def rays(self):
        """0..1: the god-rays through the Hollow Shades canopy: by day; fading with the night, and to 0.4
        through the evening's golden hour (the low sun's rays no longer fall in through the gap: the
        sunset grade turned them into muddy brown bands)."""
        h = self.g.hour % 24
        t = max(0.0, min(1.0, (h - 16.2) / 2.2)) if h >= 12 else 0.0
        return self.day() * (1 - 0.6 * t * t * (3 - 2 * t))

    # -------------------------------------------------------------- the world clock's light
    _raw = None                 # the area's frame as rendered (for the sky repaint in after_light)

    def auto_glow(self):
        """No painted spot glows by itself here (the falls' gold-rimmed lip, the meadow flowers, the inn's
        moon sign): the real lights are kept by mask (the area renders: keep_glow)."""
        return False

    def after_light(self, c, t):
        """After the relight: the kept lights (GlowKeep), then the clock's sky over the whole open sky at
        every hour. The base keeps the painted sky while the clock's looks alike (~9.5 to ~14 h): no sun on
        its arc then, and the film's sun glare in the top-left corner at 13:30."""
        super().after_light(c, t)
        raw, self._raw = self._raw, None
        if raw is None:
            return
        a, f, ix, iy = raw
        hour = self.g.hour
        if daylight.sky_weight(self.HOUR, hour) >= 0.999:
            return
        key = (self.ai, t, ix, iy)                       # the base's memo: once per step
        if self._sky_memo is None or self._sky_memo[0] != key:
            self._sky_memo = (key, daylight.sky_pixels(a, self.area.sky(f, ix, iy).a))
        mask = self._sky_memo[1]
        if mask.any():
            y0, y1 = self.area.cam_y
            span = y1 - y0 + 130                         # (as the base: to the area's horizon)
            if self.area.horizon is not None:
                span = max(span, self.area.horizon - (y0 - 10) + 4)
            daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, span, t)

    def _keep(self, c, rec, f, cx, cy):
        """End of an area render: keep its lights lit (GlowKeep) and the frame for the sky repaint."""
        self.keep_glow(c, _kept(rec, c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        return c

    # -------------------------------------------------------------- area 0: Neighagra Falls
    def render_falls(self, f, cx, cy):
        t = self.g.t
        bf = self.barrel_f(t)
        zx = self.zep_x(t)
        ab = self.spot_a('barrel', t)
        ar = self.spot_a('rainbow', t)
        k = _lamp_k(self.g.hour)                  # the lights (see the world clock's lights below)
        rec = []

        def zeppelin(c, f_, _x, y):
            x = zx - int(cx * 0.25)
            sc.draw_zeppelin(c, f_, x, y)
            if k > ZEP_V:                         # its gondola's windows lit
                _light_px(c, rec, [(x + 18 + i * 3 + q % 2, y + 16 + q // 2) for i in range(5) for q in range(4)],
                          [WARM[(i + q) % 3] for i in range(5) for q in range(4)])

        def barrel(c, _f, cam):
            if bf >= 72:
                sc.draw_barrel(c, bf, cam)
            elif bf >= BARREL_SURFACE:
                _surface_barrel(c, bf - BARREL_SURFACE, sc.CXA - 12, 76, cam)
            if 0 <= ab < PB_SPLASH:
                _player_barrel(c, ab, cam)

        def mist(c, f_, cam):
            sc.draw_mist(c, f_, cam)
            if PB_SPLASH <= ab < PB_SPLASH + 50:
                _splash(c, t, t - ab + PB_SPLASH, cam)

        day = self.day()

        def rainbow(c, f_, cam):
            if day >= 0.98:
                sc.draw_rainbow(c, f_, cam)
            elif day > 0.02:                              # a sunlit bow: it fades out with the daylight
                a0 = c.a.copy()
                sc.draw_rainbow(c, f_, cam)
                c.a[:] = (a0 + (c.a.astype(np.int16) - a0) * day).astype(np.uint8)
            k_ = _env(ar, 8, 60, 96)
            if k_ > 0:
                _bow_blaze(c, f_, ar, k_, cam)

        def comp(c, layer, ox, oy=0):
            sc.comp(c, layer, ox, oy)
            if layer is main:                     # the lookout on the right rim: its window and door
                for v, (ys, xs, src), lit in zip(LOOKOUT_V, _lookout_px(), LOOKOUT_LIT):
                    if k > v:
                        _light_seen(c, rec, ys, xs - ox, src, lit)

        main = sc._main_a()
        render = rebind(sc.render_a, cam_a=lambda _f: cx, draw_balloon=_nop, draw_zeppelin=zeppelin,
                        draw_barrel=barrel, draw_mist=mist, draw_rainbow=rainbow, comp=comp)
        c = render(f)
        _platform_lamp(c, rec, k, int(cx * 1.35))
        return self._keep(c, rec, f, cx, cy)

    # -------------------------------------------------------------- area 1: Foal Mountain & Ponhenge
    def render_foal(self, f, cx, cy):
        t = self.g.t
        tr = self.train_run(t)
        aw = t - self.whistle_t
        ast = self.spot_a('stones', t)
        hour = self.g.hour
        k = _lamp_k(hour)                         # the lights (see the world clock's lights below)
        lit_train = k > TRAIN_V
        rec = []

        def train(c, _f, ox):
            if tr is None:
                return
            kind, u = tr
            if kind == 'west':
                # draw_train puts the loco front at 122 - (f - CUT2) * 1.15
                (rebind(sc.draw_train, _train=_train_lit) if lit_train else sc.draw_train)(
                    c, sc.CUT2 + (122 - TRAIN_X0) / TRAIN_SPEED + u, ox)
                xm = TRAIN_X0 - u * TRAIN_SPEED
                chim = (int(xm) - ox + 4, sc.rail_y(int(xm) + 20) - 15 + 1)
            else:
                chim = _draw_train_east(c, u, ox, lit_train)
                if u < 26:
                    _portal_puff(c, u, PORTAL_X + 4 - ox, PORTAL_SPOT[1])
            if lit_train:                         # its windows, lit (the only pixels of these colours)
                _rec_cols(c, rec, TRAIN_LIT, 0, H)
            if 0 <= aw < 40 and chim[0] > PORTAL_X - ox + 1:
                _whistle_plume(c, chim[0], chim[1], aw)

        halo = _halo(hour)

        def runes(c, f_, ox):
            k_ = _env(ast, 3, 50, 80)
            pulse = (lambda ff: max(sc.rune_pulse(ff), k_)) if k_ > 0 else sc.rune_pulse
            rebind(sc.draw_runes, rune_pulse=pulse, fx=_HaloFx(halo))(c, f_, ox)
            if halo < 1:                          # the glyphs, the ring and the motes: magic, lit at night
                bare = sc.Canvas(W, H, KEY)
                rebind(sc.draw_runes, rune_pulse=pulse, fx=_HaloFx(0.0))(bare, f_, ox)
                ys, xs = np.nonzero(np.any(bare.a != np.array(KEY, np.uint8), axis=2))
                px = bare.a[ys, xs]
                bright = px.astype(np.int32) @ np.array([77, 150, 29]) > 110 * 256     # not the glyphs' shadows
                ys, xs, px = ys[bright], xs[bright], px[bright]
                same = np.all(c.a[ys, xs] == px, axis=1)
                rec.append((ys[same], xs[same], px[same]))
            if k_ > 0:
                _stones_blaze(c, f_, ast, k_, ox)

        render = rebind(sc.render_b, cam_b=lambda _f: cx, draw_balloon=_nop, draw_train=train,
                        draw_runes=runes, cloud_shadows=lambda c, f_, mt: _cloud_shadows(c, f_, mt, cx))
        c = render(f)
        return self._keep(c, rec, f, cx, cy)

    # -------------------------------------------------------------- area 2: Hollow Shades
    def render_shades(self, f, cx, cy):
        aw = self.spot_a('well', self.g.t)
        hour = self.g.hour
        k = _lamp_k(hour)                         # the lights (see the world clock's lights below)
        halo = _halo(hour)
        rec = []

        def well(c, f_, ox, oy):
            rebind(sc.draw_well, fx=_HaloFx(halo * WELL_DAY_HALO))(c, f_, ox, oy)   # (faint by day: its cold
                                                                                  # light reads at night)
            if halo < 1:                          # its cold light: the shaft, the rim, the mushrooms
                _well_light(c, rec, f_, ox, oy, 1 - halo)
            if 0 <= aw < 110:
                _well_surge(c, aw, ox, oy)

        def lanterns(c, f_, ox, oy):
            _draw_lanterns(c, f_, ox, oy, k, halo, rec)
            if 0 <= aw < 110:
                _lanterns_flare(c, f_, aw, ox, oy)
                for i, lx in enumerate(LANTERNS):            # the flared glass, kept lit at night
                    if aw >= _lantern_delay(lx):
                        x, y = lx - ox, sc.GYC - oy
                        _light_px(c, rec, [(x + 3 + q % 3, y - 19 + q // 3) for q in range(12)], None)

        def village(c, layer, ox, oy=0):
            sc.comp(c, layer, ox, oy)
            if layer is vil:                      # the windows: dark by day, lit one by one at dusk
                for (ys, xs, src, lit, v) in _shades_windows():
                    if k > v:
                        _light_seen(c, rec, ys - oy, xs - ox, src, lit)
                    else:
                        _light_seen(c, None, ys - oy, xs - ox, src, _unlit(src))

        def fog(c, *a, **kw):                     # a light seen through the mist: the misted colour is kept
            before = [c.a[ys, xs].copy() for ys, xs, _rgb in rec]
            sc.draw_fog(c, *a, **kw)
            for (ys, xs, rgb), b in zip(rec, before):
                now = c.a[ys, xs]
                upd = np.all(b == rgb, axis=1) & np.any(now != b, axis=1)
                rgb[upd] = now[upd]

        vil = sc._village_c()
        render = rebind(sc.render_c, cam_c=lambda _f: (cx, cy), draw_balloon=_nop, fx=_FX(self.rays()),
                        draw_well=well, draw_lanterns=lanterns, comp=village, draw_fog=fog)
        c = render(f)
        return self._keep(c, rec, f, cx, cy)


# ================================================================== the world clock's lights
# The film paints the Heartland by day (10:53). At dusk and night the region keeps its REAL lights itself
# (auto_glow is off: the relight would keep the gold sunlit rims along the falls' lip, the meadow flowers
# and the inn's moon sign glowing like lamps; GlowKeep keeps them lit): the platform lamp, the lookout's
# window and door, the zeppelin's gondola, the train's windows, the runes and their ring of light (magic),
# Hollow Shades' windows, its cold lanterns (the dead one stays dead) and the Well of Shade's light. They
# come on one by one through the dusk (_lamp_k: 18.4 -> 19.5 h, out 5.3 -> 6.3 h); by day the lamp's
# glass, the lanterns and the tower's cold window are unlit. Painted halos fade as the night comes: only
# the light itself shines then (a halo darkened by the night is a muddy ring).
KEY = (255, 0, 255)
WARM = tuple(tuple(int(v) for v in w) for w in nightwin.WARM)
LAMP = (321, 102, 4, 4)                 # the platform lamp's glass (sc._fg_a: lx - 1, 102; fg layer px)
LAMP_V = 0.1
LAMP_DAY = ('#8e9abb', '#c4cee6')       # its glass by day (glass, glint) for the painted butter / white
LOOKOUT_V = (0.2, 0.55)                 # the lookout's window, then its door
LOOKOUT_LIT = ('#ffe07a', '#e8a050')
ZEP_V = 0.4
TRAIN_V = 0.3
TRAIN_LIT = ('#ffd466', '#ffe07a', '#ffc455')      # its carriage windows, their glints, the cab
LANTERN_V = (0.15, 0.6, 0.35)           # Hollow Shades' lanterns (the middle one is the dead one)
SHADES_WINDOWS = (                      # village px (x, y, w, h, lit: 'warm' | 'cold'), sc._village_c
    (32, 145, 7, 9, 'warm'), (94, 162, 7, 8, 'warm'), (77, 123, 5, 5, 'warm'), (236, 124, 7, 8, 'cold'))


def _lamp_k(hour):
    """0 (day) .. 1 (night): how many of the lights are on. Each light has its threshold in 0..1: they come
    on one by one from 18.4 to 19.5 h and go out from 5.3 to 6.3 h."""
    h = hour % 24
    return max(0.0, min(1.0, (h - 18.4) / 1.1 if h >= 12 else (6.3 - h) / 1.0))


WELL_DAY_HALO = 0.2             # the Well of Shade's painted halo by day (the film's full one outshone its night)


def _halo(hour):
    """How much of a light's painted halo shows: all of it by day and in the dusk, none at night."""
    return max(0.0, min(1.0, 1 - 2.2 * daylight.night(hour)))


class _HaloFx:
    """engine.fx with its soft glows scaled by k."""
    def __init__(self, k):
        self.k = k

    def glow(self, c, x, y, r, color, strength=0.5, rings=4, ox=0, oy=0):
        if self.k > 0.02:
            fx.glow(c, x, y, r, color, strength=strength * self.k, rings=rings, ox=ox, oy=oy)

    def __getattr__(self, name):
        return getattr(fx, name)


def _light_px(c, rec, pts, cols):
    """Paint the screen pixels pts the colours cols (None: keep their colours) and record them in rec."""
    xs, ys = (np.array(v) for v in zip(*pts))
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if cols is not None:
        cc = np.array([col(v) for v in cols], np.uint8)[ok]
        c.a[ys[ok], xs[ok]] = cc
    rec.append((ys[ok], xs[ok], c.a[ys[ok], xs[ok]].copy()))


def _light_seen(c, rec, ys, xs, src, lit):
    """Light the pixels (ys, xs) (screen) still showing their painted colours src (nothing in front) in the
    colour lit (a colour, or one per pixel); record them in rec (None: not a light, e.g. a window put out
    by day)."""
    lit = np.asarray(col(lit) if isinstance(lit, str) else lit, np.uint8)
    if lit.ndim == 1:
        lit = np.broadcast_to(lit, src.shape)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    ys, xs, src, lit = ys[ok], xs[ok], src[ok], lit[ok]
    seen = np.all(c.a[ys, xs] == src, axis=1)
    if not seen.any():
        return
    ys, xs = ys[seen], xs[seen]
    c.a[ys, xs] = lit[seen]
    if rec is not None:
        rec.append((ys, xs, c.a[ys, xs].copy()))


def _rec_cols(c, rec, cols, y0, y1):
    """Record every pixel in rows y0..y1 of one of the colours (lights of a colour nothing else has)."""
    sub = c.a[y0:y1]
    hit = np.zeros(sub.shape[:2], bool)
    for cc in cols:
        hit |= np.all(sub == np.array(col(cc), np.uint8), axis=2)
    ys, xs = np.nonzero(hit)
    rec.append((ys + y0, xs, sub[ys, xs].copy()))


def _kept(rec, c):
    """The mask of the recorded light pixels still showing their colours (nothing drawn over them)."""
    m = np.zeros((H, W), bool)
    for ys, xs, rgb in rec:
        if len(ys):
            s = np.all(c.a[ys, xs] == rgb, axis=1)
            m[ys[s], xs[s]] = True
    return m


def _platform_lamp(c, rec, k, fcam):
    """The viewing platform's lamp: its painted butter glass lit at dusk and night (kept), plain glass by day."""
    x, y, w, h = LAMP
    xs = np.repeat(np.arange(x, x + w) - fcam, h)
    ys = np.tile(np.arange(y, y + h), w)
    ok = (xs >= 0) & (xs < W)
    xs, ys = xs[ok], ys[ok]
    if not len(xs):
        return
    px = c.a[ys, xs]
    glass = np.all(px == np.array(col('butter'), np.uint8), axis=1)
    glint = np.all(px == np.array(col('white'), np.uint8), axis=1)
    if k > LAMP_V:
        sel = glass | glint
        rec.append((ys[sel], xs[sel], px[sel]))
    else:
        c.a[ys[glass], xs[glass]] = col(LAMP_DAY[0])
        c.a[ys[glint], xs[glint]] = col(LAMP_DAY[1])


@lru_cache(maxsize=1)
def _lookout_px():
    """((ys, xs, painted rgb) of the lookout's window, then of its door): main layer px (sc._main_a)."""
    a, _m = sc._main_a()
    gx = sc.CXA + sc.AHW + 34
    gy = int(sc.rim_y(gx)) - 2
    out = []
    for pts in ([(gx - 2 + i % 2, gy - 10 + i // 2) for i in range(4)],
                [(gx - 1 + i % 2, gy - 5 + i // 2) for i in range(10)] + [(gx - 1, gy - 6)]):
        xs, ys = (np.array(v) for v in zip(*pts))
        out.append((ys, xs, a[ys, xs].copy()))
    return out


@lru_cache(maxsize=1)
def _train_lit():
    """The Friendship Express at night: its carriage windows and the cab lit."""
    s = sc._train().copy()
    rgb = s[..., :3]
    for src, dst in (('periwinkle', TRAIN_LIT[0]), ('glass_hi', TRAIN_LIT[1]), ('baby_blue', TRAIN_LIT[2])):
        hit = np.all(rgb == np.array(col(src), np.uint8), axis=2) & (s[..., 3] > 0)
        rgb[hit] = col(dst)
    return s


@lru_cache(maxsize=1)
def _shades_windows():
    """(ys, xs, painted rgb, lit colour(s), threshold) of Hollow Shades' unboarded windows' glass (village
    px; the timber cross is not glass): warm, the tower's its own cold light, brighter."""
    a, _m = sc._village_c()
    timber = np.array(col(sc.TIMBER), np.uint8)
    out = []
    for i, (x, y, w, h, kind) in enumerate(SHADES_WINDOWS):
        yy, xx = np.mgrid[y:y + h, x:x + w]
        ys, xs = yy.ravel(), xx.ravel()
        glass = ~np.all(a[ys, xs] == timber, axis=1)
        ys, xs = ys[glass], xs[glass]
        if kind == 'warm':
            lit = np.array([WARM[int(hash01(i, 5) * 3)]] * len(ys), np.uint8)
            lit[(ys == y) | (xs == x)] = WARM[2]
        else:
            lit = np.array([col(mix(sc.WIN_DARK, sc.GLOW, 0.95))] * len(ys), np.uint8)
            lit[ys >= y + h - 2] = col(mix(sc.WIN_DARK, sc.GLOW, 0.8))
        out.append((ys, xs, a[ys, xs].copy(), lit, 0.1 + 0.6 * hash01(i, 6)))
    return out


def _unlit(src):
    """A window's glass by day: the film's dark glass (the tower's faint cold light put out)."""
    return np.broadcast_to(np.array(col(sc.WIN_DARK), np.uint8), src.shape).copy()


def _draw_lanterns(c, f, ox, oy, k, halo, rec):
    """sc.draw_lanterns by the clock: the cold lanterns' glass unlit by day; lit one by one at dusk (the
    middle one, the dead one, only flickers faintly); their halos fade as the night comes."""
    for i, (lx, ly) in enumerate(((58, sc.GYC), (212, sc.GYC), (266, sc.GYC))):
        x, y = lx - ox, ly - oy
        c.rect(x, y - 22, 2, 22, '#2a2440')
        c.vline(x, y - 22, y - 1, '#3a3456')
        c.hline(x - 1, x + 4, y - 22, '#2a2440')
        c.vline(x + 4, y - 22, y - 20, 'grey')
        fl = hash01(i, f // 3)
        lit = k > LANTERN_V[i]
        on = (0.75 + 0.25 * fl if i != 1 else 0.15 + 0.1 * fl) if lit else 0.0
        c.rect(x + 3, y - 19, 3, 4, mix('#3a3a6a', 'ice', on))
        c.frame(x + 2, y - 20, 5, 6, '#2a2440')
        if lit:
            fx.glow(c, x + 4, y - 17, 7, 'periwinkle', (0.12 + 0.18 * on) * halo, rings=3)
            if i != 1:
                _light_px(c, rec, [(x + 3 + q % 3, y - 19 + q // 3) for q in range(12)], None)


def _well_light(c, rec, f, ox, oy, n):
    """The Well of Shade's own light (sc.draw_well) at night: its glowing shaft (the two inner ellipses,
    shining brighter in the dark by n = 0..1, the night), the lit rim, the cold mist curling out of it and
    the glowing mushrooms at its foot, recorded in rec where they are seen."""
    x, y = sc.WELL_X - ox, sc.WELL_Y - oy
    R, WH = 17, 14
    k = 0.55 + 0.45 * math.sin(f * 0.09)
    shaft = [(0.25 + 0.25 * k, 0.45), (0.55 + 0.40 * k, 0.4)]     # (painted mix, more at night)
    pts = [(x + i, y - WH + j) for i in range(-R, R + 1) for j in range(-24, 4)]
    xs, ys = (np.array(v) for v in zip(*pts))
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    px = c.a[ys, xs]
    hit = np.zeros(len(xs), bool)
    for m0, more in shaft:                    # the shaft, brighter at night
        on = np.all(px == np.array(col(mix('#1c1a38', sc.GLOW, m0)), np.uint8), axis=1)
        c.a[ys[on], xs[on]] = col(mix('#1c1a38', sc.GLOW, min(1.0, m0 + more * n)))
        hit |= on
    for cc in (mix(sc.STONE_C[4], sc.GLOW, 0.3 + 0.3 * k), mix('periwinkle', 'white', 0.45)):   # rim, mist
        hit |= np.all(px == np.array(col(cc), np.uint8), axis=1)
    mp = []
    for i, (mx, my) in enumerate(((-R - 4, 0), (-R - 7, 1), (R + 3, 1), (R + 6, 0))):
        cc = col(mix('#5a6aa8', 'ice', 0.5 + 0.5 * math.sin(f * 0.13 + i * 1.7)))
        mp += [(x + mx + d, y + my - 2, cc) for d in (-1, 0, 1)] + [(x + mx, y + my - 3, cc)]
    mx_, my_ = np.array([p[0] for p in mp]), np.array([p[1] for p in mp])
    ok = (mx_ >= 0) & (mx_ < W) & (my_ >= 0) & (my_ < H)
    mcol = np.array([p[2] for p in mp], np.uint8)[ok]
    mx_, my_ = mx_[ok], my_[ok]
    mh = np.all(c.a[my_, mx_] == mcol, axis=1)
    rec.append((np.concatenate([ys[hit], my_[mh]]), np.concatenate([xs[hit], mx_[mh]]),
                np.concatenate([c.a[ys[hit], xs[hit]], mcol[mh]])))


# ================================================================== falls: barrels, splash, bow
def _surface_barrel(c, a, x, y, cam, rows_per=2):
    """A barrel bobbing up out of the rapids (`a` steps after it starts to surface)."""
    k = a // rows_per + 1                                 # rows above the water: 1..6
    sx, sy = x - cam, y
    h, w = sc._BARREL.rgba.shape[:2]
    sc._BARREL.sub(0, 0, w, min(k, h)).draw(c, sx, sy + 1, anchor='bottom')
    c.hline(sx - 3, sx + 3, sy + 1, '#d8f2f8')
    if (a // 3) % 2:
        c.px(sx - 4, sy + 1, 'white')
        c.px(sx + 4, sy + 1, 'white')


def _player_barrel(c, a, cam):
    """The barrel sent over the falls with SPACE: surface, two hops, teeter on the lip, tip, plunge."""
    if a < 5:
        _surface_barrel(c, a, PB_SURF[0], PB_SURF[1], cam, rows_per=1)
        for i in range(10):                               # it pops up out of the rapids
            ang = -math.pi * (0.1 + 0.8 * hash01(i, 61))
            v = 0.8 + 0.8 * hash01(i, 62)
            c.px(int(PB_SURF[0] - cam + math.cos(ang) * v * (a + 1) * 1.5),
                 int(PB_SURF[1] + math.sin(ang) * v * (a + 1) + 0.15 * a * a), 'white')
        return
    lift = 0.0
    if a < 10:
        u = (a - 4) / 6
        x, y = lerp(PB_SURF[0], PB_HOP[0], u), lerp(PB_SURF[1], PB_HOP[1], u)
        lift = math.sin(u * math.pi) * 3
    elif a < 15:
        u = (a - 9) / 6
        x, y = lerp(PB_HOP[0], PB_LIP[0], u), lerp(PB_HOP[1], PB_LIP[1], u)
        lift = math.sin(u * math.pi) * 3
    elif a < PB_TIP:
        x, y = PB_LIP[0] - (a % 2), PB_LIP[1]          # teetering on the edge
    else:
        u = (a - PB_TIP) / 7.0
        x, y = PB_LIP[0] - u * 3, 88 + u * u * 46
        rot = sc.Sprite(np.ascontiguousarray(np.rot90(sc._BARREL.rgba, k=1 + (a // 2) % 4)))
        c.blit(rot, int(round(x)) - cam - 3, int(round(y)) - 3)
        return
    sx, sy = int(round(x)) - cam, int(round(y - lift))
    if lift < 0.5:
        c.hline(sx - 3, sx + 3, sy + 1, '#d8f2f8')
        c.px(sx - 4 - (a // 3) % 2, sy + 1, 'white')
        c.px(sx + 4 + (a // 3) % 2, sy + 1, 'white')
    sc._BARREL.draw(c, sx, sy + 1 - (a // 4) % 2 * (lift < 0.5), anchor='bottom')


_CROWN = ('#1f8a8f', '#45b3b5', '#8fd9da', 'white', '#12606e', '#c8f1f2')


def _splash(c, t, t0, cam):
    """The plunge: a crown of green water thrown high out of the mist, spray, puffs billowing up."""
    a = t - t0
    x, y = PB_LIP[0] - 3 - cam, 132
    if a < 12:
        fx.glow(c, x, y - 4, 26, 'white', 0.55 * (1 - a / 12), rings=3)
    for k in range(7):                                    # a plume of spray shooting up the curtain
        s = a - k * 2
        if 0 <= s < 36:
            rise = 50 * (1 - (1 - min(1.0, s / 20)) ** 2) * (0.55 + 0.45 * hash01(k, 5))
            r = 3 + s * 0.22 + k * 0.6
            px = x + (k - 3) * 3 + math.sin(s * 0.2 + k) * 2
            sc.draw_puff(c, px, y - 4 - rise, r, 11 + k, 1.0 if s < 22 else 0.5, tone=k % 2)
    for i in range(96):                                   # the water crown
        d = a - (i % 6)
        if not 0 <= d <= 44:
            continue
        ang = math.radians(-160 + 140 * hash01(i, 91))
        v = 1.5 + 2.1 * hash01(i, 92) * (1.2 - abs(math.cos(ang)) * 0.5)
        px = x + math.cos(ang) * v * d * 0.8
        py = y + math.sin(ang) * v * d + 0.085 * d * d
        if py > y + 8 or (d > 30 and (i + d) % 3 == 0):
            continue
        sz = 2 if d < 24 and i % 4 else 1
        c.rect(int(px), int(py), sz, sz, _CROWN[i % len(_CROWN)])
    ui.burst(c, t, t0, x, y - 2, colors=('white', '#c8f1f2', '#8fd9da'), n=20, seed=17, power=0.9,
             life=30, gravity=0.06, size=1)


@lru_cache(maxsize=64)
def _arc_geom(cx, r0, nb):
    """Pixels of each band of a bow centred at (cx, 158) (screen), like the film's draw_rainbow."""
    cy = 158
    x0, x1 = max(0, cx - r0 - 2), min(W, cx + r0 + 3)
    y0, y1 = max(0, cy - r0 - 2), min(H, 146)
    if x1 <= x0 or y1 <= y0:
        return ()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
    ang = np.arctan2(cy - yy - 0.5, xx + 0.5 - cx)
    fade = (np.clip(np.sin(ang) * 1.5, 0, 1) * np.clip((146 - yy) / 22.0, 0, 1)).astype(np.float32)
    out = []
    for k in range(nb):
        m = (d >= r0 - k - 1) & (d < r0 - k) & (ang > 0.15) & (ang < 2.99)
        out.append((yy[m], xx[m], fade[m][:, None]))
    return tuple(out)


def _arc(c, cx, r0, bands, alpha):
    for (ys, xs, fade), bc in zip(_arc_geom(cx, r0, len(bands)), bands):
        src = c.a[ys, xs].astype(np.float32)
        c.a[ys, xs] = np.clip(src + (np.array(col(bc), np.float32) - src) * (alpha * fade), 0, 255).astype(np.uint8)


def _bow_blaze(c, f, a, k, cam):
    """The rainbow blazes: brighter bow, a second bow outside it, a glint running along it, twinkles."""
    cx, cy, r0 = sc.CXA - cam + 14, 158, 58
    _arc(c, cx, r0, BOW_BANDS, 0.55 * k)
    _arc(c, cx, r0 + 11, BOW_BANDS[::-1], 0.32 * k)
    if a < 36:                                            # a glint sweeps along the bow, left to right
        th = 2.95 - (a / 36) * 2.75
        gx, gy = cx + math.cos(th) * (r0 - 3), cy - math.sin(th) * (r0 - 3)
        if gy < 144:
            fx.glow(c, gx, gy, 8, 'white', 0.6, rings=3)
            fx.sparkle(c, int(gx), int(gy), a, 'white', period=8, size=3)
    for j in range(14):
        th = 0.35 + 2.45 * hash01(j, 77)
        rr = r0 - 2 + (hash01(j, 78) - 0.3) * 14
        x, y = cx + math.cos(th) * rr, cy - math.sin(th) * rr
        if y < 142 and a > 4 + j * 2:
            fx.sparkle(c, int(x), int(y), a + j * 5, ('white', 'butter', 'cotton')[j % 3], period=18,
                       size=2 if k > 0.5 else 1)


# ================================================================== meadows: trains, stones
@lru_cache(maxsize=2)
def _train_east(lit=False):
    return np.ascontiguousarray((_train_lit() if lit else sc._train())[:, ::-1])


def _out_front(u):
    return PORTAL_X + OUT_V0 * u + OUT_ACC * u * u


def _draw_train_east(c, u, ox, lit=False):
    """The Friendship Express coming out of the tunnel eastbound (the film's train, mirrored; lit: its
    windows lit, at night); returns its chimney top (screen)."""
    xf = _out_front(u)
    portal = PORTAL_X - ox
    tx = int(xf) - 64 - ox
    ty = sc.rail_y(int(xf) - 44) - 15
    spr = _train_east(lit).copy()
    cut = portal - tx
    if cut > 0:
        spr[:, :min(cut, spr.shape[1])] = 0
    c.blit(spr, tx, ty)
    chim = tx + 59
    for i in range(6):                                    # steam trailing back over the carriages
        age = ((u // 2) + i * 3) % 18
        sx = chim - age * 1.5 - i
        sy = ty - 1 - age * 0.6
        if sx <= portal + 2 or age >= 15:
            continue
        sc.disc(c, sx, sy, 1 + age // 6, 'white' if age < 9 else '#dfe9f2', 1.0 if age < 10 else 0.5)
    return chim, ty + 1


def _portal_puff(c, a, x, y):
    """A cloud of steam rolling out of the tunnel mouth as the train bursts out."""
    for k in range(7):
        s = a - k * 2
        if 0 <= s < 22:
            px = x + s * 0.9 + k * 1.5
            py = y - 2 - s * 0.7 - (k % 3)
            sc.disc(c, px, py, 2 + s // 5, 'white' if s < 12 else '#dfe9f2', 1.0 if s < 15 else 0.5)


def _whistle_plume(c, x, y, a):
    """The whistle: a tall white plume shooting up from the chimney."""
    for k in range(11):
        s = a - k * 2
        if 0 <= s < 28:
            px = x + math.sin(s * 0.3 + k) * 1.5
            py = y - 2 - s * 1.4
            sc.disc(c, px, py, 1 + s // 6, 'white' if s < 15 else '#dfe9f2', 1.0 if s < 19 else 0.5)


_RING_ANG = np.radians(np.arange(0, 360, 2))
_B4 = bayer(4, 4)


def _ring(c, cx, cy, rx, ry, color, level=1.0):
    xs = np.round(cx + np.cos(_RING_ANG) * rx).astype(int)
    ys = np.round(cy + np.sin(_RING_ANG) * ry).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    if level < 1:
        m = _B4[ys % 4, xs % 4] < level
        xs, ys = xs[m], ys[m]
    c.a[ys, xs] = col(color)


def _stones_blaze(c, f, a, k, ox):
    """Ponhenge awakens: rune glows, rings of light rippling over the knoll, a pillar of light."""
    cx, cy = sc.KNOLL_C[0] - ox, sc.KNOLL_C[1]
    _pillar(c, cx, cy + 2, 9, 16, 'ice', 0.42 * k)
    fx.glow(c, cx, cy - 6, 30, 'turquoise', 0.35 * k, rings=4)
    for (x, y, _g) in sc.rune_spots():
        fx.glow(c, x - ox + 1, y + 2, 13, 'ice', 0.30 * k, rings=3)
    for j in range(3):
        s = a - j * 11
        if 0 <= s < 44:
            rx = 8 + s * 3.0
            lv = 1.0 if s < 22 else 0.5
            _ring(c, cx, cy + 4, rx, rx * 0.2, ('white', 'ice', 'turquoise')[min(2, s // 15)], lv)
            if s < 30:
                _ring(c, cx, cy + 4, rx - 2, (rx - 2) * 0.2, 'turquoise', 0.5)
    for i in range(16):                                   # sparks of light rising out of the circle
        s = a - (i % 8) * 3
        if 0 <= s < 40:
            x = cx - 34 + 68 * hash01(i, 31) + math.sin(s * 0.2 + i) * 2
            y = cy + 2 - s * 1.8 - 8 * hash01(i, 32)
            c.px(int(x), int(y), ('white', 'ice', 'turquoise')[i % 3])
            if s < 20:
                c.px(int(x), int(y) + 1, 'ice')


# ================================================================== Hollow Shades: the well's light
def _lantern_delay(lx):
    """Step after SPACE at which a lantern flares: the dead one you light first, then outward."""
    return 2 + abs(lx - LANTERNS[1]) // 5


def _well_surge(c, a, ox, oy):
    """The Well of Shade surges: a column of cold light up through the canopy, a bright core, motes."""
    k = _env(a, 10, 70, 110)
    x, y = sc.WELL_X - ox, sc.WELL_Y - oy - 16
    _pillar(c, x, y, 10, 24, 'crystal_lt', 0.40 * k)
    fx.glow(c, x, y, 48, 'crystal_lt', 0.30 * k, rings=5)
    fx.glow(c, x, y, 14, 'white', 0.55 * k, rings=3)
    for i in range(30):                                   # motes of light spiralling up
        s = a - i * 2
        if 0 <= s < 50:
            yy = y - 2 - s * 2.3
            xx = x + math.sin(s * 0.22 + i * 2.1) * (3 + s * 0.45)
            cc = ('white', 'ice', 'crystal_lt', 'butter')[i % 4]
            c.px(int(xx), int(yy), cc)
            if s < 26:
                c.px(int(xx) + 1, int(yy), cc)


def _lanterns_flare(c, f, a, ox, oy):
    """The village's cold lanterns flare warm gold, one after another, outward from the well."""
    k = _env(a, 1, 70, 110)
    for i, lx in enumerate(LANTERNS):
        s = a - _lantern_delay(lx)
        if s < 0:
            continue
        kk = min(1.0, s / 4) * k
        x, y = lx - ox, sc.GYC - oy
        c.rect(x + 3, y - 19, 3, 4, mix('#3a3a6a', 'butter', kk))
        c.px(x + 4, y - 18, mix('#3a3a6a', 'white', kk))
        fx.glow(c, x + 4, y - 17, 24, 'gold', 0.5 * kk, rings=4)
        fx.glow(c, x + 4, y - 17, 7, 'white', 0.6 * kk, rings=2)
        fx.glow(c, x + 4, y + 1, 16, 'butter', 0.22 * kk, rings=3)      # a warm pool on the ground
        if s < 14:
            fx.sparkle(c, x + 4, y - 17, s, 'white', period=14, size=3)


# ================================================================== shared helpers
_SHADOWS = ((-60, 92, 70, 16, 0.9), (150, 140, 90, 20, 1.1), (60, 60, 46, 12, 0.7))


def _cloud_shadows(c, f, mt_mask_screen, cam):
    """The film's sliding cloud shadows, each wrapping around off screen, anchored to the land."""
    m = np.zeros((H, W), bool)
    for (x0, y0, rx, ry, spd) in _SHADOWS:
        span = W + 2 * rx + 60
        cx = (x0 + (f - sc.CUT2) * spd - cam * 0.6 + rx) % span - rx
        X0, X1 = max(0, int(cx - rx)), min(W, int(cx + rx) + 1)
        Y0, Y1 = max(0, int(y0 - ry)), min(H, int(y0 + ry) + 1)
        if X1 <= X0 or Y1 <= Y0:
            continue
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        d = ((xx - cx) / rx) ** 2 + ((yy - y0) / ry) ** 2
        m[Y0:Y1, X0:X1] |= (d < 1) & (bayer(Y1 - Y0, X1 - X0, X0, Y0) < np.clip((1 - d) * 2.5, 0, 1))
    land = mt_mask_screen.copy()
    land[104:] = True
    sc.darken(c, m & land, (0.80, 0.85, 0.94))


def _add(c, ys, xs, color, strength):
    v = c.a[ys, xs].astype(np.float32) + np.array(col(color), np.float32) * strength
    c.a[ys, xs] = np.clip(v, 0, 255).astype(np.uint8)


@lru_cache(maxsize=512)
def _pillar_px(x, y_bot, w_bot, w_top):
    """Pixels of a dithered column of light from (x, y_bot) up to the top of the frame, densest at its
    foot and along its axis (screen px)."""
    ys, xs = [], []
    for y in range(0, min(H, y_bot)):
        t = y / max(1, y_bot)                             # 0 top .. 1 foot
        hw = w_top + (w_bot - w_top) * t
        x0, x1 = max(0, int(x - hw)), min(W, int(x + hw) + 1)
        if x1 <= x0:
            continue
        xx = np.arange(x0, x1)
        dens = (0.3 + 0.7 * t) * np.clip((1 - np.abs(xx - x) / max(1.0, hw)) * 1.8, 0, 1)
        xx = xx[bayer(1, x1 - x0, x0, y)[0] < dens]
        xs.append(xx)
        ys.append(np.full(len(xx), y))
    if not xs:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ys), np.concatenate(xs)


def _pillar(c, x, y_bot, w_bot, w_top, color, strength):
    if strength > 0:
        _add(c, *_pillar_px(int(x), int(y_bot), w_bot, w_top), color, strength)


@lru_cache(maxsize=256)
def _beam_px(top, bottom, ox, oy):
    """Pixels lit by engine.fx.light_beam(top, bottom) on a full frame (the same rows, spans and dither)."""
    (txl, ty), (txr, _) = top
    (bxl, by), (bxr, _) = bottom
    ys, xs = [], []
    for y in range(max(0, int(ty)), min(H, int(by) + 1)):
        t = seg(y, ty, by)
        xl, xr = int(lerp(txl, bxl, t)), int(lerp(txr, bxr, t))
        if xr < xl:
            xl, xr = xr, xl
        x0, x1 = max(0, xl), min(W, xr + 1)
        if x1 <= x0:
            continue
        m = bayer(1, x1 - x0, x0 + ox, y + oy)[0] < (1 - t * 0.7)
        xx = np.arange(x0, x1)[m]
        xs.append(xx)
        ys.append(np.full(len(xx), y))
    if not xs:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ys), np.concatenate(xs)


def _light_beam(c, top, bottom, color, strength=0.25, ox=0, oy=0):
    """engine.fx.light_beam with the beam's pixel set cached (the film's version loops over rows)."""
    _add(c, *_beam_px(tuple(map(tuple, top)), tuple(map(tuple, bottom)), ox, oy), color, strength)


class _FX:
    """engine.fx for the Hollow Shades renderer, with the cached light beam scaled by the daylight k (no
    god-rays at night)."""
    def __init__(self, k=1.0):
        self.k = k

    def light_beam(self, c, top, bottom, color, strength=0.25, ox=0, oy=0):
        if self.k > 0.02:
            _light_beam(c, top, bottom, color, strength * min(1.0, self.k), ox, oy)

    def __getattr__(self, name):
        return getattr(fx, name)


# ================================================================== sky hooks (game/daylight.py)
# Each reproduces only the sky its shot renderer draws first (the cached gradient and the sun's glow
# off the top-left corner); Hollow Shades gives only the bright canopy gap of its forest backdrop (the
# violet depth around it is the forest, not sky), with the god-rays and the vignette drawn over it.
_NOT_SKY = np.array((255, 0, 255), np.uint8)       # a key no painted pixel is near


def _sky_falls(f, cx, cy):
    return _covered('falls', int(cx)).copy()


def _sky_foal(f, cx, cy):
    return _covered('foal', int(cx)).copy()


@lru_cache(maxsize=8)
def _covered(shot, cam):
    """The shot's sky render with its solid scenery layers keyed out (_NOT_SKY): pale scenery pixels close
    to the sky's colours (the tunnel portal's top, the far hills' haze, sunlit ridges) must not be
    repainted as sky. Per camera (layers at the shot's parallaxes, sc.comp's offsets)."""
    if shot == 'falls':
        c = sc._sky_a().copy()
        lays = ((sc._far_a(), int(cam * 0.15)), (sc._main_a(), cam), (sc._fg_a(), int(cam * 1.35)))
    else:
        c = sc._sky_b().copy()
        lays = ((sc._ridge_b(), int(cam * 0.15)), (sc._mountain_b(), int(cam * 0.3)),
                (sc._mid_b(), int(cam * 0.55)), (sc._knoll_b(), cam), (sc._fg_b(), int(cam * 1.5)))
    hit = np.zeros((H, W), bool)
    for (_a, m), ox in lays:
        x0, x1 = max(0, ox), min(m.shape[1], ox + W)
        y1 = min(m.shape[0], H)
        if x1 > x0:
            hit[:y1, x0 - ox:x1 - ox] |= m[:y1, x0:x1]
    c.a[hit] = _NOT_SKY
    return c


@lru_cache(maxsize=1)
def _gap_c():
    """The forest backdrop with everything but the bright canopy gap keyed out (sc._sky_c's q > 0)."""
    a = sc._sky_c().a.copy()
    yy, xx = np.mgrid[0:sc.HC, 0:W]
    d = ((xx - 196) / 78.0) ** 2 + ((yy + 6) / 62.0) ** 2
    q = np.clip(np.floor(np.clip((1 - d) * 4, 0, 4) + (bayer(sc.HC, W) - 0.5) * 0.9), 0, 4)
    a[q <= 0] = _NOT_SKY
    return a


GAP_ROWS = 60                                      # the gap ends above layer row 56


@lru_cache(maxsize=1)
def _vig_top():
    return sc._vig_c()[:GAP_ROWS]


def _sky_shades(f, cx, cy, day=1.0):
    c = sc.Canvas()
    oy = int(cy * 0.2)
    c.a[:] = _gap_c()[oy:oy + H]
    for i, (tx, bx, wd) in enumerate(((150, 196, 7), (176, 238, 9), (202, 276, 6))):
        s_ = (0.09 + 0.04 * math.sin(f * 0.06 + i * 2)) * min(1.0, day)
        ox = int(cx * 0.8)
        if day > 0.02:
            _light_beam(c, ((tx - ox, 0), (tx + wd - ox, 0)), ((bx - ox - 6, H), (bx + wd + 6 - ox, H)),
                    '#c8d0ff', s_)
    top = c.a[:GAP_ROWS]                           # the render's vignette (only the gap's rows matter)
    top[:] = (top.astype(np.float32) * _vig_top()).astype(np.uint8)
    c.a[GAP_ROWS:] = _NOT_SKY
    return c
