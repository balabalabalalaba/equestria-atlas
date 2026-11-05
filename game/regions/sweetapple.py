"""Sweet Apple Acres as an explorable diorama (art: scenes/s04_sweetapple.py, used read-only).

The film's render(f) switches content by shot (orchard and farm before scene frame 216, the zap-apple
grove after), so one scene frame can never show the whole farm. This region instead re-composes the
scene's own cached layers and animation helpers along one continuous strip, in the film's layer order,
with every helper active wherever it is on screen. Timed beats are replayed on game clocks: the
beat-apples drop into their baskets again each time you come back, and the little storm over the
zap-apple trees strikes every few seconds.

Things to set off with SPACE: shake the orchard tree (its apples drop into the basket again), spin the
cider press, send a gust through the windmill, and call the zap-apple storm's lightning. Three atlas
stamps are hidden: in the barn's hay loft, high above the clubhouse tree, and over the stream by the
cottage.
World clock: the whole sky is the clock's at every hour (the film's low morning sun never stays up off
the arc), under the Everfree's gloom on the right; the sunlit dust drifts under the clock's sun. The lights
are the farm's own (auto_glow off): lit windows, the round windows, the door lantern, Canterlot's windows far
off, and by night the zap apples, sparkles and bolts.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, fx, col, mix, hash01
import scenes.s04_sweetapple as sa

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

CAM_MAX = 768           # the back orchard (p 0.7, 860 px) and foreground (p 1.3, 1320 px) end at ~770
APPLE_END = 189         # the film's beat-apple run: drops on beats 72..162, all landed by 169
ZAP_PERIOD = 240        # storm cycle: big strike + crackles (film frames 216..282), then sparkles only
ZAP_X = sa.STORM[0]
HOUSE_X = sa.HOUSE_X + sa.HOUSE_W // 2

_zx, _zcy, _zr = sa.zap_canopy(2)
LANDMARKS = [           # the film's callouts, in world px
    Landmark('SWEET APPLE ACRES', sa.SIGN[0], (sa.SIGN[0] + 8, sa.GY - 28, 'up', 44)),
    Landmark('BIG RED BARN', sa.BARN_X + sa.BARN_FW // 2 + 10, (sa.BARN_X + sa.BARN_FW + 20, sa.GY - 68, 'up', 12)),
    Landmark('CLUBHOUSE', sa.CLUB_X, (sa.CLUB_X, sa.CLUB_PY - 19 - 12, 'up', 22)),
    Landmark('ZAP APPLE TREES', _zx, (_zx, _zcy + 4, 'down', 30)),
    Landmark("FLUTTERSHY'S COTTAGE", sa.COTTAGE_X + 25, (sa.COTTAGE_X + 25, sa.GY - 24 - 14, 'up', 22)),
]


HOTSPOTS = [
    Hotspot('apples', sa.BEAT_TREE[0], 108, 'apple', cooldown=150),
    Hotspot('press', sa.PRESS_X + 13, 100, 'gear', cooldown=90),
    Hotspot('mill', sa.MILL_X, sa.GY - 57, 'wind', cooldown=90),
    Hotspot('zap', sa.STORM[0], sa.STORM[1] + 4, 'bolt', cooldown=60),
]
STAMPS = [Stamp('hayloft', sa.BARN_X + 36, sa.GY - 58), Stamp('treetop', sa.CLUB_X + 6, 40),
          Stamp('stream', sa.STREAM_X, 104)]
LEAVES = ('leaf', 'pale_leaf', 'green', 'gold')
SPIN = 90               # steps of a spin-up: the press / windmill turn fast, then ease back


def _spin(a, extra):
    """Extra scene frames after `a` steps of a spin-up adding `extra` frames in total (smooth, ends flat)."""
    a = min(max(a, 0), SPIN)
    return extra / 2 * (1 - math.cos(math.pi * a / SPIN))


@lru_cache(maxsize=64)
def _sky_base(cam):
    c = sa._sky().copy()
    if cam + W > 870:
        sa.everfree_dark(c, 300, cam)                    # the render's wash over the right edge
    return c


_NOT_SKY = np.array((255, 0, 255), np.uint8)             # a key no painted pixel is near


def _sky_farm(f, cam):
    """The farm's sky alone at camera cam, with the scenery layers and the clouds keyed out (the barn
    doors, the cottage windows and the palest cloud are within the sky test's tolerance of the sky)."""
    c = _sky_base(cam if cam + W > 870 else 0).copy()
    occ = np.zeros((sa.H, W), bool)
    for layer, p, dx in ((sa._mountains(), sa.MT_P, 40), (sa._farhills(), sa.FH_P, 0),
                         (sa._midhills(), sa.MH_P, 0), (sa._backorchard(), sa.BO_P, 0), (sa._main(), 1, 0)):
        ox = int(round(cam * p)) + dx
        m = layer[1]
        x0, x1 = max(0, ox), min(m.shape[1], ox + W)
        if x1 > x0:
            occ[:, x0 - ox:x1 - ox] |= m[:, x0:x1]
    span = W + 140                                       # the drifting clouds (as sa.draw_clouds)
    for (x, y, w, sd, p, v) in sa.CLOUDS:
        al = sa._cloud(w, sd)[..., 3] >= 128
        sx = (int(x + v * f - cam * p) + 70) % span - 70
        lh, lw = al.shape
        x0, x1, y1 = max(0, sx), min(W, sx + lw), min(sa.H, y + lh)
        if x1 > x0:
            occ[y:y1, x0:x1] |= al[:y1 - y, x0 - sx:x1 - sx]
    np.copyto(c.a, _NOT_SKY, where=occ[..., None])
    return c


# ------------------------------------------------------------------ night lights (world clock)
# The farm keeps its REAL lights itself (auto_glow is off: the relight would keep the sunlit cream path
# edge, the hay, the baskets, apples and flowers glowing like lamps): the lit windows (game/nightwin.py,
# but not the stream seen between the footbridge's balusters), Fluttershy's round windows and the
# clubhouse's (whole windows, not quadrant by quadrant), the farmhouse's door lantern, Canterlot's windows
# on the far range, the zap apples, their sparkles and the storm's bolts.
BRIDGE = (sa.STREAM_X - 22, 128, sa.STREAM_X + 22, 162)     # main-layer px: no window panes in here
_ROUND = (                                                  # (centre x, y, glass radius): main layer
    (sa.COTTAGE_X + 10, sa.GY - 14, 4), (sa.COTTAGE_X + 39, sa.GY - 14, 4),     # sa.cottage
    (sa.COTTAGE_X + 25, sa.GY - 34, 2),                                         # its attic window
    (sa.CLUB_X + 5, sa.CLUB_PY - 13, 2),                                        # sa.clubhouse
)
NOT_PANES = [BRIDGE] + [(x - r - 1, y - r - 1, x + r + 2, y + r + 2) for x, y, r in _ROUND]   # (own code)
LANTERN = (426, 133, 3, 4)          # the farmhouse's brass door lantern (sa.farmhouse: dx0 + 13, fl + 10)
_GLOW_LANTERN = np.array([col('#ffd466'), col('#fff3c0')], np.uint8)


@lru_cache(maxsize=1)
def _round_px():
    """(ys, xs, painted rgb, threshold, shade) of the round windows' glass on the main layer (the pale
    muntin cross and frame are not glass): one threshold per window, so a window lights all at once."""
    a, m = sa._main()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    rgb = a.astype(np.int16)
    glass = (rgb[..., 2] > rgb[..., 0] + 20) & (m > 0)
    ys, xs, v, kd = [], [], [], []
    for i, (x, y, r) in enumerate(_ROUND):
        sel = glass & ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
        py, px = np.nonzero(sel)
        ys.append(py)
        xs.append(px)
        v.append(np.full(len(py), 0.1 + 0.6 * hash01(i, 77), np.float32))
        kd.append(np.full(len(py), i % 3))
    ys, xs = np.concatenate(ys), np.concatenate(xs)
    return ys, xs, a[ys, xs].copy(), np.concatenate(v), np.concatenate(kd)


# Canterlot on the far range (sa._mountains, as it draws them): wall blocks (dx, h, w) and towers (dx, h, w)
_CANT_BLOCKS = ((-28, 3, 6), (-21, 5, 5), (-15, 4, 4), (-10, 6, 6), (4, 5, 6), (11, 4, 5), (17, 6, 6), (23, 3, 4))
_CANT_TOWERS = ((-24, 7, 2), (-17, 11, 3), (-11, 8, 2), (-6, 15, 3), (-1, 22, 4), (5, 13, 3), (10, 9, 2),
                (15, 12, 3), (21, 7, 2))


@lru_cache(maxsize=1)
def _castle_px():
    """(ys, xs, painted rgb, threshold, shade) of Canterlot's 1-px windows on the far range (layer px): a
    column up each white tower and a row along the wall blocks, where the wall is still seen in the layer;
    three in four of them go on one by one through the dusk."""
    a, m = sa._mountains()
    x0, y0 = sa.CANT_X, sa.CANT_Y
    pts = [(x0 + dx + (tw - 1) // 2, y) for dx, th, tw in _CANT_TOWERS for y in range(y0 - th + 3, y0 - 2, 4)]
    pts += [(x0 + dx + 1 + 2 * k, y0 - 1) for dx, _h, w_ in _CANT_BLOCKS for k in range((w_ - 1) // 2)]
    wall = {tuple(col(v)) for v in ('#fbf8ff', '#cfc6ee', sa.hz('royal', 0.2))}
    pts = [(x, y) for x, y in sorted(set(pts)) if m[y, x] and tuple(a[y, x]) in wall]
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    v = np.array([0.05 + hash01(x, y, 91) / 0.75 for x, y in pts], np.float32)
    return ys, xs, a[ys, xs].copy(), v, np.array([int(hash01(x, y, 92) * 3) for x, y in pts])


def _lights_on(c, hour, pts, ox):
    """Light a set of window pixels (layer px, the layer drawn at offset ox; seen ones only, nothing in
    front) once the dusk ramp passes each one's threshold. Returns the mask of the lit glass (None by day)."""
    k = nightwin.ramp(hour)
    if k <= 0:
        return None
    ys, xs, src, v, kd = pts
    sx = xs - ox
    ok = (v < k) & (sx >= 0) & (sx < W)
    ys, sx, src, kd = ys[ok], sx[ok], src[ok], kd[ok]
    seen = np.all(c.a[ys, sx] == src, axis=1)
    if not seen.any():
        return None
    c.a[ys[seen], sx[seen]] = nightwin.WARM[kd[seen]]
    m = np.zeros((H, W), bool)
    m[ys[seen], sx[seen]] = True
    return m


def _round_windows(c, hour, cam):
    """The round windows, lit whole (one threshold per window)."""
    return _lights_on(c, hour, _round_px(), cam)


def _door_lantern(c, hour, cam):
    """The farmhouse's door lantern: lit from early in the dusk (its glass, a hot core). Its mask."""
    x, y, w, h = LANTERN
    sx = x - cam
    if nightwin.ramp(hour) <= 0.02 or not -w < sx < W:
        return None
    a, _m = sa._main()
    src = a[y:y + h, x:x + w]
    x0, x1 = max(0, sx), min(W, sx + w)
    sub = c.a[y:y + h, x0:x1]
    seen = np.all(sub == src[:, x0 - sx:x1 - sx], axis=2)
    sub[seen] = _GLOW_LANTERN[0]
    if x0 <= sx + 1 < x1 and seen[1, sx + 1 - x0]:
        sub[1, sx + 1 - x0] = _GLOW_LANTERN[1]
    m = np.zeros((H, W), bool)
    m[y:y + h, x0:x1] = seen
    return m


def _gloom(cam):
    """sa.everfree_dark's wash at the film's end (scene frame 300): its amount per screen column."""
    xs = np.arange(W, dtype=np.float32) + cam
    return 0.62 * np.clip((xs - 870) / 190.0, 0, 1) ** 1.2


class _FxNoGlow:
    """The fx module without its soft glow: the zap strike's halo would be kept at painted strength by
    night (a dithered bright disc over the canopy); the bolts, arcs and sparkles are the light."""
    def __getattr__(self, name):
        return _nop if name == 'glow' else getattr(fx, name)


def _nop(*_a, **_k):
    return None


_ZAP_NIGHT = rebind(sa.zap_fx, fx=_FxNoGlow())


@lru_cache(maxsize=1)
def _zap_apple_px():
    """(ys, xs, rgb) of the striped zap apples in the farm's main layer (their exact colours, inside the
    zap-apple canopies): the only part of the grove kept lit by night."""
    a, m = sa._main()
    cols = []
    for s_ in list(sa.ZAP_STRIPES) + ['violet']:
        for cc in (s_, mix(s_, 'purple', 0.4)):
            cols += [col(cc), col(mix(cc, 'navy', 0.3))]
    cols.append(col('white'))
    cols = np.array(cols, np.int32)
    key = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    hit = np.isin(key, (cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]) & (m > 0)
    near = np.zeros(hit.shape, bool)
    yy, xx = np.mgrid[0:hit.shape[0], 0:hit.shape[1]]
    for i in range(len(sa.ZAP)):
        x, cy, r = sa.zap_canopy(i)
        near |= (xx - x) ** 2 + (yy - cy) ** 2 <= (r + 2) ** 2
    ys, xs = np.nonzero(hit & near)
    return ys, xs, a[ys, xs].copy()


def _zap_apples(c, cam):
    """The zap apples still seen at camera `cam` (nothing in front of them): a mask, kept lit by night."""
    ys, xs, rgb = _zap_apple_px()
    sx = xs - cam
    ok = (sx >= 0) & (sx < W) & (ys < H)
    ys, sx, rgb = ys[ok], sx[ok], rgb[ok]
    seen = np.all(c.a[ys, sx] == rgb, axis=1)
    m = np.zeros((H, W), bool)
    m[ys[seen], sx[seen]] = True
    return m


class SweetAppleAcres(GlowKeep, Region):
    SEG = 'sweetapple'
    CARD = ('APPLE FARM', ['ORCHARDS - BARN - CIDER', 'SOUTH OF PONYVILLE'])

    def build_areas(self):
        return [Area(self.render_farm, cam_x=(0, CAM_MAX), start=(80, 70), landmarks=LANDMARKS,
                     hotspots=HOTSPOTS, stamps=STAMPS, sky=self.sky_farm)]

    def auto_glow(self):
        """No painted spot glows by itself here (the cream path edge, hay, baskets, apples): the farm's
        real lights are kept by mask (render_farm)."""
        return False

    def after_light(self, c, t):
        """After the relight: the clock's sky over the whole open sky at every hour, under the Everfree's
        gloom, then the kept lights (GlowKeep). The base keeps the painted morning sky while the clock's
        sky looks alike (~6 to ~11.7 h), with the film's low sun on the far left and its haze, off the arc
        the clock's sun follows (it jumps at ~11.7 h); and a repainted sky lost the gloom the film washes
        over the sky by the Everfree (a bright band over the dark trees from ~10.5 h, gone before)."""
        raw, self._raw = getattr(self, '_raw', None), None
        if raw is not None:
            a, f, ix, iy = raw
            key = (self.ai, t, ix, iy)                   # the base's memo: once per step
            if self._sky_memo is None or self._sky_memo[0] != key:
                self._sky_memo = (key, daylight.sky_pixels(a, self.sky_farm(f, ix, iy).a))
            mask = self._sky_memo[1]
            hour = self.g.hour
            if mask.any():
                if daylight.sky_weight(self.HOUR, hour) < 0.999:
                    y0, y1 = self.area.cam_y
                    daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, y1 - y0 + 130, t)
                g = _gloom(ix)
                cols = np.nonzero(g > 0)[0]
                if len(cols):
                    x0 = cols[0]
                    sub = c.a[:, x0:]
                    sm = mask[:, x0:]
                    k = np.broadcast_to(g[x0:][None, :], sm.shape)[sm][:, None]
                    tint = np.array(col('#241c38'), np.float32)
                    px = sub[sm].astype(np.float32)
                    sub[sm] = np.clip(px * (1 - k) + tint * (k * 0.85), 0, 255).astype(np.uint8)
        super().after_light(c, t)

    def sky_farm(self, f, cx, cy):
        """Sky hook (game/daylight.py): the scene's cached sky (gradient, morning sun, its glows and
        rays), screen-fixed as render_farm draws it first, under the Everfree edge's darkening."""
        return _sky_farm(f, int(cx))

    def init_state(self):
        self.apple_t0 = None        # step the beat-apple run (scene frame 72) started
        self.zap_t0 = 0             # step of the storm's last big strike (scene frame 216)
        self.spins = {'press': (0.0, -10 ** 6), 'mill': (0.0, -10 ** 6)}   # key -> (frames banked, start)

    def spin_f(self, key, t):
        """Extra scene frames of the press / windmill: everything banked + the current spin-up."""
        banked, t0 = self.spins[key]
        return banked + _spin(t - t0, 360 if key == 'press' else 240)

    def on_enter(self):
        self.apple_t0 = self.g.t + 20

    def apple_f(self, t):
        if self.apple_t0 is None or t < self.apple_t0:
            return 72
        return min(APPLE_END, 72 + t - self.apple_t0)

    def zap_f(self, t):
        return 216 + (t - self.zap_t0) % ZAP_PERIOD

    def on_interact(self, key, t):
        g = self.g
        if key == 'apples':
            self.apple_t0 = t + 4
            g.sfx('creak', 0.3)
        elif key in ('press', 'mill'):
            self.spins[key] = (self.spin_f(key, t), t)
            g.sfx('pour' if key == 'press' else 'wind', 0.4, None if key == 'press' else 1)
        elif key == 'zap':
            self.zap_t0 = t

    def update(self, t):
        g = self.g
        fa = self.apple_f(t)
        if fa == APPLE_END and self.cx > 350:            # both baskets off screen: refill quietly
            self.apple_t0 = t + 1
        for bwx, beats, _hangs in sa.BEAT_BASKETS:
            if fa - 7 in beats and self.near_gain(bwx) > 0.3:
                g.sfx('pop', 0.3 * self.near_gain(bwx), (fa // 18) % 4)
        fz = self.zap_f(t)
        if fz == 216:
            g.sfx('zap', 0.7 * self.near_gain(ZAP_X))
        elif fz in (234, 252, 270):
            g.sfx('zap', 0.3 * self.near_gain(ZAP_X))
        if t % 72 == 0 and abs(HOUSE_X - (self.cx + W / 2)) < 160:
            g.sfx('creak', 0.18)                     # the rocking chair on the porch
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)

    def render_farm(self, f, cx, cy):
        t = self.g.t
        cam = cx
        c = sa._sky().copy()
        sa.draw_clouds(c, f, cam)
        mo = int(round(cam * sa.MT_P)) + 40
        sa.comp(c, sa._mountains(), mo)
        sa.canterlot_falls(c, mo, f)
        sa.comp(c, sa._farhills(), int(round(cam * sa.FH_P)))
        sa.comp(c, sa._midhills(), int(round(cam * sa.MH_P)))
        sa.comp(c, sa._backorchard(), int(round(cam * sa.BO_P)))
        sa.comp(c, sa._main(), cam)
        # orchard and farm (the film's shot A helpers)
        sa.beat_apples(c, self.apple_f(t), cam)
        px_, py_ = sa.CLUB_X - cam, sa.CLUB_PY - 19 - 13 - 13
        if -20 < px_ < W + 20:
            sa.pennant(c, px_, py_, f)
            sa.laundry(c, f, cam)
        hx = sa.HOUSE_X - cam
        if -100 < hx < W:
            sa.smoke(c, hx + 61, int(sa.GY - 52 - 22 - 2.5) - 10, f, seed=1)
            sa.chair_rail(c, hx, f)
        if -40 < sa.PRESS_X - cam < W:
            sa.press_dynamic(c, sa.PRESS_X - cam, f + int(self.spin_f('press', t)))
        if -40 < sa.MILL_X - cam - 16 < W:
            sa.windmill_dynamic(c, sa.MILL_X - cam, f + int(self.spin_f('mill', t)))
        # grove, stream and cottage (shot B helpers)
        sa.stream_dynamic(c, f, cam)
        if -40 < sa.COTTAGE_X - cam < W + 60:
            sa.smoke(c, sa.COTTAGE_X + 50 - 4 - cam, sa.GY - 24 + 2 - 28, f, seed=2, n=4, height=18)
        fo = int(round(cam * sa.FG_P))
        sa.comp(c, sa._foreground(), fo)
        sa.grass_sway(c, f, fo)
        sa.falling_leaves(c, f)
        hour = self.g.hour
        night = daylight.night(hour)
        kind, sun_x, _y = daylight._body_pos(hour)
        if cam < 480 and night < 0.25 and kind == 'sun':     # sunlit dust, under the clock's sun (the
            x0 = min(W - 170, max(0, sun_x - 85))             # film's are under its low morning sun)
            fx.dust_motes(c, f, (x0, 60, 170, 90), n=14, color='butter', seed=4)
        # lit windows (the house, barn, clubhouse, cottage, Canterlot far off) and the door lantern: painted
        # before the Everfree wash, which tints the cottage's colours away from its layer's (nightwin's test)
        lit = [nightwin.paint(c, hour, t, (('sa_main', nightwin.from_mask(sa._main), -cam, 0,
                                            {'exclude': NOT_PANES}),)),
               _round_windows(c, hour, cam), _door_lantern(c, hour, cam), _lights_on(c, hour, _castle_px(), mo)]
        lit = [m for m in lit if m is not None]
        sa.everfree_dark(c, 300, cam)                    # the Everfree edge, at its film-end darkness
        washed = c.a.copy() if lit else None
        zap = None
        if night > 0.1:
            apples = _zap_apples(c, cam)                 # by night the zap apples (only) keep their colour
            before = c.a.copy()
            _ZAP_NIGHT(c, self.zap_f(t), cam)            # ... and the sparkles and bolts (no soft halo)
            zap = np.any(c.a != before, axis=2) | apples
            zap_rgb = c.a.copy()
        else:
            sa.zap_fx(c, self.zap_f(t), cam)
        self._draw_events(c, t, cam)
        m = None
        for w in lit:                                    # (nothing drawn over them since)
            w &= np.all(c.a == washed, axis=2)
            m = w if m is None else m | w
        if zap is not None:
            zap &= np.all(c.a == zap_rgb, axis=2)
            m = zap if m is None else m | zap
        self.keep_glow(c, m, 0.95)
        self._raw = (c.a.copy(), f, cx, cy)              # for the sky repaint (after_light)
        return c

    def _draw_events(self, c, t, cam):
        """Visible results of the SPACE interactions (world-anchored; reads state only)."""
        t0 = self.spot_t.get('apples')
        if t0 is not None:                                  # the tree shakes: leaves fly
            ui.burst(c, t, t0, sa.BEAT_TREE[0] - cam, 104, LEAVES, n=36, seed=3, power=0.9, life=56)
        t0 = self.spot_t.get('press')
        if t0 is not None:                                  # cider splashes from the spout
            ui.burst(c, t, t0 + 6, sa.PRESS_X + 29 - cam, 134, ('amber', 'gold', 'butter', 'white'), n=28,
                     seed=4, power=0.9, life=44, gravity=0.07)
            ui.burst(c, t, t0, sa.PRESS_X + 13 - cam, 98, ('crema', 'latte', 'white'), n=14, seed=6, power=0.8,
                     life=30, gravity=0.02)
        t0 = self.spot_t.get('mill')
        if t0 is not None:                                  # the gust carries leaves past the wheel
            for k in range(3):
                ui.burst(c, t, t0 + k * 8, sa.MILL_X - 34 - cam, 96 - k * 8, LEAVES, n=18, seed=20 + k,
                         power=1.6, life=56, gravity=-0.01)

