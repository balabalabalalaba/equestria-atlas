"""Mount Aris and Seaquestria as an explorable diorama (art: scenes/s12_mountaris.py, used read-only).

The film shows four different views (dissolves between them), so the region has four areas, flown
through left to right with a cloud sweep between them:
  0  THE SEA CRAG (shot A): the crag of Mount Aris with its harbour and stairway, the sailing ship,
     the Storm King's black castle under its lightning cloud on the far left, the basalt columns of
     Basalt Beach in the foreground.
  1  THE SUMMIT (shot B): Hippogriffia's restored palace and the golden Bud Dome, a vertical
     panorama from the towers down the cliff stairway toward the sea (the film's closing swoop).
  2  BELOW THE WAVES (shot C) and
  3  SEAQUESTRIA (shot D): a hot-air balloon can't dive, so these are waterline cross-sections: the
     balloon floats on the sea (it can climb into the sky above), and the view is cut at the surface,
     sky above, the underwater world below. Descending onto the water tilts the camera down into the
     sea; the film's own dive (shot C's depth parallax, surface sweeping up the screen) is replayed by
     the player's height: high up you see the crag standing in the sea, at the waterline the whole
     kelp forest down to the pearl clams. The balloon's refracted image shows under the surface when
     it sits on the water (the film's balloon reflection, following the player's balloon), with
     ripples round the basket and a soft splash as it touches down.
Every scene helper runs on the free scene clock. The film's own balloons (drifting by the crag,
moored at the tower with its rope, reflected under the waves) are never drawn; the player's is.
Timed beats on game clocks: the storm's three lightning bolts (thunder after each) every 10 s; the
ship sailing out of the harbour (the film's start), then tacking to and fro in front of the crag
(slowing and turning at each end, never hidden behind the basalt); the spray bursts between the
basalt columns on every beat.

Things to set off with SPACE: call the lightning over the Storm King's realm, sound the ship's horn
(a gust fills its sails and it surges ahead), flash the Bud Dome, make the dive splash at the
waterline, and light the pearl on Seaquestria's conch spire. Three atlas stamps are hidden: low
among the basalt columns at the far end of the beach, high by the dome's finial, and up in the sky
above Seaquestria.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, hash01, clamp, mix, col, shade, lerp
from engine.core import bayer
import scenes.s12_mountaris as sc

from .. import ui, nightwin, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

# ----------------------------------------------------------------------------- area 0: the sea crag
CAM_A = 90              # basalt foreground (440 px from x -30, p 1) ends at cam 90; far isles at 266
STORM_P = 300           # storm cycle: bolts at storm frames 94, 130, 166, flickers 82/112/148, then quiet
STORM_X = 65            # storm tag / hotspot anchor x on the far layer (p 0.12)
SHIP_X0, SHIP_LEG = 30, 170    # the ship tacks to and fro in front of the crag (x on its p 0.7 layer)
LEG_T = 850             # steps per leg (~0.2 px per step, the film's speed); it slows, turns, sails back
LEG_PHASE = 0.425       # landing: the ship just out of the harbour (x ~96, the film's start), heading east
SURGE, SURGE_T = 100, 260   # SPACE: the gust adds SURGE_T steps of sailing over SURGE steps
CLOUDS_A = ((-10, 14, 84, 26, 3, 0.04), (104, 30, 60, 20, 5, 0.05), (170, 4, 100, 30, 7, 0.06),
            (300, 36, 50, 16, 9, 0.05))
_jx, _jw, _jt, _ = sc.COL_BACK[7]

# ----------------------------------------------------------------------------- area 1: the summit
CAM_BX, CAM_BY = 60, 100    # palace 380 x 280 (p 1), balustrade 420 px (p 1.25, -20) ends at cam 64
CLOUDS_B = ((-6, 30, 90, 28, 13), (120, 14, 70, 22, 15), (236, 50, 70, 20, 17))
DOME_HOT = (sc.DOME_X, sc.DOME_BASE - 38)

# ----------------------------------------------------------------------------- areas 2, 3: the waterline
S_LOW = 24              # screen y of the water surface with the camera at its lowest (world y of the sea)
BY_SEA = S_LOW + 1      # basket bottom resting on the water
# area 2 (shot C): the camera tilts K_C px; at the top the film's dive starts (surface at y 104, D=0),
# at the bottom its end (D=104) with the surface held at S_LOW
K_C = 80
BY_TOP_C = S_LOW + 24 - K_C         # highest basket: balloon at screen y ~48 with the camera at the top
CAM_C = 90                          # near reef 440 px from x -30 (p 1) ends at cam 90
SPLASH_X = 236
# area 3 (shot D): the palace (p 0.65 in the film) is the main layer here (p 1 both ways), so its tag
# and hotspot can sit on it while the camera tilts; the near reef runs at p 1.4, the far city at 0.46
K_D = 60
BY_TOP_D = S_LOW + 20 - K_D
CAM_D = 47              # far city 350 px (-8, p .46) ends at 47.8; palace 380 (-10) 50; reef 410 (-20, 1.4) 50
MID_Y = 15              # palace layer y (film 0): the conch's pearl stays just under the surface
FAR_Y = 4
NEAR_P = 1.4
FAR_P = 0.46
PEARL = (sc.CONCH_X - 10, sc.CONCH_TOP - 3 + MID_Y)
SPLIT = {2: (BY_TOP_C, BY_SEA, K_C), 3: (BY_TOP_D, BY_SEA, K_D)}

LANDMARKS_A = [
    Landmark("STORM KING'S REALM", 96, (STORM_X, sc.STORM_OY + sc.SBASE - 40, 'down', 30), p=sc.P_FAR),
    Landmark('BASALT BEACH', _jx + _jw // 2, (_jx + _jw // 2, _jt + 1, 'up', 12)),
]
LANDMARKS_B = [
    Landmark('MOUNT ARIS', sc.DOME_X, (sc.DOME_X + sc.DOME_R - 2, sc.DOME_BASE - 30, 'right', 14), y=60),
]
LANDMARKS_D = [
    Landmark('SEAQUESTRIA', sc.CONCH_X - 10, (sc.CONCH_X - 24, 70 + MID_Y, 'left', 14)),
]
STAMPS_A = [Stamp('basalt', 392, 116)]
STAMPS_B = [Stamp('finial', sc.DOME_X + 16, 20)]
STAMPS_D = [Stamp('skyhigh', 330, BY_TOP_D - 13)]


def _wrap(sx, w):
    """A drifting sprite's screen x, wrapped round so the sky never empties (off screen both ends)."""
    span = W + w + 16
    return int(round((sx + w + 8) % span - (w + 8)))


def _ease_extra(a, n, extra):
    """Extra distance after `a` steps of an eased surge adding `extra` over n steps (smooth, ends flat)."""
    a = min(max(a, 0), n)
    return extra / 2 * (1 - math.cos(math.pi * a / n))


# ----------------------------------------------------------------------------- the world clock's lights
# The film painted the place at 16:38 with its lanterns, cave mouths, windows and the sea city's lights
# glowing. In the game each light comes on at its own point of the windows' night ramp (nightwin.ramp:
# on through 19:20 .. 20:00, off again through 05:30 .. 06:20; nothing lit by day) and its small halo
# grows after it; the relight keeps no painted spots by itself (auto_glow False: sunlit rims and cloud
# edges would glow), the lights' own pixels are kept by mask (GlowKeep.keep_glow).
LAMP_UNLIT = '#5a4c5a'          # a lantern by day: dark glass
CAVE_UNLIT = '#3c2e3a'          # a cave mouth with no fire in it
GLASS_UNLIT = '#2c5a72'         # Seaquestria's cyan window glass by day
DOOR_UNLIT = '#9a4a86'          # ...and the pearl dome's pink door
LIGHT_FADE = 0.25               # ramp span over which a light's halo grows once it is on


def light_k(hour):
    """0 (day) .. 1 (night): the lights' ramp at a clock hour."""
    return nightwin.ramp(hour)


def lamp_thr(*key):
    """A light's own point on the ramp (0.04 .. 0.6)."""
    return 0.04 + 0.56 * hash01(*key, 77)


def lamp_fade(k, thr):
    """0 (off) .. 1: a light with threshold thr at ramp k (its halo grows over LIGHT_FADE)."""
    return clamp((k - thr) / LIGHT_FADE) if k > thr else 0.0


@lru_cache(maxsize=8)
def _glitter_c(q):
    """sc.C for the crag's sea at warmth q (warm_k): the film's sunset-orange glitter troughs and gold crests
    paled towards a white-blue daylight glint as the sun climbs (orange at noon read as golden hour)."""
    day = {'#e89a50': '#cfe6f2', 'butter': '#fffbe8'}

    def C(name):
        if name in day:
            return sc.C(mix(day[name], name, q))
        return sc.C(name)
    return C


def warm_k(hour):
    """0..1 (quarter steps): how much of the low sun's gold rim the clouds show (the golden hour and
    sunset, and the sunrise); none by day or at night."""
    h = hour % 24
    eve = clamp((h - 15.8) / 0.8) * (1 - clamp((h - 19.0) / 0.6))
    morn = clamp((h - 5.3) / 0.5) * (1 - clamp((h - 7.0) / 0.8))
    return round(max(eve, morn) * 4) / 4


_CLOUDS = {}


def cloud(w, h, seed, q):
    """sc._cloud with its gold sun rim q (0..1) of the way on (warm_k)."""
    key = (w, h, seed, q)
    if key not in _CLOUDS:
        if q >= 1:
            _CLOUDS[key] = sc._cloud(w, h, seed)
        else:
            a, b = sc._cloud(w, h, seed), sc._cloud(w, h, seed, 0)
            out = b.copy()
            out[..., :3] = (b[..., :3] * (1 - q) + a[..., :3] * q).astype(np.uint8)
            out[..., 3] = np.maximum(a[..., 3], b[..., 3])
            _CLOUDS[key] = out
    return _CLOUDS[key]


def sun_k(hour):
    """0 (night) .. 1 (high sun): the strength of the sun's rays and caustics under the water, fading as
    the sun gets low (the painted 16:38 is ~1)."""
    h = hour % 24
    day = 1.0 - daylight.night(h)
    low = clamp((19.0 - h) / 2.3) if h >= 12 else clamp((h - 5.6) / 1.6)
    return min(day, low)


def _lantern(c, x, y, f, seed, k, m=None):
    """sc.lantern by the clock: lit (its 4 px glow growing with the ramp) or a dark glass by day; the lit
    pixel goes into the keep mask m."""
    fd = lamp_fade(k, lamp_thr(seed, 3))
    if fd <= 0:
        c.px(x, y, LAMP_UNLIT)
        return
    fl = 0.75 + 0.25 * math.sin(f * 0.9 + seed * 2.3) * math.sin(f * 0.37 + seed)
    fx.glow(c, x, y, 4, 'amber', 0.35 * fl * fd, rings=2)
    c.px(x, y, 'butter' if fl > 0.7 else 'gold')
    if m is not None and 0 <= x < W and 0 <= y < H:
        m[y, x] = True


def _marked(c, m, box, draw):
    """draw(), adding the pixels it changed inside box (x0, y0, x1, y1) to the keep mask m."""
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3]))
    if x1 <= x0 or y1 <= y0:
        draw()
        return
    before = c.a[y0:y1, x0:x1].copy()
    draw()
    m[y0:y1, x0:x1] |= np.any(c.a[y0:y1, x0:x1] != before, axis=2)


def draw_crag(c, f, cam, k, m, oy=0):
    """The film's draw_crag with its lights by the clock: the 9 lanterns and the 5 glowing cave mouths
    on the stairway (lit pixels into the keep mask m, its rows shifted by oy: the canvas' row y is the
    frame's y + oy)."""
    lay, _ = sc._crag()
    ox = sc.CRX - int(round(cam * sc.P_CRAG))
    c.blit(lay, ox, 0)
    sx = lambda x: x - int(round(cam * sc.P_CRAG))
    mm = np.zeros(c.a.shape[:2], bool)
    for i, (x, y) in enumerate(sc.LANDINGS):
        _lantern(c, sx(x + 2), y - 5, f, i, k, mm)
    for i, x in enumerate((130, 144, 158)):
        c.vline(sx(x), 121, 124, '#4a3a4a')
        _lantern(c, sx(x), 120, f, 10 + i, k, mm)
    for i, (x, y) in enumerate(sc.CAVES):
        fd = lamp_fade(k, lamp_thr(i, 5))
        if fd > 0:
            q = 0.6 + 0.4 * math.sin(f * 0.5 + i * 1.7)
            cc = mix('#c8642a', '#ffb84a', q)
            cc = mix(mix('#c8642a', CAVE_UNLIT, 0.5), cc, fd)
        else:
            cc = CAVE_UNLIT
        c.rect(sx(x) - 1, y - 2, 3, 3, cc)
        c.px(sx(x), y - 3, cc)
        if fd > 0:
            x0 = max(0, sx(x) - 1)
            mm[max(0, y - 3):max(0, y + 1), x0:max(0, sx(x) + 2)] = True
    sc.flag(c, sx(184), sc.PLAT - 45, f, '#e3b24f', length=5, seed=0, ph=6)
    sc.flag(c, sx(224), sc.PLAT - 45, f, '#e3b24f', length=5, seed=1.3, ph=6)
    for i, (x, y) in enumerate(((183, sc.PLAT - 23), (223, sc.PLAT - 23), (198, sc.PLAT - 14), (207, sc.PLAT - 14),
                                (184, sc.PLAT - 16))):
        if (f // 6 + i * 3) % 11 == 0:
            c.px(sx(x), y, 'butter')
    g = (f - 72) % 54
    if g < 10:
        yy = sc.PLAT - 22 - g * 2
        c.px(sx(206), yy, 'white')
        c.px(sx(207), yy, '#fff2a0')
    for i, (x, clr) in enumerate(((116, 'hot_pink'), (126, 'aqua'))):
        b = int(round(math.sin(f * 0.18 + i * 2) * 0.6))
        y = 132 + b
        c.hline(sx(x), sx(x) + 6, y, clr)
        c.hline(sx(x) + 1, sx(x) + 5, y + 1, shade(clr, 0.6))
        c.px(sx(x) + 3, y - 1, 'paper')
    for i in range(28):
        x = int(lerp(126, 276, i / 27))
        ph = math.sin(f * 0.15 + i * 1.3)
        if ph > -0.3:
            c.hline(sx(x) - 1, sx(x) + 1 + int(ph * 2), sc.CBASE + (i % 2), 'white' if ph > 0.4 else '#cfe7f7')
    if m is not None:
        h = c.a.shape[0]
        r0, r1 = max(0, -oy), min(H, h - oy)
        if r1 > r0:
            m[r0:r1] |= mm[r0 + oy:r1 + oy]


class _NoGlow:
    """engine.fx without its glows (for drawing a light's own pixels into a keep mask)."""
    def glow(self, *a, **k):
        pass

    def __getattr__(self, k):
        return getattr(fx, k)


_anemone_body = rebind(sc.anemone, fx=_NoGlow())


def anemone(c, x, y, f, color, size, seed, k, m):
    """sc.anemone (Seaquestria's lantern anemones) by the clock: its glow grows with the ramp (none by
    day); its lit bulb and tendrils go into the keep mask m."""
    fd = lamp_fade(k, lamp_thr(seed, 9))
    p = sc.beat_pulse(f + seed * 4)
    hgt = 5 + size * 3
    cc = mix(mix(color, '#2a2458', 0.35), color, fd)       # unlit by day: a duller bulb
    if fd > 0:
        fx.glow(c, x, y - hgt - 1, 10 + size * 5 + int(p * 5), color, (0.3 + 0.3 * p) * fd, rings=3)
        r = 2 + size + 4
        _marked(c, m, (x - r - 2, y - hgt - r - 2, x + r + 6, y - hgt + r),
                lambda: _anemone_body(c, x, y, f, cc, size, seed))
    else:
        _anemone_body(c, x, y, f, cc, size, seed)


@lru_cache(maxsize=1)
def _mid_units():
    """Seaquestria's palace layer (sc._mid_d): its lit windows (cyan glass) and the dome's pink door as
    units (ys, xs, ramp threshold, unlit colour)."""
    from scipy import ndimage
    lay = sc._mid_d()[0]
    rgb = lay[..., :3]
    out = []
    for cc, unlit in (('#7ff8ff', GLASS_UNLIT), ('#ff9ae0', DOOR_UNLIT)):
        m = (lay[..., 3] >= 128) & np.all(rgb == np.array(col(cc), np.uint8), axis=2)
        lab, n = ndimage.label(m)
        for i, s_ in enumerate(ndimage.find_objects(lab), 1):
            if s_ is None:
                continue
            yy, xx = np.nonzero(lab[s_] == i)
            yy, xx = yy + s_[0].start, xx + s_[1].start
            out.append((yy, xx, lamp_thr(int(xx[0]), int(yy[0]), 11), col(unlit)))
    return tuple(out)


_MID_VAR = {}


def _mid_var(kq):
    """sc._mid_d's layer with the windows / door that are not on at ramp kq (0.05 steps) unlit, and a mask
    (layer px) of the lit ones."""
    if kq not in _MID_VAR:
        lay = sc._mid_d()[0]
        a = lay.copy()
        lit = np.zeros(lay.shape[:2], bool)
        for yy, xx, thr, unlit in _mid_units():
            if kq > thr:
                lit[yy, xx] = True
            else:
                a[yy, xx, :3] = unlit
        _MID_VAR[kq] = (a, lit)
    return _MID_VAR[kq]


# ----------------------------------------------------------------------------- film helpers, re-clocked
_RAIN = [(22 + int(hash01(i, 1) * 96), 3 + int(hash01(i, 2) * 3), hash01(i, 3) * 44) for i in range(70)]


ZAP_BOLTS = (2, 10, 19, 28, 41)  # SPACE: a volley of the film's three bolts (steps after the trigger)
ZAP_LEN = 72


def draw_storm(c, f, bf, cam, za=-1, m=None):
    """The film's draw_storm: rain and the castle's glowing cracks on the free clock f, the bolts and
    the cloud flickers on the storm clock bf (film frames, 72.. per cycle). za: steps since the player
    called the lightning (-1: not now): a volley of bolts and the castle's cracks blazing. m: keep mask
    (the bolts and the glowing cracks are light: they shine at night)."""
    lay, smask = sc._storm()
    ox = int(round(-cam * sc.P_FAR)) - 6
    oy = sc.STORM_OY
    c.blit(lay, ox, oy)
    for (x, L, h3) in _RAIN:
        y = 62 + int((h3 + f * 2.5) % 44)
        if y + L < sc.HZ:
            c.line(x + ox + (y - 60) // 3, y, x + ox + (y + L - 60) // 3, y + L, '#8f86ae')
    lit, bolt = 0.0, None
    for (f0, dur, p0, p1) in sc.BOLTS:
        if f0 <= bf < f0 + dur:
            lit = 1.0 - (bf - f0) / dur * 0.6
            bolt = (f0, p0, p1)
    for f0 in (82, 112, 148):
        if f0 <= bf < f0 + 2:
            lit = max(lit, 0.45)
    blaze = 0.0
    if 0 <= za < ZAP_LEN:
        blaze = 1 - za / ZAP_LEN
        for j, z0 in enumerate(ZAP_BOLTS):
            if z0 <= za < z0 + 3:
                f0, dur, p0, p1 = sc.BOLTS[j % 3]
                lit = max(lit, 1.0 - (za - z0) / 3 * 0.6)
                bolt = (f0 + j, p0, p1)
        if za < 50 and za % 9 in (5, 6):
            lit = max(lit, 0.4)
    if lit > 0:
        sc.add_light(c, ox, oy, smask.shape[1], smask.shape[0], smask, '#a080d0', 0.55 * lit)
    box = (ox, oy, ox + lay.shape[1], oy + lay.shape[0])
    mk = m if m is not None else np.zeros((H, W), bool)
    if bolt:
        f0, (x0, y0), (x1, y1) = bolt
        pts = sc._bolt_pts(f0, x0 + ox, y0 + oy, x1 + ox, y1 + oy)

        def lines():
            for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
                c.line(xa - 1, ya, xb - 1, yb, '#a8a0ff')
                c.line(xa + 1, ya, xb + 1, yb, '#d8d0ff')
                c.line(xa, ya, xb, yb, 'white')
        _marked(c, mk, box, lines)
        fx.glow(c, x1 + ox, y1 + oy, 12, '#c0b0ff', 0.45, rings=3)
    on = max(0.5 + 0.5 * math.sin(f * 0.21), blaze)
    cc = mix('#1f6a78', '#4fe8ff', on) if blaze < 0.5 else '#c8fcff'
    if blaze > 0:
        fx.glow(c, 71 + ox, sc.SBASE - 30 + oy, 26, '#4fe8ff', 0.4 * blaze, rings=4)

    def cracks():
        for crack in sc.CASTLE_CRACKS:
            for (xa, ya), (xb, yb) in zip(crack, crack[1:]):
                c.line(xa + ox, ya + oy, xb + ox, yb + oy, cc)
    _marked(c, mk, box, cracks)
    if on > 0.6:
        fx.glow(c, 71 + ox, sc.SBASE - 26 + oy, 7, '#4fe8ff', 0.22 * on, rings=2)


def draw_ship(c, f, X, cam, facing=1, surge=0.0):
    """The film's draw_ship at layer x X (p 0.7), facing east (1) or west (-1, mirrored); `surge` 0..1
    stretches the wake and throws a bow wave."""
    b = int(round(math.sin(f * 0.16) * 0.8))
    xi = int(round(X - cam * 0.7))
    if not -40 < xi < W + 30:
        return

    def mx(x):
        return x if facing > 0 else 2 * xi + 25 - x

    y = 120 + b
    c.blit(sc._ship(), xi, y, flip=facing < 0)
    sc.flag(c, xi + (12 if facing > 0 else 13), y + 1, f + int(surge * 20), 'hot_pink',
            length=4 + (1 if surge > 0.3 else 0), pole=False)
    for i in range(4 + int(surge * 4)):
        L = 3 + i + int(surge * 3)
        if (f // 4 + i) % 3:
            xa, xb = mx(xi - 2 - i * 5), mx(xi - 2 - i * 5 + L)
            c.hline(min(xa, xb), max(xa, xb), y + 23 + (i % 2), '#e6f2fa' if i < 2 else '#a8cbe6')
    if surge > 0.05:                                    # bow wave
        for k in range(int(3 + surge * 6)):
            h = int(hash01(k, f // 3, 5) * (2 + surge * 5))
            c.px(mx(xi + 24 + k % 3), y + 21 - h, 'white' if k % 2 else '#d6ecf8')
        xa, xb = mx(xi + 20), mx(xi + 27 + int(surge * 3))
        c.hline(min(xa, xb), max(xa, xb), y + 22, 'white')


def _sky_band(c, surf, clock=None):
    """The summit's late-afternoon sky above the waterline (bottom-anchored on the water); clock: the
    clock's sky (a screen canvas) to use instead (MountAris._sky_now)."""
    if surf <= 0:
        return 0
    hh = min(H, surf)
    if clock is not None:
        c.a[:hh] = clock.a[:hh]
        return hh
    sky = sc._sky_b()
    s0 = sky.h - surf
    c.a[:hh] = sky.a[s0:s0 + hh]
    return hh


def _sea_strip(c, f, surf, cx):
    """A thin strip of sea at the waterline seen from the side (the film's _above_water)."""
    if 0 < surf <= H:
        c.rect(0, surf - 2, W, 2, '#3d86c4')
        for i in range(20):
            x = int((hash01(i, 5) * W + f * 0.6 - cx * 0.8) % W)
            c.hline(x, x + 3, surf - 2, '#a8d0f0')


def _band_clouds(c, f, surf, cx, specs, q=1.0):
    for (x, dy, w, h, s, p) in specs:
        y = surf - dy
        if y + h > 0 and y < surf - 4:
            c.blit(cloud(w, h, s, q), _wrap(x - cx * p + f * 0.1, w), y)


def _put(c, xs, ys, rgb):
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    c.a[ys[m], xs[m]] = rgb


def kelp(c, x, ybase, hgt, f, seed, cols, amp=4.0, leaf_every=5, width=1):
    """The film's kelp strand (sc.kelp), with its blades drawn in a few numpy calls instead of ~40 puts."""
    rgb = [col(k) for k in cols]
    k = np.arange(hgt)
    u = k / max(1, hgt - 1)
    xs = np.round(x + np.sin(f * 0.07 + seed * 1.7 - k * 0.11) * amp * u ** 1.3).astype(int)
    ys = ybase - k
    _put(c, xs, ys, rgb[1])
    if width > 1:
        _put(c, xs + 1, ys, rgb[0])
    _put(c, xs[::3] - 1, ys[::3], rgb[2])
    js = np.arange(2, hgt - 2, leaf_every)
    if len(js):
        side = np.where((js // leaf_every + seed) % 2 == 1, 1, -1)[:, None]
        L = (3 + (4 * np.sin(np.pi * np.minimum(1, js / hgt + 0.15))).astype(int))[:, None]
        fl = np.sin(f * 0.09 + seed + js * 0.3)[:, None]
        q1 = np.arange(1, 8)[None, :]
        px = xs[js][:, None] + side * q1
        py = ys[js][:, None] - (q1 * (0.6 + 0.3 * fl)).astype(int)
        v = q1 <= L
        _put(c, px[v], py[v], rgb[2])
        v = q1 < L
        _put(c, px[v], py[v] + 1, rgb[1])
        _put(c, px[:, 1], py[:, 1], rgb[3])
    _put(c, xs[-1:], ys[-1:] - 1, rgb[3])


def _clip_kelp(c, x, ybase, hgt, f, seed, surf, **kw):
    """The film's kelp strand, cut short a few px under the water surface."""
    hgt = min(hgt, ybase - surf - 4)
    if hgt > 6 and -12 < x < W + 12:
        kelp(c, x, ybase, hgt, f, seed, **kw)


def _stamp(a, arr, x, y):
    """Magenta (never sky) over `a` wherever the RGBA array / opaque Canvas `arr` at (x, y) is solid."""
    rgb = arr.a if isinstance(arr, Canvas) else arr
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    hit = sub[..., 3] >= 128 if sub.shape[2] == 4 else np.ones(sub.shape[:2], bool)
    a[y0:y1, x0:x1][hit] = (255, 0, 255)


_PALACE_NIGHT = []
_ALL_SKY = np.ones((H, W), bool)


def _palace_night():
    """The summit palace at night: the low sun's hot gold rim down the dome's right side toned down
    (the re-lighting would keep it shining like a lamp); lit windows and lanterns untouched."""
    if not _PALACE_NIGHT:
        lay, dm, _ = sc._palace()
        b = lay.copy()
        r, g, bl = (b[..., i].astype(np.int16) for i in range(3))
        hot = dm & (b[..., 3] >= 128) & (r > 226) & (g > 95) & (r - bl > 70)
        b[..., 0][hot] = 226
        b[..., 1][hot] = (g[hot] * 0.9).astype(np.uint8)
        _PALACE_NIGHT.append(b)
    return _PALACE_NIGHT[0]


def beam(c, xt, yt, wt, dxl, dxr, hgt, color, k):
    """fx.light_beam, vectorised: a dithered additive ray from the top edge (xt..xt+wt, row yt) to the
    bottom edge (xt+dxl .. xt+wt+dxr, row yt+hgt), fading toward the bottom."""
    y0, y1 = max(0, yt), min(H, yt + hgt + 1)
    if y1 <= y0:
        return
    ys = np.arange(y0, y1)
    t = (ys - yt) / hgt
    xl = (xt + dxl * t).astype(int)
    xr = (xt + wt + dxr * t).astype(int)
    x0, x1 = max(0, int(xl.min())), min(W, int(xr.max()) + 1)
    if x1 <= x0:
        return
    xx = np.arange(x0, x1)[None, :]
    m = (xx >= xl[:, None]) & (xx <= xr[:, None]) & (bayer(y1 - y0, x1 - x0, x0, y0) < (1 - t * 0.7)[:, None])
    reg = c.a[y0:y1, x0:x1]
    reg[m] = np.clip(reg[m].astype(np.float32) + np.array(col(color), np.float32) * k, 0, 255).astype(np.uint8)


def _bubble_rise(c, a, x0, surf, seed, n=40):
    """After the dive splash: the plume's bubbles are driven down, then wobble back up to the surface;
    a ring of foam spreads and fades on the water."""
    for i in range(n):
        tt = a - hash01(seed, i, 1) * 8
        if tt < 0:
            continue
        d = 26 + 70 * hash01(seed, i, 2)
        y = surf + 4 + d * (1 - (1 - min(1.0, tt / 18)) ** 2) - max(0.0, tt - 18) * (0.7 + 0.8 * hash01(seed, i, 3))
        if y <= surf + 2 or y >= H:
            continue
        x = x0 + (hash01(seed, i, 4) - 0.5) * (8 + min(tt, 30) * 1.2) + math.sin(tt * 0.2 + i) * 2
        r = 0 if hash01(seed, i, 5) < 0.35 else (1 if hash01(seed, i, 5) < 0.8 else 2)
        sc.bubble(c, x, y, r)
    if 20 <= a < 90:                            # foam patches drifting apart on the surface
        k = 1 - (a - 20) / 70
        for i in range(14):
            if hash01(seed, i, 6) > k + 0.2:
                continue
            dx = (hash01(seed, i, 7) - 0.5) * (30 + a * 1.4)
            w = 1 + int(3 * hash01(seed, i, 8) * k + 1)
            c.hline(int(x0 + dx), int(x0 + dx) + w, surf + (i % 2), 'white' if i % 3 else '#d8f4ff')


class MountAris(GlowKeep, Region):
    SEG = 'mountaris'
    CARD = ('SEA KINGDOMS', ['PALACE ON A SEA CRAG', 'CORAL CITY BELOW'])

    def build_areas(self):
        self.ship_spot = Hotspot('ship', 108, 135, 'horn', cooldown=150, p=0.7)   # low: its bubble clears the storm tag
        return [
            Area(self.render_crag, cam_x=(0, CAM_A), y_range=(30, 132), start=(132, 62),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=('#ffe2c4', 0.18),
                 hotspots=[Hotspot('storm', STORM_X - 1, 50, 'bolt', cooldown=120, p=sc.P_FAR), self.ship_spot],
                 stamps=STAMPS_A, sky=self.sky_crag),
            Area(self.render_summit, cam_x=(0, CAM_BX), cam_y=(0, CAM_BY), y_range=(26, 268), start=(44, 76),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=('#ffe2c4', 0.18),
                 hotspots=[Hotspot('dome', DOME_HOT[0], DOME_HOT[1], 'spark', cooldown=90)], stamps=STAMPS_B,
                 sky=self.sky_summit),
            Area(self.render_dive, cam_x=(0, CAM_C), cam_y=(-K_C, 0), y_range=(BY_TOP_C, BY_SEA), start=(40, 0),
                 balloon='m', balloon_tint=('#ffe2c4', 0.18),
                 hotspots=[Hotspot('dive', SPLASH_X, S_LOW - 3, 'splash', cooldown=60)], sky=self.sky_band),
            Area(self.render_sea, cam_x=(0, CAM_D), cam_y=(-K_D, 0), y_range=(BY_TOP_D, BY_SEA), start=(40, 4),
                 landmarks=LANDMARKS_D, balloon='m', balloon_tint=('#ffd6d8', 0.2),
                 hotspots=[Hotspot('pearl', PEARL[0], PEARL[1], 'gem', cooldown=90)], stamps=STAMPS_D,
                 sky=self.sky_band),
        ]

    # ------------------------------------------------------------------ lights (game/daylight.py)
    def auto_glow(self):
        """No: the relight would keep the clouds' sunlit rims, the dome's gold edge, the cone roofs' bright
        sides, Seaquestria's teal columns and the water's bright band glowing at night. The real lights
        are kept by mask (_snap / _keep_lights)."""
        return False

    def _k(self):
        return light_k(self.g.hour)

    @staticmethod
    def _snap(c, m):
        """(ys, xs, colours) of the keep-mask pixels as drawn now, or None."""
        if m is None or not m.any():
            return None
        ys, xs = np.nonzero(m)
        return ys, xs, c.a[ys, xs].copy()

    def _keep_lights(self, c, snaps, grade=None):
        """At the end of a render: the snapped light pixels still showing (graded like the frame) keep
        their painted colour through the clock's light (GlowKeep)."""
        m = None
        for sn in snaps:
            if sn is None:
                continue
            ys, xs, v = sn
            if grade is not None:
                g = Canvas(len(ys), 1)
                g.a[0] = v
                fx.grade(g, mul=grade)
                v = g.a[0]
            st = nightwin.still(c, ys, xs, v)
            if st is not None:
                m = st if m is None else (m | st)
        self.keep_glow(c, m, 0.95)

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_crag(self, f, cx, cy):
        c = self._sky_now(cx, cy)
        return sc._sky_a().copy() if c is None else c

    def sky_summit(self, f, cx, cy):
        """The summit sky, with the distant sea, the palace, the cypress and the balustrade stamped out:
        the palace's pale sunlit wing near the low sun is close to the sky's glow colours."""
        c = self._sky_now(cx, cy)
        if c is None:
            c = Canvas()
            c.blit(sc._sky_b(), 0, -int(round(cy * 0.1)))
        _stamp(c.a, sc._far_b(), -int(round(cx * 0.25)), sc.HZB - int(round(cy * 0.25)))
        _stamp(c.a, sc._palace()[0], -cx, -cy)
        fy = 150 - int(round(cy * 1.25))
        _stamp(c.a, sc._fg_cypress(), -int(round(cx * 1.25)) + 300, fy - 88)
        _stamp(c.a, sc._fg_b(), -int(round(cx * 1.25)) - 20, fy)
        return c

    # ------------------------------------------------------------------ the world clock (game/daylight.py)
    def _sun_x(self, painted_x):
        """Screen x of the sun the sea's glitter path lies under: the clock's sun (also at the painted
        hour: the sky there is the clock's, _sky_now), None at night (no glitter)."""
        h = self.g.hour
        kind, bx, _by = daylight._body_pos(h)
        return bx if kind == 'sun' and daylight.night(h) < 0.3 else None

    def _sky_now(self, cx, cy):
        """The clock's own sky (a screen canvas) wherever daylight.sky_weight would keep some of the
        painted one (round the painted 16:38: the film's sun stood far from the clock's, so the sun and
        its haze jumped across the sky for a few seconds), else None (the film's sky; the clock's light
        repaints it anyway). The renders and the sky hooks start from the same sky."""
        h = self.g.hour
        if daylight.sky_weight(self.HOUR, h) >= 0.999:
            return None
        y0, y1 = self.area.cam_y
        c = Canvas()
        daylight.paint_sky(c.a, _ALL_SKY, h, 1.0, int(cx), int(cy), y0 - 10, y1 - y0 + 130, self.g.t)
        return c

    def _daylight(self):
        """0 (night) .. 1 (day): how strong the sun's rays and caustics under the water are."""
        from .. import daylight
        return 1.0 - daylight.night(self.g.hour)

    def sky_band(self, f, cx, cy):
        """The waterline areas: the summit sky above the surface; below it a colour no frame pixel has."""
        c = Canvas()
        c.a[:] = (255, 0, 255)
        _sky_band(c, S_LOW - cy, self._sky_now(cx, cy))
        return c

    # ------------------------------------------------------------------ camera: the waterline areas
    def _cam_target(self):
        tx, ty = super()._cam_target()
        sp = SPLIT.get(self.ai)
        if sp:                          # the camera tilts down into the sea as the balloon settles on it
            y0, y1, k = sp
            ty = -k * clamp((y1 - self.by) / (y1 - y0))
        return tx, ty

    def on_area(self, ai):
        self.afloat = self.by >= BY_SEA - 0.5
        t = self.g.t
        if ai == 0:
            self.storm_t0 = t + 30 - 22                  # first bolt ~1 s after arriving
        elif ai == 3:
            self.g.sfx('crystal', 0.5)                   # the film's coral reveal

    # ------------------------------------------------------------------ clocks
    def init_state(self):
        self.storm_t0 = 0
        self.ship_t0 = 0
        self.ship_bank = 0.0            # px banked from finished surges
        self.ship_go = 10 ** 9          # step of the last surge (none yet)
        self.afloat = False             # the basket is resting on the water (areas 2, 3)

    def on_enter(self):
        self.ship_t0 = self.g.t          # the ship leaves the harbour as you arrive
        self.ship_bank = 0.0
        self.ship_go = 10 ** 9

    def storm_f(self, t):
        return 72 + (t - self.storm_t0) % STORM_P

    def ship_at(self, t):
        """(layer x, facing) of the ship: legs to and fro, eased into each turn; surges add sailing time."""
        te = (t - self.ship_t0) + self.ship_bank + _ease_extra(t - self.ship_go, SURGE, SURGE_T)
        n, fr = divmod(te / LEG_T + LEG_PHASE, 1.0)
        k = fr * fr * (3 - 2 * fr)
        if int(n) % 2 == 0:
            return SHIP_X0 + SHIP_LEG * k, 1
        return SHIP_X0 + SHIP_LEG * (1 - k), -1

    def ship_x(self, t):
        return self.ship_at(t)[0]

    def surge_k(self, t):
        a = t - self.ship_go
        return math.sin(math.pi * a / SURGE) if 0 <= a < SURGE else 0.0

    def on_interact(self, key, t):
        g = self.g
        if key == 'storm':
            self.storm_t0 = t + 199 - STORM_P             # the ambient cycle rests a while after the volley
            g.sfx('wind', 0.25)
        elif key == 'ship':
            self.ship_bank += _ease_extra(t - self.ship_go, SURGE, SURGE_T)
            self.ship_go = t
            g.sfx('horn', 0.55)
        elif key == 'dome':
            g.sfx('shimmer', 0.6)
        elif key == 'dive':
            g.sfx('splash', 0.8)
        elif key == 'pearl':
            g.sfx('crystal', 0.6)

    def update(self, t):
        g = self.g
        X = self.ship_x(t)
        self.ship_spot.x = int(round(X)) + 13           # the horn hotspot rides on the ship
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.switch is not None:
            return
        cxm = self.cx + W / 2
        if self.ai == 0:
            bf = self.storm_f(t)
            gain = self.near_gain(STORM_X + self.cx * 0.88)
            za = t - self.spot_t.get('storm', -10 ** 6)
            if za in (7, 33):
                g.sfx('thunder', (0.55 if za == 7 else 0.4) * gain, 0)
            if za == 3:
                g.sfx('zap', 0.35 * gain)
            if bf == 100:
                g.sfx('thunder', 0.4 * gain, 0)
            elif bf == 138:
                g.sfx('thunder', 0.3 * gain, 0)
            elif bf == 172:
                g.sfx('thunder', 0.22 * gain, 0)
            if t % 144 == 12:
                g.sfx('waves', 0.4 * self.near_gain(_jx))
            if t % 400 == 210:
                g.sfx('wind', 0.2)
        elif self.ai == 1:
            if t % 216 == 40:
                g.sfx('shimmer', 0.18)
            if t % 300 == 120:
                g.sfx('fountain', 0.25 * clamp(1 - abs(196 - cxm) / 260, 0.1, 1))
            if t % 420 == 300:
                g.sfx('wind', 0.25)
        if self.ai in SPLIT:
            afloat = self.by >= BY_SEA - 0.5
            if afloat and not self.afloat:
                g.sfx('splash', 0.18)                   # touching down on the sea
            self.afloat = afloat
        if self.ai == 2:
            a = t - self.spot_t.get('dive', -10 ** 6)
            if a == 6:
                g.sfx('bubbles', 0.6)
            if t % 240 == 100:
                g.sfx('bubbles', 0.25)
            if t % 300 == 30:
                g.sfx('waves', 0.25)
        elif self.ai == 3:
            a = t - self.spot_t.get('pearl', -10 ** 6)
            if a == 14:
                g.sfx('shimmer', 0.4)
            if t % 216 == 60:
                g.sfx('bubbles', 0.25)
            if t % 432 == 300:
                g.sfx('crystal', 0.18)

    # ------------------------------------------------------------------ area 0: the sea crag
    def render_crag(self, f, cx, cy):
        t = self.g.t
        cam = cx
        c = self._sky_now(cx, cy)
        c = sc._sky_a().copy() if c is None else c
        q = warm_k(self.g.hour)
        k = self._k()
        for (x, y, w, h, s, p) in CLOUDS_A:
            c.blit(cloud(w, h, s, q), _wrap(x - cam * p + (f - 72) * 0.08, w), y)
        m = np.zeros((H, W), bool)
        draw_storm(c, f, self.storm_f(t), cam, t - self.spot_t.get('storm', -10 ** 6), m)
        snaps = [self._snap(c, m)]
        c.blit(sc._far_isles(), -int(round(cam * 0.15)) - 20, sc.HZ - 11)
        sx = self._sun_x(sc.SUN[0])
        draw_sea = sc.draw_sea if sx == sc.SUN[0] else rebind(sc.draw_sea, SUN=(-999 if sx is None else sx,
                                                                                sc.SUN[1]))
        if q < 1:                                       # the glitter path: pale by day, gold at the low sun
            draw_sea = rebind(draw_sea, C=_glitter_c(q))
        draw_sea(c, f, cam, under=sc.draw_reflection)
        m = np.zeros((H, W), bool)
        draw_crag(c, f, cam, k, m)
        win = nightwin.paint(c, self.g.hour, t, (('ma_crag', lambda: sc._crag()[0],
                                                  sc.CRX - int(round(cam * sc.P_CRAG)), 0),))
        snaps.append(self._snap(c, m if win is None else (m | win)))
        X, facing = self.ship_at(t)
        draw_ship(c, f, X, cam, facing, self.surge_k(t))
        sc.draw_basalt(c, 76 + f % 144, cam)            # the spray bursts on every beat, again and again
        fx.grade(c, mul=(1.03, 1.0, 0.95))
        self._keep_lights(c, snaps, (1.03, 1.0, 0.95))
        a = t - self.spot_t.get('ship', -10 ** 6)
        if 0 <= a < 80:
            t0 = self.spot_t['ship']
            xi = int(round(X - cam * 0.7))
            bow = xi + 26 if facing > 0 else xi - 1
            for k in range(3):                          # spray thrown off the bow, wave after wave
                ui.burst(c, t, t0 + k * 14, bow, 141, ('white', '#d6ecf8', '#a8cbe6'), n=18, seed=7 + k,
                         power=0.8, life=34, gravity=0.07)
            for i in range(12):                         # the gust: white streaks racing over the sails
                gx = -60 + int((a * 4.5 + hash01(i, 8) * 90) % 150)
                gy = 102 + int(hash01(i, 9) * 34)
                if a < 66 and (a + i) % 5:
                    L = 6 + int(hash01(i, 10) * 9)
                    x0 = xi + gx if facing > 0 else xi + 25 - gx - L
                    c.hline(x0, x0 + L, gy, 'white' if i % 3 else '#d6ecf8')
        return c

    # ------------------------------------------------------------------ area 1: the summit
    def render_summit(self, f, cx, cy):
        t = self.g.t
        c = self._sky_now(cx, cy)
        if c is None:
            c = Canvas()
            c.blit(sc._sky_b(), 0, -int(round(cy * 0.1)))
        q = warm_k(self.g.hour)
        k = self._k()
        for (x, y, w, h, s) in CLOUDS_B:
            c.blit(cloud(w, h, s, q), _wrap(x - cx * 0.08 + (f - 144) * 0.1, w), int(round(y - cy * 0.1)))
        hz = sc.HZB - int(round(cy * 0.25))
        c.blit(sc._far_b(), -int(round(cx * 0.25)), hz)
        gx = self._sun_x(sc.SUNB[0])
        for i in range(30 if gx is not None else 0):    # the sun's glitter path (none at night)
            r = int(hash01(i, 3) ** 1.4 * 60)
            if (f // 3 + i) % 5 > 2:
                continue
            x = gx + int((hash01(i, 5) - 0.5) * (4 + r * 0.6))
            c.hline(x - 1, x + 1 + r // 20, hz + 1 + r, 'butter' if i % 3 else 'white')
        lay, dm, _ = sc._palace()
        ox, oy = -cx, -cy
        from .. import daylight
        c.blit(_palace_night() if daylight.night(self.g.hour) > 0.2 else lay, ox, oy)
        lit = nightwin.paint(c, self.g.hour, t, (('ma_palace', _palace_night, ox, oy),))   # lit windows
        lit = None if lit is None else (*np.nonzero(lit),)
        if lit is not None:
            lit = (*lit, c.a[lit].copy())
        a = t - self.spot_t.get('dome', -10 ** 6)
        sweeps = [(f - sc.CUT1) % 36]
        if 0 <= a < 60:                                 # SPACE: three fast bright sweeps
            sweeps += [a * 2 - k * 20 for k in range(3)]
        for i, g in enumerate(sweeps):
            if 0 <= g < 14:
                bx = sc.DOME_X - sc.DOME_R + int(g / 13 * 2 * sc.DOME_R)
                band = np.zeros_like(dm)
                band[:, max(0, bx - 1 - (1 if i else 0)):bx + 2 + (1 if i else 0)] = True
                band &= dm
                if not i:
                    band[:, bx - 1] &= (np.arange(sc.BH_) % 2 == 0)
                sc.add_light(c, ox, oy, sc.BW_, sc.BH_, band, '#fff4c0', 0.55 if not i else 0.75)
        fx.sparkle(c, sc.DOME_X + ox, sc.DOME_BASE - sc.DOME_H - 17 + oy, f, 'white', period=36, size=3)
        fx.sparkle(c, sc.DOME_X + 14 + ox, sc.DOME_BASE - 40 + oy, f, 'butter', period=36, phase=0.5, size=2)
        if 0 <= a < 70:
            k = 1 - a / 70
            if a < 24:
                sc.add_light(c, ox, oy, sc.BW_, sc.BH_, dm, '#fff4c0', 0.5 * (1 - a / 24))
            fx.glow(c, sc.DOME_X + ox, sc.DOME_BASE - 36 + oy, 30 + a // 2, '#ffe9a0', 0.45 * k, rings=5)
            fx.sparkle(c, sc.DOME_X + ox, sc.DOME_BASE - sc.DOME_H - 17 + oy, a, 'white', period=70, size=6)
            ui.burst(c, t, self.spot_t['dome'], sc.DOME_X + ox, sc.DOME_BASE - sc.DOME_H - 12 + oy,
                     ('gold', 'butter', 'white', '#fff4c0'), n=40, seed=11, power=1.2, life=64, gravity=0.04)
            ui.burst(c, t, self.spot_t['dome'] + 12, sc.DOME_X + ox, sc.DOME_BASE - 40 + oy,
                     ('gold', 'white', 'butter'), n=26, seed=12, power=1.0, life=50, gravity=0.05)
        for i, (x, y) in enumerate(sc.BANNERS_B):
            X, Y = x + ox, y + oy
            c.hline(X - 1, X + 6, Y, '#a8782a')
            for k in range(22):
                sw = int(round(math.sin(f * 0.22 - k * 0.3 + i) * (k / 22) * 1.6))
                ww = 6 if k < 19 else 6 - (k - 18) * 2
                x0 = X + sw + (6 - ww) // 2
                c.hline(x0, x0 + ww - 1, Y + 1 + k, '#0b8a8f' if k % 10 else '#e3b24f')
                c.px(x0 + ww - 1, Y + 1 + k, '#0b5e65')
            c.circle(X + 3 + int(round(math.sin(f * 0.22 - 2.4 + i) * 0.5)), Y + 9, 2, '#f8d660')
            c.px(X + 3, Y + 9, '#fff0a0')
        for i, (x, y) in enumerate(sc.FLAGS_B):
            sc.flag(c, x + ox, y + oy, f, '#e3b24f' if i < 2 else '#0b8a8f', length=7, seed=i, ph=8)
        mk = np.zeros((H, W), bool)
        for i, (x, y) in enumerate(sc.LANTERNS_B):
            c.vline(x + ox, y + oy + 1, y + oy + 4, '#4a3a4a')
            _lantern(c, x + ox, y + oy, f, 20 + i, k, mk)
        lamps = self._snap(c, mk)
        sc._fountain(c, f, 196 + ox, 151 + oy)
        for i, (x, y) in enumerate(((90, 86), (268, 82), (300, 128), (52, 126))):
            if (f // 4 + i * 5) % 16 < 2:
                c.px(x + ox, y + oy, 'butter')
        fy = 150 - int(round(cy * 1.25))
        c.blit(sc._fg_cypress(), -int(round(cx * 1.25)) + 300, fy - 88)
        c.blit(sc._fg_b(), -int(round(cx * 1.25)) - 20, fy)
        fx.grade(c, mul=(1.04, 1.0, 0.94))
        self._keep_lights(c, (lit, lamps), (1.04, 1.0, 0.94))     # windows and lanterns still seen
        return c

    # ------------------------------------------------------------------ the balloon under the waterline
    def _mirror(self, c, f, cx, cy):
        """The film's refracted balloon under the surface, at the player's balloon (when it is low), and
        rings of ripples round the basket while it rests on the water."""
        gap = BY_SEA - self.by
        if gap > 20 or self.switch is not None:
            return
        surf = S_LOW - cy
        x = int(round(self.bx)) - cx
        sc.draw_balloon_reflection(c, f, x, surf + int(max(0, gap)) - 1)
        if gap < 2:
            for k in range(2):
                r = 5 + ((f + k * 12) % 24) // 2
                cc = 'white' if r < 10 else '#c8fff8'
                c.hline(x - r - 2, x - r, surf - 1, cc)
                c.hline(x + r, x + r + 2, surf - 1, cc)

    # ------------------------------------------------------------------ area 2: below the waves
    def render_dive(self, f, cx, cy):
        t = self.g.t
        D = 104.0 * (1 + cy / K_C)              # the film's dive depth for this camera height
        surf = S_LOW - cy
        c = Canvas()
        solid = np.zeros((H, W), bool)
        wt = sc._water_c()
        y0 = max(0, surf)
        c.a[y0:] = wt.a[y0 - surf:y0 - surf + (H - y0)]
        day = sun_k(self.g.hour)                # the sun's strength (fades as it gets low)
        for i in range(6 if day > 0.05 else 0):  # god rays from the surface (wrapped round; by day)
            x = int((10 + i * 70 - cx * 0.5) % 420) - 60 + int(math.sin(f * 0.03 + i * 1.9) * 8)
            wd = 4 + int(hash01(i, 9) * 6)
            k = (0.07 + 0.04 * math.sin(f * 0.05 + i * 2.3)) * day
            beam(c, x, surf + 2, wd, 40, 56, 158, '#c8fff4', k)
        fy = int(round(sc.SURF0 - 30 - D * 0.4))
        fxo = -int(round(cx * 0.4)) - 10
        c.blit(sc._sub_crag(), fxo + 30, fy - 112)
        sc.blit_track(c, sc._far_c(), fxo, fy, solid)
        pulse = 0.6 + 0.4 * math.sin(f * 2 * math.pi / 36)
        fx.glow(c, fxo + W - 4, fy + 86, 40, '#e46fb8', 0.28 * pulse, rings=4)
        fx.glow(c, fxo + W - 4, fy + 80, 18, '#4fd1c5', 0.2 * pulse, rings=3)
        my = int(round(sc.SURF0 - 13 - D * 0.7))
        mxo = -int(round(cx * 0.7)) - 20
        sc.blit_track(c, sc._mid_c(), mxo, my, solid)
        for i, (x, yb, hg) in enumerate(sc.KELP_MID):
            _clip_kelp(c, x + mxo, yb + my, hg, f, i, surf, cols=('#3a4a3a', '#5a6a3a', '#7a8a40', '#a8b060'),
                       amp=3.0)
        ny = int(round(sc.SURF0 - D)) - 4
        nxo = -cx - 30
        sc.blit_track(c, sc._near_c(), nxo, ny, solid)
        for i, (x, y) in enumerate(sc.CLAMS):
            if -12 < x + nxo < W + 12 and y + ny - 12 < H:
                sc.draw_clam(c, x + nxo, y + ny, f, i)
        sc.bubble_stream(c, f, 210 + nxo, 166 + ny, surf + 4, n=10, seed=3, speed=1.1)
        sc.bubble_stream(c, f, 70 + nxo, 166 + ny, surf + 4, n=8, seed=4, speed=0.9)
        sc.caustic_light(c, f, solid, surf + 2, k=0.32 * max(0.15, day), ox=cx * 0.7, oy=21 - D * 0.2)
        for i, (x, yb, hg, wd) in enumerate(sc.KELP_NEAR):
            _clip_kelp(c, x + nxo, yb + ny, hg, f, 10 + i, surf, cols=('#4a3a14', '#7a6a1e', '#b0982e', '#e8d060'),
                       amp=5.0, width=wd, leaf_every=4)
        # above the waterline: the summit sky, a couple of clouds and the crag of Mount Aris on the sea
        snap = None
        if surf > 0:
            _sky_band(c, surf, self._sky_now(cx, cy))
            _band_clouds(c, f, surf, cx, ((40, 92, 70, 22, 15, 0.1), (230, 78, 60, 20, 17, 0.12)),
                         warm_k(self.g.hour))
            _sea_strip(c, f, surf, cx)
            snap = self._crag_above(c, f, surf, -int(round(cx * 0.4)) + 20)
        sc._surface_line(c, surf, f)
        fx.light_rect(c, 0, surf + 2, W, 10, '#c8fff4', 0.18 * (0.25 + 0.75 * day), fade='down')
        self._mirror(c, f, cx, cy)
        a = t - self.spot_t.get('dive', -10 ** 6)
        if 0 <= a < 130:
            x0 = SPLASH_X - cx
            if a <= 28:
                sc.draw_splash(c, sc.DIVE + a, surf, x0)
            _bubble_rise(c, a, x0, surf, 5)
        self._keep_lights(c, (snap,))
        return c

    def _crag_above(self, c, f, surf, X):
        """The film's crag (with its harbour lanterns, caves, flags and rowboats) standing on the water,
        its base on the waterline: drawn on a canvas shifted so the crag's base row lands on `surf`."""
        dy = surf - sc.CBASE - 1
        if surf <= 0:
            return None
        tmp = Canvas()
        r0 = max(0, dy)
        tmp.a[r0 - dy:surf - dy] = c.a[r0:surf]
        m = np.zeros((H, W), bool)
        cam = (sc.CRX - X) * 2
        draw_crag(tmp, f, cam, self._k(), m, -dy)       # its lanterns and caves by the clock
        win = nightwin.paint(tmp, self.g.hour, self.g.t, (('ma_crag', lambda: sc._crag()[0],
                                                           sc.CRX - int(round(cam * sc.P_CRAG)), 0),))
        c.a[r0:surf] = tmp.a[r0 - dy:surf - dy]
        if win is not None:                             # (tmp rows are the frame's shifted by -dy)
            for y in np.nonzero(win.any(axis=1))[0]:
                if r0 <= y + dy < surf:
                    m[y + dy] |= win[y]
        m[surf:] = False
        return self._snap(c, m)

    # ------------------------------------------------------------------ area 3: Seaquestria
    def render_sea(self, f, cx, cy):
        t = self.g.t
        surf = S_LOW - cy
        c = Canvas()
        solid = np.zeros((H, W), bool)
        wd = sc._water_d()
        y0 = max(0, surf)
        c.a[y0:] = wd.a[y0 - surf + S_LOW:H - surf + S_LOW]
        day = sun_k(self.g.hour)
        k = self._k()
        snaps = []
        for i in range(4 if day > 0.05 else 0):  # sun rays from the surface (by day)
            x = 30 + i * 84 - int(cx * 0.3) + int(math.sin(f * 0.03 + i * 1.3) * 8)
            beam(c, x, surf + 3, 6, 30, 58, 147, '#a8f0ff', 0.06 * day)
        # far city (p .46 across, rigid with the sea vertically): its windows twinkle at night
        fo = -int(round(cx * FAR_P)) - 8
        fyy = FAR_Y - cy
        sc.blit_track(c, sc._far_d(), fo, fyy, solid)
        m = np.zeros((H, W), bool)
        for i in range(8):
            if (f // 4 + i * 3) % 10 < 4 and lamp_fade(k, lamp_thr(i, 13)) > 0:
                x = fo + int(hash01(i, 91) * (W + 30))
                y = 110 + int(hash01(i, 92) * 30) + fyy + 6
                c.px(x, y, '#4fd1c5' if i % 2 else '#e46fb8')
                if 0 <= x < W and 0 <= y < H:
                    m[y, x] = True
        # the palace (main layer): its windows and the dome's door lit by the clock
        mo, my = -cx - 10, MID_Y - cy
        lay, wl = _mid_var(round(k * 20) / 20)
        sc.blit_track(c, lay, mo, my, solid)
        lh, lw = wl.shape
        X0, Y0, X1, Y1 = max(0, mo), max(0, my), min(W, mo + lw), min(H, my + lh)
        if X1 > X0 and Y1 > Y0:
            m[Y0:Y1, X0:X1] |= wl[Y0 - my:Y1 - my, X0 - mo:X1 - mo]
        a = t - self.spot_t.get('pearl', -10 ** 6)
        flare = math.sin(math.pi * min(1.0, a / 80)) if 0 <= a < 80 else 0.0
        p = max(sc.beat_pulse(f), flare)
        kf = lamp_fade(k, 0.04)                 # the pearl and the spire tips: the city's first lights
        px_, py_ = sc.CONCH_X + mo, sc.CONCH_TOP - 3 + my
        fx.glow(c, px_, py_, 14 + int(p * 6), '#ffd8f4', (0.3 + 0.3 * p) * max(kf, flare), rings=4)
        _marked(c, m, (px_ - 3, py_ - 3, px_ + 4, py_ + 4), lambda: (c.circle(px_, py_, 2, '#fff0ff'),
                                                                      c.px(px_ - 1, py_ - 1, 'white')))
        for (x, top) in ((36, 60), (140, 74), (276, 70), (346, 50), (170, 96), (244, 92)):
            fx.glow(c, x + mo, top + my, 5 + int(flare * 4), '#7ff8ff' if x % 2 == 0 else '#ff9ae0',
                    0.3 * kf + 0.3 * flare, rings=2)
            c.px(x + mo, top + my - 1, 'white')
            if kf > 0 and 0 <= x + mo < W and 0 <= top + my - 1 < H:
                m[top + my - 1, x + mo] = True
        fx.glow(c, sc.DOME_D[0] + mo, sc.DOME_D[1] - 10 + my, 30, '#e46fb8', (0.10 + 0.1 * p) * kf + 0.15 * flare,
                rings=3)
        for i, ((xa, ya), (xb, yb), sag) in enumerate(sc.PEARL_STRINGS):
            if k > 0:                           # the strings' beads twinkle along them at night
                _marked(c, m, (min(xa, xb) + mo - 1, min(ya, yb) + my - 1, max(xa, xb) + mo + 2,
                               max(ya, yb) + my + sag + 3),
                        lambda: sc.pearl_string(c, f, (xa, ya + my), (xb, yb + my), sag, mo, i))
            else:
                sc.pearl_string(c, f, (xa, ya + my), (xb, yb + my), sag, mo, i)
        for i, (x, y, cc) in enumerate(sc.LANTERNS_D_MID):
            anemone(c, x + mo, y + my, f if not flare else f - (f % 18), cc, 0, i, k, m)
        snaps.append(self._snap(c, m))
        for i, (x, y) in enumerate(sc.VENTS_D):
            sc.bubble_stream(c, f, x + mo, y + my, surf + 3, n=7, seed=20 + i, speed=0.9)
        sc.caustic_light(c, f, solid, surf, color='#b8f8ff', k=0.22 * max(0.15, day), ox=cx * 0.5,
                         oy=-cy * 0.45, fade_h=220)
        # near reef (p 1.4 both ways: it sinks out of view as the camera rises)
        no, nyy = -int(round(cx * NEAR_P)) - 20, -int(round(cy * NEAR_P))
        if nyy < H:
            nl, _ = sc._near_d()
            c.blit(nl, no, nyy)
            m = np.zeros((H, W), bool)
            for i, (x, y, cc, sz) in enumerate(sc.LANTERNS_D_NEAR):
                anemone(c, x + no, y + nyy, f, cc, sz, 10 + i, k, m)
            snaps.append(self._snap(c, m))
            for i, (x, y, L) in enumerate(((84, 172, 16), (290, 170, 18))):
                sw = math.sin(f * 0.08 + i) * 2
                for b in range(9):
                    ang = -math.pi / 2 + (b - 4) * 0.2
                    ex = x + no + int(math.cos(ang) * L + sw * (b / 8))
                    ey = y + nyy + int(math.sin(ang) * L)
                    c.line(x + no, y + nyy, ex, ey, '#c83a8a')
                    c.px(ex, ey, '#ff9ad8')
                for r in (6, 11):
                    c.line(x + no - r // 2, y + nyy - r, x + no + r // 2 + int(sw), y + nyy - r, '#a02a7a')
            sc.draw_clam(c, 236 + no, 175 + nyy, f, 3)
        for i, (x, yb, hg) in enumerate(((-6, 182, 150), (8, 182, 120), (402, 182, 160), (390, 182, 110))):
            _clip_kelp(c, x + no, yb + nyy, hg, f, 30 + i, surf, cols=('#24183e', '#3a2a5e', '#5a4a8e', '#8a7ac8'),
                       amp=5.0, width=2, leaf_every=4)
        sc.bubble_stream(c, f, 140 + no, 168 + nyy, surf + 2, n=8, seed=40, speed=1.2, big=True)
        m = np.zeros((H, W), bool)
        for i in range(18):                     # drifting glow motes (they shine at night)
            tt = (f * 0.4 + hash01(i, 1) * 180) % 180
            x = int(hash01(i, 2) * W + math.sin(tt * 0.05 + i) * 6)
            y = int(180 - tt)
            if (f // 5 + i) % 4 and y > surf + 3:
                c.px(x, y, ('#7ff8ff', '#ff9ae0', '#fff0ff')[i % 3])
                if k > 0 and 0 <= x < W and 0 <= y < H:
                    m[y, x] = True
        snaps.append(self._snap(c, m))
        if flare > 0:                           # SPACE: the pearl blazes, a ring of light spreads
            fx.glow(c, px_, py_, 22 + a // 2, '#fff0ff', 0.5 * flare, rings=5)
            r = int(a * 1.6)
            if 3 < r < 120:
                c.circle(px_, py_, r, mix('#ffd8f4', '#7ff8ff', a / 80), fill=False)
            ui.burst(c, t, self.spot_t['pearl'], px_, py_ + 4, ('white', '#ffd8f4', '#7ff8ff', '#ff9ae0'), n=30,
                     seed=13, power=1.0, life=60, gravity=-0.015)
        # above the waterline: open sea, far isles on the horizon, the warm sky
        if surf > 0:
            _sky_band(c, surf, self._sky_now(cx, cy))
            _band_clouds(c, f, surf, cx, ((20, 70, 84, 26, 13, 0.08), (200, 52, 60, 18, 15, 0.1)),
                         warm_k(self.g.hour))
            c.blit(sc._far_isles(), -int(round(cx * 0.15)) - 20, surf - 14)
            _sea_strip(c, f, surf, cx)
        sc._surface_line(c, surf, f)
        self._mirror(c, f, cx, cy)
        self._keep_lights(c, snaps)
        return c
