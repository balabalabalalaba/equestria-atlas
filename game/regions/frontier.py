"""The frontier towns (Appleloosa + Dodge Junction) as an explorable diorama
(art: scenes/s15_frontier.py, used read-only).

The film shows two places in two shots (Appleloosa's Main Street, then Dodge Junction at sundown), so
the region has two areas, in film order. Recompose style: each area calls the scene's cached layers and
animation helpers itself, in the film's layer order, without the film's balloon, callouts, card, iris
or cloud-cut:

  area 0  APPLELOOSA      cam 0..266 (the film pans 0..260): the pastel false-front Main Street (store,
                          hotel, bakery, the SALT BLOCK saloon, the clock tower, the stage, the water
                          tower, the rail depot) and the young apple orchards with their windpump.
  area 1  DODGE JUNCTION  cam 0..200 (the film pans 0..48; its layers end at ~76, so the region runs each
                          layer on east with the scene's own generators): the little station with its
                          water tower and semaphore, the Friendship Express at the platform, and Cherry
                          Hill Ranch with its sorting shed and conveyor, the hill running down to the plain.

Timed film beats on game clocks: tumbleweeds roll down the street forever (one hop per beat, re-entering
off screen), a dust devil spins up now and then, the clock tower keeps game time (a minute every two
seconds), and the Friendship Express runs a loop: parked at the platform with idle wisps; it departs
into the sunset (the film's whistle jets, semaphore and accelerating chugs, silhouetted against the sun)
when the player sends it off, and rolls back in from the west by itself a while later.

SPACE (hotspots): at the Salt Block's lanterns the town's lights go out and come back on in a wave
(the film's lamps lighting up on the beats); in the street a gust sends tumbleweeds and a dust devil
through town; the orchard windpump spins up and water gushes into the irrigation ditch; the Friendship
Express whistles and departs (or is called back in); Cherry Hill's conveyor rushes cherries up into the
loft. Stamps: high by the clock tower's finial, low under the water tower's trestle, and at the far
east end of Dodge Junction, low over the cherry rows.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, hash01, clamp, mix, ease_out_back
import scenes.s15_frontier as sc

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp
from .everfree import GlowKeep

B = sc.B                                # 18 frames per beat

# ----------------------------------------------------------------------------- area 0: Appleloosa
# layer ends (width - 320) / parallax: sky 375, far 333, hills 363, buttes 285, plain 272, town 270,
# foreground (680 px at 1.35) 266
CAM_A = 266
TOWN, WINS, EX = sc._town_a()
SALOON_X = EX['saloon_sign'][0]
LAMPS = (SALOON_X, EX['lanterns'][0][1] - 4)          # between the saloon's two lanterns
CLOCK_X = EX['clock'][0]
GUST = (300, 150)                                     # the street in front of the stage
WP_X, WP_Y = EX['windpump']
DITCH = EX['ditch']

LANDMARKS_A = [         # the film's callouts (town layer = world)
    Landmark('SALT BLOCK', SALOON_X, (SALOON_X, EX['saloon_sign'][1] - 10, 'up', 8)),
    Landmark('APPLELOOSA', CLOCK_X, (EX['finial'][0] + 6, 40, 'right', 6)),
    Landmark('APPLE ORCHARDS', 510, (478, 124, 'up', 14)),
]
HOTSPOTS_A = [
    Hotspot('lamps', LAMPS[0], LAMPS[1], 'lantern', cooldown=120),
    Hotspot('gust', GUST[0], GUST[1], 'wind', cooldown=120),
    Hotspot('pump', WP_X, WP_Y, 'splash', cooldown=130),
]
STAMPS_A = [Stamp('finial', CLOCK_X - 9, 17),          # high by the clock tower's finial pennant
            Stamp('trestle', EX['water_tower'][0], 121)]  # low between the water tower's legs

RELIGHT_STEP, RELIGHT_PX = 6, 28        # the relit wave: one group every 6 steps per 28 px
_FAR = max(abs(x - LAMPS[0]) for x in [w[0] + w[2] // 2 for w in WINS] +
           [p[0] for k in ('lanterns', 'stage_lamps', 'depot_lamps') for p in EX[k]])
RELIGHT_END = 6 + RELIGHT_STEP * int(_FAR / RELIGHT_PX) + 40     # the last light on, then the dusk lifts
GUST_PERIOD = 690                       # an ambient dust devil every 23 s
SPIN = 90                               # windpump / conveyor / gust spin-up (steps)
SPIN_EXTRA = {'pump': 300, 'cherries': 420, 'gust': 150}    # scene frames a spin-up adds

# ----------------------------------------------------------------------------- area 1: Dodge Junction
# The film pans only 48 px here (its layers end at cam ~76: foreground 420 px at 1.3, station 400 px, ranch
# 380 px at 0.72, plain 360 px at 0.4). East of that the region continues each layer with the scene's own
# generators (see _ranch_ext, _fg_ext, _station_ext, _plain_ext), so the balloon can follow the departing
# train past Cherry Hill: the hill runs down to the plain, the track, poles, fence and cherry rows go on.
# Still limited by the sky (330 px at 0.04): cam <= 250.
CAM_B = 200
RANCH_W = int(CAM_B * sc.P_RANCH) + W + 6
FGB_W = int(CAM_B * sc.P_FGB) + W + 6
_R0, _RT0, REX = sc._ranch_b()
_S0, SWINS, SEX = sc._station_b()
_TLAY, TEX = sc.train_sprite(0)
STOP = 196                              # the loco's front when parked (sc.train_x before departure)
WHISTLE = (STOP - TEX['front'] + TEX['whistle'][0], sc.RY - 30 + TEX['whistle'][1])
DEP_ACC = 0.5 * 0.075                   # the film's departure: front = STOP + DEP_ACC * (a - 5)^2
OUT = CAM_B + W + 24                    # rear past this: no camera sees the train
A_OUT = 5 + math.sqrt((OUT + TEX['front'] - STOP) / DEP_ACC) + 1
ARR, ARR_DEC = 110, 0.02                # arrival: rolls in from off screen west, stops at STOP
TRAIN_RETURN = 420                      # steps after it left before it comes back by itself
BELT = REX['belt']


def _paste_left(cv, lay):
    """Copy a film layer (rgb, mask) over the left part of a wider canvas (its every column)."""
    a, m = lay
    h, w = m.shape
    cv.a[:h, :w] = a


@lru_cache(maxsize=1)
def _ranch_ext():
    """sc._ranch_b continued east (the same hill formula and noise, down to the plain), and its cherry rows."""
    w = RANCH_W
    cv = sc.new(w, 136)
    xs = np.arange(w, dtype=float)
    hill = 128 - 26 * np.exp(-((xs - 330) / 70) ** 2) - 6 * np.exp(-((xs - 250) / 30) ** 2)
    hill += (sc.noise1(w, 8, 101) - 0.5) * 1.2
    top = np.round(hill).astype(int)
    w0 = _R0[1].shape[1]
    for x in range(w0, w):
        if top[x] < 136:
            cv.gradient(x, top[x], 1, 136 - top[x], [(208, 136, 94), (190, 116, 86), (170, 98, 84)], soft=0.8)
            cv.a[top[x], x] = (246, 196, 128)
    _paste_left(cv, _R0)
    trees = []
    for row in range(4):                                 # sc._ranch_b's rows, run on to the new end
        sz = 0 if row < 2 else 1
        for k, x in enumerate(range(236 + row * 3, w - 4, 9 + 2 * sz)):
            by = int(hill[min(w - 1, x)]) + 2 + row * 4
            if by > 134:
                continue
            trees.append((x, by, sz, row * 31 + k, 'r' if row % 2 == 0 else 'y'))
    return sc.to_layer(cv), tuple(trees)


@lru_cache(maxsize=1)
def _fg_ext():
    """sc._fg_b continued east: the same ground line (its noise runs on), dry grass, stones and the fence."""
    w = FGB_W
    cv = sc.new(w, H)
    top = (150 + (sc.noise1(w, 13, 121) - 0.5) * 4).astype(int)
    w0 = sc._fg_b()[1].shape[1]
    for x in range(w0, w):
        cv.gradient(x, top[x], 1, H - top[x], [(176, 106, 82), (156, 90, 80), (136, 78, 78)], soft=0.8)
        cv.px(x, top[x], (232, 160, 108))
    R = sc.rng(124)
    n = (w - w0) / 420
    for i in range(int(80 * n)):
        x = w0 + int(R.random() * (w - w0))
        y = top[min(w - 1, x)]
        for k in range(3):
            cv.line(x + k - 1, y, x + k - 2 + int(R.random() * 3), y - 2 - int(R.random() * 4),
                    (214, 160, 96) if k else (140, 100, 70))
    for i in range(int(30 * n)):
        x = w0 + 3 + int(R.random() * (w - w0 - 3))
        y = int(top[min(w - 1, x)] + 4 + R.random() * 20)
        s_ = 1 + int(R.random() * 2)
        cv.ellipse(x, y, s_ + 1, s_, (120, 72, 80))
        cv.px(x + s_, y - s_ + 1, (240, 180, 120))
    for x in range(200, w, 12):
        if x >= w0 - 4:
            cv.rect(x, 140, 2, 12, (120, 76, 66))
            cv.px(x + 1, 140, (255, 210, 140))
    cv.hline(w0, w - 1, 143, (236, 186, 120))
    cv.hline(w0, w - 1, 147, (176, 120, 82))
    _paste_left(cv, sc._fg_b())
    return sc.to_layer(cv)


@lru_cache(maxsize=1)
def _station_ext():
    """sc._station_b with its eastern track run on: two poles (and their wires) repeat every 92 px."""
    a, m = _S0
    reps = -(-(CAM_B + W + 8 - 360) // 92)
    return (np.ascontiguousarray(np.concatenate([a[:, :360]] + [a[:, 268:360]] * reps, axis=1)),
            np.ascontiguousarray(np.concatenate([m[:, :360]] + [m[:, 268:360]] * reps, axis=1)))


@lru_cache(maxsize=1)
def _plain_ext():
    """sc._plain_b with 100 more px (its far fence posts repeat every 7 px)."""
    a, m = sc._plain_b()
    return (np.ascontiguousarray(np.concatenate([a, a[:, 10:110]], axis=1)),
            np.ascontiguousarray(np.concatenate([m, m[:, 10:110]], axis=1)))


FG_TREES = tuple((x, 162, 1, 200 + k, 'r' if k % 2 else 'y') for k, x in enumerate(range(214, FGB_W + 8, 22))) + \
    tuple((x, 182, 2, 300 + k, 'y' if k % 2 else 'r') for k, x in enumerate(range(236, FGB_W + 18, 34)))

LANDMARKS_B = [
    Landmark('DODGE JUNCTION', SEX['tower'][0], (SEX['tower'][0], SEX['tower'][1] - 3, 'up', 8)),
    Landmark('CHERRY HILL RANCH', 360, (REX['shed'][0], REX['shed'][1] + 2, 'up', 10), p=sc.P_RANCH),
]
HOTSPOTS_B = [
    Hotspot('train', WHISTLE[0] + 2, WHISTLE[1] - 2, 'whistle', cooldown=90),
    Hotspot('cherries', REX['gate'][0] - 12, REX['gate'][1] + 8, 'gear', cooldown=130, p=sc.P_RANCH),
]
STAMPS_B = [Stamp('east', CAM_B + W - 18, 126)]       # far east, low over the cherry rows

TINT_SUNSET = ('#ffb28c', 0.32)         # warm low sun on the balloon
CHERRY = ((232, 59, 59), (249, 210, 60), (174, 35, 52), (255, 236, 200))
WATER = ((196, 230, 255), (120, 180, 230), (255, 255, 255), (90, 140, 200))
SAND = ((246, 210, 160), (232, 182, 130), (214, 160, 100))


# ----------------------------------------------------------------------------- shared helpers
@lru_cache(maxsize=None)
def _sky(which):
    return sc.to_layer(sc._sky_a() if which == 'a' else sc._sky_b())


_NOSKY = np.array((255, 0, 255), np.uint8)     # sky hooks: a colour no frame pixel has (scenery)


SKY_ROWS = max(sc._sky_a().h, sc._sky_b().h)


def _cover(c, layers):
    """Paint _NOSKY over `c` wherever the (rgb, mask) layers at screen x are solid (in the sky's rows:
    below them the canvas is ink)."""
    hit = np.zeros((H, W), bool)
    for (_a, m), x in layers:
        x0, x1 = max(0, x), min(W, x + m.shape[1])
        h = min(SKY_ROWS, m.shape[0])
        if x1 > x0:
            hit[:h, x0:x1] |= m[:h, x0 - x:x1 - x]
    np.copyto(c.a, _NOSKY, where=hit[..., None])
    return c


@lru_cache(maxsize=8)
def _town_sky(cam):
    """Area 0's painted sky alone (gradient + baked glow, the low sun), as render_town draws it; the
    ranges, hills, buttes, plain, town and foreground blanked out (their sunlit faces are not sky)."""
    c = Canvas(W, H, 'ink')
    sun_o = int(round(cam * sc.P_SUN))
    sc.comp(c, _sky('a'), -sun_o, 0)
    r = sc.SUN_A[2]
    sc.comp(c, sc.sun_sprite(r, True), sc.SUN_A[0] - sun_o - r - 1, sc.SUN_A[1] - r - 1)
    return _cover(c, [(sc._far_a(), -int(round(cam * sc.P_FAR))), (sc._hills_a(), -int(round(cam * sc.P_HILL))),
                      (sc._buttes_a(), -int(round(cam * sc.P_BUTTE))), (sc._plain_a(), -int(round(cam * sc.P_PLAIN))),
                      (TOWN, -cam), (sc._fg_a(), -int(round(cam * sc.P_FG)))])


@lru_cache(maxsize=8)
def _junction_sky(cam):
    """Area 1's painted sky alone (gradient + glow, the setting sun), as render_junction draws it; the
    Badlands rim, plain, ranch, station and foreground blanked out."""
    c = Canvas(W, H, 'ink')
    so = int(round(cam * sc.P_SUNB))
    sc.comp(c, _sky('b'), -so, 0)
    r = sc.SUN_B[2]
    sc.comp(c, sc.sun_sprite(r, True), sc.SUN_B[0] - so - r - 1, sc.SUN_B[1] - r - 1)
    return _cover(c, [(sc._rim_b(), -int(round(cam * sc.P_RIM))), (_plain_ext(), -int(round(cam * sc.P_PLAINB))),
                      (_ranch_ext()[0], -int(round(cam * sc.P_RANCH))), (_station_ext(), -cam),
                      (_fg_ext(), -int(round(cam * sc.P_FGB)))])


# ---- the world clock (game/daylight.py): the art is painted at sundown, its sun low in the west, every lamp
# lit. The relight keeps no painted spot glowing (Frontier.auto_glow): the real lights are drawn by the clock
# here and kept by mask (GlowKeep); the sunset's clouds, long street shadows and window reflections follow
# the clock's sun.
def _ramp(x, a, b):
    """0 at a .. 1 at b, smooth."""
    u = clamp((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def _q(k, n=4):
    return round(k * n) / n


RIM_DAY = np.array((178, 128, 124), np.float32)    # the far mesas by day: dusty rose-brown in the haze


def _day_k(hour):
    """0 (golden hour, sunset, night: the painted light) .. 1 (full day): how far the far mesas are lit by day."""
    h = hour % 24
    return clamp(min((17.8 - h) / 1.2, (h - 6.0) / 1.0))


@lru_cache(maxsize=9)
def _rim_b_day(q):
    """sc._rim_b with its backlit sunset silhouette (dark violet, gold rim) moved q / 8 of the way to the far
    mesas' day colour, keeping its shading: at noon the film's violet read as a leftover of dusk."""
    a, m = sc._rim_b()
    if q <= 0:
        return a, m
    f = a.astype(np.float32)
    lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
    ref = float(lum[m > 0].mean()) if (m > 0).any() else 1.0
    tgt = np.clip(RIM_DAY[None, None, :] * (lum / max(1.0, ref))[..., None], 0, 255)
    k = q / 8 * 0.7
    return (f + (tgt - f) * k).astype(np.uint8), m


def _lamp_k(hour):
    """0 (day) .. 1 (night): how many of the lights are on. Each light has its own threshold (_thr): they
    come on one by one from 18.5 to 19.4 h (sundown into the dusk) and go out from 5.2 to 6.1 h."""
    h = hour % 24
    return clamp((h - 18.5) / 0.9) if h >= 12 else clamp((6.1 - h) / 0.9)


def _thr(x, y):
    """A light's threshold in 0.04..0.94 (lit once _lamp_k passes it)."""
    return 0.04 + 0.9 * hash01(int(x), int(y), 907)


def _day(hour):
    """0..1: full daylight, when the clouds and the window glass keep no sunrise / sunset colour."""
    h = hour % 24
    return _ramp(h, 6.8, 8.0) * (1 - _ramp(h, 15.6, 17.0))


def _shadows(hour):
    """('eve' | 'morn', 0..1): the low sun's long shadows across the street: the film's (slanting down-left,
    the sun in the west) from 15.6 h to the dusk, mirrored in the morning (the sun in the east); none by day
    and by night."""
    h = hour % 24
    if h >= 12:
        return 'eve', _q(_ramp(h, 15.6, 16.8) * (1 - _ramp(h, 19.0, 19.8)))
    return 'morn', _q(_ramp(h, 5.6, 6.4) * (1 - _ramp(h, 7.6, 8.6)))


def _clock_sun(hour):
    """The clock's sun (screen x, y, r) and how golden (low) it is: the train's silhouette against it."""
    kind, x, y = daylight._body_pos(hour)
    h = hour % 24
    k = _ramp(h, 16.6, 17.6) * (1 - _ramp(h, 19.0, 19.3)) if kind == 'sun' else 0.0
    return (x, y, 7), k


def _cloud_q(hour):
    """(day, night) weights in quarters for the sunset clouds' colours (see _cloud_v)."""
    n = daylight.night(hour)
    return _q(_day(hour)), _q(clamp((n - 0.05) / 0.45))


@lru_cache(maxsize=None)
def _cloud_v(w, th, sd, pal, dq, nq):
    """A sundown streak cloud as the clock lights it: by day (dq of the way) its pink and coral turned to the
    day's white and blue-grey; at night (nq) its sunset colours greyed, so the relight dims it like the sky."""
    a, m = sc._cloud(w, th, sd, pal)
    if dq <= 0 and nq <= 0:
        return a, m
    f = a.astype(np.float32)
    lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
    if dq > 0:
        tgt = np.stack([lum * 0.9 + 18, lum * 0.94 + 16, lum * 0.94 + 30], axis=-1)
        f += (tgt - f) * dq * 0.92
    if nq > 0:
        f += (lum[..., None] - f) * 0.6 * nq
    return np.where(m[..., None], np.clip(f, 0, 255).astype(np.uint8), a), m


def _clouds(c, f, clouds, cam, par, q=(0.0, 0.0)):
    """sc.draw_clouds, but each cloud wraps only while it is wholly off screen (the film's 430 px span
    let a wide cloud pop in at the left edge once the free-running clock wraps it); q: _cloud_q."""
    for (x, y, w, th, sd, pal, dr) in clouds:
        lay = _cloud_v(w, th, sd, pal, *q)
        lw = lay[1].shape[1]
        span = W + lw + 80
        sx = (x - 70 + lw + dr * f - cam * par) % span - lw
        sc.comp(c, lay, int(sx), y)


# the street's shadows (sc._town_a: shadowed px half violet, the sunlit ones a little warmer)
_SHADE = np.array((150, 92, 108), np.float32)
_LIFT = np.array((14, 8, -4), np.float32)
DAY_GLASS = {(70, 50, 96): (60, 78, 116), (92, 60, 110): (88, 112, 150), (214, 126, 140): (196, 218, 238)}


@lru_cache(maxsize=1)
def _street():
    """The street rows (SY..H) of the town layer: (no shadows, the film's evening shadows, the morning's),
    the film's long-shadow pass undone and run the other way."""
    a, m = TOWN
    TW = a.shape[1]
    hgt = np.zeros(TW, int)
    for x in range(TW):                                  # (as sc._town_a: occupancy above the boardwalk)
        idx = np.nonzero(m[:sc.GY, x])[0]
        if len(idx):
            hgt[x] = sc.GY - idx[0]
    og = 448
    hgt[og - 6:] = np.where(hgt[og - 6:] > 30, 0, hgt[og - 6:])
    xs = np.arange(TW)

    def shade(sign):
        sh = np.zeros((H - sc.SY, TW), bool)
        for r in range(H - sc.SY):
            src = np.clip(xs + sign * int(round((r + 2) * 2.2)), 0, TW - 1)
            sh[r] = hgt[src] > (r + 2) * 0.9
        return sh[..., None]
    eve = a[sc.SY:H]
    s = eve.astype(np.float32)
    she = shade(1)
    plain = np.clip(np.where(she, 2 * s - _SHADE, (s - _LIFT) / 1.06), 0, 255)
    morn = np.where(shade(-1), plain * 0.5 + _SHADE * 0.5, np.clip(plain * 1.06 + _LIFT, 0, 255))
    return plain.astype(np.uint8), eve.copy(), morn.astype(np.uint8)


@lru_cache(maxsize=6)
def _town_v(kind, wq, gq):
    """The town layer as the clock lights it: the street's long shadows `kind` wq of the way (none at 0),
    the window glass gq of the way from the sunset's violet and coral reflection to the day sky's."""
    a, m = TOWN
    if wq >= 1 and kind == 'eve' and gq <= 0:
        return TOWN
    b = a.copy()
    plain, eve, morn = _street()
    sh = eve if kind == 'eve' else morn
    b[sc.SY:H] = (plain + (sh.astype(np.float32) - plain) * wq).astype(np.uint8) if wq < 1 else sh
    if gq > 0:
        for (x, y, w, h, _t, _rt) in WINS:
            reg = b[y:y + h, x:x + w]
            for src, dst in DAY_GLASS.items():
                hit = np.all(reg == np.array(src, np.uint8), axis=2)
                reg[hit] = (np.array(src, np.float32) + (np.array(dst, np.float32) - src) * gq).astype(np.uint8)
    return b, m


def _pane(c, f, x, y, w, h, t_on, round_top):
    """sc.light_window without its square halo: only the pane is light (a halo darkened by the night is a
    muddy square round every window)."""
    if f < t_on:
        return
    k = seg(f, t_on, t_on + 4)
    if k < 0.5:
        cc = (236, 170, 110)
    else:
        cc = (255, 214, 120) if hash01(x, y, f // 6) > 0.06 else (246, 190, 100)
    c.rect(x, y, w, h, cc)
    c.rect(x, y + h - 2, w, 2, (246, 160, 80))
    c.px(x + 1, y + 1, (255, 248, 210))
    if round_top:
        c.px(x, y, mix(cc, (110, 70, 80), .5))
        c.px(x + w - 1, y, mix(cc, (110, 70, 80), .5))
    c.vline(x + w // 2, y, y + h - 1, (200, 130, 110))
    c.hline(x, x + w - 1, y + h // 2, (200, 130, 110))


def _lamp(c, x, y, f, on, seed=0):
    """sc.lamp without its square blend halos: the housing, and when lit only its glass shines."""
    c.rect(x - 1, y - 2, 3, 4, (74, 58, 70))
    if on:
        fl = hash01(seed, f // 5) > 0.15
        c.hline(x - 1, x + 1, y - 1, (255, 196, 100) if fl else (246, 178, 90))
        c.hline(x - 1, x + 1, y, (246, 160, 80))
        c.px(x, y - 1, (255, 248, 200))
        c.px(x, y, (255, 214, 120))
    else:
        c.px(x, y - 1, (150, 110, 110))
        c.px(x, y, (120, 84, 100))


class _Lights:
    """The lights drawn this frame: each box's warm (light-coloured) pixels as drawn, read back once the
    render is done (the mask GlowKeep keeps glowing at night, where nothing was drawn over them since)."""

    def __init__(self):
        self.parts = []
        self.beams = []                                  # the train's headlamp: (screen x, y) of its glass

    def add(self, c, x, y, w, h, warm=True):
        """Record the light just drawn in the box (screen px): its warm pixels (warm=False: all of them)."""
        x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x1 > x0 and y1 > y0:
            s = c.a[y0:y1, x0:x1].copy()
            v = s.astype(np.int16)
            warm = ((v[..., 0] > 170) & (v[..., 0] - v[..., 2] > 50) & (v[..., 1] > 100)) if warm else \
                np.ones(s.shape[:2], bool)
            if warm.any():
                self.parts.append((x0, y0, x1, y1, s, warm))

    def add_mask(self, c, x, y, m):
        """Record the pixels of the bool array m (placed at screen x, y) as lights."""
        h, w = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
            if mm.any():
                self.parts.append((x0, y0, x1, y1, c.a[y0:y1, x0:x1].copy(), mm))

    def mask(self, c):
        if not self.parts:
            return None
        m = np.zeros((H, W), bool)
        for (x0, y0, x1, y1, s, warm) in self.parts:
            m[y0:y1, x0:x1] |= warm & np.all(c.a[y0:y1, x0:x1] == s, axis=2)
        return m


def _semaphore(c, f, x, base, ox, rec):
    """sc.semaphore without the square blend halo round its lamp (a muddy square at night); the lamp (red,
    green once the arm is up) goes into rec: a signal shines day and night."""
    sx = x - ox
    c.rect(sx, base - 50, 2, 50, (110, 96, 110))
    c.vline(sx + 1, base - 50, base - 1, (236, 200, 170))
    for yy in range(base - 46, base - 2, 3):
        c.hline(sx - 2, sx - 1, yy, (125, 136, 148))
    c.vline(sx - 2, base - 46, base - 2, (125, 136, 148))
    k = ease_out_back(seg(f, 214, 222))
    ang = -0.75 * k
    L = 11
    px_, py_ = sx + 1, base - 46
    ex_, ey_ = px_ + math.cos(ang) * L, py_ + math.sin(ang) * L
    c.line(px_, py_, int(round(ex_)), int(round(ey_)), (232, 59, 59))
    c.line(px_, py_ + 1, int(round(ex_)), int(round(ey_)) + 1, (232, 59, 59))
    c.px(int(round(px_ + math.cos(ang) * (L - 3))), int(round(py_ + math.sin(ang) * (L - 3))) + 1, (255, 255, 255))
    c.px(int(round(px_ + math.cos(ang) * (L - 4))), int(round(py_ + math.sin(ang) * (L - 4))) + 1, (255, 255, 255))
    c.rect(px_ - 3, py_ - 2, 3, 5, (74, 58, 70))
    c.px(px_ - 2, py_, (107, 255, 120) if k > 0.5 else (255, 80, 60))
    rec.add(c, px_ - 2, py_, 1, 1, warm=False)
    c.circle(sx + 1, base - 51, 1, (74, 58, 70))


def _tumble_loop(c, f, cam, specs, cam_max):
    """The film's tumbleweeds, rolling forever: each re-enters at the left once it has left every view."""
    for (t0, x0, sp, gy, r, hop, par, sd) in specs:
        lo = -2 * r - 24
        hi = W + 2 * r + 24 + cam_max * par
        xw = lo + (x0 - lo + sp * (f - t0)) % (hi - lo)
        sc.draw_tumbleweeds(c, f, cam, [(f, xw, sp, gy, r, hop, par, sd)])


def _spin(a, extra):
    """Extra scene frames `a` steps into a spin-up adding `extra` frames in total (smooth, ends flat)."""
    a = min(max(a, 0), SPIN)
    return extra / 2 * (1 - math.cos(math.pi * a / SPIN))


def _clock_f(minutes):
    """Scene frame at which sc.clock_hands shows 18:00 + `minutes`."""
    return (minutes - 30) * (sc.N + sc.FLIGHT_SHIFT - 1) / 45 - sc.FLIGHT_SHIFT


_SAND_MAP = {(238, 214, 226): (232, 184, 132), (214, 182, 214): (212, 158, 104), (180, 146, 190): (184, 124, 86),
             (255, 240, 226): (248, 214, 166), (255, 214, 150): (255, 230, 176)}


@lru_cache(maxsize=None)
def _sand_puff(r, seed):
    """sc.puff_sprite recoloured as a ball of blown sand."""
    a, m = sc.puff_sprite(r, seed)
    a = a.copy()
    for src, dst in _SAND_MAP.items():
        a[np.all(a == np.array(src, np.uint8), axis=2) & m] = dst
    return a, m


def _puff(c, lay, x, y, level=1.0):
    """Composite a puff layer centred on (x, y) at a quantised opacity (sc.draw_puff's blend)."""
    a, m = lay
    hh, ww = m.shape
    x0, y0 = int(x) - ww // 2, int(y) - hh // 2
    if level >= 0.98:
        sc.comp(c, lay, x0, y0)
        return
    q = round(level * 4) / 4
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if q <= 0 or X1 <= X0 or Y1 <= Y0:
        return
    sm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    sa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    reg = c.a[Y0:Y1, X0:X1]
    np.copyto(reg, (reg.astype(np.float32) * (1 - q) + sa * q).astype(np.uint8), where=sm[:, :, None])


def _devil(c, fe, x, base, cam, s=1.0):
    """sc.dust_devil (film frames 150..214), scaled by s (a bigger one has a fourth, denser strand)."""
    if not 150 <= fe <= 214:
        return
    k = seg(fe, 150, 214)
    env = math.sin(math.pi * k) ** 0.6
    sx = x + (fe - 150) * 0.9 - cam
    c.dither(int(sx - 8 * s), base - 3, int(17 * s), 4, (236, 190, 140), 0.45 * env)
    n = 3 if s <= 1 else 4
    for i in range(int(22 * s)):
        hh = i * 1.5 * env
        r = (1.5 + i * 0.32 / s) * env * s
        for j in range(n):
            ang = fe * 0.5 + i * 0.55 / s + j * 2 * math.pi / n
            px_ = int(round(sx + r * math.cos(ang) + i * 0.3))
            py_ = int(round(base - hh))
            if (i + j + fe // 2) % (4 if s <= 1 else 6):
                cc = (255, 228, 180) if math.cos(ang) > 0 else (222, 170, 130)
                c.px(px_, py_, cc)
                if s > 1 and math.sin(ang) > 0:
                    c.px(px_ + 1, py_, (236, 190, 140))


TRAIN_LIT = ((255, 214, 120), (246, 160, 80))   # the cars' and the cab's lit windows (sc.train_sprite)...
TRAIN_DAY = ((128, 156, 198), (94, 116, 158))   # ...as day glass


@lru_cache(maxsize=None)
def _train_v(phase, lit):
    """sc.train_sprite(phase)'s layer with its windows lit (the film's) or day glass, and the lit windows'
    mask."""
    (a, m), _ex = sc.train_sprite(phase)
    hit = [np.all(a == np.array(src, np.uint8), axis=2) & m for src in TRAIN_LIT]
    if lit:
        return (a, m), hit[0] | hit[1]
    b = a.copy()
    for hm, dst in zip(hit, TRAIN_DAY):
        b[hm] = dst
    return (b, m), None


def _draw_train(c, X, ox, hour, lk, rec):
    """sc.draw_train with the loco's front at X, lit by the clock: the windows and the headlamp's beam on
    once the lamps are (day glass before), the silhouette against the clock's low sun (the film's is the
    painted sun's, which the clock's sky has replaced)."""
    phase = int((X - STOP) / 3) % 8
    lit = lk > 0.3
    lay, wm = _train_v(phase, lit)
    h, w = lay[1].shape
    x0, y0 = int(round(X - TEX['front'] - ox)), sc.RY - h
    sun, gk = _clock_sun(hour)
    if gk > 0:
        sc.silhouette(c, x0, y0, lay, sun, gk)
    else:
        sc.comp(c, lay, x0, y0)
    if wm is not None:
        rec.add_mask(c, x0, y0, wm)
    if lit:                                              # the headlamp lit (its beam: Frontier._beam)
        lx, ly = x0 + TEX['lamp'][0], y0 + TEX['lamp'][1]
        c.rect(lx - 4, ly - 1, 4, 3, (255, 206, 110))
        c.hline(lx - 4, lx - 1, ly - 1, (255, 226, 150))
        c.vline(lx - 1, ly - 1, ly + 1, (255, 248, 214))
        rec.add(c, lx - 4, ly - 1, 4, 3)
        rec.beams.append((lx, ly))


def _warm():
    """Build every cached layer and sprite up front (~70 ms once, while the iris opens), so the first
    frames of an area, a departure or a gust never stall."""
    for fn in (sc._far_a, sc._hills_a, sc._buttes_a, sc._plain_a, sc._fg_a, sc._rim_b, _ranch_ext, _fg_ext,
               _station_ext, _plain_ext, sc._orchard_trees):
        fn()
    _sky('a'), _sky('b')
    for ph in range(8):
        sc.train_sprite(ph)
    for r in range(1, 10):
        for sd in range(4):
            sc.puff_sprite(r, sd)
            _sand_puff(r, sd)


class Frontier(GlowKeep, Region):
    SEG = 'frontier'
    CARD = ('FRONTIER TOWNS', ['APPLES IN THE DESERT', 'CHERRIES AT THE JUNCTION'])
    _raw = None                 # the area's frame as rendered (for the sky repaint in after_light)
    _beams = ()                 # the train's lit headlamp(s) this frame (the beam is drawn in after_light)

    def auto_glow(self):
        """No painted spot glows by itself (sunlit mesa caps and rims, fruit, gold trims): the real lights
        are drawn by the clock and kept by mask (_Lights, GlowKeep)."""
        return False

    def after_light(self, c, t):
        """After the relight: the kept lights (GlowKeep), then the clock's sky over the whole open sky at
        every hour (the base keeps the painted sunset sky, its big sun low in the west, while the clock's
        sky looks alike, ~18.4 to 19.0 h: the sun popped from the clock's arc down to the horizon)."""
        super().after_light(c, t)
        raw, self._raw = self._raw, None
        self._beams = ()
        hour = self.g.hour
        if raw is not None and self.area.sky is not None and daylight.sky_weight(self.HOUR, hour) < 0.999:
            a, f, ix, iy = raw                           # (at 0.999+ the base has painted it already)
            key = (self.ai, t, ix, iy)                   # the base's memo: once per step
            if self._sky_memo is None or self._sky_memo[0] != key:
                self._sky_memo = (key, daylight.sky_pixels(a, self.area.sky(f, ix, iy).a))
            mask = self._sky_memo[1]
            if mask.any():
                y0, y1 = self.area.cam_y
                daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, y1 - y0 + 130, t)
        # (no beam from the headlamp: a warm cone blended over the night sky read as a muddy grey-brown
        # wedge; the lit lamp itself shines, kept by the render's mask)

    def _done(self, c, f, cx, cy, rec):
        """End of an area render: its lights kept glowing at night, the frame kept for the sky repaint."""
        self.keep_glow(c, rec.mask(c), 0.95)
        self._raw = (c.a.copy(), f, cx, cy)
        self._beams = tuple(rec.beams)

    def build_areas(self):
        return [
            Area(self.render_town, cam_x=(0, CAM_A), cam_y=(0, 0), y_range=(28, 142), start=(150, 56),
                 landmarks=LANDMARKS_A, balloon='m', balloon_tint=TINT_SUNSET,
                 hotspots=HOTSPOTS_A, stamps=STAMPS_A, sky=self.sky_town),
            Area(self.render_junction, cam_x=(0, CAM_B), cam_y=(0, 0), y_range=(28, 142), start=(40, 64),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=TINT_SUNSET,
                 hotspots=HOTSPOTS_B, stamps=STAMPS_B, sky=self.sky_junction),
        ]

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_town(self, f, cx, cy):
        return _town_sky(cx).copy()

    def sky_junction(self, f, cx, cy):
        c = _junction_sky(cx).copy()
        sxs, sy, r = sc.SUN_B[0] - int(round(cx * sc.P_SUNB)), sc.SUN_B[1], sc.SUN_B[2]
        k = 0.5 + 0.5 * math.sin(f * 0.15)             # the sun's shimmer, as render_junction
        for i in range(3):
            c.dither(sxs - r - 3, sy - 6 + i * 5, 2 * r + 6, 1, (255, 236, 180), 0.25 * k)
        return c

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        _warm()
        self.gusts = ()             # (step, camera x) of the recent gusts (their tumbleweeds enter there)
        self.devil_t0 = 0           # ambient dust devil schedule origin
        self.dep_t = None           # step the train's departure began (film frame 214)
        self.arr_t = None           # step its arrival began
        self.spins = {k: (0.0, -10 ** 6) for k in ('pump', 'cherries', 'gust')}

    def on_enter(self):
        self.dep_t = self.arr_t = None          # a new visit: the train waits at the platform
        self.devil_t0 = self.g.t + 240

    def spot_a(self, key, t):
        return t - self.spot_t.get(key, -10 ** 6)

    def spin_f(self, key, t):
        banked, t0 = self.spins[key]
        return banked + _spin(t - t0, SPIN_EXTRA[key])

    # ---- the Friendship Express
    def _phase(self, e):
        best = ('park', None)
        for kind, t0 in (('dep', self.dep_t), ('arr', self.arr_t)):
            if t0 is not None and t0 <= e and (best[1] is None or t0 > best[1]):
                best = (kind, t0)
        return best

    def train_at(self, e):
        """(front x or None, moving) at step e."""
        kind, t0 = self._phase(e)
        if kind == 'park':
            return STOP, False
        a = e - t0
        if kind == 'arr':
            if a < ARR:
                return STOP - ARR_DEC * (ARR - a) ** 2, True
            return STOP, False
        X = STOP + DEP_ACC * max(0, a - 5) ** 2
        return (X if X - TEX['front'] < OUT else None), a >= 5

    def train_kind(self, t):
        """'parked' | 'departing' | 'gone' | 'arriving'."""
        kind, t0 = self._phase(t)
        if kind == 'park':
            return 'parked'
        if kind == 'arr':
            return 'arriving' if t - t0 < ARR else 'parked'
        return 'departing' if t - t0 < A_OUT else 'gone'

    def semaphore_f(self, t):
        """Scene frame for sc.semaphore: 214 = arm down (red), 222 = up (green)."""
        kind, t0 = self._phase(t)
        if kind == 'park':
            return 214
        a = t - t0
        if kind == 'dep':
            return max(214, min(222, 214 + a) - max(0, int(a - A_OUT)))
        return max(214, min(222, 214 + a) - max(0, a - ARR - 24))

    # ---- events
    def on_interact(self, key, t):
        g = self.g
        if key == 'lamps':
            g.sfx('click', 0.4)
            g.sfx('chime', 0.35)
        elif key == 'gust':
            self.gusts = tuple(gu for gu in self.gusts if t - gu[0] < 400)[-2:] + ((t, self.cx),)
            self.spins[key] = (self.spin_f(key, t), t)
            g.sfx('wind', 0.6, 1)
            g.sfx('whoosh', 0.35)
        elif key in ('pump', 'cherries'):
            self.spins[key] = (self.spin_f(key, t), t)
            if key == 'pump':
                g.sfx('creak', 0.35)
                g.sfx('water', 0.4)
            else:
                g.sfx('cart', 0.4)
                g.sfx('pop', 0.35, 1)
        elif key == 'train':
            kind = self.train_kind(t)
            if kind == 'parked':                         # all aboard: whistle, then off it goes
                self.dep_t = t
                g.sfx('whistle', 0.8)
                g.sfx('steam', 0.5)
            elif kind == 'gone':                         # call it back in
                self.arr_t = t + 20
                g.sfx('whistle', 0.3)
            else:
                g.sfx('tick', 0.3)

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.ai == 0:
            if t % 92 == 30 and abs(SALOON_X - (self.cx + W / 2)) < 150:
                g.sfx('creak', 0.16)                     # the Salt Block's batwing doors
            if t % 160 == 80 and abs(WP_X - (self.cx + W / 2)) < 170:
                g.sfx('creak', 0.14, 1)                  # the windpump turning
            d = (t - self.devil_t0) % GUST_PERIOD
            if d == 0:
                g.sfx('wind', 0.3, 0)
            a = self.spot_a('lamps', t)
            if 6 <= a <= 6 + RELIGHT_STEP * 6 and (a - 6) % RELIGHT_STEP == 0:
                g.sfx('tick', 0.12)                      # the nearest lamps clicking on
            return
        # Dodge Junction: the train
        kind = self.train_kind(t)
        X, moving = self.train_at(t)
        if kind == 'departing':
            a = t - self.dep_t
            if a == 11:
                g.sfx('steam', 0.4)                      # the long whistle-steam jet
            if moving and t % B == 0 and X is not None:
                g.sfx('chug', 0.42 * self.near_gain(X), (t // B) % 2)
        elif kind == 'gone':
            if t - self.dep_t - A_OUT >= TRAIN_RETURN:
                self.arr_t = t
        elif kind == 'arriving':
            if t % B == 0:
                g.sfx('chug', 0.38 * self.near_gain(max(0, X)), (t // B) % 2)
            if t - self.arr_t == ARR - 30:
                g.sfx('whistle', 0.45)
        elif self.arr_t is not None and t - self.arr_t == ARR:
            g.sfx('steam', 0.5)                          # the stop hiss
        if t % 300 == 40:
            g.sfx('wind', 0.2, 0)

    # ------------------------------------------------------------------ area 0: Appleloosa
    def render_town(self, f, cx, cy):
        t = self.g.t
        cam = cx
        hour = self.g.hour
        c = Canvas(W, H, 'ink')
        sun_o = int(round(cam * sc.P_SUN))
        sc.comp(c, _sky('a'), -sun_o, 0)
        r = sc.SUN_A[2]
        sc.comp(c, sc.sun_sprite(r, True), sc.SUN_A[0] - sun_o - r - 1, sc.SUN_A[1] - r - 1)
        _clouds(c, f, sc.CLOUDS_A, cam, sc.P_SKY, _cloud_q(hour))
        sc.comp(c, sc._far_a(), -int(round(cam * sc.P_FAR)), 0)
        sc.comp(c, sc._hills_a(), -int(round(cam * sc.P_HILL)), 0)
        sc.comp(c, sc._buttes_a(), -int(round(cam * sc.P_BUTTE)), 0)
        sc.comp(c, sc._plain_a(), -int(round(cam * sc.P_PLAIN)), 0)
        sc.comp(c, _town_v(*_shadows(hour), _q(_day(hour))), -cam, 0)
        fw = f + int(self.spin_f('gust', t))             # the wind's clock: a gust speeds it up
        sc.draw_orchard(c, fw, cam)
        sc.comp(c, EX['fence'], -cam, 0)
        rec = _Lights()
        lk = _lamp_k(hour)
        self._lights(c, f, t, cam, lk, rec)
        dx, dy = EX['saloon_door']
        if lk > _thr(dx, dy):                            # the saloon's lit doorway (sc.batwing_doors)
            c.rect(dx - cam, dy, 12, 18, (176, 104, 70))
            c.rect(dx - cam + 2, dy + 2, 8, 12, (236, 160, 90))
            c.rect(dx - cam + 4, dy + 3, 4, 8, (255, 214, 130))
            rec.add(c, dx - cam, dy, 12, 18)
        sc.batwing_doors(c, fw, dx - cam, dy, False)     # the doors swing in front of it
        sc.clock_hands(c, EX['clock'][0] - cam, EX['clock'][1], _clock_f(50 + t / 60.0))
        fx_, fy_ = EX['finial']
        sc.pennant(c, fw, fx_ - cam, fy_ - 5, (240, 79, 120))
        g0, g1, gy = EX['stage']
        sc.bunting(c, fw, g0 - cam, g1 - cam, gy, 9, ((240, 79, 120), (150, 222, 190), (249, 194, 43),
                                                      (160, 206, 244)), 1)
        sc.bunting(c, fw, g0 - cam, g1 - cam, gy + 6, 6, ((168, 132, 243), (255, 159, 208), (255, 244, 214)), 3)
        sc.smoke(c, fw, 47 + 36 - cam, sc.GY - 72 - 13, 1)
        sc.smoke(c, fw, 95 + 6 - cam, sc.GY - 50 - 13, 2)
        self._windpump(c, f, t, cam)
        sc.glints(c, f, DITCH[0], DITCH[1], DITCH[2], cam, 5)
        spx, spy = EX['spout']
        if (f // 9) % 4 == 0:
            c.px(spx - cam, spy + (f % 9) // 2, (196, 230, 255))
        self._pump_water(c, t, f, cam)
        _tumble_loop(c, f, cam, [s for s in sc.TUMBLE_A if s[6] < 1.2], CAM_A)
        self._gust(c, t, f, cam)
        sc.comp(c, sc._fg_a(), -int(round(cam * sc.P_FG)), 0)
        _tumble_loop(c, f, cam, [s for s in sc.TUMBLE_A if s[6] >= 1.2], CAM_A)
        self._streaks(c, t)
        fx.dust_motes(c, f, (0, 90, W, 70), n=10, color=(255, 226, 170), seed=3)
        self._done(c, f, cx, cy, rec)
        return c

    def _lights(self, c, f, t, cam, lk, rec):
        """Windows and lamps, each lit once the clock's lamp ramp lk passes its threshold; after SPACE at
        the saloon the lit ones go out and come back on in a wave, each one with a flare as it lights (the
        film's lamps lighting up on the beats). The lit pixels go into rec (kept glowing at night)."""
        a = self.spot_a('lamps', t)
        wave = 0 <= a < RELIGHT_END and lk > 0
        lamps = ([(p, i) for i, p in enumerate(EX['lanterns'])] +
                 [(p, i + 3) for i, p in enumerate(EX['stage_lamps'])] +
                 [(p, i + 5) for i, p in enumerate(EX['depot_lamps'])])
        if lk <= 0:                                      # daylight: every lamp is out
            for (lx, ly), seed in lamps:
                if -12 < lx - cam < W + 12:
                    _lamp(c, lx - cam, ly, f, False, seed)
            return

        def on_at(x):
            return 6 + RELIGHT_STEP * int(abs(x - LAMPS[0]) / RELIGHT_PX)

        if wave and a < RELIGHT_END - 12:                # the town dims while the lights are out
            k = min(1.0, (a + 1) / 3) * max(0.0, 1 - a / (RELIGHT_END - 12))
            c.blend(0, 0, W, H, (70, 40, 96), 0.2 * k)
        flares = []
        for (x, y, w, h, t_on, rt) in WINS:
            if not -10 < x - cam < W + 10 or lk <= _thr(x, y):
                continue
            fw = f
            if wave:
                T = on_at(x + w // 2)
                if a < T:
                    continue
                if a - T < 8:
                    fw = t_on + (a - T)
                if a - T < 10:
                    flares.append((x + w // 2, y + h // 2, a - T))
            _pane(c, fw, x - cam, y, w, h, t_on, rt)
            rec.add(c, x - cam, y, w, h)
        for (lx, ly), seed in lamps:
            if not -12 < lx - cam < W + 12:
                continue
            on = lk > _thr(lx, ly)
            if wave and on:
                T = on_at(lx)
                on = a >= T
                if on and a - T < 12:
                    flares.append((lx, ly - 1, a - T))
            _lamp(c, lx - cam, ly, f, on, seed)
            if on:
                rec.add(c, lx - cam - 1, ly - 1, 3, 2)
        for (x, y, b) in flares:                         # each light flares as it comes on
            k = 1 - b / 10
            fx.glow(c, x - cam, y, 9, (255, 214, 130), 0.5 * k, rings=3)
            if b < 6:
                r = 2 + b
                c.hline(x - cam - r, x - cam + r, y, (255, 248, 210))
                c.vline(x - cam, y - r + 1, y + r - 1, (255, 248, 210))
        if wave and a < 30:                              # the match strike at the saloon
            ui.burst(c, t, self.spot_t['lamps'], LAMPS[0] - cam, LAMPS[1], ('gold', 'white', (255, 170, 90)),
                     n=22, seed=11, power=0.8, life=30, gravity=0.04)

    def _windpump(self, c, f, t, cam):
        """The rotor (spun up by SPACE, and by a gust), with a blur while it races."""
        e = self.spin_f('pump', t) + 0.5 * self.spin_f('gust', t)
        sc.windpump_rotor(c, WP_X - cam, WP_Y, f + int(e))
        a = self.spot_a('pump', t)
        if 10 <= a < SPIN - 10:                          # racing: a second, ghost rotor between
            sc.windpump_rotor(c, WP_X - cam, WP_Y, f + int(e) + 2)

    def _gust(self, c, t, f, cam):
        """The dust devil (ambient, every GUST_PERIOD) and, on SPACE, a gust: a sheet of blown sand and four
        tumbleweeds bowl down the street from the left edge of the view, a big dust devil spins up."""
        d = (t - self.devil_t0) % GUST_PERIOD
        if t >= self.devil_t0 and d <= 64:
            n = (t - self.devil_t0) // GUST_PERIOD
            x = 60 + int(hash01(n, 17) * 420)
            _devil(c, 150 + d, x, 160, cam)
        for t0, gcam in self.gusts:                       # every gust still under way (a retrigger
            a = t - t0                                   # must not cut the last one's tumbleweeds)
            if not 0 <= a < 400:
                continue
            _devil(c, 150 + a, GUST[0] + 22, 163, cam, 1.6)
            x0 = gcam - cam
            if a < 80:                                   # the rolling wall of sand
                for i in range(8):
                    px_ = x0 - 30 + a * 4.6 - i * 15 - hash01(i, 3) * 8
                    py_ = 160 + (i % 3) * 5 - int(hash01(i, 4) * 4) - a * 0.08 * (i % 2)
                    r = min(9, 4 + (i % 3) + int(a * 0.06))
                    lvl = (1 - a / 80) * (1 - i / 11)
                    if -12 < px_ < W + 12:
                        _puff(c, _sand_puff(r, i % 4), px_, py_, lvl)
            for k, (sp, gy, r, hop) in enumerate(((3.0, 156, 5, 7), (2.6, 162, 6, 8), (3.4, 167, 4, 6),
                                                  (2.3, 158, 7, 10))):
                xw = gcam - 2 * r - 8 - k * 26 + sp * max(0, a - 2 * k)
                if xw - cam > W + 30:
                    continue
                sc.draw_tumbleweeds(c, f, cam, [(f, xw, sp, gy, r, hop, 1.0, 20 + k)])

    def _streaks(self, c, t):
        """Wind streaks racing across the view during a gust (screen space, on top)."""
        a = self.spot_a('gust', t)
        if not 0 <= a < 48:
            return
        for i in range(10):
            y = 36 + int(hash01(i, 31) * 128)
            x = int((hash01(i, 32) * W + a * (8 + 3 * hash01(i, 33))) % (W + 60)) - 30
            L = 8 + int(hash01(i, 34) * 14)
            if a > 30 and (i + a) % 3 == 0:
                continue
            c.hline(x, x + L, y, (255, 238, 214))
            c.hline(x + L // 2, x + L + 4, y + 1, (246, 196, 150))

    def _pump_water(self, c, t, f, cam):
        """After SPACE at the windpump: water gushes from its pipe into the tank, then runs both ways
        along the irrigation ditch, bubbling up in little fountains among the apple trees."""
        a = self.spot_a('pump', t)
        if not 0 <= a < SPIN + 40:
            return
        t0 = self.spot_t['pump']
        tx = WP_X - 8 - cam
        end = SPIN + 40
        if 4 <= a < SPIN:                                # the pipe gushing into the tank
            for i in range(12):
                yy = 122 + i + ((a + i) % 3 == 0)
                xx = tx + 4 - (i * i) // 40
                c.px(xx, yy, WATER[(a // 2 + i) % 3])
                c.px(xx - 1, yy, WATER[1])
            for k in range(0, a - 4, 6):
                ui.burst(c, t, t0 + 4 + k, tx + 1, 138, WATER, n=10, seed=40 + k, power=0.55, life=20,
                         gravity=0.08)
        if a < 10:
            return
        reach = (a - 10) * 3.5                           # the water fronts, running both ways
        lo, hi = max(DITCH[0], WP_X - 8 - reach), min(DITCH[1] - 1, WP_X - 8 + reach)
        fade = 1.0 if a < end - 12 else (end - a) / 12
        c.dither(int(lo) - cam, DITCH[2], int(hi - lo) + 1, 1, (150, 206, 250), fade)
        c.dither(int(lo) - cam, DITCH[2] + 1, int(hi - lo) + 1, 1, (90, 150, 210), fade)
        for i in range(3):
            sc.glints(c, f + i * 13, int(lo), int(hi), DITCH[2], cam, 6 + i)
        for n, ox in enumerate(range(DITCH[0] + 10, DITCH[1] - 4, 22)):   # sprinkler fountains in the rows
            sx = ox - cam
            if not lo <= ox <= hi or not -10 < sx < W + 10:
                continue
            hgt = int((9 + 3 * math.sin(a * 0.35 + n * 1.7)) * fade)
            if hgt < 2:
                continue
            c.vline(sx, DITCH[2] - hgt, DITCH[2] - 1, WATER[0])
            c.vline(sx + 1, DITCH[2] - hgt + 2, DITCH[2] - 1, WATER[1])
            c.px(sx, DITCH[2] - hgt, (255, 255, 255))
            for j in range(8):                           # droplets arcing off both sides
                u = (a * 0.07 + j / 8 + hash01(n, 5)) % 1.0
                side = 1 if j % 2 else -1
                px_ = sx + side * int(1 + u * 7)
                py_ = DITCH[2] - hgt + int(u * u * (hgt + 2)) - int(2 * u * (1 - u) * 4)
                c.px(px_, py_, WATER[j % 3])
        for side, x in ((-1, lo), (1, hi)):              # a splash at each front
            if DITCH[0] < x < DITCH[1] - 1 and a < SPIN + 20:
                ui.burst(c, t, t + 0 - (a % 8), int(x) - cam, DITCH[2] - 1, WATER, n=6, seed=70 + side + a // 8,
                         power=0.4, life=8, gravity=0.1)

    # ------------------------------------------------------------------ area 1: Dodge Junction
    def render_junction(self, f, cx, cy):
        t = self.g.t
        cam = cx
        hour = self.g.hour
        c = Canvas(W, H, 'ink')
        so = int(round(cam * sc.P_SUNB))
        sc.comp(c, _sky('b'), -so, 0)
        sxs = sc.SUN_B[0] - so
        sc.comp(c, sc.sun_sprite(sc.SUN_B[2], True), sxs - sc.SUN_B[2] - 1, sc.SUN_B[1] - sc.SUN_B[2] - 1)
        k = 0.5 + 0.5 * math.sin(f * 0.15)
        for i in range(3):
            c.dither(sxs - sc.SUN_B[2] - 3, sc.SUN_B[1] - 6 + i * 5, 2 * sc.SUN_B[2] + 6, 1, (255, 236, 180), 0.25 * k)
        lk = _lamp_k(hour)
        rec = _Lights()
        _clouds(c, f, sc.CLOUDS_B, cam, 0.05, _cloud_q(hour))
        sc.comp(c, _rim_b_day(int(round(_day_k(hour) * 8))), -int(round(cam * sc.P_RIM)), 0)
        sc.comp(c, _plain_ext(), -int(round(cam * sc.P_PLAINB)), 0)
        rox = int(round(cam * sc.P_RANCH))
        ranch, rtrees = _ranch_ext()
        sc.comp(c, ranch, -rox, 0)
        lx, ly = REX['loft']
        if lk > _thr(lx, ly):                            # the loft door, open: lit inside after dark
            c.rect(lx - 4 - rox, ly - 6, 8, 7, (236, 160, 90))
            c.rect(lx - 3 - rox, ly - 5, 6, 5, (255, 206, 120))
            c.hline(lx - 4 - rox, lx + 3 - rox, ly, (246, 160, 80))
            rec.add(c, lx - 4 - rox, ly - 6, 8, 7)
        sc.draw_trees(c, f, rtrees, rox)
        sc.conveyor_cherries(c, f + int(self.spin_f('cherries', t)), BELT, rox, REX['legs'])
        swx, swy = REX['shed_win']
        if lk > _thr(swx, swy):
            _pane(c, f, swx - rox, swy, 4, 5, 0, False)
            rec.add(c, swx - rox, swy, 4, 5)
        self._cherries(c, t, rox)
        sc.comp(c, _station_ext(), -cam, 0)
        for (x, y, w, h, t_on, rt) in SWINS:
            if lk > _thr(x, y):
                _pane(c, f, x - cam, y, w, h, 0, rt)
                rec.add(c, x - cam, y, w, h)
        for i, (lx, ly) in enumerate(SEX['lamps']):
            on = lk > _thr(lx, ly)
            _lamp(c, lx - cam, ly, f, on, i)
            if on:
                rec.add(c, lx - cam - 1, ly - 1, 3, 2)
        spx, spy = SEX['spout']
        if (f // 7) % 3 == 0:
            c.px(spx - cam, spy + (f % 7) // 2, (196, 230, 255))
        sc.pennant(c, f, SEX['tower'][0] - cam, SEX['tower'][1] - 9, (240, 79, 120), 6)
        self._train(c, t, cam, hour, lk, rec)
        _semaphore(c, self.semaphore_f(t), 168, sc.RY + 12, cam, rec)
        fox = int(round(cam * sc.P_FGB))
        sc.comp(c, _fg_ext(), -fox, 0)
        sc.draw_trees(c, f, FG_TREES, fox)
        _tumble_loop(c, f, cam, [(220, -20, 2.4, 176, 6, 9, 1.3, 7), (240, 60, 1.6, 152, 3, 5, 1.0, 8)], CAM_B)
        fx.dust_motes(c, f, (0, 96, W, 60), n=10, color=(255, 226, 170), seed=9)
        self._done(c, f, cx, cy, rec)
        return c

    def _train(self, c, t, cam, hour, lk, rec):
        """Steam (idle wisps, whistle jets, chug puffs, stop hiss), then the train itself (its lights into
        rec)."""
        front, stack = TEX['front'], TEX['stack']
        e0 = t - 40 - (t - 40) % 3
        for e in range(e0, t + 1, 3):                    # sc.train_steam's plume, on the game clock
            X, moving = self.train_at(e)
            if X is None:
                continue
            age = t - e
            big = moving and e % B == 0
            med = moving and e % 9 == 0
            s = (1.35 if big else (1.05 if med else 0.75)) if moving else 0.6
            px_ = X - front + stack[0] - cam - age * (0.55 if moving else 0.15)
            py_ = sc.RY - 30 + stack[1] - 1 - (age * 0.85 - age * age * 0.006) * (1.1 if big else 0.9)
            r = (2.0 + age * 0.21) * s
            lvl = 1.0 if age < 28 else 1 - (age - 28) / 13
            sc.draw_puff(c, px_, py_, min(r, 9), e // 3, lvl)
        if self.dep_t is not None:                       # the whistle: a short blast, then a long one
            wx, wy = WHISTLE[0] - cam, WHISTLE[1]
            for (t0, dur) in ((self.dep_t + 2, 5), (self.dep_t + 11, 14)):
                age = t - t0
                if age < 0 or age > dur + 18:
                    continue
                top = min(34, age * 7) if age <= dur else 34
                lift = max(0, age - dur) * 2.2
                lvl = 1.0 if age <= dur + 8 else 1 - (age - dur - 8) / 10
                for i in range(0, top, 2):
                    sc.draw_puff(c, wx + i * 0.25 - lift * 0.4, wy - 2 - i - lift, 1.5 + i * 0.15, i + t0, lvl)
                if age < dur:
                    c.vline(int(wx), int(wy - 5), int(wy - 1), (255, 255, 255))
        if self.arr_t is not None and 0 <= t - self.arr_t - ARR < 30:   # the stop hiss from the cylinders
            a = t - self.arr_t - ARR
            for j, sgn in enumerate((-1, 1, -1, 1)):
                x = STOP - 30 - j * 4 + sgn * a * 0.9 - cam
                sc.draw_puff(c, x, sc.RY - 4 - a * 0.3, 2 + a * 0.12, j + 7, 1 - a / 30)
        X, _ = self.train_at(t)
        if X is not None and X - front - cam < W + 20 and X - cam > -20:
            _draw_train(c, X, cam, hour, lk, rec)

    def _cherries(self, c, t, rox):
        """After SPACE at the ranch gate: the conveyor races, cherries crowd up the belt and shower out of
        the loft door, and a few bounce out of the loading bin."""
        a = self.spot_a('cherries', t)
        if not 0 <= a < SPIN + 40:
            return
        t0 = self.spot_t['cherries']
        x0, y0, x1, y1 = BELT
        if a < SPIN:                                     # a crowd of cherries racing up the belt
            n = 16
            sh = self.spin_f('cherries', t) * 0.35 / math.hypot(x1 - x0, y1 - y0)
            for k in range(n):
                u = (sh + k / n) % 1.0
                if hash01(k, a // 3) < max(1 - a / 8, (a - SPIN + 20) / 20):
                    continue                             # the crowd thickens, then thins out
                x = int(round(x0 + (x1 - x0) * u)) - rox
                y = int(round(y0 + (y1 - y0) * u))
                cc = CHERRY[k % 2]
                c.rect(x, y - 1, 2, 2, cc)
                c.px(x + 1, y - 1, CHERRY[3])
        lx, ly = REX['loft']
        for k in range(0, min(a, SPIN), 10):            # showers out of the loft door
            ui.burst(c, t, t0 + 12 + k, lx - rox, ly - 3, CHERRY, n=14, seed=60 + k, power=0.6, life=36,
                     gravity=0.07)
        ui.burst(c, t, t0, x0 - 2 - rox, y0 - 4, CHERRY, n=20, seed=59, power=0.7, life=30, gravity=0.08)
        if a < 20:                                       # the loft door glows
            fx.glow(c, lx - rox, ly - 3, 10, (255, 214, 130), 0.5 * (1 - a / 20), rings=3)
