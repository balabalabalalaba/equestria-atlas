"""CRYSTAL EMPIRE: the city of crystal on the arctic plateau (scene frames 0..431, HUD 21:15 -> 22:00).

Beat sheet in SCENE frames (bar = 72, beat = 18; the 2-bar map flight is handled by long_flight,
film = scene + 72). Night: moon upper right = key light (right facets lit), inner crystal glow and the
pink Crystal Heart as secondary lights, aurora curtains overhead. Cameras use shotfx.cruise.
   0-71   (old flight slot; replaced by the director's 2-bar flight)
  72-161  SHOT A, wide: iris opens 72-79 on the PALACE (204, 89). Stars, moon, AURORA curtains, Mt.
          EVERHOOF's moonlit pyramid, the Crystal Mountains, glittering snowfield; the city in rising
          rings of faceted crystal houses (pink, lilac, teal, baby blue), lit windows, the slender
          ice-blue palace on crystal arches with the little Heart glowing beneath; the dome shimmer.
          Slow pan right (7 parallax layers), balloon 's'. Callout CRYSTAL EMPIRE 84-156; card 100-206.
 162      dissolve
 162-251  SHOT B, palace close: tilt down the spire (pennants, crown gem, shard balconies) to the
          arches and the CRYSTAL HEART turning on its pedestal (facets step every 3 frames), rainbow
          glints on the plaza. Callouts CRYSTAL PALACE 162-234, CRYSTAL HEART 180-251.
          `heart` HIT 216 (bar line): the Heart pulses (scale bounce + glow + flash), a ring of light
          rolls out over the plaza, the near houses light up; the aurora starts to flare.
 252      dissolve
 252-341  SHOT C, the wide city again: the wave keeps washing outward through every ring of houses
          (each lights brighter as it passes), the dome flashes when it reaches the rim (~274);
          second heartbeat `heart2` 288 (bar line) sends another ring and the rainbow aurora surges;
          slow tilt up into the flaring sky. Callout MT. EVERHOOF 270-340.
 342      glide
 342-431  SHOT D, the CRYSTAL STATION: faceted crystal train shed, the Friendship Express pulling in
          (lit windows, steam, stops ~396 with a hiss), the gift shop's glowing snow-globe window,
          snow falling. Slow crane up (366-426) to the whole lit city under the aurora with Mt.
          Everhoof behind; the balloon 'm' glides past. Callout CRYSTAL STATION 348-420. Toast from
          374 (top centre kept clear). Last 8 frames under the wipe.
Ambient: aurora waving, star twinkle, snow glitter, dome ripple, spindrift off Everhoof, window
flickers, heart turning/glow, pennants, lamp glows, falling snow, train steam, snow-globe flurry,
balloon drift.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.core import bayer
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'crystal'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

T_A, T_B, T_HEART, T_C, T_HEART2, T_D = 72, 162, 216, 252, 288, 342
KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)

# ---- wide-city world (shots A, C, D)
LM, LT = 100, 50            # layer margins (left, top)
LW, LH = W + 260, H + 120   # layer canvas size
PX = 204                    # palace screen x at cam 0 (iris centre)
CREF = 40                   # camera x where all ring centres line up exactly
P_EVER, P_RANGE, P_PLAT, P_IN, P_MID, P_OUT, P_FG = 0.18, 0.3, 0.5, 0.7, 0.8, 0.92, 1.3
PAL_BASE = 114              # palace platform top (inner layer world y)
PAL_S = 0.85                # palace scale in the wide


# ================================================================== generic helpers
def layer(w=LW, h=LH):
    return Canvas(w, h, KEY)


def to_rgba(cv):
    a = cv.a
    m = ~np.all(a == _KEYA, axis=2)
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = m.astype(np.uint8) * 255
    return out


class OC:
    """Offset proxy: draw in world coordinates onto a layer canvas."""
    __slots__ = ('c', 'ox', 'oy')

    def __init__(self, c, ox=LM, oy=LT):
        self.c, self.ox, self.oy = c, ox, oy

    def rect(self, x, y, w, h, cc):
        self.c.rect(int(round(x)) + self.ox, int(round(y)) + self.oy, int(round(w)), int(round(h)), cc)

    def px(self, x, y, cc):
        self.c.px(int(round(x)) + self.ox, int(round(y)) + self.oy, cc)

    def hline(self, x0, x1, y, cc):
        self.c.hline(int(round(x0)) + self.ox, int(round(x1)) + self.ox, int(round(y)) + self.oy, cc)

    def vline(self, x, y0, y1, cc):
        self.c.vline(int(round(x)) + self.ox, int(round(y0)) + self.oy, int(round(y1)) + self.oy, cc)

    def line(self, x0, y0, x1, y1, cc):
        self.c.line(x0 + self.ox, y0 + self.oy, x1 + self.ox, y1 + self.oy, cc)

    def poly(self, pts, cc):
        self.c.poly([(x + self.ox, y + self.oy) for x, y in pts], cc)

    def ellipse(self, cx, cy, rx, ry, cc, fill=True):
        self.c.ellipse(cx + self.ox, cy + self.oy, rx, ry, cc, fill)

    def circle(self, cx, cy, r, cc, fill=True):
        self.c.circle(cx + self.ox, cy + self.oy, r, cc, fill)

    def dither(self, x, y, w, h, cc, level):
        self.c.dither(int(round(x)) + self.ox, int(round(y)) + self.oy, int(round(w)), int(round(h)), cc, level)

    def round_rect(self, x, y, w, h, cc, r=1):
        self.c.round_rect(int(round(x)) + self.ox, int(round(y)) + self.oy, int(round(w)), int(round(h)), cc, r)

    def gradient(self, x, y, w, h, cols, soft=0.5):
        self.c.gradient(int(x) + self.ox, int(y) + self.oy, int(w), int(h), cols, soft=soft)


def add_light(c, mask_xy, color, k):
    """Additive light of `color` * k on pixels (ys, xs)."""
    ys, xs = mask_xy
    if len(xs) == 0:
        return
    reg = c.a[ys, xs].astype(np.int16) + (np.array(col(color), np.float32) * k).astype(np.int16)
    c.a[ys, xs] = np.clip(reg, 0, 255).astype(np.uint8)


def place(c, rgba, camx, camy, par, ox=LM, oy=LT):
    c.blit(rgba, -ox - int(round(camx * par)), -oy - int(round(camy * par)))


def ramp(base, sh='royal', lit=False):
    b = col(base)
    if lit:
        return dict(ol=mix(b, 'ink', 0.5), dk=mix(b, sh, 0.3), md=mix(b, 'white', 0.05),
                    lt=mix(b, 'white', 0.3), hi=mix(b, 'white', 0.75), win='white', win2='butter')
    return dict(ol=mix(b, 'ink', 0.62), dk=mix(b, sh, 0.55), md=mix(b, sh, 0.27), lt=b,
                hi=mix(b, 'white', 0.55), win='cream', win2='gold')


HUES = {
    'pink': ('crystal_pk', 'royal'),
    'lilac': (mix('lavender', 'lilac', 0.45), 'royal'),
    'teal': (mix('ice', 'turquoise', 0.25), 'blue'),
    'blue': (mix('crystal', 'baby_blue', 0.45), 'royal'),
    'violet': (mix('royal_lt', 'lavender', 0.3), 'eggplant'),
    'palace': ('crystal_lt', 'crystal_dk'),
}


def hue_ramp(name, lit=False):
    b, s = HUES[name]
    return ramp(b, s, lit)


# ================================================================== crystal geometry
def gem_shaft(o, cx, yb, yt, wb, wt, tip, rp, glow=True, edge=True):
    """Faceted crystal prism with a pointed tip: bottom yb, body top yt, tip apex at yt - tip.
    Left facet dark (moon on the right), centre facet mid->light, right facet light."""
    hb, ht = wb / 2.0, wt / 2.0
    ya = yt - tip
    o.poly([(cx - hb - 0.5, yb + 0.5), (cx - ht - 0.5, yt), (cx, ya - 0.6), (cx + ht + 0.5, yt),
            (cx + hb + 0.5, yb + 0.5)], rp['ol'])
    fb, ft = max(1.0, wb * 0.27), max(0.8, wt * 0.27)
    lb, lt_ = cx - hb + 0.5, cx - ht + 0.5
    rb, rt = cx + hb - 0.5, cx + ht - 0.5
    ab, at = lb + fb, lt_ + ft
    bb, bt = rb - fb, rt - ft
    if wb >= 3:
        o.poly([(lb, yb), (ab, yb), (at, yt), (cx, ya + 0.8), (lt_, yt)], rp['dk'])
        o.poly([(ab, yb), (bb, yb), (bt, yt), (cx + 0.2, ya + 0.8), (at, yt)], rp['md'])
        o.poly([(bb, yb), (rb, yb), (rt, yt), (cx + 0.2, ya + 0.8), (bt, yt)], rp['lt'])
        # centre-facet glow core (lighter dithered strip) + edge highlight on the lit ridge
        if glow and wb >= 5:
            h = yb - yt
            o.dither(cx - wt * 0.12, yt + 1, max(1, wt * 0.2), max(1, h - 2), rp['lt'], 0.5)
        if edge:
            o.line(bb, yb - 1, bt, yt, rp['hi'])
            o.line(bt, yt, cx, ya + 1, rp['hi'])
    else:
        o.poly([(cx - hb, yb), (cx, yb), (cx, ya + 1), (cx - ht, yt)], rp['md'])
        o.vline(cx, ya + 1, yb, rp['lt'])


def gem_bulb(o, cx, yb, ym, yt, wb, wm, rp):
    """Diamond-shaped crystal bulge (gem crown) from yb up to yt, widest (wm) at ym."""
    hb, hm = wb / 2.0, wm / 2.0
    o.poly([(cx - hb - 0.5, yb + 0.5), (cx - hm - 0.5, ym), (cx, yt - 0.6), (cx + hm + 0.5, ym),
            (cx + hb + 0.5, yb + 0.5)], rp['ol'])
    o.poly([(cx - hb + 0.5, yb), (cx - hm + 0.5, ym), (cx - hm * 0.4, ym), (cx - hb * 0.4, yb)], rp['dk'])
    o.poly([(cx - hb * 0.4, yb), (cx - hm * 0.4, ym), (cx + hm * 0.4, ym), (cx + hb * 0.4, yb)], rp['md'])
    o.poly([(cx + hb * 0.4, yb), (cx + hm * 0.4, ym), (cx + hm - 0.5, ym), (cx + hb - 0.5, yb)], rp['lt'])
    o.poly([(cx - hm + 0.5, ym), (cx, yt + 0.8), (cx - hm * 0.4, ym)], rp['md'])
    o.poly([(cx - hm * 0.4, ym), (cx, yt + 0.8), (cx + hm * 0.4, ym)], rp['lt'])
    o.poly([(cx + hm * 0.4, ym), (cx, yt + 0.8), (cx + hm - 0.5, ym)], rp['hi'])
    o.hline(cx - hm + 1, cx + hm - 1, ym, rp['hi'])


def window(o, x, y, w, h, rp, wins=None, arch=True):
    """Lit window with a dark frame; records it for flicker."""
    x, y, w, h = int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))
    if w >= 2 and h >= 3:
        o.rect(x - 1, y - 1, w + 2, h + 2, rp['ol'])
        o.rect(x, y, w, h, rp['win2'])
        o.rect(x, y, w, max(1, h // 2), rp['win'])
        if arch and w >= 2:
            o.px(x, y - 1 + 0, rp['ol'])
            o.px(x + w - 1, y, rp['ol']) if w >= 3 else None
            o.px(x, y, rp['ol'])
        o.hline(x - 1, x + w, y + h, rp['hi'])          # sill
    else:
        o.rect(x, y, w, h, rp['win'])
    if wins is not None:
        wins.append((x, y, w, h))


def house(o, cx, yb, w, h, hue, kind, seed, lit, wins=None, rp=None):
    """A faceted crystal house standing on yb, centred on cx, about w x h px."""
    rp = rp or hue_ramp(hue, lit)
    r = hash01(seed, 7)
    if kind == 'gem':           # prism body + tall pointed gem roof
        tip = max(3, int(h * 0.42))
        gem_shaft(o, cx, yb, yb - h + tip, w, w * 0.86, tip, rp)
    elif kind == 'tower':       # slim tall tower with a small side annex
        aw = max(4, int(w * 0.6))
        ah = max(4, int(h * 0.45))
        side = -1 if r < 0.5 else 1
        gem_shaft(o, cx + side * (w * 0.55), yb, yb - ah + aw * 0.6, aw, aw * 0.8, aw * 0.6,
                  rp, glow=False)
        tip = max(3, int(h * 0.3))
        gem_shaft(o, cx, yb, yb - h + tip, w * 0.75, w * 0.6, tip, rp)
    elif kind == 'bulb':        # body with a flared diamond crown
        bh = int(h * 0.5)
        gem_shaft(o, cx, yb, yb - bh, w * 0.8, w * 0.8, 1, rp, edge=False)
        gem_bulb(o, cx, yb - bh + 1, yb - bh - h * 0.12, yb - h, w * 0.7, w * 1.05, rp)
    else:                       # 'twin': two shafts of different heights
        w1 = w * 0.62
        gem_shaft(o, cx - w * 0.24, yb, yb - h * 0.62 + w1 * 0.5, w1, w1 * 0.8, w1 * 0.55, rp)
        tip = max(3, int(h * 0.35))
        gem_shaft(o, cx + w * 0.22, yb, yb - h + tip, w * 0.66, w * 0.55, tip, rp)
    # windows on the centre facet
    if w >= 9:
        ww = 2 if w < 16 else 3
        wh = 3 if w < 16 else 4
        rows = max(1, int((h * 0.55 - 6) // (wh + 3)))
        for i in range(rows):
            wy = yb - 7 - wh - i * (wh + 3) - (2 if w >= 16 else 0)
            if kind == 'bulb' and wy < yb - h * 0.5:
                break
            if kind in ('gem', 'tower', 'bulb'):
                window(o, cx - ww // 2 + (0 if w < 16 else 0), wy, ww, wh, rp, wins)
            else:
                window(o, cx + w * 0.22 - ww / 2, wy, ww, wh, rp, wins)
        # round door
        dw = 3 if w < 16 else 4
        dh = 4 if w < 16 else 6
        dx = int(round(cx - dw / 2 + (w * 0.22 if kind == 'twin' else 0)))
        o.rect(dx - 1, yb - dh - 1, dw + 2, dh + 1, rp['ol'])
        o.rect(dx, yb - dh, dw, dh, rp['win2'] if hash01(seed, 3) < 0.7 else rp['dk'])
        o.px(dx, yb - dh, rp['ol'])
        o.px(dx + dw - 1, yb - dh, rp['ol'])
    elif w >= 5:
        o.px(int(cx), yb - 3, rp['win'])
        if h > 10:
            o.px(int(cx), yb - 6, rp['win'])
        if wins is not None:
            wins.append((int(cx), yb - 3, 1, 1))
    # tiny crystal spikes at the foot (sparkly base)
    if w >= 10:
        for sx in (-1, 1):
            bx = cx + sx * (w * 0.55)
            gem_shaft(o, bx, yb, yb - 2, 3, 2, 2, rp, glow=False, edge=False)


def terrace(o, x0, x1, ytop, depth, rp, seed, lamps=True, snow=True):
    """Crystal terrace wall (rising ring): top at ytop, face height depth, snow on top."""
    o.rect(x0, ytop, x1 - x0, depth, rp['md'])
    # slab joints and facet light
    i = 0
    x = x0
    while x < x1:
        sw = 5 + int(hash01(seed, i) * 5)
        o.vline(x, ytop + 1, ytop + depth - 1, rp['ol'])
        o.rect(x + 1, ytop + 1, max(1, sw // 3), depth - 2, rp['lt'])
        o.px(x + 1, ytop + 1, rp['hi'])
        x += sw
        i += 1
    o.hline(x0, x1 - 1, ytop + depth - 1, rp['ol'])
    o.hline(x0, x1 - 1, ytop, rp['hi'])
    if snow:
        o.hline(x0, x1 - 1, ytop - 1, 'cloud')
        for k in range(int(x0), int(x1), 3):
            if hash01(seed, k, 5) < 0.4:
                o.px(k, ytop, 'cloud')


def lamp(o, x, yb, h, rp_post=None):
    """Crystal street lamp: slim post with a glowing gem lantern (glow drawn per frame)."""
    o.vline(x, yb - h, yb, mix('crystal_dk', 'ink', 0.3))
    o.px(x - 1, yb, mix('crystal_dk', 'ink', 0.3))
    o.px(x + 1, yb, mix('crystal_dk', 'ink', 0.3))
    o.rect(x - 1, yb - h - 3, 3, 3, 'cotton')
    o.px(x, yb - h - 4, 'white')
    o.px(x, yb - h - 2, 'white')


# ================================================================== sky / aurora / moon
@lru_cache(maxsize=None)
def sky_canvas():
    c = Canvas(W, H + 80)
    c.gradient(0, 0, W, H + 80, ['#0a0d20', '#0c0f24', '#141a3a', '#1e2550', 'navy', '#243e5a',
                                 'deep_teal'], soft=0.7)
    return c


def draw_sky(c, f, camx, camy):
    oy = int(round(40 + camy * 0.12))
    oy = max(0, min(80, oy))
    c.a[:] = sky_canvas().a[oy:oy + H]
    fx.stars(c, f, seed=18, n=70, region=(0, 0, W, 105),
             colors=('white', 'light_sky', 'lilac', 'baby_blue'))


_AY = np.arange(0, 130, dtype=np.float32)[:, None]
_AX = np.arange(W, dtype=np.float32)[None, :]
_AUR_STOPS_U = np.array([0.0, 0.18, 0.5, 0.8, 1.0], np.float32)
_AUR_COLS = np.array([col('#b8ffd0'), col('#5dfca4'), col('turquoise'), col('#7a6cf0'), col('#9d4fd8')],
                     np.float32)
_CURTAINS = (   # base y, amp, length, phase, speed, strength, x-envelope freq
    (58, 9, 42, 0.0, 1.0, 1.0, 0.011),
    (36, 6, 26, 2.1, 0.8, 0.7, 0.016),
    (78, 7, 30, 4.0, 1.25, 0.55, 0.009),
)


def _rainbow(h):
    r = np.clip(np.abs(h * 6 - 3) - 1, 0, 1)
    g = np.clip(2 - np.abs(h * 6 - 2), 0, 1)
    b = np.clip(2 - np.abs(h * 6 - 4), 0, 1)
    return np.stack([r, g, b], -1) * 200 + 55


def draw_aurora(c, f, camx=0, camy=0, flare=0.0, k=1.0):
    """Aurora curtains: ribbons with vertical rays, bottom edge waving (per-column sines)."""
    oy = int(round(-camy * 0.12))
    xs = _AX + camx * 0.08
    tot = np.zeros((130, W, 3), np.float32)
    for i, (by, amp, ln, ph, sp, st, ef) in enumerate(_CURTAINS):
        t = f * sp
        yb = (by + amp * np.sin(xs * 0.021 + t * 0.035 + ph)
              + amp * 0.5 * np.sin(xs * 0.051 - t * 0.047 + ph * 1.7)
              + 2.0 * np.sin(xs * 0.13 + t * 0.09 + ph))
        env = np.clip(np.sin(xs * ef + ph * 0.7 + t * 0.006) * 1.6 + 0.9, 0, 1)
        ray = 0.55 + 0.45 * np.sin(xs * 0.55 + 2.4 * np.sin(xs * 0.07 + t * 0.05 + ph) + t * 0.04)
        ray = np.round(ray * 3) / 3
        L = ln * (0.55 + 0.45 * ray) * (1 + 0.5 * flare)
        u = (yb - _AY) / L                         # 0 at the bottom edge, 1 at the top
        inside = (u >= -0.06) & (u <= 1.0)
        inten = np.where(u < 0, 1 + u * 12, np.clip(1 - u, 0, 1) ** 1.3) * env * (0.45 + 0.55 * ray) * st
        inten = np.where(inside, np.clip(inten, 0, 1), 0)
        uc = np.clip(u, 0, 1)
        colr = np.stack([np.interp(uc, _AUR_STOPS_U, _AUR_COLS[:, j]) for j in range(3)], -1)
        if flare > 0:
            hue = (xs / 260.0 + _AY / 200.0 - f * 0.012 + i * 0.2) % 1.0
            rb = _rainbow(np.broadcast_to(hue, uc.shape))
            colr = colr + (rb - colr) * min(1.0, flare)
        tot += colr * inten[..., None]
    gain = 0.5 * k * (1 + 0.6 * flare)
    lvl = np.clip(np.floor(np.clip(tot.max(-1) / 255.0, 0, 1) * 5 * gain * 1.6
                           + (bayer(130, W) - 0.5) * 0.9) / 5.0, 0, 1)
    norm = tot / np.maximum(1, tot.max(-1, keepdims=True)) * 255.0
    add = norm * lvl[..., None] * 0.8
    y0 = max(0, oy)
    y1 = min(H, 130 + oy)
    if y1 <= y0:
        return
    reg = c.a[y0:y1].astype(np.float32) + add[y0 - oy:y1 - oy]
    c.a[y0:y1] = np.clip(reg, 0, 255).astype(np.uint8)


def draw_moon(c, x, y):
    fx.glow(c, x, y, 22, 'periwinkle', strength=0.22, rings=4)
    c.circle(x, y, 6, mix('cloud', 'baby_blue', 0.3))
    c.circle(x + 1, y - 1, 5, 'cloud')
    c.circle(x + 1, y - 1, 3, mix('cloud', 'white', 0.6))
    for dx, dy in ((-3, 1), (-1, 3), (-2, -2)):
        c.px(x + dx, y + dy, mix('cloud_dk', 'periwinkle', 0.3))


# ================================================================== far: Everhoof + ranges + plateau
@lru_cache(maxsize=None)
def everhoof_layer():
    cv = layer()
    a = cv.a
    ax, ay = 150, 14
    lx, rx, fx_, by = 18, 292, 170, 126
    yy, xx = np.mgrid[0:LH, 0:LW].astype(np.float32)
    X, Y = xx - LM, yy - LT
    t = np.clip((Y - ay) / (by - ay), 0, 1)
    xl = ax + (lx - ax) * t
    xr = ax + (rx - ax) * t
    xf = ax + (fx_ - ax) * t
    inside = (Y >= ay) & (Y <= by) & (X >= xl) & (X <= xr)
    left = inside & (X < xf)
    right = inside & (X >= xf)
    # strata: wobbly horizontal bands, more rock lower down; snow gullies streak down the lit face
    wob = Y + 2.2 * np.sin(X * 0.05 + 1.3) + 1.3 * np.sin(X * 0.13)
    band = np.sin(wob * 0.45 + 0.8 * np.sin(Y * 0.05))
    smooth_n = np.sin(X * 0.071 + Y * 0.03) * 0.5 + np.sin(X * 0.023 - 0.7 + Y * 0.02) * 0.5
    rock = (band > 0.9 - 0.35 * t) & (smooth_n > 0.1 - 0.5 * t)
    gully = np.sin((X - xf) * 0.55 - (Y - ay) * 0.22) > 0.82
    pal = {k: np.array(col(v), np.uint8) for k, v in dict(
        sh_snow=mix('indigo', 'periwinkle', 0.2), sh_rock=mix('indigo', 'navy', 0.3),
        lt_snow=mix('periwinkle', 'navy', 0.2), lt_rock=mix('indigo', 'navy', 0.15),
        lt_hi=mix('baby_blue', 'periwinkle', 0.4)).items()}
    a[left] = pal['sh_snow']
    a[right] = pal['lt_snow']
    a[left & rock] = pal['sh_rock']
    a[right & rock] = pal['lt_rock']
    a[right & gully & ~rock & (t > 0.08)] = pal['lt_hi']
    # snow highlight just above each lit rock band
    up = np.zeros_like(rock)
    up[:-1] = rock[1:] & ~rock[:-1]
    a[right & up] = pal['lt_hi']
    o = OC(cv)
    o.line(ax, ay, fx_, by, mix('baby_blue', 'periwinkle', 0.3))
    o.px(ax, ay, 'white')
    o.dither(lx - 10, by - 22, rx - lx + 20, 22, mix('indigo', 'deep_teal', 0.45), 0.3)
    o.dither(lx - 10, by - 11, rx - lx + 20, 11, mix('indigo', 'deep_teal', 0.45), 0.55)
    return to_rgba(cv)


@lru_cache(maxsize=None)
def ranges_layer():
    cv = layer()
    a = cv.a
    xs = np.arange(LW) - LM
    prof = (100 - 18 * np.abs(np.sin(xs * 0.035 + 1.0)) - 10 * np.abs(np.sin(xs * 0.083 + 0.3))
            - 4 * np.abs(np.sin(xs * 0.21)))
    # keep the centre low (Everhoof stands there)
    mid = np.clip(1 - np.abs(xs - 160) / 120.0, 0, 1)
    prof = prof + mid * 22
    base = 124
    dp = np.gradient(prof)
    lit = dp > 0.15                     # slopes facing right (towards the moon)
    rock_d, rock_l = col(mix('navy', '#141a3a', 0.4)), col(mix('indigo', 'navy', 0.45))
    snow_d, snow_l = col(mix('indigo', 'periwinkle', 0.15)), col(mix('periwinkle', 'baby_blue', 0.25))
    for i in range(LW):
        y0 = int(prof[i]) + LT
        if y0 >= base + LT:
            continue
        hgt = base - prof[i]
        cap = int(hgt * 0.45)
        a[y0:base + LT, i] = rock_l if lit[i] else rock_d
        a[y0:y0 + cap, i] = snow_l if lit[i] else snow_d
        # streaky snow below the cap
        for k in range(cap, int(hgt), 3):
            if hash01(i // 2, k) < 0.35:
                a[y0 + k, i] = snow_l if lit[i] else snow_d
        a[y0, i] = col(mix('cloud', 'baby_blue', 0.5)) if lit[i] else col(mix('periwinkle', 'navy', 0.3))
    o = OC(cv)
    o.dither(-LM, base - 12, LW, 12, mix('indigo', 'deep_teal', 0.5), 0.4)
    # crystal outcrops glinting on the ranges
    for k, x in enumerate((-30, 10, 58, 250, 300, 345, 390)):
        y = int(prof[x + LM]) + 6
        rp = hue_ramp(('teal', 'pink', 'blue')[k % 3])
        gem_shaft(o, x, y + 3, y - 1, 3, 2, 3, rp, glow=False)
    return to_rgba(cv)


@lru_cache(maxsize=None)
def plateau_layer():
    cv = layer()
    o = OC(cv)
    hz = 120
    o.gradient(-LM, hz, LW, LH - LT - hz, [mix('indigo', 'deep_teal', 0.3), mix('periwinkle', 'navy', 0.5),
                                          mix('crystal', 'navy', 0.55), mix('crystal_dk', 'navy', 0.55)], soft=0.8)
    o.hline(-LM, LW, hz, mix('periwinkle', 'navy', 0.25))
    # drifts: soft moonlit crests with blue shadows
    for k in range(9):
        y0 = hz + 4 + k * 5 + k * k * 0.5
        for x in range(-LM, LW - LM, 2):
            yy = y0 + 1.6 * math.sin(x * 0.045 + k * 1.7) + 0.8 * math.sin(x * 0.13 + k)
            if math.sin(x * 0.02 + k * 2.3) > -0.2:
                o.px(x, yy, mix('periwinkle', 'baby_blue', 0.3))
                o.px(x + 1, yy, mix('periwinkle', 'baby_blue', 0.3))
                o.px(x, yy + 1, mix('crystal_dk', 'navy', 0.5))
    return to_rgba(cv)


# ================================================================== the palace (parametric)
def palace(o, cx, y0, s, lit=False, wins=None, detail=1):
    """Ice-blue crystal-spire palace on crystal arches; y0 = platform top under the arches."""
    rp = hue_ramp('palace', lit)
    rpk = hue_ramp('pink', lit)
    S = lambda v: v * s
    # ---- arches / base block
    dk_y = y0 - S(20)
    o.poly([(cx - S(30), y0 + 0.5), (cx - S(27), dk_y), (cx + S(27), dk_y), (cx + S(30), y0 + 0.5)], rp['ol'])
    o.poly([(cx - S(29) + 0.5, y0), (cx - S(26) + 0.5, dk_y + 1), (cx + S(26) - 0.5, dk_y + 1),
            (cx + S(29) - 0.5, y0)], rp['md'])
    # leg facets (dark left / light right halves)
    for lx0, lx1 in ((-29, -21), (-13, -8), (8, 13), (21, 29)):
        a_, b_ = cx + S(lx0), cx + S(lx1)
        m_ = (a_ + b_) / 2
        o.poly([(a_ + 0.5, y0), (a_ + S(1.5), dk_y + 1), (m_, dk_y + 1), (m_, y0)], rp['dk'])
        o.poly([(m_, y0), (m_, dk_y + 1), (b_ - S(0.5), dk_y + 1), (b_ - 0.5, y0)], rp['lt'])
        o.line(b_ - 1, y0 - 1, b_ - S(0.5) - 1, dk_y + 2, rp['hi'])
    # cut the arch openings (pointed arches)
    K = KEY
    o.poly([(cx - S(8), y0 + 1), (cx - S(8), y0 - S(11)), (cx, y0 - S(18)), (cx + S(8), y0 - S(11)),
            (cx + S(8), y0 + 1)], K)
    for sgn in (-1, 1):
        a_, b_ = cx + sgn * S(13), cx + sgn * S(21)
        lo, hi_ = min(a_, b_), max(a_, b_)
        o.poly([(lo, y0 + 1), (lo, y0 - S(8)), ((lo + hi_) / 2, y0 - S(13)), (hi_, y0 - S(8)),
                (hi_, y0 + 1)], K)
    # flared crystal feet under each leg
    for lx0, lx1 in ((-29, -21), (-13, -8), (8, 13), (21, 29)):
        a_, b_ = cx + S(lx0), cx + S(lx1)
        fw = S(2)
        o.poly([(a_ - fw, y0 + 0.5), (a_ + 0.5, y0 - S(4)), (b_ - 0.5, y0 - S(4)), (b_ + fw, y0 + 0.5)], rp['ol'])
        o.poly([(a_ - fw + 1, y0), (a_ + 1, y0 - S(4) + 1), ((a_ + b_) / 2, y0 - S(4) + 1), ((a_ + b_) / 2, y0)], rp['dk'])
        o.poly([((a_ + b_) / 2, y0), ((a_ + b_) / 2, y0 - S(4) + 1), (b_ - 1, y0 - S(4) + 1), (b_ + fw - 1, y0)], rp['lt'])
        o.hline(a_ + 1, b_ - 1, y0 - S(4), rp['hi'])
    # arch rims (light inner edge)
    o.line(cx - S(8) - 1, y0 - S(11), cx - 1, y0 - S(18) - 1, rp['hi'])
    o.line(cx + S(8), y0 - S(11), cx, y0 - S(18) - 1, rp['ol'])
    # deck with hanging icicle points
    dh = max(2, S(5))
    o.rect(cx - S(31), dk_y - dh, S(62), dh + 1, rp['ol'])
    o.rect(cx - S(30), dk_y - dh + 1, S(60), dh - 1, rp['md'])
    o.rect(cx + S(6), dk_y - dh + 1, S(24), dh - 1, rp['lt'])
    o.hline(cx - S(30), cx + S(30), dk_y - dh + 1, rp['hi'])
    n_ic = int(60 * s / max(2.5, 4 * s)) if s >= 1.5 else 8
    for i in range(n_ic + 1):
        ix = cx - S(30) + i * S(60) / n_ic
        if abs(ix - cx) < S(8) or (S(13) < abs(ix - cx) < S(21)):
            ln = S(3.5) if i % 2 == 0 else S(2)
            o.poly([(ix - S(1.4), dk_y), (ix + S(1.4), dk_y), (ix, dk_y + ln)], rp['lt'] if ix > cx else rp['md'])
    top = dk_y - dh
    # ---- crown of small spikes around the base of the shaft
    for sx, hh, ww in ((-26, 9, 4), (26, 9, 4), (-20, 15, 5), (20, 15, 5)):
        gem_shaft(o, cx + S(sx), top + 1, top - S(hh) + S(ww) * 0.9, S(ww), S(ww) * 0.8, S(ww) * 0.9, rp,
                  glow=False)
    # ---- side spires
    for sgn in (-1, 1):
        sx = cx + sgn * S(12.5)
        gem_shaft(o, sx, top + 1, top - S(40), S(8), S(6), S(13), rp)
        o.hline(sx - S(4.5), sx + S(4.5), top - S(22), rp['ol'])
        o.hline(sx - S(4), sx + S(4), top - S(23), rp['hi'])
        if s >= 0.8:
            window(o, sx - S(1), top - S(16), max(1, S(2)), max(2, S(5)), rp, wins)
            if s > 1.5:
                window(o, sx - S(1), top - S(34), S(2), S(4), rp, wins)
    # ---- central shaft
    sh_top = top - S(58)
    gem_shaft(o, cx, top + 1, sh_top, S(15), S(9), 1, rp)
    for by_ in (S(14), S(34), S(52)):         # ring balconies
        yy = top - by_
        frac = by_ / S(58)
        hw = S(7.5) - S(3) * frac + S(1.6)
        bt_ = yy - max(1, S(1.4))
        bh_ = max(2, S(2.2))
        o.rect(cx - hw, bt_, 2 * hw + 1, bh_, rp['ol'])
        if detail > 1:
            o.rect(cx - hw + 1, bt_ + 1, 2 * hw - 1, bh_ - 2, rp['md'])
            o.rect(cx + 1, bt_ + 1, hw - 1, bh_ - 2, rp['lt'])
            o.hline(cx - hw + 1, cx + hw - 1, bt_ + 1, rp['hi'])
            for k in range(int(2 * hw // 4)):          # little icicles under the ring
                ix = cx - hw + 2 + k * 4
                o.poly([(ix - 1, bt_ + bh_ - 1), (ix + 2, bt_ + bh_ - 1), (ix + 0.5, bt_ + bh_ + 2)],
                       rp['lt'] if ix > cx else rp['md'])
            for sgn in (-1, 1):                         # crystal shards bursting out of the ring
                ex = cx + sgn * hw
                o.poly([(ex - sgn * 1, bt_ + 1), (ex + sgn * S(3.5), bt_ - S(4)), (ex + sgn * S(1.5), bt_ + 2)],
                       rp['ol'])
                o.poly([(ex, bt_ + 1), (ex + sgn * S(3.5) - sgn, bt_ - S(4) + 1), (ex + sgn * S(1.5), bt_ + 1)],
                       rp['lt'] if sgn > 0 else rp['md'])
        else:
            o.hline(cx - hw + 1, cx + hw - 1, bt_ + 1, rp['lt'])
            o.hline(cx - hw + 1, cx - 1, bt_ + 1, rp['md'])
            o.hline(cx, cx + hw - 1, bt_, rp['hi'])
    # tall windows between the balconies
    for wy, wh_ in ((top - S(12), S(8)), (top - S(32), S(10)), (top - S(50), S(8))):
        ww = max(1, S(2.2))
        if s < 1.2:
            window(o, cx - ww / 2 + 0.5, wy - wh_, ww, wh_, rp, wins, arch=False)
        else:
            for ddx in (-S(2.6), S(2.6) - ww):
                window(o, cx + ddx, wy - wh_, ww, wh_, rp, wins)
    # crown gem (rose window) and the needle
    cb = sh_top + 1
    gem_bulb(o, cx, cb, cb - S(6), cb - S(14), S(9), S(14), rp)
    rr = max(1, S(2.2))
    o.ellipse(cx, cb - S(6), rr + 1, rr + 1, rp['ol'])
    o.ellipse(cx, cb - S(6), rr, rr, rpk['lt'] if not lit else 'white')
    if s > 1.2:
        o.px(cx, cb - S(6), 'white')
    nb = cb - S(13)
    gem_shaft(o, cx, nb, nb - 1, S(5), S(4), S(24), rp)
    o.vline(cx + 1, nb - S(16), nb - S(4), rp['hi'])
    if detail > 1:  # striations on big facets
        for k in range(30):
            yy = top - hash01(k, 3) * S(56)
            xx = cx - S(5) + hash01(k, 4) * S(10)
            o.hline(xx, xx + S(1.5), yy, rp['lt'] if xx > cx else rp['md'])
    return dict(top=top, tipy=nb - S(24), side_tip=top - S(53), heart_y=y0 - S(9), dk_y=dk_y)


# ================================================================== the Crystal Heart
_HEART_CACHE = {}


def _heart_mask(size, wf):
    key = (size, round(wf, 2))
    if key in _HEART_CACHE:
        return _HEART_CACHE[key]
    n = 2 * size + 3
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    u = (xs - (n - 1) / 2) / (size * wf)
    v = -(ys - (n - 1) / 2 - size * 0.12) / size
    m = (u * u + v * v - 1) ** 3 - u * u * v ** 3 <= 0
    _HEART_CACHE[key] = (m, u, v)
    return _HEART_CACHE[key]


_HP = [col(v) for v in (mix('crystal_pk', 'plum', 0.4), mix('crystal_pk', 'magic', 0.55), 'crystal_pk',
                        mix('crystal_pk', 'white', 0.65))]
_HL = [col(v) for v in (mix('lavender', 'royal', 0.35), 'lavender', mix('lilac', 'crystal_pk', 0.3),
                        mix('lilac', 'white', 0.65))]
_HB = [col(v) for v in (mix('crystal', 'royal', 0.35), 'crystal', 'crystal_lt', mix('crystal_lt', 'white', 0.65))]
_HRAMP = np.array([_HP, _HL, _HB], np.float32)


def draw_heart(c, cx, cy, size, f, glow_k=1.0, lit=0.0):
    """The Crystal Heart, centred at (cx, cy), half-size `size` px. A cut gem: 9 facets around a
    heart-shaped table; it turns slowly (facets advance every 3 frames), pink top -> blue point."""
    step = f // 3
    ph = step * 0.07                                   # turning phase (radians of facet rotation)
    wf = 0.84 + 0.16 * abs(math.cos(step * 0.035))
    wf = max(0.6, round(wf * size) / size)
    m, u, v = _heart_mask(int(size), wf)
    n = m.shape[0]
    ang = np.arctan2(v - 0.12, u / max(0.5, wf) * wf)
    nf = 9
    sect = np.floor(((ang + ph) / (2 * np.pi) * nf) % nf)
    ac = (sect + 0.5) * (2 * np.pi / nf) - ph
    lvl = 0.5 + 0.5 * np.cos(ac - 0.9)                 # light from the upper right
    lvl = 1 + np.clip((lvl * 2.99).astype(int), 0, 2)
    rad = np.sqrt((u / 0.62) ** 2 + ((v - 0.12) / 0.6) ** 2)
    table = rad < 0.62
    lvl = np.where(table, np.where(u + (v - 0.12) * 0.6 > 0.1 * math.sin(step * 0.2), 3, 2), lvl)
    vt = np.clip((0.95 - v) / 2.1, 0, 1)               # 0 top -> 1 bottom point
    thr = bayer(n, n, int(cx), int(cy))
    rk = np.clip((vt * 2.6 - 0.2 + (thr - 0.5) * 0.5).astype(int), 0, 2)
    rgb = _HRAMP[rk, lvl]
    if lit > 0:
        rgb = rgb + (255 - rgb) * min(1.0, lit)
    pad = np.pad(m, 1)
    edge = m & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    rgb[edge] = col(mix('plum', 'royal', 0.35)) if lit < 0.5 else col('cotton')
    if size >= 8:     # facet seams (darker lines) between sectors on the rim
        sh = np.zeros_like(m)
        sh[:, 1:] = sect[:, 1:] != sect[:, :-1]
        sh[1:, :] |= sect[1:, :] != sect[:-1, :]
        seam = sh & m & ~edge & ~table
        rgb[seam] = rgb[seam] * 0.82
        trim = table & ~(np.pad(table, 1)[:-2, 1:-1] & np.pad(table, 1)[2:, 1:-1])
        rgb[trim & m] = rgb[trim & m] * 0.85 + 255 * 0.15
    hl = (np.abs(u + 0.42) < 0.13 * max(1, 8 / size)) & (np.abs(v - 0.62) < 0.17) & m & ~edge
    rgb[hl] = col('white')
    x0, y0 = int(cx) - n // 2, int(cy) - n // 2
    ys0, xs0 = max(0, -y0), max(0, -x0)
    sub = c.a[max(0, y0):y0 + n, max(0, x0):x0 + n]
    mm = m[ys0:ys0 + sub.shape[0], xs0:xs0 + sub.shape[1]]
    rr = rgb[ys0:ys0 + sub.shape[0], xs0:xs0 + sub.shape[1]]
    sub[mm] = np.clip(rr[mm], 0, 255).astype(np.uint8)
    if size >= 6:
        gx = int(cx + size * wf * 0.5 * math.sin(step * 0.09))
        fx.sparkle(c, gx, int(cy - size * 0.3), f, 'white', period=27, size=2 if size > 10 else 1)


# ================================================================== wide city layers
def _ring_houses(seed, n, x_in, x_out, hmin, hmax, wmin, wmax, kinds):
    """House specs (dx, w, h, hue, kind) on both sides of the centre for one ring."""
    out = []
    hues = ('pink', 'lilac', 'teal', 'blue', 'violet', 'pink', 'blue', 'lilac')
    for sgn in (-1, 1):
        x = x_in
        i = 0
        while x < x_out and i < n:
            r = hash01(seed, sgn, i)
            w = int(wmin + (wmax - wmin) * r)
            t = (x - x_in) / max(1, x_out - x_in)                 # 0 near centre -> 1 outside
            h = int(hmax - (hmax - hmin) * t * (0.7 + 0.3 * hash01(seed, i, 9)))
            hue = hues[int(hash01(seed, sgn, i, 4) * len(hues))]
            kind = kinds[int(hash01(seed, sgn, i, 5) * len(kinds))]
            out.append((sgn * (x + w / 2), w, h, hue, kind, seed * 100 + sgn * 10 + i))
            x += w * (0.78 + 0.25 * hash01(seed, i, 6))
            i += 1
    # draw far (outer) ones first so inner/taller overlap nicely
    out.sort(key=lambda s: -abs(s[0]))
    return out


def ring_centre(par):
    """Layer-local world x of the city centre for a layer of parallax `par`."""
    return PX - P_IN * CREF + par * CREF


@lru_cache(maxsize=None)
def inner_layer(lit=False):
    cv = layer()
    o = OC(cv)
    wins = []
    cx = ring_centre(P_IN)
    rpw = hue_ramp('blue', lit)
    # platform: top PAL_BASE, two steps
    o.rect(cx - 66, PAL_BASE, 132, 12, rpw['ol'])
    terrace(o, cx - 64, cx + 64, PAL_BASE, 9, rpw, 3)
    terrace(o, cx - 88, cx + 88, PAL_BASE + 9, 8, rpw, 4)
    o.rect(cx - 88, PAL_BASE + 17, 176, 40, mix('crystal', 'navy', 0.4))
    # lower-step houses then top houses
    for dx, w, h, hue, kind, sd in _ring_houses(11, 2, 66, 86, 9, 12, 7, 9, ('gem', 'bulb')):
        house(o, cx + dx, PAL_BASE + 9, w, h, hue, kind, sd, lit, wins)
    for dx, w, h, hue, kind, sd in _ring_houses(12, 4, 31, 62, 11, 20, 7, 10, ('gem', 'tower', 'bulb')):
        house(o, cx + dx, PAL_BASE, w, h, hue, kind, sd, lit, wins)
    info = palace(o, cx, PAL_BASE, PAL_S, lit, wins)
    return to_rgba(cv), tuple(wins), info


@lru_cache(maxsize=None)
def mid_layer(lit=False):
    cv = layer()
    o = OC(cv)
    wins = []
    cx = ring_centre(P_MID)
    y = 134
    rpw = hue_ramp('lilac', lit)
    o.rect(cx - 124, y, 248, 60, mix('crystal', 'navy', 0.38))
    terrace(o, cx - 122, cx + 122, y, 8, rpw, 5)
    for sgn in (-1, 1):                      # snow slopes at the terrace ends
        ex = cx + sgn * 122
        o.poly([(ex, y - 1), (ex + sgn * 12, y + 14), (ex, y + 14)], mix('periwinkle', 'baby_blue', 0.3))
        o.line(ex, y - 1, ex + sgn * 12, y + 14, 'cloud')
    # grand stair up to the palace platform in the avenue gap
    rps = hue_ramp('palace', lit)
    for k in range(9):
        yy = y - 2 - k * 2
        hw = 13 - k * 0.6
        o.rect(cx - hw, yy, 2 * hw + 1, 2, rps['dk'])
        o.hline(cx - hw + 1, cx + hw - 1, yy, rps['hi'] if k % 2 == 0 else rps['lt'])
        o.px(cx - hw, yy, rps['ol'])
        o.px(cx + hw, yy, rps['ol'])
    for dx, w, h, hue, kind, sd in _ring_houses(21, 7, 17, 122, 12, 25, 9, 14, ('gem', 'tower', 'bulb', 'twin')):
        house(o, cx + dx, y, w, h, hue, kind, sd, lit, wins)
    for lx in (-15, 15, -60, 60, -100, 100):
        lamp(o, cx + lx, y, 6)
    return to_rgba(cv), tuple(wins)


@lru_cache(maxsize=None)
def outer_layer(lit=False):
    cv = layer()
    o = OC(cv)
    wins = []
    cx = ring_centre(P_OUT)
    y = 150
    # snow ground in front of the city (full width)
    o.gradient(-LM, y - 1, LW, LH - LT - y + 1, [mix('periwinkle', 'baby_blue', 0.3), mix('periwinkle', 'navy', 0.25),
                                                mix('crystal_dk', 'navy', 0.25), mix('crystal_dk', 'navy', 0.5)], soft=0.7)
    o.hline(-LM, LW, y - 1, mix('cloud', 'baby_blue', 0.4))
    # avenue path down from the gate (perspective)
    for yy in range(y, LH - LT):
        hw = 6 + (yy - y) * 0.5
        o.hline(cx - hw, cx + hw, yy, mix('periwinkle', 'crystal_lt', 0.2) if (yy - y) % 5 else mix('periwinkle', 'crystal_lt', 0.4))
        o.px(cx - hw, yy, mix('cloud', 'periwinkle', 0.3))
        o.px(cx + hw, yy, mix('cloud', 'periwinkle', 0.3))
    for k, yy in enumerate(range(y + 6, LH - LT, 9)):        # little lamp posts along the avenue
        hw = 6 + (yy - y) * 0.5 + 3
        for sgn in (-1, 1):
            o.vline(cx + sgn * hw, yy - 4 - k, yy, mix('crystal_dk', 'ink', 0.3))
            o.rect(cx + sgn * hw - 1, yy - 6 - k, 3, 2, 'cotton')
    for dx, w, h, hue, kind, sd in _ring_houses(31, 7, 20, 190, 15, 32, 12, 20, ('gem', 'tower', 'bulb', 'twin')):
        house(o, cx + dx, y, w, h, hue, kind, sd, lit, wins)
        o.ellipse(cx + dx, y + 1, w * 0.6, 1.5, 'cloud')
    # city gate: two crystal pylons + a heart emblem
    rpg = hue_ramp('pink', lit)
    for sgn in (-1, 1):
        gem_shaft(o, cx + sgn * 11, y + 1, y - 16, 5, 4, 5, hue_ramp('blue', lit))
    o.line(cx - 10, y - 18, cx, y - 23, rpg['md'])
    o.line(cx + 10, y - 18, cx, y - 23, rpg['md'])
    o.rect(cx - 2, y - 26, 2, 2, rpg['lt'])
    o.rect(cx + 1, y - 26, 2, 2, rpg['lt'])
    o.rect(cx - 1, y - 25, 3, 2, rpg['lt'])
    o.px(cx, y - 23, rpg['lt'])
    return to_rgba(cv), tuple(wins)


@lru_cache(maxsize=None)
def fg_layer():
    cv = layer()
    o = OC(cv)
    yb = 170
    for x0, x1 in ((-LM, 60), (300, LW - LM)):
        for x in range(x0, x1):
            t = min(x - x0, x1 - x) / 25.0
            yy = yb - min(1.0, t) * 6 - 2 * math.sin(x * 0.09)
            o.vline(x, yy, LH - LT, mix('periwinkle', 'baby_blue', 0.35))
            o.px(x, yy, 'cloud')
            o.vline(x, yy + 3, LH - LT, mix('crystal_dk', 'periwinkle', 0.4))
            o.vline(x, yy + 7, LH - LT, mix('crystal_dk', 'navy', 0.3))
    # crystal shard clusters
    for x, hue, hh in ((14, 'pink', 18), (24, 'teal', 11), (6, 'lilac', 9), (322, 'blue', 20),
                       (334, 'pink', 12), (312, 'teal', 9), (380, 'lilac', 16)):
        gem_shaft(o, x, yb - 2, yb - hh + 4, 6 if hh > 12 else 4, 4, 5, hue_ramp(hue))
    return to_rgba(cv)


# ================================================================== per-frame wide-city overlays
def win_flicker(c, wins, sx, sy, f, seed, rate=0.05):
    """Some windows switch off/on now and then (each a 1-2 s state)."""
    for i, (x, y, w, h) in enumerate(wins):
        r = hash01(seed, i, f // 20 + i % 7)
        if r < rate:
            c.rect(x + sx, y + sy, w, h, mix('crystal_dk', 'navy', 0.3))
        elif r > 1 - rate * 0.6:
            c.rect(x + sx, y + sy, w, h, 'white')


def dome(c, f, cx, cy, rx, ry, k=1.0):
    """Protective dome: a thin, rippling, dithered arc of light with a glassy sheen."""
    th = np.linspace(0.02, math.pi - 0.02, 1100)
    rip = np.round(1.2 * np.sin(th * 14 + f * 0.18))

    def ring(d, on, color, kk):
        xs = np.round(cx + (rx - d + rip) * np.cos(th)).astype(int)
        ys = np.round(cy - (ry - d + rip) * np.sin(th)).astype(int)
        ok = on & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        pts = np.unique(np.stack([ys[ok], xs[ok]]), axis=1)
        add_light(c, (pts[0], pts[1]), color, kk)
    wav = np.sin(th * 46 - f * 0.22) + np.sin(th * 9 + f * 0.07) * 0.8
    ring(0, wav > -0.9, 'crystal_lt', 0.22 * k)
    ring(0, wav > 0.6, 'crystal_lt', 0.3 * k)
    spec = np.abs(th - (1.0 + 0.15 * math.sin(f * 0.03))) < 0.22
    ring(1, spec, 'white', 0.35 * k)
    ring(2, spec & (np.abs(th - 1.0) < 0.15), 'white', 0.18 * k)
    for d in (3, 5, 8):
        on2 = (np.sin(th * 3 - f * 0.06 + d) > 0.2) & (np.arange(len(th)) % (d // 2 + 2) == 0)
        ring(d, on2, 'ice', 0.2 * k * 3 / d)


def snow_glitter(c, f, seed, region, n, camx, camy, par):
    x0, y0, w, h = region
    for i in range(n):
        x = int(x0 + hash01(seed, i, 1) * w - camx * par)
        x = (x % (W + 40)) - 20
        y = int(y0 + hash01(seed, i, 2) * h - camy * par)
        ph = hash01(seed, i, 3)
        t = ((f + ph * 60) % 60) / 60.0
        if t < 0.12:
            c.px(x, y, 'white')
            if t > 0.04 and t < 0.08 and hash01(seed, i, 5) > 0.5:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.px(x + dx, y + dy, 'crystal_lt')


def snowfall(c, f, n=40, seed=5, speed=0.6, camx=0, camy=0, region=(0, 0, W, H)):
    x0, y0, w, h = region
    for i in range(n):
        sp = speed * (0.6 + 0.8 * hash01(seed, i, 1))
        big = hash01(seed, i, 4) > 0.75
        y = (hash01(seed, i, 2) * (h + 20) + f * sp - camy * (1.2 if big else 0.8)) % (h + 20) - 10 + y0
        x = x0 + (hash01(seed, i, 3) * (w + 20) + 3 * math.sin(f * 0.05 + i) - camx * (1.2 if big else 0.8)) % (w + 20) - 10
        c.px(int(x), int(y), 'white' if big else 'cloud')
        if big:
            c.px(int(x) + 1, int(y), 'cloud')


def pennant(c, x, y, f, size=1, color='crystal_pk'):
    """Small waving pennant attached at its pole top (x, y), blowing to the right."""
    ph = (f // 3) % 4
    L = 4 * size
    for i in range(L):
        dy = int(round(math.sin((i * 0.9 - ph * 1.5)) * 0.8 * size * (i / L)))
        hh = max(1, int(round((L - i) / L * 2.2 * size)))
        c.rect(x + 1 + i, y + dy, 1, hh, col(color) if i < L - 1 else mix(color, 'white', 0.3))
    c.vline(x, y - 1, y + 3 * size, mix('crystal_dk', 'ink', 0.2))


def city_frame(f, camx, camy, lit_fn=None, flare=0.0, aur_k=1.0, balloon=None, heart_k=1.0, heart_glow=0.0, fg=True):
    """The wide city: all layers composited. lit_fn(par) -> per-column lit mask (or None)."""
    c = Canvas()
    draw_sky(c, f, camx, camy)
    draw_aurora(c, f, camx, camy, flare, aur_k)
    draw_moon(c, int(round(244 - camx * 0.04)), int(round(32 - camy * 0.05)))
    place(c, everhoof_layer(), camx, camy, P_EVER)
    # spindrift off the summit
    sx, sy = 150 - int(round(camx * P_EVER)), 12 - int(round(camy * P_EVER))
    for i in range(10):
        t = ((f * 0.8 + i * 9) % 70) / 70.0
        x = sx + int(t * 34)
        y = sy + int(t * 8 + math.sin(t * 9 + i) * 1.5)
        if hash01(i, f // 4) < 1 - t:
            c.px(x, y, mix('cloud', 'periwinkle', 0.4 * t))
    place(c, ranges_layer(), camx, camy, P_RANGE)
    place(c, plateau_layer(), camx, camy, P_PLAT)
    snow_glitter(c, f, 3, (0, 125, W + 40, 30), 26, camx, camy, P_PLAT)
    if balloon:
        bx, by, bsz = balloon
        draw_balloon(c, bx, by, f, bsz)
    hx = int(round(ring_centre(P_IN) - camx * P_IN))
    hy0 = -int(round(camy * P_IN))
    for name, fn, par in (('in', inner_layer, P_IN), ('mid', mid_layer, P_MID), ('out', outer_layer, P_OUT)):
        res0 = fn(False)
        sxl = -int(round(camx * par))
        syl = -int(round(camy * par))
        if lit_fn is None:
            c.blit(res0[0], sxl - LM, syl - LT)
        else:
            m = lit_fn(par)
            if m is True:
                res0 = fn(True)
                c.blit(res0[0], sxl - LM, syl - LT)
            else:
                res1 = fn(True)
                x0 = LM - sxl
                y0 = LT - syl
                a0 = res0[0][y0:y0 + H, x0:x0 + W]
                a1 = res1[0][y0:y0 + H, x0:x0 + W]
                comp = np.where(m[None, :, None], a1, a0)
                c.blit(comp, 0, 0)
                if m.any():
                    res0 = res1 if m.mean() > 0.5 else res0
        win_flicker(c, res0[1], sxl, syl, f, {"in": 1, "mid": 2, "out": 3}[name])
        if name == 'in':
            info = res0[2]
            # heart under the arches
            hy = PAL_BASE + hy0 - int(9 * PAL_S)
            fx.glow(c, hx, hy, int(16 + 30 * heart_glow), 'magic',
                    strength=0.22 + 0.06 * math.sin(f * 0.21) + 0.4 * heart_glow, rings=4)
            draw_heart(c, hx, hy + int(round(math.sin(f * 0.1))), max(3, int(round(4 * heart_k))), f,
                       lit=0.4 * heart_glow)
            # pennants on the side spires
            for sgn in (-1, 1):
                px_ = hx + sgn * int(round(12.5 * PAL_S))
                pennant(c, px_, int(info['side_tip']) + hy0 + 2, f + sgn * 5, 1,
                        'crystal_pk' if sgn < 0 else 'ice')
    # inner glow of the city + palace
    fx.glow(c, hx, PAL_BASE + hy0 + 10, 110, 'magic', strength=0.1, rings=5)
    fx.glow(c, hx, PAL_BASE + hy0 - 40, 46, 'crystal', strength=0.14, rings=4)
    if fg:
        place(c, fg_layer(), camx, camy, P_FG)
    return c, hx, hy0


# ================================================================== shot A: wide approach
def cam_a(f):
    return int(round(lerp(0, 48, cruise(seg(f, T_A, T_B + 8)))))


def render_a(f):
    camx = cam_a(f)
    bx = int(round(lerp(96, 128, seg(f, T_A, T_B))))
    c, hx, hy0 = city_frame(f, camx, 0, balloon=(bx, 74, 's'))
    dome(c, f, hx, PAL_BASE + 36, 150, 108)
    snowfall(c, f, n=14, seed=2, speed=0.35)
    # lamp glows on the mid terrace
    cm = int(round(ring_centre(P_MID) - camx * P_MID))
    for lx in (-15, 15, -60, 60, -100, 100):
        fx.glow(c, cm + lx, 134 - 8, 5, 'magic', strength=0.25 + 0.08 * math.sin(f * 0.3 + lx), rings=2)
    co = int(round(ring_centre(P_OUT) - camx * P_OUT))
    callout(c, co + 95, 128, 'CRYSTAL EMPIRE', f, 84, 156, side='up', length=22)
    atlas_card(c, f, 100, 206, 'CRYSTAL EMPIRE', ['CITY OF CRYSTAL', 'HOME OF THE HEART'])
    return c


# ================================================================== shot B: the palace close + the Heart
# Layers are drawn in "E-coords" = screen coords at the END of the tilt (camy = B_TILT).
B_TILT = 150
BOX, BOY = 30, 150          # layer margins (x drift, tilt range)
BW_, BH_ = W + 60, H + 170
PB_BACK2, PB_BACK1, PB_PAL, PB_NEAR = 0.62, 0.8, 1.0, 1.18
B_S = 2.3                   # palace scale
B_Y0 = 140                  # palace base / plaza line (E-coords)
B_HX = 160
B_HY = B_Y0 - 29             # heart centre (floats 3 px above the pedestal cup)
B_HSIZE = 13


def blayer():
    return Canvas(BW_, BH_, KEY)


def big_house(o, cx, yb, w, h, hue, lit, seed, wins, side=1):
    """A large detailed crystal house for close shots."""
    rp = hue_ramp(hue, lit)
    # side wing (towards the frame edge)
    ww, wh = int(w * 0.55), int(h * 0.5)
    wx = cx - side * (w * 0.55)
    gem_shaft(o, wx, yb, yb - wh + ww * 0.55, ww, ww * 0.9, ww * 0.55, rp)
    for k in range(2):
        window(o, wx - 2, yb - 14 - k * 12, 4, 6, rp, wins)
    # main body
    tip = int(h * 0.36)
    gem_shaft(o, cx, yb, yb - h + tip, w, w * 0.84, tip, rp)
    # crystal striations on the facets
    for k in range(int(h * 0.6)):
        yy = yb - 4 - hash01(seed, k, 1) * (h - tip - 6)
        tt = (yb - yy) / max(1, h)
        hw = w / 2 - (w * 0.08) * tt
        xx = cx - hw + 2 + hash01(seed, k, 2) * (2 * hw - 6)
        ln = 2 + int(hash01(seed, k, 3) * 4)
        cc = rp['dk'] if xx < cx - hw * 0.45 else (rp['lt'] if xx > cx + hw * 0.45 else rp['hi'] if hash01(seed, k) > 0.7 else rp['lt'])
        o.hline(xx, xx + ln, yy, cc)
    # balcony band
    by = yb - int(h * 0.42)
    o.rect(cx - w * 0.5 - 2, by, w + 4, 3, rp['ol'])
    o.hline(cx - w * 0.5 - 1, cx + w * 0.5 + 1, by, rp['hi'])
    o.hline(cx - w * 0.5 - 1, cx + w * 0.5 + 1, by + 1, rp['lt'])
    for k in range(int(w // 4)):
        o.vline(cx - w * 0.5 + 1 + k * 4, by + 3, by + 5, rp['ol'])
    o.hline(cx - w * 0.5, cx + w * 0.5, by + 6, rp['ol'])
    # windows: centre column bright, side facets dimmer
    rows = [yb - 16 - 14 * i for i in range(int((h - tip) // 14))]
    for i, wy in enumerate(rows):
        if abs(wy - by) < 9:
            continue
        window(o, cx - 3, wy - 7, 6, 8, rp, wins)
        o.vline(cx, wy - 7, wy, rp['ol'])
        o.hline(cx - 3, cx + 2, wy - 4, rp['ol'])
        for sx in (-1, 1):
            xw = cx + sx * w * 0.33
            window(o, xw - 1, wy - 5, 3, 5, rp, wins)
    # door with steps
    o.rect(cx - 5, yb - 13, 10, 13, rp['ol'])
    o.rect(cx - 4, yb - 12, 8, 12, rp['win2'])
    o.rect(cx - 4, yb - 12, 8, 5, rp['win'])
    o.px(cx - 4, yb - 12, rp['ol'])
    o.px(cx + 3, yb - 12, rp['ol'])
    o.vline(cx, yb - 11, yb - 1, rp['ol'])
    o.rect(cx - 7, yb, 14, 2, rp['hi'])
    o.rect(cx - 8, yb + 2, 16, 2, rp['lt'])
    # roof finial gem
    gem_shaft(o, cx, yb - h + tip + 2, yb - h + 2, 5, 4, 7, hue_ramp('pink' if hue != 'pink' else 'teal', lit), glow=False)


@lru_cache(maxsize=None)
def b_back_layer(which, lit=False):
    cv = blayer()
    o = OC(cv, BOX, BOY)
    wins = []
    if which == 2:      # far ring, higher terrace
        yb = 112
        o.rect(-BOX, yb, BW_, 80, mix('crystal', 'navy', 0.45))
        terrace(o, -BOX, BW_ - BOX, yb, 7, hue_ramp('blue', lit), 41)
        specs = _ring_houses(41, 9, 40, 210, 16, 38, 9, 15, ('gem', 'tower', 'bulb', 'twin'))
        for dx, w, h, hue, kind, sd in specs:
            house(o, B_HX + dx, yb, w, h, hue, kind, sd, lit, wins)
    else:               # nearer ring behind the plaza
        yb = 128
        o.rect(-BOX, yb, BW_, 80, mix('crystal', 'navy', 0.38))
        terrace(o, -BOX, BW_ - BOX, yb, 9, hue_ramp('lilac', lit), 42)
        specs = _ring_houses(42, 6, 70, 220, 26, 58, 14, 22, ('gem', 'tower', 'bulb', 'twin'))
        for dx, w, h, hue, kind, sd in specs:
            house(o, B_HX + dx, yb, w, h, hue, kind, sd, lit, wins)
        for lx in (-86, 86, -150, 150):
            lamp(o, B_HX + lx, yb, 10)
    return to_rgba(cv), tuple(wins)


_PLZ_Y, _PLZ_X = np.mgrid[0:44, 0:BW_].astype(np.float32)


@lru_cache(maxsize=None)
def b_palace_layer(lit=False):
    cv = blayer()
    o = OC(cv, BOX, BOY)
    wins = []
    # plaza floor (perspective crystal tiles), E-y B_Y0 .. B_Y0+44
    X = _PLZ_X - BOX
    Y = _PLZ_Y + B_Y0
    vx, vy = B_HX, B_Y0 - 70
    u = (X - vx) / (Y - vy) * 70.0                    # perspective column coordinate
    v = np.log(np.maximum(1.0, Y - vy)) * 14.0          # rows get taller towards the viewer
    tile = (np.floor(u / 9.0) + np.floor(v)) % 2
    seam_u = np.abs(((u / 9.0) % 1.0) - 0.5) > 0.46
    seam_v = np.abs((v % 1.0) - 0.5) > 0.42
    base0 = np.array(col(mix('crystal', 'navy', 0.38) if not lit else mix('crystal', 'white', 0.1)), np.float32)
    base1 = np.array(col(mix('crystal', 'royal', 0.3) if not lit else 'crystal'), np.float32)
    seamc = np.array(col(mix('crystal_dk', 'navy', 0.3)), np.float32)
    rgb = np.where(tile[..., None] > 0, base1, base0)
    rgb = np.where((seam_u | seam_v)[..., None], seamc, rgb)
    hi = (np.abs(((u / 9.0) % 1.0) - 0.5) > 0.4) & ~seam_u & (X > vx)
    rgb = np.where(hi[..., None], rgb * 0.7 + np.array(col('crystal_lt'), np.float32) * 0.3, rgb)
    # reflection of the palace legs + heart (light streaks under the arches)
    refl = (np.abs(X - vx) < 70) & (((X - vx) % 12) < 3) & (Y < B_Y0 + 16)
    rgb = np.where(refl[..., None], rgb * 0.75 + np.array(col('crystal_lt'), np.float32) * 0.25, rgb)
    cv.a[BOY + B_Y0:BOY + B_Y0 + 44] = rgb.astype(np.uint8)
    o.hline(-BOX, BW_ - BOX, B_Y0, mix('crystal_lt', 'white', 0.3))
    info = palace(o, B_HX, B_Y0, B_S, lit, wins, detail=2)
    # pedestal under the heart: plinth, slim column, flared cup
    rp = hue_ramp('palace', lit)
    rpk = hue_ramp('pink', lit)
    py = B_Y0
    o.rect(B_HX - 12, py - 4, 25, 5, rp['ol'])
    o.rect(B_HX - 11, py - 3, 23, 3, rp['md'])
    o.rect(B_HX + 1, py - 3, 11, 3, rp['lt'])
    o.hline(B_HX - 11, B_HX + 11, py - 3, rp['hi'])
    o.rect(B_HX - 4, py - 9, 9, 6, rp['ol'])
    o.rect(B_HX - 3, py - 9, 3, 6, rp['dk'])
    o.rect(B_HX, py - 9, 2, 6, rp['md'])
    o.rect(B_HX + 2, py - 9, 2, 6, rp['lt'])
    o.poly([(B_HX - 8, py - 13), (B_HX + 8, py - 13), (B_HX + 4, py - 9), (B_HX - 4, py - 9)], rp['ol'])
    o.poly([(B_HX - 7, py - 12), (B_HX + 7, py - 12), (B_HX + 4, py - 10), (B_HX - 4, py - 10)], rpk['lt'])
    o.hline(B_HX - 7, B_HX + 7, py - 13, rpk['hi'])
    return to_rgba(cv), tuple(wins), info


@lru_cache(maxsize=None)
def b_near_layer(lit=False):
    cv = blayer()
    o = OC(cv, BOX, BOY)
    wins = []
    big_house(o, 6, 176, 58, 122, 'pink', lit, 51, wins, side=1)
    big_house(o, 316, 178, 62, 132, 'teal', lit, 52, wins, side=-1)
    for lx, yb in ((70, 172), (250, 172)):
        o.rect(lx - 2, yb - 2, 5, 3, mix('crystal_dk', 'ink', 0.3))
        o.vline(lx, yb - 44, yb, mix('crystal_dk', 'ink', 0.3))
        o.vline(lx + 1, yb - 44, yb - 2, mix('crystal', 'navy', 0.3))
        gem_bulb(o, lx, yb - 44, yb - 50, yb - 57, 4, 8, hue_ramp('pink', True))
        o.rect(lx - 1, yb - 52, 3, 3, 'white')
    # snowy ground strip at the very bottom
    o.rect(-BOX, 176, BW_, 40, mix('periwinkle', 'navy', 0.3))
    o.hline(-BOX, BW_ - BOX, 176, 'cloud')
    for x in range(-BOX, BW_ - BOX, 2):
        o.px(x, 177 + int(hash01(x, 9) * 2), mix('cloud', 'periwinkle', 0.4))
    return to_rgba(cv), tuple(wins)


def cam_b(f):
    return int(round(keyframes(f, [(T_B, 66), (210, B_TILT), (T_C, B_TILT + 6)], ease=cruise)))


def heart_scale(f):
    if f < T_HEART:
        return 1.0
    t = (f - T_HEART) / 18.0
    return 1.0 + 0.42 * math.exp(-2.6 * t) * math.cos(t * 7.5)


def render_b(f):
    camy = cam_b(f)
    d = B_TILT - camy                      # 0 at the end of the tilt
    c = Canvas()
    sky_off = int(round(-30 + d * 0.5))
    draw_sky(c, f, 0, sky_off)
    flare = ease_in_out(seg(f, T_HEART, 270)) if f >= T_HEART else 0.0
    draw_aurora(c, f, 0, int(round(-d * 0.45 + 110)), flare, 1.15)
    ring_r = (f - T_HEART) * 9 if f >= T_HEART else -1

    def lay(fn, par, lit_cols=True):
        r0 = fn(False)
        oy = int(round(d * par)) - BOY
        ox = -BOX
        if ring_r > 0 and lit_cols:
            r1 = fn(True)
            xs = np.arange(W)
            m = np.abs(xs - B_HX) < ring_r * (0.75 + 0.25 * par)
            y0 = -oy
            a0 = r0[0][max(0, y0):y0 + H, BOX:BOX + W]
            a1 = r1[0][max(0, y0):y0 + H, BOX:BOX + W]
            comp = np.where(m[None, :, None], a1, a0)
            c.blit(comp, 0, max(0, oy))
        else:
            c.blit(r0[0], ox, oy)
        win_flicker(c, r0[1], 0, oy + BOY, f, int(par * 100))
        return oy + BOY
    lay(lambda lit: b_back_layer(2, lit), PB_BACK2)
    lay(lambda lit: b_back_layer(1, lit), PB_BACK1)
    poy = lay(b_palace_layer, PB_PAL, lit_cols=False)
    info = b_palace_layer(False)[2]
    # palace aura + pennants on the side spires
    fx.glow(c, B_HX, int(info['top']) + poy - 60, 70, 'crystal', strength=0.12, rings=4)
    for sgn in (-1, 1):
        pennant(c, B_HX + sgn * int(round(12.5 * B_S)), int(info['side_tip']) + poy + 4, f + sgn * 7, 2,
                'crystal_pk' if sgn < 0 else 'ice')
    # the Heart
    hy = B_HY + poy + int(round(math.sin(f * 2 * math.pi / 72) * 1.5))
    k = heart_scale(f)
    pulse = clamp(1 - (f - T_HEART) / 24.0) if f >= T_HEART else 0.0
    fx.glow(c, B_HX, hy, int(30 + 70 * (1 - pulse) * (pulse > 0)) if pulse > 0 else 30, 'magic',
            strength=0.3 + 0.05 * math.sin(f * 0.35) + 0.5 * pulse, rings=5)
    fx.glow(c, B_HX, B_Y0 + poy + 4, 40, 'crystal_pk', strength=0.12, rings=3)
    draw_heart(c, B_HX, hy, max(4, int(round(B_HSIZE * k))), f, lit=0.3 * pulse)
    # refracted rainbow glints on the plaza (rotate with the heart's facets) + a little spectrum fan
    step = f // 3
    fy = B_Y0 + poy
    for i in range(14):
        a_ = (i / 14.0) * 2 * math.pi + step * 0.07
        rx_ = 30 + 26 * hash01(i, 2)
        gx = B_HX + int(round(math.cos(a_) * rx_ * 1.6))
        gy = fy + 12 + int(round(math.sin(a_) * rx_ * 0.32))
        if gy <= fy + 1:
            continue
        hue_ = (i / 14.0 + step * 0.01) % 1.0
        rc = tuple(int(v) for v in _rainbow(np.array([hue_]))[0])
        tw = (step + i * 5) % 9
        if tw < 6:
            c.rect(gx - 1, gy, 3, 1, mix(rc, 'white', 0.25))
            if tw in (2, 3):
                c.px(gx, gy - 1, 'white')
                c.px(gx, gy + 1, mix(rc, 'white', 0.5))
                c.rect(gx - 2, gy, 5, 1, mix(rc, 'white', 0.4))
    for j in range(6):                      # spectrum under the heart's point onto the plaza
        hue_ = j / 6.0
        rc = tuple(int(v) for v in _rainbow(np.array([hue_]))[0])
        x1 = B_HX - 14 + j * 6 + int(round(2 * math.sin(step * 0.07)))
        pts = []
        for yy in range(fy + 2, fy + 9):
            tt = (yy - fy - 2) / 7.0
            xx = int(round(lerp(B_HX - 2 + j * 0.8, x1, tt)))
            pts.append((yy, xx))
        ys_ = np.array([p_[0] for p_ in pts]); xs_ = np.array([p_[1] for p_ in pts])
        ok = (ys_ >= 0) & (ys_ < H)
        add_light(c, (ys_[ok], xs_[ok]), rc, 0.35)
    lay(b_near_layer, PB_NEAR)
    # lamp glows
    noy = int(round(d * PB_NEAR))
    for lx in (70, 250):
        fx.glow(c, lx, 172 - 52 + noy, 12, 'magic', strength=0.3 + 0.06 * math.sin(f * 0.27 + lx), rings=3)
    # banners on the lamp posts
    for lx, cc_ in ((70, 'ice'), (250, 'crystal_pk')):
        pennant(c, lx, 172 - 40 + noy, f + lx, 3, cc_)
    # big soft snowflakes near the lens
    snowfall(c, f, n=10, seed=9, speed=0.9, camy=-d * 1.4)
    # the heartbeat: flash + ring of light rolling out over the plaza
    if f >= T_HEART:
        t = f - T_HEART
        if t < 4:
            fx.grade(c, add=(60 - 15 * t, 30 - 8 * t, 50 - 12 * t))
        for rr, kk in ((ring_r, 0.6), (ring_r - 4, 0.3)):
            if 0 < rr < 400:
                th = np.linspace(0, 2 * math.pi, 1400)
                xs = np.round(B_HX + rr * np.cos(th)).astype(int)
                ys = np.round(B_Y0 + poy + 6 + rr * 0.22 * np.sin(th)).astype(int)
                ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                pts = np.unique(np.stack([ys[ok], xs[ok]]), axis=1)
                add_light(c, (pts[0], pts[1]), 'cotton', kk * (1 - t / 30.0))
        if t < 20:
            c2 = int(t * 9)
            fx.glow(c, B_HX, hy, 24 + c2, 'cotton', strength=0.3 * (1 - t / 20.0), rings=4)
    # callouts
    callout(c, B_HX + 1, 70, 'CRYSTAL PALACE', f, 162, 234, side='right', length=26)
    callout(c, B_HX + 11, hy - 4, 'CRYSTAL HEART', f, 180, 252, side='right', length=40)
    atlas_card(c, f, 100, 206, 'CRYSTAL EMPIRE', ['CITY OF CRYSTAL', 'HOME OF THE HEART'])
    return c


# ================================================================== shot C: the wave of light over the city
C_SPEED = 2.6


def ring_r(f, t0):
    return (f - t0) * C_SPEED if f >= t0 else -1.0


def render_c(f):
    camx = int(round(lerp(58, 66, seg(f, T_C, T_D))))
    camy = int(round(keyframes(f, [(T_C, 0), (270, 0), (T_D - 6, -34)], ease=cruise)))
    r1 = ring_r(f, T_HEART)
    r2 = ring_r(f, T_HEART2)
    offs = {P_IN: 0.0, P_MID: 14.0, P_OUT: 30.0}
    xs = np.arange(W)

    def lit_fn(par):
        hxl = ring_centre(par) - camx * par
        return np.abs(xs - hxl) < r1 - offs[par]
    t2 = (f - T_HEART2) / 18.0
    hk = 1.0 + (0.6 * math.exp(-2.6 * t2) * math.cos(t2 * 7.5) if f >= T_HEART2 else 0.0)
    hg = clamp(1 - (f - T_HEART2) / 24.0) if f >= T_HEART2 else 0.25
    flare = ease_in_out(seg(f, T_HEART, 270))
    aur_k = 1.25 + (0.45 * math.exp(-(f - T_HEART2) / 22.0) if f >= T_HEART2 else 0.0)
    bx = int(round(lerp(250, 226, seg(f, T_C, T_D))))
    c, hx, hy0 = city_frame(f, camx, camy, lit_fn=lit_fn, flare=flare, aur_k=aur_k,
                            balloon=(bx, 66 - camy // 3, 's'), heart_k=hk, heart_glow=hg)
    gy = PAL_BASE + hy0 + 24                       # ground level of the rings (screen)
    # the dome flashes when the wave reaches its rim
    dk = 1.0 + 2.2 * math.exp(-abs(f - (T_HEART + 150 / C_SPEED)) / 8.0)
    dome(c, f, hx, PAL_BASE + 36 + hy0, 150, 108, dk)
    # rings of light rolling outward over the city (ellipses on the ground plane) + light wall
    for rr, kk in ((r1, 0.55), (r1 - 5, 0.25), (r2, 0.5), (r2 - 5, 0.22)):
        if rr <= 2:
            continue
        th = np.linspace(0, 2 * math.pi, 1600)
        for dy in (0, -1):
            xs_ = np.round(hx + rr * np.cos(th)).astype(int)
            ys_ = np.round(gy + dy + rr * 0.24 * np.sin(th)).astype(int)
            ok = (xs_ >= 0) & (xs_ < W) & (ys_ >= 0) & (ys_ < H)
            pts = np.unique(np.stack([ys_[ok], xs_[ok]]), axis=1)
            add_light(c, (pts[0], pts[1]), 'cotton', kk * (0.8 if dy else 1.0))
        # vertical shimmer of light where the wave front meets the houses
        for sgn in (-1, 1):
            x = int(hx + sgn * rr)
            if 0 <= x < W and kk > 0.4:
                fx.light_rect(c, x - 2, gy - 40, 5, 40, 'cotton', strength=0.25, fade='up')
    if T_HEART <= f < T_HEART + 6:
        fx.grade(c, add=(30, 15, 25))
    if T_HEART2 <= f < T_HEART2 + 4:
        k_ = f - T_HEART2
        fx.grade(c, add=(40 - 10 * k_, 20 - 5 * k_, 32 - 8 * k_))
    cm = int(round(ring_centre(P_MID) - camx * P_MID))
    for lx in (-15, 15, -60, 60, -100, 100):
        fx.glow(c, cm + lx, 134 - 8 - int(round(camy * P_MID)), 5, 'magic', strength=0.3, rings=2)
    snowfall(c, f, n=14, seed=2, speed=0.35, camy=camy)
    apx = 156 - int(round(camx * P_EVER))
    apy = 44 - int(round(camy * P_EVER))
    callout(c, apx, apy, 'MT. EVERHOOF', f, 270, 342, side='left', length=14)
    return c


# ================================================================== shot D: the Crystal Station, crane up
D_CY0 = 50                 # camera starts this much lower (looking down at the station)
DOX, DOY = 20, 10
DW_, DH_ = W + 40, 280
D_RAIL = 206               # rail top (E-coords, i.e. screen y at the end of the crane)
D_PLAT = 198
TRAIN_STOP = 116           # x of the locomotive's nose when stopped


def dlayer():
    return Canvas(DW_, DH_, KEY)


def snow_globe(c, cx, cy, r, f):
    """The gift shop's glowing snow-globe window (drawn per frame: the flurry swirls)."""
    c.circle(cx, cy, r + 2, 'brass')
    c.circle(cx, cy, r + 1, mix('brass', 'ink', 0.4))
    c.circle(cx, cy, r, mix('crystal', 'navy', 0.25))
    c.circle(cx, cy - 2, r - 2, mix('crystal', 'baby_blue', 0.3))
    c.circle(cx - 1, cy - 4, r - 5, mix('baby_blue', 'cotton', 0.35))
    # tiny palace inside
    c.rect(cx - 1, cy - 6, 3, 11, 'crystal_lt')
    c.px(cx, cy - 8, 'crystal_lt')
    c.px(cx, cy - 7, 'white')
    c.rect(cx - 4, cy - 1, 2, 6, 'crystal')
    c.rect(cx + 3, cy - 1, 2, 6, 'crystal')
    c.px(cx - 3, cy - 2, 'crystal')
    c.px(cx + 4, cy - 2, 'crystal')
    c.rect(cx - 6, cy + 4, 13, 2, 'crystal_lt')
    c.px(cx, cy + 2, 'crystal_pk')
    c.rect(cx - r + 2, cy + 6, 2 * r - 3, r - 7, 'cloud')
    # glass highlight
    c.line(cx - r + 3, cy - 4, cx - 5, cy - r + 3, 'white')
    # swirling flurry
    for i in range(16):
        a = i * 2.4 + f * (0.05 + 0.02 * (i % 3))
        rr = (r - 3) * (0.3 + 0.7 * hash01(i, 5))
        x = int(round(cx + math.cos(a) * rr))
        y = int(round(cy + math.sin(a * 0.7 + i) * rr * 0.8 + ((f * 0.4 + i * 7) % 8) - 4))
        if (x - cx) ** 2 + (y - cy) ** 2 < (r - 1) ** 2:
            c.px(x, y, 'white')
    # base
    c.rect(cx - r + 2, cy + r, 2 * r - 3, 4, 'copper')
    c.hline(cx - r + 2, cx + r - 2, cy + r, 'clay')
    c.rect(cx - r + 3, cy + r + 4, 2 * r - 5, 1, 'rust')


@lru_cache(maxsize=None)
def d_station_layer():
    cv = dlayer()
    o = OC(cv, DOX, DOY)
    rpl = hue_ramp('lilac', True)
    rpb = hue_ramp('blue', True)
    rpt = hue_ramp('teal', True)
    # ---- train shed (behind the track): back wall, platform, pillars, faceted crystal canopy
    o.rect(96, 150, 250, 48, mix('indigo', 'navy', 0.35))
    for k in range(7):                                  # back-wall arches with lit panes
        x = 104 + k * 34
        o.rect(x, 160, 22, 30, mix('indigo', 'navy', 0.1))
        o.rect(x + 1, 163, 20, 26, mix('crystal', 'navy', 0.25))
        o.rect(x + 2, 166, 18, 10, mix('crystal_lt', 'crystal', 0.5))
        o.vline(x + 11, 163, 188, mix('indigo', 'navy', 0.1))
        o.hline(x + 1, x + 20, 176, mix('indigo', 'navy', 0.1))
        o.ellipse(x + 11, 163, 10, 3, mix('crystal', 'navy', 0.25))
    o.rect(-DOX, D_PLAT, DW_, 6, mix('crystal', 'navy', 0.2))
    o.hline(-DOX, DW_, D_PLAT, mix('crystal_lt', 'white', 0.3))
    o.hline(-DOX, DW_, D_PLAT + 1, 'cream')
    for x in range(-DOX, DW_, 8):
        o.vline(x, D_PLAT + 2, D_PLAT + 5, mix('crystal_dk', 'navy', 0.3))
    for k in range(8):                                  # slim crystal pillars
        x = 100 + k * 34
        o.rect(x - 1, 146, 4, 52, mix('crystal_dk', 'ink', 0.2))
        o.rect(x, 146, 1, 52, rpb['md'])
        o.rect(x + 1, 146, 1, 52, rpb['hi'])
        o.rect(x - 2, 194, 6, 4, rpb['lt'])
    # canopy: a row of faceted gables (crystal panes with ribs)
    for k in range(8):
        x0 = 96 + k * 32
        pts = [(x0, 150), (x0 + 16, 128), (x0 + 32, 150)]
        o.poly([(x0 - 1, 151), (x0 + 16, 126), (x0 + 33, 151)], mix('crystal_dk', 'ink', 0.25))
        o.poly([(x0 + 1, 150), (x0 + 16, 129), (x0 + 16, 150)], rpb['md'])
        o.poly([(x0 + 16, 129), (x0 + 31, 150), (x0 + 16, 150)], rpb['lt'])
        o.line(x0 + 16, 129, x0 + 16, 150, rpb['hi'])
        o.line(x0 + 8, 140, x0 + 24, 140, mix('crystal_dk', 'ink', 0.1))
        o.dither(x0 + 18, 134, 10, 14, 'white', 0.25)
        o.px(x0 + 16, 127, 'white')
    o.rect(94, 150, 260, 4, mix('crystal_dk', 'ink', 0.25))
    o.hline(94, 353, 151, rpb['hi'])
    for k in range(8):                                  # hanging lanterns under the canopy
        x = 116 + k * 34
        o.vline(x, 154, 157, mix('crystal_dk', 'ink', 0.2))
        o.rect(x - 1, 158, 3, 3, 'butter')
    # ---- station house with the gift shop (left)
    o.rect(-DOX, 150, 112, 48, rpl['ol'])
    o.rect(-DOX, 151, 110, 47, rpl['md'])
    o.rect(-DOX, 151, 14, 47, rpl['dk'])
    o.rect(70, 151, 34, 47, rpl['lt'])
    for k in range(10):                                 # facet striations
        yy = 154 + int(hash01(k, 61) * 40)
        xx = int(hash01(k, 62) * 96)
        o.hline(xx, xx + 3, yy, rpl['hi'] if xx > 70 else rpl['lt'])
    o.rect(-DOX, 196, 112, 2, rpl['ol'])
    for px_ in (-2, 52, 104):                            # crystal pilasters
        o.rect(px_, 151, 4, 45, rpl['ol'])
        o.rect(px_ + 1, 151, 1, 45, rpl['dk'])
        o.rect(px_ + 2, 151, 1, 45, rpl['hi'])
        gem_shaft(o, px_ + 2, 196, 192, 6, 5, 2, rpl, glow=False)
    o.rect(-DOX, 151, 112, 2, rpl['hi'])
    o.rect(-DOX, 153, 112, 1, rpl['ol'])
    for k in range(8):                                   # scalloped awning over the globe window
        ax_ = 18 + k * 4
        o.rect(ax_, 158, 4, 3, 'crystal_pk' if k % 2 else 'cotton')
        o.px(ax_ + 1, 161, 'crystal_pk' if k % 2 else 'cotton')
        o.px(ax_ + 2, 161, 'crystal_pk' if k % 2 else 'cotton')
    o.hline(17, 50, 157, rpl['ol'])
    # shop door + icon sign (a little globe)
    o.rect(60, 176, 12, 22, rpl['ol'])
    o.rect(61, 177, 10, 21, 'gold')
    o.rect(61, 177, 10, 9, 'butter')
    o.vline(66, 177, 197, rpl['ol'])
    o.px(61, 177, rpl['ol'])
    o.px(70, 177, rpl['ol'])
    o.rect(78, 166, 2, 2, 'brass')
    o.rect(76, 168, 12, 10, 'brass')
    o.rect(77, 169, 10, 8, mix('indigo', 'navy', 0.2))
    o.circle(82, 172, 3, 'crystal_lt')
    o.rect(80, 175, 5, 1, 'copper')
    o.px(81, 171, 'white')
    # upstairs windows
    for x in (60, 84):
        window(o, x, 157, 4, 6, rpl)
        window(o, x + 7, 157, 4, 6, rpl)
    window(o, 4, 157, 4, 6, rpl)
    # roof: faceted crystal gable + clock turret
    o.poly([(-DOX - 2, 151), (48, 118), (110, 151)], mix('crystal_dk', 'ink', 0.3))
    o.poly([(-DOX, 150), (48, 120), (48, 150)], rpt['md'])
    o.poly([(48, 120), (107, 150), (48, 150)], rpt['lt'])
    o.poly([(20, 150), (48, 120), (34, 150)], rpt['dk'])
    o.poly([(48, 120), (70, 150), (60, 150)], rpt['hi'])
    o.line(48, 120, 107, 150, mix('ice', 'white', 0.5))
    o.hline(-DOX, 108, 150, mix('crystal_dk', 'ink', 0.3))
    gem_shaft(o, 48, 124, 104, 14, 12, 12, rpb)
    o.circle(48, 112, 4, mix('brass', 'ink', 0.3))
    o.circle(48, 112, 3, 'cream')
    o.vline(48, 110, 112, 'ink')
    o.hline(48, 50, 112, 'ink')
    # ---- track (in front of the platform) + snowy bank
    o.rect(-DOX, D_PLAT + 6, DW_, 2, mix('indigo', 'navy', 0.4))
    for x in range(-DOX, DW_, 5):
        o.rect(x, D_RAIL + 1, 3, 2, mix('rust', 'ink', 0.45))
    o.rect(-DOX, D_RAIL, DW_, 1, 'steel_hi')
    o.rect(-DOX, D_RAIL + 1, DW_, 1, 'steel_dk')
    o.rect(-DOX, D_RAIL + 3, DW_, DH_, mix('periwinkle', 'navy', 0.35))
    for x in range(-DOX, DW_):
        yy = D_RAIL + 8 + 2 * math.sin(x * 0.07) + math.sin(x * 0.19)
        o.vline(x, yy, DH_, mix('periwinkle', 'baby_blue', 0.35))
        o.px(x, yy, 'cloud')
        o.vline(x, yy + 6, DH_, mix('crystal', 'navy', 0.35))
    # foreground: crystal fence along the snow bank, snowy bollards, crystal shrubs
    fy = D_RAIL + 18
    for x in range(-DOX, DW_, 14):
        o.rect(x, fy - 9, 2, 10, mix('crystal_dk', 'ink', 0.2))
        o.px(x + 1, fy - 9, rpb['hi'])
        o.rect(x - 1, fy - 11, 4, 2, 'cloud')
    o.hline(-DOX, DW_, fy - 7, rpb['md'])
    o.hline(-DOX, DW_, fy - 4, rpb['md'])
    o.hline(-DOX, DW_, fy - 8, 'cloud')
    for x, hue in ((24, 'pink'), (150, 'teal'), (232, 'lilac'), (300, 'pink')):
        for dx, hh in ((-3, 6), (0, 10), (3, 7)):
            gem_shaft(o, x + dx, fy + 6, fy + 6 - hh + 2, 3, 2, 3, hue_ramp(hue, True), glow=False)
    o.rect(-DOX, fy + 6, DW_, DH_, mix('periwinkle', 'baby_blue', 0.3))
    for x in range(-DOX, DW_):
        yy = fy + 5 + 2 * math.sin(x * 0.05 + 1) + (1 if (x // 7) % 3 == 0 else 0)
        o.vline(x, yy, DH_, mix('periwinkle', 'baby_blue', 0.45))
        o.px(x, yy, 'white')
    return to_rgba(cv)


def draw_train(c, x, y, f, moving):
    """The Friendship Express facing left; x = nose, y = rail top. Wheels turn with x."""
    ph = (-x * 0.25) % (2 * math.pi)
    ink = mix('plum', 'ink', 0.4)

    def wheel(cx, r, col_):
        c.circle(cx, y - r, r, ink)
        c.circle(cx, y - r, r - 1, col_)
        c.px(cx, y - r, 'gold')
        a = ph
        c.px(int(round(cx + math.cos(a) * (r - 1))), int(round(y - r + math.sin(a) * (r - 1))), 'gold')
    # ---- locomotive
    lx = x
    c.poly([(lx, y - 2), (lx + 6, y - 9), (lx + 6, y - 2)], 'gold')          # cowcatcher
    c.line(lx + 1, y - 3, lx + 5, y - 7, 'amber')
    c.rect(lx + 5, y - 20, 28, 12, ink)                                       # boiler
    c.rect(lx + 6, y - 19, 26, 10, 'bubblegum')
    c.rect(lx + 6, y - 19, 26, 3, mix('bubblegum', 'white', 0.45))
    c.rect(lx + 6, y - 12, 26, 3, 'rose')
    for bx in (lx + 12, lx + 22):
        c.vline(bx, y - 19, y - 10, 'gold')
    c.rect(lx + 4, y - 19, 3, 10, mix('royal', 'ink', 0.2))                  # smokebox front
    c.circle(lx + 4, y - 15, 2, 'butter')                                     # headlamp
    c.px(lx + 3, y - 16, 'white')
    c.rect(lx + 9, y - 28, 5, 8, ink)                                         # stack
    c.rect(lx + 10, y - 27, 3, 7, 'royal_lt')
    c.rect(lx + 8, y - 30, 7, 3, ink)
    c.rect(lx + 9, y - 29, 5, 1, 'gold')
    c.rect(lx + 18, y - 23, 5, 3, 'gold')                                     # dome
    c.px(lx + 19, y - 24, 'butter')
    c.rect(lx + 32, y - 30, 14, 22, ink)                                      # cab
    c.rect(lx + 33, y - 29, 12, 20, 'lavender')
    c.rect(lx + 33, y - 29, 12, 2, 'lilac')
    c.rect(lx + 35, y - 26, 7, 6, 'cream')
    c.rect(lx + 35, y - 21, 7, 1, 'gold')
    c.rect(lx + 31, y - 32, 16, 3, 'royal')
    c.hline(lx + 31, lx + 46, y - 32, 'royal_lt')
    c.rect(lx + 5, y - 8, 42, 2, ink)                                         # frame
    wheel(lx + 9, 3, 'crimson')
    for k in range(3):
        wheel(lx + 19 + k * 9, 4, 'crimson')
    rx = int(round(math.cos(ph) * 2))
    c.line(lx + 19 + rx, y - 4, lx + 37 + rx, y - 4, 'steel_hi')              # coupling rod
    # ---- tender
    tx = lx + 49
    c.rect(tx, y - 22, 18, 14, ink)
    c.rect(tx + 1, y - 21, 16, 12, 'rose')
    c.rect(tx + 1, y - 21, 16, 2, 'pink')
    c.rect(tx + 2, y - 24, 14, 3, mix('ink', 'plum', 0.3))
    wheel(tx + 5, 3, 'wine')
    wheel(tx + 13, 3, 'wine')
    # ---- two coaches + caboose
    cx_ = tx + 21
    for k in range(2):
        c.rect(cx_, y - 28, 38, 21, ink)
        c.rect(cx_ + 1, y - 27, 36, 19, 'cotton')
        c.rect(cx_ + 1, y - 13, 36, 3, 'bubblegum')
        c.rect(cx_ - 1, y - 31, 40, 4, 'royal')
        c.hline(cx_ - 1, cx_ + 38, y - 31, 'royal_lt')
        for wx in range(cx_ + 4, cx_ + 34, 7):
            c.rect(wx, y - 24, 5, 7, mix('plum', 'ink', 0.3))
            c.rect(wx, y - 24, 5, 6, 'cream')
            c.rect(wx, y - 20, 5, 2, 'gold')
        c.rect(cx_ + 1, y - 8, 36, 1, ink)
        wheel(cx_ + 7, 3, 'wine')
        wheel(cx_ + 30, 3, 'wine')
        c.rect(cx_ - 3, y - 12, 3, 2, ink)
        cx_ += 41
    c.rect(cx_, y - 24, 28, 17, ink)
    c.rect(cx_ + 1, y - 23, 26, 15, 'baby_blue')
    c.rect(cx_ + 1, y - 12, 26, 3, 'crystal')
    c.rect(cx_ + 8, y - 31, 12, 8, ink)
    c.rect(cx_ + 9, y - 30, 10, 7, 'baby_blue')
    c.rect(cx_ + 11, y - 29, 6, 3, 'cream')
    c.rect(cx_ - 1, y - 26, 30, 3, 'royal')
    c.rect(cx_ + 7, y - 33, 14, 2, 'royal')
    c.rect(cx_ + 4, y - 20, 5, 5, 'cream')
    c.rect(cx_ + 19, y - 20, 5, 5, 'cream')
    c.px(cx_ + 27, y - 15, 'red')
    wheel(cx_ + 6, 3, 'wine')
    wheel(cx_ + 22, 3, 'wine')
    return lx + 11, y - 30         # stack top


def train_x(f):
    t = seg(f, T_D, T_D + 54)
    return int(round(lerp(340, TRAIN_STOP, 1 - (1 - t) ** 2.2)))


def render_d(f):
    cy = int(round(keyframes(f, [(T_D, D_CY0), (T_D + 24, D_CY0), (T_D + 84, -16), (N, -19)], ease=cruise)))
    camx = int(round(lerp(0, 16, seg(f, T_D, N))))
    c, hx, hy0 = city_frame(f, camx, cy * 0.62, lit_fn=lambda p: True, flare=0.75, aur_k=1.2, fg=False,
                            heart_glow=0.25)
    dome(c, f, hx, PAL_BASE + 36 + hy0, 150, 108, 1.2)
    cm = int(round(ring_centre(P_MID) - camx * P_MID))
    for lx in (-15, 15, -60, 60, -100, 100):
        fx.glow(c, cm + lx, 134 - 8 + int(round(-cy * 0.62 * P_MID)), 5, 'magic', strength=0.3, rings=2)
    # balloon glides past in front of the city
    bt = seg(f, T_D + 8, N)
    draw_balloon(c, int(round(lerp(352, 238, bt))), int(round(72 - cy * 0.75)), f, 'm', flame=(f // 18) % 3 == 0)
    # station + train
    oy = -cy - DOY
    c.blit(d_station_layer(), -DOX - int(round(camx * 1.1)), oy)
    ox = -int(round(camx * 1.1))
    sy = -cy                                           # E -> screen offset for the station
    fx.glow(c, 34 + ox, 176 + sy, 24, 'crystal_lt', strength=0.25 + 0.05 * math.sin(f * 0.2), rings=4)
    snow_globe(c, 34 + ox, 176 + sy, 13, f)
    for k in range(8):
        fx.glow(c, 117 + k * 34 + ox, 160 + sy, 7, 'gold', strength=0.25, rings=2)
    tx = train_x(f) + ox
    moving = f < T_D + 54
    stx, sty = draw_train(c, tx, D_RAIL + sy, f, moving)
    # headlamp beam (forward = left)
    fx.glow(c, tx + 4, D_RAIL + sy - 15, 6, 'butter', strength=0.35, rings=2)
    # steam puffs from the stack (faster while moving), cylinder hiss on stopping
    spd = 1.0 if moving else 0.4
    for i in range(7):
        per = 26
        t = ((f * spd + i * per / 7.0) % per) / per
        px_ = stx + int(t * (14 if moving else 4)) + int(math.sin(t * 6 + i) * 2)
        py_ = sty - int(t * 22)
        r_ = 1 + int(t * 4)
        if hash01(i, int(f * spd) // 9) < 1 - t * 0.6:
            c.circle(px_, py_, r_, mix('white', 'cloud_dk', t))
            c.circle(px_ - 1, py_ - 1, max(1, r_ - 1), 'white')
    if T_D + 52 <= f < T_D + 76:
        t = (f - T_D - 52) / 24.0
        for k in range(6):
            r_ = int(2 + t * 7 + k % 3)
            c.circle(tx + 14 - int(t * 14) - k * 3, D_RAIL + sy - 3 - int(t * 5), r_, mix('white', 'cloud', t))
    # falling snow (two depths)
    snowfall(c, f, n=46, seed=31, speed=0.5, camy=cy)
    snowfall(c, f, n=10, seed=32, speed=1.0, camy=cy * 1.3)
    callout(c, 50 + ox, 128 + sy, 'CRYSTAL STATION', f, T_D + 6, T_D + 78, side='right', length=30)
    return c


# ================================================================== render
def render(f):
    if f < T_A:
        return travel_frame(f, ID)
    if f < T_B:
        c = render_a(f)
    elif f < T_C:
        c = render_b(f)
    elif f < T_D:
        c = render_c(f)
    else:
        c = render_d(f)
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), PX, 89)
    return c


SFX = [
    (80, 'wind', 0.35, {'variant': 0}),
    (96, 'shimmer', 0.35),
    (162, 'crystal', 0.5),
    (198, 'sparkle', 0.35),
    (222, 'crystal', 0.75),
    (252, 'shimmer', 0.55),
    (272, 'sparkle', 0.45),
    (294, 'crystal', 0.5),
    (342, 'chug', 0.55),
    (360, 'whistle', 0.55),
    (378, 'chug', 0.3),
    (394, 'steam', 0.5),
    (408, 'wind', 0.3, {'variant': 0}),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(T_B, 'dissolve', {}), (T_C, 'dissolve', {}), (T_D, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
