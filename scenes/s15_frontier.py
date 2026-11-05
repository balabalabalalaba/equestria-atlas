"""FRONTIER TOWNS: Appleloosa + Dodge Junction
(4 bars, 288 frames, HUD clock 18:30 -> 19:15, sunset over the ochre plains, big low sun in the west = right).

Beat sheet (local frames; bar = 72, beat = 18). Chords: E@72  A@144  B@180  C#m@216  G#7@252.
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71 on dest_screen (204, 107).
  Bars 2-3 72-215 SHOT A: APPLELOOSA MAIN STREET. REVEAL 72: iris opens 72-79 on the Salt Block saloon
                  sign next to the clock tower (204, 107). One slow eased pan right along Main Street
                  (cam 0 -> 240): pastel false-front buildings (pink store, mint hotel, sky bakery, peach
                  SALT BLOCK saloon with swinging batwing doors, cream clock tower whose hands run with
                  the HUD clock, the little stage with swaying bunting, the brown water tower, the mint
                  rail depot) then the young APPLE ORCHARDS in neat rows with a spinning windpump and a
                  glinting irrigation ditch, the big low sun setting over them.
                  Sky graded violet -> coral -> gold with streaky clouds lit pink from below; layered
                  silhouettes: violet far mesas (0.12), pink Macintosh Hills (0.22), rust buttes (0.35),
                  ochre plain with the rail line and far orchards (0.55), town (1.0), fg (1.35).
                  Long shadows slant down-left across the street, light shafts through the alleys,
                  warm rim light on the right edges. Tumbleweeds roll right, ONE HOP PER BEAT (land on
                  f % 18 == 0) with a dust puff; dust swirls; chimney smoke; flags; lanterns/windows light
                  up on the beats from 144 (A chord), the rest by 200.
                  Balloon 'm' drifts high on the right. Callouts SALT BLOCK 90-162, APPLELOOSA 126-198,
                  APPLE ORCHARDS 162-212. Card 'FRONTIER TOWNS' 100-196.
  Bar 4  216-287  CUT (216, WHISTLE): DODGE JUNCTION at sundown. The Friendship Express (pink loco,
                  lavender/blue cars, no driver) at the little platform blows a tall whistle-steam jet
                  on 216 and departs right into the sunset (chug puffs on the beats), accelerating; the
                  semaphore arm is up (green lamp). Station house, water tower with spout, parked
                  stagecoach, crates of cherries. CHERRY HILL RANCH on the right: the hill with rows of
                  red- and yellow-cherry trees, the sorting shed with its conveyor belt carrying cherries
                  into the loft. The train turns to a silhouette with a gold rim as it crosses the
                  sinking sun (~250 on). Balloon 'm' with burner glow against the sunset.
                  Camera: eased pan right (cam 0 -> 64). Callouts DODGE JUNCTION 222-276,
                  CHERRY HILL RANCH 236-280. Toast from 230 (top centre kept clear).
Ambient: clouds, sun shimmer, tumbleweeds, dust, saloon doors, clock hands, bunting, flags, smoke,
windpump, irrigation glints, windows/lanterns, balloon | steam jet, chug puffs, train, semaphore lamp,
conveyor cherries, cherry trees sway, dust, clouds.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'frontier'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
B = FRAMES_PER_BEAT
KEY = (1, 2, 3)


# ================================================================== generic helpers
def C(c):
    return np.array(col(c), np.uint8)


def new(w, h=H):
    return Canvas(w, h, KEY)


def to_layer(cv):
    a = cv.a
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    return a, m


def comp(c, layer, x, y=0):
    """Composite a cached (rgb, mask) layer with its top-left at screen (x, y)."""
    a, m = layer
    x, y = int(x), int(y)
    lh, lw = a.shape[:2]
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(W, x + lw), min(H, y + lh)
    if X1 <= X0 or Y1 <= Y0:
        return
    np.copyto(c.a[Y0:Y1, X0:X1], a[Y0 - y:Y1 - y, X0 - x:X1 - x],
              where=m[Y0 - y:Y1 - y, X0 - x:X1 - x][:, :, None])


def vnoise(h, w, cy, cx, seed):
    """Smooth value noise (h, w) in 0..1 with cell sizes (cy, cx)."""
    R = rng(seed)
    gh, gw = int(h / cy) + 3, int(w / cx) + 3
    g = R.random((gh, gw))
    yy = np.arange(h) / cy
    xx = np.arange(w) / cx
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    fy = yy - y0
    fx_ = xx - x0
    fy = fy * fy * (3 - 2 * fy)
    fx_ = fx_ * fx_ * (3 - 2 * fx_)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c_ = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx_[None]
    bot = c_ + (d - c_) * fx_[None]
    return top + (bot - top) * fy[:, None]


def noise1(w, cell, seed):
    return vnoise(1, w, 1, cell, seed)[0]


def outline_of(m):
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~inner


def stops_cols(stops, h, n):
    """Sample n colours from (y, colour) stops over 0..h (for Canvas.gradient)."""
    out = []
    for i in range(n):
        y = i * (h - 1) / (n - 1)
        for j in range(len(stops) - 1):
            ya, ca = stops[j]
            yb, cb = stops[j + 1]
            if y <= yb or j == len(stops) - 2:
                out.append(mix(ca, cb, clamp((y - ya) / max(1, yb - ya))))
                break
    return out


def dith_mix(a, mask, color, level, ox=0, oy=0):
    """Dithered tint: where mask, replace ~level of the pixels by color (array version)."""
    h, w = mask.shape
    thr = bayer(h, w, ox, oy)
    sel = mask & (thr < level)
    a[sel] = C(color)


def tint(a, mask, color, k):
    """True mix of the masked pixels toward color by k (build time only)."""
    cc = np.array(col(color), np.float32)
    sub = a[mask].astype(np.float32)
    a[mask] = np.clip(sub + (cc - sub) * k, 0, 255).astype(np.uint8)


def rampc(base):
    b = col(base)
    return dict(ol=mix(shade(b, .55), 'plum', .35), dk=mix(shade(b, .80), 'violet', .12),
                md=shade(b, .91), bs=b, lt=mix(b, 'white', .28),
                rim=mix(b, (255, 236, 160), .62))


def disc_px(c, cx, cy, r, color):
    c.circle(int(cx), int(cy), int(r), color)


# ================================================================== light / sky
SUN_TINT = (255, 232, 160)


def sky_layer(w, h, stops, sun, glow_r, seed):
    """Banded dithered gradient sky + baked sun glow (wide horizontal halo). Returns Canvas."""
    cv = Canvas(w, h, 'ink')
    cv.gradient(0, 0, w, h, stops_cols(stops, h, 16), soft=0.7)
    sx, sy = sun
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - sx) / 2.4) ** 2 + (yy - sy) ** 2)
    g = np.clip(1 - d / glow_r, 0, 1) ** 1.5 * 0.9
    a = cv.a
    thr = bayer(h, w)
    q = (np.clip(np.floor(g * 7 + (thr - 0.5)), 0, 7) / 7)[:, :, None]
    glowc = np.array((255, 236, 176), np.float32)
    a[:] = np.clip(a.astype(np.float32) * (1 - q) + glowc * q, 0, 255).astype(np.uint8)
    return cv


@lru_cache(maxsize=None)
def sun_sprite(r, low):
    """Big low sun: hot core, gold body, amber limb, cloud bands crossing it."""
    s = 2 * r + 3
    cv = new(s, s)
    cx = cy = r + 1
    yy, xx = np.mgrid[0:s, 0:s]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    a = cv.a
    m = d <= r + 0.3
    a[m] = (252, 178, 74) if low else (253, 196, 90)
    a[d <= r - 1.2] = (254, 208, 112)
    a[d <= r - 3.5] = (255, 228, 150)
    a[(d <= r * 0.55)] = (255, 244, 196)
    # gentle vertical warmth: lower half deeper
    lower = m & (yy > cy + r * 0.35)
    tint(a, lower, (250, 140, 70), 0.30)
    # cloud bands crossing the disc (thin, violet-coral)
    for k, (dy, ln, cc) in enumerate(((r * 0.30, 1.0, (236, 120, 110)), (r * 0.58, 1.0, (222, 104, 112)),
                                       (r * 0.80, 1.0, (206, 92, 112)), (-r * 0.25, 0.5, (246, 160, 120)))):
        y = int(cy + dy)
        xs = np.nonzero(m[y])[0]
        if len(xs):
            x0, x1 = xs[0], xs[-1]
            L = int((x1 - x0) * ln)
            off = 0 if k % 2 == 0 else (x1 - x0 - L)
            a[y, x0 + off:x0 + off + L + 1][m[y, x0 + off:x0 + off + L + 1]] = cc
    return to_layer(cv)


def streak_cloud(w, th, seed, pal):
    """Long thin sunset cloud: violet top, body, bright under-lit belly. pal=(top, body, belly, rim)."""
    h = th + 3
    cv = new(w, h)
    xs = np.arange(w)
    u = (xs + 0.5) / w
    env = np.sin(np.pi * u) ** 0.6
    n = noise1(w, 5, seed)
    n2 = noise1(w, 13, seed + 7)
    top = np.round(1 + (1 - env) * th + (n - 0.5) * 2.4 + (n2 - 0.5) * 2).astype(int)
    bot = np.round(th + 1 - (1 - env) * 1.5 + (noise1(w, 9, seed + 3) - 0.5) * 1.2).astype(int)
    a = cv.a
    for x in range(w):
        t0, b0 = max(0, top[x]), min(h - 1, bot[x])
        if b0 - t0 < 0 or env[x] < 0.12:
            continue
        a[t0:b0 + 1, x] = col(pal[1])
        a[t0, x] = col(pal[0])
        a[b0, x] = col(pal[2])
        if b0 - t0 >= 3:
            a[b0 - 1, x] = col(mix(pal[1], pal[2], 0.5)) if (x + seed) % 2 else col(pal[1])
        if env[x] > 0.5 and b0 + 1 < h and (x * 7 + seed) % 5 == 0:
            a[b0 + 1, x] = col(pal[3])
    return to_layer(cv)


def heights_mesa(w, specs, seed, rough=1.6):
    """Height line (screen y of the top) of flat-topped mesas: (x0, x1, top, cliff, steep, talus)."""
    xs = np.arange(w, dtype=float)
    y = np.full(w, 1e9)
    for (x0, x1, top, cliff, steep, talus) in specs:
        d = np.maximum(np.maximum(x0 - xs, xs - x1), 0)
        c_ = np.minimum(d * steep, cliff)
        t = np.maximum(0, d - cliff / steep) * talus
        y = np.minimum(y, top + c_ + t)
    y = y + (noise1(w, 4, seed) - 0.5) * rough
    return y


def shade_silhouette(cv, top, base_y, ramp, seed, rim_w=2, band=5, haze=None, haze_rows=8, ox=0):
    """Fill a heightfield silhouette with stratified rock: lit right flanks, shadowed left flanks,
    lit top caps, banded strata, haze at the foot. ramp: (shadow, dark, mid, lit, rim)."""
    h, w = cv.h, cv.w
    yy = np.arange(h)[:, None]
    m = yy >= np.round(top)[None, :]
    m &= yy < base_y
    a = cv.a
    # strata bands (wavy)
    wav = (noise1(w, 17, seed) - 0.5) * 4
    strata = ((yy + wav[None, :]) // band).astype(int) % 2
    a[m & (strata == 0)] = col(ramp[2])
    a[m & (strata == 1)] = col(ramp[1])
    # erosion streaks
    st = vnoise(h, w, 9, 1.5, seed + 5) > 0.68
    a[m & st] = col(ramp[1])
    # flanks: distance to the run's left / right edge per row
    for y in range(h):
        row = m[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        for s0, e0 in zip(starts, ends):
            ln = e0 - s0
            rw = min(rim_w + (2 if ln > 30 else 0), ln // 3)
            if rw > 0:
                a[y, e0 - rw:e0] = col(ramp[3])
                a[y, e0 - 1] = col(ramp[4])
            lw = min(3, ln // 3)
            if lw > 0:
                a[y, s0:s0 + lw] = col(ramp[0])
    # lit top caps
    tops = np.round(top).astype(int)
    for x in range(w):
        t = tops[x]
        if 0 <= t < base_y:
            a[t, x] = col(ramp[4])
            if t + 1 < base_y and m[t + 1, x]:
                a[t + 1, x] = col(ramp[3])
    if haze is not None:
        for i in range(haze_rows):
            y = base_y - 1 - i
            lvl = 0.75 * (1 - i / haze_rows)
            rowm = np.zeros((h, w), bool)
            rowm[y] = m[y]
            dith_mix(a, rowm, haze, lvl, ox)
    return m


# ================================================================== SHOT A: APPLELOOSA
GY = 147          # building base = boardwalk top
SY = 150          # street top
TW = 590          # town layer width
CAMA = 260
P_SKY, P_SUN, P_FAR, P_HILL, P_BUTTE, P_PLAIN, P_FG = 0.05, 0.08, 0.12, 0.22, 0.35, 0.55, 1.35
SUN_A = (300, 98, 18)


def cam_a(f):
    return int(round(CAMA * smooth(seg(f, 74, 214))))


SKY_A_STOPS = [(0, (66, 46, 118)), (24, (104, 64, 150)), (46, (160, 84, 152)), (64, (214, 104, 132)),
               (80, (240, 132, 104)), (96, (249, 164, 86)), (110, (251, 196, 104)), (124, (253, 220, 142))]


@lru_cache(maxsize=1)
def _sky_a():
    w = W + 30
    cv = sky_layer(w, 130, SKY_A_STOPS, (SUN_A[0], SUN_A[1]), 70, 1)
    return cv


CLOUD_PALS = {
    'hi': ((132, 74, 146), (226, 124, 168), (255, 182, 196), (255, 214, 220)),
    'md': ((150, 78, 128), (222, 112, 124), (255, 190, 150), (255, 226, 180)),
    'lo': ((160, 82, 118), (204, 98, 112), (255, 214, 132), (255, 238, 170)),
}
# (x, y, w, thickness, seed, pal, drift px/frame)
CLOUDS_A = [(10, 24, 70, 3, 1, 'hi', 0.10), (100, 31, 96, 4, 2, 'hi', 0.12), (230, 20, 60, 3, 3, 'hi', 0.09),
            (300, 36, 80, 3, 4, 'hi', 0.11), (40, 52, 110, 4, 5, 'md', 0.14), (190, 58, 90, 3, 6, 'md', 0.13),
            (330, 50, 70, 3, 7, 'md', 0.15), (130, 76, 120, 3, 8, 'lo', 0.17), (270, 84, 90, 2, 9, 'lo', 0.18),
            (0, 90, 70, 2, 10, 'lo', 0.16), (360, 70, 60, 2, 11, 'lo', 0.15)]


@lru_cache(maxsize=None)
def _cloud(w, th, seed, pal):
    return streak_cloud(w, th, seed, CLOUD_PALS[pal])


def draw_clouds(c, f, clouds, cam, par, span=430):
    for (x, y, w, th, sd, pal, dr) in clouds:
        lay = _cloud(w, th, sd, pal)
        sx = (x + dr * f - cam * par) % span - 70
        comp(c, lay, int(sx), y)


# --- far mesas (violet)
@lru_cache(maxsize=1)
def _far_a():
    w = 360
    cv = new(w, 124)
    specs = [(0, 40, 101, 6, 3, 0.5), (70, 96, 104, 5, 3, 0.45), (128, 190, 99, 7, 2.5, 0.5),
             (232, 252, 106, 4, 3, 0.4), (282, 340, 109, 4, 3, 0.5)]
    top = heights_mesa(w, specs, 21, 1.0)
    top = np.minimum(top, 116 + (noise1(w, 9, 22) - 0.5) * 3)
    shade_silhouette(cv, top, 124, [(152, 98, 162), (166, 108, 170), (176, 116, 174), (206, 136, 170),
                                    (236, 162, 168)], 23, rim_w=1, band=7,
                     haze=(232, 160, 150), haze_rows=7)
    return to_layer(cv)


# --- Macintosh Hills (pink, snow-tipped)
@lru_cache(maxsize=1)
def _hills_a():
    w = 400
    cv = new(w, 126)
    xs = np.arange(w, dtype=float)
    y = np.full(w, 121.0)
    for (cx, hh, sp) in ((40, 24, 30), (88, 30, 26), (130, 22, 24), (176, 27, 30), (226, 16, 26),
                         (20, 12, 16), (110, 14, 14), (260, 9, 30), (330, 6, 40)):
        y -= hh * np.exp(-((xs - cx) / sp) ** 2)
    y += (noise1(w, 6, 31) - 0.5) * 1.6
    top = np.round(y).astype(int)
    a = cv.a
    yy = np.arange(126)[:, None]
    m = (yy >= top[None, :]) & (yy < 126)
    slope = np.gradient(y)
    body, sh, lt, hi, snow = (206, 120, 156), (178, 100, 150), (232, 146, 164), (248, 182, 176), (255, 228, 232)
    a[m] = body
    for x in range(w):
        s = slope[x]
        t = top[x]
        if s > 0.25:
            a[t:t + 30, x][m[t:t + 30, x]] = lt
            if s > 0.7:
                a[t:t + 4, x][m[t:t + 4, x]] = hi
        elif s < -0.3:
            a[t:t + 40, x][m[t:t + 40, x]] = sh
        a[t, x] = hi if s > -0.2 else lt
        if t < 98:
            a[t:t + max(1, 99 - t), x] = snow
            if s < -0.2:
                a[t + 1:t + max(2, 99 - t), x] = (232, 206, 226)
    # gullies
    g = (vnoise(126, w, 10, 2, 33) > 0.7) & m
    tint(a, g, (150, 86, 140), 0.35)
    for i in range(8):
        rowm = np.zeros(m.shape, bool)
        rowm[125 - i] = m[125 - i]
        dith_mix(a, rowm, (236, 168, 150), 0.7 * (1 - i / 8))
    return to_layer(cv)


# --- rust buttes
@lru_cache(maxsize=1)
def _buttes_a():
    w = 420
    cv = new(w, 128)
    specs = [(36, 58, 86, 22, 4, 0.6), (64, 70, 96, 10, 3, 0.5), (186, 250, 94, 16, 3, 0.55),
             (262, 270, 102, 8, 3, 0.5), (296, 314, 86, 24, 5, 0.6), (396, 420, 96, 14, 3, 0.5)]
    top = heights_mesa(w, specs, 41, 1.2)
    top = np.minimum(top, 122 + (noise1(w, 7, 42) - 0.5) * 2)
    shade_silhouette(cv, top, 128, [(120, 56, 70), (150, 72, 72), (172, 86, 76), (226, 128, 82),
                                    (252, 186, 112)], 43, rim_w=2, band=4,
                     haze=(234, 154, 118), haze_rows=10)
    return to_layer(cv)


# --- ochre plain with rail line, telegraph poles, far orchards, scrub
@lru_cache(maxsize=1)
def _plain_a():
    w = 470
    cv = new(w, 152)
    a = cv.a
    y0 = 118
    cv.gradient(0, y0, w, 152 - y0, [(238, 182, 124), (228, 164, 100), (218, 148, 86), (206, 132, 74),
                                     (196, 120, 68)], soft=0.8)
    R = rng(51)
    # streaky furrows / grass tufts
    for i in range(260):
        x = int(R.random() * w)
        y = int(y0 + 2 + (R.random() ** 0.7) * 31)
        L = 2 + int(R.random() * 6 * (y - y0) / 30)
        cc = mix((226, 160, 96), (176, 108, 70), R.random())
        cv.hline(x, x + L, y, cc)
    # far orchard rows (right part)
    for row, (yb, sp, cs) in enumerate(((121, 5, 1), (124, 6, 1), (128, 7, 2))):
        for x in range(240, w - 4, sp):
            if hash01(row, x, 3) < 0.08:
                continue
            xx = x + (row * 2)
            cv.rect(xx - cs, yb - 1 - cs, 2 * cs + 1, cs + 1, (96, 120, 64))
            cv.px(xx + cs, yb - 1 - cs, (172, 170, 80))
            cv.px(xx - cs, yb, (150, 96, 70))
            if hash01(row, x) > 0.5:
                cv.px(xx, yb - 1 - cs, (220, 70, 60))
            cv.hline(xx - cs - 3, xx - cs - 1, yb, (182, 110, 72))
    # sage scrub
    for i in range(70):
        x = int(R.random() * w)
        y = int(y0 + 4 + R.random() * 28)
        s = 1 + int((y - y0) / 12)
        cv.hline(x - s - 4, x - 1, y, (184, 114, 76))
        cv.rect(x - s, y - s, 2 * s + 1, s + 1, (118, 118, 82))
        cv.px(x + s, y - s, (196, 182, 112))
        cv.px(x, y - s, (146, 150, 96))
    # rail line
    ry = 131
    for x in range(0, w, 3):
        cv.rect(x, ry, 2, 1, (128, 84, 66))
    cv.hline(0, w - 1, ry - 1, (176, 160, 160))
    cv.hline(0, w - 1, ry - 2, (120, 96, 104))
    cv.hline(0, w - 1, ry + 1, (196, 140, 98))
    # telegraph poles + wire
    poles = list(range(8, w, 38))
    for i, x in enumerate(poles):
        cv.vline(x, ry - 16, ry - 2, (110, 72, 70))
        cv.hline(x - 2, x + 2, ry - 15, (110, 72, 70))
        cv.px(x + 1, ry - 14, (236, 186, 120))
        cv.hline(x - 8, x - 1, ry - 1, (190, 120, 80))
        if i + 1 < len(poles):
            x2 = poles[i + 1]
            for xx in range(x, x2):
                u = (xx - x) / (x2 - x)
                cv.px(xx, ry - 15 + int(round(3 * 4 * u * (1 - u))), (124, 80, 96))
    # a far farmstead (barn + silo)
    bx = 150
    cv.rect(bx, 114, 12, 7, (168, 80, 78))
    cv.tri((bx - 1, 114), (bx + 12, 114), (bx + 6, 110), (128, 66, 82))
    cv.px(bx + 11, 116, (236, 150, 110))
    cv.rect(bx + 14, 109, 4, 12, (176, 160, 168))
    cv.circle(bx + 15, 109, 2, (150, 136, 150))
    cv.vline(bx + 17, 110, 120, (226, 206, 190))
    return to_layer(cv)


# ------------------------------------------------------------------ town pieces
def clapboard(cv, x, y, w, h, r, seed, board=4):
    """Horizontal clapboard planks with staggered joints, lit top row, shadow line, knots."""
    x0, y0, x1, y1 = cv._clip(x, y, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    a = cv.a
    rows = (np.arange(y0, y1) - y) % board
    sub = a[y0:y1, x0:x1]
    sub[:] = col(r['bs'])
    sub[rows == 0] = col(r['lt'])
    sub[rows == board - 1] = col(r['md'])
    for yy in range(y0, y1):
        bi = (yy - y) // board
        if (yy - y) % board == 0:
            continue
        off = int(hash01(seed, bi) * 15)
        for jx in range(x + off, x + w, 15):
            if x0 <= jx < x1:
                a[yy, jx] = col(r['dk'])
        if hash01(seed, bi, 9) > 0.7:
            kx = x + int(hash01(seed, bi, 5) * w)
            if x0 <= kx < x1 and (yy - y) % board == 1:
                a[yy, kx] = col(r['md'])


def vplanks(cv, x, y, w, h, r, seed):
    x0, y0, x1, y1 = cv._clip(x, y, w, h)
    a = cv.a
    a[y0:y1, x0:x1] = col(r['bs'])
    for xx in range(x0, x1):
        k = (xx - x) % 3
        if k == 2:
            a[y0:y1, xx] = col(r['md'])
        elif k == 0:
            a[y0:y1, xx] = col(r['lt']) if hash01(seed, xx) > 0.4 else col(r['bs'])


WIN = []      # filled by the cached builders: (x, y, w, h, t_on, kind) in layer coords


def window(cv, x, y, w, h, frame, glass_seed, wins, t_on, round_top=True, shutters=None, sill=True):
    """Rounded-top window with frame, cross mullions, sky reflection; registers itself for lighting."""
    fr = rampc(frame)
    cv.rect(x - 1, y - 1, w + 2, h + 2, fr['dk'])
    cv.rect(x, y, w, h, (70, 50, 96))
    # reflection of the sunset sky (violet -> coral glint)
    cv.rect(x, y + h // 2, w, h - h // 2, (92, 60, 110))
    for i in range(min(w, h)):
        cv.px(x + w - 1 - i, y + i, (214, 126, 140)) if i % 3 != 2 else None
    if round_top:
        cv.px(x - 1, y - 1, KEY) if False else None
        cv.px(x, y, fr['dk'])
        cv.px(x + w - 1, y, fr['dk'])
        cv.hline(x, x + w - 1, y - 1, fr['bs'])
        cv.hline(x + 1, x + w - 2, y - 2, fr['bs'])
        cv.px(x - 1, y - 1, fr['dk'])
        cv.px(x + w, y - 1, fr['dk'])
    cv.vline(x + w // 2, y, y + h - 1, fr['bs'])
    cv.hline(x, x + w - 1, y + h // 2, fr['bs'])
    if sill:
        cv.hline(x - 2, x + w + 1, y + h + 1, fr['lt'])
        cv.hline(x - 1, x + w, y + h + 2, fr['dk'])
        cv.px(x + w + 1, y + h + 1, fr['rim'])
    if shutters:
        sr = rampc(shutters)
        for sx in (x - 4, x + w + 1):
            cv.rect(sx, y - 1, 3, h + 2, sr['bs'])
            for yy in range(y, y + h + 1, 2):
                cv.hline(sx, sx + 2, yy, sr['md'])
            cv.vline(sx + 2, y - 1, y + h, sr['rim'] if sx > x else sr['lt'])
    wins.append((x, y, w, h, t_on, round_top))


def door(cv, x, y, w, h, wood, frame):
    fr = rampc(frame)
    wd = rampc(wood)
    cv.rect(x - 1, y - 1, w + 2, h + 1, fr['dk'])
    cv.rect(x, y, w, h, wd['bs'])
    cv.px(x, y, fr['dk'])
    cv.px(x + w - 1, y, fr['dk'])
    cv.hline(x, x + w - 1, y - 2, fr['bs'])
    for yy in (y + 2, y + h // 2 + 1):
        cv.frame(x + 1, yy, w // 2 - 1, h // 2 - 3, wd['dk'])
        cv.frame(x + w // 2 + 1 - (w % 2 == 0), yy, w // 2 - 1, h // 2 - 3, wd['dk'])
    cv.vline(x + w - 1, y + 1, y + h - 1, wd['rim'])
    cv.px(x + w - 3, y + h // 2 + 1, (249, 194, 43))


def awning(cv, x, w, y, colors, posts, post_col, depth=3):
    """Porch roof seen from the front: striped edge band, scallops, dark underside, posts."""
    pr = rampc(post_col)
    for xx in range(x, x + w):
        cc = colors[((xx - x) // 4) % len(colors)]
        r = rampc(cc)
        cv.vline(xx, y, y + depth - 1, r['bs'])
        cv.px(xx, y, r['lt'])
        if (xx - x) % 4 in (1, 2):
            cv.px(xx, y + depth, r['md'])
    cv.px(x + w - 1, y, (255, 236, 160))
    cv.hline(x, x + w - 1, y + depth + 1, (96, 62, 92))
    for px_ in posts:
        cv.rect(px_, y + depth + 1, 2, GY - (y + depth + 1), pr['bs'])
        cv.vline(px_ + 1, y + depth + 1, GY - 1, pr['rim'])
        cv.vline(px_, y + depth + 1, GY - 1, pr['md'])
        cv.px(px_, GY - 1, pr['dk'])


def sign_board(cv, x, y, w, h, board, rim, icon=None):
    br = rampc(board)
    cv.round_rect(x - 1, y - 1, w + 2, h + 2, rampc(rim)['dk'], r=1)
    cv.rect(x, y, w, h, br['bs'])
    cv.hline(x, x + w - 1, y, br['lt'])
    cv.hline(x, x + w - 1, y + h - 1, br['md'])
    cv.vline(x + w, y, y + h - 1, (255, 236, 160))
    if icon is not None:
        icon.draw(cv, x + w // 2, y + h // 2, anchor='center')


ICON = {
    'apple': sprite('''
        ...g.
        .rRr.
        rRRwr
        rRRRr
        .rrr.
    ''', {'g': 'leaf', 'r': 'crimson', 'R': 'red', 'w': 'white'}),
    'bed': sprite('''
        h......
        hww....
        hwwpppp
        hbbbbbh
        h.....h
    ''', {'h': 'brew_mid', 'w': 'white', 'p': 'lavender', 'b': 'royal_lt'}),
    'pie': sprite('''
        .ccccc.
        cwcscwc
        ttttttt
        .ttttt.
    ''', {'c': 'sand', 'w': 'cream', 's': 'red', 't': 'clay'}),
    'salt': sprite('''
        ..wwww..
        .wwwwwws
        wwwwwwss
        wllllwss
        wllllws.
        wllllw..
        .ssss...
    ''', {'w': 'white', 'l': 'mist', 's': 'stone'}),
    'mug': sprite('''
        wwww.
        aaaah
        aaaa.h
        aaaah
        aaaa.
    ''', {'w': 'white', 'a': 'amber', 'h': 'cream'}),
    'shoe': sprite('''
        .ggg.
        g...g
        g...g
        g...g
        gg.gg
    ''', {'g': 'gold'}),
    'star': sprite('''
        ..g..
        ggggg
        .ggg.
        g...g
    ''', {'g': 'gold'}),
    'train': sprite('''
        p.....
        pp.ccc
        pppcwc
        ppppcc
        .b..b.
    ''', {'p': 'hot_pink', 'c': 'lavender', 'w': 'butter', 'b': 'ink'}),
    'barrel': sprite('''
        .tttt.
        bbbbbb
        tttttt
        bbbbbb
        .tttt.
    ''', {'t': 'latte', 'b': 'brew_mid'}),
    'cherry': sprite('''
        ...gg
        ..g..
        .g.g.
        rr.rr
        rw.rw
    ''', {'g': 'leaf', 'r': 'crimson', 'w': 'pink'}),
}


def crown(cv, x, w, top, r, trim, style):
    """False-front crown shapes on top of the facade rect (x, top..)."""
    tr = rampc(trim)
    if style == 'step':
        sw = max(4, w // 5)
        cv.rect(x + sw, top - 6, w - 2 * sw, 6, r['bs'])
        cv.hline(x + sw - 1, x + w - sw, top - 7, tr['bs'])
        cv.hline(x + sw - 1, x + w - sw, top - 8, tr['lt'])
        cv.px(x + w - sw, top - 8, tr['rim'])
    elif style == 'arch':
        cx = x + w // 2
        rr = w // 2 - 4
        for xx in range(x + 4, x + w - 4):
            dx = (xx - cx + 0.5) / rr
            hgt = int(round(8 * math.sqrt(max(0.0, 1 - dx * dx))))
            if hgt > 0:
                cv.vline(xx, top - hgt, top - 1, r['bs'])
                cv.px(xx, top - hgt - 1, tr['bs'])
                cv.px(xx, top - hgt - 2, tr['lt'] if xx < cx + rr // 2 else tr['rim'])
    elif style == 'gable':
        cx = x + w // 2
        hh = w // 4
        for xx in range(x + 2, x + w - 2):
            hgt = int(hh - abs(xx - cx) * hh / (w / 2 - 2))
            if hgt > 0:
                cv.vline(xx, top - hgt, top - 1, r['bs'])
                cv.px(xx, top - hgt - 1, tr['bs'] if xx <= cx else tr['rim'])
                cv.px(xx, top - hgt - 2, tr['dk'])
    # cornice with dentils
    cv.rect(x - 1, top, w + 2, 3, tr['bs'])
    cv.hline(x - 1, x + w, top, tr['lt'])
    for xx in range(x, x + w, 3):
        cv.px(xx, top + 3, tr['md'])
    cv.px(x + w, top, tr['rim'])
    cv.px(x + w, top + 1, tr['rim'])


def building(cv, x, w, h, color, trim, style, seed, wins, ton, floors=1, awn=None, sign=None,
             balcony=False, door_x=None, wins_x=(), win_y=None, upper_x=(), stove=None):
    r = rampc(color)
    top = GY - h
    clapboard(cv, x, top, w, h, r, seed)
    crown(cv, x, w, top, r, trim, style)
    # corner boards (trim) and rim light on the right edge
    tr = rampc(trim)
    cv.rect(x, top + 3, 2, h - 3, tr['md'])
    cv.vline(x, top + 3, GY - 1, r['ol'])
    cv.rect(x + w - 2, top + 3, 2, h - 3, tr['bs'])
    cv.vline(x + w - 1, top - 1, GY - 1, r['rim'])
    cv.vline(x + w, top + 3, GY - 1, mix(r['rim'], 'white', .3))
    if floors == 2:
        mid = GY - 34
        cv.hline(x + 1, x + w - 2, mid, tr['md'])
        cv.hline(x + 1, x + w - 2, mid - 1, tr['lt'])
    if sign:
        sx, sy, sw, sh, icon, board = sign
        sign_board(cv, x + sx, top + sy, sw, sh, board, trim, ICON.get(icon))
    for i, ux in enumerate(upper_x):
        window(cv, x + ux, top + 14 if floors == 1 else GY - 54, 5, 8, trim, seed + i, wins,
               ton + 36 * (i % 3) + 18 * (seed % 2), shutters=None)
    if balcony:
        by = GY - 33
        cv.rect(x - 1, by, w + 2, 2, tr['md'])
        cv.hline(x - 1, x + w, by - 7, tr['lt'])
        cv.hline(x - 1, x + w, by - 6, tr['md'])
        for xx in range(x, x + w, 3):
            cv.vline(xx, by - 5, by - 1, tr['bs'])
        cv.vline(x + w, by - 7, by + 1, (255, 236, 160))
    if awn:
        cols_, posts = awn
        awning(cv, x - 1, w + 2, GY - 24, cols_, [x + p for p in posts], 'tan')
    for i, wx in enumerate(wins_x):
        window(cv, x + wx, win_y if win_y else GY - 19, 6, 9, trim, seed + 10 + i, wins,
               ton + 18 * i, shutters=None)
    if door_x is not None:
        door(cv, x + door_x, GY - 15, 7, 15, 'latte', trim)
    if stove:
        sx = x + stove
        cv.rect(sx, top - 12, 2, 12, (90, 76, 96))
        cv.vline(sx + 1, top - 12, top - 1, (200, 150, 140))
        cv.rect(sx - 1, top - 13, 4, 1, (70, 58, 80))
    return top


@lru_cache(maxsize=1)
def _town_a():
    """Main Street: returns (layer, windows, extra dict)."""
    cv = new(TW, H)
    wins = []
    ex = {}
    # ---- general store (pink), stepped crown
    building(cv, 3, 40, 52, (246, 170, 196), 'cream', 'step', 3, wins, 126, awn=(('cream', 'hot_pink'), (0, 19, 38)),
             sign=(6, 6, 28, 9, 'apple', 'cream'), wins_x=(4, 28), door_x=16, upper_x=(8, 27))
    # ---- hotel (mint), arched, 2 floors + balcony
    building(cv, 47, 44, 72, (150, 222, 190), 'cream', 'arch', 4, wins, 108, floors=2, balcony=True,
             sign=(8, 5, 28, 9, 'bed', 'cream'), wins_x=(5, 33), door_x=18, upper_x=(5, 19, 33), stove=36)
    # ---- bakery (sky blue), gable crown
    building(cv, 95, 34, 50, (160, 206, 244), 'blush', 'gable', 5, wins, 144, awn=(('blush', 'white'), (0, 32)),
             sign=(5, 6, 24, 8, 'pie', 'blush'), wins_x=(4, 24), door_x=14, upper_x=(14,), stove=6)
    # ---- alley 131-160: water trough, barrels, parked wagon (no horse)
    wx = 133
    cv.rect(wx, GY - 22, 24, 10, (232, 232, 240))                     # canvas cover
    cv.ellipse(wx + 12, GY - 22, 12, 4, (232, 232, 240))
    for i in range(4):
        cv.vline(wx + 2 + i * 6, GY - 25, GY - 13, (196, 188, 206))
    cv.vline(wx + 23, GY - 24, GY - 13, (255, 244, 214))
    cv.rect(wx - 1, GY - 13, 26, 5, (150, 96, 70))
    cv.hline(wx - 1, wx + 24, GY - 13, (196, 134, 92))
    for (cx_, rr) in ((wx + 4, 5), (wx + 20, 5)):
        cv.circle(cx_, GY - 5, rr, (110, 70, 66), fill=False)
        cv.circle(cx_, GY - 5, rr - 1, (176, 120, 82), fill=False)
        for k in range(4):
            ang = k * math.pi / 4
            cv.line(cx_, GY - 5, cx_ + int(round(math.cos(ang) * (rr - 1))), GY - 5 + int(round(math.sin(ang) * (rr - 1))), (150, 96, 70))
        cv.px(cx_ + rr, GY - 6, (255, 220, 150))
    cv.line(wx - 1, GY - 10, wx - 9, GY - 4, (150, 96, 70))           # wagon tongue resting
    # ---- SALT BLOCK saloon (peach), big stepped crown + pediment, balcony, batwing doors
    sx0, sw = 160, 72
    top = building(cv, sx0, sw, 76, (252, 178, 150), 'cream', 'step', 6, wins, 72, floors=2, balcony=True,
                   upper_x=(6, 20, 47, 61), wins_x=(6, 59))
    # pediment star + saloon sign board with the salt-block + mug icons
    sign_board(cv, sx0 + 14, top + 3, 44, 12, (110, 60, 74), 'gold', None)
    ICON['salt'].draw(cv, sx0 + 36, top + 3 + 6, anchor='center')
    ICON['mug'].draw(cv, sx0 + 22, top + 4 + 6, anchor='center')
    ICON['mug'].flipped().draw(cv, sx0 + 51, top + 4 + 6, anchor='center')
    ICON['star'].draw(cv, sx0 + 36, top - 7, anchor='center')
    awning(cv, sx0 - 1, sw + 2, GY - 24, ('rose', 'cream'), [sx0, sx0 + 22, sx0 + 48, sx0 + 70], 'tan')
    # doorway (interior) for the batwing doors
    dx0 = sx0 + 30
    cv.rect(dx0 - 1, GY - 19, 14, 19, (176, 104, 96))
    cv.rect(dx0, GY - 18, 12, 18, (70, 40, 52))
    cv.hline(dx0 - 1, dx0 + 12, GY - 20, (255, 236, 214))
    ex['saloon_door'] = (dx0, GY - 18)
    ex['lanterns'] = [(sx0 + 25, GY - 15), (sx0 + 46, GY - 15)]
    ex['saloon_sign'] = (sx0 + 36, top + 9)
    # ---- clock tower (cream body, pink roof)
    tx0 = 236
    building(cv, tx0, 30, 44, (255, 236, 196), 'rose', 'flat', 7, wins, 162, door_x=11, wins_x=(3, 22),
             win_y=GY - 26)
    shaft_x, shaft_w = tx0 + 5, 20
    st = rampc((255, 232, 190))
    clapboard(cv, shaft_x, 62, shaft_w, GY - 44 - 62, st, 8)
    cv.vline(shaft_x, 62, GY - 45, st['ol'])
    cv.vline(shaft_x + shaft_w - 1, 62, GY - 45, st['rim'])
    cv.rect(shaft_x - 2, 88, shaft_w + 4, 3, rampc('rose')['bs'])
    cv.hline(shaft_x - 2, shaft_x + shaft_w + 1, 88, rampc('rose')['lt'])
    # clock face
    ccx, ccy = shaft_x + shaft_w // 2, 74
    cv.circle(ccx, ccy, 7, rampc('rose')['dk'])
    cv.circle(ccx, ccy, 6, (201, 162, 74))
    cv.circle(ccx, ccy, 5, (255, 248, 230))
    for k in range(12):
        ang = k * math.pi / 6
        cv.px(ccx + int(round(math.sin(ang) * 4.4)), ccy - int(round(math.cos(ang) * 4.4)),
              (150, 120, 120) if k % 3 else (110, 60, 74))
    ex['clock'] = (ccx, ccy)
    # belfry with bell
    cv.rect(shaft_x - 1, 50, shaft_w + 2, 12, rampc('rose')['bs'])
    cv.rect(shaft_x + 2, 52, shaft_w - 4, 9, (96, 56, 86))
    cv.rect(shaft_x + 1, 61, shaft_w - 2, 2, rampc('rose')['lt'])
    for k in (shaft_x + 6, shaft_x + 13):
        cv.vline(k, 52, 60, rampc('rose')['bs'])
    cv.ellipse(ccx, 56, 3, 3, (201, 162, 74))
    cv.hline(ccx - 3, ccx + 3, 59, (249, 194, 43))
    cv.px(ccx + 2, 55, (255, 240, 160))
    cv.vline(shaft_x + shaft_w, 50, 62, (255, 236, 160))
    # pyramid roof
    rr = rampc('hot_pink')
    for yy in range(30, 50):
        hw = int((yy - 30) * (shaft_w + 6) / 2 / 20)
        cv.hline(ccx - hw, ccx + hw, yy, rr['bs'])
        cv.hline(ccx - hw, ccx - hw + max(0, hw // 2), yy, rr['md'])
        cv.px(ccx + hw, yy, rr['rim'])
        if yy % 3 == 0:
            for xx in range(ccx - hw, ccx + hw, 3):
                cv.px(xx + (yy // 3) % 2, yy, rr['dk'])
    cv.hline(ccx - (shaft_w + 6) // 2 - 1, ccx + (shaft_w + 6) // 2 + 1, 50, rr['dk'])
    cv.vline(ccx, 24, 30, (201, 162, 74))
    cv.circle(ccx, 27, 1, (249, 194, 43))
    ex['finial'] = (ccx, 24)
    # ---- stage with bunting poles (bunting drawn per frame)
    gx0 = 272
    stg = rampc((214, 160, 120))
    cv.rect(gx0, GY - 10, 50, 10, stg['bs'])                          # platform front
    for xx in range(gx0, gx0 + 50, 3):
        cv.vline(xx, GY - 9, GY - 1, stg['md'])
    cv.hline(gx0, gx0 + 49, GY - 10, stg['lt'])
    cv.hline(gx0, gx0 + 49, GY - 11, (255, 226, 170))
    # skirt with rosette (star) and swags
    for k in range(5):
        cx_ = gx0 + 5 + k * 10
        for xx in range(cx_ - 5, cx_ + 5):
            u = (xx - cx_ + 5) / 10
            cv.px(xx, GY - 8 + int(round(3 * 4 * u * (1 - u))), (240, 79, 120))
        cv.px(cx_ - 5, GY - 8, (249, 194, 43))
    # back wall: a rounded proscenium shell (lavender with cream stripes)
    shell = rampc((196, 160, 232))
    cxs = gx0 + 25
    for yy in range(96, GY - 11):
        for xx in range(gx0 + 3, gx0 + 47):
            dx = (xx - cxs + 0.5) / 22
            dy = (yy - (GY - 11)) / 45
            if dx * dx + dy * dy <= 1:
                k = int((math.atan2(-dy, dx) / math.pi) * 9)
                cv.px(xx, yy, shell['bs'] if k % 2 else shell['lt'])
    for yy in range(96, GY - 11):
        for xx in range(gx0 + 3, gx0 + 47):
            dx = (xx - cxs + 0.5) / 22
            dy = (yy - (GY - 11)) / 45
            d = dx * dx + dy * dy
            if 0.82 < d <= 1:
                cv.px(xx, yy, (255, 236, 200) if xx > cxs else (226, 196, 176))
    cv.rect(cxs - 15, GY - 30, 30, 19, (110, 60, 96))                # curtain opening
    for xx in range(cxs - 15, cxs + 15):
        if (xx - cxs) % 4 == 0:
            cv.vline(xx, GY - 30, GY - 12, (150, 70, 110))
    cv.rect(cxs - 17, GY - 32, 6, 21, (214, 60, 90))                 # curtains
    cv.rect(cxs + 11, GY - 32, 6, 21, (214, 60, 90))
    cv.vline(cxs - 12, GY - 32, GY - 12, (250, 120, 140))
    cv.vline(cxs + 16, GY - 32, GY - 12, (250, 140, 150))
    cv.hline(cxs - 17, cxs + 16, GY - 33, (249, 194, 43))
    ICON['star'].draw(cv, cxs, 102, anchor='center')
    for px_ in (gx0 + 1, gx0 + 47):
        cv.rect(px_, 98, 2, GY - 98, (140, 92, 72))
        cv.vline(px_ + 1, 98, GY - 1, (236, 170, 110))
        cv.circle(px_ + 1, 97, 1, (249, 194, 43))
    ex['stage'] = (gx0 + 2, gx0 + 48, 99)
    ex['stage_lamps'] = [(gx0 + 2, 104), (gx0 + 48, 104)]
    # ---- water tower (brown staves on a trestle)
    wt0, wtw = 330, 30
    leg = rampc((128, 84, 66))
    for lx in (wt0 + 3, wt0 + 11, wt0 + 19, wt0 + 26):
        cv.rect(lx, 88, 2, GY - 88, leg['bs'])
        cv.vline(lx + 1, 88, GY - 1, leg['rim'] if lx > wt0 + 15 else leg['lt'])
    for yy in (100, 116, 132):
        cv.hline(wt0 + 3, wt0 + 27, yy, leg['md'])
        cv.line(wt0 + 4, yy, wt0 + 12, yy + 15, leg['dk'])
        cv.line(wt0 + 26, yy, wt0 + 19, yy + 15, leg['dk'])
    tk = rampc((150, 96, 70))
    vplanks(cv, wt0, 64, wtw, 24, tk, 9)
    for yy in (67, 75, 83):
        cv.hline(wt0, wt0 + wtw - 1, yy, (74, 82, 94))
        cv.hline(wt0, wt0 + wtw - 1, yy - 1, (125, 136, 148))
    cv.vline(wt0 + wtw - 1, 64, 87, tk['rim'])
    cv.vline(wt0 + wtw - 2, 64, 87, tk['lt'])
    cv.vline(wt0, 64, 87, tk['ol'])
    cv.rect(wt0 - 1, 87, wtw + 2, 2, tk['dk'])
    for yy in range(52, 64):                                          # conical roof
        hw = int((yy - 52) * (wtw / 2 + 2) / 12)
        cv.hline(wt0 + wtw // 2 - hw, wt0 + wtw // 2 + hw, yy, (122, 70, 74))
        cv.px(wt0 + wtw // 2 + hw, yy, (236, 150, 110))
        cv.hline(wt0 + wtw // 2 - hw, wt0 + wtw // 2 - hw // 2, yy, (98, 58, 70))
    cv.vline(wt0 + wtw // 2, 48, 52, (110, 72, 70))
    for yy in range(66, GY, 2):                                        # ladder
        cv.hline(wt0 - 3, wt0 - 1, yy, leg['lt'])
    cv.vline(wt0 - 3, 64, GY - 1, leg['bs'])
    cv.vline(wt0 - 1, 64, GY - 1, leg['bs'])
    cv.rect(wt0 + wtw - 4, 88, 3, 8, (74, 82, 94))                    # spout
    cv.rect(wt0 + wtw - 3, 96, 8, 2, (74, 82, 94))
    cv.px(wt0 + wtw + 4, 96, (196, 205, 214))
    ex['water_tower'] = (wt0 + wtw // 2, 64)
    ex['spout'] = (wt0 + wtw + 4, 98)
    # ---- rail depot (mint with deep hipped rose roof)
    dp0, dpw = 368, 72
    dr = rampc((176, 230, 206))
    clapboard(cv, dp0, GY - 30, dpw, 30, dr, 11)
    cv.vline(dp0, GY - 30, GY - 1, dr['ol'])
    cv.vline(dp0 + dpw - 1, GY - 30, GY - 1, dr['rim'])
    rf = rampc((214, 96, 120))
    for yy in range(GY - 48, GY - 28):
        k = yy - (GY - 48)
        x0_ = dp0 - 4 - k // 3 + 10
        x1_ = dp0 + dpw + 3 + k // 3 - 10
        cv.hline(x0_, x1_, yy, rf['bs'])
        if yy % 3 == 0:
            for xx in range(x0_, x1_, 4):
                cv.px(xx + (yy // 3) % 2 * 2, yy, rf['md'])
        cv.px(x1_, yy, rf['rim'])
        cv.px(x0_, yy, rf['dk'])
    cv.hline(dp0 + 6, dp0 + dpw - 7, GY - 49, rf['lt'])
    cv.hline(dp0 - 6, dp0 + dpw + 5, GY - 28, (110, 56, 80))
    cv.hline(dp0 - 6, dp0 + dpw + 5, GY - 27, (150, 80, 96))
    for k in range(5):                                                 # roof brackets
        bx = dp0 + 2 + k * 17
        cv.line(bx, GY - 18, bx + 3, GY - 26, (226, 196, 176))
    # depot gable dormer with the train icon
    cv.rect(dp0 + dpw // 2 - 9, GY - 58, 18, 11, dr['bs'])
    cv.tri((dp0 + dpw // 2 - 11, GY - 58), (dp0 + dpw // 2 + 10, GY - 58), (dp0 + dpw // 2, GY - 64), rf['bs'])
    cv.line(dp0 + dpw // 2, GY - 64, dp0 + dpw // 2 + 10, GY - 58, rf['rim'])
    sign_board(cv, dp0 + dpw // 2 - 7, GY - 56, 14, 8, 'cream', 'rose', ICON['train'])
    for i, wx in enumerate((dp0 + 6, dp0 + 18, dp0 + 46, dp0 + 58)):
        window(cv, wx, GY - 22, 7, 10, 'cream', 30 + i, wins, 126 + 18 * i)
    door(cv, dp0 + 31, GY - 17, 10, 17, 'latte', 'cream')
    ex['depot'] = (dp0 + dpw // 2, GY - 64)
    ex['depot_lamps'] = [(dp0 - 3, GY - 22), (dp0 + dpw + 2, GY - 22)]
    # crates of apples + baggage cart on the depot platform
    for k, (bx, by) in enumerate(((dp0 + 2, GY - 5), (dp0 + 64, GY - 5), (dp0 + 68, GY - 10))):
        cv.rect(bx, by, 7, 5, (176, 120, 82))
        cv.hline(bx, bx + 6, by, (222, 160, 104))
        cv.vline(bx + 6, by, by + 4, (250, 200, 130))
        for j in range(3):
            cv.px(bx + 1 + j * 2, by - 1, (232, 59, 59))
            cv.px(bx + 2 + j * 2, by - 1, (174, 35, 52)) if j < 2 else None
    # ---- boardwalk under the town (planks), steps at the ends
    bw = rampc((196, 140, 98))
    for x0_, x1_ in ((1, 132), (158, 270), (270, 324), (364, 444)):
        cv.rect(x0_, GY, x1_ - x0_, 3, bw['bs'])
        cv.hline(x0_, x1_ - 1, GY, bw['lt'])
        for xx in range(x0_, x1_, 5):
            cv.px(xx, GY + 1, bw['dk'])
        cv.hline(x0_, x1_ - 1, GY + 2, bw['dk'])
        cv.hline(x0_, x1_ - 1, GY + 3, (110, 70, 80))
    # hitching posts + rails in front of the boardwalk
    for hx in (34, 150, 238, 350):
        for px_ in (hx, hx + 12):
            cv.rect(px_, SY - 2, 2, 9, (140, 92, 72))
            cv.vline(px_ + 1, SY - 2, SY + 6, (236, 170, 110))
        cv.hline(hx, hx + 13, SY - 1, (176, 120, 82))
        cv.hline(hx, hx + 13, SY - 2, (236, 186, 120))
    # barrels on the boardwalk
    for bx in (92, 157, 266, 445):
        cv.rect(bx, GY - 8, 6, 8, (150, 96, 70))
        cv.ellipse(bx + 2, GY - 8, 3, 1, (176, 120, 82))
        for yy in (GY - 6, GY - 2):
            cv.hline(bx, bx + 5, yy, (74, 82, 94))
        cv.vline(bx + 5, GY - 8, GY - 1, (236, 170, 110))
    # water trough
    cv.rect(140, SY - 1, 12, 5, (150, 96, 70))
    cv.hline(141, 150, SY - 1, (110, 160, 200))
    cv.px(147, SY - 1, (220, 240, 255))
    # ---- orchard ground behind the fence (irrigation channel)
    og = 448
    cv.gradient(og - 6, 118, TW - og + 6, GY - 118 + 3, [(216, 156, 96), (204, 138, 80), (186, 118, 70)], soft=0.8)
    for xx in range(og - 6, og + 6):
        cv.vline(xx, 118, 118 + max(0, og - xx), KEY)
    ex['ditch'] = (og, TW, 140)
    cv.hline(og, TW - 1, 140, (90, 120, 150))
    cv.hline(og, TW - 1, 141, (60, 90, 120))
    cv.hline(og, TW - 1, 139, (150, 100, 80))
    # fence along the orchard with a gate arch + apple sign
    fc = new(TW, H)
    fr = rampc((176, 120, 82))
    for xx in range(og, TW, 9):
        fc.rect(xx, GY - 9, 2, 10, fr['bs'])
        fc.px(xx + 1, GY - 9, fr['rim'])
    fc.hline(og, TW - 1, GY - 7, fr['lt'])
    fc.hline(og, TW - 1, GY - 3, fr['bs'])
    gx = og + 18
    fc.rect(gx, GY - 26, 2, 27, fr['bs'])
    fc.rect(gx + 22, GY - 26, 2, 27, fr['bs'])
    fc.vline(gx + 23, GY - 26, GY, fr['rim'])
    sign_board(fc, gx + 3, GY - 27, 18, 7, 'cream', 'tan', ICON['apple'])
    fc.hline(gx - 1, gx + 24, GY - 28, fr['lt'])
    ex['gate'] = (gx + 12, GY - 28)
    ex['fence'] = to_layer(fc)
    # windpump (tower; rotor drawn per frame)
    wp = 556
    for yy in range(72, GY):
        k = (yy - 72) / (GY - 72)
        hw = int(2 + k * 7)
        cv.px(wp - hw, yy, (110, 96, 110))
        cv.px(wp + hw, yy, (196, 170, 160))
        if yy % 8 == 0:
            cv.hline(wp - hw, wp + hw, yy, (140, 120, 130))
            cv.line(wp - hw, yy, wp + hw + 1, yy + 8, (128, 110, 124))
    cv.rect(wp - 3, 70, 6, 3, (125, 136, 148))
    cv.rect(wp - 12, 140, 9, 5, (125, 136, 148))                      # tank
    cv.hline(wp - 12, wp - 4, 140, (196, 205, 214))
    ex['windpump'] = (wp, 66)
    # ---- street: dirt with ruts, pebbles, long shadows slanting down-left
    st_cols = [(224, 160, 100), (214, 146, 88), (206, 136, 82)]
    cv.gradient(0, SY, TW, H - SY, st_cols, soft=0.8)
    R = rng(61)
    for i in range(240):
        x = int(R.random() * TW)
        y = SY + 2 + int(R.random() * (H - SY - 2))
        cv.px(x, y, (240, 186, 124) if R.random() > 0.5 else (176, 112, 74))
    for ry in (157, 158, 166, 167):
        for x in range(0, TW, 1):
            if noise1(TW, 7, ry)[x] > 0.35:
                cv.px(x, ry, (186, 120, 76) if ry in (157, 166) else (232, 172, 112))
    # occupancy / height above the base for shadows
    a, m = cv.a, ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    hgt = np.zeros(TW, int)
    for x in range(TW):
        col_m = m[:GY, x]
        idx = np.nonzero(col_m)[0]
        if len(idx):
            hgt[x] = GY - idx[0]
    hgt[og - 6:] = np.where(hgt[og - 6:] > 30, 0, hgt[og - 6:])       # orchard ground isn't tall
    slope = 2.2
    sh = np.zeros((H - SY, TW), bool)
    xs = np.arange(TW)
    for r in range(H - SY):
        src = np.clip(xs + int(round((r + 2) * slope)), 0, TW - 1)
        sh[r] = hgt[src] > (r + 2) * 0.9
    sub = a[SY:H]
    tint_rows = sh
    sc = np.array((150, 92, 108), np.float32)
    s2 = sub.astype(np.float32)
    s2[tint_rows] = s2[tint_rows] * 0.5 + sc * 0.5
    # lit shafts get a warm boost
    lit = ~tint_rows
    s2[lit] = np.clip(s2[lit] * 1.06 + np.array((14, 8, -4)), 0, 255)
    a[SY:H] = s2.astype(np.uint8)
    # boardwalk and posts: shadow band at the bottom of facades (street bounce) + top warmth
    fac = m.copy()
    fac[GY:] = False
    yy = np.arange(H)[:, None]
    low = fac & (yy > GY - 14)
    tint(a, low, (150, 90, 120), 0.10)
    hi = fac & (yy < 100)
    tint(a, hi, (255, 214, 150), 0.10)
    return to_layer(cv), tuple(wins), ex


# ------------------------------------------------------------------ apple trees (per frame, sway)
@lru_cache(maxsize=None)
def apple_tree(size, seed):
    """Young apple tree sprite: slim trunk, round curly canopy (lit right), red apples."""
    R_ = (3, 5, 7)[size]
    th = (4, 6, 9)[size]
    w = 2 * R_ + 5
    h = 2 * R_ + th + 2
    cv = new(w, h)
    cx, cy = w // 2, R_ + 1
    cv.rect(cx, cy + R_ - 2, 1 + (size > 0), th + 2, (110, 72, 70))
    if size > 0:
        cv.px(cx + 1, cy + R_, (176, 120, 82))
    yy, xx = np.mgrid[0:h, 0:w]
    n = vnoise(h, w, 2.5, 2.5, seed)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) + (n - 0.5) * 2.2
    m = d <= R_
    lit = ((xx - cx) * 0.8 - (yy - cy) * 0.4) / max(1, R_)
    a = cv.a
    ramp = [(40, 100, 70), (60, 132, 76), (110, 172, 80), (176, 196, 96), (232, 214, 120)]
    idx = np.clip(np.floor((lit + 1) * 2.1 + (bayer(h, w) - 0.5) * 0.8), 0, 4).astype(int)
    for i in range(5):
        a[m & (idx == i)] = ramp[i]
    ol = outline_of(m)
    a[ol & (xx < cx)] = (40, 84, 70)
    # curls: little darker arcs
    for k in range(R_):
        px_ = cx + int((hash01(seed, k) - 0.5) * 1.6 * R_)
        py_ = cy + int((hash01(seed, k, 2) - 0.5) * 1.6 * R_)
        if 0 <= py_ < h and 0 <= px_ < w and m[py_, px_]:
            a[py_, px_] = (40, 100, 70)
    # apples
    for k in range(2 + size * 2):
        px_ = cx + int((hash01(seed, k, 4) - 0.5) * 1.5 * R_)
        py_ = cy + int((hash01(seed, k, 5) - 0.3) * 1.3 * R_)
        if 0 <= py_ < h and 0 <= px_ < w and m[py_, px_]:
            a[py_, px_] = (232, 59, 59)
            if size == 2 and px_ + 1 < w and m[py_, px_ + 1]:
                a[py_, px_ + 1] = (174, 35, 52)
            if size == 2 and py_ - 1 >= 0:
                a[py_ - 1, px_] = (255, 160, 150) if m[py_ - 1, px_] else a[py_ - 1, px_]
    return to_layer(cv)


@lru_cache(maxsize=1)
def _orchard_trees():
    """(x, base_y, size, seed) in town-layer coords, back rows first."""
    out = []
    for row, (by, size, sp, x0) in enumerate(((124, 0, 9, 452), (131, 1, 13, 458), (143, 2, 17, 462))):
        for k, x in enumerate(range(x0, TW + 10, sp)):
            if row == 2 and abs(x - 556) < 8:
                continue
            out.append((x + (row % 2) * 3, by, size, 7 * row + k))
    return tuple(out)


def draw_orchard(c, f, ox):
    for (x, by, size, sd) in _orchard_trees():
        sx = x - ox
        if sx < -20 or sx > W + 20:
            continue
        lay = apple_tree(size, sd % 6)
        h, w = lay[0].shape[:2]
        # long shadow to the left on the ground
        c.hline(sx - w // 2 - 3 - 3 * size, sx - 1, by, (176, 108, 78))
        if size == 2:
            c.hline(sx - w // 2 - 8, sx - 3, by + 1, (186, 116, 80))
        sway = 1 if math.sin(2 * math.pi * (f / 54.0 + hash01(sd, 1))) > 0.6 else 0
        comp(c, lay, sx - w // 2 + sway, by - h + 1)


# ------------------------------------------------------------------ foreground (1.35)
@lru_cache(maxsize=None)
def cactus(hgt, arms, seed):
    """Saguaro-ish cactus with rounded arms, ribs, rim light on the right, pink flowers on top."""
    w = 21
    cv = new(w, hgt + 2)
    cx = w // 2
    base = (60, 112, 84)
    dk, lt, rim = (40, 80, 72), (92, 150, 96), (226, 200, 120)

    def stem(x, y0, y1, r):
        for xx in range(x - r, x + r + 1):
            cv.vline(xx, y0, y1, base)
        cv.circle(x, y0, r, base)
        cv.vline(x - r, y0, y1, dk)
        cv.vline(x + r, y0, y1, rim)
        cv.vline(x + r - 1, y0, y1, lt)
        if r >= 2:
            cv.vline(x - 1, y0, y1, (50, 96, 80))
        cv.px(x + r - 1, y0 - r + 1, rim)
    r = 2 if hgt < 40 else 3
    stem(cx, 3, hgt + 1, r)
    for k, (side, ay, ah) in enumerate(arms):
        ax = cx + side * (r + 3)
        y_join = ay
        for xx in range(min(cx, ax), max(cx, ax) + 1):
            cv.vline(xx, y_join - 1, y_join + 1, base)
        cv.hline(min(cx, ax), max(cx, ax), y_join + 1, dk)
        stem(ax, y_join - ah, y_join + 1, 1 if r == 2 else 2)
    cv.px(cx, 1, (255, 121, 160))
    cv.px(cx - 1, 2, (240, 79, 120))
    cv.px(cx + 1, 2, (255, 180, 200))
    return to_layer(cv)


FG_W = 680


@lru_cache(maxsize=1)
def _fg_a():
    cv = new(FG_W, H)
    a = cv.a
    # front ground strip (rocks, dry grass)
    top = (171 + (noise1(FG_W, 11, 71) - 0.5) * 5).astype(int)
    for x in range(FG_W):
        cv.vline(x, top[x], H - 1, (184, 116, 76))
        cv.px(x, top[x], (232, 170, 108))
        if top[x] + 1 < H:
            cv.px(x, top[x] + 1, (210, 140, 88))
    R = rng(72)
    for i in range(60):
        x = int(R.random() * FG_W)
        y = int(top[min(FG_W - 1, x)] + 2 + R.random() * 6)
        s = 1 + int(R.random() * 3)
        cv.ellipse(x, y, s + 1, s, (150, 96, 92))
        cv.px(x + s, y - s + 1, (240, 190, 130))
        cv.hline(x - s - 6, x - s - 1, y + s - 1, (140, 84, 84))
    for i in range(90):
        x = int(R.random() * FG_W)
        y = top[min(FG_W - 1, x)]
        for k in range(3):
            cv.line(x + k - 1, y, x + k - 2 + int(R.random() * 3), y - 2 - int(R.random() * 4),
                    (214, 170, 96) if k else (160, 120, 70))
    # fence segment, barrels, wagon wheel, lantern post
    for (x0_, n) in ((120, 5), (420, 4)):
        for k in range(n):
            px_ = x0_ + k * 14
            cv.rect(px_, 150, 3, 24, (120, 76, 66))
            cv.vline(px_ + 2, 150, 173, (236, 170, 110))
            cv.px(px_ + 1, 150, (176, 120, 82))
        cv.rect(x0_, 155, (n - 1) * 14 + 3, 2, (150, 96, 70))
        cv.hline(x0_, x0_ + (n - 1) * 14 + 2, 155, (236, 186, 120))
        cv.rect(x0_, 163, (n - 1) * 14 + 3, 2, (150, 96, 70))
        cv.hline(x0_, x0_ + (n - 1) * 14 + 2, 163, (236, 186, 120))
    for bx in (232, 244, 600):
        cv.rect(bx, 160, 10, 14, (140, 86, 66))
        cv.ellipse(bx + 4, 160, 5, 2, (176, 120, 82))
        cv.ellipse(bx + 4, 160, 3, 1, (110, 70, 66))
        for yy in (163, 170):
            cv.hline(bx, bx + 9, yy, (74, 82, 94))
        cv.vline(bx + 9, 160, 173, (236, 170, 110))
        cv.vline(bx + 8, 161, 173, (190, 130, 88))
    wx, wy = 520, 164
    cv.circle(wx, wy, 9, (110, 70, 66), fill=False)
    cv.circle(wx, wy, 8, (176, 120, 82), fill=False)
    for k in range(6):
        ang = k * math.pi / 6
        cv.line(wx - int(round(math.cos(ang) * 8)), wy - int(round(math.sin(ang) * 8)),
                wx + int(round(math.cos(ang) * 8)), wy + int(round(math.sin(ang) * 8)), (150, 96, 70))
    cv.circle(wx, wy, 2, (110, 70, 66))
    for yy in range(wy - 8, wy + 4):
        cv.px(wx + 9, yy, (255, 220, 150))
    # parked apple cart, heaped with apples
    ax, ay = 548, 158
    cv.hline(ax - 30, ax + 22, 175, (140, 84, 84))
    cv.line(ax - 1, ay + 6, ax - 16, ay + 14, (110, 70, 66))
    cv.line(ax - 1, ay + 7, ax - 16, ay + 15, (176, 120, 82))
    for yy in range(ay - 3, ay + 1):
        for xx in range(ax + 1, ax + 25):
            u = (xx - ax - 1) / 24
            if yy >= ay - int(round(4 * 4 * u * (1 - u))):
                cv.px(xx, yy, (232, 59, 59) if (xx + yy * 3) % 4 else (174, 35, 52))
    for k in range(7):
        cv.px(ax + 4 + k * 3, ay - 2 - (k % 2) - (2 if 2 <= k <= 4 else 0), (255, 170, 160))
    cv.rect(ax, ay, 26, 8, (176, 120, 82))
    cv.hline(ax, ax + 25, ay, (236, 186, 120))
    cv.hline(ax, ax + 25, ay + 4, (150, 96, 70))
    cv.vline(ax + 25, ay, ay + 7, (255, 220, 150))
    cv.vline(ax, ay, ay + 7, (110, 70, 66))
    sign_board(cv, ax + 9, ay + 1, 8, 6, 'cream', 'tan', None)
    cv.px(ax + 13, ay + 3, (232, 59, 59))
    cv.px(ax + 12, ay + 4, (232, 59, 59))
    cv.px(ax + 13, ay + 4, (174, 35, 52))
    cv.px(ax + 14, ay + 2, (145, 219, 105))
    for wx_ in (ax + 6, ax + 20):
        cv.circle(wx_, ay + 11, 4, (110, 70, 66), fill=False)
        cv.circle(wx_, ay + 11, 3, (176, 120, 82), fill=False)
        cv.px(wx_, ay + 11, (110, 70, 66))
        cv.vline(wx_ + 4, ay + 9, ay + 12, (255, 220, 150))
    # cacti
    for (x, hgt, arms, sd) in ((18, 56, ((-1, 26, 12), (1, 18, 16)), 1), (330, 38, ((1, 22, 10),), 2),
                               (372, 26, (), 3), (470, 60, ((-1, 22, 14), (1, 30, 12)), 4),
                               (660, 44, ((-1, 20, 10),), 5)):
        lay = cactus(hgt, arms, sd)
        lh, lw = lay[0].shape[:2]
        y0 = top[min(FG_W - 1, x)] + 3 - lh
        # long shadow to the left
        for k in range(3):
            cv.hline(x - 34, x - 2, top[min(FG_W - 1, x)] + 1 + k, (140, 84, 90))
        src = Canvas(arr=lay[0].copy())
        aa = cv.a[y0:y0 + lh, x - lw // 2:x - lw // 2 + lw]
        np.copyto(aa, lay[0][:aa.shape[0], :aa.shape[1]], where=lay[1][:aa.shape[0], :aa.shape[1], None])
    return to_layer(cv)


# ------------------------------------------------------------------ animated bits
@lru_cache(maxsize=None)
def tumble_frames(R_, seed):
    """8 rotation frames of a tumbleweed ball (tangled twigs), lit on the right."""
    Rg = rng(seed)
    twigs = []
    for i in range(10 + R_ * 3):
        r1, a1 = R_ * (0.3 + 0.7 * Rg.random()), Rg.random() * 2 * math.pi
        r2, a2 = R_ * (0.4 + 0.6 * Rg.random()), a1 + (Rg.random() - 0.5) * 2.6
        twigs.append((r1, a1, r2, a2))
    out = []
    s = 2 * R_ + 3
    for k in range(8):
        rot = k * math.pi / 4 / 2 * 2
        cv = new(s, s)
        c0 = R_ + 1
        cv.circle(c0, c0, R_, (150, 104, 74), fill=False)
        for (r1, a1, r2, a2) in twigs:
            x1 = c0 + r1 * math.cos(a1 + rot)
            y1 = c0 + r1 * math.sin(a1 + rot)
            x2 = c0 + r2 * math.cos(a2 + rot)
            y2 = c0 + r2 * math.sin(a2 + rot)
            cc = (214, 160, 100) if (x1 + x2) / 2 > c0 else (160, 108, 76)
            cv.line(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), cc)
        # rim
        for ang in np.linspace(-1.2, 1.0, 10):
            cv.px(int(round(c0 + R_ * math.cos(ang + rot * 0))), int(round(c0 + R_ * math.sin(ang))), (250, 210, 140))
        out.append(to_layer(cv))
    return out


# (start frame, start screen-world x, speed, ground y, radius, hop, parallax, seed)
TUMBLE_A = [(80, 150, 1.9, 163, 5, 7, 1.0, 1), (120, 250, 2.7, 178, 7, 10, 1.35, 2),
            (126, 360, 1.5, 153, 3, 4, 1.0, 3), (150, 330, 2.2, 160, 4, 6, 1.0, 4)]


def draw_tumbleweeds(c, f, cam, specs):
    for (t0, x0, sp, gy, R_, hop, par, sd) in specs:
        if f < t0:
            continue
        xw = x0 + sp * (f - t0)
        sx = int(round(xw - cam * par))
        if sx < -20 or sx > W + 20:
            continue
        p = (f % B) / B
        lift = hop * 4 * p * (1 - p)
        sy = int(round(gy - lift))
        fr = tumble_frames(R_, sd)[int(xw / R_ / (math.pi / 4)) % 8]
        # shadow on the ground (shrinks while airborne)
        sw_ = max(1, int(R_ * (1 - 0.5 * lift / max(1, hop))))
        c.hline(sx - sw_ - 2, sx + sw_ - 2, gy, (150, 92, 100))
        comp(c, fr, sx - R_ - 1, sy - 2 * R_ - 2)
        # dust puff right after each landing (on the beat)
        if p < 0.4:
            k = p / 0.4
            n = 5
            for i in range(n):
                dx = int(round((i - n // 2) * (2 + 6 * k)))
                dy = int(round(-1 - 3 * k * hash01(sd, i)))
                if hash01(sd, i, f // 3) < 1 - k:
                    c.px(sx + dx - 2, gy + dy, (246, 210, 160) if i % 2 else (232, 182, 130))


def dust_devil(c, f, x, base, t0, t1, cam, par=1.0):
    if not (t0 <= f <= t1):
        return
    k = seg(f, t0, t1)
    env = math.sin(math.pi * k) ** 0.6
    sx = x + (f - t0) * 0.9 - cam * par
    c.dither(int(sx) - 8, base - 3, 17, 4, (236, 190, 140), 0.45 * env)
    for i in range(22):
        hh = i * 1.5 * env
        r = (1.5 + i * 0.32) * env
        for j in range(3):
            ang = f * 0.5 + i * 0.55 + j * 2 * math.pi / 3
            px_ = int(round(sx + r * math.cos(ang) + i * 0.3))
            py_ = int(round(base - hh))
            if (i + j + f // 2) % 4:
                c.px(px_, py_, (255, 228, 180) if math.cos(ang) > 0 else (222, 170, 130))


def smoke(c, f, x, y, seed, n=5, period=40, rise=22, drift=10):
    for i in range(n):
        t = ((f + i * period / n + seed * 7) % period) / period
        px_ = int(round(x + drift * t + math.sin(t * 6 + i) * 1.2))
        py_ = int(round(y - rise * t))
        r = 1 + int(t * 2.5)
        cc = mix((214, 170, 190), (236, 150, 150), t)
        lvl = 1 - t
        if lvl < 0.2:
            continue
        c.circle(px_, py_, r, cc) if lvl > 0.55 else c.dither(px_ - r, py_ - r, 2 * r + 1, 2 * r + 1, cc, lvl)


def pennant(c, f, x, y, color, ln=7):
    ph = (f // 4) % 3
    for i in range(ln):
        dy = int(round(math.sin(i * 0.9 - f * 0.35) * (i / ln) * 1.4))
        hgt = max(1, int(round(3 * (1 - i / ln))))
        c.vline(x + 1 + i, y + dy, y + dy + hgt - 1, color)
    c.px(x + 1, y, mix(color, 'white', .4))
    c.vline(x, y - 1, y + 6, (110, 72, 70))


def lamp(c, x, y, f, on, seed=0):
    c.rect(x - 1, y - 2, 3, 4, (74, 58, 70))
    if on:
        fl = 0.8 + 0.2 * math.sin(f * 0.9 + seed * 2.1)
        c.blend(x - 4, y - 5, 9, 9, (255, 200, 110), 0.18 * fl)
        c.blend(x - 2, y - 3, 5, 6, (255, 220, 140), 0.25 * fl)
        c.px(x, y - 1, (255, 248, 200))
        c.px(x, y, (255, 214, 120))
    else:
        c.px(x, y - 1, (150, 110, 110))
        c.px(x, y, (120, 84, 100))


def light_window(c, f, x, y, w, h, t_on, round_top):
    if f < t_on:
        return
    k = seg(f, t_on, t_on + 4)
    if k < 0.5:
        cc = (236, 170, 110)
    else:
        flick = 1 if hash01(x, y, f // 6) > 0.06 else 0
        cc = (255, 214, 120) if flick else (246, 190, 100)
    c.blend(x - 2, y - 2, w + 4, h + 4, (255, 196, 110), 0.12 * k)
    c.rect(x, y, w, h, cc)
    c.rect(x, y + h - 2, w, 2, (246, 160, 80))
    c.px(x + 1, y + 1, (255, 248, 210))
    if round_top:
        c.px(x, y, mix(cc, (110, 70, 80), .5))
        c.px(x + w - 1, y, mix(cc, (110, 70, 80), .5))
    c.vline(x + w // 2, y, y + h - 1, (200, 130, 110))
    c.hline(x, x + w - 1, y + h // 2, (200, 130, 110))


def batwing_doors(c, f, x, y, lit):
    """Swinging saloon doors in a 12-wide doorway (x, y = doorway top-left)."""
    if lit:
        c.rect(x, y, 12, 18, (176, 104, 70))
        c.rect(x + 2, y + 2, 8, 12, (236, 160, 90))
        c.rect(x + 4, y + 3, 4, 8, (255, 214, 130))
    ang = 0.55 * math.sin(2 * math.pi * f / 46) + 0.25 * math.sin(2 * math.pi * f / 19 + 1)
    for side in (0, 1):
        a_ = ang if side == 0 else -ang * 0.9
        wl = max(2, int(round(6 * math.cos(a_))))
        if side == 0:
            x0 = x
        else:
            x0 = x + 12 - wl
        tone = (236, 150, 120) if a_ > 0 else (196, 112, 100)
        c.rect(x0, y + 5, wl, 9, tone)
        c.hline(x0, x0 + wl - 1, y + 5, (255, 214, 170))
        for yy in range(y + 7, y + 13, 2):
            c.hline(x0 + 1, x0 + wl - 2, yy, (160, 88, 86))
        c.vline(x0 if side == 1 else x0 + wl - 1, y + 5, y + 13, (110, 60, 74))
        c.hline(x0, x0 + wl - 1, y + 14, (110, 60, 74))


def clock_hands(c, cx, cy, f):
    mins = 18 * 60 + 30 + 45 * (f + FLIGHT_SHIFT) / (N + FLIGHT_SHIFT - 1)
    am = (mins % 60) / 60 * 2 * math.pi
    ah = ((mins / 60) % 12) / 12 * 2 * math.pi
    c.line(cx, cy, cx + int(round(math.sin(am) * 4)), cy - int(round(math.cos(am) * 4)), (70, 40, 60))
    c.line(cx, cy, cx + int(round(math.sin(ah) * 2.6)), cy - int(round(math.cos(ah) * 2.6)), (110, 60, 74))
    c.px(cx, cy, (201, 162, 74))


def bunting(c, f, x0, x1, y, sag, cols_, seed):
    n = x1 - x0
    for xx in range(x0, x1 + 1):
        u = (xx - x0) / n
        yy = y + int(round(sag * 4 * u * (1 - u) + math.sin(f * 0.12 + seed) * u * (1 - u) * 2))
        c.px(xx, yy, (110, 72, 70))
        k = xx - x0
        if k % 5 == 2:
            cc = cols_[(k // 5 + seed) % len(cols_)]
            flut = 1 if math.sin(f * 0.4 + k * 0.7 + seed) > 0.3 else 0
            c.hline(xx - 1, xx + 1, yy + 1, cc)
            c.px(xx + flut, yy + 2, cc)
            c.px(xx + 1, yy + 1, mix(cc, 'white', .35))


def windpump_rotor(c, x, y, f):
    ang0 = f * 0.16
    for k in range(10):
        a_ = ang0 + k * 2 * math.pi / 10
        ca, sa = math.cos(a_), math.sin(a_)
        x1, y1 = x + int(round(ca * 3)), y + int(round(sa * 3))
        x2, y2 = x + int(round(ca * 9)), y + int(round(sa * 9))
        cc = (236, 214, 200) if ca > 0 else (150, 130, 150)
        c.line(x1, y1, x2, y2, cc)
    c.circle(x, y, 2, (110, 96, 110))
    c.px(x + 1, y - 1, (255, 236, 180))
    # tail vane (points left: wind from the right)
    c.line(x - 2, y, x - 11, y, (96, 70, 90))
    c.line(x - 2, y + 1, x - 11, y + 1, (150, 110, 120))
    for i in range(6):
        c.vline(x - 17 + i, y - 2 - i // 2, y + 2 + i // 3, (214, 96, 120))
    c.vline(x - 12, y - 4, y + 3, (255, 170, 160))
    c.vline(x - 17, y - 2, y + 2, (150, 60, 96))


def glints(c, f, x0, x1, y, ox, seed):
    for i in range(10):
        gx = x0 + int(hash01(seed, i) * (x1 - x0)) - ox
        ph = (f + int(hash01(seed, i, 2) * 40)) % 40
        if ph < 6:
            c.px(gx + ph // 2, y, (255, 240, 200))
            if ph in (2, 3):
                c.px(gx + ph // 2 + 1, y, (255, 220, 160))


def render_a(f):
    cam = cam_a(f)
    c = Canvas(W, H, 'ink')
    # sky + sun + clouds
    comp(c, to_layer(_sky_a()), -int(round(cam * P_SUN)), 0)
    sun = sun_sprite(SUN_A[2], True)
    sx = SUN_A[0] - int(round(cam * P_SUN))
    comp(c, sun, sx - SUN_A[2] - 1, SUN_A[1] - SUN_A[2] - 1)
    draw_clouds(c, f, CLOUDS_A, cam, P_SKY)
    # far -> near landscape
    comp(c, _far_a(), -int(round(cam * P_FAR)), 0)
    comp(c, _hills_a(), -int(round(cam * P_HILL)), 0)
    comp(c, _buttes_a(), -int(round(cam * P_BUTTE)), 0)
    comp(c, _plain_a(), -int(round(cam * P_PLAIN)), 0)
    # balloon drifting above the town (behind the town layer)
    bx = 290 + 0.1 * (f - 72) - cam * 0.4
    by = 47 - 0.03 * (f - 72)
    draw_balloon(c, int(round(bx)), int(round(by)), f, 'm', flame=(f % 72) < 30)
    # town
    town, wins, ex = _town_a()
    comp(c, town, -cam, 0)
    draw_orchard(c, f, cam)
    comp(c, ex['fence'], -cam, 0)
    ox = cam
    # windows + lamps
    for (x, y, w, h, t_on, rt) in wins:
        if -10 < x - ox < W + 10:
            light_window(c, f, x - ox, y, w, h, t_on, rt)
    lamps_on = f >= 144
    for i, (lx, ly) in enumerate(ex['lanterns']):
        lamp(c, lx - ox, ly, f, f >= 90, i)
    for i, (lx, ly) in enumerate(ex['stage_lamps']):
        lamp(c, lx - ox, ly, f, f >= 144 + 18 * i, i + 3)
    for i, (lx, ly) in enumerate(ex['depot_lamps']):
        lamp(c, lx - ox, ly, f, f >= 162 + 18 * i, i + 5)
    dx, dy = ex['saloon_door']
    batwing_doors(c, f, dx - ox, dy, True)
    clock_hands(c, ex['clock'][0] - ox, ex['clock'][1], f)
    fx_, fy_ = ex['finial']
    pennant(c, f, fx_ - ox, fy_ - 5, (240, 79, 120))
    g0, g1, gy = ex['stage']
    bunting(c, f, g0 - ox, g1 - ox, gy, 9, ((240, 79, 120), (150, 222, 190), (249, 194, 43), (160, 206, 244)), 1)
    bunting(c, f, g0 - ox, g1 - ox, gy + 6, 6, ((168, 132, 243), (255, 159, 208), (255, 244, 214)), 3)
    smoke(c, f, 47 + 36 - ox, GY - 72 - 13, 1)
    smoke(c, f, 95 + 6 - ox, GY - 50 - 13, 2)
    wpx, wpy = ex['windpump']
    windpump_rotor(c, wpx - ox, wpy, f)
    d0, d1, dy_ = ex['ditch']
    glints(c, f, d0, d1, dy_, ox, 5)
    # spout drip sparkle
    spx, spy = ex['spout']
    if (f // 9) % 4 == 0:
        c.px(spx - ox, spy + (f % 9) // 2, (196, 230, 255))
    # tumbleweeds + dust
    draw_tumbleweeds(c, f, cam, [t for t in TUMBLE_A if t[6] < 1.2])
    dust_devil(c, f, 470, 160, 150, 214, cam)
    # foreground
    comp(c, _fg_a(), -int(round(cam * P_FG)), 0)
    draw_tumbleweeds(c, f, cam, [t for t in TUMBLE_A if t[6] >= 1.2])
    fx.dust_motes(c, f, (0, 90, W, 70), n=10, color=(255, 226, 170), seed=3)
    # callouts + card
    callout(c, ex['finial'][0] - ox + 6, 40, 'APPLELOOSA', f, 126, 198, side='right', length=6)
    ssx, ssy = ex['saloon_sign']
    callout(c, ssx - ox, ssy - 10, 'SALT BLOCK', f, 90, 162, side='up', length=8)
    callout(c, 478 - ox, 124, 'APPLE ORCHARDS', f, 162, 212, side='up', length=14)
    atlas_card(c, f, 100, 196, 'FRONTIER TOWNS', ['APPLES IN THE DESERT', 'CHERRIES AT THE JUNCTION'])
    return c


# ================================================================== SHOT B: DODGE JUNCTION
RY = 134          # rail top (wheel bottom)
SUN_B = (212, 110, 22)
CAMB = 48
P_SUNB, P_RIM, P_PLAINB, P_RANCH, P_FGB = 0.04, 0.10, 0.40, 0.72, 1.30


def cam_b(f):
    return int(round(CAMB * smooth(seg(f, 218, 286))))


SKY_B_STOPS = [(0, (50, 38, 100)), (26, (84, 54, 136)), (50, (140, 72, 146)), (70, (200, 92, 124)),
               (88, (236, 120, 94)), (102, (246, 150, 78)), (114, (250, 182, 92)), (126, (252, 206, 128))]

CLOUDS_B = [(20, 22, 90, 3, 21, 'hi', 0.10), (150, 30, 70, 3, 22, 'hi', 0.12), (260, 18, 80, 3, 23, 'hi', 0.09),
            (60, 48, 120, 4, 24, 'md', 0.14), (230, 56, 100, 3, 25, 'md', 0.13), (110, 74, 130, 3, 26, 'lo', 0.17),
            (300, 82, 80, 2, 27, 'lo', 0.18), (0, 94, 60, 2, 28, 'lo', 0.16)]


@lru_cache(maxsize=1)
def _sky_b():
    return sky_layer(W + 10, 132, SKY_B_STOPS, (SUN_B[0], SUN_B[1]), 76, 2)


@lru_cache(maxsize=1)
def _rim_b():
    """The Badlands rim to the south: jagged dark-violet mesas, backlit (gold rim near the sun)."""
    w = 360
    cv = new(w, 128)
    specs = [(0, 22, 104, 8, 3, 0.5), (40, 52, 98, 14, 4, 0.6), (70, 120, 108, 6, 3, 0.5), (138, 150, 101, 10, 4, 0.6),
             (176, 196, 113, 4, 2, 0.4), (226, 246, 112, 5, 2, 0.4), (262, 300, 103, 10, 3, 0.5),
             (318, 330, 96, 16, 4, 0.6), (340, 360, 106, 8, 3, 0.5)]
    top = heights_mesa(w, specs, 81, 1.4)
    top = np.minimum(top, 121 + (noise1(w, 6, 82) - 0.5) * 2)
    m = shade_silhouette(cv, top, 128, [(78, 46, 104), (92, 54, 116), (102, 60, 124), (128, 74, 132),
                                        (176, 104, 140)], 83, rim_w=1, band=6,
                         haze=(200, 120, 132), haze_rows=6)
    # gold rim on the tops close to the sun
    a = cv.a
    tops = np.round(top).astype(int)
    for x in range(w):
        d = abs(x - SUN_B[0])
        if d < 90 and 0 <= tops[x] < 128:
            a[tops[x], x] = (255, 214, 130) if d < 50 else (236, 150, 120)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _plain_b():
    w = 360
    cv = new(w, 140)
    cv.gradient(0, 120, w, 20, [(226, 150, 104), (206, 128, 92), (180, 104, 86), (160, 90, 84)], soft=0.8)
    R = rng(91)
    for i in range(140):
        x = int(R.random() * w)
        y = int(122 + R.random() * 16)
        L = 2 + int(R.random() * 5)
        cv.hline(x, x + L, y, mix((236, 168, 110), (140, 80, 84), R.random()))
    for i in range(40):
        x = int(R.random() * w)
        y = int(123 + R.random() * 14)
        s = 1 + (y > 130)
        cv.rect(x - s, y - s, 2 * s + 1, s + 1, (110, 92, 92))
        cv.px(x + s, y - s, (240, 180, 110))
        cv.hline(x - s - 5, x - s - 1, y, (150, 84, 88))
    # a far mesa-top windmill and fence line
    for x in range(10, w, 7):
        cv.vline(x, 124, 126, (130, 80, 90))
    cv.hline(0, w - 1, 125, (150, 96, 100))
    return to_layer(cv)


@lru_cache(maxsize=None)
def cherry_tree(size, seed, kind):
    """Rounded cherry tree; kind 'r' red cherries / 'y' yellow cherries (pairs on stalks)."""
    R_ = (4, 6, 9)[size]
    th = (4, 6, 9)[size]
    w = 2 * R_ + 7
    h = 2 * R_ + th + 2
    cv = new(w, h)
    cx, cy = w // 2, R_ + 1
    tr = (98, 60, 66)
    cv.rect(cx, cy + R_ - 3, 2, th + 3, tr)
    cv.px(cx + 1, cy + R_, (190, 120, 90))
    if size == 2:
        cv.line(cx, cy + 3, cx - 3, cy - 1, tr)
        cv.line(cx + 1, cy + 3, cx + 4, cy, tr)
    yy, xx = np.mgrid[0:h, 0:w]
    n = vnoise(h, w, 2.2, 2.2, seed + 40)
    d = np.sqrt(((xx - cx) / 1.12) ** 2 + (yy - cy) ** 2) + (n - 0.5) * 2.6
    m = d <= R_
    if size == 2:
        m &= ~((yy > cy + 2) & (np.abs(xx - cx) < 2) & (yy > cy + R_ - 3))
    lit = ((xx - cx) * 0.9 - (yy - cy) * 0.5) / max(1, R_)
    a = cv.a
    ramp = [(34, 70, 66), (46, 96, 72), (80, 134, 76), (150, 168, 84), (236, 200, 116)]
    idx = np.clip(np.floor((lit + 1) * 2.0 + (bayer(h, w) - 0.5) * 0.9), 0, 4).astype(int)
    for i in range(5):
        a[m & (idx == i)] = ramp[i]
    ol = outline_of(m)
    a[ol & (xx <= cx)] = (34, 60, 64)
    a[ol & (xx > cx) & (yy < cy)] = (255, 220, 140)
    cc = ((174, 35, 52), (232, 59, 59), (255, 150, 150)) if kind == 'r' else ((214, 150, 30), (249, 210, 60), (255, 246, 170))
    for k in range(3 + size * 4):
        px_ = cx + int((hash01(seed, k, 4) - 0.5) * 1.6 * R_)
        py_ = cy + int((hash01(seed, k, 5) - 0.35) * 1.4 * R_)
        if 0 <= py_ < h - 1 and 0 <= px_ < w - 1 and m[py_, px_]:
            a[py_, px_] = cc[1]
            if size >= 1 and m[py_ + 1, px_ + 1]:
                a[py_ + 1, px_ + 1] = cc[0]
            if size == 2 and py_ > 0:
                a[py_ - 1, px_] = cc[2] if m[py_ - 1, px_] else a[py_ - 1, px_]
    return to_layer(cv)


@lru_cache(maxsize=1)
def _ranch_b():
    """Cherry Hill (behind the track, right): the hill, cherry rows, sorting shed + conveyor frame."""
    w = 380
    cv = new(w, 136)
    xs = np.arange(w, dtype=float)
    hill = 128 - 26 * np.exp(-((xs - 330) / 70) ** 2) - 6 * np.exp(-((xs - 250) / 30) ** 2)
    hill += (noise1(w, 8, 101) - 0.5) * 1.2
    top = np.round(hill).astype(int)
    a = cv.a
    for x in range(w):
        if top[x] < 136:
            cv.gradient(x, top[x], 1, 136 - top[x], [(208, 136, 94), (190, 116, 86), (170, 98, 84)], soft=0.8)
            a[top[x], x] = (246, 196, 128)
    ex = {}
    # orchard rows following the hill (back to front), alternating red / yellow
    trees = []
    for row in range(4):
        sz = 0 if row < 2 else 1
        for k, x in enumerate(range(236 + row * 3, w - 4, 9 + 2 * sz)):
            by = int(hill[min(w - 1, x)]) + 2 + row * 4
            if by > 134:
                continue
            trees.append((x, by, sz, row * 31 + k, 'r' if row % 2 == 0 else 'y'))
    # sorting shed (cream walls, red roof) on the hillside
    sx0, sw, sb = 300, 46, int(hill[322]) + 4
    sr = rampc((240, 210, 180))
    clapboard(cv, sx0, sb - 22, sw, 22, sr, 102)
    cv.vline(sx0, sb - 22, sb - 1, sr['ol'])
    cv.vline(sx0 + sw - 1, sb - 22, sb - 1, sr['rim'])
    rf = rampc((196, 60, 64))
    for yy in range(sb - 36, sb - 21):
        k = yy - (sb - 36)
        x0_ = sx0 + sw // 2 - 4 - int(k * 1.9)
        x1_ = sx0 + sw // 2 + 4 + int(k * 1.9)
        cv.hline(max(sx0 - 3, x0_), min(sx0 + sw + 2, x1_), yy, rf['bs'])
        if yy % 3 == 0:
            for xx in range(max(sx0 - 3, x0_), min(sx0 + sw + 2, x1_), 4):
                cv.px(xx + (yy // 3) % 2 * 2, yy, rf['md'])
        cv.px(min(sx0 + sw + 2, x1_), yy, (255, 200, 140))
    cv.hline(sx0 - 3, sx0 + sw + 2, sb - 21, (120, 40, 56))
    # loft door (open, lit) + big doors with X braces
    lx, ly = sx0 + 8, sb - 19
    cv.rect(lx, ly, 8, 7, (110, 60, 60))
    ex['loft'] = (lx + 4, ly + 6)
    for dx in (sx0 + 22,):
        cv.rect(dx, sb - 15, 16, 15, (196, 60, 64))
        cv.frame(dx, sb - 15, 16, 15, (240, 220, 200))
        cv.line(dx, sb - 15, dx + 15, sb - 1, (240, 220, 200))
        cv.line(dx + 15, sb - 15, dx, sb - 1, (240, 220, 200))
        cv.vline(dx + 8, sb - 15, sb - 1, (240, 220, 200))
    cv.rect(sx0 + 18, sb - 32, 9, 7, (255, 236, 214))
    ICON['cherry'].draw(cv, sx0 + 23, sb - 28, anchor='center')
    ex['shed'] = (sx0 + sw // 2, sb - 36)
    ex['shed_win'] = (sx0 + 40, sb - 12)
    # conveyor: inclined belt frame from the loading bin to the loft door
    bx0, by0 = sx0 - 44, int(hill[sx0 - 44]) + 1
    ex['belt'] = (bx0 + 4, by0 - 8, lx + 2, ly + 5)
    legs = []
    x1_, y1_ = lx + 2, ly + 5
    x0_, y0_ = bx0 + 4, by0 - 8
    L = int(math.hypot(x1_ - x0_, y1_ - y0_))
    for i in range(0, L + 1, 9):
        u = i / L
        x = int(round(x0_ + (x1_ - x0_) * u))
        y = int(round(y0_ + (y1_ - y0_) * u))
        legs.append((x, y + 3, int(hill[min(w - 1, x)]) + 3))
    ex['legs'] = tuple(legs)
    cv.rect(bx0 - 4, by0 - 10, 14, 10, (150, 96, 70))                 # loading bin, heaped with cherries
    cv.hline(bx0 - 4, bx0 + 9, by0 - 10, (236, 186, 120))
    for i in range(12):
        cv.px(bx0 - 3 + i, by0 - 11 - (1 if 3 < i < 9 else 0), (232, 59, 59) if i % 3 else (249, 210, 60))
    cv.vline(bx0 + 9, by0 - 10, by0 - 1, (255, 220, 150))
    # baskets along the hill foot
    for bxx in (230, 288, 362):
        byy = int(hill[bxx]) + 6
        cv.rect(bxx, byy - 3, 5, 3, (196, 140, 90))
        cv.hline(bxx, bxx + 4, byy - 4, (232, 59, 59) if bxx % 2 else (249, 210, 60))
    # ranch gate arch at the hill foot with the cherry sign
    gx = 238
    gb = int(hill[gx]) + 6
    cv.rect(gx, gb - 18, 2, 18, (110, 72, 70))
    cv.rect(gx + 18, gb - 18, 2, 18, (110, 72, 70))
    cv.vline(gx + 19, gb - 18, gb - 1, (255, 210, 140))
    cv.hline(gx - 1, gx + 20, gb - 19, (176, 120, 82))
    cv.hline(gx - 1, gx + 20, gb - 20, (236, 186, 120))
    sign_board(cv, gx + 5, gb - 18, 10, 7, 'cream', 'tan', ICON['cherry'])
    ex['gate'] = (gx + 10, gb - 20)
    return to_layer(cv), tuple(trees), ex


@lru_cache(maxsize=1)
def _station_b():
    """Parallax 1.0: platform, station house, water tower, track, near ground."""
    w = 400
    cv = new(w, H)
    ex = {}
    wins = []
    # near ground in front of the track
    cv.gradient(0, RY - 6, w, H - RY + 6, [(190, 118, 86), (176, 104, 82), (160, 92, 80)], soft=0.8)
    R = rng(111)
    for i in range(160):
        x = int(R.random() * w)
        y = RY + 4 + int(R.random() * (H - RY - 4))
        cv.px(x, y, (220, 150, 100) if R.random() > 0.5 else (140, 80, 80))
    # platform deck (behind the track)
    pr = rampc((176, 120, 82))
    cv.rect(2, RY - 9, 168, 4, pr['bs'])
    cv.hline(2, 169, RY - 9, pr['lt'])
    for xx in range(2, 170, 5):
        cv.px(xx, RY - 8, pr['dk'])
    cv.rect(2, RY - 5, 168, 3, (110, 70, 66))
    # station house (peach false front, canopy on posts)
    hx, hw, hh = 30, 62, 36
    base = RY - 9
    hr = rampc((240, 176, 140))
    clapboard(cv, hx, base - hh, hw, hh, hr, 112)
    crown(cv, hx, hw, base - hh, hr, 'cream', 'step')
    cv.vline(hx, base - hh + 3, base - 1, hr['ol'])
    cv.vline(hx + hw - 1, base - hh - 1, base - 1, hr['rim'])
    sign_board(cv, hx + 14, base - hh + 5, 34, 9, (110, 60, 74), 'gold', None)
    ICON['train'].draw(cv, hx + 24, base - hh + 9, anchor='center')
    ICON['cherry'].draw(cv, hx + 38, base - hh + 9, anchor='center')
    ex['station_sign'] = (hx + hw // 2, base - hh - 8)
    for i, wx in enumerate((hx + 6, hx + 47)):
        window(cv, wx, base - 21, 7, 10, 'cream', 120 + i, wins, 0)
    door(cv, hx + 27, base - 17, 9, 17, 'latte', 'cream')
    awning(cv, hx - 6, hw + 12, base - 25, ('cream', (214, 96, 120)), [], 'tan', depth=3)
    for px_ in (hx - 5, hx + 20, hx + 41, hx + hw + 4):
        cv.rect(px_, base - 21, 2, 21, (150, 96, 70))
        cv.vline(px_ + 1, base - 21, base - 1, (255, 210, 140))
    # bench, mail sacks, crates of cherries on the platform
    cv.rect(hx + 66, base - 5, 12, 2, (150, 96, 70))
    cv.vline(hx + 67, base - 3, base - 1, (110, 70, 66))
    cv.vline(hx + 76, base - 3, base - 1, (110, 70, 66))
    for k, (cx_, kind) in enumerate(((4, 'r'), (11, 'y'), (7, 'r'))):
        by = base - 6 - (6 if k == 2 else 0)
        cv.rect(cx_, by, 7, 6, (176, 120, 82))
        cv.hline(cx_, cx_ + 6, by, (236, 186, 120))
        cv.vline(cx_ + 6, by, by + 5, (255, 210, 140))
        for j in range(3):
            cv.px(cx_ + 1 + 2 * j, by - 1, (232, 59, 59) if kind == 'r' else (249, 210, 60))
    # station bell on a post + platform lamp
    cv.vline(hx + 98, base - 20, base - 1, (110, 72, 70))
    cv.ellipse(hx + 98, base - 21, 2, 2, (201, 162, 74))
    cv.px(hx + 99, base - 22, (255, 240, 160))
    ex['lamps'] = [(hx - 5, base - 26), (hx + hw + 5, base - 26), (150, base - 24)]
    cv.vline(150, base - 22, base - 1, (74, 58, 70))
    # water tower with spout arm
    wt0, wtw = 108, 26
    leg = rampc((128, 84, 66))
    for lx in (wt0 + 2, wt0 + 10, wt0 + 17, wt0 + 23):
        cv.rect(lx, 94, 2, base - 94, leg['bs'])
        cv.vline(lx + 1, 94, base - 1, leg['rim'] if lx > wt0 + 12 else leg['lt'])
    for yy in (104, 114):
        cv.hline(wt0 + 2, wt0 + 24, yy, leg['md'])
        cv.line(wt0 + 3, yy, wt0 + 11, yy + 10, leg['dk'])
    tk = rampc((140, 90, 70))
    vplanks(cv, wt0, 72, wtw, 22, tk, 113)
    for yy in (75, 83, 91):
        cv.hline(wt0, wt0 + wtw - 1, yy, (74, 82, 94))
        cv.hline(wt0, wt0 + wtw - 1, yy - 1, (125, 136, 148))
    cv.vline(wt0 + wtw - 1, 72, 93, tk['rim'])
    cv.vline(wt0, 72, 93, tk['ol'])
    cv.rect(wt0 - 1, 93, wtw + 2, 2, tk['dk'])
    for yy in range(62, 72):
        hw_ = int((yy - 62) * (wtw / 2 + 2) / 10)
        cv.hline(wt0 + wtw // 2 - hw_, wt0 + wtw // 2 + hw_, yy, (122, 70, 74))
        cv.px(wt0 + wtw // 2 + hw_, yy, (255, 190, 130))
    cv.vline(wt0 + wtw // 2, 58, 62, (110, 72, 70))
    cv.line(wt0 + wtw - 2, 95, wt0 + wtw + 12, 99, (74, 82, 94))     # swung spout
    cv.line(wt0 + wtw - 2, 96, wt0 + wtw + 12, 100, (125, 136, 148))
    ex['tower'] = (wt0 + wtw // 2, 62)
    ex['spout'] = (wt0 + wtw + 12, 101)
    # telegraph poles behind the track
    for x in range(176, w, 46):
        cv.vline(x, RY - 26, RY - 8, (110, 72, 70))
        cv.hline(x - 3, x + 3, RY - 24, (110, 72, 70))
        cv.px(x + 1, RY - 25, (255, 200, 130))
    for i, x in enumerate(range(176, w - 46, 46)):
        for xx in range(x, x + 46):
            u = (xx - x) / 46
            cv.px(xx, RY - 24 + int(round(12 * u * (1 - u))), (110, 70, 96))
    # track: ballast, ties, rails
    cv.rect(0, RY, w, 4, (150, 110, 104))
    for x in range(0, w):
        if hash01(x, 7) > 0.6:
            cv.px(x, RY + 1 + int(hash01(x, 8) * 3), (190, 160, 150))
    for x in range(0, w, 4):
        cv.rect(x, RY, 3, 2, (110, 70, 66))
    cv.hline(0, w - 1, RY - 1, (200, 196, 204))
    cv.hline(0, w - 1, RY, (110, 100, 120))
    cv.hline(0, w - 1, RY + 3, (120, 80, 84))
    return to_layer(cv), tuple(wins), ex


# ------------------------------------------------------------------ the Friendship Express
TRAIN_LEN = 162


@lru_cache(maxsize=None)
def train_sprite(phase):
    """Whole train facing right: loco (front at the right end), tender, 3 cars. phase 0..3 = wheel turn."""
    w, h = TRAIN_LEN + 4, 30
    cv = new(w, h)
    by = h - 1                      # wheel bottom row
    ex = {}

    def car(x, ln, color, roof):
        r = rampc(color)
        rr = rampc(roof)
        cv.rect(x, by - 19, ln, 13, r['bs'])
        cv.hline(x, x + ln - 1, by - 19, r['lt'])
        cv.hline(x, x + ln - 1, by - 8, r['md'])
        cv.hline(x, x + ln - 1, by - 7, (110, 70, 90))
        cv.vline(x, by - 19, by - 7, r['ol'])
        cv.vline(x + ln - 1, by - 19, by - 7, r['rim'])
        cv.hline(x - 1, x + ln, by - 20, rr['bs'])
        cv.hline(x + 1, x + ln - 2, by - 21, rr['bs'])
        cv.hline(x + 3, x + ln - 4, by - 22, rr['lt'])
        cv.px(x + ln - 4, by - 22, rr['rim'])
        cv.hline(x, x + ln - 1, by - 11, (249, 194, 43))
        for wx in range(x + 3, x + ln - 4, 6):
            cv.rect(wx, by - 17, 4, 5, (255, 214, 120))
            cv.px(wx, by - 17, r['md'])
            cv.px(wx + 3, by - 17, r['md'])
            cv.hline(wx, wx + 3, by - 13, (246, 160, 80))
        for bx in (x + 5, x + ln - 6):
            cv.rect(bx - 3, by - 6, 7, 2, (74, 58, 70))
        cv.rect(x + ln, by - 10, 2, 2, (74, 58, 70))

    def wheel(cx, cy, r, ph):
        cv.circle(cx, cy, r, (74, 58, 70))
        cv.circle(cx, cy, r - 1, (196, 60, 80) if r > 2 else (110, 70, 90))
        if r > 2:
            cv.circle(cx, cy, r - 2, (110, 50, 70))
            ang = ph * math.pi / 4
            for k in range(2):
                a_ = ang + k * math.pi / 2
                cv.line(cx - int(round(math.cos(a_) * (r - 1))), cy - int(round(math.sin(a_) * (r - 1))),
                        cx + int(round(math.cos(a_) * (r - 1))), cy + int(round(math.sin(a_) * (r - 1))), (249, 194, 43))
        cv.px(cx, cy, (255, 236, 160))

    # cars (rear first)
    x = 0
    for color, roof in (((150, 206, 244), (96, 74, 150)), ((196, 160, 240), (96, 74, 150)), ((255, 159, 208), (110, 60, 110))):
        car(x, 30, color, roof)
        for wx in (x + 4, x + 8, x + 22, x + 26):
            wheel(wx, by - 2, 2, phase)
        x += 33
    # tender
    tx = x
    tr = rampc((214, 70, 110))
    cv.rect(tx, by - 15, 17, 9, tr['bs'])
    cv.hline(tx, tx + 16, by - 15, tr['lt'])
    cv.vline(tx + 16, by - 15, by - 7, tr['rim'])
    cv.vline(tx, by - 15, by - 7, tr['ol'])
    cv.rect(tx + 1, by - 17, 15, 2, (70, 60, 80))
    cv.hline(tx + 2, tx + 14, by - 18, (98, 86, 104))
    cv.hline(tx, tx + 16, by - 10, (249, 194, 43))
    ICON['star'].draw(cv, tx + 8, by - 12, anchor='center')
    for wx in (tx + 4, tx + 12):
        wheel(wx, by - 3, 3, phase)
    x = tx + 18
    # loco: cab (rear), boiler, stack, dome, headlamp, cowcatcher (front = right)
    lx = x
    cb = rampc((196, 160, 240))
    cv.rect(lx, by - 22, 13, 16, cb['bs'])
    cv.vline(lx, by - 22, by - 7, cb['ol'])
    cv.vline(lx + 12, by - 22, by - 7, cb['rim'])
    cv.rect(lx + 3, by - 19, 7, 6, (255, 214, 120))              # cab window (no one inside)
    cv.hline(lx + 3, lx + 9, by - 14, (246, 160, 80))
    cv.hline(lx - 1, lx + 13, by - 23, (110, 60, 110))
    cv.hline(lx, lx + 12, by - 24, (150, 80, 140))
    cv.px(lx + 12, by - 24, (255, 210, 160))
    br = rampc((240, 79, 120))
    bx0, bx1 = lx + 13, lx + 37
    cv.rect(bx0, by - 18, bx1 - bx0, 10, br['bs'])
    cv.hline(bx0, bx1 - 1, by - 18, br['lt'])
    cv.hline(bx0, bx1 - 1, by - 17, br['lt'])
    cv.hline(bx0, bx1 - 1, by - 9, br['md'])
    for bx in (bx0 + 6, bx0 + 14, bx0 + 21):
        cv.vline(bx, by - 18, by - 9, (249, 194, 43))
    cv.rect(bx1, by - 18, 3, 10, (110, 60, 90))                   # smokebox front
    cv.vline(bx1 + 2, by - 18, by - 9, (255, 200, 150))
    cv.ellipse(bx0 + 10, by - 19, 3, 2, (249, 194, 43))           # steam dome
    cv.px(bx0 + 11, by - 21, (255, 240, 170))
    sx = bx1 - 5                                                  # diamond stack
    cv.rect(sx, by - 25, 3, 7, (98, 70, 96))
    cv.rect(sx - 2, by - 28, 7, 3, (110, 60, 90))
    cv.hline(sx - 2, sx + 4, by - 29, (150, 90, 120))
    cv.px(sx + 4, by - 28, (255, 210, 160))
    cv.vline(sx + 2, by - 25, by - 19, (200, 150, 170))
    ex['stack'] = (sx + 1, by - 29)
    ex['whistle'] = (bx0 + 4, by - 20)
    cv.rect(bx0 + 3, by - 20, 2, 2, (249, 194, 43))
    cv.rect(bx1 - 1, by - 23, 4, 3, (249, 194, 43))               # headlamp
    cv.px(bx1 + 2, by - 22, (255, 250, 210))
    ex['lamp'] = (bx1 + 3, by - 22)
    cv.rect(bx0 - 2, by - 8, bx1 - bx0 + 6, 2, (74, 58, 70))      # running board
    for k in range(6):                                            # cowcatcher
        cv.vline(bx1 + 1 + k, by - 7 + k // 2, by - 1, (214, 60, 70) if k % 2 == 0 else (150, 40, 56))
    for i, wx in enumerate((bx0 + 3, bx0 + 11, bx0 + 19)):
        wheel(wx, by - 4, 4, phase)
    wheel(bx0 + 27, by - 2, 2, phase)
    # connecting rod
    ang = phase * math.pi / 4
    ox_, oy_ = int(round(math.cos(ang) * 2)), int(round(math.sin(ang) * 2))
    cv.hline(bx0 + 3 + ox_, bx0 + 19 + ox_, by - 4 + oy_, (201, 205, 214))
    cv.line(bx0 + 19 + ox_, by - 4 + oy_, bx0 + 27, by - 6, (150, 156, 170))
    ex['front'] = bx1 + 7
    return to_layer(cv), ex


@lru_cache(maxsize=None)
def puff_sprite(r, seed):
    """Steam puff ball: bright cream body, warm sunset rim on the right, lilac shade lower-left."""
    s = 2 * r + 3
    cv = new(s, s)
    yy, xx = np.mgrid[0:s, 0:s]
    c0 = r + 1
    n = vnoise(s, s, 2, 2, seed)
    d = np.sqrt((xx - c0) ** 2 + (yy - c0) ** 2) + (n - 0.5) * 1.8
    m = d <= r
    l = ((xx - c0) * 0.7 - (yy - c0) * 0.7) / max(1, r)
    a = cv.a
    a[m] = (238, 214, 226)
    a[m & (l < -0.45)] = (214, 182, 214)
    a[m & (l < -0.85)] = (180, 146, 190)
    a[m & (l > 0.3)] = (255, 240, 226)
    ol = outline_of(m)
    a[ol & (l > 0.1)] = (255, 214, 150)
    return to_layer(cv)


def draw_puff(c, x, y, r, seed, level=1.0):
    r = max(1, int(r))
    lay = puff_sprite(r, seed % 4)
    a, m = lay
    if level >= 0.98:
        comp(c, lay, int(x) - r - 1, int(y) - r - 1)
        return
    q = round(level * 4) / 4
    if q <= 0:
        return
    x0, y0 = int(x) - r - 1, int(y) - r - 1
    hh, ww = m.shape
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(W, x0 + ww), min(H, y0 + hh)
    if X1 <= X0 or Y1 <= Y0:
        return
    sm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    sa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    reg = c.a[Y0:Y1, X0:X1]
    out = reg.astype(np.float32) * (1 - q) + sa * q
    np.copyto(reg, out.astype(np.uint8), where=sm[:, :, None])


def train_x(f):
    """World x of the loco's front end."""
    t = max(0, f - 219)
    return 196 + 0.5 * 0.075 * t * t


def silhouette(c, x0, y0, lay, sun, strength=1.0):
    """Composite a layer, darkening it to a backlit silhouette with a gold rim inside the sun's disc/halo."""
    a, m = lay
    h, w = m.shape
    sx, sy, sr = sun
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx + x0 - sx) ** 2 + (yy + y0 - sy) ** 2)
    k = np.clip((sr + 20 - d) / 20, 0, 1) ** 0.8 * strength
    if k.max() <= 0:
        comp(c, lay, x0, y0)
        return
    out = a.astype(np.float32)
    sil = np.array((84, 42, 82), np.float32)
    kk = (np.minimum(k, 0.78))[:, :, None]
    out = out * (1 - kk) + sil * kk
    out = out.astype(np.uint8)
    up = np.zeros_like(m)
    up[1:] = m[1:] & ~m[:-1]
    up[0] = m[0]
    edge = (outline_of(m) | up) & (k > 0.35)
    out[edge] = (255, 220, 140)
    comp(c, (out, m), x0, y0)


def draw_train(c, f, ox, sun):
    X = train_x(f)
    t = max(0, f - 219)
    dist = X - 196
    phase = int(dist / 3) % 8
    lay, ex = train_sprite(phase)
    h, w = lay[1].shape
    x0 = int(round(X - ex['front'] - ox))
    y0 = RY - h
    # headlamp beam
    lx, ly = x0 + ex['lamp'][0], y0 + ex['lamp'][1]
    for i in range(1, 22):
        hh = i // 4
        c.dither(lx + i, ly - hh, 1, 2 * hh + 1, (255, 236, 170), 0.5 * (1 - i / 22))
    silhouette(c, x0, y0, lay, sun)
    return x0, y0, ex


def train_steam(c, f, ox, sun):
    """Whistle jet on 216 + chug puffs from the stack on every 8th note (big on beats)."""
    _, ex = train_sprite(0)
    hgt = 30
    # chug plume: a puff every 3 frames from the stack (big on beats, medium on 8ths), rising,
    # growing and trailing left of the moving train; idle wisps before the departure
    f0 = (f - 50) - (f - 50) % 3
    for fe in range(max(186, f0), f + 1, 3):
        age = f - fe
        X = train_x(fe)
        moving = fe >= 219
        big = moving and fe % 18 == 0
        med = moving and fe % 9 == 0
        sc = 1.35 if big else (1.05 if med else 0.75)
        if not moving:
            sc = 0.6
        px_ = X - ex['front'] + ex['stack'][0] - ox - age * (0.55 if moving else 0.15)
        py_ = RY - hgt + ex['stack'][1] - 1 - (age * 0.85 - age * age * 0.006) * (1.1 if big else 0.9)
        r = (2.0 + age * 0.21) * sc
        if age > 40:
            continue
        lvl = 1.0 if age < 28 else 1 - (age - 28) / 13
        draw_puff(c, px_, py_, min(r, 9), fe // 3, lvl)
    # whistle: short blast on 216 then a long one an 8th later (225), white jets from the whistle
    X = train_x(216)
    wx = X - ex['front'] + ex['whistle'][0] - ox
    wy = RY - hgt + ex['whistle'][1]
    for (t0, dur) in ((216, 5), (225, 14)):
        age = f - t0
        if age < 0 or age > dur + 18:
            continue
        top = min(34, age * 7) if age <= dur else 34
        lift = max(0, age - dur) * 2.2
        lvl = 1.0 if age <= dur + 8 else 1 - (age - dur - 8) / 10
        for i in range(0, top, 2):
            rr = 1.5 + i * 0.15
            draw_puff(c, wx + i * 0.25 - lift * 0.4, wy - 2 - i - lift, rr, i + t0, lvl)
        if age < dur:
            c.vline(int(wx), int(wy - 5), int(wy - 1), (255, 255, 255))


def semaphore(c, f, x, base, ox):
    """Signal mast in front of the track: arm raises to 'clear' on 216, lamp goes green."""
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
    lc = (107, 255, 120) if k > 0.5 else (255, 80, 60)
    c.px(px_ - 2, py_, lc)
    c.blend(px_ - 5, py_ - 3, 7, 7, lc, 0.25)
    c.circle(sx + 1, base - 51, 1, (74, 58, 70))


@lru_cache(maxsize=None)
def stagecoach():
    cv = new(44, 30)
    body = rampc((178, 60, 70))
    cv.round_rect(8, 6, 26, 15, body['bs'], r=2)
    cv.hline(9, 32, 6, body['lt'])
    cv.vline(33, 7, 19, body['rim'])
    cv.vline(8, 7, 19, body['ol'])
    cv.rect(12, 9, 7, 6, (255, 214, 120))
    cv.rect(23, 9, 7, 6, (255, 214, 120))
    cv.frame(11, 8, 9, 8, (249, 194, 43))
    cv.frame(22, 8, 9, 8, (249, 194, 43))
    cv.hline(8, 33, 18, (249, 194, 43))
    cv.rect(10, 3, 22, 3, (110, 72, 70))                         # luggage rack + trunks
    cv.rect(13, 0, 6, 3, (150, 96, 70))
    cv.rect(21, 1, 7, 2, (120, 140, 180))
    cv.px(27, 1, (255, 236, 200))
    cv.rect(33, 12, 6, 6, (110, 60, 74))                         # boot
    for (cx_, rr) in ((12, 7), (34, 5)):
        cv.circle(cx_, 29 - rr, rr, (74, 58, 70), fill=False)
        cv.circle(cx_, 29 - rr, rr - 1, (249, 194, 43), fill=False)
        for k in range(4):
            ang = k * math.pi / 4
            cv.line(cx_ - int(round(math.cos(ang) * (rr - 1))), 29 - rr - int(round(math.sin(ang) * (rr - 1))),
                    cx_ + int(round(math.cos(ang) * (rr - 1))), 29 - rr + int(round(math.sin(ang) * (rr - 1))), (196, 140, 60))
        cv.px(cx_ + rr, 29 - rr - 1, (255, 236, 160))
    cv.line(0, 26, 6, 20, (110, 72, 70))                          # tongue resting on the ground
    return to_layer(cv)


@lru_cache(maxsize=1)
def _fg_b():
    w = 420
    cv = new(w, H)
    top = (150 + (noise1(w, 13, 121) - 0.5) * 4).astype(int)
    for x in range(w):
        cv.gradient(x, top[x], 1, H - top[x], [(176, 106, 82), (156, 90, 80), (136, 78, 78)], soft=0.8)
        cv.px(x, top[x], (232, 160, 108))
    R = rng(122)
    for i in range(80):
        x = int(R.random() * w)
        y = top[min(w - 1, x)]
        for k in range(3):
            cv.line(x + k - 1, y, x + k - 2 + int(R.random() * 3), y - 2 - int(R.random() * 4),
                    (214, 160, 96) if k else (140, 100, 70))
    for i in range(30):
        x = int(R.random() * w)
        y = int(top[min(w - 1, x)] + 4 + R.random() * 20)
        s = 1 + int(R.random() * 2)
        cv.ellipse(x, y, s + 1, s, (120, 72, 80))
        cv.px(x + s, y - s + 1, (240, 180, 120))
    # stagecoach (parked, no horses), crates, mail sacks, lamp post
    sc = stagecoach()
    cv.hline(6, 40, 175, (110, 66, 80))
    comp_into(cv, sc, 14, 147)
    for k, (cx_, cy_, kind) in enumerate(((66, 166, 'r'), (75, 166, 'y'), (70, 160, 'r'), (84, 167, 'y'))):
        cv.rect(cx_, cy_, 8, 7, (176, 120, 82))
        cv.hline(cx_, cx_ + 7, cy_, (236, 186, 120))
        cv.vline(cx_ + 7, cy_, cy_ + 6, (255, 210, 140))
        cv.hline(cx_, cx_ + 7, cy_ + 3, (150, 96, 70))
        for j in range(3):
            cv.px(cx_ + 1 + 2 * j, cy_ - 1, (232, 59, 59) if kind == 'r' else (249, 210, 60))
            cv.px(cx_ + 2 + 2 * j, cy_ - 1, (174, 35, 52) if kind == 'r' else (214, 150, 30))
    for (mx, my) in ((96, 168), (102, 170)):
        cv.ellipse(mx, my, 3, 4, (196, 160, 112))
        cv.px(mx + 2, my - 2, (240, 210, 150))
        cv.px(mx - 2, my + 1, (150, 110, 84))
        cv.px(mx, my - 4, (150, 96, 70))
    # fence along the track (right part) with cherry-tree rows behind it drawn per frame
    for x in range(200, w, 12):
        cv.rect(x, 140, 2, 12, (120, 76, 66))
        cv.px(x + 1, 140, (255, 210, 140))
    cv.hline(200, w - 1, 143, (236, 186, 120))
    cv.hline(200, w - 1, 147, (176, 120, 82))
    return to_layer(cv)


def comp_into(cv, lay, x, y):
    a, m = lay
    h, w = m.shape
    sub = cv.a[y:y + h, x:x + w]
    np.copyto(sub, a[:sub.shape[0], :sub.shape[1]], where=m[:sub.shape[0], :sub.shape[1], None])


FG_TREES_B = tuple((x, 162, 1, 200 + k, 'r' if k % 2 else 'y') for k, x in enumerate(range(214, 420, 22))) + \
    tuple((x, 182, 2, 300 + k, 'y' if k % 2 else 'r') for k, x in enumerate(range(236, 430, 34)))


def draw_trees(c, f, trees, ox):
    for (x, by, size, sd, kind) in trees:
        sx = int(x - ox)
        if sx < -24 or sx > W + 24:
            continue
        lay = cherry_tree(size, sd % 5, kind)
        h, w = lay[0].shape[:2]
        c.hline(sx - w // 2 - 2 - 4 * size, sx - 1, by, (140, 80, 84))
        sway = 1 if math.sin(2 * math.pi * (f / 60.0 + hash01(sd, 3))) > 0.65 else 0
        comp(c, lay, sx - w // 2 + sway, by - h + 1)


def conveyor_cherries(c, f, belt, ox, legs=()):
    x0, y0, x1, y1 = belt
    L = math.hypot(x1 - x0, y1 - y0)
    for (lx, ly0, ly1) in legs:
        c.vline(lx - ox, ly0, ly1, (110, 72, 70))
        c.px(lx - ox + 1, ly0 + 1, (236, 170, 110))
    c.line(x0 - ox, y0 + 1, x1 - ox, y1 + 1, (74, 82, 94))
    c.line(x0 - ox, y0 + 2, x1 - ox, y1 + 2, (125, 136, 148))
    c.line(x0 - ox, y0 + 3, x1 - ox, y1 + 3, (74, 58, 70))
    for k in range(0, int(L), 4):
        u = ((k + f * 0.35) % L) / L
        c.px(int(round(x0 + (x1 - x0) * u - ox)), int(round(y0 + (y1 - y0) * u)) + 2, (196, 205, 214))
    for k in range(9):
        u = ((f * 0.35 + k * L / 9) % L) / L
        x = int(round(x0 + (x1 - x0) * u - ox))
        y = int(round(y0 + (y1 - y0) * u))
        c.px(x, y, (232, 59, 59) if k % 3 else (249, 210, 60))
        if k % 2 == 0:
            c.px(x + 1, y, (174, 35, 52) if k % 3 else (214, 150, 30))


def render_b(f):
    cam = cam_b(f)
    c = Canvas(W, H, 'ink')
    comp(c, to_layer(_sky_b()), -int(round(cam * P_SUNB)), 0)
    sxs = SUN_B[0] - int(round(cam * P_SUNB))
    sun = (sxs, SUN_B[1], SUN_B[2])
    comp(c, sun_sprite(SUN_B[2], True), sxs - SUN_B[2] - 1, SUN_B[1] - SUN_B[2] - 1)
    # sun shimmer: a slowly breathing outer ring
    k = 0.5 + 0.5 * math.sin(f * 0.15)
    yy0 = SUN_B[1]
    for i in range(3):
        c.dither(sxs - SUN_B[2] - 3, yy0 - 6 + i * 5, 2 * SUN_B[2] + 6, 1, (255, 236, 180), 0.25 * k)
    draw_clouds(c, f, CLOUDS_B, cam, 0.05)
    # balloon with burner glow against the sunset
    bx = int(round(166 + 0.12 * (f - 216) - cam * 0.2))
    by = int(round(52 - 0.04 * (f - 216)))
    if (f % 36) < 20:
        fx.glow(c, bx, by - 13, 14, (90, 60, 30), 0.6, rings=3)
    draw_balloon(c, bx, by, f, 'm', flame=(f % 36) < 20)
    comp(c, _rim_b(), -int(round(cam * P_RIM)), 0)
    comp(c, _plain_b(), -int(round(cam * P_PLAINB)), 0)
    rlay, rtrees, rex = _ranch_b()
    rox = int(round(cam * P_RANCH))
    comp(c, rlay, -rox, 0)
    draw_trees(c, f, rtrees, rox)
    conveyor_cherries(c, f, rex['belt'], rox, rex['legs'])
    # shed window glow
    swx, swy = rex['shed_win']
    light_window(c, f, swx - rox, swy, 4, 5, 0, False)
    slay, swins, sex = _station_b()
    comp(c, slay, -cam, 0)
    for (x, y, w, h, t_on, rt) in swins:
        light_window(c, f, x - cam, y, w, h, 0, rt)
    for i, (lx, ly) in enumerate(sex['lamps']):
        lamp(c, lx - cam, ly, f, True, i)
    spx, spy = sex['spout']
    if (f // 7) % 3 == 0:
        c.px(spx - cam, spy + (f % 7) // 2, (196, 230, 255))
    pennant(c, f, sex['tower'][0] - cam, sex['tower'][1] - 9, (240, 79, 120), 6)
    train_steam(c, f, cam, sun)
    draw_train(c, f, cam, sun)
    semaphore(c, f, 168, RY + 12, cam)
    # foreground
    fox = int(round(cam * P_FGB))
    comp(c, _fg_b(), -fox, 0)
    draw_trees(c, f, FG_TREES_B, fox)
    draw_tumbleweeds(c, f, cam, [(220, -20, 2.4, 176, 6, 9, 1.3, 7), (240, 60, 1.6, 152, 3, 5, 1.0, 8)])
    fx.dust_motes(c, f, (0, 96, W, 60), n=10, color=(255, 226, 170), seed=9)
    # callouts
    tx_, ty_ = sex['tower']
    callout(c, tx_ - cam, ty_ - 3, 'DODGE JUNCTION', f, 222, 274, side='up', length=8)
    shx, shy = rex['shed']
    callout(c, shx - rox, shy + 2, 'CHERRY HILL RANCH', f, 236, 279, side='up', length=10)
    return c


# ================================================================== render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < 216:
        c = render_a(f)
        iris(c, 1 - seg(f, 72, 79), 204, 107)
        return c
    return render_b(f)


SFX = [
    (78, 'wind', 0.35, {'variant': 0}),     # dry evening breeze down Main Street as the iris opens
    (96, 'creak', 0.25),                     # the Salt Block's batwing doors swinging
    (144, 'wind', 0.28, {'variant': 0}),     # gust that sends the tumbleweeds; lamps light on the A chord
    (170, 'creak', 0.18),                    # windpump turning over the orchards
    (192, 'wind', 0.22, {'variant': 0}),
    (204, 'steam', 0.22),                    # pre-lap: the Friendship Express idling at Dodge Junction
    (216, 'steam', 0.55),                    # WHISTLE: the whistle-steam jet (the score plays the whistle tune)
    (216, 'chug', 0.40),                     # departure: chuffs on the 8ths, big puffs on the beats
    (252, 'chug', 0.45),
    (240, 'wind', 0.25, {'variant': 0}),     # prairie wind over Cherry Hill
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
