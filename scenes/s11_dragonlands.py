"""DRAGON LANDS: the volcanic isle + Kirin Grove
(SCENE frames 0..359: 1 travel bar + 4 diorama bars; the 2-bar map flight is added by
engine/regionfx.long_flight; HUD clock 15:15 -> 16:00, ash-red afternoon, hazy sun upper left).

Beat sheet (scene frames; bar = 72, beat = 18). Shots >= 3 s, cuts on beats (180 half-bar, 270 beat).
Cameras use shotfx.cruise (slow, even). Shot changes: zoom into the crater (180), cloud (270).
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71 on dest_screen (204, 96).
  Shot A 72-179   THE VOLCANO ISLAND FROM THE SEA. REVEAL 72: iris opens 72-79 on the cone (204, 96).
                  Ash-red sky banded with drifting ash stratus, a hazy sun (left) behind the smoke haze.
                  A big charcoal/basalt cone (lit left face, shadowed right) with a glowing crater lip and
                  orange lava rivulets pulsing down its flanks; a billowing smoke plume, underlit orange,
                  leans right; embers float. Green jungle fringe with palms, turquoise lakes in the
                  lowlands, black-sand beaches with breaking waves. Sea with the island's rippling
                  reflection, sun glitter, wave dashes. Far hazy islets. Foreground: a black sea stack with
                  a leaning palm and foam bursts (right). Balloon 'm' keeps its distance on the left.
                  Camera: eased pan right (parallax 0.2 far / 0.6 island / 1.0 sea / 1.35 fg).
                  Callouts DRAGON LANDS 90-164 (lowlands), THE VOLCANO 108-173 (crater).
                  Card 100-208.
  Shot B 180-269  ZOOM (180) INSIDE THE VOLCANIC HEART. A caldera open to the ash sky:
                  back wall of basalt columns with glowing veins and a lavafall, stepped column ledges,
                  the empty rocky throne ledge (left, under a sun-and-flame emblem), glittering gem hoards
                  (right: rubies, sapphires, emeralds, gold), a lava river with sliding crust plates,
                  bubbling pools, heat shimmer, steam vents/geysers puffing ON THE BEATS.
                  Rumble build-up 198-215 (pool swells, 1 px tremor). ERUPT 216 (bar line): flash, fountain
                  bursts from the central pool, ballistic lava blobs, ember shower, screen shake that
                  decays by ~240. Camera: slow pan right + shake. Callout GEM HOARDS 198-263.
  Shot C 270-359  CLOUD (270): KIRIN GROVE, south across the island, calm after the fire.
                  Soft jade/peach sky, the volcano faint on the far left horizon still smoking; misty
                  karst towers in 3 depths, the Lost Lagoon with rocky islets beyond, the Kirin pavilion
                  (curved green roof, red posts, glowing round windows), paper lanterns, a turquoise
                  stream stepping down in small waterfalls under a curved wooden bridge, bamboo groves
                  swaying, drifting mist and leaves. Balloon 's' between the karsts. Camera: eased pan
                  right. Callout KIRIN GROVE 288-end. Toast from N-58 (top centre kept clear).
Ambient: smoke plume, ash stratus, embers, lava pulses, waves/foam, sea glitter, reflections, balloon |
lava flow + crust, bubbles, lavafall, steam vents, gem sparkles, shimmer, eruption | bamboo sway,
waterfalls, stream ripples, lanterns, mist, leaves, lagoon glints.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'dragonlands'
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
    sa = a[Y0 - y:Y1 - y, X0 - x:X1 - x]
    sm = m[Y0 - y:Y1 - y, X0 - x:X1 - x]
    np.copyto(c.a[Y0:Y1, X0:X1], sa, where=sm[:, :, None])


def flash(c, f, f0, color='white'):
    if f == f0:
        fx.fade(c, color, 0.8)
    elif f == f0 + 1:
        fx.fade(c, color, 0.4)


def cam_ease(f, f0, f1, a, b):
    """Slow, even integer camera move (shotfx.cruise: constant speed with short ramps)."""
    return int(round(lerp(a, b, cruise(seg(f, f0, f1)))))


def vnoise(h, w, cy, cx, seed):
    """Smooth value noise (h, w) in 0..1 with cell sizes (cy, cx) (anisotropic)."""
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


def fbm(h, w, cy, cx, seed, octaves=3):
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = 2 ** o
        out += vnoise(h, w, max(1.0, cy / k), max(1.0, cx / k), seed + 31 * o) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def poly_mask(h, w, pts):
    """Boolean mask of a polygon (even-odd), vectorised over rows."""
    m = np.zeros((h, w), bool)
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    ys = np.arange(h) + 0.5
    xs = np.arange(w) + 0.5
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if y0 == y1:
            continue
        lo, hi = min(y0, y1), max(y0, y1)
        rows = np.nonzero((ys >= lo) & (ys < hi))[0]
        if len(rows) == 0:
            continue
        xi = x0 + (ys[rows] - y0) * (x1 - x0) / (y1 - y0)
        m[rows] ^= xs[None, :] < xi[:, None]
    return m


def ramp_idx(t, n, x0, y0, h, w, dith=0.7):
    thr = bayer(h, w, x0, y0)
    return np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)


def ramp_arr(ramp):
    return np.array([col(r) for r in ramp], np.uint8)


def outline_of(m):
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~inner


def blobs(cv, bl, ramp, outline=None, light=(-0.6, -0.6, 0.5), dith=0.6, rim=None, bias=0.0):
    """Shaded union of ellipsoid blobs [(cx, cy, rx, ry, z)] quantised to `ramp` (dark -> light)."""
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    ra = ramp_arr(ramp)
    n = len(ramp)
    x0 = max(0, int(math.floor(min(b[0] - b[2] for b in bl))) - 1)
    x1 = min(cv.w, int(math.ceil(max(b[0] + b[2] for b in bl))) + 2)
    y0 = max(0, int(math.floor(min(b[1] - b[3] for b in bl))) - 1)
    y1 = min(cv.h, int(math.ceil(max(b[1] + b[3] for b in bl))) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    xx += 0.5
    yy += 0.5
    best = np.full(xx.shape, -1e9, np.float32)
    NX = np.zeros(xx.shape, np.float32)
    NY = np.zeros(xx.shape, np.float32)
    NZ = np.zeros(xx.shape, np.float32)
    for (cx, cy, rx, ry, z) in bl:
        u = (xx - cx) / rx
        v = (yy - cy) / ry
        d2 = u * u + v * v
        ins = d2 <= 1.0
        zz = np.sqrt(np.clip(1 - d2, 0, 1))
        hgt = z + zz * min(rx, ry)
        upd = ins & (hgt > best)
        best[upd] = hgt[upd]
        NX[upd] = u[upd]
        NY[upd] = v[upd]
        NZ[upd] = zz[upd]
    m = best > -1e8
    lam = NX * L[0] + NY * L[1] + NZ * L[2]
    t = np.clip((lam + 1) / 2 + bias, 0, 0.999)
    lvl = ramp_idx(t, n, x0, y0, y1 - y0, x1 - x0, dith)
    out = ra[lvl]
    bound = outline_of(m)
    if outline is not None:
        out[bound] = C(outline)
        if rim is not None:
            lit = bound & ((NX * L[0] + NY * L[1]) > 0.55)
            out[lit] = C(rim)
    reg = cv.a[y0:y1, x0:x1]
    reg[m] = out[m]
    return x0, y0, m


def disc(c, cx, cy, r, color, level=1.0):
    x0, y0 = int(cx - r - 1), int(cy - r - 1)
    x1, y1 = int(cx + r + 2), int(cy + r + 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(c.w, x1), min(c.h, y1)
    if X1 <= X0 or Y1 <= Y0 or level <= 0:
        return
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    m = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    if level < 1:
        m &= bayer(Y1 - Y0, X1 - X0, X0, Y0) < level
    c.a[Y0:Y1, X0:X1][m] = col(color)


def add_light(c, mask_xy, color, k):
    """Additive light on pixels of a boolean mask placed at (x, y)."""
    m, x, y = mask_xy
    h, w = m.shape
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(c.w, x + w), min(c.h, y + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    sm = m[Y0 - y:Y1 - y, X0 - x:X1 - x]
    reg = c.a[Y0:Y1, X0:X1]
    v = reg[sm].astype(np.int16) + (np.array(col(color), np.float32) * k).astype(np.int16)
    reg[sm] = np.clip(v, 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def puff_sprite(r, tone, seed=0):
    """A lumpy shaded smoke/steam puff (RGBA array), radius ~r.
    tone 0 hot (underlit lava), 1 warm ash, 2 cool ash, 3 white steam, 4 pale mist."""
    ramps = {
        0: ['#4e2630', 'rust', 'copper', 'clay', 'sand'],
        1: ['#3f2c38', 'plum_brown', 'mauve', '#b98278', '#d6a08a'],
        2: ['#4a3c4c', '#66566a', 'grey', '#a99aa6', '#c9b6b6'],
        3: ['#9aa6b0', 'stone', 'mist', '#eef4f2', 'white'],
        4: ['#b7cfc9', '#c9ddd6', 'mint_cream', '#e7f6ee', 'white'],
        5: ['#b49aa2', '#c2aaae', '#cdb8b8', '#d8c8c4', '#e4d6d0'],
        6: ['#6a3a40', '#94534e', '#c07a66', '#e0a184', '#f4c6a4'],
    }
    lights = {0: (-0.2, 0.8, 0.5), 1: (-0.5, 0.2, 0.7), 2: (-0.6, -0.5, 0.5),
              3: (-0.5, -0.6, 0.5), 4: (-0.5, -0.6, 0.6), 5: (-0.6, -0.5, 0.6),
              6: (0.1, 0.75, 0.6)}
    sz = 2 * r + 6
    cv = new(sz, sz)
    R = rng(1000 + r * 7 + tone * 131 + seed)
    cx = cy = sz / 2
    bl = [(cx, cy + r * 0.1, r * 0.85, r * 0.75, 0.0)]
    for i in range(4 + r // 3):
        a = R.random() * 6.283
        d = r * (0.35 + 0.25 * R.random())
        rr = r * (0.4 + 0.25 * R.random())
        bl.append((cx + math.cos(a) * d, cy + math.sin(a) * d * 0.8, rr, rr * 0.9, rr * 0.3))
    blobs(cv, bl, ramps[tone], outline=None, light=lights[tone], dith=0.8)
    a, m = to_layer(cv)
    rgba = np.zeros((sz, sz, 4), np.uint8)
    rgba[:, :, :3] = a
    rgba[:, :, 3] = m * 255
    return rgba


def draw_puff(c, x, y, r, tone, seed=0, level=1.0):
    """Puff centred at (x, y); level < 1 thins it with ordered dither."""
    s = puff_sprite(max(1, int(r)), tone, seed % 3)
    sz = s.shape[0]
    x0, y0 = int(x) - sz // 2, int(y) - sz // 2
    if level >= 0.999:
        c.blit(s, x0, y0)
        return
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + sz), min(H, y0 + sz)
    if X1 <= X0 or Y1 <= Y0 or level <= 0:
        return
    sub = s[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    m = (sub[:, :, 3] > 0) & (bayer(Y1 - Y0, X1 - X0, X0, Y0) < level)
    c.a[Y0:Y1, X0:X1][m] = sub[:, :, :3][m]


def steam_puffs(c, f, x, y, period=36, n=3, height=28, rmax=6, seed=0, drift=6, tone=3):
    """Rising steam puffs that grow then shrink (no dithered fade), restarting every `period`."""
    for q in range(n):
        t = ((f + q * period / n) % period) / period
        r = rmax * math.sin(math.pi * min(1.0, 0.25 + t * 0.85)) ** 0.8
        if r < 1:
            continue
        tn = tone if (t < 0.45 or tone == 6) else 2
        draw_puff(c, x + int(t * drift) + (q % 2), y - int(t * height), int(r), tn, seed=q + seed,
                  level=1.0 if t < 0.8 else 0.7)
        if r > 4:
            draw_puff(c, x + int(t * drift) - int(r * 0.8), y - int(t * height) + 2, int(r * 0.6), tn,
                      seed=q + seed + 1)
            draw_puff(c, x + int(t * drift) + int(r * 0.8), y - int(t * height) + 1, int(r * 0.6), tn,
                      seed=q + seed + 2)


def ember(c, x, y, f, i):
    k = (f // 2 + i) % 5
    cc = ('lava_hi', 'gold', 'lava', 'amber', 'lava_hi')[k]
    c.px(x, y, cc)


# ================================================================== SHOT A: the island from the sea
A0, A1 = 72, 180
VX, VT = 204, 36            # volcano crater centre x / rim y (island layer coords = screen at cam 0)
VB = 120                    # cone base row
RT, RB = 13.0, 128.0        # half-width at crater / base
HOR = 112                   # sea horizon
WL = 130                    # island waterline
IW = 380                    # island layer width
A_PAR = 0.6


LAKES = ((94, 113, 18, 2), (150, 109, 10, 1), (262, 111, 13, 2), (326, 114, 8, 1))


def near_lake(x, y, pad=3):
    for (lx, ly, lw, lh) in LAKES:
        if abs(x - lx) < lw + pad and ly - 6 <= y <= ly + 4:
            return True
    return False


def cone_r(y):
    d = np.clip((np.asarray(y, np.float32) - VT) / (VB - VT), 0, 1)
    return RT + (RB - RT) * d ** 1.65


def cam_a(f):
    return cam_ease(f, A0, A1 - 1, 0, 34)


SKY_A = ['#3a1c33', '#4f2338', '#672b3e', 'wine', '#923c42', '#a84a44', '#bd5d4a', '#d17654',
         '#e08f60', '#eaa673', '#f0b98a']


@lru_cache(maxsize=1)
def _sky_a():
    c = Canvas()
    c.gradient(0, 0, W, HOR + 1, SKY_A, soft=0.7)
    # hazy sun behind the ash veil (low in the west, upper left)
    fx.glow(c, 40, 62, 70, 'coral', 0.30, rings=6)
    fx.glow(c, 40, 62, 34, 'sand', 0.35, rings=5)
    disc(c, 40, 62, 9, '#f6c79a')
    disc(c, 40, 62, 8, '#fbd9a8')
    disc(c, 39, 61, 6, 'cream')
    # a veil of ash crossing the sun
    for yy, x0, x1 in ((60, 18, 66), (63, 24, 60), (66, 14, 70)):
        c.hline(x0, x1, yy, '#e4a77e')
    # far islets on the horizon (hazy)
    hz = mix('#e09a74', 'mauve', 0.35)
    hz2 = mix('#e09a74', 'mauve', 0.15)
    c.poly([(0, HOR + 1), (0, 104), (6, 101), (14, 99), (20, 102), (30, HOR + 1)], hz)
    c.poly([(8, HOR + 1), (16, 104), (24, 106), (34, HOR + 1)], hz2)
    c.poly([(286, HOR + 1), (298, 100), (304, 97), (309, 99), (322, 106), (330, HOR + 1)], hz)
    c.poly([(302, 98), (304, 97), (306, 98)], '#f0b08c')
    # far sea
    c.gradient(0, HOR, W, H - HOR, ['#e7a07c', '#b9786f', '#7d5a66', '#3f5560', '#1f4a55',
                                     '#16404d', '#123844'], soft=0.6)
    c.hline(0, W - 1, HOR, '#f3c09a')
    return c


@lru_cache(maxsize=1)
def _strata():
    """Long thin ash stratus strips (RGBA layers) drifting across the sky."""
    out = []
    specs = [(420, 3, 16, '#4a2338', '#7a3a48', 3), (380, 2, 30, '#5c2a3c', '#934a4c', 5),
             (360, 3, 50, '#7a3a44', '#b8644f', 7), (300, 2, 76, '#a24e48', '#d68662', 9),
             (260, 2, 92, '#c06a54', '#eaa070', 11)]
    for w, h, y, dark, lit, seed in specs:
        cv = new(w, h + 4)
        R = rng(seed)
        x = 0
        while x < w - 20:
            L = int(30 + R.random() * 70)
            th = 1 + int(R.random() * h)
            cv.rect(x + 2, 2, L - 4, th, dark)
            cv.rect(x, 2 + th // 2, L, max(1, th - th // 2) + 1, dark)
            cv.hline(x + 3, x + L - 6, 1, lit)
            x += L + int(10 + R.random() * 40)
        out.append((to_layer(cv), y, w))
    return out


def draw_strata(c, f, cam, par=0.15):
    for i, (lay, y, w) in enumerate(_strata()):
        sp = 0.06 + 0.035 * i
        x = -int(f * sp + cam * par * (0.5 + 0.2 * i)) % w
        comp(c, lay, x - w, y)
        comp(c, lay, x, y)


@lru_cache(maxsize=1)
def _island():
    """The volcano island (layer, island coords) + extras (rivulet pixels, glow mask, palms)."""
    h = WL + 2
    cv = new(IW, h)
    yy, xx = np.mgrid[0:h, 0:IW].astype(np.float32)
    r = cone_r(yy)
    # ragged silhouette: noise on the half width
    nz = fbm(h, IW, 6, 9, 11)
    rr = r + (nz - 0.5) * 6 * np.clip((yy - VT) / 20, 0, 1)
    u = (xx + 0.5 - VX) / rr
    cone = (np.abs(u) <= 1) & (yy >= VT) & (yy <= VB + 12)
    # crater notch: the rim is a little irregular
    rimn = (np.abs(xx - VX) < RT) & (yy < VT + 1 + (np.abs(xx - VX) < 6) * 1)
    cone &= ~rimn
    # surface-of-revolution lighting: sun low upper-left + lava/ash ambient
    nzc = np.sqrt(np.clip(1 - u * u, 0, 1))
    slope = 0.55
    lam = (-u * 0.75 + nzc * 0.35 - slope * 0.25)
    # radial gullies: noise in (u, y)
    gu = vnoise(h, IW, 40, 1, 5)
    ucoord = ((u + 1) * 60).astype(np.float32)
    gi = np.clip(ucoord.astype(int), 0, IW - 1)
    gul = gu[np.clip(yy.astype(int), 0, h - 1), gi]
    ridge = vnoise(h, 160, 30, 1.6, 7)[np.clip(yy.astype(int), 0, h - 1), np.clip((ucoord * 1.0).astype(int), 0, 159)]
    t = (lam + 1) / 2 + (ridge - 0.5) * 0.45 + (gul - 0.5) * 0.15
    t = np.clip(t, 0, 0.999)
    ramp = ['#17131b', '#1f1a23', 'basalt', '#332b34', '#3f343c', '#544148', '#74514e']
    lvl = ramp_idx(t, len(ramp), 0, 0, h, IW, 0.8)
    ra = ramp_arr(ramp)
    cv.a[cone] = ra[lvl][cone]
    # crater: glowing throat
    for i, (w2, cc) in enumerate(((RT - 1, 'lava_dk'), (RT - 3, 'lava'), (RT - 7, 'lava_hi'))):
        cv.hline(int(VX - w2), int(VX + w2), VT + 1 + (i > 1), cc)
    cv.hline(VX - int(RT) + 1, VX + int(RT) - 1, VT, '#5a2a26')
    cv.hline(VX - 6, VX + 6, VT, 'lava_dk')
    cv.px(VX - int(RT), VT + 1, '#8a5e5c')
    # rim highlight on the sunlit (left) silhouette
    bound = outline_of(cone)
    left_edge = bound & (xx < VX) & (yy > VT + 1)
    right_edge = bound & (xx >= VX) & (yy > VT + 1)
    cv.a[left_edge] = C('#b0705a')
    cv.a[right_edge] = C('#1b1620')
    # glow of the crater on the upper cone
    glow_m = cone & ((xx - VX) ** 2 / 900 + (yy - VT) ** 2 / 160 < 1)
    reg = cv.a.astype(np.int16)
    reg[glow_m] += np.array([40, 10, 0], np.int16)
    cv.a[:] = np.clip(reg, 0, 255).astype(np.uint8)

    # ---- lava rivulets (pixel lists with a 0..1 run parameter)
    R = rng(21)
    riv = []
    for j, (u0, y_end, wig) in enumerate(((-0.55, 86, 0.10), (-0.15, 98, 0.07), (0.22, 92, 0.09),
                                         (0.58, 80, 0.06), (-0.80, 70, 0.05))):
        pts = []
        px_prev = None
        ph = R.random() * 6
        for y in range(VT + 2, y_end + 1):
            rw = float(cone_r(y))
            uu = u0 * (0.5 + 0.5 * min(1, (y - VT) / 14)) + wig * math.sin(y * 0.21 + ph)
            x = int(round(VX + uu * rw))
            if px_prev is None:
                xs_ = [x]
            else:
                lo, hi = sorted((px_prev, x))
                xs_ = list(range(lo, hi + 1))
            for xq in xs_:
                pts.append((xq, y, (y - VT) / (y_end - VT)))
            px_prev = x
        riv.append(pts)
    rx = np.array([p[0] for pts in riv for p in pts], int)
    ry = np.array([p[1] for pts in riv for p in pts], int)
    rs = np.array([p[2] for pts in riv for p in pts], np.float32)
    rj = np.array([j for j, pts in enumerate(riv) for p in pts], int)
    # crust edges + glow halo baked into the cone
    halo = np.zeros((h, IW), bool)
    for dx in (-2, -1, 1, 2):
        halo[ry, np.clip(rx + dx, 0, IW - 1)] = True
    halo[np.clip(ry + 1, 0, h - 1), rx] = True
    halo &= cone
    v = cv.a.astype(np.int16)
    v[halo] += np.array([70, 20, 0], np.int16)
    cv.a[:] = np.clip(v, 0, 255).astype(np.uint8)
    for dx in (-1, 1):
        sel = cone[ry, np.clip(rx + dx, 0, IW - 1)]
        cv.a[ry[sel], np.clip(rx + dx, 0, IW - 1)[sel]] = C('lava_dk')
    # lava fans where rivulets fade into the forest
    for pts in riv:
        x, y, _ = pts[-1]
        cv.px(x - 1, y + 1, 'lava_dk')
        cv.px(x + 1, y + 1, 'lava_dk')

    # ---- older eroded cone to the right
    pts = [(268, VB + 4), (290, 86), (300, 76), (306, 74), (314, 77), (330, 92), (360, VB + 4)]
    m2 = poly_mask(h, IW, pts)
    m2 &= ~cone
    nz2 = fbm(h, IW, 5, 5, 31)
    t2 = np.clip(0.55 - (xx - 306) / 90 + (nz2 - 0.5) * 0.7, 0, 0.999)
    ramp2 = ['#22201f', '#2f3330', 'slate', '#4f5a4e', '#6f6a58']
    l2 = ramp_idx(t2, len(ramp2), 0, 0, h, IW, 0.8)
    cv.a[m2] = ramp_arr(ramp2)[l2][m2]
    b2 = outline_of(m2) & (yy < 100)
    cv.a[b2 & (xx < 306)] = C('#8a7a66')
    cv.a[b2 & (xx >= 306)] = C('#1b1620')

    # ---- green fringe: meadow on the lower flanks (where the slope flattens)
    gnz = fbm(h, IW, 4, 7, 41)
    gl = 98 + (gnz - 0.5) * 10 + np.abs(xx - VX) * -0.04
    # gullies of green reaching up
    scrub = cone & (yy >= gl - 4) & (yy < gl) & (bayer(h, IW) < (yy - gl + 4) / 5)
    cv.a[scrub] = C('dark_olive')
    land = (cone | m2) | ((yy >= 112) & (xx > 44 + (WL - yy) * 1.4) & (xx < 374 - (WL - yy) * 0.8))
    land |= poly_mask(h, IW, [(44, WL), (58, 118), (72, 111), (90, 108), (120, 112), (120, WL)])
    green = land & (yy >= gl)
    mead = ['#1c5a3e', 'green', '#3aa05a', 'mint', '#6cc85e', 'leaf', '#b8e070']
    tm = np.clip(0.62 + (fbm(h, IW, 3, 6, 43) - 0.5) * 0.9 - (yy - 104) * 0.012 - (xx - 120) * 0.0006,
                 0, 0.999)
    lm = ramp_idx(tm, len(mead), 0, 0, h, IW, 0.9)
    cv.a[green] = ramp_arr(mead)[lm][green]
    edge = green & ~np.roll(green, 1, axis=0)
    cv.a[edge] = C('#0f3d33')
    cv.a[np.roll(edge, 1, axis=0) & green] = C('pale_leaf')
    # left lowland base
    lowl = land & ~green & ~cone & ~m2
    cv.a[lowl] = C('green')

    # ---- turquoise lakes in the lowlands
    for (lx, ly, lw, lh) in LAKES:
        cv.ellipse(lx, ly, lw + 2, lh + 1, '#d8c890')
        cv.ellipse(lx, ly, lw + 1, lh, 'aqua')
        cv.ellipse(lx, ly - 0, lw, max(0.5, lh - 1), 'turquoise')
        cv.hline(lx - lw // 2, lx + lw // 3, ly - lh + 1, 'ice')
        cv.hline(lx - lw + 2, lx + lw - 2, ly + lh, '#0b7a76')

    # ---- jungle: rows of round canopies (each row one shaded union -> clumps, not bushes)
    R = rng(51)
    rows = [(105, 2.2, 0.7, ['#0f3d33', 'forest', '#1f7a55', 'green', 'mint']),
            (113, 3.0, 0.85, ['#0f3d33', 'forest', 'green', 'mint', '#5cc865']),
            (121, 3.8, 1.0, ['#0b3530', 'forest', 'green', 'mint', 'leaf'])]
    for (by, rad, dens, jr) in rows:
        bl = []
        x = 46 + (WL - by) * 1.3
        while x < 372 - (WL - by) * 0.6:
            if R.random() < dens and not near_lake(x, by):
                lift = 0
                if 150 < x < 260 and by < 110:
                    lift = int(4 * (1 - abs(x - VX) / 56))
                rad_ = rad + R.random() * 1.8
                cy_ = by - lift + R.random() * 2.5
                bl.append((x, cy_, rad_ * 1.15, rad_, R.random() * 2))
                bl.append((x + rad_ * 0.9, cy_ + 1.5, rad_ * 0.8, rad_ * 0.75, R.random() * 2))
            x += int(rad * 1.9 + R.random() * 3)
        blobs(cv, bl, jr, outline='#0b2e2a', light=(-0.7, -0.65, 0.4), dith=0.75)
    # ---- black sand beach + rocks
    bnz = fbm(1, IW, 1, 8, 61)[0]
    for x in range(44, 374):
        top = WL - 3 - int(bnz[x] * 3)
        cv.vline(x, top, WL, 'basalt')
        cv.px(x, top, '#4a4048')
        if hash01(x, 7) < 0.3:
            cv.px(x, top + 1 + int(hash01(x, 8) * 2), '#5e5260')
    for (sx, sh, sw) in ((60, 10, 5), (66, 6, 3), (348, 12, 6), (356, 7, 4), (228, 5, 3)):
        cv.rect(sx, WL - sh, sw, sh, '#231f27')
        cv.vline(sx, WL - sh, WL - 1, '#6a5560')
        cv.hline(sx, sx + sw - 1, WL - sh, '#7a6268')
    # palm spots (drawn per frame): (x, base_y, height, lean)
    palms = [(58, 121, 16, -1), (84, 117, 14, 1), (128, 118, 18, 1), (176, 121, 21, -1),
             (232, 120, 22, 1), (276, 117, 15, -1), (300, 121, 19, 1), (340, 118, 16, -1),
             (362, 122, 13, 1), (196, 120, 17, 1)]
    return to_layer(cv), (rx, ry, rs, rj), palms


PALM_FROND = [(-5, 1), (-3, -1), (0, -2), (3, -1), (5, 1)]


def draw_palm(c, x, base, hgt, lean, f, seed, scale=1):
    """Small palm: curved trunk + 4-5 fronds that sway."""
    tx, ty = x, base
    pts = []
    for i in range(hgt):
        t = i / max(1, hgt - 1)
        pts.append((x + int(round(lean * t * t * 3 * scale)), base - i * scale))
    for (px_, py_) in pts:
        c.px(px_, py_, '#8a6248')
        if scale > 1:
            c.px(px_ + 1, py_, '#6b4a3a')
            c.px(px_ - 1, py_, '#b08460')
    hx, hy = pts[-1]
    sw = int(round(math.sin(f * 2 * math.pi / 54 + seed) * 1.2))
    for k, (dx, dy) in enumerate(PALM_FROND):
        ex, ey = hx + dx * scale + (sw if dx > 0 else (sw // 2)), hy + dy * scale + (1 if abs(dx) > 3 else 0)
        c.line(hx, hy - 1, ex, ey, 'mint' if k % 2 else 'leaf')
        c.px(ex, ey + 1, 'green')
    c.px(hx, hy - 1, 'leaf')
    c.px(hx, hy, '#5a3a2a')


@lru_cache(maxsize=1)
def _wave_tex():
    """Sea wave dashes for rows WL..H: values 0 none, 1 light dash, 2 dark trough."""
    h, w = H - HOR, 640
    t = np.zeros((h, w), np.uint8)
    for y in range(h):
        d = y / h                                      # 0 far .. 1 near
        L = 2 + int(d * 7)
        gap = 6 + int(d * 18)
        R = rng(300 + y)
        x = int(R.random() * gap)
        while x < w:
            if R.random() < 0.55:
                t[y, x:x + L] = 1 if R.random() < 0.6 else 2
            x += L + gap + int(R.random() * gap)
    return t


def draw_plume(c, f, ox, x0=VX, y0=VT, n=28, period=320, rise=24, drift=230, rmax=20, seed=0, hot=True):
    """Billowing smoke plume from (x0+ox, y0): rises, then bends right with the wind (oldest drawn first)."""
    items = []
    for i in range(n):
        p = ((f + i * period / n + hash01(seed, i) * 5) % period) / period
        items.append((p, i))
    items.sort(reverse=True)
    for p, i in items:
        y = y0 - rise * (1 - (1 - p) ** 2.2) - 2 + math.sin(p * 7 + i) * 2 * p
        x = x0 + ox + drift * p ** 1.6 + (hash01(seed, i, 2) - 0.5) * 6 * p
        r = 3 + rmax * p ** 0.55
        if hot:
            tone = 0 if p < 0.07 else (1 if p < 0.30 else 2)
        else:
            tone = 2
        lvl = 1.0 if p < 0.8 else 1 - (p - 0.8) / 0.2 * 0.8
        draw_puff(c, int(x), int(y), int(r), tone, seed=i, level=lvl)


@lru_cache(maxsize=1)
def _fg_a():
    """Foreground basalt sea stack (right), columnar, mossy top."""
    w, h = 96, 96
    cv = new(w, h)
    top = [(4, 95), (8, 60), (12, 50), (18, 44), (24, 38), (31, 33), (38, 30), (46, 28), (54, 30),
           (62, 34), (70, 40), (78, 48), (86, 58), (92, 70), (95, 95)]
    m = poly_mask(h, w, top)
    yy, xx = np.mgrid[0:h, 0:w]
    nz = fbm(h, w, 6, 3, 81)
    # columns: vertical joints every ~6 px, each column slightly different tone
    colid = ((xx + (nz * 4).astype(int)) // 6)
    ctone = np.array([hash01(k, 3) for k in range(40)])[np.clip(colid, 0, 39)]
    lam = 0.75 - xx / w * 0.75 + (ctone - 0.5) * 0.35 + (nz - 0.5) * 0.4
    ramp = ['#17131b', '#1f1a23', 'basalt', '#352c35', '#46393f', '#5e464b', '#80594f']
    lvl = ramp_idx(np.clip(lam, 0, 0.999), len(ramp), 0, 0, h, w, 0.8)
    cv.a[m] = ramp_arr(ramp)[lvl][m]
    joint = m & (((xx + (nz * 4).astype(int)) % 6) == 0) & (yy > 40)
    cv.a[joint] = C('#120f16')
    jl = m & (((xx + (nz * 4).astype(int)) % 6) == 1) & (yy > 40) & (xx < 50)
    cv.a[jl] = C('#4d3c43')
    # horizontal cracks
    for k in range(9):
        y0 = 46 + k * 6 + int(hash01(k, 1) * 3)
        x0 = int(hash01(k, 2) * 60)
        cv.hline(x0, x0 + 6 + int(hash01(k, 4) * 10), y0, '#120f16')
    bd = outline_of(m)
    cv.a[bd & (xx < 44)] = C('#9a6a58')
    cv.a[bd & (xx >= 44)] = C('#120f16')
    # jungle cap
    bl = []
    R = rng(83)
    for i in range(9):
        x = 14 + i * 9 + R.random() * 3
        yb = np.nonzero(m[:, int(min(w - 1, x))])[0]
        yt = yb[0] if len(yb) else 40
        bl.append((x, yt + 1, 6 + R.random() * 2.5, 4 + R.random() * 1.5, R.random()))
    blobs(cv, bl, ['#0f3d33', 'forest', 'green', 'mint', 'leaf'], outline='#0b2e2a',
          light=(-0.7, -0.6, 0.45), dith=0.7)
    # hanging vines
    for i in range(7):
        x = 18 + i * 10 + int(hash01(i, 6) * 4)
        L = 5 + int(hash01(i, 7) * 12)
        yb = np.nonzero(m[:, x])[0]
        yt = (yb[0] if len(yb) else 40) + 3
        cv.vline(x, yt, yt + L, 'forest')
        cv.px(x, yt + L + 1, 'green')
    return to_layer(cv)


def draw_big_palm(c, x, base, hgt, lean, f, seed):
    """Foreground palm: thick ringed trunk curving `lean`, drooping fronds that sway."""
    pts = []
    for i in range(hgt):
        t = i / max(1, hgt - 1)
        pts.append((x + lean * t * t, base - i))
    for i, (px_, py_) in enumerate(pts):
        px_ = int(round(px_))
        c.hline(px_ - 1, px_ + 1, py_, '#8a6248')
        c.px(px_ - 1, py_, '#c09068')
        c.px(px_ + 1, py_, '#5e4034')
        if i % 3 == 0:
            c.hline(px_ - 1, px_ + 1, py_, '#6b4a3a')
    hx, hy = int(round(pts[-1][0])), pts[-1][1]
    sw = math.sin(f * 2 * math.pi / 48 + seed)
    for k in range(7):
        a = -math.pi + k * math.pi / 6 + 0.05 * sw * (1 if k > 3 else 0.5)
        L = 13 + (k % 2) * 3
        prev = None
        for j in range(L):
            t = j / L
            ex = hx + math.cos(a) * j * 1.0 + sw * t * t * 1.5
            ey = hy + math.sin(a) * j * 0.55 + (t * t) * 7
            ex, ey = int(round(ex)), int(round(ey))
            c.px(ex, ey, 'green' if j < L - 3 else 'forest')
            c.px(ex, ey + 1, 'forest' if k > 3 else 'green')
            if j % 2 == 0 and j > 2:
                c.px(ex, ey + 1, 'mint' if k < 4 else 'green')
                c.px(ex + (1 if math.cos(a) > 0 else -1), ey + 2, 'forest')
            if j < 4:
                c.px(ex, ey - 1, 'leaf')
    c.rect(hx - 1, hy - 1, 3, 2, '#5a3a2a')
    c.px(hx - 1, hy + 1, 'clay')
    c.px(hx + 1, hy + 1, 'copper')


def render_a(f):
    cam = cam_a(f)
    c = _sky_a().copy()
    draw_strata(c, f, cam)
    ox = -int(round(cam * A_PAR))
    (lay, riv, palms) = _island()
    # plume first (behind the cone top? no: rises from the crater, drawn behind the cone silhouette)
    draw_plume(c, f, ox)
    comp(c, lay, ox, 0)
    # crater glow flicker + lava pulses down the rivulets
    rx, ry, rs, rj = riv
    ph = (rs * 3.2 - f * 0.045 + rj * 0.37) % 1.0
    cols_ = np.where(ph[:, None] < 0.18, C('lava_hi'), np.where(ph[:, None] < 0.55, C('lava'), C('#e0401a')))
    sx = rx + ox
    ok = (sx >= 0) & (sx < W)
    c.a[ry[ok], sx[ok]] = cols_[ok]
    fl = (f // 3) % 4
    c.hline(VX + ox - 5 + fl % 2, VX + ox + 4 - fl // 2, VT + 1, 'butter' if fl == 0 else 'lava_hi')
    fx.glow(c, VX + ox, VT - 2, 22, 'lava', 0.22 + 0.06 * math.sin(f * 0.3), rings=4)
    # palms (sway)
    for k, (px_, pb, ph_, ln) in enumerate(palms):
        draw_palm(c, px_ + ox, pb, ph_, ln, f, k)
    # embers rising from the crater
    for i in range(26):
        per = 70 + int(hash01(i, 3) * 50)
        t = ((f + hash01(i, 1) * per) % per) / per
        x = VX + ox + (hash01(i, 2) - 0.5) * 22 + t * (30 + hash01(i, 4) * 40) + math.sin(t * 8 + i) * 2
        y = VT - 2 - t * (40 + hash01(i, 5) * 50)
        if hash01(i, f // 4) < 0.85 - t * 0.5:
            ember(c, int(x), int(y), f, i)
    # fumaroles on the flanks
    for k, (vx_, vy_) in enumerate(((176, 70), (236, 78), (160, 88))):
        fx.steam(c, vx_ + ox, vy_, f, color='#c8b4b0', n=3, height=10, spread=2, seed=k, speed=0.25)
    # a little sailing ship on the far left horizon (no crew)
    shx = 6 + int(f * 0.09) - 6
    c.rect(shx, HOR - 1, 7, 2, '#5a3a40')
    c.hline(shx + 1, shx + 5, HOR + 1, '#7a5a5c')
    c.vline(shx + 3, HOR - 8, HOR - 2, '#5a3a40')
    c.tri((shx + 4, HOR - 8), (shx + 4, HOR - 3), (shx + 7, HOR - 3), '#f6dcc0')
    c.tri((shx + 2, HOR - 7), (shx + 2, HOR - 3), (shx - 1, HOR - 3), '#e8c4a8')
    c.px(shx + 3, HOR - 9, 'hot_pink')
    # ---- sea: reflection of everything above the waterline, rippled + tinted
    nrow = H - WL - 1
    src_rows = WL - 1 - np.arange(nrow)
    rows = WL + 1 + np.arange(nrow)
    d = np.arange(nrow) / nrow
    shift = np.round(np.sin(f * 0.16 + np.arange(nrow) * 0.9) * (1 + d * 2)).astype(int)
    xi = np.clip(np.arange(W)[None, :] + shift[:, None], 0, W - 1)
    refl = c.a[src_rows[:, None], xi].astype(np.float32)
    seacol = np.array([np.array(col(mix('#18606a', '#0d3846', t)), np.float32) for t in d])
    k = (0.50 + 0.38 * d)[:, None, None]
    out = refl * (1 - k) + seacol[:, None, :] * k
    out *= np.array([0.92, 0.95, 1.0], np.float32)
    c.a[rows] = np.clip(out, 0, 255).astype(np.uint8)
    # wave dashes (far rows + near rows), parallax by depth
    tex = _wave_tex()
    yrows = np.arange(HOR + 1, H)
    dd = (yrows - HOR) / (H - HOR)
    offs = (f * (0.15 + dd * 0.5) + cam * (0.4 + dd * 1.0)).astype(int)
    xs = (np.arange(W)[None, :] + offs[:, None]) % tex.shape[1]
    tv = tex[(yrows - HOR)[:, None], xs]
    reg = c.a[HOR + 1:H]
    reg[tv == 1] = np.clip(reg[tv == 1].astype(np.int16) + 34, 0, 255).astype(np.uint8)
    reg[tv == 2] = (reg[tv == 2] * 0.75).astype(np.uint8)
    # sun glitter path
    for i in range(40):
        yq = HOR + 2 + int(hash01(i, 11) * (H - HOR - 4))
        spread = 3 + (yq - HOR) * 0.18
        xq = 40 + int((hash01(i, 12) - 0.5) * 2 * spread)
        if hash01(i, f // 3) < 0.45:
            L = 1 + int((yq - HOR) / 20)
            c.hline(xq, xq + L, yq, 'butter' if hash01(i, 13) < 0.5 else '#f6c79a')
    # lava glow reflections (warm streaks below the rivulets' ends)
    for j, xr in enumerate((180, 200, 222)):
        for q in range(4):
            yq = WL + 3 + q * 4 + j
            if hash01(j, q, f // 4) < 0.6:
                c.hline(xr + ox - 1 + int(hash01(j, q, f // 6) * 3), xr + ox + 1, yq, '#c0502e')
    # beach waves: foam swashes that run up and back (one set per bar)
    ph = (f % 72) / 72
    run = math.sin(ph * math.pi) ** 2
    for x in range(max(0, 44 + ox), min(W, 374 + ox)):
        xi_ = x - ox
        seg_ = xi_ // 9
        on = hash01(seg_, 3) < 0.7
        if not on:
            continue
        wv = int(run * (1 + hash01(seg_, 4) * 2.5))
        c.px(x, WL - wv, 'white' if hash01(xi_, f // 6) < 0.8 else 'mist')
        if hash01(xi_, 9) < 0.5:
            c.px(x, WL + 1, '#7fb8b4')
        if wv >= 2 and hash01(xi_, 5) < 0.5:
            c.px(x, WL - wv + 1, 'mist')
    # balloon keeping its distance (left, between the sun and the cone)
    bx = 118 - int(seg(f, A0, A1) * 10)
    draw_balloon(c, bx, 78, f, 'm')
    # foreground basalt sea stack with a leaning palm (parallax 1.25)
    fxo = 250 - int(round(cam * 1.25))
    fy = H - 96 + 2
    comp(c, _fg_a(), fxo, fy)
    draw_big_palm(c, fxo + 40, fy + 30, 26, -7, f, 9)
    # foam wash around its foot; spray on the wave beat (once per bar)
    for x in range(max(0, fxo + 2), min(W, fxo + 96)):
        yb = H - 3 + int(round(math.sin(x * 0.35 + f * 0.13) * 1.2))
        if hash01(x, 2) < 0.8:
            c.px(x, yb, 'white' if hash01(x, f // 5) < 0.75 else 'mist')
        if hash01(x, 3) < 0.4:
            c.px(x, yb - 1, 'mist')
    dt = (f + 30) % 72
    if dt < 26:
        for q in range(26):
            a_ = -math.pi / 2 + (hash01(q, 1) - 0.5) * 1.8
            v = 0.9 + hash01(q, 2) * 0.9
            sx_ = fxo + 6 + hash01(q, 3) * 86
            x = sx_ + math.cos(a_) * v * dt
            y = H - 3 + math.sin(a_) * v * dt * 1.4 + 0.07 * dt * dt
            if y < H - 2 and hash01(q, dt // 3) < 1 - dt / 30:
                c.px(int(x), int(y), 'white' if q % 3 else 'mist')
    # warm key light grade: ash-red afternoon
    fx.grade(c, (1.0, 0.96, 0.94), (6, 0, 0))
    # callouts
    callout(c, 98 + ox, 112, 'DRAGON LANDS', f, 90, 164, side='down', length=10)
    callout(c, VX + ox + 6, VT + 1, 'THE VOLCANO', f, 108, 173, side='right', length=14)
    return c


# ================================================================== SHOT B: inside the volcanic heart
B0, B1 = 180, 270
ERUPT = 216
BW = 380                        # layer widths (cam pans 0 -> 24 + shake)
RIV0, RIV1 = 130, 160           # lava river rows (world = screen at cam 0, par 1.0)
POOL = (214, 143)               # eruption pool centre (river layer coords)
BAS = ['#120e15', '#19141c', '#211a23', 'basalt', '#36292f', '#4c3131', '#6c3a2e', '#97492a', '#c4602c']
LAVA_CUT = ('lava_dk', '#c22e16', 'lava', '#ff7a2a', 'amber', 'lava_hi', 'butter')


def cam_b(f):
    return cam_ease(f, B0, B1 - 1, 0, 24)


def shake_b(f):
    if ERUPT - 18 <= f < ERUPT:
        k = seg(f, ERUPT - 18, ERUPT)
        return fx.shake(f, 1 if (f % 2 == 0 and k > 0.3) else 0, seed=5)
    if ERUPT <= f < ERUPT + 24:
        amp = int(round(3 * (1 - seg(f, ERUPT, ERUPT + 24)) ** 1.2))
        return fx.shake(f, amp, seed=7)
    return 0, 0


def basalt_columns(cv, mask, seed, glow_y0, glow_y1, base=0.15, cw=(4, 7), joint=(7, 14), ramp=BAS,
                   lit_side=0, cap=True):
    """Fill `mask` with columnar basalt: vertical prisms, caps, cross joints, lava uplight."""
    h, w = mask.shape
    R = rng(seed)
    yy = np.arange(h)[:, None]
    nz = fbm(h, w, 5, 5, seed + 3)
    out = np.zeros((h, w), np.float32)
    xcol = np.zeros(w, int)
    edges = []
    x = 0
    k = 0
    while x < w:
        ww = int(cw[0] + R.random() * (cw[1] - cw[0] + 1))
        xcol[x:x + ww] = k
        edges.append((x, min(w, x + ww), R.random()))
        x += ww
        k += 1
    glow = np.clip((yy - glow_y0) / max(1, glow_y1 - glow_y0), 0, 1) ** 1.3
    t = base + glow * 0.75 + (nz - 0.5) * 0.25
    t = np.broadcast_to(t, (h, w)).copy()
    jm = np.zeros((h, w), bool)
    hm = np.zeros((h, w), bool)
    dm = np.zeros((h, w), bool)
    for (x0, x1, tone) in edges:
        t[:, x0:x1] += (tone - 0.5) * 0.14
        hm[:, x0] = True
        dm[:, x1 - 1] = True
        y = int(R.random() * joint[1])
        while y < h:
            jm[y, x0:x1] = True
            y += int(joint[0] + R.random() * (joint[1] - joint[0]))
    if lit_side < 0:
        hm, dm = dm, hm
    t[hm] += 0.10
    t[dm] -= 0.12
    t[jm] -= 0.16
    ra = ramp_arr(ramp)
    lvl = ramp_idx(np.clip(t, 0, 0.999), len(ramp), 0, 0, h, w, 0.8)
    rgb = ra[lvl]
    cv.a[mask] = rgb[mask]
    if cap:
        top = mask & ~np.vstack([np.zeros((1, w), bool), mask[:-1]])
        cv.a[top] = C('#5a4446')
        top2 = np.vstack([np.zeros((1, w), bool), top[:-1]]) & mask
        cv.a[top2] = C('#3a2c32')
    return cv


@lru_cache(maxsize=1)
def _sky_b():
    c = Canvas()
    c.gradient(0, 0, W, 70, SKY_A[1:9], soft=0.7)
    fx.glow(c, 170, 70, 90, 'lava', 0.18, rings=5)
    return c


@lru_cache(maxsize=1)
def _wall_b():
    """Far caldera wall (par 0.3): columnar basalt with glowing veins, notch for the lavafall."""
    w, h = BW, RIV0 + 6
    cv = new(w, h)
    R = rng(91)
    # ragged far rim
    xs = np.arange(w)
    rim = 44 + 8 * np.sin(xs * 0.021 + 1.2) + 5 * np.sin(xs * 0.07) + (vnoise(1, w, 1, 9, 93)[0] - 0.5) * 8
    rim -= 34 * np.clip((60 - xs) / 60, 0, 1) ** 1.2
    rim -= 30 * np.clip((xs - (w - 70)) / 70, 0, 1) ** 1.2
    # tiers: the rim steps in columns
    rim = (np.round(rim / 3) * 3).astype(int)
    yy = np.arange(h)[:, None]
    mask = yy >= rim[None, :]
    basalt_columns(cv, mask, 95, 60, RIV0 + 4, base=0.05, cw=(4, 6), joint=(8, 16))
    # a mid tier (entablature) band of rough rock
    band = mask & (yy >= rim[None, :] + 14) & (yy < rim[None, :] + 22)
    nz = fbm(h, w, 3, 3, 97)
    cv.a[band & (nz < 0.45)] = C('#19141c')
    cv.a[band & (nz >= 0.45) & (nz < 0.6)] = C('#2a2129')
    # glowing veins (static dark part; bright pulses drawn per frame)
    veins = []
    for j in range(7):
        x = 20 + j * 52 + R.random() * 20
        y = rim[int(min(w - 1, x))] + 12 + R.random() * 20
        pts = []
        for i in range(int(14 + R.random() * 26)):
            pts.append((int(x), int(y)))
            x += (R.random() - 0.5) * 2.4
            y += 1
        veins.append(pts)
        for (vx, vy) in pts:
            cv.px(vx, vy, 'lava_dk')
            cv.px(vx + 1, vy, '#4c2422')
    # lavafall: a lip on the rim and a dark channel down the wall
    LF = 118
    lip = rim[LF] + 3
    cv.rect(LF - 6, lip - 2, 13, 4, '#2a1a1c')
    cv.hline(LF - 6, LF + 6, lip - 2, '#97492a')
    # atmospheric haze: smoky red-mauve veil, stronger high up
    a, m = to_layer(cv)
    yy = np.arange(h)[:, None, None]
    k = np.clip(0.55 - (yy - 40) / 200, 0.18, 0.55)
    haze = np.array(col('#6a3038'), np.float32)
    v = a.astype(np.float32) * (1 - k) + haze * k
    a2 = np.where(m[:, :, None], np.clip(v, 0, 255).astype(np.uint8), a)
    cv.a[:] = a2
    return to_layer(cv), veins, rim, LF


@lru_cache(maxsize=1)
def _mid_b():
    """Mid ledges (par 0.6): the throne dais (left) and the gem-hoard terraces (right)."""
    w, h = BW, H
    cv = new(w, h)
    # left stepped platform: tops at 86 (dais) stepping down toward the river
    left = poly_mask(h, w, [(0, 84), (98, 84), (100, 94), (118, 94), (120, 106), (138, 106),
                            (140, 118), (156, 118), (158, RIV0 + 4), (0, RIV0 + 4)])
    right = poly_mask(h, w, [(236, RIV0 + 4), (238, 120), (252, 120), (254, 108), (300, 108),
                             (302, 96), (w, 96), (w, RIV0 + 4)])
    mask = left | right
    basalt_columns(cv, mask, 101, 80, RIV0 + 2, base=0.12, cw=(5, 7), joint=(6, 12), lit_side=0)
    # front faces (below each step top) lit warmer toward the lava
    # ---- the empty rocky throne on the dais
    tx = 56
    # steps up to the throne
    for i, (sw, sy) in enumerate(((34, 84), (28, 81), (22, 78))):
        cv.rect(tx - sw // 2, sy, sw, 3, '#3a2c32')
        cv.hline(tx - sw // 2, tx + sw // 2 - 1, sy, '#7a4a3a')
        cv.hline(tx - sw // 2, tx + sw // 2 - 1, sy + 2, '#1a151d')
    # back slab with three pointed spires
    back = poly_mask(h, w, [(tx - 9, 78), (tx - 9, 52), (tx - 7, 46), (tx - 5, 52), (tx - 3, 44),
                            (tx, 38), (tx + 3, 44), (tx + 5, 52), (tx + 7, 46), (tx + 9, 52),
                            (tx + 9, 78)])
    basalt_columns(cv, back, 103, 40, 90, base=0.25, cw=(3, 3), joint=(5, 9), cap=False)
    bo = outline_of(back)
    cv.a[bo] = C('#120e15')
    yy, xx = np.mgrid[0:h, 0:w]
    cv.a[bo & (xx > tx)] = C('#a8552e')            # lava rim light (right, toward the pool)
    # seat + armrests
    cv.rect(tx - 12, 68, 4, 10, '#2a2129')
    cv.rect(tx + 8, 68, 4, 10, '#2a2129')
    cv.hline(tx - 12, tx - 9, 68, '#7a4a3a')
    cv.hline(tx + 8, tx + 11, 68, '#b0603a')
    cv.vline(tx + 11, 69, 77, '#8a4a30')
    cv.rect(tx - 8, 71, 16, 7, '#2e242c')
    cv.hline(tx - 8, tx + 7, 71, '#8a5240')
    cv.hline(tx - 8, tx + 7, 77, '#120e15')
    # emblem: a flame-ruby set in a gold sun ring on the backrest
    cv.circle(tx, 55, 4, 'brass')
    cv.circle(tx, 55, 3, '#7a2a2a')
    for (dx, dy) in ((0, -5), (0, 5), (-5, 0), (5, 0)):
        cv.px(tx + dx, 55 + dy, 'gold')
    cv.tri((tx, 51), (tx - 2, 57), (tx + 2, 57), 'red')
    cv.px(tx, 53, 'lava_hi')
    cv.px(tx - 1, 56, 'crimson')
    # braziers (stone bowls on posts) either side
    for bx in (tx - 22, tx + 22):
        cv.rect(bx - 1, 72, 3, 12, '#2a2129')
        cv.vline(bx + 1, 72, 83, '#6c3a2e')
        cv.rect(bx - 3, 70, 7, 2, '#3a2c32')
        cv.hline(bx - 3, bx + 3, 72, '#1a151d')
    # ---- gem hoards on the right terraces
    R = rng(105)
    piles = [(278, 108, 28, 16), (330, 96, 20, 11), (247, 120, 11, 6), (360, 96, 12, 7)]
    gemcols = [('crimson', 'red', 'salmon'), ('blue', 'crystal', 'crystal_lt'), ('green', 'mint', 'ice'),
               ('purple', 'lavender', 'lilac'), ('#c27a1a', 'gold', 'butter')]
    gold = ['#7a4a1a', '#a8641c', 'amber', 'gold', '#ffe27a']
    for (px_, py_, rw, rh) in piles:
        bl = [(px_, py_, rw, rh, 0.0)]
        for i in range(4):
            bl.append((px_ + (R.random() - 0.5) * rw, py_ - R.random() * rh * 0.4, rw * 0.45, rh * 0.5, 1.0))
        sub = new(w, h)
        res = blobs(sub, bl, gold, outline='#5a3214', light=(0.5, 0.3, 0.6), dith=0.9)
        x0_, y0_, m = res
        m = m & (np.arange(y0_, y0_ + m.shape[0])[:, None] < py_ + 1)
        reg = cv.a[y0_:y0_ + m.shape[0], x0_:x0_ + m.shape[1]]
        reg[m] = sub.a[y0_:y0_ + m.shape[0], x0_:x0_ + m.shape[1]][m]
        # coin texture
        for i in range(int(rw * rh * 0.5)):
            gx = int(px_ + (R.random() - 0.5) * 2 * rw)
            gy = int(py_ - R.random() * rh)
            if 0 <= gx < w and m.shape[0] > gy - y0_ >= 0 and m.shape[1] > gx - x0_ >= 0 and m[gy - y0_, gx - x0_]:
                cv.px(gx, gy, '#ffe27a' if R.random() < 0.5 else '#a8641c')
        # gems: little faceted diamonds
        for i in range(int(rw * 0.9)):
            gx = int(px_ + (R.random() - 0.5) * 1.7 * rw)
            gy = int(py_ - R.random() * rh * 0.9)
            if not (m.shape[0] > gy - y0_ >= 0 and m.shape[1] > gx - x0_ >= 0 and m[gy - y0_, gx - x0_]):
                continue
            d, mcol, l = gemcols[int(R.random() * 5)]
            if R.random() < 0.5:
                cv.px(gx, gy - 1, l)
                cv.hline(gx - 1, gx + 1, gy, mcol)
                cv.px(gx, gy + 1, d)
                cv.px(gx + 1, gy, d)
            else:
                cv.rect(gx, gy, 2, 2, mcol)
                cv.px(gx, gy, l)
                cv.px(gx + 1, gy + 1, d)
    # treasure on the pile: a chalice and a crown (objects, no faces)
    cx_, cy_ = 268, 93
    cv.rect(cx_ - 2, cy_, 5, 3, 'gold')
    cv.hline(cx_ - 2, cx_ + 2, cy_, 'butter')
    cv.vline(cx_, cy_ + 3, cy_ + 5, 'amber')
    cv.hline(cx_ - 1, cx_ + 1, cy_ + 6, 'amber')
    cv.px(cx_, cy_ + 1, 'red')
    kx, ky = 287, 98
    cv.rect(kx - 3, ky, 7, 3, 'gold')
    for dx in (-3, 0, 3):
        cv.px(kx + dx, ky - 1, 'gold')
        cv.px(kx + dx, ky - 2, 'butter')
    cv.px(kx, ky + 1, 'blue')
    cv.px(kx - 2, ky + 1, 'red')
    cv.px(kx + 2, ky + 1, 'green')
    cv.hline(kx - 3, kx + 3, ky + 2, '#a8641c')
    # crystal clusters growing from the rock (crimson / violet shards)
    for (gx0, gy0, n, pal) in ((112, 94, 4, ('#5a1a2a', 'crimson', 'salmon', 'blush')),
                               (6, 96, 3, ('#3a2050', 'purple', 'lavender', 'lilac')),
                               (232, 126, 3, ('#5a1a2a', 'crimson', 'salmon', 'blush')),
                               (372, 96, 3, ('#3a2050', 'purple', 'lavender', 'lilac')),
                               (90, 84, 3, ('#3a2050', 'purple', 'lavender', 'lilac'))):
        for k in range(n):
            hh = 4 + (k * 7 + gx0) % 6
            xk = gx0 + k * 3 - n
            lean_ = (k - n // 2) * 0.5
            pts = [(xk - 1, gy0), (xk + 2, gy0), (xk + 1 + lean_, gy0 - hh), (xk + lean_, gy0 - hh - 1)]
            cv.poly(pts, pal[1])
            cv.line(xk, gy0 - 1, xk + lean_, gy0 - hh, pal[2])
            cv.px(int(xk + lean_), gy0 - hh, pal[3])
            cv.px(xk + 1, gy0 - 1, pal[0])
    # geyser vent on the left lower step
    cv.rect(146, 116, 6, 3, '#4c3131')
    cv.hline(147, 150, 116, '#c4602c')
    return to_layer(cv), piles


@lru_cache(maxsize=1)
def _lava_tex():
    h, w = RIV1 - RIV0, 512
    n1 = fbm(h, w, 3, 26, 111, octaves=3)
    n2 = fbm(h, w, 2, 9, 113, octaves=2)
    return n1, n2


@lru_cache(maxsize=1)
def _crusts():
    """Crust plates: small dark polygons (RGBA) that drift on the lava."""
    out = []
    R = rng(121)
    for i in range(10):
        w = int(6 + R.random() * 10)
        h = int(2 + R.random() * 2)
        cv = new(w + 2, h + 2)
        cv.round_rect(1, 1, w, h, '#2a1a1c', r=1)
        cv.hline(2, w - 1, 1, '#4c2a24')
        for k in range(int(w / 4)):
            x = 2 + int(R.random() * (w - 2))
            cv.px(x, 1 + int(R.random() * h), 'lava_dk')
        cv.frame(0, 0, w + 2, h + 2, KEY)
        cv.round_rect(0, 1, w + 2, h, KEY, r=0) if False else None
        out.append(to_layer(cv))
    return out


def draw_lava_river(c, f, ox, oy, swell):
    """Flowing lava bands (scrolling noise), crust plates, bubbles; the hot pool."""
    n1, n2 = _lava_tex()
    h = RIV1 - RIV0
    ys = np.arange(h)
    # flow speed: faster mid-stream, slower at the banks
    sp = 0.35 + 0.45 * np.sin(np.pi * (ys + 0.5) / h)
    xw = np.arange(W)[None, :] - ox
    u = (xw - (f * sp)[:, None]).astype(int) % n1.shape[1]
    u2 = (xw - (f * sp * 1.6)[:, None] + 7).astype(int) % n2.shape[1]
    v = n1[ys[:, None], u] * 0.7 + n2[ys[:, None], u2] * 0.3
    # hot pool: brighter round zone around POOL
    yw = (ys + RIV0)[:, None]
    d = np.sqrt(((xw - POOL[0]) / 24.0) ** 2 + ((yw - POOL[1]) / 7.0) ** 2)
    v = v + np.clip(1 - d, 0, 1) * (0.25 + 0.35 * swell)
    # cooler near the banks
    edge = np.minimum(ys, h - 1 - ys)[:, None]
    v = v - np.clip((3 - edge) / 3, 0, 1) * 0.25
    lv = np.clip(((v - 0.30) / 0.55) * len(LAVA_CUT) + (bayer(h, W, 0, f // 3 % 2) - 0.5) * 0.9,
                 0, len(LAVA_CUT) - 1).astype(int)
    ra = ramp_arr(LAVA_CUT)
    y0 = RIV0 + oy
    Y0, Y1 = max(0, y0), min(H, y0 + h)
    c.a[Y0:Y1] = ra[lv[Y0 - y0:Y1 - y0]]
    # crust plates drifting
    for i, lay in enumerate(_crusts()):
        yr = int(2 + hash01(i, 1) * (h - 6))
        spd = 0.35 + 0.45 * math.sin(math.pi * (yr + 2) / h)
        x = (hash01(i, 2) * 420 + f * spd) % 420 - 40 + ox
        comp(c, lay, int(x), RIV0 + yr + oy)
    # bubbles: grow, pop
    nb = 14 + int(12 * swell)
    for i in range(nb):
        per = 30 + int(hash01(i, 5) * 30)
        if swell > 0.3:
            per = max(12, int(per * (1 - swell * 0.6)))
        t = ((f + hash01(i, 6) * per) % per) / per
        near_pool = i % 3 == 0
        if near_pool:
            bx = POOL[0] + (hash01(i, 7) - 0.5) * 40 + ox
            by = POOL[1] + (hash01(i, 8) - 0.5) * 8 + oy
        else:
            bx = hash01(i, 7) * 360 + ox
            by = RIV0 + 3 + hash01(i, 8) * (h - 6) + oy
        bx, by = int(bx), int(by)
        if t < 0.75:
            r = 0.5 + 2.5 * t / 0.75
            c.ellipse(bx, by, r + 0.5, r * 0.7, 'lava')
            c.ellipse(bx, by - 1, max(0.5, r - 0.5), max(0.5, r * 0.5), 'lava_hi')
            c.px(bx - 1, by - int(r * 0.6), 'butter')
        elif t < 0.9:
            k = int((t - 0.75) / 0.15 * 4)
            for q in range(4):
                c.px(bx - 2 + q + (q > 1), by - 1 - k - (q % 2), 'lava_hi' if q % 2 else 'lava')
            c.hline(bx - 2, bx + 2, by, 'lava_dk')


def draw_lavafall(c, f, x, y0, y1):
    """Animated lavafall: pours over a lip, widens and wobbles, splashes into the river."""
    hgt = y1 - y0
    for yy in range(y0, y1):
        t = (yy - y0) / max(1, hgt)
        hw = 2 + int(t * 3.5)
        wob = int(round(math.sin(yy * 0.18 + f * 0.25) * (0.4 + t)))
        xc = x + wob + int(t * t * 3)
        for dx in range(-hw, hw + 1):
            ph = (yy * 0.45 - f * 1.6 + dx * 2.3 + (yy // 5) * 0.9) % 7
            k = abs(dx) / max(1, hw)
            if k > 0.85:
                cc = 'lava_dk' if (yy + dx) % 3 else '#c22e16'
            elif ph < 1.3:
                cc = 'butter' if k < 0.4 else 'lava_hi'
            elif ph < 4:
                cc = 'lava_hi' if k < 0.5 else 'lava'
            else:
                cc = 'lava' if k < 0.6 else '#c22e16'
            c.px(xc + dx, yy, cc)
    # pour-over lip
    c.hline(x - 3, x + 3, y0 - 1, 'lava')
    c.hline(x - 2, x + 2, y0 - 2, 'lava_hi')
    # splash + glow at the foot
    xb = x + 3
    for q in range(10):
        t = ((f + q * 3) % 12) / 12
        dx = (q - 4.5) * 1.6 * (0.5 + t)
        c.px(int(xb + dx), int(y1 - 1 - math.sin(t * math.pi) * (3 + q % 3)), 'lava_hi' if q % 2 else 'amber')
    c.ellipse(xb, y1 + 1, 9, 2, 'lava_hi')
    c.hline(xb - 5, xb + 5, y1 + 1, 'butter')


def draw_brazier_flame(c, f, x, y, seed):
    k = (f // 3 + seed) % 4
    h = 4 + (k % 2)
    c.px(x, y, 'lava')
    c.hline(x - 1, x + 1, y, 'lava')
    c.vline(x, y - h + 1, y, 'amber')
    c.px(x + (1 if k > 1 else -1), y - 2, 'lava')
    c.px(x, y - 1, 'lava_hi')
    c.px(x, y - h, 'butter' if k % 2 else 'lava_hi')


@lru_cache(maxsize=1)
def _fg_b():
    """Near crater walls (left & right) and the foreground rock shelf (par 1.3)."""
    w, h = BW + 30, H
    cv = new(w, h)
    xs = np.arange(w)
    left = poly_mask(h, w, [(0, 0), (92, 0), (84, 4), (70, 8), (58, 12), (48, 18), (42, 28), (36, 42),
                            (40, 64), (34, 92), (44, 120), (40, 140), (0, 140)])
    right = poly_mask(h, w, [(w, 0), (296, 0), (308, 5), (320, 10), (330, 18), (336, 30), (340, 44),
                             (334, 70), (342, 96), (334, 124), (340, 140), (w, 140)])
    for (sx_, sy_, sl) in ((60, 11, 6), (52, 15, 4), (72, 7, 5), (318, 9, 5), (328, 15, 7), (308, 4, 4)):
        left |= poly_mask(h, w, [(sx_ - 2, sy_), (sx_ + 2, sy_), (sx_, sy_ + sl)]) if sx_ < 200 else False
        right |= poly_mask(h, w, [(sx_ - 2, sy_), (sx_ + 2, sy_), (sx_, sy_ + sl)]) if sx_ >= 200 else False
    yy, xx = np.mgrid[0:h, 0:w]
    shelf_top = 160 + 3 * np.sin(xs * 0.09) + 2 * np.sin(xs * 0.23 + 1) + (vnoise(1, w, 1, 7, 131)[0] - 0.5) * 6
    shelf = yy >= shelf_top[None, :].astype(int)
    # stalagmite-like column stumps on the shelf
    R = rng(133)
    for i in range(16):
        x = int(R.random() * w)
        hh = int(4 + R.random() * 12)
        ww = int(3 + R.random() * 4)
        st = poly_mask(h, w, [(x - ww, 172), (x - 1, 168 - hh), (x + 1, 168 - hh), (x + ww, 172)])
        shelf |= st
    m = left | right | shelf
    basalt_columns(cv, left | right, 135, 60, 150, base=0.0, cw=(5, 8), joint=(8, 16), cap=False)
    nz = fbm(h, w, 4, 6, 137)
    t = np.clip(0.1 + (nz - 0.5) * 0.4, 0, 0.999)
    ra = ramp_arr(BAS)
    lv = ramp_idx(t, len(BAS), 0, 0, h, w, 0.8)
    cv.a[shelf] = ra[lv][shelf]
    # lava rim light on edges that face the river
    bd = outline_of(m)
    top_edge = bd & shelf & ~np.vstack([shelf[1:], np.zeros((1, w), bool)])
    cv.a[bd] = C('#0e0b11')
    up = shelf & ~np.vstack([np.zeros((1, w), bool), shelf[:-1]])
    cv.a[up] = C('#c4602c')
    up2 = np.vstack([np.zeros((1, w), bool), up[:-1]]) & shelf
    cv.a[up2] = C('#6c3a2e')
    redge_l = left & ~np.hstack([left[:, 1:], np.zeros((h, 1), bool)])
    cv.a[redge_l & (yy > 50)] = C('#97492a')
    ledge_r = right & ~np.hstack([np.zeros((h, 1), bool), right[:, :-1]])
    cv.a[ledge_r & (yy > 50)] = C('#97492a')
    # vents in the shelf
    vents = [(150, 0), (276, 1), (60, 2)]
    for (vx, _) in vents:
        top = int(shelf_top[vx])
        cv.rect(vx - 2, top, 5, 2, '#0e0b11')
        cv.hline(vx - 3, vx + 3, top - 1, '#6c3a2e')
    return to_layer(cv), vents, shelf_top.astype(int)


def draw_fountain(c, f, x, y):
    """ERUPT: lava fountain from (x, y) at f >= ERUPT: column, mushrooming crown, blobs."""
    e = f - ERUPT
    if e < 0:
        return
    if e < 5:
        hc = 128 * ease_out(e / 5)
    elif e < 16:
        hc = 128 - (e - 5) * 1.2
    else:
        hc = 115 * max(0.0, 1 - (e - 16) / 20) ** 1.3
    wbase = 9 if e < 20 else max(3, 9 - (e - 20) // 3)
    if hc > 1:
        top = int(y - hc)
        for yy in range(max(0, top), y + 1):
            t = (y - yy) / max(1, hc)
            ww = int(wbase * (1 - 0.35 * t) + 3 * max(0, t - 0.8) / 0.2)
            ww += 1 if ((yy + e * 4) // 3) % 3 == 0 else 0
            wob = int(round(math.sin(yy * 0.25 + e * 0.8) * (1 + 1.5 * t)))
            xc = x + wob
            c.hline(xc - ww - 1, xc + ww + 1, yy, 'lava_dk')
            c.hline(xc - ww, xc + ww, yy, 'lava')
            c.hline(xc - max(0, ww - 2), xc + max(0, ww - 2), yy, 'lava_hi')
            stripe = ((yy + e * 5) // 2) % 4
            c.hline(xc - max(0, ww - 5), xc + max(0, ww - 5), yy, 'butter' if stripe else 'lava_hi')
        # rounded head of the column
        wt = max(2, int(wbase * 0.65))
        c.ellipse(x, top + 1, wt + 1, 2, 'lava')
        c.ellipse(x, top + 1, max(1, wt - 1), 1, 'lava_hi')
        # mushrooming crown: lumpy lava blobs (collapses with the column)
        for q in range(9 if e < 22 else 0):
            a = q / 9 * 2 * math.pi + e * 0.2
            rr = (4 + (q % 3) * 2 + min(6, e // 2)) * (1 if e < 14 else max(0.2, 1 - (e - 14) / 8))
            bx = x + math.cos(a) * (6 + min(10, e)) * 1.3
            by = top + 3 + math.sin(a) * (3 + min(5, e // 2))
            c.ellipse(bx, by, rr * 0.7 + 1, rr * 0.5 + 1, 'lava_dk')
            c.ellipse(bx, by, rr * 0.7, rr * 0.5, 'lava')
            c.ellipse(bx - 1, by - 1, max(1, rr * 0.4), max(1, rr * 0.3), 'lava_hi')
    # ballistic lava blobs flung out in arcs
    for i in range(70):
        t0 = int(hash01(i, 1) * 16)
        dt = e - t0
        if dt < 0 or dt > 50:
            continue
        a = -math.pi / 2 + (hash01(i, 2) - 0.5) * 2.6
        v = 2.6 + hash01(i, 3) * 3.4
        h0 = min(hc, 110) * (0.4 + 0.6 * hash01(i, 4))
        bx = x + math.cos(a) * v * dt
        by = y - h0 + math.sin(a) * v * dt + 0.14 * dt * dt
        if by > RIV1 + 12 or by < -4:
            continue
        r = 1 + (i % 3 == 0) + (i % 7 == 0)
        bx, by = int(bx), int(by)
        # streak (previous position) then the hot blob
        vx_ = math.cos(a) * v
        vy_ = math.sin(a) * v + 0.28 * dt
        c.line(bx, by, bx - vx_ * 1.2, by - vy_ * 1.2, '#c22e16')
        c.rect(bx - r // 2, by - r // 2, r + 1, r + 1, 'lava')
        c.px(bx, by, 'butter' if dt < 12 else ('lava_hi' if dt < 30 else 'amber'))
    # splash ring at the pool
    if e < 26:
        for ring in (0, 1):
            r = 8 + e * (1.6 - ring * 0.6)
            for q in range(24):
                a = q / 24 * 2 * math.pi
                c.px(int(x + math.cos(a) * r), int(y + 1 + math.sin(a) * r * 0.22),
                     'lava_hi' if (q + ring) % 2 else 'lava')


def render_b(f):
    cam = cam_b(f)
    sx, sy = shake_b(f)
    e = f - ERUPT
    swell = seg(f, ERUPT - 20, ERUPT) ** 2 if f < ERUPT else max(0.0, 1 - seg(f, ERUPT, ERUPT + 20))
    c = _sky_b().copy()
    # smoke rising out of the caldera (behind the far rim)
    draw_plume(c, f, -int(cam * 0.1) + sx, x0=150, y0=60, n=18, period=240, rise=70, drift=60, rmax=16,
               seed=3, hot=True)
    if e >= 0:
        draw_plume(c, f - ERUPT, -int(cam * 0.1) + sx, x0=POOL[0] - 30, y0=50, n=10, period=120, rise=60,
                   drift=40, rmax=18, seed=4, hot=True)
    wl, veins, rim, LF = _wall_b()
    wox = -int(cam * 0.3) - 20 + sx
    comp(c, wl, wox, sy)
    # vein pulses
    for j, pts in enumerate(veins):
        n = len(pts)
        for i, (vx, vy) in enumerate(pts):
            ph = (i / n * 2 - f * 0.05 + j * 0.3) % 1.0
            if ph < 0.35:
                c.px(vx + wox, vy + sy, 'lava' if ph > 0.12 else 'lava_hi')
    draw_lavafall(c, f, LF + wox, rim[LF] + 6 + sy, RIV0 + 2 + sy)
    fx.glow(c, LF + wox, RIV0 + sy, 26, 'lava', 0.25, rings=4)
    # mid ledges
    (ml, piles) = _mid_b()
    mox = -int(cam * 0.6) - 6 + sx
    comp(c, ml, mox, sy)
    for bx in (56 - 22, 56 + 22):
        draw_brazier_flame(c, f, bx + mox, 69 + sy, bx)
    fx.glow(c, 56 + mox, 68 + sy, 30, 'amber', 0.12, rings=3)
    # gem sparkles
    for i in range(14):
        (px_, py_, rw, rh) = piles[i % len(piles)]
        gx = int(px_ + (hash01(i, 21) - 0.5) * 1.6 * rw) + mox
        gy = int(py_ - hash01(i, 22) * rh * 0.8) + sy
        fx.sparkle(c, gx, gy, f, ('white', 'butter', 'crystal_lt', 'lilac')[i % 4], period=28 + i % 5 * 4,
                   phase=hash01(i, 23), size=2 if i % 3 else 3)
    # geyser on the left step: bursts on beats 1 and 3 into a broad fan of hot spray + a steam cloud
    gx, gy = 149 + mox, 116 + sy
    bp = f % 36
    if bp < 24:
        jh = int(24 * ease_out(min(1, bp / 4))) if bp < 8 else int(24 * max(0, 1 - (bp - 8) / 16) ** 1.4)
        for yy in range(gy - jh, gy):
            t = (gy - yy) / max(1, jh)
            ww = 1 + int(t * t * 6)                 # fans out into spray toward the top
            wob = int(round(math.sin(yy * 0.6 + f * 0.9) * 0.6))
            for x in range(gx - ww + wob, gx + ww + wob + 1):
                if t < 0.6 or hash01(x, yy, f // 2) < 1.3 - t:
                    c.px(x, yy, '#f4c6a4' if (yy + x + f) % 3 else '#e0a184')
            c.vline(gx + wob, yy, yy, 'cream')
        for q in range(18):
            tt = ((bp + q * 1.3) % 14) / 14
            side = 1 if q % 2 else -1
            sx_ = gx + side * (2 + tt * (6 + q % 5 * 2))
            sy_ = gy - jh + tt * tt * (jh + 4) - math.sin(tt * math.pi) * 4
            if sy_ < gy:
                c.px(int(sx_), int(sy_), 'cream' if q % 3 else '#e0a184')
    steam_puffs(c, f - 6, gx + 6, gy - 30, period=36, n=3, height=34, rmax=10, seed=11, drift=16, tone=6)
    # lava river
    rox = -cam + sx
    draw_lava_river(c, f, rox, sy, swell)
    # pool swell dome before the eruption
    if f < ERUPT and swell > 0.05:
        dh = int(1 + swell * 5)
        c.ellipse(POOL[0] + rox, POOL[1] + sy - dh // 2, 8 + dh, dh, 'lava_hi')
        c.ellipse(POOL[0] + rox, POOL[1] + sy - dh // 2 - 1, 4 + dh, max(1, dh - 1), 'butter')
    # heat shimmer over the lava (rows 96..129)
    y0s, y1s = 96, RIV0
    rows = np.arange(y0s, y1s)
    shift = np.round(np.sin(rows * 0.7 + f * 0.35) * 0.6 * (rows - y0s) / (y1s - y0s) * 1.6).astype(int)
    xi = np.clip(np.arange(W)[None, :] - shift[:, None], 0, W - 1)
    c.a[y0s:y1s] = c.a[rows[:, None], xi]
    # uplight on everything from the river (warm glow)
    fx.light_rect(c, 0, 92, W, RIV0 - 92, 'lava', 0.10 + 0.08 * swell, fade='up')
    # eruption
    if e >= 0:
        draw_fountain(c, f, POOL[0] + rox, POOL[1] + sy)
        if e < 30:
            fx.glow(c, POOL[0] + rox, POOL[1] + sy - 40, 90, 'lava', 0.35 * (1 - e / 30), rings=5)
    # foreground walls + shelf
    fl, vents, stop = _fg_b()
    fox = -int(cam * 1.3) - 8 + sx
    comp(c, fl, fox, sy)
    # steam vents puffing on the beats
    for k, (vx, _) in enumerate(vents):
        top = stop[vx] + sy
        burst = ((f - k * 18) % 36) < 12          # each vent hisses on its own beat
        fx.steam(c, vx + fox, top - 1, f + k * 5, color='#f4d6c0', n=7 if burst else 3,
                 height=26 if burst else 14, spread=3, seed=k, speed=0.6 if burst else 0.3)
        fx.steam(c, vx + fox + 1, top - 1, f + k * 7 + 3, color='#e0a184', n=4 if burst else 2,
                 height=18, spread=2, seed=k + 9, speed=0.45)
    # embers everywhere (more after the eruption)
    ne = 34 if e < 0 else 90
    for i in range(ne):
        per = 80 + int(hash01(i, 3) * 60)
        t = ((f + hash01(i, 1) * per) % per) / per
        x = hash01(i, 2) * W + math.sin(t * 6 + i) * 4 - cam * 0.6
        if e >= 0 and i >= 34:
            # ember shower raining down after the burst
            dt = e - int(hash01(i, 9) * 10)
            if dt < 0 or dt > 60:
                continue
            x = POOL[0] + rox + (hash01(i, 2) - 0.5) * 300 * min(1, dt / 12)
            y = 10 + dt * (1.2 + hash01(i, 4)) + math.sin(dt * 0.3 + i) * 2 - 40 * max(0, 1 - dt / 8)
        else:
            y = RIV0 - t * 130
        x = int(x) % W
        if hash01(i, f // 4) < 0.85:
            ember(c, x, int(y), f, i)
    # warm grade; eruption flash
    fx.grade(c, (1.0, 0.95, 0.92), (8 + int(18 * swell), 0, 0))
    if 0 <= e < 6:
        k = (1 - e / 6) ** 1.5
        fx.grade(c, (1 + 0.5 * k, 1 + 0.35 * k, 1 + 0.1 * k), (150 * k, 110 * k, 50 * k))
    callout(c, 276 + mox, 96 + sy, 'GEM HOARDS', f, 198, 263, side='up', length=14)
    return c


# ================================================================== SHOT C: Kirin Grove
C0, C1 = 270, 360
LAG0, LAG1 = 106, 122           # Lost Lagoon rows
BANK = 118                      # grove bank top (approx)
SKY_C = ['#6aa2bd', '#7cb2c6', '#93c3cc', '#accfcf', '#c6dccb', '#dde4c6', '#efe2bd', '#f6d9b2']
KARST_FAR = ['#86a9a6', '#98b8b1', '#abc6bb', '#c0d5c7', '#d5e4d3']
KARST_MID = ['#3d5f5a', '#4f7468', '#678c7a', '#84a88f', '#a6c4a6', '#c8dcb8']
KARST_LIM = ['#5b6f6a', '#71877e', '#8ea292', '#adbfa8', '#cfdcc0']
JADE = ['#0f3d33', 'forest', 'green', 'mint', '#6fd078', 'leaf']
WATER = ['#0b6a6c', 'teal', 'aqua', 'turquoise', 'ice', 'white']


def cam_c(f):
    return cam_ease(f, C0, C1 - 1, 0, 30)


def karst(cv, cx, yt, yb, w, seed, ramp, veg=None, vines=0, rim=None, out=None, lean=0.0):
    """A karst tower: rounded top, bulging/undercut sides, vertical limestone streaks, green caps."""
    h, wd = cv.h, cv.w
    ys = np.arange(h)
    nz = vnoise(h, 1, 6, 1, seed)[:, 0]
    hw = np.zeros(h)
    span = max(1, yb - yt)
    for y in range(max(0, yt), min(h, yb)):
        d = y - yt
        top = min(1.0, math.sqrt((d + 0.5) / max(1.0, w * 0.42)))
        tt = d / span
        bulge = 1 + 0.16 * math.sin(tt * math.pi * 1.4 + seed) - 0.08 * tt
        hw[y] = w / 2 * top * bulge * (0.86 + 0.28 * nz[y])
    xx = np.arange(wd)[None, :].astype(np.float32)
    cxy = cx + lean * (ys - yt)[:, None]
    m = (np.abs(xx + 0.5 - cxy) <= hw[:, None]) & (ys[:, None] >= yt) & (ys[:, None] < yb)
    u = (xx + 0.5 - cxy) / np.maximum(hw[:, None], 0.5)
    streak = vnoise(h, wd, 16, 1.4, seed + 1)
    ledge = vnoise(h, wd, 3, 12, seed + 2)
    t = 0.62 - u * 0.38 + (streak - 0.5) * 0.4 - (ledge > 0.72) * 0.18
    ra = ramp_arr(ramp)
    lv = ramp_idx(np.clip(t, 0, 0.999), len(ramp), 0, 0, h, wd, 0.8)
    cv.a[m] = ra[lv][m]
    bd = outline_of(m)
    if out is not None:
        cv.a[bd & (u > -0.2)] = C(out)
    if rim is not None:
        cv.a[bd & (u <= -0.2)] = C(rim)
    if veg is not None:
        R = rng(seed + 5)
        bl = []
        # cap
        for i in range(5):
            bx = cx + lean * 2 + (R.random() - 0.5) * w * 0.45
            bl.append((bx, yt + 1 + R.random() * 4, 2 + w * 0.12 + R.random() * w * 0.06,
                       3 + R.random() * 2.5, R.random()))
        # ledges down the sides
        for i in range(int(span / 12)):
            yy = yt + 10 + R.random() * (span - 16)
            side = -1 if R.random() < 0.6 else 1
            yi = int(min(h - 1, yy))
            bx = cx + lean * (yy - yt) + side * hw[yi] * (0.7 + R.random() * 0.3)
            bl.append((bx, yy, 2 + R.random() * w * 0.12, 1.5 + R.random() * 1.5, R.random()))
        if bl:
            blobs(cv, bl, veg, outline=mix(veg[0], 'ink', 0.3), light=(-0.75, -0.55, 0.4), dith=0.7)
        for i in range(vines):
            vx = int(cx + (R.random() - 0.5) * w * 0.8)
            col_ = np.nonzero(m[:, min(wd - 1, max(0, vx))])[0]
            if len(col_) == 0:
                continue
            vy = col_[0] + 3
            L = int(4 + R.random() * span * 0.3)
            for k in range(L):
                if vy + k < h and m[vy + k, vx]:
                    cv.px(vx, vy + k, veg[1] if k % 4 else veg[2])
    return m


@lru_cache(maxsize=1)
def _sky_c():
    c = Canvas()
    c.gradient(0, 0, W, LAG0 + 2, SKY_C, soft=0.7)
    fx.glow(c, 34, 30, 80, 'peach', 0.16, rings=6)
    fx.glow(c, 34, 30, 30, 'cream', 0.25, rings=4)
    disc(c, 34, 30, 7, '#fff0cf')
    disc(c, 33, 29, 5, 'cream')
    # the volcano far to the north (left), still smoking: continuity with bar 2/3
    vz = mix('#b48a90', SKY_C[6], 0.35)
    c.poly([(60, LAG0), (88, 78), (93, 76), (99, 78), (126, LAG0)], vz)
    c.hline(90, 96, 77, '#f08a5a')
    c.px(93, 76, '#ffb070')
    for i in range(9, -1, -1):
        r = 2 + int(i * 0.9)
        draw_puff(c, 95 + int(i * 6.5), 73 - int(i * 3.2 - i * i * 0.12), r, 5, seed=i)
    return c


@lru_cache(maxsize=1)
def _far_karsts():
    w = 360
    cv = new(w, H)
    R = rng(201)
    x = 4
    i = 0
    while x < w:
        ww = 10 + R.random() * 16
        yt = int(40 + R.random() * 34)
        karst(cv, x + ww / 2, yt, LAG0 + 2, ww, 210 + i, KARST_FAR, veg=None,
              rim='#e2eedb', out=None)
        x += ww * (0.7 + R.random() * 0.6)
        i += 1
    # haze toward the base
    a, m = to_layer(cv)
    yy = np.arange(H)[:, None, None]
    k = np.clip((yy - 70) / 40, 0, 0.75)
    v = a.astype(np.float32) * (1 - k) + np.array(col('#d9e8dc'), np.float32) * k
    cv.a[:] = np.where(m[:, :, None], v.astype(np.uint8), a)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _lagoon():
    """Lost Lagoon water band + rocky islets (par 0.2), RGBA layer."""
    w = 360
    cv = new(w, H)
    cv.gradient(0, LAG0, w, LAG1 - LAG0, ['#bfe6dc', '#8fdccb', '#5fcab8', '#3fb2a6'], soft=0.6)
    cv.hline(0, w - 1, LAG0, '#e8f6ee')
    R = rng(221)
    isl = []
    x = 6
    while x < w:
        ww = 6 + R.random() * 14
        hh = int(5 + R.random() * 12)
        base = int(LAG0 + 4 + R.random() * 8)
        karst(cv, x, base - hh, base + 1, ww, 230 + int(x), KARST_LIM, veg=JADE[1:5], rim='#e2ecd2',
              out='#4c5e58')
        # reflection
        for k in range(1, min(6, hh)):
            cv.hline(int(x - ww * 0.3), int(x + ww * 0.3), base + k, '#4d9c94' if k % 2 else '#62b4aa')
        cv.hline(int(x - ww * 0.55), int(x + ww * 0.55), base + 1, '#e8f6ee')
        isl.append((x, base))
        x += 18 + R.random() * 40
    return to_layer(cv), isl


@lru_cache(maxsize=1)
def _mid_karsts():
    w = 380
    cv = new(w, H)
    specs = [(-2, 20, 30, 301, 0.04), (30, 8, 34, 302, -0.02), (64, 34, 22, 303, 0.0),
             (262, 22, 28, 304, 0.03), (300, 4, 38, 305, -0.03), (342, 30, 30, 306, 0.0),
             (232, 52, 16, 307, 0.0)]
    for (cx, yt, ww, sd, ln) in specs:
        karst(cv, cx, yt, 124, ww, sd, KARST_MID, veg=JADE, vines=5, rim='#d8e8c4', out='#2c4844', lean=ln)
    # mist band at the feet
    a, m = to_layer(cv)
    yy = np.arange(H)[:, None, None]
    k = np.clip((yy - 88) / 34, 0, 0.7)
    v = a.astype(np.float32) * (1 - k) + np.array(col('#d6eadf'), np.float32) * k
    cv.a[:] = np.where(m[:, :, None], v.astype(np.uint8), a)
    return to_layer(cv)


def draw_ribbon_fall(c, f, x, y0, y1):
    """Thin ribbon waterfall off a karst: white streaks sliding down."""
    for yy in range(max(0, y0), min(H, y1)):
        t = (yy - y0) / max(1, y1 - y0)
        ww = 1 + int(t * 1.6)
        xc = x + int(t * t * 2)
        for dx in range(-ww, ww + 1):
            ph = (yy - f * 2 + dx * 5) % 9
            cc = 'white' if ph < 3 else ('ice' if ph < 6 else '#9fe0d6')
            c.px(xc + dx, yy, cc)
    for q in range(3):
        draw_puff(c, x + 3 + q * 3 - 3, y1 - 2 - (f // 6 + q) % 3, 3 + q % 2, 4, seed=q)


# ---- grove bank (par 0.6): ground, pavilion, stream, bridge
GW = 400
STREAM_TOP = 122
POND_Y = 157


def stream_cx(y):
    return 112 + (y - STREAM_TOP) * 0.55


def stream_hw(y):
    return 3 + (y - STREAM_TOP) * 0.36


@lru_cache(maxsize=1)
def _bank():
    w, h = GW, H
    cv = new(w, h)
    xs = np.arange(w)
    top = (BANK + 3 * np.sin(xs * 0.035 + 0.5) + 2 * np.sin(xs * 0.11) - 4 * np.exp(-((xs - 228) / 40.0) ** 2)
           ).astype(int)
    yy, xx = np.mgrid[0:h, 0:w]
    pl = (POND_Y + 1.5 * np.sin(xs * 0.13) + (vnoise(1, w, 1, 5, 405)[0] - 0.5) * 3).astype(int)
    m = (yy >= top[None, :]) & (yy < pl[None, :])
    nz = fbm(h, w, 3, 7, 401)
    t = 0.7 - (yy - BANK) * 0.006 + (nz - 0.5) * 0.7
    ra = ramp_arr(JADE)
    lv = ramp_idx(np.clip(t, 0, 0.999), len(JADE), 0, 0, h, w, 0.9)
    cv.a[m] = ra[lv][m]
    edge = m & ~np.vstack([np.zeros((1, w), bool), m[:-1]])
    cv.a[edge] = C('#cdeb8a')
    # bank shading: darker toward the pond lip, then a stony lip with grass overhang
    shade_ = m & (yy >= pl[None, :] - 10) & (bayer(h, w) < (yy - pl[None, :] + 10) / 14)
    cv.a[shade_] = C('forest')
    for x in range(w):
        y1 = pl[x]
        cv.vline(x, y1 - 3, y1 - 1, '#6c7a6a' if (x // 4 + x // 7) % 2 else '#8a978a')
        cv.px(x, y1 - 3, 'green' if hash01(x, 1) < 0.6 else 'mint')
        if hash01(x, 2) < 0.3:
            cv.px(x, y1 - 2, 'forest')
        cv.px(x, y1 - 1, '#4c5e58')
    # grass tufts
    R = rng(403)
    for i in range(240):
        x = int(R.random() * w)
        y = int(top[x] + 1 + R.random() * (h - top[x]))
        cv.px(x, y, 'leaf' if R.random() < 0.6 else '#0f3d33')
        cv.px(x, y - 1, 'pale_leaf' if R.random() < 0.5 else 'mint')
    # stepping-stone path from the bridge to the pavilion
    for i in range(7):
        px_ = 150 + i * 11
        py_ = 150 - i * 4
        cv.ellipse(px_, py_, 4, 1.5, '#b5b0a0')
        cv.hline(px_ - 3, px_ + 2, py_ - 1, '#e2dccb')
        cv.hline(px_ - 3, px_ + 3, py_ + 1, '#6c7a6a')
    # stream bed (water drawn per frame); rocks along the banks
    sm = np.zeros((h, w), bool)
    for y in range(STREAM_TOP, POND_Y + 1):
        cxs, hws = stream_cx(y), stream_hw(y)
        sm[y, int(cxs - hws):int(cxs + hws) + 1] = True
    for y in range(STREAM_TOP, POND_Y - 3, 2):
        for side in (-1, 1):
            x = int(stream_cx(y) + side * (stream_hw(y) + 1))
            r = 1 + (y - STREAM_TOP) // 18
            if hash01(y, side) < 0.55:
                cv.ellipse(x, y, r + 1, r, '#7a8a80')
                cv.hline(x - r, x + r - 1, y - r, '#c9d4c4')
    # flowering shrubs along the bank
    R2 = rng(407)
    shr = ['#1c4a30', '#2e7a3a', '#4f9a4a', '#78bc5c', '#a8dc80']
    for (x0_, x1_) in ((16, 92), (282, 396), (160, 190)):
        x = x0_
        while x < x1_:
            y = top[min(w - 1, int(x))] + 2
            r = 3 + R2.random() * 3
            blobs(cv, [(x, y, r * 1.3, r, 0), (x + r, y + 1, r, r * 0.8, 0.5)], shr, outline='#163c28',
                  light=(-0.7, -0.6, 0.4), dith=0.8)
            for k in range(3):
                cv.px(int(x + (R2.random() - 0.5) * r * 2), int(y - R2.random() * r * 0.8),
                      ('bubblegum', 'cotton', 'cream', 'lilac')[int(R2.random() * 4)])
            x += r * 2.2 + R2.random() * 6
    # flowers
    for i in range(70):
        x = int(R.random() * w)
        y = int(top[x] + 2 + R.random() * 50)
        if not sm[min(h - 1, y), x]:
            cv.px(x, y, ('bubblegum', 'cream', 'lavender', 'gold', 'cotton')[i % 5])
    return to_layer(cv), top, sm, pl


@lru_cache(maxsize=1)
def _pavilion():
    """The Kirin pavilion: stone plinth, red posts, two curved jade roofs, moon windows."""
    w, h = 76, 62
    cv = new(w, h)
    cx = w // 2
    # plinth
    cv.rect(6, 54, w - 12, 6, '#c9c2ae')
    cv.hline(6, w - 7, 54, '#efe8d4')
    cv.hline(6, w - 7, 59, '#8a8574')
    for x in range(8, w - 8, 6):
        cv.vline(x, 55, 58, '#a8a290')
    cv.rect(cx - 7, 56, 14, 6, '#b5ae9a')            # steps
    cv.hline(cx - 7, cx + 6, 58, '#efe8d4')
    cv.hline(cx - 7, cx + 6, 60, '#efe8d4')
    # lower walls: lattice panels + moon windows between red posts
    cv.rect(12, 34, w - 24, 20, '#e8dcc0')
    for x in range(12, w - 12, 2):
        cv.vline(x, 36, 53, '#d6c8a8')
    for (wx) in (cx - 16, cx + 16):
        cv.circle(wx, 43, 6, '#7a3a2a')
        cv.circle(wx, 43, 5, '#ffd88a')
        cv.circle(wx, 43, 3, '#fff0b8')
        cv.hline(wx - 5, wx + 5, 43, '#c07a3a')
        cv.vline(wx, 38, 48, '#c07a3a')
    # door
    cv.round_rect(cx - 5, 40, 10, 14, '#7a2a2a', r=2)
    cv.rect(cx - 4, 42, 8, 12, '#ffcf7a')
    cv.vline(cx, 42, 53, '#c07a3a')
    for x in (12, cx - 8, cx + 7, w - 13):
        cv.rect(x, 32, 2, 22, 'brick')
        cv.vline(x, 32, 53, 'vermilion')
    # lower roof: wide, curved, upturned eaves
    def roof(y0, half, depth, ridge):
        pts = [(cx - half - 4, y0 + depth - 3), (cx - half, y0 + depth - 1), (cx - half + 6, y0 + depth),
               (cx + half - 6, y0 + depth), (cx + half, y0 + depth - 1), (cx + half + 4, y0 + depth - 3),
               (cx + half - 2, y0 + 3), (cx + ridge, y0), (cx - ridge, y0), (cx - half + 2, y0 + 3)]
        m = poly_mask(h, w, pts)
        yy, xx = np.mgrid[0:h, 0:w]
        stripes = ((xx // 3) % 2 == 0)
        cv.a[m] = C('#3aa06a')
        cv.a[m & stripes] = C('#2e8a5c')
        cv.a[m & (xx < cx - 6) & ~stripes] = C('#5cc07e')
        bd = outline_of(m)
        cv.a[bd] = C('#14503c')
        cv.hline(cx - half + 4, cx + half - 4, y0 + depth, '#c8a040')       # gilded eave line
        cv.hline(cx - ridge, cx + ridge, y0, '#c8a040')
        cv.px(cx - half - 4, y0 + depth - 4, 'gold')
        cv.px(cx + half + 4, y0 + depth - 4, 'gold')
    roof(20, 34, 12, 12)
    # upper storey
    cv.rect(cx - 13, 12, 26, 9, '#e8dcc0')
    cv.circle(cx, 16, 3, '#7a3a2a')
    cv.circle(cx, 16, 2, '#ffd88a')
    cv.rect(cx - 13, 12, 2, 9, 'vermilion')
    cv.rect(cx + 11, 12, 2, 9, 'vermilion')
    roof(3, 21, 10, 5)
    # finial: a golden flame-pearl (emblem)
    cv.vline(cx, 0, 3, 'brass')
    cv.circle(cx, 1, 1, 'gold')
    cv.px(cx, 0, 'butter')
    return to_layer(cv)


@lru_cache(maxsize=None)
def _bamboo(sway, seed, n, x0, x1, ytop, ybot, w, ramp=('#2e7a3a', '#4fa04a', '#86c85a', '#c8e88a'),
            canopy=True, thick=2):
    """A clump of bamboo stalks + leafy crowns (RGBA layer), tops displaced by `sway` px."""
    cv = new(w, H)
    R = rng(seed)
    dk, md, lt, hi = ramp
    tops = []
    for i in range(n):
        x = x0 + R.random() * (x1 - x0)
        top = int(ytop + R.random() * 24)
        th = thick if R.random() < 0.7 else max(1, thick - 1)
        node = int(6 + R.random() * 4) + thick * 2
        lean = (R.random() - 0.5) * 0.08
        for y in range(max(0, top), min(H, ybot)):
            t = (ybot - y) / max(1, ybot - top)
            xx = int(round(x + lean * (ybot - y) + sway * t * t))
            if th >= 3:
                cv.hline(xx - 1, xx + 1, y, md)
                cv.px(xx - 1, y, lt)
                cv.px(xx + 1, y, dk)
            elif th == 2:
                cv.px(xx, y, md)
                cv.px(xx + 1, y, dk)
            else:
                cv.px(xx, y, md)
            if (y - top) % node == 0:
                cv.hline(xx - (th >= 3), xx + (th >= 2), y, hi)
                cv.hline(xx - (th >= 3), xx + (th >= 2), y + 1, dk)
            # thin leaf sprays at the upper nodes
            if y > top + 2 and (y - top) % node == 3 and y < top + 40 and hash01(seed, i, y) < 0.8:
                side = 1 if hash01(i, y, 3) < 0.5 else -1
                L = 4 + int(hash01(i, y, 4) * 4) + thick
                for k in range(L):
                    yk = y + (k * k) // (2 * L) - 1
                    cv.px(xx + side * (k + 1 + (th >= 3)), yk, lt if k < L - 2 else md)
                    if 0 < k < L - 1:
                        cv.px(xx + side * (k + 1 + (th >= 3)), yk + 1, dk)
        tops.append((x + sway + lean * (ybot - top), top))
    if canopy:
        # feathery foliage: fans of lanceolate leaves drooping from the upper nodes
        fol = [('#1c4a30', '#2e7a3a', '#4f9a4a'), ('#24603a', '#4f9a4a', '#78bc5c'),
               ('#2e7a3a', '#78bc5c', '#a8dc80')]
        for i, (tx, ty) in enumerate(tops):
            for j in range(6):
                fy = ty + 2 + j * 5 + int(R.random() * 3)
                fxp = int(tx - sway * (j * 0.08)) + int((R.random() - 0.5) * 3)
                for k in range(3 + int(R.random() * 3)):
                    side = -1 if (k + j + i) % 2 else 1
                    a = (0.15 + R.random() * 0.9) * side
                    L = 5 + int(R.random() * 5)
                    dk_, md_, lt_ = fol[int(R.random() * 3)]
                    for q in range(L):
                        t = q / L
                        x = int(round(fxp + math.sin(a) * q * 1.1))
                        y = int(round(fy + q * 0.35 + t * t * 3 - math.cos(a) * 1.5 * (1 - t)))
                        cv.px(x, y, lt_ if q < L * 0.4 else md_)
                        if 0 < q < L - 2:
                            cv.px(x, y + 1, dk_)
    return to_layer(cv)


def draw_bamboo(c, f, ox, oy, seed, n, x0, x1, ytop, ybot, w, phase=0.0, amp=2, ramp=None, canopy=True,
                thick=2):
    s = int(round(amp * math.sin(f * 2 * math.pi / 96 + phase)))
    ramp = ramp or ('#2e7a3a', '#4fa04a', '#86c85a', '#c8e88a')
    comp(c, _bamboo(s, seed, n, x0, x1, ytop, ybot, w, ramp, canopy, thick), ox, oy)


def draw_stream(c, f, ox, oy):
    """Turquoise stream flowing toward the viewer, with two small falls (cached geometry)."""
    for y in range(STREAM_TOP, POND_Y + 1):
        yy = y + oy
        if yy < 0 or yy >= H:
            continue
        cxs, hws = stream_cx(y), stream_hw(y)
        xa, xb = int(cxs - hws) + ox, int(cxs + hws) + ox
        fall = (STREAM_TOP <= y < STREAM_TOP + 6) or (144 <= y < 149) or (y >= POND_Y - 3)
        for x in range(max(0, xa), min(W, xb + 1)):
            u = (x - ox - cxs) / max(1, hws)
            if fall:
                ph = (y * 1.0 - f * 1.5 + (x - ox) * 3.7) % 6
                cc = 'white' if ph < 2 else ('ice' if ph < 4 else 'turquoise')
            else:
                ph = (y * 0.8 - f * 0.45 + int(abs(u) * 3) * 1.7 + ((x - ox) // 3) * 0.9) % 10
                base = 'turquoise' if abs(u) < 0.55 else ('aqua' if abs(u) < 0.85 else 'teal')
                cc = 'ice' if ph < 1.0 and abs(u) < 0.8 else base
            c.a[yy, x] = col(cc)
        # foam at the foot of the falls
        if y in (STREAM_TOP + 6, 149):
            for x in range(max(0, xa), min(W, xb + 1)):
                if hash01(x - ox, f // 3, y) < 0.7:
                    c.a[yy, x] = col('white')


def draw_bridge(c, ox, oy):
    """Curved wooden (moon) bridge spanning the stream."""
    cx, y0, half, rise = 118 + ox, 141 + oy, 30, 13
    for dx in range(-half, half + 1):
        t = dx / half
        yd = int(round(y0 - rise * (1 - t * t)))
        c.vline(cx + dx, yd, yd + 3, '#7a4a2e')
        c.px(cx + dx, yd, '#e0a060' if dx % 3 else '#b07040')
        c.px(cx + dx, yd + 1, '#c0703a')
        c.px(cx + dx, yd + 3, '#4a2a22')
        # railing
        if dx % 6 == 0:
            c.vline(cx + dx, yd - 5, yd - 1, 'vermilion')
            c.px(cx + dx, yd - 6, 'gold')
        c.px(cx + dx, yd - 5, 'brick')
        c.px(cx + dx, yd - 4, 'vermilion' if dx % 2 else 'brick')


def draw_lantern(c, x, y, f, seed, size=1):
    """Hanging paper lantern (top at (x, y)) with a warm glow; gently swings."""
    sw = int(round(math.sin(f * 2 * math.pi / 60 + seed) * 0.8))
    x += sw
    if size == 1:
        c.px(x, y, '#4a2a22')
        c.rect(x - 1, y + 1, 3, 4, 'vermilion')
        c.px(x, y + 2, 'gold')
        c.px(x, y + 3, 'peach')
        c.px(x + 1, y + 3, 'brick')
        c.px(x, y + 5, 'gold')
    else:
        c.vline(x, y, y + 1, '#4a2a22')
        c.rect(x - 2, y + 2, 5, 6, 'vermilion')
        c.rect(x - 1, y + 3, 3, 4, '#ffb070')
        c.px(x, y + 4, 'butter')
        c.hline(x - 2, x + 2, y + 2, '#4a2a22')
        c.hline(x - 2, x + 2, y + 7, '#4a2a22')
        c.px(x - 2, y + 4, 'brick')
        c.vline(x, y + 8, y + 9, 'gold')
    fl = 0.12 + 0.04 * math.sin(f * 0.5 + seed * 2)
    fx.glow(c, x, y + 3 + size, 7 + 3 * size, 'amber', fl, rings=3)


def draw_pond(c, f, ox, pl):
    """Foreground pond: rippled reflection of the grove + lily pads + ring ripples."""
    y0 = int(pl.min())
    nrow = H - y0
    rows = np.arange(y0, H)
    src = np.clip(2 * y0 - rows - 2, 0, H - 1)
    d = np.arange(nrow) / max(1, nrow)
    shift = np.round(np.sin(f * 0.14 + np.arange(nrow) * 1.1) * (0.6 + d * 1.6)).astype(int)
    xi = np.clip(np.arange(W)[None, :] + shift[:, None], 0, W - 1)
    refl = c.a[src[:, None], xi].astype(np.float32)
    wc = np.array([np.array(col(mix('#3fc8b4', '#178a86', t)), np.float32) for t in d])
    k = 0.55 + 0.25 * d
    out = refl * (1 - k[:, None, None]) + wc[:, None, :] * k[:, None, None]
    xs = np.arange(W) - ox
    plx = pl[np.clip(xs, 0, len(pl) - 1)]
    mask = rows[:, None] >= plx[None, :]
    reg = c.a[y0:H]
    reg[mask] = np.clip(out, 0, 255).astype(np.uint8)[mask]
    # wave highlight dashes
    for i in range(36):
        y = y0 + 2 + int(hash01(i, 61) * (nrow - 3))
        x = int((hash01(i, 62) * 360 + f * (0.2 + hash01(i, 63) * 0.3)) % 360) - 20 + ox // 2
        L = 2 + int((y - y0) / 6)
        if 0 <= y < H:
            c.hline(x, x + L, y, '#bff4e6' if i % 3 else 'white')
    # lily pads with a bloom or two
    for i, (lx, ly, r) in enumerate(((60, 166, 4), (72, 172, 3), (196, 170, 5), (214, 164, 3), (252, 174, 4),
                                    (170, 177, 4))):
        bob = int(round(math.sin(f * 0.08 + i) * 0.5))
        x, y = lx + ox, ly + bob
        c.ellipse(x, y, r + 1, max(1, r // 2), '#1f7a4a')
        c.ellipse(x, y - 1, r, max(1, r // 2 - 1) if r > 3 else 0.5, '#3aa05a')
        c.px(x + r // 2, y - 1, '#178a86')
        if i % 2 == 0:
            c.px(x - 1, y - 2, 'cotton')
            c.px(x, y - 3, 'bubblegum')
            c.px(x + 1, y - 2, 'cotton')
            c.px(x, y - 2, 'gold')
    # ring ripples where the stream enters
    for q in range(3):
        t = ((f + q * 16) % 48) / 48
        r = 3 + t * 16
        cx_, cy_ = int(stream_cx(POND_Y)) + ox, POND_Y + 4
        for a_ in range(0, 360, 20):
            if hash01(q, a_) < 0.8 - t * 0.6:
                x = int(cx_ + math.cos(math.radians(a_)) * r)
                y = int(cy_ + math.sin(math.radians(a_)) * r * 0.25)
                if y > POND_Y:
                    c.px(x, y, '#bff4e6')


@lru_cache(maxsize=1)
def _fg_c():
    """Foreground: mossy boulders + ferns (left with a stone lantern, right with flowers)."""
    w = 380
    cv = new(w, H)
    stone = ['#4c5e58', '#62766c', '#7e9286', '#a0b2a2', '#c4d2c0']
    moss = ['#24603a', '#337a40', '#4f9a4a', '#78bc5c', '#a8dc80']
    blobs(cv, [(30, 172, 22, 12, 0), (56, 177, 14, 8, 1), (6, 162, 12, 14, 0.5)], stone, outline='#34443e',
          light=(-0.7, -0.6, 0.4), dith=0.8)
    blobs(cv, [(22, 162, 12, 4, 2), (50, 170, 8, 3, 2)], moss, outline='#1c4a30', light=(-0.7, -0.6, 0.4))
    blobs(cv, [(352, 172, 22, 12, 0), (330, 178, 12, 7, 1)], stone, outline='#34443e',
          light=(-0.7, -0.6, 0.4), dith=0.8)
    blobs(cv, [(350, 162, 14, 4, 2)], moss, outline='#1c4a30', light=(-0.7, -0.6, 0.4))
    # ferns
    for (fx_, fy_, n, sgn) in ((66, 174, 5, 1), (318, 170, 5, -1), (78, 179, 4, 1)):
        for k in range(n):
            a = -math.pi / 2 + (k - n / 2) * 0.45
            for j in range(9):
                x = int(fx_ + math.cos(a) * j * 1.2 * sgn)
                y = int(fy_ + math.sin(a) * j + j * j * 0.09)
                cv.px(x, y, 'green' if j < 6 else 'mint')
                if j % 2 == 0 and j > 1:
                    cv.px(x + 1, y, 'leaf')
    for i in range(14):
        x = 330 + int(hash01(i, 71) * 44)
        y = 158 + int(hash01(i, 72) * 8)
        cv.px(x, y, ('bubblegum', 'cream', 'lavender', 'cotton')[i % 4])
        cv.px(x, y + 1, 'green')
    # stone lantern (on the left boulder)
    lx, ly = 40, 144
    cv.rect(lx - 2, ly + 8, 5, 14, '#8e9e92')
    cv.vline(lx - 2, ly + 8, ly + 21, '#c4d2c0')
    cv.rect(lx - 4, ly + 6, 9, 2, '#a0b2a2')
    cv.rect(lx - 3, ly, 7, 6, '#7e9286')
    cv.rect(lx - 1, ly + 1, 3, 3, '#3a2a22')
    cv.tri((lx - 6, ly), (lx + 6, ly), (lx, ly - 5), '#62766c')
    cv.hline(lx - 6, lx + 6, ly, '#c4d2c0')
    cv.px(lx, ly - 6, '#a0b2a2')
    return to_layer(cv), (lx, ly + 1)


def render_c(f):
    cam = cam_c(f)
    c = _sky_c().copy()
    # drifting clouds (soft, warm)
    for i, (cy_, w_, sp) in enumerate(((22, 44, 0.08), (40, 30, 0.12), (12, 36, 0.05))):
        x = int(150 + i * 70 - f * sp - cam * 0.05) % 380 - 40
        fx.cloud(c, x, cy_, w_, color='#fff6e6', shade_color='#e8d6c8', seed=i + 3)
    comp(c, _far_karsts(), -int(cam * 0.12) - 10, 0)
    lag, isl = _lagoon()
    lox = -int(cam * 0.2) - 10
    comp(c, lag, lox, 0)
    # lagoon glints
    for i in range(18):
        gx = int(hash01(i, 31) * W)
        gy = LAG0 + 2 + int(hash01(i, 32) * (LAG1 - LAG0 - 3))
        if hash01(i, f // 5) < 0.4:
            c.hline(gx, gx + 1 + i % 2, gy, 'white')
    # balloon between the karsts
    draw_balloon(c, 150 - int(cam * 0.25) + int(seg(f, C0, C1) * 8), 50, f, 's')
    mk = _mid_karsts()
    mox = -int(cam * 0.35) - 8
    comp(c, mk, mox, 0)
    draw_ribbon_fall(c, f, 40 + mox, 52, 118)
    # mist band drifting along the karst feet
    for i in range(7):
        x = int((i * 64 + f * 0.25) % 450) - 70
        y = 104 + (i % 3) * 4
        L = 50 + (i % 3) * 18
        c.blend(x, y, L, 3, '#eef6f0', 0.32)
        c.blend(x + 8, y - 1, L - 16, 1, '#eef6f0', 0.22)
        c.blend(x + 6, y + 3, L - 12, 1, '#eef6f0', 0.18)
    # grove bank + bamboo + pavilion
    bk, top, sm, pl = _bank()
    box = -int(cam * 0.6) - 10
    draw_bamboo(c, f, box, 0, 501, 26, 170, 300, 46, 124, GW, phase=0.0, amp=2,
                ramp=('#3f7a52', '#5f9a62', '#8cbc78', '#bfe0a0'))
    draw_bamboo(c, f, box, 0, 502, 18, 0, 92, 50, 126, GW, phase=1.3, amp=2,
                ramp=('#3f7a52', '#5f9a62', '#8cbc78', '#bfe0a0'))
    comp(c, bk, box, 0)
    pv = _pavilion()
    pvx, pvy = 196 + box, BANK - 62 + 2
    comp(c, pv, pvx, pvy)
    # pavilion windows flicker warm
    fl = (f // 4) % 3
    if fl == 0:
        for wx in (pvx + 38 - 16, pvx + 38 + 16):
            c.px(wx, pvy + 43, 'butter')
    for (lx, ly, sd) in ((pvx + 1, pvy + 30, 1), (pvx + 75, pvy + 30, 2), (pvx + 15, pvy + 12, 3),
                         (pvx + 61, pvy + 12, 4)):
        draw_lantern(c, lx, ly, f, sd, size=1)
    draw_stream(c, f, box, 0)
    draw_bridge(c, box, 0)
    draw_pond(c, f, box, pl)
    # lantern string from the bridge post to the pavilion eave
    ax, ay = 148 + box, 122
    bx_, by_ = pvx + 2, pvy + 30
    for k in range(0, 41):
        t = k / 40
        x = int(lerp(ax, bx_, t))
        y = int(lerp(ay, by_, t) + 6 * math.sin(t * math.pi))
        c.px(x, y, '#5a3a2e')
    for k in range(1, 5):
        t = k / 5
        draw_lantern(c, int(lerp(ax, bx_, t)), int(lerp(ay, by_, t) + 6 * math.sin(t * math.pi)) + 1, f, k)
    c.vline(ax, ay, 140, '#7a4a2e')
    c.px(ax, ay - 1, 'gold')
    draw_lantern(c, ax + 2, ay + 1, f, 7, size=2)
    # sparkles on the stream
    for i in range(10):
        y = STREAM_TOP + 10 + int(hash01(i, 41) * (H - STREAM_TOP - 12))
        x = int(stream_cx(y) + (hash01(i, 42) - 0.5) * stream_hw(y) * 1.4) + box
        fx.sparkle(c, x, y, f, 'white', period=20 + i % 4 * 6, phase=hash01(i, 43), size=1)
    # foreground bamboo framing + rocks, stone lantern (par 1.3)
    fox = -int(cam * 1.3)
    fgl, lamp = _fg_c()
    comp(c, fgl, fox - 10, 0)
    lx_, ly_ = lamp
    fl2 = 0.18 + 0.05 * math.sin(f * 0.4)
    c.rect(lx_ + fox - 10 - 1, ly_, 3, 3, 'butter' if (f // 5) % 4 else 'lava_hi')
    fx.glow(c, lx_ + fox - 10, ly_ + 1, 14, 'amber', fl2, rings=3)
    draw_bamboo(c, f, fox, 0, 601, 6, -8, 22, -40, H, 380, phase=0.7, amp=3, canopy=False, thick=3)
    draw_bamboo(c, f, fox, 0, 602, 6, 326, 356, -40, H, 380, phase=2.1, amp=3, canopy=False, thick=3)
    # falling leaves
    for i in range(14):
        per = 120 + int(hash01(i, 51) * 80)
        t = ((f + hash01(i, 52) * per) % per) / per
        x = int(hash01(i, 53) * W + t * 40 + math.sin(t * 12 + i) * 5)
        y = int(-4 + t * (H + 8))
        c.px(x % W, y, 'leaf' if (f // 4 + i) % 2 else 'mint')
        c.px((x + 1) % W, y + ((f // 4 + i) % 2), 'green')
    fx.grade(c, (1.0, 0.99, 0.95), (6, 3, 0))
    callout(c, pvx + 38, pvy + 1, 'KIRIN GROVE', f, 288, None, side='up', length=12)
    return c


def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < B0:
        c = render_a(f)
        iris(c, 1 - seg(f, 72, 79), 204, 96)
    elif f < C0:
        c = render_b(f)
    else:
        c = render_c(f)
    atlas_card(c, f, 100, 208, 'DRAGON LANDS', ['VOLCANOES - LAVA - GEMS', 'KIRIN GROVE TO THE SOUTH'])
    return c


SFX = [
    (80, 'rumble', 0.45),                                  # the volcano breathing as the iris opens
    (108, 'waves', 0.55, {'align': 'hit'}),                # surf breaks on the black sand (foam peak)
    (144, 'wind', 0.30, {'variant': 0}),                   # sea breeze, plume drifting
    (150, 'lava', 0.30, {'variant': 0}),                   # distant crater bubbling
    (182, 'lava', 0.60, {'variant': 0}),                   # into the caldera: the lava river blorps
    (184, 'steam', 0.45),                                  # geyser burst (beat)
    (198, 'rumble', 0.85),                                 # the pool swells, tremor
    (202, 'steam', 0.35),                                  # vent hiss (beat)
    (216, 'lava', 1.0, {'variant': 1}),                    # ERUPT: lava fountain boom
    (234, 'steam', 0.40),                                  # vent hiss after the blast
    (252, 'steam', 0.30),                                  # geyser again
    (270, 'gong', 0.40),                                   # cloud into Kirin Grove, calm
    (274, 'waterfall', 0.40),                              # stream, small falls, ribbon fall
    (300, 'wind', 0.30, {'variant': 0}),                   # bamboo rustle
    (330, 'waterfall', 0.30),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(180, 'zoom', {'cx': 184, 'cy': 37}), (270, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
