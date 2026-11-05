"""19 · yakyakistan — Yakyakistan + the Frozen North + Mt. Everhoof. 4 bars / 288 f, HUD 22:15 -> 22:45.

Moonlit snow night. Key light: a big bright moon (cool silver from the upper right / left); counter-key:
warm firelight (torches, braziers, windows, chimneys). Snow falls all the time in 3 depths, drifting with
wind gusts (one shared wind function drives flakes, smoke, banners, pennants and flames). NO creatures.

Beat sheet (SCENE frames; film = scene + 72 after the 2-bar flight wrapper; 18 f / beat, 72 / bar;
G minor stomping folk). Diorama 72..359 (4 bars), cruise() cameras, cuts on beats 180 and 270:
  0-71      World-map travel shot (wrapped by engine/regionfx.long_flight).
  72-179    SHOT A "Log Gates" (3.6 s). Iris opens 72-79 on the gate (204, 105). Slow even pan right.
            Night sky (moon + halo, stars, faint aurora remnant), far range with Mt. Everhoof's distant
            pyramid, pine hills, village roofs behind the wall (watchtower, hall, smoking chimneys), the
            LOG GATES: timber palisade with iron bands, gateposts crowned with crescent horn carvings,
            painted red/blue/yellow folk band and door diamonds, snow piles, icicles, torches + braziers,
            banners and pennants in the wind; sled tracks, a parked sled; foreground pines + drifts.
            Balloon 's' drifts across the moon. Callouts YAKYAKISTAN 90-162, LOG GATES 108-172.
            Atlas card 100-208.
  180       zoom-iris into the gate doors (smooth_cuts).
  180-269   SHOT B "Inside the walls" (3.0 s): huts, rugs drying, firewood, lanterns, the GREAT HALL
            (carved roof, horn finials, big painted door, torches); the glossy ICE SLIDE winding down the
            flank above a moonlit sea of clouds. `avalanche` 216 (bar line): crack + trickle from 202,
            the hall's roof slab slides off on the hit, plume, pour-off, impact billow + shake, mound.
            Callouts GREAT HALL 180-252, ICE SLIDE 198-258.
  270       night cloud pass (smooth_cuts).
  270-359   SHOT C "Frozen North" (3.0 s): ice boulders and wind-carved ridges with spindrift, igloos
            glowing from inside, ice cave with pulsing crystal light, northern stars + aurora, and MT.
            EVERHOOF's colossal moonlit summit with a snow plume; slow tilt up + pan. Balloon 'm' with a
            glowing burner. Callout MT. EVERHOOF 288-350. Toast from ~302 (summit kept below y 18).
SFX: howling wind, torch crackle (creak of the gate timbers), gong from the hall, avalanche + rumble at
     198, crystal shimmer at the ice cave, soft wind to finish.
"""
import math
from functools import lru_cache

import numpy as np
from scipy import ndimage

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'yakyakistan'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)

# =====================================================================================  toolkit
KEY = (255, 0, 254)


def new(w, h):
    return Canvas(w, h, KEY)


def rgba(cv):
    a = cv.a
    m = ~((a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2]))
    out = np.zeros((cv.h, cv.w, 4), np.uint8)
    out[..., :3] = a
    out[..., 3] = m.astype(np.uint8) * 255
    return out


def RA(ramp):
    return np.array([col(x) for x in ramp], np.uint8)


def vnoise(h, w, cell, seed):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng(seed).random((gh, gw)).astype(np.float32)
    ys = np.arange(h) / cell
    xs = np.arange(w) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = ys - y0
    fx_ = xs - x0
    fy = (fy * fy * (3 - 2 * fy))[:, None]
    fx_ = (fx_ * fx_ * (3 - 2 * fx_))[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c_ = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx_) + b * fx_) * (1 - fy) + (c_ * (1 - fx_) + d * fx_) * fy


def fbm(h, w, cell, seed, octaves=3):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for i in range(octaves):
        out += vnoise(h, w, max(1.0, cell / 2 ** i), seed + 17 * i) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def noise1(n, cell, seed, octaves=2):
    return fbm(1, n, cell, seed, octaves)[0]


def pmask(w, h, pts):
    cv = Canvas(w, h, (0, 0, 0))
    cv.poly(pts, (255, 255, 255))
    return cv.a[..., 0] > 0


def emask(w, h, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / (rx + 0.5)) ** 2 + ((yy - cy) / (ry + 0.5)) ** 2 <= 1.0


def edge_of(m):
    return m & ~ndimage.binary_erosion(m)


def blob_t(mask, light=(0.7, -0.7), rad=5, amb=0.0):
    """Lit-ness 0..1 of a rounded blob (distance-field normals)."""
    d = ndimage.distance_transform_edt(mask)
    hgt = np.sqrt(np.minimum(d, rad) / rad)
    gy, gx = np.gradient(hgt)
    nx, ny = -gx * rad * 0.8, -gy * rad * 0.8
    lx, ly = light
    lz = 0.55
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lit = (nx * lx + ny * ly + lz) / (np.sqrt(nx * nx + ny * ny + 1) * ln)
    return np.clip(lit + amb, 0, 1).astype(np.float32)


def cam_ease(f, f0, f1, a, b, ease=ease_in_out):
    return lerp(a, b, ease(seg(f, f0, f1)))


# ------------------------------------------------------------------ materials (dark -> light)
WOOD = ['#1f1529', '#30202f', '#462b37', '#5e3a3d', '#7a4f46', '#9a6a53', '#bd8a66']
IRON = ['#1c1a2b', '#2b2a40', '#43445e', '#666a86', '#9ca2bf']
SNOW = ['#1d2352', '#29336b', '#394788', '#4e5ea6', '#6a7cc2', '#8fa1dc', '#b8c6f1', '#e4ebff']
IVORY = ['#35345c', '#55557f', '#7f7ea6', '#aeabcb', '#d8d4ea', '#f7f4ff']
RED = ['#3e1428', '#68203a', '#962c47', '#c23f55', '#e0606a']
BLUE = ['#141e46', '#1e3474', '#2a4fa4', '#3f73cc', '#6a9be6']
YEL = ['#4a3018', '#7e5824', '#b88a30', '#e0b443', '#f6d978']
STONE = ['#232540', '#33365a', '#4a4e74', '#666b8f', '#8a90ad', '#b0b6cc']
WARM = ['#a8501e', '#e07a26', '#f7a43a', '#ffd16a', '#fff2b0']
PINE = ['#0b1028', '#111a36', '#192544', '#233455', '#2f4568']
ROOF = ['#1a1428', '#271c33', '#38273e', '#4c3448', '#634352']
ICE = ['#1f3a7a', '#2c5aa8', '#3f82cc', '#62aee6', '#9ad6f6', '#d4f2ff', '#ffffff']
GAP = ['#120c1c', '#1a1226', '#22182e']
END = ['#4e3432', '#7a5442', '#a5785a', '#c99d72', '#e6c690']
FAR = ['#2a2f66', '#353c77', '#434b8a', '#565fa0', '#6e77b6', '#8b94cc', '#abb3e0', '#cfd5f2']
MIDS = ['#161b44', '#1f2656', '#29326a', '#36417f', '#475496', '#5e6cb0', '#7b8cca', '#a3b2e4']
ROCK = ['#17162c', '#211f3a', '#2c2948', '#3a3558', '#4b446a']
RUG_R = ['#3a1426', '#5e1e34', '#862c40', '#aa3e4c']
RUG_B = ['#121a3c', '#1c2a5c', '#28407e', '#3a5aa0']
RUG_Y = ['#3c2a18', '#6a4a22', '#94702e', '#bc9a46']
FUR = ['#2a1c22', '#3e2a2c', '#56393a', '#704c46', '#8c6352']
CLOUD = ['#2e3470', '#3c4380', '#4f5794', '#6770aa', '#848dc2', '#a6aed8', '#cdd3ef', '#eef0ff']
MATS = {23: CLOUD, 1: WOOD, 2: IRON, 3: SNOW, 4: IVORY, 5: RED, 6: BLUE, 7: YEL, 8: STONE, 9: WARM, 10: PINE,
        11: ROOF, 12: ICE, 13: GAP, 14: END, 16: FAR, 17: MIDS, 18: ROCK, 19: RUG_R, 20: RUG_B,
        21: RUG_Y, 22: FUR}
M_WOOD, M_IRON, M_SNOW, M_IVORY, M_RED, M_BLUE, M_YEL, M_STONE, M_WARM, M_PINE = range(1, 11)
M_ROOF, M_ICE, M_GAP, M_END, M_FAR, M_MIDS, M_ROCK = 11, 12, 13, 14, 16, 17, 18
M_RUGR, M_RUGB, M_RUGY, M_FUR, M_CLOUD = 19, 20, 21, 22, 23


class Lay:
    """Material-id + shade (0..1) buffers; colourised at the end with one dither pass."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.M = np.zeros((h, w), np.int16)
        self.T = np.zeros((h, w), np.float32)

    def put(self, x, y, mask, mat, t, under=False):
        h, w = mask.shape
        x, y = int(x), int(y)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        if under:
            m = m & (self.M[y0:y1, x0:x1] == 0)
        if np.isscalar(t):
            tt = t
        else:
            tt = np.broadcast_to(t, mask.shape)[y0 - y:y1 - y, x0 - x:x1 - x][m]
        self.M[y0:y1, x0:x1][m] = mat
        self.T[y0:y1, x0:x1][m] = tt

    def rect(self, x, y, w, h, mat, t):
        self.put(x, y, np.ones((h, w), bool), mat, t)

    def poly(self, pts, mat, t):
        m = pmask(self.w, self.h, pts)
        self.put(0, 0, m, mat, t)

    def canvas(self, dith=0.75):
        cv = new(self.w, self.h)
        thr = bayer(self.h, self.w)
        for m, ramp in MATS.items():
            mk = self.M == m
            if not mk.any():
                continue
            r = RA(ramp)
            n = len(r)
            idx = np.clip(np.floor(np.clip(self.T[mk], 0, 0.999) * n + (thr[mk] - 0.5) * dith), 0, n - 1)
            cv.a[mk] = r[idx.astype(int)]
        return cv


def cyl(w, h, peak=0.68, lo=0.1, hi=0.9):
    u = (np.arange(w) + 0.5) / w
    t = 1 - np.abs(u - peak) / max(peak, 1 - peak)
    t = lo + (hi - lo) * t
    t[0] = lo * 0.4
    return np.broadcast_to(t[None, :], (h, w)).astype(np.float32).copy()


def grain_v(h, w, seed, amp=0.3):
    return (vnoise(h, w * 5, 7, seed)[:, ::5] - 0.5) * amp


def grain_h(h, w, seed, amp=0.3):
    return (vnoise(h * 5, w, 7, seed)[::5, :] - 0.5) * amp


def vlog(lay, x, top, bot, w, seed, mat=M_WOOD, tip=3, peak=0.68, lo=0.1, hi=0.9):
    """Vertical timber log with a blunt pointed top, knots and grain."""
    h = bot - top
    t = cyl(w, h, peak, lo, hi) + grain_v(h, w, seed)
    mask = np.ones((h, w), bool)
    xs = np.arange(w)
    cx = (w - 1) / 2
    for r in range(tip):
        half = (r + 1.2) / tip * (w / 2)
        mask[r] = np.abs(xs - cx) <= half
        t[r, xs > cx] = np.maximum(t[r, xs > cx], 0.75)
    rr = rng(seed)
    for _ in range(int(rr.integers(0, 3))):
        ky = int(rr.integers(tip + 3, max(tip + 4, h - 4)))
        kx = int(rr.integers(1, max(2, w - 2)))
        t[ky, kx] = 0.02
        t[ky - 1, kx] = min(t[ky - 1, kx], 0.25)
        t[ky + 1, kx] = max(t[ky + 1, kx], 0.7)
    for _ in range(int(rr.integers(0, 2))):
        ky = int(rr.integers(tip + 2, max(tip + 3, h - 8)))
        kx = int(rr.integers(1, max(2, w - 1)))
        t[ky:ky + int(rr.integers(3, 7)), kx] = 0.05
    lay.put(x, top, mask, mat, t)


def hlog(lay, x, y, w, h, seed, mat=M_WOOD, ends=True, peak=0.3):
    """Horizontal log, lit from above, with end-grain discs."""
    v = (np.arange(h) + 0.5) / h
    tv = 1 - np.abs(v - peak) / max(peak, 1 - peak)
    t = (0.1 + 0.8 * tv)[:, None] + np.zeros((1, w), np.float32)
    t = t + grain_h(h, w, seed, 0.28)
    t[-1, :] = 0.04
    lay.put(x, y, np.ones((h, w), bool), mat, t)
    if ends:
        for ex in (x - 1, x + w - 2):
            em = emask(3, h, 1, (h - 1) / 2, 1.2, (h - 1) / 2)
            yy, xx = np.mgrid[0:h, 0:3]
            d = np.sqrt(((xx - 1) / 1.5) ** 2 + ((yy - (h - 1) / 2) / max(1, h / 2)) ** 2)
            te = 0.45 + 0.4 * ((d * 2.2) % 1 > 0.5) - 0.2 * (d > 0.9)
            lay.put(ex, y, em, M_END, te)


def snow_tops(lay, x0, x1, up, dn, seed, sel=None, y_lo=0, y_hi=None, tbase=0.9, ang=0.6):
    """Pile snow on the top surface of what is in `lay` between columns x0..x1."""
    x0, x1 = max(0, x0), min(lay.w, x1)
    if x1 <= x0:
        return
    y_hi = lay.h if y_hi is None else y_hi
    M = lay.M[y_lo:y_hi, x0:x1]
    occ = (M > 0) if sel is None else np.isin(M, sel)
    has = occ.any(0)
    tops = np.argmax(occ, 0) + y_lo
    n = x1 - x0
    nz = noise1(n, 4, seed)
    upx = np.maximum(0, np.round(up * (0.55 + 0.9 * nz))).astype(int)
    dnx = np.maximum(1, np.round(dn * (0.6 + 0.8 * noise1(n, 3, seed + 5)))).astype(int)
    sl = np.gradient(tops.astype(np.float32))
    yy = np.arange(lay.h)[:, None]
    sm = has[None, :] & (yy >= (tops - upx)[None, :]) & (yy < (tops + dnx)[None, :])
    rel = (yy - (tops - upx)[None, :]) / (upx + dnx + 1)[None, :]
    t = tbase - rel * 0.5 + np.clip(sl, -2, 2)[None, :] * 0.08 * ang
    t = np.where(yy == (tops + dnx - 1)[None, :], t - 0.25, t)
    sub_M = lay.M[:, x0:x1]
    sub_T = lay.T[:, x0:x1]
    sub_M[sm] = M_SNOW
    sub_T[sm] = np.broadcast_to(t, sm.shape)[sm]


def glow_a(a, cx, cy, r, color, k, rings=4, sx=1.0):
    """Additive stepped glow on an RGB array view."""
    h, w = a.shape[:2]
    cx, cy, r = int(cx), int(cy), max(1, int(r))
    rx = max(1, int(r * sx))
    x0, y0, x1, y1 = max(0, cx - rx), max(0, cy - r), min(w, cx + rx + 1), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xx - cx) / sx) ** 2 + (yy - cy) ** 2) / r
    lvl = np.clip(1 - d, 0, 1) * rings
    thr = bayer(y1 - y0, x1 - x0, x0, y0)
    q = np.clip(np.floor(lvl + (thr - 0.5) * 0.9) / rings, 0, 1)[:, :, None]
    reg = a[y0:y1, x0:x1].astype(np.float32)
    reg += np.array(col(color), np.float32) * q * k
    a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def dith_ell(c, cx, cy, rx, ry, color, level):
    """Ellipse drawn on Bayer-thresholded pixels only (pixel-art fade)."""
    if level <= 0:
        return
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0, x1, y1 = c._clip(cx - rx - 1, cy - ry - 1, 2 * rx + 3, 2 * ry + 3)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = ((xx - cx) / (rx + 0.5)) ** 2 + ((yy - cy) / (ry + 0.5)) ** 2 <= 1.0
    if level < 1:
        m &= bayer(y1 - y0, x1 - x0, x0, y0) < level
    c.a[y0:y1, x0:x1][m] = col(color)


def put_px(c, xs, ys, color):
    xs = np.asarray(xs, int)
    ys = np.asarray(ys, int)
    m = (xs >= 0) & (xs < c.w) & (ys >= 0) & (ys < c.h)
    c.a[ys[m], xs[m]] = col(color)


def blit_a(c, arr, x, y):
    c.blit(arr, int(x), int(y))


# ------------------------------------------------------------------ wind / weather (shared)
def wind_x(f):
    """Cumulative wind displacement (px): steady breeze + gusts."""
    return 0.45 * f + 9.0 * math.sin(2 * math.pi * f / 144.0) + 3.0 * math.sin(2 * math.pi * f / 52.0 + 1.0)


def gust(f):
    """Instant gust strength 0..1."""
    v = 0.45 + 9.0 * 2 * math.pi / 144.0 * math.cos(2 * math.pi * f / 144.0) \
        + 3.0 * 2 * math.pi / 52.0 * math.cos(2 * math.pi * f / 52.0 + 1.0)
    return clamp((v - 0.05) / 1.1, 0, 1)


@lru_cache(maxsize=None)
def _flakes(n, seed):
    r = rng(seed)
    return r.random(n), r.random(n), r.random(n), r.random(n)


def snowfall(c, f, n, seed, vy=0.6, wk=1.0, color='white', size=1, wob=1.5, alt=None, y_max=H):
    x0, y0, ph, sp = _flakes(n, seed)
    span_x, span_y = W + 40, H + 20
    X = (x0 * span_x + wind_x(f) * wk * (0.8 + 0.4 * sp) + np.sin(f * 0.06 + ph * 6.283) * wob) % span_x - 20
    Y = (y0 * span_y + f * vy * (0.75 + 0.5 * sp)) % span_y - 10
    X = X.astype(int)
    Y = Y.astype(int)
    ok = Y < y_max
    X, Y = X[ok], Y[ok]
    if size == 1:
        put_px(c, X, Y, color)
    else:
        big = sp[ok] > 0.45
        put_px(c, X, Y, color)
        put_px(c, X[big] + 1, Y[big], alt or color)
        put_px(c, X[big], Y[big] + 1, alt or color)
        put_px(c, X[big] + 1, Y[big] + 1, alt or color)


def smoke(c, x, y, f, seed, n=9, h=30, k=1.0, period=72, body='#5d6498', hi='#8a92c4'):
    """Chimney smoke column: overlapping soft puffs drifting with the wind, dither-fading out."""
    for i in range(n):
        t = ((f + i * period / n + hash01(seed, i) * 5) % period) / period
        py = y - t * h
        px = x + t * t * h * (0.45 + 0.6 * gust(f)) * k + math.sin(t * 4 + seed) * 1.0
        r = 1.2 + t * 3.5
        lv = 1.0 if t < 0.35 else max(0.0, (1 - t) / 0.65)
        dith_ell(c, px, py, r, r * 0.75, body, lv)
        dith_ell(c, px + r * 0.3, py - r * 0.3, r * 0.6, r * 0.45, hi, lv * 0.85)


def flicker(f, seed):
    return 0.78 + 0.22 * math.sin(f * 0.9 + seed * 2.3) * math.sin(f * 0.37 + seed)


def flame(c, x, y, f, seed, big=False):
    """Flame whose base sits at (x, y); leans with the wind."""
    ph = f // 3 + int(hash01(seed) * 7)
    hh = (4 if not big else 6) + int(hash01(seed, ph) * 2.99)
    lean = 1 if gust(f) > 0.55 else 0
    wv = 1 if not big else 2
    for r in range(hh):
        u = r / hh
        half = int(round(wv * (1 - u) + (0.4 if r == 0 else 0)))
        xo = x + int(round(lean * u * 1.5 + (hash01(seed, ph, r) - 0.5) * 0.9 * u))
        yy = y - r
        cc = 'gold' if u < 0.45 else ('amber' if u < 0.75 else 'orange')
        c.hline(xo - half, xo + half, yy, cc)
        if u < 0.4 and half >= 1:
            c.px(xo, yy, 'butter')
    if hash01(seed, ph, 99) > 0.6:
        sy = y - hh - 1 - int(hash01(seed, ph, 7) * 4)
        c.px(x + lean + int(hash01(seed, ph, 8) * 3) - 1, sy, 'gold' if (f // 2) % 2 else 'amber')


def torch_light(c, x, y, f, seed, r=16, k=0.3):
    fl = flicker(f, seed)
    glow_a(c.a, x, y, r, 'amber', k * fl, rings=4)
    glow_a(c.a, x, y, max(3, r // 3), 'gold', 0.25 * fl, rings=2)


def wall_torch(c, x, y, f, seed):
    """Iron bracket torch; (x, y) = top of the torch head."""
    c.vline(x, y, y + 5, '#4a2f2a')
    c.px(x, y + 1, '#2b2a40')
    c.hline(x - 1, x + 1, y, '#43445e')
    c.px(x + 1, y + 4, '#2b2a40')
    c.px(x + 2, y + 5, '#2b2a40')
    flame(c, x, y - 1, f, seed)


def brazier(c, x, base, f, seed):
    """Iron fire bowl on a tripod; (x, base) bottom centre."""
    c.line(x - 4, base, x - 1, base - 7, '#2b2a40')
    c.line(x + 4, base, x + 1, base - 7, '#2b2a40')
    c.vline(x, base - 7, base, '#1c1a2b')
    c.rect(x - 5, base - 11, 11, 3, '#2b2a40')
    c.hline(x - 6, x + 6, base - 12, '#43445e')
    c.hline(x - 4, x + 4, base - 8, '#1c1a2b')
    c.px(x - 5, base - 12, '#9ca2bf')
    c.hline(x - 4, x + 4, base - 13, 'maroon')
    c.hline(x - 3, x + 3, base - 13, 'vermilion')
    flame(c, x - 2, base - 13, f, seed, big=False)
    flame(c, x + 1, base - 14, f + 5, seed + 1, big=True)
    flame(c, x + 3, base - 13, f + 11, seed + 2, big=False)


def pennant(c, x, y, f, seed, length=14, cols=('#c23f55', '#e0b443', '#3f73cc'), pole_h=0):
    """Long tapered pennant flying right from a pole top at (x, y)."""
    if pole_h:
        c.vline(x, y, y + pole_h, '#2b2a40')
        c.px(x, y - 1, '#e0b443')
    g = gust(f)
    amp = 0.6 + 1.4 * g
    L = int(length * (0.85 + 0.25 * g))
    for i in range(L):
        u = i / max(1, L - 1)
        hh = max(1, int(round(4 * (1 - u) + 1)))
        dy = int(round(math.sin(f * 0.5 - i * 0.7 + seed) * amp * u))
        band = (i // 3) % 3
        cc = cols[band]
        c.vline(x + 1 + i, y + dy, y + dy + hh - 1, cc)
        c.px(x + 1 + i, y + dy + hh - 1, shade(cc, 0.65))
        if i % 3 == 1 and hh >= 3:
            c.px(x + 1 + i, y + dy + hh // 2, 'paper' if band != 1 else '#7a2440')


@lru_cache(maxsize=None)
def _banner_rgba(w, h, scheme):
    """Vertical folk banner pattern (RGBA)."""
    base, b2, b3 = {0: ('#962c47', '#e0b443', '#2a4fa4'), 1: ('#2a4fa4', '#e0b443', '#c23f55'),
                    2: ('#b88a30', '#962c47', '#1e3474')}[scheme]
    cv = new(w, h)
    cv.rect(0, 0, w, h - 2, base)
    cv.vline(0, 0, h - 3, shade(base, 0.7))
    cv.vline(w - 1, 0, h - 3, shade(base, 1.2))
    cv.hline(0, w - 1, 0, '#43445e')
    # border zigzag
    for y in range(2, h - 3):
        cv.px(1 + (y // 2) % 2, y, b2)
        cv.px(w - 2 - (y // 2) % 2, y, b2)
    # stacked diamonds
    cx = w // 2
    for k, cy in enumerate(range(5, h - 4, 7)):
        for r in range(3, -1, -1):
            cc = (b3, b2, b3, 'paper')[r] if k % 2 == 0 else (b2, b3, b2, 'paper')[r]
            for dy in range(-r, r + 1):
                ww = r - abs(dy)
                cv.hline(cx - ww, cx + ww, cy + dy, cc)
    # fringe
    for x in range(w):
        cv.px(x, h - 2, b2 if x % 2 == 0 else KEY)
        cv.px(x, h - 1, shade(b2, 0.7) if x % 2 == 0 else KEY)
    return rgba(cv)


def banner(c, x, y, f, seed, w=9, h=22, scheme=0):
    """Vertical banner hanging from a rod at (x, y); its lower part swings with the wind."""
    spr = _banner_rgba(w, h, scheme)
    g = gust(f)
    c.hline(x - 1, x + w, y - 1, '#2b2a40')
    c.px(x - 1, y - 1, '#e0b443')
    c.px(x + w, y - 1, '#e0b443')
    for r in range(h):
        u = r / (h - 1)
        dx = int(round((math.sin(f * 0.22 + seed - r * 0.16) * (0.6 + 1.2 * g) + g * 1.6) * u * u))
        c.blit(spr[r:r + 1], x + dx, y + r)


# ------------------------------------------------------------------ shared builders
def mountain(lay, ax, ay, base, lw, rw, seed, mat=M_FAR, lit=(0.62, 0.95), shd=(0.12, 0.42),
             jag=3.0, rock=0.0, lit_right=True, div=0.25, div_jag=8.0):
    """Snowy peak: apex (ax, ay), base line y=base, half widths lw / rw. Lit face on the moon side."""
    x0 = ax - lw - 4
    w = lw + rw + 8
    h = base - ay + 2
    xs = np.arange(w) + x0
    d = xs - ax
    sl = np.where(d < 0, (base - ay) / lw, (base - ay) / rw)
    nz = (fbm(1, w, 9, seed, 3)[0] - 0.5) * 2
    ridge = ay + np.abs(d) * sl + nz * jag * np.clip(np.abs(d) / 8.0, 0, 1) ** 0.7
    ridge = np.maximum(ridge, ay)
    yy = np.arange(h)[:, None] + ay
    mask = (yy >= ridge[None, :]) & (yy <= base)
    gnz = fbm(h, w * 4, 14, seed + 3, 3)[:, ::4]
    divx = ax + (yy - ay) * div + (fbm(h, 1, 5, seed + 7, 2) - 0.5) * div_jag + (vnoise(h, 1, 2, seed + 8) - 0.5) * div_jag * 0.3
    xx = xs[None, :]
    litm = (xx > divx) if lit_right else (xx < divx)
    t = np.where(litm, lit[0] + (lit[1] - lit[0]) * (0.35 + 0.65 * gnz), shd[0] + (shd[1] - shd[0]) * gnz)
    depth = (yy - ay) / max(1, base - ay)
    t = t - depth * 0.18
    # ridge highlight / shadow line
    e = mask & ~np.roll(mask, 1, axis=0)
    t = np.where(e & litm, 0.99, t)
    lay.put(x0, ay, mask, mat, t)
    if rock > 0:
        rk = mask & (gnz < rock) & (depth > 0.08) & ~e
        lay.put(x0, ay, rk, M_ROCK, np.where(litm, 0.75, 0.3) + gnz * 0.2)
    return mask


def pine(lay, cx, base, h, seed, mat=M_PINE, snow=True, lit_right=True, tiers=None):
    """Snowy conifer: tiered drooping triangles with snow on each tier's shoulders."""
    rr = rng(seed)
    tiers = tiers or max(3, h // 6)
    trunk_h = max(1, h // 8)
    lay.rect(cx, base - trunk_h, 1 + (h > 20), trunk_h, M_WOOD, 0.15)
    top = base - h
    body = h - trunk_h
    for k in range(tiers):
        u0 = k / tiers
        ty = top + int(body * u0 * 0.85)
        by = top + int(body * (k + 1) / tiers) + 1
        hw = max(1.5, (h * 0.34) * ((k + 1) / tiers) ** 0.9 + rr.random() * 0.8)
        wbox = int(hw * 2 + 3)
        hbox = by - ty + 1
        yy, xx = np.mgrid[0:hbox, 0:wbox]
        xc = wbox / 2 - 0.5
        frac = (yy + 1) / hbox
        droop = (np.abs(xx - xc) / max(1, hw)) ** 2 * 1.5
        m = (np.abs(xx - xc) <= hw * frac + 0.3) & (yy <= hbox - 1 - 0 * droop)
        m &= ~((yy > hbox - 1 - droop) & (np.abs(xx - xc) < hw * 0.75))
        side = (xx - xc) / max(1, hw)
        t = 0.35 + (0.3 * side if lit_right else -0.3 * side) + (rr.random() - 0.5) * 0.1
        t = t + (vnoise(hbox, wbox, 2, seed + k) - 0.5) * 0.3
        lay.put(int(cx - xc), ty, m, mat, t)
        if snow:
            occ = m
            tops = np.argmax(occ, 0)
            has = occ.any(0)
            sn = np.zeros_like(m)
            for x in range(wbox):
                if has[x] and rr.random() < 0.8:
                    y0 = tops[x]
                    th = 1 + (abs(x - xc) > hw * 0.35) * (rr.random() < 0.6)
                    sn[y0:y0 + th, x] = True
            st = 0.62 + (0.3 * side if lit_right else -0.3 * side)
            lay.put(int(cx - xc), ty, sn & m, M_SNOW, st)


# =====================================================================================  SHOT A
MOON_A = (252, 40)
GX = 206                      # gate centre in the wall layer
GY = 150                      # ground line at the wall base
P_FAR, P_MID, P_VIL, P_WALL, P_FG = 0.1, 0.3, 0.7, 0.85, 1.35


CUT1, CUT2 = 180, 270                     # shot changes (beats; smooth_cuts at the bottom)


def cam_a(f):
    return 36 * cruise(seg(f, 72, CUT1))


def moon(c, mx, my, r=13):
    a = c.a
    glow_a(a, mx, my, r * 4, '#5a62a8', 0.22, rings=5)
    glow_a(a, mx, my, int(r * 2.2), '#8a94d0', 0.22, rings=4)
    m = emask(2 * r + 1, 2 * r + 1, r, r, r, r)
    yy, xx = np.mgrid[0:2 * r + 1, 0:2 * r + 1]
    t = 0.75 - ((xx - r) * -0.4 + (yy - r) * 0.5) / r * 0.25
    nz = fbm(2 * r + 1, 2 * r + 1, 5, 77, 3)
    t = t - (nz > 0.6) * 0.22 - (nz > 0.7) * 0.1
    ramp = RA(['#a8acd2', '#c6c9e6', '#e0e2f4', '#f4f3fb', '#fffef6'])
    thr = bayer(2 * r + 1, 2 * r + 1)
    idx = np.clip(np.floor(np.clip(t, 0, 0.999) * 5 + (thr - 0.5) * 0.7), 0, 4).astype(int)
    sub = a[my - r:my + r + 1, mx - r:mx + r + 1]
    sub[m] = ramp[idx][m]
    e = edge_of(m)
    sub[e & (xx > r - 2) & (yy < r + 2)] = col('#fffef6')


@lru_cache(maxsize=None)
def _starfield(seed, n, y_max, avoid):
    r = rng(seed)
    xs = r.integers(0, W, n * 2)
    ys = (r.random(n * 2) ** 1.4 * y_max).astype(int)
    ok = np.ones(n * 2, bool)
    for (ax, ay, ar) in avoid:
        ok &= (xs - ax) ** 2 + (ys - ay) ** 2 > ar * ar
    xs, ys = xs[ok][:n], ys[ok][:n]
    return xs, ys, r.random(len(xs)), r.random(len(xs)), r.random(len(xs))


def stars(c, f, seed, n, y_max, avoid=(), dy=0, bright=1.0):
    xs, ys, ph, per, kind = _starfield(seed, n, y_max, tuple(avoid))
    v = 0.5 + 0.5 * np.sin(2 * math.pi * (f / (40 + 60 * per) + ph))
    ys = ys + dy
    dim = v < 0.35
    put_px(c, xs[dim], ys[dim], '#5a5f9a')
    mid = (v >= 0.35) & (kind < 0.7)
    put_px(c, xs[mid], ys[mid], '#b9c2ee' if bright > 0.5 else '#8b94cc')
    hi = (v >= 0.35) & (kind >= 0.7)
    put_px(c, xs[hi], ys[hi], 'white')
    big = hi & (kind > 0.93) & (v > 0.7)
    for dx, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        put_px(c, xs[big] + dx, ys[big] + dy2, '#8b94cc')


@lru_cache(maxsize=None)
def _aurora_prof(w, seed):
    a = fbm(1, w, 18, seed, 3)[0]
    b = fbm(1, w, 4, seed + 9, 2)[0]
    return np.clip((a - 0.35) * 2.2, 0, 1) * (0.55 + 0.45 * b)


def aurora(c, f, y0, h, k=0.35, seed=5, x_shift=0):
    """Faint aurora remnant: soft green/teal curtains with violet tops, slowly breathing."""
    wtot = W + 400
    prof = _aurora_prof(wtot, seed)
    sh = int(f * 0.35 + x_shift) % 400
    cur = prof[sh:sh + W]
    v = (np.arange(h) + 0.5) / h
    vert = np.clip(np.sin(v * math.pi) ** 1.5 * (0.4 + 0.6 * v), 0, 1)
    breathe = 0.75 + 0.25 * math.sin(f * 0.05)
    I = vert[:, None] * cur[None, :] * breathe
    ray = 0.75 + 0.25 * np.sin(np.arange(W) * 0.9 + f * 0.08)[None, :]
    I = I * ray
    thr = bayer(h, W, 0, y0)
    q = np.clip(np.floor(I * 4 + (thr - 0.5) * 0.9) / 4, 0, 1)
    ca = np.array(col('#3ad6a0'), np.float32)
    cb = np.array(col('#8a5ae0'), np.float32)
    mixc = ca[None, :] * v[:, None] + cb[None, :] * (1 - v[:, None])
    y1 = min(H, y0 + h)
    reg = c.a[y0:y1].astype(np.float32)
    reg += (q[:y1 - y0, :, None] * mixc[:y1 - y0, None, :]) * k
    c.a[y0:y1] = np.clip(reg, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _sky_a():
    c = Canvas(W, H, '#0b0a20')
    c.gradient(0, 0, W, 140, ['#09081c', '#0f0f2c', '#16183c', '#1f234e', '#2a2f62', '#383d76', '#4b4f8a'],
               soft=0.9)
    c.rect(0, 140, W, 40, '#4b4f8a')
    moon(c, *MOON_A)
    return c


@lru_cache(maxsize=1)
def _far_a():
    lay = Lay(330, H)
    # distant Mt. Everhoof (huge, pale) on the left + range
    mountain(lay, 76, 52, 150, 120, 100, 11, M_FAR, lit=(0.6, 0.98), shd=(0.15, 0.45), jag=3.5, rock=0.18)
    for ax, ay, lw, rw, s in ((-6, 86, 40, 46, 12), (150, 80, 46, 40, 13), (226, 92, 40, 50, 14),
                              (300, 78, 44, 46, 15), (118, 98, 30, 34, 16)):
        mountain(lay, ax, ay, 150, lw, rw, s, M_FAR, lit=(0.5, 0.86), shd=(0.12, 0.38), jag=3, rock=0.15)
    cv = lay.canvas()
    # haze toward the base
    for y in range(100, 150):
        lv = clamp((y - 100) / 50.0) * 0.8
        row = cv.a[y]
        m = ~((row[:, 0] == KEY[0]) & (row[:, 1] == KEY[1]) & (row[:, 2] == KEY[2]))
        thr = bayer(1, cv.w, 0, y)[0]
        put = m & (thr < lv)
        row[put] = col('#4b4f8a')
    return rgba(cv)


@lru_cache(maxsize=1)
def _mid_a():
    w = 340
    lay = Lay(w, H)
    xs = np.arange(w)
    ridge = 90 + 6 * np.sin(xs / 41.0 + 0.6) + 4 * np.sin(xs / 15.0 + 2) + (noise1(w, 6, 21) - 0.5) * 5
    yy = np.arange(H)[:, None]
    m = yy >= ridge[None, :]
    sl = np.gradient(ridge)
    nz = fbm(H, w, 8, 22, 3)
    t = 0.5 + np.clip(sl, -1.5, 1.5)[None, :] * 0.2 + (nz - 0.5) * 0.35 - (yy - ridge[None, :]) * 0.01
    lay.put(0, 0, m, M_MIDS, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.95
    rr = rng(23)
    for i in range(110):
        x = int(rr.integers(-4, w + 4))
        xr = min(w - 1, max(0, x))
        base = int(ridge[xr] + rr.integers(2, 20))
        h = int(rr.integers(7, 13)) + int((base - ridge[xr]) // 3)
        pine(lay, x, base, int(h), 300 + i, snow=True)
    cv = lay.canvas()
    a = cv.a
    km = (a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2])
    # cool haze on the lower part (atmospheric depth)
    for y in range(96, 130):
        lv = clamp((y - 96) / 40.0) * 0.6
        thr = bayer(1, w, 0, y)[0]
        put = ~km[y] & (thr < lv)
        a[y][put] = col('#2e3772')
    return rgba(cv)


HUTS_A = [  # (x, base, w, wall_h, roof_h, chimney side) in the village layer (rises behind the wall)
    (4, 106, 30, 20, 20, 1), (94, 108, 28, 18, 18, 0), (258, 108, 24, 16, 16, 1), (288, 104, 54, 20, 26, 0),
]
TOWER_A = (62, 50)       # watchtower roof apex in village layer


def gable_roof(lay, x, base_y, w, rh, seed, ov=3, snow_th=4, mat=M_ROOF, snow_right=True):
    """Steep gable (facing viewer): returns apex (ax, ay). Roof slopes + thick snow."""
    ax = x + w / 2 - 0.5
    ay = base_y - rh
    pts = [(x - ov, base_y + 1), (ax, ay - 1), (x + w + ov - 1, base_y + 1)]
    m = pmask(lay.w, lay.h, pts)
    yy, xx = np.mgrid[0:lay.h, 0:lay.w]
    t = np.where(xx > ax, 0.55, 0.3) + ((yy - ay) % 3 == 0) * -0.12
    lay.put(0, 0, m, mat, t)
    # snow blanket on both slopes
    inner = pmask(lay.w, lay.h, [(x - ov + snow_th * 1.2, base_y + 1), (ax, ay + snow_th * 1.6),
                                 (x + w + ov - 1 - snow_th * 1.2, base_y + 1)])
    sm = m & ~inner
    if not snow_right:
        dust = sm & (xx > ax + 1) & ~ndimage.binary_erosion(m, iterations=1)
        sm = sm & ((xx <= ax + 1) | dust)
    nz = fbm(lay.h, lay.w, 4, seed, 2)
    ts = np.where(xx > ax, 0.82, 0.6) + (nz - 0.5) * 0.25
    ts = ts - (yy - ay) / max(1, rh) * 0.15
    lay.put(0, 0, sm, M_SNOW, ts)
    # overhang lumps at the eaves
    for ex, s in ((x - ov, -1), (x + w + ov - 1, 1)):
        if s > 0 and not snow_right:
            continue
        em = emask(lay.w, lay.h, ex, base_y, 2.2, 1.6)
        lay.put(0, 0, em, M_SNOW, 0.75 if s > 0 else 0.55)
    return int(round(ax)), ay


def stone_base(lay, x, y, w, h, seed):
    rr = rng(seed)
    lay.rect(x, y, w, h, M_STONE, 0.12)
    row_h = 3
    for ry in range(y, y + h, row_h):
        cx = x - int(rr.integers(0, 3))
        while cx < x + w:
            sw = int(rr.integers(3, 6))
            sm = emask(sw, row_h, (sw - 1) / 2, 1, (sw - 1) / 2, 1)
            t = blob_t(sm, rad=2, amb=-0.1) * 0.8 + rr.random() * 0.15
            sub = np.zeros((row_h, sw), bool)
            sub[:, :] = sm
            xx0 = max(cx, x)
            off = xx0 - cx
            ww = min(sw - off, x + w - xx0)
            if ww > 0:
                lay.put(xx0, ry, sub[:, off:off + ww], M_STONE, t[:, off:off + ww])
            cx += sw
        lay.rect(x, ry + row_h - 1, w, 0, M_STONE, 0.05)


def log_wall(lay, x, y, w, h, seed, log_h=3, ends=True):
    for i, ly in enumerate(range(y, y + h, log_h)):
        hh = min(log_h, y + h - ly)
        hlog(lay, x, ly, w, hh, seed + i, ends=ends)


def window(lay, x, y, w, h, seed, shutters=True, scheme=0):
    """Round-topped warm window with frame, cross bar, shutters and a snowy sill."""
    r = w // 2
    m = np.zeros((h, w), bool)
    m[r:, :] = True
    m |= emask(w, h, (w - 1) / 2, r, (w - 1) / 2, r)
    yy, xx = np.mgrid[0:h, 0:w]
    tw = 0.55 + 0.4 * (yy / h) + (xx == w // 2) * -0.2
    lay.put(x - 1, y - 1, np.pad(m, 1, constant_values=False) | ndimage.binary_dilation(np.pad(m, 1)),
            M_WOOD, 0.08)
    lay.put(x, y, m, M_WARM, tw)
    lay.put(x, y, m & ((xx == w // 2) | (yy == r + 1)), M_WOOD, 0.2)
    if shutters:
        sc = (M_BLUE, M_RED, M_YEL)[scheme % 3]
        for sx in (x - 3, x + w + 1):
            sm = np.ones((h, 2), bool)
            ts = np.array([[0.35, 0.6]] * h, np.float32)
            ts[::3] -= 0.2
            lay.put(sx, y, sm, sc, ts)
    lay.rect(x - 2, y + h, w + 4, 1, M_WOOD, 0.5)
    lay.rect(x - 2, y + h - 1, w + 4, 1, M_SNOW, 0.85)


def door(lay, x, y, w, h, seed, scheme=0):
    r = w // 2
    m = np.zeros((h, w), bool)
    m[r:, :] = True
    m |= emask(w, h, (w - 1) / 2, r, (w - 1) / 2, r)
    big = np.pad(m, 1)
    big = big | ndimage.binary_dilation(big)
    lay.put(x - 1, y - 1, big, M_WOOD, 0.06)
    yy, xx = np.mgrid[0:h, 0:w]
    t = 0.3 + 0.35 * ((xx % 3) == 1) + grain_v(h, w, seed, 0.2)
    t = np.where(xx % 3 == 0, 0.12, t)
    lay.put(x, y, m, M_WOOD, t)
    sc = (M_RED, M_BLUE, M_YEL)[scheme % 3]
    band = m & ((yy == r + 1) | (yy == h - 4))
    lay.put(x, y, band, sc, 0.6)
    lay.put(x, y, m & (yy == r + 2), M_YEL if sc != M_YEL else M_RED, 0.6)
    lay.put(x + w - 2, y + h // 2 + 1, np.ones((1, 1), bool), M_YEL, 0.9)


def hut(lay, x, base, w, wall_h, roof_h, seed, door_x=None, windows=(), chimney=None, stone_h=5,
        scheme=0, finial=False):
    """Wood-and-stone hut, gable facing the viewer. Returns dict with apex / chimney top."""
    stone_base(lay, x, base - stone_h, w, stone_h, seed)
    log_wall(lay, x, base - wall_h, w, wall_h - stone_h, seed + 50)
    # gable triangle wall (logs) under the roof
    ax, ay = gable_roof(lay, x, base - wall_h, w, roof_h, seed + 9)
    info = {'apex': (ax, ay), 'chimney': None}
    if chimney is not None:
        cdx, extra = chimney
        cxx = x + cdx
        ax_f = x + w / 2 - 0.5
        yr = (base - wall_h) - roof_h * (1 - abs(cxx + 2 - ax_f) / (w / 2 + 3))
        ctop = int(yr - extra)
        chh = int(yr + 4 - ctop)
        cm = np.ones((chh, 5), bool)
        t = cyl(5, chh, 0.7, 0.15, 0.75)
        t[::2] -= 0.1
        lay.put(cxx, ctop, cm, M_STONE, t)
        lay.rect(cxx - 1, ctop - 1, 7, 2, M_STONE, 0.55)
        lay.rect(cxx - 1, ctop - 3, 7, 2, M_SNOW, 0.85)
        lay.rect(cxx, ctop - 4, 5, 1, M_SNOW, 0.95)
        info['chimney'] = (cxx + 2, ctop - 4)
    for (wx, wy, ww, wh) in windows:
        window(lay, x + wx, base - wy, ww, wh, seed + wx, scheme=scheme)
    if door_x is not None:
        door(lay, x + door_x, base - 11, 7, 11, seed + 3, scheme)
    if finial:
        horn_pair(lay, ax, ay + 1, 5, seed)
    return info


def horn_path(bx, by, r0, sweep, outward, n=60):
    """Spiral horn centre-line from base (bx, by) curling outward (-1 left / +1 right)."""
    cx = bx + outward * r0 * 0.55
    cy = by - r0 * 0.85
    a0 = math.atan2(by - cy, bx - cx)
    pts = []
    for i in range(n):
        s = i / (n - 1)
        r = r0 * (1 - 0.62 * s)
        a = a0 + outward * s * sweep
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), s))
    return pts


def horn(lay, bx, by, r0, outward, seed, th0=3.2, th1=0.9, sweep=5.2):
    """Carved curled horn ornament (abstract; ivory with ring grooves)."""
    pts = horn_path(bx, by, r0, sweep, outward)
    pad = int(r0 * 2 + 8)
    w = h = 2 * pad
    ox, oy = int(bx) - pad, int(by) - pad
    m = np.zeros((h, w), bool)
    sidx = np.full((h, w), -1.0, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for (px, py, s) in pts:
        th = th0 + (th1 - th0) * s
        d = (xx - (px - ox)) ** 2 + (yy - (py - oy)) ** 2 <= th * th
        sidx[d & ~m] = s
        m |= d
    t = blob_t(m, light=(0.7, -0.7), rad=3, amb=0.1)
    groove = (np.floor(sidx * 14) % 2 == 0) & (sidx > 0.05) & (sidx < 0.85)
    t = np.where(groove, t - 0.22, t)
    lay.put(ox, oy, m, M_IVORY, t)
    e = edge_of(m)
    lay.put(ox, oy, e & (t < 0.55), M_IVORY, 0.02)
    return pts


def horn_crescent(lay, bx, by, R, outward, seed, th0=4.2, th1=0.8, a_end=250.0, n=70, collar=True):
    """Big crescent horn carving: from the base it sweeps outward, then up, the tip turning inward."""
    pad = int(R * 2 + 8)
    w, h = 2 * pad, 2 * pad
    ox, oy = int(bx) - pad, int(by) - pad
    cx, cy = bx, by - R
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), bool)
    sidx = np.full((h, w), -1.0, np.float32)
    for i in range(n):
        s_ = i / (n - 1)
        th = math.radians(90 + (a_end - 90) * s_)
        px = cx - outward * -R * math.cos(th) if False else cx + R * math.cos(th) * (1 if outward < 0 else -1)
        py = cy + R * math.sin(th)
        rad = th0 + (th1 - th0) * s_ ** 0.8 + 0.6 * math.sin(s_ * math.pi) * (1 - s_)
        d = (xx - (px - ox)) ** 2 + (yy - (py - oy)) ** 2 <= rad * rad
        sidx[d & ~m] = s_
        m |= d
    t = blob_t(m, light=(0.7, -0.7), rad=3, amb=0.15)
    groove = (np.floor(sidx * 22) % 2 == 0) & (sidx > 0.04) & (sidx < 0.45)
    t = np.where(groove, t - 0.2, t)
    lay.put(ox, oy, m, M_IVORY, t)
    e = edge_of(m)
    lay.put(ox, oy, e & (t < 0.5), M_IVORY, 0.03)
    if collar:
        cm = m & (sidx >= 0) & (sidx < 0.03)
        lay.put(ox, oy, cm, M_RED, 0.6)
        cm2 = m & (sidx >= 0.03) & (sidx < 0.05)
        lay.put(ox, oy, cm2, M_YEL, 0.75)


def horn_pair(lay, ax, ay, r0, seed):
    horn(lay, ax - 1, ay, r0, -1, seed, th0=1.6, th1=0.6, sweep=4.4)
    horn(lay, ax + 1, ay, r0, 1, seed, th0=1.6, th1=0.6, sweep=4.4)


def merge_under(lay, other):
    m = (lay.M == 0) & (other.M > 0)
    lay.M[m] = other.M[m]
    lay.T[m] = other.T[m]


def cool_grade(cv, lay, mul=(0.62, 0.68, 0.92), keep=(M_SNOW, M_ICE, M_WARM, M_IVORY)):
    """Moon-behind night grade on everything except snow / ice / lit glass."""
    m = (lay.M > 0) & ~np.isin(lay.M, keep)
    reg = cv.a[m].astype(np.float32) * np.array(mul, np.float32)
    cv.a[m] = np.clip(reg, 0, 255).astype(np.uint8)


def bake_lights(cv, lights, color='amber'):
    a = cv.a
    km = (a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2])
    for (x, y, r, k) in lights:
        glow_a(a, x, y, r, color, k, rings=5)
        glow_a(a, x, y, max(4, r // 3), 'gold', k * 0.35, rings=3)
    a[km] = KEY


@lru_cache(maxsize=1)
def _village_a():
    lay = Lay(350, H)
    tx, ty = TOWER_A
    for i in range(3):
        vlog(lay, tx - 9 + i * 6, ty + 16, 150, 6, 500 + i, tip=0)
    hlog(lay, tx - 12, ty + 13, 24, 3, 509)
    lay.rect(tx - 11, ty + 16, 22, 2, M_WOOD, 0.5)
    for i in range(5):
        lay.rect(tx - 10 + i * 5, ty + 6, 1, 7, M_WOOD, 0.35)
    lay.rect(tx - 11, ty + 7, 22, 1, M_WOOD, 0.6)
    lay.rect(tx - 9, ty + 8, 18, 5, M_WARM, 0.35)
    window(lay, tx - 3, ty + 21, 6, 7, 3, shutters=True, scheme=1)
    lay.poly([(tx - 14, ty + 7), (tx, ty - 3), (tx + 13, ty + 7)], M_ROOF, 0.4)
    lay.poly([(tx - 14, ty + 7), (tx, ty - 3), (tx + 13, ty + 7), (tx + 9, ty + 7), (tx, ty + 1),
              (tx - 10, ty + 7)], M_SNOW, 0.75)
    snow_tops(lay, tx - 15, tx + 15, 2, 2, 501, y_lo=ty - 6, y_hi=ty + 10)
    chimneys = []
    for i, (x, base, w, wh, rh, cs) in enumerate(HUTS_A):
        wins = ((w // 2 - 3, wh - 3, 6, 7),) if w < 40 else ((8, wh - 3, 6, 7), (w // 2 - 3, wh - 3, 6, 7),
                                                              (w - 14, wh - 3, 6, 7))
        info = hut(lay, x, base, w, wh, rh, 600 + i * 7, windows=wins,
                   chimney=((w - 9) if cs else 4, 5), scheme=i)
        chimneys.append(info['chimney'])
    cv = lay.canvas()
    cool_grade(cv, lay, mul=(0.7, 0.74, 0.95))
    return rgba(cv), tuple(chimneys)


POST_TOP, LINTEL_Y, BAND_Y, DOOR_Y, GABLE_APEX = 64, 88, 96, 100, 70
TURRETS_A = (GX - 142, GX + 134)
LIGHTS_A = [  # (x, y, r, k, kind) wall-layer coords; kind: torch / brazier / lantern
    (GX - 40, 98, 'torch'), (GX + 40, 98, 'torch'), (GX - 76, 112, 'torch'), (GX + 72, 112, 'torch'),
    (GX - 54, GY - 12, 'brazier'), (GX + 54, GY - 12, 'brazier'),
    (TURRETS_A[0], 87, 'lantern'), (TURRETS_A[1], 87, 'lantern'),
]


def turret(lay, cx, seed):
    vlog(lay, cx - 9, 80, 100, 4, seed, tip=0)
    vlog(lay, cx + 5, 80, 100, 4, seed + 1, tip=0)
    lay.rect(cx - 5, 82, 10, 14, M_WARM, 0.18)
    hlog(lay, cx - 11, 95, 22, 3, seed + 2)
    for k in range(4):
        lay.rect(cx - 9 + k * 6, 90, 1, 5, M_WOOD, 0.3)
    lay.rect(cx - 10, 90, 20, 1, M_WOOD, 0.55)
    gable_roof(lay, cx - 10, 80, 20, 11, seed + 3, ov=2, snow_th=3)


@lru_cache(maxsize=1)
def _wall_a():
    W_ = 352
    lay = Lay(W_, H)
    rr = rng(41)
    x = -3
    i = 0
    while x < W_:
        w = 7 + (i % 3 == 0)
        if not (GX - 37 < x + w and x < GX + 37):
            top = 97 + int(rr.integers(-2, 3))
            vlog(lay, x, top, GY + 2, w, 1000 + i, tip=3, lo=0.08, hi=0.8)
        x += w
        i += 1
    for by in (111, 134):
        for x0, x1 in ((0, GX - 36), (GX + 36, W_)):
            sub_m = lay.M[by:by + 2, x0:x1]
            sub_t = lay.T[by:by + 2, x0:x1]
            band = sub_m == M_WOOD
            r0 = sub_t[0].copy()
            sub_m[band] = M_IRON
            sub_t[0][band[0]] = np.clip(r0[band[0]] * 0.5 + 0.45, 0, 1)
            sub_t[1][band[1]] = 0.12
            rv = np.zeros_like(band[0])
            rv[3::7] = True
            sub_t[0][band[0] & rv] = 0.98
    snow_tops(lay, 0, GX - 36, 3, 2, 1600, y_hi=120)
    snow_tops(lay, GX + 36, W_, 3, 2, 1601, y_hi=120)
    # wall turrets standing behind the palisade
    tl = Lay(W_, H)
    for k, cx in enumerate(TURRETS_A):
        turret(tl, cx, 1100 + 10 * k)
    merge_under(lay, tl)
    # gate posts
    for sgn in (-1, 1):
        px0 = GX - 36 if sgn < 0 else GX + 20
        vlog(lay, px0, POST_TOP, GY + 2, 8, 1200 + sgn, tip=0, lo=0.1, hi=0.85)
        vlog(lay, px0 + 8, POST_TOP, GY + 2, 8, 1210 + sgn, tip=0, lo=0.1, hi=0.85)
        for by in (78, 120, 140):
            lay.rect(px0, by, 16, 2, M_IRON, 0.3)
            lay.rect(px0, by, 16, 1, M_IRON, 0.7)
            for k in range(2, 16, 5):
                lay.rect(px0 + k, by, 1, 1, M_IRON, 0.98)
        for (yy0, mt) in ((104, M_RED), (106, M_YEL), (108, M_BLUE), (110, M_RED)):
            lay.rect(px0, yy0, 16, 2, mt, 0.55)
        for k in range(0, 16, 4):
            lay.rect(px0 + k + 1, 106, 2, 2, M_RED, 0.75)
            lay.rect(px0 + k + 3, 108, 1, 2, M_IVORY, 0.8)
        hlog(lay, px0 - 3, POST_TOP - 5, 22, 5, 1300 + sgn)
    # lintel log + painted band
    hlog(lay, GX - 40, LINTEL_Y, 80, 8, 1400)
    for k in range(76):
        xx = GX - 38 + k
        ph = (k // 3) % 3
        lay.rect(xx, BAND_Y, 1, 4, (M_RED, M_YEL, M_BLUE)[ph], 0.5 + 0.2 * (k % 3 == 1))
    for k in range(0, 76, 6):
        lay.rect(GX - 36 + k, BAND_Y + 1, 2, 2, M_IVORY, 0.85)
    # gable above the lintel (horizontal boards)
    yy, xxg = np.mgrid[0:H, 0:W_]
    gm = pmask(W_, H, [(GX - 31, LINTEL_Y), (GX, GABLE_APEX), (GX + 31, LINTEL_Y)])
    tg = 0.38 - 0.25 * ((yy % 3) == 0) + grain_h(H, W_, 1450, 0.25) + (xxg > GX) * 0.08
    lay.put(0, 0, gm, M_WOOD, tg)
    md = emask(13, 13, 6, 6, 6, 6)
    lay.put(GX - 6, GABLE_APEX + 4, md, M_YEL, blob_t(md, rad=4) * 0.9)
    md2 = emask(13, 13, 6, 6, 3, 3)
    lay.put(GX - 6, GABLE_APEX + 4, md2, M_RED, blob_t(md2, rad=2) * 0.8 + 0.1)
    lay.put(GX - 6, GABLE_APEX + 4, edge_of(md), M_WOOD, 0.05)
    for a_ in range(8):           # sun rays carved round the disc
        ang = a_ * math.pi / 4
        rx_, ry_ = int(round(GX + math.cos(ang) * 8)), int(round(GABLE_APEX + 10 + math.sin(ang) * 8))
        if ry_ < LINTEL_Y - 1:
            lay.rect(rx_, ry_, 1, 1, M_YEL, 0.7)
    for k in range(32):
        for sgn in (-1, 1):
            bx = GX + sgn * k
            by = GABLE_APEX + int(k * (LINTEL_Y - GABLE_APEX) / 31)
            lay.rect(bx, by - 1, 1, 2, (M_BLUE, M_YEL)[(k // 2) % 2], 0.6)
    snow_tops(lay, GX - 40, GX + 40, 3, 2, 1602, y_hi=LINTEL_Y + 2)
    # doors
    dh = GY - DOOR_Y
    yy2, xx2 = np.mgrid[0:dh, 0:20]
    for sgn in (-1, 1):
        dx0 = GX - 20 if sgn < 0 else GX
        t = 0.26 + 0.2 * ((xx2 % 5) >= 2) + grain_v(dh, 20, 1500 + sgn, 0.25)
        t = np.where(xx2 % 5 == 0, 0.05, t)
        lay.put(dx0, DOOR_Y, np.ones((dh, 20), bool), M_WOOD, t)
        for by in (DOOR_Y + 3, GY - 6):
            lay.rect(dx0, by, 20, 2, M_IRON, 0.3)
            lay.rect(dx0, by, 20, 1, M_IRON, 0.7)
        cy_ = (DOOR_Y + GY) // 2 - 1
        for r, mt in ((10, M_BLUE), (8, M_YEL), (6, M_RED), (4, M_YEL), (2, M_BLUE), (0, M_IVORY)):
            dm = (np.abs(xx2 - 10) + np.abs(yy2 + DOOR_Y - cy_)) <= r
            lay.put(dx0, DOOR_Y, dm, mt, 0.5 + 0.12 * (xx2 > 10))
        for yy3 in range(DOOR_Y + 7, GY - 7):
            zx = dx0 + (1 if sgn < 0 else 18) + ((yy3 // 2) % 2) * (1 if sgn < 0 else -1)
            lay.rect(zx, yy3, 1, 1, M_RED, 0.65)
        rx_ = GX - 4 if sgn < 0 else GX + 3      # iron ring handles
        rm = emask(5, 6, 2, 3, 2, 2.5) & ~emask(5, 6, 2, 3, 1, 1.5)
        lay.put(rx_ - 2, cy_ + 2, rm, M_IRON, 0.75)
    lay.rect(GX - 1, DOOR_Y, 2, dh, M_GAP, 0.5)
    lay.rect(GX, DOOR_Y + 2, 1, dh - 4, M_WARM, 0.5)
    # horns on the gatepost caps (after snow so they stay clean ivory)
    for sgn, px0 in ((-1, GX - 36), (1, GX + 20)):
        horn_crescent(lay, px0 + 8 + sgn * 2, POST_TOP - 5, 10.5, sgn, 1350 + sgn, collar=False)
        hx0 = px0 + 8 + sgn * 2
        lay.rect(hx0 - 4, POST_TOP - 7, 9, 3, M_WOOD, 0.55)
        lay.rect(hx0 - 4, POST_TOP - 6, 9, 1, M_YEL, 0.7)
        lay.rect(hx0 - 3, POST_TOP - 8, 7, 1, M_WOOD, 0.75)
    # ground
    xs = np.arange(W_)
    gl = GY - 1 + 1.5 * np.sin(xs / 23.0) + (noise1(W_, 8, 1700) - 0.5) * 3
    drift = np.where(np.abs(xs - GX) > 26, -3 - 4 * noise1(W_, 5, 1701), 0) * \
        np.clip((np.abs(xs - GX) - 22) / 8.0, 0, 1)
    gl = gl + drift
    yyg = np.arange(H)[:, None]
    gm2 = yyg >= gl[None, :]
    nz = fbm(H, W_, 6, 1702, 3)
    # moon shadow cast by the wall toward the viewer (posts / turrets cast longer)
    L = 8 + 2 * noise1(W_, 4, 1703)
    L = L + 7 * ((np.abs(xs - GX + 28) < 8) | (np.abs(xs - GX - 28) < 8)) + 4 * (np.abs(xs - GX) < 22)
    for tcx in TURRETS_A:
        L = L + 5 * (np.abs(xs - tcx) < 10)
    sh = yyg < (GY + L)[None, :]
    tl_ = 0.66 - (yyg - GY) * 0.003 + (nz - 0.5) * 0.25
    tgr = np.where(sh, 0.3 + (nz - 0.5) * 0.15, tl_)
    lay.put(0, 0, gm2, M_SNOW, tgr)
    e = gm2 & ~np.roll(gm2, 1, axis=0)
    lay.T[e] = 0.75
    edge_sh = gm2 & sh & ~np.roll(sh, -1, axis=0)
    lay.T[edge_sh] = 0.45
    for y in range(GY, H):
        u = (y - GY) / (H - GY)
        hw = 17 + u * 40
        x0 = int(GX - hw - 4 * u)
        x1 = int(GX + hw - 4 * u)
        insh = sh[y, max(0, min(W_ - 1, GX))]
        base_t = (0.2 if insh else 0.42) + 0.05 * hash01(y, 3)
        lay.rect(x0, y, x1 - x0, 1, M_SNOW, base_t)
        for tx in (-6, 6):
            xx = int(GX + tx * (1 + u * 1.6) - 4 * u)
            lay.rect(xx, y, 1, 1, M_SNOW, base_t - 0.15)
            lay.rect(xx + 1, y, 1, 1, M_SNOW, base_t + 0.3)
    cv = lay.canvas()
    cool_grade(cv, lay)
    # icicles under the lintel band, iron bands and post caps
    for x in list(range(GX - 38, GX + 38, 3)):
        if hash01(x, 10) < 0.75:
            L_ = 1 + int(hash01(x, 9) * 4)
            for k in range(L_):
                cv.px(x, BAND_Y + 4 + k, '#d4f2ff' if k < L_ - 1 else '#62aee6')
    for px0 in (GX - 39, GX + 17):
        for x in range(px0, px0 + 22, 2):
            if hash01(x, 11) < 0.6:
                L_ = 1 + int(hash01(x, 12) * 3)
                for k in range(L_):
                    cv.px(x, POST_TOP + k, '#d4f2ff' if k < L_ - 1 else '#62aee6')
    # parked sled with firewood
    sx, sy = GX + 70, GY + 5
    cv.hline(sx - 1, sx + 16, sy, '#9ca2bf')
    cv.px(sx - 2, sy - 1, '#9ca2bf')
    cv.px(sx - 3, sy - 2, '#9ca2bf')
    for k in (1, 7, 13):
        cv.vline(sx + k, sy - 3, sy - 1, '#462b37')
    cv.rect(sx, sy - 4, 16, 2, '#7a4f46')
    cv.hline(sx, sx + 15, sy - 4, '#9a6a53')
    for k in range(5):
        cv.rect(sx + 1 + k * 3, sy - 7, 3, 3, '#5e3a3d')
        cv.px(sx + 2 + k * 3, sy - 6, '#c99d72')
    for k in range(4):
        cv.rect(sx + 2 + k * 3, sy - 9, 3, 2, '#5e3a3d')
        cv.px(sx + 3 + k * 3, sy - 9, '#c99d72')
    cv.hline(sx + 1, sx + 13, sy - 10, '#b8c6f1')
    # static warm light pools
    bl = []
    for (x, y, kind) in LIGHTS_A:
        if kind == 'torch':
            bl.append((x, y, 38, 0.42))
        elif kind == 'brazier':
            bl.append((x, y + 6, 46, 0.5))
        else:
            bl.append((x, y, 16, 0.45))
    bl.append((GX, 128, 16, 0.22))
    bake_lights(cv, bl)
    return rgba(cv)


@lru_cache(maxsize=1)
def _fg_a():
    W_ = 372
    lay = Lay(W_, H)
    xs = np.arange(W_)
    gl = 169 + 4 * np.sin(xs / 31.0 + 0.5) + (noise1(W_, 7, 1800) - 0.5) * 6
    gl = gl - 12 * np.exp(-((xs - 20) / 30.0) ** 2) - 12 * np.exp(-((xs - 356) / 26.0) ** 2)
    yy = np.arange(H)[:, None]
    m = yy >= gl[None, :]
    sl = np.gradient(gl)
    nz = fbm(H, W_, 5, 1801, 3)
    t = 0.55 + np.clip(sl, -1.5, 1.5)[None, :] * 0.2 + (nz - 0.5) * 0.3 - (yy - gl[None, :]) * 0.025
    lay.put(0, 0, m, M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.97
    pine(lay, 8, 178, 124, 1810, tiers=11)
    pine(lay, 34, 178, 70, 1811, tiers=8)
    pine(lay, 360, 178, 92, 1812, tiers=9)
    for (rx, ry, rw, rh, s) in ((304, 170, 15, 8, 1820), (96, 175, 11, 6, 1821)):
        rm = emask(rw * 2 + 1, rh * 2 + 1, rw, rh, rw, rh)
        rm[rh + 2:] = False
        lay.put(rx - rw, ry - rh, rm, M_ROCK, blob_t(rm, rad=4))
    snow_tops(lay, 286, 322, 2, 3, 1822, y_lo=150)
    snow_tops(lay, 82, 110, 2, 2, 1823, y_lo=160)
    cv = lay.canvas()
    sx, sy = 70, 170
    cv.vline(sx, sy - 18, sy, '#2b1e2c')
    cv.vline(sx + 1, sy - 18, sy, '#4a2f35')
    cv.round_rect(sx - 8, sy - 22, 18, 10, '#3a2630', 1)
    cv.rect(sx - 7, sy - 21, 16, 8, '#5e3e3c')
    cv.tri((sx - 5, sy - 14), (sx, sy - 20), (sx + 5, sy - 14), '#c6d0f2')
    cv.tri((sx + 1, sy - 14), (sx + 4, sy - 18), (sx + 7, sy - 14), '#8fa1dc')
    cv.hline(sx - 7, sx + 8, sy - 23, '#e4ebff')
    cv.hline(sx - 6, sx + 7, sy - 24, '#b8c6f1')
    return rgba(cv)


def render_a(f):
    cam = cam_a(f)
    c = _sky_a().copy()
    stars(c, f, 1, 150, 110, avoid=((MOON_A[0], MOON_A[1], 22),))
    aurora(c, f, 46, 50, k=0.75, seed=5)
    c.blit(_far_a(), -int(round(cam * P_FAR)), 0)
    snowfall(c, f, 70, 101, vy=0.35, wk=0.5, color='#6e77b6')
    c.blit(_mid_a(), -int(round(cam * P_MID)), 0)
    bx = int(round(lerp(218, 272, seg(f, 72, CUT1))))
    draw_balloon(c, bx, 40, f, size='s')
    ov = -int(round(cam * P_VIL))
    vil, chim = _village_a()
    c.blit(vil, ov, 0)
    for i, (x, y) in enumerate(chim):
        smoke(c, x + ov, y - 1, f, 40 + i, n=8, h=24)
    tx, ty = TOWER_A
    c.vline(tx + ov, ty - 9, ty - 3, '#2b2a40')
    pennant(c, tx + ov, ty - 9, f, 15, length=9)
    glow_a(c.a, tx + ov, ty + 11, 9, 'amber', 0.16 * flicker(f, 3), rings=3)
    snowfall(c, f, 80, 102, vy=0.5, wk=0.8, color='#a3b2e4')
    ow = -int(round(cam * P_WALL))
    c.blit(_wall_a(), ow, 0)
    gx = GX + ow
    banner(c, gx - 33, POST_TOP + 3, f, 1, w=10, h=22, scheme=0)
    banner(c, gx + 23, POST_TOP + 3, f, 2, w=10, h=22, scheme=1)
    banner(c, gx - 102, 106, f, 3, w=9, h=22, scheme=2)
    banner(c, gx + 92, 106, f, 4, w=9, h=22, scheme=0)
    for (px_, s, L) in ((gx - 58, 11, 16), (gx + 56, 12, 15), (gx - 176, 13, 13), (gx + 166, 14, 14)):
        c.vline(px_, 76, 99, '#1c1a2b')
        c.vline(px_ + 1, 78, 99, '#2b2a40')
        c.px(px_, 75, '#e0b443')
        pennant(c, px_, 76, f, s, length=L)
    for (x, y, kind) in LIGHTS_A:
        s = x * 7 + y
        if kind == 'torch':
            wall_torch(c, x + ow, y, f, s)
            torch_light(c, x + ow, y - 3, f, s, r=14, k=0.22)
        elif kind == 'brazier':
            brazier(c, x + ow, GY + 2, f, s)
            torch_light(c, x + ow, y - 2, f, s, r=18, k=0.25)
        else:
            lx = x + ow
            c.vline(lx, 82, 84, '#1c1a2b')
            c.rect(lx - 1, 85, 3, 4, '#2b2a40')
            c.px(lx, 86, 'butter' if flicker(f, s) > 0.8 else 'gold')
            c.px(lx, 87, 'gold')
            torch_light(c, lx, 87, f, s, r=8, k=0.2)
    glow_a(c.a, gx, 126, 8, 'amber', 0.1 * flicker(f, 9), rings=3, sx=0.5)
    for i in range(16):
        x = int(hash01(i, 71) * 340) + ow
        y = GY + 10 + int(hash01(i, 72) * 12)
        if (f // 3 + i * 5) % 26 < 3:
            c.px(x, y, 'white')
    snowfall(c, f, 60, 103, vy=0.8, wk=1.1, color='#e4ebff', size=2, alt='#b8c6f1')
    c.blit(_fg_a(), -int(round(cam * P_FG)), 0)
    snowfall(c, f, 26, 104, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
    callout(c, tx + ov + 4, ty + 24, 'YAKYAKISTAN', f, 90, 162, side='right', length=12)
    callout(c, gx + 5, GY - 14, 'LOG GATES', f, 108, 172, side='down', length=22)
    return c


# =====================================================================================  SHOT B
MOON_B = (268, 30)
GB = 142                                  # village ground line
HALL_X, HALL_W, HALL_WH, HALL_RH = 56, 104, 34, 48
P_BFAR, P_BCLOUD, P_BSLOPE, P_BVIL, P_BFG = 0.08, 0.2, 0.85, 1.0, 1.3
AVA = 216                                 # avalanche hit (bar line)
SLIDE_PTS = [(244, 141), (262, 144), (284, 147), (306, 150), (318, 154), (310, 158), (288, 161), (274, 164),
             (280, 168), (304, 171), (334, 173), (358, 176), (374, 181), (384, 188)]
RUGS_B = [  # (x, rope y, w, h, scheme)
    (34, 115, 8, 12, 0), (43, 116, 8, 13, 1), (213, 121, 9, 12, 2), (223, 121, 7, 11, 0),
]


def cam_b(f):
    return 28 * cruise(seg(f, CUT1, CUT2))


@lru_cache(maxsize=1)
def _sky_b():
    c = Canvas(W, H, '#0b0a20')
    c.gradient(0, 0, W, 130, ['#09081c', '#0f0f2c', '#16183c', '#1f234e', '#2a2f62', '#3a3f78', '#50548e'],
               soft=0.9)
    c.rect(0, 130, W, 50, '#50548e')
    moon(c, *MOON_B, r=12)
    return c


@lru_cache(maxsize=1)
def _far_b():
    lay = Lay(340, H)
    for ax, ay, lw, rw, s in ((20, 92, 40, 36, 51), (96, 100, 34, 40, 52), (176, 86, 38, 44, 53),
                              (250, 98, 30, 36, 54), (318, 90, 40, 40, 55)):
        mountain(lay, ax, ay, 132, lw, rw, s, M_FAR, lit=(0.55, 0.92), shd=(0.14, 0.4), jag=3, rock=0.16)
    cv = lay.canvas()
    a = cv.a
    km = (a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2])
    for y in range(104, 132):
        lv = clamp((y - 104) / 28.0) * 0.85
        thr = bayer(1, cv.w, 0, y)[0]
        a[y][~km[y] & (thr < lv)] = col('#50548e')
    return rgba(cv)


@lru_cache(maxsize=1)
def _cloudsea():
    w = 420
    lay = Lay(w, H)
    rr = rng(61)
    for row, (y0, amp, r0, tb) in enumerate(((114, 3, 9, 0.62), (121, 4, 12, 0.7), (130, 5, 15, 0.78))):
        x = -10
        while x < w + 10:
            r = r0 * (0.6 + 0.7 * rr.random())
            cy = y0 + amp * (rr.random() - 0.5) * 2 + r * 0.55
            bm = emask(int(2 * r + 3), int(2 * r + 3), r + 1, r + 1, r, r * 0.75)
            t = blob_t(bm, light=(0.55, -0.85), rad=5, amb=-0.05) * 0.8 + tb * 0.25
            lay.put(int(x - r - 1), int(cy - r - 1), bm, M_CLOUD, t)
            x += r * (1.1 + 0.6 * rr.random())
        lay.rect(0, y0 + 10, w, H - y0 - 10, M_CLOUD, 0.3 + 0.1 * row)
    cv = lay.canvas()
    return rgba(cv)


@lru_cache(maxsize=1)
def _slide_line():
    """Dense centre-line samples of the ice slide (slope-layer coords) + cumulative length."""
    pts = np.array(SLIDE_PTS, np.float32)
    out = []
    for i in range(len(pts) - 1):
        p0 = pts[max(0, i - 1)]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[min(len(pts) - 1, i + 2)]
        for k in range(12):
            t = k / 12
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
    out.append(pts[-1])
    out = np.array(out)
    seglen = np.sqrt(((out[1:] - out[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seglen)])
    return out, cum


@lru_cache(maxsize=1)
def _slope_b():
    w = 400
    lay = Lay(w, H)
    xs = np.arange(w)
    ridge = np.where(xs < 248, 140.0, 140 + (xs - 248) * 0.13) + (noise1(w, 6, 71) - 0.5) * 3
    yy = np.arange(H)[:, None]
    m = yy >= ridge[None, :]
    nz = fbm(H, w, 7, 72, 3)
    rip = np.sin((xs[None, :] * 0.35 + yy * 1.1) + nz * 5) > 0.75
    t = 0.56 + (nz - 0.5) * 0.3 - (yy - ridge[None, :]) * 0.004 - rip * 0.12
    lay.put(0, 0, m, M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.96
    rr = rng(73)
    for i in range(5):           # flat rock ledges poking through the snow
        rx = int(rr.integers(262, 395))
        ry = int(ridge[min(w - 1, rx)] + rr.integers(8, 36))
        rw = int(rr.integers(5, 10))
        rm = pmask(2 * rw + 2, 4, [(0, 3.5), (2, 0.5), (2 * rw, 1), (2 * rw + 2, 3.5)])
        lay.put(rx - rw, ry - 2, rm, M_STONE, 0.35 + 0.3 * (np.mgrid[0:4, 0:2 * rw + 2][1] > rw))
        lay.put(rx - rw, ry - 3, rm & ~np.roll(rm, 1, axis=0), M_SNOW, 0.9)
    # the ice slide: banked snow walls + glossy ribbon
    line, cum = _slide_line()
    L = cum[-1]
    yy2, xx2 = np.mgrid[0:H, 0:w]
    best = np.full((H, w), 99.0, np.float32)
    s_at = np.zeros((H, w), np.float32)
    for (px, py), cl in zip(line, cum):
        x0, x1 = int(px) - 8, int(px) + 9
        y0, y1 = int(py) - 8, int(py) + 9
        sub = (xx2[y0:y1, x0:x1] - px) ** 2 + ((yy2[y0:y1, x0:x1] - py) * 1.5) ** 2
        d = np.sqrt(sub)
        bb = best[y0:y1, x0:x1]
        upd = d < bb
        bb[upd] = d[upd]
        s_at[y0:y1, x0:x1][upd] = cl / L
    width = 4.4 - 1.8 * s_at
    bank = best <= width + 2.0
    ice = best <= width
    lay.put(0, 0, bank, M_SNOW, 0.9)
    tice = 0.45 + 0.35 * (1 - best / np.maximum(width, 0.5)) + (nz - 0.5) * 0.2
    lay.put(0, 0, ice, M_ICE, tice)
    core = (best <= 0.7) & ice
    lay.put(0, 0, core, M_ICE, 0.97)
    lower = bank & ~ice & (yy2 > np.roll(np.where(ice, yy2, -1), 0, axis=0))
    cv = lay.canvas()
    return rgba(cv), ice


@lru_cache(maxsize=1)
def _village_b():
    w = 360
    lay = Lay(w, H)
    # plateau ground with the cornice edge on the right
    xs = np.arange(w)
    gl = GB + 1.2 * np.sin(xs / 17.0) + (noise1(w, 6, 81) - 0.5) * 2
    edge = 250
    yy = np.arange(H)[:, None]
    cut = np.clip(xs - edge, 0, None) ** 1.6 * 0.9
    gm = (yy >= gl[None, :]) & (yy <= (H + 20 - cut)[None, :]) & (xs[None, :] < edge + 18)
    nz = fbm(H, w, 6, 82, 3)
    t = 0.6 + (nz - 0.5) * 0.25 - (yy - GB) * 0.006
    lay.put(0, 0, gm, M_SNOW, t)
    e = gm & ~np.roll(gm, 1, axis=0)
    lay.T[e] = 0.95
    er = gm & ~np.roll(gm, -1, axis=1)
    lay.T[er] = 0.85
    # hut left
    infos = {}
    infos['hutL'] = hut(lay, 0, GB, 30, 20, 22, 820, door_x=19, windows=((4, 16, 6, 7),), chimney=(5, 5),
                        scheme=1)
    # great hall
    hx, hw, hwh, hrh = HALL_X, HALL_W, HALL_WH, HALL_RH
    stone_base(lay, hx, GB - 8, hw, 8, 830)
    log_wall(lay, hx, GB - hwh, hw, hwh - 8, 840, log_h=4)
    ax, ay = gable_roof(lay, hx, GB - hwh, hw, hrh, 850, ov=5, snow_th=5, snow_right=False)
    # gable face boards + oculus + painted sun band
    gm2 = pmask(w, H, [(hx + 4, GB - hwh), (ax, ay + 9), (hx + hw - 4, GB - hwh)])
    yy2, xx2 = np.mgrid[0:H, 0:w]
    tg = 0.3 + 0.25 * ((xx2 % 4) >= 2) + grain_v(H, w, 851, 0.2)
    tg = np.where(xx2 % 4 == 0, 0.08, tg)
    lay.put(0, 0, gm2, M_WOOD, tg)
    for k, mt in enumerate((M_RED, M_YEL, M_BLUE)):
        bandm = gm2 & (yy2 == GB - hwh - 2 - k)
        lay.put(0, 0, bandm, mt, 0.6)
    om = emask(15, 15, 7, 7, 7, 7)
    oy_ = ay + 14
    lay.put(ax - 7, oy_, om, M_YEL, blob_t(om, rad=3) * 0.8)
    om2 = emask(15, 15, 7, 7, 5, 5)
    lay.put(ax - 7, oy_, om2, M_WARM, 0.45 + 0.4 * np.mgrid[0:15, 0:15][0] / 15)
    lay.put(ax - 7, oy_, om2 & (np.abs(np.mgrid[0:15, 0:15][1] - 7) < 1), M_WOOD, 0.2)
    lay.put(ax - 7, oy_, om2 & (np.abs(np.mgrid[0:15, 0:15][0] - 7) < 1), M_WOOD, 0.2)
    # carved bargeboards (zigzag) along both roof edges
    for k in range(hw // 2 + 5):
        for sgn in (-1, 1):
            bx = ax + sgn * k
            by = ay + int(k * hrh / (hw / 2 + 5)) + 3
            if (k // 2) % 2 == 0:
                lay.rect(bx, by, 1, 2, M_RED, 0.6)
            else:
                lay.rect(bx, by, 1, 2, M_YEL, 0.7)
    horn_crescent(lay, ax - 2, ay + 2, 6.5, -1, 860, th0=2.6, th1=0.7, collar=False)
    horn_crescent(lay, ax + 2, ay + 2, 6.5, 1, 861, th0=2.6, th1=0.7, collar=False)
    # door: big round-topped, painted frame, steps
    dw, dh = 18, 26
    dx = ax - dw // 2 + 1
    door(lay, dx, GB - dh - 2, dw, dh, 870, scheme=0)
    fm = np.zeros((dh + 4, dw + 6), bool)
    fm[:, :] = True
    lay.rect(dx - 3, GB - dh - 1, 2, dh - 1, M_BLUE, 0.6)
    lay.rect(dx + dw + 1, GB - dh - 1, 2, dh - 1, M_BLUE, 0.6)
    for yv in range(GB - dh, GB - 2, 3):
        lay.rect(dx - 3, yv, 2, 1, M_YEL, 0.8)
        lay.rect(dx + dw + 1, yv, 2, 1, M_YEL, 0.8)
    lay.rect(dx - 4, GB - 2, dw + 8, 2, M_STONE, 0.6)
    lay.rect(dx - 6, GB, dw + 12, 2, M_STONE, 0.45)
    for wx in (hx + 9, hx + 26, hx + hw - 32, hx + hw - 15):
        window(lay, wx, GB - 26, 7, 10, wx, scheme=(wx // 7) % 3)
    # chimney on the left slope
    cx_ = hx + 18
    yr = (GB - hwh) - hrh * (1 - abs(cx_ + 2 - ax) / (hw / 2 + 5))
    ctop = int(yr - 7)
    chh = int(yr + 4 - ctop)
    lay.put(cx_, ctop, np.ones((chh, 6), bool), M_STONE, cyl(6, chh, 0.7, 0.15, 0.75))
    lay.rect(cx_ - 1, ctop - 1, 8, 2, M_STONE, 0.55)
    lay.rect(cx_ - 1, ctop - 3, 8, 2, M_SNOW, 0.85)
    infos['hall_chim'] = (cx_ + 3, ctop - 3)
    infos['hall_apex'] = (ax, ay)
    # firewood stack under a lean-to (right of the hall)
    fx0 = hx + hw + 4
    lay.rect(fx0, GB - 15, 1, 15, M_WOOD, 0.3)
    lay.rect(fx0 + 14, GB - 13, 1, 13, M_WOOD, 0.3)
    lay.poly([(fx0 - 2, GB - 15), (fx0 + 17, GB - 12), (fx0 + 17, GB - 10), (fx0 - 2, GB - 13)], M_ROOF, 0.45)
    snow_tops(lay, fx0 - 2, fx0 + 18, 2, 1, 880, y_lo=GB - 20, y_hi=GB - 8)
    for r in range(4):
        for k in range(4 - (r == 3)):
            em = emask(3, 3, 1, 1, 1, 1)
            lay.put(fx0 + 1 + k * 3 + (r % 2), GB - 3 - r * 3, em, M_END, 0.5 + 0.3 * ((k + r) % 2))
    # hut right
    infos['hutR'] = hut(lay, 186, GB, 24, 18, 18, 890, door_x=3, windows=((14, 14, 6, 6),), chimney=(15, 4),
                        scheme=2)
    # rug-line posts and ropes
    for (x0, x1, y) in ((31, 53, 114), (211, 232, 120)):
        for px in (x0, x1):
            lay.rect(px, y - 1, 2, GB - y + 1, M_WOOD, 0.45)
            lay.rect(px - 1, y - 2, 4, 1, M_SNOW, 0.9)
        cvx = np.arange(x0 + 2, x1)
        sag = (1 - ((cvx - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2) * 2
        for xx_, sg in zip(cvx, sag):
            lay.rect(int(xx_), int(y + sg), 1, 1, M_END, 0.35)
    # slide start arch at the plateau edge
    ax2 = 237
    for px in (ax2, ax2 + 12):
        lay.rect(px, GB - 18, 2, 18, M_WOOD, 0.5)
    hlog(lay, ax2 - 2, GB - 21, 17, 3, 895)
    for k in range(15):
        lay.rect(ax2 - 1 + k, GB - 18, 1, 1, (M_RED, M_YEL, M_BLUE)[(k // 2) % 3], 0.6)
    snow_tops(lay, ax2 - 3, ax2 + 16, 2, 1, 896, y_lo=GB - 26, y_hi=GB - 17)
    # lamp posts along the street
    for lx in (182, 233):
        lay.rect(lx, GB - 16, 1, 16, M_WOOD, 0.3)
        lay.rect(lx, GB - 16, 4, 1, M_WOOD, 0.45)
    snow_tops(lay, 0, 250, 2, 1, 897, y_lo=0, y_hi=GB - 1, sel=(M_ROOF,))
    cv = lay.canvas()
    cool_grade(cv, lay, mul=(0.66, 0.7, 0.93))
    bl = [(ax - 12, GB - 18, 34, 0.42), (ax + 13, GB - 18, 34, 0.42), (ax, oy_ + 7, 18, 0.25)]
    for wx in (hx + 9, hx + 26, hx + hw - 32, hx + hw - 15):
        bl.append((wx + 3, GB - 21, 13, 0.22))
    bl += [(4 + 3, GB - 12, 12, 0.2), (184 + 13 + 3, GB - 10, 12, 0.2)]
    for lx in (182, 233):
        bl.append((lx + 3, GB - 13, 16, 0.3))
    bake_lights(cv, bl)
    return rgba(cv), infos


@lru_cache(maxsize=1)
def _slab_b():
    """The roof snow slab on the hall's right slope (village coords): sprite, origin, slope unit, eave."""
    hx, hw, hwh, hrh = HALL_X, HALL_W, HALL_WH, HALL_RH
    ax = hx + hw / 2 - 0.5
    ay = GB - hwh - hrh
    ex, ey = hx + hw + 5 - 1, GB - hwh + 1
    p0 = np.array([ax + 3, ay + 2.5])
    p1 = np.array([ex + 0.5, ey + 0.5])
    dv = p1 - p0
    L = float(np.hypot(*dv))
    td = dv / L
    nrm = np.array([td[1], -td[0]])           # outward (up-right)
    yy, xx = np.mgrid[0:H, 0:W]
    rx, ry = xx + 0.5 - p0[0], yy + 0.5 - p0[1]
    u = (rx * td[0] + ry * td[1]) / L
    d = rx * nrm[0] + ry * nrm[1]
    top = 2.2 + 1.6 * np.interp(u, np.linspace(-0.1, 1.2, 40), noise1(40, 3, 871) * 1.0) - 0.8 * (u > 1.0)
    m = (u >= 0) & (u <= 1.05) & (d >= -4.5) & (d <= top)
    nz = fbm(H, W, 4, 870, 2)
    t = 0.62 + 0.32 * np.clip((d + 4.5) / (top + 4.5), 0, 1) + (nz - 0.5) * 0.2
    t = np.where(d > top - 1.0, 0.99, t)
    t = np.where(d < -3.5, 0.42, t)
    lay = Lay(W, H)
    lay.put(0, 0, m, M_SNOW, t)
    cv = lay.canvas()
    ys, xs_ = np.nonzero(m)
    x0, y0 = xs_.min(), ys.min()
    spr = rgba(cv)[y0:ys.max() + 1, x0:xs_.max() + 1]
    return spr, int(x0), int(y0), float(td[0]), float(td[1]), int(ex), int(ey), (float(p0[0]), float(p0[1])), L


def slab_d(f):
    """Distance (px) the slab has slid along the roof."""
    if f < AVA:
        return 1 if f >= AVA - 6 else 0
    t = f - AVA
    return 1 + int(1.8 * t + 0.85 * t * t)


def puff(c, x, y, r, lv, body='#d6def8', sh='#8f9dd6', hi='#ffffff'):
    """Solid snow-powder billow (dither-fades when lv < 1)."""
    if r < 0.8 or lv <= 0:
        return
    dith_ell(c, x - r * 0.25, y + r * 0.25, r, r * 0.8, sh, lv)
    dith_ell(c, x, y, r * 0.92, r * 0.74, body, lv)
    dith_ell(c, x + r * 0.3, y - r * 0.3, r * 0.5, r * 0.4, hi, lv)


def draw_avalanche(c, f, ov, oy):
    spr, x0, y0, ux, uy, ex, ey, p0, L = _slab_b()
    d = slab_d(f)
    sx = x0 + int(round(d * ux)) + ov
    sy = y0 + int(round(d * uy)) + oy
    eave = ex + ov
    gy = GB + oy
    if d < L + 6:
        cut = eave + 3 - sx
        if cut > 0:
            c.blit(spr[:, :min(spr.shape[1], cut)], sx, sy)
    if AVA - 14 <= f < AVA:                    # anticipation: a crack and a trickle off the eave
        k = (f - AVA + 14) / 14.0
        cx0 = int(p0[0] + ux * 14) + ov
        cy0 = int(p0[1] + uy * 14) + oy
        c.line(cx0, cy0 - 3, cx0 + 1 + int(k * 2), cy0 + 1, '#5262a8')
        for j in range(2 + int(k * 3)):
            tt = (f * 2 + j * 7) % 22
            c.px(eave + 2 + (j % 3), ey + oy + 1 + tt, '#e4ebff')
    if f < AVA:
        return
    t_all = f - AVA
    # break-off plume rising along the slab's upper edge (on the hit)
    if t_all < 16:
        t = t_all / 16.0
        for i in range(10):
            uu = 0.12 + 0.088 * i
            bx = p0[0] + ux * (uu * L + d * 0.6) + ov + 2
            by = p0[1] + uy * (uu * L + d * 0.6) + oy - 3 - t * (8 + 6 * hash01(i, 2))
            r = 3.2 + t * 6.5 * (0.7 + 0.3 * hash01(i, 3))
            lv = 1.0 if t < 0.45 else (1 - t) / 0.55
            puff(c, bx, by, r, lv)
    # powder pouring off the eave
    if 2 <= t_all < 16:
        k = 1.0 if t_all < 9 else (16 - t_all) / 7.0
        for yv in range(ey + oy + 1, gy, 2):
            w_ = 3 + (yv - ey - oy) // 6
            puff(c, eave + 3 + int(math.sin(yv * 0.5 + f) * 1.2), yv, w_ * 0.8, k * 0.9, body='#c7d1f4',
                 sh='#7e8cc8', hi='#eef0ff')
    # impact billow + spray
    if 4 <= t_all < 18:
        t = (t_all - 4) / 14.0
        mx = eave + 6
        for i in range(8):
            a = math.pi * (1.05 + 0.9 * hash01(i, 41))
            rr_ = (4 + 22 * t) * (0.45 + 0.55 * hash01(i, 42))
            px = mx + math.cos(a) * rr_ * 1.4
            py = gy - 4 + math.sin(a) * rr_ * 0.9
            r = 3 + 8 * t * (0.6 + 0.4 * hash01(i, 43))
            lv = 1.0 if t < 0.5 else (1 - t) / 0.5
            puff(c, px, py, r, lv)
        for i in range(16):
            sx2 = mx + int((hash01(i, 51) - 0.5) * 50 * t)
            sy2 = gy - 4 - int(hash01(i, 52) * 30 * t) + int(16 * t * t)
            if (f + i) % 3:
                c.px(sx2, sy2, 'white')
    # the landed mound stays
    if t_all >= 5:
        mh = min(6, (t_all - 4))
        mx = eave + 5
        dith_ell(c, mx, gy - 1, 8 + mh // 2, mh, '#6a7cc2', 1)
        dith_ell(c, mx - 1, gy - 2, 7 + mh // 2, max(1, mh - 1), '#b8c6f1', 1)
        dith_ell(c, mx - 2, gy - 2 - mh // 2, 4, max(1, mh // 2), '#e4ebff', 1)


@lru_cache(maxsize=None)
def _rug_rgba(w, h, scheme):
    """Shaggy rug folded over a drying line: stripes, zigzag band, ragged sides and a tassel fringe."""
    ramp = RA((RUG_R, RUG_B, RUG_Y)[scheme])
    acc = RA((RUG_Y, RUG_R, RUG_B)[scheme])
    cv = new(w, h)
    nz = vnoise(h, w * 3, 2, 400 + scheme)[:, ::3]
    body = h - 3
    for y in range(body):
        stripe = y in (3, 4, body - 4, body - 3)
        for x in range(w):
            k = 1 + int(nz[y, x] > 0.5) + (x >= w - 2)
            if x == 0:
                k = 0
            cv.px(x, y, tuple((acc if stripe else ramp)[min(3, k)]))
    for x in range(1, w - 1):
        cv.px(x, body // 2 + (1 if (x // 2) % 2 else 0), tuple(acc[3]))
        cv.px(x, body // 2 + (0 if (x // 2) % 2 else 1), tuple(acc[1]))
    cv.hline(0, w - 1, 0, tuple(ramp[3]))
    cv.hline(0, w - 1, 1, tuple(ramp[0]))
    for y in range(2, body):
        if hash01(y, scheme, 1) < 0.3:
            cv.px(0, y, KEY)
        if hash01(y, scheme, 2) < 0.3:
            cv.px(w - 1, y, KEY)
    for x in range(w):
        Lf = int(hash01(x, scheme, 3) * 3.99)
        for k in range(Lf):
            cv.px(x, body + k, tuple(ramp[1] if k < Lf - 1 else ramp[0]))
    return rgba(cv)


def rug(c, x, y, f, seed, w, h, scheme):
    spr = _rug_rgba(w, h, scheme)
    g = gust(f)
    for r in range(h):
        u = r / (h - 1)
        dx = int(round((math.sin(f * 0.17 + seed - r * 0.2) * (0.3 + 0.6 * g) + g * 0.9) * u * u))
        c.blit(spr[r:r + 1], x + dx, y + r)


@lru_cache(maxsize=1)
def _fg_b():
    W_ = 400
    lay = Lay(W_, H)
    xs = np.arange(W_)
    gl = 166 + 4 * np.sin(xs / 27.0) + (noise1(W_, 7, 91) - 0.5) * 5
    gl = gl + np.clip((xs - 210) * 0.3, 0, 30)
    yy = np.arange(H)[:, None]
    m = yy >= gl[None, :]
    nz = fbm(H, W_, 5, 92, 3)
    sl = np.gradient(gl)
    t = 0.5 + np.clip(sl, -1.5, 1.5)[None, :] * 0.2 + (nz - 0.5) * 0.3 - (yy - gl[None, :]) * 0.02
    lay.put(0, 0, m, M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.97
    for k, px in enumerate(range(8, 200, 16)):     # fence
        top = int(gl[px]) - 12
        lay.rect(px, top, 3, int(gl[px]) - top + 2, M_WOOD, 0.4)
        lay.rect(px + 2, top, 1, int(gl[px]) - top + 2, M_WOOD, 0.75)
        lay.rect(px - 1, top - 2, 5, 2, M_SNOW, 0.9)
    snow_tops(lay, 0, 210, 0, 0, 93)
    cv = lay.canvas()
    cool_grade(cv, lay, mul=(0.6, 0.64, 0.9))
    for px in range(8, 184, 16):
        y0 = int(gl[px]) - 9
        y1 = int(gl[px + 16]) - 9
        for xx in range(px + 3, px + 16):
            u = (xx - px - 3) / 13.0
            yv = int(y0 + (y1 - y0) * u + math.sin(u * math.pi) * 2)
            cv.px(xx, yv, '#4b2f36')
    return rgba(cv)


def render_b(f):
    cam = cam_b(f)
    sx_, sy_ = (0, 0)
    if AVA <= f < AVA + 9:
        sx_, sy_ = fx.shake(f, 2 if f < AVA + 4 else 1, seed=19)
    c = _sky_b().copy()
    stars(c, f, 2, 160, 110, avoid=((MOON_B[0], MOON_B[1], 20),))
    aurora(c, f, 64, 44, k=0.55, seed=8, x_shift=120)
    c.blit(_far_b(), -int(round(cam * P_BFAR)), 0)
    # light of the Crystal Empire far away
    ce = (300 - int(round(cam * P_BFAR)), 112)
    glow_a(c.a, ce[0], ce[1], 9, '#f39ce8', 0.25 + 0.08 * math.sin(f * 0.2), rings=3)
    fx.sparkle(c, ce[0], ce[1] - 2, f, '#b4e6ff', period=30, size=2)
    co = -int(round(cam * P_BCLOUD + (f - CUT1) * 0.12))
    c.blit(_cloudsea(), co, 0)
    glow_a(c.a, MOON_B[0], 126, 16, '#c9d0f2', 0.18, rings=4, sx=0.35)
    bx = int(round(lerp(284, 314, seg(f, CUT1, CUT2))))
    draw_balloon(c, bx, 98 - int(8 * seg(f, CUT1, CUT2)), f, size='s')
    snowfall(c, f, 60, 201, vy=0.35, wk=0.5, color='#6e77b6')
    os_ = -int(round(cam * P_BSLOPE)) + sx_
    slope, ice = _slope_b()
    c.blit(slope, os_, sy_)
    line, cum = _slide_line()
    # glints running down the slide (moonlight on the ice)
    Lt = cum[-1]
    for k in range(3):
        pos = ((f * 2.2 + k * Lt / 3) % Lt)
        i = int(np.searchsorted(cum, pos))
        i = min(i, len(line) - 1)
        gx_, gy_ = int(line[i][0]) + os_, int(line[i][1]) + sy_
        c.px(gx_, gy_, 'white')
        c.px(gx_ - 1, gy_, '#d4f2ff')
    # lanterns along the slide
    for j, idx in enumerate((10, 40, 70, 100, 130)):
        if idx < len(line):
            lx, ly = int(line[idx][0]) + os_ + 5, int(line[idx][1]) + sy_ - 2
            c.vline(lx, ly - 4, ly + 2, '#2b2a40')
            c.px(lx, ly - 5, 'butter' if flicker(f, j) > 0.8 else 'gold')
            glow_a(c.a, lx, ly - 5, 7, 'amber', 0.25 * flicker(f, j), rings=3)
    ov = -int(round(cam * P_BVIL)) + sx_
    vil, info = _village_b()
    c.blit(vil, ov, sy_)
    for i, key in enumerate(('hutL', 'hutR')):
        ch = info[key]['chimney']
        smoke(c, ch[0] + ov, ch[1] + sy_, f, 70 + i, n=8, h=26)
    hc = info['hall_chim']
    smoke(c, hc[0] + ov, hc[1] + sy_, f, 77, n=9, h=32)
    ax, ay = info['hall_apex']
    for (tx_, s) in ((ax - 12, 1), (ax + 13, 2)):
        wall_torch(c, tx_ + ov, GB - 22 + sy_, f, 60 + s)
        torch_light(c, tx_ + ov, GB - 25 + sy_, f, 60 + s, r=14, k=0.22)
    for j, lx in enumerate((182, 233)):
        x_ = lx + 3 + ov
        c.vline(x_, GB - 15 + sy_, GB - 14 + sy_, '#1c1a2b')
        c.rect(x_ - 1, GB - 13 + sy_, 3, 3, '#2b2a40')
        c.px(x_, GB - 12 + sy_, 'butter' if flicker(f, j + 5) > 0.8 else 'gold')
        glow_a(c.a, x_, GB - 12 + sy_, 8, 'amber', 0.2 * flicker(f, j + 5), rings=3)
    for (x, y, w, h, sch) in RUGS_B:
        rug(c, x + ov, y + sy_, f, x, w, h, sch)
    # flags on the slide arch
    pennant(c, 237 + ov, GB - 25 + sy_, f, 31, length=8)
    pennant(c, 249 + ov, GB - 25 + sy_, f, 32, length=8, cols=('#3f73cc', '#e0b443', '#c23f55'))
    c.vline(237 + ov, GB - 25 + sy_, GB - 21 + sy_, '#2b2a40')
    c.vline(249 + ov, GB - 25 + sy_, GB - 21 + sy_, '#2b2a40')
    draw_avalanche(c, f, ov, sy_)
    snowfall(c, f, 70, 202, vy=0.6, wk=0.9, color='#b8c6f1')
    for i in range(14):
        x = int(hash01(i, 171) * 230) + ov
        y = GB + 4 + int(hash01(i, 172) * 18)
        if (f // 3 + i * 5) % 26 < 3:
            c.px(x, y, 'white')
    c.blit(_fg_b(), -int(round(cam * P_BFG)) + sx_, sy_)
    snowfall(c, f, 26, 204, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
    callout(c, ax - 20 + ov, ay + 22, 'GREAT HALL', f, 180, 252, side='up', length=30)
    li = 64
    callout(c, int(line[li][0]) + os_, int(line[li][1]) - 1, 'ICE SLIDE', f, 198, 258, side='up', length=34)
    return c


# =====================================================================================  SHOT C
MOON_C = (284, 46)
EVER = (150, 36)                          # Everhoof apex (final framing, mountain layer)
TILT = 30
P_CSKY, P_CFAR, P_CEVER, P_CMID, P_CIGL, P_CFG = 0.05, 0.15, 0.3, 0.6, 0.85, 1.2
IGLOOS = [(150, 154, 20, 15), (200, 149, 11, 8), (266, 156, 16, 12)]     # (cx, base, r, h) igloo layer
SNOWLAMPS = [(112, 158), (180, 156), (226, 155), (300, 159)]
CAVE = (50, 133)                          # cave mouth centre-bottom (mid layer)


def cam_c(f):
    cy = TILT * (1 - cruise(seg(f, CUT2, 356)))
    cx = 14 * cruise(seg(f, CUT2, 360))
    return cx, cy


@lru_cache(maxsize=1)
def _sky_c():
    c = Canvas(W, H + 10, '#0b0a20')
    c.gradient(0, 0, W, 150, ['#07061a', '#0c0c28', '#13153a', '#1c204c', '#272c60', '#363b74', '#4a4e88'],
               soft=0.9)
    c.rect(0, 150, W, 40, '#4a4e88')
    moon(c, *MOON_C, r=11)
    return c


@lru_cache(maxsize=1)
def _far_c():
    lay = Lay(340, H + 10)
    for ax, ay, lw, rw, s in ((10, 84, 50, 44, 121), (300, 80, 46, 50, 122), (250, 100, 40, 40, 123),
                              (60, 104, 40, 40, 124)):
        mountain(lay, ax, ay, 150, lw, rw, s, M_FAR, lit=(0.5, 0.85), shd=(0.12, 0.36), jag=3, rock=0.15)
    cv = lay.canvas()
    a = cv.a
    km = (a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2])
    for y in range(110, 150):
        lv = clamp((y - 110) / 40.0) * 0.8
        thr = bayer(1, cv.w, 0, y)[0]
        a[y][~km[y] & (thr < lv)] = col('#4a4e88')
    return rgba(cv)


@lru_cache(maxsize=1)
def _ever_c():
    lay = Lay(340, H + 20)
    ax, ay = EVER
    mountain(lay, 34, 100, 178, 70, 60, 132, M_MIDS, lit=(0.5, 0.86), shd=(0.12, 0.38), jag=3, rock=0.14)
    mountain(lay, 292, 96, 178, 60, 70, 133, M_MIDS, lit=(0.5, 0.88), shd=(0.12, 0.38), jag=3, rock=0.14)
    m = mountain(lay, ax, ay, 178, 168, 160, 131, M_MIDS, lit=(0.6, 0.99), shd=(0.14, 0.44), jag=4.5,
                 rock=0.15, div=-0.2, div_jag=16)
    # secondary aretes: moonlit crests on the shadow face, dark gullies on the lit face
    rr = rng(134)
    cv = Canvas(340, H + 20, (0, 0, 0))
    for k in range(5):
        y0 = ay + 14 + k * 20 + int(rr.integers(-3, 4))
        x0 = ax - int((y0 - ay) * 0.2) - 2
        pts = [(x0, y0)]
        for j in range(2):
            px_, py_ = pts[-1]
            pts.append((px_ - int(rr.integers(3, 7)), py_ + int(rr.integers(3, 6))))
        for (p0, p1) in zip(pts, pts[1:]):
            cv.line(p0[0], p0[1], p1[0], p1[1], (255, 0, 0))
        x1 = ax + int((y0 - ay) * 0.1) + int(rr.integers(6, 20))
        pts = [(x1, y0 + 4)]
        for j in range(4):
            px_, py_ = pts[-1]
            pts.append((px_ + int(rr.integers(2, 6)), py_ + int(rr.integers(5, 10))))
        for (p0, p1) in zip(pts, pts[1:]):
            cv.line(p0[0], p0[1], p1[0], p1[1], (0, 255, 0))
    lit_l = (cv.a[..., 0] > 0) & (lay.M == M_MIDS)
    gul = (cv.a[..., 1] > 0) & (lay.M == M_MIDS)
    lay.T[lit_l] = np.maximum(lay.T[lit_l], 0.55)
    lay.T[gul] = lay.T[gul] - 0.22
    cvm = lay.canvas()
    a = cvm.a
    km = (a[..., 0] == KEY[0]) & (a[..., 1] == KEY[1]) & (a[..., 2] == KEY[2])
    glow_a(a, ax + 6, ay + 12, 36, '#7d88cc', 0.2, rings=4)       # summit catching moonlight
    for y in range(118, 178):
        lv = clamp((y - 118) / 60.0) * 0.75
        thr = bayer(1, cvm.w, 0, y)[0]
        a[y][~km[y] & (thr < lv)] = col('#3c4380')
    a[km] = KEY
    return rgba(cvm)


def plume(c, f, ax, ay):
    """Snow plume streaming off the summit with the wind."""
    for i in range(14):                     # soft body of the plume
        per = 80
        t = ((f + i * per / 14 + hash01(i, 11) * 10) % per) / per
        x = ax + 3 + t * (54 + 20 * gust(f))
        y = ay + 3 + t * 9 + math.sin(t * 5 + i) * 1.5
        r = 1.5 + t * 4.5
        dith_ell(c, x, y, r, r * 0.55, '#9aa6dc', (1 - t) * 0.55)
    for i in range(70):
        per = 70
        t = ((f * (1.0 + 0.4 * hash01(i, 2)) + i * per / 70 + hash01(i, 1) * 20) % per) / per
        x = ax + 2 + t * (60 + 30 * gust(f)) + math.sin(t * 6 + i) * 2
        y = ay + 2 + t * 10 + (hash01(i, 3) - 0.5) * t * 14
        lv = (1 - t) ** 1.1
        if hash01(i, f // 3) < lv:
            c.px(int(x), int(y), 'white' if t < 0.3 else ('#cfd5f2' if t < 0.6 else '#8b94cc'))
            if t < 0.35:
                c.px(int(x) + 1, int(y), '#cfd5f2')


@lru_cache(maxsize=1)
def _mid_c():
    w, hh = 340, H + 40
    lay = Lay(w, hh)
    xs = np.arange(w)
    yy = np.arange(hh)[:, None]
    # blue ice cliff (behind the snow ridge) with the cave
    cw = 112
    xc = np.arange(cw)
    top = 98 + (noise1(cw, 5, 143) - 0.5) * 14 + ((xc - 50) / 50.0) ** 2 * 16
    top = top + (vnoise(1, cw, 2, 148)[0] - 0.5) * 4
    cm = (yy >= top[None, :]) & (yy <= 150)
    cy_, cx_ = np.mgrid[0:hh, 0:cw]
    pts = rng(144).random((30, 2)) * np.array([cw, 60]) + np.array([0, 86])
    dd = (cx_[..., None] - pts[None, None, :, 0]) ** 2 + ((cy_[..., None] - pts[None, None, :, 1]) * 0.7) ** 2
    cell = np.argmin(dd, axis=2)
    ct = rng(145).random(30) * 0.25 + (pts[:, 0] / cw) * 0.25
    depth = np.clip((cy_ - top[None, :]) / 40.0, 0, 1)
    streak = (vnoise(hh, cw * 4, 6, 149)[:, ::4] - 0.5) * 0.25
    t2 = 0.42 + ct[cell] - depth * 0.3 + streak
    edge = ((cell != np.roll(cell, 1, axis=1)) | (cell != np.roll(cell, 1, axis=0))) & \
        (rng(150).random((hh, cw)) < 0.7)
    t2 = np.where(edge, t2 + 0.22, t2)
    lay.put(0, 0, cm, M_ICE, t2)
    e = cm & ~np.roll(cm, 1, axis=0)
    lay.put(0, 0, e, M_ICE, 0.95)
    cx, cb = CAVE
    mw, mh = 26, 24
    am = np.zeros((mh, mw), bool)
    am[mw // 2:, :] = True
    am |= emask(mw, mh, (mw - 1) / 2, mw // 2, (mw - 1) / 2, mw // 2)
    yy3 = np.mgrid[0:mh, 0:mw][0]
    lay.put(cx - mw // 2, cb - mh, am, M_ICE, 0.01 + 0.08 * (yy3 / mh))
    lay.put(cx - mw // 2 - 1, cb - mh - 1, edge_of(np.pad(am, 1) | ndimage.binary_dilation(np.pad(am, 1))),
            M_ICE, 0.88)
    snow_tops(lay, 0, cw, 3, 2, 146, y_lo=80, y_hi=140)
    # wind-carved snow ridge in front
    ridge = 136 + 3 * np.sin(xs / 19.0 + 1) + 2 * np.sin(xs / 7.0) + (noise1(w, 6, 141) - 0.5) * 3
    ridge = ridge - 6 * np.clip(1 - np.abs(xs - 200) / 60.0, 0, 1)
    m = yy >= ridge[None, :]
    nz = fbm(hh, w, 6, 142, 3)
    rip = np.sin(xs[None, :] * 0.25 - yy * 0.9 + nz * 6) > 0.8
    t = 0.5 + np.clip(np.gradient(ridge), -1.5, 1.5)[None, :] * 0.2 + (nz - 0.5) * 0.3 - rip * 0.12 \
        - (yy - ridge[None, :]) * 0.004
    lay.put(0, 0, m, M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.86
    cv = lay.canvas()
    for x in range(cx - 12, cx + 13, 2):
        L = 2 + int(hash01(x, 147) * 4)
        for k in range(L):
            cv.px(x, cb - mh + k, '#d4f2ff' if k < L - 1 else '#62aee6')
    return rgba(cv)


def cave_light(c, f, ox, oy):
    cx, cb = CAVE
    x, y = cx + ox, cb - 9 + oy
    pulse = 0.75 + 0.25 * math.cos(2 * math.pi * (f % 18) / 18.0)
    glow_a(c.a, x, y, 22, '#3f82cc', 0.35 * pulse, rings=5)
    glow_a(c.a, x, y + 2, 9, '#b4e6ff', 0.35 * pulse, rings=3)
    # crystals inside the mouth
    for (dx, h, w_, lean, dk, lt) in ((-8, 7, 2, -2, '#3e5bc6', '#8ad6f6'), (-4, 12, 3, -1, '#4a78d8', '#b4e6ff'),
                                      (1, 9, 2, 1, '#3e5bc6', '#a8e2ff'), (6, 7, 2, 2, '#4a78d8', '#8ad6f6'),
                                      (-1, 5, 2, 0, '#b0509c', '#f39ce8')):
        bx, by = x + dx, cb + oy - 1
        tipx = bx + lean
        c.poly([(bx - w_, by), (bx - w_, by - h * 0.55), (tipx, by - h), (bx, by)], dk)
        c.poly([(bx, by), (tipx, by - h), (bx + w_ + 1, by - h * 0.55), (bx + w_ + 1, by)], lt)
        if (f // 6 + dx) % 5 == 0:
            c.px(tipx, by - h + 1, 'white')
    fx.sparkle(c, x - 4, y - 4, f, '#d4f2ff', period=36, size=2)
    fx.sparkle(c, x + 5, y + 1, f, 'white', period=36, phase=0.5, size=2)
    # blue light spilling onto the snow in front
    glow_a(c.a, x + 4, cb + oy + 4, 18, '#2c5aa8', 0.22 * pulse, rings=4, sx=1.8)


def igloo(lay, cx, base, r, h, seed):
    w = 2 * r + 1
    m = emask(w, 2 * h + 1, r, h, r, h)[:h + 1]
    t = blob_t(m, light=(0.8, -0.5), rad=5) * 0.78 + 0.16
    yy, xx = np.mgrid[0:h + 1, 0:w]
    row = (h - yy) // 3
    hl = ((h - yy) % 3 == 0) & (yy < h)
    vj = ((xx + row * 3) % 6 == 0)
    t = np.where(hl | vj, t - 0.3, t)
    e = edge_of(m) & (yy < h)
    t = np.where(e & (xx > r + 1) & (yy < h * 0.6), 0.99, np.where(e, 0.08, t))
    lay.put(cx - r, base - h, m, M_ICE, t * 0.85 + 0.05)
    # entrance tunnel on the left front
    tr = max(3, int(r * 0.55))
    th = max(3, int(h * 0.62))
    tm = emask(2 * tr + 1, 2 * th + 1, tr, th, tr, th)[:th + 1]
    tt = blob_t(tm, light=(0.8, -0.5), rad=3) * 0.6 + 0.2
    tt = np.where(edge_of(tm), 0.1, tt)
    tx = cx - r + tr // 2
    lay.put(tx - tr, base - th, tm, M_ICE, tt * 0.85 + 0.05)
    dw = max(2, tr)
    dh = max(3, th - 1)
    dm = np.ones((dh, dw), bool)
    dm[0, 0] = dm[0, -1] = False
    lay.put(tx - dw // 2 - 1, base - dh, dm, M_WARM, 0.5 + 0.4 * np.mgrid[0:dh, 0:dw][0] / dh)
    # glowing ice-block window
    if r >= 10:
        lay.rect(cx + r // 3, base - h // 2 - 2, 3, 3, M_WARM, 0.75)
        lay.rect(cx + r // 3 - 1, base - h // 2 + 1, 5, 1, M_SNOW, 0.95)
    return (tx - 1, base - 2)


@lru_cache(maxsize=1)
def _igloo_c():
    w, hh = 340, H + 50
    lay = Lay(w, hh)
    xs = np.arange(w)
    gl = 150 + 2 * np.sin(xs / 29.0) + (noise1(w, 8, 151) - 0.5) * 3
    yy = np.arange(hh)[:, None]
    m = yy >= gl[None, :]
    nz = fbm(hh, w, 6, 152, 3)
    t = 0.55 + (nz - 0.5) * 0.25 - (yy - gl[None, :]) * 0.005
    lay.put(0, 0, m, M_SNOW, t)
    e = m & ~np.roll(m, 1, axis=0)
    lay.T[e] = 0.78
    doors = []
    for i, (cx, base, r, h) in enumerate(IGLOOS):
        doors.append(igloo(lay, cx, base, r, h, 160 + i))
    # a path of packed snow between the igloos
    for x in range(60, 320):
        y = int(160 + 3 * math.sin(x / 23.0))
        lay.rect(x, y, 1, 2, M_SNOW, 0.38)
    # snow lanterns: stacked snowballs
    for (lx, ly) in SNOWLAMPS:
        for k, (rr_, dy) in enumerate(((3, 0), (2, 4), (1, 7))):
            bm = emask(2 * rr_ + 1, 2 * rr_ + 1, rr_, rr_, rr_, rr_)
            lay.put(lx - rr_, ly - dy - 2 * rr_, bm, M_SNOW, blob_t(bm, rad=2) * 0.6 + 0.35)
        lay.rect(lx - 1, ly - 7, 2, 2, M_WARM, 0.7)
    cv = lay.canvas()
    bl = []
    for (dx, dy), (cx, base, r, h) in zip(doors, IGLOOS):
        bl.append((dx - 1, dy + 3, 8 + r // 2, 0.36))
        if r >= 10:
            bl.append((cx + r // 3 + 1, base - h // 2 - 1, 5, 0.2))
    for (lx, ly) in SNOWLAMPS:
        bl.append((lx, ly - 4, 8, 0.26))
    bake_lights(cv, bl)
    return rgba(cv), tuple(doors)


@lru_cache(maxsize=1)
def _fg_c():
    w, hh = 360, H + 64
    lay = Lay(w, hh)
    xs = np.arange(w)
    yy = np.arange(hh)[:, None]
    crests = []
    for k, (y0, amp, per, seed) in enumerate(((171, 6, 64, 1), (184, 9, 90, 2), (206, 11, 120, 3))):
        u = ((xs + seed * 53) / per) % 1.0
        g = np.sin(np.pi * np.clip(u, 0, 1) ** 0.75) ** 1.6
        var = 0.55 + 0.45 * noise1(w, 14, 170 + seed)
        prof = y0 - amp * g * var + (noise1(w, 5, 175 + seed) - 0.5) * 2
        m = yy >= prof[None, :]
        nz = fbm(hh, w, 7, 180 + seed, 2)
        sl = np.gradient(prof)
        below = (yy - prof[None, :])
        t = 0.5 + np.clip(sl, -2, 2)[None, :] * 0.3 + (nz - 0.5) * 0.12 - below * 0.008
        lee = np.broadcast_to((sl > 0.12)[None, :], t.shape)
        t = np.where(lee & (below < 7), t - 0.28, t)
        rip = np.sin(xs[None, :] * 0.5 + yy * 0.3 + nz * 4) > 0.9
        t = t - rip * 0.06
        lay.put(0, 0, m, M_SNOW, t)
        e = m & ~np.roll(m, 1, axis=0)
        crest = np.broadcast_to((g * var > 0.62)[None, :], e.shape)
        lay.T[e & crest] = 0.97
        lay.T[e & ~crest] = np.clip(lay.T[e & ~crest] + 0.08, 0, 0.75)
        pk = np.nonzero((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]) & (g[1:-1] > 0.9))[0] + 1
        for d in pk:
            crests.append((int(d), int(prof[d])))
    # framing ice boulders (bottom corners) with snow caps
    for (bx, by, bw, bh, s_) in ((-6, 196, 34, 30, 191), (322, 200, 36, 26, 192), (30, 206, 16, 12, 193)):
        pts = [(bx, by + bh), (bx + bw * 0.1, by + bh * 0.3), (bx + bw * 0.45, by),
               (bx + bw * 0.8, by + bh * 0.2), (bx + bw, by + bh)]
        bm = pmask(w, hh, pts)
        yb, xb = np.mgrid[0:hh, 0:w]
        tb = 0.35 + 0.4 * (xb > bx + bw * 0.45) + (fbm(hh, w, 4, s_, 2) - 0.5) * 0.3
        lay.put(0, 0, bm, M_ICE, tb)
        lay.put(0, 0, edge_of(bm), M_ICE, 0.12)
    snow_tops(lay, 0, 60, 3, 2, 194, y_lo=180, y_hi=hh)
    snow_tops(lay, 318, w, 3, 2, 195, y_lo=180, y_hi=hh)
    cv = lay.canvas()
    # marker pole (folk colours) planted in the drifts
    mx, my = 84, 176
    cv.vline(mx, my - 26, my, '#2b1e2c')
    cv.vline(mx + 1, my - 26, my, '#5e3e3c')
    for k, cc in enumerate(('#c23f55', '#e0b443', '#3f73cc', '#c23f55')):
        cv.rect(mx, my - 22 + k * 3, 2, 2, cc)
    cv.hline(mx - 1, mx + 2, my - 27, '#e4ebff')
    return rgba(cv), tuple(crests)


def spindrift(c, f, crests, ox, oy):
    for j, (x, y) in enumerate(crests):
        for i in range(5):
            per = 26
            t = ((f + i * per / 5 + hash01(j, i) * 9) % per) / per
            g = 0.5 + gust(f)
            px = x + ox + 1 + t * 16 * g
            py = y + oy - 1 - t * 3 + math.sin(t * 7 + i) * 1
            if hash01(j, i, f // 2) < 1 - t:
                c.px(int(px), int(py), 'white' if t < 0.4 else '#b8c6f1')


def render_c(f):
    camx, camy = cam_c(f)

    def off(p):
        return -int(round(camx * p)), -int(round(camy * p))

    sky = _sky_c()
    c = Canvas(W, H)
    oy0 = off(P_CSKY)[1]
    c.blit(sky, 0, oy0)
    stars(c, f, 3, 230, 120, avoid=((MOON_C[0], MOON_C[1] + oy0, 20),), dy=oy0)
    aurora(c, f, 64 + oy0, 56, k=1.1, seed=11, x_shift=60)
    fo = off(P_CFAR)
    c.blit(_far_c(), fo[0], fo[1])
    eo = off(P_CEVER)
    c.blit(_ever_c(), eo[0], eo[1])
    plume(c, f, EVER[0] + eo[0], EVER[1] + eo[1])
    snowfall(c, f, 70, 301, vy=0.35, wk=0.5, color='#6e77b6')
    mo = off(P_CMID)
    c.blit(_mid_c(), mo[0], mo[1])
    cave_light(c, f, mo[0], mo[1])
    # balloon with a glowing burner drifting through the snowfall
    bx = int(round(lerp(36, 112, seg(f, CUT2, 360))))
    by = int(round(lerp(66, 56, seg(f, CUT2, 360))))
    glow_a(c.a, bx, by - 6, 9, 'amber', 0.2 * flicker(f, 99), rings=3)
    draw_balloon(c, bx, by, f, size='m', flame=True)
    snowfall(c, f, 70, 302, vy=0.5, wk=0.8, color='#a3b2e4')
    io = off(P_CIGL)
    ig, doors = _igloo_c()
    c.blit(ig, io[0], io[1])
    for i, (dx, dy) in enumerate(doors):
        torch_light(c, dx + io[0], dy + io[1] - 1, f, 200 + i, r=7, k=0.16)
    for i, (cx, base, r, h) in enumerate(IGLOOS):
        if r >= 10:
            smoke(c, cx + io[0] + 1, base - h - 1 + io[1], f, 210 + i, n=6, h=16, period=60)
    for i, (lx, ly) in enumerate(SNOWLAMPS):
        c.px(lx + io[0], ly - 7 + io[1], 'butter' if flicker(f, 220 + i) > 0.8 else 'gold')
        glow_a(c.a, lx + io[0], ly - 6 + io[1], 6, 'amber', 0.18 * flicker(f, 220 + i), rings=2)
    snowfall(c, f, 60, 303, vy=0.8, wk=1.1, color='#e4ebff', size=2, alt='#b8c6f1')
    go = off(P_CFG)
    fg, crests = _fg_c()
    c.blit(fg, go[0], go[1])
    spindrift(c, f, crests, go[0], go[1])
    pennant(c, 85 + go[0], 176 - 26 + go[1], f, 41, length=12)
    snowfall(c, f, 26, 304, vy=1.3, wk=1.6, color='white', size=2, alt='#cfd5f2', wob=2.5)
    callout(c, EVER[0] + eo[0] + 5, EVER[1] + eo[1] + 12, 'MT. EVERHOOF', f, 288, 350, side='right', length=30)
    return c


# =====================================================================================  render
def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT1:
        c = render_a(f)
        iris(c, 1 - seg(f, 72, 79), 204, 105)
    elif f < CUT2:
        c = render_b(f)
    else:
        c = render_c(f)
    atlas_card(c, f, 100, 208, 'YAKYAKISTAN', ['LOG GATES - ICE SLIDES', 'TOP OF THE WORLD'])
    return c


SFX = [                                       # scene frames
    (76, 'wind', 0.6, {'variant': 1}),        # howl as the iris opens on the gates
    (100, 'creak', 0.35),                     # timber gate groans in the gust
    (140, 'wind', 0.4, {'variant': 0}),
    (184, 'gong', 0.3),                       # the great hall, inside the walls
    (202, 'rumble', 0.55),                    # anticipation: the roof slab cracks
    (216, 'avalanche', 1.0),                  # AVALANCHE hit (bar line)
    (221, 'thud', 0.6),                       # the slab lands
    (272, 'wind', 0.55, {'variant': 1}),      # cloud pass -> Frozen North
    (288, 'crystal', 0.45, {'pan': -0.6}),    # ice cave light / Mt. Everhoof tag
    (324, 'shimmer', 0.25),
    (338, 'wind', 0.35, {'variant': 0}),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(CUT1, 'zoom', {'cx': 175, 'cy': 125}), (CUT2, 'cloud', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
