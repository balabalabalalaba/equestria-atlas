"""CASTLE OF THE TWO SISTERS: the finale (5 bars, 360 frames, HUD clock 23:00 -> 24:00). D major lullaby -> climax.

Beat sheet (SCENE frames; bar = 72, beat = 18; N = 432; the 2-bar map flight is added by long_flight):
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes 64-71.
  72-161  REVEAL (72): the iris opens on the glowing sun-and-moon rose window. THE CHASM: midnight in
          the Everfree under a huge full moon. 7 layers: blue-violet sky + stars, the moon, far forest, the
          gorge's far wall, the ruined grey-blue castle on its crag (moonlit rims, ivy, broken towers,
          pennant, lantern, a thin spill), the near cliff with a gnarled tree + the swaying ROPE BRIDGE, ferns.
          Mist rolls through the chasm, magic motes drift up, the balloon drifts past the moon. Even pan
          right (cruise). Callouts CASTLE OF THE TWO SISTERS 81-153, ROPE BRIDGE 90-156. Card 100-200.
  162     ZOOM through the castle door (beat) into
  162-251 THE OLD THRONE HALL (dollhouse cross-section): collapsed vault open to the moon, moonbeams +
          dust motes, lancet windows, tattered sun/moon banners, flickering torches, broken pillars,
          fallen chandelier, rubble, the dais with two empty thrones under the great rose window, the
          stairwell glowing cyan below. Even pan right. Callout THE OLD THRONE HALL 171-243.
  252     DISSOLVE (beat) into one tall world strip: castle on top, cut-away rock, the Cave of Harmony.
  252-286 Even descent from the castle's foot through the rock (zigzag stair, glowing geodes, veins);
          the crystalline TREE OF HARMONY rises into view (lilac / cyan / white facets, six gems, a star
          at its heart, glowing roots, reflecting pool). Callout TREE OF HARMONY 270-342.
  288     MIDNIGHT (bar line): the HUD clock reaches 00:00, the tree flares (bloom, ring, sparkle wave),
          the gems pulse in sequence on the bell strikes (every 6 frames), the veins light up.
  288-341 The tree lingers in full glow.
  342-390 Calm, even rise all the way back up (motes climb with it) to the castle under the moon at
          its zenith; the balloon rests on the keep, its burner glowing; the rose window brightens.
  390-423 Final hold; a shooting star crosses the sky (396-409). Toast from 374. Wipe 424-431.
Ambient: twinkling stars, moon halo, drifting wisps, rolling mist, magic motes, swaying bridge + vines, pennant,
         flickering torches + window glow, dust motes in moonbeams, banners, pulsing veins, rising sparkles,
         pool ripples, the balloon.
SFX: wind, creak x2, wind, shimmer, rumble (descent), shimmer, crystal (tree), crystal + shimmer after the
     midnight hit, sparkle (rise), wind, star.
"""
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.shotfx import cruise
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'twosisters'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
BT = FRAMES_PER_BEAT
MIDNIGHT = 288          # named hit: the HUD clock reaches 00:00, the tree flares (bar line)
CUT_B, CUT_C = 162, 252  # zoom into the castle door / dissolve into the descent (on beats)
DESC0, DESC1 = 252, 286  # even descent (cam 140 -> 250, ~3 px/f)
RISE0, RISE1 = 342, 390  # calm, even rise back to the castle (~5 px/f)
STAR0 = 396              # shooting star over the castle in the final hold

# ------------------------------------------------------------------ palette
SKY_N = [mix('ink', 'navy', 0.45), mix('navy', 'ink', 0.15), mix('navy', 'royal', 0.3),
         mix('indigo', 'royal', 0.45), mix('indigo', 'violet', 0.42), mix('violet', 'periwinkle', 0.3)]
STONE = (mix('navy', 'ink', 0.45), mix('indigo', 'navy', 0.4), mix('indigo', 'stone', 0.42),
         mix('stone', 'periwinkle', 0.3), mix('baby_blue', 'cloud', 0.45))
ROOF = (mix('navy', 'ink', 0.5), mix('royal', 'navy', 0.55), mix('royal', 'indigo', 0.3),
        mix('royal_lt', 'periwinkle', 0.3), mix('periwinkle', 'cloud', 0.4))
ROCK_N = (mix('ink', 'navy', 0.4), mix('navy', 'indigo', 0.35), mix('indigo', 'slate', 0.35),
          mix('grey', 'indigo', 0.35), mix('stone', 'periwinkle', 0.35))
LEAF_N = (mix('ink', 'navy', 0.35), mix('deep_teal', 'navy', 0.55), mix('deep_teal', 'indigo', 0.25),
          mix('teal', 'sage', 0.5), mix('sage', 'baby_blue', 0.45))
BARK_N = (mix('ink', 'eggplant', 0.45), mix('eggplant', 'navy', 0.4), mix('plum_brown', 'indigo', 0.45),
          mix('mauve', 'periwinkle', 0.45))
MOSS_N = (mix('deep_teal', 'navy', 0.3), mix('sage', 'deep_teal', 0.45), mix('moss', 'sage', 0.55))
WOOD_N = (mix('ink', 'wine', 0.4), mix('rust', 'plum_brown', 0.4), mix('copper', 'mauve', 0.45),
          mix('clay', 'periwinkle', 0.35))
MOONLIT = mix('baby_blue', 'cloud', 0.5)
FOGC = mix('periwinkle', 'indigo', 0.35)
WARM = ('amber', 'gold', 'butter')


def haze(ramp, a, to=FOGC):
    return tuple(mix(c, to, a) for c in ramp)


# ------------------------------------------------------------------ low-level helpers
KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)
_YY, _XX = np.mgrid[0:H, 0:W]


def _new(w, h):
    return Canvas(w, h, color=KEY)


def _rgba(cv):
    a = cv.a
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = np.any(a != _KEYA, axis=2) * 255
    return out


@lru_cache(maxsize=None)
def _dmask(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r + r * 0.8


def disc(cv, cx, cy, r, c):
    r = int(r)
    if r < 0:
        return
    cx, cy = int(round(cx)), int(round(cy))
    m = _dmask(r)
    a = cv.a
    h, w = a.shape[:2]
    x0, y0 = cx - r, cy - r
    sx, sy = max(0, -x0), max(0, -y0)
    ex, ey = min(2 * r + 1, w - x0), min(2 * r + 1, h - y0)
    if ex <= sx or ey <= sy:
        return
    a[y0 + sy:y0 + ey, x0 + sx:x0 + ex][m[sy:ey, sx:ex]] = col(c)


def bez(ps, step=0.7):
    ps = [np.array(p, float) for p in ps]
    L = sum(float(np.linalg.norm(ps[i + 1] - ps[i])) for i in range(len(ps) - 1))
    n = max(2, int(L / step) + 1)
    t = np.linspace(0, 1, n)[:, None]
    if len(ps) == 2:
        P = (1 - t) * ps[0] + t * ps[1]
    elif len(ps) == 3:
        P = (1 - t) ** 2 * ps[0] + 2 * (1 - t) * t * ps[1] + t ** 2 * ps[2]
    else:
        P = (1 - t) ** 3 * ps[0] + 3 * (1 - t) ** 2 * t * ps[1] + 3 * (1 - t) * t ** 2 * ps[2] + t ** 3 * ps[3]
    return [tuple(p) for p in P]


def tube(cv, pts, r0, r1, ramp):
    """Shaded round limb along pts (radius r0 -> r1), lit from the upper right."""
    n = len(pts)
    rs = [int(round(r0 + (r1 - r0) * i / max(1, n - 1))) for i in range(n)]
    out, dk, md, lt = ramp[:4]
    for (x, y), r in zip(pts, rs):
        disc(cv, x, y, r, out)
    for (x, y), r in zip(pts, rs):
        if r >= 1:
            disc(cv, x, y, r - 1, dk)
    for (x, y), r in zip(pts, rs):
        if r >= 3:
            disc(cv, x + 1, y, r - 2, md)
        elif r == 2:
            disc(cv, x, y, 0, md)
    for (x, y), r in zip(pts, rs):
        if r >= 4:
            disc(cv, x + r * 0.5, y - 1, max(0, r // 3 - 1), lt)


def leafball(cv, x, y, r, ramp):
    out, dk, md, lt, hi = ramp
    disc(cv, x, y, r, out)
    disc(cv, x, y, r - 1, dk)
    if r >= 3:
        disc(cv, x + 1, y - 1, r - 2, md)
    if r >= 5:
        disc(cv, x + r * 0.34, y - r * 0.38, int(r * 0.42), lt)
    if r >= 8:
        disc(cv, x + r * 0.46, y - r * 0.52, max(0, int(r * 0.15)), hi)


def canopy(cv, cx, cy, rx, ry, n, rmin, rmax, ramp, seed):
    r = rng(seed)
    balls = [(cx + r.uniform(-1, 1) * rx, cy + r.uniform(-1, 1) * ry, int(r.integers(rmin, rmax + 1)))
             for _ in range(n)]
    balls.sort(key=lambda b: b[1])
    for x, y, rr in balls:
        leafball(cv, x, y, rr, ramp)


def rimlight(rgba, color, amt=0.75, dirs=((1, 0), (0, -1)), inner=0.0):
    """Moonlight rim: opaque pixels whose neighbour toward the light is empty get tinted."""
    a = rgba[..., 3] >= 128
    h, w = a.shape
    pad = np.zeros((h + 4, w + 4), bool)
    pad[2:-2, 2:-2] = a
    edge = np.zeros_like(a)
    edge2 = np.zeros_like(a)
    for dx, dy in dirs:
        nb = pad[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
        nb2 = pad[2 + 2 * dy:2 + 2 * dy + h, 2 + 2 * dx:2 + 2 * dx + w]
        edge |= a & ~nb
        edge2 |= a & nb & ~nb2
    rgb = rgba[..., :3].astype(np.float32)
    cc = np.array(col(color), np.float32)
    rgb[edge] += (cc - rgb[edge]) * amt
    if inner > 0:
        rgb[edge2] += (cc - rgb[edge2]) * inner
    rgba[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return rgba


def add_light(c, x, y, r, color, k, rings=4):
    if k > 0.01:
        fx.glow(c, int(x), int(y), int(r), color, strength=k, rings=rings)


@lru_cache(maxsize=None)
def _noise(w, h, seed, sy, sx):
    n = rng(seed).random((h, w)).astype(np.float32)
    n = gaussian_filter(n, (sy, sx), mode='wrap')
    n -= n.min()
    n /= max(1e-6, n.max())
    return n


def tone_fill(cv, mask, tone, ramp, ox=0, oy=0):
    """Paint mask pixels with ramp[int(tone + dither)] (tone: float array, same shape as mask)."""
    h, w = mask.shape
    thr = bayer(h, w, ox, oy)
    idx = np.clip(np.floor(tone + (thr - 0.5) * 0.9 + 0.5).astype(int), 0, len(ramp) - 1)
    cols = np.array([col(c) for c in ramp], np.uint8)
    cv.a[mask] = cols[idx[mask]]


def boulder_rock(w, h, poly_pts, ramp, seed, cell=(11, 6), light=(1, -1), dark_depth=0.9, y_ref=None):
    """Chunky rock of irregular boulders (anisotropic Voronoi cells): lit top/right faces, dark seams."""
    cv = _new(w, h)
    cv.poly(poly_pts, 'white')
    mask = np.all(cv.a == np.array(col('white'), np.uint8), axis=2)
    cv.a[mask] = KEY
    r = rng(seed)
    sx, sy = cell
    pts = []
    for yy0 in range(-sy, h + sy, sy):
        for xx0 in range(-sx, w + sx, sx):
            pts.append((xx0 + r.uniform(0, sx), yy0 + r.uniform(0, sy)))
    P = np.array(pts, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), 1e9, np.float32)
    lab = np.zeros((h, w), np.int32)
    for i, (px, py) in enumerate(P):
        x0, x1 = int(max(0, px - 3 * sx)), int(min(w, px + 3 * sx))
        y0, y1 = int(max(0, py - 3 * sy)), int(min(h, py + 3 * sy))
        if x1 <= x0 or y1 <= y0:
            continue
        d = ((xx[y0:y1, x0:x1] - px) / sx) ** 2 + ((yy[y0:y1, x0:x1] - py) / sy) ** 2
        sub = best[y0:y1, x0:x1]
        m = d < sub
        sub[m] = d[m]
        lab[y0:y1, x0:x1][m] = i
    # per-cell centroid offset shading
    cxs, cys = P[lab, 0], P[lab, 1]
    u = np.clip((xx - cxs) / sx, -1.2, 1.2)
    v = np.clip((yy - cys) / sy, -1.2, 1.2)
    jit = np.array([hash01(seed, i) for i in range(len(P))], np.float32)[lab]
    T = 2.0 + 0.55 * u * light[0] - 0.65 * v * (-light[1]) + (jit - 0.5) * 0.9
    seam_r = np.zeros((h, w), bool)
    seam_r[:, :-1] = lab[:, :-1] != lab[:, 1:]
    seam_b = np.zeros((h, w), bool)
    seam_b[:-1, :] = lab[:-1, :] != lab[1:, :]
    top_e = np.zeros((h, w), bool)
    top_e[1:, :] = lab[1:, :] != lab[:-1, :]
    T[top_e] += 0.9
    T[seam_r | seam_b] = 0.0
    lit = np.zeros((h, w), np.float32)
    for k in range(1, 6):
        sh = np.zeros_like(mask)
        dx, dy = light[0] * k, light[1] * k
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        sh[ys0:ys1, xs0:xs1] = mask[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        lit += (~sh).astype(np.float32) * (1.0 / k)
    lit = np.clip(lit / 1.5, 0, 1)
    ys = np.nonzero(mask.any(axis=1))[0]
    top = float(ys[0]) if (y_ref is None and len(ys)) else float(y_ref or 0)
    depth = np.clip((yy - top) / max(1.0, h - top), 0, 1)
    seam = T <= 0.01
    T = T + lit * 1.0 - depth * dark_depth
    T[seam] = np.minimum(0.6, T[seam] + 0.6 - depth[seam])
    tone_fill(cv, mask, np.clip(T, 0, 4.3), ramp)
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    cv.a[mask & ~inner] = col(ramp[0])
    return cv


def moss_top(cv, ramp=MOSS_N, seed=0, depth=2):
    """Moss/grass cap on every top edge of the (KEY-bg) canvas."""
    a = cv.a
    opaque = np.any(a != _KEYA, axis=2)
    h, w = opaque.shape
    top = opaque.copy()
    top[1:, :] &= ~opaque[:-1, :]
    ys, xs = np.nonzero(top)
    for y, x in zip(ys, xs):
        d = depth + int(hash01(seed, x) * 2.2)
        for k in range(d):
            if y + k < h and opaque[y + k, x]:
                a[y + k, x] = col(ramp[2] if k == 0 else ramp[1] if k < d - 1 else ramp[0])
        if hash01(seed, x, 3) < 0.35 and y - 1 >= 0:
            a[y - 1, x] = col(ramp[1])
            if hash01(seed, x, 4) < 0.4 and y - 2 >= 0:
                a[y - 2, x] = col(ramp[2])


# ------------------------------------------------------------------ sky, stars, moon
@lru_cache(maxsize=None)
def sky_bg(h=H, glow_x=250, glow_y=46):
    c = Canvas(W, h)
    c.gradient(0, 0, W, h, SKY_N, soft=0.7)
    return c


@lru_cache(maxsize=None)
def star_list(seed, n, w, h):
    r = rng(seed)
    out = []
    for i in range(n):
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        b = r.random()
        out.append((x, y, b, int(r.integers(14, 40)), r.random()))
    return out


def draw_stars(c, f, seed, n, w, h, ox=0, oy=0, mask_moon=None):
    for (x, y, b, per, ph) in star_list(seed, n, w, h):
        sx, sy = x - ox, y - oy
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        if mask_moon is not None:
            mx, my, mr = mask_moon
            if (sx - mx) ** 2 + (sy - my) ** 2 < (mr + 10) ** 2:
                continue
        tw = 0.5 + 0.5 * math.sin((f / per + ph) * 6.283)
        if b > 0.93:
            cc = 'white' if tw > 0.35 else 'baby_blue'
            c.px(sx, sy, cc)
            if tw > 0.6:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.px(sx + dx, sy + dy, mix('periwinkle', 'baby_blue', 0.5))
        elif b > 0.6:
            c.px(sx, sy, 'baby_blue' if tw > 0.3 else 'periwinkle')
        else:
            if tw > 0.45:
                c.px(sx, sy, mix('periwinkle', 'lilac', 0.4) if b < 0.3 else 'periwinkle')


@lru_cache(maxsize=None)
def moon_rgba(r):
    """Full moon: crisp maria, ray crater, limb shading (no faces)."""
    s = 2 * r + 3
    cv = _new(s, s)
    c0 = r + 1
    yy, xx = np.mgrid[0:s, 0:s]
    d = np.sqrt((xx - c0) ** 2 + (yy - c0) ** 2) / (r + 0.5)
    inside = d <= 1.0
    LIMB = col(mix('cloud_dk', 'periwinkle', 0.35))
    BASE = col(mix('cloud', 'cream', 0.45))
    LIGHT = col(mix('cream', 'white', 0.55))
    MARE = col(mix('cloud_dk', 'cloud', 0.4))
    MARE_D = col(mix('cloud_dk', 'periwinkle', 0.12))
    cv.a[inside] = BASE
    lt = inside & (((xx - c0 - r * 0.22) ** 2 + (yy - c0 + r * 0.22) ** 2) < (r * 0.66) ** 2)
    cv.a[lt] = LIGHT
    nz = _noise(s, s, 91, 1.5, 1.5)
    mm = np.zeros_like(inside)
    core = np.zeros_like(inside)
    for (mx, my, rx, ry) in ((-0.36, -0.3, 0.3, 0.22), (-0.58, 0.06, 0.2, 0.34), (0.1, -0.34, 0.17, 0.14),
                             (0.32, -0.06, 0.2, 0.17), (0.06, 0.12, 0.14, 0.12), (-0.18, 0.44, 0.2, 0.11),
                             (0.45, 0.28, 0.1, 0.08)):
        e = ((xx - c0 - mx * r) / (rx * r)) ** 2 + ((yy - c0 - my * r) / (ry * r)) ** 2 + (nz - 0.5) * 0.9
        mm |= e < 1.0
        core |= e < 0.35
    cv.a[mm & inside] = MARE
    cv.a[core & inside & (bayer(s, s) < 0.5)] = MARE_D
    # limb: darker ring on the lower-left, a thin bright rim on the upper-right
    ring = inside & (d > 1 - 1.5 / r)
    ll = (xx - c0) - (yy - c0) < r * 0.3
    cv.a[ring & ll] = LIMB
    ring2 = inside & (d > 1 - 2.6 / r) & ~ring & ll & (bayer(s, s) < 0.5)
    cv.a[ring2] = MARE
    # craters: small rings (dark upper-left, light lower-right)
    cr = rng(23)
    for _ in range(max(5, r // 3)):
        a = cr.uniform(0, 6.283)
        rr = cr.uniform(0.2, 0.75) * r
        cx, cy = int(c0 + math.cos(a) * rr), int(c0 + math.sin(a) * rr)
        k = int(cr.integers(1, max(2, r // 10) + 1))
        disc(cv, cx, cy, k, MARE)
        if k >= 2:
            disc(cv, cx + 1, cy + 1, k - 1, BASE)
        cv.px(cx + k, cy + k, LIGHT)
    # Tycho-like ray crater with faint rays
    tx, ty = int(c0 - r * 0.12), int(c0 + r * 0.62)
    for ang in (0.3, 1.2, 2.2, 3.4, 4.6, 5.5):
        for j in range(2, int(r * 0.35)):
            px, py = int(tx + math.cos(ang) * j), int(ty + math.sin(ang) * j)
            if 0 <= px < s and 0 <= py < s and inside[py, px] and j % 2 == 0:
                cv.a[py, px] = LIGHT
    disc(cv, tx, ty, 1, 'white')
    return _rgba(cv)


def draw_moon(c, f, x, y, r, halo=1.0):
    add_light(c, x, y, int(r * 2.8), mix('royal', 'periwinkle', 0.5), 0.13 * halo, rings=7)
    add_light(c, x, y, int(r * 1.8), 'periwinkle', 0.15 * halo, rings=6)
    add_light(c, x, y, int(r * 1.3), mix('baby_blue', 'cloud', 0.4), 0.12 * halo, rings=4)
    m = moon_rgba(r)
    c.blit(m, int(x) - r - 1, int(y) - r - 1)


@lru_cache(maxsize=None)
def wisp_rgba(w, seed):
    """Thin night cloud wisp, moonlit on top."""
    h = 12
    cv = _new(w, h)
    r = rng(seed)
    body = mix('indigo', 'royal', 0.35)
    for i in range(7):
        cx = w * (0.1 + 0.8 * i / 6) + r.uniform(-3, 3)
        rx = w * r.uniform(0.12, 0.2)
        ry = r.uniform(1.4, 3.0)
        cv.ellipse(cx, 7 + r.uniform(-1, 1), rx, ry, body)
    rgba = _rgba(cv)
    rgba = rimlight(rgba, mix('periwinkle', 'cloud_dk', 0.4), 0.9, dirs=((0, -1),))
    rgba = rimlight(rgba, mix('indigo', 'navy', 0.3), 0.6, dirs=((0, 1),))
    return rgba


def draw_wisps(c, f, specs, ox=0):
    for (w, seed, x0, y, sp) in specs:
        spr = wisp_rgba(w, seed)
        span = W + w + 40
        x = int((x0 - f * sp - ox) % span) - w - 20
        c.blit(spr, x, y)


@lru_cache(maxsize=None)
def mist_tex(w, h, seed):
    return _noise(w, h, seed, 2.0, 9.0)


def mist(c, f, y0, y1, color, speed, seed, alpha=0.45, dens=1.0, ox=0, fade_top=True, xr=None):
    """Rolling dithered mist band between y0 and y1 (screen); xr=(x0, x1, edge) limits it horizontally."""
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    if y1 <= y0:
        return
    h = y1 - y0
    tex = mist_tex(W * 2, max(8, h), seed)
    sh = int(f * speed + ox) % (W * 2)
    t = np.roll(tex, -sh, axis=1)[:h, :W]
    prof = np.ones(h, np.float32)
    if fade_top:
        prof = np.clip((np.arange(h) + 0.5) / max(1, h * 0.55), 0, 1)
    v = np.clip((t * 1.4 - 0.35) * dens, 0, 1) * prof[:, None]
    if xr is not None:
        x0, x1, e = xr
        xs = np.arange(W)
        v = v * (np.clip((xs - x0) / e, 0, 1) * np.clip((x1 - xs) / e, 0, 1))[None, :]
    thr = bayer(h, W, 0, y0)
    q = np.clip(np.floor(v * 3 + thr) / 3, 0, 1)[:, :, None]
    reg = c.a[y0:y1].astype(np.float32)
    reg += (np.array(col(color), np.float32) - reg) * q * alpha
    c.a[y0:y1] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)


def motes(c, f, region, n, seed, colors=('ice', 'crystal_lt', 'lilac', 'white'), speed=0.25, sway=4,
          ox=0, oy=0, glow=True):
    """Floating magic motes drifting upward (never creatures: just little lights)."""
    x0, y0, w, h = region
    for i in range(n):
        sx = hash01(seed, i, 1) * w
        sy = hash01(seed, i, 2) * h
        sp = speed * (0.6 + 0.8 * hash01(seed, i, 3))
        px = x0 + int(sx + math.sin(f / (30 + 20 * hash01(seed, i, 4)) + i) * sway) - ox
        py = y0 + int((sy - f * sp) % h) - oy
        tw = math.sin(f / (8 + 6 * hash01(seed, i, 5)) + i * 1.7)
        if tw < -0.55:
            continue
        cc = colors[i % len(colors)]
        c.px(px, py, cc)
        if glow and hash01(seed, i, 6) > 0.55 and tw > 0.3:
            dim = mix(cc, 'royal', 0.55)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c.px(px + dx, py + dy, dim)


# ------------------------------------------------------------------ castle parts
def masonry(cv, x, y, w, h, ramp=STONE, seed=0, round_=False, course=5, joint=8, base=2.0, ao=0.35):
    """Stone wall rect with block courses + cylinder/flat shading lit from the right."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return
    u = (np.arange(w) + 0.5) / w
    if round_:
        sh = np.sin(np.clip(u, 0, 1) * math.pi) ** 0.5 * (0.25 + 0.95 * u)
        tone = 1.0 + sh * 2.3
    else:
        tone = np.full(w, base, np.float32)
        tone[-2:] += 0.9
        tone[:1] -= 0.6
    T = np.broadcast_to(tone[None, :], (h, w)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    k = (yy + y) // course
    off = (k * (joint // 2 + 1)) % joint
    bid_x = (xx + x + off) // joint
    # block tone jitter (computed per block id via a cheap hash)
    hsh = ((k * 73856093) ^ (bid_x * 19349663) ^ (seed * 83492791)) & 0xffff
    jit = (hsh % 7 == 0) * 1.0 - (hsh % 11 == 0) * 1.0
    T += jit
    mortar = ((yy + y) % course == course - 1) | ((xx + x + off) % joint == 0)
    T[mortar] -= 1.0
    # soft top-to-bottom darkening (ambient occlusion near ground)
    T -= (yy / max(1, h)) * ao
    cv2 = Canvas(arr=cv.a[y:y + h, x:x + w])
    m = np.ones((h, w), bool)
    tone_fill(cv2, m, np.clip(T, 0.6, 4.4), ramp, x, y)
    cv.frame(x, y, w, h, ramp[0])


def crenels(cv, x, y, w, ramp=STONE, mw=3, gap=2, mh=3, broken=()):
    """Merlons on top of a wall whose top edge is y (merlons stand above y)."""
    xx = x
    i = 0
    while xx + mw <= x + w:
        if i not in broken:
            cv.rect(xx, y - mh, mw, mh + 1, ramp[0])
            cv.rect(xx + 1, y - mh + 1, mw - 2, mh, ramp[3])
            cv.px(xx + mw - 2, y - mh + 1, ramp[4])
        xx += mw + gap
        i += 1


def arch(cv, x, y, w, h, fill, frame=None, sill=True):
    """Rounded-top window/door: (x, y) top-left, w x h."""
    o = frame or STONE[0]
    r = w // 2
    cv.rect(x, y + r, w, h - r, o)
    cv.ellipse(x + (w - 1) / 2, y + r, (w - 1) / 2, r, o)
    if w >= 3:
        cv.rect(x + 1, y + r, w - 2, h - r - 1, fill)
        cv.ellipse(x + (w - 1) / 2, y + r, max(0.5, (w - 3) / 2), max(0.5, r - 1), fill)
    if sill:
        cv.hline(x - 1, x + w, y + h - 1, STONE[3])
        cv.hline(x - 1, x + w, y + h, STONE[1])


def cone(cv, cx, base, hw, top, ramp=ROOF, seed=0, broken=None):
    """Conical slate roof (side view) apex at (cx, top), eaves at y=base, half width hw."""
    hgt = base - top
    pts = [(cx - hw - 1, base + 1), (cx, top - 1), (cx + hw + 1, base + 1)]
    cv.poly(pts, ramp[0])
    for yy in range(top, base + 1):
        t = (yy - top) / max(1, hgt)
        half = hw * t
        xl, xr = int(round(cx - half)), int(round(cx + half))
        for xx in range(xl, xr + 1):
            u = min(1.0, max(0.0, (xx - (cx - half)) / max(1.0, 2 * half)))
            tone = 1.1 + 2.3 * u ** 1.3
            if (yy - top) % 4 == 3:
                tone -= 0.8                         # tile courses
            elif ((xx + (yy // 4) * 2) % 4 == 0) and (yy - top) % 4 == 2:
                tone -= 0.5
            tone += (bayer(1, 1, xx, yy)[0, 0] - 0.5) * 0.6
            cv.px(xx, yy, ramp[int(np.clip(round(tone), 1, 4))])
    cv.hline(int(cx - hw - 1), int(cx + hw + 1), base + 1, ramp[0])
    cv.hline(int(cx - hw), int(cx + hw), base + 2, STONE[1])
    if broken:
        for (bx, by, br) in broken:          # holes showing rafters
            disc(cv, bx, by, br, ramp[0])
            cv.line(bx - br, by + br - 1, bx + br - 1, by - br + 1, WOOD_N[1])
            cv.px(bx, by, WOOD_N[2])


def vine_static(cv, x, y, L, seed, ramp=MOSS_N):
    for j in range(L):
        xx = x + int(round(math.sin(j * 0.5 + seed) * 0.8))
        cv.px(xx, y + j, ramp[1])
        if hash01(seed, j) < 0.45:
            cv.px(xx + (1 if hash01(seed, j, 2) < 0.5 else -1), y + j, ramp[2])
        if hash01(seed, j, 5) < 0.12:
            cv.px(xx + 1, y + j - 1, mix(ramp[2], 'baby_blue', 0.3))


def rubble(cv, x, y, w, seed, ramp=STONE):
    """Pile of fallen blocks resting on y (bottom)."""
    r = rng(seed)
    for i in range(int(w / 2.5)):
        bw, bh = int(r.integers(3, 7)), int(r.integers(2, 4))
        t = r.uniform(0, 1)
        bx = int(x + t * w - bw / 2)
        hgt = (1 - abs(t - 0.5) * 2) * w * 0.28
        by = int(y - bh - r.uniform(0, hgt))
        cv.rect(bx, by, bw, bh, ramp[0])
        cv.rect(bx + 1, by + 1, bw - 2, bh - 1, ramp[2])
        cv.hline(bx + 1, bx + bw - 2, by + 1, ramp[3])
        cv.px(bx + bw - 2, by + 1, ramp[4])


def rose_window(cv, cx, cy, r):
    """The great sun-and-moon rose window: warm sun half (left), cool crescent half (right). No faces."""
    disc(cv, cx, cy, r + 2, STONE[0])
    disc(cv, cx, cy, r + 1, STONE[3])
    disc(cv, cx, cy, r, STONE[0])
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    inside = d <= r - 0.5
    ang = np.arctan2(yy, xx)
    warm = xx < 0
    # sun half: radiating gold/amber/orange petals; moon half: navy/blue with stars
    petal = ((ang * 8 / math.pi) % 2) < 1
    out = np.zeros((2 * r + 1, 2 * r + 1, 3), np.uint8)
    out[:] = col('ink')
    out[warm & petal] = col('amber')
    out[warm & ~petal] = col('gold')
    out[warm & (d < r * 0.45)] = col('butter')
    out[~warm & petal] = col('crystal_dk')
    out[~warm & ~petal] = col(mix('royal', 'crystal_dk', 0.5))
    # crescent on the cool half
    cxm, cym = r * 0.42, 0
    dm = np.sqrt((xx - cxm) ** 2 + (yy - cym) ** 2)
    dm2 = np.sqrt((xx - cxm - r * 0.17) ** 2 + (yy - cym - 0.0) ** 2)
    cres = (dm <= r * 0.38) & (dm2 > r * 0.3) & ~warm
    out[cres] = col('crystal_lt')
    # tiny stars on the cool half
    for (sx, sy) in ((int(r * 0.25), -int(r * 0.62)), (int(r * 0.7), int(r * 0.45)), (int(r * 0.2), int(r * 0.6))):
        if sx * sx + sy * sy < (r - 1) ** 2:
            out[sy + r, sx + r] = col('white')
    # lead lines: radial spokes + a ring + the central divide
    lead = (np.abs(((ang * 8 / math.pi) % 2) - 1) > 0.93) & (d > r * 0.45)
    lead |= np.abs(d - r * 0.45) < 0.5
    lead |= (xx == 0)
    out[lead] = col(mix('ink', 'navy', 0.5))
    sub = cv.a[cy - r:cy + r + 1, cx - r:cx + r + 1]
    sub[inside] = out[inside]


def step_px(cv, x, y, k=-1, ramp=STONE):
    """Move a pixel k steps along the ramp (only if it is a ramp colour)."""
    h, w = cv.a.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    p = tuple(int(v) for v in cv.a[y, x])
    for i, rc in enumerate(ramp):
        if p == col(rc):
            cv.a[y, x] = col(ramp[int(np.clip(i + k, 0, len(ramp) - 1))])
            return


def decay(cv, rects, seed, holes=3, streaks=3, moss=4, ramp=STONE):
    """Weathering on wall rects: chipped holes, rain streaks, moss patches."""
    r = rng(seed)
    for (x, y, w, h) in rects:
        for _ in range(holes):
            hx, hy = int(r.integers(x + 2, x + w - 3)), int(r.integers(y + 2, y + h - 3))
            hw, hh = int(r.integers(2, 4)), int(r.integers(1, 3))
            for yy in range(hy, hy + hh):
                for xx in range(hx, hx + hw):
                    step_px(cv, xx, yy, -2, ramp)
            for xx in range(hx, hx + hw):
                step_px(cv, xx, hy + hh, 1, ramp)
        for _ in range(streaks):
            sx, sy = int(r.integers(x + 1, x + w - 1)), int(r.integers(y, y + h // 2))
            for j in range(int(r.integers(5, 14))):
                if (j + sx) % 3 != 2:
                    step_px(cv, sx, sy + j, -1, ramp)
        for _ in range(moss):
            mx, my = int(r.integers(x, x + w)), int(r.integers(y + h // 2, y + h))
            for _k in range(int(r.integers(3, 9))):
                px, py = mx + int(r.integers(-2, 3)), my + int(r.integers(-1, 2))
                hh, ww = cv.a.shape[:2]
                if 0 <= px < ww and 0 <= py < hh and tuple(cv.a[py, px]) != KEY:
                    cv.a[py, px] = col(MOSS_N[int(r.integers(0, 3))])


def ivy(cv, x, y, w, L, seed):
    """Ivy curtain hanging from (x, y): several tapering vines with leaf clusters."""
    r = rng(seed)
    for i in range(w):
        LL = int(L * (0.35 + 0.65 * math.sin((i + 0.5) / w * math.pi)) * r.uniform(0.7, 1.0))
        for j in range(LL):
            xx = x + i + int(round(math.sin(j * 0.45 + i) * 0.7))
            cc = MOSS_N[1] if (j + i) % 3 else MOSS_N[0]
            cv.px(xx, y + j, cc)
            if r.random() < 0.3:
                cv.px(xx + 1, y + j, MOSS_N[2])
            if r.random() < 0.06:
                cv.px(xx, y + j, mix(MOSS_N[2], 'baby_blue', 0.35))


@lru_cache(maxsize=None)
def castle_rgba():
    """The ruined Castle of the Two Sisters, side view (160 x 124, base row 123).

    Left -> right: gatehouse, round tower with cone roof + pennant pole, great hall, the central keep
    (tallest; broken cone cap, crenellated parapet where the balloon rests; rose window; portal), a
    hall wing with a collapsed breach, the broken round tower, a ruined arcade wall + rubble.
    """
    cv = _new(160, 124)
    S = STONE
    B = 123
    # --- back hall wings
    masonry(cv, 30, 64, 34, B - 64 + 1, seed=1)
    crenels(cv, 30, 64, 34, broken=(2,))
    masonry(cv, 95, 70, 26, B - 70 + 1, seed=2)
    crenels(cv, 95, 70, 26)
    cv.poly([(99, 66), (104, 82), (108, 77), (113, 88), (117, 66)], KEY)      # collapsed breach
    for (wx, wy) in ((36, 76), (50, 76)):
        arch(cv, wx, wy, 7, 18, mix('wine', 'ink', 0.3))
        cv.rect(wx + 2, wy + 9, 3, 8, mix('amber', 'wine', 0.45))
        cv.px(wx + 3, wy + 8, mix('amber', 'wine', 0.3))
    arch(cv, 104, 92, 7, 16, mix('wine', 'ink', 0.3))
    cv.rect(106, 100, 3, 7, mix('amber', 'wine', 0.5))
    # --- gatehouse
    masonry(cv, 1, 84, 30, B - 84 + 1, seed=3)
    crenels(cv, 1, 84, 30, broken=(4,))
    arch(cv, 9, 100, 13, 24, mix('ink', 'navy', 0.3), sill=False)
    for gx in range(11, 21, 2):                       # broken portcullis
        cv.vline(gx, 102 if gx < 17 else 106, 108 + (gx * 7) % 5, mix('steel_dk', 'navy', 0.3))
    cv.hline(10, 20, 104, mix('steel_dk', 'navy', 0.3))
    for (sx, sy) in ((4, 92), (26, 92)):
        cv.rect(sx, sy, 1, 4, S[0])
    # --- left round tower + cone roof
    masonry(cv, 17, 34, 19, B - 34 + 1, seed=4, round_=True, course=4, joint=6)
    cone(cv, 26, 34, 12, 8, seed=5, broken=((30, 26, 1),))
    for (wx, wy, lit) in ((26, 44, 0), (26, 66, 1), (22, 90, 0)):
        cv.rect(wx - 1, wy - 1, 3, 7, STONE[0])
        cv.rect(wx, wy, 1, 5, mix('amber', 'wine', 0.3) if lit else mix('ink', 'navy', 0.3))
        cv.px(wx + 1, wy - 1, STONE[3])
    cv.hline(16, 36, 58, S[1])                         # string course
    cv.hline(16, 36, 57, S[3])
    # --- central keep (tallest)
    masonry(cv, 62, 24, 35, B - 24 + 1, seed=6, course=5, joint=9)
    for bx in (60, 95):                               # corner buttresses
        masonry(cv, bx, 30, 4, B - 30 + 1, seed=7 + bx, course=5, joint=4, base=2.4)
    crenels(cv, 61, 24, 37, mw=3, gap=2, mh=4, broken=(6,))
    cv.hline(61, 97, 25, S[0])
    cv.hline(61, 97, 26, S[1])
    # broken cone cap: left half stands, right half collapsed (rafters)
    cone(cv, 75, 21, 11, 0, seed=8)
    cv.poly([(73, -3), (74, 5), (76, 7), (77, 11), (80, 10), (82, 15), (85, 16), (88, 21), (92, 21), (92, -3)], KEY)
    for (rx, ry, L) in ((75, 6, 5), (77, 11, 6), (81, 15, 5)):
        cv.line(rx, ry, rx + L, ry + L - 2, WOOD_N[0])
        cv.line(rx, ry - 1, rx + L, ry + L - 3, WOOD_N[2])
        cv.px(rx + L, ry + L - 3, WOOD_N[3])
    cv.vline(75, -1, 2, 'gold')                        # finial: little sun disc
    disc(cv, 75, 0, 1, 'gold')
    cv.px(74, -1, 'butter')
    # upper windows
    for wx in (67, 87):
        arch(cv, wx, 32, 5, 11, mix('ink', 'navy', 0.25))
    rose_window(cv, 79, 56, 10)
    # portal
    cv.rect(70, 95, 19, 2, S[3])
    arch(cv, 72, 97, 15, 27, mix('ink', 'wine', 0.25), sill=False)
    cv.rect(75, 108, 9, 15, mix('wine', 'amber', 0.25))
    cv.rect(77, 112, 5, 11, mix('amber', 'wine', 0.35))
    for st in range(3):                                # steps
        cv.rect(70 - st * 2, B - 2 + st, 19 + st * 4, 1, S[3 - (st % 2)])
    # --- right round tower (broken top)
    masonry(cv, 118, 46, 19, B - 46 + 1, seed=9, round_=True, course=4, joint=6)
    cv.poly([(117, 40), (119, 50), (122, 47), (125, 53), (128, 49), (131, 56), (134, 51), (138, 58),
             (138, 40)], KEY)
    cv.rect(118, 50, 3, 3, S[0])
    cv.rect(119, 51, 1, 2, S[3])
    for (wx, wy) in ((127, 64), (127, 88)):
        cv.rect(wx - 1, wy - 1, 3, 7, STONE[0])
        cv.rect(wx, wy, 1, 5, mix('ink', 'navy', 0.3))
    # --- ruined arcade wall
    masonry(cv, 136, 92, 22, B - 92 + 1, seed=10)
    cv.poly([(135, 88), (137, 95), (141, 92), (146, 99), (150, 94), (154, 101), (159, 96), (159, 88)], KEY)
    arch(cv, 143, 104, 9, 20, KEY, frame=S[0], sill=False)
    rubble(cv, 150, B + 1, 12, 11)
    rubble(cv, 33, B + 1, 8, 12)
    # --- weathering
    decay(cv, [(31, 66, 32, 56), (96, 72, 24, 50), (2, 86, 28, 36), (63, 27, 33, 95), (18, 36, 18, 86),
               (119, 56, 18, 66), (137, 98, 20, 25)], 21)
    ivy(cv, 63, 27, 7, 30, 1)
    ivy(cv, 88, 27, 6, 22, 2)
    ivy(cv, 31, 66, 5, 18, 3)
    ivy(cv, 119, 56, 6, 26, 4)
    ivy(cv, 2, 86, 4, 14, 5)
    ivy(cv, 96, 72, 4, 30, 6)
    # --- vines
    for (vx, vy, L, sd) in ((33, 65, 14, 1), (46, 65, 9, 2), (64, 27, 22, 3), (93, 27, 30, 4), (98, 71, 12, 5),
                            (120, 52, 18, 6), (135, 56, 26, 7), (20, 37, 10, 8), (3, 85, 12, 9), (140, 96, 10, 10),
                            (84, 60, 10, 11), (157, 97, 14, 12)):
        vine_static(cv, vx, vy, L, sd)
    rgba = _rgba(cv)
    rimlight(rgba, MOONLIT, 0.7, dirs=((1, 0), (0, -1)), inner=0.25)
    return rgba


CASTLE_W, CASTLE_H = 160, 124
KEEP_TOP = (88, 22)          # parapet spot where the balloon rests (sprite coords)
PENNANT = (26, 6)            # cone tip of the left tower (sprite coords)
ROSE = (79, 56)


def pennant(c, x, y, f, color='royal_lt'):
    """Tattered pennant on a pole at (x, y) = pole top, waving right."""
    c.vline(x, y, y + 6, STONE[1])
    for i in range(8):
        wv = int(round(math.sin(f / 5.0 - i * 0.8) * (0.4 + i * 0.18)))
        hgt = max(1, 3 - i // 3)
        yy = y + 1 + wv
        cc = color if i < 5 else mix(color, 'navy', 0.3)
        if i in (6,) and (f // 4) % 2:
            continue
        c.vline(x + 1 + i, yy, yy + hgt - 1, cc)
        if i == 2:
            c.px(x + 1 + i, yy, 'gold')


# ------------------------------------------------------------------ bridge
def rope_bridge(c, f, a, b, sag=9.0):
    """Swaying rope bridge from anchor a to anchor b (screen coords, deck level)."""
    (lx, ly), (rx, ry) = a, b
    sw = math.sin(f * 2 * math.pi / 60.0)
    sg = sag + 1.5 * sw
    n = max(2, int(rx - lx))
    prev_r = None
    for i in range(n + 1):
        t = i / n
        bump = 4 * t * (1 - t)
        yd = int(round(ly + (ry - ly) * t + sg * bump))
        yr = int(round(ly - 8 + (ry - ly) * t + (sg - 3) * bump))
        x = int(lx) + i
        if not (-2 <= x < W + 2):
            continue
        c.px(x, yr, mix('tan', 'periwinkle', 0.3))
        c.px(x, yr + 1, WOOD_N[0])
        if prev_r is not None and abs(prev_r - yr) > 1:
            c.px(x, (prev_r + yr) // 2, mix('tan', 'periwinkle', 0.3))
        prev_r = yr
        if i % 6 == 3:
            c.vline(x, yr + 2, yd - 1, mix('tan', 'plum_brown', 0.5))
        k = i // 3
        if hash01(k, 41) < 0.1 or i % 3 == 2:
            c.px(x, yd + 1, WOOD_N[0])
            continue
        c.px(x, yd, WOOD_N[3] if i % 3 == 0 else WOOD_N[2])
        c.px(x, yd + 1, WOOD_N[1])
        c.px(x, yd + 2, WOOD_N[0])
        if hash01(k, 9) > 0.86:
            c.px(x, yd + 3, WOOD_N[1])
    # lower guide rope
    for i in range(0, n + 1, 1):
        t = i / n
        bump = 4 * t * (1 - t)
        y2 = int(round(ly + 3 + (ry - ly) * t + (sg + 1) * bump))
        if i % 2 == 0:
            c.px(int(lx) + i, y2, WOOD_N[0])


def post(c, x, y, h=10):
    c.rect(x - 1, y - h, 3, h + 2, WOOD_N[0])
    c.vline(x, y - h + 1, y, WOOD_N[2])
    c.px(x + 1, y - h + 1, WOOD_N[3])
    c.rect(x - 1, y - h - 1, 3, 1, WOOD_N[1])


# ------------------------------------------------------------------ SHOT A: the chasm
A_CAM = 40
A_P = dict(stars=0.05, moon=0.1, far=0.25, mid=0.5, castle=0.8, near=1.0, fg=1.25)
A_CASTLE = (125, 15)          # castle sprite origin (screen at cam 0)
A_MOON = (252, 50, 30)


def camA(f):
    return int(round(A_CAM * cruise(seg(f, 72, CUT_B + 4))))


@lru_cache(maxsize=None)
def farA():
    """Far forest hills (rounded canopies), hazy and moonlit."""
    w = 340
    cv = _new(w, H)
    ramp = haze(LEAF_N, 0.55)
    r = rng(5)
    x = -6
    while x < w + 6:
        rr = int(r.integers(5, 10))
        yb = 108 + 8 * math.sin(x / 47.0) + 5 * math.sin(x / 19.0 + 1)
        leafball(cv, x, yb, rr, ramp)
        x += rr * 1.3
    cv.rect(0, 112, w, H - 112, ramp[1])
    for k in range(40):
        leafball(cv, r.uniform(0, w), r.uniform(114, 150), int(r.integers(4, 8)), ramp)
    rgba = _rgba(cv)
    rimlight(rgba, mix(MOONLIT, FOGC, 0.4), 0.5, dirs=((0, -1), (1, 0)))
    return rgba


@lru_cache(maxsize=None)
def midA():
    """Forest on the far rim of the gorge + the gorge's far wall dropping into the deep."""
    w = 360
    cv = _new(w, H)
    wall = boulder_rock(w, H, [(-2, 140), (40, 136), (90, 141), (150, 135), (210, 139), (270, 134), (330, 140),
                            (362, 137), (362, H + 2), (-2, H + 2)], haze(ROCK_N, 0.45), 14, cell=(9, 5))
    cv.blit(_rgba(wall), 0, 0)
    ramp = haze(LEAF_N, 0.3)
    r = rng(8)
    for k in range(9):
        x = r.uniform(0, w)
        top = r.uniform(108, 122)
        tr = haze(BARK_N, 0.35)
        cv.rect(int(x) - 2, int(top), 4, 140 - int(top), tr[1])
        cv.vline(int(x) + 1, int(top), 140, tr[2])
        canopy(cv, x, top, 13, 5, 7, 5, 9, ramp, 30 + k)
    x = -6
    while x < w + 6:
        rr = int(r.integers(5, 10))
        yb = 131 + 4 * math.sin(x / 31.0) + 3 * math.sin(x / 13.0)
        leafball(cv, x, yb, rr, ramp)
        x += rr * 1.2
    rgba = _rgba(cv)
    rimlight(rgba, mix(MOONLIT, FOGC, 0.25), 0.55, dirs=((0, -1), (1, 0)))
    return rgba


@lru_cache(maxsize=None)
def cragA():
    """Castle crag under the castle (castle-layer coords: origin = castle origin - (40, 0))."""
    w, h = 250, H - 100
    # top edge at local y 38 (castle base 138 - 100) spans the castle footprint
    pts = [(0, h + 2), (10, 86), (22, 64), (34, 48), (38, 40), (42, 37), (210, 37), (216, 42), (226, 56),
           (236, 70), (246, h + 2)]
    cv = boulder_rock(w, h, pts, ROCK_N, 3, cell=(14, 8), dark_depth=2.0)
    moss_top(cv, seed=4)
    for (x0, L, sd) in ((46, 16, 1), (60, 10, 2), (120, 8, 3), (170, 12, 4), (204, 18, 5), (214, 22, 6), (30, 14, 7)):
        ivy(cv, x0, 38, 4, L, 100 + sd)
    # drain spout under the gatehouse (feeds the little waterfall)
    cv.rect(37, 41, 6, 3, STONE[0])
    cv.rect(37, 42, 5, 1, STONE[3])
    rgba = _rgba(cv)
    rimlight(rgba, MOONLIT, 0.55, dirs=((1, 0), (0, -1)))
    return rgba


@lru_cache(maxsize=None)
def nearA():
    """Near cliff (left) with the gnarled Everfree tree; bridge posts drawn per frame."""
    w = 140
    cv = _new(w, H)
    pts = [(-2, 131), (14, 128), (30, 128), (42, 130), (48, 133), (50, 140), (47, 152), (50, 166), (46, H + 2),
           (-2, H + 2)]
    rock = boulder_rock(w, H, pts, ROCK_N, 12, cell=(11, 6), dark_depth=1.8)
    moss_top(rock, seed=13, depth=3)
    cv.blit(_rgba(rock), 0, 0)
    # tree: twisted trunk + roots + crown
    trunk = bez([(12, 132), (6, 104), (20, 80), (10, 40)], 0.6)
    tube(cv, trunk, 7, 4, BARK_N)
    for rt in (bez([(10, 130), (0, 128), (-6, 136)]), bez([(16, 130), (26, 128), (34, 132)])):
        tube(cv, rt, 3, 1, BARK_N)
    br = bez([(14, 70), (36, 56), (52, 46), (66, 44)], 0.6)
    tube(cv, br, 3, 2, BARK_N)
    br2 = bez([(10, 52), (-4, 40), (-10, 30)], 0.6)
    tube(cv, br2, 3, 2, BARK_N)
    canopy(cv, 30, 22, 40, 14, 26, 6, 13, LEAF_N, 51)
    canopy(cv, 70, 36, 14, 6, 7, 5, 9, LEAF_N, 52)
    # knothole + grooves on the trunk
    cv.ellipse(10, 98, 1, 2, BARK_N[0])
    for (gx, gy) in ((14, 112), (8, 90), (16, 70)):
        cv.line(gx, gy, gx - 2, gy - 6, BARK_N[0])
    # ferns at the cliff top
    for (fx0, s) in ((22, 1), (34, 2), (3, 3)):
        for k in range(5):
            a = -math.pi / 2 + (k - 2) * 0.45
            for j in range(6):
                px = fx0 + math.cos(a) * j
                py = 128 + math.sin(a) * j + (j * j) * 0.08
                cv.px(px, py, MOSS_N[1] if j < 4 else MOSS_N[2])
    rgba = _rgba(cv)
    rimlight(rgba, MOONLIT, 0.7, dirs=((1, 0), (0, -1)), inner=0.2)
    return rgba


@lru_cache(maxsize=None)
def fgA():
    w = 400
    cv = _new(w, H)
    dk = (mix('ink', 'navy', 0.3), mix('ink', 'deep_teal', 0.45), mix('deep_teal', 'navy', 0.5),
          mix('deep_teal', 'sage', 0.35), mix('sage', 'baby_blue', 0.35))
    # bottom-left fern clump + right rock with glowing crystal flowers
    for k in range(9):
        a = -math.pi / 2 + (k - 4) * 0.33
        L = 18 + 6 * hash01(k, 2)
        pts = [(-4 + k * 3 + math.cos(a) * j, H + 2 + math.sin(a) * j + 0.05 * j * j) for j in range(int(L))]
        for j, (px, py) in enumerate(pts):
            cv.px(px, py, dk[1])
            if j % 3 == 1 and j > 3:
                cv.px(px + 1, py + 1, dk[2])
                cv.px(px - 1, py + 1, dk[2])
    rk = [(330, H + 2), (336, 168), (348, 160), (366, 158), (380, 164), (392, H + 2)]
    rock = boulder_rock(w, H, rk, ROCK_N, 33, cell=(9, 6))
    moss_top(rock, seed=6, depth=2)
    cv.blit(_rgba(rock), 0, 0)
    canopy(cv, 372, 170, 12, 4, 6, 4, 7, dk, 61)
    rgba = _rgba(cv)
    rimlight(rgba, MOONLIT, 0.5, dirs=((1, 0), (0, -1)))
    return rgba


def sway_vine(c, x, y0, L, f, seed, ramp=MOSS_N, amp=1.2):
    for j in range(L):
        t = j / max(1, L)
        dx = int(round(math.sin(f / 22.0 + seed + j * 0.12) * amp * t * 2))
        c.px(x + dx, y0 + j, ramp[1])
        if hash01(seed, j) < 0.4:
            c.px(x + dx + (1 if hash01(seed, j, 1) < 0.5 else -1), y0 + j, ramp[2])
    c.px(x + int(round(math.sin(f / 22.0 + seed + L * 0.12) * amp * 2)), y0 + L, ramp[2])


def window_glow(c, x, y, f, k=1.0):
    """Animated warm/cool light breathing out of the rose window."""
    p = 0.85 + 0.15 * math.sin(f / 9.0) + 0.06 * math.sin(f / 3.7)
    add_light(c, x - 2, y, 18, mix('amber', 'gold', 0.4), 0.16 * p * k, rings=4)
    add_light(c, x + 3, y, 18, 'crystal', 0.14 * p * k, rings=4)
    add_light(c, x, y, 10, 'butter', 0.1 * p * k, rings=3)


def waterfall(c, f, x, y0, y1, w=3):
    """Thin animated spill falling from (x, y0) to y1 (screen), with a splash of mist at the foot."""
    cols = (mix('baby_blue', 'periwinkle', 0.3), 'baby_blue', mix('cloud', 'white', 0.5), mix('periwinkle', 'indigo', 0.2))
    for yy in range(max(0, y0), min(H, y1)):
        t = (yy - y0) / max(1, y1 - y0)
        xx = x + int(t * t * 4)
        for k in range(w + int(t * 2)):
            v = (yy * 3 - f * 4 + k * 5) % 11
            cc = cols[2] if v < 2 else cols[1] if v < 6 else cols[0] if v < 9 else cols[3]
            c.px(xx + k, yy, cc)
    for i in range(5):
        ph = (f * 0.7 + i * 7) % 14
        c.px(x + 4 + int(math.sin(i * 2.1) * 5), y1 - int(ph * 0.6), mix('cloud', 'periwinkle', 0.3))


def lantern(c, x, y, f, seed=0):
    """Hanging lantern on a hook at (x, y) (screen): warm flicker + glow."""
    fl = 0.85 + 0.15 * hash01(seed, f // 3)
    add_light(c, x, y + 4, 14, 'amber', 0.2 * fl, rings=4)
    add_light(c, x, y + 4, 5, 'butter', 0.25 * fl, rings=2)
    c.rect(x - 2, y + 1, 5, 6, mix('ink', 'navy', 0.3))
    c.rect(x - 1, y + 2, 3, 4, 'gold' if fl > 0.92 else 'amber')
    c.px(x, y + 3, 'butter')
    c.hline(x - 2, x + 2, y + 1, mix('steel_dk', 'navy', 0.2))
    c.px(x, y, mix('steel_dk', 'navy', 0.2))


def render_A(f):
    cam = camA(f)
    c = sky_bg().copy()
    p = A_P
    mx, my, mr = A_MOON
    mxs = int(mx - cam * p['moon'])
    draw_stars(c, f, 3, 150, W + 20, 120, ox=int(cam * p['stars']), mask_moon=(mxs, my, mr))
    draw_moon(c, f, mxs, my, mr)
    draw_wisps(c, f, ((70, 4, 380, 66, 0.12), (54, 7, 150, 30, 0.08), (90, 9, 40, 84, 0.1)), ox=int(cam * 0.15))
    c.blit(farA(), -int(cam * p['far']), 0)
    mist(c, f, 100, 150, mix('periwinkle', 'indigo', 0.25), 0.18, 1, alpha=0.45)
    c.blit(midA(), -int(cam * p['mid']), 0)
    mist(c, f, 146, 180, mix('periwinkle', 'baby_blue', 0.25), 0.3, 2, alpha=0.55)
    # balloon drifting in past the moon (behind the castle layer)
    bx = int(318 - 46 * smooth(seg(f, 72, CUT_B)) - cam * 0.6)
    draw_balloon(c, bx, 74, f, size='s')
    # castle + crag
    ox, oy = A_CASTLE
    cx = int(ox - cam * p['castle'])
    c.blit(cragA(), cx - 40, 100)
    c.blit(castle_rgba(), cx, oy)
    pennant(c, cx + PENNANT[0], oy + PENNANT[1] - 6, f)
    window_glow(c, cx + ROSE[0], oy + ROSE[1], f)
    add_light(c, cx + 79, oy + 112, 9, 'amber', 0.12 + 0.03 * math.sin(f / 4.0), rings=3)
    for (wx, wy) in ((39, 85), (53, 85), (107, 100)):
        add_light(c, cx + wx, oy + wy, 5, 'amber', 0.10 + 0.05 * hash01(wx, f // 4), rings=2)
    nxe = 50 - cam
    # chasm mist in front of the crag foot
    mist(c, f, 150, 180, mix('periwinkle', 'baby_blue', 0.3), -0.35, 3, alpha=0.5, xr=(nxe - 6, cx - 14, 16))
    mist(c, f, 160, 180, mix('periwinkle', 'baby_blue', 0.3), -0.25, 4, alpha=0.4, xr=(cx + 150, 420, 24))
    motes(c, f, (40, 60, 300, 120), 26, 5, ox=int(cam * 0.9))
    # near cliff, tree, bridge
    nx = -cam
    a = (nx + 46, 129)
    b = (cx + 14, oy + 121)
    rope_bridge(c, f, a, b, sag=9)
    post(c, a[0] - 1, a[1] + 1, 11)
    post(c, a[0] - 5, a[1], 12)
    post(c, b[0] + 2, b[1] + 1, 9)
    post(c, b[0] + 6, b[1] + 1, 14)
    c.hline(b[0] + 6, b[0] + 9, b[1] - 13, WOOD_N[1])
    lantern(c, b[0] + 9, b[1] - 13, f, 1)
    waterfall(c, f, cx - 2, oy + 128, 181)
    c.blit(nearA(), nx, 0)
    for (vx, vy, L, sd) in ((22, 30, 26, 1), (40, 26, 18, 2), (58, 40, 14, 3), (70, 44, 22, 4), (8, 32, 12, 5)):
        sway_vine(c, nx + vx, vy, L, f, sd)
    c.blit(fgA(), -int(cam * p['fg']), 0)
    # cool grade: vignette
    fx.vignette(c, 0.28, color=mix('ink', 'navy', 0.4))
    # callouts
    callout(c, cx + 79, oy + 100, 'CASTLE OF THE TWO SISTERS', f, 81, 153, side='down', length=28)
    mid_x = (a[0] + b[0]) // 2
    callout(c, mid_x, (a[1] + b[1]) // 2 - 2, 'ROPE BRIDGE', f, 90, 156, side='up', length=14)
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), cx + ROSE[0], oy + ROSE[1])
    return c


# ------------------------------------------------------------------ SHOT B: the old throne hall
STONE_IN = (mix('navy', 'ink', 0.55), mix('indigo', 'navy', 0.62), mix('indigo', 'navy', 0.3),
            mix('indigo', 'stone', 0.3), mix('stone', 'periwinkle', 0.3))
STONE_BK = (mix('navy', 'ink', 0.6), mix('navy', 'ink', 0.25), mix('indigo', 'navy', 0.55),
            mix('indigo', 'navy', 0.2), mix('indigo', 'stone', 0.3))
FLOOR = (mix('navy', 'ink', 0.5), mix('indigo', 'navy', 0.5), mix('indigo', 'grey', 0.35),
         mix('grey', 'periwinkle', 0.35), mix('stone', 'baby_blue', 0.35))
SUNT = (mix('tan', 'navy', 0.6), mix('khaki', 'indigo', 0.45), mix('paper', 'periwinkle', 0.42),
        mix('cream', 'periwinkle', 0.2), mix('cream', 'white', 0.4))
MOONT = (mix('navy', 'ink', 0.5), mix('royal', 'navy', 0.6), mix('royal', 'indigo', 0.25),
         mix('royal_lt', 'periwinkle', 0.3), 'crystal_lt')
B_W = 400
B_CAM = 84
B_ROSE = (275, 50, 17)
B_TORCH = ((106, 92), (196, 92), (356, 92))
B_WINDOWS = ((38, 44, 18, 60), (122, 44, 18, 60), (372, 44, 18, 60))
B_HOLES = ((40, 0, 40, 20), (300, 0, 26, 16))     # roof gaps (x, y, w, h)
B_PILLARS = ((98, 0), (188, 1), (212, 0), (338, 0), (6, 2))         # (x, broken)
B_BANNERS = ((74, 30, 13, 52, 'sun'), (158, 30, 13, 48, 'moon'), (226, 26, 11, 62, 'sun'),
             (313, 26, 11, 62, 'moon'))


def camB(f):
    return int(round(B_CAM * cruise(seg(f, CUT_B - 6, CUT_C + 6))))


def pillar(cv, x, top, base, w=12, broken=0, seed=0):
    """Half column with capital and base (moonlit right side)."""
    if broken:
        top = top + 40
    masonry(cv, x, top + 5, w, base - top - 8, STONE_IN, seed=seed, round_=True, course=6, joint=w + 1)
    if not broken:
        cv.rect(x - 2, top, w + 4, 6, STONE_IN[0])
        cv.rect(x - 1, top + 1, w + 2, 4, STONE_IN[2])
        cv.hline(x - 1, x + w, top + 1, STONE_IN[3])
        cv.rect(x + w - 3, top + 1, 3, 4, STONE_IN[3])
    else:
        jag = [(x - 1, top + 8), (x + 2, top + 3), (x + 5, top + 6), (x + 8, top + 1), (x + w + 1, top + 5),
               (x + w + 1, top - 4), (x - 1, top - 4)]
        cv.poly(jag, KEY)
    cv.rect(x - 2, base - 4, w + 4, 4, STONE_IN[0])
    cv.rect(x - 1, base - 3, w + 2, 3, STONE_IN[2])
    cv.hline(x - 1, x + w, base - 3, STONE_IN[3])


def throne(cv, x, y, ramp, emblem):
    """Empty throne seen from the front; (x, y) = bottom-left on the dais. 22 x 40."""
    o, d, m, l, hi = ramp
    w = 22
    # tall back with a rounded top
    cv.rect(x + 2, y - 40, w - 4, 30, o)
    cv.ellipse(x + w / 2 - 0.5, y - 40, (w - 4) / 2, 6, o)
    cv.rect(x + 3, y - 39, w - 6, 29, m)
    cv.ellipse(x + w / 2 - 0.5, y - 40, (w - 6) / 2, 5, m)
    cv.rect(x + w - 6, y - 40, 3, 30, l)
    cv.rect(x + 3, y - 39, 2, 29, d)
    # emblem disc on the back
    ex, ey = x + w // 2 - 1, y - 32
    if emblem == 'sun':
        for k in range(8):
            a = k * math.pi / 4
            cv.px(ex + round(math.cos(a) * 6), ey + round(math.sin(a) * 6), 'gold')
            cv.px(ex + round(math.cos(a) * 5), ey + round(math.sin(a) * 5), 'amber')
        disc(cv, ex, ey, 4, 'amber')
        disc(cv, ex, ey, 3, 'gold')
        disc(cv, ex - 1, ey - 1, 1, 'butter')
    else:
        disc(cv, ex, ey, 5, mix('navy', 'ink', 0.3))
        disc(cv, ex, ey, 4, 'crystal_dk')
        disc(cv, ex, ey, 3, 'crystal_lt')
        disc(cv, ex + 2, ey - 1, 3, 'crystal_dk')
        cv.px(ex + 3, ey + 2, 'white')
        cv.px(ex + 2, ey - 3, 'white')
    # gem finials on the top corners
    for gx in (x + 3, x + w - 4):
        cv.px(gx, y - 46, 'gold' if emblem == 'sun' else 'crystal_lt')
    # seat, arms, legs
    cv.rect(x, y - 14, w, 5, o)
    cv.rect(x + 1, y - 13, w - 2, 3, l)
    cv.hline(x + 1, x + w - 2, y - 13, hi)
    cv.rect(x - 1, y - 20, 5, 8, o)
    cv.rect(x, y - 19, 3, 6, m)
    cv.rect(x + w - 4, y - 20, 5, 8, o)
    cv.rect(x + w - 3, y - 19, 3, 6, l)
    cv.px(x + 1, y - 21, 'gold' if emblem == 'sun' else 'crystal_lt')
    cv.px(x + w - 2, y - 21, 'gold' if emblem == 'sun' else 'crystal_lt')
    cv.rect(x + 1, y - 9, w - 2, 9, o)
    cv.rect(x + 2, y - 9, w - 4, 8, d)
    cv.rect(x + 4, y - 7, w - 8, 5, m)
    cv.rect(x + w - 6, y - 7, 2, 5, l)
    # cushion (faded) + trim
    cv.rect(x + 3, y - 15, w - 6, 2, mix('wine', 'royal', 0.3) if emblem == 'moon' else mix('rose', 'wine', 0.4))
    cv.hline(x + 3, x + w - 4, y - 25, 'gold' if emblem == 'sun' else 'crystal')
    # cracks + dust
    cv.line(x + 6, y - 38, x + 8, y - 30, d)


def chandelier(cv, x, y):
    """Fallen iron chandelier lying tilted on the floor at (x, y)."""
    ir = (mix('ink', 'navy', 0.3), mix('steel_dk', 'navy', 0.35), mix('steel', 'periwinkle', 0.3))
    cv.ellipse(x, y - 3, 13, 3, ir[0], fill=False)
    cv.ellipse(x, y - 4, 13, 3, ir[1], fill=False)
    for k, dx in enumerate((-10, -5, 0, 5, 10)):
        hh = 4 + (k * 3) % 4
        yy = y - 4 + (3 if abs(dx) < 8 else 1)
        cv.vline(x + dx, yy - hh, yy, mix('paper', 'periwinkle', 0.35))
        cv.px(x + dx, yy - hh, mix('cream', 'periwinkle', 0.2))
        cv.px(x + dx, yy, ir[0])
    cv.line(x - 2, y - 6, x + 6, y - 14, ir[1])
    cv.px(x + 6, y - 14, ir[2])


@lru_cache(maxsize=None)
def hallB():
    """The throne hall as a dollhouse cross-section (400 x 180), sky cut out (transparent)."""
    cv = _new(B_W, H)
    # back wall
    masonry(cv, 0, 18, B_W, 150 - 18, STONE_BK, seed=31, course=6, joint=10, ao=0.0)
    decay(cv, [(0, 30, 90, 110), (90, 30, 120, 110), (210, 80, 120, 60), (330, 30, 70, 110)], 32,
          holes=6, streaks=8, moss=8, ramp=STONE_BK)
    # broken top edge of the wall (jagged); the middle span has collapsed (sky + moon visible)
    r = rng(33)
    top = [(-1, 26)]
    for x in range(0, B_W + 8, 6):
        if 104 <= x <= 214:
            yb = 40 + 14 * math.sin((x - 104) / 110 * math.pi) ** 0.7
            top.append((x, int(yb + r.integers(-4, 5))))
        else:
            top.append((x, int(17 + r.integers(0, 9))))
    top.append((B_W + 1, 26))
    cv.poly([(-1, -1)] + top + [(B_W + 1, -1)], KEY)
    # vault ribs between pillar tops (the middle one broken) + roof slabs on the intact spans
    caps = [6, 98, 212, 338, 398]
    for i in range(len(caps) - 1):
        x0, x1 = caps[i] + 6, caps[i + 1] + 6
        pts = bez([(x0, 50), (x0 + 4, 4), (x1 - 4, 4), (x1, 50)], 0.5)
        if i == 1:
            pts = [p for k, p in enumerate(pts) if k < len(pts) * 0.2 or k > len(pts) * 0.86]
        for (px, py) in pts:
            cv.rect(int(px) - 1, int(py) - 1, 4, 4, STONE_IN[0])
        for (px, py) in pts:
            cv.rect(int(px), int(py), 2, 2, STONE_IN[2])
            cv.px(int(px) + 1, int(py), STONE_IN[4])
    for (sx0, sx1) in ((0, 104), (214, B_W)):
        cv.rect(sx0, 0, sx1 - sx0, 9, STONE_IN[0])
        for xx in range(sx0, sx1, 9):
            cv.rect(xx + 1, 1, min(7, sx1 - xx - 1), 6, STONE_IN[1])
            cv.hline(xx + 1, min(xx + 7, sx1 - 1), 6, STONE_IN[2])
    cv.poly([(100, -1), (104, 12), (108, 6), (112, 22), (116, -1)], KEY)
    cv.poly([(206, -1), (210, 14), (214, 8), (218, 18), (222, -1)], KEY)
    for (hx, hy, hw, hh) in B_HOLES:
        jag = [(hx, -1), (hx + 2, hh * 0.6), (hx + hw * 0.3, hh), (hx + hw * 0.55, hh * 0.7), (hx + hw * 0.8, hh),
               (hx + hw, hh * 0.4), (hx + hw + 2, -1)]
        cv.poly(jag, KEY)
    # windows: lancets open to the sky, a few leaded panes left at the bottom
    for (wx, wy, ww, wh) in B_WINDOWS:
        cv.rect(wx - 3, wy - 3, ww + 6, wh + 6, STONE_IN[0])
        arch(cv, wx - 2, wy - 2, ww + 4, wh + 4, STONE_IN[3], frame=STONE_IN[0], sill=False)
        arch(cv, wx, wy, ww, wh, KEY, frame=STONE_IN[0], sill=False)
        cv.vline(wx + ww // 2, wy + 6, wy + wh - 2, STONE_IN[0])
        cv.vline(wx + ww // 2 + 1, wy + 6, wy + wh - 2, STONE_IN[2])
        for k, cc in enumerate(('crystal_dk', 'royal', 'amber', 'crystal', 'wine')):
            px = wx + 1 + k * 3
            if px + 2 < wx + ww and k != 2:
                cv.rect(px, wy + wh - 8 + (k % 2) * 2, 2, 6 - (k % 2) * 2, mix(cc, 'navy', 0.3))
        cv.rect(wx - 3, wy + wh, ww + 6, 2, STONE_IN[3])
        cv.hline(wx - 3, wx + ww + 2, wy + wh + 2, STONE_IN[0])
    # the great rose window behind the thrones
    rx, ry, rr = B_ROSE
    disc(cv, rx, ry, rr + 5, STONE_IN[0])
    disc(cv, rx, ry, rr + 4, STONE_IN[3])
    disc(cv, rx, ry, rr + 3, STONE_IN[1])
    rose_window(cv, rx, ry, rr)
    # pillars
    for (px, br) in B_PILLARS:
        pillar(cv, px, 44, 150, broken=br, seed=px)
    # banner rods
    for (bx, by, bw, bl, kind) in B_BANNERS:
        cv.hline(bx - 2, bx + bw + 1, by - 1, mix('steel_dk', 'navy', 0.3))
        cv.px(bx - 2, by - 2, 'gold')
        cv.px(bx + bw + 1, by - 2, 'gold')
    # torch sconces
    for (tx, ty) in B_TORCH:
        cv.rect(tx - 2, ty + 2, 5, 2, mix('steel_dk', 'navy', 0.3))
        cv.rect(tx - 1, ty - 3, 3, 6, WOOD_N[0])
        cv.vline(tx, ty - 3, ty + 2, WOOD_N[2])
        cv.px(tx - 2, ty + 4, mix('steel_dk', 'navy', 0.3))
        cv.px(tx + 2, ty + 4, mix('steel_dk', 'navy', 0.3))
    # floor slab + foundations (cross-section below the floor)
    masonry(cv, 0, 150, B_W, 30, FLOOR, seed=34, course=7, joint=14, ao=0.0)
    cv.rect(0, 150, B_W, 2, FLOOR[3])
    cv.hline(0, B_W, 150, FLOOR[4])
    cv.hline(0, B_W, 152, FLOOR[0])
    for xx in range(0, B_W, 12):
        cv.vline(xx, 150, 151, FLOOR[1])
    cv.rect(0, 160, B_W, 20, KEY)
    masonry(cv, 0, 158, B_W, 22, STONE_BK, seed=35, course=7, joint=16, ao=0.0)
    cv.rect(0, 156, B_W, 2, FLOOR[0])
    cv.rect(0, 153, B_W, 3, FLOOR[2])
    # stairwell under the dais (leads down to the cave)
    sx = 292
    cv.rect(sx, 154, 48, 26, mix('ink', 'navy', 0.25))
    arch(cv, sx - 2, 154, 52, 30, mix('ink', 'navy', 0.25), frame=STONE_IN[0], sill=False)
    for k in range(7):
        cv.rect(sx + 4 + k * 6, 160 + k * 3, 10, 2, STONE_IN[3])
        cv.rect(sx + 4 + k * 6, 162 + k * 3, 10, 1, STONE_IN[1])
    # dais steps + carpet
    for k, (x0, x1) in enumerate(((222, 330), (226, 326), (230, 322))):
        yy = 147 - k * 3
        cv.rect(x0, yy, x1 - x0, 3, FLOOR[0])
        cv.rect(x0 + 1, yy, x1 - x0 - 2, 2, FLOOR[2])
        cv.hline(x0 + 1, x1 - 2, yy, FLOOR[4])
    carpet = (mix('wine', 'navy', 0.35), mix('wine', 'crimson', 0.3), mix('crimson', 'rose', 0.25))
    cv.rect(20, 148, 202, 2, carpet[1])
    cv.hline(20, 221, 148, carpet[2])
    for k in range(3):
        cv.rect(246 + k * 4, 147 - k * 3, 44 - k * 8, 1, carpet[2])
    for (hx, hw) in ((60, 6), (131, 4), (170, 9)):
        cv.rect(hx, 148, hw, 2, FLOOR[2])
    for k in range(0, 8, 2):
        cv.px(20 - k // 2, 149, carpet[1])
    # thrones
    throne(cv, 252, 141, SUNT, 'sun')
    throne(cv, 280, 141, MOONT, 'moon')
    # fallen things
    chandelier(cv, 150, 149)
    rubble(cv, 186, 150, 22, 41, STONE_IN)
    rubble(cv, 10, 150, 16, 42, STONE_IN)
    rubble(cv, 352, 150, 12, 43, STONE_IN)
    for (dx, dy) in ((200, 146), (176, 147)):
        cv.ellipse(dx, dy, 5, 3, STONE_IN[0])
        cv.ellipse(dx, dy, 4, 2, STONE_IN[2])
        cv.hline(dx - 3, dx + 3, dy - 2, STONE_IN[3])
    # ivy + moss
    for (vx, vy, w, L, sd) in ((0, 24, 8, 40, 51), (60, 20, 6, 22, 52), (96, 46, 4, 18, 53), (216, 26, 8, 34, 54),
                                (330, 22, 6, 26, 55), (384, 20, 8, 46, 56)):
        ivy(cv, vx, vy, w, L, sd)
    for xx in range(0, B_W, 3):
        if hash01(xx, 77) < 0.3:
            cv.px(xx, 149, MOSS_N[1])
            if hash01(xx, 78) < 0.4:
                cv.px(xx, 148, MOSS_N[2])
    rgba = _rgba(cv)
    rimlight(rgba, mix(MOONLIT, 'indigo', 0.25), 0.55, dirs=((0, -1),))
    return rgba


@lru_cache(maxsize=None)
def banner_rgba(w, L, kind):
    cv = _new(w, L)
    if kind == 'sun':
        base, dk, em, em2 = mix('cream', 'periwinkle', 0.35), mix('khaki', 'indigo', 0.4), 'gold', 'amber'
    else:
        base, dk, em, em2 = mix('royal', 'navy', 0.45), mix('navy', 'ink', 0.4), 'crystal_lt', 'crystal'
    cv.rect(0, 0, w, L, base)
    cv.vline(0, 0, L, dk)
    cv.vline(w - 1, 0, L, dk)
    cv.hline(0, w - 1, 2, em2)
    cv.hline(0, w - 1, L - 9, em2)
    cx, cy = w // 2, L // 3
    if kind == 'sun':
        for k in range(8):
            a = k * math.pi / 4
            cv.px(cx + round(math.cos(a) * 4), cy + round(math.sin(a) * 4), em2)
        disc(cv, cx, cy, 2, em)
    else:
        disc(cv, cx, cy, 3, em)
        disc(cv, cx + 2, cy - 1, 3, base)
        cv.px(cx - 2, cy + 6, 'white')
        cv.px(cx + 2, cy + 9, 'white')
    # tattered bottom + holes
    r = rng(w * 7 + L)
    for x in range(w):
        cut = int(r.integers(0, 9)) if x % 2 == 0 else int(r.integers(0, 4))
        cv.rect(x, L - cut, 1, cut, KEY)
    for _ in range(3):
        hx, hy = int(r.integers(1, w - 2)), int(r.integers(L // 2, L - 6))
        cv.rect(hx, hy, 2, 2, KEY)
    return _rgba(cv)


def draw_banner(c, x, y, f, w, L, kind, seed):
    spr = banner_rgba(w, L, kind)
    for row in range(L):
        t = row / L
        dx = int(round(math.sin(f / 21.0 + seed - row * 0.07) * 1.6 * t))
        c.blit(spr[row:row + 1], x + dx, y + row)


def torch_flame(c, x, y, f, seed):
    """Flickering torch flame with base at (x, y)."""
    ph = (f // 3 + seed) % 4
    hgt = (5, 6, 4, 6)[ph]
    lean = (0, 1, 0, -1)[ph]
    c.vline(x, y - 1, y, 'orange')
    c.px(x - 1, y - 1, 'vermilion')
    c.px(x + 1, y - 1, 'vermilion')
    for k in range(hgt):
        t = k / hgt
        xx = x + (lean if t > 0.5 else 0)
        c.px(xx, y - 1 - k, 'amber' if t < 0.4 else 'gold' if t < 0.8 else 'butter')
        if t < 0.5:
            c.px(xx - 1, y - 1 - k, 'orange')
            c.px(xx + 1, y - 1 - k, 'amber')
    if hash01(seed, f // 5) < 0.4:
        c.px(x + lean, y - hgt - 2 - (f % 3), 'gold')


def render_B(f):
    cam = camB(f)
    c = sky_bg().copy()
    mxs = 120 - int(cam * 0.25)
    draw_stars(c, f, 9, 120, W + 40, 120, ox=int(cam * 0.2), mask_moon=(mxs, 30, 24))
    draw_moon(c, f, mxs, 30, 24, halo=0.9)
    draw_wisps(c, f, ((60, 4, 300, 40, 0.1), (48, 7, 120, 18, 0.07)), ox=int(cam * 0.25))
    bx = int(150 + 34 * smooth(seg(f, CUT_B, CUT_C)) - cam * 0.35)
    draw_balloon(c, bx, 40, f, size='s')
    c.blit(hallB(), -cam, 0)
    x0 = -cam
    # moonbeams through the roof gaps + windows (light from the upper right)
    k = 0.85 + 0.15 * math.sin(f / 23.0)
    fx.light_beam(c, ((x0 + 130, 50), (x0 + 168, 50)), ((x0 + 176, 150), (x0 + 226, 150)),
                  mix('baby_blue', 'cloud', 0.4), strength=0.12 * k)
    for (hx, hy, hw, hh) in B_HOLES:
        fx.light_beam(c, ((x0 + hx + 4, hh), (x0 + hx + hw - 4, hh)), ((x0 + hx + 40, 150), (x0 + hx + hw + 50, 150)),
                      mix('baby_blue', 'cloud', 0.4), strength=0.12 * k)
    for (wx, wy, ww, wh) in B_WINDOWS:
        fx.light_beam(c, ((x0 + wx + 2, wy + 10), (x0 + wx + ww - 2, wy + 10)),
                      ((x0 + wx + 26, 150), (x0 + wx + ww + 34, 150)), 'periwinkle', strength=0.07 * k)
    # rose window: colored light on the dais
    rx, ry, rr = B_ROSE
    pulse = 0.85 + 0.15 * math.sin(f / 9.0)
    window_glow(c, x0 + rx, ry, f, 1.1)
    for (lx, cc) in ((-16, 'amber'), (14, 'crystal')):
        c.ellipse(x0 + rx + lx, 143, 13, 2, mix(cc, 'indigo', 0.55))
        c.ellipse(x0 + rx + lx, 143, 8, 1, mix(cc, 'indigo', 0.35))
    # torches
    for i, (tx, ty) in enumerate(B_TORCH):
        fl = 0.8 + 0.2 * hash01(i, f // 3)
        add_light(c, x0 + tx, ty - 4, 34, 'orange', 0.15 * fl, rings=6)
        add_light(c, x0 + tx, ty - 4, 16, 'amber', 0.18 * fl, rings=4)
        add_light(c, x0 + tx, ty - 5, 6, 'butter', 0.2 * fl, rings=2)
        torch_flame(c, x0 + tx, ty - 3, f, i)
    # stairwell glow from below (the Tree of Harmony)
    add_light(c, x0 + 322, 184, 34, 'turquoise', 0.2 + 0.06 * math.sin(f / 8.0), rings=5)
    add_light(c, x0 + 330, 186, 16, 'ice', 0.2, rings=3)
    for (bx_, by_, bw_, bl_, kind) in B_BANNERS:
        draw_banner(c, x0 + bx_, by_, f, bw_, bl_, kind, bx_)
    # dust motes in the beams + magic motes
    for i, (hx, hy, hw, hh) in enumerate(B_HOLES + ((112, 0, 90, 50),)):
        fx.dust_motes(c, f, (x0 + hx + 20, hh + 20, hw + 20, 120), n=12, color=mix('baby_blue', 'white', 0.5),
                      seed=i + 3)
    motes(c, f, (x0 + 280, 120, 70, 60), 8, 11, speed=0.3)
    # foreground: dark broken column at the left + hanging ivy
    fgx = -int(cam * 1.35)
    for (px, top) in ((fgx - 8, 60), (fgx + 470, 30)):
        c.rect(px, top, 14, H - top, mix('ink', 'navy', 0.35))
        c.rect(px + 11, top + 2, 2, H - top, mix('navy', 'indigo', 0.4))
        c.poly([(px - 1, top + 6), (px + 4, top - 3), (px + 9, top + 2), (px + 15, top - 5), (px + 15, top + 6)],
               mix('ink', 'navy', 0.35))
    for (vx, L, sd) in ((20, 30, 1), (250, 22, 2), (392, 36, 3), (470, 26, 4)):
        sway_vine(c, fgx + vx, 0, L, f, sd, ramp=(MOSS_N[0], mix(MOSS_N[0], 'ink', 0.3), MOSS_N[1]))
    fx.vignette(c, 0.3, color=mix('ink', 'navy', 0.4))
    callout(c, x0 + 276, 104, 'THE OLD THRONE HALL', f, 171, 243, side='up', length=10)
    return c


# ------------------------------------------------------------------ SHOT C: descent + Tree of Harmony
S_H = 430
GROUND = 170
CAVE_TOP = 270
TREE_X, TREE_BASE = 160, 404
HEART = (160, 368)
WATER = 414
S_CASTLE = (80, GROUND - 123)
S_MOON = (134, 44, 26)
CRY_L = (mix('royal', 'navy', 0.35), 'violet', 'lavender', 'lilac', 'white')
CRY_C = (mix('crystal_dk', 'navy', 0.35), 'crystal_dk', 'crystal', 'crystal_lt', 'white')
CRY_W = (mix('royal', 'periwinkle', 0.3), 'periwinkle', mix('cloud_dk', 'lilac', 0.3), 'cloud', 'white')
CRY_P = (mix('plum', 'navy', 0.3), 'raspberry', 'crystal_pk', mix('crystal_pk', 'white', 0.5), 'white')
CAVE = (mix('ink', 'navy', 0.55), mix('navy', 'eggplant', 0.45), mix('indigo', 'eggplant', 0.4),
        mix('violet', 'indigo', 0.55), mix('lavender', 'periwinkle', 0.45))
EARTH = (mix('ink', 'navy', 0.4), mix('navy', 'ink', 0.1), mix('navy', 'plum_brown', 0.35),
         mix('indigo', 'plum_brown', 0.3), mix('grey', 'indigo', 0.3))
# gem ornaments: (x, y, kind) in world coords, sorted left -> right (pulse order)
GEMS = ((104, 340, 'yellow'), (128, 312, 'red'), (150, 294, 'pink'), (172, 296, 'apple'), (192, 312, 'blue'),
        (216, 340, 'purple'))
GEM_COL = {'red': ('crimson', 'red', 'salmon'), 'blue': ('crystal_dk', 'sky', 'light_sky'),
           'pink': ('raspberry', 'pink', 'blush'), 'apple': ('rust', 'orange', 'clay'),
           'yellow': ('amber', 'gold', 'butter'), 'purple': ('eggplant', 'purple', 'violet')}


GEODES = ((52, 208, 9), (252, 200, 10), (104, 236, 7), (232, 240, 8), (22, 252, 7), (292, 232, 8),
          (78, 186, 6), (118, 200, 6), (214, 212, 6))


def cam_s(f):
    if f < DESC1:
        return int(round(140 + 110 * cruise(seg(f, DESC0, DESC1))))
    if f < RISE0:
        return 250
    return int(round(250 * (1 - cruise(seg(f, RISE0, RISE1)))))


def cave_void(w, h):
    """Polygon of the cave interior (world coords)."""
    return [(24, 430), (18, 384), (22, 334), (34, 304), (58, 286), (98, 276), (138, 278), (170, 270), (214, 274),
            (250, 284), (276, 300), (292, 332), (300, 384), (296, 430)]


def shard(cv, x, y, ang, L, w, ramp):
    """Crystal prism from (x, y) toward angle ang (radians), length L, width w; lit facet on the right."""
    ca, sa = math.cos(ang), math.sin(ang)
    nx, ny = -sa, ca
    hw = w / 2
    tip = (x + ca * L, y + sa * L)
    m = 0.72
    pl = (x + ca * L * m + nx * hw, y + sa * L * m + ny * hw)
    pr = (x + ca * L * m - nx * hw, y + sa * L * m - ny * hw)
    bl = (x + nx * hw, y + ny * hw)
    br = (x - nx * hw, y - ny * hw)
    mid_b = (x, y)
    mid_t = (x + ca * L * m, y + sa * L * m)
    # outline, then two facets (dark one / lit one), then a highlight spine
    cv.poly([bl, pl, tip, pr, br], ramp[0])
    shrink = 0.8
    def sh(p):
        return (x + (p[0] - x) * shrink + ca * 0.6, y + (p[1] - y) * shrink + sa * 0.6)
    lit_left = nx < 0           # which side faces right (the light)
    f1 = [sh(bl), sh(pl), (tip[0] - ca, tip[1] - sa), mid_t, mid_b]
    f2 = [mid_b, mid_t, (tip[0] - ca, tip[1] - sa), sh(pr), sh(br)]
    cv.poly(f1, ramp[3] if lit_left else ramp[1])
    cv.poly(f2, ramp[1] if lit_left else ramp[3])
    cv.poly([mid_b, mid_t, (tip[0] - ca * 2, tip[1] - sa * 2)], ramp[2])
    cv.line(mid_t[0], mid_t[1], tip[0] - ca, tip[1] - sa, ramp[4])


def crystal_cluster(cv, x, y, n, size, ramp, seed, up=-math.pi / 2, spread=1.0):
    r = rng(seed)
    items = []
    for i in range(n):
        a = up + (r.uniform(-0.5, 0.5)) * spread * 1.6
        L = size * r.uniform(0.6, 1.1)
        items.append((abs(a - up), a, L))
    items.sort(reverse=True)
    for _, a, L in items:
        shard(cv, x + r.uniform(-size * 0.25, size * 0.25), y, a, L, max(3, L * 0.38), ramp)


def crystal_puff(cv, cx, cy, r, ramp, seed, back=False):
    """A round 'canopy' made of crystal shards radiating from (cx, cy)."""
    rg = rng(seed)
    disc(cv, cx, cy + 1, int(r * 0.62), ramp[0])
    disc(cv, cx, cy + 1, int(r * 0.62) - 1, ramp[1])
    n = 11
    items = []
    for i in range(n):
        a = -math.pi - 0.35 + (i + rg.uniform(0.2, 0.8)) / n * (math.pi + 0.7)
        L = r * rg.uniform(0.75, 1.08)
        items.append((-abs(a + math.pi / 2), a, L))
    items.sort()
    for _, a, L in items:
        shard(cv, cx + math.cos(a) * r * 0.2, cy + math.sin(a) * r * 0.2 + 2, a, L, max(4, r * 0.48), ramp)
    if not back:
        for i in range(4):
            a = -math.pi / 2 + rg.uniform(-0.9, 0.9)
            shard(cv, cx + rg.uniform(-r * 0.3, r * 0.3), cy + r * 0.25, a, r * 0.55, max(3, r * 0.34),
                  (ramp[0], ramp[2], ramp[3], ramp[4], 'white'))


CRY_TRUNK = (mix('royal', 'navy', 0.3), 'royal_lt', 'lavender', 'lilac', 'crystal_lt', 'white')
PUFFS = ((112, 334, 17, 'L', True), (208, 334, 17, 'C', True), (132, 312, 18, 'C', False),
         (188, 312, 18, 'L', False), (160, 298, 20, 'W', False), (100, 346, 13, 'W', False),
         (220, 346, 13, 'W', False), (146, 328, 14, 'L', False), (174, 328, 14, 'C', False))


@lru_cache(maxsize=None)
def tree_rgba():
    """The Tree of Harmony (world coords, strip-sized RGBA 320 x 430) - static parts."""
    cv = _new(W, S_H)
    tx, tb = TREE_X, TREE_BASE
    ramps = {'L': CRY_L, 'C': CRY_C, 'W': CRY_W}
    limb = (CRY_TRUNK[0], CRY_TRUNK[1], CRY_TRUNK[2], CRY_TRUNK[3])
    # roots: crystal limbs sinking into the mound
    for (ex, ey, r0) in ((112, 410, 4), (208, 410, 4), (128, 404, 3), (192, 404, 3), (90, 412, 2), (230, 412, 2)):
        tube(cv, bez([(tx + (ex - tx) * 0.15, tb - 6), (tx + (ex - tx) * 0.6, tb - 4), (ex, ey)], 0.5), r0 + 2, 1, limb)
    # limbs from the trunk top to the canopy puffs
    for (px, py, pr, rk, bk) in PUFFS:
        if abs(px - tx) < 4:
            continue
        tube(cv, bez([(tx, tb - 52), (tx + (px - tx) * 0.35, tb - 64), (px, py + 4)], 0.5), 4, 2, limb)
    # canopy: back puffs first
    for (px, py, pr, rk, bk) in sorted(PUFFS, key=lambda p: (not p[4], -p[1])):
        crystal_puff(cv, px, py, pr, haze(ramps[rk], 0.25, 'royal') if bk else ramps[rk], int(px * 3 + py), back=bk)
    # faceted trunk (twisting facets), flaring at the base
    ftone = (1, 2, 4, 3, 2)
    for y in range(tb - 58, tb + 1):
        t = (y - (tb - 58)) / 58.0
        hw = 9 + 13 * t ** 2.2
        xl, xr = int(round(tx - hw)), int(round(tx + hw))
        for x in range(xl, xr + 1):
            u = (x - (tx - hw)) / max(1.0, 2 * hw)
            fi = int(u * 5 + (y - tb) * 0.045) % 5
            edge = abs(((u * 5 + (y - tb) * 0.045) % 1.0) - 0.0) < 0.12
            cc = CRY_TRUNK[ftone[fi]] if not edge else CRY_TRUNK[1]
            if x == xl or x == xr:
                cc = CRY_TRUNK[0]
            cv.px(x, y, cc)
        if (y * 7) % 13 == 0:
            cv.px(int(tx + hw * 0.4), y, 'white')
    rgba = _rgba(cv)
    return rgba


def draw_gem(c, x, y, kind, k=0.0):
    """Gem ornament (screen coords) with glow level k."""
    d, m, l = GEM_COL[kind]
    o = mix(d, 'ink', 0.45)
    if k > 0.05:
        add_light(c, x, y, int(8 + 10 * k), m, 0.25 + 0.35 * k, rings=3)
    if kind == 'apple':
        c.ellipse(x, y, 4, 4, o)
        c.ellipse(x, y, 3, 3, m)
        c.px(x - 1, y - 2, l)
        c.px(x - 2, y - 1, l)
        c.px(x, y - 4, mix('plum_brown', 'ink', 0.3))
        c.px(x + 1, y - 5, 'leaf')
        c.px(x + 2, y - 5, 'green')
        c.px(x, y - 5, 'leaf')
        c.px(x + 1, y - 4, 'green')
        c.px(x, y - 3, mix('plum_brown', 'ink', 0.3))
    elif kind == 'yellow':
        c.poly([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], o)
        c.poly([(x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)], m)
        c.poly([(x, y - 4), (x + 4, y), (x, y)], l)
    elif kind == 'purple':
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = 5.6 if i % 2 == 0 else 2.4
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        c.poly(pts, o)
        pts2 = [(x + (px - x) * 0.72, y + (py - y) * 0.72) for (px, py) in pts]
        c.poly(pts2, m)
        c.px(x, y - 1, l)
    else:
        c.ellipse(x, y, 4, 4, o)
        c.ellipse(x, y, 3, 3, m)
        c.ellipse(x + 1, y + 1, 1, 1, d)
        c.px(x - 1, y - 2, l)
        c.px(x - 2, y - 1, l)
        c.px(x - 1, y - 1, 'white')
    if k > 0.4:
        fx.sparkle(c, x + 3, y - 3, 6, 'white', period=12, size=2)


def draw_heart(c, x, y, f, k=0.0):
    """The star at the tree's heart (5-pointed, magenta/pink with a white core)."""
    rr = 7 + int(k * 1.5)
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rad = rr if i % 2 == 0 else rr * 0.45
        pts.append((x + math.cos(a) * rad, y + math.sin(a) * rad))
    add_light(c, x, y, 18 + int(14 * k), 'magic', 0.22 + 0.3 * k, rings=4)
    c.poly([(x + (px - x) * 1.25, y + (py - y) * 1.25) for (px, py) in pts], mix('plum', 'ink', 0.3))
    c.poly(pts, 'magic')
    c.poly([(x + (px - x) * 0.6, y + (py - y) * 0.6) for (px, py) in pts], mix('magic', 'white', 0.5))
    disc(c, x, y, 1, 'white')


@lru_cache(maxsize=None)
def strip_rgba():
    """Static world strip (320 x 430): castle on the crag, cut-away rock with the stair shaft + fissure,
    the Cave of Harmony (back wall, ceiling, stalactites, crystal clusters, mound, banks) and the tree."""
    cv = _new(W, S_H)
    # cave back wall
    back = boulder_rock(W, S_H, [(0, 240), (W, 240), (W, S_H), (0, S_H)], CAVE, 61, cell=(17, 10), dark_depth=0.3)
    br = _rgba(back)
    yy, xx = np.mgrid[0:S_H, 0:W]
    fac = np.clip(np.sqrt(((xx - TREE_X) / 1.25) ** 2 + (yy - 340) ** 2) / 150.0, 0, 1) ** 1.2
    lv = np.clip(np.floor(fac * 3 + bayer(S_H, W)) / 3, 0, 1)[..., None]
    rgb = br[..., :3].astype(np.float32)
    rgb += (np.array(col(mix('ink', 'navy', 0.3)), np.float32) - rgb) * (0.3 + lv * 0.6)
    br[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    cv.blit(br, 0, 0)
    for (x, y, n, sz, rp, sd) in ((52, 360, 5, 12, CRY_L, 1), (262, 352, 5, 14, CRY_C, 2), (84, 300, 3, 8, CRY_W, 3),
                                  (240, 304, 3, 9, CRY_L, 4), (40, 396, 4, 10, CRY_C, 5), (280, 398, 4, 10, CRY_P, 6)):
        crystal_cluster(cv, x, y, n, sz, haze(rp, 0.5, CAVE[1]), 200 + sd)
    # front rock (cut-away) around the void + below the ground
    front = boulder_rock(W, S_H, [(0, GROUND), (W, GROUND), (W, S_H), (0, S_H)], EARTH, 62, cell=(13, 8),
                         dark_depth=1.1, y_ref=GROUND)
    fr = _rgba(front)
    vmask = _new(W, S_H)
    vmask.poly(cave_void(W, S_H), 'white')
    vm = np.all(vmask.a == np.array(col('white'), np.uint8), axis=2)
    # shaft (stairwell) from the keep portal down into the cave
    sm = np.zeros_like(vm)
    sm[GROUND + 2:CAVE_TOP + 6, 150:194] = True
    fissure = _new(W, S_H)
    pts = bez([(56, GROUND), (70, 200), (58, 232), (80, 262)], 0.5)
    for (px, py) in pts:
        disc(fissure, px, py, 1, 'white')
    fm = np.all(fissure.a == np.array(col('white'), np.uint8), axis=2)
    cut = vm | sm | fm
    fr[cut, 3] = 0
    # the cut face darkens with depth (a dark frame around the lit cave)
    yy, xx = np.mgrid[0:S_H, 0:W]
    fac = np.clip((yy - 200) / 120.0, 0, 1)
    lv = np.clip(np.floor(fac * 3 + bayer(S_H, W)) / 3, 0, 1)[..., None]
    rgb = fr[..., :3].astype(np.float32)
    rgb += (np.array(col(mix('ink', 'navy', 0.25)), np.float32) - rgb) * lv * 0.6
    # crystal-lit rim (2 px) on the cut edges, strongest near the tree
    a = fr[..., 3] >= 128
    nb = np.roll(cut, 1, 0) | np.roll(cut, -1, 0) | np.roll(cut, 1, 1) | np.roll(cut, -1, 1)
    edge = a & nb
    cut2 = cut | edge
    edge2 = a & ~edge & (np.roll(cut2, 1, 0) | np.roll(cut2, -1, 0) | np.roll(cut2, 1, 1) | np.roll(cut2, -1, 1))
    dist = np.sqrt((xx - HEART[0]) ** 2 + (yy - HEART[1]) ** 2)
    near = dist < 175
    c1 = np.array(col(mix('crystal_lt', 'lilac', 0.45)), np.float32)
    c2 = np.array(col(mix('royal_lt', 'crystal_dk', 0.4)), np.float32)
    c3 = np.array(col(mix('indigo', 'periwinkle', 0.4)), np.float32)
    rgb[edge & near] = c1
    rgb[edge2 & near] = c2
    rgb[edge & ~near] = c3
    fr[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    # shaft interior: dark wall + zigzag stair + crystal lamps
    sh = Canvas(arr=cv.a)
    sh.rect(150, GROUND + 2, 44, CAVE_TOP - GROUND + 4, mix('ink', 'navy', 0.35))
    masonry(sh, 152, GROUND + 2, 40, 40, STONE_BK, seed=63, course=5, joint=8, ao=0.0)
    sh.rect(150, GROUND + 44, 44, CAVE_TOP - GROUND - 38, mix('ink', 'eggplant', 0.3))
    y = GROUND + 10
    left = True
    while y < CAVE_TOP - 4:
        for st in range(6):
            sx = (154 + st * 5) if left else (186 - st * 5)
            sh.rect(sx, y + st * 2, 6, 2, STONE_IN[3])
            sh.hline(sx, sx + 5, y + st * 2, STONE_IN[4])
            sh.rect(sx, y + st * 2 + 2, 6, 1, STONE_IN[0])
        y += 14
        left = not left
    cv.blit(fr, 0, 0)
    # soil band + grass + roots just under the surface
    soil = (mix('ink', 'eggplant', 0.5), mix('plum_brown', 'eggplant', 0.5), mix('plum_brown', 'mauve', 0.35))
    for x in range(W):
        if 150 <= x < 194:
            continue
        d = 9 + int(hash01(x, 5) * 3 + 2 * math.sin(x / 17.0))
        cv.rect(x, GROUND, 1, d, soil[1])
        if hash01(x, 9) < 0.5:
            cv.px(x, GROUND + 4 + int(hash01(x, 10) * 4), soil[0])
        cv.px(x, GROUND + d, soil[0])
        if hash01(x, 6) < 0.3:
            cv.px(x, GROUND + 2, soil[2])
    for (x0, sd) in ((276, 1), (290, 2), (20, 3), (250, 4)):
        pts = bez([(x0, GROUND + 2), (x0 + 6 - 12 * hash01(sd, 1), GROUND + 20), (x0 - 4, GROUND + 40),
                   (x0 + 8 * hash01(sd, 2), GROUND + 60)], 0.6)
        for j, (px, py) in enumerate(pts):
            cv.px(px, py, BARK_N[1] if j % 4 else BARK_N[2])
            if j > 20 and j % 7 == 0:
                cv.px(px + 1, py + 1, BARK_N[0])
    # crystal veins through the rock (static base colour; pulses drawn per frame)
    for vein in VEINS_ROCK:
        for (px, py) in vein:
            ix, iy = int(px), int(py)
            if 0 <= ix < W and 0 <= iy < S_H and fr[iy, ix, 3] >= 128:
                cv.a[iy, ix] = col(mix('crystal_dk', 'royal', 0.2))
                if ix + 1 < W and fr[iy, ix + 1, 3] >= 128:
                    cv.a[iy, ix + 1] = col(mix('navy', 'crystal_dk', 0.4))
    for i, (gx, gy, sz) in enumerate(GEODES):
        disc(cv, gx, gy, sz // 2 + 2, mix('ink', 'navy', 0.3))
        crystal_cluster(cv, gx, gy + sz // 3, 4, sz, (CRY_C, CRY_L, CRY_P)[i % 3], 700 + i, up=-1.57, spread=1.3)
    for fx0 in (136, 198):
        masonry(Canvas(arr=cv.a), fx0, GROUND + 1, 10, 52, STONE_BK, seed=fx0, course=6, joint=11, ao=0.3)
        cv.a[GROUND + 50:GROUND + 54, fx0:fx0 + 10] = col(mix('ink', 'navy', 0.4))
    # stalactites along the cave ceiling + crystal clusters on the rims
    r = rng(64)
    vpts = cave_void(W, S_H)
    for x in range(10, 310, 9):
        top = None
        for yy0 in range(240, 340):
            if vm[yy0, x]:
                top = yy0
                break
        if top is None or 150 <= x < 194:
            continue
        L = int(r.integers(4, 13))
        w = int(r.integers(2, 4))
        cv.poly([(x - w, top - 1), (x + w, top - 1), (x + (1 if L > 8 else 0), top + L)], EARTH[0])
        cv.poly([(x - w + 1, top - 1), (x + w - 1, top - 1), (x, top + L - 1)], EARTH[2])
        cv.vline(x + w - 1, top, top + L // 2, EARTH[3])
        if r.random() < 0.3:
            crystal_cluster(cv, x + 3, top, 3, 7, (CRY_C, CRY_L)[x % 2], 300 + x, up=math.pi / 2, spread=0.9)
    for (x, y, n, sz, rp, up, sd) in ((22, 300, 5, 13, CRY_C, -0.6, 7), (300, 320, 5, 13, CRY_L, -2.5, 8),
                                      (12, 405, 4, 12, CRY_L, -1.2, 9), (308, 410, 4, 12, CRY_C, -1.9, 10),
                                      (60, 268, 3, 9, CRY_L, 1.2, 11), (262, 278, 3, 9, CRY_C, 1.9, 12)):
        crystal_cluster(cv, x, y, n, sz, rp, 400 + sd, up=up, spread=0.9)
    # mound + floor + pool banks
    cv.poly([(0, 410), (40, 406), (90, 402), (120, 396), (160, 394), (200, 396), (232, 402), (280, 406),
             (W, 410), (W, S_H), (0, S_H)], CAVE[0])
    mound = boulder_rock(W, S_H, [(0, 412), (30, 408), (80, 404), (118, 398), (160, 396), (202, 398), (240, 404),
                                  (290, 408), (W, 412), (W, S_H), (0, S_H)], CAVE, 65, cell=(10, 5), dark_depth=0.4)
    cv.blit(_rgba(mound), 0, 0)
    # the tree
    cv.blit(tree_rgba(), 0, 0)
    # castle on top
    cv.blit(castle_rgba(), S_CASTLE[0], S_CASTLE[1])
    # gnarled tree on the crag top (right) + grass tufts
    tr = bez([(282, GROUND + 1), (290, 150), (280, 128), (292, 104)], 0.6)
    tube(cv, tr, 5, 3, BARK_N)
    tube(cv, bez([(286, 130), (300, 116), (316, 112)], 0.6), 2, 1, BARK_N)
    canopy(cv, 296, 92, 26, 12, 16, 6, 11, LEAF_N, 66)
    for x in range(0, W, 2):
        if hash01(x, 13) < 0.5 and not (S_CASTLE[0] + 2 < x < S_CASTLE[0] + 158):
            cv.px(x, GROUND - 1, MOSS_N[2])
            if hash01(x, 14) < 0.4:
                cv.px(x, GROUND - 2, MOSS_N[1])
    cv.hline(0, W - 1, GROUND, MOSS_N[1])
    rgba = _rgba(cv)
    return rgba


def _vein(x0, y0, x1, y1, seed, wig=3.0):
    r = rng(seed)
    pts = bez([(x0, y0), (lerp(x0, x1, 0.33) + r.uniform(-12, 12), lerp(y0, y1, 0.33) + r.uniform(-wig, wig)),
               (lerp(x0, x1, 0.66) + r.uniform(-12, 12), lerp(y0, y1, 0.66) + r.uniform(-wig, wig)), (x1, y1)], 0.8)
    out, last = [], None
    for (px, py) in pts:
        q = (int(round(px)), int(round(py)))
        if q != last:
            out.append(q)
            last = q
    return tuple(out)


# glowing root veins over the cave floor (from the tree base outward) and through the rock
VEINS_FLOOR = tuple(_vein(TREE_X + dx, 402, ex, ey, 500 + i, 2.0) for i, (dx, ex, ey) in enumerate(
    ((-20, 4, 414), (-14, 50, 412), (-28, 20, 404), (20, 316, 414), (14, 270, 412), (28, 300, 405))))
VEINS_ROCK = tuple(_vein(x0, y0, x1, y1, 600 + i, 10.0) for i, (x0, y0, x1, y1) in enumerate(
    ((20, 300, 40, 200), (300, 300, 270, 196), (90, 258, 120, 190), (230, 262, 210, 192), (8, 360, 2, 250),
     (312, 350, 318, 240))))


@lru_cache(maxsize=None)
def sky_s():
    """Sky layer for the strip (parallax 0.5): gradient + far forest at the horizon (320 x 320)."""
    hh = 320
    c = Canvas(W, hh)
    c.gradient(0, 0, W, 200, SKY_N, soft=0.7)
    c.rect(0, 200, W, hh - 200, SKY_N[-1])
    cv = _new(W, hh)
    ramp = haze(LEAF_N, 0.45)
    r = rng(15)
    x = -6
    while x < W + 6:
        rr = int(r.integers(6, 11))
        leafball(cv, x, 150 + 5 * math.sin(x / 23.0), rr, ramp)
        x += rr * 1.2
    cv.rect(0, 154, W, hh - 154, ramp[1])
    rgba = _rgba(cv)
    rimlight(rgba, mix(MOONLIT, FOGC, 0.3), 0.55, dirs=((0, -1),))
    c.blit(rgba, 0, 0)
    return c


def vein_pulse(c, f, vein, oy, k_base, t_mid=None, seed=0, bright=('crystal', 'crystal_lt', 'white')):
    """Light travelling along a vein (from its start outward)."""
    n = len(vein)
    period = 48
    for i, (x, y) in enumerate(vein):
        sy = y - oy
        if not (0 <= sy < H):
            continue
        ph = ((f - i * 0.9 + seed * 11) % period) / period
        v = k_base + 0.6 * max(0.0, 1 - abs(ph - 0.5) * 6)
        if t_mid is not None:
            dd = (f - MIDNIGHT) * 5 - i
            if 0 <= dd < 18:
                v += 1.0 - dd / 18
        if v < 0.25:
            continue
        cc = bright[0] if v < 0.55 else bright[1] if v < 0.95 else bright[2]
        c.px(x, sy, cc)


def bloom(c, hx, hy, R, k, color='white'):
    """Crisp pixel bloom: pixels near (hx, hy) step toward `color` (3 dithered levels)."""
    if k <= 0.01:
        return
    y0, y1 = max(0, int(hy - R)), min(H, int(hy + R))
    x0, x1 = max(0, int(hx - R)), min(W, int(hx + R))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2) / R
    v = np.clip(1 - d, 0, 1) ** 1.4 * k
    q = np.clip(np.floor(v * 3 + bayer(y1 - y0, x1 - x0, x0, y0)) / 3, 0, 1)[..., None]
    reg = c.a[y0:y1, x0:x1].astype(np.float32)
    reg += (np.array(col(color), np.float32) - reg) * q * 0.8
    c.a[y0:y1, x0:x1] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)


def flare(c, f, hx, hy):
    """MIDNIGHT: bloom, expanding ring and a sparkle wave from the tree's heart (screen coords)."""
    t = f - MIDNIGHT
    if t < 0 or t > 60:
        return
    k = ease_out(seg(f, MIDNIGHT, MIDNIGHT + 3)) * (1 - ease_in_out(seg(f, MIDNIGHT + 4, MIDNIGHT + 40)))
    if k > 0.01:
        add_light(c, hx, hy, int(40 + 70 * k), 'lilac', 0.3 * k, rings=6)
        add_light(c, hx, hy, int(20 + 30 * k), 'white', 0.35 * k, rings=4)
        bloom(c, hx, hy - 20, 70 + 50 * k, 0.9 * k, mix('lilac', 'white', 0.6))
    if t < 30:
        rad = t * 7.5
        y0, y1 = max(0, int(hy - rad - 2)), min(H, int(hy + rad + 3))
        x0, x1 = max(0, int(hx - rad - 2)), min(W, int(hx + rad + 3))
        if y1 > y0 and x1 > x0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
            fade = 1 - t / 30
            m = (np.abs(d - rad) < 1.0) & (bayer(y1 - y0, x1 - x0, x0, y0) < fade)
            m2 = (np.abs(d - rad + 2) < 0.8) & (bayer(y1 - y0, x1 - x0, x0, y0) < fade * 0.5)
            c.a[y0:y1, x0:x1][m2] = col('lilac')
            c.a[y0:y1, x0:x1][m] = col('ice')
    for i in range(48):
        a = hash01(i, 3) * 6.283
        sp = 2.0 + 3.0 * hash01(i, 4)
        dist = t * sp * (1 - t / 120)
        x = int(hx + math.cos(a) * dist)
        y = int(hy + math.sin(a) * dist * 0.8 - t * 0.3)
        if t > 20 + 25 * hash01(i, 5):
            continue
        cc = ('white', 'ice', 'lilac', 'crystal_pk', 'gold')[i % 5]
        if (t + i) % 6 < 3:
            fx.sparkle(c, x, y, t, cc, period=12, phase=hash01(i, 6), size=2)
        else:
            c.px(x, y, cc)


@lru_cache(maxsize=None)
def fg_crystals():
    """Foreground crystal clusters (bottom corners of the cave view), dark with glowing tips."""
    cv = _new(W, 100)
    dk = (mix('ink', 'navy', 0.4), mix('navy', 'royal', 0.3), mix('royal', 'navy', 0.2), mix('royal_lt', 'crystal', 0.3),
          'crystal_lt')
    crystal_cluster(cv, 14, 100, 6, 30, dk, 901, spread=1.0)
    crystal_cluster(cv, 306, 100, 7, 34, dk, 902, spread=1.0)
    crystal_cluster(cv, 40, 100, 3, 14, dk, 903)
    rgba = _rgba(cv)
    return rgba


def render_C(f):
    cam = cam_s(f)
    c = Canvas(W, H)
    c.blit(sky_s(), 0, -int(cam * 0.5))
    mx, my, mr = S_MOON
    msy = my - int(cam * 0.5)
    if msy + mr * 3 > 0:
        draw_stars(c, f, 21, 120, W, 160, oy=int(cam * 0.5), mask_moon=(mx, msy, mr))
        draw_moon(c, f, mx, msy, mr, halo=1.0)
    draw_wisps(c, f, ((64, 4, 260, 74, 0.1), (50, 7, 60, 30, 0.07)), ox=0)
    if STAR0 <= f < STAR0 + 14 and cam < 30:            # a shooting star over the castle
        t = (f - STAR0) / 14.0
        hx_, hy_ = int(300 - 110 * t), int(22 + 30 * t) - int(cam * 0.5)
        tail = ('white', 'white', 'cloud', 'cloud', 'baby_blue', 'baby_blue', 'baby_blue', 'periwinkle',
                'periwinkle', 'periwinkle', mix('periwinkle', 'royal', 0.5), mix('periwinkle', 'royal', 0.5))
        for j, cc in enumerate(tail[:int(6 + 6 * min(1.0, t * 3))]):
            c.px(hx_ + j, hy_ - int(round(j * 0.3)), cc)
        if t < 0.85:
            c.px(hx_ - 1, hy_, 'baby_blue')
            c.px(hx_, hy_ + 1, 'baby_blue')
    # balloon: drifting toward the keep during the descent, resting on the keep at the end
    ox, oy = S_CASTLE
    if f < RISE0:
        bx = int(262 - 22 * smooth(seg(f, CUT_C, DESC1)))
        draw_balloon(c, bx, 104 - int(cam * 1.0) + 0, f, size='m')
    c.blit(strip_rgba(), 0, -cam)
    if f >= RISE0:
        draw_balloon(c, ox + KEEP_TOP[0], oy + KEEP_TOP[1] - 2 - cam, f, size='m', flame=True, bob=False)
        if (f // 3) % 2 == 0:
            add_light(c, ox + KEEP_TOP[0], oy + KEEP_TOP[1] - 6 - cam, 6, 'amber', 0.25, rings=2)
    pennant(c, ox + PENNANT[0], oy + PENNANT[1] - 6 - cam, f)
    # rose window: breathes; flashes on the last bell strikes
    wk = 1.0
    if f >= RISE1 - 10:                                 # the window glows brighter once the light arrives
        wk = 1.0 + 0.5 * seg(f, RISE1 - 10, RISE1 + 6) * (0.8 + 0.2 * math.sin(f / 8.0))
    window_glow(c, ox + ROSE[0], oy + ROSE[1] - cam, f, wk)
    add_light(c, ox + 79, oy + 112 - cam, 9, 'amber', 0.12 + 0.03 * math.sin(f / 4.0), rings=3)
    # --- cave light (only when visible)
    hx, hy = HEART[0], HEART[1] - cam
    if hy < H + 120:
        build = seg(f, CUT_C, MIDNIGHT)
        k = 0.75 + 0.2 * math.sin(f / 10.0) + 0.25 * build + (0.35 if f >= MIDNIGHT else 0)
        add_light(c, hx, hy - 50, 80, 'royal_lt', 0.18 * k, rings=6)
        add_light(c, hx, hy - 20, 110, 'royal', 0.12 * k, rings=6)
        add_light(c, hx, hy - 24, 64, 'crystal', 0.12 * k, rings=5)
        add_light(c, hx, hy, 30, 'lilac', 0.14 * k, rings=4)
        # moonbeam from the fissure
        fx.light_beam(c, ((76, 262 - cam), (84, 262 - cam)), ((100, 402 - cam), (126, 402 - cam)),
                      mix('baby_blue', 'cloud', 0.4), strength=0.12)
        for vein in VEINS_FLOOR:
            vein_pulse(c, f, vein, cam, 0.35, t_mid=True, seed=len(vein))
        for i, (gx, gy, sz) in enumerate(GEODES):
            add_light(c, gx, gy - cam, sz + 6, ('crystal', 'lavender', 'crystal_pk')[i % 3],
                      0.12 + 0.06 * math.sin(f / 11.0 + i), rings=3)
        for i, vein in enumerate(VEINS_ROCK):
            vein_pulse(c, f, vein, cam, 0.0, seed=i, bright=('crystal_dk', 'crystal', 'crystal_lt'))
        # gems: soft breathing, then pulsing in sequence on the bell strikes (every 6 frames)
        for i, (gx, gy, kind) in enumerate(GEMS):
            gk = 0.15 + 0.1 * math.sin(f / 9.0 + i)
            for n in range(12):
                s_ = MIDNIGHT + 6 * n
                if n % 6 == i and s_ <= f < s_ + 14:
                    gk = max(gk, 1 - (f - s_) / 14)
            draw_gem(c, gx, gy - cam, kind, gk)
        hk = 0.3 + 0.15 * math.sin(f / 7.0)
        if f >= MIDNIGHT:
            hk = max(hk, 1 - seg(f, MIDNIGHT, MIDNIGHT + 24))
        draw_heart(c, hx, hy, f, hk)
        # sparkles rising from the crown
        motes(c, f, (80, 250, 160, 160), 30, 41, colors=('white', 'ice', 'lilac', 'crystal_pk'), speed=0.45,
              oy=cam, sway=6)
        flare(c, f, hx, hy)
        # pool reflection
        wl = WATER - cam
        if 0 < wl < H:
            n = min(H - wl, wl)
            for k_ in range(1, n):
                src = wl - k_ * 1
                dx = int(round(math.sin(f / 6.0 + k_ * 0.9) * (0.5 + k_ * 0.05)))
                row = np.roll(c.a[src], dx, axis=0).astype(np.float32)
                tgt = np.array(col(mix('navy', 'ink', 0.4)), np.float32)
                row += (tgt - row) * min(0.85, 0.4 + k_ * 0.03)
                c.a[wl + k_] = np.clip(row + 0.5, 0, 255).astype(np.uint8)
            c.hline(0, W - 1, wl, mix('crystal_lt', 'periwinkle', 0.5))
            for i in range(6):
                rx = int((hash01(i, 7) * W + f * (0.3 + hash01(i, 8) * 0.4)) % W)
                c.hline(rx, rx + 3 + i % 3, wl + 3 + i * 2, mix('crystal', 'navy', 0.4))
    # motes rising with the camera after midnight (the light climbs to the castle)
    if f >= 300:
        motes(c, f, (110, 40, 100, 380), 40, 77, colors=('white', 'ice', 'lilac', 'gold'), speed=1.2,
              oy=cam, sway=8)
    # foreground: hanging roots near the camera (parallax 1.25) in the rock section
    for (rx, ry, L, sd) in ((12, 176, 30, 1), (306, 180, 24, 2), (36, 262, 20, 3)):
        sy = ry - int(cam * 1.25) + 60
        if -L < sy < H:
            sway_vine(c, rx, sy, L, f, sd, ramp=(BARK_N[0], BARK_N[1], BARK_N[2]))
    fgy = int(cam * 1.2)
    c.blit(fg_crystals(), 0, 470 - fgy)
    fx.vignette(c, 0.3, color=mix('ink', 'navy', 0.4))
    callout(c, 226, 318 - cam, 'TREE OF HARMONY', f, 270, 342, side='right', length=10)
    return c


def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT_B:
        c = render_A(f)
    elif f < CUT_C:
        c = render_B(f)
    else:
        c = render_C(f)
    atlas_card(c, f, 100, 200, 'ANCIENT CASTLE', ['RUINS OF THE TWO SISTERS', 'TREE OF HARMONY BELOW'])
    return c


SFX = [
    (78, 'wind', 0.45, {'variant': 0}),      # night air over the chasm
    (96, 'creak', 0.4),                      # rope bridge sways
    (134, 'creak', 0.28),
    (166, 'wind', 0.3, {'variant': 0}),      # draught through the ruined hall
    (198, 'shimmer', 0.25),                  # dust motes in the moonbeams
    (252, 'rumble', 0.35),                   # descent through the rock
    (270, 'crystal', 0.5),                   # the Tree of Harmony comes into view
    (291, 'crystal', 0.5),                   # midnight flare (the music's bells carry 288)
    (296, 'shimmer', 0.55),
    (342, 'sparkle', 0.35),                  # the light climbs with the camera
    (360, 'wind', 0.3, {'variant': 0}),
    (396, 'star', 0.35),                     # shooting star over the castle
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(CUT_B, 'zoom', {'cx': 172, 'cy': 127}), (CUT_C, 'dissolve', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
