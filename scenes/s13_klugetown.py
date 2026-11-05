"""13 · klugetown — Klugetown, the desert trade town beyond Equestria.
Scene frames (the 2-bar map flight is added outside by engine/regionfx.long_flight; film = scene + 72
for scene >= 72). N = 360 scene frames, HUD clock 17:00 -> 17:45. D harmonic minor ragtag bazaar.

Beat sheet (scene frames; 18 f per beat, 72 per bar). Cameras use shotfx.cruise (slow, even).
  0-71     World-map travel shot (`travel_frame`). Iris closes on dest_screen (204, 97).
  72-179   SHOT A "Bone Dry Desert" (108 f): hazy amber late afternoon, soft sun disc low left, dusty
           rays, stratus drifting, heat shimmer on the far mesas + dunes. Rippled dunes, bleached rock
           arches, a cobbled road to KLUGETOWN on three banded rock pillars: stacked rust/brown shacks,
           scrap patches, tarp + tin roofs, rope bridges, string lights, a crane with a swinging hook,
           windmills, smokestacks, washing lines, a cargo lift, a tan airship moored on top.
           Iris opens 72-79 on the town (204, 97). Pan right 64 px. Balloon 's' between the pillars.
           Callouts KLUGETOWN 90-162, BONE DRY DESERT 108-172. Card 100-208. Tumbleweed 108-175.
  180      zoom cut into the town (shotfx, centred on the main pillar).
  180-269  SHOT B "The Bazaar" (90 f): empty street-level alley, striped awnings, lanterns, spice sacks,
           carpets, junk pile with a cracked bell, brazier + kettle smoke, a creaky sign swinging,
           bunting, light shafts; an arch shows a far pillar and the balloon. Callout THE BAZAAR 180-256.
  270      glide cut.
  270-359  SHOT C "Airship Docks" (90 f): piers jutting from a pillar, the patched balloon-hulled
           airship moored for a beat, then AIRSHIP 288 (bar line): lines drop, steam burst, propellers
           spin up, it drifts right over the brown-grey Sea of Clouds with black peaks; the explorer's
           balloon 'm' lifts off the pier at 298 and follows. Callouts AIRSHIP DOCKS 270-342,
           SEA OF CLOUDS 288-346. Toast from 302 (top centre clear). Keeps moving to the end.
Ambient: sun haze, stratus drift, heat shimmer, windmills, smoke, crane hook, washing, sand wisps,
tumbleweed, balloon | lamps flicker, brazier smoke, sign swing, awnings/carpets/bunting flutter,
dust motes, kettle steam | propellers, cloud-sea roll, pier lamps, ropes, steam, banners.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame
from engine.shotfx import cruise

ID = 'klugetown'
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


def cam_ease(f, f0, f1, a, b):
    return int(round(lerp(a, b, ease_in_out(seg(f, f0, f1)))))


def flash(c, f, f0, color='white'):
    if f == f0:
        fx.fade(c, color, 0.55)
    elif f == f0 + 1:
        fx.fade(c, color, 0.25)


# ================================================================== art helpers
def shift(m, dx, dy):
    """out[y, x] = m[y - dy, x - dx] (False outside)."""
    h, w = m.shape
    out = np.zeros_like(m)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def run_dist(m, dx, dy, K):
    """Run length (1..K) from each mask pixel back to the edge, looking toward (-dx, -dy)."""
    d = np.zeros(m.shape, np.int16)
    cur = m.copy()
    for _ in range(K):
        d += cur
        cur = cur & shift(cur, dx, dy)
        if not cur.any():
            break
    return d


def outline_of(m):
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~inner


def quant(t, n, x0=0, y0=0, dith=0.8):
    thr = bayer(t.shape[0], t.shape[1], x0, y0)
    return np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)


def ramp_arr(ramp):
    return np.array([col(r) for r in ramp], np.uint8)


def smooth_noise(n, cell, seed):
    """1-D smooth value noise of length n in -1..1."""
    R = rng(seed)
    g = R.random(int(n / cell) + 3) * 2 - 1
    xs = np.arange(n) / cell
    i = xs.astype(int)
    t = xs - i
    t = t * t * (3 - 2 * t)
    return g[i] + (g[i + 1] - g[i]) * t


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


def shade_solid(cv, m, ramp, outline=None, strata=None, lit=0.32, dark=0.5, dith=0.9, top_hi=True,
                seed=0, cracks=0):
    """Paint a solid mass lit from the LEFT: ramp dark -> light; strata=(period, depth) bands."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mm = m[y0:y1, x0:x1]
    dl = run_dist(mm, 1, 0, 96).astype(np.float32)
    dr = run_dist(mm, -1, 0, 96).astype(np.float32)
    du = run_dist(mm, 0, 1, 3)
    u = (dl - 0.5) / np.maximum(1, dl + dr - 1)
    t = np.clip(1.0 - (u - lit) / max(0.05, dark), 0, 1) * 0.72 + 0.14
    t = np.where(u < lit, 0.78 + 0.2 * (1 - u / max(lit, 1e-3)), t)
    if strata:
        per, amp = strata
        yy = np.arange(y0, y1)[:, None] + np.round(1.5 * smooth_noise(x1 - x0 + 2, 9, seed)[None, :x1 - x0])
        band = ((yy % per) == 0)
        band2 = ((yy % per) == 1)
        brk = smooth_noise(x1 - x0 + 2, 4, seed + 7)[None, :x1 - x0] > -0.55
        t = np.where(band & brk, t - amp, t)
        t = np.where(band2 & brk, t + amp * 0.6, t)
    n = len(ramp)
    idx = quant(np.clip(t, 0, 0.999), n, x0, y0, dith)
    if top_hi:
        idx = np.where((du == 1) & (u < 0.62), n - 1, idx)
        idx = np.where((du == 1) & (u >= 0.62), np.maximum(idx, n - 3), idx)
    idx = np.where((dl == 1) & (u < 0.5), np.minimum(n - 1, idx + 1), idx)
    if cracks:
        R = rng(seed + 99)
        for _ in range(cracks):
            cy = R.integers(0, y1 - y0)
            cx = R.integers(0, x1 - x0)
            L = R.integers(2, 6)
            for k in range(L):
                yy_, xx_ = cy + k, cx + (k // 2) * (1 if R.random() < 0.5 else 0)
                if 0 <= yy_ < y1 - y0 and 0 <= xx_ < x1 - x0 and mm[yy_, xx_]:
                    idx[yy_, xx_] = max(0, idx[yy_, xx_] - 2)
    out = ramp_arr(ramp)[idx]
    if outline is not None:
        out[outline_of(mm)] = C(outline)
    reg = cv.a[y0:y1, x0:x1]
    reg[mm] = out[mm]


def dune(cv, xa, ya, wl, wr, cols, seed=0, ripple=True, slant=0.55, y_end=None, hi_line=True):
    """A side-view dune: gentle lit windward face (left), steep shadowed slip face (right).
    cols = (shadow_dk, shadow, ripple, lit, lit_hi, crest)."""
    h, w = cv.h, cv.w
    y_end = h if y_end is None else y_end
    base = ya + (wl + wr) * 0.22
    xs = np.arange(w) + 0.5
    top = np.full(w, 1e9)
    L = (xs >= xa - wl) & (xs <= xa)
    u = (xa - xs[L]) / wl
    top[L] = ya + (base - ya) * (u * u * (3 - 2 * u)) * 1.0
    Rr = (xs > xa) & (xs <= xa + wr)
    v = (xs[Rr] - xa) / wr
    top[Rr] = ya + (base - ya) * (1 - (1 - v) ** 2.2)
    top = np.round(top)
    yy = np.arange(h)[:, None]
    m = (yy >= top[None, :]) & (yy < y_end)
    if not m.any():
        return
    depth = yy - top[None, :]
    xb = xa + np.maximum(0, yy - ya) * slant + 1.2 * np.sin((yy - ya) / 4.0 + seed)
    shadow = xs[None, :] > xb
    ca = ramp_arr(cols)
    idx = np.full(m.shape, 3)
    idx = np.where(depth > 9, 2 + (bayer(h, w) < 0.5), idx)
    idx = np.where(depth > 16, 2, idx)
    if ripple:
        wob = np.round(1.2 * np.sin(xs[None, :] * 0.45 + yy * 0.2 + seed))
        rip = ((depth + wob) % 4 == 2) & (depth > 2)
        brk = np.sin(xs[None, :] * 0.23 + yy * 0.9 + seed * 3) > -0.4
        idx = np.where(rip & brk & (depth < 26), 2, idx)
    if hi_line:
        idx = np.where(depth < 1, 5, idx)
        idx = np.where((depth >= 1) & (depth < 3), 4, idx)
    sidx = np.where(depth > 12, 0, 1)
    if ripple:
        sidx = np.where(rip & brk & (depth < 20), 0, sidx)
    edge = shadow & ~shift(shadow, 1, 0)
    idx = np.where(shadow, sidx, idx)
    idx = np.where(edge & (depth < 14), 0, idx)
    out = ca[idx]
    np.copyto(cv.a, out, where=m[:, :, None])


def haze_layer(cv, color, k, y0=0, y1=None):
    """Mix a cached layer (non-key pixels) toward a haze colour by k (rows y0..y1)."""
    y1 = cv.h if y1 is None else y1
    a = cv.a[y0:y1]
    m = ~np.all(a == np.array(KEY, np.uint8), axis=2)
    if isinstance(k, np.ndarray):
        kk = k[y0:y1, None, None]
    else:
        kk = k
    mixd = a.astype(np.float32) + (np.array(col(color), np.float32) - a) * kk
    np.copyto(a, np.clip(mixd + 0.5, 0, 255).astype(np.uint8), where=m[:, :, None])


# ------------------------------------------------------------------ shacks (shared by shots)
WALLS = {
    'rust': '#c27a3a', 'brown': '#8a4e2a', 'clay': '#a8653a', 'tan': '#b8875a', 'red': '#a84a3a',
    'ochre': '#c99a4a', 'grey': '#8f7a68', 'umber': '#9a5a3a', 'teal': '#5d7f78',
}
ROOFS = {
    'tin': '#8e8a86', 'tin2': '#76736f', 'rusty': '#9e5539', 'teal': '#4e8c84', 'mustard': '#d4a640',
    'red': '#b8483a', 'copper': '#b8693e', 'wood': '#7a5236', 'plum': '#7e4a5e',
}
WIN_DK = '#3a2531'
WIN_LIT = '#ffc861'
WIN_HI = '#fff0a8'
STEEL = '#8b8d92'


def draw_shack(cv, x, yb, w, h, wall, roof='flat', rc='wood', seed=0, door=False, win=1, lit=(), anim=None,
               s=1):
    """A ramshackle shack: (x, yb) bottom-left, w x h wall. s = detail scale (1 far, 2 near)."""
    R = rng(seed + 17)
    y = yb - h
    wl = col(WALLS.get(wall, wall))
    md, sh, dk = shade(wl, 0.88), shade(wl, 0.76), shade(wl, 0.6)
    hi = mix(wl, '#ffe2a8', 0.32)
    ol = shade(wl, 0.42)
    cv.rect(x, y, w, h, wl)
    if R.random() < 0.55:
        for px in range(x + 2, x + w - 2, 2 + s):
            for py in range(y + 1, yb - 1):
                if hash01(seed, px, py // 3) > 0.12:
                    cv.px(px, py, md)
    else:
        for py in range(y + 2, yb - 1, 1 + s):
            cv.hline(x + 1, x + w - 2, py, md)
    cv.vline(x, y, yb - 1, hi)
    cv.rect(x + w - 1 - (s - 1), y, s, h, dk)
    cv.rect(x + w - 2 - (s - 1), y, 1, h, sh)
    cv.hline(x, x + w - 1, yb - 1, dk)
    cv.hline(x + 1, x + w - 1, y, sh)
    # scrap patches
    for k in range(int(R.integers(0, 3)) if w > 7 else 0):
        pw, ph = int(R.integers(3, 5)) * s, 3 * s
        pxx = x + 1 + int(R.integers(0, max(1, w - pw - 2)))
        pyy = y + 2 + int(R.integers(0, max(1, h - ph - 2)))
        cv.rect(pxx, pyy, pw, ph, STEEL)
        cv.hline(pxx, pxx + pw - 1, pyy + ph - 1, '#5d5f66')
        cv.px(pxx, pyy, '#c9ccd0')
        cv.px(pxx + pw - 1, pyy, '#c9ccd0')
        cv.vline(pxx + 1, pyy + ph, min(yb - 2, pyy + ph + 1 + s), 'rust')
    # windows
    ww, wh = (3 * s, 3 * s) if w >= 7 else (2, 3)
    nwin = win
    for i in range(nwin):
        if nwin == 1:
            wx = x + (w - ww) // 2 + (int(R.integers(-1, 2)) if w > 9 else 0)
        else:
            wx = x + 2 + i * (w - 4 - ww) // max(1, nwin - 1)
        wy = y + 2 + s
        if door and abs((wx + ww // 2) - (x + w // 2)) < 3 and h < 12 * s:
            continue
        if wy + wh >= yb - 1:
            continue
        cv.rect(wx - 1, wy - 1, ww + 2, wh + 2, ol)
        is_lit = i in lit
        cv.rect(wx, wy, ww, wh, WIN_LIT if is_lit else WIN_DK)
        if is_lit:
            cv.px(wx, wy, WIN_HI)
            if anim is not None:
                anim.setdefault('windows', []).append((wx, wy, ww, wh, seed + i))
        else:
            cv.px(wx, wy, '#5a4050')
        cv.hline(wx - 1, wx + ww, wy + wh + 1, hi)
    if door:
        dw, dh = 3 * s, 5 * s
        dx = x + w // 2 - dw // 2 + (1 if win == 1 else 0)
        cv.rect(dx - 1, yb - dh - 1, dw + 2, dh + 1, ol)
        cv.rect(dx, yb - dh, dw, dh, '#4a2c26')
        cv.px(dx, yb - dh, ol)
        cv.px(dx + dw - 1, yb - dh, ol)
        cv.vline(dx + dw - 2, yb - dh + 1, yb - 1, '#5e3a30')
        cv.px(dx + dw - 2, yb - dh // 2, 'gold')
    # roof
    r = col(ROOFS.get(rc, rc))
    rhi, rsh, rdk = mix(r, '#fff0c8', 0.3), shade(r, 0.8), shade(r, 0.6)
    if roof == 'flat':
        cv.rect(x - 1, y - 2, w + 2, 2, r)
        cv.hline(x - 1, x + w, y - 2, rhi)
        cv.hline(x - 1, x + w, y - 1, rdk)
        for k in range(int(R.integers(0, 3))):
            bx = x + 1 + int(R.integers(0, max(1, w - 4)))
            if R.random() < 0.5:
                cv.rect(bx, y - 5, 3, 3, '#a9773f')
                cv.px(bx, y - 5, '#d1a060')
                cv.vline(bx + 2, y - 5, y - 3, '#6e4628')
            else:
                cv.rect(bx, y - 6, 3, 4, '#7a4a2e')
                cv.hline(bx, bx + 2, y - 5, '#4e2e20')
                cv.px(bx, y - 6, '#a86a40')
    elif roof == 'tin':
        hr = 2 + w // 6
        rise = 1 if R.random() < 0.5 else -1
        for i in range(-2, w + 2):
            k = (i + 2) / (w + 3)
            if rise < 0:
                k = 1 - k
            yy = y - 1 - int(round(hr * k))
            c_ = r if (i % 2 == 0) else rsh
            if hash01(seed, i, 5) < 0.18:
                c_ = '#9e5539'
            cv.vline(x + i, yy, y - 1, c_)
            cv.px(x + i, yy, rhi)
        cv.hline(x - 2, x + w + 1, y - 1, rdk)
    elif roof == 'gable':
        hr = max(3, (w + 4) // 3)
        cxr = x + w / 2 - 0.5
        for j in range(hr):
            half = (w / 2 + 2) * (j + 1) / hr
            a0, a1 = int(round(cxr - half)), int(round(cxr + half))
            yy = y - hr + j
            cv.hline(a0, int(cxr), yy, r if j % 2 else rhi if j == 0 else r)
            cv.hline(int(cxr) + 1, a1, yy, rsh if j % 2 else rdk)
            cv.px(a0, yy, rdk)
            cv.px(a1, yy, rdk)
        cv.hline(x - 2, x + w + 1, y - 1, rdk)
    elif roof == 'tarp':
        stripe = '#efe2c4'
        for i in range(-1, w + 1):
            sag = int(round(1.2 * math.sin(math.pi * (i + 1) / (w + 1))))
            yy = y - 3 + sag
            c_ = r if ((i + 1) // (1 + s)) % 2 == 0 else stripe
            if i > w * 0.6:
                c_ = shade(col(c_), 0.82)
            cv.vline(x + i, yy, y, c_)
            if i % 2 == 0:
                cv.px(x + i, y + 1, c_)
        cv.hline(x - 1, x + w, y - 3, rhi)
    elif roof == 'dome':
        rx = w // 2 + 1
        ry = max(3, int(w * 0.55))
        cx_, cy_ = x + w // 2, y
        for j in range(ry):
            half = int(round(rx * math.sqrt(max(0.0, 1 - ((ry - j) / ry) ** 2))))
            yy = cy_ - ry + j
            for i in range(-half, half + 1):
                u = (i + half) / max(1, 2 * half)
                c_ = rhi if u < 0.28 else r if u < 0.62 else rsh if u < 0.85 else rdk
                cv.px(cx_ + i, yy, c_)
        cv.vline(cx_, cy_ - ry - 3, cy_ - ry, '#5a3a2a')
        cv.px(cx_, cy_ - ry - 4, 'gold')
        cv.hline(x - 1, x + w, y, rdk)
        if anim is not None:
            anim.setdefault('pennants', []).append((cx_, cy_ - ry - 4))
    return (x, y, w, h)


ENV_RAMP = ['#6e4e3a', '#8c6a4c', '#a8865e', '#c4a274', '#d9bc8e', '#ead2a8', '#f6e6c4']


@lru_cache(maxsize=None)
def _ship(kind):
    """The balloon-hulled airship facing RIGHT. kind 0 = tiny (shot A), 1 = big (shot C).
    Returns (rgba, anchors) where anchors are sprite-relative points."""
    s = 0.34 if kind == 0 else 1.0
    Wd, Hd = int(124 * s) + 4, int(64 * s) + 4

    def P(x, y):
        return int(round(x * s)), int(round(y * s))

    cv = new(Wd, Hd)
    ex, ey, rx, ry = 62 * s, 18 * s + 1, 46 * s, 15 * s
    # tail fins (behind the envelope)
    fin = '#9e4a3a'
    for pts in ([(20, 12), (6, 1), (2, 3), (12, 17)], [(20, 24), (6, 35), (2, 33), (12, 19)]):
        cv.poly([P(*p) for p in pts], fin)
        if kind:
            cv.line(*P(*pts[1]), *P(*pts[0]), '#c86a4a')
    if kind:
        cv.poly([P(6, 16), P(-0.5, 14), P(-0.5, 22), P(6, 20)], '#7a3a2e')
    # envelope (ellipsoid shading, light from upper-left)
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    u = (xx + 0.5 - ex) / rx
    v = (yy + 0.5 - ey) / ry
    d2 = u * u + v * v
    env = d2 <= 1.0
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array([-0.55, -0.65, 0.52])
    L /= np.linalg.norm(L)
    lam = u * L[0] + v * L[1] + nz * L[2]
    t = np.clip((lam + 0.35) / 1.25, 0, 0.999)
    n = len(ENV_RAMP)
    idx = quant(t, n, 0, 0, 0.9)
    out = ramp_arr(ENV_RAMP)[idx]
    if kind:
        # gore seams + girth seam
        for k in (-3, -2, -1, 0, 1, 2, 3):
            xs_ = ex + k * 13 * np.sqrt(np.clip(1 - v * v, 0, 1))
            seam = env & (np.abs(xx + 0.5 - xs_) < 0.5)
            out[seam] = (out[seam].astype(np.float32) * 0.86).astype(np.uint8)
        girth = env & (np.abs(yy + 0.5 - (ey + 5)) < 0.5)
        out[girth] = (out[girth].astype(np.float32) * 0.86).astype(np.uint8)
        # patches (shaded with the envelope light)
        for (px, py, pw, ph, pc) in ((30, 10, 9, 7, '#b8483a'), (78, 20, 8, 6, '#4e8c84'),
                                     (52, 24, 7, 5, '#d4a640'), (92, 9, 6, 6, '#8b8d92'),
                                     (40, 21, 6, 4, '#7e4a5e')):
            pm = env & (xx >= px) & (xx < px + pw) & (yy >= py) & (yy < py + ph)
            k = (0.55 + 0.6 * t[pm])[:, None]
            out[pm] = np.clip(np.array(col(pc), np.float32) * k, 0, 255).astype(np.uint8)
            edge = pm & ~(shift(pm, 1, 0) & shift(pm, -1, 0) & shift(pm, 0, 1) & shift(pm, 0, -1))
            st = edge & (((xx + yy) % 2) == 0)
            out[st] = C('#4a3028')
    else:
        pm = env & (xx >= 9) & (xx < 12) & (yy >= 3) & (yy < 5)
        out[pm] = C('#b8483a')
        pm = env & (xx >= 25) & (xx < 27) & (yy >= 6) & (yy < 8)
        out[pm] = C('#4e8c84')
    out[outline_of(env)] = C('#4e3428')
    np.copyto(cv.a, out, where=env[:, :, None])
    if kind:
        # gold star emblem
        cxs, cys = 70, 17
        for k in range(5):
            a_ = -math.pi / 2 + k * 2 * math.pi / 5
            cv.line(cxs, cys, cxs + int(round(math.cos(a_) * 4)), cys + int(round(math.sin(a_) * 4)), '#f9c22b')
        cv.circle(cxs, cys, 1, '#f9c22b')
        cv.px(cxs - 1, cys - 1, '#fff0a8')
    # rigging
    for (x0, x1) in ((30, 40), (48, 54), (76, 76), (94, 90)):
        yb_ = ey + ry * math.sqrt(max(0.0, 1 - ((x0 * s - ex) / rx) ** 2))
        cv.line(int(round(x0 * s)), int(round(yb_)) - 1, *P(x1, 41), '#5a4034')
    # hull
    hull = [(34, 43), (98, 43), (106, 39), (103, 47), (93, 54), (45, 54), (36, 50)]
    hm = poly_mask(Hd, Wd, [(x * s, y * s) for x, y in hull])
    hr = ['#4e3020', '#6e4628', '#8a5a36', '#a06a40', '#c48a52']
    hv = np.clip(1.0 - (yy - 43 * s) / (11 * s), 0, 1)
    hidx = quant(hv * 0.85 + 0.1, len(hr), 0, 0, 0.5)
    ho = ramp_arr(hr)[hidx]
    if kind:
        plank = hm & ((np.round(yy - 43 * s) % 3) == 2)
        ho[plank] = C('#5e3a24')
    ho[outline_of(hm)] = C('#3e2418')
    np.copyto(cv.a, ho, where=hm[:, :, None])
    anchors = {}
    if kind:
        cv.hline(*P(35, 43), int(98 * s), '#c48a52')
        for x in range(36, 98, 4):
            cv.vline(x, 39, 42, '#6e4628')
        cv.hline(36, 97, 39, '#8a5a36')
        for x in (46, 56, 66, 76, 86):
            cv.circle(x, 48, 1, '#c9a24a')
            cv.px(x, 48, '#ffd27a' if x in (56, 76) else '#3a2531')
        cv.hline(46, 92, 51, '#5e3a24')
        # cabin
        cv.rect(58, 33, 18, 7, '#b8875a')
        cv.rect(57, 32, 20, 2, '#9e4a3a')
        cv.hline(57, 76, 31, '#c86a4a')
        cv.rect(61, 35, 3, 3, '#ffc861')
        cv.rect(67, 35, 3, 3, '#ffc861')
        cv.px(61, 35, '#fff0a8')
        cv.vline(75, 33, 39, '#6e4628')
        # stack (stern)
        cv.rect(39, 34, 2, 6, '#5c5656')
        cv.hline(38, 41, 34, '#3e3a3a')
        anchors['stack'] = (39, 33)
        # bow star + rudder
        cv.px(104, 40, '#f9c22b')
        cv.poly([(34, 44), (29, 46), (29, 53), (35, 51)], '#7a3a2e')
        cv.vline(29, 46, 53, '#4e2418')
        # engine pods hanging from the envelope
        for (pxp, pyp) in ((44, 36), (84, 36)):
            cv.line(pxp + 2, pyp - 4, pxp + 2, pyp, '#4a3a34')
            cv.rect(pxp, pyp, 7, 3, '#7a7672')
            cv.hline(pxp, pxp + 6, pyp, '#b0aca6')
            cv.hline(pxp, pxp + 6, pyp + 2, '#4e4a48')
        cv.hline(23, 28, 49, '#5c5656')
        anchors['props'] = [(22, 49, 6), (43, 37, 3), (83, 37, 3)]
        anchors['bow'] = (102, 44)
        anchors['stern'] = (36, 45)
        anchors['flag'] = (62, 3)
        anchors['deck'] = 43
    a, m = to_layer(cv)
    rgba = np.zeros((Hd, Wd, 4), np.uint8)
    rgba[:, :, :3] = a
    rgba[:, :, 3] = m * 255
    return rgba, anchors


def draw_prop(c, x, y, r, f, spin=1.0):
    """Side-view propeller (axis horizontal) at (x, y): a blade line that flickers in length."""
    if spin <= 0.01:
        c.vline(x, y - r, y + r, '#5a4a40')
        c.px(x, y, '#c9a24a')
        return
    ph = int(f * spin * 2.2) % 4
    L = [r, r // 2, 0, r // 2][ph] if spin < 0.6 else [r, 1, r, 1][ph]
    if spin >= 0.3:
        for yy in range(y - r, y + r + 1):
            if (yy + f) % 2 == 0:
                c.px(x, yy, '#d8c8b4')
                if abs(yy - y) < r - 1:
                    c.px(x - 1, yy, '#c0b0a0')
    c.vline(x, y - L, y + L, '#4a3a34')
    c.px(x, y - L, '#c9a24a')
    c.px(x, y + L, '#c9a24a')
    c.px(x + 1, y, '#5c5656')


# ================================================================== SHOT A: Bone Dry Desert + town
A0, A1 = 72, 180
A_CAM = 64
PAR_A = dict(mesa=0.1, far=0.22, town=0.42, mid=0.62, near=1.0, fg=1.35)
HAZE = '#f3cd95'
SUN_A = (62, 58)
SKY_A = ['#ad6c48', '#bb784c', '#c98552', '#d6955a', '#e1a663', '#e9b770', '#efc77f', '#f3d390',
         '#f6dda0', '#f8e4b0']


def cam_a(f):
    t = seg(f, A0, A1 - 1)
    return A_CAM * cruise(t)


def lw(par):
    return W + int(math.ceil(A_CAM * par)) + 4


@lru_cache(maxsize=1)
def _sky_a():
    c = Canvas(W, H, SKY_A[-1])
    c.gradient(0, 0, W, 120, SKY_A, soft=0.7)
    # sun glow: stepped rings mixing toward pale gold
    yy, xx = np.mgrid[0:H, 0:W]
    sx, sy = SUN_A
    d = np.sqrt((xx - sx) ** 2 + ((yy - sy) * 1.15) ** 2)
    lvl = np.clip(1 - d / 70.0, 0, 1) ** 1.6 * 5
    thr = bayer(H, W)
    q = np.clip(np.floor(lvl + (thr - 0.5) * 0.9), 0, 5) / 5.0 * 0.6
    a = c.a.astype(np.float32)
    a += (np.array(col('#fff2c6'), np.float32) - a) * q[:, :, None]
    c.a[:] = np.clip(a + 0.5, 0, 255).astype(np.uint8)
    # soft disc
    c.circle(sx, sy, 9, '#fbe6b0')
    c.circle(sx, sy, 8, '#fdeec4')
    c.circle(sx, sy, 7, '#fff6da')
    c.circle(sx - 1, sy - 1, 5, '#fffbec')
    for i in range(-8, 9, 2):
        c.px(sx + i, sy + 9 if abs(i) < 5 else sy + 8, '#fdeec4')
    return c


@lru_cache(maxsize=1)
def _stratus_a():
    """Long dusty cloud streaks (480 wide, wraps)."""
    cv = new(480, 70)
    R = rng(51)
    for i in range(9):
        cx = int(R.integers(0, 480))
        cy = int(R.integers(8, 62))
        rx = int(R.integers(22, 56))
        ry = int(R.integers(1, 3))
        k = cy / 70.0
        body = mix('#c98a58', '#f6dcaa', 0.25 + 0.6 * k)
        top = mix(body, '#fff1cc', 0.45)
        und = shade(body, 0.9)
        for dx in (-480, 0, 480):
            cv.ellipse(cx + dx, cy + 1, rx, ry, und)
            cv.ellipse(cx + dx, cy, rx, ry, body)
            cv.hline(cx + dx - rx // 2, cx + dx + rx // 3, cy - ry, top)
            cv.ellipse(cx + dx + rx // 3, cy - 1, rx // 3, max(1, ry - 1), top)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _mesa_a():
    w = lw(PAR_A['mesa'])
    cv = new(w)
    ramp = ['#d7a77d', '#ddb084', '#e4ba8c', '#eac495', '#efcfa2', '#f4dab1']
    shapes = [
        [(0, 112), (0, 98), (6, 96), (30, 95), (34, 99), (40, 112)],
        [(52, 112), (58, 101), (62, 90), (76, 89), (79, 94), (84, 104), (92, 112)],
        [(150, 112), (156, 104), (160, 98), (196, 97), (200, 101), (204, 108), (212, 112)],
        [(232, 112), (238, 100), (241, 93), (247, 92), (250, 99), (256, 112)],
        [(270, 112), (276, 103), (280, 96), (318, 95), (322, 100), (330, 112)],
    ]
    for i, pts in enumerate(shapes):
        m = poly_mask(H, w, pts)
        shade_solid(cv, m, ramp, outline=None, strata=(4, 0.1), lit=0.25, dark=0.6, seed=i)
    return to_layer(cv)


@lru_cache(maxsize=1)
def _far_a():
    w = lw(PAR_A['far'])
    cv = new(w)
    cols = ('#dcab78', '#e3b783', '#e8c08c', '#f0cf9c', '#f5dcae', '#f9e8c4')
    specs = [(20, 106, 50, 26), (95, 104, 60, 30), (170, 107, 50, 24), (238, 103, 64, 28),
             (300, 106, 40, 22), (60, 110, 40, 18), (140, 111, 44, 18), (210, 112, 46, 20), (280, 111, 40, 18)]
    for i, (xa, ya, wl_, wr) in enumerate(specs):
        dune(cv, xa, ya, wl_, wr, cols, seed=i, ripple=True, slant=0.5)
    haze_layer(cv, HAZE, 0.25)
    return to_layer(cv)


# town layer (pillars + stacked shacks); coords are town-layer coords (= screen at cam 0)
PILLARS = [  # cx, top, cap hw, neck hw, base hw, seed
    (136, 88, 16, 11, 19, 1),
    (206, 78, 19, 13, 23, 2),
    (276, 92, 14, 10, 17, 3),
]
DECKS = [(116, 157, 88), (183, 231, 78), (258, 296, 92)]
SHACKS = [
    # pillar 1
    (117, 88, 13, 10, 'rust', 'flat', 'wood', False, 1, (0,)),
    (130, 88, 12, 13, 'brown', 'flat', 'wood', True, 1, ()),
    (142, 88, 14, 9, 'ochre', 'tin', 'tin', False, 2, (1,)),
    (119, 78, 11, 9, 'red', 'tin', 'rusty', False, 1, ()),
    (131, 75, 13, 11, 'tan', 'flat', 'wood', False, 1, (0,)),
    (145, 79, 9, 8, 'umber', 'tarp', 'teal', False, 1, ()),
    (133, 64, 9, 8, 'rust', 'dome', 'copper', False, 1, ()),
    # pillar 2
    (183, 78, 14, 11, 'brown', 'flat', 'wood', False, 1, (0,)),
    (197, 78, 17, 14, 'rust', 'flat', 'wood', True, 2, (1,)),
    (214, 78, 16, 10, 'tan', 'flat', 'wood', False, 2, ()),
    (185, 67, 12, 10, 'ochre', 'tin', 'tin2', False, 1, ()),
    (198, 64, 14, 12, 'red', 'flat', 'wood', False, 2, (0,)),
    (213, 68, 13, 9, 'grey', 'tarp', 'mustard', False, 1, (0,)),
    (200, 52, 11, 12, 'clay', 'flat', 'wood', False, 1, (0,)),
    (212, 56, 8, 8, 'brown', 'gable', 'red', False, 1, ()),
    (201, 40, 8, 9, 'tan', 'dome', 'copper', False, 1, ()),
    (229, 74, 9, 8, 'red', 'tarp', 'teal', False, 1, ()),
    # pillar 3
    (259, 92, 12, 10, 'ochre', 'flat', 'wood', False, 1, ()),
    (271, 92, 14, 12, 'rust', 'flat', 'wood', True, 1, ()),
    (285, 92, 10, 8, 'brown', 'tin', 'tin', False, 1, (0,)),
    (261, 82, 11, 9, 'brown', 'gable', 'plum', False, 1, ()),
    (273, 80, 12, 10, 'red', 'tin', 'tin2', False, 1, (0,)),
    # ground warehouses
    (160, 124, 15, 9, 'clay', 'tin', 'tin', True, 1, ()),
    (232, 124, 13, 8, 'umber', 'tarp', 'red', True, 1, ()),
]
BRIDGES = [((157, 89), (183, 79), 4), ((231, 79), (258, 93), 5), ((144, 72), (185, 58), 3)]
STACKS_A = [(150, 80, 66), (227, 69, 50), (266, 83, 72)]
WINDMILLS_A = [(124, 55, 6, 69), (290, 69, 5, 80)]
CRANE_A = (300, 84, 62, 310)          # mast x, base y, top y, jib end x
LIFT_A = (186, 79, 126)
WASH_A = [((171, 102), (187, 100), 4), ((213, 48), (226, 52), 3)]
MAST_A = (219, 30, 56)                # mooring mast x, top y, base y


def pillar_mask(h, w, cx, top, cap, neck, base, seed, bottom=140):
    m = np.zeros((h, w), bool)
    nl = smooth_noise(h, 5, seed * 13)
    nr = smooth_noise(h, 6, seed * 13 + 5)
    for y in range(top, min(h, bottom)):
        t = (y - top) / (bottom - top)
        if t < 0.22:
            hw = lerp(cap, neck, smooth(t / 0.22))
        else:
            hw = lerp(neck, base, ((t - 0.22) / 0.78) ** 1.4)
        if y - top < 2:
            hw -= 2 - (y - top)
        l = int(round(cx - hw + 1.6 * nl[y]))
        r = int(round(cx + hw + 1.6 * nr[y]))
        m[y, max(0, l):min(w, r)] = True
    return m


@lru_cache(maxsize=1)
def _town_a():
    w = lw(PAR_A['town'])
    cv = new(w)
    anim = {}
    rock = ['#5e3a2c', '#7a4a32', '#93593a', '#ab6a42', '#c47f50', '#d99a66']
    for (cx, top, cap, neck, base, sd) in PILLARS:
        m = pillar_mask(H, w, cx, top, cap, neck, base, sd)
        shade_solid(cv, m, rock, outline='#4e2e24', strata=(6, 0.16), lit=0.3, dark=0.55, seed=sd, cracks=10)
    # pipes down the pillars
    for (px, y0, y1) in ((220, 80, 124), (148, 90, 126)):
        cv.vline(px, y0, y1, '#6f6a6a')
        cv.vline(px + 1, y0, y1, '#4c4848')
        for yy in range(y0 + 4, y1, 9):
            cv.hline(px - 1, px + 2, yy, '#9a5a3a')
    # decks with braces
    for (x0, x1, y) in DECKS:
        cv.rect(x0, y, x1 - x0, 2, '#7a5236')
        cv.hline(x0, x1 - 1, y, '#b07a48')
        for xx in range(x0 + 2, x1, 4):
            cv.px(xx, y + 1, '#4e3020')
        cv.line(x0 + 1, y + 2, x0 + 9, y + 10, '#5a3a26')
        cv.line(x1 - 2, y + 2, x1 - 10, y + 10, '#5a3a26')
        cv.vline(x0 + 1, y + 2, y + 4, '#5a3a26')
        cv.vline(x1 - 2, y + 2, y + 4, '#5a3a26')
    # ground strip at the base + road to the gate
    gcols = ('#d9a46e', '#e2b27a', '#ebc189', '#f2cf98', '#f7dcaa', '#fae7bf')
    dune(cv, 150, 117, 90, 40, gcols, seed=3, slant=0.6)
    dune(cv, 262, 119, 60, 60, gcols, seed=4, slant=0.6)
    dune(cv, 96, 121, 60, 50, gcols, seed=5, slant=0.6)
    for i, y in enumerate(range(121, 140)):
        cxr = 205 - (y - 121) * 0.9 + 3 * math.sin((y - 121) / 5.0)
        hw = 1.5 + (y - 121) * 0.18
        a0, a1 = int(round(cxr - hw)), int(round(cxr + hw))
        cv.hline(a0, a1, y, '#b49a7c' if y % 2 else '#a48a6c')
        cv.px(a0, y, '#8a6e56')
        cv.px(a1, y, '#cdb596')
    # gate arch at the pillar foot
    cv.rect(199, 109, 13, 13, '#7a4a32')
    cv.rect(201, 112, 9, 10, '#3d2622')
    cv.hline(199, 211, 108, '#c47f50')
    cv.px(201, 112, '#7a4a32')
    cv.px(209, 112, '#7a4a32')
    cv.px(205, 106, 'gold')
    cv.vline(205, 104, 107, '#5a3a2a')
    # shacks
    for i, (x, yb, sw, sh_, wall, roof, rc, door, win, lit) in enumerate(SHACKS):
        draw_shack(cv, x, yb, sw, sh_, wall, roof, rc, seed=40 + i, door=door, win=win, lit=lit, anim=anim)
    # stilts under the hanging shack
    cv.vline(230, 75, 86, '#5a3a26')
    cv.vline(237, 75, 90, '#5a3a26')
    cv.line(230, 86, 226, 90, '#5a3a26')
    # smokestacks
    for (sx, yb, yt) in STACKS_A:
        cv.rect(sx, yt, 2, yb - yt, '#5c5656')
        cv.vline(sx, yt, yb - 1, '#7d7676')
        cv.hline(sx - 1, sx + 2, yt, '#3e3a3a')
        for yy in range(yt + 3, yb, 5):
            cv.hline(sx, sx + 1, yy, '#9e5539')
    # windmill towers
    for (wx, wy, wr, wb) in WINDMILLS_A:
        cv.line(wx - 3, wb, wx, wy + 1, '#5a3a26')
        cv.line(wx + 3, wb, wx, wy + 1, '#5a3a26')
        for yy in range(wy + 4, wb, 4):
            k = (yy - wy) / max(1, wb - wy)
            cv.hline(int(wx - 3 * k), int(wx + 3 * k), yy, '#6e4a30')
        cv.rect(wx - 1, wy - 1, 3, 3, '#4e3a30')
        cv.line(wx + 1, wy, wx + 5, wy, '#4e3a30')
        cv.rect(wx + 5, wy - 2, 2, 4, '#c27a3a')
    # crane
    mx, mb, mt, je = CRANE_A
    for yy in range(mt, mb):
        cv.px(mx, yy, '#6a4630')
        cv.px(mx + 2, yy, '#6a4630')
        if (yy - mt) % 3 == 0:
            cv.px(mx + 1, yy, '#8a5a36')
    cv.hline(mx - 8, je, mt, '#7a4f30')
    cv.hline(mx - 8, je, mt + 1, '#5a3a24')
    for xx in range(mx - 8, je, 3):
        cv.px(xx, mt - 1, '#7a4f30')
    cv.line(mx + 1, mt - 5, je, mt - 1, '#5a3a24')
    cv.line(mx + 1, mt - 5, mx - 7, mt - 1, '#5a3a24')
    cv.rect(mx - 9, mt + 2, 4, 4, '#4e4a4a')
    cv.px(mx + 1, mt - 6, 'gold')
    # cargo lift frame
    lx, lt, lb = LIFT_A
    cv.hline(lx - 2, lx + 7, lt, '#5a3a26')
    cv.px(lx + 2, lt + 1, '#3e3a3a')
    # mooring mast
    ax, at, ab = MAST_A
    cv.vline(ax, at, ab, '#5a3a26')
    cv.vline(ax + 1, at + 2, ab, '#7a5236')
    for yy in range(at + 3, ab, 4):
        cv.hline(ax - 1, ax + 2, yy, '#4a2e20')
    cv.rect(ax - 1, at - 1, 4, 2, '#9a5a3a')
    cv.px(ax, at - 2, 'gold')
    haze_layer(cv, HAZE, 0.1)
    return to_layer(cv), anim


@lru_cache(maxsize=1)
def _mid_a():
    w = lw(PAR_A['mid'])
    cv = new(w)
    cols = ('#c98d58', '#d69e66', '#e3b37c', '#efc78f', '#f7d9a6', '#fdebc4')
    dune(cv, 150, 126, 120, 50, cols, seed=11, slant=0.55)
    dune(cv, 330, 124, 110, 55, cols, seed=12, slant=0.55)
    dune(cv, 60, 130, 70, 60, cols, seed=13, slant=0.55)
    # bleached arch (left)
    bone = ['#8e7462', '#b39a80', '#cbb596', '#ddcaa9', '#ebdcbe', '#f6ecd6']
    outer = poly_mask(H, w, [(14, 146), (17, 122), (20, 104), (25, 92), (34, 84), (48, 80), (62, 79),
                             (76, 81), (88, 87), (96, 97), (100, 112), (101, 128), (106, 146)])
    yy, xx = np.mgrid[0:H, 0:w]
    hole = ((xx - 60.5) / 25.0) ** 2 + ((yy - 136.0) / 42.0) ** 2 <= 1.0
    hole &= yy > 94
    arch = outer & ~hole
    shade_solid(cv, arch, bone, outline='#7e6452', strata=(5, 0.14), lit=0.3, dark=0.5, seed=21, cracks=12)
    # small rock spire (right)
    spire = poly_mask(H, w, [(300, 140), (303, 118), (306, 108), (310, 104), (315, 106), (317, 116),
                             (320, 140)])
    shade_solid(cv, spire, bone, outline='#7e6452', strata=(4, 0.14), seed=22, cracks=4)
    cap = poly_mask(H, w, [(302, 107), (305, 100), (314, 98), (321, 101), (318, 106), (306, 108)])
    shade_solid(cv, cap, bone, outline='#7e6452', seed=23)
    # front dune line covering arch feet
    dune(cv, 120, 136, 140, 45, cols, seed=14, slant=0.5)
    dune(cv, 250, 134, 80, 60, cols, seed=15, slant=0.5)
    # mid road: from the near crest up toward the town base (only on the front dune surface)
    land = ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    for y in range(128, 150):
        k = (y - 128) / 22.0
        cxr = 196 - 14 * k + 5 * math.sin(k * 3.0)
        hw = 2 + k * 4
        a0, a1 = int(round(cxr - hw)), int(round(cxr + hw))
        for xx in range(a0, a1 + 1):
            if not land[y, xx]:
                continue
            cc = '#7e6452' if xx == a0 else '#d8c0a0' if xx == a1 else (
                '#c2a888' if (xx + y) % 3 == 0 else '#a88c6e' if (xx * 2 + y) % 5 == 0 else '#b89c7c')
            cv.px(xx, y, cc)
    haze_layer(cv, HAZE, 0.05)
    return to_layer(cv)


def cobbles(cv, m, rh_of_y, seed=0, stone=('#9c8266', '#ad9374', '#bba282', '#cbb392', '#dcc6a6'),
            gap='#77604c'):
    """Irregular staggered cobblestones filling mask m; stone row height rh_of_y(y) (perspective)."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return
    xmin, xmax = xs.min(), xs.max()
    y, row = ys.min(), 0
    while y <= ys.max():
        rh = max(1, int(rh_of_y(y)))
        R = rng(seed * 1000 + row)
        x = xmin - int(R.integers(0, 2 * rh + 2))
        while x <= xmax:
            sw = int(R.integers(rh + 1, 2 * rh + 3))
            tone = int(R.integers(0, 3))
            for yy in range(y, min(y + rh, cv.h)):
                for xx in range(max(0, x), min(x + sw, cv.w)):
                    if not m[yy, xx]:
                        continue
                    corner = (xx in (x, x + sw - 1)) and (yy in (y, y + rh - 1)) and rh > 1
                    if corner:
                        cc = gap
                    elif yy == y and xx < x + sw - 1:
                        cc = stone[tone + 2]
                    elif xx == x + sw - 1 or (yy == y + rh - 1 and rh > 1):
                        cc = stone[max(0, tone - 1)] if tone else stone[0]
                    else:
                        cc = stone[tone + 1]
                    cv.a[yy, xx] = C(cc)
            if x + sw < cv.w and x + sw >= 0:
                for yy in range(y, min(y + rh, cv.h)):
                    if m[yy, x + sw]:
                        cv.a[yy, x + sw] = C(gap)
            x += sw + 1
        if y + rh < cv.h:
            gm = m[y + rh]
            cv.a[y + rh][gm] = C(gap)
        y += rh + 1
        row += 1


@lru_cache(maxsize=1)
def _near_a():
    w = lw(PAR_A['near'])
    cv = new(w)
    cols = ('#b97a48', '#c98c56', '#d9a46c', '#e9bc82', '#f3d099', '#fbe4b8')
    dune(cv, 120, 146, 140, 70, cols, seed=31, slant=0.5)
    dune(cv, 300, 143, 120, 60, cols, seed=32, slant=0.5)
    dune(cv, 390, 150, 80, 50, cols, seed=33, slant=0.5)
    # road: centreline from bottom up to the crest
    yy, xx = np.mgrid[0:H, 0:w]
    k = np.clip((yy - 147) / 33.0, 0, 1)
    cxr = 200 + 22 * np.sin(k * 2.6 - 0.5) - 10 * k
    hw = 3 + k * k * 22
    land = ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    road = (np.abs(xx - cxr) <= hw) & (yy >= 147) & land
    cobbles(cv, road, lambda y: 1 + (y - 145) // 10, seed=5)
    edge = outline_of(road)
    cv.a[edge & (xx < cxr)] = C('#8e6c50')
    cv.a[edge & (xx >= cxr) & (yy > 147)] = C('#6e543e')
    # sand drifting over the road edges
    drift = road & (np.abs(xx - cxr) > hw * (0.78 - 0.12 * np.sin(yy * 0.7 + xx * 0.13))) & (bayer(H, w) < 0.5)
    cv.a[drift] = C('#e9bc82')
    # rocks + scrub
    rock = ['#7e6452', '#a28a72', '#bfa98e', '#d6c4a6', '#e8dac0']
    for i, (rx, ry, rw, rh) in enumerate([(60, 160, 14, 8), (262, 156, 10, 6), (300, 170, 18, 10),
                                         (150, 166, 8, 5), (352, 160, 12, 7)]):
        m = poly_mask(H, w, [(rx, ry + rh), (rx + 2, ry + 2), (rx + rw // 3, ry), (rx + rw - 3, ry + 1),
                             (rx + rw, ry + rh)])
        shade_solid(cv, m, rock, outline='#6a5244', seed=60 + i)
    # cactus (two arms) beside the road
    for (cx0, base, hgt) in ((246, 152, 18), (96, 156, 12)):
        g = ['#3f5a3a', '#4f7046', '#628a52', '#7ea864']
        cv.rect(cx0, base - hgt, 4, hgt, g[1])
        cv.vline(cx0, base - hgt + 1, base - 1, g[2])
        cv.vline(cx0 + 3, base - hgt + 1, base - 1, g[0])
        cv.hline(cx0 + 1, cx0 + 2, base - hgt - 1, g[1])
        cv.rect(cx0 - 4, base - hgt + 6, 4, 3, g[1])
        cv.rect(cx0 - 4, base - hgt + 2, 3, 5, g[1])
        cv.vline(cx0 - 4, base - hgt + 2, base - hgt + 8, g[2])
        cv.rect(cx0 + 4, base - hgt + 9, 3, 2, g[1])
        cv.rect(cx0 + 5, base - hgt + 5, 3, 5, g[1])
        cv.vline(cx0 + 7, base - hgt + 5, base - hgt + 9, g[0])
        for yy_ in range(base - hgt + 1, base, 3):
            cv.px(cx0 + 1, yy_, g[3])
        cv.px(cx0 + 1, base - hgt - 2, '#e86a8a')
    # parked cart (no driver) by the road
    cx0, cy0 = 222, 162
    cv.rect(cx0, cy0 - 7, 18, 6, '#8a5a36')
    cv.hline(cx0, cx0 + 17, cy0 - 7, '#b07a48')
    cv.hline(cx0, cx0 + 17, cy0 - 2, '#5a3a24')
    for xx_ in range(cx0 + 3, cx0 + 17, 4):
        cv.vline(xx_, cy0 - 6, cy0 - 3, '#6e4628')
    cv.rect(cx0 + 2, cy0 - 12, 5, 5, '#a9773f')
    cv.px(cx0 + 2, cy0 - 12, '#d1a060')
    cv.rect(cx0 + 8, cy0 - 11, 4, 4, '#c27a3a')
    cv.ellipse(cx0 + 14, cy0 - 9, 3, 2, '#d8c39a')
    cv.line(cx0 - 6, cy0 - 1, cx0, cy0 - 4, '#6e4628')
    for wx in (cx0 + 4, cx0 + 14):
        cv.circle(wx, cy0, 3, '#4e3020', fill=False)
        cv.px(wx, cy0, '#8a5a36')
        cv.px(wx - 2, cy0 - 2, '#8a5a36')
    # signpost with an airship icon
    sx, sy = 168, 150
    cv.vline(sx, sy - 14, sy, '#6e4628')
    cv.rect(sx - 6, sy - 15, 13, 7, '#b07a48')
    cv.frame(sx - 6, sy - 15, 13, 7, '#5a3a24')
    cv.px(sx + 7, sy - 12, '#5a3a24')
    cv.ellipse(sx, sy - 13, 3, 1, '#d8c39a')
    cv.hline(sx - 1, sx + 1, sy - 11, '#5a3a24')
    return to_layer(cv)


@lru_cache(maxsize=1)
def _fg_a():
    w = lw(PAR_A['fg'])
    cv = new(w)
    cols = ('#9e6038', '#b27046', '#c58756', '#d89e68', '#e6b67c', '#f2cc96')
    dune(cv, 40, 168, 70, 60, cols, seed=41, slant=0.4)
    rock = ['#5e3a2c', '#7a4a32', '#93593a', '#ab6a42', '#c47f50', '#d99a66']
    m = poly_mask(H, w, [(336, 180), (340, 160), (348, 150), (362, 146), (378, 150), (390, 158), (398, 180)])
    shade_solid(cv, m, rock, outline='#4e2e24', strata=(4, 0.15), seed=71, cracks=8)
    m = poly_mask(H, w, [(312, 180), (316, 170), (326, 166), (338, 170), (342, 180)])
    shade_solid(cv, m, rock, outline='#4e2e24', strata=(4, 0.15), seed=72, cracks=4)
    return to_layer(cv)


def draw_smoke(c, f, x, y, n=5, period=60, rise=22, drift=16, seed=0, cols=('#8e7e78', '#b0a090', '#cdbca6')):
    """Little smoke puffs from a stack top at (x, y), drifting right with the wind."""
    for q in range(n):
        t = ((f + q * period / n + seed * 7) % period) / period
        r = 1 + int(t * 3.2)
        px = x + int(t * drift + 2 * math.sin(t * 5 + q))
        py = y - int(t * rise)
        cc = cols[min(2, int(t * 3))]
        if t > 0.8 and (q + f // 3) % 2:
            continue
        c.circle(px, py, r, cc)
        c.px(px - r + 1, py - r + 1, mix(cc, '#fff4d8', 0.4))


def wisp(c, f, x, y, h, seed=0, cols=('#e8dccb', '#d8c8b4', '#c8b6a2'), width=1, drift=6, curl=2.2):
    """A thin curling smoke/steam wisp rising from (x, y); fades out with ordered gaps."""
    n = len(cols)
    for k in range(h):
        t = k / h
        px = x + int(round(drift * t + curl * math.sin(k * 0.3 - f * 0.13 + seed) * t ** 0.6))
        py = y - k
        if t > 0.5 and (k + f // 3 + seed) % (1 + int(t * 3)) != 0:
            continue
        cc = cols[min(n - 1, int(t * n))]
        c.px(px, py, cc)
        if width > 1 and t < 0.45:
            c.px(px + 1, py, cc)


def draw_windmill(c, x, y, r, f, speed=0.22, seed=0):
    a0 = (f // 2) * speed * 2 + seed
    for k in range(6):
        a = a0 + k * math.pi / 3
        ex, ey = x + int(round(math.cos(a) * r)), y + int(round(math.sin(a) * r))
        c.line(x, y, ex, ey, '#d8c39a' if k % 2 else '#a9773f')
    c.circle(x, y, 1, '#5a3a2a')
    c.px(x, y, '#c9a06a')


def draw_washing(c, f, p0, p1, n, seed=0):
    (x0, y0), (x1, y1) = p0, p1
    pts = []
    for i in range(x1 - x0 + 1):
        t = i / max(1, x1 - x0)
        pts.append((x0 + i, int(round(y0 + (y1 - y0) * t + 2 * math.sin(math.pi * t)))))
    for (x, y) in pts:
        c.px(x, y, '#5a4a40')
    cloth = ['#e8e0cc', '#4e8c84', '#d4a640', '#b8483a', '#a884c0']
    for i in range(n):
        t = (i + 0.7) / (n + 0.4)
        x, y = pts[int(t * (len(pts) - 1))]
        cw, ch = 2 + (i + seed) % 2, 2 + (i * 3 + seed) % 2
        cc = cloth[(i + seed) % len(cloth)]
        c.rect(x, y + 1, cw, ch, cc)
        if (f // 4 + i) % 2:
            c.px(x + cw, y + ch, cc)
            c.px(x, y + ch, shade(col(cc), 0.8))
        else:
            c.px(x + cw - 1, y + ch, shade(col(cc), 0.8))


def draw_pennant(c, f, x, y, cc='#b8483a'):
    ph = (f // 4) % 3
    c.px(x, y, '#5a3a2a')
    c.px(x + 1, y - 1, cc)
    c.px(x + 2, y - 1 + (ph == 1), cc)
    c.px(x + 1, y, shade(col(cc), 0.8))
    if ph != 2:
        c.px(x + 3, y - 1 + (ph == 0), cc)


def draw_crane_hook(c, f, x, y, L, seed=0):
    a = 0.22 * math.sin(2 * math.pi * f / 54 + seed)
    hx = x + int(round(math.sin(a) * L))
    hy = y + int(round(math.cos(a) * L))
    c.line(x, y, hx, hy, '#4a3a34')
    c.px(hx, hy + 1, '#8b8d92')
    c.px(hx + 1, hy + 2, '#8b8d92')
    c.rect(hx - 2, hy + 3, 5, 4, '#a9773f')
    c.hline(hx - 2, hx + 2, hy + 3, '#d1a060')
    c.vline(hx + 2, hy + 4, hy + 6, '#6e4628')
    return hx, hy


def draw_small_airship(c, x, y, f):
    """Tiny moored airship (shot A), (x, y) = envelope left end centre."""
    s, _ = _ship(0)
    c.blit(s, x, y + int(round(math.sin(f * 2 * math.pi / 90))))


def draw_tumbleweed(c, x, y, f):
    r = 4
    ph = (f // 3) % 4
    c.circle(x, y, r, '#8a6a44', fill=False)
    c.circle(x, y, r - 1, '#a88456', fill=False)
    for k in range(4):
        a = (k + ph * 0.5) * math.pi / 4
        c.line(x - int(math.cos(a) * (r - 1)), y - int(math.sin(a) * (r - 1)),
               x + int(math.cos(a) * (r - 1)), y + int(math.sin(a) * (r - 1)), '#7a5a3a')
    c.px(x - 2, y - 3, '#c9a06a')


def sand_wisps(c, f, region, n=26, seed=0, color='#fbe4b8', speed=2.2):
    x0, y0, w, h = region
    for i in range(n):
        sx = hash01(seed, i, 1) * w
        sy = hash01(seed, i, 2) * h
        sp = speed * (0.6 + 0.8 * hash01(seed, i, 3))
        px = x0 + int((sx + f * sp) % w)
        py = y0 + int(sy + 1.5 * math.sin(f / 9 + i))
        ln = 1 + int(hash01(seed, i, 4) * 3)
        if hash01(seed, i, (f + i) // 12) > 0.3:
            c.hline(px - ln, px, py, color)


def shimmer(c, f, y0, y1, amp=0.85):
    """Heat shimmer: integer row offsets (-1/0/1) drifting slowly."""
    rows = np.arange(y0, y1)
    k = np.clip((rows - y0) / max(1, y1 - y0), 0, 1)
    s = np.round(amp * (0.4 + 0.6 * k) * np.sin(rows * 0.9 + (f // 3) * 0.55)).astype(int)
    idx = (np.arange(W)[None, :] - s[:, None]) % W
    c.a[y0:y1] = c.a[y0:y1][np.arange(y1 - y0)[:, None], idx]


@lru_cache(maxsize=1)
def _beams_a():
    yy, xx = np.mgrid[0:H, 0:W]
    sx, sy = SUN_A
    th = np.arctan2(yy - sy, xx - sx)
    d = np.hypot(xx - sx, yy - sy)
    band = (np.sin(th * 11 + 0.7) > 0.72) & (d > 13) & (d < 230) & (th > -0.5) & (th < 1.2)
    fade = np.clip(1 - d / 230.0, 0, 1)
    return band & (bayer(H, W) < 0.55 * fade + 0.1)


LIGHTS_A = [((147, 63), (199, 45), 6), ((221, 51), (268, 75), 6), ((158, 92), (182, 82), 3)]
LADDERS_A = [(124, 90, 120), (283, 94, 122)]


def render_a(f):
    cam = cam_a(f)
    c = _sky_a().copy()
    st = _stratus_a()
    comp(c, st, -int((f * 0.12 + cam * 0.04) % 480), 4)
    comp(c, st, -int((f * 0.12 + cam * 0.04) % 480) + 480, 4)
    comp(c, _mesa_a(), -int(round(cam * PAR_A['mesa'])))
    comp(c, _far_a(), -int(round(cam * PAR_A['far'])))
    shimmer(c, f, 86, 120)
    # town
    tl, anim = _town_a()
    ox = -int(round(cam * PAR_A['town']))
    for (sx, yb, yt) in STACKS_A:
        draw_smoke(c, f, sx + ox, yt - 2, seed=sx)
    draw_small_airship(c, MAST_A[0] + 2 + ox, MAST_A[1] - 4, f)
    comp(c, tl, ox)
    for (lx_, y0_, y1_) in LADDERS_A:
        c.vline(lx_ + ox, y0_, y1_, '#5a3a26')
        c.vline(lx_ + 3 + ox, y0_, y1_, '#5a3a26')
        for yy in range(y0_ + 2, y1_, 3):
            c.hline(lx_ + 1 + ox, lx_ + 2 + ox, yy, '#8a5a36')
    for li, ((x0, y0), (x1, y1), sag) in enumerate(LIGHTS_A):
        bulbs = ('#ffd27a', '#ff9a6a', '#fff0a8', '#7ac0a0', '#e86a8a')
        for i in range(x1 - x0 + 1):
            t = i / max(1, x1 - x0)
            yy = int(round(y0 + (y1 - y0) * t + sag * math.sin(math.pi * t)))
            if i % 4 == 2:
                on = hash01(li, i, (f + i * 3) // 10) > 0.2
                c.px(x0 + i + ox, yy + 1, bulbs[(i // 4 + li) % 5] if on else '#a06a40')
            else:
                c.px(x0 + i + ox, yy, '#4a3a34')
    for (x0, y0), (x1, y1), sag in BRIDGES:
        for i in range(x1 - x0 + 1):
            t = i / max(1, x1 - x0)
            yy = y0 + (y1 - y0) * t + sag * math.sin(math.pi * t)
            c.px(x0 + i + ox, int(round(yy)), '#7a5236' if i % 2 else '#a9773f')
            c.px(x0 + i + ox, int(round(yy - 3 - 0.3 * sag * math.sin(math.pi * t))), '#5a3a26')
            if i % 3 == 0:
                c.line(x0 + i + ox, int(round(yy)), x0 + i + ox,
                       int(round(yy - 3 - 0.3 * sag * math.sin(math.pi * t))), '#5a3a26')
        mxp = (x0 + x1) // 2 + ox
        myp = int(round((y0 + y1) / 2 + sag)) - 4
        if (f // 6 + x0) % 5:
            c.px(mxp, myp, '#ffd27a')
        else:
            c.px(mxp, myp, '#e8a040')
    for (wx, wy, wr, wb) in WINDMILLS_A:
        draw_windmill(c, wx + ox, wy, wr, f, seed=wx)
    mx, mb, mt, je = CRANE_A
    draw_crane_hook(c, f, je - 3 + ox, mt + 2, 16)
    for p0, p1, n in WASH_A:
        draw_washing(c, f, (p0[0] + ox, p0[1]), (p1[0] + ox, p1[1]), n, seed=p0[0])
    for (px, py) in anim.get('pennants', []):
        draw_pennant(c, f, px + ox, py)
    for (wx, wy, ww, wh, sd) in anim.get('windows', []):
        if hash01(sd, f // 9) < 0.12:
            c.rect(wx + ox, wy, ww, wh, '#e8a040')
    lx, lt, lb = LIFT_A
    ly = int(round(lerp(lb - 8, lt + 3, pingpong(((f - A0) / 120.0) % 1.0))))
    c.vline(lx + ox, lt + 1, lb, '#4a3a34')
    c.vline(lx + 5 + ox, lt + 1, lb, '#4a3a34')
    c.rect(lx + ox, ly, 6, 6, '#7a5236')
    c.rect(lx + 1 + ox, ly + 1, 4, 3, '#a9773f')
    c.hline(lx + ox, lx + 5 + ox, ly, '#b07a48')
    # balloon drifting past
    bx = int(round(lerp(241, 251, seg(f, A0, A1)))) + ox
    draw_balloon(c, bx, 66, f, size='s')
    bm = _beams_a()
    reg = c.a[bm].astype(np.int16)
    c.a[bm] = np.clip(reg + ((np.array(col('#fff4d0'), np.int16) - reg) * 3) // 20, 0, 255).astype(np.uint8)
    # mid, near, fg
    mo = -int(round(cam * PAR_A['mid']))
    comp(c, _mid_a(), mo)
    for (ax, ay) in ((150, 126), (330, 124), (120, 136)):
        for i in range(4):
            t = ((f * 0.9 + i * 11 + ax) % 40) / 40.0
            px = ax + mo + int(t * 16) + i
            py = ay - 1 + int(t * 5) - (i % 2)
            if t < 0.8:
                c.px(px, py, '#fdebc4')
    no = -int(round(cam * PAR_A['near']))
    comp(c, _near_a(), no)
    if 108 <= f < 176:
        t = (f - 108) / 68.0
        tx = int(round(lerp(-8, 300, t)))
        ty = 160 - int(abs(math.sin(t * 9.5)) * 6)
        draw_tumbleweed(c, tx, ty, f)
    comp(c, _fg_a(), -int(round(cam * PAR_A['fg'])))
    sand_wisps(c, f, (0, 150, W, 30), n=30, seed=3)
    sand_wisps(c, f, (0, 128, W, 14), n=14, seed=4, color='#fdf0d0', speed=1.2)
    fx.dust_motes(c, f, (40, 40, 200, 90), n=14, color='#fff4d0', seed=5)
    # callouts
    callout(c, 215 + ox, 47, 'KLUGETOWN', f, 90, 162, side='up', length=19)
    callout(c, 58 + mo, 92, 'BONE DRY DESERT', f, 108, 172, side='down', length=24)
    return c


# ================================================================== SHOT B: the bazaar alley
B0, B1 = 180, 270
B_CAM = 48
PAR_B = dict(sky=0.1, far=0.3, wall=0.8, stall=1.0, fg=1.4)
FLOOR = 150
SHADOW_TINT = '#5a3446'


def cam_b(f):
    t = seg(f, B0, B1 - 1)
    return B_CAM * cruise(t)


def lwb(par):
    return W + int(math.ceil(B_CAM * par)) + 4


def darken(cv, m, k=0.7, tint=SHADOW_TINT, tk=0.25):
    a = cv.a[m].astype(np.float32)
    a = a * k
    a += (np.array(col(tint), np.float32) - a) * tk
    cv.a[m] = np.clip(a, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _sky_b():
    c = Canvas(W, H, SKY_A[-1])
    c.gradient(0, 0, W, 110, SKY_A[1:], soft=0.7)
    return c


@lru_cache(maxsize=1)
def _far_b():
    """Seen through the arch: hazy dunes + a distant pillar stack."""
    w = lwb(PAR_B['far'])
    cv = new(w)
    cols = ('#d9a46e', '#e2b27a', '#ebc189', '#f2cf98', '#f7dcaa', '#fae7bf')
    rock = ['#8a5a44', '#a06a4c', '#b47c58', '#c68e66', '#d6a276', '#e2b488']
    m = pillar_mask(H, w, 252, 84, 14, 9, 17, 7, bottom=150)
    shade_solid(cv, m, rock, outline=None, strata=(5, 0.12), seed=7)
    for i, (x, yb, sw, sh_, wall, roof, rc) in enumerate([
            (237, 84, 11, 9, 'rust', 'flat', 'wood'), (248, 84, 12, 11, 'brown', 'tin', 'tin'),
            (260, 84, 9, 7, 'ochre', 'tarp', 'teal'), (240, 75, 9, 7, 'red', 'dome', 'copper'),
            (250, 73, 8, 8, 'tan', 'gable', 'red')]):
        draw_shack(cv, x, yb, sw, sh_, wall, roof, rc, seed=300 + i, win=1, lit=(0,))
    cv.rect(234, 84, 38, 2, '#7a5236')
    dune(cv, 200, 118, 90, 50, cols, seed=81)
    dune(cv, 300, 122, 70, 50, cols, seed=82)
    haze_layer(cv, HAZE, 0.22)
    return to_layer(cv)


FACADES = [  # x0, x1, top, wall colour, kind
    (0, 64, 6, '#b86e3e', 'plaster'),
    (64, 118, 20, '#8a5236', 'planks'),
    (118, 182, -10, '#c99a4a', 'plaster'),
    (182, 236, 28, '#a84a3a', 'planks'),
    (236, 278, 14, '#9a5a3a', 'arch'),
    (278, 330, 10, '#b8875a', 'plaster'),
    (330, 372, 24, '#8a4e2a', 'planks'),
]
ARCH_B = (240, 276, 74)       # arch opening x0, x1, spring y (wall coords)
WALL_SIGNS = [(110, 46, 'gear'), (226, 52, 'pot'), (324, 40, 'key')]
WALL_WINDOWS = []


def icon(cv, x, y, kind, ink='#4a2a22', fill='#d9a046'):
    """Tiny 5x5 shop icons for signs."""
    pats = {
        'gear': ['.x.x.', 'xxxxx', 'xx.xx', 'xxxxx', '.x.x.'],
        'pot': ['.....', 'x...x', 'xxxxx', 'xxxxx', '.xxx.'],
        'key': ['xx...', 'x.xxx', 'xx.x.', '.....', '.....'],
        'lamp': ['..x..', '.xxx.', '.x.x.', '.xxx.', 'xxxxx'],
        'carpet': ['xxxxx', 'x.x.x', 'xx.xx', 'x.x.x', 'xxxxx'],
        'spice': ['.xxx.', 'x...x', 'xxxxx', 'xxxxx', '.xxx.'],
        'ship': ['.xxx.', 'xxxxx', '.xxx.', '..x..', '.xxx.'],
    }
    for j, row in enumerate(pats[kind]):
        for i, ch in enumerate(row):
            if ch == 'x':
                cv.px(x + i, y + j, ink)


@lru_cache(maxsize=1)
def _wall_b():
    w = lwb(PAR_B['wall'])
    cv = new(w)
    anim = {}
    for i, (x0, x1, top, wc, kind) in enumerate(FACADES):
        base = col(wc)
        hi, md, sh, dk = mix(base, '#ffe2a8', 0.3), shade(base, 0.9), shade(base, 0.75), shade(base, 0.55)
        y0 = max(0, top)
        cv.rect(x0, y0, x1 - x0, FLOOR - y0, base)
        if kind == 'planks':
            for xx in range(x0 + 2, x1 - 1, 4):
                for yy in range(y0 + 2, FLOOR):
                    if hash01(i, xx, yy // 5) > 0.1:
                        cv.px(xx, yy, md)
            for yy in range(y0 + 6, FLOOR, 11):
                for xx in range(x0 + 1, x1 - 1):
                    if hash01(i, xx // 4, yy) > 0.4:
                        cv.px(xx, yy, sh)
        else:
            R = rng(500 + i)
            for _ in range((x1 - x0) * (FLOOR - y0) // 40):
                px_, py_ = int(R.integers(x0 + 1, x1 - 1)), int(R.integers(y0 + 2, FLOOR))
                cv.px(px_, py_, md if R.random() < 0.7 else hi)
            for _ in range(4):          # cracked plaster showing bricks
                bx, by = int(R.integers(x0 + 3, x1 - 12)), int(R.integers(y0 + 8, FLOOR - 50))
                for k in range(3):
                    cv.rect(bx + (k % 2) * 3, by + k * 3, 6, 2, '#8a4a32')
                    cv.hline(bx + (k % 2) * 3, bx + (k % 2) * 3 + 5, by + k * 3 + 2, '#6a3a2a')
        cv.vline(x0, y0, FLOOR - 1, hi)
        cv.rect(x1 - 2, y0, 2, FLOOR - y0, dk)
        cv.vline(x1 - 3, y0, FLOOR - 1, sh)
        if top >= 0:            # roof eave + tin cap
            cv.rect(x0 - 2, top - 3, x1 - x0 + 4, 3, '#7a7672')
            for xx in range(x0 - 2, x1 + 2, 2):
                cv.vline(xx, top - 3, top - 1, '#96928c')
            cv.hline(x0 - 2, x1 + 1, top - 1, '#4e4a48')
            cv.hline(x0 - 2, x1 + 1, top - 4, '#b0aca6')
            cv.hline(x0, x1 - 1, top, dk)
        # upper windows with shutters / balconies
        R = rng(700 + i)
        wy = max(y0 + 8, 14)
        for row in range(2):
            yy = wy + row * 28
            if yy > 70 or (kind == 'arch' and row == 1):
                continue
            nwin = max(1, (x1 - x0) // 26)
            for k in range(nwin):
                wx = x0 + (x1 - x0) * (k + 0.5) / nwin - 4
                wx = int(wx)
                lit = R.random() < 0.35
                cv.rect(wx - 1, yy - 1, 9, 12, shade(base, 0.45))
                cv.rect(wx, yy, 7, 10, WIN_LIT if lit else WIN_DK)
                if lit:
                    cv.rect(wx, yy, 3, 4, WIN_HI)
                    anim.setdefault('win', []).append((wx, yy, 7, 10, 700 + i * 10 + k))
                else:
                    cv.hline(wx, wx + 6, yy + 4, '#5a4050')
                    cv.vline(wx + 3, yy, yy + 9, '#5a4050')
                cv.rect(wx - 3, yy, 2, 10, '#4e8c84' if i % 2 else '#b8483a')
                cv.rect(wx + 8, yy, 2, 10, '#4e8c84' if i % 2 else '#b8483a')
                cv.hline(wx - 3, wx - 2, yy, '#8ac0b4' if i % 2 else '#e07060')
                cv.hline(wx - 2, wx + 8, yy + 11, hi)
                if row == 1 and R.random() < 0.6:     # little balcony
                    cv.rect(wx - 4, yy + 12, 15, 2, '#7a5236')
                    cv.hline(wx - 4, wx + 10, yy + 12, '#b07a48')
                    for xx in range(wx - 4, wx + 11, 2):
                        cv.vline(xx, yy + 6, yy + 11, '#5a3a26')
                    cv.hline(wx - 4, wx + 10, yy + 6, '#7a5236')
                    cv.line(wx - 3, yy + 14, wx + 1, yy + 18, '#5a3a26')
                    cv.rect(wx + 6, yy + 3, 3, 3, '#b8693e')     # flower pot
                    cv.px(wx + 7, yy + 2, '#7ea864')
                    cv.px(wx + 6, yy + 1, '#e86a8a')
    # the arch passage with the far view
    ax0, ax1, ay = ARCH_B
    yy, xx = np.mgrid[0:H, 0:w]
    rx = (ax1 - ax0) / 2
    acx = (ax0 + ax1) / 2
    hole = (((xx + 0.5 - acx) / rx) ** 2 + ((yy + 0.5 - ay) / (rx * 0.9)) ** 2 <= 1) | (
        (np.abs(xx + 0.5 - acx) <= rx) & (yy >= ay) & (yy < FLOOR))
    ring = (((xx + 0.5 - acx) / (rx + 3)) ** 2 + ((yy + 0.5 - ay) / (rx * 0.9 + 3)) ** 2 <= 1) & ~hole & (yy < ay + 1)
    for k, yy0 in enumerate(range(int(ay - rx), FLOOR)):
        pass
    cv.a[ring] = C('#c47f50')
    stones = ring & ((xx + yy) % 5 == 0)
    cv.a[stones] = C('#8a4a32')
    jamb = (np.abs(xx + 0.5 - acx) <= rx + 3) & ~hole & (yy >= ay) & (yy < FLOOR) & (np.abs(xx + 0.5 - acx) > rx)
    cv.a[jamb] = C('#a8653a')
    cv.a[jamb & (yy % 6 == 0)] = C('#7a4a32')
    cv.a[hole] = C(KEY)
    # pipes along the wall
    for (px, y0_, y1_) in ((60, 10, 150), (176, 30, 150), (330, 24, 150)):
        cv.rect(px, y0_, 3, y1_ - y0_, '#6f6a6a')
        cv.vline(px, y0_, y1_ - 1, '#9a9690')
        cv.vline(px + 2, y0_, y1_ - 1, '#4c4848')
        for yy_ in range(y0_ + 6, y1_, 14):
            cv.rect(px - 1, yy_, 5, 2, '#9e5539')
    cv.rect(60, 40, 120, 3, '#6f6a6a')
    cv.hline(60, 179, 40, '#9a9690')
    cv.hline(60, 179, 42, '#4c4848')
    # hanging shop signs (static brackets)
    for (sx, sy, kind) in WALL_SIGNS:
        cv.hline(sx - 6, sx + 1, sy, '#4a3a34')
        cv.px(sx - 6, sy + 1, '#4a3a34')
    # light shafts from the low sun (upper left): brighten diagonal bands
    shaft = np.zeros((H, w), bool)
    for (s0, sw) in ((-40, 26), (70, 18), (150, 30), (250, 22), (330, 26)):
        u = xx - (yy * 0.75) - s0
        shaft |= (u >= 0) & (u < sw)
    lm = shaft & ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    a = cv.a[lm].astype(np.float32)
    a += (np.array(col('#ffd890'), np.float32) - a) * 0.28
    cv.a[lm] = np.clip(a, 0, 255).astype(np.uint8)
    # ambient occlusion low on the wall
    low = (yy > 100) & ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    darken(cv, low & (bayer(H, w) < (yy - 100) / 60.0), k=0.85, tk=0.1)
    return to_layer(cv), anim


# ---------- stalls
STALLS = [  # x0, x1, awning y, stripe a, stripe b, goods
    (6, 74, 92, '#b8483a', '#efe2c4', 'spice'),
    (84, 150, 88, '#4e8c84', '#efe2c4', 'carpet'),
    (160, 232, 94, '#d4a640', '#7a4a2e', 'lamp'),
    (244, 306, 90, '#8e8a86', '#b8875a', 'junk'),
    (316, 376, 92, '#7e4a5e', '#efe2c4', 'fruit'),
]
LANTERNS_B = []
COUNTER = 132


def awning(cv, x0, x1, y, ca, cb, depth=10, val=4):
    """Striped awning (side-on, sloping down to the front) with a scalloped valance."""
    for xx in range(x0, x1):
        k = (xx - x0) // 5
        cc = col(ca) if k % 2 == 0 else col(cb)
        for j in range(depth):
            t = j / depth
            cv.px(xx, y + j, mix(shade(cc, 0.72), cc, t ** 0.7) if j < depth - 1 else mix(cc, '#fff4d8', 0.25))
    cv.hline(x0, x1 - 1, y, '#4a2e28')
    for xx in range(x0, x1):
        k = (xx - x0) // 5
        cc = col(ca) if k % 2 == 0 else col(cb)
        ph = (xx - x0) % 6
        L = val - (1 if ph in (0, 5) else 0)
        for j in range(L):
            cv.px(xx, y + depth + j, shade(cc, 0.88) if j else cc)
    # frame rod + posts
    cv.hline(x0 - 1, x1, y + depth - 1, '#6e4628')
    cv.rect(x0, y + depth, 2, FLOOR - (y + depth), '#6e4628')
    cv.vline(x0, y + depth, FLOOR - 1, '#9a6a40')
    cv.rect(x1 - 2, y + depth, 2, FLOOR - (y + depth), '#5a3a26')


def sack(cv, x, yb, w, h, cloth, spice, hi):
    """Open burlap sack with a mound of spice."""
    cv.rect(x, yb - h, w, h, cloth)
    cv.vline(x, yb - h, yb - 1, mix(cloth, '#fff0c8', 0.25))
    cv.vline(x + w - 1, yb - h, yb - 1, shade(col(cloth), 0.7))
    cv.hline(x - 1, x + w, yb - h, shade(col(cloth), 0.85))
    cv.hline(x - 1, x + w, yb - h + 1, mix(cloth, '#fff0c8', 0.2))
    cv.hline(x, x + w - 1, yb - 1, shade(col(cloth), 0.65))
    mh = max(2, w // 3)
    for j in range(mh):
        half = int((w / 2) * (j + 1) / mh)
        cxm = x + w // 2
        cv.hline(cxm - half, cxm + half - 1, yb - h - mh + j, spice)
    cv.px(x + w // 2 - 1, yb - h - mh, hi)
    cv.px(x + w // 2 - 2, yb - h - mh + 1, hi)
    cv.px(x + w // 2 + 1, yb - h - mh + 2, shade(col(spice), 0.8))
    cv.px(x + 2, yb - h + 3, shade(col(cloth), 0.8))
    cv.px(x + w - 3, yb - h + 5, shade(col(cloth), 0.8))


def jar(cv, x, yb, h, c_):
    cv.rect(x, yb - h, 4, h, c_)
    cv.vline(x, yb - h, yb - 1, mix(c_, '#ffffff', 0.35))
    cv.vline(x + 3, yb - h, yb - 1, shade(col(c_), 0.7))
    cv.hline(x + 1, x + 2, yb - h - 1, '#7a5236')


def gear(cv, cx, cy, r, c_, hole=True):
    for k in range(8):
        a = k * math.pi / 4
        cv.rect(cx + int(round(math.cos(a) * r)) - 1, cy + int(round(math.sin(a) * r)) - 1, 2, 2, c_)
    cv.circle(cx, cy, r - 1, c_)
    cv.px(cx - r // 2, cy - r // 2, mix(c_, '#ffffff', 0.4))
    if hole:
        cv.circle(cx, cy, max(1, r // 3), '#3e2a24')


def barrel(cv, x, yb, w, h):
    cv.round_rect(x, yb - h, w, h, '#8a5a36', r=1)
    cv.vline(x + 1, yb - h + 1, yb - 2, '#b07a48')
    cv.vline(x + w - 2, yb - h + 1, yb - 2, '#5a3a24')
    for yy in (yb - h + 2, yb - 3):
        cv.hline(x, x + w - 1, yy, '#5c5656')
        cv.px(x + 1, yy, '#8b8d92')
    cv.hline(x + 1, x + w - 2, yb - h, '#a06a40')


def crate(cv, x, yb, w, h, light=False):
    c0 = '#b8864e' if light else '#a9773f'
    cv.rect(x, yb - h, w, h, c0)
    cv.frame(x, yb - h, w, h, '#6e4628')
    cv.line(x + 1, yb - 2, x + w - 2, yb - h + 1, '#8a5a36')
    cv.hline(x + 1, x + w - 2, yb - h + 1, '#d1a060')
    cv.vline(x + w - 1, yb - h, yb - 1, '#5a3a24')


def carpet(cv, x, y, w, h, base, accent, edge, seed=0):
    cv.rect(x, y, w, h, base)
    cv.frame(x + 1, y + 1, w - 2, h - 2, edge)
    cv.frame(x + 3, y + 3, w - 6, h - 6, accent)
    cxp, cyp = x + w // 2, y + h // 2
    for k in range(min(w, h) // 2 - 4):
        cv.px(cxp - k, cyp - (min(w, h) // 2 - 5 - k) // 2, accent)
    for j in range(-3, 4):
        half = 3 - abs(j)
        cv.hline(cxp - half, cxp + half, cyp + j, accent if j % 2 else edge)
    for xx in range(x, x + w, 2):
        cv.px(xx, y + h, edge)


def bell(cv, cx, top):
    cv.rect(cx - 9, top - 4, 19, 2, '#6e4628')
    cv.vline(cx - 9, top - 2, top + 22, '#6e4628')
    cv.vline(cx + 9, top - 2, top + 22, '#6e4628')
    cv.vline(cx, top - 2, top, '#4a3a34')
    for j in range(13):
        half = 2 + int(j * 0.45) + (2 if j > 10 else 0)
        for i in range(-half, half + 1):
            u = (i + half) / (2 * half)
            cc = '#f0c060' if u < 0.25 else '#c9a24a' if u < 0.6 else '#8a6a30'
            cv.px(cx + i, top + 1 + j, cc)
    cv.line(cx + 1, top + 4, cx + 3, top + 9, '#4a3a20')
    cv.line(cx + 3, top + 9, cx + 2, top + 13, '#4a3a20')
    cv.px(cx, top + 15, '#5a4a30')


@lru_cache(maxsize=1)
def _stalls_b():
    w = lwb(PAR_B['stall'])
    cv = new(w)
    anim = {'lanterns': [], 'smoke': [], 'steam': [], 'valance': []}
    for si, (x0, x1, ay, ca, cb, goods) in enumerate(STALLS):
        R = rng(900 + si)
        # booth back panel in the awning shade
        cv.rect(x0 + 2, ay + 10, x1 - x0 - 4, FLOOR - ay - 10, '#5e3a30')
        for xx in range(x0 + 4, x1 - 3, 4):
            cv.vline(xx, ay + 10, FLOOR - 1, '#523228')
        # shelves
        for sy in (ay + 22, ay + 32):
            cv.hline(x0 + 3, x1 - 4, sy, '#8a5a36')
            cv.hline(x0 + 3, x1 - 4, sy + 1, '#3e2620')
        # counter
        cv.rect(x0 + 1, COUNTER, x1 - x0 - 2, FLOOR - COUNTER, '#8a5a36')
        cv.hline(x0, x1 - 1, COUNTER, '#c48a52')
        cv.hline(x0, x1 - 1, COUNTER + 1, '#a06a40')
        for xx in range(x0 + 3, x1 - 2, 5):
            cv.vline(xx, COUNTER + 2, FLOOR - 1, '#6e4628')
        cv.hline(x0 + 1, x1 - 2, FLOOR - 1, '#4e3020')
        # cloth drape on counter front
        drape = [ca, cb][si % 2]
        for xx in range(x0 + 4, x1 - 4):
            L = 6 + int(2 * math.sin((xx - x0) * 0.5))
            cv.vline(xx, COUNTER + 2, COUNTER + 1 + L, drape if (xx // 3) % 2 else shade(col(drape), 0.85))
        if goods == 'spice':
            jcols = ['#d94a2a', '#e8b82a', '#4e8c84', '#a884c0', '#7ea864', '#c87a3a']
            for k in range(9):
                jar(cv, x0 + 6 + k * 6, ay + 22, 5 + (k % 2) * 2, jcols[k % 6])
            for k in range(8):
                jar(cv, x0 + 8 + k * 7, ay + 32, 5, jcols[(k + 3) % 6])
            sp = [('#c9a06a', '#e8641e', '#ffb070'), ('#b8956a', '#f2c230', '#fff0a0'),
                  ('#c9a06a', '#b8301e', '#e86a50'), ('#b8956a', '#3e8a6e', '#7ac0a0'),
                  ('#c9a06a', '#8a4ab0', '#c08ae0'), ('#b8956a', '#d98a2a', '#f8c070')]
            for k in range(5):
                cl, s1, s2 = sp[k]
                sack(cv, x0 + 4 + k * 13, COUNTER, 11, 6, cl, s1, s2)
            for k in range(4):    # floor sacks in front
                cl, s1, s2 = sp[(k + 2) % 6]
                sack(cv, x0 + 8 + k * 15, FLOOR + 6 + (k % 2) * 3, 13, 9, cl, s1, s2)
            anim['smoke'].append((x0 + 60, ay + 24))
        elif goods == 'carpet':
            cv.hline(x0 + 2, x1 - 3, ay + 12, '#4a3a34')
            cps = [('#9a2e3a', '#e8b82a', '#2e4a6a'), ('#2e5a6a', '#e8d0a0', '#b8483a'),
                   ('#7a3a6a', '#f0c060', '#3e2a40'), ('#b85a2a', '#efe2c4', '#4e2a20')]
            for k in range(4):
                cxp = x0 + 4 + k * 15
                b_, a_, e_ = cps[k]
                carpet(cv, cxp, ay + 13, 13, 30 + (k % 2) * 6, b_, a_, e_)
                anim['valance'].append((cxp, ay + 13 + 30 + (k % 2) * 6, 13, e_))
            for k in range(3):      # rolled carpets standing against the counter
                rx = x0 + 10 + k * 17
                b_, a_, e_ = cps[k]
                top = COUNTER - 14 + (k % 2) * 4
                for yy_ in range(top, FLOOR + 3):
                    lean = 1 if yy_ < top + (FLOOR + 3 - top) // 2 else 0
                    for j, cc in enumerate((mix(b_, '#fff0c8', 0.3), b_, b_, shade(col(b_), 0.8),
                                            shade(col(b_), 0.6))):
                        cv.px(rx + j + lean, yy_, cc)
                    if (yy_ - top) % 7 in (3, 4):
                        cv.hline(rx + lean, rx + 3 + lean, yy_, a_)
                cv.hline(rx + 2, rx + 4, top - 1, e_)
                cv.px(rx + 3, top, a_)
        elif goods == 'lamp':
            for k in range(7):
                lx_ = x0 + 7 + k * 9
                ly_ = ay + 14 + (k % 3) * 5
                cv.vline(lx_, ay + 10, ly_ - 1, '#4a3a34')
                gc = ['#ffc861', '#ff9a50', '#7ac0a0', '#ffc861', '#e86a8a', '#ffd27a', '#a8c0ff'][k]
                anim['lanterns'].append((lx_, ly_, gc, 900 + k))
            # brass pots on the shelf, kettle on a brazier in front
            for k in range(6):
                px_ = x0 + 6 + k * 11
                cv.ellipse(px_ + 3, ay + 30, 3, 2, '#c9a24a')
                cv.px(px_ + 1, ay + 29, '#f0d080')
                cv.hline(px_ + 1, px_ + 5, ay + 31, '#8a6a30')
            bx_ = x0 + 30
            cv.rect(bx_ - 4, COUNTER - 2, 9, 2, '#3e3a3a')
            cv.ellipse(bx_, COUNTER - 5, 4, 3, '#c9a24a')
            cv.px(bx_ - 2, COUNTER - 7, '#f0d080')
            cv.line(bx_ + 4, COUNTER - 6, bx_ + 7, COUNTER - 9, '#c9a24a')
            cv.rect(bx_ - 1, COUNTER - 9, 2, 2, '#8a6a30')
            anim['steam'].append((bx_ + 7, COUNTER - 10))
            for k in range(5):
                gear(cv, x0 + 46 + k * 4, COUNTER - 3, 2, '#b8b0a0' if k % 2 else '#c9a24a', hole=False)
        elif goods == 'junk':
            bell(cv, x0 + 46, ay + 16)
            for (gx, gy, gr, gc) in ((x0 + 10, COUNTER - 5, 5, '#8b8d92'), (x0 + 20, COUNTER - 4, 4, '#b8693e'),
                                     (x0 + 15, COUNTER - 11, 4, '#a8a29a'), (x0 + 28, COUNTER - 3, 3, '#c9a24a'),
                                     (x0 + 12, FLOOR + 2, 6, '#7a7672'), (x0 + 26, FLOOR + 3, 5, '#9e5539'),
                                     (x0 + 40, FLOOR + 4, 4, '#8b8d92')):
                gear(cv, gx, gy, gr, gc)
            # pots + pans hanging on hooks
            for k in range(5):
                hx = x0 + 6 + k * 7
                cv.vline(hx, ay + 10, ay + 13, '#4a3a34')
                cv.ellipse(hx, ay + 17, 3, 3, ['#4a4a4e', '#b8693e', '#5c5656', '#c9a24a', '#4a4a4e'][k])
                cv.px(hx - 1, ay + 15, '#9a9a9e')
            # pipes + a clock on the shelf
            cv.rect(x0 + 34, ay + 26, 18, 2, '#7a7672')
            cv.circle(x0 + 8, ay + 28, 4, '#c9a24a')
            cv.circle(x0 + 8, ay + 28, 3, '#efe2c4')
            cv.vline(x0 + 8, ay + 26, ay + 28, '#3e2a24')
            cv.hline(x0 + 8, x0 + 10, ay + 28, '#3e2a24')
            for k in range(4):
                cv.rect(x0 + 32 + k * 5, COUNTER - 6, 4, 6, ['#5c5656', '#b8693e', '#8b8d92', '#4e8c84'][k])
                cv.hline(x0 + 32 + k * 5, x0 + 35 + k * 5, COUNTER - 6, '#c9ccd0')
        elif goods == 'fruit':
            for k in range(3):
                crate(cv, x0 + 4 + k * 17, COUNTER, 15, 8, light=k % 2 == 1)
                for j in range(4):
                    fc = ['#e8641e', '#d94a2a', '#f2c230'][k]
                    cv.circle(x0 + 7 + k * 17 + j * 3, COUNTER - 9, 2, fc)
                    cv.px(x0 + 6 + k * 17 + j * 3, COUNTER - 10, mix(fc, '#ffffff', 0.5))
            for k in range(4):     # date baskets on the shelf
                bx_ = x0 + 6 + k * 13
                cv.rect(bx_, ay + 18, 10, 4, '#c9a06a')
                cv.hline(bx_, bx_ + 9, ay + 18, '#e8c890')
                for j in range(4):
                    cv.px(bx_ + 2 + j * 2, ay + 17, '#7a3a2a')
            barrel(cv, x0 + 6, FLOOR + 6, 10, 14)
            barrel(cv, x0 + 18, FLOOR + 7, 10, 12)
            crate(cv, x0 + 32, FLOOR + 6, 12, 10)
            crate(cv, x0 + 34, FLOOR - 4, 10, 8, light=True)
        awning(cv, x0, x1, ay, ca, cb)
        for xx in range(x0, x1, 6):
            anim['valance'].append((xx, ay + 10 + 4, 6, (ca, cb)))
    # between-stall junk + barrels on the street
    barrel(cv, 76, FLOOR + 4, 9, 13)
    crate(cv, 152, FLOOR + 3, 10, 9)
    barrel(cv, 234, FLOOR + 5, 9, 12)
    crate(cv, 306, FLOOR + 4, 11, 9, light=True)
    # street (cobbles) 150..180
    yy, xx = np.mgrid[0:H, 0:w]
    floor = (yy >= FLOOR) & np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    street = np.zeros((H, w), bool)
    street[FLOOR:, :] = True
    tmp = new(w)
    cobbles(tmp, street, lambda y: 2 + (y - FLOOR) // 10, seed=17,
            stone=('#8a6448', '#9c7454', '#ae8662', '#c09872', '#d2ac84'), gap='#5e4434')
    np.copyto(cv.a, tmp.a, where=floor[:, :, None])
    # contact shadows under stalls and goods on the street
    sh = (yy >= FLOOR) & (yy < FLOOR + 5) & floor
    darken(cv, sh & (bayer(H, w) < 0.6 - (yy - FLOOR) * 0.1), k=0.75, tk=0.2)
    # sun shafts on the street + goods
    shaft = np.zeros((H, w), bool)
    for (s0, sw) in ((-20, 30), (96, 22), (196, 34), (306, 26)):
        u = xx - (yy * 0.75) - s0
        shaft |= (u >= 0) & (u < sw)
    lm = shaft & (yy > 120) & ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    a = cv.a[lm].astype(np.float32)
    a += (np.array(col('#ffd890'), np.float32) - a) * 0.22
    cv.a[lm] = np.clip(a, 0, 255).astype(np.uint8)
    return to_layer(cv), anim


@lru_cache(maxsize=1)
def _fg_b():
    w = lwb(PAR_B['fg'])
    cv = new(w)
    # left post with bracket
    PX = 96
    cv.rect(PX, 0, 6, H, '#5a3a26')
    cv.vline(PX, 0, H - 1, '#8a5a36')
    cv.vline(PX + 5, 0, H - 1, '#3e2620')
    for yy in range(8, H, 22):
        cv.hline(PX, PX + 5, yy, '#3e2620')
    cv.rect(PX + 6, 34, 26, 3, '#5a3a26')
    cv.hline(PX + 6, PX + 31, 34, '#8a5a36')
    cv.line(PX + 6, 52, PX + 22, 37, '#5a3a26')
    cv.rect(PX - 2, 150, 10, 30, '#4e3020')
    cv.hline(PX - 2, PX + 7, 150, '#8a5a36')
    # crates + sacks bottom right
    for (x, yb, w_, h_) in ((330, 180, 26, 20), (356, 180, 24, 16), (338, 160, 18, 14)):
        cv.rect(x, yb - h_, w_, h_, '#8a5a36')
        cv.frame(x, yb - h_, w_, h_, '#4e3020')
        cv.hline(x + 1, x + w_ - 2, yb - h_ + 1, '#c48a52')
        cv.line(x + 2, yb - 2, x + w_ - 3, yb - h_ + 2, '#6e4628')
        cv.vline(x + w_ - 1, yb - h_, yb - 1, '#3e2620')
    sack(cv, 384, 182, 16, 14, '#b8956a', '#e8641e', '#ffb070')
    sack(cv, 300, 184, 18, 12, '#c9a06a', '#f2c230', '#fff0a0')
    return to_layer(cv)


def draw_lantern(c, x, y, gc, f, seed, big=False):
    sw = int(round(math.sin(f * 2 * math.pi / 60 + seed) * 0.6))
    x += sw
    on = hash01(seed, f // 4) > 0.12
    g = gc if on else shade(col(gc), 0.8)
    if big:
        c.rect(x - 2, y, 5, 7, '#6e4628')
        c.rect(x - 1, y + 1, 3, 5, g)
        c.px(x - 1, y + 1, mix(g, '#ffffff', 0.5))
        c.hline(x - 2, x + 2, y - 1, '#4a3a34')
        c.px(x, y + 7, '#4a3a34')
    else:
        c.rect(x - 1, y, 3, 4, '#6e4628')
        c.px(x, y + 1, g)
        c.px(x, y + 2, g)
        c.px(x, y + 4, '#4a3a34')
    return x


def render_b(f):
    cam = cam_b(f)
    c = _sky_b().copy()
    fo = -int(round(cam * PAR_B['far']))
    comp(c, _far_b(), fo)
    draw_balloon(c, int(round(lerp(264, 274, seg(f, B0, B1)))) + fo, 86, f, size='s')
    wl_, wanim = _wall_b()
    wo = -int(round(cam * PAR_B['wall']))
    shimmer(c, f, 96, 150, amp=0.6)
    comp(c, wl_, wo)
    for (wx, wy, ww, wh, sd) in wanim.get('win', []):
        if hash01(sd, f // 8) < 0.15:
            c.rect(wx + wo, wy, 3, 4, WIN_LIT)
    # washing lines across the alley (between facades, upper area)
    draw_washing(c, f, (40 + wo, 64), (100 + wo, 60), 6, seed=1)
    draw_washing(c, f, (196 + wo, 70), (238 + wo, 74), 4, seed=3)
    draw_washing(c, f, (282 + wo, 58), (330 + wo, 62), 5, seed=5)
    # swinging hanging signs on the wall
    for (sx, sy, kind) in WALL_SIGNS:
        a = 0.18 * math.sin(2 * math.pi * (f - B0) / 48 + sx)
        x = sx + wo - 3
        dx = int(round(math.sin(a) * 4))
        c.line(x - 2, sy + 1, x - 2 + dx, sy + 4, '#4a3a34')
        c.line(x + 1, sy + 1, x + 1 + dx, sy + 4, '#4a3a34')
        c.rect(x - 5 + dx, sy + 4, 10, 8, '#b07a48')
        c.frame(x - 5 + dx, sy + 4, 10, 8, '#5a3a24')
        icon(c, x - 3 + dx, sy + 5, kind)
    so = -int(round(cam * PAR_B['stall']))
    sl, sanim = _stalls_b()
    comp(c, sl, so)
    for (lx, ly, gc, sd) in sanim['lanterns']:
        x = draw_lantern(c, lx + so, ly, gc, f, sd, big=True)
        fx.glow(c, x, ly + 3, 10, gc, strength=0.32, rings=3)
    for (x, y) in sanim['smoke']:
        wisp(c, f, x + so, y - 1, 22, seed=x, drift=5)
        wisp(c, f, x + so + 1, y - 1, 16, seed=x + 3, drift=8, curl=1.6)
    for (x, y) in sanim['steam']:
        wisp(c, f, x + so, y, 12, seed=x, cols=('#fbf6ea', '#f0e6d4', '#e2d6c2'), drift=4, curl=1.4)
    # street brazier: glowing coals + a tall curling smoke column
    bx_, by_ = 296 + so, 158
    c.line(bx_ - 4, by_, bx_ - 2, by_ - 6, '#3e3a3a')
    c.line(bx_ + 4, by_, bx_ + 2, by_ - 6, '#3e3a3a')
    c.vline(bx_, by_ - 6, by_, '#3e3a3a')
    c.rect(bx_ - 5, by_ - 10, 11, 4, '#4e4a48')
    c.hline(bx_ - 6, bx_ + 6, by_ - 10, '#7d7676')
    for i in range(-4, 5):
        c.px(bx_ + i, by_ - 11, ('#ff9a30', '#ffd27a', '#e85a20')[(i + f // 3) % 3])
    c.px(bx_ - 1, by_ - 12, '#ffd27a' if (f // 4) % 2 else '#ff9a30')
    fx.glow(c, bx_, by_ - 11, 12, '#ff9a40', strength=0.3, rings=3)
    wisp(c, f, bx_, by_ - 13, 46, seed=1, width=2, drift=10, curl=3.0,
         cols=('#b8a898', '#c8b8a6', '#d8c8b6', '#e4d6c4'))
    wisp(c, f, bx_ - 1, by_ - 13, 34, seed=4, drift=14, curl=2.4, cols=('#c8b8a6', '#d8c8b6', '#e4d6c4'))
    for (x, y, w_, cc) in sanim['valance']:
        if isinstance(cc, tuple):
            if (f // 5 + x // 6) % 3 == 0:
                k = ((x - 6) // 5) % 2
                c.px(x + 2 + so, y, cc[k])
        elif (f // 6 + x) % 2:
            c.px(x + 1 + so, y + 1, cc)
            c.px(x + w_ - 2 + so, y + 1, cc)
    # big creaky sign swinging from the fg bracket
    fgo = -int(round(cam * PAR_B['fg']))
    a = 0.3 * math.sin(2 * math.pi * (f - B0) / 36)
    px_, py_ = 126 + fgo, 37
    L = 6
    dx = int(round(math.sin(a) * L))
    c.line(px_ - 6, py_, px_ - 6 + dx, py_ + L, '#4a3a34')
    c.line(px_ + 2, py_, px_ + 2 + dx, py_ + L, '#4a3a34')
    sx0, sy0 = px_ - 12 + dx, py_ + L
    c.round_rect(sx0, sy0, 20, 14, '#5a3a24', r=1)
    c.rect(sx0 + 1, sy0 + 1, 18, 12, '#c48a52')
    c.hline(sx0 + 1, sx0 + 18, sy0 + 1, '#e0aa6a')
    icon(c, sx0 + 4, sy0 + 4, 'lamp', ink='#5a2a1e')
    icon(c, sx0 + 11, sy0 + 4, 'carpet', ink='#5a2a1e')
    comp(c, _fg_b(), fgo)
    # bunting across the top
    for row, (y0, y1, seedb) in enumerate(((6, 14, 1), (18, 12, 2))):
        x0b, x1b = 56, 262
        cols_ = ['#b8483a', '#efe2c4', '#4e8c84', '#d4a640', '#7e4a5e']
        for i in range(x1b - x0b):
            t = i / (x1b - x0b)
            y = int(round(y0 + (y1 - y0) * t + 5 * math.sin(math.pi * t)))
            c.px(x0b + i, y, '#4a3a34')
            if i % 8 == 0 and 2 < i < x1b - x0b - 4:
                cc = cols_[(i // 8 + seedb) % 5]
                flap = (f // 4 + i // 8) % 2
                c.rect(x0b + i, y + 1, 4, 2, cc)
                c.rect(x0b + i + 1, y + 3, 2, 1 + flap, cc)
                c.px(x0b + i + 3, y + 1, shade(col(cc), 0.8))
    fx.dust_motes(c, f, (60, 40, 220, 110), n=22, color='#fff0c0', seed=9)
    callout(c, 196 + so, 92, 'THE BAZAAR', f, 180, 256, side='up', length=20)
    return c


# ================================================================== SHOT C: airship docks
C0, C1 = 270, 360
C_CAM = 40
PAR_C = dict(peaks=0.06, cloud0=0.12, cloud1=0.3, cloud2=0.55, dock=1.0, fg=1.35)
SKY_C = ['#9e5a4a', '#b0664e', '#c27454', '#d4845a', '#e29862', '#ebac6c', '#f2c07c', '#f6d290',
         '#f8dfa6']
HORIZON_C = 104
AIRSHIP = 288


def cam_c(f):
    t = seg(f, C0, C1 - 1)
    return C_CAM * cruise(t)


def lwc(par):
    return W + int(math.ceil(C_CAM * par)) + 4


@lru_cache(maxsize=1)
def _sky_c():
    c = Canvas(W, H, SKY_C[-1])
    c.gradient(0, 0, W, 112, SKY_C, soft=0.7)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx + 30.0) ** 2 + ((yy - 70) * 1.3) ** 2)
    lvl = np.clip(1 - d / 190.0, 0, 1) ** 1.5 * 5
    q = np.clip(np.floor(lvl + (bayer(H, W) - 0.5) * 0.9), 0, 5) / 5.0 * 0.45
    a = c.a.astype(np.float32)
    a += (np.array(col('#ffe6b0'), np.float32) - a) * q[:, :, None]
    c.a[:] = np.clip(a + 0.5, 0, 255).astype(np.uint8)
    return c


@lru_cache(maxsize=1)
def _peaks_c():
    w = lwc(PAR_C['peaks'])
    cv = new(w)
    ramp = ['#1e1a24', '#28222c', '#342a32', '#46363a', '#6a4c46']
    R = rng(77)
    x = 6
    while x < w:
        hgt = int(R.integers(8, 34))
        wd = int(R.integers(10, 24))
        pts = [(x - wd // 2, HORIZON_C + 6), (x - 2, HORIZON_C - hgt + 3), (x, HORIZON_C - hgt),
               (x + 3, HORIZON_C - hgt + 4), (x + wd // 2 + 4, HORIZON_C + 6)]
        m = poly_mask(H, w, pts)
        shade_solid(cv, m, ramp, outline=None, lit=0.3, dark=0.4, top_hi=True, seed=x)
        x += int(R.integers(18, 46))
    haze_layer(cv, '#c88068', 0.07)
    return to_layer(cv)


CLOUD_RAMPS = (
    ['#b0968a', '#bea292', '#ccaf9c', '#dbbea8', '#e9ceb4', '#f3dcc0'],
    ['#8a7268', '#9a8074', '#ac9082', '#c2a490', '#d8b89e', '#ead0b0'],
    ['#66524e', '#76605a', '#8a7268', '#a6887a', '#c8a286', '#e2bc92'],
)
CLOUD_ROWS = (   # (y centre, r min, r max) rows per band, back to front
    ((HORIZON_C + 3, 3, 6), (HORIZON_C + 7, 4, 7)),
    ((HORIZON_C + 16, 6, 10), (HORIZON_C + 24, 7, 12)),
    ((HORIZON_C + 44, 10, 16), (HORIZON_C + 56, 12, 19)),
)


@lru_cache(maxsize=None)
def _cloud_band(level):
    """Tileable billowing cloud band (width 480), flattened puffs lit from the upper left."""
    TW = 480
    cv = new(TW)
    ramp = CLOUD_RAMPS[level]
    R = rng(600 + level)
    yy, xx = np.mgrid[0:H, 0:TW].astype(np.float32)
    for (yc, r0, r1) in CLOUD_ROWS[level]:
        x = float(R.integers(0, 12))
        blobs = []
        while x < TW:
            r = int(R.integers(r0, r1 + 1))
            blobs.append((x, yc + int(R.integers(-r0 // 2, r0 // 2 + 1)), r))
            x += r * (1.3 + 0.7 * R.random())
        best = np.full((H, TW), -1e9, np.float32)
        T = np.zeros((H, TW), np.float32)
        for (bx, by, r) in blobs:
            for off in (-TW, 0, TW):
                u = (xx + 0.5 - bx - off) / (r * 1.45)
                v = (yy + 0.5 - by) / (r * 0.72)
                d2 = u * u + v * v
                ins = (d2 <= 1) | ((yy > by) & (np.abs(u) <= 1))
                hgt = -by + np.sqrt(np.clip(1 - d2, 0, 1)) * 0.5
                upd = ins & (hgt > best)
                best[upd] = hgt[upd]
                lam = -0.8 * np.clip(v, -1, 1.6) - 0.3 * u + 0.25
                T[upd] = lam[upd]
        m = best > -1e8
        t = np.clip((T + 0.9) / 2.0, 0, 0.999)
        idx = quant(t, len(ramp), 0, 0, 0.85)
        out = ramp_arr(ramp)[idx]
        rim = outline_of(m) & (shift(~m, 0, 1) | shift(~m, 1, 0))
        out[rim] = C(mix(ramp[-1], '#fff0c8', 0.35))
        np.copyto(cv.a, out, where=m[:, :, None])
    return to_layer(cv)


PIER1_Y = 100
PIER2_Y = 142
SHIP_X0, SHIP_Y0 = 132, 57     # ship sprite top-left at moor (dock coords)


@lru_cache(maxsize=1)
def _dock_c():
    w = lwc(PAR_C['dock'])
    cv = new(w)
    anim = {}
    rock = ['#4e2e24', '#6a3e2c', '#83503a', '#9c6244', '#b67650', '#cc8e62', '#dca474']
    m = pillar_mask(H, w, 22, 22, 44, 36, 42, 9, bottom=H + 4)
    shade_solid(cv, m, rock, outline='#3e2420', strata=(7, 0.14), lit=0.22, dark=0.6, seed=9, cracks=22)
    # stacked shacks on the pillar top + cantilevered rooms
    specs = [(-4, 22, 26, 18, 'rust', 'flat', 'wood', True, 2, (0,)),
             (22, 22, 24, 14, 'brown', 'tin', 'tin', False, 2, (1,)),
             (46, 22, 18, 12, 'ochre', 'tarp', 'teal', False, 1, ()),
             (2, 2, 22, 16, 'red', 'flat', 'wood', False, 2, (0,)),
             (26, 6, 18, 14, 'tan', 'dome', 'copper', False, 1, (0,)),
             (52, 58, 22, 16, 'umber', 'tin', 'rusty', True, 1, (0,)),
             (58, 84, 16, 12, 'clay', 'flat', 'wood', False, 1, ()),
             (40, 130, 22, 14, 'rust', 'tarp', 'red', True, 1, (0,))]
    for i, (x, yb, sw, sh_, wall, roof, rc, door, win, lit) in enumerate(specs):
        draw_shack(cv, x, yb, sw, sh_, wall, roof, rc, seed=800 + i, door=door, win=win, lit=lit, anim=anim, s=2)
    for (x0, x1, y) in ((48, 76, 58), (54, 76, 84), (36, 64, 130)):
        cv.rect(x0, y, x1 - x0, 3, '#7a5236')
        cv.hline(x0, x1 - 1, y, '#b07a48')
        cv.line(x1 - 2, y + 3, x0 + 2, y + 16, '#5a3a26')
        cv.line(x1 - 3, y + 3, x0 + 1, y + 16, '#4a2e20')
    # piers
    for (x0, x1, y, posts) in ((40, 150, PIER1_Y, 46), (44, 124, PIER2_Y, 30)):
        cv.rect(x0, y, x1 - x0, 4, '#8a5a36')
        cv.hline(x0, x1 - 1, y, '#c48a52')
        cv.hline(x0, x1 - 1, y + 3, '#4e3020')
        for xx in range(x0, x1, 5):
            cv.vline(xx, y + 1, y + 2, '#6e4628')
        for xx in range(x0 + 14, x1, 22):
            cv.rect(xx, y + 4, 3, posts, '#6e4628')
            cv.vline(xx, y + 4, y + 3 + posts, '#8a5a36')
            cv.line(xx + 2, y + 4 + posts // 2, xx - 12, y + 4, '#5a3a26')
        cv.line(x1 - 4, y + 4, 40, y + 4 + posts, '#5a3a26')
        cv.line(x1 - 4, y + 5, 40, y + 5 + posts, '#4a2e20')
        # railing posts + bollards
        for xx in range(x0 + 4, x1 - 2, 8):
            cv.vline(xx, y - 5, y - 1, '#6e4628')
        cv.hline(x0 + 4, x1 - 6, y - 5, '#8a5a36')
    for bx in (146, 120):
        cv.rect(bx, PIER1_Y - 4, 3, 4, '#5c5656')
        cv.hline(bx - 1, bx + 3, PIER1_Y - 4, '#8b8d92')
    # cargo on pier 1
    crate(cv, 60, PIER1_Y, 12, 10)
    crate(cv, 72, PIER1_Y, 10, 8, light=True)
    crate(cv, 63, PIER1_Y - 10, 8, 7, light=True)
    barrel(cv, 86, PIER1_Y, 8, 11)
    for k in range(3):
        cv.ellipse(102, PIER1_Y - 2 - k, 4 - k, 1, '#c9a06a')
    cv.px(101, PIER1_Y - 5, '#e8c890')
    # lamp posts
    for (lx, y) in ((112, PIER1_Y), (92, PIER2_Y)):
        cv.vline(lx, y - 20, y - 1, '#4a3a34')
        cv.hline(lx, lx + 4, y - 20, '#4a3a34')
        anim.setdefault('lamps', []).append((lx + 4, y - 19))
    # little dock crane on pier 2
    cv.rect(108, PIER2_Y - 26, 2, 26, '#6a4630')
    cv.hline(100, 128, PIER2_Y - 26, '#7a4f30')
    cv.hline(100, 128, PIER2_Y - 25, '#5a3a24')
    cv.line(109, PIER2_Y - 31, 127, PIER2_Y - 26, '#5a3a24')
    cv.line(109, PIER2_Y - 31, 101, PIER2_Y - 26, '#5a3a24')
    cv.rect(99, PIER2_Y - 24, 4, 4, '#4e4a4a')
    anim['hook'] = (125, PIER2_Y - 24)
    # signal mast with flags at the pier tip
    cv.vline(140, PIER1_Y - 30, PIER1_Y - 1, '#5a3a26')
    cv.hline(136, 144, PIER1_Y - 26, '#5a3a26')
    anim['mastflags'] = [(140, PIER1_Y - 30), (144, PIER1_Y - 26)]
    # windmill + chimneys on the pillar top
    anim['windmill'] = (36, -6, 7)
    anim['stacks'] = [(14, 4), (66, 48)]
    for (sx, yt) in anim['stacks']:
        cv.rect(sx, yt, 3, 10, '#5c5656')
        cv.vline(sx, yt, yt + 9, '#7d7676')
        cv.hline(sx - 1, sx + 3, yt, '#3e3a3a')
    return to_layer(cv), anim


@lru_cache(maxsize=1)
def _fg_c():
    w = lwc(PAR_C['fg'])
    cv = new(w)
    # foreground pier corner (bottom left) with a big bollard and rope coil
    cv.rect(-2, 164, 120, 6, '#7a5236')
    cv.hline(-2, 117, 164, '#c48a52')
    cv.hline(-2, 117, 169, '#4e3020')
    for xx in range(0, 118, 7):
        cv.vline(xx, 165, 168, '#5a3a24')
    for xx in (10, 52, 96):
        cv.rect(xx, 170, 5, 10, '#5a3a26')
        cv.vline(xx, 170, 179, '#8a5a36')
    cv.rect(70, 154, 7, 10, '#5c5656')
    cv.rect(68, 152, 11, 3, '#7d7676')
    cv.hline(68, 78, 152, '#b0aca6')
    for k in range(4):
        cv.ellipse(36, 161 - k * 2, 9 - k, 2, '#c9a06a' if k % 2 else '#a88456')
    cv.hline(28, 44, 161, '#8a6a44')
    cv.line(45, 160, 70, 156, '#a88456')
    # hanging rope from top
    return to_layer(cv)


def render_c(f):
    cam = cam_c(f)
    c = _sky_c().copy()
    st = _stratus_a()
    sx = -int((f * 0.12 + cam * 0.04 + 200) % 480)
    comp(c, st, sx, 0)
    comp(c, st, sx + 480, 0)
    comp(c, _peaks_c(), -int(round(cam * PAR_C['peaks'])))
    for lvl, sp in ((0, 0.08), (1, 0.2), (2, 0.4)):
        band = _cloud_band(lvl)
        ox = -int((cam * PAR_C['cloud%d' % lvl] - f * sp) % 480)
        oy = int(round(math.sin(f / 30.0 + lvl) * (lvl * 0.6)))
        comp(c, band, ox, oy)
        comp(c, band, ox + 480, oy)
        if lvl == 0:
            shimmer(c, f, HORIZON_C - 2, HORIZON_C + 10, amp=0.7)
    do = -int(round(cam * PAR_C['dock']))
    dl, anim = _dock_c()
    # the airship: moored until 216, casts off, drifts right + up
    t = seg(f, AIRSHIP, C1)
    sx_ = SHIP_X0 + int(round(110 * ease_in(t) ** 1.0 * 0.6 + 40 * t))
    sy_ = SHIP_Y0 - int(round(14 * ease_in_out(t)))
    bob = int(round(math.sin((f - AIRSHIP) * 2 * math.pi / 60)))
    ship, an = _ship(1)
    # the explorer's balloon follows
    bt = seg(f, AIRSHIP + 10, C1)
    bx = int(round(lerp(98, 196, ease_in_out(bt)))) + do + int(round(cam * 0.0))
    by = int(round(lerp(PIER1_Y - 1, 66, ease_out(bt))))
    comp(c, dl, do)
    s0, _ = _ship(0)
    c.blit(s0, 122 + do, PIER2_Y - 22 + int(round(math.sin(f * 2 * math.pi / 80))))
    c.line(126 + do, PIER2_Y - 6, 120 + do, PIER2_Y - 2, '#5a4034')
    for (wx, wy, ww, wh, sd) in anim.get('windows', []):
        if hash01(sd, f // 9) < 0.12:
            c.rect(wx + do, wy, ww, wh, '#e8a040')
    wx, wy, wr = anim['windmill']
    for (sxk, yt) in anim['stacks']:
        draw_smoke(c, f, sxk + 1 + do, yt - 2, n=5, period=56, rise=26, drift=18, seed=sxk)
    hx, hy = anim['hook']
    draw_crane_hook(c, f, hx + do, hy, 12, seed=1)
    for (lx, ly) in anim['lamps']:
        x = draw_lantern(c, lx + do, ly, '#ffc861', f, lx, big=True)
        fx.glow(c, x, ly + 3, 9, '#ffc861', strength=0.25, rings=3)
    for i, (fx_, fy_) in enumerate(anim['mastflags']):
        draw_pennant(c, f + i * 5, fx_ + do, fy_, cc=('#b8483a', '#4e8c84')[i])
    shx, shy = sx_ + do, sy_ + bob
    # mooring lines: taut before the hit, falling + slack after
    for k, (ax, ay, px, py) in enumerate(((an['stern'][0], an['stern'][1], 120, PIER1_Y - 4),
                                          (an['bow'][0], an['bow'][1], 147, PIER1_Y - 4))):
        x0, y0 = shx + ax, shy + ay
        if f < AIRSHIP + 1:
            c.line(x0, y0, px + do, py, '#5a4034')
        else:
            drop = min(10, (f - AIRSHIP) * 2)
            if f < AIRSHIP + 14:
                c.line(px + do, py, px + do + 3 - k * 6, py + drop, '#5a4034')
                c.line(x0, y0, x0 - 2, y0 + drop, '#5a4034')
            else:
                c.line(px + do, py, px + do + 1 - k * 2, py + 8, '#5a4034')
                c.line(x0, y0, x0 - 2 + int(math.sin(f / 6.0 + k) * 1.5), y0 + 9, '#5a4034')
    c.blit(ship, shx, shy)
    spin = clamp((f - AIRSHIP) / 14.0, 0, 1)
    for (px, py, r) in an['props']:
        draw_prop(c, shx + px, shy + py, r, f, spin=spin)
    if f >= AIRSHIP:
        for q in range(4):
            tq = ((f - AIRSHIP + q * 6) % 24) / 24.0
            ex_ = shx + 24 - int(tq * 16)
            ey_ = shy + 44 - int(tq * 3)
            if tq < 0.75 or (f + q) % 2:
                c.circle(ex_, ey_, 1 + int(tq * 2), ('#e8dccb', '#d8c8b4', '#c8b6a2')[min(2, int(tq * 3))])
    if AIRSHIP <= f < AIRSHIP + 16:          # cast-off steam burst
        k = (f - AIRSHIP) / 16.0
        for i in range(7):
            a_ = i * 0.9
            rr = 2 + int(k * 6)
            px_ = shx + 50 + int(math.cos(a_) * k * 22)
            py_ = shy + 46 + int(math.sin(a_) * k * 8) + int(k * 4)
            if (i + f) % 3 or k < 0.6:
                c.circle(px_, py_, rr, '#f4ecdc' if k < 0.5 else '#dccdb8')
    # gangplank retracted at the hit
    if f < AIRSHIP:
        c.line(150 + do, PIER1_Y, shx + 98, shy + an['deck'], '#a06a40')
    fl_x, fl_y = an['flag']
    c.vline(shx + fl_x, shy + fl_y - 5, shy + fl_y, '#4a3a34')
    draw_pennant(c, f, shx + fl_x, shy + fl_y - 5, cc='#f9c22b')
    draw_balloon(c, bx, by, f, size='m', flame=bt > 0 and bt < 0.5)
    if bt <= 0:
        c.line(bx + 2, by - 1, 104 + do, PIER1_Y - 1, '#5a4034')
    comp(c, _fg_c(), -int(round(cam * PAR_C['fg'])))
    fx.dust_motes(c, f, (120, 30, 190, 90), n=12, color='#fff0c0', seed=11)
    callout(c, 86 + do, PIER2_Y + 2, 'AIRSHIP DOCKS', f, 270, 342, side='down', length=5)
    callout(c, 250, 128, 'SEA OF CLOUDS', f, 288, 346, side='down', length=10)
    return c


def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < B0:
        c = render_a(f)
        if f < 80:
            iris(c, 1 - seg(f, 72, 79), 204, 97)
    elif f < C0:
        c = render_b(f)
    else:
        c = render_c(f)
    return c


SFX = [
    (80, 'wind', 0.45),          # desert wind as the diorama opens
    (99, 'creak', 0.3),          # windmills / rope bridges
    (126, 'clank', 0.28),        # crane hook swinging
    (150, 'steam', 0.22),        # smokestacks
    (162, 'wind', 0.3),
    (189, 'creak', 0.45),        # the bazaar sign at the end of its swing
    (198, 'clank', 0.25),        # junk pile
    (216, 'steam', 0.2),         # kettle on the lamp stall
    (225, 'creak', 0.4),         # sign swings back
    (243, 'creak', 0.3),
    (272, 'creak', 0.3),         # moored hull straining at its lines
    (288, 'steam', 0.6),         # AIRSHIP: cast-off steam burst (music plays horn + boom)
    (291, 'creak', 0.5),         # lines released
    (298, 'whoosh', 0.3),        # the explorer's balloon lifts off
    (306, 'wind', 0.4),          # high air over the Sea of Clouds
    (330, 'creak', 0.28),
    (340, 'wind', 0.3),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(180, 'zoom', {'cx': 179, 'cy': 72}), (270, 'glide', {'dir': -1})])

_render_cuts = render


def render(f):
    """Atlas card drawn over the shot changes so it stays steady through the zoom at 180."""
    c = _render_cuts(f)
    atlas_card(c, f, 100, 208, 'TRADE TOWN', ['RUST - SCRAP - AIRSHIPS', 'BEYOND EQUESTRIA'])
    return c


render.cuts = _render_cuts.cuts
render.__wrapped__ = _render_cuts

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
