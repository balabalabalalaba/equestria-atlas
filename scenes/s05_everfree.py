"""EVERFREE FOREST: the wild woods (scene frames; N = 360; HUD clock 09:30 -> 10:15). Mysterious.

Scene frames (film = scene + 72 for scene >= 72: the 2-bar map flight is added by long_flight).
Beat sheet (bar = 72, beat = 18; 3 shots of 1.25 / 1.25 / 1.5 bars, glide transitions at the cuts):
  0-71      World-map travel shot (wrapped by engine/regionfx.long_flight).
  72-161    REVEAL (72): iris opens 72-79 on the winding forest path. UNDER THE CANOPY: the camera cruises
            right along the path (0 -> 120 px, ~1.2 px/f). 6 layers: wild sky through canopy holes (clouds
            race both ways: the weather runs itself), hazy far trunks, mid gnarled trees, low mist, near
            twisted trees with knotted roots + curling branches, fern/vine foreground. Key light: morning sun
            shafts through the canopy gaps (upper-left), sweeping, dust motes. Glowing poison joke, mushrooms,
            a warning sign (icon only). Balloon peeks through a canopy hole.
            108 (beat): a self-made GUST: leaves whirl through, canopy + vines sway, mist speeds up.
            Callout EVERFREE FOREST 80-152. Atlas card 98-206.
  162-251   CUT (glide): ZECORA'S HUT in a clearing (slow pan 40 px): hollowed stump, mossy thatch cone,
            carved doorway with a bead curtain, round window, abstract masks (bands, diamonds, sun discs: no
            faces), potion shelves, herb bundles, swinging lantern, chimney smoke, cauldron on a fire with
            green steam + bubbles popping on the beats. Callouts ZECORA'S HUT 170-242, POISON JOKE 188-242.
  252-359   CUT (glide): FROGGY BOTTOM BOGG + the GORGE in one wide shot (cruise 64 -> 128 px): murky water
            with rippling reflections, mangrove roots, lily pads, bubbles, will-o'-wisps; the lip, a seep
            spring, the rickety ROPE BRIDGE swaying, rock spires in the mist, a cloud raining by itself.
            GLIMPSE (288, bar line): the mist curtain parts and the ruined Castle of the Two Sisters appears
            far beyond as a cold blue silhouette, with a glint. Balloon drifts above the gorge.
            Callouts FROGGY BOTTOM BOGG 262-334, ROPE BRIDGE 280-346. Toast from 302 (top kept clear).
Ambient: racing clouds, sweeping shafts + motes, drifting mist, swaying vines, falling leaves, poison-joke glow,
         lantern/window flicker, cauldron bubbles + steam + fire, bead curtain, wisps, bog bubbles + ripples,
         bridge sway, seep, rain cloud, the balloon.
"""
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.shotfx import cruise
from engine.loc_worldmap import travel_frame

ID = 'everfree'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
BT = FRAMES_PER_BEAT
GLIMPSE = 288                 # scene frame: bar-5 downbeat
CUT1, CUT2 = 162, 252         # path -> hut -> bog/gorge (glide transitions)
GUST = 108

# ------------------------------------------------------------------ palette
AIR = [mix('slate', 'deep_teal', 0.3), mix('sage', 'slate', 0.45), mix('sage', 'pale_sage', 0.25),
       mix('pale_sage', 'mist', 0.15)]                 # under-canopy air: dark top -> misty low glow
FOG = mix(mix('sage', 'pale_sage', 0.35), 'teal', 0.12)   # haze target for far layers
FOG2 = mix('sage', 'slate', 0.3)
SKY = [mix('sky', 'teal', 0.22), mix('sky', 'light_sky', 0.45), mix('light_sky', 'mint_cream', 0.35),
       mix('mint_cream', 'pale_sage', 0.25)]
BARK = (mix('ink', 'eggplant', 0.4), mix('eggplant', 'slate', 0.4), mix('plum_brown', 'slate', 0.3),
        mix('mauve', 'sage', 0.45), mix('pale_sage', 'mauve', 0.35))
LEAF = (mix('ink', 'deep_teal', 0.4), mix('forest', 'deep_teal', 0.6), mix('forest', 'green', 0.2),
        mix('green', 'forest', 0.25), mix('green', 'leaf', 0.35))
PLUM_LEAF = (mix('ink', 'eggplant', 0.5), mix('eggplant', 'slate', 0.25), mix('berry', 'eggplant', 0.4),
             mix('berry', 'mauve', 0.35), mix('mauve', 'rose', 0.3))
GROUND = (mix('ink', 'dark_olive', 0.35), mix('dark_olive', 'eggplant', 0.35), mix('olive', 'slate', 0.5),
          mix('moss', 'sage', 0.5), mix('moss', 'pale_leaf', 0.3))
PATH = (mix('plum_brown', 'eggplant', 0.45), mix('plum_brown', 'tan', 0.4), mix('tan', 'plum_brown', 0.3),
        mix('khaki', 'tan', 0.4))
ROCK = (mix('ink', 'slate', 0.3), mix('slate', 'indigo', 0.3), mix('grey', 'sage', 0.4),
        mix('stone', 'pale_sage', 0.4), mix('mist', 'white', 0.2))
FG = (col('ink'), mix('ink', 'deep_teal', 0.35), mix('eggplant', 'deep_teal', 0.45), mix('forest', 'deep_teal', 0.4))
PJ = ('crystal_dk', 'crystal', 'light_sky', 'ice')    # poison joke ramp
WOOD = (mix('ink', 'wine', 0.4), 'rust', 'copper', 'clay')


def haze(ramp, a, to=FOG):
    return tuple(mix(c, to, a) for c in ramp)


# ------------------------------------------------------------------ low-level helpers
KEY = (1, 2, 3)
_KEYA = np.array(KEY, np.uint8)
_YY, _XX = np.mgrid[0:H, 0:W]
_YYf, _XXf = _YY.astype(np.float32), _XX.astype(np.float32)


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


def curl(x, y, ang, L, k, step=0.7):
    """Path that curls up at its end (gnarled branch / vine tendril). k = total turn (radians), signed."""
    pts, s = [], 0.0
    while s < L:
        pts.append((x, y))
        a = ang + k * (s / L) ** 2.2
        x += math.cos(a) * step
        y += math.sin(a) * step
        s += step
    return pts


def tube(cv, pts, r0, r1, ramp, light=True):
    """Shaded round limb along pts, radius r0 -> r1. ramp: (outline, dark, mid, light, ...)."""
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
            disc(cv, x - 1, y - 1, r - 2, md)
        elif r == 2:
            disc(cv, x, y - 1, 0, md)
    if light:
        for (x, y), r in zip(pts, rs):
            if r >= 4:
                disc(cv, x - r * 0.5, y - r * 0.5, max(0, r // 3 - 1), lt)


def leafball(cv, x, y, r, ramp):
    out, dk, md, lt, hi = ramp
    disc(cv, x, y, r, out)
    disc(cv, x, y, r - 1, dk)
    if r >= 3:
        disc(cv, x - 1, y - 1, r - 2, md)
    if r >= 5:
        disc(cv, x - r * 0.32, y - r * 0.36, int(r * 0.45), lt)
    if r >= 8:
        disc(cv, x - r * 0.45, y - r * 0.5, max(0, int(r * 0.16)), hi)


def canopy(cv, cx, cy, rx, ry, n, rmin, rmax, ramp, seed):
    r = rng(seed)
    balls = [(cx + r.uniform(-1, 1) * rx, cy + r.uniform(-1, 1) * ry, int(r.integers(rmin, rmax + 1)))
             for _ in range(n)]
    balls.sort(key=lambda b: b[1])
    for x, y, rr in balls:
        leafball(cv, x, y, rr, ramp)


def speckle(cv, src, dst, density, seed, horiz=True):
    """Texture pass: some pixels of colour src become dst (2-px marks)."""
    a = cv.a
    m = np.all(a == np.array(col(src), np.uint8), axis=2)
    rnd = rng(seed).random(a.shape[:2]) < density
    if horiz:
        rnd |= np.roll(rnd, 1, axis=1)
    a[m & rnd] = col(dst)


def trunk(cv, cx, base, top, w0, w1, ramp, seed, wob=5.0, lean=0.0, flare=5.0, turn=60, grooves=3,
          knots=2, streaks=0):
    """Twisted, knotted trunk from y=base up to y=top. Returns [(y, xc, hw)] spine."""
    r = rng(seed)
    ph = r.uniform(0, 6.283, 6)
    Ht = max(1, base - top)
    kn = [(r.uniform(0.12, 0.85), r.uniform(1.0, 2.6)) for _ in range(knots)]
    a = cv.a
    Hh, Ww = a.shape[:2]
    out, dk, md, lt, hi = [np.array(col(c), np.uint8) for c in ramp]
    spine = []
    prev = {}
    for y in range(top, base + 1):
        t = (base - y) / Ht
        xc = (cx + lean * t * Ht + wob * (math.sin(t * 2.7 + ph[0]) * 0.8 + math.sin(t * 6.9 + ph[1]) * 0.35)
              * min(1.0, t * 2.5))
        hw = w1 + (w0 - w1) * (1 - t) ** 1.3 + flare * max(0.0, 1 - (base - y) / 12.0) ** 2
        for tk, amp in kn:
            hw += amp * math.exp(-(((t - tk) * Ht) / 4.0) ** 2)
        spine.append((y, xc, hw))
        if not (0 <= y < Hh):
            continue
        x0 = int(math.floor(xc - hw + 0.5))
        x1 = int(math.floor(xc + hw - 0.5))
        lo, hi_ = max(0, x0), min(Ww, x1 + 1)
        if hi_ <= lo:
            continue
        xs = np.arange(lo, hi_)
        u = (xs + 0.5 - xc) / max(hw, 0.5)
        row = np.empty((len(xs), 3), np.uint8)
        row[:] = md
        row[u > 0.32] = dk
        row[u < -0.42] = lt
        if hash01(seed, y // 4) > 0.35:
            row[(u > -0.74) & (u < -0.6)] = hi
        # near-vertical bark streaks
        for k in range(streaks):
            uk = -0.8 + 1.6 * (k + 0.5) / streaks + 0.05 * math.sin(y / 7.0 + k * 1.7)
            xk = int(round(xc + uk * hw)) - lo
            if 0 <= xk < len(xs) and hash01(seed, k, y // 9) > 0.2:
                row[xk] = out if uk > 0.3 else dk
                if xk - 1 >= 0 and uk < 0.2:
                    row[xk - 1] = lt if uk < -0.2 else md
        # twisting grooves (helix), joined row to row
        for g in range(grooves):
            ang = 2 * math.pi * (y / turn) + g * 2 * math.pi / grooves + ph[2]
            if math.cos(ang) < -0.15:
                prev.pop(g, None)
                continue
            ug = math.sin(ang) * 0.86
            xg = int(round(xc + ug * hw))
            xa = prev.get(g, xg)
            for xx in range(min(xa, xg), max(xa, xg) + 1):
                i = xx - lo
                if 0 <= i < len(xs):
                    row[i] = out if ug > 0.25 else dk
            i = min(xa, xg) - 1 - lo
            if 0 <= i < len(xs) and ug < 0.15:
                row[i] = hi if ug < -0.4 else lt
            prev[g] = xg
        row[(u < -0.84) | (u > 0.84)] = out
        a[y, xs] = row
    # knots: ringed burls
    for tk, amp in kn:
        y = int(base - tk * Ht)
        sp = spine[min(len(spine) - 1, max(0, y - top))]
        kx = sp[1] + sp[2] * r.uniform(-0.45, 0.25)
        rr = 1 + int(amp)
        cv.ellipse(kx, y, rr + 1, rr, ramp[0])
        cv.ellipse(kx, y, rr, max(0.5, rr - 1), ramp[1])
        cv.px(int(kx) - 1, y - 1, ramp[3])
        cv.px(int(kx), y, ramp[0])
    return spine


def roots(cv, cx, base, hw, ramp, seed, n=4, L=22, r0=4, front=1):
    r = rng(seed)
    for i in range(n):
        s = -1 if i % 2 == 0 else 1
        u = r.uniform(0.3, 0.9)
        x0 = cx + s * hw * u
        y0 = base - r.uniform(3, 9)
        Li = L * r.uniform(0.6, 1.15)
        hump = r.uniform(2, 7)
        pts = bez([(x0, y0), (x0 + s * Li * 0.45, y0 - hump), (x0 + s * Li * 0.8, base - 2),
                   (x0 + s * Li, base + r.uniform(0, 3))])
        tube(cv, pts, r0 * r.uniform(0.7, 1.0), 1, ramp)
    for i in range(front):
        x0 = cx + r.uniform(-0.4, 0.4) * hw
        s = 1 if r.random() < 0.5 else -1
        pts = bez([(x0, base - 6), (x0 + s * 6, base - 2), (x0 + s * 10, base + 4)])
        tube(cv, pts, r0, 1, ramp)


def branch(cv, spine, t, side, L, k, r0, ramp, top):
    i = int(len(spine) * (1 - t))
    i = max(0, min(len(spine) - 1, i))
    y, xc, hw = spine[i]
    ang = -math.pi / 2 + side * 1.0
    pts = curl(xc + side * hw * 0.5, y, ang, L, side * k)
    tube(cv, pts, r0, 1, ramp)
    return pts


@lru_cache(maxsize=None)
def _noise(w, h, seed, sy, sx):
    a = rng(seed).random((h, w))
    g = gaussian_filter(a, (sy, sx), mode='wrap')
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return g.astype(np.float32)


# ------------------------------------------------------------------ small props (build-time)
def fern(cv, x, y, h, side, ramp, seed):
    """Arching fern frond rooted at (x, y)."""
    pts = curl(x, y, -math.pi / 2 + side * 0.35, h, side * 1.9, step=1.0)
    for j, (px, py) in enumerate(pts):
        cv.px(px, py, ramp[1])
        if j > 1 and j % 2 == 0 and j < len(pts) - 1:
            ln = max(1, int(3 * (1 - j / len(pts))) + 1)
            for q in range(1, ln + 1):
                cv.px(px - q, py + q * 0.5, ramp[2] if q < ln else ramp[1])
                cv.px(px + q, py + q * 0.5, ramp[3] if q == 1 else ramp[2])


def grass(cv, x, y, n, ramp, seed):
    r = rng(seed)
    for i in range(n):
        xx = x + int(r.integers(-3, 4))
        h = int(r.integers(2, 6))
        lean = int(r.integers(-1, 2))
        cv.line(xx, y, xx + lean, y - h, ramp[2] if i % 2 else ramp[3])


def poison_joke(cv, x, y, n, seed):
    """Cluster of glowing blue poison-joke flowers on stems; (x, y) = ground centre."""
    r = rng(seed)
    for i in range(n):
        fx_ = x + int(r.integers(-n * 2, n * 2 + 1))
        h = int(r.integers(3, 9))
        cv.vline(fx_, y - h, y, mix('green', 'deep_teal', 0.3))
        if i % 2:
            cv.px(fx_ + 1, y - h // 2, 'mint')
        else:
            cv.px(fx_ - 1, y - h // 2 - 1, 'mint')
        cy = y - h - 1
        # 5-petal bloom
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)):
            cv.px(fx_ + dx, cy + dy, PJ[1])
        cv.px(fx_ - 1, cy - 1, PJ[2])
        cv.px(fx_ + 1, cy + 1, PJ[0])
        cv.px(fx_, cy, 'ice' if i % 3 else 'white')
        cv.px(fx_, cy + 1, PJ[0])


def mushroom(cv, x, y, s, cap='raspberry', spots='cotton'):
    """Toadstool, (x, y) = base centre; s = cap half-width (2..5)."""
    sh = max(2, s + 1)
    cv.rect(x - max(1, s // 2), y - sh, max(2, s), sh, 'cream')
    cv.vline(x + max(1, s // 2) - 1, y - sh, y - 1, 'tan')
    cy = y - sh
    cv.ellipse(x, cy, s + 0.5, max(1, s * 0.6), shade(cap, 0.6))
    cv.ellipse(x - 0.3, cy - 0.5, s - 0.5, max(1, s * 0.6) - 0.5, cap)
    cv.hline(x - s, x + s, cy + 1, shade(cap, 0.55))
    if s >= 3:
        cv.px(x - s // 2, cy - 1, spots)
        cv.px(x + s // 2, cy, spots)
        cv.px(x, cy - max(1, s // 2), mix(cap, 'white', 0.5))


def glowshroom(cv, x, y):
    cv.vline(x, y - 2, y, 'mint_cream')
    cv.hline(x - 1, x + 1, y - 3, 'turquoise')
    cv.px(x, y - 4, 'ice')


def stone(cv, x, y, w, h, ramp=ROCK):
    cv.round_rect(x, y - h, w, h, ramp[0], r=1)
    cv.round_rect(x + 1, y - h + 1, w - 2, h - 2, ramp[2], r=1)
    cv.hline(x + 1, x + w - 3, y - h + 1, ramp[3])
    cv.hline(x + 2, x + w - 2, y - 2, ramp[1])
    if w > 5:
        cv.px(x + 2, y - h, mix(GROUND[3], 'moss', 0.3))


def bracket_fungus(cv, x, y, w, side=1):
    """Shelf fungus stuck to a trunk side at (x, y)."""
    for i in range(2):
        yy = y + i * 4
        ww = w - i * 2
        for k in range(ww):
            hgt = int(round(2.4 * math.sqrt(max(0.0, 1 - (k / ww) ** 2))))
            xx = x + side * k
            cv.vline(xx, yy - hgt, yy, 'clay')
            cv.px(xx, yy - hgt, 'sand' if k < ww - 1 else 'clay')
            cv.px(xx, yy, mix('rust', 'wine', 0.4))
        cv.px(x + side, yy - 2, 'cream')


def rockfill(cv, mask, ramp, seed, bw=9, bh=5, x0=0, y0=0, moss=True):
    """Fill a bool mask with irregular stacked rock slabs lit from the upper-left.
    ramp: (outline, dark, mid, light, hi)."""
    h, w = mask.shape
    r = rng(seed)
    lab = np.full((h, w), -1, np.int32)
    k = 0
    by = 0
    while by < h:
        bx = -int(r.integers(0, bw))
        while bx < w:
            ww = max(3, bw + int(r.integers(-bw // 2, bw // 2 + 3)))
            hh = max(2, bh + int(r.integers(-bh // 2, bh // 2 + 2)))
            dy = int(r.integers(-1, 2))
            lab[max(0, by + dy):max(0, by + dy + hh), max(0, bx):max(0, bx + ww)] = k
            k += 1
            bx += ww
        by += bh
    lab[lab < 0] = k
    up = np.roll(lab, 1, 0) != lab
    left = np.roll(lab, 1, 1) != lab
    down = np.roll(lab, -1, 0) != lab
    right = np.roll(lab, -1, 1) != lab
    var = np.array([hash01(seed, i) for i in range(k + 1)])[lab]
    out = np.empty((h, w, 3), np.uint8)
    out[:] = col(ramp[2])
    out[var < 0.35] = mix(ramp[2], ramp[1], 0.55)
    out[var > 0.85] = mix(ramp[2], ramp[3], 0.4)
    out[up | left] = col(ramp[3])
    out[up & left] = col(ramp[4])
    out[right] = col(ramp[1])
    out[down] = col(ramp[0])
    if moss:
        mz = (var > 0.55) & up & (np.array([hash01(seed, i, 3) for i in range(k + 1)])[lab] > 0.5)
        out[mz] = mix(GROUND[3], ramp[3], 0.35)
    pm = np.pad(mask, 1)
    edge = mask & ~(pm[:-2, 1:-1] & pm[2:, 1:-1] & pm[1:-1, :-2] & pm[1:-1, 2:])
    out[edge] = col(ramp[0])
    a = cv.a[y0:y0 + h, x0:x0 + w]
    a[mask] = out[mask]


# ------------------------------------------------------------------ clouds (wild sky)
@lru_cache(maxsize=None)
def cloud_sprite(w, seed, dark=0.0):
    h = max(8, w // 3 + 3)
    cv = _new(w + 6, h + 4)
    r = rng(seed)
    n = max(3, w // 7)
    lt = mix('white', 'mint_cream', 0.3 + dark * 0.3)
    md = mix('cloud', 'pale_sage', 0.25 + dark * 0.4)
    sh = mix('cloud_dk', 'sage', 0.35 + dark * 0.3)
    ol = mix('cloud_dk', 'slate', 0.35 + dark * 0.3)
    base = h + 1
    balls = []
    for i in range(n):
        u = (i + 0.5) / n
        rr = int(max(2, h * 0.5 * (0.5 + 0.5 * math.sin(u * math.pi)) * r.uniform(0.8, 1.15)))
        balls.append((3 + u * w, base - rr, rr))
    for x, y, rr in balls:
        disc(cv, x, y, rr, ol)
    for x, y, rr in balls:
        disc(cv, x, y, rr - 1, sh)
    for x, y, rr in balls:
        disc(cv, x - 1, y - 1, rr - 2, md)
    for x, y, rr in balls:
        if rr >= 3:
            disc(cv, x - rr * 0.3, y - rr * 0.4, max(0, rr // 2 - 1), lt)
    cv.a[base + 1:] = _KEYA
    cv.hline(3, w + 2, base, ol)
    m = np.all(cv.a[base - 1] != _KEYA, axis=1)
    cv.a[base - 1][m] = col(sh)
    return _rgba(cv)


def wild_clouds(c, f, specs, y_lo=0):
    """specs: [(x0, y, w, seed, speed, dark)]; clouds wrap around a 420-px loop."""
    for x0, y, w, seed, speed, dark in specs:
        spr = cloud_sprite(w, seed, dark)
        span = W + spr.shape[1] + 60
        x = int((x0 + f * speed) % span) - spr.shape[1] - 30
        c.blit(spr, x, y + y_lo)


# ------------------------------------------------------------------ per-frame light/air
def ray(c, xt, yt, yb, wt, wb, slope, color, strength, fade=0.6):
    y0, y1 = max(0, int(yt)), min(H, int(yb))
    if y1 <= y0:
        return
    xmin = int(min(xt - wt, xt + (yb - yt) * slope - wb)) - 1
    xmax = int(max(xt + wt, xt + (yb - yt) * slope + wb)) + 2
    x0, x1 = max(0, xmin), min(W, xmax)
    if x1 <= x0:
        return
    yy = _YYf[y0:y1, x0:x1]
    xx = _XXf[y0:y1, x0:x1]
    t = (yy - yt) / (yb - yt)
    xc = xt + (yy - yt) * slope
    hw = (wt + (wb - wt) * t) * 0.5
    d = np.abs(xx + 0.5 - xc) / hw
    inten = np.clip(1 - d, 0, 1) ** 0.55 * (1 - fade * t)
    q = np.clip(np.floor(inten * 3 + (bayer(y1 - y0, x1 - x0, x0, y0) - 0.5) * 0.95) / 3, 0, 1)
    reg = c.a[y0:y1, x0:x1].astype(np.float32)
    reg += np.array(col(color), np.float32) * (q * strength)[..., None]
    c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def ray_motes(c, f, xt, yt, yb, slope, wb, n, seed, color='butter'):
    for i in range(n):
        t = (hash01(seed, i, 1) + f * 0.0025 * (0.5 + hash01(seed, i, 2))) % 1.0
        u = (hash01(seed, i, 3) - 0.5) * 0.8 + 0.12 * math.sin(f / 30 + i)
        y = yt + t * (yb - yt)
        w = 4 + (wb - 4) * t
        x = xt + (y - yt) * slope + u * w
        if hash01(seed, i, f // 16) > 0.2:
            c.px(int(x), int(y), color if t < 0.7 else mix(color, 'pale_leaf', 0.5))


def mist(c, f, y0, y1, color, speed, seed, dens=1.0, a1=0.28, a2=0.5, off=0, hole=None):
    y0, y1 = max(0, y0), min(H, y1)
    h = y1 - y0
    if h <= 1:
        return
    nz = _noise(256, h, seed, max(1.0, h / 5.0), 11.0)
    xs = (np.arange(W) + int(off + f * speed)) % 256
    d = nz[:, xs]
    prof = np.sin(np.pi * (np.arange(h) + 0.5) / h) ** 0.7
    L = np.clip((d - 0.28) / 0.5, 0, 1) * prof[:, None] * dens
    if hole is not None:
        L = L * hole
    thr = bayer(h, W, 0, y0)
    m1 = thr < L
    m2 = thr < L - 0.45
    reg = c.a[y0:y1].astype(np.float32)
    cc = np.array(col(color), np.float32)
    reg[m1] += (cc - reg[m1]) * a1
    reg[m2] += (cc - reg[m2]) * a2
    c.a[y0:y1] = reg.astype(np.uint8)


def add_glow(c, x, y, r, color, k):
    fx.glow(c, int(x), int(y), int(r), color, strength=k, rings=3)


def leaf_px(c, x, y, f, i, colr):
    ph = (f // 3 + i) % 4
    x, y = int(x), int(y)
    c.px(x, y, colr)
    if ph == 0:
        c.px(x + 1, y, shade(colr, 0.75))
    elif ph == 1:
        c.px(x + 1, y + 1, shade(colr, 0.75))
    elif ph == 2:
        c.px(x, y + 1, shade(colr, 0.75))
    else:
        c.px(x - 1, y + 1, shade(colr, 0.75))


LEAF_COLS = ('leaf', 'green', 'pale_leaf', 'moss', 'clay', 'mint', 'sand')
_LEAF_SHAPES = (((0, 0), (1, 0), (2, -1)), ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 2)),
                ((0, 0), (1, -1), (1, 0)))


def big_leaf(c, x, y, f, i, colr):
    sh = _LEAF_SHAPES[(f // 3 + i) % 4]
    x, y = int(x), int(y)
    cs = (mix(colr, 'white', 0.3), colr, shade(colr, 0.7))
    for (dx, dy), cc in zip(sh, cs):
        c.px(x + dx, y + dy, cc)


def falling_leaves(c, f, n, seed, x0=0, x1=W, y0=0, y1=H, wind=0.0):
    for i in range(n):
        period = 150 + int(hash01(seed, i) * 120)
        t = ((f + hash01(seed, i, 1) * period) % period) / period
        x = x0 + hash01(seed, i, 2) * (x1 - x0) + math.sin(t * 9 + i) * 6 + wind * t * 40
        y = y0 + t * (y1 - y0)
        leaf_px(c, x % W, y, f, i, LEAF_COLS[i % len(LEAF_COLS)])


GUST_COLS = ('leaf', 'pale_leaf', 'lime', 'sand', 'clay', 'mint', 'leaf', 'gold')


def gust_leaves(c, f, t0, n=56, seed=7):
    """A self-made gust: leaves spiral in from the left and blow across in a whirling stream."""
    for i in range(n):
        s = t0 + i * 0.55
        if f < s:
            continue
        age = f - s
        sp = 3.6 + hash01(seed, i) * 2.6
        R = 8 + hash01(seed, i, 1) * 20
        w = 0.16 + hash01(seed, i, 2) * 0.12
        cy = 56 + hash01(seed, i, 3) * 80
        x = -12 + age * sp + R * math.cos(age * w + i)
        y = cy + R * math.sin(age * w + i) * 0.8 - age * 0.25
        if -4 < x < W + 4:
            big_leaf(c, x, y, f, i, GUST_COLS[i % len(GUST_COLS)])
            if i % 3 == 0:
                big_leaf(c, x + 1, y + 1, f + 3, i, GUST_COLS[(i + 3) % len(GUST_COLS)])


def vine(c, x, y0, L, f, seed, amp=1.0, ramp=LEAF):
    """Hanging vine with leaves, swaying; drawn per frame (screen coords)."""
    ph = hash01(seed) * 6.28
    px_prev = None
    for j in range(L):
        k = j / max(1, L - 1)
        dx = amp * math.sin(f * 0.045 + ph + j * 0.07) * k * k * 2.2
        xx = int(round(x + dx))
        yy = y0 + j
        c.px(xx, yy, ramp[1])
        if px_prev is not None and abs(px_prev - xx) > 1:
            c.px((px_prev + xx) // 2, yy, ramp[1])
        px_prev = xx
        if j % 4 == 2:
            sd = 1 if (j // 4) % 2 else -1
            c.px(xx + sd, yy, ramp[3])
            c.px(xx + sd * 2, yy + 1, ramp[2])
    c.px(px_prev, y0 + L, ramp[4])


# ======================================================================= SHOT A: under the canopy
A_PAN = 120
IRIS_A = (204, 156)                   # on the path, x matched to the map's dest_screen


def camA(f):
    return int(round(A_PAN * cruise(seg(f, 72, CUT1 + 8))))


@lru_cache(maxsize=None)
def bgA():
    c = Canvas(W, H, 'ink')
    c.gradient(0, 0, W, 30, SKY, soft=0.7)
    c.gradient(0, 26, W, 154, AIR, soft=0.8)
    return c


FAR_HOLES_A = [(150, 34, 16), (40, 18, 12), (300, 22, 13)]     # far-canopy sky holes (x, rx, ry)
MID_HOLES_A = [(176, 30), (50, 16), (336, 20)]                  # mid-canopy holes (x, half width)
GAPS_A = [(92, 30), (206, 38), (362, 30)]                        # near-canopy gaps: (centre x, width)


@lru_cache(maxsize=None)
def farA():
    w = 380
    cv = _new(w, H)
    ramp = haze(BARK, 0.66)
    lramp = haze(LEAF, 0.55, mix(FOG, 'slate', 0.3))
    for x in range(w):
        gy = 126 + int(2 * math.sin(x / 23.0))
        cv.vline(x, gy, H - 1, haze(GROUND, 0.6)[2])
        cv.px(x, gy, haze(GROUND, 0.6)[3])
    r = rng(11)
    for i, x in enumerate(range(8, w, 27)):
        x += int(r.integers(-7, 8))
        trunk(cv, x, 129, 10, 4.0, 2.5, ramp, 300 + i, wob=4, flare=3, grooves=1, knots=1)
    # dense far canopy band, then punch the sky holes and frill their rims
    for i, x in enumerate(range(-8, w + 8, 12)):
        canopy(cv, x, 12 + 6 * math.sin(x / 17.0), 8, 10, 4, 6, 10, lramp, 400 + i)
        canopy(cv, x + 6, 34 + 3 * math.sin(x / 11.0), 7, 4, 2, 5, 8, lramp, 450 + i)
    yy, xx = np.mgrid[0:H, 0:w]
    for hx, rx, ry in FAR_HOLES_A:
        m = ((xx - hx) / rx) ** 2 + ((yy - 12) / ry) ** 2 < 1
        cv.a[m] = _KEYA
    for j, (hx, rx, ry) in enumerate(FAR_HOLES_A):
        for k in range(10):
            a = k / 10 * 2 * math.pi
            leafball(cv, hx + math.cos(a) * rx, 12 + math.sin(a) * ry, 2 + (k % 3), lramp)
    speckle(cv, lramp[2], lramp[1], 0.08, 71)
    return _rgba(cv)


@lru_cache(maxsize=None)
def midA():
    w = 420
    cv = _new(w, H)
    ramp = haze(BARK, 0.4)
    lramp = haze(LEAF, 0.35, mix(FOG, 'slate', 0.4))
    pramp = haze(PLUM_LEAF, 0.35, mix(FOG, 'slate', 0.4))
    g = haze(GROUND, 0.35)
    p = haze(PATH, 0.3)
    for x in range(w):
        gy = 136 + int(round(2 * math.sin(x / 31.0 + 1)))
        cv.vline(x, gy, H - 1, g[2])
        cv.px(x, gy, g[3])
        cv.px(x, gy + 1, g[3] if x % 3 else g[2])
    for x in range(w):              # path snaking back into the woods
        yc = 142 + 3 * math.sin(x / 40.0)
        hw = 2.2 + 1.2 * math.sin(x / 17.0)
        cv.vline(x, int(yc - hw), int(yc + hw), p[1])
        cv.px(x, int(yc - hw), p[2])
    trees = [(18, 8.5, 3), (98, 7.0, 4), (238, 9.0, 5), (296, 7.5, 6), (392, 8.0, 7)]
    for x, w0, sd in trees:
        sp = trunk(cv, x, 140, -12, w0, w0 * 0.55, ramp, 500 + sd, wob=6, flare=4, grooves=2,
                   turn=48, knots=2)
        roots(cv, x, 140, w0, ramp, 520 + sd, n=4, L=14, r0=3, front=0)
        for j, side in enumerate((-1, 1)):
            if hash01(sd, j) < 0.75:
                pts = branch(cv, sp, 0.5 + 0.2 * hash01(sd, j, 2), side, 18 + 12 * hash01(sd, j, 3),
                             2.8, 3, ramp, -12)
                ex, ey = pts[-1]
                canopy(cv, ex - side * 2, ey - 3, 5, 3, 3, 3, 5, lramp, 540 + sd * 3 + j)
    for i, x in enumerate(range(-10, w + 10, 13)):
        canopy(cv, x, 8 + 5 * math.sin(x / 19.0), 8, 12, 4, 6, 12, pramp if i % 5 == 2 else lramp, 600 + i)
    yy, xx = np.mgrid[0:H, 0:w]
    for hx, hw in MID_HOLES_A:
        m = ((xx - hx) / hw) ** 2 + ((yy - 8) / 22.0) ** 2 < 1
        cv.a[m] = _KEYA
        for k in range(9):
            a = k / 9 * 2 * math.pi
            leafball(cv, hx + math.cos(a) * hw, 8 + math.sin(a) * 22, 3 + (k % 3), lramp)
    for i, x in enumerate(range(0, w, 6)):           # hanging moss under the canopy
        if any(abs(x - hx) < hw for hx, hw in MID_HOLES_A):
            continue
        L = int(4 + hash01(i, 77) * 16)
        y0 = 26 + int(5 * math.sin(x / 19.0))
        for j in range(L):
            if (j + i) % 3 != 2:
                cv.px(x + (j // 5) % 2, y0 + j, lramp[2] if j < L - 2 else lramp[3])
    for x in (60, 140, 200, 330, 372):
        poison_joke(cv, x, 140, 3, 900 + x)
    for x in (48, 190, 280):
        glowshroom(cv, x, 139)
        glowshroom(cv, x + 3, 140)
    speckle(cv, lramp[2], lramp[1], 0.08, 61)
    speckle(cv, lramp[3], lramp[4], 0.05, 62)
    return _rgba(cv)


def _groundA_y(x):
    return 148 + int(round(2 * math.sin(x / 40.0)))


def _pathA(x):
    return 161 + 5 * math.sin(x / 48.0 + 0.4), 6 + 1.5 * math.sin(x / 29.0)


PJ_A = [(110, 157, 7), (238, 152, 6), (318, 168, 7), (404, 156, 6)]
TREES_A = [(30, 16, 9.5, 1, 0.05), (150, 13, 8.0, 2, -0.08), (290, 17, 10.0, 3, 0.06), (432, 13, 8.0, 4, 0)]


@lru_cache(maxsize=None)
def nearA():
    w = 480
    cv = _new(w, H)
    # ground (mossy loam)
    for x in range(w):
        gy = _groundA_y(x)
        cv.vline(x, gy, H - 1, GROUND[1])
        cv.vline(x, gy, gy + 1, GROUND[3])
        cv.px(x, gy + 2, GROUND[2])
    nz = _noise(w, H, 31, 1.2, 2.5)
    gm = np.zeros((H, w), bool)
    for x in range(w):
        gm[_groundA_y(x) + 3:, x] = True
    cv.a[gm & (nz > 0.6)] = col(GROUND[2])
    cv.a[gm & (nz > 0.75)] = col(GROUND[3])
    cv.a[gm & (nz < 0.3)] = col(GROUND[0])
    # winding path with a grassy lip
    for x in range(w):
        yc, hw = _pathA(x)
        y0, y1 = int(round(yc - hw)), int(round(yc + hw))
        cv.vline(x, y0, y1, PATH[1])
        cv.vline(x, y0 + 1, y0 + 2, PATH[2])
        cv.px(x, y0, PATH[0])
        cv.px(x, y1, PATH[0])
        if hash01(x, 5) > 0.82:
            cv.px(x, y0 + 3 + int(hash01(x, 6) * max(1, y1 - y0 - 4)), PATH[3])
        if hash01(x, 7) > 0.88:
            cv.px(x, y0 + 4, PATH[0])
        if hash01(x, 8) > 0.5:
            cv.px(x, y0 - 1, GROUND[3])
            if hash01(x, 9) > 0.6:
                cv.px(x, y0, GROUND[3])
    for x in range(10, w, 21):
        yc, hw = _pathA(x)
        stone(cv, x + int(hash01(x) * 8), int(yc + hw + 3), 4 + int(hash01(x, 1) * 4), 3)
    # trees
    for x, w0, w1, sd, lean in TREES_A:
        base = _groundA_y(x) + 2
        roots(cv, x, base, w0, BARK, 700 + sd, n=6, L=30, r0=4, front=0)
        sp = trunk(cv, x, base, -20, w0, w1, BARK, 710 + sd, wob=7, lean=lean, flare=7, turn=58,
                   grooves=3, knots=3, streaks=2)
        roots(cv, x, base, w0 * 0.7, BARK, 720 + sd, n=2, L=16, r0=3, front=2)
        for j, side in enumerate((-1, 1)):
            t = 0.48 + 0.2 * j + 0.1 * hash01(sd, 9)
            pts = branch(cv, sp, t, side, 30 + 14 * hash01(sd, j), 3.6, 4, BARK, -20)
            ex, ey = pts[-1]
            if j == 0 or sd % 2:
                canopy(cv, ex, ey - 4, 7, 4, 5, 3, 6, LEAF, 730 + sd * 5 + j)
        k = len(sp) // 3
        bracket_fungus(cv, int(sp[k][1] + sp[k][2] - 2), sp[k][0], 6, 1)
        mushroom(cv, x - int(w0) - 6, base + 1, 3)
        mushroom(cv, x - int(w0) - 1, base + 2, 2, 'magenta')
        glowshroom(cv, x + int(w0) + 4, base + 1)
        glowshroom(cv, x + int(w0) + 7, base + 2)
    speckle(cv, LEAF[2], LEAF[1], 0.09, 63)
    speckle(cv, LEAF[3], LEAF[4], 0.06, 64)
    speckle(cv, PLUM_LEAF[2], PLUM_LEAF[1], 0.1, 66)
    speckle(cv, PLUM_LEAF[3], PLUM_LEAF[2], 0.08, 67)
    speckle(cv, BARK[2], BARK[1], 0.015, 65)
    # undergrowth
    for x in range(4, w, 15):
        grass(cv, x, _groundA_y(x) + 1, 4, GROUND, 940 + x)
    for x in (70, 128, 200, 262, 350, 400):
        gy = _groundA_y(x)
        fern(cv, x, gy + 2, 14, -1, LEAF, x)
        fern(cv, x + 2, gy + 2, 17, 1, LEAF, x + 1)
    for x, y, n in PJ_A:
        poison_joke(cv, x, y, n, 990 + x)
        poison_joke(cv, x + 9, y + 2, max(2, n - 3), 995 + x)
    # crooked warning sign (icon only): post + board with a hazard triangle
    sx, sy = 196, _groundA_y(196) + 6
    cv.line(sx, sy, sx - 1, sy - 26, WOOD[0])
    cv.line(sx + 1, sy, sx, sy - 26, WOOD[1])
    cv.poly([(sx - 9, sy - 30), (sx + 10, sy - 27), (sx + 9, sy - 18), (sx - 9, sy - 21)], WOOD[0])
    cv.poly([(sx - 8, sy - 29), (sx + 9, sy - 26), (sx + 8, sy - 19), (sx - 8, sy - 22)], WOOD[2])
    cv.line(sx - 7, sy - 23, sx + 7, sy - 20, WOOD[1])
    cv.tri((sx, sy - 28), (sx - 5, sy - 21), (sx + 5, sy - 20), 'gold')
    cv.vline(sx, sy - 26, sy - 24, WOOD[0])
    cv.px(sx, sy - 22, WOOD[0])
    cv.px(sx - 6, sy - 24, 'moss')
    cv.px(sx + 6, sy - 21, 'moss')
    return _rgba(cv)


@lru_cache(maxsize=None)
def canopyA():
    """The heavy near canopy band (own layer so the gust can sway it as a whole)."""
    w = 480
    cv = _new(w, 60)
    for i, x in enumerate(range(-12, w + 12, 11)):
        if any(abs(x - gx) < gw / 2 for gx, gw in GAPS_A):
            continue
        canopy(cv, x, 3 + 5 * math.sin(x / 23.0), 8, 9, 4, 7, 13, LEAF, 800 + i)
        if i % 9 == 4:
            canopy(cv, x + 3, 10 + 4 * math.sin(x / 13.0), 9, 6, 6, 5, 9, PLUM_LEAF, 830 + i)
    for gx, gw in GAPS_A:      # leafy rims on the gaps
        canopy(cv, gx - gw / 2 - 2, 16, 4, 7, 3, 3, 6, LEAF, 860 + gx)
        canopy(cv, gx + gw / 2 + 2, 14, 4, 7, 3, 3, 6, LEAF, 870 + gx)
    speckle(cv, LEAF[2], LEAF[1], 0.09, 863)
    speckle(cv, LEAF[3], LEAF[4], 0.06, 864)
    speckle(cv, PLUM_LEAF[2], PLUM_LEAF[1], 0.1, 866)
    speckle(cv, PLUM_LEAF[3], PLUM_LEAF[2], 0.08, 867)
    return _rgba(cv)


@lru_cache(maxsize=None)
def fgA():
    w = 540
    cv = _new(w, H)
    fg = (FG[0], FG[1], FG[2], mix('forest', 'deep_teal', 0.5))
    # big dark trunk on the left edge, rim-lit by the shafts
    for y in range(H):
        xc = 4 + 4 * math.sin(y / 30.0)
        hw = 14 + 3 * math.sin(y / 17.0) + max(0, (y - 150)) * 0.5
        cv.hline(int(xc - hw), int(xc + hw), y, fg[1])
        cv.px(int(xc + hw), y, fg[3])
        if y % 3 == 0:
            cv.px(int(xc + hw) - 2, y, fg[2])
    # fern silhouettes along the bottom
    for i, x in enumerate(range(40, w, 58)):
        x += int(hash01(i, 3) * 20)
        for k in range(5):
            pts = curl(x + k * 3, H + 2, -math.pi / 2 + (k - 2) * 0.45, 16 + (k * 3) % 8, (k - 2) * 0.5, 1.0)
            for j, (px, py) in enumerate(pts):
                cv.px(px, py, fg[2])
                if j % 2 == 0 and j > 2:
                    cv.px(px - 1, py + 1, fg[3])
                    cv.px(px + 1, py + 1, fg[2])
    return _rgba(cv)


def gust_streaks(c, f, t0, t1):
    """Pale curved air streaks that sell the self-made gust."""
    for i in range(6):
        s = t0 + i * 5
        age = f - s
        if age < 0 or f > t1:
            continue
        x = -40 + age * 9
        y0 = 50 + int(hash01(i, 21) * 80)
        L = 26
        for j in range(L):
            if (j + age) % 4 == 3:
                continue
            xx = int(x + j)
            yy = int(y0 + 5 * math.sin((xx + i * 30) / 22.0))
            if 0 <= xx < W and hash01(i, j, 5) > 0.15:
                c.px(xx, yy, mix('mint_cream', 'pale_sage', 0.3))


def render_A(f):
    cam = camA(f)
    c = bgA().copy()
    gust = smooth(seg(f, GUST, GUST + 10)) * (1 - smooth(seg(f, GUST + 24, GUST + 48)))
    wild_clouds(c, f, [(0, 0, 44, 1, 0.9, 0.0), (210, 8, 34, 2, 1.4, 0.1), (120, 18, 40, 3, -0.8, 0.2),
                       (330, -2, 30, 4, -1.2, 0.15)])
    # balloon peeking through a far-canopy hole
    draw_balloon(c, 150 - int(cam * 0.3) + int((f - 72) * 0.1), 30, f, 's')
    c.blit(farA(), -int(cam * 0.3), 0)
    mist(c, f, 92, 140, mix(AIR[3], 'mint_cream', 0.3), 0.25 + gust * 1.2, 5, dens=0.9, a1=0.22, a2=0.4)
    c.blit(midA(), -int(cam * 0.6), 0)
    c.blit(nearA(), -cam, 0)
    sway = int(round(gust * 2.4 * math.sin((f - GUST) / 5.5)))
    c.blit(canopyA(), -cam + sway, 0)
    # light shafts from the near-canopy gaps (sweeping)
    for i, (gx, gw) in enumerate(GAPS_A):
        sx = gx - cam
        if sx < -140 or sx > W + 20:
            continue
        slope = 0.4 + 0.12 * math.sin(f / 46.0 + i * 2.1) + gust * 0.12
        ray(c, sx - 2, 6, 170, gw * 0.6, gw * 1.6, slope, mix('butter', 'pale_leaf', 0.35), 0.3, fade=0.55)
        ray_motes(c, f, sx - 2, 14, 160, slope, gw * 1.3, 12, 40 + i)
    # vines hanging from the near canopy
    for i, vx in enumerate((48, 120, 176, 238, 318, 392, 450)):
        sx = vx - cam
        if -10 < sx < W + 10:
            vine(c, sx, 16 + int(hash01(i, 4) * 6), 22 + int(hash01(i, 5) * 34), f, 50 + i, amp=1.0 + gust * 3)
    # poison joke glow
    for i, (x, y, n) in enumerate(PJ_A):
        sx = x - cam + 4
        if -20 < sx < W + 20:
            k = 0.2 + 0.1 * math.sin(f / 9.0 + i * 1.7)
            add_glow(c, sx, y - 6, 14, 'baby_blue', k)
            fx.sparkle(c, sx + int(hash01(i, 1) * 10) - 5, y - 12, f, 'ice', period=30, phase=hash01(i, 2))
    mist(c, f, 140, 178, mix(AIR[3], 'mint_cream', 0.4), 0.5 + gust * 1.5, 6, dens=0.7, a1=0.2, a2=0.36)
    c.blit(fgA(), -int(cam * 1.35), 0)
    falling_leaves(c, f, 7, 3, wind=gust * 2)
    if f >= GUST:
        gust_streaks(c, f, GUST, GUST + 36)
        gust_leaves(c, f, GUST)
    callout(c, 150 - cam + 12, 90, 'EVERFREE FOREST', f, 80, 152, side='right', length=10)
    return c


# ======================================================================= SHOT B: Zecora's hut
HUT_X = 200          # stump centre (world, p=1)
HUT_TOP = 66         # stump rim
GB = 150             # ground line


def camB(f):
    return int(round(40 * cruise(seg(f, CUT1 - 8, CUT2 + 8))))


@lru_cache(maxsize=None)
def bgB():
    c = Canvas(W, H, 'ink')
    c.gradient(0, 0, W, 60, SKY, soft=0.7)
    c.gradient(0, 50, W, 130, [AIR[2], mix(AIR[2], AIR[1], 0.5), AIR[2], AIR[3]], soft=0.8)
    return c


@lru_cache(maxsize=None)
def farB():
    w = 340
    cv = _new(w, H)
    ramp = haze(BARK, 0.62, AIR[2])
    lramp = haze(LEAF, 0.55, AIR[2])
    for x in range(w):
        gy = 128 + int(2 * math.sin(x / 21.0))
        cv.vline(x, gy, H - 1, haze(GROUND, 0.6)[2])
        cv.px(x, gy, haze(GROUND, 0.6)[3])
    for i, x in enumerate(range(6, w, 26)):
        x += int(hash01(i, 8) * 10)
        inside = 104 < x < 236
        top = 46 if inside else 20
        trunk(cv, x, 130, top, 3.8, 2.4, ramp, 1300 + i, wob=4, flare=3, grooves=1, knots=1)
        if inside:
            canopy(cv, x, top - 2, 9, 6, 5, 4, 8, lramp, 1450 + i)
    for i, x in enumerate(range(0, w, 20)):
        if 110 < x < 230:
            continue
        canopy(cv, x, 22 + 6 * math.sin(x / 15.0), 12, 9, 4, 6, 11, lramp, 1400 + i)
    speckle(cv, lramp[2], lramp[1], 0.08, 1471)
    return _rgba(cv)


@lru_cache(maxsize=None)
def midB():
    w = 360
    cv = _new(w, H)
    ramp = haze(BARK, 0.42)
    lramp = haze(LEAF, 0.4)
    g = haze(GROUND, 0.38)
    for x in range(w):
        gy = 138 + int(round(2 * math.sin(x / 27.0)))
        cv.vline(x, gy, H - 1, g[2])
        cv.px(x, gy, g[3])
    for x, w0, sd, lean in ((20, 8.0, 1, 0.1), (88, 7.0, 2, 0.12), (268, 7.5, 3, -0.12), (340, 8.5, 4, -0.1)):
        sp = trunk(cv, x, 140, -12, w0, w0 * 0.55, ramp, 1500 + sd, wob=6, lean=lean, flare=4, grooves=2,
                   turn=48, knots=2)
        roots(cv, x, 140, w0, ramp, 1520 + sd, n=4, L=14, r0=3, front=0)
        side = 1 if x < 180 else -1
        pts = branch(cv, sp, 0.62, side, 34, 2.8, 3, ramp, -12)
        ex, ey = pts[-1]
        canopy(cv, ex, ey - 3, 7, 4, 4, 4, 7, lramp, 1540 + sd)
    # canopy arching around the clearing (open in the middle)
    for i, x in enumerate(range(-10, w + 10, 14)):
        d = abs(x - 190)
        if d < 70:
            continue
        canopy(cv, x, 6 + max(0, 30 - (d - 70) * 0.5), 10, 9, 4, 6, 12, lramp, 1600 + i)
    for x in (40, 120, 300):
        poison_joke(cv, x, 140, 3, 1900 + x)
        glowshroom(cv, x + 12, 140)
    speckle(cv, lramp[2], lramp[1], 0.08, 161)
    speckle(cv, lramp[3], lramp[4], 0.06, 162)
    return _rgba(cv)


# carved masks (abstract bands, diamonds, sun discs: no faces)
MASK_MAP = {'o': mix('ink', 'wine', 0.5), 's': 'sand', 'c': 'clay', 't': 'teal', 'w': 'cream', 'g': 'gold',
            'r': 'rust'}
MASK_A = Sprite.from_ascii('''
    ...g...
    ..ooo..
    .ossso.
    ossssso
    ottttto
    owswswo
    ottttto
    ossssso
    occccco
    .ocsco.
    .occco.
    ..oco..
    ...o...
    ..g.g..
''', MASK_MAP)
MASK_B = Sprite.from_ascii('''
    ....g....
    ....o....
    ...oco...
    ..ocwco..
    .ocwtwco.
    ocwtgtwco
    .ocwtwco.
    ..ocwco..
    ...oco...
    ....o....
    ...g.g...
''', MASK_MAP)
MASK_C = Sprite.from_ascii('''
    .....r.....
    .r..ooo..r.
    ...ossso...
    ..osctcso..
    r.ostwtso.r
    r.otwgwto.r
    r.ostwtso.r
    ..osctcso..
    ...ossso...
    .r..ooo..r.
    .....r.....
''', MASK_MAP)
LANTERN = Sprite.from_ascii('''
    .ooo.
    ..o..
    oCCCo
    oGYGo
    oGYGo
    oGYGo
    oCCCo
    .oCo.
''', {'o': mix('ink', 'rust', 0.3), 'C': 'copper', 'G': 'gold', 'Y': 'butter'})


def jar(cv, x, y, kind, liquid):
    """Potion jar standing on (x, y) (bottom-left)."""
    gl = mix('ink', 'teal', 0.35)
    lq, lq_sh = col(liquid), shade(liquid, 0.7)
    if kind == 0:      # round flask
        cv.circle(x + 2, y - 3, 2, gl)
        cv.circle(x + 2, y - 3, 1, lq)
        cv.px(x + 3, y - 2, lq_sh)
        cv.vline(x + 2, y - 7, y - 6, gl)
        cv.px(x + 2, y - 8, 'tan')
        cv.px(x + 1, y - 4, 'white')
    elif kind == 1:    # tall bottle
        cv.rect(x, y - 7, 4, 7, gl)
        cv.rect(x + 1, y - 5, 2, 4, lq)
        cv.px(x + 2, y - 2, lq_sh)
        cv.vline(x + 1, y - 9, y - 8, gl)
        cv.vline(x + 2, y - 9, y - 8, gl)
        cv.hline(x + 1, x + 2, y - 10, 'tan')
        cv.px(x + 1, y - 6, 'white')
    else:              # squat jar with lid
        cv.rect(x, y - 5, 5, 5, gl)
        cv.rect(x + 1, y - 4, 3, 3, lq)
        cv.px(x + 3, y - 2, lq_sh)
        cv.hline(x, x + 4, y - 6, 'copper')
        cv.px(x + 2, y - 7, 'rust')
        cv.px(x + 1, y - 4, 'white')


JARS_L = [(158, 0, 'magenta'), (164, 1, 'turquoise'), (170, 2, 'gold'), (176, 0, 'lavender')]
JARS_R = [(218, 1, 'leaf'), (224, 2, 'coral'), (231, 0, 'turquoise'), (237, 1, 'magic')]
CAUL_X, CAUL_Y = 306, 139        # cauldron centre (world)
WIN_B = (233, 102)               # round window centre
DOOR = (200, 128)                # door arch centre; door x 191..209, y 119..150
LANT_B = (140, 90)               # lantern hook
PJ_B = [(252, 5), (270, 3), (62, 3)]


@lru_cache(maxsize=None)
def hutB():
    w = 380
    cv = _new(w, H)
    # ---- ground: clearing (lighter, sun-lit moss)
    for x in range(w):
        gy = GB - 2 + int(round(1.5 * math.sin(x / 33.0)))
        cv.vline(x, gy, H - 1, GROUND[1])
        cv.vline(x, gy, gy + 1, GROUND[3])
    nz = _noise(w, H, 41, 1.1, 2.2)
    gm = np.zeros((H, w), bool)
    gm[GB + 1:, :] = True
    cv.a[gm & (nz > 0.6)] = col(GROUND[2])
    cv.a[gm & (nz < 0.28)] = col(GROUND[0])
    lit = np.zeros((H, w), bool)
    yy, xx = np.mgrid[0:H, 0:w]
    lit[((xx - 268) / 60.0) ** 2 + ((yy - 160) / 11.0) ** 2 < 1] = True
    cv.a[gm & lit & (nz > 0.45)] = col(GROUND[3])
    cv.a[gm & lit & (nz > 0.7)] = col(GROUND[4])
    # stepping stones to the door
    for i, (sx, sy) in enumerate(((196, 156), (186, 163), (172, 170), (154, 176))):
        cv.ellipse(sx, sy, 5, 2, ROCK[0])
        cv.ellipse(sx, sy - 0.5, 4, 1, ROCK[2])
        cv.hline(sx - 2, sx + 1, sy - 1, ROCK[3])
    # ---- framing trees (left / right)
    for x, w0, sd, side in ((14, 15, 1, 1), (378, 14, 2, -1)):
        roots(cv, x, GB, w0, BARK, 2000 + sd, n=6, L=28, r0=4, front=0)
        sp = trunk(cv, x, GB, -20, w0, 9, BARK, 2010 + sd, wob=6, lean=0.06 * side, flare=7, grooves=3,
                   knots=3, streaks=2)
        pts = branch(cv, sp, 0.6, side, 46, 3.6, 5, BARK, -20)
        ex, ey = pts[-1]
        canopy(cv, ex, ey - 5, 9, 5, 5, 5, 8, LEAF, 2030 + sd)
        canopy(cv, x + side * 10, 2, 30, 10, 9, 7, 13, LEAF, 2040 + sd)
    # ---- the stump hut
    roots(cv, HUT_X, GB, 50, BARK, 2100, n=8, L=34, r0=5, front=0)
    trunk(cv, HUT_X, GB, HUT_TOP, 51, 46, BARK, 2110, wob=2.0, flare=8, turn=400, grooves=0, knots=3,
          streaks=9)
    # jagged broken rim
    r = rng(2120)
    xs = np.linspace(HUT_X - 45, HUT_X + 45, 9)
    for i, x in enumerate(xs):
        h = int(r.integers(4, 13))
        cv.tri((x - 6, HUT_TOP + 2), (x + 6, HUT_TOP + 2), (x + r.integers(-2, 3), HUT_TOP - h), BARK[0])
        cv.tri((x - 4, HUT_TOP + 2), (x + 4, HUT_TOP + 2), (x + r.integers(-1, 2), HUT_TOP - h + 2),
               BARK[3] if x < HUT_X else BARK[2])
    # mossy thatch cone roof (bulging, tip curls right)
    cone = []
    for y in range(18, HUT_TOP + 7):
        t = (y - 18) / (HUT_TOP + 7 - 18)
        hw = 6 + 54 * (t ** 0.8) + 4 * math.sin(t * math.pi)
        xc = HUT_X + 6 * (1 - t) ** 2
        cone.append((y, xc, hw))
    th = (mix('ink', 'dark_olive', 0.5), 'dark_olive', 'olive', 'moss', mix('moss', 'pale_leaf', 0.5))
    for y, xc, hw in cone:
        x0, x1 = int(xc - hw), int(xc + hw)
        for x in range(x0, x1 + 1):
            u = (x - xc) / hw
            band = (y - 18) % 6
            cc = th[2]
            if u < -0.45:
                cc = th[3]
            if u > 0.4:
                cc = th[1]
            if band == 5:
                cc = th[1] if u < 0.4 else th[0]
            elif band == 0 and u < -0.2:
                cc = th[4]
            if abs(u) > 0.93:
                cc = th[0]
            if hash01(x, y, 3) > 0.93 and band not in (0, 5):
                cc = th[1]
            cv.px(x, y, cc)
    # fringe of the eave
    y_e = HUT_TOP + 7
    for x in range(HUT_X - 60, HUT_X + 62):
        L = 1 + int(hash01(x, 9) * 4)
        cv.vline(x, y_e, y_e + L, th[1] if x % 2 else th[2])
        cv.px(x, y_e + L, th[0])
    # tip curl + sapling sprouting on top
    tip = curl(HUT_X + 6, 19, -math.pi / 2, 14, 3.5, 0.7)
    tube(cv, tip, 2, 1, th)
    sap = curl(HUT_X - 18, 34, -math.pi / 2 - 0.4, 12, -2.5, 0.8)
    for px_, py_ in sap:
        cv.px(px_, py_, 'green')
    leafball(cv, sap[-1][0], sap[-1][1], 2, LEAF)
    leafball(cv, sap[len(sap) // 2][0] - 3, sap[len(sap) // 2][1], 2, LEAF)
    # chimney pipe
    cv.rect(HUT_X + 22, 30, 5, 14, ROCK[1])
    cv.rect(HUT_X + 21, 28, 7, 3, ROCK[0])
    cv.vline(HUT_X + 23, 31, 42, ROCK[3])
    # carved doorway: frame, warm interior, shelf silhouettes
    dx, dy = DOOR
    cv.circle(dx, dy, 12, BARK[0])
    cv.rect(dx - 12, dy, 25, GB - dy + 1, BARK[0])
    cv.circle(dx, dy, 11, mix('tan', 'khaki', 0.3))
    cv.rect(dx - 11, dy, 23, GB - dy + 1, mix('tan', 'khaki', 0.3))
    for k in range(-11, 12, 2):            # notch carving on the frame
        cv.px(dx + k, dy - int(math.sqrt(max(0, 121 - k * k))) + 1, WOOD[1])
    for yy_ in range(dy, GB, 3):
        cv.px(dx - 11, yy_, WOOD[1])
        cv.px(dx + 11, yy_ + 1, WOOD[1])
    cv.circle(dx, dy, 9, mix('brew', 'wine', 0.4))
    cv.rect(dx - 9, dy, 19, GB - dy + 1, mix('brew', 'wine', 0.4))
    cv.gradient(dx - 9, dy + 4, 19, GB - dy - 4, [mix('brew', 'wine', 0.4), mix('wine', 'rust', 0.4),
                                                  mix('rust', 'amber', 0.45)], soft=0.8)
    cv.hline(dx - 8, dx + 8, dy + 2, WOOD[0])       # interior shelf
    for k, cc in enumerate(('magenta', 'turquoise', 'gold', 'lavender', 'leaf')):
        cv.vline(dx - 7 + k * 3, dy - 1, dy + 1, cc)
    cv.hline(dx - 10, dx + 10, GB, mix('ink', 'rust', 0.3))   # threshold
    # round window
    wx, wy = WIN_B
    cv.circle(wx, wy, 8, BARK[0])
    cv.circle(wx, wy, 7, mix('tan', 'khaki', 0.3))
    cv.circle(wx, wy, 5, mix('rust', 'amber', 0.5))
    cv.circle(wx - 1, wy - 1, 3, 'amber')
    cv.vline(wx, wy - 5, wy + 5, WOOD[1])
    cv.hline(wx - 5, wx + 5, wy, WOOD[1])
    cv.hline(wx - 7, wx + 7, wy + 8, WOOD[2])
    jar(cv, wx - 4, wy + 8, 0, 'leaf')
    jar(cv, wx + 1, wy + 8, 1, 'magenta')
    # masks hanging on nails
    for spr, mx, my in ((MASK_A, 172, 103), (MASK_B, 200, 103), (MASK_C, 234, 121), (MASK_B, 158, 118)):
        cv.px(mx, my - 2, 'steel')
        cv.line(mx, my - 2, mx - spr.w // 2 + 1, my + 1, 'tan')
        cv.line(mx, my - 2, mx + spr.w // 2 - 1, my + 1, 'tan')
        spr.draw(cv, mx, my, anchor='top')
    # shelves with potion jars
    for x0, x1, y, jars in ((154, 182, 141, JARS_L), (215, 245, 141, JARS_R)):
        cv.hline(x0, x1, y, WOOD[2])
        cv.hline(x0, x1, y + 1, WOOD[0])
        for bx in (x0 + 2, x1 - 2):
            cv.line(bx, y + 2, bx + (2 if bx < (x0 + x1) // 2 else -2), y + 5, WOOD[1])
        for jx, kind, liq in jars:
            jar(cv, jx, y, kind, liq)
    # herb line under the eave
    cv.line(HUT_X - 46, HUT_TOP + 11, HUT_X + 46, HUT_TOP + 12, 'tan')
    for i, hx in enumerate((-40, -31, -22, 24, 33, 42)):
        x = HUT_X + hx
        y = HUT_TOP + 11
        cc = ('olive', 'moss', 'lavender', 'rust', 'sage', 'pale_leaf')[i]
        cv.px(x, y + 1, 'tan')
        for j in range(6 + i % 3):
            ww = 1 + j // 3
            cv.hline(x - ww, x + ww - 1, y + 2 + j, cc if (j + i) % 3 else shade(cc, 0.7))
    # lantern branch
    lx, ly = LANT_B
    br = curl(HUT_X - 44, 92, math.pi + 0.25, 18, 2.2, 0.7)
    tube(cv, br, 2, 1, BARK)
    # firewood stack (log ends)
    for i, (fx_, fy_) in enumerate(((134, 149), (140, 149), (146, 149), (137, 144), (143, 144), (140, 139))):
        cv.circle(fx_, fy_ - 2, 3, mix('ink', 'rust', 0.4))
        cv.circle(fx_, fy_ - 2, 2, mix('tan', 'cream', 0.25))
        cv.circle(fx_, fy_ - 2, 1, 'tan')
        cv.px(fx_, fy_ - 2, mix('tan', 'rust', 0.4))
        cv.px(fx_ - 2, fy_ - 3, 'cream')
    # cauldron fire pit + iron pot (liquid/bubbles/steam/flames drawn per frame)
    cx, cy = CAUL_X, CAUL_Y
    for k in range(-3, 4):
        stone(cv, cx + k * 5 - 2, GB + 4 + (abs(k) == 3), 5, 4)
    cv.line(cx - 8, GB + 1, cx + 7, GB - 3, WOOD[0])
    cv.line(cx - 7, GB - 3, cx + 8, GB + 1, WOOD[1])
    for lx_ in (cx - 9, cx + 9):
        cv.line(lx_, cy + 5, lx_ + (2 if lx_ < cx else -2), GB - 1, 'night')
    cv.ellipse(cx, cy, 14, 9, 'ink')
    cv.ellipse(cx, cy, 13, 8, 'night')
    cv.ellipse(cx - 2, cy - 1, 10, 6, mix('night', 'steel_dk', 0.5))
    cv.ellipse(cx - 6, cy - 3, 3, 2, 'steel_dk')
    cv.px(cx - 8, cy - 4, 'steel')
    cv.ellipse(cx, cy - 8, 14, 2, 'ink')
    cv.hline(cx - 13, cx + 13, cy - 9, 'steel_dk')
    cv.hline(cx - 11, cx - 4, cy - 10, 'steel')
    # ladle
    cv.line(cx + 6, cy - 9, cx + 14, cy - 22, WOOD[0])
    cv.line(cx + 7, cy - 9, cx + 15, cy - 22, WOOD[2])
    # poison joke patch (front right) + mushrooms + ferns
    for x, n in PJ_B:
        poison_joke(cv, x, GB + 6, n, 2200 + x)
        poison_joke(cv, x + 8, GB + 8, max(2, n - 2), 2250 + x)
    mushroom(cv, 238, GB + 2, 4)
    mushroom(cv, 245, GB + 3, 2, 'magenta')
    mushroom(cv, 120, GB, 3)
    for x in (230, 284, 160):
        glowshroom(cv, x, GB + 1)
    for x in (40, 80, 336, 356):
        fern(cv, x, GB + 2, 14, -1, LEAF, x)
        fern(cv, x + 2, GB + 2, 16, 1, LEAF, x + 7)
    for x in range(0, w, 13):
        grass(cv, x, GB + int(hash01(x, 2) * 3), 3, GROUND, 2300 + x)
    speckle(cv, LEAF[2], LEAF[1], 0.09, 263)
    speckle(cv, LEAF[3], LEAF[4], 0.07, 264)
    speckle(cv, BARK[2], BARK[1], 0.012, 265)
    return _rgba(cv)


@lru_cache(maxsize=None)
def fgB():
    w = 420
    cv = _new(w, H)
    for i, x in enumerate((-6, 70, 300, 380)):
        for k in range(6):
            pts = curl(x + k * 3, H + 2, -math.pi / 2 + (k - 2.5) * 0.42, 18 + (k * 5) % 9, (k - 2.5) * 0.5, 1.0)
            for j, (px, py) in enumerate(pts):
                cv.px(px, py, FG[1])
                if j % 2 == 0 and j > 2:
                    cv.px(px - 1, py + 1, FG[2])
                    cv.px(px + 1, py + 1, FG[1])
    return _rgba(cv)


def flames(c, x, y, f):
    for k, (dx, hmax) in enumerate(((-4, 6), (0, 9), (4, 6), (-2, 5), (2, 7))):
        h = int(hmax * (0.6 + 0.4 * hash01(k, f // 3)))
        for j in range(h):
            t = j / max(1, h)
            cc = 'gold' if t < 0.35 else ('amber' if t < 0.7 else 'orange')
            ww = 1 if t > 0.5 else 2
            xx = x + dx + int(round(math.sin(f / 4.0 + k + j * 0.6) * t))
            c.hline(xx - ww + 1, xx + ww - 1, y - j, cc)
        c.px(x + dx, y - h, 'butter' if k == 1 else 'orange')


def cauldron_fx(c, f, cx, cy):
    """Liquid surface, popping bubbles and green steam over the cauldron (screen coords)."""
    c.ellipse(cx, cy - 9, 12, 1, mix('green', 'lime', 0.4))
    c.hline(cx - 8, cx + 3, cy - 10, 'lime')
    c.px(cx - 4, cy - 10, 'butter')
    # bubbles: each grows for a few frames and pops; one pops on every beat
    for i in range(5):
        per = BT
        ph = (f + i * 7 + int(hash01(i) * 5)) % per
        bx = cx - 9 + int(hash01(i, f // per) * 18)
        if ph < 10:
            r = 0 if ph < 3 else 1
            by = cy - 11 - (1 if ph > 6 else 0)
            c.circle(bx, by, r, mix('lime', 'pale_leaf', 0.5), fill=(r == 0))
            if r:
                c.px(bx - 1, by - 1, 'butter')
        elif ph < 13:
            c.px(bx - 2, cy - 12, 'pale_leaf')
            c.px(bx + 2, cy - 12, 'pale_leaf')
            c.px(bx, cy - 14, 'lime')
    # steam puffs: rounded two-tone puffs that swell, drift right and thin out
    for i in range(16):
        per = 72
        t = ((f + i * per / 16.0) % per) / per
        x = cx - 3 + int(t * 24 + math.sin(t * 6 + i * 2.3) * (2 + 4 * t))
        y = cy - 12 - int(t * 74)
        r = int(2 + t * 5.5)
        body = mix('mint', 'pale_leaf', 0.25 + 0.5 * t)
        hi = mix('pale_leaf', 'mint_cream', 0.5)
        if t < 0.55:
            disc(c, x, y, r, body)
            if r >= 2:
                disc(c, x - 1, y - 1, r - 2, hi)
        else:
            m = _dmask(r)
            lvl = 1.0 - (t - 0.55) / 0.45
            th = bayer(m.shape[0], m.shape[1], x - r, y - r)
            ys, xs = np.nonzero(m & (th < lvl))
            for py, px_ in zip(ys, xs):
                c.px(x + px_ - r, y + py - r, body)


def render_B(f):
    cam = camB(f)
    c = bgB().copy()
    wild_clouds(c, f, [(40, 4, 50, 11, 0.8, 0.0), (250, 16, 36, 12, -1.0, 0.2), (150, 30, 30, 13, 0.55, 0.3),
                       (380, 0, 44, 14, 1.2, 0.1)])
    c.blit(farB(), -int(cam * 0.3), 0)
    mist(c, f, 100, 142, FOG, 0.3, 15, dens=0.85, a1=0.25, a2=0.45)
    c.blit(midB(), -int(cam * 0.6), 0)
    draw_balloon(c, 136 - int(cam * 0.4) + int((f - CUT1) * 0.12), 24, f, 's')
    c.blit(hutB(), -cam, 0)
    # per-frame hut life
    flames(c, CAUL_X - cam, GB - 1, f)
    add_glow(c, CAUL_X - cam, GB - 3, 12, 'amber', 0.22 + 0.06 * math.sin(f * 0.9))
    cauldron_fx(c, f, CAUL_X - cam, CAUL_Y)
    add_glow(c, CAUL_X - cam, CAUL_Y - 12, 16, 'lime', 0.12 + 0.04 * math.sin(f / 5.0))
    # lantern (swinging) + glow
    sw = int(round(math.sin(f / 22.0) * 1.2))
    lx, ly = LANT_B[0] - cam, LANT_B[1]
    c.line(lx, ly, lx + sw, ly + 5, 'steel_dk')
    LANTERN.draw(c, lx + sw, ly + 5, anchor='top')
    add_glow(c, lx + sw, ly + 9, 16, 'amber', 0.2 + 0.05 * math.sin(f / 3.3))
    # window + doorway light flicker
    fl = 0.12 + 0.06 * hash01(f // 4, 3)
    add_glow(c, WIN_B[0] - cam, WIN_B[1], 11, 'gold', fl)
    add_glow(c, DOOR[0] - cam, GB - 4, 14, 'amber', fl)
    # bead curtain in the doorway
    for k in range(5):
        bx = DOOR[0] - cam - 8 + k * 4
        swing = math.sin(f / 15.0 + k * 0.8) * 0.8
        for j in range(0, 22, 2):
            xx = int(round(bx + swing * j / 22))
            c.px(xx, DOOR[1] - 7 + j, ('clay', 'teal', 'sand', 'plum')[(j // 2 + k) % 4])
    # jars twinkle
    for i, (jx, kind, liq) in enumerate(JARS_L + JARS_R):
        if (f // 6 + i * 3) % 11 == 0:
            c.px(jx + 1 - cam, 141 - 5, 'white')
    # light shaft from the sky gap onto the clearing + cauldron
    slope = 0.66 + 0.06 * math.sin(f / 40.0)
    ray(c, 182 - int(cam * 0.6), 0, 166, 22, 56, slope, mix('butter', 'pale_leaf', 0.35), 0.26)
    ray_motes(c, f, 182 - int(cam * 0.6), 6, 156, slope, 50, 16, 77)
    # chimney smoke (lavender-grey wisps)
    for i in range(6):
        per = 80
        t = ((f + i * per / 6.0) % per) / per
        x = HUT_X + 24 - cam + int(t * 22 + math.sin(t * 6 + i) * 2)
        y = 27 - int(t * 30)
        if t < 0.85:
            disc(c, x, y, 1 + int(t * 2.5), mix('lilac', 'stone', 0.5 + 0.3 * t))
    for i, (vx, vy, L) in enumerate(((30, 16, 40), (52, 10, 24), (338, 14, 36), (316, 8, 22))):
        vine(c, vx - cam, vy, L, f, 90 + i)
    for i, (x, n) in enumerate(PJ_B):
        add_glow(c, x - cam, GB, 10 + n, 'baby_blue', 0.16 + 0.07 * math.sin(f / 9.0 + i))
        fx.sparkle(c, x - cam + 3 - i * 2, GB - 6, f, 'ice', period=28, phase=hash01(i, 8))
    mist(c, f, 146, 178, mix(FOG, 'mint_cream', 0.3), 0.4, 16, dens=0.6, a1=0.2, a2=0.38)
    c.blit(fgB(), -int(cam * 1.3), 0)
    falling_leaves(c, f, 8, 19)
    callout(c, HUT_X - 46 - cam, 62, "ZECORA'S HUT", f, CUT1 + 8, CUT2 - 10, side='left', length=10)
    callout(c, PJ_B[0][0] + 4 - cam, GB, 'POISON JOKE', f, CUT1 + 26, CUT2 - 10, side='down', length=8)
    return c


# ======================================================================= SHOT C: bog + gorge
WL = 150                         # bog water line
CLIFF_L, CLIFF_R = 306, 418      # near gorge rims (world, p=1)
SEEP_X = 304                     # bog seep spring on the cliff face
BR_L, BR_R = (300, 137), (425, 133)   # bridge deck anchors (world)
CASTLE_X = 254                   # far crag (p=0.1 world)
ROCK_SH = (mix('ink', 'navy', 0.3), mix('slate', 'navy', 0.35), mix('slate', 'grey', 0.35),
           mix('grey', 'sage', 0.35), mix('stone', 'sage', 0.4))


def camC(f):
    return int(round(64 + 64 * cruise(seg(f, CUT2 - 8, N))))


@lru_cache(maxsize=None)
def bgC():
    c = Canvas(W, H, 'ink')
    c.gradient(0, 0, W, 100, [mix('sky', 'teal', 0.2), 'sky', mix('light_sky', 'sky', 0.3),
                              mix('light_sky', 'mint_cream', 0.45)], soft=0.7)
    c.gradient(0, 96, W, 84, [mix('mint_cream', 'baby_blue', 0.3), mix('mint_cream', 'pale_sage', 0.4),
                              mix('pale_sage', 'sage', 0.3)], soft=0.7)
    return c


@lru_cache(maxsize=None)
def farC():
    """Distant hazy ridge (p=0.1)."""
    w = 340
    cv = _new(w, H)
    rd = mix('periwinkle', 'pale_sage', 0.6)
    rl = mix('baby_blue', 'mint_cream', 0.5)
    rs = mix(rd, 'indigo', 0.12)
    for x in range(w):
        y = 92 + int(5 * math.sin(x / 37.0) + 3 * math.sin(x / 13.0 + 1))
        cv.vline(x, y, H - 1, rd)
    for i, x in enumerate(range(0, w, 6)):
        y = 92 + int(5 * math.sin(x / 37.0) + 3 * math.sin(x / 13.0 + 1))
        rr = 2 + int(hash01(i, 3) * 3)
        disc(cv, x, y, rr, rs)
        disc(cv, x - 1, y - 1, rr - 1, rd)
        cv.px(x - 1, y - rr, rl)
    return _rgba(cv)


@lru_cache(maxsize=None)
def castle_sprite():
    """Ruined Castle of the Two Sisters on its crag: a cold, distant silhouette (60 x 64)."""
    cv = _new(60, 64)
    o, d, m, l = (mix('indigo', 'navy', 0.35), mix('indigo', 'stone', 0.42), mix('indigo', 'stone', 0.62),
                  mix('stone', 'baby_blue', 0.5))
    vine = mix('sage', 'stone', 0.45)
    cv.poly([(2, 64), (10, 48), (17, 41), (24, 38), (38, 38), (46, 44), (53, 53), (59, 64)], o)
    cv.poly([(5, 64), (12, 49), (18, 42), (24, 40), (37, 40), (45, 46), (51, 54), (56, 64)], d)
    cv.poly([(8, 64), (14, 50), (19, 44), (24, 42), (22, 64)], m)
    cv.line(14, 50, 19, 44, l)
    # curtain wall with arches + crenellations
    cv.rect(11, 29, 38, 12, o)
    cv.rect(12, 30, 36, 11, d)
    cv.rect(12, 30, 8, 11, m)
    for k in range(12, 47, 4):
        cv.rect(k, 27, 2, 2, o)
    for k in (15, 23, 37, 43):
        cv.rect(k, 33, 2, 4, o)
        cv.px(k, 32, o)
    # central keep with a broken roof + the sun-and-moon window
    cv.rect(23, 8, 14, 23, o)
    cv.rect(24, 9, 12, 22, d)
    cv.rect(24, 9, 4, 22, m)
    cv.poly([(21, 9), (30, -1), (39, 9)], o)
    cv.poly([(23, 8), (30, 1), (33, 5), (31, 8)], m)
    cv.circle(30, 16, 3, o)
    cv.px(29, 15, 'gold')
    cv.px(29, 16, 'gold')
    cv.px(31, 16, 'baby_blue')
    cv.px(31, 17, 'baby_blue')
    cv.px(30, 17, mix('gold', 'baby_blue', 0.5))
    # left tower (cone roof), right tower (broken)
    cv.rect(12, 12, 8, 19, o)
    cv.rect(13, 13, 6, 18, d)
    cv.rect(13, 13, 2, 18, m)
    cv.poly([(10, 13), (16, 1), (22, 13)], o)
    cv.poly([(12, 12), (16, 3), (16, 12)], m)
    cv.rect(15, 18, 2, 3, o)
    cv.rect(41, 18, 8, 13, o)
    cv.rect(42, 19, 6, 12, d)
    cv.poly([(41, 18), (43, 14), (45, 17), (47, 13), (49, 18)], o)
    cv.rect(44, 22, 2, 3, o)
    # vines + lit left edges
    for (vx, vy, L) in ((26, 20, 8), (34, 12, 6), (14, 22, 7), (46, 22, 6), (20, 31, 6)):
        for j in range(L):
            cv.px(vx + (j // 3) % 2, vy + j, vine)
    for (x, y0, y1) in ((12, 13, 30), (23, 9, 29), (11, 30, 40)):
        cv.vline(x, y0, y1, l)
    return _rgba(cv)


@lru_cache(maxsize=None)
def wallsC(depth):
    """Forest plateau + converging gorge walls. depth 0 = far (p=0.3), 1 = mid (p=0.6)."""
    w = 400 if depth == 0 else 430
    cv = _new(w, H)
    hz = 0.62 if depth == 0 else 0.34
    to = mix('mint_cream', 'pale_sage', 0.45) if depth == 0 else AIR[2]
    rsh, rlt = haze(ROCK_SH, hz, to), haze(ROCK, hz, to)
    lf, bk, gr = haze(LEAF, hz, to), haze(BARK, hz, to), haze(GROUND, hz, to)
    gl, grr = (252, 306) if depth == 0 else (274, 356)
    topL, topR = (104, 102) if depth == 0 else (118, 115)
    yy, xx = np.mgrid[0:H, 0:w]
    edgeL = gl + 3 * np.sin(yy / 6.0 + depth) + 2 * np.sin(yy / 2.7)
    edgeR = grr + 3 * np.sin(yy / 7.0 + 1 + depth) + 2 * np.sin(yy / 3.1)
    bank = 114 if depth == 0 else 134
    air = mix(mix(AIR[1], AIR[2], 0.5), to, hz)
    xe = gl - 46 + 8 * np.sin(yy / 11.0) + 3 * np.sin(yy / 3.0)
    cv.a[(xx < xe) & (yy < bank + 2)] = col(air)          # forest air behind the bog (no sky)

    def gtop(x):
        if x >= (gl + grr) / 2:
            return topR
        return int(round(bank + (topL - bank) * smooth((x - (gl - 62)) / 34.0)))
    # ground (far shore of the bog / plateaus)
    for x in range(w):
        top = gtop(x)
        cv.vline(x, top, H - 1, gr[1])
        cv.vline(x, top, top + 1, gr[3])
        cv.px(x, top + 2, gr[2])
        if hash01(x, depth, 11) > 0.6:
            cv.px(x, top + 3 + int(hash01(x, 12) * 8), gr[2])
    # rock faces: the left wall faces right (shadow), the right wall faces left (sunlit)
    mL = (yy >= topL) & (xx <= edgeL) & (xx > gl - 40)
    mR = (yy >= topR) & (xx >= edgeR) & (xx < grr + 40)
    gap = (xx > edgeL) & (xx < edgeR)
    cv.a[gap] = _KEYA
    rockfill(cv, mL, rsh, 5000 + depth, bw=8 + depth * 2, bh=5 + depth)
    rockfill(cv, mR, rlt, 5100 + depth, bw=8 + depth * 2, bh=5 + depth)
    for x in range(gl - 40, gl + 6):
        cv.px(x, topL, gr[3])
    for x in range(grr - 6, grr + 40):
        cv.px(x, topR, gr[3])
    # forest on the left plateau (behind the bog) and on the right plateau
    step = 22 if depth == 0 else 30
    for i, x in enumerate(range(4, gl - 50, step)):
        x += int(hash01(i, depth) * 8)
        trunk(cv, x, gtop(x) + 2, -8, 3.5 + depth * 3, 2 + depth * 1.5, bk,
              5200 + i + depth * 50, wob=4, flare=3, grooves=1 + depth, knots=1)
    for i, x in enumerate(range(-8, gl - 40, 12)):
        canopy(cv, x, 8 + 8 * math.sin(x / 17.0 + depth), 8, 14, 5, 6, 11, lf, 5300 + i + depth * 50)
    for i, y in enumerate(range(10, bank - 10, 12)):      # leafy edge of the forest wall
        canopy(cv, gl - 46 + 8 * math.sin(y / 11.0), y, 4, 5, 2, 4, 7, lf, 5350 + i + depth * 50)
    for i, x in enumerate(range(0, gl - 40, 7)):           # bushes + reeds on the far bank
        bt = gtop(x)
        leafball(cv, x, bt - 1, 2 + int(hash01(i, depth, 4) * 3), lf)
        if i % 3 == 0:
            cv.line(x + 3, bt, x + 4, bt - 6 - int(hash01(i, 6) * 5), lf[3])
    for i, x in enumerate(range(gl - 30, gl + 2, 10)):          # rim trees, lower crowns
        trunk(cv, x, topL + 1, topL - 26, 2 + depth, 1.5, bk, 5400 + i + depth * 9, wob=2, flare=1, grooves=1,
              knots=0)
        canopy(cv, x, topL - 28 + int(hash01(i, 5) * 6), 6, 4, 4, 3, 6 - depth, lf, 5450 + i + depth * 9)
    for i, x in enumerate(range(grr + 4, w + 8, 12)):
        tall = x > grr + 40
        top = topR - (60 if tall else 26)
        trunk(cv, x, topR + 1, top, 2.5 + depth * 2, 1.5 + depth, bk, 5500 + i + depth * 9, wob=3, flare=2,
              grooves=1, knots=0)
        canopy(cv, x, top - 2, 8, 5, 4, 4, 8 - depth, lf, 5550 + i + depth * 9)
    for i, x in enumerate(range(gl - 38, gl + 2, 5)):           # roots/vines over the left face
        L = 4 + int(hash01(i, depth, 7) * 16)
        for j in range(L):
            cv.px(x + (j // 4) % 2, topL + 2 + j, lf[2] if j % 3 else lf[3])
    for i, x in enumerate(range(grr, grr + 38, 6)):
        L = 3 + int(hash01(i, depth, 8) * 12)
        for j in range(L):
            cv.px(x + (j // 4) % 2, topR + 2 + j, lf[2] if j % 3 else lf[3])
    if depth == 0:                       # rock spires rising from the gorge mist
        for k, (sx, top, sw) in enumerate(((gl + 14, 124, 5), (grr - 16, 130, 4))):
            m = (xx >= sx - sw + np.sin(yy / 4.0 + k)) & (xx <= sx + sw + np.sin(yy / 5.0 + k)) & (yy >= top)
            m &= ~(((yy - top) < 3) & (np.abs(xx - sx) > sw - 2))
            rockfill(cv, m, rlt, 5700 + k, bw=5, bh=4, moss=False)
            canopy(cv, sx, top - 3, 3, 2, 3, 2, 4, lf, 5710 + k)
    speckle(cv, lf[2], lf[1], 0.08, 5600 + depth)
    return _rgba(cv)


LILY = [(30, 157, 5), (52, 163, 4), (96, 155, 3), (150, 166, 6), (176, 158, 4), (214, 170, 5), (236, 156, 3),
        (128, 174, 5), (70, 172, 4)]
WISPS = [(40, 118), (112, 134), (176, 104), (232, 128)]


def _lip_top(x):
    t = clamp((x - 248) / (CLIFF_L - 248))
    return int(WL + 2 - 13 * smooth(t) + 1.5 * math.sin(x / 5.0))


@lru_cache(maxsize=None)
def nearC():
    w = 460
    cv = _new(w, H)
    yy, xx = np.mgrid[0:H, 0:w]
    # mangrove trees with tangled prop roots standing in the water
    for i, (x, w0) in enumerate(((66, 8), (196, 9))):
        r = rng(4010 + i)
        for k in range(11):
            s = -1 if k % 2 == 0 else 1
            L = r.uniform(10, 36)
            x0 = x + s * r.uniform(0, w0 * 0.8)
            y0 = 126 - r.uniform(0, 30)
            pts = bez([(x0, y0), (x0 + s * L * 0.45, y0 - r.uniform(2, 10)), (x0 + s * L * 0.9, WL - 16),
                       (x0 + s * L * r.uniform(0.95, 1.15), WL + 5)])
            tube(cv, pts, 2 + (k % 3 == 0), 1, BARK)
        sp = trunk(cv, x, 126, -20, w0, w0 * 0.75, BARK, 4000 + i, wob=5, flare=2, grooves=3, knots=2, streaks=1)
        for j, side in enumerate((-1, 1)):
            pts = branch(cv, sp, 0.62 + 0.15 * j, side, 30, 3.0, 4, BARK, -20)
            ex, ey = pts[-1]
            canopy(cv, ex, ey - 4, 8, 4, 4, 4, 7, LEAF, 4030 + i * 3 + j)
    # canopy band over the bog
    for i, x in enumerate(range(-12, 262, 12)):
        canopy(cv, x, 4 + 6 * math.sin(x / 21.0), 8, 9, 4, 7, 13, LEAF, 4100 + i)
    for i, x in enumerate(range(0, 258, 4)):        # spanish-moss drapes
        L = int(6 + hash01(i, 41) * 24)
        y0 = 20 + int(6 * math.sin(x / 21.0))
        for j in range(L):
            if hash01(i, j, 3) > 0.25:
                cv.px(x + int(math.sin(j / 4.0 + i) * 1.2), y0 + j, mix('sage', 'pale_sage', 0.2 + 0.4 * (j / L)))
    speckle(cv, LEAF[2], LEAF[1], 0.09, 4163)
    speckle(cv, LEAF[3], LEAF[4], 0.07, 4164)
    speckle(cv, PLUM_LEAF[2], PLUM_LEAF[1], 0.1, 4165)
    # cattail islet + reeds at the left bank
    for x0, x1 in ((118, 146), (-4, 14)):
        cv.ellipse((x0 + x1) // 2, WL + 1, (x1 - x0) // 2, 3, GROUND[1])
        cv.ellipse((x0 + x1) // 2, WL, (x1 - x0) // 2 - 1, 2, GROUND[3])
        for k, x in enumerate(range(x0 + 2, x1 - 1, 3)):
            h = 10 + int(hash01(x, 2) * 12)
            cv.line(x, WL, x + (1 if k % 2 else -1), WL - h, 'olive')
            if k % 2 == 0:
                cv.rect(x - 1, WL - h - 1, 2, 4, mix('rust', 'wine', 0.3))
                cv.px(x - 1, WL - h - 3, 'olive')
    # the lip + left gorge cliff (shadowed face), right gorge cliff (sunlit face)
    lipm = np.zeros((H, w), bool)
    for x in range(246, CLIFF_L + 14):
        lipm[_lip_top(x):, x] = True
    edgeL = CLIFF_L + 2 * np.sin(yy / 5.0) - np.maximum(0, yy - 140) / 9.0
    lipm &= xx <= edgeL
    rockfill(cv, lipm, ROCK_SH, 4400, bw=11, bh=6)
    for x in range(246, CLIFF_L + 2):
        t0 = _lip_top(x)
        cv.vline(x, t0, t0 + 1, GROUND[3])
        cv.px(x, t0 + 2, GROUND[2])
        if hash01(x, 31) > 0.55:
            cv.px(x, t0 - 1, GROUND[4])
    mR = (xx >= CLIFF_R + 2 * np.sin(yy / 6.0 + 1) - np.maximum(0, yy - 136) / 10.0) & (yy >= BR_R[1] + 2)
    rockfill(cv, mR, ROCK, 4410, bw=11, bh=6)
    cv.hline(CLIFF_R - 2, w, BR_R[1] + 2, GROUND[3])
    cv.hline(CLIFF_R - 1, w, BR_R[1] + 3, GROUND[2])
    for i, x in enumerate(range(CLIFF_L - 30, CLIFF_L + 2, 4)):    # roots + vines spilling over the rims
        L = 4 + int(hash01(i, 71) * 18)
        for j in range(L):
            xx_ = int(round(x + 2 * math.sin(j / 5.0 + i)))
            cv.px(min(xx_, int(CLIFF_L + 2 * math.sin((140 + j) / 5.0)) - 1), _lip_top(x) + 3 + j,
                  LEAF[2] if j % 3 else LEAF[3])
    for i, x in enumerate(range(CLIFF_R + 2, CLIFF_R + 40, 5)):
        L = 3 + int(hash01(i, 72) * 14)
        for j in range(L):
            cv.px(x + (j // 4) % 2, BR_R[1] + 4 + j, LEAF[2] if j % 3 else LEAF[3])
    # gnarled tree on the right rim
    sp = trunk(cv, 448, BR_R[1] + 3, 30, 8, 5, BARK, 4200, wob=6, lean=-0.1, flare=5, grooves=3, knots=2)
    roots(cv, 448, BR_R[1] + 3, 8, BARK, 4210, n=4, L=16, r0=3, front=0)
    pts = branch(cv, sp, 0.7, -1, 34, 3.4, 4, BARK, 30)
    canopy(cv, pts[-1][0], pts[-1][1] - 4, 8, 4, 5, 4, 7, LEAF, 4220)
    canopy(cv, 446, 30, 16, 10, 8, 6, 11, LEAF, 4230)
    # bridge anchor posts + rope wraps
    for (ax, ay) in (BR_L, BR_R):
        for dx in (-3, 3):
            cv.rect(ax + dx - 1, ay - 14, 3, 18, WOOD[0])
            cv.vline(ax + dx, ay - 13, ay + 2, WOOD[2])
            cv.hline(ax + dx - 1, ax + dx + 1, ay - 9, 'tan')
            cv.hline(ax + dx - 1, ax + dx + 1, ay - 3, 'tan')
            cv.px(ax + dx, ay - 15, WOOD[3])
    # small life on the lip
    for x in (256, 270, 284):
        grass(cv, x, _lip_top(x), 4, GROUND, 4300 + x)
    poison_joke(cv, 268, _lip_top(268), 4, 4310)
    mushroom(cv, 282, _lip_top(282) + 1, 3)
    glowshroom(cv, 254, _lip_top(254) + 1)
    glowshroom(cv, 258, _lip_top(258) + 1)
    return _rgba(cv)


@lru_cache(maxsize=None)
def lilyC():
    w = 300
    cv = _new(w, H)
    for i, (x, y, r) in enumerate(LILY):
        cv.ellipse(x, y, r, max(1, r // 3), mix('forest', 'deep_teal', 0.3))
        cv.ellipse(x - 0.5, y - 0.5, r - 1, max(1, r // 3) - 0.5, 'green')
        cv.hline(x - r + 2, x - 1, y - 1, mix('green', 'leaf', 0.4))
        cv.line(x, y, x + r, y - 1, mix('forest', 'deep_teal', 0.3))    # notch
        if i % 3 == 0:
            cv.px(x - 1, y - 2, 'pink')
            cv.px(x, y - 2, 'cotton')
            cv.px(x - 1, y - 3, 'cotton')
            cv.px(x + 1, y - 2, 'pink')
    return _rgba(cv)


@lru_cache(maxsize=None)
def fgC():
    w = 480
    cv = _new(w, H)
    for i, x in enumerate((6, 92, 168, 236)):
        for k in range(4):
            h = 16 + int(hash01(i, k) * 16)
            xx = x + k * 4
            lean = (k - 1.5) * 0.15
            pts = bez([(xx, H + 2), (xx + lean * h * 0.5, H - h * 0.6), (xx + lean * h, H - h)])
            for px_, py_ in pts:
                cv.px(px_, py_, FG[2])
            ex, ey = pts[-1]
            if k % 2 == 0:
                cv.rect(int(ex) - 1, int(ey) - 1, 3, 6, mix('ink', 'wine', 0.45))
                cv.px(int(ex), int(ey) - 2, FG[2])
    return _rgba(cv)


def bridge(c, f, cam):
    """Rickety rope bridge between the anchors, swaying (screen coords)."""
    (lx, ly), (rx, ry) = BR_L, BR_R
    lx -= cam
    rx -= cam
    sw = math.sin(f * 2 * math.pi / 54.0)
    sag = 12 + 2.0 * sw
    n = int(rx - lx)
    rope_d = mix('tan', 'plum_brown', 0.45)
    prev = None
    for i in range(n + 1):
        t = i / n
        bump = 4 * t * (1 - t)
        yd = int(round(ly + (ry - ly) * t + sag * bump))
        yr = int(round(ly - 9 + (ry - ly) * t + (sag - 4) * bump))
        x = lx + i
        if not (-2 <= x < W + 2):
            continue
        c.px(x, yr, 'tan')
        if prev is not None and abs(prev - yr) > 1:
            c.px(x, (prev + yr) // 2, 'tan')
        prev = yr
        if i % 7 == 3:
            c.vline(x, yr + 1, yd - 1, rope_d)
        k = i // 3
        if hash01(k, 77) < 0.12 or i % 3 == 2:     # gaps + missing planks
            c.px(x, yd + 1, rope_d)
            continue
        c.px(x, yd, WOOD[3] if i % 3 == 0 else WOOD[2])
        c.px(x, yd + 1, WOOD[1])
        c.px(x, yd + 2, WOOD[0] if hash01(k, 3) < 0.6 else WOOD[1])
        if hash01(k, 9) > 0.85:
            c.px(x, yd + 3, WOOD[0])                 # a plank hanging askew


def reflect(c, x0, x1, wl, f):
    """Mirror the picture above the water line into the bog (screen x0..x1), murky + rippled."""
    x0, x1 = max(0, x0), min(W, x1)
    if x1 <= x0:
        return
    water = np.array(mix('dark_olive', 'deep_teal', 0.45), np.float32)
    for y in range(wl + 1, H):
        src = 2 * wl - y
        if src < 0:
            break
        dx = int(round(math.sin(y * 0.9 + f * 0.18) * (1 + (y - wl) / 25.0)))
        sx0, sx1 = max(0, x0 - dx), min(W, x1 - dx)
        if sx1 <= sx0:
            continue
        row = c.a[src, sx0:sx1].astype(np.float32)
        k = 0.5 + 0.2 * min(1.0, (y - wl) / 30.0)
        row = row * (1 - k) + water * k
        c.a[y, sx0 + dx:sx1 + dx] = row.astype(np.uint8)
    for y in range(wl + 3, H, 5):        # drifting surface sheen
        ph = (f * 0.4 + y * 13) % 60
        sx = x0 + int((hash01(y) * (x1 - x0) + ph * 3) % max(1, x1 - x0))
        c.hline(sx, min(x1 - 1, sx + 5), y, mix('sage', 'pale_sage', 0.4))
    c.hline(x0, x1 - 1, wl + 1, mix('sage', 'deep_teal', 0.3))


def bog_bubbles(c, f, cam):
    for i in range(8):
        per = 40 + int(hash01(i, 1) * 30)
        ph = (f + int(hash01(i, 2) * per)) % per
        cyc = (f + int(hash01(i, 2) * per)) // per
        x = int(10 + hash01(i, cyc, 3) * 230) - cam
        y = WL + 4 + int(hash01(i, cyc, 4) * 24)
        if not (0 <= x < W):
            continue
        if ph < 12:
            r = ph // 4
            c.circle(x, y, r, mix('sage', 'pale_leaf', 0.4), fill=False)
            c.px(x - 1, y - r, 'pale_leaf')
        elif ph < 20:
            r = 2 + (ph - 12) // 3
            c.ellipse(x, y, r + 1, max(1, r // 2), mix('sage', 'deep_teal', 0.3), fill=False)


def render_C(f):
    cam = camC(f)
    c = bgC().copy()
    wild_clouds(c, f, [(0, 6, 54, 21, 0.7, 0.1), (200, 26, 40, 22, -0.9, 0.25), (120, 44, 30, 23, 1.4, 0.3),
                       (330, 14, 46, 24, -0.5, 0.15)])
    c.blit(farC(), -int(cam * 0.1), 0)
    # GLIMPSE: the mist curtain parts and the far castle appears
    g = ease_out(seg(f, GLIMPSE - 2, GLIMPSE + 8))
    ccx = CASTLE_X - int(cam * 0.1)
    if g > 0:
        add_glow(c, ccx, 66, 36, 'periwinkle', 0.2 * g)
        spr = castle_sprite()
        x0, y0 = ccx - 30, 98 - 64
        sub = c.a[max(0, y0):y0 + 64, x0:x0 + 60]
        th = bayer(64, 60, x0, y0)[-sub.shape[0]:, :sub.shape[1]]
        m = (spr[-sub.shape[0]:, :sub.shape[1], 3] > 0) & (th < g)
        sub[m] = spr[-sub.shape[0]:, :sub.shape[1], :3][m]
    yb = _YYf[28:118]
    hole = 1 - g * np.exp(-(((_XXf[28:118] - ccx) / (12 + 36 * g)) ** 2 + ((yb - 72) / 44.0) ** 2))
    mist(c, f, 28, 118, mix('mint_cream', 'baby_blue', 0.25), 0.3, 25, dens=1.0 + 0.9 * (1 - g),
         a1=0.55, a2=0.9, hole=hole)
    if f >= GLIMPSE:
        fx.sparkle(c, ccx, 50, f - GLIMPSE, 'white', period=22, size=3)
    c.blit(wallsC(0), -int(cam * 0.3), 0)
    # a little cloud raining all by itself over the gorge rim (the weather runs itself)
    rcx = 217 - int(cam * 0.3) + int((f - CUT2) * 0.12)
    fx.rain(c, f, region=(rcx - 12, 34, 26, 64), n=22, color=mix('light_sky', 'mint_cream', 0.4), seed=5,
            slant=0, speed=5, length=3)
    c.blit(cloud_sprite(30, 31, 0.9), rcx - 18, 22)
    mist(c, f, 108, 168, mix(AIR[3], 'mint_cream', 0.5), 0.35, 26, dens=1.0, a1=0.35, a2=0.6)
    c.blit(wallsC(1), -int(cam * 0.6), 0)
    draw_balloon(c, 336 - int(cam * 0.6) + int((f - CUT2) * 0.08), 72, f, 'm', flame=(f // 24) % 3 == 0)
    mist(c, f, 136, 180, mix(AIR[3], 'mint_cream', 0.55), 0.6, 27, dens=1.3, a1=0.4, a2=0.7)
    c.blit(nearC(), -cam, 0)
    bridge(c, f, cam)
    # seep spring falling from the cliff face into the gorge
    sx = SEEP_X - cam
    if -4 < sx < W + 4:
        for y in range(WL + 2, H):
            j = (y - f) % 6
            xx = sx + (y - WL) // 10 + 1
            c.px(xx, y, 'mint_cream' if j < 2 else mix('pale_sage', 'mint_cream', 0.5))
            if j == 0:
                c.px(xx + 1, y, mix('sage', 'mint_cream', 0.5))
    mist(c, f, 150, 180, mix(AIR[3], 'mint_cream', 0.6), 0.8, 28, dens=0.9, a1=0.3, a2=0.55)
    reflect(c, -cam, 248 - cam, WL, f)
    c.blit(lilyC(), -cam, 0)
    bog_bubbles(c, f, cam)
    # will-o'-wisps (single drifting lights)
    for i, (wx, wy) in enumerate(WISPS):
        x = wx - cam + int(round(6 * math.sin(f / 23.0 + i * 1.9)))
        y = wy + int(round(4 * math.sin(f / 17.0 + i * 2.6)))
        if -10 < x < W + 10:
            k = 0.24 + 0.08 * math.sin(f / 6.0 + i)
            add_glow(c, x, y, 9, 'turquoise', k)
            c.px(x, y, 'white')
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c.px(x + dx, y + dy, 'ice')
            for tr in range(1, 4):
                tx = x - int(round(6 * math.cos(f / 23.0 + i * 1.9) * tr * 0.25))
                if hash01(i, tr, f // 3) > 0.35:
                    c.px(tx, y + tr + 1, 'turquoise')
    c.blit(fgC(), -int(cam * 1.3), 0)
    falling_leaves(c, f, 6, 29)
    callout(c, 150 - cam, WL + 7, 'FROGGY BOTTOM BOGG', f, CUT2 + 10, CUT2 + 82, side='down', length=6)
    callout(c, 362 - cam, 151, 'ROPE BRIDGE', f, CUT2 + 28, N - 14, side='down', length=8)
    return c


# ======================================================================= render
def render(f: int) -> Canvas:
    if f < 72:
        return travel_frame(f, ID)
    if f < CUT1:
        c = render_A(f)
        if f < 80:
            iris(c, 1 - seg(f, 72, 79), *IRIS_A)
    elif f < CUT2:
        c = render_B(f)
    else:
        c = render_C(f)
    atlas_card(c, f, 98, 206, 'WILD FOREST', ['WEATHER RUNS ITSELF', 'TREAD CAREFULLY'])
    return c


SFX = [                         # scene frames (long_flight maps them to film frames)
    (80, 'wind', 0.45),
    (GUST - 2, 'wind', 0.75),       # the self-made gust (variant 2: howling)
    (136, 'shimmer', 0.3),          # poison joke twinkle
    (CUT1 + 2, 'bubbles', 0.55),    # the cauldron
    (180, 'steam', 0.4),
    (198, 'pop', 0.4),
    (216, 'pop', 0.35),
    (234, 'pop', 0.3),
    (CUT2 + 4, 'bubbles', 0.45),    # the bog
    (270, 'creak', 0.55),           # rope bridge
    (GLIMPSE - 6, 'wind', 0.5),     # the mist parts
    (GLIMPSE, 'shimmer', 0.45),     # castle glimpse
    (318, 'creak', 0.45),
    (336, 'bubbles', 0.3),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(CUT1, 'glide', {'dir': -1}), (CUT2, 'glide', {'dir': -1})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
