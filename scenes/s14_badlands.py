"""BADLANDS: the Changeling Kingdom hive in its striped-mesa crater + the Tartarus gate
(4 bars, 288 frames, HUD clock 17:45 -> 18:30, golden hour, sun low in the west = screen left).

Beat sheet (local frames; bar = 72, beat = 18). Chords: F#m@72  F#@144  Bmaj9@216  E7b9@252.
  Bar 1   0-71    World-map travel shot (`travel_frame`). Iris closes on dest_screen (204, 103).
  Bar 2  72-143   THE DARK HIVE (shot A, wide). Iris opens 72-79 on the hive's heart (204, 103).
                  Golden-hour crater: a sky gone sickly green-grey with haze, a pale sun low on the
                  left, jagged mauve peaks ringing the crater, striped stepped mesas (far + mid) casting
                  long shadows to the right. The huge black-grey hive mound (termite hill, jagged
                  spires, honeycomb holes, tunnels) with faint green glows pulsing in its holes, slime
                  drips, dead leafless trees, bubbling green slime pools, drifting haze bands.
                  Camera: slow eased pan right (parallax 0 sky / .12 peaks / .28 far mesas /
                  .5 crater + hive / .85 near ground / 1.2 foreground rocks).
                  Balloon 's' drifting in the sky left of the hive. Callouts THE BADLANDS 90-150
                  (right mesa), CHANGELING KINGDOM 108-156 (hive spire). Card 100-208.
                  126-143: the heart's glow swells, 1-px tremor.
  Bar 3 144-215   TRANSFORM (144): flash at the heart; an expanding rainbow ring (teal/pink/lime/
                  violet, white leading edge, sparkles riding it) sweeps outward over ~1 bar: behind
                  it the chitin is translucent faceted crystal with inner glow (teal body, pink,
                  lime and violet spires), holes become warm windows, vines + flowers drape the hive
                  and the rocks, dead trees blossom, slime pools turn clear turquoise, the haze and
                  green tint clear to warm gold. Petals drift, facets glint. Callout THE HIVE IN BLOOM
                  162-208 (same spire: the name tag "changes").
  Bar 4 216-287   CUT (216) to the CLOSE-UP of the blooming hive's flank (lantern-lit vine arches,
                  crystal facets catching the sun, round windows, turquoise pool + reflection,
                  flowers). Eased pan WEST (left) 220-252 past the hive's edge into a bleak shadowed
                  canyon to the TARTARUS gate: a massive carved stone double gate in the cliff,
                  sealed by a glowing rune-ring lock emblem, red light leaking at the seams, cracks
                  pulsing. TARTARUS sting 252: seams flare white-hot, red light floods the steps, dust
                  and pebbles fall from the lintel, 2-px decaying shake. Callout TARTARUS 244-279.
                  Toast from 230: top centre kept clear. Last 8 frames: keep moving (embers, dust).
Ambient: drifting clouds + haze, hole glows, slime bubbles, slime drips, heat shimmer dust |
ring + sparkles, facet glints, petals, lantern flicker, pool ripples, flower sway |
seam pulses, rune ring rotation, embers, falling dust, balloon bob.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine.timeline import FLIGHT_SHIFT
from engine.shotfx import cruise
from engine.atlas import draw_balloon, callout, atlas_card, iris
from engine.loc_worldmap import travel_frame

ID = 'badlands'
N = seg_frames(ID) - FLIGHT_SHIFT          # scene frames (engine/regionfx.py)
B = FRAMES_PER_BEAT
KEY = (1, 2, 3)
T_FORM = 144
T_TART = 252


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
    X1, Y1 = min(c.w, x + lw), min(c.h, y + lh)
    if X1 <= X0 or Y1 <= Y0:
        return
    sa = a[Y0 - y:Y1 - y, X0 - x:X1 - x]
    sm = m[Y0 - y:Y1 - y, X0 - x:X1 - x]
    np.copyto(c.a[Y0:Y1, X0:X1], sa, where=sm[:, :, None])


def vnoise(h, w, cy, cx, seed):
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


def noise1(n, scale, seed, octaves=3):
    return fbm(1, n, 1, scale, seed, octaves)[0]


def poly_mask(h, w, pts):
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


def ramp_arr(ramp):
    return np.array([col(r) for r in ramp], np.uint8)


def ramp_idx(t, n, x0, y0, h, w, dith=0.7):
    thr = bayer(h, w, x0, y0)
    return np.clip(np.floor(t * n + (thr - 0.5) * dith), 0, n - 1).astype(int)


def outline_of(m):
    pad = np.pad(m, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~inner


def shift(m, dx, dy):
    """Boolean mask shifted by (dx, dy) (pixels moved right/down), zero-filled."""
    out = np.zeros_like(m)
    h, w = m.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def paint(cv, m, color, x0=0, y0=0):
    """Paint a boolean mask (top-left at x0, y0) onto a canvas."""
    h, w = m.shape
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(cv.w, x0 + w), min(cv.h, y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    cv.a[Y0:Y1, X0:X1][sub] = C(color)


def blob_union(h, w, bl):
    """Union of ellipsoids [(cx, cy, rx, ry, z, tag)] -> mask, normals, tag map."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx += 0.5
    yy += 0.5
    best = np.full((h, w), -1e9, np.float32)
    NX = np.zeros((h, w), np.float32)
    NY = np.zeros((h, w), np.float32)
    NZ = np.zeros((h, w), np.float32)
    TAG = np.zeros((h, w), np.int16)
    for (cx, cy, rx, ry, z, tag) in bl:
        x0, x1 = max(0, int(cx - rx - 1)), min(w, int(cx + rx + 2))
        y0, y1 = max(0, int(cy - ry - 1)), min(h, int(cy + ry + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        sx, sy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        u = (sx - cx) / rx
        v = (sy - cy) / ry
        d2 = u * u + v * v
        ins = d2 <= 1.0
        zz = np.sqrt(np.clip(1 - d2, 0, 1))
        hgt = z + zz * min(rx, ry)
        b = best[y0:y1, x0:x1]
        upd = ins & (hgt > b)
        b[upd] = hgt[upd]
        NX[y0:y1, x0:x1][upd] = u[upd]
        NY[y0:y1, x0:x1][upd] = v[upd]
        NZ[y0:y1, x0:x1][upd] = zz[upd]
        TAG[y0:y1, x0:x1][upd] = tag
    m = best > -1e8
    return m, NX, NY, NZ, TAG


LIGHT = np.array([-0.78, -0.42, 0.46], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def lambert(NX, NY, NZ):
    return np.clip((NX * LIGHT[0] + NY * LIGHT[1] + NZ * LIGHT[2] + 1) / 2, 0, 0.999)


def cam_ease(f, f0, f1, a, b, ease=ease_in_out):
    return lerp(a, b, ease(seg(f, f0, f1)))


def flash(c, f, f0, color='white', k=0.5):
    """Two-frame light flash (smooth blend, no dot screen)."""
    if f == f0:
        c.blend(0, 0, W, H, color, k)
    elif f == f0 + 1:
        c.blend(0, 0, W, H, color, k * 0.4)


def gloom(cv, k=1.0):
    """The hive's sickly green-grey haze over the golden light (dark state); keeps KEY pixels."""
    msk = ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    a = cv.a[msk].astype(np.float32)
    lum = a @ np.array([0.3, 0.55, 0.15], np.float32)
    g = a * 0.62 + lum[:, None] * 0.2 + np.array([12, 20, 8], np.float32)
    a = a + (g - a) * k
    cv.a[msk] = np.clip(a, 0, 255).astype(np.uint8)


def shadow_poly(cv, pts, k=0.32, tint='wine'):
    """Darken the pixels under a polygon (cast shadow), tinted."""
    m = poly_mask(cv.h, cv.w, pts) & ~np.all(cv.a == np.array(KEY, np.uint8), axis=2)
    a = cv.a[m].astype(np.float32)
    a = a * (1 - k) * 0.92 + np.array(col(tint), np.float32) * k * 0.9
    cv.a[m] = np.clip(a, 0, 255).astype(np.uint8)


# ================================================================== palettes
CRYSTAL = {   # dark -> light (5 steps) per hue: teal, pink, lime, violet
    0: ('deep_teal', 'teal', 'aqua', 'turquoise', 'ice'),
    1: ('raspberry', 'rose', 'pink', 'bubblegum', 'cotton'),
    2: ('olive', 'moss', 'lime', '#e6f28a', '#f6ffc8'),
    3: ('purple', 'violet', 'lavender', 'lilac', '#f6e2ff'),
}
CHITIN = ('ink', '#26232b', 'basalt', 'charcoal', '#3b3f3e', '#4d514c')
SLIME = ('dark_olive', 'olive', '#7fb83a', '#8be04b', '#c6f27a')
FLOWERS = ('bubblegum', 'cream', 'butter', 'lilac', 'magic', 'ice')

# strata bands for the mesas: (lit, mid, shadow, deep)
STRATA = [
    ('coral', 'copper', 'rust', 'wine'),
    ('sand', 'clay', 'copper', 'rust'),
    ('#f6d58e', 'sand', 'clay', 'rust'),
    ('salmon', 'brick', 'maroon', 'wine'),
    ('sand', 'clay', 'copper', 'rust'),
    ('peach', 'coral', 'rust', 'wine'),
]


# ================================================================== drawing pieces
def haze_rows(cv, y0, y1, haze, t0, t1, steps=4):
    """Stepped atmospheric blend toward `haze` on non-KEY pixels, t0 at row y0 -> t1 at row y1."""
    hz = np.array(col(haze), np.float32)
    key = np.array(KEY, np.uint8)
    for y in range(max(0, y0), min(cv.h, y1 + 1)):
        t = lerp(t0, t1, (y - y0) / max(1, y1 - y0))
        t = round(t * steps) / steps
        if t <= 0:
            continue
        row = cv.a[y]
        m = ~np.all(row == key, axis=1)
        row[m] = np.clip(row[m].astype(np.float32) * (1 - t) + hz * t, 0, 255).astype(np.uint8)


def draw_mesa(cv, cx, base, tiers, seed, band_h=4, haze=None, haze_t=0.0, talus=10, lit=0.38):
    """Striped stepped mesa. tiers: [(half_width, height), ...] bottom -> top (above the talus).

    Sun from the left: lit left faces, shadowed right faces, bright tops of every ledge.
    """
    R = rng(seed)
    total = talus + sum(t[1] for t in tiers)
    top = base - total
    rows = []
    off = 0.0
    hw_base = tiers[0][0] + talus * 1.3
    for i in range(talus):
        rows.append((base - i, hw_base - i * 1.3, 0.0, 0))
    y = base - talus
    for ti, (hw, hh) in enumerate(tiers):
        off += R.uniform(-2, 2) if ti else 0
        for j in range(hh):
            rows.append((y - j, hw + (hh - j) * 0.1, off, ti + 1))
        y -= hh
    h = total + 2
    w = int(hw_base * 2 + 8)
    x0 = int(cx - w / 2)
    yy0 = base - h + 1
    edge_n = (noise1(h + 4, 4, seed + 3) * 2.2 - 1.1).round()
    edge_n2 = (noise1(h + 4, 4, seed + 5) * 2.2 - 1.1).round()
    m = np.zeros((h, w), bool)
    u = np.zeros((h, w), np.float32)
    tier = np.zeros(h, int)
    for (y, hw, o, ti) in rows:
        r = y - yy0
        if not 0 <= r < h:
            continue
        a = max(0, int(round(cx + o - hw + edge_n[r] - x0)))
        b = min(w - 1, int(round(cx + o + hw + edge_n2[r] - x0)))
        if b <= a:
            continue
        m[r, a:b + 1] = True
        u[r, a:b + 1] = (np.arange(a, b + 1) - a) / max(1, b - a)
        tier[r] = ti
    yy, xx = np.mgrid[0:h, 0:w]
    wob = (noise1(w + 4, 7, seed + 7) * 2.6).astype(int)[:w]
    band = ((yy + yy0 + wob[None, :]) // band_h + seed) % len(STRATA)
    # crisp lit/shadow split, jittered per 3-row block
    bj = np.array([hash01(seed, k) for k in range(h // 3 + 2)])
    ub = lit + (bj[np.arange(h) // 3] - 0.5) * 0.08
    lit_side = u < ub[:, None]
    lvl = np.where(lit_side, 0, 2)
    lvl = np.where(~lit_side & (u < ub[:, None] + 0.06), 1, lvl)
    lvl = np.where(u > 0.86, 3, lvl)
    # vertical erosion grooves
    gx = [int(hash01(seed, 50 + k) * w) for k in range(max(3, w // 6))]
    for k, g in enumerate(gx):
        ln = 2 + int(hash01(seed, 80 + k) * band_h * 2)
        gy = int(hash01(seed, 90 + k) * h)
        sel = (xx == g) & (yy >= gy) & (yy < gy + ln)
        lvl = np.where(sel, np.minimum(lvl + 1, 3), lvl)
    # talus rubble
    talus_rows = (yy + yy0) > base - talus
    rub = talus_rows & (vnoise(h, w, 2, 2.5, seed + 17) > 0.7)
    lvl = np.where(rub & lit_side, 1, lvl)
    pal = np.array([[col(st[k]) for k in range(4)] for st in STRATA], np.uint8)
    rgb = pal[band, lvl]
    # ledge tops (rows whose row above is empty) + undercut shadow below each ledge
    top_rows = m & ~shift(m, 0, 1)
    rgb[top_rows & lit_side] = C('#fde3a8')
    rgb[top_rows & ~lit_side] = C('sand')
    under = shift(top_rows, 0, 1) & m & ~top_rows
    rgb[under & ~lit_side] = pal[band, 3][under & ~lit_side]
    ol = outline_of(m)
    rgb[ol & (u >= 0.5) & ~top_rows] = C('wine')
    rgb[ol & (u < 0.5) & ~top_rows] = C('#ffd7a0')
    if haze is not None and haze_t > 0:
        hz = np.array(col(haze), np.float32)
        rgb = np.clip(rgb.astype(np.float32) * (1 - haze_t) + hz * haze_t, 0, 255).astype(np.uint8)
    Y0, X0 = max(0, yy0), max(0, x0)
    sub = cv.a[Y0:yy0 + h, X0:x0 + w]
    mm = m[Y0 - yy0:, X0 - x0:][:sub.shape[0], :sub.shape[1]]
    rr = rgb[Y0 - yy0:, X0 - x0:][:sub.shape[0], :sub.shape[1]]
    sub[mm] = rr[mm]
    return (x0, yy0, w, h, top)


def draw_peaks(cv, specs, base, lit_c, mid_c, sh_c, seed, haze=None, haze_t=0.0, mist_c=None):
    """Jagged peaks [(apex_x, apex_y, half_w_left, half_w_right)] drawn back to front."""
    R = rng(seed)
    rim = mix(lit_c, 'cream', 0.4)
    for i, (ax, ay, wl, wr) in enumerate(specs):
        n = 5
        left = [(ax - wl, base)]
        for k in range(1, n):
            t = k / n
            left.append((ax - wl * (1 - t) + R.uniform(-2.5, 2.5), base - (base - ay) * t + R.uniform(-2, 2)))
        right = []
        for k in range(1, n):
            t = k / n
            right.append((ax + wr * t + R.uniform(-2.5, 2.5), ay + (base - ay) * t + R.uniform(-2, 2)))
        pts = left + [(ax, ay)] + right + [(ax + wr, base)]
        m = poly_mask(cv.h, cv.w, pts)
        cv.a[m] = C(mid_c)
        ridge = [(ax, ay)]
        x = ax
        for y in range(int(ay) + 3, base + 3, 3):
            x += R.uniform(-1.2, 2.0)
            ridge.append((x, y))
        lm = poly_mask(cv.h, cv.w, left + ridge + [(ridge[-1][0], base)])
        cv.a[lm & m] = C(lit_c)
        sm = poly_mask(cv.h, cv.w, [(ax + 1, ay + 1)] + right + [(ax + wr, base), (ax + wr * 0.45, base)])
        cv.a[sm & m & ~lm] = C(sh_c)
        # rock ribs on the lit face (darker gullies) and on the shadow face (lit spurs)
        for g in range(2 + wl // 12):
            t = R.uniform(0.2, 0.55)
            gx = ax - wl * R.uniform(0.2, 0.75) * t * 2
            gy = ay + (base - ay) * t
            cv.line(gx, gy, gx - R.uniform(1, 5), min(base, gy + R.uniform(6, 14)), mid_c)
        for g in range(1 + wr // 14):
            t = R.uniform(0.25, 0.6)
            sx = ax + wr * t * R.uniform(0.6, 1.0)
            sy = ay + (base - ay) * t
            cv.line(sx, sy, sx + R.uniform(-1, 3), min(base, sy + R.uniform(5, 12)), mid_c)
        # rim light along the sunny skyline
        for k in range(1, len(left)):
            a_, b_ = left[k - 1], left[k]
            cv.line(a_[0], a_[1], b_[0], b_[1], rim)
        cv.line(left[-1][0], left[-1][1], ax, ay, rim)
        cv.px(ax, ay, 'cream')
    if haze is not None:
        haze_rows(cv, base - 34, base, haze, 0.0, haze_t, steps=4)


def dead_tree(cv, x, y, h, seed, color='plum_brown', hi='mauve', blossom=False):
    """Gnarled leafless tree rooted at (x, y). blossom=True adds pink/white blossom clusters."""
    R = rng(seed)
    tips = []

    def branch(x0, y0, ang, ln, wd, depth):
        x1 = x0 + math.cos(ang) * ln
        y1 = y0 - math.sin(ang) * ln
        cv.line(x0, y0, x1, y1, color)
        if wd >= 2:
            cv.line(x0 + 1, y0, x1 + 1, y1, color)
            cv.line(x0 - 1, y0, x1 - 1, y1, hi)
        if depth <= 0 or ln < 2.5:
            tips.append((x1, y1))
            return
        nb = 2 if depth > 1 else R.integers(1, 3)
        for k in range(nb):
            da = R.uniform(0.35, 0.8) * (1 if k == 0 else -1) + R.uniform(-0.15, 0.15)
            branch(x1, y1, ang + da, ln * R.uniform(0.58, 0.75), wd - 1, depth - 1)
        if depth > 2 and R.random() < 0.5:
            branch(x1, y1, ang + R.uniform(-0.2, 0.2), ln * 0.5, wd - 1, depth - 2)

    branch(x, y, math.pi / 2 + R.uniform(-0.2, 0.2), h * 0.42, 2 if h > 14 else 1, 4 if h > 16 else 3)
    cv.px(x - 1, y, color)
    cv.px(x + 2, y, color)
    if blossom:
        for i, (tx, ty) in enumerate(tips):
            cc = ('bubblegum', 'cotton', 'magic', 'cream')[i % 4]
            cv.ellipse(tx, ty, 1.5, 1, 'rose')
            cv.px(tx, ty - 1, cc)
            cv.px(tx - 1, ty, cc)
            if i % 2 == 0:
                cv.px(tx + 1, ty - 1, 'cream')
    return tips


def flower_tuft(cv, x, y, seed, scale=1):
    R = rng(seed)
    for k in range(3 * scale):
        dx = R.integers(-2 * scale, 2 * scale + 1)
        hh = R.integers(1, 3 + scale)
        cv.vline(x + dx, y - hh, y, 'green' if k % 2 else 'leaf')
    for k in range(scale + 1):
        dx = R.integers(-2 * scale, 2 * scale + 1)
        cv.px(x + dx, y - 2 - R.integers(0, 2 + scale), FLOWERS[R.integers(0, len(FLOWERS))])


def vine(cv, p0, p1, sag, seed, leaf='leaf', stem='green', flowers=True, step=3):
    """A drooping vine (catenary-ish) from p0 to p1 with leaves and blossoms."""
    R = rng(seed)
    (x0, y0), (x1, y1) = p0, p1
    n = max(4, int(abs(x1 - x0) + abs(y1 - y0)))
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t + sag * 4 * t * (1 - t)
        pts.append((int(round(x)), int(round(y))))
    for (x, y) in pts:
        cv.px(x, y, stem)
    for i in range(1, len(pts) - 1, step):
        x, y = pts[i]
        cv.px(x, y + 1, leaf)
        if flowers and R.random() < 0.45:
            cv.px(x, y - 1, FLOWERS[R.integers(0, len(FLOWERS))])
        if R.random() < 0.25:   # hanging tendril
            ln = R.integers(2, 6)
            cv.vline(x, y + 1, y + ln, stem)
            cv.px(x, y + ln + 1, leaf)
    return pts


def crystalize(m, lam, TAG, NX, cell, seed, glow=None, levels=4.6, bias=-0.12, ao=None, dith=0.0):
    """Faceted translucent crystal colouring of a shape: Voronoi facets with flat shading, facet
    edges, darkest-step outline and a sun rim on the lit (left) edge. Returns (rgb, level map)."""
    h, w = m.shape
    R = rng(seed)
    cy, cx = cell
    seeds = []
    for gy in range(0, h + cy, cy):
        for gx in range(0, w + cx, cx):
            seeds.append((gx + R.uniform(-cx * 0.36, cx * 0.36), gy + R.uniform(-cy * 0.36, cy * 0.36)))
    S = np.array(seeds, np.float32)
    ys, xs = np.nonzero(m)
    cid_flat = np.zeros(len(ys), np.int64)
    for k0 in range(0, len(ys), 4096):          # chunked nearest-seed search
        sl = slice(k0, k0 + 4096)
        d = (xs[sl, None] - S[None, :, 0]) ** 2 + ((ys[sl, None] - S[None, :, 1]) * 1.15) ** 2
        cid_flat[sl] = np.argmin(d, axis=1)
    cid = np.full((h, w), -1, np.int64)
    cid[ys, xs] = cid_flat
    ncell = len(seeds)
    cnt = np.bincount(cid_flat, minlength=ncell).astype(np.float32) + 1e-6
    mlam = np.bincount(cid_flat, weights=lam[ys, xs], minlength=ncell) / cnt
    jit = np.array([hash01(seed, i) for i in range(ncell)], np.float32)
    ft = mlam[cid_flat] * 1.05 + (jit[cid_flat] - 0.5) * 0.4 + bias
    if ao is not None:                    # darken facets toward the base / right (cell centroid)
        cyy = np.bincount(cid_flat, weights=ys, minlength=ncell) / cnt
        cxx = np.bincount(cid_flat, weights=xs, minlength=ncell) / cnt
        ft = ft - (np.clip((cyy[cid_flat] - ao[0]) / ao[1], 0, 1) * ao[2]
                   + np.clip((cxx[cid_flat] - ao[3]) / ao[4], 0, 1) * ao[5])
    if glow is not None:
        gx, gy, gr, gk = glow
        dh = np.hypot(xs - gx, ys - gy)
        ft = ft + np.clip(1 - dh / gr, 0, 1) * gk
    ft = np.clip(ft, 0, 0.999)
    thr = bayer(h, w)
    lvl = np.clip(np.floor(ft * levels + (thr[ys, xs] - 0.5) * dith), 0, 4).astype(int)
    tags = TAG[ys, xs]
    pal = np.array([[col(CRYSTAL[k][j]) for j in range(5)] for k in range(4)], np.uint8)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[ys, xs] = pal[tags, lvl]
    edge = np.zeros((h, w), bool)
    edge[:, :-1] |= (cid[:, :-1] != cid[:, 1:])
    edge[:-1, :] |= (cid[:-1, :] != cid[1:, :])
    edge &= m
    L = np.zeros((h, w), np.int64)
    L[ys, xs] = lvl
    T = np.zeros((h, w), np.int64)
    T[ys, xs] = tags
    e_lit = edge & (L >= 2)
    e_dk = edge & (L < 2)
    rgb[e_lit] = pal[T[e_lit], np.minimum(L[e_lit] + 1, 4)]
    rgb[e_dk] = pal[T[e_dk], np.maximum(L[e_dk] - 1, 0)]
    ol = outline_of(m)
    rgb[ol] = pal[T[ol], 0]
    lit_edge = ol & (NX < -0.15)
    rgb[lit_edge] = pal[T[lit_edge], 4]
    return rgb, L


# ================================================================== the hive (geometry)
HW, HH = 164, 116
HEART = (82, 74)                 # local: the throne chamber opening where the burst starts
SPIRES = [  # (x_base, y_base, r0, y_tip, bend, tag, seed)
    (82, 76, 13, 2, 4, 0, 1),
    (60, 84, 11, 20, -6, 3, 2),
    (106, 82, 11, 12, 7, 1, 3),
    (40, 94, 9, 42, -8, 2, 4),
    (128, 94, 9, 34, 9, 3, 5),
    (70, 70, 6, 28, -3, 1, 6),
    (95, 72, 5, 30, 4, 2, 7),
    (20, 106, 6, 66, -6, 1, 8),
    (148, 104, 6, 60, 7, 0, 9),
    (52, 76, 4, 46, -4, 0, 10),
    (117, 74, 4, 40, 5, 0, 11),
]


@lru_cache(maxsize=None)
def hive_geo():
    bl = [
        (82, 122, 80, 38, 0, 0),
        (50, 106, 36, 30, 3, 3),
        (114, 104, 38, 32, 3, 1),
        (82, 94, 40, 34, 7, 0),
        (26, 114, 24, 16, 1, 2),
        (140, 112, 24, 18, 1, 2),
        (66, 112, 20, 14, 5, 0),
        (100, 112, 22, 14, 5, 3),
    ]
    for (xb, yb, r0, yt, bend, tag, sd) in SPIRES:
        n = max(6, int((yb - yt) / 2.2))
        for k in range(n):
            t = k / (n - 1)
            y = yb + (yt - yb) * t
            r = r0 * (1 - t) ** 0.85 + 0.7
            knob = 1 + 0.16 * math.sin(t * 17 + sd * 2.1) + 0.08 * math.sin(t * 41 + sd)
            x = xb + bend * t * t + math.sin(t * 9 + sd) * 0.8
            ry = max(1.6, (yb - yt) / n * 1.25)
            bl.append((x, y, max(0.7, r * knob), ry, 6 + 4 * (1 - t), tag))
    m, NX, NY, NZ, TAG = blob_union(HH, HW, bl)
    m[HH - 1:, :] = m[HH - 1:, :]
    # lumpy termite texture -> perturb normals
    tex = fbm(HH, HW, 3, 5, 77) - 0.5
    NX = NX + tex * 0.5
    NY = NY + (fbm(HH, HW, 2.5, 6, 78) - 0.5) * 0.8
    lam = lambert(NX, NY, NZ)
    # honeycomb holes on a jittered hex grid
    R = rng(5150)
    holes = []
    ins = m & shift(m, 3, 0) & shift(m, -3, 0) & shift(m, 0, 3) & shift(m, 0, -4)
    for gy in range(6, HH - 10, 8):
        for gx in range(4 + (gy // 8 % 2) * 5, HW - 4, 10):
            x = gx + R.integers(-2, 3)
            y = gy + R.integers(-2, 3)
            if not (0 <= x < HW and 0 <= y < HH) or not ins[y, x]:
                continue
            if math.hypot(x - HEART[0], y - HEART[1]) < 15:
                continue
            if NZ[y, x] < 0.3 or R.random() < 0.18:
                continue
            # size from the local body thickness
            row = m[y]
            l = x
            while l > 0 and row[l - 1]:
                l -= 1
            r = x
            while r < HW - 1 and row[r + 1]:
                r += 1
            span = r - l
            if span < 8:
                continue
            hw = 1 if span < 14 else (2 if span < 40 or R.random() < 0.6 else 3)
            hh = hw + 1 + R.integers(0, 2)
            holes.append((x, y, hw, hh))
    tunnels = [(30, HH - 1, 6, 12), (134, HH - 1, 5, 10), (58, HH - 1, 5, 9), (108, HH - 1, 6, 11)]
    return dict(m=m, NX=NX, NY=NY, NZ=NZ, TAG=TAG, lam=lam, holes=holes, tunnels=tunnels)


def arch_mask(h, w, x, y, hw, hh):
    """Arched opening: centre-bottom (x, y), half-width hw, height hh (round top)."""
    yy, xx = np.mgrid[0:h, 0:w]
    top = y - hh + hw
    m = (np.abs(xx - x) <= hw) & (yy <= y) & (yy >= top)
    m |= ((xx - x) ** 2 + (yy - top) ** 2 <= (hw + 0.4) ** 2)
    return m


@lru_cache(maxsize=None)
def hive_sprite(state):
    """RGBA-ish (rgb, mask) of the hive in 'dark' (chitin) or 'bloom' (crystal) state."""
    g = hive_geo()
    m, lam, TAG = g['m'], g['lam'], g['TAG']
    cv = Canvas(HW, HH, KEY)
    a = cv.a
    thr = bayer(HH, HW)
    ol = outline_of(m)
    lit_edge = ol & (g['NX'] < -0.15)
    if state == 'dark':
        tex = fbm(HH, HW, 1.5, 3, 91)
        t = np.clip(lam * 0.85 + (tex - 0.5) * 0.35, 0, 0.999)
        lvl = np.clip(np.floor(t * len(CHITIN) + (thr - 0.5) * 0.8), 0, len(CHITIN) - 1).astype(int)
        rgb = ramp_arr(CHITIN)[lvl]
        a[m] = rgb[m]
        # ridges: horizontal growth rings
        rings = (((np.arange(HH)[:, None] + (tex * 6).astype(int)) % 7) == 0) & m & (thr > 0.35)
        a[rings & (lvl > 1)] = C('basalt')
        a[ol] = C('ink')
        a[lit_edge] = C('#7d7064')       # warm golden rim on the sunny side
        a[lit_edge & (thr > 0.6)] = C('#9c8468')
    else:
        rgb, _ = crystalize(m, lam, TAG, g['NX'], (6, 7), 2024, glow=(HEART[0], HEART[1], 40, 0.2))
        a[m] = rgb[m]
    # holes / windows
    for i, (x, y, hw, hh) in enumerate(g['holes']):
        hm = arch_mask(HH, HW, x, y, hw, hh) & m
        if state == 'dark':
            a[hm] = C('ink')
            a[shift(hm, 0, 1) & ~hm & m] = C('#4d514c')
        elif hash01(41, i) > 0.3:
            # crystal cavity: a soft recessed facet (mid tone, dark lip on top, glint)
            tg = int(TAG[y, x])
            a[hm] = C(CRYSTAL[tg][2])
            a[hm & ~shift(hm, 0, 1)] = C(CRYSTAL[tg][0])
            a[hm & ~shift(hm, 0, -1)] = C(CRYSTAL[tg][3])
            a[y - hh + hw + 1, x - max(0, hw - 1)] = C(CRYSTAL[tg][4])
        else:
            a[hm] = C('gold')
            inner = hm & shift(hm, 1, 0) & shift(hm, -1, 0) & shift(hm, 0, -1)
            a[inner] = C('butter')
            a[hm & ~shift(hm, 0, 1)] = C('amber')
            a[shift(hm, 0, -1) & ~hm & m] = C('ink')
    # heart chamber
    hx, hy = HEART
    hm = arch_mask(HH, HW, hx, hy + 8, 8, 18)
    if state == 'dark':
        a[hm] = C('ink')
        inner = arch_mask(HH, HW, hx, hy + 8, 5, 13)
        a[inner] = C('#1f2a22')
        a[shift(hm, 0, -1) & ~hm] = C('#5a5e58')
    else:
        a[hm] = C('cotton')
        inner = arch_mask(HH, HW, hx, hy + 8, 5, 13)
        a[inner] = C('cream')
        a[shift(hm, 0, -1) & ~hm] = C('magic')
        a[shift(hm, 0, -2) & ~shift(hm, 0, -1)] = C('raspberry')
    # base tunnels
    for (x, y, hw, hh) in g['tunnels']:
        tm = arch_mask(HH, HW, x, y, hw, hh)
        if state == 'dark':
            a[tm] = C('ink')
            a[tm & ~shift(tm, 0, -2)] = C('basalt')
        else:
            a[tm] = C('#5a3a2a')
            a[arch_mask(HH, HW, x, y, hw - 2, hh - 2)] = C('amber')
            a[arch_mask(HH, HW, x, y, hw - 3, hh - 4)] = C('butter')
            a[shift(tm, 0, -1) & ~tm & m] = C('ink')
    if state == 'dark':
        # slime drips from some holes
        R = rng(66)
        for (x, y, hw, hh) in g['holes']:
            if R.random() < 0.35:
                ln = R.integers(2, 6)
                for k in range(ln):
                    if 0 <= y + 1 + k < HH and m[y + 1 + k, x]:
                        a[y + 1 + k, x] = C(SLIME[1] if k < ln - 1 else SLIME[3])
    else:
        # vines draped between spires + flowers on the body
        R = rng(77)
        anchors = [(14, 92), (36, 62), (52, 52), (66, 40), (82, 30), (98, 38), (114, 52), (128, 60), (150, 88)]
        for i in range(len(anchors) - 1):
            vine(cv, anchors[i], anchors[i + 1], 6 + R.integers(0, 5), 300 + i)
        anchors2 = [(22, 104), (44, 96), (64, 98), (100, 98), (120, 96), (142, 104)]
        for i in range(len(anchors2) - 1):
            if i == 2:
                continue
            vine(cv, anchors2[i], anchors2[i + 1], 4, 400 + i, step=2)
        # moss + flowers at the foot
        for k in range(26):
            x = 6 + int(hash01(9, k) * (HW - 12))
            col_ = np.nonzero(m[:, x])[0]
            if len(col_) == 0:
                continue
            flower_tuft(cv, x, HH - 1, 900 + k)
        a[~m & ~np.all(a == np.array(KEY, np.uint8), axis=2)] = a[~m & ~np.all(a == np.array(KEY, np.uint8), axis=2)]
    return to_layer(cv)


# ================================================================== SHOT A (wide crater)
SKY_CLEAR = ['#3a4796', 'blue', '#7a62b0', 'violet', 'rose', 'salmon', 'coral', '#fbb36a', 'sand', '#ffd98a']
SKY_GLOOM = ['#2c3332', '#384039', '#454c42', '#555b4a', '#686a50', '#7c7855', '#938659', '#a8925c', '#b99c60', '#c6a565']
SUN_A = (44, 84)
HAZE_A = '#f2b98a'
CAM_A = 36          # px of pan of the near-ground layer during shot A
P_PEAK, P_FARM, P_MID, P_NEAR, P_FG = 0.12, 0.28, 0.5, 0.85, 1.2
HIVE_POS = (204 - HEART[0], 103 - HEART[1])   # mid-layer coords of the hive's top-left at cam 0


def cam_a(f):
    return cam_ease(f, 72, 215, 0, CAM_A)


@lru_cache(maxsize=None)
def sky_a(state):
    cv = Canvas(W, 124)
    fx.sky(cv, SKY_CLEAR if state == 'bloom' else SKY_GLOOM, 0, 124)
    sx, sy = SUN_A
    if state == 'bloom':
        fx.glow(cv, sx, sy, 70, 'amber', 0.28, rings=5)
        fx.glow(cv, sx, sy, 30, 'gold', 0.35, rings=4)
        cv.circle(sx, sy, 8, 'butter')
        cv.circle(sx, sy, 6, 'cream')
        cv.circle(sx - 1, sy - 1, 3, 'white')
    else:
        fx.glow(cv, sx, sy, 40, '#8c8a5a', 0.25, rings=4)
        cv.circle(sx, sy, 7, '#d8d49c')
        cv.circle(sx, sy, 5, '#e8e4b4')
    return cv


@lru_cache(maxsize=None)
def cloud_sprite(w, seed, state):
    """Puffy golden-hour cloud: flat-ish base, round bumps, lit from below-left by the low sun."""
    R = rng(seed)
    n = max(3, w // 9)
    step = w / n
    h = int(step * 1.6) + 6
    cv = Canvas(w + 6, h, KEY)
    bl = []
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi) ** 0.7
        r = step * (0.55 + 0.55 * bump) * (0.85 + 0.3 * R.random())
        bl.append((3 + u * w, h - 3 - r * 0.75, r * 1.05, r * 0.85, 0, 0))
    bl.append((3 + w / 2, h - 4, w / 2 + 1, 3, 0, 0))
    m, NX, NY, NZ, _ = blob_union(h, w + 6, bl)
    m[h - 1:] = False
    lam = np.clip((NX * -0.55 + NY * 0.7 + NZ * 0.45 + 1) / 2, 0, 0.999)
    ramp = (('#8a6aa8', 'lilac', 'peach', 'blush', 'cream') if state == 'bloom'
            else ('#454b45', '#565d52', '#6a6e5c', '#7f7f66', '#949170'))
    lvl = ramp_idx(lam, 5, 0, 0, h, w + 6, 0.8)
    rgb = ramp_arr(ramp)[lvl]
    cv.a[m] = rgb[m]
    ol = outline_of(m)
    bottom = ol & ~shift(m, 0, -1)
    cv.a[bottom] = C('gold' if state == 'bloom' else '#a89f62')
    return to_layer(cv)


CLOUDS_A = [  # (x, y, w, seed, speed px/frame)
    (90, 30, 54, 1, 0.10), (196, 18, 70, 2, 0.07), (8, 50, 40, 3, 0.12), (262, 44, 44, 4, 0.09),
    (140, 62, 30, 5, 0.14),
]


def draw_clouds(c, f, state, specs, ox=0):
    for (x, y, w, sd, sp) in specs:
        lay = cloud_sprite(w, sd, state)
        span = W + w + 40
        xx = int((x + f * sp + ox) % span) - w - 20
        comp(c, lay, xx, y)


@lru_cache(maxsize=None)
def peaks_a(state):
    w = W + 12
    cv = Canvas(w, 124, KEY)
    specs_back = [(20, 70, 26, 22), (64, 62, 24, 26), (112, 74, 30, 24), (170, 66, 24, 30),
                  (226, 60, 30, 26), (276, 70, 22, 26), (318, 64, 26, 20)]
    specs_front = [(-4, 86, 20, 24), (40, 84, 24, 20), (90, 90, 22, 28), (140, 86, 26, 22),
                   (200, 88, 22, 26), (250, 82, 26, 22), (300, 88, 22, 28)]
    if state == 'bloom':
        draw_peaks(cv, specs_back, 116, '#e7a3a8', '#c08c9c', '#9a7590', 11, haze=HAZE_A, haze_t=0.5)
        draw_peaks(cv, specs_front, 118, 'salmon', 'mauve', 'plum_brown', 12, haze=HAZE_A, haze_t=0.35)
    else:
        draw_peaks(cv, specs_back, 116, '#e7a3a8', '#c08c9c', '#9a7590', 11, haze='#8c8a5a', haze_t=0.6)
        draw_peaks(cv, specs_front, 118, 'salmon', 'mauve', 'plum_brown', 12, haze='#8c8a5a', haze_t=0.45)
        gloom(cv, 0.9)
    return to_layer(cv)


@lru_cache(maxsize=None)
def farmesa_a(state):
    w = W + 16
    cv = Canvas(w, 130, KEY)
    # low far floor band
    cv.gradient(0, 112, w, 18, ['#f0b27c', 'sand', 'clay'], soft=0.6)
    for (cx, base, tiers, sd) in [(16, 116, [(16, 6), (12, 6)], 21), (98, 117, [(20, 7), (14, 5)], 22),
                                  (246, 116, [(14, 8), (10, 4)], 23), (312, 117, [(18, 6), (12, 6)], 24)]:
        draw_mesa(cv, cx, base, tiers, sd, band_h=3, haze=HAZE_A, haze_t=0.42, talus=5)
        shadow_poly(cv, [(cx, base - 2), (cx + 50, base - 2), (cx + 44, base + 1), (cx, base + 1)], 0.25)
    if state == 'dark':
        gloom(cv, 0.9)
    return to_layer(cv)


def hive_world_mask():
    g = hive_geo()
    return g['m']


@lru_cache(maxsize=None)
def mid_a(state):
    """Crater floor + mid mesas + the hive with its long shadow (parallax P_MID)."""
    w = W + int(CAM_A * P_MID) + 8
    cv = Canvas(w, H, KEY)
    # crater floor
    cv.gradient(0, 120, w, 60, ['#f6c27e', 'sand', 'clay', 'copper'], soft=0.7)
    R = rng(31)
    n = fbm(60, w, 2, 14, 32)
    a = cv.a[120:180]
    a[(n > 0.62)] = C('clay')
    a[(n < 0.33) & (bayer(60, w) > 0.4)] = C('#f6c27e')
    for k in range(40):        # cracks + pebbles
        x = R.integers(0, w)
        y = R.integers(124, 178)
        ln = R.integers(2, 6)
        cv.hline(x, x + ln, y, 'rust' if y > 150 else 'copper')
        if R.random() < 0.5:
            cv.px(x + ln + 1, y + 1, 'rust')
    for k in range(30):
        x = R.integers(0, w)
        y = R.integers(122, 178)
        cv.px(x, y, 'cream')
        cv.px(x + 1, y, 'copper')
    # mid mesas (left + right) with long shadows to the right
    hx, hy = HIVE_POS
    for (cx, base, tiers, sd, lt) in [(30, 128, [(22, 14), (16, 10), (10, 6)], 41, 0.4),
                                      (310, 126, [(20, 16), (14, 12)], 42, 0.36)]:
        x0, y0, mw, mh, top = draw_mesa(cv, cx, base, tiers, sd, band_h=4, haze=HAZE_A, haze_t=0.15)
        shadow_poly(cv, [(cx, base - 3), (cx + mw // 2 + 60, base - 3), (cx + mw // 2 + 40, base + 3),
                         (cx, base + 3)], 0.3)
    # hive shadow on the floor (long, to the right)
    shadow_poly(cv, [(hx + 70, hy + HH - 3), (hx + HW + 90, hy + HH - 3), (hx + HW + 60, hy + HH + 4),
                     (hx + 80, hy + HH + 4)], 0.3)
    # small far dead trees on the crater floor
    for (x, y, h_, sd) in [(118, 128, 12, 51), (104, 131, 9, 52), (286, 131, 11, 53), (72, 134, 10, 54)]:
        dead_tree(cv, x, y, h_, sd, color='plum_brown', hi='mauve', blossom=(state == 'bloom'))
    if state == 'bloom':
        R = rng(55)
        # grass + flower carpet racing out from the hive foot
        for k in range(70):
            x = hx + 10 + R.integers(-30, HW + 40)
            y = hy + HH + R.integers(-2, 8)
            if 0 <= x < w:
                flower_tuft(cv, x, y, 1000 + k)
        cv.dither(hx - 10, hy + HH - 1, HW + 30, 4, 'green', 0.45)
        cv.dither(hx - 20, hy + HH + 2, HW + 50, 3, 'leaf', 0.3)
        # moss on the mesa ledges
        for (x, y) in [(16, 104), (28, 92), (38, 104), (300, 99), (312, 87), (322, 99)]:
            cv.hline(x - 3, x + 3, y, 'leaf')
            cv.px(x, y - 1, 'bubblegum')
    if state == 'dark':
        gloom(cv, 0.85)
    # the hive
    comp(cv, hive_sprite(state), hx, hy)
    return to_layer(cv)


POOLS_A = [(150, 166, 34, 5), (270, 156, 26, 4), (40, 158, 18, 3)]     # near-layer coords (cx, cy, rx, ry)
TREES_A = [(112, 160, 34, 61), (300, 152, 26, 62), (8, 150, 22, 63)]


@lru_cache(maxsize=None)
def near_a(state):
    w = W + int(CAM_A * P_NEAR) + 8
    cv = Canvas(w, H, KEY)
    # rocky rise with an irregular top edge
    top = 146 + (noise1(w, 22, 71) * 10 - 5).astype(int) + (noise1(w, 5, 72) * 3).astype(int)
    yy = np.arange(H)[:, None]
    m = yy >= top[None, :]
    nn = fbm(H, w, 2, 6, 73)
    rgb = np.where((nn > 0.55)[:, :, None], C('copper'), C('clay'))
    rgb = np.where(((nn < 0.35) & (bayer(H, w) > 0.5))[:, :, None], C('rust'), rgb)
    rgb = np.where((yy > 168)[:, :, None] & (nn > 0.45)[:, :, None], C('rust'), rgb)
    cv.a[m] = rgb[m]
    edge = m & ~shift(m, 0, 1)
    cv.a[edge] = C('#f6c27e')
    cv.a[shift(edge, 0, 1) & m] = C('sand')
    # strata streaks
    R = rng(74)
    for k in range(50):
        x = R.integers(0, w)
        y = R.integers(150, 180)
        if y > top[min(w - 1, x)] + 2:
            cv.hline(x, x + R.integers(3, 9), y, 'rust' if R.random() < 0.6 else 'sand')
    # rocks
    for k, (x, y, r) in enumerate([(70, 168, 6), (214, 172, 8), (236, 162, 4), (330, 170, 7), (20, 176, 5)]):
        cv.ellipse(x, y, r + 1, r * 0.7 + 1, 'wine')
        cv.ellipse(x, y, r, r * 0.7, 'copper')
        cv.ellipse(x - 1, y - 1, r * 0.6, r * 0.4, 'clay')
        cv.px(x - r // 2, y - r // 2, 'sand')
    # pools
    for i, (cx, cy, rx, ry) in enumerate(POOLS_A):
        cv.ellipse(cx, cy, rx + 2, ry + 1, 'rust')
        cv.ellipse(cx, cy - 1, rx + 1, ry, 'wine' if state == 'dark' else 'clay')
        if state == 'dark':
            cv.ellipse(cx, cy, rx, ry, SLIME[1])
            cv.ellipse(cx - 2, cy - 1, rx - 4, ry - 1.5, SLIME[2])
            cv.hline(cx - rx // 2, cx + rx // 3, cy - ry + 1, SLIME[3])
        else:
            cv.ellipse(cx, cy, rx, ry, 'teal')
            cv.ellipse(cx, cy - 1, rx - 2, ry - 1, 'aqua')
            cv.ellipse(cx - 3, cy - 1, rx - 8, ry - 2.5, 'turquoise')
            cv.hline(cx - rx + 4, cx - rx // 2, cy - ry + 1, 'ice')
            # sky reflection streak
            cv.hline(cx + 2, cx + rx // 2, cy + 1, 'ice')
    # dead trees
    for (x, y, h_, sd) in TREES_A:
        dead_tree(cv, x, y, h_, sd, color='wine', hi='mauve', blossom=(state == 'bloom'))
    if state == 'bloom':
        for k in range(46):
            x = int(hash01(81, k) * w)
            flower_tuft(cv, x, int(top[min(w - 1, x)]) + 1 + int(hash01(82, k) * 12), 2000 + k,
                        scale=1 + (k % 3 == 0))
        for i, (cx, cy, rx, ry) in enumerate(POOLS_A):
            for k in range(5):
                flower_tuft(cv, cx - rx - 2 + k * (2 * rx + 4) // 4, cy + 1, 2100 + i * 9 + k)
    else:
        gloom(cv, 0.6)
    return to_layer(cv), top


FG_ROCKS_A = [  # polygons in fg-layer coords
    [(0, 180), (0, 132), (6, 126), (12, 134), (18, 128), (24, 146), (34, 152), (46, 166), (54, 180)],
    [(300, 180), (312, 150), (322, 140), (330, 146), (338, 128), (346, 134), (360, 140), (372, 180)],
]


@lru_cache(maxsize=None)
def fg_a(state):
    w = W + int(CAM_A * P_FG) + 10
    cv = Canvas(w, H, KEY)
    for i, pts in enumerate(FG_ROCKS_A):
        m = poly_mask(H, w, pts)
        nn = fbm(H, w, 3, 3, 90 + i)
        ys, xs = np.nonzero(m)
        cx = np.mean([p[0] for p in pts])
        lit = (xs < cx - 4 + (nn[ys, xs] - 0.5) * 12)
        rgb = np.where(lit[:, None], C('rust'), C('wine'))
        rgb[(nn[ys, xs] > 0.68)] = C('berry')
        cv.a[ys, xs] = rgb
        ol = outline_of(m)
        cv.a[ol] = C('#3a1e26')
        top_edge = m & ~shift(m, 0, 1)
        cv.a[top_edge & (np.arange(w)[None, :] < cx)] = C('coral')
    # dead branches poking up from the rocks
    dead_tree(cv, 14, 130, 30, 95, color='#3a1e26', hi='wine', blossom=(state == 'bloom'))
    dead_tree(cv, 344, 132, 26, 96, color='#3a1e26', hi='wine', blossom=(state == 'bloom'))
    if state == 'bloom':
        vine(cv, (0, 140), (24, 148), 4, 97)
        vine(cv, (314, 150), (346, 136), 5, 98)
        for k in range(10):
            flower_tuft(cv, 4 + k * 5, 180 - (k % 3), 3000 + k, 2)
            flower_tuft(cv, 304 + k * 6, 178 - (k % 2) * 3, 3100 + k, 2)
    else:
        gloom(cv, 0.5)
    return to_layer(cv)


def hive_screen(f):
    cam = cam_a(f)
    hx = HIVE_POS[0] - int(round(cam * P_MID))
    return hx, HIVE_POS[1]


def compose_a(f, state, cam, sh):
    c = sky_a(state).copy() if False else Canvas(W, H)
    c.a[:124] = sky_a(state).a
    draw_clouds(c, f, state, CLOUDS_A)
    comp(c, peaks_a(state), -int(round(cam * P_PEAK)), sh)
    comp(c, farmesa_a(state), -int(round(cam * P_FARM)), sh)
    if state == 'dark':
        # haze veil behind the hive
        pass
    mx = -int(round(cam * P_MID))
    comp(c, mid_a(state), mx, sh)
    nl, top = near_a(state)
    nx = -int(round(cam * P_NEAR))
    comp(c, nl, nx, sh)
    hx, hy = HIVE_POS[0] + mx, HIVE_POS[1] + sh
    g = hive_geo()
    if state == 'dark':
        # faint green glows in the holes (NOT eyes: soft, uneven, low in the hole)
        for i, (x, y, hw, hh) in enumerate(g['holes']):
            ph = (f / 40.0 + hash01(3, i)) % 1.0
            k = 0.5 + 0.5 * math.sin(ph * 6.283)
            if f >= 120:
                k = min(1.0, k + seg(f, 120, 143) * 0.8)
            if k > 0.45:
                c.px(hx + x, hy + y, '#3c7a3c' if k < 0.8 else '#5fb14a')
                if hw >= 2 and k > 0.7:
                    c.px(hx + x - 1, hy + y, '#2f5a32')
        for i, (x, y, hw, hh) in enumerate(g['tunnels']):
            k = 0.5 + 0.5 * math.sin(f / 23.0 + i)
            fx.glow(c, hx + x, hy + y - 2, hw + 3, '#3c8a3c', 0.25 + 0.2 * k, rings=3)
        # heart: a slow sickly pulse that swells before the transform
        hk = 0.35 + 0.15 * math.sin(f / 9.0) + 0.6 * ease_in(seg(f, 118, 144))
        fx.glow(c, hx + HEART[0], hy + HEART[1], 9 + int(10 * seg(f, 118, 144)), '#6bd04b', hk, rings=4)
        # slime pools bubbling
        for i, (cx, cy, rx, ry) in enumerate(POOLS_A):
            for k in range(3):
                per = 26 + k * 7
                t = ((f + int(hash01(i, k, 5) * per)) % per) / per
                bx = nx + cx + int((hash01(i, k, 6) - 0.5) * rx * 1.4)
                by = sh + cy + int((hash01(i, k, 7) - 0.5) * ry)
                if t < 0.6:
                    r = 0.5 + t * 2.2
                    c.circle(bx, by - int(t * 2), r, SLIME[3], fill=False)
                    c.px(bx - 1, by - int(t * 2) - 1, SLIME[4])
                elif t < 0.7:
                    c.px(bx - 2, by - 2, SLIME[4])
                    c.px(bx + 2, by - 3, SLIME[4])
            # sickly fumes rising
            fx.steam(c, nx + cx, sh + cy - 2, f, color='#93a46a', n=2, height=12, spread=4, seed=i)
        # drifting haze wisps
        hz = haze_field()
        xo = 100 - int(f * 0.4)
        k = hz[:, xo:xo + W, None]
        c.a[...] = (c.a * (1 - k) + np.array(col('#8b9273'), np.float32) * k).astype(np.uint8)
    else:
        # warm pools: glints + ripples
        for i, (cx, cy, rx, ry) in enumerate(POOLS_A):
            for k in range(3):
                ph = (f + k * 11 + i * 5) % 30
                if ph < 12:
                    gx = nx + cx - rx // 2 + int(hash01(i, k, (f + k * 11) // 30) * rx)
                    c.px(gx, sh + cy - 1 + k % 2, 'white')
            rr = ((f + i * 13) % 40) / 40.0
            c.ellipse(nx + cx + 4, sh + cy, 2 + rr * (rx - 6), 0.5 + rr * (ry - 1), 'ice', fill=False)
        # windows / tunnels glow warm, heart radiates
        for i, (x, y, hw, hh) in enumerate(g['tunnels']):
            k = 0.5 + 0.5 * math.sin(f / 7.0 + i * 1.7)
            fx.glow(c, hx + x, hy + y - 3, hw + 5, 'amber', 0.18 + 0.12 * k, rings=3)
        fx.glow(c, hx + HEART[0], hy + HEART[1], 18, 'magic', 0.3 + 0.08 * math.sin(f / 6.0), rings=4)
        # facet glints catching the sun
        for i in range(10):
            per = 34
            t0 = int(hash01(11, i) * per)
            if (f + t0) % per < 10:
                j = int(hash01(12, i, (f + t0) // per) * len(g['holes']))
                x, y = g['holes'][j][:2]
                fx.sparkle(c, hx + x - 3, hy + y - 3, f + t0, 'white', period=10, size=2)
        # drifting petals
        for i in range(18):
            sp = 0.35 + hash01(13, i) * 0.3
            px_ = int((hash01(14, i) * (W + 40) + f * sp) % (W + 40)) - 20
            py_ = int((hash01(15, i) * 140 + f * 0.25 + math.sin(f / 12.0 + i) * 3) % 150) + 20
            c.px(px_, py_, ('bubblegum', 'cotton', 'lilac', 'cream')[i % 4])
        # warm golden light
        fx.light_rect(c, 0, 40, 160, 100, 'amber', 0.07, fade='right')
    fl = fg_a(state)
    comp(c, fl, -int(round(cam * P_FG)), sh)
    return c


@lru_cache(maxsize=None)
def haze_field():
    w = W + 100
    n = fbm(H, w, 6, 40, 404)
    yy = np.arange(H)[:, None]
    env = np.clip(1 - np.abs(yy - 96) / 46.0, 0, 1)
    k = np.clip((n - 0.45) * 1.6, 0, 1) * env * 0.42
    return (np.round(k * 6) / 6).astype(np.float32)          # stepped fog bands, no dot dither


def ring_r(f):
    return 230 * ease_out(seg(f, T_FORM, 206)) ** 0.9 if f >= T_FORM else -1


RING_COLS = ('turquoise', 'bubblegum', 'lime', 'lavender')
_YY, _XX = np.mgrid[0:H, 0:W]


def render_a(f):
    cam = cam_a(f)
    sh = 0
    if 128 <= f < T_FORM:          # tremor before the burst
        sh = (1 if (f // 2) % 2 else 0) if f >= 134 else 0
    if T_FORM <= f < T_FORM + 6:
        sh = [1, -1, 1, 0, 1, 0][f - T_FORM]
    r = ring_r(f)
    hx, hy = hive_screen(f)
    cx, cy = hx + HEART[0], hy + HEART[1] + sh
    if r < 0:
        c = compose_a(f, 'dark', cam, sh)
    elif r >= 230:
        c = compose_a(f, 'bloom', cam, sh)
    else:
        dark = compose_a(f, 'dark', cam, sh)
        bloom = compose_a(f, 'bloom', cam, sh)
        d = np.hypot(_XX - cx, (_YY - cy) * 1.1)
        inside = d < r
        c = dark
        c.a[inside] = bloom.a[inside]
        # rainbow ring with a white leading edge
        ang = (np.arctan2(_YY - cy, _XX - cx) / 6.283 * 8 + f / 3.0) % 4
        band = (d >= r - 5) & (d < r - 1)
        cc = ramp_arr(RING_COLS)[ang.astype(int)]
        c.a[band] = cc[band]
        edge = (d >= r - 1) & (d < r + 0.6)
        c.a[edge] = C('white')
        # soft light ahead of the front
        fx.glow(c, cx, cy, int(r + 10), 'cream', 0.0, rings=1)
        # sparkles riding the ring
        for i in range(22):
            a_ = hash01(21, i) * 6.283 + f * 0.01
            rr = r - 3 - hash01(22, i) * 10
            fx.sparkle(c, int(cx + math.cos(a_) * rr), int(cy + math.sin(a_) * rr / 1.1), f + i * 3,
                       ('white', 'ice', 'cotton', 'butter')[i % 4], period=12, size=2)
    if T_FORM <= f < T_FORM + 10:
        k = 1 - seg(f, T_FORM, T_FORM + 10)
        fx.glow(c, cx, cy, 40, 'white', 0.9 * k, rings=5)
    flash(c, f, T_FORM, 'white', 0.45)
    # balloon drifting left of the hive in the golden sky
    bx = int(round(lerp(92, 118, seg(f, 72, 215))))
    by = int(round(lerp(70, 63, seg(f, 72, 215))))
    draw_balloon(c, bx, by, f, 's', flame=(f // 30) % 3 == 0)
    # callouts
    mx = -int(round(cam * P_MID))
    callout(c, 310 + mx + 4, 96, 'THE BADLANDS', f, 90, 150, side='up', length=14)
    sp_x, sp_y = hx + 112, hy + 16
    callout(c, sp_x, sp_y + sh, 'CHANGELING KINGDOM', f, 108, 156, side='left', length=10)
    callout(c, sp_x, sp_y, 'THE HIVE IN BLOOM', f, 162, 208, side='left', length=10)
    atlas_card(c, f, 100, 208, 'CHANGELING KINGDOM', ['A HIVE IN THE BADLANDS', 'NOW IN FULL BLOOM'])
    return c


# ================================================================== SHOT B (close-up + Tartarus)
PAN_B = 244                      # camera (main-layer px) at the close-up; 0 = gate view
P_SKYB, P_FARB, P_MIDB, P_FGB = 0.08, 0.22, 0.5, 1.3
WB = W + PAN_B + 4               # main layer width
GX, GFLOOR = 166, 166            # gate centre x (world), canyon floor y at the gate
EMB = (166, 98)                  # emblem centre
HX0 = 280                        # close-up crystal canvas origin (world x)
SUN_B = (40, 44)


def cam_b(f):
    if f < 216:
        return PAN_B
    if f < T_TART:
        return lerp(PAN_B, 8, cruise(seg(f, 216, 250), 0.15))   # director pass: even pan (~8 px/f)
    return lerp(8, 0, smooth(seg(f, T_TART, 287)))


def ground_b(x):
    """Ground line of the main layer (world x)."""
    if x < 236:
        return GFLOOR
    if x < 300:
        return int(round(lerp(GFLOOR, 150, smooth((x - 236) / 64))))
    return 150


@lru_cache(maxsize=None)
def sky_b():
    w = W + int(PAN_B * P_SKYB) + 4
    cv = Canvas(w, H)
    fx.sky(cv, SKY_CLEAR, 0, 150, 0, w)
    sx, sy = SUN_B
    fx.glow(cv, sx, sy, 70, 'amber', 0.3, rings=5)
    fx.glow(cv, sx, sy, 28, 'gold', 0.35, rings=4)
    cv.circle(sx, sy, 8, 'butter')
    cv.circle(sx, sy, 6, 'cream')
    cv.circle(sx - 1, sy - 1, 3, 'white')
    return cv


@lru_cache(maxsize=None)
def far_b():
    w = W + int(PAN_B * P_FARB) + 8
    cv = Canvas(w, H, KEY)
    specs = [(10, 70, 24, 22), (60, 60, 26, 24), (120, 72, 26, 26), (180, 62, 28, 24), (240, 74, 24, 26),
             (300, 64, 26, 24), (350, 70, 24, 22), (395, 62, 24, 20)]
    draw_peaks(cv, specs, 124, '#e7a3a8', '#c08c9c', '#9a7590', 61, haze=HAZE_A, haze_t=0.5)
    cv.gradient(0, 118, w, 20, ['#f0b27c', 'sand'], soft=0.6)
    for (cx, base, tiers, sd) in [(200, 122, [(16, 7), (11, 5)], 62), (300, 123, [(20, 6), (13, 6)], 63),
                                  (380, 122, [(14, 8)], 64)]:
        draw_mesa(cv, cx, base, tiers, sd, band_h=3, haze=HAZE_A, haze_t=0.4, talus=5)
    return to_layer(cv)


@lru_cache(maxsize=None)
def mid_b():
    """Behind the main layer: dusky canyon walls on the left, more of the hive + mesas on the right."""
    w = W + int(PAN_B * P_MIDB) + 8
    cv = Canvas(w, H, KEY)
    # back canyon walls (cool, bleak, lit rim)
    pts = [(0, H), (0, 58), (14, 54), (30, 62), (46, 50), (62, 56), (80, 66), (96, 60), (110, 74),
           (124, 90), (138, 120), (150, H)]
    m = poly_mask(H, w, pts)
    nn = fbm(H, w, 4, 6, 120)
    rgb = np.where((nn > 0.55)[:, :, None], C('#4a3e4c'), C('#3e3442'))
    cv.a[m] = rgb[m]
    top = m & ~shift(m, 0, 1)
    cv.a[top] = C('#b07a6a')
    cv.a[shift(top, 0, 1) & m] = C('#6a5058')
    # mesas on the right, warm and sunny
    for (cx, base, tiers, sd) in [(250, 136, [(20, 12), (14, 8)], 121), (330, 134, [(18, 14), (12, 8)], 122),
                                  (405, 138, [(22, 10), (14, 8)], 123)]:
        draw_mesa(cv, cx, base, tiers, sd, band_h=4, haze=HAZE_A, haze_t=0.2, talus=7)
    cv.gradient(150, 132, w - 150, 20, ['sand', 'clay'], soft=0.6)
    # the rest of the hive (smaller crystal spires) peeking behind the close-up
    bl = []
    for (xb, yb, r0, yt, tag, sd) in [(300, 140, 10, 40, 3, 1), (318, 140, 8, 60, 2, 2), (345, 140, 12, 30, 1, 3),
                                      (372, 140, 9, 52, 0, 4), (390, 140, 7, 70, 3, 5)]:
        n = 14
        for k in range(n):
            t = k / (n - 1)
            r = r0 * (1 - t) ** 0.85 + 0.7
            bl.append((xb + 3 * t * t + math.sin(t * 9 + sd), yb + (yt - yb) * t, r, (yb - yt) / n * 1.3,
                       4 * (1 - t), tag))
    bl.append((345, 150, 60, 22, 0, 0))
    m2, NX, NY, NZ, TAG = blob_union(H, w, bl)
    lam = lambert(NX, NY, NZ)
    rgb, _ = crystalize(m2, lam, TAG, NX, (5, 6), 125, bias=-0.05)
    rgb = np.clip(rgb.astype(np.float32) * 0.8 + np.array(col(HAZE_A), np.float32) * 0.2, 0, 255).astype(np.uint8)
    cv.a[m2] = rgb[m2]
    return to_layer(cv)


def rock_facets(m, cell, seed, ramp, light_x=-1.0, grad=None):
    """Chunky faceted rock: Voronoi planes with per-plane tone, bevelled edges (lit top/left, dark
    bottom/right). grad=(y0, y1, k): darken planes toward y1."""
    h, w = m.shape
    R = rng(seed)
    cy, cx = cell
    seeds = []
    for gy in range(-cy, h + cy, cy):
        for gx in range(-cx, w + cx, cx):
            seeds.append((gx + R.uniform(-cx * 0.4, cx * 0.4), gy + R.uniform(-cy * 0.4, cy * 0.4)))
    S = np.array(seeds, np.float32)
    ys, xs = np.nonzero(m)
    cid_flat = np.zeros(len(ys), np.int64)
    for k0 in range(0, len(ys), 4096):
        sl = slice(k0, k0 + 4096)
        d = ((xs[sl, None] - S[None, :, 0]) * 1.0) ** 2 + ((ys[sl, None] - S[None, :, 1]) * 0.8) ** 2
        cid_flat[sl] = np.argmin(d, axis=1)
    cid = np.full((h, w), -1, np.int64)
    cid[ys, xs] = cid_flat
    n = len(seeds)
    tone = np.array([hash01(seed, i) for i in range(n)], np.float32)
    t = 0.25 + tone[cid_flat] * 0.55
    if grad is not None:
        y0, y1, k = grad
        t = t - np.clip((ys - y0) / max(1, y1 - y0), 0, 1) * k
    ra = ramp_arr(ramp)
    lvl = np.clip((t * len(ramp)).astype(int), 0, len(ramp) - 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[ys, xs] = ra[lvl]
    L = np.zeros((h, w), np.int64)
    L[ys, xs] = lvl
    # bevels: where the cell changes below/right -> dark crack; above/left -> light lip
    dn = np.zeros((h, w), bool)
    rt = np.zeros((h, w), bool)
    dn[:-1] = (cid[:-1] != cid[1:])
    rt[:, :-1] = (cid[:, :-1] != cid[:, 1:])
    crack = (dn | rt) & m
    up = np.zeros((h, w), bool)
    up[1:] = (cid[1:] != cid[:-1])
    lip = up & m & ~crack
    rgb[crack] = ra[0]
    rgb[lip] = ra[np.minimum(L[lip] + 2, len(ramp) - 1)]
    return rgb


def rock_columns(m, seed, ramp, grad=None):
    """Bleak columnar cliff: vertical rock columns of varied width, broken by horizontal joints;
    lit lip on each column's left, dark crack on its right."""
    h, w = m.shape
    R = rng(seed)
    ra = ramp_arr(ramp)
    n = len(ramp)
    rgb = np.zeros((h, w, 3), np.uint8)
    lvl = np.zeros((h, w), np.int64)
    x = 0
    yy = np.arange(h)
    while x < w:
        cw = int(R.integers(5, 12))
        base_t = R.uniform(0.3, 0.75)
        y = int(R.integers(-20, 0))
        while y < h:
            sh_ = int(R.integers(10, 34))
            t = base_t + R.uniform(-0.12, 0.12)
            if grad is not None:
                y0, y1, k = grad
                t -= clamp((y + sh_ / 2 - y0) / max(1, y1 - y0)) * k
            li = int(clamp(t, 0, 0.999) * n)
            a, b = max(0, y), min(h, y + sh_)
            if b > a:
                lvl[a:b, x:x + cw] = li
                lvl[a:b, x] = min(n - 1, li + 2)                  # lit lip (sun from the left)
                if cw > 7:
                    lvl[a:b, x + 1] = min(n - 1, li + 1)
                lvl[a:b, min(w - 1, x + cw - 1)] = 0              # crack
                lvl[a, x:x + cw] = 0                              # joint
                if a + 1 < h:
                    lvl[a + 1, x + 1:x + cw - 1] = min(n - 1, li + 1)
            y += sh_
        x += cw
    # a few chipped notches for irregularity
    for k in range(w // 6):
        cx_, cy_ = int(R.integers(0, w)), int(R.integers(0, h))
        lvl[cy_:cy_ + 2, cx_:cx_ + 3] = 0
    rgb[:] = ra[lvl]
    return rgb


def gate_parts():
    """Static geometry of the gate (world coords): door rect, seams, cracks, steps."""
    R = rng(909)
    cracks = []
    for (x, y, ang, ln) in [(110, 60, 2.6, 22), (112, 120, 3.3, 18), (222, 52, 0.4, 20), (220, 110, -0.2, 24),
                            (150, 26, 1.9, 14), (186, 24, 1.2, 16), (98, 150, 2.9, 14), (232, 146, 0.1, 12)]:
        pts = [(x, y)]
        a = ang
        for k in range(int(ln / 3)):
            a += R.uniform(-0.5, 0.5)
            x += math.cos(a) * 3
            y -= math.sin(a) * 3
            pts.append((x, y))
        cracks.append(pts)
    return cracks


GATE_CRACKS = gate_parts()


def draw_cliff(cv, w):
    """The bleak canyon cliff with the carved Tartarus gate (world coords)."""
    top = [(0, 44), (8, 40), (16, 42), (24, 36), (32, 40), (44, 30), (52, 33), (62, 24), (72, 26), (84, 14),
           (94, 16), (104, 4), (112, -6), (232, -6), (236, 4), (242, 10), (240, 22), (248, 30), (252, 46),
           (258, 52), (256, 66), (262, 80), (266, 98), (264, 112), (270, 128), (274, 146), (278, GFLOOR + 2)]
    pts = [(0, GFLOOR + 2)] + top
    m = poly_mask(H, w, pts)
    yy, xx = np.mgrid[0:H, 0:w]
    rgb = rock_columns(m, 701, ['#221a26', '#2e2432', '#3a2e3c', '#463846', '#544452', '#66545e'],
                       grad=(20, GFLOOR, 0.3))
    cv.a[m] = rgb[m]
    # golden rim light on the sunlit skyline + upper ledges
    sky_edge = m & ~shift(m, 0, 1) & (yy > 0)
    cv.a[sky_edge] = C('#f0a070')
    cv.a[shift(sky_edge, 0, 1) & m] = C('#b0705e')
    cv.a[shift(sky_edge, 0, 2) & m & (bayer(H, w) > 0.5)] = C('#7a5058')
    right_edge = m & ~shift(m, -1, 0) & (yy > 4)
    cv.a[right_edge] = C('#1e1620')
    cv.a[shift(right_edge, -1, 0) & m & (bayer(H, w) > 0.4)] = C('#7a5a5e')     # warm bounce from the hive
    # --- the gate
    gx = GX
    # recess (deep shadow carved into the cliff)
    cv.round_rect(gx - 58, 22, 116, GFLOOR - 22, '#1e1620', r=2)
    # pediment (stepped, with chevrons)
    for k in range(4):
        cv.rect(gx - 50 + k * 8, 12 + k * 3 - 6, 100 - k * 16, 4, '#5a4a54' if k % 2 == 0 else '#4a3c48')
        cv.hline(gx - 50 + k * 8, gx + 49 - k * 8, 12 + k * 3 - 6, '#8a7478')
    cv.rect(gx - 56, 18, 112, 12, '#5a4a54')            # lintel
    cv.hline(gx - 56, gx + 55, 18, '#8a7478')
    cv.hline(gx - 56, gx + 55, 29, '#2c2230')
    for k in range(-5, 6):                               # carved rune notches on the lintel
        x = gx + k * 10
        cv.rect(x - 1, 21, 3, 5, '#3a2e3c')
        cv.px(x, 23, '#6c5a60')
    # pillars
    for side in (-1, 1):
        px0 = gx + side * 46 - 7
        cv.rect(px0, 30, 14, GFLOOR - 30, '#5a4a54')
        cv.rect(px0 - 2, 30, 18, 6, '#6c5a60')           # capital
        cv.hline(px0 - 2, px0 + 15, 30, '#8a7478')
        cv.rect(px0 - 2, GFLOOR - 8, 18, 8, '#6c5a60')   # base
        cv.hline(px0 - 2, px0 + 15, GFLOOR - 8, '#8a7478')
        for k in range(3):                                # fluting
            cv.vline(px0 + 3 + k * 4, 38, GFLOOR - 10, '#4a3c48')
        cv.vline(px0, 36, GFLOOR - 9, '#7a666a')         # sunward edge
        cv.vline(px0 + 13, 36, GFLOOR - 9, '#2c2230')
        for yk in (60, 96, 132):                          # diamond rune insets
            cv.poly([(px0 + 7, yk - 4), (px0 + 11, yk), (px0 + 7, yk + 4), (px0 + 3, yk)], '#3a2e3c')
            cv.px(px0 + 7, yk, '#8a7478')
    # door leaves
    dx0, dx1, dy0 = gx - 37, gx + 37, 38
    cv.rect(dx0, dy0, dx1 - dx0 + 1, GFLOOR - dy0, '#4a3c48')
    for side in (-1, 1):
        lx0 = gx + 2 if side == 1 else dx0 + 1
        lw = 34
        for (py, ph) in [(dy0 + 5, 30), (dy0 + 40, 44), (dy0 + 90, GFLOOR - dy0 - 96)]:
            cv.rect(lx0 + 3, py, lw - 6, ph, '#3e3242')
            cv.hline(lx0 + 3, lx0 + lw - 4, py, '#2c2230')
            cv.vline(lx0 + 3, py, py + ph - 1, '#2c2230')
            cv.hline(lx0 + 4, lx0 + lw - 4, py + ph - 1, '#6c5a60')
            cv.vline(lx0 + lw - 4, py + 1, py + ph - 1, '#6c5a60')
            cv.rect(lx0 + 7, py + 4, lw - 14, ph - 8, '#4a3c48')
        for yk in range(dy0 + 3, GFLOOR - 2, 8):           # stud rows
            cv.px(lx0 + 1, yk, '#7a666a')
            cv.px(lx0 + lw - 2, yk, '#7a666a')
    cv.hline(dx0, dx1, dy0, '#6c5a60')
    # emblem disc (stone ring) straddling the seam
    ex, ey = EMB
    cv.circle(ex, ey, 20, '#2c2230')
    cv.circle(ex, ey, 19, '#6c5a60')
    cv.circle(ex, ey, 17, '#4a3c48')
    cv.circle(ex, ey, 12, '#3a2e3c')
    cv.circle(ex, ey, 19, '#8a7478', fill=False)
    # steps
    for k, (y0, ext) in enumerate([(GFLOOR - 2, 52), (GFLOOR + 2, 60), (GFLOOR + 6, 68)]):
        cv.rect(gx - ext, y0, 2 * ext, 4, '#5a4a54')
        cv.hline(gx - ext, gx + ext - 1, y0, '#7a666a')
        cv.hline(gx - ext, gx + ext - 1, y0 + 3, '#3a2e3c')
    # cracks (dark grooves; the red glow is animated)
    for pts in GATE_CRACKS:
        for (a_, b_) in zip(pts[:-1], pts[1:]):
            cv.line(a_[0], a_[1], b_[0], b_[1], '#1e1620')
    return m


@lru_cache(maxsize=None)
def main_b():
    """Main layer (parallax 1): canyon cliff + gate, transition, the blooming hive close-up, pool."""
    w = WB
    cv = Canvas(w, H, KEY)
    # ground everywhere
    gl = np.array([ground_b(x) for x in range(w)])
    yy = np.arange(H)[:, None]
    gm = yy >= gl[None, :]
    nn = fbm(H, w, 2, 7, 801)
    canyon = np.arange(w)[None, :] < 236 + (nn * 30).astype(int)
    warm = np.where((nn > 0.55)[:, :, None], C('copper'), C('clay'))
    warm = np.where(((nn < 0.35) & (bayer(H, w) > 0.5))[:, :, None], C('sand'), warm)
    cold = np.where((nn > 0.5)[:, :, None], C('#4a3c48'), C('#5a4650'))
    cold = np.where(((nn < 0.3) & (bayer(H, w) > 0.5))[:, :, None], C('#3a2e3c'), cold)
    rgb = np.where(canyon[:, :, None] & np.ones((H, 1, 1), bool), cold, warm)
    cv.a[gm] = rgb[gm]
    edge = gm & ~shift(gm, 0, 1)
    cv.a[edge & ~canyon] = C('#f6c27e')
    cv.a[edge & canyon] = C('#7a666a')
    R = rng(802)
    for k in range(70):                     # pebbles + streaks
        x = R.integers(0, w)
        y = R.integers(150, 180)
        if y > gl[x] + 2:
            cv.hline(x, x + R.integers(1, 5), y, '#2c2230' if x < 240 else 'rust')
    # cliff + gate
    draw_cliff(cv, w)
    # transition: barren rocks + a dead tree the bloom did not reach
    for (x, y, r) in [(250, GFLOOR + 4, 7), (276, 162, 5), (238, 172, 4), (90, 172, 6), (40, 168, 8)]:
        cv.ellipse(x, y, r + 1, r * 0.7 + 1, '#2c2230')
        cv.ellipse(x, y, r, r * 0.7, '#5a4a54')
        cv.ellipse(x - 1, y - 1, r * 0.6, r * 0.4, '#6c5a60')
        cv.px(x - r // 2, y - r // 2, '#b0806c')
    dead_tree(cv, 284, 156, 44, 811, color='#3a2e3c', hi='#6c5a60')
    dead_tree(cv, 24, GFLOOR, 30, 812, color='#2c2230', hi='#5a4a54')
    # --- the blooming hive close-up
    hw_, hh_ = w - HX0, 151
    bl = [
        (440 - HX0, 196, 160, 130, 0, 0),
        (352 - HX0, 168, 54, 80, 2, 3),
        (548 - HX0, 150, 70, 96, 2, 1),
        (470 - HX0, 120, 70, 60, 6, 0),
    ]
    for (xb, yb, r0, yt, bend, tag, sd) in [(404, 96, 24, -30, -6, 2, 1), (500, 70, 22, -40, 8, 3, 2),
                                            (566, 70, 18, -20, 6, 0, 3), (340, 118, 13, 50, -6, 1, 4),
                                            (448, 60, 12, -10, 2, 1, 5)]:
        n = 22
        for k in range(n):
            t = k / (n - 1)
            r = r0 * (1 - t) ** 0.85 + 0.8
            knob = 1 + 0.12 * math.sin(t * 15 + sd * 2.1)
            bl.append((xb - HX0 + bend * t * t, yb + (yt - yb) * t, r * knob, (yb - yt) / n * 1.3,
                       8 + 4 * (1 - t), tag))
    m, NX, NY, NZ, TAG = blob_union(hh_, hw_, bl)
    tex = fbm(hh_, hw_, 4, 6, 820) - 0.5
    lam = lambert(NX + tex * 0.4, NY, NZ)
    rgb, L = crystalize(m, lam, TAG, NX, (11, 13), 821, glow=(470 - HX0, 100, 70, 0.12), bias=-0.2,
                        ao=(70, 80, 0.3, 150, 140, 0.15))
    sub = cv.a[0:hh_, HX0:HX0 + hw_]
    sub[m] = rgb[m]
    hc = Canvas(arr=sub)        # view into the main canvas, local coords
    # vine arches (lantern-lit entrances)
    arches = [(420 - HX0, 150, 15, 50), (520 - HX0, 150, 13, 44)]
    for (ax, ay, ahw, ahh) in arches:
        outer = arch_mask(hh_, hw_, ax, ay, ahw + 4, ahh + 4)
        inner = arch_mask(hh_, hw_, ax, ay, ahw, ahh)
        deep = arch_mask(hh_, hw_, ax, ay, ahw - 5, ahh - 9)
        sub[outer] = C('rose')
        sub[outer & ~shift(outer, 1, 0)] = C('cotton')
        sub[outer & ~shift(outer, -1, 0)] = C('raspberry')
        sub[outer & ~shift(outer, 0, 1)] = C('crystal_pk')
        sub[inner] = C('#7a3a2a')
        sub[inner & shift(inner, 0, 3)] = C('amber')
        sub[deep] = C('gold')
        sub[arch_mask(hh_, hw_, ax, ay, ahw - 8, ahh - 15)] = C('butter')
        sub[inner & ~shift(inner, 0, 1)] = C('#5a2a26')
        # a little crystal door frame inside + a round pane above it
        hc.rect(ax - 4, ay - 18, 9, 18, 'copper')
        hc.rect(ax - 3, ay - 17, 7, 17, 'amber')
        hc.vline(ax, ay - 17, ay - 1, 'copper')
        hc.circle(ax, ay - 28, 3, 'cream')
        hc.circle(ax, ay - 28, 3, 'copper', fill=False)
        # vines over the arch
        vine(hc, (ax - ahw - 10, ay - ahh - 2), (ax + ahw + 10, ay - ahh - 4), 9, 830 + ax, step=2)
        vine(hc, (ax - ahw - 6, ay - ahh + 6), (ax - ahw - 2, ay - 4), 2, 840 + ax, step=2)
        vine(hc, (ax + ahw + 6, ay - ahh + 4), (ax + ahw + 3, ay - 6), 2, 850 + ax, step=2)
    # round windows
    for (wx, wy, r) in [(470 - HX0, 92, 6), (376 - HX0, 112, 4), (556 - HX0, 92, 5), (448 - HX0, 54, 4),
                        (500 - HX0, 40, 3), (406 - HX0, 70, 3)]:
        hc.circle(wx, wy, r + 1, 'raspberry')
        hc.circle(wx, wy, r, 'gold')
        hc.circle(wx - 1, wy - 1, r - 2, 'butter')
        hc.hline(wx - r, wx + r, wy, 'copper')
        hc.vline(wx, wy - r, wy + r, 'copper')
    # draped vines higher up
    for (p0, p1, sg, sd) in [((300 - HX0, 120), (360 - HX0, 96), 8, 860), ((372 - HX0, 82), (430 - HX0, 66), 10, 861),
                             ((452 - HX0, 70), (520 - HX0, 64), 12, 862), ((530 - HX0, 70), (578 - HX0, 52), 8, 863),
                             ((404 - HX0, 30), (448 - HX0, 26), 7, 864)]:
        vine(hc, p0, p1, sg, sd, step=2)
    # flowers + moss along the foot
    for k in range(60):
        x = HX0 + 14 + int(hash01(870, k) * (w - HX0 - 16))
        flower_tuft(cv, x, 150 + int(hash01(871, k) * 3), 872 + k, scale=1 + (k % 4 == 0))
    # pool in front (static base colour; reflection is drawn per frame)
    pool = np.zeros((H, w), bool)
    pyy, pxx = np.mgrid[0:H, 0:w]
    pool |= (((pxx - 450) / 120.0) ** 2 + ((pyy - 165) / 9.0) ** 2) <= 1.0
    cv.a[pool] = C('teal')
    rim_ = shift(pool, 0, 1) & ~pool | shift(pool, 0, -1) & ~pool
    cv.a[rim_ & gm] = C('#c8865a')
    for k in range(16):
        flower_tuft(cv, 340 + k * 14, 175 + (k % 3), 880 + k)
    # crystal shards growing from the ground near the hive
    for (x, y, h_, tg) in [(312, 150, 12, 1), (318, 151, 7, 3), (586 - 20, 151, 10, 2), (300, 152, 6, 0)]:
        cv.poly([(x - 3, y), (x, y - h_), (x + 3, y)], CRYSTAL[tg][2])
        cv.poly([(x, y - h_), (x + 3, y), (x + 1, y)], CRYSTAL[tg][1])
        cv.vline(x - 1, y - h_ + 3, y - 1, CRYSTAL[tg][4])
    lanterns = []
    for (p0, p1, sd) in [((392 - HX0, 104), (452 - HX0, 106)), ((492 - HX0, 102), (548 - HX0, 104))] and [
            ((392 - HX0, 104), (452 - HX0, 106), 1), ((492 - HX0, 102), (548 - HX0, 104), 2)]:
        pts = vine(hc, p0, p1, 6, 890 + sd, leaf='green', stem='#5a3a2a', flowers=False, step=99)
        for k in range(1, 5):
            lx, ly = pts[k * len(pts) // 5]
            hc.vline(lx, ly + 1, ly + 2, '#5a3a2a')
            lc = ('hot_pink', 'gold', 'turquoise', 'lavender')[(k + sd) % 4]
            hc.rect(lx - 1, ly + 3, 3, 4, lc)
            hc.px(lx, ly + 4, 'cream')
            hc.hline(lx - 1, lx + 1, ly + 3, '#5a3a2a')
            lanterns.append((lx + HX0, ly + 5, lc))
    glints = []
    ys, xs = np.nonzero((L >= 3) & m)
    Rg = rng(899)
    for k in Rg.choice(len(ys), 16, replace=False):
        glints.append((int(xs[k]) + HX0, int(ys[k])))
    return to_layer(cv), pool, lanterns, glints


@lru_cache(maxsize=None)
def fg_b():
    w = W + int(PAN_B * P_FGB) + 8
    cv = Canvas(w, H, KEY)
    # left (canyon side): dark boulders and a dead thorny branch
    for pts in ([(0, H), (0, 150), (10, 144), (22, 150), (30, 162), (40, 172), (44, H)],
                [(250, H), (262, 166), (276, 160), (290, 168), (300, H)]):
        m = poly_mask(H, w, pts)
        cv.a[m] = C('#2a2030')
        top = m & ~shift(m, 0, 1)
        cv.a[top] = C('#5a4650')
    dead_tree(cv, 14, 150, 34, 931, color='#1e1620', hi='#3a2e3c')
    # right (hive side): crystal shards + flowers + a vine
    base = 400
    for (x, y, h_, tg) in [(base + 6, H, 26, 3), (base + 18, H, 14, 1), (base + 150, H, 20, 2),
                           (base + 162, H, 32, 0)]:
        cv.poly([(x - 6, y), (x - 1, y - h_), (x + 6, y)], CRYSTAL[tg][2])
        cv.poly([(x - 1, y - h_), (x + 6, y), (x + 2, y)], CRYSTAL[tg][1])
        cv.line(x - 4, y - 2, x - 1, y - h_ + 2, CRYSTAL[tg][4])
        cv.px(x - 1, y - h_, 'white')
    for k in range(22):
        flower_tuft(cv, base - 10 + k * 9, H - 1 - (k % 3), 940 + k, 2)
    return to_layer(cv)


def render_b(f):
    cam = cam_b(f)
    sh = 0
    if T_TART <= f < T_TART + 10:
        sh = [2, -2, 1, -1, 1, -1, 1, 0, 1, 0][f - T_TART]
    c = Canvas(W, H)
    sk = sky_b()
    sx = int(round(cam * P_SKYB))
    c.a[:] = sk.a[:, sx:sx + W]
    draw_clouds(c, f, 'bloom', [(60, 14, 50, 21, 0.1), (190, 30, 40, 22, 0.08), (300, 8, 60, 23, 0.06)],
                ox=-int(round(cam * 0.12)))
    comp(c, far_b(), -int(round(cam * P_FARB)), sh)
    comp(c, mid_b(), -int(round(cam * P_MIDB)), sh)
    ml, pool, lanterns, glints = main_b()
    mx = -int(round(cam))
    comp(c, ml, mx, sh)
    # ---- pool reflection (screen space)
    X0, X1 = max(0, mx), min(W, mx + WB)
    pm = pool[:, X0 - mx:X1 - mx]
    if pm.any():
        ys, xs = np.nonzero(pm)
        wl = 156 + sh
        src_y = np.clip(2 * wl - ys - 2, 0, H - 1)
        rip = (np.sin((ys * 1.7 + f * 0.35)) * 1.2).astype(int)
        src_x = np.clip(xs + X0 + rip, 0, W - 1)
        refl = c.a[src_y, src_x].astype(np.float32)
        refl = refl * 0.55 + np.array(col('teal'), np.float32) * 0.45
        c.a[ys + sh if sh >= 0 else ys, xs + X0] = np.clip(refl, 0, 255).astype(np.uint8)
        for k in range(6):
            ph = (f + k * 7) % 28
            if ph < 14:
                gx = mx + 340 + int(hash01(k, 3, (f + k * 7) // 28) * 220)
                c.hline(gx, gx + 2 + k % 3, 160 + (k * 3) % 10 + sh, 'ice')
    # ---- hive life: lanterns, glints, petals
    for i, (lx, ly, lc) in enumerate(lanterns):
        k = 0.6 + 0.4 * math.sin(f / 5.0 + i * 1.3)
        fx.glow(c, lx + mx, ly + sh, 6, lc, 0.25 * k, rings=3)
    for i, (gx, gy) in enumerate(glints):
        fx.sparkle(c, gx + mx, gy + sh, f, 'white', period=26, phase=hash01(9, i), size=2)
    for i in range(16):
        sp = 0.3 + hash01(31, i) * 0.4
        px_ = int((hash01(32, i) * (W + 40) - f * sp) % (W + 40)) - 20 + mx + PAN_B
        py_ = int((hash01(33, i) * 150 + f * 0.3 + math.sin(f / 10.0 + i) * 3) % 160) + 10
        if 0 <= px_ < W:
            c.px(px_, py_, ('bubblegum', 'cotton', 'lilac', 'cream')[i % 4])
    # ---- the gate: red seams, pulsing cracks, rune-ring seal, flare at TARTARUS
    gx = GX + mx
    flare = (1 - seg(f, T_TART, T_TART + 30)) ** 2 if f >= T_TART else 0.0
    pre = seg(f, 236, T_TART) * 0.4
    pulse = 0.5 + 0.5 * math.sin(f / 7.0)
    heat = clamp(0.35 + 0.3 * pulse + pre + flare)
    hot = mix('lava_dk', 'lava', heat) if heat < 0.7 else mix('lava', 'lava_hi', (heat - 0.7) / 0.3)
    core = 'butter' if flare > 0.3 else hot
    if gx - 60 < W and gx + 60 > 0:
        fx.glow(c, gx, GFLOOR + sh, 50 + int(40 * flare), 'red', 0.12 + 0.12 * pulse + 0.4 * flare, rings=5)
        c.vline(gx, 38 + sh, GFLOOR - 3 + sh, hot)
        c.vline(gx, 48 + sh, GFLOOR - 12 + sh, core)
        c.vline(gx - 37, 38 + sh, GFLOOR - 3 + sh, mix(hot, 'lava_dk', 0.4))
        c.vline(gx + 37, 38 + sh, GFLOOR - 3 + sh, mix(hot, 'lava_dk', 0.4))
        c.hline(gx - 37, gx + 37, 37 + sh, mix(hot, 'lava_dk', 0.3))
        if flare > 0:
            c.vline(gx - 1, 40 + sh, GFLOOR - 3 + sh, mix('lava', 'lava_dk', 1 - flare))
            c.vline(gx + 1, 40 + sh, GFLOOR - 3 + sh, mix('lava', 'lava_dk', 1 - flare))
            fx.light_rect(c, gx - 70, GFLOOR - 4 + sh, 140, 16, 'lava', 0.35 * flare, fade='down')
        for j, pts in enumerate(GATE_CRACKS):
            ck = clamp(0.25 + 0.3 * math.sin(f / 9.0 + j * 1.1) + pre + flare)
            cc = mix('lava_dk', 'red', ck / 0.7) if ck < 0.7 else mix('red', 'lava', (ck - 0.7) / 0.3)
            if flare > 0.7:
                cc = 'lava_hi'
            if flare > 0.2:
                mid_ = pts[len(pts) // 2]
                fx.glow(c, mid_[0] + mx, mid_[1] + sh, 8, 'red', 0.3 * flare, rings=3)
            for (a_, b_) in zip(pts[:-1], pts[1:]):
                c.line(a_[0] + mx, a_[1] + sh, b_[0] + mx, b_[1] + sh, cc)
        # emblem: glowing seal with a slowly turning rune ring
        ex, ey = EMB[0] + mx, EMB[1] + sh
        fx.glow(c, ex, ey, 26 + int(10 * flare), 'magic', 0.22 + 0.1 * pulse + 0.3 * flare, rings=4)
        for i in range(12):
            a_ = i / 12 * 6.283 + f * 0.012
            rx_, ry_ = ex + math.cos(a_) * 15, ey + math.sin(a_) * 15
            rc = 'lilac' if (i + f // 6) % 4 else 'white'
            c.px(rx_, ry_, rc)
            if i % 3 == 0:
                c.px(rx_ + math.cos(a_) * 1.5, ry_ + math.sin(a_) * 1.5, 'lavender')
        c.circle(ex, ey, 10, 'lavender', fill=False)
        # lock sigil: 4-point star with a keyhole
        for (dx, dy) in [(0, -8), (0, 8), (-8, 0), (8, 0)]:
            c.line(ex, ey, ex + dx, ey + dy, 'lilac')
        c.circle(ex, ey, 3, 'white')
        c.px(ex, ey - 1, 'magic')
        c.vline(ex, ey, ey + 2, 'magic')
        # embers rising from the seam
        for i in range(10):
            per = 40 + int(hash01(51, i) * 30)
            t = ((f + int(hash01(52, i) * per)) % per) / per
            exx = gx + int((hash01(53, i) - 0.5) * 70) + int(math.sin(f / 8.0 + i) * 2)
            eyy = GFLOOR - 4 - int(t * 70)
            c.px(exx, eyy + sh, 'lava_hi' if t < 0.4 else ('lava' if t < 0.7 else 'lava_dk'))
        # dust + pebbles falling from the lintel at the sting
        if f >= T_TART:
            t = f - T_TART
            # shockwave of the seal
            if t < 14:
                rr = 20 + t * 4
                c.circle(ex, ey, rr, 'lilac' if t < 7 else 'lavender', fill=False)
                if t < 8:
                    c.circle(ex, ey, rr - 2, 'magic', fill=False)
            # dust curtains pouring from the lintel joints
            for j, sx_ in enumerate((-44, -14, 18, 40)):
                tt = t - j * 2
                if 0 <= tt < 30:
                    front = 30 + int(tt * 5.5)
                    dens = 0.55 * (1 - tt / 30)
                    c.dither(gx + sx_, 30 + sh, 2, min(front, GFLOOR) - 30, '#c8a890', dens)
                    c.dither(gx + sx_ + 2, 30 + sh, 1, min(front, GFLOOR) - 34, '#8a7478', dens)
            for i in range(44):
                d = int(hash01(61, i) * 12)
                tt = t - d
                if tt < 0:
                    continue
                x0 = gx + int((hash01(62, i) - 0.5) * 120)
                y0 = 18 + int(hash01(63, i) * 16)
                y = y0 + int(0.5 * 0.14 * tt * tt + tt * 0.5)
                if y < GFLOOR - 1:
                    big = i % 6 == 0
                    if big:
                        c.rect(x0, y + sh, 2, 2, '#6c5a60')
                        c.px(x0, y + sh, '#b0806c')
                    else:
                        c.px(x0, y + sh, '#c8a890')
                        if tt > 4:
                            c.px(x0, y - 1 + sh, '#8a7478')
            # dust cloud billowing at the foot
            for i in range(14):
                k = seg(f, T_TART + 4, T_TART + 34)
                if 0 < k < 1:
                    xx = gx + int((hash01(71, i) - 0.5) * 120 * (0.5 + k))
                    yy = GFLOOR - 2 - int(hash01(72, i) * 10 * k)
                    c.dither(xx - 3, yy - 2 + sh, 7, 4, '#8a7478', 0.5 * (1 - k))
    # canyon dust motes in the red light
    fx.dust_motes(c, f, (mx + 60, 110, 220, 50), n=14, color='#b0806c', seed=7)
    # the explorer's balloon: the camera follows it west into the canyon
    bx = int(round(58 + seg(f, 216, 287) * 18))
    by = int(round(54 - seg(f, 216, 287) * 6))
    draw_balloon(c, bx, by, f, 's', flame=(f // 24) % 3 == 0)
    comp(c, fg_b(), -int(round(cam * P_FGB)), sh)
    callout(c, gx, 20 + sh, 'TARTARUS', f, 244, 279, side='down', length=4)
    return c


# ================================================================== main
def render(f):
    if f < 72:
        return travel_frame(f, ID)
    if f < 216:
        c = render_a(f)
    else:
        c = render_b(f)
    if f < 80:
        iris(c, 1 - seg(f, 72, 79), 204, 103)
    return c


SFX = [
    (76, 'wind', 0.45),        # bleak crater wind as the iris opens
    (92, 'lava', 0.35),        # slime pools blorping
    (112, 'rumble', 0.5),      # the heart swells, tremor
    (126, 'lava', 0.3),
    (148, 'shimmer', 0.6),     # just after the TRANSFORM hit: the colour wave
    (158, 'crystal', 0.5),     # chitin turning to crystal
    (176, 'sparkle', 0.4),
    (198, 'crystal', 0.35),
    (220, 'sparkle', 0.3),     # lanterns + facet glints in the close-up
    (226, 'whoosh', 0.45),     # pan west to the canyon
    (238, 'wind', 0.5),        # bleak canyon gust
    (254, 'crack', 0.4),       # seams flare after the tartarus sting, dust falls
    (262, 'rumble', 0.35),
    (270, 'wind', 0.35),
]


# ---- director pass: smooth shot changes (engine/shotfx.py) instead of hard cuts / white flashes
from engine.shotfx import smooth_cuts  # noqa: E402

render = smooth_cuts(render, ID, [(216, 'dissolve', {})])

# ---- director pass: 2-bar map flight; this module keeps working in scene frames (engine/regionfx.py)
from engine.regionfx import long_flight  # noqa: E402

render, SFX = long_flight(render, ID, SFX)
