"""Smooth shot changes inside a scene (owned by the director). Pure, pixel-art friendly.

Use in a scene's render(f) instead of a hard `if f < CUT: a(f) else: b(f)` switch:

    from engine.shotfx import shot_change
    c = shot_change(f, 216, _shot_a, _shot_b, 'dissolve')

The transition is centred on the cut frame `at` (the bar line, where the music crashes) and lasts
`dur` frames; inside the window BOTH shot functions are rendered at f and blended, so both keep
animating. Outside it, only a(f) (before) or b(f) (after) is called.

Styles (default durations in frames):
  'dissolve' 14  ordered-dither crossfade: a new angle on the same place
  'glide'    16  dissolve while both shots drift sideways (dir=-1: content moves left, the camera
                 travels right; dir=+1 the other way): neighbouring spots along a street/coast
  'cloud'    24  a bank of puffy clouds sweeps across and reveals the next place (hour= tints them
                 for the time of day; dir=+1 moves left->right): a different place in the region
  'zoom'     14  the old shot zooms toward (cx, cy) while a gold-rimmed iris opens on the new shot
                 from that point: going inside (factory, caldera, hall, alley)
"""
import math
from functools import lru_cache

import numpy as np

from .core import W, H, Canvas, col, mix, BAYER4, hash01, smooth, ease_in_out

DUR = {'dissolve': 14, 'glide': 16, 'cloud': 24, 'zoom': 14}

_YY, _XX = np.mgrid[0:H, 0:W]
_BAYER = np.tile(BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W]


# ------------------------------------------------------------------ primitives (canvas, canvas, t)
def dissolve(a, b, t):
    """Ordered-dither crossfade: t=0 -> a, t=1 -> b."""
    out = a.copy()
    m = _BAYER < t
    out.a[m] = b.a[m]
    return out


def _shift(arr, dx):
    """Shift an (H, W, 3) array horizontally by dx; returns (shifted, valid-mask)."""
    out = np.zeros_like(arr)
    valid = np.zeros((H, W), bool)
    if dx >= 0:
        out[:, dx:] = arr[:, :W - dx]
        valid[:, dx:] = True
    else:
        out[:, :W + dx] = arr[:, -dx:]
        valid[:, :W + dx] = True
    return out, valid


def glide(a, b, t, dir=-1, dist=28):
    """Both shots drift by `dist` px in direction `dir` while dissolving (eased)."""
    e = ease_in_out(t)
    da = int(round(dir * dist * e))              # a drifts out
    db = int(round(dir * dist * (e - 1)))        # b drifts in from the opposite side
    sa, va = _shift(a.a, da)
    sb, vb = _shift(b.a, db)
    show_b = (_BAYER < smooth(t)) & vb | ~va
    out = Canvas(arr=np.where(show_b[..., None], sb, sa).astype(np.uint8))
    same = (a.a == b.a).all(axis=2)              # overlays drawn identically in both shots (atlas
    out.a[same] = a.a[same]                      # card, static frames) stay put instead of drifting
    return out


# ------------------------------------------------------------------ clouds
def cloud_palette(hour):
    """(shadow, body, highlight) colours for clouds at a clock hour (0..24)."""
    h = hour % 24
    if h < 4.8 or h >= 21.0:                      # night: moonlit indigo
        return col('navy'), col('indigo'), mix('indigo', 'periwinkle', 0.38)
    if h < 6.6:                                   # dawn
        k = (h - 4.8) / 1.8
        return mix('purple', 'mauve', k), mix('rose', 'blush', k), mix('peach', 'cream', k)
    if h < 17.0:                                  # day
        return col('cloud_dk'), col('cloud'), col('white')
    if h < 19.2:                                  # golden hour / sunset
        return mix('mauve', 'rose', 0.5), mix('blush', 'peach', 0.4), col('cream')
    k = (h - 19.2) / 1.8                          # dusk
    return mix('purple', 'navy', k), mix('violet', 'indigo', k), mix('lilac', 'periwinkle', k)


@lru_cache(maxsize=None)
def _bumps(seed, n, spacing, rmin, rmax):
    r = np.random.default_rng(seed)
    pos = np.arange(n) * spacing + r.integers(-spacing // 3, spacing // 3 + 1, n)
    rad = r.integers(rmin, rmax + 1, n)
    return pos, rad


def _scallop(length, seed, spacing=22, rmin=12, rmax=24):
    """Edge profile (len `length`): how far a scalloped cloud edge bulges at each position (>= 0)."""
    n = length // spacing + 3
    pos, rad = _bumps(seed, n, spacing, rmin, rmax)
    u = np.arange(length)[:, None]
    d2 = (u - pos[None, :]) ** 2
    bulge = np.sqrt(np.clip(rad[None, :] ** 2 - d2, 0, None))
    return bulge.max(axis=1)


@lru_cache(maxsize=None)
def _cloud_texture(seed=7):
    """Static tile (2H x 2W) of overlapping cumulus puffs: 0 deep, 1 shadow, 2 body, 3 highlight."""
    r = np.random.default_rng(seed)
    th, tw = H * 2, W * 2
    tex = np.full((th, tw), 2, np.uint8)
    yy, xx = np.mgrid[0:th, 0:tw]
    puffs = [(int(r.integers(0, tw)), int(r.integers(0, th)), int(r.integers(9, 26))) for _ in range(260)]
    puffs.sort(key=lambda p: p[1])                 # back (top) to front (bottom)
    for cx, cy, rad in puffs:
        for ox in (-tw, 0, tw):                    # wrap horizontally so the tile is seamless
            for oy in (-th, 0, th):
                x0, y0 = cx + ox, cy + oy
                if x0 + rad < 0 or x0 - rad >= tw or y0 + rad < 0 or y0 - rad >= th:
                    continue
                ys, ye = max(0, y0 - rad), min(th, y0 + rad + 1)
                xs, xe = max(0, x0 - rad), min(tw, x0 + rad + 1)
                dy = yy[ys:ye, xs:xe] - y0
                dx = xx[ys:ye, xs:xe] - x0
                d2 = dx * dx + dy * dy
                inside = d2 <= rad * rad
                sub = tex[ys:ye, xs:xe]
                v = np.full(sub.shape, 2, np.uint8)
                sx_, sy_ = dx + rad * 0.18, dy + rad * 0.30              # lit disc offset up-left
                lit = sx_ * sx_ + sy_ * sy_ < (rad * 0.98) ** 2
                v[~lit] = 1                                              # crescent shadow low-right
                edge = (sx_ * sx_ + sy_ * sy_ < (rad * 1.08) ** 2) & ~lit
                v[edge & (((yy[ys:ye, xs:xe] + xx[ys:ye, xs:xe]) & 1) == 0)] = 2   # dithered terminator
                hx, hy = dx + rad * 0.35, dy + rad * 0.45
                v[(hx * hx + hy * hy < (rad * 0.52) ** 2) & (dy < rad * 0.05)] = 3   # lit cap
                v[(d2 > (rad - 1.5) ** 2) & (dy > -rad * 0.2)] = 0       # crease under each puff
                sub[inside] = v[inside]
    return tex


def _paint_clouds(out, mask, edge_dist, hour, ox=0, oy=0):
    """Fill `mask` pixels of out with textured cloud; edge_dist (px from the edge) brightens rims."""
    sh, body, hi = cloud_palette(hour)
    deep = tuple(int(v * 0.82) for v in sh)
    tex = _cloud_texture()
    tx = (_XX + ox) % (W * 2)
    ty = (_YY + oy) % (H * 2)
    t = tex[ty, tx]
    pal = np.array([deep, sh, body, hi], np.uint8)
    out.a[mask] = pal[t[mask]]
    out.a[(edge_dist < 2) & mask] = hi
    band = (edge_dist >= 2) & (edge_dist < 5) & mask & (_BAYER < 0.5)
    out.a[band] = body


def cloud(a, b, t, hour=12.0, dir=1, seed=3):
    """A cloud bank sweeps across horizontally: covers a, then reveals b."""
    e = ease_in_out(t)
    band = W + 90                                  # wide enough to fully cover mid-transition
    travel = W + band + 60
    lead = -30 + e * travel                        # leading edge position along the sweep
    trail = lead - band
    u = _XX if dir > 0 else (W - 1 - _XX)          # coordinate along the sweep
    lead_bulge = _scallop(H, seed)[_YY]            # per-row bulge of the leading edge
    trail_bulge = _scallop(H, seed + 11)[_YY]
    lead_x = lead + lead_bulge
    trail_x = trail - trail_bulge
    cover = (u < lead_x) & (u > trail_x)
    passed = u <= trail_x
    out = a.copy()
    out.a[passed] = b.a[passed]
    edge = np.minimum(lead_x - u, u - trail_x)
    off = int(e * 140) * (1 if dir > 0 else -1)
    _paint_clouds(out, cover, edge, hour, ox=-off, oy=0)
    # a few lead puffs ahead of the bank
    for i in range(5):
        py = int(hash01(seed, i, 1) * (H - 30)) + 15
        pr = 6 + int(hash01(seed, i, 2) * 8)
        pu = lead + 12 + hash01(seed, i, 3) * 40
        px = int(pu) if dir > 0 else int(W - 1 - pu)
        d2 = (_XX - px) ** 2 + (_YY - py) ** 2
        puff = d2 < pr * pr
        if puff.any():
            _paint_clouds(out, puff, pr - np.sqrt(d2), hour, ox=7 * i, oy=5 * i)
    return out


# ------------------------------------------------------------------ zoom-iris
def zoom(a, b, t, cx=W // 2, cy=H // 2):
    """a zooms toward (cx, cy) (up to 1.6x, nearest) while an iris opens on b from that point."""
    e = ease_in_out(t)
    s = 1.0 + 0.6 * e
    sx = np.clip(((_XX - cx) / s + cx).astype(np.int32), 0, W - 1)
    sy = np.clip(((_YY - cy) / s + cy).astype(np.int32), 0, H - 1)
    out = Canvas(arr=a.a[sy, sx])
    rmax = math.hypot(max(cx, W - cx), max(cy, H - cy)) + 2
    r = rmax * e
    d2 = (_XX - cx) ** 2 + (_YY - cy) ** 2
    inside = d2 <= r * r
    out.a[inside] = b.a[inside]
    if 0.02 < e < 0.98:
        ring = (d2 > r * r) & (d2 <= (r + 2) ** 2)
        out.a[ring] = col('gold')
        ring2 = (d2 > (r + 2) ** 2) & (d2 <= (r + 3) ** 2)
        out.a[ring2] = col('butter')
    same = (a.a == b.a).all(axis=2)              # overlays identical in both shots (atlas card) stay put
    out.a[same] = a.a[same]
    return out


_STYLES = {'dissolve': dissolve, 'glide': glide, 'cloud': cloud, 'zoom': zoom}


def window(at, style='dissolve', dur=None):
    """(first, end) local frames of a transition centred on `at`."""
    d = dur or DUR[style]
    return at - d // 2, at - d // 2 + d


def shot_change(f, at, a, b, style='dissolve', dur=None, **kw):
    """Render frame f around a shot change at `at` from shot function a to b (see module doc)."""
    w0, w1 = window(at, style, dur)
    if f < w0:
        return a(f)
    if f >= w1:
        return b(f)
    t = (f - w0 + 0.5) / (w1 - w0)
    return _STYLES[style](a(f), b(f), t, **kw)


def smooth_cuts(render_fn, seg_id, cuts):
    """Wrap a scene's render(f) so its hard cuts become transitions (no scene restructuring needed).

    cuts: [(at, style, kwargs), ...]. Inside a transition window the side of the cut that f is on
    stays live and the other side is held on its nearest frame (at-1 or at); the blend hides the hold.
    For 'cloud', hour defaults to the HUD clock of the segment at the cut.
    """
    from . import timeline as TL
    table = []
    for at, style, kw in cuts:
        kw = dict(kw)
        dur = kw.pop('dur', None)
        if style == 'cloud' and 'hour' not in kw:
            kw['hour'] = TL.hour_at(seg_id, at + TL.BY_ID[seg_id].get('scene_offset', 0))
        table.append((at, style, window(at, style, dur), kw))

    def wrapped(f):
        for at, style, (w0, w1), kw in table:
            if w0 <= f < w1:
                a = render_fn(f) if f < at else render_fn(at - 1)
                b = render_fn(f) if f >= at else render_fn(at)
                t = (f - w0 + 0.5) / (w1 - w0)
                return _STYLES[style](a, b, t, **kw)
        return render_fn(f)

    wrapped.__wrapped__ = render_fn
    wrapped.cuts = table
    return wrapped


def cruise(t, r=0.2):
    """Camera ease with a constant-speed middle and short acceleration ramps (fraction r at each
    end): peak speed = 1/(1-r) x average (1.25x for r=0.2), versus 3x for core.ease_in_out."""
    t = min(1.0, max(0.0, t))
    v = 1.0 / (1.0 - r)
    if t < r:
        return v * t * t / (2 * r)
    if t > 1 - r:
        u = 1 - t
        return 1 - v * u * u / (2 * r)
    return v * (r / 2 + (t - r))
