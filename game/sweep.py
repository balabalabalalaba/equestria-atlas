"""The cloud sweep between areas of a place: engine.shotfx.cloud, pixel for pixel, made cheap enough for a
60 fps frame (the film builds it once per frame offline; the game needs it in ~1 ms).

Same look: the cloud texture is pre-coloured once per palette, every per-row profile is cached, only
the columns the bank covers are textured, and the lead puffs are painted in their own small boxes.
`needs_b(t, dir)` tells whether any of the new picture shows yet, so the caller can skip rendering it
while the bank still hides it (the first half of the sweep).
"""
from functools import lru_cache

import numpy as np

from engine import W, H, ease_in_out, hash01
from engine import shotfx as S

BAND = W + 90
TRAVEL = W + BAND + 60


@lru_cache(maxsize=8)
def _bulges(seed):
    return S._scallop(H, seed).astype(np.float64), S._scallop(H, seed + 11).astype(np.float64)


@lru_cache(maxsize=64)
def _coloured(pal_key):
    """The cloud texture (2H x 2W) in RGB for one palette (deep, shadow, body, highlight)."""
    pal = np.array(pal_key, np.uint8)
    return pal[S._cloud_texture()]


def _pal(hour):
    sh, body, hi = S.cloud_palette(hour)
    deep = tuple(int(v * 0.82) for v in sh)
    return (tuple(deep), tuple(sh), tuple(body), tuple(hi))


def warm(hour=12.0):
    """Build the cached texture (~90 ms the first time) at load time, not mid-sweep."""
    _coloured(_pal(hour))
    _bulges(3)


def _edges(t, seed):
    e = ease_in_out(t)
    lead = -30 + e * TRAVEL
    lb, tb = _bulges(seed)
    return e, lead, lead + lb, lead - BAND - tb


def needs_b(t, dir=1, seed=3):
    """True once some of the new picture is uncovered (the trailing edge has entered the screen)."""
    _e, _lead, _lx, trail_x = _edges(t, seed)
    return trail_x.max() >= 0


def _wrap_slice(a, start, n, axis):
    """a[start:start+n] along axis, wrapping around (n <= the axis length): one or two slices."""
    size = a.shape[axis]
    s = start % size
    if s + n <= size:
        return a[s:s + n] if axis == 0 else a[:, s:s + n]
    k = size - s
    parts = (a[s:], a[:n - k]) if axis == 0 else (a[:, s:], a[:, :n - k])
    return np.concatenate(parts, axis=axis)


def _paint(out, mask, edge, pal, ctex, ox, oy, x0, y0):
    """shotfx._paint_clouds on the box out[y0:y0+h, x0:x0+w] (mask / edge are box-sized)."""
    h, w = mask.shape
    tex = _wrap_slice(_wrap_slice(ctex, y0 + oy, h, 0), x0 + ox, w, 1)
    box = out[y0:y0 + h, x0:x0 + w]
    box[mask] = tex[mask]
    box[(edge < 2) & mask] = pal[3]
    band = (edge >= 2) & (edge < 5) & mask & (S._BAYER[y0:y0 + h, x0:x0 + w] < 0.5)
    box[band] = pal[2]


def cloud(a, b, t, hour=12.0, dir=1, seed=3):
    """Like engine.shotfx.cloud(a, b, t, hour, dir, seed) -> new Canvas. b may be None while
    needs_b() is False."""
    e, lead, lead_x, trail_x = _edges(t, seed)
    pal = _pal(hour)
    ctex = _coloured(pal)
    out = a.copy()
    o = out.a
    # columns the bank can touch (in sweep coordinate u), then mapped to screen x
    u0 = max(0, int(np.floor(trail_x.min())))
    u1 = min(W, int(np.ceil(lead_x.max())) + 1)
    if dir > 0:
        xs0, xs1 = u0, u1
    else:
        xs0, xs1 = W - u1, W - u0
    if b is not None and trail_x.max() >= 0:
        # passed: u <= trail_x  (per row)
        u = np.arange(W)[None, :] if dir > 0 else (W - 1 - np.arange(W))[None, :]
        passed = u <= trail_x[:, None]
        o[passed] = b.a[passed]
    if xs1 > xs0:
        xx = np.arange(xs0, xs1)
        u = (xx if dir > 0 else W - 1 - xx)[None, :]
        lx, tx = lead_x[:, None], trail_x[:, None]
        cover = (u < lx) & (u > tx)
        edge = np.minimum(lx - u, u - tx)
        off = int(e * 140) * (1 if dir > 0 else -1)
        _paint(o, cover, edge, pal, ctex, -off, 0, xs0, 0)
    for i in range(5):                                  # a few lead puffs ahead of the bank
        py = int(hash01(seed, i, 1) * (H - 30)) + 15
        pr = 6 + int(hash01(seed, i, 2) * 8)
        pu = lead + 12 + hash01(seed, i, 3) * 40
        px = int(pu) if dir > 0 else int(W - 1 - pu)
        x0, x1 = max(0, px - pr), min(W, px + pr + 1)
        y0, y1 = max(0, py - pr), min(H, py + pr + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        puff = d2 < pr * pr
        if puff.any():
            _paint(o, puff, pr - np.sqrt(d2), pal, ctex, 7 * i, 5 * i, x0, y0)
    return out
