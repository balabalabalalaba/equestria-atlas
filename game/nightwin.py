"""Lit windows at night for places the film painted by day.

game/daylight.py darkens a day-painted place for the clock's hour and keeps only lights that outshine
their surroundings, so a window painted as daytime glass stays dark at night, and a gold pane painted on
a pale wall would only dim to khaki. Here the window panes of a scene layer (an RGBA array) are found
once (connected bits of blue glass, from a 2 x 2 pane to a tall 6 x 9 window; `pale_only`: just those
on a pale wall, the mean of the non-glass pixels within 3 px at least PALE); at night a share of them is painted warm where the
layer is still what is seen (nothing drawn in front), more as the night deepens (from night() 0.3 to
0.7), and now and then one more goes on. The region keeps them shining after its relight with
GlowKeep.keep_glow (game/regions/everfree.py), given the mask `paint` returns.

    mask = nightwin.paint(c, hour, t, [('key', get_rgba, x, y), ...])
    self.keep_glow(c, mask, 0.95)

`get_rgba()` returns the layer (an RGBA array, or a tuple whose first item is one); the panes are
cached by `key`. A placement may add `share` (lit share at full night) and `boost` ((x0, x1, share): a
column range of the layer with its own share, e.g. a landmark tower), and end with a dict of options:
`exclude` (boxes (x0, y0, x1, y1) in layer px: a "pane" touching one is not a window, e.g. the lavender
pickets of a fence or a blue-grey pillar), `only` (boxes: only panes inside one are windows).
"""
import numpy as np

from engine import W, H, col, clamp, hash01

from . import daylight

PALE = 150                      # a wall this bright (its non-glass pixels near a pane) counts as pale
SHARE = 0.65                    # share of the panes lit in full night
NIGHT_ON = 0.3                  # daylight.night() at which the first windows go on ...
NIGHT_FULL = 0.7                # ... and by which all of the share is on
FLIP = 0.06                     # share of the windows that go on now and then
FLIP_T = 420                    # steps between those changes
WARM = np.array([col('#ffd466'), col('#ffc455'), col('#ffe07a')], np.uint8)

_CACHE = {}


def _glassy(rgb):
    a = rgb.astype(np.int16)                    # (uint8 sums would wrap round: white would pass as glass)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b > r + 25) & (b > g + 5)           # blue glass (pale lavender stone and sills are not)


def pale_wall(lum, wall, s_):
    """Is the wall round the window at slice s_ pale? (the mean brightness of the non-glass pixels within
    3 px)"""
    y0, y1 = max(0, s_[0].start - 3), s_[0].stop + 3
    x0, x1 = max(0, s_[1].start - 3), s_[1].stop + 3
    m = wall[y0:y1, x0:x1]
    return bool(m.any()) and lum[y0:y1, x0:x1][m].mean() >= PALE


def lum_of(rgb):
    a = rgb.astype(np.int32)
    return (a[..., 0] * 77 + a[..., 1] * 150 + a[..., 2] * 29) >> 8


def _hits(s_, boxes):
    """Does the slice pair s_ (a pane's bounding box) touch any of the boxes (x0, y0, x1, y1)?"""
    y0, y1, x0, x1 = s_[0].start, s_[0].stop, s_[1].start, s_[1].stop
    return any(x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1 for bx0, by0, bx1, by1 in boxes)


def panes(key, get_rgba, share=SHARE, boost=None, extra=None, pale_only=False, exclude=(), only=()):
    """(ys, xs, v, kind, pane, src rgb) of the windows on pale walls of a layer: each window's pixels with
    its threshold v (lit once the night's ramp passes |v|; negative: a window that goes on now and then),
    its warm shade (0..2), its number and the painted colour. `extra`: a bool mask of more pixels that
    count as glass (e.g. drawn blinds)."""
    exclude, only = tuple(map(tuple, exclude)), tuple(map(tuple, only))
    ck = (key, share, boost, pale_only, exclude, only)
    if ck in _CACHE:
        return _CACHE[ck]
    from scipy import ndimage
    rgba = get_rgba()
    rgba = rgba[0] if isinstance(rgba, tuple) else rgba
    rgb = rgba[..., :3]
    op = rgba[..., 3] >= 128
    lum = lum_of(rgb)
    glass = op & _glassy(rgb)
    wall = op & ~glass
    if extra is not None:
        glass |= op & extra
    lab, n = ndimage.label(glass)
    ys, xs, vs, kind, pane = [], [], [], [], []
    for i, s_ in enumerate(ndimage.find_objects(lab) if n else (), 1):
        hh, ww = s_[0].stop - s_[0].start, s_[1].stop - s_[1].start
        own = lab[s_] == i
        if not (2 <= hh <= 9 and ww <= 6) or own.sum() > 48:      # a pane or a tall window, no more
            continue
        if pale_only and not pale_wall(lum, wall, s_):
            continue
        if (exclude and _hits(s_, exclude)) or (only and not _hits(s_, only)):
            continue
        px, py = s_[1].start, s_[0].start
        sh = boost[2] if boost is not None and boost[0] <= px <= boost[1] else share
        v = hash01(px, py, 406) / sh
        if hash01(px, py, 407) < FLIP:
            v = -v
        yy, xx = np.nonzero(own)
        ys.append(yy + py)
        xs.append(xx + px)
        vs.append(np.full(len(yy), v, np.float32))
        kind.append(np.full(len(yy), int(hash01(px, py, 405) * 3)))
        pane.append(np.full(len(yy), i))
    if ys:
        ys, xs = np.concatenate(ys), np.concatenate(xs)
        out = (ys, xs, np.concatenate(vs), np.concatenate(kind), np.concatenate(pane).astype(np.int64),
               rgb[ys, xs].copy())
    else:
        z = np.zeros(0, int)
        out = (z, z, np.zeros(0, np.float32), z, z, np.zeros((0, 3), np.uint8))
    _CACHE[ck] = out
    return out


def from_mask(get):
    """A getter for a scene layer cached as (rgb, mask) (comp-style), as an RGBA getter for `paint`."""
    def f():
        a, m = get()
        return np.dstack([a, (m > 0).astype(np.uint8) * 255])
    return f


def still(c, ys, xs, rgb):
    """The mask of the pixels (ys, xs) still showing the colours rgb (None if none)."""
    if ys is None or not len(ys):
        return None
    keep = np.all(c.a[ys, xs] == rgb, axis=1)
    if not keep.any():
        return None
    m = np.zeros((H, W), bool)
    m[ys[keep], xs[keep]] = True
    return m


def ramp(hour):
    """0 (day) .. 1 (full night): how many of the windows are on."""
    return clamp((daylight.night(hour) - NIGHT_ON) / (NIGHT_FULL - NIGHT_ON))


def paint(c, hour, t, placements):
    """Paint the lit windows of the given placements ((key, get_rgba, x, y[, share[, boost[, pale_only]]])
    of layers on the canvas) where nothing covers them. Returns the mask of the painted pixels (None by
    day)."""
    k = ramp(hour)
    if k <= 0:
        return None
    mask = None
    ep = t // FLIP_T
    for pl in placements:
        opts = pl[-1] if isinstance(pl[-1], dict) else {}
        pl = pl[:-1] if opts else pl
        key, get, x, y = pl[:4]
        share = pl[4] if len(pl) > 4 else SHARE
        boost = pl[5] if len(pl) > 5 else None
        ys, xs, v, kd, pane, src = panes(key, get, share, boost, pale_only=len(pl) > 6 and pl[6],
                                         exclude=opts.get('exclude', ()), only=opts.get('only', ()))
        if not len(ys):
            continue
        on = np.abs(v) < k
        sw = (v < 0) & (((pane * 2654435761 + ep * 40503) >> 9) % 7 < 3)   # one more goes on now and then
        on = np.where(sw, k > 0.5, on)
        sy, sx = ys + int(y), xs + int(x)
        ok = on & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        sy, sx, kd, sr = sy[ok], sx[ok], kd[ok], src[ok]
        seen = np.all(c.a[sy, sx] == sr, axis=1)                    # nothing drawn in front of it
        if not seen.any():
            continue
        sy, sx = sy[seen], sx[seen]
        c.a[sy, sx] = WARM[kd[seen]]
        if mask is None:
            mask = np.zeros((H, W), bool)
        mask[sy, sx] = True
    return mask
