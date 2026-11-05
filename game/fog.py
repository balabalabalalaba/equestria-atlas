"""Fog of war on the world map: uncharted land lies under a deck of drifting clouds that melts away
wherever the balloon flies (like the fog of an RTS map).

The map is cut into CELL x CELL px cells; `reveal(x, y)` clears every cell within REVEAL px of the
balloon (an eased, soft-edged hole: the edge is dithered and puffy, never a hard circle). The
explored grid is saved packed (zlib + base64) in the save file. `explored()` is the charted fraction
of the map.

Drawing (`draw(c, x0, y0, t, hour)`, world layer, at the integer map camera): the cells around the
view are blurred into a smooth density, bilinearly resampled to pixels, roughened by a drifting cloud
noise and then cut into a cloud deck with a bright rim, a shaded body and a dithered cloud shadow on
the land just outside it. The deck takes the map's time-of-day light. ~2 ms per frame. Over the snowy
bands of the map (the Frozen North, the southern ice: where game/weather.py snows) the deck's shaded
side turns a little lilac and its edge gets a one-step darker scallop, so fog never reads as snow.
"""
import base64
import zlib
from functools import lru_cache

import numpy as np

from engine import W, H, col, bayer
from engine import loc_worldmap as LW

CELL = 8
GW, GH = -(-LW.MAP_W // CELL), -(-LW.MAP_H // CELL)
REVEAL = 84                     # px cleared around the balloon
PLACE_REVEAL = 84               # px cleared around a discovered place (old saves)
DRIFT = 0.08                    # px / step the cloud texture drifts
SNOW_N, SNOW_S, SNOW_BLEND = 300, 1440, 40      # the snowy bands: map y < SNOW_N, y > SNOW_S (dithered)
RIM = 0.57                      # the deck's edge band (density cut 0.52 .. RIM): the scallop over snow


def _disc(r):
    n = int(np.ceil(r / CELL)) + 1
    ys, xs = np.mgrid[-n:n + 1, -n:n + 1]
    return n, (xs * CELL) ** 2 + (ys * CELL) ** 2 <= r * r


def _value_noise(cells_list, seed):
    """A 256x256 tileable noise in 0..1: octaves of smoothed value noise (cells per side per octave)."""
    rs = np.random.RandomState(seed)
    out = np.zeros((256, 256), np.float32)
    amp, tot = 1.0, 0.0
    for cells in cells_list:
        g = rs.rand(cells, cells).astype(np.float32)
        k = 256 // cells
        x = (np.arange(256) + 0.5) / k - 0.5
        i0 = np.floor(x).astype(int)
        f = x - i0
        f = f * f * (3 - 2 * f)
        a, b = i0 % cells, (i0 + 1) % cells
        rows = g[:, a] * (1 - f) + g[:, b] * f              # (cells, 256)
        up = rows[a, :] * (1 - f)[:, None] + rows[b, :] * f[:, None]
        out += amp * up
        tot += amp
        amp *= 0.5
    out /= tot
    out -= out.min()
    out /= out.max()
    return out


@lru_cache(maxsize=None)
def _noise():
    """Large soft cloud masses (features ~30-60 px)."""
    return _value_noise((6, 12, 24), 7)


@lru_cache(maxsize=None)
def _puffs():
    """Small cumulus puffs (features ~8-16 px): the billows of the deck and its scalloped rim."""
    return _value_noise((24, 48), 11)


@lru_cache(maxsize=None)
def _deck():
    """A tileable 256x256 cloud deck painted from round puffs, back (top) to front (bottom), each lit
    from the top-left: (tone index 0 light .. 3 deep shade, puff height 0..1). The height scallops the
    edge of the deck along the puffs' outlines."""
    rs = np.random.RandomState(5)
    N = 256
    tone = np.ones((N, N), np.int8)
    height = np.zeros((N, N), np.float32)
    puffs = []
    for row in range(0, N, 7):
        x = rs.rand() * 20
        while x < N:
            r = int(rs.choice((6, 7, 8, 9, 10, 12, 14)))
            puffs.append((row + rs.randint(-3, 4), x, r))
            x += r * (1.0 + rs.rand() * 0.6)
    puffs.sort()
    for cy, cx, r in puffs:
        for oy in (-N, 0, N):
            for ox in (-N, 0, N):
                y0, y1 = int(cy + oy - r), int(cy + oy + r + 1)
                x0, x1 = int(cx + ox - r), int(cx + ox + r + 1)
                if y1 <= 0 or x1 <= 0 or y0 >= N or x0 >= N:
                    continue
                yy, xx = np.mgrid[max(0, y0):min(N, y1), max(0, x0):min(N, x1)]
                dy, dx = yy - (cy + oy), xx - (cx + ox)
                dd = np.sqrt(dx * dx + dy * dy)
                m = dd <= r
                if not m.any():
                    continue
                tt = np.ones(dd.shape, np.int8)
                tt[(dx + dy * 1.4) < -r * 0.55] = 0                     # the lit top-left cap
                tt[dy > r * 0.45] = 2                                     # the shaded underside
                tt[(dd > r - 1.2) & (dy > r * 0.15)] = 3                  # a darker lower outline
                sub_t = tone[max(0, y0):min(N, y1), max(0, x0):min(N, x1)]
                sub_h = height[max(0, y0):min(N, y1), max(0, x0):min(N, x1)]
                sub_t[m] = tt[m]
                sub_h[m] = np.maximum(sub_h[m], (1 - (dd[m] / r) ** 2))
    return tone, height


@lru_cache(maxsize=64)
def _colors(hour10):
    """Fog palette graded to the map's time of day: (rim, base, shade, deep shade), then the same four
    over the snowy bands (lilac shade and deep shade)."""
    base = np.array([[col('white'), col('cloud'), col('#c3cde3'), col('#a3adcf'),
                      col('white'), col('cloud'), col('#cbc3e6'), col('#a89bd0')]], np.uint8)
    a = base.copy()
    try:
        LW._apply_grade(a, hour10 / 10.0)
    except Exception:
        pass
    return a[0]


def _lerp_axis(sub, pos, axis):
    i0 = np.floor(pos).astype(int)
    f = (pos - i0).astype(np.float32)
    n = sub.shape[axis]
    i0c, i1c = np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1)
    if axis == 1:
        return sub[:, i0c] * (1 - f) + sub[:, i1c] * f
    return sub[i0c, :] * (1 - f)[:, None] + sub[i1c, :] * f[:, None]


def _blur(a):
    """3x3 box blur, twice (edges clamped)."""
    for _ in range(2):
        p = np.pad(a, 1, mode='edge')
        a = (p[:-2] + p[1:-1] + p[2:]) / 3.0
        p = np.pad(a, ((0, 0), (1, 1)), mode='edge')
        a = (p[:, :-2] + p[:, 1:-1] + p[:, 2:]) / 3.0
    return a


class Fog:
    def __init__(self):
        self.grid = np.zeros((GH, GW), bool)
        self.rev = 0
        self._last = None
        self._n, self._mask = _disc(REVEAL)
        self._wide = _disc(REVEAL * 1.4)        # (the EXPLORER balloon's WIDE EYES, game/perks.py)

    # -------------------------------------------------------------- state
    def reveal(self, x, y, r=None, wide=False):
        """Clear the fog within r px of world (x, y) (the balloon's own reach when r is None, 40 % more
        when `wide`). Cheap when the balloon has hardly moved."""
        if r is None:
            if self._last is not None and abs(x - self._last[0]) + abs(y - self._last[1]) < 3 \
                    and self._last[2] == wide:
                return
            self._last = (x, y, wide)
            n, mask = self._wide if wide else (self._n, self._mask)
        else:
            n, mask = _disc(r)
        cx, cy = int(x // CELL), int(y // CELL)
        x0, y0 = cx - n, cy - n
        gx0, gy0 = max(0, x0), max(0, y0)
        gx1, gy1 = min(GW, cx + n + 1), min(GH, cy + n + 1)
        if gx1 <= gx0 or gy1 <= gy0:
            return
        m = mask[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
        view = self.grid[gy0:gy1, gx0:gx1]
        if (m & ~view).any():
            view |= m
            self.rev += 1

    def explored(self):
        return float(self.grid.mean())

    def is_clear(self, x, y):
        cx, cy = int(x // CELL), int(y // CELL)
        return 0 <= cx < GW and 0 <= cy < GH and bool(self.grid[cy, cx])

    def encode(self):
        return base64.b64encode(zlib.compress(np.packbits(self.grid).tobytes(), 9)).decode('ascii')

    def decode(self, s):
        try:
            bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(s)), np.uint8))
            self.grid = bits[:GH * GW].reshape(GH, GW).astype(bool)
            self.rev += 1
            return True
        except Exception:
            return False

    # -------------------------------------------------------------- draw
    def density(self, x0, y0, w=W, h=H):
        """Fog density 0..1 for every pixel of the w x h view whose top-left is world (x0, y0)."""
        cx0, cy0 = x0 // CELL - 3, y0 // CELL - 3
        cw, ch = w // CELL + 8, h // CELL + 8
        sub = np.zeros((ch, cw), np.float32)
        gx0, gy0 = max(0, cx0), max(0, cy0)
        gx1, gy1 = min(GW, cx0 + cw), min(GH, cy0 + ch)
        sub[:] = 1.0                                    # outside the map: clear (never seen anyway)
        if gx1 > gx0 and gy1 > gy0:
            sub[gy0 - cy0:gy1 - cy0, gx0 - cx0:gx1 - cx0] = self.grid[gy0:gy1, gx0:gx1]
        sub = _blur(sub)
        px = (np.arange(w) + x0 + 0.5) / CELL - 0.5 - cx0
        py = (np.arange(h) + y0 + 0.5) / CELL - 0.5 - cy0
        clear = _lerp_axis(_lerp_axis(sub, px, 1), py, 0)
        return 1.0 - clear

    def draw(self, c, x0, y0, t, hour=12.0):
        d = self.density(x0, y0)
        if d.max() < 0.3:
            return
        dx = int(t * DRIFT)
        ys = (np.arange(H) + y0) % 256
        xs = (np.arange(W) + x0 + dx) % 256
        tone_t, height_t = _deck()
        tone = tone_t[ys[:, None], xs[None, :]]
        hgt = height_t[ys[:, None], xs[None, :]]
        big = _noise()[ys[:, None], (xs[None, :] // 2)]
        # the edge follows the puffs' outlines (scalloped), loosened by large soft masses
        e = d + (hgt - 0.5) * 0.45 + (big - 0.5) * 0.3
        bay = bayer(H, W, x0, y0)                      # world-anchored: the dither never crawls
        a = c.a
        pal = _colors(int(round(hour * 10)))
        # the cloud shadow on the land just outside the deck (dithered, darker)
        sh = (e > 0.36) & (e <= 0.52) & (bay < (e - 0.36) / 0.16 * 0.6)
        if sh.any():
            a[sh] = (a[sh].astype(np.uint16) * 3 // 4).astype(np.uint8)
        fog = e > 0.52
        if not fog.any():
            return
        if y0 < SNOW_N + SNOW_BLEND or y0 + H > SNOW_S - SNOW_BLEND:     # over the snowy bands
            wy = np.arange(H) + y0
            snow = np.clip(np.maximum(SNOW_N - wy, wy - SNOW_S) / SNOW_BLEND + 0.5, 0, 1)
            snowy = bay < snow[:, None]                                  # a dithered seam
            tone = tone + 4 * snowy
            rim = snowy & (e <= RIM)                                     # the scalloped edge: 1 step darker
            tone[rim] = np.minimum(tone[rim] + 1, 7)
        a[fog] = pal[tone[fog]]

    def draw_overview(self, c, rect, t):
        """The fog on the parchment overview map (rect = x, y, w, h of the map on screen)."""
        ox, oy, ow, oh = rect
        gx = np.clip(((np.arange(ow) + 0.5) * LW.MAP_W / ow / CELL).astype(int), 0, GW - 1)
        gy = np.clip(((np.arange(oh) + 0.5) * LW.MAP_H / oh / CELL).astype(int), 0, GH - 1)
        g = _blur(self.grid.astype(np.float32))[gy[:, None], gx[None, :]]
        bay = np.tile(bayer(4, 4), (oh // 4 + 1, ow // 4 + 1))[:oh, :ow]
        fog = (1 - g) > 0.35 + bay * 0.3
        if not fog.any():
            return
        sub = c.a[oy:oy + oh, ox:ox + ow]
        tone_t, _h = _deck()
        tone = tone_t[(np.arange(oh) * 2 % 256)[:, None], ((np.arange(ow) * 2 + int(t * DRIFT)) % 256)[None, :]]
        tone = np.minimum(tone, 2)
        pal = np.array([col('white'), col('cloud'), col('#c3cde3')], np.uint8)
        sub[fog] = pal[tone[fog]]
