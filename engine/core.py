"""Core canvas, colour handling, timing and easing helpers.

Everything renders at the native resolution W x H (320 x 180) and is upscaled
6x with nearest-neighbour to 1920 x 1080 at encode time. Never draw
fractional/anti-aliased pixels: all drawing snaps to integer coordinates.

Scenes must be PURE FUNCTIONS of the frame index (no state carried between
frames) so frames can be rendered in any order and in parallel.
"""
import math
from functools import lru_cache

import numpy as np

from .palette import PAL

W, H = 320, 180
FPS = 30
BPM = 100                     # shared by picture + music; FPS*60/BPM must be an integer
assert (FPS * 60) % BPM == 0, 'pick a BPM where FPS*60/BPM is whole (e.g. 90, 100, 120, 150)'
FRAMES_PER_BEAT = FPS * 60 // BPM   # 18 at 100 BPM
FRAMES_PER_BAR = FRAMES_PER_BEAT * 4
SCALE = 6                     # 320x180 -> 1920x1080


# ---------------------------------------------------------------- colours
@lru_cache(maxsize=4096)
def _col_str(c):
    if c.startswith('#'):
        h = c[1:]
        if len(h) == 3:
            h = ''.join(ch * 2 for ch in h)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    try:
        return PAL[c]
    except KeyError:
        raise KeyError(f'unknown palette colour {c!r}') from None


def col(c):
    """Resolve a colour spec (palette name, '#rrggbb', or (r,g,b)) to an RGB tuple."""
    if isinstance(c, str):
        return _col_str(c)
    return (int(c[0]), int(c[1]), int(c[2]))


def mix(a, b, t):
    """Linear mix of two colours, t=0 -> a, t=1 -> b. Returns an RGB tuple."""
    a, b = col(a), col(b)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def shade(c, k):
    """Multiply a colour's brightness by k (k<1 darker, k>1 lighter)."""
    c = col(c)
    return tuple(max(0, min(255, int(round(v * k)))) for v in c)


# ---------------------------------------------------------------- math / easing
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def ilerp(a, b, t):
    """Integer lerp (rounded) — use for pixel positions."""
    return int(round(a + (b - a) * t))


def seg(f, a, b):
    """Progress 0..1 of frame f through the interval [a, b] (clamped)."""
    if b <= a:
        return 1.0 if f >= b else 0.0
    return clamp((f - a) / (b - a))


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_in(t):
    t = clamp(t)
    return t * t


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_bounce(t):
    t = clamp(t)
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def pingpong(t):
    """0->1->0 triangle over t in [0,1]."""
    t = t % 1.0
    return 2 * t if t < 0.5 else 2 - 2 * t


def wave(f, period, amp=1.0, phase=0.0):
    """Sine wave in frames: amp * sin(2*pi*(f/period + phase))."""
    return amp * math.sin(2 * math.pi * (f / period + phase))


def beat(f):
    """Beat index of frame f (music is 100 BPM, 18 frames per beat)."""
    return f // FRAMES_PER_BEAT


def beat_phase(f):
    """Position 0..1 within the current beat."""
    return (f % FRAMES_PER_BEAT) / FRAMES_PER_BEAT


def bar(f):
    return f // FRAMES_PER_BAR


def on_beat(f, every=1, offset=0):
    """True on the first frame of every `every`-th beat."""
    return (f - offset) % (FRAMES_PER_BEAT * every) == 0


def hash01(*xs):
    """Deterministic pseudo-random float in [0,1) from integers."""
    h = 2166136261
    for x in xs:
        h ^= (int(x) & 0xffffffff)
        h = (h * 16777619) & 0xffffffff
        h ^= h >> 13
        h = (h * 0x5bd1e995) & 0xffffffff
        h ^= h >> 15
    return (h & 0xffffff) / float(0x1000000)


def rng(seed):
    """Deterministic numpy Generator."""
    return np.random.default_rng(int(seed) & 0xffffffff)


def keyframes(f, keys, ease=smooth):
    """Interpolate a value through [(frame, value), ...] keys (numbers or tuples)."""
    if f <= keys[0][0]:
        return keys[0][1]
    for (fa, va), (fb, vb) in zip(keys, keys[1:]):
        if f <= fb:
            t = ease(seg(f, fa, fb))
            if isinstance(va, (tuple, list)):
                return tuple(lerp(a, b, t) for a, b in zip(va, vb))
            return lerp(va, vb, t)
    return keys[-1][1]


# ---------------------------------------------------------------- dithering
BAYER4 = (np.array([[0, 8, 2, 10],
                    [12, 4, 14, 6],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5]], np.float32) + 0.5) / 16.0

_BAYER_FULL = np.tile(BAYER4, (H // 4 + 64, W // 4 + 64))


def bayer(h, w, ox=0, oy=0):
    """Bayer threshold map (h x w) in (0,1), anchored to screen coords (ox, oy)."""
    yy = (np.arange(h) + oy) % 4
    xx = (np.arange(w) + ox) % 4
    return BAYER4[yy[:, None], xx[None, :]]


# ---------------------------------------------------------------- canvas
class Canvas:
    """An RGB pixel buffer (numpy uint8 array `a`, shape (h, w, 3)).

    All coordinates are integers in pixels; drawing is clipped to the canvas.
    Colours accept palette names, '#hex', or (r,g,b).
    """
    __slots__ = ('a',)

    def __init__(self, w=W, h=H, color=None, arr=None):
        if arr is not None:
            self.a = arr
        else:
            self.a = np.zeros((h, w, 3), np.uint8)
            if color is not None:
                self.a[:] = col(color)

    @property
    def w(self):
        return self.a.shape[1]

    @property
    def h(self):
        return self.a.shape[0]

    def copy(self):
        return Canvas(arr=self.a.copy())

    def fill(self, c):
        self.a[:] = col(c)
        return self

    # --- clipping helper
    def _clip(self, x, y, w, h):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x) + int(w)), min(self.h, int(y) + int(h))
        return x0, y0, x1, y1

    # --- primitives
    def px(self, x, y, c):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.a[y, x] = col(c)

    def rect(self, x, y, w, h, c):
        x0, y0, x1, y1 = self._clip(x, y, w, h)
        if x1 > x0 and y1 > y0:
            self.a[y0:y1, x0:x1] = col(c)

    def frame(self, x, y, w, h, c):
        """1px rectangle outline."""
        self.rect(x, y, w, 1, c)
        self.rect(x, y + h - 1, w, 1, c)
        self.rect(x, y, 1, h, c)
        self.rect(x + w - 1, y, 1, h, c)

    def round_rect(self, x, y, w, h, c, r=1):
        """Filled rectangle with 1-2 px chipped corners (pixel-art rounded)."""
        x, y, w, h = int(x), int(y), int(w), int(h)
        if r <= 0:
            return self.rect(x, y, w, h, c)
        self.rect(x + r, y, w - 2 * r, h, c)
        self.rect(x, y + r, w, h - 2 * r, c)
        if r >= 2:
            self.rect(x + 1, y + 1, w - 2, h - 2, c)

    def hline(self, x0, x1, y, c):
        if x1 < x0:
            x0, x1 = x1, x0
        self.rect(x0, y, x1 - x0 + 1, 1, c)

    def vline(self, x, y0, y1, c):
        if y1 < y0:
            y0, y1 = y1, y0
        self.rect(x, y0, 1, y1 - y0 + 1, c)

    def line(self, x0, y0, x1, y1, c):
        """Bresenham line (pixel perfect)."""
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        cc = col(c)
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        w, h, a = self.w, self.h, self.a
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                a[y0, x0] = cc
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx, cy, r, c, fill=True):
        """Pixel circle centred on (cx, cy) with radius r."""
        self.ellipse(cx, cy, r, r, c, fill)

    def ellipse(self, cx, cy, rx, ry, c, fill=True):
        cx, cy = int(round(cx)), int(round(cy))
        rx, ry = max(0.5, rx), max(0.5, ry)
        x0, y0, x1, y1 = self._clip(cx - rx - 1, cy - ry - 1, 2 * rx + 3, 2 * ry + 3)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = ((xx - cx) / (rx + 0.5)) ** 2 + ((yy - cy) / (ry + 0.5)) ** 2
        m = d <= 1.0
        if not fill:
            di = ((xx - cx) / (rx - 0.5)) ** 2 + ((yy - cy) / (ry - 0.5)) ** 2 if rx > 1 and ry > 1 else d * 0 + 2
            m &= di > 1.0
        self.a[y0:y1, x0:x1][m] = col(c)

    def poly(self, pts, c):
        """Filled polygon [(x,y), ...] (even-odd scanline fill, pixel centres)."""
        pts = [(float(x), float(y)) for x, y in pts]
        ys = [p[1] for p in pts]
        ymin, ymax = max(0, int(math.floor(min(ys)))), min(self.h - 1, int(math.ceil(max(ys))))
        cc = col(c)
        n = len(pts)
        for y in range(ymin, ymax + 1):
            yc = y + 0.5
            xs = []
            for i in range(n):
                (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
                if (ya <= yc < yb) or (yb <= yc < ya):
                    xs.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
            xs.sort()
            for i in range(0, len(xs) - 1, 2):
                xa = max(0, int(math.ceil(xs[i] - 0.5)))
                xb = min(self.w - 1, int(math.floor(xs[i + 1] - 0.5)))
                if xb >= xa:
                    self.a[y, xa:xb + 1] = cc

    def tri(self, p0, p1, p2, c):
        self.poly([p0, p1, p2], c)

    # --- dithered / blended fills
    def dither(self, x, y, w, h, c, level, ox=0, oy=0):
        """Ordered-dither fill: covers ~`level` (0..1) of the pixels with colour c.

        The pattern is anchored to screen coordinates so adjacent dithers tile.
        """
        x0, y0, x1, y1 = self._clip(x, y, w, h)
        if x1 <= x0 or y1 <= y0 or level <= 0:
            return
        m = bayer(y1 - y0, x1 - x0, x0 + ox, y0 + oy) < level
        self.a[y0:y1, x0:x1][m] = col(c)

    def blend(self, x, y, w, h, c, alpha):
        """True alpha blend of a solid colour over a rect. Use sparingly (glass, UI)."""
        x0, y0, x1, y1 = self._clip(x, y, w, h)
        if x1 <= x0 or y1 <= y0:
            return
        reg = self.a[y0:y1, x0:x1].astype(np.float32)
        reg += (np.array(col(c), np.float32) - reg) * float(alpha)
        self.a[y0:y1, x0:x1] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)

    def gradient(self, x, y, w, h, colors, vertical=True, soft=0.5, ox=0, oy=0):
        """Banded gradient through `colors` with Bayer-dithered band edges.

        colors: list of colour specs, first at top/left. soft: width (0..1) of
        the dithered transition zone between bands (0 = hard bands).
        """
        x0, y0, x1, y1 = self._clip(x, y, w, h)
        if x1 <= x0 or y1 <= y0:
            return
        cols = np.array([col(c) for c in colors], np.uint8)
        n = len(cols)
        if n == 1:
            self.a[y0:y1, x0:x1] = cols[0]
            return
        span = h if vertical else w
        if vertical:
            t = (np.arange(y0, y1) - int(y) + 0.5) / span
            t = np.broadcast_to(t[:, None], (y1 - y0, x1 - x0))
        else:
            t = (np.arange(x0, x1) - int(x) + 0.5) / span
            t = np.broadcast_to(t[None, :], (y1 - y0, x1 - x0))
        pos = t * (n - 1)
        i = np.clip(np.floor(pos).astype(int), 0, n - 2)
        u = pos - i
        if soft > 0:
            u2 = np.clip((u - 0.5) / soft + 0.5, 0, 1)
        else:
            u2 = (u >= 0.5).astype(np.float32)
        thr = bayer(y1 - y0, x1 - x0, x0 + ox, y0 + oy)
        idx = i + (u2 > thr)
        self.a[y0:y1, x0:x1] = cols[idx]

    # --- compositing
    def blit(self, src, x, y, flip=False):
        """Draw a Sprite / RGBA array / Canvas with its top-left at (x, y).

        RGBA pixels with alpha < 128 are transparent (hard alpha only).
        """
        from .sprite import Sprite
        if isinstance(src, Sprite):
            arr = src.rgba
        elif isinstance(src, Canvas):
            arr = src.a
        else:
            arr = src
        if flip:
            arr = arr[:, ::-1]
        x, y = int(round(x)), int(round(y))
        sh, sw = arr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + sw), min(self.h, y + sh)
        if x1 <= x0 or y1 <= y0:
            return
        sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
        if sub.shape[2] == 4:
            m = sub[:, :, 3] >= 128
            self.a[y0:y1, x0:x1][m] = sub[:, :, :3][m]
        else:
            self.a[y0:y1, x0:x1] = sub

    def crop(self, x, y, w, h):
        """Return a new Canvas copied from a region (clipped, padded with ink)."""
        out = Canvas(w, h, 'ink')
        out.blit(self, -x, -y)
        return out
