"""Atmosphere & effects: skies, light/glow, colour grading, particles, bubbles.

All functions are deterministic in their arguments (pass the frame index `f`
for animation) so frames can be rendered independently.
"""
import math

import numpy as np

from .core import (W, H, Canvas, col, mix, bayer, hash01, clamp, lerp, seg,
                   smooth, FRAMES_PER_BEAT)
from .palette import SKIES


# ---------------------------------------------------------------- sky
def sky(c, preset_or_colors, y0=0, y1=H, x0=0, x1=W, soft=0.6):
    """Banded, dithered sky gradient. preset: key of palette.SKIES or a colour list."""
    cols = SKIES[preset_or_colors] if isinstance(preset_or_colors, str) else preset_or_colors
    c.gradient(x0, y0, x1 - x0, y1 - y0, cols, vertical=True, soft=soft)


def sky_blend(c, preset_a, preset_b, t, y0=0, y1=H, x0=0, x1=W):
    """Sky that cross-fades between two presets (t: 0 -> a, 1 -> b) using dithering."""
    a = SKIES[preset_a] if isinstance(preset_a, str) else preset_a
    b = SKIES[preset_b] if isinstance(preset_b, str) else preset_b
    n = max(len(a), len(b))
    def resample(cs):
        out = []
        for i in range(n):
            p = i * (len(cs) - 1) / (n - 1)
            j = min(int(p), len(cs) - 2)
            out.append(mix(cs[j], cs[j + 1], p - j))
        return out
    a, b = resample(a), resample(b)
    cols = [mix(ca, cb, t) for ca, cb in zip(a, b)]
    c.gradient(x0, y0, x1 - x0, y1 - y0, cols, vertical=True, soft=0.6)


def stars(c, f, seed=1, n=60, region=(0, 0, W, 100), twinkle=True, colors=('white', 'light_sky', 'lilac')):
    """Scatter of 1px stars (some 'plus'-shaped bright ones) that twinkle."""
    x0, y0, w, h = region
    for i in range(n):
        x = x0 + int(hash01(seed, i, 1) * w)
        y = y0 + int(hash01(seed, i, 2) * h)
        ph = hash01(seed, i, 3)
        on = 1.0 if not twinkle else 0.5 + 0.5 * math.sin(2 * math.pi * (f / (50 + 70 * ph) + ph))
        if on < 0.25:
            continue
        cc = colors[i % len(colors)]
        big = hash01(seed, i, 4) > 0.92
        if big and on > 0.6:
            c.px(x, y, 'white')
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c.px(x + dx, y + dy, mix(cc, 'navy', 0.4))
        else:
            c.px(x, y, cc if on > 0.6 else mix(cc, 'navy', 0.5))


def cloud(c, x, y, w, color='white', shade_color=None, seed=0):
    """A puffy pixel cloud about w px wide; (x, y) is the top-left of its bounding box."""
    shade_color = shade_color or mix(color, 'stone', 0.35)
    h = max(5, w // 3)
    base = y + h
    n = max(3, w // 9)
    for i in range(n):
        u = (i + 0.5) / n
        bump = math.sin(u * math.pi)              # taller in the middle
        r = max(2, int(h * (0.35 + 0.55 * bump) * (0.8 + 0.4 * hash01(seed, i))))
        px = x + int(u * w)
        c.ellipse(px, base - r, r + 1, r, shade_color)
        c.ellipse(px, base - r - 1, r + 1, r, color)
    c.rect(x + 1, base - 2, w - 2, 2, shade_color)
    c.rect(x + 2, base - 3, w - 4, 1, color)


# ---------------------------------------------------------------- light
def glow(c, cx, cy, r, color, strength=0.5, rings=4, ox=0, oy=0):
    """Additive stepped light around (cx, cy). Dithered ring edges keep it pixel-y."""
    cx, cy, r = int(cx), int(cy), max(1, int(r))
    x0, y0, x1, y1 = c._clip(cx - r, cy - r, 2 * r + 1, 2 * r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    lvl = np.clip(1 - d, 0, 1) * rings
    thr = bayer(y1 - y0, x1 - x0, x0 + ox, y0 + oy)
    q = np.floor(lvl + (thr - 0.5) * 0.9) / rings
    q = np.clip(q, 0, 1)[:, :, None]
    reg = c.a[y0:y1, x0:x1].astype(np.float32)
    reg += np.array(col(color), np.float32) * q * strength
    c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def light_rect(c, x, y, w, h, color, strength=0.3, fade='down', ox=0, oy=0):
    """Additive light over a rectangle that fades in a direction ('down','up','left','right',None)."""
    x0, y0, x1, y1 = c._clip(x, y, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    hh, ww = y1 - y0, x1 - x0
    if fade == 'down':
        t = 1 - (np.arange(y0, y1) - y + 0.5) / h
        t = np.broadcast_to(t[:, None], (hh, ww))
    elif fade == 'up':
        t = (np.arange(y0, y1) - y + 0.5) / h
        t = np.broadcast_to(t[:, None], (hh, ww))
    elif fade == 'right':
        t = 1 - (np.arange(x0, x1) - x + 0.5) / w
        t = np.broadcast_to(t[None, :], (hh, ww))
    elif fade == 'left':
        t = (np.arange(x0, x1) - x + 0.5) / w
        t = np.broadcast_to(t[None, :], (hh, ww))
    else:
        t = np.ones((hh, ww))
    thr = bayer(hh, ww, x0 + ox, y0 + oy)
    q = np.clip(np.floor(t * 4 + (thr - 0.5) * 0.9) / 4, 0, 1)[:, :, None]
    reg = c.a[y0:y1, x0:x1].astype(np.float32)
    reg += np.array(col(color), np.float32) * q * strength
    c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def light_beam(c, top, bottom, color, strength=0.25, ox=0, oy=0):
    """Additive quad of light, e.g. window sunbeam on the floor or a lamp cone.

    top/bottom: ((xl, y), (xr, y)) pairs — the left and right x of the beam at
    the top row and bottom row. Fades toward the bottom; dithered.
    """
    (txl, ty), (txr, _) = top
    (bxl, by), (bxr, _) = bottom
    ys = range(max(0, int(ty)), min(c.h, int(by) + 1))
    for y in ys:
        t = seg(y, ty, by)
        xl, xr = int(lerp(txl, bxl, t)), int(lerp(txr, bxr, t))
        if xr < xl:
            xl, xr = xr, xl
        x0, x1 = max(0, xl), min(c.w, xr + 1)
        if x1 <= x0:
            continue
        thr = bayer(1, x1 - x0, x0 + ox, y + oy)[0]
        k = (1 - t * 0.7)
        m = thr < k
        row = c.a[y, x0:x1].astype(np.float32)
        row[m] += np.array(col(color), np.float32) * strength
        c.a[y, x0:x1] = np.clip(row, 0, 255).astype(np.uint8)


def grade(c, mul=(1, 1, 1), add=(0, 0, 0), region=None):
    """Colour grade: rgb = rgb * mul + add (per channel)."""
    if region:
        x0, y0, x1, y1 = c._clip(*region)
        a = c.a[y0:y1, x0:x1]
    else:
        a = c.a
    f = a.astype(np.float32) * np.array(mul, np.float32) + np.array(add, np.float32)
    a[...] = np.clip(f, 0, 255).astype(np.uint8)


GRADES = {
    'none':    ((1.00, 1.00, 1.00), (0, 0, 0)),
    'dawn':    ((0.92, 0.82, 0.92), (14, 4, 10)),
    'morning': ((1.00, 0.98, 0.94), (6, 4, 0)),
    'noon':    ((1.00, 1.00, 1.00), (0, 0, 0)),
    'golden':  ((1.00, 0.86, 0.72), (18, 8, 0)),
    'evening': ((0.78, 0.70, 0.80), (16, 6, 10)),
    'night':   ((0.42, 0.44, 0.66), (4, 4, 16)),
    'lamp':    ((0.70, 0.62, 0.66), (26, 12, 4)),
    'dark':    ((0.22, 0.22, 0.38), (2, 2, 10)),
}


def grade_preset(c, name, amount=1.0, region=None):
    """Apply a named grade from GRADES, optionally partially (amount 0..1)."""
    mul, add = GRADES[name]
    mul = tuple(lerp(1, m, amount) for m in mul)
    add = tuple(a * amount for a in add)
    grade(c, mul, add, region)


def vignette(c, strength=0.35, color='ink'):
    """Stepped dithered vignette toward the corners."""
    h, w = c.h, c.w
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
    lvl = np.clip((d - 0.55) / 0.45, 0, 1) * strength
    m = bayer(h, w) < lvl
    c.a[m] = col(color)


def fade(c, color, t):
    """Dithered fade of the whole frame toward a colour (t: 0..1)."""
    if t <= 0:
        return
    if t >= 1:
        c.fill(color)
        return
    m = bayer(c.h, c.w) < t
    c.a[m] = col(color)


# ---------------------------------------------------------------- particles
def steam(c, x, y, f, color='white', n=3, height=14, spread=3, seed=0, speed=0.35):
    """Wisps of steam rising from (x, y). Pixels thin out as they rise."""
    for i in range(n):
        period = int(height / speed)
        t = ((f + i * period / n + hash01(seed, i) * 7) % period) / period
        py = y - int(t * height)
        px = x + int(math.sin((t * 3 + i) * 2.1) * spread * t) + (i - n // 2)
        if hash01(seed, i, int(f // 3)) < 1 - t * 0.9:
            c.px(px, py, color)
            if t < 0.4:
                c.px(px, py - 1, color)


def sparkle(c, x, y, f, color='white', period=24, phase=0.0, size=2):
    """A 4-point twinkle that grows/shrinks over `period` frames."""
    t = ((f / period) + phase) % 1.0
    s = int(round(math.sin(t * math.pi) * size))
    if s <= 0:
        return
    c.px(x, y, color)
    for k in range(1, s + 1):
        cc = color if k < s else mix(color, 'stone', 0.4)
        c.px(x + k, y, cc)
        c.px(x - k, y, cc)
        c.px(x, y + k, cc)
        c.px(x, y - k, cc)


def sparkles(c, cx, cy, f, radius=12, n=6, colors=('white', 'gold', 'ice'), seed=0, period=30):
    for i in range(n):
        a = hash01(seed, i) * 6.283
        r = radius * (0.3 + 0.7 * hash01(seed, i, 9))
        sparkle(c, int(cx + math.cos(a) * r), int(cy + math.sin(a) * r), f,
                colors[i % len(colors)], period=period, phase=hash01(seed, i, 5))


def dust_motes(c, f, region, n=12, color='butter', seed=0):
    """Slow-floating 1px motes (for sunbeams)."""
    x0, y0, w, h = region
    for i in range(n):
        sx = hash01(seed, i, 1) * w
        sy = hash01(seed, i, 2) * h
        px = x0 + int((sx + f * 0.05 + math.sin(f / 40 + i) * 3) % w)
        py = y0 + int((sy - f * 0.08) % h)
        if hash01(seed, i, f // 20) > 0.25:
            c.px(px, py, color)


def confetti(c, f, t0, cx, cy, n=24, seed=0, colors=('gold', 'hot_pink', 'turquoise', 'lavender', 'leaf')):
    """Burst of confetti from (cx, cy) starting at frame t0 (gravity, ~2 s)."""
    dt = f - t0
    if dt < 0 or dt > 70:
        return
    for i in range(n):
        a = -math.pi / 2 + (hash01(seed, i) - 0.5) * 2.4
        v = 1.6 + 1.6 * hash01(seed, i, 2)
        x = cx + math.cos(a) * v * dt
        y = cy + math.sin(a) * v * dt + 0.06 * dt * dt
        cc = colors[i % len(colors)]
        c.px(int(x), int(y), cc)
        if (dt // 3 + i) % 2:
            c.px(int(x) + 1, int(y), cc)


def rain(c, f, region=(0, 0, W, H), n=80, color='light_sky', seed=0, slant=1, speed=6, length=3):
    x0, y0, w, h = region
    for i in range(n):
        x = x0 + (hash01(seed, i) * w + f * slant * speed / 4) % w
        y = y0 + (hash01(seed, i, 2) * h + f * speed) % h
        c.line(x, y, x - slant, y + length, color)


def shake(f, amp=1, seed=0):
    """Small deterministic camera shake offset (dx, dy)."""
    if amp <= 0:
        return 0, 0
    return (int(round((hash01(seed, f, 1) - 0.5) * 2 * amp)),
            int(round((hash01(seed, f, 2) - 0.5) * 2 * amp)))


# ---------------------------------------------------------------- bubbles / UI-in-world
def bubble(c, x, y, w, h, tail=None, fill='white', border='ink', thought=False):
    """Speech (or thought) bubble with top-left (x, y), size w x h (inner).

    tail: (tx, ty) point the tail aims at (usually the speaker's mouth/head).
    """
    x, y, w, h = int(x), int(y), int(w), int(h)
    if tail and not thought:
        tx, ty = int(tail[0]), int(tail[1])
        below = ty >= y + h
        bx = int(clamp(tx, x + 3, x + w - 7))
        by = y + h - 1 if below else y
        tri = [(bx, by), (bx + 5, by), (tx + 0.5, ty + 0.5)]
        # border triangle (1px bigger), then fill
        c.poly([(bx - 1, by), (bx + 6, by), (tx + 0.5, ty + (1.5 if below else -0.5))], border)
        c.poly(tri, fill)
    c.round_rect(x - 1, y - 1, w + 2, h + 2, border, r=2)
    c.round_rect(x, y, w, h, fill, r=1)
    if tail and not thought:
        # reopen the seam between bubble body and tail
        c.hline(bx, bx + 4, (y + h) if below else (y - 1), fill)
    if tail and thought:
        tx, ty = int(tail[0]), int(tail[1])
        mx, my = (x + w // 2 + tx) // 2, (y + h + ty) // 2
        c.circle(mx, my, 2, border)
        c.circle(mx, my, 1, fill)
        c.rect(tx - 1, ty - 1, 2, 2, border)


def pop_scale(f, t0, dur=8):
    """0..1 'pop-in' factor with overshoot for UI elements appearing at frame t0."""
    from .core import ease_out_back
    return ease_out_back(seg(f, t0, t0 + dur))
