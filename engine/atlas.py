"""Shared atlas pieces used by every segment (owned by the director).

  draw_balloon(c, x, y, f, size='m', flip=False, flame=False)
      The explorer's hot-air balloon: the "mascot" of the film. (x, y) = bottom-centre of the basket.
      size: 'xs' (5x8, far away / world map), 's' (9x14), 'm' (13x21), 'l' (19x30).
      It bobs by itself (1 px, slow) unless bob=False. flame=True shows the burner flicker.
  callout(c, x, y, label, f, t0, t1=None, side='up', length=12, font='tiny')
      Landmark name tag: a blinking marker at (x, y) (the landmark), a leader line, and a gold-rimmed
      plate with the label. Pops in at t0 (ease-out-back), fades out over 6 frames from t1.
  atlas_card(c, f, t0, t1, kind, facts, x=4, y=None)
      'ATLAS ENTRY' fact card at the bottom-left: kind line (e.g. 'ROYAL CAPITAL') + up to 2 fact lines.
  iris(c, t, cx=W//2, cy=H//2, color='ink')
      Circular iris: t=0 fully open, t=1 fully closed (to a point at cx, cy).
  map_pin(c, x, y, f, color='hot_pink')  — small map pin icon whose tip is at (x, y).
"""
import math
from functools import lru_cache

import numpy as np

from .core import W, H, Canvas, col, mix, shade, seg, clamp, ease_out_back, ease_in, hash01
from .sprite import Sprite
from .font import text, text_width

# ------------------------------------------------------------------ balloon
_B_SIZES = {          # envelope radius, envelope height, rope len, basket w, basket h
    'xs': (2, 4, 1, 3, 1),
    's': (4, 9, 2, 3, 2),
    'm': (6, 14, 3, 5, 3),
    'l': (9, 20, 4, 7, 4),
}
_STRIPES = ('hot_pink', 'lavender')
_STRIPES_SH = ('rose', 'violet')
_STRIPES_HI = ('pink', 'lilac')


_XS = Sprite.from_ascii('''
    .PAP.
    PAPAp
    PAPAp
    .PAp.
    ..g..
    .t.t.
    .bbb.
''', {'P': 'hot_pink', 'A': 'lavender', 'p': 'violet', 'g': 'gold', 't': 'tan', 'b': 'copper'})


@lru_cache(maxsize=None)
def _balloon_sprite(size, flame):
    if size == 'xs':
        return _XS
    R, He, rope, bw, bh = _B_SIZES[size]
    w = 2 * R + 3
    h = He + rope + bh + 1
    rgba = np.zeros((h, w, 4), np.uint8)
    cx = w // 2
    # envelope: round top (circle radius R centred at row R), tapering cone down to the skirt
    top_h = R
    skirt_w = max(1, bw // 2)
    inside = np.zeros((He, w), bool)
    for y in range(He):
        if y <= top_h:
            dy = (top_h - y) - 0.3
            hw = math.sqrt(max(0.0, (R + 0.5) ** 2 - dy * dy))
        else:
            t = (y - top_h) / max(1, He - 1 - top_h)
            hw = (R + 0.5) * (1 - t) ** 0.85 + skirt_w * t
        x0 = int(round(cx - hw))
        x1 = int(round(cx + hw))
        inside[y, max(0, x0):min(w, x1 + 1)] = True
    # colour the envelope: vertical gore stripes, shade on the right, highlight top-left
    n_gores = 5 if R >= 4 else 3
    for y in range(He):
        xs = np.nonzero(inside[y])[0]
        if len(xs) == 0:
            continue
        x0, x1 = xs[0], xs[-1]
        span = max(1, x1 - x0 + 1)
        for x in xs:
            u = (x - x0 + 0.5) / span           # 0..1 across this row (gores bulge with the shape)
            g = int(u * n_gores) % 2
            c = col(_STRIPES[g])
            if u > 0.68 or y > He - 3:
                c = col(_STRIPES_SH[g])
            if u < 0.34 and y < He * 0.45 and R >= 4:
                c = col(_STRIPES_HI[g])
            rgba[y, x, :3] = c
            rgba[y, x, 3] = 255
    # outline (darker purple) around the envelope
    edge = np.zeros_like(inside)
    pad = np.pad(inside, 1)
    nb = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    edge = inside & ~nb
    if R >= 4:
        rgba[:He][edge, :3] = col('eggplant')
    # specular glint
    if R >= 4:
        gy, gx = max(1, R // 2), cx - R // 2
        rgba[gy, gx, :3] = col('white')
        if R >= 6:
            rgba[gy + 1, gx, :3] = col('lilac')
    # gold five-point star emblem at the centre on bigger balloons
    if R >= 6:
        sy = R - 1
        star = ('..g..', 'ggggg', '.ggg.', '.g.g.')
        for j, row in enumerate(star):
            for i, ch in enumerate(row):
                if ch == 'g':
                    rgba[sy + j, cx - 2 + i, :3] = col('gold')
                    rgba[sy + j, cx - 2 + i, 3] = 255
        rgba[sy + 1, cx, :3] = col('butter')
    # skirt band
    sk = He - 1
    for x in range(cx - skirt_w, cx + skirt_w + 1):
        rgba[sk, x, :3] = col('gold' if R >= 4 else 'violet')
        rgba[sk, x, 3] = 255
    # ropes
    for r in range(rope):
        y = He + r
        for x in (cx - (bw // 2), cx + (bw // 2)):
            rgba[y, x, :3] = col('tan')
            rgba[y, x, 3] = 255
        if flame and r == 0:
            rgba[y, cx, :3] = col('amber')
            rgba[y, cx, 3] = 255
    # basket (wicker)
    by = He + rope
    for yy in range(bh):
        for xx in range(bw):
            x = cx - bw // 2 + xx
            c = 'clay' if yy == 0 else ('copper' if (xx + yy) % 2 else 'rust')
            if xx == bw - 1 and yy > 0:
                c = 'rust'
            rgba[by + yy, x, :3] = col(c)
            rgba[by + yy, x, 3] = 255
    return Sprite(rgba)


def balloon_size(size='m'):
    """(w, h) of the balloon sprite."""
    s = _balloon_sprite(size, False)
    return s.w, s.h


def draw_balloon(c, x, y, f=0, size='m', flip=False, flame=False, bob=True):
    """Explorer's hot-air balloon; (x, y) = bottom-centre of the basket."""
    fl = flame and (f // 3) % 2 == 0
    s = _balloon_sprite(size, fl)
    dy = 0
    if bob:
        dy = int(round(math.sin(f * 2 * math.pi / 72.0) * (1 if size in ('xs', 's') else 1.4)))
    s.draw(c, int(x), int(y) + dy, flip=flip, anchor='bottom')
    if flame and size in ('m', 'l') and fl:
        R, He, rope, bw, bh = _B_SIZES[size]
        c.px(int(x), int(y) + dy - bh - rope, 'butter')


# ------------------------------------------------------------------ map pin
_PIN = Sprite.from_ascii('''
    .ooo.
    oPPPo
    oPwPo
    oPPPo
    .oPo.
    ..o..
''', {'o': 'eggplant', 'P': 'hot_pink', 'w': 'white'})


def map_pin(c, x, y, f=0, color='hot_pink', bounce=False):
    """Small map pin whose tip touches (x, y)."""
    s = _PIN if color == 'hot_pink' else _PIN.recolor({col('hot_pink'): col(color)})
    dy = -int(abs(math.sin(f * math.pi / 18.0)) * 2) if bounce else 0
    s.draw(c, int(x), int(y) + 1 + dy, anchor='bottom')


# ------------------------------------------------------------------ callout
def callout(c, x, y, label, f, t0, t1=None, side='up', length=12, font='tiny', rim='gold'):
    """Landmark name tag pointing at (x, y). side: 'up' | 'down' | 'left' | 'right'."""
    if f < t0 or (t1 is not None and f > t1 + 6):
        return
    k = ease_out_back(seg(f, t0, t0 + 8))
    if t1 is not None and f > t1:
        k = 1 - ease_in(seg(f, t1, t1 + 6))
    if k <= 0.02:
        return
    x, y = int(x), int(y)
    # marker: blinking diamond on the landmark
    on = (f - t0) // 9 % 2 == 0
    mc = 'white' if on else rim
    for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
        c.px(x + dx, y + dy, 'ink')
    c.px(x, y, mc)
    L = int(round(length * min(1.0, k)))
    tw = text_width(label, font)
    th = 5 if font == 'tiny' else 7
    pw, ph = tw + 6, th + 4
    if side == 'up':
        ex, ey = x, y - 2 - L
        c.vline(x, ey, y - 2, 'ink')
        px, py = ex - pw // 2, ey - ph + 1
    elif side == 'down':
        ex, ey = x, y + 2 + L
        c.vline(x, y + 2, ey, 'ink')
        px, py = ex - pw // 2, ey
    elif side == 'left':
        ex, ey = x - 2 - L, y
        c.hline(ex, x - 2, y, 'ink')
        px, py = ex - pw + 1, ey - ph // 2
    else:
        ex, ey = x + 2 + L, y
        c.hline(x + 2, ex, y, 'ink')
        px, py = ex, ey - ph // 2
    if k < 0.6:            # still growing: just the leader + a dot
        c.px(ex, ey, rim)
        return
    px = max(1, min(W - pw - 1, px))
    py = max(1, min(H - ph - 1, py))
    c.round_rect(px, py, pw, ph, rim, r=1)
    c.rect(px + 1, py + 1, pw - 2, ph - 2, 'ink')
    c.blend(px + 1, py + 1, pw - 2, 1, 'white', 0.10)
    text(c, px + 3, py + 2, label, 'white', font=font)


# ------------------------------------------------------------------ atlas card
_SCROLL = Sprite.from_ascii('''
    .tttttt.
    tPPPPPPt
    .PllllP.
    .PPPPPP.
    .PllllP.
    .PPPPPP.
    tPPPPPPt
    .tttttt.
''', {'t': 'clay', 'P': 'paper', 'l': 'tan'})


def atlas_card(c, f, t0, t1, kind, facts=(), x=4, y=None, label='ATLAS ENTRY'):
    """Small fact card (bottom-left). kind: one short line; facts: up to 2 short lines (tiny caps)."""
    if f < t0 or f > t1 + 8:
        return
    facts = list(facts)[:2]
    lines = [kind] + facts
    w = max(text_width(s, 'tiny') for s in lines + [label]) + 20
    h = 9 + 7 * len(lines) + 3
    if y is None:
        y = H - h - 4
    k = ease_out_back(seg(f, t0, t0 + 9)) if f <= t1 else 1 - ease_in(seg(f, t1, t1 + 8))
    xx = int(-w - 2 + (w + 2 + x) * k)
    c.round_rect(xx, y, w, h, 'gold', r=1)
    c.rect(xx + 1, y + 1, w - 2, h - 2, 'ink')
    c.blend(xx + 1, y + 1, w - 2, 1, 'white', 0.10)
    c.blit(_SCROLL, xx + 3, y + 3)
    text(c, xx + 14, y + 3, label, 'gold', font='tiny')
    for i, s in enumerate(lines):
        text(c, xx + 14, y + 11 + 7 * i, s, 'white' if i == 0 else 'stone', font='tiny')


# ------------------------------------------------------------------ iris
_YY, _XX = np.mgrid[0:H, 0:W]


def iris(c, t, cx=W // 2, cy=H // 2, color='ink'):
    """Circular iris wipe. t=0 open, t=1 closed. Use 1-t to open."""
    if t <= 0:
        return
    if t >= 1:
        c.a[:] = col(color)
        return
    rmax = math.hypot(max(cx, W - cx), max(cy, H - cy)) + 2
    r = rmax * (1 - t)
    d2 = (_XX - cx) ** 2 + (_YY - cy) ** 2
    c.a[d2 > r * r] = col(color)
    # thin gold rim on the iris edge
    ring = (d2 > (r - 1) ** 2) & (d2 <= r * r)
    c.a[ring] = col('gold')
