"""Pixel props for the places' troubles (game/quests.py): each drawn broken (the problem the player
notices) and mended (how the place looks for good once helped).

kinds: 'garden' (a flower bed: wilted -> blooming), 'lanterns' (a festival string: dark -> lit),
'lamp' (a lamp post: dark, cracked -> crystal light), 'fountain' (dry, cracked -> rainbow jets),
'stall' (a market cart: faded, empty -> bright awning and the goods delivered), 'bunting' (bare poles
-> rainbow pennants), 'fireworks' (an empty rack -> rockets that go up now and then), 'board' (an
empty notice board -> a trail map and posters).

`draw_prop(c, kind, x, ground, t, fixed, item, tint)`: x = centre, ground = the y the prop stands on
(screen px). The body is a cached sprite, lit like the scene (tint, as the balloon), then lights,
water and rockets are drawn live over it (untinted: lights glow). Pure: depends only on t.

A helped place's festival (game/quests.py draws it in every area): `pennant_string`, `roof_flag`,
`maypole`, `balloons_rising` (party balloons by day, sky lanterns at night), `confetti_drift`, each in
the place's own colours (FESTIVE), and `reveal_band`, the light band that sweeps it in after the mend.
`question` / `tiny_question`: the '?' over a place still to explore (world map / overview).
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, col, fx, hash01

from . import ui
from .items import ITEMS

KEY = (255, 0, 254)
SIZE = {'garden': (34, 18), 'lanterns': (48, 30), 'lamp': (12, 30), 'fountain': (28, 20),
        'stall': (34, 28), 'bunting': (40, 26), 'fireworks': (22, 18), 'board': (24, 24)}
RAINBOW = ('red', 'orange', 'butter', 'leaf', 'sky', 'violet')


def height(kind):
    return SIZE[kind][1]


# ------------------------------------------------------------------ bodies (local coords, ground = h)
def _garden(c, w, h, fixed, ph, item):
    c.rect(0, h - 8, w, 1, 'brew_dk')
    c.rect(0, h - 7, w, 6, 'clay')
    for x in range(3, w, 8):
        c.vline(x, h - 7, h - 2, 'brew')
    c.hline(1, w - 2, h - 6, 'tan')
    c.rect(0, h - 1, w, 1, 'brew_dk')
    c.rect(1, h - 9, w - 2, 1, 'brew_dk')                    # the soil
    for i, x in enumerate(range(3, w - 2, 4)):
        if not fixed:
            ln = 3 + (i * 5) % 3
            c.vline(x, h - 9 - ln, h - 10, 'tan')
            c.px(x + (1 if i % 2 else -1), h - 10 - ln, 'sand')      # drooping heads
            c.px(x + (2 if i % 2 else -2), h - 9 - ln, 'stone')
            continue
        ln = 5 + (i * 7) % 4
        sw = (1 if (i + ph) % 2 else 0) if i % 3 else 0
        c.vline(x, h - 9 - ln, h - 10, 'green')
        c.px(x + 1, h - 12, 'leaf')
        fc = ('hot_pink', 'butter', 'violet', 'white', 'salmon', 'sky')[i % 6]
        fy, fx_ = h - 10 - ln, x + sw
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            c.px(fx_ + dx, fy + dy, fc)
        c.px(fx_, fy, 'gold')


def _rope_y(u, sag):
    return 5 + sag * (1 - (2 * u - 1) ** 2)


def _lanterns(c, w, h, fixed, ph, item):
    for px in (1, w - 3):
        c.rect(px, 4, 2, h - 4, 'brew')
        c.px(px, 4, 'tan')
        c.rect(px - 1, 2, 4, 2, 'gold' if fixed else 'stone')
        c.rect(px - 1, h - 2, 4, 2, 'brew_dk')
    sag = 6 if fixed else 9
    pts = [(int(2 + u * (w - 5)), int(round(_rope_y(u, sag)))) for u in np.linspace(0, 1, w - 4)]
    for x, y in pts:
        c.px(x, y, 'brew_dk')
    cols = ('red', 'orange', 'hot_pink', 'gold', 'red')
    for k, u in enumerate((0.17, 0.33, 0.5, 0.67, 0.83)):
        if not fixed and k == 3:
            continue                                              # one has fallen off
        x = int(2 + u * (w - 5))
        y = int(round(_rope_y(u, sag))) + 1 + ((k + ph) % 2 if fixed else 0)
        if fixed:
            c.rect(x - 1, y, 3, 1, 'ink')
            c.rect(x - 1, y + 1, 3, 3, cols[k])
            c.px(x, y + 2, 'butter')
            c.px(x, y + 4, 'gold')
        else:                                                     # unlit paper lanterns: dull colours
            dull = ('maroon', 'brick', 'wine', 'rust', 'maroon')[k]
            c.rect(x - 1, y, 3, 1, 'ink')
            c.rect(x - 1, y + 1, 3, 3, dull)
            c.px(x, y + 2, 'plum')
            c.px(x, y + 4, 'brew')
    if not fixed:                                                 # the fallen lantern on the ground
        fx_ = int(2 + 0.67 * (w - 5)) + 2
        c.rect(fx_, h - 3, 3, 2, 'brick')
        c.px(fx_ + 1, h - 3, 'ink')


def _lamp(c, w, h, fixed, ph, item):
    c.rect(3, h - 2, 6, 2, 'steel_dk')
    c.rect(5, 8, 2, h - 9, 'steel_dk')
    c.px(5, 10, 'steel')
    c.rect(2, 0, 8, 1, 'ink')
    c.rect(1, 1, 10, 1, 'steel_dk')
    c.rect(2, 2, 8, 6, 'ink')
    if fixed:
        c.rect(3, 3, 6, 4, 'crystal_lt')
        c.rect(4, 3, 2, 3, 'white')
        c.px(7, 5, 'crystal_pk')
        c.px(6, 4, 'ice')
    else:
        c.rect(3, 3, 6, 4, 'slate')
        c.px(4, 3, 'charcoal')
        c.px(5, 4, 'charcoal')
        c.px(6, 5, 'charcoal')
        c.px(7, 6, 'charcoal')
    c.rect(3, 8, 6, 1, 'steel_dk')


def _fountain(c, w, h, fixed, ph, item):
    cx = w // 2
    c.ellipse(cx, h - 4, 13, 3, 'stone')
    c.ellipse(cx, h - 5, 12, 2, 'light_sky' if fixed else 'tan')
    c.rect(1, h - 4, w - 2, 3, 'stone')
    c.hline(1, w - 2, h - 4, 'cloud')
    c.rect(1, h - 1, w - 2, 1, 'slate')
    c.rect(cx - 2, 6, 4, h - 10, 'cloud')
    c.vline(cx + 1, 6, h - 5, 'stone')
    c.rect(cx - 5, 5, 10, 2, 'stone')
    c.hline(cx - 5, cx + 4, 5, 'cloud')
    if fixed:
        c.hline(cx - 9 + ph * 2, cx - 6 + ph * 2, h - 5, 'white')
        c.hline(cx + 3 - ph * 2, cx + 5 - ph * 2, h - 6, 'white')
        c.hline(cx - 4, cx + 3, 4, 'sky')
    else:
        c.px(cx - 6, h - 5, 'clay')
        c.px(cx - 5, h - 6, 'clay')
        c.px(cx + 5, h - 5, 'clay')
        c.px(cx - 1, 9, 'stone')
        c.px(cx, 10, 'slate')


def _stall(c, w, h, fixed, ph, item):
    for px in (2, w - 4):
        c.rect(px, 4, 2, h - 6, 'brew')
    c.rect(1, h - 10, w - 2, 7, 'brew')
    c.hline(1, w - 2, h - 10, 'tan')
    for x in range(4, w - 2, 6):
        c.vline(x, h - 9, h - 4, 'brew_dk')
    for wx in (6, w - 7):                                          # wheels
        c.circle(wx, h - 3, 2, 'brew_dk')
        c.px(wx, h - 3, 'tan')
    a, b = ('hot_pink', 'white') if fixed else ('sand', 'tan')
    for i, x in enumerate(range(0, w, 4)):
        c.rect(x, 2, 4, 6, a if i % 2 == 0 else b)
    c.hline(0, w - 1, 1, 'wine' if fixed else 'clay')
    for x in range(0, w, 4):                                       # the scalloped edge
        c.px(x + 1, 8, a if (x // 4) % 2 == 0 else b)
        c.px(x + 2, 8, a if (x // 4) % 2 == 0 else b)
    if fixed:
        for k, gx in enumerate((4, 13, 22)):
            ITEMS[item].draw(c, gx, h - 17 - (1 if (k + ph) % 3 == 0 else 0))
        c.px(w // 2, 10, 'gold')
    else:
        for x in (9, 18, 27):                                      # holes torn in the awning
            c.px(x, 6, KEY)
            c.px(x + 1, 7, KEY)
        c.rect(6, h - 15, 8, 5, 'brew_dk')                         # an empty crate
        c.hline(6, 13, h - 15, 'brew')
        c.rect(19, h - 12, 7, 2, 'tan')


def _bunting(c, w, h, fixed, ph, item):
    for px in (1, w - 3):
        c.rect(px, 3, 2, h - 3, 'brew')
        c.rect(px - 1, 1, 4, 2, 'gold' if fixed else 'stone')
        c.rect(px - 1, h - 2, 4, 2, 'brew_dk')
    sag = 4 if fixed else 8
    n = w - 4
    for i in range(n):
        u = i / max(1, n - 1)
        c.px(2 + i, int(round(_rope_y(u, sag))), 'brew_dk')
    if fixed:
        for k, x in enumerate(range(5, w - 5, 5)):
            u = (x - 2) / max(1, n - 1)
            y = int(round(_rope_y(u, sag))) + 1
            cc = RAINBOW[k % 6]
            c.rect(x - 1, y, 4, 2, cc)
            c.rect(x, y + 2, 2, 2, cc)
            c.px(x + ((k + ph) % 2), y + 4, cc)
    else:
        x = w // 2 + 3
        y = int(round(_rope_y((x - 2) / max(1, n - 1), sag))) + 1
        c.rect(x, y, 3, 2, 'stone')
        c.px(x + 1, y + 2, 'stone')
        c.px(x - 1, y + 4, 'slate')


def _fireworks(c, w, h, fixed, ph, item):
    c.rect(1, h - 6, w - 2, 5, 'brew')
    c.hline(1, w - 2, h - 6, 'tan')
    c.rect(0, h - 1, w, 1, 'brew_dk')
    c.rect(2, h - 12, 2, 6, 'brew_dk')
    c.rect(w - 4, h - 12, 2, 6, 'brew_dk')
    c.hline(2, w - 3, h - 12, 'brew')
    for k, x in enumerate((6, 11, 16)):
        c.rect(x, h - 11, 2, 5, 'charcoal')
        if fixed:
            c.rect(x, h - 15, 2, 4, ('red', 'sky', 'butter')[k])
            c.px(x, h - 16, 'white')
            c.px(x + 1, h - 16, ('red', 'sky', 'butter')[k])
    if not fixed:
        c.px(12, h - 12, 'slate')
        c.px(13, h - 13, 'slate')


def _board(c, w, h, fixed, ph, item):
    c.rect(3, 12, 2, h - 12, 'brew')
    c.rect(w - 5, 12, 2, h - 12, 'brew')
    c.rect(0, 2, w, 13, 'brew_dk')
    c.rect(1, 3, w - 2, 11, 'tan')
    c.hline(1, w - 2, 3, 'sand')
    c.rect(2, 0, w - 4, 2, 'clay')                                  # a little roof
    c.hline(3, w - 4, 0, 'brick')
    if fixed:
        c.rect(3, 5, 9, 7, 'paper')                                 # a trail map
        c.hline(4, 10, 7, 'leaf')
        c.hline(5, 9, 9, 'sky')
        c.px(8, 8, 'hot_pink')
        c.px(7, 6, 'ink')
        c.rect(14, 5, 7, 8, 'cream')                                # a poster with a star
        ui.star(c, 14, 6)
        c.px(7, 4, 'red')
        c.px(17, 4, 'red')
    else:
        c.px(5, 6, 'paper')
        c.px(6, 6, 'paper')
        c.px(5, 7, 'paper')
        c.px(16, 9, 'paper')
        c.px(17, 10, 'paper')
        c.px(9, 5, 'steel')
        c.px(18, 6, 'steel')


_DRAW = {'garden': _garden, 'lanterns': _lanterns, 'lamp': _lamp, 'fountain': _fountain, 'stall': _stall,
         'bunting': _bunting, 'fireworks': _fireworks, 'board': _board}


@lru_cache(maxsize=512)
def _sprite(kind, fixed, ph, item, tint):
    w, h = SIZE[kind]
    c = Canvas(w, h, color=KEY)
    _DRAW[kind](c, w, h, fixed, ph, item)
    a = c.a
    m = np.any(a != np.array(KEY, np.uint8), axis=2)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = a
    rgba[..., 3] = np.where(m, 255, 0)
    if tint is not None:
        tc, amount = tint
        rgb = rgba[:, :, :3].astype(np.float32)
        lit = rgb * (np.array(col(tc), np.float32) / 255.0)
        rgba[:, :, :3] = np.clip(rgb + (lit - rgb) * amount, 0, 255).astype(np.uint8)
    return rgba


LIGHTS = {'lamp': lambda w, h: [(3, 3, 6, 4)],                          # the lamp's crystal glass
          'lanterns': lambda w, h: [(3, 0, w - 6, h - 2)]}              # the paper lanterns on the rope


@lru_cache(maxsize=32)
def _lights(kind, ph, item):
    """The mended prop's own lights, untinted (RGBA, transparent elsewhere): pasted over the clock-lit
    prop at night so the lamp's glass and the lanterns shine instead of dimming with the scene."""
    w, h = SIZE[kind]
    rgba = _sprite(kind, True, ph, item, None).copy()
    keep = np.zeros((h, w), bool)
    for x, y, bw, bh in LIGHTS[kind](w, h):
        keep[y:y + bh, x:x + bw] = True
    if kind == 'lanterns':                                      # the lanterns, not the rope between them
        rgb = rgba[..., :3].astype(np.int16)
        keep &= (rgb[..., 0] > 150) & (rgb.max(axis=2) - rgb.min(axis=2) > 60)
    rgba[~keep, 3] = 0
    return rgba


def _paste(c, rgba, x, y):
    h, w = rgba.shape[:2]
    H_, W_ = c.a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[:, :, 3] > 127
    c.a[y0:y1, x0:x1][m] = s[:, :, :3][m]


def draw_prop(c, kind, x, ground, t, fixed, item='parcel', tint=None, night=False):
    """The prop centred on x, standing on `ground` (screen px)."""
    w, h = SIZE[kind]
    ph = (t // 8) % 2
    left, top = int(x) - w // 2, int(ground) - h
    _paste(c, _sprite(kind, bool(fixed), ph if fixed else 0, item, tint), left, top)
    if not fixed:
        return
    # night: 0..1 (or a bool), how far the festival's lights are on: each light has its own threshold
    if night >= 0.3 and tint is not None and kind in LIGHTS:  # its lights shine as painted, not dimmed
        _paste(c, _lights(kind, ph, item), left, top)
    if kind == 'lanterns':
        if night >= 0.3:                                        # (by day they are just paper lanterns)
            sag = 6
            for k, u in enumerate((0.17, 0.33, 0.5, 0.67, 0.83)):
                lx = left + int(2 + u * (w - 5))
                ly = top + int(round(_rope_y(u, sag))) + 3
                fx.glow(c, lx, ly, 4, ('orange', 'butter')[k % 2], 0.45 * min(1.0, night * 1.5), rings=2)
    elif kind == 'lamp':
        if night >= 0.3:                                        # a small pool of light at the glass only
            fx.glow(c, left + 6, top + 5, 6, 'crystal_lt', 0.35, rings=2)
        if (t // 5) % 9 == 0:
            c.px(left + 9, top + 1, 'white')
    elif kind == 'fountain':
        cx = left + w // 2
        for i in range(14):                                         # rainbow jets arcing into the basin
            side = 1 if i % 2 else -1
            p = ((t * 0.04) + hash01(i, 3)) % 1.0
            px = cx + side * int(round(p * 11))
            py = top + 4 - int(round(10 * math.sin(math.pi * p) * (0.7 + 0.3 * hash01(i, 5))))
            c.px(px, py, RAINBOW[i % 6])
        c.vline(cx, top, top + 3, 'white')
    elif kind == 'fireworks':
        a = (t + int(x)) % 96
        cols = (('hot_pink', 'butter', 'white'), ('sky', 'mint', 'white'), ('gold', 'orange', 'white'))[(t // 96) % 3]
        if a < 24:                                                  # a rocket climbs...
            ry = top + 2 - int(a * 2.4)
            c.px(left + 11, ry, 'white')
            c.px(left + 11, ry + 1, 'gold')
            c.px(left + 11, ry + 3, 'orange')
        ui.burst(c, a, 24, left + 11, top + 2 - 58, cols, n=26, seed=int(x) % 97 + (t // 96), power=0.9, life=40,
                 gravity=0.03)
        if 24 <= a < 30:
            fx.glow(c, left + 11, top + 2 - 58, 12, cols[0], 0.5, rings=3)
    elif kind in ('garden', 'stall', 'bunting', 'board') and (t // 7) % 11 == 0:
        c.px(left + int(hash01(t // 77, 1) * w), top + int(hash01(t // 77, 2) * h // 2), 'white')


def question(c, x, y, t, bob=None):
    """A bobbing '?' bubble over a trouble not yet noticed (screen px, tail at (x, y)). bob: the lift in
    px (default: a 1 px hop every 10 steps)."""
    if bob is None:
        bob = 1 if (t // 10) % 2 else 0
    bx, by = x - 5, y - 13 - bob
    fx.bubble(c, bx, by, 11, 10, tail=(x, y - 1), fill='white', border=ui.INK)
    for px, py in ((3, 2), (4, 1), (5, 1), (6, 1), (7, 2), (7, 3), (6, 4), (5, 5), (5, 7)):
        c.px(bx + px, by + py, 'hot_pink')


def question_icon(c, x, y):
    """9x9 '?' badge for toasts."""
    c.circle(x + 4, y + 4, 4, ui.INK)
    c.circle(x + 4, y + 4, 3, 'white')
    for px, py in ((3, 2), (4, 1), (5, 2), (5, 3), (4, 4), (4, 6)):
        c.px(x + px, y + py, 'hot_pink')


_HEART = ('.rr.rr.', 'rpprppr', 'rpwpppr', '.rpppr.', '..rpr..', '...r...')


def heart_icon(c, x, y):
    """9x9-ish heart for toasts: a place has been helped."""
    for j, row in enumerate(_HEART):
        for i, ch in enumerate(row):
            if ch != '.':
                c.px(x + 1 + i, y + 1 + j, {'r': 'crimson', 'p': 'hot_pink', 'w': 'white'}[ch])


def tiny_heart(c, x, y):
    for px, py in ((0, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
        c.px(x + px, y + py, 'hot_pink')


_TINY_Q = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (1, 4))


def tiny_question(c, x, y):
    """A 3x5 grey '?' with an ink outline (the overview map: a trouble still to find there)."""
    for px, py in _TINY_Q:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            c.px(x + px + dx, y + py + dy, ui.INK)
    for px, py in _TINY_Q:
        c.px(x + px, y + py, 'cloud')


def map_bunting(c, x, y, t, seed=0):
    """A helped place on the world map (x, y = its icon centre on screen): a string of pennants over the
    icon (lit by the map's clock: game/world.py dims it at night; map_festive adds its bulbs)."""
    y0 = y - 18
    for i in range(-14, 15):
        sag = int(round(3 * (1 - (i / 14) ** 2)))
        c.px(x + i, y0 + sag, 'brew_dk')
    for k, i in enumerate(range(-12, 13, 4)):
        sag = int(round(3 * (1 - (i / 14) ** 2)))
        cc = RAINBOW[(k + seed) % 6]
        c.rect(x + i - 1, y0 + sag + 1, 3, 2, cc)
        c.px(x + i + ((t // 10 + k) % 2) - 1 + 1 - 1, y0 + sag + 3, cc)


def map_festive(c, x, y, t, night, seed=0):
    """A helped place on the world map, over its bunting (map_bunting): a heart, twinkles, and at night
    warm bulbs between the pennants (the light itself, no halo) and a little firework now and then."""
    y0 = y - 18
    if night:
        for k, i in enumerate(range(-10, 11, 4)):
            if hash01(seed, k, 31) >= night:                        # (they come on one by one at dusk)
                continue
            sag = int(round(3 * (1 - (i / 14) ** 2)))
            on = (t // 16 + k + seed) % 7 != 0                     # now and then one blinks
            c.px(x + i, y0 + sag, 'butter' if on else 'gold')
            if on and (k + seed) % 2:
                c.px(x + i, y0 + sag + 1, 'gold')
    bob = int(round(math.sin(t * 0.1 + seed) * 1.5))                # a heart floats over a helped place
    hx, hy = x - 4, y0 - 9 + bob
    for j, row in enumerate(_HEART):
        for i, ch in enumerate(row):
            if ch != '.':
                c.px(hx + i, hy + j, {'r': 'crimson', 'p': 'hot_pink', 'w': 'white'}[ch])
    a = (t + seed * 37) % 70
    if a < 6:
        fx.sparkle(c, x + int(hash01(seed, t // 70) * 24) - 12, y - 4 - int(hash01(seed, t // 70, 2) * 10),
                   a, 'white', period=6, size=2)
    if night >= 0.8:
        b = (t + seed * 53) % 150
        if b < 34:
            cols = (('hot_pink', 'butter', 'white'), ('sky', 'mint', 'white'), ('gold', 'orange', 'white'))[seed % 3]
            if b < 10:
                c.px(x + 6, y0 - b * 2, 'butter')
            ui.burst(c, b, 10, x + 6, y0 - 20, cols, n=14, seed=seed + t // 150, power=0.55, life=24,
                     gravity=0.02, size=1)


@lru_cache(maxsize=None)
def heart_sprite():
    """The heart as a Sprite (counters)."""
    from engine import Sprite
    art = '\n'.join(_HEART)
    return Sprite.from_ascii(art, {'r': 'crimson', 'p': 'hot_pink', 'w': 'white'})


# ------------------------------------------------------------------ a helped place celebrates
# Pennant colours of each place's festival (its own palette: candy for Ponyville, crystal for the
# Crystal Empire, neon for Las Pegasus...). The bunting, flags and confetti of game/quests.py use them.
FESTIVE = {
    'canterlot': ('gold', 'royal', 'white', 'lavender'),
    'cloudsdale': ('sky', 'white', 'bubblegum', 'butter'),
    'ponyville': ('hot_pink', 'butter', 'sky', 'leaf'),
    'sweet_apple_acres': ('red', 'butter', 'leaf', 'cream'),
    'everfree': ('mint', 'violet', 'gold', 'teal'),
    'neighagra_falls': ('turquoise', 'white', 'orange', 'leaf'),
    'rainbow_falls': RAINBOW,
    'manehattan': ('crimson', 'cream', 'sky', 'gold'),
    'baltimare': ('coral', 'white', 'turquoise', 'butter'),
    'griffonstone': ('gold', 'brick', 'cream', 'sky'),
    'dragon_lands': ('red', 'gold', 'mint', 'orange'),
    'mount_aris': ('turquoise', 'gold', 'white', 'coral'),
    'klugetown': ('orange', 'turquoise', 'butter', 'magenta'),
    'changeling_kingdom': ('magenta', 'mint', 'lilac', 'butter'),
    'appleloosa': ('red', 'butter', 'sky', 'cream'),
    'las_pegasus': ('neon_pink', 'neon_cyan', 'neon_gold', 'lilac'),
    'vanhoover': ('red', 'white', 'teal', 'butter'),
    'crystal_empire': ('crystal_lt', 'crystal_pk', 'sky', 'white'),
    'yakyakistan': ('red', 'gold', 'sky', 'white'),
    'castle_two_sisters': ('lavender', 'gold', 'mint', 'royal_lt'),
}
ROPE = 'brew_dk'
# a pennant hanging under the rope: (dx, dy, tone) with tone 0 highlight, 1 base, 2 shade; 6 x 5 px
_PENNANT = ((-2, 1, 0), (-1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 2),
            (-1, 2, 0), (0, 2, 1), (1, 2, 1), (2, 2, 2),
            (-1, 3, 1), (0, 3, 1), (1, 3, 1), (2, 3, 2),
            (0, 4, 1), (1, 4, 2))
_TIP = 5                                                            # the tip row (it flutters)
_GAP = 8                                                            # rope px between pennants


def _lit(rgb, tint):
    """An RGB array lit like the scene (multiply blend by the tint, as the props and the balloon)."""
    rgb = np.asarray(rgb, np.float32)
    if tint is None:
        return rgb
    tc, amount = tint
    return rgb + (rgb * (np.array(col(tc), np.float32) / 255.0) - rgb) * amount


@lru_cache(maxsize=96)
def _tones(cols, tint):
    """(len(cols), 3 tones, rgb) uint8: highlight, base and shade of each pennant colour, lit by tint."""
    out = []
    for name in cols:
        base = np.array(col(name), np.float32)
        hi = base + (255 - base) * 0.35
        out.append([hi, base, base * 0.72])
    return np.clip(_lit(np.array(out, np.float32), tint), 0, 255).astype(np.uint8)


@lru_cache(maxsize=32)
def _rope_rgb(tint):
    return tuple(int(v) for v in np.clip(_lit(col(ROPE), tint), 0, 255))


def _put(a, xs, ys, rgb):
    """Set the pixels (xs, ys) inside the canvas array a to rgb (one colour, or one per pixel)."""
    m = (xs >= 0) & (xs < a.shape[1]) & (ys >= 0) & (ys < a.shape[0])
    if m.any():
        rgb = np.asarray(rgb, np.uint8)
        a[ys[m], xs[m]] = rgb[m] if rgb.ndim == 2 else rgb


_PLUS = np.array([(0, 0, 1.6), (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)], np.float32)
# a sky lantern's glow (offsets from its top-left, strength): round its 4 x 5 body
_LGLOW = np.array([(x, y, max(0.0, 1.25 - 0.32 * (abs(x - 1.5) + abs(y - 2.0))))
                   for x in range(-2, 6) for y in range(-1, 7)
                   if 1.25 - 0.32 * (abs(x - 1.5) + abs(y - 2.0)) > 0.1], np.float32)


def _halo(a, xs, ys, rgb, k, kernel=_PLUS):
    """A soft light (additive, k = strength, one or one per point) around each point: festival bulbs
    (a plus), sky lanterns. One pass for all points (where two lights overlap, one of them counts)."""
    if len(xs) == 0:
        return
    kx, ky, kf = kernel[:, 0].astype(int), kernel[:, 1].astype(int), kernel[:, 2]
    x = (np.asarray(xs)[:, None] + kx[None, :]).ravel()
    y = (np.asarray(ys)[:, None] + ky[None, :]).ravel()
    kk = np.broadcast_to(np.asarray(k, np.float32).reshape(-1, 1), (len(xs), 1))
    f = (kk * kf[None, :]).ravel()
    m = (x >= 0) & (x < a.shape[1]) & (y >= 0) & (y < a.shape[0])
    if m.any():
        add = np.array(col(rgb), np.float32)
        px = a[y[m], x[m]].astype(np.float32) + f[m][:, None] * add
        a[y[m], x[m]] = np.minimum(px, 255).astype(np.uint8)


@lru_cache(maxsize=512)
def _string_geom(dx, dy, sag8):
    """The shape of a pennant string from (0, 0) to (dx, dy) sagging sag8 / 8 px, relative to its start
    (built once per shape: the same string keeps every pixel as the camera pans): rope (xs, ys), pennant
    pixels (X, Y, tone, pennant number), tip anchors (px, py), night bulbs (bx, by)."""
    sag = sag8 / 8.0
    n = int(abs(dx) + abs(dy) + 4 * sag) + 2
    u = np.linspace(0.0, 1.0, n)
    rx = np.round(dx * u).astype(int)
    ry = np.round(dy * u + sag * 4 * u * (1 - u)).astype(int)
    m = int(max(abs(dx), 1) // _GAP)                   # one pennant every _GAP px (by x: mostly level)
    if m < 2:
        return rx, ry, None
    k = np.arange(1, m)
    uk = k / m
    px = np.round(dx * uk).astype(int)
    py = np.round(dy * uk + sag * 4 * uk * (1 - uk)).astype(int)
    ox = np.array([d[0] for d in _PENNANT]); oy = np.array([d[1] for d in _PENNANT])
    tn = np.array([d[2] for d in _PENNANT])
    X = (px[:, None] + ox[None, :]).ravel(); Y = (py[:, None] + oy[None, :]).ravel()
    T = np.broadcast_to(tn[None, :], (len(k), len(tn))).ravel()
    K = np.repeat(k, len(tn))
    bu = (np.arange(m) + 0.5) / m
    bx = np.round(dx * bu).astype(int)
    by = np.round(dy * bu + sag * 4 * bu * (1 - bu)).astype(int) + 1
    return rx, ry, (k, X, Y, T, K, px, py, bx, by)


def pennant_string(c, x0, y0, x1, y1, t, cols, tint=None, night=False, sag=None, seed=0, knots=True):
    """A string of pennants sagging from (x0, y0) to (x1, y1) (screen px), the pennants in turn in the
    colours cols, fluttering in the wind; a gold knot at each end. At night a warm bulb glows between
    the pennants (a gentle chase). Returns the rope's lowest screen y."""
    a = c.a
    dx, dy = x1 - x0, y1 - y0
    if sag is None:
        sag = 3 + abs(dx) * 0.07
    rx, ry, pen = _string_geom(dx, dy, int(round(sag * 8)))
    _put(a, x0 + rx, y0 + ry, _rope_rgb(tint))
    if pen is not None:
        k, X, Y, T, K, px, py, bx, by = pen
        tones = _tones(tuple(cols), tint)
        _put(a, x0 + X, y0 + Y, tones[(K + seed) % len(cols), T])
        wob = (k + t // 9 + seed) % 2                               # the tips flutter left / right
        _put(a, x0 + px + wob, y0 + py + _TIP, tones[(k + seed) % len(cols), 1])
        if night:                                                   # festival bulbs between pennants
            lit = ((np.arange(len(bx)) * 37 + seed * 11) % 100) / 100.0 < night     # (one by one at dusk)
            bx, by = bx[lit], by[lit]
            on = ((np.arange(len(bx)) + t // 20 + seed) % 5) != 0
            _halo(a, x0 + bx[on], y0 + by[on], 'orange', 0.3)
            _put(a, x0 + bx[on], y0 + by[on], col('butter'))
            _put(a, x0 + bx[~on], y0 + by[~on], col('brass'))
    if knots:
        for x, y in ((x0, y0), (x1, y1)):
            _put(a, np.array([x - 1, x + 1, x, x]), np.array([y, y, y - 1, y + 1]),
                 np.clip(_lit(col('brass'), tint), 0, 255).astype(np.uint8))
            _put(a, np.array([x]), np.array([y]), col('butter'))
            if night >= 0.6:                                        # a lit knot (the light, no halo)
                _put(a, np.array([x]), np.array([y]), col('white'))
    return int(y0 + ry.max())


def roof_flag(c, x, y, t, color, tint=None, night=False, seed=0):
    """A little pennant flag waving on a mast planted at (x, y) (a rooftop, screen px)."""
    a = c.a
    pole = tuple(int(v) for v in np.clip(_lit(col('steel_dk'), tint), 0, 255))
    _put(a, np.full(7, x), np.arange(y - 7, y), pole)
    _put(a, np.array([x]), np.array([y - 8]), col('butter'))
    tones = _tones((color,), tint)[0]
    ph = (t // 7 + seed) % 3                                        # the cloth ripples
    for i in range(5):
        h = 3 - (i * 3) // 5
        dy = (1 if (i + ph) % 3 == 0 else 0) if i else 0
        _put(a, np.full(h, x + 1 + i), np.arange(y - 7 + dy, y - 7 + dy + h),
             tones[0 if i < 2 else 1] if i < 4 else tones[2])
    if night > hash01(seed, x, 41) * 0.8:                          # a little lamp at the flag's top
        _put(a, np.array([x, x + 1]), np.array([y - 8, y - 8]), col('butter'))


_STAR = np.array(((0, -3), (-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0),
                  (1, 0)))
_STAR_RGB = np.array([col('butter' if k in (0, 2, 6) else 'gold') for k in range(len(_STAR))], np.uint8)


def _PEG_RGB(pole):
    return np.array([col('gold')] + [pole] * 4, np.uint8)


def _ribbons():
    """The maypole's four ribbons as flat arrays: ribbon j, pixel k, side, ribbon length, x offset."""
    J, K, S, N, O = [], [], [], [], []
    for j in range(4):
        n = 13 + 3 * (j % 2)
        side = -1 if j < 2 else 1
        for k in range(n):
            J.append(j); K.append(k); S.append(side); N.append(n)
            O.append((1 if side > 0 else 0) + side * (1 + (k * (2 + j % 2)) // 5))
    return tuple(np.array(v) for v in (J, K, S, N, O))


_RIBBONS = _ribbons()


def maypole(c, x, ground, top, t, cols, tint=None, night=False, spread=24, outer=None):
    """A festival mast rising from behind a mended prop (x = centre, top = the mast's top, screen px)
    with a gold star on it and pennant strings down to pegs on either side (outer: a second, wider pair
    from halfway up the mast)."""
    a = c.a
    pole = tuple(int(v) for v in np.clip(_lit(col('brew'), tint), 0, 255))
    hi = tuple(int(v) for v in np.clip(_lit(col('tan'), tint), 0, 255))
    _put(a, np.full(ground - top, x), np.arange(top, ground), pole)
    _put(a, np.full(ground - top, x + 1), np.arange(top, ground), hi)
    _put(a, x + _STAR[:, 0], top + _STAR[:, 1], _STAR_RGB)              # the star
    pairs = [(spread, top + 2, 2, 0)]
    if outer:
        pairs.append((outer, top + (ground - top) // 3, 4, 1))
    for sp, y0, sag, s in pairs:
        for side in (-1, 1):
            ex = x + side * sp
            _put(a, np.full(5, ex), np.arange(ground - 5, ground), _PEG_RGB(pole))   # a peg, gold-capped
            pennant_string(c, x + (side if side < 0 else 2), y0, ex, ground - 5, t, cols, tint, night, sag=sag,
                           seed=(3 if side > 0 else 0) + s, knots=False)
    tones = _tones(tuple(cols), tint)
    j, k, side, n, ox = _RIBBONS                                    # ribbons streaming from the star
    wave = np.round(np.sin(t * 0.15 + k * 0.55 + j * 1.7) * (k / n) * 2).astype(int)
    xs, ys = x + ox + wave, top + 1 + k
    ci = (j + 1) % len(cols)
    _put(a, np.concatenate([xs, xs + side]), np.concatenate([ys, ys + 1]),
         np.concatenate([tones[ci, 1], tones[ci, 2]]))
    if (t // 6) % 8 == 0:
        c.px(x + 2, top - 3, 'white')
    if night:
        fx.glow(c, x, top - 1, 10, 'butter', 0.5 * min(1.0, night * 1.5), rings=3)


@lru_cache(maxsize=32)
def _confetti_seeds(seed, n):
    h = np.array([[hash01(seed, k, j) for j in range(6)] for k in range(n)], np.float32)
    return h


def confetti_drift(c, t, ix, iy, cols, seed=0, n=20, tint=None):
    """About n petals and paper confetti drifting down through the place (world-anchored: they move with
    the main layer and wrap round the view). Petals are 2 x 2; confetti strips turn as they fall."""
    a = c.a
    h = _confetti_seeds(seed, n)
    PX, PY = W + 48, H + 36
    vx = 0.12 + (h[:, 2] - 0.5) * 0.3
    vy = 0.22 + h[:, 3] * 0.25
    sway = np.round(np.sin(t * 0.08 + h[:, 4] * 6.3) * 2).astype(int)
    xs = (np.floor(h[:, 0] * PX + vx * t) - ix) % PX - 24 + sway
    ys = (np.floor(h[:, 1] * PY + vy * t) - iy) % PY - 18
    xs, ys = xs.astype(int), ys.astype(int)
    tones = _tones(tuple(cols) + ('white',), tint)
    ci = np.arange(n) % (len(cols) + 1)
    spin = (t // 4 + np.arange(n)) % 4
    petal = h[:, 5] < 0.4
    base, hi_, sh = tones[ci, 1], tones[ci, 0], tones[ci, 2]
    _put(a, xs, ys, np.where(petal[:, None], hi_, base))
    m = petal | (spin == 0)                                          # petals and flat strips: 2 wide
    _put(a, xs[m] + 1, ys[m], np.where(petal[m][:, None], base[m], sh[m]))
    m = petal | (spin == 2)                                          # petals and upright strips: 2 tall
    _put(a, xs[m], ys[m] + 1, sh[m])
    m = petal
    _put(a, xs[m] + 1, ys[m] + 1, sh[m])


# a party balloon (5 x 6, tone per pixel: 0 highlight, 1 base, 2 shade) and its knot; a sky lantern (3 x 4)
_PARTY = ((1, 0, 1), (2, 0, 1), (3, 0, 1),
          (0, 1, 1), (1, 1, 0), (2, 1, 1), (3, 1, 1), (4, 1, 2),
          (0, 2, 1), (1, 2, 0), (2, 2, 1), (3, 2, 1), (4, 2, 2),
          (0, 3, 1), (1, 3, 1), (2, 3, 1), (3, 3, 2), (4, 3, 2),
          (1, 4, 2), (2, 4, 1), (3, 4, 2),
          (2, 5, 2))
_LANTERN = ((1, 0, 'brass'), (2, 0, 'brass'),
            (0, 1, 'orange'), (1, 1, 'butter'), (2, 1, 'butter'), (3, 1, 'orange'),
            (0, 2, 'orange'), (1, 2, 'cream'), (2, 2, 'cream'), (3, 2, 'orange'),
            (0, 3, 'brick'), (1, 3, 'orange'), (2, 3, 'orange'), (3, 3, 'brick'),
            (1, 4, 'gold'), (2, 4, 'butter'))


@lru_cache(maxsize=32)
def _float_seeds(seed, n):
    return np.array([[hash01(seed, k, j, 5) for j in range(6)] for k in range(n)], np.float32)


def balloons_rising(c, t, ix, iy, cols, seed=0, n=12, tint=None, night=False):
    """A slow release of party balloons on wavy strings (by day) or glowing sky lanterns (at night)
    rising through the place: world-anchored on the main layer, wrapping round the view like the
    confetti."""
    a = c.a
    h = _float_seeds(seed, n)
    PX, PY = W + 60, H + 60
    vy = 0.16 + h[:, 2] * 0.14
    sway = np.round(np.sin(t * 0.035 + h[:, 3] * 6.3) * (1.5 + h[:, 4])).astype(int)
    xs = ((np.floor(h[:, 0] * PX) - ix) % PX - 30 + sway).astype(int)
    ys = ((np.floor(h[:, 1] * PY - vy * t) - iy) % PY - 20).astype(int)
    lantern = h[:, 5] < float(night)                                # at dusk they turn into lanterns one by one
    if lantern.any():
        lx = np.array([d[0] for d in _LANTERN]); ly = np.array([d[1] for d in _LANTERN])
        lc = np.array([np.clip(_lit(col(d[2]), None), 0, 255) for d in _LANTERN], np.uint8)
        flick = (t // 5 + np.arange(n)) % 7 != 0                    # the flame inside breathes
        lxs, lys = xs[lantern], ys[lantern]
        _halo(a, lxs, lys, 'orange', np.where(flick, 0.2, 0.13)[lantern], kernel=_LGLOW)
        X = (lxs[:, None] + lx[None, :]).ravel(); Y = (lys[:, None] + ly[None, :]).ravel()
        _put(a, X, Y, np.tile(lc, (int(lantern.sum()), 1)))
    if lantern.all():
        return
    idx = np.nonzero(~lantern)[0]                                  # (each balloon keeps its own colour)
    xs, ys = xs[idx], ys[idx]
    tones = _tones(tuple(cols), tint)
    ci = (idx + seed) % len(cols)
    ox = np.array([d[0] for d in _PARTY]); oy = np.array([d[1] for d in _PARTY])
    tn = np.array([d[2] for d in _PARTY])
    X = (xs[:, None] + ox[None, :]).ravel(); Y = (ys[:, None] + oy[None, :]).ravel()
    _put(a, X, Y, tones[ci[:, None], tn[None, :]].reshape(-1, 3))
    _put(a, xs + 1, ys + 1, col('white'))                         # the shine
    string = tuple(int(v) for v in np.clip(_lit(col('cloud_dk'), tint), 0, 255))
    for k in range(5):                                              # the string waves under it
        wig = ((t // 8 + k + idx) % 4 == 0).astype(int) * (1 if k % 2 else -1)
        _put(a, xs + 2 + wig, ys + 6 + k, string)


def reveal_band(c, x, t):
    """The mend's reveal: a dithered gold light band at screen x sweeping right (trailing glow on its
    left, a bright leading edge with sparkles)."""
    a = c.a
    x0, x1 = max(0, x - 16), min(W, x + 2)
    if x1 <= x0:
        return
    from engine import bayer
    cols = np.arange(x0, x1)
    lvl = np.clip(1 - (x - cols) / 16.0, 0, 1) ** 1.5                # brightest at the leading edge
    thr = bayer(a.shape[0], x1 - x0, x0, 0)
    m = thr < lvl[None, :] * 0.9
    reg = a[:, x0:x1].astype(np.float32)
    reg[m] += np.array(col('gold'), np.float32) * 0.42
    a[:, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    if 0 <= x < W:
        reg = a[:, x].astype(np.float32) * 0.5 + np.array(col('butter'), np.float32) * 0.5
        a[:, x] = reg.astype(np.uint8)
    for k in range(6):
        sy = int(hash01(k, t, 9) * H)
        if 0 <= x + 1 < W:
            fx.sparkle(c, x + 1, sy, (t + k) % 6, 'white', period=6, size=2)
