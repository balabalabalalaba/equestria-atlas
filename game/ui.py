"""Small game-only UI pieces for the HUD layer (game/compose.py): key caps, the star counter, hints,
an edge arrow towards an off-screen goal and a name plate. All opaque (the HUD layer has no alpha).
"""
import math

from engine import W, H, Sprite, mix, text, text_width
from engine.atlas import draw_balloon

INK = 'ink'
PLATE = '#3a2440'          # the plum fill of the map's name plates
PLATE_HI = '#5a3a60'
RIM = 'gold'

_STAR = Sprite.from_ascii('''
    ...y...
    ..yYy..
    yyyYyyy
    .yYWYy.
    ..yYy..
    .yy.yy.
    .y...y.
''', {'y': 'gold', 'Y': 'butter', 'W': 'white'})

_ARROWS = {
    'up': Sprite.from_ascii('''
        ..o..
        .ooo.
        ooooo
    ''', {'o': INK}),
    'down': Sprite.from_ascii('''
        ooooo
        .ooo.
        ..o..
    ''', {'o': INK}),
    'left': Sprite.from_ascii('''
        ..o
        .oo
        ooo
        .oo
        ..o
    ''', {'o': INK}),
    'right': Sprite.from_ascii('''
        o..
        oo.
        ooo
        oo.
        o..
    ''', {'o': INK}),
}


_LIGHT_ARROWS = {k: v.recolor({INK: 'butter'}) for k, v in _ARROWS.items()}


def keycap(c, x, y, label, lit=False):
    """A little key cap (tiny caps or an arrow glyph name) at top-left (x, y); returns its width."""
    if label in _ARROWS:
        g = _ARROWS[label]
        w, h = g.w + 4, 9
    else:
        w, h = text_width(label, 'tiny') + 5, 9
    face = 'butter' if lit else 'cream'
    c.round_rect(x, y, w, h, INK, r=1)
    c.rect(x + 1, y + 1, w - 2, h - 3, face)
    c.hline(x + 1, x + w - 2, y + h - 2, mix(face, 'tan', .6))
    if label in _ARROWS:
        g = _ARROWS[label]
        g.draw(c, x + (w - g.w) // 2, y + 1 + (h - 3 - g.h) // 2)
    else:
        text(c, x + 3, y + 2, label, INK, font='tiny')
    return w


def hint_row(c, groups, y=None, x=None, right=None):
    """A row of key-cap groups on a dark strip, centred (or at x, or ending at `right`).
    groups: [([labels], icon_fn or None), ...]."""
    widths = []
    for keys, icon in groups:
        w = sum(keycap_width(k) + 1 for k in keys) - 1
        if icon is not None:
            w += 3 + icon(None, 0, 0)
        widths.append(w)
    total = sum(widths) + 8 * (len(groups) - 1) + 8
    y = H - 14 if y is None else y
    if x is None:
        x = (W - total) // 2 if right is None else right - total
    c.round_rect(x, y - 2, total, 13, INK, r=1)          # opaque: the HUD layer has no alpha
    gx = x + 3
    for (keys, icon), w in zip(groups, widths):
        cx = gx
        for k in keys:
            cx += keycap(c, cx, y, k) + 1
        if icon is not None:
            icon(c, cx + 2, y)
        gx += w + 8
    return total


def keycap_width(label):
    if label in _ARROWS:
        return _ARROWS[label].w + 4
    return text_width(label, 'tiny') + 5


def icon_balloon(c, x, y):
    """Tiny balloon glyph (6x9) for hints; returns its width when c is None."""
    if c is None:
        return 6
    draw_balloon(c, x + 3, y + 9, 0, 'xs', bob=False)
    c.px(x + 3, y + 1, 'hot_pink')
    return 6


def icon_land(c, x, y):
    """Balloon over a down arrow: 'land here'."""
    if c is None:
        return 11
    draw_balloon(c, x + 3, y + 7, 0, 'xs', bob=False)
    _LIGHT_ARROWS['down'].draw(c, x + 6, y + 4)
    c.hline(x, x + 10, y + 8, 'gold')
    return 11


def icon_takeoff(c, x, y):
    if c is None:
        return 11
    draw_balloon(c, x + 3, y + 8, 0, 'xs', bob=False)
    _LIGHT_ARROWS['up'].draw(c, x + 6, y + 1)
    return 11


def icon_fast(c, x, y):
    """Burner flame: boost."""
    if c is None:
        return 5
    for i, cc in enumerate(('butter', 'gold', 'orange', 'vermilion')):
        c.hline(x + 1 + (i == 0), x + 3 - (i == 0), y + 2 + i, cc)
    c.px(x + 2, y + 1, 'white')
    c.hline(x + 1, x + 3, y + 6, 'red')
    return 5


def icon_map(c, x, y):
    """Folded parchment map."""
    if c is None:
        return 9
    c.rect(x, y + 1, 9, 7, 'clay')
    c.rect(x + 1, y + 2, 7, 5, 'paper')
    c.vline(x + 3, y + 2, y + 6, 'tan')
    c.vline(x + 6, y + 2, y + 6, 'tan')
    c.px(x + 5, y + 4, 'hot_pink')
    return 9


def star(c, x, y):
    """The 7x7 gold star at top-left (x, y)."""
    _STAR.draw(c, x, y)


def star_counter(c, n, total, f=0, flash_t=None, x_right=W - 3, y=3):
    """Gold star + 'n/total' on a dark plate in the top-right corner. flash_t = frame of the last gain."""
    s = f'{n}/{total}'
    w = _STAR.w + 4 + text_width(s, 'tiny') + 5
    x = x_right - w
    hot = flash_t is not None and 0 <= f - flash_t < 24
    c.round_rect(x, y, w, 11, RIM if hot and (f - flash_t) // 3 % 2 == 0 else INK, r=1)
    c.rect(x + 1, y + 1, w - 2, 9, PLATE)
    c.hline(x + 2, x + w - 3, y + 1, PLATE_HI)
    dy = -1 if hot and f - flash_t < 8 else 0
    _STAR.draw(c, x + 2, y + 2 + dy)
    text(c, x + _STAR.w + 5, y + 3, s, 'butter' if hot else 'white', font='tiny')
    return x


def plate_width(label):
    return text_width(label, 'normal') + 12


def plate(c, cx, y, label, k=1.0):
    """Name plate in the map's arrival style (normal font, gold rim, plum fill) centred on cx.
    k < 1 = growing in (width scales)."""
    tw = text_width(label, 'normal')
    pw, ph = tw + 12, 13
    x = int(max(4, min(W - pw - 4, cx - pw // 2)))
    if k < 0.999:
        cw = max(4, int(pw * max(0.0, k)))
        c.round_rect(x + (pw - cw) // 2, y, cw, ph, RIM, r=2)
        c.rect(x + (pw - cw) // 2 + 1, y + 1, cw - 2, ph - 2, PLATE)
        return
    c.round_rect(x - 1, y - 1, pw + 2, ph + 2, '#6e2727', r=2)
    c.round_rect(x, y, pw, ph, RIM, r=2)
    c.rect(x + 1, y + 1, pw - 2, ph - 2, PLATE)
    c.rect(x + 2, y + 2, pw - 4, 1, PLATE_HI)
    text(c, x + 6, y + 3, label, '#fff6c0', font='normal', shadow='#6e2727')


# Edge arrows: a chunky 7x7 dart (with its ink outline 9x9) in 8 directions, kept clear of the HUD
# corners. Designed facing east and north-east; the other six are flips and a transpose of those two.
_DART_E = ('.......',
           'yy.....',
           '.yyy...',
           '.yyyyy.',
           '.yyy...',
           'yy.....',
           '.......')
_DART_NE = ('.......',
            '.yyyyy.',
            '..yyyy.',
            '....yy.',
            '....yy.',
            '.....y.',
            '.......')
# HUD boxes an edge arrow keeps out of (screen px, x0 y0 x1 y1): the clock, the counters and the errand
# plate under them, and the land prompt / key hints at the bottom centre (the widest prompt, Castle of the
# Two Sisters with its NEW tag, is 201 px)
ARROW_AVOID = ((0, 0, 60, 24), (W - 72, 0, W, 48), (W // 2 - 101, H - 29, W // 2 + 101, H))
_ARROW_M = 10                   # the arrows run along a ring this far inside the screen edge
_DARTS = {}


def _dart_masks():
    """Fill masks (7x7 bool) of the 8 darts, index k = direction k * 45 degrees (0 east, 2 south)."""
    if not _DARTS.get('masks'):
        import numpy as np
        e = np.array([[ch == 'y' for ch in r] for r in _DART_E])
        ne = np.array([[ch == 'y' for ch in r] for r in _DART_NE])
        s = e.T
        _DARTS['masks'] = [e, ne[::-1], s, ne[::-1, ::-1], e[:, ::-1], ne[:, ::-1], s[::-1], ne]
    return _DARTS['masks']


def _dart(k, color):
    """The 9x9 dart sprite for direction k in `color`: ink outline, a lit top edge."""
    key = (k, color)
    spr = _DARTS.get(key)
    if spr is None:
        import numpy as np
        from engine import col
        m = np.pad(_dart_masks()[k], 1)
        out = ~m & (np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1))
        hi = m & ~np.roll(m, 1, 0)                          # fill pixels with nothing above: lit
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[out, :3] = col(INK)
        rgba[m, :3] = col(color)
        rgba[hi, :3] = mix(color, 'white', 0.45)
        rgba[out | m, 3] = 255
        spr = _DARTS[key] = Sprite(rgba)
    return spr


def _arrow_ring():
    """Legal arrow centres: integer points on the ring _ARROW_M px inside the screen edge, outside every
    ARROW_AVOID box (grown by the dart's half size). Built once."""
    ring = _DARTS.get('ring')
    if ring is None:
        import numpy as np
        m = _ARROW_M
        xs, ys = np.arange(m, W - m + 1), np.arange(m, H - m + 1)
        pts = np.concatenate([np.stack([xs, np.full_like(xs, m)], 1), np.stack([xs, np.full_like(xs, H - m)], 1),
                              np.stack([np.full_like(ys, m), ys], 1), np.stack([np.full_like(ys, W - m), ys], 1)])
        ok = np.ones(len(pts), bool)
        for x0, y0, x1, y1 in ARROW_AVOID:
            ok &= ~((pts[:, 0] + 5 > x0) & (pts[:, 0] - 5 < x1) & (pts[:, 1] + 5 > y0) & (pts[:, 1] - 5 < y1))
        ring = _DARTS['ring'] = pts[ok].astype(float)
    return ring


def edge_arrow(c, tx, ty, f, color='hot_pink'):
    """Blinking dart at the screen edge pointing at an off-screen target (tx, ty) in screen px. It sits
    where the ray from the screen centre meets the edge ring, moved along the edge out of the HUD
    corners and the prompt band (ARROW_AVOID), and points from there at the target (8 directions)."""
    cx, cy = W / 2, H / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return
    m = _ARROW_M
    k = min((W / 2 - m) / abs(dx) if dx else 1e9, (H / 2 - m) / abs(dy) if dy else 1e9)
    ax, ay = cx + dx * k, cy + dy * k
    ring = _arrow_ring()
    i = int(((ring[:, 0] - ax) ** 2 + (ring[:, 1] - ay) ** 2).argmin())
    ax, ay = int(ring[i, 0]), int(ring[i, 1])
    d = int(round(math.atan2(ty - ay, tx - ax) / (math.pi / 4))) % 8
    bob = int(round(math.sin(f * 0.25) * 1.5))                 # a nudge along its heading
    ax += ((1, 1, 0, -1, -1, -1, 0, 1)[d]) * bob
    ay += ((0, 1, 1, 1, 0, -1, -1, -1)[d]) * bob
    _dart(d, color if (f // 8) % 2 == 0 else mix(color, 'white', .4)).draw(c, ax - 4, ay - 4)


# ------------------------------------------------------------------ interaction icons (9x9)
# Pixel icons for the "press SPACE" bubbles of hotspots (game/region.py Hotspot.icon). Regions may
# also pass their own Sprite. Palette letters: o ink, w white, y gold, Y butter, r red, p hot_pink,
# b baby_blue, B royal, g leaf, G green, s stone, S steel, c cloud, n tan, N brew, v violet, m magic
_ICON_PAL = {'o': INK, 'w': 'white', 'y': 'gold', 'Y': 'butter', 'r': 'red', 'p': 'hot_pink',
             'b': 'baby_blue', 'B': 'royal', 'g': 'leaf', 'G': 'green', 's': 'stone', 'S': 'steel',
             'c': 'cloud', 'n': 'tan', 'N': 'brew', 'v': 'violet', 'm': 'magic', 'k': 'grey'}
_ICON_ART = {
    'bell': '''
        ....o....
        ...oyo...
        ..oyYyo..
        ..oyYyo..
        ..oyyyo..
        .oyyyyyo.
        ooooooooo
        ....o....
        ...oyo...''',
    'whistle': '''
        ..c.c.c..
        ...ccc...
        ...oyo...
        ..oyYyo..
        ..oyYyo..
        ..oyyyo..
        ..ooooo..
        ...oyo...
        ..ooooo..''',
    'rain': '''
        ..ooooo..
        .occcwco.
        occcccwco
        okcccccko
        .ooooooo.
        .b..b..b.
        ..b..b..b
        .b..b..b.
        .........''',
    'spark': '''
        ....y....
        ....Y....
        .y..Y..y.
        ..YYwYY..
        yYYwwwYYy
        ..YYwYY..
        .y..Y..y.
        ....Y....
        ....y....''',
    'firework': '''
        .p..y..b.
        ..p.y.b..
        ...pyb...
        ypyyYyypy
        ...byp...
        ..b.y.p..
        .b..y..p.
        ....o....
        ....o....''',
    'steam': '''
        ..ccc....
        .cwwwc.c.
        ..ccc.cwc
        ...cc..c.
        ..ccwc...
        ...cc....
        ...oo....
        ..oNNo...
        ..oNNo...''',
    'lantern': '''
        ....o....
        ...ooo...
        ..oyYyo..
        ..oYwYo..
        ..oYwYo..
        ..oyYyo..
        ...ooo...
        ....o....
        .........''',
    'note': '''
        ....ooooo
        ....o...o
        ....o...o
        ....o...o
        ..ooo.ooo
        .oyyooyyo
        .oyyooyyo
        ..oo..oo.
        .........''',
    'bolt': '''
        ....oooo.
        ...oYYo..
        ..oYYo...
        .oYYYooo.
        .ooYYYYo.
        ...oYYo..
        ..oYo....
        .oYo.....
        .oo......''',
    'wind': '''
        .........
        oooooo...
        .....oo..
        ooooooo..
        .........
        oooooooo.
        .......oo
        oooo..oo.
        .........''',
    'splash': '''
        .b.....b.
        ..b.b.b..
        b..b.b..b
        .b.www.b.
        ..wwbww..
        .wbbbbbw.
        bbbBBBbbb
        .BBBBBBB.
        .........''',
    'gem': '''
        ..ooooo..
        .ommwmmo.
        omwwmmmmo
        .ommmmmo.
        ..ommmo..
        ...omo...
        ....o....
        .........
        .........''',
    'flag': '''
        .o.......
        .oppppo..
        .opwppppo
        .oppppo..
        .o.......
        .o.......
        .o.......
        ooo......
        .........''',
    'gear': '''
        ...o.o...
        .oosssoo.
        .osSSSso.
        osSoooSso
        .sSo.oSs.
        osSoooSso
        .osSSSso.
        .oosssoo.
        ...o.o...''',
    'horn': '''
        .......oo
        ......oyo
        ....ooyYo
        oooooyyYo
        oyyyyyyYo
        oooooyyyo
        ....ooyyo
        ......oyo
        .......oo''',
    'apple': '''
        ....g....
        ...gN....
        .oooNooo.
        orrrrwrro
        orrrrrwro
        orrrrrrro
        .orrrrro.
        ..ooooo..
        .........''',
    'heart': '''
        .........
        .oo...oo.
        oppo.oppo
        opwpopppo
        oppppppo.
        .oppppo..
        ..oppo...
        ...oo....
        .........''',
    'moon': '''
        ...oooo..
        ..oYYoo..
        .oYYo....
        .oYYo....
        .oYYo....
        .oYYYo...
        ..oYYYoo.
        ...oooo..
        .........''',
    # menu icons (title screen, pause menu, controls page)
    'speaker': '''
        .........
        ...oo..b.
        ..oco...b
        ooccoob.b
        occccob.b
        ooccoob.b
        ..oco...b
        ...oo..b.
        .........''',
    'window': '''
        ooooooooo
        oBBBBBwBo
        ooooooooo
        obbbbbbbo
        obcbbbbbo
        obbbbbbbo
        obbbbgbbo
        oggggGggo
        ooooooooo''',
    'fullscreen': '''
        ooo...ooo
        oYo...oYo
        ooYo.oYoo
        ...o.o...
        .........
        ...o.o...
        ooYo.oYoo
        oYo...oYo
        ooo...ooo''',
    'pad': '''
        .........
        .ooooooo.
        oSSSSSSSo
        oSoSSSgSo
        oooSSrSyo
        oSoSSSbSo
        oSSoooSSo
        oSo...oSo
        .o.....o.''',
    'door': '''
        .ooooooo.
        .oNNNNNo.
        .oNnnnNo.
        .oNnnnNo.
        .oNnnyNo.
        .oNnnnNo.
        .oNnnnNo.
        .oNNNNNo.
        ooooooooo''',
    'play': '''
        .........
        ..oo.....
        ..oYoo...
        ..oYYYoo.
        ..oYYYYYo
        ..oYYYoo.
        ..oYoo...
        ..oo.....
        .........''',
    'back': '''
        .........
        ...oo....
        ..oYo....
        .oYYooooo
        oYYYYYYYo
        .oYYooooo
        ..oYo....
        ...oo....
        .........''',
    'book': '''
        .........
        oooo.oooo
        orrrorrro
        orwrorwro
        orrrorrro
        orrrorrro
        orrrorrro
        ooooooooo
        ....o....''',
    'camera': '''
        .........
        ..ooo....
        oooYoooo.
        osssssswo
        osooosssO
        osowoSsso
        osooosSSo
        ooooooooo
        .........''',
    'pause': '''
        .........
        .ooo.ooo.
        .oYo.oYo.
        .oYo.oYo.
        .oYo.oYo.
        .oYo.oYo.
        .oYo.oYo.
        .ooo.ooo.
        .........''',
    'newmap': '''
        .......y.
        oooooo.Y.
        onnnnoYwY
        onpnno.Y.
        onnnno.y.
        onnpno...
        onnnno...
        oooooo...
        .........''',
}
_ICON_PAL['O'] = INK
ICONS = {k: Sprite.from_ascii(v, _ICON_PAL) for k, v in _ICON_ART.items()}


def icon(c, name, x, y):
    """Draw an interaction icon (name in ICONS, or a Sprite) with its top-left at (x, y)."""
    spr = name if isinstance(name, Sprite) else ICONS[name]
    spr.draw(c, x, y)
    return spr.w


def prompt_bubble(c, x, y, icon_name, f, ready=True):
    """World-anchored 'press SPACE' bubble whose tail points at (x, y): [icon][SPACE]. Dimmed while
    the hotspot is cooling down."""
    kw = keycap_width('SPACE')
    w, h = 9 + 3 + kw + 2, 11
    bob = 1 if ready and (f // 10) % 2 else 0
    bx, by = int(x - w // 2), int(y - h - 7 - bob)
    tail = (x, y - 1)
    if by < 3:                                  # no room above: hang the bubble below its target
        by, tail = int(y + 8 + bob), (x, y + 1)
    bx = max(2, min(W - w - 2, bx))
    fill = 'white' if ready else 'stone'
    from engine import fx
    fx.bubble(c, bx, by, w, h, tail=tail, fill=fill, border=INK)
    icon(c, icon_name, bx + 1, by + 1)
    keycap(c, bx + 12, by + 1, 'SPACE', lit=ready and (f // 10) % 2 == 0)


_STAMP = Sprite.from_ascii('''
    .w.w.w.w.
    wpppppppw
    .pnnnnnp.
    wpnnynnpw
    .pnyYynp.
    wpnnynnpw
    .pnnnnnp.
    wpppppppw
    .w.w.w.w.
''', {'w': 'white', 'p': 'hot_pink', 'n': 'cream', 'y': 'gold', 'Y': 'butter'})
STAMP_W, STAMP_H = _STAMP.w, _STAMP.h


def stamp(c, x, y, f=0, seed=0):
    """A hidden atlas stamp centred on (x, y), bobbing and glinting."""
    dy = int(round(math.sin(f * 0.12 + seed) * 1.5))
    _STAMP.draw(c, x - STAMP_W // 2, y - STAMP_H // 2 + dy)
    if (f + seed * 7) % 40 < 6:
        c.px(x + 3, y - 4 + dy, 'white')
        c.px(x + 4, y - 5 + dy, 'white')


def counter(c, sprite, n, total, f=0, flash_t=None, x_right=W - 3, y=3):
    """A counter plate (icon + 'n/total') in the top-right column; returns its left x."""
    s = f'{n}/{total}'
    w = sprite.w + 4 + text_width(s, 'tiny') + 5
    x = x_right - w
    hot = flash_t is not None and 0 <= f - flash_t < 24
    c.round_rect(x, y, w, 11, RIM if hot and (f - flash_t) // 3 % 2 == 0 else INK, r=1)
    c.rect(x + 1, y + 1, w - 2, 9, PLATE)
    c.hline(x + 2, x + w - 3, y + 1, PLATE_HI)
    dy = -1 if hot and f - flash_t < 8 else 0
    sprite.draw(c, x + 2, y + 1 + (9 - sprite.h) // 2 + dy)
    text(c, x + sprite.w + 5, y + 3, s, 'butter' if hot else 'white', font='tiny')
    return x


def stamp_counter(c, n, total, f=0, flash_t=None, y=16):
    return counter(c, _STAMP, n, total, f, flash_t, y=y)


def _outline_star():
    """The 7x7 star as a hollow outline (a place found but not landed in yet)."""
    spr = _DARTS.get('ostar')
    if spr is None:
        import numpy as np
        from engine import col
        m = _STAR.rgba[..., 3] > 0
        p = np.pad(m, 1)
        inner = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[m & ~inner, :3] = col('tan')
        rgba[m & ~inner, 3] = 255
        spr = _DARTS['ostar'] = Sprite(rgba)
    return spr


def place_counter(c, visited, found, total, f=0, flash_t=None, x_right=W - 3, y=3, found_t=None):
    """The map's progress plate (top right): a gold star with the places landed in, 'visited/total', and
    a small outline star with how many more have been found but not landed in yet (when any).
    flash_t = step of the last first visit (the plate flashes); found_t = step of the last discovery
    (the outline star hops). Returns its left x."""
    s = f'{visited}/{total}'
    more = found - visited
    sm = str(more) if more > 0 else ''
    w = _STAR.w + 4 + text_width(s, 'tiny') + 5
    if sm:
        w += 4 + _STAR.w + 2 + text_width(sm, 'tiny')
    x = x_right - w
    hot = flash_t is not None and 0 <= f - flash_t < 24
    c.round_rect(x, y, w, 11, RIM if hot and (f - flash_t) // 3 % 2 == 0 else INK, r=1)
    c.rect(x + 1, y + 1, w - 2, 9, PLATE)
    c.hline(x + 2, x + w - 3, y + 1, PLATE_HI)
    dy = -1 if hot and f - flash_t < 8 else 0
    _STAR.draw(c, x + 2, y + 2 + dy)
    if flash_t is not None:                                 # a first visit: sparks fly off the star
        burst(c, f, flash_t, x + 5, y + 5, ('gold', 'white', 'butter'), n=10, seed=7, power=0.7, life=20,
              gravity=0.03, size=1)
    tx = x + _STAR.w + 5
    text(c, tx, y + 3, s, 'butter' if hot else 'white', font='tiny')
    if sm:
        sx = tx + text_width(s, 'tiny') + 4
        c.vline(sx - 2, y + 3, y + 7, PLATE_HI)                 # a soft divider
        new = found_t is not None and 0 <= f - found_t < 24
        _outline_star().draw(c, sx, y + 2 - (1 if new and f - found_t < 8 else 0))
        text(c, sx + _STAR.w + 2, y + 3, sm, 'butter' if new and (f - found_t) // 3 % 2 == 0 else 'tan',
             font='tiny')
    return x


NEW_TAG_W = text_width('NEW', 'tiny') + 4


def new_tag(c, x, y, f=0):
    """A small gold 'NEW' tag (7 px tall) at top-left (x, y): a place not landed in yet. A glint runs
    across it now and then."""
    c.round_rect(x, y, NEW_TAG_W, 7, 'gold', r=1)
    c.hline(x + 1, x + NEW_TAG_W - 2, y, 'butter')
    text(c, x + 2, y + 1, 'NEW', PLATE, font='tiny')
    a = f % 50
    if a < NEW_TAG_W:
        c.px(x + a, y + 1 + a % 2, 'white')


def place_stats(c, x, y, found, total, stamps, stamp_total, chip=None):
    """One row (9 px tall) of what waits in a place: a star with landmarks found/total, a stamp with
    stamps/total ('?' and 3 while unknown), and a chip (a Sprite or a draw function fn(c, x, y), 9x9).
    c = None only measures. Returns the width."""
    a = f'{found}/{total if total else "?"}'
    b = f'{stamps}/{stamp_total if stamp_total else 3}'
    w = _STAR.w + 2 + text_width(a, 'tiny') + 5 + STAMP_W + 2 + text_width(b, 'tiny') + (5 + 9 if chip else 0)
    if c is None:
        return w
    _STAR.draw(c, x, y + 1)
    tx = x + _STAR.w + 2
    text(c, tx, y + 2, a, 'butter' if total and found >= total else 'white', font='tiny')
    sx = tx + text_width(a, 'tiny') + 5
    _STAMP.draw(c, sx, y)
    tx = sx + STAMP_W + 2
    text(c, tx, y + 2, b, 'butter' if stamp_total and stamps >= stamp_total else 'white', font='tiny')
    if chip is not None:
        cx = tx + text_width(b, 'tiny') + 5
        if callable(chip) and not isinstance(chip, Sprite):
            chip(c, cx, y)
        else:
            chip.draw(c, cx + (9 - chip.w) // 2, y + (9 - chip.h) // 2)
    return w


def icon_interact(c, x, y):
    """Hint glyph for SPACE in a region: the spark icon."""
    if c is None:
        return 9
    ICONS['spark'].draw(c, x, y)
    return 9


def burst(c, t, t0, x, y, colors=('gold', 'white', 'hot_pink'), n=24, seed=0, power=1.0, life=48,
          gravity=0.05, size=2):
    """A bold particle burst for interaction events (world layer): n chunky sparks fly out of (x, y)
    from step t0, arc under gravity and wink out over `life` steps. Deterministic (hash-based)."""
    import math as _m
    from engine import hash01
    a = t - t0
    if a < 0 or a >= life:
        return
    for i in range(n):
        ang = hash01(seed, i) * 6.283
        v = power * (0.8 + 1.6 * hash01(seed, i, 2))
        px = x + _m.cos(ang) * v * a
        py = y + _m.sin(ang) * v * a * 0.8 + gravity * a * a
        if a > life * 0.6 and (i + a) % 3 == 0:          # wink out towards the end
            continue
        cc = colors[i % len(colors)]
        s = size if a < life * 0.5 else 1
        c.rect(int(px), int(py), s, s, cc)


def clock(c, hour, x=3, y=3):
    """The world clock, top-left: a sun (day), a rising / setting sun (dawn, dusk) or a moon, and the
    time in 10-minute steps."""
    h = hour % 24
    hh, mm = int(h), int((h % 1) * 60) // 10 * 10
    s = f'{hh:02d}:{mm:02d}'
    w = 9 + 4 + text_width(s, 'tiny') + 5
    c.round_rect(x, y, w, 11, INK, r=1)
    c.rect(x + 1, y + 1, w - 2, 9, PLATE)
    c.hline(x + 2, x + w - 3, y + 1, PLATE_HI)
    ix, iy = x + 2, y + 1
    if 6.5 <= h < 18.5:                                     # the sun
        c.circle(ix + 4, iy + 4, 2, 'gold')
        for dx, dy in ((0, -4), (0, 4), (-4, 0), (4, 0), (-3, -3), (3, -3), (-3, 3), (3, 3)):
            c.px(ix + 4 + dx, iy + 4 + dy, 'butter')
        c.px(ix + 3, iy + 3, 'white')
    elif 5 <= h < 6.5 or 18.5 <= h < 20:                    # the sun on the horizon
        c.circle(ix + 4, iy + 6, 3, 'orange')
        c.rect(ix, iy + 7, 9, 2, PLATE)
        c.hline(ix, ix + 8, iy + 7, 'salmon')
    else:                                                   # the moon and a star
        c.circle(ix + 4, iy + 4, 3, 'cream')
        c.circle(ix + 6, iy + 3, 3, PLATE)
        c.px(ix + 8, iy + 7, 'white')
    text(c, x + 13, y + 3, s, 'white', font='tiny')
    return w
