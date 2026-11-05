"""Shared props, food, icons and the kiwi mascot for ONE GOOD DAY.

    from engine.props import *          # or import the names you need
    draw_lunchbox(c, 200, 132, 'open', items=3)
    draw_iced_glass(c, 150, 134, level=0.8, milk=0.4, f=f)
    draw_kiwi(c, 40, 150, 'walk', f, facing='right')
    draw_icon(c, 100, 30, 'atom')                     # 9x9, top-left anchored

Conventions (see docs/PROPS.md for the full catalogue):
  * (x, y) is the BOTTOM-CENTRE of the object unless documented otherwise: the
    object rests ON row y, i.e. its lowest pixel row is y-1. So a prop on the
    kitchen counter (top y=132) is drawn with y=132, on the floor with y=150.
    Odd/even widths: the left edge is x - w//2.
  * Icons (draw_icon) and bubbles use the TOP-LEFT corner.
  * Every animated prop takes the frame index `f`; output is a pure function of
    the arguments (deterministic, no state).
  * Light comes from the upper-left. Outlines are 'ink' (or a darker shade);
    glass is see-through (true alpha blend over whatever is behind it).
  * All sprites are built once and cached; a draw call is one small masked blit
    (typically 5-40 microseconds).
"""
import math
from functools import lru_cache

import numpy as np

from .core import (Canvas, col, mix, shade, clamp, lerp, seg, smooth, ease_out, ease_in,
                   ease_out_back, hash01, bayer, FRAMES_PER_BEAT)
from .sprite import Sprite
from .font import text as _text, text_width as _text_width

__all__ = [
    # containers & coffee
    'LUNCH_ITEMS', 'draw_lunchbox', 'draw_lunchbox_lid', 'lunchbox_slot', 'lunchbox_size',
    'draw_breakfast_box', 'draw_brew_jar', 'draw_jar_lid', 'draw_iced_glass', 'iced_glass_size',
    'draw_ice_cube', 'draw_milk', 'draw_coffee_bag', 'draw_scoop', 'draw_pour',
    # gadgets & bags
    'draw_phone', 'draw_backpack', 'draw_laptop', 'draw_sticky_note', 'draw_gym_bag',
    # kitchen
    'draw_pan', 'draw_pot', 'draw_rice_cooker', 'draw_toaster', 'draw_kettle', 'draw_plate',
    'draw_bowl', 'draw_mug', 'draw_fork', 'draw_knife', 'draw_cutting_board', 'draw_spatula',
    # home / gym / shop
    'draw_broom', 'draw_grocery_bag', 'draw_basket', 'draw_towel', 'draw_water_bottle',
    'draw_dumbbell', 'draw_kettlebell', 'draw_trolley', 'draw_book', 'draw_clothes', 'draw_receipt',
    'draw_shopping_list', 'draw_plant',
    # food & icons
    'FOOD', 'draw_food', 'food_size', 'ICONS', 'draw_icon', 'icon_bubble',
    # kiwi
    'KIWI_POSES', 'draw_kiwi', 'kiwi_size', 'draw_kiwi_silhouette', 'draw_kiwi_doodle',
    'draw_kiwi_plush', 'draw_kiwi_sticker', 'draw_kiwi_magnet',
    # fx helpers
    'float_text', 'draw_sweat', 'speed_lines', 'shake_lines', 'draw_growl', 'puff',
    'burst', 'heart_pop', 'zzz', 'draw_steam_heart',
]


# =====================================================================================
# infrastructure
# =====================================================================================
_ART = {}          # name -> (ascii, mapping)

# letters every sprite understands unless it overrides them
_BASE = {'o': 'ink', 'w': 'white', 'W': 'paper', 'k': 'night'}


def _reg(name, art, mapping=None):
    _ART[name] = (art, mapping or {})


def _parse(art, mapping):
    """ASCII -> RGBA uint8. mapping values: colour spec, or (spec, alpha 0..1)."""
    lines = art.strip('\n').split('\n')
    ind = min((len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()), default=0)
    lines = [ln[ind:].rstrip() for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    h, w = len(lines), max(len(ln) for ln in lines)
    out = np.zeros((h, w, 4), np.uint8)
    m = dict(_BASE)
    m.update(mapping)
    lut = {}
    for k, v in m.items():
        if v is None:
            continue
        if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], (int, float)):
            lut[k] = col(v[0]) + (int(round(255 * v[1])),)
        else:
            lut[k] = col(v) + (255,)
    for y, ln in enumerate(lines):
        for x, ch in enumerate(ln):
            if ch in '. ':
                continue
            if ch not in lut:
                raise KeyError(f'prop art: char {ch!r} at ({x},{y}) not mapped')
            out[y, x] = lut[ch]
    return out


@lru_cache(maxsize=None)
def _rgba(name, flip=False):
    if flip:
        return np.ascontiguousarray(_rgba(name)[:, ::-1])
    art, mapping = _ART[name]
    a = _parse(art, mapping)
    a.setflags(write=False)
    return a


def _blit(c, arr, x, y):
    """Blit an RGBA array with its top-left at (x, y). Alpha 255 = opaque,
    0 = transparent, in between = true alpha blend (glass)."""
    x, y = int(x), int(y)
    sh, sw = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(c.w, x + sw), min(c.h, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
    al = sub[:, :, 3]
    dst = c.a[y0:y1, x0:x1]
    full = al == 255
    dst[full] = sub[:, :, :3][full]
    part = (al > 0) & ~full
    if part.any():
        a = al[part].astype(np.float32)[:, None] / 255.0
        d = dst[part].astype(np.float32)
        dst[part] = (d + (sub[:, :, :3][part].astype(np.float32) - d) * a + 0.5).astype(np.uint8)


def _put(c, arr, x, y, anchor='bottom'):
    h, w = arr.shape[:2]
    if anchor == 'bottom':
        _blit(c, arr, x - w // 2, y - h)
    elif anchor == 'center':
        _blit(c, arr, x - w // 2, y - h // 2)
    else:
        _blit(c, arr, x, y)


def _draw(c, name, x, y, flip=False, anchor='bottom'):
    _put(c, _rgba(name, flip), x, y, anchor)


def _tint(arr, color):
    out = arr.copy()
    out[out[:, :, 3] > 0, :3] = col(color)
    return out


def _scale(arr, k):
    if k == 1:
        return arr
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _canvas_rgba(w, h):
    return np.zeros((h, w, 4), np.uint8)


def _px(a, x, y, color, alpha=255):
    if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        a[y, x, :3] = col(color)
        a[y, x, 3] = alpha


def _outline_rgba(arr, color='ink', diagonal=False):
    """Add a 1px outline around opaque pixels (array grows by 2)."""
    m = arr[:, :, 3] > 0
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = m
    g = pad.copy()
    g[1:, :] |= pad[:-1, :]
    g[:-1, :] |= pad[1:, :]
    g[:, 1:] |= pad[:, :-1]
    g[:, :-1] |= pad[:, 1:]
    if diagonal:
        g[1:, 1:] |= pad[:-1, :-1]
        g[1:, :-1] |= pad[:-1, 1:]
        g[:-1, 1:] |= pad[1:, :-1]
        g[:-1, :-1] |= pad[1:, 1:]
    out = np.zeros((h + 2, w + 2, 4), np.uint8)
    ring = g & ~pad
    out[ring, :3] = col(color)
    out[ring, 3] = 255
    out[1:-1, 1:-1][m] = arr[m]
    return out


# =====================================================================================
# FOOD — ~6-9 px icons, bottom-centre anchored
# =====================================================================================
_EGG_SH = (226, 208, 182)
_RICE_SH = (214, 212, 204)

_reg('food_egg', '''
    ..oo..
    .owWo.
    owWWWo
    oWWWWo
    oWWWso
    .oWsso
    ..ooo.
''', {'s': _EGG_SH})

_reg('food_fried_egg', '''
    ..oooo...
    .owwWWoo.
    owWWaaWWo
    oWWaYgaWo
    oWWaggaso
    .oWWaasso
    ..ooooooo
''', {'a': 'amber', 'g': 'gold', 'Y': 'butter', 's': _EGG_SH})

_reg('food_toast', '''
    .oooooo.
    oCCCCCCo
    oCTTttCo
    oCTtttco
    .oCttco.
    .oCtdco.
    .oCddco.
    .oooooo.
''', {'C': 'copper', 'c': 'rust', 'T': 'butter', 't': 'sand', 'd': 'clay'})

_reg('food_rice', '''
    ...oo...
    ..owWo..
    .owWWWo.
    .oWWWso.
    owWWWWso
    onNnnnno
    onnnnnmo
    .oooooo.
''', {'s': _RICE_SH, 'n': 'slate', 'N': 'sage', 'm': 'charcoal'})

_reg('food_broccoli', '''
    ..oo.oo..
    .oLLoLLo.
    oLGGLGGGo
    oLGGGGGFo
    oGGFGGFFo
    .oFoFFFo.
    ..oPPMo..
    ..oPPMo..
    ..ooooo..
''', {'L': 'leaf', 'G': 'green', 'F': 'forest', 'P': 'pale_leaf', 'M': 'moss'})

_reg('food_carrot', '''
    ......gg.
    ...o.gLg.
    ....oGGo.
    ...oaAAo.
    ..oaAAvo.
    .oaAAvo..
    .oAAvo...
    oAvvo....
    ooo......
''', {'g': 'green', 'L': 'leaf', 'G': 'forest', 'a': 'amber', 'A': 'orange', 'v': 'vermilion'})

_reg('food_tomato', '''
    ...g...
    .ooGoo.
    oHGgGRo
    oHRRRRo
    oRRRRro
    oRRRrro
    .ooooo.
''', {'g': 'green', 'G': 'leaf', 'H': 'peach', 'R': 'vermilion', 'r': 'brick'})

_reg('food_chicken', '''
    .oooo....
    oSSCCo...
    oSCCCCo..
    oCCCCCco.
    oCCCCcco.
    .ocCccco.
    ..oocco..
    ....oBoo.
    ....oBBBo
    .....oBo.
    ......o..
''', {'S': 'sand', 'C': 'clay', 'c': 'copper', 'B': 'paper'})

_reg('food_apple', '''
    ....b.l.
    ....bll.
    .ooobooo
    oHRRRRro
    oHRRRRro
    oRRRRrro
    oRRRrrro
    .oRrrro.
    ..oooo..
''', {'b': 'plum_brown', 'l': 'leaf', 'H': 'salmon', 'R': 'red', 'r': 'crimson'})

_reg('food_banana', '''
    .......bb
    .......oo
    ......oYo
    .....oYGo
    o...oYGao
    bo.oYGGao
    oYoYGGGao
    .oGGGGao.
    ..ooooo..
''', {'b': 'brew_mid', 'Y': 'butter', 'G': 'gold', 'a': 'amber'})

_reg('food_kiwifruit', '''
    ..ooooo..
    .oKKKKKko
    oKGLLLGko
    oKLsWsLko
    oKLWWWLko
    oKLsWsLko
    oKGLLLGko
    .okkkkko.
    ..ooooo..
''', {'K': 'latte', 'k': 'brew_mid', 'G': 'green', 'L': 'leaf', 'W': 'pale_leaf', 's': 'ink'})

_reg('food_milk', '''
    ..oo..
    .owWo.
    oWWWWo
    oBBBbo
    oWWWso
    oWlWso
    oWWWso
    oBBBbo
    .oooo.
''', {'B': 'sky', 'b': 'blue', 'l': 'light_sky', 's': 'mist'})

_reg('food_bread', '''
    ..oooooo..
    .oSbSSbSo.
    oSCbCCbCCo
    oCCCCCCCco
    occcccccco
    .oooooooo.
''', {'S': 'sand', 'b': 'butter', 'C': 'clay', 'c': 'copper'})

_reg('food_cheese', '''
    ......oo
    ....ooYo
    ..ooYYYo
    ooYYYYYo
    oGGaGGGo
    oGaGGaGo
    oGGGGGao
    oooooooo
''', {'Y': 'butter', 'G': 'gold', 'a': 'amber'})

_reg('food_avocado', '''
    ...ooo...
    ..oFFFo..
    .oFLLLFo.
    .oFLPLFo.
    oFLPPPLFo
    oFLhrrLFo
    oFLrrRLFo
    oFLPRPLFo
    .oFLLLFo.
    ..ooooo..
''', {'F': 'forest', 'L': 'leaf', 'P': 'pale_leaf', 'h': 'clay', 'r': 'copper', 'R': 'rust'})

_reg('food_onion', '''
    ...g...
    ...go..
    ..oVo..
    .oLVVo.
    oLVvVVo
    oLVvVvo
    oVVvVvo
    .oVvvo.
    ..ooo..
''', {'g': 'green', 'L': 'lilac', 'V': 'violet', 'v': 'purple'})

_reg('food_noodles', '''
    .......cc
    ......cc.
    ..YyYccy.
    oYyYycyYo
    oWBWBWBso
    .oWWWWso.
    ..ooooo..
''', {'c': 'copper', 'Y': 'butter', 'y': 'gold', 'B': 'sky', 's': 'mist'})

_reg('food_pepper', '''
    ...gg..
    ..oGo..
    .ooGoo.
    oYGGGYo
    oYGGGao
    oYGGGao
    oGGaGao
    .oaoao.
    ..o.o..
''', {'g': 'green', 'Y': 'butter', 'G': 'gold', 'a': 'amber'})

_reg('food_lettuce', '''
    ..oo.oo.
    .oPLoLLo
    oPLLPLGo
    oLPLLGGo
    oLLGLGGo
    oGLGGGFo
    .oGGGFo.
    ..ooooo.
''', {'P': 'pale_leaf', 'L': 'leaf', 'G': 'green', 'F': 'forest'})

_reg('food_coffee_bag', '''
    .oooooo.
    .ossssso
    .oKKKKKo
    oKkKKKKko
    oKpppppko
    oKpbbppko
    oKpppppko
    oKKKKKKko
    .ooooooo.
''', {'s': 'steel', 'K': 'crema', 'k': 'latte', 'p': 'paper', 'b': 'brew_mid'})

_reg('food_yoghurt', '''
    ..ooooo..
    .ossssso.
    .oWWWWWo.
    .oPPPPPo.
    .oPwPPpo.
    .oWWWWso.
    ..ooooo..
''', {'s': 'steel_hi', 'P': 'pink', 'p': 'rose', 'w': 'white', 'W': 'paper', 'S': 'mist'})

_reg('food_cereal', '''
    oooooooo
    oRRRRRRo
    oRYYYYRo
    oRYyyYRo
    oRRRRRro
    oRWWWWro
    oRWccWro
    oRRWWRro
    oRRRRRro
    oooooooo
''', {'R': 'red', 'r': 'brick', 'Y': 'gold', 'y': 'butter', 'W': 'paper', 'c': 'sand'})

_reg('food_juice', '''
    ..oo..
    .oGGo.
    .oGGo.
    oAAAAo
    oAOOao
    oAOoao
    oAOOao
    oAAAao
    .oooo.
''', {'G': 'green', 'A': 'orange', 'a': 'vermilion', 'O': 'amber'})

_reg('food_chocolate', '''
    oooooooo.
    oPPPPPPPo
    oPLPPPPpo
    oPPGGGGpo
    oBbBbBbBo
    oBBbBBbBo
    oooooooo.
''', {'P': 'violet', 'L': 'lilac', 'p': 'purple', 'G': 'gold', 'B': 'brew_mid', 'b': 'brew'})

_reg('food_strawberry', '''
    ..gGg..
    .oGgGo.
    oHRRRRo
    oRwRRRo
    oRRRwro
    .oRRro.
    ..oro..
    ...o...
''', {'g': 'green', 'G': 'leaf', 'H': 'salmon', 'R': 'red', 'r': 'crimson', 'w': 'butter'})

_reg('food_grapes', '''
    ....g..
    ..oogo.
    .oLvLvo
    oLvpvpo
    ovpLvpo
    .oLvpo.
    .ovpo..
    ..oo...
''', {'g': 'moss', 'L': 'lilac', 'v': 'violet', 'p': 'purple'})

_reg('food_mushroom', '''
    ..oooo..
    .oHHRRo.
    oHRRRRro
    oRRrrrro
    .ooPPoo.
    ..oPPo..
    ..oPso..
    ..oooo..
''', {'H': 'blush', 'R': 'tan', 'r': 'mauve', 'P': 'paper', 's': 'mist'})

_reg('food_salmon', '''
    .ooooooo.
    oPCwCCwCo
    oCCCwCCwo
    occcccwco
    osssssss.
    .ooooooo.
''', {'P': 'peach', 'C': 'coral', 'c': 'clay', 'w': 'blush', 's': 'stone'})

FOOD = ['egg', 'fried_egg', 'toast', 'rice', 'broccoli', 'carrot', 'tomato', 'chicken', 'apple',
        'banana', 'kiwifruit', 'milk', 'bread', 'cheese', 'avocado', 'onion', 'noodles', 'pepper',
        'lettuce', 'coffee_bag', 'yoghurt', 'cereal', 'juice', 'chocolate', 'strawberry',
        'grapes', 'mushroom', 'salmon']


def draw_food(c, x, y, name, flip=False):
    """Food icon (~6-9 px) with bottom-centre at (x, y). Names: FOOD."""
    _draw(c, 'food_' + name, x, y, flip)


def food_size(name):
    a = _rgba('food_' + name)
    return a.shape[1], a.shape[0]


# =====================================================================================
# LUNCH BOX (mint-teal bento with a coral band) & BREAKFAST BOX (glass tub, gold lid)
# =====================================================================================
LUNCH_ITEMS = ['rice', 'broccoli', 'chicken', 'egg', 'carrot', 'tomato']

_LB = {'I': 'ice', 'T': 'turquoise', 'A': 'aqua', 'a': 'teal', 'd': 'deep_teal',
       'P': 'peach', 'p': 'salmon', 'c': 'coral', 'r': 'vermilion', 'g': 'amber', 'G': 'gold',
       'Y': 'butter', 'h': 'white', 't': (12, 116, 122)}

# closed: the lid overhangs the body by 1 px on each side (18 x 10)
_reg('lunchbox_closed', '''
    ..oooooooooooooo..
    .oIwIIIIIPpIIIIIo.
    oIITTTTTTpcTTTTTTo
    oAAAAAAgYGgAAAAAAo
    oooooooooooooooooo
    .oTTTTTTTcrTTTTAo.
    .oTTTTTTTcrTTTTAo.
    .oTTTTTTTcrTTTTAo.
    .oaAAAAAAcrAAAAao.
    ..oooooooooooooo..
''', _LB)

# the lid alone (18 x 5)
_reg('lunchbox_lid', '''
    ..oooooooooooooo..
    .oIwIIIIIPpIIIIIo.
    oIITTTTTTpcTTTTTTo
    oAAAAAAgYGgAAAAAAo
    oooooooooooooooooo
''', _LB)

# open: lid leaning behind (rows 0-2), back rim, 3x2 compartments (4x3 each), front wall
_LB_OPEN_BOX = '''
    .oIIIIIIIIIIIIIIo.
    .oddddtddddtddddo.
    .oaaaaAaaaaAaaaao.
    .oaaaaAaaaaAaaaao.
    .oAAAAIAAAAIAAAAo.
    .oddddtddddtddddo.
    .oaaaaAaaaaAaaaao.
    .oaaaaAaaaaAaaaao.
    .oIIIIIIIIIIIIIIo.
    .oTTTTTTTcrTTTTAo.
    .oTTTTTTTcrTTTTAo.
    .oaAAAAAAcrAAAAao.
    ..oooooooooooooo..
'''
_reg('lunchbox_open', '''
    ..oooooooooooooo..
    .otttttttcrttttto.
    oAAAAAAAAcrAAAAAAo
''' + _LB_OPEN_BOX, _LB)
_reg('lunchbox_open_nolid', _LB_OPEN_BOX, _LB)

# 4x3 food tiles that fill a compartment
_TILES = {
    'rice': ('''
        wWwW
        WwWs
        sWsW
    ''', {'s': _RICE_SH}),
    'broccoli': ('''
        LGLG
        GFGF
        FPFP
    ''', {'L': 'leaf', 'G': 'green', 'F': 'forest', 'P': 'pale_leaf'}),
    'chicken': ('''
        CSCS
        cCcC
        rcrc
    ''', {'S': 'clay', 'C': 'copper', 'c': 'rust', 'r': 'wine'}),
    'egg': ('''
        WWWW
        WgYW
        sggs
    ''', {'g': 'gold', 'Y': 'butter', 's': _EGG_SH}),
    'carrot': ('''
        gAgA
        AaAa
        aAaA
    ''', {'g': 'sand', 'A': 'amber', 'a': 'orange'}),
    'tomato': ('''
        HgHg
        RRRR
        rRrR
    ''', {'H': 'salmon', 'g': 'leaf', 'R': 'red', 'r': 'crimson'}),
}
for _k, (_a, _m) in _TILES.items():
    _reg('tile_' + _k, _a, _m)

_LB_W = 18
# compartment top-left corners relative to the box rows (without the lid)
_LB_CELLS = [(2, 1), (7, 1), (12, 1), (2, 5), (7, 5), (12, 5)]


def lunchbox_size(state='closed', lid='behind'):
    """(w, h) of the lunch box sprite for a state."""
    if state == 'closed':
        return 18, 10
    return (18, 16) if lid == 'behind' else (18, 13)


def lunchbox_slot(i, lid='behind'):
    """(dx, dy) of the centre of compartment i (0..5, LUNCH_ITEMS order) relative to
    the anchor (x, y) of draw_lunchbox(c, x, y, 'open'). Use it as the landing target
    for food hopping in (e.g. draw_food(c, x+dx, y+dy+2, name) sits in the slot); the
    lid setting does not change the result (the box body is the same)."""
    cx, cy = _LB_CELLS[i]
    return (cx + 2 - _LB_W // 2, cy + 1 - 13)


def draw_lunchbox(c, x, y, state='closed', items=6, f=0, lid='behind'):
    """The hero's mint-teal bento box with a coral band and a gold clip.

    state 'closed' (18x10, the lid overhangs the 16 px body) or 'open' (18x16 with the
    lid leaning behind, or 18x13 with lid='none'). items 0..6 fills compartments in
    LUNCH_ITEMS order. A glint sweeps over the closed lid every 96 frames (f).
    """
    if state == 'closed':
        _draw(c, 'lunchbox_closed', x, y)
        t = f % 96
        if t < 9:
            gx = x - 7 + t * 2
            c.px(gx, y - 9, 'white')
            if t < 8:
                c.px(gx - 1, y - 8, 'white')
        return
    name = 'lunchbox_open' if lid == 'behind' else 'lunchbox_open_nolid'
    arr = _rgba(name)
    x0 = x - _LB_W // 2
    _blit(c, arr, x0, y - arr.shape[0])
    by = y - 13
    for i in range(min(6, max(0, int(items)))):
        cx, cy = _LB_CELLS[i]
        _blit(c, _rgba('tile_' + LUNCH_ITEMS[i]), x0 + cx, by + cy)


def draw_lunchbox_lid(c, x, y):
    """The lid alone (18x5), bottom-centre — for animating it shut: on a closed box
    it sits at draw_lunchbox_lid(c, x, y-5)."""
    _draw(c, 'lunchbox_lid', x, y)


_BB = {'Y': 'butter', 'G': 'gold', 'A': 'amber', 'r': 'orange',
       's': 'stone', 'S': 'grey', 'q': ('glass', 0.35), 'Q': ('glass_hi', 0.8),
       'T': 'sand', 't': 'clay', 'c': 'copper', 'e': 'paper', 'y': 'gold', 'u': 'butter',
       'E': _EGG_SH, 'b': ('glass', 0.6)}

_BB_LID = '''
    ..oooooooooooo..
    .oYwYYYYYYYYYYo.
    oAGGGGGGGGGGGGAo
    oAooooooooooooAo
'''
_reg('bbox_closed_full', _BB_LID + '''
    .sQqqqqqqqqqqqS.
    .sQqcTTcqeeeqqS.
    .sQcTTtcEeyueqS.
    .sQcTttceEyyeES.
    .sbctttcEEeeEES.
    ..sSSSSSSSSSSS..
''', _BB)

_reg('bbox_closed_empty', _BB_LID + '''
    .sQqqqqqqqqqqqS.
    .sQqqqqqqqqqqqS.
    .sQqqqqqqqqqqqS.
    .sQqqqqqqqqqqqS.
    .sbbbbbbbbbbbbS.
    ..sSSSSSSSSSSS..
''', _BB)

_reg('bbox_open_full', '''
    .sQssssssssssS.
    .sQqcTTcqeeeeS.
    .sQcTTtcEeyueeS
    .sQcTttceEyyeES
    .sQcTttcEEeeEES
    .sbctttcEEeeEES
    ..sSSSSSSSSSSS.
''', _BB)

_reg('bbox_open_empty', '''
    .sQssssssssssS.
    .sQqqqqqqqqqqS.
    .sQqqqqqqqqqqqS
    .sQqqqqqqqqqqqS
    .sQqqqqqqqqqqqS
    .sbbbbbbbbbbbbS
    ..sSSSSSSSSSSS.
''', _BB)

_reg('bbox_lid', _BB_LID, _BB)


def draw_breakfast_box(c, x, y, state='closed', full=True):
    """Glass tub with a sunny gold clip-lid; toast + fried egg inside (full=True).
    state 'closed' (16x10), 'open' (15x7, no lid) or 'lid' (the lid alone, 16x4).
    The lid sits on the open tub at y-6."""
    if state == 'lid':
        _draw(c, 'bbox_lid', x, y)
        return
    _draw(c, f"bbox_{'closed' if state == 'closed' else 'open'}_{'full' if full else 'empty'}", x, y)


# =====================================================================================
# COLD BREW: mason jar, iced glass, ice cube, milk, coffee bag, scoop, pour stream
# =====================================================================================
# jar template (11 x 17): g = interior (brew or see-through glass), h = glass highlight
# streak, p/Y/G/R/a = paper label with a little sun, lid on top.
_JAR_ART = '''
    ..ooooooo..
    .oHwHHHHHo.
    .oMmMmMmMo.
    .ommmmmmmo.
    ..sgggggS..
    .shggggggS.
    shggggggggS
    shggggggggS
    shggggggggS
    sgpppappppS
    sgppRGRpppS
    sgpaGYGappS
    sgppRGRpppS
    sgpppappppS
    shggggggggS
    sbbbbbbbbbS
    .sSSSSSSSS.
'''
_JAR_MAP = {'H': 'steel_hi', 'M': 'steel', 'm': 'steel_dk', 's': 'stone', 'S': 'grey',
            'p': 'paper', 'Y': 'butter', 'G': 'gold', 'R': 'amber', 'a': 'orange',
            'b': ('glass', 0.6)}
_JAR_LINES = [ln.strip() for ln in _JAR_ART.strip('\n').split('\n')]
_JAR_TOP, _JAR_BOT = 4, 14          # interior rows
_JAR_ROWS = _JAR_BOT - _JAR_TOP + 1  # 11 rows of liquid at level 1


@lru_cache(maxsize=1024)
def _jar_rgba(level_rows, steeping, phase, label, lid):
    lines = _JAR_LINES
    w = len(lines[0])
    art = '\n'.join(lines).replace('g', '.').replace('h', '.')
    if not label:
        for ch in 'pYGRa':
            art = art.replace(ch, '.')
    a = _parse(art, _JAR_MAP).copy()
    interior = np.array([[ch in 'ghpYGRa' for ch in ln] for ln in lines])
    hi = np.array([[ch == 'h' for ch in ln] for ln in lines])
    labelm = (np.array([[ch in 'pYGRa' for ch in ln] for ln in lines]) if label
              else np.zeros_like(hi))
    top_row = _JAR_BOT + 1 - level_rows
    glass = col('glass')
    if steeping:
        body, light, dark, surf = col('brew_mid'), col('latte'), col('brew'), col('crema')
    else:
        body, light, dark, surf = col('brew'), col('brew_mid'), col('brew_dk'), col('latte')
    for yy in range(_JAR_TOP, _JAR_BOT + 1):
        row = lines[yy]
        xs = [i for i, ch in enumerate(row) if interior[yy, i]]
        lo, hi_x = min(xs), max(xs)
        for xx in xs:
            if labelm[yy, xx]:
                continue
            if yy >= top_row:
                cc = light if xx <= lo + 1 else dark if xx >= hi_x else body
                if yy == top_row:
                    cc = surf if xx < hi_x else light
                a[yy, xx, :3] = cc
                a[yy, xx, 3] = 255
            else:
                a[yy, xx, :3] = glass
                a[yy, xx, 3] = 90
            if hi[yy, xx]:
                a[yy, xx, :3] = col('glass_hi')
                a[yy, xx, 3] = 255 if yy >= top_row else 210
    if steeping and level_rows > 2:
        span = level_rows - 1
        for i in range(10):          # suspended grounds drifting slowly
            gx = 2 + int(hash01(i, 7) * 7)
            gy = _JAR_BOT - int((hash01(i, 3) * span + phase * (0.12 + 0.1 * hash01(i, 5))) % span)
            if interior[gy, gx] and not labelm[gy, gx] and gy > top_row and not hi[gy, gx]:
                a[gy, gx, :3] = col('brew_dk') if i % 3 else col('dark_olive')
        for xx in range(1, 10):      # bloom of grounds floating on top
            if interior[top_row, xx] and hash01(xx, phase // 8) > 0.4:
                a[top_row, xx, :3] = col('brew')
        for i in range(2):           # tiny bubbles rising
            by = _JAR_BOT - int((phase * 0.6 + i * span / 2) % max(1, span - 1))
            bx = 3 + i * 4 + (phase // 5 + i) % 2
            if by > top_row and interior[by, bx] and not labelm[by, bx]:
                a[by, bx, :3] = col('glass_hi')
    if not lid:
        a[0:4] = 0
        for xx in range(2, 9):       # rim of the open mouth
            if xx in (2, 8):
                a[4, xx] = col('stone') + (255,)
            elif a[4, xx, 3] < 255:
                a[4, xx] = col('glass_hi') + (170,)
    a.setflags(write=False)
    return a


def draw_brew_jar(c, x, y, level=1.0, steeping=False, f=0, label=True, lid=True):
    """The hero's big mason jar of cold brew (11x17): see-through glass, steel lid, paper
    label with a little sun. level 0..1 (fill height), steeping=True: lighter brew with
    suspended grounds, a floating bloom and tiny rising bubbles (animated by f).
    lid=False shows the open mouth (scooping / pouring); the lid alone: draw_jar_lid."""
    rows = int(round(clamp(level) * _JAR_ROWS))
    phase = (f // 3) % 240 if steeping else 0
    _put(c, _jar_rgba(rows, bool(steeping), phase, bool(label), bool(lid)), x, y)


@lru_cache(maxsize=None)
def _jar_lid():
    a = _parse('\n'.join(_JAR_LINES[:4]), _JAR_MAP)
    return np.ascontiguousarray(a[:, 1:10])


def draw_jar_lid(c, x, y):
    """The jar's steel lid alone (9x4), bottom-centre. On a jar drawn at (x, y) it sits
    at draw_jar_lid(c, x, y-13)."""
    _put(c, _jar_lid(), x, y)


# ---- iced glass (procedural so 'small' and 'big' share one look)
_GLASS_DIMS = {'small': (7, 10), 'big': (11, 16)}


def iced_glass_size(size='small'):
    """(w, h) of the glass body (the straw adds 4 px above the rim)."""
    return _GLASS_DIMS[size]


@lru_cache(maxsize=4096)
def _iced_rgba(size, level_q, milk_q, phase, ice, straw, drip):
    gw, gh = _GLASS_DIMS[size]
    big = size == 'big'
    top = 4 if straw else 0
    a = np.zeros((gh + top, gw, 4), np.uint8)

    def P(x, y, cc, al=255):
        y += top
        if 0 <= x < gw and 0 <= y < a.shape[0]:
            a[y, x, :3] = col(cc)
            a[y, x, 3] = al

    def get(x, y):
        return a[y + top, x]

    ib = gh - 3                      # last interior row (thick base below it)
    x0, x1 = 1, gw - 2               # interior columns
    for yy in range(0, ib + 1):      # see-through interior
        for xx in range(x0, x1 + 1):
            P(xx, yy, 'glass', 70)
    max_rows = ib - 1                # one row of headroom under the rim
    rows = int(round(level_q / 20 * max_rows))
    surf = ib - rows + 1
    milk = milk_q / 20
    thr = bayer(gh + 4, gw)
    for yy in range(surf, ib + 1):
        for xx in range(x0, x1 + 1):
            u = (xx - x0) / max(1, x1 - x0)
            v = (yy - surf) / max(1, rows - 1)
            if milk <= 0:
                cc = 'brew'
            elif milk >= 0.97:
                cc = 'latte' if v < 0.75 or xx < x1 else 'brew_mid'
            else:
                # milk sinks from the top in a slow swirl, clouding more as milk grows
                sw = 0.5 + 0.5 * math.sin(5.0 * u - 4.0 * v + phase * 0.3)
                m = milk * 1.5 + 0.55 * sw - 0.6 * v - 0.25
                if m > 0.75:
                    cc = 'crema' if sw > 0.9 and v < 0.5 else 'latte'
                elif m > 0.4 + 0.25 * (thr[yy, xx] - 0.5):
                    cc = 'brew_mid'
                else:
                    cc = 'brew'
            if xx == x1:
                cc = {'brew': 'brew_dk', 'brew_mid': 'brew', 'latte': 'brew_mid',
                      'crema': 'latte'}[cc]
            P(xx, yy, cc)
        if yy == surf and rows > 0:   # the surface catches the light
            for xx in range(x0, x1):
                P(xx, yy, 'crema' if milk > 0.5 else 'latte' if milk > 0.1 else 'brew_mid')
    # vertical highlight streak on the left (short; looks like a reflection)
    hl = range(max(1, surf + 1), min(ib, surf + (6 if big else 3)))
    for yy in hl:
        P(1, yy, 'glass_hi', 235)
    if big:
        for yy in range(2, 6):
            P(x1, yy, 'glass_hi', 120)
    # ice cubes: floating at the surface (bobbing), or on the bottom when nearly empty
    cs = 3 if big else 2
    if big:
        spots = [(x0, -2, 0), (x0 + 4, -1, 3), (x0 + 2, 2, 6)]
    else:
        spots = [(x0, -1, 0), (x0 + 2, 0, 4), (x0 + 1, 2, 7)]
    for i in range(min(int(ice), 3)):
        sx, depth, ph = spots[i]
        bob = 1 if ((phase + ph) % 12) < 6 else 0
        if rows >= cs + 1:
            sy = surf + depth + bob
        else:
            sx = x0 + i * (cs - (0 if big else 1))
            sy = ib - cs + 1 - (1 if i == 2 else 0)
        sy = max(1, sy)
        for dy in range(cs):
            for dx in range(cs):
                xx, yy = sx + dx, sy + dy
                if not (x0 <= xx <= x1) or yy > ib:
                    continue
                if dx == 0 and dy == 0:
                    cc = 'white'
                elif dx == cs - 1 and dy == cs - 1:
                    cc = 'stone'
                elif dx == cs - 1 or dy == cs - 1:
                    cc = 'glass'
                else:
                    cc = 'glass_hi'
                if yy > surf:
                    cc = mix(cc, 'latte' if milk > 0.6 else 'brew_mid', 0.5)
                P(xx, yy, cc)
    # straw: from the bottom (right of centre) up past the rim
    if straw:
        sx0, sy0 = (x1 - 1, ib)
        sx1, sy1 = (x1 + (1 if big else 1), -top)
        if big:
            sx0, sx1 = x1 - 2, x1
        n = sy0 - sy1
        for k in range(n + 1):
            yy = sy0 - k
            xx = sx0 + int(round((sx1 - sx0) * (k / n) ** 2))
            if yy < 0:
                cc = 'hot_pink'
            elif yy >= surf:
                cc = mix('hot_pink', 'brew' if milk < 0.6 else 'latte', 0.55)
            else:
                cc = mix('hot_pink', 'glass', 0.2)
            P(xx, yy, cc)
            if big:
                P(xx + 1, yy, shade(cc, 0.72))
        P(sx1, sy1, 'pink')
    # walls, thick base, bottom edge, rim glint
    for yy in range(0, gh - 1):
        P(0, yy, 'stone')
        P(gw - 1, yy, 'grey')
    for xx in range(1, gw - 1):
        P(xx, ib + 1, 'glass_hi', 160)
        P(xx, gh - 1, 'grey')
    P(1, gh - 1, 'stone')
    P(0, 0, 'glass_hi')
    # condensation: beads glint on and off, one drip slides down the front
    if rows > 2:
        beads = [(x1 - 1, surf + 3), (3, ib - 1), (x1 - 3, surf + 6), (2, surf + 8)]
        for i, (bx, by) in enumerate(beads[:4 if big else 1]):
            if surf < by <= ib and (drip + i * 13) % 30 < 20:
                P(bx, by, 'white')
        dyy = surf + 2 + (drip // 2) % max(2, ib - surf - 1)
        if dyy <= ib:
            P(x1 - (2 if big else 2), dyy, 'glass_hi')
    a.setflags(write=False)
    return a


def draw_iced_glass(c, x, y, level=1.0, milk=0.0, f=0, straw=True, ice=3, size='small'):
    """Tall iced glass of cold brew, see-through. size 'small' (7x10, in hand / on a
    table) or 'big' (11x16, close-ups). level 0..1, milk 0..1 (0 = black cold brew,
    ~0.3 = a swirl sinking in, 1 = fully clouded iced latte), ice = cubes 0..3 (they
    bob with f), condensation beads glint and a drip slides down. The straw adds
    4 px above the rim (the sprite is then 14 / 20 px tall)."""
    lq = int(round(clamp(level) * 20))
    mq = int(round(clamp(milk) * 20))
    phase = (f // 3) % 60
    _put(c, _iced_rgba(size, lq, mq, phase, int(ice), bool(straw), (f // 4) % 60), x, y)

_reg('ice_cube', '''
    .ooo.
    oWIIo
    oIIGo
    oIGGo
    .ooo.
''', {'o': 'stone', 'W': 'white', 'I': 'glass_hi', 'G': 'glass'})

_reg('ice_cube_big', '''
    .oooo.
    oWWIIo
    oWIIIo
    oIIIGo
    oIIGGo
    .oooo.
''', {'o': 'stone', 'W': 'white', 'I': 'glass_hi', 'G': 'glass'})


def draw_ice_cube(c, x, y, big=False):
    """A single ice cube (5x5, big: 6x6), bottom-centre — for cubes dropping into a glass."""
    _draw(c, 'ice_cube_big' if big else 'ice_cube', x, y)


_reg('milk', '''
    ..ooo..
    ..oBo..
    .owWWo.
    owWWWso
    oWBBBbo
    oWBlBbo
    oWBBBbo
    oWWWWso
    oWWWWso
    .ooooo.
''', {'B': 'sky', 'b': 'blue', 'l': 'white', 's': 'mist'})


def draw_milk(c, x, y):
    """Bottle of milk (7x10) — NZ 'blue top' standard milk."""
    _draw(c, 'milk', x, y)


_reg('coffee_bag', '''
    .oooooooo.
    .oSSSSSSo.
    .oKHKKKko.
    oKHKKKKKko
    oKHppppKko
    oKpbbbppko
    oKpbBblpko
    oKppbpppko
    oKHppppKko
    oKKKKKKKko
    okkkkkkkko
    .oooooooo.
''', {'S': 'steel_hi', 'K': 'crema', 'H': (226, 190, 146), 'k': 'latte', 'p': 'paper',
      'b': 'latte', 'B': 'brew_mid', 'l': 'tan'})


def draw_coffee_bag(c, x, y):
    """Kraft coffee bag (10x12) with a folded tin-tie top and a tiny kiwi on the label."""
    _draw(c, 'coffee_bag', x, y)


_reg('scoop', '''
    oooo.....
    oBBoooooo
    obBoSSSSo
    .oo.oooo.
''', {'B': 'brew', 'b': 'brew_mid', 'S': 'steel_hi'})

_reg('scoop_empty', '''
    oooo.....
    oSsoooooo
    osSoSSSSo
    .oo.oooo.
''', {'S': 'steel_hi', 's': 'steel'})


def draw_scoop(c, x, y, full=True, flip=False):
    """Coffee scoop (9x4) — bowl on the left, handle to the right (flip for the other way).
    full=True shows ground coffee in the bowl."""
    _draw(c, 'scoop' if full else 'scoop_empty', x, y, flip)


def draw_pour(c, x, y0, y1, f, color='brew', width=1, splash=True):
    """A pouring liquid stream from (x, y0) down to the surface at y1 with a wobble
    and (optional) splash droplets at the bottom. color: 'brew', 'milk' (white),
    'water', or any colour."""
    cc = {'milk': 'paper', 'water': 'light_sky', 'brew': 'brew'}.get(color, color)
    hi = {'brew': 'brew_mid', 'paper': 'white', 'light_sky': 'white'}.get(cc, 'white')
    for yy in range(int(y0), int(y1)):
        wob = 1 if (yy + f // 2) % 7 == 0 and yy > y0 + 2 else 0
        for k in range(width):
            c.px(x + k + wob, yy, cc)
        if (yy + f) % 5 == 0:
            c.px(x + wob, yy, hi)
    if splash:
        for i in range(3):
            t = ((f + i * 4) % 10) / 10
            dx = int((i - 1) * 3 * t) + (1 if i == 1 else 0)
            dy = int(-4 * t * (1 - t) * 3)
            c.px(x + dx, int(y1) - 1 + dy, hi if i == 1 else cc)


# =====================================================================================
# THE KIWI — procedural, cached per (pose, step, scale, facing). Facing right in model
# space; origin = ground point under the body centre (y up is negative).
# =====================================================================================
KIWI_POSES = ['idle', 'walk', 'run', 'peck', 'look_up', 'call', 'hop', 'squat', 'sleep']
_KIWI_PERIOD = {'idle': 72, 'walk': 18, 'run': 12, 'peck': 36, 'look_up': 72, 'call': 72,
                'hop': 18, 'squat': 36, 'sleep': 72}
KIWI_SPEED = {'walk': 0.5, 'run': 1.5, 'hop': 0.6}   # suggested px / frame at scale 1

KIWI_COLORS = {'out': 'ink', 'deep': 'brew', 'shade': 'brew_mid', 'mid': (128, 84, 60), 'base': 'latte',
               'hi': 'crema', 'beak': 'blush', 'beak_sh': 'tan', 'leg': 'tan',
               'leg_far': 'mauve', 'eye': 'ink', 'glint': 'white'}


def kiwi_period(pose):
    """Loop length in frames of a kiwi pose (walk/hop = one beat, idle = one bar)."""
    return _KIWI_PERIOD[pose]


def _kiwi_params(pose, k):
    """Geometry for step k of a pose, in scale-1 units (floats)."""
    P = dict(bx=0.0, by=-5.6, rx=4.7, ry=3.6, hx=3.0, hy=-7.0, hr=2.4, ba=32.0, bl=5.2,
             eye='open', legs=[(0.9, 1.6, 0.0, 0), (-0.9, -0.6, 0.0, 1)], lift=0.0,
             sound=0, dust=0, beak_open=False, air=0.0)
    # legs: (hip x, foot x, foot lift, far?)
    if pose == 'idle':
        if 12 <= k < 22:
            P['by'] -= 0.25
            P['ry'] += 0.3
            P['breathe'] = 1
        if k in (9, 21):
            P['eye'] = 'closed'
        dip = {14: 8, 15: 18, 16: 18, 17: 8}.get(k, 0)
        P['ba'] += dip
        P['hy'] += 0.6 if dip > 10 else 0
    elif pose == 'walk':
        xs = [1.8, 1.0, -0.6, -1.6, -0.6, 1.0]
        lf = [0, 0, 0, 0, 1, 1]
        j = (k + 3) % 6
        P['legs'] = [(0.9, xs[k], lf[k], 0), (-0.9, xs[j] - 0.4, lf[j], 1)]
        bob = [0, -1, 0, 0, -1, 0][k]
        P['lift'] = -bob
        P['ba'] += [0, -4, 0, 4, -4, 0][k]
        P['hx'] += [0.0, 0.3, 0.3, 0.0, 0.3, 0.3][k]
    elif pose == 'run':
        P['bx'] += 0.4
        P['hx'] += 1.2
        P['hy'] += 1.0
        P['ba'] = 18.0
        P['rx'] += 0.3
        P['ry'] -= 0.2
        xs = [3.0, 0.6, -2.6, 0.6]
        lf = [0, 1.5, 0, 1.5]
        j = (k + 2) % 4
        P['legs'] = [(1.0, xs[k], lf[k], 0), (-0.6, xs[j] - 0.4, lf[j], 1)]
        P['lift'] = [0, 1, 0, 1][k]
        P['dust'] = 1 if k in (0, 2) else 0
        P['shear'] = 1
    elif pose == 'peck':
        down = [0, 0, 1, 2, 3, 3, 2, 3, 3, 2, 1, 0][k]
        ang = [32, 36, 48, 60, 74, 74, 62, 74, 74, 60, 44, 34][k]
        P['hx'] += 0.4 * down
        P['hy'] += 0.9 * down
        P['bx'] += 0.15 * down
        P['ba'] = ang
        P['ry'] -= 0.1 * down
        P['shear'] = 1 if down >= 2 else 0
        if down >= 3:
            P['dust'] = 2 if k in (5, 8) else 0
    elif pose in ('look_up', 'call'):
        P['hx'] -= 0.6
        P['hy'] -= 0.9
        P['bx'] -= 0.2
        P['ba'] = -38.0 if pose == 'look_up' else -46.0
        P['shear'] = -1
        if pose == 'look_up':
            P['ba'] += [0, 0, -3, -3, 0, 0][(k // 4) % 6]
            if k in (8, 19):
                P['eye'] = 'closed'
        else:
            calling = (k % 12) < 6
            if calling:
                P['ry'] += 0.3
                P['by'] -= 0.3
                P['beak_open'] = True
                P['breathe'] = 1
                P['sound'] = 1 + (k % 6) // 2
            if k in (9, 22):
                P['eye'] = 'closed'
    elif pose == 'hop':
        # f % 18 == 0 is the landing (squash) — land your hops on the beat
        P['air'] = [0.0, 0.0, 2.6, 4.0, 3.0, 1.0][k]
        if k == 0:
            P['rx'] += 0.6
            P['ry'] -= 0.7
            P['by'] += 0.7
            P['hy'] += 0.8
            P['legs'] = [(0.9, 1.6, 0.0, 0), (-0.9, -0.8, 0.0, 1)]
            P['sq'] = -1
        elif k in (2, 3):
            P['sq'] = 1
            P['rx'] -= 0.3
            P['ry'] += 0.4
            P['by'] -= 0.2
            P['legs'] = [(0.9, 0.4, 0.0, 0), (-0.9, -1.4, 0.0, 1)]
            P['ba'] -= 6
        elif k == 5:
            P['legs'] = [(0.9, 1.4, 0.0, 0), (-0.9, -0.6, 0.0, 1)]
        P['dust'] = 3 if k == 0 else 0
    elif pose == 'squat':
        d = [0, 0, 0.6, 1.2, 1.6, 1.6, 1.2, 0.6, 0, 0, 0, 0][k]
        P['lift'] = -d
        P['rx'] += 0.25 * d
        P['sq'] = -1 if d >= 1.2 else 0
        P['ba'] += 6 * d
        if k in (4, 5):
            P['eye'] = 'closed'
    elif pose == 'sleep':
        P['by'] = -3.9
        P['rx'] = 5.0
        P['ry'] = 3.6
        P['hx'] = 2.9
        P['hy'] = -4.9
        P['hr'] = 2.3
        P['ba'] = 74.0
        P['bl'] = 3.8
        P['eye'] = 'closed'
        P['legs'] = []
        P['sleep'] = True
        if k >= 12:
            P['breathe'] = 1
            P['ry'] += 0.35
            P['by'] -= 0.3
    return P


def _kiwi_build(pose, k, s, mode):
    """Rasterise one kiwi frame facing right -> (rgba, ox, oy, eye_xy)."""
    P = _kiwi_params(pose, k)
    KC = {n: np.array(col(v), np.uint8) for n, v in KIWI_COLORS.items()}
    Ww, Hh = 20 * s + 12, 18 * s + 12
    ox, oy = Ww // 2 - 1, Hh - 2
    a = np.zeros((Hh, Ww, 4), np.uint8)
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    X = (xx + 0.5 - ox) / s
    Y = (yy + 0.5 - oy) / s
    lift, air = P['lift'], P.get('air', 0.0)
    by, hy = P['by'] - lift - air, P['hy'] - lift - air
    bx, hx = P['bx'], P['hx']
    rx, ry, hr = P['rx'], P['ry'], P['hr']
    eb = ((X - bx) / rx) ** 2 + ((Y - by) / ry) ** 2
    eh = ((X - hx) / hr) ** 2 + ((Y - hy) / hr) ** 2
    body = eb <= 1.0
    head = eh <= 1.0
    m = body | head
    # --- shading (light from the upper left), quantised to a 4-step ramp
    inb = eb <= eh * (hr / rx) ** 0  # pick the nearer surface for normals
    nx = np.where(body & ~head, (X - bx) / rx, np.where(head, (X - hx) / hr, (X - bx) / rx))
    ny = np.where(body & ~head, (Y - by) / ry, np.where(head, (Y - hy) / hr, (Y - by) / ry))
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lum = nx * -0.55 + ny * -0.62 + nz * 0.56 - 0.12 * np.clip(ny, 0, 1)
    ramp = np.where(lum > 0.60, 3, np.where(lum > 0.16, 2, np.where(lum > -0.38, 1, 0)))
    ramp = np.where(head & (ramp < 2) & (Y < hy + hr * 0.5), ramp + 1, ramp)   # lit face
    if mode == 'color':
        # hair-like feather strands: short '/' dashes (down-and-back), anchored to the body
        u = xx - int(math.floor(ox + bx * s))
        v = yy - int(math.floor(oy + by * s))
        if s == 1:
            strand = (((u * 7 + v * 13) % 11) == 0) & (ramp == 2) & body & ~head
        else:
            cell = (((u + v) // 3) * 31 + (v // 2) * 17) % 7
            strand = (((u + v) % 3) == 0) & (cell < 3) & (ramp >= 1) & body & ~head
            strand &= ~(((u + v) % 3) == 0) | (((v % 4) < 2))
        ramp = np.where(strand & (ramp >= 1), ramp - 1, ramp)
    cols = np.stack([KC['deep'], KC['shade'], KC['base'], KC['hi']])
    rgb = cols[ramp]
    a[m, :3] = rgb[m]
    a[m, 3] = 255
    # --- outline (4-connected ring) around body+head
    pad = np.zeros((Hh + 2, Ww + 2), bool)
    pad[1:-1, 1:-1] = m
    g = pad[1:-1, 1:-1] | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    ring = g & ~m
    a[ring, :3] = KC['out']
    a[ring, 3] = 255

    def P_(px, py, key, over=True):
        if 0 <= px < Ww and 0 <= py < Hh and (over or a[py, px, 3] == 0):
            a[py, px, :3] = KC[key] if isinstance(key, str) else key
            a[py, px, 3] = 255

    def to_px(mx, my):
        return int(math.floor(ox + mx * s)), int(math.floor(oy + my * s))

    # --- legs (far leg first), stout, feet with toes forward + a back toe
    lw = 1 if s == 1 else 2 if s == 2 else 2
    for (hipx, footx, fl, far) in sorted(P['legs'], key=lambda t: -t[3]):
        key = 'leg_far' if far else 'leg'
        hx0, hy0 = to_px(bx + hipx, by + ry * 0.80)
        fx0, fy0 = to_px(footx, -0.5 - fl - air)
        fy0 = min(fy0, oy - 1)
        n = max(abs(fx0 - hx0), abs(fy0 - hy0), 1)
        for i in range(n + 1):
            px_ = int(round(hx0 + (fx0 - hx0) * i / n))
            py_ = int(round(hy0 + (fy0 - hy0) * i / n))
            for t in range(lw):
                if not (0 <= py_ < Hh and 0 <= px_ + t < Ww) or not m[py_, px_ + t]:
                    P_(px_ + t, py_, key, over=not far)
        for t in range(1, 2 + s):
            P_(fx0 + t, fy0, key)
        P_(fx0 - 1, fy0, key)
        if s >= 2:
            P_(fx0 - 1, fy0, 'shade')
    # --- beak: tapering, slightly decurved; pale with a darker underside
    ang = math.radians(P['ba'])
    sx = hx + hr * 0.80 * math.cos(ang * 0.6)
    sy = hy + hr * 0.50 * math.sin(ang * 0.6) + 0.45
    L_ = P['bl']
    steps = int(L_ * s * 2) + 2
    curve = math.radians(14) * (1 if P['ba'] >= 0 else -0.4)
    tip = None
    for i in range(steps + 1):
        t = i / steps
        a2 = ang + curve * t
        px_, py_ = to_px(sx + math.cos(a2) * L_ * t, sy + math.sin(a2) * L_ * t)
        if py_ > oy - 1:
            continue
        tip = (px_, py_)
        thick = s if t < 0.5 else max(1, s - 1)
        if s == 1:
            P_(px_, py_, 'beak' if t < 0.6 else 'beak_sh')
            if t < 0.25:
                P_(px_, py_ + 1, 'beak_sh')
        else:
            for d in range(thick):
                P_(px_, py_ + d, 'beak' if d < thick - 1 else 'beak_sh')
    if P['beak_open'] and s >= 2 and tip:
        mx_, my_ = to_px(sx + math.cos(ang) * L_ * 0.75, sy + math.sin(ang) * L_ * 0.75)
        P_(mx_ + 1, my_ + s, 'beak_sh')
        P_(mx_ + 2, my_ + s, 'beak_sh')
    # --- eye: a dark dot (2x2 + glint from scale 2)
    ex, ey = to_px(hx + hr * 0.42, hy - hr * 0.05)
    if P['eye'] == 'open':
        es = 1 if s == 1 else 2
        for dy in range(es):
            for dx in range(es):
                P_(ex + dx, ey + dy, 'eye')
        if s >= 2:
            P_(ex, ey, 'glint')
    else:
        for dx in range(2 if s == 1 else s + 1):
            P_(ex + dx - (0 if s == 1 else 1), ey + (0 if s == 1 else 1), 'deep')
    # --- extras: dust puffs, calling sound arcs
    gy = oy - 1
    dust = P['dust']
    pts = []
    if dust == 1:      # run kick-up behind
        gx, _ = to_px(-rx - 0.6, 0)
        pts = [(gx, gy), (gx - 1, gy - 1), (gx - 2, gy)]
    elif dust == 2:    # peck: soil bits at the beak tip
        if tip:
            pts = [(tip[0] - 1, gy - 1), (tip[0] + 1, gy - 2), (tip[0] + 2, gy)]
    elif dust == 3:    # hop landing
        l0, _ = to_px(-rx - 0.8, 0)
        r0, _ = to_px(rx + 0.8, 0)
        pts = [(l0, gy), (l0 - 1, gy - 1), (r0, gy), (r0 + 1, gy - 1)]
    for (px_, py_) in pts:
        P_(px_, py_, col('mist'), over=False)
    if P['sound'] and tip:
        r = 1 + P['sound'] * max(1, s)
        cx_, cy_ = tip
        for (dx, dy) in ((r, -r - 1), (r + 1, -r), (r + 1, -r + 1)):
            P_(cx_ + dx, cy_ + dy, col('white'), over=False)
        if P['sound'] >= 2:
            r2 = r + 2
            for (dx, dy) in ((r2, -r2 - 1), (r2 + 1, -r2), (r2 + 1, -r2 + 1), (r2 + 1, -r2 + 2)):
                P_(cx_ + dx, cy_ + dy, col('white'), over=False)
    if mode == 'sil':
        a[a[:, :, 3] > 0, :3] = 255
    a.setflags(write=False)
    return a, ox, oy, (ex, ey, P['eye'] == 'open')


# ---- scale-1 kiwi: hand-drawn interiors (outline is recomputed after the shape ops)
# H highlight, B base, S shade, D deep, F face (light), E eye spot (on base)
_KIWI1_ART = {
    'normal': [
        '..HHHHB...',
        '.HHHHBBBB.',
        'HHHBBBBBEF',
        'HHBBBMBBBB',
        'HBBBMBBMBS',
        'BBSBBBMBSS',
        '.SSSBSSSS.',
    ],
    'sleep': [
        '...HHHB...',
        '.HHHHBBBB.',
        'HHHBBMBBEF',
        'HHBBBBBBBB',
        'BBSBBBMBSS',
        '.SSSSBSSS.',
    ],
}


def _kiwi1_grid(P):
    g = [list(r) for r in _KIWI1_ART['sleep' if P.get('sleep') else 'normal']]
    if P.get('breathe'):
        g.insert(3, list(g[3]))
    sq = P.get('sq', 0)
    if sq < 0:            # squash: lose a row, gain a column
        del g[3]
        for r in g:
            r.insert(4, r[4])
    elif sq > 0:          # stretch: gain a row, lose a column
        g.insert(3, list(g[3]))
        for r in g:
            del r[4]
    sh = P.get('shear', 0)
    if sh:
        n = len(g)
        for i in range(n):
            if i <= 2:
                r = g[i]
                if sh > 0:
                    g[i] = ['.'] + r[:-1] if r[-1] == '.' else ['.'] + r
                else:
                    g[i] = r[1:] + ['.'] if r[0] == '.' else r + ['.']
        w = max(len(r) for r in g)
        g = [r + ['.'] * (w - len(r)) for r in g]
    return g


def _kiwi_build1(pose, k, mode):
    P = _kiwi_params(pose, k)
    KC = {n: np.array(col(v), np.uint8) for n, v in KIWI_COLORS.items()}
    g = _kiwi1_grid(P)
    gh, gw = len(g), len(g[0])
    Ww, Hh = 32, 30
    ox, oy = 15, 28
    a = np.zeros((Hh, Ww, 4), np.uint8)
    lift = int(round(P['lift'] + P.get('air', 0.0)))
    legs_h = 0 if P.get('sleep') else 2
    bot = oy - 2 - legs_h - lift          # last interior row
    top = bot - gh + 1
    left = ox - gw // 2 + int(round(P['bx']))
    cmap = {'H': 'hi', 'B': 'base', 'S': 'shade', 'D': 'deep', 'F': 'hi', 'E': 'base', 'M': 'mid'}
    m = np.zeros((Hh, Ww), bool)
    eye = None
    for r, row in enumerate(g):
        for q, ch in enumerate(row):
            if ch == '.':
                continue
            X, Y = left + q, top + r
            a[Y, X, :3] = KC[cmap[ch]]
            a[Y, X, 3] = 255
            m[Y, X] = True
            if ch == 'E':
                eye = (X, Y)
    pad = np.zeros((Hh + 2, Ww + 2), bool)
    pad[1:-1, 1:-1] = m
    gg = pad[1:-1, 1:-1] | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    ring = gg & ~m
    a[ring, :3] = KC['out']
    a[ring, 3] = 255

    def P_(px, py, key, over=True):
        if 0 <= px < Ww and 0 <= py < Hh and (over or a[py, px, 3] == 0):
            a[py, px, :3] = KC[key] if isinstance(key, str) else key
            a[py, px, 3] = 255

    # legs
    for (hipx, footx, fl, far) in sorted(P['legs'], key=lambda t: -t[3]):
        key = 'leg_far' if far else 'leg'
        hx0 = ox + int(math.floor(hipx + P['bx'])) - (0 if hipx > 0 else 1)
        fx0 = ox + int(math.floor(footx))
        fy0 = oy - 1 - int(round(fl + P.get('air', 0.0)))
        y0 = bot + 2
        n = max(fy0 - y0, 0)
        for i in range(n + 1):
            yy_ = y0 + i
            xx_ = int(round(hx0 + (fx0 - hx0) * (i / n if n else 1)))
            P_(xx_, yy_, key, over=not far)
        P_(fx0 + 1, fy0, key, over=not far)
    # beak from the front of the face row below the eye
    ex, ey = eye
    by_ = ey + 1
    row = np.nonzero(m[by_])[0]
    bx0 = row.max() + 1 if len(row) else ex + 2
    ang = math.radians(P['ba'])
    L_ = P['bl'] - (1.2 if P.get('sleep') else 0)
    tx = bx0 + int(round(math.cos(ang) * L_))
    ty = by_ + int(round(math.sin(ang) * L_))
    pts = []
    x0_, y0_ = bx0, by_
    dx_, dy_ = abs(tx - x0_), -abs(ty - y0_)
    sx_, sy_ = (1 if x0_ < tx else -1), (1 if y0_ < ty else -1)
    err = dx_ + dy_
    while True:
        if y0_ <= oy - 1:
            pts.append((x0_, y0_))
        if x0_ == tx and y0_ == ty:
            break
        e2 = 2 * err
        if e2 >= dy_:
            err += dy_
            x0_ += sx_
        if e2 <= dx_:
            err += dx_
            y0_ += sy_
    tip = None
    for i, (px_, py_) in enumerate(pts):
        P_(px_, py_, 'beak' if i < max(2, len(pts) * 0.6) else 'beak_sh')
        tip = (px_, py_)
    # eye
    if P['eye'] == 'open':
        P_(ex, ey, 'eye')
    else:
        P_(ex, ey, 'shade')
        P_(ex - 1, ey, 'shade')
    # extras
    gy = oy - 1
    pts2 = []
    if P['dust'] == 1:
        gx = left - 2
        pts2 = [(gx, gy), (gx - 1, gy - 1), (gx - 2, gy)]
    elif P['dust'] == 2 and tip:
        pts2 = [(tip[0] - 1, gy - 1), (tip[0] + 1, gy - 2), (tip[0] + 2, gy)]
    elif P['dust'] == 3:
        pts2 = [(left - 2, gy), (left - 3, gy - 1), (left + gw + 1, gy), (left + gw + 2, gy - 1)]
    for (px_, py_) in pts2:
        P_(px_, py_, col('mist'), over=False)
    if P['sound'] and tip:
        r = 1 + P['sound']
        for (dx, dy) in ((r, -r - 1), (r + 1, -r), (r + 1, -r + 1)):
            P_(tip[0] + dx, tip[1] + dy, col('white'), over=False)
    if mode == 'sil':
        a[a[:, :, 3] > 0, :3] = 255
    a.setflags(write=False)
    return a, ox, oy, (ex, ey, P['eye'] == 'open')


@lru_cache(maxsize=None)
def _kiwi(pose, k, s, flip, mode):
    a, ox, oy, eye = (_kiwi_build1(pose, k, mode) if s == 1 else _kiwi_build(pose, k, s, mode))
    if flip:
        a = np.ascontiguousarray(a[:, ::-1])
        a.setflags(write=False)
        ox = a.shape[1] - ox
        eye = (a.shape[1] - 1 - eye[0], eye[1], eye[2])
    return a, ox, oy, eye


def _kiwi_step(pose, f):
    if pose not in _KIWI_PERIOD:
        raise KeyError(f'kiwi pose {pose!r}; one of {KIWI_POSES}')
    return (int(f) % _KIWI_PERIOD[pose]) // 3


def draw_kiwi(c, x, y, pose='idle', f=0, facing='right', scale=1):
    """The mascot kiwi. (x, y) = ground point under the body centre (feet on row y-1).
    pose: KIWI_POSES. scale 1 (~11x9 body + 5 px beak), 2 or 3 (re-rasterised, not
    blown up). Poses step every 3 frames; walk/hop loop once per beat (hop lands on
    f % 18 == 0), idle/look_up/call once per bar."""
    a, ox, oy, _ = _kiwi(pose, _kiwi_step(pose, f), int(scale), facing == 'left', 'color')
    _blit(c, a, x - ox, y - oy)


@lru_cache(maxsize=None)
def _kiwi_tinted(pose, k, s, flip, rgb):
    a, ox, oy, eye = _kiwi(pose, k, s, flip, 'sil')
    t = a.copy()
    t[t[:, :, 3] > 0, :3] = rgb
    t.setflags(write=False)
    return t, ox, oy, eye


def draw_kiwi_silhouette(c, x, y, f=0, facing='right', color='ink', pose='idle', scale=1,
                         eye=None):
    """Single-colour kiwi (dark-window / backlit cameo), same poses/anchor as draw_kiwi.
    eye: optional colour for a glinting eye dot (e.g. 'gold' or 'mist' at night)."""
    t, ox, oy, (ex, ey, open_) = _kiwi_tinted(pose, _kiwi_step(pose, f), int(scale),
                                              facing == 'left', col(color))
    _blit(c, t, x - ox, y - oy)
    if eye is not None and open_:
        c.px(x - ox + ex, y - oy + ey, eye)


def kiwi_size(scale=1):
    """Approximate (w, h) of the kiwi (body + beak) at a scale, for layout."""
    return (16 * scale, 10 * scale)


# =====================================================================================
# GADGETS & BAGS: phone, backpack, gym bag, laptop, sticky note
# =====================================================================================
_reg('phone_stand', '''
    .ooo.
    okSko
    o...o
    o...o
    o...o
    o...o
    okkko
    .ooo.
''', {'S': 'shadow'})

_reg('phone_big', '''
    .ooooooo.
    okkkSkkko
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    ok.......
    okkkkkkko
    .ooooooo.
''', {'S': 'shadow'})

_reg('phone_flat', '''
    ooooooo
    oCcccco
    .ooooo.
''', {'C': 'salmon', 'c': 'coral'})

_PHONE_SCREENS = {
    'off': [['screen', 'screen', 'navy'], ['screen', 'navy', 'screen'],
            ['navy', 'screen', 'screen'], ['screen', 'screen', 'screen']],
    'on': [['sky', 'sky', 'light_sky'], ['white', 'light_sky', 'white'],
           ['blue', 'sky', 'lavender'], ['indigo', 'violet', 'violet']],
    'ring0': [['coral', 'butter', 'coral'], ['butter', 'butter', 'butter'],
              ['butter', 'butter', 'butter'], ['coral', 'amber', 'coral']],
    'ring1': [['orange', 'orange', 'butter'], ['orange', 'butter', 'butter'],
              ['butter', 'butter', 'butter'], ['orange', 'amber', 'orange']],
}


def _phone_buzz(f):
    """Ringing pattern: buzz for 12 frames, rest 6 (one burst per beat)."""
    return (f % FRAMES_PER_BEAT) < 12


def draw_phone(c, x, y, state='off', f=0, size='small', pose='stand', glow=True):
    """The hero's phone. state 'off' | 'on' | 'ringing'.
    size 'small' (5x8, on the nightstand / in a hand) or 'big' (9x14, close-up with a
    lock screen or a ringing alarm bell). pose 'stand' (upright, screen to camera) or
    'flat' (7x3 slab lying on a table; the screen faces up and only glows).
    Ringing: buzzes in bursts once per beat (1 px jitter + shake marks + flashing
    screen); on/ringing also spill light around (glow=False to skip)."""
    buzz = state == 'ringing' and _phone_buzz(f)
    dx = (1 if (f // 1) % 2 else -1) if buzz else 0
    dy = 0
    if pose == 'flat':
        dy = -1 if buzz and f % 4 < 2 else 0
        if state != 'off' and glow:
            from . import fx
            g = 'gold' if state == 'ringing' else 'light_sky'
            fx.glow(c, x, y - 4, 7, g, 0.30 if state == 'ringing' else 0.18)
        _draw(c, 'phone_flat', x + dx, y + dy)
        if state != 'off':
            cc = ('butter' if f % 6 < 3 else 'coral') if state == 'ringing' else 'light_sky'
            c.hline(x - 2 + dx, x + 2 + dx, y - 3 + dy, cc)
        if buzz:
            shake_lines(c, x - 4, y - 4, 9, 4, f)
        return
    big = size == 'big'
    w, h = (9, 14) if big else (5, 8)
    x0, y0 = x - w // 2 + dx, y - h
    if state != 'off' and glow:
        from . import fx
        g = ('gold' if (f // 3) % 2 else 'coral') if state == 'ringing' else 'light_sky'
        fx.glow(c, x, y - h // 2, 9 if big else 7, g, 0.30 if state == 'ringing' else 0.2)
    _blit(c, _rgba('phone_big' if big else 'phone_stand'), x0, y0)
    if not big:
        key = state if state != 'ringing' else ('ring0' if (f // 3) % 2 == 0 else 'ring1')
        scr = _PHONE_SCREENS[key]
        for r in range(4):
            for q in range(3):
                c.px(x0 + 1 + q, y0 + 2 + r, scr[r][q])
    else:
        _phone_big_screen(c, x0 + 2, y0 + 2, state, f)
    if buzz:
        shake_lines(c, x0, y0, w, h, f)


def _phone_big_screen(c, sx, sy, state, f):
    """7 x 10 screen at (sx, sy)."""
    if state == 'off':
        c.rect(sx, sy, 7, 10, 'screen')
        for i in range(4):
            c.px(sx + 5 - i, sy + 1 + i, 'navy')
        c.px(sx + 6, sy, 'navy')
        return
    if state == 'on':
        c.gradient(sx, sy, 7, 10, ['sky', 'light_sky', 'lavender', 'violet'], soft=0.8)
        c.rect(sx + 1, sy + 2, 5, 2, 'white')            # the big clock
        c.px(sx + 3, sy + 2, 'light_sky')
        c.px(sx + 3, sy + 3, 'light_sky')
        c.hline(sx + 2, sx + 4, sy + 5, 'white')            # date line
        c.rect(sx + 1, sy + 7, 5, 2, 'paper')              # a notification card
        c.px(sx + 1, sy + 7, 'gold')
        return
    # ringing: flashing background, wiggling bell, slide-to-stop bar
    c.rect(sx, sy, 7, 10, 'coral' if (f // 3) % 2 == 0 else 'orange')
    tilt = [0, 1, 0, -1][(f // 2) % 4] if _phone_buzz(f) else 0
    bx, by = sx + 1 + tilt, sy + 2
    bell = ['..y..', '.YYY.', '.YYY.', 'YYYYY', '..a..']
    for r, row in enumerate(bell):
        for q, ch in enumerate(row):
            if ch != '.':
                c.px(bx + q, by + r, {'y': 'gold', 'Y': 'butter', 'a': 'amber'}[ch])
    c.hline(sx + 1, sx + 5, sy + 8, 'blush')
    c.px(sx + 1 + (f // 4) % 5, sy + 8, 'white')


PACK_COLORS = {'dark': 'maroon', 'shade': 'rust', 'base': 'copper', 'hi': 'clay',
               'strap': 'brew', 'zip': 'gold'}
_PK = {'v': PACK_COLORS['dark'], 'C': PACK_COLORS['shade'], 'c': PACK_COLORS['base'],
       'K': PACK_COLORS['hi'], 'V': PACK_COLORS['strap'], 'z': PACK_COLORS['zip'],
       'b': 'brew_mid', 'g': 'sand'}

# side view, front of the pack (pocket) to the right, straps on the back (left)
_reg('backpack_side', '''
    ....ooo.....
    ...o...o....
    ..ooooooo...
    .oKKKccCCo..
    oVKKccccCCo.
    oVKcccccCCo.
    oVKccccccCo.
    oVKcccccoooo
    oVKccccoKccCo
    oVKccccoKczCo
    oVKccccoccccCo
    oVccccCocccCCo
    oVcCCCCoCCCCvo
    .oVvvvvoovvvo.
    ..oooooo.ooo..
''', _PK)

# front view (standing on the floor, straps hidden behind)
_reg('backpack_front', '''
    ....oooo....
    ...o.oo.o...
    ..oooooooo..
    .oKKKccccCo.
    oVKKccccccVo
    oVKcccccccVo
    oVKcbbbbbcVo
    oVoooooooooVo
    oVoKKcccCCoVo
    oVoKcczccCoVo
    oVoKcccccCoVo
    oVoCCCCCCvoVo
    oVvoooooooovo
    .oooooooooooo
''', _PK)


def draw_backpack(c, x, y, facing='right', view='side', open_=False):
    """The hero's rust/copper backpack (matches the character's; colours in PACK_COLORS).
    view 'side' (14x15, pocket towards `facing`, straps on the back) or 'front' (13x14).
    open_=True: the top flap gapes open (packing scenes) — a dark opening on top."""
    name = 'backpack_side' if view == 'side' else 'backpack_front'
    arr = _rgba(name, facing == 'left' and view == 'side')
    _put(c, arr, x, y)
    if open_:
        h, w = arr.shape[:2]
        x0 = x - w // 2
        c.hline(x0 + 3, x0 + 8, y - h + 2, 'ink')
        c.hline(x0 + 3, x0 + 8, y - h + 3, 'brew')


_reg('gym_bag', '''
    .....oooooo.....
    ....o......o....
    ..ooooooooooooo.
    .oSSSSSSSSSSSSso
    oSssssssssssssNo
    oSsTTTTTTTTTTsNo
    oSssssssssssssNo
    oNsssssssssssNNo
    .oNNNNNNNNNNNNo.
    ..oooooooooooo..
''', {'S': 'indigo', 's': 'navy', 'N': '#25263f', 'T': 'teal'})


def draw_gym_bag(c, x, y, facing='right'):
    """Navy duffel gym bag (16x10) with a teal stripe and a carry handle on top."""
    _draw(c, 'gym_bag', x, y, facing == 'left')


# ---- laptop
_reg('laptop_lid', '''
    oooooooooooooo
    o............o
    o............o
    o............o
    o............o
    o............o
    o............o
    o............o
    oSSSSSSSSSSSSo
''', {'S': 'steel_dk'})
_reg('laptop_base', '''
    .oooooooooooooo.
    oHhhhhhhhhhhhhHo
    oMMMMMMMMMMMMMMo
    .oooooooooooooo.
''', {'H': 'steel_hi', 'h': 'stone', 'M': 'steel'})
_reg('laptop_closed', '''
    .oooooooooooooo.
    oHHHHHHHHHHHHHMo
    oMMMMMMMMMMMMMMo
    .oooooooooooooo.
''', {'H': 'steel_hi', 'M': 'steel'})
_reg('laptop_side', '''
    o.........
    oo........
    .oo.......
    .oSo......
    ..oSo.....
    ..oSSo....
    ...oSo....
    ...oSSo...
    ....oooooooooo
    ....oMMMMMMMMo
    .....ooooooooo
''', {'S': 'steel', 'M': 'steel_hi'})


def _laptop_screen(c, sx, sy, w, h, screen, f):
    if screen == 'off':
        c.rect(sx, sy, w, h, 'screen')
        c.px(sx + w - 2, sy + 1, 'navy')
        c.px(sx + w - 3, sy + 2, 'navy')
        return
    if screen == 'code':
        c.rect(sx, sy, w, h, 'screen')
        scroll = f // 6
        palette = ['lavender', 'turquoise', 'gold', 'pink', 'light_sky', 'stone']
        for r in range(h - 1):
            i = r + scroll
            ind = int(hash01(i, 1) * 3)
            ln = 2 + int(hash01(i, 2) * (w - 4 - ind))
            c.hline(sx + 1 + ind, sx + ind + ln, sy + 1 + r, palette[int(hash01(i, 3) * 6)])
        if (f // 9) % 2 == 0:
            c.px(sx + 2, sy + h - 1, 'white')
        return
    if screen == 'chat':
        c.rect(sx, sy, w, h, 'paper')
        c.rect(sx, sy, w, 1, 'mist')
        n = (f // 18) % 4
        rows = [(1, 6, 'stone'), (w - 8, 6, 'light_sky'), (1, 8, 'stone'), (w - 7, 5, 'light_sky')]
        for i in range(min(n + 1, 3)):
            bx, bw, cc = rows[(i + max(0, n - 2)) % 4]
            c.rect(sx + bx, sy + 2 + i * 2, bw, 1, cc)
        c.rect(sx + 1, sy + h - 1, w - 2, 1, 'mist')
        if (f // 9) % 2 == 0:
            c.px(sx + 2, sy + h - 1, 'ink')
        return
    if screen == 'plot':
        c.rect(sx, sy, w, h, 'screen')
        c.vline(sx + 1, sy + 1, sy + h - 2, 'stone')
        c.hline(sx + 1, sx + w - 2, sy + h - 2, 'stone')
        for i in range(w - 3):
            u = (i - (w - 4) / 2) / ((w - 4) / 4)
            v = math.exp(-u * u)
            py = sy + h - 3 - int(round(v * (h - 4)))
            c.px(sx + 2 + i, py, 'gold' if i % 2 else 'turquoise')
        return
    # any other value: a colour fill (e.g. 'light_sky')
    c.rect(sx, sy, w, h, screen)


def draw_laptop(c, x, y, open=True, f=0, screen='code', view='front', facing='right'):
    """Laptop. view 'front' (16x13, screen facing camera: screen 'code' scrolling,
    'chat', 'plot', 'off' or a colour), 'side' (14x11 L-shape, the screen faces
    `facing`; a little screen light spills out). open=False: closed (16x4)."""
    if not open:
        _draw(c, 'laptop_closed', x, y)
        return
    if view == 'side':
        arr = _rgba('laptop_side', facing == 'left')
        _put(c, arr, x, y)
        return
    _draw(c, 'laptop_base', x, y)
    lid = _rgba('laptop_lid')
    lx, ly = x - lid.shape[1] // 2, y - 4 - lid.shape[0] + 1
    _blit(c, lid, lx, ly)
    _laptop_screen(c, lx + 1, ly + 1, 12, 7, screen, f)


# ---- sticky note
_NOTE_COLS = {'yellow': ('butter', 'gold', 'sand'), 'pink': ('blush', 'pink', 'salmon'),
              'blue': ('light_sky', 'sky', 'blue'), 'green': ('pale_leaf', 'leaf', 'moss')}
_NOTE_ICONS = {
    'sun': (['.y.y.', 'yGGGy', '.GGG.', 'yGGGy', '.y.y.'], {'y': 'amber', 'G': 'gold'}),
    'heart': (['RR.RR', 'RRRRR', 'RRRRR', '.RRR.', '..R..'], {'R': 'red'}),
    'check': (['....g', '...g.', 'g.g..', '.g...', '.....'], {'g': 'green'}),
    'star': (['..y..', '.yyy.', 'yyyyy', '.yyy.', '.y.y.'], {'y': 'amber'}),
    'kiwi': (['.bb..', 'bbbbe', 'bbbbtt', '.bb..', '.l.l.'], {'b': 'latte', 'e': 'ink', 't': 'tan', 'l': 'mauve'}),
    'lines': (['kkkk.', '.....', 'kkk..', '.....', 'kkkk.'], {'k': 'grey'}),
    'atom': (['k...k', '.kRk.', '.RRR.', '.kRk.', 'k...k'], {'k': 'blue', 'R': 'red'}),
    'moon': (['.yy..', 'yy...', 'yy...', 'yy...', '.yy..'], {'y': 'indigo'}),
    'coffee': (['.....', 'bbbb.', 'bbbbb', 'bbbb.', '.bb..'], {'b': 'brew_mid'}),
}


@lru_cache(maxsize=None)
def _note_rgba(color, icon, curl):
    base, mid, edge = _NOTE_COLS[color]
    a = np.zeros((8, 8, 4), np.uint8)
    for yy in range(7):
        for xx in range(7):
            cc = mid if yy == 0 else base
            if xx == 6 or yy == 6:
                cc = edge
            a[yy, xx, :3] = col(cc)
            a[yy, xx, 3] = 255
    # soft drop shadow
    for i in range(1, 8):
        a[7, i, :3] = col('ink')
        a[7, i, 3] = 70
        a[i, 7, :3] = col('ink')
        a[i, 7, 3] = 70
    if curl:
        a[6, 6, 3] = 0
        a[5, 6, :3] = col(edge)
        a[6, 5, :3] = col(edge)
        a[5, 5, :3] = col(mid)
    if icon:
        art, cm = _NOTE_ICONS[icon]
        for r, row in enumerate(art):
            for q, ch in enumerate(row):
                if ch != '.' and 1 + q < 7 and 1 + r < 7:
                    a[1 + r, 1 + q, :3] = col(cm[ch])
    a.setflags(write=False)
    return a


def draw_sticky_note(c, x, y, icon='sun', color='yellow', curl=True):
    """Sticky note (7x7 + soft shadow) with a 5x5 doodle. icon: 'sun', 'heart', 'check',
    'star', 'kiwi', 'lines', 'atom', 'moon', 'coffee' or None. color: 'yellow', 'pink',
    'blue', 'green'. Bottom-centre anchored like the rest (stick it on the jar with
    y = jar_y - 4)."""
    _put(c, _note_rgba(color, icon, bool(curl)), x, y)


# =====================================================================================
# KITCHEN: pan, pot, rice cooker, toaster, kettle, plate, bowl, mug, utensils, board
# =====================================================================================
_reg('pan', '''
    ..oooooooooooo..........
    .oddddddddddddo.........
    oHkkkkkkkkkkkkMo......oo
    oMMMMMMMMMMMMMMooooooWWo
    .oCCCCCCCCCCCCooWWWWwwo.
    ..oooooooooooo.oooooooo.
''', {'d': 'night', 'k': 'charcoal', 'H': 'steel_hi', 'M': 'steel', 'C': 'steel_dk',
      'W': 'copper', 'w': 'rust'})
_PAN_BODY_W = 16


def draw_pan(c, x, y, contents=None, f=0, flip=False, sizzle=True):
    """Frying pan, side view (24x6, the 16 px pan body is centred on x; the wooden
    handle sticks out to the right, flip=True for the left). contents None | 'egg'
    (fried egg, yolk dome) | 'stirfry' (colourful veg + chicken, tossed up on every
    beat: f % 18 == 0 starts a toss). sizzle: oil spits / steam dots while cooking."""
    arr = _rgba('pan', flip)
    w, h = arr.shape[1], arr.shape[0]
    x0 = x - _PAN_BODY_W // 2 if not flip else x + _PAN_BODY_W // 2 - w
    _blit(c, arr, x0, y - h)
    bx = x - _PAN_BODY_W // 2      # left edge of pan body
    top = y - h                     # rim row
    if contents == 'egg':
        c.hline(bx + 4, bx + 11, top + 1, 'paper')
        c.hline(bx + 5, bx + 10, top, 'white')
        c.hline(bx + 3, bx + 5, top + 1, 'white')
        c.px(bx + 12, top + 1, (226, 208, 182))
        c.rect(bx + 7, top - 1, 3, 2, 'gold')
        c.px(bx + 7, top - 1, 'butter')
        c.px(bx + 9, top, 'amber')
        c.px(bx + 8, top - 2, 'gold')
    elif contents == 'stirfry':
        t = f % FRAMES_PER_BEAT
        toss = t < 9
        pieces = [('copper', 3), ('leaf', 5), ('vermilion', 7), ('green', 9), ('orange', 11),
                  ('clay', 6), ('forest', 8), ('gold', 10), ('red', 4), ('pale_leaf', 12)]
        for i, (cc, px_) in enumerate(pieces):
            hgt = 0
            if toss:
                hgt = int(round((3 + 3 * hash01(i, 9)) * math.sin(math.pi * t / 9)))
            row = top + (0 if i < 5 else -1) - hgt
            c.px(bx + px_ + (1 if toss and i % 3 == 0 and t > 4 else 0), row, cc)
            if i < 5:
                c.px(bx + px_ + 1, row, shade(cc, 0.8))
        c.hline(bx + 3, bx + 12, top + 1, 'clay')
    if contents and sizzle:
        for i in range(4):
            ph = (f + i * 5) % 12
            if ph < 6:
                sx = bx + 3 + int(hash01(i, f // 12) * 10)
                c.px(sx, top - 1 - ph // 2, 'white' if i % 2 else 'butter')


def draw_flame(c, x, y, f, size=1):
    """Gas-hob flame ring under a pan/pot: (x, y) = bottom-centre (the burner top);
    size 1 (10 px wide) or 2 (14, high heat for the stir-fry). Flickers every 2 frames."""
    w = 10 if size == 1 else 14
    for i in range(w // 2):
        xx = x - w // 2 + 1 + i * 2
        hgt = 1 + int(hash01(i, f // 2) * (2 if size == 1 else 3))
        c.vline(xx, y - hgt, y - 1, 'laser_blu')
        c.px(xx, y - 1, 'light_sky')
        if hgt >= 3:
            c.px(xx, y - hgt, 'amber')


_reg('pot', '''
    ......oooo......
    .....oHSSo......
    ...oooooooooo...
    ..oHHSSSSSSMMo..
    .oooooooooooooo.
    oooHRRRRRRRRrooo
    oWoHRRRRRRRRroWo
    ooooHRRRRRRRrooo
    ...oHRRRRRRrro..
    ...oCrrrrrrrro..
    ....oooooooooo..
''', {'H': 'steel_hi', 'S': 'stone', 'M': 'grey', 'R': 'vermilion', 'r': 'brick',
      'C': 'coral', 'W': 'brick'})


def draw_pot(c, x, y, f=0, steam=True, lid=True):
    """Red enamel cooking pot with a steel lid (16x11). steam=True: wisps rise from
    under the lid and the lid rattles 1 px now and then. lid=False: open pot."""
    arr = _rgba('pot')
    w, h = arr.shape[1], arr.shape[0]
    x0, y0 = x - w // 2, y - h
    if lid:
        rattle = 1 if steam and (f % 36) in (0, 1, 2, 9, 10) else 0
        _blit(c, arr[4:], x0, y0 + 4)
        _blit(c, arr[:4], x0, y0 - rattle)
    else:
        _blit(c, arr[4:], x0, y0 + 4)
        c.hline(x0 + 2, x0 + 13, y0 + 4, 'ink')
        c.hline(x0 + 3, x0 + 12, y0 + 5, 'maroon')
    if steam:
        from . import fx
        fx.steam(c, x - 4, y0 + 2, f, 'white', n=2, height=10, seed=1)
        fx.steam(c, x + 4, y0 + 2, f + 7, 'mist', n=2, height=12, seed=2)


_reg('rice_cooker', '''
    .....oooo.....
    ...ooWWWWoo...
    ..oWWWWWWWWo..
    .oWWWWWWWWWso.
    ooooooooooooso
    oHWWWWWWWWWWso
    oHWkkkkkkWWWso
    oHWkkkkkkWpWso
    oHWWWWWWWpPpso
    oHWWWWWWWWpsso
    .osssssssssso.
    ..oooooooooo..
''', {'W': 'paper', 'H': 'white', 's': 'mist', 'k': 'night', 'p': 'pink', 'P': 'gold'})


def draw_rice_cooker(c, x, y, f=0, steaming=False):
    """Cream rice cooker (14x12) with a little flower decal and a front panel.
    steaming=True: 'COOK' LED red + puffs of steam from the vent (bigger puff every beat);
    otherwise the 'WARM' LED glows amber."""
    arr = _rgba('rice_cooker')
    w, h = arr.shape[1], arr.shape[0]
    x0, y0 = x - w // 2, y - h
    _blit(c, arr, x0, y0)
    c.px(x0 + 4, y0 + 7, 'red' if steaming and (f // 9) % 2 == 0 else 'amber' if not steaming else 'brick')
    c.px(x0 + 6, y0 + 7, 'screen_txt' if steaming else 'shadow')
    c.px(x0 + 7, y0 + 7, 'screen_txt' if steaming else 'shadow')
    c.px(x0 + 6, y0, 'steel')
    c.px(x0 + 7, y0, 'steel')
    if steaming:
        from . import fx
        fx.steam(c, x0 + 7, y0 - 1, f, 'white', n=3, height=14, spread=2, seed=5)
        t = f % FRAMES_PER_BEAT
        if t < 8:
            r = 1 + t // 3
            c.circle(x0 + 7, y0 - 3 - t, r, 'white')
            c.circle(x0 + 8, y0 - 4 - t, max(0, r - 1), 'mist')


_reg('toaster', '''
    .oooooooooooo..
    oHkkkkWkkkkWso.
    oWWWWWWWWWWWsoo
    oRRRRRRRRRRRro.
    oWWWWWWWWWWWso.
    oWWWWWWWWWWWso.
    oHWWWWWWWWWssoo
    .oooooooooooo..
    .oo........oo..
''', {'W': 'paper', 'H': 'white', 's': 'tan', 'k': 'ink', 'R': 'red', 'r': 'brick'})
_reg('toast_slice', '''
    .oooo.
    oCCCCo
    oCTtco
    oCttco
    oCttco
    oCttco
    oCttco
''', {'C': 'copper', 'c': 'rust', 'T': 'butter', 't': 'sand'})


def draw_toaster(c, x, y, toast=0.0, f=0, heating=False):
    """Retro cream toaster with a red stripe (15x9). toast 0..1: two slices rise out of
    the slots (1 = popped up 6 px; animate with ease_out_back for a springy pop).
    The lever on the side is down while toasting. heating=True: slots glow orange."""
    arr = _rgba('toaster')
    w, h = arr.shape[1], arr.shape[0]
    x0, y0 = x - w // 2, y - h
    rise = int(round(clamp(toast, -0.2, 1.3) * 6))
    if toast > 0 or heating:
        sl = _rgba('toast_slice')
        for sx in (x0 + 1, x0 + 7):
            _blit(c, sl[:max(0, rise + 1)], sx, y0 + 1 - rise)
    _blit(c, arr, x0, y0)
    if heating and toast <= 0.05:
        glow_c = 'orange' if (f // 4) % 2 else 'amber'
        c.hline(x0 + 2, x0 + 5, y0 + 1, glow_c)
        c.hline(x0 + 7, x0 + 10, y0 + 1, glow_c)
    lever_y = y0 + (2 if toast > 0.5 or not heating else 5)
    c.rect(x0 + 13, lever_y, 2, 1, 'ink')
    c.px(x0 + 14, lever_y, 'grey')


_reg('kettle', '''
    ...ooooo....
    ..oHSSSSo...
    .ooooooooo..
    oo.oHSSSSSooo
    o..oHSSSSSSoWo
    o..oHSSSSSSooo
    o..oHSWWSSSo..
    oooHSSWbSSMo..
    ..oHSSSSSSMo..
    ..oHSSSSSMMo..
    .ooooooooooooo
    .oBBBBBBBBBBBo
    .oooooooooooo.
''', {'H': 'steel_hi', 'S': 'stone', 'M': 'grey', 'W': 'light_sky', 'b': 'sky',
      'B': 'steel_dk'})


def draw_kettle(c, x, y, f=0, steaming=False):
    """Electric jug/kettle (14x13, handle on the left, spout right) with a water
    window. steaming=True: steam jets from the spout, the power light glows red and it
    rumbles (1 px shake on alternate frames during the last beat of each bar)."""
    arr = _rgba('kettle')
    w, h = arr.shape[1], arr.shape[0]
    shake_x = 1 if steaming and (f % 72) >= 54 and f % 2 else 0
    x0, y0 = x - w // 2 + shake_x, y - h
    _blit(c, arr, x0, y0)
    c.px(x0 + 3, y0 + 11, 'red' if steaming else 'shadow')
    if steaming:
        from . import fx
        fx.steam(c, x0 + 13, y0 + 3, f, 'white', n=3, height=12, spread=3, seed=8, speed=0.5)
        for i in range(2):
            t = (f + i * 7) % 14
            c.px(x0 + 15 + t // 3, y0 + 3 - t // 2, 'mist')


_reg('plate', '''
    .oooooooooooooo.
    oWWWWWWWWWWWWWWo
    .osWWWWWWWWWWso.
    ..oooooooooooo..
''', {'W': 'white', 's': 'mist'})


def draw_plate(c, x, y, kind='dinner'):
    """Plate seen from the side (16x4 + food). kind 'dinner' (rice + stir-fry),
    'breakfast' (fried egg + toast), 'empty', 'crumbs' (finished)."""
    _draw(c, 'plate', x, y)
    t = y - 4               # plate surface row
    x0 = x - 8
    if kind == 'dinner':
        c.hline(x0 + 2, x0 + 7, t, 'paper')
        c.hline(x0 + 3, x0 + 6, t - 1, 'white')
        c.hline(x0 + 4, x0 + 5, t - 2, 'white')
        c.px(x0 + 3, t, (214, 212, 204))
        c.px(x0 + 6, t - 1, (214, 212, 204))
        for i, cc in enumerate(['copper', 'leaf', 'orange', 'green', 'vermilion', 'clay', 'forest']):
            c.px(x0 + 8 + i, t, cc)
        for i, cc in enumerate(['clay', 'green', 'red', 'leaf', 'copper']):
            c.px(x0 + 9 + i, t - 1, cc)
        c.px(x0 + 11, t - 2, 'orange')
        c.px(x0 + 12, t - 2, 'leaf')
    elif kind == 'breakfast':
        c.hline(x0 + 2, x0 + 8, t, 'paper')
        c.hline(x0 + 3, x0 + 7, t - 1, 'white')
        c.rect(x0 + 4, t - 2, 3, 2, 'gold')
        c.px(x0 + 4, t - 2, 'butter')
        _blit(c, _rgba('food_toast'), x0 + 8, t - 7)
    elif kind == 'crumbs':
        for i, cc in ((3, 'sand'), (6, 'clay'), (9, 'sand'), (11, 'copper')):
            c.px(x0 + i, t, cc)
        c.px(x0 + 12, t - 1, 'gold')
    if kind in ('empty', 'crumbs'):
        c.px(x0 + 3, t + 1, 'white')


_reg('bowl', '''
    oooooooooooo
    oWWWWWWWWWWo
    oBWBWBWBWBWo
    .oWWWWWWWso.
    ..oWWWWsso..
    ...oooooo...
''', {'W': 'paper', 'B': 'sky', 's': 'mist'})


def draw_bowl(c, x, y, kind='rice', f=0, steam=False):
    """Ceramic bowl with a blue band (12x6 + contents). kind 'rice' (white dome),
    'noodles' (with chopsticks), 'soup', 'empty'. steam=True adds rising wisps."""
    _draw(c, 'bowl', x, y)
    t = y - 6
    x0 = x - 6
    if kind == 'rice':
        c.hline(x0 + 1, x0 + 10, t, 'white')
        c.hline(x0 + 2, x0 + 9, t - 1, 'white')
        c.hline(x0 + 4, x0 + 7, t - 2, 'white')
        for px_, py_ in ((3, 0), (6, -1), (8, 0), (5, -2), (9, -1)):
            c.px(x0 + px_, t + py_, (214, 212, 204))
        c.px(x0 + 3, t - 1, 'white')
    elif kind == 'noodles':
        c.hline(x0 + 1, x0 + 10, t, 'gold')
        c.hline(x0 + 2, x0 + 9, t - 1, 'butter')
        for i in range(2, 10, 2):
            c.px(x0 + i, t, 'amber')
            c.px(x0 + i + 1, t - 1, 'gold')
        c.px(x0 + 3, t - 1, 'leaf')
        c.px(x0 + 7, t - 1, 'vermilion')
        c.line(x0 + 7, t - 1, x0 + 11, t - 7, 'copper')
        c.line(x0 + 8, t - 1, x0 + 12, t - 6, 'clay')
    elif kind == 'soup':
        c.hline(x0 + 1, x0 + 10, t, 'clay')
        c.px(x0 + 4, t, 'leaf')
        c.px(x0 + 7, t, 'sand')
    else:
        c.hline(x0 + 1, x0 + 10, t + 1, 'mist')
    if steam:
        from . import fx
        fx.steam(c, x, t - 2, f, 'white', n=3, height=10, seed=11)


_MUG_LIQ = {'tea': 'clay', 'coffee': 'brew', 'latte': 'latte', 'cocoa': 'brew_mid',
            'empty': 'shadow'}


@lru_cache(maxsize=None)
def _mug_rgba(kind, body, stripe, size, design):
    if size == 'big':
        art = '''
            .ooooooo...
            oLLLLLLLo..
            oHBBBBBSoooo
            oHBBBBBSo..o
            oHBBBBBSo..o
            oHBBBBBSo..o
            oHBBBBBSooo.
            oHBBBBSSo...
            .ooooooo....
        '''
    else:
        art = '''
            ooooo..
            oLLLooo
            oHBSo.o
            oHBSo.o
            oHSSoo.
            .ooo...
        '''
    a = _parse(art, {'L': _MUG_LIQ.get(kind, kind), 'H': mix(body, 'white', 0.45), 'B': body,
                     'S': shade(body, 0.8)}).copy()
    if stripe:
        r = 4 if size == 'big' else 3
        for xx in range(1, a.shape[1]):
            if a[r, xx, 3] and tuple(a[r, xx, :3]) != col('ink') and xx < (8 if size == 'big' else 4):
                a[r, xx, :3] = col(stripe)
    if size == 'big' and design:
        ic = {'heart': ['R.R', 'RRR', '.R.'], 'kiwi': ['bb.', 'bbe', '.l.'],
              'sun': ['g.g', '.G.', 'g.g'], 'atom': ['b.b', '.R.', 'b.b']}[design]
        cm = {'R': 'red', 'b': 'latte', 'e': 'ink', 'l': 'mauve', 'g': 'amber', 'G': 'gold'}
        if design == 'atom':
            cm['b'] = 'blue'
        if design == 'kiwi':
            for (dx, dy) in ((5, 4), (6, 4)):
                a[4 + 0, 3 + dx - 2 + 0, :3] = a[4, 3, :3]
        for r, row in enumerate(ic):
            for q, ch in enumerate(row):
                if ch != '.':
                    a[3 + r, 3 + q, :3] = col(cm[ch])
        if design == 'kiwi':
            a[4, 6, :3] = col('tan')
    a.setflags(write=False)
    return a


def draw_mug(c, x, y, kind='tea', f=0, color='paper', stripe='teal', size='small',
             design=None, steam=None):
    """Mug, handle to the right. size 'small' (7x6, the 5x5 body is centred on x) or
    'big' (12x9 close-up; design None | 'heart' | 'kiwi' | 'sun' | 'atom' printed on it).
    kind 'tea' | 'coffee' | 'latte' | 'cocoa' | 'empty' (liquid seen at the rim).
    color/stripe: body colour and band colour (stripe=None for plain). Hot drinks steam
    (steam=None -> automatic: everything except 'empty')."""
    a = _mug_rgba(kind, col(color), col(stripe) if stripe else None, size, design)
    bw = 9 if size == 'big' else 5
    _blit(c, a, x - bw // 2, y - a.shape[0])
    if steam is None:
        steam = kind != 'empty'
    if steam:
        from . import fx
        fx.steam(c, x, y - a.shape[0] - 1, f, 'white', n=2 if size == 'small' else 3,
                 height=8 if size == 'small' else 12, spread=2, seed=3)


draw_cup = draw_mug   # alias (character.py's docstring example uses draw_cup)


_reg('fork', '''
    s.s
    s.s
    sSs
    .s.
    .s.
    .h.
    .h.
    .h.
''', {'s': 'steel_hi', 'S': 'steel', 'h': 'steel'})
_reg('knife', '''
    .s
    ss
    ss
    sS
    sS
    .h
    .h
    .h
''', {'s': 'steel_hi', 'S': 'steel', 'h': 'steel'})
_reg('chef_knife', '''
    ooooooooo.....
    oHHHHHHHHooooo
    .oSSSSSSoWwWwo
    ..ooooooooooo.
''', {'H': 'steel_hi', 'S': 'steel', 'W': 'brew_mid', 'w': 'brass'})
_reg('spatula', '''
    oooo.........
    oHHoooooooooo
    oHSoWWWWWWWWo
    oSSoooooooooo
    oooo.........
''', {'H': 'steel_hi', 'S': 'steel', 'W': 'copper'})


def _orient(name, orient, flip=False):
    a = _rgba(name)
    if orient == 'flat':        # lying down, tip/business end to the right
        a = np.ascontiguousarray(np.rot90(a, -1))
    elif orient == 'down':
        a = np.ascontiguousarray(a[::-1])
    if flip:
        a = np.ascontiguousarray(a[:, ::-1])
    return a


@lru_cache(maxsize=None)
def _utensil(name, orient, flip):
    a = _orient(name, orient, flip)
    a.setflags(write=False)
    return a


def draw_fork(c, x, y, orient='up', flip=False):
    """Fork (3x8). orient 'up' (tines up, e.g. held), 'down', 'flat' (lying on the table,
    8x3, tines to the right; flip=True for left)."""
    _put(c, _utensil('fork', orient, flip), x, y)


def draw_knife(c, x, y, kind='table', orient='up', flip=False):
    """Table knife (2x8, orient 'up' | 'down' | 'flat') or kind='chef': a chef's knife
    (14x4, always side-on, blade to the left, handle right; flip=True mirrors). For
    chopping, hold the chef's knife at the hand and bob it 2-3 px on the beats."""
    if kind == 'chef':
        _draw(c, 'chef_knife', x, y, flip)
        return
    _put(c, _utensil('knife', orient, flip), x, y)


def draw_spatula(c, x, y, flip=False):
    """Kitchen spatula/turner (13x5, blade left, wooden handle right)."""
    _draw(c, 'spatula', x, y, flip)


_reg('cutting_board', '''
    oooooooooooooooooooooo
    oHhhhhhhhhhhhhhhhhooho
    oWWWWWWWWWWWWWWWWWWWWo
    .oooooooooooooooooooo.
''', {'H': 'sand', 'h': 'clay', 'W': 'copper'})

_VEG = {'carrot': ('orange', 'amber', 'leaf'), 'pepper': ('red', 'vermilion', 'green'),
        'onion': ('lilac', 'violet', None), 'broccoli': ('green', 'leaf', None),
        'cucumber': ('green', 'pale_leaf', None)}


def draw_cutting_board(c, x, y, veg=None, cut=0.0):
    """Wooden cutting board (22x4). veg 'carrot' | 'pepper' | 'onion' | 'broccoli' |
    'cucumber': the whole vegetable lies on the right and gets shorter as cut 0..1
    grows, while a heap of slices builds up on the left (bump cut on every beat)."""
    _draw(c, 'cutting_board', x, y)
    if not veg:
        return
    main, light, top = _VEG[veg]
    t = y - 5                  # rests on the board (board top row is y-4)
    x0 = x - 11
    n_slices = int(round(clamp(cut) * 8))
    length = int(round(9 * (1 - clamp(cut))))
    if length > 0:
        vx = x0 + 19 - length
        c.hline(vx, x0 + 18, t, main)
        c.hline(vx, x0 + 18, t - 1, main)
        c.hline(vx, x0 + 17, t - 1, light)
        if top:
            c.px(x0 + 19, t - 1, top)
            c.px(x0 + 20, t - 2, top)
        c.px(vx - 1, t, 'ink')
    for i in range(n_slices):
        sx = x0 + 3 + (i % 5) * 2
        sy = t - (i // 5)
        c.px(sx, sy, main)
        c.px(sx + 1, sy, light)


# =====================================================================================
# HOME / GYM / SHOP: broom, grocery bag, baskets, towel, bottle, dumbbell, kettlebell,
# trolley, books, clothes, receipt, shopping list, potted plants
# =====================================================================================
@lru_cache(maxsize=None)
def _broom_rgba(angle):
    lean = angle * 0.22                       # handle dx per px of height
    L = 24
    Ww, Hh = 32, L + 8
    a = np.zeros((Hh, Ww, 4), np.uint8)
    cx = Ww // 2
    bot = Hh - 1

    def P(x, y, cc):
        if 0 <= x < Ww and 0 <= y < Hh:
            a[y, x, :3] = col(cc)
            a[y, x, 3] = 255
    # bristles: trapezoid 6 rows, top 5 wide -> bottom 9 wide, dragged by the lean
    for r in range(6):
        half = 2 + (r * 2) // 5
        drag = -int(round(lean * (r - 1) * 0.6))
        y = bot - 5 + r
        for dx in range(-half - 1, half + 2):
            xx = cx + dx + drag
            if abs(dx) == half + 1:
                P(xx, y, 'ink')
            else:
                streak = (dx + r) % 3 == 0
                P(xx, y, 'clay' if streak else 'gold' if dx < 0 else 'sand')
        if r == 5:
            for dx in range(-half - 1, half + 2):
                if (dx % 2) == 0:
                    P(cx + dx + drag, y, 'clay')
    for dx in range(-3, 4):                    # binding
        P(cx + dx, bot - 6, 'ink' if abs(dx) == 3 else 'red')
    P(cx - 2, bot - 6, 'salmon')
    for dx in range(-2, 3):
        P(cx + dx, bot - 7, 'ink')
    # handle
    hx_top = None
    for i in range(L):
        y = bot - 8 - i
        x = cx + int(round(lean * i))
        P(x, y, 'copper' if i % 7 else 'clay')
        P(x + 1, y, 'rust')
        hx_top = (x, y)
    P(hx_top[0], hx_top[1] - 1, 'red')
    P(hx_top[0] + 1, hx_top[1] - 1, 'brick')
    a.setflags(write=False)
    return a, cx, bot


def draw_broom(c, x, y, angle=0):
    """Broom (handle 24 px + bristles 8 px). angle -2..2 = handle lean steps (+ leans
    the top to the right; the bristles drag the other way). (x, y) = bottom-centre of
    the bristles. Returns ((x_hi, y_hi), (x_lo, y_lo)): two grip points on the handle
    for the hands."""
    angle = int(max(-2, min(2, round(angle))))
    a, cx, bot = _broom_rgba(angle)
    _blit(c, a, x - cx, y - 1 - bot)
    lean = angle * 0.22
    return ((x + int(round(lean * 19)), y - 1 - 8 - 19), (x + int(round(lean * 9)), y - 1 - 8 - 9))


_reg('grocery_bag', '''
    .oooooooooooo.
    oPPPPPPPPPPPpo
    oKKKKKKKKKKKko
    oKKKKKKKKKKKko
    oKkKKKKKKKKKko
    oKkKKrRrKKKKko
    oKkKKRRRKKKKko
    oKkKKKrKKKKKko
    oKkKKKKKKKKkko
    oKKKKKKKKKKkko
    okkkkkkkkkkkko
    .oooooooooooo.
''', {'P': 'sand', 'p': 'clay', 'K': 'crema', 'k': 'latte', 'r': 'brick', 'R': 'red'})


def draw_grocery_bag(c, x, y, f=0, full=True):
    """Kraft paper grocery bag (14x12) with a little red apple logo; full=True: leafy
    greens and a baguette poke out of the top (the leaves sway gently with f)."""
    arr = _rgba('grocery_bag')
    w, h = arr.shape[1], arr.shape[0]
    x0, y0 = x - w // 2, y - h
    if full:
        sw = 1 if (f // 12) % 4 == 1 else -1 if (f // 12) % 4 == 3 else 0
        # baguette (behind, leaning right)
        for i in range(9):
            bx, by = x0 + 8 + i // 2, y0 + 1 - i
            c.px(bx, by, 'sand' if i % 3 else 'clay')
            c.px(bx + 1, by, 'copper')
        c.px(x0 + 12, y0 - 8, 'ink')
        c.px(x0 + 13, y0 - 8, 'ink')
        c.px(x0 + 12, y0 - 7, 'ink')
        c.px(x0 + 13, y0 - 7, 'rust')
        c.px(x0 + 10, y0 - 3, 'butter')
        # greens (leek / lettuce leaves)
        for i, (lx, lh, cc) in enumerate(((2, 6, 'green'), (4, 8, 'leaf'), (6, 5, 'forest'),
                                          (3, 4, 'pale_leaf'))):
            for j in range(lh):
                d = (sw if j > lh // 2 else 0) + (1 if j > lh - 2 and i % 2 else 0)
                c.px(x0 + lx + d, y0 + 1 - j, cc)
            c.px(x0 + lx + 1 + (sw if lh > 4 else 0), y0 + 1 - lh + 1, 'leaf' if cc != 'leaf' else 'pale_leaf')
        # carrot top
        c.px(x0 + 7, y0, 'orange')
        c.px(x0 + 7, y0 - 1, 'leaf')
        c.px(x0 + 6, y0 - 2, 'leaf')
    _blit(c, arr, x0, y0)


_reg('basket_laundry', '''
    oooooooooooooooo
    oSsSsSsSsSsSsSso
    oWtWtWtWtWtWtWto
    otWtWtWtWtWtWtto
    oWtWtWtWtWtWtWto
    otWtWtWtWtWtWtto
    .oCCCCCCCCCCCCo.
    ..oooooooooooo..
''', {'S': 'sand', 's': 'clay', 'W': 'sand', 't': 'clay', 'C': 'copper'})
_reg('basket_shop', '''
    ...oooooooo...
    ..o........o..
    .o..........o.
    oooooooooooooo
    oRRRRRRRRRRRro
    oRkRkRkRkRkRro
    oRRRRRRRRRRRro
    oRkRkRkRkRkRro
    .orrrrrrrrrro.
    ..oooooooooo..
''', {'R': 'red', 'r': 'brick', 'k': 'maroon'})


def draw_basket(c, x, y, kind='laundry', fill=0.0, items=()):
    """kind 'laundry': wicker laundry basket (16x8); fill 0..1 piles clothes above the
    rim. kind 'shopping': red supermarket hand basket (14x10) with a handle; `items` =
    FOOD names shown poking out of it."""
    if kind == 'shopping':
        arr = _rgba('basket_shop')
        x0, y0 = x - arr.shape[1] // 2, y - arr.shape[0]
        for i, n in enumerate(list(items)[:4]):
            fa = _rgba('food_' + n)
            _blit(c, fa, x0 + 2 + i * 3, y0 + 5 - fa.shape[0] + 1)
        _blit(c, arr, x0, y0)
        return
    arr = _rgba('basket_laundry')
    x0, y0 = x - arr.shape[1] // 2, y - arr.shape[0]
    n = int(round(clamp(fill) * 5))
    heap = [('teal', 2, 5), ('paper', 7, 4), ('indigo', 10, 4), ('coral', 4, 3), ('sky', 9, 3)]
    for i in range(n):
        cc, hx, hw = heap[i]
        hh = 1 + (i < 3)
        c.round_rect(x0 + hx, y0 - hh - (i // 3), hw, hh + 1, cc, r=1)
        c.px(x0 + hx + 1, y0 - hh - (i // 3), mix(cc, 'white', 0.4))
    _blit(c, arr, x0, y0)


_reg('towel_folded', '''
    .oooooooooo.
    oWWWWWWWWWWo
    oBBBBBBBBBbo
    oWWWWWWWWWWo
    oBBBBBBBBBbo
    obbbbbbbbbbo
    .oooooooooo.
''', {'W': 'paper', 'B': 'sky', 'b': 'blue'})
_reg('towel_hanging', '''
    oooooooo
    oHBBBBbo
    oBBBBBbo
    oWWWWWWo
    oBBBBBbo
    oBBBBBbo
    oBBBBBbo
    oWWWWWWo
    oBBBBBbo
    oBBBBbbo
    owowowoo
''', {'W': 'paper', 'B': 'sky', 'b': 'blue', 'H': 'light_sky'})


def draw_towel(c, x, y, kind='folded'):
    """Sky-blue gym towel with white stripes: 'folded' (12x7 stack) or 'hanging'
    (8x11, hung from a rail/hook whose row is y-11)."""
    _draw(c, 'towel_' + kind, x, y)


_reg('water_bottle', '''
    .ooo.
    .oko.
    ooooo
    oHBbo
    oHBbo
    oWWso
    oWWso
    oHBbo
    oHBbo
    oHBbo
    .ooo.
''', {'k': 'steel_dk', 'H': 'ice', 'B': 'turquoise', 'b': 'aqua', 'W': 'paper', 's': 'mist'})


def draw_water_bottle(c, x, y, color=None):
    """Turquoise sports water bottle (5x11) with a white label band. color: optional
    body colour to recolour it."""
    if color is None:
        _draw(c, 'water_bottle', x, y)
        return
    _put(c, _recol('water_bottle', col(color)), x, y)


@lru_cache(maxsize=None)
def _recol(name, rgb):
    a = _rgba(name).copy()
    for src, k in (('turquoise', 1.0), ('ice', 1.35), ('aqua', 0.8)):
        m = (a[:, :, :3] == np.array(col(src), np.uint8)).all(axis=2) & (a[:, :, 3] > 0)
        a[m, :3] = shade(rgb, k)
    a.setflags(write=False)
    return a


_reg('dumbbell_side', '''
    .oo......oo.
    oPPo....oPPo
    oPpoooooopPo
    oPpSSSSSSpPo
    oPpoooooopPo
    opPo....opPo
    .oo......oo.
''', {'P': 'steel_dk', 'p': 'charcoal', 'S': 'steel_hi'})
_reg('dumbbell_end', '''
    .oooo.
    oPPPpo
    oPSSpo
    oPSSpo
    opppno
    .oooo.
''', {'P': 'steel_dk', 'p': 'charcoal', 'S': 'steel_hi', 'n': 'night'})
_reg('kettlebell', '''
    ..ooooo..
    .o.....o.
    .o.....o.
    .ooooooo.
    oHPPPPPpo
    oPPPPPPpo
    oPPPPPppo
    oPPPPPppo
    .opppppo.
    ..ooooo..
''', {'P': 'steel_dk', 'p': 'charcoal', 'H': 'steel'})


def draw_dumbbell(c, x, y, view='side'):
    """Hex dumbbell: view 'side' (12x7, lying / seen lengthwise) or 'end' (6x6 — how it
    looks in the hand of a side-view hero doing curls)."""
    _draw(c, 'dumbbell_' + view, x, y)


def draw_kettlebell(c, x, y):
    """Cast-iron kettlebell (9x10)."""
    _draw(c, 'kettlebell', x, y)


# ---- shopping trolley (side view, handle on the LEFT so it is pushed to the right)
_TROLLEY_W, _TROLLEY_H = 26, 21
_TR_SLOTS = [(9, 0), (14, 0), (19, 0), (11, -3), (17, -3), (14, -6)]


@lru_cache(maxsize=None)
def _trolley_layers():
    back = Canvas(_TROLLEY_W, _TROLLEY_H, color=(255, 0, 255))
    front = Canvas(_TROLLEY_W, _TROLLEY_H, color=(255, 0, 255))
    # basket: top rim y=3, bottom y=12; x from 5 (back) to 24 (front, slanted)
    for yy in range(3, 13):
        xl = 5 + (yy - 3) // 4
        xr = 24 - (yy - 3) // 3
        back.hline(xl, xr, yy, 'steel_dk')
        if yy % 3 == 0:
            front.hline(xl, xr, yy, 'steel')
        for xx in range(xl, xr + 1):
            if (xx - xl) % 3 == 0:
                front.px(xx, yy, 'steel')
        front.px(xl, yy, 'stone')
        front.px(xr, yy, 'stone')
    front.hline(5, 24, 3, 'steel_hi')
    front.hline(7, 21, 12, 'stone')
    # handle: from the rim back up-left, red grip
    front.line(5, 3, 2, 0, 'stone')
    front.hline(0, 3, 0, 'red')
    front.px(0, 0, 'brick')
    front.hline(0, 3, 1, 'brick')
    # chassis + wheels
    front.line(7, 13, 5, 17, 'steel')
    front.line(20, 13, 21, 17, 'steel')
    front.hline(4, 22, 17, 'steel')
    front.hline(4, 22, 18, 'steel_dk')
    for wx in (5, 20):
        front.rect(wx, 19, 3, 2, 'ink')
        front.px(wx + 1, 19, 'grey')
    out = []
    for cv in (back, front):
        a = np.zeros((_TROLLEY_H, _TROLLEY_W, 4), np.uint8)
        a[:, :, :3] = cv.a
        a[:, :, 3] = np.where((cv.a == np.array([255, 0, 255])).all(axis=2), 0, 255)
        a[(cv.a == np.array([255, 0, 255])).all(axis=2)] = 0
        a.setflags(write=False)
        out.append(a)
    # the back panel is see-through (blend)
    b = out[0].copy()
    b[b[:, :, 3] > 0, 3] = 110
    b.setflags(write=False)
    return b, out[1]


def trolley_slot(i):
    """(dx, dy) landing point (bottom-centre for draw_food) of item i (0..5) inside the
    trolley, relative to the draw_trolley anchor. Items stack: 3 on the floor, 2 above,
    1 on top."""
    sx, sy = _TR_SLOTS[i % len(_TR_SLOTS)]
    return (sx - _TROLLEY_W // 2, -(_TROLLEY_H - 12) + sy)


def draw_trolley(c, x, y, items=(), f=0, moving=False, flip=False):
    """Supermarket trolley (26x21), side view, red handle on the left (push it to the
    right; flip=True for the other way). items: FOOD names placed in trolley_slot order
    (draw a hopping item yourself until it lands, then add it here). moving=True makes
    the wheels' hubs spin."""
    back, front = _trolley_layers()
    if flip:
        back, front = back[:, ::-1], front[:, ::-1]
    x0, y0 = x - _TROLLEY_W // 2, y - _TROLLEY_H
    _blit(c, back, x0, y0)
    for i, n in enumerate(items[:len(_TR_SLOTS)]):
        dx, dy = trolley_slot(i)
        if flip:
            dx = -dx
        draw_food(c, x + dx, y + dy, n)
    _blit(c, front, x0, y0)
    if moving:
        for wx in (5, 20):
            xx = (x0 + _TROLLEY_W - 1 - wx - 2) if flip else x0 + wx
            c.px(xx + (1 + (f // 3) % 2), y0 + 19 + (f // 3) % 2, 'stone')


# ---- books & clothes (night tidy-up)
@lru_cache(maxsize=None)
def _book_rgba(color, orient):
    b = col(color)
    hi, sh = mix(b, 'white', 0.35), shade(b, 0.7)
    if orient == 'stand':
        art = ['ooooo', 'oHBSo', 'oHBSo', 'oWWWo', 'oHBSo', 'oHBSo', 'oHBSo',
               'oWWWo', 'oHBSo', 'ooooo']
    elif orient == 'flat':
        art = ['oooooooooo', 'oHHHHHHHHo', 'oPPPPPPPPo', 'oSSSSSSSSo', 'oooooooooo']
    else:  # open
        art = ['.oooo.oooo.', 'oPPPPoPPPPo', 'oPllPoPllPo', 'oPPPPoPPPPo',
               'oPllPoPllPo', 'oBBBBBBBBBo', '.ooooooooo.']
    a = _parse('\n'.join(art), {'H': hi, 'B': b, 'S': sh, 'W': 'gold', 'P': 'paper',
                                'l': 'stone'})
    a.setflags(write=False)
    return a


def draw_book(c, x, y, color='blue', orient='stand'):
    """A book: orient 'stand' (5x10, spine out, on a shelf), 'flat' (10x5, lying) or
    'open' (11x7). color: cover colour."""
    _put(c, _book_rgba(col(color), orient), x, y)


_CLOTHES = {
    'hoodie': ('''
        ...oooo....
        ..oTTTToo..
        .oTTtTTTTo.
        oTTTTtTTTto
        otttTTTtTto
        .oooooooooo
    ''', {'T': 'teal', 't': 'deep_teal'}),
    'shirt': ('''
        ..oooooo..
        .oWWWWWWo.
        oWWWsWWWWo
        osWWWWsWWo
        .oooooooo.
    ''', {'W': 'paper', 's': 'mist'}),
    'sock': ('''
        ooo..
        oWoo.
        oRWWo
        .ooo.
    ''', {'W': 'paper', 'R': 'red'}),
    'jeans': ('''
        .ooooooo.
        oPPPPPPPo
        oPpPPPPpo
        opppPPppo
        .ooooooo.
    ''', {'P': 'indigo', 'p': 'navy'}),
    'folded': ('''
        oooooooooo
        oTTTTTTTto
        oooooooooo
        oWWWWWWWso
        oooooooooo
        oPPPPPPPpo
        oooooooooo
    ''', {'T': 'teal', 't': 'deep_teal', 'W': 'paper', 's': 'mist', 'P': 'indigo', 'p': 'navy'}),
}
for _k, (_a, _m) in _CLOTHES.items():
    _reg('clothes_' + _k, _a, _m)


def draw_clothes(c, x, y, kind='hoodie', flip=False):
    """Crumpled clothes for the tidy-up: 'hoodie' (teal, 11x6), 'shirt' (10x5), 'sock'
    (5x4), 'jeans' (9x5), 'folded' (neat stack, 10x7)."""
    _draw(c, 'clothes_' + kind, x, y, flip)


def draw_receipt(c, x, y, length=12, f=0):
    """Till receipt 7 px wide growing UP from (x, y) (e.g. out of a register slot):
    length in px (animate it), grey item lines, a bold total and a zig-zag torn top."""
    length = int(max(0, length))
    if length <= 0:
        return
    x0 = x - 3
    top = y - length
    c.rect(x0, top, 7, length, 'paper')
    c.vline(x0 + 6, top, y - 1, 'mist')
    for i in range(length - 2):
        yy = y - 2 - i
        if i % 2 == 0:
            ln = 2 + int(hash01(i, 4) * 3)
            c.hline(x0 + 1, x0 + ln, yy, 'stone')
            c.px(x0 + 5, yy, 'grey')
    if length >= 6:
        c.hline(x0 + 1, x0 + 5, top + 2, 'ink')
    for xx in range(7):
        c.px(x0 + xx, top + (xx % 2), 'paper' if xx % 2 else (0, 0, 0, 0) and 'paper')
        if xx % 2 == 0:
            c.px(x0 + xx, top, 'paper')
        else:
            pass


def draw_shopping_list(c, x, y, items=('apple', 'broccoli', 'kiwifruit', 'milk'), checked=0,
                       f=0, t_check=None):
    """Shopping-list card, TOP-LEFT anchored (UI-like; put it in a bubble or on a fridge):
    one row (10 px) per FOOD item with a checkbox; the first `checked` items are ticked
    (green tick, item greyed). t_check: frame of the latest tick, makes it pop."""
    n = len(items)
    w, h = 22, 3 + n * 10
    c.rect(x, y, w, h, 'paper')
    c.frame(x - 1, y - 1, w + 2, h + 2, 'ink')
    c.rect(x, y, w, 2, 'sky')
    for i, it in enumerate(items):
        ry = y + 3 + i * 10
        c.frame(x + 2, ry + 2, 5, 5, 'grey')
        draw_food(c, x + 14, ry + 9, it)
        if i < checked:
            pop = 0
            if t_check is not None and i == checked - 1:
                pop = 1 if 0 <= f - t_check < 4 else 0
            for (dx, dy) in ((0, 2), (1, 3), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)):
                c.px(x + 2 + dx - pop, ry + 1 + dy - pop, 'green')
            c.hline(x + 9, x + 19, ry + 5, 'red')
        if i < n - 1:
            c.hline(x + 1, x + w - 2, ry + 9, 'mist')


@lru_cache(maxsize=None)
def _plant_base(kind):
    pot = '''
        .oooooooo.
        oCCCCCCCCo
        .oKKKKKko.
        .oKKKKKko.
        ..oKKKko..
        ..oooooo..
    '''
    a = _parse(pot, {'C': 'clay', 'K': 'copper', 'k': 'rust'})
    a.setflags(write=False)
    return a


def draw_plant(c, x, y, kind='leafy', f=0):
    """Potted plant on a terracotta pot (10x6 pot). kind 'leafy' (round leaves, ~12x13),
    'daffodil' (spring! yellow trumpets), 'succulent', 'herb'. Leaves sway with f."""
    pot = _plant_base(kind)
    x0, y0 = x - 5, y - 6
    sw = [0, 0, 1, 1, 0, 0, -1, -1][(f // 9) % 8]
    if kind == 'leafy':
        for (lx, ly, sx) in ((-4, -6, 0), (3, -7, 1), (-2, -10, 1), (2, -11, 0), (0, -4, 0)):
            d = sw if ly < -8 else 0
            xx, yy = x + lx + d, y0 + ly
            c.line(x, y0, xx, yy + 2, 'forest')
            c.round_rect(xx - 2, yy - 1, 5, 4, 'ink', r=1)
            c.rect(xx - 1, yy, 3, 2, 'green')
            c.px(xx - 1, yy, 'leaf')
            c.px(xx + 1, yy + 1, 'forest')
    elif kind == 'daffodil':
        for (lx, h_) in ((-3, 9), (0, 12), (3, 8)):
            d = sw if h_ > 9 else 0
            c.line(x + lx // 2, y0, x + lx + d, y0 - h_ + 2, 'green')
            fx_, fy_ = x + lx + d, y0 - h_
            for (dx, dy) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c.px(fx_ + dx, fy_ + dy, 'butter')
            c.px(fx_, fy_, 'amber')
            c.px(fx_ + 1, fy_ + 1, 'gold')
        c.line(x - 2, y0, x - 4, y0 - 5, 'leaf')
        c.line(x + 2, y0, x + 4 + sw, y0 - 6, 'leaf')
    elif kind == 'succulent':
        for (dx, dy, cc) in ((-3, -1, 'sage'), (3, -1, 'sage'), (-2, -3, 'pale_sage'),
                             (2, -3, 'pale_sage'), (0, -4, 'khaki'), (0, -2, 'sage')):
            c.round_rect(x + dx - 1, y0 + dy - 1, 3, 3, cc, r=1)
        c.px(x, y0 - 5, 'pink')
    else:  # herb
        for i in range(-3, 4):
            h_ = 4 + (3 - abs(i))
            d = sw if h_ > 5 else 0
            c.line(x + i, y0, x + i + d, y0 - h_, 'green')
            c.px(x + i + d, y0 - h_, 'leaf')
            c.px(x + i + d - 1, y0 - h_ + 1, 'leaf')
    _blit(c, pot, x0, y0)


# =====================================================================================
# ICONS — 9x9 pixel icons for speech / thought bubbles. TOP-LEFT anchored.
# =====================================================================================
_ICON_ART = {
    'atom': ('''
        .bb...bb.
        b..b.b..b
        b...b...b
        .b.b.b.b.
        ..bbRbb..
        .b.b.b.b.
        b...b...y
        b..b.b..b
        .bb...bb.
    ''', {'b': 'blue', 'R': 'red', 'y': 'gold'}),
    'heart': ('''
        .........
        .ooo.ooo.
        oHHRoRRRo
        oHRRRRRro
        oRRRRRRro
        .oRRRRro.
        ..oRRro..
        ...oro...
        ....o....
    ''', {'H': 'pink', 'R': 'red', 'r': 'crimson'}),
    'bulb': ('''
        ..ooooo..
        .oYYYYGo.
        oYWYYYYGo
        oYWYYYYGo
        oYYYYYGGo
        .oYYYGGo.
        ..ossso..
        ..oSSSo..
        ...ooo...
    ''', {'Y': 'butter', 'G': 'gold', 'W': 'white', 's': 'stone', 'S': 'grey'}),
    'note': ('''
        ....oo...
        ....oVo..
        ....o.Vo.
        ....o..o.
        ....o....
        .oooo....
        oLVVo....
        oVVVo....
        .ooo.....
    ''', {'V': 'violet', 'L': 'lavender'}),
    'notes': ('''
        ..ooooooo
        ..oVVVVVo
        ..ooooooo
        ..o....o.
        ..o....o.
        oooo.oooo
        LVVooLVVo
        VVVo.VVVo
        ooo..ooo.
    ''', {'V': 'violet', 'L': 'lavender'}),
    'coffee': ('''
        ..s..s...
        ...s..s..
        ..s..s...
        .oooooo..
        .oBBBBooo
        .oWWWWo.o
        .oWWWsooo
        ..oWso...
        .oooooo..
    ''', {'s': 'stone', 'B': 'brew_mid', 'W': 'paper'}),
    'brew': ('''
        ......pp.
        .ooooopo.
        .oGIGBpo.
        .oIIBBpo.
        .oBBBBpo.
        .oBMBBBo.
        .oBBBBBo.
        .oBBBBBo.
        ..ooooo..
    ''', {'G': 'glass_hi', 'I': 'glass', 'B': 'brew', 'M': 'brew_mid', 'p': 'hot_pink'}),
    'question': ('''
        ..oooo...
        .oBBBBo..
        oBBooBBo.
        .oo.oBBo.
        ...oBBo..
        ...oBo...
        ...ooo...
        ...oBo...
        ...ooo...
    ''', {'B': 'blue'}),
    'exclaim': ('''
        ...oooo..
        ...oRRo..
        ...oRRo..
        ...oRRo..
        ...oRro..
        ...oooo..
        ...oooo..
        ...oRro..
        ...oooo..
    ''', {'R': 'red', 'r': 'brick'}),
    'graph': ('''
        o.......B
        o......B.
        o....RB..
        o....B...
        o..RB....
        o..B.....
        oRB......
        oB.......
        ooooooooo
    ''', {'B': 'blue', 'R': 'red'}),
    'gauss': ('''
        o........
        o...RR...
        o...RR...
        o..R..R..
        o..R..R..
        o.R....R.
        o.R....R.
        oR......R
        ooooooooo
    ''', {'R': 'red'}),
    'laser': ('''
        .........
        ........p
        oooooo.pR
        oSSSSo.RR
        oSyySRRRR
        oSSSSo.RR
        oooooo.pR
        .o..o...p
        .........
    ''', {'S': 'steel', 'y': 'gold', 'R': 'laser_red', 'p': 'pink'}),
    'levels': ('''
        ooooooooo
        ....R....
        ...RRR...
        ....R....
        ...R.....
        ....R....
        ...R.....
        ..b.R....
        ooooooooo
    ''', {'R': 'laser_red', 'b': 'blue'}),
    'star': ('''
        ....o....
        ...oYo...
        ...oYo...
        ooooYYooo
        oYYYYYGGo
        .oGYYGGo.
        ..oYGGo..
        .oGGoGGo.
        .ooo.ooo.
    ''', {'Y': 'butter', 'G': 'gold'}),
    'check': ('''
        .......oo
        ......oGo
        .....oGGo
        oo...oGo.
        oGo.oGGo.
        oGGoGGo..
        .oGGGo...
        ..oGo....
        ...o.....
    ''', {'G': 'mint'}),
    'zzz': ('''
        ....BBBBB
        .......B.
        ......B..
        .....B...
        BBB.BBBBB
        ..B......
        .B.......
        BBB......
        .........
    ''', {'B': 'blue'}),
    'moon': ('''
        ...ooo...
        .ooYYo...
        oYYYo....
        oYYo.....
        oYYo.....
        oGYYo....
        .oGYYooo.
        ..oGGGGo.
        ...oooo..
    ''', {'Y': 'butter', 'G': 'gold'}),
    'sun': ('''
        ....a....
        .a.....a.
        ...ooo...
        ..oYYGo..
        aaoYGGoaa
        ..oGGAo..
        ...ooo...
        .a.....a.
        ....a....
    ''', {'Y': 'butter', 'G': 'gold', 'A': 'amber', 'a': 'amber'}),
    'cart': ('''
        oo.......
        .o.......
        .oooooooo
        .oRRRRRRo
        .oRRRRRo.
        .oooooo..
        .o.......
        .ooooooo.
        ..o...o..
    ''', {'R': 'red'}),
    'dumbbell': ('''
        .........
        .oo...oo.
        oPpo.oPpo
        oPpoooPpo
        oPpSSSPpo
        oPpoooPpo
        oPpo.oPpo
        .oo...oo.
        .........
    ''', {'P': 'steel', 'p': 'steel_dk', 'S': 'steel_hi'}),
    'book': ('''
        .........
        .oo...oo.
        oWWoooWWo
        oWlWoWlWo
        oWWWoWWWo
        oWlWoWlWo
        oWWWoWWWo
        oBBBBBBBo
        .ooooooo.
    ''', {'W': 'paper', 'l': 'stone', 'B': 'blue'}),
    'sparkle': ('''
        ....o....
        ...oYo...
        ...oYo...
        .ooYWYoo.
        oYYWWWYYo
        .ooYWYoo.
        ...oYo...
        ...oYo...
        ....o....
    ''', {'Y': 'gold', 'W': 'butter'}),
    'fork': ('''
        .s.s.....
        .s.s..k..
        .sss.kk..
        ..s..kk..
        ..s..kk..
        ..s...k..
        ..s...k..
        ..g...g..
        ..g...g..
    ''', {'s': 'steel', 'k': 'steel', 'g': 'grey'}),
    'kiwi': ('''
        .........
        ..oooo...
        .oHHBBo..
        oHHBBeBo.
        oHBBBBBbb
        oBBBBBSo.b
        .oSSSSo..
        ..l..l...
        .ll.ll...
    ''', {'H': 'crema', 'B': 'latte', 'S': 'brew_mid', 'e': 'ink', 'b': 'tan', 'l': 'mauve'}),
    'equation': ('''
        .........
        B..B..B..
        B..B..B..
        B..B..B..
        .B.B.B...
        ..BBB....
        ...B.....
        ...B.....
        ..BBB....
    ''', {'B': 'indigo'}),
    'robot': ('''
        ....y....
        ....o....
        .ooooooo.
        oSSSSSSSo
        oSCSSSCSo
        oSSSSSSSo
        oSSkkkSSo
        oSSSSSSso
        .ooooooo.
    ''', {'y': 'red', 'S': 'steel_hi', 's': 'steel', 'C': 'turquoise', 'k': 'grey'}),
    'rocket': ('''
        ....o....
        ...oWo...
        ...oWo...
        ..oWBWo..
        ..oWWso..
        .ooWWsoo.
        oRoWWsoRo
        oo.oAo.oo
        ....Y....
    ''', {'W': 'paper', 's': 'mist', 'B': 'sky', 'R': 'red', 'A': 'amber', 'Y': 'gold'}),
    'cat': ('''
        o.......o
        oOo...oOo
        oOOoooOOo
        oOOOOOOOo
        oOkOOOkOo
        oOOOpOOOo
        .oOOOOOo.
        ..ooooo..
        .........
    ''', {'O': 'clay', 'k': 'ink', 'p': 'pink'}),
    'flower': ('''
        ..oo.oo..
        .oPPoPPo.
        oPPpYpPPo
        .ooYGYoo.
        oPPpYpPPo
        .oPPoPPo.
        ..oogoo..
        ...lg....
        ....g....
    ''', {'P': 'pink', 'p': 'rose', 'Y': 'butter', 'G': 'gold', 'g': 'green', 'l': 'leaf'}),
    'plane': ('''
        ........o
        ......ooW
        ....ooWWo
        ..ooWWWo.
        ooWWWWso.
        ..oosWso.
        ....oso..
        .....oo..
        .........
    ''', {'W': 'paper', 's': 'mist'}),
    'camera': ('''
        .........
        ..ooo....
        ooSSSoooo
        oSSoooSRo
        oSoBBBoSo
        oSoBwBoSo
        oSSoooSSo
        ooooooooo
        .........
    ''', {'S': 'steel', 'R': 'red', 'B': 'blue', 'w': 'white'}),
    'list': ('''
        ...ooo...
        .ooOOOoo.
        oWWoooWWo
        oWgWlllWo
        oWWWWWWWo
        oWgWlllWo
        oWWWWWWWo
        oWgWllWWo
        ooooooooo
    ''', {'W': 'paper', 'O': 'steel', 'g': 'mint', 'l': 'stone'}),
    'snowflake': ('''
        ....B....
        .B..B..B.
        ..B.B.B..
        ...BWB...
        BBBWWWBBB
        ...BWB...
        ..B.B.B..
        .B..B..B.
        ....B....
    ''', {'B': 'sky', 'W': 'light_sky'}),
    'clock': ('''
        ..ooooo..
        .oWWWWWo.
        oWWWkWWWo
        oWWWkWWWo
        oWWWkkkWo
        oWWWWWWWo
        oWWWWWWso
        .oWWWWso.
        ..ooooo..
    ''', {'W': 'paper', 'k': 'ink', 's': 'mist'}),
    'battery': ('''
        .........
        .........
        oooooooo.
        oRWWWWWoo
        oRWWWWWoo
        oRWWWWWoo
        oooooooo.
        .........
        .........
    ''', {'R': 'red', 'W': 'paper'}),
    'battery_full': ('''
        .........
        .........
        oooooooo.
        oGGGGGgoo
        oGGGGGgoo
        oGGGGGgoo
        oooooooo.
        .........
        .........
    ''', {'G': 'leaf', 'g': 'mint'}),
    'lunchbox': ('''
        .........
        .ooooooo.
        oIIIcIIIo
        oAAAcAAAo
        ooooooooo
        oTTTcTTTo
        oTTTcTTTo
        oaaacaaao
        .ooooooo.
    ''', {'I': 'ice', 'A': 'aqua', 'T': 'turquoise', 'a': 'teal', 'c': 'coral'}),
    'home': ('''
        ....o....
        ...oRo...
        ..oRRRo..
        .oRRRRRo.
        ooWWWWWoo
        .oWYoWWo.
        .oWWoWWo.
        .oWWoWWo.
        .ooooooo.
    ''', {'R': 'red', 'W': 'paper', 'Y': 'gold'}),
    'magnet': ('''
        .ooo.ooo.
        oRRo.oRRo
        oRRo.oRRo
        oRRo.oRRo
        oRRoooRRo
        oRRRRRRRo
        .oRRRRRo.
        oWo...oWo
        ooo...ooo
    ''', {'R': 'red', 'W': 'steel_hi'}),
    'thumbs': ('''
        ...oo....
        ..oSo....
        ..oSo....
        .oSSoooo.
        oSSSSSSSo
        oSSSoooSo
        oSSSSSSso
        oSSSooSso
        .ooooooo.
    ''', {'S': 'skin', 's': 'skin_sh'}),
    'galaxy': ('''
        ...LLL...
        .LL...L..
        L..vvv.L.
        L.v.W.v.L
        L.v.Wv..L
        .L.v...L.
        ..L.vvL..
        ...L..L..
        ....LL...
    ''', {'L': 'lavender', 'v': 'violet', 'W': 'white'}),
    'egg': ('''
        .........
        ..ooooo..
        .oWWWWWo.
        oWWoooWWo
        oWoYYGoWo
        oWoYGGoWo
        oWWoooWso
        .oWWWsso.
        ..ooooo..
    ''', {'W': 'white', 's': 'mist', 'Y': 'butter', 'G': 'gold'}),
    'chat': ('''
        .........
        .ooooooo.
        oWWWWWWWo
        oWkWkWkWo
        oWWWWWWWo
        .oooWooo.
        ...oWo...
        ...oo....
        .........
    ''', {'W': 'paper', 'k': 'grey'}),
    'phone': ('''
        ..ooooo..
        ..okkko..
        ..oSSSo..
        ..oSsSo..
        ..oSSSo..
        ..osSso..
        ..oSSSo..
        ..okkko..
        ..ooooo..
    ''', {'k': 'night', 'S': 'sky', 's': 'light_sky'}),
    'bed': ('''
        .........
        o........
        o........
        oWWo.....
        oWWBBBBBo
        ooooooooo
        oBBBBBBBo
        o.......o
        .........
    ''', {'W': 'paper', 'B': 'indigo'}),
}
ICONS = list(_ICON_ART.keys())
for _k, (_a, _m) in _ICON_ART.items():
    _reg('icon_' + _k, _a, _m)
_reg('icon_wave', '''
    .........
    .........
    .BB....BB
    B..B..B..
    B..B..B..
    ....BB...
    .........
    .........
    .........
''', {'B': 'sky'})
_reg('icon_dna', '''
    R.....B..
    .R...B...
    ..RkkB...
    ...BR....
    ..B..R...
    .BkkkkR..
    B......R.
    .B....R..
    ..BkkR...
''', {'R': 'red', 'B': 'blue', 'k': 'stone'})
ICONS += ['wave', 'dna']


def draw_icon(c, x, y, name):
    """9x9 icon with its TOP-LEFT at (x, y) (fits an 11x11 bubble: draw it at bubble+1).
    Names: ICONS."""
    _blit(c, _rgba('icon_' + name), x, y)


def icon_bubble(c, x, y, name, f=0, t0=None, tail=None, thought=False, fill='white'):
    """Speech (or thought) bubble with one icon: 11x11 inside, TOP-LEFT (x, y).
    tail = (tx, ty) point at the speaker's head. t0: frame the bubble appears — it pops
    in over 4 frames (dot -> small -> full with 1 px overshoot)."""
    from . import fx
    if t0 is not None:
        dt = f - t0
        if dt < 0:
            return
        if dt < 2:
            c.round_rect(x + 3, y + 3, 5, 5, 'ink', r=1)
            c.rect(x + 4, y + 4, 3, 3, fill)
            return
        if dt < 4:
            fx.bubble(c, x - 1, y - 1, 13, 13, tail=tail, fill=fill, thought=thought)
            draw_icon(c, x + 2, y + 2, name)
            return
    fx.bubble(c, x, y, 11, 11, tail=tail, fill=fill, thought=thought)
    draw_icon(c, x + 1, y + 1, name)


# =====================================================================================
# FX HELPERS: float_text, sweat, speed lines, shake lines, growl, puff, burst, hearts,
# zzz, steam heart
# =====================================================================================
def _text_mask(s, font):
    from .font import _render_mask, FONTS
    if FONTS[font]['upper']:
        s = s.upper()
    return _render_mask(s, font, None)


def float_text(c, x, y, f, t0, text, color='gold', dur=36, rise=12, font='tiny',
               outline='ink', align='center'):
    """Game-style pop-up text ('+1 STR') that pops in at frame t0 at (x, y) (top-centre
    by default), rises `rise` px over `dur` frames and dissolves (Bayer dither) over the
    last 40%. Outlined for readability. No-op outside [t0, t0+dur)."""
    dt = f - t0
    if dt < 0 or dt >= dur:
        return
    t = dt / dur
    m = _text_mask(text, font)
    h, w = m.shape
    pop = 2 if dt < 2 else 1 if dt < 4 else 0
    yy = int(y - round(ease_out(t) * rise)) - pop
    xx = int(x - (w // 2 if align == 'center' else w if align == 'right' else 0))
    vis = 1.0 if t < 0.6 else 1.0 - (t - 0.6) / 0.4
    if outline is not None:
        g = np.zeros((h + 2, w + 2), bool)
        for dy in range(3):
            for dx in range(3):
                g[dy:dy + h, dx:dx + w] |= m
        _stamp_dither(c, g, xx - 1, yy - 1, outline, vis)
    _stamp_dither(c, m, xx, yy, color, vis)


def _stamp_dither(c, m, x, y, color, vis):
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(c.w, x + w), min(c.h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    if vis < 1.0:
        sub = sub & (bayer(y1 - y0, x1 - x0, x0, y0) < vis)
    c.a[y0:y1, x0:x1][sub] = col(color)


_reg('sweat', '''
    .o.
    oIo
    oWo
    .o.
''', {'I': 'light_sky', 'W': 'white', 'o': 'sky'})


def draw_sweat(c, x, y, f=0, t0=None, side='right'):
    """Sweat drop (3x4, bottom-centre at (x, y), e.g. the hero's temple). t0=None: a drop
    flies off in an arc once per beat; t0=frame: a single fling starting at t0;
    f=None: a static drop."""
    if f is None:
        _draw(c, 'sweat', x, y)
        return
    dt = (f % FRAMES_PER_BEAT) if t0 is None else f - t0
    if dt < 0 or dt > 14:
        return
    s = 1 if side == 'right' else -1
    px_ = x + s * int(dt * 0.6)
    py_ = y - int(dt * 1.2 - dt * dt * 0.09)
    _draw(c, 'sweat', px_, py_)


def speed_lines(c, x, y, w, h, f, direction='left', color='white', n=5, seed=0):
    """Animated horizontal speed streaks inside the box (x, y, w, h) trailing
    behind a dashing character (direction = the side the streaks trail to)."""
    for i in range(n):
        ry = y + int(hash01(seed, i) * h)
        ln = 4 + int(hash01(seed, i, 2) * 8)
        ph = (f * 3 + int(hash01(seed, i, 3) * w)) % (w + ln)
        if direction == 'left':
            x1 = x + w - ph
            c.hline(max(x, x1), min(x + w - 1, x1 + ln), ry, color)
        else:
            x0 = x + ph - ln
            c.hline(max(x, x0), min(x + w - 1, x0 + ln), ry, color)


def shake_lines(c, x, y, w, h, f, color='white'):
    """Vibration marks '( )' on both sides of a box (x, y, w, h) — ringing phone, a
    rattling lid... They jump between 2 and 3 px away every 2 frames."""
    d = 2 + (f // 2) % 2
    cy = y + h // 2
    for (sx, sgn) in ((x - d, -1), (x + w - 1 + d, 1)):
        c.px(sx, cy - 2, color)
        c.px(sx + sgn, cy - 1, color)
        c.px(sx + sgn, cy, color)
        c.px(sx, cy + 1, color)


def draw_growl(c, x, y, f, color='ink'):
    """Stomach-growl squiggles around the belly at (x, y): three little zig-zag
    marks that wobble (draw them for a beat or two)."""
    k = (f // 3) % 2
    for (dx, dy) in ((-6, -2), (5, -3), (4, 2)):
        for i in range(3):
            c.px(x + dx + i, y + dy + ((i + k) % 2), color)


def puff(c, x, y, f, t0, size=1, color='white', dur=18):
    """A dust/smoke puff centred on (x, y) that pops at t0, expands and dissolves
    (outfit change, landing dust, broom dust: size 1-3)."""
    dt = f - t0
    if dt < 0 or dt >= dur:
        return
    t = dt / dur
    r = 2 + int(ease_out(t) * 4 * size)
    shade_c = mix(color, 'stone', 0.45)
    for i in range(5 + 2 * size):
        a = hash01(t0, i) * 6.283
        d = r * (0.6 + 0.5 * hash01(t0, i, 2))
        px_, py_ = x + int(math.cos(a) * d), y + int(math.sin(a) * d * 0.7)
        rr = max(0, int(round((1 - t) * (1 + size) * (0.6 + 0.6 * hash01(t0, i, 3)))))
        if t < 0.75 or hash01(t0, i, dt) > (t - 0.75) * 4:
            c.circle(px_ + 1, py_ + 1, rr, shade_c)
            c.circle(px_, py_, rr, color)


def burst(c, x, y, f, t0, color='gold', n=8, r=10, dur=12):
    """Radial impact/eureka lines bursting out from (x, y) at t0 (lines fly outward
    and shrink)."""
    dt = f - t0
    if dt < 0 or dt >= dur:
        return
    t = dt / dur
    r0 = 2 + ease_out(t) * r
    ln = max(1, int((1 - t) * 4))
    for i in range(n):
        a = 2 * math.pi * i / n + 0.2
        ca, sa = math.cos(a), math.sin(a)
        c.line(x + ca * r0, y + sa * r0, x + ca * (r0 + ln), y + sa * (r0 + ln), color)


_reg('heart_small', '''
    oo.oo
    oHRro
    oRRro
    .oro.
    ..o..
''', {'H': 'pink', 'R': 'red', 'r': 'crimson', 'o': 'crimson'})
_reg('heart_tiny', '''
    R.R
    RRR
    .R.
''', {'R': 'hot_pink'})


def heart_pop(c, x, y, f, t0, dur=30, small=False):
    """A little heart that pops at (x, y) (bottom-centre) at t0, floats up ~8 px with a
    side-to-side sway and dissolves."""
    dt = f - t0
    if dt < 0 or dt >= dur:
        return
    t = dt / dur
    yy = y - int(ease_out(t) * 8)
    xx = x + int(round(math.sin(dt / 5) * 1.2))
    if t > 0.75 and (dt % 2):
        return
    _draw(c, 'heart_tiny' if small else 'heart_small', xx, yy)


def zzz(c, x, y, f, color='light_sky', period=54):
    """Sleeping 'z z Z' drifting up-right from (x, y) in a loop (3 letters staggered)."""
    for i in range(3):
        t = ((f + i * period / 3) % period) / period
        px_ = x + int(t * 8) + i
        py_ = y - int(t * 12)
        if t > 0.85 and f % 2:
            continue
        big = t > 0.45
        if big:
            c.hline(px_, px_ + 2, py_, color)
            c.px(px_ + 1, py_ + 1, color)
            c.hline(px_, px_ + 2, py_ + 2, color)
        else:
            c.hline(px_, px_ + 1, py_, color)
            c.hline(px_, px_ + 1, py_ + 1, color)


def draw_steam_heart(c, x, y, f, color='white', period=72):
    """Steam rising from a dish at (x, y) that curls into a little heart near the top
    of every bar (period frames), then fades."""
    from . import fx
    t = f % period
    fx.steam(c, x, y, f, color, n=2, height=8, spread=2, seed=21)
    if 30 <= t < 66:
        u = (t - 30) / 36
        hy = y - 9 - int(u * 4)
        pts = [(-1, 0), (1, 0), (-2, 1), (0, 1), (2, 1), (-1, 2), (1, 2), (0, 3)]
        if u < 0.2:
            pts = pts[-3:]
        for i, (dx, dy) in enumerate(pts):
            if u > 0.7 and hash01(i, t // 2) < (u - 0.7) * 3:
                continue
            c.px(x + dx, hy + dy, color)


# =====================================================================================
# KIWI VARIANTS: doodle (whiteboard), plush, sticker, fridge magnet
# =====================================================================================
_reg('kiwi_doodle', '''
    ....ooooo.........
    ..oo.....oo.......
    .o.........o......
    o.......o...o.....
    o............ooo..
    o...........o...oo
    .o.........o......
    ..oo.....oo.......
    ....ooooo.........
    .....o..o.........
    ....oo.oo.........
''')


@lru_cache(maxsize=None)
def _kiwi_doodle(rgb):
    return _tint(_rgba('kiwi_doodle'), rgb)


def draw_kiwi_doodle(c, x, y, color='ink'):
    """Marker-line kiwi doodle (18x11) for the whiteboard; bottom-centre of the body
    at (x, y) (the beak sticks out to the right)."""
    a = _kiwi_doodle(col(color))
    _blit(c, a, x - 6, y - a.shape[0])


_reg('kiwi_plush', '''
    ...ooooo.......
    ..oHHHHHoo.....
    .oHHHHHHBBo....
    oHHHHHBBBeBo...
    oHHBBBBBBBBooo.
    oHBBBBBBBBBpppo
    oBBBWBBBBSSoooo
    oBBBRBBBSSSo...
    .oSSSSSSSSo....
    ..oFFooFFo.....
    ...oo..oo......
''', {'H': 'blush', 'B': 'crema', 'S': 'latte', 'e': 'ink', 'p': 'peach', 'F': 'tan',
      'W': 'white', 'R': 'red'})


def draw_kiwi_plush(c, x, y, facing='right'):
    """Soft kiwi plushie (15x11) for a supermarket shelf / the bedroom: lighter felt
    colours, button eye, stubby beak, a little tag on its side."""
    a = _rgba('kiwi_plush', facing == 'left')
    _blit(c, a, x - 6 if facing == 'right' else x - a.shape[1] + 6, y - a.shape[0])


@lru_cache(maxsize=None)
def _kiwi_sticker(border, edge):
    a, ox, oy, _ = _kiwi('idle', 0, 1, False, 'color')
    m = a[:, :, 3] > 0
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    core = a[y0:y1, x0:x1]
    cm = core[:, :, 3] > 0
    h, w = cm.shape
    out = np.zeros((h + 4, w + 4, 4), np.uint8)
    pad = np.zeros((h + 4, w + 4), bool)
    pad[2:-2, 2:-2] = cm
    g = pad.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            g |= np.roll(np.roll(pad, dy, 0), dx, 1)
    out[g, :3] = col(border)
    out[g, 3] = 255
    e = np.roll(np.roll(g, 1, 0), 1, 1) & ~g
    out[e, :3] = col(edge)
    out[e, 3] = 255
    out[2:-2, 2:-2][cm] = core[cm]
    out.setflags(write=False)
    return out, ox - x0 + 2


def draw_kiwi_sticker(c, x, y, border='white', edge='stone'):
    """Die-cut kiwi sticker (~17x13): the 1x kiwi with a white border and a thin edge
    shadow — for the laser enclosure, a laptop lid, a water bottle."""
    a, cx = _kiwi_sticker(border, edge)
    _blit(c, a, x - cx, y - a.shape[0])


def draw_kiwi_magnet(c, x, y, f=0):
    """Fridge magnet: a little kiwi (the 9x9 kiwi icon) on a round leaf-green disc with
    a white rim (13x13). Bottom-centre anchored."""
    x0, y0 = x - 6, y - 13
    c.round_rect(x0, y0 + 1, 13, 12, 'ink', r=2)
    c.round_rect(x0 + 1, y0, 11, 12, 'ink', r=2)
    c.round_rect(x0 + 1, y0 + 1, 11, 11, 'white', r=2)
    c.round_rect(x0 + 2, y0 + 2, 9, 9, 'leaf', r=1)
    c.px(x0 + 3, y0 + 3, 'pale_leaf')
    c.rect(x0 + 2, y0 + 12, 11, 1, 'ink')
    _blit(c, _rgba('icon_kiwi'), x0 + 2, y0 + 2)
