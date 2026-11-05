"""Errand cargo: the things the explorer's balloon carries for atlas requests (game/quests.py).

Each item is a 7x7 pixel sprite (shown on the request board, the HUD plate and hanging under the
balloon's basket on a short rope) and a 4x3 tiny parcel for the small balloon sizes. No creatures:
goods only (fruit, parcels, lights, books...).
"""
from functools import lru_cache

import numpy as np

from engine import Sprite, col

_PAL = {'o': 'ink', 'w': 'white', 'r': 'red', 'R': 'crimson', 'y': 'gold', 'Y': 'butter', 'b': 'brew',
        'B': 'brew_dk', 't': 'tan', 'T': 'sand', 'p': 'hot_pink', 'P': 'pink', 'c': 'cream', 'g': 'leaf',
        'G': 'green', 's': 'sky', 'S': 'light_sky', 'v': 'violet', 'V': 'lilac', 'k': 'crystal',
        'K': 'crystal_lt', 'n': 'orange', 'm': 'magic', 'a': 'amber', 'l': 'clay', 'i': 'ice',
        'e': 'teal', 'E': 'mint', 'u': 'royal', 'U': 'royal_lt', 'q': 'cloud', 'h': 'brick'}

_ART = {
    'apples': '''
        .r.r.R.
        rYrRrrR
        brbrbbr
        ttttttt
        tbtbtbt
        ttttttt
        .bbbbb.''',
    'cupcakes': '''
        ...y...
        PPPyPPP
        PwPyPwP
        yyyyyyy
        PPPyPPP
        PwPyPwP
        ppppppp''',
    'scroll': '''
        TcccccT
        tcccccT
        .clllc.
        .ccccc.
        .clllc.
        TccrccT
        tcrRrcT''',
    'crystal': '''
        ..ooo..
        .oKKKo.
        oKkkkmo
        oKkWkmo
        oKkkkmo
        .ommmo.
        ..ooo..''',
    'rainbow': '''
        .ttttt.
        .iiiii.
        iriririi
        iYnYnYi
        igsgsgi
        isvsvsi
        .iiiii.''',
    'trophy': '''
        yyyyyyy
        yYYYYYy
        .yYYYy.
        ..yYy..
        ...y...
        ..aya..
        .aaaaa.''',
    'parcel': '''
        ttTTTtt
        tTTyTTt
        ttTyTtt
        yyyyyyy
        ttTyTtt
        ttTyTtt
        bbbybbb''',
    'cider': '''
        .bbbbb.
        bttttTb
        aaaaaaa
        bttttTb
        bttttTb
        aaaaaaa
        .bbbbb.''',
    'fireworks': '''
        r.s.y..
        rRsSyY.
        rRsSyY.
        hhhhhhh
        hThThTh
        hhhhhhh
        BBBBBBB''',
    'tea': '''
        ..eee..
        .EEEEE.
        eeeeeee
        eEcccEe
        eEcgcEe
        eEcccEe
        eeeeeee''',
    'gems': '''
        .k.p.y.
        tktptyt
        ttttttt
        tTTTTTt
        tTtttTt
        tTTTTTt
        .ttttt.''',
    'lantern': '''
        ...o...
        .ooooo.
        rnnYnnr
        rnYYYnr
        rnnYnnr
        .ooooo.
        ...y...''',
    'pearl': '''
        .PPPPP.
        PpPpPpP
        PPPPPPP
        .twWwt.
        twwWwwt
        tPPPPPt
        .ttttt.''',
    'book': '''
        uuuuuuo
        uUUUUuo
        uUyyUuo
        uUUUUuo
        uUUUUuo
        uuuuuuo
        cccccco''',
    'seeds': '''
        ..g.g..
        .gGgGg.
        ..pGp..
        ...G...
        lllllll
        .lhhhl.
        .lllll.''',
}

ITEM_NAMES = {'apples': 'APPLE BASKET', 'cupcakes': 'CUPCAKE BOX', 'scroll': 'ROYAL SCROLL',
              'crystal': 'CRYSTAL LIGHT', 'rainbow': 'RAINBOW JAR', 'trophy': 'GOLD TROPHY',
              'parcel': 'PARCEL', 'cider': 'CIDER BARREL', 'fireworks': 'FIREWORKS', 'tea': 'TEA CHEST',
              'gems': 'GEM SACK', 'lantern': 'PAPER LANTERNS', 'pearl': 'PEARL BOX', 'book': 'STAR ATLAS',
              'seeds': 'FLOWER SEEDS'}


def _fix(art):
    """Rows padded / trimmed to 7 columns (the sprites are 7x7)."""
    rows = [r.strip() for r in art.strip().splitlines()]
    return '\n'.join((r + '.......')[:7] for r in rows)


ITEMS = {k: Sprite.from_ascii(_fix(v), dict(_PAL, W='white')) for k, v in _ART.items()}
# the main colour of each item: the tiny parcel shown under small balloons
_TINY = {'apples': 'red', 'cupcakes': 'pink', 'scroll': 'cream', 'crystal': 'crystal', 'rainbow': 'sky',
         'trophy': 'gold', 'parcel': 'tan', 'cider': 'brew', 'fireworks': 'brick', 'tea': 'teal',
         'gems': 'crystal', 'lantern': 'orange', 'pearl': 'pink', 'book': 'royal', 'seeds': 'leaf'}


def icon(key):
    return ITEMS.get(key, ITEMS['parcel'])


@lru_cache(maxsize=None)
def cargo_rgba(key, size):
    """RGBA of what hangs under the basket: a rope and the item, top-centre = the basket's bottom
    centre. 'l' / 'm' balloons carry the 7x7 item on a 2 px rope; 's' / 'xs' a 4x3 tiny parcel."""
    if size in ('l', 'm'):
        spr = icon(key).rgba
        h, w = spr.shape[:2]
        out = np.zeros((h + 2, w, 4), np.uint8)
        out[2:] = spr
        out[0:2, w // 2] = (*col('brew'), 255)               # the rope
        return out
    out = np.zeros((4, 4, 4), np.uint8)
    out[1:4, 0:4] = (*col(_TINY.get(key, 'tan')), 255)
    out[1, 0:4] = (*col('ink'), 255)
    out[0, 1:3] = (*col('brew'), 255)
    return out
