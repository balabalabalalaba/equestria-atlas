"""A plain 32 px mannequin for checking scale while the real character is built.

Location builders: blit this in YOUR preview scripts only (never in shipped
location code) to check that doors, counters, beds and chairs fit the hero.
"""
from .sprite import Sprite

DUMMY = Sprite.from_ascii('''
    ...oooooo...
    ..oxxxxxxo..
    .oxxxxxxxxo.
    .oxxxxxxxxo.
    .oxxxxxxxxo.
    .oxxxxxxxxo.
    .oxxxxxxxxo.
    .oxxxxxxxxo.
    ..oxxxxxxo..
    ...oxxxxo...
    ...oooooo...
    ..oyyyyyyo..
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    .oyyyyyyyyo.
    ..ozzzzzzo..
    ..ozzzzzzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    ..ozzoozzo..
    .ooooooooo..
    .ooooooooo..
''', {'o': 'ink', 'x': 'skin', 'y': 'teal', 'z': 'indigo'})


def draw_dummy(c, x, y, sitting=False):
    """Feet bottom-centre at (x, y) — i.e. y is the floor line. 32 px tall."""
    if sitting:
        top = DUMMY.sub(0, 0, 12, 21)
        c.blit(top, x - 6, y - 21 - 10)
    else:
        c.blit(DUMMY, x - 6, y - 32)
