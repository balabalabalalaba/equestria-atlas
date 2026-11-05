"""Postcard snapshots (P): the current view, framed like a picture postcard with the place's name and an
atlas stamp in the corner, saved as a PNG (window scale 4). Each one also goes into the game's photo book
(game/photos.py), which the journal's PHOTOS page shows.
"""
import itertools
import os
import time

import numpy as np

from engine import Canvas, text, text_width
from engine.loc_worldmap import LOCS

from . import ui
from .compose import compose, KEY, VIEW_W, VIEW_H

K = 4
BORDER, CAPTION = 8, 16
_COUNT = itertools.count(1)         # postcards taken this session (in the file name: none overwrite another)


def place_name(game):
    lid = getattr(game.mode, 'ID', None)
    return LOCS[lid]['name'] if lid in LOCS else 'Equestria'


def make(game):
    """The postcard as an RGB array (art px x K): the view without the HUD, a cream frame, a caption."""
    fr = game.render(0.0)
    fr.hud = Canvas(color=KEY)
    fr.iris = None
    fr.flash = 0.0
    view = compose(fr, K)
    pw, ph = VIEW_W + 2 * BORDER, VIEW_H + BORDER + CAPTION
    c = Canvas(pw, ph, color='cream')
    c.rect(1, 1, pw - 2, ph - 2, 'paper')
    for x in range(3, pw - 3, 4):                     # a dotted inner rule
        c.px(x, 3, 'bubblegum')
        c.px(x, ph - 4, 'bubblegum')
    for y in range(3, ph - 3, 4):
        c.px(3, y, 'bubblegum')
        c.px(pw - 4, y, 'bubblegum')
    name = place_name(game)
    sub = 'GREETINGS FROM'
    text(c, BORDER, VIEW_H + BORDER + 6, sub, 'clay', font='tiny')
    text(c, BORDER + text_width(sub, 'tiny') + 5, VIEW_H + BORDER + 4, name, 'wine', font='normal')
    ui._STAMP.draw(c, pw - BORDER - 10, VIEW_H + BORDER + 3)
    big = np.repeat(np.repeat(c.a, K, 0), K, 1)
    big[BORDER * K:BORDER * K + view.shape[0], BORDER * K:BORDER * K + view.shape[1]] = view
    return big


def save(game, folder):
    """Write the postcard, put it in the photo book and return its path."""
    from PIL import Image
    from .photos import slug, from_postcard
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'postcard_{slug(place_name(game))}_{int(time.time() * 1000)}_{next(_COUNT)}.png')
    img = make(game)
    Image.fromarray(img).save(path)
    lid = getattr(game.mode, 'ID', None)
    game.photos.add(path, lid if lid in LOCS else None, from_postcard(img), game.t)
    return path
