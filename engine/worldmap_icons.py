"""Hand-designed 3/4 landmark icons for the world map (owned by the map builder).

icon(lid) -> Icon with .rgba (h, w, 4), .lights (h, w, 4: night-glow colours), .ax/.ay anchor
(the world point of the place sits at sprite pixel (ax, ay)), .grey (desaturated 'undiscovered').
Light from the top-left; outlines in a dark ink. Window pixels are recorded in .lights so the
renderer can light them at night. See docs/LOC_WORLDMAP.md for the description of each icon.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from .core import col, shade, mix

INK = '#2e222f'


class Ico:
    def __init__(self, w, h, ax=None, ay=None):
        self.a = np.zeros((h, w, 4), np.uint8)
        self.L = np.zeros((h, w, 4), np.uint8)
        self.w, self.h = w, h
        self.ax = w // 2 if ax is None else ax
        self.ay = h - 1 if ay is None else ay

    # -------- primitives
    def c(self, x, y, color):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.a[y, x, :3] = col(color)
            self.a[y, x, 3] = 255

    def rect(self, x, y, w, h, color):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x + w)), min(self.h, int(y + h))
        if x1 > x0 and y1 > y0:
            self.a[y0:y1, x0:x1, :3] = col(color)
            self.a[y0:y1, x0:x1, 3] = 255

    def hl(self, x0, x1, y, color):
        self.rect(min(x0, x1), y, abs(x1 - x0) + 1, 1, color)

    def vl(self, x, y0, y1, color):
        self.rect(x, min(y0, y1), 1, abs(y1 - y0) + 1, color)

    def poly(self, pts, color):
        im = Image.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=1, outline=1)
        m = np.asarray(im).astype(bool)
        self.a[m, :3] = col(color)
        self.a[m, 3] = 255

    def ell(self, cx, cy, rx, ry, color):
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        m = ((xx - cx) / (rx + 0.5)) ** 2 + ((yy - cy) / (ry + 0.5)) ** 2 <= 1
        self.a[m, :3] = col(color)
        self.a[m, 3] = 255

    def win(self, x, y, day='#3a4466', night='#ffd86a'):
        self.c(x, y, day)
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.L[y, x, :3] = col(night)
            self.L[y, x, 3] = 255

    def glow(self, x, y, color, night=None):
        self.c(x, y, color)
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.L[y, x, :3] = col(night or color)
            self.L[y, x, 3] = 255

    # -------- building blocks
    def house(self, x, yb, w, h, wall, roof, style='gable', rh=None, door=True, wins=True, trim=None):
        self.rect(x, yb - h, w, h, wall)
        self.vl(x + w - 1, yb - h, yb - 1, shade(wall, 0.78))
        if trim:
            self.hl(x, x + w - 1, yb - h, trim)
        rh = rh if rh is not None else max(2, (w + 1) // 2)
        top = yb - h
        if style == 'gable':
            for i in range(rh):
                y = top - 1 - i
                hw = (w + 1) / 2 - i * (w + 1) / 2 / rh
                x0 = int(round(x + (w - 1) / 2 - hw))
                x1 = int(round(x + (w - 1) / 2 + hw))
                mid = x + (w - 1) / 2
                for xx in range(x0, x1 + 1):
                    self.c(xx, y, roof if xx <= mid else shade(roof, 0.75))
            self.hl(x - 1, x + w, top - 1, shade(roof, 0.65))
        elif style == 'thatch':
            for i in range(rh):
                y = top - 1 - i
                k = i / max(1, rh - 1)
                hw = (w / 2 + 0.6) * (1 - k * k * 0.85)
                cx = x + (w - 1) / 2
                for xx in range(int(round(cx - hw)), int(round(cx + hw)) + 1):
                    c = roof
                    if xx > cx + hw * 0.3:
                        c = shade(roof, 0.78)
                    if i == 0:
                        c = shade(roof, 0.62)
                    elif xx < cx - hw * 0.3 and i > rh // 2:
                        c = shade(roof, 1.15)
                    self.c(xx, y, c)
        elif style == 'flat':
            self.hl(x - 1, x + w, top - 1, roof)
            self.hl(x - 1, x + w, top, shade(roof, 0.7))
        elif style == 'cone':
            self.cone(x + (w - 1) / 2, top - 1, (w + 1) / 2 + 0.5, rh, roof)
        if door and w >= 3 and h >= 3:
            dx = x + w // 2 - (1 if w % 2 == 0 else 0)
            self.c(dx, yb - 1, shade(wall, 0.45))
            if h >= 4:
                self.c(dx, yb - 2, shade(wall, 0.45))
        if wins and h >= 3:
            for wx in range(x + 1, x + w - 1, 3):
                if door and abs(wx - (x + w // 2)) < 1:
                    continue
                self.win(wx, yb - h + 1 + (h > 4))

    def cone(self, cx, ybase, hw, h, roof, tip=None):
        for i in range(int(h)):
            y = ybase - i
            r = hw * (1 - i / h)
            for xx in range(int(round(cx - r)), int(round(cx + r)) + 1):
                self.c(xx, y, roof if xx <= cx else shade(roof, 0.72))
        if tip:
            self.c(round(cx), ybase - int(h), tip)

    def tower(self, x, yb, w, h, wall, roof, rh, tip=None, trim=None, wins=True):
        self.rect(x, yb - h, w, h, wall)
        self.vl(x, yb - h, yb - 1, shade(wall, 1.08))
        self.vl(x + w - 1, yb - h, yb - 1, shade(wall, 0.75))
        if trim:
            self.hl(x, x + w - 1, yb - h, trim)
        self.cone(x + (w - 1) / 2, yb - h - 1, w / 2 + 0.6, rh, roof, tip)
        if wins:
            for wy in range(yb - h + 2, yb - 1, 3):
                self.win(x + w // 2, wy)

    def dome(self, cx, ybase, r, color):
        for i in range(r + 1):
            y = ybase - i
            hw = (r * r - i * i) ** 0.5
            for xx in range(int(round(cx - hw)), int(round(cx + hw)) + 1):
                c = color
                if xx > cx + hw * 0.25:
                    c = shade(color, 0.75)
                if xx < cx - hw * 0.2 and i > r * 0.4:
                    c = shade(color, 1.15)
                self.c(xx, y, c)

    def tree(self, x, yb, c1='#3f8d45', c2='#2c6b3d', r=1, fruit=None):
        self.vl(x, yb - 1, yb, '#6b4a36')
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r + 0.5:
                    self.c(x + dx, yb - 2 - r + dy, c1 if dx + dy <= 0 else c2)
        if fruit:
            self.c(x, yb - 2 - r, fruit)

    def falls(self, x, y0, y1, w, colors):
        for i in range(w):
            self.vl(x + i, y0, y1, colors[i % len(colors)])

    def flag(self, x, y, color):
        self.vl(x, y, y + 3, '#5a4a48')
        self.c(x + 1, y, color)
        self.c(x + 2, y, color)
        self.c(x + 1, y + 1, shade(color, 0.8))

    def star(self, cx, cy, color='#f9c22b', hi='#fbff86'):
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)):
            self.c(cx + dx, cy + dy, color)
        self.c(cx, cy, hi)

    def finish(self, outline=INK):
        """Pad by 1, outline the silhouette, and shift the anchor."""
        a = np.pad(self.a, ((1, 1), (1, 1), (0, 0)))
        L = np.pad(self.L, ((1, 1), (1, 1), (0, 0)))
        if outline:
            m = a[:, :, 3] > 0
            g = np.zeros_like(m)
            g[1:, :] |= m[:-1, :]
            g[:-1, :] |= m[1:, :]
            g[:, 1:] |= m[:, :-1]
            g[:, :-1] |= m[:, 1:]
            ring = g & ~m
            a[ring, :3] = col(outline)
            a[ring, 3] = 255
        return Icon(a, L, self.ax + 1, self.ay + 1)


class Icon:
    def __init__(self, rgba, lights, ax, ay):
        self.rgba = rgba
        self.lights = lights
        self.ax, self.ay = ax, ay
        self.h, self.w = rgba.shape[:2]
        a = rgba[:, :, :3].astype(np.float32)
        luma = a @ np.array([0.3, 0.55, 0.15], np.float32)
        g = (luma * 0.62 + 30)[..., None] * np.array([0.92, 0.95, 1.08], np.float32)
        grey = rgba.copy()
        grey[:, :, :3] = np.clip(g, 0, 255).astype(np.uint8)
        self.grey = grey
        white = rgba.copy()
        white[:, :, :3] = 255
        self.white = white


# ======================================================================== icon designs
def _canterlot():
    ic = Ico(46, 50, ax=23, ay=49)
    # rock ledges (violet-grey cliff) the city is built on
    ic.poly([(4, 38), (42, 34), (40, 44), (30, 49), (16, 49), (6, 45)], '#8f80b6')
    ic.poly([(24, 36), (42, 34), (40, 44), (30, 49), (24, 49)], '#71649a')
    ic.hl(6, 41, 37, '#ad9ed0')
    ic.hl(8, 38, 42, '#584e80')
    # waterfalls pouring from under the ledges
    ic.falls(12, 38, 48, 2, ('#d8f6ff', '#8fd3ff'))
    ic.falls(21, 38, 49, 2, ('#ffffff', '#8fd3ff'))
    ic.falls(31, 37, 46, 2, ('#d8f6ff', '#8fd3ff'))
    for x in (11, 13, 20, 23, 30, 33):
        ic.c(x, 48 if x < 25 else 46, '#ffffff')
    # outer wall
    ic.rect(6, 31, 34, 6, '#f4eff7')
    ic.rect(30, 31, 10, 6, '#d6cce4')
    ic.hl(6, 39, 31, '#f9c22b')
    for x in range(7, 39, 3):
        ic.win(x, 33)
    # buildings and towers (back to front)
    ic.tower(8, 31, 4, 9, '#f4eff7', '#5b3fa8', 6, tip='#f9c22b', trim='#f9c22b')
    ic.tower(33, 31, 4, 11, '#f4eff7', '#5b3fa8', 7, tip='#f9c22b', trim='#f9c22b')
    ic.house(13, 31, 6, 5, '#f4eff7', '#f9c22b', 'gable', rh=3, door=False)
    ic.house(27, 31, 5, 6, '#e6def0', '#f9c22b', 'gable', rh=3, door=False)
    ic.tower(16, 31, 5, 15, '#ffffff', '#f9c22b', 8, tip='#905ea9', trim='#f9c22b')
    ic.tower(24, 31, 4, 13, '#f4eff7', '#905ea9', 7, tip='#f9c22b', trim='#f9c22b')
    # the great central tower
    ic.tower(20, 31, 6, 21, '#ffffff', '#f9c22b', 10, tip='#fbff86', trim='#f9c22b')
    ic.c(22, 18, '#905ea9')
    ic.c(23, 18, '#905ea9')
    ic.flag(26, 2, '#905ea9')
    ic.flag(19, 6, '#f04f78')
    # stained-glass glints
    ic.c(22, 24, '#8fd8f0')
    ic.c(23, 26, '#f39ce8')
    return ic.finish('#3e365e')


def _cloudsdale():
    ic = Ico(50, 46, ax=25, ay=45)
    rain = ('#ff5a5a', '#ffa54a', '#ffe15a', '#6bd66b', '#5ab4ff', '#a275e6')
    for x0, y1 in ((8, 44), (20, 45), (34, 42)):
        for i, c in enumerate(rain):
            ic.vl(x0 + i, 24, y1 - (i % 2), c)
    # cloud base
    for cx, cy, rx, ry in ((12, 24, 10, 5), (25, 22, 13, 6), (38, 24, 10, 5), (25, 27, 18, 4)):
        ic.ell(cx, cy, rx, ry, '#e2eaff')
    for cx, cy, rx, ry in ((10, 22, 6, 3), (22, 19, 7, 3), (36, 22, 6, 3)):
        ic.ell(cx, cy, rx, ry, '#ffffff')
    for x in range(8, 44):
        ic.c(x, 29 if x % 3 else 30, '#b6c2e8')
    # colonnade + domes (Greek style)
    ic.rect(14, 12, 22, 2, '#ffffff')
    ic.rect(14, 19, 22, 1, '#cfd8f2')
    for x in range(15, 36, 3):
        ic.vl(x, 14, 18, '#ffffff')
        ic.vl(x + 1, 14, 18, '#b6c2e8')
    ic.poly([(13, 12), (25, 7), (37, 12)], '#f2f5ff')
    ic.poly([(25, 7), (37, 12), (25, 12)], '#cfd8f2')
    ic.dome(9, 18, 4, '#f6f8ff')
    ic.dome(41, 19, 4, '#f6f8ff')
    ic.vl(9, 11, 13, '#b6c2e8')
    ic.c(9, 10, '#f9c22b')
    # the colosseum ring (right)
    ic.rect(38, 19, 8, 3, '#e2eaff')
    for x in range(38, 46, 2):
        ic.c(x, 20, '#9aa8d8')
    ic.flag(44, 12, '#5ab4ff')
    ic.flag(6, 10, '#ff5a5a')
    return ic.finish('#6a74a8')


def _ponyville():
    ic = Ico(36, 24, ax=18, ay=23)
    walls = ('#f7d7e6', '#fbefd0', '#cdebc4', '#e4d4f4', '#fbe0c8', '#d8ecf8')
    th = '#e8c77a'
    spots = [(2, 14, 6, 4), (10, 12, 6, 5), (24, 13, 6, 4), (29, 17, 5, 4), (4, 21, 6, 4), (12, 22, 6, 4),
             (21, 22, 6, 4), (28, 23, 6, 4)]
    for i, (x, yb, w, h) in enumerate(spots[:3]):
        ic.house(x, yb, w, h, walls[i], th, 'thatch', rh=4)
    # town hall with spire (centre)
    ic.house(16, 16, 7, 6, '#fbefd0', '#c8313a', 'gable', rh=4, trim='#ffffff')
    ic.tower(18, 9, 3, 3, '#ffffff', '#c8313a', 4, tip='#f9c22b')
    for i, (x, yb, w, h) in enumerate(spots[3:]):
        ic.house(x, yb, w, h, walls[(i + 3) % len(walls)], th, 'thatch', rh=4)
    # sugarcube corner: gingerbread + cupcake roof (little pink dome)
    ic.dome(31, 13, 2, '#ff9fd0')
    ic.c(31, 10, '#e83b3b')
    return ic.finish('#5a3a2e')


def _castle_friendship():
    ic = Ico(20, 30, ax=10, ay=29)
    # roots / trunk
    ic.poly([(6, 29), (8, 18), (12, 18), (14, 29)], '#905ea9')
    ic.poly([(10, 18), (12, 18), (14, 29), (10, 29)], '#6b3e75')
    ic.c(4, 29, '#905ea9')
    ic.c(15, 29, '#6b3e75')
    ic.vl(9, 20, 27, '#c8a0e0')
    # branches holding turrets
    ic.poly([(2, 16), (10, 12), (18, 16), (10, 20)], '#905ea9')
    ic.poly([(10, 12), (18, 16), (10, 20)], '#6b3e75')
    for x, yb, h in ((2, 16, 4), (15, 16, 4), (8, 12, 6)):
        ic.rect(x, yb - h, 3 if x != 8 else 4, h, '#a884f3')
        ic.vl(x + (2 if x != 8 else 3), yb - h, yb - 1, '#6f6ac8')
        ic.cone(x + (1 if x != 8 else 1.5), yb - h - 1, 2.2 if x != 8 else 2.8, 4 if x != 8 else 5, '#30e1b9')
        ic.win(x + 1, yb - h + 1, night='#ffb0f0')
    ic.star(10, 3, '#f39ce8', '#ffffff')
    ic.win(10, 24, night='#ffd0f8')
    return ic.finish('#3e2858')


def _school():
    ic = Ico(18, 15, ax=9, ay=14)
    ic.house(2, 14, 14, 6, '#e4d4f4', '#6b3e75', 'gable', rh=4, trim='#f9c22b')
    ic.tower(1, 14, 3, 9, '#efe4fa', '#905ea9', 4, tip='#f9c22b', wins=False)
    ic.tower(14, 14, 3, 9, '#efe4fa', '#905ea9', 4, tip='#f9c22b', wins=False)
    ic.star(9, 5, '#f9c22b')
    return ic.finish('#3e2858')


def _sweet_apple():
    ic = Ico(28, 18, ax=14, ay=17)
    # farmhouse
    ic.house(16, 15, 7, 6, '#f4e2c8', '#7a5a8a', 'gable', rh=3, trim='#ffffff')
    # big red barn
    ic.house(4, 16, 10, 8, '#c8313a', '#8a2a2e', 'gable', rh=5, door=False, wins=False)
    ic.hl(4, 13, 8, '#ffffff')
    for i in range(4):
        ic.c(7 + i, 12 + i, '#ffffff')
        ic.c(10 - i, 12 + i, '#ffffff')
    ic.rect(7, 12, 4, 4, '#ffffff')
    ic.rect(8, 13, 2, 2, '#8a2a2e')
    ic.c(8, 9, '#3a2a2a')
    # orchard trees
    for x, yb in ((2, 17), (24, 16), (26, 12), (23, 9)):
        ic.tree(x, yb, '#43883e', '#2f6a32', 1, fruit='#e8323c')
    # fence
    for x in range(14, 27, 2):
        ic.c(x, 17, '#b98a55')
    return ic.finish('#3a2a2a')


def _everfree():
    ic = Ico(34, 30, ax=17, ay=28)
    # low mist band
    ic.ell(17, 27, 16, 2, '#c7dcd0')
    ic.ell(17, 27, 12, 1, '#e2eeea')
    # flanking small twisted trees
    for x, c1, c2 in ((5, '#3f6a5a', '#24443e'), (29, '#4f4070', '#2e2648')):
        ic.vl(x, 19, 26, '#4a3a4a')
        ic.c(x - 1, 26, '#4a3a4a')
        ic.c(x + 1, 25, '#4a3a4a')
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy * 1.3 <= 10:
                    ic.c(x + dx, 16 + dy, c1 if dx + dy < 0 else c2)
    # the great gnarled tree: twisted trunk with roots
    ic.poly([(13, 27), (15, 17), (14, 12), (17, 11), (19, 13), (18, 18), (21, 27)], '#5a4458')
    ic.poly([(17, 11), (19, 13), (18, 18), (21, 27), (17, 27)], '#3e2e40')
    for x0, d in ((12, -1), (22, 1)):
        for i in range(4):
            ic.c(x0 + d * i, 27 - (i == 0), '#4a3a4a')
    ic.c(16, 20, '#7a6478')
    ic.c(15, 23, '#7a6478')
    # lumpy spooky canopy (teal + plum), lit top-left
    for cx, cy, r, c1, c2 in ((11, 9, 5, '#4f8a76', '#2e5a50'), (22, 8, 5, '#6a5a9a', '#3e3466'),
                              (17, 5, 6, '#5a9a80', '#35685a')):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy * 1.2 <= r * r:
                    ic.c(cx + dx, cy + dy, c1 if dx + dy < -1 else c2)
    for x, y in ((9, 6), (15, 2), (13, 3), (20, 5), (24, 5)):
        ic.c(x, y, '#8fc8a8' if x < 18 else '#a898d8')
    # bare twisted branches poking out
    for i in range(4):
        ic.c(27 + i // 2, 8 - i, '#4a3a4a')
        ic.c(6 - i // 2, 10 - i, '#4a3a4a')
    # glowing poison-joke flowers
    for x, y in ((3, 27), (8, 28), (11, 26), (24, 27), (27, 28), (31, 26), (19, 28)):
        ic.glow(x, y, '#6fc0ff', '#bfe8ff')
    return ic.finish('#141c1c')


def _zecora():
    ic = Ico(12, 12, ax=6, ay=11)
    ic.ell(6, 7, 4, 4, '#7a5236')
    ic.rect(2, 7, 9, 5, '#7a5236')
    ic.vl(9, 4, 11, '#5a3a26')
    ic.vl(10, 6, 11, '#5a3a26')
    ic.hl(3, 9, 3, '#3f8d45')
    ic.c(5, 2, '#5fae4f')
    ic.rect(5, 8, 2, 4, '#2e1e18')
    ic.c(3, 6, '#f9c22b')
    ic.c(8, 6, '#e83b3b')
    ic.glow(1, 10, '#5fd06a', '#8af07a')
    ic.c(1, 8, '#9ae8a0')
    return ic.finish('#2e1e18')


def _two_sisters():
    ic = Ico(32, 26, ax=16, ay=25)
    # chasm in front with the rope bridge
    ic.rect(0, 22, 32, 4, '#1e1a22')
    ic.hl(0, 31, 22, '#5e7a4e')
    for x in range(2, 30, 2):
        ic.c(x, 23 + (1 if 8 < x < 24 else 0), '#a8784a')
    # ruined castle
    ic.rect(4, 12, 24, 10, '#8a93a6')
    ic.rect(20, 12, 8, 10, '#6c7488')
    for x in range(4, 28, 2):
        ic.c(x, 11, '#8a93a6' if x < 20 else '#6c7488')
    ic.rect(3, 5, 5, 17, '#9aa3b6')
    ic.vl(7, 5, 21, '#6c7488')
    ic.rect(13, 2, 6, 20, '#9aa3b6')
    ic.vl(18, 2, 21, '#6c7488')
    ic.rect(24, 7, 5, 15, '#7c849a')
    # broken tops
    for x, y in ((3, 4), (5, 3), (7, 4), (13, 1), (15, 0), (17, 2), (24, 6), (27, 5)):
        ic.c(x, y, '#9aa3b6')
    ic.c(14, 2, '#00000000') if False else None
    # sun & moon stained glass
    ic.glow(15, 6, '#f9c22b', '#ffe07a')
    ic.glow(16, 6, '#5a7ae0', '#9ab8ff')
    ic.win(5, 9, night='#ffb06a')
    ic.win(26, 11, night='#ffb06a')
    # vines
    for x, y in ((4, 14), (4, 15), (5, 16), (12, 18), (13, 19), (22, 15), (22, 16), (23, 17), (19, 9)):
        ic.c(x, y, '#5e7a4e')
    ic.rect(9, 17, 3, 5, '#2b2d4a')
    return ic.finish('#2b2d4a')


def _tree_harmony():
    ic = Ico(13, 13, ax=6, ay=12)
    ic.poly([(5, 12), (6, 6), (7, 12)], '#b9a7f2')
    for x, y, c in ((2, 5, '#7fd3f5'), (10, 5, '#b9a7f2'), (4, 2, '#ffffff'), (8, 2, '#7fd3f5'),
                    (6, 1, '#b9a7f2'), (3, 7, '#b9a7f2'), (9, 7, '#7fd3f5')):
        ic.glow(x, y, c)
        ic.c(x + 1, y + 1, shade(c, 0.75))
    ic.poly([(6, 6), (2, 5), (4, 2), (6, 1), (8, 2), (10, 5)], '#cfc2f8')
    for x, y, c in ((3, 4, '#e83b3b'), (5, 3, '#f9c22b'), (7, 3, '#4d9be6'), (9, 4, '#ea4fa0'),
                    (4, 5, '#a884f3'), (8, 5, '#1ebc73')):
        ic.glow(x, y, c)
    ic.glow(6, 4, '#ff79e1', '#ffffff')
    ic.hl(3, 9, 12, '#5a4a7a')
    return ic.finish('#3a2f5a')


def _neighagra():
    ic = Ico(42, 28, ax=21, ay=21)
    # flanking cliffs (rock) left and right of the horseshoe
    ic.poly([(0, 20), (3, 9), (9, 6), (10, 22), (2, 24)], '#8a7a6a')
    ic.poly([(41, 20), (38, 9), (32, 6), (31, 22), (39, 24)], '#6e5e50')
    ic.hl(2, 9, 8, '#3f8d45')
    ic.hl(32, 39, 8, '#2c6b3d')
    # river surface above the lip (flowing towards the viewer)
    ic.ell(21, 7, 13, 5, '#56aedc')
    for x, y in ((14, 5), (19, 4), (25, 6), (29, 5), (17, 8), (23, 9)):
        ic.c(x, y, '#9fe0f6')
    # the horseshoe curtain: lip arc (white) then falling streaks
    for x in range(8, 35):
        t = (x - 21) / 13.0
        y0 = int(round(10 + 3 * (1 - t * t)))
        ic.c(x, y0, '#ffffff')
        for y in range(y0 + 1, 21):
            k = (x * 3 + y // 3) % 4
            ic.c(x, y, '#ffffff' if k == 0 else '#d8f6ff' if k == 1 else '#8fd3ff' if k == 2 else '#bfe8ff')
    # foam + mist at the foot
    ic.ell(21, 22, 15, 2, '#eafcff')
    for x in range(6, 37, 4):
        ic.c(x, 20, '#ffffff')
        ic.c(x + 2, 24, '#ffffff')
    ic.ell(21, 25, 11, 1, '#6cc2e6')
    return ic.finish('#2e4a6a')


def _rainbow_falls():
    ic = Ico(34, 34, ax=17, ay=33)
    # mesa cliff
    ic.poly([(3, 12), (31, 12), (29, 33), (5, 33)], '#a98466')
    ic.poly([(20, 12), (31, 12), (29, 33), (21, 33)], '#86644e')
    for y in range(15, 33, 4):
        ic.hl(5, 29, y, '#94704c')
    ic.rect(3, 10, 28, 3, '#7fc96b')
    ic.hl(3, 30, 13, '#5aa64c')
    # rainbow falls down the face
    rain = ('#ff5a5a', '#ffa54a', '#ffe15a', '#6bd66b', '#5ab4ff', '#a275e6')
    for x0, y1 in ((7, 31), (15, 33), (24, 30)):
        for i, c in enumerate(rain[::2] if x0 != 15 else rain):
            ic.vl(x0 + i, 13, y1 - (i % 2), c)
    # village on top
    ic.house(5, 10, 5, 4, '#c8a070', '#8a4e2a', 'gable', rh=3)
    ic.house(12, 10, 6, 5, '#d8b080', '#a0432a', 'gable', rh=3)
    ic.house(21, 10, 5, 4, '#c8a070', '#8a4e2a', 'gable', rh=3)
    ic.flag(28, 4, '#5ab4ff')
    return ic.finish('#4a3426')


def _manehattan():
    ic = Ico(48, 44, ax=24, ay=40)
    # piers into the harbour
    for x in (3, 9, 37, 43):
        ic.rect(x, 36, 3, 7, '#8a5a3a')
        ic.hl(x, x + 2, 36, '#b98a55')
    ic.rect(6, 34, 36, 4, '#d8c8a8')
    ic.hl(6, 41, 37, '#a8987a')
    blds = [  # x, w, h, wall, top
        (7, 6, 14, '#b8c8b8', '#5f8f8a'), (12, 5, 20, '#eadbc0', '#b5654a'), (33, 6, 17, '#8fa3b8', '#5f7f9a'),
        (38, 5, 11, '#eadbc0', '#b5654a'), (16, 6, 24, '#9fd0c8', '#3f8f8a'), (28, 6, 22, '#eadbc0', '#c8a24a'),
        (22, 7, 30, '#d8e4e0', '#c9a24a'),
    ]
    for x, w, h, wall, top in blds:
        ic.rect(x, 34 - h, w, h, wall)
        ic.vl(x + w - 1, 34 - h, 33, shade(wall, 0.72))
        ic.vl(x, 34 - h, 33, shade(wall, 1.08))
        ic.rect(x + 1, 34 - h - 2, w - 2, 2, top)
        ic.c(x + w // 2, 34 - h - 3, top)
        for wy in range(34 - h + 2, 33, 3):
            for wx in range(x + 1, x + w - 1, 2):
                ic.win(wx, wy, day=shade(wall, 0.6))
    # the Crystaller deco crown + spire
    ic.poly([(23, 4), (25, 0), (27, 4)], '#e8e0c8')
    ic.vl(25, -1, 4, '#f9c22b')
    ic.c(25, 0, '#fbff86')
    for x in (23, 25, 27):
        ic.glow(x, 6, '#f9c22b', '#fff0a0')
    return ic.finish('#2e3a4a')


def _baltimare():
    ic = Ico(34, 24, ax=17, ay=22)
    ic.rect(1, 20, 32, 3, '#c8b088')
    for x in range(2, 32, 4):
        ic.vl(x, 21, 23, '#7a5a3a')
    ic.house(3, 20, 7, 6, '#f0eaD8'.lower(), '#3e7fb0', 'gable', rh=3)
    ic.tower(23, 20, 4, 12, '#f0ead8', '#3e7fb0', 4, tip='#f9c22b')
    ic.tower(28, 20, 3, 8, '#e0d8c4', '#d9b65a', 3)
    # domed civic hall with columns
    ic.rect(11, 14, 12, 6, '#f6f2e6')
    for x in range(12, 22, 2):
        ic.vl(x, 15, 19, '#c8c0aa')
    ic.hl(10, 23, 14, '#ffffff')
    ic.dome(16.5, 13, 5, '#f6f2e6')
    ic.c(16, 7, '#d9b65a')
    ic.c(16, 6, '#d9b65a')
    # harbour crane
    ic.vl(31, 9, 19, '#b86b4b')
    ic.hl(26, 32, 9, '#b86b4b')
    ic.vl(27, 10, 12, '#5a4a48')
    return ic.finish('#2e3a4a')


def _fillydelphia():
    ic = Ico(32, 26, ax=16, ay=25)
    ic.house(1, 25, 8, 7, '#b86b4b', '#6e2727', 'gable', rh=3, trim='#f0ead8')
    ic.house(23, 25, 8, 6, '#b86b4b', '#6e2727', 'gable', rh=3, trim='#f0ead8')
    # glass towers
    for x, h in ((9, 20), (14, 16)):
        ic.rect(x, 25 - h, 5, h, '#b4e6ff')
        ic.vl(x + 4, 25 - h, 24, '#6fa8f0')
        ic.vl(x + 1, 25 - h + 1, 23, '#e8ffff')
        ic.poly([(x, 25 - h), (x + 2, 25 - h - 3), (x + 4, 25 - h)], '#e8ffff')
        for wy in range(25 - h + 2, 24, 3):
            ic.win(x + 2, wy, day='#8fb8e0', night='#fff0a0')
    # white clock spire
    ic.tower(19, 25, 3, 12, '#f6f2e6', '#3e7fb0', 4, tip='#f9c22b')
    ic.c(20, 16, '#3a3040')
    # bronze bell in its pavilion
    ic.rect(23, 11, 7, 1, '#6e2727')
    ic.vl(23, 12, 18, '#f0ead8')
    ic.vl(29, 12, 18, '#f0ead8')
    ic.poly([(24, 17), (25, 13), (27, 13), (28, 17)], '#c9a24a')
    ic.poly([(26, 13), (27, 13), (28, 17), (26, 17)], '#9a6e32')
    ic.c(26, 18, '#5a3a26')
    return ic.finish('#3a2420')


def _griffonstone():
    ic = Ico(38, 50, ax=19, ay=49)
    # the dead-tree rock spire
    ic.poly([(12, 49), (15, 30), (14, 18), (18, 12), (20, 12), (24, 18), (22, 30), (26, 49)], '#8e8a82')
    ic.poly([(19, 12), (20, 12), (24, 18), (22, 30), (26, 49), (19, 49)], '#6e6a64')
    for y in range(22, 48, 5):
        ic.c(16, y, '#a8a49c')
        ic.c(21, y + 2, '#56524c')
    # branches
    for pts, c in ((((15, 20), (6, 12), (4, 6), (6, 6), (9, 11), (16, 17)), '#8e8a82'),
                   (((22, 18), (31, 11), (33, 4), (35, 5), (33, 12), (23, 21)), '#7e7a73'),
                   (((17, 13), (14, 5), (16, 3), (19, 11)), '#8e8a82'),
                   (((21, 13), (26, 4), (28, 5), (23, 14)), '#7e7a73')):
        ic.poly(pts, c)
    # plank houses clinging to the branches
    for x, yb, w in ((3, 7, 4), (11, 11, 4), (28, 9, 5), (24, 15, 4), (8, 20, 4), (25, 27, 4), (12, 33, 4)):
        ic.house(x, yb, w, 3, '#a68e6b', '#6b5236', 'gable', rh=2, door=False)
    # ruined castle on top
    ic.rect(15, 6, 7, 6, '#9ea6ae')
    ic.vl(21, 6, 11, '#7e868e')
    for x in (15, 17, 19, 21):
        ic.c(x, 5, '#9ea6ae')
    ic.flag(18, 0, '#c8a24a')
    ic.win(18, 8, night='#ffc06a')
    # rocky base
    ic.poly([(6, 49), (12, 42), (26, 42), (32, 49)], '#5e6b5a')
    ic.hl(9, 29, 45, '#7e8a6a')
    return ic.finish('#33302c')


def _trottingham():
    ic = Ico(24, 16, ax=12, ay=15)
    for x, yb, w, h in ((1, 15, 6, 5), (8, 14, 5, 5), (17, 15, 6, 5)):
        ic.house(x, yb, w, h, '#f4efe4', '#b33831', 'gable', rh=3)
        ic.vl(x, yb - h, yb - 1, '#4c3e24')
        ic.c(x + 2, yb - h + 1, '#4c3e24')
        ic.c(x + 3, yb - h + 2, '#4c3e24')
    ic.tower(13, 12, 4, 9, '#9babb2', '#6e2727', 3, wins=False)
    for x in (13, 15):
        ic.c(x, 2, '#9babb2')
    ic.flag(15, 0, '#e83b3b')
    return ic.finish('#3a2a2a')


def _station():
    ic = Ico(12, 9, ax=6, ay=8)
    ic.house(1, 8, 8, 4, '#c08a58', '#6e2727', 'gable', rh=3)
    ic.hl(0, 11, 8, '#9a948c')
    ic.vl(10, 2, 7, '#5a4a48')
    ic.c(10, 1, '#f9c22b')
    return ic.finish()


def _dragon_lands():
    ic = Ico(58, 42, ax=29, ay=41)
    ic.poly([(2, 41), (22, 8), (36, 8), (56, 41)], '#3c3236')
    ic.poly([(29, 8), (36, 8), (56, 41), (29, 41)], '#2a2426')
    ic.poly([(2, 41), (22, 8), (26, 8), (14, 41)], '#524449')
    # strata / ash streaks
    for y in range(14, 40, 5):
        ic.hl(int(29 - (y - 8) * 0.6), int(29 - (y - 8) * 0.4), y, '#6a595b')
    # crater + lava
    ic.ell(29, 8, 7, 2, '#8a1c12')
    ic.ell(29, 8, 5, 1, '#ff551e')
    for x in (26, 29, 32):
        ic.glow(x, 8, '#ffd04a', '#ffe07a')
    # lava rivers
    for sx, d in ((24, -1), (31, 1), (34, 1)):
        x = sx
        for y in range(10, 36):
            if y % 3 == 0:
                x += d * (1 if (y // 3) % 2 else 0)
            ic.glow(x, y, '#ff551e' if y % 4 else '#ffd04a', '#ff8a3a')
    # green foothills
    ic.poly([(0, 41), (8, 34), (14, 41)], '#3f8d45')
    ic.poly([(44, 41), (51, 33), (57, 41)], '#2c6b3d')
    return ic.finish('#1a1416')


def _kirin():
    ic = Ico(18, 16, ax=9, ay=15)
    for x in (1, 3, 15, 16):
        ic.vl(x, 4, 15, '#9ad45a')
        ic.c(x, 3, '#bfe87a')
    ic.house(4, 15, 10, 5, '#c8a070', '#3e8048', 'gable', rh=4, trim='#e83b3b')
    ic.c(3, 9, '#5aa858')
    ic.c(14, 9, '#5aa858')
    ic.glow(5, 12, '#ff7a3a', '#ffc06a')
    ic.glow(12, 12, '#ff7a3a', '#ffc06a')
    ic.hl(2, 15, 15, '#30e1b9')
    return ic.finish('#1e3a26')


def _mount_aris():
    ic = Ico(28, 46, ax=14, ay=45)
    ic.poly([(4, 45), (9, 18), (13, 12), (16, 12), (20, 20), (24, 45)], '#9a8a76')
    ic.poly([(14, 12), (16, 12), (20, 20), (24, 45), (14, 45)], '#76685a')
    # stairway zigzag
    x = 8
    for y in range(44, 16, -1):
        ic.c(x, y, '#e8dcc4')
        x += 1 if (y // 5) % 2 else -1
        x = max(6, min(20, x))
    # palace + bud dome
    ic.rect(7, 12, 14, 5, '#d9cbb0')
    ic.vl(20, 12, 16, '#b0a288')
    ic.tower(5, 15, 3, 7, '#d9cbb0', '#2d7fb8', 3, wins=False)
    ic.tower(20, 15, 3, 7, '#d9cbb0', '#2d7fb8', 3, wins=False)
    ic.poly([(10, 12), (11, 5), (14, 1), (17, 5), (18, 12)], '#e3b24f')
    ic.poly([(14, 1), (17, 5), (18, 12), (14, 12)], '#b88a3a')
    ic.vl(14, -2, 1, '#e3b24f')
    for x in (9, 13, 17):
        ic.win(x, 14)
    # flowers on terraces
    for x in (7, 10, 19):
        ic.c(x, 17, '#e46fb8')
    ic.hl(2, 25, 45, '#eafcff')
    return ic.finish('#3a3026')


def _seaquestria():
    ic = Ico(30, 18, ax=15, ay=12)
    ic.ell(15, 14, 13, 3, '#2a1f5e')
    for x, h, c in ((5, 6, '#e46fb8'), (9, 9, '#4fd1c5'), (15, 13, '#e46fb8'), (21, 9, '#a275e6'),
                    (25, 6, '#4fd1c5'), (12, 7, '#a275e6'), (18, 8, '#4fd1c5')):
        ic.rect(x - 1, 15 - h, 3, h, c)
        ic.vl(x + 1, 15 - h, 14, shade(c, 0.7))
        ic.glow(x, 15 - h, '#ffffff', '#ffd8f0')
        ic.glow(x, 15 - h + 2, shade(c, 1.2), '#ffd8f0')
    ic.ell(15, 4, 3, 2, '#ff79e1')
    ic.glow(15, 4, '#ffd8f0', '#ffffff')
    out = ic.finish('#1a1440')
    # submerged: thin water dither over the icon
    yy, xx = np.mgrid[0:out.h, 0:out.w]
    m = ((xx + yy) % 2 == 0) & (yy < out.h * 0.35)
    out.rgba[m & (out.rgba[:, :, 3] > 0), :3] = col('#3f7fc2')
    out.grey = Icon(out.rgba, out.lights, out.ax, out.ay).grey
    return out


def _storm_king():
    ic = Ico(28, 36, ax=14, ay=35)
    ic.poly([(4, 35), (8, 22), (12, 14), (14, 4), (16, 14), (20, 22), (24, 35)], '#1c1a26')
    ic.poly([(14, 4), (16, 14), (20, 22), (24, 35), (14, 35)], '#2a2638')
    for x, y0, h in ((6, 35, 12), (21, 35, 14), (10, 35, 18), (18, 35, 16)):
        ic.poly([(x - 2, y0), (x, y0 - h), (x + 2, y0)], '#3c2e3e')
        ic.c(x, y0 - h - 1, '#1c1a26')
    for x, y in ((13, 12), (14, 16), (15, 20), (11, 26), (17, 28), (14, 30), (8, 30), (20, 31)):
        ic.glow(x, y, '#4fe8ff', '#8ff8ff')
    return ic.finish('#100c16')


def _klugetown():
    ic = Ico(34, 40, ax=17, ay=39)
    # rock pillars
    for x, w in ((6, 6), (16, 7), (26, 5)):
        ic.rect(x, 26, w, 14, '#a0784e')
        ic.vl(x + w - 1, 26, 39, '#7a5634')
    # stacked shacks
    stack = [(3, 26, 8, 5, '#c27a3a'), (13, 26, 10, 6, '#8a4e2a'), (24, 26, 8, 5, '#c27a3a'),
             (5, 21, 7, 5, '#b86b3a'), (14, 20, 8, 5, '#c8884a'), (23, 21, 6, 5, '#9e5a30'),
             (9, 15, 7, 5, '#c27a3a'), (18, 15, 7, 5, '#8a4e2a'), (13, 10, 6, 5, '#b86b3a')]
    for x, yb, w, h, c in stack:
        ic.house(x, yb, w, h, c, '#5a3a26', 'flat', door=False)
    # chimneys + crane + windmill
    for x, y in ((11, 9), (24, 12), (15, 5)):
        ic.vl(x, y, y + 4, '#5a4a48')
    ic.vl(31, 8, 26, '#8a4e2a')
    ic.hl(26, 33, 8, '#8a4e2a')
    ic.vl(27, 9, 13, '#3a3040')
    for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        ic.c(5 + dx, 12 + dy, '#ebcb8b')
    ic.vl(5, 12, 15, '#5a4a48')
    # airship docked (balloon hull)
    ic.ell(26, 2, 6, 2, '#d8a050')
    ic.hl(21, 31, 3, '#a87030')
    ic.rect(25, 5, 3, 1, '#7a4a2a')
    return ic.finish('#3a2416')


def _changeling():
    ic = Ico(34, 28, ax=17, ay=27)
    ic.poly([(1, 27), (6, 14), (11, 10), (14, 3), (17, 9), (21, 5), (24, 12), (29, 15), (33, 27)], '#3a3f45')
    ic.poly([(17, 9), (21, 5), (24, 12), (29, 15), (33, 27), (17, 27)], '#2b2f33')
    for x, y in ((8, 20), (13, 15), (20, 18), (26, 22), (16, 23), (11, 24), (23, 13)):
        ic.ell(x, y, 1, 1, '#14161a')
        ic.glow(x, y + 1, '#8be04b', '#a8ff6a')
    for x, h in ((6, 9), (27, 7), (14, 5)):
        ic.poly([(x - 1, 27 - 3), (x, 27 - 3 - h), (x + 1, 27 - 3)], '#4a5058')
    ic.ell(29, 26, 3, 1, '#8be04b')
    ic.ell(4, 26, 2, 1, '#6aa83a')
    return ic.finish('#101214')


def _tartarus():
    ic = Ico(18, 16, ax=9, ay=15)
    ic.poly([(0, 15), (2, 2), (16, 1), (17, 15)], '#3a1a1a')
    ic.rect(4, 4, 10, 11, '#5a5050')
    ic.vl(9, 4, 14, '#1a0f12')
    ic.hl(4, 13, 4, '#7a6a66')
    for y in range(6, 14, 2):
        ic.glow(9, y, '#ff5a1f', '#ff8a3a')
    ic.glow(8, 9, '#ffd04a')
    ic.glow(10, 9, '#ffd04a')
    ic.c(6, 7, '#7a6a66')
    ic.c(12, 7, '#7a6a66')
    return ic.finish('#100808')


def _appleloosa():
    ic = Ico(32, 20, ax=16, ay=19)
    fronts = [(1, 6, 7, '#f2a7a1'), (8, 5, 9, '#9fd6c4'), (14, 6, 7, '#d58a4f'), (21, 5, 8, '#f2d29a'),
              (26, 5, 6, '#b7794a')]
    for x, w, h, c in fronts:
        ic.rect(x, 19 - h, w, h, c)
        ic.vl(x + w - 1, 19 - h, 18, shade(c, 0.75))
        ic.hl(x, x + w - 1, 19 - h, shade(c, 1.15))
        ic.c(x + w // 2, 18, '#4a3020')
        ic.win(x + 1, 19 - h + 2, day='#5a4a3a')
    # clock tower + water tower
    ic.tower(12, 12, 3, 6, '#f2d29a', '#a0432a', 3, wins=False)
    ic.c(13, 8, '#3a3040')
    ic.rect(28, 4, 4, 4, '#8b5a3a')
    ic.vl(28, 8, 13, '#5a3a26')
    ic.vl(31, 8, 13, '#5a3a26')
    ic.hl(28, 31, 3, '#a0702a')
    return ic.finish('#3a2416')


def _dodge():
    ic = Ico(22, 14, ax=11, ay=13)
    ic.house(2, 13, 8, 5, '#d8b47a', '#8b5a3a', 'gable', rh=3)
    ic.hl(0, 21, 13, '#9a948c')
    ic.house(12, 13, 5, 4, '#c0303a', '#8b5a3a', 'flat')
    ic.rect(18, 3, 4, 3, '#8b5a3a')
    ic.vl(18, 6, 12, '#5a3a26')
    ic.vl(21, 6, 12, '#5a3a26')
    ic.c(19, 2, '#f2d04a')
    return ic.finish('#3a2416')


def _rock_farm():
    ic = Ico(16, 15, ax=8, ay=14)
    ic.house(1, 14, 7, 8, '#9a9490', '#5e5858', 'gable', rh=3)
    ic.rect(10, 6, 3, 8, '#b8b0aa')
    ic.dome(11, 5, 1, '#8a8480')
    ic.vl(14, 4, 13, '#5a4a48')
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (-2, -2), (2, 2)):
        ic.c(14 + dx, 4 + dy, '#d8d0c8')
    return ic.finish()


def _las_pegasus():
    ic = Ico(44, 34, ax=22, ay=33)
    for cx, cy, rx, ry in ((10, 29, 9, 4), (22, 30, 12, 4), (34, 29, 9, 4)):
        ic.ell(cx, cy, rx, ry, '#e2eaff')
    for x in range(3, 41):
        ic.c(x, 33 if x % 4 else 32, '#b6c2e8')
    for cx, cy in ((8, 27), (20, 27), (32, 27)):
        ic.ell(cx, cy, 4, 2, '#ffffff')
    towers = [(6, 4, 12, '#ff9fd0', '#ff4fd8'), (11, 5, 18, '#f4ecf8', '#ffe05a'), (17, 6, 22, '#4f3a7a', '#4ff6ff'),
              (24, 5, 16, '#f4ecf8', '#ff4fd8'), (30, 4, 13, '#9fd8ff', '#ffe05a')]
    for x, w, h, wall, neon in towers:
        ic.rect(x, 27 - h, w, h, wall)
        ic.vl(x + w - 1, 27 - h, 26, shade(wall, 0.7))
        ic.hl(x, x + w - 1, 27 - h, neon)
        for wy in range(27 - h + 2, 26, 2):
            for wx in range(x + 1, x + w - 1, 2):
                ic.win(wx, wy, day=shade(wall, 0.65), night=neon if (wx + wy) % 3 == 0 else '#ffe8a0')
        ic.glow(x + w // 2, 27 - h - 1, neon)
    # ferris wheel
    for k in range(12):
        import math
        a = k / 12 * 2 * math.pi
        ic.glow(38 + round(math.cos(a) * 4), 18 + round(math.sin(a) * 4), '#ff4fd8' if k % 2 else '#ffe05a')
    ic.vl(38, 18, 27, '#5a4a6a')
    # neon sign star
    ic.star(20, 2, '#ffe05a', '#ffffff')
    return ic.finish('#2a1a4a')


def _applewood():
    ic = Ico(26, 12, ax=13, ay=11)
    ic.poly([(0, 11), (8, 5), (18, 4), (25, 11)], '#7cbc48')
    for i, x in enumerate(range(2, 22, 2)):
        ic.vl(x, 5 - (i % 2 == 0) + (1 if x < 6 or x > 18 else 0), 7 + (1 if x < 6 or x > 18 else 0), '#ffffff')
    ic.star(23, 3, '#f9c22b')
    return ic.finish('#2e4a2a')


def _somnambula():
    ic = Ico(22, 16, ax=11, ay=15)
    ic.poly([(3, 15), (11, 3), (19, 15)], '#e8c88a')
    ic.poly([(11, 3), (19, 15), (11, 15)], '#c09a5e')
    for y in range(6, 15, 3):
        ic.hl(int(11 - (y - 3) * 0.66), 11, y, '#d4b070')
    ic.c(11, 3, '#f9c22b')
    ic.house(0, 15, 4, 3, '#f0d8a8', '#b08850', 'flat', door=False)
    ic.house(18, 15, 4, 3, '#f0d8a8', '#b08850', 'flat', door=False)
    return ic.finish('#5a4026')


def _vanhoover():
    ic = Ico(30, 28, ax=15, ay=27)
    for x, w, h in ((3, 5, 12), (8, 5, 20), (14, 6, 24), (20, 5, 16), (25, 4, 10)):
        ic.rect(x, 27 - h, w, h, '#e8f2f6')
        ic.vl(x + w - 1, 27 - h, 26, '#a8c4d0')
        ic.vl(x + 1, 27 - h + 1, 25, '#ffffff')
        for wy in range(27 - h + 2, 26, 2):
            ic.win(x + 2, wy, day='#8fb0c8', night='#fff0b0')
            if w > 4:
                ic.win(x + 3, wy, day='#a8c4d8', night='#ffe08a')
    ic.poly([(14, 3), (16, 0), (19, 3)], '#e8f2f6')
    ic.vl(16, -2, 1, '#9babb2')
    for x in (1, 28):
        ic.poly([(x - 2, 27), (x, 20), (x + 2, 27)], '#2d6a4a')
    ic.hl(0, 29, 27, '#5a7a8a')
    return ic.finish('#2a3a4a')


def _small_town(walls=('#c8a070', '#e0c8a0', '#b88860'), roofs=('#8a4e2a', '#6b3e2a'), n=3, w=18, h=12):
    ic = Ico(w, h, ax=w // 2, ay=h - 1)
    xs = [1, w // 2 - 2, w - 6][:n]
    for i, x in enumerate(xs):
        ic.house(x, h - 1 - (i == 1), 5, 4, walls[i % len(walls)], roofs[i % len(roofs)], 'gable', rh=3)
    return ic


def _tall_tale():
    ic = _small_town(w=18, h=12)
    ic.vl(9, 0, 5, '#7a5236')
    ic.c(9, 0, '#f9c22b')
    return ic.finish()


def _silver_shoals():
    ic = Ico(20, 14, ax=10, ay=13)
    ic.rect(0, 10, 20, 4, '#3f7fc2')
    for x in range(1, 20, 2):
        ic.c(x, 12, '#8fd3ff')
    for x, yb in ((2, 9), (8, 8), (14, 9)):
        ic.house(x, yb, 5, 4, '#d8d4cc', '#5a7a9a', 'gable', rh=3)
        ic.vl(x, yb, yb + 3, '#5a4030')
        ic.vl(x + 4, yb, yb + 3, '#5a4030')
    ic.hl(0, 19, 9, '#8a6a4a')
    return ic.finish('#22303e')


def _hope_hollow():
    ic = _small_town(walls=('#f2c8a0', '#f7e0c0', '#e8b890'), roofs=('#c06a4a', '#a0583a'), w=20, h=14)
    rain = ('#ff5a5a', '#ffa54a', '#ffe15a', '#6bd66b', '#5ab4ff', '#a275e6')
    import math
    for i, c in enumerate(rain):
        r = 7 - i * 0.7
        for k in range(17):
            a = math.pi * k / 16
            ic.c(round(10 + math.cos(a) * r * 1.3), round(8 - math.sin(a) * r * 0.9), c)
    return ic.finish('#3a2a2a')


def _crystal_empire():
    ic = Ico(46, 56, ax=23, ay=55)
    # snowy ground disc + ring of crystal houses
    ic.ell(23, 50, 21, 5, '#eef6fb')
    ic.ell(23, 51, 18, 3, '#d8ecf6')
    houses = [(4, 50, '#f39ce8'), (9, 53, '#8fb8f2'), (36, 53, '#e7a8d9'), (41, 50, '#7e5db0'),
              (14, 55, '#bfe8ff'), (30, 55, '#f39ce8'), (6, 46, '#bfe8ff'), (39, 46, '#8fb8f2')]
    for x, yb, c in houses[:6]:
        ic.poly([(x - 2, yb), (x - 2, yb - 4), (x, yb - 7), (x + 2, yb - 4), (x + 2, yb)], c)
        ic.vl(x + 1, yb - 5, yb, shade(c, 0.72))
        ic.vl(x - 1, yb - 4, yb - 1, shade(c, 1.15))
        ic.win(x, yb - 2, day=shade(c, 0.6), night='#ffe8f8')
    # the slender crystal spire palace on crystal arches
    ic.poly([(16, 49), (18, 40), (28, 40), (30, 49)], '#8fb8f2')
    ic.rect(20, 43, 2, 6, '#3e5bc6')
    ic.rect(25, 43, 2, 6, '#3e5bc6')
    ic.poly([(18, 40), (20, 22), (23, 2), (26, 22), (28, 40)], '#bfe8ff')
    ic.poly([(23, 2), (26, 22), (28, 40), (23, 40)], '#6fa8f0')
    ic.vl(21, 18, 38, '#ffffff')
    for x, yb, h in ((15, 41, 14), (31, 41, 14)):
        ic.poly([(x - 2, yb), (x, yb - h), (x + 2, yb)], '#b4e6ff')
        ic.poly([(x, yb - h), (x + 2, yb), (x, yb)], '#6fa8f0')
    for wy in range(14, 38, 5):
        ic.win(23, wy, day='#3e5bc6', night='#ffd8f8')
    # the crystal heart beneath
    ic.glow(22, 46, '#f39ce8', '#ffb0f0')
    ic.glow(24, 46, '#f39ce8', '#ffb0f0')
    ic.glow(23, 47, '#ff79e1', '#ff9ff0')
    ic.glow(23, 46, '#ffffff')
    for x, yb, c in houses[6:]:
        ic.poly([(x - 2, yb), (x - 2, yb - 4), (x, yb - 7), (x + 2, yb - 4), (x + 2, yb)], c)
        ic.vl(x + 1, yb - 5, yb, shade(c, 0.72))
        ic.win(x, yb - 2, day=shade(c, 0.6), night='#ffe8f8')
    return ic.finish('#2a3a78')


def _yakyakistan():
    ic = Ico(34, 24, ax=17, ay=23)
    # log fortress walls
    ic.rect(2, 12, 30, 11, '#8a5a3a')
    for x in range(2, 32, 2):
        ic.vl(x, 12, 22, '#6e4428')
        ic.c(x, 11, '#8a5a3a')
    ic.rect(26, 12, 6, 11, '#6e4428')
    # gate with horn carvings
    ic.rect(13, 15, 8, 8, '#4a2c1a')
    ic.c(12, 14, '#f2f6fa')
    ic.c(11, 13, '#f2f6fa')
    ic.c(21, 14, '#f2f6fa')
    ic.c(22, 13, '#f2f6fa')
    # painted folk bands
    for x in range(3, 31, 3):
        ic.c(x, 13, ('#c0392b', '#2e6db4', '#f2c84a')[x % 3])
    # great hall + huts with snowy roofs
    ic.house(6, 12, 8, 5, '#a06a42', '#f2f6fa', 'gable', rh=4, door=False)
    ic.house(19, 12, 9, 6, '#a06a42', '#f2f6fa', 'gable', rh=5, door=False)
    ic.win(9, 10, night='#ffc06a')
    ic.win(23, 9, night='#ffc06a')
    ic.flag(16, 0, '#c0392b')
    ic.glow(12, 16, '#ff9a3a', '#ffc06a')
    ic.glow(22, 16, '#ff9a3a', '#ffc06a')
    return ic.finish('#2e1e14')


def _our_town():
    ic = Ico(24, 14, ax=12, ay=13)
    for row, yb in ((0, 6), (1, 12)):
        for x in range(1, 20, 5):
            ic.house(x, yb, 4, 3, '#d9c49a', '#b5523b', 'gable', rh=2, door=False)
    ic.tree(22, 12, '#5aa64c', '#3f8d45', 2)
    ic.hl(0, 20, 9, '#e8d8b0')
    return ic.finish('#4a3426')


def _starlight_cave():
    ic = Ico(14, 10, ax=7, ay=9)
    ic.poly([(0, 9), (4, 1), (9, 0), (13, 9)], '#8a8699')
    ic.poly([(9, 0), (13, 9), (8, 9)], '#6c687c')
    ic.ell(6, 7, 2, 2, '#1e1a22')
    ic.rect(4, 7, 5, 3, '#1e1a22')
    return ic.finish('#2a2632')


def _hollow_shades():
    ic = Ico(22, 15, ax=11, ay=14)
    for x, yb, h in ((1, 14, 5), (7, 12, 6), (14, 14, 5), (18, 11, 4)):
        ic.house(x, yb, 4, h, '#4a3f6a', '#2d2f4a', 'gable', rh=5, door=False, wins=False)
        ic.win(x + 1, yb - 2, day='#2a2848', night='#9ab0ff')
    ic.glow(11, 14, '#8fa0e0', '#c0d0ff')
    return ic.finish('#141428')


def _farthest_reaches():
    ic = Ico(22, 14, ax=11, ay=13)
    ic.rect(0, 11, 8, 2, '#8a6a4a')
    for x in range(0, 8, 3):
        ic.vl(x, 12, 13, '#5a4030')
    for x, yb in ((5, 10), (11, 11), (16, 10)):
        ic.house(x, yb, 5, 4, '#f4efe4', '#c8313a', 'gable', rh=3)
    ic.vl(2, 3, 10, '#7a5236')
    ic.c(3, 4, '#ffffff')
    ic.c(3, 5, '#ffffff')
    return ic.finish('#3a2a2a')


DESIGNS = {
    'canterlot': _canterlot, 'cloudsdale': _cloudsdale, 'ponyville': _ponyville,
    'sweet_apple_acres': _sweet_apple, 'everfree': _everfree, 'neighagra_falls': _neighagra,
    'rainbow_falls': _rainbow_falls, 'manehattan': _manehattan, 'baltimare': _baltimare,
    'griffonstone': _griffonstone, 'dragon_lands': _dragon_lands, 'mount_aris': _mount_aris,
    'klugetown': _klugetown, 'changeling_kingdom': _changeling, 'appleloosa': _appleloosa,
    'las_pegasus': _las_pegasus, 'vanhoover': _vanhoover, 'crystal_empire': _crystal_empire,
    'yakyakistan': _yakyakistan, 'castle_two_sisters': _two_sisters,
    'castle_friendship': _castle_friendship, 'school_friendship': _school, 'zecoras_hut': _zecora,
    'tree_of_harmony': _tree_harmony, 'hollow_shades': _hollow_shades, 'our_town': _our_town,
    'starlights_cave': _starlight_cave, 'fillydelphia': _fillydelphia, 'trottingham': _trottingham,
    'griffonstone_station': _station, 'kirin_grove': _kirin, 'seaquestria': _seaquestria,
    'storm_kings_realm': _storm_king, 'tartarus': _tartarus, 'dodge_junction': _dodge,
    'rock_farm': _rock_farm, 'applewood': _applewood, 'somnambula': _somnambula, 'tall_tale': _tall_tale,
    'silver_shoals': _silver_shoals, 'hope_hollow': _hope_hollow, 'farthest_reaches': _farthest_reaches,
}


@lru_cache(maxsize=None)
def icon(lid):
    """Icon for a location id, or None if the place has no icon (natural features, regions, seas)."""
    fn = DESIGNS.get(lid)
    return fn() if fn else None


@lru_cache(maxsize=None)
def icon_half(lid):
    """Half-size icon (nearest 2x2 pick, keeps opaque majority) for distant mode-7 billboards."""
    ic = icon(lid)
    if ic is None:
        return None
    a = ic.rgba
    h2, w2 = a.shape[0] // 2, a.shape[1] // 2
    sub = a[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 4)
    alpha = (sub[..., 3] > 0).sum(axis=(1, 3)) >= 2
    rgb = sub[:, 0, :, 0, :3].copy()
    alt = sub[:, 1, :, 1, :3]
    use_alt = sub[:, 0, :, 0, 3] == 0
    rgb[use_alt] = alt[use_alt]
    out = np.zeros((h2, w2, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.where(alpha, 255, 0)
    return Icon(out, np.zeros_like(out), ic.ax // 2, ic.ay // 2)
