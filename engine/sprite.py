"""Sprites authored as ASCII art.

    CUP = Sprite.from_ascii('''
        .oooo.
        owwwwo
        owbbwoo
        owbbwo.o
        owwwwoo
        .oooo.
    ''', {'o': 'ink', 'w': 'white', 'b': 'brew'})

'.' and ' ' are transparent. Every other character must be in the mapping.
Leading/trailing blank lines and the common indentation are removed, so you can
indent art freely inside code. Rows shorter than the widest row are padded
with transparency on the right.
"""
import textwrap

import numpy as np

from .core import col

TRANSPARENT = '. '


class Sprite:
    __slots__ = ('rgba', '_flip')

    def __init__(self, rgba):
        self.rgba = rgba
        self._flip = None

    @property
    def w(self):
        return self.rgba.shape[1]

    @property
    def h(self):
        return self.rgba.shape[0]

    @classmethod
    def from_ascii(cls, art, mapping):
        lines = textwrap.dedent(art).split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        lines = [ln.rstrip() for ln in lines]
        h = len(lines)
        w = max((len(ln) for ln in lines), default=0)
        rgba = np.zeros((h, w, 4), np.uint8)
        cmap = {k: col(v) for k, v in mapping.items() if v is not None}
        for y, ln in enumerate(lines):
            for x, ch in enumerate(ln):
                if ch in TRANSPARENT:
                    continue
                if ch not in mapping:
                    raise KeyError(f'sprite char {ch!r} at ({x},{y}) not in mapping')
                if mapping[ch] is None:
                    continue
                rgba[y, x, :3] = cmap[ch]
                rgba[y, x, 3] = 255
        return cls(rgba)

    @classmethod
    def from_canvas(cls, canvas, key=None):
        """Wrap a Canvas as a sprite; pixels equal to `key` colour become transparent."""
        a = canvas.a
        rgba = np.zeros(a.shape[:2] + (4,), np.uint8)
        rgba[:, :, :3] = a
        rgba[:, :, 3] = 255
        if key is not None:
            k = np.array(col(key), np.uint8)
            rgba[(a == k).all(axis=2), 3] = 0
        return cls(rgba)

    def flipped(self):
        if self._flip is None:
            self._flip = Sprite(np.ascontiguousarray(self.rgba[:, ::-1]))
        return self._flip

    def recolor(self, mapping):
        """New sprite with colours replaced: {old_spec: new_spec}."""
        out = self.rgba.copy()
        src = self.rgba[:, :, :3]
        for old, new in mapping.items():
            m = (src == np.array(col(old), np.uint8)).all(axis=2) & (self.rgba[:, :, 3] > 0)
            out[m, :3] = col(new)
        return Sprite(out)

    def tinted(self, color):
        """Silhouette: every opaque pixel set to one colour."""
        out = self.rgba.copy()
        out[out[:, :, 3] > 0, :3] = col(color)
        return Sprite(out)

    def outlined(self, color='ink', diagonal=False):
        """New sprite (2px larger) with a 1px outline around opaque pixels."""
        a = self.rgba[:, :, 3] > 0
        h, w = a.shape
        pad = np.zeros((h + 2, w + 2), bool)
        pad[1:-1, 1:-1] = a
        grow = pad.copy()
        grow[1:, :] |= pad[:-1, :]
        grow[:-1, :] |= pad[1:, :]
        grow[:, 1:] |= pad[:, :-1]
        grow[:, :-1] |= pad[:, 1:]
        if diagonal:
            grow[1:, 1:] |= pad[:-1, :-1]
            grow[1:, :-1] |= pad[:-1, 1:]
            grow[:-1, 1:] |= pad[1:, :-1]
            grow[:-1, :-1] |= pad[1:, 1:]
        out = np.zeros((h + 2, w + 2, 4), np.uint8)
        ring = grow & ~pad
        out[ring, :3] = col(color)
        out[ring, 3] = 255
        out[1:-1, 1:-1][a] = self.rgba[a]
        return Sprite(out)

    def scaled(self, k):
        """Integer nearest-neighbour upscale (for big title letters etc.)."""
        return Sprite(np.repeat(np.repeat(self.rgba, k, axis=0), k, axis=1))

    def sub(self, x, y, w, h):
        return Sprite(self.rgba[y:y + h, x:x + w].copy())

    def draw(self, canvas, x, y, flip=False, anchor='topleft'):
        """Blit onto canvas. anchor: 'topleft' | 'bottom' (x=center, y=bottom row+1)
        | 'center'."""
        if anchor == 'bottom':
            x, y = x - self.w // 2, y - self.h
        elif anchor == 'center':
            x, y = x - self.w // 2, y - self.h // 2
        canvas.blit(self.flipped() if flip else self, x, y)


def sprite(art, mapping):
    """Shorthand for Sprite.from_ascii."""
    return Sprite.from_ascii(art, mapping)


def sheet(frames_art, mapping):
    """Build a list of sprites from a list of ASCII arts sharing one mapping."""
    return [Sprite.from_ascii(a, mapping) for a in frames_art]
