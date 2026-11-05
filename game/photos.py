"""The photo book: every postcard (P) the player has taken, for the journal's PHOTOS page and its
full-screen viewer (game/journal.py). No pygame.

Postcards are the PNGs game/postcard.py writes (postcard_<place>_<ms>_<n>.png: the view framed in a
cream border at 4x). The book lists the folders the front end gives it (game/play.py: Pictures/Equestria
Atlas and the game's own game_shots), newest first, and reads each picture back at the art's own size:
the view inside the frame, every 4th pixel, an exact copy of what was on screen. Pictures load lazily,
at most one per step (a PNG decodes in ~8 ms) and only the ones the page shows; a postcard taken this
session is added with its picture at once. The book is listed again each time the journal opens, so
postcards moved or deleted in the folder drop out.
"""
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from engine.loc_worldmap import LOCS

from .compose import VIEW_W, VIEW_H

PREFIX = 'postcard_'
KEEP = 24                       # pictures kept decoded (the thumbnails are kept for every postcard)
MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')


def slug(name):
    """A place name as it appears in a postcard's file name."""
    return name.lower().replace(' ', '_').replace("'", '')


_BY_SLUG = {slug(v['name']): lid for lid, v in LOCS.items()}


@dataclass
class Photo:
    path: str
    lid: object                 # the map location id it was taken in, or None (the world map)
    ms: int                     # when it was taken (ms since the epoch)

    @property
    def name(self):
        return LOCS[self.lid]['name'].upper() if self.lid in LOCS else 'THE WORLD MAP'

    def date(self):
        """'2 NOV 2026'."""
        tm = time.localtime(self.ms / 1000)
        return f'{tm.tm_mday} {MONTHS[tm.tm_mon - 1]} {tm.tm_year}'

    def clock(self):
        """'7:09 PM'."""
        tm = time.localtime(self.ms / 1000)
        return f'{(tm.tm_hour - 1) % 12 + 1}:{tm.tm_min:02d} {"AM" if tm.tm_hour < 12 else "PM"}'


def parse(path):
    """The Photo a postcard file name describes, or None if it is not a postcard."""
    fn = os.path.basename(path)
    if not (fn.startswith(PREFIX) and fn.lower().endswith('.png')):
        return None
    parts = fn[len(PREFIX):-4].rsplit('_', 2)
    lid, ms = None, None
    if len(parts) == 3:
        lid = _BY_SLUG.get(parts[0])
        if parts[1].isdigit():
            ms = int(parts[1])
    if ms is None:
        try:
            ms = int(os.path.getmtime(path) * 1000)
        except OSError:
            return None
    return Photo(path, lid, ms)


def from_postcard(a):
    """The view inside a postcard (an RGB array as game/postcard.py makes it) at the art's own size,
    or any other picture squeezed to the view's size."""
    from .postcard import K, BORDER, CAPTION
    if a.shape[:2] == ((VIEW_H + BORDER + CAPTION) * K, (VIEW_W + 2 * BORDER) * K):
        b = BORDER * K
        return np.ascontiguousarray(a[b:b + VIEW_H * K:K, b:b + VIEW_W * K:K, :3])
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(a[..., :3])).resize((VIEW_W, VIEW_H), Image.BOX))


def shrink(img, w, h):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BOX))


class PhotoBook:
    def __init__(self):
        self.folders = []
        self.items = []                 # Photo, newest first
        self.last_t = None              # step the last postcard was taken (the journal opens on it)
        self._img = OrderedDict()       # path -> (VIEW_H, VIEW_W, 3) uint8, or None (unreadable)
        self._small = {}                # (path, w, h) -> thumbnail
        self._new = {}                  # path -> Photo, taken this session (kept if a folder can't be listed)

    def set_folders(self, folders):
        seen = []
        for f in folders:
            if f and os.path.abspath(f) not in seen:
                seen.append(os.path.abspath(f))
        self.folders = seen
        self.scan()

    def scan(self):
        """List the postcards in the folders again (newest first)."""
        found = {}
        for f in self.folders:
            try:
                names = os.listdir(f)
            except OSError:
                continue
            for fn in names:
                p = parse(os.path.join(f, fn))
                if p is not None:
                    found[p.path] = p
        for path, p in self._new.items():
            if path not in found and os.path.exists(path):
                found[path] = p
        self.items = sorted(found.values(), key=lambda p: (p.ms, p.path), reverse=True)

    def add(self, path, lid, img, t):
        """A postcard just taken (its picture is already at hand)."""
        p = parse(path) or Photo(path, lid, int(time.time() * 1000))
        p.lid = lid
        self._new[path] = p
        self.items = [q for q in self.items if q.path != path]
        self.items.insert(0, p)
        self._keep(path, img)
        self.last_t = t

    def __len__(self):
        return len(self.items)

    # -------------------------------------------------------------- pictures
    def loaded(self, p):
        return p.path in self._img

    def image(self, p):
        """The picture (VIEW_H x VIEW_W RGB) if it has been read, else None."""
        img = self._img.get(p.path)
        if img is not None:
            self._img.move_to_end(p.path)
        return img

    def thumb(self, p, w, h):
        """A w x h copy of the picture (None until it has been read)."""
        key = (p.path, w, h)
        if key not in self._small:
            img = self._img.get(p.path)
            if img is None:
                return None
            self._small[key] = shrink(img, w, h)
        return self._small[key]

    def pump(self, full=(), small=()):
        """Read one photo that is wanted and not read yet (one per call: ~8 ms): first the `full`
        ones (their whole picture), then the `small` ones (thumbnails only). Returns it, or None."""
        for p in full:
            if p.path not in self._img:
                self._keep(p.path, self._read(p.path))
                return p
        for p in small:
            if (p.path, SIZES[0][0], SIZES[0][1]) not in self._small and p.path not in self._img:
                self._keep(p.path, self._read(p.path))
                return p
        return None

    def _read(self, path):
        try:
            from PIL import Image
            with Image.open(path) as im:
                a = np.asarray(im.convert('RGB'))
            return from_postcard(a)
        except Exception as e:                      # moved, half-written or not a picture
            print('photo unreadable:', path, e)
            return None

    def _keep(self, path, img):
        self._img[path] = img
        self._img.move_to_end(path)
        if img is not None:                         # the thumbnails are made while the picture is here
            for w, h in SIZES:
                self._small.setdefault((path, w, h), shrink(img, w, h))
        while len(self._img) > KEEP:                # (its thumbnails stay; the picture is read again if wanted)
            self._img.popitem(last=False)


SIZES = ((32, 18), (128, 72))       # the thumbnails the PHOTOS page draws (filmstrip, preview)
