"""The photo album: 20 postcard targets. Taking a postcard (P) that shows a target ticks it off: a toast
with a sparkle, and a small copy of the photo pinned in the album page of the journal.

A target is a place (and area) plus what must be in the picture: a point in view (a landmark), a
hotspot's event just set off and on screen (the bell ringing, fireworks, the aurora...), or something
on the world map (sky traffic, night). The checks read the game state at the moment of the snapshot;
targets named for a time of day (DAWN, SUNSET, NIGHT) also need the clock in that hour range.
Thumbnails (64 x 36) are saved next to the save file in game_album/<key>.png.

Photo chances: while a postcard would tick a target, viewfinder brackets blink in the corners of the
view for a while and a [camera][P] chip sits bottom-right (the first two times in a session with a
'PICTURE PERFECT!' plate). An event stays a target for EVENT_WINDOW steps after it was set off.
"""
import math
import os
from dataclasses import dataclass

import numpy as np

from engine import W, H, text, text_width

from . import ui, perks

THUMB = (64, 36)
EVENT_WINDOW = 240              # steps an event (the bell, the zap...) stays in the picture
CHANCE_EVERY = 6                # steps between two checks for a photo chance
CHANCE_GAP = 30                 # a chance that is gone for fewer steps than this is the same chance
BRACKETS = 300                  # steps the viewfinder brackets blink at the start of a chance
PERFECT = 150                   # steps of the 'PICTURE PERFECT!' plate (the first PERFECT_N chances)
PERFECT_N = 2
HOURS = {'DAWN': (5, 8), 'SUNSET': (17, 20), 'NIGHT': (20, 5)}     # a word in the title -> hours [a, b)


@dataclass
class Target:
    key: str
    title: str
    place: str                  # map location id, or 'map'
    hint: str
    check: callable


# ------------------------------------------------------------------ checks
def _region(g, lid, area=None):
    r = g.mode
    if getattr(r, 'ID', None) != lid or getattr(r, 'switch', 1) is not None:
        return None
    if area is not None and r.ai != area:
        return None
    return r


def _in_view(r, x, y, p=1.0, margin=24):
    sx, sy = x - r.cx * p, y - r.cy * p
    return margin <= sx <= W - margin and 8 <= sy <= H - 8


def _spot(r, key, within):
    """The hotspot `key` of the current area was set off less than `within` steps ago and is in view."""
    t0 = r.spot_t.get(key)
    if t0 is None or r.g.t - t0 > within * perks.photo_window(r.g):
        return False
    for h in r.area.hotspots:
        if h.key == key:
            py = h.p if h.py is None else h.py
            sx, sy = h.x - r.cx * h.p, h.y - r.cy * py
            return -20 <= sx <= W + 20 and -20 <= sy <= H + 20
    return False


def view(lid, area, x, y, p=1.0):
    def chk(g):
        r = _region(g, lid, area)
        return r is not None and _in_view(r, x, y, p)
    return chk


def event(lid, area, key, within=EVENT_WINDOW):
    def chk(g):
        r = _region(g, lid, area)
        return r is not None and _spot(r, key, within)
    return chk


def _trestle_train(g):
    r = _region(g, 'rainbow_falls', 0)
    if r is None:
        return False
    try:
        from .regions import rainbowfalls as RF
        import scenes.s07_rainbowfalls as sc
        tc = r.train_clock(g.t)
        if tc is None or tc > RF.TRAIN_EMIT:
            return False
        tox = int(round(sc.REF[0] + (r.cx - sc.REF[0]) * sc.TRK))
        x = RF._train_front(tc) - tox
        return -40 <= x <= W - 10
    except Exception:
        return False


def _map_traffic(g):
    from .world import MapMode, TRAFFIC, traffic_pos, traffic_count
    m = g.mode
    if not isinstance(m, MapMode) or m.overview:
        return False
    x0, y0 = m.cx - W / 2, m.cy - H / 2
    for k in range(traffic_count(len(g.quests.fixed))):
        if g.flock.hidden(k, g.t):
            continue
        tx, ty = g.flock.pos(k, g.t)
        if 20 < tx - x0 < W - 20 and 30 < ty - y0 < H:
            return True
    return False


def _map_night(g):
    from .world import MapMode
    return isinstance(g.mode, MapMode) and not g.mode.overview and (g.hour >= 20 or g.hour < 5)


TARGETS = [
    Target('canterlot_dawn', 'CANTERLOT AT DAWN', 'canterlot', 'THE CASTLE AT FIRST LIGHT, 05:00 TO 08:00',
           view('canterlot', 0, 186, 76)),
    Target('ponyville_train', 'THE EXPRESS IN PONYVILLE', 'ponyville', 'CALL THE TRAIN AT THE STATION',
           event('ponyville', 0, 'train')),
    Target('zap_apples', 'ZAP APPLE LIGHTNING', 'sweet_apple_acres', 'MAKE LIGHTNING STRIKE THE TREES',
           event('sweet_apple_acres', 0, 'zap')),
    Target('liquid_rainbow', 'LIQUID RAINBOW', 'cloudsdale', 'START THE RAINBOW FALLS',
           event('cloudsdale', 2, 'falls')),
    Target('trestle_train', 'TRAIN ON THE TRESTLE', 'rainbow_falls', 'THE EXPRESS CROSSING THE GORGE',
           _trestle_train),
    Target('neighagra_barrel', 'OVER THE FALLS', 'neighagra_falls', 'SEND A BARREL OVER THE FALLS',
           event('neighagra_falls', 0, 'barrel')),
    Target('crystaller', 'THE CRYSTALLER BEACON', 'manehattan', 'LIGHT THE TOWER TOP',
           event('manehattan', 0, 'beacon')),
    Target('big_bell', 'THE BIG BELL RINGS', 'baltimare', 'RING THE BELL OF BALTIMARE',
           event('baltimare', 0, 'bell')),
    Target('griffon_storm', 'STORM OVER THE ABYSS', 'griffonstone', 'STIR UP THE STORM CLOUDS',
           event('griffonstone', 1, 'storm')),
    Target('volcano', 'THE VOLCANO ERUPTS', 'dragon_lands', 'WAKE THE VOLCANO',
           lambda g: event('dragon_lands', 0, 'crater')(g) or event('dragon_lands', 1, 'erupt')(g)),
    Target('seaquestria', 'SEAQUESTRIA', 'mount_aris', 'THE CITY UNDER THE SEA',
           view('mount_aris', 3, 196, 40)),
    Target('airship', 'AIRSHIP AT THE DOCKS', 'klugetown', 'SEND OFF THE KLUGETOWN AIRSHIP',
           event('klugetown', 2, 'airship')),
    Target('hive_bloom', 'THE HIVE IN BLOOM', 'changeling_kingdom', 'LIGHT THE HIVE LANTERNS',
           event('changeling_kingdom', 1, 'lanterns')),
    Target('appleloosa_sunset', 'APPLELOOSA AT SUNSET', 'appleloosa', 'THE TOWN IN THE EVENING SUN, 17:00 TO 20:00',
           view('appleloosa', 0, 251, 60)),
    Target('lp_fireworks', 'FIREWORKS AT LAS PEGASUS', 'las_pegasus', 'LAUNCH FIREWORKS OVER THE STRIP',
           event('las_pegasus', 0, 'fireworks')),
    Target('lighthouse', 'THE LIGHTHOUSE AT NIGHT', 'vanhoover', 'LIGHT THE SILVER SHOALS BEAM AFTER 20:00',
           event('vanhoover', 2, 'lighthouse')),
    Target('aurora', 'AURORA OVER THE CRYSTAL EMPIRE', 'crystal_empire', 'OPEN THE NORTHERN LIGHTS',
           event('crystal_empire', 0, 'aurora')),
    Target('harmony', 'THE TREE OF HARMONY', 'castle_two_sisters', 'MAKE THE OLD TREE CHIME',
           event('castle_two_sisters', 2, 'tree')),
    Target('map_balloons', 'BALLOONS OVER THE MAP', 'map', 'ANOTHER BALLOON IN THE PICTURE',
           _map_traffic),
    Target('map_night', 'THE MAP BY NIGHT', 'map', 'THE WORLD MAP AFTER NIGHTFALL',
           _map_night),
]
BY_KEY = {t.key: t for t in TARGETS}


def _hour_ok(tg, h):
    for word, (a, b) in HOURS.items():
        if word in tg.title:
            return a <= h < b if a < b else (h >= a or h < b)
    return True


def shows(g, tg):
    """Would a postcard taken now tick target tg? (its check, and the hour for DAWN / SUNSET / NIGHT)"""
    try:
        return _hour_ok(tg, g.hour % 24) and bool(tg.check(g))
    except Exception:
        return False


class Album:
    def __init__(self, game):
        self.g = game
        self.taken = {}                 # target key -> step the photo was taken
        self.thumbs = {}                # target key -> (36, 64, 3) uint8 thumbnail (loaded lazily)
        self.snap_t = None
        self._ch_key = None             # memo of the photo chances: (step // CHANCE_EVERY, mode, area)
        self._ch = []                   # ... the untaken targets a postcard would tick now
        self._win_t = None              # step the current chance began (None: no chance)
        self._seen_t = None             # step a chance was last seen
        self._wins = 0                  # chances this session that showed the 'PICTURE PERFECT!' plate
        self._plate_t = None            # step the plate of the current chance appeared (None: not yet)

    def folder(self):
        sp = self.g.save_path
        return os.path.join(os.path.dirname(os.path.abspath(sp)), 'game_album') if sp else None

    def save(self):
        return dict(taken=dict(self.taken))

    def load(self, d):
        d = d or {}
        self.taken = {k: int(v) for k, v in d.get('taken', {}).items() if k in BY_KEY}
        self._ch_key, self._ch, self._win_t, self._seen_t = None, [], None, None

    def thumb(self, key):
        if key not in self.thumbs and key in self.taken:
            f = self.folder()
            img = None
            if f:
                try:
                    from PIL import Image
                    img = np.array(Image.open(os.path.join(f, key + '.png')).convert('RGB').resize(THUMB))
                except Exception:
                    img = None
            self.thumbs[key] = img
        return self.thumbs.get(key)

    def snap(self):
        """The postcard camera went off: tick every target the current view shows."""
        g = self.g
        hits = []
        for tg in TARGETS:
            if tg.key not in self.taken and shows(g, tg):
                hits.append(tg)
        if not hits:
            self._saved_toast()
            return []
        img = self._thumb_now()
        before = len(self.taken)
        for tg in hits:
            self.taken[tg.key] = g.t
            self.thumbs[tg.key] = img
            self._write(tg.key, img)
            g.toasts.push('PHOTO ALBUM', tg.title, 'camera', sound='jackpot', gain=0.45, urgent=True)
        self.snap_t = g.t
        self._ch_key = None                         # check the chances again at once
        if before < 10 <= len(self.taken):
            from .compose import PATTERNS
            name = dict((p[0], p[1]) for p in PATTERNS).get('snapshot', 'SNAPSHOT')
            g.toasts.push('NEW BALLOON', name + ' PATTERN (J)', ui.icon_balloon, sound='quest')
        g.save()
        return [t.key for t in hits]

    def _saved_toast(self):
        """A postcard that ticks no album target still says it was saved (and where it was taken). It
        shows next, after the toast on screen (never cutting it short); presses in a row make one toast."""
        g = self.g
        q = g.toasts
        if any(d.get('title') == 'POSTCARD SAVED' for d in [q.cur or {}] + q.queue):
            return
        from engine.loc_worldmap import LOCS
        lid = getattr(g.mode, 'ID', None)
        where = LOCS[lid]['name'].upper() if lid in LOCS else 'THE WORLD MAP'
        label = getattr(g, 'photo_label', None)         # the folder it went to, so it can be shared
        q.push('POSTCARD SAVED', (label + '  -  J TO SEE IT') if label else (where + '  -  PRESS J TO SEE IT'),
               'camera', sound=None)
        q.queue.insert(0, q.queue.pop())
        q.queue[0]['urgent'] = True         # it answers a key press: no early-session pacing wait (no cut)

    def _thumb_now(self):
        from PIL import Image
        from .compose import compose, KEY
        from engine import Canvas
        fr = self.g.render(0.0)
        fr.hud = Canvas(color=KEY)
        fr.iris = None
        fr.flash = 0.0
        a = compose(fr, 1)
        return np.array(Image.fromarray(a).resize(THUMB, Image.BOX))

    def _write(self, key, img):
        f = self.folder()
        if not f or img is None:
            return
        try:
            from PIL import Image
            os.makedirs(f, exist_ok=True)
            Image.fromarray(img).save(os.path.join(f, key + '.png'))
        except OSError as e:
            print('album photo not saved:', e)

    # -------------------------------------------------------------- photo chances
    def _chances(self, t):
        """The untaken targets a postcard taken now would tick (checked every CHANCE_EVERY steps, and
        at once when the place, area or a photo changes). Also keeps the start of the current chance
        (a gap shorter than CHANCE_GAP steps does not end it)."""
        g = self.g
        key = (t // CHANCE_EVERY, id(g.mode), getattr(g.mode, 'ai', None), getattr(g.mode, 'switch', None) is None,
               getattr(g.mode, 'overview', None))
        if key != self._ch_key:
            self._ch_key = key
            self._ch = [tg for tg in TARGETS if tg.key not in self.taken and shows(g, tg)]
            if self._ch:
                if self._win_t is None or self._seen_t is None or t - self._seen_t > CHANCE_GAP:
                    self._win_t = t
                    self._plate_t = None
                self._seen_t = t
        return self._ch

    def draw_sparkle(self, c, t):
        """HUD layer, every frame: the photo-chance cue (viewfinder brackets, the [camera][P] chip) and
        a ring of sparkles around the frame just after a target is ticked."""
        if self._chances(t):
            self._draw_chance(c, t)
        if self.snap_t is None or not 0 <= t - self.snap_t < 36:
            return
        a = t - self.snap_t
        import math as _m
        for k in range(14):
            ang = k / 14 * 2 * _m.pi + a * 0.05
            r = 20 + a * 3
            x, y = int(W / 2 + _m.cos(ang) * r * 1.6), int(H / 2 + _m.sin(ang) * r)
            if 0 <= x < W - 2 and 0 <= y < H - 2 and (k + a // 3) % 3:
                c.rect(x, y, 2, 2, ('white', 'gold', 'hot_pink')[k % 3])

    def _draw_chance(self, c, t):
        a = t - (self._win_t if self._win_t is not None else t)
        if a < BRACKETS and (a // 16) % 3 != 2:             # slow blink: 32 steps on, 16 off
            _brackets(c, 6)
        kw = ui.keycap_width('P')
        w, h = 3 + 9 + 2 + kw + 3, 13
        x, y = W - 44, H - 32
        c.round_rect(x, y, w, h, ui.INK, r=1)
        c.rect(x + 1, y + 1, w - 2, h - 2, ui.PLATE)
        c.round_rect(x + 2, y + 2, 11, 9, 'cream', r=1)       # a light card behind the camera
        ui.icon(c, 'camera', x + 3, y + 2)
        ui.keycap(c, x + 14, y + 2, 'P', lit=(t // 15) % 2 == 0)
        tu = self.g.tutor                                    # the plate waits for a quiet moment
        if self._plate_t is None and self._wins < PERFECT_N and getattr(tu, 'cur', None) is None \
                and not getattr(tu, 'card', None):
            self._plate_t = t
            self._wins += 1
        if self._plate_t is not None and 0 <= t - self._plate_t < PERFECT:
            s = 'PICTURE PERFECT!'
            pw = text_width(s, 'tiny') + 8
            px, py = x + w - pw, y - 12 + (1 if t - self._plate_t < 4 else 0)
            c.round_rect(px, py, pw, 10, ui.INK, r=1)
            c.rect(px + 1, py + 1, pw - 2, 8, ui.PLATE)
            text(c, px + 4, py + 3, s, 'butter' if (t // 8) % 2 else 'white', font='tiny')


def _brackets(c, inset):
    """Four 5 px viewfinder corners, `inset` px in from the sides; each one moves inwards past any HUD
    plate in its corner (the HUD layer's opaque pixels), the two of a side by the same amount."""
    from .compose import KEY
    a = c.a
    key = np.array(KEY, a.dtype)
    free = np.all(a == key, axis=2)

    def clear(x, y):                                # a 7x7 box around a corner is free of HUD
        y0, x0 = max(0, y - 1), max(0, x - 1)
        return free[y0:y + 6, x0:x + 6].all()
    L, R = inset, W - inset - 5
    top = next((y for y in range(inset, 70) if clear(L, y) and clear(R, y)), None)
    bot = next((y for y in range(H - inset - 5, 100, -1) if clear(L, y) and clear(R, y)), None)
    for x, y, dx, dy in ((L, top, 1, 1), (R + 4, top, -1, 1), (L, None if bot is None else bot + 4, 1, -1),
                         (R + 4, None if bot is None else bot + 4, -1, -1)):
        if y is None:
            continue
        for col, o in ((ui.INK, 1), ('white', 0)):  # an ink shadow, then the white corner
            c.hline(min(x, x + 4 * dx) + o, max(x, x + 4 * dx) + o, y + o, col)
            c.vline(x + o, min(y, y + 4 * dy) + o, max(y, y + 4 * dy) + o, col)
