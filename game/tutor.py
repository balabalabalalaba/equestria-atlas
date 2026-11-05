"""A very light tutorial: a welcome card on a new game, a first gentle request, then small tips that
show up when they fit.

The welcome card (an overlay over the dimmed map, like the pause menu) says what the game is about,
in this order: wander through Equestria and enjoy it; help the places you find; meet friends in the
sky as you travel. It also shows the keys you need. SPACE: let's go. ESC: no tips (they can be turned
back on in the pause menu, TIPS).

The first request: when the card closes on a brand-new journey, Ponyville's empty market cart is
already noticed (quests.seen['ponyville']) and a toast says PONYVILLE NEEDS A HAND. The map's goods
bubble, the gold arrow and the errand plate then lead to Sweet Apple Acres next door: the short apple
loop is everyone's first errand.

Tips: one small plate at a time, above the key hints on the map, and in a place in whichever band
(under the counters or above the street) is clear of the balloon. Each has a moment it shows up
(`when`) and a sign it has been learnt (`done`, read from the game state, so a player who already
does it never sees it). Some are worded for where they show (on the map, in the place that has the
goods, somewhere else). Learnt tips tick off with a green check and are kept in the save (`tutor`).
They never pause or push: a tip shows for ~6 s, may come back once a few minutes later, and then is
considered learnt. The world is there to wander.

One message speaks at a time: a tip waits (and one on screen folds away, its clock paused) while a
toast is up or waiting its turn (it speaks first), a place's discovery plate shows on the map, a
place's atlas card or completion plate is up, a race runs, or the iris is closing or opening.
"""
from dataclasses import dataclass

import numpy as np

from engine import W, H, Canvas, col, mix, seg, ease_out_back, text, text_width

from . import ui
from .mend_art import heart_icon, question_icon

CHECK = 24                      # steps the green check shows when a tip is learnt
DELAY = 45                      # steps between one tip going and the next appearing
SHOW = 180                      # a tip shows ~6 s, then steps aside
SNOOZE = 5400                   # it may come back once, 3 minutes later
MAX_SHOWS = 2                   # after that it counts as learnt: tips never nag
QUIET = 30                      # steps the message lane must have been free before a new tip starts
RESUME = 12                     # ... before a folded-away tip comes back
FOLD = 8                        # steps a tip takes to fold away when another message needs the screen
CLASH = 8                       # steps the balloon may sit under a tip before it moves to the other band
GIFT_DELAY = 36                 # steps between the welcome card closing and the first request's toast
GIFT = 'ponyville'              # the first request: its market cart (the apples are next door)
PLATE_H = 26
TOP_Y = 48                      # a tip's top band in a place / on the map (under counters and plates)
LOW_Y = H - 46                  # the bottom band (above the key hints)
LOW_PROMPT_Y = H - 57           # the map's bottom band while the land plate is up under it
BALLOON_BOX = {'l': (10, 29), 'm': (7, 22), 's': (5, 14), 'xs': (3, 8)}   # half width, height (px)
PLATE_RIM = '#6e2727'
TITLE_COL = '#fff6c0'


def _map(g):
    from .world import MapMode
    return isinstance(g.mode, MapMode) and not g.mode.overview and g.mode.landing is None


def _is_region(m):
    from .region import Region
    return isinstance(m, Region)


def _region(g):
    return _is_region(g.mode) and g.mode.switch is None and g.mode.leaving is None


def _in_place_for(g, n):
    return _region(g) and g.t - g.mode.t_enter > n


def _friend_on_screen(g):
    if not _map(g):
        return False
    m = g.mode
    for k in range(g.flock.visible()):
        if g.flock.hidden(k, g.t):
            continue
        x, y = g.flock.pos(k, g.t)
        if abs(x - m.cx) < 150 and abs(y - m.cy) < 80:
            return True
    return False


def _used_any(g):
    return any(r.used for r in g._regions.values()) or any(s.get('used') for s in g.saved_regions.values())


def _noticed_any(g):
    """The player has noticed a trouble by flying close to its '?' (the first request does not count)."""
    gift = g.tutor.gift
    return any(not (k == GIFT and v == gift) for k, v in g.quests.seen.items())


def _trouble_here(g):
    """In a place whose trouble is still to be found."""
    from .quests import BY_KEY
    q = g.quests
    return g.mode.ID in BY_KEY and g.mode.ID not in q.fixed and g.mode.ID not in q.seen


def _goal_here(g):
    """Can the errand be advanced in this place (the goods wait here, or the trouble they go to is here)?"""
    return any(tg[0] == g.mode.ID for tg in g.quests.targets())


def _goal_on_screen(g):
    """(item) of a goods / delivery bubble the map shows on screen now, or None (then the gold edge
    arrow is up instead)."""
    from .world import icon_centre
    m = g.mode
    ox, oy = m.cx - W / 2, m.cy - H / 2
    for lid, _ai, _x, _y, _kind, item in g.quests.targets():
        x, y = icon_centre(lid)
        sx, sy = x - ox, y - oy - 22
        if 8 <= sx < W - 8 and 8 <= sy < H - 8:
            return item
    return None


def _errand_row(g, away):
    """The fetch / mend tip, worded for where it shows. away: the row in a place without the goal."""
    if _map(g):
        item = _goal_on_screen(g)
        if item is not None:
            return ('[SPACE]', ' LAND AT THE ', '{%s}' % item, ' BUBBLE')
        return ('FOLLOW THE GOLD ARROW ', '>')
    if _region(g) and not _goal_here(g):
        return away
    return None


def _fetch_row(g):
    row = _errand_row(g, ('[ESC]', ' TAKE OFF - FOLLOW THE GOLD ARROW'))
    if row is not None:
        return row
    item = next((tg[5] for tg in g.quests.targets() if tg[0] == getattr(g.mode, 'ID', None)), None)
    if item is not None and _region(g):
        return ('[SPACE]', ' AT THE GLOWING ', '{%s}' % item, ' TO TAKE IT')
    return ('FOLLOW THE GOLD ARROW ', '>')


def _mend_row(g):
    row = _errand_row(g, ('[ESC]', ' TAKE OFF - FOLLOW THE GOLD ARROW'))
    return row if row is not None else ('[SPACE]', ' AT THE BUBBLE TO HELP')


def _playing(g):
    return _map(g) or _region(g)


@dataclass
class Tip:
    key: str
    title: str
    row: object                 # text pieces, or fn(game) -> pieces (worded for where it shows): a
    when: callable              # piece in [brackets] is a key cap ([up] = arrow), '{item}' an item icon
    done: callable
    icon: object = None


TIPS = [
    Tip('fly', 'FLY ANYWHERE YOU LIKE', ('[left]', '[up]', '[down]', '[right]', ' FLY   ', '[SHIFT]', ' FASTER'),
        lambda g: _map(g), lambda g: g.map.moved > 70 or bool(g.visited), ui.icon_balloon),
    Tip('land', 'VISIT A PLACE', ('FLY OVER A PLACE TO FIND IT, THEN ', '[SPACE]', ' TO LAND'),
        lambda g: _map(g) and g.map.moved > 70, lambda g: bool(g.visited), ui.icon_land),
    Tip('explore', 'WANDER AND PLAY', ('[left]', '[up]', '[down]', '[right]', ' LOOK AROUND   ', '[SPACE]',
                                       ' AT A SPARKLE: MAKE IT GO'),
        lambda g: _in_place_for(g, 170), lambda g: _used_any(g), ui.icon_interact),
    Tip('notice', 'THIS PLACE NEEDS A HAND', ('FIND THE ', '?', ' IN THIS PLACE AND FLY CLOSE'),
        lambda g: _in_place_for(g, 300) and _trouble_here(g), _noticed_any, question_icon),
    Tip('fetch', 'HELP THE PLACE', _fetch_row,
        lambda g: _playing(g) and bool(g.quests.open_troubles()) and not g.quests.carrying,
        lambda g: bool(g.quests.carrying) or bool(g.quests.fixed), heart_icon),
    Tip('mend', 'BRING IT HOME', _mend_row,
        lambda g: _playing(g) and bool(g.quests.carrying), lambda g: bool(g.quests.fixed), heart_icon),
    Tip('takeoff', 'BACK TO THE SKY', ('[ESC]', ' TAKES OFF TO THE WORLD MAP'),
        lambda g: _in_place_for(g, 1100), lambda g: _map(g) and bool(g.visited), ui.icon_takeoff),
    Tip('friends', 'BALLOONS ARE FRIENDS', ('FLY ALONGSIDE ONE AND PRESS ', '[SPACE]', ' TO SAY HELLO'),
        lambda g: bool(g.visited) and _friend_on_screen(g), lambda g: bool(g.flock.met), ui.icon_balloon),
    Tip('journal', 'YOUR TRAVEL BOOKS', ('[J]', ' JOURNAL   ', '[M]', ' MAP   ', '[P]', ' POSTCARD'),
        lambda g: _map(g) and (bool(g.flock.met) or bool(g.quests.fixed)),
        lambda g: g.tutor.saw_journal, ui.icon_map),
]


def _lane_busy(g):
    """Is another message using the screen (or about to)? Then tips wait: one message at a time."""
    to = g.toasts
    if to.cur is not None or to.queue:                              # a toast is up, or waiting its turn
        return True
    t = g.t
    if any(0 <= t - t0 < 90 for t0 in g.found_t.values()):         # a discovery plate on the map
        return True
    m = g.mode
    if type(m).__name__ == 'Cutscene':
        return True
    if _is_region(m):
        if m.t_enter <= t < m.t_enter + 150:                        # arriving: the atlas card is up
            return True
        dt = getattr(m, 'done_t', None)
        if dt is not None and 0 <= t - dt < 100:                    # the place's completion plate
            return True
    if getattr(g.races, 'run', None) is not None or getattr(g.flock, 'race', None) is not None:
        return True
    return False


class Tutor:
    def __init__(self, game):
        self.g = game
        self.done = set()               # tips learnt
        self.welcomed = False           # the welcome card has been seen
        self.card = False               # the welcome card is open
        self.card_t = 0
        self.saw_journal = False
        self.old_player = False         # loaded a save from before the tutor (a seasoned player)
        self.gift = None                # step the first request was given (quests.seen[GIFT]), or None
        self.gift_due = None            # step its toast goes up, or None
        self.cur = None                 # (tip key, step it appeared; moved on while the tip waits)
        self.row = ()                   # the current tip's row, worded for where it shows
        self.shown = None               # step the current tip's plate (re)appeared, or None: folded away
        self.y = LOW_Y                  # the current plate's top (screen px) and its band
        self.band = 'low'
        self.clash_n = 0                # steps the balloon has been under the plate
        self.ticked = None              # (tip key, step it was learnt): the green check
        self.tick_row = ()
        self.gone = None                # (tip key, row, y, step): a plate folding away
        self.idle_t = -10 ** 6          # step the last tip went away
        self.busy_t = -10 ** 6          # last step another message used the screen
        self.snooze = {}                # tip key -> step before which it stays away
        self.shows = {}                 # tip key -> times it has been shown

    # -------------------------------------------------------------- save
    def save(self):
        d = dict(done=sorted(self.done), welcomed=self.welcomed, journal=self.saw_journal)
        if self.gift is not None:
            d['gift'] = self.gift
        return d

    def load(self, d, old_player=False):
        self.old_player = old_player
        if d is None:                   # a save from before the tutor: no welcome card for a seasoned player
            self.welcomed = old_player
            return
        keys = {tp.key for tp in TIPS}
        self.done = {k for k in d.get('done', ()) if k in keys}
        self.welcomed = bool(d.get('welcomed', True))
        self.saw_journal = bool(d.get('journal', False))
        gift = d.get('gift')
        self.gift = int(gift) if isinstance(gift, (int, float)) else None

    @property
    def on(self):
        return getattr(self.g.settings, 'tips', True)

    # -------------------------------------------------------------- welcome card
    def want_card(self):
        if self.welcomed or not _map(self.g) or self.g.trans is not None:
            return False
        if not self.on:                 # tips off: no card, but a new journey still gets its first request
            self._close_card()
            return False
        return True

    def open_card(self):
        self.card = True
        self.card_t = self.g.t
        self.g.sfx('pop', 0.35, 0)

    def card_step(self, inp):
        """Returns False when the card closes."""
        if self.g.t - self.card_t < 12:
            return True
        if inp.action:
            self._close_card()
            self.g.sfx('quest', 0.4)
            return False
        if inp.back:
            self._close_card()
            self.g.settings.set('tips', False)
            self.g.settings.save()
            self.g.sfx('pop', 0.35, 1)
            return False
        return True

    def _close_card(self):
        self.welcomed = True
        self.card = False               # (g.t reads the world clock again from here)
        g = self.g
        from .quests import BY_KEY
        q = g.quests
        if not self.old_player and not q.seen and not q.fixed and not g.visited and GIFT in BY_KEY:
            q.seen[GIFT] = g.t          # a new journey: the first request (saved as the card closes)
            self.gift = g.t
            self.gift_due = g.t + GIFT_DELAY

    # -------------------------------------------------------------- tips
    def step(self):
        g = self.g
        t = g.t
        if self.gift_due is not None and t >= self.gift_due and g.trans is None:
            self.gift_due = None
            g.toasts.push('PONYVILLE NEEDS A HAND', 'ITS MARKET CART IS EMPTY', ui.icon_land, sound='idea',
                          gain=0.35)
        if g.trans is not None or self.gift_due is not None or _lane_busy(g):
            self.busy_t = t
            self._wait(t)
            return
        if self.cur is not None:
            key = self.cur[0]
            tp = _BY_KEY[key]
            if tp.done(g):
                self.done.add(key)
                if self.shown is not None:
                    self.ticked = (key, t)
                    self.tick_row = self.row
                    g.sfx('coin', 0.3)
                self._end(t, fold=False)
            elif not tp.when(g) or not self.on or t - self.cur[1] > SHOW:
                self.snooze[key] = t + SNOOZE
                if self.shows.get(key, 0) >= MAX_SHOWS:
                    self.done.add(key)
                self._end(t, fold=t - self.cur[1] <= SHOW)
            elif t - self.busy_t <= RESUME:              # the lane just came free: a breath first
                self._wait(t)
            else:
                self._keep(t, tp)
            return
        if not self.on or t - self.idle_t < DELAY or (self.ticked and t - self.ticked[1] < CHECK):
            return
        if t - self.busy_t <= QUIET:
            return
        for tp in TIPS:
            if tp.key in self.done:
                continue
            if tp.done(g):                                   # already knows it: never shown
                self.done.add(tp.key)
                continue
            if tp.when(g) and t >= self.snooze.get(tp.key, 0):
                self.cur = (tp.key, t)
                self.shown = None
                self.shows[tp.key] = self.shows.get(tp.key, 0) + 1
                self._keep(t, tp)
                return

    def _wait(self, t):
        """Another message has the screen: the tip folds away and its clock waits."""
        if self.cur is None:
            return
        if self.shown is not None:
            self.gone = (self.cur[0], self.row, self.y, t)
            self.shown = None
        self.cur = (self.cur[0], self.cur[1] + 1)

    def _end(self, t, fold):
        if fold and self.shown is not None:
            self.gone = (self.cur[0], self.row, self.y, t)
        self.cur = None
        self.shown = None
        self.idle_t = t

    def _keep(self, t, tp):
        """The tip is up: its row (worded for where it shows) and its band, clear of the balloon."""
        row = _row_of(tp, self.g)
        fresh = self.shown is None
        if row != self.row:
            self.row = row
            fresh = True                                    # the wording changed: it pops in again
        bands = self._bands()
        if fresh:
            self.clash_n = 0
            self.band = min(bands, key=lambda b: self._clash(b, tp))   # (in order on a tie)
            self.y = self._band_y(self.band)
            self.shown = t
            return
        if self.band not in bands:                         # the mode changed under the tip
            self.band = bands[0]
            self.y = self._band_y(self.band)
            self.shown = t
            return
        here = self._clash(self.band, tp)
        if here:                                            # the balloon has flown under it
            self.clash_n += 1
            other = [b for b in bands if b != self.band and self._clash(b, tp) < here]
            if self.clash_n >= CLASH and other:
                self.band, self.clash_n = other[0], 0
                self.y = self._band_y(self.band)
                self.shown = t
                return
        else:
            self.clash_n = 0
        goal = self._band_y(self.band)                      # (the map's land plate came or went)
        if goal != self.y:
            self.y += max(-2, min(2, goal - self.y))

    def _bands(self):
        g = self.g
        if _is_region(g.mode):
            return ['top', 'low']                           # the street band is where troubles stand
        return ['low', 'top']

    def _band_y(self, band):
        if band == 'top':
            return TOP_Y
        m = self.g.mode
        if not _is_region(m) and getattr(m, 'near', None) is not None:
            return LOW_PROMPT_Y
        return LOW_Y

    def _clash(self, band, tp):
        """Would the plate in this band cover the balloon (2) or, in a place, a landmark's name tag (1)?
        0: it is clear. Covering the balloon is the worse of the two."""
        box = _balloon_box(self.g)
        y = self._band_y(band)
        w = _plate_w(tp, self.row, False)
        x0 = (W - w) // 2
        if box is not None:
            bx0, by0, bx1, by1 = box
            if bx0 < x0 + w + 2 and x0 - 2 < bx1 and by0 < y + PLATE_H + 3 and y - 3 < by1:
                return 2
        m = self.g.mode
        if _is_region(m):
            from . import quests
            hit = getattr(quests, '_tags_hit', None)
            if hit is not None:
                try:
                    return 1 if hit(m, self.g.t, (x0 - 1, y - 1, w + 2, PLATE_H + 2)) else 0
                except Exception:
                    return 0
        return 0

    def showing(self):
        return (self.cur is not None and self.shown is not None) or \
            (self.ticked is not None and self.g.t - self.ticked[1] < CHECK)

    # -------------------------------------------------------------- drawing
    def draw(self, c, t, y=None):
        """The tip plate, a learnt tip's green check, or a plate folding away (HUD layer)."""
        if self.cur is not None and self.shown is not None:
            a = t - self.cur[1]
            k = ease_out_back(seg(t - self.shown, 0, 10))
            fade = 0.0
            if a > SHOW - 10:                                       # its time is up: it folds away
                u = seg(a, SHOW - 10, SHOW)
                k, fade = min(k, 1 - u), u
            self._plate(c, t, self.cur[0], self.row, False, k, self.y if y is None else y, fade)
        elif self.ticked is not None and 0 <= t - self.ticked[1] < CHECK:
            u = seg(t - self.ticked[1], CHECK - 8, CHECK)
            self._plate(c, t, self.ticked[0], self.tick_row, True, 1 - u, self.y if y is None else y, u)
        elif self.gone is not None and 0 <= t - self.gone[3] < FOLD:
            key, row, yy, t0 = self.gone
            u = (t - t0) / FOLD
            self._plate(c, t, key, row, False, 1 - u, yy if y is None else y, u)

    def _plate(self, c, t, key, row, learnt, k, y, fade=0.0):
        """The plate at full width k = 1, or narrowing to the centre (k < 1) with its words clipped and
        faded with it, never an empty box."""
        if k <= 0.02:
            return
        tp = _BY_KEY[key]
        lit = not learnt and (t // 12) % 2 == 0
        step = 0 if fade < 0.25 else 1 if fade < 0.6 else 2
        if k < 0.98:                    # narrowing: its clipped words are never shown at full strength
            step = max(step, 1 if k > 0.6 else 2)
        img = _plate_img(key, tuple(row), learnt, lit, step)
        w, h = img.shape[1] - 2, PLATE_H
        x = (W - w) // 2
        if k >= 0.98:
            c.blit(img, x - 1, y - 1)
            return
        cw = max(6, int(w * k))
        cx = x + (w - cw) // 2
        c.round_rect(cx - 1, y - 1, cw + 2, h + 2, PLATE_RIM, r=2)
        c.round_rect(cx, y, cw, h, 'leaf' if learnt else 'gold', r=2)
        if cw > 2:
            c.blit(img[2:h, cx - x + 2:cx - x + cw], cx + 1, y + 1)

    # -------------------------------------------------------------- welcome card drawing
    def draw_card(self, c, t):
        a = t - self.card_t
        k = ease_out_back(seg(a, 0, 12))
        w, h = 262, 150
        x = (W - w) // 2
        y = int((H - h) // 2 + (1 - k) * 40)
        c.round_rect(x - 1, y - 1, w + 2, h + 2, PLATE_RIM, r=3)
        c.round_rect(x, y, w, h, 'gold', r=3)
        c.rect(x + 1, y + 1, w - 2, h - 2, ui.PLATE)
        c.rect(x + 2, y + 2, w - 4, 1, ui.PLATE_HI)
        title = 'WELCOME TO EQUESTRIA!'
        text(c, x + (w - text_width(title, 'normal')) // 2, y + 6, title, TITLE_COL, font='normal', shadow=PLATE_RIM)
        rows = [
            (ui.icon_balloon, 'ENJOY THE WANDERING', 'FLY WHERE YOU LIKE - EVERY PLACE IS ALIVE TO EXPLORE'),
            (heart_icon, 'HELP THE PLACES', 'FIND A ? AND BRING WHAT IT NEEDS: THE WORLD BLOOMS'),
            (_two_balloons, 'MAKE FRIENDS ON THE WAY', 'GREET THE BALLOONS YOU MEET IN THE SKY'),
        ]
        yy = y + 22
        for i, (ic, head, sub) in enumerate(rows):
            if a < 14 + i * 6:
                continue
            ic(c, x + 10, yy + 2)
            text(c, x + 24, yy, head, 'butter' if i == 0 else 'cream', font='normal')
            text(c, x + 24, yy + 10, sub, 'bubblegum', font='tiny')
            yy += 22
        c.hline(x + 10, x + w - 11, y + 92, '#5a3a60')
        keys = [(['left', 'up', 'down', 'right'], 'FLY'), (['SHIFT'], 'FASTER'), (['SPACE'], 'LAND / DO'),
                (['ESC'], 'TAKE OFF')]
        _key_line(c, x + 10, y + 98, keys)
        _key_line(c, x + 10, y + 111, [(['J'], 'JOURNAL'), (['M'], 'MAP'), (['P'], 'POSTCARD'),
                                       (['ESC'], 'ON THE MAP: PAUSE')])
        if a >= 30:
            lit = (t // 12) % 2 == 0
            bx = x + w // 2 - 70
            ui.keycap(c, bx, y + h - 17, 'SPACE', lit=lit)
            text(c, bx + ui.keycap_width('SPACE') + 4, y + h - 15, "LET'S GO", 'butter', font='tiny')
            bx2 = x + w // 2 + 16
            ui.keycap(c, bx2, y + h - 17, 'ESC')
            text(c, bx2 + ui.keycap_width('ESC') + 4, y + h - 15, 'NO TIPS', 'tan', font='tiny')


_BY_KEY = {tp.key: tp for tp in TIPS}


def _row_of(tp, g):
    return tuple(tp.row(g) if callable(tp.row) else tp.row)


def _balloon_box(g):
    """The player's balloon on screen (x0, y0, x1, y1), from the step state, or None."""
    m = g.mode
    try:
        if _is_region(m):
            hw, hh = BALLOON_BOX.get(m.area.balloon, BALLOON_BOX['l'])
            x, y = m.bx - m.cx, m.by - m.cy
        elif hasattr(m, 'alt') and hasattr(m, 'cx'):
            hw, hh = BALLOON_BOX['m']
            x, y = m.bx - (m.cx - W / 2), m.by - m.alt - (m.cy - H / 2)
        else:
            return None
    except AttributeError:
        return None
    return x - hw, y - hh, x + hw, y + 1


# ------------------------------------------------------------------ the plate
def _icon_w(ic):
    """Width of a tip's icon: hint icons report it when called with c = None; toast icons are 9."""
    if ic is None:
        return 0
    try:
        w = ic(None, 0, 0)
    except (AttributeError, TypeError):
        return 9
    return w if isinstance(w, int) else 9


def _geometry(tp, row, learnt):
    """(plate width, title x, check x, row width): the icon at 4, the title 3 px after it, the learnt
    check 4 px after the title, 6 px of plate to the right; the hint row centred under them."""
    iw = _icon_w(tp.icon)
    tx = 4 + iw + 3 if iw else 6
    tw = text_width(tp.title, 'normal')
    rw = _row_width(row)
    head = tx + tw + 6
    w = max(head + (13 if learnt else 0), rw + 10)
    return w, tx, tx + tw + 4, rw


def _plate_w(tp, row, learnt):
    return _geometry(tp, row, learnt)[0]


_IMG = {}


def _plate_img(key, row, learnt, lit, fade):
    """The whole plate (rim included, 1 px margin) as an RGBA array; fade 0..2 steps the words toward
    the plate colour (it folds away). Cached: a plate is the same for every step of its show."""
    k = (key, row, learnt, lit, fade)
    img = _IMG.get(k)
    if img is not None:
        return img
    tp = _BY_KEY[key]
    w, tx, chx, rw = _geometry(tp, row, learnt)
    h = PLATE_H
    cv = Canvas(w + 2, h + 2, color=(255, 0, 254))
    x, y = 1, 1
    cv.round_rect(x - 1, y - 1, w + 2, h + 2, PLATE_RIM, r=2)
    cv.round_rect(x, y, w, h, 'leaf' if learnt else 'gold', r=2)
    cv.rect(x + 1, y + 1, w - 2, h - 2, ui.PLATE)
    cv.rect(x + 2, y + 2, w - 4, 1, ui.PLATE_HI)
    if tp.icon is not None:
        tp.icon(cv, x + 4, y + 3)
    text(cv, x + tx, y + 3, tp.title, TITLE_COL, font='normal', shadow=PLATE_RIM)
    if learnt:
        _check(cv, x + chx, y + 3)
    _draw_row(cv, x + (w - rw) // 2, y + 14, row, lit)
    a = cv.a
    if fade:                                                # the words fade into the plate
        inner = a[y + 1:y + h - 1, x + 1:x + w - 1]
        fill = np.array(col(ui.PLATE), np.float32)
        m = np.any(inner != fill.astype(np.uint8), axis=2)
        kf = (0.45, 0.8)[fade - 1]
        inner[m] = (inner[m] * (1 - kf) + fill * kf).astype(np.uint8)
    rgba = np.dstack([a, np.where(np.all(a == (255, 0, 254), axis=2), 0, 255).astype(np.uint8)])
    if len(_IMG) > 64:
        _IMG.clear()
    _IMG[k] = rgba
    return rgba


def _piece_width(p):
    if p.startswith('[') and p.endswith(']'):
        return ui.keycap_width(p[1:-1]) + 1
    if p.startswith('{') and p.endswith('}'):
        return 9
    if p in ('?', '>'):
        return 8
    return text_width(p, 'tiny')


def _row_width(row):
    return sum(_piece_width(p) for p in row)


def _draw_row(c, x, y, row, lit=False):
    for p in row:
        if p.startswith('[') and p.endswith(']'):
            ui.keycap(c, x, y - 1, p[1:-1], lit=lit)
        elif p.startswith('{') and p.endswith('}'):
            from .items import ITEMS
            spr = ITEMS.get(p[1:-1])
            if spr is not None:
                spr.draw(c, x + 1, y - 1)
        elif p == '?':
            for px, py in ((1, 0), (2, 0), (3, 0), (4, 1), (3, 2), (2, 3), (2, 5)):
                c.px(x + 1 + px, y + py, 'hot_pink')
        elif p == '>':
            c.poly([(x + 1, y), (x + 6, y + 3), (x + 1, y + 6)], 'gold')
        else:
            text(c, x, y + 1, p, 'cream', font='tiny')
        x += _piece_width(p)


def _key_line(c, x, y, groups):
    for keys, label in groups:
        for kk in keys:
            x += ui.keycap(c, x, y, kk) + 1
        text(c, x + 2, y + 2, label, 'cream', font='tiny')
        x += text_width(label, 'tiny') + 9


def _two_balloons(c, x, y):
    from engine.atlas import draw_balloon
    draw_balloon(c, x + 2, y + 8, 0, 'xs', bob=False)
    draw_balloon(c, x + 7, y + 9, 0, 'xs', bob=False)


def _check(c, x, y):
    for px, py in ((0, 4), (1, 5), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)):
        c.rect(x + px, y + py, 2, 2, 'leaf')
