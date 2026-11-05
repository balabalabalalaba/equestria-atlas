"""The atlas journal (J): a parchment book of the 20 places and what the player found there, the
request board, the race records, the photo album, the badges and the balloon wardrobe.

Pages (LEFT / RIGHT turn them; the tabs along the top show where you are):
  ATLAS     the 20 places (below)
  HELP      the places' troubles (game/quests.py): found or not, what each needs and from where
  RACES     the ring races (game/races.py): where, best time, medal, par times, a sketch of the course
  ALBUM     the 20 photo targets (game/album.py) and the photos taken
  PHOTOS    every postcard taken (P), newest first (game/photos.py): a filmstrip and a big print of the
            chosen one; SPACE shows it full screen, where LEFT / RIGHT turn the photos, SPACE shows the
            file in its folder and ESC goes back. J right after a postcard opens the journal here.
  FRIENDS   the 15 balloon friends (game/flock.py): met or not, home, job, status
  BADGES    the 18 milestone badges (game/badges.py): a trophy case with their progress, the next ones
  BALLOONS  the wardrobe (below)

Atlas page: left, every tour stop in route order with its status (undiscovered "? ? ?", discovered on
the map, visited, complete) and its landmark count; right, the selected place: its map icon, name,
atlas card, the landmarks (found ones by name, the rest as "? ? ?"), the 3 hidden stamps and the events
set off there. Places are only described once visited. Up / down choose a place. The footer counts
the places helped toward the grand finale (it starts on take-off once every place is helped); once it
has been seen (or every place is complete), SPACE replays it.

Wardrobe page: the balloon patterns, unlocked by stamps, deliveries, gold medals and album photos;
UP / DOWN choose, SPACE puts one on. J or ESC closes the journal.
"""
import math

import numpy as np

from engine import W, H, Canvas, col, text, text_width, clamp, ease_in, ease_out, smooth
from engine import timeline as TL
from engine import loc_worldmap as LW
from engine import worldmap_icons as WI

from . import ui
from .compose import Frame, PATTERNS, balloon_rgba, need_text
from . import quests as Q, races as RC, album as AL, badges as BG, mend_art as MA, flock as FL
from .items import ITEMS, ITEM_NAMES
from .perks import PERKS

INK_BROWN = 'brew'
SOFT = 'skin_dk'                   # secondary words (4.4:1 on the paper page)
FAINT = 'grey'                     # not found yet, '? ? ?' (4.0:1 on the paper page)
DARK = (INK_BROWN, 'wine', 'ink', 'maroon', 'plum')    # readable on the bubblegum selection bar
ROW0, ROW_H = 28, 6
LEFT = (12, 11, 146, 150)          # x, y, w, h of the left page
RIGHT = (163, 11, 146, 150)
BOOK_H = 160                       # the cover (y 6..166); the key hints sit below it
PAGES = ('atlas', 'requests', 'races', 'album', 'photos', 'friends', 'badges', 'wardrobe')
TITLES = {'atlas': 'ATLAS', 'requests': 'HELP', 'races': 'RACES', 'album': 'ALBUM', 'photos': 'PHOTOS', 'friends': 'FRIENDS',
          'badges': 'BADGES', 'wardrobe': 'BALLOONS'}
RECENT = 450                       # steps after a postcard in which J opens the journal on the PHOTOS page
FILM_Y0, FILM_ROW, FILM_ROWS = 27, 22, 6     # the PHOTOS page's filmstrip
VIEW_FADE = 6                      # steps of the viewer's cross-fades (open, close, next photo)
VIEW_IDLE = 90                     # steps without a key before the viewer's captions slide away
PAGE_FADE = 5                      # steps of the cross-fade when a page turns


def _mask(c, rgba, x, y, color):
    """Paint the opaque pixels of an RGBA sprite array in one colour (a silhouette)."""
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = rgba[y0 - y:y1 - y, x0 - x:x1 - x, 3] > 127
    c.a[y0:y1, x0:x1][m] = col(color)


def _blit_rgba(c, rgba, x, y):
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[:, :, 3] > 127
    c.a[y0:y1, x0:x1][m] = s[:, :, :3][m]


def _sel(color, on):
    """A row's text colour: on the bubblegum selection bar a light one turns wine (4.8:1 there)."""
    return color if not on or color in DARK else 'wine'


class Journal:
    def __init__(self, game):
        self.g = game
        here = getattr(game.mode, 'ID', None)
        self.sel = TL.TOUR.index(here) if here in TL.TOUR else 0
        self.page = 0                   # index in PAGES
        self.wsel = [p[0] for p in PATTERNS].index(game.pattern) if game.pattern in [p[0] for p in PATTERNS] else 0
        keys = [tr.key for tr in Q.TROUBLES]
        self.qsel = keys.index(here) if here in keys else 0
        self.rsel = 0
        self.asel = 0
        self.bsel = 0
        self.fsel = 0
        self.psel = 0                   # the chosen photo (index in g.photos.items, newest first)
        self.ptop = 0                   # the photo in the filmstrip's first row
        self.view = None                # the photo open full screen (an index), or None
        self.view_t = None              # step the viewer opened, closed or turned a photo
        self.fade_from = None           # what that cross-fade starts from: 'page' or a Photo
        self.idle_t = game.t            # the viewer's last key press (its captions hide when idle)
        self.wake_t = None              # step the hidden captions were called back
        game.photos.scan()
        book = game.photos
        if book.items and book.last_t is not None and 0 <= game.t - book.last_t <= RECENT:
            self.page = PAGES.index('photos')
        self.flip_t = game.t
        self.flip_from = None           # the page turned from (it fades out for PAGE_FADE steps)
        self.t0 = game.t
        self.equip_t = None
        self.held = {}                  # direction -> step it went down (for key repeat)

    # -------------------------------------------------------------- input
    def step(self, inp):
        """Returns False when the journal closes."""
        if self.view is not None:
            on = self._view_step(inp)
        else:
            on = self._step(inp)
        if on:
            self.g.photos.pump(*self._photo_wants())
        return on

    def _step(self, inp):
        g = self.g
        if inp.journal or inp.back:
            return False
        go = self._repeat(inp)
        if go['right'] and self.page < len(PAGES) - 1:
            self.flip_from = self.page
            self.page += 1
            self.flip_t = g.t
            g.sfx('pop', 0.3, 2)
            return True
        if go['left'] and self.page > 0:
            self.flip_from = self.page
            self.page -= 1
            self.flip_t = g.t
            g.sfx('pop', 0.3, 1)
            return True
        page = PAGES[self.page]
        d = (1 if go['down'] else 0) - (1 if go['up'] else 0)
        if d:
            g.sfx('tick', 0.25)
        if page == 'atlas':
            self.sel = (self.sel + d) % len(TL.TOUR)
            if inp.action and self._can_replay():
                g.start_finale()
                return False
        elif page == 'requests':
            self.qsel = (self.qsel + d) % len(Q.TROUBLES)
        elif page == 'races':
            self.rsel = (self.rsel + d) % len(RC.COURSES)
        elif page == 'album':
            self.asel = (self.asel + d) % len(AL.TARGETS)
            if inp.action:                              # SPACE: show the postcards folder (game/play.py)
                g.open_photos = True
                g.sfx('pop', 0.3, 0)
        elif page == 'photos':
            n = len(g.photos.items)
            if n:
                self.psel = (min(self.psel, n - 1) + d) % n
                self._scroll()
                if inp.action:
                    self._open_view(self.psel)
            elif inp.action:
                g.sfx('tick', 0.4)
        elif page == 'badges':
            self.bsel = (self.bsel + d) % len(BG.BADGES)
        elif page == 'friends':
            self.fsel = (self.fsel + d) % len(FL.FRIENDS)
        else:
            self.wsel = (self.wsel + d) % len(PATTERNS)
            if inp.action:
                key = PATTERNS[self.wsel][0]
                if key in g.unlocked_patterns():
                    g.pattern = key
                    self.equip_t = g.t
                    g.sfx('coin', 0.5)
                else:
                    g.sfx('tick', 0.4)
        return True

    # -------------------------------------------------------------- the photo viewer
    def _scroll(self):
        n = len(self.g.photos.items)
        self.psel = max(0, min(self.psel, n - 1))
        self.ptop = max(0, min(self.ptop, self.psel, n - FILM_ROWS))
        self.ptop = max(self.ptop, self.psel - FILM_ROWS + 1)

    def _photo_wants(self):
        """(whole pictures, thumbnails) the viewer or the PHOTOS page shows now (read one per step)."""
        items = self.g.photos.items
        if self.view is not None:
            return [items[i] for i in (self.view, self.view + 1, self.view - 1) if 0 <= i < len(items)], ()
        if PAGES[self.page] != 'photos' or not items:
            return (), ()
        self._scroll()
        return (), [items[self.psel]] + items[self.ptop:self.ptop + FILM_ROWS]

    def _open_view(self, i):
        g = self.g
        self.view, self.view_t, self.fade_from = i, g.t, 'page'
        self.idle_t, self.wake_t = g.t, None
        g.sfx('pop', 0.35, 2)

    def _close_view(self):
        g = self.g
        items = g.photos.items
        self.fade_from = items[self.view] if self.view < len(items) else None
        self.psel = self.view
        self._scroll()
        self.view, self.view_t = None, g.t
        g.sfx('pop', 0.35, 1)

    def _view_step(self, inp):
        g = self.g
        if inp.journal:
            return False
        items = g.photos.items
        go = self._repeat(inp)
        if any(go.values()) or inp.action or inp.back:
            if g.t - self.idle_t >= VIEW_IDLE:          # the captions had gone: call them back
                self.wake_t = g.t
            self.idle_t = g.t
        if inp.back or not items:
            self._close_view()
            return True
        self.view = min(self.view, len(items) - 1)
        d = (1 if go['right'] else 0) - (1 if go['left'] else 0)
        if d:
            i = self.view + d
            if 0 <= i < len(items):
                self.fade_from = items[self.view]
                self.view, self.view_t = i, g.t
                g.sfx('pop', 0.3, 2 if d > 0 else 1)
            else:
                g.sfx('tick', 0.4)                      # the first / last photo
        if inp.action:                                  # SPACE: show the file (game/play.py)
            g.open_photos = items[self.view].path
            g.sfx('pop', 0.3, 0)
        return True

    def _photo_canvas(self, p, t):
        """A full-screen canvas with photo p (a soft 'developing' card until it has been read)."""
        book = self.g.photos
        img = book.image(p)
        if img is None:
            c = Canvas(color=_developing(t) if not book.loaded(p) else 'tan')
            ui.ICONS['camera'].scaled(3).draw(c, W // 2 - 13, H // 2 - 14)
            return c
        c = Canvas(color='ink')
        c.a[:img.shape[0], :img.shape[1]] = img
        return c

    def _render_view(self, t):
        g = self.g
        items = g.photos.items
        i = min(self.view, len(items) - 1)
        p = items[i]
        c = self._photo_canvas(p, t)
        k = (t - self.view_t) / VIEW_FADE
        if k < 1 and self.fade_from is not None:
            src = self._spread(t)[0] if self.fade_from == 'page' else self._photo_canvas(self.fade_from, t)
            _blend(c, src, smooth(clamp(k)))
        # the captions: slid away after VIEW_IDLE steps without a key, back in when one is pressed
        dy = int(round(ease_in(clamp((t - self.idle_t - VIEW_IDLE) / 10)) * 40))
        if self.wake_t is not None and t - self.wake_t < 8:
            dy = max(dy, int(round((1 - ease_out(clamp((t - self.wake_t) / 8))) * 40)))
        fr = Frame(c)
        if dy < 40:
            _postcard_caption(c, p, 5, H - 30 + dy)
            s = f'{i + 1} / {len(items)}'
            cw = text_width(s, 'tiny') + 8
            c.round_rect(W - cw - 6, 5 - dy, cw, 11, 'ink', r=2)
            text(c, W - cw - 2, 8 - dy, s, 'cream', font='tiny')
            if i > 0:
                ui.keycap(c, 4 - dy, H // 2 - 5, 'left')
            if i < len(items) - 1:
                ui.keycap(c, W - 14 + dy, H // 2 - 5, 'right')
            keys = [(['left', 'right'], _icon_photo), (['SPACE'], _icon_folder), (['ESC'], None)]
            ui.hint_row(fr.hud, keys, right=W - 6, y=H - 13 + dy)
        return fr

    def _repeat(self, inp):
        """Arrow keys act on the first step they are down, then repeat every 4 steps after a short wait."""
        t = self.g.t
        out = {}
        for d in ('up', 'down', 'left', 'right'):
            if getattr(inp, d):
                t0 = self.held.setdefault(d, t)
                a = t - t0
                out[d] = a == 0 or (a >= 12 and (a - 12) % 4 == 0)
            else:
                self.held.pop(d, None)
                out[d] = False
        return out

    # -------------------------------------------------------------- status
    def status(self, lid):
        """'unknown' | 'seen' | 'visited' | 'complete'."""
        g = self.g
        n, tot, _s, _st = g.place_progress(lid)
        if tot and n >= tot:
            return 'complete'
        if lid in g.visited:
            return 'visited'
        return 'seen' if lid in g.found else 'unknown'

    # -------------------------------------------------------------- render
    def render(self, t):
        if self.view is not None and self.g.photos.items:
            return self._render_view(t)
        c, keys = self._spread(t)
        if self.flip_from is not None and t - self.flip_t < PAGE_FADE:    # a page turning: fade from the last
            cur, self.page = self.page, self.flip_from
            try:
                old = self._spread(t)[0]
            finally:
                self.page = cur
            _blend(c, old, smooth(clamp((t - self.flip_t + 1) / PAGE_FADE)))
        if (self.fade_from not in (None, 'page') and self.view_t is not None and PAGES[self.page] == 'photos'
                and t - self.view_t < VIEW_FADE):         # the viewer just closed: fade from its photo
            _blend(c, self._photo_canvas(self.fade_from, t), smooth(clamp((t - self.view_t) / VIEW_FADE)))
        fr = Frame(c)
        ui.hint_row(fr.hud, keys, right=W - 8, y=H - 11)
        return fr

    def _spread(self, t):
        """The open book (a canvas) and its key hints."""
        c = Canvas(color='#1b1934')
        self._book(c)
        page = PAGES[self.page]
        if page == 'atlas':
            self._left(c, t)
            self._right(c, t)
        elif page == 'wardrobe':
            self._wardrobe(c, t)
        else:
            getattr(self, '_page_' + page)(c, t)
        self._tabs(c, t)
        keys = [(['left', 'right'], _icon_pages), (['up', 'down'], None)]
        if page == 'wardrobe':
            keys.append((['SPACE'], ui.icon_balloon))
        elif page == 'album' and getattr(self.g, 'photo_label', None):
            keys.append((['SPACE'], _icon_folder))
        elif page == 'photos' and self.g.photos.items:
            keys.append((['SPACE'], _icon_photo))
        keys.append((['J'], None))
        return c, keys

    def _tabs(self, c, t):
        """Bookmark tabs along the top of the cover, one per page; the open one stands up."""
        n = len(PAGES)
        tw, gap = 22, 4
        x0 = (W - (n * tw + (n - 1) * gap)) // 2
        for i, p in enumerate(PAGES):
            x = x0 + i * (tw + gap)
            on = i == self.page
            y = 0 if on else 2
            c.round_rect(x, y, tw, 12 - y, 'maroon', r=1)
            c.rect(x + 1, y + 1, tw - 2, 10 - y, 'bubblegum' if on else 'clay')
            _TAB_ICONS[p](c, x + (tw - 9) // 2, y + 1)
            if on and (t // 10) % 2:
                c.px(x + tw - 3, y + 2, 'white')

    def _header(self, c, x0, y0, w, title, s=None, icon=None):
        text(c, x0 + 6, y0 + 4, title, INK_BROWN, font='normal')
        if s:
            sx = x0 + w - 6 - text_width(s, 'tiny')
            text(c, sx, y0 + 6, s, INK_BROWN, font='tiny')
            if icon is not None:
                icon(c, sx - 11, y0 + 4)

    def _list_row(self, c, x0, w, y, i, sel, t):
        if i == sel:
            c.rect(x0 + 2, y - 1, w - 4, 7, 'bubblegum')
            if (t // 8) % 2:
                c.px(x0 + 3, y + 2, 'hot_pink')

    # -------------------------------------------------------------- requests page
    def _page_requests(self, c, t):
        """HELP: the 20 places' troubles: unknown (?), noticed (what it needs, where from), mended."""
        g = self.g
        qs = g.quests
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'HELP', f'{len(qs.fixed)}/{len(Q.TROUBLES)}',
                     lambda c_, x, y: MA.heart_icon(c_, x, y - 1))
        for i, tr in enumerate(Q.TROUBLES):
            y = 24 + i * 6
            self._list_row(c, x0, w, y, i, self.qsel, t)
            st = qs.status(tr.key)
            on = i == self.qsel
            known = tr.key in g.found or tr.key in g.visited or st != 'unknown'   # (a request names its place)
            name = Q.place_name(tr.key) if known else '? ? ?'
            if st == 'fixed':
                MA.tiny_heart(c, x0 + 7, y + 1)
                text(c, x0 + 14, y, name, 'wine', font='tiny')
            elif st == 'seen':
                c.rect(x0 + 8, y, 1, 3, 'hot_pink')
                c.px(x0 + 8, y + 4, 'hot_pink')
                text(c, x0 + 14, y, name, INK_BROWN, font='tiny')
                spr = ITEMS[tr.item]
                if g.quests.carrying == tr.item and (t // 8) % 2:
                    c.rect(x0 + w - 16, y - 1, 9, 7, 'butter')
                spr.sub(0, 1, 7, 5).draw(c, x0 + w - 15, y)
            else:
                c.px(x0 + 8, y + 2, FAINT)
                text(c, x0 + 14, y, name, _sel(SOFT if known else FAINT, on), font='tiny')
        # right page: the selected place's trouble
        tr = Q.TROUBLES[self.qsel]
        st = qs.status(tr.key)
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        pw, ph = MA.SIZE[tr.kind]
        box = Canvas(pw + 8, ph + 8, color='paper')
        if st == 'unknown':
            MA.draw_prop(box, tr.kind, (pw + 8) // 2, ph + 6, t, False, tr.item)
            _mask(c, _canvas_rgba(box), cx - pw - 8, y0 + 6, 'sand')
            MA.question(c, cx, y0 + 10 + ph + 8, t)
            msg = ['NOT FOUND YET', 'EXPLORE ' + (Q.place_name(tr.key) if tr.key in g.found else 'THE MAP'),
                   'AND LOOK FOR ?']
            for k, line in enumerate(msg):
                text(c, cx - text_width(line, 'tiny') // 2, y0 + 2 * ph + 30 + k * 8, line, SOFT, font='tiny')
            return
        MA.draw_prop(box, tr.kind, (pw + 8) // 2, ph + 6, t, st == 'fixed', tr.item)
        big = np.repeat(np.repeat(box.a, 2, 0), 2, 1)
        bx, by = cx - big.shape[1] // 2, y0 + 4
        c.a[by:by + big.shape[0], bx:bx + big.shape[1]] = big
        y = by + big.shape[0] + 3
        name = Q.place_name(tr.key)
        text(c, cx - text_width(name, 'normal') // 2, y, name, INK_BROWN, font='normal')
        y += 11
        lines = _wrap(tr.done if st == 'fixed' else tr.title, 34)
        for k, line in enumerate(lines):
            text(c, x0 + 7, y + k * 7, line, 'wine' if st == 'fixed' else SOFT, font='tiny')
        y += 7 * len(lines) + 4
        c.hline(x0 + 6, x0 + w - 7, y - 3, 'sand')
        if st == 'fixed':
            MA.heart_icon(c, x0 + 6, y)
            text(c, x0 + 18, y + 2, 'HELPED - THANK YOU!', INK_BROWN, font='tiny')
            return
        src = Q.SOURCES[tr.item]
        ITEMS[tr.item].draw(c, x0 + 7, y)
        text(c, x0 + 18, y + 1, 'NEEDS ' + ITEM_NAMES[tr.item], INK_BROWN, font='tiny')
        text(c, x0 + 18, y + 8, 'FROM ' + Q.place_name(src[0]), SOFT, font='tiny')
        yb = RIGHT[1] + RIGHT[3] - 10
        if g.quests.carrying == tr.item:
            text(c, x0 + 7, yb, 'IT IS IN YOUR BASKET: FLY THERE!', 'wine', font='tiny')
        elif g.quests.carrying:
            text(c, x0 + 7, yb, 'YOUR BASKET IS FULL', SOFT, font='tiny')

    # -------------------------------------------------------------- races page
    def _page_races(self, c, t):
        g = self.g
        rc = g.races
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'RACES', f'{rc.golds()}/{len(RC.COURSES)}',
                     lambda c_, x, y: RC.draw_medal(c_, x + 1, y - 1, 'gold'))
        for i, co in enumerate(RC.COURSES):
            y = 28 + i * 20
            if i == self.rsel:
                c.rect(x0 + 2, y - 2, w - 4, 19, 'bubblegum')
            m = rc.medal(co.key)
            if m in RC.MEDALS:
                RC.draw_medal(c, x0 + 6, y, m)
            else:
                RC.draw_medal(c, x0 + 6, y, 'none')
            on = i == self.rsel
            text(c, x0 + 19, y, co.name, INK_BROWN, font='tiny')
            text(c, x0 + 19, y + 7, co.hint, _sel(SOFT, on), font='tiny')
            b = rc.best.get(co.key)
            s = RC.fmt(b) if b is not None else '-:--.-'
            text(c, x0 + w - 6 - text_width(s, 'tiny'), y, s, 'wine' if b is not None else _sel(FAINT, on), font='tiny')
        # right page: the selected course
        co = RC.COURSES[self.rsel]
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        text(c, cx - text_width(co.name, 'normal') // 2, y0 + 5, co.name, INK_BROWN, font='normal')
        text(c, cx - text_width(co.hint, 'tiny') // 2, y0 + 16, co.hint, SOFT, font='tiny')
        # a sketch of the course: the gates joined by a dotted line, fitted in a box
        bx, by, bw, bh = x0 + 12, y0 + 26, w - 24, 64
        c.rect(bx - 2, by - 2, bw + 4, bh + 4, 'cream')
        c.frame(bx - 2, by - 2, bw + 4, bh + 4, 'sand')
        xs = [p[0] for p in co.gates]
        ys = [p[1] for p in co.gates]
        sx = (bw - 12) / max(1, max(xs) - min(xs))
        sy = (bh - 12) / max(1, max(ys) - min(ys))
        k = min(sx, sy)
        ox = bx + 6 + ((bw - 12) - (max(xs) - min(xs)) * k) / 2
        oy = by + 6 + ((bh - 12) - (max(ys) - min(ys)) * k) / 2
        pts = [(int(ox + (x - min(xs)) * k), int(oy + (y - min(ys)) * k)) for x, y in co.gates]
        for (ax, ay), (bx2, by2) in zip(pts, pts[1:]):
            n = max(1, int(math.hypot(bx2 - ax, by2 - ay) / 3))
            for j in range(n):
                if (j + t // 6) % 2 == 0:
                    c.px(int(ax + (bx2 - ax) * j / n), int(ay + (by2 - ay) * j / n), 'hot_pink')
        for i, (px, py) in enumerate(pts):
            if i == 0:
                RC.FLAG.draw(c, px - 1, py - 6)
            else:
                c.ellipse(px, py, 2, 3, 'gold' if i < len(pts) - 1 else 'wine', fill=False)
        y = by + bh + 8
        for m, p in zip(RC.MEDALS, co.par):
            RC.draw_medal(c, x0 + 10 + RC.MEDALS.index(m) * 44, y, m)
            text(c, x0 + 22 + RC.MEDALS.index(m) * 44, y + 3, RC.fmt(int(p * 30)), INK_BROWN, font='tiny')
        b = rc.best.get(co.key)
        y += 18
        msg = ('BEST ' + RC.fmt(b)) if b is not None else 'FLY THROUGH THE START GATE'
        text(c, cx - text_width(msg, 'tiny') // 2, y, msg, 'wine', font='tiny')

    # -------------------------------------------------------------- album page
    def _page_album(self, c, t):
        g = self.g
        al = g.album
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'ALBUM', f'{len(al.taken)}/{len(AL.TARGETS)}',
                     lambda c_, x, y: ui.ICONS['camera'].draw(c_, x, y - 1))
        for i, tg in enumerate(AL.TARGETS):
            y = 24 + i * 6
            self._list_row(c, x0, w, y, i, self.asel, t)
            got = tg.key in al.taken
            if got:
                c.px(x0 + 7, y + 2, 'gold')
                c.px(x0 + 8, y + 3, 'gold')
                c.px(x0 + 9, y + 1, 'gold')
                c.px(x0 + 10, y, 'gold')
            else:
                c.rect(x0 + 7, y + 1, 3, 3, 'sand')
            text(c, x0 + 14, y, tg.title, INK_BROWN if got else _sel(SOFT, i == self.asel), font='tiny')
        tg = AL.TARGETS[self.asel]
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        # a polaroid: the photo (2x the 64 x 36 thumbnail) or an empty frame with a camera
        pw, ph = 128 + 8, 72 + 18
        px, py = cx - pw // 2, y0 + 8
        c.rect(px + 2, py + 2, pw, ph, 'tan')
        c.rect(px, py, pw, ph, 'white')
        img = al.thumb(tg.key)
        if img is not None:
            big = img.repeat(2, 0).repeat(2, 1)
            c.a[py + 4:py + 4 + 72, px + 4:px + 4 + 128] = big
        else:
            c.rect(px + 4, py + 4, 128, 72, 'cream')
            ui.ICONS['camera'].scaled(2).draw(c, cx - 9, py + 30)
        title = tg.title
        text(c, cx - text_width(title, 'tiny') // 2, py + 79, title, INK_BROWN, font='tiny')
        y = py + ph + 8
        where = 'THE WORLD MAP' if tg.place == 'map' else LW.LOCS[tg.place]['name'].upper()
        text(c, x0 + 7, y, 'WHERE', 'wine', font='tiny')
        text(c, x0 + 33, y, where if tg.place == 'map' or tg.place in g.found else '? ? ?', INK_BROWN, font='tiny')
        text(c, x0 + 7, y + 9, 'TIP', 'wine', font='tiny')
        for k, line in enumerate(_wrap(tg.hint, 28)):
            text(c, x0 + 33, y + 9 + k * 7, line, SOFT, font='tiny')
        if tg.key in al.taken:
            ui.star(c, px + pw - 10, py + 2)
        label = getattr(g, 'photo_label', None)       # where the postcards (P) are saved
        if label:
            x0, y0, w, _h = LEFT
            y = y0 + 136
            c.hline(x0 + 6, x0 + w - 6, y - 2, 'sand')
            text(c, x0 + 7, y, 'POSTCARDS (P) ARE IN', 'wine', font='tiny')
            text(c, x0 + 7, y + 7, label, INK_BROWN, font='tiny')

    # -------------------------------------------------------------- photos page
    def _page_photos(self, c, t):
        g = self.g
        book = g.photos
        items = book.items
        self._scroll()
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'PHOTOS', str(len(items)), lambda c_, x, y: ui.ICONS['camera'].draw(c_, x, y - 1))
        for r in range(FILM_ROWS):                      # the filmstrip: a small print, where and when
            i = self.ptop + r
            y = FILM_Y0 + r * FILM_ROW
            fx = x0 + 6
            if i >= len(items):
                if not items and r < 3:                 # empty frames waiting for postcards
                    c.frame(fx, y, 34, 20, 'sand')
                continue
            p = items[i]
            on = i == self.psel
            if on:
                c.rect(x0 + 2, y - 2, w - 4, FILM_ROW - 1, 'bubblegum')
                if (t // 8) % 2:
                    c.px(x0 + 3, y + 8, 'hot_pink')
            fy = y - (1 if on else 0)
            c.rect(fx + 1, fy + 1, 34, 20, 'tan')
            c.rect(fx, fy, 34, 20, 'white')
            th = book.thumb(p, 32, 18)
            if th is not None:
                c.a[fy + 1:fy + 19, fx + 1:fx + 33] = th
            else:
                c.rect(fx + 1, fy + 1, 32, 18, _developing(t + i * 5) if not book.loaded(p) else 'sand')
            tx = fx + 40
            text(c, tx, y + 3, p.name, _sel(INK_BROWN, on), font='tiny')
            text(c, tx, y + 11, p.date(), _sel(SOFT, on), font='tiny')
            if p.path in book._new:
                ui.new_tag(c, x0 + w - 8 - ui.NEW_TAG_W, y + 10, t)
        n = len(items)
        if n > FILM_ROWS:                               # a scrollbar along the page edge
            y, L = FILM_Y0 - 1, FILM_ROWS * FILM_ROW - 2
            bl = max(6, L * FILM_ROWS // n)
            by = y + (L - bl) * self.ptop // (n - FILM_ROWS)
            c.rect(x0 + w - 4, y, 2, L, 'sand')
            c.rect(x0 + w - 4, by, 2, bl, 'clay')

        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        pw, ph = 128 + 8, 72 + 18                       # the chosen photo as a big print
        px, py = cx - pw // 2, y0 + 8
        c.rect(px + 2, py + 2, pw, ph, 'tan')
        c.rect(px, py, pw, ph, 'white')
        if not items:
            c.rect(px + 4, py + 4, 128, 72, 'cream')
            ui.ICONS['camera'].scaled(2).draw(c, cx - 9, py + 30)
            y = py + ph + 8
            for k, line in enumerate(_wrap('NO POSTCARDS YET. PRESS P ANYWHERE ON YOUR TRIP TO TAKE ONE.', 30)):
                text(c, x0 + 7, y + k * 8, line, SOFT, font='tiny')
        else:
            p = items[self.psel]
            img = book.thumb(p, 128, 72)
            if img is not None:
                c.a[py + 4:py + 76, px + 4:px + 132] = img
            else:
                c.rect(px + 4, py + 4, 128, 72, _developing(t) if not book.loaded(p) else 'sand')
                ui.ICONS['camera'].scaled(2).draw(c, cx - 9, py + 30)
            text(c, cx - text_width(p.name, 'tiny') // 2, py + 79, p.name, INK_BROWN, font='tiny')
            y = py + ph + 8
            text(c, x0 + 7, y, 'WHEN', 'wine', font='tiny')
            text(c, x0 + 33, y, p.date() + '   ' + p.clock(), INK_BROWN, font='tiny')
            kw = ui.keycap(c, x0 + 7, y + 10, 'SPACE')
            text(c, x0 + 7 + kw + 4, y + 12, 'FULL SCREEN', 'wine', font='tiny')
        label = getattr(g, 'photo_label', None)       # where the postcards are saved
        if label:
            y = y0 + 136
            c.hline(x0 + 6, x0 + w - 6, y - 2, 'sand')
            text(c, x0 + 7, y, 'SAVED IN', 'wine', font='tiny')
            text(c, x0 + 7, y + 7, label, INK_BROWN, font='tiny')

    # -------------------------------------------------------------- friends page
    def _page_friends(self, c, t):
        """FRIENDS: the 15 balloons of the sky (game/flock.py)."""
        g = self.g
        fl = g.flock
        from .world import TRAFFIC
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'FRIENDS', f'{len(fl.met)}/{len(FL.FRIENDS)}',
                     lambda c_, x, y: ui.icon_balloon(c_, x + 2, y))
        vis = fl.visible()
        for k, (name, job, item) in enumerate(FL.FRIENDS):
            y = 24 + k * 8
            if k == self.fsel:
                c.rect(x0 + 2, y - 1, w - 4, 8, 'bubblegum')
            met = k in fl.met
            _mini_balloon(c, x0 + 8, y + 3, TRAFFIC[k][1], met)
            text(c, x0 + 15, y + 1, name if met else '? ? ?', INK_BROWN if met else _sel(FAINT, k == self.fsel), font='tiny')
            if met:
                ic = fl.bubble_icon(k)
                spr = ui.ICONS[ic] if isinstance(ic, str) else ic
                spr.draw(c, x0 + w - 16, y - 1)
                if k in fl.beaten:
                    c.px(x0 + w - 20, y + 2, 'gold')
                    c.px(x0 + w - 21, y + 3, 'red')
            elif k >= vis:
                c.rect(x0 + w - 13, y + 2, 4, 3, 'sand')
        k = self.fsel
        name, job, item = FL.FRIENDS[k]
        lid, pat, size, _ph = TRAFFIC[k]
        met = k in fl.met
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        big = balloon_rgba('l', pat).repeat(2, 0).repeat(2, 1)
        bob = int(round(np.sin(t * 0.09 + k) * 2))
        bx, by = cx - big.shape[1] // 2, y0 + 6 + bob
        if met:
            _blit_rgba(c, big, bx, by)
            if job == 'carrier' and fl.has_goods(k):
                ITEMS[item].scaled(2).draw(c, cx - 7, y0 + 6 + big.shape[0] + 1 + bob)
        else:
            _mask(c, big, bx, by, 'sand')
        y = y0 + 6 + big.shape[0] + 18
        title = name if met else '? ? ?'
        text(c, cx - text_width(title, 'normal') // 2, y, title, INK_BROWN, font='normal')
        y += 12
        home = LW.LOCS[lid]['name'].upper()
        if not met:
            lines = (['FLIES NEAR ' + home, 'FLY ALONGSIDE AND PRESS SPACE'] if k < vis else
                     ['NOT IN THE SKY YET', 'HELP MORE PLACES TO SEE IT'])
        else:
            lines = ['HOME: ' + home, FL.JOB_TEXT[job] + (ITEM_NAMES[item] if job == 'carrier' else '')]
            if job == 'carrier':
                lines.append('HAS GOODS TO SHARE' if fl.has_goods(k) else 'RESTOCKING')
            elif job == 'racer':
                lines.append('BEATEN - A RIBBON!' if k in fl.beaten else 'NOT BEATEN YET')
            else:
                lines.append('ASK IT TO LEAD YOU')
        for j, line in enumerate(lines):
            text(c, cx - text_width(line, 'tiny') // 2, y + j * 8, line, SOFT if j else 'wine', font='tiny')

    # -------------------------------------------------------------- badges page
    def _page_badges(self, c, t):
        """BADGES: left, the trophy case (3 columns of 6, in BG.BADGES order: exploring, helping and
        friends, the sky), each medallion with its count and bar; right, the selected badge large,
        and the next few badges closest to being earned."""
        g = self.g
        bd = g.badges
        n = len(BG.BADGES)
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'BADGES', f'{len(bd.earned)}/{n}',
                     lambda c_, x, y: BG.ICON.draw(c_, x + 1, y))
        rows = (n + 2) // 3
        for i, b in enumerate(BG.BADGES):
            cx0, cy0 = x0 + 4 + (i // rows) * 46, y0 + 19 + (i % rows) * 21
            on = i == self.bsel
            earned = b.key in bd.earned
            if on:
                c.round_rect(cx0, cy0, 45, 20, 'bubblegum', r=2)
            BG.badge_sprite(b, earned).draw(c, cx0 + 1, cy0 + 1)
            if bd.fresh(b.key) and (t // 6) % 2:                 # earned this session: a twinkle
                c.px(cx0 + 16, cy0 + 3, 'white')
                c.px(cx0 + 17, cy0 + 2, 'butter')
            k, tot = bd.progress(b)
            s = f'{k}%' if b.key in BG.PCT else f'{k}/{tot}'
            text(c, cx0 + 21, cy0 + 4, s, _sel('wine' if earned else SOFT, on), font='tiny')
            self._bar(c, cx0 + 21, cy0 + 11, 21, 3, k, tot, earned)
        self._badge_detail(c, BG.BADGES[self.bsel], t)

    def _bar(self, c, x, y, bw, bh, k, tot, earned):
        c.rect(x, y, bw, bh, 'sand')
        fill = int(bw * k / max(1, tot))
        if fill:
            c.rect(x, y, fill, bh, 'gold' if earned else 'hot_pink')
            c.hline(x, x + fill - 1, y, 'butter' if earned else 'pink')

    def _badge_detail(self, c, b, t):
        bd = self.g.badges
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        earned = b.key in bd.earned
        k, tot = bd.progress(b)
        bx, by = cx - 19, y0 + 6
        if earned:                                               # two ribbon tails under it
            for dx, cc, dk in ((-10, 'crimson', 'maroon'), (4, 'royal', 'navy')):
                c.rect(cx + dx, by + 30, 6, 10, cc)
                c.vline(cx + dx + 5, by + 30, by + 39, dk)
                c.px(cx + dx + 2, by + 39, 'paper')
                c.px(cx + dx + 3, by + 39, 'paper')
        BG.badge_sprite(b, earned, k=2).draw(c, bx, by)          # the medallion at 2x
        name = b.name
        text(c, cx - text_width(name, 'normal') // 2, y0 + 49, name, 'wine' if earned else INK_BROWN, font='normal')
        text(c, cx - text_width(b.desc, 'tiny') // 2, y0 + 59, b.desc, SOFT, font='tiny')
        s = f'{k}%' if b.key in BG.PCT else f'{k}/{tot}'
        bw = 96
        bx = cx - (bw + 4 + text_width(s, 'tiny')) // 2
        self._bar(c, bx, y0 + 68, bw, 4, k, tot, earned)
        text(c, bx + bw + 4, y0 + 68, s, 'wine' if earned else INK_BROWN, font='tiny')
        if earned and bd.fresh(b.key):
            ui.new_tag(c, cx + 16, y0 + 8, t)
        # next up: the unearned badges with the fewest steps left (scaled: one race < 98 hotspots)
        todo = [(bd.progress(o), i, o) for i, o in enumerate(BG.BADGES) if o.key not in bd.earned]
        todo.sort(key=lambda e: ((e[0][1] - e[0][0]) / math.sqrt(max(1, e[0][1])), e[1]))   # few steps left
        y = y0 + 80
        for xx in range(x0 + 6, x0 + w - 6, 2):
            c.px(xx, y, 'tan')
        if not todo:
            ui.star(c, cx - 3, y + 10)
            return
        text(c, x0 + 6, y + 4, 'NEXT UP', 'wine', font='tiny')
        for j, ((kk, tt), _i, o) in enumerate(todo[:3]):
            yy = y + 13 + j * 18
            BG.badge_sprite(o, False, small=True).draw(c, x0 + 3, yy - 3)
            text(c, x0 + 21, yy, o.name, INK_BROWN, font='tiny')
            s = f'{kk}%' if o.key in BG.PCT else f'{kk}/{tt}'
            text(c, x0 + w - 6 - text_width(s, 'tiny'), yy, s, SOFT, font='tiny')
            text(c, x0 + 21, yy + 6, o.desc, SOFT, font='tiny')
            self._bar(c, x0 + 21, yy + 12, w - 27, 2, kk, tt, False)

    # -------------------------------------------------------------- wardrobe page
    def _wardrobe(self, c, t):
        g = self.g
        unlocked = set(g.unlocked_patterns())
        key, name, need, _spec = PATTERNS[self.wsel]
        # left page: the chosen balloon, large, drifting
        x0, y0, w, _h = LEFT
        self._header(c, x0, y0, w, 'BALLOONS', f'{len(unlocked)}/{len(PATTERNS)}',
                     lambda c_, x, y: ui.icon_balloon(c_, x + 2, y))
        big = balloon_rgba('l', key)
        big = big.repeat(3, 0).repeat(3, 1)                 # 3x, nearest
        bob = int(round(np.sin(t * 0.09) * 2))
        bx, by = x0 + w // 2 - big.shape[1] // 2, y0 + 22 + bob
        if key in unlocked:
            _blit_rgba(c, big, bx, by)
        else:
            _mask(c, big, bx, by, 'sand')
        text(c, x0 + w // 2 - text_width(name, 'normal') // 2, y0 + 106, name, INK_BROWN, font='normal')
        if key == g.pattern:
            msg, colr = 'ON YOUR BALLOON', 'wine'
        elif key in unlocked:
            msg, colr = 'SPACE TO WEAR IT', SOFT
        else:
            msg, colr = need_text(need), SOFT
        text(c, x0 + w // 2 - text_width(msg, 'tiny') // 2, y0 + 116, msg, colr, font='tiny')
        perk = PERKS.get(key)                           # what it does (game/perks.py)
        if perk is not None:
            c.hline(x0 + 10, x0 + w - 11, y0 + 124, 'sand')
            pt = 'PERK: ' + perk[0]
            text(c, x0 + w // 2 - text_width(pt, 'tiny') // 2, y0 + 128, pt, 'wine', font='tiny')
            for k, line in enumerate(_wrap(perk[1], 34)[:2]):
                text(c, x0 + w // 2 - text_width(line, 'tiny') // 2, y0 + 136 + k * 7, line,
                     INK_BROWN if key in unlocked else SOFT, font='tiny')
        if self.equip_t is not None and 0 <= t - self.equip_t < 30:
            ui.burst(c, t, self.equip_t, x0 + w // 2, y0 + 60, ('gold', 'butter', 'hot_pink', 'white'), n=30,
                     seed=7, power=1.1, life=30)
        # right page: every pattern (4 columns x 4 rows, column by column)
        x1, y1, w1, _h1 = RIGHT
        for i, (k2, _n2, need2, _s2) in enumerate(PATTERNS):
            cc, rr = divmod(i, 4)
            cx, cy = x1 + 4 + cc * 35, y1 + 8 + rr * 35
            if i == self.wsel:
                c.round_rect(cx - 1, cy - 3, 34, 29, 'bubblegum', r=2)
            spr = balloon_rgba('m', k2)
            sx = cx + 14 - spr.shape[1] // 2
            if k2 in unlocked:
                _blit_rgba(c, spr, sx, cy + 1)
            else:
                _mask(c, spr, sx, cy + 1, 'sand')
                c.rect(sx + 3, cy + 9, 7, 6, 'clay')                 # a little padlock
                c.rect(sx + 4, cy + 6, 5, 4, 'clay')
                c.rect(sx + 5, cy + 7, 3, 3, 'sand')
            if k2 == g.pattern:
                ui.star(c, cx + 24, cy - 1)

    def _book(self, c):
        c.round_rect(6, 6, 308, BOOK_H, 'maroon', r=3)
        c.round_rect(7, 7, 306, BOOK_H - 2, 'wine', r=3)
        yb = 6 + BOOK_H - 6
        for x, y in ((8, 8), (W - 12, 8), (8, yb), (W - 12, yb)):               # gold corners
            c.rect(x, y, 4, 4, 'gold')
            c.px(x + 1, y + 1, 'butter')
        for (x, y, w, h) in (LEFT, RIGHT):
            c.rect(x, y, w, h, 'tan')
            c.rect(x + 1, y + 1, w - 2, h - 2, 'paper')
            c.hline(x + 2, x + w - 3, y + 1, 'cream')
        for i, cc in enumerate(('clay', 'tan', 'sand', 'tan', 'clay')):           # the spine
            c.vline(158 + i, 11, 11 + LEFT[3] - 1, cc)

    def _left(self, c, t):
        g = self.g
        x0, y0, w, _h = LEFT
        text(c, x0 + 6, y0 + 4, 'ATLAS', INK_BROWN, font='normal')
        n_vis = sum(1 for lid in TL.TOUR if lid in g.visited)
        n_st = sum(g.place_progress(lid)[2] for lid in TL.TOUR)
        s = f'{n_vis}/20'
        sx = x0 + w - 6 - text_width(s, 'tiny')
        text(c, sx, y0 + 6, s, INK_BROWN, font='tiny')
        ui._ARROWS['down'].recolor({col('ink'): col('hot_pink')}).draw(c, sx - 8, y0 + 7)
        s2 = f'{n_st}/60'
        sx2 = sx - 14 - text_width(s2, 'tiny')
        text(c, sx2, y0 + 6, s2, INK_BROWN, font='tiny')
        ui._STAMP.draw(c, sx2 - 11, y0 + 4)
        hx = x0 + 6 + text_width('ATLAS', 'normal') + 3
        if g.all_complete():                                 # every place complete: a gold star
            ui.star(c, hx, y0 + 4)
            hx += 9
        pct = int(g.fog.explored() * 100)                    # the fog of war: how much is charted
        ui.icon_map(c, hx + 2, y0 + 4)
        text(c, hx + 13, y0 + 6, f'{pct}%', INK_BROWN, font='tiny')
        self._finale_line(c, x0, y0 + LEFT[3] - 11, t)
        for i, lid in enumerate(TL.TOUR):
            y = ROW0 + i * ROW_H
            st = self.status(lid)
            if i == self.sel:
                c.rect(x0 + 2, y - 1, w - 4, ROW_H, 'bubblegum')
                if (t // 8) % 2:
                    c.px(x0 + 3, y + 2, 'hot_pink')
            self._mark(c, x0 + 7, y, st)
            name = LW.LOCS[lid]['name'].upper() if st != 'unknown' else '? ? ?'
            on = i == self.sel
            color = {'complete': 'wine', 'visited': INK_BROWN, 'seen': SOFT, 'unknown': FAINT}[st]
            text(c, x0 + 15, y, name, _sel(color, on), font='tiny')
            n, tot, _s, _st = g.place_progress(lid)
            if tot:
                s = f'{n}/{tot}'
                text(c, x0 + w - 6 - text_width(s, 'tiny'), y, s, 'wine' if n == tot else _sel(SOFT, on), font='tiny')

    def _finale_line(self, c, x0, y, t):
        """The atlas footer: how far the grand finale is (it comes when every place is helped), or,
        once it has been seen, SPACE to see it again."""
        g = self.g
        n, tot = len(g.quests.fixed), len(Q.TROUBLES)
        MA.heart_icon(c, x0 + 6, y - 1)
        if self._can_replay():
            ui.keycap(c, x0 + 18, y - 1, 'SPACE', lit=(t // 12) % 2 == 0)
            s = 'GRAND FINALE AGAIN' if g.finale_done else 'GRAND FINALE'
            text(c, x0 + 20 + ui.keycap_width('SPACE'), y + 1, s, 'wine', font='tiny')
            return
        text(c, x0 + 18, y + 1, f'GRAND FINALE: {n}/{tot} PLACES HELPED', 'wine' if n else SOFT, font='tiny')

    def _can_replay(self):
        """SPACE plays the finale from here once it has been seen, or once every place is helped (the
        same rule as the finale's own cue on take-off and THE END badge)."""
        g = self.g
        return bool(g.finale_done) or g.world_mended()

    @staticmethod
    def _mark(c, x, y, st):
        if st == 'complete':
            ui.star(c, x - 1, y - 1)
        elif st == 'visited':
            c.rect(x, y + 1, 4, 3, 'hot_pink')
            c.hline(x + 1, x + 2, y, 'hot_pink')
            c.hline(x + 1, x + 2, y + 4, 'hot_pink')
        elif st == 'seen':
            c.rect(x + 1, y + 1, 2, 3, SOFT)
        else:
            c.px(x + 1, y + 2, FAINT)

    def _right(self, c, t):
        g = self.g
        lid = TL.TOUR[self.sel]
        st = self.status(lid)
        x0, y0, w, _h = RIGHT
        cx = x0 + w // 2
        ic = WI.icon(lid)
        if ic is not None:                      # the map icon, bottom-aligned in a 52 px box
            img = ic.rgba if st in ('visited', 'complete') else ic.grey
            top = y0 + 2 + max(0, 52 - ic.h)
            if st == 'unknown':
                _mask(c, img, cx - ic.w // 2, top, 'sand')
            else:
                _blit_rgba(c, img, cx - ic.w // 2, top)
        name = LW.LOCS[lid]['name'] if st != 'unknown' else '? ? ?'
        text(c, cx - text_width(name, 'normal') // 2, y0 + 55, name, INK_BROWN, font='normal')
        me = g.meta.get(lid)
        y = y0 + 66
        if st in ('unknown', 'seen') or me is None:
            msg = 'NOT VISITED YET' if st == 'seen' else 'NOT DISCOVERED YET'
            if st == 'seen' and not g.can_enter(lid):
                msg = 'NO DIORAMA YET'
            text(c, cx - text_width(msg, 'tiny') // 2, y + 4, msg, SOFT, font='tiny')
            return
        kind, facts = me['card']
        text(c, x0 + 7, y, kind, 'wine', font='tiny')
        for k, fct in enumerate(facts[:2]):
            text(c, x0 + 7, y + 7 + k * 7, fct, SOFT, font='tiny')
        y += 24
        c.hline(x0 + 6, x0 + w - 7, y - 3, 'sand')
        found = set(g._regions[lid].found) if lid in g._regions else set(g.saved_regions.get(lid, {}).get('found', ()))
        for k, name in enumerate(me['landmarks'][:7]):
            yy = y + k * 6
            if name in found:
                c.px(x0 + 8, yy + 2, 'gold')
                c.px(x0 + 9, yy + 2, 'gold')
                text(c, x0 + 13, yy, name, INK_BROWN, font='tiny')
            else:
                text(c, x0 + 13, yy, '? ? ?', FAINT, font='tiny')
        # stamps and events along the bottom
        yb = y0 + LEFT[3] - 13
        if lid in g._regions:
            r = g._regions[lid]
            got, used = len(r.stamps_got), set(r.used)
        else:
            sv = g.saved_regions.get(lid, {})
            got, used = len(sv.get('stamps', ())), set(sv.get('used', ()))
        for k in range(me.get('stamps', 3)):
            sx = x0 + 7 + k * 11
            if k < got:
                ui._STAMP.draw(c, sx, yb)
            else:
                _mask(c, ui._STAMP.rgba, sx, yb, 'sand')
        ex = x0 + 44
        for key, icon in me.get('hotspots', []):
            spr = ui.ICONS.get(icon, ui.ICONS['spark'])
            if key in used:
                spr.draw(c, ex, yb)
            else:
                _mask(c, spr.rgba, ex, yb, 'sand')
            ex += 11


def _canvas_rgba(cv, key='paper'):
    a = cv.a
    m = np.any(a != np.array(col(key), np.uint8), axis=2)
    out = np.zeros(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = np.where(m, 255, 0)
    return out.repeat(2, 0).repeat(2, 1)


def _wrap(s, n):
    """Split a line of words into lines of at most n characters."""
    out, line = [], ''
    for word in s.split():
        if line and len(line) + 1 + len(word) > n:
            out.append(line)
            line = word
        else:
            line = (line + ' ' + word).strip()
    if line:
        out.append(line)
    return out


def _blend(c, src, k):
    """c becomes src * (1 - k) + c * k (a cross-fade; k in 0..1)."""
    if k < 1:
        c.a[:] = (src.a.astype(np.float32) * (1 - k) + c.a.astype(np.float32) * k + 0.5).astype(np.uint8)


def _developing(t):
    """The colour of a print still being read: a slow cream shimmer, like an instant photo developing."""
    return 'cream' if (t // 6) % 3 else 'paper'


def _postcard_caption(c, p, x, y):
    """The viewer's caption, in the postcard's style: 'GREETINGS FROM' the place, when, the stamp."""
    sub = 'GREETINGS FROM'
    when = p.date() + '   ' + p.clock()
    w = max(text_width(sub, 'tiny') + 5 + text_width(p.name, 'normal'), text_width(when, 'tiny')) + 30
    h = 25
    c.round_rect(x, y, w, h, 'tan', r=2)
    c.round_rect(x + 1, y + 1, w - 2, h - 2, 'paper', r=2)
    for xx in range(x + 3, x + w - 3, 4):                # the dotted rule of the postcard's frame
        c.px(xx, y + 3, 'bubblegum')
        c.px(xx, y + h - 4, 'bubblegum')
    text(c, x + 7, y + 8, sub, 'clay', font='tiny')
    text(c, x + 7 + text_width(sub, 'tiny') + 5, y + 6, p.name, 'wine', font='normal')
    text(c, x + 7, y + 15, when, SOFT, font='tiny')
    ui._STAMP.draw(c, x + w - 17, y + 8)


def _icon_photo(c, x, y):
    """A little print (a hill under a sky): the PHOTOS page and 'look at a photo'."""
    if c is None:
        return 9
    c.rect(x, y + 1, 9, 8, 'white')
    c.rect(x + 1, y + 2, 7, 4, 'baby_blue')
    c.rect(x + 1, y + 6, 7, 2, 'leaf')
    c.hline(x + 3, x + 6, y + 5, 'leaf')
    c.px(x + 6, y + 3, 'butter')
    return 9


def _icon_pages(c, x, y):
    if c is None:
        return 9
    ui.ICONS['book'].recolor({col('ink'): col('tan')}).draw(c, x, y)
    return 9


def _icon_folder(c, x, y):
    """A folder: 'show the postcards folder' (the ALBUM page) or 'show this postcard's file' (the viewer)."""
    if c is None:
        return 10
    c.rect(x, y + 1, 4, 2, 'clay')
    c.rect(x, y + 2, 10, 7, 'clay')
    c.rect(x + 1, y + 3, 8, 5, 'gold')
    c.hline(x + 1, x + 8, y + 3, 'butter')
    return 10


def _tab_atlas(c, x, y):
    ui.icon_map(c, x, y)


def _tab_requests(c, x, y):
    MA.heart_icon(c, x, y)


def _tab_races(c, x, y):
    RC.FLAG.draw(c, x + 2, y + 1)


def _tab_album(c, x, y):
    ui.ICONS['camera'].draw(c, x, y)


def _tab_photos(c, x, y):
    c.rect(x, y + 1, 9, 8, 'brew')
    _icon_photo(c, x, y)
    c.frame(x, y + 1, 9, 8, 'brew')


def _tab_friends(c, x, y):
    _mini_balloon(c, x + 2, y + 2, 'classic', True)
    _mini_balloon(c, x + 7, y + 3, 'sky', True)


def _mini_balloon(c, x, y, pattern, met):
    """A 5 x 7 balloon in a friend's colours (centre x, envelope centre y)."""
    from .compose import PATTERNS
    spec = dict((p[0], p[3]) for p in PATTERNS).get(pattern)
    a, b = ('hot_pink', 'lavender') if spec is None else (('red', 'sky') if spec == 'rainbow' else spec[:2])
    if not met:
        a = b = 'sand'
    c.circle(x, y, 2, a)
    c.vline(x, y - 2, y + 2, b)
    c.px(x - 1, y + 3, 'brew')
    c.px(x + 1, y + 3, 'brew')
    c.rect(x - 1, y + 4, 3, 1, 'brew')


def _tab_badges(c, x, y):
    BG.ICON.draw(c, x + 1, y + 1)


def _tab_wardrobe(c, x, y):
    ui.icon_balloon(c, x + 2, y)


_TAB_ICONS = {'atlas': _tab_atlas, 'requests': _tab_requests, 'races': _tab_races, 'album': _tab_album,
              'photos': _tab_photos, 'friends': _tab_friends,
              'badges': _tab_badges, 'wardrobe': _tab_wardrobe}
