"""Toasts: short celebration banners at the top centre of the screen (HUD layer), one at a time.

`Toasts.push(title, sub='', icon=None, sound=None)` queues one; `draw(c, t)` shows the current one
sliding down, holding and sliding back up (TOAST steps). Used for errands (picked up / delivered),
race results, album photos, new balloon patterns and badges. An icon is a 9x9-ish Sprite, a name in
game.ui.ICONS or a draw function fn(c, x, y). Draw state lives in the queue (it is set when step()
advances), so drawing stays pure.

Pacing: for the first EARLY steps of a session a toast that is not urgent waits until GAP steps after
the previous one has gone, and while a tutor tip is on screen (game/tutor.py), so the first minutes
speak one message at a time. A toast's sound is skipped when the same cue played less than ECHO steps
ago (from any source: Game.sfx lists every step's cues in g.sounds).

Placement: a toast never covers the errand plate under the counters (on the map before the first landing
it is level with the toasts). One that would moves left of it (clear of the clock), or, too wide for
that, shows just under the plate, sliding out from beneath it. The place is latched as the toast first
shows and never moves mid-display.
"""
from engine import W, H, Sprite, seg, ease_out_back, ease_in, text, text_width

from . import ui

TOAST = 96                      # steps per toast (in 8, hold, out 10)
EARLY = 3600                    # steps of a session during which toasts are paced (2 minutes)
GAP = 90                        # ... a calm toast waits this long after the last one went
ECHO = 45                       # a toast's sound is skipped if the same cue played this recently
SEEN = 50                       # a toast cut short by an urgent one before this many steps is shown again
CLOCK_W = 54                    # the clock's corner (top left): a toast moved left stops short of it
PLUM, PLUM_HI = ui.PLATE, ui.PLATE_HI


class Toasts:
    def __init__(self, game):
        self.g = game
        self.queue = []                 # [dict(title, sub, icon, sound, gain, urgent)]
        self.cur = None                 # the toast on screen, with its start step 't0'
        self.start = None               # step of the first step(): the session's paced minutes count from it
        self.end_t = -10 ** 6           # step the last toast went
        self.heard = {}                 # cue -> step it last played
        self._snd = None                # the g.sounds list of the last step (its cues are noted next step)

    def push(self, title, sub='', icon=None, sound=None, gain=0.5, urgent=False, ttl=None):
        """Queue a toast; urgent: show it next, cutting the one on screen short (its slide-out).
        ttl: steps it may wait in the queue; one still waiting after that is dropped (news that is only
        true now, like the weather turning, never shows late)."""
        d = dict(title=title, sub=sub, icon=icon, sound=sound, gain=gain, urgent=urgent)
        if ttl is not None:
            d['due'] = self.g.t + int(ttl)
        if not urgent:
            self.queue.append(d)
            return
        self.queue.insert(0, d)
        if self.cur is not None and self.g.t - self.cur['t0'] < TOAST - 10:
            if self.g.t - self.cur['t0'] < SEEN and not self.cur.get('again'):
                # barely seen: it comes back (quietly) right after the urgent one instead of being lost
                back = {k: v for k, v in self.cur.items() if k not in ('t0', 'due') and not k.startswith('_')}
                back.update(again=True, sound=None)
                self.queue.insert(1, back)
            self.cur['t0'] = self.g.t - (TOAST - 10)       # slide out now

    def clear(self):
        self.queue = []
        self.cur = None

    def ready(self, t):
        """Would the next queued toast start now? (While any waits, the tutor keeps its tips away.)"""
        if not self.queue or self.cur is not None:
            return False
        d = self.queue[0]
        if d.get('urgent') or t - (t if self.start is None else self.start) >= EARLY:
            return True
        if t - self.end_t < GAP:
            return False
        tu = getattr(self.g, 'tutor', None)
        return not (tu is not None and tu.cur is not None and tu.shown is not None)

    def step(self, t):
        if self.start is None:
            self.start = t
        self._hear(t)
        if self.cur is not None and t - self.cur['t0'] >= TOAST:
            self.cur = None
            self.end_t = t
        if self.queue and any('due' in d for d in self.queue):     # stale news is dropped
            self.queue = [d for d in self.queue if d.get('due', t) >= t]
        if self.cur is None and self.ready(t):
            self.cur = dict(self.queue.pop(0), t0=t)
            snd = self.cur['sound']
            if snd and t - self.heard.get(snd, -10 ** 6) >= ECHO:
                self.g.sfx(snd, self.cur['gain'])
                self.heard[snd] = t

    def _hear(self, t):
        """Note the cues played since the last step. Game.step starts a new g.sounds list every step, so
        the last step's list (kept here) holds everything played after this ran then."""
        cur = getattr(self.g, 'sounds', None)
        for lst, when in ((self._snd, t - 1), (cur, t)):
            if lst is None:
                continue
            for s in lst:
                name = s[0] if isinstance(s, (tuple, list)) else s
                if self.heard.get(name, -10 ** 6) < when:
                    self.heard[name] = when
        self._snd = cur

    def _plate_rect(self):
        """The errand plate under the counters (x, y, w, h) on the world map or in a place, or None. Its
        y mirrors world.MapMode.render (29 once a place is visited, else 16) and region.Region._hud (16,
        plus 13 per counter row above it)."""
        g = self.g
        q = getattr(g, 'quests', None)
        m = getattr(g, 'mode', None)
        if q is None or m is None:
            return None
        try:
            if m is getattr(g, 'map', None):
                if getattr(m, 'overview', False):
                    return None
                pw = q._plate_w(q._plate_tg(None))
                py = 29 if g.visited else 16
            elif hasattr(m, 'stamp_total') and hasattr(m, 'spot_total'):
                pw = q._plate_w(q.active(m))
                py = 16 + 13 * bool(m.stamp_total) + 13 * bool(m.spot_total)
            else:
                return None
        except Exception:
            return None
        return (W - 3 - pw, py, pw, 11) if pw else None

    def _place(self, d, y):
        """(dx, y, clip) of a toast drawn at y: centred, or moved left / dropped under the errand plate
        so it never covers it (clip: rows above it are kept clear while it slides). Low toasts (the
        balloon is up top: world.Game._toast_y) are already clear of it."""
        w, h = _size(d)
        x = (W - w) // 2
        pr = None if d.get('_low') else self._plate_rect()
        if pr is None:
            return 0, y, None
        px, py, pw, ph = pr
        if x + w + 2 < px or x - 1 > px + pw or y + h < py or y - 1 > py + ph:     # clear of it
            return 0, y, None
        over = x + w + 3 - px
        if over <= x - (CLOCK_W if y < 16 else 3):                  # room on the left (by the clock)
            return -over, y, None
        y2 = py + ph + 3                                            # too wide: just under the plate
        box = None
        try:
            from .tutor import _balloon_box
            box = _balloon_box(self.g)
        except Exception:
            pass
        if box is not None and box[0] < x + w + 2 and x - 2 < box[2] and box[1] < y2 + h + 2 and y2 - 2 < box[3]:
            return 0, H - 78, None                                  # (the balloon is there: the low slot)
        return 0, y2, py + ph + 1

    @property
    def busy(self):
        return self.cur is not None or bool(self.queue)

    def draw(self, c, t, y=4):
        cur = self.cur
        if cur is None:
            return
        a = t - cur['t0']
        if a < 0 or a >= TOAST:
            return
        if a < 10:
            k = ease_out_back(seg(a, 0, 10))
        elif a > TOAST - 10:
            k = 1 - ease_in(seg(a, TOAST - 10, TOAST))
        else:
            k = 1.0
        title, sub, ic = cur['title'], cur['sub'], cur['icon']
        iw = 0 if ic is None else 12
        w, h = _size(cur)
        pl = cur.get('_pl')                             # its place, latched as it first shows (per slot y)
        if pl is None or pl[0] != y:
            pl = (y,) + self._place(cur, y)
            cur['_pl'] = pl
        _y0, dx, y, clip = pl
        x = (W - w) // 2 + dx
        yy = int(round(y - (h + 6) * (1 - k)))
        keep = None
        if clip is not None and yy - 1 < clip:          # sliding out from under the errand plate
            x0, x1 = max(0, x - 1), min(W, x + w + 1)
            keep = (x0, c.a[:clip, x0:x1].copy())
        c.round_rect(x - 1, yy - 1, w + 2, h + 2, '#6e2727', r=2)
        c.round_rect(x, yy, w, h, 'gold' if (a // 4) % 2 or a > 24 else 'butter', r=2)
        c.rect(x + 1, yy + 1, w - 2, h - 2, PLUM)
        c.rect(x + 2, yy + 2, w - 4, 1, PLUM_HI)
        if ic is not None:
            _icon(c, ic, x + 4, yy + (h - 9) // 2)
        text(c, x + 6 + iw, yy + 4, title, '#fff6c0', font='normal', shadow='#6e2727')
        if sub:
            text(c, x + 6 + iw, yy + 14, sub, 'bubblegum', font='tiny')
        if a < 30 and (a // 3) % 2 == 0:                     # a glint running along the rim
            gx = x + int((w - 4) * a / 30)
            c.px(gx + 2, yy, 'white')
            c.px(gx + 3, yy, 'white')
        if keep is not None:
            x0, rows = keep
            c.a[:clip, x0:x0 + rows.shape[1]] = rows


def _size(d):
    """(w, h) of a toast's plate."""
    iw = 0 if d['icon'] is None else 12
    tw = max(text_width(d['title'], 'normal'), text_width(d['sub'], 'tiny') if d['sub'] else 0)
    return iw + tw + 12, 22 if d['sub'] else 15


def _icon(c, ic, x, y):
    if callable(ic) and not isinstance(ic, Sprite):
        ic(c, x, y)
    elif isinstance(ic, str):
        ui.ICONS[ic].draw(c, x, y)
    else:
        ic.draw(c, x + (9 - ic.w) // 2, y + (9 - ic.h) // 2)
