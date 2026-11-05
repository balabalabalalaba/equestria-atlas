"""Badges: milestones of the whole journey, shown on the journal's badge page. Each has a progress
(n, total); reaching the total awards it once (a toast with its medallion and a stamp) and it is kept
in the save. A badge with an `earn` test is awarded by that instead (THE END: its bar counts the places
helped, the badge comes with the finale they lead to).

Pacing: besides the big 100% milestones, a few come along the way (GOOD NEIGHBOUR: help 5 places,
LOCAL EXPERT: every landmark and stamp of one place, FIRST POSTCARD, RING DASH: 10 map rings), so a
badge turns up every few minutes of play rather than twice and then never. The list order is the
journal's trophy case, column by column: exploring, helping and friends, the sky (THE END last).
"""
from dataclasses import dataclass

from engine import timeline as TL

TOTAL_SPOTS = 98                # every hotspot of the 20 places (game/README.md)

from engine import Sprite        # noqa: E402
ICON = Sprite.from_ascii('''
    .rr.bb.
    .rr.bb.
    ..ooo..
    .oyYyo.
    oyYWYyo
    .oyYyo.
    ..ooo..
''', {'r': 'red', 'b': 'sky', 'o': 'amber', 'y': 'gold', 'Y': 'butter', 'W': 'white'})   # a badge (counters)


@dataclass
class Badge:
    key: str
    name: str
    desc: str
    color: tuple                # medallion (dark, mid, light)
    glyph: str                  # what is drawn on it: see draw_badge
    progress: callable          # game -> (n, total)
    earn: callable = None       # game -> bool: awarded by this instead of a full bar (or None)


def _found(g):
    return len(g.found), len(TL.TOUR)


def _stamps(g):
    return g.stamps_got(), 3 * len(TL.TOUR)


def _spots(g):
    used, total = 0, 0
    for lid in TL.TOUR:
        if lid in g._regions:
            r = g._regions[lid]
            used += len(r.used)
        else:
            used += len(g.saved_regions.get(lid, {}).get('used', ()))
        me = g.meta.get(lid)
        total += len(me['hotspots']) if me else 0
    if not all(lid in g.meta for lid in TL.TOUR):
        total = TOTAL_SPOTS
    return used, total


def _rings(g):
    from .world import RINGS
    return len(g.rings_flown), len(RINGS)


def _complete(g):
    n = 0
    for lid in TL.TOUR:
        a, tot, _s, _st = g.place_progress(lid)
        n += bool(tot) and a >= tot
    return n, len(TL.TOUR)


def _charted(g):
    return min(90, int(g.fog.explored() * 100)), 90


def _expert(g):
    """How near the closest place is to having every landmark and every stamp, in % (100: earned)."""
    best = (0, 1)
    for lid in TL.TOUR:
        a, tot, s, st = g.place_progress(lid)
        if not tot or st is None:
            continue
        n, m = a + s, tot + st
        if n * best[1] > best[0] * m:
            best = (n, m)
    return min(100, best[0] * 100 // best[1]), 100


def _helped(g):
    from .quests import TROUBLES
    return len(g.quests.fixed), len(TROUBLES)


def _the_end(g):
    """THE END's bar: the places helped (the finale comes when every one is); full once it is seen."""
    n, tot = _helped(g)
    return (tot, tot) if g.finale_done else (min(n, tot), tot)


BADGES = [
    # exploring
    Badge('explorer', 'EXPLORER', 'FIND ALL 20 PLACES', ('wine', 'hot_pink', 'pink'), 'balloon', _found),
    Badge('cartographer', 'CARTOGRAPHER', 'CHART 90% OF THE MAP', ('brew', 'tan', 'paper'), 'map', _charted),
    Badge('expert', 'LOCAL EXPERT', 'ALL SIGHTS + STAMPS OF A PLACE', ('rust', 'copper', 'peach'), 'star', _expert),
    Badge('sightseer', 'SIGHTSEER', 'COMPLETE EVERY PLACE', ('amber', 'gold', 'butter'), 'star', _complete),
    Badge('collector', 'STAMP COLLECTOR', 'ALL 60 HIDDEN STAMPS', ('maroon', 'red', 'salmon'), 'stamp', _stamps),
    Badge('sparks', 'SPARK MAKER', 'SET OFF ALL 98 HOTSPOTS', ('indigo', 'violet', 'lilac'), 'spark', _spots),
    # helping, friends and postcards
    Badge('courier', 'GOOD DEED', 'HELP A PLACE', ('brew', 'clay', 'sand'), 'heart',
          lambda g: (min(1, len(g.quests.fixed)), 1)),
    Badge('neighbour', 'GOOD NEIGHBOUR', 'HELP 5 PLACES', ('maroon', 'hot_pink', 'pink'), 'heart',
          lambda g: (min(5, len(g.quests.fixed)), 5)),
    Badge('postmaster', 'WORLD MENDER', 'HELP ALL 20 PLACES', ('forest', 'leaf', 'pale_leaf'), 'heart', _helped),
    Badge('flock', 'FRIENDLY SKIES', 'MEET ALL 15 BALLOONS', ('teal', 'sky', 'white'), 'balloon',
          lambda g: (len(g.flock.met), 15)),
    Badge('postcard', 'FIRST POSTCARD', 'TICK A PHOTO IN THE ALBUM', ('periwinkle', 'baby_blue', 'white'), 'camera',
          lambda g: (min(1, len(g.album.taken)), 1)),
    Badge('photographer', 'PHOTOGRAPHER', 'FILL THE PHOTO ALBUM', ('navy', 'royal', 'royal_lt'), 'camera',
          lambda g: (len(g.album.taken), 20)),
    # the sky
    Badge('ringdash', 'RING DASH', 'FLY 10 MAP RINGS', ('deep_teal', 'turquoise', 'aqua'), 'ring',
          lambda g: (min(10, len(g.rings_flown)), 10)),
    Badge('rings', 'RAINBOW RIDER', 'FLY EVERY MAP RING', ('teal', 'sky', 'light_sky'), 'ring', _rings),
    Badge('racer', 'RACER', 'WIN A RACE MEDAL', ('rust', 'copper', 'peach'), 'flag',
          lambda g: (min(1, g.races.medals()), 1)),
    Badge('champion', 'CHAMPION', 'GOLD IN ALL 6 RACES', ('amber', 'gold', 'white'), 'medal',
          lambda g: (g.races.golds(), 6)),
    Badge('skyracer', 'SKY RACER', 'BEAT EVERY RACING BALLOON', ('rust', 'orange', 'butter'), 'flag',
          lambda g: (len(g.flock.beaten), 5)),
    Badge('finale', 'THE END', 'HELP EVERY PLACE', ('plum', 'magic', 'crystal_pk'), 'heart', _the_end,
          earn=lambda g: bool(g.finale_done)),
]
BY_KEY = {b.key: b for b in BADGES}
PCT = ('cartographer', 'expert')        # their bars count per cent


class Badges:
    def __init__(self, game):
        self.g = game
        self.earned = {}                # badge key -> step it was earned
        self.new_t = None
        self.session_t0 = getattr(game, 't', 0)    # this session's first step: badges earned since blink

    def save(self):
        return dict(earned=dict(self.earned))

    def load(self, d):
        d = d or {}
        self.earned = {k: int(v) for k, v in d.get('earned', {}).items() if k in BY_KEY}
        self.session_t0 = self.g.t      # (Game._load_state sets the resumed clock first)

    def progress(self, b):
        try:
            n, tot = b.progress(self.g)
        except Exception:
            n, tot = 0, 1
        return min(n, tot), tot

    def earned_now(self, b):
        """Would this badge be awarded now?"""
        if b.earn is not None:
            try:
                return bool(b.earn(self.g))
            except Exception:
                return False
        n, tot = self.progress(b)
        return bool(tot) and n >= tot

    def fresh(self, key):
        """Earned in this session (it blinks on the badge page)."""
        t = self.earned.get(key)
        return t is not None and t >= self.session_t0

    def check(self, quiet=False):
        """Award every badge whose progress is full. quiet: no toasts (loading an old save): those
        were earned before this session and do not blink."""
        g = self.g
        for b in BADGES:
            if b.key in self.earned:
                continue
            if self.earned_now(b):
                self.earned[b.key] = g.t
                if quiet:
                    self.session_t0 = max(self.session_t0, g.t + 1)
                else:
                    self.new_t = g.t
                    g.toasts.push('BADGE: ' + b.name, b.desc, _badge_icon(b), sound='stamp', gain=0.55)


_SPR = {}


def badge_sprite(b, earned=True, small=False, k=1):
    """The medallion as a Sprite (cached), drawn k times bigger (nearest): blit its centre at (cx, cy) with
    .draw(c, cx - 9 * k, cy - 9 * k)."""
    key = (b.key, earned, small, k)
    spr = _SPR.get(key)
    if spr is None:
        from engine import Canvas
        import numpy as np
        cv = Canvas(19, 19, color=(255, 0, 254))
        draw_badge(cv, 9, 9, b, earned=earned, small=small)
        if k > 1:
            cv.a = np.repeat(np.repeat(cv.a, k, axis=0), k, axis=1)
        spr = _SPR[key] = Sprite.from_canvas(cv, key=(255, 0, 254))
    return spr


def _badge_icon(b):
    def draw(c, x, y):
        draw_badge(c, x + 4, y + 4, b, small=True)
    return draw


def draw_badge(c, cx, cy, b, earned=True, small=False):
    """A round medallion centred on (cx, cy): r 8 (r 5 small), coloured rim, a glyph."""
    from . import ui
    dk, mid, hi = b.color if earned else ('tan', 'sand', 'paper')
    r = 5 if small else 8
    c.circle(cx, cy, r, ui.INK if earned else 'tan')
    c.circle(cx, cy, r - 1, dk)
    c.circle(cx, cy, r - 2, mid)
    c.px(cx - r + 3, cy - r + 3, hi)
    if small:
        c.px(cx, cy, 'white' if earned else 'tan')
        return
    ink = 'white' if earned else 'tan'
    gl = b.glyph
    if gl == 'star':
        ui.star(c, cx - 3, cy - 3) if earned else c.rect(cx - 2, cy - 2, 5, 5, ink)
    elif gl == 'stamp':
        ui._STAMP.draw(c, cx - 4, cy - 4) if earned else c.rect(cx - 3, cy - 3, 7, 7, ink)
    elif gl in ui.ICONS or gl in ('balloon', 'map', 'ring', 'parcel', 'medal', 'flag'):
        if gl == 'balloon':
            ui.icon_balloon(c, cx - 3, cy - 5)
        elif gl == 'map':
            ui.icon_map(c, cx - 4, cy - 4)
        elif gl == 'ring':
            for k, cc in enumerate(('red', 'butter', 'sky')):
                c.ellipse(cx, cy, 4 - k, 5 - k, cc if earned else ink, fill=False)
        elif gl == 'parcel':
            from .items import ITEMS
            ITEMS['parcel'].draw(c, cx - 3, cy - 3)
        elif gl == 'medal':
            from .races import draw_medal
            draw_medal(c, cx - 4, cy - 5, 'gold' if earned else 'none')
        elif gl == 'flag':
            from .races import FLAG
            FLAG.draw(c, cx - 3, cy - 3)
        else:
            ui.ICONS[gl].draw(c, cx - 4, cy - 4)
    if not earned:                                      # greyed: a faint wash over the glyph
        for yy in range(cy - r + 2, cy + r - 1):
            for xx in range(cx - r + 2, cx + r - 1):
                if (xx + yy) % 2 == 0 and (xx - cx) ** 2 + (yy - cy) ** 2 < (r - 2) ** 2:
                    c.px(xx, yy, 'sand')
