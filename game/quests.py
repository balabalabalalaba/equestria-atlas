"""Helping the places: every place has a trouble to find by exploring, and fixing it changes the world.

Each of the 20 places has one trouble, drawn in its diorama (game/mend_art.py): wilted flower beds, a
festival string with dark lanterns, a dark cracked lamp, a dry fountain, an empty market stall, bare
festival poles, an empty fireworks rack, an empty notice board. A '?' bobs over it until the balloon
comes close: then the player notices it (a toast says what it needs and where that comes from) and
it is listed on the journal's HELP page. Troubles can be found and fixed in any order.

What a trouble needs comes from another place that has plenty of it (seeds from Klugetown's bazaar,
crystal light from the Crystal Empire...). Once a trouble needing it has been noticed, the goods wait
at their source spot, glowing: fly the envelope to them, SPACE, and they hang under the basket (one
kind of goods at a time, enough for every noticed trouble that needs it: the plate shows X2...). On the
map a '?' bobs over every found place whose trouble is still to be found, and the plate names the
nearest job (FROM / TO a place, +N more). Bring the goods to the trouble, SPACE: a burst of magic mends
it for good, a light band sweeps the place's festival in (bunting between its landmarks, a festival
mast over the mended prop, party balloons or sky lanterns, confetti: in every area, for good),
fireworks go up and the world map changes too (game/world.py: the helped place gets bunting and
festival lights, a thin rainbow arcs back to the place whose goods helped, route legs between helped
neighbours turn into rainbows, more balloons fill the sky). Nobody is ever shown: the places
themselves are helped (no creatures).

Kept from the first version (the engine hooks in game/region.py and game/world.py call these): the
cargo, the spot markers and bubbles, the plate under the counters, edge arrows and map markers.
Spot coordinates are the envelope centre in the area's main-layer world px (tools/phaseb_spots.py
checks them).
"""
import math
from functools import lru_cache
from dataclasses import dataclass

from engine import W, H, fx
from engine import loc_worldmap as LW
from engine import timeline as TL

from . import ui, perks
from . import mend_art as MA
from .items import ITEMS, ITEM_NAMES

REACH = 16                      # envelope <-> spot distance that shows the SPACE bubble
NOTICE = 70                     # envelope <-> trouble distance at which the player notices it
MEND = 16                       # steps of the mending magic before the prop changes
REVEAL = 40                     # steps of the light band that sweeps the festive layer in after a mend
ENV = {'l': 18, 'm': 13, 's': 8, 'xs': 4}


@dataclass
class Trouble:
    key: str                    # = the place id
    area: int
    kind: str                   # a prop kind in game/mend_art.py
    item: str                   # what it needs (game/items.py)
    x: int                      # the prop's centre and the ground it stands on (area main-layer px)
    ground: int
    title: str                  # what is wrong
    done: str                   # how it looks once mended


TROUBLES = [
    Trouble('canterlot', 1, 'garden', 'seeds', 100, 128, 'THE ROYAL FLOWER BEDS HAVE WILTED',
            'THE ROYAL GARDENS BLOOM AGAIN'),
    Trouble('cloudsdale', 0, 'bunting', 'parcel', 99, 99, 'THE GAMES HAVE NO FLAGS',
            'FLAGS FLY OVER CLOUDSDALE'),
    Trouble('ponyville', 0, 'stall', 'apples', 690, 146, 'THE MARKET CART STANDS EMPTY',
            'FRESH APPLES AT THE MARKET'),
    Trouble('sweet_apple_acres', 0, 'lanterns', 'lantern', 680, 146, 'THE BARN DANCE HAS NO LIGHTS',
            'THE ORCHARD LANTERNS GLOW'),
    Trouble('everfree', 1, 'lamp', 'crystal', 110, 146, 'THE FOREST PATH LAMP IS DARK',
            'A CRYSTAL LIGHT ON THE PATH'),
    Trouble('neighagra_falls', 0, 'board', 'book', 55, 100, 'THE TRAIL BOARD IS BARE',
            'A TRAIL MAP FOR THE FALLS'),
    Trouble('rainbow_falls', 0, 'garden', 'seeds', 455, 143, 'THE CLIFFTOP PLANTER IS DRY',
            'FLOWERS ON THE CLIFFTOP'),
    Trouble('manehattan', 2, 'stall', 'cider', 416, 133, 'THE PARK CART HAS NOTHING TO SELL',
            'HOT CIDER IN THE PARK'),
    Trouble('baltimare', 2, 'lanterns', 'lantern', 120, 152, 'THE BEACH PARTY IS IN THE DARK',
            'LANTERNS ALONG THE BEACH'),
    Trouble('griffonstone', 1, 'stall', 'cupcakes', 205, 61, 'THE CASTLE STALL IS BARE',
            'CUPCAKES ATOP GRIFFONSTONE'),
    Trouble('dragon_lands', 2, 'fountain', 'rainbow', 360, 121, 'THE GROVE FOUNTAIN RAN DRY',
            'RAINBOW WATER IN THE GROVE'),
    Trouble('mount_aris', 1, 'fireworks', 'fireworks', 245, 141, 'THE FESTIVAL RACK IS EMPTY',
            'FIREWORKS OVER MOUNT ARIS'),
    Trouble('klugetown', 1, 'lamp', 'crystal', 560, 152, 'THE BAZAAR LAMP IS BROKEN',
            'THE BAZAAR SHINES AT NIGHT'),
    Trouble('changeling_kingdom', 1, 'stall', 'gems', 355, 150, 'THE HIVE MARKET IS EMPTY',
            'GEMS SPARKLE AT THE HIVE'),
    Trouble('appleloosa', 0, 'board', 'scroll', 455, 148, 'THE TOWN BOARD HAS NO NEWS',
            'NEWS FROM CANTERLOT POSTED'),
    Trouble('las_pegasus', 1, 'stall', 'pearl', 290, 134, 'THE PLAZA STALL IS CLOSED',
            'PEARLS ON THE PLAZA'),
    Trouble('vanhoover', 2, 'fireworks', 'fireworks', 120, 155, 'THE HARBOUR RACK IS EMPTY',
            'FIREWORKS OVER THE BAY'),
    Trouble('crystal_empire', 2, 'stall', 'tea', 410, 194, 'THE COLD STATION HAS NO TEA',
            'WARM TEA AT THE STATION'),
    Trouble('yakyakistan', 1, 'lanterns', 'lantern', 385, 142, 'THE SNOW FESTIVAL IS DARK',
            'LANTERNS IN THE SNOW'),
    Trouble('castle_two_sisters', 2, 'lamp', 'crystal', 240, 398, 'THE OLD TREE SITS IN THE DARK',
            'CRYSTAL LIGHT BY THE TREE OF HARMONY'),
]
BY_KEY = {tr.key: tr for tr in TROUBLES}

# where each kind of goods comes from: (place, area, envelope x, y, landmark)
SOURCES = {
    'apples': ('sweet_apple_acres', 0, 205, 60, 'BIG RED BARN'),
    'cupcakes': ('ponyville', 0, 700, 44, 'SUGARCUBE CORNER'),
    'scroll': ('canterlot', 0, 150, 70, 'CANTERLOT CASTLE'),
    'crystal': ('crystal_empire', 1, 215, 80, 'CRYSTAL HEART'),
    'rainbow': ('cloudsdale', 1, 120, 100, 'WEATHER FACTORY'),
    'parcel': ('manehattan', 0, 300, 60, 'MANEHATTAN'),
    'cider': ('appleloosa', 0, 470, 80, 'APPLE ORCHARDS'),
    'fireworks': ('las_pegasus', 0, 300, 90, 'THE STRIP'),
    'tea': ('vanhoover', 1, 120, 120, 'SMOKEY MOUNTAINS'),
    'gems': ('yakyakistan', 2, 200, 100, 'MT. EVERHOOF'),
    'lantern': ('dragon_lands', 2, 280, 60, 'KIRIN GROVE'),
    'pearl': ('mount_aris', 3, 270, 0, 'SEAQUESTRIA'),
    'book': ('baltimare', 0, 130, 60, 'THE BIG BELL'),
    'seeds': ('klugetown', 1, 140, 70, 'THE BAZAAR'),
}
REWARDS = {5: 'courier', 10: 'parcel_post', 20: 'aurora'}     # places helped -> balloon pattern


def festival_night(hour):
    """0..1: how many of the festival's lights are on (bulbs, sky lanterns, glows): they come on one by one
    with the windows of the places (game/nightwin.py), from about 19:00 to 19:40 and go out again 05:30 to
    06:00, instead of all at once (game/mend_art.py takes it as each light's own threshold)."""
    from .daylight import night
    from engine import clamp
    return round(clamp((night(hour) - 0.12) / 0.4) * 20) / 20


def place_name(lid):
    return LW.LOCS[lid]['name'].upper() if lid in LW.LOCS else lid.upper()


def spot(r, tr):
    """The envelope spot over a trouble's prop in region r (clamped to where the envelope can go)."""
    A = r.areas[tr.area]
    ey = ENV.get(A.balloon, 13)
    y = tr.ground - MA.height(tr.kind) - 6
    return tr.x, min(max(y, A.y_range[0] - ey), A.y_range[1] - ey)


class Errands:
    """The help system (named for the hooks it replaced)."""

    def __init__(self, game):
        self.g = game
        self.seen = {}                  # trouble key -> step it was noticed
        self.fixed = {}                 # trouble key -> step it was mended
        self._carrying = None           # the item in the basket (see carrying)
        self.crates = 0                 # how many of it: one for every noticed open trouble needing it
        self.intro = False              # the 'look for ?' hint has been shown
        self.pick_t = None
        self.drop_t = None              # (step, place id) of the last mending: fireworks
        self.near = None                # 'pick' | 'drop' | 'need' while at a spot

    @property
    def carrying(self):
        """The item key in the basket, or None."""
        return self._carrying

    @carrying.setter
    def carrying(self, item):
        # goods put in the basket from anywhere (a flock friend sharing them too) fill it for every open
        # need, as a pick-up does; emptied, it holds none
        if item != self._carrying:
            self.crates = max(1, self.wanting(item)) if item else 0
        self._carrying = item

    # -------------------------------------------------------------- save
    def save(self):
        return dict(seen=dict(self.seen), fixed=dict(self.fixed), carrying=self.carrying,
                    load=self.crates if self.carrying else 0, intro=self.intro)

    def load(self, d):
        d = d or {}
        if 'seen' not in d and 'fixed' not in d:          # a version 2 save: the old errand chain
            return
        self.seen = {k: int(v) for k, v in d.get('seen', {}).items() if k in BY_KEY}
        self.fixed = {k: int(v) for k, v in d.get('fixed', {}).items() if k in BY_KEY}
        self.seen.update({k: v for k, v in self.fixed.items() if k not in self.seen})
        c = d.get('carrying')
        self.carrying = c if c in SOURCES and c in self.needed() else None
        try:
            n = int(d.get('load', 1))
        except (TypeError, ValueError):
            n = 1
        self.crates = min(max(1, n), self.wanting(self.carrying)) if self.carrying else 0
        self.intro = bool(d.get('intro', False))

    # -------------------------------------------------------------- state
    @property
    def done(self):
        """Places helped (in the order they were helped)."""
        return sorted(self.fixed, key=self.fixed.get)

    def open_troubles(self):
        return [tr for tr in TROUBLES if tr.key in self.seen and tr.key not in self.fixed]

    def needed(self):
        """Items some noticed, unmended trouble needs."""
        return {tr.item for tr in self.open_troubles()}

    def wanting(self, item):
        """How many noticed, unmended troubles need `item`."""
        return sum(1 for tr in self.open_troubles() if tr.item == item)

    def status(self, key):
        """'fixed' | 'seen' | 'unknown'."""
        return 'fixed' if key in self.fixed else 'seen' if key in self.seen else 'unknown'

    def cargo(self):
        return self.carrying

    def targets(self):
        """[(place id, area, x, y, 'drop'|'pick', item)]: where the player can make progress now."""
        out = []
        if self.carrying:
            for tr in self.open_troubles():
                if tr.item == self.carrying:
                    out.append((tr.key, tr.area, tr.x, None, 'drop', tr.item))
        else:
            for item in sorted(self.needed()):
                lid, ai, x, y, _l = SOURCES[item]
                out.append((lid, ai, x, y, 'pick', item))
        return out

    def active(self, here=None):
        """The nearest target (for the plate), or None. here = the Region the balloon is in: its own
        targets come first, then the others by map distance from its icon; None = on the world map,
        by distance from the balloon."""
        tg = self.targets()
        if len(tg) > 1:
            from .world import icon_centre
            if here is not None:
                ox, oy = icon_centre(here.ID)
            else:
                ox, oy = self.g.map.bx, self.g.map.by
            def key(tgt):
                x, y = icon_centre(tgt[0])
                return (0 if here is not None and tgt[0] == here.ID else 1, (x - ox) ** 2 + (y - oy) ** 2)
            tg.sort(key=key)
        return tg[0] if tg else None

    # -------------------------------------------------------------- in a place
    def _here(self, r):
        tr = BY_KEY.get(r.ID)
        return tr if tr is not None and tr.area == r.ai else None

    def _pick_here(self, r):
        """(item, x, y) of the goods waiting in this area, or None."""
        if self.carrying:
            return None
        for item in sorted(self.needed()):
            lid, ai, x, y, _l = SOURCES[item]
            if lid == r.ID and ai == r.ai:
                return item, x, y
        return None

    def region_step(self, r, inp):
        """Called every step in a place before its hotspots. Returns True when the balloon is at a spot
        (SPACE belongs to it then)."""
        g = self.g
        self.near = None
        if r.leaving is not None:
            return False
        if not self.intro and g.t - r.t_enter > 40:
            self.intro = True
            g.toasts.push('PLACES NEED HELP', 'LOOK FOR ? AND HELP THEM', MA.question_icon, sound='sparkle')
        ex, ey = r.envelope()
        tr = self._here(r)
        if tr is not None and tr.key not in self.fixed:
            sx, sy = spot(r, tr)
            if tr.key not in self.seen and math.hypot(ex - tr.x, ey - (tr.ground - MA.height(tr.kind) // 2)) < NOTICE * perks.notice(g):
                self.seen[tr.key] = g.t
                src = SOURCES[tr.item]
                if self.carrying == tr.item:            # the basket has room for one more of it
                    self.crates += 1
                sub = ('YOU HAVE ' + ITEM_NAMES[tr.item] + '!' if self.carrying == tr.item else
                       'NEEDS ' + ITEM_NAMES[tr.item] + ' FROM ' + place_name(src[0]))
                g.toasts.push(tr.title, sub, ITEMS[tr.item], sound='idea', gain=0.4, urgent=True)
                g.save()
            if tr.key in self.seen and math.hypot(ex - sx, ey - sy) < REACH * perks.parcel_reach(g):
                if self.carrying == tr.item:
                    self.near = 'drop'
                    if inp.action:
                        self._mend(r, tr)
                else:
                    self.near = 'need'
                    if inp.action:
                        g.sfx('tick', 0.35)
                return True
        pk = self._pick_here(r)
        if pk is not None:
            item, x, y = pk
            if math.hypot(ex - x, ey - y) < REACH * perks.parcel_reach(g):
                self.near = 'pick'
                if inp.action:
                    self.carrying = item
                    self.crates = max(1, self.wanting(item))     # enough for every open need
                    self.pick_t = g.t
                    dests = sorted({place_name(tr.key) for tr in self.open_troubles() if tr.item == item})
                    title = ITEM_NAMES[item] + (' X%d' % self.crates if self.crates > 1 else '')
                    g.toasts.push(title, 'TAKE IT TO ' + ' / '.join(dests[:2]) + (' ...' if len(dests) > 2 else ''),
                                  ITEMS[item], sound='coin', gain=0.5, urgent=True)
                    g.sfx('pop', 0.4, 2)
                return True
        return False

    def _mend(self, r, tr):
        g = self.g
        self.fixed[tr.key] = g.t
        self.crates -= 1                                # one unit used; the rest stays in the basket
        if self.crates <= 0 or self.wanting(self.carrying) == 0:
            self.carrying, self.crates = None, 0
        self.drop_t = (g.t, r.ID)
        g.sfx('powerup', 0.8)                           # the fanfare; 'stamp' lands a beat later
        g.sfx('shimmer', 0.6)
        g.toasts.push(place_name(tr.key) + ' IS BRIGHTER', tr.done, MA.heart_icon, urgent=True)
        n = len(self.fixed)
        if n in REWARDS:
            from .compose import PATTERNS
            name = dict((p[0], p[1]) for p in PATTERNS).get(REWARDS[n], REWARDS[n].upper())
            g.toasts.push('NEW BALLOON', name + ' PATTERN (J)', ui.icon_balloon, sound='quest', gain=0.5)
        if n == len(TROUBLES):
            g.toasts.push('THE WORLD IS BRIGHTER', 'EVERY PLACE HAS BEEN HELPED', MA.heart_icon, sound='quest')
        elif n == 1:
            g.toasts.push('THE MAP CHANGES TOO', 'TAKE OFF AND HAVE A LOOK', ui.icon_map)
        g.save()

    def region_update(self, r, t):
        if self.drop_t is None or self.drop_t[1] != r.ID:
            return
        a = t - self.drop_t[0] - MEND
        if a == 18:                                     # one beat after the prop changes
            self.g.sfx('stamp', 0.7)
        if 0 <= a < 6 * 16 and a % 16 == 0:
            self.g.sfx('firework', 0.4)

    def region_draw(self, r, c, t, ix, iy):
        """The festival of a helped place (swept in by a light band after the mend), the trouble's prop
        (broken / mending / mended), the goods waiting at a source, bubbles."""
        night = festival_night(self.g.hour)
        tr = self._here(r)
        ft = self.fixed.get(r.ID)
        if ft is not None and t - ft >= MEND:            # a helped place celebrates, in every area
            reveal = None
            if tr is not None and t - ft < MEND + REVEAL:
                reveal = int((t - ft - MEND) / REVEAL * (W + 24)) - 8
            keep = c.a[:, max(0, reveal):].copy() if reveal is not None and reveal < W else None
            _festive(r, c, t, ix, iy, night, tr)
            if keep is not None:                         # the reveal: only left of the light band yet
                c.a[:, max(0, reveal):] = keep
        if tr is not None:
            px, gy = tr.x - ix, tr.ground - iy
            if -50 < px < W + 50 and -60 < gy < H + 80:
                ft = self.fixed.get(tr.key)
                a = t - ft if ft is not None else None
                mended = a is not None and a >= MEND - 4
                MA.draw_prop(c, tr.kind, px, gy, t, mended, tr.item, r.light_tint(), night)
                top = gy - MA.height(tr.kind)
                if tr.key not in self.seen:
                    MA.question(c, px, top - 2, t)
                if a is not None and 0 <= a < MEND + 70:
                    _mend_fx(c, t, a, px, gy, top)
                sx, sy = spot(r, tr)
                if self.near in ('drop', 'need'):
                    if self.near == 'drop':
                        ui.prompt_bubble(c, sx - ix, sy - iy - 8, ITEMS[tr.item], t)
                    else:
                        _need_bubble(c, sx - ix, sy - iy - 8, tr.item, t)
        pk = self._pick_here(r)
        if pk is not None:
            item, x, y = pk
            sx, sy = x - ix, y - iy
            if -30 < sx < W + 30 and -30 < sy < H + 30:
                _pick_marker(c, sx, sy, t, item)
                if self.near == 'pick':
                    ui.prompt_bubble(c, sx, sy - 6, ITEMS[item], t)
        if self.drop_t is not None and self.drop_t[1] == r.ID:
            from .region import SHELLS
            for k, (x, y, cols) in enumerate(SHELLS):
                t0 = self.drop_t[0] + MEND + k * 16
                a = t - t0
                if -12 <= a < 0:
                    yy = y + int(-a * 4)
                    c.px(x, yy, 'butter')
                    c.px(x, yy + 2, 'gold')
                if 0 <= a < 6:
                    fx.glow(c, x, y, 14 - a, cols[0], 0.6 * (1 - a / 6), rings=3)
                ui.burst(c, t, t0, x, y, cols, n=48, seed=71 + k, power=1.5, life=54, gravity=0.02, size=2)
        if tr is not None and ft is not None and 0 <= t - ft - MEND < REVEAL:
            MA.reveal_band(c, int((t - ft - MEND) / REVEAL * (W + 24)) - 8, t)

    def region_hud(self, r, c, t, y=29):
        """Plate under the counters (a compact one while a landmark tag would sit under the full one);
        a gold edge arrow to the spot in this place (or its area)."""
        w = self._plate_w(self.active(r))
        if w and _tags_hit(r, t, (W - 3 - w - 1, y - 1, w + 2, 13)) is not None:
            for _k in range(3):                         # a tag right at the edge: slip under it
                b = _tags_hit(r, t, (W - 22, y - 1, 20, 13))
                if b is None:
                    break
                y = b + 2
            self._plate_compact(c, t, y, r)
        else:
            self._plate(c, t, y, r)
        for lid, ai, x, sy, kind, _item in self.targets():
            if lid != r.ID:
                continue
            if sy is None:
                sx_, sy = spot(r, BY_KEY[lid])
            if ai != r.ai:
                ui.edge_arrow(c, W + 40 if ai > r.ai else -40, H // 2, t, 'gold')
                return
            px, py = x - r.cx, sy - r.cy
            if not (0 <= px < W and 0 <= py < H):
                ui.edge_arrow(c, px, py, t, 'gold')
            return

    # -------------------------------------------------------------- world map
    def unknown_places(self):
        """Places found on the map, with a diorama, whose trouble is still to be found ('?' on the map),
        not counting those that already carry a goods bubble."""
        g = self.g
        busy = {tg[0] for tg in self.targets()}
        return [lid for lid in TL.TOUR if lid in g.found and lid not in busy and g.can_enter(lid)
                and self.status(lid) == 'unknown']

    def map_draw(self, c, t, x0, y0):
        """A '?' over every found place whose trouble is still to be found, and bobbing bubbles with the
        goods over the places to fly to (world layer, over the fog). Returns the screen point of the
        nearest goods bubble, or None (map_lines draws the rainbows under them first). A place not discovered
        yet gets neither (the gold arrow follows the returned point)."""
        from .world import icon_centre
        near = getattr(self.g.map, 'near', None)
        for lid in self.unknown_places():
            if lid == near:                             # landing there: the prompt shows its trouble chip
                continue
            x, y = _q_tip(lid)
            sx, sy = x - x0, y - y0
            if not (-12 < sx < W + 12 and -2 < sy < H + 16):
                continue
            ph = TL.TOUR.index(lid) * 1.7
            bob = int(round(math.sin(t * 0.07 + ph)))       # 1 px, slow, each place in its own time
            MA.question(c, sx, sy, t, bob=bob)
        best, bd = None, 1e18
        seen_lids = set()
        for lid, _ai, _x, _y, kind, item in self.targets():
            if lid in seen_lids:
                continue
            seen_lids.add(lid)
            if lid not in self.g.found:                 # not discovered yet: no bubble and no arrow give it
                continue                                # away (the plate names it; it is found by exploring)
            x, y = icon_centre(lid)
            sx, sy = int(round(x)) - x0, int(round(y)) - y0 - 22
            d = (sx - W / 2) ** 2 + (sy - H / 2) ** 2
            if d < bd:
                best, bd = (sx, sy), d
            tx, ty = _q_tip(lid, goods=True)            # like the '?': off the places' name plates
            sx, sy = tx - x0, ty - y0 - 9
            if not (-12 < sx < W + 12 and -12 < sy < H + 20):
                continue
            bob = int(round(math.sin(t * 0.15 + len(seen_lids)) * 1.5))
            bx, by = sx - 6, sy - 6 + bob
            fx.bubble(c, bx, by, 13, 11, tail=(sx, sy + 9 + bob), fill='white' if kind == 'pick' else 'butter',
                      border=ui.INK)
            ITEMS[item].draw(c, bx + 3, by + 2)
            if (t // 10) % 3 == 0:
                c.px(bx + 11, by - 1, 'white')
        return best

    def map_hud(self, c, t, screen_xy, y=29, arrow=True):
        self._plate(c, t, y)
        if arrow and screen_xy is not None:
            sx, sy = screen_xy
            if not (0 <= sx < W and 12 <= sy < H - 26):    # off screen, or hidden behind the HUD bands
                ui.edge_arrow(c, sx, sy, t, 'gold')

    def _plate_tg(self, here):
        tg = self.active(here)
        if tg is None and self.carrying:
            tg = (None, 0, 0, 0, 'drop', self.carrying)
        return tg

    def _plate_text(self, tg):
        """(place text, 'X2' or '', '+N' or '') of the plate for target tg."""
        lid, _ai, _x, _y, kind, _item = tg
        s = ('TO ' if kind == 'drop' else 'FROM ') + (place_name(lid) if lid else '?')
        cnt = 'X%d' % self.crates if self.carrying and self.crates > 1 else ''
        more = len({tgt[0] for tgt in self.targets()}) - 1
        return s, cnt, ('+%d' % more if more > 0 else '')

    def _plate_w(self, tg):
        """Width of the full plate for target tg (0: no plate)."""
        if tg is None:
            tg = (None, 0, 0, 0, 'drop', self.carrying) if self.carrying else None
        if tg is None:
            return 0
        s, cnt, more = self._plate_text(tg)
        return (11 + ui.text_width(s, 'tiny') + 5 + (ui.text_width(cnt, 'tiny') + 3 if cnt else 0)
                + (ui.text_width(more, 'tiny') + 3 if more else 0))

    def _plate(self, c, t, y, here=None):
        """[item] FROM PLACE (goods to fetch) / [item] X2 TO PLACE (carrying) +N (more places to go), under
        the counters. here = the Region the balloon is in (None: the world map)."""
        tg = self._plate_tg(here)
        if tg is None:
            return
        item = tg[5]
        s, cnt, more = self._plate_text(tg)
        w = self._plate_w(tg)
        x = W - 3 - w
        hot = self.pick_t is not None and 0 <= t - self.pick_t < 24
        c.round_rect(x, y, w, 11, 'gold' if self.carrying else ui.INK, r=1)
        c.rect(x + 1, y + 1, w - 2, 9, ui.PLATE)
        c.hline(x + 2, x + w - 3, y + 1, ui.PLATE_HI)
        ITEMS[item].draw(c, x + 2, y + 2 - (1 if hot and t - self.pick_t < 8 else 0))
        tx = x + 11
        if cnt:                                         # how many the basket holds
            ui.text(c, tx, y + 3, cnt, 'white', font='tiny')
            tx += ui.text_width(cnt, 'tiny') + 3
        ui.text(c, tx, y + 3, s, 'butter' if self.carrying else 'bubblegum', font='tiny')
        if more:                                        # more places to go: a soft '+N' at the end
            ui.text(c, x + w - 4 - ui.text_width(more, 'tiny'), y + 3, more, 'lavender', font='tiny')

    def _plate_compact(self, c, t, y, r):
        """The plate shrunk to make room for a landmark tag: the goods and a little arrow (towards the
        spot in this place, or up: take off), gold rim when carrying, ink when fetching."""
        tg = self._plate_tg(r)
        if tg is None:
            return
        lid, ai, wx, wy, kind, item = tg
        w = 18
        x = W - 3 - w
        c.round_rect(x, y, w, 11, 'gold' if self.carrying else ui.INK, r=1)
        c.rect(x + 1, y + 1, w - 2, 9, ui.PLATE)
        c.hline(x + 2, x + w - 3, y + 1, ui.PLATE_HI)
        ITEMS[item].draw(c, x + 2, y + 2)
        d = 'up'
        if lid == r.ID:
            if ai != r.ai:
                d = 'right' if ai > r.ai else 'left'
            else:
                sx = (wx - r.cx) if wy is not None else (spot(r, BY_KEY[lid])[0] - r.cx)
                d = 'right' if sx >= W / 2 else 'left'
        ax, ay = x + 13, y + 5
        cc = 'butter' if self.carrying else 'bubblegum'
        if (t // 15) % 2:                               # a gentle nudge in its direction
            ax += {'left': -1, 'right': 1}.get(d, 0)
            ay -= 1 if d == 'up' else 0
        for k in range(3):                              # a 3 px deep arrow head
            if d == 'up':
                c.hline(ax - k, ax + k, ay - 1 + k, cc)
            elif d == 'right':
                c.vline(ax + 1 - k, ay - k, ay + k, cc)
            else:
                c.vline(ax - 1 + k, ay - k, ay + k, cc)

    def overview_draw(self, c, t):
        from .world import icon_centre
        for lid in self.unknown_places():                       # a grey '?': something to find there
            if self.g.fog.is_clear(*icon_centre(lid)):
                MA.tiny_question(c, *_ov_q_xy(lid))
        done = set()
        for lid, _ai, _x, _y, kind, item in self.targets():
            if lid in done:
                continue
            done.add(lid)
            x, y = LW.overview_xy(lid)
            if (t // 6) % 2:
                c.rect(x - 5, y - 15, 11, 10, ui.INK)
                c.rect(x - 4, y - 14, 9, 8, 'butter' if kind == 'drop' else 'white')
            ITEMS[item].draw(c, x - 3, y - 14)
        for lid in self.fixed:                                  # helped places: a little heart
            x, y = LW.overview_xy(lid)
            MA.tiny_heart(c, x + 3, y - 6)

    # -------------------------------------------------------------- the changing map
    def map_lines(self, c, t, x0, y0):
        """The helped world's things in the sky, lit by the clock like the map under them (the world map
        dims what this draws at dusk and night): the thin rainbows from helped places back to their goods'
        sources, rainbows over the route between helped neighbours, and the bunting over helped places."""
        tour = TL.TOUR
        for lid in self.fixed:                          # a thin rainbow back to the place whose goods helped
            tr = BY_KEY.get(lid)
            if tr is not None and tr.item in SOURCES and SOURCES[tr.item][0] != lid:
                _goods_rainbow(lid, SOURCES[tr.item][0], c, t, x0, y0)
        for i in range(len(tour) - 1):
            if tour[i] in self.fixed and tour[i + 1] in self.fixed:
                _rainbow_leg(c, i, t, x0, y0)
        for lid, sx, sy, seed in self._festive_spots(x0, y0):
            MA.map_bunting(c, sx, sy, t, seed=seed)

    def _festive_spots(self, x0, y0):
        from .world import icon_centre
        tour = TL.TOUR
        for lid in self.fixed:
            x, y = icon_centre(lid)
            sx, sy = int(round(x)) - x0, int(round(y)) - y0
            if -40 < sx < W + 40 and -50 < sy < H + 30:
                yield lid, sx, sy, tour.index(lid) if lid in tour else 0

    def map_world(self, c, t, x0, y0, hour):
        """The helped places' festival over the map (drawn over the fog: it is in the sky), after
        map_lines and the goods bubbles: the heart and twinkles, and at night its own lights only (bulbs
        on the bunting, a little firework now and then)."""
        night = festival_night(hour)
        for lid, sx, sy, seed in self._festive_spots(x0, y0):
            MA.map_festive(c, sx, sy, t, night, seed=seed)


# ------------------------------------------------------------------ a helped place celebrates
# Areas with a roof or rock overhead (no balloons or sky lanterns rising there) and under the sea (no
# balloons, no confetti either): the festival keeps to what fits the place.
SHELTERED = {('cloudsdale', 1), ('castle_two_sisters', 1), ('dragon_lands', 1)}
UNDERSEA = {('mount_aris', 2), ('mount_aris', 3)}
# Landmark tags whose anchor is nothing to tie bunting to (a waterfall's face, a hoop in the sky, a point in
# front of a cliff): no string starts or ends there.
NO_TIE = {('rainbow_falls', 'RAINBOW FALLS'), ('rainbow_falls', 'HOOP COURSE'), ('canterlot', 'CANTERLOT CASTLE')}
# Festival mast pegs of a prop on a narrow footing: (inner spread, outer spread or None) in px from its
# centre (default w // 2 + 10, w // 2 + 30). Griffonstone's stall stands at the edge of the treetop.
PEGS = {'griffonstone': (25, None)}


def _festive(r, c, t, ix, iy, night, tr):
    """The festival of a helped place (world layer, under the prop): party balloons (sky lanterns at
    night) rising, pennant bunting between the landmarks of this area that sit on the main layer and the
    festival mast over the mended prop (two pairs of strings down to pegs), a flag on the rooftops, and
    petals and confetti drifting through every area (see SHELTERED / UNDERSEA)."""
    cols = MA.FESTIVE.get(r.ID, MA.RAINBOW)
    tint = r.light_tint()
    seed = TL.TOUR.index(r.ID) if r.ID in TL.TOUR else 0
    where = (r.ID, r.ai)
    if where not in SHELTERED and where not in UNDERSEA:
        MA.balloons_rising(c, t, ix, iy, cols, seed=seed, n=20, tint=tint, night=night)
    pts = []
    for L in r.area.landmarks:
        ax, ay, side, _ln = L.tag
        if not 0.85 <= L.p <= 1.2 or side == 'down':    # far layers: a string there would cross nearer art;
            continue                                    # 'down' tags point at a foot: it would run to the ground
        if (r.ID, L.name) in NO_TIE:
            continue
        pts.append((ax - int(round(ix * L.p)), ay - int(round(iy * (L.p if L.py is None else L.py))), L.p, side))
    mast = None
    if tr is not None:
        w, h = MA.SIZE[tr.kind]
        px, gy = tr.x - ix, tr.ground - iy
        mast = (px, gy, gy - h - 20, w)
        pts.append((px + 1, gy - h - 17, 1.0, 'mast'))   # the bunting ties the mast into the town too
    pts.sort()
    for k in range(len(pts) - 1):
        (x0, y0, p0, s0), (x1, y1, p1, s1) = pts[k], pts[k + 1]
        gap = x1 - x0
        steep = 1.0 if 'mast' in (s0, s1) else 0.6      # a mast can take a steeper string
        if abs(p0 - p1) > 0.1 or not 28 <= gap <= 300 or abs(y1 - y0) > max(24, steep * gap):
            continue
        if x1 < -4 or x0 > W + 4 or max(y0, y1) < -40 or min(y0, y1) > H + 4:
            continue
        MA.pennant_string(c, x0, y0, x1, y1, t, cols, tint, night, seed=seed + k)
    for k, (x, y, _p, side) in enumerate(pts):
        if side == 'up' and -8 < x < W + 8 and 0 < y < H + 8:
            MA.roof_flag(c, x, y, t, cols[(seed + k) % len(cols)], tint, night, seed=k)
    if mast is not None:
        px, gy, top, w = mast
        if -90 < px < W + 90 and -40 < gy < H + 90:
            spread, outer = PEGS.get(tr.key, (w // 2 + 10, w // 2 + 30))
            MA.maypole(c, px, gy, top, t, cols, tint, night, spread=spread, outer=outer)
    if where not in UNDERSEA:
        MA.confetti_drift(c, t, ix, iy, cols, seed=seed, n=24, tint=tint)


# ------------------------------------------------------------------ the map '?'
@lru_cache(maxsize=None)
def _q_tip(lid, goods=False):
    """World px of the tail tip of a place's map '?' (goods: of its bobbing goods bubble, a little bigger).
    Tried in turn: over its icon, just above the icon's top, over its name plate (when that plate sits
    above the icon; clear of the pin on it), then beside the plate like a badge. The first that keeps the
    bubble off every name plate wins; failing that, the first that leaves the places' own plates (the
    tour's) readable."""
    from .world import icon_centre
    from engine import worldmap_icons as WI
    cx, cy = (int(round(v)) for v in icon_centre(lid))
    labels = LW._label_layout()
    plates = {k: (lx, ly, ui.text_width(LW.LOCS[k]['name'].upper(), 'tiny') + 6, 9)
              for k, (lx, ly, _kind) in labels.items()}
    ys = [(cx, cy - (13 if goods else 11))]             # (the goods bubble as it always sat)
    ic = WI.icon(lid)
    top = int(LW.loc_xy(lid)[1]) - ic.ay if ic is not None else cy - 6
    ys.append((cx, top + 1))
    lx, ly, w, _h = plates.get(lid, (cx, cy + 4, 0, 9))
    if ly < top:
        ys.append((lx + w // 2, ly - 9))
    cands = [(x + dx, y) for x, y in ys for dx in (0, -5, 5, -9, 9)]
    if lid in plates:
        cands += [(lx + w + 7, ly + 12), (lx - 8, ly + 12)]

    ox, oy, ow, oh = (6, 17, 14, 20) if goods else (5, 14, 11, 14)   # the bubble (with its bob) and its tail

    def hits(tip, only):
        x, y = tip[0] - ox, tip[1] - oy
        for k, (bx, by, bw, bh) in plates.items():
            if k in only and x < bx + bw and bx < x + ow and y < by + bh and by < y + oh:
                return True
        return False
    for only in (plates, set(TL.TOUR) | {lid}, {lid}):
        for tip in cands:
            if not hits(tip, only):
                return tip
    return cands[0]


@lru_cache(maxsize=None)
def _ov_q_xy(lid):
    """Screen px (top left) of a place's tiny '?' on the overview map: beside its pin, on the first side
    that keeps it off the overview's region labels and the other places' pins, stars and hearts."""
    from engine.font import text_width
    ox, oy, ow, oh = LW.OVERVIEW_RECT
    keep_off = []
    for txt, nx, ny, _cc in getattr(LW, '_OV_LABELS', ()):  # as loc_worldmap.draw_overview lays them out
        tw = text_width(txt, 'tiny')
        lx = int(min(max(ox + int(nx * ow) - tw // 2, ox + 2), ox + ow - tw - 2))
        keep_off.append((lx - 1, oy + int(ny * oh) - 3, tw + 2, 7))
    for k in TL.TOUR:
        if k != lid:
            px, py = LW.overview_xy(k)
            keep_off += [(px - 3, py - 12, 7, 15), (px + 2, py - 7, 5, 5)]   # pin and star, heart
    x, y = LW.overview_xy(lid)
    cands = [(x + 4, y - 9), (x - 7, y - 9), (x + 4, y - 3), (x - 7, y - 3), (x - 1, y + 4), (x - 1, y - 18)]
    for gx, gy in cands:
        if not any(gx - 1 < bx + bw and bx < gx + 4 and gy - 1 < by + bh and by < gy + 6
                   for bx, by, bw, bh in keep_off):
            return gx, gy
    return cands[0]


# ------------------------------------------------------------------ the plate vs the landmark tags
def _tag_box(L, ix, iy, side=None):
    """Screen box (x, y, w, h) a landmark's name tag (engine.atlas.callout, tiny font) covers at
    integer camera (ix, iy), leader and marker included (side: the side it shows on, Region.tag_side)."""
    ax, ay, side0, length = L.tag
    side = side or side0
    x = ax - int(round(ix * L.p))
    y = ay - int(round(iy * (L.p if L.py is None else L.py)))
    pw, ph = ui.text_width(L.name, 'tiny') + 6, 9
    if side == 'up':
        px, py = x - pw // 2, y - 2 - length - ph + 1
    elif side == 'down':
        px, py = x - pw // 2, y + 2 + length
    elif side == 'left':
        px, py = x - 2 - length - pw + 1, y - ph // 2
    else:
        px, py = x + 2 + length, y - ph // 2
    px = max(1, min(W - pw - 1, px))
    py = max(1, min(H - ph - 1, py))
    x0, y0 = min(px, x - 1), min(py, y - 1)
    x1, y1 = max(px + pw, x + 2), max(py + ph, y + 2)
    return x0, y0, x1 - x0, y1 - y0


def _tags_hit(r, t, rect):
    """The bottom edge (screen y) of the lowest landmark tag showing now (or about to) inside rect
    (x, y, w, h), or None."""
    tags = getattr(r, 'tags', None)
    if not tags:
        return None
    A = r.area
    ix, iy = int(math.floor(r.cx)), int(math.floor(r.cy))
    rx, ry, rw, rh = rect
    low = None
    for i, (t0, t1) in tags.items():
        if i >= len(A.landmarks) or (t1 is not None and t > t1 + 6):
            continue
        x, y, w, h = _tag_box(A.landmarks[i], ix, iy, getattr(r, 'tag_side', {}).get(i))
        if x < rx + rw and rx < x + w and y < ry + rh and ry < y + h:
            low = max(low or 0, y + h)
    return low


# ------------------------------------------------------------------ markers
def _pick_marker(c, x, y, t, item):
    """The goods waiting at their source: floating in a soft glow, glinting, with a dotted ring."""
    bob = int(round(math.sin(t * 0.12) * 1.5))
    fx.glow(c, x, y + bob, 10, 'butter', 0.45, rings=3)
    for k in range(10):
        a = k / 10 * 2 * math.pi + t * 0.05
        if (k + t // 4) % 2 == 0:
            c.px(int(round(x + math.cos(a) * 9)), int(round(y + bob + math.sin(a) * 6)), 'white')
    spr = ITEMS[item]
    spr.outlined('white').draw(c, x - spr.w // 2 - 1, y - spr.h // 2 - 1 + bob)
    fx.sparkle(c, x + 5, y - 5 + bob, t % 40, 'white', period=40, size=2)


def _need_bubble(c, x, y, item, t):
    """At a noticed trouble without its goods: a grey bubble with what it needs (no SPACE)."""
    bx, by = int(x - 7), int(y - 18)
    fx.bubble(c, bx, by, 15, 12, tail=(x, y - 1), fill='stone', border=ui.INK)
    ITEMS[item].draw(c, bx + 4, by + 2)
    if (t // 12) % 2:
        c.px(bx + 13, by + 1, 'hot_pink')


def _mend_fx(c, t, a, x, ground, top):
    """The mending magic: sparkles spiral in, a flash, then sparkles drift up for a while."""
    cy = (ground + top) // 2
    if a < MEND:
        k = 1 - a / MEND
        for i in range(12):
            ang = i / 12 * 2 * math.pi + a * 0.35
            r = 6 + 34 * k
            c.rect(int(x + math.cos(ang) * r), int(cy + math.sin(ang) * r * 0.7), 2, 2,
                   ('white', 'gold', 'hot_pink', 'sky')[i % 4])
    elif a < MEND + 8:
        fx.glow(c, x, cy, 26, 'white', 0.7 * (1 - (a - MEND) / 8), rings=4)
    else:
        b = a - MEND
        for i in range(8):
            p = (b * 0.6 + i * 9) % 40
            sx = x + int((i - 4) * 5 + math.sin((b + i * 7) * 0.15) * 2)
            sy = top - int(p)
            if (i + b // 3) % 3:
                c.px(sx, sy, ('white', 'gold', 'hot_pink')[i % 3])


@lru_cache(maxsize=None)
def _leg_arc(i):
    """World px (x ints, top-band y floats) of the rainbow over route leg i (the curve is fixed: built
    once, not every frame)."""
    import numpy as np
    leg = LW._bezier_leg(i)
    (ax, ay), (bx, by) = LW._bez(leg, 0.0), LW._bez(leg, 1.0)
    length = math.hypot(bx - ax, by - ay)
    hgt = min(70, 0.3 * length)
    n = max(8, int(length * 1.5))
    us = np.linspace(0.04, 0.96, n)
    pts = np.array([LW._bez(leg, float(u)) for u in us], np.float32)
    lift = np.sin(np.pi * us) * hgt + 26
    return np.round(pts[:, 0]).astype(int), pts[:, 1] - lift


def _rainbow_leg(c, i, t, x0, y0):
    """A rainbow arcing over route leg i (between two helped neighbours), drawn translucent."""
    import numpy as np
    wx, wy = _leg_arc(i)
    n = len(wx)
    xs = wx - x0
    if xs.max() < -5 or xs.min() > W + 5:
        return
    base = wy - y0
    a = c.a
    for k, name in enumerate(('red', 'orange', 'butter', 'leaf', 'sky', 'violet')):
        ys = np.round(base + k).astype(int)
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if not m.any():
            continue
        cc = np.array(_col(name), np.float32)
        px = a[ys[m], xs[m]].astype(np.float32)
        a[ys[m], xs[m]] = (px * 0.3 + cc * 0.7).astype(np.uint8)
    j = int((t * 0.6) % n)                                       # a glint running along the arc
    if 0 <= xs[j] < W and 0 <= int(base[j]) < H - 2:
        c.px(int(xs[j]), int(base[j]) + 2, 'white')


@lru_cache(maxsize=None)
def _goods_arc(lid, src):
    """World px (xs, ys, alpha, bbox) of the arc from a helped place's goods' source up and over to its
    icon: vivid near both places, fading out in the middle of a long one (the map stays readable)."""
    import numpy as np
    from .world import icon_centre
    (bx, by), (ax, ay) = icon_centre(lid), icon_centre(src)
    ay, by = ay - 9, by - 9                             # out of the top of each icon
    d = math.hypot(bx - ax, by - ay)
    hgt = min(96.0, 0.3 * d + 10)
    n = max(8, int(d * 1.6))
    u = np.linspace(0.0, 1.0, n)
    xs = np.round(ax + (bx - ax) * u).astype(int)
    ys = np.round(ay + (by - ay) * u - np.sin(np.pi * u) * hgt).astype(int)
    keep = np.ones(n, bool)                             # one point per pixel step (no doubled pixels)
    keep[1:] = (np.diff(xs) != 0) | (np.diff(ys) != 0)
    for k, (lx, ly, _kind) in LW._label_layout().items():   # it passes behind the places' name plates
        pw = ui.text_width(LW.LOCS[k]['name'].upper(), 'tiny') + 6
        keep &= ~((xs >= lx - 1) & (xs <= lx + pw) & (ys + 2 >= ly - 1) & (ys <= ly + 9))
    xs, ys, u = xs[keep], ys[keep], u[keep]
    if len(xs) == 0:
        return xs, ys, np.zeros(0, np.float32), (10 ** 9, 10 ** 9, -10 ** 9, -10 ** 9)
    e = np.minimum(u, 1 - u) * (d + 2 * hgt)            # px along the arc from the nearer end
    alpha = (0.62 - 0.42 * np.clip((e - 50) / 110.0, 0, 1)).astype(np.float32)
    return xs, ys, alpha, (int(xs.min()), int(ys.min()) - 2, int(xs.max()), int(ys.max()) + 3)


_ARC_COLS = None


def _goods_rainbow(lid, src, c, t, x0, y0):
    """A thin, translucent three-band rainbow arcing from src, the place that gave the goods, to the
    helped place lid; a glint runs along it from the helper to the helped (world layer of the map)."""
    import numpy as np
    global _ARC_COLS
    xs, ys, al, (bx0, by0, bx1, by1) = _goods_arc(lid, src)
    if bx1 - x0 < 0 or bx0 - x0 >= W or by1 - y0 < 0 or by0 - y0 >= H:
        return
    if _ARC_COLS is None:
        _ARC_COLS = [np.array(_col(n), np.float32) for n in ('hot_pink', 'butter', 'sky')]
    a = c.a
    sx = xs - x0
    for k, cc in enumerate(_ARC_COLS):
        sy = ys - y0 + k
        m = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        if m.any():
            px = a[sy[m], sx[m]].astype(np.float32)
            w = al[m][:, None]
            a[sy[m], sx[m]] = (px * (1 - w) + cc * w).astype(np.uint8)
    n = len(xs)
    j = int((t * 0.8) % (n + 40))                       # the glint: from the source to the helped place
    if j < n:
        gx, gy = int(sx[j]), int(ys[j] - y0) + 1
        if 0 <= gx < W and 0 <= gy < H:
            c.px(gx, gy, 'white')


def _col(name):
    from engine import col
    return col(name)
