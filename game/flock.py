"""Balloon friends: the crewless balloons drifting over the world map (game/world.py TRAFFIC) can be met.

Each one loops slowly around its home place, never closer than LOOP_CLEAR px to the place's icon, so
the landing spot stays free. A friend you haven't met yet carries a little pink heart over its envelope.
Fly alongside one (ground points within REACH) and a bubble pops up beside it (on the side away from
your balloon); SPACE greets it: both burners puff and a burst of hearts flies between the two
balloons. The first greeting adds it to the FRIENDS page of the journal (meet all 15: a badge and the
FLOCK balloon pattern) and its toast says what a second SPACE does. After that, SPACE asks it to do
its job:

  carrier  goods hang under it; if a trouble you noticed needs them and your basket is empty, it
           hands them over (a shortcut instead of flying to the source place). It restocks later.
  racer    challenges you to a race to a nearby place: it sets off at once; reach that place's icon
           first to win (beating every racer: the SKY RACER badge).
  guide    leads the way to where you are needed: a place you have found whose trouble you haven't
           noticed yet, else the nearest undiscovered place, else a place you visited with stamps or
           sights left, else a race start gate never flown, else a map rainbow ring never flown. It
           flies a little slower than your balloon, trailing sparkles, waits there for a while, then
           drifts home.

Landing wins: while the land plate of a place is up (MapMode.near), no friend takes SPACE and its
bubble hides. When helping places sends a new balloon into the sky, a toast says near which place.

No crew: the balloons themselves are the friends (no creatures). Positions are pure functions of the
step and the friend's current outing (`pos(k, t)`), so the 60 fps interpolation stays smooth. The
bubble, the hearts and the greeting burst are world-anchored art over the balloons (Frame.front).
"""
import math

from engine import W, H, Canvas, Sprite, fx, hash01, ease_in_out, text, text_width
from engine import loc_worldmap as LW
from engine import timeline as TL

from . import ui, perks
from .compose import KEY, bob_dy
from .items import ITEMS, ITEM_NAMES

REACH = 22                      # balloon ground point <-> friend ground point
RESTOCK = 2400                  # steps before a carrier has goods again
HOLD = 210                      # steps a guide / racer waits at its goal before drifting home
HOME = 150                      # steps of the drift back to its loop
GREET = 30                      # steps of the greeting effect
FIRST_GREET = 44                # ... of the bigger burst of the first meeting
LOOP_CLEAR = 30                 # a home loop keeps this far (ground px) from its home icon's centre
GUIDE_UP = 26                   # a guide waits this far north of a place's icon (not on the landing spot)
NEW_ARROW = 300                 # steps an edge arrow points at a balloon that has just taken to the sky

# per TRAFFIC index: (name, job, goods for carriers / race speed px per step for racers)
FRIENDS = [
    ('CIRRUS', 'guide', None),
    ('BOWLER', 'carrier', 'cider'),
    ('SUNDOWN', 'racer', 2.6),
    ('PRISM', 'carrier', 'lantern'),
    ('NEON', 'racer', 2.7),
    ('LEMON DROP', 'carrier', 'crystal'),
    ('ORCHARD', 'guide', None),
    ('MINTY', 'carrier', 'seeds'),
    ('SEASPRAY', 'racer', 2.8),
    ('SPECTRUM', 'guide', None),
    ('NIGHTCAP', 'carrier', 'rainbow'),
    ('NORTHWIND', 'racer', 2.9),
    ('GALE', 'carrier', 'fireworks'),
    ('SANDY', 'guide', None),
    ('SCEPTRE', 'racer', 3.0),
]
JOB_TEXT = {'carrier': 'IT CARRIES ', 'racer': 'IT LOVES TO RACE', 'guide': 'IT KNOWS THE WAY'}
RACERS = [k for k, f in enumerate(FRIENDS) if f[1] == 'racer']
RACE_WAIT = 24                      # steps a racing friend waits after the GO toast (a human's reaction time)

# the heart an unmet friend carries over its envelope (5x5, outlined: 7x7)
_HEART = Sprite.from_ascii('''
    .p.p.
    pwppp
    ppppp
    .ppp.
    ..p..''', {'p': 'hot_pink', 'w': 'white'}).outlined('wine')

_ICONS = {}


def _ground_up(size):
    return 16 if size in ('s', 'xs') else 20


def _home_icon(lid):
    if lid not in _ICONS:
        from .world import icon_centre
        _ICONS[lid] = icon_centre(lid)
    return _ICONS[lid]


_LOOPS = {}
_WIDE = (52, 8, 35, 4)              # a loop around the home icon: rx = 52 +- 8, ry = 35 +- 4
_SIDE = (30, 4, 19, 2)              # a smaller loop beside it (places in a cluster)


def _loop(k):
    """(centre x, centre y, rx, rx swing, ry, ry swing) of friend k's home loop. A wide loop around its
    home icon when that keeps LOOP_CLEAR px from every place's icon; else (a cluster of places, like
    Ponyville and Sweet Apple Acres) the smaller loop beside the home icon that stays clearest of all
    of them. Picked once, from fixed candidates (no randomness)."""
    if k not in _LOOPS:
        import numpy as np
        from .world import TRAFFIC
        icons = np.array([_home_icon(lid) for lid in TL.TOUR], float)
        hx, hy = _home_icon(TRAFFIC[k][0])
        u = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        qx, qy = np.cos(u), np.sin(u)

        def clear(cx, cy, shape):
            """The closest the loop (at every swing extreme) comes to a place's icon (-1: off the map)."""
            rx, sx, ry, sy = shape
            px = np.concatenate([cx + (rx + a * sx) * qx for a in (-1, 1) for _b in (-1, 1)])
            py = np.concatenate([cy + (ry + b * sy) * qy for _a in (-1, 1) for b in (-1, 1)])
            if px.min() < 24 or py.min() < 24 or px.max() > LW.MAP_W - 24 or py.max() > LW.MAP_H - 24:
                return -1.0
            return float(np.hypot(px[:, None] - icons[None, :, 0], py[:, None] - icons[None, :, 1]).min())
        ok = LOOP_CLEAR                             # clear enough: then the closest loop to home wins
        d = clear(hx, hy, _WIDE)
        best = (min(d, ok), 0, hx, hy, _WIDE)
        if d < ok:
            for j in range(16):
                for dist in (56, 66, 76):
                    a = j * math.pi / 8
                    cx, cy = hx + dist * math.cos(a), hy + dist * 0.8 * math.sin(a)
                    cand = (min(clear(cx, cy, _SIDE), ok), -dist, cx, cy, _SIDE)
                    if cand[:2] > best[:2]:
                        best = cand
        _d, _o, cx, cy, shape = best
        _LOOPS[k] = (cx, cy) + shape
    return _LOOPS[k]


def home_pos(k, t):
    """Ground point of friend k on its home loop at (fractional) step t: a slow, breathing ellipse
    around (or, in a cluster of places, beside) its home place's icon (half the friends turn one way,
    half the other) that keeps LOOP_CLEAR px from the icons, so the landing spots stay free."""
    from .world import TRAFFIC
    ph = TRAFFIC[k][3]
    cx, cy, rx, sx, ry, sy = _loop(k)
    a = (t * 0.0031 * (52 / rx) + ph) * (1 if k % 2 else -1)       # the same ground speed on a small loop
    rx = rx + sx * math.sin(t * 0.0107 + 2 * ph)
    ry = ry + sy * math.sin(t * 0.0023 + ph * 1.7)
    return cx + rx * math.cos(a), cy + ry * math.sin(a)


class Flock:
    def __init__(self, game):
        self.g = game
        self.met = set()                # TRAFFIC indices greeted at least once
        self.beaten = set()             # racers the player has beaten
        self.stock_t = {}               # carrier -> step it gave its goods away
        self.out = {}                   # k -> outing: dict(kind, t0, sx, sy, gx, gy, v, lid)
        self.greet_t = {}               # k -> step of the last greeting (effect)
        self.greet_xy = {}              # k -> world point between the two envelopes at that greeting
        self.first_t = {}               # k -> step of its first greeting (the bigger burst)
        self.near = None
        self.race = None                # dict(k, lid, t0) while racing a balloon
        self.result = None              # (step, won, k) of the last balloon race
        self.new_t = {}                 # k -> step it took to the sky (a help unlocked it)
        self._vis = None                # balloons in the sky at the last map step (None: not counted yet)
        for k in range(len(FRIENDS)):   # pick the home loops now (a few ms), not on the first map frame
            _loop(k)

    # -------------------------------------------------------------- save
    def save(self):
        return dict(met=sorted(self.met), beaten=sorted(self.beaten))

    def load(self, d):
        d = d or {}
        n = len(FRIENDS)
        self.met = {int(k) for k in d.get('met', ()) if 0 <= int(k) < n}
        self.beaten = {int(k) for k in d.get('beaten', ()) if int(k) in RACERS}
        self._vis = self.visible()      # the help is loaded first: no 'new balloon' toast for old ones

    # -------------------------------------------------------------- motion
    def visible(self):
        from .world import traffic_count
        return traffic_count(len(self.g.quests.fixed))

    def pos(self, k, t):
        """Ground point of friend k at (fractional) step t."""
        o = self.out.get(k)
        if o is None:
            return home_pos(k, t)
        a = max(0.0, t - o['t0'])                       # (a racer waits at its start until t0)
        d = math.hypot(o['gx'] - o['sx'], o['gy'] - o['sy'])
        travel = max(1.0, d / o['v'])
        if a < travel:
            u = a / travel
            u = u * u * (3 - 2 * u) if o['kind'] == 'guide' else min(1.0, u)
            return o['sx'] + (o['gx'] - o['sx']) * u, o['sy'] + (o['gy'] - o['sy']) * u
        b = a - travel
        hx = o['gx'] + 7 * math.sin(b * 0.05)
        hy = o['gy'] + 3 * math.cos(b * 0.05)
        if b < HOLD:
            return hx, hy
        k2 = ease_in_out(min(1.0, (b - HOLD) / HOME))
        tx, ty = home_pos(k, t)
        return hx + (tx - hx) * k2, hy + (ty - hy) * k2

    def arrive_t(self, k):
        o = self.out[k]
        return o['t0'] + math.hypot(o['gx'] - o['sx'], o['gy'] - o['sy']) / o['v']

    def _send(self, k, kind, gx, gy, v, lid=None, wait=0):
        """Send friend k off to (gx, gy) at v px per step, after `wait` steps."""
        t = self.g.t
        sx, sy = self.pos(k, t)
        self.out[k] = dict(kind=kind, t0=t + wait, sx=sx, sy=sy, gx=gx, gy=gy, v=v, lid=lid)

    def hidden(self, k, t):
        """A friend inside uncharted fog is not seen (unless it is out leading you or racing you)."""
        if k in self.out:
            return False
        x, y = self.pos(k, t)
        return not self.g.fog.is_clear(x, y)

    def has_goods(self, k):
        name, job, item = FRIENDS[k]
        return job == 'carrier' and self.g.t - self.stock_t.get(k, -10 ** 6) >= RESTOCK

    # -------------------------------------------------------------- update (world map)
    def map_step(self, m, inp):
        """Called every map step before landing (MapMode has just set m.near, the place whose land
        plate is up). Returns True when SPACE went to a balloon friend. Landing wins: while m.near is
        set no friend takes SPACE (and its bubble hides)."""
        g = self.g
        t = g.t
        self._new_balloons()
        for k in list(self.out):                                  # outings that are over
            if t - self.arrive_t(k) > HOLD + HOME:
                del self.out[k]
        self._race_step(m)
        self._guide_step(m)
        self.near = None
        if m.landing is not None or m.takeoff is not None:
            return False
        if getattr(m, 'near', None) is not None:                  # the land plate is up
            return False
        best, bd = None, REACH * perks.friend_reach(self.g)
        for k in range(self.visible()):
            if self.hidden(k, t):
                continue
            x, y = self.pos(k, t)
            d = math.hypot(m.bx - x, m.by - y)
            if d < bd:
                best, bd = k, d
        self.near = best
        if best is None or not inp.action:
            return False
        self._greet(best, m)
        return True

    def _new_balloons(self):
        """Helping places sends more balloons into the sky: say near which place each new one flies."""
        n = self.visible()
        if self._vis is None:
            self._vis = n
        from .world import TRAFFIC
        while self._vis < n:
            k = self._vis
            self._vis += 1
            self.new_t[k] = self.g.t
            name = LW.LOCS[TRAFFIC[k][0]]['name'].upper()
            self.g.toasts.push('A NEW BALLOON IS FLYING', 'NEAR ' + name, ui.icon_balloon, sound='chime', gain=0.4)

    def _greet(self, k, m):
        g = self.g
        name, job, item = FRIENDS[k]
        self.greet_t[k] = g.t
        fx_, fy_ = self.pos(k, g.t)
        from .world import TRAFFIC
        up = _ground_up(TRAFFIC[k][2])
        self.greet_xy[k] = ((fx_ + m.bx) / 2, (fy_ - up - 8 + m.by - m.alt - 12) / 2)
        m.hop_t = g.t
        g.sfx('puff', 0.45)
        g.sfx('sparkle', 0.35)
        if k not in self.met:
            self.met.add(k)
            self.first_t[k] = g.t
            nxt = {'guide': 'SPACE AGAIN: FOLLOW ME', 'racer': 'SPACE AGAIN: RACE ME'}.get(job)
            if nxt is None:
                nxt = 'SPACE AGAIN: TAKE MY ' + ITEM_NAMES[item]
            g.toasts.push('MET ' + name, nxt, ui.ICONS['heart'], sound='chime', gain=0.5, urgent=True)
            if len(self.met) == len(FRIENDS):
                g.toasts.push('NEW BALLOON', 'FLOCK PATTERN (J)', ui.icon_balloon, sound='quest')
            g.save()
            return
        if job == 'carrier':
            q = g.quests
            if not self.has_goods(k):
                g.toasts.push(name, 'NOTHING LEFT TO SHARE TODAY', ui.ICONS['heart'], urgent=True)
            elif item not in q.needed():
                g.toasts.push(name, 'NO PLACE NEEDS ' + ITEM_NAMES[item] + ' YET', ITEMS[item], urgent=True)
            elif q.carrying:
                g.toasts.push(name, 'YOUR BASKET IS FULL', ITEMS[item], urgent=True)
                g.sfx('tick', 0.35)
            else:
                q.carrying = item
                q.pick_t = g.t
                self.stock_t[k] = g.t
                g.toasts.push(name + ' SHARES', ITEM_NAMES[item] + ' FOR YOUR BASKET', ITEMS[item], sound='coin',
                              urgent=True)
                g.save()
        elif job == 'racer':
            self._start_race(k)
        else:
            self._start_guide(k, m)

    # -------------------------------------------------------------- balloon races
    def _start_race(self, k):
        g = self.g
        if self.race is not None or g.races.run is not None or k in self.out:
            g.sfx('tick', 0.35)
            return
        from .world import icon_centre
        x0, y0 = self.pos(k, g.t)
        best = None
        for lid in TL.TOUR:
            x, y = icon_centre(lid)
            d = math.hypot(x - x0, y - y0)
            if 220 <= d <= 460 and (best is None or abs(d - 330) < best[0]):
                best = (abs(d - 330), lid, x, y)
        if best is None:
            return
        _d, lid, gx, gy = best
        self._send(k, 'race', gx, gy, FRIENDS[k][2], lid, wait=RACE_WAIT)
        self.race = dict(k=k, lid=lid, t0=g.t)
        self.result = None
        g.toasts.push('RACE TO ' + LW.LOCS[lid]['name'].upper() + '!', 'FLY! ' + FRIENDS[k][0] + ' IS RIGHT BEHIND',
                      _flag, sound='whistle', urgent=True)

    def _race_step(self, m):
        r = self.race
        if r is None:
            return
        g = self.g
        from .world import icon_centre, FOCUS_DY
        gx, gy = icon_centre(r['lid'])
        if m.landing is not None or r['k'] not in self.out:   # landed elsewhere / its outing is over
            self.race = None
            return
        mine = math.hypot(m.bx - gx, m.by - FOCUS_DY - gy) < 22         # the balloon itself on the icon
        theirs = g.t >= self.arrive_t(r['k'])
        if not mine and not theirs:
            return
        k = r['k']
        self.race = None
        self.result = (g.t, mine, k)
        name = FRIENDS[k][0]
        if mine:
            first = k not in self.beaten
            self.beaten.add(k)
            m.hop_t = g.t
            g.sfx('jackpot', 0.5)
            g.sfx('firework', 0.35)
            g.toasts.push('YOU BEAT ' + name + '!', 'WELL FLOWN' + ('' if not first else ' - A RIBBON FOR YOU'),
                          _ribbon, urgent=True)
            if first and self.beaten == set(RACERS):
                g.toasts.push('EVERY RACER BEATEN', 'THE FASTEST BALLOON IN THE SKY', _ribbon, sound='quest')
            g.save()
        else:
            g.sfx('tick', 0.4)
            g.toasts.push(name + ' WINS', 'GREET IT TO RACE AGAIN', _flag, urgent=True)

    # -------------------------------------------------------------- guides
    def _guide_target(self, x0, y0):
        """(x, y, lead, arrived) of where a guide leads, or None. By priority: the errand under way (where
        the plate points: goods to pick up or a place to bring them), a place you have found
        whose trouble you haven't noticed yet, an undiscovered place, a visited place with stamps or
        sights still to find, a race start gate never flown, a map rainbow ring never flown (in each
        kind the nearest one at least 60 px away). lead / arrived: the toast lines on setting off /
        on arrival."""
        from .world import icon_centre, RINGS
        from .races import COURSES
        from .quests import BY_KEY as TROUBLE
        g = self.g

        def place(lid):
            x, y = icon_centre(lid)
            return x, y - GUIDE_UP, LW.LOCS[lid]['name'].upper()

        def left_to_find(lid):
            n, tot, s, stot = g.place_progress(lid)
            return (tot is not None and n < tot) or (stot is not None and s < stot)

        job = g.quests.active()
        tiers = (
            lambda: [(*place(job[0])[:2], 'IT KNOWS THE WAY TO ' + place(job[0])[2], 'YOUR ERRAND IS HERE')]
            if job is not None and job[0] in LW.LOCS else [],
            lambda: [(*place(lid)[:2], place(lid)[2] + ' NEEDS HELP', 'LAND AND LOOK FOR THE ?')
                     for lid in TL.TOUR if lid in g.found and lid in TROUBLE and g.quests.status(lid) == 'unknown'],
            lambda: [(*place(lid)[:2], 'IT KNOWS WHERE ' + place(lid)[2] + ' IS', FRIENDS_FOUND + place(lid)[2])
                     for lid in TL.TOUR if lid not in g.found],
            lambda: [(*place(lid)[:2], 'MORE TO FIND IN ' + place(lid)[2], 'STAMPS AND SIGHTS ARE LEFT HERE')
                     for lid in TL.TOUR if lid in g.visited and left_to_find(lid)],
            lambda: [(*co.gates[0], 'IT KNOWS WHERE A RACE START IS', FRIENDS_FOUND + 'A RACE START')
                     for co in COURSES if co.where == 'map' and co.key not in g.races.best],
            lambda: [(x, y, 'IT KNOWS WHERE A RAINBOW RING IS', FRIENDS_FOUND + 'A RAINBOW RING')
                     for i, (x, y) in enumerate(RINGS) if i not in g.rings_flown],
        )
        for tier in tiers:
            best = None
            for x, y, lead, here in tier():
                d = math.hypot(x - x0, y - y0)
                if d > 60 and (best is None or d < best[0]):
                    best = (d, x, y, lead, here)
            if best is not None:
                return best[1:]
        return None

    def _start_guide(self, k, m):
        g = self.g
        if k in self.out:
            g.sfx('tick', 0.35)
            return
        x0, y0 = self.pos(k, g.t)
        tg = self._guide_target(x0, y0)
        name = FRIENDS[k][0]
        if tg is None:
            g.toasts.push(name, 'YOU HAVE FOUND EVERY CORNER!', ui.ICONS['heart'], urgent=True)
            return
        gx, gy, lead, here = tg
        self._send(k, 'guide', gx, gy, 1.35, here.replace(FRIENDS_FOUND, name + ' FOUND '))
        g.toasts.push('FOLLOW ' + name, lead, ui.icon_map, sound='whoosh', urgent=True)

    def _guide_step(self, m):
        g = self.g
        for k, o in self.out.items():
            if o['kind'] != 'guide' or o.get('told'):
                continue
            if g.t >= self.arrive_t(k) and math.hypot(m.bx - o['gx'], m.by - o['gy']) < 40:
                o['told'] = True
                g.sfx('sparkle', 0.5)
                g.toasts.push('HERE IT IS!', o['lid'], ui.ICONS['heart'])

    # -------------------------------------------------------------- draw (world map)
    def draw_map(self, c, fr, t, alpha, x0, y0, tint=None):
        """The friends (sprites, with their goods), guide trails (world layer); the hearts of unmet
        friends, greeting bursts and the SPACE bubble (over the balloons: fr.front)."""
        from .world import TRAFFIC
        px = fr.focus[0] - x0 if fr.focus is not None else W / 2        # the player's balloon (screen x)
        for k in range(self.visible()):
            if self.hidden(k, t):
                continue
            lid, pat, size, _ph = TRAFFIC[k]
            tx, ty = self.pos(k, t + alpha)
            up = _ground_up(size)
            sx, sy = int(round(tx)) - x0, int(round(ty)) - y0
            o = self.out.get(k)
            if o is not None and o['kind'] == 'guide' and t + alpha < self.arrive_t(k) + 20:
                for j in range(1, 9):                              # a sparkle trail behind a guide
                    qx, qy = self.pos(k, t - j * 5)
                    if (j + t // 3) % 2:
                        c.px(int(round(qx)) - x0, int(round(qy)) - y0 - up - 4, ('white', 'gold', 'hot_pink')[j % 3])
            ga = t - self.greet_t.get(k, -10 ** 6)
            fa = t - self.first_t.get(k, -10 ** 6)
            if 0 <= ga < FIRST_GREET:                              # the burst stays where it went off
                gx, gy = self.greet_xy[k]
                first = 0 <= fa < FIRST_GREET
                _hearts(_front(fr), t, self.greet_t[k], int(round(gx)) - x0, int(round(gy)) - y0, seed=k,
                        big=first)
            if not (-20 < sx < W + 20 and -40 < sy < H + 30):
                continue
            flame = (ga < 20 and (ga // 3) % 2 == 0) or (o is not None and o['kind'] == 'race')
            cargo = FRIENDS[k][2] if self.has_goods(k) else None
            fr.shadow(tx, ty, 3 if size == 's' else 5)
            fr.balloon(tx, ty - up, t + alpha + k * 9, size, pattern=pat, flame=flame, cargo=cargo, tint=tint)
            top = sy - up - (14 if size in ('s', 'xs') else 21)   # the envelope's top (screen)
            # the heart and the bubble ride the balloon (Frame.ride): its sub-pixel offset and bob, so they
            # glide with it instead of stepping a whole pixel against it
            rdx = tx - round(tx)
            rdy = ty - round(ty) + bob_dy(t + alpha + k * 9, size)
            if k not in self.met and k != self.near and not (0 <= fa < FIRST_GREET):  # unmet: a heart
                bob = 1 if (t // 12) % 2 else 0                    # (the bubble carries it when alongside)
                fr.ride(_heart_layer(), sx - 3, top - 6 - _HEART.h - bob, rdx, rdy)
            if k == self.near:
                lay = Canvas(color=KEY)
                _side_bubble(lay, sx, top, px, self.bubble_icon(k), t)
                fr.ride(lay, 0, 0, rdx, rdy)

    def bubble_icon(self, k):
        name, job, item = FRIENDS[k]
        if k not in self.met:
            return 'heart'
        if job == 'carrier':
            return ITEMS[item] if self.has_goods(k) else 'heart'
        if job == 'racer':
            from .races import FLAG
            return FLAG
        return 'flag'

    def hud(self, c, t, x0, y0):
        """Balloon race plate (top-left) + an edge arrow to the goal; the guide's arrow; an arrow to a
        balloon that has just taken to the sky."""
        r = self.race
        from .world import icon_centre
        if r is not None:
            k = r['k']
            name = LW.LOCS[r['lid']]['name'].upper()
            s = 'RACE TO ' + name
            w = text_width(s, 'tiny') + 12
            c.round_rect(3, 3, w, 11, ui.INK, r=1)
            c.rect(4, 4, w - 2, 9, ui.PLATE)
            _flag(c, 5, 4)
            text(c, 13, 6, s, 'butter', font='tiny')
            gx, gy = icon_centre(r['lid'])
            sx, sy = gx - x0, gy - y0
            if not (0 <= sx < W and 0 <= sy < H):
                ui.edge_arrow(c, sx, sy, t, 'sky')
        for k, o in self.out.items():
            if o['kind'] == 'guide' and not o.get('told'):
                px, py = self.pos(k, t)
                sx, sy = px - x0, py - y0 - 20
                if not (0 <= sx < W and 0 <= sy < H):
                    ui.edge_arrow(c, sx, sy, t, 'mint')
        for k, t0 in self.new_t.items():
            if 0 <= t - t0 < NEW_ARROW and k not in self.met and k not in self.out:
                px, py = self.pos(k, t)
                sx, sy = px - x0, py - y0 - 20
                if not (0 <= sx < W and 0 <= sy < H):
                    ui.edge_arrow(c, sx, sy, t, 'bubblegum')


FRIENDS_FOUND = '\0'                # stands for '<NAME> FOUND ' in a guide's arrival line


_HEART_LAYER = []


def _heart_layer():
    """The unmet friend's heart on its own small KEY canvas (cached), for Frame.ride."""
    if not _HEART_LAYER:
        lay = Canvas(_HEART.w, _HEART.h, color=KEY)
        _HEART.draw(lay, 0, 0)
        _HEART_LAYER.append(lay)
    return _HEART_LAYER[0]


def _front(fr):
    """The frame's world-anchored layer over the balloons (made on first use)."""
    if fr.front is None:
        from .compose import KEY
        fr.front = Canvas(color=KEY)
    return fr.front


def _side_bubble(c, x, top, px, icon_name, f):
    """The friend's [icon][SPACE] bubble beside its envelope (top = the envelope's top, screen px), on
    the side away from the player's balloon (screen x px), its tail pointing at the envelope."""
    kw = ui.keycap_width('SPACE')
    w, h = 9 + 3 + kw + 2, 11
    side = 1 if x >= px else -1
    bob = 1 if (f // 10) % 2 else 0
    bx = x + 7 if side > 0 else x - 7 - w
    if not 2 <= bx <= W - w - 3:                    # no room on that side: the other one
        side = -side
        bx = x + 7 if side > 0 else x - 7 - w
    bx = max(2, min(W - w - 3, bx))
    by = top - h - 3 - bob
    tail = (x + 2 * side, top + 1)
    if by < 3:                                      # no room above: hang it beside the envelope, lower
        by = top + 4
        tail = (x + 4 * side, top + 3)
        bx = x + 9 if side > 0 else x - 9 - w
        bx = max(2, min(W - w - 3, bx))
        fx.bubble(c, bx, by, w, h, fill='white', border=ui.INK)
        c.hline(min(tail[0], bx if side > 0 else bx + w), max(tail[0], bx if side > 0 else bx + w), by + h // 2,
                ui.INK)
    else:
        fx.bubble(c, bx, by, w, h, tail=tail, fill='white', border=ui.INK)
    ui.icon(c, icon_name, bx + 1, by + 1)
    ui.keycap(c, bx + 12, by + 1, 'SPACE', lit=(f // 10) % 2 == 0)


def _hearts(c, t, t0, x, y, seed=0, big=False):
    """The greeting: hearts and sparks fly out between the two balloons, a big heart rises (a bigger
    burst for the first meeting)."""
    a = t - t0
    life = FIRST_GREET if big else GREET
    if not 0 <= a < life:
        return
    n, power = (30, 1.0) if big else (16, 0.6)
    cols = ('hot_pink', 'white', 'butter', 'pink')
    for i in range(n):
        ang = hash01(seed + 300, i) * 6.283
        v = power * (0.8 + 1.6 * hash01(seed + 300, i, 2))
        qx = int(x + math.cos(ang) * v * a)
        qy = int(y + math.sin(ang) * v * a * 0.8 - 0.01 * a * a)
        if a > life * 0.6 and (i + a) % 3 == 0:          # wink out towards the end
            continue
        if i % 3 == 0 and a < life * 0.75:               # every third spark is a little heart
            cc = 'hot_pink' if i % 2 else 'pink'
            for hx, hy in ((0, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
                c.px(qx - 1 + hx, qy - 1 + hy, cc)
        else:
            s = 2 if a < life * 0.5 else 1
            c.rect(qx, qy, s, s, cols[i % len(cols)])
    if a < 18 + (10 if big else 0):
        from .mend_art import heart_icon
        heart_icon(c, x - 4, y - 6 - a // 2)


def _flag(c, x, y):
    from .races import FLAG
    FLAG.draw(c, x + 1, y)


def _ribbon(c, x, y):
    c.rect(x + 3, y, 3, 5, 'hot_pink')
    c.circle(x + 4, y + 3, 3, 'gold')
    c.px(x + 4, y + 3, 'butter')
    c.px(x + 2, y + 7, 'red')
    c.px(x + 6, y + 7, 'red')
