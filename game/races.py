"""Ring races: timed courses of gates, on the world map and inside two places.

A course is a list of gates (world px; on the map a gate's point is the ground under the ring, like
the map's rainbow rings; in a place it is the envelope centre on the area's main layer). Gate 0 is
the start gate (chequered): fly through it and the clock starts. Then fly through every gate in order
(the next one pulses gold with an arrow over it; the one after shows faintly; an edge arrow points
to the next gate when it is off screen). The last gate stops the clock: a medal (gold / silver /
bronze, by the course's par times) and the best time are saved; the result plate also says how much
faster the next medal (or your best time) is. The run ends if no gate is flown for TIMEOUT steps (a
clock ticks, faster and faster, through its last 5 s; a run that never got past the start gate just
stops, with no 'too slow'), when the balloon lands / takes off / changes area, or on ESC. Gate cues
climb in pitch as the course goes on, and a whistle greets the finish. A start gate works only after
the balloon has been outside it (so a save made inside one does not start a race on loading).

Par times: gold = the autopilot's time (tools/phaseb_race_cal.py) x 1.3, silver x 1.6, bronze x 2.2.

Courses: four on the world map, the hoop course at Rainbow Falls (through the film's cloud hoops)
and a slalom along the Las Pegasus strip.
"""
import math
from dataclasses import dataclass

from engine import W, H, Sprite, fx, seg, ease_out_back, text, text_width

from . import ui, perks

TIMEOUT = 20 * 30               # steps without a gate before a run is called off
MAP_REACH = 11                  # map: envelope centre <-> ring centre (as ground points: + MAP_RING_UP)
REGION_REACH = 12               # place: envelope centre <-> gate centre
MAP_RING_UP = 34                # a map gate's ring floats this far above its ground point
REARM = 45                      # steps after a run ends before the start gate works again
RESULT = 150                    # steps the result plate shows
HURRY, HURRY2 = 150, 60         # the last steps of the timeout: a tick every 15 steps, then every 8
# the cue of each gate flown, climbing in pitch: the 'pop' variants low to high (A5 B5 D6 F#6, see
# audio/sfx.py), then the 'blip' (D6 -> A6); the finish gate has its own fanfare (and a 'whistle')
GATE_CUES = (('pop', 0), ('pop', 3), ('pop', 1), ('pop', 2), ('blip', 0))


@dataclass
class Course:
    key: str
    name: str
    where: str                  # 'map' or the segment id of a place
    area: int
    gates: list                 # [(x, y)], gate 0 = start
    par: tuple                  # (gold, silver, bronze) in seconds
    hint: str                   # where to find the start gate (journal)


COURSES = [
    Course('canterlot', 'CANTERLOT CIRCUIT', 'map', 0,
           [(590, 580), (680, 575), (760, 520), (770, 430), (690, 340), (580, 380), (560, 452)],
           (8.0, 10.0, 14.0), 'MAP: BY SADDLE LAKE'),
    Course('coast', 'COASTAL DASH', 'map', 0,
           [(1150, 480), (1120, 545), (1065, 590), (1180, 640), (1245, 565), (1215, 475), (1160, 402)],
           (8.0, 10.0, 14.0), 'MAP: OFF MANEHATTAN'),
    Course('badlands', 'BADLANDS RALLY', 'map', 0,
           [(720, 830), (800, 760), (900, 770), (930, 860), (860, 920), (760, 900), (700, 880)],
           (7.5, 9.5, 13.0), 'MAP: BY THE BADLANDS'),
    Course('frost', 'FROSTBITE RUN', 'map', 0,
           [(560, 250), (480, 215), (400, 160), (330, 192), (255, 160), (320, 102)],
           (6.0, 7.5, 10.5), 'MAP: WEST OF THE CRYSTAL EMPIRE'),
    Course('hoops', 'RAINBOW HOOPS', 'rainbowfalls', 0, [],
           (11.0, 13.5, 18.5), 'AT RAINBOW FALLS'),
    Course('strip', 'THE STRIP SLALOM', 'laspegasus', 0,
           [(100, 50), (160, 58), (232, 120), (292, 70), (345, 132), (448, 100), (380, 170), (262, 176),
            (150, 165)],
           (15.0, 18.5, 25.5), 'AT LAS PEGASUS'),
]


def _hoop_gates():
    """The Rainbow Falls course flies through the film's cloud hoops, in the film's order."""
    try:
        import scenes.s07_rainbowfalls as sc
        return [(int(round(x)), int(round(y))) for x, y, _a in sc.RINGS]
    except Exception:
        return [(118, 100), (164, 84), (251, 54), (282, 63), (297, 95), (303, 178), (368, 166), (373, 122),
                (378, 82)]


for _c in COURSES:
    if _c.key == 'hoops':
        _c.gates = _hoop_gates()
BY_KEY = {c.key: c for c in COURSES}
MEDALS = ('gold', 'silver', 'bronze')

_MEDAL_COLS = {'gold': ('amber', 'gold', 'butter'), 'silver': ('steel', 'cloud', 'white'),
               'bronze': ('rust', 'copper', 'peach'), 'none': ('sand', 'tan', 'paper')}


def medal_for(course, steps):
    if steps is None:
        return None
    s = steps / 30.0
    for m, p in zip(MEDALS, course.par):
        if s <= p:
            return m
    return 'none'                   # finished, but slower than bronze


def fmt(steps):
    s = steps / 30.0
    return f'{int(s // 60)}:{s % 60:04.1f}'


def draw_medal(c, x, y, kind):
    """A 9x11 medal on a ribbon (top-left x, y)."""
    dk, mid, hi = _MEDAL_COLS.get(kind, _MEDAL_COLS['none'])
    c.rect(x + 2, y, 2, 3, 'red')
    c.rect(x + 5, y, 2, 3, 'sky')
    c.circle(x + 4, y + 6, 4, dk)
    c.circle(x + 4, y + 6, 3, mid)
    c.px(x + 3, y + 4, hi)
    c.px(x + 2, y + 5, hi)
    c.px(x + 4, y + 6, dk)


def stopwatch(c, x, y):
    c.circle(x + 4, y + 5, 4, ui.INK)
    c.circle(x + 4, y + 5, 3, 'white')
    c.rect(x + 3, y, 3, 1, ui.INK)
    c.px(x + 4, y + 1, ui.INK)
    c.vline(x + 4, y + 3, y + 5, 'red')
    c.px(x + 5, y + 5, 'red')
    return 9


FLAG = Sprite.from_ascii('''
    o.....
    owkwk.
    okwkw.
    owkwk.
    o.....
    o.....
    o.....''', {'o': 'brew', 'w': 'white', 'k': 'ink'})


class Races:
    def __init__(self, game):
        self.g = game
        self.best = {}                  # course key -> best time in steps
        self.run = None                 # dict(key, t0, next, last) of the race being flown
        self.result = None              # dict(key, steps, medal, best, t) of the last finish
        self.ended_t = -10 ** 6
        self.gate_t = {}                # gate index -> step it was passed (burst), current run only
        self._armed = {}                # course key -> False while the balloon is still in its start gate

    # -------------------------------------------------------------- save
    def save(self):
        return dict(best=dict(self.best))

    def load(self, d):
        d = d or {}
        self.best = {k: int(v) for k, v in d.get('best', {}).items() if k in BY_KEY}
        self._armed = {c.key: False for c in COURSES}      # re-armed once the balloon is out of the gate

    def medal(self, key):
        return medal_for(BY_KEY[key], self.best.get(key))

    def golds(self):
        return sum(1 for c in COURSES if self.medal(c.key) == 'gold')

    def medals(self):
        return sum(1 for c in COURSES if self.medal(c.key) in MEDALS)

    # -------------------------------------------------------------- the run
    def cancel(self, why=None):
        if self.run is not None:
            self.run = None
            self.ended_t = self.g.t
            if why:
                self.g.toasts.push('RACE OVER', why, stopwatch, sound='tick', gain=0.4)

    def _courses_here(self, where, area=0):
        return [c for c in COURSES if c.where == where and c.area == area]

    def _gate_hit(self, gx, gy, px, py, reach):
        return math.hypot(px - gx, py - gy) < reach

    def _advance(self, px, py, reach, where, area, heading=None):
        """heading: the balloon's velocity on the map: a run starts only when the balloon flies through the
        start gate towards the course (gate 1), so crossing it on the way somewhere else starts nothing."""
        g = self.g
        t = g.t
        if self.run is None:
            if t - self.ended_t < REARM:
                return
            for c in self._courses_here(where, area):
                x, y = c.gates[0]
                inside = self._gate_hit(x, y, px, py, reach)
                if heading is not None and len(c.gates) > 1:
                    gx, gy = c.gates[1]
                    inside = inside and heading[0] * (gx - x) + heading[1] * (gy - y) > 0
                if inside and self._armed.get(c.key, True):
                    self.run = dict(key=c.key, t0=t, next=1, last=t)
                    self.gate_t = {0: t}
                    self.result = None
                    self._armed[c.key] = False
                    g.toasts.push('GO!', c.name, FLAG, sound='whistle', gain=0.5, urgent=True)
                    g.sfx('whoosh', 0.4)
                    return
                if not inside and math.hypot(px - x, py - y) > reach * 3:
                    self._armed[c.key] = True
            return
        c = BY_KEY[self.run['key']]
        if c.where != where or c.area != area:
            self.cancel('YOU LEFT THE COURSE')
            return
        i = self.run['next']
        x, y = c.gates[i]
        if self._gate_hit(x, y, px, py, reach):
            self.gate_t[i] = t
            self.run['next'] = i + 1
            self.run['last'] = t
            if i + 1 >= len(c.gates):
                self._finish(c, t - self.run['t0'])
            else:
                perks.gate(g)                           # the RACER balloon dashes
                name, var = GATE_CUES[min(i - 1, len(GATE_CUES) - 1)]
                g.sfx(name, 0.45, var)
                g.sfx('sparkle', 0.25)
        elif t - self.run['last'] > TIMEOUT:
            # only the start gate flown (an errand hop through it, most likely): no 'too slow'
            self.cancel('TOO SLOW, TRY AGAIN' if self.run['next'] > 1 else None)
        else:
            left = TIMEOUT - (t - self.run['last'])            # hurry: the clock ticks, then faster
            fast = left <= HURRY2
            if left <= HURRY and ((HURRY2 - left) % 8 == 0 if fast else (HURRY - left) % 15 == 0):
                g.sfx('tick', 0.38 if fast else 0.3, (left // 8) % 2)

    def _finish(self, c, steps):
        g = self.g
        old = self.best.get(c.key)
        new_best = old is None or steps < old
        if new_best:
            self.best[c.key] = steps
        m = medal_for(c, steps)
        self.result = dict(key=c.key, steps=steps, medal=m, best=new_best, t=g.t, gap=_gap_line(c, steps, old))
        self.run = None
        self.ended_t = g.t
        g.sfx('jackpot' if m == 'gold' else 'quest', 0.55)
        g.sfx('whistle', 0.4)
        g.sfx('firework', 0.35)
        if new_best and m == 'gold' and (old is None or medal_for(c, old) != 'gold') and self.golds() == 1:
            from .compose import PATTERNS
            name = dict((p[0], p[1]) for p in PATTERNS).get('checkered', 'CHEQUERED')
            g.toasts.push('NEW BALLOON', name + ' PATTERN (J)', ui.icon_balloon, sound='quest', gain=0.5)
        g.save()

    # -------------------------------------------------------------- map
    def map_step(self, m):
        if m.landing is not None:
            self.cancel()
            return
        ex, ey = m.envelope()                           # the envelope's centre, in the gates' ground frame:
        self._advance(ex, ey + MAP_RING_UP, MAP_REACH, 'map', 0, heading=(m.vx, m.vy))   # (issue #5)

    def map_draw(self, c, t, x0, y0, active):
        """active False: the idle start gates (under the fog); True: the course being flown (over it)."""
        for co in self._courses_here('map'):
            racing = self.run is not None and self.run['key'] == co.key
            if racing != active:
                continue
            pts = [(int(round(x)) - x0, int(round(y)) - y0 - MAP_RING_UP) for x, y in co.gates]
            self._draw_course(c, t, co, pts, racing, small=True)

    # -------------------------------------------------------------- places
    def region_step(self, r):
        seg_ = r.SEG
        if self.run is not None and BY_KEY[self.run['key']].where != seg_:
            self.cancel()
        if not self._courses_here(seg_, r.ai) and self.run is None:
            return
        if r.leaving is not None or r.switch is not None:
            if self.run is not None:
                self.cancel('YOU LEFT THE COURSE')
            return
        ex, ey = r.envelope()
        self._advance(ex, ey, REGION_REACH, seg_, r.ai)

    def region_draw(self, r, c, t, ix, iy):
        for co in self._courses_here(r.SEG, r.ai):
            racing = self.run is not None and self.run['key'] == co.key
            pts = [(x - ix, y - iy) for x, y in co.gates]
            self._draw_course(c, t, co, pts, racing, small=False)

    # -------------------------------------------------------------- drawing
    def _draw_course(self, c, t, co, pts, racing, small):
        rx, ry = (6, 9) if small else (6, 12)
        if not racing:
            x, y = pts[0]
            if -20 < x < W + 20 and -30 < y < H + 30:
                _gate(c, x, y, rx, ry, t, 'start')
                FLAG.draw(c, x + rx + 1, y - ry - 4)
                if small and len(pts) > 1:              # marching dots: fly through it this way to start
                    dx, dy = pts[1][0] - x, pts[1][1] - y
                    n = math.hypot(dx, dy) or 1.0
                    for j in range(3):
                        d = rx + 5 + 5 * j
                        lit = (t // 6) % 3 == j
                        c.px(int(round(x + dx / n * d)), int(round(y + dy / n * d)), 'butter' if lit else 'gold')
                best = self.best.get(co.key)
                if best is not None:
                    draw_medal(c, x - 4, y + ry + 3, medal_for(co, best))
            return
        nxt = self.run['next']
        for i, (x, y) in enumerate(pts):
            if not (-30 < x < W + 30 and -40 < y < H + 40):
                continue
            a = t - self.gate_t.get(i, -10 ** 6)
            if i < nxt:
                if a < 30:
                    ui.burst(c, t, self.gate_t[i], x, y, ('white', 'gold', 'hot_pink', 'sky'), n=20, seed=i + 5,
                             power=0.9, life=30, gravity=0.02)
                continue
            if i == nxt:
                _gate(c, x, y, rx, ry, t, 'finish' if i == len(pts) - 1 else 'next')
                bob = int(round(math.sin(t * 0.25) * 1.5))
                ui._LIGHT_ARROWS['down'].outlined(ui.INK).draw(c, x - 3, y - ry - 9 + bob)
            elif i == nxt + 1:
                _gate(c, x, y, rx, ry, t, 'later')

    def hud(self, c, t, focus=None):
        """Timer plate (top-left) while racing, the result plate after a finish, and an edge arrow to
        the next gate. focus: fn(gate x, gate y) -> screen (x, y) for the current view, or None."""
        if self.run is not None:
            co = BY_KEY[self.run['key']]
            s = fmt(t - self.run['t0'])
            g = f"{self.run['next']}/{len(co.gates)}"
            w = 9 + 4 + text_width(s, 'normal') + 6 + text_width(g, 'tiny') + 6
            c.round_rect(3, 3, w, 13, ui.INK, r=1)
            c.rect(4, 4, w - 2, 11, ui.PLATE)
            c.hline(5, w, 4, ui.PLATE_HI)
            stopwatch(c, 5, 5)
            text(c, 17, 6, s, 'white', font='normal')
            text(c, 17 + text_width(s, 'normal') + 6, 7, g, 'butter', font='tiny')
            left = TIMEOUT - (t - self.run['last'])
            if left < 150 and (t // 4) % 2:                    # hurry: the plate rim flashes
                c.round_rect(3, 3, w, 13, 'red', r=1)
                c.rect(4, 4, w - 2, 11, ui.PLATE)
                stopwatch(c, 5, 5)
                text(c, 17, 6, s, 'white', font='normal')
                text(c, 17 + text_width(s, 'normal') + 6, 7, g, 'butter', font='tiny')
            if focus is not None:
                x, y = focus(*co.gates[self.run['next']])
                if not (0 <= x < W and 0 <= y < H):
                    ui.edge_arrow(c, x, y, t, 'sky')
        res = self.result
        if res is not None and 0 <= t - res['t'] < RESULT:
            co = BY_KEY[res['key']]
            a = t - res['t']
            k = ease_out_back(seg(a, 0, 10)) if a < RESULT - 10 else 1 - seg(a, RESULT - 10, RESULT)
            title = {'gold': 'GOLD!', 'silver': 'SILVER!', 'bronze': 'BRONZE!'}.get(res['medal'], 'FINISHED')
            s = fmt(res['steps'])
            sub = ('NEW BEST ' if res['best'] else 'BEST ' + fmt(self.best[co.key]) + '  ') + co.name
            gap = res.get('gap')
            w = max(text_width(title + '  ' + s, 'normal'), text_width(sub, 'tiny'),
                    text_width(gap or '', 'tiny')) + 26
            ph = 31 if gap else 24
            x = (W - w) // 2
            y = int(46 - 60 * (1 - k))
            c.round_rect(x - 1, y - 1, w + 2, ph + 2, '#6e2727', r=2)
            c.round_rect(x, y, w, ph, 'gold', r=2)
            c.rect(x + 1, y + 1, w - 2, ph - 2, ui.PLATE)
            draw_medal(c, x + 5, y + 6, res['medal'])
            text(c, x + 18, y + 4, title, '#fff6c0', font='normal', shadow='#6e2727')
            text(c, x + 18 + text_width(title + '  ', 'normal'), y + 4, s, 'white', font='normal')
            text(c, x + 18, y + 15, sub, 'bubblegum' if res['best'] else 'tan', font='tiny')
            if gap:
                text(c, x + 18, y + 22, gap, 'mint' if gap.endswith('UNDER YOUR BEST') else 'butter', font='tiny')
            if res['medal'] == 'gold' and a < 60:
                fx.sparkles(c, x + 9, y + 12, t, radius=7, n=6, colors=('white', 'gold', 'butter'), seed=3,
                            period=14)


def _gap_line(co, steps, old):
    """The result plate's third line: how much faster the next better medal is ('GOLD 1.8S FASTER');
    for a gold, how it compares with the best time before it (None: nothing to say)."""
    s = steps / 30.0
    m = medal_for(co, steps)
    if m != 'gold':
        better = 'bronze' if m == 'none' else MEDALS[MEDALS.index(m) - 1]
        d = s - co.par[MEDALS.index(better)]
        return f'{better.upper()} {max(0.1, d):.1f}S FASTER'
    if old is None:
        return None
    d = (old - steps) / 30.0
    if d > 0:
        return f'{d:.1f}S UNDER YOUR BEST'
    return f'YOUR BEST IS {max(0.1, -d):.1f}S FASTER' if d < 0 else 'JUST AS FAST AS YOUR BEST'


def _gate(c, x, y, rx, ry, t, state):
    """A race gate ring at (x, y): 'start' chequered, 'next' pulsing gold, 'finish' gold + white,
    'later' faint dotted."""
    if state == 'later':
        for k in range(16):
            a = k / 16 * 2 * math.pi
            if k % 2 == 0:
                c.px(int(round(x + math.cos(a) * rx)), int(round(y + math.sin(a) * ry)), 'white')
        return
    if state == 'start':
        for k in range(24):
            a = k / 24 * 2 * math.pi
            px, py = int(round(x + math.cos(a) * rx)), int(round(y + math.sin(a) * ry))
            c.rect(px, py, 2, 2, 'ink' if (k // 2 + t // 8) % 2 else 'white')
        return
    p = (t // 4) % 2
    outer = 'amber'
    inner = ('gold', 'butter') if state == 'next' else ('white', 'butter')
    fx.glow(c, x, y, ry + 2, 'gold', 0.18, rings=2)
    c.ellipse(x, y, rx + 1, ry + 1, outer, fill=False)
    c.ellipse(x, y, rx, ry, inner[p], fill=False)
    c.ellipse(x, y, rx - 1, ry - 1, inner[1 - p], fill=False)
    if (t // 3) % 6 == 0:
        c.px(x - rx + 1, y - ry // 2, 'white')
