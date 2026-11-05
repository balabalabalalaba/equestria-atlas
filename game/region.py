"""Base class for an explorable diorama: one region of the atlas, built from its film scene.

A region is one or more `Area`s: side-view panoramas the player pans through with the balloon. Fly
past the right (left) edge of an area into the next (previous) one, with a cloud bank sweeping across.
The base class handles everything shared: balloon physics, the eased camera (with frame interpolation
for smooth scrolling), landmark name tags + the star counter, the atlas card, the completion plate,
area changes and taking off back to the world map. A subclass (game/<region>.py) only provides:

    SEG        segment id in engine/timeline.py (gives the map location id, name and music bars)
    CARD       (kind, [fact, fact]) for the atlas card (copy the film scene's atlas_card call)
    build_areas()  -> [Area(...)]: how to draw each panorama from the film scene (see Area)
    init_state() / on_enter() / on_area(ai) / update(t)   optional: persistent state, landing reset,
                   arrival in an area, per-step events
    on_interact(key, t)  the player pressed SPACE at a Hotspot of the current area (start its event)

Interaction: each Area may list Hotspots (things to set off with SPACE: a bell, a train whistle,
fireworks...; a bubble with the hotspot's icon pops up when the balloon is close, and a faint glint
marks them from afar) and Stamps (hidden atlas stamps, collected by flying through them).
    scene_f(t)     optional: the free-running scene frame used for ambient animation

Phase B systems hook in here without touching the region files: the errand spot (game/quests.py) takes
SPACE before the hotspots when the balloon is at it, race gates (game/races.py) are checked and drawn
for the courses of the current area, both draw into the world layer and the HUD, and the balloon
carries the errand's cargo.

Coordinates: an Area's world is the scene's main layer in art px; camera (cx, cy) is the world point at
the screen's top-left; the balloon's (bx, by) is the bottom-centre of its basket in world px.
"""
import json
import math
import os
import types
from dataclasses import dataclass, field

import numpy as np

from engine import W, H, Canvas, seg, clamp, ease_out_back, fx, text_width
from engine import timeline as TL
from engine.loc_worldmap import LOCS
from engine.atlas import callout, atlas_card

from . import ui, sweep, daylight, perks
from .compose import Frame, paste_balloon, KEY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEED, BOOST, CLIMB = 1.5, 3.0, 1.1     # art px / frame (30 fps)
ACC = 0.12
LEAD = 26                               # camera lead: frames of velocity looked ahead
CAM_EASE, LEAD_EASE = 0.10, 0.06
NEAR = 62                               # balloon <-> landmark centre distance that shows its tag
HOLD = 45                               # a tag stays at least this long once shown
TAKEOFF = 14                            # frames of climbing before the iris closes
AREA_FRAMES = 24                        # cloud sweep between areas
BALLOON_H = 28                          # 'l' balloon height (basket bottom -> envelope top)
CELEBRATE = 150                         # steps of fireworks when a place is completed
SHELLS = [(70, 40, ('hot_pink', 'butter', 'white')), (240, 34, ('sky', 'mint', 'white')),
          (160, 26, ('gold', 'orange', 'butter')), (100, 52, ('violet', 'lilac', 'white')),
          (220, 54, ('red', 'gold', 'white')), (150, 44, ('leaf', 'butter', 'sky'))]
ENVELOPE_Y = {'l': 18, 'm': 13, 's': 8, 'xs': 4}   # basket bottom -> envelope centre, by size
STAMP_REACH = 11                        # envelope centre <-> stamp centre distance that collects it
SPOT_BURST, SPOT_GLOW = 40, 22          # steps of the spark burst / light pulse when a hotspot is set off
SPOT_GLOW_R, SPOT_GLOW_K = (22, 26), 0.85   # the pulse's radius (start, growth) and strength (>= ~3% of the screen)
SPOT_HIDE = 45                          # the bubble steps aside this long (at most the cooldown) for the show
PLUS_ONE = 30                           # steps the '+1 spark' floats up from a hotspot's first use
JOB_TOAST_TTL = 150                     # a return trip's 'STRAIGHT TO ...' toast not shown by then is dropped
# the light pulse and sparks of a hotspot set off, by its icon: (glow colour, burst colours)
SPOT_FX = {'bell': ('gold', ('gold', 'butter', 'white')), 'whistle': ('cloud', ('white', 'cloud', 'butter')),
           'rain': ('sky', ('sky', 'baby_blue', 'white')), 'spark': ('butter', ('gold', 'white', 'hot_pink')),
           'firework': ('hot_pink', ('hot_pink', 'butter', 'sky')), 'steam': ('cloud', ('white', 'cloud', 'light_sky')),
           'lantern': ('orange', ('orange', 'butter', 'gold')), 'note': ('lilac', ('pink', 'lilac', 'white')),
           'bolt': ('butter', ('butter', 'white', 'sky')), 'wind': ('mint', ('white', 'cloud', 'mint')),
           'splash': ('sky', ('sky', 'aqua', 'white')), 'gem': ('crystal', ('crystal_lt', 'crystal_pk', 'white')),
           'flag': ('salmon', ('red', 'butter', 'white')), 'gear': ('gold', ('gold', 'cream', 'white')),
           'horn': ('gold', ('gold', 'butter', 'cream')), 'apple': ('salmon', ('red', 'leaf', 'butter')),
           'heart': ('pink', ('hot_pink', 'pink', 'white')), 'moon': ('periwinkle', ('cream', 'periwinkle', 'white'))}
SPOT_FX_DEFAULT = ('butter', ('gold', 'white', 'hot_pink'))


def scene_render(module):
    """A film scene's own `render(f)`, without the director-pass wrappers the scene module applies at
    its end (engine.shotfx.smooth_cuts, engine.regionfx.long_flight): the wrapped function keeps the
    raw one in its closure; this finds the function named `render` defined in the scene's own file."""
    stack, seen = [module.render], set()
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        code = getattr(fn, '__code__', None)
        if code is None:
            continue
        if fn.__name__ == 'render' and os.path.abspath(code.co_filename) == os.path.abspath(module.__file__):
            return fn
        stack.extend(c.cell_contents for c in (fn.__closure__ or ()) if callable(c.cell_contents))
    raise LookupError(f'raw render() not found in {module.__name__}')


def rebind(fn, **names):
    """A copy of function `fn` whose module-level names in `names` are replaced (fn is untouched).
    This is how a film scene's draw function is driven by the game without editing the scene file."""
    g = dict(fn.__globals__)
    g.update(names)
    return types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)


@dataclass
class Landmark:
    """A named place in an area. x (and y, for areas that pan vertically) = the building's centre in
    world px, used for proximity; tag = (anchor x, anchor y, side, leader) for engine.atlas.callout,
    in world px (copy them from the scene's callout calls, converting screen -> world with the camera).
    p: parallax of the layer the tag points at (its anchor moves p x the camera; 1 = the main layer);
    py: its vertical parallax, if different (None = p)."""
    name: str
    x: int
    tag: tuple
    y: int = None
    p: float = 1.0
    py: float = None


@dataclass
class Hotspot:
    """Something the player sets off with SPACE when the balloon is close: a bell to ring, a train to
    call, fireworks to launch... x, y: world px of the thing (the bubble's tail points at it); icon: a
    name in game.ui.ICONS or a Sprite; key: handed to Region.on_interact(key, t); r: reach in px from
    the balloon's envelope centre; cooldown: steps before it can be used again; p: parallax of the
    layer it sits on (like Landmark.p; py = vertical parallax, None = p)."""
    key: str
    x: int
    y: int
    icon: object = 'spark'
    r: int = 44
    cooldown: int = 75
    p: float = 1.0
    py: float = None


@dataclass
class Stamp:
    """A hidden atlas stamp (about 3 per region, tucked away: up high, under a bridge, behind
    something): the player collects it by flying the balloon's envelope through it. x, y: world px of
    its centre on the main layer, somewhere the envelope can reach."""
    key: str
    x: int
    y: int


@dataclass
class Area:
    """One panorama. render(f, cx, cy) -> Canvas: the world at integer camera (cx, cy) and scene frame f,
    WITHOUT the film's own balloon (the base draws the player's). cam_x / cam_y: camera ranges (the
    largest camera that still has art on every parallax layer). y_range: basket-bottom range in world
    px. start: balloon spawn (world px) when arriving from the map or from the previous area; arriving
    from the next area mirrors it to the right end. balloon: sprite size that suits the area's scale."""
    render: callable
    cam_x: tuple = (0, 0)
    cam_y: tuple = (0, 0)
    y_range: tuple = (36, 128)
    start: tuple = (104, 78)
    landmarks: list = field(default_factory=list)
    balloon: str = 'l'              # balloon sprite size here: 'l' (28 px tall), 'm' (20), 's' (13)
    balloon_tint: tuple = None      # (colour, 0..1): light the balloon like the scene (dusk, night, mist)
    climb: float = 1.0              # vertical speed factor (tall areas: climb faster)
    hotspots: list = field(default_factory=list)
    stamps: list = field(default_factory=list)
    sky: callable = None            # sky(f, cx, cy) -> Canvas: the area's sky alone, as render draws it
                                    # first (game/daylight.py repaints it for the world clock's hour)
    horizon: int = None             # world row where the open sky meets the land / sea, if the sky shows
                                    # lower than the clock's sky strip reaches (cam_y span + 130 rows)

    @property
    def x_range(self):
        return self.cam_x[0] + 14, self.cam_x[1] + W - 14


def _music_spans():
    try:
        with open(os.path.join(ROOT, 'audio', 'out', 'sections.json')) as fh:
            return {s['id']: (s['start_s'], s['end_s']) for s in json.load(fh)}
    except OSError:
        return {}


_SECTIONS = _music_spans()
FLIGHT_S = TL.FLIGHT / 30.0             # the 2-bar map flight at the start of every region's music


class Region:
    SEG = None
    CARD = ('', [])

    def __init__(self, game):
        s = TL.BY_ID[self.SEG]
        self.g = game
        self.ID = s['loc']
        self.NAME = LOCS[self.ID]['name'].upper()
        self.HOUR = TL.hour_at(self.SEG, TL.seg_frames(self.SEG) // 2)
        a, b = _SECTIONS.get(self.SEG, (0.0, 0.0))
        self.MUSIC = (a + FLIGHT_S, b)
        self.found = set()              # landmark names discovered
        self.found_t = None
        self.done_t = None
        self.visits = 0
        self.used = set()               # hotspot keys set off at least once
        self.lasting = set()            # big set-pieces done for good (a subclass's own keys: 'lit', 'bloom')
        self.spot_t = {}                # hotspot key -> step it was last set off
        self.stamps_got = set()         # stamp keys collected
        self.stamp_t = {}               # stamp key -> step it was collected (burst)
        self.stamp_flash = None
        self.near_spot = None
        self.spot_fx = {}               # hotspot key -> step SPACE set it off (its burst and light pulse)
        self.spot_first = {}            # hotspot key -> step of its first use ever (the '+1 spark' float)
        self.spot_flash = None          # step the spark counter last flashed (a first use)
        self._memo = None               # (key, world) of the frame's world: the 60 Hz in-between frames reuse it
        self._job_toast = None          # the queued 'STRAIGHT TO ...' toast of a return trip (dropped if stale)
        self.areas = self.build_areas()
        self.total = sum(len(a.landmarks) for a in self.areas)
        self.stamp_total = sum(len(a.stamps) for a in self.areas)
        self.spot_total = sum(len(a.hotspots) for a in self.areas)
        self.t_enter = 0
        self._sky_memo = None
        self.init_state()
        self._place(0)

    # -------------------------------------------------------------- progress (save game, journal)
    def save_state(self):
        return dict(found=sorted(self.found), stamps=sorted(self.stamps_got), used=sorted(self.used),
                    visits=self.visits, lasting=sorted(self.lasting))

    def load_state(self, d):
        names = {L.name for A in self.areas for L in A.landmarks}
        keys = {st.key for A in self.areas for st in A.stamps}
        spots = {h.key for A in self.areas for h in A.hotspots}
        self.found = set(d.get('found', ())) & names
        self.stamps_got = set(d.get('stamps', ())) & keys
        self.used = set(d.get('used', ())) & spots
        self.visits = int(d.get('visits', 0))
        ls = d.get('lasting', ())
        self.lasting = {k for k in ls if isinstance(k, str)} if isinstance(ls, (list, tuple)) else set()
        if self.total and len(self.found) == self.total:
            self.done_t = -10 ** 6                      # completed earlier: no plate again

    def meta(self):
        """What the atlas journal shows about this place (also stored in the save)."""
        return dict(name=self.NAME, card=[self.CARD[0], list(self.CARD[1])],
                    landmarks=[L.name for A in self.areas for L in A.landmarks],
                    stamps=self.stamp_total,
                    hotspots=[[h.key, h.icon if isinstance(h.icon, str) else 'spark']
                              for A in self.areas for h in A.hotspots])

    @property
    def complete(self):
        return self.total > 0 and len(self.found) == self.total

    # -------------------------------------------------------------- hooks
    def build_areas(self):
        raise NotImplementedError

    def init_state(self):
        pass

    def on_enter(self):
        pass

    def update(self, t):
        pass

    def on_interact(self, key, t):
        """The player set off hotspot `key` at step t: start its event (sounds via self.g.sfx)."""
        pass

    def on_area(self, ai):
        """Called after arriving in area ai (from the map: ai 0; or flying in from a neighbour)."""
        pass

    def front(self, f, ix, iy):
        """Optional: a KEY Canvas (or None) of the scene's art that stands IN FRONT of the balloon (the
        near half of a hoop it flies through...), at integer camera (ix, iy). The base class keeps only
        the part around the balloon, so the rest of the scene's layer order is untouched."""
        return None

    def scene_f(self, t):
        """Scene frame for ambient animation: free running, past the film's scripted beats."""
        return 360 + t

    # -------------------------------------------------------------- lifecycle
    def enter(self):
        """Arrive from the world map: in the first area (left side), or, on a return trip, straight in
        the area where the errand that brought the player here is (game/quests.py targets: goods to
        pick up or a trouble to mend here)."""
        self._prune_job_toast()                         # (one left over from the last visit goes)
        ai, job = 0, None
        if self.visits >= 1:
            for tg in self.g.quests.targets():          # (place id, area, x, y, 'drop' | 'pick', item)
                if tg[0] == self.ID and isinstance(tg[1], int) and 0 <= tg[1] < len(self.areas):
                    ai, job = tg[1], tg
                    break
        self._place(ai)
        self.t_enter = self.g.t
        self.g.weather.drift = (0.0, 0.0)
        self.visits += 1
        self.on_enter()
        self.on_area(ai)
        if job is not None and ai > 0:
            from .items import ITEM_NAMES, icon as item_icon
            item = job[5]
            sub = ('STRAIGHT TO THE ' + ITEM_NAMES.get(item, 'GOODS') if job[4] == 'pick' else
                   'STRAIGHT TO WHERE IT IS NEEDED')
            q = self.g.toasts.queue
            self.g.toasts.push(self.NAME, sub, item_icon(item))
            self._job_toast = q[-1] if q else None      # (step drops it if it would show too late)
        return self

    def _prune_job_toast(self):
        """A return trip's landing toast only makes sense on arrival: if the queue (paced tips and
        toasts) has kept it waiting past JOB_TOAST_TTL steps, or the balloon is taking off, drop it so
        it never greets the player at the next place."""
        jt = self._job_toast
        if jt is None:
            return
        q = self.g.toasts.queue
        if not any(d is jt for d in q):
            self._job_toast = None                      # on screen (or cleared): nothing to do
        elif self.leaving is not None or self.g.t - self.t_enter > JOB_TOAST_TTL:
            q[:] = [d for d in q if d is not jt]
            self._job_toast = None

    def _place(self, area, from_right=False):
        self.ai = area
        A = self.areas[area]
        x, y = A.start
        if from_right:
            x = A.x_range[1] - (x - A.x_range[0])
        self.bx, self.by = float(x), float(y)
        self.vx = self.vy = 0.0
        self.boost = self.climb = False
        self.look_x = self.look_y = 0.0
        self.cx, self.cy = self._cam_target()
        self.prev = self.snapshot()
        self.leaving = None
        self.switch = None              # (frame, old area's last frame, sweep direction)
        self.back_pending = False       # ESC pressed during a cloud sweep (it takes off when the sweep ends)
        self.tags = {}
        self.tag_side = {i: self._tag_side(L) for i, L in enumerate(A.landmarks) if L.tag}   # (see _tag_side)
        self._shown = None              # (t, world canvas, ix, iy, bx, by) of the last frame drawn
        self._sweep_new = None          # (t, the new area's world) during a sweep: drawn once per step

    def warm(self):
        """Build every area's cached art now (at landing), so no area stalls on its first frame."""
        for A in self.areas:
            for cx in sorted(set(A.cam_x)):
                for cy in sorted(set(A.cam_y)):
                    A.render(self.scene_f(self.g.t), int(cx), int(cy))
        sweep.warm(self.sweep_hour())

    def sweep_hour(self):
        """The hour the cloud sweep between areas is coloured for: the world clock's (a quarter-hour step, so
        the palettes stay few), or the painted hour indoors."""
        if (self.ID, self.ai) in daylight.INTERIORS:
            return self.HOUR
        return round(self.g.hour * 4) / 4.0

    def snapshot(self):
        return (self.cx, self.cy, self.bx, self.by)

    def view(self, alpha):
        p, c = self.prev, self.snapshot()
        return tuple(a + (b - a) * alpha for a, b in zip(p, c))

    def iris_center(self):
        return int(round(self.bx - self.cx)), int(round(self.by - self.cy)) - 16

    @property
    def area(self):
        return self.areas[self.ai]

    # -------------------------------------------------------------- update
    def step(self, inp):
        g = self.g
        self._memo = None                               # the world changes from here on
        self._prune_job_toast()
        self.prev = self.snapshot()
        if self.switch is not None:
            if inp.back:                                # ESC during the cloud sweep: take off after it
                self.back_pending = True
            if g.t - self.switch[0] >= AREA_FRAMES:
                self.switch = None
            return
        if self.leaving is not None:
            self.vy = -2.4
            # the climb stops a little above the area's top edge (steps skipped by the journal or the
            # pause menu must not leave it climbing forever)
            self.by = max(self.area.cam_y[0] - 60, self.by + self.vy)
            if g.t >= self.leaving + TAKEOFF and g.trans is None:    # >=: the exact step may be skipped
                sx, sy = self.iris_center()
                g.iris_to(sx, max(8, sy), lambda: g.map.take_off_from(self.ID))
            self.update(g.t)
            return
        if (inp.back or self.back_pending) and g.t - self.t_enter > 20:
            self.back_pending = False
            self.leaving = g.t
            g.races.cancel()
            g.sfx('puff', 0.5)
            return
        self._fly(inp)
        g.weather.region_step(self)                     # gusts push the balloon; rain / snow settle
        self._edges(inp)
        self._landmarks()
        # Phase B hooks (game/quests.py, game/races.py): an errand spot takes SPACE before the hotspots
        at_errand = g.quests.region_step(self, inp)
        g.races.region_step(self)
        self._interactions(inp, block=at_errand)
        g.quests.region_update(self, g.t)
        self.update(g.t)

    def _fly(self, inp):
        A = self.area
        ax = (1 if inp.right else 0) - (1 if inp.left else 0)
        ay = (1 if inp.down else 0) - (1 if inp.up else 0)
        self.boost = inp.boost and ax != 0
        sm = self.g.speed_mult * perks.speed(self.g, self.boost)   # testing (--cheats) x the balloon's perk
        vmax = (BOOST if self.boost else SPEED) * sm
        if ax:
            self.vx += ax * ACC * (1.6 if self.boost else 1) * sm
        else:
            self.vx *= perks.glide(self.g, 0.9)
        if abs(self.vx) > vmax:          # ease down after a boost instead of snapping
            self.vx = math.copysign(max(vmax, abs(self.vx) * 0.94), self.vx)
        self.vy = self.vy + ay * 0.12 * A.climb * sm if ay else self.vy * 0.85
        self.vy = clamp(self.vy, -CLIMB * A.climb * sm, CLIMB * A.climb * sm)
        self.climb = ay < 0
        x0, x1 = A.x_range
        self.bx = clamp(self.bx + self.vx + self.g.weather.drift[0], x0, x1)   # + a gust's push
        if self.bx in (x0, x1):
            self.vx = 0.0
        self.by = clamp(self.by + self.vy, A.y_range[0], A.y_range[1])
        self.look_x += ((self.vx + self.g.weather.drift[0]) * LEAD - self.look_x) * LEAD_EASE
        self.look_y += (self.vy * LEAD * 0.5 - self.look_y) * LEAD_EASE
        tx, ty = self._cam_target()
        self.cx += (tx - self.cx) * CAM_EASE
        self.cy += (ty - self.cy) * CAM_EASE
        self.cx = clamp(self.cx, *A.cam_x)
        self.cy = clamp(self.cy, *A.cam_y)

    def _cam_target(self):
        A = self.area
        tx = clamp(self.bx - W / 2 + self.look_x, *A.cam_x)
        ty = clamp(self.by - BALLOON_H / 2 - H / 2 + self.look_y, *A.cam_y)
        return tx, ty

    def _edges(self, inp):
        x0, x1 = self.area.x_range
        if inp.right and self.bx >= x1 and self.ai + 1 < len(self.areas):
            self._switch(self.ai + 1, from_right=False, dir=-1)
        elif inp.left and self.bx <= x0 and self.ai > 0:
            self._switch(self.ai - 1, from_right=True, dir=1)

    def _switch(self, ai, from_right, dir):
        old = self._old_picture()
        y0, y1 = self.area.y_range                    # keep the balloon's relative height
        k = (self.by - y0) / max(1, y1 - y0)
        self._place(ai, from_right)
        y0, y1 = self.area.y_range
        self.by = y0 + k * (y1 - y0)
        self.cx, self.cy = self._cam_target()        # frame the new spot at once (no camera swoop)
        self.prev = self.snapshot()
        self.switch = (self.g.t, old, dir)
        self.g.sfx('wind', 0.3)
        self.on_area(ai)

    def _old_picture(self):
        """The picture the sweep starts from: the frame just shown (no second render of the area in
        the switch step, which used to stall that frame), with the balloon pasted in."""
        sh = self._shown
        if sh is None or self.g.t - sh[0] > 1:
            return self._world(self.g.t, with_balloon=True)
        t, c, ix, iy, bx, by = sh
        c = c.copy()
        A = self.area
        paste_balloon(c, int(round(bx)) - ix, int(round(by)) - iy, t, A.balloon, flame=True,
                      tint=self.light_tint(), pattern=self.g.pattern, cargo=self.g.quests.cargo(),
                      wet=self.g.weather.wet, snow=self.g.weather.snow)
        fl = self._front(t, ix, iy, bx, by)
        if fl is not None:
            m = np.any(fl.a != np.array(KEY, np.uint8), axis=2)
            c.a[m] = fl.a[m]
        return c

    def _landmarks(self):
        g = self.g
        for i, L in enumerate(self.area.landmarks):
            near = abs(self.bx - L.x) < NEAR and (L.y is None or abs(self.by - L.y) < 90)
            tag = self.tags.get(i)
            if near:
                if tag is None or tag[1] is not None:
                    self.tags[i] = [g.t, None]
                    if L.name not in self.found:
                        self.found.add(L.name)
                        self.found_t = g.t
                        if len(self.found) == self.total:
                            self.done_t = g.t + 20
                        else:
                            g.sfx('sparkle', 0.45)
            elif tag is not None and tag[1] is None and g.t - tag[0] >= HOLD:
                tag[1] = g.t
        if self.done_t == g.t:
            g.sfx('quest', 0.6)
        if self.done_t is not None and 0 <= g.t - self.done_t - 6 < len(SHELLS) * 16 and (g.t - self.done_t - 6) % 16 == 0:
            g.sfx('firework', 0.4)

    def _tag_side(self, L):
        """The side a landmark's tag shows on in the current area (fixed per area, so it never jumps): its
        own, or for a 'left' / 'right' tag whose plate can come under the counters (top right) or the clock
        (top left) at the cameras it shows at (the balloon within NEAR of it, looking ahead), the other
        side if that one stays clear at all of them."""
        ax, ay, side, length = L.tag
        if side not in ('left', 'right'):
            return side
        A = self.area
        rows = 16 + 13 * bool(getattr(self, 'stamp_total', 0)) + 13 * bool(getattr(self, 'spot_total', 0)) + 13
        hud = ((W - 50, 0, 50, rows), (0, 0, 58, 18))
        pw, ph = text_width(L.name, 'tiny') + 6, 9
        cams = [(int(clamp(L.x - W / 2 + d, *A.cam_x)), int(clamp(cy, *A.cam_y)))
                for d in range(-NEAR - 50, NEAR + 51, 16) for cy in (A.cam_y[0], A.cam_y[1])]

        def hits(sd):
            for ix, iy in cams:
                sx, sy = ax - int(round(ix * L.p)), ay - int(round(iy * (L.p if L.py is None else L.py)))
                if not (0 <= sx < W and 0 <= sy < H):
                    continue
                px = sx - 2 - length - pw + 1 if sd == 'left' else sx + 2 + length
                px, py = max(1, min(W - pw - 1, px)), max(1, min(H - ph - 1, sy - ph // 2))
                if any(px < hx + hw and hx < px + pw and py < hy + hh and hy < py + ph for hx, hy, hw, hh in hud):
                    return True
            return False
        other = 'left' if side == 'right' else 'right'
        return other if hits(side) and not hits(other) else side

    def envelope(self):
        """World px of the balloon's envelope centre."""
        return self.bx, self.by - ENVELOPE_Y.get(self.area.balloon, 18)

    def spot_ready(self, h, t):
        return t - self.spot_t.get(h.key, -10 ** 6) >= h.cooldown

    def _interactions(self, inp, block=False):
        g, A = self.g, self.area
        ex, ey = self.envelope()
        if block:                                   # at an errand spot: SPACE and the bubble are its
            self.near_spot = None
            inp = type(inp)(**dict(vars(inp), action=False))
        # nearest hotspot within reach (compared on screen: hotspots may sit on parallax layers)
        best, bd = None, 1e9
        for i, h in enumerate(A.hotspots if not block else ()):
            hx = h.x - self.cx * (h.p - 1)
            hy = h.y - self.cy * ((h.p if h.py is None else h.py) - 1)
            d = math.hypot(ex - hx, ey - hy)
            if d < h.r * perks.spark_reach(self.g) and d < bd:
                best, bd = i, d
        self.near_spot = best
        if inp.action and best is not None:
            h = A.hotspots[best]
            if self.spot_ready(h, g.t):
                self.spot_t[h.key] = g.t
                self.spot_fx[h.key] = g.t               # every SPACE: a spark burst and a light pulse
                first = h.key not in self.used
                self.used.add(h.key)
                self.on_interact(h.key, g.t)
                if first:                               # a first use: '+1 spark', the counter flashes
                    self.spot_first[h.key] = g.t
                    self.spot_flash = g.t
                    if len(self.used) == self.spot_total:
                        g.sfx('jackpot', 0.5)           # ... and every one here has been set off
                    else:
                        g.sfx('blip', 0.4)
            else:
                g.sfx('tick', 0.3)
        for st in A.stamps:
            if st.key not in self.stamps_got and math.hypot(ex - st.x, ey - st.y) < STAMP_REACH * perks.stamp_reach(self.g):
                before = set(g.unlocked_patterns())
                self.stamps_got.add(st.key)
                self.stamp_t[st.key] = g.t
                self.stamp_flash = g.t
                full = len(self.stamps_got) == self.stamp_total
                g.sfx('jackpot' if full else 'coin', 0.5 if full else 0.45)
                from .compose import PATTERNS
                names = {p[0]: p[1] for p in PATTERNS}
                for key in g.unlocked_patterns():       # a stamp count that earns a balloon pattern
                    if key not in before:
                        g.toasts.push('NEW BALLOON', names.get(key, key.upper()) + ' PATTERN (J)',
                                      ui.icon_balloon, sound='powerup')

    def near_gain(self, x):
        """0.08..1 volume for a sound at world x, by its distance from the screen centre."""
        return clamp(1 - abs(x - (self.cx + W / 2)) / 520, 0.08, 1.0)

    # -------------------------------------------------------------- draw
    def light_tint(self):
        """How the balloon and sprites are lit here now: the area's own tint (as the place was painted),
        moving towards the world clock's as far as the clock's light differs from the painted one (a
        night-painted place at another night hour keeps its own, a day-painted one at night takes the
        clock's), so the balloon never turns suddenly dark or light."""
        if abs(self.g.hour - self.HOUR) < 0.3 or (self.ID, self.ai) in daylight.INTERIORS:
            return self.area.balloon_tint
        w = abs(daylight.night(self.g.hour) - daylight.night(self.HOUR)) * 2.5
        # ...and by how different the two hours' light is (a golden-painted place at noon: same 'night',
        # but the balloon must lose its evening glow)
        w = max(w, daylight.tint_gap(daylight.light_tint(self.HOUR), daylight.light_tint(self.g.hour)) * 6)
        if daylight.light_tint(self.g.hour) is None and self.area.balloon_tint is None:
            return None
        return daylight.mix_tint(self.area.balloon_tint, daylight.light_tint(self.g.hour), w)

    MEMO = True                         # reuse the world for the 60 Hz in-between frames (tools/b_memo_check.py)
    OWN_MOON = False                    # the place paints a moon of its own: the clock's sky draws none

    def auto_glow(self):
        """Should the relight keep the area's small warm / magic-hued spots glowing at dusk and night
        (daylight._lights)? Right where the painted spots are real lights; an area whose painted warm
        trims, fruit or signs would glow like lamps returns False and keeps its real lights itself
        (GlowKeep, game/nightwin.py)."""
        return True

    def after_light(self, c, t):
        """Called on the area's picture right after the clock's light (and its sky), before the stamps and
        the hotspot effects go on top: a place's own lights get their colour back here (GlowKeep), and
        light that only exists at night can be drawn onto the real night sky."""

    def _world(self, t, cam=None, with_balloon=False):
        A = self.area
        cx, cy = (self.cx, self.cy) if cam is None else cam
        ix, iy = int(math.floor(cx)), int(math.floor(cy))
        # the frame's own world (render) is the same for every 60 Hz frame of a step at the same integer
        # camera: drawn once, handed out as copies (step() clears it; the sweep's pictures are not kept)
        memo = self.MEMO and cam is not None and not with_balloon
        mkey = (self.ai, t, ix, iy, self.g.hour, self.near_spot, self.leaving)
        if memo and self._memo is not None and self._memo[0] == mkey:
            return self._memo[1].copy()
        f = self.scene_f(t)
        c = A.render(f, ix, iy)
        hour = self.g.hour
        if (self.ID, self.ai) in daylight.INTERIORS:     # indoors: lit by its own lamps, as painted
            hour = self.HOUR
        w = daylight.sky_weight(self.HOUR, hour) if A.sky is not None else 0.0
        mask = None
        if w > 0.01:                                     # open sky: exactly the pixels still equal to it
            key = (self.ai, t, ix, iy)                   # once per step (the 60 fps frames share it)
            if self._sky_memo is None or self._sky_memo[0] != key:
                self._sky_memo = (key, daylight.sky_pixels(c.a, A.sky(f, ix, iy).a))
            mask = self._sky_memo[1]
        daylight.relight(c.a, self.HOUR, hour, t, snow=self.ID in daylight.SNOWY,    # the clock's light
                         glow=self.auto_glow())
        if mask is not None and mask.any():
            y0, y1 = A.cam_y
            span = y1 - y0 + 130
            if A.horizon is not None:                    # a low sky: the gradient reaches its horizon
                span = max(span, A.horizon - (y0 - 10) + 4)
            daylight.paint_sky(c.a, mask, hour, w, ix, iy, y0 - 10, span, t, moon=not self.OWN_MOON)
        self.after_light(c, t)                           # lights the relight would dim; night lights (GlowKeep)
        for k, st in enumerate(A.stamps):
            sx, sy = st.x - ix, st.y - iy
            if not (-12 < sx < W + 12 and -12 < sy < H + 12):
                continue
            if st.key not in self.stamps_got:
                ui.stamp(c, sx, sy, t, seed=k * 3)
            elif 0 <= t - self.stamp_t.get(st.key, -999) < 40:
                fx.confetti(c, t, self.stamp_t[st.key], sx, sy, n=16, seed=k)
                fx.sparkles(c, sx, sy, t, radius=10, n=6, colors=('white', 'gold', 'hot_pink'), seed=k, period=12)
        for i, h in enumerate(A.hotspots):
            sx, sy = self._spot_xy(h, ix, iy)
            if not (-30 < sx < W + 30 and -30 < sy < H + 30):
                continue
            a = t - self.spot_fx.get(h.key, -10 ** 6)
            if 0 <= a < SPOT_BURST:                     # just set off: a light pulse and a spark burst
                glow, cols = SPOT_FX.get(h.icon, SPOT_FX_DEFAULT) if isinstance(h.icon, str) else SPOT_FX_DEFAULT
                if a < SPOT_GLOW:                       # a bloom of light that swells and fades
                    k = a / SPOT_GLOW
                    fx.glow(c, sx, sy, SPOT_GLOW_R[0] + int(SPOT_GLOW_R[1] * math.sqrt(k)), glow,
                            SPOT_GLOW_K * (1 - k) ** 1.2, rings=4)
                if 1 <= a < 13 and a % 4 != 3:          # a ring of light runs out from it
                    c.circle(sx, sy, 4 + a * 3, 'white' if a < 7 else glow, fill=False)
                ui.burst(c, t, self.spot_fx[h.key], sx, sy, cols, n=24, seed=17 + i * 5, power=1.0,
                         life=SPOT_BURST, gravity=0.03, size=2)
            if i == self.near_spot and self.leaving is None:
                continue                                # its SPACE bubble is on the front layer (render)
            if h.key not in self.used:                  # not tried yet: a bigger golden double twinkle
                p = (t + i * 29) % 54
                if p < 6 or 9 <= p < 15:
                    fx.sparkle(c, sx, sy, p % 9, 'butter', period=6, size=3)
                    c.px(sx, sy, 'white')
            elif (t + i * 37) % 70 < 8:                 # tried: a faint white glint marks it from afar
                fx.sparkle(c, sx, sy, (t + i * 37) % 70, 'white', period=70, size=2)
        self.g.races.region_draw(self, c, t, ix, iy)
        self.g.quests.region_draw(self, c, t, ix, iy)
        before = None if daylight.is_day(hour) else c.a.copy()
        self.g.weather.region_draw(self, c, t, ix, iy)
        if before is not None:                           # wind streaks and rain lit like the place (no white
            daylight.dim_new(c.a, before, hour, 0.85)    # dashes racing over a night scene)
        for i, (t0, t1) in self.tags.items():            # the landmark names last: nothing hides them
            L = A.landmarks[i]
            ax, ay, side, length = L.tag
            side = self.tag_side.get(i, side)
            sx, sy = ax - int(round(ix * L.p)), ay - int(round(iy * (L.p if L.py is None else L.py)))
            if -40 < sx < W + 40 and -40 < sy < H + 40:
                callout(c, sx, sy, L.name, t, t0, t1, side=side, length=length)
        if with_balloon:
            paste_balloon(c, int(round(self.bx)) - ix, int(round(self.by)) - iy, t, A.balloon, flame=True,
                          tint=self.light_tint(), pattern=self.g.pattern, cargo=self.g.quests.cargo(),
                          wet=self.g.weather.wet, snow=self.g.weather.snow)
            fl = self._front(t, ix, iy, self.bx, self.by)
            if fl is not None:
                m = np.any(fl.a != np.array(KEY, np.uint8), axis=2)
                c.a[m] = fl.a[m]
        self._celebrate(c, t)
        if memo:
            self._memo = (mkey, c.copy())
        return c

    @staticmethod
    def _spot_xy(h, ix, iy):
        """Screen (= world canvas) point of hotspot h at integer camera (ix, iy)."""
        return h.x - int(round(ix * h.p)), h.y - int(round(iy * (h.p if h.py is None else h.py)))

    def _spot_overlay(self, c, t, ix, iy):
        """Front-layer UI of the hotspots (over the balloon, world-anchored): the near one's SPACE bubble
        (it steps aside while the effect it set off plays, then waits dimmed until it is ready again)
        and the '+1 spark' rising from a first use. Draws on c (a KEY canvas); returns whether it drew."""
        A, drew = self.area, False
        for i, h in enumerate(A.hotspots):
            a = t - self.spot_first.get(h.key, -10 ** 6)
            if 0 <= a < PLUS_ONE and not (a >= PLUS_ONE - 8 and (a // 2) % 2):    # blinks out at the end
                sx, sy = self._spot_xy(h, ix, iy)
                spr = ui.ICONS['spark']
                w = spr.w + 2 + ui.text_width('+1', 'normal')
                y0, x = sy - 12, sx - w // 2
                if y0 - 14 < 3:                         # a hotspot near the top (Ponyville's star): the float
                    y0 = 17                             # starts lower, beside the hotspot, so it stays in view
                    x = sx + 9 if sx + 9 + w < W - 2 else sx - 9 - w
                y = y0 - int(round(14 * (1 - (1 - a / PLUS_ONE) ** 2)))
                x = max(2, min(W - w - 2, x))
                ui.text(c, x, y + 1, '+1', 'butter', font='normal', shadow='ink')
                spr.draw(c, x + ui.text_width('+1', 'normal') + 2, y)
                drew = True
        i = self.near_spot
        if i is None or self.leaving is not None or i >= len(A.hotspots):
            return drew
        h = A.hotspots[i]
        if 0 <= t - self.spot_t.get(h.key, -10 ** 6) < min(h.cooldown, SPOT_HIDE):
            return drew
        sx, sy = self._spot_xy(h, ix, iy)
        if -30 < sx < W + 30 and -30 < sy < H + 30:
            ui.prompt_bubble(c, sx, sy, h.icon, t, ready=self.spot_ready(h, t))
            drew = True
        return drew

    def _celebrate(self, c, t):
        """Fireworks over the place when its last landmark is found (screen-anchored sky bursts)."""
        if self.done_t is None or not 0 <= t - self.done_t < CELEBRATE:
            return
        for k in range(len(SHELLS)):
            t0 = self.done_t + 6 + k * 16
            x, y, cols = SHELLS[k]
            a = t - t0
            if -12 <= a < 0:                                  # the shell climbs, trailing sparks
                yy = y + int(-a * 4)
                c.px(x, yy, 'butter')
                c.px(x, yy + 2, 'gold')
            if 0 <= a < 6:                                    # the bloom flash
                fx.glow(c, x, y, 14 - a, cols[0], 0.6 * (1 - a / 6), rings=3)
            ui.burst(c, t, t0, x, y, cols, n=64, seed=31 + k, power=1.6, life=58, gravity=0.02, size=2)

    def _front(self, t, ix, iy, bx, by):
        """The region's front art, kept only in a box around the balloon (None if there is none)."""
        fl = self.front(self.scene_f(t), ix, iy)
        if fl is None:
            return None
        sx, sy = int(round(bx)) - ix, int(round(by)) - iy
        up = BALLOON_H + 14
        keep = np.zeros(fl.a.shape[:2], bool)
        keep[max(0, sy - up):max(0, sy + 12), max(0, sx - 16):max(0, sx + 17)] = True
        fl.a[~keep] = KEY
        hour = self.HOUR if (self.ID, self.ai) in daylight.INTERIORS else self.g.hour
        art = keep & np.any(fl.a != np.array(KEY, np.uint8), axis=2)
        if art.any() and abs(hour - self.HOUR) >= 0.05:   # lit like the rest of the place (world clock)
            y0, y1 = np.nonzero(art.any(axis=1))[0][[0, -1]]
            x0, x1 = np.nonzero(art.any(axis=0))[0][[0, -1]]
            box = fl.a[y0:y1 + 1, x0:x1 + 1]
            m = art[y0:y1 + 1, x0:x1 + 1]
            lit = box.copy()
            daylight.relight(lit, self.HOUR, hour, t, snow=self.ID in daylight.SNOWY, glow=self.auto_glow())
            box[m] = lit[m]
        return fl

    def render(self, t, alpha=0.0):
        if self.switch is not None:
            t0, old, dir = self.switch
            k = clamp((t - t0 + alpha) / AREA_FRAMES)
            new = None
            if sweep.needs_b(k, dir):           # the new area is drawn only once it shows, once per step
                if self._sweep_new is None or self._sweep_new[0] != t:
                    self._sweep_new = (t, self._world(t, with_balloon=True))
                new = self._sweep_new[1]
            fr = Frame(sweep.cloud(old, new, k, hour=self.sweep_hour(), dir=dir))
            self._hud(fr.hud, t)
            return fr
        cx, cy, bx, by = self.view(alpha)
        ix, iy = math.floor(cx), math.floor(cy)
        world = self._world(t, (cx, cy))
        self._shown = (t, world, ix, iy, bx, by)
        self._sweep_new = None
        fr = Frame(world, (ix, iy), (cx, cy), (bx, by))
        flame = self.boost or self.climb or self.leaving is not None
        hop = 0.0                                           # a happy hop when a stamp is grabbed
        if self.stamp_flash is not None and 0 <= t - self.stamp_flash < 14:
            hop = 4 * math.sin(math.pi * (t - self.stamp_flash + alpha) / 14)
        fr.balloon(bx, by, t + alpha, self.area.balloon, flame=flame, tint=self.light_tint(),
                   pattern=self.g.pattern, hop=hop, cargo=self.g.quests.cargo(),
                   wet=self.g.weather.wet, snow=self.g.weather.snow)
        self.g.weather.balloon_fx(world, t, bx - ix, by - iy, ENVELOPE_Y.get(self.area.balloon, 18))
        fr.front = self._front(t, ix, iy, bx, by)
        over = fr.front if fr.front is not None else Canvas(color=KEY)
        if self._spot_overlay(over, t, ix, iy):          # bubble / '+1': over the balloon, world-anchored
            fr.front = over
        self._hud(fr.hud, t)
        return fr

    def _hud(self, c, t):
        ui.star_counter(c, len(self.found), self.total, t, self.found_t)
        y = 16
        if self.stamp_total:
            ui.stamp_counter(c, len(self.stamps_got), self.stamp_total, t, self.stamp_flash, y=y)
            y += 13
        if self.spot_total:                             # the hotspots set off here at least once
            ui.counter(c, ui.ICONS['spark'], len(self.used), self.spot_total, t, self.spot_flash, y=y)
            y += 13
        self.g.quests.region_hud(self, c, t, y=y)
        if self.g.races.run is None:
            ui.clock(c, self.g.hour)
        self.g.races.hud(c, t, focus=lambda x, y: (x - self.cx, y - self.cy))
        kind, facts = self.CARD
        if kind:
            atlas_card(c, t, self.t_enter + 24, self.t_enter + 150, kind, facts)
        if self.done_t is not None and 0 <= t - self.done_t < 100:
            k = ease_out_back(seg(t, self.done_t, self.done_t + 8))
            if t - self.done_t > 88:
                k = 1 - seg(t, self.done_t + 88, self.done_t + 100)
            ui.plate(c, W // 2, 20, self.NAME, k)
            if k >= 0.999:
                half = ui.plate_width(self.NAME) // 2
                for s in (-1, 1):
                    ui.star(c, W // 2 + s * (half + 8) - 3, 23 - (1 if (t // 6) % 2 else 0))
        if self.visits == 1 and 150 < t - self.t_enter < 600:
            # compact, bottom-right (the card is bottom-left; tags and hotspots are mostly higher up)
            groups = ([(['SPACE'], ui.icon_interact)] if self.spot_total else []) + [(['ESC'], ui.icon_takeoff)]
            ui.hint_row(c, groups, right=W - 4, y=H - 14)
