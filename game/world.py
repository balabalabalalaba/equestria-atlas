"""EQUESTRIA ATLAS - the explorable prototype: game state, the world-map mode and transitions.

The game reuses the film's engine read-only: the world map (`engine.loc_worldmap`), the explorer's
balloon sprite, the iris / callout / atlas-card helpers, and the film scenes for the dioramas (see
game/region.py). Game logic runs at the film's 30 steps per second (`Game.step(inp)`); the display
runs at 60 fps and `Game.render(alpha)` interpolates the camera and balloon between the last two steps
(alpha 0..1) into a layered `Frame` (game/compose.py) for smooth, sub-pixel scrolling.

Sound requests collect in `Game.sounds` (name, gain, variant), the wanted music in `Game.music` ('map'
or a region's location id; `Game.music_spans` gives each region's bars of the film score). The pygame
front end (game/play.py) plays them. Nothing here imports pygame, so the game can be driven headless
(tools/game_shots.py).
"""
import importlib
import json
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np

from engine import W, H, Canvas, fx, seg, clamp, ease_in, ease_out, ease_in_out, ease_out_back, text, text_width
from engine import timeline as TL
from engine import loc_worldmap as LW
from engine import worldmap_icons as WI
from engine.atlas import map_pin, draw_balloon

from . import ui, perks, daylight
from .compose import Frame, compose, VIEW_W, VIEW_H, KEY
from .fog import PLACE_REVEAL

HOUR = 8.0                      # a new journey starts in the morning
DAY_STEPS = 24 * 60 * 30         # the world clock: one day in 24 real minutes (an hour a minute)
FLASH = 0.6                      # the postcard camera's flash at its brightest (0..1 white), easing out
OVERLAY_FADE = 5                 # steps of the cross-fade when the journal opens or closes
MAP_SPEED, MAP_BOOST = 1.6, 3.2  # art px / step
MAP_ACC = 0.14
LEAD, CAM_EASE, LEAD_EASE = 24, 0.10, 0.06
LEAD_Y = 11                     # vertical look-ahead (the view is short: the balloon stays clear of the HUD)
ALT = 22                        # balloon height above its shadow on the map
NORTH_EDGE = ALT + 26           # the balloon's northmost ground y (its envelope stays on screen)
NEAR, NEAR_NEW = 30, 45         # landing reach (px); a place found in the last NEW_T steps reaches further
FOCUS_DY = 30                   # reach is measured from the balloon's body (its basket), this far above its
                                # ground point: a player puts the balloon, not its faint shadow, on a place
NEW_T = 150
SETTLE_T, SETTLE_DAMP = 30, 0.85  # after a discovery an unsteered balloon slows harder (it stops near it)
PROMPT_IN = 6                   # steps the land prompt takes to slide up
PINK_HOLD, PINK_NEAR = 600, 60  # the pink arrow keeps its place this long, or until the balloon is near it
LAND_FRAMES = 30                # glide to the parking spot + descend, then the iris closes
TAKEOFF_FRAMES = 26
IRIS_FRAMES = 14
START = 'canterlot'
RING_TS = (0.3, 0.55, 0.8)       # rainbow rings along every leg of the tour route (bezier t)
RING_REACH, RING_RESPAWN = 10, 450
RING_UP = 34                     # a map ring floats this far above its ground point (the route)
RING_RX, RING_RY = 10, 8         # hit ellipse (half axes) around the ring's centre, for the envelope's centre
MAP_ENVELOPE = 14                # the map balloon ('m'): its envelope centre above the sprite's anchor
RING_BOOST = 4.4                 # speed (px / step) a ring flings the balloon at; it eases back down
# friendly sky traffic: other (crewless) balloons drifting over the map: (place, pattern, size, phase)
TRAFFIC = [('cloudsdale', 'sky', 's', 0.0), ('manehattan', 'royal', 's', 1.3), ('appleloosa', 'sunset', 'm', 2.1),
           ('crystal_empire', 'crystal', 's', 3.4), ('las_pegasus', 'night', 's', 4.2),
           ('mount_aris', 'lemon', 's', 5.0), ('sweet_apple_acres', 'orchard', 's', 0.7),
           # more balloons take to the sky as the places are helped (traffic_count)
           ('ponyville', 'mint', 's', 1.9), ('baltimare', 'sunset', 's', 2.8), ('rainbow_falls', 'rainbow', 's', 3.9),
           ('vanhoover', 'night', 's', 4.6), ('yakyakistan', 'royal', 's', 5.5), ('griffonstone', 'sky', 's', 0.4),
           ('klugetown', 'lemon', 'm', 1.1), ('canterlot', 'royal', 'm', 2.5)]


def traffic_count(helped):
    """Balloons in the sky: 4 at first, more as places are helped (all 15 when all 20 are)."""
    return min(len(TRAFFIC), 4 + (helped * 11 + 10) // 20)


def _rings():
    """World px (ground point) of every map ring: along the route legs, clear of every place's icon and
    label (short legs near clustered places get fewer rings)."""
    places = [icon_centre(lid) for lid in LW.LOCS]
    out = []
    for i in range(len(TL.TOUR) - 1):
        leg = LW._bezier_leg(i)
        for t in RING_TS:
            x, y = (float(v) for v in LW._bez(leg, t))
            if all(math.hypot(x - px, y - 30 - py) > 34 for px, py in places) and \
                    all(math.hypot(x - ox, y - oy) > 40 for ox, oy in out):
                out.append((x, y))
    return [_off_labels(x, y) for x, y in out]


def _off_labels(x, y):
    """A ring (drawn 6 x 9 px around y - 34) that would sit on a map name plate moves down (or up) off
    it; its place in RINGS (saved as rings_flown indices) stays the same."""
    boxes = []
    for lid, (lx, ly, kind) in LW._label_layout().items():
        label = LW.LOCS[lid]['name'].upper()
        spr = LW._plate_cached(label, 'normal') if kind == 'plate' else LW._soft_label(label, 'white', '#2e222f')
        boxes.append((lx - 2, ly - 2, spr.shape[1] + 4, spr.shape[0] + 4))

    def clear(yy):
        return not any(x - 7 < bx + bw and bx < x + 7 and yy - 44 < by + bh and by < yy - 24
                       for bx, by, bw, bh in boxes)
    for dy in (0, 8, 12, 16, -8, -12, -16):
        if clear(y + dy):
            return (x, y + dy)
    return (x, y)




def traffic_pos(k, t):
    """World ground point of sky-traffic balloon k at step t (slow loops around its place)."""
    lid, _pat, _size, ph = TRAFFIC[k]
    x, y = LW.loc_xy(lid)
    return (x + 46 * math.sin(t * 0.0031 + ph) + 14 * math.sin(t * 0.0107 + 2 * ph),
            y - 20 + 28 * math.sin(t * 0.0023 + ph * 1.7))


@dataclass
class Input:
    """Held directions/boost plus keys pressed since the last step."""
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    boost: bool = False
    action: bool = False        # SPACE / ENTER: land (map), set off a hotspot (region)
    back: bool = False          # ESC: take off / close the overview or journal
    map: bool = False           # M: overview map
    journal: bool = False       # J: the atlas journal
    pause: bool = False         # gamepad Start: the pause menu (ESC on the map opens it too)


def regions():
    """Explorable dioramas: map location id -> module name in game/regions (named after the segment id).
    Modules are imported on first landing (the film scenes they wrap are big)."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regions')
    out = {}
    for fn in sorted(os.listdir(folder)):
        seg_id = fn[:-3]
        if fn.endswith('.py') and seg_id in TL.BY_ID and TL.BY_ID[seg_id].get('region'):
            out[TL.BY_ID[seg_id]['loc']] = f'game.regions.{seg_id}'
    return out


def region_class(module_name):
    """The Region subclass defined in a game/regions module."""
    from .region import Region
    mod = importlib.import_module(module_name)
    for v in vars(mod).values():
        if isinstance(v, type) and issubclass(v, Region) and v is not Region and v.__module__ == mod.__name__:
            return v
    raise LookupError(f'no Region subclass in {module_name}')


def icon_centre(lid):
    """World px of the centre of a place's map icon."""
    x, y = LW.loc_xy(lid)
    ic = WI.icon(lid)
    return (x, y - (ic.ay - ic.h // 2)) if ic else (x, y - 6)


SAVE_VERSION = 3                # 2: + fog, map rings flown, errands, races, album, badges; 3: help (troubles)
AUTOSAVE = 300                  # steps between autosaves (when something changed)
RESUME_GAP = 150                # a loaded journey's clock starts this far past its last saved stamp
MAX_STEPS = 10 ** 9             # a saved clock or stamp past this (over a year of play) means a damaged save
FINALE_SEG = 'outro'            # the film's finale plays once every place is complete


RINGS = _rings()


class Game:
    def __init__(self, save_path=None, settings=None, title=False):
        from .settings import Settings
        self._wt = self._ot = 0         # the world clock and the overlay clock (see Game.t)
        self._stepped = True            # the mode stepped in the last Game.step (render interpolates)
        self.t = 0
        self.cheats = False             # testing keys (game/play.py --cheats): clock and balloon speed
        self.clock_mult = 1.0           # how fast the world clock runs (testing)
        self.speed_mult = 1.0           # how fast the balloon flies (testing)
        self.settings = settings if settings is not None else Settings()
        self.sounds = []
        self.music = 'map'
        self.quit = False
        self.save_path = save_path
        from .photos import PhotoBook
        self.photos = PhotoBook()       # the postcards taken (game/photos.py; game/play.py gives it the folders)
        self.photo_label = None         # where postcards are saved, as the journal shows it (set by game/play.py)
        self._reset()
        if save_path and os.path.exists(save_path):
            self._load(save_path)
        if title:
            from .menu import TitleMode
            self.mode = TitleMode(self)
        sx, sy = self.mode.iris_center()
        self.trans = dict(phase='open', t0=self.t, dur=IRIS_FRAMES + 6, c=(sx, sy))

    # -------------------------------------------------------------- the step clock
    @property
    def t(self):
        """The step clock everyone reads. It is saved (so mend / badge / photo stamps from earlier
        sessions stay in the past). While an overlay is open (the journal, the pause menu, the welcome
        card) it is the overlay's own clock, which starts where the world paused; the world clock waits,
        so everything on the map or in a place resumes exactly where it stopped."""
        return self._ot if self._overlay() else self._wt

    @t.setter
    def t(self, v):
        if self._overlay():
            self._ot = v
        else:
            self._wt = self._ot = v

    def _overlay(self):
        tu = getattr(self, 'tutor', None)
        return (getattr(self, 'journal', None) is not None or getattr(self, 'menu', None) is not None
                or (tu is not None and tu.card))

    def _reset(self):
        """Fresh progress: a new journey from Canterlot."""
        self.found = {START}            # tour stops discovered on the map
        self.found_t = {}               # lid -> step it was discovered (arrival pop)
        self.visited = set()            # dioramas entered at least once
        self.hour = HOUR                # the world clock (0..24): the map and the places are lit by it
        self.meta = {}                  # lid -> what the journal knows about a visited place
        self.saved_regions = {}         # lid -> saved progress of regions not loaded yet
        self.finale_done = False
        self._finale_from = None        # the place the finale was replayed from (the journal), or None
        self.visit_t = None             # step of the last first visit to a place (the place counter flashes)
        self.pattern = 'classic'        # the balloon's envelope pattern (wardrobe: game/compose.PATTERNS)
        self.flash_t = None             # step of the last postcard snapshot (camera flash)
        self.open_photos = False        # the journal asked to show the postcards folder, or a postcard file (a path)
                                        # in it (game/play.py does it)
        self.journal = None             # the open atlas journal (an overlay), or None
        self._ov_fade = None            # (picture faded from, 'journal' | 'world', clock step it began)
        self.menu = None                # the open pause menu (an overlay), or None
        self._last_frame = None         # the last Frame rendered (frozen behind the pause menu)
        self.hud_hidden = False         # H (game/play.py): no HUD over the view, for clean screenshots
        self._region_mod = regions()
        self.region_ids = tuple(self._region_mod)
        self._regions = {}
        self._prep = {}                 # lid -> (thread, result box): a place being built in the background
        # Phase B: errands, races, photo album, badges, the fog of war and the toasts that announce them
        from .toast import Toasts
        from .quests import Errands
        from .races import Races
        from .album import Album
        from .badges import Badges
        from .fog import Fog
        self.toasts = Toasts(self)
        self.quests = Errands(self)
        self.races = Races(self)
        self.album = Album(self)
        self.badges = Badges(self)
        self.fog = Fog()
        from .flock import Flock
        self.flock = Flock(self)        # the balloon friends (game/flock.py)
        from .tutor import Tutor
        self.tutor = Tutor(self)        # the welcome card and tips (game/tutor.py)
        from .weather import Weather
        self.weather = Weather(self)    # snow, rain, wind and the balloon's wet / snowy state
        self.rings_flown = set()        # map rainbow rings flown through at least once (indices in RINGS)
        self.map = MapMode(self)
        self.mode = self.map
        self.fog.reveal(*LW._balloon_end(START), r=96)
        from .fog import _deck
        _deck()                         # the cloud deck texture (~0.15 s), built now, not on the first frame
        self._saved_sig = None

    def has_save(self):
        return bool(self.save_path) and os.path.exists(self.save_path)

    def new_game(self):
        """Start over: the old save is kept as <save>.old."""
        if self.has_save():
            os.replace(self.save_path, self.save_path + '.old')
            print('previous save kept as', self.save_path + '.old')
        self._reset()

    def open_pause(self):
        from .menu import PauseMenu
        fr = self._last_frame if self._last_frame is not None else self.mode.render(self.t, 0.0)
        self.menu = PauseMenu(self, fr)
        self.sfx('pop', 0.35, 0)

    # -------------------------------------------------------------- save game
    def _state(self):
        regions = dict(self.saved_regions)
        regions.update({lid: r.save_state() for lid, r in self._regions.items()})
        m = self.map
        return dict(version=SAVE_VERSION, steps=self._wt, found=sorted(self.found), visited=sorted(self.visited),
                    hour=self.hour, balloon=[round(m.bx, 1), round(m.by, 1)], regions=regions,
                    meta=self.meta, finale_done=self.finale_done, pattern=self.pattern,
                    fog=self.fog.encode(), rings=sorted(self.rings_flown), help=self.quests.save(),
                    races=self.races.save(), album=self.album.save(), badges=self.badges.save(),
                    flock=self.flock.save(), tutor=self.tutor.save(),
                    weather=self.weather.save())

    def save(self, force=False):
        """Write the save file if anything changed (atomic: temp file + rename)."""
        if not self.save_path or self.on_title:
            return
        st = self._state()
        sig = json.dumps({k: v for k, v in st.items() if k not in ('balloon', 'steps')}, sort_keys=True)
        if sig == self._saved_sig and not force:
            return
        tmp = self.save_path + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
            with open(tmp, 'w') as fh:
                json.dump(st, fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.save_path)
        except OSError as e:                # a full disk or a read-only folder: keep playing, try again later
            if not getattr(self, '_save_warned', False):
                print('could not save the journey:', e)
                self._save_warned = True
            return
        self._saved_sig = sig

    def _load(self, path):
        try:
            with open(path) as fh:
                st = json.load(fh)
        except OSError:
            print('save file unreadable, starting fresh:', path)
            return
        except ValueError as e:                 # not JSON (cut short, garbled): kept aside, as below
            print(f'save file damaged ({type(e).__name__}), kept as {path}.bad; starting fresh')
            try:
                os.replace(path, path + '.bad')
            except OSError:
                pass
            return
        try:
            if not isinstance(st, dict):
                raise ValueError(f'the save is a {type(st).__name__}, not a dict')
            self._load_state(st)
        except Exception as e:                  # a damaged save: keep it aside and start a new journey
            print(f'save file damaged ({type(e).__name__}: {e}), kept as {path}.bad; starting fresh')
            try:
                os.replace(path, path + '.bad')
            except OSError:
                pass
            self._reset()
            self._wt = self._ot = 0

    def _resume_clock(self, st):
        """The step clock of a saved journey: the saved 'steps', and never before the latest stamp in
        the save (a save from before the clock was saved has none), plus a short gap so nothing that
        happened before the save replays (a mend's magic lasts MEND + 70 steps)."""
        stamps = [0]
        help_, badges, album = (st.get(k) for k in ('help', 'badges', 'album'))
        for d, key in ((help_, 'fixed'), (help_, 'seen'), (badges, 'earned'), (album, 'taken')):
            sub = d.get(key) if isinstance(d, dict) else None
            if isinstance(sub, dict):
                stamps += [int(v) for v in sub.values() if isinstance(v, (int, float)) and math.isfinite(v)]
        steps = st.get('steps')
        steps = int(steps) if isinstance(steps, (int, float)) and math.isfinite(steps) and steps >= 0 else 0
        t = max(steps, max(stamps) + 1) + RESUME_GAP
        if t > MAX_STEPS:               # (the art's clock arithmetic would overflow)
            raise ValueError(f'step clock {t} out of range')
        return t

    def _load_state(self, st):
        self._wt = self._ot = self._resume_clock(st)   # first: the subsystems below read and compare g.t
        self.found = set(st.get('found', [START])) & set(TL.TOUR) | {START}
        self.visited = set(st.get('visited', ())) & set(TL.TOUR)
        self.hour = float(st.get('hour', HOUR))
        self.hour = self.hour % 24.0 if math.isfinite(self.hour) else HOUR   # (a corrupt value is dropped)
        self.meta = dict(st.get('meta', {}))
        self.saved_regions = dict(st.get('regions', {}))
        self.finale_done = bool(st.get('finale_done', False))
        self.quests.load(st.get('help'))       # (a version 2 'errands' chain is dropped)
        self.races.load(st.get('races'))
        self.flock.load(st.get('flock'))
        self.weather.load(st.get('weather'))
        self.tutor.load(st.get('tutor'), old_player=bool(self.visited))
        self.album.load(st.get('album'))
        self.badges.load(st.get('badges'))
        self.rings_flown = {int(i) for i in st.get('rings', ()) if 0 <= int(i) < len(RINGS)}
        bx, by = (float(v) for v in st.get('balloon', (self.map.bx, self.map.by)))
        if not (math.isfinite(bx) and math.isfinite(by)):
            bx, by = self.map.bx, self.map.by
        bx, by = clamp(bx, 12, LW.MAP_W - 12), clamp(by, NORTH_EDGE, LW.MAP_H - 6)
        if not (isinstance(st.get('fog'), str) and self.fog.decode(st['fog'])):
            # a version 1 save: chart the land around every place already found and the balloon
            for lid in self.found:
                self.fog.reveal(*icon_centre(lid), r=PLACE_REVEAL)
            self.fog.reveal(float(bx), float(by), r=96)
        self.pattern = st.get('pattern', 'classic')
        if self.pattern not in self.unlocked_patterns():
            self.pattern = 'classic'
        self.map.place(float(bx), float(by))
        self.badges.check(quiet=True)
        self._saved_sig = None

    # -------------------------------------------------------------- progress
    def place_progress(self, lid):
        """(landmarks found, landmarks total or None, stamps got, stamps total or None) of a place."""
        if lid in self._regions:
            r = self._regions[lid]
            return len(r.found), r.total, len(r.stamps_got), r.stamp_total
        st, me = self.saved_regions.get(lid, {}), self.meta.get(lid)
        tot = len(me['landmarks']) if me else None
        return len(st.get('found', ())), tot, len(st.get('stamps', ())), (me['stamps'] if me else None)

    def stamps_got(self):
        return sum(self.place_progress(lid)[2] for lid in TL.TOUR)

    def unlocked_patterns(self):
        """Wardrobe patterns earned so far: by stamps collected, or (Phase B) by deliveries, gold
        medals and album photos (need = 'helped:N' / 'gold:N' / 'photos:N')."""
        from .compose import PATTERNS
        n = self.stamps_got()
        have = {'helped': len(self.quests.fixed), 'friends': len(self.flock.met), 'gold': self.races.golds(), 'photos': len(self.album.taken)}
        out = []
        for key, _name, need, _spec in PATTERNS:
            if isinstance(need, int):
                ok = n >= need
            else:
                kind, k = need.split(':')
                ok = have.get(kind, 0) >= int(k)
            if ok:
                out.append(key)
        return out

    def snapshot_flash(self):
        """The postcard camera went off (game/play.py saves the postcard); the photo album checks
        whether the picture shows one of its targets."""
        self.flash_t = self._wt                 # (the flash plays on the world clock, as render reads it)
        self.sfx('shutter', 0.6)
        if not self.on_title and not isinstance(self.mode, Cutscene):
            self.album.snap()

    def all_complete(self):
        for lid in TL.TOUR:
            n, tot, _s, _st = self.place_progress(lid)
            if not tot or n < tot:
                return False
        return True

    # -------------------------------------------------------------- services for the modes
    def sfx(self, name, gain=0.6, variant=None):
        self.sounds.append((name, float(gain), variant))

    def region(self, lid):
        if lid not in self._regions:
            r = region_class(self._region_mod[lid])(self)
            if lid in self.saved_regions:
                r.load_state(self.saved_regions.pop(lid))
            self._regions[lid] = r
            self.meta[lid] = r.meta()
        return self._regions[lid]

    @property
    def on_title(self):
        return type(self.mode).__name__ == 'TitleMode'

    @property
    def music_spans(self):
        spans = {lid: r.MUSIC for lid, r in self._regions.items()}
        spans[FINALE_SEG] = _section_span(FINALE_SEG)
        spans['title'] = _section_span('intro')
        return spans

    def can_enter(self, lid):
        return lid in self._region_mod

    def try_region(self, lid):
        """The region object, or None if it fails to load (it is then padlocked for this session).
        A first load also builds all its art, so nothing stalls once inside; one prepared in the
        background (prepare_region) is adopted (waiting for it if it is not quite done)."""
        try:
            fresh = lid not in self._regions
            if fresh and lid in self._prep:
                th, box = self._prep.pop(lid)
                th.join()
                if 'r' in box:
                    r = box['r']
                    if lid in self.saved_regions:
                        r.load_state(self.saved_regions.pop(lid))
                    self._regions[lid] = r
                    self.meta[lid] = r.meta()
                    return r
            r = self.region(lid)
            if fresh:
                r.warm()
            return r
        except Exception:
            self._broken(lid)
            return None

    PREP_KEEP = 2                       # places built in the background but not landed in, kept at most

    def prepare_region(self, lid):
        """Build and warm place `lid` in a background thread, so landing there does not stall the game
        (the art takes ~0.1-0.3 s to build; numpy lets the main thread keep drawing meanwhile). The map
        calls it while the balloon hovers where it can land and when it starts to land; try_region adopts
        the result. The thread only builds its own Region object: the game's tables are left to the main
        thread."""
        if lid in self._regions or lid in self._prep or lid not in self._region_mod:
            return
        import threading
        box = {}
        mod = self._region_mod[lid]

        def work():
            try:
                r = region_class(mod)(self)
                r.warm()
                box['r'] = r
            except Exception:           # try_region builds it again on the main thread (and reports it)
                pass

        th = threading.Thread(target=work, name='prepare_' + lid, daemon=True)
        self._prep[lid] = (th, box)
        th.start()
        done = [k for k, (t_, _b) in self._prep.items() if k != lid and not t_.is_alive()]
        for k in done[:max(0, len(self._prep) - 1 - self.PREP_KEEP)]:   # (memory: a place is 10-60 MB)
            self._prep.pop(k, None)

    def _broken(self, lid):
        print(f'region {lid} failed and is closed for this session:')
        traceback.print_exc()
        self._region_mod.pop(lid, None)
        self._regions.pop(lid, None)
        self.region_ids = tuple(self._region_mod)

    def _bail_out(self):
        """A region raised while running: back to the map, beside its icon."""
        lid = getattr(self.mode, 'ID', None)
        self._broken(lid)
        self.trans = None
        self.mode = self.map.take_off_from(lid) if lid in LW.LOCS else self.map

    def iris_to(self, cx, cy, then):
        """Close the iris on screen point (cx, cy), then switch to the mode returned by then()."""
        self.trans = dict(phase='close', t0=self.t, dur=IRIS_FRAMES, c=(int(cx), int(cy)), then=then)
        self.sfx('whoosh', 0.35)

    def start_finale(self):
        """Iris out to the film's finale (it returns to the map when it ends, or to the place it was
        replayed from: the balloon takes off from there)."""
        if self._overlay():             # a replay from the journal (it closes this step): the iris
            self._ot = self._wt         # runs on the world clock, which waited behind the journal
        lid = getattr(self.mode, 'ID', None)
        self._finale_from = lid if lid in LW.LOCS and lid in self._regions else None

        def go():
            self.finale_done = True
            self.music = FINALE_SEG
            return Cutscene(self, FINALE_SEG)
        sx, sy = self.mode.iris_center()
        self.iris_to(sx, sy, go)

    def world_mended(self):
        """Every place has been helped: the finale's cue (on the next take-off)."""
        from . import quests
        return len(self.quests.fixed) >= len(quests.TROUBLES)

    # -------------------------------------------------------------- loop
    def step(self, inp):
        # Overlays (the welcome card, the journal, the pause menu) run on their own clock: g.t reads the
        # overlay clock while one is open and the world clock waits (see Game.t), so nothing on the map
        # or in a place moves on behind them. _stepped = False: the mode did not step, so render holds
        # its view still instead of interpolating the last step again.
        self.sounds = []
        self._stepped = False
        if self.tutor.card:                              # the welcome card (a new game)
            if not self.tutor.card_step(inp):
                self.save()
            self.t += 1
            return
        if self.journal is not None:
            if not self.journal.step(inp):
                shown = compose(self.journal.render(self.t), 1)    # the world fades in from the last page
                self.journal = None
                self._ov_fade = (shown, 'world', self._wt)
                self.sfx('pop', 0.35, 1)
            self.t += 1
            return
        if self.menu is not None:
            if not self.menu.step(inp):
                self.menu = None
                self.sfx('pop', 0.35, 1)
            self.t += 1
            return
        if inp.pause and self.trans is None and not self.on_title and not isinstance(self.mode, Cutscene):
            self.open_pause()
            self.t += 1
            return
        if inp.journal and self.trans is None and not self.on_title and not isinstance(self.mode, Cutscene):
            from .journal import Journal
            under = self._last_frame                    # the journal fades in over the view it covers
            self.journal = Journal(self)
            if under is not None:
                self._ov_fade = (compose(under, 1), 'journal', self._ot)
            self.tutor.saw_journal = True
            self.sfx('pop', 0.35, 0)
            self.t += 1
            return
        if self.save_path and self.t % AUTOSAVE == 0:
            self.save()
        if not self.on_title and not isinstance(self.mode, Cutscene):
            self.hour = (self.hour + 24.0 / DAY_STEPS * self.clock_mult * perks.clock(self)) % 24.0   # the world clock runs
        if not self.on_title:
            if self.tutor.want_card():
                self.tutor.open_card()
                self.t += 1
                return
            self.tutor.step()
            if not isinstance(self.mode, Cutscene):      # the finale shows no toasts: they wait for its end
                self.toasts.step(self.t)
                if self.t % 15 == 0:
                    self.badges.check()
        tr = self.trans
        if tr is not None and self.t - tr['t0'] >= tr['dur']:
            if tr['phase'] == 'close':
                self.mode = tr['then']()
                self.trans = dict(phase='hold', t0=self.t, dur=4, c=tr['c'])
            elif tr['phase'] == 'hold':
                self.trans = dict(phase='open', t0=self.t, dur=IRIS_FRAMES, c=self.mode.iris_center())
            else:
                self.trans = None
        try:
            self._stepped = True
            self.mode.step(inp if self.trans is None else Input())
        except Exception:
            if self.mode is self.map:
                raise
            self._bail_out()
        self.t += 1

    def render(self, alpha=0.0):
        """The current frame as layers; alpha = fraction of the next step already elapsed."""
        if self.journal is not None:
            return self._faded(self.journal.render(self.t), 'journal', self.t, 0.0)
        if self.menu is not None:
            return self.menu.render(self.t)
        t = self._wt                                    # the world's clock (the welcome card has its own)
        if not self._stepped:                           # the world waited this step: hold the view still
            alpha = 1.0
        try:
            fr = self.mode.render(t, alpha)
        except Exception:
            if self.mode is self.map:
                raise
            self._bail_out()
            fr = self.mode.render(t, alpha)
        if not self.on_title and not isinstance(self.mode, Cutscene):
            self.album.draw_sparkle(fr.hud, t)
            done_t = getattr(self.mode, 'done_t', None)       # a place's completion plate holds y 20..33
            low = done_t is not None and 0 <= t - done_t < 100
            racing = self.races.run is not None or self.flock.race is not None   # a race plate holds the top-left
            self.toasts.draw(fr.hud, t, y=self._toast_y(fr, 38 if low else 18 if racing else 4))
            on_map = self.mode is self.map and not self.map.overview
            if self.tutor.card:
                if on_map:
                    self.map.draw_prompt(fr.hud, t, alpha)
                fr.dim = True
                self.tutor.draw_card(fr.hud, self.t)
            else:
                self.tutor.draw(fr.hud, t)
                if on_map:                              # the land prompt: over a tip folding away above it
                    self.map.draw_prompt(fr.hud, t, alpha)
            if not self.tutor.card:                       # the balloon shows through what covers it
                from .compose import see_through
                see_through(fr)
            if self.cheats:                               # testing: what the cheat keys have set
                s = f'TEST  CLOCK x{self.clock_mult:g}  BALLOON x{self.speed_mult:g}'
                fr.hud.rect(3, H - 10, text_width(s, 'tiny') + 4, 8, '#2e222f')
                text(fr.hud, 5, H - 9, s, 'mint', font='tiny')
        if self.flash_t is not None and t - self.flash_t < 11:
            # a soft click: the first frames drawn after P (half a step to a step later) rise over three
            # frames, then it eases out
            u = max(0.0, t - self.flash_t + alpha - 0.5)
            fr.flash = FLASH * min(1.0, (u + 0.25) / 1.5) * max(0.0, 1 - u / 10) ** 2
        tr = self.trans
        if tr is not None:
            k = clamp((t - tr['t0'] + alpha) / tr['dur'])
            if tr['phase'] == 'close':
                fr.iris = (ease_in(k), *tr['c'])
            elif tr['phase'] == 'hold':
                fr.iris = (1.0, *tr['c'])
            else:
                fr.iris = (1 - ease_out(k), *tr['c'])
        if self.hud_hidden and not self.on_title and not self.tutor.card:
            fr.hud = Canvas(color=KEY)                  # (before the journal's fade bakes the frame)
        self._last_frame = fr
        return self._faded(fr, 'world', t, alpha)

    def _faded(self, fr, side, t, alpha):
        """fr, cross-faded from the picture the journal opened over (side 'journal') or the journal's
        last page (side 'world') during the OVERLAY_FADE steps after it opened or closed."""
        f = self._ov_fade
        if f is None or f[1] != side:
            return fr
        k = (t - f[2] + alpha) / OVERLAY_FADE
        if not 0 <= k < 1:
            return fr
        k = k * k * (3 - 2 * k)
        a = compose(fr, 1).astype(np.float32)
        c = Canvas()
        c.a[:VIEW_H, :VIEW_W] = (f[0].astype(np.float32) * (1 - k) + a * k + 0.5).astype(np.uint8)
        c.a[VIEW_H, :VIEW_W] = c.a[VIEW_H - 1, :VIEW_W]
        c.a[:, VIEW_W] = c.a[:, VIEW_W - 1]
        out = Frame(c)
        out.focus, out.cam = fr.focus, (0.0, 0.0)
        return out

    def _toast_y(self, fr, y):
        """Toasts sit at the top centre. One that starts while the balloon's envelope is up there (in the
        toast's columns) shows low instead, at y = H - 78, for its whole life: the slot is latched in the
        toast itself when it first shows and never moves mid-display."""
        cur = self.toasts.cur
        if cur is None:
            return y
        low = cur.get('_low')
        if low is None:
            low = False
            if fr.focus is not None and fr.cam is not None:
                lift = 40 if self.mode is self.map else 2 * _envelope_y(getattr(self.mode, 'area', None))
                ex, top = fr.focus[0] - fr.cam[0], fr.focus[1] - fr.cam[1] - lift   # envelope top (screen)
                tw = max(text_width(cur['title'], 'normal'), text_width(cur['sub'], 'tiny') if cur['sub'] else 0)
                half = (tw + (12 if cur['icon'] is not None else 0) + 12) // 2 + 12
                low = y - 26 < top < y + 28 and abs(ex - W / 2) < half   # (the slot actually used: the
                # completion plate pushes it down to y 38, a race plate to 18)
            cur['_low'] = low
        return H - 78 if low else y


class MapMode:
    """Fly the balloon over the top-down world map; discover the 20 tour stops; land to explore."""

    def __init__(self, game):
        self.g = game
        x, y = LW._balloon_end(START)
        self.bx, self.by = float(x), float(y)
        self.vx = self.vy = 0.0
        self.alt = float(ALT)
        self.look_x = self.look_y = 0.0
        self.cx, self.cy = self.bx, self.by - 10
        self.boost = False
        self.landing = None             # (lid, step, start x, start y)
        self.takeoff = None             # step the balloon lifted off a place
        self.overview = False
        self.near = None
        self.locked_t = None            # step the player tried to land at a closed place
        self.moved = 0                  # steps of steering so far (hides the key hints)
        self.ring_t = {}                # ring index -> step it was flown through (it pops, then returns)
        self.hop_t = None               # step of the last happy hop (ring, discovery)
        self.discover_t = None          # step of the last discovery (the balloon settles after it)
        self.pink_lid = None            # the place the pink arrow points at (held, see _pick_pink)
        self.pink_t = 0
        self.first_visit = None         # a place entered for the first time (the counter flashes on take-off)
        self.visit_flash_t = None
        self.prompt_on = False          # the land prompt showed last step; prompt_t = step it came up
        self.prompt_t = None
        self._clamp_cam()
        self.prev = self.snapshot()

    # -------------------------------------------------------------- geometry
    def _clamp_cam(self):
        self.cx = clamp(self.cx, W / 2, LW.MAP_W - W / 2)
        self.cy = clamp(self.cy, H / 2, LW.MAP_H - H / 2)

    def snapshot(self):
        return (self.cx, self.cy, self.bx, self.by, self.alt)

    def view(self, alpha):
        return tuple(a + (b - a) * alpha for a, b in zip(self.prev, self.snapshot()))

    def iris_center(self):
        ox, oy = self.cx - W / 2, self.cy - H / 2
        return int(round(self.bx - ox)), int(round(self.by - self.alt - oy)) - 7

    def place(self, x, y):
        """Put the balloon (and the camera) at world (x, y), airborne."""
        self.bx, self.by = x, max(y, NORTH_EDGE)
        self.vx = self.vy = self.look_x = self.look_y = 0.0
        self.cx, self.cy = x, self.by - 10
        self._clamp_cam()
        self.prev = self.snapshot()

    def take_off_from(self, lid):
        """Back from a diorama: the balloon rises off its parking spot beside lid's icon. The map takes
        the time of day of the place just visited."""
        g = self.g
        if lid in g._regions:
            g.save()
        x, y = LW._balloon_end(lid)
        self.bx, self.by = float(x), float(y)
        self.vx = self.vy = 0.0
        self.look_x = self.look_y = 0.0
        self.alt = 4.0
        self.cx, self.cy = self.bx + 20, self.by - 10
        self._clamp_cam()
        self.prev = self.snapshot()
        self.takeoff = self.g.t
        self.landing = None
        if self.first_visit == lid:                  # the place counter shows the visit as the balloon rises
            self.visit_flash_t = self.g.t + 4 + IRIS_FRAMES  # (once the iris is open)
        self.first_visit = None
        self.g.weather.drift = (0.0, 0.0)
        self.g.music = 'map'
        return self

    # -------------------------------------------------------------- update
    def step(self, inp):
        g = self.g
        self.prev = self.snapshot()
        if self.overview:
            self._hold_races()
            if inp.map or inp.back or inp.action:
                self.overview = False
                g.sfx('pop', 0.4, 1)
            return
        if inp.back and self.landing is None and g.trans is None:   # ESC on the map: the pause menu
            g.open_pause()
            return
        if inp.map and self.landing is None:
            self.overview = True
            g.sfx('pop', 0.4, 0)
            return
        if self.landing is not None:
            self._step_landing()
            return
        if self.takeoff is not None:
            k = seg(g.t, self.takeoff, self.takeoff + TAKEOFF_FRAMES)
            self.alt = 4 + (ALT - 4) * ease_out(k)
            if g.t == self.takeoff + 2:
                g.sfx('puff', 0.4)
            if k >= 1:
                self.takeoff = None
                if not g.finale_done and g.world_mended():   # every place helped: the finale
                    g.start_finale()
        self._fly(inp)
        g.weather.map_step(self)                        # wind carries the balloon; rain / snow settle
        g.fog.reveal(self.bx, self.by, wide=perks.fog_wide(g))
        self._rings()
        g.races.map_step(self)
        if self.takeoff is None and math.hypot(self.vx, self.vy) > 0.3:   # only while flying
            self._discover()
        self._near()
        if self.near is not None and self.near not in g._regions:   # build it now, in the background
            g.prepare_region(self.near)
        on = self.prompt_up()
        if on and not self.prompt_on:                   # the land prompt comes up: it slides in
            self.prompt_t = g.t
        self.prompt_on = on
        self._pick_pink()
        if g.flock.map_step(self, inp):                   # SPACE went to a balloon friend
            inp = Input(**dict(vars(inp), action=False))
        if inp.action and self.near is not None and self.takeoff is None and g.trans is None:
            if g.can_enter(self.near):                  # (not mid-climb: the land prompt is hidden then)
                g.races.cancel()
                g.flock.race = None                     # landing ends a balloon friend's race too
                self.landing = (self.near, g.t, self.bx, self.by)
                g.sfx('puff', 0.5)
            else:
                self.locked_t = g.t
                g.sfx('tick', 0.5)

    def _hold_races(self):
        """The overview map is up: the race clocks (a ring course, a friend's race and the friends out
        on an outing) wait with it, so a look at the map never costs a race."""
        g = self.g
        run = g.races.run
        if run is not None:
            run['t0'] += 1
            run['last'] += 1
        fl = g.flock
        if fl.race is not None:
            fl.race['t0'] += 1
        for o in fl.out.values():
            o['t0'] += 1

    def _pick_pink(self):
        """The place the pink edge arrow points at: the nearest place still to visit, held until it is
        visited, the balloon comes within PINK_NEAR px of it, or PINK_HOLD steps have passed (so the
        arrow does not flick between two places at the same distance). A place reached but not landed
        in hands the arrow on to the next one."""
        g = self.g
        lid = self.pink_lid
        reached = None
        if lid is not None and lid not in g.visited and lid in g.region_ids:
            x, y = icon_centre(lid)
            if math.hypot(self.bx - x, self.by - y) < PINK_NEAR:
                reached = lid
            elif g.t - self.pink_t < PINK_HOLD:
                return
        best, bd = None, 1e18
        for k in g.region_ids:
            if k in g.visited or k == reached:
                continue
            x, y = icon_centre(k)
            d = (x - self.bx) ** 2 + (y - self.by) ** 2
            if d < bd:
                best, bd = k, d
        self.pink_lid = best if best is not None else reached
        self.pink_t = g.t

    def _fly(self, inp):
        ax = (1 if inp.right else 0) - (1 if inp.left else 0)
        ay = (1 if inp.down else 0) - (1 if inp.up else 0)
        n = math.hypot(ax, ay)
        if n:
            ax, ay = ax / n, ay / n
            self.moved += 1
        self.boost = inp.boost and n > 0
        pk = perks.speed(self.g, self.boost)            # the worn balloon's perk (game/perks.py)
        acc = MAP_ACC * (1.7 if self.boost else 1.0) * self.g.speed_mult * pk
        dt = self.g.t - self.discover_t if self.discover_t is not None else SETTLE_T
        damp = SETTLE_DAMP if n == 0 and dt < SETTLE_T else perks.glide(self.g, 0.9)   # just found a place: it stops near it
        self.vx = self.vx + ax * acc if ax else self.vx * damp
        self.vy = self.vy + ay * acc if ay else self.vy * damp
        vmax = (MAP_BOOST if self.boost else MAP_SPEED) * self.g.speed_mult * pk
        sp = math.hypot(self.vx, self.vy)
        if sp > vmax:                    # ease down after a boost instead of snapping
            k = max(vmax, sp * 0.94) / sp
            self.vx *= k
            self.vy *= k
        wx, wy = self.g.weather.drift                     # the wind carries the balloon
        nx, ny = self.bx + self.vx + wx, self.by + self.vy + wy
        if not 12 <= nx <= LW.MAP_W - 12:                 # at the map's edge: no speed kept against it,
            self.vx = 0.0                                 # so turning back answers at once
        if not NORTH_EDGE <= ny <= LW.MAP_H - 6:
            self.vy = 0.0
        self.bx = clamp(nx, 12, LW.MAP_W - 12)
        self.by = clamp(ny, NORTH_EDGE, LW.MAP_H - 6)
        self._follow(self.bx, self.by - 10, LEAD, (wx, wy))

    def _follow(self, x, y, lead, drift=(0.0, 0.0)):
        """Eased camera: the look-ahead eases towards the velocity (with the wind's drift), the camera eases
        towards the target. The view is short, so it looks ahead less vertically (LEAD_Y): the balloon
        stays clear of the top and bottom HUD."""
        self.look_x += ((self.vx + drift[0]) * lead - self.look_x) * LEAD_EASE
        self.look_y += ((self.vy + drift[1]) * lead * LEAD_Y / LEAD - self.look_y) * LEAD_EASE
        self.cx += (x + self.look_x - self.cx) * CAM_EASE
        self.cy += (y + self.look_y - self.cy) * CAM_EASE
        self._clamp_cam()

    def envelope(self):
        """World px of the balloon's envelope centre (it flies `alt` above its shadow, by)."""
        return self.bx, self.by - self.alt - MAP_ENVELOPE

    def _rings(self):
        # The envelope (the round part you steer through the ring) must meet the ring's centre. This
        # used to compare the ground points (shadow vs route) in a 10 px box, which put the hit window
        # 2 px low and lopsided: 12 px below the envelope's centre (down at the basket) but 8 above, so a
        # ring flown into from above counted at the basket (GitHub issue #5).
        g = self.g
        ex, ey = self.envelope()
        for i, (x, y) in enumerate(RINGS):
            if g.t - self.ring_t.get(i, -10 ** 6) < RING_RESPAWN:
                continue
            if ((ex - x) / RING_RX) ** 2 + ((ey - (y - RING_UP)) / RING_RY) ** 2 < 1:
                self.ring_t[i] = g.t
                g.rings_flown.add(i)
                self.hop_t = g.t
                sp = math.hypot(self.vx, self.vy)
                if sp > 0.2:                            # fling the balloon on along its heading
                    rb = RING_BOOST * perks.ring(g)
                    self.vx, self.vy = self.vx / sp * rb, self.vy / sp * rb
                g.sfx('sparkle', 0.4)
                g.sfx('whoosh', 0.3)

    def _discover(self):
        g = self.g
        for lid in TL.TOUR:
            if lid in g.found:
                continue
            x, y = icon_centre(lid)
            ic = WI.icon(lid)
            r = 18 + (ic.w // 4 if ic else 0)
            if math.hypot(self.bx - x, self.by - FOCUS_DY - y) < r:
                g.found.add(lid)
                g.found_t[lid] = g.t
                self.hop_t = self.discover_t = g.t
                g.sfx('star', 0.6)

    def _near(self):
        """The found place under the balloon (it can land there). A place found in the last NEW_T steps
        and not visited yet reaches further (NEAR_NEW), so a fly-over still offers the landing."""
        g = self.g
        best, bd = None, 1e9
        for lid in g.found:
            x, y = icon_centre(lid)
            d = math.hypot(self.bx - x, self.by - FOCUS_DY - y)
            fresh = lid not in g.visited and g.t - g.found_t.get(lid, -10 ** 9) < NEW_T
            if d < (NEAR_NEW if fresh else NEAR) and d < bd:
                best, bd = lid, d
        self.near = best

    def _step_landing(self):
        g = self.g
        lid, t0, sx, sy = self.landing
        k = seg(g.t, t0, t0 + LAND_FRAMES)
        ex, ey = LW._balloon_end(lid)
        e = ease_in_out(k)
        self.bx = sx + (ex - sx) * e
        self.by = sy + (ey - sy) * e
        self.alt = ALT - (ALT - 4) * e
        self.vx = self.vy = 0.0
        self._follow(self.bx + 20, self.by - 10, 0)
        if g.t >= t0 + LAND_FRAMES and g.trans is None:     # (>=: an overlay may have taken that step)
            ox, oy = self.cx - W / 2, self.cy - H / 2
            ix, iy = icon_centre(lid)

            def enter():                    # runs behind the closed iris: loading time is not seen
                r = g.try_region(lid)
                if r is None:                   # it failed to load: stay on the map (now padlocked)
                    self.landing = None
                    self.alt = float(ALT)
                    g.sfx('tick', 0.5)
                    return self
                first = lid not in g.visited
                g.visited.add(lid)
                g.music = lid
                self.landing = None             # landed: the map is free again when the balloon comes back
                try:
                    mode = r.enter()
                except Exception:               # its arrival hooks raised: stay on the map (padlocked)
                    g._broken(lid)
                    if first:
                        g.visited.discard(lid)
                    g.music = 'map'
                    self.alt = float(ALT)
                    g.sfx('tick', 0.5)
                    return self
                if first:                       # a first visit: the place counter goes up. Its toast
                    g.visit_t = g.t             # greets the arrival: it shows first (urgent), ahead of
                    self.first_visit = lid      # the place's own news and anything older still waiting
                    g.toasts.push('FIRST VISIT', LW.LOCS[lid]['name'].upper(), ui.star, sound='coin', gain=0.5,
                                  urgent=True)
                return mode
            g.iris_to(ix - ox, iy - oy, enter)

    # -------------------------------------------------------------- render
    def render(self, t, alpha=0.0):
        if self.overview:
            return self._render_overview(t)
        g = self.g
        cx, cy, bx, by, alt = self.view(alpha)
        ox, oy = cx - W / 2, cy - H / 2
        x0, y0 = math.floor(ox), math.floor(oy)
        icon_fx = {}
        for lid in g.found:
            t0 = g.found_t.get(lid)
            if t0 is not None and t - t0 < 16:
                icon_fx.update(LW._arrival_fx(t, lid, t0))
            else:
                icon_fx[lid] = {'state': 'color'}
        c = Canvas()
        LW.draw_map(c, x0 + W // 2, y0 + H // 2, t, hour=g.hour, found=0, icon_fx=icon_fx,
                    label_fx=self._label_fx(t))
        g.weather.map_draw_ground(c, t, x0, y0)             # rain-cloud shadows
        lit = c.a.copy()
        self._draw_rings(c, t, x0, y0)
        g.races.map_draw(c, t, x0, y0, active=False)        # idle start gates: under the fog
        daylight.dim_new(c.a, lit, g.hour)                 # lit by the clock like the map under them
        g.fog.draw(c, x0, y0, t, g.hour)
        self._draw_discoveries(c, t, x0, y0)                # sparkle ring + name plate (over the fog)
        fr = Frame(c, (x0, y0), (ox, oy), (bx, by))
        # quest markers: a pin over the label of each place spotted but not yet visited, and one edge
        # arrow towards the place the pink arrow holds (_pick_pink: no clutter of arrows)
        labels = LW._label_layout()
        pink = None
        for lid in g.region_ids:
            if lid in g.visited:
                continue
            lx, ly, _kind = labels[lid]
            sx = lx - x0 + (text_width(LW.LOCS[lid]['name'].upper(), 'tiny') + 6) // 2
            sy = ly - y0 - 1
            onscreen = -4 <= sx < W + 4 and 6 <= sy < H + 4
            if onscreen and (lid in g.found or lid == self.pink_lid):   # the pink target: always marked
                map_pin(c, sx, sy, t, bounce=True)
            if lid == self.pink_lid:
                pink = (sx, sy, onscreen)
        racing = g.races.run is not None
        lit = c.a.copy()
        g.races.map_draw(c, t, x0, y0, active=True)          # the course being flown: over the fog
        g.quests.map_lines(c, t, x0, y0)                      # the helped world: rainbows, bunting
        daylight.dim_new(c.a, lit, g.hour)                    # (lit by the clock: no neon rainbows at night)
        errand_xy = g.quests.map_draw(c, t, x0, y0)
        g.quests.map_world(c, t, x0, y0, g.hour)              # festive towns: hearts, night bulbs, fireworks
        g.weather.map_draw_sky(c, t, x0, y0, g.hour)          # snow, wind, rain clouds
        tint = daylight.map_tint(g.hour)                      # the balloons, softly lit like the map
        g.flock.draw_map(c, fr, t, alpha, x0, y0, tint=tint)    # sky traffic: the balloon friends
        # the balloon and its shadow
        flame = self.boost or (self.takeoff is not None and (t - self.takeoff) < 14)
        hop = 0.0
        if self.hop_t is not None and 0 <= t - self.hop_t < 14:
            hop = 4 * math.sin(math.pi * (t - self.hop_t + alpha) / 14)
        fr.shadow(bx, by, 5)
        fr.balloon(bx, by - alt, t + alpha, 'm', flame=flame, pattern=g.pattern, hop=hop, cargo=g.quests.cargo(),
                   wet=g.weather.wet, snow=g.weather.snow, tint=tint)
        g.weather.balloon_fx(c, t, bx - x0, by - alt - y0, 13)
        # the HUD: the plates first, then the edge arrows (ui.edge_arrow keeps them clear of the plates)
        prompt = self.prompt_up()                       # (Game.render draws it, over the tutor's tip)
        self._draw_counter(fr.hud, t)
        if g.visited:                                   # stamps collected so far (the wardrobe's currency)
            ui.stamp_counter(fr.hud, g.stamps_got(), 3 * len(TL.TOUR), t)
        if not racing and g.flock.race is None:
            ui.clock(fr.hud, g.hour)
            if pink is not None and not pink[2] and g.quests.active() is None:   # (an errand: its gold arrow)
                ui.edge_arrow(fr.hud, pink[0], pink[1], t)
        g.quests.map_hud(fr.hud, t, errand_xy, y=29 if g.visited else 16, arrow=not racing)
        royal = perks.compass(g) if not racing and g.flock.race is None else None
        if royal is not None:                           # the ROYAL balloon's compass: the nearest unfound place
            rx, ry = icon_centre(royal)
            sx, sy = rx - x0, ry - y0
            if not (0 <= sx < W and 12 <= sy < H - 26):
                ui.edge_arrow(fr.hud, sx, sy, t + 7, color='gold')
            elif (t // 6) % 3 == 0:                     # in view (under the fog): a royal twinkle over it
                fx.sparkle(fr.hud, int(sx), int(sy), t % 6, 'butter', period=6, size=3)
        g.races.hud(fr.hud, t, focus=lambda gx, gy: (gx - x0, gy - y0 - 34))
        g.flock.hud(fr.hud, t, x0, y0)
        plates = any(0 <= t - t0 < 90 for t0 in g.found_t.values())
        if not prompt and not plates and self.moved < 90 and self.landing is None and not g.tutor.showing() and not g.tutor.card:
            ui.hint_row(fr.hud, [(['left', 'up', 'down', 'right'], ui.icon_balloon), (['SHIFT'], ui.icon_fast),
                                 (['SPACE'], ui.icon_land), (['M'], ui.icon_map),
                                 (['ESC'], _icon_pause)])
        return fr

    def _draw_rings(self, c, t, x0, y0):
        """Rainbow rings floating at balloon height above the route (world layer)."""
        for i, (x, y) in enumerate(RINGS):
            sx, sy = int(round(x)) - x0, int(round(y)) - y0
            if not (-12 < sx < W + 12 and -50 < sy < H + 12):
                continue
            a = t - self.ring_t.get(i, -10 ** 6)
            ry = sy - RING_UP + int(round(math.sin(t * 0.08 + i) * 1.5))
            if a < 40:                                      # just flown through: it bursts
                ui.burst(c, t, self.ring_t[i], sx, ry, ('red', 'orange', 'butter', 'leaf', 'sky', 'violet'),
                         n=24, seed=i, power=1.0, life=40, gravity=0.02)
                continue
            if a < RING_RESPAWN:
                continue
            if a < RING_RESPAWN + 20 and (a // 2) % 2:      # winks back in
                continue
            for k, cc in enumerate(('red', 'butter', 'sky')):
                c.ellipse(sx, ry, 6 - k, 9 - k, cc, fill=False)
            if (t + i * 13) % 50 < 4:
                c.px(sx - 3, ry - 6, 'white')

    def _render_overview(self, t):
        g = self.g
        c = Canvas()
        LW.draw_overview(c, t, hour=g.hour, found=0, labels=True)
        ox, oy, ow, oh = LW.OVERVIEW_RECT
        for lid in g.found:
            x, y = LW.overview_xy(lid)
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (-2, 0), (2, 0), (0, 2)):
                c.px(x + dx, y + dy, '#2e222f')
            for (dx, dy), cc in (((0, -1), '#f9c22b'), ((-1, 0), '#f9c22b'), ((1, 0), '#f9c22b'),
                                 ((0, 1), '#f7961d'), ((0, 0), '#fbff86')):
                c.px(x + dx, y + dy, cc)
        g.fog.draw_overview(c, LW.OVERVIEW_RECT, t)
        for lid in g.region_ids:
            if lid not in g.visited and g.fog.is_clear(*icon_centre(lid)):
                x, y = LW.overview_xy(lid)
                map_pin(c, x, y - 2, t, bounce=True)
        g.quests.overview_draw(c, t)
        bx = ox + int(self.bx * ow / LW.MAP_W)
        by = oy + int(self.by * oh / LW.MAP_H)
        r = 7 + (t // 4) % 3                                 # pulsing ring: "you are here"
        c.circle(bx, by - 8, r, 'hot_pink' if (t // 4) % 2 else 'white', fill=False)
        draw_balloon(c, bx, by, t, 's', bob=False)
        fr = Frame(c)
        self._draw_counter(fr.hud, t)
        ui.hint_row(fr.hud, [(['M'], None)], x=LW.RIGHT_PANEL[0] + 2, y=H - 14)
        return fr

    def _draw_counter(self, c, t):
        """The place counter (top right): places landed in, and found but not landed in yet. It flashes
        as the balloon rises from a first visit."""
        g = self.g
        ui.place_counter(c, len(g.visited), len(g.found | g.visited), len(TL.TOUR), t, self.visit_flash_t,
                         found_t=max(g.found_t.values(), default=None))

    def _label_fx(self, t):
        """Map name plates by status: lit under the balloon, gold rim for places found but not landed in
        (a white glint now and then), plain once visited; undiscovered places stay 'unfound'."""
        g = self.g
        fx = {}
        for i, lid in enumerate(TL.TOUR):
            if lid == self.near:
                fx[lid] = 'lit'
            elif lid in g.visited:
                fx[lid] = 'normal'
            elif lid in g.found:
                fx[lid] = 'lit' if (t + i * 23) % 120 < 4 else 'found'
        return fx

    @staticmethod
    def _covers_icon(px, py, pw, lid, x0, y0):
        """Would a 14 px high name plate at screen (px, py) cover another place's icon or the bubble
        over it (about 24 px above the icon)?"""
        for other in TL.TOUR:
            if other == lid:
                continue
            ox, oy = icon_centre(other)
            ox, oy = ox - x0, oy - y0
            if px - 8 < ox < px + pw + 8 and py - 4 < oy + 6 and oy - 30 < py + 14:
                return True
        return False

    def _draw_discoveries(self, c, t, x0, y0):
        """The arrival pop of the (at most two) latest discoveries, anchored to the map: a sparkle ring
        round the icon and its name plate under it, kept above the land prompt at the bottom. The newer
        plate is drawn on top; the older one steps up if they would overlap."""
        g = self.g
        recent = sorted(((t0, lid) for lid, t0 in g.found_t.items() if 0 <= t - t0 < 90), reverse=True)[:2]
        plates = []
        for t0, lid in recent:
            x, y = LW.loc_xy(lid)
            sx, sy = x - x0, y - y0
            ic = WI.icon(lid)
            cx, cy = sx, sy - ((ic.ay - ic.h // 2) if ic else 6)
            pw = ui.plate_width(LW.LOCS[lid]['name'])
            px = int(max(4, min(W - pw - 4, cx - pw // 2)))
            py = int(clamp(sy + 8, 50, H - 46))
            if self._covers_icon(px, py, pw, lid, x0, y0):      # under the icon it would hide a neighbour
                up = int(clamp(cy - (ic.h // 2 if ic else 6) - 30, 34, H - 46))   # (and its bubble): beside
                for qx, qy in ((cx + 14, py), (cx - 14 - pw, py), (px, up)):  # it, or above (under the toasts)
                    qx = int(max(4, min(W - pw - 4, qx)))
                    if not self._covers_icon(qx, qy, pw, lid, x0, y0):
                        px, py = qx, qy
                        break
            for qx, qy in plates:
                if abs(qy - py) < 16 and abs(qx - px) < pw:
                    py = max(20, qy - 16)
            plates.append((px, py))
            _discovery_ring(c, t, t0, cx, cy, ic)
        for (t0, lid), (px, py) in reversed(list(zip(recent, plates))):
            kk = ease_out_back(seg(t, t0 + 1, t0 + 7))
            if kk <= 0.05:
                continue
            name = LW.LOCS[lid]['name']
            pw = ui.plate_width(name)
            ui.plate(c, px + pw // 2, py, name, min(1.0, kk))
            if kk >= 0.999:
                for xx in (px + 2, px + pw - 3):
                    c.px(xx, py + 6, '#fbff86')

    def prompt_up(self):
        """The land prompt shows (a found place under the balloon, not landing or taking off)."""
        return self.near is not None and self.landing is None and self.takeoff is None

    def draw_prompt(self, c, t, alpha=0.0):
        """Bottom-centre plate for the place under the balloon: [SPACE] + land icon + its name (a gold NEW
        tag until the first visit) over what waits inside: landmarks found, stamps, and the place's
        trouble (? not noticed yet / the goods it needs / a heart once helped). A padlock for places
        without a diorama (it shakes when the player tries to land there). Shown from the first step
        a place is found, so a fly-over always offers the landing. It slides up from the bottom edge as
        it comes (in step with the tip plate above it, which moves up out of its way)."""
        if not self.prompt_up():
            return False
        g = self.g
        lid = self.near
        name = LW.LOCS[lid]['name']
        enter = g.can_enter(lid)
        k = 1.0 if self.prompt_t is None else clamp((t - self.prompt_t + alpha) / PROMPT_IN)
        dy = int(round((1 - ease_out(k)) * 30))
        if not enter:
            w = 3 + 7 + 5 + text_width(name, 'normal') + 6
            x, y = (W - w) // 2, H - 18 + dy
            _prompt_plate(c, x, y, w, 13, 'stone')
            shake = (1 if t % 4 < 2 else -1) if self.locked_t is not None and t - self.locked_t < 12 else 0
            _padlock(c, x + 3 + shake, y + 1)
            text(c, x + 15, y + 3, name, 'stone', font='normal', shadow='#6e2727')
            return True
        new = lid not in g.visited
        fresh = 0 <= t - g.found_t.get(lid, -999) < 90      # just found: the key pulses faster
        n, tot, s, st = g.place_progress(lid)
        chip = _trouble_chip(g, lid)
        kw = ui.keycap_width('SPACE')
        lead = 3 + kw + 2 + ui.icon_land(None, 0, 0) + 5
        w1 = text_width(name, 'normal') + (3 + ui.NEW_TAG_W if new else 0)
        w2 = ui.place_stats(None, 0, 0, n, tot, s, st, chip)
        w = lead + max(w1, w2) + 5
        x, y = (W - w) // 2, H - 27 + dy
        _prompt_plate(c, x, y, w, 22, ui.RIM)
        ui.keycap(c, x + 3, y + 7, 'SPACE', lit=(t // (5 if fresh else 12)) % 2 == 0)
        ui.icon_land(c, x + 3 + kw + 2, y + 7)
        tx = x + lead
        text(c, tx, y + 3, name, '#fff6c0', font='normal', shadow='#6e2727')
        if new:
            ui.new_tag(c, tx + text_width(name, 'normal') + 3, y + 4, t)
        ui.place_stats(c, tx, y + 12, n, tot, s, st, chip)
        return True


def _envelope_y(area):
    """Basket bottom -> envelope centre (px) of a place's balloon (game/region.py ENVELOPE_Y)."""
    from .region import ENVELOPE_Y
    return ENVELOPE_Y.get(getattr(area, 'balloon', 'l'), 18)


def _prompt_plate(c, x, y, w, h, rim):
    c.round_rect(x - 1, y - 1, w + 2, h + 2, '#6e2727', r=2)
    c.round_rect(x, y, w, h, rim, r=2)
    c.rect(x + 1, y + 1, w - 2, h - 2, ui.PLATE)
    c.rect(x + 2, y + 2, w - 4, 1, ui.PLATE_HI)


def _trouble_chip(g, lid):
    """The land prompt's trouble chip for lid (a draw function or a Sprite), or None: '?' while its
    trouble is not noticed, the goods it needs once seen, a heart once helped."""
    from . import quests, mend_art as MA
    from .items import ITEMS
    tr = quests.BY_KEY.get(lid)
    if tr is None:
        return None
    st = g.quests.status(lid)
    return MA.heart_icon if st == 'fixed' else ITEMS[tr.item] if st == 'seen' else MA.question_icon


def _discovery_ring(c, t, t0, cx, cy, ic):
    """The sparkle ring that bursts round a place's icon as it is found (as the film's arrival,
    engine.loc_worldmap._arrival_overlay), centred on screen (cx, cy)."""
    k = seg(t, t0, t0 + 14)
    if k >= 1:
        return
    r = 6 + ((ic.h if ic else 12) * 0.45 + 20) * ease_out(k)
    for i in range(12):
        if k > 0.7 and (i + t) % 2:
            continue
        ang = i / 12 * 2 * math.pi + t * 0.05
        x = int(cx + math.cos(ang) * r * 1.15)
        y = int(cy + math.sin(ang) * r * 0.85)
        white = (i + t // 2) % 3 == 0
        c.px(x, y, '#ffffff' if white else '#f9c22b')
        if k < 0.6:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c.px(x + dx, y + dy, '#f9c22b' if white else '#f7961d')


def _icon_pause(c, x, y):
    if c is None:
        return 9
    ui.ICONS['pause'].recolor({'ink': 'tan'}).draw(c, x, y)
    return 9


def _padlock(c, x, y):
    """Closed diorama (not built for the prototype yet)."""
    c.rect(x, y + 4, 7, 6, 'ink')
    c.rect(x + 1, y + 5, 5, 4, 'gold')
    c.px(x + 3, y + 6, 'ink')
    c.px(x + 3, y + 7, 'ink')
    c.rect(x + 1, y, 5, 5, 'ink')
    c.rect(x + 2, y + 1, 3, 4, 'stone')
    c.rect(x + 3, y + 1, 1, 3, 'ink')
    c.hline(x + 2, x + 4, y + 9, 'amber')


def _section_span(seg_id):
    """(start, end) seconds of a segment in the film score (audio/out/sections.json)."""
    from .region import _SECTIONS
    return _SECTIONS.get(seg_id, (0.0, 0.0))


class Cutscene:
    """A film segment played as it is (the finale: scenes/s21_outro.py); SPACE / ESC skips after 2 s.
    Returns to the map when it ends."""

    def __init__(self, game, seg_id):
        self.g = game
        self.mod = importlib.import_module(TL.BY_ID[seg_id]['module'])
        self.n = self.mod.N
        self.f0 = game.t

    def iris_center(self):
        return W // 2, H // 2

    def step(self, inp):
        g = self.g
        f = g.t - self.f0
        if f >= self.n - 1 or ((inp.action or inp.back) and f > 60):
            def back():
                g.music = 'map'
                g.save(force=True)
                lid, g._finale_from = g._finale_from, None
                if lid is not None:             # replayed from inside a place: take off from it
                    return g.map.take_off_from(lid)
                g.map.landing = None            # never back to a landing that already happened
                g.map.alt = float(ALT)
                return g.map
            if g.trans is None:
                g.iris_to(W // 2, H // 2, back)

    def render(self, t, alpha=0.0):
        f = min(self.n - 1, max(0, t - self.f0))
        return Frame(self.mod.render(f))
