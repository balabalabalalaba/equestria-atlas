"""The master timeline. Music and picture share it: BPM in engine/core.py, 4/4 bars.

EQUESTRIA: THE WORLD ATLAS — one day across Equestria, sunrise over Canterlot to midnight at the
Castle of the Two Sisters. 20 regions; each region segment = FLIGHT_BARS (2) bars of world-map travel
(engine.loc_worldmap.travel_frame) + 3 to 5 bars of diorama. Scene modules are written in "scene frames"
(their diorama starts at scene frame 72); engine/regionfx.long_flight maps film frame f >= FLIGHT to scene
frame f - FLIGHT_SHIFT.

Per segment:
  id, module      scene id and its python module (scenes/sNN_<id>.py)
  bars            length in bars (FRAMES_PER_BAR frames each). FIXED: the music is composed to them.
  clock           ('HH:MM', 'HH:MM') HUD clock range ('24:00' allowed = midnight), or None
  clock_end       optional local frame where the clock reaches its end time and holds (default: last frame)
  quest           (title, subtitle) for the HUD 'NEW AREA' banner and the toast, or None
  found           (areas found before this segment, total) for the HUD counter (ticks at frames-58)
  loc             world-map location id the balloon flies to (engine.loc_worldmap.LOCS)
  places          other map locations this region introduces (labels light up on arrival)
  hud             False to switch the HUD off (intro, outro)
  tod             time-of-day note for artists
  mood            music mood word for the composer
"""
from .core import FRAMES_PER_BAR, FPS

HUD = {'banner': 'NEW AREA', 'toast': 'AREA DISCOVERED'}
FLIGHT_BARS = 2                         # map flight length per region (bars)
FLIGHT = FLIGHT_BARS * FRAMES_PER_BAR   # 144 frames: film frames 0..143 of a region are the flight
FLIGHT_SHIFT = FLIGHT - FRAMES_PER_BAR  # 72: scene frame = film frame - FLIGHT_SHIFT (for f >= FLIGHT)
TOTAL_AREAS = 20


def _r(n, id, bars, clock, title, sub, loc, places, tod, mood):
    return dict(id=id, module=f'scenes.s{n:02d}_{id}', bars=bars, clock=clock, quest=(title, sub),
                meter=None, loc=loc, places=places, tod=tod, mood=mood, region=True, scene_offset=FLIGHT_SHIFT)


SEGMENTS = [
    dict(id='intro', module='scenes.s00_intro', bars=6, clock=None, quest=None, meter=None,
         hud=False, tod='night->dawn', mood='overture', region=False),
    _r(1, 'canterlot', 7, ('05:30', '06:30'), 'Canterlot', 'royal capital', 'canterlot',
       ['mt_canterhorn'], 'pre-dawn -> sunrise', 'regal fanfare'),
    _r(2, 'cloudsdale', 6, ('06:30', '07:15'), 'Cloudsdale', 'city in the sky', 'cloudsdale',
       ['galloping_gorge'], 'sunrise gold over a sea of clouds', 'airy, soaring'),
    _r(3, 'ponyville', 6, ('07:30', '08:30'), 'Ponyville', 'country town', 'ponyville',
       ['castle_friendship', 'school_friendship', 'saddle_lake'], 'fresh morning', 'cozy folk'),
    _r(4, 'sweetapple', 5, ('08:30', '09:15'), 'Sweet Apple Acres', 'apple farm', 'sweet_apple_acres',
       ['white_tail_woods'], 'morning, warm sun', 'country hoedown'),
    _r(5, 'everfree', 6, ('09:30', '10:15'), 'Everfree Forest', 'wild woods', 'everfree',
       ['zecoras_hut', 'froggy_bottom_bogg'], 'dim green forest light, mist', 'mysterious'),
    _r(6, 'heartland', 6, ('10:30', '11:15'), 'Neighagra Falls', 'heartland wonders', 'neighagra_falls',
       ['foal_mountain', 'hollow_shades', 'rambling_rock_ridge'], 'late morning', 'adventurous'),
    _r(7, 'rainbowfalls', 5, ('11:30', '12:00'), 'Rainbow Falls', 'clifftop village', 'rainbow_falls',
       ['our_town', 'starlights_cave', 'crystal_mountains'], 'bright midday', 'sporty, bright'),
    _r(8, 'manehattan', 7, ('12:15', '13:00'), 'Manehattan', 'island metropolis', 'manehattan',
       [], 'high noon, sparkling harbour', 'big-city swing'),
    _r(9, 'eastcoast', 6, ('13:15', '14:00'), 'Baltimare', 'coastal cities', 'baltimare',
       ['fillydelphia', 'horseshoe_bay', 'hayseed_swamps'], 'early afternoon, sea breeze', 'maritime'),
    _r(10, 'griffonstone', 5, ('14:15', '15:00'), 'Griffonstone', 'griffon kingdom', 'griffonstone',
       ['trottingham', 'griffonstone_station', 'bug_bear_territory'], 'overcast, windy', 'rugged minor'),
    _r(11, 'dragonlands', 6, ('15:15', '16:00'), 'Dragon Lands', 'volcanic isle', 'dragon_lands',
       ['kirin_grove', 'lost_lagoon'], 'ash-red afternoon', 'heavy riff'),
    _r(12, 'mountaris', 7, ('16:15', '17:00'), 'Mount Aris', 'sea kingdoms', 'mount_aris',
       ['seaquestria', 'basalt_beach', 'storm_kings_realm'], 'warm late afternoon, then underwater',
       'mediterranean -> underwater'),
    _r(13, 'klugetown', 6, ('17:00', '17:45'), 'Klugetown', 'desert trade town', 'klugetown',
       ['bone_dry_desert', 'sea_of_clouds', 'caves_of_conundrum'], 'hazy amber, heat shimmer', 'ragtag bazaar'),
    _r(14, 'badlands', 5, ('17:45', '18:30'), 'Changeling Kingdom', 'the badlands', 'changeling_kingdom',
       ['badlands', 'tartarus', 'forbidden_jungle'], 'golden hour', 'dark -> blooming'),
    _r(15, 'frontier', 5, ('18:30', '19:15'), 'Appleloosa', 'frontier towns', 'appleloosa',
       ['dodge_junction', 'macintosh_hills', 'rock_farm', 'ghastly_gorge'], 'sunset', 'western twang'),
    _r(16, 'laspegasus', 6, ('19:30', '20:30'), 'Las Pegasus', 'resort city', 'las_pegasus',
       ['applewood', 'san_palomino', 'somnambula'], 'dusk -> neon night', 'showtime funk'),
    _r(17, 'westcoast', 6, ('20:30', '21:00'), 'Vanhoover', 'northwest coast', 'vanhoover',
       ['tall_tale', 'smokey_mountains', 'silver_shoals', 'luna_bay', 'hope_hollow', 'unicorn_range'],
       'night, drizzle, moonrise', 'lo-fi rain'),
    _r(18, 'crystal', 7, ('21:15', '22:00'), 'Crystal Empire', 'crystal city', 'crystal_empire',
       ['frozen_north'], 'night, aurora', 'majestic shimmer'),
    _r(19, 'yakyakistan', 6, ('22:15', '22:45'), 'Yakyakistan', 'top of the world', 'yakyakistan',
       ['mt_everhoof', 'yaket_range'], 'moonlit snow', 'stomping folk'),
    _r(20, 'twosisters', 7, ('23:00', '24:00'), 'Castle of the Two Sisters', 'ancient ruins',
       'castle_two_sisters', ['tree_of_harmony'], 'midnight, full moon', 'lullaby -> climax'),
    dict(id='outro', module='scenes.s21_outro', bars=7, clock=None, quest=None, meter=None,
         hud=False, tod='night -> new dawn', mood='finale', region=False),
]

# the finale's clock reaches 00:00 exactly on the music's `midnight` hit and holds there (film frame)
next(s for s in SEGMENTS if s['id'] == 'twosisters')['clock_end'] = 288 + FLIGHT_SHIFT

TRANSITION = 10  # frames of the cloud-deck transition on each side of a segment boundary

_start = 0
_found = 0
_prev_loc = None
for _s in SEGMENTS:
    _s['frames'] = _s['bars'] * FRAMES_PER_BAR
    _s['start'] = _start
    _s['end'] = _start + _s['frames']
    _start = _s['end']
    if _s.get('region'):
        _s['found'] = (_found, TOTAL_AREAS)
        _s['src'] = _prev_loc            # where the balloon flies from (None = from the intro sky)
        _prev_loc = _s['loc']
        _found += 1
TOTAL_FRAMES = _start
DURATION = TOTAL_FRAMES / FPS
BY_ID = {s['id']: s for s in SEGMENTS}
REGIONS = [s for s in SEGMENTS if s.get('region')]
TOUR = [s['loc'] for s in REGIONS]       # the balloon's route, in order


def seg_frames(sid):
    """Number of local frames in segment `sid`."""
    return BY_ID[sid]['frames']


def locate(g):
    """Global frame -> (segment dict, local frame)."""
    for s in SEGMENTS:
        if g < s['end']:
            return s, g - s['start']
    s = SEGMENTS[-1]
    return s, s['frames'] - 1


def clock_minutes(s, f):
    """Clock time (minutes since midnight, snapped to 5 min) at local frame f."""
    if not s.get('clock'):
        return None
    a, b = (int(t[:2]) * 60 + int(t[3:]) for t in s['clock'])
    t = min(1.0, f / max(1, s.get('clock_end', s['frames'] - 1)))
    m = a + (b - a) * t
    return int(m // 5 * 5)


def hour_at(sid, f):
    """Continuous clock hour (0..24) at local frame f of segment sid (intro 4.5, outro 0.0)."""
    s = BY_ID[sid]
    if not s.get('clock'):
        return 4.5 if sid == 'intro' else 0.0
    a, b = (int(t[:2]) * 60 + int(t[3:]) for t in s['clock'])
    t = min(1.0, f / max(1, s.get('clock_end', s['frames'] - 1)))
    return (a + (b - a) * t) / 60.0
