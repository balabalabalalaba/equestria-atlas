"""World map of Equestria: the detailed map, the travel shots, the overview and a Mode-7 view.

Coordinates: world pixels on the detailed map (MAP_W x MAP_H), from the normalised positions of
docs/RESEARCH_EQUESTRIA.md (2017 expanded official map; x 0 = west, y 0 = north). Full doc:
docs/LOC_WORLDMAP.md.

API (keep these names/signatures; scenes and the intro/outro depend on them):
  MAP_W, MAP_H                      detailed map size in world px
  LOCS[id] = dict(name, x, y, kind) every labelled place (kind: capital city town village landmark
                                    natural region forest swamp desert sea foreign)
  loc_xy(id) -> (x, y)              world px of a location (= ground anchor of its icon)
  travel_frame(f, seg_id) -> Canvas 2-bar map flight of a region segment (film-local f 0..143):
                                    lift-off 0..16, even-speed flight 14..112 (Mode 7 on long hops),
                                    ARRIVE hit at f 126, places light up 128..134, iris 136..143.
  draw_map(c, cx, cy, f, hour=12.0, found=None, zoom=1, labels=True, route_upto=None, balloon=None)
                                    top-down view of the detailed map centred on world (cx, cy).
  draw_overview(c, f, hour=12.0, found=None, route_upto=None, labels=True, reveal=1.0, frame=True)
                                    the whole map fitted to the 320x180 screen (decorated frame).
  overview_xy(id) -> (sx, sy)       screen position of a location on the overview.
  draw_mode7(c, cx, cy, heading, f, altitude=60, hour=12.0, found=None, pitch=24, route_upto=None)
                                    SNES-style perspective view (camera looks at world (cx, cy)).
  dest_screen(seg_id) -> (sx, sy)   where travel_frame frames the destination icon (iris centre).
"""
import math
from functools import lru_cache

import numpy as np

from .core import (W, H, Canvas, col, mix, shade, seg, clamp, lerp, smooth, ease_in_out, ease_out,
                   ease_in, ease_out_back, hash01)
from .font import text, text_width
from . import timeline as TL

MAP_W, MAP_H = 1536, 1608


def _L(name, x, y, kind):
    return dict(name=name, nx=x, ny=y, x=int(round(x * MAP_W)), y=int(round(y * MAP_H)), kind=kind)


# (x, y) = ground anchor of the place's icon (nudged from the dossier so icons and coasts fit)
LOCS = {
    # --- the 20 tour stops (TL.TOUR) ---
    'canterlot': _L('Canterlot', 0.411, 0.288, 'capital'),
    'cloudsdale': _L('Cloudsdale', 0.318, 0.232, 'city'),
    'ponyville': _L('Ponyville', 0.367, 0.331, 'town'),
    'sweet_apple_acres': _L('Sweet Apple Acres', 0.349, 0.347, 'landmark'),
    'everfree': _L('Everfree Forest', 0.372, 0.372, 'forest'),
    'neighagra_falls': _L('Neighagra Falls', 0.462, 0.236, 'natural'),
    'rainbow_falls': _L('Rainbow Falls', 0.62, 0.146, 'town'),
    'manehattan': _L('Manehattan', 0.711, 0.266, 'city'),
    'baltimare': _L('Baltimare', 0.664, 0.372, 'city'),
    'griffonstone': _L('Griffonstone', 0.893, 0.345, 'foreign'),
    'dragon_lands': _L('Dragon Lands', 0.905, 0.545, 'foreign'),
    'mount_aris': _L('Mount Aris', 0.717, 0.893, 'foreign'),
    'klugetown': _L('Klugetown', 0.45, 0.695, 'foreign'),
    'changeling_kingdom': _L('Changeling Kingdom', 0.535, 0.522, 'foreign'),
    'appleloosa': _L('Appleloosa', 0.4, 0.445, 'town'),
    'las_pegasus': _L('Las Pegasus', 0.158, 0.428, 'city'),
    'vanhoover': _L('Vanhoover', 0.118, 0.226, 'city'),
    'crystal_empire': _L('Crystal Empire', 0.4, 0.127, 'foreign'),
    'yakyakistan': _L('Yakyakistan', 0.212, 0.083, 'foreign'),
    'castle_two_sisters': _L('Castle of the Two Sisters', 0.42, 0.386, 'landmark'),
    # --- other places shown/labelled on the map ---
    'mt_canterhorn': _L('Mt. Canterhorn', 0.432, 0.24, 'natural'),
    'galloping_gorge': _L('Galloping Gorge', 0.255, 0.218, 'natural'),
    'castle_friendship': _L('Castle of Friendship', 0.33, 0.305, 'landmark'),
    'school_friendship': _L('School of Friendship', 0.346, 0.302, 'landmark'),
    'saddle_lake': _L('Saddle Lake', 0.42, 0.335, 'natural'),
    'white_tail_woods': _L('White Tail Woods', 0.222, 0.367, 'forest'),
    'zecoras_hut': _L("Zecora's Hut", 0.392, 0.36, 'landmark'),
    'froggy_bottom_bogg': _L('Froggy Bottom Bogg', 0.38, 0.4, 'swamp'),
    'tree_of_harmony': _L('Tree of Harmony', 0.438, 0.393, 'landmark'),
    'foal_mountain': _L('Foal Mountain', 0.522, 0.29, 'natural'),
    'hollow_shades': _L('Hollow Shades', 0.575, 0.276, 'village'),
    'rambling_rock_ridge': _L('Rambling Rock Ridge', 0.48, 0.352, 'natural'),
    'our_town': _L('Our Town', 0.6, 0.218, 'village'),
    'starlights_cave': _L("Starlight's Cave", 0.758, 0.168, 'landmark'),
    'crystal_mountains': _L('Crystal Mountains', 0.53, 0.18, 'natural'),
    'fillydelphia': _L('Fillydelphia', 0.628, 0.312, 'city'),
    'horseshoe_bay': _L('Horseshoe Bay', 0.68, 0.408, 'natural'),
    'hayseed_swamps': _L('Hayseed Swamps', 0.69, 0.462, 'swamp'),
    'trottingham': _L('Trottingham', 0.813, 0.24, 'town'),
    'griffonstone_station': _L('Griffonstone Station', 0.918, 0.222, 'landmark'),
    'bug_bear_territory': _L('Bug Bear Territory', 0.87, 0.1, 'region'),
    'kirin_grove': _L('Kirin Grove', 0.935, 0.742, 'foreign'),
    'lost_lagoon': _L('Lost Lagoon', 0.862, 0.752, 'natural'),
    'seaquestria': _L('Seaquestria', 0.717, 0.927, 'foreign'),
    'basalt_beach': _L('Basalt Beach', 0.56, 0.857, 'natural'),
    'storm_kings_realm': _L("Storm King's Realm", 0.235, 0.9, 'foreign'),
    'bone_dry_desert': _L('Bone Dry Desert', 0.53, 0.645, 'desert'),
    'sea_of_clouds': _L('Sea of Clouds', 0.47, 0.735, 'natural'),
    'caves_of_conundrum': _L('Caves of Conundrum', 0.3, 0.67, 'natural'),
    'badlands': _L('Badlands', 0.52, 0.49, 'region'),
    'tartarus': _L('Tartarus', 0.405, 0.552, 'landmark'),
    'forbidden_jungle': _L('Forbidden Jungle', 0.6, 0.585, 'forest'),
    'dodge_junction': _L('Dodge Junction', 0.49, 0.418, 'town'),
    'macintosh_hills': _L('Macintosh Hills', 0.385, 0.49, 'natural'),
    'rock_farm': _L('Rock Farm', 0.278, 0.468, 'landmark'),
    'ghastly_gorge': _L('Ghastly Gorge', 0.322, 0.428, 'natural'),
    'applewood': _L('Applewood', 0.205, 0.43, 'town'),
    'san_palomino': _L('San Palomino Desert', 0.155, 0.485, 'desert'),
    'somnambula': _L('Somnambula', 0.19, 0.484, 'village'),
    'tall_tale': _L('Tall Tale', 0.13, 0.252, 'town'),
    'smokey_mountains': _L('Smokey Mountains', 0.14, 0.31, 'natural'),
    'silver_shoals': _L('Silver Shoals', 0.05, 0.303, 'town'),
    'luna_bay': _L('Luna Bay', 0.07, 0.165, 'natural'),
    'hope_hollow': _L('Hope Hollow', 0.08, 0.372, 'village'),
    'unicorn_range': _L('Unicorn Range', 0.19, 0.25, 'natural'),
    'frozen_north': _L('Frozen North', 0.18, 0.175, 'region'),
    'mt_everhoof': _L('Mt. Everhoof', 0.41, 0.03, 'natural'),
    'yaket_range': _L('Yaket Range', 0.61, 0.1, 'natural'),
    'arimaspi': _L('Arimaspi Territory', 0.318, 0.585, 'natural'),
    'farthest_reaches': _L('Farthest Reaches', 0.235, 0.604, 'town'),
    'nightmare_cliffs': _L('Nightmare Cliffs', 0.15, 0.725, 'natural'),
    'black_skull_island': _L('Black Skull Island', 0.71, 0.724, 'natural'),
    'slug_troll_swamp': _L('Slug Troll Swamp', 0.41, 0.815, 'swamp'),
    'pine_needle_barrens': _L('Pine Needle Barrens', 0.57, 0.81, 'natural'),
    'iceberg_barrier': _L('Great Iceberg Barrier', 0.55, 0.97, 'region'),
    'north_luna_ocean': _L('North Luna Ocean', 0.04, 0.23, 'sea'),
    'south_luna_ocean': _L('South Luna Ocean', 0.05, 0.66, 'sea'),
    'celestial_sea': _L('Celestial Sea', 0.76, 0.38, 'sea'),
}


def loc_xy(lid):
    L = LOCS[lid]
    return L['x'], L['y']


# ====================================================================== static data
def _icon(lid):
    from . import worldmap_icons as WI
    return WI.icon(lid)


def _clearings():
    out = []
    for lid in LOCS:
        ic = _icon(lid)
        if ic is None:
            continue
        x, y = loc_xy(lid)
        out.append((x - ic.ax + ic.w // 2, y - ic.h * 0.3, ic.w // 2 + 3, ic.h * 0.35 + 3))
    return tuple((int(a), int(b), int(c), int(d)) for a, b, c, d in out)


@lru_cache(maxsize=1)
def _static():
    from . import worldmap_build as WB
    r = WB.load_or_build(_clearings())
    tex = r['rgb']
    # mip pyramid for Mode 7 (box filtered)
    mips = [tex]
    a = tex.astype(np.float32)
    for _ in range(4):
        h2, w2 = a.shape[0] // 2, a.shape[1] // 2
        a = a[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 3).mean(axis=(1, 3))
        mips.append(np.clip(a + 0.5, 0, 255).astype(np.uint8))
    rg = np.random.default_rng(99)
    phase = rg.integers(0, 256, (64, 64)).astype(np.uint8)
    from scipy import ndimage
    ph = ndimage.zoom(phase.astype(np.float32), 4, order=1)
    ph = np.tile(ph, (MAP_H // ph.shape[0] + 1, MAP_W // ph.shape[1] + 1))[:MAP_H, :MAP_W]
    return dict(tex=tex, mips=mips, wdist=r['wdist'], cls=r['cls'], peaks=r['peaks'], glint=r['glint'],
                forest=r['forest'], phase=(ph / 255.0 * 2 * math.pi).astype(np.float32))


# ---------------------------------------------------------------- lines (rails, sea lanes, route)
@lru_cache(maxsize=1)
def _lines():
    from . import worldmap_build as WB
    from . import worldmap_data as WD
    rails = []
    for pts in WD.RAILS:
        clean = [(p[0], p[1]) for p in pts]
        line, ts = WB.polyline_px(clean, 1.0)
        tun = np.zeros(len(line), bool)
        cps = [WB.P(p) for p in clean]
        cidx = [int(np.argmin(np.hypot(line[:, 0] - cx, line[:, 1] - cy))) for cx, cy in cps]
        for k, p in enumerate(pts):
            if len(p) > 2 and p[2] == 'T':
                tun[cidx[k]:cidx[k + 1] + 1] = True
        rails.append((line, tun))
    lanes = [WB.polyline_px(p, 1.0)[0] for p in WD.SEA_LANES]
    return rails, lanes


def _bezier_leg(i):
    """Control points of route leg i (TOUR[i] -> TOUR[i+1]) as a gentle quadratic arc."""
    a = np.array(loc_xy(TL.TOUR[i]), float)
    b = np.array(loc_xy(TL.TOUR[i + 1]), float)
    return _arc(a, b, 1 if i % 2 == 0 else -1)


def _arc(a, b, side):
    d = b - a
    L = float(np.hypot(*d)) + 1e-6
    nrm = np.array([-d[1], d[0]]) / L
    c = (a + b) / 2 + nrm * side * min(0.16 * L, 70)
    return a, c, b


def _bez(p, t):
    a, c, b = p
    t = np.asarray(t, float)[..., None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t * t * b


def _bez_tan(p, t):
    a, c, b = p
    return 2 * (1 - t) * (c - a) + 2 * t * (b - c)


@lru_cache(maxsize=None)
def _leg_dots(i, spacing=5.0):
    p = _bezier_leg(i)
    ts = np.linspace(0, 1, 400)
    pts = _bez(p, ts)
    sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    n = max(2, int(sl[-1] / spacing))
    s = np.linspace(0, sl[-1], n)
    tt = np.interp(s, sl, ts)
    return tt, _bez(p, tt)


# ====================================================================== colour grading
_GRADE_KEYS = [
    (0.0, (0.30, 0.34, 0.58), (6, 8, 26)),
    (4.0, (0.33, 0.36, 0.62), (8, 8, 30)),
    (5.0, (0.48, 0.45, 0.70), (22, 10, 36)),
    (5.75, (0.78, 0.62, 0.72), (40, 16, 22)),
    (6.5, (1.0, 0.82, 0.70), (30, 14, 0)),
    (7.5, (1.0, 0.95, 0.88), (8, 4, 0)),
    (9.0, (1.0, 1.0, 1.0), (0, 0, 0)),
    (15.5, (1.0, 1.0, 1.0), (0, 0, 0)),
    (16.5, (1.02, 0.96, 0.86), (8, 4, 0)),
    (17.5, (1.05, 0.86, 0.66), (22, 8, 0)),
    (18.5, (0.95, 0.68, 0.62), (24, 6, 18)),
    (19.25, (0.62, 0.50, 0.72), (16, 8, 34)),
    (20.25, (0.36, 0.38, 0.64), (8, 8, 30)),
    (24.0, (0.30, 0.34, 0.58), (6, 8, 26)),
]


def grade_params(hour):
    """(mul (r,g,b), add (r,g,b), night 0..1) for a clock hour 0..24."""
    h = hour % 24.0 if hour != 24.0 else 24.0
    for (h0, m0, a0), (h1, m1, a1) in zip(_GRADE_KEYS, _GRADE_KEYS[1:]):
        if h <= h1:
            t = smooth(seg(h, h0, h1))
            mul = tuple(lerp(x, y, t) for x, y in zip(m0, m1))
            add = tuple(lerp(x, y, t) for x, y in zip(a0, a1))
            break
    night = 1.0 - smooth(seg(h, 5.0, 6.6)) if h < 12 else smooth(seg(h, 18.6, 20.2))
    return mul, add, night


def _apply_grade(a, hour, amount=1.0):
    mul, add, night = grade_params(hour)
    if amount != 1.0:
        mul = tuple(lerp(1, m, amount) for m in mul)
        add = tuple(x * amount for x in add)
    if mul == (1.0, 1.0, 1.0) and add == (0, 0, 0):
        return
    f = a.astype(np.float32)
    if night > 0.05:
        luma = f @ np.array([0.3, 0.55, 0.15], np.float32)
        f += (luma[..., None] - f) * (0.3 * night * amount)
    f = f * np.array(mul, np.float32) + np.array(add, np.float32)
    a[...] = np.clip(f, 0, 255).astype(np.uint8)


# ====================================================================== found-state helpers
def _found_set(found):
    n = len(TL.TOUR) if found is None else int(found)
    return set(TL.TOUR[:max(0, n)])


# ====================================================================== labels
_PLATE_KINDS = ('capital', 'city', 'town', 'village', 'landmark', 'foreign')


def _plate_sprite(label, state):
    return _plate_cached(label, state)


@lru_cache(maxsize=None)
def _plate_cached(label, state):
    """Label plate RGBA: tiny caps on a dark plate with a 1px rim. state: normal|found|unfound|lit|dest."""
    tw = text_width(label, 'tiny')
    w, h = tw + 6, 9
    c = Canvas(w, h, (1, 2, 3))
    rim, fill, ink = {
        'normal': ('#9babb2', '#2e222f', '#ffffff'),
        'found': ('#f9c22b', '#3a2440', '#fbff86'),
        'unfound': ('#62556a', '#2a2230', '#9a8ea8'),
        'lit': ('#ffffff', '#4a2a50', '#ffffff'),
        'dim': ('#4a4052', '#262028', '#7a7088'),
    }[state]
    c.round_rect(0, 0, w, h, rim, r=1)
    c.rect(1, 1, w - 2, h - 2, fill)
    c.blend(1, 1, w - 2, 1, 'white', 0.12)
    text(c, 3, 2, label, ink, font='tiny')
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = c.a
    a[..., 3] = np.where((c.a == np.array((1, 2, 3), np.uint8)).all(-1), 0, 255)
    return a


@lru_cache(maxsize=None)
def _soft_label(label, color, outline, font='tiny', spacing=None):
    """Outlined text (natural features / regions / seas) as RGBA."""
    tw = text_width(label, font, spacing=spacing)
    hh = 7 if font == 'tiny' else 10
    c = Canvas(tw + 2, hh, (1, 2, 3))
    text(c, 1, 1, label, color, font=font, outline=outline, spacing=spacing)
    a = np.zeros(c.a.shape[:2] + (4,), np.uint8)
    a[..., :3] = c.a
    a[..., 3] = np.where((c.a == np.array((1, 2, 3), np.uint8)).all(-1), 0, 255)
    return a


@lru_cache(maxsize=1)
def _lettering_lids():
    """name(upper) -> lid for places drawn as big region / sea lettering instead of a small label."""
    from . import worldmap_data as WD
    names = {t[0] for t in WD.REGION_LABELS} | {t[0] for t in WD.SEA_LABELS}
    return {L['name'].upper(): lid for lid, L in LOCS.items() if L['name'].upper() in names}


@lru_cache(maxsize=1)
def _label_layout():
    """Greedy collision-free placement of every label in world px. -> dict lid -> (x, y, kind)."""
    from . import worldmap_icons as WI
    boxes = []
    for lid in LOCS:
        ic = WI.icon(lid)
        if ic is not None:
            x, y = loc_xy(lid)
            boxes.append((x - ic.ax, y - ic.ay, ic.w, ic.h))

    def hit(b, boxes_, own=None):
        x, y, w, h = b
        for bb in boxes_:
            if bb is own:
                continue
            if x < bb[0] + bb[2] and bb[0] < x + w and y < bb[1] + bb[3] and bb[1] < y + h:
                return True
        return False
    placed = []
    out = {}
    order = list(TL.TOUR) + [k for k in LOCS if k not in TL.TOUR]
    for lid in order:
        L = LOCS[lid]
        if L['kind'] in ('sea',) or L['name'].upper() in _lettering_lids():
            continue
        ic = WI.icon(lid)
        x, y = loc_xy(lid)
        label = L['name'].upper()
        if ic is not None:
            spr = _plate_cached(label, 'normal')
            w, h = spr.shape[1], spr.shape[0]
            x0i, y0i = x - ic.ax, y - ic.ay
            own = (x0i, y0i, ic.w, ic.h)
            cands = [(x - w // 2, y + 2), (x - w // 2, y0i - h - 1), (x0i + ic.w + 1, y0i + ic.h // 2 - 4),
                     (x0i - w - 1, y0i + ic.h // 2 - 4), (x - w // 2, y + 11), (x - w // 2, y0i - h - 10)]
        else:
            if L['kind'] in ('region',) and lid in ('frozen_north', 'badlands', 'iceberg_barrier',
                                                   'bug_bear_territory'):
                continue           # drawn as big region lettering instead
            spr = _soft_label(label, 'white', '#2e222f')
            w, h = spr.shape[1], spr.shape[0]
            own = None
            cands = [(x - w // 2, y - h // 2), (x - w // 2, y + 6), (x - w // 2, y - h - 6),
                     (x + 6, y - h // 2), (x - w - 6, y - h // 2)]
        chosen = cands[0]
        for cnd in cands:
            b = (cnd[0], cnd[1], w, h)
            if not hit(b, placed) and not hit(b, boxes, own if own is None else None) or False:
                chosen = cnd
                break
            if not hit(b, placed) and not _hits_other_icons(b, boxes, own):
                chosen = cnd
                break
        b = (chosen[0], chosen[1], w, h)
        placed.append(b)
        out[lid] = (int(chosen[0]), int(chosen[1]), 'plate' if ic is not None else 'soft')
    return out


def _hits_other_icons(b, boxes, own):
    x, y, w, h = b
    for bb in boxes:
        if own is not None and bb == own:
            continue
        if x < bb[0] + bb[2] and bb[0] < x + w and y < bb[1] + bb[3] and bb[1] < y + h:
            return True
    return False


def _blit_rgba(dst, spr, x, y, alpha_mul=None):
    h, w = spr.shape[:2]
    x, y = int(x), int(y)
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = spr[Y0 - y:Y1 - y, X0 - x:X1 - x]
    m = sub[:, :, 3] > 0
    if alpha_mul is not None and alpha_mul < 0.999:
        if alpha_mul <= 0.001:
            return
        yy = (np.arange(Y0, Y1) % 4)[:, None]
        xx = (np.arange(X0, X1) % 4)[None, :]
        m &= _BAYER4[yy, xx] < alpha_mul
    dst[Y0:Y1, X0:X1][m] = sub[:, :, :3][m]


_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16


def _draw_labels(a, x0, y0, zoom, found_set, label_fx=None, which=None, pos_fn=None, alpha=1.0):
    """Draw all labels whose world position is inside the view. label_fx: lid -> state override.
    pos_fn(wx, wy) -> (sx, sy, k) | None projects world points (perspective); labels need k >= 0.75.
    alpha < 1 fades them with an ordered dither."""
    from . import worldmap_data as WD
    if alpha <= 0.001:
        return
    vw, vh = a.shape[1], a.shape[0]
    if pos_fn is None:
        def pos_fn(wx, wy):
            return (wx - x0) * zoom, (wy - y0) * zoom, 1.0

    def place(wx, wy):
        p = pos_fn(wx, wy)
        if p is None or p[2] < 0.75:
            return None
        return p[0], p[1]
    # big region lettering and seas first
    lets = _lettering_lids()
    for txt, nx, ny, colr, sp, font in WD.REGION_LABELS:
        st = None if label_fx is None else label_fx.get(lets.get(txt))
        if st == 'hide':
            continue
        colr = '#8a8098' if st == 'dim' else '#fbff86' if st == 'lit' else colr
        spr = _soft_label(txt, colr, '#3a2f4a' if st != 'lit' else '#5a2a50', font, sp)
        q = place(nx * MAP_W, ny * MAP_H)
        if q is None:
            continue
        sx, sy = int(q[0]) - spr.shape[1] // 2, int(q[1]) - spr.shape[0] // 2
        if -spr.shape[1] < sx < vw and -12 < sy < vh:
            _blit_rgba(a, spr, sx, sy, alpha)
    for txt, nx, ny, sp in WD.SEA_LABELS:
        st = None if label_fx is None else label_fx.get(lets.get(txt))
        cc = '#8a9ab8' if st == 'dim' else '#fbff86' if st == 'lit' else '#c8f0ff'
        spr = _soft_label(txt, cc, '#2a4a8a', 'normal', sp)
        q = place(nx * MAP_W, ny * MAP_H)
        if q is None:
            continue
        sx, sy = int(q[0]) - spr.shape[1] // 2, int(q[1]) - spr.shape[0] // 2
        if -spr.shape[1] < sx < vw and -12 < sy < vh:
            _blit_rgba(a, spr, sx, sy, alpha)
    lay = _label_layout()
    for lid, (lx, ly, kind) in lay.items():
        if which is not None and lid not in which:
            continue
        wx, wy = loc_xy(lid)
        q = place(wx, wy)
        if q is None:
            continue
        sx = int(round(q[0])) + (lx - wx)
        sy = int(round(q[1])) + (ly - wy)
        if sx > vw or sy > vh or sx < -130 or sy < -12:
            continue
        label = LOCS[lid]['name'].upper()
        st = None if label_fx is None else label_fx.get(lid)
        if st == 'hide':
            continue
        if kind == 'plate':
            if st is None:
                st = ('found' if lid in found_set else 'unfound') if lid in TL.TOUR else 'normal'
            spr = _plate_cached(label, st)
        else:
            if st == 'dim':
                spr = _soft_label(label, '#8a8098', '#2e222f')
            elif st == 'lit':
                spr = _soft_label(label, '#fbff86', '#5a2a50')
            else:
                spr = _soft_label(label, '#ffffff' if LOCS[lid]['kind'] != 'swamp' else '#e8ffd8', '#2e222f')
        _blit_rgba(a, spr, sx, sy, alpha)


def _draw_flags(a, f, found_set, icon_fx=None, pos_fn=None, alpha=1.0):
    """Quest flags on the 20 tour stops (top-right of each icon). pos_fn as in _draw_labels."""
    from . import worldmap_icons as WI
    if alpha <= 0.001:
        return
    for lid in TL.TOUR:
        ic = WI.icon(lid)
        x, y = loc_xy(lid)
        p = pos_fn(x, y)
        if p is None or p[2] < 0.75:
            continue
        fx_ = int(round(p[0])) - ic.ax + ic.w - 2
        fy_ = int(round(p[1])) - ic.ay + 6
        if not (-8 < fx_ < W + 8 and -2 < fy_ < H + 12):
            continue
        st = (icon_fx or {}).get(lid, {}).get('state')
        fnd = (lid in found_set) if st is None else st != 'grey'
        spr = _flag_sprite(fnd, (f // 6) % 2)
        _blit_rgba(a, spr, fx_, fy_ - 9, alpha)


@lru_cache(maxsize=4)
def _flag_sprite(found, wave):
    a = np.zeros((11, 9, 3), np.uint8)
    a[:] = (1, 2, 3)
    _flag(a, 0, 9, found, wave * 6)
    out = np.zeros((11, 9, 4), np.uint8)
    out[..., :3] = a
    out[..., 3] = np.where((a == np.array((1, 2, 3), np.uint8)).all(-1), 0, 255)
    return out


# ---------------------------------------------------------------- quest flags (tour stops)
def _flag(a, x, y, found, f):
    """Little quest flag: gold star (discovered) or grey '?' (undiscovered). (x, y) = pole foot."""
    x, y = int(x), int(y)
    pole = '#5a4a48'
    for k in range(9):
        _px(a, x, y - k, pole)
    wave = (f // 6) % 2
    if found:
        cols_ = ('#f9c22b', '#f7961d')
        for dy in range(5):
            for dx in range(6):
                _px(a, x + 1 + dx, y - 8 + dy + (1 if (dx > 3 and wave) else 0), cols_[0] if dx < 4 else cols_[1])
        for dx, dy in ((3, 1), (2, 2), (3, 2), (4, 2), (3, 3)):
            _px(a, x + 1 + dx - 1, y - 8 + dy, '#fff6c0')
        _px(a, x, y - 9, '#fbff86')
    else:
        for dy in range(5):
            for dx in range(6):
                _px(a, x + 1 + dx, y - 8 + dy + (1 if (dx > 3 and wave) else 0), '#7a7088' if dx < 4 else '#62556a')
        for dx, dy in ((2, 0), (3, 0), (4, 1), (3, 2), (3, 4)):
            _px(a, x + dx, y - 8 + dy, '#ffffff')


def _px(a, x, y, c):
    if 0 <= x < a.shape[1] and 0 <= y < a.shape[0]:
        a[y, x] = col(c)


# ====================================================================== world-layer rendering
_DEEP = np.array(col('#2a539a'), np.uint8)


def _crop(arr, x0, y0, w, h, fill):
    out = np.empty((h, w) + arr.shape[2:], arr.dtype)
    out[...] = fill
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(arr.shape[1], x0 + w), min(arr.shape[0], y0 + h)
    if X1 > X0 and Y1 > Y0:
        out[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = arr[Y0:Y1, X0:X1]
    return out


def _water_anim(a, x0, y0, f):
    S = _static()
    h, w = a.shape[:2]
    wd = _crop(S['wdist'], x0, y0, w, h, 255)
    ph = _crop(S['phase'], x0, y0, w, h, 0.0)
    lap = 1.0 + 2.2 * (0.5 + 0.5 * np.sin(f * (2 * math.pi / 54.0) + ph))
    m1 = (wd >= 1) & (wd <= 1.5)
    m2 = (wd >= 2) & (wd <= lap)
    a[m2] = col('#c8f0fa')
    a[m1 & (lap > 1.6)] = col('#eafcff')
    # glints
    g = S['glint']
    sel = (g[:, 0] >= x0) & (g[:, 0] < x0 + w - 2) & (g[:, 1] >= y0 + 1) & (g[:, 1] < y0 + h)
    gg = g[sel]
    if len(gg):
        on = ((f + gg[:, 2]) % 84) < 12
        gg = gg[on]
        ph2 = ((f + gg[:, 2]) % 84)
        xs = gg[:, 0] - x0
        ys = gg[:, 1] - y0
        a[ys, xs + 1] = col('#ffffff')
        big = (ph2 >= 3) & (ph2 < 9)
        a[ys[big], xs[big]] = col('#c8f0ff')
        a[ys[big], xs[big] + 2] = col('#c8f0ff')


# trains: (rail index, speed px/frame, phase offset)
_TRAINS = [(0, 0.55, 0), (1, 0.6, 300), (2, 0.5, 90), (4, 0.5, 40), (5, 0.55, 200), (8, 0.6, 60), (9, 0.5, 30),
           (1, 0.6, 900)]
_TRAIN_COLS = ('#f04f78', '#a884f3', '#bfe3ff', '#f04f78')


def _trains(a, x0, y0, f, L=None):
    rails, _ = _lines()
    h, w = a.shape[:2]
    for k, (ri, sp, off) in enumerate(_TRAINS):
        line, tun = rails[ri]
        n = len(line)
        if n < 20:
            continue
        period = 2 * (n - 1)
        s = (off + f * sp) % period
        fwd = s < n - 1
        pos = s if fwd else period - s
        for car in range(3):
            q = pos - car * 4 if fwd else pos + car * 4
            qi = int(clamp(q, 0, n - 1))
            if tun[qi]:
                continue
            px_, py_ = line[qi]
            sx, sy = int(px_ - x0), int(py_ - y0)
            if not (-4 < sx < w + 4 and -4 < sy < h + 4):
                continue
            c0 = _TRAIN_COLS[min(car, 3)]
            for dy in (-1, 0):
                for dx in (-1, 0, 1):
                    _px(a, sx + dx, sy + dy, c0 if (dx, dy) != (-1, -1) else '#ffffff')
            _px(a, sx - 1, sy + 1, '#2e222f')
            _px(a, sx, sy + 1, '#2e222f')
            _px(a, sx + 1, sy + 1, '#2e222f')
            if car == 0 and L is not None and 0 <= sx < w and 0 <= sy - 1 < h:
                L[sy - 1, sx, :3] = col('#fff0a0')
                L[sy - 1, sx, 3] = 255
        # smoke puffs from the loco
        qi = int(clamp(pos, 0, n - 1))
        if tun[qi]:
            continue
        px_, py_ = line[qi]
        for j in range(4):
            age = (f + j * 5 + k * 3) % 20
            back = -age * 0.5 if fwd else age * 0.5
            bi = int(clamp(pos + back, 0, n - 1))
            bx, by = line[bi]
            sx, sy = int(bx - x0), int(by - y0 - 3 - age // 3)
            c = '#ffffff' if age < 8 else '#dfe8ee' if age < 14 else '#b8c4cc'
            _px(a, sx, sy, c)
            if age < 12:
                _px(a, sx + 1, sy, c)
                _px(a, sx, sy - 1, c)


_SHIP = np.array([
    '....p..',
    '...ws..',
    '..wws..',
    '.wwws..',
    '..wws..',
    'bbbbbbb',
    '.BBBBB.',
], dtype=object)
_SHIP_COL = {'p': '#f04f78', 'w': '#ffffff', 's': '#c8d4e8', 'b': '#9e4539', 'B': '#6e2727'}


@lru_cache(maxsize=2)
def _ship_sprite(flip):
    h, w = len(_SHIP), len(_SHIP[0])
    a = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(_SHIP):
        for x, ch in enumerate(row):
            if ch != '.':
                a[y, x, :3] = col(_SHIP_COL[ch])
                a[y, x, 3] = 255
    return a[:, ::-1].copy() if flip else a


_SHIPS = [(0, 0.18, 0), (0, 0.18, 900), (1, 0.12, 50), (2, 0.2, 100), (2, 0.2, 1200), (3, 0.15, 200),
          (4, 0.08, 0), (5, 0.14, 120)]


def _ships(a, x0, y0, f):
    _, lanes = _lines()
    h, w = a.shape[:2]
    for li, sp, off in _SHIPS:
        line = lanes[li]
        n = len(line)
        period = 2 * (n - 1)
        s = (off + f * sp) % period
        fwd = s < n - 1
        pos = s if fwd else period - s
        i = int(pos)
        x, y = line[min(n - 1, i)]
        j = min(n - 1, i + 3)
        dx = line[j][0] - line[max(0, i - 3)][0]
        right = (dx >= 0) == fwd
        sx, sy = int(x - x0), int(y - y0)
        if not (-10 < sx < w + 10 and -10 < sy < h + 10):
            continue
        spr = _ship_sprite(not right)
        bob = 1 if (f // 20 + li) % 2 else 0
        _blit_rgba(a, spr, sx - 3, sy - 6 + bob)
        # wake
        for k in range(1, 4):
            wx = sx + (-3 - k * 2 if right else 3 + k * 2)
            if (f // 4 + k) % 3:
                _px(a, wx, sy + 1, '#d8f6ff')


# clouds: (x, y, size seed)
@lru_cache(maxsize=1)
def _cloud_defs():
    rg = np.random.default_rng(5)
    out = []
    for k in range(16):
        out.append((rg.uniform(0, MAP_W + 200), rg.uniform(0.05, 0.95) * MAP_H, int(rg.integers(0, 6)),
                    rg.uniform(0.1, 0.22)))
    return out


@lru_cache(maxsize=None)
def _cloud_sprite(v):
    from . import worldmap_art as A
    rg = np.random.default_rng(v + 40)
    w, h = 44 + v * 4, 22
    a = np.zeros((h, w, 4), np.uint8)
    ramp = ('#ffffff', '#f2f6ff', '#d8e2f6', '#b6c2e8', '#9aa8d8')
    for k in range(6 + v):
        r = int(rg.integers(5, 9))
        p = A.puff_sprite(r, ramp, int(rg.integers(0, 99)))
        px_ = int(rg.uniform(2, w - p.shape[1] - 2))
        py_ = int(rg.uniform(1, h - p.shape[0] - 1))
        m = p[:, :, 3] > 0
        a[py_:py_ + p.shape[0], px_:px_ + p.shape[1]][m] = p[m]
    return a


def _cloud_positions(f):
    out = []
    span = MAP_W + 260
    for k, (cx, cy, v, sp) in enumerate(_cloud_defs()):
        x = (cx + f * sp) % span - 130
        out.append((x, cy, v))
    return out


def _clouds(a, x0, y0, f, avoid=None, shadows=True, tops=True):
    h, w = a.shape[:2]
    for (x, y, v) in _cloud_positions(f):
        spr = _cloud_sprite(v)
        sh, sw = spr.shape[:2]
        if avoid is not None:
            ax_, ay_, ar = avoid
            if abs(x - ax_) < ar + sw / 2 and abs(y - ay_) < ar + sh:
                continue
        sx, sy = int(x - x0 - sw // 2), int(y - y0 - sh)
        if shadows:
            ssx, ssy = sx + 10, sy + 26
            X0, Y0 = max(0, ssx), max(0, ssy)
            X1, Y1 = min(w, ssx + sw), min(h, ssy + sh)
            if X1 > X0 and Y1 > Y0:
                m = spr[Y0 - ssy:Y1 - ssy, X0 - ssx:X1 - ssx, 3] > 0
                reg = a[Y0:Y1, X0:X1]
                reg[m] = (reg[m].astype(np.uint16) * 3 // 4).astype(np.uint8)
        if tops:
            _blit_rgba(a, spr, sx, sy)


# ---------------------------------------------------------------- icon life (smoke, sparkle...)
def _icon_anim(a, L, lid, sx, sy, ic, f, night):
    """Animated overlays for an icon whose anchor is at screen (sx, sy)."""
    top = sy - ic.ay
    left = sx - ic.ax
    if lid == 'dragon_lands':
        _smoke(a, sx, top + 6, f, 7, '#8a7a7a', '#5a4a4a', big=True)
        if (f // 4) % 3 == 0:
            _px(a, sx + (f // 4) % 5 - 2, top + 8, '#ffd04a')
    elif lid == 'klugetown':
        for k, (dx, dy) in enumerate(((11 - 17, 9 - 39), (24 - 17, 12 - 39), (15 - 17, 5 - 39))):
            _smoke(a, sx + dx, sy + dy, f + k * 7, 3, '#d8c8b8', '#9a8a7a')
    elif lid == 'yakyakistan':
        _smoke(a, sx - 7, top + 4, f, 3, '#e8eef4', '#b8c4cc')
    elif lid == 'cloudsdale':
        rain = ('#ff5a5a', '#ffa54a', '#ffe15a', '#6bd66b', '#5ab4ff', '#a275e6')
        for x0r in (8, 20, 34):
            yy = top + 26 + (f // 2 + x0r) % 18
            _px(a, left + x0r + 1 + (f // 3) % 6, yy, '#ffffff')
    elif lid == 'crystal_empire':
        for k in range(4):
            t = (f + k * 11) % 44
            if t < 8:
                x = left + 6 + int(hash01(k, (f + k * 11) // 44) * (ic.w - 12))
                y = top + 8 + int(hash01(k, 7, (f + k * 11) // 44) * (ic.h - 16))
                c = '#ffffff' if t < 4 else '#bfe8ff'
                _px(a, x, y, c)
                if t < 4:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        _px(a, x + dx, y + dy, '#bfe8ff')
    elif lid in ('neighagra_falls', 'canterlot', 'rainbow_falls'):
        n = 10 if lid == 'neighagra_falls' else 5
        for k in range(n):
            t = (f + k * 7) % 21
            if lid == 'neighagra_falls':
                x = left + 8 + (k * 13) % 26
                y = top + 12 + t // 2
            elif lid == 'canterlot':
                x = left + (13, 22, 32)[k % 3]
                y = top + 39 + t // 2
            else:
                x = left + (8, 17, 26)[k % 3]
                y = top + 15 + t
            _px(a, x, y, '#ffffff')
        if lid == 'neighagra_falls':
            for k in range(5):
                t = (f + k * 9) % 36
                _px(a, left + 6 + k * 7 + t // 9, top + 24 - t // 6, '#ffffff' if t < 18 else '#d8f6ff')
    elif lid == 'storm_kings_realm':
        cx = left + ic.w // 2
        for k, (dx, dy, rx) in enumerate(((-6, -2, 8), (4, -4, 9), (12, -1, 6), (-14, 0, 5))):
            _ell(a, cx + dx + int(math.sin(f / 40 + k) * 2), top + dy, rx, 3, '#3c2e3e')
            _ell(a, cx + dx + int(math.sin(f / 40 + k) * 2) - 1, top + dy - 1, rx - 2, 1, '#4e3e52')
        if f % 37 < 4:
            bx = cx - 4 + (f // 37 % 3) * 5
            for i in range(8):
                _px(a, bx + (i % 3 == 1) - (i % 4 == 3), top + 2 + i, '#bff8ff' if f % 37 < 2 else '#4fe8ff')
            if L is not None:
                pass
    elif lid == 'las_pegasus' and night > 0.3:
        if (f // 9) % 2:
            for k in range(6):
                _px(a, left + 8 + k * 5, top + 6 + (k % 2) * 3, '#ff4fd8' if k % 2 else '#4ff6ff')
    elif lid == 'changeling_kingdom':
        if (f // 12) % 2:
            _px(a, left + 30, top + 26, '#c8ff9a')
    elif lid == 'mount_aris' or lid == 'seaquestria':
        if lid == 'seaquestria':
            for k in range(3):
                t = (f + k * 13) % 30
                _px(a, left + 8 + k * 7, top + 10 - t // 4, '#c8f0ff')
    if lid == 'canterlot':
        # banners flutter
        w_ = (f // 5) % 2
        _px(a, left + 29 + w_, top + 3, '#905ea9')
        _px(a, left + 22 + w_, top + 7, '#f04f78')


def _smoke(a, x, y, f, n, c1, c2, big=False):
    for k in range(n):
        age = (f * 0.5 + k * (30 / n)) % 30
        sx = int(x + math.sin(age / 5 + k) * (1 + age / 10) + age / 5)
        sy = int(y - age * (1.2 if big else 0.7))
        r = 1 + int(age / 10) + (1 if big else 0)
        c = c1 if age < 15 else c2
        _ell(a, sx, sy, r, max(1, r - 1), c)


def _ell(a, cx, cy, rx, ry, c):
    h, w = a.shape[:2]
    cc = col(c)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / (rx + 0.5)) ** 2 + (dy / (ry + 0.5)) ** 2 <= 1:
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    a[y, x] = cc


def _shade_ell(a, cx, cy, rx, ry, k=3):
    h, w = a.shape[:2]
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / (rx + 0.5)) ** 2 + (dy / (ry + 0.5)) ** 2 <= 1:
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    a[y, x] = (a[y, x].astype(np.uint16) * k // 4).astype(np.uint8)


def _scale_sprite(spr, k):
    if abs(k - 1) < 0.02:
        return spr
    h, w = spr.shape[:2]
    nh, nw = max(1, int(round(h * k))), max(1, int(round(w * k)))
    yi = (np.arange(nh) / k).astype(int).clip(0, h - 1)
    xi = (np.arange(nw) / k).astype(int).clip(0, w - 1)
    return spr[yi[:, None], xi[None, :]]


def _draw_icons(a, L, x0, y0, f, found_set, night, icon_fx=None, only=None):
    """Icons in view, y-sorted. icon_fx: lid -> dict(scale, state) overrides (arrival pop)."""
    from . import worldmap_icons as WI
    h, w = a.shape[:2]
    items = []
    for lid in LOCS:
        ic = WI.icon(lid)
        if ic is None:
            continue
        x, y = loc_xy(lid)
        sx, sy = x - x0, y - y0
        if sx + ic.w < -2 or sx - ic.w > w + 2 or sy - ic.h > h + 2 or sy + 30 < -2:
            continue
        items.append((y, lid, ic, sx, sy))
    items.sort(key=lambda t: t[0])
    for _, lid, ic, sx, sy in items:
        fx = (icon_fx or {}).get(lid, {})
        state = fx.get('state')
        if state is None:
            state = 'grey' if (lid in TL.TOUR and lid not in found_set) else 'color'
        spr = ic.grey if state == 'grey' else ic.white if state == 'white' else ic.rgba
        k = fx.get('scale', 1.0)
        lights = ic.lights
        if k != 1.0:
            spr = _scale_sprite(spr, k)
            lights = _scale_sprite(lights, k)
        ax, ay = int(round(ic.ax * k)), int(round(ic.ay * k))
        if lid in ('cloudsdale', 'las_pegasus'):
            _shade_ell(a, sx + 6, sy + 3, ic.w // 3, 3)
        _blit_rgba(a, spr, sx - ax, sy - ay)
        if L is not None and state == 'color':
            hh, ww = lights.shape[:2]
            X0, Y0 = max(0, sx - ax), max(0, sy - ay)
            X1, Y1 = min(w, sx - ax + ww), min(h, sy - ay + hh)
            if X1 > X0 and Y1 > Y0:
                sub = lights[Y0 - (sy - ay):Y1 - (sy - ay), X0 - (sx - ax):X1 - (sx - ax)]
                m = sub[:, :, 3] > 0
                L[Y0:Y1, X0:X1][m] = sub[m]
        if state == 'color' and k == 1.0:
            _icon_anim(a, L, lid, sx, sy, ic, f, night)


def _aurora(a, x0, y0, f, zoom, strength):
    """Aurora ribbons over the far north (screen-space blend), strength 0..1."""
    if strength <= 0.02:
        return
    h, w = a.shape[:2]
    ys = (np.arange(h) / zoom + y0)
    if ys[0] > 0.24 * MAP_H:
        return
    xs = np.arange(w) / zoom + x0
    for k, (base, col1, col2, amp) in enumerate(((0.07, '#5fffc0', '#30e1b9', 18), (0.13, '#a884f3', '#5fe0ff', 14),
                                                 (0.03, '#7fffd0', '#a884f3', 12))):
        cy = base * MAP_H + np.sin(xs * 0.012 + f * 0.035 + k * 2) * amp + np.sin(xs * 0.031 - f * 0.02) * 6
        inten = 0.5 + 0.5 * np.sin(xs * 0.05 + f * 0.06 + k)
        top = cy - 26
        yy = ys[:, None]
        band = (yy > top[None, :]) & (yy < cy[None, :])
        t = np.clip((yy - top[None, :]) / 26.0, 0, 1)
        alpha = (t ** 2) * inten[None, :] * strength * 0.55
        alpha = np.where(band, alpha, 0)
        if alpha.max() <= 0.01:
            continue
        cc = np.array(col(col1), np.float32)
        c2 = np.array(col(col2), np.float32)
        colr = cc[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
        # quantise the alpha into 3 dithered steps (pixel-art friendly)
        thr = ((np.arange(h)[:, None] * 3 + np.arange(w)[None, :] * 2) % 4 + 0.5) / 4.0
        q = np.floor(alpha * 4 + thr * 0.999) / 4.0
        f32 = a.astype(np.float32)
        a[...] = np.clip(f32 + (colr - f32) * q[..., None] * 0.9, 0, 255).astype(np.uint8)


def _render_world(x0, y0, vw, vh, f, hour, found_set, avoid=None, icon_fx=None, clouds=True):
    """World layer (terrain + life + icons), ungraded, plus the night-light layer."""
    S = _static()
    a = _crop(S['tex'], x0, y0, vw, vh, _DEEP)
    L = np.zeros((vh, vw, 4), np.uint8)
    _, _, night = grade_params(hour)
    _water_anim(a, x0, y0, f)
    _ships(a, x0, y0, f)
    _trains(a, x0, y0, f, L)
    if clouds:
        _clouds(a, x0, y0, f, avoid, shadows=True, tops=False)
    _draw_icons(a, L, x0, y0, f, found_set, night, icon_fx)
    bx, by = int(0.714 * MAP_W) - x0, int(0.699 * MAP_H) - y0      # Black Skull Island smoke
    if -20 < bx < vw + 20 and -40 < by < vh + 10:
        _smoke(a, bx, by, f, 5, '#7a6a6a', '#4a3e3e')
        if (f // 5) % 4 == 0:
            _px(a, bx, by + 1, '#ff551e')
    if clouds:
        _clouds(a, x0, y0, f, avoid, shadows=False, tops=True)
    return a, L, night


def _emissive(a, L, night):
    if night <= 0.02:
        return
    m = L[:, :, 3] > 0
    if not m.any():
        return
    f = a[m].astype(np.float32)
    a[m] = np.clip(f + (L[m][:, :3].astype(np.float32) - f) * min(1.0, night * 1.1), 0, 255).astype(np.uint8)


def _route_dots(a, x0, y0, route_upto, zoom=1, color='#f04f78', shadow='#6e2727'):
    if route_upto is None or route_upto <= 0:
        return
    n_legs = len(TL.TOUR) - 1
    r = min(float(route_upto), n_legs)
    full = int(r)
    for i in range(min(full + 1, n_legs)):
        frac = 1.0 if i < full else r - full
        if frac <= 0:
            continue
        tt, pts = _leg_dots(i)
        for t, (x, y) in zip(tt, pts):
            if t > frac:
                break
            sx = int(round((x - x0) * zoom))
            sy = int(round((y - y0) * zoom))
            if 0 <= sx < a.shape[1] - 1 and 0 <= sy < a.shape[0] - 1:
                a[sy + 1, sx:sx + 2] = col(shadow)
                a[sy, sx:sx + 2] = col(color)


def _route_partial(a, x0, y0, i, t_end, zoom=1):
    """Leg i drawn up to bezier parameter t_end (the trail behind the balloon)."""
    if i < 0 or t_end <= 0:
        return
    tt, pts = _leg_dots(i)
    for t, (x, y) in zip(tt, pts):
        if t > t_end:
            break
        sx = int(round((x - x0) * zoom))
        sy = int(round((y - y0) * zoom))
        if 0 <= sx < a.shape[1] - 1 and 0 <= sy < a.shape[0] - 1:
            a[sy + 1, sx:sx + 2] = col('#6e2727')
            a[sy, sx:sx + 2] = col('#f04f78')


def _balloon(c, sx, sy, f, size, alt):
    """Balloon with its ground shadow: (sx, sy) = shadow (ground) point on screen."""
    from .atlas import draw_balloon
    rx = {'xs': 2, 's': 3, 'm': 5, 'l': 7}[size]
    sx, sy = int(sx), int(sy)
    reg_x0, reg_x1 = sx - rx, sx + rx + 1
    if 0 <= sy < H:
        x0_, x1_ = max(0, reg_x0), min(W, reg_x1)
        if x1_ > x0_:
            c.a[sy, x0_:x1_] = (c.a[sy, x0_:x1_].astype(np.uint16) * 3 // 5).astype(np.uint8)
            if sy + 1 < H and x1_ - 1 > x0_ + 1:
                c.a[sy + 1, x0_ + 1:x1_ - 1] = (c.a[sy + 1, x0_ + 1:x1_ - 1].astype(np.uint16) * 3 // 5).astype(np.uint8)
    draw_balloon(c, sx, sy - alt, f, size=size)


# ====================================================================== public: top-down map
def draw_map(c, cx, cy, f, hour=12.0, found=None, zoom=1, labels=True, route_upto=None, balloon=None,
             avoid=None, icon_fx=None, label_fx=None, flags=True, trail=None, label_alpha=1.0):
    """Top-down view centred on world (cx, cy). zoom 1 or 2. balloon=(wx, wy, size[, alt]).

    Extra (optional): avoid=(wx, wy, r) keeps clouds off a spot; icon_fx / label_fx override icon /
    label states (used by the travel shots); flags=False hides the quest flags; trail=(leg, t) draws
    leg `leg` up to bezier parameter t.
    """
    zoom = 2 if zoom >= 2 else 1
    vw, vh = W // zoom, H // zoom
    x0 = int(round(cx - vw / 2))
    y0 = int(round(cy - vh / 2))
    fs = _found_set(found)
    a, L, night = _render_world(x0, y0, vw, vh, f, hour, fs, avoid, icon_fx)
    if zoom == 2:
        a = np.repeat(np.repeat(a, 2, 0), 2, 1)
        L = np.repeat(np.repeat(L, 2, 0), 2, 1)
    _apply_grade(a, hour)
    _emissive(a, L, night)
    _aurora(a, x0, y0, f, zoom, night)
    # UI layer (lightly graded so it stays readable)
    ui = a
    _route_dots(ui, x0, y0, route_upto, zoom)
    if trail is not None:
        _route_partial(ui, x0, y0, trail[0], trail[1], zoom)
    if flags:
        _draw_flags(ui, f, fs, icon_fx, lambda wx, wy: ((wx - x0) * zoom, (wy - y0) * zoom, 1.0), label_alpha)
    if labels:
        _draw_labels(ui, x0, y0, zoom, fs, label_fx, alpha=label_alpha)
    c.a[...] = ui
    if balloon is not None:
        bx, by, size = balloon[:3]
        alt = balloon[3] if len(balloon) > 3 else {'xs': 9, 's': 14, 'm': 22, 'l': 30}[size]
        _balloon(c, (bx - x0) * zoom, (by - y0) * zoom, f, size, alt)
    return c


# ====================================================================== Mode 7
_F = 160.0


class Cam:
    """Perspective camera over the map plane. T = ground point at the principal point (W/2, cyp)."""

    def __init__(self, tx, ty, heading, pitch_deg, dist, cyp=90.0):
        self.T = np.array([tx, ty], float)
        self.psi = heading
        self.th = math.radians(pitch_deg)
        self.D = dist
        self.cyp = cyp
        self.h = np.array([math.sin(heading), -math.cos(heading)])      # forward on the ground
        self.r = np.array([math.cos(heading), math.sin(heading)])       # right on the ground
        ct, st = math.cos(self.th), math.sin(self.th)
        self.Fw = np.array([ct * self.h[0], ct * self.h[1], -st])
        self.U = np.array([st * self.h[0], st * self.h[1], ct])
        self.R = np.array([self.r[0], self.r[1], 0.0])
        self.C = np.array([tx, ty, 0.0]) - dist * self.Fw

    def project(self, wx, wy, wz=0.0):
        d = np.array([wx, wy, wz]) - self.C
        depth = d @ self.Fw
        if depth < 1.0:
            return None
        sx = W / 2 + _F * (d @ self.R) / depth
        sy = self.cyp - _F * (d @ self.U) / depth
        return sx, sy, _F / depth

    def project_many(self, pts):
        d = np.concatenate([pts, np.zeros((len(pts), 1))], 1) - self.C
        depth = d @ self.Fw
        ok = depth > 1.0
        depth = np.where(ok, depth, 1.0)
        sx = W / 2 + _F * (d @ self.R) / depth
        sy = self.cyp - _F * (d @ self.U) / depth
        return sx, sy, _F / depth, ok

    def horizon(self):
        """Screen row of the horizon (may be off-screen)."""
        if self.th >= math.radians(89.9):
            return -1e9
        return self.cyp - _F * math.tan(self.th)


def cam_for(G, sxo, vt, heading, pitch, dist, cyp):
    """Camera such that ground point G appears at screen (W/2 + sxo, cyp + vt)."""
    th = math.radians(pitch)
    h = np.array([math.sin(heading), -math.cos(heading)])
    r = np.array([math.cos(heading), math.sin(heading)])
    a = vt * dist / (_F * math.sin(th) + vt * math.cos(th))
    depth = dist - a * math.cos(th)
    b = sxo * depth / _F
    T = np.array(G, float) + a * h - b * r
    return Cam(T[0], T[1], heading, pitch, dist, cyp)


_SKY_BY_HOUR = [(0, 'night'), (4.8, 'predawn'), (5.6, 'dawn'), (6.4, 'sunrise'), (7.5, 'morning'),
                (11, 'noon'), (15, 'afternoon'), (17.2, 'golden'), (18.6, 'sunset'), (19.4, 'dusk'),
                (20.4, 'night')]


def _sky_colors(hour):
    from .palette import SKIES
    name = 'night'
    for h0, nm in _SKY_BY_HOUR:
        if hour >= h0:
            name = nm
    return [col(c) for c in SKIES[name]]


def _sky(a, y1, hour, f, cam):
    if y1 <= 0:
        return
    y1 = int(min(H, y1))
    cols = _sky_colors(hour)
    c = Canvas(W, y1)
    hz = max(1, y1)
    c.gradient(0, 0, W, hz, cols, soft=0.6)
    _, _, night = grade_params(hour)
    if night > 0.3:
        rg = np.random.default_rng(3)
        n = 70
        xs = rg.integers(0, W, n)
        ys = rg.integers(0, max(1, y1 - 4), n)
        shift = int(cam.psi * 60) % W
        for i in range(n):
            if (f // 6 + i) % 9 == 0:
                continue
            x = (xs[i] - shift) % W
            c.a[ys[i], x] = col('#ffffff' if i % 3 else '#bfe3ff')
    a[:y1] = c.a


def _mode7_ground(cam, f, hour, y_start, live=False, avoid=None):
    """Sample the textured plane for rows y_start..H-1 -> (rows, W, 3) + per-row footprint t.
    live=True samples the near field (mip 0) from an animated patch (waves, foam, ships, trains,
    cloud shadows) so tilting from the top-down map does not pop."""
    S = _static()
    mips = S['mips']
    rows = np.arange(y_start, H)
    v = rows + 0.5 - cam.cyp
    st, ct = math.sin(cam.th), math.cos(cam.th)
    denom = _F * st + v * ct
    denom = np.maximum(denom, 1e-3)
    cz = cam.C[2]
    t = cz / denom
    h = cam.h
    base = cam.C[:2][None, :] + t[:, None] * (h[None, :] * (_F * ct - v * st)[:, None])
    u = np.arange(W) + 0.5 - W / 2
    px = base[:, 0:1] + t[:, None] * u[None, :] * cam.r[0]
    py = base[:, 1:2] + t[:, None] * u[None, :] * cam.r[1]
    # per-row footprint (world px per screen px) for mip selection
    depth_fp = t * np.abs(_F * ct * ct + _F * st * st) / denom * 1.0
    fp = np.maximum(t, depth_fp * 0.6)
    lvl = np.clip(np.floor(np.log2(np.maximum(fp, 1e-3)) + 0.25), 0, len(mips) - 1).astype(int)
    out = np.empty((len(rows), W, 3), np.uint8)
    patch = None
    if live and (lvl == 0).any():
        r0 = lvl == 0
        qx, qy = px[r0], py[r0]
        X0 = int(max(-40, np.floor(qx.min()) - 2))
        Y0 = int(max(-40, np.floor(qy.min()) - 2))
        X1 = int(min(MAP_W + 40, np.ceil(qx.max()) + 3))
        Y1 = int(min(MAP_H + 40, np.ceil(qy.max()) + 3))
        if 0 < X1 - X0 <= 1000 and 0 < Y1 - Y0 <= 800:
            arr = _crop(S['tex'], X0, Y0, X1 - X0, Y1 - Y0, _DEEP)
            _water_anim(arr, X0, Y0, f)
            _ships(arr, X0, Y0, f)
            _trains(arr, X0, Y0, f)
            _clouds(arr, X0, Y0, f, avoid, shadows=True, tops=False)
            patch = (X0, Y0, arr)
    for L_ in np.unique(lvl):
        sel = lvl == L_
        if L_ == 0 and patch is not None:
            X0, Y0, m = patch
            xi = np.floor(px[sel]).astype(np.int64) - X0
            yi = np.floor(py[sel]).astype(np.int64) - Y0
        else:
            m = mips[L_]
            sc = 2 ** L_
            xi = np.floor(px[sel] / sc).astype(np.int64)
            yi = np.floor(py[sel] / sc).astype(np.int64)
        inside = (xi >= 0) & (xi < m.shape[1]) & (yi >= 0) & (yi < m.shape[0])
        xi = xi.clip(0, m.shape[1] - 1)
        yi = yi.clip(0, m.shape[0] - 1)
        g = m[yi, xi]
        if not inside.all():                      # beyond the edge: the coarsest mip, no streaks
            mc = mips[-1]
            sc = 2 ** (len(mips) - 1)
            xo = np.floor(px[sel][~inside] / sc).astype(np.int64).clip(0, mc.shape[1] - 1)
            yo = np.floor(py[sel][~inside] / sc).astype(np.int64).clip(0, mc.shape[0] - 1)
            g[~inside] = mc[yo, xo]
        out[sel] = g
    # distance beyond the map edge (world px): render_persp fades it into the horizon haze
    d_out = np.maximum.reduce([-px, px - MAP_W, -py, py - MAP_H, np.zeros_like(px)])
    return out, t, px, py, d_out


def render_persp(c, cam, f, hour=12.0, found=None, route_upto=None, trail=None, labels=False,
                 icon_fx=None, balloon=None, avoid=None, sky_hour=None, live=False, label_alpha=0.0,
                 label_fx=None, clouds=False):
    """Generic perspective renderer (Mode 7).
    balloon = ('fixed', sx, sy, size, gx, gy) or ('world', wx, wy, size, alt).
    live: animated near field; label_alpha > 0 draws (dither-faded) labels + quest flags where the
    ground scale k >= 0.75; clouds: the drifting map clouds as ground-anchored billboards."""
    from . import worldmap_icons as WI
    fs = _found_set(found)
    a = c.a
    hz = cam.horizon()
    y_start = int(max(0, math.ceil(hz + 1)))
    _, _, night = grade_params(hour)
    sky_cols = _sky_colors(sky_hour if sky_hour is not None else hour)
    if y_start > 0:
        _sky(a, y_start, sky_hour if sky_hour is not None else hour, f, cam)
    if y_start < H:
        g, t, px, py, d_out = _mode7_ground(cam, f, hour, y_start, live, avoid)
        _apply_grade(g, hour)
        # haze toward the horizon; the world's edge dissolves into the mist
        haze_c = np.array(sky_cols[-1], np.float32)
        far = np.clip((t - cam.D * 1.4) / (cam.D * 9.0), 0, 1) ** 0.8 * 0.85
        far = np.maximum(far[:, None], np.clip(d_out / 60.0, 0, 1))
        if far.max() > 0.01:
            thr = ((np.arange(len(t))[:, None] * 5 + np.arange(W)[None, :] * 3) % 8 + 0.5) / 8.0
            q = np.minimum(1.0, np.floor(far * 6 + thr) / 6.0)
            gf = g.astype(np.float32)
            g = np.clip(gf + (haze_c[None, None, :] - gf) * q[..., None], 0, 255).astype(np.uint8)
        a[y_start:] = g
    k_top = _F / cam.D
    # billboards: icons (far first)
    items = []
    for lid in LOCS:
        ic = WI.icon(lid)
        if ic is None:
            continue
        x, y = loc_xy(lid)
        p = cam.project(x, y)
        if p is None:
            continue
        sx, sy, k = p
        if not (-60 < sx < W + 60 and -10 < sy < H + 70):
            continue
        items.append((k, lid, ic, sx, sy))
    items.sort(key=lambda it: it[0])
    for k, lid, ic, sx, sy in items:
        fx = (icon_fx or {}).get(lid, {})
        state = fx.get('state') or ('grey' if (lid in TL.TOUR and lid not in fs) else 'color')
        kk = k * fx.get('scale', 1.0)
        if kk >= 0.72:
            spr, ax, ay = (ic.grey if state == 'grey' else ic.rgba), ic.ax, ic.ay
            if kk >= 1.6:
                spr = _scale_sprite(spr, 2.0)
                ax, ay = ax * 2, ay * 2
        elif kk >= 0.36:
            hi = WI.icon_half(lid)
            spr, ax, ay = hi.rgba, hi.ax, hi.ay
            if state == 'grey':
                spr = _greyed(spr)
        elif kk >= 0.12:
            cc = '#f9c22b' if (lid in TL.TOUR and lid in fs) else '#d8d0e0'
            _px(a, int(sx), int(sy) - 1, cc)
            _px(a, int(sx), int(sy), '#2e222f')
            continue
        else:
            continue
        tmp = spr.copy()
        if state != 'grey':
            g2 = tmp[..., :3].copy()
            _apply_grade(g2, hour)
            tmp[..., :3] = g2
            if night > 0.05 and kk >= 0.72 and spr.shape == ic.lights.shape:
                m = ic.lights[:, :, 3] > 0
                tmp[m, :3] = ic.lights[m, :3]
        else:
            g2 = tmp[..., :3].copy()
            _apply_grade(g2, hour)
            tmp[..., :3] = g2
        _blit_rgba(a, tmp, int(sx) - ax, int(sy) - ay)
    # route dots
    if route_upto:
        n_legs = len(TL.TOUR) - 1
        r = min(float(route_upto), n_legs)
        for i in range(min(int(r) + 1, n_legs)):
            frac = 1.0 if i < int(r) else r - int(r)
            if frac > 0:
                _persp_dots(a, cam, i, frac)
    if trail is not None:
        _persp_dots(a, cam, trail[0], trail[1])
    if clouds:
        for (x, y, v) in _cloud_positions(f):
            spr = _cloud_sprite(v)
            if avoid is not None:
                ax_, ay_, ar = avoid
                if abs(x - ax_) < ar + spr.shape[1] / 2 and abs(y - ay_) < ar + spr.shape[0]:
                    continue
            pr = cam.project(x, y)
            if pr is None or pr[2] < 0.36:
                continue
            sx, sy, k = pr
            if not (-60 < sx < W + 60 and -10 < sy < H + 40):
                continue
            if k < 0.72:
                spr = _scale_sprite(spr, 0.5)
            elif k >= 1.6:
                spr = _scale_sprite(spr, 2.0)
            tmp = spr.copy()
            g2 = tmp[..., :3].copy()
            _apply_grade(g2, hour)
            tmp[..., :3] = g2
            _blit_rgba(a, tmp, int(sx) - tmp.shape[1] // 2, int(sy) - tmp.shape[0])
    if label_alpha > 0.001:
        def pos_fn(wx, wy):
            return cam.project(wx, wy)
        _draw_flags(a, f, fs, icon_fx, pos_fn, label_alpha)
        _draw_labels(a, 0, 0, 1, fs, label_fx, pos_fn=pos_fn, alpha=label_alpha)
    # balloon
    if balloon is not None:
        from .atlas import draw_balloon
        if balloon[0] == 'fixed':
            _, bsx, bsy, size, gx, gy = balloon
            p = cam.project(gx, gy)
            if p is not None:
                shx, shy, k = p
                rx = int(clamp(k * 5, 2, 8)) if size != 's' else 3
                _shadow_ell(a, int(shx), int(shy), rx, max(1, rx // 3))
            draw_balloon(c, int(round(bsx)), int(round(bsy)), f, size=size)
        else:
            _, wx, wy, size, alt = balloon
            p = cam.project(wx, wy)
            if p is not None:
                shx, shy, k = p
                rx = {'xs': 2, 's': 3, 'm': 5, 'l': 7}[size]
                _shadow_ell(a, int(shx), int(shy), rx, 1)
                draw_balloon(c, int(shx), int(shy - alt * max(0.6, min(1.6, k))), f, size=size)
    return c


def _greyed(spr):
    a = spr[:, :, :3].astype(np.float32)
    luma = a @ np.array([0.3, 0.55, 0.15], np.float32)
    g = (luma * 0.62 + 30)[..., None] * np.array([0.92, 0.95, 1.08], np.float32)
    out = spr.copy()
    out[:, :, :3] = np.clip(g, 0, 255).astype(np.uint8)
    return out


def _shadow_ell(a, cx, cy, rx, ry):
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / (rx + 0.5)) ** 2 + (dy / (ry + 0.5)) ** 2 <= 1:
                x, y = cx + dx, cy + dy
                if 0 <= x < W and 0 <= y < H:
                    a[y, x] = (a[y, x].astype(np.uint16) * 3 // 5).astype(np.uint8)


def _persp_dots(a, cam, i, frac):
    tt, pts = _leg_dots(i)
    m = tt <= frac
    if not m.any():
        return
    sx, sy, k, ok = cam.project_many(pts[m])
    for x, y, kk, o in zip(sx, sy, k, ok):
        if not o:
            continue
        x, y = int(x), int(y)
        if kk < 0.3:
            continue
        if 0 <= x < W - 1 and 0 <= y < H - 1:
            if kk > 0.6:
                a[y + 1, x:x + 2] = col('#6e2727')
                a[y, x:x + 2] = col('#f04f78')
            else:
                a[y, x] = col('#f04f78')


def draw_mode7(c, cx, cy, heading, f, altitude=60, hour=12.0, found=None, pitch=24, route_upto=None):
    """SNES-style Mode-7 view looking at world (cx, cy) with heading (radians, 0 = north) and pitch
    (degrees above straight-ahead; 90 = top-down). altitude = camera height in world px."""
    dist = altitude / max(0.05, math.sin(math.radians(pitch)))
    cam = Cam(cx, cy, heading, pitch, dist, cyp=112)
    return render_persp(c, cam, f, hour, found, route_upto)


# ====================================================================== travel shots
# 2-bar flight contract (film-local frames of a region segment; the music follows it)
FLIGHT = getattr(TL, 'FLIGHT', 144)
T_LIFT = 16                    # departure hold: the balloon lifts off from the source (0..16)
T_FLY0, T_FLY1 = 14, 112       # balloon progress window (even speed, engine.shotfx.cruise)
T_IN0, T_IN1 = 4, 40           # long hops: tilt into Mode 7
T_OUT0, T_OUT1 = 84, 120       # long hops: tilt back to top-down
T_ARRIVE = 126                 # ARRIVE hit
T_PLACES = 128                 # seg['places'] labels light up 128..134
T_IRIS = 136                   # iris closes 136..143
_DEST_SX, _DEST_SY = 204, 116          # where the destination anchor sits on screen at arrival
SHORT_HOP = 260
_LH_PITCH, _LH_CYP, _LH_VT = 30.0, 100.0, 44.0   # long-hop Mode-7 cruise framing
_LH_SPEED = 6.5                                  # target max ground speed (screen px/frame, bottom row)


def _cruise(t):
    from .shotfx import cruise
    return cruise(t)


def _leg_index(seg_id):
    s = TL.BY_ID[seg_id]
    return TL.TOUR.index(s['loc']) - 1


def dest_screen(seg_id):
    """Screen position of the iris centre (destination icon centre) in the travel shot."""
    from . import worldmap_icons as WI
    s = TL.BY_ID[seg_id]
    ic = WI.icon(s['loc'])
    return _DEST_SX, _DEST_SY - (ic.ay - ic.h // 2 if ic else 0)


def _arrival_fx(f, dest, t0=None):
    """icon_fx for the destination: grey until the ARRIVE hit, then pop + white flash."""
    t0 = T_ARRIVE if t0 is None else t0
    if f < t0:
        return {dest: {'state': 'grey'}}
    t = seg(f, t0, t0 + 8)
    k = 1.0 + 0.3 * math.sin(math.pi * t) * (1 - t * 0.5) if t < 1 else 1.0
    k = round(k * 8) / 8.0
    st = 'white' if f in (t0, t0 + 1) else 'color'
    return {dest: {'state': st, 'scale': k}}


def _arrival_overlay(c, f, dest, sx, sy, ic, places, t0=None):
    """Sparkle ring + big name plate (screen space), starting at the ARRIVE hit t0."""
    t0 = T_ARRIVE if t0 is None else t0
    if f < t0:
        return
    cx, cy = sx, sy - (ic.ay - ic.h // 2)
    t = seg(f, t0, t0 + 14)
    if t < 1:
        r = 6 + (ic.h * 0.45 + 20) * ease_out(t)
        n = 12
        for i in range(n):
            ang = i / n * 2 * math.pi + f * 0.05
            x = int(cx + math.cos(ang) * r * 1.15)
            y = int(cy + math.sin(ang) * r * 0.85)
            colr = '#ffffff' if (i + f // 2) % 3 == 0 else '#f9c22b'
            if t > 0.7 and (i + f) % 2:
                continue
            _px(c.a, x, y, colr)
            if t < 0.6:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    _px(c.a, x + dx, y + dy, '#f9c22b' if colr == '#ffffff' else '#f7961d')
    # big name plate (normal font, gold rim)
    name = LOCS[dest]['name']
    tw = text_width(name, 'normal')
    pw, ph = tw + 12, 13
    kk = ease_out_back(seg(f, t0 + 1, t0 + 7))
    if kk <= 0.05:
        return
    px_ = int(clamp(cx - pw // 2, 4, W - pw - 4))
    py_ = int(clamp(sy + 8, 50, H - ph - 6))
    if kk < 0.999:
        cw = max(4, int(pw * min(1.0, kk)))
        c.round_rect(px_ + (pw - cw) // 2, py_, cw, ph, '#f9c22b', r=2)
        c.rect(px_ + (pw - cw) // 2 + 1, py_ + 1, cw - 2, ph - 2, '#3a2440')
        return
    c.round_rect(px_ - 1, py_ - 1, pw + 2, ph + 2, '#6e2727', r=2)
    c.round_rect(px_, py_, pw, ph, '#f9c22b', r=2)
    c.rect(px_ + 1, py_ + 1, pw - 2, ph - 2, '#3a2440')
    c.rect(px_ + 2, py_ + 2, pw - 4, 1, '#5a3a60')
    text(c, px_ + 6, py_ + 3, name, '#fff6c0', font='normal', shadow='#6e2727')
    for xx in (px_ + 2, px_ + pw - 3):
        _px(c.a, xx, py_ + 6, '#fbff86')


def _label_fx(f, s):
    """Labels of seg['places'] light up one by one at f 128..134 (dim before, flash, then normal)."""
    fx = {s['loc']: 'hide'} if f > T_ARRIVE else {}
    pl = s.get('places', [])
    n = len(pl)
    for i, lid in enumerate(pl):
        t0 = T_PLACES + (int(round(i * 6 / (n - 1))) if n > 1 else 0)
        if f < t0:
            fx[lid] = 'dim'
        elif f < t0 + 4:
            fx[lid] = 'lit'
    return fx


def _balloon_end(dest):
    """Where the balloon parks (ground point) beside a stop's icon."""
    from . import worldmap_icons as WI
    ic = WI.icon(dest)
    x, y = loc_xy(dest)
    half = (ic.w // 2 if ic else 8)
    return np.array([x - half - 9.0, y - 4.0])


def travel_frame(f, seg_id):
    """2-bar travel shot of a region segment (film-local f 0..143). See docs/LOC_WORLDMAP.md."""
    from .atlas import iris
    s = TL.BY_ID[seg_id]
    st = _flight_state(f, seg_id)
    c = Canvas()
    if st['mode'] == 'top':
        x0, y0 = _topdown_tail(c, f, s, st['leg'], st['p'], st['end'], st['hour'], st['found'], st['cam_c'],
                               st['q'], G=st['G'], alt=st['alt'])
        if st['puff']:
            _departure_puff(c, f, st['G'][0] - x0, st['G'][1] - y0)
    else:
        _render_flight_persp(c, f, s, st)
    if f >= T_IRIS:
        sx, sy = dest_screen(seg_id)
        iris(c, seg(f, T_IRIS - 1, FLIGHT - 1), sx, sy)
    return c


@lru_cache(maxsize=4096)
def _flight_state(f, seg_id):
    """Pure camera/balloon state of a travel frame:
    mode 'top' (cam_c = top-down camera centre) or 'persp' (cam = Cam), G = balloon ground point."""
    s = TL.BY_ID[seg_id]
    dest = s['loc']
    k = TL.TOUR.index(dest)
    st = dict(hour=TL.hour_at(seg_id, f), found=k + (1 if f >= T_ARRIVE else 0), dest=dest, puff=False)
    if s.get('src') is None:
        dxy = np.array(loc_xy(dest), float)
        p = _arc(dxy + np.array([-30.0, 420.0]), dxy, 1)
        end = _balloon_end(dest)
        q = _cruise(seg(f, 0, T_FLY1))
        G = _ground_pos(p, end, q)
        tilt = 1 - smooth(seg(f, T_SKY_TILT0, T_SKY_TILT1))
        st.update(leg=-1, p=p, end=end, q=q, G=G, tilt=tilt, alt=14, route=None, trail=None)
        if tilt <= 0.001:
            st.update(mode='top', cam_c=np.round(_cam_center_for(dxy)))
            return st
        tq = min(0.999, max(0.001, q))
        tan = _bez_tan(p, tq)
        psi_leg = math.atan2(tan[0], -tan[1])
        lift = 1 - smooth(seg(f, 0, T_SKY_LIFT))
        psi = _ang_lerp(0.0, psi_leg, smooth(tilt))
        pitch = lerp(90.0, lerp(_M7_PITCH, 12.0, lift), tilt)
        dist = lerp(_F, _M7_DIST * (900.0 / _M7_DIST) ** lift, tilt)   # geometric: even zoom rate
        cyp = lerp(H / 2, lerp(_M7_CYP, 150.0, lift), tilt)
        sxo = lerp(_DEST_SX - W / 2, 0.0, tilt)
        vt = lerp(_DEST_SY - cyp, lerp(_M7_VT, 20.0, lift), tilt)
        cam = cam_for(dxy * (1 - tilt) + G * tilt, sxo, vt, psi, pitch, dist, cyp)
        if tilt > 0.5:
            pr = cam.project(*G)
            bsx, bsy = (pr[0], pr[1] - 28) if pr else (W / 2, 120)
            bal = ('fixed', bsx, max(40, bsy), 'm', G[0], G[1])
        else:
            bal = _balloon_fixed(cam, G, tilt, 14)
        st.update(mode='persp', cam=cam, bal=bal, la=1 - smooth(seg(tilt, 0.0, 0.22)))
        return st
    leg = k - 1
    p = _bezier_leg(leg)
    start = _balloon_end(s['src'])
    end = _balloon_end(dest)
    u = _cruise(seg(f, T_FLY0, T_FLY1))
    G = _ground_pos2(p, start, end, u)
    st.update(leg=leg, p=p, start=start, end=end, q=u, G=G, alt=_lift_alt(f), puff=f < 22,
              route=leg if leg > 0 else None, trail=(leg, u))
    L = float(np.hypot(*(p[2] - p[0])))
    if L < SHORT_HOP:
        cam = _cam_center_for(p[0]) * (1 - u) + _cam_center_for(p[2]) * u
        bs = G - cam + np.array([W / 2, H / 2])
        push = np.array([min(0.0, bs[0] - 80) + max(0.0, bs[0] - (W - 50)),
                         min(0.0, bs[1] - 64) + max(0.0, bs[1] - (H - 24))])
        st.update(mode='top', cam_c=np.round(cam + push), tilt=0.0)
        return st
    tilt = smooth(seg(f, T_IN0, T_IN1)) * (1 - smooth(seg(f, T_OUT0, T_OUT1)))
    st['tilt'] = tilt
    if tilt <= 0.001:
        st.update(mode='top', cam_c=np.round(_cam_center_for(p[0] if f < 64 else p[2])))
        return st
    Dc = _leg_cruise_dist(leg)
    pitch = lerp(90.0, _LH_PITCH, tilt)
    dist = lerp(_F, Dc, tilt)
    cyp = lerp(H / 2, _LH_CYP, tilt)
    sxo = lerp(_DEST_SX - W / 2, 0.0, tilt)
    vt = lerp(_DEST_SY - cyp, _LH_VT, tilt)
    anchor = (p[0] if f < 64 else p[2]) * (1 - tilt) + G * tilt
    cam = cam_for(anchor, sxo, vt, 0.0, pitch, dist, cyp)        # north stays up: no camera spin
    st.update(mode='persp', cam=cam, bal=_balloon_fixed(cam, G, tilt, _lift_alt(f)),
              la=1 - smooth(seg(tilt, 0.0, 0.22)))
    return st


def _render_flight_persp(c, f, s, st):
    dest = st['dest']
    cam = st['cam']
    render_persp(c, cam, f, st['hour'], st['found'], route_upto=st['route'], trail=st['trail'],
                 icon_fx=_arrival_fx(f, dest), balloon=st['bal'], avoid=(*loc_xy(dest), 60), live=True,
                 label_alpha=st['la'], label_fx=_label_fx(f, s), clouds=True)
    _cloud_billboards(c, cam, f, st['hour'], alpha=smooth(seg(st['tilt'], 0.3, 0.9)))
    if st['puff'] and 'start' in st:
        pr = cam.project(*st['start'])
        if pr is not None:
            _departure_puff(c, f, pr[0], pr[1])


def _ground_pos(p, end, q):
    """Balloon ground point at progress q: along the leg arc, easing onto the parking spot."""
    b = _bez(p, q)
    w = smooth(seg(q, 0.7, 1.0))
    return b + (end - p[2]) * w


def _ground_pos2(p, start, end, q):
    """Like _ground_pos, also easing off the source parking spot at the start."""
    b = _bez(p, q)
    w0 = 1 - smooth(seg(q, 0.0, 0.3))
    w1 = smooth(seg(q, 0.7, 1.0))
    return b + (start - p[0]) * w0 + (end - p[2]) * w1


def _lift_alt(f):
    """Balloon height above its shadow (top-down 's'): lifts off during the departure hold."""
    return 3 + 11 * smooth(seg(f, 3, T_LIFT))


def _departure_puff(c, f, sx, sy):
    """Little burner puff at lift-off (screen point = the balloon's ground spot)."""
    if not (2 <= f < 20):
        return
    t = (f - 2) / 18.0
    for i, (dx, dy) in enumerate(((-1, 0), (1, 0), (-0.6, -0.5), (0.6, -0.5), (0, -1))):
        r = 2 + 9 * ease_out(t)
        x = int(sx + dx * r)
        y = int(sy - 2 + dy * r * 0.7 - t * 4)
        if t > 0.6 and (i + f) % 2:
            continue
        cc = '#ffffff' if t < 0.4 else '#dfe8ee' if t < 0.75 else '#b8c4cc'
        _ell(c.a, x, y, 1 if t > 0.5 else 2, 1, cc)


def _topdown_tail(c, f, s, leg, p, end, hour, found, cam_c, q, G=None, alt=14):
    """Top-down frame (camera centre cam_c), balloon at ground point G (or progress q), arrival fx."""
    from . import worldmap_icons as WI
    dest = s['loc']
    ic = WI.icon(dest)
    if G is None:
        G = _ground_pos(p, end, q) if p is not None else end
    route_prev = leg if leg > 0 else None
    draw_map(c, cam_c[0], cam_c[1], f, hour=hour, found=found, route_upto=route_prev,
             avoid=(*loc_xy(dest), 60), icon_fx=_arrival_fx(f, dest), label_fx=_label_fx(f, s),
             trail=(leg, min(1.0, q)) if leg >= 0 else None, balloon=(G[0], G[1], 's', alt))
    x0 = int(round(cam_c[0] - W / 2))
    y0 = int(round(cam_c[1] - H / 2))
    dx, dy = loc_xy(dest)
    _arrival_overlay(c, f, dest, dx - x0, dy - y0, ic, s.get('places', []))
    return x0, y0


def _cam_center_for(world_pt):
    """Camera centre that puts world_pt at the destination screen spot."""
    return np.array([world_pt[0] - (_DEST_SX - W / 2), world_pt[1] - (_DEST_SY - H / 2)], float)


_M7_PITCH, _M7_DIST, _M7_CYP, _M7_VT = 26.0, 150.0, 104.0, 46.0   # Canterlot sky approach (+ intro)


def _ang_lerp(a, b, t):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


@lru_cache(maxsize=None)
def _leg_cruise_dist(leg):
    """Mode-7 camera distance for a long leg, high enough that the ground moves <= _LH_SPEED px/frame."""
    p = _bezier_leg(leg)
    start = _balloon_end(TL.TOUR[leg])
    end = _balloon_end(TL.TOUR[leg + 1])
    fs = np.arange(T_FLY0, T_FLY1 + 1)
    G = np.array([_ground_pos2(p, start, end, _cruise(seg(f, T_FLY0, T_FLY1))) for f in fs])
    vmax = float(np.hypot(*np.diff(G, axis=0).T).max())
    th = math.radians(_LH_PITCH)
    vb = H - _LH_CYP
    kb = _LH_SPEED / max(0.5, vmax)
    D = (_F * math.sin(th) + vb * math.cos(th)) / (kb * math.sin(th))
    return float(clamp(D, 190.0, 1100.0))


def _balloon_fixed(cam, G, tilt, alt_td):
    """Balloon sprite position for perspective frames: blends the top-down height into the Mode-7
    foreground framing (28 px above its shadow, 'm' size)."""
    pr = cam.project(G[0], G[1])
    if pr is None:
        return None
    shx, shy, k = pr
    a = lerp(alt_td * max(0.6, min(1.6, k)), 28.0, smooth(seg(tilt, 0.15, 0.85)))
    size = 'm' if tilt > 0.5 else 's'
    return ('fixed', shx, max(40.0, shy - a), size, G[0], G[1])


def _cloud_billboards(c, cam, f, hour, alpha=1.0):
    """A few cloud puffs floating at altitude between camera and ground (parallax in Mode 7)."""
    if alpha <= 0.001:
        return
    rg = np.random.default_rng(11)
    pts = rg.uniform(0, 1, (40, 2)) * np.array([MAP_W, MAP_H])
    for i, (x, y) in enumerate(pts):
        x = (x + f * 0.3) % MAP_W
        p = cam.project(x, y, 26.0)
        if p is None:
            continue
        sx, sy, k = p
        if not (-40 < sx < W + 40 and 0 < sy < H + 20) or k < 0.5:
            continue
        spr = _cloud_sprite(i % 6)
        if k > 1.5:
            spr = _scale_sprite(spr, 2.0)
        elif k < 0.8:
            spr = _scale_sprite(spr, 0.5)
        tmp = spr.copy()
        g2 = tmp[..., :3].copy()
        _apply_grade(g2, hour, 0.8)
        tmp[..., :3] = g2
        _blit_rgba(c.a, tmp, int(sx) - tmp.shape[1] // 2, int(sy) - tmp.shape[0], alpha)


T_SKY_LIFT, T_SKY_TILT0, T_SKY_TILT1 = 76, 76, 122   # Canterlot sky approach phases


# ====================================================================== overview
OVERVIEW_RECT = (78, 4, 165, 173)            # x, y, w, h of the map inside the frame
LEFT_PANEL = (4, 4, 68, 173)                 # free side panels for titles / credits
RIGHT_PANEL = (248, 4, 68, 173)
_OX, _OY, _OW, _OH = OVERVIEW_RECT


def overview_xy(lid):
    x, y = loc_xy(lid)
    return _OX + int(x * _OW / MAP_W), _OY + int(y * _OH / MAP_H)


def _ov(x, y):
    return x * _OW / MAP_W, y * _OH / MAP_H


@lru_cache(maxsize=1)
def _overview_base():
    """Crisp small-scale map (OH, OW, 3), rendered from the class map (majority per block) with its own
    small-scale styling (tiny tree crowns, curated mountain marks, ice floes)."""
    from . import worldmap_build as WB
    from . import worldmap_art as A
    from . import worldmap_data as WD
    from scipy import ndimage
    S = _static()
    cls = S['cls']
    forest = S['forest']
    n = len(WB.CLASSES)
    by = (np.arange(MAP_H) * _OH // MAP_H)
    bx = (np.arange(MAP_W) * _OW // MAP_W)
    bidx = (by[:, None] * _OW + bx[None, :]).ravel()
    counts = np.bincount(bidx * n + cls.ravel().astype(np.int64), minlength=_OW * _OH * n).reshape(_OH * _OW, n)
    land_frac = 1 - counts[:, 0] / np.maximum(1, counts.sum(1))
    cl = counts.copy()
    cl[:, 0] = 0
    maj = cl.argmax(1).reshape(_OH, _OW)
    land = (land_frac > 0.42).reshape(_OH, _OW)
    fcount = np.bincount(bidx, weights=forest.ravel().astype(np.float64), minlength=_OW * _OH)
    ffrac = (fcount / np.maximum(1, counts.sum(1))).reshape(_OH, _OW)
    ramps = WB._RAMP_ARR
    yy, xx = np.mgrid[0:_OH, 0:_OW]
    tex = ((xx * 7 + yy * 13) % 11) < 2
    out = np.where(tex[..., None], ramps[maj, 1], ramps[maj, 2])
    # forests: dark canopy with tiny lit tree crowns
    fm = land & (ffrac > 0.4) & ~np.isin(maj, [WB.CI['snow'], WB.CI['tundra']])
    canopy = (ramps[maj, 0].astype(np.int16) * 9 // 10).astype(np.uint8)
    out[fm] = canopy[fm]
    crown = fm & (yy % 2 == 0) & (((xx + (yy // 2) % 2 * 2) % 3) == 0)
    out[crown] = ramps[maj, 2][crown]
    shadow = np.zeros_like(crown)
    shadow[1:, :] = crown[:-1, :]
    shadow &= fm
    out[shadow] = (ramps[maj, 0][shadow].astype(np.int16) * 7 // 10).astype(np.uint8)
    # water
    dw = ndimage.distance_transform_edt(~land)
    wat = np.where(dw <= 1, 0, np.where(dw <= 2, 1, np.where(dw <= 4, 2, np.where(dw <= 8, 3, 4))))
    wp = np.array([col(c) for c in ('#9adcea', '#6fbfe0', '#4f97cf', '#3f7fc2', '#2f62aa')], np.uint8)
    o2 = np.where(land[..., None], out, wp[wat]).astype(np.uint8)
    # coast lip
    below = np.zeros_like(land)
    below[:-1] = ~land[1:]
    lip = land & below
    o2[lip] = (o2[lip].astype(np.int16) * 3 // 4).astype(np.uint8)
    # rivers (1 px)
    for pts, w0, w1 in WD.RIVERS:
        line, _ = WB.polyline_px(pts, 3.0)
        for (x, y) in line:
            ox, oy = int(x * _OW / MAP_W), int(y * _OH / MAP_H)
            if 0 <= ox < _OW and 0 <= oy < _OH and land[oy, ox]:
                o2[oy, ox] = col('#56aedc')
    # ice floes along the south edge and a few arctic bergs
    rg = np.random.default_rng(21)
    for y in range(int(_OH * 0.875), _OH, 3):
        for x in range(0, _OW, 4):
            px_, py_ = x + int(rg.integers(0, 3)), y + int(rg.integers(0, 2))
            if py_ >= _OH or px_ >= _OW - 2 or land[py_, px_]:
                continue
            if rg.random() > (py_ / _OH - 0.875) / 0.06:
                continue
            w_ = 2 + int(rg.integers(0, 2))
            o2[py_, px_:px_ + w_] = col('#f4faff')
            if py_ + 1 < _OH:
                o2[py_ + 1, px_ + 1:px_ + w_ + 1] = col('#9cc4e4')
    for k in range(14):
        px_, py_ = int(rg.uniform(0.55, 0.98) * _OW), int(rg.uniform(0.005, 0.06) * _OH)
        if not land[py_, px_]:
            o2[py_, px_] = col('#ffffff')
            o2[min(_OH - 1, py_ + 1), px_] = col('#9cc4e4')
    # curated mountain marks along the ranges + named peaks (back to front)
    marks = []
    for scheme, line_n, pw, ph, spacing, rows, jit, seed in WD.RANGES:
        line, ts = WB.polyline_px(line_n, 1.0)
        step = 52.0 if scheme not in ('rim', 'mauve', 'cloudpk') else 40.0
        s_ = step / 2
        while s_ < ts[-1]:
            x, y = line[min(len(line) - 1, int(s_))]
            mw = 6 if scheme in ('snow', 'yaket', 'crag', 'storm') else 5
            mh = 5 if scheme in ('snow', 'yaket', 'crag') else 4
            marks.append((y, x, mw, mh, scheme))
            if rows > 1:
                marks.append((y - 30, x + 26, mw, mh, scheme))
            s_ += step
    for scheme, xn, yn, w, h, seed in WD.PEAKS:
        mw = int(clamp(w / 9.3 * 1.6, 5, 17))
        mh = int(clamp(h / 9.3 * 1.5, 4, 19))
        marks.append((yn * MAP_H, xn * MAP_W, mw, mh, scheme))
    marks.append((0.545 * MAP_H, 0.905 * MAP_W, 9, 7, 'cloudpk'))
    marks.sort()
    for y, x, mw, mh, scheme in marks:
        ox, oy = int(x * _OW / MAP_W), int(y * _OH / MAP_H)
        hi, lit, mid, sh, dsh, outc, snl, sns, snowline = A.MTN_SCHEMES[scheme]
        for r in range(mh):
            yy_ = oy - mh + 1 + r
            half = (r + 0.5) * (mw / 2) / mh
            x0_, x1_ = int(round(ox - half)), int(round(ox + half))
            for xx_ in range(x0_, x1_ + 1):
                if not (0 <= xx_ < _OW and 0 <= yy_ < _OH):
                    continue
                left_ = xx_ < ox or (xx_ == ox and r < 2)
                snowy = snowline > 0.1 and r < max(1, int(mh * snowline))
                cc = (snl if left_ else sns) if snowy else (lit if left_ else sh)
                if (xx_ == x1_ and r > 0) or r == mh - 1 and xx_ > ox:
                    cc = outc
                o2[yy_, xx_] = col(cc)
    vx, vy = int(0.905 * _OW), int(0.545 * _OH) - 6
    o2[vy, vx - 1:vx + 2] = col('#ff551e')
    o2[vy - 1, vx] = col('#8a7a7a')
    return o2, land


def _ov_frame(c):
    """Ornate atlas frame + side panels around the overview rect."""
    a = c.a
    c.gradient(0, 0, W, H, ['#1b1934', '#231f40', '#1b1934'], soft=0.5)
    yy, xx = np.mgrid[0:H, 0:W]
    pat = ((xx + yy) % 8 == 0) & ((xx - yy) % 8 == 0)
    a[pat] = col('#2e2a52')
    x, y, w, h = OVERVIEW_RECT
    # panels: thin gold-lined insets
    for (px_, py_, pw, ph) in (LEFT_PANEL, RIGHT_PANEL):
        c.frame(px_, py_, pw, ph, '#5a4a6a')
        c.frame(px_ + 2, py_ + 2, pw - 4, ph - 4, '#3a3258')
        for (cx_, cy_) in ((px_, py_), (px_ + pw - 1, py_), (px_, py_ + ph - 1), (px_ + pw - 1, py_ + ph - 1)):
            _px(a, cx_, cy_, '#f9c22b')
    # map frame: dark, gold, dark bands
    c.rect(x - 4, y - 4, w + 8, h + 8, '#2e222f')
    c.rect(x - 3, y - 3, w + 6, h + 6, '#c9a24a')
    c.rect(x - 2, y - 2, w + 4, h + 4, '#f9c22b')
    c.rect(x - 1, y - 1, w + 2, h + 2, '#6e2727')
    for i in range(x - 2, x + w + 2, 4):
        _px(a, i, y - 2, '#fbff86')
        _px(a, i + 2, y + h + 1, '#c9a24a')
    # corner ornaments (gold flourishes)
    orn = ['..ggg..', '.gGyGg.', 'gGy.yGg', 'gy.r.yg', 'gGy.yGg', '.gGyGg.', '..ggg..']
    cm = {'g': '#c9a24a', 'G': '#f9c22b', 'y': '#fbff86', 'r': '#f04f78'}
    for (cx_, cy_) in ((x - 4, y - 4), (x + w - 3, y - 4), (x - 4, y + h - 3), (x + w - 3, y + h - 3)):
        for j, row in enumerate(orn):
            for i, ch in enumerate(row):
                if ch != '.':
                    _px(a, cx_ - 1 + i, cy_ - 1 + j, cm[ch])


def _ov_compass(a, cx, cy):
    pts = {
        (0, -6): '#f04f78', (0, -5): '#f04f78', (0, -4): '#ffb0c8', (0, -3): '#ffffff',
        (0, 6): '#c9d2e0', (0, 5): '#c9d2e0', (0, 4): '#9babb2', (-6, 0): '#c9d2e0', (-5, 0): '#c9d2e0',
        (6, 0): '#c9d2e0', (5, 0): '#c9d2e0', (-4, 0): '#9babb2', (4, 0): '#9babb2', (0, 3): '#9babb2',
        (-3, 0): '#ffffff', (3, 0): '#ffffff', (-1, -2): '#f9c22b', (1, -2): '#f9c22b', (-2, -1): '#f9c22b',
        (2, -1): '#f9c22b', (-1, 2): '#f9c22b', (1, 2): '#f9c22b', (-2, 1): '#f9c22b', (2, 1): '#f9c22b',
        (0, 0): '#fbff86', (-1, 0): '#f9c22b', (1, 0): '#f9c22b', (0, -1): '#f9c22b', (0, 1): '#f9c22b',
        (-2, -2): '#c9a24a', (2, -2): '#c9a24a', (-2, 2): '#c9a24a', (2, 2): '#c9a24a',
    }
    for (dx, dy) in list(pts):
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (dx + ox, dy + oy) not in pts:
                _px(a, cx + dx + ox, cy + dy + oy, '#1e2a4a')
    for (dx, dy), cc in pts.items():
        _px(a, cx + dx, cy + dy, cc)
    # 'N'
    for dy in range(3):
        _px(a, cx - 1, cy - 11 + dy, '#ffffff')
        _px(a, cx + 1, cy - 11 + dy, '#ffffff')
    _px(a, cx, cy - 10, '#ffffff')


_OV_LABELS = [
    ('FROZEN NORTH', 0.2, 0.155, 'mist'), ('CELESTIAL SEA', 0.765, 0.43, '#c8f0ff'),
    ('LUNA OCEAN', 0.1, 0.63, '#c8f0ff'), ('BADLANDS', 0.545, 0.5, 'cream'),
    ('BONE DRY', 0.43, 0.655, 'cream'), ('EQUESTRIA', 0.33, 0.27, 'cream'),
    ('GRIFFONS', 0.92, 0.3, 'cream'), ('ICEBERG BARRIER', 0.52, 0.965, 'white'),
]


def draw_overview(c, f, hour=12.0, found=None, route_upto=None, labels=True, reveal=1.0, frame=True):
    """The whole map fitted to the screen (map rect OVERVIEW_RECT) with an ornate atlas frame and
    side panels. labels: bool or 0..1 (fraction of the overview labels shown, in order).
    reveal 0..1: the map unrolls from the top (intro helper). route_upto: legs drawn (float ok)."""
    base, land = _overview_base()
    if frame:
        _ov_frame(c)
    x, y, w, h = OVERVIEW_RECT
    m = base.copy()
    # animated water glints + drifting tiny clouds
    rg = np.random.default_rng(8)
    gx = rg.integers(0, w, 160)
    gy = rg.integers(0, h, 160)
    for i in range(160):
        if not land[gy[i], gx[i]] and (f + i * 7) % 60 < 8:
            m[gy[i], gx[i]] = col('#ffffff' if (f + i * 7) % 60 < 4 else '#c8f0ff')
    # tiny trains (pink dots) on the rails
    rails, _ = _lines()
    for k, (ri, sp, off) in enumerate(_TRAINS[:5]):
        line, tun = rails[ri]
        n = len(line)
        s = (off + f * sp * 2) % (2 * (n - 1))
        pos = int(s if s < n - 1 else 2 * (n - 1) - s)
        if tun[pos]:
            continue
        ox, oy = _ov(*line[pos])
        if 0 <= int(oy) < h and 0 <= int(ox) < w:
            m[int(oy), int(ox)] = col('#f04f78')
            if 0 <= int(oy) - 1:
                m[int(oy) - 1, int(ox)] = col('#ffffff') if (f // 6) % 2 else col('#dfe8ee')
    # tiny clouds with shadows
    for k in range(5):
        cx_ = int((rg.uniform(0, w) + f * 0.06) % (w + 20)) - 10
        cy_ = int(rg.uniform(10, h - 30))
        for dx in range(-3, 4):
            for dy in (-1, 0):
                if abs(dx) < 3 or dy == 0:
                    sx_, sy_ = cx_ + dx + 2, cy_ + dy + 4
                    if 0 <= sx_ < w and 0 <= sy_ < h:
                        m[sy_, sx_] = (m[sy_, sx_].astype(np.uint16) * 3 // 4).astype(np.uint8)
        for dx in range(-3, 4):
            for dy in (-1, 0):
                if abs(dx) < 3 or dy == 0:
                    sx_, sy_ = cx_ + dx, cy_ + dy
                    if 0 <= sx_ < w and 0 <= sy_ < h:
                        m[sy_, sx_] = col('#ffffff' if dy < 0 else '#dfe8f6')
    _apply_grade(m, hour, 0.85)
    _, _, night = grade_params(hour)
    fs = _found_set(found)
    # aurora at night (top band)
    if night > 0.05:
        for xx_ in range(w):
            yb = int(10 + 4 * math.sin(xx_ * 0.08 + f * 0.04))
            for dy in range(6):
                if (xx_ + dy + f // 4) % 3 == 0:
                    yy_ = yb - dy
                    if 0 <= yy_ < h:
                        cc = np.array(col('#5fffc0' if dy < 3 else '#a884f3'), np.float32)
                        m[yy_, xx_] = np.clip(m[yy_, xx_] * (1 - 0.35 * night) + cc * 0.35 * night, 0, 255)
    # route
    if route_upto:
        n_legs = len(TL.TOUR) - 1
        r = min(float(route_upto), n_legs)
        for i in range(min(int(r) + 1, n_legs)):
            frac = 1.0 if i < int(r) else r - int(r)
            if frac <= 0:
                continue
            tt, pts = _leg_dots(i, 9.0)
            for t, (px_, py_) in zip(tt, pts):
                if t > frac:
                    break
                ox, oy = _ov(px_, py_)
                if 0 <= int(oy) < h and 0 <= int(ox) < w:
                    m[int(oy), int(ox)] = col('#f04f78')
    # tour-stop markers: gold star (found) / grey dot (unfound); lit at night
    for i, lid in enumerate(TL.TOUR):
        ox, oy = overview_xy(lid)
        ox -= x
        oy -= y
        fnd = lid in fs
        if fnd:
            pts = (((0, -1), '#f9c22b'), ((-1, 0), '#f9c22b'), ((1, 0), '#f9c22b'), ((0, 1), '#f7961d'),
                   ((0, 0), '#fbff86'))
            ring = ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (-2, 0), (2, 0), (0, 2))
        else:
            pts = (((0, 0), '#c8c0d0'), ((1, 0), '#9a90a8'), ((0, 1), '#9a90a8'), ((1, 1), '#7a7088'))
            ring = ((-1, 0), (-1, 1), (0, -1), (1, -1), (2, 0), (2, 1), (0, 2), (1, 2))
        for dx, dy in ring:
            if 0 <= oy + dy < h and 0 <= ox + dx < w:
                m[oy + dy, ox + dx] = col('#2e222f')
        for (dx, dy), cc in pts:
            if 0 <= oy + dy < h and 0 <= ox + dx < w:
                m[oy + dy, ox + dx] = col(cc)
        if fnd and night > 0.3 and (f // 8 + i) % 4 == 0:
            if 0 <= oy - 2 < h:
                m[oy - 2, ox] = col('#ffffff')
    # unroll reveal
    rows = int(round(h * clamp(reveal)))
    if rows > 0:
        c.a[y:y + rows, x:x + w] = m[:rows]
    if 0 < reveal < 1:
        ry = y + rows
        for dy, cc in ((-1, '#fff4d6'), (0, '#e8d0a0'), (1, '#c9a24a'), (2, '#8a6a3a'), (3, '#4a3426')):
            if 0 <= ry + dy < H:
                c.a[ry + dy, x - 2:x + w + 2] = col(cc)
    if reveal >= 0.999:
        # compass rose (west sea) + scale bar (south-east sea)
        _ov_compass(c.a, x + 9, y + 92)
        sbx, sby = x + w - 36, y + h - 34
        c.rect(sbx, sby, 26, 3, '#2e222f')
        for i in range(4):
            c.rect(sbx + 1 + i * 6, sby + 1, 6, 1, '#ffffff' if i % 2 == 0 else '#c9a24a')
        text(c, sbx + 1, sby - 6, '0', '#ffffff', font='tiny')
        text(c, sbx + 17, sby - 6, '500', '#ffffff', font='tiny')
        # labels
        if labels:
            frac = 1.0 if labels is True else float(labels)
            nl = int(round(len(_OV_LABELS) * clamp(frac)))
            for txt, nx, ny, cc in _OV_LABELS[:nl]:
                tw = text_width(txt, 'tiny')
                lx = int(clamp(x + int(nx * w) - tw // 2, x + 2, x + w - tw - 2))
                text(c, lx, y + int(ny * h) - 2, txt, cc, font='tiny', outline='#2e222f')
    return c
