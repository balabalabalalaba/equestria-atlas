"""The East Coast as an explorable diorama (art: scenes/s09_eastcoast.py, used read-only).

The film visits three different places, one per shot (cloud / glide transitions between them), so the
region has three areas, flown through left to right with a cloud sweep between them:
  0  FILLYDELPHIA (shot A): the brick plaza with the Big Bell in its white pavilion, Georgian rows, the
     clock spire, the domed hall and the glass towers behind.
  1  BALTIMARE (shot B): the harbour under the civic dome: warehouses, dock cranes, moored ships.
  2  HORSESHOE BAY (shot C): the sandy bay with the tall ship and the lighthouse, running south into
     the HAYSEED SWAMPS (stilt cabins, cypresses, lanterns).
Each area is the film's own shot renderer (`render_a/b/c`) bound to replacement globals (`rebind`): the
player drives the camera; the film's balloon, name tags and iris are switched off (the base class draws
the player's balloon and the tags; tags on slower layers ride them with Landmark.p). The clouds wrap
round instead of drifting away for good on the free-running scene clock. Timed film beats on game clocks:
  - the BELL HIT (film frames 82..179): the Big Bell swings up, strikes (sound rings, glint), echoes
    and settles, again every 12 s while you are in Fillydelphia (first strike ~2 s after arriving). The
    film's 1-px strike shake is left out (it would jolt the world under the player's balloon);
  - the ship's horn sounds when you arrive in Baltimare; foam waves wash up the bay on every beat
    (periodic in the film already), the cranes swing and the lighthouse beam turns on the scene frame.
Interaction (SPACE at a hotspot), each with the film's own helpers where it can:
  - BELL (Fillydelphia): the Big Bell swings up and strikes (the film's swing, sound rings and glint) and
    music notes float up out of the pavilion. It also rings once on landing, as in the film.
  - CRANE (Baltimare): the loaded dock crane luffs its jib up, hauling the crate high (it swings), then
    dips and sets the crate down on the quay (thud, dust) and hoists it back.
  - HORN (Baltimare): the white schooner sounds its horn: the film's sound rings off its deckhouse,
    ripples round the hull, and signal flags run up its rigging from bowsprit to stern for ~2 s.
  - LIGHTHOUSE (Horseshoe Bay): the lantern flares with a burst of sparks and the beam (brighter, with
    a bright core ray) spins round three times fast.
  - SAILS (Horseshoe Bay): the tall ship drops all nine square sails (a gold star on the main course),
    holds them, then furls them again; the sails are drawn into the ship so the bay reflects them.
Hidden stamps: above the bell pavilion's cupola, low by the pier at Baltimare's far end, and under the
boardwalk between the stilt cabins of the Hayseed Swamps.
World clock: the whole sky is the clock's at every hour (the film's 13:30 sun never stays up off the arc),
and the water's sun sparkles follow the clock's sun (moonlit ones by night). The lights are the region's
own (auto_glow off): by day nothing is lit (the lighthouse is out, the cabins' gold panes are glass, the
porch lanterns cold); from the dusk on the windows, lamp posts, office towers, clock faces, cabins,
lanterns, ships' riding lights and the lighthouse come on one by one, mirrored in the harbour and bayou.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, fx, seg, smooth, bayer, col, hash01
import scenes.s09_eastcoast as sc

from .. import ui, daylight, nightwin
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind
from .everfree import GlowKeep

_MAGENTA = np.array((255, 0, 255), np.uint8)


def _block(c, layers):
    """Stamp cached (rgb, mask) scenery layers' opaque pixels out of a sky hook (each placed as sc.comp
    places it at offset ox): one union mask, one write."""
    hit = np.zeros((H, W), bool)
    for (a, m), ox in layers:
        lh, lw = m.shape[:2]
        x0, x1 = max(0, ox), min(lw, ox + W)
        y1 = min(lh, H)
        if x1 > x0:
            hit[:y1, x0 - ox:x1 - ox] |= m[:y1, x0:x1]
    np.copyto(c.a, _MAGENTA, where=hit[..., None])


# camera limits (the largest camera with art on every layer):
#   A: fg railings 400 (p 1.35) -> 59; near 390 (p 1) -> 70; mid 380 (p .55) -> 109; glass 360 (p .3) -> 133
#   B: quay 380 (p .7) -> 86; fg pier 440 (p 1.35) -> 89; mid 360 (p .4) -> 100; far 340 (p .15) -> 133
#   C: fg reeds 560 (p 1.4) -> 171; swamp back 460 (p .75) -> 186; main 520 (p 1) -> 200
CAM_A, CAM_B, CAM_C = 59, 86, 171

BELL_REST = 236         # bell frame where the swing has died away (angle rounds to 0)
BELL_LEAD = 26          # bell frame 82 this many steps after landing (strike ~1.8 s in)

LANDMARKS_A = [         # the film's callouts, in world px (main = the plaza layer)
    Landmark('THE BIG BELL', sc.BELL_X, (sc.BELL_X - 14, sc.BELL_PIVOT + 10, 'left', 26)),
    Landmark('FILLYDELPHIA', sc.SPIRE_X + 22, (sc.SPIRE_X, 50, 'up', 14), p=sc.MID_P),
]
LANDMARKS_B = [         # the civic dome sits on the mid layer
    Landmark('BALTIMARE', sc.CIVIC_X + 24, (sc.CIVIC_X - 15, 58, 'left', 18), p=sc.B_MID_P),
]
LANDMARKS_C = [         # the bay tag hangs off the tall ship (ship layer); the swamp tag off the first cabin
    Landmark('HORSESHOE BAY', 200, (150 + 70, 118, 'down', 14), p=sc.C_SHIP_P),
    Landmark('HAYSEED SWAMPS', sc.CABINS[0][0] + 11, (sc.CABINS[0][0] + 11, 132 - 16 - 22, 'up', 18)),
]
BEACH_X = 160
SWAMP_X = sc.CABINS[0][0] + 40

# ------------------------------------------------------------------ interaction
CRANE_X = 178           # waterfront-layer x of the loaded crane's left leg (p .7)
SCHOONER = dict(x=150, L=84, seed=1, scale=1.15, wl=154)      # the white schooner (main layer)
TALL = dict(x=150, L=100, seed=3, scale=1.25, wl=114)         # the tall ship (layer p .6)
HOTSPOTS_A = [Hotspot('bell', sc.BELL_X, 70, 'bell', cooldown=110)]     # bubble over the pediment
HOTSPOTS_B = [
    Hotspot('crane', CRANE_X + 16, 102, 'gear', cooldown=100, p=sc.B_WF_P),    # bubble over the cab
    Hotspot('horn', SCHOONER['x'] + 10, 132, 'horn', cooldown=95),          # the stern deckhouse
]
HOTSPOTS_C = [
    Hotspot('lighthouse', sc.LH_X, 62, 'lantern', cooldown=100, p=sc.C_LH_P),  # bubble below the lantern
    Hotspot('sails', TALL['x'] + 57, 40, 'wind', cooldown=115, p=sc.C_SHIP_P),
]
STAMPS_A = [Stamp('cupola', sc.BELL_X, 28)]             # high above the bell pavilion's cupola
STAMPS_B = [Stamp('pier', 388, 124)]                    # low over the water at the harbour's far end
STAMPS_C = [Stamp('boardwalk', 372, 128)]               # under the boardwalk between the stilt cabins
BUNTING = ('hot_pink', 'butter', 'baby_blue', 'mint', 'white', 'lavender', 'gold')
SAIL = ('#b8a88e', '#d9cdb8', '#ece4d4', '#f8f3e8', '#fffdf6')
STAR = ('...x...', '..xox..', 'xxxoxxx', '.xxoxx.', '..xxx..', '.xx.xx.', '.x...x.')


def _nop(*_a, **_k):
    return None


def _clouds(c, f, cam, clouds, drift=0.12):
    """The film's draw_clouds, wrapped: a cloud that drifts off the left edge comes back in from the
    right (both off screen), so the sky never empties on the free-running clock."""
    for (x, y, w, h, seed, par) in clouds:
        sx = x - cam * par - f * drift * (0.6 + par * 3)
        span = W + w + 12
        sx = int(round((sx + w + 8) % span - (w + 8)))
        sc.comp(c, sc._cumulus(w, h, seed), -sx, -y)


def _beam(c, lx, ly, f, gain):
    """sc.lighthouse_beam with a brighter, longer wedge (gain 0..1) for the LIGHTHOUSE event."""
    phi = f * 2 * math.pi / 72.0
    reach, facing = math.cos(phi), math.sin(phi)
    if facing > 0.6:
        fx.glow(c, lx, ly, 16, '#fff6c0', strength=min(1.0, (0.55 + 0.4 * gain) * (facing - 0.6) * 2.5), rings=4)
    L = int(abs(reach) * (120 + 80 * gain))
    if L < 6:
        return
    x0, x1 = (lx, lx + L) if reach > 0 else (lx - L, lx)
    x0, x1 = max(0, x0), min(W, x1)
    if x1 <= x0:
        return
    y0 = max(0, ly - 30)
    yy, xx = np.mgrid[y0:min(sc.H, ly + 31), x0:x1]
    dist = np.abs(xx - lx) + 1
    m = np.abs(yy - ly - dist * 0.05) < 1.5 + dist * 0.12
    strength = (1 - dist / (L + 1)) * (0.55 + 0.4 * gain) * (0.4 + 0.6 * (1 - abs(facing)))
    m &= sc.bayer(m.shape[0], m.shape[1], x0, y0) < strength
    reg = c.a[y0:min(sc.H, ly + 31), x0:x1]
    reg[m] = (reg[m].astype(np.int32) + np.array(sc.col('#fffbe0'), np.int32) * 2) // 3
    core = (np.abs(yy - ly - dist * 0.05) < 0.6) & (dist < L * 0.8 * gain)       # the bright ray
    reg[core] = (reg[core].astype(np.int32) + np.array(sc.col('white'), np.int32) * 3) // 4


_LAMP_GLASS = np.array(sc.col('#fff3c4'), np.uint8)      # sc.lamp_post's glass


def _lamps_at_night(c, hour):
    """The film's lamp posts are painted unlit (pale glass) at 13:30: when the clock says night their
    glass (and its white glint) turns a warm lit yellow. Returns the mask of the lit glass (None by day),
    kept shining after the relight (GlowKeep)."""
    if daylight.night(hour) <= 0.3:
        return None
    from scipy import ndimage
    m = np.all(c.a == _LAMP_GLASS, axis=2)
    if not m.any():
        return None
    lab, n = ndimage.label(m)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    size[0] = 1 << 20
    glass = size[lab] <= 40                                  # small panes only, never a pale wall
    glint = np.all(c.a == 255, axis=2) & ndimage.binary_dilation(glass)
    lit = glass | glint
    c.a[lit] = _NB[0]
    return lit


_NB = np.array([sc.col(v) for v in ('#ffe27a', '#ffd060', '#f8c050', '#f0b050', '#c8843a')], np.uint8)


# ------------------------------------------------------------------ night lights (world clock)
# The film paints the coast at 13:30. At night the region keeps its REAL lights itself (auto_glow is off:
# the relight would otherwise keep painted gold trims, finials and the pediment emblem glowing): lit
# windows (game/nightwin.py), the lamp posts, the office windows of the glass towers and the far skyline,
# the clock faces, the swamp cabins' windows and porch lanterns, the lighthouse, and their reflections in
# the harbour. Each set is (ys, xs, painted rgb, threshold, shade) in its layer's px: a light goes on once
# nightwin.ramp passes its threshold (so they come on one by one through the dusk), where nothing covers it.
_GLASS_TOWERS = ((22, 46, 15), (92, 34, 18), (113, 54, 12), (286, 40, 20), (310, 58, 13))   # as sc._a_glass
OFFICE = np.array([col('#ffe7a0'), col('#fff3c8'), col('#ffd466')], np.uint8)
FACE = np.array([col('#fff2c8')], np.uint8)                 # a lit clock face


def _pts(ys, xs, rgb, gid, share, seed):
    """A light set: per pixel its threshold (hash of its group / share) and shade (0..2)."""
    gid = np.asarray(gid)
    ug, inv = np.unique(gid, return_inverse=True)
    v = np.array([hash01(int(g), seed) / share for g in ug], np.float32)[inv]
    kind = np.array([int(hash01(int(g), seed + 1) * 3) for g in ug], np.int64)[inv]
    return np.asarray(ys), np.asarray(xs), rgb, v, kind


@lru_cache(maxsize=1)
def _office_pts():
    """The glass towers' office windows (2 x 2 cells between the floor lines and mullions of the lit face,
    one cell across the shaded face) and the far skyline blocks' 1-px windows (sc._a_glass)."""
    a, m = sc._a_glass()
    cell = np.zeros(m.shape, bool)
    for x, top, w in _GLASS_TOWERS:
        lw = max(3, w // 3)
        for yy in range(top + 3, 129, 3):                  # between the floor lines (top + 2 + 3k)
            for xx in range(x + lw, x + w - 2, 3):          # between the mullions (x + lw + 2 + 3k)
                cell[yy:yy + 2, xx:xx + 2] = True
            cell[yy:yy + 2, x + 1:x + lw - 1] = True
    towers = np.zeros(m.shape, bool)
    for x, top, w in _GLASS_TOWERS:
        towers[top:130, x:x + w] = True
    sky_win = np.all(a == np.array(col(sc.hz('#e9f2fb', 0.3)), np.uint8), axis=2) & ~towers
    from scipy import ndimage
    lab = ndimage.label(cell & (m > 0))[0]                  # one cell, one office
    ys, xs = np.nonzero(lab)
    gid = lab[ys, xs]
    ys2, xs2 = np.nonzero(sky_win & (m > 0))
    gid2 = 10 ** 6 + ys2 * 1000 + xs2
    ys, xs, gid = np.r_[ys, ys2], np.r_[xs, xs2], np.r_[gid, gid2]
    return _pts(ys, xs, a[ys, xs].copy(), gid, 0.42, 611)


@lru_cache(maxsize=4)
def _face_pts(layer):
    """The clock faces of a scene layer (its '#fbf6e8' dial pixels): lit early in the dusk."""
    a, m = getattr(sc, layer)()
    ys, xs = np.nonzero(np.all(a == np.array(col('#fbf6e8'), np.uint8), axis=2) & (m > 0))
    return ys, xs, a[ys, xs].copy(), np.full(len(ys), 0.05, np.float32), np.zeros(len(ys), np.int64)


def _light_on(c, hour, pts, ox, cols=nightwin.WARM, oy=0):
    """Paint a light set (layer drawn at ox, oy) where the dusk ramp has switched it on and nothing covers
    it. Returns the mask of the painted pixels (None if none)."""
    k = nightwin.ramp(hour)
    if k <= 0:
        return None
    ys, xs, src, v, kind = pts
    sy, sx = ys + oy, xs + ox
    ok = (v < k) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    sy, sx, src, kind = sy[ok], sx[ok], src[ok], kind[ok]
    seen = np.all(c.a[sy, sx] == src, axis=1)
    if not seen.any():
        return None
    sy, sx = sy[seen], sx[seen]
    c.a[sy, sx] = cols[kind[seen] % len(cols)]
    m = np.zeros((H, W), bool)
    m[sy, sx] = True
    return m


def _union(*ms):
    out = None
    for m in ms:
        if m is not None:
            out = m.copy() if out is None else out | m
    return out


def _mirror(wl, f, amp):
    """sc.reflect's geometry for the water rows wl..H: (rows, depth 0..1, source row per row, source x
    per row and screen x)."""
    rows = np.arange(wl, H)
    depth = (rows - wl) / max(1, len(rows))
    srcr = np.clip(2 * wl - rows - 1 - (depth * 6).astype(int), 0, H - 1)
    dx = np.round(np.sin(rows * 1.3 + f * 0.33) * amp * (0.5 + depth)).astype(int)
    xs = np.clip(np.arange(W)[None, :] + dx[:, None], 0, W - 1)
    return rows, depth, srcr, xs


def _reflect_lights(c, lit, f, wl, still, amp, x0=0):
    """The lit windows and lamps mirrored into the water below wl, as sc.reflect mirrors the scene (same
    rows, same ripple): warm streaks broken into ripple lines (two rows of three, drifting down slowly),
    only on the water still showing (`still`: rows wl.. as drawn, nothing in front). Returns its mask."""
    if lit is None:
        return None
    rows, depth, srcr, xs = _mirror(wl, f, amp)
    hit = lit[srcr[:, None], xs] & still
    hit[:, :x0] = False
    hit &= ((rows - f // 6) % 3 != 0)[:, None]
    if not hit.any():
        return None
    reg = c.a[wl:]
    src = c.a[srcr[:, None], xs]
    k = np.where(depth < 0.5, 5, 4).astype(np.uint16)[:, None, None]       # fainter further out
    mixed = (src.astype(np.uint16) * k + reg.astype(np.uint16) * (8 - k)) >> 3
    reg[hit] = mixed[hit].astype(np.uint8)
    m = np.zeros((H, W), bool)
    m[wl:] = hit
    return m


def _sky_in_water(c, sky, refl, from_h, to_h):
    """The sky mirrored in the water in the clock's colours (after the sky repaint): sc.reflect mirrors
    the painted 13:30 sky, so a sunset or a dawn left a blue day sky in the water between the buildings'
    reflections. refl = (pure: water rows still showing sc.reflect's own pixels, f, wl, amp, k0, k1, tint,
    x0) as the area render recorded it; sky: the frame's sky mask."""
    pure, f, wl, amp, k0, k1, tint, x0 = refl
    rows, depth, srcr, xs = _mirror(wl, f, amp)
    m = pure & sky[srcr[:, None], xs]
    m[:, :x0] = False
    if not m.any():
        return
    t = np.array([[col(tint)]], np.uint8)            # the water's tint, lit for the clock
    daylight.relight(t, from_h, to_h, glow=False)
    k = (k0 + (k1 - k0) * depth)[:, None, None]
    ref = c.a[srcr[:, None], xs].astype(np.float32)
    out = ref * (1 - k) + t[0, 0].astype(np.float32) * k
    reg = c.a[wl:]
    reg[m] = np.clip(out[m] + 0.5, 0, 255).astype(np.uint8)


MOON_GLINT = (236, 232, 214)


def _glint_x(hour):
    """(sun x or None, moon x or None) for the water's sparkles: the sun's, on its arc, while it is up;
    the moon's in the night; none in the deep dusk."""
    kind, x, _y = daylight._body_pos(hour)
    n = daylight.night(hour)
    if kind == 'sun' and n < 0.25:
        return x, None
    return None, (x if kind == 'moon' and n > 0.5 else None)


def _moon_glints(c, f, mx, y0, y1, seed, x1=W):
    """A few moonlit sparkles twinkling on the water under the moon (screen x mx). Returns their mask
    (kept after the relight)."""
    m = np.zeros((H, W), bool)
    for i in range(12):
        x = mx - 22 + int(hash01(seed, i, 7) * 44)
        y = y0 + 2 + int(hash01(seed, i, 8) ** 1.4 * (y1 - y0 - 4))
        ph = ((f + hash01(seed, i, 9) * 36) % 36) / 36.0
        if ph < 0.3 and 0 <= x < min(W, x1) and 0 <= y < H:
            c.a[y, x] = MOON_GLINT
            m[y, x] = True
    return m


class _FxNight:
    """The fx module without its sunlit dust motes (the film's plaza motes are sunbeams)."""
    def __getattr__(self, name):
        return _nop if name == 'dust_motes' else getattr(fx, name)


_FX_NIGHT = _FxNight()


# the stilt cabins' windows (sc.stilt_cabin paints them lit gold, '#ffcf6a', at 13:30): glass by day,
# lit at night; rows top -> bottom of the day glass
_CABIN_GLASS = np.array([col(v) for v in ('#9cc8ee', '#5f8fca', '#5f8fca', '#3d5f9a', '#3d5f9a', '#e3f6ff')],
                        np.uint8)


@lru_cache(maxsize=1)
def _cabin_px():
    """(ys, xs, painted rgb, day-glass rgb, cabin) of the cabins' window panes on the bay's main layer."""
    a, m = sc._c_main()
    ys, xs = np.nonzero(np.all(a == np.array(col('#ffcf6a'), np.uint8), axis=2) & (m > 0))
    cab = (xs >= sc.CABINS[1][0]).astype(np.int64)
    tops = np.array([132 + sd * 4 - 16 - 12 for _x, sd in sc.CABINS])[cab]
    left = np.array([x + 3 for x, _sd in sc.CABINS])[cab]
    r, q = ys - tops, xs - left
    shade_ = np.where((q - r == 1) | (q - r == 2), 0, np.clip(r, 1, 4))     # a light streak across
    shade_ = np.where((r == 0) & (q == 0), 5, shade_)
    return ys, xs, a[ys, xs].copy(), _CABIN_GLASS[shade_], cab


def _cabin_windows(c, hour, cam):
    """The cabin windows: day glass (sky blue, a light streak) until the dusk ramp lights them (the left
    cabin first), then their painted gold. Returns the mask of the lit panes."""
    ys, xs, src, glass, cab = _cabin_px()
    sx = xs - cam
    ok = (sx >= 0) & (sx < W)
    ys, sx, src, glass, cab = ys[ok], sx[ok], src[ok], glass[ok], cab[ok]
    seen = np.all(c.a[ys, sx] == src, axis=1)
    ys, sx, src, glass, cab = ys[seen], sx[seen], src[seen], glass[seen], cab[seen]
    on = nightwin.ramp(hour) > np.array([0.08, 0.3])[cab]
    c.a[ys[~on], sx[~on]] = glass[~on]
    if not on.any():
        return None
    m = np.zeros((H, W), bool)
    m[ys[on], sx[on]] = True
    return m


SHIP_LAMP = (col('#ffe27a'), col('#fff6d0'))


def _ship_lights(c, kw, hour, out):
    """A moored ship's lights by night (drawn into the ship, so its reflection shows them too): a stern
    lantern on the taffrail and a riding light below the foremast's truck. Their boxes go to `out`."""
    if nightwin.ramp(hour) <= 0.1:
        return
    gm = _ship_geom(kw['x'], kw['wl'], kw['f'], kw.get('L', 70), kw.get('masts', 2), kw.get('seed', 0),
                    kw.get('scale', 1.0))
    (sx, sy), (tx, ty) = gm['stern'], gm['tips'][0]
    for x, y in ((sx, sy - 3), (tx, ty + 3)):
        c.px(x, y - 1, '#3a3248')
        c.rect(x - 1, y, 2, 2, SHIP_LAMP[0])
        c.px(x - 1, y, SHIP_LAMP[1])
        out.append((x - 1, y))


def _ship_mask(c, boxes):
    """The ship lights still seen (their colours where they were drawn: nothing in front)."""
    if not boxes:
        return None
    m = np.zeros((H, W), bool)
    for x, y in boxes:
        x0, x1, y0, y1 = max(0, x), min(W, x + 2), max(0, y), min(H, y + 2)
        if x1 > x0 and y1 > y0:
            sub = c.a[y0:y1, x0:x1]
            m[y0:y1, x0:x1] = np.all(sub == SHIP_LAMP[0], axis=2) | np.all(sub == SHIP_LAMP[1], axis=2)
    return m


_LAMP_ROOM = np.array([col('#fff6c8'), col('#ffeeb0')], np.uint8)


@lru_cache(maxsize=1)
def _lamp_room_pts():
    """The lighthouse's lantern-room glass ('#fff3b0'): lit with the beam."""
    a, m = sc._c_lighthouse()
    ys, xs = np.nonzero(np.all(a == np.array(col('#fff3b0'), np.uint8), axis=2) & (m > 0))
    return ys, xs, a[ys, xs].copy(), np.zeros(len(ys), np.float32), (ys + xs) % 2


def _lanterns(c, f, cam, hour, out):
    """sc.lanterns (the cabins' porch lanterns and stove smoke) by the world clock: by day a cold lantern
    (pale glass, no glow); from the dusk on lit, a flickering hot core, no halo. The lit glass is added to
    `out` (a list) as (mask, frame just after) to keep it shining after the relight."""
    lit = nightwin.ramp(hour) > 0.04
    m = np.zeros((H, W), bool)
    for (cx_, sd) in sc.CABINS:
        lx = cx_ + 26 - cam
        ly = 132 + sd * 4 - 16 - 9
        if -10 < lx < W + 10:
            c.px(lx, ly - 2, '#3a3248')
            if lit:
                c.rect(lx - 1, ly - 1, 3, 3, '#ffe9a0')
                c.px(lx, ly, 'white' if hash01(sd, f // 3) < 0.7 else '#fff3c0')
                c.px(lx + (1 if hash01(sd, f // 4, 2) < 0.5 else -1), ly + 1, '#ffcf6a')
                m[max(0, ly - 1):ly + 2, max(0, lx - 1):max(0, lx + 2)] = True
            else:
                c.rect(lx - 1, ly - 1, 3, 3, '#d9d2b8')
                c.px(lx - 1, ly - 1, '#f4f0e2')
        sx_ = cx_ + 18 - cam
        if -10 < sx_ < W + 10:
            sc.smoke(c, sx_, 132 + sd * 4 - 16 - 32, f, seed=40 + sd, n=4, height=16, color='#c9d2cc')
    if m.any():
        out.append((m, c.a.copy()))


def _night_light(c, lx, ly, f, gain, allow, k=1.0, halo=None):
    """The lighthouse at night, drawn onto the re-lit night picture (Region.after_light), the way
    Vanhoover's is painted: a dithered wedge of warm white that widens and fades as it sweeps
    the bay (longer while the SPACE turns run, gain), a brighter core, and a soft glow round the lamp
    that flares when the beam faces us. `allow`: the pixels it may light (nothing drawn in front of the
    light since); k: how dark it is (0..1, dusk to night); halo: where the lamp's glow may show (the
    sky and the lamp room: no dithered halo over the tower and the rocks)."""
    phi = f * 2 * math.pi / 72.0
    d = math.cos(phi)
    L = int(abs(d) * (150 + 60 * gain))
    if L > 4:
        sgn = 1 if d > 0 else -1
        x0, x1 = (lx, lx + L) if sgn > 0 else (lx - L, lx)
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, ly - 16), min(H, ly + 17)
        if x1 > x0 and y1 > y0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = np.abs(xx - lx).astype(np.float32)
            half = 0.6 + dist * 0.07
            dy = np.abs(yy - ly)
            inside = (dy <= half) & allow[y0:y1, x0:x1]
            inten = np.clip(1 - dist / max(1, L), 0, 1) ** 1.3 * (0.55 + 0.45 * abs(d)) * k
            inten = inten * (1.15 - 0.6 * dy / np.maximum(half, 1))
            m = inside & (bayer(y1 - y0, x1 - x0, x0, y0) < inten)
            reg = c.a[y0:y1, x0:x1].astype(np.float32)
            beam = np.array([255, 246, 214], np.float32)     # mixed toward warm white (an added fixed
            reg[m] += (beam - reg[m]) * 0.42                  # amount reads as a grey band on the night)
            core = inside & (dy <= np.maximum(0.6, half * 0.3)) & (dist < L * 0.65)
            reg[core] += (beam - reg[core]) * 0.55 * k
            c.a[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    flare = 1 - min(1, abs(d) * 3)                  # the beam facing us: the lamp blazes
    r = 10 + int(10 * flare)
    y0, y1, x0, x1 = max(0, ly - r), min(H, ly + r + 1), max(0, lx - r), min(W, lx + r + 1)
    if y1 > y0 and x1 > x0:
        before = c.a[y0:y1, x0:x1].copy()
        fx.glow(c, lx, ly, r, '#fff2c0', (0.30 + 0.35 * flare) * k, rings=4)
        fx.glow(c, lx, ly, 5 + int(4 * flare), '#ffe6a0', (0.45 + 0.4 * flare) * k, rings=3)
        if halo is not None:
            off = ~halo[y0:y1, x0:x1]
            c.a[y0:y1, x0:x1][off] = before[off]
    c.px(lx, ly, 'white')


def _lamp_flash(c, a, lx, ly):
    """SPACE at night: the lamp flashes three times (rings and a white star), as at Vanhoover."""
    for p0 in (0, 14, 28):
        q = a - p0
        if not 0 <= q < 16:
            continue
        kk = 1 - q / 16
        fx.glow(c, lx, ly, 14 + q * 2, '#fff2c0', 0.85 * kk, rings=5)
        c.circle(lx, ly, 6 + q * 3, '#ffe6a0' if q < 8 else '#c8b890', fill=False)
        L = 16 - q
        if L > 2:
            c.hline(lx - L, lx + L, ly, 'white')
            c.vline(lx, ly - L // 2, ly + L // 2, 'white')
            for j in range(2, L // 3 + 1):
                for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    c.px(lx + sx_ * j, ly + sy_ * j, '#fff6d0')


def _ship_geom(x, wl, f, L, masts, seed, scale):
    """The geometry sc.sailing_ship uses (same maths): hull top line, masts, yards, rigging points."""
    ph = seed * 1.7
    bob = int(round(math.sin(f * 2 * math.pi / 60.0 + ph) * 1.0))
    lean = math.sin(f * 2 * math.pi / 60.0 + ph + 0.9) * 1.6
    wy = wl + bob
    hh = int(10 * scale)
    top = [wy - hh - int(round(3 * scale * ((1 - i / L) ** 2 * 0.8 + (i / L) ** 3 * 1.6))) for i in range(L + 1)]
    mxs = [x + int(L * (0.3 + 0.55 * k / max(1, masts - 1))) for k in range(masts)] if masts > 1 else [x + L // 2]
    mh = [int((40 + 8 * (k == masts // 2)) * scale) for k in range(masts)]
    bases = [top[mx - x] - 1 for mx in mxs]
    tips = [(mx + int(round(lean * mh[k] / 40.0)), bases[k] - mh[k]) for k, mx in enumerate(mxs)]
    bx0, by0 = x + L + 2, top[L] - 2
    bow = (bx0 + int(14 * scale), by0 - int(5 * scale))
    return dict(wy=wy, hh=hh, top=top, lean=lean, mxs=mxs, mh=mh, bases=bases, tips=tips, bow=bow,
                stern=(x + 2, top[0] - 2))


def _sails(c, gm, scale, a):
    """Square sails dropped from the tall ship's yards (a = steps since SPACE; 0 = furled)."""
    lean = gm['lean']
    for k, mx in enumerate(gm['mxs']):
        base, m = gm['bases'][k], gm['mh'][k]
        yards = [(int(base - m * yf), int((11 - j * 3) * scale)) for j, yf in enumerate((0.3, 0.55, 0.78))]
        for j, (yy, half) in enumerate(yards):
            d = k * 4 + (2 - j) * 3                  # the topsails drop first, mast by mast
            u = smooth(seg(a, d, d + 14)) * (1 - smooth(seg(a, 78 + d, 92 + d)))
            if u <= 0:
                continue
            y_bot = yards[j - 1][0] - 1 if j else base - 4
            full = y_bot - (yy + 2)
            hgt = int(round(full * u))
            bh = yards[j - 1][1] if j else half + 2
            for r in range(hgt):
                y = yy + 2 + r
                q = r / max(1, full)
                hw = half + (bh - half) * q
                xc = mx + lean * (base - y) / 40.0
                x0, x1 = int(round(xc - hw)), int(round(xc + hw))
                c.hline(x0, x1, y, SAIL[3] if r > 1 else SAIL[1])
                c.hline(x0, x0 + max(1, (x1 - x0) // 3), y, SAIL[2] if r > 1 else SAIL[1])
                c.px(x1 - 1, y, SAIL[4])
                c.px(x0, y, SAIL[0])
                c.px(x1, y, SAIL[2])
                for sx in range(x0 + 3, x1 - 1, 4):  # cloth panels
                    if (sx + y) % 5:
                        c.px(sx, y, SAIL[2])
            if hgt >= 3:                             # the foot bellies a little, then a shadow line
                y = yy + 2 + hgt
                xc = mx + lean * (base - y) / 40.0
                hw = half + (bh - half) * min(1.0, hgt / max(1, full))
                c.hline(int(round(xc - hw * 0.6)), int(round(xc + hw * 0.6)), y, SAIL[1])
                c.hline(int(round(xc - hw * 0.3)), int(round(xc + hw * 0.3)), y + 1, SAIL[0])
                if k == 1 and j == 0 and u > 0.95:   # a gold star on the main course
                    sy = yy + 2 + hgt // 2 - 3
                    sx = int(round(mx + lean * (base - sy) / 40.0)) - 3
                    for r_, row in enumerate(STAR):
                        for q_, ch in enumerate(row):
                            if ch != '.':
                                c.px(sx + q_, sy + r_, 'gold' if ch == 'x' else '#fff1a8')


def _bunting(c, gm, a, f):
    """Signal flags running up the schooner's rigging (stern -> masts -> bowsprit), held, then lowered."""
    pts = [gm['stern']] + gm['tips'] + [gm['bow']]
    flags = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 4))
        for i in range(n):
            u = (i + 0.5) / n
            flags.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
    k = smooth(seg(a, 4, 26)) * (1 - smooth(seg(a, 68, 86)))
    m = int(round(len(flags) * k))
    for i, (x, y) in enumerate(flags[::-1][:m]):     # they run up from the bowsprit
        x, y = int(round(x)), int(round(y))
        cc = BUNTING[i % len(BUNTING)]
        dx = int(round(math.sin(f * 0.5 + i * 0.8) * 0.7))
        c.hline(x - 1, x + 1, y + 1, cc)
        c.hline(x - 1 + (dx > 0), x + (dx >= 0), y + 2, cc)
        c.px(x + dx, y + 3, cc)


def _crane_up(a):
    """CRANE event, part 1 (0..1): the jib luffs up and hauls the crate up under its tip."""
    return smooth(seg(a, 0, 20)) * (1 - smooth(seg(a, 30, 52))) if 0 <= a < 90 else 0.0


def _crane_low(a):
    """CRANE event, part 2 (0..1): the crate is lowered onto the quay (lands at 54), then hoisted back."""
    return smooth(seg(a, 30, 54)) * (1 - smooth(seg(a, 62, 90))) if 0 <= a < 90 else 0.0


def _crane(c, x, gy, f, a):
    """sc.crane_dyn for the loaded crane (phase 0) plus the CRANE event (a = steps since SPACE): the jib
    luffs up hauling the crate high, it swings, then the jib dips and sets the crate down on the quay
    (thud, dust) and hoists it back up. Identical to the film's crane outside the event."""
    up, low = _crane_up(a), _crane_low(a)
    px_, py_ = x + 18, gy - 40
    ang = math.radians(28 + 9 * math.sin(f * 2 * math.pi / 144.0) + 24 * up - 5 * low)
    L = 46
    tx = px_ + math.cos(ang) * L
    ty = py_ - math.sin(ang) * L
    sc.lattice(c, px_, py_, tx, ty, 4, sc.CRANE)
    c.line(x + 6, gy - 52, int(tx), int(ty), '#5a5272')
    c.vline(x + 6, gy - 54, gy - 46, sc.CRANE[1])
    tx, ty = int(round(tx)), int(round(ty))
    c.px(tx, ty, '#3a3248')
    base = 18 + 8 * math.sin(f * 2 * math.pi / 96.0)
    drop = int(round(base + (5 - base) * up + (gy - 12 - ty - base) * low))
    sw = math.sin(f * 2 * math.pi / 40.0) * 1.2 * (1 - max(up, low))
    if 16 <= a < 44:                                   # swinging high under the tip
        sw += 4.0 * math.sin((a - 16) * 2 * math.pi / 16) * (1 - seg(a, 16, 44))
    if 64 <= a < 92:                                   # and again as it leaves the quay
        sw += 2.5 * math.sin((a - 64) * 2 * math.pi / 22) * (1 - seg(a, 64, 92))
    sw = int(round(sw))
    c.line(tx, ty + 1, tx + sw, ty + drop, '#4a4e5c')
    hx, hy = tx + sw, ty + drop
    c.px(hx, hy + 1, '#9aa3b4')
    c.line(hx, hy + 1, hx - 4, hy + 4, '#4a4e5c')
    c.line(hx, hy + 1, hx + 4, hy + 4, '#4a4e5c')
    sc.crate(c, hx - 5, hy + 12, 11, 8, '#d4a852')
    c.px(hx - 3, hy + 7, '#323353')
    if 54 <= a < 72:                                   # dust puffs where it lands
        e = a - 54
        for s_ in (-1, 1):
            for j in range(3):
                dx = hx + s_ * (7 + e // 2 + j * 2)
                dy = gy - 1 - (e * (j + 1)) // 6
                if (e + j) % 5 < 4 - e // 6:
                    c.px(dx, dy, ('#efe9ee', '#d8d0d8', 'white')[j])
                    if e < 8:
                        c.px(dx + s_, dy, '#efe9ee')


_rgba = nightwin.from_mask


class EastCoast(GlowKeep, Region):
    SEG = 'eastcoast'
    CARD = ('COASTAL CITIES', ['BRICK - BELLS - HARBOURS', 'ON THE CELESTIAL SEA'])

    def build_areas(self):
        return [
            Area(self.render_philly, cam_x=(0, CAM_A), y_range=(34, 140), start=(124, 66),
                 landmarks=LANDMARKS_A, balloon='m', hotspots=HOTSPOTS_A, stamps=STAMPS_A,
                 sky=self.sky_philly),
            Area(self.render_harbour, cam_x=(0, CAM_B), y_range=(34, 140), start=(70, 70),
                 landmarks=LANDMARKS_B, balloon='m', hotspots=HOTSPOTS_B, stamps=STAMPS_B,
                 sky=self.sky_harbour),
            Area(self.render_bay, cam_x=(0, CAM_C), y_range=(34, 140), start=(112, 62),
                 landmarks=LANDMARKS_C, balloon='m', hotspots=HOTSPOTS_C, stamps=STAMPS_C,
                 sky=self.sky_bay),
        ]

    # -------------------------------------------------------------- clocks + sound
    def init_state(self):
        self.bell_t0 = -10 ** 6     # step at which the bell clock was at frame 72 (82 = swing starts)

    def on_enter(self):
        """The film's reveal: the Big Bell swings and strikes just after landing (counts as a ring, so
        a SPACE right away waits for the swing to settle)."""
        t = self.g.t
        self.bell_t0 = t + BELL_LEAD - 10
        self.spot_t['bell'] = t + BELL_LEAD

    def on_area(self, ai):
        if ai == 1:
            self.g.sfx('horn', 0.5)                 # the film's ship horn on the cut to Baltimare

    def bell_f(self, t):
        """The bell's film frame: 72..82 at rest, 82 swing starts, 108 strike, 126 echo, 236 still."""
        return min(BELL_REST, max(72, 72 + t - self.bell_t0))

    def spot_a(self, key, t):
        """Steps since hotspot `key` was set off (large if never)."""
        return t - self.spot_t.get(key, -10 ** 6)

    def on_interact(self, key, t):
        g = self.g
        if key == 'bell':
            self.bell_t0 = t - 10                   # swing starts now; the sounds follow the bell clock
        elif key == 'crane':
            g.sfx('creak', 0.35)
        elif key == 'horn':
            g.sfx('horn', 0.7)
        elif key == 'lighthouse':
            g.sfx('shimmer', 0.5)
            g.sfx('whoosh', 0.3)
        elif key == 'sails':
            g.sfx('banner', 0.5)
            g.sfx('creak', 0.25)

    def update(self, t):
        g = self.g
        if t - self.t_enter == 6:
            g.sfx('wind', 0.3)
        if self.switch is not None:
            return
        if self.ai == 0:
            bf = self.bell_f(t)
            gain = self.near_gain(sc.BELL_X)
            if bf == 88:
                g.sfx('creak', 0.22 * gain)
            elif bf == 108:
                g.sfx('bell', 0.7 * gain)
            elif bf == 126:
                g.sfx('bell', 0.25 * gain)
            elif bf == 150:
                g.sfx('creak', 0.12 * gain)
            if t % 300 == 150:
                g.sfx('fountain', 0.15 * self.near_gain(318))
        elif self.ai == 1:
            f = self.scene_f(t)
            a = self.spot_a('crane', t)
            if a == 18:
                g.sfx('clank', 0.3)
            elif a == 32:
                g.sfx('creak', 0.25)
            elif a == 54:
                g.sfx('thud', 0.5)
            elif a == 64:
                g.sfx('clank', 0.3)
            if self.spot_a('horn', t) == 8:
                g.sfx('banner', 0.35)
            if t % 240 == 60:
                g.sfx('waves', 0.18)
            if f % 144 == 20:
                g.sfx('creak', 0.16 * self.near_gain(178))
            elif f % 144 == 92:
                g.sfx('clank', 0.16 * self.near_gain(178))
            if t % 360 == 200:
                g.sfx('wind', 0.2)
        else:
            if self.spot_a('sails', t) == 84:
                g.sfx('banner', 0.3)
            if self.spot_a('lighthouse', t) == 30:
                g.sfx('shimmer', 0.25)
            if t % 144 == 0:
                g.sfx('waves', 0.32 * self.near_gain(BEACH_X))
            if t % 400 == 130:
                g.sfx('creak', 0.14 * self.near_gain(220))
            if t % 420 == 300:
                g.sfx('wind', 0.2)

    # -------------------------------------------------------------- the world clock's light
    def auto_glow(self):
        """The relight keeps no painted spot glowing here (the gold pediment emblem, finials, the tall
        ship's gilt stripe and the cabins' orange door are paint): the real lights are kept by mask."""
        return False

    def _keep(self, c, *masks, k=0.95):
        self.keep_glow(c, _union(*masks), k)

    def _done(self, c, f, cx, cy):
        """End of an area render: the frame as rendered, for the sky repaint (after_light)."""
        self._raw = (c.a.copy(), f, cx, cy)
        return c

    def _repaint_sky(self, c, t):
        """The whole open sky in the clock's colours at every hour (also near the painted one). The base
        keeps the painted sky while the clock's sky looks alike (sky_weight < 1), which here means the
        film's sun (high on the right at 13:30) and its glow stay up from ~11.9 to ~15.4 h, off the arc
        the clock's sun follows, and the sun jumps when the clock leaves that span. Returns this frame's
        sky mask (None if unknown)."""
        raw, self._raw = getattr(self, '_raw', None), None
        A = self.area
        if raw is None or A.sky is None or (self.ID, self.ai) in daylight.INTERIORS:
            return None
        a, f, ix, iy = raw
        key = (self.ai, t, ix, iy)                       # the base's memo: once per step
        if self._sky_memo is None or self._sky_memo[0] != key:
            self._sky_memo = (key, daylight.sky_pixels(a, A.sky(f, ix, iy).a))
        mask = self._sky_memo[1]
        hour = self.g.hour
        if mask.any() and daylight.sky_weight(self.HOUR, hour) < 0.999:
            y0, y1 = A.cam_y
            daylight.paint_sky(c.a, mask, hour, 1.0, ix, iy, y0 - 10, y1 - y0 + 130, t)
        return mask

    # -------------------------------------------------------------- the skies alone
    # the film's screen-fixed sky (gradient, sun disc and its glows), as each render draws it first;
    # clouds are scenery (game/daylight.py repaints the sky for the world clock's hour). The sun's wide
    # pale glow is close to the white pavilion, the glass towers and the ships' hulls, so every opaque
    # scenery layer is stamped out of the hook (magenta: never sky), at its render offset.
    def sky_philly(self, f, cx, cy):
        c = sc._sky().copy()
        _block(c, [(layer, int(round(cx * p))) for layer, p in (
            (sc._a_fields(), sc.FIELD_P), (sc._a_glass(), sc.GLASS_P), (sc._a_mid(), sc.MID_P),
            (sc._a_near(), 1.0), (sc._a_fg(), sc.FG_P))])
        return c

    def sky_harbour(self, f, cx, cy):
        c = sc._sky().copy()
        _block(c, [(layer, int(round(cx * p))) for layer, p in (
            (sc._b_far(), sc.B_FAR_P), (sc._b_mid(), sc.B_MID_P), (sc._b_wf(), sc.B_WF_P),
            (sc._b_fg(), sc.B_FG_P))])
        c.a[sc.WL:] = _MAGENTA                          # the harbour water, its ships and reflections
        return c

    def sky_bay(self, f, cx, cy):
        c = sc._sky(0.12).copy()
        _block(c, [(layer, int(round(cx * p))) for layer, p in (
            (sc._c_sea(), 0.0), (sc._c_far(), sc.C_FAR_P), (sc._c_lighthouse(), sc.C_LH_P),
            (sc._c_swamp_back(), sc.C_SWB_P), (sc._c_main(), 1.0), (sc._c_palms()[0], 1.0),
            (sc._c_fg(), sc.C_FG_P))])
        return c

    # -------------------------------------------------------------- area 0: Fillydelphia
    def render_philly(self, f, cx, cy):
        self._water = None                               # (the sky in its water: harbour, bayou)
        bf = self.bell_f(self.g.t)
        hour = self.g.hour
        sun = daylight.night(hour) < 0.25           # the sun's glint on the glass and its dusty beams
        render = rebind(sc.render_a, a_cam=lambda _f: cx, a_balloon=_nop, callout=_nop, iris=_nop,
                        draw_clouds=_clouds, tower_glints=sc.tower_glints if sun else _nop,
                        fx=fx if sun else _FX_NIGHT,
                        draw_bell=lambda c, x, py, _f: sc.draw_bell(c, x, py, bf),
                        sound_rings=lambda c, x, y, _f, t0, s=1.0, n=3: sc.sound_rings(c, x, y, bf, t0, s, n))
        c = render(f)
        bx = sc.BELL_X - cx
        if 108 <= bf <= 112:                     # the glint on the strike
            fx.sparkle(c, bx + 5, sc.BELL_PIVOT + 12, bf - 108, 'white', period=10, size=3)
        if 108 <= bf < 150:                      # a shower of gold motes off the strike
            ui.burst(c, bf, 108, bx, sc.BELL_PIVOT + 30, colors=('gold', 'butter', 'white'), n=18, seed=3,
                     power=0.7, life=40, gravity=0.03, size=1)
        for i, (ox, d) in enumerate(((-20, 0), (17, 7), (-3, 15))):  # notes float up out of the pavilion
            a = bf - 108 - d
            if 0 <= a < 48 and not (a > 36 and a % 4 < 2):
                x = bx + ox + int(round(math.sin(a * 0.18 + i * 2) * 3))
                ui.icon(c, 'note', x - 4, sc.BELL_PIVOT + 2 - int(a * 1.1))
        mo, go = int(round(cx * sc.MID_P)), int(round(cx * sc.GLASS_P))
        self._keep(c, _lamps_at_night(c, hour),                     # night: the town lights up
                   nightwin.paint(c, hour, self.g.t, (('ec_a_mid', _rgba(sc._a_mid), -mo, 0),
                                                      ('ec_a_near', _rgba(sc._a_near), -cx, 0))),
                   _light_on(c, hour, _office_pts(), -go, OFFICE),
                   _light_on(c, hour, _face_pts('_a_mid'), -mo, FACE))
        return self._done(c, f, cx, cy)

    # -------------------------------------------------------------- area 1: Baltimare
    def render_harbour(self, f, cx, cy):
        self._water = None                               # (the sky in its water: harbour, bayou)
        t = self.g.t
        ac = self.spot_a('crane', t)
        ah = self.spot_a('horn', t)

        def crane_dyn(c, x, gy, f_, phase=0.0, load=True):
            if load:
                _crane(c, x, gy, f_, ac)
            else:
                sc.crane_dyn(c, x, gy, f_, phase, load)

        hour = self.g.hour
        ships = []

        def sailing_ship(c, **kw):                       # called as sailing_ship(tmp, wl=, f=, x=, ...)
            box = sc.sailing_ship(c, **kw)
            if kw.get('seed') == SCHOONER['seed'] and 0 <= ah < 90:
                gm = _ship_geom(kw['x'], kw['wl'], kw['f'], kw['L'], kw.get('masts', 2), kw['seed'], kw['scale'])
                _bunting(c, gm, ah, kw['f'])
            _ship_lights(c, kw, hour, ships)
            return box

        swr0 = rebind(sc.ship_with_reflection, sailing_ship=sailing_ship)

        def ship_with_reflection(c, f_, wl, tint, **kw):
            swr0(c, f_, wl, tint, **kw)
            if kw.get('seed') == SCHOONER['seed'] and 0 <= ah < 60:
                self._horn_fx(c, f_, wl, kw['x'], ah)

        sunx, moonx = _glint_x(hour)
        water = {}

        def reflect(c, wl, y1, f_, tint, **kw):          # the harbour mirror (its own pixels, for the sky)
            sc.reflect(c, wl, y1, f_, tint, **kw)
            water['refl'] = c.a[wl:].copy()

        def water_glints(c, wl, y1, f_, cam, seed=0):  # the sun's sparkles where the clock's sun is
            sc.water_glints(c, wl, y1, f_, cam, seed, sunx=-500 if sunx is None else sunx)
            if moonx is not None:
                water['moon'] = _moon_glints(c, f_, moonx, wl, y1, seed + 3)
            water['drawn'] = c.a[wl:].copy()            # the water as drawn (what is in front differs)

        render = rebind(sc.render_b, b_cam=lambda _f: cx, b_balloon=_nop, callout=_nop, draw_clouds=_clouds,
                        crane_dyn=crane_dyn, ship_with_reflection=ship_with_reflection, water_glints=water_glints,
                        reflect=reflect)
        c = render(f)
        mo = int(round(cx * sc.B_MID_P))
        lit = _union(_lamps_at_night(c, hour),
                     nightwin.paint(c, hour, t, (('ec_b_mid', _rgba(sc._b_mid), -mo, 0),
                                                 ('ec_b_wf', _rgba(sc._b_wf), -int(round(cx * sc.B_WF_P)), 0))),
                     _light_on(c, hour, _face_pts('_b_mid'), -mo, FACE), _ship_mask(c, ships))
        refl = glints = None
        if 'refl' in water:                              # sc.reflect's sky, for after_light
            self._water = (np.all(c.a[sc.WL:] == water['refl'], axis=2), f, sc.WL, 1.2, 0.48, 0.72,
                           sc.WATER[1], 0)
        if 'drawn' in water:
            still = np.all(c.a[sc.WL:] == water['drawn'], axis=2)
            if 'moon' in water:
                glints = water['moon'] & np.all(c.a == np.array(MOON_GLINT, np.uint8), axis=2)
            refl = _reflect_lights(c, lit, f, sc.WL, still, 1.2)     # the lights in the harbour
        self._keep(c, lit, refl, glints)
        return self._done(c, f, cx, cy)

    @staticmethod
    def _horn_fx(c, f, wl, x, a):
        """The HORN: sound rings off the stern deckhouse and ripples spreading round the hull."""
        S = SCHOONER
        gm = _ship_geom(x, wl, f, S['L'], 2, S['seed'], S['scale'])
        sc.sound_rings(c, x + 9, gm['top'][3] - 8, a, 0, 0.9, 3)          # off the deckhouse
        cxr, wy = x + S['L'] // 2, gm['wy'] + 1
        for i in range(3):
            e = a - i * 8
            if not 0 <= e < 40:
                continue
            rx, ry = S['L'] * 0.55 + e * 1.6, 2 + e * 0.22
            n = int(rx * 1.6)
            ang = np.linspace(0.05, math.pi - 0.05, n)
            xs = np.round(cxr + rx * np.cos(ang)).astype(int)
            ys = np.round(wy + ry * np.sin(ang)).astype(int)
            keep = (xs >= 0) & (xs < W) & (ys < sc.H) & ((np.arange(n) + e // 3) % 5 != 0)
            if e > 26:
                keep &= (np.arange(n) % 2 == 0)
            c.a[ys[keep], xs[keep]] = sc.C('white' if i == 0 else '#e8f8ff')

    # -------------------------------------------------------------- area 2: Horseshoe Bay
    def _lighthouse(self, c, x, y, f, fb, gain):
        """The lighthouse light by the world clock. At night it is drawn after the relight (after_light):
        here only where the lamp is and what has been drawn so far (what comes later is in front of the
        light). By day the lamp is out (no searchlight at noon): only the SPACE event turns the beam."""
        n = daylight.night(self.g.hour)
        if n >= 0.3:
            self._lh = (x, y, fb, gain, c.a.copy())
        elif gain > 0:
            _beam(c, x, y, fb, gain)

    def render_bay(self, f, cx, cy):
        self._water = None                               # (the sky in its water: harbour, bayou)
        t = self.g.t
        al = self.spot_a('lighthouse', t)
        asl = self.spot_a('sails', t)
        fb = f + (int(round(216 * smooth(seg(al, 0, 84)))) if 0 <= al < 84 else 0)    # 3 fast turns
        gain = (smooth(seg(al, 0, 8)) * (1 - smooth(seg(al, 70, 90)))) if 0 <= al < 90 else 0.0

        hour = self.g.hour
        ships = []

        def sailing_ship(c, **kw):                       # called as sailing_ship(tmp, wl=, f=, x=, ...)
            box = sc.sailing_ship(c, **kw)
            if kw.get('yards') and 0 <= asl < 110:
                gm = _ship_geom(kw['x'], kw['wl'], kw['f'], kw['L'], kw['masts'], kw['seed'], kw['scale'])
                _sails(c, gm, kw['scale'], asl)
            _ship_lights(c, kw, hour, ships)
            return box

        sunx, moonx = _glint_x(hour)
        lamps, glints = [], []

        def c_glints(c, f_, cam):                       # the sun's path sparkles under the clock's sun
            rebind(sc.c_glints, SUN=(-500 if sunx is None else sunx, 0))(c, f_, cam)
            if moonx is not None:
                glints.append(_moon_glints(c, f_, moonx, sc.HORIZON, sc.HORIZON + 36, 17,
                                           x1=sc.SWAMP_X0 - 120 - cam))

        bayou = {}

        def reflect(c, wl, y1, f_, tint, x0=0, **kw):   # the bayou's mirror (its own pixels, for the sky)
            sc.reflect(c, wl, y1, f_, tint, x0=x0, **kw)
            bayou['refl'] = (c.a[wl:].copy(), wl, x0, kw.get('amp', 1.0), kw.get('k0', 0.38), kw.get('k1', 0.62),
                             tint)

        render = rebind(sc.render_c, c_cam=lambda _f: cx, c_balloon=_nop, callout=_nop, draw_clouds=_clouds,
                        lighthouse_beam=lambda c, x, y, _f: self._lighthouse(c, x, y, f, fb, gain),
                        ship_with_reflection=rebind(sc.ship_with_reflection, sailing_ship=sailing_ship),
                        c_glints=c_glints, lanterns=lambda c, f_, cam: _lanterns(c, f_, cam, hour, lamps),
                        bayou_water=rebind(sc.bayou_water, reflect=reflect))
        self._lh = self._lh_post = None
        c = render(f)
        night = daylight.night(hour) >= 0.3
        if self._lh is not None:                         # night: the light is drawn after the relight
            x, y, fb_, gain_, snap = self._lh
            self._lh_post = (x, y, fb_, gain_, np.all(c.a == snap, axis=2), al)
            self._lh = None
        if 0 <= al < 90 and not night:                   # the lantern flares (by night: after_light)
            lx, ly = sc.LH_X - int(round(cx * sc.C_LH_P)), 44 - 9
            k = 1 - seg(al, 10, 90)
            fx.glow(c, lx, ly, 22, '#fff6c0', strength=0.7 * k, rings=5)
            fx.glow(c, lx, ly, 8, 'white', strength=0.8 * k, rings=3)
            ui.burst(c, al, 0, lx, ly, colors=('white', 'butter', 'gold'), n=22, seed=7, power=0.9, life=40,
                     gravity=0.02)
            if al < 60:
                fx.sparkles(c, lx, ly, f, radius=14, n=6, colors=('white', 'butter'), seed=9, period=14)
        lo = int(round(cx * sc.C_LH_P))                  # night: the lit windows, lanterns, lamp room
        cabins = _union(_cabin_windows(c, hour, cx),
                        *[m & np.all(c.a == snap, axis=2) for m, snap in lamps])   # (nothing over them since)
        lit = [cabins, _ship_mask(c, ships),
               nightwin.paint(c, hour, t, (
                   ('ec_c_lh', _rgba(sc._c_lighthouse), -lo, 0, nightwin.SHARE, (sc.LH_X - 4, sc.LH_X + 4, 1.0)),
                   ('ec_c_main', _rgba(sc._c_main), -cx, 0)))]
        if night:
            lit.append(_light_on(c, hour, _lamp_room_pts(), -lo, _LAMP_ROOM))
        for m in glints:
            lit.append(m & np.all(c.a == np.array(MOON_GLINT, np.uint8), axis=2))
        if 'refl' in bayou:                              # sc.reflect's sky (for after_light), and the
            snap, wl, x0, amp, k0, k1, tint = bayou['refl']          # cabins' lights in the bayou
            pure = np.all(c.a[wl:] == snap, axis=2)
            self._water = (pure, f, wl, amp, k0, k1, tint, x0)
            lit.append(_reflect_lights(c, cabins, f, wl, pure, amp, x0))
        self._keep(c, *lit)
        return self._done(c, f, cx, cy)

    def after_light(self, c, t):
        """After the relight (and the night sky): the clock's whole sky, the kept lights, then the
        lighthouse's beam and lamp, and its flashes."""
        sky = self._repaint_sky(c, t)
        water, self._water = getattr(self, '_water', None), None
        if water is not None and sky is not None:
            _sky_in_water(c, sky, water, self.HOUR, self.g.hour)
        super().after_light(c, t)
        post, self._lh_post = getattr(self, '_lh_post', None), None
        if post is None:
            return
        x, y, fb, gain, allow, al = post
        k = min(1.0, max(0.0, (daylight.night(self.g.hour) - 0.3) / 0.3))     # from 0 as it comes on
        halo = np.zeros((H, W), bool) if sky is None else sky.copy()
        halo[max(0, y - 7):y + 4, max(0, x - 4):max(0, x + 5)] = True     # the lamp room itself
        _night_light(c, x, y, fb, gain, allow, k, halo)
        if 0 <= al < 50:
            _lamp_flash(c, al, x, y)
            ui.burst(c, al, 0, x, y, colors=('white', 'butter', 'gold'), n=22, seed=7, power=0.9, life=40,
                     gravity=0.02)
