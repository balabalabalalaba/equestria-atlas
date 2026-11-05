"""Las Pegasus as an explorable diorama (art: scenes/s16_laspegasus.py, used read-only).

The film's render(f) switches content by shot and runs dusk -> neon night on film frames, so the
region re-composes the scene's own cached layers and helpers at the player's camera (recompose style,
see sweetapple.py), without the film's balloon, callouts, card, iris or dissolves. Two areas, flown
through left to right with a cloud sweep between them:
  0  THE STRIP (shots A + C): the resort towers on their cloud bank, the Applewood hills and the sea
     behind. The camera pans the whole strip (0..160) and cranes up 40 px into the firework sky (the
     film's shot-C tilt, done as a parallax crane: sky and coast move 0.2, back clouds 0.5, the city 1,
     the drifting front puffs 1.4). World y = film y + 40 on the city layer.
  1  FOUNTAIN PLAZA (shot B): the dancing fountain, the jackpot casino, the Ferris wheel and the Wild
     Blue Yonder coaster at night. Camera 0..100 (the plaza floor ends at the film's 420 px).
Timed film beats on game clocks:
  - THE NEON HIT (film 140..216): the first landing starts at dusk with the backlit strip. SPACE at the
    Grand Tower's entrance marquee (or flying east past the Strip, or waiting 12 s) runs the film's
    cascade: the tower lights and its star blazes, then one building per 8th down the strip while the
    sky darkens to violet night. SPACE during the cascade blazes the star again; afterwards it sends
    a power surge down the strip (windows go dark, each building relights bottom-up). Once lit, the
    strip stays lit on later landings (Region.lasting 'neon', saved).
  - THE FIREWORKS (film 276..380): the film's bar-5 show replays once after the strip is lit and on
    SPACE at the gold resort's star; single shells pop now and then over the lit strip.
  - the jackpot sign, fountain finale and premiere searchlights replay from the press (below).
  - drifting front puffs and the sailboat run on seamless loops (the film's drifts only last 5 bars).
Things to set off with SPACE: the Grand Tower's neon (cascade / surge), the Applewood premiere
(searchlights blaze and the sign's letters chase), the fireworks show, the fountain's finale and the
jackpot lever (reels spin and land three stars). Three atlas stamps are hidden: high in the firework
sky above the Grand Tower (only when the camera cranes up), low in the cloud bank under the theatre,
and past the fountain's east urn at the plaza's far end.
The world clock: by day the strip and the plaza are drawn as day art (true colours, neon and windows
off, white clouds, a blue sea), at dusk and night as the film's art, blended in between (see "the clock's
light" below); the moon is the clock's.
No creatures anywhere (the edition's rule): towers, neon, water and light only.
"""
import math
from functools import lru_cache

import numpy as np

from engine import W, H, Canvas, fx, seg, hash01, col, clamp, ease_out, mix, shade, smooth, text
from engine.core import BAYER4
import scenes.s16_laspegasus as sc

from .. import ui, daylight
from ..region import Region, Area, Landmark, Hotspot, Stamp, rebind

# ----------------------------------------------------------------------------- strip geometry
TOP = 40                    # crane range: world y = film y + TOP on the city layer
CAM_A = 160                 # city / bank 480 px (p 1) and back clouds 420 (p .5, -20) end at cam 160
CAM_B = 100                 # the plaza floor (p 1, 420 px) ends at cam 100
P_SKY, P_FAR, P_BACK, P_FG = 0.2, sc.PAR_FAR, sc.PAR_BACK, sc.PAR_FG
P_STREAK, P_FW = 0.15, 0.35
FW_DX = 35                  # firework sky: screen x = film x + FW_DX - cam * P_FW (film framing at cam 100)
FG_LOOP = 560               # fg_puffs width (art spans 26..548, so it tiles seamlessly)
RIDGE = [(0, 88), (14, 84), (34, 78), (52, 74), (68, 75), (84, 80), (100, 90), (116, 102),
         (134, 112), (150, 118), (160, 122), (0, 122)]          # far_layer's ridge (Applewood sign)


def oyp(cy, p):
    """Vertical screen offset of a layer of parallax p at camera y cy (0 at the film framing cy=TOP);
    the same rounding as the base class uses for tags (ay - round(iy * p))."""
    return int(round(TOP * p)) - int(round(cy * p))


def wy(film_y, p=1.0):
    """World (tag / hotspot) y of a film y on a layer of parallax p."""
    return film_y + int(round(TOP * p))


# ----------------------------------------------------------------------------- landmarks
TOWER_X = 204
STAR_A = (TOWER_X, 18)                  # Grand Tower star (film px)
GOLD_STAR = (402, 14)                   # gold resort star (film px)
LANDMARKS_A = [
    Landmark('APPLEWOOD', 64, (84, wy(80, P_FAR), 'up', 10), p=P_FAR),
    Landmark('LAS PEGASUS', TOWER_X, (220, wy(36), 'right', 10)),
    Landmark('THE STRIP', 317, (317, wy(80), 'up', 12)),
]
LANDMARKS_B = [
    Landmark('FOUNTAIN PLAZA', 196, (196, sc.RIM_Y + 1, 'down', 10)),
    Landmark('WILD BLUE YONDER', 330, (320, 76, 'up', 12), p=sc.PAR_MID),
]

PREMIERE = (48, 116)                    # far layer px: the studio town under the searchlights
HOTSPOTS_A = [
    Hotspot('premiere', PREMIERE[0], wy(PREMIERE[1], P_FAR), 'lantern', r=40, cooldown=110, p=P_FAR),
    Hotspot('neon', STAR_A[0], wy(117), 'spark', cooldown=100),        # the grand entrance's marquee
    Hotspot('fireworks', GOLD_STAR[0], wy(GOLD_STAR[1] + 4), 'firework', cooldown=110),
]
LEVER = (sc.JX + 52, sc.JY + 5)         # jackpot lever knob (mid layer)
HOTSPOTS_B = [
    Hotspot('jackpot', LEVER[0], LEVER[1], 'bell', r=40, cooldown=110, p=sc.PAR_MID),
    Hotspot('fountain', sc.jet_x(5), 98, 'splash', cooldown=100),
]
STAMPS_A = [Stamp('sky', 232, 12), Stamp('undercloud', 372, wy(154))]
STAMPS_B = [Stamp('plaza', 404, 124)]

# ----------------------------------------------------------------------------- clocks
NEON_F0, NEON_END = 140, 216            # film cascade: sky darkens 140..212, last building lit at 216
AUTO_NEON_X = 330                       # flying east past the Strip lights it
AUTO_NEON_T = 360                       # ... and so does waiting 12 s at dusk
SURGE_F0 = 138                          # surge: buildings relight from film 144 on, 6 steps after SPACE
SHOW_F0, SHOW_END = 274, 382            # the film's bar-5 show (first trail rises at 276)
SHOW_HITS = {288: (0.8, 2), 294: (0.3, 0), 299: (0.3, 1), 306: (0.45, 0), 313: (0.25, 2),
             318: (0.3, 0), 324: (0.7, 1), 333: (0.35, 2), 342: (0.8, 2), 346: (0.3, 1), 350: (0.4, 0)}
AMB_PERIOD = 150                        # a single shell pops over the lit strip every 5 s
PREM_LEN = 120
FIN_LEN = 84
SPIN_STOPS = (12, 18, 24)               # jackpot reels land this many steps after the pull
WIN_LEN = 96

TINT_DUSK = ('#a27ab4', 0.36)           # backlit by the afterglow
TINT_NIGHT = ('#6658a8', 0.48)          # violet night
TINT_NEON = ('#c86ad0', 0.40)           # down among the lit towers
TINT_PLAZA = ('#8a68c8', 0.42)


def _tint_between(a, b, q):
    """A tint q (0..1, in eighths: the compositor caches one sprite per tint) of the way from a to b."""
    q = round(clamp(q) * 8) / 8
    return mix(a[0], b[0], q), round(a[1] + (b[1] - a[1]) * q, 3)


# ----------------------------------------------------------------------------- cached strip art
@lru_cache(maxsize=None)
def _lit_city():
    """The fully lit strip (film state 2): the city layer, the night bank and one light map that equals
    the film's order (neon halos, the bank drawn over them, the neon pooled on the bank)."""
    layers, glows, _, _ = sc.city_layers()
    bank = sc.front_bank(True)
    halo = np.zeros_like(glows[0])
    for g in glows:
        halo += g
    pool = np.zeros_like(halo)
    for g in sc.bank_glow_maps():
        pool += g
    on_bank = bank[..., 3] >= 128
    light = np.where(on_bank[..., None], pool, halo).astype(np.int16)
    halo_off = np.where(on_bank[..., None], 0, halo).astype(np.int16)   # the halos off the bank (sky,
    return layers[2], bank, light, halo_off                             # towers): faded out by day


@lru_cache(maxsize=None)
def _letters():
    """Applewood sign letters: (x, y) top-lefts on the far layer."""
    return tuple((16 + k * 7, sc.ridge_y(RIDGE, 16 + k * 7 + 3) + 4) for k in range(9))


_FAST = {}


def _paste_rgba(c, rgba, x, y=0):
    """c.blit(rgba, x, y) for a cached RGBA layer: same pixels, clipped to the layer's opaque bounding
    box and copied with a precomputed mask (several times faster for big, mostly empty layers)."""
    e = _FAST.get(id(rgba))
    if e is None:
        m = rgba[..., 3] >= 128
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        r0, r1, c0, c1 = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) if rows.size else (0, 0, 0, 0)
        e = _FAST[id(rgba)] = (rgba, np.ascontiguousarray(rgba[r0:r1, c0:c1, :3]), m[r0:r1, c0:c1, None],
                               r0, c0)
    _, rgb, m, r0, c0 = e
    x, y = int(x) + c0, int(y) + r0
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(W, x + rgb.shape[1]), min(H, y + rgb.shape[0])
    if X1 > X0 and Y1 > Y0:
        np.copyto(c.a[Y0:Y1, X0:X1], rgb[Y0 - y:Y1 - y, X0 - x:X1 - x], where=m[Y0 - y:Y1 - y, X0 - x:X1 - x])


@lru_cache(maxsize=64)
def _blend(name, tq):
    """sc.xpaste's cross-fade of scene layer `name` (dusk -> night) at t = tq / 12, as a whole RGBA
    layer (xpaste blends per pixel with no screen-anchored dither, so this is pixel-identical)."""
    fn = getattr(sc, name)
    a, b = fn(False), fn(True)
    t = tq / 12.0
    ma, mb = a[..., 3] >= 128, b[..., 3] >= 128
    rgb = (a[..., :3].astype(np.float32) * (1 - t) + b[..., :3].astype(np.float32) * t + 0.5).astype(np.uint8)
    rgb[ma & ~mb] = a[..., :3][ma & ~mb]
    rgb[mb & ~ma] = b[..., :3][mb & ~ma]
    out = np.zeros_like(a)
    out[..., :3] = rgb
    out[..., 3] = (ma | mb) * 255
    return out


def _xp(c, name, x, y, t):
    """sc.xpaste(c, sc.<name>(False), sc.<name>(True), x, y, t), with the cross-fades cached."""
    fn = getattr(sc, name)
    if t <= 0.0:
        _paste_rgba(c, fn(False), x, y)
    elif t >= 1.0:
        _paste_rgba(c, fn(True), x, y)
    else:
        _paste_rgba(c, _blend(name, int(round(t * 12))), x, y)


def _beam(c, top, bottom, color, strength):
    """fx.light_beam (same pixels), all rows at once."""
    (txl, ty), (txr, _) = top
    (bxl, by), (bxr, _) = bottom
    y0, y1 = max(0, int(ty)), min(H, int(by) + 1)
    if y1 <= y0:
        return
    ys = np.arange(y0, y1)
    if by <= ty:
        t = np.where(ys >= by, 1.0, 0.0)
    else:
        t = np.clip((ys - ty) / (by - ty), 0.0, 1.0)
    xl = np.trunc(txl + (bxl - txl) * t).astype(np.int64)
    xr = np.trunc(txr + (bxr - txr) * t).astype(np.int64)
    x0 = np.maximum(0, np.minimum(xl, xr))
    x1 = np.minimum(W, np.maximum(xl, xr) + 1)
    X0, X1 = int(x0.min()), int(x1.max())
    if X1 <= X0:
        return
    xs = np.arange(X0, X1)
    thr = BAYER4[(ys % 4)[:, None], (xs % 4)[None, :]]
    m = (xs[None, :] >= x0[:, None]) & (xs[None, :] < x1[:, None]) & (thr < (1 - t * 0.7)[:, None])
    reg = c.a[y0:y1, X0:X1].astype(np.float32)
    reg[m] += np.array(col(color), np.float32) * strength
    c.a[y0:y1, X0:X1] = np.clip(reg, 0, 255).astype(np.uint8)


def _gs_dusk(_f, _i):
    return 0, None, 0.0


def _gs_cascade(nf):
    def gs(_f, i):
        return sc.group_state(nf, i)
    return gs


def _gs_surge(nf):
    """Power surge: the film's cascade states, but nothing drops back to the dusk silhouette (state 0
    is the sunset-rimmed layer): unlit buildings keep their neon tubes on and their windows dark."""
    def gs(_f, i):
        st, row, k = sc.group_state(nf, i)
        if st == 0:
            return 1, None, 0.25
        return st, row, k
    return gs


def _shift(shells, dx):
    return tuple((t0, x + dx, y, R, c1, c2, n, kind) for (t0, x, y, R, c1, c2, n, kind) in shells)


_PAIRS = (('neon_pink', 'magic'), ('neon_cyan', 'baby_blue'), ('neon_gold', 'amber'),
          ('lavender', 'neon_pink'), ('leaf', 'neon_cyan'), ('neon_cyan', 'white'))
_KINDS = ('peony', 'ring', 'peony', 'willow', 'heart', 'ring')


def _amb_shell(k):
    """Ambient single shell number k (t0 = k * 1000, sky px at film framing)."""
    h = [hash01(k, j, 977) for j in range(6)]
    kind = _KINDS[int(h[0] * len(_KINDS))]
    R = 14 + int(h[1] * 10)
    c1, c2 = _PAIRS[int(h[2] * len(_PAIRS))]
    return (k * 1000, 30 + int(h[3] * 260), 22 + int(h[4] * 34), R, c1, c2, 22 + R, kind)


def _disc_offsets(r):
    """(shadow, body, light) pixel offsets of a puff of radius r: a lower-right rim, the rest, and an
    upper-left highlight (a 2-step ramp + highlight, so each puff reads as a round ball of smoke)."""
    pts = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r + r * 0.8]
    inner = {(dx, dy) for dx, dy in pts if (dx + 1) ** 2 + (dy + 1) ** 2 <= r * r + r * 0.8}
    shadow = [p for p in pts if p not in inner]
    light = [(dx, dy) for dx, dy in pts if (dx + 1) ** 2 + (dy + 1) ** 2 <= max(0.5, (r - 1) ** 2)]
    body = [p for p in pts if p not in shadow and p not in light]
    return (np.array(shadow, np.int64).reshape(-1, 2), np.array(body, np.int64).reshape(-1, 2),
            np.array(light, np.int64).reshape(-1, 2))


_PUFF = {r: _disc_offsets(r) for r in (1, 2, 3)}
DAY_EXTRA = 26                          # by day a shell's smoke hangs this much longer than its stars


def _puffs(c, xs, ys, r, cols):
    """Draw puffs of radius r at integer centres (xs, ys) with (shadow, body, light) colours, all at once."""
    if not len(xs):
        return
    a = c.a
    for offs, cc in zip(_PUFF[r], cols):
        if not len(offs):
            continue
        X = (xs[:, None] + offs[None, :, 0]).ravel()
        Y = (ys[:, None] + offs[None, :, 1]).ravel()
        m = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        a[Y[m], X[m]] = col(cc)


def _day_fireworks(c, shells, f, oy, dc):
    """Fireworks in daylight (the clock's, not the painted night): a glow would be left as a dark blot
    on the repainted day sky, so each shell is drawn the way daytime fireworks really look: a pale smoke
    trail climbs, a white pop, its stars flash bright and leave a ring of coloured smoke puffs that
    swell, drift on the wind and thin out one by one (each puff leaves for good: no flicker). Same
    shells, same timing. dc(colour) -> the colour to paint so it shows true after the day's relight."""
    for sh in shells:
        t0, x, y, R, c1, c2, n, kind = sh
        a = f - t0
        yy0 = y + oy
        life = 30 if kind != 'willow' else 44
        end = life + DAY_EXTRA
        if a < -12 or a >= end:
            continue
        # the smoke trail: rises with the shell, then hangs and thins out from the bottom
        u = clamp((a + 12) / 12.0, 0.0, 1.0)
        if a < 22:
            keep = [k for k in np.arange(0.0, u + 1e-6, 0.06)
                    if a < 0 or hash01(t0, int(k * 100), 3) > (a / 22.0) * (1.4 - k)]
            if keep:
                tx = np.array([x + int(round(math.sin(k * 3))) for k in keep], np.int64)
                ty = np.array([int(lerp_i(150 + oy, yy0, ease_out(k))) for k in keep], np.int64)
                _puffs(c, tx, ty, 1, (dc('#c4c0dc'), dc('#eceaf6'), dc('white')))
        if a < 0:
            hx = x + int(round(math.sin(u * 3)))
            hy = int(lerp_i(150 + oy, yy0, ease_out(u)))
            c.px(hx, hy, dc('white'))
            c.px(hx, hy + 1, dc('cream'))
            continue
        drift = max(0, a - life * 0.6)
        pts = []
        for i in range(n):
            if a > life * 0.55 and hash01(t0, i, 11) < (a - life * 0.55) / (end - life * 0.55):
                continue                                        # this puff has thinned away
            px_, py_ = sc.shell_pos(sh, min(a, life * 0.7), i)
            pts.append((int(round(px_ + drift * 0.12)), int(round(py_ + oy + drift * 0.05)),
                        kind == 'peony' and i % 3 == 0))
        if a < 3:                                               # the pop
            c.circle(x, yy0, 4 - a, dc('white'))
        q = clamp((a - 9) / float(end - 9), 0.0, 1.0)           # the smoke pales as it ages
        for core in (False, True):
            xs = np.array([p[0] for p in pts if p[2] == core], np.int64)
            ys = np.array([p[1] for p in pts if p[2] == core], np.int64)
            cA = c2 if core else c1
            if a < 3:
                _puffs(c, xs, ys, 1, (dc('white'),) * 3)
            elif a < 9:                                         # the stars: small and bright
                _puffs(c, xs, ys, 1, (dc(cA), dc(mix(cA, 'white', 0.3)), dc('white')))
            else:                                               # the smoke: swells, fades to cloud
                r = 1 if a >= end - 6 else (3 if core and R >= 40 and a >= 14 else 2)
                _puffs(c, xs, ys, r, (dc(mix(mix(cA, 'periwinkle', 0.45), 'cloud', 0.15 + 0.35 * q)),
                                      dc(mix(cA, 'cloud', 0.2 + 0.5 * q)),
                                      dc(mix(cA, 'white', 0.6 + 0.3 * q))))


def lerp_i(a, b, t):
    return a + (b - a) * t


def _searchlights(c, f, ox, oy, k, boost=0.0, pa=0, kn=1.0):
    """The film's two premiere searchlights (sc.searchlights, here with a vertical offset). While the
    premiere's `boost` (0..1) is up they swing wide and criss-cross, brighter, and a pink and a mint
    beam join them (pa: steps since the premiere started). kn = daylight.sky_light: a beam is light
    added to the painted night sky, and on a brighter clock sky (dawn, day, sunset) that sky is
    repainted around it, so the beam would be left as a dark stripe: none then."""
    if kn <= 0.05 or (k <= 0.05 and boost <= 0):
        return
    beams = [(36, 0.0, '#a8b0ff'), (86, 2.1, '#a8b0ff')]
    if boost > 0:
        beams += [(18, 1.0, '#ffb0e8'), (62, 3.3, '#b0fff0')]
    for j, (bx, ph, cc) in enumerate(beams):
        if j < 2:
            a = -math.pi / 2 + 0.55 * math.sin(f * 0.045 + ph) + 0.5 * boost * math.sin(pa * 0.08 + ph * 2)
            s = (0.12 * max(k, 0.0) + 0.26 * boost) * min(1.0, kn)
        else:
            a = -math.pi / 2 + 0.7 * math.sin(pa * 0.1 + ph)
            s = 0.30 * boost * min(1.0, kn)
        x0, y0 = bx + ox, 114 + oy
        L = 120 + 40 * boost
        ex, ey = x0 + math.cos(a) * L, y0 + math.sin(a) * L
        w = 9 + 4 * boost
        if s > 0.01:
            _beam(c, ((ex - w, ey), (ex + w, ey)), ((x0, y0), (x0 + 1, y0)), cc, s)


def _sailboat(c, f, fxo, oy, t):
    """The film's tiny sailboat, drifting left on a loop (enters and leaves off-screen)."""
    bx = 350 - int((f * 0.08) % 380) + fxo // 2
    by = 136 + oy
    if not -12 < bx < W + 12:
        return
    c.hline(bx - 6, bx + 8, by + 2, mix('#c89ad8', '#6a5aa0', t))
    c.hline(bx - 4, bx + 4, by, '#8a4a5a')
    c.hline(bx - 3, bx + 3, by + 1, '#5a2e44')
    c.vline(bx, by - 8, by - 1, '#d8b090')
    c.tri((bx + 1, by - 8), (bx + 1, by - 2), (bx + 5, by - 2), mix('#ffd8e0', '#b8a0d8', t))
    c.tri((bx - 1, by - 6), (bx - 1, by - 2), (bx - 4, by - 2), mix('#ffb8c8', '#9a80c0', t))
    c.px(bx + 4, by - 1, 'butter' if (f // 12) % 2 else '#ffb35a')


_M32 = np.uint64(0xffffffff)


def _hash01v(*xs):
    """engine.hash01 over numpy integer arrays (same values, broadcast)."""
    h = np.uint64(2166136261)
    for x in xs:
        x = np.asarray(x, np.int64) & 0xffffffff
        h = h ^ x.astype(np.uint64)
        h = (h * np.uint64(16777619)) & _M32
        h ^= h >> np.uint64(13)
        h = (h * np.uint64(0x5bd1e995)) & _M32
        h ^= h >> np.uint64(15)
    return (h & np.uint64(0xffffff)).astype(np.float64) / float(0x1000000)


def _mixv(a, b, t):
    """engine.mix for an array of t: (n, 3) uint8 colours."""
    a, b = np.array(col(a), np.float64), np.array(col(b), np.float64)
    t = np.asarray(t, np.float64)[..., None]
    return np.round(a + (b - a) * t).astype(np.uint8)


@lru_cache(maxsize=None)
def _drops(i):
    """The fountain droplets' per-jet constants: age offsets, x speeds, colour classes."""
    k = np.arange(22)
    h1, h2 = _hash01v(i, k, 61), _hash01v(i, k, 62)
    off = (h1 * 24).astype(np.int64)
    vx = (0.15 + 0.45 * h2) * np.where(k % 2 == 1, 1, -1)
    return k, off, vx


_JCOLS = (-3, -2, -1, 0, 1, 2, 3)


def _fountain(c, f, ox, jet_h, jet_col, glow=True):
    """sc.fountain (same pixels), with each jet's column, droplets and foam drawn with numpy (glow False:
    by day, no coloured light on the water)."""
    white = np.array(col('white'), np.uint8)
    dark = col('#2a1a4a')
    J = np.array(_JCOLS)
    for i in range(sc.NJ):
        x = sc.jet_x(i) + ox
        if x < -16 or x > W + 16:
            continue
        h = jet_h(f, i)
        cc = col(jet_col(f, i))
        top = sc.RIM_Y - h
        light = mix(cc, 'white', 0.6)
        if glow:
            fx.glow(c, x, sc.RIM_Y - 2, 18, cc, 0.45, rings=3)
            fx.glow(c, x, top + 2, 7, light, 0.30, rings=2)
        ys = np.arange(top + 1, sc.RIM_Y)
        ys = ys[(ys >= 0) & (ys < H)]
        if ys.size:
            u = (ys - top) / max(1, h)
            wob = np.where(u > 0.15, (_hash01v(i, ys // 2, (f + ys) // 3) * 3).astype(np.int64) - 1, 0)
            half = np.where(u < 0.12, 0, np.where(u < 0.7, 1, 2))
            lo = -half + np.minimum(0, wob)
            hi = half + np.maximum(0, wob)
            m = (J[None, :] >= lo[:, None]) & (J[None, :] <= hi[:, None])
            edge = _mixv(cc, 'white', 0.25 + 0.5 * (1 - u))
            core = _mixv('white', cc, 0.2 * u)
            patch = np.broadcast_to(edge[:, None, :], (ys.size, 7, 3)).copy()
            patch[:, 3] = core
            m[:, 3] = True
            xs0, xs1 = max(0, x - 3), min(W, x + 4)
            if xs1 > xs0:
                j0, j1 = xs0 - (x - 3), xs1 - (x - 3)
                dst = c.a[ys[0]:ys[-1] + 1, xs0:xs1]
                np.copyto(dst, patch[:, j0:j1], where=m[:, j0:j1, None])
        c.hline(x - 2, x + 2, top, light)
        c.hline(x - 1, x + 1, top - 1, 'white')
        c.px(x, top - 2, light)
        # droplets (in the film's draw order: each drop, then its dark trail pixel)
        g = 2.0 * (h + 6) / (24.0 * 24.0)
        k, off, vx = _drops(i)
        age = (f + off) % 24
        dx = np.round(vx * age).astype(np.int64)
        dy = np.round(-0.9 * age + 0.5 * g * age * age + 0.9 * age * (age / 24.0)).astype(np.int64)
        yy = top + dy
        keep = yy < sc.RIM_Y - 1
        if keep.any():
            lt, mid = np.array(light, np.uint8), np.array(mix(cc, 'white', 0.3), np.uint8)
            pc = np.where((age < 9)[:, None], lt, np.where((k % 3 != 0)[:, None], mid, white))
            sh = np.round(pc.astype(np.float64) + (np.array(dark, np.float64) - pc) * 0.5).astype(np.uint8)
            trail = keep & (age > 4) & (k % 2 == 1)
            px = np.stack([x + dx, x + dx], 1).ravel()
            py = np.stack([yy, yy - 1], 1).ravel()
            pcs = np.stack([pc, sh], 1).reshape(-1, 3)
            ok = np.stack([keep, trail], 1).ravel() & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            c.a[py[ok], px[ok]] = pcs[ok]
        # foam at the nozzle
        kk = np.arange(5)
        fxs = x - 4 + (_hash01v(i, kk, f // 2) * 9).astype(np.int64)
        fys = sc.RIM_Y - 1 - (_hash01v(kk, i, f // 3) * 2).astype(np.int64)
        ok = (fxs >= 0) & (fxs < W)
        c.a[fys[ok], fxs[ok]] = white


@lru_cache(maxsize=4)
def _plaza_sky():
    c = sc.backdrop(12).copy()
    fx.glow(c, 160, 128, 150, '#ff4fd8', 0.10, rings=5)
    return c


@lru_cache(maxsize=None)
def _far_resort_day():
    """sc.far_resort with its window dots dark (by day): the same towers, walked as the scene does."""
    a = sc.far_resort().copy()
    x, i = 0, 0
    while x < 400:
        w = 12 + int(hash01(i, 1, 51) * 16)
        h = 36 + int(hash01(i, 2, 51) * 58)
        top = 130 - h
        glass = col(shade(mix('#2c2058', '#3c2c6c', hash01(i, 3, 51)), 0.78))
        for yy in range(top + 3, 140, 4):
            for xx in range(x + 2, x + w - 3, 3):
                if hash01(i, xx, yy) < 0.5 and 0 <= xx < a.shape[1]:
                    a[yy, xx, :3] = glass
        x += w + 1 + int(hash01(i, 5, 51) * 5)
        i += 1
    return a


def _no_sea(c, y):
    """Rows from y down (sea, horizon line) set to their inverse so they never count as sky."""
    if y < H:
        y = max(0, y)
        c.a[y:] = 255 - c.a[y:]
    return c


def _not_sky(c, rgba, x, y=0):
    """Invert the sky under an opaque scenery layer's pixels (pasted at x, y) so they never match."""
    _paste_rgba(c, _inv_layer(rgba), x, y)


_INV = {}


def _inv_layer(rgba):
    """A cached layer with its colours inverted (same alpha)."""
    e = _INV.get(id(rgba))
    if e is None:
        e = _INV[id(rgba)] = (rgba, np.dstack([255 - rgba[..., :3], rgba[..., 3:]]))
    return e[1]


def _wrap_paste(c, name, x, y, t, period):
    """_xp a horizontally looping scene layer (art narrower than its period) at offset x (mod period)."""
    w = getattr(sc, name)(True).shape[1]
    x0 = (x % period) - period
    while x0 < W:
        if x0 + w > 0:
            _xp(c, name, x0, y, t)
        x0 += period


# ----------------------------------------------------------------------------- the clock's light
# The film painted Las Pegasus at night (20:00), lit neon and all. The world clock's relight only lifts a
# night picture evenly, so by day the strip stayed a violet night with its neon, halos and floodlights on,
# a dark sea and night clouds. So the place is drawn for the clock in two ways and blended:
#   DAY ART (day_k 1, about 06:15 .. 18:50): the same city in daylight - true facade colours, unlit glass
#     neon tubes, dark window glass, white clouds, a blue sea with the sun's glitter under the clock's sun,
#     tan coast hills - graded for the clock's hour like a day-painted place (golden in the evening);
#   FILM ART (day_k 0): the film's dusk (about 19:15) -> night (from about 19:55) art, chosen by the clock
#     (art_night), the neon lit by the player's cascade (a strip not lit yet stands dark, never sunset-rimmed).
# The blended picture is in the colours it should show: where the base relight (painted 20:00 -> the
# clock) would brighten it, after_light puts it back as it was drawn (the clock's sky repainted around it).
# Later than 20:00 the base relight dims the film's night gently, as for every place.
DAY_ART_H = 12.0                # the day art is painted for noon (the world grade's neutral hour)


def day_k(hour):
    """1 by day .. 0 by night: how much of the day art shows (the lamps come on as it fades)."""
    return clamp(1.0 - (daylight.night(hour) - 0.05) / 0.45)


def art_night(hour):
    """0 (the film's dusk art: afterglow, backlit city) .. 1 (its night art), by the clock. Its afterglow
    is a sunset in the west: in the morning the night art stays until the day art has taken over."""
    if (hour % 24) < 12:
        return 1.0
    return clamp((daylight.night(hour) - 0.5) / 0.4)


@lru_cache(maxsize=48)
def _brightens(fq, tq):
    """Does the relight from the painted hour fq to the clock's tq brighten the picture?"""
    lut = daylight._tables(fq, tq)[0]
    return bool(lut[:, 128].astype(np.int32).mean() > 128)


def _mix_into(a, b, k):
    """a = a * k + b * (1 - k) (uint8 pictures)."""
    if k >= 0.999:
        return
    w = int(round(k * 256))
    a[...] = ((a.astype(np.uint16) * w + b.astype(np.uint16) * (256 - w)) >> 8).astype(np.uint8)


def _clock_moon(c, hour):
    """The clock's moon where game/daylight.paint_sky draws it (so the painted sky, kept within minutes of
    20:00, never shows a second moon somewhere else)."""
    kind, bx, by = daylight._body_pos(hour)
    if kind == 'moon':
        daylight._disc(c.a, bx, by, 6, col('cream'), col('#d8d0c0'))
        c.a[by - 2, bx + 1] = col('#c8c0b0')
        c.a[by + 1, bx - 2] = col('#c8c0b0')


# ---- the city by day: true facade colours, unlit tubes, window glass (no lit doors, flood, glows, bulbs)
DAY_GLASS = ('#8fb2dc', '#7a9ed2', '#a8c8ea', '#6c8cc4', '#9ab8e0')


def _wc_day(_st, h):
    return col(DAY_GLASS[int(h * 997) % len(DAY_GLASS)])


def _ne_day(c, _st):
    return mix(c, '#e6e2ee', 0.6)                     # a glass tube, switched off, in daylight


def _t_day(c, _st, k=0.66):
    return col(c)


@lru_cache(maxsize=None)
def _day_city():
    win = rebind(sc.windows, WC=_wc_day)

    def windows(cv, x0, y0, nx, ny, gx, gy, ww, wh, st, seed, **kw):
        return win(cv, x0, y0, nx, ny, gx, gy, ww, wh, 2, seed, **kw)      # (2: the glass's glint)
    cv = sc.layer(sc.WA)
    for g in sc.GROUPS:                               # state 3: no lit doors, no floodlight
        rebind(g[0], T=_t_day, NE=_ne_day, WC=_wc_day, windows=windows)(cv, 3)
    return sc.to_rgba(cv)


@lru_cache(maxsize=None)
def _dark_city():
    """The strip's unlit state (film state 0) without the sunset rim: the city standing dark at night."""
    cv = sc.layer(sc.WA)
    for g in sc.GROUPS:
        g[0](cv, 0)
    return sc.to_rgba(cv)


@lru_cache(maxsize=16)
def _city_unlit(q):
    """Film state 0 for art_night q/12: the sunset-rimmed dusk silhouettes fading to the dark city."""
    a, b = sc.city_layers()[0][0], _dark_city()
    t = q / 12.0
    out = a.copy()
    out[..., :3] = (a[..., :3] * (1 - t) + b[..., :3] * t + 0.5).astype(np.uint8)
    return out


@lru_cache(maxsize=16)
def _city_layers_at(q):
    layers, glows, bulbs, gid = sc.city_layers()
    return (_city_unlit(q), layers[1], layers[2]), glows, bulbs, gid


# ---- clouds by day: the film's dusk clouds recoloured (cloud_mass: mid, hi, lo, rim, dk + two mixes)
_CLOUD_DUSK = {'back_clouds': ('#8e62a6', '#d2a2d4', '#ff8aa8', '#ffd0dc', '#7a5296'),
               'front_bank': ('#a874b6', '#e6b8e0', '#ff9fd0', '#ffe2f0', '#8e5ea6'),
               'fg_puffs': ('#b882bc', '#f2c8e6', '#ff96c4', '#fff0f6', '#9a68ae')}
_CLOUD_DAY = {'back_clouds': ('#dde3f4', '#f6f8ff', '#b2bcdc', '#ffffff', '#c2c8e2'),
              'front_bank': ('#f3e6f2', '#fff8fc', '#d4bedc', '#ffffff', '#dac4e0'),
              'fg_puffs': ('#f7eef8', '#ffffff', '#dacae6', '#ffffff', '#e2d2ea')}


def _cloud_cols(p):
    mid, hi, lo, rim, dk = p
    return [col(mid), mix(mid, lo, 0.45), col(lo), mix(mid, hi, 0.5), col(hi), col(rim), col(dk)]


@lru_cache(maxsize=None)
def _day_clouds(name):
    a = getattr(sc, name)(False).copy()
    src = a[..., :3].copy()
    for s, d in zip(_cloud_cols(_CLOUD_DUSK[name]), _cloud_cols(_CLOUD_DAY[name])):
        a[..., :3][(src == np.array(s, np.uint8)).all(axis=2)] = d
    return a


@lru_cache(maxsize=None)
def _day_far():
    """sc.far_layer by day: tan coast hills with sage scrub, a hazy blue far ridge, a sandy shore, the
    pale studio town (dark glass), green palms; the APPLEWOOD letters white, the star gold."""
    cv = sc.layer(sc.FW)
    ridge2 = [(0, 102), (24, 96), (60, 98), (100, 104), (140, 108), (180, 112), (180, 116), (0, 116)]
    cv.poly(ridge2, '#a9b6d6')
    hill, hill_lt = '#c4a47e', '#e0c49a'
    cv.poly(RIDGE, hill)
    for i in range(len(RIDGE) - 3):
        (xa, ya), (xb, yb) = RIDGE[i], RIDGE[i + 1]
        if yb > ya:
            cv.line(xa, ya, xb, yb, hill_lt)
            cv.line(xa, ya + 1, xb, yb + 1, mix(hill_lt, hill, 0.5))
    for i in range(60):
        x = int(hash01(i, 3, 5) * 150)
        y0 = 80 + int(hash01(i, 4, 5) * 40)
        if cv.a[y0, x].tolist() == list(col(hill)):
            cv.px(x, y0, '#8e9660')
    cv.rect(0, 122, 162, 2, '#f0dcae')
    cv.rect(0, 124, 158, 1, '#c4ab8a')
    for k, (x, w, h) in enumerate(((4, 12, 6), (18, 9, 8), (30, 14, 5), (48, 10, 7), (62, 13, 6),
                                   (79, 8, 9), (90, 12, 5), (106, 10, 6), (120, 9, 4))):
        y = 122 - h
        body = '#e6d8e6'
        cv.rect(x, y, w, h, body)
        cv.hline(x + 1, x + w - 2, y - 1, body)
        cv.hline(x + 1, x + w - 2, y, '#fff6fa')
        cv.vline(x + w - 1, y, 121, '#c8b4cc')
        for j in range(w // 3):
            if hash01(k, j, 9) < 0.5:
                cv.px(x + 1 + j * 3, y + 2, '#7e98c2')
    cv.vline(44, 112, 121, '#b4a4b8')
    cv.rect(42, 109, 5, 4, '#d4c4d2')
    for x in (26, 74, 100):
        cv.vline(x, 114, 121, '#8a6a52')
        cv.hline(x - 2, x + 2, 113, '#4e8a58')
        cv.px(x - 3, 114, '#4e8a58')
        cv.px(x + 3, 114, '#4e8a58')
    for k, ch in enumerate('APPLEWOOD'):
        x = 16 + k * 7
        yr = sc.ridge_y(RIDGE, x + 3) + 4
        text(cv, x + 1, yr + 1, ch, '#a8907c', font='normal')
        text(cv, x, yr, ch, 'white', font='normal')
    sx, sy = 16 + 9 * 7 + 2, sc.ridge_y(RIDGE, 16 + 66) - 3
    sc.icon(cv, 'star7', sx, sy, '#c98a1e')
    sc.icon(cv, 'star7', sx - 1, sy - 1, 'gold')
    cv.px(sx + 2, sy + 1, 'butter')
    return sc.to_rgba(cv)


@lru_cache(maxsize=None)
def _day_backdrop():
    """backdrop(0)'s sky (the clock repaints it) over a day sea: turquoise at the horizon, deep blue below."""
    c = sc.backdrop(0).copy()
    fx.sky_blend(c, ['#a8dcf0', '#78c0e8', '#58a4dc', '#4488cc', '#3670b4'],
                 ['#a8dcf0', '#78c0e8', '#58a4dc', '#4488cc', '#3670b4'], 0.0, y0=sc.HORIZ, y1=H)
    c.hline(0, W - 1, sc.HORIZ, '#e4f6ff')
    return c


def _day_glitter(c, f, y0, sunx):
    """The sun's glitter on the sea: white dashes in a column under the clock's sun, a few scattered."""
    for i in range(40):
        h1, h2, h3 = hash01(i, 1, 78), hash01(i, 2, 78), hash01(i, 3, 78)
        y = y0 + 2 + int(h2 ** 1.5 * 64)
        if y >= H:
            continue
        spread = 6 + (y - y0) * 0.8
        x = int(sunx + (h1 - 0.5) * 2 * spread) if i % 3 else int(h1 * W)
        if math.sin(f * 0.21 + h3 * 6.28) <= 0.25:
            continue
        c.hline(x, x + 1 + int(h3 * 3), y, 'white' if i % 3 else '#d4eeff')


def _day_streaks(c, f, cam, oy):
    """sc.streaks by day: thin white stratus."""
    for x0, y, w in sc.STREAKS:
        x = int(x0 - cam * 0.12 + f * 0.05) % 560 - 120
        yy = y + oy
        c.hline(x + 4, x + w - 6, yy, '#ffffff')
        c.hline(x, x + w, yy + 1, '#eef2fa')
        c.hline(x + 3, x + w - 2, yy + 2, '#d6deee')
        c.hline(x + w // 3, x + w - 10, yy - 1, '#ffffff')


def _day_sailboat(c, f, fxo, oy):
    bx = 350 - int((f * 0.08) % 380) + fxo // 2
    by = 136 + oy
    if not -12 < bx < W + 12:
        return
    c.hline(bx - 6, bx + 8, by + 2, '#d8f0ff')
    c.hline(bx - 4, bx + 4, by, '#b05a6a')
    c.hline(bx - 3, bx + 3, by + 1, '#7a3a50')
    c.vline(bx, by - 8, by - 1, '#d8b090')
    c.tri((bx + 1, by - 8), (bx + 1, by - 2), (bx + 5, by - 2), '#ffffff')
    c.tri((bx - 1, by - 6), (bx - 1, by - 2), (bx - 4, by - 2), '#ffe0e8')


# ---- the plaza by day: its night palette (built for neon) turned into day colours, one lookup per pixel
class _NoGlow:
    """engine.fx for a scene draw function by day: no glows or light washes (fx.<other> as usual)."""
    @staticmethod
    def glow(*_a, **_k):
        return None

    light_rect = glow

    def __getattr__(self, name):
        return getattr(fx, name)


NOGLOW = _NoGlow()


@lru_cache(maxsize=1)
def _day_lut():
    """32768-entry lookup on 5-bit colours: the plaza's night colours by day. Lights (cream, butter and
    gold windows, lamp globes) turn to window glass, neon (bright, very saturated, not gold) to a glass
    tube switched off, and the indigo / violet buildings and floor to light pastel lavender (the value
    lifted, the saturation lowered); white, gold and the sky are kept."""
    q = np.arange(32768)
    rgb = np.stack([((q >> 10) & 31) * 8 + 4, ((q >> 5) & 31) * 8 + 4, (q & 31) * 8 + 4], 1).astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1), rgb.min(1)
    v, s = mx / 255.0, np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    lum = r * 0.3 + g * 0.55 + b * 0.15
    golden = (r >= g) & (g >= b) & (r > 180) & (s > 0.7)               # gold paint and trim
    light = (lum > 190) & (r > 200) & (g > 170) & (b < g - 15) & (s < 0.6)    # lit windows, lamp globes
    neon = (mx > 200) & (s >= 0.5) & ~golden & ~light
    # pastel lift: value up (keeping the shading's order), saturation down, the hue kept
    v2 = 0.35 + 0.65 * v ** 0.7
    s2 = s * 0.55
    hi = v2 * 255
    lo = hi * (1 - s2)
    k = np.where((mx > mn)[:, None], (rgb - mn[:, None]) / np.maximum(mx - mn, 1)[:, None], 0)
    out = lo[:, None] + (hi - lo)[:, None] * k
    glass = np.array(col('#9cb8dc'), np.float32)
    tube = np.array(col('#e6e2ee'), np.float32)
    out = np.where(light[:, None], rgb * 0.3 + glass * 0.7, out)
    out = np.where(neon[:, None], rgb * 0.4 + tube * 0.6, out)
    keep = golden | ((s < 0.08) & (lum > 200))
    out = np.where(keep[:, None], rgb, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _day_px(px):
    """Day colours for an (n, 3) uint8 array of night colours."""
    qq = ((px[:, 0] >> 3).astype(np.int32) << 10) | ((px[:, 1] >> 3).astype(np.int32) << 5) | (px[:, 2] >> 3)
    return _day_lut()[qq]


@lru_cache(maxsize=None)
def _day_layer(name):
    """A plaza layer of the scene (or far_resort with dark windows) in day colours."""
    a = (_far_resort_day() if name == 'far_resort' else getattr(sc, name)()).copy()
    m = a[..., 3] >= 128
    a[..., :3][m] = _day_px(a[..., :3][m])
    return a


def _day_new(c, before):
    """Day colours for what was drawn on c since `before` (a copy of c.a)."""
    m = (c.a != before).any(axis=2)
    if m.any():
        c.a[m] = _day_px(c.a[m])


DAY_JET = '#d4eaff'             # the fountain's water by day (no coloured lights)


# jackpot sign: idle = three stars showing, rim bulbs chasing, no win flashing
_FAR = 10 ** 9
_reel_idle = rebind(sc.reel_icon, REEL_STOP=(-_FAR, -_FAR, -_FAR))
_jackpot_idle = rebind(sc.jackpot, REEL_STOP=(-_FAR, -_FAR, _FAR), reel_icon=_reel_idle)


class LasPegasus(Region):
    SEG = 'laspegasus'
    CARD = ('RESORT CITY', ['NEON ON THE CLOUDS', 'SHOWS ALL NIGHT'])

    def build_areas(self):
        return [
            Area(self.render_strip, cam_x=(0, CAM_A), cam_y=(0, TOP), y_range=(22, wy(168)),
                 start=(92, wy(112)), landmarks=LANDMARKS_A, balloon='m', balloon_tint=TINT_DUSK,
                 hotspots=HOTSPOTS_A, stamps=STAMPS_A, sky=self.sky_strip),
            Area(self.render_plaza, cam_x=(0, CAM_B), cam_y=(0, 0), y_range=(24, 146), start=(40, 66),
                 landmarks=LANDMARKS_B, balloon='m', balloon_tint=TINT_PLAZA,
                 hotspots=HOTSPOTS_B, stamps=STAMPS_B, sky=self.sky_plaza),
        ]

    def warm(self):
        """Also build the lit strip, every dusk -> night cross-fade and the fountain tables now."""
        super().warm()
        _lit_city()
        sc.bank_glow_maps()
        for name in ('far_layer', 'back_clouds', 'front_bank', 'fg_puffs'):
            for tq in range(13):
                _blend(name, tq)
        for i in range(sc.NJ):
            _drops(i)

    # ------------------------------------------------------------------ state + clocks
    def init_state(self):
        self.neon_t0 = None         # step the neon cascade started (film frame NEON_F0); None = dusk
        self.surge_t0 = None        # step of the last power surge
        self.blaze_t0 = None        # step the star last blazed on its own (SPACE during the cascade)
        self.show_t0 = None         # step the fireworks show started (film frame SHOW_F0)
        self.show_auto = False      # the show played by itself after the cascade
        self.lit_t = None           # step the strip was fully lit (ambient shells count from it)
        self.area_t0 = 0

    def on_enter(self):
        self.neon_t0 = self.surge_t0 = self.show_t0 = self.lit_t = self.blaze_t0 = None
        self.show_auto = False
        self.areas[0].balloon_tint = TINT_DUSK
        if 'neon' in self.lasting or 'neon' in self.used:
            # the strip has been lit before: it stays lit (SPACE at the marquee sends a power surge); the
            # opening show was its first-time welcome, the single shells still pop over the lit strip
            self.neon_t0 = -10 ** 6
            self.lit_t = self.g.t
            self.show_auto = True
            self.areas[0].balloon_tint = TINT_NIGHT

    def on_area(self, ai):
        self.area_t0 = self.g.t
        if ai == 1:
            self.g.sfx('fountain', 0.35)

    def neon_f(self, t):
        """Cascade clock in film frames: NEON_F0 - 1 before it starts, capped at NEON_END."""
        if self.neon_t0 is None or t < self.neon_t0:
            return NEON_F0 - 1
        return min(NEON_END, NEON_F0 + (t - self.neon_t0))

    def night_k(self, t):
        return sc.night_t(self.neon_f(t))

    def lit(self, t):
        return self.neon_f(t) >= NEON_END

    def surge_f(self, t):
        if self.surge_t0 is None:
            return None
        nf = SURGE_F0 + (t - self.surge_t0)
        return nf if nf < NEON_END else None

    def show_f(self, t):
        if self.show_t0 is None:
            return None
        sf = SHOW_F0 + (t - self.show_t0)
        return sf if sf < SHOW_END else None

    def amb_shell(self, t, span=70):
        """(shell, age) of the current ambient shell over the lit strip, or None (span: steps it shows)."""
        if self.lit_t is None:
            return None
        k, a = divmod(t - self.lit_t - 60, AMB_PERIOD)
        if k < 0 or a >= span:
            return None
        if self.show_t0 is not None and 0 <= (t - a) - self.show_t0 < SHOW_END - SHOW_F0:
            return None                 # launched while the show played: skipped (never pops in later)
        sh = _amb_shell(int(k))
        return sh, sh[0] + a - 14

    def _start_neon(self, t):
        self.neon_t0 = t
        self.g.sfx('wind', 0.25, 1)

    def jack_f(self, t):
        """Steps since the jackpot lever was pulled, or None (idle)."""
        a = t - self.spot_t.get('jackpot', -10 ** 6)
        return a if a < SPIN_STOPS[2] + WIN_LEN else None

    # ------------------------------------------------------------------ events
    def on_interact(self, key, t):
        g = self.g
        if key == 'neon':
            if self.neon_t0 is None:
                self._start_neon(t)
            elif self.lit(t):
                self.surge_t0 = t
                g.sfx('zap', 0.35)
            else:                                   # mid-cascade: the star blazes again
                self.blaze_t0 = t
                g.sfx('neon', 0.45)
        elif key == 'fireworks':
            self.show_t0 = t
            g.sfx('whoosh', 0.35)
        elif key == 'premiere':
            g.sfx('whoosh', 0.4, 1)
            g.sfx('shimmer', 0.4)
        elif key == 'fountain':
            g.sfx('fountain', 0.6)
            g.sfx('splash', 0.35)
        elif key == 'jackpot':
            g.sfx('click', 0.45)
            g.sfx('clank', 0.25)

    def update(self, t):
        g = self.g
        k = t - self.area_t0
        if t - self.t_enter == 6:
            g.sfx('wind', 0.35)
        if self.ai == 0:
            if self.neon_t0 is None and (self.bx > AUTO_NEON_X or t - self.t_enter > AUTO_NEON_T):
                self._start_neon(t)
            nf = self.neon_f(t)
            if self.neon_t0 is not None and t - self.neon_t0 == nf - NEON_F0:
                cue = {144: 0.8, 162: 0.4, 180: 0.4, 198: 0.45}.get(nf)
                if cue:
                    g.sfx('neon', cue * max(0.4, self.near_gain(sc.GROUPS[(nf - 144) // 9][1])))
            if nf >= NEON_END and self.lit_t is None:
                self.lit_t = t
                self.lasting.add('neon')                     # saved: the strip stays lit (on_enter)
            if self.lit_t is not None and not self.show_auto and t - self.lit_t >= 30:   # >=: may be skipped
                self.show_auto = True
                self.show_t0 = t
            sf = self.surge_f(t)
            if sf in (144, 171, 198):
                g.sfx('neon', 0.35 if sf == 144 else 0.25)
            sf = self.show_f(t)
            if sf in SHOW_HITS:
                gain, var = SHOW_HITS[sf]
                g.sfx('firework', gain, var)
            am = self.amb_shell(t)
            if am is not None and am[1] == am[0][0]:
                g.sfx('firework', 0.22, int(am[0][0] // 1000) % 3)
            if k % 330 == 200 and self.lit(t):
                g.sfx('neon', 0.12)
            self._tint_strip(t)
        else:
            a = self.jack_f(t)
            if a is not None:
                if a in SPIN_STOPS[:2]:
                    g.sfx('tick', 0.4)
                elif a == SPIN_STOPS[2]:
                    g.sfx('jackpot', 0.65)
                elif a == SPIN_STOPS[2] + 10:
                    g.sfx('coin', 0.4)
            a = t - self.spot_t.get('fountain', -999)
            if a == 14:
                g.sfx('splash', 0.4, 1)
            if k % 216 == 108:
                g.sfx('fountain', 0.22 * self.near_gain(196))

    def _tint_strip(self, t):
        A = self.areas[0]
        q = self.night_k(t)
        tint = _tint_between(TINT_DUSK, TINT_NIGHT, q)
        if q >= 1:
            low = clamp((self.by - wy(40)) / 50)            # down among the lit towers: neon light
            tint = _tint_between(TINT_NIGHT, TINT_NEON, low * clamp((self.bx - 96) / 30))
        A.balloon_tint = tint

    # ------------------------------------------------------------------ sky hooks (game/daylight.py)
    def sky_strip(self, f, cx, cy):
        """render_strip's sky alone (the same blend of day and film art, given to the relight the same
        way); the sea (and the horizon line) below it is masked out."""
        return self._by_clock(lambda: self._sky_strip_day(cx, cy), lambda: self._sky_strip_film(f, cx, cy))

    def _sky_strip_day(self, cx, cy):
        oys = oyp(cy, P_SKY)
        bd = _day_backdrop()
        if oys:
            c = Canvas(W, H, bd.a[0, 0])
            c.blit(bd, 0, oys)
        else:
            c = bd.copy()
        daylight.relight(c.a, DAY_ART_H, self.g.hour, glow=False)
        _not_sky(c, _day_city(), -cx, oyp(cy, 1.0))     # (a navy facade at night matched the night sky)
        return _no_sea(c, sc.HORIZ + oys)

    def _sky_strip_film(self, f, cx, cy):
        """The backdrop's dusk/night gradient + afterglow, stars and the clock's moon on the crane offsets."""
        nt = art_night(self.g.hour)
        oys = oyp(cy, P_SKY)
        bd = sc.backdrop(int(round(nt * 12)))
        if oys:
            c = Canvas(W, H, bd.a[0, 0])
            c.blit(bd, 0, oys)
        else:
            c = bd.copy()
        fx.stars(c, f, seed=16, n=int(18 + 70 * nt), region=(0, oys - 58, W, 146))
        _clock_moon(c, self.g.hour)
        _not_sky(c, _lit_city()[0], -cx, oyp(cy, 1.0))  # the city: its dark facades are not the night sky
        return _no_sea(c, sc.HORIZ + oys)

    def _by_clock(self, day, film, keep=False):
        """The picture for the clock: day(), film() or both blended by day_k (each returns a new Canvas).
        keep: the area render (not its sky hook): kept for after_light where the relight would brighten."""
        hour = self.g.hour
        k = day_k(hour)
        if k >= 0.999:
            c = day()
        elif k <= 0.001:
            c = film()
        else:
            c = day()
            _mix_into(c.a, film().a, k)
        if keep and _brightens(daylight._q(self.HOUR), daylight._q(hour)) and abs(hour - self.HOUR) >= 0.05:
            self._final = c.a.copy()
        return c

    _final = None                   # the area's picture as drawn, to be put back after the relight

    def _world(self, t, cam=None, with_balloon=False):
        self._final = None
        return super()._world(t, cam, with_balloon)

    def after_light(self, c, t):
        """Put the picture back as drawn (in the colours it should show), around the clock's sky."""
        fin, self._final = self._final, None
        if fin is None:
            return
        memo = self._sky_memo
        if (memo is not None and memo[0][:2] == (self.ai, t)
                and daylight.sky_weight(self.HOUR, self.g.hour) > 0.01):
            np.copyto(c.a, fin, where=~memo[1][..., None])
        else:
            c.a[...] = fin

    def sky_plaza(self, f, cx, cy):
        """render_plaza's sky alone (day and film art blended as the render does it)."""
        return self._by_clock(lambda: self._sky_plaza_day(cx), lambda: self._sky_plaza_film(f, cx))

    def _sky_plaza_day(self, cx):
        c = sc.backdrop(12).copy()
        _no_sea(c, sc.HORIZ)
        _not_sky(c, _day_layer('far_resort'), -int(round(cx * sc.PAR_FAR_B)) - 20)
        _not_sky(c, _day_layer('casino_layer'), -int(round(cx * sc.PAR_MID)))
        daylight.relight(c.a, DAY_ART_H, self.g.hour, glow=False)
        return c

    def _sky_plaza_film(self, f, cx):
        """The night backdrop + violet wash, stars and the clock's moon (sea masked)."""
        c = _plaza_sky().copy()
        fx.stars(c, f, seed=26, n=80, region=(0, 0, W, 90))
        _clock_moon(c, self.g.hour)
        fx.light_rect(c, 0, 112, W, 30, '#8a3aa8', 0.18, fade='up')     # the city haze over the sky
        mo = -int(round(cx * sc.PAR_MID))
        fx.glow(c, sc.WCX + mo, sc.WCY, sc.WR + 8, 'magic', 0.10, rings=3)   # the wheel's glow (sc.ferris)
        _no_sea(c, sc.HORIZ)
        _not_sky(c, sc.far_resort(), -int(round(cx * sc.PAR_FAR_B)) - 20)   # dark towers on a dark sky
        _not_sky(c, sc.casino_layer(), mo)                              # its lit facade is sky-pink
        return c

    # ------------------------------------------------------------------ the strip
    def render_strip(self, f, cx, cy):
        return self._by_clock(lambda: self._strip_day(f, cx, cy), lambda: self._strip_film(f, cx, cy), True)

    def _strip_day(self, f, cx, cy):
        """The strip in daylight (see DAY ART above), graded for the clock's hour."""
        t = self.g.t
        hour = self.g.hour
        oys = oyp(cy, P_SKY)
        bd = _day_backdrop()
        if oys:
            c = Canvas(W, H, bd.a[0, 0])
            c.blit(bd, 0, oys)
        else:
            c = bd.copy()
        kind, sunx, _sy = daylight._body_pos(hour)
        _day_glitter(c, f, sc.HORIZ + oys, sunx if kind == 'sun' else W - 40)
        _day_streaks(c, f, cx, oyp(cy, P_STREAK))
        self._fireworks(c, t, cx, cy, 0.0)
        fxo = -int(round(cx * P_FAR))
        oyf = oyp(cy, P_FAR)
        _paste_rgba(c, _day_far(), fxo, oyf)
        pa = t - self.spot_t.get('premiere', -999)
        if 0 <= pa < PREM_LEN:
            boost = smooth(seg(pa, 0, 10)) * (1 - smooth(seg(pa, PREM_LEN - 30, PREM_LEN)))
            self._premiere(c, t, pa, fxo, oyf, boost, 0.0)
        _day_sailboat(c, f, fxo, oyf)
        _paste_rgba(c, _day_clouds('back_clouds'), -int(round(cx * P_BACK)) - 20, oyp(cy, P_BACK))
        oy = oyp(cy, 1.0)
        _paste_rgba(c, _day_city(), -cx, oy)
        _paste_rgba(c, _day_clouds('front_bank'), -cx, oy)
        self._neon_fx(c, t, self.neon_f(t), cx, oy, 0.0)
        x = -int(round(cx * P_FG)) + int(f * 0.12) - 120
        y = 142 + oyp(cy, P_FG)
        x0 = (x % FG_LOOP) - FG_LOOP
        while x0 < W:
            _paste_rgba(c, _day_clouds('fg_puffs'), x0, y)
            x0 += FG_LOOP
        daylight.relight(c.a, DAY_ART_H, hour, glow=False)          # graded for the clock (golden...)
        return c

    def _strip_film(self, f, cx, cy):
        """The film's strip: its dusk or night art by the clock (art_night), the neon by the cascade."""
        t = self.g.t
        nf = self.neon_f(t)
        nt = art_night(self.g.hour)
        lit = nf >= NEON_END
        surge = self.surge_f(t) if lit else None
        # sky + sea (+ glitter) on the sky's crane offset
        oys = oyp(cy, P_SKY)
        bd = sc.backdrop(int(round(nt * 12)))
        if oys:
            sky = bd.copy()
            sc.sea_glitter(sky, f, nt)
            c = Canvas(W, H, bd.a[0, 0])
            c.blit(sky, 0, oys)
        else:
            c = bd.copy()
            sc.sea_glitter(c, f, nt)
        fx.stars(c, f, seed=16, n=int(18 + 70 * nt), region=(0, oys - 58, W, 146))
        _clock_moon(c, self.g.hour)
        sc.streaks(c, f, cx, oyp(cy, P_STREAK), nt)
        # light added to the painted night sky (glows, beams, halos) is left as dark blots on a brighter
        # clock sky (dawn, day, sunset): kn says how much of it may show (daylight.sky_light)
        kn = daylight.sky_light(self.HOUR, self.g.hour)
        flashes = self._fireworks(c, t, cx, cy, kn)
        # far coast + Applewood, searchlights, sailboat
        fxo = -int(round(cx * P_FAR))
        oyf = oyp(cy, P_FAR)
        _xp(c, 'far_layer', fxo, oyf, nt)
        pa = t - self.spot_t.get('premiere', -999)
        boost = 0.0
        if 0 <= pa < PREM_LEN:
            boost = smooth(seg(pa, 0, 10)) * (1 - smooth(seg(pa, PREM_LEN - 30, PREM_LEN)))
        _searchlights(c, f, fxo, oyf, nt, boost, pa, kn)
        if nt > 0.3 and kn > 0.05:
            fx.glow(c, fxo + 48, 82 + oyf, 34, '#fff0d0', 0.10 * nt * kn, rings=4)
        if boost > 0:
            self._premiere(c, t, pa, fxo, oyf, boost, kn)
        _sailboat(c, f, fxo, oyf, nt)
        # back clouds
        _xp(c, 'back_clouds', -int(round(cx * P_BACK)) - 20,
                  oyp(cy, P_BACK), nt)
        # the city on its cloud bank
        oy = oyp(cy, 1.0)
        if lit and surge is None:
            self._city_lit(c, f, cx, oy, kn)
        else:
            gs = _gs_surge(surge) if surge is not None else (_gs_cascade(nf) if nf >= NEON_F0 else _gs_dusk)
            # the neon halos are light added to the painted night sky: they go with it (kn; by day the
            # sky is repainted around them and they would be left as dark blots)
            dl = sc.add_light if kn >= 0.999 else (lambda cc, arr, k=1.0: sc.add_light(cc, arr, k * kn))
            q = int(round(nt * 12))                 # unlit: sunset-rimmed at dusk, dark at night
            rebind(sc.draw_city, group_state=gs, add_light=dl,
                   city_layers=lambda: _city_layers_at(q))(c, f, cx, oy, night=False)
            _xp(c, 'front_bank', -cx, oy, nt)
            rebind(sc.bank_glow, group_state=gs)(c, f, cx, oy, False)
        self._neon_fx(c, t, nf, cx, oy, kn)
        # drifting front puffs (loop)
        _wrap_paste(c, 'fg_puffs', -int(round(cx * P_FG)) + int(f * 0.12) - 120,
                    142 + oyp(cy, P_FG), nt, FG_LOOP)
        # burst light on the towers, clouds and sky (render_c); none by day (sunlight outshines it)
        for (x, y, R, cc, a) in (flashes if kn > 0.05 else ()):
            k = (1 - a / 10.0) * min(1.0, kn)
            fx.glow(c, x, y, int(R * 1.6), cc, 0.30 * k, rings=4)
            if R >= 40:
                fx.glow(c, x, y + 40, 170, cc, 0.16 * k, rings=4)
                if a < 3:
                    c.blend(0, 0, W, H, mix(cc, 'white', 0.5), 0.10 * (3 - a) / 3.0)
        return c

    def _city_lit(self, c, f, cx, oy, kn=1.0):
        """sc.draw_city(night=True) + front bank + bank_glow, with the static light pre-summed."""
        L2, bank, light, halo_off = _lit_city()
        x0, x1 = max(0, cx), min(sc.WA, cx + W)
        if x1 <= x0:
            return
        _paste_rgba(c, L2, -cx, oy)
        _paste_rgba(c, bank, -cx, oy)
        full = np.zeros((H, W, 3), np.int16)
        full[oy:, x0 - cx:x1 - cx] = light[:H - oy, x0:x1]
        if kn < 0.999:                  # by day the neon halos fade (on the open sky they would turn into
            full[oy:, x0 - cx:x1 - cx] -= (halo_off[:H - oy, x0:x1] * (1 - kn)).astype(np.int16)  # dark blots)
        sc.add_light(c, full)
        _, _, bulbs, _ = sc.city_layers()
        for i in range(len(sc.GROUPS)):
            b = bulbs[i]
            if len(b):
                ph = (np.arange(len(b)) + f // 3) % 3
                for (bx, by), p in zip(b, ph):
                    c.px(bx - cx, by + oy, 'white' if p == 0 else ('neon_gold' if p == 1 else '#b0702a'))
            sc.sign_anim(c, f, i, 2, cx, oy)

    def _neon_fx(self, c, t, nf, cx, oy, kn=1.0):
        """The Grand Tower's star blazes on the hit (and on a surge): glow, rays and neon sparks. By day
        (sky light kn under 0.5: dawn, day, sunset) a glow on the bright sky would be left as a dark blot: the
        star flares as a solid burst of rays instead, in colours that show true after the relight."""
        day = kn < 0.5
        dc = self._dc if day else (lambda cc: cc)
        for t0 in (None if self.neon_t0 is None else self.neon_t0 + 4, self.surge_t0, self.blaze_t0):
            if t0 is None:
                continue
            a = t - t0
            if not 0 <= a < 56:
                continue
            sx, sy = STAR_A[0] - cx, STAR_A[1] + oy
            k = 1 - a / 56
            if day:
                if a < 14:                          # a white-hot core, then a gold one, shrinking
                    c.circle(sx, sy, 4 - a // 4, dc('neon_gold'))
                    c.circle(sx, sy, max(1, 2 - a // 5), dc('white'))
                if a < 22:                          # long diagonal rays (solid pixels, no glow)
                    d = 11 - a // 2
                    for j in range(3, d + 1):
                        if (j + a) % 3:
                            for ex, ey in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                                c.px(sx + ex * j, sy + ey * j, dc('neon_gold' if j > d - 3 else 'cream'))
            else:
                fx.glow(c, sx, sy, 24 + a // 2, 'neon_gold', 0.7 * k, rings=5)
                fx.glow(c, sx, sy, 9, 'white', 0.8 * k, rings=3)
            if a < 16:
                r = 16 - a // 2
                c.hline(sx - r, sx + r, sy, dc('white'))
                c.vline(sx, sy - r, sy + r, dc('white'))
            ui.burst(c, t, t0, sx, sy, tuple(dc(q) for q in ('neon_gold', 'neon_pink', 'neon_cyan', 'white')),
                     n=30, seed=7, power=0.9, life=50, gravity=0.04)
            if a < 20:                              # the entrance marquee sparks where SPACE was pressed
                if not day:
                    fx.glow(c, sx, 118 + oy, 14, 'neon_pink', 0.5 * (1 - a / 20), rings=3)
                ui.burst(c, t, t0, sx, 118 + oy, tuple(dc(q) for q in ('white', 'neon_pink', 'butter')), n=12,
                         seed=8, power=0.6, life=20, gravity=0.05, size=1)

    def _premiere(self, c, t, pa, fxo, oyf, boost, kn=1.0):
        """Applewood premiere: the sign's letters light up white in a chase, its star sparkles. By day
        (sky light kn under 0.5: dawn, day, sunset) there is no glow on the bright sky (it would be left as a dark
        blot): the letters chase in neon pink and the star throws pink and white sparks."""
        day = kn < 0.5
        lc = self._dc('neon_pink') if day else 'white'
        for k, (x, y) in enumerate(_letters()):
            if (pa // 3 - k) % 9 < 4 and boost > 0.3:
                if not day:
                    fx.glow(c, x + 2 + fxo, y + 3 + oyf, 8, 'neon_gold', 0.5 * boost, rings=2)
                text(c, x + fxo, y + oyf, 'APPLEWOOD'[k], lc, font='normal')
        sx = 16 + 9 * 7 + 2 + fxo + 3
        sy = sc.ridge_y(RIDGE, 16 + 66) - 3 + oyf + 3
        if not day:
            fx.glow(c, sx, sy, 16, 'neon_gold', 0.6 * boost, rings=3)
        fx.sparkle(c, sx, sy, pa, self._dc('white') if day else 'white', period=16, size=3)
        cols = tuple(self._dc(k) for k in ('neon_pink', 'white', 'baby_blue')) if day else ('gold', 'butter', 'white')
        ui.burst(c, t, self.spot_t['premiere'], sx, sy, cols, n=22, seed=11, power=0.7, life=44, gravity=0.03)
        for j in range(5):                              # flashbulbs popping along the studio town
            q = (pa + j * 17) % 23
            if q < 3 and pa < PREM_LEN - 20:
                fx.glow(c, fxo + 10 + int(hash01(j, pa // 23, 5) * 110), 116 + oyf, 6 - q * 2, 'white', 0.8,
                        rings=2)

    @staticmethod
    def _dc(cc):
        """A day effect's colour: as it should show (the whole picture is given to the relight in the
        colours it should show, _by_clock)."""
        return col(cc)

    def _fireworks(self, c, t, cx, cy, kn=1.0):
        """The show (film shells) and ambient shells, in the sky behind the coast (parallax P_FW). By
        day (sky light kn under 0.5: dawn, day, sunset) they are puffs of coloured smoke instead (_day_fireworks)."""
        dx = FW_DX - int(round(cx * P_FW))
        oy = oyp(cy, P_FW)
        flashes = []
        if kn < 0.5:                    # by day the smoke hangs on after the show's (and a shell's) end
            if self.show_t0 is not None and 0 <= t - self.show_t0 < SHOW_END - SHOW_F0 + DAY_EXTRA:
                _day_fireworks(c, _shift(sc.SHELLS, dx), SHOW_F0 + t - self.show_t0, oy, self._dc)
            am = self.amb_shell(t, 70 + DAY_EXTRA)
            if am is not None:
                sh, af = am
                _day_fireworks(c, _shift((sh,), dx), af, oy, self._dc)
            return flashes
        sf = self.show_f(t)
        if sf is not None:
            flashes += rebind(sc.fireworks, SHELLS=_shift(sc.SHELLS, dx))(c, sf, oy)
        am = self.amb_shell(t)
        if am is not None:
            sh, af = am
            flashes += rebind(sc.fireworks, SHELLS=_shift((sh,), dx))(c, af, oy)
        return flashes

    # ------------------------------------------------------------------ the plaza
    def render_plaza(self, f, cx, cy):
        return self._by_clock(lambda: self._plaza_day(f, cx), lambda: self._plaza_film(f, cx), True)

    def _plaza_day(self, f, cx):
        """The plaza in daylight: pastel towers and casino, unlit neon and bulbs, window glass, white
        water in the fountain, no glows, white clouds; graded for the clock's hour."""
        t = self.g.t
        cam = cx
        c = sc.backdrop(12).copy()
        _paste_rgba(c, _day_layer('far_resort'), -int(round(cam * sc.PAR_FAR_B)) - 20, 0)
        mo = -int(round(cam * sc.PAR_MID))
        before = c.a.copy()
        rebind(sc.ferris, fx=NOGLOW)(c, f, mo)
        _day_new(c, before)
        _paste_rgba(c, _day_layer('casino_layer'), mo, 0)
        before = c.a.copy()
        self._jackpot(c, t, f, mo, NOGLOW)
        _day_new(c, before)
        _paste_rgba(c, _day_layer('coaster_layer'), mo, 0)
        before = c.a.copy()
        rebind(sc.coaster_dyn, fx=NOGLOW)(c, f, mo)
        _day_new(c, before)
        po = -cam
        fa = t - self.spot_t.get('fountain', -999)
        fin = 0 <= fa < FIN_LEN
        jh = self._finale_h(fa) if fin else sc.jet_h
        _fountain(c, f, po, jh, lambda _f, _i: DAY_JET, glow=False)
        _paste_rgba(c, _day_layer('plaza_layer'), po, 0)
        for i in range(sc.NJ):
            x = sc.jet_x(i) + po
            if -20 < x < W + 20:
                c.hline(x - 3, x + 3, sc.RIM_Y, mix(DAY_JET, 'white', 0.4))
        if fin:
            self._finale_fx(c, t, fa, f, po, glow=False)
        fo = -int(round(cam * sc.PAR_FG_B))
        _paste_rgba(c, _day_layer('plaza_fg'), fo, 0)
        before = c.a.copy()
        sc.palm(c, 136 + fo, f, 0)
        sc.palm(c, 356 + fo, f, 2)
        _day_new(c, before)
        x = -int(round(cam * 1.4)) - 60 + int((f - 216) * 0.2)
        x0 = (x % FG_LOOP) - FG_LOOP
        while x0 < W:
            _paste_rgba(c, _day_clouds('fg_puffs'), x0, 150)
            x0 += FG_LOOP
        daylight.relight(c.a, DAY_ART_H, self.g.hour, glow=False)
        return c

    def _plaza_film(self, f, cx):
        t = self.g.t
        cam = cx
        c = _plaza_sky().copy()
        fx.stars(c, f, seed=26, n=80, region=(0, 0, W, 90))
        _clock_moon(c, self.g.hour)
        _paste_rgba(c, sc.far_resort(), -int(round(cam * sc.PAR_FAR_B)) - 20, 0)
        fx.light_rect(c, 0, 112, W, 30, '#8a3aa8', 0.18, fade='up')
        mo = -int(round(cam * sc.PAR_MID))
        sc.ferris(c, f, mo)
        _paste_rgba(c, sc.casino_layer(), mo, 0)
        self._jackpot(c, t, f, mo)
        _paste_rgba(c, sc.coaster_layer(), mo, 0)
        sc.coaster_dyn(c, f, mo)
        po = -cam
        fa = t - self.spot_t.get('fountain', -999)
        fin = 0 <= fa < FIN_LEN
        jh, jc = (self._finale_h(fa), self._finale_c(fa)) if fin else (sc.jet_h, sc.jet_col)
        _fountain(c, f, po, jh, jc)
        _paste_rgba(c, sc.plaza_layer(), po, 0)
        for i in range(sc.NJ):
            x = sc.jet_x(i) + po
            if -20 < x < W + 20:
                cc = jc(f, i)
                c.hline(x - 3, x + 3, sc.RIM_Y, mix(cc, 'white', 0.4))
                fx.light_rect(c, x - 9, sc.RIM_Y + 3, 19, 18, cc, 0.16, fade='down')
                fx.light_rect(c, x - 6, sc.FLOOR_B, 13, 12, cc, 0.18, fade='down')
        if fin:
            self._finale_fx(c, t, fa, f, po)
        fo = -int(round(cam * sc.PAR_FG_B))
        _paste_rgba(c, sc.plaza_fg(), fo, 0)
        for x in (30, 250, 470):
            if -20 < x + fo < W + 20:
                fx.glow(c, x + fo, 108, 16, 'butter', 0.30, rings=3)
        sc.palm(c, 136 + fo, f, 0)
        sc.palm(c, 356 + fo, f, 2)
        _wrap_paste(c, 'fg_puffs', -int(round(cam * 1.4)) - 60 + int((f - 216) * 0.2), 150, 1.0, FG_LOOP)
        return c

    def _jackpot(self, c, t, f, mo, fx_=fx):
        a = self.jack_f(t)
        if a is None:
            (_jackpot_idle if fx_ is fx else rebind(_jackpot_idle, fx=fx_))(c, f, mo)
            return
        f0 = f - a                                  # scene frame of the pull
        stops = tuple(f0 + s for s in SPIN_STOPS)
        reel = rebind(sc.reel_icon, REEL_STOP=stops)
        rebind(sc.jackpot, REEL_STOP=stops, reel_icon=reel, fx=fx_)(c, f, mo)
        jx, jy = sc.JX + mo, sc.JY
        if a < 8:                                   # the lever is pulled: the knob dips, sparks fly
            c.vline(jx + 51, jy + 6, jy + 16, '#7a7a9a')
            c.circle(jx + 52, jy + 5 + a, 2, 'red')
            c.px(jx + 51, jy + 4 + a, 'salmon')
        ui.burst(c, t, t - a, jx + 52, jy + 6, ('white', 'butter', 'neon_gold'), n=10, seed=3, power=0.5,
                 life=20, gravity=0.02)
        w = a - SPIN_STOPS[2]
        if 0 <= w:                                  # win: coins burst out of the sign
            ui.burst(c, t, t - w, jx + 25, jy + 13, ('gold', 'butter', 'white', 'neon_gold'), n=34, seed=9,
                     power=1.1, life=56, gravity=0.09)
            if w < 10:
                fx_.glow(c, jx + 25, jy + 13, 60, 'neon_gold', 0.35 * (1 - w / 10), rings=4)
            fx.confetti(c, t, t - w, jx + 25, jy + 4, n=30, seed=4,
                        colors=('gold', 'hot_pink', 'neon_cyan', 'lavender', 'butter'))
            for k in range(6):                      # gold stars twinkle around the sign
                ang = k / 6 * 6.283 + w * 0.05
                fx.sparkle(c, jx + 25 + int(math.cos(ang) * 36), jy + 13 + int(math.sin(ang) * 22), w + k * 4,
                           'butter', period=24, size=3)

    @staticmethod
    def _finale_h(fa):
        """Finale: jets surge from the centre outward to full height, hold, then settle back."""
        def jh(f, i):
            d = abs(i - (sc.NJ - 1) / 2) * 3
            u = fa - d
            base = sc.jet_h(f, i)
            if u < 0:
                return base
            k = ease_out(min(1.0, u / 9.0)) * (1 - smooth(seg(u, FIN_LEN - 40, FIN_LEN - 14)))
            top = int((120 - 4 * abs(i - (sc.NJ - 1) / 2)) * k + 2 * math.sin(u * 0.5 + i))
            return max(base, top)
        return jh

    @staticmethod
    def _finale_c(fa):
        def jc(f, i):
            return sc.JET_COLS[(i + fa // 4) % 4]
        return jc

    def _finale_fx(self, c, t, fa, f, po, glow=True):
        """Spray crowns and a cool wash of light over the plaza while the finale plays (by day: no wash)."""
        t0 = t - fa
        k = 1 - smooth(seg(fa, FIN_LEN - 30, FIN_LEN))
        if glow:
            fx.glow(c, 196 + po, 110, 110, 'neon_cyan', 0.16 * k, rings=4)
        for i in range(sc.NJ):
            d = int(abs(i - (sc.NJ - 1) / 2) * 3)
            x = sc.jet_x(i) + po
            if -30 < x < W + 30:
                top = sc.RIM_Y - (120 - 4 * abs(i - (sc.NJ - 1) / 2))
                ui.burst(c, t, t0 + d + 8, x, top, ('white', 'neon_cyan', 'baby_blue', 'magic'), n=10,
                         seed=40 + i, power=0.55, life=44, gravity=0.06, size=1 if i % 2 else 2)
