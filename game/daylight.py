"""The world clock's light in the places: re-light a diorama painted at one hour to the clock's hour.

Each place was painted at a fixed time of day (the film's hour for its scene: Region.HOUR). The world
map's grade (engine.loc_worldmap.grade_params: a per-channel multiply + add, and a desaturation at
night) is what turns daylight art into any hour. `relight` undoes the painted hour's grade and applies
the clock hour's, per channel through a lookup table (fast: three `take`s, ~1 ms), then desaturates if
the clock is darker. Warm light sources (lit windows, lamps, lanterns, fire: bright, saturated
yellow-orange pixels) keep their painted colour when the place gets darker, so towns glow at night.
Brightening a scene (a night scene by day) is an even, hue-neutral lift, limited, so it reads as soft
evening light instead of shifting colours.

`light_tint(hour)` lights the balloon and other sprites like the place (Frame.balloon tint).
"""
from functools import lru_cache

import numpy as np

from engine import loc_worldmap as LW

DARKEN_STRENGTH, BRIGHTEN_STRENGTH = 0.92, 0.62


def _q(h):
    return round(h * 10) / 10.0          # 6-minute steps: the tables are cached


SKY_STEP = 20                   # the clock's sky (and what places derive from it) moves on in 1/20 h steps


def sky_q(hour):
    """The hour the clock's sky is drawn for: 3-minute steps (3 s of play), small enough that the sky never
    visibly jumps (quarter hours did: the whole sky changed by up to 20 levels in one frame at dusk).
    Places that build their own pieces of the clock's sky use this too, so all of it moves together."""
    return round(hour * SKY_STEP) / SKY_STEP


@lru_cache(maxsize=256)
def _tables(from_h, to_h, soft=False):
    """(3 x 256 uint8 lookup, desaturation amount, darker?) from the painted hour to the clock hour."""
    mf, af, nf = LW.grade_params(from_h)
    mt, at, nt = LW.grade_params(to_h)
    v = np.arange(256, dtype=np.float32)
    lum_f = sum(m for m in mf) / 3 + sum(af) / 765
    lum_t = sum(m for m in mt) / 3 + sum(at) / 765
    darker = lum_t < lum_f - 0.02
    s = DARKEN_STRENGTH if darker else BRIGHTEN_STRENGTH * (0.45 if soft else 1.0)
    if darker and nf > 0.5:          # painted at night already (all its lights on): only a gentle dimming
        s *= 0.3
    lut = np.empty((3, 256), np.uint8)
    if darker:                       # undo the painted hour's grade, apply the clock's (per channel)
        for ch in range(3):
            base = np.maximum((v - af[ch]) / max(0.2, mf[ch]), 0)      # no cap: only the result is clipped
            out = base * mt[ch] + at[ch]
            lut[ch] = np.clip(v + (out - v) * s, 0, 255).astype(np.uint8)
    else:                            # brighter: an even, hue-neutral lift (undoing a blue night grade
        gain = 1 + (min(1.7, lum_t / max(0.05, lum_f)) - 1) * s          # per channel turns it green)
        # a strong lift (a night-painted place seen by day) gets a soft knee; the small day-to-day lifts
        # of day-painted places (gain under ~1.04) stay exactly as they were
        kw = min(1.0, max(0.0, (gain - 1.04) / 0.12))
        for ch, warm in enumerate((1.03, 1.0, 0.96)):
            lin = v * gain * warm + (gain - 1) * 10
            # soft knee: the full lift up to the knee, then it rolls off smoothly to none at 255, so lit
            # neon and pale towers keep their detail instead of clipping to flat white. The knee sits at
            # 150, lower for a strong lift (it keeps the top ~70 levels for the highlights).
            knee = min(150.0, (185 - (gain - 1) * 10) / (gain * warm))
            x = np.clip((v - knee) / (255 - knee), 0, 1)
            soft = np.maximum(v, v + (lin - v) * (1 - x) ** 1.5)      # never darker than painted
            out = lin + (soft - lin) * kw
            lut[ch] = np.clip(np.maximum.accumulate(out), 0, 255).astype(np.uint8)
    desat = max(0.0, 0.3 * (nt - nf)) * s
    glow = darker and (nt > 0.2 or lum_t < lum_f * 0.85)      # only real dusk / night saves the lamps
    return lut, desat, glow


# interiors: lit by their own lamps whatever the hour, always shown as painted
INTERIORS = {('cloudsdale', 1), ('castle_two_sisters', 1)}
SNOW_DAY = ('#f4f8ff', '#dfe8fb', '#b9c8ee')    # day snow: sunlit, plain, shadow
SNOWY = {'yakyakistan', 'crystal_empire'}       # night snow scenes: their snow turns white by day


@lru_cache(maxsize=8)
def _snow_lut(kq):
    """32768-entry lookup on 5-bit-per-channel colours: (is night snow, day-snow colour blended by k)."""
    from engine import col
    q = np.arange(32768)
    r, g, b = ((q >> 10) & 31) * 8 + 4, ((q >> 5) & 31) * 8 + 4, (q & 31) * 8 + 4
    lum = (r * 77 + g * 150 + b * 29) >> 8
    snow = (b >= r) & (b >= g) & (b - np.minimum(r, g) < 60) & (lum > 105)
    hi, mid, lo = (np.array(col(c), np.int32) for c in SNOW_DAY)
    tgt = np.where((lum > 175)[:, None], hi, np.where((lum > 140)[:, None], mid, lo))
    w = kq / 10 * 0.6
    rgb = np.stack([r, g, b], 1)
    out = np.clip(rgb * (1 - w) + tgt * w, 0, 255).astype(np.uint8)
    return snow, out


def _day_snow(a, k, t):
    """Night snow (light lilac / periwinkle) seen by day turns white with blue shadows, and the sun
    glints on it. k 0..1: how much daylight. One table lookup per pixel (~0.6 ms)."""
    snow_t, out_t = _snow_lut(int(round(k * 10)))
    q = ((a[..., 0] >> 3).astype(np.uint16) << 10) | ((a[..., 1] >> 3).astype(np.uint16) << 5) | (a[..., 2] >> 3)
    snow = snow_t[q]
    np.copyto(a, out_t[q], where=snow[..., None])
    if k > 0.5:                     # sun glints: a few snow pixels sparkle (re-picked every 6 steps)
        rs = np.random.RandomState((t // 6) % 997)
        ys = rs.randint(0, a.shape[0], 60)
        xs = rs.randint(0, a.shape[1], 60)
        hit = snow[ys, xs]
        a[ys[hit][:14], xs[hit][:14]] = (255, 255, 255)


def relight(a, from_h, to_h, t=0, snow=False, glow=True):
    """Re-light the RGB array `a` (in place) from the painted hour to the clock hour. snow: the place's
    snow was painted at night and turns into day snow when the clock says day. glow: keep the small warm
    and magic-hued spots glowing at dusk and night (_lights; False: the region keeps its own lights)."""
    fq, tq = _q(from_h), _q(to_h)
    if abs(fq - tq) < 0.05:
        return
    lut, desat, keep = _tables(fq, tq, snow)          # snow scenes: the day snow already brightens them
    keep = keep and glow
    if snow:
        day = 1.0 - night(to_h)
        day_painted = 1.0 - night(from_h)
        if day - day_painted > 0.2:
            _day_snow(a, day - day_painted, t)
    if keep:
        glow = _lights(a)                                        # warm lights keep shining
    off = night(from_h) - night(to_h)                            # painted at night, seen by day:
    lamps = _lamps(a) if off > 0.3 else None                     # its lamps and windows switch off
    out = np.empty_like(a)
    for ch in range(3):
        out[..., ch] = lut[ch].take(a[..., ch])
    if desat > 0.01:
        f = out.astype(np.float32)
        lum = f @ np.array([0.3, 0.55, 0.15], np.float32)
        f += (lum[..., None] - f) * desat
        out = np.clip(f, 0, 255).astype(np.uint8)
    _finish(out, a, glow if keep else None, lamps, off)
    a[...] = out


def unlight(rgb, from_h, to_h):
    """The colour to paint into a scene painted at from_h so that relight(.., from_h, to_h) shows `rgb`
    (as near as the light allows): for effects drawn into a night scene that must keep their own colour
    by day (Las Pegasus' daytime fireworks). A warm bright colour is held just under the lamp test
    (_lamps), or it would be switched off as a painted lamp."""
    return _unlight(tuple(int(v) for v in rgb), _q(from_h), _q(to_h))


def unlight_array(a, from_h, to_h):
    """unlight for a whole RGB array at once (vectorised: the same colour per pixel as unlight gives)."""
    fq, tq = _q(from_h), _q(to_h)
    if abs(fq - tq) < 0.05:
        return a.copy()
    lut = _tables(fq, tq)[0]
    p = np.empty(a.shape, np.int32)
    for ch in range(3):
        p[..., ch] = np.minimum(255, np.searchsorted(lut[ch], a[..., ch]))
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    m = (r > 200) & (g > 140) & (b < 175) & (r > b + 55)            # it would pass for a lamp
    if m.any():
        p[m] = (p[m] * (200.0 / r[m])[:, None]).astype(np.int32)
    return p.astype(np.uint8)


@lru_cache(maxsize=1024)
def _unlight(rgb, fq, tq):
    if abs(fq - tq) < 0.05:
        return rgb
    lut = _tables(fq, tq)[0]
    p = [min(255, int(np.searchsorted(lut[ch], rgb[ch]))) for ch in range(3)]
    r, g, b = p
    if r > 200 and g > 140 and b < 175 and r > b + 55:          # it would pass for a lamp
        k = 200.0 / r
        p = [int(v * k) for v in p]
    return tuple(p)


def _finish(out, a, glow, lamps, off):
    """Keep glowing pixels as painted; switch painted lamps off by day."""
    if glow is not None:
        out[glow] = a[glow]
    if lamps is not None and lamps.any():
        k = min(1.0, off)
        unlit = np.array((74, 80, 120), np.float32)              # dark window glass / a cold lamp
        o = out[lamps].astype(np.float32)
        out[lamps] = (o + (unlit - o) * k * 0.85).astype(np.uint8)


def _lamps(a):
    """Warm lights painted in a night scene (lit windows and doors, lamps, torches) and their dithered
    glow: warm yellow-orange pixels brighter than their 5x5 surroundings."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    warm = (r > 200) & (g > 140) & (b < 175) & (r > b + 55)
    rows = np.nonzero(warm.any(axis=1))[0]
    if len(rows) == 0:
        return warm
    y0, y1 = max(0, rows[0] - 2), min(a.shape[0], rows[-1] + 3)     # only the rows with candidates
    sub = a[y0:y1]
    lum = (sub[..., 0] >> 2).astype(np.uint16) + (sub[..., 1] >> 1) + (sub[..., 2] >> 3)
    p = np.pad(lum, 2, mode='edge').astype(np.uint32)
    cs = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = lum.shape
    box = cs[5:5 + h, 5:5 + w] - cs[:h, 5:5 + w] - cs[5:5 + h, :w] + cs[:h, :w]
    out = np.zeros_like(warm)
    out[y0:y1] = warm[y0:y1] & (lum.astype(np.int32) * 25 - box.astype(np.int32) > 5 * 25)
    return out


def _lights(a):
    """Light sources kept glowing at night: small warm spots (windows, lamps, lanterns, fire) and things
    that glow by themselves in magic / neon hues (rings, crystals, neon, rainbows), clearly brighter
    than their 7x7 surroundings; never big painted areas, sunlit rims or foliage."""
    r, g, b = (a[..., k].astype(np.int32) for k in range(3))     # (int16 sums wrapped: bright lamps went
    warm = (r > 238) & (g > 95) & (b < 140) & (r - b > 95)            # negative and were never kept)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    # vivid: bright and saturated ((mx - mn) / mx > 0.42), in the magic / neon hues only (cyan, blue,
    # violet, magenta): warm colours must pass the stricter lamp test, foliage greens never glow
    vivid = (mx > 200) & ((mx - mn) * 12 > mx * 5)
    vivid &= ~((r >= b) & (g >= b) & (r + 20 >= g))           # red .. orange .. yellow
    vivid &= ~((r > g) & (b * 5 < r * 4))                     # rose / pink (sunset-lit clouds)
    vivid &= ~((g >= r) & (g > b + 25))                       # green .. lime
    out = np.zeros(warm.shape, bool)
    if not (warm.any() or vivid.any()):
        return out
    lum = ((r * 77 + g * 150 + b * 29) >> 8).astype(np.int32)
    cs = np.pad(np.pad(lum, 3, mode='edge').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = lum.shape
    lift = lum * 49 - (cs[7:7 + h, 7:7 + w] - cs[:h, 7:7 + w] - cs[7:7 + h, :w] + cs[:h, :w])
    warm &= lift > 22 * 49
    if warm.any():                  # a lamp is a small spot: long warm runs are sunlit rims, not lights
        from scipy import ndimage
        lab, n = ndimage.label(warm, structure=np.ones((3, 3)))
        wy, wx = np.nonzero(lab)
        ls = lab[wy, wx]
        order = np.argsort(ls, kind='stable')
        ls, wy, wx = ls[order], wy[order], wx[order]
        starts = np.r_[0, np.nonzero(np.diff(ls))[0] + 1]
        ww = np.maximum.reduceat(wx, starts) - np.minimum.reduceat(wx, starts)
        hh = np.maximum.reduceat(wy, starts) - np.minimum.reduceat(wy, starts)
        ok = np.zeros(n + 1, bool)
        ok[ls[starts]] = (ww < 8) & (hh < 10)                  # a bounding box of at most 8 x 10
        out |= ok[lab]
    out |= vivid & (lift > 10 * 49)
    return out


@lru_cache(maxsize=64)
def _tint(hq):
    mul, add, night = LW.grade_params(hq)
    if min(mul) > 0.93:
        return None
    rgb = tuple(int(max(0, min(255, m * 255 + a))) for m, a in zip(mul, add))
    return ('#%02x%02x%02x' % rgb, 0.85)


def light_tint(hour):
    """The tint (colour, amount) for the balloon and sprites at a clock hour, or None in full daylight."""
    return _tint(round(hour * 4) / 4.0)


def night(hour):
    return LW.grade_params(hour)[2]


MAP_DIM = 0.8                   # how far the map's sky things (rings, rainbows, bunting) take the clock's grade
MAP_TINT = 0.6                  # ...and the balloons on the map (they stay easy to see)


def is_day(hour):
    """True while the clock's grade leaves colours as they are (about 09:00-15:30)."""
    mul, add, nt = LW.grade_params(hour)
    return min(mul) > 0.97 and max(add) < 3 and nt < 0.02


def dim_new(a, before, hour, k=MAP_DIM):
    """Light what was drawn since `before` (a copy of the already lit picture: the map's rings, race gates,
    rainbows and bunting; a place's wind streaks and rain) like the picture under it: the clock's grade,
    k of the way. Nothing by day."""
    if is_day(hour):
        return
    m = np.any(a != before, axis=2)
    if not m.any():
        return
    px = a[m][None].copy()
    LW._apply_grade(px, hour, k)
    a[m] = px[0]


def map_tint(hour):
    """The balloons' tint on the world map at a clock hour (softer than in the places), or None by day."""
    return mix_tint(None, light_tint(hour), MAP_TINT)


def tint_gap(a, b):
    """0..1: how differently tints a and b (colour, amount; None: none) light a white sprite (mean channel
    difference of their multiply colours)."""
    from engine import col

    def eff(tn):
        if tn is None:
            return np.ones(3)
        c, k = tn
        return 1 - k * (1 - np.array(col(c), np.float64) / 255)
    return float(np.abs(eff(a) - eff(b)).mean())


def mix_tint(a, b, w):
    """A tint (colour, amount) between tints a and b (either may be None: no tint), w = 0..1 (in steps of
    0.1, so the tinted balloon sprites stay few)."""
    w = round(min(1.0, max(0.0, w)) * 10) / 10
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    ca, ka = a if a else ('#ffffff', 0.0)
    cb, kb = b if b else ('#ffffff', 0.0)
    from engine import col
    pa, pb = col(ca), col(cb)
    rgb = tuple(int(round(x + (y - x) * w)) for x, y in zip(pa, pb))
    k = ka + (kb - ka) * w
    return ('#%02x%02x%02x' % rgb, round(k, 2)) if k > 0.02 else None


# ------------------------------------------------------------------ the sky
# Each place's Area may give `sky(f, cx, cy)`: its sky alone (gradient, stars, moon, sun, aurora, as
# its render draws them first). Every pixel of the full frame still equal to that is open sky; when
# the clock's hour looks different from the painted one, those pixels are repainted with the clock's
# sky (gradient, sun or moon and stars), blended in as the difference grows.

SKY_TOL = 40                    # L1: a light haze or glow drawn over the sky still counts as sky
SKY_HOLE = 500                  # enclosed bits up to this size inside the sky (a sun disc) count too...
SKY_HOLE_D = 110                # ...if their mean L1 to the sky render is under this (clouds are not)


def sky_pixels(frame, sky):
    """The frame's open-sky pixels: (nearly) equal to the area's sky alone, plus small enclosed holes."""
    from scipy import ndimage
    d = (np.maximum(frame, sky) - np.minimum(frame, sky)).sum(axis=2, dtype=np.uint16)
    m = d < SKY_TOL
    rows = np.nonzero(m.any(axis=1))[0]
    if len(rows) == 0:
        return m
    y1 = rows[-1] + 1                                   # the sky's rows only (it is up top)
    sub = m[:y1]
    # holes: non-sky regions that touch no edge of the sky rows (one labelling pass; full size, so thin
    # things standing in the sky such as a windpump stay scenery)
    hl, hn = ndimage.label(~sub)
    if hn:
        edge = np.unique(np.concatenate([hl[0], hl[-1], hl[:, 0], hl[:, -1]]))
        sizes = np.bincount(hl.ravel(), minlength=hn + 1)
        sizes[edge] = 1 << 30
        cand = np.nonzero(sizes <= SKY_HOLE)[0]
        cand = cand[cand > 0]
    # lone specks inside the sky (dither dots, a missed star): a non-sky pixel with 7+ of its 8
    # neighbours sky is sky (a thin pole or wire has at most 6)
    nb = np.zeros(sub.shape, np.uint8)
    p = np.pad(sub, 1, mode='edge').view(np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy != 1 or dx != 1:
                nb += p[dy:dy + sub.shape[0], dx:dx + sub.shape[1]]
    sub |= nb >= 7
    if hn and len(cand):
        dsum = np.bincount(hl.ravel(), weights=d[:y1].ravel(), minlength=hn + 1)
        near = dsum / np.maximum(sizes, 1)
        # a hole is the sky's own (a hazed sun disc) when it is small AND close to the sky render;
        # a small cloud or a pole floating in the sky is far from it and stays scenery
        small = (sizes <= SKY_HOLE) & (np.nan_to_num(near, nan=999) < SKY_HOLE_D)
        small[0] = False
        sub |= small[hl]
    return m


def sky_weight(from_h, to_h):
    """0 (keep the painted sky) .. 1 (the clock's sky): how different the two hours' skies look (the sky
    presets' colours compared: a sunset sky and a night sky differ even when both count as 'night').
    Quick to reach 1: a half-blended sky looks muddy."""
    def five(h):                     # presets have 4 or 5 colours: sample both at 5 points
        cs = _sky_cols(h)
        return np.array([cs[min(len(cs) - 1, i * len(cs) // 5)] for i in range(5)], np.float32)
    a, b = five(from_h), five(to_h)
    ka, xa, _ya = _body_pos(from_h)
    kb, xb, _yb = _body_pos(to_h)
    if ka == kb == 'sun' and abs(xa - xb) > 60:      # same colours, but the sun is elsewhere (morning vs
        return 1.0                                   # afternoon): the painted sun must go
    d = float(np.abs(a - b).sum(axis=1).mean())
    return max(0.0, min(1.0, (d - 25) / 15))        # steep: a half-blended sky shows a ghost sun


def sky_light(from_h, to_h):
    """0..1: how much of the light a scene paints into its own sky at from_h (searchlight beams, firework
    glows, neon halos) may show at the clock's to_h. Such light makes its pixels differ from the sky
    layer, so paint_sky leaves them in the painted sky's colours: right while the painted sky is kept
    or the clock's sky is darker (a pale beam on the night), wrong once the clock's sky is brighter
    (the painted dusk shows through as dark stripes and blots on a dawn, day or sunset sky)."""
    return _sky_light(_q(from_h), _q(to_h))


@lru_cache(maxsize=512)
def _sky_light(fq, tq):
    def lum(h):
        return float((np.array(_sky_cols(h), np.float32) @ np.array([0.3, 0.55, 0.15], np.float32)).mean())
    r = lum(tq) / max(1.0, lum(fq))
    return max(1.0 - sky_weight(fq, tq), min(1.0, max(0.0, (1.0 - r) / 0.25)))


_KEYS = [(0, 'night'), (4.8, 'predawn'), (5.6, 'dawn'), (6.4, 'sunrise'), (7.5, 'morning'), (11, 'noon'),
         (15, 'afternoon'), (17.2, 'golden'), (18.6, 'sunset'), (19.4, 'dusk'), (20.4, 'night'), (24, 'night')]


def _sky_cols(hour):
    """The sky preset colours at an hour, blended between the neighbouring presets (top -> horizon)."""
    from engine.palette import SKIES
    from engine import col
    h = hour % 24
    for (h0, n0), (h1, n1) in zip(_KEYS, _KEYS[1:]):
        if h0 <= h < h1:
            k = (h - h0) / (h1 - h0)
            k = max(0.0, (k - 0.35) / 0.65)                    # hold each preset a while, then blend
            k = k * k * (3 - 2 * k)                            # smoothly into the next (no step at the keys)
            a, b = _stops(n0), _stops(n1)
            return [tuple(int(round(x + (y - x) * k)) for x, y in zip(ca, cb)) for ca, cb in zip(a, b)]
    return list(_stops('night'))


SKY_OVERRIDE = {   # the film's 'afternoon' ends in a peach horizon: by 16:00 it read as sunset arriving early
    'afternoon': ['sky', '#6db6f0', 'light_sky', '#e4f2fb'],
}
SKY_STOPS = 6                   # every preset as the same 6 evenly spaced stops: neighbouring presets line up


@lru_cache(maxsize=16)
def _stops(name):
    """A sky preset's colours resampled (by position, linear) to SKY_STOPS stops, top -> horizon. The presets
    have 4 to 6 colours; mapped by index into each blend's own count, the bands jumped where a 4- or 5-colour
    preset met a longer one (04:48, 06:24, 07:30: the whole sky changed by up to 24 levels in one frame)."""
    from engine.palette import SKIES
    from engine import col
    cs = np.array([col(c) for c in SKY_OVERRIDE.get(name, SKIES[name])], np.float64)
    out = []
    for j in range(SKY_STOPS):
        p = j * (len(cs) - 1) / (SKY_STOPS - 1)
        i = min(int(p), len(cs) - 2)
        f = p - i
        out.append(tuple(int(round(v)) for v in cs[i] * (1 - f) + cs[i + 1] * f))
    return tuple(out)


@lru_cache(maxsize=160)
def _sky_strip(hq, top, height):
    """The clock's sky as a W-wide strip from world y `top`, `height` rows (dithered gradient)."""
    from engine import Canvas, W
    c = Canvas(W, height)
    c.gradient(0, 0, W, height, _sky_cols(hq), soft=0.6)
    return c.a


def _body_pos(hour):
    """(kind, screen x, screen y) of the sun (day) or the moon (night) on its arc, or None."""
    from engine import W
    h = hour % 24
    def sink(k):                    # 0..1 -> how far the disc has gone down behind the horizon (px)
        k = min(1.0, max(0.0, k))
        return 70 * k * k * (3 - 2 * k)
    if 5.6 <= h <= 19.2:            # the sun rises from below the land and sets behind it (it used to
        u = (h - 5.6) / (19.2 - 5.6)                    # wink out mid-sky at 19:12 and pop in at 05:36)
        y = 70 - 58 * math.sin(math.pi * u) + sink((h - 18.7) / 0.5) + sink((6.1 - h) / 0.5)
        return 'sun', int(W * (0.06 + 0.88 * u)), int(y)
    hn = (h - 19.2) % 24
    u = hn / (24 - 19.2 + 5.6)
    y = 64 - 50 * math.sin(math.pi * u) + sink((19.7 - h) / 0.5 if h > 12 else 0) + sink((h - 5.1) / 0.5 if h < 12 else 0)
    return 'moon', int(W * (0.06 + 0.88 * u)), int(y)


def paint_sky(a, mask, hour, w, cx, cy, top, height, t, moon=True):
    """Repaint the sky pixels of frame `a` for the clock's hour (blend weight w): gradient anchored in
    the area's world (top / height in world y), stars with a little parallax at night, the sun or moon
    on its arc (drawn only on sky, so it sets behind the scenery). moon=False: a place with a moon of its
    own (it would show two at once while the skies cross-fade)."""
    from engine import H, W, hash01, col
    hq = sky_q(hour)
    strip = _sky_strip(hq, top, height)
    y0 = int(cy) - top
    rows = np.clip(np.arange(H) + y0, 0, height - 1)
    sky = strip[rows].copy()
    night = LW.grade_params(hour)[2]
    if night > 0.25:                                           # stars (a little parallax), twinkling
        sx, sy, sc_ = _STARS
        n = int(len(sx) * night)
        xs = (sx[:n] - int(cx * 0.06)) % 400 - 40
        ok = (xs >= 0) & (xs < W) & (((t // 7) + np.arange(n)) % 11 != 0)
        sky[sy[:n][ok], xs[ok]] = sc_[:n][ok]
    kind, bx, by = _body_pos(hour)
    if kind == 'sun':
        _disc(sky, bx, by, 7, col('butter'), col('gold'), glow=col('cream'))
    elif moon:
        _disc(sky, bx, by, 6, col('cream'), col('#d8d0c0'), glow=None)
        sky[by - 2, bx + 1] = col('#c8c0b0')
        sky[by + 1, bx - 2] = col('#c8c0b0')
    if w < 0.999:
        k = int(round(w * 256))
        sky = ((a.astype(np.uint16) * (256 - k) + sky.astype(np.uint16) * k) >> 8).astype(np.uint8)
    np.copyto(a, sky, where=mask[..., None])


def _stars():
    from engine import H, hash01, col
    n = 70
    sx = np.array([int(hash01(i, 71) * 400) for i in range(n)])
    sy = np.array([int(hash01(i, 72) * H * 0.7) for i in range(n)])
    sc_ = np.array([col('white') if i % 4 else col('light_sky') for i in range(n)], np.uint8)
    return sx, sy, sc_


_STARS = _stars()


def _disc(img, x, y, r, c1, c2, glow=None):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[max(0, y - r - 4):min(h, y + r + 5), max(0, x - r - 4):min(w, x + r + 5)]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    sub = img[max(0, y - r - 4):min(h, y + r + 5), max(0, x - r - 4):min(w, x + r + 5)]
    if glow is not None:
        g = (d2 <= (r + 3) ** 2) & (d2 > r * r) & ((xx + yy) % 2 == 0)
        sub[g] = (sub[g] * 0.5 + np.array(glow) * 0.5).astype(np.uint8)
    sub[d2 <= r * r] = c2
    sub[d2 <= (r - 1) ** 2] = c1


import math  # noqa: E402
