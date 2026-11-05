"""Frame layers and the compositor that turns them into the window image.

Why layers: the world is drawn at an integer camera (crisp pixel art), but a camera that only moves in
whole art pixels judders, and a balloon rounded separately from the camera wobbles against the ground.
So a `Frame` carries
  - `world`  : the 320x180 world drawn with its top-left at the integer world point `origin`
               (plus world-anchored tags / plates),
  - `cam`    : the true (float) camera = world point at the view's top-left,
  - `focus`  : the world point the camera follows (the balloon), or None,
  - `sprites`: the balloon (and its shadow) at float WORLD positions,
  - `front`  : world-anchored art IN FRONT of the balloon (KEY canvas, same origin as `world`; e.g. the
               near half of a cloud hoop the balloon flies through), or None,
  - `hud`    : screen-fixed UI on a KEY-coloured canvas (opaque pixels only),
  - `iris`   : (t, cx, cy) of an iris wipe over everything,
  - `dim`    : darken the world and sprites under the HUD (the pause menu).
`compose(frame, k)` scales by k (nearest) and places the world and the sprites with 1/k art-px
precision, so motion is smooth while every art pixel stays a sharp k x k block. The camera's window
pixel is derived from the focus: cam_px = round(focus*k) - round((focus - cam)*k), so the balloon moves
smoothly both on screen and over the ground (rounding them separately makes it flicker by a pixel).
The output shows art px 0..318 x 0..178 (one column/row less than the canvas) so the shifted world
always covers the view.
"""
import math

import numpy as np

from engine import W, H, Canvas, col
from engine.atlas import _balloon_sprite, _B_SIZES, iris as draw_iris

KEY = (255, 0, 254)
VIEW_W, VIEW_H = W - 1, H - 1
_KEY = np.array(KEY, np.uint8)


class Frame:
    def __init__(self, world=None, origin=(0, 0), cam=None, focus=None):
        self.world = world if world is not None else Canvas()
        self.origin = origin
        self.cam = cam if cam is not None else tuple(float(v) for v in origin)
        self.focus = focus
        self.sprites = []
        self.front = None               # world-anchored layer over the sprites (see the module doc)
        self.hud = Canvas(color=KEY)
        self.iris = None
        self.flash = 0.0                # 0..1 white over everything (the postcard camera)
        self.dim = False                # darken the world and sprites (under the HUD): the pause menu

    def balloon(self, wx, wy, f, size='m', flame=False, bob=True, tint=None, pattern='classic', hop=0.0,
                cargo=None, wet=0.0, snow=0.0):
        """Balloon whose basket bottom-centre is at world (wx, wy) (floats); f may be fractional.
        tint = (colour, amount): light the balloon like its surroundings (multiply blend, 0..1).
        pattern: an envelope pattern from PATTERNS (the balloon wardrobe); hop: extra lift in px;
        cargo: an errand item key (game/items.py) hanging under the basket, or None; wet / snow (0..1):
        rain darkens the envelope, snow settles on its top (game/weather.py)."""
        self.sprites.append(('balloon', wx, wy - hop, f, size, flame, bob, tint, pattern, cargo,
                             _q(wet), _q(snow)))

    def shadow(self, wx, wy, rx):
        self.sprites.append(('shadow', wx, wy, rx))

    def ride(self, layer, x, y, dx=0.0, dy=0.0):
        """Small art that rides a moving sprite (a balloon friend's heart or SPACE bubble): the non-KEY
        pixels of `layer` (a KEY canvas) with its top-left at art px (x, y) of the integer world view (the
        coordinates of `world` / `front`), shifted by (dx, dy) art px, the sprite's sub-pixel offset
        from those whole pixels (e.g. tx - round(tx), plus its bob), so it glides with the sprite instead
        of stepping against it. Drawn in sprite order (over what was added before it)."""
        m = np.any(layer.a != _KEY, axis=2)
        if m.any():
            self.sprites.append(('ride', layer.a, m, int(x), int(y), float(dx), float(dy)))


def _up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


def see_through(fr, margin=2):
    """Where the HUD layer (counters, plates, toasts, tips) covers the player's balloon, a checkerboard of
    its pixels turns clear, so the balloon shows through the overlay instead of vanishing under it.
    The player's balloon is the balloon sprite at the frame's focus x."""
    if fr.focus is None or fr.cam is None:
        return
    best = None
    for s in fr.sprites:
        if s[0] == 'balloon':
            d = abs(s[1] - fr.focus[0]) * 10 + abs(s[2] - fr.focus[1])
            if best is None or d < best[0]:
                best = (d, s)
    if best is None:
        return
    _, wx, wy, f, size, flame, bob, tint, pattern, cargo, wet, snow = best[1]
    R, He, rope, bw, bh = _B_SIZES[size]
    sx, sy = wx - fr.cam[0], wy - fr.cam[1]
    x0 = int(math.floor(sx - R - 2 - margin))
    x1 = int(math.ceil(sx + R + 3 + margin))
    y0 = int(math.floor(sy - (He + rope + bh + 2) - margin))
    y1 = int(math.ceil(sy + (9 if cargo else 1) + margin))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = fr.hud.a[y0:y1, x0:x1]
    on = np.any(sub != _KEY, axis=2)
    if not on.any():
        return
    yy, xx = np.indices(on.shape)
    on &= (yy + y0 + xx + x0) % 2 == 0
    sub[on] = _KEY


def bob_dy(f, size):
    """A bobbing balloon sprite's float lift (art px) at frame f: what rides it (Frame.ride) adds it too."""
    return math.sin(f * 2 * math.pi / 72.0) * (1 if size in ('xs', 's') else 1.4)


_SPR = {}


# The balloon wardrobe: envelope patterns unlocked by collecting stamps. Each swaps the six stripe
# colours of the shared sprite (engine.atlas: base, shade and highlight of the two alternating gores);
# 'rainbow' paints the gores in rainbow bands instead.
PATTERNS = [   # (key, name, stamps needed, (base0, base1, shade0, shade1, hi0, hi1))
    ('classic', 'EXPLORER', 0, None),
    ('orchard', 'ORCHARD', 6, ('red', 'leaf', 'crimson', 'green', 'salmon', 'pale_leaf')),
    ('sky', 'CLEAR SKY', 12, ('baby_blue', 'white', 'sky', 'cloud', 'light_sky', 'white')),
    ('sunset', 'SUNSET', 18, ('orange', 'hot_pink', 'vermilion', 'rose', 'amber', 'pink')),
    ('mint', 'MINT CREAM', 24, ('mint', 'cream', 'teal', 'sage', 'mint_cream', 'white')),
    ('crystal', 'CRYSTAL', 30, ('crystal_lt', 'crystal_pk', 'crystal', 'magic', 'ice', 'pink')),
    ('lemon', 'LEMONADE', 36, ('butter', 'white', 'gold', 'cloud', 'cream', 'white')),
    ('royal', 'ROYAL', 42, ('royal', 'gold', 'indigo', 'amber', 'royal_lt', 'butter')),
    ('night', 'STARRY NIGHT', 48, ('navy', 'royal', 'night', 'indigo', 'royal_lt', 'periwinkle')),
    ('rainbow', 'RAINBOW', 54, 'rainbow'),
    # Phase B: earned by helping places, races and the photo album (need = 'helped:N' / 'gold:N' / 'photos:N')
    ('courier', 'COURIER', 'helped:5', ('tan', 'red', 'clay', 'crimson', 'sand', 'salmon')),
    ('parcel_post', 'PARCEL POST', 'helped:10', ('sky', 'cream', 'blue', 'tan', 'light_sky', 'white')),
    ('aurora', 'AURORA', 'helped:20', ('mint', 'violet', 'teal', 'purple', 'mint_cream', 'lilac')),
    ('checkered', 'CHEQUERED', 'gold:1', ('white', 'charcoal', 'cloud', 'ink', 'white', 'slate')),
    ('snapshot', 'SNAPSHOT', 'photos:10', ('turquoise', 'butter', 'teal', 'gold', 'aqua', 'cream')),
    ('flock', 'FLOCK', 'friends:15', ('pink', 'sky', 'rose', 'blue', 'cotton', 'light_sky')),
]


def need_text(need):
    """How a pattern is unlocked, for the wardrobe page."""
    if isinstance(need, int):
        return f'{need} STAMPS'
    kind, n = need.split(':')
    return {'helped': f'HELP {n} PLACES', 'gold': 'A GOLD MEDAL' if n == '1' else f'{n} GOLD MEDALS',
            'photos': f'{n} ALBUM PHOTOS', 'friends': f'MEET {n} BALLOONS'}.get(kind, need)
PATTERN_KEYS = [p[0] for p in PATTERNS]
_STRIPE_COLS = ('hot_pink', 'lavender', 'rose', 'violet', 'pink', 'lilac')
_RAINBOW = (('red', 'crimson', 'salmon'), ('orange', 'vermilion', 'amber'), ('butter', 'gold', 'cream'),
            ('leaf', 'green', 'pale_leaf'), ('sky', 'blue', 'light_sky'), ('violet', 'purple', 'lilac'))


def pattern_rgba(rgba, pattern):
    """A copy of a balloon sprite's RGBA with its envelope recoloured to `pattern`."""
    spec = dict((p[0], p[3]) for p in PATTERNS).get(pattern)
    if spec is None:
        return rgba
    out = rgba.copy()
    rgb = out[:, :, :3]
    src = [np.array(col(c), np.uint8) for c in _STRIPE_COLS]
    w = out.shape[1]
    for i, sc in enumerate(src):
        m = np.all(rgb == sc, axis=2) & (out[:, :, 3] > 0)
        if not m.any():
            continue
        if spec == 'rainbow':
            tone = (0, 0, 1, 1, 2, 2)[i]                # base / shade / highlight
            ys, xs = np.nonzero(m)
            for y, x in zip(ys, xs):
                band = _RAINBOW[min(5, x * 6 // max(1, w))]
                rgb[y, x] = col((band[0], band[1], band[2])[tone])
        else:
            rgb[m] = col(spec[i])
    return out


def balloon_rgba(size, pattern='classic', flame_on=False):
    """The balloon sprite (art px, RGBA) in a wardrobe pattern: for previews (journal, postcards)."""
    return pattern_rgba(_balloon_sprite(size, flame_on).rgba, pattern)


def _q(v):
    """0..1 -> 0..4 (sprite cache steps)."""
    return int(round(min(1.0, max(0.0, v)) * 4))


def paste_balloon(c, x, y, f, size, flame=False, tint=None, pattern='classic', cargo=None, wet=0.0, snow=0.0):
    """Draw the balloon straight onto an art-px Canvas (basket bottom-centre at integer (x, y)), with
    its pattern, tint and cargo: for frames composed as one picture (the cloud sweep between areas)."""
    flame_on = bool(flame) and (int(f) // 3) % 2 == 0
    spr = _balloon_rgba(size, flame_on, 1, tint, pattern, _q(wet), _q(snow))
    dy = int(round(math.sin(f * 2 * math.pi / 72.0) * (1 if size in ('xs', 's') else 1.4)))
    _paste_rgba(c.a, spr, int(x) - spr.shape[1] // 2, int(y) + dy - spr.shape[0])
    if cargo:
        cg = _cargo_rgba(cargo, size, 1, tint)
        _paste_rgba(c.a, cg, int(x) - cg.shape[1] // 2, int(y) + dy)


def _cargo_rgba(key, size, k, tint=None):
    """The basket's cargo (rope + item) scaled by k and lit like the balloon."""
    ck = ('cargo', key, size, k, tint)
    if ck not in _SPR:
        from .items import cargo_rgba
        rgba = cargo_rgba(key, size).copy()
        if tint is not None:
            c, amount = tint
            rgb = rgba[:, :, :3].astype(np.float32)
            lit = rgb * (np.array(col(c), np.float32) / 255.0)
            rgba[:, :, :3] = np.clip(rgb + (lit - rgb) * amount, 0, 255).astype(np.uint8)
        _SPR[ck] = _up(rgba, k)
    return _SPR[ck]


def _balloon_rgba(size, flame_on, k, tint=None, pattern='classic', wet=0, snow=0):
    key = (size, flame_on, k, tint, pattern, wet, snow)
    if key not in _SPR:
        if len(_SPR) > 256:                         # smoothly changing tints: keep the cache bounded
            _SPR.clear()
        rgba = pattern_rgba(_balloon_sprite(size, flame_on).rgba, pattern).copy()
        if wet or snow:
            _weather_rgba(rgba, size, wet / 4.0, snow / 4.0)
        if tint is not None:
            c, amount = tint
            rgb = rgba[:, :, :3].astype(np.float32)
            lit = rgb * (np.array(col(c), np.float32) / 255.0)
            rgba[:, :, :3] = np.clip(rgb + (lit - rgb) * amount, 0, 255).astype(np.uint8)
        if flame_on and size in ('m', 'l'):
            _R, _He, rope, _bw, bh = _B_SIZES[size]
            h, w = rgba.shape[:2]
            rgba[h - bh - rope, w // 2] = (*col('butter'), 255)     # the flame stays bright
        _SPR[key] = _up(rgba, k)
    return _SPR[key]


def _weather_rgba(rgba, size, wet, snow):
    """Soak (darker, bluer envelope) and snow-cap (white on the envelope's top) a balloon sprite in place."""
    _R, He, _rope, _bw, _bh = _B_SIZES[size]
    h, w = rgba.shape[:2]
    env = np.zeros((h, w), bool)
    env[:He + 1] = rgba[:He + 1, :, 3] > 0                 # the envelope rows (the basket stays as it is)
    if wet > 0:
        rgb = rgba[:, :, :3].astype(np.float32)
        wetc = np.array(col('#3e4f86'), np.float32)
        k = 0.42 * wet
        rgb[env] = rgb[env] * (1 - k) + (rgb[env] * wetc / 255.0) * k * 1.6
        rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    if snow > 0:
        n = max(1, int(round(snow * (2 + He // 6))))
        for x in range(w):
            ys = np.nonzero(env[:, x])[0]
            if len(ys) == 0:
                continue
            top = ys[0]
            lump = n + (1 if (x * 7 + 3) % 5 == 0 and snow > 0.5 else 0) - (1 if (x * 5) % 7 == 0 else 0)
            for y in range(top, min(top + max(0, lump), h)):
                if env[y, x]:
                    rgba[y, x, :3] = col('white') if y == top or (x + y) % 3 else col('cloud')


def _paste_rgba(out, spr, x, y):
    h, w = spr.shape[:2]
    H_, W_ = out.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[:, :, 3] > 127
    out[y0:y1, x0:x1][m] = s[:, :, :3][m]


def _darken(out, x, y, w, h, num=3, den=5):
    H_, W_ = out.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = (out[y0:y1, x0:x1].astype(np.uint16) * num // den).astype(np.uint8)


def _overlay_keyed(out, layer, k):
    """Paste the non-KEY pixels of a 320x180 layer (only the rows that have any) scaled by k."""
    m = np.any(layer != _KEY, axis=2)[:VIEW_H, :VIEW_W]
    rows = np.nonzero(m.any(axis=1))[0]
    if len(rows) == 0:
        return
    # paste row bands (contiguous runs) to keep the work small for a corner counter + a hint strip
    starts = [rows[0]]
    ends = []
    for a, b in zip(rows[:-1], rows[1:]):
        if b != a + 1:
            ends.append(a + 1)
            starts.append(b)
    ends.append(rows[-1] + 1)
    # out seen as (art row, k, art col, k, rgb): each art pixel's k x k block is written straight from
    # the layer by broadcasting (no upscaled temporaries: a full-screen iris costs ~3 ms, not ~30)
    o5 = out.reshape(VIEW_H, k, VIEW_W, k, 3)
    assert np.shares_memory(o5, out)              # (splitting axes of a strided view never copies)
    for r0, r1 in zip(starts, ends):
        cols = np.nonzero(m[r0:r1].any(axis=0))[0]
        c0, c1 = cols[0], cols[-1] + 1
        np.copyto(o5[r0:r1, :, c0:c1], layer[r0:r1, None, c0:c1, None, :],
                  where=m[r0:r1, None, c0:c1, None, None])


def _overlay_world(out, layer, k, sx, sy):
    """Paste the non-KEY pixels of a world-anchored layer, scaled by k and shifted like the world
    (window pixel = art pixel * k - (sx, sy)): only its bounding box is upscaled."""
    m = np.any(layer != _KEY, axis=2)
    rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
    if len(rows) == 0:
        return
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    patch = _up(layer[r0:r1, c0:c1], k)
    pm = _up(m[r0:r1, c0:c1], k)
    y, x = r0 * k - sy, c0 * k - sx
    H_, W_ = out.shape[:2]
    y0, x0, y1, x1 = max(0, y), max(0, x), min(H_, y + patch.shape[0]), min(W_, x + patch.shape[1])
    if y1 <= y0 or x1 <= x0:
        return
    np.copyto(out[y0:y1, x0:x1], patch[y0 - y:y1 - y, x0 - x:x1 - x],
              where=pm[y0 - y:y1 - y, x0 - x:x1 - x, None])


_DIM_ADD = np.array((10, 4, 14), np.uint8)


def _dim(a):
    """The pause dim of RGB pixels (any shape ending in 3): 3/8 brightness, plum-tinted, integer ops.
    It is applied to each layer at art px before it is scaled (the same result as dimming the window
    image, at a fraction of the cost)."""
    d = a >> 2
    d += d >> 1
    d += _DIM_ADD
    return d


def _dim_rgba(spr):
    out = spr.copy()
    out[..., :3] = _dim(spr[..., :3])
    return out


def _paste_ride(out, a, m, x, y, k):
    """Paste the masked pixels of a ride layer (Frame.ride), scaled by k, at window pixel (x, y)."""
    rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    patch, pm = _up(a[r0:r1, c0:c1], k), _up(m[r0:r1, c0:c1], k)
    y, x = y + r0 * k, x + c0 * k
    H_, W_ = out.shape[:2]
    y0, x0, y1, x1 = max(0, y), max(0, x), min(H_, y + patch.shape[0]), min(W_, x + patch.shape[1])
    if y1 > y0 and x1 > x0:
        np.copyto(out[y0:y1, x0:x1], patch[y0 - y:y1 - y, x0 - x:x1 - x], where=pm[y0 - y:y1 - y, x0 - x:x1 - x, None])


def _cam_px(o, c, focus, k):
    """Window pixel of the camera on one axis (see the module doc)."""
    if focus is None:
        return int(round(c * k))
    return int(round(focus * k)) - int(round((focus - c) * k))


_BUF = {}                       # k -> the window buffer compose_window reuses frame after frame


def compose(fr, k):
    """The frame as a (VIEW_H*k, VIEW_W*k, 3) uint8 image (a new array)."""
    big, sx, sy = _compose(fr, k, np.empty((H * k, W * k), np.uint32))
    return np.ascontiguousarray(big.view(np.uint8).reshape(H * k, W * k, 4)[sy:sy + VIEW_H * k,
                                                                             sx:sx + VIEW_W * k, :3])


def compose_window(fr, k):
    """The frame drawn into a reused (H*k, W*k) buffer of RGBX pixels (uint32); returns (buffer, sx, sy):
    the window shows its (VIEW_W*k x VIEW_H*k) part from (sx, sy). The front end hands the buffer to the
    display as is (no copy): the buffer is overwritten by the next call."""
    if k not in _BUF:
        _BUF.clear()
        _BUF[k] = np.zeros((H * k, W * k), np.uint32)
    return _compose(fr, k, _BUF[k])


def _compose(fr, k, big):
    (ix, iy), (cx, cy) = fr.origin, fr.cam
    fx, fy = fr.focus if fr.focus is not None else (None, None)
    sx = min(k - 1, max(0, _cam_px(ix, cx, fx, k) - ix * k))
    sy = min(k - 1, max(0, _cam_px(iy, cy, fy, k) - iy * k))
    x0, y0 = ix * k + sx, iy * k + sy                 # window origin in world*k pixels
    dim = _dim if fr.dim else None                    # the pause dim: applied per art px (cheap), not
    wa = _dim(fr.world.a) if dim else fr.world.a      # over the whole window image
    # the upscale works on whole 4-byte pixels (RGBX as one uint32): one broadcast write, ~6x faster than
    # moving 3 separate bytes; `out` is the RGB view of the window's part of it
    a4 = np.zeros((H, W, 4), np.uint8)
    a4[..., :3] = wa
    big.reshape(H, k, W, k)[:] = a4.view(np.uint32)[:, None, :, :1].reshape(H, 1, W, 1)
    out = big.view(np.uint8).reshape(H * k, W * k, 4)[sy:sy + VIEW_H * k, sx:sx + VIEW_W * k, :3]
    for s in fr.sprites:
        if s[0] == 'shadow':
            _, wx, wy, rx = s
            x, y = int(round(wx * k)) - x0, int(round(wy * k)) - y0
            _darken(out, x - rx * k, y, (2 * rx + 1) * k, k)
            _darken(out, x - (rx - 1) * k, y + k, (2 * rx - 1) * k, k)
        elif s[0] == 'ride':
            _, a, m, rx_, ry_, dx, dy = s
            _paste_ride(out, _dim(a) if dim else a, m, rx_ * k - sx + int(round(dx * k)),
                        ry_ * k - sy + int(round(dy * k)), k)
        else:
            _, wx, wy, f, size, flame, bob, tint, pattern, cargo, wet, snow = s
            fi = int(f)
            flame_on = bool(flame) and (fi // 3) % 2 == 0
            spr = _balloon_rgba(size, flame_on, k, tint, pattern, wet, snow)
            dy = bob_dy(f, size) if bob else 0.0
            h, w = spr.shape[0] // k, spr.shape[1] // k
            x = int(round(wx * k)) - x0 - (w // 2) * k
            y = int(round((wy + dy) * k)) - y0 - h * k
            _paste_rgba(out, _dim_rgba(spr) if dim else spr, x, y)
            if cargo:
                cg = _cargo_rgba(cargo, size, k, tint)
                _paste_rgba(out, _dim_rgba(cg) if dim else cg,
                            int(round(wx * k)) - x0 - (cg.shape[1] // k // 2) * k, y + h * k)
    if fr.front is not None:
        la = fr.front.a
        if dim:                                       # (dimmed pixels never come out as KEY: max 108)
            la = la.copy()
            m = np.any(la != _KEY, axis=2)
            la[m] = _dim(la[m])
        _overlay_world(out, la, k, sx, sy)
    _overlay_keyed(out, fr.hud.a, k)
    if fr.iris is not None:
        t, icx, icy = fr.iris
        if t > 0:
            layer = Canvas(color=KEY)
            draw_iris(layer, t, icx, icy)
            _overlay_keyed(out, layer.a, k)
    if fr.flash > 0:
        a = min(1.0, fr.flash)
        out[:] = (out.astype(np.float32) * (1 - a) + 255 * a).astype(np.uint8)
    return big, sx, sy
