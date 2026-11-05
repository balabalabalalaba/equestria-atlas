"""Procedural terrain sprites for the world map: trees, mountains, mesas, spires, ice floes.

Every generator returns an RGBA uint8 array (h, w, 4) (alpha 0/255), deterministic from its seed.
Light comes from the top-left. Anchor convention: bottom-centre of the sprite = ground point.
"""
import math
from functools import lru_cache

import numpy as np

from .core import col


def C(c):
    return np.array(col(c), np.uint8)


def _rng(seed):
    return np.random.default_rng(int(seed) & 0xffffffff)


def _new(h, w):
    return np.zeros((h, w, 4), np.uint8)


def _put(a, y, x, c):
    if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        a[y, x, :3] = col(c)
        a[y, x, 3] = 255


def _outline(a, color, sides='all'):
    """Add a 1px outline in `color` on transparent pixels next to opaque ones (inside the box)."""
    m = a[:, :, 3] > 0
    g = np.zeros_like(m)
    g[1:, :] |= m[:-1, :]
    g[:-1, :] |= m[1:, :]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    ring = g & ~m
    a[ring, :3] = col(color)
    a[ring, 3] = 255
    return a


def _pad(a, p=1):
    return np.pad(a, ((p, p), (p, p), (0, 0)))


# ------------------------------------------------------------------ trees
TREE_RAMPS = {
    # outline, dark, mid, light, highlight, trunk
    'round': ('#1d4a33', '#2c6b3d', '#3f8d45', '#5fae4f', '#8fd062', '#6b4a36'),
    'round2': ('#1d4a33', '#2f7444', '#469a4c', '#6cbb57', '#9ad86a', '#6b4a36'),
    'dark': ('#162a2a', '#1f3d36', '#2a5544', '#3b6e4e', '#5a8a5a', '#3e2e34'),
    'dark2': ('#132a2a', '#1d3a3a', '#2a5250', '#3a6a62', '#4f8070', '#3e2e34'),
    'autumn': ('#5a2a1e', '#a0432a', '#d0663a', '#ef9a48', '#ffcf6a', '#5a3a2a'),
    'autumn2': ('#5a3a1e', '#a86a22', '#d99a2e', '#f2c64a', '#fff08a', '#5a3a2a'),
    'autumn3': ('#4a1e2a', '#8a2e3a', '#b8483e', '#de6e4a', '#f7a066', '#5a3a2a'),
    'violet': ('#1e1838', '#2e2858', '#463d7e', '#5f58a2', '#8c86c8', '#2a2238'),
    'olive': ('#232a1c', '#34402a', '#4a5838', '#637048', '#82905c', '#4a3a2a'),
    'jungle': ('#0f3a26', '#185a34', '#237a3e', '#36994a', '#5cbb58', '#5a3a2a'),
    'orchard': ('#244a26', '#2f6a32', '#43883e', '#5ea64a', '#86c45e', '#7a5236'),
    'cherry': ('#244a26', '#2f6a32', '#43883e', '#5ea64a', '#86c45e', '#7a5236'),
    'mushroom': ('#3a2a2a', '#7a5a52', '#a07a68', '#c09a80', '#e2c4a2', '#d8ccb4'),
}


def _canopy(r, ramp, seed, squash=1.0, lumpy=0.0):
    """Round 3/4 canopy with top-left light and a dark outline."""
    o, dk, md, lt, hi, _ = ramp
    rg = _rng(seed)
    d = 2 * r + 1
    h = int(round(d * squash))
    a = _new(h + 2, d + 2)
    cx, cy = r + 1, (h - 1) / 2 + 1
    for y in range(h + 2):
        for x in range(d + 2):
            dx = (x - cx) / (r + 0.35)
            dy = (y - cy) / ((h / 2) + 0.35)
            rr = dx * dx + dy * dy
            if lumpy:
                ang = math.atan2(dy, dx)
                rr *= 1 + lumpy * math.sin(ang * 5 + seed)
            if rr <= 1.0:
                # lighting: normal from sphere
                nz = math.sqrt(max(0.0, 1 - min(1.0, rr)))
                lv = (-dx * 0.6 - dy * 0.8) * 0.75 + nz * 0.45 + (rg.random() - 0.5) * 0.25
                if lv > 0.62:
                    c = hi
                elif lv > 0.3:
                    c = lt
                elif lv > -0.05:
                    c = md
                else:
                    c = dk
                _put(a, y, x, c)
    _outline(a, o)
    return a


def tree_sprite(kind, variant=0):
    return _tree_sprite(kind, variant).copy()


@lru_cache(maxsize=None)
def _tree_sprite(kind, variant):
    seed = hash_str(kind) * 31 + variant * 7 + 3
    rg = _rng(seed)
    if kind in ('round', 'round2', 'olive', 'jungle', 'autumn', 'autumn2', 'autumn3', 'violet_round',
                'orchard', 'cherry'):
        ramp = TREE_RAMPS[kind]
        r = {'jungle': 3 + (variant % 2), 'orchard': 2, 'cherry': 2}.get(kind, 2 + (variant % 3 == 0))
        can = _canopy(r, ramp, seed, squash=0.9 if kind != 'jungle' else 0.85)
        ch, cw = can.shape[:2]
        trunk = 2 if kind not in ('orchard', 'cherry') else 1
        a = _new(ch + trunk, cw)
        a[ch - 1:ch - 1 + trunk + 1, cw // 2, :3] = C(ramp[5])
        a[ch - 1:ch - 1 + trunk + 1, cw // 2, 3] = 255
        m = can[:, :, 3] > 0
        a[:ch][m] = can[m]
        if kind in ('orchard', 'cherry'):
            fr = ['#e8323c', '#ff5a5a'] if kind == 'orchard' else (['#d0203a', '#f2d04a'][variant % 2],) * 2
            for (yy, xx) in ((2, 2), (3, 4), (4, 2)):
                if variant % 2 or (yy, xx) != (4, 2):
                    _put(a, yy, xx, fr[(yy + xx) % 2])
        return a
    if kind == 'dark':
        ramp = TREE_RAMPS['dark2' if variant % 3 == 2 else 'dark']
        r = 2 + (variant % 2)
        can = _canopy(r, ramp, seed, squash=0.8, lumpy=0.35)
        ch, cw = can.shape[:2]
        a = _new(ch + 3, cw + 2)
        tx = cw // 2 + 1
        # twisted trunk
        for i, dx in enumerate((0, -1 if variant % 2 else 1, 0)):
            _put(a, ch - 1 + i, tx + dx, ramp[5])
        if variant % 4 == 1:     # gnarled bare branch poking out
            _put(a, ch - 2, cw + 1, ramp[5])
            _put(a, ch - 3, cw, ramp[5])
        m = can[:, :, 3] > 0
        a[:ch, 1:cw + 1][m] = can[m]
        if variant % 5 == 3:     # glowing blue poison-joke flower at the foot
            _put(a, ch + 2, tx - 2, '#5ba3e0')
        return a
    if kind in ('pine', 'pine_snow'):
        h = 7 + (variant % 3)
        w = 5 + 2 * (variant % 2)
        a = _new(h + 1, w)
        dk, md, lt, o = '#1f4a3a', '#2d6a4a', '#4a8a5a', '#163428'
        cxp = w // 2
        for y in range(h):
            t = y / max(1, h - 1)
            # layered tiers
            tier = (y % 3) / 2.0
            hw = int(round((0.5 + t * (w / 2 - 0.5)) * (0.75 + 0.25 * tier)))
            for x in range(cxp - hw, cxp + hw + 1):
                c = lt if x < cxp else (md if x == cxp else dk)
                if kind == 'pine_snow' and (y % 3 == 0) and x <= cxp:
                    c = '#eaf4ff'
                elif kind == 'pine_snow' and (y % 3 == 0):
                    c = '#b8cce8'
                _put(a, y, x, c)
        _put(a, h, cxp, '#5a3a2a')
        a = _pad(a)
        _outline(a, o)
        return a
    if kind == 'palm':
        a = _new(10, 9)
        for y in range(4, 10):
            _put(a, y, 4 + (1 if y < 6 else 0), '#8a5a3a' if y % 2 else '#a8784a')
        fr = '#2a8a3a'
        frl = '#5cc04a'
        for (y, x) in ((3, 1), (2, 2), (2, 3), (3, 4), (3, 5), (2, 6), (2, 7), (3, 8), (1, 4), (0, 5),
                       (4, 2), (4, 7), (1, 6), (1, 3)):
            _put(a, y, x, frl if x < 5 else fr)
        _put(a, 3, 5, '#7a4a2a')
        a = _pad(a)
        _outline(a, '#123a22')
        return a
    if kind == 'bamboo':
        h = 10 + variant % 3
        a = _new(h, 6)
        for k, x in enumerate((1, 3, 4)):
            hh = h - (k * 2 + variant) % 4
            for y in range(h - hh, h):
                c = '#9ad45a' if (y % 4) else '#5a8a3a'
                _put(a, y, x, c if x != 4 else '#6aa84a')
            _put(a, h - hh, x - 1, '#bfe87a')
            _put(a, h - hh + 1, x + 1, '#7ab84a')
        return a
    if kind == 'mushroom':
        a = _new(9, 9)
        o, dk, md, lt, hi, stalk = TREE_RAMPS['mushroom']
        caps = [('#b07aa0', '#d8a2c0', '#f2cce0'), ('#8aa06a', '#b0c47e', '#d6e4a2'),
                ('#c0806a', '#e0a488', '#f6cdb0')][variant % 3]
        for y in range(5, 9):
            _put(a, y, 4, stalk)
            _put(a, y, 5, '#a89c88')
        for y in range(0, 5):
            hw = [2, 3, 4, 4, 3][y]
            for x in range(4 - hw, 4 + hw + 1):
                c = caps[2] if (x < 4 and y < 2) else caps[1] if x < 6 else caps[0]
                if y == 4:
                    c = caps[0]
                _put(a, y, x, c)
        _put(a, 2, 3, '#fff6e0')
        _put(a, 1, 5, '#fff6e0')
        a = _pad(a)
        _outline(a, '#2e2230')
        return a
    if kind == 'violet':
        h = 8 + variant % 3
        a = _new(h, 5)
        o, dk, md, lt, hi, tr = TREE_RAMPS['violet']
        for y in range(h - 1):
            hw = min(2, (y + 1) // 2)
            for x in range(2 - hw, 2 + hw + 1):
                _put(a, y, x, hi if (x < 2 and y < 3) else lt if x < 2 else md if x == 2 else dk)
        _put(a, h - 1, 2, tr)
        a = _pad(a)
        _outline(a, o)
        return a
    if kind == 'dead':
        a = _new(9, 7)
        c = '#5a4a48'
        for y in range(3, 9):
            _put(a, y, 3, c)
        for (y, x) in ((2, 2), (1, 1), (3, 4), (2, 5), (1, 5), (4, 2), (0, 1), (0, 6)):
            _put(a, y, x, c if x != 5 else '#7a6a66')
        return a
    raise KeyError(kind)


def tree_shadow(w):
    """Ground shadow offsets (dx, dy) list for a tree of width w (bottom-right ellipse)."""
    out = []
    r = max(2, w // 2)
    for dx in range(-r + 1, r + 2):
        out.append((dx + 1, 0))
        if abs(dx) < r - 1:
            out.append((dx + 1, 1))
    return out


def hash_str(s):
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xffffffff
    return h


# ------------------------------------------------------------------ mountains
MTN_SCHEMES = {
    # hi, lit, mid, shadow, deep shadow, outline, snow lit, snow shadow, snowline(0..1 of height)
    'snow': ('#d8e2f6', '#a9b8de', '#8494c6', '#6672aa', '#4c5590', '#343a6a', '#ffffff', '#c6d6f2', 0.62),
    'yak': ('#d8e2f6', '#a9b8de', '#8494c6', '#6672aa', '#4c5590', '#343a6a', '#ffffff', '#c6d6f2', 0.55),
    'everhoof': ('#e2eafa', '#b2c0e4', '#8f9dce', '#6e78b2', '#525a96', '#383e70', '#ffffff', '#cddcf6', 0.72),
    'yaket': ('#cfc6dc', '#aea4c4', '#8f86a8', '#706a8c', '#554f70', '#3b3650', '#ffffff', '#cfd6ee', 0.45),
    'crystalm': ('#e2f6ff', '#b4e2fa', '#8cc4ee', '#6a9ad8', '#5276bc', '#3a4f90', '#ffffff', '#d4ecff', 0.4),
    'unicorn': ('#8ed8cc', '#62b8b0', '#46989a', '#357a84', '#285e6c', '#1c4250', '#f4fbff', '#bfe0ec', 0.3),
    'mauve': ('#f0bcc8', '#d89aae', '#bc7e96', '#9a6280', '#7a4a68', '#5a3450', '#fff4f6', '#e8cad6', 0.2),
    'rim': ('#d6a0b4', '#b98298', '#9c687e', '#7e5068', '#623c54', '#47293e', '#fff0f4', '#e2c4d0', 0.12),
    'crag': ('#c2c4b2', '#a3a594', '#86887a', '#6a6c62', '#52544c', '#3a3c36', '#f6f8f2', '#cdd2cc', 0.22),
    'greycrag': ('#c8c4d2', '#a8a4b8', '#8a8699', '#6c687c', '#524e62', '#3a3648', '#ffffff', '#d2d4e6', 0.25),
    'cloudpk': ('#6a6270', '#544c5a', '#423b48', '#332d38', '#262129', '#1a161c', '#8a8290', '#6a6270', 0.0),
    'storm': ('#b07a62', '#94624e', '#7a4c3c', '#603a2e', '#4a2c24', '#33201a', '#c89a80', '#a07a66', 0.0),
    'canterhorn': ('#cfc2e6', '#ad9ed0', '#8f80b6', '#71649a', '#584e80', '#3e365e', '#f4f0ff', '#d6cff0', 0.08),
    'foal': ('#9ccaf0', '#78aedc', '#5a92c6', '#4676aa', '#365c8c', '#26446a', '#f4fbff', '#c8e2f6', 0.22),
    'smokey': ('#a2c08e', '#84a472', '#6a885c', '#54704a', '#40583a', '#2c3e2a', '#f2f6ee', '#c6d6c2', 0.12),
}


def mountain_sprite(w, h, scheme, seed):
    return _mountain(int(w), int(h), scheme, int(seed)).copy()


def _noise1d(n, rg, octaves=((8, 1.0), (4, 0.5), (2, 0.25))):
    """Smooth 1D fractal noise in about [-1, 1] of length n."""
    out = np.zeros(n)
    for step, amp in octaves:
        k = n // step + 3
        v = rg.uniform(-1, 1, k)
        xs = np.arange(n) / step
        i = xs.astype(int)
        t = xs - i
        t = t * t * (3 - 2 * t)
        out += amp * (v[i] * (1 - t) + v[i + 1] * t)
    return out / sum(a for _, a in octaves)


@lru_cache(maxsize=None)
def _mountain(w, h, scheme, seed):
    hi, lit, mid, sh, dsh, out, snl, sns, snowline = MTN_SCHEMES[scheme]
    rg = _rng(seed * 977 + 13)
    big = scheme in ('everhoof', 'canterhorn')
    rugged = {'rim': 1.6, 'cloudpk': 1.5, 'storm': 1.5, 'crag': 1.3, 'greycrag': 1.4}.get(scheme, 1.0)
    px = w / 2 + rg.uniform(-0.12, 0.12) * w * (0.3 if big else 1)
    xs = np.arange(w) + 0.5
    # --- silhouette: power profile + fractal noise + shoulders (secondary summits)
    d = np.abs(xs - px) / np.where(xs < px, max(1.0, px), max(1.0, w - px))
    p = rg.uniform(0.85, 1.1) if not big else (1.0 if scheme == 'everhoof' else 1.3)
    top = h * np.clip(d, 0, 1) ** p
    amp = (0.07 if big else 0.11) * h * rugged
    top += _noise1d(w, rg, ((max(2, w // 5), 1.0), (max(2, w // 10), 0.5), (2, 0.2))) * amp * np.clip(d * 2, 0, 1)
    if not big:
        for _ in range(rg.integers(1, 3)):
            sx = rg.uniform(0.12, 0.88) * w
            if abs(sx - px) < w * 0.18:
                continue
            bump = rg.uniform(0.12, 0.3) * h
            top -= bump * np.exp(-((xs - sx) / (w * 0.09)) ** 2) * np.clip(d * 1.6, 0, 1)
    top = np.clip(top, 0, h)
    top[int(px)] = 0
    # --- facets: spurs radiating from the summit
    yy, xx = np.mgrid[0:h + 1, 0:w]
    inside = yy >= top[None, :]
    phi = np.degrees(np.arctan2(xx + 0.5 - px, yy + 1.0))       # 0 = straight down, <0 = left
    nl = 3 if w < 30 else 4 if w < 70 else 6
    spl = np.sort(rg.uniform(-78, -12, nl))
    spr = np.sort(rg.uniform(10, 78, nl))
    main = rg.uniform(-4, 6)
    bounds = np.concatenate([[-180], spl, [main], spr, [180]])
    fidx = np.searchsorted(bounds, phi) - 1
    nsl = nl + 1
    lit_side = fidx < nsl
    alt = (fidx % 2 == 0)
    rgb = np.zeros((h + 1, w, 3), np.uint8)
    hy = yy / max(1, h)
    # wobble the facet boundaries a little with height for a hand-drawn feel
    fr = rg.uniform(-0.35, 0.35, len(bounds) + 1)[np.clip(fidx, 0, len(bounds))]
    v = np.where(lit_side, np.where(alt, 3.55, 2.9), np.where(alt, 1.25, 0.55)) + fr
    v = v - np.clip(hy - 0.35, 0, 1) * 1.6 + np.where(lit_side & (hy < 0.25) & (phi > -40), 0.7, 0)
    tex = rg.uniform(-0.45, 0.45, v.shape) * np.clip(hy * 1.5, 0.2, 1)
    by = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0
    thr = by[yy % 4, xx % 4]
    vv = v + tex * 0.5
    lvl = np.floor(vv).astype(int) + ((vv - np.floor(vv)) > (0.5 + (thr - 0.5) * 0.5)).astype(int)
    lvl = np.where(lit_side, np.clip(lvl, 2, 4), np.clip(lvl, 0, 2))
    pal = np.array([C(dsh), C(sh), C(mid), C(lit), C(hi)])
    rgb[:] = pal[np.clip(lvl, 0, 4)]
    # creases: facet boundaries on the lit side drawn in mid
    edge = np.zeros_like(inside)
    edge[:, 1:] = fidx[:, 1:] != fidx[:, :-1]
    crease = edge & inside & (phi < main) & (hy > 0.12)
    rgb[crease] = C(mid)
    rgb[edge & inside & (phi > main) & (hy > 0.12)] = C(dsh)
    # --- snow: smooth jagged snowline; streaks run down along the spurs
    if scheme == 'canterhorn':
        for ly in range(int(h * 0.35), h - 4, 9):
            band = inside & (yy == ly) & lit_side
            rgb[band] = C(hi)
            rgb[inside & (yy == ly + 1) & lit_side] = C(mid)
            rgb[inside & (yy == ly + 1) & ~lit_side] = C(dsh)
    if snowline > 0:
        rr = np.hypot(xx + 0.5 - px, (yy + 0.5) * 1.0)
        nphi = _noise1d(361, rg, ((24, 1.0), (8, 0.5), (3, 0.25)))
        R0 = snowline * h * (1.0 + 0.22 * nphi[np.clip((phi + 180).astype(int), 0, 360)])
        tongue = np.zeros_like(R0)
        for b in list(spl) + list(spr):
            ln = h * rg.uniform(0.12, 0.3)
            tongue = np.maximum(tongue, ln * np.clip(1 - np.abs(phi - b) / 5.5, 0, 1))
        snow = inside & (rr < R0 + tongue)
        snow_rgb = np.where(lit_side[..., None], C(snl), C(sns))
        rgb = np.where(snow[..., None], snow_rgb, rgb)
        rgb[snow & lit_side & (lvl >= 4)] = C('#ffffff')
        rgb[snow & ~lit_side & (lvl == 0)] = C(sh)
    a = np.zeros((h + 1, w, 4), np.uint8)
    a[..., :3] = rgb
    a[..., 3] = np.where(inside, 255, 0)
    # dithered fade of the base rows into the ground
    base = (yy >= h - 1) & ((xx + yy) % 2 == 0)
    a[base, 3] = 0
    a = _pad(a)
    m = a[:, :, 3] > 0
    g = np.zeros_like(m)
    g[1:, :] |= m[:-1, :]
    g[:-1, :] |= m[1:, :]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    ring = g & ~m
    ring[-4:, :] = False
    a[ring, :3] = C(out)
    a[ring, 3] = 255
    return a


# ------------------------------------------------------------------ mesas & spires
MESA_SCHEMES = {
    # top lit, top, side bands (2-3), side shadow, outline
    'red': ('#f2b07a', '#e09060', ('#c8603e', '#d8784a', '#b04e36'), '#8a3a2e', '#5a2420'),
    'orange': ('#f6c27a', '#eaa560', ('#d47a42', '#e8964e', '#c0663a'), '#9a4a30', '#62301e'),
    'ochre': ('#ead09a', '#d8b47a', ('#c08a58', '#d09c66', '#ac7a4c'), '#8a5a3a', '#5a3a26'),
    'ridge': ('#eee6d4', '#d8cfbc', ('#b8ad98', '#c8bea8', '#a89d88'), '#857a6a', '#5a5248'),
}


def mesa_sprite(w, h, td, scheme, seed):
    return _mesa(int(w), int(h), int(td), scheme, int(seed)).copy()


@lru_cache(maxsize=None)
def _mesa(w, h, td, scheme, seed):
    tl, tp, bands, shd, out = MESA_SCHEMES[scheme]
    rg = _rng(seed + 5)
    a = _new(h + td, w)
    # top face (ellipse-ish blob), then the cliff sides below it
    cx = w / 2
    for y in range(td):
        t = (y + 0.5) / td
        hw = (w / 2 - 1) * math.sqrt(max(0.0, 1 - (2 * t - 1) ** 2 * 0.55))
        for x in range(int(cx - hw), int(cx + hw) + 1):
            _put(a, y, x, tl if (x < cx - hw * 0.3 and y < td * 0.6) else tp)
    # sides: slightly flaring downwards, striped
    for y in range(td // 2, h + td):
        k = (y - td // 2) / max(1, h + td // 2)
        hw = (w / 2 - 1) * (0.88 + 0.12 * k) + rg.uniform(-0.4, 0.4)
        band = bands[(y // 2 + (1 if y > (h + td) * 0.6 else 0)) % len(bands)]
        for x in range(int(cx - hw), int(cx + hw) + 1):
            if y < td and a[y, x, 3]:
                continue
            c = band if x < cx + hw * 0.35 else shd
            _put(a, y, x, c)
    a = _pad(a)
    _outline(a, out)
    a[-1, :, 3] = 0
    return a


def spire_sprite(kind, w, h, seed):
    return _spire(kind, int(w), int(h), int(seed)).copy()


@lru_cache(maxsize=None)
def _spire(kind, w, h, seed):
    rg = _rng(seed + 77)
    a = _new(h, w)
    cx = (w - 1) / 2
    if kind == 'needle':
        cols = ('#e6e8f0', '#b8bccc', '#8a8ea2', '#62667c', '#3e4256')
    elif kind == 'karst':
        cols = ('#d2dcd2', '#a8b8ac', '#84968a', '#62746a', '#3e4e46')
    elif kind == 'cone':
        cols = ('#f6dca0', '#e2be7c', '#c89c5e', '#a07a46', '#6a4e30')
    elif kind == 'hoodoo':
        cols = ('#f0d2a4', '#d8b080', '#bc9064', '#94704c', '#62482e')
    else:  # sea stack
        cols = ('#ffffff', '#e6e2da', '#c4beb4', '#9a948c', '#6a6660')
    for y in range(h):
        t = y / max(1, h - 1)
        if kind == 'needle':
            hw = 0.5 + t * (w / 2 - 0.5)
        elif kind == 'karst':
            hw = (w / 2 - 0.5) * (0.75 + 0.25 * min(1, t * 3)) if y > 1 else w / 2 - 2
        elif kind == 'cone':
            hw = 0.5 + t * (w / 2 - 0.5)
        elif kind == 'hoodoo':
            hw = (w / 2 - 0.5) if y < 2 else (w / 2 - 1.5) * (0.6 + 0.4 * t)
        else:
            hw = (w / 2 - 0.5) * (0.6 + 0.4 * math.sin(min(1, t * 1.4) * math.pi / 2))
        for x in range(w):
            if abs(x - cx) <= hw:
                rel = (x - cx) / max(0.5, hw)
                c = cols[1] if rel < -0.2 else cols[2] if rel < 0.35 else cols[3]
                if rel < -0.55 and t < 0.8:
                    c = cols[0]
                if kind == 'karst' and y < max(3, h // 6):
                    c = '#5aa858' if rel < 0.3 else '#3e8048'
                if kind == 'hoodoo' and y < 2:
                    c = '#9a6a4a'
                if kind in ('needle', 'karst') and (y + int(x * 1.7)) % 7 == 0 and t > 0.2:
                    c = cols[3] if rel < 0.35 else cols[4]
                _put(a, y, x, c)
    a = _pad(a)
    _outline(a, cols[4])
    a[-1, :, 3] = 0
    return a


# ------------------------------------------------------------------ ice floes & icebergs
def floe_sprite(r, seed):
    return _floe(int(r), int(seed)).copy()


@lru_cache(maxsize=None)
def _floe(r, seed):
    rg = _rng(seed + 991)
    n = rg.integers(7, 10)
    angs = np.sort(rg.uniform(0, 2 * math.pi, n))
    rad = r * rg.uniform(0.78, 1.0, n)
    pts = [(math.cos(t) * q, math.sin(t) * q * 0.62) for t, q in zip(angs, rad)]
    th = max(2, r // 4)
    w = 2 * r + 3
    hh = int(2 * r * 0.62) + th + 4
    a = _new(hh, w)
    cx, cy = r + 1, r * 0.62 + 1

    def inside(x, y):
        c = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                c = not c
            j = i
        return c
    top = np.zeros((hh, w), bool)
    for y in range(hh):
        for x in range(w):
            top[y, x] = inside(x - cx, y - cy)
    # side face: top shifted down by th
    side = np.zeros_like(top)
    for k in range(1, th + 1):
        side[k:, :] |= top[:-k, :]
    side &= ~top
    a[side, :3] = C('#7fbcd8')
    a[side, 3] = 255
    # darker lower side line
    low = np.zeros_like(side)
    low[1:, :] = side[1:, :] & ~side[:-1, :] | (side[1:, :] & ~np.vstack([side[2:, :], np.zeros((1, w), bool)]))
    a[side & np.vstack([~side[1:, :], np.ones((1, w), bool)]), :3] = C('#5f9cc0')
    a[top, :3] = C('#f2f8ff')
    a[top, 3] = 255
    # lit edge top-left, blue shade bottom-right inside top
    yy, xx = np.nonzero(top)
    for y, x in zip(yy, xx):
        if y > 0 and not top[y - 1, x]:
            a[y, x, :3] = C('#ffffff')
        elif x + 1 < w and not top[y, x + 1] or (y + 1 < hh and not top[y + 1, x]):
            a[y, x, :3] = C('#cfe4f6')
    # crack lines
    if r > 6:
        x0 = int(cx + rg.integers(-r // 3, r // 3 + 1))
        for i in range(r // 2):
            yv = int(cy - r * 0.3 + i)
            xv = x0 + i // 2
            if 0 <= yv < hh and 0 <= xv < w and top[yv, xv]:
                a[yv, xv, :3] = C('#cfe4f6')
    return a


def berg_sprite(w, h, seed):
    """Pointed arctic iceberg (3/4): white lit face, blue shadow face."""
    return _berg(int(w), int(h), int(seed)).copy()


@lru_cache(maxsize=None)
def _berg(w, h, seed):
    rg = _rng(seed + 31)
    a = _new(h + 2, w)
    px = w / 2 + rg.uniform(-w * 0.15, w * 0.15)
    for y in range(h):
        t = y / max(1, h - 1)
        hw = 0.5 + t * (w / 2 - 0.5) + rg.uniform(-0.4, 0.4)
        for x in range(w):
            if abs(x - px) <= hw:
                c = '#ffffff' if x < px - 0.5 else '#c8dff2' if x < px + hw * 0.5 else '#9cc4e4'
                _put(a, y, x, c)
    for x in range(w):
        if a[h - 1, x, 3]:
            _put(a, h, x, '#4f8fbf')
    a = _pad(a)
    _outline(a, '#4a7fae')
    return a


def puff_sprite(r, ramp, seed):
    """Cloud puff (Sea of Clouds / drifting clouds): lit top-left, shaded bottom-right."""
    return _puff(int(r), tuple(ramp), int(seed)).copy()


@lru_cache(maxsize=None)
def _puff(r, ramp, seed):
    hi, lt, md, dk, out = ramp
    rg = _rng(seed + 3)
    w = 2 * r + 1
    h = int(r * 1.5) + 2
    a = _new(h, w)
    for y in range(h):
        for x in range(w):
            dx = (x - r) / (r + 0.4)
            dy = (y - h * 0.62) / (h * 0.62)
            if dx * dx + dy * dy <= 1 and y < h - 1:
                lv = -dx * 0.5 - dy * 0.9 + (rg.random() - 0.5) * 0.15
                c = hi if lv > 0.55 else lt if lv > 0.0 else md if lv > -0.45 else dk
                if dx * dx + dy * dy > 0.8 and dy > 0.1:
                    c = out
                _put(a, y, x, c)
    return a
