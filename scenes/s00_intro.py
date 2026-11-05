"""INTRO / title sequence (6 bars, 432 f, HUD off). Owner: intro director.

Bar 1  (0-71)    night sky over a moonlit sea of clouds, crescent moon, shooting star on `star` 36,
                 the camera tilts slowly down toward the clouds.
Bar 2  (72-143)  the camera dives into the clouds, they part like curtains and reveal the atlas board;
                 the whole map unrolls inside its gold frame (reveal 0 -> 1).
Bar 3  (144-215) `title`: the cartouche slams open, E-Q-U-E-S-T-R-I-A drop in on the 9 plings
                 (144 + 9k), the subtitle types on from 180; first light sweeps the map from the east.
Bar 4  (216-287) `dawn`: dawn reaches the whole map, the sun emblem flares, region names pop in on 8ths.
Bar 5  (288-359) `prompt`: pixel zoom (integer nearest 1x..9x) + mosaic into the detailed map at
                 Canterlot; the balloon drops in and moors; BEGIN JOURNEY blinks on every beat.
Bar 6  (360-431) `liftoff`: puff + burner, the card and prompt clear, the balloon rises and the camera
                 swings up into the pre-dawn Mode-7 sky = exactly the first frame of the Canterlot
                 travel shot (seamless under the diamond wipe 424-431).
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine import timeline as TL
from engine import loc_worldmap as WM
from engine.loc_worldmap import (draw_overview, draw_map, loc_xy, OVERVIEW_RECT, LEFT_PANEL,
                                 RIGHT_PANEL, MAP_W, MAP_H)
from engine.atlas import draw_balloon
from engine.title import draw_title_card

ID = 'intro'
N = seg_frames(ID)
BEAT = FRAMES_PER_BEAT
BAR = FRAMES_PER_BAR

T_STAR, T_TITLE, T_DAWN, T_PROMPT, T_LIFT = 36, 144, 216, 288, 360
H_NIGHT, H_DAWN, H_END = 4.5, 5.85, 5.5
CARD_Y = 3

MX, MY, MW, MH = OVERVIEW_RECT
_YY, _XX = np.mgrid[0:H, 0:W]
_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
_BAYER = _B4[_YY % 4, _XX % 4]


# =========================================================================== bar 1-2: sky + clouds
SKY_H = 320


@lru_cache(maxsize=1)
def _sky_tall():
    c = Canvas(W, SKY_H)
    c.gradient(0, 0, W, SKY_H, ['#0e0c20', '#141127', '#1b1934', '#232445', '#2c2f5a', '#3a3d6e',
                                '#4a4a80'], soft=0.7)
    return c


@lru_cache(maxsize=None)
def _moon(r=10):
    n = 2 * r + 3
    yy, xx = np.mgrid[0:n, 0:n] - (r + 1)
    d = np.sqrt(xx ** 2 + yy ** 2)
    disc = d <= r + 0.3
    ox, oy = r * 0.5, r * 0.3
    dc = np.sqrt((xx + ox) ** 2 + (yy + oy) ** 2)
    cres = disc & ~(dc <= r - 0.6)
    a = np.zeros((n, n, 4), np.uint8)
    a[disc] = (*col('#2b2d58'), 255)                    # earthshine on the dark side
    a[cres] = (*col('#eef0ff'), 255)
    a[cres & (dc <= r + 1.0)] = (*col('baby_blue'), 255)
    a[cres & (d > r - 1.2) & (xx > 0)] = (*col('white'), 255)
    a[cres & (dc > r + 1.0) & (d < r - 1.2) & (yy > r * 0.3)] = (*col('#d4dcff'), 255)
    for cx_, cy_ in ((0.4, 0.3), (0.6, -0.2), (0.2, 0.7), (0.75, 0.15)):
        X, Y = int(round(cx_ * r)) + r + 1, int(round(cy_ * r)) + r + 1
        if cres[Y, X] and dc[Y, X] > r + 1.5:
            a[Y, X, :3] = col('cloud_dk')
    return a


def _blit(c, rgba, x, y):
    x, y = int(x), int(y)
    sh, sw = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    m = sub[..., 3] >= 128
    c.a[y0:y1, x0:x1][m] = sub[..., :3][m]


# cloud layers: (top world y, parallax, drift px/f, ramp rim/mid/body/shadow/deep, seed, part speed)
_RAMPS = {
    'far': ('#9ea8dc', '#7880b8', '#5a6097', '#434678', '#2e3058'),
    'mid': ('#c3cbf2', '#949dd4', '#6d74ad', '#50548a', '#3a3c6a'),
    'near': ('#e6ecff', '#b2bcea', '#868fc9', '#60669f', '#454878'),
}
LAYERS = [
    (134, 0.75, 0.10, 'far', 11, 0.8),
    (148, 1.0, 0.17, 'mid', 22, 1.0),
    (166, 1.3, 0.26, 'near', 33, 1.25),
]
LX0, LWID, LH = -170, W + 340, 330
_SEAM = W // 2 - LX0


def _cloud_puff(a, cx, cy, r, rim, lit, body, shd):
    """One shaded cloud puff (light from the upper right) painted into RGBA array a."""
    LH_, LW_ = a.shape[:2]
    y0, y1 = max(0, cy - r), min(LH_, cy + r + 1)
    x0, x1 = max(0, cx - r), min(LW_, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    ins = d <= r + 0.2
    nl = (dx * 0.38 - dy * 0.92) / max(1, r)
    reg = a[y0:y1, x0:x1]
    # lit cap = a circle offset toward the light (curved terminator), thin rim on the lit edge
    dl = np.sqrt((dx - 0.22 * r) ** 2 + (dy + 0.42 * r) ** 2)
    cc = np.where((dl < 0.86 * r)[..., None], lit, body)
    cc = np.where(((d > r - 2.2) & (nl > 0.45))[..., None], rim, cc)
    cc = np.where(((d > r - 1.2) & (nl < -0.30))[..., None], shd, cc)
    reg[ins, :3] = cc[ins]
    reg[ins, 3] = 255


@lru_cache(maxsize=None)
def _cloud_half(li, side):
    """RGBA (LH, LWID) of one half (0 left, 1 right) of cloud layer li; the top bumps start near y 10.
    Silhouette = one row of big puffs over a dithered body gradient with sparse, soft inner banks."""
    top, par, drift, ramp, seed, _ = LAYERS[li]
    rim, lit, body, shd, deep = (np.array(col(x), np.float32) for x in _RAMPS[ramp])
    a = np.zeros((LH, LWID, 4), np.uint8)
    R = rng(seed)
    seam = _SEAM
    y_top = 34

    def depth(y):
        return clamp((y - y_top - 10) / 170.0) * 0.75

    def tone(cc, k):
        return np.clip(cc * (1 - k) + deep * k, 0, 255).astype(np.uint8)
    # body: dithered vertical gradient (5 tones)
    xa, xb = (0, seam + 3) if side == 0 else (seam - 3, LWID)
    tones = np.array([tone(body, k) for k in np.linspace(0, 0.75, 5)])

    def grad(yy, xx):
        lvl = np.clip((yy - y_top - 10) / 170.0, 0, 1) * 4
        return tones[np.clip(np.floor(lvl + _B4[yy % 4, xx % 4] - 0.5).astype(int), 0, 4)]
    yy, xx = np.mgrid[y_top + 4:LH, xa:xb]
    a[y_top + 4:, xa:xb, :3] = grad(yy, xx)
    a[y_top + 4:, xa:xb, 3] = 255
    rows = [y_top] + list(range(y_top + 56, LH - 10, 44))
    for k, yk in enumerate(rows):
        if k == 0:
            cr, cl, cb, cs = (tone(x, 0) for x in (rim, lit, body, shd))
        else:
            loc = body * (1 - depth(yk + 8)) + deep * depth(yk + 8)
            cb = np.clip(loc, 0, 255).astype(np.uint8)
            cl = np.clip(loc * 0.78 + lit * 0.22, 0, 255).astype(np.uint8)
            cr = np.clip(loc * 0.6 + lit * 0.4, 0, 255).astype(np.uint8)
            cs = np.clip(loc * 0.6 + shd * 0.4, 0, 255).astype(np.uint8)
        x = int(R.integers(-6, 10))
        while x < LWID:
            if k == 0:
                r = int(R.integers(10, 25))
                jy = int(R.integers(-6, 6)) + (r - 16) // -2
            else:
                r = int(R.integers(11, 20))
                jy = int(R.integers(-5, 6))
            skip = k > 0 and R.random() < 0.5
            inside = (x < seam) if side == 0 else (x >= seam)
            if inside and not skip:
                _cloud_puff(a, x, yk + 8 + jy, r, cr, cl, cb, cs)
                if k == 0 and R.random() < 0.5:      # small cauliflower bumps on the big puffs
                    rr = max(5, r // 2)
                    _cloud_puff(a, x + r // 2, yk + 8 + jy - r // 2 - 1, rr, cr, cl, cb, cs)
            x += int(R.integers(int(r * 0.9), int(r * 1.6) + 2))
    # edge puffs down the seam: body-gradient coloured (invisible while the halves touch), they
    # give the curtain a scalloped silhouette when it parts
    for ye in range(y_top + 8, LH, 15):
        r = int(R.integers(9, 13))
        ex = (seam + int(R.integers(0, 5))) if side == 0 else (seam - int(R.integers(0, 5)))
        y0, y1 = max(0, ye - r), min(LH, ye + r + 1)
        yy, xx = np.mgrid[y0:y1, ex - r:ex + r + 1]
        m = (yy - ye) ** 2 + (xx - ex) ** 2 <= r * r + 0.5
        reg = a[y0:y1, ex - r:ex + r + 1]
        reg[m, :3] = grad(np.maximum(yy, y_top + 4), xx)[m]
        reg[m, 3] = 255
    return a


def _cloud_layer(c, li, f, cam, part):
    top, par, drift, ramp, seed, pspd = LAYERS[li]
    sy = int(round(top - cam * par - part * 26 * pspd)) - 34
    dxd = int(f * drift)
    spread = int(round(part * (W / 2 + 120) * pspd))
    deep = col(_RAMPS[ramp][4])
    for side in (1, 0):
        spr = _cloud_half(li, side)
        x = LX0 - (dxd % 40) + (spread if side else -spread)
        _blit(c, spr, x, sy)
        if sy + LH < H:          # solid below the sprite
            if side == 0:
                xa, xb = max(0, x), max(0, min(W, x + _SEAM))
            else:
                xa, xb = max(0, x + _SEAM), W
            if xb > xa:
                c.a[max(0, sy + LH):, xa:xb] = deep


def _cam_sky(f):
    if f < BAR:
        return 50 * ease_in_out(seg(f, 0, 71))
    return 50 + 150 * ease_in(seg(f, 72, 95))


def _shooting_star(c, f):
    if not (30 <= f <= 52):
        return
    sx, sy, ex, ey = 30, 10, 176, 62
    t = ease_out(seg(f, 30, 45))
    hx, hy = sx + (ex - sx) * t, sy + (ey - sy) * t
    fade = 1 - seg(f, 45, 52)
    ux, uy = (ex - sx), (ey - sy)
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    L = int(30 * (0.4 + 0.6 * seg(f, 30, 35)) * (1 - 0.6 * seg(f, 42, 52)))
    trail = ['white', 'white', 'cream', 'baby_blue', 'baby_blue', 'light_sky', 'periwinkle', '#7d86c4',
             '#5a5fa0', '#3e4274']
    for i in range(L, 0, -1):
        u = i / max(1, L)
        px = int(round(hx - ux * i))
        py = int(round(hy - uy * i))
        ci = min(len(trail) - 1, int(u * len(trail) + (1 - fade) * 4))
        if u > 0.55 and (i % 2):
            continue
        c.px(px, py, trail[ci])
        if u < 0.35:
            c.px(px, py + 1, trail[min(len(trail) - 1, ci + 2)])
    if fade > 0.05:
        hxi, hyi = int(round(hx)), int(round(hy))
        fx.glow(c, hxi, hyi, 8, 'baby_blue', 0.45 * fade)
        c.rect(hxi, hyi, 2, 2, 'white')
        if 34 <= f <= 39:
            s_ = 4 if f in (36, 37) else 2
            for k in range(2, s_ + 2):
                cc = 'white' if k < s_ else 'light_sky'
                for dx, dy in ((k, 0), (-k + 1, 0), (0, k), (0, -k + 1)):
                    c.px(hxi + dx, hyi + dy, cc)


def _render_sky(f, part=0.0, board=None):
    cam = _cam_sky(f)
    c = Canvas()
    off = int(round(cam * 0.45))
    if board is None or part <= 0:
        c.a[:] = _sky_tall().a[off:off + H]
        fx.stars(c, f, seed=5, n=120, region=(0, -off, W, 190))
        fx.stars(c, f, seed=9, n=30, region=(0, -off, W, 110), colors=('white', 'butter', 'cotton'))
        for i, (bx, by) in enumerate(((58, 40), (204, 22), (122, 70), (286, 84), (20, 96))):
            fx.sparkle(c, bx, by - off, f, 'white', period=40 + 9 * i, phase=i * 0.37, size=2)
        mx, my = 246, 38 - off
        fx.glow(c, mx, my, 34, '#6c74c0', 0.30)
        fx.glow(c, mx, my, 18, '#9aa8ff', 0.25)
        _blit(c, _moon(13), mx - 14, my - 14)
        _shooting_star(c, f)
    else:
        c.a[:] = board.a
    for li in range(3):
        _cloud_layer(c, li, f, cam, part)
        if li == 0 and (board is None or part <= 0):
            # moonlight sheen on the far cloud tops under the moon
            fx.glow(c, 246, int(134 - cam * 0.75) + 6, 40, '#b8c4ff', 0.16)
    return c


# =========================================================================== the atlas board
def _backdrop(c, x, y, w, h):
    """Empty velvet area (map not yet unrolled)."""
    if h <= 0:
        return
    c.rect(x, y, w, h, '#211d3c')
    sub = c.a[y:y + h, x:x + w]
    yy, xx = np.mgrid[y:y + h, x:x + w]
    pat = ((xx + yy) % 6 == 0) & ((xx - yy) % 6 == 0)
    sub[pat] = col('#2e2a52')


_ROLLC = ['#4a3426', 'plum_brown', 'clay', 'sand', 'cream', 'paper', 'tan', 'clay', 'rust', '#4a3426']


def _map_roll(c, ry):
    """Horizontal parchment roll whose top edge is at row ry, across the map frame, with gold knobs."""
    x0, x1 = MX - 3, MX + MW + 3
    c.blend(MX, ry + len(_ROLLC), MW, 3, '#0c0a18', 0.5)
    for i, cc in enumerate(_ROLLC):
        c.hline(x0, x1 - 1, ry + i, cc)
    # spiral hint on the right end
    c.px(x1 - 3, ry + 4, 'rust')
    c.px(x1 - 4, ry + 5, 'rust')
    for kx in (x0 - 3, x1):
        c.rect(kx, ry + 2, 3, 6, 'brass')
        c.rect(kx, ry + 3, 3, 4, 'gold')
        c.px(kx + 1, ry + 3, 'butter')
        c.frame(kx - 1, ry + 1, 5, 8, '#4a3426')


def _overview_pair(f, hour_day):
    """(night canvas incl. frame, day map rect array) for the dawn composite."""
    cn = Canvas()
    draw_overview(cn, f, hour=H_NIGHT, found=0, labels=0, reveal=1.0)
    if hour_day is None:
        return cn, None
    cd = Canvas()
    draw_overview(cd, f, hour=hour_day, found=0, labels=0, reveal=1.0, frame=False)
    return cn, cd.a[MY:MY + MH, MX:MX + MW]


def _board(f, reveal=1.0, front=None, hour_day=H_DAWN, panels=True):
    """The atlas: frame + panels + overview (dawn region right of `front`, x in screen px)."""
    if front is None or front >= MX + MW + 12:
        c, day = _overview_pair(f, None)
    elif front <= MX - 14:
        c = Canvas()
        draw_overview(c, f, hour=hour_day, found=0, labels=0, reveal=1.0)
        day = None
    else:
        c, day = _overview_pair(f, hour_day)
        reg = c.a[MY:MY + MH, MX:MX + MW]
        xs = np.arange(MX, MX + MW)[None, :]
        ys = np.arange(MY, MY + MH)[:, None]
        wob = np.sin(ys * 0.11 + f * 0.05) * 2.0
        thr = front + wob + (_BAYER[MY:MY + MH, MX:MX + MW] - 0.5) * 8
        m = xs >= thr
        reg[m] = day[m]
        # terminator: warm first-light band just east of the front
        dist = xs - (front + wob)
        k = np.clip(1 - (dist - 2) / 10.0, 0, 1) * (dist >= -1)
        k = np.floor(k * 3 + 0.5) / 3
        add = np.array(col('#ff9a5a'), np.float32)
        r2 = reg.astype(np.float32) + add[None, None] * k[..., None] * 0.22
        line = np.broadcast_to((np.abs(dist) < 1.0) & (((ys + f // 2) % 3) != 0), r2.shape[:2])
        r2[line] = r2[line] * 0.5 + np.array(col('butter'), np.float32) * 0.5
        reg[:] = np.clip(r2, 0, 255).astype(np.uint8)
    if reveal < 1:
        rows = int(round(MH * clamp(reveal)))
        _backdrop(c, MX, MY + rows, MW, MH - rows)
        if rows < MH:
            _map_roll(c, MY + rows - 2)
    if panels:
        _panels(c, f)
    return c


# --------------------------------------------------------------- panel emblems (sun / moon)
def _ring_pts(cx, cy, r):
    out = []
    for i in range(int(2 * math.pi * r * 1.5)):
        a = i / (2 * math.pi * r * 1.5) * 2 * math.pi
        out.append((int(round(cx + math.cos(a) * r)), int(round(cy + math.sin(a) * r))))
    return sorted(set(out))


def _medallion(c, cx, cy, r, rim1, rim2, fill_top, fill_bot, stud):
    c.circle(cx + 1, cy + 2, r + 2, '#120f26')
    c.circle(cx, cy, r + 2, rim2)
    c.circle(cx, cy, r + 1, rim1)
    c.circle(cx, cy, r, fill_bot)
    c.ellipse(cx, cy - r // 3, r - 2, r - 3, fill_top)
    for i in range(8):
        a = i * math.pi / 4
        c.px(int(round(cx + math.cos(a) * (r + 1))), int(round(cy + math.sin(a) * (r + 1))), stud)


def _moon_medal(c, f, k):
    """Left panel: crescent moon medallion; k = 1 night .. 0 day (dims)."""
    cx, cy, r = 38, 92, 18
    dim = 1 - k
    rim1 = mix('stone', '#5a5478', dim * 0.6)
    _medallion(c, cx, cy, r, rim1, mix('cloud_dk', '#3a3258', dim * 0.6),
               mix('#2c2f62', '#3c3560', dim), mix('#1b1d45', '#2c2648', dim), 'gold')
    if k > 0.02:
        fx.glow(c, cx, cy, 22, '#6c74c0', 0.22 * k)
    rgba = _moon().copy()
    if dim > 0:
        rgb = rgba[..., :3].astype(np.float32)
        tgt = np.array(col('#6a6290'), np.float32)
        rgba[..., :3] = np.clip(rgb * (1 - dim * 0.6) + tgt * dim * 0.6, 0, 255).astype(np.uint8)
    _blit(c, rgba, cx - 11, cy - 11)
    for i, (sx, sy) in enumerate(((cx - 9, cy - 8), (cx + 9, cy + 7), (cx - 6, cy + 10), (cx + 11, cy - 6))):
        if k > 0.2:
            fx.sparkle(c, sx, sy, f, 'white' if i % 2 == 0 else 'baby_blue', period=34 + 7 * i, phase=i * 0.3,
                       size=1 + (i == 0))
        else:
            c.px(sx, sy, '#4a4470')


def _sun_medal(c, f, k):
    """Right panel: sun medallion with turning rays; k = 0 night (dim) .. 1 lit."""
    cx, cy, r = 282, 92, 18
    _medallion(c, cx, cy, r, mix('#5a4a6a', 'gold', k), mix('#3a3258', 'brass', k),
               mix('#2a2550', '#ffcf9a', k), mix('#221d44', '#f6a07a', k), mix('#5a4a6a', 'butter', k))
    if k > 0.02:
        fx.glow(c, cx, cy, 30, '#ffb070', 0.30 * k)
    ang0 = f * 0.012
    core = mix('#6a5a5a', 'gold', k)
    ray_l = mix('#5a4a50', 'amber', k)
    ray_s = mix('#4a3f50', 'sand', k)
    for i in range(12):
        a = ang0 + i * math.pi / 6
        L = 13 if i % 2 == 0 else 10
        x0, y0 = cx + math.cos(a) * 8, cy + math.sin(a) * 8
        x1, y1 = cx + math.cos(a) * L, cy + math.sin(a) * L
        c.line(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)), ray_l if i % 2 == 0 else ray_s)
    c.circle(cx, cy, 7, mix('#4a3f50', 'amber', k))
    c.circle(cx, cy, 6, core)
    c.circle(cx - 1, cy - 1, 4, mix('#7a6a60', 'butter', k))
    c.px(cx - 2, cy - 3, mix('#8a7a70', 'white', k))
    if k > 0.6:
        for i, (sx, sy) in enumerate(((cx - 12, cy - 12), (cx + 13, cy + 11), (cx + 12, cy - 13))):
            fx.sparkle(c, sx, sy, f, 'butter', period=30 + 6 * i, phase=i * 0.4, size=2)


def _filigree(c, cx, y0, y1, color, f, seed):
    c.vline(cx, y0, y1, mix(color, '#2a2550', 0.45))
    for y in range(y0 + 3, y1 - 2, 10):
        c.px(cx - 1, y, color)
        c.px(cx + 1, y, color)
        c.px(cx, y - 1, color)
        c.px(cx, y + 1, color)
        c.px(cx, y, 'hot_pink' if (y // 10 + seed) % 3 == 0 else 'butter')
    for y in (y0, y1):
        c.rect(cx - 2, y - 1, 5, 3, color)
        c.px(cx, y, 'butter')


def _panel_light(f):
    """(moon k night 1..0, sun k 0..1)."""
    sun = smooth(seg(f, 138, 156))
    moon = 1 - smooth(seg(f, 200, 216))
    return moon, sun


def _panels(c, f):
    moon_k, sun_k = _panel_light(f)
    lx = LEFT_PANEL[0] + LEFT_PANEL[2] // 2
    rx = RIGHT_PANEL[0] + RIGHT_PANEL[2] // 2
    _filigree(c, lx, 16, 66, mix('stone', '#5a5478', 1 - moon_k), f, 0)
    _filigree(c, lx, 118, 166, mix('stone', '#5a5478', 1 - moon_k), f, 1)
    _filigree(c, rx, 16, 66, mix('#6a5a6a', 'gold', sun_k), f, 2)
    _filigree(c, rx, 118, 166, mix('#6a5a6a', 'gold', sun_k), f, 0)
    _moon_medal(c, f, moon_k)
    _sun_medal(c, f, sun_k)
    if T_DAWN <= f < T_DAWN + 14:      # dawn flare on the sun emblem
        u = seg(f, T_DAWN, T_DAWN + 14)
        fx.glow(c, 282, 92, int(20 + 26 * u), 'butter', 0.5 * (1 - u))
    if sun_k > 0:                      # warm light spilling from the east edge
        fx.light_rect(c, RIGHT_PANEL[0] - 6, 0, 6, H, '#ff9a6a', 0.10 * sun_k, fade='left')


# --------------------------------------------------------------- region labels (bar 4)
_LBL = [l for l in WM._OV_LABELS if l[0] != 'FROZEN NORTH']   # the far north sits under the title


def _labels(c, f):
    for i, (txt, nx, ny, cc) in enumerate(_LBL):
        t0 = T_DAWN + 4 + i * 9
        if f < t0 or (f >= T_PROMPT - 5 and ny > 0.9):
            continue
        tw = text_width(txt, 'tiny')
        lx = int(clamp(MX + int(nx * MW) - tw // 2, MX + 2, MX + MW - tw - 2))
        ly = MY + int(ny * MH) - 2
        u = f - t0
        if u < 2:
            text(c, lx, ly - 1, txt, 'white', font='tiny', outline='#2e222f')
        elif u < 5:
            text(c, lx, ly, txt, 'butter', font='tiny', outline='#2e222f')
        else:
            text(c, lx, ly, txt, cc, font='tiny', outline='#2e222f')
        if u < 10:
            fx.sparkle(c, lx - 3, ly + 2, u, 'white', period=10, size=2)


def _dawn_sparkles(c, f):
    """Twinkles across the land as the light arrives (bars 3-4)."""
    for i in range(26):
        sx = MX + 4 + int(hash01(7, i, 1) * (MW - 8))
        sy = MY + 40 + int(hash01(7, i, 2) * (MH - 60))
        front = _front(f)
        if sx < front + 4:
            continue
        ph = hash01(7, i, 3)
        fx.sparkle(c, sx, sy, f, ('white', 'butter', 'cotton')[i % 3], period=38 + int(ph * 30), phase=ph,
                   size=1 + (i % 4 == 0))


def _front(f):
    """x of the dawn terminator (moves east -> west across bar 3, a steady sweep)."""
    u = seg(f, T_TITLE - 2, T_DAWN)
    return lerp(MX + MW + 12, MX - 16, 0.35 * u + 0.65 * smooth(u))


# =========================================================================== bar 5-6: Canterlot
CANT = np.array(loc_xy('canterlot'), float)
_SEG_C = TL.BY_ID['canterlot']
_P_ARC = WM._arc(CANT + np.array([-30.0, 420.0]), CANT, 1)
_END = WM._balloon_end('canterlot')
_CAM_C = np.round(WM._cam_center_for(CANT))          # top-down camera centre of the travel tail
_PAN0 = np.array([18.0, 10.0])                       # bar-5 camera start offset (eases to 0)


def _cam5(f):
    k = ease_out(seg(f, 296, 356))
    return np.round(_CAM_C + _PAN0 * (1 - k))


def _ov_to_screen_scale():
    return MW / MAP_W, MH / MAP_H


def _zoom_frame(src, z, A, B):
    """Integer nearest zoom of canvas src by z, overview point A drawn at screen point B."""
    sx = np.floor(A[0] + (_XX - B[0] + 0.5) / z).astype(np.int32)
    sy = np.floor(A[1] + (_YY - B[1] + 0.5) / z).astype(np.int32)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    out = Canvas(color='#1b1934')
    out.a[ok] = src.a[sy[ok], sx[ok]]
    return out


def _mosaic(c, b, ax, ay):
    if b <= 1:
        return c
    sx = np.clip(((_XX - ax) // b) * b + ax + b // 2, 0, W - 1)
    sy = np.clip(((_YY - ay) // b) * b + ay + b // 2, 0, H - 1)
    c.a[:] = c.a[sy, sx]
    return c


_ZOOM_STEPS = {289: 2, 290: 2, 291: 3, 292: 4, 293: 5, 294: 6, 295: 7, 296: 9}
_MOSAIC_STEPS = {297: 8, 298: 6, 299: 4, 300: 3, 301: 2}


def _cant_screen(cam):
    return CANT - cam + np.array([W / 2, H / 2])


def _shadow(c, x, y, rx, ry=1):
    c.blend(x - rx, y - ry, 2 * rx + 1, 2 * ry + 1, '#1a1428', 0.35)
    c.blend(x - rx + 1, y, 2 * rx - 1, 1, '#1a1428', 0.25)


def _smoke_puff(c, f, x, y, t0):
    """Liftoff puff: a ring of small shaded smoke balls spreading out from the basket and fading."""
    u = f - t0
    if not (0 <= u < 24):
        return
    k = u / 24.0
    for i in range(9):
        a = i / 9 * 2 * math.pi + 0.35
        r = 2 + ease_out(k) * 15
        px = int(round(x + math.cos(a) * r * 1.25))
        py = int(round(y + math.sin(a) * r * 0.5 - k * 5))
        rad = 3 if k < 0.35 else (2 if k < 0.7 else 1)
        if k > 0.6 and (i + u) % 3 == 0:
            continue
        c.circle(px + 1, py + 1, rad, mix('cloud_dk', 'indigo', k * 0.5))
        c.circle(px, py, rad, mix('white', 'cloud_dk', k))
        if rad > 1:
            c.px(px - 1, py - 1, 'white')
    if u < 4:
        fx.glow(c, x, y - 4, 14, 'butter', 0.35 * (1 - u / 4))


def _prompt(c, f):
    if not (T_PROMPT - 4 <= f < T_LIFT + 10):
        return
    s = 'BEGIN JOURNEY'
    tw = text_width(s)
    pw, ph = tw + 22, 15
    cx, cy = W // 2, 160
    k = ease_out_back(seg(f, T_PROMPT - 4, T_PROMPT + 2))
    if f >= T_LIFT + 3:
        k = 1 - ease_in(seg(f, T_LIFT + 3, T_LIFT + 9))
    if k <= 0.05:
        return
    w = max(6, int(pw * k))
    x0, y0 = cx - w // 2, cy - ph // 2
    c.blend(x0 + 1, y0 + 2, w, ph, '#0c0a18', 0.45)
    c.round_rect(x0, y0, w, ph, 'gold', r=2)
    c.round_rect(x0 + 1, y0 + 1, w - 2, ph - 2, '#231c40', r=1)
    c.hline(x0 + 2, x0 + w - 3, y0 + 1, '#3b3266')
    if k < 0.9:
        return
    tx = cx - tw // 2 + 5
    sel = T_LIFT <= f < T_LIFT + 3
    on = ((f - T_PROMPT) % BEAT) < 12 or sel
    if on:
        text(c, tx, y0 + 4, s, 'white' if (sel or (f - T_PROMPT) % BEAT < 2) else 'cream', shadow='#4a2a6a')
    # pointing-triangle cursor, nudges right on each beat
    nb = 1 if ((f - T_PROMPT) % BEAT) < 4 else 0
    ax = tx - 9 + nb
    ay = y0 + 4
    tri = ['#...', '##..', '###.', '####', '###.', '##..', '#...']
    for j, row in enumerate(tri):
        for i, ch in enumerate(row):
            if ch == '#':
                c.px(ax + i, ay + j, 'gold' if j < 3 else 'amber')
    c.px(ax, ay + 1, 'butter')
    if sel:
        fx.glow(c, cx, cy, 30, 'butter', 0.4)


def _balloon_drop(f):
    """(alt offset above the moored height, visible) for the bar-5 arrival."""
    if f < 298:
        return None
    return 70 * (1 - ease_out(seg(f, 298, 306)))


def _lift(f):
    """Extra balloon height after liftoff (px at top-down scale)."""
    return 26 * ease_out(seg(f, T_LIFT, T_LIFT + 22))


T_ASC = 368          # top-down detailed map hands over to the perspective ascent


def _bar5(f):
    """Top-down detailed map at Canterlot with the moored balloon (bars 5 and early 6)."""
    hour = H_DAWN
    cam = _cam5(f)
    c = Canvas()
    draw_map(c, int(cam[0]), int(cam[1]), f, hour=hour, found=0, labels=306 <= f < T_LIFT + 6)
    G = _END - cam + np.array([W / 2, H / 2])
    gx, gy = int(round(G[0])), int(round(G[1]))
    drop = _balloon_drop(f)
    if drop is not None:
        alt = 18 + drop
        lift = _lift(f)
        _shadow(c, gx, gy, 5)
        if f < T_LIFT and drop < 1:
            # mooring rope + stake
            c.vline(gx + 2, gy - 17, gy, 'tan')
            c.px(gx + 2, gy, '#4a3426')
            c.px(gx + 3, gy - 8, 'tan')
            c.px(gx + 1, gy, '#4a3426')
        flame = f >= T_LIFT - 10
        draw_balloon(c, gx, int(round(gy - alt - lift)), f, size='m', flame=flame)
        if 306 <= f < 320:
            fx.sparkles(c, gx, gy - 20, f - 306, radius=11, n=7, period=14, seed=3,
                        colors=('white', 'gold', 'cotton'))
        _smoke_puff(c, f, gx, gy - 6, T_LIFT)
    return c


def _ascent_cam(fv):
    """Camera of the Canterlot sky approach at virtual travel frame fv (mirror of _travel_sky)."""
    q = ease_in_out(seg(fv, 0, 46))
    G = WM._ground_pos(_P_ARC, _END, q)
    tilt = 1 - smooth(seg(fv, 30, 47))
    tq = min(0.999, max(0.001, q))
    tan = WM._bez_tan(_P_ARC, tq)
    psi_leg = math.atan2(tan[0], -tan[1])
    lift = 1 - smooth(seg(fv, 0, 30))
    psi = WM._ang_lerp(0.0, psi_leg, smooth(tilt))
    pitch = lerp(90.0, lerp(WM._M7_PITCH, 12.0, lift), tilt)
    dist = lerp(WM._F, lerp(WM._M7_DIST, 900.0, lift), tilt)
    cyp = lerp(H / 2, lerp(WM._M7_CYP, 150.0, lift), tilt)
    sxo = lerp(WM._DEST_SX - W / 2, 0.0, tilt)
    vt = lerp(WM._DEST_SY - cyp, lerp(WM._M7_VT, 20.0, lift), tilt)
    anchor = CANT * (1 - tilt) + G * tilt
    cam = WM.cam_for(anchor, sxo, vt, psi, pitch, dist, cyp)
    return cam, G, tilt


def _bar6(f):
    """Rise into the pre-dawn sky: the Canterlot approach played backwards, ends on its frame 0."""
    u = seg(f, T_ASC, 419)
    fv = 47 * (1 - (0.35 * u + 0.65 * smooth(u)))
    hour = lerp(H_DAWN, H_END, smooth(seg(f, T_ASC, 390)))
    cam, G, tilt = _ascent_cam(fv)
    c = Canvas()
    WM.render_persp(c, cam, f, hour, 0, balloon=None)
    WM._cloud_billboards(c, cam, f, hour)
    p = cam.project(G[0], G[1])
    if p is not None:
        shx, shy, k = p
        if tilt > 0.5:
            rx = int(clamp(k * 5, 2, 8))
            WM._shadow_ell(c.a, int(shx), int(shy), rx, max(1, rx // 3))
        else:
            WM._shadow_ell(c.a, int(shx), int(shy), 5, 1)
        # balloon height: the lifted top-down height blends into the travel shot's fixed framing
        a_world = (18 + _lift(f)) * max(0.6, min(1.6, k))
        a = lerp(a_world, 28, smooth(seg(tilt, 0.25, 0.75)))
        by = shy - a
        if tilt > 0.5:
            by = max(40, by)
        draw_balloon(c, int(shx), int(round(by)), f, size='m', flame=f < 412)
        _smoke_puff(c, f, int(shx), int(shy) - 6, T_LIFT)
    return c


# =========================================================================== render
def render(f):
    if f < 92:
        c = _render_sky(f)
        if f < 14:
            fx.fade(c, 'ink', 1 - smooth(seg(f, 0, 14)))
        return c
    if f < T_TITLE:
        reveal = ease_in_out(seg(f, 100, 140))
        board = _board(f, reveal=reveal)
        part = ease_in_out(seg(f, 94, 124))
        if part < 1:
            c = _render_sky(f, part=part, board=board)
        else:
            c = board
        if 140 <= f < T_TITLE:      # compass + scale bar land with a glint
            fx.sparkle(c, MX + 9, MY + 92, f - 140, 'white', period=8, size=3)
        if f >= T_TITLE - 8:        # the cartouche unrolls into the downbeat
            draw_title_card(c, f, 160, CARD_Y, t0=T_TITLE, sub_t0=180, step=9)
        return c
    if f < T_PROMPT + 1:
        c = _board(f, front=_front(f))
        if f >= T_DAWN:
            _labels(c, f)
        _dawn_sparkles(c, f)
        draw_title_card(c, f, 160, CARD_Y, t0=T_TITLE, sub_t0=180, step=9, shine_t0=T_DAWN + 4,
                        shine_every=T_PROMPT - T_DAWN)
        if T_TITLE <= f < T_TITLE + 6:          # title boom flash
            fx.glow(c, 160, CARD_Y + 18, 70, 'white', 0.35 * (1 - seg(f, T_TITLE, T_TITLE + 6)))
        if f >= 270:                             # zoom box locks onto Canterlot
            _zoom_box(c, f)
        _prompt(c, f)
        return c
    if f <= 296:                                  # pixel zoom into Canterlot
        src = _board(f, front=MX - 40)
        z = _ZOOM_STEPS.get(f, 1)
        sxs, sys_ = _ov_to_screen_scale()
        A = (MX + CANT[0] * sxs, MY + CANT[1] * sys_)
        Bend = _cant_screen(_cam5(297))
        t = (z - 1) / 8.3
        B = (lerp(A[0], Bend[0], t), lerp(A[1], Bend[1], t))
        c = _zoom_frame(src, z, A, B)
    elif f < T_ASC:
        c = _bar5(f)
        if f in _MOSAIC_STEPS:
            S = _cant_screen(_cam5(f))
            _mosaic(c, _MOSAIC_STEPS[f], int(S[0]), int(S[1]))
    else:
        c = _bar6(f)
    # UI layer: title card stays through bar 5, slides away on liftoff
    if f < T_LIFT + 18:
        dy = -int(round(48 * ease_in(seg(f, T_LIFT + 2, T_LIFT + 16))))
        draw_title_card(c, f, 160, CARD_Y, t0=T_TITLE, sub_t0=180, step=9, shine_t0=T_DAWN + 4,
                        shine_every=T_PROMPT - T_DAWN, dy=dy)
    _prompt(c, f)
    return c


def _zoom_box(c, f):
    """Gold viewfinder brackets around the Canterlot screen footprint on the overview (270-288)."""
    sxs, sys_ = _ov_to_screen_scale()
    cam = _cam5(297)
    cx = MX + cam[0] * sxs
    cy = MY + cam[1] * sys_
    bw, bh = W * sxs, H * sys_
    k = ease_out_back(seg(f, 270, 278))
    grow = 1 + (1 - k) * 1.5
    x0 = int(round(cx - bw * grow / 2))
    x1 = int(round(cx + bw * grow / 2))
    y0 = int(round(cy - bh * grow / 2))
    y1 = int(round(cy + bh * grow / 2))
    on = (f // 3) % 2 == 0 or f >= 279
    colr = 'white' if on else 'gold'
    L = 5
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for i in range(-1, L + 1):          # ink backing for contrast on the busy map
            for j in (-1, 1):
                c.px(x + sx * i, y + j, 'ink')
                c.px(x + j, y + sy * i, 'ink')
        for i in range(L):
            c.px(x + sx * i, y, colr)
            c.px(x, y + sy * i, colr)
    if f >= 279:
        sxp, syp = int(MX + CANT[0] * sxs), int(MY + CANT[1] * sys_)
        fx.sparkle(c, sxp, syp - 3, f, 'white', period=9, size=2)


SFX = [
    (33, 'star', 0.55),
    (78, 'wind', 0.30),
    (96, 'whoosh', 0.35),
    (T_DAWN, 'shimmer', 0.32),
    (T_PROMPT - 6, 'whoosh', 0.40),
    (306, 'sparkle', 0.35),
    (T_LIFT, 'puff', 0.60),
    (T_LIFT + 2, 'steam', 0.22),
    (376, 'wind', 0.38),
]
