"""OUTRO / finale (7 bars, 504 f, HUD off). Owner: outro director.

The atlas board of the intro returns at midnight, now complete.

Bars 1-2 (0-143)    the whole journey redraws itself on the map: the pink route runs from Canterlot to
                    the Castle of the Two Sisters, every arrival landing on a 16th of the `route` blips
                    (f 9 .. 135); each stop flashes, the explorer's balloon (xs) rides the pen. Left
                    panel: AREAS 07/20 counter + a 4x5 grid of star slots; right panel: a postcard of
                    the stop just reached, its name, a progress bar and the percentage.
Bar 3   (144-215)   `complete`: a big rose "MAP COMPLETE / 20/20 . 100%" rubber stamp slams onto the
                    map (white flash, shake, sparkle ring, confetti); the title plate unrolls at the
                    top: EQUESTRIA / A PIXEL WORLD ATLAS (silver lettering, moonlit).
Bar 4   (216-287)   `keychange`: the plate flips to "MAP, ART & MUSIC / MADE WITH CODE"; the panels
                    turn into the moon / sun medallions of the intro; first light warms the east.
Bar 5   (288-359)   "FAN-MADE TRIBUTE / FIRST EDITION . LANDSCAPES ONLY" + a small trademark line.
Bars 6-7 (360-503)  a new dawn sweeps the map east -> west; the balloon lifts off the castle and
                    floats away over the sun medallion, out of the top-right corner; `final` 432:
                    THE END (gold) on a cartouche; everything fades to ink by f 496.
"""
import math
from functools import lru_cache

import numpy as np

from engine import *
from engine import timeline as TL
from engine import loc_worldmap as WM
from engine.core import bayer
from engine.loc_worldmap import draw_overview, overview_xy, OVERVIEW_RECT, LEFT_PANEL, RIGHT_PANEL
from engine.atlas import draw_balloon
from engine.title import draw_title, title_size, draw_cartouche, draw_subtitle

ID = 'outro'
N = seg_frames(ID)
BEAT = FRAMES_PER_BEAT
BAR = FRAMES_PER_BAR

T_R0, T_R1 = 9, 135                       # route: first and last arrival (8th-note blips)
T_COMPLETE, T_KEY, T_B5, T_B6, T_FINAL = 144, 216, 288, 360, 432
T_FADE0, T_FADE1 = 462, 498

MX, MY, MW, MH = OVERVIEW_RECT
TOUR = TL.TOUR
NLEG = len(TOUR) - 1
BG = '#1b1934'
LCX = LEFT_PANEL[0] + LEFT_PANEL[2] // 2      # 38
RCX = RIGHT_PANEL[0] + RIGHT_PANEL[2] // 2    # 282

_YY, _XX = np.mgrid[0:H, 0:W]
_BAY = bayer(H, W)

NAMES = {
    'canterlot': 'CANTERLOT', 'cloudsdale': 'CLOUDSDALE', 'ponyville': 'PONYVILLE',
    'sweet_apple_acres': 'SWEET APPLE\nACRES', 'everfree': 'EVERFREE\nFOREST',
    'neighagra_falls': 'NEIGHAGRA\nFALLS', 'rainbow_falls': 'RAINBOW\nFALLS', 'manehattan': 'MANEHATTAN',
    'baltimare': 'BALTIMARE', 'griffonstone': 'GRIFFONSTONE', 'dragon_lands': 'DRAGON\nLANDS',
    'mount_aris': 'MOUNT ARIS', 'klugetown': 'KLUGETOWN', 'changeling_kingdom': 'CHANGELING\nKINGDOM',
    'appleloosa': 'APPLELOOSA', 'las_pegasus': 'LAS PEGASUS', 'vanhoover': 'VANHOOVER',
    'crystal_empire': 'CRYSTAL\nEMPIRE', 'yakyakistan': 'YAKYAKISTAN',
    'castle_two_sisters': 'CASTLE OF THE\nTWO SISTERS',
}


# =========================================================================== route timing
def _leg_len(i):
    _, pts = WM._leg_dots(i, 9.0)
    sp = np.array([WM._ov(x, y) for x, y in pts])
    return float(np.hypot(*np.diff(sp, axis=0).T).sum())


@lru_cache(maxsize=1)
def _arrivals():
    """Arrival frame of every tour stop: legs get 16th notes (4.5 f) in proportion to their length
    (min 1), 28 16ths in all, so every arrival lands on a 16th and the last one on the f-135 blip."""
    L = [_leg_len(i) for i in range(NLEG)]
    units = int(round((T_R1 - T_R0) / 4.5))
    tot = sum(L)
    alloc = [1] * NLEG
    extra = [L[i] / tot * units - 1 for i in range(NLEG)]
    for _ in range(units - NLEG):
        k = max(range(NLEG), key=lambda i: extra[i])
        alloc[k] += 1
        extra[k] -= 1
    cum = np.concatenate([[0], np.cumsum(alloc)])
    return tuple(T_R0 + int(4.5 * u) for u in cum)


def _route(f):
    """route_upto (legs drawn, float) at frame f."""
    A = _arrivals()
    if f <= A[0]:
        return 0.0
    if f >= A[-1]:
        return float(NLEG)
    for i in range(NLEG):
        if f < A[i + 1]:
            u = (f - A[i]) / (A[i + 1] - A[i])
            return i + 0.45 * u + 0.55 * smooth(u)
    return float(NLEG)


def _reached(f):
    """Number of stops reached (0..20)."""
    return sum(1 for a in _arrivals() if f >= a)


def _head(r):
    """Screen position of the route pen for route_upto r."""
    if r <= 0:
        return overview_xy(TOUR[0])
    i = min(int(r), NLEG - 1)
    t = 1.0 if r >= NLEG else r - i
    x, y = WM._bez(WM._bezier_leg(i), t)
    ox, oy = WM._ov(float(x), float(y))
    return MX + ox, MY + oy


# =========================================================================== map + dawn
def _hours(f):
    """(night-side hour, day-side hour, dawn front x)."""
    hn = keyframes(f, [(0, 0.0), (T_KEY, 0.5), (T_B5, 4.3), (T_B6, 4.9), (T_FINAL, 5.2)])
    hd = keyframes(f, [(T_KEY, 5.7), (T_FINAL, 6.4), (N, 6.9)], ease=lambda t: t)
    if f < T_KEY:
        fr = MX + MW + 20
    elif f < T_B6:
        fr = lerp(MX + MW + 4, MX + MW - 30, ease_out(seg(f, T_KEY, T_B6)))
    else:
        u = seg(f, T_B6, T_FINAL - 4)
        fr = lerp(MX + MW - 30, MX - 18, 0.3 * u + 0.7 * smooth(u))
    return hn, hd, fr


def _board(f, r):
    hn, hd, front = _hours(f)
    c = Canvas()
    if front <= MX - 16:
        draw_overview(c, f, hour=hd, found=20, route_upto=r, labels=False)
        return c, front
    draw_overview(c, f, hour=hn, found=20, route_upto=r, labels=False)
    if front >= MX + MW + 16:
        return c, front
    cd = Canvas()
    draw_overview(cd, f, hour=hd, found=20, route_upto=r, labels=False, frame=False)
    day = cd.a[MY:MY + MH, MX:MX + MW]
    reg = c.a[MY:MY + MH, MX:MX + MW]
    xs = np.arange(MX, MX + MW)[None, :]
    ys = np.arange(MY, MY + MH)[:, None]
    by = _BAY[MY:MY + MH, MX:MX + MW]
    thr = front + np.sin(ys * 0.11 + f * 0.05) * 2.0 + (by - 0.5) * 8
    m = xs >= thr
    reg[m] = day[m]
    dist = xs - thr
    k = np.clip(1 - dist / 10.0, 0, 1) * (dist >= 0)
    k = np.floor(k * 3 + by * 0.9) / 3
    add = np.array(col('#ffb070'), np.float32)
    reg[:] = np.clip(reg.astype(np.float32) + add * k[..., None] * 0.45, 0, 255).astype(np.uint8)
    return c, front


# =========================================================================== small sprites
_STAR = ['...#...',
         '..###..',
         '#######',
         '.#####.',
         '..###..',
         '.##.##.',
         '.#...#.']


@lru_cache(maxsize=None)
def _star_rgba(state):
    """7x7 star + 1px outline (9x9 RGBA). state: 'empty' | 'gold' | 'white'."""
    m = np.array([[ch == '#' for ch in row] for row in _STAR])
    big = np.zeros((9, 9), bool)
    big[1:8, 1:8] = m
    out = big.copy()
    out[1:, :] |= big[:-1, :]
    out[:-1, :] |= big[1:, :]
    out[:, 1:] |= big[:, :-1]
    out[:, :-1] |= big[:, 1:]
    rgba = np.zeros((9, 9, 4), np.uint8)
    if state == 'empty':
        rgba[out & ~big] = (*col('#4a4270'), 255)
        rgba[big] = (*col('#262246'), 255)
        return rgba
    rgba[out & ~big] = (*col('#2e1f3a'), 255)
    if state == 'white':
        rgba[big] = (255, 255, 255, 255)
        return rgba
    for y in range(9):
        for x in range(9):
            if big[y, x]:
                cc = 'butter' if y <= 3 else ('gold' if y <= 5 else 'amber')
                rgba[y, x] = (*col(cc), 255)
    rgba[3, 3] = (*col('white'), 255)
    return rgba


@lru_cache(maxsize=None)
def _icon(lid):
    from engine import worldmap_icons as WI
    return WI.icon(lid)


@lru_cache(maxsize=None)
def _postcard_bg():
    c = Canvas(58, 58)
    c.gradient(0, 0, 58, 58, ['#2c2f5a', '#4a4a80', '#7a6ab8', '#c48ad0', '#ffb0c8'], soft=0.6)
    for i in range(14):
        x, y = int(hash01(3, i, 1) * 58), int(hash01(3, i, 2) * 26)
        c.px(x, y, 'white' if i % 3 == 0 else 'baby_blue')
    c.ellipse(29, 56, 26, 4, '#5a4a7a')
    c.ellipse(29, 56, 22, 3, '#6a5a8a')
    return c


def _postcard(c, f, k_idx, t_arr):
    """Gold-framed postcard of tour stop k_idx (-1 = empty), x 252..311, y 18..77."""
    x0, y0, w, h = 252, 18, 60, 60
    c.rect(x0 - 2, y0 - 2, w + 4, h + 4, '#2e222f')
    c.rect(x0 - 1, y0 - 1, w + 2, h + 2, '#c9a24a')
    c.frame(x0, y0, w, h, '#f9c22b')
    sub = _postcard_bg().copy()
    if k_idx >= 0:
        ic = _icon(TOUR[k_idx])
        u = f - t_arr
        spr = ic.white if u < 2 else ic.rgba
        sx = 29 - ic.w // 2
        sy = 56 - ic.h + (1 if u < 2 else 0)
        sub.blit(spr, sx, max(sy, 0))
        if u < 10:
            fx.sparkle(sub, 50, 10, u, 'white', period=10, size=3)
            fx.sparkle(sub, 8, 22, u + 3, 'butter', period=10, size=2)
    else:
        text(sub, 29, 24, '?', '#7a6ab8', align='center', scale=2)
    c.blit(sub, x0 + 1, y0 + 1)
    for (gx, gy) in ((x0, y0), (x0 + w - 1, y0), (x0, y0 + h - 1), (x0 + w - 1, y0 + h - 1)):
        c.px(gx, gy, 'butter')


# =========================================================================== panels
def _filigree(c, cx, y0, y1, color, seed):
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


_GRID = [(19 + 13 * (i % 4), 52 + 13 * (i // 4)) for i in range(20)]


def _panels_journey(c, f):
    """Bars 1-3: counter + star grid (left), postcard + progress (right)."""
    A = _arrivals()
    n = _reached(f)
    r = _route(f)
    done = f >= T_COMPLETE
    # ---- left: AREAS nn/20
    text(c, LCX, 10, 'AREAS', 'lilac', font='tiny', align='center')
    s = f'{n:02d}/20'
    flash = n > 0 and f - A[n - 1] < 3
    if done and T_COMPLETE <= f < T_COMPLETE + 4:
        flash = True
    text(c, LCX, 19, s, '#45293f', scale=2, align='center')
    text(c, LCX - 1, 18, s, 'white' if flash else 'gold', scale=2, align='center')
    c.hline(LCX - 28, LCX + 28, 36, '#5a4a6a')
    for d in (-30, 30):
        c.px(LCX + d, 36, 'gold')
    # star slots
    for i, (gx, gy) in enumerate(_GRID):
        if i < n:
            u = f - A[i]
            st = 'white' if u < 2 else 'gold'
            if done and 0 <= f - T_COMPLETE - i < 2:
                st = 'white'
            c.blit(_star_rgba(st), gx - 4, gy - 4 - (1 if 2 <= u < 4 else 0))
            if u < 9:
                fx.sparkle(c, gx + 4, gy - 4, u, 'white', period=9, size=2)
            elif done:
                fx.sparkle(c, gx + 3, gy - 3, f, 'white', period=60, phase=hash01(5, i), size=1)
        else:
            c.blit(_star_rgba('empty'), gx - 4, gy - 4)
    _filigree(c, LCX, 124, 166, '#7a6a8a', 0)
    # ---- right: postcard + name + progress
    text(c, RCX, 8, 'JOURNEY', 'lilac', font='tiny', align='center')
    k = n - 1
    _postcard(c, f, k, A[k] if k >= 0 else 0)
    if k >= 0:
        nm = NAMES[TOUR[k]]
        lines = nm.split('\n')
        yy = 84 if len(lines) == 2 else 87
        for j, ln in enumerate(lines):
            text(c, RCX, yy + 7 * j, ln, 'white' if f - A[k] < 2 else 'cream', font='tiny', align='center')
    pct = 5 * n
    ps = f'{pct}%'
    text(c, RCX + 1, 102, ps, '#45293f', scale=2, align='center')
    text(c, RCX, 101, ps, 'white' if (done and f - T_COMPLETE < 4) else 'gold', scale=2, align='center')
    bx, by, bw = RCX - 27, 120, 54
    c.rect(bx - 1, by - 1, bw + 2, 6, '#2e222f')
    c.rect(bx, by, bw, 4, '#262246')
    fw = int(round(bw * r / NLEG))
    if fw > 0:
        c.rect(bx, by, fw, 4, 'hot_pink')
        c.hline(bx, bx + fw - 1, by, 'bubblegum')
        c.hline(bx, bx + fw - 1, by + 3, 'magenta')
        c.vline(bx + fw - 1, by, by + 3, 'white')
    _filigree(c, RCX, 134, 166, '#7a6a8a', 2)


def _medallion(c, cx, cy, r, rim1, rim2, fill_top, fill_bot, stud):
    c.circle(cx + 1, cy + 2, r + 2, '#120f26')
    c.circle(cx, cy, r + 2, rim2)
    c.circle(cx, cy, r + 1, rim1)
    c.circle(cx, cy, r, fill_bot)
    c.ellipse(cx, cy - r // 3, r - 2, r - 3, fill_top)
    for i in range(8):
        a = i * math.pi / 4
        c.px(int(round(cx + math.cos(a) * (r + 1))), int(round(cy + math.sin(a) * (r + 1))), stud)


@lru_cache(maxsize=None)
def _crescent(r=10):
    n = 2 * r + 3
    yy, xx = np.mgrid[0:n, 0:n] - (r + 1)
    d = np.hypot(xx, yy)
    d2 = np.hypot(xx + 5, yy - 3)
    m = (d <= r) & (d2 > r - 1)
    rgba = np.zeros((n, n, 4), np.uint8)
    lit = np.clip((xx * 0.5 - yy * 0.4) / r, -1, 1)
    for y in range(n):
        for x in range(n):
            if m[y, x]:
                edge = d[y, x] > r - 1.2 or d2[y, x] < r + 0.3
                cc = 'baby_blue' if edge else ('white' if lit[y, x] > 0.15 else 'cloud')
                rgba[y, x] = (*col(cc), 255)
    return rgba


def _panels_medal(c, f, moon_k, sun_k):
    """Bars 4-7: moon medallion (left, fading) and sun medallion (right, rising), as in the intro."""
    fil_l = mix('stone', '#5a5478', 1 - moon_k)
    _filigree(c, LCX, 16, 66, fil_l, 0)
    _filigree(c, LCX, 118, 166, fil_l, 1)
    fil_r = mix('#6a5a6a', 'gold', sun_k)
    _filigree(c, RCX, 16, 66, fil_r, 2)
    _filigree(c, RCX, 118, 166, fil_r, 0)
    # moon
    cx, cy, r = LCX, 92, 18
    dim = 1 - moon_k
    _medallion(c, cx, cy, r, mix('stone', '#5a5478', dim * 0.6), mix('cloud_dk', '#3a3258', dim * 0.6),
               mix('#2c2f62', '#3c3560', dim), mix('#1b1d45', '#2c2648', dim), 'gold')
    if moon_k > 0.02:
        fx.glow(c, cx, cy, 22, '#6c74c0', 0.22 * moon_k)
    cr = _crescent().copy()
    if dim > 0:
        rgb = cr[..., :3].astype(np.float32)
        cr[..., :3] = np.clip(rgb * (1 - dim * 0.6) + np.array(col('#6a6290')) * dim * 0.6, 0, 255)
    c.blit(cr, cx - 11, cy - 11)
    for i, (sx, sy) in enumerate(((cx - 9, cy - 8), (cx + 9, cy + 7), (cx - 6, cy + 10), (cx + 11, cy - 6))):
        if moon_k > 0.2:
            fx.sparkle(c, sx, sy, f, 'white' if i % 2 == 0 else 'baby_blue', period=34 + 7 * i, phase=i * 0.3,
                       size=1 + (i == 0))
        else:
            c.px(sx, sy, '#4a4470')
    # sun
    cx, cy, r = RCX, 92, 18
    k = sun_k
    _medallion(c, cx, cy, r, mix('#5a4a6a', 'gold', k), mix('#3a3258', 'brass', k),
               mix('#2a2550', '#ffcf9a', k), mix('#221d44', '#f6a07a', k), mix('#5a4a6a', 'butter', k))
    if k > 0.02:
        fx.glow(c, cx, cy, 30, '#ffb070', 0.30 * k)
    ang0 = f * 0.012
    for i in range(12):
        a = ang0 + i * math.pi / 6
        L = 13 if i % 2 == 0 else 10
        c.line(int(round(cx + math.cos(a) * 8)), int(round(cy + math.sin(a) * 8)),
               int(round(cx + math.cos(a) * L)), int(round(cy + math.sin(a) * L)),
               mix('#5a4a50', 'amber', k) if i % 2 == 0 else mix('#4a3f50', 'sand', k))
    c.circle(cx, cy, 7, mix('#4a3f50', 'amber', k))
    c.circle(cx, cy, 6, mix('#6a5a5a', 'gold', k))
    c.circle(cx - 1, cy - 1, 4, mix('#7a6a60', 'butter', k))
    c.px(cx - 2, cy - 3, mix('#8a7a70', 'white', k))
    if k > 0.6:
        for i, (sx, sy) in enumerate(((cx - 12, cy - 12), (cx + 13, cy + 11), (cx + 12, cy - 13))):
            fx.sparkle(c, sx, sy, f, 'butter', period=30 + 6 * i, phase=i * 0.4, size=2)
    if k > 0:
        fx.light_rect(c, RIGHT_PANEL[0] - 6, 0, 6, H, '#ff9a6a', 0.10 * k, fade='left')


def _panel_light(f):
    moon = 1 - smooth(seg(f, T_B6, T_FINAL))
    sun = 0.35 * smooth(seg(f, T_KEY, T_KEY + 24)) + 0.65 * smooth(seg(f, T_B6 - 12, T_FINAL - 12))
    return moon, sun


_PANEL_RECTS = [(LEFT_PANEL[0] + 3, LEFT_PANEL[1] + 3, LEFT_PANEL[2] - 6, LEFT_PANEL[3] - 6),
                (RIGHT_PANEL[0] + 3, RIGHT_PANEL[1] + 3, RIGHT_PANEL[2] - 6, RIGHT_PANEL[3] - 6)]
T_SWAP0, T_SWAP1 = T_KEY - 8, T_KEY + 8


def _panels(c, f):
    """Journey panels dissolve out over the last 8 frames of bar 3; the medallions dissolve in from
    the key change (never both at once)."""
    if f < T_SWAP0:
        _panels_journey(c, f)
        return
    moon_k, sun_k = _panel_light(f)
    if f >= T_SWAP1:
        _panels_medal(c, f, moon_k, sun_k)
        return
    b = c.copy()
    if f < T_KEY:
        _panels_journey(b, f)
        m = _BAY >= seg(f, T_SWAP0 - 1, T_KEY - 1)
    else:
        _panels_medal(b, f, moon_k, sun_k)
        m = _BAY < seg(f, T_KEY - 1, T_SWAP1 - 1)
    for (x, y, w, h) in _PANEL_RECTS:
        sl = (slice(y, y + h), slice(x, x + w))
        c.a[sl] = np.where(m[sl][..., None], b.a[sl], c.a[sl])
    if T_KEY <= f <= T_KEY + 8:              # flare on the sun emblem at the key change
        u = seg(f, T_KEY, T_KEY + 8)
        fx.glow(c, RCX, 92, int(14 + 22 * u), 'butter', 0.5 * (1 - u))


def _east_light(c, f):
    """Warm first light spilling over the east of the map from the key change."""
    k = 0.6 * smooth(seg(f, T_KEY, T_KEY + 30)) + 0.4 * smooth(seg(f, T_B6, T_FINAL))
    if k <= 0 or f >= T_FADE1:
        return
    fx.light_rect(c, MX + MW - 70, MY, 70, MH, '#ff9a6a', 0.12 * k, fade='left')


# =========================================================================== route fx + balloon
def _stop_fx(c, f):
    A = _arrivals()
    for k, lid in enumerate(TOUR):
        u = f - A[k]
        sx, sy = overview_xy(lid)
        if 0 <= u < 12:
            if u < 3:
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    c.px(sx + dx, sy + dy, 'white')
            rr = 2 + int(u * 0.7)
            cc = 'white' if u < 5 else ('butter' if u < 8 else 'gold')
            for dx, dy in ((rr, 0), (-rr, 0), (0, rr), (0, -rr)):
                c.px(sx + dx, sy + dy, cc)
            if u < 8:
                fx.sparkle(c, sx, sy - 1, u, 'white', period=8, size=3)
        # map COMPLETE: all stops flash in a ripple, then twinkle
        v = f - T_COMPLETE - (k % 5)
        if 0 <= v < 3:
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                c.px(sx + dx, sy + dy, 'white')
        elif T_COMPLETE <= f < T_FINAL:
            fx.sparkle(c, sx, sy - 1, f, 'white', period=48 + (k * 7) % 23, phase=hash01(11, k), size=1)


def _pen(c, f, r):
    """Bright pen head + short glitter trail while the route draws."""
    if not (T_R0 <= f < T_R1 + 2) or r <= 0:
        return
    hx, hy = _head(r)
    for j in range(1, 5):
        rr = max(0.0, r - j * 0.08)
        tx, ty = _head(rr)
        if j % 2 == 0 and (f + j) % 3:
            c.px(int(tx), int(ty) - 1, 'bubblegum')
    c.px(int(hx), int(hy), 'white')
    if (f // 2) % 2 == 0:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c.px(int(hx) + dx, int(hy) + dy, 'cotton')


def _shadow(c, x, y, w=3):
    c.blend(x - w // 2, y, w, 1, '#120c20', 0.45)


_FLIGHT = [(T_B6 + 4, (147, 68)), (380, (168, 62)), (396, (204, 64)), (410, (242, 78)), (424, (279, 98)),
           (436, (302, 76)), (448, (324, 30)), (460, (342, -14)), (472, (356, -50))]


def _flight(f):
    """Balloon basket point (Catmull-Rom through the _FLIGHT keys)."""
    ks = _FLIGHT
    if f <= ks[0][0]:
        return np.array(ks[0][1], float)
    if f >= ks[-1][0]:
        return np.array(ks[-1][1], float)
    for i in range(len(ks) - 1):
        if f < ks[i + 1][0]:
            break
    t = (f - ks[i][0]) / (ks[i + 1][0] - ks[i][0])
    p = [np.array(ks[max(0, min(len(ks) - 1, j))][1], float) for j in (i - 1, i, i + 1, i + 2)]
    t2, t3 = t * t, t * t * t
    return 0.5 * (2 * p[1] + (-p[0] + p[2]) * t + (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) * t2
                  + (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) * t3)


def _balloon(c, f, r):
    """The explorer's balloon: rides the pen (xs), parks at the castle, lifts off at bar 6."""
    if f < T_B6:
        hx, hy = _head(r)
        x, y = int(round(hx)), int(round(hy))
        _shadow(c, x + 1, y + 1)
        draw_balloon(c, x, y - 2, f, size='xs')
        return
    # departure: puff, then drift east, across the rising sun and away out of the top-right corner
    p0 = np.array(overview_xy(TOUR[-1]), float) + (0, -2)
    if f < T_B6 + 4:
        x, y = int(p0[0]), int(p0[1])
        _shadow(c, x + 1, y + 1)
        draw_balloon(c, x, y, f, size='xs', flame=True)
    else:
        q = _flight(f)
        x, y = int(round(q[0])), int(round(q[1]))
        lift = int(round(min(1.0, (f - T_B6 - 4) / 30.0) * 6))
        if lift < 6 and MX <= x < MX + MW:
            _shadow(c, x + 1 + lift, y + 1 + lift)
        size = 's' if x < RIGHT_PANEL[0] - 4 else 'm'
        draw_balloon(c, x, y + (0 if size == 's' else 4), f, size=size,
                     flame=(f - T_B6) < 40 or 0 <= (f - 412) < 14)
        if 0 <= x - (RIGHT_PANEL[0] - 4) < 3:      # size change hidden in a glint
            fx.sparkle(c, x, y - 8, 1, 'white', period=4, size=3)


def _puff(c, f):
    """Puff of smoke at lift-off (drawn under the balloon)."""
    v = f - T_B6
    if not (0 <= v < 18):
        return
    x0, y0 = overview_xy(TOUR[-1])
    for i in range(7):
        a = i * 2 * math.pi / 7 + 0.3
        d = 2 + 9 * ease_out(seg(v, 0, 16)) * (0.7 + 0.3 * hash01(9, i))
        px_, py_ = int(round(x0 + math.cos(a) * d)), int(round(y0 - 3 + math.sin(a) * d * 0.6))
        rr = 2 if v < 8 else (1 if v < 14 else 0)
        if hash01(9, i, v // 3) < seg(v, 10, 18):
            continue
        c.circle(px_, py_ + 1, rr, 'cloud_dk')
        c.circle(px_, py_, rr, 'cloud' if v > 4 else 'white')
        c.px(px_ - rr // 2, py_ - rr // 2, 'white')


# =========================================================================== stamp
STAMP_C = (160, 100)
_ANG = -5.0


@lru_cache(maxsize=None)
def _stamp_layers():
    """(rgb, level) of the rotated stamp; level 0 = none, 1 = translucent backing, 2 = ink."""
    sw, sh = 150, 42
    cv = Canvas(sw, sh, (1, 2, 3))
    key = np.array([1, 2, 3], np.uint8)
    ink, ink_dk, gold = col('hot_pink'), col('magenta'), col('gold')
    cv.round_rect(0, 0, sw, sh, ink, r=4)
    cv.round_rect(2, 2, sw - 4, sh - 4, (9, 9, 9), r=3)            # backing
    cv.round_rect(4, 4, sw - 8, sh - 8, ink, r=2)
    cv.round_rect(5, 5, sw - 10, sh - 10, (9, 9, 9), r=2)
    tw = text_width('MAP COMPLETE', 'normal', 2)
    text(cv, sw // 2 - tw // 2 + 1, 10, 'MAP COMPLETE', ink_dk, scale=2)
    text(cv, sw // 2 - tw // 2, 9, 'MAP COMPLETE', ink, scale=2)
    s2 = '20/20  ·  100%'
    tw2 = text_width(s2, 'normal')
    text(cv, sw // 2 - tw2 // 2, 28, s2, gold)
    for sx in (sw // 2 - tw2 // 2 - 12, sw // 2 + tw2 // 2 + 6):
        for dx, dy in ((2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3),
                       (3, 3), (1, 4), (3, 4)):
            cv.px(sx + dx, 28 + dy + 1, gold)
    a = cv.a
    lvl = np.where((a == key).all(-1), 0, np.where((a == 9).all(-1), 1, 2)).astype(np.uint8)
    # worn rubber: speckle holes in the ink
    yy, xx = np.mgrid[0:sh, 0:sw]
    hol = np.vectorize(lambda x, y: hash01(71, x, y))(xx, yy) < 0.05
    lvl[(lvl == 2) & hol & ((a == np.array(ink, np.uint8)).all(-1))] = 1
    # dark rim around the whole stamp (reads on any map colour)
    pad = 12
    sw2, sh2 = sw + 2, sh + 2
    big = np.zeros((sh2, sw2), np.uint8)
    big[1:-1, 1:-1] = lvl
    a2 = np.zeros((sh2, sw2, 3), np.uint8)
    a2[1:-1, 1:-1] = a
    on = big > 0
    dil = on.copy()
    dil[1:] |= on[:-1]
    dil[:-1] |= on[1:]
    dil[:, 1:] |= on[:, :-1]
    dil[:, :-1] |= on[:, 1:]
    rim = dil & ~on
    big[rim] = 2
    a2[rim] = col('#3a1030')
    lvl, a, sw, sh = big, a2, sw2, sh2
    # rotate (nearest) into a padded buffer
    ow, oh = sw + 2 * pad, sh + 2 * pad
    oy, ox = np.mgrid[0:oh, 0:ow]
    ca, sa = math.cos(math.radians(_ANG)), math.sin(math.radians(_ANG))
    dx, dy = ox - ow / 2 + 0.5, oy - oh / 2 + 0.5
    srcx = np.floor(ca * dx + sa * dy + sw / 2).astype(int)
    srcy = np.floor(-sa * dx + ca * dy + sh / 2).astype(int)
    ok = (srcx >= 0) & (srcx < sw) & (srcy >= 0) & (srcy < sh)
    rgb = np.zeros((oh, ow, 3), np.uint8)
    lv = np.zeros((oh, ow), np.uint8)
    rgb[ok] = a[srcy[ok], srcx[ok]]
    lv[ok] = lvl[srcy[ok], srcx[ok]]
    return rgb, lv


def _stamp(c, f):
    if f < T_COMPLETE - 6 or f >= T_KEY:
        return
    rgb, lv = _stamp_layers()
    oh, ow = lv.shape
    x0, y0 = STAMP_C[0] - ow // 2, STAMP_C[1] - oh // 2
    reg = c.a[y0:y0 + oh, x0:x0 + ow]
    if f < T_COMPLETE:                     # the stamp comes down: its shadow slides in under it
        u = seg(f, T_COMPLETE - 6, T_COMPLETE - 1)
        d = int(round((1 - u) * 6)) + 1
        m = np.zeros_like(lv, bool)
        m[d:, d:] = lv[:-d, :-d] > 0
        m &= _BAY[y0:y0 + oh, x0:x0 + ow] < 0.25 + 0.5 * u
        reg[m] = (reg[m].astype(np.uint16) * 2 // 5).astype(np.uint8)
        return
    vis = np.ones_like(lv, bool)
    if f >= T_KEY - 10:                    # lifts away: dither dissolve
        vis = _BAY[y0:y0 + oh, x0:x0 + ow] >= seg(f, T_KEY - 10, T_KEY - 1)
    back = (lv == 1) & vis
    r2 = reg.astype(np.float32)
    r2[back] = r2[back] * 0.35 + np.array(col('#2a1430'), np.float32) * 0.65
    reg[:] = r2.astype(np.uint8)
    inkm = (lv == 2) & vis
    if f - T_COMPLETE < 2:
        reg[inkm] = 255
    else:
        reg[inkm] = rgb[inkm]


def _burst(c, f):
    """Sparkle ring + confetti when the stamp lands."""
    u = f - T_COMPLETE
    if not (0 <= u < 40):
        return
    cx, cy = STAMP_C
    for i in range(18):
        a = i * 2 * math.pi / 18 + 0.17
        d = 30 + 50 * ease_out(seg(u, 0, 18))
        x = int(cx + math.cos(a) * d * 1.25)
        y = int(cy + math.sin(a) * d * 0.55)
        if u < 18:
            fx.sparkle(c, x, y, u, ('white', 'butter', 'bubblegum')[i % 3], period=18, size=3)
    fx.confetti(c, f, T_COMPLETE, cx, cy - 10, n=36, seed=21,
                colors=('gold', 'hot_pink', 'butter', 'lavender', 'baby_blue'))


# =========================================================================== cards (top plate)
def _plate_dims(lines_w, h):
    w = lines_w + 2 * 8 + 12
    return w, h


def _plate_t(f, t_in, t_out, unroll=6, roll=4):
    """Cartouche unroll amount (0..1) and whether its content shows."""
    if f < t_in - unroll or f >= t_out + roll:
        return 0.0, False
    if f < t_in:
        return ease_out_back(seg(f, t_in - unroll, t_in), 1.2), False
    if f >= t_out:
        return 1 - smooth(seg(f, t_out, t_out + roll)), False
    return 1.0, True


def _card_title(c, f):
    t_in, t_out = T_COMPLETE + 8, T_KEY - 4
    tw, th = title_size('EQUESTRIA')
    sub = 'A PIXEL WORLD ATLAS'
    sw = text_width(sub, 'normal', spacing=1) + 16
    w = max(tw, sw) + 16 + 12
    h = 4 + th + 11 + 3
    t, show = _plate_t(f, t_in, t_out)
    if t <= 0:
        return
    draw_cartouche(c, 160, 3, w, h, t)
    if not show:
        return
    sh = None
    if f >= t_in + 34:
        uu = (f - t_in - 34) / 16.0
        sh = uu if uu < 1 else None
    draw_title(c, 160, 8, 'EQUESTRIA', f=f, t0=t_in, step=3, style='silver', shine=sh)
    draw_subtitle(c, 160, 8 + th, sub, f, t_in + 8, rate=1)


def _card_lines(c, f, t_in, t_out, lines, y=3):
    """Plate with typed lines: [(text, font, color, shade)]."""
    widths = [text_width(s, fn, spacing=1 if fn == 'normal' else None) for s, fn, _, _ in lines]
    lh = [10 if fn == 'normal' else 8 for _, fn, _, _ in lines]
    w = max(wd + (36 if fn == 'normal' else 18) for wd, (_, fn, _, _) in zip(widths, lines)) + 12
    h = 6 + sum(lh) + 2
    t, show = _plate_t(f, t_in, t_out)
    if t <= 0:
        return
    draw_cartouche(c, 160, y, w, h, t)
    if not show:
        return
    yy = y + 5
    t0 = t_in
    for (s, fn, cc, shc), lhh in zip(lines, lh):
        if fn == 'normal':
            draw_subtitle(c, 160, yy, s, f, t0, rate=1, color=cc, shade_color=shc)
        else:
            n = int(clamp((f - t0) + 1, 0, len(s)))
            if n > 0:
                tw = text_width(s, 'tiny')
                text(c, 160 - tw // 2, yy + 1, s[:n], cc, font='tiny')
        t0 += len(s) // 2 + 4
        yy += lhh
    if f >= t0 + 6:
        for i, (gx, gy) in enumerate(((160 - w // 2 + 8, y + 3), (160 + w // 2 - 9, y + h - 4))):
            fx.sparkle(c, gx, gy, f + i * 17, 'white', period=48, size=2)


def _cards(c, f):
    _card_title(c, f)
    _card_lines(c, f, T_KEY + 6, T_B5 - 4,
                [('MAP, ART & MUSIC', 'normal', 'royal', 'lilac'),
                 ('MADE WITH CODE', 'normal', 'royal', 'lilac')])
    _card_lines(c, f, T_B5 + 6, T_B6 - 10,
                [('FAN-MADE TRIBUTE', 'normal', 'royal', 'lilac'),
                 ('FIRST EDITION · LANDSCAPES ONLY', 'tiny', 'plum', None)])
    if T_B5 + 10 <= f < T_B6 - 8:          # small, respectful trademark line at the map's foot
        s = 'MY LITTLE PONY IS A TRADEMARK OF HASBRO'
        tw = text_width(s, 'tiny')
        text(c, 160 - tw // 2, MY + MH - 8, s, 'cloud_dk', font='tiny', outline='#2e222f')


def _the_end(c, f):
    t0 = T_FINAL
    if f < t0 - 8:
        return
    tw, th = title_size('THE END')
    w = tw + 2 * 14 + 12
    h = 4 + th + 3
    k = seg(f, t0 - 8, t0)
    y = 74
    draw_cartouche(c, 160, y, w, h, ease_out_back(k, 1.2) if k < 1 else 1.0)
    if f < t0:
        return
    sh = None
    if f >= t0 + 24:
        uu = (f - t0 - 24) / 16.0
        sh = uu if uu < 1 else None
    draw_title(c, 160, y + 4, 'THE END', f=f, t0=t0, step=3, style='gold', shine=sh)
    if f >= t0 + 20:
        for i, (gx, gy) in enumerate(((160 - w // 2 + 8, y + 3), (160 + w // 2 - 9, y + h - 4),
                                      (160 + w // 2 - 9, y + 3), (160 - w // 2 + 8, y + h - 4))):
            fx.sparkle(c, gx, gy, f + i * 13, 'white', period=40, size=2)


# =========================================================================== final fade
_KEY = (1, 2, 3)


def _end_mask(f):
    """Mask of the THE END plate pixels (rolls, plate, letters; not its soft shadow)."""
    L = Canvas(W, H, _KEY)
    _the_end(L, f)
    key = np.array(_KEY, np.uint8)
    shd = np.clip(key + (np.array(col('#120c20'), np.float32) - key) * 0.45 + 0.5, 0, 255).astype(np.uint8)
    return (L.a != key).any(-1) & (L.a != shd).any(-1)


def _dim(a, k, bay):
    """Dithered darkening of an RGB block toward ink (k 0..1, 4 levels + ordered dither)."""
    if k <= 0:
        return
    if k >= 1:
        a[:] = col('ink')
        return
    v = k * 4
    base = int(v)
    lvl = (base + (bay < (v - base))).astype(np.float32) / 4.0
    ink = np.array(col('ink'), np.float32)
    af = a.astype(np.float32)
    a[:] = np.clip(af + (ink - af) * lvl[..., None], 0, 255).astype(np.uint8)


def _fade(c, f):
    """The world fades first; THE END lingers on the decaying chord, then goes too."""
    k_world = seg(f, T_FADE0, T_FADE1 - 12)
    k_end = seg(f, T_FADE0 + 14, T_FADE1)
    if k_world <= 0:
        return
    m = _end_mask(f)
    keep = c.a.copy()
    _dim(c.a, k_world, _BAY)
    _dim(keep, k_end, _BAY)
    c.a[m] = keep[m]


def _shake(c, f):
    u = f - T_COMPLETE
    if not (0 <= u < 9):
        return
    amp = [3, 3, 2, 2, 1, 1, 1, 0, 0][u]
    dx, dy = fx.shake(f, amp, seed=4)
    if dx == 0 and dy == 0:
        return
    a = np.roll(np.roll(c.a, dy, 0), dx, 1)
    if dy > 0:
        a[:dy] = col(BG)
    elif dy < 0:
        a[dy:] = col(BG)
    if dx > 0:
        a[:, :dx] = col(BG)
    elif dx < 0:
        a[:, dx:] = col(BG)
    c.a[:] = a


# =========================================================================== render
def render(f):
    if f >= T_FADE1:
        return Canvas(W, H, 'ink')
    r = _route(f)
    c, front = _board(f, r)
    _panels(c, f)
    _east_light(c, f)
    _stop_fx(c, f)
    _pen(c, f, r)
    _stamp(c, f)
    _puff(c, f)
    _balloon(c, f, r)
    _burst(c, f)
    _cards(c, f)
    _the_end(c, f)
    _shake(c, f)
    _fade(c, f)
    return c


SFX = [
    (T_COMPLETE + 3, 'sparkle', 0.35),
    (T_COMPLETE + 8, 'shimmer', 0.22),
    (T_KEY, 'whoosh', 0.18),
    (T_B5, 'whoosh', 0.15),
    (T_B6, 'puff', 0.45),
    (T_B6 + 6, 'wind', 0.18),
    (T_B6 + 44, 'whoosh', 0.28),
]
