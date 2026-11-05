"""Game-style HUD drawn over story segments (configure it in engine/timeline.py).

Per segment: clock=('HH:MM','HH:MM') | None, quest=(title, subtitle) | None,
meter=[(t 0..1, level 0..5), ...] | None, hud=False to switch it all off.
Labels come from timeline.HUD (banner/toast text). Restyle freely per project.

  top-left : clock panel (sun / sunset / moon icon + HH:MM, colon blinks on the beat)
  top-right: cold-brew meter (iced glass icon + 5 pips)
  banner   : 'NEW QUEST' ribbon sliding in under the clock at the start of a segment
  toast    : small 'QUEST COMPLETE' stamp near the end of a segment

HUD-safe area for scenes: keep key action out of y < 24 at the left (x < 120)
and right (x > 250); the banner covers x 4..~170, y 20..44 during local frames
10..110.
"""
import math

from .core import (W, H, col, mix, seg, ease_out_back, ease_in, keyframes, beat_phase,
                   FRAMES_PER_BEAT)
from .font import text, text_width
from .sprite import Sprite
from . import timeline as TL

_SUN = Sprite.from_ascii('''
    ...y...
    .y.y.y.
    ..ggg..
    yggGggy
    ..ggg..
    .y.y.y.
    ...y...
''', {'y': 'gold', 'g': 'sand', 'G': 'butter'})

_SUNSET = Sprite.from_ascii('''
    .......
    ..ooo..
    .oaaao.
    oaaaaao
    rrrrrrr
    .......
    .r.r.r.
''', {'o': 'orange', 'a': 'amber', 'r': 'coral'})

_MOON = Sprite.from_ascii('''
    ..mmm..
    .mmw...
    mmw....
    mmw....
    mmw....
    .mmw...
    ..mmm..
''', {'m': 'butter', 'w': 'sand'})

_GLASS = Sprite.from_ascii('''
    ....s..
    ....s..
    o..o.oo
    o.....o
    ob.ibbo
    obbbbbo
    obibbbo
    .obbbo.
    ..ooo..
''', {'o': 'glass', 's': 'white', 'b': 'brew_mid', 'i': 'glass_hi'})

_COMPASS = Sprite.from_ascii('''
    ...r...
    ..rRr..
    .s.r.s.
    sSSwSSs
    .s.b.s.
    ..bBb..
    ...b...
''', {'r': 'red', 'R': 'coral', 's': 'stone', 'S': 'mist', 'w': 'white', 'b': 'blue', 'B': 'sky'})


def draw_counter(c, s, f):
    """Areas-discovered counter (top-right): compass + 'NN/TT'. Segment key found=(n_before, total);
    the count ticks up when the 'AREA DISCOVERED' toast appears."""
    fd = s.get('found')
    if not fd:
        return
    n0, tot = fd
    t_up = s['frames'] - 58
    n = n0 + 1 if f >= t_up else n0
    w = 45
    x, y = W - 4 - w, 4
    _panel(c, x, y, w, 13)
    c.blit(_COMPASS, x + 3, y + 3)
    lab = f'{n:02d}/{tot:02d}'
    just = t_up <= f < t_up + 12
    text(c, x + 13, y + 3, lab, 'butter' if just else 'white')
    if just:
        ph = (f - t_up) / 12
        r = 1 + int(3 * math.sin(ph * math.pi))
        sx, sy = x + 13 + text_width(f'{n:02d}') // 2, y + 6
        for dx, dy in ((r + 4, 0), (-r - 4, 0), (0, r + 2), (0, -r - 2)):
            c.px(sx + dx, sy + dy, 'butter')


def _label(k):
    return getattr(TL, 'HUD', {}).get(k, {'banner': 'NEW QUEST', 'toast': 'QUEST COMPLETE'}[k])


PANEL = 'ink'
PANEL_EDGE = 'shadow'


def _panel(c, x, y, w, h):
    c.round_rect(x, y, w, h, PANEL_EDGE, r=1)
    c.rect(x + 1, y + 1, w - 2, h - 2, PANEL)
    c.blend(x + 1, y + 1, w - 2, 1, 'white', 0.08)


def draw_clock(c, s, f):
    m = TL.clock_minutes(s, f)
    if m is None:
        return
    hh, mm = (m // 60) % 24, m % 60
    x, y = 4, 4
    _panel(c, x, y, 45, 13)
    icon = _MOON if hh >= 20 or hh < 5 else _SUNSET if hh >= 17 else _SUN
    c.blit(icon, x + 3, y + 3)
    colon_on = beat_phase(f + s['start']) < 0.5
    tx = x + 12
    tx += text(c, tx, y + 3, f'{hh:02d}', 'white')
    tx += 1
    text(c, tx, y + 3, ':', 'white' if colon_on else 'shadow')
    tx += 2
    text(c, tx, y + 3, f'{mm:02d}', 'white')


def draw_coffee(c, s, f):
    keys = s.get('meter') or s.get('coffee')
    if not keys:
        return
    t = f / max(1, s['frames'] - 1)
    lvl = keyframes(t, keys)
    w = 49
    x, y = W - 4 - w, 4
    _panel(c, x, y, w, 13)
    c.blit(_GLASS, x + 3, y + 2)
    for i in range(5):
        px = x + 13 + i * 7
        c.rect(px, y + 4, 6, 5, 'night')
        fill = max(0.0, min(1.0, lvl - i))
        if fill >= 0.99:
            c.rect(px, y + 4, 6, 5, 'brew_mid')
            c.rect(px, y + 4, 6, 1, 'latte')
            c.px(px + 1, y + 5, 'crema')
        elif fill > 0:
            fh = max(1, int(round(5 * fill)))
            c.rect(px, y + 9 - fh, 6, fh, 'brew_mid')
    # sparkle on the newest pip while filling
    prev = keyframes(max(0, t - 6 / max(1, s['frames'])), keys)
    if lvl > prev + 0.01:
        i = min(4, int(lvl))
        sx, sy = x + 13 + i * 7 + 3, y + 3
        ph = (f % 8) / 8
        r = 1 + int(2 * math.sin(ph * math.pi))
        for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
            c.px(sx + dx, sy + dy, 'butter')
        c.px(sx, sy, 'white')


def draw_banner(c, s, f):
    q = s.get('quest')
    if not q:
        return
    t_in, t_out = 10, 104
    if f < t_in or f > t_out + 10:
        return
    title, sub = q
    w = max(text_width(title), text_width(_label('banner'), 'tiny') + text_width(sub, 'tiny') + 6) + 12
    k = ease_out_back(seg(f, t_in, t_in + 10)) if f < t_out else 1 - ease_in(seg(f, t_out, t_out + 10))
    x = int(-w + (w + 4) * k)
    y = 21
    h = 23
    c.round_rect(x, y, w, h, 'gold', r=1)
    c.rect(x + 1, y + 1, w - 2, h - 2, PANEL)
    c.rect(x + 1, y + 1, 3, h - 2, 'gold')
    tx = x + 7
    text(c, tx, y + 3, _label('banner'), 'gold', font='tiny')
    text(c, tx + text_width(_label('banner'), 'tiny') + 5, y + 3, sub, 'stone', font='tiny')
    text(c, tx, y + 11, title, 'white', shadow='night')


def draw_toast(c, s, f):
    if not s.get('quest'):
        return
    t0 = s['frames'] - 58
    t1 = s['frames'] - 14
    if not (t0 <= f <= t1):
        return
    k = ease_out_back(seg(f, t0, t0 + 8))
    msg = _label('toast')
    w = text_width(msg, 'tiny') + 16
    h = 11
    x = W // 2 - w // 2
    y = int(-h + (h + 5) * k)
    c.round_rect(x, y, w, h, 'mint', r=1)
    c.rect(x + 1, y + 1, w - 2, h - 2, PANEL)
    # check mark
    cx, cy = x + 4, y + 5
    for dx, dy in ((0, 0), (1, 1), (2, 0), (3, -1), (4, -2)):
        c.px(cx + dx, cy + dy, 'leaf')
    text(c, x + 11, y + 3, msg, 'white', font='tiny')


def draw(c, s, f):
    """Draw the full HUD for segment `s` at local frame `f`."""
    if s.get('hud') is False or not (s.get('clock') or s.get('quest') or s.get('meter') or s.get('coffee')
                                     or s.get('found')):
        return
    draw_clock(c, s, f)
    draw_coffee(c, s, f)
    draw_counter(c, s, f)
    draw_banner(c, s, f)
    draw_toast(c, s, f)
