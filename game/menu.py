"""Title screen, pause menu, settings and the controls page.

TitleMode (a game mode, like MapMode): the parchment overview map unrolls inside its gold frame while
the explorer's balloon rises in front of it and the title card pops in; the menu plates slide into the
right-hand panel: CONTINUE (when there is a save) / NEW GAME / SETTINGS / QUIT. NEW GAME over a save
asks once more (the old save is kept as .old). The left panel shows the saved progress, a caption
under each counter (for a new player: the journey ahead, greyed); a ring marks where the balloon is
(a pin chip on the title card's edge when the card hides the spot).

PauseMenu (an overlay, like the journal): ESC on the world map or a gamepad Start. The frozen, dimmed
view stays behind a plum panel: RESUME, MUSIC and SOUND volume, WINDOW scale, FULLSCREEN, CONTROLS
(a page of keyboard + gamepad bindings), QUIT (progress is saved).

Settings live in game.settings (game/settings.py); the pygame front end applies them. Everything here
draws into the layered Frame (game/compose.py) and stays pygame-free.
"""
import math

from engine import W, H, Canvas, Sprite, col, seg, clamp, ease_in_out, ease_out, ease_out_back, mix, text, text_width
from engine import loc_worldmap as LW
from engine import timeline as TL
from engine.title import draw_title_card

from . import ui
from .compose import Frame
from .settings import VOL_STEPS

ROW_H = 13
GOLD_TXT = '#fff6c0'
DARK_RIM = '#6e2727'
PIP_OFF = '#2a1a30'


class Repeat:
    """Held arrows act on the first step they are down, then repeat every 4 steps after a short wait."""

    def __init__(self):
        self.held = {}

    def __call__(self, inp, t):
        out = {}
        for d in ('up', 'down', 'left', 'right'):
            if getattr(inp, d):
                t0 = self.held.setdefault(d, t)
                a = t - t0
                out[d] = a == 0 or (a >= 12 and (a - 12) % 4 == 0)
            else:
                self.held.pop(d, None)
                out[d] = False
        return out


# ------------------------------------------------------------------ drawing helpers
_LIGHT = {}


def micon(c, name, x, y, lit=False):
    """An interaction icon for the dark plum panels: its ink outline turned light so it reads."""
    key = (name, lit)
    if key not in _LIGHT:
        _LIGHT[key] = ui.ICONS[name].recolor({col(ui.INK): col('butter' if lit else 'tan')})
    _LIGHT[key].draw(c, x, y)


def keyboard(c, x, y):
    """A tiny keyboard glyph (13x8): the keyboard column head of the controls page."""
    c.round_rect(x, y, 13, 8, ui.INK, r=1)
    c.rect(x + 1, y + 1, 11, 6, 'cloud_dk')
    for r in range(2):
        for i in range(5):
            c.px(x + 2 + i * 2, y + 2 + r * 2, 'white')
    c.hline(x + 4, x + 8, y + 6, 'white')
    return 13

def panel(c, x, y, w, h):
    """The plum plate with the gold rim (the map's name-plate style), any size."""
    c.round_rect(x - 1, y - 1, w + 2, h + 2, DARK_RIM, r=3)
    c.round_rect(x, y, w, h, ui.RIM, r=3)
    c.round_rect(x + 1, y + 1, w - 2, h - 2, ui.PLATE, r=2)
    c.hline(x + 3, x + w - 4, y + 2, ui.PLATE_HI)
    for cx, cy in ((x + 2, y + 2), (x + w - 3, y + 2), (x + 2, y + h - 3), (x + w - 3, y + h - 3)):
        c.px(cx, cy, 'butter')


def _arrow(c, x, y, d, color):
    """A 3x5 arrow head pointing left (d<0) or right (d>0) with its tip column at x."""
    for i in range(3):
        xx = x - d * i
        c.vline(xx, y + 2 - i, y + 2 + i, color)


def _cursor(c, x, y, t):
    """The bobbing gold pointer left of the selected row (tip at x)."""
    dx = 1 if (t // 8) % 2 else 0
    _arrow(c, x + dx, y, 1, 'gold')
    c.px(x + dx - 2, y + 2, 'butter')


def slider(c, x_right, y, v, lit):
    """VOL_STEPS pips (3x5, 1 apart) ending at x_right; filled up to v."""
    w = VOL_STEPS * 4 - 1
    x0 = x_right - w
    for i in range(VOL_STEPS):
        on = i < v
        cc = ('butter' if i == v - 1 and lit else 'gold') if on else PIP_OFF
        c.rect(x0 + i * 4, y + 1, 3, 5, cc)
        if on:
            c.px(x0 + i * 4, y + 1, 'butter' if lit else 'amber')
    if v == 0:                                          # muted: a little cross
        c.line(x0 - 8, y + 1, x0 - 4, y + 5, 'red')
        c.line(x0 - 8, y + 5, x0 - 4, y + 1, 'red')
    return x0


def toggle(c, x_right, y, on):
    """A 13x7 switch: knob left (off, stone) or right (on, leaf)."""
    x = x_right - 13
    c.round_rect(x, y, 13, 7, ui.INK, r=2)
    c.rect(x + 1, y + 1, 11, 5, 'leaf' if on else 'stone')
    c.hline(x + 2, x + 10, y + 1, 'pale_leaf' if on else 'cloud_dk')
    kx = x + 7 if on else x + 1
    c.rect(kx, y, 5, 7, ui.INK)
    c.rect(kx + 1, y + 1, 3, 5, 'white' if on else 'cloud')
    return x


# ------------------------------------------------------------------ settings rows
SETTINGS_ROWS = [('music', 'note', 'MUSIC', 'slider'), ('sound', 'speaker', 'SOUND', 'slider'),
                 ('scale', 'window', 'WINDOW', 'scale'), ('fullscreen', 'fullscreen', 'FULLSCREEN', 'toggle'),
                 ('tips', 'book', 'TIPS', 'toggle'), ('controls', 'pad', 'CONTROLS', 'page')]


class RowMenu:
    """A vertical list of rows: buttons (SPACE), sliders / choices (LEFT / RIGHT), toggles, pages.
    step() returns the key of a button or page that was chosen, 'back' for ESC, else None."""

    def __init__(self, game, rows):
        self.g = game
        self.rows = rows
        self.sel = 0
        self.rep = Repeat()
        self.t_sel = game.t
        self.t_val = {}                 # row key -> step its value last changed (a little flash)

    def step(self, inp):
        g, s = self.g, self.g.settings
        go = self.rep(inp, g.t)
        if inp.back or getattr(inp, 'pause', False):
            return 'back'
        if go['up'] or go['down']:
            self.sel = (self.sel + (1 if go['down'] else -1)) % len(self.rows)
            self.t_sel = g.t
            g.sfx('tick', 0.25)
        key, _icon, _label, kind = self.rows[self.sel]
        d = (1 if go['right'] else 0) - (1 if go['left'] else 0)
        if kind == 'slider' and d:
            v = int(clamp(getattr(s, key) + d, 0, VOL_STEPS))
            if v != getattr(s, key):
                s.set(key, v)
                self.t_val[key] = g.t
                g.sfx('coin' if key == 'sound' else 'tick', 0.45 if key == 'sound' else 0.3)
        elif kind == 'scale' and d:
            opts = s.scales()
            i = opts.index(s.scale) if s.scale in opts else 0
            j = int(clamp(i + d, 0, len(opts) - 1))
            if j != i:
                s.set('scale', opts[j])
                self.t_val[key] = g.t
                g.sfx('tick', 0.3)
        elif kind == 'toggle' and (d or inp.action):
            s.set(key, not getattr(s, key))
            self.t_val[key] = g.t
            g.sfx('pop', 0.35, 0 if getattr(s, key) else 1)
        elif kind in ('button', 'page') and inp.action:
            g.sfx('pop', 0.35, 0)
            return key
        return None

    def draw(self, c, x, y, w, t):
        """Rows from (x, y), w wide."""
        s = self.g.settings
        for i, (key, icon, label, kind) in enumerate(self.rows):
            ry = y + i * ROW_H
            lit = i == self.sel
            if lit:
                c.round_rect(x + 2, ry, w - 4, ROW_H - 1, ui.PLATE_HI, r=1)
                c.hline(x + 4, x + w - 5, ry + ROW_H - 2, '#4a2e50')
                _cursor(c, x + 6, ry + 3, t)
            micon(c, icon, x + 11, ry + 1, lit)
            text(c, x + 23, ry + 3, label, GOLD_TXT if lit else 'cream', font='normal',
                 shadow=DARK_RIM if lit else None)
            xr = x + w - 8
            flash = t - self.t_val.get(key, -99) < 6
            if kind == 'slider':
                x0 = slider(c, xr - (6 if lit else 0), ry + 2, getattr(s, key), lit or flash)
                if lit:
                    _arrow(c, x0 - 4 - (10 if getattr(s, key) == 0 else 0), ry + 3, -1, 'butter')
                    _arrow(c, xr, ry + 3, 1, 'butter')
            elif kind == 'scale':
                v = s.scale
                label_v = 'FIT' if v == 0 else f'x{v}'
                tw = text_width(label_v, 'normal')
                tx = xr - 6 - tw if lit else xr - tw
                text(c, tx, ry + 3, label_v, 'butter' if lit or flash else 'cream', font='normal')
                if lit:
                    _arrow(c, tx - 4, ry + 4, -1, 'butter')
                    _arrow(c, xr, ry + 4, 1, 'butter')
            elif kind == 'toggle':
                toggle(c, xr, ry + 3, getattr(s, key))
            elif kind == 'page':
                _arrow(c, xr - (1 if lit and (t // 8) % 2 else 0), ry + 4, 1, 'butter' if lit else 'tan')

    def height(self):
        return len(self.rows) * ROW_H


# ------------------------------------------------------------------ the controls page
def _face(c, x, y, letter, fill):
    """A round gamepad face button (9x9) with its letter."""
    c.circle(x + 4, y + 4, 4, ui.INK)
    c.circle(x + 4, y + 4, 3, fill)
    c.px(x + 2, y + 2, mix(fill, 'white', 0.5))
    text(c, x + 3, y + 2, letter, 'white', font='tiny')
    return 9


def _pill(c, x, y, label):
    """A shoulder / Start / Back button: a dark pill with tiny caps."""
    w = text_width(label, 'tiny') + 6
    c.round_rect(x, y, w, 9, ui.INK, r=2)
    c.round_rect(x + 1, y + 1, w - 2, 7, 'steel', r=2)
    c.hline(x + 2, x + w - 3, y + 1, 'stone')
    text(c, x + 3, y + 2, label, 'white', font='tiny')
    return w


def _stick(c, x, y):
    c.circle(x + 4, y + 4, 4, ui.INK)
    c.circle(x + 4, y + 4, 3, 'steel')
    c.circle(x + 4, y + 4, 1, 'stone')
    c.px(x + 3, y + 3, 'cloud')
    return 9


def _dpad(c, x, y):
    c.rect(x + 3, y, 3, 9, ui.INK)
    c.rect(x, y + 3, 9, 3, ui.INK)
    c.rect(x + 4, y + 1, 1, 7, 'steel')
    c.rect(x + 1, y + 4, 7, 1, 'steel')
    return 9


_FACE = {'A': 'green', 'B': 'red', 'X': 'royal', 'Y': 'amber'}
# (icon, keyboard caps, gamepad glyphs): icons are names in ui.ICONS or ui icon functions
CONTROLS = [
    (ui.icon_balloon, ['left', 'up', 'down', 'right', '/', 'W', 'A', 'S', 'D'], ['stick', 'dpad']),
    (ui.icon_fast, ['SHIFT'], ['LB', 'RB']),
    (ui.icon_land, ['SPACE', 'ENTER'], ['A']),
    (ui.icon_takeoff, ['ESC'], ['B']),
    ('book', ['J'], ['X']),
    (ui.icon_map, ['M'], ['Y']),
    ('camera', ['P'], ['BACK']),
    ('pause', ['ESC'], ['START']),
    ('fullscreen', ['F'], []),
    ('window', ['H'], []),                  # hide / show the HUD (clean screenshots)
]
CTRL_W, CTRL_H = 216, 152


def draw_controls(c, t, cx=W // 2, y=14):
    """The controls page: a plum panel, one row per action: its icon, the keys, the gamepad buttons."""
    x = cx - CTRL_W // 2
    panel(c, x, y, CTRL_W, CTRL_H)
    hx = x + 30
    micon(c, 'pad', x + CTRL_W - 52, y + 4)                 # column heads: a keyboard / a gamepad
    keyboard(c, hx, y + 5)
    for i, (icon, keys, pad) in enumerate(CONTROLS):
        ry = y + 18 + i * 13
        if i % 2 == 0:
            c.rect(x + 4, ry - 2, CTRL_W - 8, 13, '#432a4a')
        if callable(icon):
            icon(c, x + 8, ry)
        else:
            micon(c, icon, x + 8, ry)
        kx = hx
        for k in keys:
            if k == '/':
                c.line(kx + 1, ry + 8, kx + 3, ry, 'tan')
                kx += 6
                continue
            kx += ui.keycap(c, kx, ry, k, lit=False) + 1
        px = x + CTRL_W - 52
        for p in pad:
            if p in _FACE:
                px += _face(c, px, ry, p, _FACE[p]) + 2
            elif p == 'stick':
                px += _stick(c, px, ry) + 2
            elif p == 'dpad':
                px += _dpad(c, px, ry) + 2
            else:
                px += _pill(c, px, ry, p) + 2
    c.vline(x + CTRL_W - 57, y + 16, y + CTRL_H - 6, '#4a2e50')


# ------------------------------------------------------------------ pause menu
PAUSE_ROWS = [('resume', 'play', 'RESUME', 'button')] + SETTINGS_ROWS + [('quit', 'door', 'QUIT', 'button')]
PAUSE_W = 168


class PauseMenu:
    """Overlay over the frozen, dimmed view (ESC on the map / gamepad Start)."""

    def __init__(self, game, frozen):
        self.g = game
        self.frozen = frozen            # the Frame on screen when the game paused
        self.t0 = game.t
        self.rows = RowMenu(game, PAUSE_ROWS)
        self.page = 'main'

    def step(self, inp):
        """Returns False when the menu closes."""
        g = self.g
        if self.page == 'controls':
            if inp.back or inp.action or inp.pause or inp.left:
                self.page = 'main'
                g.sfx('pop', 0.3, 1)
            return True
        r = self.rows.step(inp)
        if r in ('back', 'resume'):
            g.settings.save()
            return False
        if r == 'controls':
            self.page = 'controls'
        elif r == 'quit':
            g.settings.save()
            g.quit = True
        return True

    def render(self, t):
        src = self.frozen
        fr = Frame(src.world, src.origin, src.cam, src.focus)
        fr.sprites = list(src.sprites)
        fr.snap_y = getattr(src, 'snap_y', False)       # (else the frozen view slips a sub-pixel)
        fr.dim = True
        c = fr.hud
        if self.page == 'controls':
            draw_controls(c, t)
            ui.hint_row(c, [(['ESC'], _icon_back)], right=W - 8, y=H - 11)
            return fr
        h = self.rows.height() + 24
        x, y = (W - PAUSE_W) // 2, (H - h) // 2 + 4
        k = ease_out_back(seg(t, self.t0, self.t0 + 6))
        if k < 0.999:                                   # the panel pops open
            cw, ch = max(6, int(PAUSE_W * k)), max(6, int(h * k))
            panel(c, W // 2 - cw // 2, y + h // 2 - ch // 2, cw, ch)
            return fr
        panel(c, x, y, PAUSE_W, h)
        ui.plate(c, W // 2, y - 7, 'PAUSE')
        self.rows.draw(c, x + 2, y + 14, PAUSE_W - 4, t)
        ui.hint_row(c, [(['up', 'down'], None), (['left', 'right'], _icon_slider), (['SPACE'], None),
                        (['ESC'], _icon_play)], y=H - 13)
        return fr


def _icon_slider(c, x, y):
    if c is None:
        return 11
    for i in range(3):
        c.rect(x + i * 4, y + 2, 3, 5, 'gold' if i < 2 else PIP_OFF)
    return 11


def _icon_play(c, x, y):
    if c is None:
        return 9
    ui.ICONS['play'].draw(c, x, y)
    return 9


# ------------------------------------------------------------------ title screen
CARD_T0 = 30            # first title letter (steps after the title appears)
UNROLL = (6, 58)        # the map unrolls
RISE = (8, 66)          # the balloon rises in front of it
MENU_T0 = 62            # menu plates slide in, one every 4 steps
INTRO_END = 92
HOVER = (124.0, 150.0)  # the balloon's basket once it has risen (screen px)
MENU_X, MENU_Y, MENU_W = 251, 52, 62


def _overview_pos(bx, by):
    ox, oy, ow, oh = LW.OVERVIEW_RECT
    return ox + int(bx * ow / LW.MAP_W), oy + int(by * oh / LW.MAP_H)


class TitleMode:
    """The title screen (see the module doc). A game mode: step(inp) / render(t, alpha) / iris_center()."""

    def __init__(self, game):
        self.g = game
        self.t0 = game.t
        self.has_save = game.has_save()
        self.items = ([('continue', 'play', 'CONTINUE')] if self.has_save else []) + \
            [('new', 'newmap', 'NEW GAME'), ('settings', 'gear', 'SETTINGS'), ('quit', 'door', 'QUIT')]
        self.sel = 0
        self.rep = Repeat()
        self.confirm = None             # step NEW GAME asked "sure?" (a save would be replaced)
        self.page = 'menu'              # 'menu' | 'settings' | 'controls'
        self.settings = RowMenu(game, SETTINGS_ROWS + [('back', 'back', 'BACK', 'button')])
        self.chose_t = None
        self.t_sel = game.t
        self.prev = self.t0
        game.music = 'title'

    # -------------------------------------------------------------- geometry
    def _f(self, t):
        return t - self.t0

    def balloon_pos(self, f):
        """Basket bottom-centre (screen px, floats) at title step f (may be fractional)."""
        k = ease_out(clamp((f - RISE[0]) / (RISE[1] - RISE[0])))
        x0, y0 = HOVER
        x = x0 + 8 * math.sin(f * 0.021) * k
        y = 214 + (y0 - 214) * k + 2.5 * math.sin(f * 0.033) * k
        return x, y

    def iris_center(self):
        f = self._f(self.g.t)
        if f < RISE[1]:                             # the opening iris: from the middle of the board
            return W // 2, H // 2
        x, y = self.balloon_pos(f)
        return int(round(x)), int(round(y)) - 18

    # -------------------------------------------------------------- update
    def step(self, inp):
        g = self.g
        f = self._f(g.t)
        if f < INTRO_END:
            if inp.action or inp.back or inp.pause:     # any key skips the intro
                self.t0 = g.t - INTRO_END
                g.sfx('pop', 0.3, 2)
            return
        if self.page == 'controls':
            if inp.back or inp.action or inp.left:
                self.page = 'settings'
                g.sfx('pop', 0.3, 1)
            return
        if self.page == 'settings':
            r = self.settings.step(inp)
            if r in ('back',):
                self.page = 'menu'
                g.settings.save()
            elif r == 'controls':
                self.page = 'controls'
            return
        go = self.rep(inp, g.t)
        if go['up'] or go['down']:
            self.sel = (self.sel + (1 if go['down'] else -1)) % len(self.items)
            self.confirm = None
            self.t_sel = g.t
            g.sfx('tick', 0.25)
        if inp.back and self.confirm is not None:
            self.confirm = None
            g.sfx('pop', 0.3, 1)
        if not inp.action:
            return
        key = self.items[self.sel][0]
        if key == 'continue':
            self._start(new=False)
        elif key == 'new':
            if self.has_save and self.confirm is None:
                self.confirm = g.t
                g.sfx('tick', 0.5)
            else:
                self._start(new=True)
        elif key == 'settings':
            self.page = 'settings'
            self.settings.sel = 0
            g.sfx('pop', 0.35, 0)
        elif key == 'quit':
            g.quit = True

    def _start(self, new):
        g = self.g
        self.chose_t = g.t
        g.sfx('coin', 0.5)
        sx, sy = self.iris_center()

        def go():                       # behind the closed iris
            if new:
                g.new_game()
            g.music = 'map'
            return g.map
        g.iris_to(sx, sy, go)

    # -------------------------------------------------------------- render
    def render(self, t, alpha=0.0):
        g = self.g
        f = self._f(t)
        ff = f + alpha
        c = Canvas()
        reveal = ease_in_out(clamp((f - UNROLL[0]) / (UNROLL[1] - UNROLL[0])))
        labels = clamp((f - UNROLL[1]) / 24.0) if reveal >= 0.999 else False
        LW.draw_overview(c, t, hour=g.hour, found=0, labels=labels or False, reveal=reveal)
        if reveal < 0.999:                          # the board shows below the unrolling map
            x, y, w, h = LW.OVERVIEW_RECT
            top = y + int(round(h * reveal)) + (4 if reveal > 0 else 0)
            if top < y + h:
                c.rect(x, top, w, y + h - top, '#1b1934')
                for yy in range(top + (top % 4), y + h, 4):
                    for xx in range(x + (yy // 4 % 2) * 4, x + w, 8):
                        c.px(xx, yy, '#2e2a52')
        fr = Frame(c)
        # the balloon (a sub-pixel sprite) and its shadow on the map
        bx, by = self.balloon_pos(ff)
        if reveal > 0.6:
            fr.shadow(bx, HOVER[1] + 14, 4)
        rising = f < RISE[1] + 6
        fr.balloon(bx, by, t + alpha, 'l', flame=rising or (t // 40) % 5 == 0, pattern=g.pattern)
        # the title card: drawn on the world (its shadow blends with the map), then lifted onto the
        # HUD so it stays in front of the balloon
        before = c.a.copy()
        card = draw_title_card(c, f, 160, 3, t0=CARD_T0, title='EQUESTRIA', subtitle='THE WORLD ATLAS', step=5,
                               sub_t0=CARD_T0 + 30, shine_every=240)
        changed = (c.a != before).any(axis=2)
        fr.hud.a[changed] = c.a[changed]
        h = fr.hud
        if reveal >= 0.999:                         # in front of the card and the balloon
            self._you_are_here(h, t, card if f >= CARD_T0 else None)
        if f >= MENU_T0 - 4:
            self._progress(h, t, f)
        if self.page == 'menu':
            self._menu(h, t, f)
        elif self.page == 'settings':
            w = PAUSE_W
            hh = self.settings.height() + 10
            x, y = (W - w) // 2, 52
            panel(h, x, y, w, hh)
            ui.plate(h, W // 2, y - 7, 'SETTINGS')
            self.settings.draw(h, x + 2, y + 6, w - 4, t)
            ui.hint_row(h, [(['up', 'down'], None), (['left', 'right'], _icon_slider), (['SPACE'], None)],
                        y=H - 13)
        else:
            draw_controls(h, t)
            ui.hint_row(h, [(['ESC'], _icon_back)], right=W - 8, y=H - 11)
        return fr

    def _you_are_here(self, h, t, card=None):
        """Pulsing ring on the overview where the saved balloon is. When the title card hides that spot
        (a northern save), a pin chip hangs from the card's bottom edge right below it instead."""
        if not self.has_save:
            return
        x, y = _overview_pos(self.g.map.bx, self.g.map.by)
        lit = (t // 6) % 2
        if card is not None:
            cx0, cy0, cw, ch = card
            if cx0 - 5 <= x < cx0 + cw + 5 and y < cy0 + ch + 5:
                px = int(clamp(x, cx0 + 9, cx0 + cw - 10))
                _pin_chip(h, px, cy0 + ch - 1, lit, t)
                return
        r = 3 + (t // 6) % 3
        h.circle(x, y, r + 1, ui.INK, fill=False)
        h.circle(x, y, r, 'hot_pink' if lit else 'white', fill=False)
        h.px(x, y, 'butter')

    def _menu(self, h, t, f):
        for i, (key, icon, label) in enumerate(self.items):
            k = ease_out_back(clamp((f - MENU_T0 - 4 * i) / 8.0))
            if k <= 0:
                continue
            x = MENU_X + int((1 - k) * 80)
            y = MENU_Y + i * 18
            lit = i == self.sel and f >= INTRO_END
            sure = key == 'new' and self.confirm is not None
            h.round_rect(x - 1, y - 1, MENU_W + 2, 16, DARK_RIM, r=2)
            h.round_rect(x, y, MENU_W, 14, ui.RIM if lit else '#8a6a3a', r=2)
            h.rect(x + 1, y + 1, MENU_W - 2, 12, ui.PLATE_HI if lit else ui.PLATE)
            h.hline(x + 2, x + MENU_W - 3, y + 1, '#7a5a80' if lit else ui.PLATE_HI)
            bob = -1 if lit and (t // 10) % 2 else 0
            micon(h, icon, x + 3, y + 3 + bob, lit)
            if sure:
                lab, colr = 'SURE?', ('butter' if (t // 6) % 2 else 'hot_pink')
            else:
                lab, colr = label, GOLD_TXT if lit else 'cream'
            text(h, x + 14, y + 4, lab, colr, font='normal', shadow=DARK_RIM if lit else None)
            if lit:
                _cursor(h, x - 6, y + 4, t)
        if f >= INTRO_END:
            y = MENU_Y + len(self.items) * 18 + 6
            if self.confirm is not None:            # SPACE = yes, ESC = no
                ui.keycap(h, MENU_X + 3, y, 'SPACE', lit=(t // 8) % 2 == 0)
                micon(h, 'newmap', MENU_X + 3 + ui.keycap_width('SPACE') + 3, y)
                ui.keycap(h, MENU_X + 3, y + 11, 'ESC')
                _icon_back(h, MENU_X + 3 + ui.keycap_width('ESC') + 3, y + 11)
            else:
                ui.keycap(h, MENU_X + 3, y, 'up')
                ui.keycap(h, MENU_X + 13, y, 'down')
                ui.keycap(h, MENU_X + 25, y, 'SPACE', lit=(t // 12) % 2 == 0)

    def _progress(self, h, t, f):
        """Left panel: the saved journey (places found, places complete, stamps, places helped, badges),
        each counter with a tiny caption; for a new player the journey ahead, greyed (places, places
        helped, balloon friends, races, postcards: all at 0)."""
        g = self.g
        from .mend_art import heart_sprite          # Phase B: places helped and badges
        from .quests import TROUBLES
        from .badges import BADGES, ICON as BADGE_ICON
        if self.has_save:
            n_done = 0
            for lid in TL.TOUR:
                n, tot, _s, _st = g.place_progress(lid)
                n_done += bool(tot) and n >= tot
            rows = [(ui._STAR, len(g.found), len(TL.TOUR), 'PLACES'),
                    (_DONE, n_done, len(TL.TOUR), 'COMPLETE'),
                    (ui._STAMP, g.stamps_got(), 3 * len(TL.TOUR), 'STAMPS'),
                    (heart_sprite(), len(g.quests.fixed), len(TROUBLES), 'HELPED'),
                    (BADGE_ICON, len(g.badges.earned), len(BADGES), 'BADGES')]
        else:
            from .flock import FRIENDS
            from .races import COURSES, FLAG
            from .album import TARGETS
            rows = [(ui._STAR, 0, len(TL.TOUR), 'PLACES'), (heart_sprite(), 0, len(TROUBLES), 'HELPED'),
                    (_balloon_sprite(), 0, len(FRIENDS), 'FRIENDS'), (FLAG, 0, len(COURSES), 'RACES'),
                    (ui.ICONS['camera'], 0, len(TARGETS), 'POSTCARDS')]
        x0 = LW.LEFT_PANEL[0] + 6
        xc = LW.LEFT_PANEL[0] + 3                       # rows slide in from under the panel's left rim
        keep = h.a[:, :xc].copy() if f - MENU_T0 + 4 - 2 * len(rows) < 8 else None
        for i, (spr, n, tot, cap) in enumerate(rows):
            k = ease_out_back(clamp((f - MENU_T0 + 4 - 2 * i) / 8.0))
            if k <= 0:
                continue
            _stat(h, spr, n, tot, cap, x0 - int((1 - k) * 80), MENU_Y - 4 + i * 21, ghost=not self.has_save)
        if keep is not None:
            h.a[:, :xc] = keep


STAT_W = 56


def _stat(h, spr, n, tot, caption, x, y, ghost=False):
    """One left-panel counter: a plate (icon, n/total on the right) with a tiny caption under it.
    ghost: the journey ahead of a new player, in muted greys."""
    s = f'{n}/{tot}'
    h.round_rect(x, y, STAT_W, 11, '#3a3258' if ghost else ui.INK, r=1)
    h.rect(x + 1, y + 1, STAT_W - 2, 9, PIP_OFF if ghost else ui.PLATE)
    h.hline(x + 2, x + STAT_W - 3, y + 1, '#3a3258' if ghost else ui.PLATE_HI)
    (_ghost(spr) if ghost else spr).draw(h, x + 2, y + 1 + (9 - spr.h) // 2)
    text(h, x + STAT_W - 4 - text_width(s, 'tiny'), y + 3, s, 'grey' if ghost else 'white', font='tiny')
    text(h, x + 1, y + 13, caption, 'grey' if ghost else 'periwinkle', font='tiny')


_GHOSTS = {}


def _ghost(spr):
    """A greyed copy of an icon sprite (dark parts indigo, the rest grey), cached."""
    g = _GHOSTS.get(id(spr))
    if g is None:
        a = spr.rgba.copy()
        lum = a[:, :, :3].astype(int) @ [3, 6, 1]
        on = a[:, :, 3] > 0
        a[on & (lum < 900), :3] = col('indigo')
        a[on & (lum >= 900), :3] = col('grey')
        g = _GHOSTS[id(spr)] = (spr, Sprite(a))
    return g[1]


_BALLOON = []


def _balloon_sprite():
    """ui.icon_balloon as a Sprite (the friends counter)."""
    if not _BALLOON:
        cv = Canvas(8, 11, color=(255, 0, 254))
        ui.icon_balloon(cv, 1, 1)
        _BALLOON.append(Sprite.from_canvas(cv, key=(255, 0, 254)))
    return _BALLOON[0]


def _pin_chip(h, x, y, lit, t):
    """A small 'you are here' chip hanging from the title card's bottom edge at x: a pin on a plate,
    with a notch pointing up at the spot the card hides."""
    w, hh = 13, 10
    x0 = x - w // 2
    h.round_rect(x0, y, w, hh, ui.INK, r=2)
    h.round_rect(x0 + 1, y + 1, w - 2, hh - 2, 'hot_pink' if lit else 'wine', r=1)
    h.hline(x0 + 2, x0 + w - 3, y + 1, 'pink' if lit else 'hot_pink')
    for dy in range(3):                                 # the notch, up into the card
        h.hline(x - dy, x + dy, y - 3 + dy, ui.INK)
    h.px(x, y - 2, 'hot_pink' if lit else 'wine')
    h.hline(x - 1, x + 1, y - 1, 'hot_pink' if lit else 'wine')
    bob = (t // 12) % 2
    px, py = x - 2, y + 2 + bob                         # the pin: a round head and a point
    h.hline(px + 1, px + 3, py, 'white')
    h.rect(px, py + 1, 5, 2, 'white')
    h.px(px + 2, py + 1, 'crimson')
    h.hline(px + 1, px + 3, py + 3, 'cream')
    h.px(px + 2, py + 4, 'cream')


def _icon_back(c, x, y):
    if c is None:
        return 9
    ui.ICONS['back'].draw(c, x, y)
    return 9


_DONE = Sprite.from_ascii('''
    .ooooooo.
    oyyyyyyyo
    oyyyyyyWo
    oyWyyyWyo
    oyyWyWyyo
    oyyyWyyyo
    .ooooooo.
''', {'o': ui.INK, 'y': 'leaf', 'W': 'white'})
