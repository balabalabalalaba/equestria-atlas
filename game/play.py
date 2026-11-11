"""Play the EQUESTRIA ATLAS in a window.

    .venv/bin/python game/play.py      # options: --mute, --scale N, --new, --skip-title, --frames N

It opens on the title screen (Continue / New Game / Settings / Quit).
Keys: arrows / WASD fly the balloon, SHIFT boost, SPACE / ENTER land (map) or set off the thing the
balloon is next to (in a place), ESC take off (in a place), close a screen, or pause (on the map),
M overview map, J atlas journal (and the balloon wardrobe), P postcard snapshot (Pictures/Equestria Atlas;
the journal's PHOTOS page shows them all, full screen with SPACE),
F fullscreen, H hide / show the HUD (clean screenshots), F12 raw screenshot; quit from the pause menu (QUIT) or by closing the window. Gamepad: stick / d-pad fly, A (button 0) land / set off,
B (1) take off, X (2) journal, Y (3) map, shoulders boost, Back (6) postcard, Start (7) pause.
When the window loses focus the world freezes quietly (no menu over the view, so a screenshot tool
catches the scenery) and runs on when the window is clicked again; ESC opens the pause menu.
Progress is saved to out/game_save.json, settings (volumes, window scale, fullscreen) to
out/game_settings.json.

Timing: the game logic steps at the film's 30 per second; the window redraws at the display rate
(vsync, 60 Hz) and interpolates between steps, with sub-pixel scrolling (game/compose.py).
"""
import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FROZEN = getattr(sys, 'frozen', False)      # running as the packaged app (packaging/, PyInstaller)


def user_dir():
    """Where the packaged app keeps saves, settings and pictures (the app itself is read-only);
    the project's out/ folder when run from source."""
    if not FROZEN:
        return os.path.join(ROOT, 'out')
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    elif os.name == 'nt':
        base = os.environ.get('APPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
    d = os.path.join(base, 'Equestria Atlas')
    os.makedirs(d, exist_ok=True)
    return d

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np        # noqa: E402
import pygame             # noqa: E402

from engine import FPS                                # noqa: E402
from game.world import Game, Input, FINALE_SEG         # noqa: E402
from game.compose import compose, compose_window, VIEW_W, VIEW_H       # noqa: E402
from game.settings import Settings                     # noqa: E402

SOUNDTRACK = os.path.join(ROOT, 'audio', 'out', 'soundtrack.wav')     # the film score (finale, fallback)
GAME_MUSIC = os.path.join(ROOT, 'audio', 'out', 'game')                  # the game's own loops (audio/game_music.py)
MUSIC_VOL, SFX_VOL = 0.75, 1.0        # at full volume in the settings
GAME_MUSIC_VOL = 0.88                 # the game loops are mastered 1.5 dB under the film score: level them
SR = 48000
STEP = 1.0 / FPS
STALL = 0.1                     # a frame gap longer than this (s) is a stall: no catch-up, no fade jump
SAVE_FILE = os.path.join(user_dir(), 'game_save.json')
SETTINGS_FILE = os.path.join(user_dir(), 'game_settings.json')

# Rewards duck the music: while one of these cues plays (at most DUCK_MAX s) the score sinks to DUCK
# (about -6 dB), with a DUCK_ATTACK s attack and a DUCK_RELEASE s release; the journal and the pause
# menu dip it to DIP (about -4 dB). Ambient cues (wind, water, puffs...) never duck it.
REWARD = {'quest', 'jackpot', 'shimmer', 'stamp', 'star', 'powerup', 'coin', 'idea', 'chime'}
DUCK, DIP = 0.5, 0.63
DUCK_ATTACK, DUCK_RELEASE, DUCK_MAX = 0.06, 0.6, 1.2
DUCK_MIN_GAIN = 0.35            # quieter reward cues (the coin of a tip learnt, 0.3) leave the music alone
FADE_IN, FADE_OUT = 0.9, 0.7    # s: a track fades in / out when the music changes
MAP_FADE = 1.0                  # s: the map music's fade out (it pauses, keeping its place) and back in
MUSIC_CHANNELS = 4              # reserved mixer channels: map by day, map by night, and two for the rest
DECODED = ('title', 'map_day', 'map_night', 'race')    # decoded at start-up (before any music plays); the
# places' themes are streamed from their files (pygame.mixer.music): decoding a whole track with
# pygame.mixer.Sound holds the audio device's lock for ~60-90 ms, which stalled the game and made the
# music skip whenever a place's theme was decoded mid-game


def night_weight(hour):
    """How much of the night arrangement of the map theme plays at this hour (0 by day, 1 at night): it
    crossfades in from 18:00 to 20:00 and out from 05:00 to 07:00, in step with the map's light."""
    h = hour % 24.0
    if 7.0 <= h < 18.0:
        return 0.0
    if 18.0 <= h < 20.0:
        u = (h - 18.0) / 2.0
    elif 5.0 <= h < 7.0:
        u = 1.0 - (h - 5.0) / 2.0
    else:
        return 1.0
    return u * u * (3 - 2 * u)


class Audio:
    """Music and procedural SFX through pygame.mixer; silent if there is no audio device.

    The music is the game's own soundtrack (audio/out/game: one seamless loop per screen): the title
    theme, the map theme in two arrangements that play in sync and crossfade with the world clock (day /
    night), a race theme while a ring race or a balloon friend's race runs, and one theme per place. The
    map theme pauses (keeping its place) while another plays. The finale plays the film's outro, and a
    screen without a game track falls back to its bars of the film score. The title, map and race tracks
    are decoded at start-up, before any music plays; a place's theme is streamed from its file, so
    nothing is decoded while music plays (see DECODED)."""

    def __init__(self, enabled=True):
        self.ok = False
        if not enabled:
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(SR, -16, 2, 1024)
            if pygame.mixer.get_init()[0] != SR:          # every sound here is built at 48 kHz
                pygame.mixer.quit()
                pygame.mixer.init(SR, -16, 2, 1024)
            pygame.mixer.set_num_channels(24)
            pygame.mixer.set_reserved(MUSIC_CHANNELS)
        except pygame.error as e:
            print('audio off:', e)
            return
        if pygame.mixer.get_init()[0] != SR:
            print('audio: the mixer runs at', pygame.mixer.get_init()[0], 'Hz, not', SR)
        import audio.sfx as sfx
        import threading
        self.sfx = sfx
        self.ok = True
        self.cache = {}
        self.tracks = self._manifest()
        self.music = {}                  # track id -> decoded Sound (the map, title and recent ones)
        self.lru = []
        self.loading = {}                # track id -> thread decoding it
        self.lock = threading.Lock()
        self.chans = [pygame.mixer.Channel(i) for i in range(MUSIC_CHANNELS)]
        self.map_on = False              # the two map channels are playing (maybe paused)
        self.map_paused = False
        self.map_vol = 0.0               # the map music's fade (0..1)
        self.solo = {}                   # channel index (2 or 3) -> dict(id, vol, target)
        self.stream = None               # the streamed place theme: dict(id, vol, target, gain), or None
        self.track = None
        self.music_gain = 1.0            # the settings' volumes (0..1)
        self.sound_gain = 1.0
        self.clock = time.perf_counter   # (a test can drive the ducking with its own clock)
        self.duck_until = 0.0            # clock time a reward cue stops ducking the music
        self.duck = 1.0                  # the music's duck factor now (eased)
        self.hour = 12.0

    @staticmethod
    def _manifest():
        import json
        try:
            with open(os.path.join(GAME_MUSIC, 'tracks.json')) as fh:
                return {t['id']: t for t in json.load(fh)
                        if os.path.exists(os.path.join(GAME_MUSIC, t['id'] + '.ogg'))}
        except (OSError, ValueError):
            return {}

    def warm(self):
        """Synthesize every sound the game asks for now (~0.5 s at start-up), so none is built
        mid-game (the first cloud sweep's 'wind' used to stall a frame by ~25 ms); decode the title and
        map music."""
        if not self.ok:
            return
        import glob
        import re
        from game.races import GATE_CUES
        keys = {('pop', 0), ('pop', 1), ('pop', 2), ('pop', 3), ('tick', 1)} | set(GATE_CUES)
        for fn in glob.glob(os.path.join(ROOT, 'game', '**', '*.py'), recursive=True):
            with open(fn) as fh:
                src = fh.read()
            for m in re.finditer(r"sfx\('([a-z_0-9]+)'(?:\s*,\s*[^,()]+(?:\([^)]*\))?\s*,\s*(\d+))?", src):
                keys.add((m.group(1), int(m.group(2) or 0)))
        for key in sorted(keys):
            try:
                self._sound(key)
            except Exception as e:                  # an unknown name: it fails the same way in play
                print('sfx', key, 'not built:', e)
        for tid in DECODED:                              # all four at once, and done before any music plays
            self.prefetch(tid)
        for th in list(self.loading.values()):
            th.join()

    def _sound(self, key):
        if key not in self.cache:
            self.cache[key] = self._stereo(self.sfx.make(*key))
        return self.cache[key]

    def set_volumes(self, music, sound):
        self.music_gain, self.sound_gain = music, sound

    def _stereo(self, x):
        a = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.ascontiguousarray(np.repeat(a[:, None], 2, 1)))

    def play(self, name, gain, variant=None):
        if not self.ok or gain <= 0.01:
            return
        snd = self._sound((name, variant or 0))
        ch = snd.play()
        g = min(1.0, gain * SFX_VOL * self.sound_gain)
        if ch is not None:
            ch.set_volume(g)
        if name in REWARD and g >= DUCK_MIN_GAIN:        # (a muted reward cue leaves the music alone)
            self.duck_until = max(self.duck_until, self.clock() + min(DUCK_MAX, snd.get_length()))

    # -------------------------------------------------------------- music tracks
    def _decode(self, tid):
        try:
            snd = pygame.mixer.Sound(os.path.join(GAME_MUSIC, tid + '.ogg'))
        except pygame.error as e:
            print('music', tid, 'not loaded:', e)
            snd = None
        with self.lock:
            if snd is not None:
                self.music[tid] = snd
            self.loading.pop(tid, None)

    def prefetch(self, tid):
        """Start decoding a track in the background (no-op if it is ready, loading, unknown or streamed)."""
        if not self.ok or tid not in self.tracks or tid not in DECODED:
            return
        import threading
        with self.lock:
            if tid in self.music or tid in self.loading:
                return
            th = threading.Thread(target=self._decode, args=(tid,), daemon=True)
            self.loading[tid] = th
        th.start()

    def _get(self, tid):
        """The decoded Sound of a game track (waiting for its decode if it is under way), or None."""
        if tid not in self.tracks:
            return None
        self.prefetch(tid)
        th = self.loading.get(tid)
        if th is not None:
            th.join()
        snd = self.music.get(tid)
        if snd is not None and tid not in ('title', 'map_day', 'map_night', 'race'):
            if tid in self.lru:
                self.lru.remove(tid)
            self.lru.append(tid)
            while len(self.lru) > 4:                     # keep the last few places' themes decoded
                old = self.lru.pop(0)
                if all(s.get('id') != old for s in self.solo.values()):
                    self.music.pop(old, None)
        return snd

    def _film_loop(self, tid, span):
        """A screen without a game track: its bars of the film score, as a loop (10 ms edge ramps)."""
        key = 'film:' + tid
        if key not in self.music:
            from scipy.io import wavfile
            sr, a = wavfile.read(SOUNDTRACK, mmap=True)
            s0, s1 = span
            x = np.array(a[int(s0 * sr):int(s1 * sr)], np.float32)
            n = int(0.01 * sr)
            ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]
            self.music[key] = pygame.sndarray.make_sound(np.ascontiguousarray(x.astype(np.int16)))
        return self.music[key]

    def _start_solo(self, tid, spans):
        """Play a track (a place, the title, the race, the finale) from its start: a place's theme as the
        stream, the others on a free solo channel."""
        if tid in self.tracks and tid not in DECODED:
            try:
                pygame.mixer.music.load(os.path.join(GAME_MUSIC, tid + '.ogg'))
                pygame.mixer.music.set_volume(0.0)
                pygame.mixer.music.play(loops=-1)
                self.stream = dict(id=tid, vol=0.0, target=1.0, gain=GAME_MUSIC_VOL)
                return
            except pygame.error as e:
                print('music', tid, 'not streamed:', e)
                self.stream = None
        snd, film = self._get(tid), False
        if snd is None and tid in spans:
            snd, film = self._film_loop(tid, spans[tid]), True
        if snd is None:
            return
        i = 2 if 2 not in self.solo else 3
        if i in self.solo:                                 # both busy (a quick change): cut the older
            self.chans[i].stop()
        ch = self.chans[i]
        ch.set_volume(0.0)
        ch.play(snd, loops=-1)
        self.solo[i] = dict(id=tid, vol=0.0, target=1.0, gain=MUSIC_VOL if film else GAME_MUSIC_VOL)

    def update(self, want, spans, dt, dip=False, hour=None):
        """Called every display frame with the wanted music: 'map', 'race', 'title', a place's id, or the
        finale's section id; dip: the journal or the pause menu is open; hour: the world clock."""
        if not self.ok:
            return
        if hour is not None:
            self.hour = hour
        target = DUCK if self.clock() < self.duck_until else 1.0     # the reward duck, eased
        if dip:
            target = min(target, DIP)
        if self.duck > target:
            self.duck = max(target, self.duck - (1 - DUCK) * dt / DUCK_ATTACK)
        elif self.duck < target:
            self.duck = min(target, self.duck + (1 - DUCK) * dt / DUCK_RELEASE)
        use_map = want == 'map' and 'map_day' in self.tracks and 'map_night' in self.tracks
        if want != self.track:
            self.track = want
            for s in self.solo.values():                 # whatever plays alone fades out
                s['target'] = 0.0
            if self.stream is not None:
                self.stream['target'] = 0.0
            if use_map:
                if not self.map_on:
                    day, night = self._get('map_day'), self._get('map_night')
                    if day is not None and night is not None:
                        for ch, snd in ((self.chans[0], day), (self.chans[1], night)):
                            ch.set_volume(0.0)
                        self.chans[0].play(day, loops=-1)    # started together: they stay in step
                        self.chans[1].play(night, loops=-1)
                        self.map_on, self.map_paused = True, False
                elif self.map_paused:
                    self.chans[0].unpause()
                    self.chans[1].unpause()
                    self.map_paused = False
            elif want is not None and (want in self.tracks or want in spans) and want != 'map':
                self._start_solo(want, spans)
            elif want == 'map':                          # no game map tracks: the film score loops
                self._start_solo('map_film', {'map_film': (0.0, 10 ** 6)})
        # fades
        base = self.music_gain * self.duck
        self.map_vol += (1 if use_map and self.map_on else -1) * dt / MAP_FADE
        self.map_vol = min(1.0, max(0.0, self.map_vol))
        if self.map_on:
            w = night_weight(self.hour)
            v = self.map_vol * base * GAME_MUSIC_VOL
            # equal-power crossfade between the two arrangements
            self.chans[0].set_volume(round(v * math.cos(w * math.pi / 2), 3))
            self.chans[1].set_volume(round(v * math.sin(w * math.pi / 2), 3))
            if self.map_vol <= 0 and not self.map_paused and not use_map:
                self.chans[0].pause()
                self.chans[1].pause()
                self.map_paused = True
        for i in list(self.solo):
            s = self.solo[i]
            rate = dt / (FADE_IN if s['target'] > s['vol'] else FADE_OUT)
            s['vol'] = min(s['target'], s['vol'] + rate) if s['target'] > s['vol'] else max(s['target'], s['vol'] - rate)
            self.chans[i].set_volume(round(s['vol'] * base * s['gain'], 3))
            if s['vol'] <= 0 and s['target'] <= 0:
                self.chans[i].stop()
                del self.solo[i]
        s = self.stream
        if s is not None:
            rate = dt / (FADE_IN if s['target'] > s['vol'] else FADE_OUT)
            s['vol'] = min(s['target'], s['vol'] + rate) if s['target'] > s['vol'] else max(s['target'], s['vol'] - rate)
            pygame.mixer.music.set_volume(round(s['vol'] * base * s['gain'], 3))
            if s['vol'] <= 0 and s['target'] <= 0:
                pygame.mixer.music.stop()
                self.stream = None

    def pause_all(self, paused):
        """The window is minimised: hold every sound (pygame.mixer.pause keeps the places)."""
        if not self.ok:
            return
        if paused:
            pygame.mixer.pause()
            pygame.mixer.music.pause()
        else:
            pygame.mixer.unpause()
            pygame.mixer.music.unpause()
            if self.map_paused:                          # (it was resting while a place played)
                self.chans[0].pause()
                self.chans[1].pause()


def calm_ime():
    """Keep an input method (Chinese / Japanese / Korean IME) out of the game window (GitHub issue #1).

    pygame turns SDL's text input on by default, and on Windows that hands every key to the IME: with a
    Chinese IME in Chinese mode W / A / S / D went into its composition window and never reached the game,
    and tapping SHIFT (to boost) flipped the IME between Chinese and English. The game never needs typed
    text, so text input is turned off (SDL then also detaches the IME from the window), and on Windows the
    window's IME context is detached explicitly as well. Called after every display (re)creation and
    whenever the window gets the focus back (another app may have switched the IME on)."""
    try:
        pygame.key.stop_text_input()
    except (pygame.error, AttributeError):
        pass
    if os.name != 'nt':
        return
    try:
        import ctypes
        ctypes.windll.imm32.ImmDisableIME(ctypes.c_uint32(0xFFFFFFFF).value)   # every thread of the game
        hwnd = pygame.display.get_wm_info().get('window')
        if hwnd:
            imm = ctypes.windll.imm32
            imm.ImmAssociateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            imm.ImmAssociateContext.restype = ctypes.c_void_p
            imm.ImmAssociateContext(ctypes.c_void_p(hwnd), None)
    except Exception as e:                              # (no imm32: nothing to detach)
        print('ime: could not detach the input method:', e)


class ShortcutKeys:
    """Windows' accessibility shortcuts fire on the boost key: SHIFT pressed five times asks to turn on
    Sticky Keys, right SHIFT held for 8 s asks to turn on Filter Keys. Their dialog takes the focus away
    in the middle of a flight (GitHub issue #1). As Microsoft recommends for games ("Disabling shortcut
    keys in games"), while the game window has the focus the *shortcuts* of features that are OFF are
    disabled, for this session only (not saved to the user's profile); the player's own settings are put
    back as soon as the window loses the focus and when the game quits. A feature the player has turned
    ON is left alone. Does nothing outside Windows."""

    SPI = {'sticky': (0x003A, 0x003B, 2), 'toggle': (0x0034, 0x0035, 2), 'filter': (0x0032, 0x0033, 6)}
    ON, HOTKEY, CONFIRM = 0x1, 0x4, 0x8

    def __init__(self, path=None):
        self.saved = None                   # name -> original flags while disabled
        self.ok = os.name == 'nt'
        self.path = path                    # the originals, on disk while disabled: a game that was killed
        if self.ok:                         # (no atexit) puts them back on its next start
            import atexit
            atexit.register(self.restore)
            self._recover()

    def _recover(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            import json
            with open(self.path) as fh:
                self.saved = {k: int(v) for k, v in json.load(fh).items() if k in self.SPI}
        except (OSError, ValueError, AttributeError, TypeError):
            self.saved = None
        if self.saved is None:
            self._forget()
        else:
            print('shortcut keys: putting back the settings a previous session left changed')
            self.restore()

    def _struct(self, n):
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [('cbSize', ctypes.c_uint), ('dwFlags', ctypes.c_uint)] + \
                       [('f%d' % i, ctypes.c_uint) for i in range(n - 2)]
        s = S()
        s.cbSize = ctypes.sizeof(S)
        return s

    def disable(self):
        if not self.ok or self.saved is not None:
            return
        import ctypes
        spi = ctypes.windll.user32.SystemParametersInfoW
        self.saved = {}
        try:
            for name, (get, put, n) in self.SPI.items():
                s = self._struct(n)
                if not spi(get, s.cbSize, ctypes.byref(s), 0):
                    continue
                flags = s.dwFlags
                if flags & self.ON:                     # the player uses this feature: leave it be
                    continue
                self.saved[name] = flags
                self._keep()                                # on disk BEFORE anything changes
                s.dwFlags = flags & ~(self.HOTKEY | self.CONFIRM)
                spi(put, s.cbSize, ctypes.byref(s), 0)      # fWinIni 0: this session only
        except Exception as e:
            print('shortcut keys: left as they are:', e)

    def _keep(self):
        if self.path:
            try:
                import json
                with open(self.path, 'w') as fh:
                    json.dump(self.saved, fh)
            except OSError:
                pass

    def _forget(self):
        if self.path:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def restore(self):
        if not self.ok or self.saved is None:
            return
        import ctypes
        spi = ctypes.windll.user32.SystemParametersInfoW
        try:
            for name, flags in self.saved.items():
                get, put, n = self.SPI[name]
                s = self._struct(n)
                if spi(get, s.cbSize, ctypes.byref(s), 0):
                    s.dwFlags = flags
                    spi(put, s.cbSize, ctypes.byref(s), 0)
        except Exception as e:
            print('shortcut keys: could not restore:', e)
            return                                      # (the file stays: the next start tries again)
        self.saved = None
        self._forget()


PAD_DEAD = 0.35
PAD_BUTTONS = {0: 'action', 1: 'back', 2: 'journal', 3: 'map', 6: 'postcard', 7: 'pause'}


def pad_state(pads):
    """Held directions and boost from every connected gamepad (left stick, d-pad, shoulders)."""
    d = dict(left=False, right=False, up=False, down=False, boost=False)
    for js in pads:
        try:
            if js.get_numaxes() >= 2:
                ax, ay = js.get_axis(0), js.get_axis(1)
                d['left'] |= ax < -PAD_DEAD
                d['right'] |= ax > PAD_DEAD
                d['up'] |= ay < -PAD_DEAD
                d['down'] |= ay > PAD_DEAD
            if js.get_numhats() > 0:
                hx, hy = js.get_hat(0)
                d['left'] |= hx < 0
                d['right'] |= hx > 0
                d['up'] |= hy > 0
                d['down'] |= hy < 0
            for b in (4, 5):
                if js.get_numbuttons() > b and js.get_button(b):
                    d['boost'] = True
        except pygame.error:
            continue
    return d


def held_input(keys, pressed, pads=()):
    k = pygame
    p = pad_state(pads)
    return Input(left=bool(keys[k.K_LEFT] or keys[k.K_a] or p['left']),
                 right=bool(keys[k.K_RIGHT] or keys[k.K_d] or p['right']),
                 up=bool(keys[k.K_UP] or keys[k.K_w] or p['up']),
                 down=bool(keys[k.K_DOWN] or keys[k.K_s] or p['down']),
                 boost=bool(keys[k.K_LSHIFT] or keys[k.K_RSHIFT] or p['boost']),
                 action='action' in pressed, back='back' in pressed, map='map' in pressed,
                 journal='journal' in pressed, pause='pause' in pressed)


CLOCK_SPEEDS = (0.0, 1.0, 4.0, 16.0, 60.0)
BALLOON_SPEEDS = (1.0, 1.5, 2.0, 3.0)


def cheat_key(game, key):
    """Testing keys (--cheats). Returns True when the key was a cheat."""
    if key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
        game.hour = (game.hour + (1 if key == pygame.K_RIGHTBRACKET else -1)) % 24.0
    elif key == pygame.K_BACKSLASH:
        i = CLOCK_SPEEDS.index(game.clock_mult) if game.clock_mult in CLOCK_SPEEDS else 0
        game.clock_mult = CLOCK_SPEEDS[(i + 1) % len(CLOCK_SPEEDS)]
    elif key in (pygame.K_COMMA, pygame.K_PERIOD):
        i = BALLOON_SPEEDS.index(game.speed_mult) if game.speed_mult in BALLOON_SPEEDS else 0
        i = max(0, min(len(BALLOON_SPEEDS) - 1, i + (1 if key == pygame.K_PERIOD else -1)))
        game.speed_mult = BALLOON_SPEEDS[i]
    else:
        return False
    print(f'cheat: {int(game.hour):02d}:{int(game.hour % 1 * 60):02d}  clock x{game.clock_mult:g}  '
          f'balloon x{game.speed_mult:g}')
    return True


def fit_scale():
    """The largest window scale that fits the screen."""
    try:
        dw, dh = pygame.display.get_desktop_sizes()[0]
    except (pygame.error, IndexError):
        return 4
    return max(2, min(int(dw * 0.92) // VIEW_W, int(dh * 0.88) // VIEW_H))


class Display:
    """The window: (re)made from the settings' scale and fullscreen. Windowed, the window is the view
    times k. Fullscreen, the screen is the desktop size and the view is drawn at the largest whole scale
    that fits, centred on black: every art pixel is k x k screen pixels (a fractional stretch would make
    pixels uneven and shimmer as the camera glides)."""

    def __init__(self, settings, forced_scale=0):
        self.s = settings
        self.forced = forced_scale
        self.screen = None
        self.k = None
        self.full = None
        self.vsync = True
        self.win = None                 # pygame._sdl2 Window wrapper of the display (see apply)
        self.offset = (0, 0)
        self.apply()

    def _desktop(self):
        try:
            return pygame.display.get_desktop_sizes()[0]
        except (pygame.error, IndexError):
            return (VIEW_W * 4, VIEW_H * 4)

    def apply(self):
        s = self.s
        k = self.forced or s.scale or s.max_scale
        k = max(2, min(k, max(s.max_scale, self.forced or 0)))
        if s.fullscreen:
            dw, dh = self._desktop()
            k = max(1, min(dw // VIEW_W, dh // VIEW_H))
            size = (dw, dh)
        else:
            size = (VIEW_W * k, VIEW_H * k)
        if k == self.k and s.fullscreen == self.full:
            return
        flags = pygame.SCALED | (pygame.FULLSCREEN if s.fullscreen else 0)
        try:
            self.screen = pygame.display.set_mode(size, flags, vsync=1)
            self.vsync = True
        except pygame.error:
            self.screen = pygame.display.set_mode(size, flags)
            self.vsync = False
        if not s.fullscreen:                # SCALED picks its own (bigger) window for small sizes
            try:
                from pygame._sdl2.video import Window
                if self.win is None:            # keep ONE wrapper for the life of the display: a
                    self.win = Window.from_display_module()   # dropped wrapper frees the SDL window
                win = self.win
                if tuple(win.size) != size:
                    win.size = size
                    win.position = pygame.WINDOWPOS_CENTERED
            except (ImportError, AttributeError, pygame.error):
                pass
        self.k, self.full = k, s.fullscreen
        calm_ime()                          # a new / changed window: keep the IME out of it (issue #1)
        self.size = (VIEW_W * k, VIEW_H * k)
        sw, sh = self.screen.get_size()
        self.offset = (max(0, (sw - self.size[0]) // 2), max(0, (sh - self.size[1]) // 2))
        self.screen.fill((0, 0, 0))

    def show(self, buf, sx, sy):
        """Blit the view from a compose_window buffer (RGBX pixels) without copying it. Returns the
        Surface (it shares the buffer)."""
        surf = pygame.image.frombuffer(buf, (buf.shape[1], buf.shape[0]), 'RGBX')
        self.screen.blit(surf, self.offset, pygame.Rect(sx, sy, self.size[0], self.size[1]))
        return surf


def photo_dir():
    """Where postcards go: the player's Pictures folder (Pictures/Equestria Atlas), where people look
    for pictures; the game's own folder if there is no Pictures folder."""
    pics = os.path.join(os.path.expanduser('~'), 'Pictures')
    if os.path.isdir(pics):
        return os.path.join(pics, 'Equestria Atlas')
    return os.path.join(user_dir(), 'game_shots')


def open_folder(path):
    """Show a folder in Finder / Explorer / the file manager."""
    import subprocess
    try:
        os.makedirs(path, exist_ok=True)
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif os.name == 'nt':
            os.startfile(path)                          # noqa: (Windows only)
        else:
            subprocess.Popen(['xdg-open', path])
    except OSError as e:
        print('could not open', path, e)


def reveal_file(path):
    """Show a file selected in its folder (Finder / Explorer); the folder elsewhere."""
    import subprocess
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', path])
        elif os.name == 'nt':
            subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
        else:
            open_folder(os.path.dirname(path))
    except OSError as e:
        print('could not show', path, e)


def selftest():
    """Load every place, draw a frame in each, synthesize every sound: the check for a packaged build.
    Returns the exit status (0 = all good). The report also goes to selftest.txt in the user folder."""
    import traceback
    lines = []

    def say(s):
        print(s)
        lines.append(s)
    t0 = time.perf_counter()
    say(f'Equestria Atlas self-test  python {sys.version.split()[0]}  frozen={FROZEN}  {sys.platform}')
    bad = 0
    try:
        game = Game(save_path=None)
        game.tutor.welcomed = True
        compose(game.render(0.0), 2)
        say('world map: OK')
    except Exception:
        say('world map: FAILED\n' + traceback.format_exc())
        return 1
    for lid in list(game.region_ids):
        try:
            r = game.region(lid)
            r.warm()
            game.mode = r.enter()
            game.trans = None
            for h in (r.HOUR, 13.0, 23.0):
                game.hour = h
                compose(game.render(0.0), 2)
            say(f'{lid}: OK')
        except Exception:
            bad += 1
            say(f'{lid}: FAILED\n' + traceback.format_exc())
    try:
        import audio.sfx as sfx
        import glob
        import re
        names = set()
        for fn in glob.glob(os.path.join(ROOT, 'game', '**', '*.py'), recursive=True):
            src = open(fn).read()               # the g.sfx cues and the toasts' sound= cues
            names |= set(re.findall(r"sfx\('([a-z_0-9]+)'", src)) | set(re.findall(r"sound='([a-z_0-9]+)'", src))
        for n in sorted(names):
            sfx.make(n, 0)
        from scipy.io import wavfile
        sr, a = wavfile.read(SOUNDTRACK, mmap=True)
        say(f'sounds: {len(names)} OK, soundtrack {len(a) / sr:.0f} s')
        tracks = Audio._manifest()
        from game.regions import __path__ as _rp      # noqa: F401
        want = {'title', 'map_day', 'map_night', 'race'} | set(game.region_ids)
        missing = sorted(want - set(tracks))
        if missing:
            bad += 1
            say('music: MISSING ' + ', '.join(missing))
        else:
            say(f'music: {len(tracks)} tracks OK ({sum(t["seconds"] for t in tracks.values()) / 60:.0f} min)')
    except Exception:
        bad += 1
        say('sounds: FAILED\n' + traceback.format_exc())
    say(f'{"ALL OK" if not bad else f"{bad} PROBLEM(S)"} in {time.perf_counter() - t0:.0f} s')
    try:
        with open(os.path.join(user_dir(), 'selftest.txt'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
    except OSError:
        pass
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mute', action='store_true')
    ap.add_argument('--scale', type=int, default=0, help='window = 319x179 art px times N (default: settings)')
    ap.add_argument('--frames', type=int, default=0, help='quit after N game steps (smoke test)')
    ap.add_argument('--new', action='store_true', help='ignore the saved progress and start over')
    ap.add_argument('--skip-title', action='store_true', help='go straight to the world map')
    ap.add_argument('--selftest', action='store_true',
                    help='load every place, draw a frame in each and build every sound; print a report (for checking a packaged build)')
    ap.add_argument('--cheats', action='store_true',
                    help='testing keys: [ ] clock -/+ 1 hour, \\ clock speed, , . balloon speed')
    ap.add_argument('--save', default=SAVE_FILE, help='save file (default out/game_save.json)')
    ap.add_argument('--settings', default=SETTINGS_FILE, help='settings file (default out/game_settings.json)')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    print('Loading the world map...')
    t0 = time.perf_counter()
    if args.new and os.path.exists(args.save):
        os.replace(args.save, args.save + '.old')
        print('previous save kept as', args.save + '.old')
    settings = Settings(args.settings)
    game = Game(save_path=args.save, settings=settings, title=not args.skip_title)
    game.cheats = args.cheats
    if args.cheats:
        print('cheats on: [ ] clock -/+ 1 hour, \\ clock speed (stop/1/4/16/60), , . balloon speed (1/1.5/2/3)')
    print(f'ready in {time.perf_counter() - t0:.1f} s; {len(game.region_ids)} places to explore')

    try:
        pygame.mixer.pre_init(SR, -16, 2, 1024)          # before pygame.init(), which opens the mixer
    except pygame.error:
        pass
    pygame.init()
    audio = Audio(not args.mute)
    audio.set_volumes(settings.music_gain, settings.sound_gain)
    audio.warm()
    settings.max_scale = fit_scale()
    disp = Display(settings, args.scale)
    pygame.display.set_caption('Equestria Atlas')
    applied = settings.version
    pygame.joystick.init()
    pads = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for js in pads:
        js.init()
        print('gamepad:', js.get_name())
    clock = pygame.time.Clock()
    shots = os.path.join(user_dir(), 'game_shots')
    photos = photo_dir()
    home = os.path.expanduser('~')
    game.photo_label = (os.path.relpath(photos, home) if photos.startswith(home) else photos).upper().replace(
        os.sep, ' / ')
    game.photos.set_folders([photos, shots])      # the journal's PHOTOS page (earlier postcards are in shots)
    try:
        run(game, settings, args, audio, disp, clock, pads, shots, photos, applied)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:                                            # a trace for the player to send, and their progress
            with open(os.path.join(user_dir(), 'crash.txt'), 'w') as fh:
                fh.write(f'Equestria Atlas crashed {time.ctime()}  python {sys.version.split()[0]}  {sys.platform}\n')
                fh.write(tb)
            print('crash report written to', os.path.join(user_dir(), 'crash.txt'))
        except OSError:
            pass
        try:
            game.save(force=True)
        except Exception:
            pass
        settings.save()
        pygame.quit()
        sys.exit(1)
    game.save(force=True)
    settings.save()
    pygame.quit()


def run(game, settings, args, audio, disp, clock, pads, shots, photos, applied):
    pressed = set()
    acc = 0.0
    last = time.perf_counter()
    steps_run = 0                   # (--frames counts this session's steps: a save resumes its own clock)
    hidden = False                  # the window is minimised: nothing runs, the sound is held
    fast = 0                        # frames in a row that came quicker than any display refresh
    capped = not disp.vsync
    away = False                    # the window lost the focus: the world is frozen (no menu: issue #2)
    held_alpha = 0.0                # the in-between position the frozen view is drawn at
    shortcuts = ShortcutKeys(os.path.join(user_dir(), 'shortcut_keys.json'))   # Windows' Sticky / Filter
    if pygame.key.get_focused():    # Keys shortcuts: off only while the window has the focus (issue #1)
        shortcuts.disable()
    calm_ime()
    while not game.quit:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                game.quit = True
            elif ev.type == pygame.KEYDOWN:
                if game.cheats and cheat_key(game, ev.key):
                    pass
                elif ev.key in (pygame.K_f, pygame.K_F11):
                    settings.set('fullscreen', not settings.fullscreen)
                    settings.save()
                elif ev.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    pressed.add('action')
                elif ev.key == pygame.K_ESCAPE:
                    pressed.add('back')
                elif ev.key == pygame.K_m:
                    pressed.add('map')
                elif ev.key == pygame.K_j:
                    pressed.add('journal')
                elif ev.key == pygame.K_p:
                    pressed.add('postcard')
                elif ev.key == pygame.K_F12:
                    pressed.add('shot')
                elif ev.key == pygame.K_h:
                    game.hud_hidden = not game.hud_hidden
            elif ev.type == pygame.JOYBUTTONDOWN and ev.button in PAD_BUTTONS:
                pressed.add(PAD_BUTTONS[ev.button])
            elif ev.type == pygame.JOYDEVICEADDED:
                js = pygame.joystick.Joystick(ev.device_index)
                js.init()
                pads.append(js)
                print('gamepad:', js.get_name())
            elif ev.type == getattr(pygame, 'WINDOWFOCUSLOST', -1):
                # the player went to another window (or a screenshot tool took the focus): the world
                # freezes where it is, with nothing drawn over it, and the sound holds; it runs on when
                # the window gets the focus back (ESC still opens the pause menu). GitHub issue #2: the
                # pause menu used to open here and covered the view the player wanted to capture.
                shortcuts.restore()
                if not away:
                    away = True
                    held_alpha = min(1.0, acc / STEP)
                    audio.pause_all(True)
            elif ev.type == getattr(pygame, 'WINDOWFOCUSGAINED', -1):
                shortcuts.disable()
                calm_ime()
                if away:
                    away = False
                    if not hidden:
                        audio.pause_all(False)
                    last = time.perf_counter()
                    acc = held_alpha * STEP
            elif ev.type == getattr(pygame, 'WINDOWMINIMIZED', -1):
                hidden = True
                audio.pause_all(True)
            elif ev.type in (getattr(pygame, 'WINDOWRESTORED', -1), getattr(pygame, 'WINDOWSHOWN', -1)):
                if hidden:
                    hidden = False
                    if not away:
                        audio.pause_all(False)
                    last = time.perf_counter()
        if hidden:                                      # minimised: no drawing, no clock; ~10 checks a second
            clock.tick(10)
            continue
        if away:                                        # frozen: redraw the same view, ~30 times a second
            last = time.perf_counter()
            buf, sx, sy = compose_window(game.render(held_alpha), disp.k)
            disp.show(buf, sx, sy)
            pygame.display.flip()
            clock.tick(30)
            continue
        now = time.perf_counter()
        dt = now - last
        last = now
        # a slow machine catches up (at most 6 steps); after a real stall (a place loading, the OS busy)
        # the world just waits instead of rushing ahead, so nothing on screen jumps
        acc = min(acc + dt, (1 if dt > STALL else 6) * STEP)
        while acc >= STEP:
            game.step(held_input(pygame.key.get_pressed(), pressed, pads))
            steps_run += 1
            pressed -= {'action', 'back', 'map', 'journal', 'pause'}
            for name, gain, variant in game.sounds:
                audio.play(name, gain, variant)
            acc -= STEP
        if settings.version != applied:              # the menus changed a setting
            applied = settings.version
            audio.set_volumes(settings.music_gain, settings.sound_gain)
            disp.apply()
        audio.update(music_wanted(game), game.music_spans, min(dt, 2 * STEP), dip=game.journal is not None or game.menu is not None,
                     hour=game.hour)
        nxt = upcoming_place(game)                    # decode a place's theme while the balloon lands there
        if nxt is not None:
            audio.prefetch(nxt)
        if game.open_photos:                          # the journal: the postcards folder, or one postcard
            if isinstance(game.open_photos, str) and os.path.isfile(game.open_photos):
                reveal_file(game.open_photos)
            else:
                open_folder(photos)
            game.open_photos = False
        buf, sx, sy = compose_window(game.render(acc / STEP), disp.k)
        frame = disp.show(buf, sx, sy)
        if 'postcard' in pressed and game.journal is None and game.menu is None and not game.on_title:
            from game import postcard
            try:
                print('postcard saved:', postcard.save(game, photos))
            except OSError as e:                        # Pictures not writable: the game's own folder
                print('postcard: could not write to', photos, e, '- using', shots)
                photos = shots
                game.photo_label = shots.upper()
                print('postcard saved:', postcard.save(game, photos))
            n = len(game.sounds)                     # this step's sounds have already played
            game.snapshot_flash()
            for name, gain, variant in game.sounds[n:]:   # the shutter (sounds only play after a step,
                audio.play(name, gain, variant)          # and each step starts a new list: no replay)
        pressed.discard('postcard')
        if 'shot' in pressed:
            pressed.discard('shot')
            os.makedirs(shots, exist_ok=True)
            path = os.path.join(shots, f'shot_{int(time.time())}_{game.t:06d}.png')
            pygame.image.save(frame.subsurface(pygame.Rect(sx, sy, disp.size[0], disp.size[1])), path)
            print('saved', path)
        pygame.display.flip()
        if not capped and dt < 1 / 150:               # vsync was promised but frames come too fast
            fast += 1
            capped = fast > 60
        elif dt >= 1 / 150:
            fast = 0
        if capped:
            clock.tick(120 if disp.vsync else 60)
        if args.frames and steps_run >= args.frames:
            game.quit = True
    shortcuts.restore()


def music_wanted(game):
    """The music for this moment: the game's choice (game.music), or the race theme while a ring race or
    a balloon friend's race is on."""
    races = getattr(game, 'races', None)
    flock = getattr(game, 'flock', None)
    if game.music != FINALE_SEG and not game.on_title and \
            ((races is not None and races.run is not None) or (flock is not None and flock.race is not None)):
        return 'race'
    return game.music


def upcoming_place(game):
    """The place whose theme will be wanted next: the one the balloon hovers to land at, or is landing at."""
    m = getattr(game, 'map', None)
    if m is None or game.mode is not m:
        return None
    if m.landing is not None:
        return m.landing[0]
    return m.near


if __name__ == '__main__':
    main()
