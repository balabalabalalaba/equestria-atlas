"""Player settings (music and sound volume, window scale, fullscreen, tips), kept in out/game_settings.json.

The menus (game/menu.py) change a `Settings`; every change bumps `version`, and the pygame front end
(game/play.py) applies the new values when it sees a new version. No pygame in here.
"""
import json
import os

VOL_STEPS = 10                  # volume sliders: 0..10


class Settings:
    def __init__(self, path=None):
        self.path = path
        self.music = 7          # 0..VOL_STEPS
        self.sound = 9
        self.scale = 0          # window scale (art px -> window px); 0 = fit to the screen
        self.fullscreen = False
        self.tips = True        # the tutorial's tips (game/tutor.py)
        self.max_scale = 6      # the largest scale that fits the screen (set by the front end)
        self.version = 0
        self._saved = None
        if path and os.path.exists(path):
            self.load()

    def as_dict(self):
        return dict(music=self.music, sound=self.sound, scale=self.scale, fullscreen=self.fullscreen, tips=self.tips)

    def load(self):
        try:
            with open(self.path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            print('settings unreadable, using the defaults:', self.path)
            return
        if not isinstance(d, dict):
            print('settings damaged, using the defaults:', self.path)
            return

        def num(key, lo, hi):
            v = d.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                setattr(self, key, int(min(hi, max(lo, v))))
        num('music', 0, VOL_STEPS)
        num('sound', 0, VOL_STEPS)
        num('scale', 0, 32)
        for key in ('fullscreen', 'tips'):
            if isinstance(d.get(key), bool):
                setattr(self, key, d[key])
        self._saved = self.as_dict()

    def save(self):
        """Write the file if anything changed (atomic)."""
        if not self.path or self.as_dict() == self._saved:
            return
        tmp = self.path + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            with open(tmp, 'w') as fh:
                json.dump(self.as_dict(), fh, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:                # not writable: the settings still apply for this session
            print('could not save the settings:', e)
            return
        self._saved = self.as_dict()

    # -------------------------------------------------------------- values the front end applies
    @property
    def music_gain(self):
        return self.music / VOL_STEPS

    @property
    def sound_gain(self):
        return self.sound / VOL_STEPS

    def scales(self):
        """Window scale choices: 0 (fit) and 2..max_scale."""
        return [0] + list(range(2, max(2, self.max_scale) + 1))

    def set(self, name, value):
        if getattr(self, name) != value:
            setattr(self, name, value)
            self.version += 1
