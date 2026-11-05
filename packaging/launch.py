"""Entry point of the packaged game (PyInstaller). The game's own sources (engine/, game/, scenes/,
audio/) ship as plain files next to this launcher, so the game imports them exactly as it does when
run from the project; only the third-party libraries are frozen."""
import os
import sys

ROOT = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import importlib  # noqa: E402

importlib.import_module('game.play').main()
