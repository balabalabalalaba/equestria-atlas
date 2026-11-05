"""ONE GOOD DAY — pixel-art video engine.

Scenes typically do:  from engine import *
which exposes the canvas, colour, easing/timing helpers, sprites, fonts and fx.
Character, props and locations live in their own modules:
    from engine.character import draw_hero, draw_npc, ...
    from engine.props import ...
    from engine.loc_home import ...
"""
from .core import (W, H, FPS, BPM, FRAMES_PER_BEAT, FRAMES_PER_BAR, SCALE, Canvas, col, mix, shade,
                   clamp, lerp, ilerp, seg, smooth, ease_in, ease_out, ease_in_out, ease_out_back,
                   ease_out_bounce, pingpong, wave, beat, beat_phase, bar, on_beat, hash01, rng,
                   keyframes, bayer, BAYER4)
from .palette import PAL, SKIES
from .sprite import Sprite, sprite, sheet
from .font import text, text_width
from . import fx
from .timeline import seg_frames

__all__ = [
    'W', 'H', 'FPS', 'BPM', 'FRAMES_PER_BEAT', 'FRAMES_PER_BAR', 'SCALE', 'Canvas', 'col', 'mix',
    'shade', 'clamp', 'lerp', 'ilerp', 'seg', 'smooth', 'ease_in', 'ease_out', 'ease_in_out',
    'ease_out_back', 'ease_out_bounce', 'pingpong', 'wave', 'beat', 'beat_phase', 'bar',
    'on_beat', 'hash01', 'rng', 'keyframes', 'bayer', 'BAYER4', 'PAL', 'SKIES', 'Sprite',
    'sprite', 'sheet', 'text', 'text_width', 'fx', 'seg_frames',
]
