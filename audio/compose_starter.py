"""Starter score — a simple, pleasant placeholder soundtrack that follows engine/timeline.py.

    .venv/bin/python audio/compose_starter.py     # -> audio/out/soundtrack.wav + sections.json

It exists so the pipeline (mix + full render) works from day one. The composer
replaces it with a real arrangement in audio/compose.py (see the skill's
references/music.md and references/examples/compose_example.py), writing the same
outputs: soundtrack.wav (exact timeline length), sections.json (chords + hits with
local frames, consumed by docs/MUSIC.md and the SFX key-fit), events.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audio import synth as S  # noqa: E402
from engine import timeline as TL  # noqa: E402
from engine.core import FPS, FRAMES_PER_BAR, FRAMES_PER_BEAT  # noqa: E402

OUT = os.path.join(ROOT, 'audio', 'out')
SPB = S.SPB
TOTAL = int(round(TL.DURATION * S.SR))

# D major: I  V  vi  IV  (root midi, chord tones as offsets)
PROG = [('D', 50, (0, 4, 7)), ('A', 45, (0, 4, 7)), ('Bm', 47, (0, 3, 7)), ('G', 43, (0, 4, 7))]
QUIET = {'dawn', 'end', 'lullaby'}


def add(bus, x, at, pan=0.0, gain=1.0):
    if x.ndim == 1:
        gl, gr = S.pan_gains(pan)
        x = np.stack([x * gl, x * gr])
    n = min(x.shape[1], bus.shape[1] - at)
    if n > 0:
        bus[:, at:at + n] += x[:, :n] * gain


def main():
    os.makedirs(OUT, exist_ok=True)
    bus = np.zeros((2, TOTAL))
    sections, events = [], []
    for s in TL.SEGMENTS:
        beat0 = s['start'] // FRAMES_PER_BEAT
        quiet = s.get('mood') in QUIET
        chords = []
        for b in range(s['bars']):
            name, root, tones = PROG[b % 4]
            gb = beat0 + b * 4
            chords.append({'bar': b + 1, 'beat': 1.0, 'chord': name, 'local_frame': b * FRAMES_PER_BAR})
            for t in tones:                                    # pad
                add(bus, S.render_note('padsoft', root + 12 + t, 2.3, 0.5, seed=gb + t), gb * SPB, gain=0.9)
            for k in range(4):                                 # bass on beats
                add(bus, S.render_note('bass', root - 12 + (7 if k == 2 else 0), 0.5, 0.6), (gb + k) * SPB)
            for k in range(8):                                 # arpeggio on 8ths
                m = root + 24 + tones[k % 3] + (12 if k >= 4 else 0)
                add(bus, S.render_note('arp', m, 0.25, 0.5, seed=k), (gb * 2 + k) * SPB // 2,
                    pan=0.3 if k % 2 else -0.3, gain=0.6 if quiet else 1.0)
                events.append({'section': s['id'], 'voice': 'arp', 'start_beat': gb + k / 2,
                               'dur_beats': 0.5, 'midi': m, 'velocity': 0.5})
            if not quiet:                                      # drums
                for k in range(4):
                    add(bus, S.drum('kick_soft' if k % 2 == 0 else 'snare_soft', 0.7), (gb + k) * SPB, gain=0.8)
                for k in range(8):
                    add(bus, S.drum('hat_soft', 0.4, seed=k), (gb * 2 + k) * SPB // 2, gain=0.5)
        sections.append({'id': s['id'], 'start_s': s['start'] / FPS, 'end_s': s['end'] / FPS, 'bars': s['bars'],
                         'mood': s.get('mood', ''), 'key': 'D major', 'tempo_bpm': S.BPM,
                         'start_frame': s['start'], 'end_frame': s['end'], 'start_sample': beat0 * SPB,
                         'chords': chords, 'hits': []})
    fade = np.linspace(1, 0, S.SR * 2)                         # 2 s fade at the very end
    bus[:, -fade.size:] *= fade
    bus *= 10 ** ((-16 - S.lufs(bus)) / 20)                    # loudness ~ -16 LUFS
    bus, _ = S.limiter(bus, -1.3)
    S.write_wav(os.path.join(OUT, 'soundtrack.wav'), S.to_int16(bus))
    json.dump(sections, open(os.path.join(OUT, 'sections.json'), 'w'), indent=1)
    json.dump(events, open(os.path.join(OUT, 'events.json'), 'w'))
    print(f'wrote audio/out/soundtrack.wav ({TOTAL / S.SR:.1f} s, {S.lufs(bus):.1f} LUFS) + sections.json')


if __name__ == '__main__':
    main()
