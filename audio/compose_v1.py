"""EQUESTRIA: THE WORLD ATLAS — the score (22 sections, 101 bars, 100 BPM) and the render/mix pipeline.

    .venv/bin/python audio/compose.py            # compose + render -> audio/out/soundtrack.wav, events.json, sections.json
    .venv/bin/python audio/compose.py --score    # compose only (fast): writes events.json + sections.json
    .venv/bin/python audio/compose.py --stems    # also print per-role / per-section levels (mix debugging)
    .venv/bin/python tools/music_check.py        # measure the result (length, levels, clicks, clashes ...)

Leitmotif "The Atlas Theme" (4 bars, written in D, restated in every section in a new costume):
    | A4 D5 E5 F#5~ A5 | B5~ A5 F#5 D5 | B4 E5 F#5 G5~ B5 | A5 G5 F#5 E5 |     (D | Bm | Em | A)
Every region section = bar 1 TRAVEL FIGURE (theme head over a rising arpeggio, IVadd9 -> V of the new
key, `arrive` pling at local frame 54, swell + riser into `reveal` crash at 72) + 3-4 region bars.
Sections are written "relative to D" (D major / D minor) and transposed by the section's key shift.

Light-weight helpers for scenes (no numpy/scipy import, JSON cached):

    from audio.compose import hits, hit_frame, note_frames, beat_frames
    hits('canterlot')                       -> [{'name': 'arrive', 'local_frame': 54, ...}, ...]
    hit_frame('canterlot', 'sunrise')       -> 144
    note_frames('ponyville', voice='lead')  -> [(local_frame, midi, dur_frames, velocity), ...]
    beat_frames('laspegasus', voice='kick') -> [local_frame, ...]

Timing: 100 BPM, 4/4. 1 beat = 0.6 s = 18 video frames = 28 800 samples. 1 bar = 72 frames.
Global beat b starts at sample b * 28800 exactly; global frame = beat * 18.
"""
import json
import math
import os
import re
import sys
from functools import lru_cache
from itertools import combinations, product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'audio', 'out')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BPM = 100
SR = 48000
SPB = 28800                    # samples per beat
FPB = 18                       # video frames per beat
TOTAL_BARS = 101
TOTAL_SAMPLES = TOTAL_BARS * 4 * SPB   # 11 635 200 = 242.4 s

# --------------------------------------------------------------------------- section table
# (id, bars, mood, key label, tonic, mode)  mirrors engine/timeline.py (verified at compose time)
SECTIONS = [
    ('intro', 6, 'overture', 'D major', 'D', 'major'),
    ('canterlot', 5, 'regal fanfare', 'D major', 'D', 'major'),
    ('cloudsdale', 5, 'airy, soaring', 'A major', 'A', 'major'),
    ('ponyville', 5, 'cozy folk', 'G major', 'G', 'major'),
    ('sweetapple', 4, 'country hoedown', 'C major', 'C', 'major'),
    ('everfree', 4, 'mysterious', 'D dorian', 'D', 'minor'),
    ('heartland', 4, 'adventurous', 'F major', 'F', 'major'),
    ('rainbowfalls', 4, 'sporty, bright', 'Bb major', 'Bb', 'major'),
    ('manehattan', 5, 'big-city swing', 'Eb major', 'Eb', 'major'),
    ('eastcoast', 4, 'maritime shuffle', 'G major', 'G', 'major'),
    ('griffonstone', 4, 'rugged minor', 'E minor', 'E', 'minor'),
    ('dragonlands', 4, 'heavy riff', 'C# minor', 'C#', 'minor'),
    ('mountaris', 5, 'mediterranean -> underwater', 'A major', 'A', 'major'),
    ('klugetown', 4, 'ragtag bazaar', 'D harmonic minor', 'D', 'minor'),
    ('badlands', 4, 'dark -> blooming', 'F# minor -> F# major', 'F#', 'minor'),
    ('frontier', 4, 'western twang', 'E major', 'E', 'major'),
    ('laspegasus', 5, 'showtime funk', 'Ab major', 'Ab', 'major'),
    ('westcoast', 4, 'lo-fi rain', 'Db major', 'Db', 'major'),
    ('crystal', 5, 'majestic shimmer', 'B major', 'B', 'major'),
    ('yakyakistan', 4, 'stomping folk', 'G minor', 'G', 'minor'),
    ('twosisters', 5, 'lullaby -> climax', 'D major', 'D', 'major'),
    ('outro', 7, 'finale', 'D major -> E major', 'D', 'major'),
]
_PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9,
       'Bb': 10, 'B': 11}
FLAT_KEYS = {'F', 'Bb', 'Eb', 'Ab', 'Db'}


def _section_table():
    t, bar = [], 0
    for sid, bars, mood, key, tonic, mode in SECTIONS:
        k = (_PC[tonic] - 2 + 6) % 12 - 6          # transposition from "relative D" to the real key
        flat = tonic in FLAT_KEYS or (mode == 'minor' and tonic in ('D', 'G'))
        t.append(dict(id=sid, bars=bars, mood=mood, key=key, start_bar=bar, start_beat=bar * 4,
                      end_beat=(bar + bars) * 4, k=k, mode=mode, flat=flat,
                      region=sid not in ('intro', 'outro')))
        bar += bars
    assert bar == TOTAL_BARS
    return t


SEC = _section_table()
SEC_BY_ID = {s['id']: s for s in SEC}


def section_of_beat(b):
    for s in SEC:
        if b < s['end_beat'] - 1e-9:
            return s['id']
    return SEC[-1]['id']


# --------------------------------------------------------------------------- pitch / chords
_NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def P(name):
    """'C#5' -> 73, 'Bb4' -> 70, 'A4' -> 69."""
    m = re.fullmatch(r'([A-G])([#b]?)(-?\d)', name.strip())
    if not m:
        raise ValueError(f'bad pitch {name!r}')
    pc = _NOTE[m.group(1)] + {'#': 1, 'b': -1, '': 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + pc


def pname(m):
    m = int(round(m))
    return f'{_NAMES[m % 12]}{m // 12 - 1}'


QUAL = {
    '': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'maj7': (0, 4, 7, 11), 'm7': (0, 3, 7, 10),
    'm9': (0, 3, 7, 10, 14), 'maj9': (0, 4, 7, 11, 14), '9': (0, 4, 7, 10, 14), '13': (0, 4, 7, 10, 14, 21),
    '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9), 'add9': (0, 4, 7, 14), 'madd9': (0, 3, 7, 14), 'sus4': (0, 5, 7),
    '7sus4': (0, 5, 7, 10), '9sus4': (0, 5, 7, 10, 14), 'm7b5': (0, 3, 6, 10), '7b9': (0, 4, 7, 10, 13),
    'maj7#11': (0, 4, 7, 11, 18), '69': (0, 4, 7, 9, 14), 'maj13': (0, 4, 7, 11, 14, 21),
}


class Chord:
    __slots__ = ('sym', 'root', 'ivs', 'pcs', 'bass_pc')

    def __init__(self, sym):
        m = re.fullmatch(r'([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?', sym)
        if not m or m.group(2) not in QUAL:
            raise ValueError(f'bad chord {sym!r}')
        self.sym = sym
        self.root = P(m.group(1) + '0') % 12
        self.ivs = QUAL[m.group(2)]
        self.pcs = frozenset((self.root + i) % 12 for i in self.ivs)
        self.bass_pc = P(m.group(3) + '0') % 12 if m.group(3) else self.root

    def tones(self, lo, hi):
        return [m for m in range(int(lo), int(hi) + 1) if m % 12 in self.pcs]

    def classify(self, midi):
        """'chord' | 'tension' | 'avoid' for a melody pitch against this chord."""
        pc = int(round(midi)) % 12
        if pc in self.pcs:
            return 'chord'
        if (pc - 1) % 12 in self.pcs:
            return 'avoid'
        return 'tension'

    def __repr__(self):
        return self.sym


def _iv_priority(iv):
    """Voicing priority of a chord interval (lower = more important)."""
    r = iv % 12
    if r in (3, 4):
        return 0          # 3rd
    if r in (10, 11):
        return 1          # 7th
    if r in (5,) and iv < 12:
        return 1          # sus4
    if iv > 12 or r in (9, 2, 6) and iv not in (6,):
        return 2          # extensions / 6
    if r == 6:
        return 1          # b5
    if r == 7:
        return 4          # 5th
    return 3              # root


def voicing(ch, lo, hi, n=4, prev=None, want_root=True):
    """Pick an n-note voicing of chord `ch` inside [lo, hi] with smooth voice leading from `prev`."""
    ivs = sorted(set(ch.ivs), key=_iv_priority)
    pcs_all = []
    for iv in ivs:
        pc = (ch.root + iv) % 12
        if pc not in pcs_all:
            pcs_all.append(pc)
    k = min(n, len(pcs_all))
    best, best_s = None, 1e9
    centre = (lo + hi) / 2
    subsets = [(s_, 0.0) for s_ in combinations(pcs_all, k)]
    if k >= 3:                                        # thinner voicings are allowed (small penalty)
        subsets += [(s_, 2.5) for s_ in combinations(pcs_all, k - 1)]
    for subset, sp in subsets:
        if pcs_all[0] not in subset:
            sp += 6                                   # missing 3rd
        if ch.root not in subset and want_root:
            sp += 1.2
        if len(pcs_all) > 1 and pcs_all[1] not in subset:
            sp += 2
        opts = [[m for m in range(lo, hi + 1) if m % 12 == pc] for pc in subset]
        if any(not o for o in opts):
            continue
        for combo in product(*opts):
            s = sorted(combo)
            sc = sp
            for a, b in zip(s, s[1:]):
                d = b - a
                if d <= 1:
                    sc += 40                          # minor 2nd inside a voicing: never in a pad
                elif d == 2 and a < 55:
                    sc += 3
                elif d <= 4 and a < 50:
                    sc += 2                           # muddy close intervals in the low register
            for a, b in combinations(s, 2):
                if b - a == 13:
                    sc += 4
            span = s[-1] - s[0]
            if span > 16:
                sc += (span - 16) * 0.8
            if prev and len(prev) == len(s):
                sc += 0.5 * sum(abs(x - y) for x, y in zip(s, sorted(prev)))
            else:
                sc += 0.35 * abs(sum(s) / len(s) - centre)
            if sc < best_s:
                best, best_s = s, sc
    return best




# --------------------------------------------------------------------------- voices / mix table
DRUM_GM = {'kick': 36, 'kick_soft': 36, 'kick_lofi': 36, 'snare': 38, 'snare_soft': 38, 'snare_lofi': 38,
           'brush': 40, 'clap': 39, 'rim': 37, 'hat': 42, 'hat_soft': 42, 'ohat': 46, 'shaker': 70, 'crash': 49,
           'crash_soft': 57, 'ride': 51, 'tom_hi': 50, 'tom_mid': 47, 'tom_lo': 45, 'boom': 35, 'tick': 75}
DRUM_VOICE = {'kick_soft': 'kick', 'kick_lofi': 'kick', 'snare_soft': 'snare', 'snare_lofi': 'snare',
              'hat_soft': 'hat', 'crash_soft': 'crash', 'tom_hi': 'tom', 'tom_mid': 'tom', 'tom_lo': 'tom'}
DRUM_MIX = {  # pan, reverb send
    'kick': (0.0, 0.02), 'snare': (0.0, 0.12), 'clap': (0.05, 0.16), 'rim': (-0.22, 0.12), 'hat': (0.34, 0.04),
    'ohat': (0.36, 0.08), 'shaker': (-0.45, 0.06), 'crash': (-0.12, 0.12), 'ride': (0.3, 0.06),
    'tom': (0.0, 0.14), 'boom': (0.0, 0.14), 'tick': (0.35, 0.1), 'brush': (0.1, 0.12)}

# extra instrument presets (registered into audio.synth.INSTRUMENTS at render time)
CUSTOM_INST = {
    'harpsi':   dict(kind='chip', wave='pulse', duty=0.125, wave2='pulse', duty2=0.25, oct2=1, mix2=0.35, a=0.002,
                     d=0.22, s=0.0, r=0.05, lp=7000, gain=0.15, fenv=0.6, fenv_lo=1800, fenv_tau=0.05),
    'fiddle':   dict(kind='chip', wave='saw', wave2='pulse', duty2=0.25, mix2=0.3, a=0.02, d=0.3, s=0.8, r=0.08,
                     vib=22, vib_rate=6.2, vib_delay=0.12, glide=0.03, lp=3600, gain=0.32),
    'banjo':    dict(kind='chip', wave='pulse', duty=0.125, a=0.002, d=0.11, s=0.0, r=0.03, lp=6500,
                     gain=0.15, fenv=0.8, fenv_lo=1100, fenv_tau=0.025),
    'kalimba':  dict(kind='mbox', tau=0.55, gain=0.34, pulse_mix=0.04),
    'accordion': dict(kind='chip', wave='pulse', duty=0.33, wave2='pulse', duty2=0.5, det2=11, mix2=0.45,
                      a=0.025, d=0.2, s=0.9, r=0.08, vib=6, vib_rate=5.5, lp=3200, gain=0.2),
    'mando':    dict(kind='chip', wave='pulse', duty=0.2, a=0.002, d=0.08, s=0.15, r=0.03, lp=5200,
                     gain=0.12, fenv=0.6, fenv_lo=1400, fenv_tau=0.03),
    'twang':    dict(kind='chip', wave='pulse', duty=0.125, wave2='saw', mix2=0.25, a=0.004, d=0.4, s=0.55,
                     r=0.08, vib=26, vib_rate=5.2, vib_delay=0.2, lp=5200, gain=0.25),
    'whistle':  dict(kind='chip', wave='sine', wave2='tri', mix2=0.25, a=0.03, d=0.4, s=0.9, r=0.1, vib=24,
                     vib_rate=5.6, vib_delay=0.12, glide=0.05, gain=0.42),
    'choir':    dict(kind='pad', waves=('tri', 'saw', 'tri'), detune=(-6, 0, 7), a=0.45, d=1.2, s=0.9, r=1.0,
                     lp=1500, hp=140, gain=0.12),
    'timp':     dict(kind='chip', wave='tri', wave2='sine', mix2=0.5, a=0.003, d=0.55, s=0.0, r=0.2, lp=900,
                     hp=30, gain=0.8, bend=-0.12, bend_time=0.25),
    'reed':     dict(kind='chip', wave='pulse', duty=0.18, wave2='tri', mix2=0.4, a=0.02, d=0.3, s=0.8, r=0.08,
                     vib=28, vib_rate=6.0, vib_delay=0.1, glide=0.04, lp=3000, gain=0.3),
}

VOICES = {
    # role: melody (protected, checked vs chords) | harmony (protected) | accomp (auto-avoids clashes) | bass
    'lead':    dict(inst='lead', pan=0.0, echo=0.20, rev=0.14, role='melody', gate=0.96, glide=True),
    'head':    dict(inst='lead', pan=0.0, echo=0.22, rev=0.16, role='melody', gate=0.95, glide=True),
    'mbox':    dict(inst='mbox', pan=0.08, echo=0.20, rev=0.40, role='melody', gate=1.0),
    'brass':   dict(inst='brass', pan=-0.08, echo=0.10, rev=0.18, role='melody', gate=0.9),
    'whistle': dict(inst='whistle', pan=0.2, echo=0.2, rev=0.25, role='melody', gate=0.95),
    'lead2':   dict(inst='thin', pan=0.45, echo=0.12, rev=0.18, role='harmony', gate=0.94),
    'brass2':  dict(inst='brass', pan=0.3, echo=0.10, rev=0.18, role='harmony', gate=0.9),
    'glock':   dict(inst='glock', pan=0.35, echo=0.16, rev=0.32, role='ornament', gate=1.0),
    'pling':   dict(inst='glock', pan=0.0, echo=0.25, rev=0.35, role='ornament', gate=1.0),
    'bell':    dict(inst='bell', pan=0.0, echo=0.0, rev=0.40, role='ornament', gate=1.0),
    'blip':    dict(inst='blip', pan=0.0, echo=0.28, rev=0.18, role='ornament', gate=0.8),
    'arp':     dict(inst='arp', pan=0.0, echo=0.22, rev=0.14, role='accomp', bus='duck', gate=0.85),
    'triarp':  dict(inst='triarp', pan=-0.38, echo=0.10, rev=0.32, role='accomp', gate=0.95),
    'harpsi':  dict(inst='harpsi', pan=0.3, echo=0.08, rev=0.2, role='accomp', gate=0.9),
    'banjo':   dict(inst='banjo', pan=0.35, echo=0.05, rev=0.1, role='accomp', gate=0.9),
    'mando':   dict(inst='mando', pan=0.32, echo=0.05, rev=0.2, role='accomp', gate=0.9),
    'pad':     dict(inst='pad', pan=0.0, echo=0.0, rev=0.28, role='accomp', bus='duck', gate=1.0),
    'choir':   dict(inst='choir', pan=0.0, echo=0.0, rev=0.4, role='accomp', gate=1.0),
    'keys':    dict(inst='epiano', pan=-0.15, echo=0.0, rev=0.20, role='accomp', gate=0.95),
    'stab':    dict(inst='pluck', pan=0.42, echo=0.06, rev=0.10, role='accomp', bus='duck', gate=0.5),
    'power':   dict(inst='power', pan=-0.25, echo=0.0, rev=0.06, role='accomp', bus='duck', gate=0.7),
    'bass':    dict(inst='bass', pan=0.0, echo=0.0, rev=0.02, role='bass', bus='duck', gate=0.88),
    'timp':    dict(inst='timp', pan=-0.1, echo=0.0, rev=0.2, role='bass', gate=1.0),
    'fx':      dict(inst=None, pan=0.0, echo=0.0, rev=0.2, role='fx', gate=1.0),
}
MELODY_ROLES = ('melody', 'harmony')
PITCHED_ROLES = ('melody', 'harmony', 'accomp', 'bass', 'ornament')


class Ev(dict):
    """One score event (a dict so it serialises straight to JSON)."""


# --------------------------------------------------------------------------- the score builder
class Score:
    def __init__(self):
        self.events = []
        self.chords = []            # (beat0, beat1, Chord, section)
        self.hits = []              # dicts
        self.sec = None
        self.duck = []              # (beat0, beat1, depth)

    # ---- section context
    def section(self, sid):
        self.sec = SEC_BY_ID[sid]
        return self

    def at(self, bar, beat=0.0):
        """Global beat of local (0-based) bar / beat of the current section."""
        return self.sec['start_beat'] + bar * 4 + beat

    # ---- primitive
    def add(self, voice, beat, dur, midi, vel=0.8, **opts):
        v = VOICES.get(voice, {})
        beat = round(float(beat) * SPB) / SPB            # every onset lands on an exact sample
        e = Ev(section=self.sec['id'], voice=voice, start_beat=round(beat, 7),
               dur_beats=round(float(dur), 6), midi=midi, velocity=round(float(min(1.0, max(0.0, vel))), 3))
        e['_o'] = dict(opts)
        e['_o'].setdefault('inst', v.get('inst'))
        self.events.append(e)
        return e

    def drum(self, name, beat, vel=0.8, **opts):
        e = self.add(DRUM_VOICE.get(name, name), beat, 0.25, DRUM_GM[name], vel, inst='drum:' + name, **opts)
        e['drum'] = name
        return e

    def fx(self, name, beat, dur, vel=0.8, ends_at=None, **opts):
        """FX event; with ends_at=global beat the effect is placed so that it ENDS there (risers, swells)."""
        if ends_at is not None:
            beat = ends_at - dur
        e = self.add('fx', beat, dur, None, vel, inst='fx:' + name, **opts)
        e['fx'] = name
        return e

    def hit(self, name, bar, beat, desc):
        g = self.at(bar, beat)
        self.hits.append(dict(section=self.sec['id'], name=name, desc=desc, bar=bar + 1, beat=beat + 1,
                              global_beat=g))

    # ---- melody notation
    def mel(self, voice, bar, spec, unit=0.5, vel=0.8, tr=0, swing=None, late=0.0, check=True, shape=True,
            **opts):
        """Write a line in compact notation starting at local `bar` (beat 0).

        Tokens: PITCH[:dur][!][_][~] or r[:dur]; '|' separates bars (checked to sum to 4 beats).
        dur is in `unit` beats (default eighths), may be '3', '1.5' or '2/3'.  '!' accent, '_' staccato.
        shape=True adds gentle metric phrasing (downbeats lean in, off-beats lighter, long notes sing).
        """
        t = self.at(bar)
        out = []
        for bi, chunk in enumerate(spec.split('|')):
            acc = 0.0
            for tok in chunk.split():
                m = re.fullmatch(r'([A-Gr][#b]?-?\d?)(?::([\d./]+))?([!_~]*)', tok)
                if not m:
                    raise ValueError(f'bad token {tok!r} in {spec!r}')
                d = m.group(2) or '1'
                if '/' in d:
                    a_, b_ = d.split('/')
                    d = float(a_) / float(b_)
                d = float(d) * unit
                if m.group(1) != 'r':
                    mods = m.group(3)
                    vv = vel + (0.12 if '!' in mods else 0.0)
                    if shape:
                        pb = (t + acc - self.at(0)) % 4
                        vv *= (1.0 if abs(pb) < 1e-6 else 0.97 if abs(pb - 2) < 1e-6 else
                               0.94 if abs(pb - round(pb)) < 1e-6 else 0.9) * (1.03 if d >= 1.0 else 1.0)
                    o = dict(opts)
                    if '_' in mods:
                        o['gate'] = 0.5
                    pos = t + acc
                    if swing:
                        pos = self.at(0) + swing_pos(pos - self.at(0), swing)
                    e = self.add(voice, pos + late, d, P(m.group(1)) + tr, vv, **o)
                    out.append(e)
                acc += d
            if check and '|' in spec and abs(acc - 4.0) > 1e-6:
                raise ValueError(f'bar {bi} of {spec!r} lasts {acc} beats in {self.sec["id"]}')
            t += acc if '|' not in spec else 4.0
        return out

    # ---- harmony
    def progression(self, spec):
        """Chords for the whole current section: 'D | A | Em7:1 A7:1 D:2' (durations in beats)."""
        bars = [b.split() for b in spec.split('|')]
        assert len(bars) == self.sec['bars'], (self.sec['id'], len(bars))
        for i, toks in enumerate(bars):
            fixed = sum(float(t.split(':')[1]) for t in toks if ':' in t)
            free = [t for t in toks if ':' not in t]
            each = (4.0 - fixed) / len(free) if free else 0.0
            pos = self.at(i)
            for tok in toks:
                sym, _, d = tok.partition(':')
                d = float(d) if d else each
                self.chords.append((pos, pos + d, Chord(sym), self.sec['id']))
                pos += d
            assert abs(pos - self.at(i + 1)) < 1e-9, (self.sec['id'], i, spec)

    def chord_at(self, beat):
        for b0, b1, c, _ in self.chords:
            if b0 - 1e-9 <= beat < b1 - 1e-9:
                return c
        return self.chords[-1][2]

    def chord_end(self, beat):
        for b0, b1, c, _ in self.chords:
            if b0 - 1e-9 <= beat < b1 - 1e-9:
                return b1
        return self.chords[-1][1]

    def chord_spans(self, beat0, beat1):
        out = []
        for b0, b1, c, _ in self.chords:
            s, e = max(b0, beat0), min(b1, beat1)
            if e > s + 1e-9:
                out.append((s, e, c))
        return out

    # ---- generators
    def pad(self, voice, bar0, bar1, lo=50, hi=67, n=4, vel=0.5, tie=True, end_beat=None, **opts):
        prev = None
        last = {}
        b1 = self.at(bar1) if end_beat is None else end_beat
        for s, e, c in self.chord_spans(self.at(bar0), b1):
            v = voicing(c, lo, hi, n, prev)
            new_last = {}
            for m in v:
                pe = last.get(m)
                if tie and pe is not None and abs(pe['start_beat'] + pe['dur_beats'] - s) < 1e-6:
                    pe['dur_beats'] = round(e - pe['start_beat'], 6)
                    new_last[m] = pe
                else:
                    new_last[m] = self.add(voice, s, e - s, m, vel, **opts)
            last, prev = new_last, v

    def comp(self, voice, bar0, bar1, rhythm, lo=52, hi=69, n=4, vel=0.55, swing=None, human=0.0, **opts):
        """Chord hits: rhythm = [(pos_in_bar_beats, dur_beats, vel_mult)], voiced per current chord."""
        prev = None
        for bar in range(bar0, bar1):
            for k, (pos, dur, vm) in enumerate(rhythm):
                b = self.at(bar, pos)
                c = self.chord_at(b)
                v = voicing(c, lo, hi, n, prev)
                prev = v
                bb = self.at(bar) + (swing_pos(pos, swing) if swing else pos)
                cend = self.chord_end(b)
                for j, m in enumerate(v):
                    jit = human * _h(bar, k, j) * 0.025 + 0.006 * j * human   # laid-back: only ever late
                    d = max(0.1, min(dur, cend - (bb + jit) - 0.02))           # never ring into the next chord
                    self.add(voice, bb + jit, d, m, vel * vm * (0.92 + 0.08 * _h(bar, k, j, 7)), **opts)

    def arp(self, voice, beat0, beat1, step=0.25, lo=69, hi=88, vel=0.5, pattern='up', group=None,
            pans=(-0.55, 0.55), vel_ramp=None, lp_ramp=None, swing=None, reset=True, skip=(), **opts):
        t, i, last_c = beat0, 0, None
        k = 0
        while t < beat1 - 1e-9:
            c = self.chord_at(t)
            if reset and c is not last_c:
                i = 0
            last_c = c
            tones = c.tones(lo, hi)
            if group:
                idx = group[i % len(group)] % len(tones)
            elif pattern == 'up':
                idx = i % len(tones)
            elif pattern == 'down':
                idx = len(tones) - 1 - i % len(tones)
            else:  # updown
                cyc = max(1, 2 * len(tones) - 2)
                j = i % cyc
                idx = j if j < len(tones) else cyc - j
            x = (t - beat0) / max(beat1 - beat0, 1e-9)
            vv = vel * (vel_ramp[0] + (vel_ramp[1] - vel_ramp[0]) * x if vel_ramp else 1.0)
            o = dict(opts)
            if lp_ramp:
                o['lp'] = lp_ramp[0] * (lp_ramp[1] / lp_ramp[0]) ** x
            if pans:
                o['pan'] = pans[k % len(pans)]
            pos = t
            if swing:
                sb = math.floor(t - self.at(0) + 1e-9)
                pos = self.at(0) + sb + swing_pos(t - self.at(0) - sb, swing)
            if round((t - self.at(0)) / step) not in skip:
                self.add(voice, pos, step, tones[idx], vv, **o)
            t += step
            i += 1
            k += 1

    def bassline(self, voice, bar0, bar1, pattern, lo=33, vel=0.8, swing=None, **opts):
        """pattern = [(pos_beats, spec, dur_beats[, vel_mult])]; spec: 'R','R+12','5','5-12','3','b7','8' or int."""
        for bar in range(bar0, bar1):
            for item in pattern:
                pos, spec, dur = item[:3]
                vm = item[3] if len(item) > 3 else 1.0
                b = self.at(bar, pos)
                c = self.chord_at(b)
                root = lo + (c.bass_pc - lo) % 12
                third = 3 if 3 in c.ivs else 4
                iv = {'R': 0, 'R+12': 12, '8': 12, '5': 7, '5-12': -5, '3': third, 'b7': 10, '6': 9}.get(spec, spec)
                if spec in ('5', '5-12') and 6 in c.ivs and 7 not in c.ivs:
                    iv -= 1
                bb = self.at(bar) + (swing_pos(pos, swing) if swing else pos)
                self.add(voice, bb, dur, root + iv, vel * vm, **opts)

    def drums(self, bar0, bar1, pats, swing=None, swing16=0.0, human=0.0, vel=1.0):
        """pats: {drum_name: '16-step pattern per bar'} (X=1.0 x=.8 o=.55 g=.35 .=rest).
        A pattern of 32/48/64 steps spans 2/3/4 bars and repeats."""
        lv = {'X': 1.0, 'x': 0.8, 'o': 0.55, 'g': 0.35}
        for name, p in pats.items():
            p = p.replace(' ', '')
            nb = len(p) // 16
            assert len(p) % 16 == 0, (name, p)
            for bar in range(bar0, bar1):
                seg = p[((bar - bar0) % nb) * 16:((bar - bar0) % nb + 1) * 16]
                for st, ch in enumerate(seg):
                    if ch == '.':
                        continue
                    pos = st / 4
                    if swing:
                        pos = swing_pos(pos, swing)
                    if swing16 and st % 2 == 1:
                        pos += swing16
                    jit = human * (_h(bar, st, len(name)) - 0.5) * 0.03 if human and st else 0.0
                    vv = lv[ch] * vel * (1 - human * 0.15 * _h(st, bar, 3)) * (0.96 + 0.08 * _h(bar, st, 11))
                    self.drum(name, self.at(bar) + pos + jit, vv)

    def roll(self, name, beat0, beat1, step=0.25, v0=0.2, v1=0.8):
        n = int(round((beat1 - beat0) / step))
        for i in range(n):
            self.drum(name, beat0 + i * step, v0 + (v1 - v0) * i / max(1, n - 1))


def swing_pos(pos, ratio):
    """Map a straight position (beats) to its swung position: 8th offbeats move to `ratio` of the beat."""
    b = math.floor(pos + 1e-9)
    f = pos - b
    if abs(f - 0.5) < 1e-6:
        return b + ratio
    if abs(f - 0.25) < 1e-6:
        return b + ratio * 0.5
    if abs(f - 0.75) < 1e-6:
        return b + ratio + (1 - ratio) * 0.5
    return pos


def _h(*xs):
    h = 2166136261
    for x in xs:
        h ^= int(x * 1000003) & 0xffffffff
        h = (h * 16777619) & 0xffffffff
        h ^= h >> 13
    return (h & 0xffff) / 65536.0


def tr(spec, semis):
    """Transpose every pitch in a notation string by `semis` semitones."""
    def rep(m):
        return pname(P(m.group(0)) + semis)
    return re.sub(r'[A-G][#b]?-?\d', rep, spec)



def trc(spec, semis, flat=False):
    """Transpose the chord roots / slash basses of a progression spec ('Gadd9:3 A:1 | Bm7')."""
    names = _FLATS if flat else _NAMES

    def rep(m):
        return names[(P(m.group(0) + '4') + semis) % 12]
    return re.sub(r'[A-G][#b]?', rep, spec)


_FLATS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


class Atlas(Score):
    """Score with "relative to D" writing: S.k = semitones from D to the section's key."""

    @property
    def k(self):
        return self.sec['k']

    def prog(self, spec):
        self.progression(trc(spec, self.k, self.sec['flat']))

    def m(self, voice, bar, spec, o=0, **kw):
        return self.mel(voice, bar, spec, tr=self.k + o, **kw)

    def pp(self, name):
        return P(name) + self.k


# =========================================================================== the leitmotif
# "The Atlas Theme" (written in D; minor variant in D minor). 8th-note units.
TH = {1: 'A4 D5 E5 F#5:3 A5:2', 2: 'B5:3 A5 F#5:2 D5:2', 3: 'B4 E5 F#5 G5:3 B5:2', 4: 'A5:2 G5 F#5 E5:4',
      'c': 'G5:2 E5 C#5 D5:4'}
THM = {1: 'A4 D5 E5 F5:3 A5:2', 2: 'Bb5:3 A5 F5:2 D5:2', 3: 'G4 C5 D5 E5:3 G5:2', 4: 'A5:2 G5 F5 E5:4',
       'c': 'G5:2 E5 C#5 D5:4'}
TRAVEL, TRAVELM = 'Gadd9:3 A:1', 'Gmadd9:3 A:1'


def th(*bars, minor=False, o=0):
    src = THM if minor else TH
    return ' | '.join(tr(src[b], o) if o else src[b] for b in bars)


def tonic_midi(sec):
    pc = (2 + sec['k']) % 12
    return 67 + (pc - 7) % 12          # G4..F#5


# =========================================================================== the travel figure
def travel(S, lead='lead', arp='arp', night=False, crash='crash', cv=0.85, hv=0.7):
    """Bar 1 of every region: theme head (5-1-2-3) over IVadd9 -> V of the new key, a rising 16th
    arpeggio (left -> right), light percussion; ARRIVE pling at beat 4 (local frame 54), reverse
    swell + riser into the REVEAL crash on the next downbeat (local frame 72)."""
    minor = S.sec['mode'] == 'minor'
    t = tonic_midi(S.sec)
    third = 3 if minor else 4
    for pos, iv, d, v in [(0, -5, 0.5, 0.95), (0.5, 0, 0.5, 0.9), (1.0, 2, 0.5, 0.92), (1.5, third, 1.5, 1.0)]:
        S.add('head', S.at(0, pos), d, t + iv, hv * v, inst=lead)
        S.add('glock', S.at(0, pos), d, t + iv + 12, 0.26 if night else 0.2)
    for i in range(16):
        b = S.at(0, i * 0.25)
        c = S.chord_at(b)
        target = t - 14 + i * 2.0
        m = min(x for x in c.tones(40, 104) if x >= target)
        S.add(arp, b, 0.25, m, (0.26 if night else 0.3) + 0.24 * i / 15, pan=-0.55 + 1.1 * i / 15)
    c4, c5 = S.chord_at(S.at(0)), S.chord_at(S.at(0, 3))
    S.add('bass', S.at(0), 2.9, 36 + (c4.root - 36) % 12, 0.6)
    S.add('bass', S.at(0, 3), 0.95, 36 + (c5.root - 36) % 12, 0.64)
    S.pad('pad', 0, 1, lo=52, hi=67, n=3, vel=0.36, inst='padsoft' if night else 'pad')
    if night:
        S.drums(0, 1, {'shaker': 'g.g.g.g.g.g.g.g.', 'kick_soft': 'o.......o.......'})
    else:
        S.drums(0, 1, {'kick_soft': 'x.......o.......', 'shaker': 'gogogogogogogogo', 'hat_soft': '..g...g...g...g.'})
    p1 = 84 + (t - 84) % 12
    S.add('pling', S.at(0, 3), 1.5, p1, 0.55)
    S.add('pling', S.at(0, 3), 1.5, p1 + 7, 0.45)
    S.fx('shimmer', S.at(0, 3), 1.2, 0.32, f0=1760.0 * 2 ** (((t - 69) % 12) / 12))
    S.fx('swell', 0, 1.5, 0.5, ends_at=S.at(1))
    S.fx('riser', 0, 1.0, 0.3, ends_at=S.at(1), lo=700.0, hi=9000.0, f0=330.0, octs=2.0)
    S.drum(crash, S.at(1), cv)
    S.drum('kick' if crash == 'crash' else 'kick_soft', S.at(1), 0.7)
    S.hit('arrive', 0, 3, 'ARRIVE: bright glock pling (tonic + fifth, 2 octaves up) + shimmer; reverse '
                          'cymbal swell grows toward the reveal')
    S.hit('reveal', 1, 0, 'REVEAL: riser lands, crash + the region groove starts on the downbeat')


BOUNCE = [(0, 'R', 0.75), (1, 'R+12', 0.4, 0.75), (1.5, 'R', 0.4, 0.85), (2.5, '5', 0.4, 0.8),
          (3, 'R+12', 0.4, 0.75), (3.5, 'R', 0.4, 0.85)]
OCT8 = [(i * 0.5, 'R' if i % 2 == 0 else 'R+12', 0.42, 1.0 if i % 2 == 0 else 0.78) for i in range(8)]
OFF8 = [(0.5, 0.25, 0.8), (1.5, 0.25, 0.7), (2.5, 0.25, 0.8), (3.5, 0.25, 0.7)]


def fill(S, bar, beat=3, v=0.55, kind='tom'):
    names = ['tom_hi', 'tom_hi', 'tom_mid', 'tom_lo'] if kind == 'tom' else ['snare_soft'] * 4
    for i, nm in enumerate(names):
        S.drum(nm, S.at(bar, beat) + 0.25 * i, v + 0.06 * i)


# =========================================================================== SECTIONS
def s_intro(S):
    S.section('intro')
    S.prog('Dadd9 | Bm7 | Gmaj9 | A7sus4:2 A7:2 | D | Bm7:2 D7:2')
    # bar 1: night sky - the music box states the theme alone (continues through bar 4)
    S.m('mbox', 0, th(1, 2, 3, 4), vel=0.6)
    for i, nm in enumerate(['A7', 'F#7', 'E7', 'D7', 'A6']):
        S.add('glock', S.at(0, 2) + 0.125 * i, 0.5, P(nm), 0.2 - 0.025 * i, pan=0.5 - 0.22 * i)
    S.hit('star', 0, 2, 'shooting star: soft falling glock sparkle (beat 3)')
    # bar 2: strings + glock build
    S.pad('pad', 1, 4, lo=50, hi=69, n=4, vel=0.5, inst='padsoft')
    S.m('glock', 1, 'r:4 D6 F#6 A6 C#7', vel=0.3)
    S.m('bass', 1, 'B2:8 | G2:8 | A2:8', vel=0.42, inst='upright', gate=1.0)
    S.fx('swell', 0, 2.0, 0.7, ends_at=S.at(2))
    S.roll('tom_lo', S.at(1, 3), S.at(2), 0.125, 0.15, 0.5)
    # bar 3: TITLE
    S.drum('boom', S.at(2), 0.75)
    S.drum('crash', S.at(2), 0.85)
    S.fx('shimmer', S.at(2), 4.0, 0.6, f0=1760.0)
    S.add('timp', S.at(2), 2.0, P('G2'), 0.8)
    S.pad('pad', 2, 4, lo=55, hi=74, n=4, vel=0.38, inst='padbright')
    S.m('lead', 2, th(3, 4, o=-12), vel=0.55, inst='mellow')
    for i, nm in enumerate(['D6', 'G6', 'A6', 'B6', 'D7', 'B6', 'A6', 'G6', 'A6']):
        S.add('glock', S.at(2) + 0.5 * i, 0.5, P(nm), 0.5 if i in (0, 8) else 0.42, pan=0.5 * math.sin(i * 1.3))
    S.hit('title', 2, 0, 'TITLE DROP: boom + crash + shimmer; letter plings (glock) on every 8th: local frames '
                         '144,153,162,171,180,189,198,207,216 (E-Q-U-E-S-T-R-I-A, the 9th on the bar-4 downbeat)')
    # bar 4: dawn build
    S.add('timp', S.at(3), 1.0, P('A2'), 0.7)
    S.arp('triarp', S.at(3), S.at(4), step=0.25, lo=57, hi=86, pattern='up', vel=0.42, vel_ramp=(0.5, 1.0))
    S.roll('snare_soft', S.at(3, 3), S.at(4), 0.125, 0.2, 0.7)
    S.fx('swell', 0, 2.0, 0.55, ends_at=S.at(4))
    S.hit('dawn', 3, 0, 'dawn light sweeps the map: rising arpeggio build + snare roll into bar 5')
    # bars 5-6: begin journey
    S.drum('crash_soft', S.at(4), 0.7)
    S.m('lead', 4, th(1, 2), vel=0.72, inst='lead')
    S.m('lead2', 4, 'F#4 A4 C#5 D5:3 F#5:2 | F#5:3 F#5 D5:2 B4:2', vel=0.4)
    S.bassline('bass', 4, 6, BOUNCE, lo=36, vel=0.75)
    S.comp('stab', 4, 6, OFF8, lo=57, hi=69, n=3, vel=0.5)
    S.pad('pad', 4, 6, lo=50, hi=66, n=3, vel=0.3)
    S.drums(4, 6, {'kick_soft': 'x.....o.x.......', 'clap': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.',
                   'shaker': '.g.g.g.g.g.g.g.g'})
    for q in range(4):
        S.add('blip', S.at(4, q), 0.25, P('A6'), 0.36 if q == 0 else 0.3, pan=0.3)
    S.hit('prompt', 4, 0, 'BEGIN JOURNEY prompt: blip on every beat (288, 306, 324, 342)')
    S.fx('riser', 0, 1.0, 0.45, ends_at=S.at(5), lo=400.0, hi=8000.0, f0=250.0, octs=2.0)
    S.drum('kick', S.at(5), 0.8)
    S.drum('tick', S.at(5), 0.75)
    S.drum('crash_soft', S.at(5), 0.6)
    S.add('blip', S.at(5), 0.25, P('D7'), 0.45)
    for i, nm in enumerate(['D5', 'F#5', 'A5', 'B5', 'D6', 'F#6', 'A6', 'B6', 'D7']):
        S.add('glock', S.at(5) + 0.125 * i, 0.4, P(nm), 0.22 + 0.025 * i, pan=-0.5 + 0.12 * i)
    S.hit('liftoff', 5, 0, 'LIFTOFF: riser lands, pop (kick + tick + blip), whoosh-up glock glissando')
    S.roll('snare_soft', S.at(5, 3), S.at(6), 0.25, 0.4, 0.7)
    S.fx('swell', 0, 1.0, 0.4, ends_at=S.at(6))


def s_canterlot(S):
    S.section('canterlot')
    S.prog(TRAVEL + ' | Dsus4:2 D:2 | D | Bm7 | G:2 A7:2')
    travel(S, crash='crash_soft', cv=0.7)
    # bar 2: pre-dawn - harpsichord, timpani, a low fanfare call
    S.arp('harpsi', S.at(1), S.at(2), step=0.25, lo=50, hi=69, pattern='updown', vel=0.36, pans=(-0.3, 0.3))
    S.m('brass2', 1, 'D4:2 A3 D4 F#4:2 A4:2', vel=0.5)
    S.add('timp', S.at(1), 1.0, P('D2'), 0.6)
    S.add('timp', S.at(1, 2), 1.0, P('A1'), 0.55)
    S.roll('tom_lo', S.at(1, 2), S.at(2), 0.125, 0.15, 0.7)
    S.pad('pad', 1, 2, lo=50, hi=66, n=3, vel=0.34, inst='padsoft')
    S.m('bass', 1, 'D2:8', vel=0.45, inst='upright')
    S.fx('swell', 0, 2.0, 0.8, ends_at=S.at(2))
    # bar 3: SUNRISE - the theme in full brass
    S.drum('crash', S.at(2), 0.95)
    S.drum('boom', S.at(2), 0.7)
    S.fx('shimmer', S.at(2), 4.0, 0.55, f0=1760.0)
    S.hit('sunrise', 2, 0, 'SUNRISE: reverse swell lands on crash + boom; the Atlas Theme in full brass')
    S.m('brass', 2, th(1, 2, 4), vel=0.85)
    S.m('brass2', 2, 'F#4 A4 C#5 D5:3 F#5:2 | F#5:3 F#5 D5:2 B4:2 | D5:2 D5 D5 C#5:4', vel=0.56)
    S.m('glock', 2, th(1, o=12), vel=0.22)
    S.arp('harpsi', S.at(2), S.at(5), step=0.25, lo=50, hi=69, pattern='updown', vel=0.42, pans=(-0.35, 0.35))
    S.pad('pad', 2, 5, lo=50, hi=67, n=4, vel=0.45)
    S.bassline('timp', 2, 4, [(0, 'R', 1.0), (2, '5-12', 1.0, 0.8)], lo=38, vel=0.75)
    S.bassline('bass', 2, 5, [(0, 'R', 1.9), (2, 'R', 0.9, 0.85), (3, '5-12', 0.9, 0.8)], lo=38, vel=0.72)
    S.drums(2, 4, {'kick': 'x.......x.......', 'snare_soft': '....x.......x.gg', 'hat_soft': 'g.g.g.g.g.g.g.g.'})
    S.drums(4, 5, {'kick': 'x.......x.......', 'snare_soft': '....x...........', 'hat_soft': 'g.g.g.g.........'})
    S.roll('snare_soft', S.at(4, 2), S.at(5), 0.125, 0.25, 0.8)
    S.add('timp', S.at(4), 1.0, P('G2'), 0.75)
    S.add('timp', S.at(4, 2), 2.0, P('A2'), 0.8)


def s_cloudsdale(S):
    S.section('cloudsdale')
    S.prog(TRAVEL + ' | Dmaj7 | Bm7 | Gmaj7:2 A:2 | Bbmaj7:2 C:2')
    travel(S, crash='crash_soft', cv=0.75)
    S.m('lead', 1, th(1, 2, 3, o=12), vel=0.6, inst='lead50')
    S.m('glock', 1, th(1, 2, 3, o=12), vel=0.2)
    S.arp('triarp', S.at(1), S.at(4), step=0.5, lo=55, hi=74, pattern='updown', vel=0.42, pans=(-0.4, 0.4))
    S.arp('arp', S.at(2), S.at(4), step=0.25, lo=62, hi=79, pattern='up', vel=0.26, pans=(-0.6, 0.6))
    S.pad('pad', 1, 5, lo=52, hi=69, n=4, vel=0.4, inst='padbright')
    S.bassline('bass', 1, 5, [(0, 'R', 1.9), (2, '5-12', 0.9, 0.8), (3, 'R', 0.9, 0.85)], lo=36, vel=0.65)
    S.drums(1, 4, {'kick_soft': 'x.........x.....', 'snare_soft': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.',
                   'shaker': '.g.g.g.g.g.g.g.g'})
    # bar 5: RAINBOW glissando (short, so the HUD toast at frame 302 stays clear)
    S.drum('crash_soft', S.at(4), 0.75)
    for i, nm in enumerate(['Bb5', 'D6', 'F6', 'A6', 'Bb6', 'D7', 'F7', 'A7']):
        S.add('glock', S.at(4) + 0.0625 * i, 0.15, S.pp(nm), 0.28 + 0.03 * i, pan=-0.6 + 0.16 * i)
    S.fx('shimmer', S.at(4), 1.5, 0.4, f0=1400.0)
    S.hit('rainbow', 4, 0, 'RAINBOW: fast glissando sparkle (glock run up over half a beat, 288-297) + shimmer')
    S.m('lead', 4, 'D5:2 C5 A4 G4:2 E4:2', o=12, vel=0.6, inst='lead50')
    S.arp('arp', S.at(4, 1), S.at(5), step=0.25, lo=55, hi=74, pattern='up', vel=0.28)
    S.drums(4, 5, {'kick_soft': 'x.......x.......', 'snare_soft': '............x.x.', 'hat_soft': 'g.o.g.o.g.o.g.o.'})


def s_ponyville(S):
    S.section('ponyville')
    S.prog(TRAVEL + ' | D | Bm7 | Em7:2 C:2 | G:2 G7:2')
    travel(S)
    S.m('lead', 1, th(1, 2, 3, o=-12), vel=0.78, inst='ocarina')
    S.arp('stab', S.at(1), S.at(5), step=0.5, lo=67, hi=84, group=[0, 2, 1, 3], vel=0.46, pans=(0.35, 0.5),
          inst='pluck')
    S.bassline('bass', 1, 5, [(0, 'R', 0.9), (1, '5-12', 0.9, 0.8), (2, 'R', 0.9, 0.9), (3, '5-12', 0.9, 0.8)],
               lo=36, vel=0.75, inst='upright')
    S.pad('pad', 1, 5, lo=50, hi=65, n=3, vel=0.32, inst='padsoft')
    S.drums(1, 4, {'kick_soft': 'x.......x.......', 'clap': '....o.......o...', 'shaker': 'gogggogggogggogg'})
    S.m('lead2', 2, 'r:6 F#5 G5 | A5:2 r:6', o=-12, vel=0.36, inst='thin')
    # bar 5: TRAIN whistle (two notes) + chugging shaker, ocarina answers
    S.m('whistle', 4, 'D5:0.75 r:0.25 B4:1.0 r:2', unit=1.0, vel=0.72, check=False)
    S.m('lead', 4, 'r:4 A4 G4 F4 D4', vel=0.65, inst='ocarina')
    S.drums(4, 5, {'kick_soft': 'x.......x.......', 'shaker': 'xgogxgogxgogxgog', 'brush': '....o.......o...'})
    S.hit('train', 4, 0, 'TRAIN: two-note whistle (G5 then E5, "whoo-whooo") at 288 and 306; chugging shaker')


def s_sweetapple(S):
    S.section('sweetapple')
    S.prog(TRAVEL + ' | D | G:2 A7:2 | D:2 E7:2')
    travel(S)
    S.m('lead', 1, 'A4 D5 E5 F#5 E5 F#5 A5:2 | B5:3 A5 G5:2 E5 C#5', vel=0.8, inst='fiddle')
    S.m('lead2', 1, 'F#4 A4 C#5 D5 C#5 D5 F#5:2 | G5:3 F#5 E5:2 C#5 A4', vel=0.42, inst='fiddle')
    S.arp('banjo', S.at(1), S.at(3), step=0.25, lo=55, hi=74, group=[0, 2, 4, 1, 3, 5], vel=0.44, pans=(0.3, 0.45))
    S.bassline('bass', 1, 4, [(0, 'R', 0.9), (2, '5-12', 0.9, 0.85)], lo=36, vel=0.8)
    S.comp('stab', 1, 4, [(1, 0.4, 1.0), (3, 0.4, 0.95)], lo=55, hi=67, n=3, vel=0.55)
    S.drums(1, 3, {'kick': 'x.......x.......', 'clap': '....x.......x...', 'brush': '..o...o...o...o.',
                   'hat_soft': 'g.g.g.g.g.g.g.g.'})
    # bar 4: ZAP - rainbow arpeggio
    S.fx('zap', S.at(3), 0.8, 0.6, f0=3200.0, f1=400.0)
    S.drum('crash', S.at(3), 0.8)
    tones = S.chord_at(S.at(3)).tones(60, 88)
    seq = tones + tones[-2::-1]
    for i, m in enumerate(seq[:16]):
        S.add('blip', S.at(3) + 0.125 * i, 0.15, m, 0.42 if i < 8 else 0.3, inst='arp25', pan=-0.6 + 0.08 * i)
    S.hit('zap', 3, 0, 'ZAP: zappy rainbow arpeggio (pulse 32nds up and back down, 216-251) + laser zap')
    S.m('lead', 3, 'r:4 E5 G#5 B5 D6', vel=0.72, inst='fiddle')
    S.drums(3, 4, {'kick': 'x.......x.......', 'clap': '....x.......x...', 'hat_soft': 'g.g.g.g.g.g.....'})
    S.roll('snare_soft', S.at(3, 3), S.at(4), 0.125, 0.3, 0.75)


def s_everfree(S):
    S.section('everfree')
    S.prog(TRAVELM + ' | Dm9 | G7 | Gm7:2 A7b9:1 F7:1')
    travel(S, crash='crash_soft', cv=0.6, arp='triarp')
    S.m('lead', 1, th(1, minor=True) + ' | B5:3 A5 F5:2 D5:2 | Bb4:2 A4 F4 r:4', vel=0.74, inst='kalimba')
    S.m('lead2', 1, 'r:7 D6 | r:7 B5 | r:8', vel=0.3, inst='kalimba')
    S.pad('pad', 1, 4, lo=38, hi=57, n=3, vel=0.5, inst='padsoft', end_beat=S.at(3, 2))
    S.m('bass', 1, 'D2:8 | G2:8 | G2:4 A2:2 F2:2', vel=0.5, inst='upright')
    S.drums(1, 3, {'tom_lo': 'x.......o.......', 'tom_mid': '......o.......g.', 'shaker': 'g...g...g...g...',
                   'rim': '...........g....'}, vel=0.8)
    for bar, beat, nm in [(1, 1.5, 'D6'), (1, 3.5, 'A6'), (2, 0.5, 'B6'), (2, 2.5, 'F6'), (3, 0.5, 'D6')]:
        S.add('glock', S.at(bar, beat), 0.5, P(nm), 0.17)
    # GLIMPSE: eerie stab on beat 3 of bar 4
    S.fx('swell', 0, 1.0, 0.45, ends_at=S.at(3, 2))
    for nm in ['A2', 'C#4', 'G4', 'Bb4']:
        S.add('keys', S.at(3, 2), 0.9, P(nm), 0.62)
    S.add('bell', S.at(3, 2), 2.0, P('G5'), 0.3)
    S.drum('tom_lo', S.at(3, 2), 0.6)
    S.drum('crash_soft', S.at(3, 2), 0.35)
    S.hit('glimpse', 3, 2, 'GLIMPSE: eerie A7b9 chord stab (e-piano + bell, reverse swell into it)')
    S.add('lead', S.at(3, 3), 0.5, P('A4'), 0.5, inst='kalimba')
    S.add('lead', S.at(3, 3.5), 0.5, P('C5'), 0.55, inst='kalimba')
    S.drum('shaker', S.at(3, 3), 0.3)


def s_heartland(S):
    S.section('heartland')
    S.prog(TRAVEL + ' | D:2 A/C#:2 | Bm7:2 G:2 | Em7:2 G7:2')
    travel(S)
    S.m('lead', 1, th(1, 2) + ' | B5:2 A5 G5 F5:2 D5:2', vel=0.8, inst='lead')
    S.m('lead2', 2, 'F#5:3 F#5 D5:2 B4:2 | G5:2 F#5 E5 D5:2 B4:2', vel=0.4)
    S.bassline('bass', 1, 4, OCT8, lo=36, vel=0.78, inst='bassoct')
    S.comp('stab', 1, 4, OFF8, lo=55, hi=67, n=3, vel=0.48, inst='brass')
    S.arp('arp', S.at(1), S.at(4), step=0.25, lo=72, hi=88, pattern='updown', vel=0.28, pans=(-0.6, 0.6))
    S.pad('pad', 1, 4, lo=50, hi=66, n=3, vel=0.34)
    S.drums(1, 4, {'kick': 'x.....x.x.......', 'snare': '....x.......x...', 'hat': 'g.o.g.o.g.o.g.o.'})
    for bar, nm in ((2, 'cut2'), (3, 'cut3')):
        S.drum('crash', S.at(bar), 0.85)
        S.drum('boom', S.at(bar), 0.4)
        S.hit(nm, bar, 0, f'postcard cut #{bar}: crash + boom on the bar line')
    fill(S, 3)


def s_rainbowfalls(S):
    S.section('rainbowfalls')
    S.prog(TRAVEL + ' | D:2 G:2 | Bm7:2 A:2 | Em7:2 G7:2')
    travel(S)
    S.m('lead', 1, th(1, 2) + ' | G5:2 F#5 E5 D5:2 F5:2', vel=0.8, inst='lead', drive=1.3)
    S.bassline('bass', 1, 4, OCT8, lo=36, vel=0.8, inst='pulsebass')
    S.arp('arp', S.at(1), S.at(4), step=0.25, lo=67, hi=86, pattern='up', vel=0.34, pans=(-0.55, 0.55),
          lp_ramp=(3000, 7000))
    S.pad('pad', 1, 4, lo=52, hi=67, n=3, vel=0.36, inst='padbright')
    S.drums(1, 4, {'kick': 'x...x...x...x...', 'snare': '....x.......x...', 'hat': 'gogggogggogggogg',
                   'ohat': '..............o.'})
    S.duck.append((S.at(1), S.at(4), 0.3))
    S.drum('crash', S.at(2), 0.9)
    S.drum('boom', S.at(2), 0.55)
    S.comp('stab', 2, 3, [(0, 0.5, 1.0)], lo=60, hi=72, n=3, vel=0.72, inst='brass')
    S.hit('trestle', 2, 0, 'TRESTLE: crash + boom + brass stab (the train crosses the trestle)')
    S.roll('snare', S.at(3, 3), S.at(4), 0.125, 0.3, 0.8)


def s_manehattan(S):
    S.section('manehattan')
    S.prog(TRAVEL + ' | Dmaj7:2 B7:2 | Em7:2 A7:2 | Bm7:2 Gmaj7:2 | Gmaj7:2 F#7:2')
    travel(S)
    sw = 0.64
    S.m('brass', 1, th(1) + ' | B5:3 A5 G5:2 E5:2 | ' + TH[3], vel=0.82, swing=sw)
    S.m('brass2', 1, 'F#4 A4 C#5 D5:3 F#5:2 | G5:3 F#5 C#5:2 A4:2 | F#4 B4 D5 D5:3 F#5:2', vel=0.52, swing=sw)
    S.m('bass', 1, 'D2:2 F#2:2 B1:2 D#2:2 | E2:2 G2:2 A1:2 C#2:2 | B1:2 D2:2 G1:2 A1:2 | G1:2 B1:2 F#1:2 A#1:2',
        vel=0.8, inst='upright', gate=0.85)
    S.comp('keys', 1, 5, [(0, 1.0, 1.0), (1.5, 0.4, 0.8)], lo=55, hi=67, n=3, vel=0.48, swing=sw, human=0.6)
    S.drums(1, 5, {'ride': 'x...x.o.x...x.o.', 'hat_soft': '....x.......x...', 'kick_soft': 'g.......g.......',
                   'snare_soft': '......g.......g.'}, swing=sw)
    # bar 5: BRIDGE shout (rest at beat 1.78 for the HUD toast)
    S.fx('swell', 0, 1.0, 0.5, ends_at=S.at(4))
    S.drum('crash', S.at(4), 0.9)
    S.drum('boom', S.at(4), 0.6)
    S.m('brass', 4, 'D6 r B5 F#5 A#5:2 C#6:2', vel=0.88, swing=sw)
    S.m('brass2', 4, 'B5 r G5 D5 F#5:2 A#5:2', vel=0.55, swing=sw)
    S.hit('bridge', 4, 0, 'BRIDGE: brass shout + crash + boom (the big suspension bridge)')
    S.roll('snare_soft', S.at(4, 3), S.at(5), 0.125, 0.3, 0.7)


def s_eastcoast(S):
    S.section('eastcoast')
    S.prog(TRAVEL + ' | D | G:2 A7:2 | Em7:2 B7:2')
    travel(S)
    sw = 0.667
    S.m('lead', 1, th(1) + ' | B5:3 A5 G5:2 E5 C#5 | B5:2 G5 E5 D#5:2 F#5:2', o=-12, vel=0.8, inst='accordion',
        swing=sw)
    S.comp('keys', 1, 4, [(0.5, 0.3, 1.0), (1.5, 0.3, 0.8), (2.5, 0.3, 1.0), (3.5, 0.3, 0.8)], lo=60, hi=72, n=3,
           vel=0.42, swing=sw, inst='accordion')
    S.bassline('bass', 1, 4, [(0, 'R', 0.9), (2, '5-12', 0.9, 0.85)], lo=36, vel=0.75, inst='upright')
    S.pad('pad', 1, 4, lo=50, hi=65, n=3, vel=0.3, inst='padsoft')
    S.drums(1, 4, {'kick_soft': 'x.......x.......', 'snare_soft': '....o.......o...', 'shaker': 'g.o.g.o.g.o.g.o.',
                   'tick': '..g.......g.....'}, swing=sw)
    S.m('glock', 2, 'r:6 D6 E6 | r:8', vel=0.2)
    S.add('bell', S.at(1, 2), 6.0, S.pp('D3'), 0.68)
    S.drum('boom', S.at(1, 2), 0.35)
    S.hit('bell', 1, 2, 'BELL: deep harbour-bell stroke (G3), rings ~4 s')
    fill(S, 3, kind='snare')


def s_griffonstone(S):
    S.section('griffonstone')
    S.prog(TRAVELM + ' | Dm | Bb:2 C:2 | Gm:2 B7:2')
    travel(S, crash='crash', cv=0.8)
    S.m('brass', 1, th(1, minor=True, o=-12) + ' | ' + tr('Bb5:3 A5 G5:2 E5:2', -12) +
        ' | D4:2 C4 Bb3 A3:2 F#3:2', vel=0.85, drive=1.4)
    S.m('lead2', 1, 'r:4 A4:4 | r:4 G4:4 | r:8', vel=0.34, inst='thin')
    S.pad('pad', 1, 4, lo=40, hi=57, n=3, vel=0.5, inst='padsoft')
    S.bassline('bass', 1, 4, [(0, 'R', 1.9), (2, 'R', 1.9, 0.9)], lo=36, vel=0.75)
    S.drums(1, 3, {'boom': 'x.......x.......', 'tom_lo': 'x..x..x.x..x..x.', 'tom_mid': '......o.......o.',
                   'snare_soft': '....o.......o...'}, vel=0.85)
    S.fx('sweep', S.at(1), 4.0, 0.3, hi=4000.0, lo=600.0)
    # GUST
    S.fx('riser', 0, 1.0, 0.5, ends_at=S.at(3), lo=500.0, hi=6000.0, f0=200.0, octs=1.5)
    S.fx('sweep', S.at(3), 3.0, 0.6, hi=8000.0, lo=300.0)
    S.drum('crash_soft', S.at(3), 0.7)
    S.hit('gust', 3, 0, 'GUST: wind whoosh (noise riser into a long downward sweep) + soft cymbal')
    S.drums(3, 4, {'boom': 'x.......x.......', 'tom_lo': 'x..x..x.........', 'snare_soft': '....o.......o...'})
    fill(S, 3, v=0.6)


def s_dragonlands(S):
    S.section('dragonlands')
    S.prog(TRAVELM + ' | Dm | Bb:2 C:2 | Gm:2 Bb7:2')
    travel(S, crash='crash', cv=0.9)
    S.m('lead', 1, th(1, minor=True) + ' | Bb5:3 A5 G5:2 E5:2 | D5:2 C5 Bb4 F5:2 D5:2', vel=0.85, inst='lead',
        drive=1.8)
    offs = [0, 12, 0, 7]
    for bar in range(1, 4):
        for st in range(16):
            b = S.at(bar, st / 4)
            c = S.chord_at(b)
            r = 38 + (c.root - 38) % 12
            S.add('power', b, 0.25, r + offs[st % 4], 0.62 if st % 4 == 0 else 0.46)
    S.bassline('bass', 1, 4, [(i * 0.5, 'R', 0.45, 1.0 if i % 2 == 0 else 0.8) for i in range(8)], lo=33, vel=0.85,
               inst='bassoct')
    S.drums(1, 4, {'kick': 'x..x..x.x..x....', 'snare': '....x.......x...', 'hat': 'x.x.x.x.x.x.x.x.'})
    S.duck.append((S.at(1), S.at(4), 0.35))
    S.fx('riser', 0, 2.0, 0.55, ends_at=S.at(2, 2), lo=300.0, hi=7000.0, f0=150.0, octs=2.5)
    S.drum('boom', S.at(2, 2), 1.0)
    S.drum('crash', S.at(2, 2), 0.95)
    S.hit('erupt', 2, 2, 'ERUPT: riser into boom + crash + kick (the volcano blows) on beat 3')
    fill(S, 3, v=0.65)


def s_mountaris(S):
    S.section('mountaris')
    S.prog(TRAVEL + ' | D | Bm7:2 G:2 | Bm9:2 Gmaj9:2 | Em9:2 G7:2')
    travel(S)
    S.m('lead', 1, th(1, 2), vel=0.78, inst='warm')
    for bar in (1, 2):
        for q in range(32):
            b = S.at(bar, q * 0.125)
            v = voicing(S.chord_at(b), 64, 77, 2)
            for m in v:
                S.add('mando', b, 0.125, m, 0.4 if q % 2 == 0 else 0.3)
    S.comp('keys', 1, 3, [(0, 1.9, 1.0), (2, 1.9, 0.9)], lo=52, hi=64, n=3, vel=0.42, inst='brass')
    S.bassline('bass', 1, 3, [(0, 'R', 0.9), (1, '5-12', 0.9, 0.8), (2, 'R', 0.9, 0.9), (3, '5-12', 0.9, 0.8)],
               lo=36, vel=0.72, inst='upright')
    S.drums(1, 3, {'shaker': 'gogggogggogggogg', 'rim': '....o.......o..g', 'kick_soft': 'x.......x.......'})
    # DIVE: splash + filter drop
    S.fx('riser', 0, 0.75, 0.3, ends_at=S.at(3), lo=800.0, hi=9000.0, f0=400.0, octs=1.5)
    S.fx('sweep', S.at(3), 2.0, 0.6, hi=7000.0, lo=200.0)
    S.drum('crash_soft', S.at(3), 0.6)
    S.drum('boom', S.at(3), 0.6)
    S.hit('dive', 3, 0, 'DIVE: splash (cymbal + boom + noise sweep down) and filter drop: bars 4-5 underwater')
    S.m('lead', 3, 'A4:2 D5:2 E5:2 F#5:2 | F#5:3 E5 D5:2 B4:2', vel=0.72, inst='mellow', bus='uw')
    S.pad('pad', 3, 5, lo=50, hi=67, n=4, vel=0.55, inst='padsoft', bus='uw')
    S.arp('triarp', S.at(3), S.at(5), step=0.5, lo=74, hi=88, pattern='updown', vel=0.3, pans=(-0.5, 0.5), bus='uw',
          inst='glock')
    S.m('bass', 3, 'B2:4 G2:4 | E2:4 G2:4', vel=0.6, inst='upright', bus='uw')
    for bar, beat, nm in [(3, 0.75, 'F#6'), (3, 1.25, 'B6'), (3, 3.25, 'D7'), (4, 2.25, 'A6'), (4, 2.75, 'E7')]:
        S.add('blip', S.at(bar, beat), 0.25, S.pp(nm), 0.2, bus='uw')
    S.drums(3, 5, {'kick_soft': 'o.......o.......'}, vel=0.6)


def s_klugetown(S):
    S.section('klugetown')
    S.prog(TRAVELM + ' | Dm | Gm:2 A7:2 | Bb:2 F#7:2')
    travel(S)
    S.m('lead', 1, th(1, minor=True) + ' | Bb5:3 A5 G5 F5 E5 C#5 | D5:2 F5 D5 C#5:2 A#4:2', vel=0.78, inst='reed')
    S.arp('stab', S.at(1), S.at(4), step=0.5, lo=57, hi=74, group=[0, 2, 1, 3], vel=0.42, inst='pluck',
          pans=(0.3, -0.3))
    S.m('bass', 1, 'D2:2 A1:2 D2:2 A1:2 | G1:2 D2:2 A1:2 E2:2 | A#1:2 F2:2 F#1:2 C#2:2', vel=0.75, inst='upright')
    S.drums(1, 4, {'kick_soft': 'x.......x..x....', 'rim': 'x..x..x...x..x..', 'tick': '..g...g.g...g..g',
                   'tom_hi': '.......o.......o', 'clap': '....o.......o...', 'shaker': 'g.g.g.g.g.g.g.g.'}, human=0.6)
    S.pad('pad', 1, 4, lo=50, hi=65, n=3, vel=0.3, inst='padsoft')
    # AIRSHIP: low horn
    S.fx('swell', 0, 1.5, 0.5, ends_at=S.at(3))
    S.add('brass2', S.at(3), 2.0, P('Bb2'), 0.72, inst='brass')
    S.add('brass2', S.at(3), 2.0, P('F3'), 0.6, inst='brass')
    S.drum('boom', S.at(3), 0.5)
    S.drum('crash_soft', S.at(3), 0.6)
    S.fx('sweep', S.at(3), 2.0, 0.3, hi=3000.0, lo=200.0)
    S.hit('airship', 3, 0, 'AIRSHIP: low horn blast (Bb2 + F3) + boom + soft cymbal + propeller whoosh')


def s_badlands(S):
    S.section('badlands')
    S.prog(TRAVELM + ' | Dm | D | Gmaj9:2 C7b9:2')
    travel(S, crash='crash_soft', cv=0.6)
    # bar 2: ominous
    S.pad('pad', 1, 2, lo=38, hi=55, n=3, vel=0.55, inst='padsoft')
    for i in range(16):
        S.add('lead2', S.at(1, i * 0.25), 0.25, S.pp('D3' if i % 2 == 0 else 'Eb3'), 0.34, inst='arp25', drive=2.0,
              lp=1800)
    S.m('brass', 1, th(1, minor=True, o=-12), vel=0.6)
    S.m('bass', 1, 'D2:8', vel=0.55)
    S.drums(1, 2, {'tom_lo': 'x.......x.......', 'kick_soft': 'x...............'})
    S.fx('swell', 0, 2.0, 0.75, ends_at=S.at(2))
    S.fx('riser', 0, 1.0, 0.45, ends_at=S.at(2), lo=400.0, hi=9000.0, f0=200.0, octs=3.0)
    # bar 3: TRANSFORM -> radiant major
    S.drum('crash', S.at(2), 1.0)
    S.drum('boom', S.at(2), 0.8)
    S.fx('shimmer', S.at(2), 4.0, 0.7, f0=1760.0)
    S.m('lead', 2, th(1), vel=0.85, inst='lead')
    S.m('glock', 2, th(1, o=12), vel=0.28)
    S.pad('pad', 2, 3, lo=52, hi=69, n=4, vel=0.5, inst='padbright')
    S.arp('arp', S.at(2), S.at(3), step=0.25, lo=62, hi=86, pattern='up', vel=0.36, pans=(-0.6, 0.6))
    S.bassline('bass', 2, 3, BOUNCE, lo=36, vel=0.78)
    S.drums(2, 3, {'kick': 'x.....x.x.......', 'snare': '....x.......x...', 'hat': 'g.o.g.o.g.o.g.o.'})
    S.hit('transform', 2, 0, 'TRANSFORM: crash + boom + shimmer; F# minor bursts into radiant F# major')
    # bar 4: wonder, then the dark sting
    S.m('lead', 3, 'B5:2 A5 F#5 r:4', vel=0.75, inst='lead')
    S.pad('pad', 3, 4, lo=52, hi=69, n=4, vel=0.45, inst='padbright', end_beat=S.at(3, 2))
    S.arp('triarp', S.at(3), S.at(3, 2), step=0.25, lo=67, hi=88, pattern='updown', vel=0.34)
    S.add('bass', S.at(3), 1.9, S.pp('G2'), 0.7)
    for nm in ['C3', 'G3', 'Bb3', 'E4', 'Db5']:
        S.add('stab', S.at(3, 2), 1.8, S.pp(nm), 0.7, inst='power')
    S.add('bass', S.at(3, 2), 1.9, S.pp('C2'), 0.8)
    S.drum('boom', S.at(3, 2), 0.9)
    S.drum('tom_lo', S.at(3, 2), 0.8)
    S.drum('crash_soft', S.at(3, 2), 0.5)
    S.hit('tartarus', 3, 2, 'dark sting (E7b9: low power chord + boom + tom): the gates of Tartarus')


def s_frontier(S):
    S.section('frontier')
    S.prog(TRAVEL + ' | D | G:2 A:2 | Bm:2 F#7:2')
    travel(S)
    S.m('lead', 1, th(1) + ' | B5:3 A5 G5:2 E5 C#5', vel=0.8, inst='twang', slide=1.0)
    S.bassline('bass', 1, 4, [(0, 'R', 0.9), (2, '5-12', 0.9, 0.85)], lo=36, vel=0.78)
    S.comp('stab', 1, 4, [(1, 0.35, 1.0), (3, 0.35, 0.95)], lo=55, hi=67, n=3, vel=0.55, inst='banjo')
    S.drums(1, 4, {'tick': 'x.o.x.o.x.o.x.o.', 'rim': '...o...o...o...o', 'kick_soft': 'x.......x.......',
                   'snare_soft': '....o.......o...'})
    S.pad('pad', 1, 4, lo=50, hi=65, n=3, vel=0.3, inst='padsoft')
    S.m('whistle', 3, 'F#5:3 E5 C#5:2 A#4 C#5', vel=0.75)
    S.hit('whistle', 3, 0, 'WHISTLE: the whistle lead takes the tune at sunset (G#5 F#5 E5 ...)')
    fill(S, 3, kind='snare', v=0.45)


def s_laspegasus(S):
    S.section('laspegasus')
    S.prog(TRAVEL + ' | Dmaj7:2 Bm7:2 | Em7:2 A7:2 | Bm7:2 Gmaj7:2 | Em9:2 G13:2')
    travel(S)
    S.m('lead', 1, th(1) + ' | B5:3 A5 G5:2 E5:2 | ' + TH[3], o=12, vel=0.78, inst='lead', drive=1.3)
    funk = [(0, 'R', 0.4), (0.75, 'R+12', 0.2, 0.7), (1.5, 'R', 0.3, 0.85), (2, 'R+12', 0.2, 0.7),
            (2.5, 'R', 0.4, 0.9), (3.25, 'R+12', 0.2, 0.7), (3.5, '5', 0.3, 0.8)]
    S.bassline('bass', 1, 5, funk, lo=36, vel=0.82, inst='pulsebass')
    S.drums(1, 5, {'kick': 'x...x...x...x...', 'clap': '....x.......x...', 'ohat': '..o...o...o...o.',
                   'hat': 'g.g.g.g.g.g.g.g.'})
    S.comp('keys', 1, 5, [(0.5, 0.25, 0.8), (1.25, 0.25, 0.7), (2.5, 0.25, 0.8), (3.25, 0.25, 0.7)], lo=60, hi=72,
           n=3, vel=0.4, inst='pluck')
    S.comp('stab', 1, 2, [(1.5, 0.25, 1.0), (3.75, 0.25, 0.8)], lo=58, hi=70, n=3, vel=0.6, inst='brass')
    S.comp('stab', 3, 4, [(1.5, 0.25, 1.0), (3.75, 0.25, 0.8)], lo=58, hi=70, n=3, vel=0.6, inst='brass')
    S.pad('pad', 1, 5, lo=52, hi=67, n=3, vel=0.28)
    S.duck.append((S.at(1), S.at(5), 0.3))
    # NEON: rising 8th-note stab line
    for i in range(8):
        b = S.at(2, 0.5 * i)
        for m in voicing(S.chord_at(b), 55 + 2 * i, 67 + 2 * i, 3):
            S.add('stab', b, 0.3, m, 0.55 + 0.04 * i, inst='brass')
    S.hit('neon', 2, 0, 'NEON: rising brass-stab line on every 8th (144,153,...,207): one building lights per stab')
    # FIREWORKS
    S.drum('crash', S.at(4), 0.95)
    S.drum('boom', S.at(4), 0.8)
    S.fx('shimmer', S.at(4), 3.0, 0.5, f0=2000.0)
    for beat in (0, 2, 3):
        ch = S.chord_at(S.at(4, beat))
        c = [m for m in ch.tones(76, 91) if (m - ch.root) % 12 in (0, 3, 4, 7)][::-1]
        for i, m in enumerate(c[:4]):
            S.add('glock', S.at(4, beat) + 0.125 * i, 0.5, m, 0.34 - 0.05 * i, pan=(-0.5, 0.5, 0.0)[beat % 3])
    S.m('lead', 4, 'D6 r:1 B5 A5 G5:2 F5:2', vel=0.8, inst='lead', drive=1.3)
    S.hit('fireworks', 4, 0, 'FIREWORKS: crash + boom + glock bursts on beats 1, 3 and 4 (288, 324, 342)')


def s_westcoast(S):
    S.section('westcoast')
    S.prog(TRAVEL + ' | Dmaj9 | Bm9:2 Em9:2 | Gmaj9:2 C9:2')
    travel(S, night=True, crash='crash_soft', cv=0.55, lead='mellow')
    sw = 0.6
    S.m('lead', 1, th(1, 2) + ' | B5:3 A5 F#5:2 r:2', vel=0.72, inst='mellow', swing=sw, late=0.03)
    S.comp('keys', 1, 4, [(0, 1.9, 1.0), (2.5, 1.4, 0.8)], lo=52, hi=67, n=4, vel=0.52, swing=sw, human=1.0)
    S.bassline('bass', 1, 4, [(0, 'R', 1.2), (1.5, '5-12', 0.4, 0.7), (2.5, 'R', 0.9, 0.9), (3.5, '8', 0.4, 0.6)],
               lo=36, vel=0.8, inst='upright', swing=sw)
    S.drums(1, 4, {'kick_lofi': 'x......x..x.....', 'snare_lofi': '....x.......x...', 'hat': 'o.g.o.g.o.g.o.gg'},
            swing=sw, human=1.0, vel=0.85)
    S.fx('crackle', S.at(1), 7.2, 0.8)
    S.pad('pad', 1, 4, lo=48, hi=62, n=3, vel=0.35, inst='padsoft')
    S.fx('swell', 0, 1.5, 0.35, ends_at=S.at(3))
    S.add('bell', S.at(3), 4.0, S.pp('D5'), 0.32)
    S.add('glock', S.at(3, 0.5), 1.0, S.pp('A5'), 0.18)
    S.fx('shimmer', S.at(3), 3.0, 0.28, f0=1200.0)
    S.hit('moonrise', 3, 0, 'MOONRISE: gentle bell swell (soft reverse cymbal + bell C#5 + shimmer)')


def s_crystal(S):
    S.section('crystal')
    S.prog(TRAVEL + ' | D:2 A/C#:2 | Bm7:2 G:2 | Gmaj9 | Em7:2 Bb7:2')
    travel(S, night=True, crash='crash', cv=0.75)
    S.m('lead', 1, th(1, 2, 3) + ' | A5:2 G5 F#5 F5:2 D5:2', vel=0.8, inst='warm')
    S.m('glock', 1, th(1, 2, 3, o=12), vel=0.22)
    S.pad('choir', 1, 5, lo=55, hi=72, n=4, vel=0.45)
    S.pad('pad', 1, 5, lo=50, hi=64, n=3, vel=0.32, inst='padbright')
    S.arp('triarp', S.at(1), S.at(5), step=0.5, lo=55, hi=74, pattern='updown', vel=0.36)
    S.bassline('bass', 1, 5, [(0, 'R', 1.9), (2, 'R', 1.9, 0.85)], lo=36, vel=0.7)
    S.drums(1, 3, {'kick_soft': 'x.........x.....', 'snare': '........x.......', 'hat_soft': 'g.g.g.g.g.g.g.g.'})
    # HEART
    S.drum('crash', S.at(3), 0.85)
    S.fx('shimmer', S.at(3), 4.0, 0.7, f0=2093.0)
    for b in (0, 2):
        S.drum('boom', S.at(3, b), 0.75)
        S.drum('kick_soft', S.at(3, b + 1 / 3), 0.6)
    for i, nm in enumerate(['B5', 'D6', 'A6', 'B6']):
        S.add('glock', S.at(3) + 0.08 * i, 2.0, S.pp(nm), 0.32, pan=-0.4 + 0.2 * i)
    S.hit('heart', 3, 0, 'HEART: heartbeat (boom + kick "lub-dub" on beats 1 and 3) + big shimmering Gmaj9 chord')
    S.drums(4, 5, {'kick_soft': 'x.........x.....', 'snare': '........x.......', 'hat_soft': 'g.g.g.g.g.g.g.g.'})
    fill(S, 4, kind='snare', v=0.5)


def s_yakyakistan(S):
    S.section('yakyakistan')
    S.prog(TRAVELM + ' | Dm | Bb:2 C:2 | Gm:2 A7:2')
    travel(S, night=True, crash='crash', cv=0.85)
    S.m('brass', 1, th(1, minor=True, o=-12) + ' | ' + tr('Bb5:3 A5 G5:2 E5:2', -12) +
        ' | D4:2 C4 Bb3 A3:2 C#4:2', vel=0.85, drive=1.3)
    S.comp('keys', 1, 4, [(1, 0.5, 1.0), (3, 0.5, 0.9)], lo=55, hi=67, n=3, vel=0.5, inst='accordion')
    S.bassline('bass', 1, 4, [(0, 'R', 0.9), (1, 'R', 0.9, 0.7), (2, '5-12', 0.9, 0.85), (3, 'R', 0.9, 0.7)],
               lo=36, vel=0.8)
    S.drums(1, 3, {'boom': 'x.......x.......', 'kick': 'x.......x.......', 'clap': '....x.......x...',
                   'tom_lo': '......o.......o.'})
    S.drum('crash', S.at(2, 3), 0.9)
    S.drum('boom', S.at(2, 3), 0.9)
    S.fx('sweep', S.at(2, 3), 3.0, 0.65, hi=5000.0, lo=150.0)
    S.roll('tom_lo', S.at(2, 3), S.at(3), 0.125, 0.5, 0.9)
    S.hit('avalanche', 2, 3, 'AVALANCHE: crash + boom + tom roll + rumbling noise sweep (beat 4 of bar 3)')
    S.drums(3, 4, {'boom': 'x.......x.......', 'kick': 'x.......x.......', 'clap': '....x.......x...'})
    fill(S, 3, v=0.6)


def s_twosisters(S):
    S.section('twosisters')
    S.prog(TRAVEL + ' | Dmaj7 | Bm7 | Em7:2 A7sus4:1 A7:1 | D')
    travel(S, night=True, crash='crash_soft', cv=0.45, lead='mellow', arp='triarp')
    # bars 2-3: lullaby
    S.m('mbox', 1, th(1, 2), vel=0.6)
    S.pad('pad', 1, 3, lo=50, hi=66, n=4, vel=0.4, inst='padsoft')
    S.arp('triarp', S.at(1), S.at(3), step=1.0, lo=50, hi=69, group=[0, 2, 1, 3], vel=0.3)
    S.m('bass', 1, 'D2:8 | B1:8', vel=0.4, inst='upright')
    # bar 4: building
    S.m('lead', 3, TH[3], vel=0.72, inst='mellow')
    S.m('glock', 3, TH[3], o=12, vel=0.24)
    S.arp('arp', S.at(3), S.at(4), step=0.25, lo=62, hi=86, pattern='up', vel=0.3, vel_ramp=(0.6, 1.25),
          pans=(-0.5, 0.5))
    S.pad('pad', 3, 4, lo=52, hi=69, n=4, vel=0.45)
    S.bassline('bass', 3, 4, [(i * 0.5, 'R', 0.45, 0.8 + 0.03 * i) for i in range(8)], lo=38, vel=0.75,
               inst='bassoct')
    S.roll('snare_soft', S.at(3, 2), S.at(4), 0.125, 0.15, 0.85)
    S.roll('tom_lo', S.at(3), S.at(3, 2), 0.5, 0.3, 0.5)
    S.fx('riser', 0, 2.0, 0.5, ends_at=S.at(4), lo=400.0, hi=9000.0, f0=200.0, octs=2.5)
    S.fx('swell', 0, 2.0, 0.7, ends_at=S.at(4))
    S.hit('build', 3, 0, 'build-up: arpeggio + snare roll + riser toward midnight')
    # bar 5: MIDNIGHT
    S.drum('crash', S.at(4), 1.0)
    S.drum('boom', S.at(4), 0.85)
    for i in range(12):
        S.add('bell', S.at(4) + i / 3.0, 1.0, P('D4'), 0.62 - 0.015 * i, tau=1.0)
    S.hit('midnight', 4, 0, 'MIDNIGHT: 12 bell strikes on triplet 8ths (local frames 288, 294, ..., 354) under a '
                            'full statement of the theme')
    S.m('brass', 4, TH[1], vel=0.85)
    S.m('brass2', 4, 'F#4 A4 C#5 D5:3 F#5:2', vel=0.55)
    S.m('glock', 4, TH[1], o=12, vel=0.24)
    S.pad('pad', 4, 5, lo=50, hi=69, n=4, vel=0.5, inst='padbright')
    S.arp('arp', S.at(4), S.at(5), step=0.25, lo=62, hi=84, pattern='updown', vel=0.3)
    S.bassline('bass', 4, 5, [(0, 'R', 1.9), (2, 'R', 0.9), (3, '5-12', 0.9)], lo=38, vel=0.8)
    S.drums(4, 5, {'kick': 'x.......x.......', 'snare': '....x.......x...', 'hat': 'g.o.g.o.g.o.g.o.'})


def s_outro(S):
    S.section('outro')
    S.prog('Bm7:2 G:2 | Em7:2 A7:2 | D:2 B7sus4:1 B7:1 | E | C#m7:2 A:2 | F#m7:2 B7:2 | Eadd9')
    S.m('lead', 0, TH[2] + ' | B4 E5 F#5 G5:3 E5 C#5 | D6:2 A5 F#5 E5:2 D#5:2', vel=0.8, inst='lead')
    S.m('brass2', 0, 'F#5:3 F#5 D5:2 B4:2 | G4 B4 D5 E5:3 C#5 A4 | A5:2 F#5 D5 B4:4', vel=0.5)
    for i in range(16):
        b = S.at(0, 0.5 * i)
        tones = S.chord_at(b).tones(81, 93)
        S.add('blip', b, 0.25, tones[i % len(tones)], 0.22, pan=-0.5 + i / 15)
    S.hit('route', 0, 0, 'route draws itself: blip on every 8th, bars 1-2 (local frames 0, 9, 18, ... 135)')
    S.pad('pad', 0, 6, lo=50, hi=67, n=4, vel=0.42)
    S.bassline('bass', 0, 6, BOUNCE, lo=36, vel=0.75)
    S.drums(0, 2, {'kick_soft': 'x.......x.......', 'clap': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.'})
    S.arp('arp', S.at(0), S.at(2), step=0.25, lo=62, hi=79, pattern='updown', vel=0.26)
    # MAP COMPLETE stamp
    S.drum('crash', S.at(2), 1.0)
    S.drum('boom', S.at(2), 0.75)
    S.drum('clap', S.at(2), 0.8)
    S.drum('kick', S.at(2), 0.75)
    S.comp('stab', 2, 3, [(0, 0.6, 1.0)], lo=57, hi=74, n=4, vel=0.8, inst='brass')
    S.hit('complete', 2, 0, 'MAP COMPLETE stamp: crash + boom + clap + brass chord')
    S.drums(2, 3, {'kick_soft': 'x.......x.......', 'hat_soft': 'g.o.g.o.g.o.....'})
    S.fx('swell', 0, 1.5, 0.6, ends_at=S.at(3))
    S.roll('snare', S.at(2, 3), S.at(3), 0.125, 0.3, 0.8)
    # KEY CHANGE: E major, last statement
    S.drum('crash', S.at(3), 0.9)
    S.hit('keychange', 3, 0, 'KEY CHANGE up a tone (E major): last theme statement; credits one card per bar')
    S.m('lead', 3, th(1, 2, o=2) + ' | C#5 F#5 G#5 A5:3 F#5 D#5', vel=0.85, inst='lead')
    S.m('brass', 3, th(1, 2, o=-10) + ' | C#4 F#4 G#4 A4:3 F#4 D#4', vel=0.55)
    S.m('glock', 3, th(1, 2, o=14), vel=0.2)
    S.m('brass2', 3, tr('F#4 A4 C#5 D5:3 F#5:2 | F#5:3 F#5 D5:2 B4:2', 2) + ' | A4 C#5 E5 F#5:3 D#5 B4', vel=0.5)
    S.arp('arp', S.at(3), S.at(6), step=0.25, lo=64, hi=86, pattern='updown', vel=0.28)
    S.drums(3, 6, {'kick': 'x.....x.x.......', 'snare': '....x.......x...', 'hat': 'g.o.g.o.g.o.g.o.',
                   'ohat': '..............o.'})
    S.duck.append((S.at(3), S.at(6), 0.25))
    fill(S, 5, v=0.6)
    # FINAL Eadd9
    S.drum('crash_soft', S.at(6), 0.7)
    S.drum('boom', S.at(6), 0.6)
    S.add('bass', S.at(6), 2.5, P('E2'), 0.65, inst='upright')
    for i, nm in enumerate(['E5', 'G#5', 'B5', 'F#6']):
        S.add('mbox', S.at(6) + 0.25 * i, 3.0, P(nm), 0.5 - 0.04 * i)
    for nm in ['G#3', 'B3', 'E4', 'F#4']:
        S.add('keys', S.at(6), 3.0, P(nm), 0.5, tau=1.2)
    for nm in ['E3', 'B3', 'F#4']:
        S.add('pad', S.at(6), 1.0, P(nm), 0.42, inst='padsoft', r=1.6)
    S.add('lead', S.at(6), 2.0, P('E5'), 0.66, inst='mellow')
    S.add('glock', S.at(6), 2.0, P('B6'), 0.24)
    S.hit('final', 6, 0, 'FINAL Eadd9 chord on the last downbeat; decays to digital silence exactly at the last sample')


SECTION_FUNCS = [s_intro, s_canterlot, s_cloudsdale, s_ponyville, s_sweetapple, s_everfree, s_heartland,
                 s_rainbowfalls, s_manehattan, s_eastcoast, s_griffonstone, s_dragonlands, s_mountaris, s_klugetown,
                 s_badlands, s_frontier, s_laspegasus, s_westcoast, s_crystal, s_yakyakistan, s_twosisters, s_outro]


def hud_space(S):
    """Leave the high register free for the HUD toast chime (D6 F#6 A6) at local frame frames-58."""
    n = 0
    for e in S.events:
        sec = SEC_BY_ID[e['section']]
        if not sec['region'] or e['midi'] is None or 'drum' in e or e['midi'] < 84:
            continue
        lo = sec['end_beat'] - 62 / FPB
        hi = sec['end_beat'] - 36 / FPB
        if lo <= e['start_beat'] < hi:
            e['velocity'] = round(e['velocity'] * 0.45, 3)
            n += 1
    return n

# =========================================================================== post-processing
def assign_glides(events):
    by = {}
    for e in events:
        if e['midi'] is not None and VOICES.get(e['voice'], {}).get('glide'):
            by.setdefault(e['voice'], []).append(e)
    for v, evs in by.items():
        evs.sort(key=lambda e: e['start_beat'])
        for a, b in zip(evs, evs[1:]):
            end = a['start_beat'] + a['dur_beats']
            if abs(end - b['start_beat']) < 0.06 and 0 < abs(a['midi'] - b['midi']) <= 7 and \
                    a['section'] == b['section']:
                b['_o']['glide_from'] = 440.0 * 2 ** ((a['midi'] - 69) / 12)


MONO_VOICES = ('lead', 'head', 'whistle', 'lead2', 'brass', 'brass2', 'bass')
ACCOMP_ORDER = {'keys': 0, 'stab': 0, 'power': 0, 'harpsi': 1, 'banjo': 1, 'mando': 1, 'triarp': 1, 'arp': 2,
                'choir': 3, 'pad': 3}


def tidy_lines(S):
    """Monophonic lines: a note never overlaps the next note of the same voice (swing / 'late' offsets
    can push a note a hair past the next onset) and never rings past the end of its section."""
    by = {}
    for e in S.events:
        if e['voice'] in MONO_VOICES and e['midi'] is not None and 'drum' not in e:
            by.setdefault(e['voice'], []).append(e)
    n = 0
    for evs in by.values():
        evs.sort(key=lambda e: e['start_beat'])
        for a, b in zip(evs, evs[1:]):
            over = a['start_beat'] + a['dur_beats'] - b['start_beat']
            if 1e-6 < over < 0.55 and b['start_beat'] > a['start_beat']:
                a['dur_beats'] = round(b['start_beat'] - a['start_beat'], 6)
                n += 1
    for e in S.events:
        if e['midi'] is None or 'drum' in e or e['voice'] in ('mbox', 'glock', 'bell'):
            continue
        end = SEC_BY_ID[e['section']]['end_beat']
        if e['start_beat'] + e['dur_beats'] > end + 1e-6 and end < TOTAL_BARS * 4:
            e['dur_beats'] = round(end - e['start_beat'], 6)
            n += 1
    return n


def resolve_clashes(S):
    """No semitone rubs in the same octave: accompaniment notes (keys, stabs, arps, pads) that sit a
    semitone from anything already sounding (melody, harmony, bass, ornaments, or accompaniment placed
    before them) are re-pitched to the nearest non-clashing chord tone (short notes) or moved an
    octave (long notes); dropped if nothing fits."""
    import bisect
    pitched = [e for e in S.events if e['midi'] is not None and 'drum' not in e and e['voice'] != 'fx']
    acc = [e for e in pitched if VOICES.get(e['voice'], {}).get('role') == 'accomp']
    placed = [e for e in pitched if VOICES.get(e['voice'], {}).get('role') != 'accomp']
    placed.sort(key=lambda e: e['start_beat'])
    starts = [e['start_beat'] for e in placed]
    acc.sort(key=lambda e: (ACCOMP_ORDER.get(e['voice'], 1), e['start_beat']))
    fixed = 0
    for e in acc:
        s, en = e['start_beat'], e['start_beat'] + e['dur_beats']
        lo_i = bisect.bisect_left(starts, s - 16)
        hi_i = bisect.bisect_right(starts, en)
        near = [p for p in placed[lo_i:hi_i]
                if p['start_beat'] < en - 1e-6 and p['start_beat'] + p['dur_beats'] > s + 1e-6 and p is not e]

        def clash(m):
            return any(abs(p['midi'] - m) == 1 for p in near)
        if clash(e['midi']):
            fixed += 1
            if e['dur_beats'] <= 0.6:
                c = S.chord_at(s)
                cands = sorted(c.tones(e['midi'] - 7, e['midi'] + 7), key=lambda m: abs(m - e['midi']))
                cands = [m for m in cands if m != e['midi'] and not clash(m) and
                         not any(p['midi'] == m and p['voice'] == e['voice'] for p in near)]
                if cands:
                    e['midi'] = cands[0]
                else:
                    e['_drop'] = True
            else:
                for m in (e['midi'] - 12, e['midi'] + 12):
                    if 40 <= m <= 86 and not clash(m) and not any(p['midi'] == m for p in near):
                        e['midi'] = m
                        break
                else:
                    e['_drop'] = True
        if not e.get('_drop'):
            i = bisect.bisect_right(starts, e['start_beat'])
            starts.insert(i, e['start_beat'])
            placed.insert(i, e)
    S.events = [e for e in S.events if not e.get('_drop')]
    return fixed


def dedupe_drums(S):
    """Two hits of the same drum voice on the same sample (e.g. the travel figure's reveal kick and the
    groove's first kick) -> keep the louder one."""
    best = {}
    for e in S.events:
        if 'drum' in e:
            key = (e['voice'], e['start_beat'])
            if key not in best or e['velocity'] > best[key]['velocity']:
                best[key] = e
    keep = {id(e) for e in best.values()}
    S.events = [e for e in S.events if 'drum' not in e or id(e) in keep]


def compose():
    S = Atlas()
    for f in SECTION_FUNCS:
        f(S)
    S.sec = None
    dedupe_drums(S)
    tidy_lines(S)
    nfix = resolve_clashes(S)
    hud_space(S)
    assign_glides(S.events)
    S.events.sort(key=lambda e: (e['start_beat'], e['voice']))
    for e in S.events:
        assert 0 <= e['start_beat'] < TOTAL_BARS * 4, e
        assert section_of_beat(e['start_beat'] + 1e-6) == e['section'] or e['voice'] == 'fx', e
    return S, nfix


# =========================================================================== outputs (JSON)
def _frame(beat):
    return beat * FPB


def export_json(S, out=OUT):
    os.makedirs(out, exist_ok=True)
    evs = []
    for e in S.events:
        sec = SEC_BY_ID[e['section']]
        d = {k: e[k] for k in ('section', 'voice', 'start_beat', 'dur_beats', 'midi', 'velocity')}
        d['time_s'] = round(e['start_beat'] * 0.6, 4)
        d['frame'] = round(_frame(e['start_beat']), 2)
        d['local_frame'] = round(_frame(e['start_beat'] - sec['start_beat']), 2)
        d['inst'] = e['_o'].get('inst')
        d['role'] = VOICES.get(e['voice'], {}).get('role', 'drums' if 'drum' in e else 'fx')
        if 'drum' in e:
            d['drum'] = e['drum']
        if 'fx' in e:
            d['fx'] = e['fx']
        if e['midi'] is not None and 'drum' not in e:
            d['note'] = pname(e['midi'])
        evs.append(d)
    with open(os.path.join(out, 'events.json'), 'w') as f:
        json.dump(evs, f, separators=(',', ':'))
    secs = []
    for s in SEC:
        hs = []
        for h in S.hits:
            if h['section'] != s['id']:
                continue
            g = h['global_beat']
            hs.append(dict(name=h['name'], bar=h['bar'], beat=h['beat'], desc=h['desc'],
                           time_s=round(g * 0.6, 3), frame=int(round(g * FPB)),
                           local_frame=int(round((g - s['start_beat']) * FPB))))
        chords = [dict(bar=int((b0 - s['start_beat']) // 4) + 1, beat=round((b0 - s['start_beat']) % 4 + 1, 3),
                       chord=c.sym, local_frame=int(round((b0 - s['start_beat']) * FPB)))
                  for b0, b1, c, sid in S.chords if sid == s['id']]
        secs.append(dict(id=s['id'], start_s=round(s['start_beat'] * 0.6, 3), end_s=round(s['end_beat'] * 0.6, 3),
                         bars=s['bars'], mood=s['mood'], key=s['key'], tempo_bpm=BPM, start_bar=s['start_bar'],
                         start_frame=s['start_beat'] * FPB, end_frame=s['end_beat'] * FPB,
                         start_sample=s['start_beat'] * SPB, chords=chords, hits=hs))
    with open(os.path.join(out, 'sections.json'), 'w') as f:
        json.dump(secs, f, indent=1)
    return evs, secs


# =========================================================================== render / mix
SECTION_REV = {'intro': 1.5, 'everfree': 1.4, 'westcoast': 1.2, 'crystal': 1.4, 'twosisters': 1.5,
               'outro': 1.3, 'cloudsdale': 1.3, 'mountaris': 1.2, 'griffonstone': 1.2}
# loudness map: relative level of each section (dB, applied per note so tails carry their own level)
SECTION_GAIN_DB = {'intro': -1.5, 'canterlot': 0.5, 'cloudsdale': -0.5, 'ponyville': -1.0, 'sweetapple': 1.0,
                   'everfree': -2.5, 'heartland': 0.0, 'rainbowfalls': 0.3, 'manehattan': 0.5, 'eastcoast': -1.0,
                   'griffonstone': 0.0, 'dragonlands': 0.3, 'mountaris': -0.8, 'klugetown': -0.8, 'badlands': 0.0,
                   'frontier': 0.0, 'laspegasus': 0.3, 'westcoast': -2.5, 'crystal': 0.0, 'yakyakistan': 0.0,
                   'twosisters': -1.0, 'outro': 0.3}
# mix trims (dB): voice -> dB, or (section, voice) -> dB
VOICE_GAIN_DB = {'head': -1.0, 'pling': -1.0, ('canterlot', 'timp'): -3.0, ('canterlot', 'brass'): 2.5,
                 ('manehattan', 'brass'): 3.5, ('sweetapple', 'lead'): 3.0, ('eastcoast', 'lead'): 2.0, ('klugetown', 'lead'): 1.5,
                 ('yakyakistan', 'boom'): -3.0}
# role trims (dB): role -> dB, or (section, role) -> dB
ROLE_GAIN_DB = {'bass': -2.0, 'melody': 1.5, ('canterlot', 'bass'): -3.0, ('sweetapple', 'bass'): -3.5,
                ('manehattan', 'bass'): -4.0, ('klugetown', 'bass'): -3.0, ('frontier', 'bass'): -3.0,
                ('yakyakistan', 'bass'): -2.0, ('griffonstone', 'bass'): -2.0, ('dragonlands', 'drums'): -2.0,
                ('rainbowfalls', 'drums'): -2.0, ('laspegasus', 'drums'): -2.0, ('heartland', 'drums'): -1.5,
                ('sweetapple', 'drums'): -3.0}


def render(S, path=os.path.join(OUT, 'soundtrack.wav'), stems=False, verbose=True):
    import time
    import numpy as np
    from audio import synth as Y
    for k_, v_ in CUSTOM_INST.items():
        Y.INSTRUMENTS.setdefault(k_, v_)
    t0 = time.time()
    N = TOTAL_SAMPLES
    main = np.zeros((2, N), np.float32)
    duck = np.zeros((2, N), np.float32)
    uw = np.zeros((2, N), np.float32)
    esend = np.zeros(N, np.float32)
    rsend = np.zeros((2, N), np.float32)
    roles = {}
    kicks = []
    for i, e in enumerate(S.events):
        o = dict(e['_o'])
        inst = o.pop('inst', None)
        v = VOICES.get(e['voice'], {})
        gate = o.pop('gate', v.get('gate', 0.95))
        pan = o.pop('pan', None)
        bus_o = o.pop('bus', None)
        o.pop('seed_', None)
        seed = 1000 + i * 7
        if inst and inst.startswith('drum:'):
            name = inst[5:]
            y = Y.drum(name, e['velocity'], seed=seed)
            dv = DRUM_VOICE.get(name, name)
            pan_, rv = DRUM_MIX.get(dv, (0.0, 0.05))
            pan = pan_ if pan is None else pan
            ev_echo, bus, role = 0.0, 'main', 'drums'
            if dv == 'kick':
                kicks.append(int(round(e['start_beat'] * SPB)))
        elif inst and inst.startswith('fx:'):
            name = inst[3:]
            y = Y.fx(name, e['dur_beats'] * 0.6, e['velocity'], seed=seed, **o)
            pan = 0.0 if pan is None else pan
            rv, ev_echo, bus, role = 0.25 if name != 'crackle' else 0.0, 0.0, 'main', 'fx'
        else:
            midi = e['midi']
            sl = o.pop('slide', None)
            if sl:
                midi -= sl
                o.update(bend=sl, bend_time=0.06)
            y = Y.render_note(inst, midi, e['dur_beats'] * 0.6 * gate, e['velocity'], seed=seed, **o)
            pan = v.get('pan', 0.0) if pan is None else pan
            rv, ev_echo, bus, role = v.get('rev', 0.1), v.get('echo', 0.0), v.get('bus', 'main'), v.get('role')
        if bus_o:
            bus = bus_o
            if bus == 'uw':
                rv *= 2.2
                ev_echo = max(ev_echo, 0.3)
        rv *= SECTION_REV.get(e['section'], 1.0)
        sg = 10 ** ((SECTION_GAIN_DB.get(e['section'], 0.0) +
                     VOICE_GAIN_DB.get((e['section'], e['voice']), VOICE_GAIN_DB.get(e['voice'], 0.0)) +
                     ROLE_GAIN_DB.get(role, 0.0) + ROLE_GAIN_DB.get((e['section'], role), 0.0)) / 20)
        y = y * sg
        gl, gr = Y.pan_gains(pan)
        s0 = int(round(e['start_beat'] * SPB))
        if s0 >= N:
            continue
        if y.ndim == 1:
            n = min(len(y), N - s0)
            st = np.stack([y[:n] * gl, y[:n] * gr])
            mono = y[:n]
        else:
            n = min(y.shape[1], N - s0)
            st = y[:, :n] * np.array([[gl], [gr]])
            mono = y[:, :n].mean(0)
        (duck if bus == 'duck' else uw if bus == 'uw' else main)[:, s0:s0 + n] += st
        if ev_echo:
            esend[s0:s0 + n] += mono * ev_echo
        if rv:
            rsend[:, s0:s0 + n] += st * rv
        if stems:
            r = roles.setdefault(role, np.zeros((2, N), np.float32))
            r[:, s0:s0 + n] += st
    t1 = time.time()
    # sidechain-style pump on the duck bus
    g = np.ones(N)
    for b0, b1, depth in S.duck:
        s0, s1 = int(b0 * SPB), int(b1 * SPB)
        hits = [k - s0 for k in kicks if s0 <= k < s1]
        g[s0:s1] = Y.duck_curve(s1 - s0, hits, depth=depth)
    duck *= g.astype(np.float32)
    echo = Y.pingpong_echo(esend.astype(np.float64), delay_s=0.45, feedback=0.36, lp=3600, hp=380)
    ir = Y.make_reverb_ir(2.4, (1.9, 1.7, 0.85), seed=11)
    rev = Y.reverb(rsend.astype(np.float64) + 0.25 * echo, ir)
    uwf = Y.lowpass(uw.astype(np.float64), 1100.0, order=2) * 1.25     # underwater: filter drop
    mix = main.astype(np.float64) + duck + uwf + 0.55 * echo + 0.42 * rev
    t2 = time.time()
    y, info = master(mix, Y)
    t3 = time.time()
    x16 = Y.to_int16(y, dither=True, dither_mask=info['dither_mask'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Y.write_wav(path, x16)
    if verbose:
        print(f'render: notes {t1 - t0:.1f}s  buses {t2 - t1:.1f}s  master {t3 - t2:.1f}s  -> {path}')
        print('master:', {k: v for k, v in info.items() if k != 'dither_mask'})
    if stems:
        stem_report(roles, Y)
    return y, info


TARGET_LUFS = -14.5
CEILING_DB = -1.5
END_DECAY = (240.45, 242.385, 0.35, 30.0)   # start s, digital-zero s, knee s, dB per second


def master(mix, Y):
    import numpy as np
    N = mix.shape[1]
    x = Y.highpass(mix, 22.0, order=4)
    x = Y.eq(x, 'lowshelf', 110.0, -1.5, 0.7)      # keep the lows tidy (no boom)
    x = Y.eq(x, 'peak', 300.0, -1.0, 0.9)          # de-mud
    x = Y.eq(x, 'highshelf', 9000.0, -1.0, 0.7)    # soften chip fizz a touch
    x -= x.mean(axis=1, keepdims=True)
    x, gr = Y.glue_comp(x, thresh_db=-20.0, ratio=1.6, attack=0.015, release=0.3)
    L = Y.lufs(x)
    x *= 10 ** ((TARGET_LUFS - L) / 20)
    # end: the last chord (struck at 240.0 s) decays exponentially - a straight line in dB, like a
    # room tail - reaching ~-55 dB of extra attenuation at 242.385 s; the last 15 ms are digital zero.
    t = np.arange(N) / SR
    f0, f1, knee, depth = END_DECAY
    u = np.maximum(t - f0, 0.0)
    att_db = np.where(u < knee, depth * u * u / (2 * knee), depth * (u - knee / 2))
    fade = 10 ** (-att_db / 20)
    fade[t >= f1] = 0.0
    # start: from silence
    fs = int(0.05 * SR)
    fade[:fs] *= np.linspace(0, 1, fs) ** 2
    x *= fade
    ceiling = CEILING_DB
    for _ in range(4):
        y, g = Y.limiter(x, ceiling_db=ceiling, lookahead=0.003, hold=0.02)
        tp = Y.true_peak_db(y)
        if tp <= -1.2:
            break
        ceiling -= (tp + 1.2) + 0.05
    y[:, t >= f1] = 0.0
    mask = (fade > 1e-4).astype(np.float64)
    info = dict(lufs_pre=round(L, 2), lufs=round(Y.lufs(y), 2), true_peak=round(tp, 2), ceiling=round(ceiling, 2),
                max_gr_db=round(float(-20 * np.log10(g.min())), 2),
                gr_over_1db_pct=round(float(np.mean(g < 10 ** (-1 / 20)) * 100), 3),
                comp_max_db=round(float(-gr.min()), 2), max_gr_at_s=round(float(np.argmin(g)) / SR, 2),
                dither_mask=mask)
    return y, info


def stem_report(roles, Y):
    import numpy as np
    print(f'{"section":10s}' + ''.join(f'{r:>10s}' for r in sorted(roles)))
    for s in SEC:
        a, b = s['start_beat'] * SPB, s['end_beat'] * SPB
        row = f'{s["id"]:10s}'
        for r in sorted(roles):
            x = roles[r][:, a:b]
            rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
            row += f'{20 * math.log10(rms):10.1f}'
        print(row)


# =========================================================================== helpers for scenes
@lru_cache(maxsize=1)
def _load_sections():
    with open(os.path.join(OUT, 'sections.json')) as f:
        return {s['id']: s for s in json.load(f)}


@lru_cache(maxsize=1)
def _load_events():
    with open(os.path.join(OUT, 'events.json')) as f:
        return json.load(f)


def hits(section):
    """Named musical accents of a section (list of dicts with local_frame, frame, time_s, desc)."""
    return _load_sections()[section]['hits']


def hit_frame(section, name):
    """Local frame of the named accent, e.g. hit_frame('lab', 'bec') -> 360."""
    for h in hits(section):
        if h['name'] == name:
            return h['local_frame']
    raise KeyError(f'{section}: no hit named {name!r}; have {[h["name"] for h in hits(section)]}')


def note_frames(section, voice='lead'):
    """[(local_frame, midi, dur_frames, velocity), ...] of a voice's notes inside a section."""
    return [(e['local_frame'], e['midi'], e['dur_beats'] * FPB, e['velocity'])
            for e in _load_events() if e['section'] == section and e['voice'] == voice]


def beat_frames(section, voice='kick'):
    """Local frames of every hit of a drum voice (kick, snare, clap, hat, ohat, shaker, crash, tom, rim...)."""
    return [e['local_frame'] for e in _load_events() if e['section'] == section and e['voice'] == voice]


def _check_timeline():
    from engine import timeline as TL
    a = [(s['id'], s['bars']) for s in TL.SEGMENTS]
    b = [(s[0], s[1]) for s in SECTIONS]
    assert a == b, f'timeline mismatch: {a} vs {b}'


if __name__ == '__main__':
    _check_timeline()
    S, nfix = compose()
    export_json(S)
    print(f'{len(S.events)} events, {len(S.hits)} hits, {nfix} accompaniment clashes auto-resolved')
    if '--score' not in sys.argv:
        render(S, stems='--stems' in sys.argv)
