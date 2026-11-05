"""EQUESTRIA: THE WORLD ATLAS — the score (22 sections, 101 bars, 100 BPM) and the render/mix pipeline.

    .venv/bin/python audio/compose.py            # compose + render -> audio/out/soundtrack.wav, events.json, sections.json
    .venv/bin/python audio/compose.py --score    # compose only (fast): writes events.json + sections.json
    .venv/bin/python audio/compose.py --stems    # also print per-role / per-section levels (mix debugging)
    .venv/bin/python tools/music_check.py        # measure the result (length, levels, clicks, clashes ...)

v2 "one continuous journey": five movements following the day, one key centre each (max 6 in the film):
    Dawn D (intro-cloudsdale) | Heartland morning G (ponyville-rainbowfalls, Everfree = E minor) |
    The wide world A (manehattan-mountaris, Griffonstone/Dragon Lands = F# minor) |
    Golden hour F (klugetown-laspegasus, Klugetown/Badlands = D minor) | Night D (westcoast-twosisters,
    Yakyakistan = B minor) | finale D -> E.
A continuous atlas band (pad bed tied across bar lines, round bass, gentle arpeggio, light drums, lead) plays
throughout; each region adds one signature colour layer; every section boundary is bridged by sustained
'bed' + 'halo' voices on common tones. The Atlas Theme (| A4 D5 E5 F#5~ A5 | B5~ A5 F#5 D5 | B4 E5 F#5 G5~ B5 |
A5 G5 F#5 E5 |, relative to D) is phrased across travel/diorama boundaries with B-phrase answers; each
region opens with a 2-bar map flight (bar 1 calm departure, bar 2 builds: rising arpeggio, swell, arrive pling
at local 126, reveal crash at 144); every in-scene cut (CUTS, from docs/HITS_v2.json) gets a soft crash/swell.
Earlier versions: audio/compose_v1.py (per-section blocks), audio/compose_v2.py (fluent, 101-bar timeline).

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
TOTAL_BARS = 133
TOTAL_SAMPLES = TOTAL_BARS * 4 * SPB   # 15 321 600 = 319.2 s

# --------------------------------------------------------------------------- section table
# (id, bars, mood, key label, tonic, mode)  mirrors engine/timeline.py (verified at compose time)
SECTIONS = [
    ('intro', 6, 'overture (dawn)', 'D major', 'D', 'major'),
    ('canterlot', 7, 'regal fanfare (dawn)', 'D major', 'D', 'major'),
    ('cloudsdale', 6, 'airy, soaring (dawn)', 'D major (-> G)', 'D', 'major'),
    ('ponyville', 6, 'cozy folk (morning)', 'G major', 'G', 'major'),
    ('sweetapple', 5, 'country hoedown (morning)', 'G major', 'G', 'major'),
    ('everfree', 6, 'mysterious (morning)', 'E minor (relative of G)', 'G', 'major'),
    ('heartland', 6, 'adventurous (morning)', 'G major', 'G', 'major'),
    ('rainbowfalls', 5, 'sporty, bright (morning)', 'G major (-> A)', 'G', 'major'),
    ('manehattan', 7, 'big-city swing (noon peak)', 'A major', 'A', 'major'),
    ('eastcoast', 6, 'maritime shuffle', 'A major', 'A', 'major'),
    ('griffonstone', 5, 'rugged minor', 'F# minor (relative of A)', 'A', 'major'),
    ('dragonlands', 6, 'heavy riff', 'F# minor (relative of A)', 'A', 'major'),
    ('mountaris', 7, 'mediterranean -> underwater', 'A major (-> F)', 'A', 'major'),
    ('klugetown', 6, 'ragtag bazaar (golden hour)', 'D harmonic minor (relative of F)', 'F', 'major'),
    ('badlands', 5, 'dark -> blooming', 'D minor -> F major', 'F', 'major'),
    ('frontier', 5, 'western twang (sunset)', 'F major', 'F', 'major'),
    ('laspegasus', 6, 'showtime funk (dusk)', 'F major (-> D)', 'F', 'major'),
    ('westcoast', 6, 'lo-fi rain (night)', 'D major', 'D', 'major'),
    ('crystal', 7, 'majestic shimmer (night peak)', 'D major', 'D', 'major'),
    ('yakyakistan', 6, 'stomping folk', 'B minor (relative of D)', 'D', 'major'),
    ('twosisters', 7, 'lullaby -> midnight climax', 'D major', 'D', 'major'),
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
    'bed':     dict(inst='padsoft', pan=-0.2, echo=0.0, rev=0.3, role='harmony', gate=1.0),
    'halo':    dict(inst='choir', pan=0.25, echo=0.0, rev=0.4, role='harmony', gate=1.0),
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


# =========================================================================== v3: one continuous journey, roomier
# Five movements following the day, one key centre each (+ the finale lift). Everything is written
# "relative to D" and transposed by the movement's shift (rel D = the movement tonic, rel Bm = its
# relative minor), so a region colours the movement with its relative minor instead of a new key.
# Every region = 2 flight bars (F1 calm departure, F2 build -> arrive 126, reveal 144) + diorama bars.
def _mstart(sid, extra=0):
    return SEC_BY_ID[sid]['start_bar'] + extra


MOVES = [(0, 0, 'D', 'Dawn'), (_mstart('ponyville'), -7, 'G', 'Heartland morning'),
         (_mstart('manehattan'), -5, 'A', 'The wide world'), (_mstart('klugetown'), 3, 'F', 'Golden hour'),
         (_mstart('westcoast'), 0, 'D', 'Night'), (_mstart('outro', 3), 2, 'E', 'Finale lift')]


def kshift(g):
    k = 0
    for b0, s, _, _ in MOVES:
        if g >= b0:
            k = s
    return k


def kname(g):
    n = 'D'
    for b0, _, name, _ in MOVES:
        if g >= b0:
            n = name
    return n


# --------------------------------------------------------------------------- the leitmotif
# "The Atlas Theme" (relative to D) and its relative-minor form (relative B minor). 8th-note units.
TH = {1: 'A4 D5 E5 F#5:3 A5:2', 2: 'B5:3 A5 F#5:2 D5:2', 3: 'B4 E5 F#5 G5:3 B5:2', 4: 'A5:2 G5 F#5 E5:4'}
THM1 = 'F#4 B4 C#5 D5:3 F#5:2'
THM2 = 'G5:3 F#5 D5:2 B4:2'
LIFT = 'B4:4 A4:2 C#5 E5'           # the recurring flight phrase (rises into the reveal)
T2S = 'B5:3 A5 G5:2 E5:2'            # T2 over ii-V
BPH = ['B5:4 A5:2 G5 F#5', 'E5:4 F#5:2 A5:2', 'G5:3 F#5 E5:2 D5:2']   # the B phrase (IV iii ii ...)


def th(*bars, o=0):
    return ' | '.join(tr(TH[b], o) if o else TH[b] for b in bars)


FLIGHT2 = 'Em7:2 A7sus4:1 A7:1'
# Per section: chords per bar (relative to D) and melody entries (voice, local bar, spec, options).
SCORE = {
    'intro': dict(ch=['Dadd9', 'Bm7', 'Gmaj9', 'A7sus4:2 A7:2', 'D', 'Bm7:2 G:2'], mel=[
        ('mbox', 0, th(1, 2, 3, 4), dict(vel=0.6)),
        ('lead2', 2, th(3, 4, o=-12), dict(vel=0.5, inst='mellow')),
        ('lead', 4, 'F#5:2 A5:2 D6:3 C#6 | D6:2 B5:2 G5:2 A5:2 | B5:4 G5:2 E5:2', dict(vel=0.7, inst='lead'))]),
    'canterlot': dict(ch=['Gmaj7:2 Em7:2', 'Bm7:2 A7sus4:1 A7:1', 'A7sus4:2 A7:2', 'D', 'Bm7', 'Em7:2 A7:2',
                          'Gmaj7:2 A:2'], mel=[
        ('lead', 1, 'D5:4 E5:2 F#5:2', dict(vel=0.66, inst='lead')),
        ('brass', 2, 'A4:2 D5 E5 A4:2 C#5 E5 | ' + th(1, 2, 3, 4), dict(vel=0.82)),
        ('brass2', 3, 'F#4 A4 C#5 D5:3 F#5:2 | F#5:3 F#5 D5:2 B4:2 | G4 B4 D5 E5:3 G5:2 | D5:2 D5 D5 C#5:4',
         dict(vel=0.52)),
        ('glock', 3, th(1, o=12), dict(vel=0.2))]),
    'cloudsdale': dict(ch=['Dmaj9', FLIGHT2, 'Gmaj7', 'F#m7', 'Em7:2 F#m7:2', 'Em7:2 D7:2'], mel=[
        ('lead', 0, 'F#5:8 | E5:4 D5:2 C#5:2 | ' + ' | '.join(BPH) + ' | E5:4 D5:2 C5:2+4',
         dict(vel=0.66, inst='lead50')),
        ('glock', 2, ' | '.join(tr(b, 12) for b in BPH), dict(vel=0.16))]),
    'ponyville': dict(ch=['Gadd9', FLIGHT2, 'D', 'Bm7', 'Em7:2 C:2', 'G:2 A7:2'], mel=[
        ('lead', 0, 'r:4 D5:4 | ' + LIFT + ' | ' + th(1, 2, 3), dict(vel=0.8, inst='ocarina')),
        ('whistle', 5, 'D5:1.5 r:0.5 B4:1 r:1', dict(o=12, unit=1.0, vel=0.7)),
        ('lead', 5, 'r:4 E5 F#5 G5 A5', dict(vel=0.72, inst='ocarina'))]),
    'sweetapple': dict(ch=['Bm7', 'Em7:2 D7:2', 'G', 'F#m7:2 Bm7:2', 'Em7:2 A7:2'], mel=[
        ('lead', 0, 'B5:8 | G5:4 A5:2 C6:2 | B5:2 D6 B5 A5:2 G5 F#5 | ' + ' | '.join(BPH[1:]),
         dict(vel=0.8, inst='fiddle')),
        ('lead2', 2, 'G5:2 B5 G5 F#5:2 E5 D5 | C#5:4 D5:2 F#5:2', dict(vel=0.36, inst='fiddle'))]),
    'everfree': dict(ch=['Gmaj7', 'Em7:2 F#7:2', 'Bmadd9', 'G', 'Em7:2 F#7:2', 'F#7b9:2 F#7:2'], mel=[
        ('lead', 0, 'D5:8 | E5:4 C#5:2 A#4:2', dict(vel=0.72, inst='fiddle')),
        ('lead', 2, THM1 + ' | ' + THM2 + ' | E5:3 D5 C#5:2 A#4:2 | r:6 C#5 E5', dict(o=12, vel=0.8, inst='kalimba')),
        ('lead2', 2, 'r:7 F#5 | r:7 D5 | r:8', dict(o=12, vel=0.3, inst='kalimba'))]),
    'heartland': dict(ch=['Bm7', 'Gmaj7:2 A7sus4:1 A7:1', 'Em7:2 A7:2', 'D/F#:2 G:2', 'Em7:2 A7sus4:1 A7:1',
                          'Bm7:2 G:1 A:1'], mel=[
        ('lead', 0, 'r:4 B4:4 | ' + LIFT + ' | B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 B5:4 | G5:3 E5 A5:2 C#6:2 | '
         'D6:4 B5:2 C#6:2', dict(vel=0.8, inst='lead'))]),
    'rainbowfalls': dict(ch=['D', 'G:2 A7sus4:1 A7:1', 'D', 'G:2 A:2', 'Em7:2 B7:2'], mel=[
        ('lead', 0, 'D6:8 | B5:4 A5:2 G5 E5 | D6:3 C#6 B5 A5 F#5 A5 | B5:3 A5 G5:2 E5:2 | G5:4 F#5:2 D#5:2',
         dict(vel=0.8, inst='lead')),
        ('lead2', 2, 'F#5:3 E5 D5 C#5 A4 C#5 | D5:3 C#5 B4:2 C#5:2', dict(vel=0.34))]),
    'manehattan': dict(ch=['Gadd9', FLIGHT2, 'Dmaj7:2 B7:2', 'Em7:2 A7:2', 'Bm7:2 Gmaj7:2', 'Em7:2 A7:2',
                           'Gmaj7:2 F#7:2'], mel=[
        ('brass', 0, 'r:4 D5:4 | ' + LIFT + ' | ' + th(1) + ' | ' + T2S + ' | ' + TH[3] +
         ' | A5:2 G5 F#5 E5:2 C#5 E5 | D6 r B5 F#5 A#5:2 C#6:2', dict(vel=0.84, swing=0.64)),
        ('brass2', 2, 'F#4 A4 C#5 D5:3 F#5:2 | G5:3 F#5 C#5:2 A4:2 | F#4 B4 D5 D5:3 F#5:2 | '
         'E5:2 D5 C#5 C#5:2 A4 C#5 | B5 r G5 D5 F#5:2 A#5:2', dict(vel=0.52, swing=0.64)),
        ('glock', 2, th(1, o=12), dict(vel=0.18, swing=0.64))]),
    'eastcoast': dict(ch=['Bm7', FLIGHT2, 'Gmaj7', 'F#m7', 'Em7:2 A7:2', 'Gmaj7:2 F#7:2'], mel=[
        ('lead', 0, 'D6:8 | B5:4 E6:2 C#6:2 | ' + ' | '.join(BPH) + ' | D5:4 C#5:2 A#4:2',
         dict(vel=0.8, inst='accordion', swing=0.667))]),
    'griffonstone': dict(ch=['Bmadd9', 'Em7:2 F#7:2', 'Bm', 'G', 'Em7:2 F#7:2'], mel=[
        ('brass', 0, 'F#4:8 | B4:4 C#5:2 A#4:2 | ' + THM1 + ' | ' + THM2 + ' | E5:3 D5 C#5:2 A#4:2',
         dict(vel=0.85, drive=1.3))]),
    'dragonlands': dict(ch=['Bm', 'Gmaj7:2 F#7sus4:1 F#7:1', 'Bm', 'G:2 A:2', 'Bm:2 G:2', 'Bm:2 F#7:2'], mel=[
        ('lead', 0, 'D5:8 | B4:4 C#5:2 E5:2 | F#5:3 E5 D5:2 C#5 B4 | D5:2 E5 F#5 E5:4 | B5:3 A5 F#5:2 D5:2 | '
         'F#5:3 G5 F#5:2 A#4:2', dict(vel=0.84, inst='lead', drive=1.8))]),
    'mountaris': dict(ch=['Bm7', 'Gadd9:2 A7sus4:1 A7:1', 'D', 'Bm7:2 G:2', 'Em7:2 A7:2', 'Bm9:2 Gmaj9:2',
                          'Gmaj9:2 D7:2'], mel=[
        ('lead', 0, 'F#5:8 | ' + LIFT + ' | ' + th(1, 2, 3), dict(vel=0.8, inst='warm')),
        ('lead', 5, 'A4:2 D5:2 E5:2 F#5:2 | F#5:3 E5 D5:2 C5:2+4', dict(vel=0.72, inst='mellow', bus='uw'))]),
    'klugetown': dict(ch=['Em7', 'Gmaj7:2 F#7:2', 'Bm', 'Em:2 F#7:2', 'Bm:2 Em:2', 'G:2 F#7:2'], mel=[
        ('lead', 0, 'r:4 E5:4 | D5:4 F#5:2 A#4:2 | F#5:3 E5 D5 C#5 B4 A#4 | G5:3 F#5 E5:2 C#5 A#4 | '
         'B4:3 C#5 E5:2 F#5 G5 | B4:3 D5 C#5:2 A#4:2', dict(vel=0.8, inst='reed'))]),
    'badlands': dict(ch=['Bm', 'Em7:2 F#7:2', 'Bm', 'D', 'Gmaj9:2 C7b9:2'], mel=[
        ('lead', 0, 'B4:8 | E5:4 C#5:4', dict(vel=0.7, inst='reed')),
        ('brass', 2, 'F#4:3 G4 F#4:4', dict(o=-12, vel=0.62)),
        ('lead', 3, th(1) + ' | B5:2 A5 F#5 r:4', dict(vel=0.85, inst='lead')),
        ('glock', 3, th(1, o=12), dict(vel=0.26))]),
    'frontier': dict(ch=['Gadd9', FLIGHT2, 'Bm7', 'Em7:2 A7:2', 'G:2 A7:2'], mel=[
        ('lead', 0, 'B4:4 D5:4 | ' + LIFT + ' | ' + th(2, 3), dict(vel=0.8, inst='twang', slide=1.0)),
        ('whistle', 4, TH[4], dict(vel=0.75))]),
    'laspegasus': dict(ch=['Gmaj7', FLIGHT2, 'Dmaj7:2 Bm7:2', 'Em7:2 A7:2', 'Bm7:2 Gmaj7:2',
                           'Em7:2 F#7sus4:1 F#7:1'], mel=[
        ('lead', 0, 'r:4 B4:4 | ' + LIFT + ' | ' + th(1) + ' | ' + T2S + ' | ' + TH[3] + ' | G5:4 F#5:2 C#5:2+4',
         dict(vel=0.8, inst='lead', drive=1.3))]),
    'westcoast': dict(ch=['Gmaj9', 'Em9:2 A7sus4:1 A7:1', 'Dmaj9', 'Bm9:2 Em9:2', 'Gmaj9:2 A7sus4:1 A7:1',
                          'Bm9:2 Gmaj9:2'], mel=[
        ('lead', 0, 'r:4 D5:4 | B4:4 D5:2 C#5:2 | r:2 F#5 A5 C#6:3 B5 | A5:3 F#5 G5:2 E5:2 | F#5:3 E5 D5:2 E5 C#5 | '
         'F#5:4 D5:4', dict(vel=0.74, inst='mellow', swing=0.6))]),
    'crystal': dict(ch=['Gmaj9', 'Em9:2 A7sus4:1 A7:1', 'D:2 A/C#:2', 'Bm7:2 G:2', 'Gmaj9', 'Bm7:2 G:2',
                        'Em7:2 F#7:2'], mel=[
        ('lead', 0, 'B4:8 | D5:4 E5:2 C#5:2 | ' + th(1, 2, 3) + ' | D6:4 B5:2 A5:2 | ' + TH[4],
         dict(vel=0.82, inst='warm')),
        ('glock', 2, th(1, 2, 3, o=12), dict(vel=0.2))]),
    'yakyakistan': dict(ch=['Bm7', 'Em7:2 F#7sus4:1 F#7:1', 'Bm', 'G:2 A:2', 'Em:2 F#7:2', 'Em7:2 A7:2'], mel=[
        ('brass', 0, 'D5:8 | B4:4 C#5:2 A#4:2 | ' + THM1 + ' | G5:3 F#5 E5:2 C#5:2 | E5:3 D5 C#5:2 A#4:2 | '
         'B4:3 D5 E5:2 G5:2', dict(vel=0.85, drive=1.3))]),
    'twosisters': dict(ch=['Gmaj7', FLIGHT2, 'Dmaj7', 'Bm7', 'Em7:2 A7sus4:1 A7:1', 'D', 'Bm7:2 G:2'], mel=[
        ('lead', 0, 'F#5:8 | B4:4 E5:4', dict(vel=0.6, inst='mellow')),
        ('mbox', 2, th(1, 2), dict(vel=0.6)),
        ('lead', 4, TH[3], dict(vel=0.72, inst='mellow')),
        ('glock', 4, th(3, o=12), dict(vel=0.22)),
        ('brass', 5, 'A5:2 B5 C#6 D6:4 | ' + TH[2], dict(vel=0.9)),
        ('brass2', 5, 'F#5:2 G5 A5 F#5:4 | F#5:3 F#5 D5:2 B4:2', dict(vel=0.56)),
        ('lead', 5, 'A5:2 B5 C#6 D6:4 | ' + TH[2], dict(vel=0.8, inst='lead'))]),
    'outro': dict(ch=['Em7:2 A7:2', 'G:2 A7sus4:1 A7:1', 'D:2 B7sus4:1 B7:1', 'D', 'Bm7:2 G:2', 'Em7:2 A7:2',
                      'Dadd9'], mel=[
        ('lead', 0, 'B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 E5:2 C#5 E5 | D6:2 A5 F#5 E5:2 D#5:2 | ' + th(1, 2) +
         ' | B4 E5 F#5 G5:3 E5 C#5', dict(vel=0.82, inst='lead')),
        ('brass2', 0, 'G4 B4 D5 E5:3 G5:2 | D5:2 D5 D5 D5:2 A4 C#5 | A5:2 F#5 D5 B4:4 | F#4 A4 C#5 D5:3 F#5:2 | '
         'F#5:3 F#5 D5:2 B4:2 | G4 B4 D5 E5:3 C#5 A4', dict(vel=0.5)),
        ('brass', 3, th(1, 2, o=-12) + ' | B3 E4 F#4 G4:3 E4 C#4', dict(vel=0.55)),
        ('glock', 3, th(1, 2, o=12), dict(vel=0.2))]),
}
CH = []
for _s in SECTIONS:
    _c = SCORE[_s[0]]['ch']
    assert len(_c) == _s[1], (_s[0], len(_c), _s[1])
    CH += _c
assert len(CH) == TOTAL_BARS, len(CH)

# dynamics arc per section (local bar, energy 0..1): soft dawn, morning build, noon peak, warm afternoon,
# sunset dip, neon, night, crystal peak, lullaby, midnight climax, finale
E_LOC = {
    'intro': [(0, .22), (2, .36), (3, .42), (4, .5)], 'canterlot': [(0, .48), (1, .52), (2, .5), (3, .68), (5, .66),
                                                                     (6, .6)],
    'cloudsdale': [(0, .56), (1, .58), (2, .6), (5, .6)], 'ponyville': [(0, .55), (1, .56), (2, .58), (5, .6)],
    'sweetapple': [(0, .58), (1, .6), (2, .66), (3, .64), (4, .6)], 'everfree': [(0, .62), (1, .54), (2, .46), (5, .46)],
    'heartland': [(0, .5), (1, .56), (2, .62), (5, .68)], 'rainbowfalls': [(0, .66), (1, .68), (2, .7), (4, .7)],
    'manehattan': [(0, .7), (1, .76), (2, .95), (6, .98)], 'eastcoast': [(0, .8), (1, .76), (2, .7), (5, .7)],
    'griffonstone': [(0, .66), (1, .7), (2, .72), (4, .72)], 'dragonlands': [(0, .66), (1, .68), (2, .7), (5, .68)],
    'mountaris': [(0, .68), (1, .72), (2, .72), (4, .72), (5, .56), (6, .56)],
    'klugetown': [(0, .58), (1, .6), (2, .62), (5, .62)], 'badlands': [(0, .58), (1, .56), (2, .48), (3, .76), (4, .64)],
    'frontier': [(0, .7), (1, .6), (2, .53), (4, .55)], 'laspegasus': [(0, .6), (1, .64), (2, .78), (5, .8)],
    'westcoast': [(0, .64), (1, .58), (2, .5), (5, .52)], 'crystal': [(0, .58), (1, .64), (2, .82), (4, .88), (5, .9),
                                                                      (6, .82)],
    'yakyakistan': [(0, .78), (1, .76), (2, .76), (5, .72)],
    'twosisters': [(0, .56), (1, .5), (2, .32), (3, .36), (4, .6), (5, 1.0), (6, .9)],
    'outro': [(0, .86), (2, .9), (3, .95), (5, .95), (6, .8)],
}
E_PTS = [(SEC_BY_ID[sid]['start_bar'] + b, e) for sid, _b, *_ in SECTIONS for b, e in E_LOC[sid]]


def energy(beat):
    x = beat / 4.0
    for (b0, e0), (b1, e1) in zip(E_PTS, E_PTS[1:]):
        if x <= b1:
            return e0 + (e1 - e0) * max(0.0, x - b0) / (b1 - b0)
    return E_PTS[-1][1]


def section_bars(sid):
    s = SEC_BY_ID[sid]
    return s['start_bar'], s['start_bar'] + s['bars']


GLOBAL = dict(id='*', start_beat=0, bars=TOTAL_BARS)


class Journey(Score):
    """Score written on the global bar grid (bar g = global beat 4g); each event's section is derived from
    its onset, so notes may tie across section boundaries."""

    def __init__(self):
        super().__init__()
        self.sec = GLOBAL

    def add(self, voice, beat, dur, midi, vel=0.8, **opts):
        e = Score.add(self, voice, beat, dur, midi, vel, **opts)
        e['section'] = section_of_beat(e['start_beat'] + 1e-6)
        return e

    def ghit(self, sid, name, lf, desc):
        s = SEC_BY_ID[sid]
        self.hits.append(dict(section=sid, name=name, desc=desc, bar=lf // 72 + 1,
                              beat=round((lf % 72) / FPB + 1, 3), global_beat=s['start_beat'] + lf / FPB))

    def pp(self, name, g, o=0):
        return P(name) + kshift(g) + o

    def gmel(self, voice, g, spec, o=0, unit=0.5, vel=0.8, swing=None, late=0.0, **opts):
        """Melody on the global grid from bar g, written relative to D (each note transposed by the shift of
        the bar it starts in). Token PITCH[:dur[+extra]][!_] ; '+extra' lets a note ring past the bar line."""
        out = []
        for bi, chunk in enumerate(spec.split('|')):
            t0 = 4.0 * (g + bi)
            acc = 0.0
            for tok in chunk.split():
                m = re.fullmatch(r'([A-Gr][#b]?-?\d?)(?::([\d./]+)(?:\+([\d.]+))?)?([!_]*)', tok)
                if not m:
                    raise ValueError(f'bad token {tok!r}')
                ds = m.group(2) or '1'
                d = (float(ds.split('/')[0]) / float(ds.split('/')[1]) if '/' in ds else float(ds)) * unit
                extra = float(m.group(3) or 0) * unit
                if m.group(1) != 'r':
                    pb = acc
                    vv = vel * (1.0 if pb < 1e-6 else 0.97 if abs(pb - 2) < 1e-6 else
                                0.94 if abs(pb - round(pb)) < 1e-6 else 0.9) * (1.03 if d >= 1.0 else 1.0)
                    if '!' in m.group(4):
                        vv += 0.1
                    o_ = dict(opts)
                    if '_' in m.group(4):
                        o_['gate'] = 0.5
                    pos = t0 + (swing_pos(acc, swing) if swing else acc) + late
                    k = kshift(int((t0 + acc + 1e-6) // 4))
                    out.append(self.add(voice, pos, d + extra, P(m.group(1)) + k + o, vv, **o_))
                acc += d
            if abs(acc - 4.0) > 1e-6:
                raise ValueError(f'bar {g + bi} of {spec!r} lasts {acc} beats')
        return out

    def gdrums(self, g0, g1, pats, swing=None, vel=1.0, human=0.0):
        sec = self.sec
        for g in range(g0, g1):
            self.sec = dict(GLOBAL, start_beat=4 * g)
            self.drums(0, 1, pats, swing=swing, vel=vel, human=human)
        self.sec = sec

    def gbass(self, g0, g1, pattern, lo=36, vel=0.8, swing=None, **opts):
        sec = self.sec
        self.sec = GLOBAL
        self.bassline('bass', g0, g1, pattern, lo=lo, vel=vel, swing=swing, **opts)
        self.sec = sec


def build_chords(S):
    for g, spec in enumerate(CH):
        k = kshift(g)
        toks = trc(spec, k, kname(g) == 'F').split()
        fixed = sum(float(t.split(':')[1]) for t in toks if ':' in t)
        free = [t for t in toks if ':' not in t]
        each = (4.0 - fixed) / len(free) if free else 0.0
        pos = 4.0 * g
        for tok in toks:
            sym, _, d = tok.partition(':')
            d = float(d) if d else each
            S.chords.append((pos, pos + d, Chord(sym), section_of_beat(pos + 1e-6)))
            pos += d
        assert abs(pos - 4.0 * (g + 1)) < 1e-9, (g, spec)


def melody(S):
    for sid, *_ in SECTIONS:
        g0 = SEC_BY_ID[sid]['start_bar']
        for voice, lb, spec, kw in SCORE[sid]['mel']:
            S.gmel(voice, g0 + lb, spec, **kw)


# =========================================================================== the band (continuous)
BOUNCE = [(0, 'R', 0.75), (1, 'R+12', 0.4, 0.75), (1.5, 'R', 0.4, 0.85), (2.5, '5', 0.4, 0.8),
          (3, 'R+12', 0.4, 0.75), (3.5, 'R', 0.4, 0.85)]
OCT8 = [(i * 0.5, 'R' if i % 2 == 0 else 'R+12', 0.42, 1.0 if i % 2 == 0 else 0.78) for i in range(8)]
BASS = {
    'long': [(0, 'R', 3.9)], 'half': [(0, 'R', 1.9), (2, 'R', 0.9, 0.85), (3, '5-12', 0.9, 0.8)],
    'lift': [(0, 'R', 1.9), (2, 'R', 0.9, 0.9), (3, 'R', 0.9, 0.9)],
    'folk': [(0, 'R', 0.9), (1, '5-12', 0.9, 0.8), (2, 'R', 0.9, 0.9), (3, '5-12', 0.9, 0.8)],
    'boomchick': [(0, 'R', 0.9), (1, 'R+12', 0.4, 0.55), (2, '5-12', 0.9, 0.85), (3, 'R+12', 0.4, 0.55)],
    'bounce': BOUNCE, 'oct8': OCT8, 'pulse8': [(i * 0.5, 'R', 0.45, 1.0 if i % 2 == 0 else 0.8) for i in range(8)],
    'funk': [(0, 'R', 0.4), (0.75, 'R+12', 0.2, 0.7), (1.5, 'R', 0.3, 0.85), (2, 'R+12', 0.2, 0.7),
             (2.5, 'R', 0.4, 0.9), (3.25, 'R+12', 0.2, 0.7), (3.5, '5', 0.3, 0.8)],
    'lofi': [(0, 'R', 1.2), (1.5, '5-12', 0.4, 0.7), (2.5, 'R', 0.9, 0.9), (3.5, '8', 0.4, 0.6)],
    'stomp': [(0, 'R', 0.9), (1, 'R', 0.9, 0.7), (2, '5-12', 0.9, 0.85), (3, 'R', 0.9, 0.7)],
}
CORE_DRUMS = [
    (0.45, {'shaker': 'g.g.g.g.g.g.g.g.', 'kick_soft': 'o.......o.......'}),
    (0.7, {'kick_soft': 'x.......o.......', 'shaker': 'gogogogogogogogo', 'hat_soft': '..g...g...g...g.'}),
    (9.0, {'kick_soft': 'x.......x.......', 'snare_soft': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.',
           'shaker': '.g.g.g.g.g.g.g.g'}),
]
# regional groove = core band + one signature layer (bars after the lift bar)
REG = {
    'canterlot': dict(bass='half', drums={'kick': 'x.......x.......', 'snare_soft': '....x.......x.gg',
                                          'hat_soft': 'g.g.g.g.g.g.g.g.'}),
    'cloudsdale': dict(bass='half', drums={'kick_soft': 'x.........x.....', 'snare_soft': '....o.......o...',
                                           'hat_soft': 'g.o.g.o.g.o.g.o.', 'shaker': '.g.g.g.g.g.g.g.g'}),
    'ponyville': dict(bass='folk', binst='upright', drums={'kick_soft': 'x.......x.......', 'clap': '....o.......o...',
                                                          'shaker': 'gogggogggogggogg'}),
    'sweetapple': dict(bass='boomchick', drums={'kick_soft': 'x.......x.......', 'clap': '....x.......x...',
                                                'brush': '..o...o...o...o.', 'hat_soft': 'g.g.g.g.g.g.g.g.'}),
    'everfree': dict(bass='half', binst='upright', drums={'tom_lo': 'x.......o.......', 'tom_mid': '......o.......g.',
                                                          'shaker': 'g...g...g...g...', 'kick_soft': 'o...............'}),
    'heartland': dict(bass='oct8', binst='bassoct', drums={'kick': 'x.....x.x.......', 'snare': '....x.......x...',
                                                           'hat': 'g.o.g.o.g.o.g.o.'}),
    'rainbowfalls': dict(bass='oct8', binst='pulsebass', drums={'kick': 'x...x...x...x...', 'snare': '....x.......x...',
                                                                'hat': 'gogggogggogggogg', 'ohat': '..............o.'}),
    'manehattan': dict(bass='walk', binst='upright', swing=0.64,
                       drums={'ride': 'x...x.o.x...x.o.', 'hat_soft': '....x.......x...', 'kick_soft': 'x.......x.......',
                              'snare_soft': '....o.g.....o.g.'}),
    'eastcoast': dict(bass='boomchick', binst='upright', swing=0.667,
                      drums={'kick_soft': 'x.......x.......', 'snare_soft': '....o.......o...',
                             'shaker': 'g.o.g.o.g.o.g.o.', 'tick': '..g.......g.....'}),
    'griffonstone': dict(bass='half', drums={'boom': 'x.......x.......', 'tom_lo': 'x..x..x.x..x..x.',
                                             'tom_mid': '......o.......o.', 'snare_soft': '....o.......o...'}),
    'dragonlands': dict(bass='pulse8', binst='bassoct', drums={'kick': 'x..x..x.x..x....', 'snare': '....x.......x...',
                                                               'hat': 'x.x.x.x.x.x.x.x.'}),
    'mountaris': dict(bass='folk', binst='upright', drums={'shaker': 'gogggogggogggogg', 'rim': '....o.......o..g',
                                                           'kick_soft': 'x.......x.......'}),
    'klugetown': dict(bass='folk', binst='upright', drums={'kick_soft': 'x.......x..x....', 'rim': 'x..x..x...x..x..',
                                                           'tick': '..g...g.g...g..g', 'tom_hi': '.......o.......o',
                                                           'clap': '....o.......o...'}),
    'badlands': dict(bass='half', drums={'kick': 'x.....x.x.......', 'snare': '....x.......x...',
                                         'hat': 'g.o.g.o.g.o.g.o.'}),
    'frontier': dict(bass='boomchick', drums={'tick': 'x.o.x.o.x.o.x.o.', 'rim': '...o...o...o...o',
                                              'kick_soft': 'x.......x.......', 'snare_soft': '....o.......o...'}),
    'laspegasus': dict(bass='funk', binst='pulsebass', drums={'kick': 'x...x...x...x...', 'clap': '....x.......x...',
                                                              'ohat': '..o...o...o...o.', 'hat': 'g.g.g.g.g.g.g.g.'}),
    'westcoast': dict(bass='lofi', binst='upright', swing=0.6,
                      drums={'kick_lofi': 'x......x..x.....', 'snare_lofi': '....x.......x...', 'hat': 'o.g.o.g.o.g.o.gg'}),
    'crystal': dict(bass='half', drums={'kick_soft': 'x.........x.....', 'snare': '........x.......',
                                        'hat_soft': 'g.g.g.g.g.g.g.g.'}),
    'yakyakistan': dict(bass='stomp', drums={'boom': 'x.......x.......', 'kick': 'x.......x.......',
                                             'clap': '....x.......x...', 'tom_lo': '......o.......o.'}),
    'twosisters': dict(bass='long', binst='upright', drums={'shaker': 'g...g...g...g...'}),
}


def walking(S, g):
    """Quarter-note walking bass for bar g: root, chord tone, chord tone, chromatic approach."""
    for q in range(4):
        b = 4 * g + q
        c = S.chord_at(b)
        r = 31 + (c.bass_pc - 31) % 12
        third = r + (3 if 3 in c.ivs else 4)
        if q == 0:
            m = r
        elif q == 1:
            m = third if (b + 1) % 2 else r + 7
        elif q == 2:
            m = r if S.chord_at(b - 1) is not c else r + 7
        else:
            nr = S.chord_at(b + 1)
            tgt = 31 + (nr.bass_pc - 31) % 12
            m = tgt - 1 if tgt - 1 >= 31 else tgt + 1
        S.add('bass', b, 0.92, m, 0.8 if q % 2 == 0 else 0.7, inst='upright', gate=0.85)


def band(S):
    """The atlas band: pad bed (tied across bar lines), round bass, gentle arpeggio, light drums."""
    prev, last = None, {}
    for b0, b1, c, _ in S.chords:
        e = energy(b0)
        v = voicing(c, 50, 67, 4 if e > 0.6 else 3, prev)
        new_last = {}
        for mm in v:
            pe = last.get(mm)
            if pe is not None and abs(pe['start_beat'] + pe['dur_beats'] - b0) < 1e-6:
                pe['dur_beats'] = round(b1 - pe['start_beat'], 6)
                new_last[mm] = pe
            else:
                new_last[mm] = S.add('pad', b0, b1 - b0, mm, 0.3 + 0.2 * e, inst='padsoft' if e < 0.55 else 'pad')
        last, prev = new_last, v
    regions = {s['id']: s for s in SEC}
    last = TOTAL_BARS - 1
    tw0 = SEC_BY_ID['twosisters']['start_bar'] + 2         # lullaby bars tw0, tw0+1; build tw0+2; midnight tw0+3
    uw0 = SEC_BY_ID['mountaris']['start_bar'] + 5          # underwater bars uw0, uw0+1
    ou0 = SEC_BY_ID['outro']['start_bar']
    for g in range(TOTAL_BARS):
        sid = section_of_beat(4 * g + 1e-6)
        s0 = SEC_BY_ID[sid]['start_bar']
        e = energy(4 * g)
        cfg = REG.get(sid, {})
        lift = regions[sid]['region'] and g in (s0, s0 + 1)
        calm = regions[sid]['region'] and g == s0          # flight bar 1: calm departure
        # gentle arpeggio (8ths) everywhere except the night-sky opening and the outro's final chord
        if 1 <= g < last:
            step = 1.0 if tw0 <= g <= tw0 + 1 else 0.5
            S.arp('triarp', 4 * g, 4 * g + 4, step=step, lo=55, hi=74, pattern='updown', vel=0.16 + 0.2 * e,
                  pans=(-0.4, 0.4))
        # bass
        if sid == 'intro':
            pat = None if g == 0 else 'long' if g < 4 else 'bounce'
        elif sid == 'outro':
            pat = 'bounce' if g < last else None
        elif calm:
            pat = 'long'
        elif lift:
            pat = 'lift'
        elif cfg.get('bass') == 'walk':
            walking(S, g)
            pat = None
        else:
            pat = cfg.get('bass', 'half')
        if uw0 <= g <= uw0 + 1 or tw0 <= g <= tw0 + 1:
            pat = 'long'
        if g == tw0 + 2:
            pat = 'pulse8'
        if pat:
            inst = 'upright' if pat == 'long' else 'bass' if lift else cfg.get('binst', 'bass')
            S.gbass(g, g + 1, BASS[pat], lo=36, vel=0.62 + 0.25 * e, swing=None if lift else cfg.get('swing'),
                    inst=inst)
        # drums
        if sid == 'intro':
            pats = {1: {'shaker': 'g.......g.......'}, 2: {'shaker': 'g...g...g...g...'},
                    3: {'shaker': 'g.g.g.g.gogogogo'}}.get(g) if g < 4 else \
                {'kick_soft': 'x.....o.x.......', 'clap': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.',
                 'shaker': '.g.g.g.g.g.g.g.g'}
        elif sid == 'outro':
            pats = None if g == last else {'kick': 'x.....x.x.......', 'snare': '....x.......x...',
                                           'hat': 'g.o.g.o.g.o.g.o.', 'ohat': '..............o.'} if g >= ou0 + 3 else \
                {'kick_soft': 'x.......x.......', 'clap': '....o.......o...', 'hat_soft': 'g.o.g.o.g.o.g.o.'}
        elif calm:
            pats = next(p for th_, p in CORE_DRUMS if e - 0.15 < th_)
        elif lift or 'drums' not in cfg:
            pats = next(p for th_, p in CORE_DRUMS if e < th_)
        else:
            pats = cfg['drums']
        if tw0 <= g <= tw0 + 1:
            pats = {'shaker': 'g...g...g...g...'}
        if g == tw0 + 2:
            pats = {'kick_soft': 'x.......x.......', 'shaker': 'gogogogogogogogo'}
        if g in (tw0 + 3, tw0 + 4):
            pats = {'kick': 'x.......x.......', 'snare': '....x.......x...', 'hat': 'g.o.g.o.g.o.g.o.'}
        if uw0 <= g <= uw0 + 1:
            pats = {'kick_soft': 'o.......o.......'}
        if pats:
            S.gdrums(g, g + 1, pats, swing=None if lift else cfg.get('swing'), vel=0.75 + 0.3 * e)
    for sid in [s['id'] for s in SEC if s['region']]:
        if sid in ('cloudsdale', 'everfree', 'heartland', 'badlands', 'twosisters'):
            continue                                # last bar has its own accent/fill
        g1 = section_bars(sid)[1] - 1
        for i in range(4):
            S.drum('snare_soft', 4 * g1 + 3 + 0.25 * i, 0.25 + 0.08 * i)


def bridges(S):
    """Every section boundary is bridged by two sustained voices (low 'bed' + high 'halo') on tones common to
    the chords either side, entering a beat early and ringing 1.5 beats past the bar line."""
    pitched = [e for e in S.events if e['midi'] is not None and 'drum' not in e and e['voice'] != 'fx' and
               VOICES.get(e['voice'], {}).get('role') != 'accomp']      # accompaniment moves around the bridges
    for s in SEC[1:]:
        B = s['start_beat']
        ca, cb = S.chord_at(B - 0.01), S.chord_at(B)
        near = [p for p in pitched if p['start_beat'] < B + 1.5 and p['start_beat'] + p['dur_beats'] > B - 1.0]
        def free(mm):
            return not any(abs(p['midi'] - mm) == 1 for p in near)
        cand = [mm for mm in range(48, 85) if mm % 12 in ca.pcs and mm % 12 in cb.pcs and free(mm)]
        if len(cand) < 2:
            cand += [mm for mm in cb.tones(48, 84) if free(mm) and mm not in cand and
                     all((mm - x) % 12 not in (1, 11) for x in ca.pcs)]
        cand.sort()
        picks = []
        if cand:
            picks.append(('bed', min(cand, key=lambda x: abs(x - 57)), 'padsoft', 0.42))
            hi = [mm for mm in cand if mm >= picks[0][1] + 7]
            if hi:
                picks.append(('halo', min(hi, key=lambda x: abs(x - 74)), 'choir', 0.3))
        for voice, mm, inst, v in picks:
            S.add(voice, B - 1.0, 2.5, mm, v, inst=inst)
            near.append(dict(midi=mm, start_beat=B - 1.0, dur_beats=2.5))


def lift(S, sid, crash='crash', cv=0.85):
    """The 2-bar map flight: F1 = calm departure (a gentle rising 8th figure), F2 = the flight builds (rising 16th
    arpeggio left -> right), ARRIVE pling on beat 4 of F2 (local 126), reverse swell + riser into the REVEAL crash
    on the diorama's first downbeat (local 144)."""
    g = SEC_BY_ID[sid]['start_bar']
    t = 67 + ((2 + kshift(g)) % 12 - 7) % 12           # movement tonic in G4..F#5
    for i in range(8):
        b = 4 * g + 0.5 * i
        c = S.chord_at(b)
        mm = min(x for x in c.tones(40, 104) if x >= t - 19 + i * 1.6)
        S.add('arp', b, 0.5, mm, 0.13 + 0.07 * i / 7, pan=-0.5 + i / 14, inst='triarp')
    e = energy(4 * (g + 1))
    for i in range(16):
        b = 4 * (g + 1) + i * 0.25
        c = S.chord_at(b)
        mm = min(x for x in c.tones(40, 104) if x >= t - 14 + i * 2.0)
        S.add('arp', b, 0.25, mm, 0.18 + 0.12 * e + 0.22 * i / 15, pan=-0.55 + 1.1 * i / 15)
    gb = 4 * (g + 1)
    p1 = 84 + (t - 84) % 12
    S.add('pling', gb + 3, 1.5, p1, 0.5)
    S.add('pling', gb + 3, 1.5, p1 + 7, 0.4)
    S.fx('shimmer', gb + 3, 1.2, 0.3, f0=1760.0 * 2 ** (((t - 69) % 12) / 12))
    S.fx('swell', 0, 2.5, 0.45, ends_at=gb + 4)
    S.fx('riser', 0, 1.5, 0.26, ends_at=gb + 4, lo=700.0, hi=9000.0, f0=330.0, octs=2.0)
    S.drum(crash, gb + 4, cv)
    S.drum('kick' if crash == 'crash' else 'kick_soft', gb + 4, 0.7)
    S.ghit(sid, 'arrive', 126, 'ARRIVE: glock pling (tonic + fifth) + shimmer on beat 4 of the second flight bar; '
                               'the reverse swell grows toward the reveal')
    S.ghit(sid, 'reveal', 144, 'REVEAL: riser lands, crash on the diorama downbeat; the regional colour layer is in')


# in-scene transitions (film-local frames, docs/HITS_v2.json): a soft crash (and a short swell for cloud passes
# and dissolves) lands on each cut frame
CUTS = {
    'canterlot': [(288, 'dissolve'), (396, 'glide')], 'cloudsdale': [(288, 'zoom'), (360, 'cloud')],
    'ponyville': [(288, 'glide')], 'sweetapple': [(288, 'glide')], 'everfree': [(234, 'glide'), (324, 'glide')],
    'heartland': [(252, 'cloud'), (342, 'cloud')], 'rainbowfalls': [(288, 'cloud')],
    'manehattan': [(234, 'dissolve'), (324, 'glide'), (414, 'glide')], 'eastcoast': [(252, 'cloud'), (342, 'glide')],
    'griffonstone': [(216, 'cloud')], 'dragonlands': [(252, 'zoom'), (342, 'cloud')],
    'mountaris': [(252, 'dissolve'), (360, 'dissolve'), (396, 'dissolve')], 'klugetown': [(252, 'zoom'), (342, 'glide')],
    'badlands': [(288, 'dissolve')], 'frontier': [(288, 'cloud')], 'laspegasus': [(288, 'dissolve'), (360, 'dissolve')],
    'westcoast': [(252, 'cloud'), (342, 'cloud')], 'crystal': [(234, 'dissolve'), (324, 'dissolve'), (414, 'glide')],
    'yakyakistan': [(252, 'zoom'), (342, 'cloud')], 'twosisters': [(234, 'zoom'), (324, 'dissolve')],
}


def L(sid, f):
    """Global beat of film-local frame f of section sid."""
    return SEC_BY_ID[sid]['start_beat'] + f / FPB


def cuts(S):
    for sid, lst in CUTS.items():
        for f, style in lst:
            b = L(sid, f)
            S.drum('crash_soft', b, 0.36 if style in ('cloud', 'dissolve') else 0.3)
            if style in ('cloud', 'dissolve'):
                S.fx('swell', 0, 0.75, 0.24, ends_at=b)


def colours(S):
    """One signature colour per region on top of the band, fading in over the flight, out over the last bar."""
    a = S.add

    def fade(g, g0, g1):
        return 0.4 if g == g0 else 0.6 if g == g0 + 1 else 0.75 if g == g1 - 1 else 1.0

    # ---------------- intro (night sky -> title -> dawn -> begin journey -> liftoff); unchanged positions
    for i, nm in enumerate(['A7', 'F#7', 'E7', 'D7', 'A6']):
        a('glock', 2 + 0.125 * i, 0.5, P(nm), 0.2 - 0.025 * i, pan=0.5 - 0.22 * i)
    S.ghit('intro', 'star', 36, 'shooting star: soft falling glock sparkle (beat 3)')
    S.gmel('glock', 1, 'r:4 D6 F#6 A6 C#7', vel=0.28)
    S.fx('swell', 0, 2.0, 0.65, ends_at=8)
    S.roll('tom_lo', 7, 8, 0.125, 0.15, 0.5)
    S.drum('boom', 8, 0.7)
    S.drum('crash', 8, 0.8)
    S.fx('shimmer', 8, 4.0, 0.6, f0=1760.0)
    a('timp', 8, 2.0, P('G2'), 0.75)
    for i, nm in enumerate(['D6', 'G6', 'A6', 'B6', 'D7', 'B6', 'A6', 'G6', 'A6']):
        a('glock', 8 + 0.5 * i, 0.5, P(nm), 0.48 if i in (0, 8) else 0.4, pan=0.5 * math.sin(i * 1.3))
    S.ghit('intro', 'title', 144, 'TITLE DROP: boom + crash + shimmer; letter plings (glock) on every 8th: local '
                                  'frames 144,153,...,216 (9 letters, the 9th on the bar-4 downbeat)')
    a('timp', 12, 1.0, P('A2'), 0.65)
    S.arp('arp', 12, 16, step=0.25, lo=57, hi=86, pattern='up', vel=0.3, vel_ramp=(0.5, 1.0))
    S.roll('snare_soft', 15, 16, 0.125, 0.2, 0.6)
    S.fx('swell', 0, 2.0, 0.5, ends_at=16)
    S.ghit('intro', 'dawn', 216, 'dawn light sweeps the map: rising arpeggio build + snare roll into bar 5')
    S.drum('crash_soft', 16, 0.6)
    for q in range(4):
        a('blip', 16 + q, 0.25, P('A6'), 0.34 if q == 0 else 0.28, pan=0.3)
    S.ghit('intro', 'prompt', 288, 'BEGIN JOURNEY prompt: blip on every beat (288, 306, 324, 342)')
    S.fx('riser', 0, 1.0, 0.42, ends_at=20, lo=400.0, hi=8000.0, f0=250.0, octs=2.0)
    S.drum('tick', 20, 0.7)
    S.drum('crash_soft', 20, 0.55)
    a('blip', 20, 0.25, P('D7'), 0.42)
    for i, nm in enumerate(['D5', 'F#5', 'A5', 'B5', 'D6', 'F#6', 'A6', 'B6', 'D7']):
        a('glock', 20 + 0.125 * i, 0.4, P(nm), 0.22 + 0.025 * i, pan=-0.5 + 0.12 * i)
    S.ghit('intro', 'liftoff', 360, 'LIFTOFF: riser lands, pop (tick + blip + soft cymbal), whoosh-up glock glissando')
    S.sec = GLOBAL
    # ---------------- canterlot: harpsichord + timpani (royal); pre-dawn pedal, sunrise
    sid = 'canterlot'
    g0, g1 = section_bars(sid)
    r0 = g0 + 2
    lift(S, sid, crash='crash_soft', cv=0.6)
    for g in range(g0, g1):
        S.arp('harpsi', 4 * g, 4 * g + 4, step=0.25 if g >= r0 else 0.5, lo=50, hi=69, pattern='updown',
              vel=0.38 * fade(g, g0, g1), pans=(-0.3, 0.3))
    a('timp', 4 * r0, 1.0, P('A1'), 0.55)
    a('timp', 4 * r0 + 2, 1.0, P('A1'), 0.6)
    S.roll('tom_lo', 4 * r0 + 2, L(sid, 216), 0.125, 0.15, 0.6)
    S.fx('swell', 0, 2.0, 0.75, ends_at=L(sid, 216))
    S.drum('crash', L(sid, 216), 0.9)
    S.drum('boom', L(sid, 216), 0.65)
    S.fx('shimmer', L(sid, 216), 4.0, 0.5, f0=1760.0)
    for g in range(r0 + 1, g1):
        a('timp', 4 * g, 1.0, 38 + (S.chord_at(4 * g).root - 38) % 12, 0.6)
    S.ghit(sid, 'sunrise', 216, 'SUNRISE: reverse swell lands on crash + boom; the Atlas Theme in full brass')
    # ---------------- cloudsdale: airy lead50 + glock doubling, padbright shimmer, rainbow glissando
    sid = 'cloudsdale'
    g0, g1 = section_bars(sid)
    lift(S, sid, crash='crash_soft', cv=0.7)
    S.pad('choir', g0 + 2, g1, lo=62, hi=79, n=3, vel=0.32, inst='padbright')
    rb = L(sid, 360)
    S.drum('crash_soft', rb, 0.7)
    for i, nm in enumerate(['Bb5', 'D6', 'F6', 'A6', 'Bb6', 'D7', 'F7', 'A7']):
        a('glock', rb + 0.0625 * i, 0.15, P(nm), 0.28 + 0.03 * i, pan=-0.6 + 0.16 * i)
    S.fx('shimmer', rb, 1.5, 0.38, f0=1400.0)
    S.ghit(sid, 'rainbow', 360, 'RAINBOW: fast glissando sparkle (glock run up, 360-369) + shimmer')
    # ---------------- ponyville: pluck picking + ocarina, train whistle
    sid = 'ponyville'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    for g in range(g0 + 1, g1):
        S.arp('stab', 4 * g, 4 * g + 4, step=0.5, lo=67, hi=84, group=[0, 2, 1, 3], vel=0.44 * fade(g, g0, g1),
              pans=(0.35, 0.5), inst='pluck')
    tb = L(sid, 360)
    for i in range(16):
        S.drum('shaker', tb + 0.25 * i, 0.7 if i % 4 == 0 else 0.4)
    S.ghit(sid, 'train', 360, 'TRAIN: two-note whistle (G5 then E5, "whoo-whooo") at 360 and 378; chugging shaker')
    # ---------------- sweetapple: banjo rolls + fiddle; zap
    sid = 'sweetapple'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    for g in range(g0 + 1, g1):
        S.arp('banjo', 4 * g, 4 * g + 4, step=0.25, lo=55, hi=74, group=[0, 2, 4, 1, 3, 5],
              vel=0.4 * fade(g, g0, g1), pans=(0.3, 0.45))
    zb = L(sid, 288)
    S.fx('zap', zb, 0.8, 0.55, f0=3200.0, f1=400.0)
    S.drum('crash', zb, 0.75)
    tones = S.chord_at(zb).tones(76, 91)
    for i, mm in enumerate((tones + tones[-2::-1])[:16]):
        a('blip', zb + 0.125 * i, 0.15, mm, 0.4 if i < 8 else 0.28, inst='arp25', pan=-0.6 + 0.08 * i)
    S.ghit(sid, 'zap', 288, 'ZAP: zappy rainbow arpeggio (pulse 32nds up and back down, 288-323) + laser zap')
    # ---------------- everfree (E minor): kalimba, soft toms, low shadows; glimpse
    sid = 'everfree'
    g0, g1 = section_bars(sid)
    r0 = g0 + 2
    lift(S, sid, crash='crash_soft', cv=0.5)
    for bar, beat, nm in [(0, 1.5, 'D6'), (0, 3.5, 'A6'), (1, 0.5, 'B6'), (2, 2.5, 'F#6')]:
        a('glock', 4 * (r0 + bar) + beat, 0.5, S.pp(nm, g0), 0.16)
    gb = L(sid, 360)
    S.fx('swell', 0, 1.0, 0.42, ends_at=gb)
    for mm in voicing(S.chord_at(gb), 52, 70, 4):
        a('keys', gb, 1.4, mm, 0.6)
    a('bell', gb, 2.0, S.chord_at(gb).tones(76, 88)[0], 0.28)
    S.drum('tom_lo', gb, 0.6)
    S.ghit(sid, 'glimpse', 360, 'GLIMPSE: eerie B7b9 chord stab on the downbeat (e-piano + bell, reverse swell into it)')
    for i in range(3):
        S.drum('tom_mid', gb + 1.5 + 0.33 * i, 0.3 + 0.08 * i)
    # ---------------- heartland: brass stabs + toms, postcard cuts
    sid = 'heartland'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    S.comp('stab', g0 + 2, g1, [(0.5, 0.25, 0.8), (1.5, 0.25, 0.7), (2.5, 0.25, 0.8), (3.5, 0.25, 0.7)],
           lo=55, hi=67, n=3, vel=0.46, inst='brass')
    for f, nm in ((252, 'cut2'), (342, 'cut3')):
        S.drum('crash', L(sid, f), 0.8)
        S.drum('boom', L(sid, f), 0.4)
        S.ghit(sid, nm, f, f'postcard cut ({nm}): crash + boom on the beat (cloud pass)')
    for i, nm in enumerate(['tom_hi', 'tom_hi', 'tom_mid', 'tom_lo']):
        S.drum(nm, 4 * (g1 - 1) + 3 + 0.25 * i, 0.5 + 0.06 * i)
    # ---------------- rainbowfalls: driving pulse arp; trestle
    sid = 'rainbowfalls'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    S.arp('arp', 4 * (g0 + 2), 4 * g1, step=0.25, lo=67, hi=86, pattern='up', vel=0.3, pans=(-0.55, 0.55),
          lp_ramp=(3000, 7000))
    S.duck.append((4 * (g0 + 2), 4 * g1, 0.25))
    tb = L(sid, 216)
    S.drum('crash', tb, 0.85)
    S.drum('boom', tb, 0.5)
    S.comp('stab', g0 + 3, g0 + 4, [(0, 0.5, 1.0)], lo=60, hi=72, n=3, vel=0.68, inst='brass')
    S.ghit(sid, 'trestle', 216, 'TRESTLE: crash + boom + brass stab (the train crosses the trestle)')
    # ---------------- manehattan (noon peak): swing ride + walking bass + e-piano comping; bridge shout
    sid = 'manehattan'
    g0, g1 = section_bars(sid)
    lift(S, sid, cv=0.9)
    S.comp('keys', g0 + 2, g1, [(0, 1.0, 1.0), (1.5, 0.4, 0.8)], lo=55, hi=67, n=3, vel=0.46, swing=0.64, human=0.6)
    bb = L(sid, 432)
    S.fx('swell', 0, 1.0, 0.45, ends_at=bb)
    S.drum('crash', bb, 0.88)
    S.drum('boom', bb, 0.55)
    S.ghit(sid, 'bridge', 432, 'BRIDGE: brass shout + crash + boom (the big suspension bridge)')
    # ---------------- eastcoast: accordion (lead + off-beat pulses), harbour bell
    sid = 'eastcoast'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    S.comp('keys', g0 + 2, g1, [(0.5, 0.3, 1.0), (1.5, 0.3, 0.8), (2.5, 0.3, 1.0), (3.5, 0.3, 0.8)], lo=60, hi=72,
           n=3, vel=0.38, swing=0.667, inst='accordion')
    a('bell', L(sid, 180), 6.0, S.pp('D4', g0), 0.62)
    S.drum('boom', L(sid, 180), 0.32)
    S.ghit(sid, 'bell', 180, 'BELL: deep harbour-bell stroke (A3) on beat 3, rings ~4 s')
    # ---------------- griffonstone (F# minor): low brass, war toms, wind; gust
    sid = 'griffonstone'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    S.fx('sweep', 4 * g0, 6.0, 0.24, hi=3000.0, lo=500.0)
    S.fx('sweep', 4 * (g0 + 2), 4.0, 0.28, hi=4000.0, lo=600.0)
    gb = L(sid, 288)
    S.fx('riser', 0, 1.0, 0.45, ends_at=gb, lo=500.0, hi=6000.0, f0=200.0, octs=1.5)
    S.fx('sweep', gb, 3.0, 0.55, hi=8000.0, lo=300.0)
    S.drum('crash_soft', gb, 0.65)
    S.ghit(sid, 'gust', 288, 'GUST: wind whoosh (noise riser into a long downward sweep) + soft cymbal')
    # ---------------- dragonlands: power-octave riff; erupt
    sid = 'dragonlands'
    g0, g1 = section_bars(sid)
    lift(S, sid, cv=0.9)
    offs = [0, 12, 0, 7]
    for g in range(g0 + 1, g1):
        for st in range(16):
            if g == g0 + 1 and st % 4:
                continue                       # the riff knocks during the flight, full from the reveal
            b = 4 * g + st / 4
            r = 38 + (S.chord_at(b).root - 38) % 12
            a('power', b, 0.25, r + offs[st % 4], (0.55 if st % 4 == 0 else 0.4) * fade(g, g0, g1))
    S.duck.append((4 * (g0 + 2), 4 * g1, 0.3))
    eb = L(sid, 288)
    S.fx('riser', 0, 2.0, 0.5, ends_at=eb, lo=300.0, hi=7000.0, f0=150.0, octs=2.5)
    S.drum('boom', eb, 0.95)
    S.drum('crash', eb, 0.9)
    S.ghit(sid, 'erupt', 288, 'ERUPT: riser into boom + crash (the volcano blows) on the downbeat')
    # ---------------- mountaris: mandolin tremolo, then underwater (filter drop) from the dive
    sid = 'mountaris'
    g0, g1 = section_bars(sid)
    db_ = L(sid, 360)
    for g in range(g0, g0 + 5):
        for q in range(32 if g >= g0 + 1 else 16):
            step = 0.125 if g >= g0 + 1 else 0.25
            b = 4 * g + q * step
            for mm in voicing(S.chord_at(b), 64, 77, 2):
                a('mando', b, step, mm, (0.36 if q % 2 == 0 else 0.27) * fade(g, g0, g0 + 5))
    lift(S, sid)
    S.fx('riser', 0, 0.75, 0.28, ends_at=db_, lo=800.0, hi=9000.0, f0=400.0, octs=1.5)
    S.fx('sweep', db_, 2.0, 0.55, hi=7000.0, lo=200.0)
    S.drum('crash_soft', db_, 0.55)
    S.drum('boom', db_, 0.55)
    S.ghit(sid, 'dive', 360, 'DIVE: splash (cymbal + boom + noise sweep down) and filter drop: the last 2 bars are '
                             'underwater (the whole band low-passed, more echo)')
    S.arp('triarp', db_, 4 * g1, step=0.5, lo=74, hi=88, pattern='updown', vel=0.26, pans=(-0.5, 0.5), inst='glock')
    for bar, beat, nm in [(5, 0.75, 'F#6'), (5, 1.25, 'B6'), (5, 3.25, 'D7'), (6, 2.25, 'A6'), (6, 2.75, 'E7')]:
        a('blip', 4 * (g0 + bar) + beat, 0.25, S.pp(nm, g0), 0.18)
    # ---------------- klugetown (D harmonic minor): reed + junkyard percussion; airship
    sid = 'klugetown'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    for g in range(g0 + 2, g1):
        S.arp('stab', 4 * g, 4 * g + 4, step=0.5, lo=57, hi=74, group=[0, 2, 1, 3], vel=0.38, inst='pluck',
              pans=(0.3, -0.3))
    ab = L(sid, 360)
    S.fx('swell', 0, 1.5, 0.45, ends_at=ab)
    a('brass2', ab, 2.0, S.pp('G2', g0), 0.66, inst='brass')
    a('brass2', ab, 2.0, S.pp('D3', g0), 0.55, inst='brass')
    S.drum('boom', ab, 0.45)
    S.drum('crash_soft', ab, 0.55)
    S.fx('sweep', ab, 2.0, 0.28, hi=3000.0, lo=200.0)
    S.ghit(sid, 'airship', 360, 'AIRSHIP: low horn blast (Bb2 + F3) + boom + soft cymbal + propeller whoosh')
    # ---------------- badlands: ominous buzz (D minor) -> transform into radiant F major; Tartarus sting
    sid = 'badlands'
    g0, g1 = section_bars(sid)
    lift(S, sid, crash='crash_soft', cv=0.55)
    for i in range(16):
        a('lead2', 4 * (g0 + 2) + 0.25 * i, 0.25, S.pp('B2' if i % 2 == 0 else 'C3', g0), 0.3, inst='arp25',
          drive=2.0, lp=1800)
    tb = L(sid, 216)
    S.fx('swell', 0, 2.0, 0.7, ends_at=tb)
    S.fx('riser', 0, 1.0, 0.42, ends_at=tb, lo=400.0, hi=9000.0, f0=200.0, octs=3.0)
    S.drum('crash', tb, 1.0)
    S.drum('boom', tb, 0.75)
    S.fx('shimmer', tb, 4.0, 0.65, f0=1760.0)
    S.pad('choir', g0 + 3, g0 + 4, lo=60, hi=76, n=4, vel=0.4, inst='padbright')
    S.arp('arp', tb, tb + 4, step=0.25, lo=62, hi=86, pattern='up', vel=0.34, pans=(-0.6, 0.6))
    S.ghit(sid, 'transform', 216, 'TRANSFORM: crash + boom + shimmer; D minor blooms into radiant F major')
    sb = L(sid, 324)
    for mm in voicing(S.chord_at(sb), 43, 66, 4):
        a('stab', sb, 1.8, mm, 0.46, inst='power')
    S.drum('boom', sb, 0.68)
    S.drum('tom_lo', sb, 0.62)
    S.drum('crash_soft', sb, 0.45)
    S.ghit(sid, 'tartarus', 324, 'dark sting (Eb7b9: low power chord + boom + tom): the gates of Tartarus')
    # ---------------- frontier: twang lead, clip-clop, banjo stabs; whistle
    sid = 'frontier'
    g0, g1 = section_bars(sid)
    lift(S, sid)
    S.comp('stab', g0 + 2, g1, [(1, 0.35, 1.0), (3, 0.35, 0.95)], lo=55, hi=67, n=3, vel=0.5, inst='banjo')
    S.ghit(sid, 'whistle', 288, 'WHISTLE: the whistle lead takes the tune at sunset')
    # ---------------- laspegasus: funk chops + brass stabs + disco hats; neon, fireworks
    sid = 'laspegasus'
    g0, g1 = section_bars(sid)
    r0 = g0 + 2
    lift(S, sid)
    S.comp('keys', r0, g1, [(0.5, 0.25, 0.8), (1.25, 0.25, 0.7), (2.5, 0.25, 0.8), (3.25, 0.25, 0.7)],
           lo=60, hi=72, n=3, vel=0.38, inst='pluck')
    for g in (r0, r0 + 2):
        S.comp('stab', g, g + 1, [(1.5, 0.25, 1.0), (3.75, 0.25, 0.8)], lo=58, hi=70, n=3, vel=0.55, inst='brass')
    S.duck.append((4 * r0, 4 * g1, 0.28))
    nb = L(sid, 216)
    for i in range(8):
        b = nb + 0.5 * i
        for mm in voicing(S.chord_at(b), 55 + 2 * i, 67 + 2 * i, 3):
            a('stab', b, 0.3, mm, 0.5 + 0.04 * i, inst='brass')
    S.ghit(sid, 'neon', 216, 'NEON: rising brass-stab line on every 8th (216,225,...,279): one building per stab')
    fb = L(sid, 360)
    S.drum('crash', fb, 0.9)
    S.drum('boom', fb, 0.7)
    S.fx('shimmer', fb, 3.0, 0.45, f0=2000.0)
    for beat in (0, 2, 3):
        ch = S.chord_at(fb + beat)
        cs = [mm for mm in ch.tones(76, 91) if (mm - ch.root) % 12 in (0, 3, 4, 7)][::-1]
        for i, mm in enumerate(cs[:4]):
            a('glock', fb + beat + 0.125 * i, 0.5, mm, 0.32 - 0.05 * i, pan=(-0.5, 0.5, 0.0)[beat % 3])
    S.ghit(sid, 'fireworks', 360, 'FIREWORKS: crash + boom + glock bursts on beats 1, 3 and 4 (360, 396, 414)')
    # ---------------- westcoast: e-piano + lo-fi drums + rain crackle; moonrise
    sid = 'westcoast'
    g0, g1 = section_bars(sid)
    lift(S, sid, crash='crash_soft', cv=0.5)
    S.comp('keys', g0 + 2, g1, [(0, 1.9, 1.0), (2.5, 1.4, 0.8)], lo=52, hi=67, n=4, vel=0.48, swing=0.6, human=1.0)
    S.fx('crackle', 4 * g0, 14.4, 0.75)
    mb = L(sid, 360)
    S.fx('swell', 0, 1.5, 0.32, ends_at=mb)
    a('bell', mb, 4.0, S.pp('A4', g1 - 1), 0.3)
    a('glock', mb + 0.5, 1.0, S.pp('E5', g1 - 1), 0.16)
    S.fx('shimmer', mb, 3.0, 0.26, f0=1200.0)
    S.ghit(sid, 'moonrise', 360, 'MOONRISE: gentle bell swell (soft reverse cymbal + bell + shimmer)')
    # ---------------- crystal (night peak): choir + glock shimmer; heart, heart2
    sid = 'crystal'
    g0, g1 = section_bars(sid)
    lift(S, sid, cv=0.75)
    S.pad('choir', g0 + 2, g1, lo=62, hi=76, n=3, vel=0.36)
    S.arp('glock', 4 * (g0 + 2), 4 * (g0 + 4), step=0.5, lo=81, hi=93, pattern='updown', vel=0.12, pans=(-0.5, 0.5))
    for f, nm, big in ((288, 'heart', 0.62), (360, 'heart2', 0.72)):
        hb = L(sid, f)
        S.drum('crash', hb, 0.8 if nm == 'heart' else 0.88)
        S.fx('shimmer', hb, 4.0, big, f0=2093.0)
        for b in (0, 2):
            S.drum('boom', hb + b, 0.7)
            S.drum('kick_soft', hb + b + 1 / 3, 0.55)
        ch = S.chord_at(hb)
        cs = [mm for mm in ch.tones(83, 98) if (mm - ch.root) % 12 in (2, 3, 4, 7, 10)][:5]
        for i, mm in enumerate(cs):
            a('glock', hb + 0.08 * i, 2.0, mm, 0.28, pan=-0.4 + 0.2 * i)
        S.ghit(sid, nm, f, f'{nm.upper()}: heartbeat (boom + kick "lub-dub" on beats 1 and 3) + big shimmering chord')
    # ---------------- yakyakistan (B minor): stomp + accordion pulses; avalanche
    sid = 'yakyakistan'
    g0, g1 = section_bars(sid)
    lift(S, sid, cv=0.8)
    S.comp('keys', g0 + 2, g1, [(1, 0.5, 1.0), (3, 0.5, 0.9)], lo=55, hi=67, n=3, vel=0.46, inst='accordion')
    vb = L(sid, 288)
    S.fx('riser', 0, 1.0, 0.35, ends_at=vb, lo=300.0, hi=5000.0, f0=150.0, octs=1.5)
    S.drum('crash', vb, 0.85)
    S.drum('boom', vb, 0.85)
    S.fx('sweep', vb, 3.0, 0.6, hi=5000.0, lo=150.0)
    S.roll('tom_lo', vb, vb + 1, 0.125, 0.85, 0.45)
    S.ghit(sid, 'avalanche', 288, 'AVALANCHE: crash + boom + tom roll + rumbling noise sweep (downbeat of bar 5)')
    # ---------------- twosisters: lullaby -> build -> midnight -> afterglow
    sid = 'twosisters'
    g0, g1 = section_bars(sid)
    lift(S, sid, crash='crash_soft', cv=0.4)
    bb = L(sid, 288)
    S.arp('arp', bb, bb + 4, step=0.25, lo=62, hi=86, pattern='up', vel=0.26, vel_ramp=(0.6, 1.3), pans=(-0.5, 0.5))
    S.roll('snare_soft', bb + 2, bb + 4, 0.125, 0.15, 0.8)
    S.fx('riser', 0, 2.0, 0.45, ends_at=bb + 4, lo=400.0, hi=9000.0, f0=200.0, octs=2.5)
    S.fx('swell', 0, 2.0, 0.65, ends_at=bb + 4)
    S.ghit(sid, 'build', 288, 'build-up: arpeggio + snare roll + riser toward midnight')
    mb = L(sid, 360)
    S.drum('crash', mb, 1.0)
    S.drum('boom', mb, 0.8)
    for i in range(12):
        a('bell', mb + i / 3.0, 1.0, P('D4'), 0.6 - 0.015 * i, tau=1.0)
    S.ghit(sid, 'midnight', 360, 'MIDNIGHT: 12 bell strikes on triplet 8ths (local 360, 366, ..., 426) under the climax '
                                 'of the theme')
    # ---------------- outro: route, complete, key change, final (positions unchanged)
    sid = 'outro'
    g0 = section_bars(sid)[0]
    for i in range(16):
        b = 4 * g0 + 0.5 * i
        ts = S.chord_at(b).tones(81, 93)
        a('blip', b, 0.25, ts[i % len(ts)], 0.2, pan=-0.5 + i / 15)
    S.ghit(sid, 'route', 0, 'route draws itself: blip on every 8th, bars 1-2 (local frames 0, 9, 18, ... 135)')
    cb = 4 * (g0 + 2)
    S.drum('crash', cb, 1.0)
    S.drum('boom', cb, 0.75)
    S.drum('clap', cb, 0.8)
    S.comp('stab', g0 + 2, g0 + 3, [(0, 0.6, 1.0)], lo=57, hi=74, n=4, vel=0.75, inst='brass')
    S.ghit(sid, 'complete', 144, 'MAP COMPLETE stamp: crash + boom + clap + brass chord')
    S.fx('swell', 0, 1.5, 0.55, ends_at=cb + 4)
    S.roll('snare', cb + 3, cb + 4, 0.125, 0.3, 0.75)
    S.drum('crash', cb + 4, 0.88)
    S.ghit(sid, 'keychange', 216, 'KEY CHANGE up a tone (E major): last theme statement; credits one card per bar')
    S.duck.append((cb + 4, cb + 16, 0.22))
    for i, nm in enumerate(['tom_hi', 'tom_hi', 'tom_mid', 'tom_lo']):
        S.drum(nm, cb + 15 + 0.25 * i, 0.55 + 0.06 * i)
    fb = 4 * (g0 + 6)
    S.drum('crash_soft', fb, 0.65)
    S.drum('boom', fb, 0.55)
    a('bass', fb, 2.5, P('E2'), 0.65, inst='upright')
    for i, nm in enumerate(['E5', 'G#5', 'B5', 'F#6']):
        a('mbox', fb + 0.25 * i, 3.0, P(nm), 0.5 - 0.04 * i)
    for nm in ['G#3', 'B3', 'E4', 'F#4']:
        a('keys', fb, 3.0, P(nm), 0.48, tau=1.2)
    for nm in ['E3', 'B3', 'F#4']:
        a('pad', fb, 1.0, P(nm), 0.42, inst='padsoft', r=1.6)
    a('lead', fb, 2.0, P('E5'), 0.66, inst='mellow')
    a('glock', fb, 2.0, P('B6'), 0.24)
    S.ghit(sid, 'final', 432, 'FINAL Eadd9 chord on the last downbeat; decays to digital silence exactly at the '
                              'last sample')
    cuts(S)


def underwater(S):
    """Mountaris after the dive: everything that starts there goes through the low-passed underwater bus."""
    b0 = L('mountaris', 360)
    b1 = SEC_BY_ID['mountaris']['end_beat']
    for e in S.events:
        if b0 - 1e-6 <= e['start_beat'] < b1 - 1e-6 and e.get('fx') != 'sweep':
            e['_o']['bus'] = 'uw'


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


def compose_journey():
    S = Journey()
    build_chords(S)
    melody(S)
    band(S)
    colours(S)
    S.sec = GLOBAL
    return S

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
        end = TOTAL_BARS * 4                     # v2: notes tie across sections; clamp only at the very end
        if e['start_beat'] + e['dur_beats'] > end + 1e-6:
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
    S = compose_journey()
    dedupe_drums(S)
    tidy_lines(S)
    bridges(S)
    nfix = resolve_clashes(S)
    underwater(S)
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
                         key_centres=list(dict.fromkeys(kname(g) for g in range(s['start_bar'],
                                                                             s['start_bar'] + s['bars']))),
                         start_frame=s['start_beat'] * FPB, end_frame=s['end_beat'] * FPB,
                         start_sample=s['start_beat'] * SPB, chords=chords, hits=hs))
    with open(os.path.join(out, 'sections.json'), 'w') as f:
        json.dump(secs, f, indent=1)
    return evs, secs


# =========================================================================== render / mix
SECTION_REV = {'intro': 1.5, 'everfree': 1.4, 'westcoast': 1.2, 'crystal': 1.4, 'twosisters': 1.5,
               'outro': 1.3, 'cloudsdale': 1.3, 'mountaris': 1.2, 'griffonstone': 1.2}
# loudness map: relative level of each section (dB, applied per note so tails carry their own level)
SECTION_GAIN_DB = {}                      # v2: loudness follows the energy arc (see energy())
ARC_DB = 7.0                              # dB per unit of energy, applied per note at its onset
# mix trims (dB): voice -> dB, or (section, voice) -> dB
VOICE_GAIN_DB = {'pling': -1.0, 'bed': -1.0, 'halo': -2.0, ('sweetapple', 'lead'): 3.0, ('ponyville', 'lead'): -2.5,
                 ('manehattan', 'brass'): 3.0, ('manehattan', 'brass2'): 2.0}
# role trims (dB): role -> dB, or (section, role) -> dB
ROLE_GAIN_DB = {'bass': -2.0, 'melody': 1.5, ('dragonlands', 'drums'): -3.5, ('rainbowfalls', 'drums'): -3.0,
                ('heartland', 'drums'): -2.0, ('manehattan', 'drums'): 1.5, ('manehattan', 'accomp'): 1.5, ('manehattan', 'melody'): 1.0,
                ('manehattan', 'harmony'): 1.0,
                ('sweetapple', 'drums'): 1.5, ('sweetapple', 'accomp'): 1.5}


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
        sg = 10 ** ((SECTION_GAIN_DB.get(e['section'], 0.0) + ARC_DB * (energy(e['start_beat']) - 0.7) +
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
END_DECAY = (TOTAL_SAMPLES / SR - 2.4 + 0.45, TOTAL_SAMPLES / SR - 0.015, 0.35, 30.0)   # start, zero, knee, dB/s


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
    # end: the last chord (struck 2.4 s before the end) decays exponentially - a straight line in dB, like a
    # room tail - reaching ~-55 dB of extra attenuation 15 ms before the end; the last 15 ms are digital zero.
    t = np.arange(N) / SR
    f0, f1, knee, depth = END_DECAY
    u = np.maximum(t - f0, 0.0)
    att_db = np.where(u < knee, depth * u * u / (2 * knee), depth * (u - knee / 2))
    fade = 10 ** (-att_db / 20)
    fade[t >= f1] = 0.0
    # start: from silence
    fs = int(0.01 * SR)
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
