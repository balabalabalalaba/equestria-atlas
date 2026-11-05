"""Procedural sound effects (48 kHz, mono, float32 in [-1, 1]) - EQUESTRIA: THE WORLD ATLAS
(built on the ONE GOOD DAY library; atlas landscape / weather / travel effects at the end).

Chiptune-flavoured but polished: band-limited (polyBLEP) pulse/saw waves, modal
(struck-object) synthesis, swept-noise whooshes and bell partials.  Every tonal
effect sits in **D major** (D E F# G A B C#), the key of the soundtrack, and the
melodic ones are timed to the 100 BPM grid (32nd note = 0.075 s).

API
---
    import audio.sfx as sfx            # or load the file directly
    sfx.SFX_NAMES                      # every valid name (preview order)
    x = sfx.make('coin')               # np.float32 mono array (cached; you get a copy)
    x = sfx.make('step', variant=2)    # some sounds have variants (see VARIANTS)
    sfx.describe('coin')               # one-line description incl. duration / hit time
    sfx.resolve('footstep') -> 'step'  # alias resolution (KeyError with suggestions)
    sfx.hit('door')                    # seconds from sound start to its main transient
    sfx.notes('quest')                 # ['D6', 'F#6', 'A6']  (pitched notes; [] if unpitched)
    sfx.pitch_classes('pop', 2)        # {6}  (F#; used by the mixer's key-fit)
    sfx.stats('bell')                  # dur / peak / loudness / RMS / centroid numbers
    sfx.MIX['sizzle']                  # mix defaults: trim dB, reverb send dB, pan, alt-pan

Every effect returned by make() is loudness-normalised: the loudest 100 ms
(K-weighted) sits at NOMINAL_LK (-20 LUFS-ish), sample peak capped at -1 dBFS,
starts and ends exactly at zero (raised-cosine fades) and has no DC.

Scene authors normally never call this module: they write
    SFX = [(local_frame, 'name', gain), ...]      # gain ~0.3 .. 1.0
in their scene and audio/mix.py does the rest.

CLI:  .venv/bin/python audio/sfx.py            -> audio/out/sfx_preview.wav + table
      .venv/bin/python audio/sfx.py coin bell  -> preview just those (use --out to keep the full one)
      .venv/bin/python audio/sfx.py --list     -> names + descriptions
"""
from __future__ import annotations

import difflib
import os
import zlib
from functools import lru_cache

import numpy as np
from scipy import signal

SR = 48000
PEAK = 10 ** (-1 / 20)          # -1 dBFS sample-peak cap
NOMINAL_LK = -20.0              # loudest-100ms K-weighted level of every effect
KEY_PCS = (2, 4, 6, 7, 9, 11, 1)  # D major pitch classes (C=0)
PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
       'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
_HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 2 * np.pi


def hz(note):
    """'F#6' -> frequency in Hz (equal temperament, A4 = 440)."""
    name, octv = note[:-1], int(note[-1])
    midi = 12 * (octv + 1) + _PC[name]
    return 440.0 * 2 ** ((midi - 69) / 12)


# ============================================================== primitives
def _n(d):
    return max(1, int(round(d * SR)))


def _t(n):
    return np.arange(n) / SR


def _ramp(n):
    """Raised-cosine ramp 0 -> ~1 over n samples (first sample exactly 0)."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)


def _fade(x, fin=0.0003, fout=0.002):
    y = np.array(x, dtype=np.float64)
    a, b = min(len(y), _n(fin)), min(len(y), _n(fout))
    if a:
        y[:a] *= _ramp(a)
    if b:
        y[len(y) - b:] *= _ramp(b)[::-1]
    if len(y):
        y[-1] = 0.0
    return y


def env(n, a=0.003, tau=None, hold=0.0, rel=0.01, sus=0.0):
    """Attack (raised cosine, `a` s) -> hold -> exp decay (time constant `tau`) toward `sus`
    -> release fade to exactly 0 over the last `rel` s.  tau=None: flat sustain."""
    t = _t(n)
    if tau:
        td = np.clip(t - a - hold, 0, None)
        e = sus + (1 - sus) * np.exp(-td / tau)
    else:
        e = np.ones(n)
    na = min(n, _n(a))
    e[:na] *= _ramp(na)
    nr = min(n, _n(rel))
    if nr:
        e[n - nr:] *= _ramp(nr)[::-1]
    return e


def pw(n, pts):
    """Smooth piecewise envelope through (time_s, value) points, smoothstep between points."""
    t = _t(n)
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    i = np.clip(np.searchsorted(xs, t, side='right') - 1, 0, len(xs) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    u = np.clip((t - x0) / np.maximum(x1 - x0, 1e-9), 0, 1)
    u = u * u * (3 - 2 * u)
    return y0 + (y1 - y0) * u


def put(dst, src, t, gain=1.0):
    """Add `src` into `dst` starting at time t (s), with tiny safety fades (never clicks)."""
    i = int(round(t * SR))
    if i >= len(dst) or gain == 0:
        return dst
    s = _fade(src)
    j = min(len(dst), i + len(s))
    dst[i:j] += gain * s[: j - i]
    return dst


def pk(x):
    m = np.max(np.abs(x)) if len(x) else 0
    return x / m if m > 0 else x


def delay(x, d, g):
    """Single feedforward echo (d seconds, gain g) - same length as x."""
    k = _n(d)
    y = np.array(x, float)
    if k < len(x):
        y[k:] += g * x[:-k]
    return y


# ---------------------------------------------------------------- oscillators
def _freq(f, n):
    return np.array(np.broadcast_to(np.asarray(f, dtype=np.float64), (n,)))


def _phase(f, n):
    f = _freq(f, n)
    ph = np.empty(n)
    ph[0] = 0.0
    np.cumsum(f[:-1] / SR, out=ph[1:])
    return ph


def sine(f, n):
    return np.sin(TAU * _phase(f, n))


def tri(f, n):
    p = _phase(f, n)
    return 1 - 4 * np.abs(((p + 0.25) % 1.0) - 0.5)


def _blep(t, dt):
    y = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    y[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt[m]
    y[m] = x * x + x + x + 1.0
    return y


def pulse(f, n, duty=0.5):
    """Band-limited (polyBLEP) pulse wave, DC removed. duty may be an array."""
    f = _freq(f, n)
    t = _phase(f, n) % 1.0
    dt = np.clip(np.abs(f) / SR, 1e-9, 0.5)
    d = np.asarray(duty, float)
    y = np.where(t < d, 1.0, -1.0)
    y += _blep(t, dt)
    y -= _blep((t + 1.0 - d) % 1.0, dt)
    return y - (2 * d - 1)


def saw(f, n):
    f = _freq(f, n)
    t = _phase(f, n) % 1.0
    dt = np.clip(np.abs(f) / SR, 1e-9, 0.5)
    return 2 * t - 1 - _blep(t, dt)


def white(n, r):
    return r.standard_normal(n)


def crunch(n, rate, r):
    """Chip-style sample&hold +-1 noise at `rate` Hz."""
    hold = SR / rate
    k = int(np.ceil(n / hold)) + 1
    v = r.choice([-1.0, 1.0], k)
    return v[(np.arange(n) / hold).astype(int)]


# ---------------------------------------------------------------- filters
@lru_cache(maxsize=None)
def _sos(kind, f1, f2, order):
    if kind == 'bandpass':
        return signal.butter(order, [f1, f2], 'bandpass', fs=SR, output='sos')
    return signal.butter(order, f1, kind, fs=SR, output='sos')


def lp(x, fc, order=2):
    return signal.sosfilt(_sos('lowpass', float(min(fc, SR * 0.45)), 0.0, order), x)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos('highpass', float(fc), 0.0, order), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos('bandpass', float(lo), float(min(hi, SR * 0.45)), order), x)


def reson(x, f, q):
    b, a = signal.iirpeak(f, q, fs=SR)
    return signal.lfilter(b, a, x)


def modes(n, spec, a=0.0002, rel=0.006):
    """Struck-object modal synthesis: sum of decaying sines [(freq, amp, tau), ...]."""
    t = _t(n)
    y = np.zeros(n)
    for f, amp, tau in spec:
        if f < SR * 0.47:
            y += amp * np.sin(TAU * f * t) * np.exp(-t / tau)
    return y * env(n, a, None, rel=min(rel, n / SR / 4))


def bell(f0, n, partials, beat=0.0):
    """Bell partials [(ratio, amp, tau)], each a slightly detuned doublet (warble)."""
    t = _t(n)
    y = np.zeros(n)
    for r_, amp, tau in partials:
        f = f0 * r_
        if f >= SR * 0.47:
            continue
        e = amp * np.exp(-t / tau)
        if beat:
            y += e * (0.62 * np.sin(TAU * (f - beat / 2) * t) + 0.38 * np.sin(TAU * (f + beat / 2) * t))
        else:
            y += e * np.sin(TAU * f * t)
    return y


def swept(n, fc, r, width=0.7, lo=120.0, hi=14000.0, bands=18, src=None):
    """Noise through a bank of band-passes whose weights follow a moving centre `fc` (Hz, array)."""
    x = white(n, r) if src is None else src
    lfc = np.log2(np.clip(_freq(fc, n), lo, hi))
    edges = np.geomspace(lo, hi, bands + 1)
    y = np.zeros(n)
    for i in range(bands):
        c = np.log2(np.sqrt(edges[i] * edges[i + 1]))
        w = np.exp(-0.5 * ((lfc - c) / width) ** 2)
        if w.max() < 1e-3:
            continue
        y += bp(x, edges[i], edges[i + 1]) * w
    return y


def note(freq, dur, wave='pulse', duty=0.25, a=0.002, tau=0.15, sus=0.0, rel=0.02, hold=0.0,
         vib=0.0, vib_rate=6.0, vib_delay=0.05, glide=0.0, glide_t=0.012, lpf=7000.0, sub=0.0):
    """One chip note. glide: start `glide` semitones off and settle; sub: sine layer level."""
    n = _n(dur)
    t = _t(n)
    f = freq * np.ones(n)
    if glide:
        f *= 2 ** (glide / 12 * np.exp(-t / glide_t))
    if vib:
        f *= 1 + vib * np.sin(TAU * vib_rate * t) * np.clip((t - vib_delay) / 0.08, 0, 1)
    if wave == 'pulse':
        y = pulse(f, n, duty)
    elif wave == 'saw':
        y = saw(f, n)
    elif wave == 'tri':
        y = tri(f, n)
    else:
        y = sine(f, n)
    if lpf and wave in ('pulse', 'saw'):
        y = lp(y, lpf)
    if sub:
        y = y + sub * sine(f, n)
    return y * env(n, a, tau, hold, rel, sus)


def twinkle(f, dur=0.35, tau=0.1):
    """Small glassy bell ping used for sparkles."""
    n = _n(dur)
    t = _t(n)
    y = np.sin(TAU * f * t) + 0.25 * np.sin(TAU * f * 2.001 * t) * np.exp(-t / (tau * 0.5))
    return y * env(n, 0.0008, tau, rel=0.03)


def events(n, rate, r):
    """Poisson event mask; rate in events/s (scalar or array)."""
    return r.random(n) < np.asarray(rate, float) / SR


# ============================================================== registry
_GEN = {}
_DESC = {}
_HIT = {}
_HPF = {}
VARIANTS = {}
TONAL = {}       # name -> ('notes'|'glide', [intended note names])  (used by the checker/docs)


def _sound(name, desc, hit=0.0, variants=1, hpf=25.0, tonal=None):
    def deco(fn):
        _GEN[name] = fn
        _DESC[name] = desc
        _HIT[name] = hit
        _HPF[name] = hpf
        VARIANTS[name] = variants
        if tonal:
            TONAL[name] = tonal
        return fn
    return deco


# ============================================================== the sounds
# ---------------------------------------------------------------- home / morning
@_sound('alarm', 'Cute phone alarm: two 1-beat groups of 32nd-note marimba-chip arpeggios '
        '(D6 F#6 A6 F#6 D7 / D6 F#6 A6 B6 A6) over a rattling phone vibration; exactly 2 beats.',
        tonal=('notes', ['D6', 'F#6', 'A6', 'B6', 'D7']))
def _alarm(v, r):
    n = _n(1.2)
    y = np.zeros(n)
    s = 0.075
    seq = [(0, 'D6', 1), (1, 'F#6', 1), (2, 'A6', 1), (3, 'F#6', 1), (4, 'D7', 2),
           (8, 'D6', 1), (9, 'F#6', 1), (10, 'A6', 1), (11, 'B6', 1), (12, 'A6', 2)]
    for slot, nm, ln in seq:
        nt = note(hz(nm), ln * s + 0.1, 'pulse', 0.25, a=0.002, tau=0.065 if ln == 1 else 0.11,
                  rel=0.03, lpf=6000, sub=0.7)
        put(y, nt, slot * s, 0.85 if ln == 1 else 1.0)
    # phone vibration buzz on the nightstand (D3, rattly)
    t = _t(n)
    bz = lp(pulse(hz('D3'), n, 0.5), 520) * (0.55 + 0.45 * np.sin(TAU * 31 * t))
    gate = pw(n, [(0, 0), (0.02, 1), (0.47, 1), (0.5, 0), (0.6, 0), (0.62, 1), (1.07, 1), (1.1, 0)])
    return y + 0.16 * pk(bz) * gate


@_sound('tap', 'Fingertip tap on a phone screen / button: soft low "tk".', variants=3)
def _tap(v, r):
    k = (1.0, 1.07, 0.94)[v]
    n = _n(0.07)
    y = modes(n, [(210 * k, 0.9, 0.010), (640 * k, 0.35, 0.006), (1500 * k, 0.2, 0.003)])
    y += 0.45 * pk(bp(white(n, r), 1800, 6000)) * env(n, 0.0003, 0.002, rel=0.002)
    return y


@_sound('step', 'Single soft footstep (heel thump + sole scuff). Variants alternate automatically.',
        variants=4, hpf=30)
def _step(v, r):
    k = (1.0, 0.93, 1.06, 0.97)[v]
    n = _n(0.17)
    t = _t(n)
    thump = sine(70 * k + 85 * k * np.exp(-t / 0.018), n) * env(n, 0.002, 0.035, rel=0.02)
    heel = pk(lp(white(n, r), 2200)) * env(n, 0.001, 0.016, rel=0.02)
    scuff = pk(bp(white(n, r), 1200, 4500)) * env(n, 0.02, 0.03, rel=0.03)
    return 0.9 * thump + 0.5 * heel + 0.16 * scuff


@_sound('curtain', 'Curtains drawn open: fabric swish with curtain rings sliding along the rod.', hit=0.12)
def _curtain(v, r):
    n = _n(0.85)
    t = _t(n)
    fc = 700 * 3.2 ** np.sin(np.pi * np.clip(t / 0.75, 0, 1))
    y = 0.75 * pk(swept(n, fc, r, width=0.8)) * pw(n, [(0, 0), (0.13, 1), (0.4, 0.55), (0.85, 0)])
    times = [0.04, 0.09, 0.13, 0.17, 0.21, 0.26, 0.32, 0.41, 0.52]
    for i, tt in enumerate(times):
        m = _n(0.05)
        kk = 1 + 0.03 * ((i * 7) % 5 - 2)
        ring = modes(m, [(3150 * kk, 1, 0.012), (4720 * kk, 0.6, 0.008), (6900 * kk, 0.35, 0.005)])
        ring += 0.4 * pk(hp(white(m, r), 4000)) * env(m, 0.0002, 0.001, rel=0.002)
        put(y, pk(ring), tt, 0.3 * (1 - 0.06 * i))
    return y


@_sound('sizzle', 'Egg/stir-fry hitting a hot pan: bright "tsss" onset settling into crackly frying.',
        hpf=60)
def _sizzle(v, r):
    n = _n(1.5)
    t = _t(n)
    hiss = pk(bp(white(n, r), 2500, 11000))
    flutter = lp(white(n, r), 8)
    flutter = 1 + 0.35 * flutter / (np.std(flutter) + 1e-12)
    hiss *= np.clip(flutter, 0.3, 2.0)
    rate = 70 * np.exp(-t / 0.6) + 28
    ev = events(n, rate, r)
    imp = np.where(ev, r.choice([-1.0, 1.0], n) * np.minimum(0.3 + r.pareto(3.0, n), 4.0), 0.0)
    cr = reson(imp, 2600, 4) + 0.7 * reson(imp, 4300, 5) + 0.5 * reson(imp, 6800, 6)
    sp_ev = events(n, 4.0, r)
    spit = reson(np.where(sp_ev, r.uniform(0.5, 1.0, n), 0.0), 1500, 10)
    body = pk(bp(white(n, r), 250, 700)) * (0.6 + 0.4 * np.sin(TAU * 5 * t))
    e = pw(n, [(0, 0), (0.012, 1), (0.08, 0.95), (0.45, 0.62), (1.15, 0.55), (1.5, 0)])
    return (0.45 * hiss + 0.5 * pk(cr) + 0.25 * pk(spit) + 0.07 * body) * e


@_sound('crack', 'Egg cracked on the pan rim: crisp shell crack + tiny crumble + soft "plip".',
        variants=2)
def _crack(v, r):
    k = (1.0, 1.08)[v]
    n = _n(0.32)
    y = np.zeros(n)
    for tt, a in [(0.0, 1.0), (0.009, 0.55), (0.021, 0.4), (0.034, 0.18)]:
        m = _n(0.03)
        b = pk(hp(white(m, r), 1500)) * env(m, 0.0002, 0.0025, rel=0.002)
        b += modes(m, [(1950 * k, 0.5, 0.006), (3350 * k, 0.35, 0.004), (5200 * k, 0.2, 0.002)])
        put(y, b, tt, a)
    m = _n(0.1)
    tm = _t(m)
    ev = events(m, 160 * np.exp(-tm / 0.03), r)
    crum = reson(np.where(ev, r.uniform(-1, 1, m), 0.0), 3100, 3)
    put(y, pk(crum) * env(m, 0.001, 0.03, rel=0.02), 0.035, 0.22)
    m = _n(0.05)
    tm = _t(m)
    plip = sine(650 * (1 + 0.7 * tm / 0.05), m) * env(m, 0.001, 0.012, rel=0.01)
    put(y, plip, 0.16, 0.25)
    return y


@_sound('pop', 'Cartoon pop (toast popping, items hopping, bubbles). Variants cycle in-key pitches '
        'A5 D6 F#6 B5.', variants=4, tonal=('glide', ['A5', 'D6', 'F#6', 'B5']))
def _pop(v, r):
    fend = hz(('A5', 'D6', 'F#6', 'B5')[v])
    n = _n(0.16)
    t = _t(n)
    f = fend * (0.35 + 0.65 * (1 - np.exp(-t / 0.012)))
    y = sine(f, n) * env(n, 0.001, 0.045, rel=0.02)
    y += 0.25 * tri(f, n) * env(n, 0.001, 0.02, rel=0.02)
    y += 0.35 * pk(lp(white(n, r), 5000)) * env(n, 0.0002, 0.0015, rel=0.002)
    return y


@_sound('chop', 'Knife chop through veg onto a wooden board (crunch + board knock).', variants=4, hpf=40)
def _chop(v, r):
    k = (1.0, 1.05, 0.95, 1.02)[v]
    n = _n(0.15)
    t = _t(n)
    cr = pk(bp(white(n, r), 1800, 7000)) * env(n, 0.0003, 0.004, rel=0.003)
    board = modes(n, [(360 * k, 0.8, 0.02), (790 * k, 0.5, 0.012), (1580 * k, 0.3, 0.006),
                      (2900 * k, 0.15, 0.004)])
    thump = sine(150 * k * (1 + 0.6 * np.exp(-t / 0.008)), n) * env(n, 0.001, 0.02, rel=0.02)
    return 0.6 * cr + board + 0.5 * thump


@_sound('click', 'Light switch / mouse / latch click: crisp double "tik-k". 2 variants (on/off).',
        variants=2, hpf=60)
def _click(v, r):
    n = _n(0.07)
    y = np.zeros(n)
    f1, f2 = ((3200, 5600), (2700, 4900))[v]

    def one(fa, fb):
        m = _n(0.03)
        c = 0.5 * pk(hp(white(m, r), 2000)) * env(m, 0.0001, 0.0008, rel=0.001)
        return c + modes(m, [(fa, 0.6, 0.004), (fb, 0.35, 0.003), (900, 0.2, 0.005)])
    put(y, one(f1, f2), 0.0, 1.0)
    put(y, one(f1 * 1.1, f2 * 0.95), 0.024, 0.35)
    return y


@_sound('clink', 'Ice cube clinking into a glass (two glassy hits). Variants cycle D7 A6 E7 F#7.',
        variants=4, tonal=('notes', ['D7', 'A6', 'E7', 'F#7']), hpf=200)
def _clink(v, r):
    f0 = hz(('D7', 'A6', 'E7', 'F#7')[v])
    n = _n(0.5)
    y = np.zeros(n)

    def hit(d):
        m = _n(0.42)
        g = modes(m, [(f0, 1.0, 0.16), (f0 * 2.32, 0.45 * d, 0.07), (f0 * 3.86, 0.25 * d, 0.035),
                      (f0 * 5.6, 0.1 * d, 0.02)])
        return g + 0.3 * pk(hp(white(m, r), 4000)) * env(m, 0.0002, 0.0015, rel=0.002)
    put(y, hit(1.0), 0.0, 1.0)
    put(y, hit(0.7), 0.072, 0.5)
    return y


@_sound('pour', 'Liquid poured into a glass: stream whose resonance rises as it fills + bubbles.',
        hpf=80)
def _pour(v, r):
    n = _n(1.0)
    t = _t(n)
    stream = pk(swept(n, 650 * 2 ** (1.4 * t), r, width=0.55))
    bub = np.zeros(n)
    for tt in np.sort(r.uniform(0.03, 0.9, 42)):
        f0 = r.uniform(500, 1500) * (1 + 0.6 * tt)
        m = _n(0.03)
        tm = _t(m)
        b = sine(f0 * (1 + 0.25 * tm / 0.03), m) * env(m, 0.001, r.uniform(0.006, 0.014), rel=0.004)
        put(bub, b, tt, r.uniform(0.3, 1.0))
    glug = pk(lp(white(n, r), 400)) * (0.5 + 0.5 * np.sin(TAU * 7 * t))
    e = pw(n, [(0, 0), (0.04, 1), (0.8, 1), (1.0, 0)])
    return (stream + 0.55 * pk(bub) + 0.15 * glug) * e


@_sound('gulp', 'A happy gulp/sip: wet click + low "glk" bloop gliding down to D3.', hit=0.03,
        tonal=('glide', ['D3', 'A3']), hpf=40)
def _gulp(v, r):
    n = _n(0.36)
    y = np.zeros(n)
    m = _n(0.02)
    put(y, pk(bp(white(m, r), 900, 3500)) * env(m, 0.0003, 0.003, rel=0.004), 0.0, 0.35)
    m = _n(0.17)
    tm = _t(m)
    f = hz('D3') * 2.25 ** np.exp(-tm / 0.022)
    bl = lp(0.6 * tri(f, m) + sine(f, m), 1400) * env(m, 0.004, 0.08, rel=0.03)
    put(y, bl, 0.025, 1.0)
    f = hz('A3') * 1.5 ** np.exp(-tm / 0.018)
    bl2 = lp(0.6 * tri(f, m) + sine(f, m), 1400) * env(m, 0.004, 0.06, rel=0.03)
    put(y, bl2, 0.17, 0.45)
    return y


@_sound('fridge', 'Fridge door opening: seal "thup", jars jingle, compressor hum (G2) swells.',
        hpf=35)
def _fridge(v, r):
    n = _n(0.95)
    t = _t(n)
    y = 0.7 * pk(lp(white(n, r), 900)) * env(n, 0.004, 0.04, rel=0.02)
    y += 0.5 * sine(60 + 40 * np.exp(-t / 0.02), n) * env(n, 0.003, 0.05, rel=0.02)
    for tt, nm, a in [(0.11, 'A7', 0.12), (0.18, 'E7', 0.08)]:
        m = _n(0.2)
        put(y, modes(m, [(hz(nm), 1, 0.05), (hz(nm) * 2.4, 0.4, 0.02)]), tt, a)
    f = hz('G2')
    hum = sine(f, n) + 0.5 * sine(2 * f, n) + 0.35 * sine(3 * f, n) + 0.15 * sine(4 * f, n)
    hum = lp(hum, 600) * pw(n, [(0, 0), (0.15, 0), (0.35, 1), (0.65, 1), (0.95, 0)])
    return y + 0.2 * pk(hum)


@_sound('door', 'Wooden door: latch click, air swing, closing thud + latch (main thud at 0.38 s).',
        hit=0.38, hpf=30)
def _door(v, r):
    n = _n(0.8)
    y = np.zeros(n)

    def latch():
        m = _n(0.05)
        l_ = modes(m, [(2150, 0.6, 0.012), (3400, 0.45, 0.008), (5300, 0.3, 0.005)])
        return l_ + 0.5 * pk(hp(white(m, r), 3000)) * env(m, 0.0001, 0.0008, rel=0.002)
    put(y, latch(), 0.0, 0.45)
    m = _n(0.42)
    air = pk(bp(white(m, r), 250, 1400)) * pw(m, [(0, 0), (0.05, 0), (0.22, 1), (0.37, 0.4), (0.42, 0)])
    put(y, air, 0.0, 0.12)
    m = _n(0.4)
    tm = _t(m)
    th = sine(55 + 35 * np.exp(-tm / 0.015), m) * env(m, 0.001, 0.08, rel=0.05)
    th += modes(m, [(165, 0.6, 0.06), (318, 0.45, 0.035), (530, 0.3, 0.02), (980, 0.15, 0.01)])
    th += 0.4 * pk(lp(white(m, r), 1800)) * env(m, 0.0005, 0.008, rel=0.01)
    put(y, th, 0.38, 1.0)
    put(y, latch(), 0.392, 0.25)
    return y


@_sound('whoosh', 'Fast air whoosh / dash / swipe (band sweeps up to ~2.6 kHz, peaks at 0.2 s).',
        hit=0.2, hpf=60)
def _whoosh(v, r):
    n = _n(0.5)
    t = _t(n)
    pkt = 0.2
    fc = np.where(t < pkt, 350 * (2600 / 350) ** ((np.clip(t, 0, pkt) / pkt) ** 1.2),
                  2600 * (900 / 2600) ** ((np.clip(t - pkt, 0, None) / (0.5 - pkt)) ** 0.8))
    src = white(n, r) + 0.35 * crunch(n, 9000, r)
    y = pk(swept(n, fc, r, width=0.6, src=src))
    return y * pw(n, [(0, 0), (0.08, 0.25), (0.2, 1), (0.3, 0.45), (0.5, 0)])


@_sound('sparkle', 'Magic twinkle: quick run of glassy pings A6 D7 F#7 A7 E7 A7 with glitter.',
        tonal=('notes', ['A6', 'D7', 'F#7', 'A7', 'E7']), hpf=300)
def _sparkle(v, r):
    n = _n(0.85)
    t = _t(n)
    y = np.zeros(n)
    for i, (nm, tt) in enumerate([('A6', 0.0), ('D7', 0.055), ('F#7', 0.1), ('A7', 0.15),
                                  ('E7', 0.215), ('A7', 0.27)]):
        put(y, twinkle(hz(nm), 0.5, 0.12 + 0.02 * i), tt, 1 - 0.08 * i)
    ev = events(n, 90 * pw(n, [(0, 0), (0.05, 1), (0.4, 0.5), (0.7, 0)]), r)
    gl = reson(np.where(ev, r.uniform(-1, 1, n), 0.0), 9000, 12) + reson(
        np.where(ev, r.uniform(-1, 1, n), 0.0), 11500, 14)
    return y + 0.18 * pk(gl) * pw(n, [(0, 0), (0.03, 1), (0.6, 0.3), (0.85, 0)])


@_sound('powerup', 'Classic power-up: rising chip arpeggio D5 F#5 A5 D6 F#6 A6 -> D7 with vibrato.',
        tonal=('notes', ['D5', 'F#5', 'A5', 'D6', 'F#6', 'A6', 'D7']))
def _powerup(v, r):
    n = _n(0.85)
    y = np.zeros(n)
    notes = ['D5', 'F#5', 'A5', 'D6', 'F#6', 'A6', 'D7']
    step = 0.045
    for i, nm in enumerate(notes):
        last = i == len(notes) - 1
        d = step + (0.5 if last else 0.035)
        nt = note(hz(nm), d, 'pulse', 0.125 if i % 2 == 0 else 0.25, a=0.0015,
                  tau=0.28 if last else 0.06, rel=0.06 if last else 0.012, glide=-0.6,
                  glide_t=0.008, vib=0.006 if last else 0.0, vib_rate=7, lpf=7500, sub=0.5)
        put(y, nt, i * step, 0.75 + 0.04 * i)
    for tt, nm in [(0.33, 'A7'), (0.4, 'F#7'), (0.47, 'D7')]:
        put(y, twinkle(hz(nm), 0.3, 0.07), tt, 0.18)
    return y


@_sound('clank', 'Metal clank (barbell plates / pan on hob): inharmonic plate ring + second plate.',
        variants=3, hpf=40)
def _clank(v, r):
    f1, f2 = ((hz('A4'), hz('B4')), (hz('B4'), hz('D5')), (hz('F#4'), hz('A4')))[v]
    n = _n(0.65)
    y = np.zeros(n)

    def hit(f, d):
        m = _n(0.6)
        tm = _t(m)
        h = modes(m, [(f, 1.0, 0.32 * d), (f * 2.76, 0.7, 0.2 * d), (f * 5.40, 0.45, 0.11 * d),
                      (f * 8.93, 0.3, 0.06 * d), (f * 13.34, 0.18, 0.035 * d)])
        h += 0.5 * pk(bp(white(m, r), 1500, 9000)) * env(m, 0.0002, 0.004, rel=0.004)
        h += 0.4 * sine(80 + 40 * np.exp(-tm / 0.01), m) * env(m, 0.001, 0.03, rel=0.02)
        return h
    put(y, hit(f1, 1.0), 0.0, 1.0)
    put(y, hit(f2, 0.7), 0.055, 0.45)
    return y


@_sound('thud', 'Heavy dull thud (weights set down, bag dropped, body on bed). Dark.', variants=2,
        hpf=22)
def _thud(v, r):
    k = (1.0, 0.9)[v]
    n = _n(0.42)
    t = _t(n)
    body = sine(42 * k + 48 * k * np.exp(-t / 0.03), n) * env(n, 0.002, 0.11, rel=0.05)
    knock = modes(n, [(170 * k, 0.5, 0.03), (260 * k, 0.3, 0.02)])
    dust = pk(lp(white(n, r), 450)) * env(n, 0.001, 0.03, rel=0.02)
    return body + 0.4 * knock + 0.45 * dust


@_sound('beep', 'Supermarket scanner beep: clean electronic A6 (dominant of the key) - cue on beats.',
        tonal=('notes', ['A6']), hpf=200)
def _beep(v, r):
    n = _n(0.13)
    f = hz('A6')
    y = 0.8 * sine(f, n) + 0.2 * lp(pulse(f, n, 0.5), 6000)
    return y * env(n, 0.003, None, rel=0.02)


@_sound('ding', 'Microwave / counter-bell ding on D6 with a ~1.2 s shimmering decay.',
        tonal=('notes', ['D6']), hpf=150)
def _ding(v, r):
    f = hz('D6')
    n = _n(1.4)
    y = bell(f, n, [(1, 1, 0.5), (2.0, 0.38, 0.3), (2.76, 0.1, 0.09), (3.0, 0.18, 0.16),
                    (4.07, 0.07, 0.06), (5.4, 0.04, 0.04)], beat=1.3)
    y += 0.15 * pk(hp(white(n, r), 3000)) * env(n, 0.0002, 0.0015, rel=0.002)
    return y * env(n, 0.0008, None, rel=0.25)


@_sound('typing', 'Burst of ~11 keyboard keystrokes incl. a spacebar thock (about 1 s).', variants=2,
        hpf=60)
def _typing(v, r):
    n = _n(1.0)
    y = np.zeros(n)
    gaps = [0.085, 0.072, 0.11, 0.064, 0.092, 0.13, 0.078, 0.083, 0.104, 0.071]
    if v:
        gaps = list(r.permutation(gaps))
    tt = 0.0
    for i, g in enumerate([0.0] + gaps):
        tt += g
        space = i == 6
        kk = 0.6 if space else 1 + 0.08 * (r.random() - 0.5)
        m = _n(0.06)
        c = 0.5 * pk(bp(white(m, r), 2500, 8000)) * env(m, 0.0002, 0.0012, rel=0.002)
        c += modes(m, [(1050 * kk, 0.5, 0.007), (2250 * kk, 0.3, 0.005),
                       ((180 if space else 320) * (1 + 0.05 * r.random()), 0.35, 0.01)])
        put(y, c, tt, 1.0 if space else 0.65 + 0.3 * r.random())
    return y


# ---------------------------------------------------------------- lab / work
@_sound('laser', 'Laser power-on: rising zap A3 -> A6 that "pings" and settles into a soft D/A hum.',
        hit=0.17, tonal=('glide', ['A6', 'D4', 'A4', 'A3']), hpf=60)
def _laser(v, r):
    n = _n(1.25)
    t = _t(n)
    y = np.zeros(n)
    zt = 0.17
    m = _n(0.32)
    tm = _t(m)
    f = hz('A3') * 8 ** (np.clip(tm / zt, 0, 1) ** 0.6)
    zap = 0.5 * pulse(f, m, 0.25) + sine(f * (1 + 0.02 * np.sin(TAU * 45 * tm)), m)
    zap = lp(zap, 9000) * pw(m, [(0, 0), (0.01, 0.7), (zt, 1), (0.32, 0)])
    put(y, zap, 0.0, 0.8)
    m = _n(0.6)
    put(y, sine(hz('A6'), m) * env(m, 0.001, 0.14, rel=0.05), zt, 0.5)
    trem = 1 + 0.08 * np.sin(TAU * 4.5 * t)
    hum = (0.5 * sine(hz('D4'), n) + 0.35 * sine(hz('A4'), n) + 0.4 * sine(hz('A3') + 0.4, n)
           + 0.12 * sine(hz('A5') * 1.0005, n)) * trem
    hiss = 0.05 * pk(bp(white(n, r), 6000, 11000))
    he = pw(n, [(0, 0), (0.12, 0), (0.3, 1), (0.8, 0.75), (1.25, 0)])
    return y + 0.45 * (pk(hum) + hiss) * he


@_sound('hum', 'Electronics powering on: relay tick + "vwoom" spin-up D2 -> D3 and a steady hum.',
        tonal=('glide', ['D3']), hpf=30)
def _hum(v, r):
    n = _n(1.1)
    t = _t(n)
    y = np.zeros(n)
    m = _n(0.03)
    put(y, modes(m, [(2800, 0.6, 0.004), (4100, 0.3, 0.003)]), 0.0, 0.25)
    u = np.clip(t / 0.4, 0, 1)
    f = hz('D2') * 2 ** (1 - (1 - u) ** 2)
    buzz = lp(0.6 * saw(f, n) + 0.3 * pulse(f, n, 0.5), 1100) + 0.6 * sine(f, n)
    whine = 0.04 * sine(hz('A5') * 2 ** (1 - (1 - u) ** 2), n)
    e = pw(n, [(0, 0), (0.03, 0.15), (0.3, 1), (0.7, 0.85), (1.1, 0)])
    return y + (pk(buzz) + whine) * e


@_sound('blip', 'UI blip: tiny two-step chip note D6 -> A6 (counters, menus, results screen).',
        tonal=('notes', ['D6', 'A6']), hpf=150)
def _blip(v, r):
    n = _n(0.09)
    t = _t(n)
    f = np.where(t < 0.022, hz('D6'), hz('A6'))
    y = 0.7 * lp(pulse(f, n, 0.25), 8000) + 0.4 * sine(f, n)
    return y * env(n, 0.001, 0.05, rel=0.015)


@_sound('idea', 'Eureka / light-bulb: "bwing" glide A5 -> D7 into a bright bell + D-major pad + '
        'glitter (bell lands at 0.12 s).', hit=0.12, tonal=('notes', ['D7', 'D6', 'F#6', 'A6']),
        hpf=150)
def _idea(v, r):
    n = _n(1.15)
    t = _t(n)
    y = np.zeros(n)
    m = _n(0.14)
    tm = _t(m)
    f = hz('A5') * 2 ** (1.41667 * np.clip(tm / 0.12, 0, 1) ** 1.5)
    gl = (sine(f, m) + 0.4 * tri(f, m)) * pw(m, [(0, 0), (0.1, 0.8), (0.12, 1), (0.14, 0)])
    put(y, gl, 0.0, 0.45)
    m = _n(1.0)
    b = bell(hz('D7'), m, [(1, 1, 0.35), (2, 0.25, 0.15), (3, 0.08, 0.08)], beat=2.0)
    put(y, b * env(m, 0.001, None, rel=0.2), 0.12, 1.0)
    tm = _t(m)
    pad = sum(sine(hz(nm), m) for nm in ('D6', 'F#6', 'A6')) * (0.75 + 0.25 * np.sin(TAU * 7 * tm))
    put(y, pad * env(m, 0.02, 0.45, rel=0.2), 0.12, 0.22)
    for tt, nm in [(0.2, 'A7'), (0.28, 'F#7'), (0.36, 'D7'), (0.5, 'A7')]:
        put(y, twinkle(hz(nm), 0.3, 0.06), tt, 0.15)
    return y


@_sound('marker', 'Whiteboard marker squeaks: three short strokes (E7 / D7 / F#7 squeal + felt hiss).',
        tonal=('notes', ['E7', 'D7', 'F#7']), hpf=400)
def _marker(v, r):
    n = _n(0.78)
    y = np.zeros(n)
    for t0, d, nm in [(0.0, 0.17, 'E7'), (0.23, 0.2, 'D7'), (0.5, 0.25, 'F#7')]:
        m = _n(d)
        tm = _t(m)
        jit = lp(white(m, r), 30)
        jit = jit / (np.std(jit) + 1e-12)
        f = hz(nm) * (1 + 0.006 * jit + 0.01 * np.sin(TAU * 3 * tm / d))
        sq = sine(f, m) + 0.25 * sine(2 * f, m)
        am = np.abs(lp(white(m, r), 60))
        am = 0.6 + 0.4 * am / (am.max() + 1e-12)
        fr = 0.35 * pk(bp(white(m, r), 2500, 8000))
        e = pw(m, [(0, 0), (0.015, 1), (d - 0.03, 0.8), (d, 0)])
        put(y, (sq * am + fr) * e, t0, 1.0)
    return y


@_sound('cart', 'Shopping trolley rolling: wheel rumble + wire-basket rattle (~1.2 s, loopable feel).',
        hpf=40)
def _cart(v, r):
    n = _n(1.2)
    t = _t(n)
    rumble = pk(lp(white(n, r), 300)) * (0.55 + 0.45 * np.abs(np.sin(TAU * 3.5 * t)))
    imp = np.where(events(n, 55, r), r.choice([-1.0, 1.0], n) * r.uniform(0.2, 1.0, n), 0.0)
    metal = reson(imp, 2900, 30) + 0.8 * reson(imp, 4400, 35) + 0.5 * reson(imp, 6300, 40)
    clk = np.zeros(n)
    for tt in np.arange(0.05, 1.15, 0.14):
        m = _n(0.04)
        clk_m = modes(m, [(140, 0.8, 0.012), (420, 0.3, 0.006)])
        put(clk, clk_m, tt, 0.6)
    e = pw(n, [(0, 0), (0.1, 1), (0.95, 1), (1.2, 0)])
    return (0.6 * rumble + 0.55 * pk(metal) + 0.3 * pk(clk)) * e


@_sound('bag', 'Paper shopping-bag rustle / crinkle in three handfuls (~0.6 s).', variants=2, hpf=200)
def _bag(v, r):
    n = _n(0.62)
    dens = pw(n, [(0, 0), (0.03, 1), (0.15, 0.6), (0.2, 0.1), (0.24, 1), (0.36, 0.5), (0.41, 0.1),
                  (0.44, 0.9), (0.56, 0.4), (0.62, 0)])
    imp = np.where(events(n, dens * 900, r),
                   r.choice([-1.0, 1.0], n) * np.minimum(r.pareto(2.0, n), 8.0), 0.0)
    cr = bp(imp, 1500, 9000) + 0.5 * reson(imp, 3000, 2)
    rustle = pk(bp(white(n, r), 1000, 6000)) * dens
    return pk(cr) + 0.25 * rustle


@_sound('chime', 'Door / notification chime: vibraphone "ding-dong" F#6 -> D6 (an 8th note apart).',
        tonal=('notes', ['F#6', 'D6']), hpf=150)
def _chime(v, r):
    n = _n(1.4)
    y = np.zeros(n)

    def vib(f, d, tau):
        m = _n(d)
        tm = _t(m)
        s = np.sin(TAU * f * tm) + 0.22 * np.sin(TAU * 4 * f * tm) * np.exp(-tm / (tau * 0.25))
        s *= 1 + 0.18 * np.sin(TAU * 5.2 * tm)
        s = s * env(m, 0.0015, tau, rel=0.15)
        return s + 0.12 * pk(hp(white(m, r), 2500)) * env(m, 0.0002, 0.0015, rel=0.002)
    put(y, vib(hz('F#6'), 1.4, 0.55), 0.0, 1.0)
    put(y, vib(hz('D6'), 1.1, 0.6), 0.3, 0.95)
    return y


# ---------------------------------------------------------------- outdoors / nature
@_sound('birds', 'Spring birdsong (~2 s): bellbird-like liquid notes A6 F#6 D7 B6 D7, silvereye '
        'chirps and a little trill.', tonal=('glide', ['A6', 'F#6', 'D7', 'B6']), hpf=500)
def _birds(v, r):
    n = _n(2.0)
    y = np.zeros(n)
    for nm, t0, d in [('A6', 0.12, 0.16), ('F#6', 0.36, 0.14), ('D7', 0.55, 0.22), ('B6', 1.22, 0.18),
                      ('D7', 1.42, 0.26)]:
        m = _n(d + 0.05)
        tm = _t(m)
        f = hz(nm) * (1 - 0.04 * np.exp(-tm / 0.015))
        s = sine(f, m) + 0.12 * sine(2 * f, m)
        put(y, s * env(m, 0.006, d * 0.6, rel=0.03), t0, 0.8)
    for t0 in (0.82, 0.9, 0.975, 1.05):
        m = _n(0.045)
        tm = _t(m)
        f = 6200 * 0.55 ** (tm / 0.045)
        put(y, sine(f, m) * env(m, 0.003, 0.015, rel=0.01), t0, 0.45)
    m = _n(0.26)
    tm = _t(m)
    f = hz('D8') + 450 * np.sin(TAU * 28 * tm)
    s = sine(f, m) * pw(m, [(0, 0), (0.03, 1), (0.2, 0.7), (0.26, 0)]) * (0.6 + 0.4 * np.sin(TAU * 28 * tm) ** 2)
    put(y, s, 1.62, 0.3)
    return y


@_sound('gull', 'Seagull "kyow kyow kyow" (three nasal calls, glide F#6 -> A5).',
        tonal=('glide', ['F#6', 'A5']), hpf=300)
def _gull(v, r):
    n = _n(1.3)
    y = np.zeros(n)
    for t0, a, d in [(0.0, 1.0, 0.3), (0.38, 0.8, 0.27), (0.72, 0.62, 0.33)]:
        m = _n(d)
        tm = _t(m)
        u = tm / d
        f = np.interp(u, [0, 0.12, 0.42, 0.62, 1], [1150, hz('F#6'), hz('F#6') * 0.995, hz('A5'),
                                                     hz('A5') * 0.99])
        s = saw(f, m)
        s = reson(s, 1650, 3) + 0.7 * reson(s, 2900, 4) + 0.3 * s
        rough = 1 + 0.25 * np.sin(TAU * 70 * tm)
        e = pw(m, [(0, 0), (0.025, 1), (0.6 * d, 0.8), (d, 0)])
        put(y, pk(s) * rough * e, t0, a)
    return y


@_sound('bell', 'Clocktower bell "BONG": deep B-tuned bell (hum B1, prime B2, minor-third tierce D3, '
        'quint F#3, nominal B3...) - all partials in D major; 2.5 s decay.',
        tonal=('notes', ['B2', 'D3', 'F#3', 'B3']), hpf=28)
def _bell(v, r):
    f = hz('B2')
    n = _n(2.5)
    e = lambda st: 2 ** (st / 12)
    parts = [(0.5, 0.55, 0.95), (1.0, 0.5, 0.8), (e(3), 0.6, 0.7), (e(7), 0.25, 0.5), (2.0, 0.9, 0.6),
             (e(17), 0.35, 0.32), (e(19), 0.3, 0.26), (4.0, 0.25, 0.19), (e(29), 0.15, 0.12),
             (e(31), 0.12, 0.1), (8.0, 0.07, 0.06)]
    y = bell(f, n, parts, beat=0.7)
    m = _n(0.1)
    strike = 0.35 * pk(bp(white(m, r), 800, 5000)) * env(m, 0.0003, 0.006, rel=0.01)
    strike += modes(m, [(1720, 0.25, 0.03), (2380, 0.2, 0.02)])
    put(y, strike, 0.0, 1.0)
    return y * env(n, 0.0015, None, rel=0.5)


@_sound('kiwi', 'Kiwi call: six shrill rising whistles F#6 -> D7, crescendo then fade (~1.9 s).',
        tonal=('glide', ['D7']), hpf=500)
def _kiwi(v, r):
    n = _n(1.95)
    y = np.zeros(n)
    for i, amp in enumerate([0.55, 0.75, 0.9, 1.0, 0.9, 0.75]):
        t0 = 0.02 + i * 0.31
        d = 0.24
        m = _n(d)
        u = _t(m) / d
        f_lo = hz('F#6') * (1 + 0.02 * i)
        uu = np.clip(u / 0.55, 0, 1)
        f = f_lo * (hz('D7') / f_lo) ** (1 - (1 - uu) ** 3)
        f *= 1 + 0.004 * np.sin(TAU * 9 * _t(m)) * (u > 0.55)
        s = sine(f, m) + 0.18 * sine(2 * f, m) + 0.06 * pk(bp(white(m, r), 1500, 4000))
        e = pw(m, [(0, 0), (0.03, 0.8), (0.17, 1), (0.24, 0)])
        put(y, s * e, t0, amp)
    return y


@_sound('zzz', 'Cartoon sleep "Zzz": soft fluttery snore (D3) then a little whistle out A5 -> F#5.',
        tonal=('glide', ['D3', 'A5', 'F#5']), hpf=40)
def _zzz(v, r):
    n = _n(1.3)
    y = np.zeros(n)
    m = _n(0.65)
    tm = _t(m)
    f = hz('D3') * (1 + 0.03 * np.sin(TAU * 1.5 * tm))
    buzz = lp(saw(f, m), 900) * (0.45 + 0.55 * np.abs(np.sin(TAU * 14 * tm)))
    breath = 0.25 * pk(bp(white(m, r), 300, 1800))
    put(y, (0.7 * pk(buzz) + breath) * pw(m, [(0, 0), (0.25, 0.8), (0.5, 1), (0.65, 0)]), 0.0, 1.0)
    m = _n(0.55)
    tm = _t(m)
    f = np.interp(tm, [0, 0.2, 0.23, 0.55], [hz('A5'), hz('A5'), hz('F#5'), hz('F#5')])
    wh = sine(f, m) * pw(m, [(0, 0), (0.08, 0.5), (0.3, 0.45), (0.55, 0)])
    put(y, wh, 0.72, 0.45)
    return y


@_sound('sweep', 'Broom sweeping: two bristly swish strokes.', variants=1, hpf=300)
def _sweep(v, r):
    n = _n(0.7)
    y = np.zeros(n)
    for t0, a in [(0.0, 1.0), (0.33, 0.8)]:
        m = _n(0.3)
        tm = _t(m)
        src = white(m, r) * (0.5 + 0.5 * (r.random(m) < 0.3))
        s = pk(swept(m, 2500 * 2 ** (1.2 * tm / 0.3), r, width=0.7, src=src))
        put(y, s * pw(m, [(0, 0), (0.04, 1), (0.12, 0.8), (0.3, 0)]), t0, a)
    return y


@_sound('shimmer', 'Magic shimmer (~2 s): swelling D-major cloud D6 F#6 A6 D7 F#7 A7 with tremolo, '
        'glitter grains and air.', tonal=('notes', ['D6', 'F#6', 'A6', 'D7', 'F#7', 'A7']), hpf=200)
def _shimmer(v, r):
    n = _n(2.0)
    t = _t(n)
    y = np.zeros(n)
    for i, (nm, t0) in enumerate([('D6', 0.0), ('F#6', 0.12), ('A6', 0.24), ('D7', 0.36),
                                  ('F#7', 0.48), ('A7', 0.6)]):
        f = hz(nm)
        m = n - _n(t0)
        tm = _t(m)
        s = sine(f * (1 + 0.002 * np.sin(TAU * 0.7 * tm)), m) + 0.6 * sine(f * 1.0035, m)
        trem = 0.7 + 0.3 * np.sin(TAU * (5.5 + 0.7 * i) * tm)
        e = pw(m, [(0, 0), (0.25, 1), (max(0.3, 1.3 - t0), 0.6), (2.0 - t0, 0)])
        put(y, s * trem * e, t0, 0.5 - 0.04 * i)
    pool = ['D7', 'E7', 'F#7', 'A7', 'B7', 'D8']
    for tt in np.sort(r.uniform(0.05, 1.7, 36)):
        a = 0.3 * np.interp(tt, [0, 0.7, 1.8], [0.3, 1, 0.2])
        put(y, twinkle(hz(pool[int(r.integers(len(pool)))]), 0.25, r.uniform(0.04, 0.1)), tt, a)
    air = pk(hp(white(n, r), 6000)) * pw(n, [(0, 0), (0.6, 1), (2.0, 0)])
    return y + 0.08 * air


@_sound('coin', 'Coin / reward: classic chip two-note B5 -> E6.', tonal=('notes', ['B5', 'E6']),
        hpf=150)
def _coin(v, r):
    n = _n(0.55)
    t = _t(n)
    f = np.where(t < 0.065, hz('B5'), hz('E6'))
    y = 0.8 * lp(pulse(f, n, 0.5), 7000) + 0.4 * sine(f, n)
    e = np.where(t < 0.065, 1.0, np.exp(-(t - 0.065) / 0.16))
    return y * e * env(n, 0.001, None, rel=0.04)


@_sound('tick', 'Clock tick; variants alternate tick / tock automatically (time-lapses).', variants=2,
        hpf=300)
def _tick(v, r):
    n = _n(0.06)
    spec = ([(3400, 0.8, 0.006), (5150, 0.5, 0.004), (1250, 0.3, 0.005)],
            [(2250, 0.8, 0.007), (3650, 0.45, 0.005), (820, 0.35, 0.007)])[v]
    return modes(n, spec) + 0.4 * pk(hp(white(n, r), 3000)) * env(n, 0.0001, 0.0007, rel=0.001)


@_sound('water', 'Running tap water / rinsing (~1.5 s): rush + many small bubbles + splash.', hpf=80)
def _water(v, r):
    n = _n(1.5)
    rush = bp(white(n, r), 300, 6500)
    fl = lp(white(n, r), 6)
    rush = pk(rush) * (1 + 0.15 * fl / (np.std(fl) + 1e-12))
    bub = np.zeros(n)
    for tt in np.sort(r.uniform(0.0, 1.45, 130)):
        f0 = r.uniform(400, 2600)
        m = _n(0.03)
        tm = _t(m)
        b = sine(f0 * (1 + 0.4 * tm / 0.03), m) * env(m, 0.001, r.uniform(0.004, 0.012), rel=0.004)
        put(bub, b, tt, r.uniform(0.2, 1.0))
    splash = pk(hp(white(n, r), 4000))
    e = pw(n, [(0, 0), (0.06, 1), (1.2, 1), (1.5, 0)])
    return (0.8 * rush + 0.7 * pk(bub) + 0.2 * splash) * e


@_sound('microwave', 'Microwave running (~2 s): relay clunk, A2 magnetron buzz, fan, turntable wobble.',
        tonal=('notes', ['A2']), hpf=40)
def _microwave(v, r):
    n = _n(2.0)
    t = _t(n)
    y = np.zeros(n)
    m = _n(0.05)
    put(y, modes(m, [(1900, 0.6, 0.006), (3100, 0.35, 0.004), (240, 0.5, 0.012)]), 0.0, 0.5)
    f = hz('A2')
    buzz = 0.6 * lp(pulse(f, n, 0.3), 1100) + 0.5 * sine(f, n) + 0.25 * sine(2 * f, n)
    buzz = pk(buzz) * (1 + 0.08 * np.sin(TAU * 0.45 * t))
    fan = pk(lp(white(n, r), 750, 4)) * (1 + 0.05 * np.sin(TAU * 0.9 * t))
    e = pw(n, [(0, 0), (0.02, 0), (0.12, 1), (1.7, 1), (2.0, 0)])
    return y + (0.6 * buzz + 0.3 * fan) * e


@_sound('boing', 'Cartoon spring "boiiing" wobbling around D4 (jumps, bounces, surprises).',
        tonal=('glide', ['D4']), hpf=60)
def _boing(v, r):
    n = _n(0.8)
    t = _t(n)
    w = 0.45 * np.exp(-t / 0.17) * np.sin(TAU * 10 * t) - 0.7 * np.exp(-t / 0.02)
    f = hz('D4') * 2 ** w
    s = saw(f, n)
    s = 0.6 * reson(s, 1200, 2) + 0.4 * lp(s, 2000)
    return (0.6 * pk(s) + sine(f, n)) * env(n, 0.002, 0.28, rel=0.05)


@_sound('shutter', 'Camera shutter "ka-chick" (mirror up / down with a tiny whirr).', hpf=200)
def _shutter(v, r):
    n = _n(0.26)
    y = np.zeros(n)
    m = _n(0.04)
    c1 = pk(hp(white(m, r), 2500)) * env(m, 0.0002, 0.0018, rel=0.002)
    c1 += modes(m, [(4200, 0.5, 0.006), (7000, 0.3, 0.004), (1500, 0.3, 0.006)])
    put(y, c1, 0.0, 1.0)
    m = _n(0.07)
    tm = _t(m)
    wh = pk(bp(white(m, r), 900, 3200)) * (0.5 + 0.5 * np.sin(TAU * 140 * tm)) * pw(m, [(0, 0), (0.02, 1), (0.07, 0)])
    put(y, wh, 0.01, 0.18)
    m = _n(0.05)
    c2 = 0.8 * pk(hp(white(m, r), 2000)) * env(m, 0.0002, 0.0015, rel=0.002)
    c2 += modes(m, [(2600, 0.6, 0.008), (5200, 0.3, 0.005), (1100, 0.3, 0.008)])
    put(y, c2, 0.085, 0.8)
    return y


# ---------------------------------------------------------------- HUD (added by the mixer)
@_sound('banner', 'HUD "NEW QUEST": airy whoosh + soft rising two-note blip A5 -> D6; the D6 lands '
        '0.267 s (8 frames) after the cue = on the beat when cued at local frame 10.', hit=0.267,
        tonal=('notes', ['A5', 'D6']))
def _banner(v, r):
    n = _n(0.62)
    t = _t(n)
    y = np.zeros(n)
    fc = np.interp(t, [0, 0.2, 0.62], [450, 3400, 1500])
    wh = pk(swept(n, fc, r, width=0.7)) * pw(n, [(0, 0), (0.16, 1), (0.3, 0.35), (0.62, 0)])
    put(y, wh, 0.0, 0.32)
    n1 = note(hz('A5'), 0.12, 'pulse', 0.25, a=0.002, tau=0.06, rel=0.03, lpf=5000, sub=0.8)
    n2 = note(hz('D6'), 0.34, 'pulse', 0.25, a=0.002, tau=0.16, rel=0.08, lpf=5000, sub=0.8,
              vib=0.004, vib_delay=0.08)
    put(y, n1, 0.192, 0.8)
    put(y, n2, 0.267, 1.0)
    return y


@_sound('quest', 'HUD "QUEST COMPLETE": bright 3-note chime D6 F#6 A6 (2 frames apart; the A6 lands '
        '4 frames after the cue = on the beat when cued at frames-58) with a 16th-note echo.',
        hit=0.133, tonal=('notes', ['D6', 'F#6', 'A6']))
def _quest(v, r):
    n = _n(1.35)
    y = np.zeros(n)
    sp = 1 / 15
    for i, (nm, d, tau) in enumerate([('D6', 0.3, 0.09), ('F#6', 0.3, 0.1), ('A6', 1.0, 0.42)]):
        f = hz(nm)
        m = _n(d)
        tm = _t(m)
        b = (np.sin(TAU * f * tm) + 0.3 * np.sin(TAU * 2 * f * 1.002 * tm) * np.exp(-tm / (tau * 0.5))
             + 0.08 * np.sin(TAU * 3 * f * tm) * np.exp(-tm / (tau * 0.3)))
        c = 0.35 * lp(pulse(f, m, 0.25), 6000) * np.exp(-tm / 0.05)
        put(y, (b + c) * env(m, 0.0015, tau, rel=min(0.1, d * 0.3)), i * sp, (0.75, 0.85, 1.0)[i])
    for tt, nm in [(0.15, 'D7'), (0.22, 'A7')]:
        put(y, twinkle(hz(nm), 0.3, 0.07), tt, 0.14)
    y = delay(y, 0.15, 0.22)
    return delay(y, 0.3, 0.06)


# ---------------------------------------------------------------- bonus sounds
@_sound('growl', 'Stomach growl: low gurgly rumble with a wobbly A2 -> F#2 groan (hungry!).',
        hpf=35)
def _growl(v, r):
    n = _n(0.9)
    t = _t(n)
    am = lp(white(n, r), 9)
    am = np.clip(0.5 + 0.5 * am / (np.abs(am).max() + 1e-12), 0.05, 1)
    rum = pk(bp(white(n, r), 110, 520)) * am
    f = hz('A2') * (hz('F#2') / hz('A2')) ** (t / 0.9) * (1 + 0.06 * np.sin(TAU * 11 * t))
    gro = lp(saw(f, n), 700) + sine(f, n)
    gur = np.zeros(n)
    for tt in (0.15, 0.32, 0.41, 0.62):
        m = _n(0.06)
        tm = _t(m)
        put(gur, sine(r.uniform(200, 380) * (1 + 0.5 * tm / 0.06), m) * env(m, 0.002, 0.02, rel=0.01), tt,
            r.uniform(0.5, 1))
    e = pw(n, [(0, 0), (0.1, 1), (0.6, 0.8), (0.9, 0)])
    return (0.6 * rum + 0.45 * pk(gro) + 0.35 * gur) * e


@_sound('clap', 'Single hand clap / high-five slap (layered 3-burst clap).', variants=2, hpf=150)
def _clap(v, r):
    n = _n(0.22)
    y = np.zeros(n)
    for tt, a in [(0.0, 0.8), (0.009, 0.9), (0.017, 1.0)]:
        m = _n(0.012)
        put(y, pk(bp(white(m, r), 900, 3200)) * env(m, 0.0003, 0.003, rel=0.003), tt, a)
    m = _n(0.2)
    put(y, pk(bp(white(m, r), 800, 3000)) * env(m, 0.0005, 0.03, rel=0.02), 0.022, 0.8)
    return y


@_sound('stamp', 'Big rubber-stamp impact: heavy thump + chip D-major stab + sparkle (rank stamp, '
        '"DAY COMPLETE").', tonal=('notes', ['D4', 'A4', 'D5', 'F#5']), hpf=30)
def _stamp(v, r):
    n = _n(0.7)
    t = _t(n)
    y = sine(hz('D2') + 60 * np.exp(-t / 0.015), n) * env(n, 0.001, 0.09, rel=0.05)
    y += modes(n, [(hz('B3'), 0.5, 0.04), (hz('A4') * 1.07, 0.3, 0.025)])
    y += 0.4 * pk(lp(white(n, r), 1500)) * env(n, 0.0005, 0.01, rel=0.01)
    stab = sum(note(hz(nm), 0.45, 'pulse', 0.25, a=0.002, tau=0.12, rel=0.05, lpf=5000)
               for nm in ('D4', 'A4', 'D5', 'F#5'))
    put(y, pk(stab), 0.0, 0.55)
    for tt, nm in [(0.06, 'A7'), (0.13, 'D7')]:
        put(y, twinkle(hz(nm), 0.3, 0.06), tt, 0.2)
    return y


@_sound('grind', 'Coffee grinder (~1.3 s): A3 motor whine under load + crunchy grinding beans.',
        tonal=('glide', ['A3']), hpf=60)
def _grind(v, r):
    n = _n(1.3)
    load = lp(white(n, r), 4)
    load = load / (np.abs(load).max() + 1e-12)
    f = hz('A3') * (1 - 0.03 * load)
    motor = lp(saw(f, n), 2500) + 0.5 * sine(2 * f, n)
    gr = pk(bp(crunch(n, 7000, r) + 0.5 * white(n, r), 1200, 6000))
    am = np.abs(lp(white(n, r), 40))
    gr *= 0.4 + 0.6 * am / (am.max() + 1e-12)
    e = pw(n, [(0, 0), (0.08, 1), (1.05, 1), (1.3, 0)])
    return (0.5 * pk(motor) + 0.6 * gr) * e


@_sound('puff', 'Soft "poof" puff of smoke (outfit change, dust) with a two-ping sparkle tail.',
        hpf=60)
def _puff(v, r):
    n = _n(0.55)
    t = _t(n)
    fc = np.interp(t, [0, 0.05, 0.55], [900, 700, 250])
    y = pk(swept(n, fc, r, width=0.9)) * pw(n, [(0, 0), (0.012, 1), (0.12, 0.5), (0.55, 0)])
    put(y, twinkle(hz('D7'), 0.3, 0.07), 0.05, 0.2)
    put(y, twinkle(hz('A7'), 0.3, 0.07), 0.1, 0.15)
    return y


@_sound('star', 'Shooting star: high descending sparkle-whoosh with pings A7 F#7 D7 A6 F#6 D6.',
        tonal=('notes', ['A7', 'F#7', 'D7', 'A6', 'F#6', 'D6']), hpf=300)
def _star(v, r):
    n = _n(1.3)
    t = _t(n)
    y = 0.5 * pk(swept(n, 9000 * (1800 / 9000) ** (t / 1.3), r, width=0.5)) * pw(
        n, [(0, 0), (0.15, 1), (0.7, 0.6), (1.3, 0)])
    for i, nm in enumerate(['A7', 'F#7', 'D7', 'A6', 'F#6', 'D6']):
        put(y, twinkle(hz(nm), 0.4, 0.16), 0.05 + i * 0.11, 0.6 - 0.05 * i)
    return y


# ============================================================== ATLAS: landscapes, weather, travel
# (EQUESTRIA: THE WORLD ATLAS - no creature sounds: never add hooves, animal or bird calls here)
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


def pink(n, r):
    """Pink-ish noise (-3 dB/octave, Kellet filter), unit std."""
    w = 4800
    y = signal.lfilter(_PINK_B, _PINK_A, white(n + w, r))[w:]
    return y / (np.std(y) + 1e-12)


def _lfo(n, r, rate):
    """Smooth random control curve (zero mean, ~unit std), new random knot every 1/rate s."""
    k = int(n / SR * rate) + 3
    return pw(n, list(zip(np.arange(k) / rate, r.standard_normal(k))))


def _bursts(n, spec):
    """Sum of swells [(t0, amp, attack, tau)]: squared ramp up over `attack`, then exp decay."""
    t = _t(n)
    e = np.zeros(n)
    for t0, a, att, tau in spec:
        u = t - t0
        m = u > 0
        e[m] += a * np.minimum(1.0, u[m] / att) ** 2 * np.exp(-np.maximum(u[m] - att, 0) / tau)
    return e


def _grains(n, rate, r, lo, hi):
    """Granular crackle: Poisson impulses (rate events/s, scalar or array) band-passed and smoothed."""
    imp = np.where(events(n, rate, r), r.uniform(0.25, 1.0, n) * r.choice([-1.0, 1.0], n), 0.0)
    return pk(lp(bp(imp, lo, hi), hi * 1.2))


def _gate(n, spans, smooth=350.0):
    """0/1 gate open during [(t0, t1), ...], edges smoothed (no clicks)."""
    g = np.zeros(n)
    for a, b in spans:
        g[_n(a):_n(b)] = 1.0
    return lp(g, smooth, 2)


@_sound('wind', 'Wind: soft gusting air (two swells) with a faint breathy whistle hovering A5 -> B5 '
        '-> D6; variant 2 = strong howling gust (wailing F#5 -> A5 -> D6 whistle, rougher air). '
        'Soft ends, layer/re-cue freely.', variants=2, hpf=40)
def _wind(v, r):
    n = _n(2.0)
    strong = v == 1
    if strong:
        g = pw(n, [(0, 0.2), (0.35, 0.6), (0.9, 1.0), (1.3, 0.75), (1.6, 0.95), (2.0, 0.4)])
        wpts = [(0, 'F#5'), (0.5, 'F#5'), (0.85, 'A5'), (1.3, 'A5'), (1.6, 'D6'), (2.0, 'D6')]
    else:
        g = pw(n, [(0, 0.25), (0.5, 0.75), (0.85, 0.5), (1.3, 0.9), (2.0, 0.3)])
        wpts = [(0, 'A5'), (0.6, 'A5'), (0.95, 'B5'), (1.25, 'B5'), (1.5, 'D6'), (2.0, 'D6')]
    g = np.clip(g * (1 + 0.12 * _lfo(n, r, 6)), 0.05, None)
    fc = (450 + 1800 * g) if strong else (350 + 1100 * g)
    body = pk(swept(n, fc, r, width=0.55, lo=80, hi=9000, bands=14, src=pink(n, r))) * g
    low = pk(lp(white(n, r), 160)) * g
    f = pw(n, [(tt, hz(nm)) for tt, nm in wpts]) * (1 + 0.004 * _lfo(n, r, 9))
    breath = 0.55 + 0.45 * np.abs(_lfo(n, r, 22))
    wh = (sine(f, n) + (0.25 * sine(2 * f, n) if strong else 0)) * breath * g ** 2
    wh += 0.3 * pk(bp(white(n, r), 700, 1500)) * g ** 2
    y = 0.8 * body + 0.35 * low + (0.24 if strong else 0.1) * pk(wh)
    return y * env(n, 0.25, None, rel=0.35)


@_sound('waterfall', 'Waterfall: roaring cascade (pink roar + low thunder of the plunge pool), '
        'spray fizz and tumbling water chunks; steady, loop-friendly.', hpf=35)
def _waterfall(v, r):
    n = _n(2.0)
    roar = pk(lp(hp(pink(n, r), 60), 3500))
    rumble = pk(lp(white(n, r), 110, 2))
    spray = pk(hp(white(n, r), 3500)) * (0.7 + 0.5 * np.abs(_lfo(n, r, 18)))
    chunks = _grains(n, 140, r, 600, 4000)
    surge = 1 + 0.12 * _lfo(n, r, 1.5)
    y = (0.9 * roar + 0.45 * rumble + 0.18 * spray + 0.3 * chunks) * surge
    return y * env(n, 0.15, None, rel=0.25)


@_sound('whistle', 'Steam-train whistle "toot-TOOOT": short blast then a long one an 8th later '
        '(0.3 s), three-chime chord A4 D5 F#5 with a pressure scoop, breathy steam and hiss.',
        tonal=('notes', ['A4', 'D5', 'F#5']), hpf=120)
def _whistle(v, r):
    n = _n(1.2)
    y = np.zeros(n)

    def blast(d):
        m = _n(d)
        tm = _t(m)
        bend = 2 ** ((-0.9 * np.exp(-tm / 0.03) - 0.2 * np.clip((tm - (d - 0.12)) / 0.12, 0, 1)) / 12)
        out = np.zeros(m)
        for i, (nm, a) in enumerate((('A4', 0.8), ('D5', 1.0), ('F#5', 0.7))):
            f = hz(nm) * bend * (1 + 0.003 * np.sin(TAU * 5.5 * tm + i))
            out += a * (sine(f, m) + 0.3 * lp(saw(f, m), 2400) + 0.2 * sine(2 * f, m))
        out = pk(out) + 0.12 * pk(bp(white(m, r), 900, 5000))
        return out * pw(m, [(0, 0), (0.035, 1), (d - 0.1, 0.85), (d, 0)])
    put(y, blast(0.25), 0.0, 0.8)
    put(y, blast(0.86), 0.3, 1.0)
    hiss = pk(hp(white(n, r), 2500)) * pw(n, [(0, 0), (0.02, 0.6), (0.3, 0.3), (1.12, 0.45), (1.2, 0)])
    return y + 0.1 * hiss


@_sound('chug', 'Steam locomotive running: four chuffs on 100 BPM 8ths (0 / 0.3 / 0.6 / 0.9 s, '
        'accent on 1 and 3), rail "clickety-clack", axle rumble, steam tail. Re-cue every 36 frames '
        '(2 beats) for a continuous run; variants alternate.', variants=2, hpf=40)
def _chug(v, r):
    n = _n(1.6)
    y = np.zeros(n)
    for tt, a in [(0.0, 1.0), (0.3, 0.6), (0.6, 0.85), (0.9, 0.55)]:
        m = _n(0.34)
        tm = _t(m)
        ch = pk(swept(m, 400 + 1800 * np.exp(-tm / 0.08), r, width=0.8, lo=150, hi=8000, bands=10))
        ch = ch * env(m, 0.004, 0.075, rel=0.06)
        thump = sine(70 * (1 + 0.6 * np.exp(-tm / 0.02)), m) * env(m, 0.002, 0.05, rel=0.06)
        put(y, ch + 0.5 * thump, tt, a)
    for tt in (0.45, 0.525, 1.05, 1.125):
        m = _n(0.08)
        clack = modes(m, [(r.uniform(1150, 1250), 0.6, 0.012), (2630, 0.4, 0.008), (430, 0.5, 0.015)])
        clack += 0.3 * pk(hp(white(m, r), 2000)) * env(m, 0.0002, 0.002, rel=0.003)
        put(y, clack, tt, 0.35)
    rum = pk(lp(white(n, r), 140)) * pw(n, [(0, 0), (0.25, 1), (1.2, 1), (1.6, 0)])
    hiss = pk(bp(white(n, r), 2000, 9000)) * pw(n, [(0, 0), (0.9, 0.2), (1.1, 0.5), (1.6, 0)])
    return y + 0.22 * rum + 0.12 * hiss


@_sound('steam', 'Steam hiss / vent "PSSHHH": valve pop, broadband hiss that darkens and fades.',
        hpf=200)
def _steam(v, r):
    n = _n(0.9)
    t = _t(n)
    y = pk(swept(n, 5500 * (0.55 + 0.45 * np.exp(-t / 0.25)), r, width=0.9, lo=400, hi=14000, bands=14))
    y = y * (1 + 0.15 * _lfo(n, r, 30)) * pw(n, [(0, 0), (0.015, 1), (0.25, 0.7), (0.9, 0)])
    m = _n(0.03)
    put(y, pk(bp(white(m, r), 300, 3000)) * env(m, 0.0005, 0.006, rel=0.005), 0.0, 0.5)
    return y


@_sound('thunder', 'Thunder: distant rolling thunder - soft crack, then a long rumbling roll with '
        'three swells; variant 2 = close strike: tearing crack, huge boom, heavy roll.', variants=2,
        hit=0.02, hpf=35)
def _thunder(v, r):
    n = _n(2.5)
    t = _t(n)
    flutter = np.clip(1 + 0.3 * _lfo(n, r, 12), 0.3, None)
    if v == 0:
        e = _bursts(n, [(0.0, 0.35, 0.02, 0.12), (0.25, 1.0, 0.15, 0.5), (0.9, 0.7, 0.2, 0.45),
                        (1.55, 0.45, 0.2, 0.5)])
        roll = pk(lp(pink(n, r), 450, 2)) + 0.6 * pk(lp(white(n, r), 90, 2))
        y = pk(roll) * e * flutter
        y += 0.25 * pk(lp(white(n, r), 1500)) * _bursts(n, [(0.0, 1.0, 0.01, 0.06)])
    else:
        m = _n(0.2)
        tm = _t(m)
        tear = pk(hp(crunch(m, 3000, r) + white(m, r), 900)) * np.clip(1 + 0.8 * _lfo(m, r, 60), 0, None)
        tear *= env(m, 0.0005, 0.05, rel=0.03)
        boom = sine(30 + 45 * np.exp(-t / 0.15), n) * _bursts(n, [(0.02, 1.0, 0.01, 0.45)])
        roll = pk(lp(pink(n, r), 700, 2)) * _bursts(n, [(0.03, 1.0, 0.03, 0.3), (0.5, 0.6, 0.12, 0.45),
                                                        (1.2, 0.4, 0.2, 0.5)]) * flutter
        y = 0.8 * boom + 0.9 * roll + 0.45 * pk(lp(white(n, r), 220)) * _bursts(n, [(0.01, 1, 0.01, 0.35)])
        put(y, tear, 0.0, 0.9)
    return y * env(n, 0.002, None, rel=0.4)


@_sound('zap', 'Magic electric zap (rainbow lightning): bright crackling arc, descending zappy buzz, '
        'glassy D7 F#7 A7 sparkle on top.', hpf=150)
def _zap(v, r):
    n = _n(0.7)
    t = _t(n)
    cr = _grains(n, 2400 * pw(n, [(0, 1), (0.25, 0.7), (0.55, 0.15), (0.7, 0)]), r, 1500, 11000)
    gate = np.clip(lp((_lfo(n, r, 35) > -0.4).astype(float), 300), 0, 1)
    f = 260 + 1500 * 0.2 ** (t / 0.35)
    f = f * (1 + 0.06 * np.sign(np.sin(TAU * 55 * t)))
    buzz = lp(saw(f, n), 9000) * env(n, 0.001, 0.2, rel=0.05)
    y = 0.65 * cr * gate * pw(n, [(0, 0), (0.005, 1), (0.3, 0.6), (0.7, 0)]) + 0.4 * pk(buzz)
    for nm, tt, a in (('D7', 0.02, 0.35), ('A7', 0.09, 0.3), ('F#7', 0.16, 0.25)):
        put(y, twinkle(hz(nm), 0.45, 0.1), tt, a)
    return y


@_sound('splash', 'Big splash / dive into water: low "ploonk" cavity, white-water burst, spray '
        'droplets falling back, rising bubble chirps.', hit=0.02, hpf=50)
def _splash(v, r):
    n = _n(0.9)
    t = _t(n)
    y = pk(swept(n, 900 + 3500 * np.exp(-t / 0.12), r, width=0.9, lo=150, hi=12000, bands=14))
    y = y * pw(n, [(0, 0), (0.02, 1), (0.12, 0.55), (0.45, 0.12), (0.9, 0)])
    m = _n(0.25)
    tm = _t(m)
    put(y, sine(70 + 260 * np.exp(-tm / 0.06), m) * env(m, 0.004, 0.07, rel=0.05), 0.0, 0.7)
    for tt in np.sort(r.uniform(0.12, 0.8, 45)):
        mm = _n(0.03)
        d = sine(r.uniform(1200, 4500) * (1 + 0.6 * _t(mm) / 0.03), mm)
        put(y, d * env(mm, 0.0008, r.uniform(0.004, 0.01), rel=0.004), tt, r.uniform(0.1, 0.35) * (1 - tt))
    for tt in np.sort(r.uniform(0.2, 0.85, 14)):
        mm = _n(0.06)
        b = sine(r.uniform(350, 1100) * (1 + 0.8 * _t(mm) / 0.06), mm)
        put(y, b * env(mm, 0.002, 0.02, rel=0.01), tt, 0.22)
    return y


@_sound('bubbles', 'Underwater bubble stream: liquid "bloop"s on 32nd notes in D-major pentatonic '
        '(each slides up into its note) over a muffled water murmur.',
        tonal=('notes', ['D5', 'E5', 'F#5', 'A5', 'B5', 'D6', 'F#6', 'A6']), hpf=80)
def _bubbles(v, r):
    n = _n(1.2)
    y = np.zeros(n)
    pool = ['D5', 'E5', 'F#5', 'A5', 'B5', 'D6', 'F#6', 'A6']
    k = 3
    for s in range(15):
        if s and r.random() < 0.15:
            continue
        k = int(np.clip(k + r.integers(-2, 3), 0, len(pool) - 1))
        f = hz(pool[k])
        b = note(f, 0.14, 'sine', a=0.002, tau=0.045, rel=0.03, glide=-2, glide_t=0.006)
        b += 0.15 * note(2 * f, 0.14, 'sine', a=0.002, tau=0.02, rel=0.03, glide=-2, glide_t=0.006)
        put(y, b, s * 0.075, r.uniform(0.55, 1.0))
    mur = pk(lp(pink(n, r), 450)) * (0.8 + 0.2 * _lfo(n, r, 4))
    return y + 0.07 * mur * env(n, 0.1, None, rel=0.2)


@_sound('lava', 'Lava pool: deep viscous "blorp" bubbles swelling and bursting (low glides + wet '
        'pop), magma rumble, faint crackle; variant 2 = eruption: huge boom, roaring blast, '
        'falling rock debris.', variants=2, hpf=28)
def _lava(v, r):
    n = _n(1.8)
    t = _t(n)
    rum = pk(lp(pink(n, r), 120, 2))
    if v == 0:
        y = 0.4 * rum * np.clip(0.75 + 0.25 * _lfo(n, r, 2), 0.2, None)
        for tt, f0, d in [(0.03, 70, 0.32), (0.5, 95, 0.26), (0.85, 60, 0.38), (1.3, 85, 0.3)]:
            m = _n(d)
            tm = _t(m)
            f = f0 * (1 + 1.2 * (tm / d) ** 2)
            b = tri(f, m) + 0.5 * sine(f, m)
            put(y, b * pw(m, [(0, 0), (0.04, 0.8), (d * 0.85, 1), (d, 0)]), tt, 0.8)
            pm = _n(0.07)
            put(y, pk(bp(white(pm, r), 200, 1800)) * env(pm, 0.0005, 0.012, rel=0.01), tt + d * 0.85, 0.6)
        y += 0.12 * _grains(n, 25, r, 2500, 9000)
        return y * env(n, 0.03, None, rel=0.2)
    boom = sine(32 + 90 * np.exp(-t / 0.08), n) * _bursts(n, [(0.0, 1.0, 0.004, 0.5)])
    blast = pk(lp(white(n, r), 450)) * _bursts(n, [(0.0, 1.0, 0.01, 0.25)])
    roar = pk(lp(pink(n, r), 1000)) * _bursts(n, [(0.05, 1.0, 0.25, 0.7)])
    debris = _grains(n, 60 * pw(n, [(0, 0), (0.3, 1), (1.2, 0.6), (1.8, 0)]), r, 400, 3000)
    y = 0.9 * boom + 0.6 * blast + 0.45 * roar + 0.3 * debris + 0.3 * rum
    return y * env(n, 0.002, None, rel=0.35)


@_sound('rumble', 'Low earth rumble (volcano waking, avalanche start, quake): sub-bass roll that '
        'swells in, grinding-rock texture, fades out.', hpf=35)
def _rumble(v, r):
    n = _n(2.0)
    sub = pk(bp(white(n, r), 32, 80, 2))
    mid = pk(lp(pink(n, r), 300, 2))
    grind = _grains(n, 45, r, 300, 1500)
    flutter = np.clip(1 + 0.35 * _lfo(n, r, 7), 0.25, None)
    return (0.9 * sub + 0.6 * mid + 0.2 * grind) * flutter * pw(n, [(0, 0), (0.6, 1), (1.4, 0.85), (2.0, 0)])


@_sound('avalanche', 'Avalanche: rushing snow-slide whoosh (hiss rises to 0.9 s then recedes) with '
        'icy granular crunch over a growing low rumble.', hpf=30)
def _avalanche(v, r):
    n = _n(2.0)
    sh = pk(swept(n, pw(n, [(0, 400), (0.9, 2600), (2.0, 700)]), r, width=0.7, lo=150, hi=12000, bands=16,
                  src=pink(n, r)))
    gr = _grains(n, 500 * pw(n, [(0, 0.2), (0.9, 1), (2.0, 0.3)]), r, 2000, 9000)
    rum = pk(lp(white(n, r), 90, 2)) * pw(n, [(0, 0.2), (0.7, 1), (1.6, 1), (2.0, 0.6)])
    return (0.8 * sh + 0.25 * gr + 0.6 * rum) * pw(n, [(0, 0), (0.5, 0.6), (0.9, 1), (1.4, 0.8), (2.0, 0)])


@_sound('neon', 'Neon tube switching on: starter tick, three buzzy flickers, then a steady A2 mains '
        'hum with bright tube buzz (fades by 0.5 s).', hpf=60)
def _neon(v, r):
    n = _n(0.5)
    f = hz('A2')
    buzz = lp(pulse(f, n, 0.12), 6000) + 0.8 * sine(f, n) + 0.4 * sine(2 * f, n)
    g = _gate(n, [(0.0, 0.035), (0.07, 0.09), (0.12, 0.135), (0.17, 0.5)])
    flick = _gate(n, [(0.0, 0.17)], 200) * pk(hp(white(n, r), 3000)) * 0.25
    y = pk(buzz) * g * pw(n, [(0, 1), (0.17, 1), (0.25, 0.6), (0.5, 0.5)]) + flick * g
    m = _n(0.03)
    put(y, modes(m, [(3100, 0.7, 0.004), (5200, 0.4, 0.003)]), 0.0, 0.6)
    return y * env(n, 0.0005, None, rel=0.12)


@_sound('firework', 'Firework: launch thump + rising whistle, BURST at 0.5 s (boom + bright crack), '
        'crackling glitter tail. Variants: small pop / medium shell / big shell (deeper boom, '
        'longer crackle).', hit=0.5, variants=3, hpf=35)
def _firework(v, r):
    n = _n(1.6)
    t = _t(n)
    tb = 0.5
    y = np.zeros(n)
    m = _n(0.08)
    put(y, sine(60 + 150 * np.exp(-_t(m) / 0.02), m) * env(m, 0.001, 0.02, rel=0.02), 0.0, 0.5)
    m = _n(tb - 0.04)
    tm = _t(m)
    f = 900 * 2.6 ** (tm / (tb - 0.04))
    wh = sine(f * (1 + 0.01 * np.sin(TAU * 30 * tm)), m) + 0.4 * pk(bp(white(m, r), 1500, 6000))
    put(y, wh * pw(m, [(0, 0), (0.05, 0.5), (tb - 0.07, 1), (tb - 0.04, 0)]), 0.02, (0.2, 0.28, 0.32)[v])
    mb = n - _n(tb)
    tm = _t(mb)
    size = (0.5, 0.8, 1.0)[v]
    boom = sine((110, 75, 55)[v] * (1 + 1.5 * np.exp(-tm / 0.015)), mb)
    boom *= env(mb, 0.001, (0.12, 0.22, 0.35)[v], rel=0.05)
    crack = pk(hp(white(mb, r), 1200)) * env(mb, 0.0004, (0.02, 0.03, 0.04)[v], rel=0.01)
    body = pk(lp(white(mb, r), 900)) * env(mb, 0.002, (0.08, 0.15, 0.25)[v], rel=0.05)
    put(y, 0.9 * size * boom + 0.8 * crack + 0.6 * size * body, tb, 1.0)
    t_end = min(1.55, tb + 0.6 + 0.45 * v)
    rate = (140, 280, 450)[v] * pw(n, [(0, 0), (tb + 0.1, 0), (tb + 0.2, 1), (t_end, 0), (1.6, 0)])
    y += 0.35 * _grains(n, rate, r, 2500, 9500)
    return y


@_sound('crystal', 'Crystal resonance: glass-harmonica chime F#6 A6 D7 on 32nds (soft glassy strike, '
        'singing sustain with slow beating, swells then fades).', tonal=('notes', ['F#6', 'A6', 'D7']),
        hpf=200)
def _crystal(v, r):
    n = _n(1.6)
    y = np.zeros(n)
    for i, (nm, t0, a) in enumerate([('F#6', 0.0, 1.0), ('A6', 0.075, 0.85), ('D7', 0.15, 0.75)]):
        f = hz(nm)
        m = n - _n(t0)
        tm = _t(m)
        d = m / SR
        s = 0.65 * sine(f, m) + 0.35 * sine(f * 1.0024, m) + 0.1 * sine(2 * f, m) * np.exp(-tm / 0.3)
        s *= pw(m, [(0, 0), (0.12, 1), (0.55, 0.75), (d, 0)])
        s += modes(m, [(f, 0.35, 0.05), (2 * f, 0.2, 0.025), (3 * f, 0.12, 0.015)])
        put(y, s, t0, 0.6 * a)
    return y


@_sound('horn', 'Ship / ferry horn "BWAAAMP": low warm reedy D3 + A3 fifth over a D2 sub, breathy '
        'onset scoop, 1.25 s blast and a harbour echo.', tonal=('notes', ['D3', 'A3']), hpf=40)
def _horn(v, r):
    n = _n(1.6)
    d = 1.25
    m = _n(d)
    tm = _t(m)
    bend = 2 ** ((-0.7 * np.exp(-tm / 0.05) - 0.2 * np.clip((tm - (d - 0.15)) / 0.15, 0, 1)) / 12)
    out = np.zeros(m)
    for nm, a in (('D3', 1.0), ('A3', 0.55)):
        s = saw(hz(nm) * bend * (1 + 0.002 * np.sin(TAU * 4.5 * tm + a)), m)
        out += a * (0.6 * lp(s, 900, 2) + 0.4 * reson(s, 520, 4) + 0.25 * reson(s, 1100, 5))
    out = pk(out) + 0.35 * sine(hz('D2') * bend, m) + 0.05 * pk(bp(white(m, r), 400, 3000))
    out *= pw(m, [(0, 0), (0.08, 0.9), (0.3, 1), (d - 0.2, 0.9), (d, 0)])
    y = np.zeros(n)
    put(y, out, 0.0, 1.0)
    return delay(y, 0.32, 0.18) * env(n, 0.001, None, rel=0.02)


@_sound('creak', 'Timber creak: stick-slip wooden groan (rope bridge / ship hull / mill) - a long '
        '"crrreeak" then a short "crk"; variants differ.', variants=2, hpf=120)
def _creak(v, r):
    n = _n(0.8)
    y = np.zeros(n)

    def groan(d, f0, f1):
        m = _n(d)
        tm = _t(m)
        rate = f0 * (f1 / f0) ** (tm / d) * np.clip(1 + 0.15 * _lfo(m, r, 25), 0.5, None)
        ph = np.cumsum(rate / SR)
        imp = lp(np.diff(np.floor(ph), prepend=0.0) * r.uniform(0.5, 1.0, m), 3000)
        s = reson(imp, 380, 5) + 0.8 * reson(imp, 820, 6) + 0.6 * reson(imp, 1650, 8) + 0.3 * reson(imp, 2900, 10)
        return pk(s) * pw(m, [(0, 0), (0.03, 0.8), (d * 0.6, 1), (d, 0)])
    if v == 0:
        put(y, groan(0.45, 45, 140), 0.02, 1.0)
        put(y, groan(0.18, 90, 60), 0.55, 0.7)
    else:
        put(y, groan(0.16, 70, 110), 0.02, 0.6)
        put(y, groan(0.5, 130, 50), 0.24, 1.0)
    return y


@_sound('rain', 'Steady rain: dense soft patter + airy hiss and a low wash (loop-friendly); variant 2 '
        '= drizzle on roofs: sparse tile / tin "tik"s, gutter drips, light hiss.', variants=2, hpf=60)
def _rain(v, r):
    n = _n(2.0)
    if v == 0:
        hiss = pk(bp(pink(n, r), 800, 9000))
        wash = pk(lp(pink(n, r), 400))
        drops = _grains(n, 900, r, 2000, 7000)
        y = (0.55 * hiss + 0.25 * wash + 0.45 * drops) * (1 + 0.08 * _lfo(n, r, 0.8))
    else:
        y = 0.22 * pk(bp(pink(n, r), 1500, 8000))
        for tt in np.sort(r.uniform(0.02, 1.9, 48)):
            m = _n(0.06)
            f = r.uniform(1200, 3600)
            tin = r.random() < 0.3
            tau = r.uniform(0.012, 0.03) if tin else r.uniform(0.003, 0.009)
            put(y, modes(m, [(f, 1.0, tau), (f * 1.73, 0.4, tau * 0.6)]), tt, r.uniform(0.25, 0.7))
        for tt in (0.31, 0.92, 1.53):
            m = _n(0.07)
            tm = _t(m)
            put(y, sine(900 * (1 + 0.7 * tm / 0.07), m) * env(m, 0.002, 0.02, rel=0.01), tt + r.uniform(0, 0.05), 0.5)
    return y * env(n, 0.12, None, rel=0.2)


@_sound('gong', 'Deep gong: soft-mallet strike on a D-tuned gong (D2 hum, A2, D3, F#3, A3, D4 ... '
        'all in D major); high partials bloom after the strike; 2.5 s decay.',
        tonal=('notes', ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4']), hpf=28)
def _gong(v, r):
    n = _n(2.5)
    t = _t(n)
    st = lambda s: 2 ** (s / 12)
    parts = [(1.0, 0.9, 1.8), (st(7), 0.45, 1.2), (2.0, 0.8, 1.4), (st(16), 0.5, 1.0), (st(19), 0.45, 0.9),
             (4.0, 0.35, 0.7), (st(28), 0.25, 0.5), (st(31), 0.2, 0.45), (st(33), 0.15, 0.35),
             (st(38), 0.1, 0.3)]
    f0 = hz('D2')
    y = bell(f0, n, parts, beat=0.9)
    bloom = sum(a * sine(f0 * st(s), n) for s, a in ((28, 0.5), (31, 0.4), (38, 0.3), (43, 0.2)))
    y += 0.35 * bloom * (1 - np.exp(-t / 0.15)) * np.exp(-t / 0.6)
    m = _n(0.08)
    put(y, pk(lp(white(m, r), 1200)) * env(m, 0.001, 0.015, rel=0.02), 0.0, 0.3)
    return y * env(n, 0.003, None, rel=0.5)


@_sound('fountain', 'Fountain: splashing jets falling into a basin - bright bubbly splatter, pulsing '
        'jet hiss and a low pool burble (loop-friendly).', hpf=80)
def _fountain(v, r):
    n = _n(2.0)
    t = _t(n)
    splat = pk(bp(white(n, r), 900, 7000)) * (0.55 + 0.45 * np.abs(_lfo(n, r, 20)))
    jet = pk(hp(white(n, r), 5000)) * (1 + 0.25 * np.sin(TAU * 1.2 * t))
    drops = np.zeros(n)
    for tt in np.sort(r.uniform(0.0, 1.95, 200)):
        m = _n(0.03)
        b = sine(r.uniform(800, 3000) * (1 + 0.4 * _t(m) / 0.03), m)
        put(drops, b * env(m, 0.001, r.uniform(0.004, 0.012), rel=0.004), tt, r.uniform(0.2, 1.0))
    for tt in np.sort(r.uniform(0.0, 1.9, 22)):
        m = _n(0.08)
        b = sine(r.uniform(200, 600) * (1 + 0.6 * _t(m) / 0.08), m)
        put(drops, b * env(m, 0.003, 0.03, rel=0.02), tt, 0.8)
    y = 0.6 * splat + 0.15 * jet + 0.6 * pk(drops)
    return y * env(n, 0.15, None, rel=0.25)


@_sound('waves', 'Sea wave on a beach: swell builds, breaks at 0.8 s (soft crash), foamy fizz washes '
        'up the sand and recedes with bubbly crackle; variants alternate.', hit=0.8, variants=2, hpf=35)
def _waves(v, r):
    n = _n(2.5)
    tb = 0.8 + 0.05 * v
    fc = pw(n, [(0, 250), (tb, 1800), (tb + 0.25, 2600), (1.6, 3200), (2.5, 4500)])
    body = pk(swept(n, fc, r, width=0.8, lo=80, hi=12000, bands=16, src=pink(n, r)))
    body *= pw(n, [(0, 0), (0.5, 0.35), (tb, 1.0), (tb + 0.3, 0.75), (1.6, 0.45), (2.5, 0)])
    low = pk(lp(white(n, r), 150)) * pw(n, [(0, 0), (0.6, 0.8), (tb + 0.1, 1), (1.5, 0.2), (2.5, 0)])
    foam = _grains(n, 1500 * pw(n, [(0, 0), (tb, 0), (tb + 0.2, 1), (2.5, 0.1)]), r, 2500, 9000)
    foam *= pw(n, [(0, 0), (tb, 0), (tb + 0.2, 1), (2.5, 0)])
    return 0.85 * body + 0.4 * low + 0.3 * foam


@_sound('jackpot', 'Slot-machine JACKPOT: chip arpeggio D6 F#6 A6 D7 on 32nds, an electric bell '
        'ringing (D7), then a cascade of coins clinking into the tray (all in key).',
        tonal=('notes', ['D6', 'F#6', 'A6', 'D7']), hpf=150)
def _jackpot(v, r):
    n = _n(1.2)
    y = np.zeros(n)
    for i, nm in enumerate(['D6', 'F#6', 'A6', 'D7']):
        put(y, note(hz(nm), 0.3, 'pulse', duty=0.25, tau=0.12, rel=0.03, sub=0.5), i * 0.075, 0.6)
    f = hz('D7')
    for k in range(14):
        m = _n(0.2)
        put(y, modes(m, [(f, 1.0, 0.12), (hz('A7'), 0.4, 0.06), (hz('F#8'), 0.25, 0.03)]), 0.3 + k / 18,
            0.3 * (1 - k / 20))
    pool = ['A7', 'B7', 'D8', 'E8', 'F#8']
    for tt in np.sort(r.uniform(0.35, 1.1, 30)):
        m = _n(0.12)
        fc = hz(pool[int(r.integers(len(pool)))])
        c = modes(m, [(fc, 1.0, r.uniform(0.03, 0.06)), (2 * fc, 0.3, 0.015)])
        c += 0.2 * pk(hp(white(m, r), 4000)) * env(m, 0.0002, 0.002, rel=0.003)
        put(y, c, tt, r.uniform(0.12, 0.35))
    return y


# ============================================================== names, aliases, mix defaults
STORY_NAMES = ['alarm', 'tap', 'step', 'curtain', 'sizzle', 'crack', 'pop', 'chop', 'click', 'clink',
               'pour', 'gulp', 'fridge', 'door', 'whoosh', 'sparkle', 'powerup', 'clank', 'thud',
               'beep', 'ding', 'typing', 'laser', 'hum', 'blip', 'idea', 'marker', 'cart', 'bag',
               'chime', 'birds', 'gull', 'bell', 'kiwi', 'zzz', 'sweep', 'shimmer', 'coin', 'tick',
               'water', 'microwave', 'boing', 'shutter']
HUD_NAMES = ['banner', 'quest']
BONUS_NAMES = ['growl', 'clap', 'stamp', 'grind', 'puff', 'star']
# EQUESTRIA atlas: landscapes / weather / travel (no creatures)
ATLAS_NAMES = ['wind', 'waterfall', 'whistle', 'chug', 'steam', 'thunder', 'zap', 'splash', 'bubbles',
               'lava', 'rumble', 'avalanche', 'neon', 'firework', 'crystal', 'horn', 'creak', 'rain',
               'gong', 'fountain', 'waves', 'jackpot']
# ATLAS RULE: never cue birds, gull, kiwi, growl - no creatures in this edition (kept for old scenes)
CREATURE_NAMES = ('birds', 'gull', 'kiwi', 'growl')
SFX_NAMES = STORY_NAMES + HUD_NAMES + BONUS_NAMES + ATLAS_NAMES
assert set(SFX_NAMES) == set(_GEN), set(SFX_NAMES) ^ set(_GEN)

ALIASES = {
    'footstep': 'step', 'footsteps': 'step', 'steps': 'step', 'walk': 'step',
    'swoosh': 'whoosh', 'swish': 'whoosh', 'dash': 'whoosh',
    'scanner': 'beep', 'scan': 'beep', 'snore': 'zzz', 'sleep': 'zzz',
    'squeak': 'marker', 'trolley': 'cart', 'keyboard': 'typing', 'type': 'typing', 'keys': 'typing',
    'clocktower': 'bell', 'bong': 'bell', 'highfive': 'clap', 'high-five': 'clap', 'poof': 'puff',
    'smoke': 'puff', 'bird': 'birds', 'tui': 'birds', 'bellbird': 'birds', 'seagull': 'gull',
    'switch': 'click', 'ice': 'clink', 'sip': 'gulp', 'drink': 'gulp', 'eureka': 'idea',
    'lightbulb': 'idea', 'camera': 'shutter', 'photo': 'shutter', 'magic': 'shimmer',
    'twinkle': 'sparkle', 'spring': 'boing', 'bounce': 'boing', 'fry': 'sizzle', 'egg': 'crack',
    'knife': 'chop', 'toast': 'pop', 'beam': 'laser', 'poweron': 'hum', 'buzz': 'hum',
    'reward': 'coin', 'stomach': 'growl', 'hungry': 'growl', 'grinder': 'grind',
    'shootingstar': 'star', 'tock': 'tick', 'clock': 'tick', 'tap_water': 'water', 'sink': 'water',
    'rustle': 'bag', 'paper': 'bag', 'broom': 'sweep', 'metal': 'clank', 'weights': 'clank',
    'drop': 'thud', 'impact': 'stamp', 'newquest': 'banner', 'questcomplete': 'quest',
    'complete': 'quest', 'phone': 'alarm',
    # atlas
    'gust': 'wind', 'breeze': 'wind', 'gale': 'wind', 'cascade': 'waterfall', 'falls': 'waterfall',
    'trainwhistle': 'whistle', 'choochoo': 'whistle', 'toot': 'whistle', 'train': 'chug',
    'locomotive': 'chug', 'steamtrain': 'chug', 'chuff': 'chug', 'hiss': 'steam', 'vent': 'steam',
    'storm': 'thunder', 'lightning': 'thunder', 'electric': 'zap', 'rainbow': 'zap', 'crackle': 'zap',
    'dive': 'splash', 'plunge': 'splash', 'bubble': 'bubbles', 'bloop': 'bubbles',
    'underwater': 'bubbles', 'magma': 'lava', 'eruption': 'lava', 'volcano': 'rumble',
    'quake': 'rumble', 'earthquake': 'rumble', 'snowslide': 'avalanche', 'neonsign': 'neon',
    'flicker': 'neon', 'fireworks': 'firework', 'rocket': 'firework', 'glass': 'crystal',
    'crystals': 'crystal', 'ship': 'horn', 'ferry': 'horn', 'foghorn': 'horn', 'boat': 'horn',
    'bridge': 'creak', 'timber': 'creak', 'rope': 'creak', 'drizzle': 'rain', 'shower': 'rain',
    'temple': 'gong', 'jets': 'fountain', 'surf': 'waves', 'sea': 'waves', 'ocean': 'waves',
    'beach': 'waves', 'wave': 'waves', 'slots': 'jackpot', 'slotmachine': 'jackpot', 'casino': 'jackpot',
}

# per-name mix defaults: (trim dB, reverb send dB, pan -1..1, alternate-pan amount)
# trim is relative to the SFX bus (nominal loudness); alternate pan flips +/- on successive cues.
MIX = {
    'alarm': (0, -18, -0.15, 0), 'tap': (-3, -24, -0.15, 0), 'step': (-2, -22, 0, 0.12),
    'curtain': (-2, -18, 0.1, 0), 'sizzle': (-4, -24, 0.1, 0), 'crack': (-1, -20, 0.05, 0),
    'pop': (-1, -18, 0, 0.12), 'chop': (-2, -20, 0.05, 0.08), 'click': (-3, -22, 0, 0.1),
    'clink': (-2, -15, 0, 0.12), 'pour': (-3, -20, 0, 0), 'gulp': (-2, -22, 0, 0),
    'fridge': (-3, -18, 0, 0), 'door': (-2, -16, 0, 0), 'whoosh': (-2, -16, 0, 0.2),
    'sparkle': (-2, -12, 0, 0.18), 'powerup': (0, -14, 0, 0), 'clank': (0, -14, 0, 0.15),
    'thud': (-1, -18, 0, 0), 'beep': (-4, -18, 0.05, 0), 'ding': (-1, -13, 0, 0),
    'typing': (-3, -22, 0, 0.08), 'laser': (-1, -14, 0, 0), 'hum': (-4, -18, 0, 0),
    'blip': (-3, -18, 0, 0.1), 'idea': (0, -12, 0, 0), 'marker': (-3, -20, 0, 0.1),
    'cart': (-4, -22, 0, 0), 'bag': (-4, -20, 0, 0.1), 'chime': (-1, -12, 0, 0),
    'birds': (-5, -12, 0, 0.35), 'gull': (-5, -12, 0, 0.3), 'bell': (0, -10, 0, 0),
    'kiwi': (-3, -10, 0.2, 0), 'zzz': (-5, -18, 0, 0), 'sweep': (-4, -20, 0, 0.12),
    'shimmer': (-2, -11, 0, 0), 'coin': (-2, -15, 0, 0.1), 'tick': (-4, -20, 0, 0.06),
    'water': (-4, -20, 0, 0), 'microwave': (-5, -22, 0, 0), 'boing': (-1, -16, 0, 0),
    'shutter': (-3, -20, 0, 0), 'banner': (-2, -14, -0.3, 0), 'quest': (-1, -12, 0, 0),
    'growl': (-2, -22, 0, 0), 'clap': (-2, -14, 0, 0), 'stamp': (0, -12, 0, 0),
    'grind': (-4, -20, 0, 0), 'puff': (-2, -14, 0, 0), 'star': (-2, -10, 0, 0),
    # atlas
    'wind': (-4, -16, 0, 0.25), 'waterfall': (-4, -18, 0, 0), 'whistle': (-2, -12, 0, 0),
    'chug': (-3, -20, 0, 0), 'steam': (-4, -18, 0, 0.15), 'thunder': (-1, -10, 0, 0.3),
    'zap': (-2, -14, 0, 0.2), 'splash': (-1, -16, 0, 0.1), 'bubbles': (-3, -16, 0, 0.15),
    'lava': (-2, -18, 0, 0.12), 'rumble': (-2, -18, 0, 0), 'avalanche': (-1, -14, 0, 0),
    'neon': (-5, -20, 0, 0.2), 'firework': (-1, -10, 0, 0.3), 'crystal': (-2, -10, 0, 0.15),
    'horn': (-1, -10, 0, 0), 'creak': (-4, -20, 0, 0.15), 'rain': (-5, -18, 0, 0),
    'gong': (0, -10, 0, 0), 'fountain': (-5, -18, 0, 0), 'waves': (-3, -16, 0, 0.25),
    'jackpot': (-2, -14, 0, 0),
}
# suggested mono-compatible stereo width for ambiences (mix.py WIDTH can adopt these)
WIDTH_HINT = {'wind': 0.6, 'waterfall': 0.5, 'rain': 0.7, 'fountain': 0.4, 'waves': 0.6,
              'avalanche': 0.4, 'thunder': 0.4, 'firework': 0.35, 'crystal': 0.35, 'steam': 0.25}
assert set(MIX) == set(SFX_NAMES)


def resolve(name):
    """Canonical SFX name for `name` (case/space-insensitive, aliases allowed). KeyError if unknown."""
    key = str(name).strip().lower().replace(' ', '')
    if key in _GEN:
        return key
    if key in ALIASES:
        return ALIASES[key]
    sug = difflib.get_close_matches(key, SFX_NAMES + list(ALIASES), n=3, cutoff=0.6)
    raise KeyError(f'unknown sfx {name!r}' + (f' (did you mean {", ".join(sug)}?)' if sug else ''))


# ============================================================== loudness / finishing
_KW = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241,
                 0.73248077421585],
                [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])


def kweight(x):
    """ITU-R BS.1770 K-weighting filter (48 kHz coefficients)."""
    return signal.sosfilt(_KW, x)


def loudness_peak(x, win=0.1):
    """Loudest `win`-second K-weighted level (LUFS-like, dB)."""
    y = kweight(np.asarray(x, np.float64))
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    w = _n(win)
    ms = c[-1] / w if len(y) <= w else np.max(c[w:] - c[:-w]) / w
    return -0.691 + 10 * np.log10(ms + 1e-20)


def _finish(x, hpf):
    x = hp(np.asarray(x, np.float64), hpf, 2)
    x = _fade(x, 0.0003, 0.008)
    lk = loudness_peak(x)
    g = 10 ** ((NOMINAL_LK - lk) / 20)
    p = np.max(np.abs(x))
    if p > 0:
        g = min(g, PEAK / p)
    return (x * g).astype(np.float32)


@lru_cache(maxsize=None)
def _make(name, v):
    r = np.random.default_rng(zlib.crc32(f'{name}:{v}'.encode()))
    return _finish(_GEN[name](v, r), _HPF[name])


def make(name, variant=0):
    """Mono float32 sample array (48 kHz) for effect `name` (aliases ok). Cached; returns a copy.
    `variant` is taken modulo VARIANTS[name] (most sounds have a single variant)."""
    key = resolve(name)
    return _make(key, int(variant) % VARIANTS[key]).copy()


def duration(name):
    return len(_make(resolve(name), 0)) / SR


def hit(name):
    """Seconds from the start of the sound to its main transient (cue = frame - hit*30)."""
    return _HIT[resolve(name)]


# pitched parts of otherwise noisy / inharmonic effects, per variant (used by the mixer's key-fit only;
# tools/sfx_check.py does not pitch-track these)
EXTRA_NOTES = {
    'clank': [['A4', 'B4'], ['B4', 'D5'], ['F#4', 'A4']],
    'fridge': [['G2', 'E7', 'A7']],
    'growl': [['A2', 'F#2']],
    'wind': [['A5', 'B5', 'D6'], ['F#5', 'A5', 'D6']],
    'zap': [['D7', 'F#7', 'A7']],
    'neon': [['A2', 'A3', 'E4']],
}


def notes(name, variant=0):
    """Pitched notes an effect plays (e.g. ['D6', 'F#6', 'A6']); [] for unpitched effects.
    For effects whose variants are single notes (pop, clink) only that variant's note."""
    key = resolve(name)
    if key in EXTRA_NOTES:
        lst = EXTRA_NOTES[key]
        return list(lst[int(variant) % len(lst)])
    if key not in TONAL:
        return []
    ns = list(TONAL[key][1])
    nv = VARIANTS[key]
    if nv > 1 and len(ns) == nv:
        return [ns[int(variant) % nv]]
    return ns


def pitch_classes(name, variant=0):
    """Set of pitch classes (C=0 .. B=11) of notes(name, variant)."""
    return {_PC[n[:-1]] for n in notes(name, variant)}


def describe(name):
    key = resolve(name)
    extra = f'; main hit at {_HIT[key]:.2f} s' if _HIT[key] else ''
    var = f'; {VARIANTS[key]} variants' if VARIANTS[key] > 1 else ''
    return f'{key}: {_DESC[key]} ({duration(key):.2f} s{extra}{var})'


def centroid(x):
    x = np.asarray(x, np.float64)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float((X * f).sum() / (X.sum() + 1e-20))


def stats(name, variant=0):
    """Numbers for docs / checks: duration, peak dBFS, loudness, RMS dBFS, centroid Hz."""
    x = make(name, variant).astype(np.float64)
    return dict(name=resolve(name), dur=len(x) / SR, peak_db=20 * np.log10(np.max(np.abs(x)) + 1e-12),
                lk=loudness_peak(x), rms_db=10 * np.log10(np.mean(x * x) + 1e-20),
                centroid=centroid(x), hit=hit(name), variants=VARIANTS[resolve(name)])


def write_wav(path, x, sr=SR):
    """Write mono or (n, ch) float audio as 16-bit PCM."""
    from scipy.io import wavfile
    y = np.clip(np.round(np.asarray(x, np.float64) * 32767), -32767, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wavfile.write(path, sr, y)


# ============================================================== CLI
def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Render the SFX preview and print a table.')
    ap.add_argument('names', nargs='*', help='only these effects (default: all)')
    ap.add_argument('--out', default=os.path.join(_HERE, 'out', 'sfx_preview.wav'))
    ap.add_argument('--gap', type=float, default=0.4)
    ap.add_argument('--list', action='store_true', help='just list names and descriptions')
    a = ap.parse_args(argv)
    names = [resolve(n) for n in a.names] or SFX_NAMES
    if a.list:
        for nm in names:
            print(describe(nm))
        return
    gap = np.zeros(_n(a.gap), np.float32)
    parts = [gap]
    print(f'{"name":10s} {"start":>7s} {"dur s":>6s} {"peak":>6s} {"LK100":>6s} {"centroid":>8s}  description')
    pos = len(gap)
    for nm in names:
        x = make(nm)
        s = stats(nm)
        print(f'{nm:10s} {pos / SR:7.2f} {s["dur"]:6.2f} {s["peak_db"]:6.1f} {s["lk"]:6.1f} '
              f'{s["centroid"]:8.0f}  {_DESC[nm][:70]}')
        parts += [x, gap]
        pos += len(x) + len(gap)
    y = np.concatenate(parts)
    write_wav(a.out, y)
    print(f'wrote {a.out}  ({len(y) / SR:.1f} s, {len(names)} effects, 48 kHz mono 16-bit)')


if __name__ == '__main__':
    _main()
