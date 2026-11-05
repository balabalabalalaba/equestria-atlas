"""ONE GOOD DAY — chiptune / lo-fi synthesiser and mixing DSP (numpy + scipy only).

Everything is vectorised numpy; one note is rendered at a time into a float array and
added into a bus by the arranger (`audio/compose.py`).

Grid facts: 48 kHz, 100 BPM -> 28 800 samples per beat, 115 200 per bar (exact integers),
so a note on the beat grid always starts on an exact sample.

Main entry points
-----------------
    render_note(inst, midi, dur_s, vel, seed=0, **overrides) -> np.ndarray  (n,) mono or (2, n) stereo
    drum(name, vel, seed=0, **p)                             -> (n,) mono
    fx(name, dur_s, vel, seed=0, **p)                        -> (n,) or (2, n)
    pingpong_echo(mono, delay_s, feedback, lp, hp)           -> (2, n)
    make_reverb_ir(...) / reverb(stereo, ir)                 -> (2, n)
    duck_curve(n, hit_samples, depth, attack_s, release_s)   -> (n,) gain
    lufs(x), true_peak_db(x), limiter(x, ceiling_db), to_int16(x), write_wav(path, x), read_wav(path)

Instrument presets live in INSTRUMENTS (name -> params); `render_note` merges overrides.
All envelopes have >= 2 ms attack and >= 3 ms release and end exactly at 0 (no clicks).
"""
import math
import wave
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
try:                          # tempo comes from the engine so picture and music never drift
    from engine.core import BPM
except Exception:  # pragma: no cover
    BPM = 100
SPB = SR * 60 // BPM          # 28 800 samples per beat
SPBAR = SPB * 4               # 115 200 samples per bar
MIN_ATTACK = 0.002
MIN_RELEASE = 0.003


def mtof(m):
    """MIDI note number -> Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def amp_of(vel):
    """Velocity 0..1 -> linear amplitude (gentle perceptual curve)."""
    return float(max(0.0, vel)) ** 1.5


# ======================================================================= filters
@lru_cache(maxsize=512)
def _butter(kind, fc, order):
    fc = float(min(max(fc, 5.0), SR * 0.45))
    return signal.butter(order, fc, btype=kind, fs=SR, output='sos')


@lru_cache(maxsize=512)
def _butter_bp(lo, hi, order):
    lo = float(max(lo, 5.0))
    hi = float(min(hi, SR * 0.45))
    return signal.butter(order, [lo, hi], btype='bandpass', fs=SR, output='sos')


def lowpass(x, fc, order=2):
    return signal.sosfilt(_butter('lowpass', round(fc, 1), order), x, axis=-1)


def highpass(x, fc, order=2):
    return signal.sosfilt(_butter('highpass', round(fc, 1), order), x, axis=-1)


def bandpass(x, lo, hi, order=2):
    return signal.sosfilt(_butter_bp(round(lo, 1), round(hi, 1), order), x, axis=-1)


def onepole_lp(x, fc):
    a = math.exp(-2 * math.pi * fc / SR)
    return signal.lfilter([1 - a], [1, -a], x, axis=-1)


def _rbj(kind, f0, gain_db=0.0, q=0.707):
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    if kind == 'peak':
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == 'lowshelf':
        sa = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == 'highshelf':
        sa = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return b, a


def eq(x, kind, f0, gain_db, q=0.707):
    """RBJ peaking / shelving EQ ('peak', 'lowshelf', 'highshelf')."""
    b, a = _rbj(kind, f0, gain_db, q)
    return signal.lfilter(b, a, x, axis=-1)


# ======================================================================= oscillators
def _phase(freq, n, phase0=0.0):
    """Phase in cycles [0,1) and per-sample increment for a (possibly time-varying) frequency."""
    if np.isscalar(freq):
        dt = np.full(n, float(freq) / SR)
    else:
        dt = np.asarray(freq, dtype=np.float64)[:n] / SR
    ph = np.cumsum(dt)
    ph -= dt[0] - phase0
    ph %= 1.0
    return ph, dt


def _blep(t, dt):
    """PolyBLEP residual for a unit step at phase 0 (t, dt arrays)."""
    y = np.zeros_like(t)
    m = t < dt
    if m.any():
        x = t[m] / dt[m]
        y[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    if m.any():
        x = (t[m] - 1.0) / dt[m]
        y[m] = x * x + x + x + 1.0
    return y


def osc(wave_, freq, n, duty=0.25, phase0=0.0):
    """Band-limited oscillator. wave_: pulse | saw | tri | nestri (4-bit stepped) | sine.

    Pulse waves are DC-free (the duty-cycle offset is removed), amplitude ~ +-1.
    `duty` may be a scalar or an array (PWM).
    """
    ph, dt = _phase(freq, n, phase0)
    if wave_ == 'pulse':
        d = duty
        y = np.where(ph < d, 1.0, -1.0)
        y += _blep(ph, dt)
        y -= _blep((ph - d) % 1.0, dt)
        y -= 2.0 * np.asarray(d) - 1.0
        return y
    if wave_ == 'saw':
        return 2.0 * ph - 1.0 - _blep(ph, dt)
    if wave_ in ('tri', 'nestri'):
        p2 = (ph + 0.75) % 1.0
        y = 4.0 * np.abs(p2 - 0.5) - 1.0
        if wave_ == 'nestri':
            y = np.round((y + 1.0) * 7.5) / 7.5 - 1.0
        return y
    if wave_ == 'sine':
        return np.sin(2 * np.pi * ph)
    raise ValueError(wave_)


@lru_cache(maxsize=4)
def _lfsr_seq(short):
    """NES-style 15-bit LFSR noise sequence as +-1 floats (long: 32767 steps, short: 93)."""
    reg = 1
    tap = 6 if short else 1
    out = []
    seen = {}
    for i in range(32767):
        bit = (reg ^ (reg >> tap)) & 1
        reg = (reg >> 1) | (bit << 14)
        out.append(1.0 if reg & 1 else -1.0)
        if short:
            if reg in seen:
                out = out[seen[reg]:i + 1]
                break
            seen[reg] = i + 1
    return np.array(out)


def lfsr_noise(n, rate=SR, short=False, seed=0):
    """Chip noise channel: LFSR sequence clocked at `rate` Hz (sample-and-hold)."""
    seq = _lfsr_seq(bool(short))
    off = int(seed * 7919) % len(seq)
    idx = (np.floor(np.arange(n) * (rate / SR)).astype(np.int64) + off) % len(seq)
    return seq[idx]


def white(n, seed):
    return np.random.default_rng(int(seed) & 0xffffffff).standard_normal(n)


# ======================================================================= envelopes
def _ns(t):
    return max(1, int(round(t * SR)))


def adsr(gate_s, a=0.005, d=0.2, s=0.7, r=0.08, attack_curve='cos'):
    """ADSR envelope (gate + release), starts and ends at exactly 0.

    a: attack seconds (>= 2 ms), d: decay *time constant*, s: sustain level, r: release seconds (>= 3 ms).
    """
    a = max(a, MIN_ATTACK)
    r = max(r, MIN_RELEASE)
    ng = max(_ns(gate_s), 2)
    na, nr = _ns(a), _ns(r)
    env = np.empty(ng)
    k = min(na, ng)
    x = np.arange(k) / na
    env[:k] = 0.5 - 0.5 * np.cos(np.pi * x) if attack_curve == 'cos' else x
    if ng > na:
        td = np.arange(ng - na) / SR
        env[na:] = s + (1.0 - s) * np.exp(-td / max(d, 1e-4))
    lvl = env[-1]
    xr = np.arange(1, nr + 1) / nr
    rel = lvl * (1.0 - xr) ** 2
    return np.concatenate([env, rel])


def perc_env(n, a=0.001, tau=0.1, hold=0.0, fade=0.004):
    """Percussive envelope of n samples: short attack, optional hold, exponential decay, end fade to 0."""
    t = np.arange(n) / SR
    env = np.exp(-np.maximum(t - a - hold, 0.0) / tau)
    na = max(_ns(a), 2)
    env[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    nf = min(n, _ns(fade))
    env[n - nf:] *= np.linspace(1.0, 0.0, nf) ** 2
    return env


def _fade_ends(x, a=0.002, r=0.004):
    """Force a click-free start/end on an arbitrary buffer (in place, returns x)."""
    n = x.shape[-1]
    na, nr = min(n // 2, _ns(a)), min(n // 2, _ns(r))
    if na > 0:
        x[..., :na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    if nr > 0:
        x[..., n - nr:] *= (1.0 - np.arange(1, nr + 1) / nr) ** 2
    return x


# ======================================================================= pitch
def pitch_curve(f0, n, vib=0.0, vib_rate=5.5, vib_delay=0.18, vib_fade=0.25, glide_from=None,
                glide=0.04, bend=0.0, bend_time=None, bend_curve=2.0, wow=0.0, wow_rate=0.7, seed=0):
    """Instantaneous frequency array: optional portamento from `glide_from` Hz, delayed vibrato
    (cents), pitch bend (semitones reached at `bend_time`), lo-fi tape wow (cents)."""
    t = np.arange(n) / SR
    cents = np.zeros(n)
    if vib:
        ramp = np.clip((t - vib_delay) / max(vib_fade, 1e-3), 0.0, 1.0)
        cents += vib * ramp * np.sin(2 * np.pi * vib_rate * t)
    if wow:
        ph = (seed % 97) / 97.0 * 2 * np.pi
        cents += wow * np.sin(2 * np.pi * wow_rate * t + ph) + 0.35 * wow * np.sin(2 * np.pi * 5.3 * wow_rate * t)
    if bend:
        bt = bend_time if bend_time else n / SR
        x = np.clip(t / bt, 0.0, 1.0)
        cents += 100.0 * bend * (1.0 - (1.0 - x) ** bend_curve)
    if glide_from and glide > 0:
        g = 1200.0 * math.log2(glide_from / f0)
        x = np.clip(t / glide, 0.0, 1.0)
        cents += g * (1.0 - (3 * x * x - 2 * x * x * x))
    return f0 * 2.0 ** (cents / 1200.0)


# ======================================================================= instruments
INSTRUMENTS = {
    # --- leads / melodic chip voices -------------------------------------------------
    'lead':     dict(kind='chip', wave='pulse', duty=0.25, a=0.006, d=0.35, s=0.72, r=0.07, vib=16,
                     vib_delay=0.16, glide=0.035, lp=7200, gain=0.30, fenv=0.35, fenv_lo=2600, fenv_tau=0.12),
    'lead50':   dict(kind='chip', wave='pulse', duty=0.5, a=0.006, d=0.35, s=0.7, r=0.07, vib=14,
                     vib_delay=0.18, glide=0.03, lp=5200, gain=0.27, fenv=0.3, fenv_lo=1800, fenv_tau=0.1),
    'thin':     dict(kind='chip', wave='pulse', duty=0.125, a=0.004, d=0.25, s=0.6, r=0.06, vib=12,
                     vib_delay=0.2, glide=0.0, lp=6500, gain=0.24),
    'mellow':   dict(kind='chip', wave='tri', wave2='pulse', duty2=0.5, mix2=0.22, lp2=1400, a=0.012, d=0.5,
                     s=0.8, r=0.12, vib=22, vib_rate=5.0, vib_delay=0.2, glide=0.05, gain=0.42, wow=4),
    'ocarina':  dict(kind='chip', wave='tri', wave2='sine', mix2=0.45, oct2=0, a=0.018, d=0.6, s=0.85,
                     r=0.14, vib=18, vib_rate=4.8, vib_delay=0.22, glide=0.06, gain=0.55, wow=3),
    'warm':     dict(kind='chip', wave='tri', wave2='pulse', duty2=0.5, mix2=0.34, lp2=2600, a=0.016, d=0.5,
                     s=0.8, r=0.15, vib=20, vib_rate=5.0, vib_delay=0.22, glide=0.05, gain=0.36, wow=3),
    'arp':      dict(kind='chip', wave='pulse', duty=0.125, a=0.002, d=0.07, s=0.25, r=0.03, lp=7000,
                     gain=0.17, fenv=0.5, fenv_lo=1500, fenv_tau=0.05),
    'arp25':    dict(kind='chip', wave='pulse', duty=0.25, a=0.002, d=0.06, s=0.2, r=0.03, lp=6000,
                     gain=0.15, fenv=0.5, fenv_lo=1300, fenv_tau=0.04),
    'triarp':   dict(kind='chip', wave='tri', a=0.004, d=0.45, s=0.35, r=0.12, gain=0.33),
    'blip':     dict(kind='chip', wave='pulse', duty=0.125, a=0.002, d=0.04, s=0.0, r=0.02, lp=8000, gain=0.16),
    'pluck':    dict(kind='chip', wave='pulse', duty=0.25, a=0.002, d=0.09, s=0.0, r=0.03, lp=5000,
                     gain=0.16, fenv=0.7, fenv_lo=900, fenv_tau=0.035),
    'power':    dict(kind='chip', wave='pulse', duty=0.5, wave2='saw', mix2=0.35, a=0.003, d=0.12, s=0.35,
                     r=0.04, lp=3800, gain=0.16, drive=1.8, fenv=0.6, fenv_lo=700, fenv_tau=0.06),
    'brass':    dict(kind='chip', wave='pulse', duty=0.25, wave2='pulse', duty2=0.5, mix2=0.5, det2=7,
                     a=0.012, d=0.3, s=0.75, r=0.09, vib=14, vib_delay=0.2, lp=6000, gain=0.25, fenv=0.8,
                     fenv_lo=900, fenv_tau=0.09),
    # --- bass ----------------------------------------------------------------------
    'bass':     dict(kind='chip', wave='nestri', wave2='pulse', duty2=0.5, mix2=0.18, lp2=600, a=0.004,
                     d=0.25, s=0.75, r=0.04, lp=3500, hp=36, gain=0.62),
    'bassoct':  dict(kind='chip', wave='nestri', wave2='pulse', duty2=0.25, mix2=0.28, lp2=1100, a=0.003,
                     d=0.12, s=0.55, r=0.03, lp=3800, hp=36, gain=0.6, drive=1.4),
    'upright':  dict(kind='chip', wave='tri', wave2='sine', mix2=0.5, a=0.006, d=0.32, s=0.25, r=0.08,
                     lp=1600, hp=32, gain=0.75, bend=-0.08, bend_time=0.06),
    'pulsebass': dict(kind='chip', wave='pulse', duty=0.5, wave2='nestri', mix2=0.6, a=0.003, d=0.11, s=0.4,
                      r=0.04, lp=1000, hp=36, gain=0.5, fenv=0.6, fenv_lo=380, fenv_tau=0.05),
    # --- harmony -------------------------------------------------------------------
    'pad':      dict(kind='pad', waves=('saw', 'pulse', 'saw'), detune=(-9, 2, 11), a=0.6, d=1.5, s=0.85,
                     r=0.9, lp=1900, gain=0.085),
    'padsoft':  dict(kind='pad', waves=('saw', 'tri', 'saw'), detune=(-7, 0, 8), a=1.0, d=2.0, s=0.9,
                     r=1.4, lp=1100, gain=0.10),
    'padbright': dict(kind='pad', waves=('saw', 'pulse', 'saw', 'saw'), detune=(-12, -3, 5, 13), a=0.25,
                      d=1.0, s=0.85, r=0.7, lp=3200, gain=0.07),
    'epiano':   dict(kind='epiano', I0=1.7, Itau=0.45, Isus=0.12, tau=1.5, r=0.18, trem=0.22, trem_rate=3.6,
                     gain=0.20, wow=4),
    # --- bells / music box -----------------------------------------------------------
    'mbox':     dict(kind='mbox', tau=1.1, gain=0.30, pulse_mix=0.16),
    'glock':    dict(kind='glock', tau=0.7, gain=0.22),
    'bell':     dict(kind='bell', tau=3.2, gain=0.30),
}


def _vel_gain(p, vel):
    return p.get('gain', 0.3) * amp_of(vel)


def inst_chip(f0, dur, vel, seed, p):
    env = adsr(dur, p.get('a', 0.005), p.get('d', 0.2), p.get('s', 0.7), p.get('r', 0.06))
    n = len(env)
    freq = pitch_curve(f0, n, vib=p.get('vib', 0.0) if dur > 0.22 else 0.0, vib_rate=p.get('vib_rate', 5.5),
                       vib_delay=p.get('vib_delay', 0.18), glide_from=p.get('glide_from'),
                       glide=p.get('glide', 0.0), bend=p.get('bend', 0.0), bend_time=p.get('bend_time'),
                       wow=p.get('wow', 0.0), seed=seed)
    duty = p.get('duty', 0.25)
    if p.get('pwm'):
        t = np.arange(n) / SR
        duty = duty + p['pwm'] * np.sin(2 * np.pi * p.get('pwm_rate', 0.8) * t)
    y = osc(p.get('wave', 'pulse'), freq, n, duty)
    if p.get('wave2'):
        f2 = freq * 2.0 ** (p.get('oct2', 0) + p.get('det2', 0) / 1200.0)
        y2 = osc(p['wave2'], f2, n, p.get('duty2', 0.5))
        if p.get('lp2'):
            y2 = lowpass(y2, p['lp2'])
        m = p.get('mix2', 0.3)
        y = (1 - m) * y + m * y2
    if p.get('fenv'):
        dark = lowpass(y, p.get('fenv_lo', 1500))
        t = np.arange(n) / SR
        b = (1 - p['fenv']) + p['fenv'] * np.exp(-t / p.get('fenv_tau', 0.08))
        y = dark + (y - dark) * b
    if p.get('lp'):
        y = lowpass(y, p['lp'])
    if p.get('hp'):
        y = highpass(y, p['hp'])
    if p.get('drive'):
        k = p['drive']
        y = np.tanh(k * y) / math.tanh(k)
    y *= env * _vel_gain(p, vel)
    return y


def inst_pad(f0, dur, vel, seed, p):
    env = adsr(dur, p.get('a', 0.6), p.get('d', 1.5), p.get('s', 0.85), p.get('r', 0.9))
    n = len(env)
    rg = np.random.default_rng(seed)
    waves, det = p.get('waves', ('saw', 'saw', 'saw')), p.get('detune', (-8, 0, 8))
    out = np.zeros((2, n))
    t = np.arange(n) / SR
    for ch in range(2):
        acc = np.zeros(n)
        for i, (w, c) in enumerate(zip(waves, det)):
            cc = c * (1 if ch == 0 else -1) + (2.5 if ch else -2.5)
            drift = 3.0 * np.sin(2 * np.pi * (0.13 + 0.07 * i) * t + rg.uniform(0, 6.28))
            fr = f0 * 2.0 ** ((cc + drift) / 1200.0)
            acc += osc(w, fr, n, 0.5 if w == 'pulse' else 0.25, phase0=rg.uniform())
        out[ch] = acc / math.sqrt(len(waves))
    out = lowpass(out, p.get('lp', 1800))
    out = highpass(out, p.get('hp', 90), order=1)
    out *= env * _vel_gain(p, vel)
    return out


def inst_epiano(f0, dur, vel, seed, p):
    tau = p.get('tau', 1.5) * (261.6 / f0) ** 0.35
    gate = dur
    n_g = _ns(gate)
    nr = _ns(max(p.get('r', 0.18), MIN_RELEASE))
    n = n_g + nr
    t = np.arange(n) / SR
    freq = pitch_curve(f0, n, wow=p.get('wow', 0.0), seed=seed)
    ph = 2 * np.pi * np.cumsum(freq) / SR
    I = (p.get('I0', 1.7) * (0.5 + 0.7 * vel)) * np.exp(-t / p.get('Itau', 0.45)) + p.get('Isus', 0.12)
    y = np.sin(ph + I * np.sin(ph))
    y += 0.12 * vel * np.sin(7.0 * ph) * np.exp(-t / 0.025)          # tine
    y += 0.25 * np.sin(2 * ph) * np.exp(-t / (tau * 0.4))            # 2nd harmonic body
    env = np.exp(-t / tau)
    na = _ns(0.003)
    env[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    env[n_g:] *= (1 - np.arange(1, nr + 1) / nr) ** 2
    y *= env * _vel_gain(p, vel)
    tr = p.get('trem', 0.0)
    if tr:
        lfo = np.sin(2 * np.pi * p.get('trem_rate', 3.6) * t + (seed % 13) * 0.5)
        return np.stack([y * (1 + tr * lfo), y * (1 - tr * lfo)])
    return y


def inst_mbox(f0, dur, vel, seed, p):
    tau = p.get('tau', 1.1) * min(1.6, (523.0 / f0) ** 0.45)
    ring = max(dur, min(3.2, tau * 3.2))
    n = _ns(ring)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f0 * t
    y = np.sin(ph) * np.exp(-t / tau)
    y += 0.30 * np.sin(2 * ph) * np.exp(-t / (tau * 0.45))
    y += 0.07 * np.sin(5.33 * ph) * np.exp(-t / 0.05)
    pm = p.get('pulse_mix', 0.16)
    if pm:
        pl = lowpass(osc('pulse', f0, n, 0.125), 3200)
        y += pm * pl * np.exp(-t / (tau * 0.35))
    y = _fade_ends(y, 0.002, 0.03)
    return y * _vel_gain(p, vel)


def inst_glock(f0, dur, vel, seed, p):
    tau = p.get('tau', 0.7) * min(1.5, (880.0 / f0) ** 0.4)
    n = _ns(max(dur, tau * 4.0))
    t = np.arange(n) / SR
    ph = 2 * np.pi * f0 * t
    y = np.sin(ph) * np.exp(-t / tau)
    y += 0.22 * np.sin(2.76 * ph) * np.exp(-t / (tau * 0.3))
    y += 0.08 * np.sin(5.40 * ph) * np.exp(-t / (tau * 0.12))
    y += 0.03 * np.sin(8.93 * ph) * np.exp(-t / 0.02)
    y = _fade_ends(y, 0.002, 0.02)
    return y * _vel_gain(p, vel)


_BELL = [  # (ratio to prime, amp, decay factor)
    (0.5, 0.55, 1.6), (1.0, 0.75, 1.0), (1.183, 0.45, 0.75), (1.506, 0.30, 0.6), (2.0, 0.65, 0.65),
    (2.514, 0.18, 0.4), (2.662, 0.20, 0.38), (3.011, 0.14, 0.3), (4.166, 0.08, 0.2), (5.433, 0.05, 0.12)]


def inst_bell(f0, dur, vel, seed, p):
    """Church-bell additive model; f0 is the PRIME (the hum sounds an octave below)."""
    tau = p.get('tau', 3.2)
    n = _ns(max(dur, tau * 2.2))
    t = np.arange(n) / SR
    rg = np.random.default_rng(seed)
    y = np.zeros(n)
    for r_, a_, k in _BELL:
        beat_ = 1.0 + rg.uniform(-0.0015, 0.0015)
        y += a_ * np.sin(2 * np.pi * f0 * r_ * beat_ * t + rg.uniform(0, 6.28)) * np.exp(-t / (tau * k))
    strike = highpass(white(n, seed), 1500) * np.exp(-t / 0.006) * 0.25
    y += strike
    y = _fade_ends(y, 0.002, 0.05)
    return y * _vel_gain(p, vel)


_KINDS = {'chip': inst_chip, 'pad': inst_pad, 'epiano': inst_epiano, 'mbox': inst_mbox,
          'glock': inst_glock, 'bell': inst_bell}


def render_note(inst, midi, dur_s, vel, seed=0, **overrides):
    """Render one note. `inst` is a key of INSTRUMENTS (or a full params dict).
    Overrides (e.g. glide_from=Hz, lp=..., duty=...) are merged into the preset.
    Returns mono (n,) or stereo (2, n) float64; the array starts at the note onset and includes the tail."""
    p = dict(INSTRUMENTS[inst]) if isinstance(inst, str) else dict(inst)
    p.update({k: v for k, v in overrides.items() if v is not None})
    f0 = mtof(midi)
    return _KINDS[p['kind']](f0, max(float(dur_s), 0.01), float(vel), int(seed), p)


# ======================================================================= drums
_HAT_FREQS = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _metal(n, scale=1.0):
    t = np.arange(n) / SR
    return sum(np.sign(np.sin(2 * np.pi * f * scale * 1.7 * t + i)) for i, f in enumerate(_HAT_FREQS)) / 6.0


def drum(name, vel=0.8, seed=0, **p):
    """Synthesised drum hit -> mono float array starting at the hit.

    kick, kick_soft, kick_lofi, snare, snare_soft, snare_lofi, brush, clap, rim, hat, hat_soft, ohat,
    shaker, crash, crash_soft, ride, tom_hi, tom_mid, tom_lo, boom, tick
    """
    g = amp_of(vel)
    if name.startswith('kick'):
        soft = name != 'kick'
        dur = 0.42 if soft else 0.5
        n = _ns(dur)
        t = np.arange(n) / SR
        f_hi, f_lo = (120, 58) if soft else (160, 57)
        if name == 'kick_lofi':
            f_hi, f_lo = 110, 60
        f = f_lo + (f_hi - f_lo) * np.exp(-t / (0.032 if soft else 0.04))
        body = np.sin(2 * np.pi * np.cumsum(f) / SR)
        body *= perc_env(n, 0.0015, 0.15 if soft else 0.165, hold=0.01)
        click = highpass(white(n, seed), 1800) * np.exp(-t / 0.0025) * (0.08 if soft else 0.22)
        y = body + click
        if name == 'kick':
            y = np.tanh(1.6 * y) / math.tanh(1.6)
        if soft:
            y = lowpass(y, 2500)
        return _fade_ends(y * g * (0.72 if soft else 0.85), 0.001, 0.01)
    if name.startswith('snare') or name == 'brush':
        n = _ns(0.4)
        t = np.arange(n) / SR
        if name == 'brush':
            nz = bandpass(white(n, seed), 1500, 9000) * perc_env(n, 0.018, 0.13)
            return _fade_ends(nz * g * 0.22, 0.002, 0.01)
        tau_n = {'snare': 0.11, 'snare_soft': 0.09, 'snare_lofi': 0.13}[name]
        f = 190 * (1 + 0.25 * np.exp(-t / 0.01))
        tone = (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(f * 1.74) / SR))
        tone *= perc_env(n, 0.001, 0.05) * 0.5
        nz = 0.6 * white(n, seed) + 0.4 * lfsr_noise(n, 24000, seed=seed)
        nz = bandpass(nz, 900, 9000) * perc_env(n, 0.001, tau_n)
        y = tone + nz * 0.9
        if name == 'snare_lofi':
            y = lowpass(y, 4200)
        if name == 'snare_soft':
            y = lowpass(y, 6500)
        lvl = {'snare': 0.5, 'snare_soft': 0.38, 'snare_lofi': 0.42}[name]
        return _fade_ends(y * g * lvl, 0.001, 0.01)
    if name == 'clap':
        n = _ns(0.35)
        t = np.arange(n) / SR
        nz = bandpass(white(n, seed), 800, 4200)
        env = np.zeros(n)

        def burst(off, tau):     # 0.6 ms raised-cosine onset: snappy but no discontinuity
            tt = t - off
            on = np.clip(tt / 0.0006, 0.0, 1.0)
            return (0.5 - 0.5 * np.cos(np.pi * on)) * np.exp(-np.maximum(tt, 0) / tau)
        for k, off in enumerate((0.0, 0.009, 0.018, 0.026)):
            env += burst(off, 0.006) * (0.8 if k < 3 else 1.0)
        env += burst(0.026, 0.085) * 0.55
        return _fade_ends(nz * env * g * 0.42, 0.001, 0.01)
    if name == 'rim':
        n = _ns(0.12)
        t = np.arange(n) / SR
        y = np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.012) + 0.6 * bandpass(white(n, seed), 2000, 7000) * np.exp(-t / 0.006)
        return _fade_ends(y * g * 0.32, 0.001, 0.005)
    if name == 'tick':
        n = _ns(0.05)
        t = np.arange(n) / SR
        y = highpass(lfsr_noise(n, 48000, short=True, seed=seed), 5000) * np.exp(-t / 0.008)
        return _fade_ends(y * g * 0.2, 0.001, 0.004)
    if name in ('hat', 'hat_soft', 'ohat'):
        dur, tau = (0.09, 0.028) if name != 'ohat' else (0.5, 0.22)
        if name == 'hat_soft':
            tau = 0.022
        n = _ns(dur)
        y = 0.55 * highpass(_metal(n), 7000) + 0.45 * highpass(white(n, seed), 8000)
        y = y * perc_env(n, 0.001, tau)
        lvl = {'hat': 0.2, 'hat_soft': 0.14, 'ohat': 0.16}[name]
        return _fade_ends(y * g * lvl, 0.001, 0.005)
    if name == 'shaker':
        n = _ns(0.12)
        y = bandpass(white(n, seed), 4000, 11000) * perc_env(n, 0.012, 0.035)
        return _fade_ends(y * g * 0.18, 0.003, 0.005)
    if name in ('crash', 'crash_soft', 'ride'):
        dur, tau = {'crash': (2.6, 0.9), 'crash_soft': (2.8, 1.0), 'ride': (1.2, 0.4)}[name]
        n = _ns(dur)
        t = np.arange(n) / SR
        y = 0.5 * highpass(_metal(n, 1.37), 4000) + 0.5 * highpass(white(n, seed), 3500)
        y = y * perc_env(n, 0.002 if name != 'crash_soft' else 0.02, tau)
        if name == 'ride':
            y += 0.25 * np.sin(2 * np.pi * 3100 * t) * np.exp(-t / 0.25)
        if name == 'crash_soft':
            y = lowpass(y, 7000)
        lvl = {'crash': 0.26, 'crash_soft': 0.2, 'ride': 0.12}[name]
        return _fade_ends(y * g * lvl, 0.001, 0.05)
    if name.startswith('tom'):
        f0 = p.get('f', {'tom_hi': 210, 'tom_mid': 155, 'tom_lo': 110}[name])
        n = _ns(0.45)
        t = np.arange(n) / SR
        f = f0 * (1 + 0.6 * np.exp(-t / 0.04))
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * perc_env(n, 0.001, 0.2)
        y += 0.15 * bandpass(white(n, seed), 400, 4000) * np.exp(-t / 0.02)
        return _fade_ends(y * g * 0.5, 0.001, 0.01)
    if name == 'boom':
        n = _ns(1.6)
        t = np.arange(n) / SR
        f = 48 + 50 * np.exp(-t / 0.08)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * perc_env(n, 0.004, 0.45)
        y += 0.3 * lowpass(white(n, seed), 400) * perc_env(n, 0.002, 0.06)
        y = highpass(y, 35)
        return _fade_ends(y * g * 0.5, 0.002, 0.05)
    raise ValueError(name)


# ======================================================================= fx
def fx(name, dur_s=1.0, vel=0.8, seed=0, **p):
    """Effects. riser (noise+tone sweep up, ends at dur), swell (reverse cymbal, peaks at dur),
    zap (laser), powerdown (low-battery slide), crackle (vinyl bed, stereo), sweep (downlifter),
    shimmer (stereo glassy cluster)."""
    g = amp_of(vel)
    n = _ns(dur_s)
    t = np.arange(n) / SR
    x = t / max(dur_s, 1e-3)
    if name == 'riser':
        nz = white(n, seed)
        lo, hi = p.get('lo', 300.0), p.get('hi', 7000.0)
        bands = np.geomspace(lo, hi, 9)
        pos = x * (len(bands) - 1)
        y = np.zeros(n)
        for i, fc in enumerate(bands):
            w = np.clip(1.0 - np.abs(pos - i), 0.0, 1.0)
            if w.any():
                y += bandpass(nz, fc / 1.35, fc * 1.35) * w
        tone = osc('saw', p.get('f0', 200.0) * 2.0 ** (x * p.get('octs', 3.0)), n)
        y = y * 1.6 + 0.18 * lowpass(tone, 4000)
        env = x ** p.get('curve', 2.2)
        y = y * env
        nf = min(n, _ns(0.01))
        y[n - nf:] *= np.linspace(1, 0, nf)
        return _fade_ends(y * g * 0.22, 0.01, 0.003)
    if name == 'swell':
        c = drum('crash_soft', 1.0, seed)
        c = c[:n] if len(c) >= n else np.pad(c, (0, n - len(c)))
        y = c[::-1].copy() * x ** 1.5
        y = lowpass(y, p.get('lp', 9000))
        return _fade_ends(y * g * 1.1, 0.02, 0.004)
    if name == 'zap':
        f0 = p.get('f0', 2600.0)
        f1 = p.get('f1', 260.0)
        f = f1 + (f0 - f1) * np.exp(-t / (dur_s * 0.25))
        y = osc('pulse', f, n, 0.125) * perc_env(n, 0.001, dur_s * 0.35)
        y = lowpass(y, 9000)
        return _fade_ends(y * g * 0.12, 0.001, 0.01)
    if name == 'powerdown':
        f0 = p.get('f0', 988.0)
        f = f0 * 2.0 ** (-p.get('octs', 2.5) * x ** 1.3) * (1 + 0.03 * np.sin(2 * np.pi * 9 * t))
        y = osc('pulse', f, n, 0.5) * 0.6 + osc('tri', f * 0.5, n) * 0.4
        y = lowpass(y, 3000) * (1 - x) ** 0.6 * (0.5 - 0.5 * np.cos(np.pi * np.clip(t / 0.01, 0, 1)))
        return _fade_ends(y * g * 0.2, 0.003, 0.02)
    if name == 'sweep':
        nz = white(n, seed)
        y = np.zeros(n)
        bands = np.geomspace(p.get('hi', 6000.0), p.get('lo', 300.0), 7)
        pos = x * (len(bands) - 1)
        for i, fc in enumerate(bands):
            w = np.clip(1.0 - np.abs(pos - i), 0.0, 1.0)
            if w.any():
                y += bandpass(nz, fc / 1.4, fc * 1.4) * w
        y *= (1 - x) ** 1.5
        return _fade_ends(y * g * 0.25, 0.01, 0.01)
    if name == 'crackle':
        rg = np.random.default_rng(seed)
        out = np.zeros((2, n))
        for ch in range(2):
            k = int(dur_s * p.get('rate', 9))
            pos = rg.integers(0, n, k)
            imp = np.zeros(n)
            imp[pos] = rg.uniform(0.2, 1.0, k) * rg.choice([-1, 1], k)
            y = bandpass(imp, 1200, 7000)
            hiss = bandpass(rg.standard_normal(n), 2000, 9000) * 0.012
            out[ch] = y * 0.5 + hiss
        return _fade_ends(out * g * 0.05, 0.3, 0.3)
    if name == 'shimmer':
        rg = np.random.default_rng(seed)
        out = np.zeros((2, n))
        for ch in range(2):
            for i in range(6):
                f = p.get('f0', 1760.0) * 2.0 ** (rg.integers(0, 4) * 0.25 + i * 0.5 / 6)
                out[ch] += np.sin(2 * np.pi * f * t + rg.uniform(0, 6.28)) * (0.5 + 0.5 * np.sin(2 * np.pi * rg.uniform(3, 8) * t))
        out *= np.sin(np.pi * x) ** 2 / 6
        return _fade_ends(out * g * 0.12, 0.01, 0.01)
    raise ValueError(name)


# ======================================================================= buses
def pingpong_echo(x, delay_s=0.45, feedback=0.38, lp=3200.0, hp=350.0):
    """Ping-pong echo of a mono send. First repeat left, second right, ... each repeat
    filtered (LP/HP) and scaled by `feedback`. Returns (2, n) wet signal only."""
    n = len(x)
    D = int(round(delay_s * SR))
    P = 2 * D
    src = highpass(lowpass(x, lp), hp)
    xd = np.zeros(n)
    xd[D:] = src[:n - D]
    sos = _butter('lowpass', round(lp * 1.4, 1), 1)
    sos2 = np.vstack([sos, sos])
    zi = np.zeros((sos2.shape[0], 2))
    eL = np.zeros(n)
    fb2 = feedback * feedback
    for s in range(0, n, P):
        e = min(n, s + P)
        if s == 0:
            eL[s:e] = xd[s:e]
        else:
            prev, zi = signal.sosfilt(sos2, eL[s - P:e - P], zi=zi)
            eL[s:e] = xd[s:e] + fb2 * prev
    eR = np.zeros(n)
    eR[D:] = feedback * signal.sosfilt(sos, eL)[:n - D]
    return np.stack([eL, eR])


def make_reverb_ir(dur=2.4, t60=(1.9, 1.7, 0.9), predelay=0.014, seed=11, width=1.0):
    """Synthetic stereo room/plate IR: sparse early reflections + dense decorrelated noise tail with
    frequency-dependent decay (t60 for low <450 Hz, mid, high >3.5 kHz). Unit energy per channel."""
    n = _ns(dur)
    t = np.arange(n) / SR
    rg = np.random.default_rng(seed)
    out = np.zeros((2, n))
    pd = _ns(predelay)
    for ch in range(2):
        w = rg.standard_normal(n)
        lo = lowpass(w, 450, 2)
        hi = highpass(w, 3500, 2)
        mid = w - lo - hi
        ir = sum(b * 10.0 ** (-3.0 * t / T) for b, T in zip((lo, mid, hi), t60))
        ir *= 1.0 - np.exp(-t / 0.03)                       # diffuse build-up
        er = np.zeros(n)
        for k in range(14):
            tt = rg.uniform(0.004, 0.07)
            er[_ns(tt)] += rg.choice([-1, 1]) * 2.2 * math.exp(-tt / 0.05)
        er = lowpass(er, 6000)
        ir = ir + er * 3.0
        ir = np.concatenate([np.zeros(pd), ir[:n - pd]])
        nf = n // 8
        ir[n - nf:] *= np.cos(np.linspace(0, np.pi / 2, nf)) ** 2
        out[ch] = ir / math.sqrt(np.sum(ir ** 2))
    if width < 1.0:
        m = out.mean(0)
        out = m + (out - m) * width
    return out


def reverb(x, ir):
    """Convolve a stereo send (2, n) with a stereo IR; output trimmed to n samples."""
    n = x.shape[-1]
    return np.stack([signal.oaconvolve(x[c], ir[c])[:n] for c in range(2)])


def duck_curve(n, hits, depth=0.35, attack=0.004, release=0.16, length=0.6):
    """Sidechain-style gain curve: dips by `depth` at each hit sample index, recovers with `release`."""
    k = _ns(length)
    t = np.arange(k) / SR
    na = max(_ns(attack), 2)
    shape = np.exp(-np.maximum(t - attack, 0) / release)
    shape[:na] = np.arange(na) / na
    shape[-_ns(0.02):] *= np.linspace(1, 0, _ns(0.02))
    imp = np.zeros(n)
    hits = np.asarray([h for h in hits if 0 <= h < n], dtype=np.int64)
    if len(hits) == 0:
        return np.ones(n)
    np.add.at(imp, hits, 1.0)
    c = signal.oaconvolve(imp, shape)[:n]
    return 1.0 - depth * np.clip(c, 0.0, 1.0)


def pan_gains(pan):
    """Constant-power pan law, pan in [-1, 1] -> (gL, gR)."""
    a = (pan + 1.0) * np.pi / 4.0
    return math.cos(a) * math.sqrt(2) * 0.7071, math.sin(a) * math.sqrt(2) * 0.7071


# ======================================================================= master / metering
_K1 = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
       np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K2 = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))


def k_weight(x):
    y = signal.lfilter(_K1[0], _K1[1], x, axis=-1)
    return signal.lfilter(_K2[0], _K2[1], y, axis=-1)


def lufs_blocks(x, block=0.4, overlap=0.75):
    """Momentary loudness (LUFS) per 400 ms block (75 % overlap) for a (2, n) signal."""
    y = k_weight(x) ** 2
    ms = y.sum(0) if y.ndim == 2 else y
    nb, hop = _ns(block), _ns(block * (1 - overlap))
    c = np.concatenate([[0.0], np.cumsum(ms)])
    starts = np.arange(0, max(1, len(ms) - nb + 1), hop)
    e = (c[starts + nb] - c[starts]) / nb
    return -0.691 + 10 * np.log10(np.maximum(e, 1e-12)), starts


def lufs(x):
    """Integrated loudness (ITU-R BS.1770-4 with absolute -70 and relative -10 LU gates)."""
    L, _ = lufs_blocks(x)
    e = 10 ** ((L + 0.691) / 10)
    g = L > -70
    if not g.any():
        return -120.0
    rel = -0.691 + 10 * math.log10(e[g].mean()) - 10
    g &= L > rel
    return -0.691 + 10 * math.log10(e[g].mean())


def true_peak_db(x):
    """4x oversampled true peak in dBTP."""
    x = np.atleast_2d(x)
    m = 0.0
    for c in x:
        for i in range(0, len(c), SR * 20):
            seg = c[max(0, i - 64): i + SR * 20 + 64]
            m = max(m, float(np.max(np.abs(signal.resample_poly(seg, 4, 1)))))
    return 20 * math.log10(max(m, 1e-12))


def limiter(x, ceiling_db=-1.3, lookahead=0.004, hold=0.03):
    """Look-ahead peak limiter (zero-overshoot by construction for sample peaks).
    Gain = box-smoothed running-min of the required gain. Returns (y, gain)."""
    c = 10 ** (ceiling_db / 20)
    pk = np.max(np.abs(x), axis=0)
    req = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = max(3, _ns(lookahead))
    H = max(2 * L + 1, _ns(hold))
    g1 = minimum_filter1d(req, size=H, mode='nearest')
    g2 = uniform_filter1d(g1, size=L, mode='nearest')
    g2 = np.minimum(g2, req)                     # numerical safety
    return x * g2, g2


def glue_comp(x, thresh_db=-16.0, ratio=1.8, attack=0.012, release=0.25, knee=6.0):
    """Gentle RMS bus compressor (feed-forward, smoothed in the dB domain). Returns (y, gain_db)."""
    m = (x ** 2).mean(0)
    rms = np.sqrt(onepole_lp(m, 1.0 / (2 * np.pi * 0.03)) + 1e-12)
    lvl = 20 * np.log10(rms + 1e-9)
    over = lvl - thresh_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    # attack/release smoothing on a decimated control signal
    dec = 32
    gd = gr[::dec]
    out = np.empty_like(gd)
    aa = math.exp(-dec / (attack * SR))
    ar = math.exp(-dec / (release * SR))
    s = 0.0
    for i, v in enumerate(gd):
        s = aa * s + (1 - aa) * v if v > s else ar * s + (1 - ar) * v
        out[i] = s
    g = np.interp(np.arange(len(gr)), np.arange(len(gd)) * dec, out)
    return x * 10 ** (-g / 20), -g


def to_int16(x, dither=True, seed=1234, dither_mask=None):
    """Float (2, n) in [-1, 1] -> int16 (n, 2) with optional TPDF dither (mask gates dither per sample)."""
    y = x * 32767.0
    if dither:
        rg = np.random.default_rng(seed)
        d = rg.uniform(-0.5, 0.5, y.shape) + rg.uniform(-0.5, 0.5, y.shape)
        if dither_mask is not None:
            d *= dither_mask
        y = y + d
    return np.clip(np.round(y), -32767, 32767).astype(np.int16).T


def write_wav(path, x16):
    """Write int16 (n, 2) to a 48 kHz stereo 16-bit WAV."""
    with wave.open(path, 'wb') as w:
        w.setnchannels(x16.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.ascontiguousarray(x16).tobytes())


def read_wav(path):
    """Read a 16-bit WAV -> (float (channels, n) in [-1, 1], sample rate)."""
    with wave.open(path, 'rb') as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        assert sw == 2, 'expects 16-bit'
        a = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).T
    return a.astype(np.float64) / 32767.0, sr
