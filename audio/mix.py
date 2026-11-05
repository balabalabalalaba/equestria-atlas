"""ONE GOOD DAY - the final mix: soundtrack + scene SFX + HUD sounds -> audio/out/final_mix.wav

    .venv/bin/python audio/mix.py              # full mix (~15 s) + cue report
    .venv/bin/python audio/mix.py --check      # only import the scenes and print the cue report (no audio)
    .venv/bin/python audio/mix.py --seg lab    # mix one segment only (fast) -> <tmp>/ogd_mix_lab.wav
    .venv/bin/python audio/mix.py --cues       # also list every placed cue
    .venv/bin/python audio/mix.py --verify     # re-read audio/out/final_mix.wav and check format/levels

    more options: --music PATH  --sections PATH  --out PATH  --sfx-db -6  --duck-db 2.5
                  --no-scenes  --no-hud  --no-keyfit  --sfx-only  --music-only

Scene contract (see docs/AUDIO.md):

    SFX = [(local_frame, 'name', gain), ...]          # gain ~0.3 .. 1.0 (max 2.0)
    SFX = [(local_frame, 'name', gain, {'pan': -0.5, 'align': 'hit'}), ...]   # optional options

Pipeline
--------
1. music: audio/out/soundtrack.wav (48 kHz stereo). Missing -> silence of the same length (reported).
2. cues: every scene module from engine/timeline.py is imported (missing / failing modules are
   skipped and reported) and its SFX list read; plus the automatic HUD sounds of every story
   segment: 'banner' at local frame 10, 'quest' at local frame frames-58.
3. each cue starts at (segment start + local frame) / 30 s. Variants cycle automatically per name
   (step, tap, chop, ...); deterministic stereo spread (per-name pan, alternating offsets, golden-
   ratio jitter); mono-compatible width for ambiences; per-name trim + reverb send (sfx.MIX); the
   SFX bus sits SFX_DB (-6 dB) under the music's loudness and follows each segment's music level
   by 50 % (quiet scenes -> slightly quieter SFX).
4. key-fit: tonal effects are written in D major. Where the score is NOT in D major (unipol's
   E-major gym groove, the E-major climax of `ai`) a tonal cue is transposed (resampled) by the
   smallest/most consonant shift into the local key, read from audio/out/sections.json.
5. a short stereo room reverb (per-segment room size), a light sidechain duck of the music
   (up to -2.5 dB, 10 ms attack, 150 ms release) under loud SFX, the same end fade as the score.
6. master: soft-knee look-ahead limiter on the 4x-oversampled (true) peak, true peak <= -1 dBTP,
   TPDF dither, 16-bit 48 kHz stereo, exactly the film length (TOTAL_FRAMES * 1600 samples).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audio import sfx                     # noqa: E402
from engine import timeline as TL         # noqa: E402

# ------------------------------------------------------------------------------- constants
SR = sfx.SR                                # 48 000
FPS = 30
SPF = SR // FPS                            # 1600 samples per video frame
TOTAL_FRAMES = TL.TOTAL_FRAMES             # from engine/timeline.py
N_TOTAL = TOTAL_FRAMES * SPF               # samples in the whole film

OUT_DIR = os.path.join(ROOT, 'audio', 'out')
MUSIC_PATH = os.path.join(OUT_DIR, 'soundtrack.wav')
SECTIONS_PATH = os.path.join(OUT_DIR, 'sections.json')
FINAL_PATH = os.path.join(OUT_DIR, 'final_mix.wav')

SFX_DB = -6.0               # nominal effect (gain 1, trim 0) sits this far under the music loudness
DEFAULT_MUSIC_LUFS = -14.5  # assumed music loudness when there is no soundtrack
TRACK = 0.5                 # how much the SFX level follows each segment's music loudness
TRACK_MAX = 5.0             # ... clamped to +-5 dB
MAX_GAIN = 2.0              # per-cue gain clamp

DUCK_DB = 2.5               # max music gain reduction under loud SFX
DUCK_ATTACK = 0.010         # s
DUCK_RELEASE = 0.150        # s
DUCK_LOOKAHEAD = 0.004      # s
DUCK_WIN = 0.030            # s, detector window (trailing RMS, K-weighted)
DUCK_START = -8.0           # dB re the SFX reference level: ducking starts here ...
DUCK_FULL = 0.0             # ... and is full here

TP_TARGET = -1.0            # dBTP
CEILING_DB = -1.25          # limiter ceiling on the 4x-oversampled peak (iterated if needed)
KNEE_DB = 1.5               # soft knee below the ceiling
LIMIT_LOOKAHEAD = 0.0015    # s
LIMIT_RELEASE = 20.0        # dB / s

END_FADE = (TL.DURATION - 1.7, TL.DURATION - 0.015)  # s, last 1.7 s of the film; digital silence after
BANNER_FRAME = 10            # HUD 'NEW QUEST' banner slides in at local frame 10
QUEST_BEFORE_END = 58        # HUD 'QUEST COMPLETE' toast at local frame frames-58

# reverb send offset per segment (dB): small rooms drier, big / magic spaces wetter
ROOM = {'intro': 2, 'rise': -3, 'campus': 0, 'unipol': 2, 'brunch': -2, 'lab': 1, 'lunch': -2,
        'discuss': -1, 'newworld': 1, 'dinner': -3, 'ai': 3, 'night': -1, 'outro': 3}
# mono-compatible stereo width (decorrelated side signal) per effect, 0..1
WIDTH = {'birds': 0.6, 'shimmer': 0.5, 'star': 0.5, 'sparkle': 0.35, 'water': 0.4, 'sizzle': 0.35,
         'cart': 0.3, 'gull': 0.3, 'bell': 0.35, 'microwave': 0.2, 'pour': 0.25, 'kiwi': 0.2,
         'sweep': 0.3, 'curtain': 0.3, 'whoosh': 0.3, 'idea': 0.3, 'powerup': 0.2, 'quest': 0.3,
         'banner': 0.3, 'chime': 0.25, 'ding': 0.2, 'laser': 0.3, 'hum': 0.2, 'bag': 0.3,
         'typing': 0.2, 'grind': 0.2, 'puff': 0.3, 'alarm': 0.1, 'fridge': 0.2, 'growl': 0.1}
WIDTH.update(getattr(sfx, 'WIDTH_HINT', {}))      # atlas ambiences (wind, rain, waves...)
OPTION_KEYS = ('pan', 'variant', 'pitch', 'align', 'reverb', 'keyfit', 'width')
HUD_NAMES = ('banner', 'quest')
PC_NAMES = sfx.PC_NAMES


def _db(x):
    return 20 * math.log10(max(float(x), 1e-12))


# ------------------------------------------------------------------------------- loudness / peaks
def lufs(x):
    """BS.1770-4 integrated loudness (LUFS) of mono (n,) or multichannel (ch, n) audio."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    n = x.shape[1]
    B, H = int(0.4 * SR), int(0.1 * SR)
    if n < B:
        return -70.0
    p = np.zeros(n)
    for ch in x:
        p += sfx.kweight(ch) ** 2
    c = np.concatenate([[0.0], np.cumsum(p)])
    s = np.arange(0, n - B + 1, H)
    ms = (c[s + B] - c[s]) / B
    ms = ms[ms > 10 ** ((-70 + 0.691) / 10)]
    if not len(ms):
        return -70.0
    rel = -0.691 + 10 * np.log10(ms.mean()) - 10
    ms = ms[ms > 10 ** ((rel + 0.691) / 10)]
    return float(-0.691 + 10 * np.log10(ms.mean())) if len(ms) else -70.0


def tp_env(x, chunk=1 << 19):
    """Per-sample peak of the 4x-oversampled signal (max over channels and the 4 sub-samples)."""
    x = np.atleast_2d(x)
    ch, n = x.shape
    out = np.abs(x).max(axis=0)
    pad = 64
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        a1, b1 = max(0, a - pad), min(n, b + pad)
        up = signal.resample_poly(x[:, a1:b1], 4, 1, axis=1)
        off = (a - a1) * 4
        u = np.abs(up[:, off:off + (b - a) * 4]).reshape(ch, b - a, 4).max(axis=(0, 2))
        out[a:b] = np.maximum(out[a:b], u)
    return out


def true_peak_db(x):
    return _db(tp_env(x).max())


# ------------------------------------------------------------------------------- music
def load_music(path=MUSIC_PATH):
    """(stereo float64 (2, N_TOTAL), status string)."""
    if not path or not os.path.exists(path):
        return np.zeros((2, N_TOTAL)), f'MISSING ({path}) - mixing the SFX over silence'
    try:
        sr, d = wavfile.read(path)
    except Exception as ex:          # e.g. being rewritten by the music agent right now
        return np.zeros((2, N_TOTAL)), f'UNREADABLE ({path}: {ex}) - mixing the SFX over silence'
    if d.dtype == np.int16:
        x = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32:
        x = d.astype(np.float64) / 2147483648.0
    elif d.dtype == np.uint8:
        x = (d.astype(np.float64) - 128) / 128.0
    else:
        x = d.astype(np.float64)
    x = x[:, None] if x.ndim == 1 else x
    x = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x[:, :2]
    x = x.T.copy()
    notes = []
    if sr != SR:
        fr = Fraction(SR, sr).limit_denominator(1000)
        x = signal.resample_poly(x, fr.numerator, fr.denominator, axis=1)
        notes.append(f'resampled {sr} -> {SR} Hz')
    n = x.shape[1]
    if n != N_TOTAL:
        notes.append(f'length {n} samples ({n / SR:.3f} s) -> {"padded" if n < N_TOTAL else "trimmed"} '
                     f'to {N_TOTAL}')
        x = np.pad(x, ((0, 0), (0, max(0, N_TOTAL - n))))[:, :N_TOTAL]
    return x, 'ok ' + os.path.relpath(path, ROOT) + (' (' + '; '.join(notes) + ')' if notes else '')


def music_refs(music, sfx_db=SFX_DB):
    """SFX reference loudness (dB, the level a gain-1/trim-0 effect is normalised to) per segment."""
    I = lufs(music)
    silent = I < -60
    base = (DEFAULT_MUSIC_LUFS if silent else I) + sfx_db
    refs, seg_l = {}, {}
    for s in TL.SEGMENTS:
        if silent:
            refs[s['id']] = base
            seg_l[s['id']] = None
            continue
        L = lufs(music[:, s['start'] * SPF:s['end'] * SPF])
        seg_l[s['id']] = L
        d = 0.0 if L < -60 else float(np.clip(L - I, -TRACK_MAX, TRACK_MAX))
        refs[s['id']] = base + TRACK * d
    return refs, dict(music_lufs=None if silent else round(I, 2), seg_lufs=seg_l, sfx_ref=round(base, 2))


# ------------------------------------------------------------------------------- harmony / key-fit
_ROOT_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_MAJOR = (0, 2, 4, 5, 7, 9, 11)


def _pc(name):
    return (_ROOT_PC[name[0]] + {'#': 1, 'b': -1}.get(name[1:2], 0)) % 12


def chord_pcs(sym):
    """Pitch classes of a chord symbol ('Dmaj9', 'F#7b9', 'E/D', 'A7sus4', 'C#m7b5' ...), bass included."""
    m = re.fullmatch(r'\s*([A-G][#b]?)([^/]*?)(?:/([A-G][#b]?))?\s*', str(sym))
    if not m:
        raise ValueError(f'bad chord {sym!r}')
    root, q, bass = _pc(m.group(1)), m.group(2), m.group(3)
    minor = bool(re.match(r'(m(?!aj)|min|-|dim)', q))
    third = 3 if minor else 4
    if 'sus4' in q or 'sus' == q:
        third = 5
    elif 'sus2' in q:
        third = 2
    fifth = 6 if ('dim' in q or 'b5' in q) else 8 if ('aug' in q or '#5' in q or q == '+') else 7
    ivs = {0, third, fifth}
    ext = re.sub(r'add\d+|69', '', q)
    if re.search(r'(maj|M|Δ)(7|9|11|13)', q):
        ivs.add(11)
    elif 'dim7' in q:
        ivs.add(9)
    elif re.search(r'(?<![#b])(7|9|11|13)', ext):
        ivs.add(10)
    if re.search(r'(?<![#b])(9|11|13)', ext) or 'add9' in q or '69' in q:
        ivs.add(2)
    if 'b9' in q:
        ivs.add(1)
    if '#9' in q:
        ivs.add(3)
    if '#11' in q:
        ivs.add(6)
    elif re.search(r'(?<![#b])11', q):
        ivs.add(5)
    if '13' in q or re.search(r'(?<!1)6', q):
        ivs.add(9)
    pcs = {(root + i) % 12 for i in ivs}
    if bass:
        pcs.add(_pc(bass))
    return frozenset(pcs)


def _scale(k):
    return frozenset((k + i) % 12 for i in _MAJOR)


def _fifths_from_d(k):
    d = abs((k * 7) % 12 - (2 * 7) % 12) % 12
    return min(d, 12 - d)


class Harmony:
    """Chord timeline of the score (from audio/out/sections.json) for the key-fit of tonal SFX."""

    def __init__(self, path=SECTIONS_PATH):
        self.ok, self.status, self.chords = False, '', {}
        if not path or not os.path.exists(path):
            self.status = f'no {os.path.relpath(path, ROOT) if path else "sections.json"}: key-fit off'
            return
        try:
            with open(path) as f:
                secs = json.load(f)
            nch = 0
            for s in secs:
                if s.get('id') not in TL.BY_ID:
                    continue
                seg_ = TL.BY_ID[s['id']]
                rows = sorted((c['local_frame'], chord_pcs(c['chord'])) for c in s.get('chords', []))
                lst = []
                for i, (lf, pcs) in enumerate(rows):
                    end = rows[i + 1][0] if i + 1 < len(rows) else seg_['frames']
                    lst.append((seg_['start'] + lf, seg_['start'] + end, pcs))
                self.chords[s['id']] = lst
                nch += len(lst)
            self.ok = nch > 0
            self.status = f'{nch} chords from {os.path.relpath(path, ROOT)}' if self.ok else 'no chords: key-fit off'
        except Exception as ex:
            self.status = f'cannot read {path}: {ex}: key-fit off'

    def chord_at(self, gf):
        s, _ = TL.locate(int(gf))
        for a, b, pcs in self.chords.get(s['id'], []):
            if a <= gf < b:
                return pcs
        return None

    def _fit(self, sid, a, b):
        """Per major key (12,): share of chord tones inside its scale, weighted by chord time in [a, b)."""
        fit, tot = np.zeros(12), 0.0
        for c0, c1, pcs in self.chords.get(sid, []):
            w = min(b, c1) - max(a, c0)
            if w <= 0 or not pcs:
                continue
            tot += w
            for k in range(12):
                fit[k] += w * len(pcs & _scale(k)) / len(pcs)
        return fit / tot if tot > 0 else None

    def local_key(self, gf):
        """Major key (pc) of the music around global frame gf (None if no chords there).

        Every major key is scored by the share of chord tones it contains, weighted by how long each
        chord sounds in the window [gf - 1 beat, gf + 5 beats] (an effect rings after its cue). D major
        wins unless another key is clearly better (by >= 0.1), so passing chromatic chords (F#7, Gm6,
        E/D ...) never trigger a transposition - only real key areas (unipol / ai in E major) do.
        Ties (a lone E chord fits A, E and B major) are broken by the previous two bars."""
        sid = TL.locate(int(gf))[0]['id']
        near = self._fit(sid, gf - 18, gf + 90)
        if near is None:
            return None
        top = float(near.max())
        if near[2] >= 0.85 or near[2] >= top - 0.1:
            return 2
        cands = [k for k in range(12) if near[k] >= top - 0.02]
        ctx = self._fit(sid, gf - 144, gf + 90)
        return max(cands, key=lambda k: (round(float(ctx[k]), 6), -_fifths_from_d(k)))

    def shift_for(self, pcs, gf):
        """(semitones, key name) that move a D-major effect with pitch classes `pcs` into the local key."""
        if not self.ok or not pcs:
            return 0, None
        key = self.local_key(gf)
        if key is None:
            return 0, None
        sc = _scale(key)
        if set(pcs) <= sc:
            return 0, PC_NAMES[key]
        cur = self.chord_at(gf) or frozenset()
        best = None
        for s in range(-5, 7):
            sh = {(p + s) % 12 for p in pcs}
            if not sh <= sc:
                continue
            score = sum(1.0 if q in cur else (-1.5 if (q - 1) % 12 in cur else 0.0) for q in sh) - 0.3 * abs(s)
            if best is None or score > best[0]:
                best = (score, s)
        if best is None:
            return ((key - 2 + 5) % 12) - 5, PC_NAMES[key]
        return best[1], PC_NAMES[key]


# ------------------------------------------------------------------------------- cues
def _parse_entry(e):
    """(frame, name, gain, opts, warnings) from a scene SFX entry."""
    warns = []
    if isinstance(e, dict):
        d = dict(e)
        frame, name, gain = d.pop('frame'), d.pop('name'), d.pop('gain', 1.0)
        opts = d
    else:
        e = list(e)
        if not 2 <= len(e) <= 4:
            raise ValueError('expected (local_frame, name, gain[, options])')
        frame, name = e[0], e[1]
        gain = e[2] if len(e) > 2 else 1.0
        opts = dict(e[3]) if len(e) > 3 and e[3] else {}
    if isinstance(frame, str) and not isinstance(name, str):
        frame, name = name, frame
        warns.append('(name, frame) order swapped')
    frame, gain = float(frame), float(gain)
    if not math.isfinite(frame) or not math.isfinite(gain):
        raise ValueError('non-finite frame/gain')
    bad = [k for k in opts if k not in OPTION_KEYS]
    if bad:
        warns.append(f'ignored option(s) {bad}')
        opts = {k: v for k, v in opts.items() if k in OPTION_KEYS}
    return frame, str(name), gain, opts, warns


def _import_scene(modname):
    """(SFX list or None, status)."""
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as ex:
        if ex.name and modname.startswith(ex.name):
            return None, 'missing'
        return None, f'error: {type(ex).__name__}: {ex}'
    except (Exception, SystemExit) as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        where = (f' ({os.path.basename(tb[-1].filename)}:{tb[-1].lineno})'
                 if tb and not isinstance(ex, SyntaxError) else '')
        return None, f'error: {type(ex).__name__}: {ex}{where}'
    raw = getattr(mod, 'SFX', None)
    if raw is None:
        return [], 'ok, no SFX'
    try:
        if callable(raw):
            raw = raw()
        return list(raw), 'ok'
    except Exception as ex:
        return None, f'error reading SFX: {type(ex).__name__}: {ex}'


def collect_cues(scenes=True, hud=True, only=None, harmony=None, keyfit=True):
    """Gather and annotate every cue. Returns (cues, rows): one row per segment for the report.

    Each cue is a dict: seg, local, frame (global), name, variant, gain, trim, send, pan, width,
    semis (transposition), key, src ('scene'|'hud'), opts."""
    cues, rows = [], []
    for s in TL.SEGMENTS:
        if only and s['id'] not in only:
            continue
        row = dict(id=s['id'], module=s['module'], status='skipped', scene=0, hud=0, unknown=[], problems=[],
                   shifts=[], ms=0.0)
        entries = []
        if scenes:
            t0 = time.time()
            got, row['status'] = _import_scene(s['module'])
            row['ms'] = (time.time() - t0) * 1000
            entries = got or []
        auto = {'banner': BANNER_FRAME, 'quest': s['frames'] - QUEST_BEFORE_END} if (hud and s.get('quest')) else {}
        sc = []
        for k, e in enumerate(entries):
            try:
                frame, name, gain, opts, warns = _parse_entry(e)
            except Exception as ex:
                row['problems'].append(f'bad entry {e!r}: {ex}')
                continue
            try:
                key = sfx.resolve(name)
            except KeyError:
                row['unknown'].append(name)
                continue
            for w in warns:
                row['problems'].append(f'{name}@{frame:g}: {w}')
            if not 0 <= frame < s['frames']:
                row['problems'].append(f'{key}@{frame:g}: frame outside 0..{s["frames"] - 1}, skipped')
                continue
            if gain <= 0:
                continue
            if gain > MAX_GAIN:
                row['problems'].append(f'{key}@{frame:g}: gain {gain:g} clamped to {MAX_GAIN:g}')
                gain = MAX_GAIN
            if key in auto and abs(frame - auto[key]) <= 6:
                row['problems'].append(f'{key}@{frame:g}: the mixer adds this HUD sound itself, skipped')
                continue
            sc.append(dict(seg=s['id'], local=frame, name=key, gain=gain, opts=opts, src='scene', order=k))
        for nm, lf in auto.items():
            sc.append(dict(seg=s['id'], local=float(lf), name=nm, gain=1.0, opts={}, src='hud', order=10 ** 6))
        sc.sort(key=lambda c: (c['local'], c['order']))
        count = {}
        for i, c in enumerate(sc):
            nm, o = c['name'], c['opts']
            k = count.get(nm, 0)
            count[nm] = k + 1
            c['frame'] = s['start'] + c['local']
            c['variant'] = int(o.get('variant', k)) % sfx.VARIANTS[nm]
            trim, send, pan0, alt = sfx.MIX[nm]
            if 'pan' in o:
                pan = float(o['pan'])
            else:
                pan = pan0 + (alt if k % 2 == 0 else -alt)
                if c['src'] == 'scene':
                    pan += 0.07 * ((((i + 1) * 0.6180339887498949) % 1.0) * 2 - 1)
            c['pan'] = float(np.clip(pan, -0.9, 0.9))
            c['trim'] = float(trim)
            c['send'] = float(send) + (0.0 if c['src'] == 'hud' else ROOM.get(s['id'], 0)) + float(o.get('reverb', 0))
            c['width'] = float(np.clip(o.get('width', WIDTH.get(nm, 0.0)), 0, 1))
            c['semis'], c['key'] = 0.0, None
            if 'pitch' in o:
                c['semis'] = float(o['pitch'])
            elif keyfit and harmony is not None and o.get('keyfit', True):
                hit_f = c['frame'] + sfx.hit(nm) * FPS
                st, kname = harmony.shift_for(sfx.pitch_classes(nm, c['variant']), hit_f)
                c['semis'], c['key'] = float(st), kname
                if st:
                    row['shifts'].append(f'{nm}@{c["local"]:g} {st:+d} ({kname})')
        row['scene'] = sum(c['src'] == 'scene' for c in sc)
        row['hud'] = sum(c['src'] == 'hud' for c in sc)
        cues += sc
        rows.append(row)
    return cues, rows


# ------------------------------------------------------------------------------- per-cue audio
def _edge_fade(x, a=0.0005, b=0.004):
    x = np.array(x, np.float64)
    na, nb = min(len(x), int(a * SR)), min(len(x), int(b * SR))
    if na:
        x[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    if nb:
        x[len(x) - nb:] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(nb) / nb))[::-1]
    if len(x):
        x[-1] = 0.0
    return x


@lru_cache(maxsize=None)
def _mono(name, variant, semis):
    """The effect (float64), transposed by `semis` (resampling), with 40 ms of silence appended."""
    x = sfx.make(name, variant).astype(np.float64)
    if semis:
        fr = Fraction(2 ** (semis / 12)).limit_denominator(400)
        x = _edge_fade(signal.resample_poly(x, fr.denominator, fr.numerator))
    return np.concatenate([x, np.zeros(int(0.04 * SR))])


@lru_cache(maxsize=None)
def _side(name, variant, semis):
    """Decorrelated copy (Schroeder all-pass chain) for mono-compatible width."""
    y = _mono(name, variant, semis)
    for d, g in ((0.0031, 0.55), (0.0053, -0.5), (0.0087, 0.45), (0.0123, -0.4)):
        k = int(d * SR)
        b = np.zeros(k + 1)
        a = np.zeros(k + 1)
        b[0], b[k], a[0], a[k] = -g, 1.0, 1.0, -g
        y = signal.lfilter(b, a, y)
    return _edge_fade(y, 0.0005, 0.03)


def _start_time(c):
    """Seconds (global) at which the cue's audio starts."""
    r = 2 ** (c['semis'] / 12)
    h = sfx.hit(c['name'])
    t = c['frame'] / FPS + h * (1 - 1 / r)       # transposed: keep the main hit where it was
    if c['opts'].get('align') == 'hit':
        t -= h / r
    return t


def place(c, dry, send, a0, bus_db):
    """Mix cue `c` into the stereo dry buffer and the mono reverb send (both start at sample a0)."""
    x = _mono(c['name'], c['variant'], c['semis'])
    i0 = int(round(_start_time(c) * SR)) - a0
    n = dry.shape[1]
    j0, j1 = max(0, i0), min(n, i0 + len(x))
    if j1 <= j0:
        return False
    s0, s1 = j0 - i0, j1 - i0
    lin = c['gain'] * 10 ** ((c['trim'] + bus_db) / 20)
    th = (c['pan'] + 1) * math.pi / 4
    gl, gr = math.cos(th), math.sin(th)
    m = x[s0:s1]
    w = c['width']
    if w > 0:
        d = _side(c['name'], c['variant'], c['semis'])[s0:s1]
        k = lin / math.sqrt(1 + w * w)
        L, R = gl * k * (m + w * d), gr * k * (m - w * d)
    else:
        L, R = gl * lin * m, gr * lin * m
    if s0 > 0:          # cue started before this buffer: soften the cut
        f = np.ones(len(m))
        nf = min(len(m), int(0.003 * SR))
        f[:nf] = np.linspace(0, 1, nf)
        L, R = L * f, R * f
    dry[0, j0:j1] += L
    dry[1, j0:j1] += R
    send[j0:j1] += lin * 10 ** (c['send'] / 20) * m
    return True


# ------------------------------------------------------------------------------- reverb / duck / limiter
@lru_cache(maxsize=2)
def reverb_ir(rt60=1.15, pre=0.014, seed=29):
    """Stereo room IR (2, n): early reflections + two-band exponential tail, unit energy per channel."""
    r = np.random.default_rng(seed)
    n = int((pre + rt60 * 1.05) * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for ch in range(2):
        z = r.standard_normal(n)
        lo = sfx.lp(z, 2500)
        hi = z - lo
        tail = (lo * np.exp(-6.91 * t / rt60) + 0.6 * hi * np.exp(-6.91 * t / (rt60 * 0.45)))
        tail *= np.clip((t - pre) / 0.012, 0, 1)
        er = np.zeros(n)
        for j, ms in enumerate((5.3, 8.9, 12.7, 17.1, 23.5, 31.0, 41.0) if ch == 0 else
                               (6.1, 9.7, 13.9, 19.3, 25.1, 33.7, 44.0)):
            er[int(ms * SR / 1000)] += (0.9 - 0.1 * j) * (1 if (j + ch) % 2 == 0 else -1)
        y = 0.55 * er / np.sqrt((er ** 2).sum()) + tail / np.sqrt((tail ** 2).sum())
        y = sfx.lp(sfx.hp(y, 180), 8500)
        ir[ch] = _edge_fade(y, 0.0, 0.05) / np.sqrt((y ** 2).sum())
    return ir


def apply_reverb(send):
    if not np.any(send):
        return np.zeros((2, len(send)))
    ir = reverb_ir()
    return np.stack([signal.oaconvolve(send, ir[ch])[:len(send)] for ch in range(2)])


def duck_gain(dry, ref_per_sample, depth_db=DUCK_DB):
    """Music gain (n,) under the SFX bus: K-weighted 20 ms trailing level vs the SFX reference."""
    n = dry.shape[1]
    if depth_db <= 0 or not np.any(dry):
        return np.ones(n), 0.0
    p = sfx.kweight(dry[0]) ** 2 + sfx.kweight(dry[1]) ** 2
    W = int(DUCK_WIN * SR)
    c = np.concatenate([[0.0], np.cumsum(p)])
    blk = 240
    idx = np.arange(0, n, blk)
    lo = np.maximum(0, idx + 1 - W)
    ms = (c[idx + 1] - c[lo]) / W
    L = -0.691 + 10 * np.log10(ms + 1e-20)
    rel = L - ref_per_sample[idx]
    tgt = -depth_db * np.clip((rel - DUCK_START) / (DUCK_FULL - DUCK_START), 0, 1)
    ga = 1 - math.exp(-blk / SR / DUCK_ATTACK)
    gr = 1 - math.exp(-blk / SR / DUCK_RELEASE)
    out = np.empty(len(tgt))
    g = 0.0
    for i, t in enumerate(tgt.tolist()):
        g += (t - g) * (ga if t < g else gr)
        out[i] = g
    gdb = np.interp(np.arange(n), idx - int(DUCK_LOOKAHEAD * SR), out)
    return 10 ** (gdb / 20), float(-out.min())


def soft_limit(x, ceiling_db=CEILING_DB, knee_db=KNEE_DB):
    """Linked stereo look-ahead limiter on the oversampled peak with a tanh soft knee.
    Output peak (4x oversampled) never exceeds ceiling_db. Returns (y, gain_db (n,))."""
    n = x.shape[1]
    pk = tp_env(x)
    L = 20 * np.log10(pk + 1e-12)
    thr = ceiling_db - knee_db
    over = L - thr
    req = np.where(over > 0, thr + knee_db * np.tanh(np.maximum(over, 0) / knee_db) - L, 0.0)
    if req.min() > -1e-6:
        return x.copy(), np.zeros(n)
    W = int(LIMIT_LOOKAHEAD * SR)
    g = uniform_filter1d(minimum_filter1d(req, 2 * W + 1, mode='nearest'), 2 * W + 1, mode='nearest')
    rate = LIMIT_RELEASE / SR
    ramp = rate * np.arange(n)
    g = np.minimum(np.minimum.accumulate(g - ramp) + ramp, 0.0)
    W2 = int(0.001 * SR)
    g = uniform_filter1d(minimum_filter1d(g, 2 * W2 + 1, mode='nearest'), 2 * W2 + 1, mode='nearest')
    return x * 10 ** (g / 20), g


def end_fade(a0, n):
    """The film's end fade (cos^2 over END_FADE, then silence) for samples a0..a0+n."""
    t = (a0 + np.arange(n)) / SR
    f = np.ones(n)
    m = (t >= END_FADE[0]) & (t < END_FADE[1])
    f[m] = np.cos(0.5 * np.pi * (t[m] - END_FADE[0]) / (END_FADE[1] - END_FADE[0])) ** 2
    f[t >= END_FADE[1]] = 0.0
    return f


def to_int16(y, seed=1601):
    """TPDF-dithered 16-bit (n, 2) array; exact digital silence stays silent."""
    r = np.random.default_rng(seed)
    d = (r.random(y.shape) - r.random(y.shape)) / 32768.0
    act = maximum_filter1d((np.abs(y).max(axis=0) > 1e-7).astype(np.uint8), 193) > 0
    q = np.round((y + d * act) * 32767.0)
    return np.clip(q, -32767, 32767).astype(np.int16).T.copy()


# ------------------------------------------------------------------------------- render
def render(cues, music, refs, a=0, b=N_TOTAL, sfx_only=False, music_only=False, duck_db=DUCK_DB):
    """Mix samples a..b. Returns (stereo float64 (2, b-a), info dict)."""
    pre = min(a, 3 * SR)                       # pre-roll so earlier cues / reverb tails are present
    a0 = a - pre
    n = b - a0
    dry = np.zeros((2, n))
    send = np.zeros(n)
    placed = 0
    if not music_only:
        for c in cues:
            placed += place(c, dry, send, a0, refs[c['seg']] - sfx.NOMINAL_LK)
    bus = (dry + apply_reverb(send)) * end_fade(a0, n)
    ref_s = np.empty(n)
    for s in TL.SEGMENTS:
        i, j = max(0, s['start'] * SPF - a0), min(n, s['end'] * SPF - a0)
        if j > i:
            ref_s[i:j] = refs[s['id']]
    m = np.zeros((2, n)) if sfx_only else music[:, a0:b].copy()
    g, duck_max = duck_gain(dry, ref_s, 0.0 if (sfx_only or music_only) else duck_db)
    mix = (m * g + bus)[:, pre:]
    ceiling = CEILING_DB
    for _ in range(4):
        y, gdb = soft_limit(mix, ceiling)
        tp = true_peak_db(y)
        if tp <= TP_TARGET - 0.04:
            break
        ceiling -= (tp - (TP_TARGET - 0.04)) + 0.03
    y *= end_fade(a, y.shape[1]) > 0            # exact digital silence after the end fade
    gg = g[pre:]
    info = dict(placed=placed, duck_max_db=round(duck_max, 2),
                duck_time_pct=round(float(np.mean(gg < 10 ** (-1 / 20)) * 100), 2),
                limiter_max_db=round(float(-gdb.min()), 2),
                limiter_time_pct=round(float(np.mean(gdb < -0.5) * 100), 3),
                ceiling_db=round(ceiling, 2), true_peak_db=round(tp, 2),
                sfx_bus_peak_db=round(_db(np.abs(bus).max()), 2))
    return y, info


def write_wav(path, y):
    q = to_int16(y)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp.wav'
    wavfile.write(tmp, SR, q)
    os.replace(tmp, path)
    return q


def verify(path=FINAL_PATH, expect_len=N_TOTAL):
    """Re-read a written mix and check format, length, clipping, true peak, loudness."""
    sr, d = wavfile.read(path)
    x = d.astype(np.float64).T / 32768.0
    probs = []
    if sr != SR:
        probs.append(f'sample rate {sr}')
    if d.dtype != np.int16:
        probs.append(f'dtype {d.dtype}')
    if d.ndim != 2 or d.shape[1] != 2:
        probs.append(f'shape {d.shape}')
    if expect_len and d.shape[0] != expect_len:
        probs.append(f'length {d.shape[0]} != {expect_len}')
    clip = int(np.sum(np.abs(d.astype(np.int32)) >= 32767))
    if clip:
        probs.append(f'{clip} full-scale samples')
    tp = true_peak_db(x)
    if tp > TP_TARGET + 1e-3:
        probs.append(f'true peak {tp:.2f} dBTP > {TP_TARGET}')
    tail = int(np.abs(d[-int(0.01 * SR):]).max()) if len(d) else 0
    res = dict(path=os.path.relpath(path, ROOT), sr=sr, dtype=str(d.dtype), shape=tuple(d.shape),
               seconds=round(d.shape[0] / sr, 4), sample_peak_db=round(_db(np.abs(x).max()), 2),
               true_peak_db=round(tp, 2), lufs=round(lufs(x), 2), last_10ms_max=tail, problems=probs)
    return res


# ------------------------------------------------------------------------------- report
def print_report(rows, cues, harmony, list_cues=False):
    print(f'{"segment":9s} {"frames":>6s} {"scene":>5s} {"hud":>3s}  {"module status":34s} notes')
    tot = 0
    for r in rows:
        notes = []
        if r['unknown']:
            notes.append('UNKNOWN: ' + ', '.join(sorted(set(r['unknown']))))
        if r['shifts']:
            notes.append('key-fit: ' + ', '.join(r['shifts']))
        st = r['status'] + (f' {r["ms"]:.0f}ms' if r['status'].startswith('ok') and r['ms'] > 500 else '')
        fr = TL.BY_ID[r['id']]['frames']
        print(f'{r["id"]:9s} {fr:6d} {r["scene"]:5d} {r["hud"]:3d}  {st[:34]:34s} {"; ".join(notes)}')
        if len(st) > 34:
            print(f'{"":30s}{st}')
        for p in r['problems']:
            print(f'{"":30s}! {p}')
        tot += r['scene'] + r['hud']
    unk = sorted({u for r in rows for u in r['unknown']})
    print(f'total cues: {tot}  (scene {sum(r["scene"] for r in rows)}, hud {sum(r["hud"] for r in rows)})'
          f'   unknown names: {", ".join(unk) if unk else "none"}   harmony: {harmony.status}')
    if list_cues:
        print(f'\n{"time s":>8s} {"seg":9s} {"local":>6s} {"name":10s} {"v":>1s} {"gain":>5s} {"pan":>5s} '
              f'{"semis":>5s} src')
        for c in sorted(cues, key=lambda c: c['frame']):
            print(f'{_start_time(c):8.3f} {c["seg"]:9s} {c["local"]:6g} {c["name"]:10s} {c["variant"]:1d} '
                  f'{c["gain"]:5.2f} {c["pan"]:+5.2f} {c["semis"]:+5.1f} {c["src"]}')


# ------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description='ONE GOOD DAY final mix (soundtrack + SFX + HUD sounds).')
    ap.add_argument('--check', action='store_true', help='only collect cues and print the report')
    ap.add_argument('--seg', help='mix only this segment (fast preview)')
    ap.add_argument('--cues', action='store_true', help='list every cue')
    ap.add_argument('--verify', action='store_true', help='verify the written final mix and exit')
    ap.add_argument('--music', default=MUSIC_PATH)
    ap.add_argument('--sections', default=SECTIONS_PATH)
    ap.add_argument('--out')
    ap.add_argument('--sfx-db', type=float, default=SFX_DB)
    ap.add_argument('--duck-db', type=float, default=DUCK_DB)
    ap.add_argument('--no-scenes', action='store_true')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--no-keyfit', action='store_true')
    ap.add_argument('--sfx-only', action='store_true', help='no music (SFX bus alone)')
    ap.add_argument('--music-only', action='store_true', help='no SFX (music through the master only)')
    a = ap.parse_args(argv)

    if a.verify:
        v = verify(a.out or FINAL_PATH)
        print(v)
        return 1 if v['problems'] else 0
    t0 = time.time()
    if a.seg and a.seg not in TL.BY_ID:
        ap.error(f'unknown segment {a.seg!r}; have {[s["id"] for s in TL.SEGMENTS]}')
    harmony = Harmony(a.sections)
    cues, rows = collect_cues(scenes=not a.no_scenes, hud=not a.no_hud, harmony=harmony,
                              keyfit=not a.no_keyfit)
    print_report(rows, cues, harmony, a.cues)
    if a.check:
        return 0
    t1 = time.time()
    music, mstat = load_music(None if a.sfx_only else a.music)
    if a.sfx_only:
        mstat = 'off (--sfx-only)'
    refs, linfo = music_refs(music, a.sfx_db)
    if a.seg:
        s = TL.BY_ID[a.seg]
        lo, hi = s['start'] * SPF, s['end'] * SPF
        out = a.out or os.path.join(tempfile.gettempdir(), f'ogd_mix_{a.seg}.wav')
    else:
        lo, hi = 0, N_TOTAL
        out = a.out or FINAL_PATH
    y, info = render(cues, music, refs, lo, hi, sfx_only=a.sfx_only, music_only=a.music_only,
                     duck_db=a.duck_db)
    write_wav(out, y)
    t2 = time.time()
    v = verify(out, hi - lo)
    print(f'\nmusic: {mstat}')
    print(f'music loudness {linfo["music_lufs"]} LUFS -> SFX reference {linfo["sfx_ref"]} dB '
          f'(sfx-db {a.sfx_db:+g}, per-segment tracking {TRACK:.0%})')
    print(f'cues placed {info["placed"]}/{len(cues)}   duck max {info["duck_max_db"]} dB '
          f'({info["duck_time_pct"]}% of time > 1 dB)   limiter max {info["limiter_max_db"]} dB '
          f'({info["limiter_time_pct"]}% of time > 0.5 dB, ceiling {info["ceiling_db"]})')
    print(f'wrote {v["path"]}: {v["seconds"]} s, {v["shape"][0]} samples x {v["shape"][1]} ch, {v["dtype"]}, '
          f'{v["sr"]} Hz, {v["lufs"]} LUFS, sample peak {v["sample_peak_db"]} dBFS, '
          f'true peak {v["true_peak_db"]} dBTP')
    print('verify: ' + ('OK' if not v['problems'] else 'PROBLEMS: ' + '; '.join(v['problems'])) +
          f'   (cues {t1 - t0:.1f}s, mix {t2 - t1:.1f}s)')
    return 1 if v['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())
