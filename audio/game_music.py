"""EQUESTRIA ATLAS - the game soundtrack: seamless music loops for the explorable game.

    .venv/bin/python audio/game_music.py                # compose + render every track
    .venv/bin/python audio/game_music.py map_day race   # only these tracks
    .venv/bin/python audio/game_music.py --list         # the track list
    .venv/bin/python audio/game_music.py trailer        # the trailer cue (EXTRA, not a game loop): wav/trailer.wav
    .venv/bin/python audio/game_music.py --check        # compose only: melody / chord report, no audio
    .venv/bin/python audio/game_music.py --stems ...    # also print each role's level against the mix

Outputs `audio/out/game/<id>.ogg` (what the game plays), `audio/out/game/wav/<id>.wav` (for checks and the
trailer) and `audio/out/game/tracks.json` (id, title, bpm, bars, beats per bar, loop length in samples, LUFS).

Every track is one exact loop: notes are written on a bar grid of their own tempo and metre, rendered into a
buffer one tail longer than the loop, and the tail (reverb, echo, pad releases) is folded back onto the
start, so the last bar flows into the first without a seam. The master chain (EQ, glue compressor,
loudness, limiter) runs over three copies of the loop and keeps the middle one, so its state is seamless too.

The tracks themselves (melodies, chords, arrangements) are in `audio/game_tracks.py`. This module reuses the
film score's note helpers (`audio/compose.py`: pitches, chords, voicings, the Score builder, clash
resolution) and its synthesiser (`audio/synth.py`).
"""
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audio import compose as C          # noqa: E402

OUT = os.path.join(ROOT, 'audio', 'out', 'game')
SR = 48000
TAIL_S = 8.0                    # rendered past the loop end and folded back onto its start
TARGET_LUFS = -16.0             # a touch under the film (-14.5): the game plays sound effects over it
FFMPEG = os.path.join(ROOT, 'bin', 'ffmpeg')

# voices the film score does not have (registered into compose.VOICES so its clash resolver knows them)
EXTRA_VOICES = {
    'fiddle':    dict(inst='fiddle', pan=-0.08, echo=0.08, rev=0.16, role='melody', gate=0.95, glide=True),
    'accordion': dict(inst='accordion', pan=-0.1, echo=0.06, rev=0.16, role='melody', gate=0.95),
    'reed':      dict(inst='reed', pan=0.0, echo=0.14, rev=0.24, role='melody', gate=0.95, glide=True),
    'twang':     dict(inst='twang', pan=0.0, echo=0.22, rev=0.18, role='melody', gate=0.95, glide=True),
    'ocarina':   dict(inst='ocarina', pan=0.0, echo=0.18, rev=0.26, role='melody', gate=0.95, glide=True),
    'warm':      dict(inst='warm', pan=0.0, echo=0.18, rev=0.22, role='melody', gate=0.95, glide=True),
    'flute':     dict(inst='mellow', pan=0.05, echo=0.16, rev=0.24, role='melody', gate=0.95, glide=True),
    'counter':   dict(inst='thin', pan=-0.42, echo=0.12, rev=0.2, role='harmony', gate=0.92),
    'harm':      dict(inst='lead50', pan=0.38, echo=0.1, rev=0.2, role='harmony', gate=0.94),
    'kalimba':   dict(inst='kalimba', pan=0.3, echo=0.18, rev=0.3, role='ornament', gate=1.0),
    'sparkle':   dict(inst='glock', pan=-0.3, echo=0.22, rev=0.36, role='ornament', gate=1.0),
    'upright':   dict(inst='upright', pan=0.0, echo=0.0, rev=0.04, role='bass', gate=0.9),
    'pbass':     dict(inst='pulsebass', pan=0.0, echo=0.0, rev=0.02, role='bass', bus='duck', gate=0.8),
    'harp':      dict(inst='triarp', pan=0.32, echo=0.1, rev=0.34, role='accomp', gate=0.98),
    'strum':     dict(inst='mando', pan=-0.3, echo=0.04, rev=0.16, role='accomp', gate=0.9),
    'squeeze':   dict(inst='accordion', pan=0.3, echo=0.0, rev=0.14, role='accomp', gate=0.6),
}
C.VOICES.update({k: v for k, v in EXTRA_VOICES.items() if k not in C.VOICES})
MONO = set(C.MONO_VOICES) | {'fiddle', 'accordion', 'reed', 'twang', 'ocarina', 'warm', 'flute', 'counter',
                             'harm'}
# mix trims by role (dB)
ROLE_DB = {'melody': 2.0, 'harmony': -0.5, 'bass': -1.5, 'accomp': 1.5, 'ornament': -1.0, 'drums': -4.5,
           'fx': -4.0}
# and by voice (some instruments are quieter than others for the same velocity)
VOICE_DB = {'fiddle': 7.0, 'brass': 3.0, 'brass2': 2.0, 'accordion': 2.0, 'reed': 1.5}


class Track(C.Score):
    """A Score on its own tempo and metre: one section (the whole loop) of `bars` bars of `bpb` beats."""

    def __init__(self, tid, title, bpm, bars, bpb=4, tr=0, rev=1.0, lufs=TARGET_LUFS, echo_beats=0.75):
        super().__init__()
        self.tid, self.title, self.bpm, self.bars, self.bpb = tid, title, bpm, bars, bpb
        self.tr = tr                    # every mel / prog is written in C (or A minor) and moved by tr
        self.rev, self.lufs, self.echo_beats = rev, lufs, echo_beats
        self.loop = True                # False: a cue with an ending (the trailer): its tail rings out
        self.sec = dict(id=tid, start_beat=0, bars=bars, end_beat=bars * bpb)
        self._cbar = 0
        self.marks = []                 # (bar, name): the form, for the report

    # ---- grid
    def at(self, bar, beat=0.0):
        return bar * self.bpb + beat

    @property
    def beats(self):
        return self.bars * self.bpb

    def mark(self, bar, name):
        self.marks.append((bar, name))

    # ---- harmony: appended bar by bar
    def prog(self, spec):
        """Chords for the next bars: 'C | G/B | Am7:2 D7:2 | ...' (written in C, moved by tr)."""
        spec = C.trc(spec, self.tr, flat=self.tr in (1, 3, 5, 8, 10, -2, -4, -7, -9, -11)) if self.tr else spec
        for toks in [b.split() for b in spec.split('|')]:
            fixed = sum(float(t.split(':')[1]) for t in toks if ':' in t)
            free = [t for t in toks if ':' not in t]
            each = (self.bpb - fixed) / len(free) if free else 0.0
            pos = self.at(self._cbar)
            for tok in toks:
                sym, _, d = tok.partition(':')
                d = float(d) if d else each
                self.chords.append((pos, pos + d, C.Chord(sym), self.tid))
                pos += d
            assert abs(pos - self.at(self._cbar + 1)) < 1e-9, (self.tid, self._cbar, spec)
            self._cbar += 1

    # ---- melody (bars sum to the metre)
    def mel(self, voice, bar, spec, unit=0.5, vel=0.8, tr=0, swing=None, late=0.0, octave=0, **opts):
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
                    pb = (acc % self.bpb)
                    vv *= (1.0 if pb < 1e-6 else 0.94 if abs(pb - round(pb)) < 1e-6 else 0.88) * \
                        (1.03 if d >= 1.0 else 1.0)
                    o = dict(opts)
                    if '_' in mods:
                        o['gate'] = 0.5
                    pos = t + acc
                    if swing:
                        bb = math.floor(acc + 1e-9)
                        pos = t + bb + C.swing_pos(acc - bb, swing)
                    out.append(self.add(voice, pos + late, d, C.P(m.group(1)) + self.tr + tr + 12 * octave,
                                        vv, **o))
                acc += d
            if abs(acc - self.bpb) > 1e-6:
                raise ValueError(f'{self.tid}: bar {bar + bi} of a {voice} line lasts {acc} beats: {chunk!r}')
            t += self.bpb
        return out

    # ---- derived lines
    def double(self, events, voice, octave=0, vel=0.7, **opts):
        """Copy notes to another voice (octave doubling, an echo instrument...)."""
        out = []
        for e in events:
            out.append(self.add(voice, e['start_beat'], e['dur_beats'], e['midi'] + 12 * octave,
                                e['velocity'] * vel, **opts))
        return out

    def harmonize(self, events, voice, vel=0.72, below=True, **opts):
        """A consonant second line under (or over) a melody: the nearest chord tone 3-9 semitones away."""
        out = []
        for e in events:
            c = self.chord_at(e['start_beat'] + 1e-6)
            m0 = e['midi']
            best = None
            for d in (3, 4, 5, 8, 9):
                m = m0 - d if below else m0 + d
                if m % 12 in c.pcs:
                    best = m
                    break
            if best is None:
                continue
            out.append(self.add(voice, e['start_beat'], e['dur_beats'], best, e['velocity'] * vel, **opts))
        return out

    def drums(self, bar0, bar1, pats, swing=None, swing16=0.0, human=0.0, vel=1.0):
        """pats: {drum: pattern of 16ths, 4 x bpb steps per bar (X x o g .), a multiple of a bar long}."""
        lv = {'X': 1.0, 'x': 0.8, 'o': 0.55, 'g': 0.35}
        spb = 4 * self.bpb
        for name, p in pats.items():
            p = p.replace(' ', '')
            assert len(p) % spb == 0, (self.tid, name, p)
            nb = len(p) // spb
            for bar in range(bar0, bar1):
                seg_ = p[((bar - bar0) % nb) * spb:((bar - bar0) % nb + 1) * spb]
                for st, ch in enumerate(seg_):
                    if ch == '.':
                        continue
                    pos = st / 4
                    if swing:
                        bb = math.floor(pos + 1e-9)
                        pos = bb + C.swing_pos(pos - bb, swing)
                    if swing16 and st % 2 == 1:
                        pos += swing16
                    jit = human * (C._h(bar, st, len(name)) - 0.5) * 0.03 if human and st else 0.0
                    vv = lv[ch] * vel * (1 - human * 0.15 * C._h(st, bar, 3)) * (0.96 + 0.08 * C._h(bar, st, 11))
                    self.drum(name, self.at(bar) + pos + jit, vv)

    def pump(self, bar0, bar1, depth=0.3):
        self.duck.append((self.at(bar0), self.at(bar1), depth))


# --------------------------------------------------------------------------- tidy + checks
def tidy(T):
    """Monophonic voices never overlap their next note; nothing starts at or after the loop end."""
    by = {}
    for e in T.events:
        if e['voice'] in MONO and e['midi'] is not None and 'drum' not in e:
            by.setdefault(e['voice'], []).append(e)
    for evs in by.values():
        evs.sort(key=lambda e: e['start_beat'])
        for a, b in zip(evs, evs[1:]):
            over = a['start_beat'] + a['dur_beats'] - b['start_beat']
            if over > 1e-6 and b['start_beat'] > a['start_beat']:
                a['dur_beats'] = round(b['start_beat'] - a['start_beat'], 6)
    T.events = [e for e in T.events if 0 <= e['start_beat'] < T.beats - 1e-9]


def melody_report(T):
    """Long or strong melody notes that rub against their chord (a semitone over a chord tone)."""
    bad = []
    for e in T.events:
        if C.VOICES.get(e['voice'], {}).get('role') != 'melody' or e['midi'] is None:
            continue
        pos = e['start_beat'] % T.bpb
        strong = abs(pos - round(pos)) < 1e-6
        if not strong and e['dur_beats'] < 1.0:
            continue
        c = T.chord_at(e['start_beat'] + 1e-6)
        if c.classify(e['midi']) == 'avoid' and e['dur_beats'] >= 0.5:
            bar = int(e['start_beat'] // T.bpb)
            bad.append(f"bar {bar + 1} beat {pos + 1:g}: {e['voice']} {C.pname(e['midi'])} over {c.sym}")
    return bad


def finish(T):
    C.dedupe_drums(T)
    tidy(T)
    nfix = C.resolve_clashes(T)
    C.assign_glides(T.events)
    T.events.sort(key=lambda e: (e['start_beat'], e['voice']))
    return nfix


# --------------------------------------------------------------------------- render
def render(T, verbose=True, stems=False):
    import time
    import numpy as np
    from audio import synth as Y
    for k_, v_ in C.CUSTOM_INST.items():
        Y.INSTRUMENTS.setdefault(k_, v_)
    t0 = time.time()
    spb = SR * 60.0 / T.bpm
    sec_per_beat = 60.0 / T.bpm
    N = int(round(T.beats * spb))
    NT = N + int(TAIL_S * SR)
    main = np.zeros((2, NT), np.float32)
    duck = np.zeros((2, NT), np.float32)
    uw = np.zeros((2, NT), np.float32)
    esend = np.zeros(NT, np.float32)
    rsend = np.zeros((2, NT), np.float32)
    kicks = []
    roles = {}
    for i, e in enumerate(T.events):
        o = dict(e['_o'])
        inst = o.pop('inst', None)
        v = C.VOICES.get(e['voice'], {})
        gate = o.pop('gate', v.get('gate', 0.95))
        pan = o.pop('pan', None)
        bus_o = o.pop('bus', None)
        gain_db = o.pop('db', 0.0)
        seed = 1000 + i * 7 + hash(T.tid) % 997
        if inst and inst.startswith('drum:'):
            name = inst[5:]
            y = Y.drum(name, e['velocity'], seed=seed)
            dv = C.DRUM_VOICE.get(name, name)
            pan_, rv = C.DRUM_MIX.get(dv, (0.0, 0.05))
            pan = pan_ if pan is None else pan
            ev_echo, bus, role = 0.0, 'main', 'drums'
            if dv == 'kick':
                kicks.append(int(round(e['start_beat'] * spb)))
        elif inst and inst.startswith('fx:'):
            name = inst[3:]
            y = Y.fx(name, e['dur_beats'] * sec_per_beat, e['velocity'], seed=seed, **o)
            pan = 0.0 if pan is None else pan
            rv, ev_echo, bus, role = (0.25 if name != 'crackle' else 0.0), 0.0, 'main', 'fx'
        else:
            midi = e['midi']
            sl = o.pop('slide', None)
            if sl:
                midi -= sl
                o.update(bend=sl, bend_time=0.06)
            y = Y.render_note(inst, midi, e['dur_beats'] * sec_per_beat * gate, e['velocity'], seed=seed, **o)
            pan = v.get('pan', 0.0) if pan is None else pan
            rv, ev_echo, bus, role = v.get('rev', 0.1), v.get('echo', 0.0), v.get('bus', 'main'), v.get('role')
        if bus_o:
            bus = bus_o
            if bus == 'uw':
                rv *= 2.0
                ev_echo = max(ev_echo, 0.3)
        rv *= T.rev
        y = y * 10 ** ((ROLE_DB.get(role, 0.0) + VOICE_DB.get(e['voice'], 0.0) + gain_db) / 20)
        gl, gr = Y.pan_gains(pan)
        s0 = int(round(e['start_beat'] * spb))
        if s0 >= N:
            continue
        if y.ndim == 1:
            n = min(len(y), NT - s0)
            st = np.stack([y[:n] * gl, y[:n] * gr])
            mono = y[:n]
        else:
            n = min(y.shape[1], NT - s0)
            st = y[:, :n] * np.array([[gl], [gr]])
            mono = y[:, :n].mean(0)
        (duck if bus == 'duck' else uw if bus == 'uw' else main)[:, s0:s0 + n] += st
        if ev_echo:
            esend[s0:s0 + n] += mono * ev_echo
        if rv:
            rsend[:, s0:s0 + n] += st * rv
        if stems:
            r_ = roles.setdefault(role, np.zeros((2, NT), np.float32))
            r_[:, s0:s0 + n] += st
    t1 = time.time()
    if T.duck:
        g = np.ones(NT)
        for b0, b1, depth in T.duck:
            s0, s1 = int(b0 * spb), int(b1 * spb)
            hits = [k - s0 for k in kicks if s0 <= k < s1]
            g[s0:s1] = Y.duck_curve(s1 - s0, hits, depth=depth)
        duck *= g.astype(np.float32)
    echo = Y.pingpong_echo(esend.astype(np.float64), delay_s=T.echo_beats * sec_per_beat, feedback=0.34,
                           lp=3600, hp=380)
    ir = Y.make_reverb_ir(2.4, (1.9, 1.7, 0.85), seed=11)
    rev = Y.reverb(rsend.astype(np.float64) + 0.25 * echo, ir)[:, :NT]
    uwf = Y.lowpass(uw.astype(np.float64), 1100.0, order=2) * 1.25
    mix = main.astype(np.float64) + duck + uwf + 0.5 * echo + 0.42 * rev
    if not T.loop:                      # a cue with an ending: the tail rings out after the last bar
        y, info = master_once(mix, T.lufs, Y)
        if stems:
            tot = float(np.sqrt(np.mean(mix ** 2)) + 1e-12)
            info['stems'] = {r: round(20 * math.log10((float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12)
                                                       / tot), 1) for r, x in sorted(roles.items(), key=str)}
        if verbose:
            print(f'  {T.tid}: {len(T.events)} notes, {N / SR:.1f} s + tail, {info["lufs"]} LUFS, '
                  f'peak {info["true_peak"]} dBTP')
        return y, info, N
    # fold the tail back onto the start: the loop's end rings into its own beginning
    loop = mix[:, :N].copy()
    tail = mix[:, N:]
    k = 0
    while k < tail.shape[1]:
        n = min(N, tail.shape[1] - k)
        loop[:, :n] += tail[:, k:k + n]
        k += n
    y, info = master_loop(loop, T.lufs, Y)
    t2 = time.time()
    if stems:
        tot = float(np.sqrt(np.mean(mix[:, :N] ** 2)) + 1e-12)
        info['stems'] = {r: round(20 * math.log10((float(np.sqrt(np.mean(x[:, :N].astype(np.float64) ** 2))) + 1e-12)
                                                   / tot), 1) for r, x in sorted(roles.items(), key=str)}
    if verbose:
        print(f'  {T.tid}: {len(T.events)} notes, {N / SR:.1f} s loop, render {t1 - t0:.1f}s + mix {t2 - t1:.1f}s, '
              f'{info["lufs"]} LUFS, peak {info["true_peak"]} dBTP')
    return y, info, N


def master_loop(loop, target, Y):
    import numpy as np
    N = loop.shape[1]
    x = np.concatenate([loop, loop, loop], axis=1)
    x = Y.highpass(x, 24.0, order=4)
    x = Y.eq(x, 'lowshelf', 110.0, -1.5, 0.7)
    x = Y.eq(x, 'peak', 300.0, -1.0, 0.9)
    x = Y.eq(x, 'highshelf', 9000.0, -1.5, 0.7)
    x -= x.mean(axis=1, keepdims=True)
    x, gr = Y.glue_comp(x, thresh_db=-20.0, ratio=1.6, attack=0.015, release=0.3)
    L = Y.lufs(x[:, N:2 * N])
    x *= 10 ** ((target - L) / 20)
    ceiling = -1.5
    for _ in range(4):
        y, g = Y.limiter(x, ceiling_db=ceiling, lookahead=0.003, hold=0.02)
        tp = Y.true_peak_db(y[:, N:2 * N])
        if tp <= -1.2:
            break
        ceiling -= (tp + 1.2) + 0.05
    y = y[:, N:2 * N]
    return y, dict(lufs=round(Y.lufs(y), 2), true_peak=round(tp, 2))


def master_once(x, target, Y):
    """The master chain for a cue with an ending; the tail fades to silence over its last 3 s."""
    import numpy as np
    x = Y.highpass(x, 24.0, order=4)
    x = Y.eq(x, 'lowshelf', 110.0, -1.5, 0.7)
    x = Y.eq(x, 'peak', 300.0, -1.0, 0.9)
    x = Y.eq(x, 'highshelf', 9000.0, -1.5, 0.7)
    x -= x.mean(axis=1, keepdims=True)
    x, gr = Y.glue_comp(x, thresh_db=-20.0, ratio=1.6, attack=0.015, release=0.3)
    x *= 10 ** ((target - Y.lufs(x)) / 20)
    n = int(3.0 * SR)
    x[:, -n:] *= np.linspace(1, 0, n) ** 2
    ceiling = -1.5
    for _ in range(4):
        y, g = Y.limiter(x, ceiling_db=ceiling, lookahead=0.003, hold=0.02)
        tp = Y.true_peak_db(y)
        if tp <= -1.2:
            break
        ceiling -= (tp + 1.2) + 0.05
    return y, dict(lufs=round(Y.lufs(y), 2), true_peak=round(tp, 2))


def write(T, y, info, N):
    import numpy as np
    from audio import synth as Y
    os.makedirs(os.path.join(OUT, 'wav'), exist_ok=True)
    wav = os.path.join(OUT, 'wav', f'{T.tid}.wav')
    Y.write_wav(wav, Y.to_int16(y, dither=True, dither_mask=np.ones(y.shape[1])))
    if not T.loop:                      # (not a game track: the WAV is for the trailer)
        print('  wrote', wav)
        return None
    ogg = os.path.join(OUT, f'{T.tid}.ogg')
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', wav, '-c:a', 'libvorbis', '-q:a', '6', ogg],
                   check=True)
    return dict(id=T.tid, title=T.title, bpm=T.bpm, bars=T.bars, bpb=T.bpb, samples=N, seconds=round(N / SR, 3),
                lufs=info['lufs'], true_peak=info['true_peak'], form=T.marks)


def build(tid, check_only=False):
    from audio import game_tracks as GT
    T = (GT.TRACKS.get(tid) or GT.EXTRA[tid])()
    nfix = finish(T)
    bad = melody_report(T)
    if check_only or bad:
        print(f'  {tid}: {len(T.events)} events, {T.bars} bars x {T.bpb} @ {T.bpm} BPM '
              f'({T.beats * 60 / T.bpm:.1f} s), {nfix} accompaniment clashes fixed')
        for b in bad:
            print('     rub:', b)
    if check_only:
        return None
    y, info, N = render(T, stems=STEMS)
    if STEMS:
        print('     role levels vs mix (dB):', info['stems'])
    return write(T, y, info, N)


STEMS = False


def main(argv):
    global STEMS
    from audio import game_tracks as GT
    STEMS = '--stems' in argv
    if '--list' in argv:
        for k in GT.TRACKS:
            print(k)
        return
    check = '--check' in argv
    ids = [a for a in argv if not a.startswith('--')] or list(GT.TRACKS)
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, 'tracks.json')
    man = {}
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = {t['id']: t for t in json.load(f)}
    for tid in ids:
        r = build(tid, check_only=check)
        if r:
            man[tid] = r
    if not check:
        order = [k for k in GT.TRACKS if k in man]
        with open(man_path, 'w') as f:
            json.dump([man[k] for k in order], f, indent=1)
        print('wrote', man_path)


if __name__ == '__main__':
    main(sys.argv[1:])
