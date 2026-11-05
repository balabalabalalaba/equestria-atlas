"""EQUESTRIA ATLAS - the game soundtrack's tracks (rendered by audio/game_music.py).

Every track is a seamless loop. Melodies and chords are written in compact notation (see Track.mel and
Track.prog in audio/game_music.py; '|' separates bars, durations in eighths unless `unit` says otherwise).
The place themes are written in C major / A minor and moved to their key with `tr`, so their lines read
alike; the title, the map themes and the race theme are written in their own keys.

    title           The World Atlas          D major, 84 BPM: the Atlas theme on a music box, then the band
    map_day         Wind in the Envelope     G major, 104 BPM, 48 bars: the overworld by day
    map_night       Lanterns over Equestria  the same 48 bars arranged as a nocturne; the game crossfades the two
                                             with the world clock, in sync (same tempo, length and chords)
    race            Ring Rush                A minor, 138 BPM: ring races and balloon-friend races
    <place>         one theme per place, in its own style (see PLACES)
"""
from audio.game_music import Track

# =========================================================================== style kits
# Each kit lays down bass, harmony and drums for bars [b0, b1) at an energy e: 0 light, 1 normal, 2 full.


def _drums(T, b0, b1, pats, **kw):
    T.drums(b0, b1, {k: v for k, v in pats.items() if v}, **kw)


def crash(T, bar, e):
    if e >= 2:
        T.drum('crash', T.at(bar), 0.75)
    elif e >= 1:
        T.drum('crash_soft', T.at(bar), 0.6)


def fill(T, bar, kind='snare', vel=0.6):
    """A one-beat fill at the end of `bar` (leads into the next section)."""
    b = T.at(bar + 1) - 1
    if kind == 'snare':
        T.roll('snare_soft', b, b + 1, 0.25, vel * 0.4, vel)
    elif kind == 'tom':
        for i, t in enumerate(('tom_hi', 'tom_hi', 'tom_mid', 'tom_lo')):
            T.drum(t, b + i * 0.25, vel * (0.7 + 0.1 * i))
    elif kind == 'brush':
        T.roll('brush', b, b + 1, 0.25, vel * 0.3, vel * 0.8)


def k_folk(T, b0, b1, e, swing=None):
    if e >= 1:
        T.bassline('upright', b0, b1, [(0, 'R', 1.5), (2, '5', 1.5)] + ([(3.5, 'R', 0.5, 0.55)] if e >= 2 else []),
                   lo=38, vel=0.72, swing=swing)
        T.comp('strum', b0, b1, [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)],
               lo=55, hi=72, n=3, vel=0.5, swing=swing, human=0.6)
    else:
        T.bassline('upright', b0, b1, [(0, 'R', 3.5)], lo=38, vel=0.6)
    T.pad('bed' if e < 2 else 'pad', b0, b1, lo=50, hi=67, n=3, vel=0.32 if e < 2 else 0.28)
    _drums(T, b0, b1, dict(
        shaker='g.o.g.o.g.o.g.o.' if e >= 1 else 'g...g...g...g...',
        kick_soft='X.......x.......' if e >= 1 else None,
        snare_soft='....x.......x...' if e >= 1 else None,
        hat_soft='..g...g...g...g.' if e >= 2 else None), swing=swing, human=0.5, vel=0.8 + 0.1 * e)


def k_hoedown(T, b0, b1, e, swing=None):
    T.bassline('upright', b0, b1, [(0, 'R', 0.9), (2, '5', 0.9)], lo=38, vel=0.8)
    T.comp('strum', b0, b1, [(1, 0.4, 0.8), (3, 0.4, 0.8)], lo=55, hi=72, n=3, vel=0.55)
    if e >= 1:
        T.arp('banjo', T.at(b0), T.at(b1), step=0.5, lo=62, hi=81, vel=0.5, group=[0, 2, 4, 2, 1, 3, 4, 3],
              pans=(0.35,))
    _drums(T, b0, b1, dict(kick='X.......X.......', snare_soft='....x.......x...' if e >= 1 else None,
                           brush='g.o.g.o.g.o.g.o.' if e >= 1 else 'g...g...g...g...',
                           hat_soft='..o...o...o...o.' if e >= 2 else None), human=0.4, vel=0.8 + 0.1 * e)


def k_march(T, b0, b1, e, swing=None):
    T.bassline('bass', b0, b1, [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.7)
    T.arp('harpsi', T.at(b0), T.at(b1), step=0.5, lo=62, hi=81, vel=0.45 if e else 0.35, pattern='updown',
          pans=(0.3, 0.2))
    T.pad('pad', b0, b1, lo=50, hi=67, n=4, vel=0.26 + 0.05 * e)
    _drums(T, b0, b1, dict(kick='X.......X.......' if e >= 1 else None,
                           snare_soft='x..ox.o.x..ox.o.' if e >= 1 else '....o.......o...',
                           hat_soft='o.o.o.o.o.o.o.o.' if e >= 2 else None), human=0.3, vel=0.75 + 0.1 * e)
    if e >= 2:
        for b in range(b0, b1, 2):
            T.add('timp', T.at(b), 1.0, 26 + (T.chord_at(T.at(b)).root - 26) % 12 + 12, 0.7)


def k_airy(T, b0, b1, e, swing=None):
    if e >= 1:
        T.bassline('bass', b0, b1, [(0, 'R', 3.6)], lo=36, vel=0.55)
    T.arp('harp', T.at(b0), T.at(b1), step=0.25 if e >= 1 else 0.5, lo=64, hi=86, vel=0.36, pattern='updown',
          pans=(-0.4, 0.4))
    T.pad('bed', b0, b1, lo=52, hi=69, n=4, vel=0.34)
    _drums(T, b0, b1, dict(shaker='g.g.o.g.g.g.o.g.', kick_soft='x.......x.......' if e >= 1 else None,
                           rim='........o.......' if e >= 1 else None,
                           hat_soft='..g...g...g...g.' if e >= 2 else None), human=0.5, vel=0.75)


def k_pop(T, b0, b1, e, swing=None):
    T.bassline('bass', b0, b1, [(0, 'R', 0.9), (1, 'R', 0.4, 0.6), (1.5, 'R', 0.4, 0.7), (2, '5', 0.9),
                                (3, 'R+12', 0.4, 0.6), (3.5, '5', 0.4, 0.65)] if e >= 1 else [(0, 'R', 3.6)],
               lo=33, vel=0.72)
    if e >= 1:
        T.comp('stab', b0, b1, [(0.5, 0.3, 0.7), (1.5, 0.3, 0.6), (2.5, 0.3, 0.7), (3.5, 0.3, 0.6)],
               lo=57, hi=74, n=3, vel=0.5)
    T.pad('pad', b0, b1, lo=50, hi=67, n=4, vel=0.3)
    _drums(T, b0, b1, dict(kick='X.....x.X.......' if e >= 1 else 'X.......X.......',
                           snare='....X.......X...' if e >= 1 else None,
                           hat='x.o.x.o.x.o.x.o.' if e >= 1 else 'o...o...o...o...',
                           ohat='..............o.' if e >= 2 else None), human=0.3, vel=0.75 + 0.08 * e)


def k_rock(T, b0, b1, e, swing=None):
    T.bassline('pbass', b0, b1, [(i * 0.5, 'R' if i % 4 != 3 else 'R+12', 0.45) for i in range(8)], lo=33,
               vel=0.75)
    if e >= 1:
        T.comp('power', b0, b1, [(0, 0.4, 1.0), (0.75, 0.25, 0.7), (1.5, 0.4, 0.8), (2.5, 0.4, 0.8),
                                 (3, 0.4, 0.9), (3.5, 0.4, 0.7)], lo=50, hi=64, n=3, vel=0.5)
    if e >= 2:
        T.arp('arp', T.at(b0), T.at(b1), step=0.25, lo=69, hi=88, vel=0.42, pattern='up')
    T.pump(b0, b1, 0.25)
    _drums(T, b0, b1, dict(kick='X...x.X.X...x...' if e >= 1 else 'X.......X.......',
                           snare='....X.......X...', clap='....x.......x...' if e >= 2 else None,
                           hat='xoxoxoxoxoxoxoxo' if e >= 1 else 'x.x.x.x.x.x.x.x.',
                           ohat='..............x.' if e >= 2 else None), human=0.2, vel=0.8 + 0.06 * e)


def walk(T, b0, b1, voice='upright', lo=36, vel=0.75, swing=None):
    """Walking bass: root, a chord tone, the fifth, then a step into the next bar's root."""
    for bar in range(b0, b1):
        c = T.chord_at(T.at(bar))
        nxt = T.chord_at(T.at(bar + 1) if bar + 1 < T.bars else 0)
        r = lo + (c.bass_pc - lo) % 12
        third = r + (3 if 3 in c.ivs else 4)
        fifth = r + (6 if 6 in c.ivs and 7 not in c.ivs else 7)
        nr = lo + (nxt.bass_pc - lo) % 12
        app = nr - 1 if abs(nr - 1 - fifth) <= abs(nr + 1 - fifth) else nr + 1
        if c is not T.chord_at(T.at(bar, 2)):            # two chords in the bar: root, fifth / root, fifth
            c2 = T.chord_at(T.at(bar, 2))
            r2 = lo + (c2.bass_pc - lo) % 12
            notes = [r, fifth, r2, app]
        else:
            notes = [r, third, fifth, app]
        for i, m in enumerate(notes):
            T.add(voice, T.at(bar, i), 0.9, m, vel * (1.0 if i == 0 else 0.85))


def k_swing(T, b0, b1, e, swing=0.64):
    walk(T, b0, b1, vel=0.72)
    T.comp('keys', b0, b1, [(0, 0.9, 0.8), (1.5, 0.5, 0.7)] if e < 2 else [(1, 0.5, 0.8), (2.5, 0.9, 0.75)],
           lo=55, hi=72, n=4, vel=0.48, swing=swing, human=0.6)
    _drums(T, b0, b1, dict(ride='x...x.o.x...x.o.', hat_soft='....x.......x...',
                           snare_soft='....g.......x..g' if e >= 1 else None,
                           kick_soft='o.......g.......' if e >= 1 else None,
                           clap='....x.......x...' if e >= 2 else None), swing=swing, human=0.5, vel=0.8)


def k_shuffle(T, b0, b1, e, swing=0.66):
    T.bassline('upright', b0, b1, [(0, 'R', 0.9), (1, '5', 0.9, 0.8), (2, 'R+12', 0.9, 0.85), (3, '5', 0.9, 0.8)],
               lo=36, vel=0.72, swing=swing)
    T.comp('squeeze', b0, b1, [(0.5, 0.4, 0.8), (1.5, 0.4, 0.65), (2.5, 0.4, 0.8), (3.5, 0.4, 0.65)],
           lo=55, hi=72, n=3, vel=0.42, swing=swing)
    T.pad('bed', b0, b1, lo=50, hi=67, n=3, vel=0.28)
    _drums(T, b0, b1, dict(kick_soft='X.......x.......', brush='....x.......x...' if e >= 1 else None,
                           shaker='o.g.o.g.o.g.o.g.', hat_soft='..o...o...o...o.' if e >= 2 else None),
           swing=swing, human=0.5, vel=0.8)


def k_mystic(T, b0, b1, e, swing=None):
    T.bassline('upright', b0, b1, [(0, 'R', 3.8)], lo=33, vel=0.6)
    T.arp('kalimba', T.at(b0), T.at(b1), step=0.5, lo=57, hi=76, vel=0.42, group=[0, 2, 1, 3, 0, 2, 4, 3],
          pans=(0.3, -0.2))
    T.pad('bed', b0, b1, lo=48, hi=64, n=4, vel=0.38)
    if e >= 1:
        T.pad('choir', b0, b1, lo=57, hi=72, n=3, vel=0.22)
    _drums(T, b0, b1, dict(tom_lo='o.......o..g....' if e >= 1 else 'o...............',
                           tick='..g...g...g...g.' if e >= 1 else None,
                           shaker='g.g.g.g.g.g.g.g.' if e >= 2 else None), human=0.6, vel=0.7)


def k_heavy(T, b0, b1, e, swing=None):
    T.bassline('pbass', b0, b1, [(0, 'R', 0.45), (0.5, 'R', 0.45), (1, 'R+12', 0.45), (1.5, 'R', 0.45),
                                 (2, '3', 0.45, 0.8), (2.5, 'R', 0.45), (3, 'b7', 0.45, 0.8), (3.5, 'R', 0.45)],
               lo=33, vel=0.78)
    if e >= 1:
        T.comp('power', b0, b1, [(0, 0.9, 1.0), (1.5, 0.4, 0.75), (2, 0.9, 0.9), (3.5, 0.4, 0.75)],
               lo=48, hi=62, n=2, vel=0.48, lp=2600)
    T.pad('pad', b0, b1, lo=48, hi=64, n=3, vel=0.28)
    _drums(T, b0, b1, dict(kick='X..x..X.X..x....' if e >= 1 else 'X.......X.......',
                           snare='........X.......' if e < 2 else '....X.......X...',
                           tom_lo='..........x...x.' if e >= 1 else None,
                           hat='x.x.x.x.x.x.x.x.' if e >= 1 else None), human=0.2, vel=0.82)
    for b in range(b0, b1, 2 if e >= 1 else 4):
        T.add('timp', T.at(b), 1.0, 26 + (T.chord_at(T.at(b)).root - 26) % 12 + 12, 0.65)


def k_waltz(T, b0, b1, e, swing=None):
    T.bassline('bass', b0, b1, [(0, 'R', 0.9)], lo=36, vel=0.62)
    T.comp('keys', b0, b1, [(1, 0.8, 0.7), (2, 0.8, 0.6)], lo=57, hi=74, n=3, vel=0.42)
    if e >= 1:
        T.arp('harp', T.at(b0), T.at(b1), step=0.5, lo=64, hi=86, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.pad('choir' if e >= 1 else 'bed', b0, b1, lo=55, hi=72, n=3, vel=0.24)
    _drums(T, b0, b1, dict(kick_soft='X...........' if e >= 1 else None, hat_soft='....o...o...',
                           tick='..g...g...g.' if e >= 2 else None), human=0.4, vel=0.7)


def k_funk(T, b0, b1, e, swing=None):
    T.bassline('pbass', b0, b1, [(0, 'R', 0.4), (0.75, 'R', 0.2, 0.7), (1.5, 'R+12', 0.25, 0.8), (2, 'b7', 0.4, 0.8),
                                 (2.75, 'R', 0.2, 0.7), (3, '5', 0.4, 0.85), (3.5, 'R+12', 0.25, 0.8)],
               lo=33, vel=0.78)
    T.comp('keys', b0, b1, [(0.5, 0.3, 0.8), (1.75, 0.2, 0.6), (2.5, 0.3, 0.8), (3.75, 0.2, 0.6)],
           lo=57, hi=74, n=4, vel=0.5)
    if e >= 1:
        T.comp('stab', b0, b1, [(1, 0.2, 0.8), (3, 0.2, 0.8)], lo=64, hi=79, n=3, vel=0.42)
    T.pump(b0, b1, 0.2)
    _drums(T, b0, b1, dict(kick='X..x..x...X..x..', snare='....X.......X...' if e >= 1 else '............X...',
                           clap='....x.......x...' if e >= 1 else None,
                           hat='xoxoxoxoxoxoxoxo' if e >= 1 else 'x.x.x.x.x.x.x.x.',
                           ohat='..............x.' if e >= 2 else None), human=0.3, vel=0.8)


def k_lofi(T, b0, b1, e, swing=None):
    T.bassline('upright', b0, b1, [(0, 'R', 1.4), (1.5, '5', 0.5, 0.7), (2.5, 'R', 1.4, 0.85)], lo=36, vel=0.7)
    T.comp('keys', b0, b1, [(0, 1.4, 0.8), (1.75, 0.5, 0.55), (2.5, 1.4, 0.7)], lo=55, hi=74, n=4, vel=0.5,
           human=0.8)
    T.pad('bed', b0, b1, lo=50, hi=67, n=3, vel=0.25)
    _drums(T, b0, b1, dict(kick_lofi='X......x..X.....' if e >= 1 else 'X.........x.....',
                           snare_lofi='....X.......X...' if e >= 1 else None,
                           hat_soft='x.o.x.o.x.o.x.o.' if e >= 1 else 'o...o...o...o...',
                           shaker='..g...g...g...g.' if e >= 2 else None), swing16=0.045, human=0.7, vel=0.75)


def k_bazaar(T, b0, b1, e, swing=None):
    T.bassline('upright', b0, b1, [(0, 'R', 0.9), (1.5, 'R', 0.4, 0.7), (2, '5', 0.9, 0.8), (3.5, 'R', 0.4, 0.7)],
               lo=36, vel=0.72)
    T.arp('stab', T.at(b0), T.at(b1), step=0.5, lo=57, hi=76, vel=0.45, group=[0, 1, 2, 1, 3, 2, 1, 2],
          pans=(0.42, 0.3))
    T.pad('bed', b0, b1, lo=50, hi=65, n=3, vel=0.3)
    _drums(T, b0, b1, dict(tom_hi='X..x..x.....x...' if e >= 1 else 'X.....x.........',
                           tom_mid='..........x.....' if e >= 1 else None,
                           rim='..o...o.o...o.o.', clap='....x.......x...' if e >= 2 else None,
                           shaker='g.g.g.g.g.g.g.g.' if e >= 1 else None), human=0.5, vel=0.75)


def k_stomp(T, b0, b1, e, swing=None):
    T.bassline('bass', b0, b1, [(0, 'R', 0.9), (2, '5', 0.9, 0.85)], lo=33, vel=0.78, inst='bassoct')
    T.comp('squeeze', b0, b1, [(1, 0.5, 0.85), (3, 0.5, 0.85)] + ([(1.5, 0.3, 0.55), (3.5, 0.3, 0.55)] if e >= 2
                                                                   else []), lo=55, hi=72, n=3, vel=0.46)
    T.pad('bed', b0, b1, lo=50, hi=65, n=3, vel=0.26)
    _drums(T, b0, b1, dict(boom='X.......X.......', clap='....X.......X...' if e >= 1 else '....x.......x...',
                           tom_lo='......x.......x.' if e >= 1 else None,
                           shaker='o.g.o.g.o.g.o.g.' if e >= 2 else None), human=0.3, vel=0.8)
    if e >= 2:
        for b in range(b0, b1, 2):
            T.add('timp', T.at(b), 1.0, 26 + (T.chord_at(T.at(b)).root - 26) % 12 + 12, 0.6)


def k_lullaby(T, b0, b1, e, swing=None):
    T.arp('harp', T.at(b0), T.at(b1), step=0.5, lo=57, hi=81, vel=0.36, pattern='updown', pans=(-0.3, 0.3))
    if e >= 1:
        T.bassline('upright', b0, b1, [(0, 'R', 2.8)], lo=38, vel=0.55)
    T.pad('choir' if e >= 1 else 'bed', b0, b1, lo=55, hi=72, n=3, vel=0.24 + 0.04 * e)
    for b in range(b0, b1, 2):
        c = T.chord_at(T.at(b))
        T.add('bell', T.at(b), 2.0, 72 + (c.root - 72) % 12, 0.32 + 0.06 * e)
    if e >= 2:
        _drums(T, b0, b1, dict(tick='o...g...g...', kick_soft='x...........'), human=0.4, vel=0.6)


def k_medi(T, b0, b1, e, swing=None):
    T.bassline('upright', b0, b1, [(0, 'R', 0.9), (1.5, '5', 0.4, 0.7), (2, 'R', 0.9, 0.85), (3, '5', 0.9, 0.8)],
               lo=38, vel=0.7)
    T.comp('strum', b0, b1, [(i * 0.25, 0.22, 0.75 if i % 4 == 0 else 0.5) for i in range(16)], lo=60, hi=76, n=2,
           vel=0.36)
    T.pad('bed', b0, b1, lo=50, hi=67, n=3, vel=0.28)
    _drums(T, b0, b1, dict(kick_soft='X.....x.X.......' if e >= 1 else 'X.......X.......',
                           hat_soft='g.o.g.o.g.o.g.o.', rim='....x.......x...' if e >= 1 else None,
                           shaker='..o...o...o...o.' if e >= 2 else None), human=0.5, vel=0.78)


KITS = dict(folk=k_folk, hoedown=k_hoedown, march=k_march, airy=k_airy, pop=k_pop, rock=k_rock, swing=k_swing,
            shuffle=k_shuffle, mystic=k_mystic, heavy=k_heavy, waltz=k_waltz, funk=k_funk, lofi=k_lofi,
            bazaar=k_bazaar, stomp=k_stomp, lullaby=k_lullaby, medi=k_medi)


# =========================================================================== the standard place form
def place(tid, title, bpm, tr, kit, sections, bpb=4, swing=None, rev=1.0, extra=None):
    """sections: list of dicts: name, prog, mel (or None), voice, e (energy), harm (voice for a line under the
    melody), dbl ((voice, octave) doubling), bus ('uw' underwater), vel, fill ('snare'/'tom'/'brush' at the
    end of the section)."""
    def make():
        bars = sum(len(s['prog'].split('|')) for s in sections)
        T = Track(tid, title, bpm, bars, bpb=bpb, tr=tr, rev=rev)
        b = 0
        for s in sections:
            n = len(s['prog'].split('|'))
            T.mark(b, s['name'])
            T.prog(s['prog'])
            b += n
        b = 0
        for s in sections:
            n = len(s['prog'].split('|'))
            e = s.get('e', 1)
            KITS[s.get('kit', kit)](T, b, b + n, e, swing=swing)
            crash(T, b, e if s.get('crash', True) else 0)
            if s.get('mel'):
                o = dict(bus=s['bus']) if s.get('bus') else {}
                ev = T.mel(s.get('voice', 'lead'), b, s['mel'], vel=s.get('vel', 0.8), swing=swing,
                           unit=s.get('unit', 0.5), tr=s.get('tr', 0), **o)
                if s.get('harm'):
                    T.harmonize(ev, s['harm'], **o)
                if s.get('dbl'):
                    v, octv = s['dbl']
                    T.double(ev, v, octave=octv, vel=s.get('dbl_vel', 0.55), **o)
            if s.get('fill'):
                fill(T, b + n - 1, s['fill'])
            b += n
        if extra:
            extra(T)
        return T
    return make


TRACKS = {}

# =========================================================================== title
ATLAS_A = 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 E5:4'


def title():
    T = Track('title', 'The World Atlas', 84, 20, rev=1.35)
    T.prog('D | Gmaj7 | Em7 | A7sus4:2 A7:2')                       # 0 intro
    T.prog('D | Gmaj7 | Em7 | A7sus4:2 A7:2')                       # 4 theme
    T.prog('G | Bm7 | Em7 | A7')                                    # 8 answer
    T.prog('D | Gmaj7 | Em7 | A7:2 D:2')                            # 12 theme again
    T.prog('Gmaj7 | D/F# | Em7 | A7sus4:2 A7:2')                    # 16 coda -> intro
    for b, n in ((0, 'intro'), (4, 'theme'), (8, 'answer'), (12, 'theme 2'), (16, 'coda')):
        T.mark(b, n)
    T.add('bell', 0, 4, 74, 0.45)
    T.mel('mbox', 0, 'r:4 F#5:2 A5:2 | B5:6 A5:2 | G5:4 E5:2 B4:2 | D5:4 C#5:2 E5:2', vel=0.7)
    T.arp('harp', 0, T.at(20), step=0.5, lo=62, hi=83, vel=0.3, pattern='updown', pans=(-0.35, 0.35))
    T.pad('bed', 0, 20, lo=50, hi=67, n=4, vel=0.34)
    T.pad('choir', 4, 16, lo=57, hi=72, n=3, vel=0.2)
    T.bassline('upright', 4, 20, [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.6)
    a = T.mel('warm', 4, ATLAS_A, vel=0.82)
    T.double(a, 'sparkle', octave=1, vel=0.25)
    T.mel('flute', 8, 'B5:3 A5 G5:2 F#5 G5 | A5:3 G5 F#5:2 E5 F#5 | G5:2 F#5 E5 D5:2 E5 F#5 | E5:3 D5 C#5:2 A4:2',
          vel=0.8)
    T.mel('counter', 8, 'D5:4 G4:4 | F#4:4 D4:4 | E4:4 B4:4 | C#5:4 E5:4', vel=0.5)
    a2 = T.mel('warm', 12, 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 D5:4',
               vel=0.86)
    T.harmonize(a2, 'counter', vel=0.6)
    T.mel('mbox', 16, 'B5:3 A5 F#5:4 | A5:3 F#5 D5:4 | G5:3 F#5 E5:2 B4:2 | D5:4 C#5:4', vel=0.66)
    T.drums(8, 16, {'brush': '....o.......o...', 'kick_soft': 'o.......g.......', 'shaker': 'g.g.g.g.g.g.g.g.'},
            human=0.5, vel=0.7)
    T.drum('crash_soft', T.at(4), 0.45)
    T.drum('crash_soft', T.at(12), 0.55)
    T.fx('swell', 0, 2.0, 0.4, ends_at=T.at(12))
    for b in (12, 14):
        T.add('timp', T.at(b), 1.0, 38, 0.45)
    T.add('bell', T.at(16), 4, 74, 0.35)
    return T


TRACKS['title'] = title

# =========================================================================== the overworld (day + night)
MAP_PROG = [
    ('intro', 'G | Cadd9 | Em7 | D7sus4:2 D7:2'),
    ('A', 'G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | D7sus4:2 D7:2'),
    ("A'", 'G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | G'),
    ('B', 'Em | C | G | D/F# | Em | C | Am7 | D7sus4:2 D7:2'),
    ('C', 'Cmaj9 | Em7 | Am7 | D7sus4:2 D7:2 | Cmaj9 | Bm7 | Am7:2 Bm7:2 | C:2 D7sus4:1 D7:1'),
    ("A''", 'G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | G'),
    ('outro', 'G | Cadd9 | Em7 | D7sus4:2 D7:2'),
]
# night: the same harmony with sevenths and ninths
MAP_PROG_NIGHT = [
    ('intro', 'Gmaj7 | Cmaj7 | Em7 | D7sus4:2 D7:2'),
    ('A', 'Gmaj7 | D/F# | Em7 | Cmaj7 | G/B | Cmaj7 | Am7:2 D7:2 | D7sus4:2 D7:2'),
    ("A'", 'Gmaj7 | D/F# | Em7 | Cmaj7 | G/B | Cmaj7 | Am7:2 D7:2 | Gmaj7'),
    ('B', 'Em7 | Cmaj7 | Gmaj7 | D/F# | Em7 | Cmaj7 | Am7 | D7sus4:2 D7:2'),
    ('C', 'Cmaj9 | Em7 | Am7 | D7sus4:2 D7:2 | Cmaj9 | Bm7 | Am7:2 Bm7:2 | Cmaj7:2 D7sus4:1 D7:1'),
    ("A''", 'Gmaj7 | D/F# | Em7 | Cmaj7 | G/B | Cmaj7 | Am7:2 D7:2 | Gmaj7'),
    ('outro', 'Gmaj7 | Cmaj7 | Em7 | D7sus4:2 D7:2'),
]
MAP_A = ('D5:2 G5:2 A5 B5:3 | A5:2 F#5 E5 D5:4 | E5 F#5 G5 B5 A5:2 G5 E5 | D5:6 r:2 | D5:2 G5:2 A5 B5:3 | '
         'C6:2 B5 A5 G5:2 E5:2 | A5:2 G5 E5 F#5:2 A5:2 | G5:4 F#5:2 r:2')
MAP_A2 = ('D5:2 G5:2 A5 B5:3 | A5:2 F#5 E5 D5:4 | E5 F#5 G5 B5 A5:2 G5 E5 | D5:6 r:2 | D5:2 G5:2 A5 B5:3 | '
          'C6:2 B5 A5 G5:2 E5:2 | A5:2 B5 C6 B5:2 A5:2 | G5:6 r:2')
MAP_B = ('B4:3 E5 G5:2 F#5 E5 | E5:3 D5 C5:2 B4 C5 | D5:3 B4 G4:2 A4 B4 | A4:6 r:2 | B4:3 E5 G5:2 A5 B5 | '
         'C6:3 B5 A5:2 G5 E5 | E5:2 A5:2 G5 E5 D5 C5 | D5:4 r:4')
MAP_C = ('D4 G4 A4 B4:3 D5:2 | E5:3 D5 B4:2 G4:2 | E4 A4 B4 C5:3 E5:2 | D5:2 C5 B4 A4:4 | G5:3 E5 D5:2 E5 G5 | '
         'F#5:3 D5 B4:2 D5 F#5 | E5:2 C5 A4 D5:2 F#5:2 | G5:3 E5 A5:2 F#5:2')


def _map_base(tid, title, night):
    T = Track(tid, title, 104, 48, rev=1.25 if night else 1.0)
    b = 0
    for name, p in (MAP_PROG_NIGHT if night else MAP_PROG):
        T.mark(b, name)
        T.prog(p)
        b += len(p.split('|'))
    return T


def map_day():
    T = _map_base('map_day', 'Wind in the Envelope', False)
    riff = dict(step=0.5, lo=62, hi=81, group=[0, 2, 1, 3, 2, 4, 3, 1], pans=(0.3, -0.1))
    # intro / outro: the wind riff, bass and a light pulse
    for b0 in (0, 44):
        T.arp('kalimba', T.at(b0), T.at(b0 + 4), vel=0.5, **riff)
        T.bassline('bass', b0, b0 + 4, [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.62)
        T.pad('bed', b0, b0 + 4, lo=50, hi=67, n=4, vel=0.34)
        T.drums(b0, b0 + 4, {'shaker': 'g.o.g.o.g.o.g.o.', 'kick_soft': 'x.......x.......'}, human=0.5, vel=0.7)
    T.drums(3, 4, {'snare_soft': '........g.o.x.x.'}, vel=0.7)
    # A, A': the theme over the folk band
    bounce = [(0, 'R', 1.4), (1.5, '5', 0.45, 0.6), (2, 'R', 0.9, 0.85), (3, 'R+12', 0.45, 0.6), (3.5, '5', 0.45, 0.6)]
    T.bassline('bass', 4, 20, bounce, lo=36, vel=0.72)
    T.comp('strum', 4, 20, [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)], lo=55, hi=72,
           n=3, vel=0.48, human=0.6)
    T.pad('pad', 4, 20, lo=50, hi=67, n=4, vel=0.26)
    T.drums(4, 20, {'kick': 'X.......X.....x.', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.4, vel=0.78)
    T.drum('crash_soft', T.at(4), 0.6)
    T.mel('lead', 4, MAP_A, vel=0.8)
    a2 = T.mel('lead', 12, MAP_A2, vel=0.82)
    T.harmonize(a2[8:], 'counter', vel=0.55)
    T.arp('harp', T.at(12), T.at(20), step=0.5, lo=67, hi=84, vel=0.26, pattern='updown', pans=(-0.4, 0.4))
    T.mel('sparkle', 7, 'r:4 B5 D6 G6:2', vel=0.45)
    T.mel('sparkle', 11, 'r:4 A5 D6 F#6:2', vel=0.45)
    # B: softer, the ocarina sings low
    T.bassline('bass', 20, 28, [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.62)
    T.arp('harp', T.at(20), T.at(28), step=0.5, lo=62, hi=81, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.pad('bed', 20, 28, lo=50, hi=67, n=4, vel=0.36)
    T.drums(20, 28, {'kick_soft': 'x.......x.......', 'rim': '............o...', 'shaker': 'g.g.g.g.g.g.g.g.'},
            human=0.5, vel=0.7)
    T.drum('crash_soft', T.at(20), 0.45)
    T.mel('ocarina', 20, MAP_B, vel=0.82)
    T.mel('counter', 24, 'G4:8 | E4:8 | C5:4 A4:4 | A4:4 r:4', vel=0.45)
    # C: the wide world - the Atlas theme on brass
    T.bassline('bass', 28, 36, [(0, 'R', 1.8), (2, 'R', 0.9, 0.7), (3, '5', 0.9, 0.7)], lo=36, vel=0.72)
    T.pad('choir', 28, 36, lo=55, hi=72, n=3, vel=0.26)
    T.pad('pad', 28, 36, lo=48, hi=64, n=4, vel=0.3)
    T.arp('harp', T.at(28), T.at(36), step=0.25, lo=67, hi=86, vel=0.24, pattern='up', pans=(-0.4, 0.4))
    T.drums(28, 36, {'boom': 'X...............', 'kick_soft': '........x.......', 'snare_soft': '....x.......x...',
                     'hat_soft': '..g...g...g...g.'}, human=0.3, vel=0.75)
    T.drum('crash', T.at(28), 0.65)
    T.drum('crash_soft', T.at(32), 0.55)
    for b in (28, 30, 32, 34):
        T.add('timp', T.at(b), 1.0, 26 + (T.chord_at(T.at(b)).root - 26) % 12 + 12, 0.55)
    c = T.mel('brass', 28, MAP_C, vel=0.82)
    T.harmonize(c[18:], 'brass2', vel=0.55)
    T.roll('snare_soft', T.at(35, 2), T.at(36), 0.25, 0.25, 0.75)
    # A'': everyone
    T.bassline('bass', 36, 44, bounce, lo=36, vel=0.75)
    T.comp('strum', 36, 44, [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)], lo=55, hi=72,
           n=3, vel=0.5, human=0.6)
    T.pad('pad', 36, 44, lo=50, hi=67, n=4, vel=0.3)
    T.arp('harp', T.at(36), T.at(44), step=0.25, lo=67, hi=86, vel=0.24, pattern='updown', pans=(-0.4, 0.4))
    T.drums(36, 44, {'kick': 'X.......X.....x.', 'snare': '....x.......x...', 'hat': 'x.o.x.o.x.o.x.o.',
                     'shaker': 'g.g.g.g.g.g.g.g.'}, human=0.3, vel=0.78)
    T.drum('crash', T.at(36), 0.7)
    a3 = T.mel('lead', 36, MAP_A2, vel=0.86)
    T.harmonize(a3, 'harm', vel=0.6)
    T.drum('crash_soft', T.at(44), 0.5)
    return T


def map_night():
    T = _map_base('map_night', 'Lanterns over Equestria', True)
    riff = dict(step=0.5, lo=62, hi=81, group=[0, 2, 1, 3, 2, 4, 3, 1], pans=(0.3, -0.1))
    for b0 in (0, 44):
        T.arp('kalimba', T.at(b0), T.at(b0 + 4), vel=0.38, **riff)
        T.add('bell', T.at(b0), 4, 67, 0.3)
    T.bassline('upright', 0, 48, [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.55)
    T.pad('bed', 0, 48, lo=50, hi=67, n=4, vel=0.36)
    T.comp('keys', 4, 20, [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.4, human=0.7)
    T.comp('keys', 36, 44, [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.42, human=0.7)
    T.mel('mbox', 4, MAP_A, vel=0.72)
    a2 = T.mel('flute', 12, MAP_A2, vel=0.72)
    T.harmonize(a2, 'kalimba', vel=0.5)
    # B: harp quarters and the ocarina
    T.arp('harp', T.at(20), T.at(28), step=1.0, lo=62, hi=81, vel=0.34, pattern='updown', pans=(-0.3, 0.3))
    T.mel('ocarina', 20, MAP_B, vel=0.72)
    # C: choir and the warm lead
    T.pad('choir', 28, 36, lo=55, hi=72, n=3, vel=0.26)
    T.arp('harp', T.at(28), T.at(36), step=0.5, lo=64, hi=84, vel=0.26, pattern='up', pans=(-0.35, 0.35))
    T.mel('warm', 28, MAP_C, vel=0.76)
    for b in range(28, 36, 2):
        T.add('bell', T.at(b), 2, 72 + (T.chord_at(T.at(b)).root - 72) % 12, 0.24)
    # A'': music box with the flute an octave under, brushes
    a3 = T.mel('mbox', 36, MAP_A2, vel=0.74)
    T.double(a3, 'flute', octave=-1, vel=0.55)
    T.drums(36, 44, {'brush': '....o.......o...', 'kick_soft': 'g.......g.......'}, human=0.5, vel=0.6)
    T.drums(4, 20, {'shaker': 'g...g...g...g...'}, human=0.5, vel=0.5)
    T.mel('sparkle', 7, 'r:4 B5 D6 G6:2', vel=0.36)
    T.mel('sparkle', 11, 'r:4 A5 D6 F#6:2', vel=0.36)
    return T


TRACKS['map_day'] = map_day
TRACKS['map_night'] = map_night

# =========================================================================== race
RACE_A = ('A5 E5 A5 B5 C6:2 B5 A5 | F5:2 A5 C6 A5:2 G5 F5 | E5 G5 C6 E6 D6:2 C6 G5 | B5:3 A5 G5:2 D5:2 | '
          'A5 E5 A5 B5 C6:2 B5 A5 | F5:2 A5 C6 F6:2 E6 C6 | D6:2 C6 A5 F5:2 A5:2 | G#5:4 B5:2 E6:2')
RACE_B = ('C6:3 A5 F5:2 G5 A5 | B5:3 G5 D5:2 E5 G5 | B5:2 G5 E5 D6:2 B5:2 | C6:4 A5:4 | C6:3 A5 F5:2 G5 A5 | '
          'B5:3 G5 D6:2 B5 D6 | E6:4 D6:2 B5:2 | G#5:2 B5:2 D6:2 E6:2')


def race():
    T = Track('race', 'Ring Rush', 138, 32)
    for name, p in (('A', 'Am | F | C | G | Am | F | Dm7 | E7'), ('B', 'F | G | Em7 | Am | F | G | E7sus4 | E7'),
                    ("A'", 'Am | F | C | G | Am | F | Dm7 | E7'), ('break', 'Am | F | C | G | Am | F | Dm7 | E7')):
        T.mark(len(T.marks) * 8, name)
        T.prog(p)
    k_rock(T, 0, 8, 1)
    k_rock(T, 8, 16, 1)
    k_rock(T, 16, 24, 2)
    k_rock(T, 24, 32, 2)
    T.pad('pad', 8, 16, lo=52, hi=69, n=4, vel=0.3)
    for b, v in ((0, 0.75), (8, 0.6), (16, 0.8), (24, 0.6)):
        T.drum('crash', T.at(b), v)
    T.mel('lead', 0, RACE_A, vel=0.82)
    T.mel('lead', 8, RACE_B, vel=0.8)
    a2 = T.mel('lead', 16, RACE_A, vel=0.86)
    T.harmonize(a2, 'harm', vel=0.58)
    # the break: a call-and-response riff, then a snare build back into the top
    T.mel('counter', 24, 'A4 C5 E5 A5 r:2 G5 E5 | F5:2 A4 C5 r:2 F5 E5 | E5 G5 C6 E5 r:2 D5 B4 | '
                         'D5:2 B4 G4 r:4 | A4 C5 E5 A5 r:2 G5 E5 | F5:2 A4 C5 r:2 F5 E5 | D5 F5 A5 D6 r:2 C6 A5 | '
                         'B5:2 G#5 E5 r:4', vel=0.7)
    T.mel('sparkle', 25, 'r:4 C6 F6 A6:2', vel=0.4)
    T.mel('sparkle', 29, 'r:4 C6 F6 A6:2', vel=0.4)
    T.roll('snare', T.at(31), T.at(32), 0.25, 0.2, 0.85)
    T.fx('riser', 0, 4.0, 0.35, ends_at=T.at(32))
    return T


TRACKS['race'] = race

# =========================================================================== the 20 places
# (written in C major / A minor, moved by tr)

TRACKS['canterlot'] = place('canterlot', 'Sunrise over Canterlot', 96, -2, 'march', [
    dict(name='fanfare', prog='C | G7', mel='C5 C5 G4 C5 E5:4 | D5 D5 B4 D5 G5:4', voice='brass', e=0, vel=0.78),
    dict(name='A', prog='C | G/B | Am | Em/G | F | C/E | Dm7:2 G7:2 | C',
         mel='E5:3 F5 G5:2 C6:2 | B5:3 A5 G5:4 | A5:3 B5 C6:2 A5:2 | G5:3 F#5 G5:2 B5:2 | A5:3 G5 F5:2 C6:2 | '
             'C6:2 B5 A5 G5:4 | F5:2 A5:2 G5:2 B4:2 | C5:6 r:2', voice='brass', e=1, dbl=('sparkle', 1), dbl_vel=0.25),
    dict(name='trio', prog='F | C/E | Dm7 | G | Em7 | Am | D7 | G7',
         mel='C6:4 A5:2 F5:2 | G5:4 E5:2 C5:2 | D5:2 F5:2 A5:2 C6:2 | B5:6 r:2 | B5:3 A5 G5:2 E5:2 | '
             'A5:3 G5 E5:2 C5:2 | F#5:2 A5:2 C6:2 D6:2 | B5:2 G5:2 F5:2 D5:2', voice='whistle', e=0, fill='snare'),
    dict(name="A'", prog='C | G/B | Am | Em/G | F | C/E | Dm7:2 G7:2 | C',
         mel='E5:3 F5 G5:2 C6:2 | B5:3 A5 G5:4 | A5:3 B5 C6:2 A5:2 | G5:3 F#5 G5:2 B5:2 | A5:3 G5 F5:2 C6:2 | '
             'C6:2 B5 A5 G5:4 | F5:2 A5:2 G5:2 B4:2 | C5:6 r:2', voice='brass', e=2, harm='brass2'),
])

TRACKS['cloudsdale'] = place('cloudsdale', 'Above the Clouds', 112, -3, 'airy', [
    dict(name='intro', prog='Cmaj7 | G7sus4', mel=None, e=0),
    dict(name='A', prog='Cmaj7 | G/B | Am7 | Fmaj7 | Cmaj7 | Em7 | Fmaj7 | G7sus4:2 G:2',
         mel='C5:2 G5:4 E5:2 | D5:2 B5:4 G5:2 | C6:3 B5 A5:2 E5:2 | F5:2 E5:2 C5:4 | C5:2 G5:4 E5:2 | '
             'D5:2 B5:4 G5:2 | A5:3 C6 E6:2 C6:2 | D6:4 B5:4', voice='whistle', e=1),
    dict(name='B', prog='Am7 | Em7 | Fmaj7 | Cmaj7 | Dm7 | Em7 | Fmaj7 | G7sus4:2 G7:2',
         mel='E5:3 G5 A5:4 | G5:3 E5 D5:4 | C5:2 F5:2 A5:2 G5:2 | E5:6 r:2 | F5:3 A5 C6:4 | B5:3 G5 E5:4 | '
             'A5:2 G5 F5 E5:2 C5:2 | D5:4 F5:2 G5:2', voice='ocarina', e=1),
    dict(name="A'", prog='Cmaj7 | G/B | Am7 | Fmaj7 | Cmaj7 | Em7 | Fmaj7 | G7sus4:2 G:2',
         mel='C5:2 G5:4 E5:2 | D5:2 B5:4 G5:2 | C6:3 B5 A5:2 E5:2 | F5:2 E5:2 C5:4 | C5:2 G5:4 E5:2 | '
             'D5:2 B5:4 G5:2 | A5:3 C6 E6:2 C6:2 | D6:4 B5:4', voice='whistle', e=2, dbl=('sparkle', 0), dbl_vel=0.4),
], rev=1.3)

TRACKS['ponyville'] = place('ponyville', 'Ponyville Morning', 104, -5, 'folk', [
    dict(name='intro', prog='C | G7', mel='r:4 G5 E5 C5 D5 | E5:2 D5:2 B4:2 r:2', voice='kalimba', e=0),
    dict(name='A', prog='C | F | C | G7 | C | F | G7 | C',
         mel='E5 G5 A5 G5 E5:2 C5:2 | F5 A5 C6 A5 G5:2 F5:2 | E5 G5 C6 B5 A5 G5 E5:2 | D5:2 F5 E5 D5:4 | '
             'E5 G5 A5 G5 E5:2 C5:2 | F5 A5 C6 D6 C6:2 A5:2 | B5:2 D6 B5 G5:2 F5 D5 | C5:4 r:4', voice='whistle',
         e=1),
    dict(name='B', prog='Am | Em | F | C | Dm7 | G | Em7:2 A7:2 | Dm7:2 G7:2',
         mel='A5:3 G5 E5:2 C5:2 | B4:3 D5 E5:2 G5:2 | A5:3 G5 F5:2 C5:2 | E5:6 r:2 | D5 F5 A5 C6 A5:2 F5:2 | '
             'G5 B5 D6 B5 G5:4 | E5:2 G5:2 C#6:2 A5:2 | D6:2 C6:2 B5:2 G5:2', voice='flute', e=1),
    dict(name="A'", prog='C | F | C | G7 | C | F | G7 | C',
         mel='E5 G5 A5 G5 E5:2 C5:2 | F5 A5 C6 A5 G5:2 F5:2 | E5 G5 C6 B5 A5 G5 E5:2 | D5:2 F5 E5 D5:4 | '
             'E5 G5 A5 G5 E5:2 C5:2 | F5 A5 C6 D6 C6:2 A5:2 | B5:2 D6 B5 G5:2 F5 D5 | C5:4 r:4', voice='whistle',
         e=2, harm='kalimba'),
], swing=0.6)

TRACKS['sweet_apple_acres'] = place('sweet_apple_acres', 'Apple Bucking Hoedown', 126, 2, 'hoedown', [
    dict(name='intro', prog='C | G7', mel=None, e=1),
    dict(name='A', prog='C | C | F | C | C | C | G7 | C',
         mel='G5 E5 G5 A5 G5 E5 C5 E5 | G5:2 A5 G5 E5:2 C5:2 | A5 F5 A5 C6 A5 F5 A5 C6 | G5 E5 C5 E5 G5:4 | '
             'G5 E5 G5 A5 G5 E5 C5 E5 | C6 B5 A5 G5 E5 D5 C5:2 | D5 F5 G5 B5 D6 B5 G5 F5 | E5 C5 D5 E5 C5:4',
         voice='fiddle', e=1),
    dict(name='B', prog='F | C | G | C | F | C | G7 | C',
         mel='A5 C6 A5 F5 C5 F5 A5 C6 | E6:2 D6 C6 G5:2 E5:2 | D6 B5 G5 B5 D6 B5 G5 B5 | C6:2 G5 E5 C5:4 | '
             'A5 C6 A5 F5 C5 F5 A5 C6 | E6:2 D6 C6 G5 A5 G5 E5 | D6 C6 B5 A5 G5 F5 E5 D5 | C5:2 E5:2 G5:2 C6:2',
         voice='fiddle', e=1, harm='counter'),
    dict(name="A'", prog='C | C | F | C | C | C | G7 | C',
         mel='G5 E5 G5 A5 G5 E5 C5 E5 | G5:2 A5 G5 E5:2 C5:2 | A5 F5 A5 C6 A5 F5 A5 C6 | G5 E5 C5 E5 G5:4 | '
             'G5 E5 G5 A5 G5 E5 C5 E5 | C6 B5 A5 G5 E5 D5 C5:2 | D5 F5 G5 B5 D6 B5 G5 F5 | E5 C5 D5 E5 C5:4',
         voice='fiddle', e=2, dbl=('whistle', 0), dbl_vel=0.35),
])

TRACKS['everfree'] = place('everfree', 'Whispering Everfree', 84, -5, 'mystic', [
    dict(name='intro', prog='Am | D/F# | Fmaj7 | E7', mel=None, e=0),
    dict(name='A', prog='Am | D/F# | Fmaj7 | Em7 | Am | D/F# | Fmaj7 | E7sus4:2 E7:2',
         mel='r:2 E5:3 D5 C5:2 | D5:3 E5 F#5:4 | E5:3 C5 A4:4 | B4:6 r:2 | r:2 E5:3 G5 A5:2 | A5:3 F#5 D5:4 | '
             'C5:2 E5:2 A5:2 G5:2 | A5:4 G#5:4', voice='reed', e=0),
    dict(name='B', prog='Fmaj7 | G | Em7 | Am | Fmaj7 | G | Dm7 | E7',
         mel='C6:3 B5 A5:2 E5:2 | D6:3 C6 B5:2 G5:2 | B5:3 A5 G5:2 E5:2 | A5:6 r:2 | C6:3 B5 A5:2 F5:2 | '
             'G5:3 A5 B5:2 D6:2 | C6:2 A5:2 F5:2 D5:2 | B5:4 G#5:4', voice='flute', e=1),
    dict(name="A'", prog='Am | D/F# | Fmaj7 | Em7 | Am | D/F# | Fmaj7 | E7sus4:2 E7:2',
         mel='r:2 E5:3 D5 C5:2 | D5:3 E5 F#5:4 | E5:3 C5 A4:4 | B4:6 r:2 | r:2 E5:3 G5 A5:2 | A5:3 F#5 D5:4 | '
             'C5:2 E5:2 A5:2 G5:2 | A5:4 G#5:4', voice='reed', e=2, dbl=('kalimba', 1), dbl_vel=0.4),
], rev=1.4)

TRACKS['neighagra_falls'] = place('neighagra_falls', 'The Roar of Neighagra', 116, -7, 'airy', [
    dict(name='intro', prog='C | Bb', mel=None, e=1, kit='pop'),
    dict(name='A', prog='C | Bb | F | C | C | Bb | F | G',
         mel='G5:2 C6:2 D6 E6:3 | D6:3 C6 Bb5:2 F5:2 | A5:2 C6:2 A5:2 G5 F5 | G5:6 r:2 | G5:2 C6:2 D6 E6:3 | '
             'F6:2 D6:2 Bb5:2 C6:2 | A5:3 G5 F5:2 A5:2 | G5:2 B5:2 D6:4', voice='warm', e=1, kit='pop'),
    dict(name='B', prog='Am | F | C | G | Am | F | Dm7 | G7sus4:2 G7:2',
         mel='E6:3 D6 C6:2 A5:2 | C6:3 A5 F5:4 | G5 A5 C6 D6 E6:4 | D6:6 r:2 | E6:3 D6 C6:2 E6:2 | '
             'F6:3 E6 C6:2 A5:2 | D6:2 C6:2 A5:2 F5:2 | G5:4 B5:2 D6:2', voice='lead', e=1, fill='tom'),
    dict(name="A'", prog='C | Bb | F | C | C | Bb | F | G',
         mel='G5:2 C6:2 D6 E6:3 | D6:3 C6 Bb5:2 F5:2 | A5:2 C6:2 A5:2 G5 F5 | G5:6 r:2 | G5:2 C6:2 D6 E6:3 | '
             'F6:2 D6:2 Bb5:2 C6:2 | A5:3 G5 F5:2 A5:2 | G5:2 B5:2 D6:4', voice='warm', e=2, harm='counter',
         kit='pop'),
], extra=lambda T: T.arp('harp', 0, T.at(T.bars), step=0.25, lo=67, hi=88, vel=0.26, pattern='down',
                         pans=(-0.45, 0.45)))

TRACKS['rainbow_falls'] = place('rainbow_falls', 'Rainbow Hoops', 132, -3, 'rock', [
    dict(name='intro', prog='C | G', mel=None, e=1),
    dict(name='A', prog='C | G | Am | F | C | G | F | G',
         mel='C6 C6 G5 C6 E6:2 D6 C6 | D6:2 B5 G5 D6:2 B5:2 | C6 C6 A5 C6 E6:2 D6 C6 | A5:4 r:2 F5 G5 | '
             'C6 C6 G5 C6 E6:2 E6 G6 | G6:2 D6 B5 G5:2 B5:2 | A5:2 C6:2 F6:2 E6:2 | D6:4 B5:2 G5:2', voice='lead',
         e=1),
    dict(name='B', prog='Am | F | C | G | Am | F | Dm7 | G7',
         mel='E6:3 C6 A5:2 C6:2 | C6:3 A5 F5:2 A5:2 | G5:3 C6 E6:2 G6:2 | D6:6 r:2 | E6:3 C6 A5:2 E6:2 | '
             'F6:3 E6 C6:2 A5:2 | D6:2 F6:2 A6:2 F6:2 | G6:2 F6:2 D6:2 B5:2', voice='lead', e=1, harm='harm',
         fill='tom'),
    dict(name="A'", prog='C | G | Am | F | C | G | F | G',
         mel='C6 C6 G5 C6 E6:2 D6 C6 | D6:2 B5 G5 D6:2 B5:2 | C6 C6 A5 C6 E6:2 D6 C6 | A5:4 r:2 F5 G5 | '
             'C6 C6 G5 C6 E6:2 E6 G6 | G6:2 D6 B5 G5:2 B5:2 | A5:2 C6:2 F6:2 E6:2 | D6:4 B5:2 G5:2', voice='lead',
         e=2, harm='harm'),
])

TRACKS['manehattan'] = place('manehattan', 'Big City Swing', 120, -2, 'swing', [
    dict(name='intro', prog='C6 | G7', mel='r:2 G5 r:2 G5 A5:2 | B5 r B5 r D6:4', voice='brass', e=1),
    dict(name='A', prog='C6 | A7 | Dm7 | G7 | C6 | A7 | Dm7:2 G7:2 | C6',
         mel='E5:2 G5 A5 r G5 E5:2 | C#6:3 A5 G5:2 E5:2 | F5:2 A5 C6 r D6 C6:2 | B5:6 r:2 | '
             'E5:2 G5 A5 r C6 E6:2 | E6:3 C#6 A5:2 G5:2 | F5:2 A5:2 G5 F5 D5:2 | C5:6 r:2', voice='brass', e=1),
    dict(name='bridge', prog='Fmaj7 | Fm6 | Em7 | A7 | Dm7 | G7 | Cmaj7 | G7sus4:2 G7:2',
         mel='A5:3 C6 E6:4 | Ab5:3 C6 D6:4 | G5:3 B5 D6:4 | C#6:2 E6:2 G6:2 E6:2 | F6:3 E6 D6:2 A5:2 | '
             'B5:3 A5 G5:2 F5:2 | E5:2 G5:2 B5:2 C6:2 | C6:4 B5:4', voice='warm', e=1, harm='counter'),
    dict(name="A'", prog='C6 | A7 | Dm7 | G7 | C6 | A7 | Dm7:2 G7:2 | C6',
         mel='E5:2 G5 A5 r G5 E5:2 | C#6:3 A5 G5:2 E5:2 | F5:2 A5 C6 r D6 C6:2 | B5:6 r:2 | '
             'E5:2 G5 A5 r C6 E6:2 | E6:3 C#6 A5:2 G5:2 | F5:2 A5:2 G5 F5 D5:2 | C5:6 r:2', voice='brass', e=2,
         harm='brass2'),
], swing=0.64)

TRACKS['baltimare'] = place('baltimare', 'Harbour Shuffle', 98, 3, 'shuffle', [
    dict(name='intro', prog='C | G7', mel=None, e=0),
    dict(name='A', prog='C | C | F | C | C | Am | D7 | G7',
         mel='G4:2 C5:2 E5:3 D5 | C5:2 E5:2 G5:4 | A5:3 G5 F5:2 C5:2 | E5:6 r:2 | G4:2 C5:2 E5:3 F5 | '
             'E5:2 A5:2 C6:3 B5 | A5:3 F#5 D5:2 C5:2 | B4:4 D5:2 G5:2', voice='accordion', e=1),
    dict(name='B', prog='F | C | G7 | C | F | C | D7:2 G7:2 | C',
         mel='C6:3 A5 F5:4 | G5:3 E5 C5:4 | D5:2 F5:2 B5:2 A5 G5 | E5:2 G5 E5 C5:4 | C6:3 A5 F5:2 A5:2 | '
             'G5:3 E5 C5:2 E5:2 | F#5:2 A5:2 G5:2 F5:2 | C5:4 r:4', voice='whistle', e=1),
    dict(name="A'", prog='C | C | F | C | C | Am | D7 | G7',
         mel='G4:2 C5:2 E5:3 D5 | C5:2 E5:2 G5:4 | A5:3 G5 F5:2 C5:2 | E5:6 r:2 | G4:2 C5:2 E5:3 F5 | '
             'E5:2 A5:2 C6:3 B5 | A5:3 F#5 D5:2 C5:2 | B4:4 D5:2 G5:2', voice='accordion', e=2, harm='counter'),
], swing=0.66)

TRACKS['griffonstone'] = place('griffonstone', 'Griffonstone Ramparts', 92, -3, 'march', [
    dict(name='intro', prog='Am | E7', mel=None, e=0),
    dict(name='A', prog='Am | G | F | E7 | Am | G | F:2 G:2 | Am',
         mel='A4:3 B4 C5:2 E5:2 | D5:3 C5 B4:2 G4:2 | C5:3 A4 F4:2 A4:2 | G#4:4 B4:2 D5:2 | A4:3 B4 C5:2 E5:2 | '
             'D5:3 E5 F5:2 D5:2 | C5:2 A4:2 B4:2 D5:2 | A4:6 r:2', voice='brass', e=1),
    dict(name='B', prog='F | C | G | Am | F | C | Dm | E7',
         mel='A5:3 G5 F5:2 C5:2 | E5:3 D5 C5:2 G4:2 | B4:3 C5 D5:4 | C5:2 A4:2 E5:4 | A5:3 G5 F5:2 A5:2 | '
             'G5:3 E5 C5:2 E5:2 | F5:3 E5 D5:2 A4:2 | G#4:2 B4:2 E5:4', voice='reed', e=1, fill='snare'),
    dict(name="A'", prog='Am | G | F | E7 | Am | G | F:2 G:2 | Am',
         mel='A4:3 B4 C5:2 E5:2 | D5:3 C5 B4:2 G4:2 | C5:3 A4 F4:2 A4:2 | G#4:4 B4:2 D5:2 | A4:3 B4 C5:2 E5:2 | '
             'D5:3 E5 F5:2 D5:2 | C5:2 A4:2 B4:2 D5:2 | A4:6 r:2', voice='brass', e=2, tr=12, harm='brass2'),
], rev=1.25)

TRACKS['dragon_lands'] = place('dragon_lands', 'Land of Fire', 100, -7, 'heavy', [
    dict(name='riff', prog='Am | E7', mel=None, e=0),
    dict(name='A', prog='Am | Bb | Am | G | Am | Bb | F | E7',
         mel='A4:2 E5:2 r C5 B4 A4 | Bb4:2 F5:2 r D5 C5 Bb4 | A4:2 E5:2 A5:3 G5 | G5:2 D5:2 B4:4 | '
             'A4:2 E5:2 r C5 B4 A4 | Bb4:2 F5:2 Bb5:3 A5 | A5:2 F5:2 C5:2 A4:2 | G#4:2 B4:2 E5:2 D5:2', voice='brass',
         e=1),
    dict(name='B', prog='F | G | Am | Am | F | G | E7 | E7',
         mel='C5:3 F5 A5:4 | D5:3 G5 B5:4 | C6:3 B5 A5:2 E5:2 | A5:6 r:2 | C6:3 A5 F5:2 C6:2 | D6:3 B5 G5:2 D6:2 | '
             'E6:4 D6:2 B5:2 | G#5:4 E5:4', voice='lead', e=1, harm='harm', fill='tom'),
    dict(name="A'", prog='Am | Bb | Am | G | Am | Bb | F | E7',
         mel='A4:2 E5:2 r C5 B4 A4 | Bb4:2 F5:2 r D5 C5 Bb4 | A4:2 E5:2 A5:3 G5 | G5:2 D5:2 B4:4 | '
             'A4:2 E5:2 r C5 B4 A4 | Bb4:2 F5:2 Bb5:3 A5 | A5:2 F5:2 C5:2 A4:2 | G#4:2 B4:2 E5:2 D5:2', voice='brass',
         e=2, dbl=('lead', 1), dbl_vel=0.5),
])

TRACKS['mount_aris'] = place('mount_aris', 'Mount Aris and the Sea', 104, -3, 'medi', [
    dict(name='intro', prog='C | G7', mel=None, e=0),
    dict(name='A', prog='C | F | G | C | Am | F | G | C',
         mel='E5 F5 G5:2 G5 F5 E5 D5 | C5:2 F5:2 A5:3 G5 | B5 A5 G5:2 D5:2 G5:2 | E5:6 r:2 | E5 F5 G5:2 A5 G5 E5 C5 | '
             'F5:2 A5:2 C6:3 A5 | B5:2 D6:2 B5 G5 A5 B5 | C6:4 r:4', voice='whistle', e=1),
    dict(name='under the sea', prog='Am7 | Fmaj7 | Cmaj7 | G | Am7 | Fmaj7 | Dm7 | G7sus4',
         mel='E5:4 C5:4 | A4:4 E5:4 | G5:4 B5:4 | D5:8 | E5:4 G5:4 | A5:4 C6:4 | F5:4 A5:4 | C6:4 D6:4',
         voice='mbox', e=0, kit='airy', bus='uw', crash=False),
    dict(name="A'", prog='C | F | G | C | Am | F | G | C',
         mel='E5 F5 G5:2 G5 F5 E5 D5 | C5:2 F5:2 A5:3 G5 | B5 A5 G5:2 D5:2 G5:2 | E5:6 r:2 | E5 F5 G5:2 A5 G5 E5 C5 | '
             'F5:2 A5:2 C6:3 A5 | B5:2 D6:2 B5 G5 A5 B5 | C6:4 r:4', voice='whistle', e=2, harm='counter'),
])

TRACKS['klugetown'] = place('klugetown', 'Klugetown Bazaar', 112, -7, 'bazaar', [
    dict(name='intro', prog='A | Bb', mel=None, e=0),
    dict(name='A', prog='A | Bb | Gm | A | A | Bb | Gm:2 Bb:2 | A',
         mel='A4 Bb4 C#5 D5 E5:2 D5 C#5 | F5:2 E5 D5 Bb4:4 | G5 F5 E5 D5 Bb4:2 D5:2 | C#5:2 E5:2 A4:4 | '
             'A4 Bb4 C#5 D5 E5:2 F5 E5 | F5:2 D5 F5 Bb5:2 A5:2 | G5:2 Bb5:2 F5:2 D5:2 | E5:2 C#5:2 A4:4', voice='reed',
         e=1),
    dict(name='B', prog='Dm | Gm | A | Dm | Bb | Gm | A | A',
         mel='D5:2 F5:2 A5:3 G5 | Bb5:3 A5 G5:2 D5:2 | C#6:3 Bb5 A5:2 E5:2 | F5:3 E5 D5:4 | D6:3 C6 Bb5:2 F5:2 | '
             'G5:3 A5 Bb5:2 D6:2 | C#6:2 E6:2 D6 C#6 Bb5 A5 | A5:6 r:2', voice='twang', e=1, fill='tom'),
    dict(name="A'", prog='A | Bb | Gm | A | A | Bb | Gm:2 Bb:2 | A',
         mel='A4 Bb4 C#5 D5 E5:2 D5 C#5 | F5:2 E5 D5 Bb4:4 | G5 F5 E5 D5 Bb4:2 D5:2 | C#5:2 E5:2 A4:4 | '
             'A4 Bb4 C#5 D5 E5:2 F5 E5 | F5:2 D5 F5 Bb5:2 A5:2 | G5:2 Bb5:2 F5:2 D5:2 | E5:2 C#5:2 A4:4', voice='reed',
         e=2, dbl=('kalimba', 1), dbl_vel=0.35),
])

TRACKS['changeling_kingdom'] = place('changeling_kingdom', 'The Blooming Hive', 96, -5, 'mystic', [
    dict(name='intro', prog='Am | E7', mel=None, e=0),
    dict(name='dark', prog='Am | F | Dm | E7sus4:2 E7:2 | Am | F | Dm | E7',
         mel='A4:2 C5:2 E5:2 D5:2 | C5:4 A4:4 | D5:2 F5:2 A5:2 G5:2 | A5:4 G#5:4 | A4:2 C5:2 E5:2 A5:2 | '
             'C6:3 B5 A5:4 | F5:3 E5 D5:4 | E5:6 r:2', voice='reed', e=1),
    dict(name='bloom', prog='C | G/B | Am | F | C | G/B | F | G7',
         mel='G5:2 C6:2 E6:4 | D6:3 C6 B5:4 | C6:3 B5 A5:2 E5:2 | F5:2 A5:2 C6:4 | G5:2 C6:2 E6:2 G6:2 | '
             'D6:3 E6 D6:2 B5:2 | C6:2 A5:2 F5:2 A5:2 | B5:4 D6:4', voice='warm', e=1, kit='airy',
         dbl=('sparkle', 0), dbl_vel=0.35),
    dict(name='full bloom', prog='C | G/B | Am | F | C | G/B | F | G7',
         mel='G5:2 C6:2 E6:4 | D6:3 C6 B5:4 | C6:3 B5 A5:2 E5:2 | F5:2 A5:2 C6:4 | G5:2 C6:2 E6:2 G6:2 | '
             'D6:3 E6 D6:2 B5:2 | C6:2 A5:2 F5:2 A5:2 | B5:4 D6:4', voice='warm', e=2, kit='pop', harm='counter'),
], rev=1.2)

TRACKS['appleloosa'] = place('appleloosa', 'Appleloosa Sundown', 96, 4, 'folk', [
    dict(name='intro', prog='C | G7', mel=None, e=1),
    dict(name='A', prog='C | Em | F | C | Am | Em | F:2 G7:2 | C',
         mel='G4:2 C5:3 D5 E5:2 | G5:3 E5 B4:4 | A4:2 C5:2 F5:3 E5 | G5:6 r:2 | A5:3 G5 E5:2 C5:2 | '
             'B4:3 D5 E5:2 G5:2 | A5:2 C6:2 B5:2 D6:2 | C6:6 r:2', voice='twang', e=1),
    dict(name='B', prog='F | C | G | Am | F | C | D7 | G7',
         mel='C6:3 A5 F5:4 | E5:3 G5 C6:4 | B5:3 A5 G5:2 D5:2 | C5:2 E5:2 A5:4 | A5:3 C6 F5:4 | G5:3 E5 C5:4 | '
             'F#5:3 E5 D5:2 A4:2 | B4:2 D5:2 F5:2 G5:2', voice='whistle', e=1),
    dict(name="A'", prog='C | Em | F | C | Am | Em | F:2 G7:2 | C',
         mel='G4:2 C5:3 D5 E5:2 | G5:3 E5 B4:4 | A4:2 C5:2 F5:3 E5 | G5:6 r:2 | A5:3 G5 E5:2 C5:2 | '
             'B4:3 D5 E5:2 G5:2 | A5:2 C6:2 B5:2 D6:2 | C6:6 r:2', voice='twang', e=2, harm='counter'),
], swing=0.62, extra=lambda T: T.arp('banjo', 0, T.at(T.bars), step=0.5, lo=62, hi=79, vel=0.3,
                                     group=[0, 2, 1, 3], pans=(0.35,)))

TRACKS['las_pegasus'] = place('las_pegasus', 'Neon Strip', 112, -7, 'funk', [
    dict(name='groove', prog='Am7 | D9', mel=None, e=1),
    dict(name='A', prog='Am7 | D9 | Am7 | D9 | Fmaj7 | Em7 | Dm7 | E7sus4:2 E7:2',
         mel='r A5 r C6 r:2 A5 G5 | A5:2 F#5 E5 r:2 D5 E5 | r A5 r C6 r E6 D6 C6 | A5:4 r:4 | '
             'C6:2 A5 C6 E6:2 C6:2 | B5:2 G5 B5 D6:2 B5:2 | A5:2 F5 A5 C6:2 A5:2 | A5:4 G#5:4', voice='lead', e=1),
    dict(name='showtime', prog='Fmaj7 | G | Em7 | Am7 | Dm7 | G7 | Cmaj7 | E7',
         mel='C6 r C6 r A5 C6 r:2 | D6 r D6 r B5 D6 r:2 | E6:2 D6 B5 G5:2 B5:2 | C6:6 r:2 | '
             'F6 r F6 r D6 F6 r:2 | D6:2 B5 G5 F5:2 D5:2 | E5:2 G5:2 B5:2 C6:2 | B5:4 G#5:4', voice='brass', e=2,
         harm='brass2'),
    dict(name="A'", prog='Am7 | D9 | Am7 | D9 | Fmaj7 | Em7 | Dm7 | E7sus4:2 E7:2',
         mel='r A5 r C6 r:2 A5 G5 | A5:2 F#5 E5 r:2 D5 E5 | r A5 r C6 r E6 D6 C6 | A5:4 r:4 | '
             'C6:2 A5 C6 E6:2 C6:2 | B5:2 G5 B5 D6:2 B5:2 | A5:2 F5 A5 C6:2 A5:2 | A5:4 G#5:4', voice='lead', e=2,
         harm='harm'),
])

TRACKS['vanhoover'] = place('vanhoover', 'Rain on Vanhoover', 80, -9, 'lofi', [
    dict(name='intro', prog='Fmaj7 | Em7', mel=None, e=0),
    dict(name='A', prog='Fmaj7 | Em7 | Dm7 | Cmaj7 | Fmaj7 | Em7 | Dm7:2 G7sus4:2 | Cmaj7',
         mel='r:2 A5 C6 E6:2 C6 A5 | G5:4 r:2 E5 G5 | A5:2 F5 E5 D5:2 C5:2 | E5:6 r:2 | r:2 A5 C6 E6:2 G6 E6 | '
             'D6:4 B5:2 G5:2 | F5:2 A5:2 G5:2 C6:2 | B5:6 r:2', voice='warm', e=1),
    dict(name='B', prog='Am7 | Dm7 | G7 | Cmaj7 | Am7 | D7 | Fmaj7 | G7sus4:2 G7:2',
         mel='E5:3 G5 C6:4 | A5:3 C6 F5:4 | B5:3 A5 G5:2 F5:2 | E5:6 r:2 | C6:3 B5 A5:2 E5:2 | F#5:3 A5 C6:4 | '
             'C6:2 A5:2 E5:2 F5:2 | G5:4 F5:4', voice='flute', e=1),
    dict(name="A'", prog='Fmaj7 | Em7 | Dm7 | Cmaj7 | Fmaj7 | Em7 | Dm7:2 G7sus4:2 | Cmaj7',
         mel='r:2 A5 C6 E6:2 C6 A5 | G5:4 r:2 E5 G5 | A5:2 F5 E5 D5:2 C5:2 | E5:6 r:2 | r:2 A5 C6 E6:2 G6 E6 | '
             'D6:4 B5:2 G5:2 | F5:2 A5:2 G5:2 C6:2 | B5:6 r:2', voice='warm', e=2, dbl=('kalimba', 0), dbl_vel=0.35),
], extra=lambda T: T.fx('crackle', 0, T.beats, 0.25))

TRACKS['crystal_empire'] = place('crystal_empire', 'The Crystal Waltz', 132, 4, 'waltz', [
    dict(name='intro', prog='C | C | F | G7', mel=None, e=0),
    dict(name='A', prog='C | C | Am | Am | F | G | C | G7 | C | C | Am | Am | Dm | G7 | C | C',
         mel='G5:4 E5 F5 | G5:4 C6:2 | B5:2 A5:2 E5:2 | A5:6 | A5:4 F5 G5 | B5:4 D6:2 | C6:2 E6:2 D6 C6 | B5:6 | '
             'G5:4 E5 F5 | G5:4 E6:2 | D6:2 C6:2 A5:2 | E5:6 | F5:2 A5:2 D6:2 | C6:2 B5:2 G5:2 | C6:6 | r:6',
         voice='mbox', e=1, dbl=('sparkle', 0), dbl_vel=0.3),
    dict(name='B', prog='F | F | C | C | G | G7 | C | C | F | Fm | C/E | A7 | Dm | G7 | C | G7',
         mel='C6:4 A5:2 | A5:4 G5:2 | G5:4 E5:2 | E5:6 | D5:2 G5:2 B5:2 | D6:4 B5:2 | C6:2 G5:2 E5:2 | G5:6 | '
             'A5:4 C6:2 | Ab5:4 C6:2 | G5:4 E5:2 | C#5:2 E5:2 A5:2 | F5:4 A5:2 | B5:2 D6:2 G5:2 | C6:6 | r:6',
         voice='flute', e=1),
    dict(name="A'", prog='C | C | Am | Am | F | G | C | G7 | C | C | Am | Am | Dm | G7 | C | C',
         mel='G5:4 E5 F5 | G5:4 C6:2 | B5:2 A5:2 E5:2 | A5:6 | A5:4 F5 G5 | B5:4 D6:2 | C6:2 E6:2 D6 C6 | B5:6 | '
             'G5:4 E5 F5 | G5:4 E6:2 | D6:2 C6:2 A5:2 | E5:6 | F5:2 A5:2 D6:2 | C6:2 B5:2 G5:2 | C6:6 | r:6',
         voice='mbox', e=2, harm='flute', dbl=('sparkle', 0), dbl_vel=0.3),
], bpb=3, rev=1.45, extra=lambda T: [T.add('bell', T.at(b), 3, 72 + (T.chord_at(T.at(b)).root - 72) % 12, 0.3)
                                    for b in range(0, T.bars, 4)])

TRACKS['yakyakistan'] = place('yakyakistan', 'Yak Stomp', 110, 2, 'stomp', [
    dict(name='intro', prog='Am | E7', mel=None, e=1),
    dict(name='A', prog='Am | Am | G | G | F | G | Am:2 E7:2 | Am',
         mel='A4 A4 C5 E5 A5:2 E5:2 | G5 E5 C5 E5 A4:4 | G4 G4 B4 D5 G5:2 D5:2 | F5 D5 B4 D5 G4:4 | '
             'A4 C5 F5 A5 G5:2 F5:2 | G5 B5 D6 B5 G5:2 D5:2 | C5:2 A4:2 G#4:2 B4:2 | A4:4 r:2 E5:2', voice='accordion',
         e=1),
    dict(name='B', prog='C | G | Am | E7 | F | C | Dm:2 E7:2 | Am',
         mel='E5 E5 G5 C6 E6:2 C6:2 | D6 D6 B5 G5 D5:4 | C6 C6 A5 E5 A5:4 | B5 B5 G#5 E5 B4:4 | '
             'A5 A5 F5 C5 F5:2 A5:2 | G5 G5 E5 C5 E5:2 G5:2 | F5:2 D5:2 E5:2 G#5:2 | A5:4 r:4', voice='fiddle', e=2,
         harm='counter'),
    dict(name="A'", prog='Am | Am | G | G | F | G | Am:2 E7:2 | Am',
         mel='A4 A4 C5 E5 A5:2 E5:2 | G5 E5 C5 E5 A4:4 | G4 G4 B4 D5 G5:2 D5:2 | F5 D5 B4 D5 G4:4 | '
             'A4 C5 F5 A5 G5:2 F5:2 | G5 B5 D6 B5 G5:2 D5:2 | C5:2 A4:2 G#4:2 B4:2 | A4:4 r:2 E5:2', voice='accordion',
         e=2, dbl=('fiddle', 1), dbl_vel=0.45),
], rev=1.2)

TRACKS['castle_two_sisters'] = place('castle_two_sisters', 'The Tree of Harmony', 84, 5, 'lullaby', [
    dict(name='intro', prog='Am | F | Dm | E7', mel=None, e=0),
    dict(name='lullaby', prog='Am | F | C | G | Am | F | Dm | E7',
         mel='A4:2 C5:2 E5:2 | F5:4 E5:2 | G5:2 E5:2 C5:2 | D5:6 | A4:2 C5:2 E5:2 | A5:4 G5:2 | F5:2 E5:2 D5:2 | E5:6',
         voice='mbox', e=0),
    dict(name="lullaby'", prog='Am | F | C | G | Am | F | Dm | E7',
         mel='A4:2 C5:2 E5:2 | F5:4 E5:2 | G5:2 E5:2 C5:2 | D5:6 | A4:2 C5:2 E5:2 | A5:4 G5:2 | F5:2 E5:2 D5:2 | E5:6',
         voice='mbox', e=1, dbl=('flute', -1), dbl_vel=0.5),
    dict(name='harmony', prog='A | E/G# | F#m | D | A | E/G# | D | E7',
         mel='C#5:2 E5:2 A5:2 | B5:4 G#5:2 | A5:4 F#5:2 | F#5:6 | E5:2 A5:2 C#6:2 | B5:4 E5:2 | F#5:2 A5:2 D6:2 | '
             'B5:2 G#5:2 E5:2', voice='warm', e=1),
    dict(name="harmony'", prog='A | E/G# | F#m | D | A | E/G# | D | E7',
         mel='C#5:2 E5:2 A5:2 | B5:4 G#5:2 | A5:4 F#5:2 | F#5:6 | E5:2 A5:2 C#6:2 | B5:4 E5:2 | F#5:2 A5:2 D6:2 | '
             'B5:2 G#5:2 E5:2', voice='warm', e=2, harm='counter', dbl=('mbox', 1), dbl_vel=0.3),
], bpb=3, rev=1.5)

# the places in the order of the atlas route (map location id -> track id: the same names)
PLACES = ['canterlot', 'cloudsdale', 'ponyville', 'sweet_apple_acres', 'everfree', 'neighagra_falls',
          'rainbow_falls', 'manehattan', 'baltimare', 'griffonstone', 'dragon_lands', 'mount_aris', 'klugetown',
          'changeling_kingdom', 'appleloosa', 'las_pegasus', 'vanhoover', 'crystal_empire', 'yakyakistan',
          'castle_two_sisters']
assert all(p in TRACKS for p in PLACES)


# =========================================================================== the trailer cue (not a game loop)
# tools/trailer2_make.py cuts the picture to these bars (100 BPM: one bar = 2.4 s = 144 video frames).
# It tells the game's story with the game's own themes, all in D: the Atlas theme (title), the map theme
# (flying, the places), its B phrase (helping), the 'wide world' brass, the night music box, the race
# riff (in B minor), and the Atlas theme again over the end card, ending on a ringing D.
EXTRA = {}
TRAILER_BARS = [  # (first bar, shot section) - the edit (tools/trailer2_make.py SHOTS)
    (0, 'title'), (4, 'fly anywhere'), (9, 'places montage'), (15, 'harmony'), (17, 'help'),
    (23, 'the world changes'), (25, 'nightfall'), (27, 'friends + races'), (34, 'postcards + journal'),
    (38, 'end card')]


def trailer():
    from audio import compose as CC
    T = Track('trailer', 'Equestria Atlas - trailer', 100, 42, rev=1.1, lufs=-14.0)
    T.loop = False
    for b, n in TRAILER_BARS:
        T.mark(b, n)
    d = lambda spec: CC.trc(spec, -5)             # the map theme's chords, G -> D
    T.prog('D | D | Gmaj7 | Em7 A7:2')            # 0 title (bar 3: Em7 2 beats, A7 2)
    T.prog(d('G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | D7sus4:2 D7:2'))      # 4..11 map theme A
    T.prog(d('G | D/F# | Em7'))                   # 12..14 A' (the montage builds)
    T.prog('Gmaj7 | A7sus4:2 A7:2')               # 15..16 harmony
    T.prog(d('Em | C | G | D/F# | Em | C'))       # 17..22 help: the B phrase
    T.prog('Gmaj9 | Bm7')                         # 23..24 the world changes: Atlas theme, brass
    T.prog('Em7 | A7sus4:2 A7:2')                 # 25..26 nightfall
    T.prog('Bm | G | D | A | Bm | G | Em7')       # 27..33 friends + races (B minor riff)
    T.prog(d('G/B | C | Am7:2 D7:2 | G'))         # 34..37 postcards + journal
    T.prog('D | Gmaj7 | Em7 | A7:2 D:2')          # 38..41 end card
    # ---- 0..3 the title
    T.fx('riser', 0, 4.0, 0.3, ends_at=T.at(1))
    T.mel('mbox', 0, 'r:4 F#5:2 A5:2', vel=0.7)
    T.drum('crash', T.at(1), 0.8)
    T.add('timp', T.at(1), 1.5, 38, 0.8)
    T.add('bell', T.at(1), 4, 74, 0.5)
    a = T.mel('warm', 1, 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2', vel=0.85)
    T.double(a, 'sparkle', octave=1, vel=0.25)
    T.pad('bed', 0, 4, lo=50, hi=67, n=4, vel=0.36)
    T.pad('choir', 1, 4, lo=57, hi=72, n=3, vel=0.22)
    T.arp('harp', T.at(1), T.at(4), step=0.5, lo=62, hi=83, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.bassline('upright', 1, 4, [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.6)
    T.roll('snare_soft', T.at(3, 3), T.at(4), 0.25, 0.25, 0.7)
    # ---- 4..14 fly anywhere + the places: the map theme over the folk band, building
    bounce = [(0, 'R', 1.4), (1.5, '5', 0.45, 0.6), (2, 'R', 0.9, 0.85), (3, 'R+12', 0.45, 0.6), (3.5, '5', 0.45, 0.6)]
    T.bassline('bass', 4, 15, bounce, lo=36, vel=0.74)
    T.comp('strum', 4, 15, [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)], lo=55, hi=72,
           n=3, vel=0.5, human=0.6)
    T.pad('pad', 4, 15, lo=50, hi=67, n=4, vel=0.27)
    T.drums(4, 9, {'kick': 'X.......X.....x.', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.4, vel=0.78)
    T.drums(9, 15, {'kick': 'X.......X.....x.', 'snare': '....x.......x...', 'hat': 'x.o.x.o.x.o.x.o.',
                    'shaker': 'g.g.g.g.g.g.g.g.'}, human=0.3, vel=0.8)
    T.drum('crash_soft', T.at(4), 0.6)
    for b in range(9, 15):                        # one cut per bar in the montage: a crash on each
        T.drum('crash_soft' if b % 2 else 'crash', T.at(b), 0.5)
    T.mel('lead', 4, MAP_A, tr=-5, vel=0.82)
    a2 = T.mel('lead', 12, 'D5:2 G5:2 A5 B5:3 | A5:2 F#5 E5 D5:4 | E5 F#5 G5 B5 A5:2 G5 E5', tr=-5, vel=0.86)
    T.harmonize(a2, 'harm', vel=0.58)
    T.arp('harp', T.at(9), T.at(15), step=0.25, lo=67, hi=86, vel=0.24, pattern='updown', pans=(-0.4, 0.4))
    T.roll('snare', T.at(14, 3), T.at(15), 0.25, 0.3, 0.8)
    # ---- 15..16 harmony: the Tree of Harmony - choir, bells, music box
    T.drum('crash', T.at(15), 0.6)
    T.pad('choir', 15, 17, lo=57, hi=74, n=4, vel=0.3)
    T.pad('bed', 15, 17, lo=48, hi=64, n=4, vel=0.36)
    T.mel('mbox', 15, 'B5:3 A5 F#5:4 | E5:4 C#5:2 A4:2', vel=0.8)
    T.arp('harp', T.at(15), T.at(17), step=0.25, lo=69, hi=88, vel=0.28, pattern='up', pans=(-0.4, 0.4))
    for bt in (0, 2, 4, 6):
        T.add('bell', T.at(15, bt), 2, 74 if bt % 4 == 0 else 81, 0.3)
    T.bassline('upright', 15, 17, [(0, 'R', 3.8)], lo=36, vel=0.6)
    # ---- 17..22 help the places: the B phrase on the ocarina, rising into the mend
    T.mel('ocarina', 17, 'B4:3 E5 G5:2 F#5 E5 | E5:3 D5 C5:2 B4 C5 | D5:3 B4 G4:2 A4 B4 | A4:6 r:2 | '
                         'B4:3 E5 G5:2 A5 B5 | C6:3 B5 A5:2 G5 E5', tr=-5, vel=0.84)
    T.bassline('bass', 17, 23, [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.64)
    T.arp('harp', T.at(17), T.at(23), step=0.5, lo=62, hi=81, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.pad('bed', 17, 23, lo=50, hi=67, n=4, vel=0.36)
    T.drums(17, 21, {'kick_soft': 'x.......x.......', 'rim': '............o...', 'shaker': 'g.g.g.g.g.g.g.g.'},
            human=0.5, vel=0.72)
    T.drums(21, 23, {'kick': 'X.......X.......', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.4, vel=0.78)
    T.mel('counter', 19, 'D5:8 | C#5:8 | D5:4 B4:4 | E5:8', vel=0.45)
    T.roll('snare', T.at(22, 2), T.at(23), 0.25, 0.25, 0.85)
    T.fx('swell', 0, 4.0, 0.35, ends_at=T.at(23))
    # ---- 23..24 the world changes: the Atlas theme on brass, wide
    T.drum('crash', T.at(23), 0.8)
    T.add('timp', T.at(23), 1.0, 31, 0.75)
    T.add('timp', T.at(24), 1.0, 35, 0.7)
    c = T.mel('brass', 23, 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2', vel=0.88)
    T.harmonize(c, 'brass2', vel=0.6)
    T.pad('choir', 23, 25, lo=57, hi=74, n=4, vel=0.3)
    T.pad('pad', 23, 25, lo=48, hi=64, n=4, vel=0.32)
    T.arp('harp', T.at(23), T.at(25), step=0.25, lo=67, hi=88, vel=0.26, pattern='up', pans=(-0.4, 0.4))
    T.bassline('bass', 23, 25, [(0, 'R', 1.8), (2, 'R', 0.9, 0.7), (3, '5', 0.9, 0.7)], lo=36, vel=0.74)
    T.drums(23, 25, {'boom': 'X.......X.......', 'snare_soft': '....x.......x...', 'hat_soft': '..g...g...g...g.'},
            human=0.3, vel=0.8)
    # ---- 25..26 nightfall: the music box and the keys
    T.mel('mbox', 25, 'B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 E5:4', vel=0.8)
    T.comp('keys', 25, 27, [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.42, human=0.7)
    T.pad('bed', 25, 27, lo=50, hi=67, n=4, vel=0.38)
    T.bassline('upright', 25, 27, [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.55)
    for bt in (0, 4):
        T.add('bell', T.at(25, bt), 3, 79, 0.28)
    T.arp('kalimba', T.at(25), T.at(27), step=0.5, lo=62, hi=81, vel=0.36, group=[0, 2, 1, 3, 2, 4, 3, 1],
          pans=(0.3, -0.1))
    T.fx('riser', 0, 4.0, 0.35, ends_at=T.at(27))
    T.roll('snare', T.at(26, 3), T.at(27), 0.125, 0.2, 0.8)
    # ---- 27..33 friends + races: the race riff in B minor, driving
    k_rock(T, 27, 30, 1)
    k_rock(T, 30, 34, 2)
    T.drum('crash', T.at(27), 0.8)
    T.drum('crash', T.at(30), 0.7)
    r = T.mel('lead', 27, 'A5 E5 A5 B5 C6:2 B5 A5 | F5:2 A5 C6 A5:2 G5 F5 | E5 G5 C6 E6 D6:2 C6 G5 | '
                          'B5:3 A5 G5:2 D5:2 | A5 E5 A5 B5 C6:2 B5 A5 | F5:2 A5 C6 F6:2 E6 C6 | D6:2 C6 A5 F5:2 A5:2',
              tr=2, vel=0.84)
    T.harmonize(r[16:], 'harm', vel=0.55)
    # ---- 34..37 postcards + journal: the map theme's close, light again
    T.mel('whistle', 34, 'D5:2 G5:2 A5 B5:3 | C6:2 B5 A5 G5:2 E5:2 | A5:2 B5 C6 B5:2 A5:2 | G5:6 r:2', tr=-5,
          vel=0.84)
    T.bassline('bass', 34, 38, bounce, lo=36, vel=0.7)
    T.comp('strum', 34, 38, [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)], lo=55, hi=72,
           n=3, vel=0.48, human=0.6)
    T.pad('pad', 34, 38, lo=50, hi=67, n=4, vel=0.27)
    T.drums(34, 38, {'kick_soft': 'X.......x.......', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.4, vel=0.75)
    T.mel('sparkle', 36, 'r:4 B5 D6 G6:2', tr=-5, vel=0.45)
    T.roll('snare', T.at(37, 2), T.at(38), 0.25, 0.25, 0.85)
    # ---- 38..41 the end card: the Atlas theme, everyone, and a ringing D
    T.drum('crash', T.at(38), 0.85)
    T.add('timp', T.at(38), 1.0, 38, 0.8)
    e1 = T.mel('brass', 38, 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 D5:4',
               vel=0.86)
    T.double(e1, 'warm', octave=1, vel=0.5)
    T.harmonize(e1, 'brass2', vel=0.55)
    T.pad('choir', 38, 42, lo=57, hi=74, n=4, vel=0.3)
    T.pad('pad', 38, 42, lo=48, hi=67, n=4, vel=0.3)
    T.arp('harp', T.at(38), T.at(41, 2), step=0.25, lo=67, hi=88, vel=0.26, pattern='updown', pans=(-0.4, 0.4))
    T.bassline('bass', 38, 41, [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.72)
    T.drums(38, 41, {'kick': 'X.......X.......', 'snare_soft': '....x.......x...', 'hat_soft': 'g.g.g.g.g.g.g.g.'},
            human=0.3, vel=0.78)
    T.add('bass', T.at(41, 2), 2.0, 38, 0.75)
    T.add('timp', T.at(41, 2), 1.0, 38, 0.8)
    T.drum('crash', T.at(41, 2), 0.75)
    T.add('bell', T.at(41, 2), 2, 74, 0.5)
    T.add('bell', T.at(41, 2), 2, 81, 0.35)
    T.mel('sparkle', 41, 'r:4 A5 D6 F#6 A6', vel=0.5)
    return T


EXTRA['trailer'] = trailer


# =========================================================================== the intro video (4 min 19 s)
INTRO_BARS = [  # (first bar, chapter) - the edit (tools/intro_make.py SHOTS)
    (0, 'title'), (4, 'morning'), (12, 'places'), (28, 'help'), (36, 'the world changes'),
    (38, 'friends + races'), (48, 'golden hour'), (56, 'nightfall'), (60, 'after dark'), (80, 'postcards'),
    (86, 'photo book'), (98, 'wardrobe'), (102, 'end card')]
GOLD, GOLD_BARS = 48, 8         # the golden-hour chapter (Nov 2026) was put in at bar 48: later bars moved on 8


def _B(bar):
    """A bar of the first cut's score (before the golden-hour chapter) -> its bar now."""
    return bar + GOLD_BARS if bar >= GOLD else bar
# the postcard theme (G major): a light, curious tune for the camera and the photo book
SNAP_A = ('B5:2 D6 B5 G5:2 A5 B5 | G5:3 E5 D5:2 E5 G5 | E5:2 G5 B5 C6:2 B5 G5 | A5:4 F#5:2 r:2 | '
          'B5:2 D6 B5 G5:2 A5 B5 | G5:3 E5 D5:2 B4 D5 | C5:2 E5 A5 F#5:2 D5 F#5 | G5:6 r:2')
SNAP_B = ('E5:3 G5 C6:2 B5 A5 | F#5:3 A5 D6:2 C6 A5 | B5:2 A5 F#5 D5:2 F#5 A5 | G5:4 E5:2 r:2 | '
          'E5:3 A5 C6:2 B5 A5 | F#5:3 A5 C6:2 A5 F#5 | G5:2 B5 D6 B5:2 G5:2 | A5:4 r:4')


def intro():
    from audio import compose as CC
    T = Track('intro', 'Equestria Atlas - a tour of the game', 100, 100 + GOLD_BARS, rev=1.15, lufs=-14.0)
    T.loop = False
    for b, n in INTRO_BARS:
        T.mark(b, n)                              # (INTRO_BARS are already the new bars)
    d = lambda spec: CC.trc(spec, -5)             # the map theme's chords, G -> D
    T.prog('D | D | Gmaj7 | Em7 A7:2')                                                   # 0 title
    T.prog(d('G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | D7sus4:2 D7:2'))          # 4 morning (A)
    T.prog(d('G | D/F# | Em7 | Cadd9 | G/B | C | Am7:2 D7:2 | G'))                      # 12 places (A')
    T.prog(d('Cmaj9 | Em7 | Am7 | D7sus4:2 D7:2 | Cmaj9 | Bm7 | Am7:2 Bm7:2 | C:2 D7sus4:1 D7:1'))  # 20 (C)
    T.prog(d('Em | C | G | D/F# | Em | C | Am7 | D7sus4:2 D7:2'))                        # 28 help (B)
    T.prog('Gmaj9 | A7sus4:2 A7:2')                                                      # 36 the world changes
    T.prog('Bm | G | Bm | G | D | A | Bm | G | Em7 | F#7')                                # 38 friends + races
    T.prog(d('Em | C | G | D/F# | Em | C | Am7 | D7sus4:2 D7:2'))                         # 48 golden hour (B)
    T.prog('Bm7 | Gmaj7 | Em7 | A7sus4:2 A7:2')                                          # 56 nightfall
    T.prog('Dmaj7 | A7sus4:2 A7:2')                                                      # 60 after dark
    T.prog(d('Gmaj7 | D/F# | Em7 | Cmaj7 | G/B | Cmaj7 | Am7:2 D7:2 | Gmaj7'))           # 62 (night A')
    T.prog(d('Cmaj9 | Em7 | Am7 | D7sus4:2 D7:2 | Cmaj9 | Bm7 | Am7:2 Bm7:2 | Cmaj7:2 D7sus4:1 D7:1'))  # 70
    T.prog('Dmaj7 | D7sus4:2 D7:2')                                                      # 78 -> G
    T.prog('G | Em7 | Cmaj7 | D7sus4:2 D7:2 | G | Em7 | Am7:2 D7:2 | G')                 # 80 postcards (G)
    T.prog('C | D | Bm7 | Em7 | Am7 | D7 | G | D7sus4:2 D7:2')                           # 88 photo book
    T.prog('G | A7sus4:2 A7:2')                                                          # 96 -> D
    T.prog(d('G | D/F# | Em7 | Cadd9'))                                                  # 98 wardrobe
    T.prog('D | Gmaj7 | Em7 | A7:2 D:2 | Gmaj7 | D')                                     # 102 end card
    bounce = [(0, 'R', 1.4), (1.5, '5', 0.45, 0.6), (2, 'R', 0.9, 0.85), (3, 'R+12', 0.45, 0.6), (3.5, '5', 0.45, 0.6)]
    strum = [(0.5, 0.4, 0.75), (1.5, 0.4, 0.6), (2.5, 0.4, 0.75), (3.5, 0.4, 0.6)]
    riff = dict(step=0.5, lo=62, hi=81, group=[0, 2, 1, 3, 2, 4, 3, 1], pans=(0.3, -0.1))

    # ---- 0..3 the title: the Atlas theme
    T.fx('riser', 0, 4.0, 0.3, ends_at=T.at(_B(1)))
    T.mel('mbox', _B(0), 'r:4 F#5:2 A5:2', vel=0.7)
    T.drum('crash', T.at(_B(1)), 0.8)
    T.add('timp', T.at(_B(1)), 1.5, 38, 0.8)
    T.add('bell', T.at(_B(1)), 4, 74, 0.5)
    a = T.mel('warm', _B(1), 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2', vel=0.85)
    T.double(a, 'sparkle', octave=1, vel=0.25)
    T.pad('bed', _B(0), _B(4), lo=50, hi=67, n=4, vel=0.36)
    T.pad('choir', _B(1), _B(4), lo=57, hi=72, n=3, vel=0.22)
    T.arp('harp', T.at(_B(1)), T.at(_B(4)), step=0.5, lo=62, hi=83, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.bassline('upright', _B(1), _B(4), [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.6)
    T.roll('snare_soft', T.at(_B(3), 3), T.at(_B(4)), 0.25, 0.25, 0.7)

    # ---- 4..11 morning: the map theme over a light folk band and the wind riff
    T.arp('kalimba', T.at(_B(4)), T.at(_B(12)), vel=0.42, **riff)
    T.bassline('bass', _B(4), _B(12), bounce, lo=36, vel=0.7)
    T.comp('strum', _B(4), _B(12), strum, lo=55, hi=72, n=3, vel=0.46, human=0.6)
    T.pad('bed', _B(4), _B(12), lo=50, hi=67, n=4, vel=0.3)
    T.drums(_B(4), _B(12), {'kick_soft': 'X.......x.......', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.45, vel=0.76)
    T.drum('crash_soft', T.at(_B(4)), 0.6)
    T.mel('lead', _B(4), MAP_A, tr=-5, vel=0.8)
    T.mel('sparkle', _B(7), 'r:4 B5 D6 G6:2', tr=-5, vel=0.42)
    T.roll('snare_soft', T.at(_B(11), 3), T.at(_B(12)), 0.25, 0.3, 0.75)

    # ---- 12..27 the places: A' with the full band (a crash every two bars, one per place), then the wide C
    T.bassline('bass', _B(12), _B(20), bounce, lo=36, vel=0.74)
    T.comp('strum', _B(12), _B(20), strum, lo=55, hi=72, n=3, vel=0.5, human=0.6)
    T.pad('pad', _B(12), _B(20), lo=50, hi=67, n=4, vel=0.27)
    T.drums(_B(12), _B(20), {'kick': 'X.......X.....x.', 'snare': '....x.......x...', 'hat': 'x.o.x.o.x.o.x.o.',
                     'shaker': 'g.g.g.g.g.g.g.g.'}, human=0.3, vel=0.8)
    T.arp('harp', T.at(_B(12)), T.at(_B(20)), step=0.25, lo=67, hi=86, vel=0.22, pattern='updown', pans=(-0.4, 0.4))
    a2 = T.mel('lead', _B(12), MAP_A2, tr=-5, vel=0.86)
    T.harmonize(a2, 'harm', vel=0.56)
    for b in range(_B(12), _B(28), 2):
        T.drum('crash' if b in (12, 20) else 'crash_soft', T.at(b), 0.7 if b in (12, 20) else 0.5)
    T.bassline('bass', _B(20), _B(28), [(0, 'R', 1.8), (2, 'R', 0.9, 0.7), (3, '5', 0.9, 0.7)], lo=36, vel=0.72)
    T.pad('choir', _B(20), _B(28), lo=55, hi=72, n=3, vel=0.26)
    T.pad('pad', _B(20), _B(28), lo=48, hi=64, n=4, vel=0.3)
    T.arp('harp', T.at(_B(20)), T.at(_B(28)), step=0.25, lo=67, hi=86, vel=0.24, pattern='up', pans=(-0.4, 0.4))
    T.drums(_B(20), _B(28), {'boom': 'X...............', 'kick_soft': '........x.......', 'snare_soft': '....x.......x...',
                     'hat_soft': '..g...g...g...g.'}, human=0.3, vel=0.75)
    for b in (_B(20), _B(22), _B(24), _B(26)):
        T.add('timp', T.at(b), 1.0, 26 + (T.chord_at(T.at(b)).root - 26) % 12 + 12, 0.55)
    c = T.mel('brass', _B(20), MAP_C, tr=-5, vel=0.82)
    T.harmonize(c[18:], 'brass2', vel=0.55)
    T.roll('snare_soft', T.at(_B(27), 2), T.at(_B(28)), 0.25, 0.25, 0.75)

    # ---- 28..35 help the places: the B phrase on the ocarina, rising into the mend
    T.mel('ocarina', _B(28), MAP_B, tr=-5, vel=0.84)
    T.bassline('bass', _B(28), _B(36), [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.64)
    T.arp('harp', T.at(_B(28)), T.at(_B(36)), step=0.5, lo=62, hi=81, vel=0.32, pattern='updown', pans=(-0.35, 0.35))
    T.pad('bed', _B(28), _B(36), lo=50, hi=67, n=4, vel=0.36)
    T.drums(_B(28), _B(32), {'kick_soft': 'x.......x.......', 'rim': '............o...', 'shaker': 'g.g.g.g.g.g.g.g.'},
            human=0.5, vel=0.72)
    T.drums(_B(32), _B(36), {'kick': 'X.......X.......', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.4, vel=0.78)
    T.drum('crash_soft', T.at(_B(28)), 0.5)
    T.mel('counter', _B(32), 'G4:8 | E4:8 | C5:4 A4:4 | A4:4 r:4', tr=-5 + 12, vel=0.42)
    T.roll('snare', T.at(_B(35), 2), T.at(_B(36)), 0.25, 0.25, 0.85)
    T.fx('swell', 0, 4.0, 0.35, ends_at=T.at(_B(36)))

    # ---- 36..37 the world changes: the Atlas theme on brass, wide
    T.drum('crash', T.at(_B(36)), 0.8)
    T.add('timp', T.at(_B(36)), 1.0, 31, 0.75)
    T.add('timp', T.at(_B(37)), 1.0, 33, 0.7)
    w = T.mel('brass', _B(36), 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 E5:2 C#5:2', vel=0.88)
    T.harmonize(w, 'brass2', vel=0.6)
    T.pad('choir', _B(36), _B(38), lo=57, hi=74, n=4, vel=0.3)
    T.pad('pad', _B(36), _B(38), lo=48, hi=64, n=4, vel=0.32)
    T.arp('harp', T.at(_B(36)), T.at(_B(38)), step=0.25, lo=67, hi=88, vel=0.26, pattern='up', pans=(-0.4, 0.4))
    T.bassline('bass', _B(36), _B(38), [(0, 'R', 1.8), (2, 'R', 0.9, 0.7), (3, '5', 0.9, 0.7)], lo=36, vel=0.74)
    T.drums(_B(36), _B(38), {'boom': 'X.......X.......', 'snare_soft': '....x.......x...', 'hat_soft': '..g...g...g...g.'},
            human=0.3, vel=0.8)

    # ---- 38..47 friends + races: the race riff in B minor (two bars of groove, then the tune)
    k_rock(T, _B(38), _B(44), 1)
    k_rock(T, _B(44), GOLD, 2)
    T.drum('crash', T.at(_B(38)), 0.8)
    T.drum('crash', T.at(_B(44)), 0.7)
    T.arp('kalimba', T.at(_B(38)), T.at(_B(40)), vel=0.45, **riff)
    r = T.mel('lead', _B(40), RACE_A, tr=2, vel=0.84)
    T.harmonize(r[16:], 'harm', vel=0.55)
    T.roll('snare', T.at(_B(47), 3), T.at(GOLD), 0.125, 0.3, 0.8)

    # ---- 48..55 golden hour: the B phrase again, on the warm voice, the band softening into the evening
    T.drum('crash_soft', T.at(GOLD), 0.55)
    T.mel('warm', GOLD, MAP_B, tr=-5, vel=0.8)
    T.mel('counter', GOLD + 4, 'G4:8 | E4:8 | C5:4 A4:4 | A4:4 r:4', tr=-5 + 12, vel=0.38)
    T.comp('strum', GOLD, GOLD + 8, strum, lo=55, hi=72, n=3, vel=0.38, human=0.6)
    T.pad('bed', GOLD, GOLD + 8, lo=50, hi=67, n=4, vel=0.34)
    T.arp('harp', T.at(GOLD), T.at(GOLD + 8), step=0.5, lo=62, hi=81, vel=0.28, pattern='updown', pans=(-0.35, 0.35))
    T.bassline('upright', GOLD, GOLD + 8, [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.6)
    T.drums(GOLD, GOLD + 8, {'kick_soft': 'x.......x.......', 'brush': '....o.......o...', 'shaker': 'g.g.g.g.g.g.g.g.'},
            human=0.5, vel=0.66)
    T.add('bell', T.at(GOLD + 4), 3, 78, 0.24)
    T.roll('brush', T.at(GOLD + 7, 3), T.at(GOLD + 8), 0.25, 0.25, 0.6)

    # ---- 56..59 nightfall: the music box and the keys, the sky darkening
    T.drum('crash_soft', T.at(_B(48)), 0.55)
    T.mel('mbox', _B(48), 'F#5:3 D5 B4:2 D5 F#5 | B5:4 A5:2 F#5:2 | G5:3 F#5 E5:2 B4 D5 | E5:4 C#5:2 A4:2', vel=0.8)
    T.comp('keys', _B(48), _B(52), [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.42, human=0.7)
    T.pad('bed', _B(48), _B(52), lo=50, hi=67, n=4, vel=0.38)
    T.bassline('upright', _B(48), _B(52), [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.55)
    for b in (_B(48), _B(50)):
        T.add('bell', T.at(b), 3, 78, 0.26)
    T.arp('kalimba', T.at(_B(48)), T.at(_B(52)), vel=0.34, **riff)
    T.fx('swell', 0, 4.0, 0.3, ends_at=T.at(_B(52)))

    # ---- 60..79 after dark: the night map theme (music box, flute, keys, harp, bells), a slow glow
    T.add('bell', T.at(_B(52)), 4, 74, 0.34)
    T.arp('kalimba', T.at(_B(52)), T.at(_B(54)), vel=0.36, **riff)
    T.bassline('upright', _B(52), _B(72), [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.56)
    T.pad('bed', _B(52), _B(72), lo=50, hi=67, n=4, vel=0.36)
    T.comp('keys', _B(54), _B(62), [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.4, human=0.7)
    n1 = T.mel('flute', _B(54), MAP_A2, tr=-5, vel=0.74)
    T.harmonize(n1, 'kalimba', vel=0.48)
    T.drums(_B(54), _B(62), {'brush': '....o.......o...', 'kick_soft': 'g.......g.......', 'shaker': 'g...g...g...g...'},
            human=0.5, vel=0.6)
    T.mel('sparkle', _B(57), 'r:4 B5 D6 G6:2', tr=-5, vel=0.36)
    T.mel('sparkle', _B(61), 'r:4 A5 D6 F#6:2', tr=-5, vel=0.36)
    T.drum('crash_soft', T.at(_B(62)), 0.45)
    T.pad('choir', _B(62), _B(70), lo=55, hi=72, n=3, vel=0.26)
    T.arp('harp', T.at(_B(62)), T.at(_B(70)), step=0.5, lo=64, hi=84, vel=0.26, pattern='up', pans=(-0.35, 0.35))
    T.mel('warm', _B(62), MAP_C, tr=-5, vel=0.78)
    for b in range(_B(62), _B(70), 2):
        T.add('bell', T.at(b), 2, 67 + (T.chord_at(T.at(b)).root - 67) % 12, 0.24)
    T.drums(_B(62), _B(70), {'brush': '....o.......o...', 'kick_soft': 'g.......g.......'}, human=0.5, vel=0.6)
    T.comp('keys', _B(70), _B(72), [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.4, human=0.7)
    T.mel('mbox', _B(70), 'F#5:3 A5 D6:2 C#6 A5 | G5:4 F#5:2 r:2', vel=0.76)
    T.roll('brush', T.at(_B(71), 3), T.at(_B(72)), 0.25, 0.25, 0.6)

    # ---- 80..97 postcards and the photo book: the postcard theme (G), plucky and light
    T.drum('crash_soft', T.at(_B(72)), 0.55)
    T.arp('kalimba', T.at(_B(72)), T.at(_B(88)), vel=0.36, **riff)
    T.bassline('upright', _B(72), _B(88), [(0, 'R', 0.9), (1.5, '5', 0.45, 0.6), (2, 'R', 0.9, 0.8), (3, '5', 0.9, 0.7)],
               lo=38, vel=0.66)
    T.comp('strum', _B(72), _B(88), strum, lo=55, hi=72, n=3, vel=0.4, human=0.6)
    T.pad('bed', _B(72), _B(90), lo=50, hi=67, n=4, vel=0.3)
    T.drums(_B(72), _B(80), {'kick_soft': 'X.......x.......', 'rim': '....o.......o...', 'shaker': 'g.o.g.o.g.o.g.o.'},
            human=0.45, vel=0.74)
    s1 = T.mel('whistle', _B(72), SNAP_A, vel=0.82)
    T.harmonize(s1[16:], 'counter', vel=0.5)
    T.drums(_B(80), _B(88), {'kick_soft': 'X.......x.......', 'snare_soft': '....x.......x...', 'shaker': 'g.o.g.o.g.o.g.o.',
                     'hat_soft': '..g...g...g...g.'}, human=0.4, vel=0.76)
    T.drum('crash_soft', T.at(_B(80)), 0.55)
    s2 = T.mel('flute', _B(80), SNAP_B, vel=0.8)
    T.double(s2, 'mbox', octave=1, vel=0.32)
    T.arp('harp', T.at(_B(80)), T.at(_B(88)), step=0.5, lo=64, hi=84, vel=0.22, pattern='updown', pans=(-0.35, 0.35))
    T.add('bell', T.at(_B(78)), 3, 79, 0.32)                  # J: the journal opens on the photos
    T.mel('sparkle', _B(78), 'r:4 B5 D6 G6:2', vel=0.42)
    T.comp('keys', _B(88), _B(90), [(0, 1.8, 0.75), (2, 1.8, 0.6)], lo=55, hi=72, n=4, vel=0.4, human=0.7)
    T.mel('mbox', _B(88), 'B5:3 A5 G5:2 D5:2 | E5:4 C#5:2 r:2', vel=0.74)
    T.bassline('upright', _B(88), _B(90), [(0, 'R', 1.8), (2, '5', 1.8)], lo=38, vel=0.58)
    T.roll('snare_soft', T.at(_B(89), 2), T.at(_B(90)), 0.25, 0.25, 0.75)

    # ---- 98..101 the wardrobe: the map theme on the whistle over a bright pop groove
    k_pop(T, _B(90), _B(94), 1)
    T.drum('crash', T.at(_B(90)), 0.7)
    T.mel('whistle', _B(90), 'D5:2 G5:2 A5 B5:3 | A5:2 F#5 E5 D5:4 | E5 F#5 G5 B5 A5:2 G5 E5 | D5:6 r:2', tr=-5,
          vel=0.84)
    T.mel('sparkle', _B(93), 'r:4 B5 D6 G6:2', tr=-5, vel=0.45)
    T.roll('snare', T.at(_B(93), 2), T.at(_B(94)), 0.25, 0.25, 0.85)

    # ---- 102..107 the end card: the Atlas theme, everyone, then the music box and a ringing D
    T.drum('crash', T.at(_B(94)), 0.85)
    T.add('timp', T.at(_B(94)), 1.0, 38, 0.8)
    e1 = T.mel('brass', _B(94), 'A4 D5 E5 F#5:3 A5:2 | B5:3 A5 F#5:2 D5:2 | B4 E5 F#5 G5:3 B5:2 | A5:2 G5 F#5 D5:4',
               vel=0.86)
    T.double(e1, 'warm', octave=1, vel=0.5)
    T.harmonize(e1, 'brass2', vel=0.55)
    T.pad('choir', _B(94), _B(100), lo=57, hi=74, n=4, vel=0.3)
    T.pad('pad', _B(94), _B(100), lo=48, hi=67, n=4, vel=0.3)
    T.arp('harp', T.at(_B(94)), T.at(_B(98)), step=0.25, lo=67, hi=88, vel=0.26, pattern='updown', pans=(-0.4, 0.4))
    T.bassline('bass', _B(94), _B(98), [(0, 'R', 1.8), (2, '5', 1.8)], lo=36, vel=0.72)
    T.drums(_B(94), _B(97), {'kick': 'X.......X.......', 'snare_soft': '....x.......x...', 'hat_soft': 'g.g.g.g.g.g.g.g.'},
            human=0.3, vel=0.78)
    T.add('timp', T.at(_B(97), 2), 1.0, 38, 0.7)
    T.mel('mbox', _B(98), 'B5:3 A5 F#5:4 | D5:8', vel=0.74)
    T.arp('harp', T.at(_B(98)), T.at(_B(99), 2), step=0.5, lo=62, hi=83, vel=0.28, pattern='updown', pans=(-0.35, 0.35))
    T.bassline('upright', _B(98), _B(100), [(0, 'R', 3.8)], lo=38, vel=0.6)
    T.add('bell', T.at(_B(99)), 4, 74, 0.45)
    T.add('bell', T.at(_B(99)), 4, 81, 0.3)
    T.mel('sparkle', _B(99), 'r:2 A5 D6 F#6 A6 r:2', vel=0.42)
    return T


EXTRA['intro'] = intro
