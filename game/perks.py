"""Balloon perks: every envelope in the wardrobe (game/compose.PATTERNS, the journal's BALLOONS page)
does something of its own while it is worn. The perks are small helps, never needed to finish:

  EXPLORER      WIDE EYES      the fog clears 40 % further round the balloon
  ORCHARD       GREEN THUMB    picks up the hidden stamps from twice as far
  CLEAR SKY     CALM AIR       wind and gusts never push it
  SUNSET        TIME FLIES     the world clock runs three times as fast (dawn, sunset and night sooner)
  MINT CREAM    GLIDER         keeps gliding when you let go
  CRYSTAL       SPARKLE SENSE  sets off sparkles (SPACE) from half as far again
  LEMONADE      FIZZ           SHIFT boosts a quarter faster
  ROYAL         ROYAL COMPASS  a gold arrow on the map points to the nearest place not found yet
  STARRY NIGHT  NIGHT FLYER    flies 30 % faster at night
  RAINBOW       RING RIDER     the map's rainbow rings fling it 40 % further
  COURIER       EXPRESS        flies 30 % faster while it carries a parcel
  PARCEL POST   LONG REACH     picks up and delivers parcels from twice as far
  AURORA        TROUBLE SENSE  notices a place's trouble (the ?) from twice as far
  CHEQUERED     RACER          a dash of speed at every race gate
  SNAPSHOT      QUICK EYE      a photo moment (the bell, the fireworks...) lasts twice as long
  FLOCK         FRIENDLY       balloon friends notice you from twice as far

The systems ask here (`speed`, `glide`, `fog_wide`, ...) with the game; nothing is saved (the worn
pattern is, as before).
"""
from . import daylight

PERKS = {   # pattern key -> (perk name, what it does, for the wardrobe page)
    'classic': ('WIDE EYES', 'THE FOG CLEARS FURTHER AROUND YOU'),
    'orchard': ('GREEN THUMB', 'GRABS HIDDEN STAMPS FROM TWICE AS FAR'),
    'sky': ('CALM AIR', 'WIND AND GUSTS NEVER PUSH IT'),
    'sunset': ('TIME FLIES', 'THE WORLD CLOCK RUNS 3 TIMES AS FAST'),
    'mint': ('GLIDER', 'KEEPS GLIDING WHEN YOU LET GO'),
    'crystal': ('SPARKLE SENSE', 'SETS OFF SPARKLES FROM FURTHER AWAY'),
    'lemon': ('FIZZ', 'SHIFT BOOSTS A QUARTER FASTER'),
    'royal': ('ROYAL COMPASS', 'POINTS TO THE NEAREST PLACE NOT FOUND YET'),
    'night': ('NIGHT FLYER', 'FLIES 30% FASTER AT NIGHT'),
    'rainbow': ('RING RIDER', 'RAINBOW RINGS FLING IT 40% FURTHER'),
    'courier': ('EXPRESS', 'FLIES 30% FASTER WITH A PARCEL ON BOARD'),
    'parcel_post': ('LONG REACH', 'PICKS UP AND DELIVERS FROM TWICE AS FAR'),
    'aurora': ('TROUBLE SENSE', 'NOTICES A PLACE\'S TROUBLE FROM TWICE AS FAR'),
    'checkered': ('RACER', 'A DASH OF SPEED AT EVERY RACE GATE'),
    'snapshot': ('QUICK EYE', 'PHOTO MOMENTS LAST TWICE AS LONG'),
    'flock': ('FRIENDLY', 'BALLOON FRIENDS NOTICE YOU FROM TWICE AS FAR'),
}
DASH = 20                       # steps of the RACER's dash after a gate


def worn(g):
    return getattr(g, 'pattern', 'classic')


def on(g, key):
    return worn(g) == key


def speed(g, boost):
    """Speed factor of the worn balloon now (top speed and acceleration)."""
    k, p = 1.0, worn(g)
    if p == 'lemon' and boost:
        k *= 1.25
    elif p == 'night' and daylight.night(g.hour) > 0.5:
        k *= 1.3
    elif p == 'courier' and g.quests.carrying is not None:
        k *= 1.3
    elif p == 'checkered' and 0 <= g.t - getattr(g, 'dash_t', -10 ** 6) < DASH:
        k *= 1.5
    return k


def glide(g, damp):
    """The drag when no key is held (GLIDER keeps most of its speed)."""
    return 0.975 if on(g, 'mint') else damp


def calm(g):
    return on(g, 'sky')


def clock(g):
    return 3.0 if on(g, 'sunset') else 1.0


def fog_wide(g):
    return on(g, 'classic')


def ring(g):
    return 1.4 if on(g, 'rainbow') else 1.0


def spark_reach(g):
    return 1.5 if on(g, 'crystal') else 1.0


def stamp_reach(g):
    return 2.0 if on(g, 'orchard') else 1.0


def parcel_reach(g):
    return 2.0 if on(g, 'parcel_post') else 1.0


def notice(g):
    return 2.0 if on(g, 'aurora') else 1.0


def photo_window(g):
    return 2.0 if on(g, 'snapshot') else 1.0


def friend_reach(g):
    return 2.0 if on(g, 'flock') else 1.0


def gate(g):
    """A race gate was passed: the RACER dashes."""
    if on(g, 'checkered'):
        g.dash_t = g.t


def compass(g):
    """ROYAL COMPASS: the nearest tour stop not found yet, or None."""
    if not on(g, 'royal'):
        return None
    import math
    from engine import timeline as TL
    from .world import icon_centre
    m = g.map
    best, bd = None, 1e9
    for lid in TL.TOUR:
        if lid in g.found:
            continue
        x, y = icon_centre(lid)
        d = math.hypot(m.bx - x, m.by - y)
        if d < bd:
            best, bd = lid, d
    return best
