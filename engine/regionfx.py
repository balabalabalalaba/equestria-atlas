"""Region-segment wrapper (owned by the director): 2-bar map flights.

Region scenes are written in "scene frames": scene frames 0..71 used to be the 1-bar map flight and
the diorama starts at scene frame 72. The film now gives every region a FLIGHT-frame (2-bar) flight,
so at the bottom of each region scene module:

    render, SFX = long_flight(render, ID, SFX)

film frame f < FLIGHT          -> travel_frame(f, ID) (the map builder's 2-bar flight)
film frame f >= FLIGHT         -> scene render(f - FLIGHT_SHIFT)
SFX cue at scene frame f >= 72 -> film frame f + FLIGHT_SHIFT; cues inside the old 1-bar flight
                                  (f < 72) are stretched to 2f.
Inside a scene module, N = seg_frames(ID) - FLIGHT_SHIFT is the length in scene frames.
"""
from .timeline import FLIGHT, FLIGHT_SHIFT, FRAMES_PER_BAR


def scene_to_film(f):
    """Scene frame -> film-local frame of the segment."""
    return f + FLIGHT_SHIFT if f >= FRAMES_PER_BAR else 2 * f


def long_flight(render_fn, seg_id, sfx=()):
    from .loc_worldmap import travel_frame

    def render(f):
        if f < FLIGHT:
            return travel_frame(f, seg_id)
        return render_fn(f - FLIGHT_SHIFT)

    render.__wrapped__ = render_fn
    render.cuts = [(at + FLIGHT_SHIFT, st, (w0 + FLIGHT_SHIFT, w1 + FLIGHT_SHIFT), kw)
                   for at, st, (w0, w1), kw in getattr(render_fn, 'cuts', [])]
    out = []
    for cue in sfx:
        out.append((scene_to_film(cue[0]),) + tuple(cue[1:]))
    return render, out
