"""Cut transitions applied by the renderer at segment boundaries."""
import numpy as np

from .core import W, H, col

_yy, _xx = np.mgrid[0:H, 0:W]
_CELL = 16
_DIAMOND = (np.abs((_xx % _CELL) - _CELL / 2 + 0.5) + np.abs((_yy % _CELL) - _CELL / 2 + 0.5)) / _CELL
_SWEEP = _xx / W * 0.6 + _yy / H * 0.4  # wipe travels from top-left to bottom-right


def diamond(c, t, color='ink', reverse=False):
    """Retro diamond wipe. t=0: no cover, t=1: fully covered.

    The diamonds grow in a sweep from top-left (or bottom-right if reverse).
    """
    if t <= 0:
        return
    if t >= 1:
        c.a[:] = col(color)
        return
    sweep = (1 - _SWEEP) if reverse else _SWEEP
    k = t * 1.6 - sweep * 0.6          # per-pixel local progress
    m = _DIAMOND < k
    c.a[m] = col(color)


def clouds(c, t, hour=12.0, reveal=False, seed=5):
    """Cloud-layer transition between segments (the balloon climbs through a cloud deck).

    The cloud mass always drifts DOWN the screen:
      reveal=False (end of a segment): t 0->1, the deck descends from the top until it covers all.
      reveal=True  (start of the next): t 0->1, the deck keeps descending and uncovers from the top.
    hour tints the clouds for the time of day (engine.shotfx.cloud_palette).
    """
    from . import shotfx
    if t <= 0 and not reveal:
        return
    if t >= 1 and reveal:
        return
    span = H + 90
    xs = np.arange(W)
    if not reveal:
        edge = (-50 + t * span + shotfx._scallop(W, seed, spacing=26, rmin=14, rmax=30))[None, :]
        cover = _yy < edge
        dist = edge - _yy
        oy = -int(t * 70)
    else:
        edge = (-50 + t * span - shotfx._scallop(W, seed + 4, spacing=26, rmin=14, rmax=30))[None, :]
        cover = _yy > edge
        dist = _yy - edge
        oy = -int(70 + t * 70)
    shotfx._paint_clouds(c, cover, dist, hour, ox=seed * 37, oy=oy)
