# Making a film region explorable (brief for region builders)

The prototype game (`game/`, see `game/README.md`) lets the player fly the explorer's balloon over the
world map and land at a place to explore its diorama. Each explorable place is one file
`game/regions/<seg>.py` that turns the film's scene `scenes/sNN_<seg>.py` into a pannable panorama.
The film files are **read-only**. Never edit `engine/`, `scenes/`, `audio/` or the shared `game/*.py`.
If the base class is missing something, work around it in your file and report what you'd change.

Reference implementations (read both first, they are short):
- `game/regions/ponyville.py`: **rebind style**. The scene has a clean `render_diorama(f)` that reads
  the camera from `cam_x(f)`. A copy of it is bound to replacement globals with `rebind(...)`: the
  camera, a no-op balloon, and game-driven train / raincloud / clock.
- `game/regions/sweetapple.py`: **recompose style**. The film's `render(f)` branches per shot (`if f < 216:`), so the region
  calls the scene's own cached layer builders and animation helpers itself, in the film's layer order,
  along one continuous strip, with every helper active wherever it is on screen.
  Use this style whenever the film's render is shot-gated or a simple rebind can't show everything.

Base class: `game/region.py` (read its docstrings). You provide:

```python
from ..region import Region, Area, Landmark, rebind, scene_render
import scenes.sNN_<seg> as sc

class MyPlace(Region):
    SEG = '<seg>'                                  # segment id (engine/timeline.py)
    CARD = ('KIND', ['FACT LINE', 'FACT LINE'])    # copy the scene's atlas_card(...) call
    def build_areas(self):
        return [Area(self.render_x, cam_x=(0, CAM_MAX), cam_y=(0, 0), y_range=(36, 128),
                     start=(x, y), landmarks=[Landmark(...), ...], balloon='l')]
    def init_state(self): ...      # optional persistent state (e.g. 'the train has arrived')
    def on_enter(self): ...        # optional: reset when landing from the map
    def on_area(self, ai): ...     # optional: arrived in area ai (landing: 0, or flown in from a neighbour)
    def update(self, t): ...       # optional per-step events: sounds, triggers (t = game step)
    def render_x(self, f, cx, cy): # -> Canvas 320x180: the world at INTEGER camera (cx, cy)
        ...
```

- `render(f, cx, cy)`: `f` is a free-running scene frame (`360 + t` by default; override
  `scene_f(t)`). Use it for periodic ambient animation such as smoke, flags, sparkles, water and
  clouds. Use `self.g.t` (the game step, 30 per second) for your own clocks. The render must be
  **pure** (it may be called twice per step at 60 fps): no state changes inside render.
- **Never draw the film's balloon.** The base class draws the player's balloon on top. Rebind
  `draw_balloon` to a no-op, or skip the scene's balloon helper. Never draw `callout`, `atlas_card` or
  `iris` from the scene. The base class draws tags and the card; rebind those names to no-ops if you
  use `scene_render(sc)` (the scene's raw `render`, without the film's wrappers).
- **Camera range**: `cam_x` / `cam_y` are the integer camera limits. Find the largest camera at which
  every parallax layer still has art. For each layer (width `Wl`, parallax `p`, offset `o`):
  `p * cam + o + 320 <= Wl`. Check it visually with the `edges.png` sheet (a layer that runs out shows
  a gap or flat strip at the right / bottom edge). The film's own camera path is a good starting
  point, but the whole strip is usually wider.
- **Shots**: a film shot change is usually a jump to another part of the same strip (keep one
  continuous Area). If it really is a different view (another angle, interior, close-up,
  underwater), make it a second `Area`. Flying past the right edge of an area leads into the next
  one, with a cloud sweep. Mirror any shot-dependent offsets (e.g. a mountain layer offset that
  differs per shot) into one formula that is continuous over the whole strip.
- **Vertical pans**: set `cam_y` and give `y_range` in world px (basket bottom), so the balloon can
  reach the whole height. Landmark `y` (optional) restricts the tag to that height band (±90 px).
- **Timed film beats** (a train arriving, lightning, fireworks, a fountain burst, a ship sailing in)
  belong on game clocks. Either replay them in a loop (`216 + t % 240`), trigger them when the
  balloon reaches a spot (see the Ponyville train), or freeze them at their best frame. Avoid visible
  pops: restart a non-periodic effect only while it is off screen. Ambient helpers gated by `f`
  ranges (`if f < 216`) should run wherever they are on screen.
- **Landmarks**: one `Landmark(name, centre_x, (tag_x, tag_y, side, leader), y=None, p=1.0, py=None)` per
  film callout, in **world** px (`p`: parallax of the layer the tag points at, if not the main layer;
  `py`: its vertical parallax if different). The scene's callouts are written in screen px as `X - cam`, so drop the
  `- cam`. Use the official names exactly as the film writes them. The tag shows when the balloon is
  within 62 px (horizontally) of `centre_x`. Order them left to right.
- **start**: where the balloon appears when landing (world px, basket bottom). Use the film's opening
  framing (where its iris opens) and keep the balloon in the sky, not inside a building.
- **front(f, cx, cy)** (optional): art that stands in front of the balloon, on a KEY canvas at the
  integer camera; the base class keeps only the part around the balloon, so the scene's own layer
  order elsewhere is untouched. Rainbow Falls uses it for the near half of its cloud hoops, so the
  balloon flies through them instead of over them.
- **sky(f, cx, cy)** (required where the area shows open sky): the area's sky alone, exactly as the
  render draws it first (gradient, stars, moon, sun, aurora, same offsets); scenery that could be
  mistaken for sky stamped out in magenta. The world clock repaints those pixels for its hour
  (game/daylight.py). Check with `tools/phaseb_skycheck.py <place id>`.
- **balloon_tint**: `(colour, amount)` lights the balloon like its surroundings (multiply blend): use it
  in dusk, night, misty or dark scenes so the balloon isn't brighter than the world (its flame stays
  bright). Example: `balloon_tint=('#5a5a9a', 0.5)`.
- **balloon**: `'l'` (28 px) by default. Use `'m'` or `'s'` where the scene's scale is vast or
  distant and an `'l'` balloon would look like a giant.
- **Sound** (`update`): `self.g.sfx(name, gain, variant=None)`. Names come from `docs/AUDIO.md` /
  `audio.sfx.SFX_NAMES`. Scale gain with `self.near_gain(world_x)`. Keep it sparse (the film's `SFX`
  list shows what fits). The music is automatic: the scene's own bars of the film score loop.
- **Speed**: render under ~7 ms per frame (the compositor adds 3–5 ms; the 60 fps budget is
  16 ms). Cache static layers (the scenes already use `lru_cache`), use numpy, and don't loop in
  Python over pixels.
- Project rules still apply (CLAUDE.md): **no creatures** of any kind, no character statues, official
  place names only, nothing drawn on the HUD from the region except what the base class draws.

## Interaction (required): hotspots and hidden stamps

The player should be able to *do* things in every place, not only look. Each region has:

- **3 to 5 `Hotspot`s** (`Area(hotspots=[...])`): the place's signature events, set off on demand
  with SPACE. When the balloon's envelope comes within `r` px, a bubble with the hotspot's icon and
  `SPACE` pops up over it. From afar, a faint glint marks it. Pick the events the film already
  animates and make them player-triggered: ring a bell, send a train off or call it in, launch
  fireworks, make lightning strike, light lanterns, start a fountain or geyser, set an airship or ship
  sailing, swing a crane, flash a lighthouse, start an eruption, open the aurora. Ponyville
  (`game/regions/ponyville.py`) is the reference: star blaze, raincloud, bell, cupcake sprinkles, and
  the train departing or arriving.
  - `Hotspot(key, x, y, icon, r=44, cooldown=75, p=1.0)` in world px, at the thing itself, where the
    balloon can reach it. Icons are listed in `game.ui.ICONS` (bell, whistle, rain, spark, firework,
    steam, lantern, note, bolt, wind, splash, gem, flag, gear, horn, apple, heart, moon), or pass
    your own 9x9 `Sprite`.
  - In `on_interact(key, t)`, start the event: set your clock (e.g. `self.boom_t0 = t`) and play its
    sound. The base class records `self.spot_t[key]`, the step it was last set off, so your render can
    draw the result from `t - self.spot_t[key]`. Every interaction needs a **clearly visible** result
    (not just a sound) lasting 1 to 3 s. `game.ui.burst(...)` gives chunky particle bursts, and
    `engine.fx` has glow, sparkles, confetti, steam and fireworks-like helpers. Reuse the film's own
    effect helpers wherever you can.
  - Keep the events beautiful when retriggered, and give each a sensible `cooldown`.
- **3 hidden `Stamp`s** (`Area(stamps=[...])`, spread over the areas): atlas stamps the player
  collects by flying the balloon's envelope through them. Hide them where a curious player looks:
  up high near a spire, low under a bridge or among market stalls, at the far end of an area, behind
  a foreground layer's gap. They must be reachable: the envelope centre is `by - 18` for an `'l'`
  balloon, `- 13` for `'m'`, `- 8` for `'s'`.
- Hotspot bubbles must not sit on top of landmark tags. Move one of them if they collide.
- The shared hint strip (bottom-right, first visit only) shows SPACE and ESC. ESC takes off; SPACE
  only interacts.

## Procedure

1. Read `CLAUDE.md` (project rules), this file, `game/region.py`, both reference regions, and the
   scene's docstring + `render` (grep for `def `, `_W`, `_P`, `cam`, `callout`, `atlas_card`,
   `balloon`, `SFX`). Read big scene files selectively.
2. Decide the areas, camera ranges, landmarks, start, clocks for timed beats, sounds, hotspots and
   stamps. Write
   `game/regions/<seg>.py` early and save often.
3. Check: `.venv/bin/python tools/game_region_check.py <seg>`. It writes
   `out/game_checks/<seg>/edges.png`, `tour.png` and `back.png`, and prints landmarks found, ms per
   frame, sounds, purity and problems. Single frames:
   `... game_region_check.py <seg> --cam <area> <x>,<y> [--t step]`.
   **Look at the images with the Read tool** and fix what looks wrong. Watch for gaps at layer edges,
   things popping in or out, a film element frozen mid-action, tags in odd places, a stray film
   balloon, darkness or fades from the film's end frames, and black or empty strips. Do several
   look → fix rounds until it truly looks good and the check prints `OK`.
   Then test every hotspot and stamp: `... game_region_check.py <seg> --interact` writes
   `interact.png` (each hotspot's bubble, then 10 and 28 steps after SPACE, and each stamp
   collected) and prints the sounds. Look at it and make every result clearly visible.
4. Report: areas and camera ranges, landmarks, what you replay on game clocks, timings, the final
   check output, and anything you could not solve or would change in the shared files.

Throwaway scripts go in `tools/<seg>_*.py` or your scratchpad, never in `game/` or `engine/`.
