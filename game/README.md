# EQUESTRIA ATLAS: explorable prototype

A playable version of the atlas. You pilot the explorer's balloon over the world map, discover the
20 tour stops, and land to explore each place's living diorama: fly through every area, find the
landmarks, set things off with SPACE and collect the hidden atlas stamps. Phase B adds reasons to
fly all over the land: every place has a trouble to find and mend (helping changes the place and
the whole map), 6 timed ring
races, a 20-photo album, 15 balloon friends to meet, 18 badges and a fog of war over the uncharted map. All 20 film regions are
playable (53 areas, 95 landmarks, 98 interactions, 60 stamps). It reuses the film's engine and art
read-only (two film fixes made for the game are listed at the end).

## Run

```bash
.venv/bin/python game/play.py
```

It opens on the title screen. Options: `--mute` (no sound), `--scale N` (force a window scale),
`--new` (start over; the old save is kept as `.old`), `--skip-title` (straight to the map),
`--frames N` (quit after N steps of this session: a smoke test). Needs `pygame` in the venv
(`.venv/bin/pip install pygame`). Progress is saved to `out/game_save.json` automatically (every 10 s,
on every take-off, and on quit); settings to `out/game_settings.json`. A save that is damaged or cut
short is kept next to it as `game_save.json.bad` and a new journey starts (CONTINUE is then not offered).

Testing cheats (not part of the game, only with this flag): `--cheats` turns on hidden keys:
`[` / `]` move the world clock back / forward an hour, `\` cycles the clock speed (stop, x1, x4, x16,
x60), `,` / `.` change the balloon speed (x1, x1.5, x2, x3). A small mint line bottom-left shows the
current values; each change is printed in the terminal.

```bash
.venv/bin/python game/play.py --cheats
```

Headless smoke run:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python game/play.py --frames 200
```

## Sharing the game (packaged apps)

`packaging/` turns the game into double-click apps for friends (PyInstaller; the game's own sources
ship as plain files inside, so nothing in the game changes). A packaged app keeps saves, settings,
screenshots and album photos in the player's folder (`~/Library/Application Support/Equestria Atlas`
on a Mac, `%APPDATA%\Equestria Atlas` on Windows) instead of `out/`.

- **Mac**: `bash packaging/build_mac.sh` -> `dist/Equestria-Atlas-mac.zip` (~100 MB: the app + `READ ME FIRST.txt`).
  Built on Apple Silicon, it runs on M1 or newer Macs. It is unsigned: friends right-click > Open the first time.
- **Windows**: a Windows app must be built on a Windows PC. Run `bash packaging/make_source_zip.sh` here
  (-> `dist/Equestria-Atlas-source.zip`, ~60 MB), copy it to a Windows PC with Python 3.11+ installed,
  unzip, and double-click `packaging\build_windows.bat` -> `dist\Equestria-Atlas-windows.zip`.
- `READ ME FIRST.txt` (in both zips) explains how to open the app, the controls and how to send feedback.
- `--selftest` (any build) loads every place, draws a frame in each at three hours and builds every sound,
  and writes `selftest.txt` next to the saves: the check for a fresh build or a friend whose game won't start.
- `packaging/make_icon.py` makes the balloon app icon (`icon.icns`, `icon.ico`).

## Title screen and pause menu

- **Title**: the parchment overview map unrolls inside its gold frame while the explorer's balloon
  rises in front of it and the title card pops in (any key skips this). The menu plates slide into
  the right panel: CONTINUE (only with a save), NEW GAME, SETTINGS, QUIT. NEW GAME over a save asks
  "SURE?" first (SPACE yes, ESC no), and the old save is kept as `out/game_save.json.old`. The left
  panel shows the saved journey as captioned counters that slide in from under the panel's rim one
  after another (PLACES, COMPLETE, STAMPS, HELPED, BADGES); for a new player it shows the journey
  ahead greyed out at 0 (PLACES, HELPED, FRIENDS, RACES, POSTCARDS). A pulsing ring, drawn in front
  of the title card, marks where the balloon is on the map; when the card hides that spot, a pin chip
  hangs from the card's bottom edge instead. The title loops the film's opening bars.
- **Pause** (ESC on the world map, or Start on a gamepad anywhere): the whole world freezes and dims
  behind a plum panel, and the music dips: RESUME, MUSIC and SOUND volume (LEFT / RIGHT, 0 to 10), WINDOW scale (FIT or x2 up to
  what fits the screen), FULLSCREEN, CONTROLS (a page of every keyboard key and gamepad button) and
  QUIT (progress is saved). The same settings are on the title screen.
- Settings are applied at once (the window is remade when the scale changes) and kept in
  `out/game_settings.json`. F still toggles fullscreen and is remembered too.

## Controls

| key | world map | in a place |
|---|---|---|
| arrows / WASD | fly | fly (left/right pan the place; up/down change height) |
| SHIFT | boost (burner flame) | boost |
| SPACE / ENTER | land at the place under the balloon (not during the take-off climb) | set off the thing the balloon is next to; pick up goods / mend a trouble |
| ESC | pause menu (or close the overview) | take off back to the map |
| M | overview map of the whole land | |
| J | atlas journal (← → turn its pages) | atlas journal |
| P | postcard snapshot (ticks album targets) | postcard snapshot (ticks album targets) |
| F / F11 | fullscreen (remembered) | fullscreen |
| F12 | screenshot to `out/game_shots/` | screenshot |

To quit, choose QUIT in the pause menu (ESC on the map; in a place, ESC takes off first) or close
the window. Progress is saved either way.

A gamepad works too. The stick or d-pad flies, and the shoulder buttons boost. Button 0 (A) lands
or sets off a hotspot, 1 (B) takes off, 2 (X) opens the journal, 3 (Y) opens the map, 6 (Back)
takes a postcard and 7 (Start) pauses. The menus' CONTROLS page shows all of this.

## What is in it

- **World map** (`engine.loc_worldmap.draw_map`): the balloon flies with eased inertia, and the
  camera leads it (less vertically, so the balloon stays clear of the HUD; at the map's north edge it
  stays on screen). Flying over a tour stop discovers it with a star chime: the icon pops from grey to
  colour, its flag turns into a gold star, a sparkle ring and name plate appear, and if you let go of
  the keys the balloon settles near it. M opens the parchment overview with your balloon (ring races
  and friend races wait while it is open). The world clock lights the map (see Day and night).
  - **Counter** (top right): a gold star counts the places you have landed in (n/20), a small outline
    star the places found but not visited yet. A first landing shows a FIRST VISIT toast on arrival,
    and the counter flashes as you take off again.
  - **Name plates** show each place's status: grey = undiscovered, gold rim = found but not visited,
    plain = visited, bright = under the balloon.
  - **Pins and arrows**: a pink pin marks found places you haven't visited yet, and one pink target
    (the nearest place still to visit) is always marked, by its pin or, off screen, by an edge arrow.
    It keeps its place until you visit or reach it, or for 20 s, and hides while an errand's gold
    arrow is showing. Edge arrows are chunky darts in 8 directions that stay clear of the clock, the
    counters and the land prompt.
  - **Land prompt**: it slides up from the bottom as soon as a place is found (for 5 s after you find
    a place you can land on it from a bit further away, 45 px instead of 30). The first row shows
    `SPACE`, the land icon, the name and a gold NEW tag on places you have not visited; the second
    shows landmarks found n/total, stamps n/3 and the place's trouble (`?` not noticed yet, the goods
    it needs, a heart once helped). While it is up, SPACE always lands (a balloon friend nearby gives
    way).
- **The places** (`game/regions/`, one per film region): each has one or more areas (the film's
  shots) joined by cloud sweeps.
  - **Arrival:** a first visit starts at the left of area 0. On a return trip with an errand there
    (goods to pick up or a trouble to mend), the balloon lands straight in that area with a STRAIGHT
    TO THE <ITEM> (or STRAIGHT TO WHERE IT IS NEEDED) toast; if queued tips keep that toast waiting
    more than ~5 s, or the balloon takes off, it is dropped, so it never shows at the wrong place.
  - **Landmarks:** the film's tags appear as you drift past, drawn on top of props and weather. A tag
    that would sit under the counters or the clock shows on its other side (fixed per area, so it
    never jumps). A star counter tracks them, and a plate appears when all are found.
  - **Hotspots:** 3 to 5 per place, set off with SPACE. A bubble shows the icon when you're close
    (drawn above the balloon; it steps aside while the effect plays). Every SPACE gives a light pulse,
    a ring and a burst of sparks. The first use of each hotspot floats a '+1' spark with a blip (beside
    it when the hotspot is near the top of the screen), and setting off a place's last one plays a
    jackpot. Untried hotspots twinkle gold from afar; tried ones keep a faint white glint. A third HUD
    counter, sparks set off / total, sits under the landmarks and stamps (the errand plate moves below
    it). Examples: Ponyville's bell, raincloud and train; Sweet Apple Acres' apple tree and zap-apple
    lightning; the Changeling hive's transformation; Griffonstone's lighthouse.
  - **Set-pieces stay done** on later landings and across saves (`lasting` in the save): the Crystal
    Empire stays lit, the changeling hive stays in bloom, the Las Pegasus strip stays neon, and SPACE
    then replays the pulse, ring or surge.
  - **Stamps:** 3 hidden atlas stamps per place, collected by flying through them. A stamp that
    unlocks a balloon pattern shows a NEW BALLOON - <NAME> PATTERN (J) toast.
  - **Timed events:** the film's beats replay on game clocks (trains, storms, ships).
- **Map fun**: 42 rainbow rings float along the routes between places. Flying through one bursts
  it with a chime and flings the balloon onward, and it comes back later. A few other crewless
  balloons drift around the land in their own colours. Discoveries and rings make your balloon
  hop.
- **Celebrations**: completing a place (all its landmarks) sets off fireworks over it, and grabbing
  a stamp makes the balloon hop.
- **Balloon wardrobe**: every 6 stamps unlock a new envelope pattern (Explorer, Orchard, Clear Sky,
  Sunset, Mint Cream, Crystal, Lemonade, Royal, Starry Night, Rainbow), and six more come from
  Phase B (Courier, Parcel Post, Aurora by helping 5, 10 and 20 places, Chequered by a gold medal,
  Snapshot by 10 album photos, Flock by meeting all 15 balloon friends): 16 in all. Choose one on the journal's BALLOONS page; it shows on the map and in every place.
- **Balloon perks** (`game/perks.py`): every balloon does something of its own while it is worn (the
  BALLOONS page shows it under the name): EXPLORER clears the fog 40 % further; ORCHARD grabs stamps
  from twice as far; CLEAR SKY is never pushed by wind or gusts; SUNSET runs the world clock 3 x as fast
  (dawn, sunset and night photos sooner); MINT CREAM glides on when you let go; CRYSTAL sets off sparkles
  from half as far again; LEMONADE boosts a quarter faster; ROYAL shows a gold compass arrow to the
  nearest place not found yet; STARRY NIGHT flies 30 % faster at night; RAINBOW is flung 40 % further by
  the map's rings; COURIER flies 30 % faster with a parcel; PARCEL POST picks up and delivers from twice
  as far; AURORA notices a place's trouble from twice as far; CHEQUERED dashes (x1.5 speed, 20 steps)
  at every race gate; SNAPSHOT keeps photo moments twice as long; FLOCK is noticed by balloon friends
  from twice as far. Nothing new is saved. `tools/claude_perks_check.py` measures each one.
- **Postcards (P)**: a camera flash and shutter save the current view framed as a postcard
  ("Greetings from ...", with an atlas stamp) to the player's Pictures folder,
  `~/Pictures/Equestria Atlas/postcard_<place>_<ms>_<n>.png` (the game's own `game_shots` folder if there
  is no Pictures folder or it can't be written; names never collide, so no postcard overwrites another).
  A POSTCARD SAVED toast names the place (THE WORLD MAP on the map) and says PRESS J TO SEE IT. A postcard
  that ticks an album target shows the album's toast instead. F12 screenshots still go to `game_shots`.
- **Photo book** (`game/photos.py`, the journal's PHOTOS page): every postcard taken, newest first, read
  back from the postcard folders (Pictures/Equestria Atlas and the older `game_shots`, so postcards from
  before this page are there too). Left page: a filmstrip (a small print, the place, the date; NEW on
  this session's; a scrollbar past 6), right page: a big print of the chosen one, when it was taken and
  where the files are. SPACE shows it full screen, exactly as it was on screen (the view inside the
  postcard's frame, every 4th pixel), with a postcard-style caption ("Greetings from ...", the date,
  the stamp) and "3 / 12"; LEFT / RIGHT turn the photos with a short cross-fade, the captions slide away
  after 3 s without a key (any key brings them back), SPACE shows the file selected in Finder / Explorer,
  ESC fades back to the page. J pressed within 15 s of a postcard opens the journal on this page. Prints
  are read lazily, one per step (~8 ms, a soft "developing" card until then); a postcard deleted or
  moved in the folder drops out the next time the journal opens, an unreadable file shows a blank print.
  The ALBUM page still says where postcards are, and SPACE there opens the folder.
- **Atlas journal (J)**: a parchment book of all 20 places in route order. Each shows its status
  and landmark count, and the selected place's map icon, atlas card, landmarks found, stamps and the
  events you have set off. Places stay "? ? ?" until discovered and are described once visited.
- **Finale**: once every place has been helped (all 20 troubles mended), the film's outro plays on
  the next take-off. (Finding every landmark earns SIGHTSEER but no longer triggers it.) The journey
  redraws on the map, "MAP COMPLETE 20/20" stamps down, then the title and THE END. The journal's
  ATLAS footer counts toward it (GRAND FINALE: n/20 PLACES HELPED); once every place is helped or the
  finale has been seen, SPACE there plays it (SPACE GRAND FINALE, then GRAND FINALE AGAIN). A replay
  started inside a place takes off from that place when it ends.
- **Music**: the game has its own soundtrack (see Soundtrack below): a title theme, the map theme in a
  day and a night arrangement that crossfade with the world clock, a race theme for ring races and
  balloon friends' races, and a theme for every place. The map music pauses (keeping its place) while a
  place or a race plays, and comes back where it was. The finale plays the film's outro. Reward sounds
  (stamps, mends, stars, coins, chimes...) briefly lower the music by about 6 dB (not when the sound
  effects are muted), and the music dips by about 4 dB while the journal or the pause menu is open.
  The sound effects are the film's procedural library. Everything plays at 48 kHz (the mixer is opened
  at that rate before pygame starts).
- **Window**: clicking away to another window opens the pause menu (the world waits); a minimised
  window stops drawing and holds the sound. Fullscreen draws the view at the largest whole scale that
  fits the screen, centred on black (every art pixel the same size: no shimmer as the camera glides).
- **Robustness**: a place that fails to load, fails on arrival (its enter hooks raise) or crashes
  sends you back to (or keeps you on) the map, padlocked for that session, and the error is printed.
  If the game itself crashes, it saves the journey and writes `crash.txt` (the traceback) next to the
  save. A save or settings file that can't be written (a full disk) no longer stops the game, and a
  damaged settings file falls back to the defaults.
  A place loads behind the closed landing iris, so its first frames never stall. Landing somewhere
  else in the middle of a balloon friend's race ends that race.

## Welcome card and tips

`game/tutor.py`. A new game opens with a welcome card over the dimmed map that says what the game is
about, wandering first: ENJOY THE WANDERING (fly where you like, every place is alive to explore),
HELP THE PLACES (find a ? and bring what it needs) and MAKE FRIENDS ON THE WAY (greet the balloons in
the sky), plus the keys. SPACE starts, ESC turns the tips off.

The first request comes as the card closes on a brand-new journey: a toast says PONYVILLE NEEDS A
HAND (its market cart is empty, already noticed), and the goods bubble, the gold arrow and the errand
plate lead to Sweet Apple Acres next door for apples: the short apple loop is everyone's first errand.

After that, small tips appear only when they fit, one at a time: on the map above the key hints (and
above the land prompt while it is up), in a place in whichever band (under the counters or above the
street) does not cover the balloon. They cover: fly, visit a place, wander and set things off, find
the ?, fetch what it needs, bring it home, take off, greet a balloon, the journal / map / postcard keys.
A tip shows for about 6 s and then slides away. It comes back at most once, a few minutes later, if
you still haven't done it, and then counts as learnt. Doing the thing ticks it off with a green check,
and a player who already does something never sees its tip.

One message speaks at a time, and a toast always goes first: while a toast is up or waiting its turn,
a tip waits, and one already on screen folds away with its timer paused. Tips also wait while a
discovery plate, a place's atlas card or completion plate, a race or the iris is on screen. In the
first 2 minutes of a session toasts are paced too (one at a time, with a short breath between them),
and a toast's sound is skipped if the same sound has just played.

TIPS ON / OFF is in the pause and title settings (`out/game_settings.json`). Saves made before the tutor
skip the welcome card when they have visited places. The test tools start as seasoned players (no
card); `tools/phaseb_tutor.py` plays a fresh game through the card and the tips (`out/phaseb/tutor.png`),
`tools/d_gift.py` checks the first request, and `tools/d_tips.py` draws every tip plate on one sheet.

## Day and night

One world clock runs for the whole world: a day lasts 24 real minutes (an hour a minute), and a new
journey starts at 08:00. The journal, the pause menu and the welcome card freeze the whole world, not
just the clock: the map, the places, races, toasts and tips all wait and go on exactly where they
stopped (the overlays run on a clock of their own). The hour is kept in the save (`hour`), and so is
the journey's step clock (`steps`), so places you helped stay mended after a relaunch. A small clock
plate top-left shows the time with a sun, a low sun at dawn and dusk, or a moon (it steps aside while
a race plate is shown).

- The world map is lit by the clock with the film's own day grade: dawn, bright day, golden dusk,
  blue night with glowing towns, the aurora in the north at night. Rain clouds darken at night too.
- The places were painted at fixed hours (the film's times). `game/daylight.py` makes them follow the
  clock in two parts:
  - **The sky is repainted.** Every area that shows open sky gives `Area.sky(f, cx, cy)`: its sky
    alone, drawn as its render draws it first (gradient, stars, moon, sun, aurora), with the scenery
    that could be mistaken for sky stamped out. The frame's pixels that still match it (within a
    little haze; plus small enclosed holes close to it, such as a hazed sun disc, but never a small
    cloud or a thin pole) are the open sky. When the clock's hour looks different from the painted
    one, those pixels are repainted with the clock's sky: the film's sky presets blended by the hour,
    the sun on its arc by day, the moon and twinkling stars by night (both set behind the scenery).
    Clouds stay and are re-lit. Interiors, caves and the underwater views have no hook.
  - **The landscape is re-lit.** A day scene at night undoes its painted grade and applies the night's
    (per-channel lookup tables), keeping its warm lights (lit windows, lamps, lanterns, fire: small hot
    yellow-orange spots) glowing. A night scene by day gets an even, hue-neutral lift; a strong lift
    has a soft knee, so lit neon and pale towers keep their detail instead of clipping to white
    (day-painted places are unchanged). Las Pegasus by day also drops the light it paints into its own
    sky (`daylight.sky_light`: no searchlights), and its fireworks become coloured smoke puffs painted
    to keep their colour (`daylight.unlight`). The balloon, cargo and trouble props are lit to match
    (`Region.light_tint`).
  Light painted into the art itself (long morning shadows, a sunset glow on one side of a building)
  stays as painted. Cost: ~1-2 ms per frame where the sky is repainted (the sky mask is computed once
  per step). Checks: `tools/phaseb_daylight.py` (every place painted / at 06:00 / 12:30 / 18:45 / 23:00,
  `out/phaseb/daylight_*.png`) and `tools/phaseb_skycheck.py <place>` (the detected sky in magenta
  per area and camera, then the four hours).
- The photo album's "map by night" and the festival lights over helped places follow the clock.

## Phase B: requests, races, album, badges, fog

- **Helping the places** (`game/quests.py`, props in `game/mend_art.py`, goods in `game/items.py`): every
  one of the 20 places has a trouble, found by exploring and fixed in any order. Nobody is ever shown:
  the places themselves are helped (no creatures).
  - The troubles: wilted royal flower beds (Canterlot), no flags for the games (Cloudsdale), an empty
    market cart (Ponyville), a barn dance with no lights (Sweet Apple Acres), a dark path lamp
    (Everfree), a bare trail board (Neighagra Falls), a dry clifftop planter (Rainbow Falls), a park cart
    with nothing to sell (Manehattan), a beach party in the dark (Baltimare), a bare castle stall
    (Griffonstone), a dry grove fountain (Dragon Lands), an empty festival rack (Mount Aris), a broken
    bazaar lamp (Klugetown), an empty hive market (Changeling Kingdom), a town board with no news
    (Appleloosa), a closed plaza stall (Las Pegasus), an empty harbour rack (Vanhoover), a cold station
    with no tea (Crystal Empire), a dark snow festival (Yakyakistan), the Tree of Harmony in the dark
    (Castle of the Two Sisters).
  - Each is drawn in its place (8 prop kinds: flower bed, lantern string, lamp post, fountain, market
    cart, festival poles, fireworks rack, notice board) with a bobbing `?` over it. On the world map a
    white `?` bubble bobs gently over every found place whose trouble you have not found yet (the
    overview map shows a small grey `?` beside its pin where the fog is clear). Fly near the trouble
    and you notice it with a soft 'idea' chime: a toast says what it needs and where that comes from,
    it joins the journal's HELP page, and the goods bubble takes over from the `?`.
  - The goods come from the place that has plenty (seeds from Klugetown's bazaar, crystal light from
    the Crystal Heart, lanterns from the Kirin Grove, apples from Sweet Apple Acres...). Once a trouble
    needing them is noticed, they wait at their source in a glow: fly the envelope to them and press
    SPACE. They hang under the basket: one kind at a time, but enough of it for every noticed trouble
    that needs it (the plate and the pick-up toast show X2 and so on, each mend uses one, and the save
    keeps the count). Goods from a carrier friend fill the basket the same way.
  - Bring them to the trouble and press SPACE: magic sparkles spiral in, a flash, and the place is
    mended for good (flowers bloom, lanterns glow, the lamp shines with crystal light, rainbow water
    jets, an awning and goods on the cart, rainbow pennants, rockets that go up now and then, a trail
    map on the board), with a fanfare, a stamp a beat later, fireworks and a toast. Without the goods,
    a grey bubble shows what is needed.
  - **The festival**: after a mend a gold light band sweeps across the screen and reveals the place's
    festival, which stays for good in every area that fits it: pennant bunting between the landmarks,
    a festival mast with ribbons and strings over the mended thing, flags on the rooftops, party
    balloons rising (glowing sky lanterns at night), petals and confetti, festival bulbs at night.
    Indoors and in caves no balloons or lanterns rise, and under the sea there are no balloons or
    confetti.
  - **The map changes too**: a helped place gets a string of pennants and a floating heart over its
    icon (and at night a festival glow with little fireworks); a thin rainbow arcs from it back to the
    place whose goods helped it (under the bubbles, behind the name plates); a route leg between two
    helped neighbours turns into a rainbow; and more balloons take to the sky as more places are helped
    (4 at first, up to 15). The THE MAP CHANGES TOO toast appears only after your first help.
  - Guidance: a plate under the counters names the nearest job (`FROM <place>` where to pick goods up,
    `TO <place>` where to bring them, with a small `+N` when more places are waiting). Inside a place it
    shrinks to the goods icon and a little arrow (towards the spot, or up: take off) whenever a landmark
    name tag would sit under it, and toasts never cover it. Bubbles over those places on the map (kept
    off the place names) and gold edge arrows point the way; the overview map (M) marks them and shows
    a heart on each helped place.
  - Helping 5, 10 and 20 places unlocks the COURIER, PARCEL POST and AURORA balloons.
- **Balloon friends** (`game/flock.py`): the 15 crewless balloons drifting over the map (4 at first,
  more as places are helped: a toast says A NEW BALLOON IS FLYING, NEAR <place>, and an edge arrow
  points to it for a while) can be met. A friend you have not met carries a small pink heart. Fly
  alongside one (within 22 px) and a bubble pops up beside it, on the side away from your balloon; the
  heart and the bubble ride its balloon smoothly. SPACE greets it: both burners puff and hearts and
  confetti burst (a bigger burst the first time). The first greeting adds it to the journal's FRIENDS
  page, and its toast says what SPACE does next (FOLLOW ME, RACE ME or TAKE MY <item>). Landing wins:
  while a place's land prompt is up, a nearby friend gives way and hides its bubble, and friends' home
  loops keep at least 30 px from the place icons. After the first greeting, SPACE asks a friend to do
  its job:
  - **carriers** (BOWLER, PRISM, LEMON DROP, MINTY, NIGHTCAP, GALE) have goods hanging under them
    (cider, lanterns, crystal light, seeds, a rainbow jar, fireworks): if a trouble you noticed needs
    them and your basket is empty, they hand them over, a shortcut instead of the source place. They
    restock after a while.
  - **racers** (SUNDOWN, NEON, SEASPRAY, NORTHWIND, SCEPTRE, each faster than the last) challenge you to
    a race to a place 220 to 460 px away: they set off at once; reach its icon first (a plate top-left
    and a sky-blue edge arrow show the goal). Beating one gives a ribbon (shown on the FRIENDS page).
  - **guides** (CIRRUS, ORCHARD, SPECTRUM, SANDY) lead the way, trailing sparkles and a little slower
    than you, first to a found place whose trouble you have not seen yet, else the nearest
    undiscovered place, else a visited place with stamps or sights left, else a race start never
    flown, else a map ring never flown ("HERE IT IS!"), then drift home.
  Meeting all 15 gives the FRIENDLY SKIES badge and the FLOCK balloon pattern; beating every racer the
  SKY RACER badge. No crew is ever shown: the balloons themselves are the friends.
- **Weather** (`game/weather.py`): on the world map, snow falls over the Frozen North and the icy
  south, four dark rain clouds drift in slow loops with their shadows on the land (over the Everfree
  country, the Manehattan-Baltimare coast, the west coast and the eastern forests), and windy zones
  blow (swooshes over the Celestial Sea and the Luna Ocean, blowing sand over the Bone Dry Desert, a
  breeze over the Cloudsdale highlands). In places: sand gusts (Appleloosa, Klugetown), breezes
  (Cloudsdale, Mount Aris, Baltimare), drizzle (Griffonstone), and the snow / rain the scenes already
  draw (Yakyakistan, the Crystal Empire, Vanhoover). The balloon takes it:
  - wind carries it (fly with it for speed, against it to struggle); in places, gusts come and go and
    push it sideways. The camera's look-ahead includes the push, so it stays smooth. On the map the
    wind calms near places you have found (none within 30 px of the icon, full again at 70 px), so you
    can hover over a place and land even in windy regions.
  - rain soaks it: the envelope darkens and drips; flying out of the rain while soaked leaves a little
    rainbow behind.
  - snow settles on top of the envelope; boosting (SHIFT) shakes it off in a puff of flakes.
  The first meeting with each kind shows a toast that says how it works (it waits for a free toast
  lane and is dropped rather than shown late). Balloon friends inside
  uncharted fog are hidden (and can't be greeted) unless one is leading or racing you.
- **Ring races** (`game/races.py`): four courses on the world map (CANTERLOT CIRCUIT by Saddle Lake,
  COASTAL DASH off Manehattan, BADLANDS RALLY round the crater, FROSTBITE RUN from the Crystal Empire
  to Yakyakistan), RAINBOW HOOPS through the film's cloud hoops at Rainbow Falls and THE STRIP SLALOM
  along Las Pegasus. Fly through the chequered start gate (on the map, towards the course: three
  marching gold dots point the way, and crossing it on the way somewhere else starts nothing) (it has a little flag, and your best medal
  under it) and the clock starts: a timer plate top-left, the next gate pulses gold with an arrow over
  it, the one after shows dotted, a sky-blue edge arrow points to the next gate when it is off screen.
  Gate sounds climb in pitch as the course goes on. The last gate stops the clock with a whistle:
  GOLD / SILVER / BRONZE by the course's par times, NEW BEST, saved; the result plate also shows the
  gap to the next medal, or how the time compares with your best. A run ends after 20 s without a gate
  (the timer rim flashes red and a clock ticks, then ticks faster, through the last 5 s), on landing,
  take-off or an area change. Flying through only a start gate by accident ends quietly, with no TOO
  SLOW, and a start gate works only after the balloon has been outside it (loading a save made inside
  one does not start a race). The first gold unlocks the CHEQUERED balloon. Par times were set from an
  autopilot run (`tools/phaseb_race_cal.py`: gold = autopilot x1.3, silver x1.6, bronze x2.2).
- **Photo album** (`game/album.py`): 20 targets, e.g. Canterlot at dawn, the Express in Ponyville, zap
  apple lightning, the train on the Rainbow Falls trestle, the Crystaller beacon, the Big Bell ringing,
  the volcano erupting, Seaquestria, fireworks at Las Pegasus, the lighthouse at night, the aurora over
  the Crystal Empire, the Tree of Harmony, balloons over the map, the map by night. A postcard (P)
  whose view matches a target ticks it off (toast + a ring of sparkles) and pins a 64 x 36 copy in the
  album page (`out/game_album/<key>.png`). The DAWN (05-08), SUNSET (17-20) and NIGHT (20-05)
  targets need the clock in that hour. **Photo chances**: while a postcard would tick a target,
  viewfinder brackets blink in the screen corners and a [camera][P] chip sits bottom-right (the first
  two chances in a session also show PICTURE PERFECT!); an event stays photographable for about 8 s
  after it is set off. 10 photos unlock the SNAPSHOT balloon.
- **Badges** (`game/badges.py`): EXPLORER, CARTOGRAPHER (90 % charted), LOCAL EXPERT (every sight and
  stamp of one place), SIGHTSEER, STAMP COLLECTOR, SPARK MAKER (all 98 hotspots), GOOD DEED, GOOD
  NEIGHBOUR (help 5 places), WORLD MENDER (help all 20 places), FRIENDLY SKIES (meet all 15 balloons),
  FIRST POSTCARD (tick an album photo), PHOTOGRAPHER, RING DASH (10 map rings), RAINBOW RIDER (every
  map ring), RACER, CHAMPION (gold in all 6 races), SKY RACER (beat every racing balloon), THE END
  (help every place, n/20; earned with the grand finale): 18 in all. Each shows its progress in the
  journal and is announced by a toast with a stamp when earned.
- **Fog of war** (`game/fog.py`): uncharted land lies under a puffy cloud deck that melts away around the
  balloon (84 px) as it flies, with a scalloped, lit rim and a dithered cloud shadow on the land just
  outside. Over the snowy north and south the deck has a lilac shade and a darker rim, so it reads
  apart from the snow. The deck drifts, takes the map's time-of-day light and is also drawn on the
  overview map (M). The explored grid (8 px cells) is saved packed. Idle race start gates and map rings
  sit under the fog; markers, pins, the active race and discovery plates are drawn over it. ~3 ms per
  frame.
- **Journal** (J): tabbed pages, ← → to turn: ATLAS (MAP CHARTED %, and the grand finale footer, see
  Finale), HELP (the 20 troubles: not found yet, what each needs and from where, or mended, with the
  prop drawn; a place whose request you know is listed by name, Ponyville's first request even before
  Ponyville is found), RACES (medals, best times, par times, a sketch of the course), ALBUM (the list
  and a polaroid of each photo with a tip), PHOTOS (every postcard, see Photo book), FRIENDS (the 15 balloons: met or not, home, job), BADGES (a
  trophy case of 3 columns of 6, the selected badge shown large with its progress, and a NEXT UP list
  of the closest ones; badges earned this session twinkle), BALLOONS (16 patterns, 4 x 4). Secondary
  words are a darker brown, so they read better on the paper.
- **Toasts** (`game/toast.py`): one celebration banner at a time, top centre (lower while a place's
  completion plate or a race plate shows): troubles noticed, pick-ups, places helped, GO!, album
  photos, postcards, new balloons, badges, first visits. A toast that would cover the balloon shows low
  on the screen instead, for its whole life. Toasts never cover the errand plate: on the map before
  the first landing (where the plate sits level with them) one moves left of the plate, or, too wide
  for that, slides out from just under it. FIRST VISIT shows at once on arrival; the paced and the
  sound-echo rules are under Welcome card and tips. Toasts and the race clock wait with the world while
  the journal or pause menu is open.

### Save file

`out/game_save.json` is version 3: version 1 keys plus `steps` (the journey's step clock), `fog`
(packed explored grid), `rings` (map rings flown), `help` (`seen`, `fixed`, `carrying`, `load`: how
many of it the basket holds, `intro`), `flock` (`met`, `beaten`), `tutor` (`done`, `welcomed`,
`journal`, and `gift`: when the first request was given), `weather` (`met`), `races` (`best` steps
per course), `album` (`taken`) and `badges` (`earned`); each place under `regions` also keeps
`lasting` (its set-pieces done for good: `lit`, `bloom`, `neon`). On load the clock resumes at
max(`steps`, 1 + the latest mend / badge / photo stamp) + 150 steps, so nothing from before the save
replays, and a save from before `steps` existed resumes just after its latest stamp (a load / save
round trip therefore shows `steps` + 150, as intended). A save without `load` carries one of its goods.
A version 2 save loads with its old errand chain dropped (the troubles start fresh). A version 1 save loads as before: the fog is charted around every place already
found and around the balloon, everything else starts empty, and badges already deserved are given
quietly. Unknown keys or corrupt values are dropped (a bad fog string falls back the same way). A
save that is not valid JSON, is cut short or fails to load is renamed `<save>.bad` and a new journey
starts.

### Phase B checks

| tool | what |
|---|---|
| `tools/game_autopilot_b.py` | headless play-through by real inputs: notice Ponyville's empty cart, call the train + postcard, fetch apples from Sweet Apple Acres, mend the cart, race the Canterlot Circuit, turn every journal page; prints OK / FAIL per stage and timings; sheets in `out/phaseb/autopilot/` |
| `tools/phaseb_shots.py` | stills of the new systems (journal pages, fog, balloon friends, weather, the helped world map by day and night, races, album) in `out/phaseb/` |
| `tools/phaseb_spots.py` | every trouble (broken / mended) and goods source: reachable, clear of hotspot reach, SPACE works (the NOTICE check parks 70 px away on whichever side has room); sheets `out/phaseb/troubles_*.png`, `sources.png` |
| `tools/phaseb_race_cal.py` | flies every course with an autopilot (times for the par table), sheets `out/phaseb/gates_<key>.png` |
| `tools/phaseb_album_check.py` | stages and photographs all 20 album targets (DAWN, SUNSET and NIGHT in their hours), sheet `out/phaseb/album.png` |
| `tools/phaseb_mapgrid.py`, `tools/phaseb_pano.py` | whole-map and area panoramas with grids (for placing gates and spots) |

## Improvement round (2026-10-28)

A review played the game headless and its findings were built in five batches (map core, places,
helping, onboarding and journal, sky / album / sound), then integrated and checked together. What
changed for players (details in the sections above):

- **The goal**: the grand finale now comes after every place has been helped, on the next take-off
  (THE END counts places helped). The journal's ATLAS footer shows how far it is.
- **First minutes**: PONYVILLE NEEDS A HAND as the welcome card closes, then one message at a time
  (toasts before tips, paced in the first 2 minutes, no echoing sounds). The title screen's counters
  have captions, and a new player sees the journey ahead greyed out.
- **The map**: a place counter for visited / found places with a FIRST VISIT toast, name plates by
  status, a two-row land prompt (NEW tag, landmarks, stamps, trouble) that comes up at once and reaches
  further for a just-found place, a steadier pink arrow, chunky edge darts clear of the HUD, a
  camera that keeps the balloon clear of the HUD, a star chime on discovery, wind that calms near
  found places, and a fog that clears a wider circle (84 px) and reads apart from snow.
- **Freezing**: the journal, the pause menu and the welcome card freeze the whole world; the overview
  map pauses races. The journey's clock is saved, so helped places stay mended after a relaunch.
- **Places**: every hotspot press gives a pulse, a ring and sparks, with '+1' on a first use, a
  jackpot for the last, a spark counter, gold twinkles on untried hotspots and NEW BALLOON pattern
  toasts. A return trip lands straight at the errand. Big set-pieces stay done. Las Pegasus by day
  looks like day. Landmark names stay on top and move off the counters.
- **Helping**: `?` marks on the map, FROM / TO plates with `+N` (compact under landmark tags), a basket
  that carries enough for every open need (X2...), a fanfare, a light band that reveals a lasting
  festival, and rainbows on the map back to the place whose goods helped. Manehattan's park cart now
  stands on the promenade by the pavilion.
- **Friends**: hearts on unmet friends, a longer reach (22 px), the bubble beside the friend, a toast
  saying what SPACE does next, NEW BALLOON IS FLYING toasts with an arrow, landing always wins over a
  friend, and guides lead to unseen troubles first.
- **Album, postcards, races, badges**: photo-chance brackets and chip, hour rules for DAWN / SUNSET /
  NIGHT, a shutter and POSTCARD SAVED toast for every postcard (files never overwrite), kinder medal
  times, the gap to the next medal, rising gate sounds, a finish whistle and a hurry tick, no TOO SLOW
  for a start gate flown by accident, and 4 new badges (18) in a trophy-case page with NEXT UP.
- **Sound**: rewards duck the music, the journal and the menu dip it, and each place's music now plays
  at the menu's volume.
- **Fixes**: no softlock from J or pausing during take-off, SPACE does nothing during the take-off
  climb, no crash after landing mid-way through a friend's race, damaged saves are kept as `.bad`, a
  place that fails on arrival is padlocked, a map ring that sat on a name plate (CLOUDSDALE) moved off
  it, and `play.py --frames` counts this session's steps.

Still open: a mend changes ~2 % of the frame (the aim was 3 %); in the first 2 minutes a backlog of
paced toasts can make a badge or NEW BALLOON toast show 10-20 s late (FIRST VISIT and weather toasts
are exempt); a toast that took the low slot in a place stays low over the map after take-off;
Yakyakistan stays at ~13 ms per frame at x4, as before the round.

The round's own check scripts (each file's docstring gives its usage):

| tool | what |
|---|---|
| `tools/b_enter_areas.py` | lands every place straight into each of its areas (a return trip) and flies about: no area-0 assumptions |
| `tools/b_land_job.py` | a return trip lands at the errand's area; first visits and places without one start in area 0 |
| `tools/b_second_landing.py` | a set-piece set off, then landed on again (same session and after a save round trip): still done? |
| `tools/b_pattern_toast.py` | the stamp that earns a balloon pattern shows its toast; first-use blip and last-hotspot jackpot |
| `tools/b_shots.py` | full frames around a hotspot press (bubble, burst, '+1', HUD counters) |
| `tools/b_memo_check.py` | the place world memo against drawing every frame afresh (pixel-identical), and its timing |
| `tools/b_knee_compare.py` | night- or dawn-painted places by day with the old brightening vs the soft knee |
| `tools/b_canterlot_tags.py` | Canterlot's ROYAL GARDENS tag, broken / mended, noon / night |
| `tools/c_plate_sweep.py` | flies places with a trouble open and counts steps where a landmark tag sits under the errand plate |
| `tools/d_gift.py` | a new game from the welcome card: the first request's toast, then the first tips |
| `tools/d_tips.py` | every tip plate (each wording), learnt and folding, on one sheet |
| `tools/d_journal.py` | every journal page from a new game and from a scratch copy of a save |
| `tools/d_title.py` | the title screen at a few steps, from a new game and from a scratch copy of a save |

## Soundtrack (2026-11-01)

`audio/game_music.py` composes and renders the game's music; the tracks themselves (melodies, chords,
arrangements) are in `audio/game_tracks.py`. Every track is one seamless loop: notes are written on a
bar grid of its own tempo and metre, rendered one tail longer than the loop, and the tail (reverb, echo,
pad releases) is folded back onto the start; the master chain runs over three copies of the loop and
keeps the middle one. Output: `audio/out/game/<id>.ogg` (what the game plays, ~37 MB in all),
`audio/out/game/wav/` (for checks and the trailer) and `audio/out/game/tracks.json`.

| track | title | style |
|---|---|---|
| `title` | The World Atlas | D major, 84 BPM: the Atlas theme on a music box, then the band |
| `map_day` / `map_night` | Wind in the Envelope / Lanterns over Equestria | G major, 104 BPM, 48 bars (1:51): the same tune as a folk-band overworld theme and as a nocturne (music box, keys, harp); they play in sync and crossfade 18:00-20:00 and 05:00-07:00 |
| `race` | Ring Rush | A minor, 138 BPM |
| 20 places | one each, 47-80 s | Canterlot fanfare, airy Cloudsdale, Ponyville folk, Sweet Apple Acres hoedown, mysterious Everfree, Neighagra Falls, sporty Rainbow Falls, Manehattan swing, Baltimare harbour shuffle, Griffonstone minor march, Dragon Lands riff, Mount Aris (with an underwater middle), Klugetown bazaar, the Changeling hive (dark, then blooming), Appleloosa western, Las Pegasus funk, Vanhoover lo-fi rain, the Crystal Empire waltz (3/4), Yakyakistan stomp, the Tree of Harmony lullaby (3/4) |

```bash
.venv/bin/python audio/game_music.py              # every track (~2.5 min)
.venv/bin/python audio/game_music.py --check      # compose only: melody notes that rub against their chord
.venv/bin/python audio/game_music.py --stems race # also each role's level against the mix
.venv/bin/python tools/game_music_check.py        # loop seams, levels, spectrum bands, spectrograms (out/music_check/)
```

In the game (`game/play.py` `Audio`): four mixer channels are reserved for music (map by day, map by
night, and two for the rest, so a change crossfades). A track is decoded on first use; a place's theme
is decoded in a background thread while the balloon hovers to land there. A screen without a game
track falls back to its bars of the film score.

## Fixes (2026-11-01)

A headless review (world map, the 20 places, the front end) found about 30 problems; these were fixed:
- the mixer ran at 44.1 kHz, so every effect and place loop played 9 % slow and 1.5 semitones flat;
- the land prompt followed the balloon's faint shadow (FOCUS_DY: reach is now measured from the
  balloon itself), so hovering the envelope on Ponyville offered Sweet Apple Acres;
- the finale played the last mend's toasts and sounds unseen (toasts wait for the finale to end);
- a new game with tips off got no first request (and turning tips on later popped the welcome card);
- the first goal's bubble could hide behind the key hints with no arrow (HUD bands count as off screen);
- a toast cut short by an urgent one was lost (it now comes back once);
- racing friends set off before a human could react (a 0.8 s head start); guides now lead to the
  errand under way first; the map edges no longer keep the balloon's speed against them; the '?' hides
  while you land there; a map race starts only when you fly through its start gate towards the course
  (no accidental races on the way to Ponyville or Everfree);
- the balloon could vanish under the counters, plates, toasts and tips (the HUD turns see-through over
  it: `compose.see_through`); ESC during a cloud sweep was lost;
- fullscreen stretched pixels by a fraction; big windows missed the 60 fps budget (the window path,
  `compose_window`, is ~2.6x faster and pixel-identical); the game kept running in the background;
  the music dipped for muted reward sounds; write errors and damaged settings crashed the game;
  no crash report; uncapped frame rate when vsync silently failed;
- postcards were hard to find (now in Pictures, see Postcards);
- in the places: the Dragon Lands eruption flash and screen shake (it now erupts once per visit by
  itself, then on SPACE), the Badlands sky snap, flash and canyon shake, the Yakyakistan avalanche
  shake, the Griffonstone double lightning strobe, a Cloudsdale stamp wedged under the counters, and day/night
  fixes in the Two Sisters throne hall and the Vanhoover bay (see the region files).
Not changed: the volume sliders stay linear; gamepads still use raw button numbers (no pad to test).

## Photo book (2026-11-02)

Postcards can now be looked at in the game: the journal's PHOTOS page and its full-screen viewer (see
Photo book above). Also: the journal fades in over the view and back out (`OVERLAY_FADE`, 5 steps) and
its pages cross-fade when they turn, instead of jumping; the postcard camera's flash rises over three
frames to a soft glow of about 45 % and eases out (`FLASH` in game/world.py), never a full white frame; NEW GAME no longer forgets where postcards are saved (the album's folder line went blank until
the game was restarted).

Night lights (reported from the video): places the film painted by day kept their windows dark at
night, because game/daylight.py only keeps lights that outshine their surroundings and nothing switched
the windows on (a gold pane on a pale wall even dimmed to khaki). `game/nightwin.py` now finds each scene
layer's window panes once (blue glass, 2 x 2 to 6 x 9 px, cached) and at night lights a share of them
warm where nothing covers them, more as the night deepens (night() 0.3 to 0.7), now and then one more;
the region keeps them shining after its relight with GlowKeep (game/regions/everfree.py). Used in
Manehattan (pale walls only: the Crystaller Building, the cream towers, the theatres, the brownstones,
the skyline from the bridge), Baltimare (the brick square and harbour houses, the lighthouse windows,
the keeper's cottage, the swamp cabins), Ponyville, Sweet Apple Acres, the Mount Aris palace and
Canterlot's gardens, docks and city. Baltimare's lighthouse beam (a long warm ray the relight took for
sunlit trim, so it showed as a dull brown band) and the Crystaller's beacon gold are kept shining the
same way. Costs ~2-6 ms per layer once, nothing by day. Then, from more reports: Manehattan lights
60-80 % of its windows (the Crystaller all of them, its sunburst crown lit gold every night), the beacon's
halo and rays shine at night, the rail bridge has a necklace of bulbs along its cables, deck lamps, lit
tower emblems and red lights on the tower tops; at Sweet Apple Acres only the zap apples themselves (their
exact colours, found once in the farm layer) and the rainbow lightning keep their light by night (a halo
round the trees looked muddy); Baltimare's lighthouse beam is brighter and sweeps in front of the clouds,
and its lantern's flare (SPACE) keeps its light instead of dulling to a dark blue dome. GlowKeep
now gives the colour back right after the relight (Region.after_light), before the stamps and hotspot
effects are drawn over it, so a hotspot's burst no longer hides a glow.

Landing stall (reported: "when I meet a new city there is a delay and a jump of music and video"): a
place's first landing built all its art on the main thread (Region.warm, ~0.1-0.3 s, more in the packaged
app on a first import) at the moment the iris closed; the frame froze, then the loop ran up to 6 steps at
once (the iris jumped) and the music fades took the whole gap in one go. Now the map starts building the
place in a background thread as soon as the land prompt is up (`Game.prepare_region`; the thread builds
its own Region object only, try_region adopts it at the landing, waiting if it is not quite done; at most
3 such places are kept), and the front end treats a frame gap over 0.1 s as a stall: no catch-up burst,
and the music fades never move more than two steps' worth in one frame. A first landing now has no frame
gap over ~21 ms (it was 110-190 ms, then a 3-6 step jump); `tools/claude_landing_hitch2.py` measures it.

The bigger hitch came earlier, the moment a new city was found: the land prompt came up and the front
end started decoding that place's theme (pygame.mixer.Sound on an OGG, in a thread). That decode holds
the audio device's lock for ~60-90 ms, so the game's next mixer call (a volume change, every frame)
waited for it, and the sound card was starved too: the picture froze and the music skipped. The places'
themes are now streamed from their files with pygame.mixer.music (no decode, ~0.5 ms to start, fades as
before; one place theme plays at a time, the old one has faded out long before the next landing), and
the title, map and race tracks are decoded at start-up before any music plays (DECODED in game/play.py;
start-up ~1 s longer). Found-a-city and landing round trips now run without a frame gap over ~28 ms
(`tools/claude_discover_hitch.py`, `tools/claude_roundtrip_hitch.py`, real time, the game's own audio).

## Lights at every hour (2026-11-03 .. 05)

Reported: "the lights are not right in the afternoon, twilight or night". Every place was checked at ten clock
hours (its painted hour, 09:00, 13:30, 16:00, 17:48, 18:48, 19:36, 21:30, 02:00, 05:36) and fixed, then the
shared light code:

- **Only real lights glow.** The relight's own keeping of small warm spots (`daylight._lights`) took painted
  trims, fences, pillars, fruit and sunlit rims for lamps; it also had an int16 overflow, so bright lamps were
  never kept. Most places now return `Region.auto_glow() -> False` and keep exactly their own lights (window
  glass, lamp heads, lanterns, fire, neon, crystals, beams) with GlowKeep masks (the night-painted Las Pegasus,
  Vanhoover, Crystal Empire, Yakyakistan and the Two Sisters keep the relight's own, checked per place); `game/nightwin.py` placements
  take `exclude` / `only` boxes for "panes" that are paint (Ponyville's lavender fence was lit as windows).
  No dithered halos over scenery; lamps lit by day are switched off; the trouble props' lamp and lanterns
  light only their glass at night (`mend_art.LIGHTS`).
- **Lights come on gradually.** Windows ramp with `nightwin.ramp`; the festival's bulbs and sky lanterns come
  on one by one about 19:00-19:40 (`quests.festival_night` is 0..1), off again 05:30-06:00.
- **No jumps.** The clock's sky moves in 3-minute steps (`daylight.sky_q`, was quarter hours: the whole sky
  jumped by up to 20 levels at dusk); places that draw their own pieces of it use the same steps. The sky
  presets blend smoothly, the sun sets and the moon rises behind the land (`_body_pos`), Canterlot's evening
  keeps the film's sun down and its sunrise burst is a gentle swell. `Area.horizon` lets a low sky reach its
  horizon (no flat band). Places build their dusk caches at landing (`warm`): no 20-70 ms stall at dusk.
- **Afternoon.** Painted morning suns, glare, haze and reflections give way to the clock's sun on its arc;
  golden-painted places lose their evening light by day (and the balloon its evening tint: `Region.light_tint`
  fades by the light difference, `daylight.tint_gap`).
- **Everything over the art is lit too.** The cloud sweep between areas takes the clock's colours
  (`Region.sweep_hour`); wind streaks and rain in places and on the map are dimmed at night
  (`daylight.dim_new`); on the world map the rings, race gates, rainbows and bunting take the clock's grade,
  the festival shows warm bulbs instead of a halo, and the balloons are softly tinted (`daylight.map_tint`).
- **Errands:** goods from a place not discovered yet get no bubble and no gold arrow on the map (the plate
  still names it); they appear once the place is found.

Tools: `tools/claude_light_survey.py` (sheets of every place at the ten hours, `one` for a 4x zoom),
`tools/claude_light_mask.py` (what the relight keeps glowing), `tools/claude_light_map.py` (the map),
`tools/claude_light_props.py` (the trouble props), `tools/claude_light_smoke.py` (every place at five hours:
errors and ms), `tools/claude_light_hitch.py --areas` (the clock running through dusk / dawn in every area:
slow frames and light steps). Notes per place: `out/light_audit/work/<place>/bugs.md`.

## Smooth motion

The game logic steps 30 times a second (the film's rate). The window redraws at 60 fps (vsync) and
interpolates the camera and the balloon between steps. The world is drawn at an integer camera, and
the compositor (`game/compose.py`) shifts it by the camera's fractional part, at 1/4 art-pixel
precision. The balloon and the HUD are separate layers, and the camera's pixel position is derived
from the balloon's, so the balloon never flickers against the ground. Art pixels stay sharp blocks.
Small art that rides a moving balloon (a friend's heart and SPACE bubble) is a `Frame.ride` sprite
drawn at the balloon's sub-pixel position with the same bob, so it never wobbles against it. While an
overlay holds the world still, the view is held too (no interpolation). `tools/game_smooth.py`
measures it (ground steps per frame, balloon back-and-forth jumps).

Inside a place, the world is drawn once per step and reused for the 60 Hz in-between frames at the
same integer camera (`Region.MEMO`; `tools/b_memo_check.py` checks that it is pixel-identical to
drawing every frame afresh). The pause / welcome-card dim is applied per art pixel before scaling
(the card frame went from ~18 to ~14 ms at x4, the pause frame from ~11 to ~7 ms).

No stalls: the cloud sweep between areas of a place is drawn by `game/sweep.py` (the film's
`shotfx.cloud`, pixel for pixel, in ~2 ms: a pre-coloured cloud texture, the bank textured only where
it is, the new area not drawn while the bank still hides it and drawn once per step after that). The
sweep starts from the frame just shown instead of drawing the old area again, the cloud texture is
built when a place loads, and every sound effect is synthesized at start-up (~0.5 s). The compositor
writes the HUD and iris layers by broadcasting each art pixel into its block, so a full-screen iris
costs ~10 ms instead of ~35. `tools/game_sweep.py` times every frame of every sweep (all stay under
16 ms at scale 4).

## Files

| file | what |
|---|---|
| `game/play.py` | pygame front end: window (settings scale / fullscreen), keys, 60 fps loop, music, SFX |
| `game/menu.py` | title screen (`TitleMode`), pause menu (`PauseMenu`), settings rows, controls page |
| `game/settings.py` | `Settings`: volumes, window scale, fullscreen; `out/game_settings.json` |
| `game/sweep.py` | the fast cloud sweep between areas (same picture as `engine.shotfx.cloud`) |
| `game/world.py` | `Game` (modes, iris transitions, discoveries, the world / overlay clocks, save and load) and `MapMode` (land prompt, place counter, pink target); no pygame inside |
| `game/region.py` | base class for explorable regions: areas, balloon physics, eased camera, tags (and their side), card, hotspot effects, lasting set-pieces |
| `game/regions/<seg>.py` | one per explorable place (auto-discovered by file name = segment id) |
| `game/compose.py` | frame layers (world / balloon / HUD / iris, ride sprites) and the sub-pixel compositor |
| `game/ui.py` | key caps, hints, counters (the place counter), name plate, NEW tag, place stats, edge arrow, interaction icons, bubbles, stamps, bursts |
| `game/journal.py` | the atlas journal (J): atlas, help, races, album, badges and wardrobe pages |
| `game/quests.py` | helping the places: the 20 troubles, goods sources, noticing, mending, the festival, the errand plate, the changing map (`?` marks, goods rainbows) |
| `game/daylight.py` | the world clock's light in the places (re-lighting painted scenes, lights kept glowing) |
| `game/weather.py` | snow, rain clouds and wind on the map and in places; the wet / snowy / pushed balloon |
| `game/tutor.py` | the welcome card, the first request and the short contextual tips |
| `game/flock.py` | balloon friends on the map: greeting, carriers, racers, guides |
| `game/mend_art.py` | the trouble props, broken and mended; the festival art (bunting, mast, roof flags, balloons, confetti, the reveal band); the map's festive marks and `?` |
| `game/items.py` | the goods' 7x7 sprites and the cargo hanging under the basket |
| `game/races.py` | ring races: courses, gates, timer, medals, best times |
| `game/album.py` | the photo album: 20 targets, their checks (and hours), photo chances, thumbnails |
| `game/badges.py` | the 18 milestone badges and their medallions |
| `game/fog.py` | the fog of war over the world map (and the overview) |
| `game/toast.py` | celebration banners (one at a time, top centre; paced, placed clear of the balloon and the errand plate) |
| `game/postcard.py` | postcard snapshots (P) |
| `game/perks.py` | balloon perks: what each wardrobe balloon does while worn |
| `game/nightwin.py` | lit windows at night for day-painted places (found per scene layer, kept by GlowKeep) |
| `game/photos.py` | the photo book: lists and reads back the postcards for the journal's PHOTOS page and viewer |
| `audio/game_music.py`, `audio/game_tracks.py` | the game's soundtrack: engine and tracks (see Soundtrack) |
| `tools/game_music_check.py` | soundtrack checks: seams, levels, bands, spectrograms |
| `tools/trailer2_make.py` | the second trailer (`out/trailer2/equestria_atlas_trailer.mp4`) |
| `game/REGIONS.md` | how to turn a film scene into an explorable region |
| `tools/game_region_check.py` | check one region: edge / tour sheets, ms per frame, purity, landmarks; `--interact` tests every hotspot and stamp |
| `tools/game_shots.py` | headless autopilot play-through: key frames, contact sheets, ms per frame |
| `tools/game_smooth.py` | smoothness numbers for the camera |
| `tools/game_sweep.py` | frame times through every in-place cloud sweep (step + render + compose) |
| `tools/game_sweep_eq.py` | `game/sweep.py` against `engine.shotfx.cloud`: pixel equality and speed |
| `tools/game_menus.py` | headless shots of the title, settings, controls and pause screens (`out/game_checks/menus/`; the title shots are timed from the start clock) |
| `tools/game_all_check.py` | the region check (`--interact`) for every place in parallel workers (`--jobs 4`): one summary table of landmarks, hotspots, stamps, ms per frame and problems (`out/game_checks/SUMMARY.md`; its frame times run high with 4 workers at once) |

## Regions

Each region file wraps its film scene (read-only) in one of two ways. In **rebind** style
(`regions/ponyville.py`), a copy of the scene's draw function is bound to game-driven globals:
camera, balloon, timed events. In **recompose** style (`regions/sweetapple.py`), the scene's cached
layers and animation helpers are called directly along one continuous strip. Film beats (trains,
lightning, ...) replay on game clocks. A region can have several areas (shots); flying past an
edge sweeps a cloud bank to the next one. Full guide: `game/REGIONS.md`.

## Film fixes made for the game

- `scenes/s02_cloudsdale.py` `cloud_mass`: a tileable strip (`bank_strip`) now treats its first column
  as the right neighbour of its last one, so no gold rim seam is drawn at the tile edge.
- `scenes/s08_manehattan.py` `a_far`: the tiny robed torch-bearer on the far islet is now an abstract
  torch column (patina shaft, capital, gold bowl and flame), per the no-statues rule. The game's
  Manehattan region used to patch this itself and now uses the film layer as it is.

Both scenes pass `vid.py check`.
