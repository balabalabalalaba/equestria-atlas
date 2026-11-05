# Equestria Atlas

A cozy pixel-art balloon adventure. Fly the explorer's balloon over a hand-drawn world map, land in 20 living
places, help each one with the goods it needs, race through rainbow rings, make friends with the other balloons,
snap postcards for your photo book, and watch the world turn from morning to golden hour to a night full of
lanterns, neon and auroras.

## Play it

**[Download the game for Windows](../../releases/latest/download/Equestria-Atlas-windows.zip)** (Windows 10 / 11)

1. Right-click the downloaded `Equestria-Atlas-windows.zip` > *Extract All*.
2. Open the folder and double-click `Equestria Atlas.exe`.
3. The first time Windows may say "Windows protected your PC": click *More info*, then *Run anyway* (the game
   is not signed by a big publisher).

No installation is needed and nothing else has to be downloaded. Progress is saved automatically.

| key | what it does |
|---|---|
| arrow keys / WASD | fly |
| SHIFT | go faster |
| SPACE | land, set something off, pick up and deliver goods |
| ESC | take off (in a place) / pause menu (on the map) |
| J / M / P | journal, overview map, postcard |
| F | fullscreen |

A gamepad works too. More tips are in `READ ME FIRST.txt` inside the zip.

## About

A fan-made, non-commercial project: all art and music are original, made with a small Python pixel engine
(numpy, pygame). Landscapes only, no characters.

## Building it yourself

Python 3.11 or newer, then:

- Windows: double-click `packaging\build_windows.bat` (makes `dist\Equestria-Atlas-windows.zip`)
- Or run it from source: `pip install -r packaging/requirements.txt`, then `python game/play.py`

Every version tag (`git tag v1.1 && git push origin v1.1`) builds the Windows zip on GitHub and publishes it as a Release.
