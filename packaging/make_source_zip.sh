#!/bin/bash
# The game's sources and assets only (no film output, no virtualenv): copy this zip to a Windows PC,
# unzip it, and double-click packaging\build_windows.bat.   ->  dist/Equestria-Atlas-source.zip
set -e
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/Equestria-Atlas-source.zip
zip -q -r dist/Equestria-Atlas-source.zip engine game scenes packaging audio/*.py \
    audio/out/soundtrack.wav audio/out/sections.json audio/out/game/*.ogg audio/out/game/tracks.json \
    out/cache/worldmap_*.npz \
    -x '*/__pycache__/*' '*.pyc' 'packaging/*.iconset/*'
echo "Done: dist/Equestria-Atlas-source.zip  ($(du -h dist/Equestria-Atlas-source.zip | cut -f1))"
