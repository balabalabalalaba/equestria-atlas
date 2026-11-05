#!/bin/bash
# Build "Equestria Atlas.app" and a zip to share: dist/Equestria-Atlas-mac.zip
# Run from anywhere:  bash packaging/build_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv; PY=.venv/bin/python; }
$PY -m pip install -q -r packaging/requirements.txt
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging/equestria_atlas.spec
rm -rf "dist/Equestria Atlas (Mac)" dist/Equestria-Atlas-mac.zip
mkdir -p "dist/Equestria Atlas (Mac)"
cp -R "dist/Equestria Atlas.app" "dist/Equestria Atlas (Mac)/"
cp "packaging/READ ME FIRST.txt" "dist/Equestria Atlas (Mac)/"
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "dist/Equestria Atlas.app/Contents/MacOS/Equestria Atlas" --selftest | tail -1
(cd dist && ditto -c -k --norsrc --keepParent "Equestria Atlas (Mac)" Equestria-Atlas-mac.zip)
echo
echo "Done: dist/Equestria-Atlas-mac.zip  ($(du -h dist/Equestria-Atlas-mac.zip | cut -f1))"
